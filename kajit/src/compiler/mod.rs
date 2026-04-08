mod dwarf;
mod hir_to_ir;
mod shape_utils;

use dwarf::*;
pub(crate) use shape_utils::*;

use std::collections::{BTreeMap, HashSet};
use std::path::{Path, PathBuf};

use facet::{Def, ScalarType, Shape, Type, UserType};
use kajit_hir as hir;

use crate::format::{DecoderKind, FieldEmitInfo, SkippedFieldInfo};
use crate::intrinsics;
use crate::ir::RegionBuilder;
use crate::pipeline_opts::PipelineOptions;

pub use hir_to_ir::lower_hir_module;
pub(crate) use kajit_postcard::{build_postcard_decoder_hir, supports_postcard_decoder_hir};

/// A compiled deserializer. Owns the executable buffer containing JIT'd machine code.
pub struct CompiledDecoder {
    buf: crate::ir_backend::BackendBuf,
    cfg_mir_line_text_by_line: Vec<String>,
    entry: usize,
    func: *const u8,
    trusted_utf8_input: bool,
    _jit_registration: Option<crate::jit_debug::JitRegistration>,
    asm_program: Option<kajit_emit::aarch64_asm::Program>,
}

/// A compiled scalar function. Owns the executable buffer containing JIT'd machine code.
/// Uses standard calling convention: args in x0..x7, return value in x0.
pub struct CompiledFunction {
    buf: crate::ir_backend::BackendBuf,
    entry: usize,
}

impl CompiledFunction {
    /// Get the entry point as a raw function pointer.
    /// The caller is responsible for casting to the correct signature.
    pub fn as_ptr(&self) -> *const u8 {
        unsafe { self.buf.code_ptr().add(self.entry) }
    }

    /// The raw executable code buffer.
    pub fn code(&self) -> &[u8] {
        self.buf.code()
    }
}

// Safety: the JIT code buffer is immutable and pinned for the lifetime of CompiledDecoder.
unsafe impl Send for CompiledDecoder {}
unsafe impl Sync for CompiledDecoder {}

impl CompiledDecoder {
    /// Raw entry point pointer. Caller casts to the appropriate signature.
    pub(crate) fn func_ptr(&self) -> *const u8 {
        self.func
    }

    /// The raw executable code buffer.
    pub fn code(&self) -> &[u8] {
        self.buf.code()
    }

    /// Byte offset of the entry point within the code buffer.
    pub fn entry_offset(&self) -> usize {
        self.entry
    }

    /// Whether `from_str` can safely enable trusted UTF-8 mode for this format.
    pub fn supports_trusted_utf8_input(&self) -> bool {
        self.trusted_utf8_input
    }

    /// Deterministic machine-emission trace annotated with CFG-MIR provenance.
    pub fn emission_trace_text(&self) -> Result<String, kajit_emit::TraceError> {
        let entries = match &self.buf {
            crate::ir_backend::BackendBuf::X86_64(buf) => buf.trace_entries()?,
            crate::ir_backend::BackendBuf::Aarch64(buf) => buf.trace_entries()?,
        };

        Ok(format_emission_trace_entries(
            &entries,
            &self.cfg_mir_line_text_by_line,
        ))
    }

    /// CFG-MIR listing text (the same text used for DWARF debug info).
    pub fn cfg_mir_text(&self) -> String {
        self.cfg_mir_line_text_by_line.join("\n")
    }

    /// CFG-MIR listing lines (individual lines, for harness generation).
    pub fn cfg_mir_lines(&self) -> &[String] {
        &self.cfg_mir_line_text_by_line
    }

    /// Build DWARF sections for a standalone binary (code_address=0).
    ///
    /// The linker will relocate the addresses. This is used by the harness
    /// generator to produce DWARF that works in a standalone executable.
    pub fn build_standalone_dwarf(
        &self,
        listing_path: &std::path::Path,
    ) -> Option<crate::jit_dwarf::JitDwarfSections> {
        let source_map = self.source_map()?;
        let file_name = listing_path.file_name()?.to_str()?.to_owned();
        let directory = listing_path
            .parent()
            .and_then(std::path::Path::to_str)
            .map(str::to_owned);
        let rows = normalize_debug_line_rows(source_map);

        let debug_info = crate::jit_dwarf::JitDebugInfo {
            target_arch: crate::compiler::dwarf::jit_dwarf_target_arch(),
            code_address: 0, // standalone binary — linker relocates
            code_size: self.code().len() as u64,
            line_table: crate::jit_dwarf::JitDebugLineTable {
                file_name,
                directory,
                rows,
            },
            subprogram: crate::jit_dwarf::JitDebugSubprogram {
                name: "kajit_decode".to_owned(),
                frame_base_expression: Vec::new(),
                variables: Vec::new(),
                lexical_blocks: Vec::new(),
            },
        };

        crate::jit_dwarf::build_jit_dwarf_sections_from_debug_info(&debug_info).ok()
    }

    /// Source map (code offset → DWARF line).
    fn source_map(&self) -> Option<&kajit_emit::SourceMap> {
        Some(self.buf.source_map())
    }

    /// ARM64 assembly text (captured instructions before encoding).
    pub fn assembly_text(&self) -> Option<String> {
        self.asm_program.as_ref().map(|p| format!("{}", p))
    }
}

pub(crate) const DEFAULT_PRE_LINEARIZATION_PASSES_ENABLED: bool = true;

pub(crate) fn materialize_backend_result(
    result: crate::ir_backend::LinearBackendResult,
) -> (
    crate::ir_backend::BackendBuf,
    usize,
    Option<kajit_emit::SourceMap>,
    Option<crate::ir_backend::BackendDebugInfo>,
    Option<kajit_emit::aarch64_asm::Program>,
) {
    let crate::ir_backend::LinearBackendResult {
        buf,
        entry,
        source_map,
        backend_debug_info,
        asm_program,
        intrinsic_call_sites: _,
        data_relocs: _,
        extern_addr_relocs: _,
    } = result;
    (
        buf,
        entry as usize,
        source_map,
        backend_debug_info,
        asm_program,
    )
}

/// All intermediate artifacts from a compilation pipeline run.
///
/// This is THE canonical way to compile a decoder. Both the CLI and tests
/// use this, so debug dumps and JIT execution see identical vreg numbering.
pub struct PipelineArtifacts {
    /// HIR module text
    pub hir_text: String,
    /// IR text after each optimization pass: (pass_name, ir_text)
    pub ir_opt_timeline: Vec<(String, String)>,
    /// Linearized IR text
    pub linear_text: String,
    /// CFG-MIR text (debug line listing, after CFG optimizations)
    pub cfg_text: String,
    /// CFG-MIR canonical text (round-trippable, after CFG optimizations)
    pub cfg_canonical_text: String,
    /// Assembly text (aarch64 only, empty on other platforms)
    pub asm_text: String,
    /// VReg → physical location map (for lockstep debugger, legacy static map)
    pub alloc_map: crate::harness::AllocationMap,
    /// Per-program-point vreg location map (call-clobber aware, replaces alloc_map for lockstep)
    pub location_map: crate::harness::LocationMap,
    /// Intrinsic call sites in the JIT code (for harness relocation)
    pub intrinsic_call_sites: Vec<crate::ir_backend::IntrinsicCallSiteInfo>,
    /// External address relocs (vtable function pointers) for harness relocation
    pub extern_addr_relocs: Vec<crate::ir_backend::ExternAddrRelocInfo>,
    /// Exact machine-code ranges for emitted CFG ops.
    pub backend_debug_info: Option<crate::ir_backend::BackendDebugInfo>,
    /// The post-optimization CFG-MIR program (same one the JIT compiled).
    /// Used by the lockstep debugger to run the interpreter on the exact same IR.
    pub cfg_program: kajit_mir::cfg_mir::Program,
    /// The compiled decoder (ready to execute)
    pub decoder: CompiledDecoder,
}

/// Extract data_arg layout metadata from HIR function parameters.
///
/// Walks the HIR type of each parameter and records which u64-word positions
/// contain pointers. This lets the debugger seed shadow memory so loads through
/// data_arg pointers recover provenance.
fn extract_data_arg_layouts(module: &hir::Module) -> Vec<kajit_types::TypeLayout> {
    let Some((_, function)) = module.functions.iter().next() else {
        return Vec::new();
    };

    function
        .params
        .iter()
        .map(|param| hir_type_to_layout(module, &param.ty))
        .collect()
}

/// Convert an HIR Type to a TypeLayout for runtime use (debugger, shadow memory, etc.).
fn hir_type_to_layout(module: &hir::Module, ty: &hir::Type) -> kajit_types::TypeLayout {
    use kajit_types::TypeLayout;
    match ty {
        hir::Type::Unit => TypeLayout::Scalar { size: 0 },
        hir::Type::Bool => TypeLayout::Scalar { size: 1 },
        hir::Type::Integer(int_ty) => TypeLayout::Scalar {
            size: (int_ty.bits / 8) as u8,
        },
        hir::Type::Ref { pointee, .. } | hir::Type::Handle { value: pointee, .. } => {
            TypeLayout::Ptr {
                pointee: Box::new(hir_type_to_layout(module, pointee)),
            }
        }
        hir::Type::Address { .. } => TypeLayout::Ptr {
            pointee: Box::new(TypeLayout::Opaque { size: 0 }),
        },
        hir::Type::Slice { element, .. } => TypeLayout::Slice {
            element: Box::new(hir_type_to_layout(module, element)),
        },
        hir::Type::Str { .. } => TypeLayout::Str,
        hir::Type::Array { element, len } => TypeLayout::Array {
            element: Box::new(hir_type_to_layout(module, element)),
            len: *len,
        },
        hir::Type::Named { def, .. } => {
            let type_def = &module.type_defs[*def];
            match &type_def.kind {
                hir::TypeDefKind::Struct { fields } => {
                    let mut field_layouts = Vec::new();
                    let mut byte_offset: u64 = 0;
                    for field in fields {
                        field_layouts.push(kajit_types::FieldLayout {
                            name: field.name.clone(),
                            offset: byte_offset,
                            layout: hir_type_to_layout(module, &field.ty),
                        });
                        let word_count = hir_to_ir::word_count_for_type(module, &field.ty);
                        byte_offset += (word_count * 8) as u64;
                    }
                    TypeLayout::Struct {
                        name: type_def.name.clone(),
                        fields: field_layouts,
                    }
                }
                hir::TypeDefKind::Enum {
                    variants,
                    discriminant_width,
                } => {
                    let disc_size = discriminant_width.unwrap_or(1) as u8;
                    let variant_layouts = variants
                        .iter()
                        .enumerate()
                        .map(|(i, variant)| {
                            let mut field_layouts = Vec::new();
                            let mut byte_offset: u64 = 0;
                            for field in &variant.fields {
                                field_layouts.push(kajit_types::FieldLayout {
                                    name: field.name.clone(),
                                    offset: byte_offset,
                                    layout: hir_type_to_layout(module, &field.ty),
                                });
                                let word_count = hir_to_ir::word_count_for_type(module, &field.ty);
                                byte_offset += (word_count * 8) as u64;
                            }
                            kajit_types::VariantLayout {
                                name: variant.name.clone(),
                                discriminant: variant.discriminant.unwrap_or(i as i64),
                                fields: field_layouts,
                            }
                        })
                        .collect();
                    TypeLayout::Enum {
                        name: type_def.name.clone(),
                        discriminant_size: disc_size,
                        variants: variant_layouts,
                    }
                }
            }
        }
    }
}

/// Compile an HIR module into a callable scalar function.
///
/// This is the entry point for Vixen: takes an HIR `Module` containing a
/// plain scalar function (params, locals, return value — no cursor or
/// destination), runs it through the full pipeline (IR → passes → linearize
/// → CFG-MIR → regalloc → backend), and returns executable machine code
/// with standard calling convention.
pub fn compile_hir_module(module: &kajit_hir::Module) -> CompiledFunction {
    // Phase 1: HIR → IR
    let mut func = lower_hir_module(module);

    // Phase 2: IR optimization passes
    crate::ir_passes::run_default_passes(&mut func);

    // Phase 3: Linearize
    let linear = crate::linearize::linearize(&mut func);

    // Phase 4: CFG-MIR lowering + optimization
    let hints = Default::default();
    let cfg_program = crate::regalloc_engine::cfg_mir::lower_and_optimize(&linear, hints);

    // Phase 5: Register allocation
    let alloc = crate::regalloc_engine::allocate_cfg_program_regalloc3_native(&cfg_program)
        .unwrap_or_else(|err| panic!("regalloc3 allocation failed: {err}"));

    // Phase 6: Backend compilation
    let empty_symbols = kajit_types::SymbolTable::new();
    #[cfg(target_arch = "aarch64")]
    let result = crate::backends::aarch64::regalloc3_backend::compile_regalloc3(
        &alloc,
        &empty_symbols,
        crate::pipeline_opts::CompileTarget::Jit,
    );
    #[cfg(target_arch = "x86_64")]
    let result = crate::backends::x86_64::regalloc3_backend::compile_regalloc3(
        &alloc,
        &empty_symbols,
        crate::pipeline_opts::CompileTarget::Jit,
    );
    let entry = result.entry as usize;

    CompiledFunction {
        buf: result.buf,
        entry,
    }
}

/// Run the full compilation pipeline, producing all artifacts in one pass.
pub fn compile_pipeline(
    shape: &'static Shape,
    kind: DecoderKind,
    pipeline_opts: &PipelineOptions,
) -> PipelineArtifacts {
    let registry = symbol_registry_for_shape(shape);
    let (module, symbol_table) = build_decoder_hir(shape, kind);
    let mut opts = pipeline_opts.clone();
    opts.symbol_table = symbol_table;
    compile_pipeline_from_hir_module(&module, &registry, &opts)
}

/// Run the full compilation pipeline from an already-built HIR module.
///
/// This is primarily intended for handwritten HIR debugging and tooling.
pub fn compile_pipeline_from_hir_module(
    module: &kajit_hir::Module,
    registry: &crate::ir::IntrinsicRegistry,
    pipeline_opts: &PipelineOptions,
) -> PipelineArtifacts {
    let hir_text = module.to_string();

    if let Ok(path) = std::env::var("KAJIT_DUMP_HIR_DEBUG") {
        std::fs::write(&path, format!("{module:#?}")).unwrap();
        eprintln!("[debug] dumped HIR debug to {path}");
    }
    if let Ok(path) = std::env::var("KAJIT_DUMP_HIR_TEXT") {
        std::fs::write(&path, &hir_text).unwrap();
        eprintln!("[debug] dumped HIR text to {path}");
    }

    // Phase 2: IR + passes with timeline
    let mut func = lower_hir_module(module);
    let mut ir_opt_timeline = vec![(
        "initial".to_string(),
        format!("{}", func.display_with_registry(registry)),
    )];
    run_configured_default_passes_with_observer(&mut func, pipeline_opts, |pass_name, func| {
        ir_opt_timeline.push((
            pass_name.to_string(),
            format!("{}", func.display_with_registry(registry)),
        ));
    });

    // Phase 3: Linearize
    if let Ok(path) = std::env::var("KAJIT_DUMP_IR_BEFORE_LINEAR") {
        let ir_text = format!("{}", func.display_with_registry(registry));
        std::fs::write(&path, &ir_text).unwrap();
        eprintln!("[debug] dumped IR to {path} ({} bytes)", ir_text.len());
    }
    let linear = crate::linearize::linearize(&mut func);
    let linear_text = format!("{linear}");

    // Phase 4: CFG-MIR + optimize
    let trusted_utf8_input = false;

    // Phase 5: CFG-MIR lowering + optimization (ONCE — used for everything)
    let hints = Default::default();
    let mut cfg_program = crate::regalloc_engine::cfg_mir::lower_and_optimize(&linear, hints);

    // Attach data_arg layout metadata from HIR types (debug info for pointer tracking)
    cfg_program.data_arg_layouts = extract_data_arg_layouts(module);

    // Phase 6: Register allocation + backend compilation from the ONE cfg_program
    let ra3_alloc = crate::regalloc_engine::allocate_cfg_program_regalloc3_native(&cfg_program)
        .unwrap_or_else(|err| panic!("regalloc3 allocation failed: {err}"));

    #[cfg(target_arch = "aarch64")]
    let base_frame = crate::backends::aarch64::regalloc3_backend::compute_base_frame(&ra3_alloc);
    #[cfg(target_arch = "x86_64")]
    let base_frame = crate::backends::x86_64::regalloc3_backend::compute_base_frame(&ra3_alloc);
    let alloc_map = ra3_alloc
        .functions
        .first()
        .map(|f| crate::harness::AllocationMap::from_regalloc3(f, base_frame))
        .unwrap_or_default();

    let location_map = crate::harness::LocationMap::from_alloc_map_and_cfg(
        &alloc_map,
        &ra3_alloc.cfg_program,
        &ra3_alloc,
    );

    #[cfg(target_arch = "aarch64")]
    let result = crate::backends::aarch64::regalloc3_backend::compile_regalloc3(
        &ra3_alloc,
        &pipeline_opts.symbol_table,
        pipeline_opts.compile_target,
    );
    #[cfg(target_arch = "x86_64")]
    let result = crate::backends::x86_64::regalloc3_backend::compile_regalloc3(
        &ra3_alloc,
        &pipeline_opts.symbol_table,
        pipeline_opts.compile_target,
    );
    let intrinsic_call_sites = result.intrinsic_call_sites.clone();
    let extern_addr_relocs = result.extern_addr_relocs.clone();
    let (buf, entry, _source_map, backend_debug_info, asm_program) =
        materialize_backend_result(result);

    let func = unsafe { buf.code_ptr().add(entry) };

    let emitted_cfg_program = &ra3_alloc.cfg_program;
    let listing = dwarf::build_cfg_mir_listing(emitted_cfg_program, Some(registry));

    let decoder = CompiledDecoder {
        buf,
        cfg_mir_line_text_by_line: listing.line_text_by_line,
        entry,
        func,
        trusted_utf8_input,
        _jit_registration: None, // JIT debug registration handled by the old path if needed
        #[cfg(target_arch = "aarch64")]
        asm_program,
    };

    let cfg_text = decoder.cfg_mir_text();

    // Capture ASM
    #[cfg(target_arch = "aarch64")]
    let asm_text = decoder.assembly_text().unwrap_or_default();
    #[cfg(not(target_arch = "aarch64"))]
    let asm_text = String::new();

    PipelineArtifacts {
        hir_text,
        ir_opt_timeline,
        linear_text,
        cfg_text,
        cfg_canonical_text: String::new(),
        asm_text,
        alloc_map,
        location_map,
        intrinsic_call_sites,
        extern_addr_relocs,
        backend_debug_info,
        cfg_program: emitted_cfg_program.clone(),
        decoder,
    }
}

/// Produce the pre-optimization CFG-MIR Program for a given type/format.
///
/// This runs HIR → IR → linearize → CFG-MIR lowering but stops before any
/// CFG-MIR optimization passes. Used by the reducer to get the raw input CFG.
pub fn compile_pre_opt_cfg(
    shape: &'static facet::Shape,
    kind: DecoderKind,
    pipeline_opts: &PipelineOptions,
) -> kajit_mir::cfg_mir::Program {
    // Phase 1: HIR
    let (module, _symbol_table) = build_decoder_hir(shape, kind);

    // Phase 2: IR + passes
    let mut func = lower_hir_module(&module);
    run_configured_default_passes_with_observer(&mut func, pipeline_opts, |_, _| {});

    // Phase 3: Linearize
    let linear = crate::linearize::linearize(&mut func);

    // Phase 4: Lower to CFG-MIR (NO optimization passes)
    let hints = Default::default();
    crate::regalloc_engine::cfg_mir::lower_linear_ir(&linear, hints)
}

/// Compile a deserializer through RVSDG + linearization + backend adapter.
pub fn compile_decoder(shape: &'static Shape, kind: DecoderKind) -> CompiledDecoder {
    let pipeline_opts = PipelineOptions::from_env();
    compile_decoder_with_options(shape, kind, &pipeline_opts)
}

// r[impl compiler.opts.api]
pub fn compile_decoder_with_options(
    shape: &'static Shape,
    kind: DecoderKind,
    pipeline_opts: &PipelineOptions,
) -> CompiledDecoder {
    // Delegate to compile_pipeline so all paths share identical codegen
    compile_pipeline(shape, kind, pipeline_opts).decoder
}

/// Legacy entry point kept for backward compatibility.
#[allow(dead_code)]
fn compile_decoder_with_options_legacy(
    shape: &'static Shape,
    kind: DecoderKind,
    pipeline_opts: &PipelineOptions,
) -> CompiledDecoder {
    match kind {
        DecoderKind::Postcard if supports_postcard_decoder_hir(shape) => {
            compile_postcard_decoder_via_hir_with_options(shape, pipeline_opts.clone())
        }
        _ => {
            panic!(
                "unsupported shape for {kind:?} HIR: {}",
                shape.type_identifier,
            );
        }
    }
}

/// Build IR + linear form, compile through the backend, and return a deterministic emission trace.
pub fn emission_trace_text(shape: &'static Shape, kind: DecoderKind) -> String {
    let pipeline_opts = PipelineOptions::from_env();
    emission_trace_text_with_options(shape, kind, &pipeline_opts)
}

/// Build a decoder and return ARM64 assembly text.
#[cfg(target_arch = "aarch64")]
pub fn assembly_text(shape: &'static Shape, kind: DecoderKind) -> String {
    let pipeline_opts = PipelineOptions::from_env();
    assembly_text_with_options(shape, kind, &pipeline_opts)
}

#[cfg(target_arch = "aarch64")]
pub fn assembly_text_with_options(
    shape: &'static Shape,
    kind: DecoderKind,
    pipeline_opts: &PipelineOptions,
) -> String {
    let decoder = compile_decoder_with_options(shape, kind, pipeline_opts);
    decoder.assembly_text().unwrap_or_else(|| {
        panic!("assembly capture not available (should always be enabled on aarch64)")
    })
}

// r[impl compiler.opts.api]
pub fn emission_trace_text_with_options(
    shape: &'static Shape,
    kind: DecoderKind,
    pipeline_opts: &PipelineOptions,
) -> String {
    let decoder = compile_decoder_with_options(shape, kind, pipeline_opts);
    decoder
        .emission_trace_text()
        .unwrap_or_else(|err| panic!("failed to format emission trace: {err:?}"))
}

pub(crate) fn build_decoder_hir(
    shape: &'static Shape,
    kind: DecoderKind,
) -> (hir::Module, kajit_types::SymbolTable) {
    match kind {
        DecoderKind::Postcard => build_postcard_decoder_hir(shape),
    }
}

pub(crate) fn compile_postcard_decoder_via_hir_with_options(
    shape: &'static Shape,
    pipeline_opts: PipelineOptions,
) -> CompiledDecoder {
    let registry = symbol_registry_for_shape(shape);
    let (module, symbol_table) = build_postcard_decoder_hir(shape);
    let mut func = lower_hir_module(&module);
    eprintln!(
        "=== RVSDG IR before passes ===\n{}\n=== END IR ===",
        func.display_with_registry(&registry)
    );
    run_configured_default_passes(&mut func, &pipeline_opts);
    eprintln!(
        "=== RVSDG IR after passes ===\n{}\n=== END IR ===",
        func.display_with_registry(&registry)
    );
    let linear = crate::linearize::linearize(&mut func);
    let mut pipeline_opts = pipeline_opts;
    pipeline_opts.symbol_table = symbol_table;
    compile_linear_ir_decoder_with_options(
        &linear,
        false,
        pipeline_opts,
        Some(&registry),
        Some(shape),
    )
}

pub(crate) fn run_default_passes_from_env(func: &mut crate::ir::IrFunc) {
    let pipeline_opts = PipelineOptions::from_env();
    run_configured_default_passes(func, &pipeline_opts);
}

pub(crate) fn run_configured_default_passes(
    func: &mut crate::ir::IrFunc,
    pipeline_opts: &PipelineOptions,
) {
    run_configured_default_passes_with_observer(func, pipeline_opts, |_, _| {});
}

pub(crate) fn run_configured_default_passes_with_observer<F>(
    func: &mut crate::ir::IrFunc,
    pipeline_opts: &PipelineOptions,
    mut observe_after_pass: F,
) where
    F: FnMut(&str, &crate::ir::IrFunc),
{
    // r[impl compiler.opts.all-opts]
    if !pipeline_opts.resolve_all_opts(DEFAULT_PRE_LINEARIZATION_PASSES_ENABLED) {
        return;
    }

    for pass in crate::ir_passes::default_pass_registry() {
        if !pipeline_opts.resolve_pass(pass.name, true) {
            continue;
        }
        pass.run(func);
        observe_after_pass(pass.name, func);
    }
}

/// Compile a deserializer from already-linearized IR.
///
/// This is the first backend-adapter entrypoint used by the IR migration.
pub fn compile_linear_ir_decoder(
    ir: &crate::linearize::LinearIr,
    trusted_utf8_input: bool,
) -> CompiledDecoder {
    compile_linear_ir_decoder_with_options(
        ir,
        trusted_utf8_input,
        PipelineOptions::from_env(),
        None,
        None,
    )
}

/// Compile a deserializer directly from CFG-MIR.
///
/// This is primarily intended for regression tests and minimization workflows
/// where a failing CFG-MIR program is edited by hand and recompiled quickly.
pub fn compile_cfg_mir_decoder(
    cfg_program: &crate::regalloc_engine::cfg_mir::Program,
    trusted_utf8_input: bool,
) -> CompiledDecoder {
    compile_cfg_mir_decoder_with_registry(cfg_program, None, trusted_utf8_input)
}

pub(crate) fn compile_cfg_mir_decoder_with_registry(
    cfg_program: &crate::regalloc_engine::cfg_mir::Program,
    registry: Option<&crate::ir::IntrinsicRegistry>,
    trusted_utf8_input: bool,
) -> CompiledDecoder {
    compile_cfg_mir_decoder_with_options(
        cfg_program,
        registry,
        trusted_utf8_input,
        PipelineOptions::from_env(),
    )
}

fn compile_linear_ir_decoder_with_options(
    ir: &crate::linearize::LinearIr,
    trusted_utf8_input: bool,
    pipeline_opts: PipelineOptions,
    registry: Option<&crate::ir::IntrinsicRegistry>,
    root_shape: Option<&'static Shape>,
) -> CompiledDecoder {
    let jit_debug = jit_debug_enabled();
    let hints = Default::default(); // TODO: Call analyze_spill_costs before linearization
    let cfg_program = crate::regalloc_engine::cfg_mir::lower_and_optimize(ir, hints);

    let alloc = crate::regalloc_engine::allocate_cfg_program_regalloc3_native(&cfg_program)
        .unwrap_or_else(|err| panic!("regalloc3 allocation failed: {err}"));
    #[cfg(target_arch = "aarch64")]
    let result = crate::backends::aarch64::regalloc3_backend::compile_regalloc3(
        &alloc,
        &pipeline_opts.symbol_table,
        pipeline_opts.compile_target,
    );
    #[cfg(target_arch = "x86_64")]
    let result = crate::backends::x86_64::regalloc3_backend::compile_regalloc3(
        &alloc,
        &pipeline_opts.symbol_table,
        pipeline_opts.compile_target,
    );
    let (buf, entry, source_map, backend_debug_info, asm_program) =
        materialize_backend_result(result);
    #[cfg(target_arch = "aarch64")]
    let base_frame = crate::backends::aarch64::regalloc3_backend::compute_base_frame(&alloc);
    #[cfg(target_arch = "x86_64")]
    let base_frame = crate::backends::x86_64::regalloc3_backend::compute_base_frame(&alloc);
    let alloc_map = alloc
        .functions
        .first()
        .map(|f| crate::harness::AllocationMap::from_regalloc3(f, base_frame))
        .unwrap_or_default();
    let dwarf_location_map =
        crate::harness::LocationMap::from_alloc_map_and_cfg(&alloc_map, &alloc.cfg_program, &alloc);
    let func = unsafe { buf.code_ptr().add(entry) };
    let listing = build_cfg_mir_listing(&cfg_program, registry);
    let root_label = ir.ops.iter().find_map(|op| match op {
        crate::linearize::LinearOp::FuncStart {
            lambda_id, label, ..
        } if lambda_id.index() == 0 => Some(label.as_str()),
        _ => None,
    });
    let root_display_name = root_label
        .map(|l| format!("kajit::decode::{l}"))
        .unwrap_or_else(|| "kajit::decode::<ir-root>".to_string());
    let root_mangled_name = root_label
        .map(|l| crate::jit_debug::rust_v0_mangle(&["kajit", "decode", l]))
        .unwrap_or_else(|| crate::jit_debug::rust_v0_mangle(&["kajit", "decode", "ir_root"]));
    let symbol = crate::jit_debug::JitSymbolEntry {
        name: root_mangled_name,
        offset: entry,
        size: buf.len().saturating_sub(entry),
    };
    let registration = if jit_debug {
        let listing_path = write_cfg_mir_listing_file(&root_display_name, &listing.text);
        let mut debug_subprogram = cfg_mir_dwarf_variables(
            root_shape,
            &cfg_program,
            &dwarf_location_map,
            backend_debug_info.as_ref(),
            buf.code_ptr(),
            jit_dwarf_target_arch(),
        );
        debug_subprogram.name = root_display_name.clone();
        let dwarf = listing_path.as_deref().and_then(|path| {
            let debug_info = build_jit_debug_info_from_source_map(
                buf.code_ptr(),
                buf.len(),
                source_map.as_ref(),
                path,
                debug_subprogram.clone(),
            )?;
            crate::jit_dwarf::build_jit_dwarf_sections_from_debug_info(&debug_info).ok()
        });
        crate::jit_debug::register_jit_code_with_dwarf(
            buf.code_ptr(),
            buf.len(),
            &[symbol],
            dwarf.as_ref(),
        )
    } else {
        crate::jit_debug::register_jit_code(buf.code_ptr(), buf.len(), &[symbol])
    };

    CompiledDecoder {
        buf,
        cfg_mir_line_text_by_line: listing.line_text_by_line,
        entry,
        func,
        trusted_utf8_input,
        _jit_registration: Some(registration),
        #[cfg(target_arch = "aarch64")]
        asm_program,
    }
}

fn compile_cfg_mir_decoder_with_options(
    cfg_program: &crate::regalloc_engine::cfg_mir::Program,
    registry: Option<&crate::ir::IntrinsicRegistry>,
    trusted_utf8_input: bool,
    pipeline_opts: PipelineOptions,
) -> CompiledDecoder {
    let jit_debug = jit_debug_enabled();

    let alloc = crate::regalloc_engine::allocate_cfg_program_regalloc3_native(cfg_program)
        .unwrap_or_else(|err| panic!("regalloc3 allocation failed: {err}"));

    #[cfg(target_arch = "aarch64")]
    let result = crate::backends::aarch64::regalloc3_backend::compile_regalloc3(
        &alloc,
        &pipeline_opts.symbol_table,
        pipeline_opts.compile_target,
    );
    #[cfg(target_arch = "x86_64")]
    let result = crate::backends::x86_64::regalloc3_backend::compile_regalloc3(
        &alloc,
        &pipeline_opts.symbol_table,
        pipeline_opts.compile_target,
    );
    let (buf, entry, source_map, backend_debug_info, asm_program) =
        materialize_backend_result(result);
    #[cfg(target_arch = "aarch64")]
    let base_frame = crate::backends::aarch64::regalloc3_backend::compute_base_frame(&alloc);
    #[cfg(target_arch = "x86_64")]
    let base_frame = crate::backends::x86_64::regalloc3_backend::compute_base_frame(&alloc);
    let alloc_map = alloc
        .functions
        .first()
        .map(|f| crate::harness::AllocationMap::from_regalloc3(f, base_frame))
        .unwrap_or_default();
    let dwarf_location_map =
        crate::harness::LocationMap::from_alloc_map_and_cfg(&alloc_map, &alloc.cfg_program, &alloc);
    let func = unsafe { buf.code_ptr().add(entry) };
    let listing = build_cfg_mir_listing(cfg_program, registry);

    let root_display_name = "kajit::decode::cfg_mir_text".to_string();
    let root_mangled_name = crate::jit_debug::rust_v0_mangle(&["kajit", "decode", "cfg_mir_text"]);
    let symbol = crate::jit_debug::JitSymbolEntry {
        name: root_mangled_name,
        offset: entry,
        size: buf.len().saturating_sub(entry),
    };
    let registration = if jit_debug {
        let listing_path = write_cfg_mir_listing_file(&root_display_name, &listing.text);
        let mut debug_subprogram = cfg_mir_dwarf_variables(
            None,
            cfg_program,
            &dwarf_location_map,
            backend_debug_info.as_ref(),
            buf.code_ptr(),
            jit_dwarf_target_arch(),
        );
        debug_subprogram.name = root_display_name.clone();
        let dwarf = listing_path.as_deref().and_then(|path| {
            let debug_info = build_jit_debug_info_from_source_map(
                buf.code_ptr(),
                buf.len(),
                source_map.as_ref(),
                path,
                debug_subprogram.clone(),
            )?;
            crate::jit_dwarf::build_jit_dwarf_sections_from_debug_info(&debug_info).ok()
        });
        crate::jit_debug::register_jit_code_with_dwarf(
            buf.code_ptr(),
            buf.len(),
            &[symbol],
            dwarf.as_ref(),
        )
    } else {
        crate::jit_debug::register_jit_code(buf.code_ptr(), buf.len(), &[symbol])
    };

    CompiledDecoder {
        buf,
        cfg_mir_line_text_by_line: listing.line_text_by_line,
        entry,
        func,
        trusted_utf8_input,
        _jit_registration: Some(registration),
        #[cfg(target_arch = "aarch64")]
        asm_program,
    }
}

#[cfg(test)]
mod tests;
