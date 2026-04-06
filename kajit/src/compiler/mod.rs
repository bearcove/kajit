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
use crate::ir::{RegionBuilder, Width as IrWidth};
use crate::pipeline_opts::PipelineOptions;

#[cfg(test)]
pub(crate) use hir_to_ir::build_structural_hir_ir;
pub use hir_to_ir::lower_hir_module;
pub(crate) use kajit_postcard::{build_postcard_decoder_hir, supports_postcard_decoder_hir};

/// A compiled deserializer. Owns the executable buffer containing JIT'd machine code.
pub struct CompiledDecoder {
    #[cfg(target_arch = "x86_64")]
    buf: kajit_emit::x64::FinalizedEmission,
    #[cfg(target_arch = "aarch64")]
    buf: kajit_emit::aarch64::FinalizedEmission,
    cfg_mir_line_text_by_line: Vec<String>,
    entry: usize,
    func: unsafe extern "C" fn(*mut u8, *mut crate::context::DeserContext),
    root_data_abi: RootDecoderDataAbi,
    trusted_utf8_input: bool,
    _jit_registration: Option<crate::jit_debug::JitRegistration>,
    #[cfg(target_arch = "aarch64")]
    asm_program: Option<kajit_emit::aarch64_asm::Program>,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum RootDecoderDataAbi {
    #[default]
    None,
    CursorRef,
}

/// A compiled scalar function. Owns the executable buffer containing JIT'd machine code.
/// Uses standard calling convention: args in x0..x7, return value in x0.
pub struct CompiledFunction {
    #[cfg(target_arch = "x86_64")]
    buf: kajit_emit::x64::FinalizedEmission,
    #[cfg(target_arch = "aarch64")]
    buf: kajit_emit::aarch64::FinalizedEmission,
    entry: usize,
}

impl CompiledFunction {
    /// Get the entry point as a raw function pointer.
    /// The caller is responsible for casting to the correct signature.
    pub fn as_ptr(&self) -> *const u8 {
        #[cfg(target_arch = "aarch64")]
        {
            unsafe { self.buf.code_ptr().add(self.entry) }
        }
        #[cfg(target_arch = "x86_64")]
        {
            unsafe { self.buf.exec.as_ptr().add(self.entry) }
        }
    }

    /// The raw executable code buffer.
    pub fn code(&self) -> &[u8] {
        #[cfg(target_arch = "x86_64")]
        {
            return self.buf.exec.as_ref();
        }
        #[cfg(target_arch = "aarch64")]
        {
            &self.buf.code
        }
    }
}

impl CompiledDecoder {
    pub(crate) fn func(&self) -> unsafe extern "C" fn(*mut u8, *mut crate::context::DeserContext) {
        self.func
    }

    pub(crate) fn root_data_abi(&self) -> RootDecoderDataAbi {
        self.root_data_abi
    }

    /// The raw executable code buffer.
    pub fn code(&self) -> &[u8] {
        #[cfg(target_arch = "x86_64")]
        {
            return self.buf.exec.as_ref();
        }

        #[cfg(target_arch = "aarch64")]
        {
            &self.buf.code
        }
    }

    /// Byte offset of the entry point within the code buffer.
    pub fn entry_offset(&self) -> usize {
        self.entry
    }

    /// Whether `from_str` can safely enable trusted UTF-8 mode for this format.
    pub fn supports_trusted_utf8_input(&self) -> bool {
        self.trusted_utf8_input
    }

    pub fn uses_root_cursor_arg(&self) -> bool {
        matches!(self.root_data_abi, RootDecoderDataAbi::CursorRef)
    }

    /// Deterministic machine-emission trace annotated with CFG-MIR provenance.
    pub fn emission_trace_text(&self) -> Result<String, kajit_emit::TraceError> {
        #[cfg(target_arch = "x86_64")]
        let entries = self.buf.trace_entries()?;

        #[cfg(target_arch = "aarch64")]
        let entries = self.buf.trace_entries()?;

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
        #[cfg(target_arch = "aarch64")]
        {
            Some(&self.buf.source_map)
        }
        #[cfg(target_arch = "x86_64")]
        {
            Some(&self.buf.source_map)
        }
    }

    /// ARM64 assembly text (captured instructions before encoding).
    #[cfg(target_arch = "aarch64")]
    pub fn assembly_text(&self) -> Option<String> {
        self.asm_program.as_ref().map(|p| format!("{}", p))
    }
}

pub(crate) const DEFAULT_PRE_LINEARIZATION_PASSES_ENABLED: bool = true;

#[cfg(target_arch = "aarch64")]
pub(crate) fn materialize_backend_result(
    result: crate::ir_backend::LinearBackendResult,
) -> (
    kajit_emit::aarch64::FinalizedEmission,
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
    } = result;
    (
        buf,
        entry as usize,
        source_map,
        backend_debug_info,
        asm_program,
    )
}

#[cfg(target_arch = "x86_64")]
pub(crate) fn materialize_backend_result(
    result: crate::ir_backend::LinearBackendResult,
) -> (
    kajit_emit::x64::FinalizedEmission,
    usize,
    Option<kajit_emit::SourceMap>,
    Option<crate::ir_backend::BackendDebugInfo>,
    (), // placeholder for asm_program (aarch64-only)
) {
    let crate::ir_backend::LinearBackendResult {
        buf,
        entry,
        source_map,
        backend_debug_info,
        intrinsic_call_sites: _,
        data_relocs: _,
    } = result;
    (buf, entry as usize, source_map, backend_debug_info, ())
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
    #[cfg(target_arch = "aarch64")]
    pub intrinsic_call_sites:
        Vec<crate::backends::aarch64::regalloc3_backend::IntrinsicCallSiteInfo>,
    #[cfg(target_arch = "x86_64")]
    pub intrinsic_call_sites:
        Vec<crate::backends::x86_64::regalloc3_backend::IntrinsicCallSiteInfo>,
    /// Exact machine-code ranges for emitted CFG ops.
    pub backend_debug_info: Option<crate::ir_backend::BackendDebugInfo>,
    /// The post-optimization CFG-MIR program (same one the JIT compiled).
    /// Used by the lockstep debugger to run the interpreter on the exact same IR.
    pub cfg_program: kajit_mir::cfg_mir::Program,
    /// The compiled decoder (ready to execute)
    pub decoder: CompiledDecoder,
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
    #[cfg(target_arch = "aarch64")]
    let result = crate::backends::aarch64::regalloc3_backend::compile_regalloc3_with_root_data_abi(
        &alloc,
        RootDecoderDataAbi::None,
    );
    #[cfg(target_arch = "x86_64")]
    let result = crate::backends::x86_64::regalloc3_backend::compile_regalloc3_with_root_data_abi(
        &alloc,
        RootDecoderDataAbi::None,
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
    let module = build_decoder_hir(shape, kind);
    compile_pipeline_from_hir_module(&module, &registry, pipeline_opts)
}

/// Run the full compilation pipeline from an already-built HIR module.
///
/// This is primarily intended for handwritten HIR debugging and tooling.
pub fn compile_pipeline_from_hir_module(
    module: &kajit_hir::Module,
    registry: &crate::ir::IntrinsicRegistry,
    pipeline_opts: &PipelineOptions,
) -> PipelineArtifacts {
    let root_data_abi = infer_root_decoder_data_abi(module);
    let hir_text = module.to_string();

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
    let linear = crate::linearize::linearize(&mut func);
    let linear_text = format!("{linear}");

    // Phase 4: CFG-MIR + optimize
    let trusted_utf8_input = false;

    // Phase 5: CFG-MIR lowering + optimization (ONCE — used for everything)
    let hints = Default::default();
    let mut cfg_program = crate::regalloc_engine::cfg_mir::lower_and_optimize(&linear, hints);

    // For leaf decoder functions: exclude ABI arg registers from allocation
    // (kept for output_ptr/ctx_ptr). Scalar functions don't need this.
    if !cfg_program.is_scalar {
        use kajit_mir::regalloc3::machine_inst::PReg;

        if let Some(func) = cfg_program.funcs.first() {
            #[cfg(target_arch = "aarch64")]
            {
                cfg_program.extra_excluded_regs = (0..func.data_args.len())
                    .map(|i| PReg(i as u8 + 2))
                    .collect();
            }
            #[cfg(target_arch = "x86_64")]
            {
                // SysV: data_args arrive at rdx(2), rcx(1), r8(8), r9(9)
                #[cfg(not(windows))]
                const DATA_ARG_ENCS: &[u8] = &[2, 1, 8, 9];
                #[cfg(windows)]
                const DATA_ARG_ENCS: &[u8] = &[8, 9];
                cfg_program.extra_excluded_regs = func
                    .data_args
                    .iter()
                    .enumerate()
                    .filter_map(|(i, _)| DATA_ARG_ENCS.get(i).map(|&enc| PReg(enc)))
                    .collect();
            }
        }

        let is_leaf = cfg_program.funcs.iter().all(|func| {
            func.insts.iter().all(|inst| {
                !matches!(
                    inst.op,
                    kajit_lir::LinearOp::CallIntrinsic { .. }
                        | kajit_lir::LinearOp::CallPure { .. }
                        | kajit_lir::LinearOp::CallEffect { .. }
                        | kajit_lir::LinearOp::CallLambda { .. }
                )
            })
        });
        if is_leaf {
            #[cfg(target_arch = "aarch64")]
            cfg_program
                .extra_excluded_regs
                .extend([PReg(0), PReg(1), PReg(15)]);
            #[cfg(target_arch = "x86_64")]
            {
                #[cfg(not(windows))]
                cfg_program.extra_excluded_regs.extend([PReg(7), PReg(6)]); // rdi, rsi
                #[cfg(windows)]
                cfg_program.extra_excluded_regs.extend([PReg(1), PReg(2)]); // rcx, rdx
            }
        }
    }

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
    let result = crate::backends::aarch64::regalloc3_backend::compile_regalloc3_with_root_data_abi(
        &ra3_alloc,
        root_data_abi,
    );
    #[cfg(target_arch = "x86_64")]
    let result = crate::backends::x86_64::regalloc3_backend::compile_regalloc3_with_root_data_abi(
        &ra3_alloc,
        root_data_abi,
    );
    let intrinsic_call_sites = result.intrinsic_call_sites.clone();
    let (buf, entry, _source_map, backend_debug_info, asm_program) =
        materialize_backend_result(result);

    let func: unsafe extern "C" fn(*mut u8, *mut crate::context::DeserContext) =
        unsafe { core::mem::transmute(buf.code_ptr().add(entry)) };

    let emitted_cfg_program = &ra3_alloc.cfg_program;
    let listing = dwarf::build_cfg_mir_listing(emitted_cfg_program, Some(registry));

    let decoder = CompiledDecoder {
        buf,
        cfg_mir_line_text_by_line: listing.line_text_by_line,
        entry,
        func,
        root_data_abi,
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
    let module = build_decoder_hir(shape, kind);

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

// r[impl ir.regalloc.regressions]
/// Build IR + linear form and run regalloc over it, returning total edit count.
///
/// This is a full-pipeline diagnostic helper, not a lightweight metric.
pub fn regalloc_edit_count(shape: &'static Shape, kind: DecoderKind) -> usize {
    let pipeline_opts = PipelineOptions::from_env();
    regalloc_edit_count_with_options(shape, kind, &pipeline_opts)
}

/// Build IR + linear form and run regalloc, returning a detailed edits dump.
pub fn regalloc_edits_text(shape: &'static Shape, kind: DecoderKind) -> String {
    let pipeline_opts = PipelineOptions::from_env();
    regalloc_edits_text_with_options(shape, kind, &pipeline_opts)
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
pub fn regalloc_edit_count_with_options(
    shape: &'static Shape,
    kind: DecoderKind,
    pipeline_opts: &PipelineOptions,
) -> usize {
    let module = build_decoder_hir(shape, kind);
    let mut func = lower_hir_module(&module);
    run_configured_default_passes(&mut func, pipeline_opts);
    let linear = crate::linearize::linearize(&mut func);
    let hints = Default::default(); // TODO: Call analyze_spill_costs(&func) before linearization
    let cfg_program = crate::regalloc_engine::cfg_mir::lower_linear_ir(&linear, hints);
    let alloc = crate::regalloc_engine::allocate_cfg_program(&cfg_program)
        .unwrap_or_else(|err| panic!("regalloc2 allocation failed while counting edits: {err}"));
    alloc.functions.iter().map(|f| f.edits.len()).sum()
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

/// Same as [`regalloc_edits_text`], but with explicit pipeline options.
pub fn regalloc_edits_text_with_options(
    shape: &'static Shape,
    kind: DecoderKind,
    pipeline_opts: &PipelineOptions,
) -> String {
    let module = build_decoder_hir(shape, kind);
    let mut func = lower_hir_module(&module);
    run_configured_default_passes(&mut func, pipeline_opts);
    let linear = crate::linearize::linearize(&mut func);
    let hints = Default::default(); // TODO: Call analyze_spill_costs(&func) before linearization
    let cfg_program = crate::regalloc_engine::cfg_mir::lower_linear_ir(&linear, hints);
    let alloc = crate::regalloc_engine::allocate_cfg_program(&cfg_program)
        .unwrap_or_else(|err| panic!("regalloc2 allocation failed while formatting edits: {err}"));
    format_allocated_regalloc_edits(&alloc)
}

pub(crate) fn format_allocated_regalloc_edits(
    alloc: &crate::regalloc_engine::AllocatedCfgProgram,
) -> String {
    let mut out = String::new();
    let total_pp_edits: usize = alloc.functions.iter().map(|f| f.edits.len()).sum();
    let total_edge_edits: usize = alloc.functions.iter().map(|f| f.edge_edits.len()).sum();
    let _ = std::fmt::Write::write_fmt(
        &mut out,
        format_args!(
            "total_progpoint_edits: {total_pp_edits}\ntotal_edge_edits: {total_edge_edits}\n"
        ),
    );

    for func in &alloc.functions {
        let _ = std::fmt::Write::write_fmt(
            &mut out,
            format_args!(
                "\nlambda @{}:\n  num_spillslots: {}\n  progpoint_edits ({}):\n",
                func.lambda_id.index(),
                func.num_spillslots,
                func.edits.len()
            ),
        );
        for (prog_point, edit) in &func.edits {
            let _ = std::fmt::Write::write_fmt(
                &mut out,
                format_args!("    - {:?}: {:?}\n", prog_point, edit),
            );
        }

        let _ = std::fmt::Write::write_fmt(
            &mut out,
            format_args!("  edge_edits ({}):\n", func.edge_edits.len()),
        );
        for edge in &func.edge_edits {
            let _ = std::fmt::Write::write_fmt(
                &mut out,
                format_args!(
                    "    - edge e{} pos={:?} move {:?} -> {:?}\n",
                    edge.edge.0, edge.pos, edge.from, edge.to
                ),
            );
        }
    }

    out
}

pub(crate) fn build_decoder_hir(shape: &'static Shape, kind: DecoderKind) -> hir::Module {
    match kind {
        DecoderKind::Postcard => build_postcard_decoder_hir(shape),
    }
}

pub(crate) fn compile_postcard_decoder_via_hir_with_options(
    shape: &'static Shape,
    pipeline_opts: PipelineOptions,
) -> CompiledDecoder {
    let registry = symbol_registry_for_shape(shape);
    let module = build_postcard_decoder_hir(shape);
    let root_data_abi = infer_root_decoder_data_abi(&module);
    let mut func = lower_hir_module(&module);
    run_configured_default_passes(&mut func, &pipeline_opts);
    let linear = crate::linearize::linearize(&mut func);
    compile_linear_ir_decoder_with_options(
        &linear,
        false,
        pipeline_opts,
        Some(&registry),
        Some(shape),
        root_data_abi,
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
        RootDecoderDataAbi::None,
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
    _pipeline_opts: PipelineOptions,
    registry: Option<&crate::ir::IntrinsicRegistry>,
    root_shape: Option<&'static Shape>,
    root_data_abi: RootDecoderDataAbi,
) -> CompiledDecoder {
    let jit_debug = jit_debug_enabled();
    let hints = Default::default(); // TODO: Call analyze_spill_costs before linearization
    let mut cfg_program = crate::regalloc_engine::cfg_mir::lower_and_optimize(ir, hints);
    let root_data_abi = match root_data_abi {
        RootDecoderDataAbi::None => infer_root_decoder_data_abi_from_cfg(&cfg_program),
        explicit => explicit,
    };

    // Exclude ABI registers from allocation
    #[cfg(target_arch = "aarch64")]
    {
        use kajit_mir::regalloc3::machine_inst::PReg;

        if !cfg_program.is_scalar
            && let Some(func) = cfg_program.funcs.first()
        {
            cfg_program.extra_excluded_regs = (0..func.data_args.len())
                .map(|i| PReg(i as u8 + 2))
                .collect();
        }

        let is_leaf = cfg_program.funcs.iter().all(|func| {
            func.insts.iter().all(|inst| {
                !matches!(
                    inst.op,
                    kajit_lir::LinearOp::CallIntrinsic { .. }
                        | kajit_lir::LinearOp::CallPure { .. }
                        | kajit_lir::LinearOp::CallEffect { .. }
                        | kajit_lir::LinearOp::CallLambda { .. }
                )
            })
        });
        if is_leaf {
            // x0/x1: keep output_ptr/ctx_ptr in place (no moves to x21/x22)
            // x15: reserved for cursor writeback (RestoreCursor writes here
            //      instead of x19, avoiding callee-save overhead)
            cfg_program
                .extra_excluded_regs
                .extend([PReg(0), PReg(1), PReg(15)]);
        }
    }

    #[cfg(target_arch = "x86_64")]
    {
        use kajit_mir::regalloc3::machine_inst::PReg;

        if !cfg_program.is_scalar {
            if let Some(func) = cfg_program.funcs.first() {
                // Exclude data_arg ABI positions: SysV rdx(2), rcx(1), r8(8), r9(9)
                #[cfg(not(windows))]
                const DATA_ARG_ENCS: &[u8] = &[2, 1, 8, 9];
                #[cfg(windows)]
                const DATA_ARG_ENCS: &[u8] = &[8, 9];
                cfg_program.extra_excluded_regs = func
                    .data_args
                    .iter()
                    .enumerate()
                    .filter_map(|(i, _)| DATA_ARG_ENCS.get(i).map(|&enc| PReg(enc)))
                    .collect();
            }
        }

        let is_leaf = cfg_program.funcs.iter().all(|func| {
            func.insts.iter().all(|inst| {
                !matches!(
                    inst.op,
                    kajit_lir::LinearOp::CallIntrinsic { .. }
                        | kajit_lir::LinearOp::CallPure { .. }
                        | kajit_lir::LinearOp::CallEffect { .. }
                        | kajit_lir::LinearOp::CallLambda { .. }
                )
            })
        });
        if is_leaf {
            // rdi(7)/rsi(6): keep output_ptr/ctx_ptr in ABI arg registers
            #[cfg(not(windows))]
            cfg_program.extra_excluded_regs.extend([PReg(7), PReg(6)]);
            #[cfg(windows)]
            cfg_program.extra_excluded_regs.extend([PReg(1), PReg(2)]);
        }
    }

    let alloc = crate::regalloc_engine::allocate_cfg_program_regalloc3_native(&cfg_program)
        .unwrap_or_else(|err| panic!("regalloc3 allocation failed: {err}"));
    #[cfg(target_arch = "aarch64")]
    let result = crate::backends::aarch64::regalloc3_backend::compile_regalloc3_with_root_data_abi(
        &alloc,
        root_data_abi,
    );
    #[cfg(target_arch = "x86_64")]
    let result = crate::backends::x86_64::regalloc3_backend::compile_regalloc3_with_root_data_abi(
        &alloc,
        root_data_abi,
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
    let func: unsafe extern "C" fn(*mut u8, *mut crate::context::DeserContext) =
        unsafe { core::mem::transmute(buf.code_ptr().add(entry)) };
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
        root_data_abi,
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
    _pipeline_opts: PipelineOptions,
) -> CompiledDecoder {
    let jit_debug = jit_debug_enabled();
    let apply_regalloc_edits = true;

    let root_data_abi = infer_root_decoder_data_abi_from_cfg(cfg_program);
    let regalloc_alloc = crate::regalloc_engine::allocate_cfg_program(cfg_program)
        .unwrap_or_else(|err| panic!("regalloc2 allocation failed: {err}"));
    let dwarf_location_map = crate::harness::LocationMap::default();

    let shim_linear = crate::linearize::LinearIr {
        ops: Vec::new(),
        label_count: 0,
        vreg_count: cfg_program.vreg_count,
        slot_count: cfg_program.slot_count,
        param_slot_count: cfg_program.param_slot_count,
        is_scalar: cfg_program.is_scalar,
        debug: Default::default(),
        data_blobs: cfg_program.data_blobs.clone(),
    };
    let (buf, entry, source_map, backend_debug_info, asm_program) = {
        let result = crate::ir_backend::compile_linear_ir_with_alloc_and_mode(
            &shim_linear,
            cfg_program,
            &regalloc_alloc,
            apply_regalloc_edits,
            root_data_abi,
            registry,
        );
        materialize_backend_result(result)
    };
    let func: unsafe extern "C" fn(*mut u8, *mut crate::context::DeserContext) =
        unsafe { core::mem::transmute(buf.code_ptr().add(entry)) };
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
        root_data_abi,
        trusted_utf8_input,
        _jit_registration: Some(registration),
        #[cfg(target_arch = "aarch64")]
        asm_program,
    }
}

fn infer_root_decoder_data_abi(module: &hir::Module) -> RootDecoderDataAbi {
    let Some((_, function)) = module.functions.iter().next() else {
        return RootDecoderDataAbi::None;
    };
    let non_destination_params: Vec<_> = function
        .params
        .iter()
        .filter(|param| !param.is_destination())
        .collect();
    match non_destination_params.as_slice() {
        [] => RootDecoderDataAbi::None,
        [param] if matches!(param.ty, hir::Type::Ref { mutable: true, .. }) => {
            RootDecoderDataAbi::CursorRef
        }
        _ => RootDecoderDataAbi::None,
    }
}

fn infer_root_decoder_data_abi_from_cfg(
    cfg_program: &crate::regalloc_engine::cfg_mir::Program,
) -> RootDecoderDataAbi {
    let Some(root) = cfg_program.funcs.first() else {
        return RootDecoderDataAbi::None;
    };
    match root.data_args.as_slice() {
        [] => RootDecoderDataAbi::None,
        [_] => RootDecoderDataAbi::CursorRef,
        _ => RootDecoderDataAbi::None,
    }
}

#[cfg(test)]
mod tests;
