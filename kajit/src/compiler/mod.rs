mod dwarf;
mod hir_to_ir;
mod shape_utils;

use dwarf::*;
pub(crate) use shape_utils::*;

use std::collections::{BTreeMap, HashMap, HashSet};
use std::path::{Path, PathBuf};

use facet::{Def, OptionDef, ScalarType, Shape, Type, UserType};
use kajit_hir as hir;

use crate::format::{DecoderKind, FieldEmitInfo, SkippedFieldInfo};
use crate::intrinsics;
use crate::ir::{RegionBuilder, Width as IrWidth};
use crate::pipeline_opts::PipelineOptions;

pub(crate) use hir_to_ir::{build_postcard_decoder_ir_via_hir, build_structural_hir_ir};
pub(crate) use kajit_json::{build_json_decoder_hir, supports_json_decoder_hir};
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
    trusted_utf8_input: bool,
    _jit_registration: Option<crate::jit_debug::JitRegistration>,
    #[cfg(target_arch = "aarch64")]
    asm_program: Option<kajit_emit::aarch64_asm::Program>,
}

impl CompiledDecoder {
    pub(crate) fn func(&self) -> unsafe extern "C" fn(*mut u8, *mut crate::context::DeserContext) {
        self.func
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

        let rows: Vec<crate::jit_dwarf::JitDebugLineRow> = source_map
            .iter()
            .filter(|entry| entry.location.line != 0)
            .map(|entry| crate::jit_dwarf::JitDebugLineRow {
                code_offset: entry.offset,
                line: entry.location.line,
            })
            .collect();

        eprintln!(
            "[dwarf] source map: {} entries, {} with line!=0",
            source_map.len(),
            rows.len()
        );
        if let Some(first) = rows.first() {
            eprintln!(
                "[dwarf] first row: offset={}, line={}",
                first.code_offset, first.line
            );
        }
        if let Some(last) = rows.last() {
            eprintln!(
                "[dwarf] last row: offset={}, line={}",
                last.code_offset, last.line
            );
        }

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
) {
    let crate::ir_backend::LinearBackendResult {
        buf,
        entry,
        source_map,
        backend_debug_info,
    } = result;
    (buf, entry as usize, source_map, backend_debug_info)
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
    /// VReg → physical location map (for lockstep debugger)
    pub alloc_map: crate::harness::AllocationMap,
    /// The post-optimization CFG-MIR program (same one the JIT compiled).
    /// Used by the lockstep debugger to run the interpreter on the exact same IR.
    pub cfg_program: kajit_mir::cfg_mir::Program,
    /// The compiled decoder (ready to execute)
    pub decoder: CompiledDecoder,
}

/// Run the full compilation pipeline, producing all artifacts in one pass.
pub fn compile_pipeline(
    shape: &'static Shape,
    kind: DecoderKind,
    pipeline_opts: &PipelineOptions,
) -> PipelineArtifacts {
    let registry = symbol_registry_for_shape(shape);

    // Phase 1: HIR
    let module = match kind {
        DecoderKind::Postcard => build_postcard_decoder_hir(shape),
        DecoderKind::Json => build_json_decoder_hir(shape),
    };
    let hir_text = module.to_string();

    // Phase 2: IR + passes with timeline
    let mut func = build_structural_hir_ir(shape, &module);
    let mut ir_opt_timeline = vec![(
        "initial".to_string(),
        format!("{}", func.display_with_registry(&registry)),
    )];
    run_configured_default_passes_with_observer(&mut func, pipeline_opts, |pass_name, func| {
        ir_opt_timeline.push((
            pass_name.to_string(),
            format!("{}", func.display_with_registry(&registry)),
        ));
    });

    // Phase 3: Linearize
    let linear = crate::linearize::linearize(&mut func);
    let linear_text = format!("{linear}");

    // Phase 4: CFG-MIR + optimize
    let trusted_utf8_input = matches!(kind, DecoderKind::Json);

    // Phase 5: CFG-MIR lowering + optimization (ONCE — used for everything)
    let hints = Default::default();
    let cfg_program = crate::regalloc_engine::cfg_mir::lower_and_optimize(&linear, hints);

    // Phase 6: Register allocation + backend compilation from the ONE cfg_program
    let ra3_alloc = crate::regalloc_engine::allocate_cfg_program_regalloc3_native(&cfg_program)
        .unwrap_or_else(|err| panic!("regalloc3 allocation failed: {err}"));

    let alloc_map = ra3_alloc
        .functions
        .first()
        .map(|f| crate::harness::AllocationMap::from_regalloc3(f))
        .unwrap_or_default();

    let result = crate::backends::aarch64::regalloc3_backend::compile_regalloc3(&ra3_alloc);
    let (buf, entry, source_map, _backend_debug_info, asm_program) =
        materialize_backend_result(result);

    let func: unsafe extern "C" fn(*mut u8, *mut crate::context::DeserContext) =
        unsafe { core::mem::transmute(buf.code_ptr().add(entry)) };

    let listing = dwarf::build_cfg_mir_listing(&cfg_program, Some(&registry));

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
        cfg_program: ra3_alloc.cfg_program.clone(),
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
    let module = match kind {
        DecoderKind::Postcard => build_postcard_decoder_hir(shape),
        DecoderKind::Json => build_json_decoder_hir(shape),
    };

    // Phase 2: IR + passes
    let mut func = build_structural_hir_ir(shape, &module);
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
fn compile_decoder_with_options_legacy(
    shape: &'static Shape,
    kind: DecoderKind,
    pipeline_opts: &PipelineOptions,
) -> CompiledDecoder {
    match kind {
        DecoderKind::Json if supports_json_decoder_hir(shape) => {
            compile_json_decoder_via_hir_with_options(shape, pipeline_opts.clone())
        }
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
    if !pipeline_opts.resolve_regalloc(true) {
        return 0;
    }
    let mut func = build_decoder_ir_via_hir(shape, kind);
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
    let mut func = build_decoder_ir_via_hir(shape, kind);
    run_configured_default_passes(&mut func, pipeline_opts);
    let linear = crate::linearize::linearize(&mut func);
    let hints = Default::default(); // TODO: Call analyze_spill_costs(&func) before linearization
    let cfg_program = crate::regalloc_engine::cfg_mir::lower_linear_ir(&linear, hints);
    let alloc = if pipeline_opts.resolve_regalloc(true) {
        let mut alloc =
            crate::regalloc_engine::allocate_cfg_program(&cfg_program).unwrap_or_else(|err| {
                panic!("regalloc2 allocation failed while formatting edits: {err}")
            });
        maybe_disable_regalloc_edits_cfg(&mut alloc, pipeline_opts);
        alloc
    } else {
        no_regalloc_alloc_for_cfg_program(&cfg_program)
    };
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

/// Build decoder IR through HIR.
pub(crate) fn build_decoder_ir_via_hir(
    shape: &'static Shape,
    kind: DecoderKind,
) -> crate::ir::IrFunc {
    match kind {
        DecoderKind::Json => {
            let module = build_json_decoder_hir(shape);
            build_structural_hir_ir(shape, &module)
        }
        DecoderKind::Postcard => {
            let module = build_postcard_decoder_hir(shape);
            build_structural_hir_ir(shape, &module)
        }
    }
}

pub(crate) fn compile_postcard_decoder_via_hir_with_options(
    shape: &'static Shape,
    pipeline_opts: PipelineOptions,
) -> CompiledDecoder {
    let registry = symbol_registry_for_shape(shape);
    let module = build_postcard_decoder_hir(shape);
    let mut func = build_structural_hir_ir(shape, &module);
    run_configured_default_passes(&mut func, &pipeline_opts);
    let linear = crate::linearize::linearize(&mut func);
    compile_linear_ir_decoder_with_options(
        &linear,
        false,
        pipeline_opts,
        Some(&registry),
        Some(shape),
    )
}

pub(crate) fn compile_json_decoder_via_hir_with_options(
    shape: &'static Shape,
    pipeline_opts: PipelineOptions,
) -> CompiledDecoder {
    let registry = symbol_registry_for_shape(shape);
    let module = build_json_decoder_hir(shape);
    let mut func = build_structural_hir_ir(shape, &module);
    run_configured_default_passes(&mut func, &pipeline_opts);
    let linear = crate::linearize::linearize(&mut func);
    compile_linear_ir_decoder_with_options(
        &linear,
        true,
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

fn maybe_disable_regalloc_edits_cfg(
    alloc: &mut crate::regalloc_engine::AllocatedCfgProgram,
    pipeline_opts: &PipelineOptions,
) {
    if pipeline_opts.resolve_regalloc(true) {
        return;
    }

    for func in &mut alloc.functions {
        func.edits.clear();
        func.edge_edits.clear();
    }
}

fn no_regalloc_alloc_for_cfg_program(
    cfg_program: &crate::regalloc_engine::cfg_mir::Program,
) -> crate::regalloc_engine::AllocatedCfgProgram {
    let functions = cfg_program
        .funcs
        .iter()
        .map(|func| crate::regalloc_engine::AllocatedCfgFunction {
            lambda_id: func.lambda_id,
            num_spillslots: 0,
            edits: Vec::new(),
            op_allocs: std::collections::HashMap::new(),
            op_operands: std::collections::HashMap::new(),
            edge_edits: Vec::new(),
            return_result_allocs: Vec::new(),
        })
        .collect();

    crate::regalloc_engine::AllocatedCfgProgram {
        cfg_program: cfg_program.clone(),
        functions,
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
    let apply_regalloc_edits = pipeline_opts.resolve_regalloc(true);

    let hints = Default::default(); // TODO: Call analyze_spill_costs before linearization
    let cfg_program = crate::regalloc_engine::cfg_mir::lower_and_optimize(ir, hints);

    // Use regalloc3 by default, opt out with KAJIT_USE_REGALLOC2=1
    let use_regalloc3 = std::env::var("KAJIT_USE_REGALLOC2").is_err();

    let (buf, entry, source_map, backend_debug_info, asm_program, regalloc_alloc) = if use_regalloc3
    {
        let alloc = crate::regalloc_engine::allocate_cfg_program_regalloc3_native(&cfg_program)
            .unwrap_or_else(|err| panic!("regalloc3 allocation failed: {err}"));
        let result = crate::backends::aarch64::regalloc3_backend::compile_regalloc3(&alloc);
        let (buf, entry, source_map, backend_debug_info, asm_program) =
            materialize_backend_result(result);
        // Create a dummy regalloc2 alloc for DWARF/debug info (just needs cfg_program)
        let dummy_alloc = no_regalloc_alloc_for_cfg_program(&cfg_program);
        (
            buf,
            entry,
            source_map,
            backend_debug_info,
            asm_program,
            dummy_alloc,
        )
    } else {
        let regalloc_alloc = if apply_regalloc_edits {
            let mut alloc = crate::regalloc_engine::allocate_cfg_program(&cfg_program)
                .unwrap_or_else(|err| panic!("regalloc2 allocation failed: {err}"));
            maybe_disable_regalloc_edits_cfg(&mut alloc, &pipeline_opts);
            alloc
        } else {
            no_regalloc_alloc_for_cfg_program(&cfg_program)
        };

        let (buf, entry, source_map, backend_debug_info, asm_program) = {
            let result = crate::ir_backend::compile_linear_ir_with_alloc_and_mode(
                ir,
                &cfg_program,
                &regalloc_alloc,
                apply_regalloc_edits,
                registry,
            );
            materialize_backend_result(result)
        };
        (
            buf,
            entry,
            source_map,
            backend_debug_info,
            asm_program,
            regalloc_alloc,
        )
    };
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
            &regalloc_alloc,
            backend_debug_info.as_ref(),
            buf.code_ptr(),
            jit_dwarf_target_arch(),
            apply_regalloc_edits,
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
    let apply_regalloc_edits = pipeline_opts.resolve_regalloc(true);

    let regalloc_alloc = if apply_regalloc_edits {
        let mut alloc = crate::regalloc_engine::allocate_cfg_program(cfg_program)
            .unwrap_or_else(|err| panic!("regalloc2 allocation failed: {err}"));
        maybe_disable_regalloc_edits_cfg(&mut alloc, &pipeline_opts);
        alloc
    } else {
        no_regalloc_alloc_for_cfg_program(cfg_program)
    };

    let shim_linear = crate::linearize::LinearIr {
        ops: Vec::new(),
        label_count: 0,
        vreg_count: cfg_program.vreg_count,
        slot_count: cfg_program.slot_count,
        debug: Default::default(),
    };
    let (buf, entry, source_map, backend_debug_info, asm_program) = {
        let result = crate::ir_backend::compile_linear_ir_with_alloc_and_mode(
            &shim_linear,
            cfg_program,
            &regalloc_alloc,
            apply_regalloc_edits,
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
            &regalloc_alloc,
            backend_debug_info.as_ref(),
            buf.code_ptr(),
            jit_dwarf_target_arch(),
            apply_regalloc_edits,
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
