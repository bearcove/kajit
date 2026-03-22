mod dwarf;
mod hir_to_ir;
mod shape_utils;

use dwarf::*;
pub(crate) use shape_utils::*;

use std::collections::{BTreeMap, HashMap, HashSet};
use std::path::{Path, PathBuf};

use facet::{Def, Facet, OptionDef, ScalarType, Shape, Type, UserType};
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
}

pub(crate) const DEFAULT_PRE_LINEARIZATION_PASSES_ENABLED: bool = true;

#[cfg(target_arch = "aarch64")]
fn materialize_backend_result(
    result: crate::ir_backend::LinearBackendResult,
) -> (
    kajit_emit::aarch64::FinalizedEmission,
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

#[cfg(target_arch = "x86_64")]
fn materialize_backend_result(
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
    let cfg_program = crate::regalloc_engine::cfg_mir::lower_linear_ir(&linear);
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
    let cfg_program = crate::regalloc_engine::cfg_mir::lower_linear_ir(&linear);
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

fn format_allocated_regalloc_edits(alloc: &crate::regalloc_engine::AllocatedCfgProgram) -> String {
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
    let module = build_postcard_decoder_hir(shape);
    let mut func = build_structural_hir_ir(shape, &module);
    run_configured_default_passes(&mut func, &pipeline_opts);
    let linear = crate::linearize::linearize(&mut func);
    compile_linear_ir_decoder_with_options(&linear, false, pipeline_opts)
}

pub(crate) fn compile_json_decoder_via_hir_with_options(
    shape: &'static Shape,
    pipeline_opts: PipelineOptions,
) -> CompiledDecoder {
    let module = build_json_decoder_hir(shape);
    let mut func = build_structural_hir_ir(shape, &module);
    run_configured_default_passes(&mut func, &pipeline_opts);
    let linear = crate::linearize::linearize(&mut func);
    compile_linear_ir_decoder_with_options(&linear, true, pipeline_opts)
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
    compile_linear_ir_decoder_with_options(ir, trusted_utf8_input, PipelineOptions::from_env())
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
) -> CompiledDecoder {
    let jit_debug = jit_debug_enabled();
    let apply_regalloc_edits = pipeline_opts.resolve_regalloc(true);

    let cfg_program = crate::regalloc_engine::cfg_mir::lower_and_optimize(ir);
    let regalloc_alloc = if apply_regalloc_edits {
        let mut alloc = crate::regalloc_engine::allocate_cfg_program(&cfg_program)
            .unwrap_or_else(|err| panic!("regalloc2 allocation failed: {err}"));
        maybe_disable_regalloc_edits_cfg(&mut alloc, &pipeline_opts);
        alloc
    } else {
        no_regalloc_alloc_for_cfg_program(&cfg_program)
    };

    let (buf, entry, source_map, backend_debug_info) = {
        let result = crate::ir_backend::compile_linear_ir_with_alloc_and_mode(
            ir,
            &cfg_program,
            &regalloc_alloc,
            apply_regalloc_edits,
        );
        materialize_backend_result(result)
    };
    let func: unsafe extern "C" fn(*mut u8, *mut crate::context::DeserContext) =
        unsafe { core::mem::transmute(buf.code_ptr().add(entry)) };
    let root_shape = ir.ops.iter().find_map(|op| match op {
        crate::linearize::LinearOp::FuncStart {
            lambda_id, shape, ..
        } if lambda_id.index() == 0 => Some(*shape),
        _ => None,
    });
    let registry = root_shape.map(symbol_registry_for_shape);
    let listing = build_cfg_mir_listing(&cfg_program, registry.as_ref());
    let root_display_name = ir
        .ops
        .iter()
        .find_map(|op| match op {
            crate::linearize::LinearOp::FuncStart {
                lambda_id, shape, ..
            } if lambda_id.index() == 0 => {
                Some(format!("kajit::decode::{}", shape.type_identifier))
            }
            _ => None,
        })
        .unwrap_or_else(|| "kajit::decode::<ir-root>".to_string());
    let root_mangled_name = ir
        .ops
        .iter()
        .find_map(|op| match op {
            crate::linearize::LinearOp::FuncStart {
                lambda_id, shape, ..
            } if lambda_id.index() == 0 => Some(crate::jit_debug::rust_v0_mangle(&[
                "kajit",
                "decode",
                shape.type_identifier,
            ])),
            _ => None,
        })
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
    let (buf, entry, source_map, backend_debug_info) = {
        let result = crate::ir_backend::compile_linear_ir_with_alloc_and_mode(
            &shim_linear,
            cfg_program,
            &regalloc_alloc,
            apply_regalloc_edits,
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
    }
}

#[cfg(test)]
mod tests;
