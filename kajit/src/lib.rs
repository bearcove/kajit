#![allow(clippy::approx_constant)]

pub mod alt_asm;
pub(crate) mod arch;
pub mod backends;
pub mod compiler;
pub mod context;
pub mod disasm_normalize;
pub mod format;
pub mod intrinsics;
pub mod ir;
pub mod ir_backend;

pub mod harness;
pub mod ir_parse;
pub mod ir_passes;
pub mod jit_debug;
pub mod jit_dwarf;
pub mod jit_f64;
pub mod linearize;
pub mod lockstep;
pub mod malum;
pub mod pipeline_opts;
mod pow10tab;
pub mod regalloc_engine;
pub mod solver;

pub use compiler::{
    CompiledDecoder, CompiledFunction, PipelineArtifacts, compile_hir_module, compile_pipeline,
    compile_pipeline_from_hir_module, compile_pre_opt_cfg, lower_hir_module,
};
use context::{DeserContext, ErrorCode};
pub use format::DecoderKind;
pub use pipeline_opts::{CompileTarget, PipelineOptions};

/// Compile a deserializer for the given shape and format.
pub fn compile_decoder(shape: &'static facet::Shape, kind: DecoderKind) -> CompiledDecoder {
    compiler::compile_decoder(shape, kind)
}

/// Compile a deserializer with explicit pipeline options.
pub fn compile_decoder_with_options(
    shape: &'static facet::Shape,
    kind: DecoderKind,
    pipeline_opts: &PipelineOptions,
) -> CompiledDecoder {
    compiler::compile_decoder_with_options(shape, kind, pipeline_opts)
}

/// Return a deterministic machine-emission trace annotated with CFG-MIR provenance.
pub fn emission_trace_text(shape: &'static facet::Shape, kind: DecoderKind) -> String {
    compiler::emission_trace_text(shape, kind)
}

/// Return a deterministic emission trace with explicit pipeline options.
pub fn emission_trace_text_with_options(
    shape: &'static facet::Shape,
    kind: DecoderKind,
    pipeline_opts: &PipelineOptions,
) -> String {
    compiler::emission_trace_text_with_options(shape, kind, pipeline_opts)
}

/// Return ARM64 assembly text (captured instructions before encoding).
#[cfg(target_arch = "aarch64")]
pub fn assembly_text(shape: &'static facet::Shape, kind: DecoderKind) -> String {
    compiler::assembly_text(shape, kind)
}

/// Compile a deserializer from already-linearized IR.
pub fn compile_decoder_linear_ir(
    ir: &linearize::LinearIr,
    trusted_utf8_input: bool,
) -> CompiledDecoder {
    compiler::compile_linear_ir_decoder(ir, trusted_utf8_input)
}

/// Compile a deserializer from IR text (RVSDG representation).
///
/// Parses the IR text, runs the full pipeline (IR → passes → LIR → CFG-MIR →
/// regalloc → codegen), and returns an executable decoder.
///
/// This is intended for regression tests and bug minimization: paste a failing
/// IR snapshot, edit it down to a minimal reproducer, and re-run.
pub fn compile_decoder_from_ir_text(
    ir_text: &str,
    registry: &ir::IntrinsicRegistry,
    with_passes: bool,
) -> CompiledDecoder {
    let mut func = ir_parse::parse_ir(ir_text, registry).expect("IR text should parse");
    if with_passes {
        compiler::run_default_passes_from_env(&mut func);
    }
    let linear = linearize::linearize(&mut func);
    compiler::compile_linear_ir_decoder(&linear, false)
}

/// Compile a deserializer from canonical CFG-MIR text.
///
/// Parses the CFG-MIR text with the provided symbol registry, runs regalloc,
/// and returns an executable decoder.
pub fn compile_decoder_from_cfg_mir_text(
    cfg_mir_text: &str,
    registry: &ir::IntrinsicRegistry,
    trusted_utf8_input: bool,
) -> CompiledDecoder {
    let cfg_program = kajit_mir_text::parse_cfg_mir_with_registry(cfg_mir_text, registry)
        .expect("CFG-MIR text should parse");
    compiler::compile_cfg_mir_decoder_with_registry(
        &cfg_program,
        Some(registry),
        trusted_utf8_input,
    )
}

/// Compile from IR text and immediately deserialize one input.
pub fn deserialize_from_ir_text<'input, T: facet::Facet<'input>>(
    ir_text: &str,
    registry: &ir::IntrinsicRegistry,
    with_passes: bool,
    input: &'input [u8],
) -> Result<T, DeserError> {
    let decoder = compile_decoder_from_ir_text(ir_text, registry, with_passes);
    deserialize(&decoder, input)
}

/// Compile from CFG-MIR text and immediately deserialize one input.
pub fn deserialize_from_cfg_mir_text<'input, T: facet::Facet<'input>>(
    cfg_mir_text: &str,
    input: &'input [u8],
) -> Result<T, DeserError> {
    let registry = symbol_registry_for_shape(T::SHAPE);
    let decoder = compile_decoder_from_cfg_mir_text(cfg_mir_text, &registry, false);
    deserialize(&decoder, input)
}

/// Build decoder IR (after default pre-regalloc passes) and return textual RVSDG + CFG-MIR dumps.
///
/// Intended for snapshot tests and debugging.
pub fn debug_ir_and_cfg_mir_text(
    shape: &'static facet::Shape,
    kind: DecoderKind,
) -> (String, String) {
    let (module, _symbol_table) = compiler::build_decoder_hir(shape, kind);
    let mut func = compiler::lower_hir_module(&module);
    compiler::run_default_passes_from_env(&mut func);
    let registry = symbol_registry_for_shape(shape);
    let ir_text = format!("{}", func.display_with_registry(&registry));
    let cfg = debug_cfg_mir(shape, kind);
    let cfg_text = format!("{}", cfg.display_with_registry(&registry));
    (ir_text, cfg_text)
}

/// Build decoder IR (after default pre-regalloc passes) and return a canonical CFG-MIR dump.
///
/// This renderer is intended for interactive debugging and LLM-assisted analysis.
pub fn debug_cfg_mir_text(shape: &'static facet::Shape, kind: DecoderKind) -> String {
    let cfg = debug_cfg_mir(shape, kind);
    let registry = symbol_registry_for_shape(shape);
    format!("{}", cfg.display_with_registry(&registry))
}

/// Build decoder IR (after default pre-regalloc passes) and return the CFG-MIR program.
///
/// This preserves real intrinsic function pointers and is intended for
/// in-process debugging tools (e.g. differential checking).
pub fn debug_cfg_mir(
    shape: &'static facet::Shape,
    kind: DecoderKind,
) -> regalloc_engine::cfg_mir::Program {
    let (module, _symbol_table) = compiler::build_decoder_hir(shape, kind);
    let mut func = compiler::lower_hir_module(&module);
    compiler::run_default_passes_from_env(&mut func);
    let linear = linearize::linearize(&mut func);
    let hints = Default::default(); // TODO: Call analyze_spill_costs(&func) before linearization
    regalloc_engine::cfg_mir::lower_and_optimize(&linear, hints)
}

/// Build the current prototype postcard HIR for a shape.
///
/// This is an inspectable prototype producer path from facet `Shape` into the
/// new semantic HIR. It currently targets postcard semantics only and supports
/// a deliberately narrow subset needed to exercise borrowed-output decoding.
pub fn debug_postcard_hir(shape: &'static facet::Shape) -> kajit_hir::Module {
    let (module, _symbol_table) = compiler::build_postcard_decoder_hir(shape);
    module
}

/// Build the current prototype postcard HIR and render it in canonical text form.
pub fn debug_postcard_hir_text(shape: &'static facet::Shape) -> String {
    debug_postcard_hir(shape).to_string()
}

/// Build HIR for a decoder and render it in canonical text form.
///
/// Dispatches to the appropriate HIR builder based on the decoder kind.
pub fn debug_hir_text(shape: &'static facet::Shape, kind: DecoderKind) -> String {
    let (module, _symbol_table) = compiler::build_decoder_hir(shape, kind);
    module.to_string()
}

/// Build postcard RVSDG by building postcard HIR and lowering it to IR.
pub fn debug_postcard_ir_via_hir_text(shape: &'static facet::Shape) -> String {
    let (module, _symbol_table) = compiler::build_postcard_decoder_hir(shape);
    let func = compiler::lower_hir_module(&module);
    let registry = symbol_registry_for_shape(shape);
    format!("{}", func.display_with_registry(&registry))
}

/// Compile a postcard decoder by building postcard HIR and lowering it.
pub fn compile_postcard_decoder_via_hir(shape: &'static facet::Shape) -> CompiledDecoder {
    compiler::compile_postcard_decoder_via_hir_with_options(shape, PipelineOptions::from_env())
}

/// Compile a postcard decoder through the postcard-HIR pipeline.
///
/// This is retained as a compatibility alias for callers that still use the
/// older name.
pub fn compile_postcard_decoder_via_structural_hir(
    shape: &'static facet::Shape,
) -> CompiledDecoder {
    compiler::compile_postcard_decoder_via_hir_with_options(shape, PipelineOptions::from_env())
}

/// Build decoder IR (after default pre-regalloc passes) and return the Linear IR.
pub fn debug_linear_ir(shape: &'static facet::Shape, kind: DecoderKind) -> linearize::LinearIr {
    let (module, _symbol_table) = compiler::build_decoder_hir(shape, kind);
    let mut func = compiler::lower_hir_module(&module);
    compiler::run_default_passes_from_env(&mut func);
    linearize::linearize(&mut func)
}

/// Build decoder IR (after default pre-regalloc passes) and return textual Linear IR dump.
pub fn debug_linear_ir_text(shape: &'static facet::Shape, kind: DecoderKind) -> String {
    let linear = debug_linear_ir(shape, kind);
    let registry = symbol_registry_for_shape(shape);
    format!("{}", linear.display_with_registry(&registry))
}

/// Build decoder IR and return textual RVSDG checkpoints before/after each enabled optimization pass.
///
/// The first entry is always `("initial", ir_before_passes)`.
pub fn debug_ir_opt_timeline_text(
    shape: &'static facet::Shape,
    kind: DecoderKind,
) -> Vec<(String, String)> {
    let pipeline_opts = PipelineOptions::from_env();
    debug_ir_opt_timeline_text_with_options(shape, kind, &pipeline_opts)
}

/// Same as [`debug_ir_opt_timeline_text`], but with explicit pipeline options.
pub fn debug_ir_opt_timeline_text_with_options(
    shape: &'static facet::Shape,
    kind: DecoderKind,
    pipeline_opts: &PipelineOptions,
) -> Vec<(String, String)> {
    let (module, _symbol_table) = compiler::build_decoder_hir(shape, kind);
    let mut func = compiler::lower_hir_module(&module);
    let registry = symbol_registry_for_shape(shape);
    let mut checkpoints = vec![(
        "initial".to_string(),
        format!("{}", func.display_with_registry(&registry)),
    )];

    compiler::run_configured_default_passes_with_observer(
        &mut func,
        pipeline_opts,
        |pass, func| {
            checkpoints.push((
                pass.to_string(),
                format!("{}", func.display_with_registry(&registry)),
            ));
        },
    );

    checkpoints
}

/// Build a registry containing all built-in postcard and JSON intrinsics.
pub fn known_intrinsic_registry() -> ir::IntrinsicRegistry {
    let mut registry = ir::IntrinsicRegistry::empty();
    for (name, func) in intrinsics::known_intrinsics() {
        registry.register(name, func);
    }
    registry
}

/// Build a registry containing built-in intrinsics plus shape-specific named constants.
pub fn symbol_registry_for_shape(shape: &'static facet::Shape) -> ir::IntrinsicRegistry {
    compiler::symbol_registry_for_shape(shape)
}

/// Render the shape-specific named constant table used by text debug helpers.
pub fn debug_const_registry_text(shape: &'static facet::Shape) -> String {
    let mut entries = symbol_registry_for_shape(shape)
        .const_entries()
        .map(|(name, value)| format!("const @{name} = {value:#x}"))
        .collect::<Vec<_>>();
    entries.sort();
    let mut text = entries.join("\n");
    if !text.is_empty() {
        text.push('\n');
    }
    text
}

// r[impl api.output]
/// Deserialize a value of type `T` from the given input bytes using a compiled deserializer.
///
/// # Safety
/// The compiled deserializer must have been compiled for the same shape as `T`.
pub fn deserialize<'input, T: facet::Facet<'input>>(
    deser: &CompiledDecoder,
    input: &'input [u8],
) -> Result<T, DeserError> {
    from_bytes(deser, input)
}

/// Deserialize a value of type `T` from raw bytes.
pub fn from_bytes<'input, T: facet::Facet<'input>>(
    deser: &CompiledDecoder,
    input: &'input [u8],
) -> Result<T, DeserError> {
    let mut ctx = DeserContext::from_bytes(input);
    deserialize_with_ctx(deser, &mut ctx)
}

/// Deserialize a value of type `T` from UTF-8 input text.
pub fn from_str<'input, T: facet::Facet<'input>>(
    deser: &CompiledDecoder,
    input: &'input str,
) -> Result<T, DeserError> {
    let _ = deser;
    let mut ctx = DeserContext::from_bytes(input.as_bytes());
    deserialize_with_ctx(deser, &mut ctx)
}

/// Deserialize into a raw output buffer and return its bytes.
///
/// This is intended for CFG-MIR differential testing and minimization workflows
/// where only output memory bytes and error slots matter.
pub fn deserialize_raw(
    deser: &CompiledDecoder,
    input: &[u8],
    output_size: usize,
) -> Result<Vec<u8>, DeserError> {
    let mut ctx = DeserContext::from_bytes(input);
    let mut output = vec![0u8; output_size];

    invoke_decoder(deser, output.as_mut_ptr(), &mut ctx);

    if ctx.error.code != 0 {
        let code: ErrorCode = unsafe { core::mem::transmute(ctx.error.code) };
        return Err(DeserError {
            code,
            offset: ctx.error.offset,
        });
    }

    Ok(output)
}

fn deserialize_with_ctx<'input, T: facet::Facet<'input>>(
    deser: &CompiledDecoder,
    ctx: &mut DeserContext,
) -> Result<T, DeserError> {
    // Allocate output on the stack as MaybeUninit
    let mut output = core::mem::MaybeUninit::<T>::uninit();

    invoke_decoder(deser, output.as_mut_ptr() as *mut u8, ctx);

    if ctx.error.code != 0 {
        let code: ErrorCode = unsafe { core::mem::transmute(ctx.error.code) };
        return Err(DeserError {
            code,
            offset: ctx.error.offset,
        });
    }

    Ok(unsafe { output.assume_init() })
}

fn invoke_decoder(deser: &CompiledDecoder, output: *mut u8, ctx: &mut DeserContext) {
    unsafe {
        (deser.func())(output, ctx);
    }
}

/// Error returned by `deserialize`.
#[derive(Debug)]
pub struct DeserError {
    pub code: ErrorCode,
    pub offset: u32,
}

impl core::fmt::Display for DeserError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "{} at offset {}", self.code, self.offset)
    }
}

impl std::error::Error for DeserError {}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DifferentialFailure {
    pub code: ErrorCode,
    pub offset: u32,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DifferentialOutcome {
    Success { output: Vec<u8> },
    Failure(DifferentialFailure),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DifferentialMismatch {
    FirstDivergentByte {
        index: usize,
        interpreter: u8,
        jit: u8,
    },
    OutputLength {
        shared_prefix: usize,
        interpreter_len: usize,
        jit_len: usize,
    },
    FailureMismatch {
        interpreter: DifferentialFailure,
        jit: DifferentialFailure,
    },
    OutcomeKindMismatch {
        interpreter: DifferentialOutcome,
        jit: DifferentialOutcome,
    },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DifferentialReport {
    pub interpreter: DifferentialOutcome,
    pub jit: DifferentialOutcome,
    pub mismatch: Option<DifferentialMismatch>,
}

impl DifferentialReport {
    pub fn is_match(&self) -> bool {
        self.mismatch.is_none()
    }
}

#[derive(Debug)]
pub enum DifferentialHarnessError {
    InterpreterTimeout,
}

impl core::fmt::Display for DifferentialHarnessError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::InterpreterTimeout => write!(f, "ideal interpreter timed out"),
        }
    }
}

impl std::error::Error for DifferentialHarnessError {}

fn normalize_interp_outcome(
    outcome: kajit_mir::opt::reduce::InterpOutcome,
    output_size: usize,
) -> Result<DifferentialOutcome, DifferentialHarnessError> {
    use kajit_mir::opt::reduce::InterpOutcome;
    match outcome {
        InterpOutcome::Returned(mut output) => {
            if output.len() < output_size {
                output.resize(output_size, 0);
            } else if output.len() > output_size {
                output.truncate(output_size);
            }
            Ok(DifferentialOutcome::Success { output })
        }
        InterpOutcome::Trapped(trap) => Ok(DifferentialOutcome::Failure(DifferentialFailure {
            code: trap.code,
            offset: trap.offset,
        })),
        InterpOutcome::TimedOut => Err(DifferentialHarnessError::InterpreterTimeout),
    }
}

pub fn infer_linear_ir_output_size(ir: &linearize::LinearIr) -> usize {
    let from_fields = ir
        .ops
        .iter()
        .filter_map(|op| match op {
            linearize::LinearOp::StoreToAddr { .. } | linearize::LinearOp::LoadFromAddr { .. } => {
                // General-purpose IR no longer carries static offsets;
                // output size is determined by FuncStart.
                None
            }
            _ => None,
        })
        .max()
        .unwrap_or(0);
    let from_func_start = ir
        .ops
        .iter()
        .find_map(|op| match op {
            linearize::LinearOp::FuncStart { output_size, .. } => Some(*output_size),
            _ => None,
        })
        .unwrap_or(0);
    from_fields.max(from_func_start)
}

fn normalize_jit_outcome(
    outcome: Result<Vec<u8>, DeserError>,
    output_size: usize,
) -> DifferentialOutcome {
    match outcome {
        Ok(mut output) => {
            if output.len() < output_size {
                output.resize(output_size, 0);
            } else if output.len() > output_size {
                output.truncate(output_size);
            }
            DifferentialOutcome::Success { output }
        }
        Err(err) => DifferentialOutcome::Failure(DifferentialFailure {
            code: err.code,
            offset: err.offset,
        }),
    }
}

fn compare_differential_outcomes(
    interpreter: &DifferentialOutcome,
    jit: &DifferentialOutcome,
) -> Option<DifferentialMismatch> {
    compare_differential_outcomes_with_ignored_output_bytes(interpreter, jit, None)
}

pub(crate) fn compare_differential_outcomes_with_ignored_output_bytes(
    interpreter: &DifferentialOutcome,
    jit: &DifferentialOutcome,
    ignored_output_bytes: Option<&[bool]>,
) -> Option<DifferentialMismatch> {
    match (interpreter, jit) {
        (
            DifferentialOutcome::Success {
                output: interpreter,
            },
            DifferentialOutcome::Success { output: jit },
        ) => {
            let shared = interpreter.len().min(jit.len());
            for i in 0..shared {
                if ignored_output_bytes
                    .and_then(|mask| mask.get(i))
                    .copied()
                    .unwrap_or(false)
                {
                    continue;
                }
                if interpreter[i] != jit[i] {
                    return Some(DifferentialMismatch::FirstDivergentByte {
                        index: i,
                        interpreter: interpreter[i],
                        jit: jit[i],
                    });
                }
            }
            if interpreter.len() != jit.len() {
                return Some(DifferentialMismatch::OutputLength {
                    shared_prefix: shared,
                    interpreter_len: interpreter.len(),
                    jit_len: jit.len(),
                });
            }
            None
        }
        (DifferentialOutcome::Failure(interpreter), DifferentialOutcome::Failure(jit)) => {
            if interpreter == jit {
                None
            } else {
                Some(DifferentialMismatch::FailureMismatch {
                    interpreter: *interpreter,
                    jit: *jit,
                })
            }
        }
        _ => Some(DifferentialMismatch::OutcomeKindMismatch {
            interpreter: interpreter.clone(),
            jit: jit.clone(),
        }),
    }
}

pub fn differential_check_linear_ir_vs_jit(
    ir: &linearize::LinearIr,
    input: &[u8],
) -> Result<DifferentialReport, DifferentialHarnessError> {
    let output_size = infer_linear_ir_output_size(ir);
    differential_check_linear_ir_vs_jit_with_output_size(ir, input, output_size)
}

pub fn differential_check_linear_ir_vs_jit_with_output_size(
    ir: &linearize::LinearIr,
    input: &[u8],
    output_size: usize,
) -> Result<DifferentialReport, DifferentialHarnessError> {
    let hints = Default::default();
    let cfg_program = regalloc_engine::cfg_mir::lower_and_optimize(ir, hints);
    let interp_outcome = kajit_mir::opt::reduce::interpret(&cfg_program, input);
    let interpreter = normalize_interp_outcome(interp_outcome, output_size)?;
    let decoder = compile_decoder_linear_ir(ir, false);
    let jit = normalize_jit_outcome(deserialize_raw(&decoder, input, output_size), output_size);
    let mismatch = compare_differential_outcomes(&interpreter, &jit);
    Ok(DifferentialReport {
        interpreter,
        jit,
        mismatch,
    })
}

#[cfg(test)]
mod differential_tests {
    use super::*;

    #[test]
    fn differential_harness_reports_first_divergent_byte() {
        let interpreter = DifferentialOutcome::Success {
            output: vec![1, 2, 3],
        };
        let jit = DifferentialOutcome::Success {
            output: vec![1, 9, 3],
        };
        let mismatch = compare_differential_outcomes(&interpreter, &jit);
        assert_eq!(
            mismatch,
            Some(DifferentialMismatch::FirstDivergentByte {
                index: 1,
                interpreter: 2,
                jit: 9,
            })
        );
    }
}
