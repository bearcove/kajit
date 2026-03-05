#![allow(clippy::approx_constant)]

pub mod arch;
pub mod backends;
pub mod compiler;
pub mod context;
pub mod disasm_normalize;
pub mod format;
pub mod intrinsics;
pub mod ir;
pub mod ir_backend;

pub mod ir_parse;
pub mod ir_passes;
pub mod jit_debug;
pub mod jit_dwarf;
pub mod jit_f64;
pub mod json;
pub mod json_intrinsics;
pub mod linearize;
pub mod malum;
pub mod pipeline_opts;
pub mod postcard;
mod pow10tab;
pub mod recipe;
pub mod regalloc_engine;
pub mod solver;

use compiler::CompiledDecoder;
use context::{DeserContext, ErrorCode};
pub use pipeline_opts::PipelineOptions;

/// Compile a deserializer for the given shape and format.
pub fn compile_decoder(
    shape: &'static facet::Shape,
    decoder: &dyn format::Decoder,
) -> CompiledDecoder {
    compiler::compile_decoder(shape, decoder)
}

/// Compile a deserializer with explicit pipeline options.
pub fn compile_decoder_with_options(
    shape: &'static facet::Shape,
    decoder: &dyn format::Decoder,
    pipeline_opts: &PipelineOptions,
) -> CompiledDecoder {
    compiler::compile_decoder_with_options(shape, decoder, pipeline_opts)
}

/// Return the number of regalloc edit instructions produced by IR lowering.
pub fn regalloc_edit_count<F: format::Decoder>(shape: &'static facet::Shape, decoder: &F) -> usize {
    compiler::regalloc_edit_count(shape, decoder)
}

/// Return the number of regalloc edit instructions produced by IR lowering with explicit options.
pub fn regalloc_edit_count_with_options<F: format::Decoder>(
    shape: &'static facet::Shape,
    decoder: &F,
    pipeline_opts: &PipelineOptions,
) -> usize {
    compiler::regalloc_edit_count_with_options(shape, decoder, pipeline_opts)
}

/// Return a detailed regalloc edits dump for the compiled decoder pipeline.
pub fn regalloc_edits_text<F: format::Decoder>(
    shape: &'static facet::Shape,
    decoder: &F,
) -> String {
    compiler::regalloc_edits_text(shape, decoder)
}

/// Return a detailed regalloc edits dump with explicit pipeline options.
pub fn regalloc_edits_text_with_options<F: format::Decoder>(
    shape: &'static facet::Shape,
    decoder: &F,
    pipeline_opts: &PipelineOptions,
) -> String {
    compiler::regalloc_edits_text_with_options(shape, decoder, pipeline_opts)
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
    shape: &'static facet::Shape,
    registry: &ir::IntrinsicRegistry,
    with_passes: bool,
) -> CompiledDecoder {
    let mut func = ir_parse::parse_ir(ir_text, shape, registry).expect("IR text should parse");
    if with_passes {
        compiler::run_default_passes_from_env(&mut func);
    }
    let linear = linearize::linearize(&mut func);
    compiler::compile_linear_ir_decoder(&linear, false)
}

/// Compile a deserializer from canonical CFG-MIR text.
///
/// Parses the CFG-MIR text with Kajit's built-in intrinsic registry, runs
/// regalloc (unless `KAJIT_OPTS='-regalloc'`), and returns an executable
/// decoder.
pub fn compile_decoder_from_cfg_mir_text(
    cfg_mir_text: &str,
    trusted_utf8_input: bool,
) -> CompiledDecoder {
    let registry = known_intrinsic_registry();
    let cfg_program = kajit_mir_text::parse_cfg_mir_with_registry(cfg_mir_text, &registry)
        .expect("CFG-MIR text should parse");
    compiler::compile_cfg_mir_decoder(&cfg_program, trusted_utf8_input)
}

/// Compile from IR text and immediately deserialize one input.
pub fn deserialize_from_ir_text<'input, T: facet::Facet<'input>>(
    ir_text: &str,
    shape: &'static facet::Shape,
    registry: &ir::IntrinsicRegistry,
    with_passes: bool,
    input: &'input [u8],
) -> Result<T, DeserError> {
    let decoder = compile_decoder_from_ir_text(ir_text, shape, registry, with_passes);
    deserialize(&decoder, input)
}

/// Compile from CFG-MIR text and immediately deserialize one input.
pub fn deserialize_from_cfg_mir_text<'input, T: facet::Facet<'input>>(
    cfg_mir_text: &str,
    input: &'input [u8],
) -> Result<T, DeserError> {
    let decoder = compile_decoder_from_cfg_mir_text(cfg_mir_text, false);
    deserialize(&decoder, input)
}

/// Build decoder IR (after default pre-regalloc passes) and return textual RVSDG + CFG-MIR dumps.
///
/// Intended for snapshot tests and debugging.
pub fn debug_ir_and_cfg_mir_text(
    shape: &'static facet::Shape,
    ir_decoder: &dyn format::Decoder,
) -> (String, String) {
    let mut func = compiler::build_decoder_ir(shape, ir_decoder);
    compiler::run_default_passes_from_env(&mut func);
    let registry = known_intrinsic_registry();
    let ir_text = scrub_volatile_const_addrs(&format!("{}", func.display_with_registry(&registry)));
    let cfg = debug_cfg_mir(shape, ir_decoder);
    let cfg_text = scrub_volatile_const_addrs(&format!("{}", cfg.display_with_registry(&registry)));
    (ir_text, cfg_text)
}

/// Build decoder IR (after default pre-regalloc passes) and return a canonical CFG-MIR dump.
///
/// This renderer is intended for interactive debugging and LLM-assisted analysis.
pub fn debug_cfg_mir_text(
    shape: &'static facet::Shape,
    ir_decoder: &dyn format::Decoder,
) -> String {
    let cfg = debug_cfg_mir(shape, ir_decoder);
    let registry = known_intrinsic_registry();
    scrub_volatile_const_addrs(&format!("{}", cfg.display_with_registry(&registry)))
}

/// Build decoder IR (after default pre-regalloc passes) and return the CFG-MIR program.
///
/// This preserves real intrinsic function pointers and is intended for
/// in-process debugging tools (e.g. differential checking).
pub fn debug_cfg_mir(
    shape: &'static facet::Shape,
    ir_decoder: &dyn format::Decoder,
) -> regalloc_engine::cfg_mir::Program {
    let mut func = compiler::build_decoder_ir(shape, ir_decoder);
    compiler::run_default_passes_from_env(&mut func);
    let linear = linearize::linearize(&mut func);
    regalloc_engine::cfg_mir::lower_linear_ir(&linear)
}

/// Build decoder IR (after default pre-regalloc passes) and return textual Linear IR dump.
///
/// Intended for snapshot tests and debugging.
pub fn debug_linear_ir(
    shape: &'static facet::Shape,
    ir_decoder: &dyn format::Decoder,
) -> linearize::LinearIr {
    let mut func = compiler::build_decoder_ir(shape, ir_decoder);
    compiler::run_default_passes_from_env(&mut func);
    linearize::linearize(&mut func)
}

/// Build decoder IR (after default pre-regalloc passes) and return textual Linear IR dump.
///
/// Intended for snapshot tests and debugging.
pub fn debug_linear_ir_text(
    shape: &'static facet::Shape,
    ir_decoder: &dyn format::Decoder,
) -> String {
    let linear = debug_linear_ir(shape, ir_decoder);
    let registry = known_intrinsic_registry();
    scrub_volatile_const_addrs(&format!("{}", linear.display_with_registry(&registry)))
}

/// Build decoder IR and return textual RVSDG checkpoints before/after each enabled optimization pass.
///
/// The first entry is always `("initial", ir_before_passes)`.
pub fn debug_ir_opt_timeline_text(
    shape: &'static facet::Shape,
    ir_decoder: &dyn format::Decoder,
) -> Vec<(String, String)> {
    let pipeline_opts = PipelineOptions::from_env();
    debug_ir_opt_timeline_text_with_options(shape, ir_decoder, &pipeline_opts)
}

/// Same as [`debug_ir_opt_timeline_text`], but with explicit pipeline options.
pub fn debug_ir_opt_timeline_text_with_options(
    shape: &'static facet::Shape,
    ir_decoder: &dyn format::Decoder,
    pipeline_opts: &PipelineOptions,
) -> Vec<(String, String)> {
    let mut func = compiler::build_decoder_ir(shape, ir_decoder);
    let registry = known_intrinsic_registry();
    let mut checkpoints = vec![(
        "initial".to_string(),
        scrub_volatile_const_addrs(&format!("{}", func.display_with_registry(&registry))),
    )];

    compiler::run_configured_default_passes_with_observer(
        &mut func,
        pipeline_opts,
        |pass, func| {
            checkpoints.push((
                pass.to_string(),
                scrub_volatile_const_addrs(&format!("{}", func.display_with_registry(&registry))),
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
    for (name, func) in json_intrinsics::known_intrinsics() {
        registry.register(name, func);
    }
    registry
}

fn scrub_volatile_const_addrs(text: &str) -> String {
    // Pointer-valued const operands are still process-local and not yet
    // symbolically rendered. Scrub only those constant payloads.
    let text = scrub_hex_after_prefix_with_min_len(&text, "Const(0x", 9);
    scrub_hex_after_prefix_with_min_len(&text, "const(0x", 9)
}

fn scrub_hex_after_prefix_with_min_len(input: &str, prefix: &str, min_hex_len: usize) -> String {
    let mut out = String::with_capacity(input.len());
    let bytes = input.as_bytes();
    let prefix_bytes = prefix.as_bytes();
    let mut i = 0;

    while i < bytes.len() {
        if i + prefix_bytes.len() <= bytes.len()
            && &bytes[i..i + prefix_bytes.len()] == prefix_bytes
        {
            out.push_str(prefix);
            i += prefix_bytes.len();

            let start = i;
            while i < bytes.len() && bytes[i].is_ascii_hexdigit() {
                i += 1;
            }

            if i - start >= min_hex_len {
                out.push_str("<ptr>");
            } else {
                out.push_str(&input[start..i]);
            }
            continue;
        }

        out.push(bytes[i] as char);
        i += 1;
    }

    out
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
///
/// Trusted UTF-8 mode is enabled only when the compiled format supports it.
pub fn from_str<'input, T: facet::Facet<'input>>(
    deser: &CompiledDecoder,
    input: &'input str,
) -> Result<T, DeserError> {
    let mut ctx = if deser.supports_trusted_utf8_input() {
        DeserContext::from_str(input)
    } else {
        DeserContext::from_bytes(input.as_bytes())
    };
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

    unsafe {
        (deser.func())(output.as_mut_ptr(), &mut ctx);
    }

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

    unsafe {
        (deser.func())(output.as_mut_ptr() as *mut u8, ctx);
    }

    if ctx.error.code != 0 {
        let code: ErrorCode = unsafe { core::mem::transmute(ctx.error.code) };
        return Err(DeserError {
            code,
            offset: ctx.error.offset,
        });
    }

    Ok(unsafe { output.assume_init() })
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
    Simulation(kajit_mir::RegallocEngineError),
}

impl core::fmt::Display for DifferentialHarnessError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::Simulation(err) => write!(f, "cfg simulation failed: {err}"),
        }
    }
}

impl std::error::Error for DifferentialHarnessError {}

impl From<kajit_mir::RegallocEngineError> for DifferentialHarnessError {
    fn from(value: kajit_mir::RegallocEngineError) -> Self {
        Self::Simulation(value)
    }
}

pub fn infer_linear_ir_output_size(ir: &linearize::LinearIr) -> usize {
    ir.ops
        .iter()
        .filter_map(|op| match op {
            linearize::LinearOp::WriteToField { offset, width, .. }
            | linearize::LinearOp::ReadFromField { offset, width, .. } => {
                Some(*offset as usize + width.bytes() as usize)
            }
            _ => None,
        })
        .max()
        .unwrap_or(0)
}

fn normalize_simulation_outcome(
    outcome: kajit_mir::ExecutionResult,
    output_size: usize,
) -> DifferentialOutcome {
    match outcome.trap {
        Some(trap) => DifferentialOutcome::Failure(DifferentialFailure {
            code: trap.code,
            offset: trap.offset,
        }),
        None => {
            let mut output = outcome.output;
            if output.len() < output_size {
                output.resize(output_size, 0);
            } else if output.len() > output_size {
                output.truncate(output_size);
            }
            DifferentialOutcome::Success { output }
        }
    }
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
    match (interpreter, jit) {
        (
            DifferentialOutcome::Success {
                output: interpreter,
            },
            DifferentialOutcome::Success { output: jit },
        ) => {
            let shared = interpreter.len().min(jit.len());
            for i in 0..shared {
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
    let cfg_program = regalloc_engine::cfg_mir::lower_linear_ir(ir);
    let alloc = regalloc_engine::allocate_cfg_program(&cfg_program)?;
    let simulation = regalloc_engine::simulate_execution_cfg(&alloc, input)?;
    let interpreter = normalize_simulation_outcome(simulation, output_size);
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
    use facet::Facet;

    const POSTCARD_U32_V0_RVSDG_SNAPSHOT: &str = include_str!(
        "../tests/snapshots/corpus__generated_rvsdg_postcard_scalar_u32__v0_x86_64.snap"
    );

    fn snapshot_body(snapshot: &'static str) -> &'static str {
        let snapshot = snapshot
            .strip_prefix("---\n")
            .expect("insta snapshot should start with frontmatter");
        let (_, body) = snapshot
            .split_once("\n---\n")
            .expect("insta snapshot frontmatter should end with separator");
        body.trim()
    }

    #[test]
    fn differential_harness_matches_postcard_u32_linear_ir_snapshot() {
        let ir_text = snapshot_body(POSTCARD_U32_V0_RVSDG_SNAPSHOT);
        let registry = ir::IntrinsicRegistry::new();
        let mut ir_func =
            ir_parse::parse_ir(ir_text, <u32 as Facet>::SHAPE, &registry).expect("valid RVSDG");
        let linear = linearize::linearize(&mut ir_func);
        let report =
            differential_check_linear_ir_vs_jit(&linear, &[0x2a]).expect("harness should execute");
        assert!(
            report.is_match(),
            "unexpected mismatch: {:?}",
            report.mismatch
        );
        assert_eq!(
            report.interpreter,
            DifferentialOutcome::Success {
                output: vec![0x2a, 0x00, 0x00, 0x00]
            }
        );
    }

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
