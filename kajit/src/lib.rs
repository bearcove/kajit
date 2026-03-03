#![allow(clippy::approx_constant)]

pub mod arch;
pub mod compiler;
pub mod context;
pub mod disasm_normalize;
pub mod format;
pub mod intrinsics;
pub mod ir;
pub mod ir_backend;
#[cfg(target_arch = "aarch64")]
pub mod ir_backend_aarch64;
#[cfg(target_arch = "x86_64")]
pub mod ir_backend_x64;
pub mod ir_parse;
pub mod ir_passes;
pub mod jit_debug;
pub mod jit_f64;
pub mod json;
pub mod json_intrinsics;
pub mod linearize;
pub mod malum;
pub mod postcard;
mod pow10tab;
pub mod recipe;
pub mod regalloc_engine;
pub mod regalloc_mir;
pub mod regalloc_mir_parse;
pub mod solver;

use compiler::CompiledDecoder;
use context::{DeserContext, ErrorCode};

/// Compile a deserializer for the given shape and format.
pub fn compile_decoder(
    shape: &'static facet::Shape,
    decoder: &dyn format::Decoder,
) -> CompiledDecoder {
    compiler::compile_decoder(shape, decoder)
}

/// Return the number of regalloc edit instructions produced by IR lowering.
pub fn regalloc_edit_count<F: format::Decoder>(shape: &'static facet::Shape, decoder: &F) -> usize {
    compiler::regalloc_edit_count(shape, decoder)
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
/// Parses the IR text, runs the full pipeline (IR → passes → LIR → RA-MIR →
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
        ir_passes::run_default_passes(&mut func);
    }
    let linear = linearize::linearize(&mut func);
    compiler::compile_linear_ir_decoder(&linear, false)
}

/// Compile a deserializer from an already-constructed RaProgram.
///
/// Runs regalloc2 + codegen, skipping IR/LIR entirely.
pub fn compile_decoder_from_ra_program(program: &regalloc_mir::RaProgram) -> CompiledDecoder {
    compiler::compile_ra_program_decoder(program)
}

/// Compile a deserializer from RA-MIR text.
///
/// Parses the RA-MIR text, runs regalloc2 + codegen, and returns an
/// executable decoder. Skips the IR → LIR pipeline entirely.
///
/// This is intended for regression tests and bug minimization: paste a
/// failing RA-MIR snapshot, edit it down to a minimal reproducer, and re-run.
pub fn compile_decoder_from_ra_mir_text(mir_text: &str) -> CompiledDecoder {
    let program = regalloc_mir_parse::parse_ra_mir(mir_text).expect("RA-MIR text should parse");
    compile_decoder_from_ra_program(&program)
}

/// Build decoder IR (after default pre-regalloc passes) and return textual RVSDG + RA-MIR dumps.
///
/// Intended for snapshot tests and debugging.
pub fn debug_ir_and_ra_mir_text(
    shape: &'static facet::Shape,
    ir_decoder: &dyn format::Decoder,
) -> (String, String) {
    let mut func = compiler::build_decoder_ir(shape, ir_decoder);
    ir_passes::run_default_passes(&mut func);
    let ir_text = scrub_volatile_intrinsic_addrs(&format!("{func}"));
    let linear = linearize::linearize(&mut func);
    let ra = regalloc_mir::lower_linear_ir(&linear);
    let ra_text = scrub_volatile_intrinsic_addrs(&format!("{ra}"));
    (ir_text, ra_text)
}

/// Build decoder IR (after default pre-regalloc passes) and return a human-readable RA-MIR dump.
///
/// This renderer is intended for interactive debugging and LLM-assisted analysis.
pub fn debug_ra_mir_human_text(
    shape: &'static facet::Shape,
    ir_decoder: &dyn format::Decoder,
) -> String {
    let mut func = compiler::build_decoder_ir(shape, ir_decoder);
    ir_passes::run_default_passes(&mut func);
    let linear = linearize::linearize(&mut func);
    let ra = regalloc_mir::lower_linear_ir(&linear);
    scrub_volatile_intrinsic_addrs(&format!("{}", ra.human()))
}

/// Build decoder IR (after default pre-regalloc passes) and return textual Linear IR dump.
///
/// Intended for snapshot tests and debugging.
pub fn debug_linear_ir_text(
    shape: &'static facet::Shape,
    ir_decoder: &dyn format::Decoder,
) -> String {
    let mut func = compiler::build_decoder_ir(shape, ir_decoder);
    ir_passes::run_default_passes(&mut func);
    let linear = linearize::linearize(&mut func);
    scrub_volatile_intrinsic_addrs(&format!("{linear}"))
}

fn scrub_volatile_intrinsic_addrs(text: &str) -> String {
    // Display dumps include intrinsic function pointer addresses which are process-local and
    // unstable across runs. Replace only those pointer fields to keep snapshots deterministic.
    let text = scrub_hex_after_prefix(text, "CallIntrinsic(0x");
    let text = scrub_hex_after_prefix(&text, "call_intrinsic(0x");
    let text = scrub_hex_after_prefix(&text, "CallPure(0x");
    let text = scrub_hex_after_prefix(&text, "call_pure(0x");
    let text = scrub_hex_after_prefix_with_min_len(&text, "Const(0x", 9);
    scrub_hex_after_prefix_with_min_len(&text, "const(0x", 9)
}

fn scrub_hex_after_prefix(input: &str, prefix: &str) -> String {
    scrub_hex_after_prefix_with_min_len(input, prefix, 1)
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

#[cfg(all(test, not(target_os = "windows")))]
mod disasm_tests;
