#![allow(clippy::approx_constant)]

pub mod arch;
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

use compiler::{CompiledDecoder, CompiledEncoder};
use context::{DeserContext, ErrorCode};
use std::sync::OnceLock;

// r[impl api.compile]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DecoderBackend {
    Legacy,
    Ir,
}

impl DecoderBackend {
    fn parse_env(s: &str) -> Option<Self> {
        if s.eq_ignore_ascii_case("legacy") {
            Some(Self::Legacy)
        } else if s.eq_ignore_ascii_case("ir") {
            Some(Self::Ir)
        } else {
            None
        }
    }
}

static DECODER_BACKEND: OnceLock<DecoderBackend> = OnceLock::new();

/// Returns the default backend used by [`compile_decoder`].
///
/// Read once from `KAJIT_DECODER_BACKEND` (`legacy` or `ir`), then cached.
/// When unset, defaults to `legacy`.
pub fn decoder_backend() -> DecoderBackend {
    *DECODER_BACKEND.get_or_init(|| match std::env::var("KAJIT_DECODER_BACKEND") {
        Ok(value) => DecoderBackend::parse_env(&value).unwrap_or_else(|| {
            panic!("invalid KAJIT_DECODER_BACKEND={value:?}; expected \"legacy\" or \"ir\"")
        }),
        Err(std::env::VarError::NotPresent) => DecoderBackend::Legacy,
        Err(std::env::VarError::NotUnicode(_)) => {
            panic!("KAJIT_DECODER_BACKEND must be valid UTF-8")
        }
    })
}

/// Compile a deserializer for the given shape and format.
pub fn compile_decoder(
    shape: &'static facet::Shape,
    decoder: &dyn format::Decoder,
) -> CompiledDecoder {
    compile_decoder_with_backend(shape, decoder, decoder_backend())
}

/// Compile a deserializer with an explicit backend choice.
pub fn compile_decoder_with_backend(
    shape: &'static facet::Shape,
    decoder: &dyn format::Decoder,
    backend: DecoderBackend,
) -> CompiledDecoder {
    match backend {
        DecoderBackend::Legacy => compile_decoder_legacy(shape, decoder),
        DecoderBackend::Ir => {
            let ir_decoder = decoder.as_ir_decoder().unwrap_or_else(|| {
                panic!(
                    "IR backend requested for format {:?}, but it does not implement IrDecoder",
                    core::any::type_name_of_val(decoder)
                )
            });
            compiler::compile_decoder_via_ir_dyn(shape, decoder, ir_decoder)
        }
    }
}

/// Compile a deserializer using the legacy direct dynasm emission path.
pub fn compile_decoder_legacy(
    shape: &'static facet::Shape,
    decoder: &dyn format::Decoder,
) -> CompiledDecoder {
    compiler::compile_decoder(shape, decoder)
}

/// Compile a deserializer using IR lowering + linearization + backend adapter.
pub fn compile_decoder_via_ir<F: format::Decoder + format::IrDecoder>(
    shape: &'static facet::Shape,
    decoder: &F,
) -> CompiledDecoder {
    compiler::compile_decoder_via_ir(shape, decoder)
}

/// Return the number of regalloc edit instructions produced by IR lowering.
pub fn regalloc_edit_count_via_ir<F: format::IrDecoder>(
    shape: &'static facet::Shape,
    decoder: &F,
) -> usize {
    compiler::regalloc_edit_count_via_ir(shape, decoder)
}

/// Compile a deserializer from already-linearized IR.
pub fn compile_decoder_linear_ir(
    ir: &linearize::LinearIr,
    trusted_utf8_input: bool,
) -> CompiledDecoder {
    compiler::compile_linear_ir_decoder(ir, trusted_utf8_input)
}

/// Build decoder IR (after default pre-regalloc passes) and return textual RVSDG + RA-MIR dumps.
///
/// Intended for snapshot tests and debugging.
pub fn debug_ir_and_ra_mir_text(
    shape: &'static facet::Shape,
    ir_decoder: &dyn format::IrDecoder,
) -> (String, String) {
    let mut func = compiler::build_decoder_ir(shape, ir_decoder);
    ir_passes::run_default_passes(&mut func);
    let ir_text = scrub_volatile_intrinsic_addrs(&format!("{func}"));
    let linear = linearize::linearize(&mut func);
    let ra = regalloc_mir::lower_linear_ir(&linear);
    let ra_text = scrub_volatile_intrinsic_addrs(&format!("{ra}"));
    (ir_text, ra_text)
}

/// Build decoder IR (after default pre-regalloc passes) and return textual Linear IR dump.
///
/// Intended for snapshot tests and debugging.
pub fn debug_linear_ir_text(
    shape: &'static facet::Shape,
    ir_decoder: &dyn format::IrDecoder,
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

/// Compile an encoder (serializer) for the given shape and format.
pub fn compile_encoder(
    shape: &'static facet::Shape,
    encoder: &dyn format::Encoder,
) -> CompiledEncoder {
    compiler::compile_encoder(shape, encoder)
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

/// Serialize a value using a compiled encoder, returning the output bytes.
pub fn serialize<T: facet::Facet<'static>>(encoder: &CompiledEncoder, value: &T) -> Vec<u8> {
    let mut ctx = context::EncodeContext::new();
    unsafe {
        (encoder.func())(value as *const T as *const u8, &mut ctx);
    }
    ctx.into_vec()
}

#[cfg(test)]
mod tests {
    use super::*;
    use facet::Facet;

    #[derive(Facet, Debug, PartialEq)]
    struct Friend {
        age: u32,
        name: String,
    }

    #[test]
    fn compile_decoder_with_backend_ir_supports_json_scalars() {
        let via_ir = compile_decoder_with_backend(
            <u32 as facet::Facet>::SHAPE,
            &json::KajitJson,
            DecoderBackend::Ir,
        );
        let got: u32 = from_str(&via_ir, "42").unwrap();
        assert_eq!(got, 42);
    }

    #[test]
    fn compile_decoder_with_backend_ir_supports_json_structs() {
        let via_ir =
            compile_decoder_with_backend(Friend::SHAPE, &json::KajitJson, DecoderBackend::Ir);
        let got: Friend = from_str(&via_ir, r#"{"name":"Alice","age":42}"#).unwrap();
        assert_eq!(
            got,
            Friend {
                age: 42,
                name: "Alice".into(),
            }
        );
    }

    // --- Milestone 4: All scalar types ---

    #[derive(Facet, Debug, PartialEq)]
    struct AllScalars {
        a_bool: bool,
        a_u8: u8,
        a_u16: u16,
        a_u32: u32,
        a_u64: u64,
        a_u128: u128,
        a_usize: usize,
        a_i8: i8,
        a_i16: i16,
        a_i32: i32,
        a_i64: i64,
        a_i128: i128,
        a_isize: isize,
        a_f32: f32,
        a_f64: f64,
        a_char: char,
        a_name: String,
    }

    // --- Milestone 5: Nested structs ---

    #[derive(Facet, Debug, PartialEq)]
    struct Address {
        city: String,
        zip: u32,
    }

    #[derive(Facet, Debug, PartialEq)]
    struct Person {
        name: String,
        age: u32,
        address: Address,
    }

    #[derive(Facet, Debug, PartialEq)]
    struct Inner {
        x: u32,
    }

    #[derive(Facet, Debug, PartialEq)]
    struct Middle {
        inner: Inner,
        y: u32,
    }

    #[derive(Facet, Debug, PartialEq)]
    struct Outer {
        middle: Middle,
        z: u32,
    }

    // --- Milestone 6: Enums ---

    #[derive(Facet, Debug, PartialEq)]
    #[repr(u8)]
    enum Animal {
        Cat,
        Dog { name: String, good_boy: bool },
        Parrot(String),
    }

    // --- Milestone 7: Flatten ---
    #[derive(Facet, Debug, PartialEq)]
    struct Metadata {
        version: u32,
        author: String,
    }

    #[derive(Facet, Debug, PartialEq)]
    struct Document {
        title: String,
        #[facet(flatten)]
        meta: Metadata,
    }

    // r[verify deser.flatten.conflict]
    #[test]
    #[should_panic(expected = "field name collision")]
    fn flatten_name_collision() {
        #[derive(Facet)]
        struct Collider {
            x: u32,
        }

        #[derive(Facet)]
        struct HasCollision {
            x: u32,
            #[facet(flatten)]
            inner: Collider,
        }

        compile_decoder(HasCollision::SHAPE, &json::KajitJson);
    }

    // ═══════════════════════════════════════════════════════════════════
    // #20: rename / rename_all
    // ═══════════════════════════════════════════════════════════════════

    #[derive(Facet, Debug, PartialEq)]
    struct RenameField {
        #[facet(rename = "user_name")]
        name: String,
        age: u32,
    }

    #[derive(Facet, Debug, PartialEq)]
    #[facet(rename_all = "camelCase")]
    struct CamelCaseStruct {
        user_name: String,
        birth_year: u32,
    }

    // ═══════════════════════════════════════════════════════════════════
    // #21: deny_unknown_fields
    // ═══════════════════════════════════════════════════════════════════

    #[derive(Facet, Debug, PartialEq)]
    #[facet(deny_unknown_fields)]
    struct Strict {
        x: u32,
        y: u32,
    }

    // ═══════════════════════════════════════════════════════════════════
    // #19: default
    // ═══════════════════════════════════════════════════════════════════

    #[derive(Facet, Debug, PartialEq)]
    struct WithDefault {
        name: String,
        #[facet(default)]
        score: u32,
    }

    #[derive(Facet, Debug, PartialEq)]
    struct WithDefaultString {
        #[facet(default)]
        label: String,
        value: u32,
    }

    #[derive(Facet, Debug, PartialEq)]
    #[facet(default)]
    struct AllDefault {
        x: u32,
        y: u32,
    }

    impl Default for AllDefault {
        fn default() -> Self {
            AllDefault { x: 10, y: 20 }
        }
    }

    // ═══════════════════════════════════════════════════════════════════
    // #22: skip / skip_deserializing
    // ═══════════════════════════════════════════════════════════════════

    #[derive(Facet, Debug, PartialEq)]
    struct WithSkip {
        name: String,
        #[facet(skip, default)]
        cached: u32,
    }

    #[derive(Facet, Debug, PartialEq)]
    struct WithSkipDeser {
        name: String,
        #[facet(skip_deserializing, default)]
        internal: u32,
    }

    #[derive(Facet, Debug, PartialEq)]
    struct SkipWithCustomDefault {
        value: u32,
        #[facet(skip, default = 42)]
        magic: u32,
    }

    // ── Smart pointer tests ──────────────────────────────────────────

    #[derive(Facet, Debug, PartialEq)]
    struct BoxedScalar {
        value: Box<u32>,
    }

    #[derive(Facet, Debug, PartialEq)]
    struct BoxedString {
        #[allow(clippy::box_collection)]
        name: Box<String>,
    }

    #[derive(Facet, Debug, PartialEq)]
    struct BoxedNested {
        inner: Box<Friend>,
    }

    // r[verify deser.pointer.nesting]
    #[derive(Facet, Debug, PartialEq)]
    struct OptionBox {
        value: Option<Box<u32>>,
    }

    #[derive(Facet, Debug, PartialEq)]
    struct VecBox {
        #[allow(clippy::vec_box)]
        items: Vec<Box<u32>>,
    }

    #[derive(Facet, Debug, PartialEq)]
    struct ArcScalar {
        value: std::sync::Arc<u32>,
    }

    #[derive(Facet, Debug, PartialEq)]
    struct UnitField {
        geo: (),
        name: String,
    }

    // --- Encode tests ---
}

#[cfg(all(test, not(target_os = "windows")))]
mod disasm_tests;
