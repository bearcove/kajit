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

fn scrub_volatile_intrinsic_addrs(text: &str) -> String {
    // Display dumps include intrinsic function pointer addresses which are process-local and
    // unstable across runs. Replace only those pointer fields to keep snapshots deterministic.
    let text = scrub_hex_after_prefix(text, "CallIntrinsic(0x");
    scrub_hex_after_prefix(&text, "call_intrinsic(0x")
}

fn scrub_hex_after_prefix(input: &str, prefix: &str) -> String {
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

            if i > start {
                out.push_str("<ptr>");
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
    use std::borrow::Cow;

    use super::*;
    use facet::Facet;

    #[derive(Facet, Debug, PartialEq)]
    struct Friend {
        age: u32,
        name: String,
    }

    // r[verify deser.postcard.struct]
    #[test]
    fn postcard_flat_struct() {
        // age=42 → postcard varint 0x2A (42 < 128, so single byte)
        // name="Alice" → varint(5)=0x05 + b"Alice"
        let input = [0x2A, 0x05, b'A', b'l', b'i', b'c', b'e'];
        let deser = compile_decoder(Friend::SHAPE, &postcard::KajitPostcard);
        let result: Friend = deserialize(&deser, &input).unwrap();
        assert_eq!(
            result,
            Friend {
                age: 42,
                name: "Alice".into()
            }
        );
    }

    #[test]
    fn postcard_flat_struct_via_ir() {
        let input = [0x2A, 0x05, b'A', b'l', b'i', b'c', b'e'];
        let deser = compile_decoder_via_ir(Friend::SHAPE, &postcard::KajitPostcard);
        let result: Friend = deserialize(&deser, &input).unwrap();
        assert_eq!(
            result,
            Friend {
                age: 42,
                name: "Alice".into()
            }
        );
    }

    #[test]
    fn postcard_nested_struct_ir_uses_apply_nodes() {
        #[derive(Facet)]
        struct Inner {
            x: u32,
        }

        #[derive(Facet)]
        struct Outer {
            inner: Inner,
            y: u32,
        }

        let func = crate::compiler::build_decoder_ir(Outer::SHAPE, &postcard::KajitPostcard);
        assert!(
            func.lambdas.len() >= 2,
            "expected at least root + nested lambda"
        );
        let apply_count = func
            .nodes
            .iter()
            .filter(|(_, n)| matches!(&n.kind, crate::ir::NodeKind::Apply { .. }))
            .count();
        assert!(
            apply_count >= 1,
            "expected at least one apply node for nested shape"
        );
    }

    #[test]
    fn postcard_nested_struct_via_ir() {
        #[derive(Facet, Debug, PartialEq)]
        struct Inner {
            x: u32,
        }

        #[derive(Facet, Debug, PartialEq)]
        struct Outer {
            inner: Inner,
            y: u32,
        }

        #[derive(serde::Serialize)]
        struct InnerSerde {
            x: u32,
        }

        #[derive(serde::Serialize)]
        struct OuterSerde {
            inner: InnerSerde,
            y: u32,
        }

        let source = OuterSerde {
            inner: InnerSerde { x: 7 },
            y: 99,
        };
        let encoded = ::postcard::to_allocvec(&source).unwrap();
        let deser = compile_decoder_via_ir(Outer::SHAPE, &postcard::KajitPostcard);
        let result: Outer = deserialize(&deser, &encoded).unwrap();
        assert_eq!(
            result,
            Outer {
                inner: Inner { x: 7 },
                y: 99,
            }
        );
    }

    #[test]
    fn postcard_legacy_and_ir_match() {
        let input = [0x2A, 0x05, b'A', b'l', b'i', b'c', b'e'];
        let legacy = compile_decoder_legacy(Friend::SHAPE, &postcard::KajitPostcard);
        let via_ir = compile_decoder_via_ir(Friend::SHAPE, &postcard::KajitPostcard);

        let a: Friend = deserialize(&legacy, &input).unwrap();
        let b: Friend = deserialize(&via_ir, &input).unwrap();
        assert_eq!(a, b);
    }

    #[test]
    fn postcard_compile_decoder_with_backend_routes_to_selected_path() {
        let input = [0x2A, 0x05, b'A', b'l', b'i', b'c', b'e'];

        let legacy = compile_decoder_with_backend(
            Friend::SHAPE,
            &postcard::KajitPostcard,
            DecoderBackend::Legacy,
        );
        let via_ir = compile_decoder_with_backend(
            Friend::SHAPE,
            &postcard::KajitPostcard,
            DecoderBackend::Ir,
        );

        let a: Friend = deserialize(&legacy, &input).unwrap();
        let b: Friend = deserialize(&via_ir, &input).unwrap();
        assert_eq!(a, b);
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

    #[test]
    fn json_struct_ir_orders_key_read_before_key_compare() {
        let mut func = compiler::build_decoder_ir(Friend::SHAPE, &json::KajitJson);
        ir_passes::run_default_passes(&mut func);
        let linear = linearize::linearize(&mut func);
        let linear_text = format!("{linear}");

        let read_key_pat = format!(
            "call_intrinsic 0x{:x}",
            json_intrinsics::kajit_json_read_key as *const () as usize
        );
        let key_eq_pat = format!(
            "call_pure 0x{:x}",
            json_intrinsics::kajit_json_key_equals as *const () as usize
        );
        let slot_read_pat = " = slot[0]";

        let read_key_pos = linear_text
            .find(&read_key_pat)
            .unwrap_or_else(|| panic!("missing read_key call in linear IR:\n{linear_text}"));
        let slot_read_pos = linear_text
            .find(slot_read_pat)
            .unwrap_or_else(|| panic!("missing slot read in linear IR:\n{linear_text}"));
        let key_eq_pos = linear_text
            .find(&key_eq_pat)
            .unwrap_or_else(|| panic!("missing key_equals call in linear IR:\n{linear_text}"));

        assert!(
            read_key_pos < slot_read_pos && slot_read_pos < key_eq_pos,
            "expected read_key -> slot_read -> key_compare order in linear IR:\n{linear_text}"
        );
    }

    #[test]
    fn postcard_from_str_entrypoint() {
        let bytes = [0x2A, 0x05, b'A', b'l', b'i', b'c', b'e'];
        let input = core::str::from_utf8(&bytes).unwrap();
        let deser = compile_decoder(Friend::SHAPE, &postcard::KajitPostcard);
        let result: Friend = from_str(&deser, input).unwrap();
        assert_eq!(
            result,
            Friend {
                age: 42,
                name: "Alice".into()
            }
        );
    }

    // r[verify deser.json.struct]
    #[test]
    fn json_flat_struct() {
        let input = br#"{"age": 42, "name": "Alice"}"#;
        let deser = compile_decoder(Friend::SHAPE, &json::KajitJson);
        let result: Friend = deserialize(&deser, input).unwrap();
        assert_eq!(
            result,
            Friend {
                age: 42,
                name: "Alice".into()
            }
        );
    }

    #[test]
    fn json_from_str_entrypoint() {
        let input = r#"{"age": 42, "name": "Alice"}"#;
        let deser = compile_decoder(Friend::SHAPE, &json::KajitJson);
        let result: Friend = from_str(&deser, input).unwrap();
        assert_eq!(
            result,
            Friend {
                age: 42,
                name: "Alice".into()
            }
        );
    }

    // r[verify deser.json.struct]
    #[test]
    fn json_reversed_key_order() {
        let input = br#"{"name": "Alice", "age": 42}"#;
        let deser = compile_decoder(Friend::SHAPE, &json::KajitJson);
        let result: Friend = deserialize(&deser, input).unwrap();
        assert_eq!(
            result,
            Friend {
                age: 42,
                name: "Alice".into()
            }
        );
    }

    // r[verify deser.json.struct.unknown-keys]
    #[test]
    fn json_unknown_keys_skipped() {
        let input = br#"{"age": 42, "extra": true, "name": "Alice"}"#;
        let deser = compile_decoder(Friend::SHAPE, &json::KajitJson);
        let result: Friend = deserialize(&deser, input).unwrap();
        assert_eq!(
            result,
            Friend {
                age: 42,
                name: "Alice".into()
            }
        );
    }

    // r[verify deser.json.struct]
    #[test]
    fn json_empty_object_missing_fields() {
        let input = b"{}";
        let deser = compile_decoder(Friend::SHAPE, &json::KajitJson);
        let result = deserialize::<Friend>(&deser, input);
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert_eq!(err.code, context::ErrorCode::MissingRequiredField);
    }

    #[derive(Facet, Debug, PartialEq)]
    struct BorrowedFriend<'a> {
        age: u32,
        name: &'a str,
    }

    #[derive(Facet, Debug, PartialEq)]
    struct CowFriend<'a> {
        age: u32,
        name: Cow<'a, str>,
    }

    #[test]
    fn postcard_borrowed_str_zero_copy() {
        let input = [0x2A, 0x05, b'A', b'l', b'i', b'c', b'e'];
        let deser = compile_decoder(BorrowedFriend::SHAPE, &postcard::KajitPostcard);
        let result: BorrowedFriend<'_> = deserialize(&deser, &input).unwrap();
        assert_eq!(result.age, 42);
        assert_eq!(result.name, "Alice");
        assert_eq!(result.name.as_ptr(), unsafe { input.as_ptr().add(2) });
    }

    #[test]
    fn postcard_cow_str_borrowed_zero_copy() {
        let input = [0x2A, 0x05, b'A', b'l', b'i', b'c', b'e'];
        let deser = compile_decoder(CowFriend::SHAPE, &postcard::KajitPostcard);
        let result: CowFriend<'_> = deserialize(&deser, &input).unwrap();
        assert_eq!(result.age, 42);
        assert!(matches!(result.name, Cow::Borrowed("Alice")));
    }

    #[test]
    fn json_borrowed_str_zero_copy_fast_path() {
        let input = br#"{"age":42,"name":"Alice"}"#;
        let name_start = input.windows(5).position(|w| w == b"Alice").unwrap();
        let deser = compile_decoder(BorrowedFriend::SHAPE, &json::KajitJson);
        let result: BorrowedFriend<'_> = deserialize(&deser, input).unwrap();
        assert_eq!(result.age, 42);
        assert_eq!(result.name, "Alice");
        assert_eq!(result.name.as_ptr(), unsafe {
            input.as_ptr().add(name_start)
        });
    }

    #[test]
    fn json_borrowed_str_escape_is_error() {
        let input = br#"{"age":42,"name":"A\nB"}"#;
        let deser = compile_decoder(BorrowedFriend::SHAPE, &json::KajitJson);
        let err = deserialize::<BorrowedFriend<'_>>(&deser, input).unwrap_err();
        assert_eq!(err.code, context::ErrorCode::InvalidEscapeSequence);
    }

    #[test]
    fn json_cow_str_fast_path_borrowed() {
        let input = br#"{"age":42,"name":"Alice"}"#;
        let deser = compile_decoder(CowFriend::SHAPE, &json::KajitJson);
        let result: CowFriend<'_> = deserialize(&deser, input).unwrap();
        assert_eq!(result.age, 42);
        assert!(matches!(result.name, Cow::Borrowed("Alice")));
    }

    #[test]
    fn json_cow_str_escape_slow_path_owned() {
        let input = br#"{"age":42,"name":"A\nB"}"#;
        let deser = compile_decoder(CowFriend::SHAPE, &json::KajitJson);
        let result: CowFriend<'_> = deserialize(&deser, input).unwrap();
        assert_eq!(result.age, 42);
        assert!(matches!(result.name, Cow::Owned(ref s) if s == "A\nB"));
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

    // r[verify deser.postcard.scalar.varint]
    // r[verify deser.postcard.scalar.float]
    // r[verify deser.postcard.scalar.bool]
    #[test]
    fn postcard_all_scalars() {
        use serde::Serialize;

        #[derive(Serialize)]
        struct AllScalarsSerde {
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

        let source = AllScalarsSerde {
            a_bool: true,
            a_u8: 200,
            a_u16: 1000,
            a_u32: 70000,
            a_u64: 1_000_000_000_000,
            a_u128: 18_446_744_073_709_551_621u128,
            a_usize: 12345,
            a_i8: -42,
            a_i16: -1000,
            a_i32: -70000,
            a_i64: -1_000_000_000_000,
            a_i128: -18_446_744_073_709_551_621i128,
            a_isize: -12345,
            #[allow(clippy::approx_constant)]
            a_f32: 3.14,
            #[allow(clippy::approx_constant)]
            a_f64: 2.718281828459045,
            a_char: 'ß',
            a_name: "hello".into(),
        };

        let encoded = ::postcard::to_allocvec(&source).unwrap();
        let deser = compile_decoder(AllScalars::SHAPE, &postcard::KajitPostcard);
        let result: AllScalars = deserialize(&deser, &encoded).unwrap();

        assert!(result.a_bool);
        assert_eq!(result.a_u8, 200);
        assert_eq!(result.a_u16, 1000);
        assert_eq!(result.a_u32, 70000);
        assert_eq!(result.a_u64, 1_000_000_000_000);
        assert_eq!(result.a_u128, 18_446_744_073_709_551_621u128);
        assert_eq!(result.a_usize, 12345);
        assert_eq!(result.a_i8, -42);
        assert_eq!(result.a_i16, -1000);
        assert_eq!(result.a_i32, -70000);
        assert_eq!(result.a_i64, -1_000_000_000_000);
        assert_eq!(result.a_i128, -18_446_744_073_709_551_621i128);
        assert_eq!(result.a_isize, -12345);
        assert_eq!(result.a_f32, 3.14);
        assert_eq!(result.a_f64, 2.718281828459045);
        assert_eq!(result.a_char, 'ß');
        assert_eq!(result.a_name, "hello");
    }

    // r[verify deser.json.scalar.integer]
    // r[verify deser.json.scalar.float]
    // r[verify deser.json.scalar.bool]
    #[test]
    fn json_all_scalars() {
        let input = br#"{
            "a_bool": true,
            "a_u8": 200,
            "a_u16": 1000,
            "a_u32": 70000,
            "a_u64": 1000000000000,
            "a_u128": 18446744073709551621,
            "a_usize": 12345,
            "a_i8": -42,
            "a_i16": -1000,
            "a_i32": -70000,
            "a_i64": -1000000000000,
            "a_i128": -18446744073709551621,
            "a_isize": -12345,
            "a_f32": 3.14,
            "a_f64": 2.718281828459045,
            "a_char": "\u00df",
            "a_name": "hello"
        }"#;

        let deser = compile_decoder(AllScalars::SHAPE, &json::KajitJson);
        let result: AllScalars = deserialize(&deser, input).unwrap();

        assert!(result.a_bool);
        assert_eq!(result.a_u8, 200);
        assert_eq!(result.a_u16, 1000);
        assert_eq!(result.a_u32, 70000);
        assert_eq!(result.a_u64, 1_000_000_000_000);
        assert_eq!(result.a_u128, 18_446_744_073_709_551_621u128);
        assert_eq!(result.a_usize, 12345);
        assert_eq!(result.a_i8, -42);
        assert_eq!(result.a_i16, -1000);
        assert_eq!(result.a_i32, -70000);
        assert_eq!(result.a_i64, -1_000_000_000_000);
        assert_eq!(result.a_i128, -18_446_744_073_709_551_621i128);
        assert_eq!(result.a_isize, -12345);
        assert_eq!(result.a_f32, 3.14);
        assert_eq!(result.a_f64, 2.718281828459045);
        assert_eq!(result.a_char, 'ß');
        assert_eq!(result.a_name, "hello");
    }

    // r[verify deser.json.scalar.bool]
    #[test]
    fn json_bool_true_false() {
        #[derive(Facet, Debug, PartialEq)]
        struct Bools {
            a: bool,
            b: bool,
        }

        let input = br#"{"a": true, "b": false}"#;
        let deser = compile_decoder(Bools::SHAPE, &json::KajitJson);
        let result: Bools = deserialize(&deser, input).unwrap();
        assert!(result.a);
        assert!(!result.b);
    }

    // r[verify deser.postcard.scalar.bool]
    #[test]
    fn postcard_bool_true_false() {
        #[derive(Facet, Debug, PartialEq)]
        struct Bools {
            a: bool,
            b: bool,
        }

        // postcard: true=1, false=0
        let input = [1u8, 0u8];
        let deser = compile_decoder(Bools::SHAPE, &postcard::KajitPostcard);
        let result: Bools = deserialize(&deser, &input).unwrap();
        assert!(result.a);
        assert!(!result.b);
    }

    // r[verify deser.json.scalar.integer]
    #[test]
    fn json_boundary_values() {
        #[derive(Facet, Debug, PartialEq)]
        struct Boundaries {
            u8_max: u8,
            u16_max: u16,
            i8_min: i8,
            i8_max: i8,
            i16_min: i16,
            i32_min: i32,
        }

        let input = br#"{
            "u8_max": 255,
            "u16_max": 65535,
            "i8_min": -128,
            "i8_max": 127,
            "i16_min": -32768,
            "i32_min": -2147483648
        }"#;

        let deser = compile_decoder(Boundaries::SHAPE, &json::KajitJson);
        let result: Boundaries = deserialize(&deser, input).unwrap();
        assert_eq!(result.u8_max, 255);
        assert_eq!(result.u16_max, 65535);
        assert_eq!(result.i8_min, -128);
        assert_eq!(result.i8_max, 127);
        assert_eq!(result.i16_min, -32768);
        assert_eq!(result.i32_min, -2147483648);
    }

    // r[verify deser.json.scalar.integer]
    #[test]
    fn json_u8_out_of_range() {
        #[derive(Facet, Debug)]
        struct Tiny {
            val: u8,
        }

        let input = br#"{"val": 256}"#;
        let deser = compile_decoder(Tiny::SHAPE, &json::KajitJson);
        let result = deserialize::<Tiny>(&deser, input);
        assert!(result.is_err());
        assert_eq!(result.unwrap_err().code, ErrorCode::NumberOutOfRange);
    }

    // r[verify deser.json.scalar.float]
    #[test]
    fn json_float_scientific() {
        #[derive(Facet, Debug, PartialEq)]
        struct Floats {
            a: f64,
            b: f64,
        }

        let input = br#"{"a": 1.5e2, "b": -3.14}"#;
        let deser = compile_decoder(Floats::SHAPE, &json::KajitJson);
        let result: Floats = deserialize(&deser, input).unwrap();
        assert_eq!(result.a, 150.0);
        assert_eq!(result.b, -3.14);
    }

    // r[verify deser.json.scalar.float]
    // r[verify deser.json.scalar.float.sign]
    // r[verify deser.json.scalar.float.digits]
    // r[verify deser.json.scalar.float.overflow-digits]
    // r[verify deser.json.scalar.float.dot]
    // r[verify deser.json.scalar.float.exponent]
    // r[verify deser.json.scalar.float.zero]
    // r[verify deser.json.scalar.float.exact-int]
    // r[verify deser.json.scalar.float.uscale]
    // r[verify deser.json.scalar.float.pack]
    // r[verify deser.json.scalar.float.pack.subnormal]
    // r[verify deser.json.scalar.float.pack.overflow]
    #[test]
    fn json_f64_edge_cases() {
        #[derive(Facet, Debug, PartialEq)]
        struct F {
            x: f64,
        }

        let cases: &[(&[u8], f64)] = &[
            // Exact integers
            (br#"{"x":0}"#, 0.0),
            (br#"{"x":1}"#, 1.0),
            (br#"{"x":42}"#, 42.0),
            (br#"{"x":9007199254740992}"#, 9007199254740992.0), // 2^53
            // Simple decimals
            (br#"{"x":0.5}"#, 0.5),
            (br#"{"x":1.0}"#, 1.0),
            #[allow(clippy::approx_constant)]
            (br#"{"x":3.14}"#, 3.14),
            (br#"{"x":2.718281828459045}"#, 2.718281828459045),
            // Negative values
            (br#"{"x":-1.0}"#, -1.0),
            (br#"{"x":-0.0}"#, -0.0_f64),
            (br#"{"x":-3.14}"#, -3.14),
            // Scientific notation
            (br#"{"x":1e10}"#, 1e10),
            (br#"{"x":1.5e2}"#, 150.0),
            (br#"{"x":1e-10}"#, 1e-10),
            (br#"{"x":1E10}"#, 1e10),
            (br#"{"x":1e+10}"#, 1e10),
            (br#"{"x":1e0}"#, 1.0),
            // Leading zeros (valid JSON numbers can have 0.xxx)
            (br#"{"x":0.001}"#, 0.001),
            (br#"{"x":0.0000000000000000000001}"#, 1e-22),
            // Large/small exponents
            (br#"{"x":1e308}"#, 1e308),
            (br#"{"x":1.7976931348623157e308}"#, f64::MAX),
            (br#"{"x":1e-308}"#, 1e-308),
            // Min normal
            (br#"{"x":2.2250738585072014e-308}"#, 2.2250738585072014e-308),
            // Overflow → infinity
            (br#"{"x":1e309}"#, f64::INFINITY),
            (br#"{"x":-1e309}"#, f64::NEG_INFINITY),
            // Underflow → zero
            (br#"{"x":1e-400}"#, 0.0),
            // > 19 significant digits (overflow digits get dropped)
            (br#"{"x":12345678901234567890.0}"#, 12345678901234567890.0),
            // With leading whitespace after colon (tests ws skip)
            (br#"{"x": 1.0}"#, 1.0),
            (br#"{"x":	1.0}"#, 1.0), // tab
        ];

        let deser = compile_decoder(F::SHAPE, &json::KajitJson);
        for (input, expected) in cases {
            let result: F = deserialize(&deser, input).unwrap();
            assert_eq!(
                result.x.to_bits(),
                expected.to_bits(),
                "input={:?}: got {} ({:#018x}), expected {} ({:#018x})",
                std::str::from_utf8(input).unwrap(),
                result.x,
                result.x.to_bits(),
                expected,
                expected.to_bits(),
            );
        }
    }

    // r[verify deser.json.scalar.float.uscale.table]
    // r[verify deser.json.scalar.float.uscale.mul128]
    // r[verify deser.json.scalar.float.uscale.clz]
    #[test]
    fn json_f64_canada_roundtrip() {
        #[derive(Facet, Debug)]
        struct Coord {
            v: f64,
        }

        let compressed = include_bytes!("../fixtures/canada.json.br");
        let mut json_bytes = Vec::new();
        brotli::BrotliDecompress(&mut std::io::Cursor::new(compressed), &mut json_bytes).unwrap();

        let deser = compile_decoder(Coord::SHAPE, &json::KajitJson);

        let mut mismatches = 0;
        let mut total = 0;
        let mut i = 0;
        while i < json_bytes.len() {
            if json_bytes[i] == b'-' || json_bytes[i].is_ascii_digit() {
                let start = i;
                i += 1;
                while i < json_bytes.len() {
                    match json_bytes[i] {
                        b'0'..=b'9' | b'.' | b'e' | b'E' | b'+' | b'-' => i += 1,
                        _ => break,
                    }
                }
                let s = &json_bytes[start..i];
                if s.contains(&b'.') || s.contains(&b'e') || s.contains(&b'E') {
                    total += 1;
                    let std_val: f64 = std::str::from_utf8(s).unwrap().parse().unwrap();
                    let json_input = format!(r#"{{"v":{}}}"#, std::str::from_utf8(s).unwrap());
                    let result: Coord = deserialize(&deser, json_input.as_bytes()).unwrap();
                    if std_val.to_bits() != result.v.to_bits() {
                        if mismatches < 10 {
                            eprintln!(
                                "MISMATCH: {:?} → std={std_val:?} jit={:?}",
                                std::str::from_utf8(s).unwrap(),
                                result.v,
                            );
                        }
                        mismatches += 1;
                    }
                }
            } else {
                i += 1;
            }
        }
        eprintln!("{total} floats checked, {mismatches} mismatches");
        assert_eq!(mismatches, 0, "{mismatches}/{total} values differ from std");
    }

    // r[verify deser.postcard.scalar.varint]
    #[test]
    fn postcard_boundary_values() {
        use serde::Serialize;

        #[derive(Facet, Debug, PartialEq)]
        struct Boundaries {
            u8_max: u8,
            u64_big: u64,
            i8_min: i8,
            i64_min: i64,
        }

        #[derive(Serialize)]
        struct BoundariesSerde {
            u8_max: u8,
            u64_big: u64,
            i8_min: i8,
            i64_min: i64,
        }

        let source = BoundariesSerde {
            u8_max: 255,
            u64_big: u64::MAX,
            i8_min: i8::MIN,
            i64_min: i64::MIN,
        };

        let encoded = ::postcard::to_allocvec(&source).unwrap();
        let deser = compile_decoder(Boundaries::SHAPE, &postcard::KajitPostcard);
        let result: Boundaries = deserialize(&deser, &encoded).unwrap();

        assert_eq!(result.u8_max, 255);
        assert_eq!(result.u64_big, u64::MAX);
        assert_eq!(result.i8_min, i8::MIN);
        assert_eq!(result.i64_min, i64::MIN);
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

    // r[verify deser.nested-struct]
    // r[verify deser.nested-struct.offset]
    #[test]
    fn postcard_nested_struct() {
        use serde::Serialize;

        #[derive(Serialize)]
        struct AddressSerde {
            city: String,
            zip: u32,
        }

        #[derive(Serialize)]
        struct PersonSerde {
            name: String,
            age: u32,
            address: AddressSerde,
        }

        let source = PersonSerde {
            name: "Alice".into(),
            age: 30,
            address: AddressSerde {
                city: "Portland".into(),
                zip: 97201,
            },
        };

        let encoded = ::postcard::to_allocvec(&source).unwrap();
        let deser = compile_decoder(Person::SHAPE, &postcard::KajitPostcard);
        let result: Person = deserialize(&deser, &encoded).unwrap();

        assert_eq!(
            result,
            Person {
                name: "Alice".into(),
                age: 30,
                address: Address {
                    city: "Portland".into(),
                    zip: 97201,
                },
            }
        );
    }

    // r[verify deser.nested-struct]
    // r[verify deser.nested-struct.offset]
    #[test]
    fn json_nested_struct() {
        let input =
            br#"{"name": "Alice", "age": 30, "address": {"city": "Portland", "zip": 97201}}"#;
        let deser = compile_decoder(Person::SHAPE, &json::KajitJson);
        let result: Person = deserialize(&deser, input).unwrap();

        assert_eq!(
            result,
            Person {
                name: "Alice".into(),
                age: 30,
                address: Address {
                    city: "Portland".into(),
                    zip: 97201,
                },
            }
        );
    }

    // r[verify deser.nested-struct]
    #[test]
    fn json_nested_struct_reversed_keys() {
        let input =
            br#"{"address": {"zip": 97201, "city": "Portland"}, "age": 30, "name": "Alice"}"#;
        let deser = compile_decoder(Person::SHAPE, &json::KajitJson);
        let result: Person = deserialize(&deser, input).unwrap();

        assert_eq!(
            result,
            Person {
                name: "Alice".into(),
                age: 30,
                address: Address {
                    city: "Portland".into(),
                    zip: 97201,
                },
            }
        );
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

    // r[verify deser.nested-struct]
    #[test]
    fn postcard_deeply_nested() {
        use serde::Serialize;

        #[derive(Serialize)]
        struct InnerSerde {
            x: u32,
        }
        #[derive(Serialize)]
        struct MiddleSerde {
            inner: InnerSerde,
            y: u32,
        }
        #[derive(Serialize)]
        struct OuterSerde {
            middle: MiddleSerde,
            z: u32,
        }

        let source = OuterSerde {
            middle: MiddleSerde {
                inner: InnerSerde { x: 1 },
                y: 2,
            },
            z: 3,
        };

        let encoded = ::postcard::to_allocvec(&source).unwrap();
        let deser = compile_decoder(Outer::SHAPE, &postcard::KajitPostcard);
        let result: Outer = deserialize(&deser, &encoded).unwrap();

        assert_eq!(
            result,
            Outer {
                middle: Middle {
                    inner: Inner { x: 1 },
                    y: 2,
                },
                z: 3,
            }
        );
    }

    // r[verify deser.nested-struct]
    #[test]
    fn json_deeply_nested() {
        let input = br#"{"middle": {"inner": {"x": 1}, "y": 2}, "z": 3}"#;
        let deser = compile_decoder(Outer::SHAPE, &json::KajitJson);
        let result: Outer = deserialize(&deser, input).unwrap();

        assert_eq!(
            result,
            Outer {
                middle: Middle {
                    inner: Inner { x: 1 },
                    y: 2,
                },
                z: 3,
            }
        );
    }

    // r[verify deser.nested-struct]
    #[test]
    fn postcard_shared_inner_type() {
        use serde::Serialize;

        #[derive(Facet, Debug, PartialEq)]
        struct TwoAddresses {
            home: Address,
            work: Address,
        }

        #[derive(Serialize)]
        struct AddressSerde {
            city: String,
            zip: u32,
        }
        #[derive(Serialize)]
        struct TwoAddressesSerde {
            home: AddressSerde,
            work: AddressSerde,
        }

        let source = TwoAddressesSerde {
            home: AddressSerde {
                city: "Portland".into(),
                zip: 97201,
            },
            work: AddressSerde {
                city: "Seattle".into(),
                zip: 98101,
            },
        };

        let encoded = ::postcard::to_allocvec(&source).unwrap();
        let deser = compile_decoder(TwoAddresses::SHAPE, &postcard::KajitPostcard);
        let result: TwoAddresses = deserialize(&deser, &encoded).unwrap();

        assert_eq!(
            result,
            TwoAddresses {
                home: Address {
                    city: "Portland".into(),
                    zip: 97201,
                },
                work: Address {
                    city: "Seattle".into(),
                    zip: 98101,
                },
            }
        );
    }

    // --- Milestone 6: Enums ---

    #[derive(Facet, Debug, PartialEq)]
    #[repr(u8)]
    enum Animal {
        Cat,
        Dog { name: String, good_boy: bool },
        Parrot(String),
    }

    // r[verify deser.postcard.enum]
    // r[verify deser.postcard.enum.unit]
    #[test]
    fn postcard_enum_unit_variant() {
        use serde::Serialize;

        #[derive(Serialize)]
        #[repr(u8)]
        enum AnimalSerde {
            Cat,
            #[allow(dead_code)]
            Dog {
                name: String,
                good_boy: bool,
            },
            #[allow(dead_code)]
            Parrot(String),
        }

        let encoded = ::postcard::to_allocvec(&AnimalSerde::Cat).unwrap();
        let deser = compile_decoder(Animal::SHAPE, &postcard::KajitPostcard);
        let result: Animal = deserialize(&deser, &encoded).unwrap();
        assert_eq!(result, Animal::Cat);
    }

    // r[verify deser.postcard.enum]
    // r[verify deser.postcard.enum.dispatch]
    #[test]
    fn postcard_enum_struct_variant() {
        use serde::Serialize;

        #[derive(Serialize)]
        #[repr(u8)]
        enum AnimalSerde {
            #[allow(dead_code)]
            Cat,
            Dog {
                name: String,
                good_boy: bool,
            },
            #[allow(dead_code)]
            Parrot(String),
        }

        let encoded = ::postcard::to_allocvec(&AnimalSerde::Dog {
            name: "Rex".into(),
            good_boy: true,
        })
        .unwrap();
        let deser = compile_decoder(Animal::SHAPE, &postcard::KajitPostcard);
        let result: Animal = deserialize(&deser, &encoded).unwrap();
        assert_eq!(
            result,
            Animal::Dog {
                name: "Rex".into(),
                good_boy: true
            }
        );
    }

    // r[verify deser.postcard.enum]
    #[test]
    fn postcard_enum_tuple_variant() {
        use serde::Serialize;

        #[derive(Serialize)]
        #[repr(u8)]
        enum AnimalSerde {
            #[allow(dead_code)]
            Cat,
            #[allow(dead_code)]
            Dog {
                name: String,
                good_boy: bool,
            },
            Parrot(String),
        }

        let encoded = ::postcard::to_allocvec(&AnimalSerde::Parrot("Polly".into())).unwrap();
        let deser = compile_decoder(Animal::SHAPE, &postcard::KajitPostcard);
        let result: Animal = deserialize(&deser, &encoded).unwrap();
        assert_eq!(result, Animal::Parrot("Polly".into()));
    }

    // r[verify deser.postcard.enum.dispatch]
    #[test]
    fn postcard_enum_unknown_discriminant() {
        // Discriminant 99 doesn't exist
        let input = [99u8];
        let deser = compile_decoder(Animal::SHAPE, &postcard::KajitPostcard);
        let result = deserialize::<Animal>(&deser, &input);
        assert!(result.is_err());
        assert_eq!(result.unwrap_err().code, ErrorCode::UnknownVariant);
    }

    // r[verify deser.postcard.enum]
    #[test]
    fn postcard_enum_as_struct_field() {
        use serde::Serialize;

        #[derive(Facet, Debug, PartialEq)]
        struct Zoo {
            name: String,
            star: Animal,
        }

        #[derive(Serialize)]
        #[repr(u8)]
        enum AnimalSerde {
            #[allow(dead_code)]
            Cat,
            Dog {
                name: String,
                good_boy: bool,
            },
            #[allow(dead_code)]
            Parrot(String),
        }

        #[derive(Serialize)]
        struct ZooSerde {
            name: String,
            star: AnimalSerde,
        }

        let encoded = ::postcard::to_allocvec(&ZooSerde {
            name: "City Zoo".into(),
            star: AnimalSerde::Dog {
                name: "Rex".into(),
                good_boy: true,
            },
        })
        .unwrap();
        let deser = compile_decoder(Zoo::SHAPE, &postcard::KajitPostcard);
        let result: Zoo = deserialize(&deser, &encoded).unwrap();
        assert_eq!(
            result,
            Zoo {
                name: "City Zoo".into(),
                star: Animal::Dog {
                    name: "Rex".into(),
                    good_boy: true
                },
            }
        );
    }

    // r[verify deser.json.enum.external]
    // r[verify deser.json.enum.external.unit-as-string]
    #[test]
    fn json_enum_unit_as_string() {
        let input = br#""Cat""#;
        let deser = compile_decoder(Animal::SHAPE, &json::KajitJson);
        let result: Animal = deserialize(&deser, input).unwrap();
        assert_eq!(result, Animal::Cat);
    }

    // r[verify deser.json.enum.external]
    // r[verify deser.json.enum.external.struct-variant]
    #[test]
    fn json_enum_struct_variant() {
        let input = br#"{"Dog": {"name": "Rex", "good_boy": true}}"#;
        let deser = compile_decoder(Animal::SHAPE, &json::KajitJson);
        let result: Animal = deserialize(&deser, input).unwrap();
        assert_eq!(
            result,
            Animal::Dog {
                name: "Rex".into(),
                good_boy: true
            }
        );
    }

    // r[verify deser.json.enum.external]
    // r[verify deser.json.enum.external.tuple-variant]
    #[test]
    fn json_enum_tuple_variant() {
        let input = br#"{"Parrot": "Polly"}"#;
        let deser = compile_decoder(Animal::SHAPE, &json::KajitJson);
        let result: Animal = deserialize(&deser, input).unwrap();
        assert_eq!(result, Animal::Parrot("Polly".into()));
    }

    // r[verify deser.json.enum.external]
    #[test]
    fn json_enum_unit_in_object() {
        // Unit variant inside object form: { "Cat": null }
        let input = br#"{"Cat": null}"#;
        let deser = compile_decoder(Animal::SHAPE, &json::KajitJson);
        let result: Animal = deserialize(&deser, input).unwrap();
        assert_eq!(result, Animal::Cat);
    }

    // r[verify deser.json.enum.external]
    #[test]
    fn json_enum_unknown_variant() {
        let input = br#""Snake""#;
        let deser = compile_decoder(Animal::SHAPE, &json::KajitJson);
        let result = deserialize::<Animal>(&deser, input);
        assert!(result.is_err());
        assert_eq!(result.unwrap_err().code, ErrorCode::UnknownVariant);
    }

    // r[verify deser.json.enum.external]
    #[test]
    fn json_enum_as_struct_field() {
        #[derive(Facet, Debug, PartialEq)]
        struct Zoo {
            name: String,
            star: Animal,
        }

        let input = br#"{"name": "City Zoo", "star": {"Dog": {"name": "Rex", "good_boy": true}}}"#;
        let deser = compile_decoder(Zoo::SHAPE, &json::KajitJson);
        let result: Zoo = deserialize(&deser, input).unwrap();
        assert_eq!(
            result,
            Zoo {
                name: "City Zoo".into(),
                star: Animal::Dog {
                    name: "Rex".into(),
                    good_boy: true
                },
            }
        );
    }

    // r[verify deser.json.enum.external.struct-variant]
    #[test]
    fn json_enum_struct_variant_reversed_keys() {
        let input = br#"{"Dog": {"good_boy": true, "name": "Rex"}}"#;
        let deser = compile_decoder(Animal::SHAPE, &json::KajitJson);
        let result: Animal = deserialize(&deser, input).unwrap();
        assert_eq!(
            result,
            Animal::Dog {
                name: "Rex".into(),
                good_boy: true
            }
        );
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

    // r[verify deser.flatten]
    // r[verify deser.flatten.offset-accumulation]
    // r[verify deser.flatten.inline]
    #[test]
    fn json_flatten_basic() {
        let input = br#"{"title": "Hello", "version": 1, "author": "Amos"}"#;
        let deser = compile_decoder(Document::SHAPE, &json::KajitJson);
        let result: Document = deserialize(&deser, input).unwrap();
        assert_eq!(
            result,
            Document {
                title: "Hello".into(),
                meta: Metadata {
                    version: 1,
                    author: "Amos".into(),
                },
            }
        );
    }

    // r[verify deser.flatten]
    #[test]
    fn json_flatten_reversed_keys() {
        let input = br#"{"author": "Amos", "version": 1, "title": "Hello"}"#;
        let deser = compile_decoder(Document::SHAPE, &json::KajitJson);
        let result: Document = deserialize(&deser, input).unwrap();
        assert_eq!(
            result,
            Document {
                title: "Hello".into(),
                meta: Metadata {
                    version: 1,
                    author: "Amos".into(),
                },
            }
        );
    }

    // r[verify deser.flatten]
    #[test]
    fn postcard_flatten_basic() {
        // Postcard flatten: fields appear in declaration order at parent level.
        // Document = { title: String, meta: Metadata { version: u32, author: String } }
        // With flatten, wire order is: title, version, author
        //
        // title="Hi" → varint(2)=0x02 + b"Hi"
        // version=1 → varint 0x01
        // author="A" → varint(1)=0x01 + b"A"
        let input = [0x02, b'H', b'i', 0x01, 0x01, b'A'];
        let deser = compile_decoder(Document::SHAPE, &postcard::KajitPostcard);
        let result: Document = deserialize(&deser, &input).unwrap();
        assert_eq!(
            result,
            Document {
                title: "Hi".into(),
                meta: Metadata {
                    version: 1,
                    author: "A".into(),
                },
            }
        );
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

    // --- Milestone 7: Adjacently tagged enums ---

    #[derive(Facet, Debug, PartialEq)]
    #[facet(tag = "type", content = "data")]
    #[repr(u8)]
    enum AdjAnimal {
        Cat,
        Dog { name: String, good_boy: bool },
        Parrot(String),
    }

    // r[verify deser.json.enum.adjacent]
    // r[verify deser.json.enum.adjacent.unit-variant]
    #[test]
    fn json_adjacent_unit_no_content() {
        let input = br#"{"type": "Cat"}"#;
        let deser = compile_decoder(AdjAnimal::SHAPE, &json::KajitJson);
        let result: AdjAnimal = deserialize(&deser, input).unwrap();
        assert_eq!(result, AdjAnimal::Cat);
    }

    // r[verify deser.json.enum.adjacent]
    // r[verify deser.json.enum.adjacent.unit-variant]
    #[test]
    fn json_adjacent_unit_with_null_content() {
        let input = br#"{"type": "Cat", "data": null}"#;
        let deser = compile_decoder(AdjAnimal::SHAPE, &json::KajitJson);
        let result: AdjAnimal = deserialize(&deser, input).unwrap();
        assert_eq!(result, AdjAnimal::Cat);
    }

    // r[verify deser.json.enum.adjacent]
    #[test]
    fn json_adjacent_struct_variant() {
        let input = br#"{"type": "Dog", "data": {"name": "Rex", "good_boy": true}}"#;
        let deser = compile_decoder(AdjAnimal::SHAPE, &json::KajitJson);
        let result: AdjAnimal = deserialize(&deser, input).unwrap();
        assert_eq!(
            result,
            AdjAnimal::Dog {
                name: "Rex".into(),
                good_boy: true,
            }
        );
    }

    // r[verify deser.json.enum.adjacent]
    #[test]
    fn json_adjacent_struct_variant_reversed_fields() {
        let input = br#"{"type": "Dog", "data": {"good_boy": true, "name": "Rex"}}"#;
        let deser = compile_decoder(AdjAnimal::SHAPE, &json::KajitJson);
        let result: AdjAnimal = deserialize(&deser, input).unwrap();
        assert_eq!(
            result,
            AdjAnimal::Dog {
                name: "Rex".into(),
                good_boy: true,
            }
        );
    }

    // r[verify deser.json.enum.adjacent]
    // r[verify deser.json.enum.adjacent.tuple-variant]
    #[test]
    fn json_adjacent_tuple_variant() {
        let input = br#"{"type": "Parrot", "data": "Polly"}"#;
        let deser = compile_decoder(AdjAnimal::SHAPE, &json::KajitJson);
        let result: AdjAnimal = deserialize(&deser, input).unwrap();
        assert_eq!(result, AdjAnimal::Parrot("Polly".into()));
    }

    // r[verify deser.json.enum.adjacent]
    #[test]
    fn json_adjacent_unknown_variant() {
        let input = br#"{"type": "Snake", "data": null}"#;
        let deser = compile_decoder(AdjAnimal::SHAPE, &json::KajitJson);
        let result = deserialize::<AdjAnimal>(&deser, input);
        assert!(result.is_err());
        assert_eq!(result.unwrap_err().code, ErrorCode::UnknownVariant);
    }

    // r[verify deser.json.enum.adjacent.key-order]
    #[test]
    fn json_adjacent_wrong_first_key() {
        let input = br#"{"data": null, "type": "Cat"}"#;
        let deser = compile_decoder(AdjAnimal::SHAPE, &json::KajitJson);
        let result = deserialize::<AdjAnimal>(&deser, input);
        assert!(result.is_err());
        assert_eq!(result.unwrap_err().code, ErrorCode::ExpectedTagKey);
    }

    // --- Milestone 7: Internally tagged enums ---

    #[derive(Facet, Debug, PartialEq)]
    #[facet(tag = "type")]
    #[repr(u8)]
    enum IntAnimal {
        Cat,
        Dog { name: String, good_boy: bool },
    }

    // r[verify deser.json.enum.internal]
    // r[verify deser.json.enum.internal.unit-variant]
    #[test]
    fn json_internal_unit_variant() {
        let input = br#"{"type": "Cat"}"#;
        let deser = compile_decoder(IntAnimal::SHAPE, &json::KajitJson);
        let result: IntAnimal = deserialize(&deser, input).unwrap();
        assert_eq!(result, IntAnimal::Cat);
    }

    // r[verify deser.json.enum.internal]
    #[test]
    fn json_internal_struct_variant() {
        let input = br#"{"type": "Dog", "name": "Rex", "good_boy": true}"#;
        let deser = compile_decoder(IntAnimal::SHAPE, &json::KajitJson);
        let result: IntAnimal = deserialize(&deser, input).unwrap();
        assert_eq!(
            result,
            IntAnimal::Dog {
                name: "Rex".into(),
                good_boy: true,
            }
        );
    }

    // r[verify deser.json.enum.internal]
    #[test]
    fn json_internal_struct_variant_reversed_fields() {
        let input = br#"{"type": "Dog", "good_boy": true, "name": "Rex"}"#;
        let deser = compile_decoder(IntAnimal::SHAPE, &json::KajitJson);
        let result: IntAnimal = deserialize(&deser, input).unwrap();
        assert_eq!(
            result,
            IntAnimal::Dog {
                name: "Rex".into(),
                good_boy: true,
            }
        );
    }

    // r[verify deser.json.enum.internal]
    #[test]
    fn json_internal_unknown_variant() {
        let input = br#"{"type": "Snake"}"#;
        let deser = compile_decoder(IntAnimal::SHAPE, &json::KajitJson);
        let result = deserialize::<IntAnimal>(&deser, input);
        assert!(result.is_err());
        assert_eq!(result.unwrap_err().code, ErrorCode::UnknownVariant);
    }

    // r[verify deser.json.enum.internal]
    #[test]
    fn json_internal_wrong_first_key() {
        let input = br#"{"name": "Rex", "type": "Dog", "good_boy": true}"#;
        let deser = compile_decoder(IntAnimal::SHAPE, &json::KajitJson);
        let result = deserialize::<IntAnimal>(&deser, input);
        assert!(result.is_err());
        assert_eq!(result.unwrap_err().code, ErrorCode::ExpectedTagKey);
    }

    // r[verify deser.json.enum.internal.struct-only]
    #[test]
    #[should_panic(expected = "tuple variants")]
    fn json_internal_tuple_variant_panics() {
        #[derive(Facet)]
        #[facet(tag = "type")]
        #[repr(u8)]
        enum BadInternal {
            #[allow(dead_code)]
            Wrapper(String),
        }

        compile_decoder(BadInternal::SHAPE, &json::KajitJson);
    }

    // --- Milestone 8: Untagged enums ---

    #[derive(Facet, Debug, PartialEq)]
    #[facet(untagged)]
    #[repr(u8)]
    enum Untagged {
        Cat,
        Dog { name: String, good_boy: bool },
        Parrot(String),
    }

    // r[verify deser.json.enum.untagged]
    // r[verify deser.json.enum.untagged.string-trie]
    #[test]
    fn json_untagged_unit() {
        let input = br#""Cat""#;
        let deser = compile_decoder(Untagged::SHAPE, &json::KajitJson);
        let result: Untagged = deserialize(&deser, input).unwrap();
        assert_eq!(result, Untagged::Cat);
    }

    // r[verify deser.json.enum.untagged]
    // r[verify deser.json.enum.untagged.bucket]
    // r[verify deser.json.enum.untagged.peek]
    #[test]
    fn json_untagged_struct() {
        let input = br#"{"name": "Rex", "good_boy": true}"#;
        let deser = compile_decoder(Untagged::SHAPE, &json::KajitJson);
        let result: Untagged = deserialize(&deser, input).unwrap();
        assert_eq!(
            result,
            Untagged::Dog {
                name: "Rex".into(),
                good_boy: true,
            }
        );
    }

    // r[verify deser.json.enum.untagged]
    // r[verify deser.json.enum.untagged.string-trie]
    #[test]
    fn json_untagged_newtype_string() {
        let input = br#""Polly""#;
        let deser = compile_decoder(Untagged::SHAPE, &json::KajitJson);
        let result: Untagged = deserialize(&deser, input).unwrap();
        assert_eq!(result, Untagged::Parrot("Polly".into()));
    }

    // r[verify deser.json.enum.untagged]
    #[test]
    fn json_untagged_struct_reversed_keys() {
        let input = br#"{"good_boy": true, "name": "Rex"}"#;
        let deser = compile_decoder(Untagged::SHAPE, &json::KajitJson);
        let result: Untagged = deserialize(&deser, input).unwrap();
        assert_eq!(
            result,
            Untagged::Dog {
                name: "Rex".into(),
                good_boy: true,
            }
        );
    }

    // Multi-struct solver test
    #[derive(Facet, Debug, PartialEq)]
    #[facet(untagged)]
    #[repr(u8)]
    enum Config {
        Database { host: String, port: u32 },
        Redis { host: String, db: u32 },
    }

    // r[verify deser.json.enum.untagged.object-solver]
    #[test]
    fn json_untagged_solver_database() {
        let input = br#"{"host": "localhost", "port": 5432}"#;
        let deser = compile_decoder(Config::SHAPE, &json::KajitJson);
        let result: Config = deserialize(&deser, input).unwrap();
        assert_eq!(
            result,
            Config::Database {
                host: "localhost".into(),
                port: 5432,
            }
        );
    }

    // r[verify deser.json.enum.untagged.object-solver]
    #[test]
    fn json_untagged_solver_redis() {
        let input = br#"{"host": "localhost", "db": 0}"#;
        let deser = compile_decoder(Config::SHAPE, &json::KajitJson);
        let result: Config = deserialize(&deser, input).unwrap();
        assert_eq!(
            result,
            Config::Redis {
                host: "localhost".into(),
                db: 0,
            }
        );
    }

    // r[verify deser.json.enum.untagged.object-solver]
    #[test]
    fn json_untagged_solver_key_order_independent() {
        // Key order doesn't matter — "db" narrows to Redis regardless of position
        let input = br#"{"db": 0, "host": "localhost"}"#;
        let deser = compile_decoder(Config::SHAPE, &json::KajitJson);
        let result: Config = deserialize(&deser, input).unwrap();
        assert_eq!(
            result,
            Config::Redis {
                host: "localhost".into(),
                db: 0,
            }
        );
    }

    // r[verify deser.json.enum.untagged.scalar-unique]
    #[test]
    fn json_untagged_newtype_number() {
        #[derive(Facet, Debug, PartialEq)]
        #[facet(untagged)]
        #[repr(u8)]
        enum StringOrNum {
            Text(String),
            Num(u32),
        }

        let input = br#"42"#;
        let deser = compile_decoder(StringOrNum::SHAPE, &json::KajitJson);
        let result: StringOrNum = deserialize(&deser, input).unwrap();
        assert_eq!(result, StringOrNum::Num(42));

        let input = br#""hello""#;
        let deser = compile_decoder(StringOrNum::SHAPE, &json::KajitJson);
        let result: StringOrNum = deserialize(&deser, input).unwrap();
        assert_eq!(result, StringOrNum::Text("hello".into()));
    }

    // r[verify deser.json.enum.untagged.scalar-unique]
    #[test]
    fn json_untagged_newtype_bool() {
        #[derive(Facet, Debug, PartialEq)]
        #[facet(untagged)]
        #[repr(u8)]
        enum StringOrBool {
            Text(String),
            Flag(bool),
        }

        let input = br#"true"#;
        let deser = compile_decoder(StringOrBool::SHAPE, &json::KajitJson);
        let result: StringOrBool = deserialize(&deser, input).unwrap();
        assert_eq!(result, StringOrBool::Flag(true));

        let input = br#""hello""#;
        let deser = compile_decoder(StringOrBool::SHAPE, &json::KajitJson);
        let result: StringOrBool = deserialize(&deser, input).unwrap();
        assert_eq!(result, StringOrBool::Text("hello".into()));
    }

    // r[verify deser.json.enum.untagged.scalar-unique]
    #[test]
    #[should_panic(expected = "number variants")]
    fn json_untagged_ambiguous_number_panics() {
        #[derive(Facet)]
        #[facet(untagged)]
        #[repr(u8)]
        enum BadNum {
            #[allow(dead_code)]
            A(u32),
            #[allow(dead_code)]
            B(i64),
        }

        compile_decoder(BadNum::SHAPE, &json::KajitJson);
    }

    // r[verify deser.json.enum.untagged.value-type]
    #[test]
    fn json_untagged_value_type_number_vs_string() {
        #[derive(Facet, Debug, PartialEq)]
        #[facet(untagged)]
        #[repr(u8)]
        enum ValueTyped {
            NumField { value: u32 },
            StrField { value: String },
        }

        let deser = compile_decoder(ValueTyped::SHAPE, &json::KajitJson);

        let input = br#"{"value": 42}"#;
        let result: ValueTyped = deserialize(&deser, input).unwrap();
        assert_eq!(result, ValueTyped::NumField { value: 42 });

        let input = br#"{"value": "hello"}"#;
        let result: ValueTyped = deserialize(&deser, input).unwrap();
        assert_eq!(
            result,
            ValueTyped::StrField {
                value: "hello".into()
            }
        );
    }

    // r[verify deser.json.enum.untagged.value-type]
    #[test]
    fn json_untagged_value_type_bool_vs_number() {
        #[derive(Facet, Debug, PartialEq)]
        #[facet(untagged)]
        #[repr(u8)]
        enum BoolOrNum {
            Flag { active: bool },
            Count { active: u32 },
        }

        let deser = compile_decoder(BoolOrNum::SHAPE, &json::KajitJson);

        let input = br#"{"active": true}"#;
        let result: BoolOrNum = deserialize(&deser, input).unwrap();
        assert_eq!(result, BoolOrNum::Flag { active: true });

        let input = br#"{"active": 5}"#;
        let result: BoolOrNum = deserialize(&deser, input).unwrap();
        assert_eq!(result, BoolOrNum::Count { active: 5 });
    }

    // r[verify deser.json.enum.untagged.nested-key]
    #[test]
    fn json_untagged_nested_key_evidence() {
        #[derive(Facet, Debug, PartialEq)]
        struct SuccessPayload {
            items: u32,
        }

        #[derive(Facet, Debug, PartialEq)]
        struct ErrorPayload {
            message: String,
        }

        #[derive(Facet, Debug, PartialEq)]
        #[facet(untagged)]
        #[repr(u8)]
        enum ApiResponse {
            Success { status: u32, data: SuccessPayload },
            Error { status: u32, data: ErrorPayload },
        }

        let deser = compile_decoder(ApiResponse::SHAPE, &json::KajitJson);

        let input = br#"{"status": 200, "data": {"items": 5}}"#;
        let result: ApiResponse = deserialize(&deser, input).unwrap();
        assert_eq!(
            result,
            ApiResponse::Success {
                status: 200,
                data: SuccessPayload { items: 5 }
            }
        );

        let input = br#"{"status": 500, "data": {"message": "fail"}}"#;
        let result: ApiResponse = deserialize(&deser, input).unwrap();
        assert_eq!(
            result,
            ApiResponse::Error {
                status: 500,
                data: ErrorPayload {
                    message: "fail".into()
                }
            }
        );
    }

    // r[verify deser.json.enum.untagged.nested-key]
    #[test]
    fn json_untagged_nested_key_order_independent() {
        #[derive(Facet, Debug, PartialEq)]
        struct SuccessPayload {
            items: u32,
        }

        #[derive(Facet, Debug, PartialEq)]
        struct ErrorPayload {
            message: String,
        }

        #[derive(Facet, Debug, PartialEq)]
        #[facet(untagged)]
        #[repr(u8)]
        enum ApiResponse {
            Success { status: u32, data: SuccessPayload },
            Error { status: u32, data: ErrorPayload },
        }

        let deser = compile_decoder(ApiResponse::SHAPE, &json::KajitJson);

        // data before status — key order shouldn't matter
        let input = br#"{"data": {"items": 5}, "status": 200}"#;
        let result: ApiResponse = deserialize(&deser, input).unwrap();
        assert_eq!(
            result,
            ApiResponse::Success {
                status: 200,
                data: SuccessPayload { items: 5 }
            }
        );
    }

    // r[verify deser.json.enum.untagged.ambiguity-error]
    #[test]
    #[should_panic(expected = "indistinguishable")]
    fn json_untagged_truly_ambiguous_panics() {
        #[derive(Facet)]
        #[facet(untagged)]
        #[repr(u8)]
        enum TrulyAmbiguous {
            #[allow(dead_code)]
            A { x: u32 },
            #[allow(dead_code)]
            B { x: u32 },
        }

        compile_decoder(TrulyAmbiguous::SHAPE, &json::KajitJson);
    }

    // --- Option<T> support ---

    // r[verify deser.postcard.option]
    #[test]
    fn postcard_option_some_scalar() {
        use serde::Serialize;

        #[derive(Facet, Debug, PartialEq)]
        struct WithOptU32 {
            value: Option<u32>,
        }

        #[derive(Serialize)]
        struct WithOptU32Serde {
            value: Option<u32>,
        }

        let source = WithOptU32Serde { value: Some(42) };
        let encoded = ::postcard::to_allocvec(&source).unwrap();
        let deser = compile_decoder(WithOptU32::SHAPE, &postcard::KajitPostcard);
        let result: WithOptU32 = deserialize(&deser, &encoded).unwrap();
        assert_eq!(result, WithOptU32 { value: Some(42) });
    }

    // r[verify deser.postcard.option]
    #[test]
    fn postcard_option_none_scalar() {
        use serde::Serialize;

        #[derive(Facet, Debug, PartialEq)]
        struct WithOptU32 {
            value: Option<u32>,
        }

        #[derive(Serialize)]
        struct WithOptU32Serde {
            value: Option<u32>,
        }

        let source = WithOptU32Serde { value: None };
        let encoded = ::postcard::to_allocvec(&source).unwrap();
        let deser = compile_decoder(WithOptU32::SHAPE, &postcard::KajitPostcard);
        let result: WithOptU32 = deserialize(&deser, &encoded).unwrap();
        assert_eq!(result, WithOptU32 { value: None });
    }

    #[test]
    fn postcard_option_some_scalar_via_ir() {
        use serde::Serialize;

        #[derive(Facet, Debug, PartialEq)]
        struct WithOptU32 {
            value: Option<u32>,
        }

        #[derive(Serialize)]
        struct WithOptU32Serde {
            value: Option<u32>,
        }

        let source = WithOptU32Serde { value: Some(42) };
        let encoded = ::postcard::to_allocvec(&source).unwrap();
        let deser = compile_decoder_via_ir(WithOptU32::SHAPE, &postcard::KajitPostcard);
        let result: WithOptU32 = deserialize(&deser, &encoded).unwrap();
        assert_eq!(result, WithOptU32 { value: Some(42) });
    }

    #[test]
    fn postcard_option_none_scalar_via_ir() {
        use serde::Serialize;

        #[derive(Facet, Debug, PartialEq)]
        struct WithOptU32 {
            value: Option<u32>,
        }

        #[derive(Serialize)]
        struct WithOptU32Serde {
            value: Option<u32>,
        }

        let source = WithOptU32Serde { value: None };
        let encoded = ::postcard::to_allocvec(&source).unwrap();
        let deser = compile_decoder_via_ir(WithOptU32::SHAPE, &postcard::KajitPostcard);
        let result: WithOptU32 = deserialize(&deser, &encoded).unwrap();
        assert_eq!(result, WithOptU32 { value: None });
    }

    // r[verify deser.postcard.option]
    #[test]
    fn postcard_option_some_string() {
        use serde::Serialize;

        #[derive(Facet, Debug, PartialEq)]
        struct WithOptStr {
            name: Option<String>,
        }

        #[derive(Serialize)]
        struct WithOptStrSerde {
            name: Option<String>,
        }

        let source = WithOptStrSerde {
            name: Some("Alice".into()),
        };
        let encoded = ::postcard::to_allocvec(&source).unwrap();
        let deser = compile_decoder(WithOptStr::SHAPE, &postcard::KajitPostcard);
        let result: WithOptStr = deserialize(&deser, &encoded).unwrap();
        assert_eq!(
            result,
            WithOptStr {
                name: Some("Alice".into())
            }
        );
    }

    // r[verify deser.postcard.option]
    #[test]
    fn postcard_option_none_string() {
        use serde::Serialize;

        #[derive(Facet, Debug, PartialEq)]
        struct WithOptStr {
            name: Option<String>,
        }

        #[derive(Serialize)]
        struct WithOptStrSerde {
            name: Option<String>,
        }

        let source = WithOptStrSerde { name: None };
        let encoded = ::postcard::to_allocvec(&source).unwrap();
        let deser = compile_decoder(WithOptStr::SHAPE, &postcard::KajitPostcard);
        let result: WithOptStr = deserialize(&deser, &encoded).unwrap();
        assert_eq!(result, WithOptStr { name: None });
    }

    // r[verify deser.postcard.option]
    #[test]
    fn postcard_option_some_struct() {
        use serde::Serialize;

        #[derive(Facet, Debug, PartialEq)]
        struct WithOptAddr {
            addr: Option<Address>,
        }

        #[derive(Serialize)]
        struct AddressSerde {
            city: String,
            zip: u32,
        }

        #[derive(Serialize)]
        struct WithOptAddrSerde {
            addr: Option<AddressSerde>,
        }

        let source = WithOptAddrSerde {
            addr: Some(AddressSerde {
                city: "Portland".into(),
                zip: 97201,
            }),
        };
        let encoded = ::postcard::to_allocvec(&source).unwrap();
        let deser = compile_decoder(WithOptAddr::SHAPE, &postcard::KajitPostcard);
        let result: WithOptAddr = deserialize(&deser, &encoded).unwrap();
        assert_eq!(
            result,
            WithOptAddr {
                addr: Some(Address {
                    city: "Portland".into(),
                    zip: 97201
                }),
            }
        );
    }

    // r[verify deser.postcard.option]
    #[test]
    fn postcard_option_none_struct() {
        use serde::Serialize;

        #[derive(Facet, Debug, PartialEq)]
        struct WithOptAddr {
            addr: Option<Address>,
        }

        #[derive(Serialize)]
        struct AddressSerde {
            city: String,
            zip: u32,
        }

        #[derive(Serialize)]
        struct WithOptAddrSerde {
            addr: Option<AddressSerde>,
        }

        let source = WithOptAddrSerde { addr: None };
        let encoded = ::postcard::to_allocvec(&source).unwrap();
        let deser = compile_decoder(WithOptAddr::SHAPE, &postcard::KajitPostcard);
        let result: WithOptAddr = deserialize(&deser, &encoded).unwrap();
        assert_eq!(result, WithOptAddr { addr: None });
    }

    // r[verify deser.postcard.option]
    #[test]
    fn postcard_multiple_options() {
        use serde::Serialize;

        #[derive(Facet, Debug, PartialEq)]
        struct MultiOpt {
            a: Option<u32>,
            b: String,
            c: Option<String>,
        }

        #[derive(Serialize)]
        struct MultiOptSerde {
            a: Option<u32>,
            b: String,
            c: Option<String>,
        }

        let source = MultiOptSerde {
            a: Some(7),
            b: "hello".into(),
            c: None,
        };
        let encoded = ::postcard::to_allocvec(&source).unwrap();
        let deser = compile_decoder(MultiOpt::SHAPE, &postcard::KajitPostcard);
        let result: MultiOpt = deserialize(&deser, &encoded).unwrap();
        assert_eq!(
            result,
            MultiOpt {
                a: Some(7),
                b: "hello".into(),
                c: None,
            }
        );
    }

    // r[verify deser.json.option]
    #[test]
    fn json_option_some_scalar() {
        #[derive(Facet, Debug, PartialEq)]
        struct WithOptU32 {
            value: Option<u32>,
        }

        let input = br#"{"value": 42}"#;
        let deser = compile_decoder(WithOptU32::SHAPE, &json::KajitJson);
        let result: WithOptU32 = deserialize(&deser, input).unwrap();
        assert_eq!(result, WithOptU32 { value: Some(42) });
    }

    // r[verify deser.json.option]
    #[test]
    fn json_option_none_scalar() {
        #[derive(Facet, Debug, PartialEq)]
        struct WithOptU32 {
            value: Option<u32>,
        }

        let input = br#"{"value": null}"#;
        let deser = compile_decoder(WithOptU32::SHAPE, &json::KajitJson);
        let result: WithOptU32 = deserialize(&deser, input).unwrap();
        assert_eq!(result, WithOptU32 { value: None });
    }

    // r[verify deser.json.option]
    #[test]
    fn json_option_some_string() {
        #[derive(Facet, Debug, PartialEq)]
        struct WithOptStr {
            name: Option<String>,
        }

        let input = br#"{"name": "Alice"}"#;
        let deser = compile_decoder(WithOptStr::SHAPE, &json::KajitJson);
        let result: WithOptStr = deserialize(&deser, input).unwrap();
        assert_eq!(
            result,
            WithOptStr {
                name: Some("Alice".into())
            }
        );
    }

    // r[verify deser.json.option]
    #[test]
    fn json_option_none_string() {
        #[derive(Facet, Debug, PartialEq)]
        struct WithOptStr {
            name: Option<String>,
        }

        let input = br#"{"name": null}"#;
        let deser = compile_decoder(WithOptStr::SHAPE, &json::KajitJson);
        let result: WithOptStr = deserialize(&deser, input).unwrap();
        assert_eq!(result, WithOptStr { name: None });
    }

    // r[verify deser.json.option]
    #[test]
    fn json_option_some_struct() {
        #[derive(Facet, Debug, PartialEq)]
        struct WithOptAddr {
            addr: Option<Address>,
        }

        let input = br#"{"addr": {"city": "Portland", "zip": 97201}}"#;
        let deser = compile_decoder(WithOptAddr::SHAPE, &json::KajitJson);
        let result: WithOptAddr = deserialize(&deser, input).unwrap();
        assert_eq!(
            result,
            WithOptAddr {
                addr: Some(Address {
                    city: "Portland".into(),
                    zip: 97201
                }),
            }
        );
    }

    // r[verify deser.json.option]
    #[test]
    fn json_option_none_struct() {
        #[derive(Facet, Debug, PartialEq)]
        struct WithOptAddr {
            addr: Option<Address>,
        }

        let input = br#"{"addr": null}"#;
        let deser = compile_decoder(WithOptAddr::SHAPE, &json::KajitJson);
        let result: WithOptAddr = deserialize(&deser, input).unwrap();
        assert_eq!(result, WithOptAddr { addr: None });
    }

    // r[verify deser.json.option]
    #[test]
    fn json_multiple_options() {
        #[derive(Facet, Debug, PartialEq)]
        struct MultiOpt {
            a: Option<u32>,
            b: String,
            c: Option<String>,
        }

        let input = br#"{"a": 7, "b": "hello", "c": null}"#;
        let deser = compile_decoder(MultiOpt::SHAPE, &json::KajitJson);
        let result: MultiOpt = deserialize(&deser, input).unwrap();
        assert_eq!(
            result,
            MultiOpt {
                a: Some(7),
                b: "hello".into(),
                c: None,
            }
        );
    }

    // r[verify deser.json.option]
    #[test]
    fn json_option_reversed_keys() {
        #[derive(Facet, Debug, PartialEq)]
        struct MultiOpt {
            a: Option<u32>,
            b: String,
            c: Option<String>,
        }

        let input = br#"{"c": "world", "b": "hello", "a": null}"#;
        let deser = compile_decoder(MultiOpt::SHAPE, &json::KajitJson);
        let result: MultiOpt = deserialize(&deser, input).unwrap();
        assert_eq!(
            result,
            MultiOpt {
                a: None,
                b: "hello".into(),
                c: Some("world".into()),
            }
        );
    }

    // r[verify deser.postcard.seq]
    #[test]
    fn postcard_vec_u32() {
        #[derive(Facet, Debug, PartialEq)]
        struct Nums {
            vals: Vec<u32>,
        }

        #[derive(serde::Serialize)]
        struct NumsSerde {
            vals: Vec<u32>,
        }

        let source = NumsSerde {
            vals: vec![1, 2, 3],
        };
        let encoded = ::postcard::to_allocvec(&source).unwrap();
        let deser = compile_decoder(Nums::SHAPE, &postcard::KajitPostcard);
        let result: Nums = deserialize(&deser, &encoded).unwrap();
        assert_eq!(
            result,
            Nums {
                vals: vec![1, 2, 3]
            }
        );
    }

    #[test]
    fn postcard_vec_u32_via_ir() {
        #[derive(Facet, Debug, PartialEq)]
        struct Nums {
            vals: Vec<u32>,
        }

        #[derive(serde::Serialize)]
        struct NumsSerde {
            vals: Vec<u32>,
        }

        let source = NumsSerde {
            vals: vec![1, 2, 3],
        };
        let encoded = ::postcard::to_allocvec(&source).unwrap();
        let deser = compile_decoder_via_ir(Nums::SHAPE, &postcard::KajitPostcard);
        let result: Nums = deserialize(&deser, &encoded).unwrap();
        assert_eq!(
            result,
            Nums {
                vals: vec![1, 2, 3]
            }
        );
    }

    // r[verify ir.regalloc.regressions]
    #[test]
    fn postcard_vec_u32_medium_large_ir_matches_legacy_and_serde() {
        #[derive(Facet, Debug, PartialEq)]
        struct Nums {
            vals: Vec<u32>,
        }

        #[derive(serde::Serialize, serde::Deserialize, Debug, PartialEq)]
        struct NumsSerde {
            vals: Vec<u32>,
        }

        let legacy = compile_decoder_legacy(Nums::SHAPE, &postcard::KajitPostcard);
        let ir = compile_decoder_via_ir(Nums::SHAPE, &postcard::KajitPostcard);

        for len in [100usize, 10_000usize] {
            let source = NumsSerde {
                vals: (0..len as u32).collect(),
            };
            let encoded = ::postcard::to_allocvec(&source).unwrap();

            let serde_val: NumsSerde = ::postcard::from_bytes(&encoded).unwrap();
            let legacy_val: Nums = deserialize(&legacy, &encoded).unwrap();
            let ir_val: Nums = deserialize(&ir, &encoded).unwrap();

            assert_eq!(
                legacy_val.vals, serde_val.vals,
                "legacy mismatch at len={len}"
            );
            assert_eq!(ir_val.vals, serde_val.vals, "ir mismatch at len={len}");
            assert_eq!(ir_val, legacy_val, "legacy/ir mismatch at len={len}");
        }
    }

    // r[verify deser.postcard.seq]
    #[test]
    fn postcard_vec_empty() {
        #[derive(Facet, Debug, PartialEq)]
        struct Nums {
            vals: Vec<u32>,
        }

        #[derive(serde::Serialize)]
        struct NumsSerde {
            vals: Vec<u32>,
        }

        let source = NumsSerde { vals: vec![] };
        let encoded = ::postcard::to_allocvec(&source).unwrap();
        let deser = compile_decoder(Nums::SHAPE, &postcard::KajitPostcard);
        let result: Nums = deserialize(&deser, &encoded).unwrap();
        assert_eq!(result, Nums { vals: vec![] });
    }

    // r[verify deser.postcard.seq]
    #[test]
    fn postcard_vec_string() {
        #[derive(Facet, Debug, PartialEq)]
        struct Names {
            items: Vec<String>,
        }

        #[derive(serde::Serialize)]
        struct NamesSerde {
            items: Vec<String>,
        }

        let source = NamesSerde {
            items: vec!["hello".into(), "world".into()],
        };
        let encoded = ::postcard::to_allocvec(&source).unwrap();
        let deser = compile_decoder(Names::SHAPE, &postcard::KajitPostcard);
        let result: Names = deserialize(&deser, &encoded).unwrap();
        assert_eq!(
            result,
            Names {
                items: vec!["hello".into(), "world".into()],
            }
        );
    }

    #[test]
    fn postcard_vec_string_via_ir() {
        #[derive(Facet, Debug, PartialEq)]
        struct Names {
            items: Vec<String>,
        }

        #[derive(serde::Serialize)]
        struct NamesSerde {
            items: Vec<String>,
        }

        let source = NamesSerde {
            items: vec!["hello".into(), "world".into()],
        };
        let encoded = ::postcard::to_allocvec(&source).unwrap();
        let deser = compile_decoder_via_ir(Names::SHAPE, &postcard::KajitPostcard);
        let result: Names = deserialize(&deser, &encoded).unwrap();
        assert_eq!(
            result,
            Names {
                items: vec!["hello".into(), "world".into()],
            }
        );
    }

    // r[verify deser.postcard.seq]
    #[test]
    fn postcard_vec_nested_struct() {
        #[derive(Facet, Debug, PartialEq)]
        struct AddressList {
            addrs: Vec<Address>,
        }

        #[derive(serde::Serialize)]
        struct AddrSerde {
            city: String,
            zip: u32,
        }

        #[derive(serde::Serialize)]
        struct AddressListSerde {
            addrs: Vec<AddrSerde>,
        }

        let source = AddressListSerde {
            addrs: vec![
                AddrSerde {
                    city: "Portland".into(),
                    zip: 97201,
                },
                AddrSerde {
                    city: "Seattle".into(),
                    zip: 98101,
                },
            ],
        };
        let encoded = ::postcard::to_allocvec(&source).unwrap();
        let deser = compile_decoder(AddressList::SHAPE, &postcard::KajitPostcard);
        let result: AddressList = deserialize(&deser, &encoded).unwrap();
        assert_eq!(
            result,
            AddressList {
                addrs: vec![
                    Address {
                        city: "Portland".into(),
                        zip: 97201
                    },
                    Address {
                        city: "Seattle".into(),
                        zip: 98101
                    },
                ],
            }
        );
    }

    // r[verify compiler.recursive.one-func-per-shape]
    #[test]
    fn json_shared_inner_type() {
        #[derive(Facet, Debug, PartialEq)]
        struct TwoAddresses {
            home: Address,
            work: Address,
        }

        let input = br#"{"home": {"city": "Portland", "zip": 97201}, "work": {"city": "Seattle", "zip": 98101}}"#;
        let deser = compile_decoder(TwoAddresses::SHAPE, &json::KajitJson);
        let result: TwoAddresses = deserialize(&deser, input).unwrap();

        assert_eq!(
            result,
            TwoAddresses {
                home: Address {
                    city: "Portland".into(),
                    zip: 97201,
                },
                work: Address {
                    city: "Seattle".into(),
                    zip: 98101,
                },
            }
        );
    }

    // r[verify deser.json.seq]
    #[test]
    fn json_vec_u32() {
        #[derive(Facet, Debug, PartialEq)]
        struct Nums {
            vals: Vec<u32>,
        }

        let input = br#"{"vals": [1, 2, 3]}"#;
        let deser = compile_decoder(Nums::SHAPE, &json::KajitJson);
        let result: Nums = deserialize(&deser, input).unwrap();
        assert_eq!(
            result,
            Nums {
                vals: vec![1, 2, 3]
            }
        );
    }

    // r[verify deser.json.seq]
    #[test]
    fn json_vec_empty() {
        #[derive(Facet, Debug, PartialEq)]
        struct Nums {
            vals: Vec<u32>,
        }

        let input = br#"{"vals": []}"#;
        let deser = compile_decoder(Nums::SHAPE, &json::KajitJson);
        let result: Nums = deserialize(&deser, input).unwrap();
        assert_eq!(result, Nums { vals: vec![] });
    }

    // r[verify deser.json.seq]
    #[test]
    fn json_vec_string() {
        #[derive(Facet, Debug, PartialEq)]
        struct Names {
            items: Vec<String>,
        }

        let input = br#"{"items": ["hello", "world"]}"#;
        let deser = compile_decoder(Names::SHAPE, &json::KajitJson);
        let result: Names = deserialize(&deser, input).unwrap();
        assert_eq!(
            result,
            Names {
                items: vec!["hello".into(), "world".into()],
            }
        );
    }

    // r[verify deser.json.seq]
    #[test]
    fn json_vec_nested_struct() {
        #[derive(Facet, Debug, PartialEq)]
        struct AddressList {
            addrs: Vec<Address>,
        }

        let input = br#"{"addrs": [{"city": "Portland", "zip": 97201}, {"city": "Seattle", "zip": 98101}]}"#;
        let deser = compile_decoder(AddressList::SHAPE, &json::KajitJson);
        let result: AddressList = deserialize(&deser, input).unwrap();
        assert_eq!(
            result,
            AddressList {
                addrs: vec![
                    Address {
                        city: "Portland".into(),
                        zip: 97201
                    },
                    Address {
                        city: "Seattle".into(),
                        zip: 98101
                    },
                ],
            }
        );
    }

    // r[verify seq.malum.json]
    #[test]
    fn json_vec_growth() {
        // More than 4 elements exercises the growth path (initial cap=4)
        #[derive(Facet, Debug, PartialEq)]
        struct Nums {
            vals: Vec<u32>,
        }

        let input = br#"{"vals": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]}"#;
        let deser = compile_decoder(Nums::SHAPE, &json::KajitJson);
        let result: Nums = deserialize(&deser, input).unwrap();
        assert_eq!(
            result,
            Nums {
                vals: vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
            }
        );
    }

    // ── JSON string escape sequence tests ────────────────────────────────

    // r[verify deser.json.string.escape]

    #[test]
    fn json_string_escape_newline() {
        let input = br#"{"age": 1, "name": "hello\nworld"}"#;
        let deser = compile_decoder(Friend::SHAPE, &json::KajitJson);
        let result: Friend = deserialize(&deser, input).unwrap();
        assert_eq!(result.name, "hello\nworld");
    }

    #[test]
    fn json_string_escape_tab() {
        let input = br#"{"age": 1, "name": "hello\tworld"}"#;
        let deser = compile_decoder(Friend::SHAPE, &json::KajitJson);
        let result: Friend = deserialize(&deser, input).unwrap();
        assert_eq!(result.name, "hello\tworld");
    }

    #[test]
    fn json_string_escape_backslash() {
        let input = br#"{"age": 1, "name": "hello\\world"}"#;
        let deser = compile_decoder(Friend::SHAPE, &json::KajitJson);
        let result: Friend = deserialize(&deser, input).unwrap();
        assert_eq!(result.name, "hello\\world");
    }

    #[test]
    fn json_string_escape_quote() {
        let input = br#"{"age": 1, "name": "hello\"world"}"#;
        let deser = compile_decoder(Friend::SHAPE, &json::KajitJson);
        let result: Friend = deserialize(&deser, input).unwrap();
        assert_eq!(result.name, "hello\"world");
    }

    #[test]
    fn json_string_escape_all_simple() {
        let input = br#"{"age": 1, "name": "a\"b\\c\/d\be\ff\ng\rh\ti"}"#;
        let deser = compile_decoder(Friend::SHAPE, &json::KajitJson);
        let result: Friend = deserialize(&deser, input).unwrap();
        assert_eq!(result.name, "a\"b\\c/d\x08e\x0Cf\ng\rh\ti");
    }

    #[test]
    fn json_string_unicode_escape_bmp() {
        // \u0041 = 'A'
        let input = br#"{"age": 1, "name": "\u0041lice"}"#;
        let deser = compile_decoder(Friend::SHAPE, &json::KajitJson);
        let result: Friend = deserialize(&deser, input).unwrap();
        assert_eq!(result.name, "Alice");
    }

    #[test]
    fn json_string_unicode_escape_non_ascii() {
        // \u00E9 = 'e' with acute accent
        let input = br#"{"age": 1, "name": "caf\u00E9"}"#;
        let deser = compile_decoder(Friend::SHAPE, &json::KajitJson);
        let result: Friend = deserialize(&deser, input).unwrap();
        assert_eq!(result.name, "caf\u{00E9}");
    }

    #[test]
    fn json_string_unicode_surrogate_pair() {
        // \uD83D\uDE00 = U+1F600 (grinning face emoji)
        let input = br#"{"age": 1, "name": "\uD83D\uDE00"}"#;
        let deser = compile_decoder(Friend::SHAPE, &json::KajitJson);
        let result: Friend = deserialize(&deser, input).unwrap();
        assert_eq!(result.name, "\u{1F600}");
    }

    #[test]
    fn json_key_with_unicode_escape() {
        // "na\u006De" unescapes to "name"
        let input = br#"{"age": 42, "na\u006De": "Alice"}"#;
        let deser = compile_decoder(Friend::SHAPE, &json::KajitJson);
        let result: Friend = deserialize(&deser, input).unwrap();
        assert_eq!(
            result,
            Friend {
                age: 42,
                name: "Alice".into()
            }
        );
    }

    #[test]
    fn json_string_invalid_escape() {
        let input = br#"{"age": 1, "name": "hello\xworld"}"#;
        let deser = compile_decoder(Friend::SHAPE, &json::KajitJson);
        let result = deserialize::<Friend>(&deser, input);
        assert!(result.is_err());
        assert_eq!(result.unwrap_err().code, ErrorCode::InvalidEscapeSequence);
    }

    #[test]
    fn json_string_lone_high_surrogate() {
        let input = br#"{"age": 1, "name": "\uD800"}"#;
        let deser = compile_decoder(Friend::SHAPE, &json::KajitJson);
        let result = deserialize::<Friend>(&deser, input);
        assert!(result.is_err());
        assert_eq!(result.unwrap_err().code, ErrorCode::InvalidEscapeSequence);
    }

    #[test]
    fn json_string_truncated_unicode() {
        let input = br#"{"age": 1, "name": "\u00"}"#;
        let deser = compile_decoder(Friend::SHAPE, &json::KajitJson);
        let result = deserialize::<Friend>(&deser, input);
        assert!(result.is_err());
    }

    #[test]
    fn json_skip_value_with_unicode_escape() {
        // Unknown field "extra" has a string with \uXXXX — should be skipped correctly
        let input = br#"{"age": 42, "extra": "test\uD83D\uDE00end", "name": "Alice"}"#;
        let deser = compile_decoder(Friend::SHAPE, &json::KajitJson);
        let result: Friend = deserialize(&deser, input).unwrap();
        assert_eq!(
            result,
            Friend {
                age: 42,
                name: "Alice".into()
            }
        );
    }

    #[test]
    fn json_skip_value_with_backslash_escape() {
        // Unknown field with simple escapes in its value
        let input = br#"{"age": 42, "extra": "test\n\t\\end", "name": "Alice"}"#;
        let deser = compile_decoder(Friend::SHAPE, &json::KajitJson);
        let result: Friend = deserialize(&deser, input).unwrap();
        assert_eq!(
            result,
            Friend {
                age: 42,
                name: "Alice".into()
            }
        );
    }

    // --- Milestone 8: Map deserialization ---

    // r[verify deser.postcard.map]
    #[test]
    fn postcard_map_string_to_u32() {
        use std::collections::HashMap;

        #[derive(facet::Facet, Debug, PartialEq)]
        struct Config {
            scores: HashMap<String, u32>,
        }

        #[derive(serde::Serialize)]
        struct ConfigSerde {
            scores: HashMap<String, u32>,
        }

        let mut scores = HashMap::new();
        scores.insert("alice".to_string(), 42u32);
        scores.insert("bob".to_string(), 7u32);
        let source = ConfigSerde {
            scores: scores.clone(),
        };
        let encoded = ::postcard::to_allocvec(&source).unwrap();
        let deser = compile_decoder(Config::SHAPE, &postcard::KajitPostcard);
        let result: Config = deserialize(&deser, &encoded).unwrap();
        assert_eq!(result, Config { scores });
    }

    // r[verify deser.postcard.map]
    #[test]
    fn postcard_map_empty() {
        use std::collections::HashMap;

        #[derive(facet::Facet, Debug, PartialEq)]
        struct Config {
            scores: HashMap<String, u32>,
        }

        #[derive(serde::Serialize)]
        struct ConfigSerde {
            scores: HashMap<String, u32>,
        }

        let source = ConfigSerde {
            scores: HashMap::new(),
        };
        let encoded = ::postcard::to_allocvec(&source).unwrap();
        let deser = compile_decoder(Config::SHAPE, &postcard::KajitPostcard);
        let result: Config = deserialize(&deser, &encoded).unwrap();
        assert_eq!(
            result,
            Config {
                scores: HashMap::new()
            }
        );
    }

    // r[verify deser.postcard.map]
    #[test]
    fn postcard_map_string_to_string() {
        use std::collections::HashMap;

        #[derive(facet::Facet, Debug, PartialEq)]
        struct Env {
            vars: HashMap<String, String>,
        }

        #[derive(serde::Serialize)]
        struct EnvSerde {
            vars: HashMap<String, String>,
        }

        let mut vars = HashMap::new();
        vars.insert("HOME".to_string(), "/root".to_string());
        vars.insert("PATH".to_string(), "/usr/bin".to_string());
        let source = EnvSerde { vars: vars.clone() };
        let encoded = ::postcard::to_allocvec(&source).unwrap();
        let deser = compile_decoder(Env::SHAPE, &postcard::KajitPostcard);
        let result: Env = deserialize(&deser, &encoded).unwrap();
        assert_eq!(result, Env { vars });
    }

    // r[verify deser.postcard.map]
    #[test]
    fn postcard_btreemap_string_to_u32() {
        use std::collections::BTreeMap;

        #[derive(facet::Facet, Debug, PartialEq)]
        struct Config {
            scores: BTreeMap<String, u32>,
        }

        #[derive(serde::Serialize)]
        struct ConfigSerde {
            scores: BTreeMap<String, u32>,
        }

        let mut scores = BTreeMap::new();
        scores.insert("alice".to_string(), 42u32);
        scores.insert("bob".to_string(), 7u32);
        let source = ConfigSerde {
            scores: scores.clone(),
        };
        let encoded = ::postcard::to_allocvec(&source).unwrap();
        let deser = compile_decoder(Config::SHAPE, &postcard::KajitPostcard);
        let result: Config = deserialize(&deser, &encoded).unwrap();
        assert_eq!(result, Config { scores });
    }

    // r[verify deser.json.map]
    #[test]
    fn json_map_string_to_u32() {
        use std::collections::HashMap;

        #[derive(facet::Facet, Debug, PartialEq)]
        struct Config {
            scores: HashMap<String, u32>,
        }

        let input = br#"{"scores": {"alice": 42, "bob": 7}}"#;
        let deser = compile_decoder(Config::SHAPE, &json::KajitJson);
        let result: Config = deserialize(&deser, input).unwrap();
        let mut expected = HashMap::new();
        expected.insert("alice".to_string(), 42u32);
        expected.insert("bob".to_string(), 7u32);
        assert_eq!(result, Config { scores: expected });
    }

    // r[verify deser.json.map]
    #[test]
    fn json_map_empty() {
        use std::collections::HashMap;

        #[derive(facet::Facet, Debug, PartialEq)]
        struct Config {
            scores: HashMap<String, u32>,
        }

        let input = br#"{"scores": {}}"#;
        let deser = compile_decoder(Config::SHAPE, &json::KajitJson);
        let result: Config = deserialize(&deser, input).unwrap();
        assert_eq!(
            result,
            Config {
                scores: HashMap::new()
            }
        );
    }

    // r[verify deser.json.map]
    #[test]
    fn json_map_string_to_string() {
        use std::collections::HashMap;

        #[derive(facet::Facet, Debug, PartialEq)]
        struct Env {
            vars: HashMap<String, String>,
        }

        let input = br#"{"vars": {"HOME": "/root", "PATH": "/usr/bin"}}"#;
        let deser = compile_decoder(Env::SHAPE, &json::KajitJson);
        let result: Env = deserialize(&deser, input).unwrap();
        let mut expected = HashMap::new();
        expected.insert("HOME".to_string(), "/root".to_string());
        expected.insert("PATH".to_string(), "/usr/bin".to_string());
        assert_eq!(result, Env { vars: expected });
    }

    // r[verify deser.json.map]
    #[test]
    fn json_map_growth() {
        use std::collections::HashMap;

        #[derive(facet::Facet, Debug, PartialEq)]
        struct Big {
            data: HashMap<String, u32>,
        }

        // More than initial cap=4 to trigger growth
        let input = br#"{"data": {"a": 1, "b": 2, "c": 3, "d": 4, "e": 5, "f": 6}}"#;
        let deser = compile_decoder(Big::SHAPE, &json::KajitJson);
        let result: Big = deserialize(&deser, input).unwrap();
        let mut expected = HashMap::new();
        for (k, v) in [("a", 1), ("b", 2), ("c", 3), ("d", 4), ("e", 5), ("f", 6)] {
            expected.insert(k.to_string(), v as u32);
        }
        assert_eq!(result, Big { data: expected });
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

    // r[verify deser.rename]
    // r[verify deser.rename.json]
    #[test]
    fn json_rename_field() {
        let input = br#"{"user_name": "Alice", "age": 30}"#;
        let deser = compile_decoder(RenameField::SHAPE, &json::KajitJson);
        let result: RenameField = deserialize(&deser, input).unwrap();
        assert_eq!(
            result,
            RenameField {
                name: "Alice".into(),
                age: 30,
            }
        );
    }

    // r[verify deser.rename]
    // r[verify deser.rename.json]
    #[test]
    fn json_rename_field_original_name_rejected() {
        // Using the original Rust field name "name" should fail
        // because the key dispatch only knows about "user_name".
        let input = br#"{"name": "Alice", "age": 30}"#;
        let deser = compile_decoder(RenameField::SHAPE, &json::KajitJson);
        let result = deserialize::<RenameField>(&deser, input);
        assert!(result.is_err(), "original field name should not match");
    }

    #[derive(Facet, Debug, PartialEq)]
    #[facet(rename_all = "camelCase")]
    struct CamelCaseStruct {
        user_name: String,
        birth_year: u32,
    }

    // r[verify deser.rename.all]
    // r[verify deser.rename.json]
    #[test]
    fn json_rename_all_camel_case() {
        let input = br#"{"userName": "Bob", "birthYear": 1990}"#;
        let deser = compile_decoder(CamelCaseStruct::SHAPE, &json::KajitJson);
        let result: CamelCaseStruct = deserialize(&deser, input).unwrap();
        assert_eq!(
            result,
            CamelCaseStruct {
                user_name: "Bob".into(),
                birth_year: 1990,
            }
        );
    }

    // r[verify deser.rename.postcard-irrelevant]
    #[test]
    fn postcard_rename_ignored() {
        // Postcard is positional — rename has no effect.
        // Fields are deserialized in declaration order regardless of name.
        let wire = ::postcard::to_allocvec(&("Alice".to_string(), 30u32)).unwrap();
        let deser = compile_decoder(RenameField::SHAPE, &postcard::KajitPostcard);
        let result: RenameField = deserialize(&deser, &wire).unwrap();
        assert_eq!(
            result,
            RenameField {
                name: "Alice".into(),
                age: 30,
            }
        );
    }

    // ═══════════════════════════════════════════════════════════════════
    // #23: transparent
    // ═══════════════════════════════════════════════════════════════════

    #[derive(Facet, Debug, PartialEq)]
    #[facet(transparent)]
    struct Wrapper(u32);

    // r[verify deser.transparent]
    // r[verify deser.transparent.forwarding]
    #[test]
    fn json_transparent_scalar() {
        let input = b"42";
        let deser = compile_decoder(Wrapper::SHAPE, &json::KajitJson);
        let result: Wrapper = deserialize(&deser, input).unwrap();
        assert_eq!(result, Wrapper(42));
    }

    // r[verify deser.transparent]
    // r[verify deser.transparent.forwarding]
    #[test]
    fn postcard_transparent_scalar() {
        let wire = ::postcard::to_allocvec(&42u32).unwrap();
        let deser = compile_decoder(Wrapper::SHAPE, &postcard::KajitPostcard);
        let result: Wrapper = deserialize(&deser, &wire).unwrap();
        assert_eq!(result, Wrapper(42));
    }

    #[derive(Facet, Debug, PartialEq)]
    #[facet(transparent)]
    struct StringWrapper(String);

    // r[verify deser.transparent]
    #[test]
    fn json_transparent_string() {
        let input = br#""hello""#;
        let deser = compile_decoder(StringWrapper::SHAPE, &json::KajitJson);
        let result: StringWrapper = deserialize(&deser, input).unwrap();
        assert_eq!(result, StringWrapper("hello".into()));
    }

    // r[verify deser.transparent]
    #[test]
    fn postcard_transparent_string() {
        let wire = ::postcard::to_allocvec(&"hello".to_string()).unwrap();
        let deser = compile_decoder(StringWrapper::SHAPE, &postcard::KajitPostcard);
        let result: StringWrapper = deserialize(&deser, &wire).unwrap();
        assert_eq!(result, StringWrapper("hello".into()));
    }

    #[derive(Facet, Debug, PartialEq)]
    #[facet(transparent)]
    struct StructWrapper(Friend);

    // r[verify deser.transparent]
    // r[verify deser.transparent.composite]
    #[test]
    fn json_transparent_composite() {
        let input = br#"{"age": 25, "name": "Eve"}"#;
        let deser = compile_decoder(StructWrapper::SHAPE, &json::KajitJson);
        let result: StructWrapper = deserialize(&deser, input).unwrap();
        assert_eq!(
            result,
            StructWrapper(Friend {
                age: 25,
                name: "Eve".into()
            })
        );
    }

    // r[verify deser.transparent]
    // r[verify deser.transparent.composite]
    #[test]
    fn postcard_transparent_composite() {
        let wire = ::postcard::to_allocvec(&(25u32, "Eve".to_string())).unwrap();
        let deser = compile_decoder(StructWrapper::SHAPE, &postcard::KajitPostcard);
        let result: StructWrapper = deserialize(&deser, &wire).unwrap();
        assert_eq!(
            result,
            StructWrapper(Friend {
                age: 25,
                name: "Eve".into()
            })
        );
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

    // r[verify deser.deny-unknown-fields]
    // r[verify deser.deny-unknown-fields.json]
    #[test]
    fn json_deny_unknown_fields_rejects() {
        let input = br#"{"x": 1, "y": 2, "z": 3}"#;
        let deser = compile_decoder(Strict::SHAPE, &json::KajitJson);
        let result = deserialize::<Strict>(&deser, input);
        assert!(result.is_err());
        assert_eq!(result.unwrap_err().code, ErrorCode::UnknownField);
    }

    // r[verify deser.deny-unknown-fields]
    // r[verify deser.deny-unknown-fields.json]
    #[test]
    fn json_deny_unknown_fields_allows_known() {
        let input = br#"{"x": 1, "y": 2}"#;
        let deser = compile_decoder(Strict::SHAPE, &json::KajitJson);
        let result: Strict = deserialize(&deser, input).unwrap();
        assert_eq!(result, Strict { x: 1, y: 2 });
    }

    // r[verify deser.deny-unknown-fields.postcard-irrelevant]
    #[test]
    fn postcard_deny_unknown_fields_irrelevant() {
        // Postcard is positional — deny_unknown_fields has no effect.
        let wire = ::postcard::to_allocvec(&(10u32, 20u32)).unwrap();
        let deser = compile_decoder(Strict::SHAPE, &postcard::KajitPostcard);
        let result: Strict = deserialize(&deser, &wire).unwrap();
        assert_eq!(result, Strict { x: 10, y: 20 });
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

    // r[verify deser.default]
    #[test]
    fn json_default_field_missing() {
        // "score" is missing — should get its Default (0).
        let input = br#"{"name": "Alice"}"#;
        let deser = compile_decoder(WithDefault::SHAPE, &json::KajitJson);
        let result: WithDefault = deserialize(&deser, input).unwrap();
        assert_eq!(
            result,
            WithDefault {
                name: "Alice".into(),
                score: 0,
            }
        );
    }

    // r[verify deser.default]
    #[test]
    fn json_default_field_present() {
        // "score" is present — use the provided value.
        let input = br#"{"name": "Alice", "score": 99}"#;
        let deser = compile_decoder(WithDefault::SHAPE, &json::KajitJson);
        let result: WithDefault = deserialize(&deser, input).unwrap();
        assert_eq!(
            result,
            WithDefault {
                name: "Alice".into(),
                score: 99,
            }
        );
    }

    // r[verify deser.default]
    #[test]
    fn json_default_field_required_still_errors() {
        // "name" has no default — missing it should error.
        let input = br#"{"score": 50}"#;
        let deser = compile_decoder(WithDefault::SHAPE, &json::KajitJson);
        let result = deserialize::<WithDefault>(&deser, input);
        assert!(result.is_err());
        assert_eq!(result.unwrap_err().code, ErrorCode::MissingRequiredField);
    }

    #[derive(Facet, Debug, PartialEq)]
    struct WithDefaultString {
        #[facet(default)]
        label: String,
        value: u32,
    }

    // r[verify deser.default]
    #[test]
    fn json_default_string_field() {
        let input = br#"{"value": 42}"#;
        let deser = compile_decoder(WithDefaultString::SHAPE, &json::KajitJson);
        let result: WithDefaultString = deserialize(&deser, input).unwrap();
        assert_eq!(
            result,
            WithDefaultString {
                label: String::new(),
                value: 42,
            }
        );
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

    // r[verify deser.default]
    #[test]
    fn json_container_default_empty_object() {
        // Container-level #[facet(default)] — all fields optional.
        let input = br#"{}"#;
        let deser = compile_decoder(AllDefault::SHAPE, &json::KajitJson);
        let result: AllDefault = deserialize(&deser, input).unwrap();
        assert_eq!(result, AllDefault { x: 0, y: 0 });
    }

    // r[verify deser.default]
    #[test]
    fn json_container_default_partial() {
        let input = br#"{"x": 5}"#;
        let deser = compile_decoder(AllDefault::SHAPE, &json::KajitJson);
        let result: AllDefault = deserialize(&deser, input).unwrap();
        assert_eq!(result, AllDefault { x: 5, y: 0 });
    }

    // r[verify deser.default.postcard-irrelevant]
    #[test]
    fn postcard_default_irrelevant() {
        // Postcard is positional — all fields are always present, defaults don't apply.
        let wire = ::postcard::to_allocvec(&("hello".to_string(), 7u32)).unwrap();
        let deser = compile_decoder(WithDefault::SHAPE, &postcard::KajitPostcard);
        let result: WithDefault = deserialize(&deser, &wire).unwrap();
        assert_eq!(
            result,
            WithDefault {
                name: "hello".into(),
                score: 7,
            }
        );
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

    // r[verify deser.skip]
    // r[verify deser.skip.json]
    #[test]
    fn json_skip_field() {
        // "cached" is skipped — it should NOT appear in input and gets its default (0).
        let input = br#"{"name": "Alice"}"#;
        let deser = compile_decoder(WithSkip::SHAPE, &json::KajitJson);
        let result: WithSkip = deserialize(&deser, input).unwrap();
        assert_eq!(
            result,
            WithSkip {
                name: "Alice".into(),
                cached: 0,
            }
        );
    }

    // r[verify deser.skip]
    // r[verify deser.skip.json]
    #[test]
    fn json_skip_field_in_input_treated_as_unknown() {
        // If the skipped field's name appears in input, it's treated as unknown.
        // Without deny_unknown_fields, it's silently skipped.
        let input = br#"{"name": "Alice", "cached": 99}"#;
        let deser = compile_decoder(WithSkip::SHAPE, &json::KajitJson);
        let result: WithSkip = deserialize(&deser, input).unwrap();
        assert_eq!(
            result,
            WithSkip {
                name: "Alice".into(),
                cached: 0, // default, NOT 99
            }
        );
    }

    #[derive(Facet, Debug, PartialEq)]
    struct WithSkipDeser {
        name: String,
        #[facet(skip_deserializing, default)]
        internal: u32,
    }

    // r[verify deser.skip]
    // r[verify deser.skip.json]
    #[test]
    fn json_skip_deserializing_field() {
        let input = br#"{"name": "Bob"}"#;
        let deser = compile_decoder(WithSkipDeser::SHAPE, &json::KajitJson);
        let result: WithSkipDeser = deserialize(&deser, input).unwrap();
        assert_eq!(
            result,
            WithSkipDeser {
                name: "Bob".into(),
                internal: 0,
            }
        );
    }

    // r[verify deser.skip]
    // r[verify deser.skip.postcard]
    #[test]
    fn postcard_skip_field() {
        // Postcard: skipped field is NOT on the wire. Only "name" is serialized.
        let wire = ::postcard::to_allocvec(&("Alice".to_string(),)).unwrap();
        let deser = compile_decoder(WithSkip::SHAPE, &postcard::KajitPostcard);
        let result: WithSkip = deserialize(&deser, &wire).unwrap();
        assert_eq!(
            result,
            WithSkip {
                name: "Alice".into(),
                cached: 0,
            }
        );
    }

    #[derive(Facet, Debug, PartialEq)]
    struct SkipWithCustomDefault {
        value: u32,
        #[facet(skip, default = 42)]
        magic: u32,
    }

    // r[verify deser.skip]
    // r[verify deser.skip.default-required]
    #[test]
    fn json_skip_with_custom_default() {
        let input = br#"{"value": 10}"#;
        let deser = compile_decoder(SkipWithCustomDefault::SHAPE, &json::KajitJson);
        let result: SkipWithCustomDefault = deserialize(&deser, input).unwrap();
        assert_eq!(
            result,
            SkipWithCustomDefault {
                value: 10,
                magic: 42,
            }
        );
    }
    // ── Smart pointer tests ──────────────────────────────────────────

    #[derive(Facet, Debug, PartialEq)]
    struct BoxedScalar {
        value: Box<u32>,
    }

    // r[verify deser.pointer]
    // r[verify deser.pointer.scratch]
    // r[verify deser.pointer.new-into]
    // r[verify deser.pointer.format-transparent]
    #[test]
    fn json_box_scalar() {
        let input = br#"{"value": 42}"#;
        let deser = compile_decoder(BoxedScalar::SHAPE, &json::KajitJson);
        let result: BoxedScalar = deserialize(&deser, input).unwrap();
        assert_eq!(
            result,
            BoxedScalar {
                value: Box::new(42)
            }
        );
    }

    // r[verify deser.pointer]
    // r[verify deser.pointer.format-transparent]
    #[test]
    fn postcard_box_scalar() {
        // Box<u32> is wire-transparent, so the postcard encoding is identical
        // to a struct with a bare u32 field: just a varint.
        #[derive(serde::Serialize)]
        struct Ref {
            value: u32,
        }
        let input = ::postcard::to_allocvec(&Ref { value: 42 }).unwrap();
        let deser = compile_decoder(BoxedScalar::SHAPE, &postcard::KajitPostcard);
        let result: BoxedScalar = deserialize(&deser, &input).unwrap();
        assert_eq!(
            result,
            BoxedScalar {
                value: Box::new(42)
            }
        );
    }

    #[derive(Facet, Debug, PartialEq)]
    struct BoxedString {
        #[allow(clippy::box_collection)]
        name: Box<String>,
    }

    // r[verify deser.pointer]
    #[test]
    fn json_box_string() {
        let input = br#"{"name": "hello"}"#;
        let deser = compile_decoder(BoxedString::SHAPE, &json::KajitJson);
        let result: BoxedString = deserialize(&deser, input).unwrap();
        assert_eq!(
            result,
            BoxedString {
                name: Box::new("hello".to_string()),
            }
        );
    }

    #[derive(Facet, Debug, PartialEq)]
    struct BoxedNested {
        inner: Box<Friend>,
    }

    // r[verify deser.pointer]
    #[test]
    fn json_box_nested_struct() {
        let input = br#"{"inner": {"age": 30, "name": "Bob"}}"#;
        let deser = compile_decoder(BoxedNested::SHAPE, &json::KajitJson);
        let result: BoxedNested = deserialize(&deser, input).unwrap();
        assert_eq!(
            result,
            BoxedNested {
                inner: Box::new(Friend {
                    age: 30,
                    name: "Bob".to_string(),
                }),
            }
        );
    }

    // r[verify deser.pointer]
    #[test]
    fn postcard_box_nested_struct() {
        // Box<Friend> is wire-transparent — same encoding as a bare Friend.
        #[derive(serde::Serialize)]
        struct Ref {
            inner: RefInner,
        }
        #[derive(serde::Serialize)]
        struct RefInner {
            age: u32,
            name: String,
        }
        let input = ::postcard::to_allocvec(&Ref {
            inner: RefInner {
                age: 30,
                name: "Bob".to_string(),
            },
        })
        .unwrap();
        let deser = compile_decoder(BoxedNested::SHAPE, &postcard::KajitPostcard);
        let result: BoxedNested = deserialize(&deser, &input).unwrap();
        assert_eq!(
            result,
            BoxedNested {
                inner: Box::new(Friend {
                    age: 30,
                    name: "Bob".to_string(),
                }),
            }
        );
    }

    // r[verify deser.pointer.nesting]
    #[derive(Facet, Debug, PartialEq)]
    struct OptionBox {
        value: Option<Box<u32>>,
    }

    #[test]
    fn json_option_box_some() {
        let input = br#"{"value": 7}"#;
        let deser = compile_decoder(OptionBox::SHAPE, &json::KajitJson);
        let result: OptionBox = deserialize(&deser, input).unwrap();
        assert_eq!(
            result,
            OptionBox {
                value: Some(Box::new(7)),
            }
        );
    }

    #[test]
    fn json_option_box_none() {
        let input = br#"{"value": null}"#;
        let deser = compile_decoder(OptionBox::SHAPE, &json::KajitJson);
        let result: OptionBox = deserialize(&deser, input).unwrap();
        assert_eq!(result, OptionBox { value: None });
    }

    #[derive(Facet, Debug, PartialEq)]
    struct VecBox {
        #[allow(clippy::vec_box)]
        items: Vec<Box<u32>>,
    }

    // r[verify deser.pointer.nesting]
    #[test]
    fn json_vec_box() {
        let input = br#"{"items": [1, 2, 3]}"#;
        let deser = compile_decoder(VecBox::SHAPE, &json::KajitJson);
        let result: VecBox = deserialize(&deser, input).unwrap();
        assert_eq!(
            result,
            VecBox {
                items: vec![Box::new(1), Box::new(2), Box::new(3)],
            }
        );
    }

    #[derive(Facet, Debug, PartialEq)]
    struct ArcScalar {
        value: std::sync::Arc<u32>,
    }

    // r[verify deser.pointer]
    #[test]
    fn json_arc_scalar() {
        let input = br#"{"value": 99}"#;
        let deser = compile_decoder(ArcScalar::SHAPE, &json::KajitJson);
        let result: ArcScalar = deserialize(&deser, input).unwrap();
        assert_eq!(
            result,
            ArcScalar {
                value: std::sync::Arc::new(99),
            }
        );
    }

    #[derive(Facet, Debug, PartialEq)]
    struct RcScalar {
        value: std::rc::Rc<u32>,
    }

    // r[verify deser.pointer]
    #[test]
    fn json_rc_scalar() {
        let input = br#"{"value": 77}"#;
        let deser = compile_decoder(RcScalar::SHAPE, &json::KajitJson);
        let result: RcScalar = deserialize(&deser, input).unwrap();
        assert_eq!(
            result,
            RcScalar {
                value: std::rc::Rc::new(77),
            }
        );
    }
    #[derive(Facet, Debug, PartialEq)]
    struct UnitField {
        geo: (),
        name: String,
    }

    #[test]
    fn json_unit_field() {
        let input = br#"{"geo": null, "name": "test"}"#;
        let deser = compile_decoder(UnitField::SHAPE, &json::KajitJson);
        let result: UnitField = deserialize(&deser, input).unwrap();
        assert_eq!(
            result,
            UnitField {
                geo: (),
                name: "test".into(),
            }
        );
    }

    // --- Encode tests ---

    #[test]
    fn postcard_encode_simple_struct() {
        use serde::Serialize;

        #[derive(Serialize)]
        struct FriendSerde {
            age: u32,
            name: String,
        }

        let serde_val = FriendSerde {
            age: 42,
            name: "Alice".into(),
        };
        let expected = ::postcard::to_allocvec(&serde_val).unwrap();

        let facet_val = Friend {
            age: 42,
            name: "Alice".into(),
        };
        let enc = compile_encoder(Friend::SHAPE, &postcard::KajitPostcard);
        let got = serialize(&enc, &facet_val);
        assert_eq!(got, expected);
    }

    #[test]
    fn postcard_encode_all_scalars() {
        use serde::Serialize;

        #[derive(Serialize)]
        struct AllScalarsSerde {
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

        let serde_val = AllScalarsSerde {
            a_bool: true,
            a_u8: 200,
            a_u16: 1000,
            a_u32: 70000,
            a_u64: 1_000_000_000_000,
            a_u128: 18_446_744_073_709_551_621u128,
            a_usize: 12345,
            a_i8: -42,
            a_i16: -1000,
            a_i32: -70000,
            a_i64: -1_000_000_000_000,
            a_i128: -18_446_744_073_709_551_621i128,
            a_isize: -12345,
            a_f32: 3.14,
            a_f64: 2.718281828459045,
            a_char: 'ß',
            a_name: "hello".into(),
        };
        let expected = ::postcard::to_allocvec(&serde_val).unwrap();

        let facet_val = AllScalars {
            a_bool: true,
            a_u8: 200,
            a_u16: 1000,
            a_u32: 70000,
            a_u64: 1_000_000_000_000,
            a_u128: 18_446_744_073_709_551_621u128,
            a_usize: 12345,
            a_i8: -42,
            a_i16: -1000,
            a_i32: -70000,
            a_i64: -1_000_000_000_000,
            a_i128: -18_446_744_073_709_551_621i128,
            a_isize: -12345,
            a_f32: 3.14,
            a_f64: 2.718281828459045,
            a_char: 'ß',
            a_name: "hello".into(),
        };
        let enc = compile_encoder(AllScalars::SHAPE, &postcard::KajitPostcard);
        let got = serialize(&enc, &facet_val);
        assert_eq!(got, expected);
    }

    #[test]
    fn postcard_encode_nested_struct() {
        use serde::Serialize;

        #[derive(Serialize)]
        struct InnerSerde {
            x: u32,
            y: u32,
        }
        #[derive(Serialize)]
        struct OuterSerde {
            inner: InnerSerde,
            label: String,
        }

        let serde_val = OuterSerde {
            inner: InnerSerde { x: 10, y: 20 },
            label: "test".into(),
        };
        let expected = ::postcard::to_allocvec(&serde_val).unwrap();

        #[derive(Facet)]
        struct Inner {
            x: u32,
            y: u32,
        }
        #[derive(Facet)]
        struct Outer {
            inner: Inner,
            label: String,
        }

        let facet_val = Outer {
            inner: Inner { x: 10, y: 20 },
            label: "test".into(),
        };
        let enc = compile_encoder(Outer::SHAPE, &postcard::KajitPostcard);
        let got = serialize(&enc, &facet_val);
        assert_eq!(got, expected);
    }

    #[test]
    fn postcard_encode_empty_string() {
        use serde::Serialize;

        #[derive(Serialize)]
        struct WithStringSerde {
            name: String,
        }

        let serde_val = WithStringSerde {
            name: String::new(),
        };
        let expected = ::postcard::to_allocvec(&serde_val).unwrap();

        #[derive(Facet)]
        struct WithString {
            name: String,
        }

        let facet_val = WithString {
            name: String::new(),
        };
        let enc = compile_encoder(WithString::SHAPE, &postcard::KajitPostcard);
        let got = serialize(&enc, &facet_val);
        assert_eq!(got, expected);
    }

    #[test]
    fn postcard_encode_long_string() {
        use serde::Serialize;

        #[derive(Serialize)]
        struct WithStringSerde {
            data: String,
        }

        // 300 bytes — varint length requires 2 bytes
        let long = "a".repeat(300);

        let serde_val = WithStringSerde { data: long.clone() };
        let expected = ::postcard::to_allocvec(&serde_val).unwrap();

        #[derive(Facet)]
        struct WithString {
            data: String,
        }

        let facet_val = WithString { data: long };
        let enc = compile_encoder(WithString::SHAPE, &postcard::KajitPostcard);
        let got = serialize(&enc, &facet_val);
        assert_eq!(got, expected);
    }

    #[test]
    fn postcard_encode_roundtrip() {
        // Encode with kajit, decode with kajit — full roundtrip.
        let original = Friend {
            age: 42,
            name: "Alice".into(),
        };
        let enc = compile_encoder(Friend::SHAPE, &postcard::KajitPostcard);
        let bytes = serialize(&enc, &original);
        let dec = compile_decoder(Friend::SHAPE, &postcard::KajitPostcard);
        let decoded: Friend = deserialize(&dec, &bytes).unwrap();
        assert_eq!(decoded, original);
    }

    // --- JSON Encode tests ---

    #[test]
    fn json_encode_simple_struct() {
        use serde::Serialize;

        #[derive(Serialize)]
        struct FriendSerde {
            age: u32,
            name: String,
        }

        let serde_val = FriendSerde {
            age: 42,
            name: "Alice".into(),
        };
        let expected = serde_json::to_string(&serde_val).unwrap();

        let facet_val = Friend {
            age: 42,
            name: "Alice".into(),
        };
        let enc = compile_encoder(Friend::SHAPE, &json::KajitJsonEncoder);
        let got = String::from_utf8(serialize(&enc, &facet_val)).unwrap();
        assert_eq!(got, expected);
    }

    #[test]
    fn json_encode_all_scalars() {
        use serde::Serialize;

        #[derive(Serialize)]
        struct AllScalarsSerde {
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

        let serde_val = AllScalarsSerde {
            a_bool: true,
            a_u8: 200,
            a_u16: 1000,
            a_u32: 70000,
            a_u64: 1_000_000_000_000,
            a_u128: 99999999999999999999,
            a_usize: 42,
            a_i8: -100,
            a_i16: -1000,
            a_i32: -70000,
            a_i64: -1_000_000_000_000,
            a_i128: -99999999999999999999,
            a_isize: -42,
            #[allow(clippy::approx_constant)]
            a_f32: 3.14,
            #[allow(clippy::approx_constant)]
            a_f64: 2.718281828,
            a_char: 'ß',
            a_name: "hello".into(),
        };
        let expected = serde_json::to_string(&serde_val).unwrap();

        let facet_val = AllScalars {
            a_bool: true,
            a_u8: 200,
            a_u16: 1000,
            a_u32: 70000,
            a_u64: 1_000_000_000_000,
            a_u128: 99999999999999999999,
            a_usize: 42,
            a_i8: -100,
            a_i16: -1000,
            a_i32: -70000,
            a_i64: -1_000_000_000_000,
            a_i128: -99999999999999999999,
            a_isize: -42,
            a_f32: 3.14,
            #[allow(clippy::approx_constant)]
            #[allow(clippy::approx_constant)]
            a_f64: 2.718281828,
            a_char: 'ß',
            a_name: "hello".into(),
        };
        let enc = compile_encoder(AllScalars::SHAPE, &json::KajitJsonEncoder);
        let got = String::from_utf8(serialize(&enc, &facet_val)).unwrap();
        assert_eq!(got, expected);
    }

    #[test]
    fn json_encode_nested_struct() {
        use serde::Serialize;

        #[derive(Serialize)]
        struct InnerSerde {
            x: u32,
            y: u32,
        }
        #[derive(Serialize)]
        struct OuterSerde {
            inner: InnerSerde,
            label: String,
        }

        let serde_val = OuterSerde {
            inner: InnerSerde { x: 10, y: 20 },
            label: "test".into(),
        };
        let expected = serde_json::to_string(&serde_val).unwrap();

        #[derive(Facet)]
        struct Inner {
            x: u32,
            y: u32,
        }
        #[derive(Facet)]
        struct Outer {
            inner: Inner,
            label: String,
        }

        let facet_val = Outer {
            inner: Inner { x: 10, y: 20 },
            label: "test".into(),
        };
        let enc = compile_encoder(Outer::SHAPE, &json::KajitJsonEncoder);
        let got = String::from_utf8(serialize(&enc, &facet_val)).unwrap();
        assert_eq!(got, expected);
    }

    #[test]
    fn json_encode_string_escaping() {
        use serde::Serialize;

        #[derive(Serialize)]
        struct WithStringSerde {
            text: String,
        }

        #[derive(Facet)]
        struct WithString {
            text: String,
        }

        // Test various characters that need JSON escaping
        let cases = [
            "hello world",      // no escaping
            "hello \"world\"",  // quotes
            "back\\slash",      // backslash
            "line\nbreak",      // newline
            "tab\there",        // tab
            "",                 // empty
            "emoji: \u{1F600}", // unicode (no escaping needed, just UTF-8)
            "\x00\x01\x1F",     // control chars
        ];

        for text in cases {
            let serde_val = WithStringSerde { text: text.into() };
            let expected = serde_json::to_string(&serde_val).unwrap();

            let facet_val = WithString { text: text.into() };
            let enc = compile_encoder(WithString::SHAPE, &json::KajitJsonEncoder);
            let got = String::from_utf8(serialize(&enc, &facet_val)).unwrap();
            assert_eq!(got, expected, "mismatch for text: {:?}", text);
        }
    }

    #[test]
    fn json_encode_roundtrip() {
        // Encode with kajit JSON, decode with kajit JSON — full roundtrip.
        let original = Friend {
            age: 42,
            name: "Alice".into(),
        };
        let enc = compile_encoder(Friend::SHAPE, &json::KajitJsonEncoder);
        let bytes = serialize(&enc, &original);
        let dec = compile_decoder(Friend::SHAPE, &json::KajitJson);
        let decoded: Friend = deserialize(&dec, &bytes).unwrap();
        assert_eq!(decoded, original);
    }

    // ── Tuple and array tests ─────────────────────────────────────────────────

    #[test]
    fn json_tuple_deser() {
        let input = br#"[42, "Alice"]"#;
        let dec = compile_decoder(<(u32, String)>::SHAPE, &json::KajitJson);
        let result: (u32, String) = deserialize(&dec, input).unwrap();
        assert_eq!(result, (42, "Alice".into()));
    }

    #[test]
    fn postcard_tuple_deser() {
        let original: (u32, String) = (42, "Alice".into());
        let bytes = ::postcard::to_allocvec(&original).unwrap();
        let dec = compile_decoder(<(u32, String)>::SHAPE, &postcard::KajitPostcard);
        let result: (u32, String) = deserialize(&dec, &bytes).unwrap();
        assert_eq!(result, original);
    }

    #[test]
    fn json_tuple_triple_deser() {
        let input = br#"[1, 2, 3]"#;
        let dec = compile_decoder(<(u32, u32, u32)>::SHAPE, &json::KajitJson);
        let result: (u32, u32, u32) = deserialize(&dec, input).unwrap();
        assert_eq!(result, (1, 2, 3));
    }

    #[test]
    fn json_array_deser() {
        let input = br#"[10, 20, 30, 40]"#;
        let dec = compile_decoder(<[u32; 4]>::SHAPE, &json::KajitJson);
        let result: [u32; 4] = deserialize(&dec, input).unwrap();
        assert_eq!(result, [10, 20, 30, 40]);
    }

    #[test]
    fn postcard_array_deser() {
        let original: [u32; 4] = [10, 20, 30, 40];
        let bytes = ::postcard::to_allocvec(&original).unwrap();
        let dec = compile_decoder(<[u32; 4]>::SHAPE, &postcard::KajitPostcard);
        let result: [u32; 4] = deserialize(&dec, &bytes).unwrap();
        assert_eq!(result, original);
    }

    #[test]
    fn json_tuple_nested_deser() {
        let input = br#"[[1, 2], [3, 4]]"#;
        let dec = compile_decoder(<([u32; 2], [u32; 2])>::SHAPE, &json::KajitJson);
        let result: ([u32; 2], [u32; 2]) = deserialize(&dec, input).unwrap();
        assert_eq!(result, ([1, 2], [3, 4]));
    }
}

#[cfg(all(test, not(target_os = "windows")))]
mod disasm_tests;
