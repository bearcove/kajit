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
    fn postcard_long_varints_via_ir() {
        #[derive(Debug, PartialEq, serde::Serialize, serde::Deserialize, Facet)]
        struct Address {
            city: String,
            zip: u32,
        }

        #[derive(Debug, PartialEq, serde::Serialize, serde::Deserialize, Facet)]
        struct Person {
            name: String,
            age: u32,
            address: Address,
        }

        let source = Person {
            name: "a".repeat(128),
            age: 128,
            address: Address {
                city: "b".repeat(128),
                zip: 128,
            },
        };
        let encoded = ::postcard::to_allocvec(&source).unwrap();
        let legacy = compile_decoder_legacy(Person::SHAPE, &postcard::KajitPostcard);
        let via_ir = compile_decoder_via_ir(Person::SHAPE, &postcard::KajitPostcard);
        let legacy_out: Person = deserialize(&legacy, &encoded).unwrap();
        let ir_out: Person = deserialize(&via_ir, &encoded).unwrap();
        assert_eq!(legacy_out, source);
        assert_eq!(ir_out, source);
    }

    #[test]
    fn postcard_all_integers_wide_via_ir() {
        #[derive(Debug, PartialEq, serde::Serialize, serde::Deserialize, Facet)]
        struct AllIntegers {
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
        }

        let source = AllIntegers {
            a_u8: 0,
            a_u16: 128,
            a_u32: 128,
            a_u64: 128,
            a_u128: 0,
            a_usize: 2_310_817_621_330_714usize,
            a_i8: 82,
            a_i16: -27_214,
            a_i32: -753_462_665,
            a_i64: 5_113_149_701_919_602_663,
            a_i128: 111_719_190_169_970_084_407_522_330_417_561_111_272i128,
            a_isize: 5_474_093_000_439_056_201isize,
        };
        let encoded = ::postcard::to_allocvec(&source).unwrap();
        let legacy = compile_decoder_legacy(AllIntegers::SHAPE, &postcard::KajitPostcard);
        let via_ir = compile_decoder_via_ir(AllIntegers::SHAPE, &postcard::KajitPostcard);
        let legacy_out: AllIntegers = deserialize(&legacy, &encoded).unwrap();
        let ir_out: AllIntegers = deserialize(&via_ir, &encoded).unwrap();
        assert_eq!(legacy_out, source);
        assert_eq!(ir_out, source);
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
}

#[cfg(all(test, not(target_os = "windows")))]
mod disasm_tests;
