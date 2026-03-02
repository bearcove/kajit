//! Deser/ser cases

use crate::Case;
use proc_macro2::TokenStream;
use quote::{format_ident, quote};

pub(crate) fn types_rs() -> TokenStream {
    quote! {
        use serde::{Serialize, Deserialize};

        #[derive(Debug, PartialEq, Serialize, Deserialize, Facet, proptest_derive::Arbitrary)]
        struct Friend {
            age: u32,
            #[proptest(strategy = "proptest::string::string_regex(\"(?s).{0,64}\").unwrap()")]
            name: String,
        }

        #[derive(Debug, PartialEq, Serialize, Deserialize, Facet, proptest_derive::Arbitrary)]
        struct Address {
            #[proptest(strategy = "proptest::string::string_regex(\"(?s).{0,64}\").unwrap()")]
            city: String,
            zip: u32,
        }

        #[derive(Debug, PartialEq, Serialize, Deserialize, Facet, proptest_derive::Arbitrary)]
        struct Person {
            #[proptest(strategy = "proptest::string::string_regex(\"(?s).{0,64}\").unwrap()")]
            name: String,
            age: u32,
            address: Address,
        }

        #[derive(Debug, PartialEq, Serialize, Deserialize, Facet, proptest_derive::Arbitrary)]
        struct Inner {
            x: u32,
        }

        #[derive(Debug, PartialEq, Serialize, Deserialize, Facet, proptest_derive::Arbitrary)]
        struct Middle {
            inner: Inner,
            y: u32,
        }

        #[derive(Debug, PartialEq, Serialize, Deserialize, Facet, proptest_derive::Arbitrary)]
        struct Outer {
            middle: Middle,
            z: u32,
        }

        #[derive(Debug, PartialEq, Serialize, Deserialize, Facet, proptest_derive::Arbitrary)]
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

        #[derive(Debug, PartialEq, Serialize, Deserialize, Facet, proptest_derive::Arbitrary)]
        struct BoolField {
            value: bool,
        }

        #[derive(Debug, PartialEq, Serialize, Deserialize, Facet, proptest_derive::Arbitrary)]
        struct ScalarVec {
            #[proptest(strategy = "proptest::collection::vec(proptest::arbitrary::any::<u32>(), 0..256)")]
            values: Vec<u32>,
        }

        #[allow(dead_code)]
        type Pair = (u32, String);
    }
}

pub(crate) fn cases() -> Vec<Case> {
    vec![
        Case {
            name: "flat_struct",
            values: vec![quote!(Friend {
                age: 42,
                name: "Alice".into()
            })],
        },
        Case {
            name: "nested_struct",
            values: vec![quote!(Person {
                name: "Alice".into(),
                age: 30,
                address: Address {
                    city: "Portland".into(),
                    zip: 97201
                }
            })],
        },
        Case {
            name: "deep_struct",
            values: vec![quote!(Outer {
                middle: Middle {
                    inner: Inner { x: 1 },
                    y: 2
                },
                z: 3
            })],
        },
        Case {
            name: "all_integers",
            values: vec![quote!(AllIntegers {
                a_u8: 255,
                a_u16: 65535,
                a_u32: 1_000_000,
                a_u64: 1_000_000_000_000,
                a_u128: 340282366920938463463374607431768211455u128,
                a_usize: 123_456usize,
                a_i8: -128,
                a_i16: -32768,
                a_i32: -1_000_000,
                a_i64: -1_000_000_000_000,
                a_i128: -170141183460469231731687303715884105728i128,
                a_isize: -123_456isize
            })],
        },
        Case {
            name: "bool_field",
            values: vec![quote!(BoolField { value: true })],
        },
        Case {
            name: "tuple_pair",
            values: vec![quote!((42u32, "Alice".to_string()))],
        },
        Case {
            name: "vec_scalar_small",
            values: vec![quote!(ScalarVec {
                values: (0..16).map(|i| i as u32).collect()
            })],
        },
        Case {
            name: "vec_scalar_large",
            values: vec![quote!(ScalarVec {
                values: (0..2048).map(|i| i as u32).collect()
            })],
        },
    ]
}

pub(crate) fn render_bench_file() -> String {
    let cases = cases();
    let types = types_rs();
    let bench_calls: Vec<TokenStream> = cases
        .iter()
        .flat_map(|case| {
            case.values.iter().enumerate().map(|(sample_idx, value)| {
                let sample_name = if case.values.len() == 1 {
                    case.name.to_string()
                } else {
                    format!("{}__v{}", case.name, sample_idx)
                };
                let value = value.clone();
                quote! {
                    register_bench_case(&mut v, #sample_name, #value);
                }
            })
        })
        .collect();
    let file_tokens = quote! {
        #[path = "harness.rs"]
        mod harness;

        use facet::Facet;
        use std::hint::black_box;
        use std::sync::Arc;

        #types

        fn register_bench_case<T>(v: &mut Vec<harness::Bench>, group: &str, value: T)
        where
            for<'input> T: Facet<'input> + serde::Serialize + serde::de::DeserializeOwned + 'static,
        {
            let json_data = Arc::new(serde_json::to_string(&value).unwrap());
            let postcard_data = Arc::new(postcard::to_allocvec(&value).unwrap());
            let value = Arc::new(value);

            let json_decoder =
                Arc::new(kajit::compile_decoder_legacy(T::SHAPE, &kajit::json::KajitJson));
            let json_encoder =
                Arc::new(kajit::compile_encoder(T::SHAPE, &kajit::json::KajitJsonEncoder));

            let postcard_decoder =
                Arc::new(kajit::compile_decoder_legacy(T::SHAPE, &kajit::postcard::KajitPostcard));
            let postcard_ir_decoder =
                Arc::new(kajit::compile_decoder_via_ir(T::SHAPE, &kajit::postcard::KajitPostcard));
            let postcard_encoder =
                Arc::new(kajit::compile_encoder(T::SHAPE, &kajit::postcard::KajitPostcard));

            let json_prefix = format!("{group}/json");
            let postcard_prefix = format!("{group}/postcard");

            v.push(harness::Bench {
                name: format!("{json_prefix}/serde_deser"),
                func: Box::new({
                    let data = Arc::clone(&json_data);
                    move |runner| {
                        runner.run(|| {
                            black_box(serde_json::from_str::<T>(black_box(data.as_str())).unwrap());
                        });
                    }
                }),
            });
            v.push(harness::Bench {
                name: format!("{json_prefix}/kajit_dynasm_deser"),
                func: Box::new({
                    let data = Arc::clone(&json_data);
                    let decoder = Arc::clone(&json_decoder);
                    move |runner| {
                        let decoder = &*decoder;
                        runner.run(|| {
                            black_box(kajit::from_str::<T>(decoder, black_box(data.as_str())).unwrap());
                        });
                    }
                }),
            });
            v.push(harness::Bench {
                name: format!("{json_prefix}/serde_ser"),
                func: Box::new({
                    let value = Arc::clone(&value);
                    move |runner| {
                        runner.run(|| {
                            black_box(serde_json::to_vec(black_box(&*value)).unwrap());
                        });
                    }
                }),
            });
            v.push(harness::Bench {
                name: format!("{json_prefix}/kajit_dynasm_ser"),
                func: Box::new({
                    let value = Arc::clone(&value);
                    let encoder = Arc::clone(&json_encoder);
                    move |runner| {
                        let encoder = &*encoder;
                        runner.run(|| {
                            black_box(kajit::serialize(encoder, black_box(&*value)));
                        });
                    }
                }),
            });

            v.push(harness::Bench {
                name: format!("{postcard_prefix}/serde_deser"),
                func: Box::new({
                    let data = Arc::clone(&postcard_data);
                    move |runner| {
                        runner.run(|| {
                            black_box(postcard::from_bytes::<T>(black_box(&data[..])).unwrap());
                        });
                    }
                }),
            });
            v.push(harness::Bench {
                name: format!("{postcard_prefix}/kajit_dynasm_deser"),
                func: Box::new({
                    let data = Arc::clone(&postcard_data);
                    let decoder = Arc::clone(&postcard_decoder);
                    move |runner| {
                        let decoder = &*decoder;
                        runner.run(|| {
                            black_box(kajit::deserialize::<T>(decoder, black_box(&data[..])).unwrap());
                        });
                    }
                }),
            });
            v.push(harness::Bench {
                name: format!("{postcard_prefix}/kajit_ir_deser"),
                func: Box::new({
                    let data = Arc::clone(&postcard_data);
                    let decoder = Arc::clone(&postcard_ir_decoder);
                    move |runner| {
                        let decoder = &*decoder;
                        runner.run(|| {
                            black_box(kajit::deserialize::<T>(decoder, black_box(&data[..])).unwrap());
                        });
                    }
                }),
            });
            v.push(harness::Bench {
                name: format!("{postcard_prefix}/serde_ser"),
                func: Box::new({
                    let value = Arc::clone(&value);
                    move |runner| {
                        runner.run(|| {
                            black_box(postcard::to_allocvec(black_box(&*value)).unwrap());
                        });
                    }
                }),
            });
            v.push(harness::Bench {
                name: format!("{postcard_prefix}/kajit_dynasm_ser"),
                func: Box::new({
                    let value = Arc::clone(&value);
                    let encoder = Arc::clone(&postcard_encoder);
                    move |runner| {
                        let encoder = &*encoder;
                        runner.run(|| {
                            black_box(kajit::serialize(encoder, black_box(&*value)));
                        });
                    }
                }),
            });
        }

        fn main() {
            let mut v: Vec<harness::Bench> = Vec::new();
            #(#bench_calls)*
            harness::run_benchmarks(v);
        }
    };
    let file: syn::File =
        syn::parse2(file_tokens).expect("generated synthetic bench file should parse");
    format!(
        "// @generated by xtask generate-synthetic. Do not edit manually.\n{}",
        prettyplease::unparse(&file)
    )
}

pub(crate) fn render_test_file() -> String {
    let cases = cases();
    let types = types_rs();
    let json_tests: Vec<TokenStream> = cases
        .iter()
        .flat_map(|case| {
            case.values
                .iter()
                .enumerate()
                .map(|(sample_idx, value)| {
                    let test_name = if case.values.len() == 1 {
                        format_ident!("{}", case.name)
                    } else {
                        format_ident!("{}_v{}", case.name, sample_idx)
                    };
                    let case_name = if case.values.len() == 1 {
                        case.name.to_string()
                    } else {
                        format!("{}__v{}", case.name, sample_idx)
                    };
                    let value = value.clone();
                    quote! {
                        #[test]
                        fn #test_name() {
                            let value = #value;
                            assert_codegen_snapshots("json", #case_name, &kajit::json::KajitJson, &value);
                            assert_json_case(value);
                        }
                    }
                })
        })
        .collect();
    let postcard_tests: Vec<TokenStream> = cases
        .iter()
        .flat_map(|case| {
            case.values
                .iter()
                .enumerate()
                .map(|(sample_idx, value)| {
                    let test_name = if case.values.len() == 1 {
                        format_ident!("{}", case.name)
                    } else {
                        format_ident!("{}_v{}", case.name, sample_idx)
                    };
                    let case_name = if case.values.len() == 1 {
                        case.name.to_string()
                    } else {
                        format!("{}__v{}", case.name, sample_idx)
                    };
                    let value = value.clone();
                    quote! {
                        #[test]
                        fn #test_name() {
                            let value = #value;
                            assert_codegen_snapshots("postcard", #case_name, &kajit::postcard::KajitPostcard, &value);
                            assert_postcard_case(value);
                        }
                    }
                })
        })
        .collect();
    let prop_tests: Vec<TokenStream> = cases
        .iter()
        .map(|case| {
            let test_name = format_ident!("{}", case.name);
            let value = case
                .values
                .first()
                .cloned()
                .expect("each case should define at least one sample value");
            quote! {
                #[test]
                fn #test_name() {
                    let marker = #value;
                    assert_prop_case(&marker);
                }
            }
        })
        .collect();
    let file_tokens = quote! {
        use facet::Facet;
        use proptest::arbitrary::Arbitrary;
        use std::fmt::Write;
        #[cfg(target_arch = "x86_64")]
        use yaxpeax_arch::LengthedInstruction;
        use yaxpeax_arch::{Decoder, U8Reader};

        #types

        fn assert_json_case<T>(value: T)
        where
            for<'input> T: Facet<'input> + serde::Serialize + serde::de::DeserializeOwned + PartialEq + std::fmt::Debug,
        {
            let encoded = serde_json::to_string(&value).unwrap();
            let expected: T = serde_json::from_str(&encoded).unwrap();
            let decoder = kajit::compile_decoder_legacy(T::SHAPE, &kajit::json::KajitJson);
            let got: T = kajit::from_str(&decoder, &encoded).unwrap();
            assert_eq!(got, expected);
        }

        fn assert_postcard_case<T>(value: T)
        where
            for<'input> T: Facet<'input> + serde::Serialize + serde::de::DeserializeOwned + PartialEq + std::fmt::Debug,
        {
            let encoded = ::postcard::to_allocvec(&value).unwrap();
            let expected: T = ::postcard::from_bytes(&encoded).unwrap();
            let legacy = kajit::compile_decoder_legacy(T::SHAPE, &kajit::postcard::KajitPostcard);
            let legacy_out: T = kajit::deserialize(&legacy, &encoded).unwrap();
            assert_eq!(legacy_out, expected);
            let ir = kajit::compile_decoder_via_ir(T::SHAPE, &kajit::postcard::KajitPostcard);
            let ir_out: T = kajit::deserialize(&ir, &encoded).unwrap();
            assert_eq!(ir_out, expected);
        }

        fn assert_prop_case<T>(_marker: &T)
        where
            for<'input> T: Facet<'input> + serde::Serialize + serde::de::DeserializeOwned + PartialEq + std::fmt::Debug + Arbitrary + 'static,
        {
            let json_decoder = kajit::compile_decoder_legacy(T::SHAPE, &kajit::json::KajitJson);
            let postcard_legacy =
                kajit::compile_decoder_legacy(T::SHAPE, &kajit::postcard::KajitPostcard);
            let postcard_ir =
                kajit::compile_decoder_via_ir(T::SHAPE, &kajit::postcard::KajitPostcard);
            let mut runner = proptest::test_runner::TestRunner::new(proptest::test_runner::Config {
                cases: 64,
                ..proptest::test_runner::Config::default()
            });
            let strategy = proptest::arbitrary::any::<T>();
            runner
                .run(&strategy, |value| {
                    let json_encoded = serde_json::to_string(&value).unwrap();
                    let json_expected: T = serde_json::from_str(&json_encoded).unwrap();
                    let json_got: T = kajit::from_str(&json_decoder, &json_encoded).unwrap();
                    assert_eq!(json_got, json_expected);

                    let postcard_encoded = ::postcard::to_allocvec(&value).unwrap();
                    let postcard_expected: T = ::postcard::from_bytes(&postcard_encoded).unwrap();
                    let postcard_legacy_out: T =
                        kajit::deserialize(&postcard_legacy, &postcard_encoded).unwrap();
                    assert_eq!(postcard_legacy_out, postcard_expected);

                    let postcard_ir_out: T =
                        kajit::deserialize(&postcard_ir, &postcard_encoded).unwrap();
                    assert_eq!(postcard_ir_out, postcard_expected);
                    Ok(())
                })
                .unwrap();
        }

        fn disasm_bytes(code: &[u8], marker_offset: Option<usize>) -> String {
            let mut out = String::new();

            #[cfg(target_arch = "aarch64")]
            {
                use yaxpeax_arm::armv8::a64::InstDecoder;

                let decoder = InstDecoder::default();
                let mut reader = U8Reader::new(code);
                let mut offset = 0usize;
                let mut ret_count = 0u32;

                while offset + 4 <= code.len() {
                    let prefix = if marker_offset == Some(offset) { "> " } else { "  " };
                    match decoder.decode(&mut reader) {
                        Ok(inst) => {
                            let text = kajit::disasm_normalize::normalize_inst(&format!("{inst}"));
                            writeln!(&mut out, "{prefix}{text}").unwrap();
                            if text.trim() == "ret" {
                                ret_count += 1;
                                if ret_count >= 2 {
                                    break;
                                }
                            }
                        }
                        Err(e) => {
                            let word = u32::from_le_bytes(code[offset..offset + 4].try_into().unwrap());
                            writeln!(&mut out, "{prefix}<{e}> (0x{word:08x})").unwrap();
                        }
                    }
                    offset += 4;
                }
            }

            #[cfg(target_arch = "x86_64")]
            {
                use yaxpeax_x86::amd64::InstDecoder;

                let decoder = InstDecoder::default();
                let mut reader = U8Reader::new(code);
                let mut offset = 0usize;
                let mut ret_count = 0u32;

                while offset < code.len() {
                    let prefix = if marker_offset == Some(offset) { "> " } else { "  " };
                    match decoder.decode(&mut reader) {
                        Ok(inst) => {
                            let len = inst.len().to_const() as usize;
                            let text = kajit::disasm_normalize::normalize_inst(&format!("{inst}"));
                            writeln!(&mut out, "{prefix}{text}").unwrap();
                            if text.trim() == "ret" {
                                ret_count += 1;
                                if ret_count >= 2 {
                                    break;
                                }
                            }
                            offset += len;
                        }
                        Err(_) => {
                            writeln!(&mut out, "{prefix}<decode error> (0x{:02x})", code[offset]).unwrap();
                            offset += 1;
                        }
                    }
                }
            }

            out
        }

        fn codegen_artifacts<T, F>(decoder: &F) -> (String, String, usize, String)
        where
            for<'input> T: Facet<'input>,
            F: kajit::format::Decoder + kajit::format::IrDecoder,
        {
            let shape = T::SHAPE;
            let (ir_text, ra_text) = kajit::debug_ir_and_ra_mir_text(shape, decoder);
            let edits = kajit::regalloc_edit_count_via_ir(shape, decoder);
            let compiled = kajit::compile_decoder_with_backend(shape, decoder, kajit::DecoderBackend::Ir);
            let disasm = disasm_bytes(compiled.code(), Some(compiled.entry_offset()));
            (ir_text, ra_text, edits, disasm)
        }

        fn assert_codegen_snapshots<T, F>(
            format_label: &str,
            case: &str,
            decoder: &F,
            _marker: &T,
        )
        where
            for<'input> T: Facet<'input>,
            F: kajit::format::Decoder + kajit::format::IrDecoder,
        {
            let (ir_text, ra_text, edits, disasm) = codegen_artifacts::<T, F>(decoder);
            insta::assert_snapshot!(
                format!(
                    "generated_rvsdg_{}_{}_{}",
                    format_label,
                    case,
                    std::env::consts::ARCH
                ),
                ir_text
            );
            insta::assert_snapshot!(
                format!(
                    "generated_ra_mir_{}_{}_{}",
                    format_label,
                    case,
                    std::env::consts::ARCH
                ),
                ra_text
            );
            insta::assert_snapshot!(
                format!(
                    "generated_postreg_disasm_{}_{}_{}",
                    format_label,
                    case,
                    std::env::consts::ARCH
                ),
                disasm
            );
            insta::assert_snapshot!(
                format!(
                    "generated_postreg_edits_{}_{}_{}",
                    format_label,
                    case,
                    std::env::consts::ARCH
                ),
                format!("{edits}")
            );
        }

        mod json {
            use super::*;
            #(#json_tests)*
        }

        mod postcard {
            use super::*;
            #(#postcard_tests)*
        }

        mod prop {
            use super::*;
            #(#prop_tests)*
        }

        mod postreg {
            use super::*;

            #[test]
            fn vec_scalar_large_hotpath_asserts() {
                let (ir_text, ra_text, edits, disasm) =
                    codegen_artifacts::<ScalarVec, _>(&kajit::postcard::KajitPostcard);

                assert!(
                    ir_text.contains("theta") || ir_text.contains("apply @"),
                    "expected loop form (`theta`) or outlined loop body (`apply`) in IR"
                );
                assert!(
                    ra_text.contains("branch_if"),
                    "expected loop backedge in RA-MIR"
                );
                assert!(
                    ra_text.contains("call_intrinsic"),
                    "expected intrinsic-heavy vec decode path in RA-MIR"
                );
                assert!(edits <= 128, "expected edit budget <= 128, got {edits}");

                assert!(!disasm.is_empty(), "expected non-empty disassembly artifact");
            }
        }
    };
    let file: syn::File =
        syn::parse2(file_tokens).expect("generated synthetic test file should parse");
    format!(
        "// @generated by xtask generate-synthetic. Do not edit manually.\n{}",
        prettyplease::unparse(&file)
    )
}
