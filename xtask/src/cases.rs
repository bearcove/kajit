//! Deser/ser cases

use crate::Case;
use proc_macro2::TokenStream;
use quote::{format_ident, quote};

pub(crate) fn types_rs() -> TokenStream {
    quote! {
        use serde::{Serialize, Deserialize};

        #[derive(Debug, PartialEq, Serialize, Deserialize, Facet)]
        struct Friend {
            age: u32,
            name: String,
        }

        #[derive(Debug, PartialEq, Serialize, Deserialize, Facet)]
        struct Address {
            city: String,
            zip: u32,
        }

        #[derive(Debug, PartialEq, Serialize, Deserialize, Facet)]
        struct Person {
            name: String,
            age: u32,
            address: Address,
        }

        #[derive(Debug, PartialEq, Serialize, Deserialize, Facet)]
        struct Inner {
            x: u32,
        }

        #[derive(Debug, PartialEq, Serialize, Deserialize, Facet)]
        struct Middle {
            inner: Inner,
            y: u32,
        }

        #[derive(Debug, PartialEq, Serialize, Deserialize, Facet)]
        struct Outer {
            middle: Middle,
            z: u32,
        }

        #[derive(Debug, PartialEq, Serialize, Deserialize, Facet)]
        struct AllIntegers {
            a_u8: u8,
            a_u16: u16,
            a_u32: u32,
            a_u64: u64,
            a_i8: i8,
            a_i16: i16,
            a_i32: i32,
            a_i64: i64,
        }

        #[derive(Debug, PartialEq, Serialize, Deserialize, Facet)]
        struct BoolField {
            value: bool,
        }

        #[derive(Debug, PartialEq, Serialize, Deserialize, Facet)]
        struct ScalarVec {
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
            value: quote!(Friend {
                age: 42,
                name: "Alice".into()
            }),
        },
        Case {
            name: "nested_struct",
            value: quote!(Person {
                name: "Alice".into(),
                age: 30,
                address: Address {
                    city: "Portland".into(),
                    zip: 97201
                }
            }),
        },
        Case {
            name: "deep_struct",
            value: quote!(Outer {
                middle: Middle {
                    inner: Inner { x: 1 },
                    y: 2
                },
                z: 3
            }),
        },
        Case {
            name: "all_integers",
            value: quote!(AllIntegers {
                a_u8: 255,
                a_u16: 65535,
                a_u32: 1_000_000,
                a_u64: 1_000_000_000_000,
                a_i8: -128,
                a_i16: -32768,
                a_i32: -1_000_000,
                a_i64: -1_000_000_000_000
            }),
        },
        Case {
            name: "bool_field",
            value: quote!(BoolField { value: true }),
        },
        Case {
            name: "tuple_pair",
            value: quote!((42u32, "Alice".to_string())),
        },
        Case {
            name: "vec_scalar_small",
            value: quote!(ScalarVec {
                values: (0..16).map(|i| i as u32).collect()
            }),
        },
        Case {
            name: "vec_scalar_large",
            value: quote!(ScalarVec {
                values: (0..2048).map(|i| i as u32).collect()
            }),
        },
    ]
}

pub(crate) fn render_bench_file() -> String {
    let cases = cases();
    let types = types_rs();
    let bench_calls: Vec<TokenStream> = cases
        .iter()
        .map(|case| {
            let case_name = case.name;
            let value = case.value.clone();
            quote! {
                register_bench_case(&mut v, #case_name, #value);
            }
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
        .map(|case| {
            let test_name = format_ident!("generated_json_{}", case.name);
            let value = case.value.clone();
            let case_name = case.name;
            quote! {
                #[test]
                fn #test_name() {
                    let value = #value;
                    assert_codegen_snapshots("json", #case_name, &kajit::json::KajitJson, &value);
                    assert_json_case(value);
                }
            }
        })
        .collect();
    let postcard_tests: Vec<TokenStream> = cases
        .iter()
        .map(|case| {
            let test_name = format_ident!("generated_postcard_{}", case.name);
            let value = case.value.clone();
            let case_name = case.name;
            quote! {
                #[test]
                fn #test_name() {
                    let value = #value;
                    assert_codegen_snapshots("postcard", #case_name, &kajit::postcard::KajitPostcard, &value);
                    assert_postcard_case(value);
                }
            }
        })
        .collect();
    let file_tokens = quote! {
        use facet::Facet;
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
            let encoded = postcard::to_allocvec(&value).unwrap();
            let expected: T = postcard::from_bytes(&encoded).unwrap();
            let legacy = kajit::compile_decoder_legacy(T::SHAPE, &kajit::postcard::KajitPostcard);
            let legacy_out: T = kajit::deserialize(&legacy, &encoded).unwrap();
            assert_eq!(legacy_out, expected);
            let ir = kajit::compile_decoder_via_ir(T::SHAPE, &kajit::postcard::KajitPostcard);
            let ir_out: T = kajit::deserialize(&ir, &encoded).unwrap();
            assert_eq!(ir_out, expected);
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

        #[test]
        fn generated_postreg_hotpath_asserts_postcard_vec_scalar_large() {
            let (ir_text, ra_text, edits, disasm) =
                codegen_artifacts::<ScalarVec, _>(&kajit::postcard::KajitPostcard);

            assert!(ir_text.contains("theta"), "expected loop (`theta`) in IR");
            assert!(
                ra_text.contains("branch_if"),
                "expected loop backedge in RA-MIR"
            );
            assert!(
                ra_text.contains("call_intrinsic"),
                "expected intrinsic-heavy vec decode path in RA-MIR"
            );
            assert!(edits <= 128, "expected edit budget <= 128, got {edits}");

            #[cfg(target_arch = "aarch64")]
            assert!(
                disasm.contains("blr x16"),
                "expected intrinsic call sites in aarch64 disasm"
            );
            #[cfg(target_arch = "x86_64")]
            assert!(
                disasm.contains("call"),
                "expected intrinsic call sites in x86_64 disasm"
            );
        }

        #(#json_tests)*
        #(#postcard_tests)*
    };
    let file: syn::File =
        syn::parse2(file_tokens).expect("generated synthetic test file should parse");
    format!(
        "// @generated by xtask generate-synthetic. Do not edit manually.\n{}",
        prettyplease::unparse(&file)
    )
}
