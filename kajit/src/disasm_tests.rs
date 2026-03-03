#![allow(dead_code)]

use crate::compiler::CompiledDecoder;
use facet::Facet;
use std::fmt::Write;
#[cfg(target_arch = "x86_64")]
use yaxpeax_arch::LengthedInstruction;
use yaxpeax_arch::{Decoder, U8Reader};

#[derive(Facet)]
struct AllScalars {
    a_bool: bool,
    a_u8: u8,
    a_u16: u16,
    a_u32: u32,
    a_u64: u64,
    a_i8: i8,
    a_i16: i16,
    a_i32: i32,
    a_i64: i64,
    a_f32: f32,
    a_f64: f64,
    a_string: String,
}

#[derive(Facet)]
struct Inner {
    x: u32,
}

#[derive(Facet)]
struct Middle {
    inner: Inner,
    y: u32,
}

#[derive(Facet)]
struct Outer {
    middle: Middle,
    z: u32,
}

#[derive(Facet)]
struct Metadata {
    version: u32,
    author: String,
}

#[derive(Facet)]
struct Document {
    title: String,
    #[facet(flatten)]
    meta: Metadata,
}

#[derive(Facet)]
struct Address {
    city: String,
    zip: u32,
}

#[derive(Facet)]
struct OptionModel {
    value: Option<u32>,
    name: Option<String>,
    addr: Option<Address>,
}

#[derive(Facet)]
struct VecModel {
    vals: Vec<u32>,
    items: Vec<String>,
    addrs: Vec<Address>,
}

#[derive(Facet)]
#[repr(u8)]
enum Animal {
    Cat,
    Dog { name: String, good_boy: bool },
    Parrot(String),
}

#[derive(Facet)]
#[facet(tag = "type", content = "data")]
#[repr(u8)]
enum AdjAnimal {
    Cat,
    Dog { name: String, good_boy: bool },
    Parrot(String),
}

#[derive(Facet)]
#[facet(tag = "type")]
#[repr(u8)]
enum IntAnimal {
    Cat,
    Dog { name: String, good_boy: bool },
}

#[derive(Facet)]
#[facet(untagged)]
#[repr(u8)]
enum UntaggedAnimal {
    Cat,
    Dog { name: String, good_boy: bool },
    Parrot(String),
}

fn assert_case_snapshot(
    format_label: &str,
    case_label: &str,
    shape: &'static facet::Shape,
    decoder: &dyn crate::format::Decoder,
) {
    let deser = crate::compile_decoder(shape, decoder);
    let mut out = String::new();
    let label = format!("{format_label}/{case_label}");
    writeln!(out, "=== {label} ===").unwrap();
    out.push_str(&disasm_jit(&deser));
    out.push('\n');
    insta::assert_snapshot!(
        format!(
            "disasm_{}_{}_{}",
            format_label,
            case_label,
            std::env::consts::ARCH
        ),
        out
    );
}

fn disasm_jit(deser: &CompiledDecoder) -> String {
    disasm_bytes(deser.code(), Some(deser.entry_offset()))
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
            let prefix = if marker_offset == Some(offset) {
                "> "
            } else {
                "  "
            };
            match decoder.decode(&mut reader) {
                Ok(inst) => {
                    let text = crate::disasm_normalize::normalize_inst(&format!("{inst}"));
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
            let prefix = if marker_offset == Some(offset) {
                "> "
            } else {
                "  "
            };
            match decoder.decode(&mut reader) {
                Ok(inst) => {
                    let len = inst.len().to_const() as usize;
                    let text = crate::disasm_normalize::normalize_inst(&format!("{inst}"));
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

#[test]
fn disasm_postcard_supported_surface() {
    assert_case_snapshot(
        "postcard",
        "all_scalars",
        AllScalars::SHAPE,
        &crate::postcard::KajitPostcard,
    );
    assert_case_snapshot(
        "postcard",
        "nested_struct",
        Outer::SHAPE,
        &crate::postcard::KajitPostcard,
    );
    assert_case_snapshot(
        "postcard",
        "flatten",
        Document::SHAPE,
        &crate::postcard::KajitPostcard,
    );
    assert_case_snapshot(
        "postcard",
        "enum_external",
        Animal::SHAPE,
        &crate::postcard::KajitPostcard,
    );
    assert_case_snapshot(
        "postcard",
        "option",
        OptionModel::SHAPE,
        &crate::postcard::KajitPostcard,
    );
    assert_case_snapshot(
        "postcard",
        "vec",
        VecModel::SHAPE,
        &crate::postcard::KajitPostcard,
    );
}

#[test]
fn disasm_json_supported_surface() {
    assert_case_snapshot(
        "json",
        "all_scalars",
        AllScalars::SHAPE,
        &crate::json::KajitJson,
    );
    assert_case_snapshot(
        "json",
        "nested_struct",
        Outer::SHAPE,
        &crate::json::KajitJson,
    );
    assert_case_snapshot("json", "vec", VecModel::SHAPE, &crate::json::KajitJson);
}
