use facet::Facet;
use std::fmt::Write;
#[cfg(target_arch = "x86_64")]
use yaxpeax_arch::LengthedInstruction;
use yaxpeax_arch::{Decoder, U8Reader};

#[derive(Debug, PartialEq, serde::Serialize, serde::Deserialize, Facet)]
struct ScalarVec {
    values: Vec<u32>,
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
            let prefix = if marker_offset == Some(offset) {
                "> "
            } else {
                "  "
            };
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

fn vec_hotpath_artifacts() -> (String, String, usize, String) {
    let (ir_text, ra_text) =
        kajit::debug_ir_and_ra_mir_text(ScalarVec::SHAPE, &kajit::postcard::KajitPostcard);
    let edits =
        kajit::regalloc_edit_count_via_ir(ScalarVec::SHAPE, &kajit::postcard::KajitPostcard);
    let dec = kajit::compile_decoder_with_backend(
        ScalarVec::SHAPE,
        &kajit::postcard::KajitPostcard,
        kajit::DecoderBackend::Ir,
    );
    let disasm = disasm_bytes(dec.code(), Some(dec.entry_offset()));
    (ir_text, ra_text, edits, disasm)
}

#[test]
fn postreg_vec_hotpath_snapshot_ir_ra_disasm() {
    let (ir_text, ra_text, edits, disasm) = vec_hotpath_artifacts();
    insta::assert_snapshot!("postreg_vec_hotpath_ir", ir_text);
    insta::assert_snapshot!("postreg_vec_hotpath_ra_mir", ra_text);
    insta::assert_snapshot!("postreg_vec_hotpath_disasm", disasm);
    insta::assert_snapshot!("postreg_vec_hotpath_edits", format!("{edits}"));
}

#[test]
fn postreg_vec_hotpath_asserts_loop_shape_and_budget() {
    let (ir_text, ra_text, edits, disasm) = vec_hotpath_artifacts();

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
