use facet::Facet;
use std::fmt::Write;
use yaxpeax_arch::Decoder;
#[cfg(target_arch = "aarch64")]
use yaxpeax_arch::U8Reader;
#[cfg(target_arch = "x86_64")]
use yaxpeax_arch::{LengthedInstruction, U8Reader};

#[derive(Facet, Debug)]
struct ScalarVec {
    values: Vec<u32>,
}

fn main() {
    let legacy = kajit::compile_decoder(ScalarVec::SHAPE, &kajit::postcard::KajitPostcard);
    let ir = kajit::compile_decoder(ScalarVec::SHAPE, &kajit::postcard::KajitPostcard);

    println!("=== kajit postcard ScalarVec (legacy) ===");
    println!(
        "{}",
        disasm_bytes(
            legacy.code(),
            legacy.code().as_ptr() as u64,
            Some(legacy.entry_offset()),
        )
    );

    println!("\n=== kajit postcard ScalarVec (ir) ===");
    println!(
        "{}",
        disasm_bytes(
            ir.code(),
            ir.code().as_ptr() as u64,
            Some(ir.entry_offset()),
        )
    );
}

fn disasm_bytes(code: &[u8], base_addr: u64, marker_offset: Option<usize>) -> String {
    let mut out = String::new();

    #[cfg(target_arch = "aarch64")]
    {
        use yaxpeax_arm::armv8::a64::InstDecoder;

        let decoder = InstDecoder::default();
        let mut reader = U8Reader::new(code);
        let mut offset = 0usize;
        let mut ret_count = 0u32;

        while offset + 4 <= code.len() {
            let marker = match marker_offset {
                Some(m) if m == offset => " <entry>",
                _ => "",
            };
            match decoder.decode(&mut reader) {
                Ok(inst) => {
                    let addr = base_addr + offset as u64;
                    writeln!(&mut out, "{addr:12x}:{marker}  {inst}").unwrap();
                    if format!("{inst}").trim() == "ret" {
                        ret_count += 1;
                        if ret_count >= 4 {
                            break;
                        }
                    }
                }
                Err(e) => {
                    let word = u32::from_le_bytes(code[offset..offset + 4].try_into().unwrap());
                    let addr = base_addr + offset as u64;
                    writeln!(&mut out, "{addr:12x}:{marker}  <{e}> (0x{word:08x})").unwrap();
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
            let marker = match marker_offset {
                Some(m) if m == offset => " <entry>",
                _ => "",
            };
            match decoder.decode(&mut reader) {
                Ok(inst) => {
                    let len = inst.len().to_const() as usize;
                    let addr = base_addr + offset as u64;
                    writeln!(&mut out, "{addr:12x}:{marker}  {inst}").unwrap();
                    if format!("{inst}").trim() == "ret" {
                        ret_count += 1;
                        if ret_count >= 4 {
                            break;
                        }
                    }
                    offset += len;
                }
                Err(_) => {
                    let addr = base_addr + offset as u64;
                    writeln!(
                        &mut out,
                        "{addr:12x}:{marker}  <decode error> (0x{:02x})",
                        code[offset]
                    )
                    .unwrap();
                    offset += 1;
                }
            }
        }
    }

    out
}
