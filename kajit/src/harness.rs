//! Standalone test harness generator.
//!
//! Generates a Mach-O executable containing:
//! - The JIT-compiled decoder code in `.text`
//! - DWARF debug sections (`.debug_line`, `.debug_info`, `.debug_abbrev`)
//! - A C harness wrapper that sets up input/output and calls the decoder
//!
//! Usage: `kajit compile postcard u32 -s harness`
//! Produces: `harness_postcard_u32` executable + source listing

use std::collections::HashMap;
use std::path::Path;

/// Where a vreg lives after register allocation.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub enum VRegLocation {
    /// Physical register (aarch64 GPR index: 0=x0, 1=x1, ..., 28=x28)
    Register(u8),
    /// Stack slot (offset from frame pointer in bytes)
    StackSlot(u32),
    /// Rematerializable constant (re-emitted as movz/movk, not loaded from stack)
    Constant(u64),
}

/// Maps vreg index → physical location. Used by the lockstep debugger
/// to read JIT register/stack state and compare with interpreter vreg values.
#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize)]
pub struct AllocationMap {
    /// vreg_index → location
    pub locations: HashMap<u32, VRegLocation>,
    /// Number of spill slots allocated (for computing frame size)
    pub num_spill_slots: usize,
}

impl AllocationMap {
    /// Build from a regalloc3 allocation result.
    pub fn from_regalloc3(alloc: &kajit_mir::regalloc3_result::AllocatedCfgFunctionRa3) -> Self {
        let mut locations = HashMap::new();

        for (&vreg, allocation) in &alloc.allocations {
            match allocation {
                kajit_mir::regalloc3::linear_scan::Allocation::Reg(preg) => {
                    locations.insert(vreg.index() as u32, VRegLocation::Register(preg.0));
                }
                kajit_mir::regalloc3::linear_scan::Allocation::Spill => {
                    // Check if it's rematerializable first
                    if let Some(&value) = alloc.rematerializable.get(&vreg) {
                        locations.insert(vreg.index() as u32, VRegLocation::Constant(value));
                    } else if let Some(slot) = alloc.spill_slots.get(&vreg) {
                        locations.insert(
                            vreg.index() as u32,
                            VRegLocation::StackSlot(slot.0 * 8), // each slot is 8 bytes
                        );
                    }
                }
            }
        }

        Self {
            locations,
            num_spill_slots: alloc.num_spillslots,
        }
    }

    /// Write as JSON to a file.
    pub fn write_json(&self, path: &Path) -> Result<(), std::io::Error> {
        let json = serde_json::to_string_pretty(self)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))?;
        std::fs::write(path, json)
    }

    /// Get the aarch64 register name for a physical register index.
    pub fn reg_name(preg: u8) -> &'static str {
        match preg {
            0 => "x0",
            1 => "x1",
            2 => "x2",
            3 => "x3",
            4 => "x4",
            5 => "x5",
            6 => "x6",
            7 => "x7",
            8 => "x8",
            9 => "x9",
            10 => "x10",
            11 => "x11",
            12 => "x12",
            13 => "x13",
            14 => "x14",
            15 => "x15",
            16 => "x16",
            17 => "x17",
            18 => "x18",
            19 => "x19",
            20 => "x20",
            21 => "x21",
            22 => "x22",
            23 => "x23",
            24 => "x24",
            25 => "x25",
            26 => "x26",
            27 => "x27",
            28 => "x28",
            29 => "fp",
            30 => "lr",
            31 => "sp",
            _ => "???",
        }
    }
}

/// Information needed to generate a standalone harness.
pub struct HarnessInput<'a> {
    /// Raw JIT machine code bytes.
    pub code: &'a [u8],
    /// Entry point offset within the code buffer.
    pub entry_offset: usize,
    /// Output buffer size in bytes (sizeof the target type).
    pub output_size: usize,
    /// DWARF sections (if available).
    pub dwarf: Option<crate::jit_dwarf::JitDwarfSections>,
    /// CFG-MIR listing lines (for the source file).
    pub cfg_mir_lines: &'a [String],
    /// Name for the generated function symbol.
    pub function_name: &'a str,
    /// Allocation map (vreg → physical location).
    pub alloc_map: Option<&'a AllocationMap>,
}

/// Generate a standalone test harness.
///
/// Returns the path to the generated executable.
pub fn generate_harness(
    input: &HarnessInput,
    output_dir: &Path,
    base_name: &str,
) -> Result<std::path::PathBuf, HarnessError> {
    use object::write::{Object, Relocation, StandardSection, Symbol, SymbolSection};
    use object::{
        Architecture, BinaryFormat, Endianness, RelocationEncoding, RelocationFlags,
        RelocationKind, SectionKind, SymbolFlags, SymbolKind, SymbolScope,
    };

    std::fs::create_dir_all(output_dir).map_err(|e| HarnessError::Io("create output dir", e))?;

    // Write the CFG-MIR listing file (DWARF source)
    let listing_path = output_dir.join(format!("{base_name}.cfg-mir"));
    let listing_text = input.cfg_mir_lines.join("\n");
    std::fs::write(&listing_path, &listing_text)
        .map_err(|e| HarnessError::Io("write listing", e))?;

    // Build the object file with JIT code
    let obj_path = output_dir.join(format!("{base_name}.o"));
    build_object_file(input, &obj_path)?;

    // Write the C harness
    let c_path = output_dir.join(format!("{base_name}_main.c"));
    write_c_harness(input, &c_path)?;

    // Write allocation map (for lockstep debugger)
    if let Some(alloc_map) = input.alloc_map {
        let map_path = output_dir.join(format!("{base_name}.alloc.json"));
        alloc_map
            .write_json(&map_path)
            .map_err(|e| HarnessError::Io("write alloc map", e))?;
    }

    // Link: cc -o harness harness_main.c jit.o -lSystem
    let exe_path = output_dir.join(base_name);
    link_harness(&c_path, &obj_path, &exe_path)?;

    // Post-link: patch DWARF addresses with actual symbol address
    if let Some(dwarf) = &input.dwarf {
        if let Err(e) = patch_dwarf_addresses(&exe_path, dwarf, input.function_name) {
            eprintln!("[harness] warning: failed to patch DWARF: {e}");
        }
    }

    eprintln!("[harness] generated: {}", exe_path.display());
    eprintln!("[harness] listing:   {}", listing_path.display());
    eprintln!("[harness] usage:     {} <input-hex>", exe_path.display());
    eprintln!(
        "[harness] debug:     lldb {} -- <input-hex>",
        exe_path.display()
    );

    Ok(exe_path)
}

fn build_object_file(input: &HarnessInput, path: &Path) -> Result<(), HarnessError> {
    use object::write::{Object, Relocation, Symbol, SymbolSection};
    use object::{
        Architecture, BinaryFormat, Endianness, RelocationEncoding, RelocationFlags,
        RelocationKind, SectionKind, SymbolFlags, SymbolKind, SymbolScope,
    };

    let mut obj = Object::new(
        BinaryFormat::MachO,
        Architecture::Aarch64,
        Endianness::Little,
    );

    // Set macOS platform version (prevents "no platform load command" warning)
    let mut build_ver = object::write::MachOBuildVersion::default();
    build_ver.platform = object::macho::PLATFORM_MACOS;
    build_ver.minos = (14 << 16) | (0 << 8); // macOS 14.0
    build_ver.sdk = (14 << 16) | (0 << 8);
    obj.set_macho_build_version(build_ver);

    // Add .text section with JIT code
    let text_section = obj.section_id(object::write::StandardSection::Text);
    obj.append_section_data(text_section, input.code, 16);

    // Add the entry point symbol (global, so the C harness can call it)
    let symbol_name = input.function_name.to_string();
    let text_symbol = obj.add_symbol(Symbol {
        name: symbol_name.into_bytes(),
        value: input.entry_offset as u64,
        size: (input.code.len() - input.entry_offset) as u64,
        kind: SymbolKind::Text,
        scope: SymbolScope::Dynamic,
        weak: false,
        section: SymbolSection::Section(text_section),
        flags: SymbolFlags::None,
    });

    // Add DWARF sections with relocations
    if let Some(dwarf) = &input.dwarf {
        let mut debug_info_section_id = None;
        let mut debug_line_section_id = None;

        if !dwarf.debug_line.is_empty() {
            let sid = obj.add_section(Vec::new(), b"__debug_line".to_vec(), SectionKind::Debug);
            obj.append_section_data(sid, &dwarf.debug_line, 8);
            debug_line_section_id = Some(sid);
        }
        if !dwarf.debug_info.is_empty() {
            let sid = obj.add_section(Vec::new(), b"__debug_info".to_vec(), SectionKind::Debug);
            obj.append_section_data(sid, &dwarf.debug_info, 8);
            debug_info_section_id = Some(sid);
        }
        if !dwarf.debug_abbrev.is_empty() {
            let sid = obj.add_section(Vec::new(), b"__debug_abbrev".to_vec(), SectionKind::Debug);
            obj.append_section_data(sid, &dwarf.debug_abbrev, 1);
        }
        if !dwarf.debug_loc.is_empty() {
            let sid = obj.add_section(Vec::new(), b"__debug_loc".to_vec(), SectionKind::Debug);
            obj.append_section_data(sid, &dwarf.debug_loc, 1);
        }
        if !dwarf.debug_ranges.is_empty() {
            let sid = obj.add_section(Vec::new(), b"__debug_ranges".to_vec(), SectionKind::Debug);
            obj.append_section_data(sid, &dwarf.debug_ranges, 1);
        }

        // Note: we don't add relocations here because Mach-O requires
        // 8-byte aligned relocation targets and DWARF addresses aren't aligned.
        // Instead, we patch the DWARF addresses post-link (see patch_dwarf_addresses).
    }

    let data = obj.write().map_err(HarnessError::ObjectWrite)?;
    std::fs::write(path, data).map_err(|e| HarnessError::Io("write object", e))?;

    Ok(())
}

fn write_c_harness(input: &HarnessInput, path: &Path) -> Result<(), HarnessError> {
    let output_size = input.output_size;
    let func_name = input.function_name;

    let c_code = format!(
        r#"// Auto-generated test harness for kajit JIT code.
// Usage: ./{func_name} <input-hex>
// Example: ./{func_name} 8001

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

// DeserContext — must match kajit::context::DeserContext layout
typedef struct {{
    const uint8_t *cursor;
    const uint8_t *end;
    uint8_t *out_ptr;
    struct {{
        uint32_t code;
        uint32_t offset;
    }} error;
}} DeserContext;

// The JIT-compiled decoder function (linked from the .o file)
extern void {func_name}(uint8_t *output, DeserContext *ctx);

static int hex_digit(char c) {{
    if (c >= '0' && c <= '9') return c - '0';
    if (c >= 'a' && c <= 'f') return c - 'a' + 10;
    if (c >= 'A' && c <= 'F') return c - 'A' + 10;
    return -1;
}}

static size_t parse_hex(const char *s, uint8_t *buf, size_t max) {{
    size_t len = 0;
    while (*s && *(s+1) && len < max) {{
        int hi = hex_digit(*s);
        int lo = hex_digit(*(s+1));
        if (hi < 0 || lo < 0) break;
        buf[len++] = (uint8_t)((hi << 4) | lo);
        s += 2;
    }}
    return len;
}}

int main(int argc, char **argv) {{
    if (argc < 2) {{
        fprintf(stderr, "usage: %s <input-hex>\n", argv[0]);
        return 1;
    }}

    // Parse hex input
    uint8_t input[4096];
    size_t input_len = parse_hex(argv[1], input, sizeof(input));

    // Allocate output buffer
    uint8_t output[{output_size}];
    memset(output, 0, sizeof(output));

    // Set up context
    DeserContext ctx;
    memset(&ctx, 0, sizeof(ctx));
    ctx.cursor = input;
    ctx.end = input + input_len;
    ctx.out_ptr = output;

    // Call the JIT decoder
    {func_name}(output, &ctx);

    // Check for errors
    if (ctx.error.code != 0) {{
        fprintf(stderr, "error: code=%u offset=%u\n", ctx.error.code, ctx.error.offset);
        return 1;
    }}

    // Print output as hex
    for (size_t i = 0; i < {output_size}; i++) {{
        printf("%02x", output[i]);
    }}
    printf("\n");

    return 0;
}}
"#
    );

    std::fs::write(path, c_code).map_err(|e| HarnessError::Io("write C harness", e))?;
    Ok(())
}

/// Patch DWARF addresses in a linked binary.
///
/// After linking, we know the actual address of kajit_decode. We find the
/// symbol address with `nm`, then binary-patch the DWARF sections to replace
/// code_address=0 with the real address.
fn patch_dwarf_addresses(
    exe_path: &Path,
    dwarf: &crate::jit_dwarf::JitDwarfSections,
    function_name: &str,
) -> Result<(), HarnessError> {
    // Get the symbol address from the linked binary
    let output = std::process::Command::new("nm")
        .arg(exe_path)
        .output()
        .map_err(|e| HarnessError::Io("invoke nm", e))?;

    let nm_output = String::from_utf8_lossy(&output.stdout);
    let symbol_to_find = format!("_{function_name}");
    let addr = nm_output
        .lines()
        .find_map(|line| {
            let parts: Vec<&str> = line.split_whitespace().collect();
            if parts.len() >= 3 && parts[2] == symbol_to_find {
                u64::from_str_radix(parts[0], 16).ok()
            } else {
                None
            }
        })
        .ok_or_else(|| {
            HarnessError::Link(format!("symbol {symbol_to_find} not found in nm output"))
        })?;

    eprintln!("[harness] patching DWARF: {symbol_to_find} @ 0x{addr:x}");

    // Read the binary
    let mut binary =
        std::fs::read(exe_path).map_err(|e| HarnessError::Io("read binary for patch", e))?;

    // Find and patch zero addresses in DWARF sections.
    // We search for the pattern of 8 zero bytes at the relocation offsets
    // within the DWARF section data, which appears somewhere in the binary.
    let addr_bytes = addr.to_le_bytes();
    let zero_bytes = [0u8; 8];

    // For each relocation, find the DWARF section data in the binary and patch it
    for (section, reloc) in &dwarf.relocations {
        let section_data = match section {
            crate::jit_dwarf::DwarfSection::DebugInfo => &dwarf.debug_info,
            crate::jit_dwarf::DwarfSection::DebugLine => &dwarf.debug_line,
        };

        // Find where this section's data appears in the binary
        // We use a sliding window match on a unique prefix around the relocation offset
        let offset = reloc.offset as usize;
        if offset + 8 > section_data.len() {
            continue;
        }

        // Create a search pattern: bytes before the address + 8 zero bytes
        let context_start = offset.saturating_sub(4);
        let context_end = (offset + 8).min(section_data.len());
        let pattern = &section_data[context_start..context_end];

        // Find this pattern in the binary
        if let Some(pos) = find_bytes(&binary, pattern) {
            let addr_pos = pos + (offset - context_start);
            // Verify it's still zeros
            if binary[addr_pos..addr_pos + 8] == zero_bytes {
                binary[addr_pos..addr_pos + 8].copy_from_slice(&addr_bytes);
                eprintln!(
                    "[harness]   patched {:?} offset {} → 0x{:x}",
                    section, reloc.offset, addr
                );
            }
        }
    }

    // Write back
    std::fs::write(exe_path, &binary).map_err(|e| HarnessError::Io("write patched binary", e))?;

    // Re-sign the binary (macOS code signing)
    let _ = std::process::Command::new("codesign")
        .args(["--force", "--sign", "-"])
        .arg(exe_path)
        .output();

    Ok(())
}

/// Find a byte pattern in a haystack.
fn find_bytes(haystack: &[u8], needle: &[u8]) -> Option<usize> {
    haystack.windows(needle.len()).position(|w| w == needle)
}

fn link_harness(c_path: &Path, obj_path: &Path, exe_path: &Path) -> Result<(), HarnessError> {
    let output = std::process::Command::new("cc")
        .arg("-g") // keep debug info
        .arg("-O0") // no optimization (so DWARF is accurate)
        .arg("-o")
        .arg(exe_path)
        .arg(c_path)
        .arg(obj_path)
        .output()
        .map_err(|e| HarnessError::Io("invoke cc", e))?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Err(HarnessError::Link(stderr.into_owned()));
    }

    Ok(())
}

#[derive(Debug)]
pub enum HarnessError {
    Io(&'static str, std::io::Error),
    ObjectWrite(object::write::Error),
    Link(String),
}

impl std::fmt::Display for HarnessError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            HarnessError::Io(ctx, e) => write!(f, "{ctx}: {e}"),
            HarnessError::ObjectWrite(e) => write!(f, "object write: {e}"),
            HarnessError::Link(msg) => write!(f, "link failed: {msg}"),
        }
    }
}

impl std::error::Error for HarnessError {}
