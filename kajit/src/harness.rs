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

    // Build dSYM bundle: copy DWARF sections with patched addresses
    if let Some(dwarf) = &input.dwarf {
        if let Err(e) = build_dsym(&exe_path, dwarf, input.function_name, input.entry_offset) {
            eprintln!("[harness] warning: dSYM creation failed: {e}");
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

        // Mach-O DWARF sections must be in the __DWARF segment
        let dwarf_segment = b"__DWARF".to_vec();
        if !dwarf.debug_line.is_empty() {
            let sid = obj.add_section(
                dwarf_segment.clone(),
                b"__debug_line".to_vec(),
                SectionKind::Debug,
            );
            obj.append_section_data(sid, &dwarf.debug_line, 1);
            debug_line_section_id = Some(sid);
        }
        if !dwarf.debug_info.is_empty() {
            let sid = obj.add_section(
                dwarf_segment.clone(),
                b"__debug_info".to_vec(),
                SectionKind::Debug,
            );
            obj.append_section_data(sid, &dwarf.debug_info, 1);
            debug_info_section_id = Some(sid);
        }
        if !dwarf.debug_abbrev.is_empty() {
            let sid = obj.add_section(
                dwarf_segment.clone(),
                b"__debug_abbrev".to_vec(),
                SectionKind::Debug,
            );
            obj.append_section_data(sid, &dwarf.debug_abbrev, 1);
        }

        // Add relocations so the linker/dsymutil fixes up DWARF addresses
        for (section, reloc) in &dwarf.relocations {
            let target_section = match section {
                crate::jit_dwarf::DwarfSection::DebugInfo => debug_info_section_id,
                crate::jit_dwarf::DwarfSection::DebugLine => debug_line_section_id,
            };
            if let Some(sid) = target_section {
                obj.add_relocation(
                    sid,
                    Relocation {
                        offset: reloc.offset as u64,
                        symbol: text_symbol,
                        addend: reloc.addend + input.entry_offset as i64,
                        flags: RelocationFlags::MachO {
                            r_type: object::macho::ARM64_RELOC_UNSIGNED,
                            r_pcrel: false,
                            r_length: 3, // 8 bytes (2^3)
                        },
                    },
                )
                .map_err(HarnessError::ObjectWrite)?;
            }
        }
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

/// Build a dSYM bundle by hand: read UUID from exe, patch DWARF addresses,
/// write a Mach-O with DWARF into the dSYM directory structure.
fn build_dsym(
    exe_path: &Path,
    dwarf: &crate::jit_dwarf::JitDwarfSections,
    function_name: &str,
    entry_offset: usize,
) -> Result<(), HarnessError> {
    use object::read::{Object, ObjectSection, ObjectSegment, ObjectSymbol};

    let exe_data = std::fs::read(exe_path).map_err(|e| HarnessError::Io("read exe for dSYM", e))?;
    let exe_obj = object::read::File::parse(&*exe_data)
        .map_err(|e| HarnessError::Link(format!("parse exe: {e}")))?;

    // Get UUID
    let uuid = exe_obj.mach_uuid().ok().flatten().unwrap_or([0u8; 16]);

    // Get symbol address
    let mangled = format!("_{function_name}");
    let symbol_addr = exe_obj
        .symbols()
        .find(|s| s.name() == Ok(&mangled))
        .map(|s| s.address())
        .ok_or_else(|| HarnessError::Link(format!("symbol {mangled} not found for dSYM")))?;

    // Get __TEXT segment address range (needed for LLDB address resolution)
    let (text_vmaddr, text_vmsize) = exe_obj
        .segments()
        .find(|s| s.name() == Ok(Some("__TEXT")))
        .map(|s| (s.address(), s.size()))
        .unwrap_or((symbol_addr, 0x10000)); // fallback: use symbol addr + generous size

    drop(exe_obj);

    eprintln!(
        "[harness] building dSYM: {} @ 0x{:x}, uuid={}",
        mangled,
        symbol_addr,
        uuid.iter().map(|b| format!("{b:02X}")).collect::<String>()
    );

    // Patch DWARF: copy sections and fix addresses at relocation offsets
    let addr_bytes = symbol_addr.to_le_bytes();

    let mut debug_info = dwarf.debug_info.clone();
    let mut debug_line = dwarf.debug_line.clone();

    for (section, reloc) in &dwarf.relocations {
        let data = match section {
            crate::jit_dwarf::DwarfSection::DebugInfo => &mut debug_info,
            crate::jit_dwarf::DwarfSection::DebugLine => &mut debug_line,
        };
        let offset = reloc.offset as usize;
        if offset + 8 <= data.len() {
            data[offset..offset + 8].copy_from_slice(&addr_bytes);
        }
    }

    // Build the dSYM Mach-O by hand (need LC_UUID which the object crate can't emit)
    let dsym_data = build_dsym_macho(
        &uuid,
        &debug_info,
        &debug_line,
        &dwarf.debug_abbrev,
        text_vmaddr,
        text_vmsize,
    );

    // Write dSYM bundle
    let dsym_dir = exe_path.with_extension("dSYM");
    let dwarf_dir = dsym_dir.join("Contents/Resources/DWARF");
    std::fs::create_dir_all(&dwarf_dir).map_err(|e| HarnessError::Io("create dSYM dir", e))?;

    let dsym_file = dwarf_dir.join(exe_path.file_name().unwrap());
    std::fs::write(&dsym_file, &dsym_data).map_err(|e| HarnessError::Io("write dSYM Mach-O", e))?;

    // Write Info.plist
    let plist = format!(
        r#"<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>CFBundleDevelopmentRegion</key>
    <string>English</string>
    <key>CFBundleIdentifier</key>
    <string>com.kajit.harness.{}</string>
    <key>CFBundleInfoDictionaryVersion</key>
    <string>6.0</string>
    <key>CFBundlePackageType</key>
    <string>dSYM</string>
    <key>CFBundleVersion</key>
    <string>1</string>
</dict>
</plist>"#,
        exe_path.file_stem().unwrap().to_str().unwrap()
    );
    let plist_path = dsym_dir.join("Contents/Info.plist");
    std::fs::write(&plist_path, plist).map_err(|e| HarnessError::Io("write Info.plist", e))?;

    eprintln!("[harness] created dSYM: {}", dsym_dir.display());
    Ok(())
}

/// Build a dSYM Mach-O: header + LC_UUID + LC_SEGMENT_64(__TEXT) + LC_SEGMENT_64(__DWARF).
///
/// The __TEXT segment is a stub (no file data) that tells LLDB the virtual
/// address range of the code, so it can resolve addresses to compile units.
fn build_dsym_macho(
    uuid: &[u8; 16],
    debug_info: &[u8],
    debug_line: &[u8],
    debug_abbrev: &[u8],
    text_vmaddr: u64,
    text_vmsize: u64,
) -> Vec<u8> {
    // Count DWARF sections
    let mut dwarf_nsects = 0u32;
    if !debug_info.is_empty() {
        dwarf_nsects += 1;
    }
    if !debug_line.is_empty() {
        dwarf_nsects += 1;
    }
    if !debug_abbrev.is_empty() {
        dwarf_nsects += 1;
    }

    // Sizes
    let header_size = 32u32; // mach_header_64
    let uuid_cmd_size = 24u32; // LC_UUID
    let segment_cmd_size = 72u32; // LC_SEGMENT_64 (without sections)
    let section_size = 80u32; // section_64 per section
    let text_nsects = 1u32; // __text section stub
    let text_segment_size = segment_cmd_size + text_nsects * section_size;
    let dwarf_segment_size = segment_cmd_size + dwarf_nsects * section_size;
    let ncmds = 3u32; // LC_UUID + LC_SEGMENT_64(__TEXT) + LC_SEGMENT_64(__DWARF)
    let load_cmds_size = uuid_cmd_size + text_segment_size + dwarf_segment_size;
    let header_and_cmds = header_size + load_cmds_size;

    // Align data start to 8 bytes
    let data_start = (header_and_cmds + 7) & !7;
    let padding = data_start - header_and_cmds;

    // Layout DWARF sections sequentially after headers
    let mut section_offsets = Vec::new();
    let mut offset = data_start;
    for data in [debug_info, debug_line, debug_abbrev] {
        if !data.is_empty() {
            section_offsets.push((offset, data.len() as u32));
            offset += data.len() as u32;
        }
    }
    let total_data = offset - data_start;

    let mut out = Vec::with_capacity(offset as usize);

    // --- Mach-O header (mach_header_64) ---
    out.extend_from_slice(&0xFEEDFACFu32.to_le_bytes()); // magic (MH_MAGIC_64)
    out.extend_from_slice(&(12u32 | 0x01000000).to_le_bytes()); // cputype: CPU_TYPE_ARM64
    out.extend_from_slice(&0u32.to_le_bytes()); // cpusubtype: ALL
    out.extend_from_slice(&0x0Au32.to_le_bytes()); // filetype: MH_DSYM
    out.extend_from_slice(&ncmds.to_le_bytes()); // ncmds
    out.extend_from_slice(&load_cmds_size.to_le_bytes()); // sizeofcmds
    out.extend_from_slice(&0u32.to_le_bytes()); // flags
    out.extend_from_slice(&0u32.to_le_bytes()); // reserved

    // --- LC_UUID ---
    out.extend_from_slice(&0x1Bu32.to_le_bytes()); // cmd: LC_UUID
    out.extend_from_slice(&uuid_cmd_size.to_le_bytes()); // cmdsize
    out.extend_from_slice(uuid); // 16 bytes UUID

    // --- LC_SEGMENT_64 (__TEXT) — stub for address resolution ---
    out.extend_from_slice(&0x19u32.to_le_bytes()); // cmd: LC_SEGMENT_64
    out.extend_from_slice(&text_segment_size.to_le_bytes()); // cmdsize
    let mut text_segname = [0u8; 16];
    text_segname[..6].copy_from_slice(b"__TEXT");
    out.extend_from_slice(&text_segname); // segname
    out.extend_from_slice(&text_vmaddr.to_le_bytes()); // vmaddr
    out.extend_from_slice(&text_vmsize.to_le_bytes()); // vmsize
    out.extend_from_slice(&0u64.to_le_bytes()); // fileoff (no file data)
    out.extend_from_slice(&0u64.to_le_bytes()); // filesize (no file data)
    out.extend_from_slice(&5i32.to_le_bytes()); // maxprot: VM_PROT_READ | VM_PROT_EXECUTE
    out.extend_from_slice(&5i32.to_le_bytes()); // initprot: VM_PROT_READ | VM_PROT_EXECUTE
    out.extend_from_slice(&text_nsects.to_le_bytes()); // nsects
    out.extend_from_slice(&0u32.to_le_bytes()); // flags

    // __text section stub (no file data, just address range)
    let mut text_sectname = [0u8; 16];
    text_sectname[..6].copy_from_slice(b"__text");
    out.extend_from_slice(&text_sectname); // sectname
    out.extend_from_slice(&text_segname); // segname
    out.extend_from_slice(&text_vmaddr.to_le_bytes()); // addr
    out.extend_from_slice(&text_vmsize.to_le_bytes()); // size
    out.extend_from_slice(&0u32.to_le_bytes()); // offset (no file data)
    out.extend_from_slice(&0u32.to_le_bytes()); // align
    out.extend_from_slice(&0u32.to_le_bytes()); // reloff
    out.extend_from_slice(&0u32.to_le_bytes()); // nreloc
    out.extend_from_slice(&0x80000400u32.to_le_bytes()); // flags: S_REGULAR | S_ATTR_PURE_INSTRUCTIONS | S_ATTR_SOME_INSTRUCTIONS
    out.extend_from_slice(&0u32.to_le_bytes()); // reserved1
    out.extend_from_slice(&0u32.to_le_bytes()); // reserved2
    out.extend_from_slice(&0u32.to_le_bytes()); // reserved3

    // --- LC_SEGMENT_64 (__DWARF) ---
    out.extend_from_slice(&0x19u32.to_le_bytes()); // cmd: LC_SEGMENT_64
    out.extend_from_slice(&dwarf_segment_size.to_le_bytes());
    let mut segname = [0u8; 16];
    segname[..7].copy_from_slice(b"__DWARF");
    out.extend_from_slice(&segname);
    out.extend_from_slice(&0u64.to_le_bytes()); // vmaddr
    out.extend_from_slice(&0u64.to_le_bytes()); // vmsize
    out.extend_from_slice(&(data_start as u64).to_le_bytes()); // fileoff
    out.extend_from_slice(&(total_data as u64).to_le_bytes()); // filesize
    out.extend_from_slice(&0i32.to_le_bytes()); // maxprot
    out.extend_from_slice(&0i32.to_le_bytes()); // initprot
    out.extend_from_slice(&dwarf_nsects.to_le_bytes()); // nsects
    out.extend_from_slice(&0u32.to_le_bytes()); // flags

    // --- section_64 entries ---
    let section_names: Vec<&[u8]> = {
        let mut names = Vec::new();
        if !debug_info.is_empty() {
            names.push(b"__debug_info" as &[u8]);
        }
        if !debug_line.is_empty() {
            names.push(b"__debug_line" as &[u8]);
        }
        if !debug_abbrev.is_empty() {
            names.push(b"__debug_abbrev" as &[u8]);
        }
        names
    };

    for (i, (name, &(off, size))) in section_names.iter().zip(section_offsets.iter()).enumerate() {
        let mut sectname = [0u8; 16];
        let len = name.len().min(16);
        sectname[..len].copy_from_slice(&name[..len]);
        out.extend_from_slice(&sectname);
        out.extend_from_slice(&segname); // segname: __DWARF
        out.extend_from_slice(&0u64.to_le_bytes()); // addr
        out.extend_from_slice(&(size as u64).to_le_bytes()); // size
        out.extend_from_slice(&off.to_le_bytes()); // offset
        out.extend_from_slice(&0u32.to_le_bytes()); // align
        out.extend_from_slice(&0u32.to_le_bytes()); // reloff
        out.extend_from_slice(&0u32.to_le_bytes()); // nreloc
        out.extend_from_slice(&0x02000000u32.to_le_bytes()); // flags: S_REGULAR | S_ATTR_DEBUG
        out.extend_from_slice(&0u32.to_le_bytes()); // reserved1
        out.extend_from_slice(&0u32.to_le_bytes()); // reserved2
        out.extend_from_slice(&0u32.to_le_bytes()); // reserved3 (padding for 64-bit)
    }

    // Padding
    out.extend(std::iter::repeat_n(0u8, padding as usize));

    // Section data
    if !debug_info.is_empty() {
        out.extend_from_slice(debug_info);
    }
    if !debug_line.is_empty() {
        out.extend_from_slice(debug_line);
    }
    if !debug_abbrev.is_empty() {
        out.extend_from_slice(debug_abbrev);
    }

    out
}

/// Patch the LC_UUID in a Mach-O binary.
/// LC_UUID has cmd=0x1B, cmdsize=24, followed by 16 bytes of UUID.
fn patch_macho_uuid(data: &mut [u8], uuid: &[u8; 16]) {
    const LC_UUID: u32 = 0x1b;
    // Walk load commands to find LC_UUID
    // Mach-O 64 header: 32 bytes, then load commands
    if data.len() < 32 {
        return;
    }
    let ncmds = u32::from_le_bytes(data[16..20].try_into().unwrap()) as usize;
    let mut offset = 32; // past mach_header_64

    for _ in 0..ncmds {
        if offset + 8 > data.len() {
            break;
        }
        let cmd = u32::from_le_bytes(data[offset..offset + 4].try_into().unwrap());
        let cmdsize = u32::from_le_bytes(data[offset + 4..offset + 8].try_into().unwrap()) as usize;

        if cmd == LC_UUID && cmdsize >= 24 && offset + 24 <= data.len() {
            data[offset + 8..offset + 24].copy_from_slice(uuid);
            return;
        }

        offset += cmdsize;
    }
}

/// Find the address of a symbol in a linked binary.
fn find_symbol_address(exe_path: &Path, function_name: &str) -> Result<u64, HarnessError> {
    use object::read::{Object, ObjectSymbol};

    let binary = std::fs::read(exe_path).map_err(|e| HarnessError::Io("read binary", e))?;
    let obj = object::read::File::parse(&*binary)
        .map_err(|e| HarnessError::Link(format!("parse linked binary: {e}")))?;

    let mangled = format!("_{function_name}");
    obj.symbols()
        .find(|s| s.name() == Ok(&mangled))
        .map(|s| s.address())
        .ok_or_else(|| HarnessError::Link(format!("symbol {mangled} not found")))
}

/// Patch DWARF addresses in a .o file so dsymutil picks them up.
fn patch_object_dwarf(
    obj_path: &Path,
    dwarf: &crate::jit_dwarf::JitDwarfSections,
    symbol_addr: u64,
) -> Result<(), HarnessError> {
    use object::read::{Object, ObjectSection};

    let mut data = std::fs::read(obj_path).map_err(|e| HarnessError::Io("read .o for patch", e))?;

    let obj = object::read::File::parse(&*data)
        .map_err(|e| HarnessError::Link(format!("parse .o: {e}")))?;

    let debug_info_offset = obj
        .section_by_name("__debug_info")
        .and_then(|s| s.file_range())
        .map(|(off, _)| off);
    let debug_line_offset = obj
        .section_by_name("__debug_line")
        .and_then(|s| s.file_range())
        .map(|(off, _)| off);

    drop(obj); // release borrow on data

    eprintln!("[harness] patching .o DWARF: addr=0x{:x}", symbol_addr);
    let addr_bytes = symbol_addr.to_le_bytes();

    for (section, reloc) in &dwarf.relocations {
        let base = match section {
            crate::jit_dwarf::DwarfSection::DebugInfo => debug_info_offset,
            crate::jit_dwarf::DwarfSection::DebugLine => debug_line_offset,
        };
        let Some(base) = base else { continue };
        let offset = base as usize + reloc.offset as usize;
        if offset + 8 <= data.len() {
            data[offset..offset + 8].copy_from_slice(&addr_bytes);
            eprintln!(
                "[harness]   patched {:?} @ 0x{:x} (section+{})",
                section, offset, reloc.offset
            );
        }
    }

    std::fs::write(obj_path, &data).map_err(|e| HarnessError::Io("write patched .o", e))?;
    Ok(())
}

fn link_harness(c_path: &Path, obj_path: &Path, exe_path: &Path) -> Result<(), HarnessError> {
    let output = std::process::Command::new("cc")
        .arg("-g") // keep debug info
        .arg("-O0") // no optimization (so DWARF is accurate)
        .arg("-Wl,-no_deduplicate") // don't mess with our sections
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
