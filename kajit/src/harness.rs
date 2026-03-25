//! Standalone test harness generator.
//!
//! Generates a Mach-O executable containing:
//! - The JIT-compiled decoder code in `.text`
//! - DWARF debug sections (`.debug_line`, `.debug_info`, `.debug_abbrev`)
//! - A C harness wrapper that sets up input/output and calls the decoder
//!
//! Usage: `kajit compile postcard u32 -s harness`
//! Produces: `harness_postcard_u32` executable + source listing

use std::path::Path;

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

    // Link: cc -o harness harness_main.c jit.o -lSystem
    let exe_path = output_dir.join(base_name);
    link_harness(&c_path, &obj_path, &exe_path)?;

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
    use object::write::{Object, Symbol, SymbolSection};
    use object::{
        Architecture, BinaryFormat, Endianness, SectionKind, SymbolFlags, SymbolKind, SymbolScope,
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
    // The object crate adds the Mach-O `_` prefix automatically for MachO format
    let symbol_name = input.function_name.to_string();
    obj.add_symbol(Symbol {
        name: symbol_name.into_bytes(),
        value: input.entry_offset as u64,
        size: (input.code.len() - input.entry_offset) as u64,
        kind: SymbolKind::Text,
        scope: SymbolScope::Dynamic,
        weak: false,
        section: SymbolSection::Section(text_section),
        flags: SymbolFlags::None,
    });

    // Add DWARF sections if available
    if let Some(dwarf) = &input.dwarf {
        if !dwarf.debug_line.is_empty() {
            let section_id =
                obj.add_section(Vec::new(), b"__debug_line".to_vec(), SectionKind::Debug);
            obj.append_section_data(section_id, &dwarf.debug_line, 1);
        }
        if !dwarf.debug_info.is_empty() {
            let section_id =
                obj.add_section(Vec::new(), b"__debug_info".to_vec(), SectionKind::Debug);
            obj.append_section_data(section_id, &dwarf.debug_info, 1);
        }
        if !dwarf.debug_abbrev.is_empty() {
            let section_id =
                obj.add_section(Vec::new(), b"__debug_abbrev".to_vec(), SectionKind::Debug);
            obj.append_section_data(section_id, &dwarf.debug_abbrev, 1);
        }
        if !dwarf.debug_loc.is_empty() {
            let section_id =
                obj.add_section(Vec::new(), b"__debug_loc".to_vec(), SectionKind::Debug);
            obj.append_section_data(section_id, &dwarf.debug_loc, 1);
        }
        if !dwarf.debug_ranges.is_empty() {
            let section_id =
                obj.add_section(Vec::new(), b"__debug_ranges".to_vec(), SectionKind::Debug);
            obj.append_section_data(section_id, &dwarf.debug_ranges, 1);
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
