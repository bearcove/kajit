use std::path::Path;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TargetArch {
    Aarch64,
    X86_64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DwarfRelocation {
    pub offset: u32,
    pub addend: i64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DwarfSection {
    DebugInfo,
    DebugLine,
    DebugAranges,
}

#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct DwarfSections {
    pub debug_line: Vec<u8>,
    pub debug_abbrev: Vec<u8>,
    pub debug_info: Vec<u8>,
    pub debug_aranges: Vec<u8>,
    pub debug_loc: Vec<u8>,
    pub debug_ranges: Vec<u8>,
    pub relocations: Vec<(DwarfSection, DwarfRelocation)>,
}

#[derive(Debug, Clone)]
pub struct IntrinsicCallSite {
    pub code_offset: usize,
    pub baked_addr: u64,
    pub symbol_name: String,
}

#[derive(Debug, Clone)]
pub struct ExternAddrReloc {
    pub code_offset: usize,
    pub symbol: String,
}

pub struct ObjectInput<'a> {
    pub target_arch: TargetArch,
    pub code: &'a [u8],
    pub entry_offset: usize,
    pub function_name: &'a str,
    pub dwarf: Option<&'a DwarfSections>,
    pub intrinsic_calls: &'a [IntrinsicCallSite],
    pub extern_addr_relocs: &'a [ExternAddrReloc],
}

#[derive(Debug)]
pub enum WaresError {
    Io(&'static str, std::io::Error),
    ObjectWrite(object::write::Error),
    UnsupportedTarget(&'static str),
    Link(String),
}

impl std::fmt::Display for WaresError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            WaresError::Io(ctx, e) => write!(f, "{ctx}: {e}"),
            WaresError::ObjectWrite(e) => write!(f, "object write: {e}"),
            WaresError::UnsupportedTarget(msg) => write!(f, "{msg}"),
            WaresError::Link(msg) => write!(f, "{msg}"),
        }
    }
}

impl std::error::Error for WaresError {}

pub fn write_object_file(input: &ObjectInput<'_>, path: &Path) -> Result<(), WaresError> {
    match input.target_arch {
        TargetArch::Aarch64 => write_aarch64_object_file(input, path),
        TargetArch::X86_64 => write_x64_object_file(input, path),
    }
}

pub struct PrintMainExecutableInput<'a> {
    pub object: ObjectInput<'a>,
}

pub fn write_print_main_executable(
    input: &PrintMainExecutableInput<'_>,
    output_dir: &Path,
    base_name: &str,
) -> Result<std::path::PathBuf, WaresError> {
    std::fs::create_dir_all(output_dir).map_err(|e| WaresError::Io("create output dir", e))?;

    let obj_path = output_dir.join(format!("{base_name}.o"));
    write_object_file(&input.object, &obj_path)?;

    let c_path = output_dir.join(format!("{base_name}_main.c"));
    write_print_main_wrapper(input.object.function_name, &c_path)?;

    let exe_path = output_dir.join(base_name);
    link_print_main_executable(input.object.target_arch, &c_path, &obj_path, &exe_path)?;

    Ok(exe_path)
}

fn write_aarch64_object_file(input: &ObjectInput<'_>, path: &Path) -> Result<(), WaresError> {
    use object::write::{Object, Symbol, SymbolSection};
    use object::{
        Architecture, BinaryFormat, Endianness, SectionKind, SymbolFlags, SymbolKind, SymbolScope,
    };

    let mut obj = Object::new(
        BinaryFormat::native_object(),
        Architecture::Aarch64,
        Endianness::Little,
    );

    #[cfg(target_os = "macos")]
    {
        let mut build_ver = object::write::MachOBuildVersion::default();
        build_ver.platform = object::macho::PLATFORM_MACOS;
        build_ver.minos = 14 << 16;
        build_ver.sdk = 14 << 16;
        obj.set_macho_build_version(build_ver);
    }

    let mut code = input.code.to_vec();
    let mut intrinsic_relocs: Vec<(usize, object::write::SymbolId)> = Vec::new();

    for site in input.intrinsic_calls {
        let sym_id = obj.add_symbol(Symbol {
            name: site.symbol_name.as_bytes().to_vec(),
            value: 0,
            size: 0,
            kind: SymbolKind::Text,
            scope: SymbolScope::Dynamic,
            weak: false,
            section: SymbolSection::Undefined,
            flags: SymbolFlags::None,
        });

        let off = site.code_offset;
        let adrp = 0x90000010u32;
        let add = 0x91000210u32;
        let nop = 0xD503201Fu32;
        code[off..off + 4].copy_from_slice(&adrp.to_le_bytes());
        code[off + 4..off + 8].copy_from_slice(&add.to_le_bytes());
        code[off + 8..off + 12].copy_from_slice(&nop.to_le_bytes());
        intrinsic_relocs.push((off, sym_id));
    }

    let mut extern_addr_reloc_entries: Vec<(usize, object::write::SymbolId)> = Vec::new();
    for reloc in input.extern_addr_relocs {
        let sym_id = obj.add_symbol(Symbol {
            name: reloc.symbol.as_bytes().to_vec(),
            value: 0,
            size: 0,
            kind: SymbolKind::Data,
            scope: SymbolScope::Dynamic,
            weak: false,
            section: SymbolSection::Undefined,
            flags: SymbolFlags::None,
        });

        let off = reloc.code_offset;
        let adrp = 0x90000010u32;
        let add = 0x91000210u32;
        let nop = 0xD503201Fu32;
        code[off..off + 4].copy_from_slice(&adrp.to_le_bytes());
        code[off + 4..off + 8].copy_from_slice(&add.to_le_bytes());
        code[off + 8..off + 12].copy_from_slice(&nop.to_le_bytes());
        code[off + 12..off + 16].copy_from_slice(&nop.to_le_bytes());
        extern_addr_reloc_entries.push((off, sym_id));
    }

    let text_section = obj.section_id(object::write::StandardSection::Text);
    obj.append_section_data(text_section, &code, 16);

    for &(off, sym_id) in &intrinsic_relocs {
        add_page_relocations(&mut obj, text_section, off, sym_id);
    }

    for &(off, sym_id) in &extern_addr_reloc_entries {
        add_page_relocations(&mut obj, text_section, off, sym_id);
    }

    let text_symbol = obj.add_symbol(Symbol {
        name: input.function_name.as_bytes().to_vec(),
        value: input.entry_offset as u64,
        size: (input.code.len() - input.entry_offset) as u64,
        kind: SymbolKind::Text,
        scope: SymbolScope::Dynamic,
        weak: false,
        section: SymbolSection::Section(text_section),
        flags: SymbolFlags::None,
    });

    if let Some(dwarf) = input.dwarf {
        let mut debug_info_section_id = None;
        let mut debug_line_section_id = None;

        if !dwarf.debug_line.is_empty() {
            let sid = obj.add_section(
                dwarf_segment_name(),
                dwarf_debug_section_name("debug_line"),
                SectionKind::Debug,
            );
            obj.append_section_data(sid, &dwarf.debug_line, 1);
            debug_line_section_id = Some(sid);
        }
        if !dwarf.debug_info.is_empty() {
            let sid = obj.add_section(
                dwarf_segment_name(),
                dwarf_debug_section_name("debug_info"),
                SectionKind::Debug,
            );
            obj.append_section_data(sid, &dwarf.debug_info, 1);
            debug_info_section_id = Some(sid);
        }
        if !dwarf.debug_abbrev.is_empty() {
            let sid = obj.add_section(
                dwarf_segment_name(),
                dwarf_debug_section_name("debug_abbrev"),
                SectionKind::Debug,
            );
            obj.append_section_data(sid, &dwarf.debug_abbrev, 1);
        }
        let mut debug_aranges_section_id = None;
        if !dwarf.debug_aranges.is_empty() {
            let sid = obj.add_section(
                dwarf_segment_name(),
                dwarf_debug_section_name("debug_aranges"),
                SectionKind::Debug,
            );
            obj.append_section_data(sid, &dwarf.debug_aranges, 1);
            debug_aranges_section_id = Some(sid);
        }

        for (section, reloc) in &dwarf.relocations {
            let target_section = match section {
                DwarfSection::DebugInfo => debug_info_section_id,
                DwarfSection::DebugLine => debug_line_section_id,
                DwarfSection::DebugAranges => debug_aranges_section_id,
            };
            if let Some(sid) = target_section {
                obj.add_relocation(
                    sid,
                    dwarf_text_relocation(text_symbol, reloc, input.entry_offset),
                )
                .map_err(WaresError::ObjectWrite)?;
            }
        }
    }

    let data = obj.write().map_err(WaresError::ObjectWrite)?;
    std::fs::write(path, data).map_err(|e| WaresError::Io("write object", e))?;
    Ok(())
}

fn write_x64_object_file(input: &ObjectInput<'_>, path: &Path) -> Result<(), WaresError> {
    use object::write::{Object, Symbol, SymbolSection};
    use object::{
        Architecture, BinaryFormat, Endianness, SectionKind, SymbolFlags, SymbolKind, SymbolScope,
    };

    if !input.intrinsic_calls.is_empty() || !input.extern_addr_relocs.is_empty() {
        return Err(WaresError::UnsupportedTarget(
            "x86_64 object writing does not support external relocations yet",
        ));
    }

    let mut obj = Object::new(
        BinaryFormat::native_object(),
        Architecture::X86_64,
        Endianness::Little,
    );

    #[cfg(target_os = "macos")]
    {
        let mut build_ver = object::write::MachOBuildVersion::default();
        build_ver.platform = object::macho::PLATFORM_MACOS;
        build_ver.minos = 14 << 16;
        build_ver.sdk = 14 << 16;
        obj.set_macho_build_version(build_ver);
    }

    let text_section = obj.section_id(object::write::StandardSection::Text);
    obj.append_section_data(text_section, input.code, 16);

    let text_symbol = obj.add_symbol(Symbol {
        name: input.function_name.as_bytes().to_vec(),
        value: input.entry_offset as u64,
        size: (input.code.len() - input.entry_offset) as u64,
        kind: SymbolKind::Text,
        scope: SymbolScope::Dynamic,
        weak: false,
        section: SymbolSection::Section(text_section),
        flags: SymbolFlags::None,
    });

    if let Some(dwarf) = input.dwarf {
        let mut debug_info_section_id = None;
        let mut debug_line_section_id = None;

        if !dwarf.debug_line.is_empty() {
            let sid = obj.add_section(
                dwarf_segment_name(),
                dwarf_debug_section_name("debug_line"),
                SectionKind::Debug,
            );
            obj.append_section_data(sid, &dwarf.debug_line, 1);
            debug_line_section_id = Some(sid);
        }
        if !dwarf.debug_info.is_empty() {
            let sid = obj.add_section(
                dwarf_segment_name(),
                dwarf_debug_section_name("debug_info"),
                SectionKind::Debug,
            );
            obj.append_section_data(sid, &dwarf.debug_info, 1);
            debug_info_section_id = Some(sid);
        }
        if !dwarf.debug_abbrev.is_empty() {
            let sid = obj.add_section(
                dwarf_segment_name(),
                dwarf_debug_section_name("debug_abbrev"),
                SectionKind::Debug,
            );
            obj.append_section_data(sid, &dwarf.debug_abbrev, 1);
        }
        let mut debug_aranges_section_id = None;
        if !dwarf.debug_aranges.is_empty() {
            let sid = obj.add_section(
                dwarf_segment_name(),
                dwarf_debug_section_name("debug_aranges"),
                SectionKind::Debug,
            );
            obj.append_section_data(sid, &dwarf.debug_aranges, 1);
            debug_aranges_section_id = Some(sid);
        }

        for (section, reloc) in &dwarf.relocations {
            let target_section = match section {
                DwarfSection::DebugInfo => debug_info_section_id,
                DwarfSection::DebugLine => debug_line_section_id,
                DwarfSection::DebugAranges => debug_aranges_section_id,
            };
            if let Some(sid) = target_section {
                obj.add_relocation(
                    sid,
                    dwarf_text_relocation(text_symbol, reloc, input.entry_offset),
                )
                .map_err(WaresError::ObjectWrite)?;
            }
        }
    }

    let data = obj.write().map_err(WaresError::ObjectWrite)?;
    std::fs::write(path, data).map_err(|e| WaresError::Io("write object", e))?;
    Ok(())
}

fn add_page_relocations(
    obj: &mut object::write::Object<'_>,
    text_section: object::write::SectionId,
    off: usize,
    sym_id: object::write::SymbolId,
) {
    use object::RelocationFlags;
    use object::write::Relocation;

    #[cfg(target_os = "macos")]
    {
        obj.add_relocation(
            text_section,
            Relocation {
                offset: off as u64,
                symbol: sym_id,
                flags: RelocationFlags::MachO {
                    r_type: object::macho::ARM64_RELOC_PAGE21,
                    r_pcrel: true,
                    r_length: 2,
                },
                addend: 0,
            },
        )
        .expect("adrp relocation");

        obj.add_relocation(
            text_section,
            Relocation {
                offset: (off + 4) as u64,
                symbol: sym_id,
                flags: RelocationFlags::MachO {
                    r_type: object::macho::ARM64_RELOC_PAGEOFF12,
                    r_pcrel: false,
                    r_length: 2,
                },
                addend: 0,
            },
        )
        .expect("add relocation");
    }

    #[cfg(target_os = "linux")]
    {
        obj.add_relocation(
            text_section,
            Relocation {
                offset: off as u64,
                symbol: sym_id,
                flags: RelocationFlags::Elf {
                    r_type: object::elf::R_AARCH64_ADR_PREL_PG_HI21,
                },
                addend: 0,
            },
        )
        .expect("adrp relocation");

        obj.add_relocation(
            text_section,
            Relocation {
                offset: (off + 4) as u64,
                symbol: sym_id,
                flags: RelocationFlags::Elf {
                    r_type: object::elf::R_AARCH64_ADD_ABS_LO12_NC,
                },
                addend: 0,
            },
        )
        .expect("add relocation");
    }
}

fn dwarf_segment_name() -> Vec<u8> {
    #[cfg(target_os = "macos")]
    {
        b"__DWARF".to_vec()
    }
    #[cfg(target_os = "linux")]
    {
        Vec::new()
    }
}

fn dwarf_debug_section_name(name: &str) -> Vec<u8> {
    #[cfg(target_os = "macos")]
    {
        format!("__{name}").into_bytes()
    }
    #[cfg(target_os = "linux")]
    {
        format!(".{name}").into_bytes()
    }
}

fn dwarf_text_relocation(
    text_symbol: object::write::SymbolId,
    reloc: &DwarfRelocation,
    entry_offset: usize,
) -> object::write::Relocation {
    #[cfg(target_os = "macos")]
    {
        object::write::Relocation {
            offset: reloc.offset as u64,
            symbol: text_symbol,
            addend: reloc.addend + entry_offset as i64,
            flags: object::RelocationFlags::MachO {
                r_type: object::macho::ARM64_RELOC_UNSIGNED,
                r_pcrel: false,
                r_length: 3,
            },
        }
    }
    #[cfg(target_os = "linux")]
    {
        object::write::Relocation {
            offset: reloc.offset as u64,
            symbol: text_symbol,
            addend: reloc.addend + entry_offset as i64,
            flags: object::RelocationFlags::Generic {
                kind: object::RelocationKind::Absolute,
                encoding: object::RelocationEncoding::Generic,
                size: 64,
            },
        }
    }
}

fn write_print_main_wrapper(function_name: &str, path: &Path) -> Result<(), WaresError> {
    let c_code = format!(
        r#"#include <stdint.h>
#include <stdio.h>

extern uint64_t {function_name}(void);

int main(void) {{
    uint64_t value = {function_name}();
    printf("%llu\n", (unsigned long long)value);
    return 0;
}}
"#
    );

    std::fs::write(path, c_code).map_err(|e| WaresError::Io("write C wrapper", e))
}

fn link_print_main_executable(
    target_arch: TargetArch,
    c_path: &Path,
    obj_path: &Path,
    exe_path: &Path,
) -> Result<(), WaresError> {
    let mut command = std::process::Command::new("cc");
    command.arg("-O0");

    add_target_arch_flags(&mut command, target_arch)?;

    #[cfg(target_os = "macos")]
    {
        command.arg("-g");
    }

    #[cfg(target_os = "linux")]
    {
        command.arg("-g0");
    }

    command.arg("-o").arg(exe_path).arg(c_path).arg(obj_path);

    #[cfg(target_os = "macos")]
    {
        command.arg("-lSystem");
    }

    let output = command
        .output()
        .map_err(|e| WaresError::Io("invoke cc", e))?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Err(WaresError::Link(format!("link failed: {stderr}")));
    }

    Ok(())
}

fn add_target_arch_flags(
    command: &mut std::process::Command,
    target_arch: TargetArch,
) -> Result<(), WaresError> {
    #[cfg(target_os = "macos")]
    {
        command.arg("-arch");
        command.arg(match target_arch {
            TargetArch::Aarch64 => "arm64",
            TargetArch::X86_64 => "x86_64",
        });
        return Ok(());
    }

    #[cfg(not(target_os = "macos"))]
    {
        if matches!(
            (target_arch, std::env::consts::ARCH),
            (TargetArch::Aarch64, "aarch64") | (TargetArch::X86_64, "x86_64")
        ) {
            return Ok(());
        }

        Err(WaresError::UnsupportedTarget(
            "cross-arch standalone executable linking is only implemented on macOS right now",
        ))
    }
}
