//! DWARF preparation utilities for JIT code.
//!
//! This module builds a minimal DWARF v4 `.debug_line` section from a source
//! map of `(code_offset, ra_mir_inst_index)` pairs.

/// Owned DWARF sections ready to be attached to the JIT ELF.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct JitDwarfSections {
    pub debug_line: Vec<u8>,
    pub debug_str: Vec<u8>,
    pub debug_line_str: Vec<u8>,
}

impl JitDwarfSections {
    pub fn is_empty(&self) -> bool {
        self.debug_line.is_empty() && self.debug_str.is_empty() && self.debug_line_str.is_empty()
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DwarfPrepError {
    InteriorNul {
        field: &'static str,
    },
    SourceMapNotStrictlyIncreasing {
        previous_offset: u32,
        next_offset: u32,
    },
    SourceOffsetOutOfBounds {
        offset: u32,
        code_size: u64,
    },
    CodeSizeOutOfBounds {
        code_size: u64,
    },
}

const DW_LNS_COPY: u8 = 1;
const DW_LNS_ADVANCE_PC: u8 = 2;
const DW_LNS_ADVANCE_LINE: u8 = 3;

const DW_LNE_END_SEQUENCE: u8 = 1;
const DW_LNE_SET_ADDRESS: u8 = 2;

const LINE_VERSION: u16 = 4;
const MIN_INSN_LEN: u8 = 1;
const MAX_OPS_PER_INSN: u8 = 1;
const DEFAULT_IS_STMT: u8 = 1;
const LINE_BASE: i8 = -5;
const LINE_RANGE: u8 = 14;
const OPCODE_BASE: u8 = 13;
const STANDARD_OPCODE_LENGTHS: [u8; 12] = [0, 1, 1, 1, 1, 0, 0, 0, 1, 0, 0, 1];

/// Build DWARF sections for one JIT function.
///
/// `source_map` entries are interpreted as:
/// - address: `code_address + code_offset`
/// - line: `ra_mir_inst_index + 1`
pub fn build_jit_dwarf_sections(
    code_address: u64,
    code_size: u64,
    source_map: &[(u32, u32)],
    file_name: &str,
    directory: Option<&str>,
) -> Result<JitDwarfSections, DwarfPrepError> {
    Ok(JitDwarfSections {
        debug_line: build_debug_line_section(
            code_address,
            code_size,
            source_map,
            file_name,
            directory,
        )?,
        debug_str: Vec::new(),
        debug_line_str: Vec::new(),
    })
}

pub fn build_debug_line_section(
    code_address: u64,
    code_size: u64,
    source_map: &[(u32, u32)],
    file_name: &str,
    directory: Option<&str>,
) -> Result<Vec<u8>, DwarfPrepError> {
    if code_size > u32::MAX as u64 {
        return Err(DwarfPrepError::CodeSizeOutOfBounds { code_size });
    }
    if file_name.as_bytes().contains(&0) {
        return Err(DwarfPrepError::InteriorNul { field: "file_name" });
    }
    if let Some(dir) = directory {
        if dir.as_bytes().contains(&0) {
            return Err(DwarfPrepError::InteriorNul { field: "directory" });
        }
    }

    for window in source_map.windows(2) {
        let previous = window[0].0;
        let next = window[1].0;
        if next <= previous {
            return Err(DwarfPrepError::SourceMapNotStrictlyIncreasing {
                previous_offset: previous,
                next_offset: next,
            });
        }
    }

    for (offset, _) in source_map {
        if (*offset as u64) > code_size {
            return Err(DwarfPrepError::SourceOffsetOutOfBounds {
                offset: *offset,
                code_size,
            });
        }
    }

    let mut header_body = Vec::new();
    header_body.push(MIN_INSN_LEN);
    header_body.push(MAX_OPS_PER_INSN);
    header_body.push(DEFAULT_IS_STMT);
    header_body.push(LINE_BASE as u8);
    header_body.push(LINE_RANGE);
    header_body.push(OPCODE_BASE);
    header_body.extend_from_slice(&STANDARD_OPCODE_LENGTHS);

    let has_dir = directory.is_some_and(|dir| !dir.is_empty());
    if let Some(dir) = directory.filter(|dir| !dir.is_empty()) {
        header_body.extend_from_slice(dir.as_bytes());
        header_body.push(0);
    }
    header_body.push(0); // end include_directories

    header_body.extend_from_slice(file_name.as_bytes());
    header_body.push(0);
    push_uleb128(&mut header_body, if has_dir { 1 } else { 0 });
    push_uleb128(&mut header_body, 0); // mtime
    push_uleb128(&mut header_body, 0); // size
    header_body.push(0); // end file_names

    let mut program = Vec::new();
    // Extended opcode: set absolute text address for sequence.
    program.push(0);
    push_uleb128(&mut program, 1 + 8);
    program.push(DW_LNE_SET_ADDRESS);
    program.extend_from_slice(&code_address.to_le_bytes());

    let mut current_offset = 0u64;
    let mut current_line = 1i64;

    for (offset, ra_mir_inst_index) in source_map {
        let offset = *offset as u64;
        let line = (*ra_mir_inst_index as i64) + 1;

        if offset > current_offset {
            program.push(DW_LNS_ADVANCE_PC);
            push_uleb128(&mut program, offset - current_offset);
            current_offset = offset;
        }

        let delta_line = line - current_line;
        if delta_line != 0 {
            program.push(DW_LNS_ADVANCE_LINE);
            push_sleb128(&mut program, delta_line);
            current_line = line;
        }

        program.push(DW_LNS_COPY);
    }

    if code_size > current_offset {
        program.push(DW_LNS_ADVANCE_PC);
        push_uleb128(&mut program, code_size - current_offset);
    }

    program.push(0);
    push_uleb128(&mut program, 1);
    program.push(DW_LNE_END_SEQUENCE);

    let header_length = header_body.len() as u32;
    let unit_length = 2u32 + 4u32 + header_length + (program.len() as u32);

    let mut section = Vec::with_capacity(4 + unit_length as usize);
    section.extend_from_slice(&unit_length.to_le_bytes());
    section.extend_from_slice(&LINE_VERSION.to_le_bytes());
    section.extend_from_slice(&header_length.to_le_bytes());
    section.extend_from_slice(&header_body);
    section.extend_from_slice(&program);
    section.shrink_to_fit();
    Ok(section)
}

fn push_uleb128(out: &mut Vec<u8>, mut value: u64) {
    loop {
        let mut byte = (value & 0x7f) as u8;
        value >>= 7;
        if value != 0 {
            byte |= 0x80;
        }
        out.push(byte);
        if value == 0 {
            break;
        }
    }
}

fn push_sleb128(out: &mut Vec<u8>, mut value: i64) {
    loop {
        let byte = (value as u8) & 0x7f;
        let sign_bit_set = (byte & 0x40) != 0;
        value >>= 7;
        let done = (value == 0 && !sign_bit_set) || (value == -1 && sign_bit_set);
        out.push(if done { byte } else { byte | 0x80 });
        if done {
            break;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parse_uleb(bytes: &[u8], i: &mut usize) -> u64 {
        let mut shift = 0u32;
        let mut out = 0u64;
        loop {
            let byte = bytes[*i];
            *i += 1;
            out |= ((byte & 0x7f) as u64) << shift;
            if (byte & 0x80) == 0 {
                return out;
            }
            shift += 7;
        }
    }

    fn parse_sleb(bytes: &[u8], i: &mut usize) -> i64 {
        let mut shift = 0u32;
        let mut out = 0i64;
        let mut byte;
        loop {
            byte = bytes[*i];
            *i += 1;
            out |= ((byte & 0x7f) as i64) << shift;
            shift += 7;
            if (byte & 0x80) == 0 {
                break;
            }
        }
        if shift < 64 && (byte & 0x40) != 0 {
            out |= !0i64 << shift;
        }
        out
    }

    fn parse_debug_line_rows(section: &[u8]) -> Vec<(u64, i64)> {
        let unit_length = u32::from_le_bytes(section[0..4].try_into().unwrap()) as usize;
        let version = u16::from_le_bytes(section[4..6].try_into().unwrap());
        assert_eq!(version, 4);
        let header_length = u32::from_le_bytes(section[6..10].try_into().unwrap()) as usize;
        let mut program_i = 10 + header_length;
        assert_eq!(section.len(), 4 + unit_length);

        let mut rows = Vec::new();
        let mut address = 0u64;
        let mut line = 1i64;

        while program_i < section.len() {
            let opcode = section[program_i];
            program_i += 1;
            match opcode {
                0 => {
                    let len = parse_uleb(section, &mut program_i) as usize;
                    let sub = section[program_i];
                    program_i += 1;
                    match sub {
                        DW_LNE_SET_ADDRESS => {
                            assert_eq!(len, 9);
                            address = u64::from_le_bytes(
                                section[program_i..program_i + 8].try_into().unwrap(),
                            );
                            program_i += 8;
                        }
                        DW_LNE_END_SEQUENCE => {
                            break;
                        }
                        _ => panic!("unexpected extended opcode {sub}"),
                    }
                }
                DW_LNS_ADVANCE_PC => {
                    address += parse_uleb(section, &mut program_i);
                }
                DW_LNS_ADVANCE_LINE => {
                    line += parse_sleb(section, &mut program_i);
                }
                DW_LNS_COPY => rows.push((address, line)),
                other => panic!("unexpected standard opcode {other}"),
            }
        }

        rows
    }

    #[test]
    fn debug_line_v4_header_has_max_ops_field() {
        let section = build_debug_line_section(0x3000, 0, &[], "decoder.ra", None).unwrap();
        // unit_length (4), version (2), header_length (4), then header body.
        assert_eq!(u16::from_le_bytes(section[4..6].try_into().unwrap()), 4);
        assert_eq!(section[10], MIN_INSN_LEN);
        assert_eq!(section[11], MAX_OPS_PER_INSN);
        assert_eq!(section[12], DEFAULT_IS_STMT);
    }

    #[test]
    fn debug_line_rows_match_source_map() {
        let section = build_debug_line_section(
            0x1000,
            12,
            &[(0, 0), (4, 3), (9, 7)],
            "decoder.ra",
            Some("jit"),
        )
        .unwrap();

        let rows = parse_debug_line_rows(&section);
        assert_eq!(rows, vec![(0x1000, 1), (0x1004, 4), (0x1009, 8)]);
    }

    #[test]
    fn debug_line_allows_empty_source_map() {
        let section = build_debug_line_section(0x2000, 5, &[], "decoder.ra", None).unwrap();
        let rows = parse_debug_line_rows(&section);
        assert!(rows.is_empty());
    }

    #[test]
    fn rejects_invalid_source_map_and_inputs() {
        let err = build_debug_line_section(0, 8, &[(4, 1), (4, 2)], "f", None).unwrap_err();
        assert!(matches!(
            err,
            DwarfPrepError::SourceMapNotStrictlyIncreasing {
                previous_offset: 4,
                next_offset: 4
            }
        ));

        let err = build_debug_line_section(0, 7, &[(8, 0)], "f", None).unwrap_err();
        assert!(matches!(
            err,
            DwarfPrepError::SourceOffsetOutOfBounds {
                offset: 8,
                code_size: 7
            }
        ));

        let err = build_debug_line_section(0, 8, &[], "bad\0file", None).unwrap_err();
        assert!(matches!(
            err,
            DwarfPrepError::InteriorNul { field: "file_name" }
        ));
    }
}
