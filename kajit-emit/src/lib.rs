pub type FileId = u16;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct SourceLocation {
    pub file: FileId,
    pub line: u32,
    pub column: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SourceMapEntry {
    pub offset: u32,
    pub location: SourceLocation,
}

pub type SourceMap = Vec<SourceMapEntry>;

pub mod aarch64;
pub mod x64;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SourceMapError {
    TruncatedBinary { len: usize },
    UnsortedOffsets { previous: u32, next: u32 },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TraceError {
    InvalidSourceMap(SourceMapError),
    OffsetOutOfBounds { offset: u32, code_len: usize },
}

impl From<SourceMapError> for TraceError {
    fn from(value: SourceMapError) -> Self {
        Self::InvalidSourceMap(value)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TraceEntry {
    pub offset: u32,
    pub location: SourceLocation,
    pub bytes: Vec<u8>,
}

pub fn validate_source_map(source_map: &[SourceMapEntry]) -> Result<(), SourceMapError> {
    for window in source_map.windows(2) {
        let previous = window[0].offset;
        let next = window[1].offset;
        if next <= previous {
            return Err(SourceMapError::UnsortedOffsets { previous, next });
        }
    }
    Ok(())
}

pub fn encode_source_map_le(source_map: &[SourceMapEntry]) -> Result<Vec<u8>, SourceMapError> {
    validate_source_map(source_map)?;
    let mut out = Vec::with_capacity(source_map.len() * 14);
    for entry in source_map {
        out.extend_from_slice(&entry.offset.to_le_bytes());
        out.extend_from_slice(&entry.location.file.to_le_bytes());
        out.extend_from_slice(&entry.location.line.to_le_bytes());
        out.extend_from_slice(&entry.location.column.to_le_bytes());
    }
    Ok(out)
}

pub fn decode_source_map_le(bytes: &[u8]) -> Result<SourceMap, SourceMapError> {
    if !bytes.len().is_multiple_of(14) {
        return Err(SourceMapError::TruncatedBinary { len: bytes.len() });
    }
    let mut out = Vec::with_capacity(bytes.len() / 14);
    for chunk in bytes.chunks_exact(14) {
        out.push(SourceMapEntry {
            offset: u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]),
            location: SourceLocation {
                file: u16::from_le_bytes([chunk[4], chunk[5]]),
                line: u32::from_le_bytes([chunk[6], chunk[7], chunk[8], chunk[9]]),
                column: u32::from_le_bytes([chunk[10], chunk[11], chunk[12], chunk[13]]),
            },
        });
    }
    validate_source_map(&out)?;
    Ok(out)
}

pub fn build_trace(code: &[u8], source_map: &[SourceMapEntry]) -> Result<Vec<TraceEntry>, TraceError> {
    validate_source_map(source_map)?;
    let mut out = Vec::with_capacity(source_map.len());
    for (index, entry) in source_map.iter().copied().enumerate() {
        if entry.offset as usize >= code.len() {
            return Err(TraceError::OffsetOutOfBounds {
                offset: entry.offset,
                code_len: code.len(),
            });
        }
        let next_offset = source_map
            .get(index + 1)
            .map(|next| next.offset as usize)
            .unwrap_or(code.len());
        if next_offset > code.len() {
            return Err(TraceError::OffsetOutOfBounds {
                offset: next_offset as u32,
                code_len: code.len(),
            });
        }
        let start = entry.offset as usize;
        out.push(TraceEntry {
            offset: entry.offset,
            location: entry.location,
            bytes: code[start..next_offset].to_vec(),
        });
    }
    Ok(out)
}

pub fn format_trace_entries(entries: &[TraceEntry]) -> String {
    entries
        .iter()
        .map(|entry| {
            let hex = entry
                .bytes
                .iter()
                .map(|byte| format!("{byte:02x}"))
                .collect::<String>();
            format!(
                "{:08x} file={} line={} col={} bytes={}",
                entry.offset,
                entry.location.file,
                entry.location.line,
                entry.location.column,
                hex
            )
        })
        .collect::<Vec<_>>()
        .join("\n")
}

pub fn format_trace(code: &[u8], source_map: &[SourceMapEntry]) -> Result<String, TraceError> {
    let entries = build_trace(code, source_map)?;
    Ok(format_trace_entries(&entries))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn source_map_roundtrip_binary_codec() {
        let source_map = vec![
            SourceMapEntry {
                offset: 0,
                location: SourceLocation {
                    file: 1,
                    line: 12,
                    column: 4,
                },
            },
            SourceMapEntry {
                offset: 4,
                location: SourceLocation {
                    file: 1,
                    line: 13,
                    column: 9,
                },
            },
            SourceMapEntry {
                offset: 7,
                location: SourceLocation {
                    file: 2,
                    line: 14,
                    column: 3,
                },
            },
        ];
        let encoded = encode_source_map_le(&source_map).unwrap();
        let decoded = decode_source_map_le(&encoded).unwrap();
        assert_eq!(decoded, source_map);
    }

    #[test]
    fn source_map_codec_rejects_invalid_input() {
        let err = decode_source_map_le(&[0u8; 3]).unwrap_err();
        assert!(matches!(err, SourceMapError::TruncatedBinary { len: 3 }));

        let err = encode_source_map_le(&[
            SourceMapEntry {
                offset: 4,
                location: SourceLocation {
                    file: 1,
                    line: 1,
                    column: 1,
                },
            },
            SourceMapEntry {
                offset: 4,
                location: SourceLocation {
                    file: 1,
                    line: 1,
                    column: 2,
                },
            },
        ])
        .unwrap_err();
        assert!(matches!(
            err,
            SourceMapError::UnsortedOffsets {
                previous: 4,
                next: 4
            }
        ));
    }

    #[test]
    fn trace_builds_and_formats() {
        let code = vec![0x90, 0x90, 0x0f, 0x84, 0x34, 0x12, 0x00, 0x00];
        let source_map = vec![
            SourceMapEntry {
                offset: 0,
                location: SourceLocation {
                    file: 7,
                    line: 10,
                    column: 2,
                },
            },
            SourceMapEntry {
                offset: 2,
                location: SourceLocation {
                    file: 7,
                    line: 11,
                    column: 8,
                },
            },
        ];
        let trace = build_trace(&code, &source_map).unwrap();
        assert_eq!(trace[0].bytes, vec![0x90, 0x90]);
        assert_eq!(trace[1].bytes, vec![0x0f, 0x84, 0x34, 0x12, 0x00, 0x00]);

        let text = format_trace_entries(&trace);
        assert_eq!(
            text,
            "00000000 file=7 line=10 col=2 bytes=9090\n00000002 file=7 line=11 col=8 bytes=0f8434120000"
        );
    }

    #[test]
    fn trace_rejects_out_of_bounds_offsets() {
        let err = build_trace(
            &[0x90],
            &[SourceMapEntry {
                offset: 1,
                location: SourceLocation::default(),
            }],
        )
        .unwrap_err();
        assert!(matches!(
            err,
            TraceError::OffsetOutOfBounds {
                offset: 1,
                code_len: 1
            }
        ));
    }
}
