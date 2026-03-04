pub type RaMirInstIndex = u32;
pub type SourceMap = Vec<(u32, RaMirInstIndex)>;

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
    pub ra_mir_inst_index: RaMirInstIndex,
    pub bytes: Vec<u8>,
}

pub fn validate_source_map(source_map: &[(u32, RaMirInstIndex)]) -> Result<(), SourceMapError> {
    for window in source_map.windows(2) {
        let previous = window[0].0;
        let next = window[1].0;
        if next <= previous {
            return Err(SourceMapError::UnsortedOffsets { previous, next });
        }
    }
    Ok(())
}

pub fn encode_source_map_le(
    source_map: &[(u32, RaMirInstIndex)],
) -> Result<Vec<u8>, SourceMapError> {
    validate_source_map(source_map)?;
    let mut out = Vec::with_capacity(source_map.len() * 8);
    for (offset, inst) in source_map {
        out.extend_from_slice(&offset.to_le_bytes());
        out.extend_from_slice(&inst.to_le_bytes());
    }
    Ok(out)
}

pub fn decode_source_map_le(bytes: &[u8]) -> Result<SourceMap, SourceMapError> {
    if !bytes.len().is_multiple_of(8) {
        return Err(SourceMapError::TruncatedBinary { len: bytes.len() });
    }
    let mut out = Vec::with_capacity(bytes.len() / 8);
    for chunk in bytes.chunks_exact(8) {
        let offset = u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
        let inst = u32::from_le_bytes([chunk[4], chunk[5], chunk[6], chunk[7]]);
        out.push((offset, inst));
    }
    validate_source_map(&out)?;
    Ok(out)
}

pub fn build_trace(
    code: &[u8],
    source_map: &[(u32, RaMirInstIndex)],
) -> Result<Vec<TraceEntry>, TraceError> {
    validate_source_map(source_map)?;
    let mut out = Vec::with_capacity(source_map.len());
    for (index, (offset, inst)) in source_map.iter().copied().enumerate() {
        if offset as usize >= code.len() {
            return Err(TraceError::OffsetOutOfBounds {
                offset,
                code_len: code.len(),
            });
        }
        let next_offset = source_map
            .get(index + 1)
            .map(|(next, _)| *next as usize)
            .unwrap_or(code.len());
        if next_offset > code.len() {
            return Err(TraceError::OffsetOutOfBounds {
                offset: next_offset as u32,
                code_len: code.len(),
            });
        }
        let start = offset as usize;
        let bytes = code[start..next_offset].to_vec();
        out.push(TraceEntry {
            offset,
            ra_mir_inst_index: inst,
            bytes,
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
                "{:08x} inst={} bytes={}",
                entry.offset, entry.ra_mir_inst_index, hex
            )
        })
        .collect::<Vec<_>>()
        .join("\n")
}

pub fn format_trace(
    code: &[u8],
    source_map: &[(u32, RaMirInstIndex)],
) -> Result<String, TraceError> {
    let entries = build_trace(code, source_map)?;
    Ok(format_trace_entries(&entries))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn source_map_roundtrip_binary_codec() {
        let source_map = vec![(0, 12), (4, 13), (7, 14)];
        let encoded = encode_source_map_le(&source_map).unwrap();
        let decoded = decode_source_map_le(&encoded).unwrap();
        assert_eq!(decoded, source_map);
    }

    #[test]
    fn source_map_codec_rejects_invalid_input() {
        let err = decode_source_map_le(&[0u8; 3]).unwrap_err();
        assert!(matches!(err, SourceMapError::TruncatedBinary { len: 3 }));

        let err = encode_source_map_le(&[(4, 1), (4, 2)]).unwrap_err();
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
        let source_map = vec![(0, 10), (2, 11)];
        let trace = build_trace(&code, &source_map).unwrap();
        assert_eq!(trace[0].bytes, vec![0x90, 0x90]);
        assert_eq!(trace[1].bytes, vec![0x0f, 0x84, 0x34, 0x12, 0x00, 0x00]);

        let text = format_trace_entries(&trace);
        assert_eq!(
            text,
            "00000000 inst=10 bytes=9090\n00000002 inst=11 bytes=0f8434120000"
        );
    }

    #[test]
    fn trace_rejects_out_of_bounds_offsets() {
        let err = build_trace(&[0x90], &[(1, 0)]).unwrap_err();
        assert!(matches!(
            err,
            TraceError::OffsetOutOfBounds {
                offset: 1,
                code_len: 1
            }
        ));
    }
}
