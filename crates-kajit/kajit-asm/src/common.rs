use kajit_types::{SourceLocation, SourceMap, SourceMapEntry};

#[derive(Debug, Clone, Default)]
pub(crate) struct EmissionState {
    buf: Vec<u8>,
    source_map: SourceMap,
    current_location: SourceLocation,
    last_recorded_location: Option<SourceLocation>,
    labels: Vec<Option<u32>>,
}

impl EmissionState {
    pub(crate) fn current_offset(&self) -> u32 {
        self.buf.len() as u32
    }

    pub(crate) fn code_len(&self) -> usize {
        self.buf.len()
    }

    pub(crate) fn bytes(&self) -> &[u8] {
        &self.buf
    }

    pub(crate) fn emit_raw_bytes(&mut self, bytes: &[u8]) {
        self.buf.extend_from_slice(bytes);
    }

    pub(crate) fn source_map(&self) -> &[SourceMapEntry] {
        &self.source_map
    }

    pub(crate) fn set_source_location(&mut self, loc: SourceLocation) {
        self.current_location = loc;
    }

    pub(crate) fn current_source_location(&self) -> SourceLocation {
        self.current_location
    }

    pub(crate) fn maybe_record_source_map(&mut self) {
        if Some(self.current_location) != self.last_recorded_location {
            self.source_map.push(SourceMapEntry {
                offset: self.current_offset(),
                location: self.current_location,
            });
            self.last_recorded_location = Some(self.current_location);
        }
    }

    pub(crate) fn new_label(&mut self) -> u32 {
        let id = self.labels.len() as u32;
        self.labels.push(None);
        id
    }

    pub(crate) fn bind_label(&mut self, label: u32) -> Result<u32, Option<u32>> {
        let current_offset = self.current_offset();
        let Some(slot) = self.labels.get_mut(label as usize) else {
            return Err(None);
        };
        if let Some(existing_offset) = *slot {
            return Err(Some(existing_offset));
        }
        *slot = Some(current_offset);
        Ok(current_offset)
    }

    pub(crate) fn has_label(&self, label: u32) -> bool {
        self.labels.get(label as usize).is_some()
    }

    pub(crate) fn label_offset(&self, label: u32) -> Option<Option<u32>> {
        self.labels.get(label as usize).copied()
    }

    pub(crate) fn label_slot_mut(&mut self, label: u32) -> Option<&mut Option<u32>> {
        self.labels.get_mut(label as usize)
    }

    pub(crate) fn extend_buffer(&mut self, bytes: &[u8]) {
        self.buf.extend_from_slice(bytes);
    }

    pub(crate) fn patch_buffer(&mut self, offset: usize, bytes: &[u8]) {
        self.buf[offset..offset + bytes.len()].copy_from_slice(bytes);
    }

    pub(crate) fn read_u32_le(&self, offset: usize) -> u32 {
        u32::from_le_bytes([
            self.buf[offset],
            self.buf[offset + 1],
            self.buf[offset + 2],
            self.buf[offset + 3],
        ])
    }

    pub(crate) fn into_parts(self) -> (Vec<u8>, SourceMap) {
        (self.buf, self.source_map)
    }
}
