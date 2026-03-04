use crate::{SourceLocation, SourceMap, SourceMapEntry};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Mem {
    pub base: u8,
    pub disp: i32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct LabelId(u32);

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FinalizedEmission {
    pub code: Vec<u8>,
    pub source_map: SourceMap,
}

impl FinalizedEmission {
    pub fn trace_entries(&self) -> Result<Vec<crate::TraceEntry>, crate::TraceError> {
        crate::build_trace(&self.code, &self.source_map)
    }

    pub fn trace_text(&self) -> Result<String, crate::TraceError> {
        crate::format_trace(&self.code, &self.source_map)
    }

    pub fn source_map_le(&self) -> Result<Vec<u8>, crate::SourceMapError> {
        crate::encode_source_map_le(&self.source_map)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Operand {
    Reg(u8),
    Mem(Mem),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EmitError {
    InvalidRegister {
        reg: u8,
    },
    LabelOutOfBounds {
        label: LabelId,
    },
    LabelAlreadyBound {
        label: LabelId,
        existing_offset: u32,
    },
    UnboundLabel {
        label: LabelId,
    },
    RelativeOutOfRange {
        at_offset: u32,
        target_offset: u32,
        delta: i64,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Fixup {
    disp_offset: u32,
    next_ip: u32,
    label: LabelId,
}

#[derive(Debug, Clone, Default)]
pub struct Emitter {
    buf: Vec<u8>,
    source_map: SourceMap,
    current_location: SourceLocation,
    last_recorded_location: Option<SourceLocation>,
    labels: Vec<Option<u32>>,
    fixups: Vec<Fixup>,
}

impl Emitter {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn current_offset(&self) -> u32 {
        self.buf.len() as u32
    }

    pub fn bytes(&self) -> &[u8] {
        &self.buf
    }

    pub fn source_map(&self) -> &[SourceMapEntry] {
        &self.source_map
    }

    pub fn set_source_location(&mut self, loc: SourceLocation) {
        self.current_location = loc;
    }

    fn maybe_record_source_map(&mut self) {
        if Some(self.current_location) != self.last_recorded_location {
            self.source_map.push(SourceMapEntry {
                offset: self.current_offset(),
                location: self.current_location,
            });
            self.last_recorded_location = Some(self.current_location);
        }
    }

    pub fn new_label(&mut self) -> LabelId {
        let id = LabelId(self.labels.len() as u32);
        self.labels.push(None);
        id
    }

    pub fn bind_label(&mut self, label: LabelId) -> Result<(), EmitError> {
        let current_offset = self.current_offset();
        let Some(slot) = self.labels.get_mut(label.0 as usize) else {
            return Err(EmitError::LabelOutOfBounds { label });
        };
        if let Some(existing_offset) = *slot {
            return Err(EmitError::LabelAlreadyBound {
                label,
                existing_offset,
            });
        }
        *slot = Some(current_offset);
        Ok(())
    }

    pub fn emit_with<F>(&mut self, f: F) -> Result<(), EmitError>
    where
        F: FnOnce(&mut Vec<u8>) -> Result<(), EmitError>,
    {
        let mut inst = Vec::new();
        f(&mut inst)?;
        self.maybe_record_source_map();
        self.buf.extend_from_slice(&inst);
        Ok(())
    }

    pub fn emit_bytes(&mut self, bytes: &[u8]) {
        self.maybe_record_source_map();
        self.buf.extend_from_slice(bytes);
    }

    fn ensure_label_exists(&self, label: LabelId) -> Result<(), EmitError> {
        if self.labels.get(label.0 as usize).is_some() {
            Ok(())
        } else {
            Err(EmitError::LabelOutOfBounds { label })
        }
    }

    pub fn emit_je_label(
        &mut self,
        label: LabelId,
    ) -> Result<(), EmitError> {
        self.ensure_label_exists(label)?;
        let start = self.current_offset();
        self.emit_with(|buf| encode_je_rel32(buf, 0))?;
        self.fixups.push(Fixup {
            disp_offset: start + 2,
            next_ip: self.current_offset(),
            label,
        });
        Ok(())
    }

    pub fn emit_jz_label(
        &mut self,
        label: LabelId,
    ) -> Result<(), EmitError> {
        self.emit_je_label(label)
    }

    pub fn emit_jnz_label(
        &mut self,
        label: LabelId,
    ) -> Result<(), EmitError> {
        self.ensure_label_exists(label)?;
        let start = self.current_offset();
        self.emit_with(|buf| encode_jnz_rel32(buf, 0))?;
        self.fixups.push(Fixup {
            disp_offset: start + 2,
            next_ip: self.current_offset(),
            label,
        });
        Ok(())
    }

    pub fn emit_jmp_label(
        &mut self,
        label: LabelId,
    ) -> Result<(), EmitError> {
        self.ensure_label_exists(label)?;
        let start = self.current_offset();
        self.emit_with(|buf| encode_jmp_rel32(buf, 0))?;
        self.fixups.push(Fixup {
            disp_offset: start + 1,
            next_ip: self.current_offset(),
            label,
        });
        Ok(())
    }

    pub fn emit_call_label(
        &mut self,
        label: LabelId,
    ) -> Result<(), EmitError> {
        self.ensure_label_exists(label)?;
        let start = self.current_offset();
        self.emit_with(|buf| encode_call_rel32(buf, 0))?;
        self.fixups.push(Fixup {
            disp_offset: start + 1,
            next_ip: self.current_offset(),
            label,
        });
        Ok(())
    }

    pub fn finalize(mut self) -> Result<FinalizedEmission, EmitError> {
        for fixup in self.fixups.iter().copied() {
            let target = self
                .labels
                .get(fixup.label.0 as usize)
                .ok_or(EmitError::LabelOutOfBounds { label: fixup.label })?
                .ok_or(EmitError::UnboundLabel { label: fixup.label })?;

            let delta = target as i64 - fixup.next_ip as i64;
            if delta < i32::MIN as i64 || delta > i32::MAX as i64 {
                return Err(EmitError::RelativeOutOfRange {
                    at_offset: fixup.disp_offset,
                    target_offset: target,
                    delta,
                });
            }

            let disp = (delta as i32).to_le_bytes();
            self.buf[fixup.disp_offset as usize..fixup.disp_offset as usize + 4]
                .copy_from_slice(&disp);
        }

        Ok(FinalizedEmission {
            code: self.buf,
            source_map: self.source_map,
        })
    }
}

fn check_reg(reg: u8) -> Result<u8, EmitError> {
    if reg <= 15 {
        Ok(reg)
    } else {
        Err(EmitError::InvalidRegister { reg })
    }
}

fn push_rex(buf: &mut Vec<u8>, w: bool, r: u8, x: u8, b: u8, force: bool) {
    let rex =
        0x40 | ((w as u8) << 3) | (((r >> 3) & 1) << 2) | (((x >> 3) & 1) << 1) | ((b >> 3) & 1);
    if force || rex != 0x40 {
        buf.push(rex);
    }
}

fn emit_modrm(buf: &mut Vec<u8>, reg_field: u8, rm: Operand) {
    match rm {
        Operand::Reg(rm_reg) => {
            let modrm = 0b11_000_000 | ((reg_field & 0x7) << 3) | (rm_reg & 0x7);
            buf.push(modrm);
        }
        Operand::Mem(mem) => {
            let base = mem.base;
            let base_low = base & 0x7;

            let (mod_bits, disp8, disp32) = if mem.disp == 0 && base_low != 0b101 {
                (0b00, None, None)
            } else if (-128..=127).contains(&mem.disp) {
                (0b01, Some(mem.disp as i8), None)
            } else {
                (0b10, None, Some(mem.disp))
            };

            let rm_field = if base_low == 0b100 { 0b100 } else { base_low };
            let modrm = (mod_bits << 6) | ((reg_field & 0x7) << 3) | rm_field;
            buf.push(modrm);

            if rm_field == 0b100 {
                // SIB with no index: scale=0, index=100, base=base_low
                buf.push((0b100 << 3) | base_low);
            }

            if let Some(disp8) = disp8 {
                buf.push(disp8 as u8);
            }
            if let Some(disp32) = disp32 {
                buf.extend_from_slice(&disp32.to_le_bytes());
            }
        }
    }
}

fn emit_op_rm(
    buf: &mut Vec<u8>,
    legacy_prefix: Option<u8>,
    opcodes: &[u8],
    rex_w: bool,
    reg_field: u8,
    rm: Operand,
    force_rex_for_rm8: bool,
) -> Result<(), EmitError> {
    let reg_field = check_reg(reg_field)?;
    let b = match rm {
        Operand::Reg(reg) => check_reg(reg)?,
        Operand::Mem(mem) => check_reg(mem.base)?,
    };

    if let Some(prefix) = legacy_prefix {
        buf.push(prefix);
    }

    let force = force_rex_for_rm8 && matches!(rm, Operand::Reg(r) if (4..=7).contains(&r));
    push_rex(buf, rex_w, reg_field, 0, b, force);
    buf.extend_from_slice(opcodes);
    emit_modrm(buf, reg_field, rm);
    Ok(())
}

pub fn encode_mov_r64_r64(dst: u8, src: u8, buf: &mut Vec<u8>) -> Result<(), EmitError> {
    emit_op_rm(buf, None, &[0x89], true, src, Operand::Reg(dst), false)
}

pub fn encode_mov_m_r64(dst: Mem, src: u8, buf: &mut Vec<u8>) -> Result<(), EmitError> {
    emit_op_rm(buf, None, &[0x89], true, src, Operand::Mem(dst), false)
}

pub fn encode_mov_r64_m(dst: u8, src: Mem, buf: &mut Vec<u8>) -> Result<(), EmitError> {
    emit_op_rm(buf, None, &[0x8B], true, dst, Operand::Mem(src), false)
}

pub fn encode_mov_r32_r32(dst: u8, src: u8, buf: &mut Vec<u8>) -> Result<(), EmitError> {
    emit_op_rm(buf, None, &[0x89], false, src, Operand::Reg(dst), false)
}

pub fn encode_mov_m_r32(dst: Mem, src: u8, buf: &mut Vec<u8>) -> Result<(), EmitError> {
    emit_op_rm(buf, None, &[0x89], false, src, Operand::Mem(dst), false)
}

pub fn encode_mov_r32_m(dst: u8, src: Mem, buf: &mut Vec<u8>) -> Result<(), EmitError> {
    emit_op_rm(buf, None, &[0x8B], false, dst, Operand::Mem(src), false)
}

pub fn encode_movzx_r32_rm8(dst: u8, src: Operand, buf: &mut Vec<u8>) -> Result<(), EmitError> {
    emit_op_rm(buf, None, &[0x0F, 0xB6], false, dst, src, true)
}

pub fn encode_movzx_r32_rm16(dst: u8, src: Operand, buf: &mut Vec<u8>) -> Result<(), EmitError> {
    emit_op_rm(buf, None, &[0x0F, 0xB7], false, dst, src, false)
}

pub fn encode_movzx_r64_rm8(dst: u8, src: Operand, buf: &mut Vec<u8>) -> Result<(), EmitError> {
    emit_op_rm(buf, None, &[0x0F, 0xB6], true, dst, src, true)
}

pub fn encode_movzx_r64_rm16(dst: u8, src: Operand, buf: &mut Vec<u8>) -> Result<(), EmitError> {
    emit_op_rm(buf, None, &[0x0F, 0xB7], true, dst, src, false)
}

pub fn encode_movsx_r64_rm8(dst: u8, src: Operand, buf: &mut Vec<u8>) -> Result<(), EmitError> {
    emit_op_rm(buf, None, &[0x0F, 0xBE], true, dst, src, true)
}

pub fn encode_movsx_r64_rm16(dst: u8, src: Operand, buf: &mut Vec<u8>) -> Result<(), EmitError> {
    emit_op_rm(buf, None, &[0x0F, 0xBF], true, dst, src, false)
}

pub fn encode_movsxd_r64_rm32(dst: u8, src: Operand, buf: &mut Vec<u8>) -> Result<(), EmitError> {
    emit_op_rm(buf, None, &[0x63], true, dst, src, false)
}

pub fn encode_lea_r64_m(dst: u8, src: Mem, buf: &mut Vec<u8>) -> Result<(), EmitError> {
    emit_op_rm(buf, None, &[0x8D], true, dst, Operand::Mem(src), false)
}

pub fn encode_add_r64_r64(dst: u8, src: u8, buf: &mut Vec<u8>) -> Result<(), EmitError> {
    emit_op_rm(buf, None, &[0x01], true, src, Operand::Reg(dst), false)
}

pub fn encode_add_r64_m(dst: u8, src: Mem, buf: &mut Vec<u8>) -> Result<(), EmitError> {
    emit_op_rm(buf, None, &[0x03], true, dst, Operand::Mem(src), false)
}

pub fn encode_sub_r64_r64(dst: u8, src: u8, buf: &mut Vec<u8>) -> Result<(), EmitError> {
    emit_op_rm(buf, None, &[0x29], true, src, Operand::Reg(dst), false)
}

pub fn encode_sub_r64_m(dst: u8, src: Mem, buf: &mut Vec<u8>) -> Result<(), EmitError> {
    emit_op_rm(buf, None, &[0x2B], true, dst, Operand::Mem(src), false)
}

pub fn encode_and_r64_r64(dst: u8, src: u8, buf: &mut Vec<u8>) -> Result<(), EmitError> {
    emit_op_rm(buf, None, &[0x21], true, src, Operand::Reg(dst), false)
}

pub fn encode_and_r64_m(dst: u8, src: Mem, buf: &mut Vec<u8>) -> Result<(), EmitError> {
    emit_op_rm(buf, None, &[0x23], true, dst, Operand::Mem(src), false)
}

pub fn encode_or_r64_r64(dst: u8, src: u8, buf: &mut Vec<u8>) -> Result<(), EmitError> {
    emit_op_rm(buf, None, &[0x09], true, src, Operand::Reg(dst), false)
}

pub fn encode_or_r64_m(dst: u8, src: Mem, buf: &mut Vec<u8>) -> Result<(), EmitError> {
    emit_op_rm(buf, None, &[0x0B], true, dst, Operand::Mem(src), false)
}

pub fn encode_xor_r64_r64(dst: u8, src: u8, buf: &mut Vec<u8>) -> Result<(), EmitError> {
    emit_op_rm(buf, None, &[0x31], true, src, Operand::Reg(dst), false)
}

pub fn encode_xor_r64_m(dst: u8, src: Mem, buf: &mut Vec<u8>) -> Result<(), EmitError> {
    emit_op_rm(buf, None, &[0x33], true, dst, Operand::Mem(src), false)
}

pub fn encode_shl_r64_cl(dst: u8, buf: &mut Vec<u8>) -> Result<(), EmitError> {
    emit_op_rm(buf, None, &[0xD3], true, 4, Operand::Reg(dst), false)
}

pub fn encode_shr_r64_cl(dst: u8, buf: &mut Vec<u8>) -> Result<(), EmitError> {
    emit_op_rm(buf, None, &[0xD3], true, 5, Operand::Reg(dst), false)
}

pub fn encode_shl_r64_imm8(dst: u8, imm8: u8, buf: &mut Vec<u8>) -> Result<(), EmitError> {
    emit_op_rm(buf, None, &[0xC1], true, 4, Operand::Reg(dst), false)?;
    buf.push(imm8);
    Ok(())
}

pub fn encode_shr_r64_imm8(dst: u8, imm8: u8, buf: &mut Vec<u8>) -> Result<(), EmitError> {
    emit_op_rm(buf, None, &[0xC1], true, 5, Operand::Reg(dst), false)?;
    buf.push(imm8);
    Ok(())
}

pub fn encode_neg_r64(dst: u8, buf: &mut Vec<u8>) -> Result<(), EmitError> {
    emit_op_rm(buf, None, &[0xF7], true, 3, Operand::Reg(dst), false)
}

pub fn encode_cmp_r64_r64(lhs: u8, rhs: u8, buf: &mut Vec<u8>) -> Result<(), EmitError> {
    emit_op_rm(buf, None, &[0x39], true, rhs, Operand::Reg(lhs), false)
}

pub fn encode_cmp_r64_m(lhs: u8, rhs: Mem, buf: &mut Vec<u8>) -> Result<(), EmitError> {
    emit_op_rm(buf, None, &[0x3B], true, lhs, Operand::Mem(rhs), false)
}

pub fn encode_test_r64_r64(lhs: u8, rhs: u8, buf: &mut Vec<u8>) -> Result<(), EmitError> {
    emit_op_rm(buf, None, &[0x85], true, rhs, Operand::Reg(lhs), false)
}

pub fn encode_test_r32_r32(lhs: u8, rhs: u8, buf: &mut Vec<u8>) -> Result<(), EmitError> {
    emit_op_rm(buf, None, &[0x85], false, rhs, Operand::Reg(lhs), false)
}

pub fn encode_test_m_r64(lhs: Mem, rhs: u8, buf: &mut Vec<u8>) -> Result<(), EmitError> {
    emit_op_rm(buf, None, &[0x85], true, rhs, Operand::Mem(lhs), false)
}

pub fn encode_setne_r8(dst: u8, buf: &mut Vec<u8>) -> Result<(), EmitError> {
    emit_op_rm(buf, None, &[0x0F, 0x95], false, 0, Operand::Reg(dst), true)
}

pub fn encode_je_rel32(buf: &mut Vec<u8>, disp: i32) -> Result<(), EmitError> {
    buf.extend_from_slice(&[0x0F, 0x84]);
    buf.extend_from_slice(&disp.to_le_bytes());
    Ok(())
}

pub fn encode_jz_rel32(buf: &mut Vec<u8>, disp: i32) -> Result<(), EmitError> {
    encode_je_rel32(buf, disp)
}

pub fn encode_jnz_rel32(buf: &mut Vec<u8>, disp: i32) -> Result<(), EmitError> {
    buf.extend_from_slice(&[0x0F, 0x85]);
    buf.extend_from_slice(&disp.to_le_bytes());
    Ok(())
}

pub fn encode_jmp_rel32(buf: &mut Vec<u8>, disp: i32) -> Result<(), EmitError> {
    buf.push(0xE9);
    buf.extend_from_slice(&disp.to_le_bytes());
    Ok(())
}

pub fn encode_call_rel32(buf: &mut Vec<u8>, disp: i32) -> Result<(), EmitError> {
    buf.push(0xE8);
    buf.extend_from_slice(&disp.to_le_bytes());
    Ok(())
}

pub fn encode_call_r64(reg: u8, buf: &mut Vec<u8>) -> Result<(), EmitError> {
    emit_op_rm(buf, None, &[0xFF], false, 2, Operand::Reg(reg), false)
}

pub fn encode_push_r64(reg: u8, buf: &mut Vec<u8>) -> Result<(), EmitError> {
    let reg = check_reg(reg)?;
    push_rex(buf, false, 0, 0, reg, false);
    buf.push(0x50 + (reg & 0x7));
    Ok(())
}

pub fn encode_pop_r64(reg: u8, buf: &mut Vec<u8>) -> Result<(), EmitError> {
    let reg = check_reg(reg)?;
    push_rex(buf, false, 0, 0, reg, false);
    buf.push(0x58 + (reg & 0x7));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn encode_mov_variants() {
        let mut buf = Vec::new();
        encode_mov_r64_r64(11, 10, &mut buf).unwrap();
        assert_eq!(buf, [0x4d, 0x89, 0xd3]);

        buf.clear();
        encode_mov_m_r64(Mem { base: 4, disp: 0 }, 10, &mut buf).unwrap();
        assert_eq!(buf, [0x4c, 0x89, 0x14, 0x24]);

        buf.clear();
        encode_mov_r64_m(10, Mem { base: 4, disp: 0 }, &mut buf).unwrap();
        assert_eq!(buf, [0x4c, 0x8b, 0x14, 0x24]);

        buf.clear();
        encode_mov_r32_r32(11, 10, &mut buf).unwrap();
        assert_eq!(buf, [0x45, 0x89, 0xd3]);

        buf.clear();
        encode_mov_m_r32(Mem { base: 4, disp: 0 }, 10, &mut buf).unwrap();
        assert_eq!(buf, [0x44, 0x89, 0x14, 0x24]);

        buf.clear();
        encode_mov_r32_m(10, Mem { base: 4, disp: 0 }, &mut buf).unwrap();
        assert_eq!(buf, [0x44, 0x8b, 0x14, 0x24]);
    }

    #[test]
    fn encode_extend_and_lea_variants() {
        let mut buf = Vec::new();
        encode_movzx_r32_rm8(
            10,
            Operand::Mem(Mem {
                base: 14,
                disp: 127,
            }),
            &mut buf,
        )
        .unwrap();
        assert_eq!(buf, [0x45, 0x0f, 0xb6, 0x56, 0x7f]);

        buf.clear();
        encode_movzx_r32_rm16(
            10,
            Operand::Mem(Mem {
                base: 14,
                disp: 126,
            }),
            &mut buf,
        )
        .unwrap();
        assert_eq!(buf, [0x45, 0x0f, 0xb7, 0x56, 0x7e]);

        buf.clear();
        encode_movsx_r64_rm8(
            10,
            Operand::Mem(Mem {
                base: 14,
                disp: 127,
            }),
            &mut buf,
        )
        .unwrap();
        assert_eq!(buf, [0x4d, 0x0f, 0xbe, 0x56, 0x7f]);

        buf.clear();
        encode_movsx_r64_rm16(
            10,
            Operand::Mem(Mem {
                base: 14,
                disp: 126,
            }),
            &mut buf,
        )
        .unwrap();
        assert_eq!(buf, [0x4d, 0x0f, 0xbf, 0x56, 0x7e]);

        buf.clear();
        encode_movsxd_r64_rm32(
            10,
            Operand::Mem(Mem {
                base: 14,
                disp: 124,
            }),
            &mut buf,
        )
        .unwrap();
        assert_eq!(buf, [0x4d, 0x63, 0x56, 0x7c]);

        buf.clear();
        encode_movzx_r64_rm8(10, Operand::Reg(10), &mut buf).unwrap();
        assert_eq!(buf, [0x4d, 0x0f, 0xb6, 0xd2]);

        buf.clear();
        encode_movzx_r64_rm16(10, Operand::Reg(10), &mut buf).unwrap();
        assert_eq!(buf, [0x4d, 0x0f, 0xb7, 0xd2]);

        buf.clear();
        encode_lea_r64_m(10, Mem { base: 4, disp: 16 }, &mut buf).unwrap();
        assert_eq!(buf, [0x4c, 0x8d, 0x54, 0x24, 0x10]);
    }

    #[test]
    fn encode_integer_and_shift_ops() {
        let mut buf = Vec::new();
        encode_add_r64_r64(10, 11, &mut buf).unwrap();
        assert_eq!(buf, [0x4d, 0x01, 0xda]);

        buf.clear();
        encode_add_r64_m(10, Mem { base: 4, disp: 0 }, &mut buf).unwrap();
        assert_eq!(buf, [0x4c, 0x03, 0x14, 0x24]);

        buf.clear();
        encode_sub_r64_r64(10, 11, &mut buf).unwrap();
        assert_eq!(buf, [0x4d, 0x29, 0xda]);

        buf.clear();
        encode_sub_r64_m(10, Mem { base: 4, disp: 0 }, &mut buf).unwrap();
        assert_eq!(buf, [0x4c, 0x2b, 0x14, 0x24]);

        buf.clear();
        encode_and_r64_r64(10, 11, &mut buf).unwrap();
        assert_eq!(buf, [0x4d, 0x21, 0xda]);

        buf.clear();
        encode_and_r64_m(10, Mem { base: 4, disp: 0 }, &mut buf).unwrap();
        assert_eq!(buf, [0x4c, 0x23, 0x14, 0x24]);

        buf.clear();
        encode_or_r64_r64(10, 11, &mut buf).unwrap();
        assert_eq!(buf, [0x4d, 0x09, 0xda]);

        buf.clear();
        encode_or_r64_m(10, Mem { base: 4, disp: 0 }, &mut buf).unwrap();
        assert_eq!(buf, [0x4c, 0x0b, 0x14, 0x24]);

        buf.clear();
        encode_xor_r64_r64(10, 11, &mut buf).unwrap();
        assert_eq!(buf, [0x4d, 0x31, 0xda]);

        buf.clear();
        encode_xor_r64_m(10, Mem { base: 4, disp: 0 }, &mut buf).unwrap();
        assert_eq!(buf, [0x4c, 0x33, 0x14, 0x24]);

        buf.clear();
        encode_shl_r64_cl(10, &mut buf).unwrap();
        assert_eq!(buf, [0x49, 0xd3, 0xe2]);

        buf.clear();
        encode_shr_r64_cl(10, &mut buf).unwrap();
        assert_eq!(buf, [0x49, 0xd3, 0xea]);

        buf.clear();
        encode_shl_r64_imm8(10, 5, &mut buf).unwrap();
        assert_eq!(buf, [0x49, 0xc1, 0xe2, 0x05]);

        buf.clear();
        encode_shr_r64_imm8(10, 7, &mut buf).unwrap();
        assert_eq!(buf, [0x49, 0xc1, 0xea, 0x07]);

        buf.clear();
        encode_neg_r64(10, &mut buf).unwrap();
        assert_eq!(buf, [0x49, 0xf7, 0xda]);
    }

    #[test]
    fn encode_cmp_test_setcc() {
        let mut buf = Vec::new();
        encode_cmp_r64_r64(10, 11, &mut buf).unwrap();
        assert_eq!(buf, [0x4d, 0x39, 0xda]);

        buf.clear();
        encode_cmp_r64_m(10, Mem { base: 4, disp: 0 }, &mut buf).unwrap();
        assert_eq!(buf, [0x4c, 0x3b, 0x14, 0x24]);

        buf.clear();
        encode_test_r64_r64(10, 10, &mut buf).unwrap();
        assert_eq!(buf, [0x4d, 0x85, 0xd2]);

        buf.clear();
        encode_test_r32_r32(10, 10, &mut buf).unwrap();
        assert_eq!(buf, [0x45, 0x85, 0xd2]);

        buf.clear();
        encode_setne_r8(10, &mut buf).unwrap();
        assert_eq!(buf, [0x41, 0x0f, 0x95, 0xc2]);
    }

    #[test]
    fn encode_control_flow_and_stack() {
        let mut buf = Vec::new();
        encode_je_rel32(&mut buf, 0x13d).unwrap();
        assert_eq!(buf, [0x0f, 0x84, 0x3d, 0x01, 0x00, 0x00]);

        buf.clear();
        encode_jz_rel32(&mut buf, 0x137).unwrap();
        assert_eq!(buf, [0x0f, 0x84, 0x37, 0x01, 0x00, 0x00]);

        buf.clear();
        encode_jnz_rel32(&mut buf, 0x131).unwrap();
        assert_eq!(buf, [0x0f, 0x85, 0x31, 0x01, 0x00, 0x00]);

        buf.clear();
        encode_call_rel32(&mut buf, 0x12c).unwrap();
        assert_eq!(buf, [0xe8, 0x2c, 0x01, 0x00, 0x00]);

        buf.clear();
        encode_call_r64(0, &mut buf).unwrap();
        assert_eq!(buf, [0xff, 0xd0]);

        buf.clear();
        encode_call_r64(10, &mut buf).unwrap();
        assert_eq!(buf, [0x41, 0xff, 0xd2]);

        buf.clear();
        encode_push_r64(10, &mut buf).unwrap();
        assert_eq!(buf, [0x41, 0x52]);

        buf.clear();
        encode_pop_r64(10, &mut buf).unwrap();
        assert_eq!(buf, [0x41, 0x5a]);
    }

    #[test]
    fn encode_addressing_edge_cases() {
        let mut buf = Vec::new();
        encode_mov_m_r64(Mem { base: 13, disp: 0 }, 10, &mut buf).unwrap();
        assert_eq!(buf, [0x4d, 0x89, 0x55, 0x00]);

        buf.clear();
        encode_mov_r64_m(10, Mem { base: 13, disp: 0 }, &mut buf).unwrap();
        assert_eq!(buf, [0x4d, 0x8b, 0x55, 0x00]);

        buf.clear();
        encode_mov_m_r64(
            Mem {
                base: 12,
                disp: 4096,
            },
            10,
            &mut buf,
        )
        .unwrap();
        assert_eq!(buf, [0x4d, 0x89, 0x94, 0x24, 0x00, 0x10, 0x00, 0x00]);

        buf.clear();
        encode_lea_r64_m(10, Mem { base: 13, disp: 0 }, &mut buf).unwrap();
        assert_eq!(buf, [0x4d, 0x8d, 0x55, 0x00]);

        buf.clear();
        encode_lea_r64_m(
            10,
            Mem {
                base: 13,
                disp: 4096,
            },
            &mut buf,
        )
        .unwrap();
        assert_eq!(buf, [0x4d, 0x8d, 0x95, 0x00, 0x10, 0x00, 0x00]);

        buf.clear();
        encode_test_m_r64(
            Mem {
                base: 13,
                disp: 4096,
            },
            10,
            &mut buf,
        )
        .unwrap();
        assert_eq!(buf, [0x4d, 0x85, 0x95, 0x00, 0x10, 0x00, 0x00]);
    }

    #[test]
    fn emitter_records_source_map_and_resolves_fixups() {
        let mut emitter = Emitter::new();
        let start = emitter.new_label();
        let done = emitter.new_label();

        emitter.bind_label(start).unwrap();
        emitter.set_source_location(crate::SourceLocation {
            file: 3,
            line: 10,
            column: 1,
        });
        emitter.emit_je_label(done).unwrap();
        emitter.set_source_location(crate::SourceLocation {
            file: 3,
            line: 11,
            column: 1,
        });
        emitter.emit_call_label(start).unwrap();
        emitter.bind_label(done).unwrap();
        emitter.set_source_location(crate::SourceLocation {
            file: 3,
            line: 12,
            column: 1,
        });
        emitter.emit_jnz_label(start).unwrap();

        let finalized = emitter.finalize().unwrap();
        assert_eq!(
            finalized.source_map,
            vec![
                crate::SourceMapEntry {
                    offset: 0,
                    location: crate::SourceLocation {
                        file: 3,
                        line: 10,
                        column: 1,
                    },
                },
                crate::SourceMapEntry {
                    offset: 6,
                    location: crate::SourceLocation {
                        file: 3,
                        line: 11,
                        column: 1,
                    },
                },
                crate::SourceMapEntry {
                    offset: 11,
                    location: crate::SourceLocation {
                        file: 3,
                        line: 12,
                        column: 1,
                    },
                },
            ]
        );
        assert_eq!(
            finalized.code,
            vec![
                0x0f, 0x84, 0x05, 0x00, 0x00, 0x00, // je +5
                0xe8, 0xf5, 0xff, 0xff, 0xff, // call -11
                0x0f, 0x85, 0xef, 0xff, 0xff, 0xff, // jnz -17
            ]
        );

        let trace = finalized.trace_text().unwrap();
        assert_eq!(
            trace,
            "00000000 file=3 line=10 col=1 bytes=0f8405000000\n00000006 file=3 line=11 col=1 bytes=e8f5ffffff\n0000000b file=3 line=12 col=1 bytes=0f85efffffff"
        );

        let source_map_encoded = finalized.source_map_le().unwrap();
        assert_eq!(
            crate::decode_source_map_le(&source_map_encoded).unwrap(),
            finalized.source_map
        );
    }

    #[test]
    fn emitter_reports_unbound_label() {
        let mut emitter = Emitter::new();
        let dangling = emitter.new_label();
        emitter.emit_call_label(dangling).unwrap();
        let err = emitter.finalize().unwrap_err();
        assert!(matches!(err, EmitError::UnboundLabel { .. }));
    }

    #[test]
    fn emitter_reports_out_of_range_fixup() {
        let mut emitter = Emitter::new();
        let far = emitter.new_label();
        emitter.emit_call_label(far).unwrap();
        emitter.labels[far.0 as usize] = Some((i32::MAX as u32).saturating_add(1024));
        let err = emitter.finalize().unwrap_err();
        assert!(matches!(err, EmitError::RelativeOutOfRange { .. }));
    }

    #[test]
    fn invalid_register_is_rejected() {
        let mut buf = Vec::new();
        let err = encode_mov_r64_r64(0, 16, &mut buf).unwrap_err();
        assert!(matches!(err, EmitError::InvalidRegister { reg: 16 }));
    }

    #[test]
    fn encode_disp_size_boundaries() {
        let mut buf = Vec::new();
        encode_mov_m_r64(Mem { base: 5, disp: 0 }, 10, &mut buf).unwrap();
        assert_eq!(buf, [0x4c, 0x89, 0x55, 0x00]);

        buf.clear();
        encode_mov_m_r64(Mem { base: 5, disp: 127 }, 10, &mut buf).unwrap();
        assert_eq!(buf, [0x4c, 0x89, 0x55, 0x7f]);

        buf.clear();
        encode_mov_m_r64(Mem { base: 5, disp: 128 }, 10, &mut buf).unwrap();
        assert_eq!(buf, [0x4c, 0x89, 0x95, 0x80, 0x00, 0x00, 0x00]);
    }

    #[test]
    fn encode_rex_for_low_byte_registers() {
        let mut buf = Vec::new();
        encode_setne_r8(4, &mut buf).unwrap(); // spl
        assert_eq!(buf, [0x40, 0x0f, 0x95, 0xc4]);

        buf.clear();
        encode_movzx_r64_rm8(0, Operand::Reg(4), &mut buf).unwrap(); // movzx rax, spl
        assert_eq!(buf, [0x48, 0x0f, 0xb6, 0xc4]);
    }

    #[test]
    fn emitter_resolves_forward_and_backward_jumps() {
        let mut emitter = Emitter::new();
        let start = emitter.new_label();
        let target = emitter.new_label();
        emitter.bind_label(start).unwrap();
        emitter.set_source_location(crate::SourceLocation {
            file: 4,
            line: 1,
            column: 1,
        });
        emitter.emit_jmp_label(target).unwrap();
        emitter.set_source_location(crate::SourceLocation {
            file: 4,
            line: 2,
            column: 1,
        });
        emitter.emit_bytes(&[0x90, 0x90]); // 2 nops
        emitter.bind_label(target).unwrap();
        emitter.set_source_location(crate::SourceLocation {
            file: 4,
            line: 3,
            column: 1,
        });
        emitter.emit_jz_label(start).unwrap();

        let finalized = emitter.finalize().unwrap();
        assert_eq!(
            finalized.code,
            vec![
                0xe9, 0x02, 0x00, 0x00, 0x00, // jmp +2
                0x90, 0x90, // nops
                0x0f, 0x84, 0xf3, 0xff, 0xff, 0xff, // jz -13
            ]
        );
    }
}
