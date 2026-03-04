use crate::{SourceLocation, SourceMap, SourceMapEntry};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Width {
    W32,
    X64,
}

impl Width {
    fn sf(self) -> u32 {
        match self {
            Self::W32 => 0,
            Self::X64 => 1,
        }
    }

    fn bits(self) -> u8 {
        match self {
            Self::W32 => 32,
            Self::X64 => 64,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Shift {
    Lsl = 0,
    Lsr = 1,
    Asr = 2,
    Ror = 3,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Condition {
    Eq = 0,
    Ne = 1,
    Hs = 2,
    Lo = 3,
    Mi = 4,
    Pl = 5,
    Vs = 6,
    Vc = 7,
    Hi = 8,
    Ls = 9,
    Ge = 10,
    Lt = 11,
    Gt = 12,
    Le = 13,
}

impl Condition {
    fn invert(self) -> Self {
        match self {
            Self::Eq => Self::Ne,
            Self::Ne => Self::Eq,
            Self::Hs => Self::Lo,
            Self::Lo => Self::Hs,
            Self::Mi => Self::Pl,
            Self::Pl => Self::Mi,
            Self::Vs => Self::Vc,
            Self::Vc => Self::Vs,
            Self::Hi => Self::Ls,
            Self::Ls => Self::Hi,
            Self::Ge => Self::Lt,
            Self::Lt => Self::Ge,
            Self::Gt => Self::Le,
            Self::Le => Self::Gt,
        }
    }
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

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EmitError {
    InvalidRegister {
        reg: u8,
    },
    InvalidShiftAmount {
        width: Width,
        amount: u8,
    },
    InvalidMovWideShift {
        width: Width,
        shift: u8,
    },
    InvalidImmediate {
        instruction: &'static str,
        value: i64,
    },
    InvalidOffset {
        instruction: &'static str,
        offset: u32,
        align: u32,
        max: u32,
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
    BranchTargetNotAligned {
        at_offset: u32,
        target_offset: u32,
    },
    BranchOutOfRange {
        bits: u8,
        at_offset: u32,
        target_offset: u32,
        delta_words: i64,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum FixupKind {
    Imm26,
    Imm19,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Fixup {
    at_offset: u32,
    label: LabelId,
    kind: FixupKind,
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

    pub fn emit_word(&mut self, word: u32) {
        self.maybe_record_source_map();
        self.buf.extend_from_slice(&word.to_le_bytes());
    }

    pub fn emit_b_label(
        &mut self,
        label: LabelId,
    ) -> Result<(), EmitError> {
        self.emit_word(encode_b(0)?);
        self.fixups.push(Fixup {
            at_offset: self.current_offset() - 4,
            label,
            kind: FixupKind::Imm26,
        });
        Ok(())
    }

    pub fn emit_bl_label(
        &mut self,
        label: LabelId,
    ) -> Result<(), EmitError> {
        self.emit_word(encode_bl(0)?);
        self.fixups.push(Fixup {
            at_offset: self.current_offset() - 4,
            label,
            kind: FixupKind::Imm26,
        });
        Ok(())
    }

    pub fn emit_cbz_label(
        &mut self,
        width: Width,
        rt: u8,
        label: LabelId,
    ) -> Result<(), EmitError> {
        self.emit_word(encode_cbz(width, rt, 0)?);
        self.fixups.push(Fixup {
            at_offset: self.current_offset() - 4,
            label,
            kind: FixupKind::Imm19,
        });
        Ok(())
    }

    pub fn emit_cbnz_label(
        &mut self,
        width: Width,
        rt: u8,
        label: LabelId,
    ) -> Result<(), EmitError> {
        self.emit_word(encode_cbnz(width, rt, 0)?);
        self.fixups.push(Fixup {
            at_offset: self.current_offset() - 4,
            label,
            kind: FixupKind::Imm19,
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

            if (target & 0b11) != 0 || (fixup.at_offset & 0b11) != 0 {
                return Err(EmitError::BranchTargetNotAligned {
                    at_offset: fixup.at_offset,
                    target_offset: target,
                });
            }

            let delta_words = (target as i64 - fixup.at_offset as i64) / 4;
            let word = u32::from_le_bytes([
                self.buf[fixup.at_offset as usize],
                self.buf[fixup.at_offset as usize + 1],
                self.buf[fixup.at_offset as usize + 2],
                self.buf[fixup.at_offset as usize + 3],
            ]);

            let patched = match fixup.kind {
                FixupKind::Imm26 => {
                    check_signed_bits("branch26", delta_words, 26)?;
                    (word & !0x03ff_ffff) | ((delta_words as u32) & 0x03ff_ffff)
                }
                FixupKind::Imm19 => {
                    check_signed_bits("branch19", delta_words, 19)?;
                    (word & !(0x7ffff << 5)) | (((delta_words as u32) & 0x7ffff) << 5)
                }
            };

            self.buf[fixup.at_offset as usize..fixup.at_offset as usize + 4]
                .copy_from_slice(&patched.to_le_bytes());
        }

        Ok(FinalizedEmission {
            code: self.buf,
            source_map: self.source_map,
        })
    }
}

fn check_reg(reg: u8) -> Result<u32, EmitError> {
    if reg <= 31 {
        Ok(reg as u32)
    } else {
        Err(EmitError::InvalidRegister { reg })
    }
}

fn check_shift_amount(width: Width, amount: u8) -> Result<u32, EmitError> {
    if amount < width.bits() {
        Ok(amount as u32)
    } else {
        Err(EmitError::InvalidShiftAmount { width, amount })
    }
}

fn check_signed_bits(instruction: &'static str, value: i64, bits: u8) -> Result<(), EmitError> {
    let min = -(1i64 << (bits - 1));
    let max = (1i64 << (bits - 1)) - 1;
    if value < min || value > max {
        Err(EmitError::InvalidImmediate { instruction, value })
    } else {
        Ok(())
    }
}

fn emit_logical_shifted_reg(
    base: u32,
    width: Width,
    rd: u8,
    rn: u8,
    rm: u8,
    shift: Shift,
    amount: u8,
) -> Result<u32, EmitError> {
    let rd = check_reg(rd)?;
    let rn = check_reg(rn)?;
    let rm = check_reg(rm)?;
    let amount = check_shift_amount(width, amount)?;
    Ok((width.sf() << 31)
        | base
        | ((shift as u32) << 22)
        | (rm << 16)
        | (amount << 10)
        | (rn << 5)
        | rd)
}

pub fn encode_mov_reg(width: Width, rd: u8, rm: u8) -> Result<u32, EmitError> {
    encode_orr_reg(width, rd, 31, rm, Shift::Lsl, 0)
}

pub fn encode_movz(width: Width, rd: u8, imm16: u16, shift: u8) -> Result<u32, EmitError> {
    let rd = check_reg(rd)?;
    if !matches!(shift, 0 | 16 | 32 | 48) {
        return Err(EmitError::InvalidMovWideShift { width, shift });
    }
    if width == Width::W32 && shift > 16 {
        return Err(EmitError::InvalidMovWideShift { width, shift });
    }
    let hw = (shift / 16) as u32;
    Ok((width.sf() << 31) | 0x5280_0000 | (hw << 21) | ((imm16 as u32) << 5) | rd)
}

pub fn encode_movk(width: Width, rd: u8, imm16: u16, shift: u8) -> Result<u32, EmitError> {
    let rd = check_reg(rd)?;
    if !matches!(shift, 0 | 16 | 32 | 48) {
        return Err(EmitError::InvalidMovWideShift { width, shift });
    }
    if width == Width::W32 && shift > 16 {
        return Err(EmitError::InvalidMovWideShift { width, shift });
    }
    let hw = (shift / 16) as u32;
    Ok((width.sf() << 31) | 0x7280_0000 | (hw << 21) | ((imm16 as u32) << 5) | rd)
}

fn encode_load_store_unsigned(
    base: u32,
    rt: u8,
    rn: u8,
    offset: u32,
    align: u32,
) -> Result<u32, EmitError> {
    let rt = check_reg(rt)?;
    let rn = check_reg(rn)?;
    let max = 4095 * align;
    if !offset.is_multiple_of(align) || offset > max {
        return Err(EmitError::InvalidOffset {
            instruction: "ld/st unsigned",
            offset,
            align,
            max,
        });
    }
    let imm12 = offset / align;
    Ok(base | (imm12 << 10) | (rn << 5) | rt)
}

pub fn encode_ldr_imm(width: Width, rt: u8, rn: u8, offset: u32) -> Result<u32, EmitError> {
    let base = match width {
        Width::W32 => 0xB940_0000,
        Width::X64 => 0xF940_0000,
    };
    let align = match width {
        Width::W32 => 4,
        Width::X64 => 8,
    };
    encode_load_store_unsigned(base, rt, rn, offset, align)
}

pub fn encode_str_imm(width: Width, rt: u8, rn: u8, offset: u32) -> Result<u32, EmitError> {
    let base = match width {
        Width::W32 => 0xB900_0000,
        Width::X64 => 0xF900_0000,
    };
    let align = match width {
        Width::W32 => 4,
        Width::X64 => 8,
    };
    encode_load_store_unsigned(base, rt, rn, offset, align)
}

pub fn encode_ldrb_imm(rt: u8, rn: u8, offset: u32) -> Result<u32, EmitError> {
    encode_load_store_unsigned(0x3940_0000, rt, rn, offset, 1)
}

pub fn encode_ldrh_imm(rt: u8, rn: u8, offset: u32) -> Result<u32, EmitError> {
    encode_load_store_unsigned(0x7940_0000, rt, rn, offset, 2)
}

pub fn encode_strb_imm(rt: u8, rn: u8, offset: u32) -> Result<u32, EmitError> {
    encode_load_store_unsigned(0x3900_0000, rt, rn, offset, 1)
}

pub fn encode_strh_imm(rt: u8, rn: u8, offset: u32) -> Result<u32, EmitError> {
    encode_load_store_unsigned(0x7900_0000, rt, rn, offset, 2)
}

pub fn encode_add_reg(width: Width, rd: u8, rn: u8, rm: u8) -> Result<u32, EmitError> {
    let rd = check_reg(rd)?;
    let rn = check_reg(rn)?;
    let rm = check_reg(rm)?;
    Ok((width.sf() << 31) | 0x0B00_0000 | (rm << 16) | (rn << 5) | rd)
}

pub fn encode_sub_reg(width: Width, rd: u8, rn: u8, rm: u8) -> Result<u32, EmitError> {
    let rd = check_reg(rd)?;
    let rn = check_reg(rn)?;
    let rm = check_reg(rm)?;
    Ok((width.sf() << 31) | 0x4B00_0000 | (rm << 16) | (rn << 5) | rd)
}

fn encode_add_sub_imm(
    width: Width,
    rd: u8,
    rn: u8,
    imm12: u16,
    shift12: bool,
    is_sub: bool,
) -> Result<u32, EmitError> {
    let rd = check_reg(rd)?;
    let rn = check_reg(rn)?;
    if imm12 > 0x0fff {
        return Err(EmitError::InvalidImmediate {
            instruction: "add/sub imm12",
            value: imm12 as i64,
        });
    }
    let opcode = match (width, is_sub) {
        (Width::W32, false) => 0x1100_0000,
        (Width::X64, false) => 0x9100_0000,
        (Width::W32, true) => 0x5100_0000,
        (Width::X64, true) => 0xD100_0000,
    };
    Ok(opcode | ((shift12 as u32) << 22) | ((imm12 as u32) << 10) | (rn << 5) | rd)
}

pub fn encode_add_imm(
    width: Width,
    rd: u8,
    rn: u8,
    imm12: u16,
    shift12: bool,
) -> Result<u32, EmitError> {
    encode_add_sub_imm(width, rd, rn, imm12, shift12, false)
}

pub fn encode_sub_imm(
    width: Width,
    rd: u8,
    rn: u8,
    imm12: u16,
    shift12: bool,
) -> Result<u32, EmitError> {
    encode_add_sub_imm(width, rd, rn, imm12, shift12, true)
}

pub fn encode_and_reg(
    width: Width,
    rd: u8,
    rn: u8,
    rm: u8,
    shift: Shift,
    amount: u8,
) -> Result<u32, EmitError> {
    emit_logical_shifted_reg(0x0A00_0000, width, rd, rn, rm, shift, amount)
}

pub fn encode_orr_reg(
    width: Width,
    rd: u8,
    rn: u8,
    rm: u8,
    shift: Shift,
    amount: u8,
) -> Result<u32, EmitError> {
    emit_logical_shifted_reg(0x2A00_0000, width, rd, rn, rm, shift, amount)
}

pub fn encode_eor_reg(
    width: Width,
    rd: u8,
    rn: u8,
    rm: u8,
    shift: Shift,
    amount: u8,
) -> Result<u32, EmitError> {
    emit_logical_shifted_reg(0x4A00_0000, width, rd, rn, rm, shift, amount)
}

pub fn encode_lsl_reg(width: Width, rd: u8, rn: u8, rm: u8) -> Result<u32, EmitError> {
    let rd = check_reg(rd)?;
    let rn = check_reg(rn)?;
    let rm = check_reg(rm)?;
    Ok((width.sf() << 31) | 0x1AC0_2000 | (rm << 16) | (rn << 5) | rd)
}

pub fn encode_lsr_reg(width: Width, rd: u8, rn: u8, rm: u8) -> Result<u32, EmitError> {
    let rd = check_reg(rd)?;
    let rn = check_reg(rn)?;
    let rm = check_reg(rm)?;
    Ok((width.sf() << 31) | 0x1AC0_2400 | (rm << 16) | (rn << 5) | rd)
}

pub fn encode_neg(width: Width, rd: u8, rm: u8) -> Result<u32, EmitError> {
    encode_sub_reg(width, rd, 31, rm)
}

pub fn encode_cmp_reg(width: Width, rn: u8, rm: u8) -> Result<u32, EmitError> {
    let rn = check_reg(rn)?;
    let rm = check_reg(rm)?;
    Ok((width.sf() << 31) | 0x6B00_001F | (rm << 16) | (rn << 5))
}

pub fn encode_cmp_imm(width: Width, rn: u8, imm12: u16, shift12: bool) -> Result<u32, EmitError> {
    let rn = check_reg(rn)?;
    if imm12 > 0x0fff {
        return Err(EmitError::InvalidImmediate {
            instruction: "cmp imm12",
            value: imm12 as i64,
        });
    }
    let base = match width {
        Width::W32 => 0x7100_001F,
        Width::X64 => 0xF100_001F,
    };
    Ok(base | ((shift12 as u32) << 22) | ((imm12 as u32) << 10) | (rn << 5))
}

pub fn encode_cbz(width: Width, rt: u8, imm19: i32) -> Result<u32, EmitError> {
    check_reg(rt)?;
    check_signed_bits("cbz", imm19 as i64, 19)?;
    let base = match width {
        Width::W32 => 0x3400_0000,
        Width::X64 => 0xB400_0000,
    };
    Ok(base | (((imm19 as u32) & 0x7ffff) << 5) | (rt as u32))
}

pub fn encode_cbnz(width: Width, rt: u8, imm19: i32) -> Result<u32, EmitError> {
    check_reg(rt)?;
    check_signed_bits("cbnz", imm19 as i64, 19)?;
    let base = match width {
        Width::W32 => 0x3500_0000,
        Width::X64 => 0xB500_0000,
    };
    Ok(base | (((imm19 as u32) & 0x7ffff) << 5) | (rt as u32))
}

pub fn encode_cset(width: Width, rd: u8, condition: Condition) -> Result<u32, EmitError> {
    let rd = check_reg(rd)?;
    let inv = condition.invert() as u32;
    let base = match width {
        Width::W32 => 0x1A9F_07E0,
        Width::X64 => 0x9A9F_07E0,
    };
    Ok(base | (inv << 12) | rd)
}

pub fn encode_sxtb(rd: u8, rn: u8) -> Result<u32, EmitError> {
    let rd = check_reg(rd)?;
    let rn = check_reg(rn)?;
    Ok(0x9340_1C00 | (rn << 5) | rd)
}

pub fn encode_sxth(rd: u8, rn: u8) -> Result<u32, EmitError> {
    let rd = check_reg(rd)?;
    let rn = check_reg(rn)?;
    Ok(0x9340_3C00 | (rn << 5) | rd)
}

pub fn encode_sxtw(rd: u8, rn: u8) -> Result<u32, EmitError> {
    let rd = check_reg(rd)?;
    let rn = check_reg(rn)?;
    Ok(0x9340_7C00 | (rn << 5) | rd)
}

pub fn encode_b(imm26: i32) -> Result<u32, EmitError> {
    check_signed_bits("b", imm26 as i64, 26)?;
    Ok(0x1400_0000 | ((imm26 as u32) & 0x03ff_ffff))
}

pub fn encode_bl(imm26: i32) -> Result<u32, EmitError> {
    check_signed_bits("bl", imm26 as i64, 26)?;
    Ok(0x9400_0000 | ((imm26 as u32) & 0x03ff_ffff))
}

pub fn encode_blr(rn: u8) -> Result<u32, EmitError> {
    let rn = check_reg(rn)?;
    Ok(0xD63F_0000 | (rn << 5))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn word(bytes: &[u8]) -> u32 {
        let arr: [u8; 4] = bytes.try_into().expect("need 4 bytes");
        u32::from_le_bytes(arr)
    }

    #[test]
    fn encode_mov_and_mov_wide() {
        assert_eq!(encode_mov_reg(Width::X64, 1, 2).unwrap(), 0xAA02_03E1);
        assert_eq!(encode_mov_reg(Width::W32, 1, 2).unwrap(), 0x2A02_03E1);

        assert_eq!(encode_movz(Width::X64, 3, 0x1234, 0).unwrap(), 0xD282_4683);
        assert_eq!(encode_movz(Width::X64, 3, 0x1234, 16).unwrap(), 0xD2A2_4683);
        assert_eq!(encode_movz(Width::W32, 3, 0x1234, 0).unwrap(), 0x5282_4683);
        assert_eq!(encode_movz(Width::W32, 3, 0x1234, 16).unwrap(), 0x52A2_4683);

        assert_eq!(encode_movk(Width::X64, 3, 0x5678, 0).unwrap(), 0xF28A_CF03);
        assert_eq!(encode_movk(Width::X64, 3, 0x5678, 32).unwrap(), 0xF2CA_CF03);
        assert_eq!(encode_movk(Width::W32, 3, 0x5678, 0).unwrap(), 0x728A_CF03);
        assert_eq!(encode_movk(Width::W32, 3, 0x5678, 16).unwrap(), 0x72AA_CF03);
    }

    #[test]
    fn encode_load_store_unsigned_offset() {
        assert_eq!(encode_ldr_imm(Width::X64, 4, 5, 16).unwrap(), 0xF940_08A4);
        assert_eq!(encode_ldr_imm(Width::W32, 4, 5, 16).unwrap(), 0xB940_10A4);
        assert_eq!(encode_ldrb_imm(4, 5, 15).unwrap(), 0x3940_3CA4);
        assert_eq!(encode_ldrh_imm(4, 5, 14).unwrap(), 0x7940_1CA4);

        assert_eq!(encode_str_imm(Width::X64, 4, 5, 16).unwrap(), 0xF900_08A4);
        assert_eq!(encode_str_imm(Width::W32, 4, 5, 16).unwrap(), 0xB900_10A4);
        assert_eq!(encode_strb_imm(4, 5, 15).unwrap(), 0x3900_3CA4);
        assert_eq!(encode_strh_imm(4, 5, 14).unwrap(), 0x7900_1CA4);
    }

    #[test]
    fn encode_integer_arithmetic_and_bitwise() {
        assert_eq!(encode_add_reg(Width::X64, 6, 7, 8).unwrap(), 0x8B08_00E6);
        assert_eq!(encode_add_reg(Width::W32, 6, 7, 8).unwrap(), 0x0B08_00E6);
        assert_eq!(
            encode_add_imm(Width::X64, 6, 7, 123, false).unwrap(),
            0x9101_ECE6
        );
        assert_eq!(
            encode_add_imm(Width::W32, 6, 7, 123, false).unwrap(),
            0x1101_ECE6
        );
        assert_eq!(
            encode_add_imm(Width::X64, 6, 7, 0x123, true).unwrap(),
            0x9144_8CE6
        );

        assert_eq!(encode_sub_reg(Width::X64, 6, 7, 8).unwrap(), 0xCB08_00E6);
        assert_eq!(encode_sub_reg(Width::W32, 6, 7, 8).unwrap(), 0x4B08_00E6);
        assert_eq!(
            encode_sub_imm(Width::X64, 6, 7, 123, false).unwrap(),
            0xD101_ECE6
        );
        assert_eq!(
            encode_sub_imm(Width::W32, 6, 7, 123, false).unwrap(),
            0x5101_ECE6
        );
        assert_eq!(
            encode_sub_imm(Width::X64, 6, 7, 0x123, true).unwrap(),
            0xD144_8CE6
        );

        assert_eq!(
            encode_and_reg(Width::X64, 9, 10, 11, Shift::Lsl, 0).unwrap(),
            0x8A0B_0149
        );
        assert_eq!(
            encode_orr_reg(Width::X64, 9, 10, 11, Shift::Lsl, 0).unwrap(),
            0xAA0B_0149
        );
        assert_eq!(
            encode_eor_reg(Width::X64, 9, 10, 11, Shift::Lsl, 0).unwrap(),
            0xCA0B_0149
        );

        assert_eq!(
            encode_and_reg(Width::X64, 9, 10, 11, Shift::Lsr, 5).unwrap(),
            0x8A4B_1549
        );
        assert_eq!(
            encode_orr_reg(Width::X64, 9, 10, 11, Shift::Lsr, 5).unwrap(),
            0xAA4B_1549
        );
        assert_eq!(
            encode_eor_reg(Width::X64, 9, 10, 11, Shift::Lsr, 5).unwrap(),
            0xCA4B_1549
        );
    }

    #[test]
    fn encode_shifts_cmp_and_misc() {
        assert_eq!(encode_lsl_reg(Width::X64, 9, 10, 11).unwrap(), 0x9ACB_2149);
        assert_eq!(encode_lsl_reg(Width::W32, 9, 10, 11).unwrap(), 0x1ACB_2149);
        assert_eq!(encode_lsr_reg(Width::X64, 9, 10, 11).unwrap(), 0x9ACB_2549);
        assert_eq!(encode_lsr_reg(Width::W32, 9, 10, 11).unwrap(), 0x1ACB_2549);

        assert_eq!(encode_neg(Width::X64, 12, 13).unwrap(), 0xCB0D_03EC);
        assert_eq!(encode_neg(Width::W32, 12, 13).unwrap(), 0x4B0D_03EC);

        assert_eq!(encode_cmp_reg(Width::X64, 14, 15).unwrap(), 0xEB0F_01DF);
        assert_eq!(encode_cmp_reg(Width::W32, 14, 15).unwrap(), 0x6B0F_01DF);
        assert_eq!(
            encode_cmp_imm(Width::X64, 14, 0xff, false).unwrap(),
            0xF103_FDDF
        );
        assert_eq!(
            encode_cmp_imm(Width::W32, 14, 0xff, false).unwrap(),
            0x7103_FDDF
        );
        assert_eq!(
            encode_cmp_imm(Width::X64, 14, 0x123, true).unwrap(),
            0xF144_8DDF
        );

        assert_eq!(
            encode_cset(Width::X64, 12, Condition::Ne).unwrap(),
            0x9A9F_07EC
        );
        assert_eq!(
            encode_cset(Width::W32, 12, Condition::Ne).unwrap(),
            0x1A9F_07EC
        );
        assert_eq!(
            encode_cset(Width::X64, 12, Condition::Eq).unwrap(),
            0x9A9F_17EC
        );
        assert_eq!(
            encode_cset(Width::X64, 12, Condition::Gt).unwrap(),
            0x9A9F_D7EC
        );

        assert_eq!(encode_sxtb(18, 19).unwrap(), 0x9340_1E72);
        assert_eq!(encode_sxth(18, 19).unwrap(), 0x9340_3E72);
        assert_eq!(encode_sxtw(18, 19).unwrap(), 0x9340_7E72);

        assert_eq!(encode_blr(20).unwrap(), 0xD63F_0280);
    }

    #[test]
    fn encode_branches_and_cb_branches() {
        assert_eq!(encode_b(1).unwrap(), 0x1400_0001);
        assert_eq!(encode_b(-1).unwrap(), 0x17FF_FFFF);
        assert_eq!(encode_bl(3).unwrap(), 0x9400_0003);
        assert_eq!(encode_bl(-4).unwrap(), 0x97FF_FFFC);

        assert_eq!(encode_cbz(Width::X64, 16, 10).unwrap(), 0xB400_0150);
        assert_eq!(encode_cbnz(Width::W32, 17, 9).unwrap(), 0x3500_0131);
        assert_eq!(encode_cbz(Width::X64, 16, -2).unwrap(), 0xB4FF_FFD0);
        assert_eq!(encode_cbnz(Width::W32, 17, -3).unwrap(), 0x35FF_FFB1);
    }

    #[test]
    fn emitter_records_source_map_and_resolves_fixups() {
        let mut emitter = Emitter::new();
        let start = emitter.new_label();
        let done = emitter.new_label();

        emitter.bind_label(start).unwrap();
        emitter.set_source_location(crate::SourceLocation {
            file: 1,
            line: 10,
            column: 1,
        });
        emitter.emit_b_label(done).unwrap();
        emitter.set_source_location(crate::SourceLocation {
            file: 1,
            line: 11,
            column: 1,
        });
        emitter.emit_cbz_label(Width::X64, 16, start).unwrap();
        emitter.bind_label(done).unwrap();
        emitter.set_source_location(crate::SourceLocation {
            file: 1,
            line: 12,
            column: 1,
        });
        emitter.emit_bl_label(start).unwrap();

        let finalized = emitter.finalize().unwrap();
        assert_eq!(
            finalized.source_map,
            vec![
                crate::SourceMapEntry {
                    offset: 0,
                    location: crate::SourceLocation {
                        file: 1,
                        line: 10,
                        column: 1,
                    },
                },
                crate::SourceMapEntry {
                    offset: 4,
                    location: crate::SourceLocation {
                        file: 1,
                        line: 11,
                        column: 1,
                    },
                },
                crate::SourceMapEntry {
                    offset: 8,
                    location: crate::SourceLocation {
                        file: 1,
                        line: 12,
                        column: 1,
                    },
                },
            ]
        );

        assert_eq!(word(&finalized.code[0..4]), 0x1400_0002);
        assert_eq!(word(&finalized.code[4..8]), 0xB4FF_FFF0);
        assert_eq!(word(&finalized.code[8..12]), 0x97FF_FFFE);

        let trace = finalized.trace_text().unwrap();
        assert_eq!(
            trace,
            "00000000 file=1 line=10 col=1 bytes=02000014\n00000004 file=1 line=11 col=1 bytes=f0ffffb4\n00000008 file=1 line=12 col=1 bytes=feffff97"
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
        emitter.emit_b_label(dangling).unwrap();
        let err = emitter.finalize().unwrap_err();
        assert!(matches!(err, EmitError::UnboundLabel { .. }));
    }

    #[test]
    fn emitter_reports_out_of_range_fixup() {
        let mut emitter = Emitter::new();
        let far = emitter.new_label();
        emitter.emit_cbz_label(Width::X64, 0, far).unwrap();
        for _ in 0..=262_143 {
            emitter.emit_word(0xD503_201F); // nop
        }
        emitter.bind_label(far).unwrap();

        let err = emitter.finalize().unwrap_err();
        assert!(matches!(
            err,
            EmitError::InvalidImmediate {
                instruction: "branch19",
                ..
            }
        ));
    }

    #[test]
    fn encode_boundaries_and_rejections() {
        assert_eq!(encode_b(-(1 << 25)).unwrap(), 0x1600_0000);
        assert_eq!(encode_b((1 << 25) - 1).unwrap(), 0x15FF_FFFF);
        assert!(matches!(
            encode_b(-(1 << 25) - 1),
            Err(EmitError::InvalidImmediate {
                instruction: "b",
                ..
            })
        ));
        assert!(matches!(
            encode_b(1 << 25),
            Err(EmitError::InvalidImmediate {
                instruction: "b",
                ..
            })
        ));

        assert_eq!(encode_cbz(Width::X64, 0, -(1 << 18)).unwrap(), 0xB480_0000);
        assert_eq!(
            encode_cbz(Width::X64, 0, (1 << 18) - 1).unwrap(),
            0xB47F_FFE0
        );
        assert!(matches!(
            encode_cbz(Width::X64, 0, -(1 << 18) - 1),
            Err(EmitError::InvalidImmediate {
                instruction: "cbz",
                ..
            })
        ));
        assert!(matches!(
            encode_cbz(Width::X64, 0, 1 << 18),
            Err(EmitError::InvalidImmediate {
                instruction: "cbz",
                ..
            })
        ));

        assert_eq!(
            encode_ldr_imm(Width::X64, 1, 2, 4095 * 8).unwrap(),
            0xF97F_FC41
        );
        assert!(matches!(
            encode_ldr_imm(Width::X64, 1, 2, 4095 * 8 + 8),
            Err(EmitError::InvalidOffset { .. })
        ));
        assert!(matches!(
            encode_ldr_imm(Width::X64, 1, 2, 10),
            Err(EmitError::InvalidOffset { .. })
        ));

        assert!(matches!(
            encode_movz(Width::W32, 1, 0x1234, 32),
            Err(EmitError::InvalidMovWideShift { .. })
        ));
        assert!(matches!(
            encode_add_imm(Width::X64, 0, 0, 0x1000, false),
            Err(EmitError::InvalidImmediate {
                instruction: "add/sub imm12",
                ..
            })
        ));
    }

    #[test]
    fn emitter_rejects_double_label_bind() {
        let mut emitter = Emitter::new();
        let label = emitter.new_label();
        emitter.bind_label(label).unwrap();
        let err = emitter.bind_label(label).unwrap_err();
        assert!(matches!(err, EmitError::LabelAlreadyBound { .. }));
    }
}
