//! Instruction emission: binop, unary, read/write, branch, jump table.

use super::*;
use kajit_emit::aarch64::{self, Condition, Reg};

impl Lowerer {
    pub(super) fn emit_read_from_field(&mut self, dst: crate::ir::VReg, offset: u32, width: Width) {
        match width {
            Width::W1 => self.ectx
                .emit
                .emit_word(aarch64::encode_ldrb_imm(Reg::X9, Reg::X21, offset).expect("ldrb")),
            Width::W2 => self.ectx
                .emit
                .emit_word(aarch64::encode_ldrh_imm(Reg::X9, Reg::X21, offset).expect("ldrh")),
            Width::W4 => self.ectx
                .emit
                .emit_word(
                    aarch64::encode_ldr_imm(aarch64::Width::W32, Reg::X9, Reg::X21, offset)
                        .expect("ldr"),
                ),
            Width::W8 => self.ectx
                .emit
                .emit_word(
                    aarch64::encode_ldr_imm(aarch64::Width::X64, Reg::X9, Reg::X21, offset)
                        .expect("ldr"),
                ),
        }
        self.emit_store_def_x9(dst, 0);
        self.set_const(dst, None);
    }

    pub(super) fn emit_write_to_field(&mut self, src: crate::ir::VReg, offset: u32, width: Width) {
        self.emit_load_use_x9(src, 0);
        match width {
            Width::W1 => self.ectx
                .emit
                .emit_word(aarch64::encode_strb_imm(Reg::X9, Reg::X21, offset).expect("strb")),
            Width::W2 => self.ectx
                .emit
                .emit_word(aarch64::encode_strh_imm(Reg::X9, Reg::X21, offset).expect("strh")),
            Width::W4 => self.ectx.emit.emit_word(
                aarch64::encode_str_imm(aarch64::Width::W32, Reg::X9, Reg::X21, offset)
                    .expect("str"),
            ),
            Width::W8 => self.ectx.emit.emit_word(
                aarch64::encode_str_imm(aarch64::Width::X64, Reg::X9, Reg::X21, offset)
                    .expect("str"),
            ),
        }
    }

    pub(super) fn emit_save_out_ptr(&mut self, dst: crate::ir::VReg) {
        self.ectx
            .emit
            .emit_word(aarch64::encode_mov_reg(aarch64::Width::X64, Reg::X9, Reg::X21).expect("mov"));
        self.emit_store_def_x9(dst, 0);
        self.set_const(dst, None);
    }

    pub(super) fn emit_set_out_ptr(&mut self, src: crate::ir::VReg) {
        self.emit_load_use_x9(src, 0);
        self.ectx
            .emit
            .emit_word(aarch64::encode_mov_reg(aarch64::Width::X64, Reg::X21, Reg::X9).expect("mov"));
    }

    pub(super) fn emit_slot_addr(&mut self, dst: crate::ir::VReg, slot: crate::ir::SlotId) {
        let slot_off = self.slot_off(slot);
        self.ectx
            .emit
            .emit_word(aarch64::encode_add_imm(
                aarch64::Width::X64,
                Reg::X9,
                Reg::SP,
                slot_off as u16,
                false,
            ).expect("add"));
        self.emit_store_def_x9(dst, 0);
        self.set_const(dst, None);
    }

    pub(super) fn emit_read_bytes(&mut self, dst: crate::ir::VReg, count: u32) {
        self.emit_recipe_ops(vec![Op::BoundsCheck { count }]);
        match count {
            1 => self.ectx
                .emit
                .emit_word(aarch64::encode_ldrb_imm(Reg::X9, Reg::X19, 0).expect("ldrb")),
            2 => self.ectx
                .emit
                .emit_word(aarch64::encode_ldrh_imm(Reg::X9, Reg::X19, 0).expect("ldrh")),
            4 => self.ectx
                .emit
                .emit_word(aarch64::encode_ldr_imm(aarch64::Width::W32, Reg::X9, Reg::X19, 0).expect("ldr")),
            8 => self.ectx
                .emit
                .emit_word(aarch64::encode_ldr_imm(aarch64::Width::X64, Reg::X9, Reg::X19, 0).expect("ldr")),
            _ => panic!("unsupported ReadBytes count: {count}"),
        }
        self.emit_store_def_x9(dst, 0);
        self.set_const(dst, None);
        self.ectx.emit_advance_cursor_by(count);
    }

    pub(super) fn emit_peek_byte(&mut self, dst: crate::ir::VReg) {
        self.emit_recipe_ops(vec![Op::BoundsCheck { count: 1 }]);
        self.ectx
            .emit
            .emit_word(aarch64::encode_ldrb_imm(Reg::X9, Reg::X19, 0).expect("ldrb"));
        self.emit_store_def_x9(dst, 0);
        self.set_const(dst, None);
    }

    pub(super) fn emit_binop(
        &mut self,
        kind: BinOpKind,
        dst: crate::ir::VReg,
        lhs: crate::ir::VReg,
        rhs: crate::ir::VReg,
    ) {
        if kind == BinOpKind::CmpNe {
            let lhs_alloc = self.current_alloc(0);
            let rhs_alloc = self.current_alloc(1);
            let rhs_const = self.const_of(rhs);

            if let Some(reg) = lhs_alloc.as_reg() {
                assert!(
                    reg.class() == regalloc2::RegClass::Int,
                    "unsupported register allocation class {:?} for CmpNe lhs",
                    reg.class()
                );
            }
            if let Some(reg) = rhs_alloc.as_reg() {
                assert!(
                    reg.class() == regalloc2::RegClass::Int,
                    "unsupported register allocation class {:?} for CmpNe rhs",
                    reg.class()
                );
            }

            match (
                lhs_alloc.as_reg(),
                lhs_alloc.as_stack(),
                rhs_const,
                rhs_alloc.as_reg(),
                rhs_alloc.as_stack(),
            ) {
                (Some(lhs_reg), None, Some(c), _, _) => {
                    let lhs_r = lhs_reg.hw_enc() as u8;
                    self.emit_load_u64_x10(c);
                    self.ectx.emit.emit_word(
                        aarch64::encode_cmp_reg(aarch64::Width::X64, Reg::from_raw(lhs_r), Reg::X10)
                            .expect("cmp"),
                    );
                }
                (None, Some(lhs_stack), Some(c), _, _) => {
                    let lhs_off = self.spill_off(lhs_stack);
                    self.ectx
                        .emit
                        .emit_word(aarch64::encode_ldr_imm(aarch64::Width::X64, Reg::X9, Reg::SP, lhs_off).expect("ldr"));
                    if c <= 4095 {
                        self.ectx.emit.emit_word(
                            aarch64::encode_cmp_imm(aarch64::Width::X64, Reg::X9, c as u16, false)
                                .expect("cmp"),
                        );
                    } else {
                        self.emit_load_u64_x10(c);
                        self.ectx.emit.emit_word(
                            aarch64::encode_cmp_reg(aarch64::Width::X64, Reg::X9, Reg::X10)
                                .expect("cmp"),
                        );
                    }
                }
                (Some(lhs_reg), None, None, Some(rhs_reg), None) => {
                    let lhs_r = lhs_reg.hw_enc() as u8;
                    let rhs_r = rhs_reg.hw_enc() as u8;
                    self.ectx.emit.emit_word(
                        aarch64::encode_cmp_reg(
                            aarch64::Width::X64,
                            Reg::from_raw(lhs_r),
                            Reg::from_raw(rhs_r),
                        )
                        .expect("cmp"),
                    );
                }
                (Some(lhs_reg), None, None, None, Some(rhs_stack)) => {
                    let lhs_r = lhs_reg.hw_enc() as u8;
                    let rhs_off = self.spill_off(rhs_stack);
                    self.ectx.emit.emit_word(
                        aarch64::encode_ldr_imm(aarch64::Width::X64, Reg::X10, Reg::SP, rhs_off)
                            .expect("ldr"),
                    );
                    self.ectx.emit.emit_word(
                        aarch64::encode_cmp_reg(aarch64::Width::X64, Reg::from_raw(lhs_r), Reg::X10)
                            .expect("cmp"),
                    );
                }
                (None, Some(lhs_stack), None, Some(rhs_reg), None) => {
                    let lhs_off = self.spill_off(lhs_stack);
                    let rhs_r = rhs_reg.hw_enc() as u8;
                    self.ectx.emit.emit_word(
                        aarch64::encode_ldr_imm(aarch64::Width::X64, Reg::X9, Reg::SP, lhs_off)
                            .expect("ldr"),
                    );
                    self.ectx.emit.emit_word(
                        aarch64::encode_cmp_reg(aarch64::Width::X64, Reg::X9, Reg::from_raw(rhs_r))
                            .expect("cmp"),
                    );
                }
                (None, Some(lhs_stack), None, None, Some(rhs_stack)) => {
                    let lhs_off = self.spill_off(lhs_stack);
                    let rhs_off = self.spill_off(rhs_stack);
                    self.ectx.emit.emit_word(
                        aarch64::encode_ldr_imm(aarch64::Width::X64, Reg::X9, Reg::SP, lhs_off)
                            .expect("ldr"),
                    );
                    self.ectx.emit.emit_word(
                        aarch64::encode_ldr_imm(aarch64::Width::X64, Reg::X10, Reg::SP, rhs_off)
                            .expect("ldr"),
                    );
                    self.ectx.emit.emit_word(
                        aarch64::encode_cmp_reg(aarch64::Width::X64, Reg::X9, Reg::X10)
                            .expect("cmp"),
                    );
                }
                _ => panic!("unexpected none allocation for CmpNe operands"),
            }

            let dst_alloc = self.current_alloc(2);
            if let Some(dst_reg) = dst_alloc.as_reg() {
                assert!(
                    dst_reg.class() == regalloc2::RegClass::Int,
                    "unsupported register allocation class {:?} for CmpNe dst",
                    dst_reg.class()
                );
                let dst_r = dst_reg.hw_enc() as u8;
                self.ectx.emit.emit_word(
                    aarch64::encode_cset(aarch64::Width::X64, Reg::from_raw(dst_r), Condition::Ne)
                        .expect("cset"),
                );
            } else if let Some(dst_stack) = dst_alloc.as_stack() {
                let dst_off = self.spill_off(dst_stack);
                self.ectx
                    .emit
                    .emit_word(aarch64::encode_cset(aarch64::Width::X64, Reg::X9, Condition::Ne).expect("cset"));
                self.ectx.emit.emit_word(
                    aarch64::encode_str_imm(aarch64::Width::X64, Reg::X9, Reg::SP, dst_off)
                        .expect("str"),
                );
            } else {
                panic!("unexpected none allocation for CmpNe dst");
            }
            self.set_const(dst, None);
            return;
        }

        let lhs_alloc = self.current_alloc(0);
        let rhs_alloc = self.current_alloc(1);
        let dst_alloc = self.current_alloc(2);
        if let (Some(lhs_reg), Some(rhs_reg), Some(dst_reg)) =
            (lhs_alloc.as_reg(), rhs_alloc.as_reg(), dst_alloc.as_reg())
        {
            assert!(
                lhs_reg.class() == regalloc2::RegClass::Int,
                "unsupported register allocation class {:?} for binop lhs",
                lhs_reg.class()
            );
            assert!(
                rhs_reg.class() == regalloc2::RegClass::Int,
                "unsupported register allocation class {:?} for binop rhs",
                rhs_reg.class()
            );
            assert!(
                dst_reg.class() == regalloc2::RegClass::Int,
                "unsupported register allocation class {:?} for binop dst",
                dst_reg.class()
            );

            let lhs_r = lhs_reg.hw_enc() as u8;
            let rhs_r = rhs_reg.hw_enc() as u8;
            let dst_r = dst_reg.hw_enc() as u8;

            let handled = match kind {
                BinOpKind::Add => {
                    if dst_r == lhs_r {
                        self.ectx.emit.emit_word(aarch64::encode_add_reg(
                            aarch64::Width::X64,
                            Reg::from_raw(dst_r),
                            Reg::from_raw(dst_r),
                            Reg::from_raw(rhs_r),
                        ).expect("add"));
                    } else if dst_r == rhs_r {
                        self.ectx.emit.emit_word(aarch64::encode_add_reg(
                            aarch64::Width::X64,
                            Reg::from_raw(dst_r),
                            Reg::from_raw(dst_r),
                            Reg::from_raw(lhs_r),
                        ).expect("add"));
                    } else {
                        self.ectx.emit.emit_word(aarch64::encode_mov_reg(
                            aarch64::Width::X64,
                            Reg::from_raw(dst_r),
                            Reg::from_raw(lhs_r),
                        ).expect("mov"));
                        self.ectx.emit.emit_word(aarch64::encode_add_reg(
                            aarch64::Width::X64,
                            Reg::from_raw(dst_r),
                            Reg::from_raw(dst_r),
                            Reg::from_raw(rhs_r),
                        ).expect("add"));
                    }
                    true
                }
                BinOpKind::Sub => {
                    if dst_r == lhs_r {
                        self.ectx.emit.emit_word(aarch64::encode_sub_reg(
                            aarch64::Width::X64,
                            Reg::from_raw(dst_r),
                            Reg::from_raw(dst_r),
                            Reg::from_raw(rhs_r),
                        ).expect("sub"));
                        true
                    } else if dst_r != rhs_r {
                        self.ectx.emit.emit_word(aarch64::encode_mov_reg(
                            aarch64::Width::X64,
                            Reg::from_raw(dst_r),
                            Reg::from_raw(lhs_r),
                        ).expect("mov"));
                        self.ectx.emit.emit_word(aarch64::encode_sub_reg(
                            aarch64::Width::X64,
                            Reg::from_raw(dst_r),
                            Reg::from_raw(dst_r),
                            Reg::from_raw(rhs_r),
                        ).expect("sub"));
                        true
                    } else {
                        false
                    }
                }
                BinOpKind::And => {
                    if dst_r == lhs_r {
                        self.ectx.emit.emit_word(aarch64::encode_and_reg(
                            aarch64::Width::X64,
                            Reg::from_raw(dst_r),
                            Reg::from_raw(dst_r),
                            Reg::from_raw(rhs_r),
                            aarch64::Shift::Lsl,
                            0,
                        ).expect("and"));
                    } else if dst_r == rhs_r {
                        self.ectx.emit.emit_word(aarch64::encode_and_reg(
                            aarch64::Width::X64,
                            Reg::from_raw(dst_r),
                            Reg::from_raw(dst_r),
                            Reg::from_raw(lhs_r),
                            aarch64::Shift::Lsl,
                            0,
                        ).expect("and"));
                    } else {
                        self.ectx.emit.emit_word(aarch64::encode_mov_reg(
                            aarch64::Width::X64,
                            Reg::from_raw(dst_r),
                            Reg::from_raw(lhs_r),
                        ).expect("mov"));
                        self.ectx.emit.emit_word(aarch64::encode_and_reg(
                            aarch64::Width::X64,
                            Reg::from_raw(dst_r),
                            Reg::from_raw(dst_r),
                            Reg::from_raw(rhs_r),
                            aarch64::Shift::Lsl,
                            0,
                        ).expect("and"));
                    }
                    true
                }
                BinOpKind::Or => {
                    if dst_r == lhs_r {
                        self.ectx.emit.emit_word(aarch64::encode_orr_reg(
                            aarch64::Width::X64,
                            Reg::from_raw(dst_r),
                            Reg::from_raw(dst_r),
                            Reg::from_raw(rhs_r),
                            aarch64::Shift::Lsl,
                            0,
                        ).expect("orr"));
                    } else if dst_r == rhs_r {
                        self.ectx.emit.emit_word(aarch64::encode_orr_reg(
                            aarch64::Width::X64,
                            Reg::from_raw(dst_r),
                            Reg::from_raw(dst_r),
                            Reg::from_raw(lhs_r),
                            aarch64::Shift::Lsl,
                            0,
                        ).expect("orr"));
                    } else {
                        self.ectx.emit.emit_word(aarch64::encode_mov_reg(
                            aarch64::Width::X64,
                            Reg::from_raw(dst_r),
                            Reg::from_raw(lhs_r),
                        ).expect("mov"));
                        self.ectx.emit.emit_word(aarch64::encode_orr_reg(
                            aarch64::Width::X64,
                            Reg::from_raw(dst_r),
                            Reg::from_raw(dst_r),
                            Reg::from_raw(rhs_r),
                            aarch64::Shift::Lsl,
                            0,
                        ).expect("orr"));
                    }
                    true
                }
                BinOpKind::Xor => {
                    if dst_r == lhs_r {
                        self.ectx.emit.emit_word(aarch64::encode_eor_reg(
                            aarch64::Width::X64,
                            Reg::from_raw(dst_r),
                            Reg::from_raw(dst_r),
                            Reg::from_raw(rhs_r),
                            aarch64::Shift::Lsl,
                            0,
                        ).expect("eor"));
                    } else if dst_r == rhs_r {
                        self.ectx.emit.emit_word(aarch64::encode_eor_reg(
                            aarch64::Width::X64,
                            Reg::from_raw(dst_r),
                            Reg::from_raw(dst_r),
                            Reg::from_raw(lhs_r),
                            aarch64::Shift::Lsl,
                            0,
                        ).expect("eor"));
                    } else {
                        self.ectx.emit.emit_word(aarch64::encode_mov_reg(
                            aarch64::Width::X64,
                            Reg::from_raw(dst_r),
                            Reg::from_raw(lhs_r),
                        ).expect("mov"));
                        self.ectx.emit.emit_word(aarch64::encode_eor_reg(
                            aarch64::Width::X64,
                            Reg::from_raw(dst_r),
                            Reg::from_raw(dst_r),
                            Reg::from_raw(rhs_r),
                            aarch64::Shift::Lsl,
                            0,
                        ).expect("eor"));
                    }
                    true
                }
                BinOpKind::Shr => {
                    if dst_r == lhs_r {
                        self.ectx.emit.emit_word(aarch64::encode_lsr_reg(
                            aarch64::Width::X64,
                            Reg::from_raw(dst_r),
                            Reg::from_raw(dst_r),
                            Reg::from_raw(rhs_r),
                        ).expect("lsr"));
                        true
                    } else if dst_r != rhs_r {
                        self.ectx.emit.emit_word(aarch64::encode_mov_reg(
                            aarch64::Width::X64,
                            Reg::from_raw(dst_r),
                            Reg::from_raw(lhs_r),
                        ).expect("mov"));
                        self.ectx.emit.emit_word(aarch64::encode_lsr_reg(
                            aarch64::Width::X64,
                            Reg::from_raw(dst_r),
                            Reg::from_raw(dst_r),
                            Reg::from_raw(rhs_r),
                        ).expect("lsr"));
                        true
                    } else {
                        false
                    }
                }
                BinOpKind::Shl => {
                    if dst_r == lhs_r {
                        self.ectx.emit.emit_word(aarch64::encode_lsl_reg(
                            aarch64::Width::X64,
                            Reg::from_raw(dst_r),
                            Reg::from_raw(dst_r),
                            Reg::from_raw(rhs_r),
                        ).expect("lsl"));
                        true
                    } else if dst_r != rhs_r {
                        self.ectx.emit.emit_word(aarch64::encode_mov_reg(
                            aarch64::Width::X64,
                            Reg::from_raw(dst_r),
                            Reg::from_raw(lhs_r),
                        ).expect("mov"));
                        self.ectx.emit.emit_word(aarch64::encode_lsl_reg(
                            aarch64::Width::X64,
                            Reg::from_raw(dst_r),
                            Reg::from_raw(dst_r),
                            Reg::from_raw(rhs_r),
                        ).expect("lsl"));
                        true
                    } else {
                        false
                    }
                }
                BinOpKind::CmpNe => unreachable!("CmpNe handled above"),
            };
            if handled {
                self.set_const(dst, None);
                return;
            }
        }

        self.emit_load_use_x9(lhs, 0);
        let rhs_const = self.const_of(rhs);
        match kind {
            BinOpKind::Add => {
                if let Some(c) = rhs_const
                    && c <= 4095
                {
                    dynasm!(self.ectx.ops ; .arch aarch64 ; add x9, x9, c as u32);
                } else {
                    self.emit_load_use_x10(rhs, 1);
                    dynasm!(self.ectx.ops ; .arch aarch64 ; add x9, x9, x10);
                }
            }
            BinOpKind::Sub => {
                if let Some(c) = rhs_const
                    && c <= 4095
                {
                    dynasm!(self.ectx.ops ; .arch aarch64 ; sub x9, x9, c as u32);
                } else {
                    self.emit_load_use_x10(rhs, 1);
                    dynasm!(self.ectx.ops ; .arch aarch64 ; sub x9, x9, x10);
                }
            }
            BinOpKind::And => {
                if matches!(rhs_const, Some(0x7f | 0x7e | 0x80 | 0x1)) {
                    let c = rhs_const.expect("just matched Some");
                    dynasm!(self.ectx.ops ; .arch aarch64 ; and x9, x9, c);
                } else {
                    self.emit_load_use_x10(rhs, 1);
                    dynasm!(self.ectx.ops ; .arch aarch64 ; and x9, x9, x10);
                }
            }
            BinOpKind::Or => {
                self.emit_load_use_x10(rhs, 1);
                dynasm!(self.ectx.ops ; .arch aarch64 ; orr x9, x9, x10);
            }
            BinOpKind::Xor => {
                self.emit_load_use_x10(rhs, 1);
                dynasm!(self.ectx.ops ; .arch aarch64 ; eor x9, x9, x10);
            }
            BinOpKind::CmpNe => unreachable!("CmpNe handled above"),
            BinOpKind::Shr => {
                if let Some(c) = rhs_const
                    && c <= 63
                {
                    dynasm!(self.ectx.ops ; .arch aarch64 ; lsr x9, x9, c as u32);
                } else {
                    self.emit_load_use_x10(rhs, 1);
                    dynasm!(self.ectx.ops ; .arch aarch64 ; lsr x9, x9, x10);
                }
            }
            BinOpKind::Shl => {
                if let Some(c) = rhs_const
                    && c <= 63
                {
                    dynasm!(self.ectx.ops ; .arch aarch64 ; lsl x9, x9, c as u32);
                } else {
                    self.emit_load_use_x10(rhs, 1);
                    dynasm!(self.ectx.ops ; .arch aarch64 ; lsl x9, x9, x10);
                }
            }
        }
        self.emit_store_def_x9(dst, 2);
        self.set_const(dst, None);
    }

    pub(super) fn emit_unary(
        &mut self,
        kind: UnaryOpKind,
        dst: crate::ir::VReg,
        src: crate::ir::VReg,
    ) {
        self.emit_load_use_x9(src, 0);
        match kind {
            UnaryOpKind::ZigzagDecode { wide: true } => {
                dynasm!(self.ectx.ops
                    ; .arch aarch64
                    ; lsr x10, x9, #1
                    ; and x16, x9, #1
                    ; neg x16, x16
                    ; eor x9, x10, x16
                );
            }
            UnaryOpKind::ZigzagDecode { wide: false } => {
                dynasm!(self.ectx.ops
                    ; .arch aarch64
                    ; lsr w10, w9, #1
                    ; and w16, w9, #1
                    ; neg w16, w16
                    ; eor w9, w10, w16
                );
            }
            UnaryOpKind::SignExtend { from_width } => match from_width {
                Width::W1 => dynasm!(self.ectx.ops ; .arch aarch64 ; sxtb x9, w9),
                Width::W2 => dynasm!(self.ectx.ops ; .arch aarch64 ; sxth x9, w9),
                Width::W4 => dynasm!(self.ectx.ops ; .arch aarch64 ; sxtw x9, w9),
                Width::W8 => {}
            },
        }
        self.emit_store_def_x9(dst, 1);
        self.set_const(dst, None);
    }

    pub(super) fn emit_branch_if(
        &mut self,
        cond: crate::ir::VReg,
        target: LabelId,
        invert: bool,
    ) {
        let _ = cond;
        let alloc = self.current_alloc(0);
        self.emit_branch_if_allocation(alloc, target, invert);
    }

    pub(super) fn emit_branch_if_allocation(
        &mut self,
        alloc: Allocation,
        target: LabelId,
        invert: bool,
    ) {
        if let Some(reg) = alloc.as_reg() {
            assert!(
                reg.class() == regalloc2::RegClass::Int,
                "unsupported register allocation class {:?} for branch condition",
                reg.class()
            );
            let r = reg.hw_enc() as u8;
            if invert {
                dynasm!(self.ectx.ops ; .arch aarch64 ; cbz X(r), =>target);
            } else {
                dynasm!(self.ectx.ops ; .arch aarch64 ; cbnz X(r), =>target);
            }
            return;
        }
        if let Some(slot) = alloc.as_stack() {
            let off = self.spill_off(slot);
            dynasm!(self.ectx.ops ; .arch aarch64 ; ldr x9, [sp, #off]);
            if invert {
                dynasm!(self.ectx.ops ; .arch aarch64 ; cbz x9, =>target);
            } else {
                dynasm!(self.ectx.ops ; .arch aarch64 ; cbnz x9, =>target);
            }
            return;
        }
        panic!("unexpected none allocation for branch condition");
    }

    pub(super) fn emit_jump_table(
        &mut self,
        lambda_id: u32,
        predicate: crate::ir::VReg,
        term: &RaTerminator,
        linear_op_index: usize,
    ) {
        let RaTerminator::JumpTable {
            targets, default, ..
        } = term
        else {
            unreachable!();
        };
        let _ = predicate;
        let alloc = self.current_alloc(0);
        let pred_reg = alloc.as_reg().map(|r| {
            assert!(
                r.class() == regalloc2::RegClass::Int,
                "unsupported register allocation class {:?} for jumptable predicate",
                r.class()
            );
            r.hw_enc() as u8
        });
        if let Some(slot) = alloc.as_stack() {
            let off = self.spill_off(slot);
            dynasm!(self.ectx.ops ; .arch aarch64 ; ldr x9, [sp, #off]);
        } else if pred_reg.is_none() {
            panic!("unexpected none allocation for jumptable predicate");
        }
        for (index, target_block) in targets.iter().enumerate() {
            let resolved = self.resolve_forwarded_block(lambda_id, *target_block);
            let target = self.edge_target_label(
                linear_op_index,
                index,
                self.block_label(lambda_id, resolved),
            );
            let idx = index as u32;
            if let Some(r) = pred_reg {
                self.emit_load_u32_w10(idx);
                dynasm!(self.ectx.ops
                    ; .arch aarch64
                    ; cmp X(r), x10
                    ; b.eq =>target
                );
            } else if idx <= 4095 {
                dynasm!(self.ectx.ops
                    ; .arch aarch64
                    ; cmp w9, idx
                    ; b.eq =>target
                );
            } else {
                self.emit_load_u32_w10(idx);
                dynasm!(self.ectx.ops
                    ; .arch aarch64
                    ; cmp w9, w10
                    ; b.eq =>target
                );
            }
        }
        let default_succ_index = targets.len();
        let resolved_default = self.resolve_forwarded_block(lambda_id, *default);
        let default_target = self.edge_target_label(
            linear_op_index,
            default_succ_index,
            self.block_label(lambda_id, resolved_default),
        );
        self.ectx.emit_branch(default_target);
    }
}
