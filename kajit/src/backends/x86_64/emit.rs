use super::*;
use kajit_emit::x64::{self, LabelId, Mem, Operand};

impl Lowerer {
    pub(super) fn emit_read_from_field(&mut self, dst: crate::ir::VReg, offset: u32, width: Width) {
        let off = offset as i32;
        match width {
            Width::W1 => self
                .ectx
                .emit
                .emit_with(|buf| {
                    x64::encode_movzx_r32_rm8(
                        10,
                        Operand::Mem(Mem {
                            base: 14,
                            disp: off,
                        }),
                        buf,
                    )
                })
                .expect("movzx"),
            Width::W2 => self
                .ectx
                .emit
                .emit_with(|buf| {
                    x64::encode_movzx_r32_rm16(
                        10,
                        Operand::Mem(Mem {
                            base: 14,
                            disp: off,
                        }),
                        buf,
                    )
                })
                .expect("movzx"),
            Width::W4 => self
                .ectx
                .emit
                .emit_with(|buf| {
                    x64::encode_mov_r32_m(
                        10,
                        Mem {
                            base: 14,
                            disp: off,
                        },
                        buf,
                    )
                })
                .expect("mov"),
            Width::W8 => self
                .ectx
                .emit
                .emit_with(|buf| {
                    x64::encode_mov_r64_m(
                        10,
                        Mem {
                            base: 14,
                            disp: off,
                        },
                        buf,
                    )
                })
                .expect("mov"),
        }
        self.emit_store_def_r10(dst, 0);
        self.set_const(dst, None);
    }

    pub(super) fn emit_write_to_field(&mut self, src: crate::ir::VReg, offset: u32, width: Width) {
        self.emit_load_use_r10(src, 0);
        let off = offset as i32;
        match width {
            Width::W1 => self
                .ectx
                .emit
                .emit_with(|buf| x64::encode_mov_m_r8(14, off, 10, buf))
                .expect("mov"),
            Width::W2 => self
                .ectx
                .emit
                .emit_with(|buf| x64::encode_mov_m_r16(14, off, 10, buf))
                .expect("mov"),
            Width::W4 => self
                .ectx
                .emit
                .emit_with(|buf| {
                    x64::encode_mov_m_r32(
                        Mem {
                            base: 14,
                            disp: off,
                        },
                        10,
                        buf,
                    )
                })
                .expect("mov"),
            Width::W8 => self
                .ectx
                .emit
                .emit_with(|buf| {
                    x64::encode_mov_m_r64(
                        Mem {
                            base: 14,
                            disp: off,
                        },
                        10,
                        buf,
                    )
                })
                .expect("mov"),
        }
    }

    pub(super) fn emit_save_out_ptr(&mut self, dst: crate::ir::VReg) {
        self.ectx
            .emit
            .emit_with(|buf| x64::encode_mov_r64_r64(10, 14, buf))
            .expect("mov");
        self.emit_store_def_r10(dst, 0);
        self.set_const(dst, None);
    }

    pub(super) fn emit_set_out_ptr(&mut self, src: crate::ir::VReg) {
        self.emit_load_use_r10(src, 0);
        self.ectx
            .emit
            .emit_with(|buf| x64::encode_mov_r64_r64(14, 10, buf))
            .expect("mov");
    }

    pub(super) fn emit_slot_addr(&mut self, dst: crate::ir::VReg, slot: crate::ir::SlotId) {
        let slot_off = self.slot_off(slot) as i32;
        self.ectx
            .emit
            .emit_with(|buf| {
                x64::encode_lea_r64_m(
                    10,
                    Mem {
                        base: 4,
                        disp: slot_off,
                    },
                    buf,
                )
            })
            .expect("lea");
        self.emit_store_def_r10(dst, 0);
        self.set_const(dst, None);
    }

    pub(super) fn emit_read_bytes(&mut self, dst: crate::ir::VReg, count: u32) {
        self.emit_recipe_ops(vec![Op::BoundsCheck { count }]);
        match count {
            1 => self
                .ectx
                .emit
                .emit_with(|buf| {
                    x64::encode_movzx_r32_rm8(10, Operand::Mem(Mem { base: 12, disp: 0 }), buf)
                })
                .expect("movzx"),
            2 => self
                .ectx
                .emit
                .emit_with(|buf| {
                    x64::encode_movzx_r32_rm16(10, Operand::Mem(Mem { base: 12, disp: 0 }), buf)
                })
                .expect("movzx"),
            4 => self
                .ectx
                .emit
                .emit_with(|buf| x64::encode_mov_r32_m(10, Mem { base: 12, disp: 0 }, buf))
                .expect("mov"),
            8 => self
                .ectx
                .emit
                .emit_with(|buf| x64::encode_mov_r64_m(10, Mem { base: 12, disp: 0 }, buf))
                .expect("mov"),
            _ => panic!("unsupported ReadBytes count: {count}"),
        }
        self.emit_store_def_r10(dst, 0);
        self.set_const(dst, None);
        self.ectx.emit_advance_cursor_by(count);
    }

    pub(super) fn emit_peek_byte(&mut self, dst: crate::ir::VReg) {
        self.emit_recipe_ops(vec![Op::BoundsCheck { count: 1 }]);
        self.ectx
            .emit
            .emit_with(|buf| {
                x64::encode_movzx_r32_rm8(10, Operand::Mem(Mem { base: 12, disp: 0 }), buf)
            })
            .expect("movzx");
        self.emit_store_def_r10(dst, 0);
        self.set_const(dst, None);
    }

    pub(super) fn emit_binop(
        &mut self,
        kind: BinOpKind,
        dst: crate::ir::VReg,
        lhs: crate::ir::VReg,
        _rhs: crate::ir::VReg,
    ) {
        self.emit_load_use_r10(lhs, 0);
        match kind {
            BinOpKind::Add => {
                let rhs_alloc = self.current_alloc(1);
                if let Some(reg) = rhs_alloc.as_reg() {
                    let enc = reg.hw_enc() as u8;
                    self.ectx
                        .emit
                        .emit_with(|buf| x64::encode_add_r64_r64(10, enc, buf))
                        .expect("add");
                } else if let Some(slot) = rhs_alloc.as_stack() {
                    let off = self.spill_off(slot) as i32;
                    self.ectx
                        .emit
                        .emit_with(|buf| x64::encode_add_r64_m(10, Mem { base: 4, disp: off }, buf))
                        .expect("add");
                }
            }
            BinOpKind::Sub => {
                let rhs_alloc = self.current_alloc(1);
                if let Some(reg) = rhs_alloc.as_reg() {
                    let enc = reg.hw_enc() as u8;
                    self.ectx
                        .emit
                        .emit_with(|buf| x64::encode_sub_r64_r64(10, enc, buf))
                        .expect("sub");
                } else if let Some(slot) = rhs_alloc.as_stack() {
                    let off = self.spill_off(slot) as i32;
                    self.ectx
                        .emit
                        .emit_with(|buf| x64::encode_sub_r64_m(10, Mem { base: 4, disp: off }, buf))
                        .expect("sub");
                }
            }
            BinOpKind::And => {
                let rhs_alloc = self.current_alloc(1);
                if let Some(reg) = rhs_alloc.as_reg() {
                    let enc = reg.hw_enc() as u8;
                    self.ectx
                        .emit
                        .emit_with(|buf| x64::encode_and_r64_r64(10, enc, buf))
                        .expect("and");
                } else if let Some(slot) = rhs_alloc.as_stack() {
                    let off = self.spill_off(slot) as i32;
                    self.ectx
                        .emit
                        .emit_with(|buf| x64::encode_and_r64_m(10, Mem { base: 4, disp: off }, buf))
                        .expect("and");
                }
            }
            BinOpKind::Or => {
                let rhs_alloc = self.current_alloc(1);
                if let Some(reg) = rhs_alloc.as_reg() {
                    let enc = reg.hw_enc() as u8;
                    self.ectx
                        .emit
                        .emit_with(|buf| x64::encode_or_r64_r64(10, enc, buf))
                        .expect("or");
                } else if let Some(slot) = rhs_alloc.as_stack() {
                    let off = self.spill_off(slot) as i32;
                    self.ectx
                        .emit
                        .emit_with(|buf| x64::encode_or_r64_m(10, Mem { base: 4, disp: off }, buf))
                        .expect("or");
                }
            }
            BinOpKind::Xor => {
                let rhs_alloc = self.current_alloc(1);
                if let Some(reg) = rhs_alloc.as_reg() {
                    let enc = reg.hw_enc() as u8;
                    self.ectx
                        .emit
                        .emit_with(|buf| x64::encode_xor_r64_r64(10, enc, buf))
                        .expect("xor");
                } else if let Some(slot) = rhs_alloc.as_stack() {
                    let off = self.spill_off(slot) as i32;
                    self.ectx
                        .emit
                        .emit_with(|buf| x64::encode_xor_r64_m(10, Mem { base: 4, disp: off }, buf))
                        .expect("xor");
                }
            }
            BinOpKind::CmpNe => {
                let rhs_alloc = self.current_alloc(1);
                if let Some(reg) = rhs_alloc.as_reg() {
                    let enc = reg.hw_enc() as u8;
                    self.ectx
                        .emit
                        .emit_with(|buf| x64::encode_cmp_r64_r64(10, enc, buf))
                        .expect("cmp");
                } else if let Some(slot) = rhs_alloc.as_stack() {
                    let off = self.spill_off(slot) as i32;
                    self.ectx
                        .emit
                        .emit_with(|buf| x64::encode_cmp_r64_m(10, Mem { base: 4, disp: off }, buf))
                        .expect("cmp");
                }
                self.ectx
                    .emit
                    .emit_with(|buf| {
                        x64::encode_setne_r8(10, buf)?;
                        x64::encode_movzx_r64_rm8(10, Operand::Reg(10), buf)
                    })
                    .expect("cmpne");
            }
            BinOpKind::Shr => {
                let rhs_alloc = self.current_alloc(1);
                if let Some(reg) = rhs_alloc.as_reg() {
                    let enc = reg.hw_enc() as u8;
                    self.ectx
                        .emit
                        .emit_with(|buf| x64::encode_mov_r64_r64(1, enc, buf))
                        .expect("mov");
                } else if let Some(slot) = rhs_alloc.as_stack() {
                    let off = self.spill_off(slot) as i32;
                    self.ectx
                        .emit
                        .emit_with(|buf| x64::encode_mov_r64_m(1, Mem { base: 4, disp: off }, buf))
                        .expect("mov");
                }
                self.ectx
                    .emit
                    .emit_with(|buf| x64::encode_shr_r64_cl(10, buf))
                    .expect("shr");
            }
            BinOpKind::Shl => {
                let rhs_alloc = self.current_alloc(1);
                if let Some(reg) = rhs_alloc.as_reg() {
                    let enc = reg.hw_enc() as u8;
                    self.ectx
                        .emit
                        .emit_with(|buf| x64::encode_mov_r64_r64(1, enc, buf))
                        .expect("mov");
                } else if let Some(slot) = rhs_alloc.as_stack() {
                    let off = self.spill_off(slot) as i32;
                    self.ectx
                        .emit
                        .emit_with(|buf| x64::encode_mov_r64_m(1, Mem { base: 4, disp: off }, buf))
                        .expect("mov");
                }
                self.ectx
                    .emit
                    .emit_with(|buf| x64::encode_shl_r64_cl(10, buf))
                    .expect("shl");
            }
        }
        self.emit_store_def_r10(dst, 2);
        self.set_const(dst, None);
    }

    pub(super) fn emit_unary(
        &mut self,
        kind: UnaryOpKind,
        dst: crate::ir::VReg,
        src: crate::ir::VReg,
    ) {
        self.emit_load_use_r10(src, 0);
        match kind {
            UnaryOpKind::ZigzagDecode { wide: true } => {
                self.ectx
                    .emit
                    .emit_with(|buf| {
                        x64::encode_mov_r64_r64(11, 10, buf)?;
                        x64::encode_shr_r64_imm8(11, 1, buf)?;
                        x64::encode_mov_r64_imm32_sext(13, 1, buf)?;
                        x64::encode_and_r64_r64(10, 13, buf)?;
                        x64::encode_neg_r64(10, buf)?;
                        x64::encode_xor_r64_r64(10, 11, buf)
                    })
                    .expect("zigzag");
            }
            UnaryOpKind::ZigzagDecode { wide: false } => {
                self.ectx
                    .emit
                    .emit_with(|buf| {
                        x64::encode_mov_r32_r32(10, 10, buf)?;
                        x64::encode_mov_r32_r32(11, 10, buf)?;
                        x64::encode_shr_r64_imm8(11, 1, buf)?;
                        x64::encode_mov_r64_imm32_sext(13, 1, buf)?;
                        x64::encode_and_r64_r64(10, 13, buf)?;
                        x64::encode_neg_r64(10, buf)?;
                        x64::encode_xor_r64_r64(10, 11, buf)?;
                        x64::encode_mov_r32_r32(10, 10, buf)
                    })
                    .expect("zigzag");
            }
            UnaryOpKind::SignExtend { from_width } => match from_width {
                Width::W1 => self
                    .ectx
                    .emit
                    .emit_with(|buf| x64::encode_movsx_r64_rm8(10, Operand::Reg(10), buf))
                    .expect("movsx"),
                Width::W2 => self
                    .ectx
                    .emit
                    .emit_with(|buf| x64::encode_movsx_r64_rm16(10, Operand::Reg(10), buf))
                    .expect("movsx"),
                Width::W4 => self
                    .ectx
                    .emit
                    .emit_with(|buf| x64::encode_movsxd_r64_rm32(10, Operand::Reg(10), buf))
                    .expect("movsxd"),
                Width::W8 => {}
            },
        }
        self.emit_store_def_r10(dst, 1);
        self.set_const(dst, None);
    }

    pub(super) fn emit_branch_if(&mut self, cond: crate::ir::VReg, target: LabelId, invert: bool) {
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
                reg.class() == RegClass::Int,
                "unsupported register class {:?} for branch condition",
                reg.class()
            );
            let enc = reg.hw_enc() as u8;
            self.ectx
                .emit
                .emit_with(|buf| x64::encode_test_r64_r64(enc, enc, buf))
                .expect("test");
            if invert {
                self.ectx.emit.emit_jz_label(target).expect("jz");
            } else {
                self.ectx.emit.emit_jnz_label(target).expect("jnz");
            }
            return;
        }
        if let Some(slot) = alloc.as_stack() {
            let off = self.spill_off(slot) as i32;
            self.ectx
                .emit
                .emit_with(|buf| {
                    x64::encode_mov_r64_m(10, Mem { base: 4, disp: off }, buf)?;
                    x64::encode_test_r64_r64(10, 10, buf)
                })
                .expect("test");
            if invert {
                self.ectx.emit.emit_jz_label(target).expect("jz");
            } else {
                self.ectx.emit.emit_jnz_label(target).expect("jnz");
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
        from_block_id: u32,
    ) {
        let RaTerminator::JumpTable {
            targets, default, ..
        } = term
        else {
            unreachable!();
        };
        let _ = predicate;
        let alloc = self.current_alloc(0);
        if let Some(reg) = alloc.as_reg() {
            let enc = reg.hw_enc() as u8;
            self.ectx
                .emit
                .emit_with(|buf| x64::encode_mov_r64_r64(10, enc, buf))
                .expect("mov");
        } else if let Some(slot) = alloc.as_stack() {
            let off = self.spill_off(slot) as i32;
            self.ectx
                .emit
                .emit_with(|buf| x64::encode_mov_r64_m(10, Mem { base: 4, disp: off }, buf))
                .expect("mov");
        } else {
            panic!("unexpected none allocation for jumptable predicate");
        }
        self.ectx
            .emit
            .emit_with(|buf| x64::encode_mov_r32_r32(10, 10, buf))
            .expect("mov");
        for (index, target_block) in targets.iter().enumerate() {
            let resolved = self.resolve_forwarded_block(lambda_id, *target_block);
            let target_label =
                self.edge_target_label(from_block_id, index, self.block_label(lambda_id, resolved));
            self.ectx
                .emit
                .emit_with(|buf| x64::encode_cmp_r64_imm32(10, index as u32, buf))
                .expect("cmp");
            self.ectx.emit.emit_je_label(target_label).expect("je");
        }
        let default_succ_index = targets.len();
        let resolved_default = self.resolve_forwarded_block(lambda_id, *default);
        let default_target = self.edge_target_label(
            from_block_id,
            default_succ_index,
            self.block_label(lambda_id, resolved_default),
        );
        self.ectx.emit_branch(default_target);
    }
}
