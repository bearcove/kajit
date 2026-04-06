//! Instruction emission: binop, unary, read/write, branch, jump table.

use super::*;
use kajit_emit::aarch64::{self, Condition, Reg};

impl Lowerer<'_> {
    pub(super) fn emit_read_from_field(&mut self, dst: crate::ir::VReg, offset: u32, width: Width) {
        match width {
            Width::W1 => self
                .ectx
                .emit
                .emit_ldrb_imm(Reg::X9, Reg::X21, offset)
                .expect("ldrb"),
            Width::W2 => self
                .ectx
                .emit
                .emit_ldrh_imm(Reg::X9, Reg::X21, offset)
                .expect("ldrh"),
            Width::W4 => self
                .ectx
                .emit
                .emit_ldr_imm(aarch64::Width::W32, Reg::X9, Reg::X21, offset)
                .expect("ldr"),
            Width::W8 => self
                .ectx
                .emit
                .emit_ldr_imm(aarch64::Width::X64, Reg::X9, Reg::X21, offset)
                .expect("ldr"),
        }
        self.emit_store_def_x9(dst, 0);
        self.set_const(dst, None);
    }

    pub(super) fn emit_write_to_field(&mut self, src: crate::ir::VReg, offset: u32, width: Width) {
        // Use the allocated register directly to avoid unnecessary movs
        let alloc = self
            .alloc_for_vreg(src)
            .expect("emit_write_to_field: src should have allocation");

        let src_reg = if let Some(preg) = alloc.as_reg() {
            // Value is in a register - use it directly!
            assert_eq!(
                preg.class(),
                regalloc2::RegClass::Int,
                "store source must be in integer register"
            );
            Reg::from_raw(preg.hw_enc() as u8)
        } else {
            // Value is spilled - load into x9
            self.emit_load_x9_from_allocation(alloc);
            Reg::X9
        };

        match width {
            Width::W1 => self
                .ectx
                .emit
                .emit_strb_imm(src_reg, Reg::X21, offset)
                .expect("strb"),
            Width::W2 => self
                .ectx
                .emit
                .emit_strh_imm(src_reg, Reg::X21, offset)
                .expect("strh"),
            Width::W4 => self
                .ectx
                .emit
                .emit_str_imm(aarch64::Width::W32, src_reg, Reg::X21, offset)
                .expect("str"),
            Width::W8 => self
                .ectx
                .emit
                .emit_str_imm(aarch64::Width::X64, src_reg, Reg::X21, offset)
                .expect("str"),
        }
    }

    pub(super) fn emit_save_out_ptr(&mut self, dst: crate::ir::VReg) {
        self.ectx
            .emit
            .emit_mov_reg(aarch64::Width::X64, Reg::X9, Reg::X21)
            .expect("mov");
        self.emit_store_def_x9(dst, 0);
        self.set_const(dst, None);
    }

    pub(super) fn emit_set_out_ptr(&mut self, src: crate::ir::VReg) {
        self.emit_load_use_x9_vreg(src);
        self.ectx
            .emit
            .emit_mov_reg(aarch64::Width::X64, Reg::X21, Reg::X9)
            .expect("mov");
    }

    pub(super) fn emit_slot_addr(&mut self, dst: crate::ir::VReg, slot: crate::ir::SlotId) {
        let slot_off = self.slot_off(slot);
        self.emit_stack_addr(Reg::X9, slot_off);
        self.emit_store_def_x9(dst, 0);
        self.set_const(dst, None);
    }

    pub(super) fn emit_store_to_addr(
        &mut self,
        addr: crate::ir::VReg,
        src: crate::ir::VReg,
        width: Width,
    ) {
        self.emit_load_use_x10(addr, 0);
        self.emit_load_use_x9(src, 1);
        match width {
            Width::W1 => self
                .ectx
                .emit
                .emit_strb_imm(Reg::X9, Reg::X10, 0)
                .expect("strb"),
            Width::W2 => self
                .ectx
                .emit
                .emit_strh_imm(Reg::X9, Reg::X10, 0)
                .expect("strh"),
            Width::W4 => self
                .ectx
                .emit
                .emit_str_imm(aarch64::Width::W32, Reg::X9, Reg::X10, 0)
                .expect("str"),
            Width::W8 => self
                .ectx
                .emit
                .emit_str_imm(aarch64::Width::X64, Reg::X9, Reg::X10, 0)
                .expect("str"),
        }
    }

    pub(super) fn emit_load_from_addr(
        &mut self,
        dst: crate::ir::VReg,
        addr: crate::ir::VReg,
        width: Width,
    ) {
        // Try to use allocated registers directly to avoid scratch copies
        let addr_alloc = self.alloc_for_vreg(addr);
        let dst_alloc = self.alloc_for_vreg(dst);
        let addr_reg = addr_alloc.and_then(|a| a.as_reg());
        let dst_reg = dst_alloc.and_then(|a| a.as_reg());
        if let (Some(ar), Some(dr)) = (addr_reg, dst_reg)
            && ar.class() == regalloc2::RegClass::Int
            && dr.class() == regalloc2::RegClass::Int
        {
            let base = Reg::from_raw(ar.hw_enc() as u8);
            let dest = Reg::from_raw(dr.hw_enc() as u8);
            match width {
                Width::W1 => self.ectx.emit.emit_ldrb_imm(dest, base, 0).expect("ldrb"),
                Width::W2 => self.ectx.emit.emit_ldrh_imm(dest, base, 0).expect("ldrh"),
                Width::W4 => self
                    .ectx
                    .emit
                    .emit_ldr_imm(aarch64::Width::W32, dest, base, 0)
                    .expect("ldr"),
                Width::W8 => self
                    .ectx
                    .emit
                    .emit_ldr_imm(aarch64::Width::X64, dest, base, 0)
                    .expect("ldr"),
            }
            self.set_const(dst, None);
            return;
        }

        // Fallback: use scratch registers
        self.emit_load_use_x10(addr, 0);
        match width {
            Width::W1 => self
                .ectx
                .emit
                .emit_ldrb_imm(Reg::X9, Reg::X10, 0)
                .expect("ldrb"),
            Width::W2 => self
                .ectx
                .emit
                .emit_ldrh_imm(Reg::X9, Reg::X10, 0)
                .expect("ldrh"),
            Width::W4 => self
                .ectx
                .emit
                .emit_ldr_imm(aarch64::Width::W32, Reg::X9, Reg::X10, 0)
                .expect("ldr"),
            Width::W8 => self
                .ectx
                .emit
                .emit_ldr_imm(aarch64::Width::X64, Reg::X9, Reg::X10, 0)
                .expect("ldr"),
        }
        self.emit_store_def_x9(dst, 1);
        self.set_const(dst, None);
    }

    pub(super) fn emit_read_bytes(&mut self, dst: crate::ir::VReg, count: u32) {
        self.emit_bounds_check(count);
        match count {
            1 => self
                .ectx
                .emit
                .emit_ldrb_imm(Reg::X9, Reg::X19, 0)
                .expect("ldrb"),
            2 => self
                .ectx
                .emit
                .emit_ldrh_imm(Reg::X9, Reg::X19, 0)
                .expect("ldrh"),
            4 => self
                .ectx
                .emit
                .emit_ldr_imm(aarch64::Width::W32, Reg::X9, Reg::X19, 0)
                .expect("ldr"),
            8 => self
                .ectx
                .emit
                .emit_ldr_imm(aarch64::Width::X64, Reg::X9, Reg::X19, 0)
                .expect("ldr"),
            _ => panic!("unsupported ReadBytes count: {count}"),
        }
        self.emit_store_def_x9(dst, 0);
        self.set_const(dst, None);
        self.ectx.emit_advance_cursor_by(count);
    }

    pub(super) fn emit_peek_byte(&mut self, dst: crate::ir::VReg) {
        self.emit_bounds_check(1);
        self.ectx
            .emit
            .emit_ldrb_imm(Reg::X9, Reg::X19, 0)
            .expect("ldrb");
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
        if matches!(
            kind,
            BinOpKind::CmpEq
                | BinOpKind::CmpNe
                | BinOpKind::CmpLt
                | BinOpKind::CmpLe
                | BinOpKind::CmpGt
                | BinOpKind::CmpGe
        ) {
            let lhs_alloc = self
                .alloc_for_vreg(lhs)
                .expect("compare lhs should have alloc");
            let rhs_alloc = self.alloc_for_vreg(rhs); // May be None for immediate-only const
            let rhs_const = self.const_of(rhs);

            if let Some(reg) = lhs_alloc.as_reg() {
                assert!(
                    reg.class() == regalloc2::RegClass::Int,
                    "unsupported register allocation class {:?} for compare lhs",
                    reg.class()
                );
            }
            if let Some(alloc) = rhs_alloc
                && let Some(reg) = alloc.as_reg()
            {
                assert!(
                    reg.class() == regalloc2::RegClass::Int,
                    "unsupported register allocation class {:?} for compare rhs",
                    reg.class()
                );
            }

            // For rhs_alloc, extract reg/stack if present
            let rhs_reg = rhs_alloc.and_then(|a| a.as_reg());
            let rhs_stack = rhs_alloc.and_then(|a| a.as_stack());

            match (
                lhs_alloc.as_reg(),
                lhs_alloc.as_stack(),
                rhs_const,
                rhs_reg,
                rhs_stack,
            ) {
                (Some(lhs_reg), None, Some(c), _, _) => {
                    let lhs_r = lhs_reg.hw_enc() as u8;
                    self.emit_load_u64_x10(c);
                    self.ectx
                        .emit
                        .emit_cmp_reg(aarch64::Width::X64, Reg::from_raw(lhs_r), Reg::X10)
                        .expect("cmp");
                }
                (None, Some(lhs_stack), Some(c), _, _) => {
                    let lhs_off = self.spill_off(lhs_stack);
                    self.emit_stack_load(aarch64::Width::X64, Reg::X9, lhs_off);
                    if c <= 4095 {
                        self.ectx
                            .emit
                            .emit_cmp_imm(aarch64::Width::X64, Reg::X9, c as u16)
                            .expect("cmp");
                    } else {
                        self.emit_load_u64_x10(c);
                        self.ectx
                            .emit
                            .emit_cmp_reg(aarch64::Width::X64, Reg::X9, Reg::X10)
                            .expect("cmp");
                    }
                }
                (Some(lhs_reg), None, None, Some(rhs_reg), None) => {
                    let lhs_r = lhs_reg.hw_enc() as u8;
                    let rhs_r = rhs_reg.hw_enc() as u8;
                    self.ectx
                        .emit
                        .emit_cmp_reg(
                            aarch64::Width::X64,
                            Reg::from_raw(lhs_r),
                            Reg::from_raw(rhs_r),
                        )
                        .expect("cmp");
                }
                (Some(lhs_reg), None, None, None, Some(rhs_stack)) => {
                    let lhs_r = lhs_reg.hw_enc() as u8;
                    let rhs_off = self.spill_off(rhs_stack);
                    self.emit_stack_load(aarch64::Width::X64, Reg::X10, rhs_off);
                    self.ectx
                        .emit
                        .emit_cmp_reg(aarch64::Width::X64, Reg::from_raw(lhs_r), Reg::X10)
                        .expect("cmp");
                }
                (None, Some(lhs_stack), None, Some(rhs_reg), None) => {
                    let lhs_off = self.spill_off(lhs_stack);
                    let rhs_r = rhs_reg.hw_enc() as u8;
                    self.emit_stack_load(aarch64::Width::X64, Reg::X9, lhs_off);
                    self.ectx
                        .emit
                        .emit_cmp_reg(aarch64::Width::X64, Reg::X9, Reg::from_raw(rhs_r))
                        .expect("cmp");
                }
                (None, Some(lhs_stack), None, None, Some(rhs_stack)) => {
                    let lhs_off = self.spill_off(lhs_stack);
                    let rhs_off = self.spill_off(rhs_stack);
                    self.emit_stack_load(aarch64::Width::X64, Reg::X9, lhs_off);
                    self.emit_stack_load(aarch64::Width::X64, Reg::X10, rhs_off);
                    self.ectx
                        .emit
                        .emit_cmp_reg(aarch64::Width::X64, Reg::X9, Reg::X10)
                        .expect("cmp");
                }
                _ => panic!("unexpected none allocation for compare operands"),
            }

            let dst_alloc = self
                .alloc_for_vreg(dst)
                .expect("compare dst should have alloc");
            let condition = match kind {
                BinOpKind::CmpEq => Condition::Eq,
                BinOpKind::CmpNe => Condition::Ne,
                BinOpKind::CmpLt => Condition::Lo,
                BinOpKind::CmpLe => Condition::Ls,
                BinOpKind::CmpGt => Condition::Hi,
                BinOpKind::CmpGe => Condition::Hs,
                _ => unreachable!(),
            };
            if let Some(dst_reg) = dst_alloc.as_reg() {
                assert!(
                    dst_reg.class() == regalloc2::RegClass::Int,
                    "unsupported register allocation class {:?} for compare dst",
                    dst_reg.class()
                );
                let dst_r = dst_reg.hw_enc() as u8;
                self.ectx
                    .emit
                    .emit_cset(aarch64::Width::X64, Reg::from_raw(dst_r), condition)
                    .expect("cset");
            } else if let Some(dst_stack) = dst_alloc.as_stack() {
                let dst_off = self.spill_off(dst_stack);
                self.ectx
                    .emit
                    .emit_cset(aarch64::Width::X64, Reg::X9, condition)
                    .expect("cset");
                self.emit_stack_store(aarch64::Width::X64, Reg::X9, dst_off);
            } else {
                panic!("unexpected none allocation for compare dst");
            }
            self.set_const(dst, None);
            return;
        }

        let lhs_alloc = self
            .alloc_for_vreg(lhs)
            .expect("binop lhs should have alloc");
        let rhs_alloc = self.alloc_for_vreg(rhs); // May be None for immediate-only const
        let dst_alloc = self
            .alloc_for_vreg(dst)
            .expect("binop dst should have alloc");

        // Check if rhs is an immediate-only const (no allocation)
        let rhs_const = self.const_of(rhs);
        let rhs_is_immediate_only = rhs_alloc.is_none() && rhs_const.is_some();

        let rhs_reg = rhs_alloc.and_then(|a| a.as_reg());
        if let (Some(lhs_reg), Some(dst_reg)) = (lhs_alloc.as_reg(), dst_alloc.as_reg())
            && (rhs_is_immediate_only || rhs_reg.is_some())
        {
            assert!(
                lhs_reg.class() == regalloc2::RegClass::Int,
                "unsupported register allocation class {:?} for binop lhs",
                lhs_reg.class()
            );
            if let Some(rhs_reg) = rhs_reg {
                assert!(
                    rhs_reg.class() == regalloc2::RegClass::Int,
                    "unsupported register allocation class {:?} for binop rhs",
                    rhs_reg.class()
                );
            }
            assert!(
                dst_reg.class() == regalloc2::RegClass::Int,
                "unsupported register allocation class {:?} for binop dst",
                dst_reg.class()
            );

            let lhs_r = lhs_reg.hw_enc() as u8;
            // rhs_r is only used when immediate encoding isn't possible.
            // If rhs_is_immediate_only is true, we should always take the immediate path,
            // so unwrap_or(0) is safe (the value won't be used).
            let rhs_r = rhs_reg.map(|r| r.hw_enc() as u8).unwrap_or(0);
            let dst_r = dst_reg.hw_enc() as u8;

            let handled = match kind {
                BinOpKind::Add => {
                    // Try immediate encoding first (12-bit, 0-4095)
                    if let Some(c) = rhs_const
                        && c <= 4095
                    {
                        self.ectx
                            .emit
                            .emit_add_imm(
                                aarch64::Width::X64,
                                Reg::from_raw(dst_r),
                                Reg::from_raw(lhs_r),
                                c as u16,
                                false,
                            )
                            .expect("add imm");
                    } else if dst_r == lhs_r {
                        self.ectx
                            .emit
                            .emit_add_reg(
                                aarch64::Width::X64,
                                Reg::from_raw(dst_r),
                                Reg::from_raw(dst_r),
                                Reg::from_raw(rhs_r),
                            )
                            .expect("add");
                    } else {
                        // add is a three-operand instruction, all distinct regs are fine
                        self.ectx
                            .emit
                            .emit_add_reg(
                                aarch64::Width::X64,
                                Reg::from_raw(dst_r),
                                Reg::from_raw(lhs_r),
                                Reg::from_raw(rhs_r),
                            )
                            .expect("add");
                    }
                    true
                }
                BinOpKind::Sub => {
                    // Try immediate encoding first (12-bit, 0-4095)
                    if let Some(c) = rhs_const
                        && c <= 4095
                    {
                        self.ectx
                            .emit
                            .emit_sub_imm(
                                aarch64::Width::X64,
                                Reg::from_raw(dst_r),
                                Reg::from_raw(lhs_r),
                                c as u16,
                                false,
                            )
                            .expect("sub imm");
                        true
                    } else {
                        // sub is a three-operand instruction, all distinct regs are fine
                        self.ectx
                            .emit
                            .emit_sub_reg(
                                aarch64::Width::X64,
                                Reg::from_raw(dst_r),
                                Reg::from_raw(lhs_r),
                                Reg::from_raw(rhs_r),
                            )
                            .expect("sub");
                        true
                    }
                }
                BinOpKind::Mul => {
                    if dst_r == lhs_r {
                        self.ectx
                            .emit
                            .emit_mul_reg(
                                aarch64::Width::X64,
                                Reg::from_raw(dst_r),
                                Reg::from_raw(dst_r),
                                Reg::from_raw(rhs_r),
                            )
                            .expect("mul");
                    } else {
                        // mul is a three-operand instruction, all distinct regs are fine
                        self.ectx
                            .emit
                            .emit_mul_reg(
                                aarch64::Width::X64,
                                Reg::from_raw(dst_r),
                                Reg::from_raw(lhs_r),
                                Reg::from_raw(rhs_r),
                            )
                            .expect("mul");
                    }
                    true
                }
                BinOpKind::And => {
                    // Track power-of-2 masks for potential tbz/tbnz optimization
                    if let Some(c) = rhs_const
                        && c.is_power_of_two()
                    {
                        self.set_masked_value(
                            dst,
                            Some(super::MaskedValueInfo {
                                src: lhs,
                                bit: c.trailing_zeros() as u8,
                            }),
                        );
                    } else {
                        self.set_masked_value(dst, None);
                    }

                    // Try logical immediate encoding first
                    if let Some(c) = rhs_const
                        && self
                            .ectx
                            .emit
                            .emit_and_imm(
                                aarch64::Width::X64,
                                Reg::from_raw(dst_r),
                                Reg::from_raw(lhs_r),
                                c,
                            )
                            .is_ok()
                    {
                        // emitted
                    } else if dst_r == lhs_r {
                        self.ectx
                            .emit
                            .emit_and_reg(
                                aarch64::Width::X64,
                                Reg::from_raw(dst_r),
                                Reg::from_raw(dst_r),
                                Reg::from_raw(rhs_r),
                            )
                            .expect("and");
                    } else if dst_r == rhs_r {
                        self.ectx
                            .emit
                            .emit_and_reg(
                                aarch64::Width::X64,
                                Reg::from_raw(dst_r),
                                Reg::from_raw(dst_r),
                                Reg::from_raw(lhs_r),
                            )
                            .expect("and");
                    } else {
                        self.ectx
                            .emit
                            .emit_mov_reg(
                                aarch64::Width::X64,
                                Reg::from_raw(dst_r),
                                Reg::from_raw(lhs_r),
                            )
                            .expect("mov");
                        self.ectx
                            .emit
                            .emit_and_reg(
                                aarch64::Width::X64,
                                Reg::from_raw(dst_r),
                                Reg::from_raw(dst_r),
                                Reg::from_raw(rhs_r),
                            )
                            .expect("and");
                    }
                    true
                }
                BinOpKind::Or => {
                    // Try logical immediate encoding first
                    if let Some(c) = rhs_const
                        && self
                            .ectx
                            .emit
                            .emit_orr_imm(
                                aarch64::Width::X64,
                                Reg::from_raw(dst_r),
                                Reg::from_raw(lhs_r),
                                c,
                            )
                            .is_ok()
                    {
                        // emitted
                    } else if dst_r == lhs_r {
                        self.ectx
                            .emit
                            .emit_orr_reg(
                                aarch64::Width::X64,
                                Reg::from_raw(dst_r),
                                Reg::from_raw(dst_r),
                                Reg::from_raw(rhs_r),
                            )
                            .expect("orr");
                    } else if dst_r == rhs_r {
                        self.ectx
                            .emit
                            .emit_orr_reg(
                                aarch64::Width::X64,
                                Reg::from_raw(dst_r),
                                Reg::from_raw(dst_r),
                                Reg::from_raw(lhs_r),
                            )
                            .expect("orr");
                    } else {
                        self.ectx
                            .emit
                            .emit_mov_reg(
                                aarch64::Width::X64,
                                Reg::from_raw(dst_r),
                                Reg::from_raw(lhs_r),
                            )
                            .expect("mov");
                        self.ectx
                            .emit
                            .emit_orr_reg(
                                aarch64::Width::X64,
                                Reg::from_raw(dst_r),
                                Reg::from_raw(dst_r),
                                Reg::from_raw(rhs_r),
                            )
                            .expect("orr");
                    }
                    true
                }
                BinOpKind::Xor => {
                    // Try logical immediate encoding first
                    if let Some(c) = rhs_const
                        && self
                            .ectx
                            .emit
                            .emit_eor_imm(
                                aarch64::Width::X64,
                                Reg::from_raw(dst_r),
                                Reg::from_raw(lhs_r),
                                c,
                            )
                            .is_ok()
                    {
                        // emitted
                    } else if dst_r == lhs_r {
                        self.ectx
                            .emit
                            .emit_eor_reg(
                                aarch64::Width::X64,
                                Reg::from_raw(dst_r),
                                Reg::from_raw(dst_r),
                                Reg::from_raw(rhs_r),
                            )
                            .expect("eor");
                    } else if dst_r == rhs_r {
                        self.ectx
                            .emit
                            .emit_eor_reg(
                                aarch64::Width::X64,
                                Reg::from_raw(dst_r),
                                Reg::from_raw(dst_r),
                                Reg::from_raw(lhs_r),
                            )
                            .expect("eor");
                    } else {
                        self.ectx
                            .emit
                            .emit_mov_reg(
                                aarch64::Width::X64,
                                Reg::from_raw(dst_r),
                                Reg::from_raw(lhs_r),
                            )
                            .expect("mov");
                        self.ectx
                            .emit
                            .emit_eor_reg(
                                aarch64::Width::X64,
                                Reg::from_raw(dst_r),
                                Reg::from_raw(dst_r),
                                Reg::from_raw(rhs_r),
                            )
                            .expect("eor");
                    }
                    true
                }
                BinOpKind::Shr => {
                    // Try immediate encoding first (shift amount 0-63)
                    if let Some(c) = rhs_const
                        && c <= 63
                    {
                        self.ectx
                            .emit
                            .emit_lsr_imm(
                                aarch64::Width::X64,
                                Reg::from_raw(dst_r),
                                Reg::from_raw(lhs_r),
                                c as u8,
                            )
                            .expect("lsr imm");
                        true
                    } else {
                        // lsr is a three-operand instruction, all distinct regs are fine
                        self.ectx
                            .emit
                            .emit_lsr_reg(
                                aarch64::Width::X64,
                                Reg::from_raw(dst_r),
                                Reg::from_raw(lhs_r),
                                Reg::from_raw(rhs_r),
                            )
                            .expect("lsr");
                        true
                    }
                }
                BinOpKind::Sar => {
                    // Try immediate encoding first (shift amount 0-63)
                    if let Some(c) = rhs_const
                        && c <= 63
                    {
                        self.ectx
                            .emit
                            .emit_asr_imm(
                                aarch64::Width::X64,
                                Reg::from_raw(dst_r),
                                Reg::from_raw(lhs_r),
                                c as u8,
                            )
                            .expect("asr imm");
                        true
                    } else {
                        // asr is a three-operand instruction, all distinct regs are fine
                        self.ectx
                            .emit
                            .emit_asr_reg(
                                aarch64::Width::X64,
                                Reg::from_raw(dst_r),
                                Reg::from_raw(lhs_r),
                                Reg::from_raw(rhs_r),
                            )
                            .expect("asr");
                        true
                    }
                }
                BinOpKind::Shl => {
                    // Try immediate encoding first (shift amount 1-63, not 0)
                    if let Some(c) = rhs_const
                        && (1..=63).contains(&c)
                    {
                        self.ectx
                            .emit
                            .emit_lsl_imm(
                                aarch64::Width::X64,
                                Reg::from_raw(dst_r),
                                Reg::from_raw(lhs_r),
                                c as u8,
                            )
                            .expect("lsl imm");
                        true
                    } else if dst_r == lhs_r {
                        self.ectx
                            .emit
                            .emit_lsl_reg(
                                aarch64::Width::X64,
                                Reg::from_raw(dst_r),
                                Reg::from_raw(dst_r),
                                Reg::from_raw(rhs_r),
                            )
                            .expect("lsl");
                        true
                    } else {
                        // dst != lhs, but lsl reads both operands before writing dst,
                        // so dst == rhs is safe
                        self.ectx
                            .emit
                            .emit_lsl_reg(
                                aarch64::Width::X64,
                                Reg::from_raw(dst_r),
                                Reg::from_raw(lhs_r),
                                Reg::from_raw(rhs_r),
                            )
                            .expect("lsl");
                        true
                    }
                }
                BinOpKind::CmpEq
                | BinOpKind::CmpNe
                | BinOpKind::CmpLt
                | BinOpKind::CmpLe
                | BinOpKind::CmpGt
                | BinOpKind::CmpGe => unreachable!("compare handled above"),
            };
            if handled {
                self.set_const(dst, None);
                return;
            }
        }

        self.emit_load_use_x9_vreg(lhs);
        let rhs_const = self.const_of(rhs);
        match kind {
            BinOpKind::Add => {
                if let Some(c) = rhs_const
                    && c <= 4095
                {
                    self.ectx
                        .emit
                        .emit_add_imm(aarch64::Width::X64, Reg::X9, Reg::X9, c as u16, false)
                        .expect("add");
                } else {
                    self.emit_load_use_x10_vreg(rhs);
                    self.ectx
                        .emit
                        .emit_add_reg(aarch64::Width::X64, Reg::X9, Reg::X9, Reg::X10)
                        .expect("add");
                }
            }
            BinOpKind::Sub => {
                if let Some(c) = rhs_const
                    && c <= 4095
                {
                    self.ectx
                        .emit
                        .emit_sub_imm(aarch64::Width::X64, Reg::X9, Reg::X9, c as u16, false)
                        .expect("sub");
                } else {
                    self.emit_load_use_x10_vreg(rhs);
                    self.ectx
                        .emit
                        .emit_sub_reg(aarch64::Width::X64, Reg::X9, Reg::X9, Reg::X10)
                        .expect("sub");
                }
            }
            BinOpKind::Mul => {
                self.emit_load_use_x10_vreg(rhs);
                self.ectx
                    .emit
                    .emit_mul_reg(aarch64::Width::X64, Reg::X9, Reg::X9, Reg::X10)
                    .expect("mul");
            }
            BinOpKind::And => {
                // Track power-of-2 masks for potential tbz/tbnz optimization
                if let Some(c) = rhs_const
                    && c.is_power_of_two()
                {
                    self.set_masked_value(
                        dst,
                        Some(super::MaskedValueInfo {
                            src: lhs,
                            bit: c.trailing_zeros() as u8,
                        }),
                    );
                } else {
                    self.set_masked_value(dst, None);
                }

                if let Some(c) = rhs_const
                    && self
                        .ectx
                        .emit
                        .emit_and_imm(aarch64::Width::X64, Reg::X9, Reg::X9, c)
                        .is_ok()
                {
                    // emitted
                } else {
                    self.emit_load_use_x10_vreg(rhs);
                    self.ectx
                        .emit
                        .emit_and_reg(aarch64::Width::X64, Reg::X9, Reg::X9, Reg::X10)
                        .expect("and");
                }
            }
            BinOpKind::Or => {
                self.emit_load_use_x10_vreg(rhs);
                self.ectx
                    .emit
                    .emit_orr_reg(aarch64::Width::X64, Reg::X9, Reg::X9, Reg::X10)
                    .expect("orr");
            }
            BinOpKind::Xor => {
                self.emit_load_use_x10_vreg(rhs);
                self.ectx
                    .emit
                    .emit_eor_reg(aarch64::Width::X64, Reg::X9, Reg::X9, Reg::X10)
                    .expect("eor");
            }
            BinOpKind::CmpEq
            | BinOpKind::CmpNe
            | BinOpKind::CmpLt
            | BinOpKind::CmpLe
            | BinOpKind::CmpGt
            | BinOpKind::CmpGe => unreachable!("compare handled above"),
            BinOpKind::Shr => {
                if let Some(c) = rhs_const
                    && c <= 63
                {
                    self.ectx
                        .emit
                        .emit_lsr_imm(aarch64::Width::X64, Reg::X9, Reg::X9, c as u8)
                        .expect("lsr");
                } else {
                    self.emit_load_use_x10_vreg(rhs);
                    self.ectx
                        .emit
                        .emit_lsr_reg(aarch64::Width::X64, Reg::X9, Reg::X9, Reg::X10)
                        .expect("lsr");
                }
            }
            BinOpKind::Sar => {
                if let Some(c) = rhs_const
                    && c <= 63
                {
                    self.ectx
                        .emit
                        .emit_asr_imm(aarch64::Width::X64, Reg::X9, Reg::X9, c as u8)
                        .expect("asr");
                } else {
                    self.emit_load_use_x10_vreg(rhs);
                    self.ectx
                        .emit
                        .emit_asr_reg(aarch64::Width::X64, Reg::X9, Reg::X9, Reg::X10)
                        .expect("asr");
                }
            }
            BinOpKind::Shl => {
                if let Some(c) = rhs_const
                    && c <= 63
                {
                    self.ectx
                        .emit
                        .emit_lsl_imm(aarch64::Width::X64, Reg::X9, Reg::X9, c as u8)
                        .expect("lsl");
                } else {
                    self.emit_load_use_x10_vreg(rhs);
                    self.ectx
                        .emit
                        .emit_lsl_reg(aarch64::Width::X64, Reg::X9, Reg::X9, Reg::X10)
                        .expect("lsl");
                }
            }
        }
        self.emit_store_def_x9_vreg(dst);
        self.set_const(dst, None);
    }

    pub(super) fn emit_unary(
        &mut self,
        kind: UnaryOpKind,
        dst: crate::ir::VReg,
        src: crate::ir::VReg,
    ) {
        self.emit_load_use_x9_vreg(src);
        match kind {
            UnaryOpKind::ZigzagDecode { wide: true } => {
                self.ectx
                    .emit
                    .emit_lsr_imm(aarch64::Width::X64, Reg::X10, Reg::X9, 1)
                    .expect("lsr");
                self.ectx
                    .emit
                    .emit_and_imm(aarch64::Width::X64, Reg::X16, Reg::X9, 1)
                    .expect("and");
                self.ectx
                    .emit
                    .emit_neg_reg(aarch64::Width::X64, Reg::X16, Reg::X16)
                    .expect("neg");
                self.ectx
                    .emit
                    .emit_eor_reg(aarch64::Width::X64, Reg::X9, Reg::X10, Reg::X16)
                    .expect("eor");
            }
            UnaryOpKind::ZigzagDecode { wide: false } => {
                self.ectx
                    .emit
                    .emit_lsr_imm(aarch64::Width::W32, Reg::X10, Reg::X9, 1)
                    .expect("lsr");
                self.ectx
                    .emit
                    .emit_and_imm(aarch64::Width::W32, Reg::X16, Reg::X9, 1)
                    .expect("and");
                self.ectx
                    .emit
                    .emit_neg_reg(aarch64::Width::W32, Reg::X16, Reg::X16)
                    .expect("neg");
                self.ectx
                    .emit
                    .emit_eor_reg(aarch64::Width::W32, Reg::X9, Reg::X10, Reg::X16)
                    .expect("eor");
            }
            UnaryOpKind::SignExtend { from_width } => match from_width {
                Width::W1 => self
                    .ectx
                    .emit
                    .emit_sxtb(aarch64::Width::X64, Reg::X9, Reg::X9)
                    .expect("sxtb"),
                Width::W2 => self
                    .ectx
                    .emit
                    .emit_sxth(aarch64::Width::X64, Reg::X9, Reg::X9)
                    .expect("sxth"),
                Width::W4 => self.ectx.emit.emit_sxtw(Reg::X9, Reg::X9).expect("sxtw"),
                Width::W8 => {}
            },
        }
        self.emit_store_def_x9_vreg(dst);
        self.set_const(dst, None);
    }

    pub(super) fn emit_branch_if(&mut self, cond: crate::ir::VReg, target: LabelId, invert: bool) {
        // Check if we can use tbz/tbnz optimization for masked values
        if let Some(masked) = self.masked_value_of(cond) {
            // In no-edit mode, load the source vreg and use tbz/tbnz directly
            // This eliminates the And instruction that created the masked value
            if self.no_edit_mode() {
                let src_alloc = self.canonical_alloc_for_vreg(masked.src);
                if let Some(slot) = src_alloc.as_stack() {
                    let off = self.spill_off(slot);
                    self.emit_stack_load(aarch64::Width::X64, Reg::X9, off);
                    // invert=true means branch_if_zero, so we want tbz (branch if bit is zero)
                    // invert=false means branch_if, so we want tbnz (branch if bit is non-zero)
                    if invert {
                        self.ectx
                            .emit
                            .emit_tbz_label(Reg::X9, masked.bit, target)
                            .expect("tbz");
                    } else {
                        self.ectx
                            .emit
                            .emit_tbnz_label(Reg::X9, masked.bit, target)
                            .expect("tbnz");
                    }
                    return;
                }
            }
            // In edit mode with regalloc, check if source vreg has a current allocation
            // This would happen if src is also an operand of the branch (unusual but possible)
            if let Some(src_alloc) = self.alloc_for_vreg(masked.src)
                && let Some(reg) = src_alloc.as_reg()
            {
                let r = reg.hw_enc() as u8;
                if invert {
                    self.ectx
                        .emit
                        .emit_tbz_label(Reg::from_raw(r), masked.bit, target)
                        .expect("tbz");
                } else {
                    self.ectx
                        .emit
                        .emit_tbnz_label(Reg::from_raw(r), masked.bit, target)
                        .expect("tbnz");
                }
                return;
            }
        }

        let alloc = self
            .alloc_for_vreg(cond)
            .expect("branch_if cond should have allocation");
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
                self.ectx
                    .emit
                    .emit_cbz_label(aarch64::Width::X64, Reg::from_raw(r), target)
                    .expect("cbz");
            } else {
                self.ectx
                    .emit
                    .emit_cbnz_label(aarch64::Width::X64, Reg::from_raw(r), target)
                    .expect("cbnz");
            }
            return;
        }
        if let Some(slot) = alloc.as_stack() {
            let off = self.spill_off(slot);
            self.emit_stack_load(aarch64::Width::X64, Reg::X9, off);
            if invert {
                self.ectx
                    .emit
                    .emit_cbz_label(aarch64::Width::X64, Reg::X9, target)
                    .expect("cbz");
            } else {
                self.ectx
                    .emit
                    .emit_cbnz_label(aarch64::Width::X64, Reg::X9, target)
                    .expect("cbnz");
            }
            return;
        }
        panic!("unexpected none allocation for branch condition");
    }

    pub(super) fn emit_jump_table(
        &mut self,
        lambda_id: u32,
        predicate: crate::ir::VReg,
        targets: &[cfg_mir::EdgeId],
        default: cfg_mir::EdgeId,
        func: &cfg_mir::Function,
    ) {
        let alloc = self
            .alloc_for_vreg(predicate)
            .expect("jump_table predicate should have allocation");
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
            self.emit_stack_load(aarch64::Width::X64, Reg::X9, off);
        } else if pred_reg.is_none() {
            panic!("unexpected none allocation for jumptable predicate");
        }
        for (index, edge_id) in targets.iter().enumerate() {
            let target_block = func
                .edge(*edge_id)
                .expect("jump-table target edge should exist")
                .to;
            let resolved = self.resolve_forwarded_block(lambda_id, target_block);
            let target = self.edge_target_label(*edge_id, self.block_label(lambda_id, resolved));
            let idx = index as u32;
            if let Some(r) = pred_reg {
                self.emit_load_u32_w10(idx);
                self.ectx
                    .emit
                    .emit_cmp_reg(aarch64::Width::X64, Reg::from_raw(r), Reg::X10)
                    .expect("cmp");
                self.ectx
                    .emit
                    .emit_b_cond_label(Condition::Eq, target)
                    .expect("b.eq");
            } else if idx <= 4095 {
                self.ectx
                    .emit
                    .emit_cmp_imm(aarch64::Width::W32, Reg::X9, idx as u16)
                    .expect("cmp");
                self.ectx
                    .emit
                    .emit_b_cond_label(Condition::Eq, target)
                    .expect("b.eq");
            } else {
                self.emit_load_u32_w10(idx);
                self.ectx
                    .emit
                    .emit_cmp_reg(aarch64::Width::W32, Reg::X9, Reg::X10)
                    .expect("cmp");
                self.ectx
                    .emit
                    .emit_b_cond_label(Condition::Eq, target)
                    .expect("b.eq");
            }
        }
        let default_block = func
            .edge(default)
            .expect("jump-table default edge should exist")
            .to;
        let resolved_default = self.resolve_forwarded_block(lambda_id, default_block);
        let default_target =
            self.edge_target_label(default, self.block_label(lambda_id, resolved_default));
        self.ectx.emit_branch(default_target);
    }
}
