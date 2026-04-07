//! Instruction emission for aarch64 regalloc3 backend.

use kajit_emit::aarch64::{Condition, Reg, Width};
use kajit_mir::cfg_mir::Inst;

use crate::ir_backend::{DataRelocInfo, ExternAddrRelocInfo};
use kajit_lir::{BinOpKind, LinearOp, UnaryOpKind};

use super::context::EmitContext;

impl<'a> EmitContext<'a> {
    /// Emit a single instruction.
    pub(super) fn emit_inst(&mut self, inst: &Inst) {
        self.current_inst = Some(inst.id);
        // Skip instructions whose outputs were fused (bfi, bit-test, etc.)
        match &inst.op {
            LinearOp::BinOp { dst, .. }
            | LinearOp::Const { dst, .. }
            | LinearOp::DataAddr { dst, .. } => {
                if self.fused_skip.contains(dst) {
                    self.current_inst = None;
                    return;
                }
            }
            _ => {}
        }
        match &inst.op {
            LinearOp::Copy { dst, src } => {
                // Elide copy when src and dst are in the same register
                if let (Some(sp), Some(dp)) = (self.preg_for_vreg(*src), self.preg_for_vreg(*dst))
                    && sp == dp
                {
                    self.current_inst = None;
                    return; // nop
                }
                let src_reg = self.reg_for_vreg_with_temp(*src, Reg::X16);
                self.store_to_vreg(*dst, src_reg);
            }

            LinearOp::Const { dst, value } => {
                // Skip immediate-only consts (operands cleared by elim_imm).
                if inst.operands.is_empty() {
                    self.current_inst = None;
                    return;
                }
                if let Some(preg) = self.preg_for_vreg(*dst) {
                    self.emit_load_u64(self.preg_to_reg(preg), *value);
                } else if self.alloc_func.rematerializable.contains_key(dst) {
                    // Rematerializable constant: skip store to spill slot.
                    // All reads of this vreg will re-emit movz instead.
                } else {
                    // Spilled - load into x9, store to spill slot
                    self.emit_load_u64(Reg::X16, *value);
                    self.store_to_vreg(*dst, Reg::X16);
                }
            }

            LinearOp::DataAddr { dst, blob_id } => {
                // Emit a fixed 4-instruction movz/movk sequence with placeholder 0.
                // The actual address will be patched after JIT finalization.
                let code_offset = self.ectx.emit.code_len();
                let dest_reg = if let Some(preg) = self.preg_for_vreg(*dst) {
                    self.preg_to_reg(preg)
                } else {
                    Reg::X16
                };
                self.emit_load_u64_fixed(dest_reg, 0);
                self.data_relocs.push(DataRelocInfo {
                    code_offset,
                    blob_id: *blob_id,
                });
                if self.preg_for_vreg(*dst).is_none() {
                    self.store_to_vreg(*dst, Reg::X16);
                }
            }

            LinearOp::ExternAddr { dst, symbol, value } => {
                // Emit a fixed 4-instruction sequence with placeholder 0.
                // Patched at JIT time with the actual value, or relocated in harness mode.
                let code_offset = self.ectx.emit.code_len();
                let dest_reg = if let Some(preg) = self.preg_for_vreg(*dst) {
                    self.preg_to_reg(preg)
                } else {
                    Reg::X16
                };
                self.emit_load_u64_fixed(dest_reg, 0);
                self.extern_addr_relocs.push(ExternAddrRelocInfo {
                    code_offset,
                    value: *value,
                    symbol: symbol.clone(),
                });
                if self.preg_for_vreg(*dst).is_none() {
                    self.store_to_vreg(*dst, Reg::X16);
                }
            }

            LinearOp::BinOp { op, dst, lhs, rhs } => {
                self.emit_binop(*op, *dst, *lhs, *rhs);
            }

            LinearOp::UnaryOp { op, dst, src } => {
                self.emit_unary(*op, *dst, *src);
            }

            LinearOp::SlotAddr { dst, slot } => {
                let rd = self.dst_reg_or_temp(*dst, Reg::X16);
                let off = self.slot_off(slot.index() as u32);
                self.ectx
                    .emit
                    .emit_add_imm(Width::X64, rd, Reg::SP, off as u16, false)
                    .expect("add");
                if rd == Reg::X16 {
                    self.store_to_vreg(*dst, Reg::X16);
                }
            }

            LinearOp::StoreToAddr { addr, src, width } => {
                let addr_reg = self.reg_for_vreg_with_temp(*addr, Reg::X16);
                let src_reg = self.reg_for_vreg_with_temp(*src, Reg::X10);
                match width {
                    kajit_ir::Width::W1 => {
                        self.ectx
                            .emit
                            .emit_strb_imm(src_reg, addr_reg, 0)
                            .expect("strb");
                    }
                    kajit_ir::Width::W2 => {
                        self.ectx
                            .emit
                            .emit_strh_imm(src_reg, addr_reg, 0)
                            .expect("strh");
                    }
                    kajit_ir::Width::W4 => {
                        self.ectx
                            .emit
                            .emit_str_imm(Width::W32, src_reg, addr_reg, 0)
                            .expect("str");
                    }
                    kajit_ir::Width::W8 => {
                        self.ectx
                            .emit
                            .emit_str_imm(Width::X64, src_reg, addr_reg, 0)
                            .expect("str");
                    }
                }
            }

            LinearOp::LoadFromAddr { dst, addr, width } => {
                // Check for fused base+offset: skip the Add, use [base, #offset]
                let (base_reg, offset) =
                    if let Some(&(base_vreg, off)) = self.fused_addr_offsets.get(addr) {
                        (self.reg_for_vreg_with_temp(base_vreg, Reg::X10), off as u32)
                    } else {
                        (self.reg_for_vreg_with_temp(*addr, Reg::X10), 0)
                    };
                let mut used_scratch = false;
                let assigned = self.dst_reg_or_temp(*dst, Reg::X16);
                let rd = if assigned == base_reg {
                    used_scratch = true;
                    if base_reg != Reg::X16 {
                        Reg::X16
                    } else if base_reg != Reg::X10 {
                        Reg::X10
                    } else {
                        Reg::X11
                    }
                } else {
                    assigned
                };
                match width {
                    kajit_ir::Width::W1 => {
                        self.ectx
                            .emit
                            .emit_ldrb_imm(rd, base_reg, offset)
                            .expect("ldrb");
                    }
                    kajit_ir::Width::W2 => {
                        self.ectx
                            .emit
                            .emit_ldrh_imm(rd, base_reg, offset)
                            .expect("ldrh");
                    }
                    kajit_ir::Width::W4 => {
                        self.ectx
                            .emit
                            .emit_ldr_imm(Width::W32, rd, base_reg, offset)
                            .expect("ldr");
                    }
                    kajit_ir::Width::W8 => {
                        self.ectx
                            .emit
                            .emit_ldr_imm(Width::X64, rd, base_reg, offset)
                            .expect("ldr");
                    }
                }
                if used_scratch || rd == Reg::X16 {
                    self.store_to_vreg(*dst, rd);
                }
            }

            LinearOp::WriteToSlot { slot, src } => {
                let src_reg = self.reg_for_vreg_with_temp(*src, Reg::X16);
                let off = self.slot_off(slot.index() as u32);
                self.ectx
                    .emit
                    .emit_str_imm(Width::X64, src_reg, Reg::SP, off)
                    .expect("str slot");
            }

            LinearOp::ReadFromSlot { dst, slot } => {
                let rd = self.dst_reg_or_temp(*dst, Reg::X16);
                let off = self.slot_off(slot.index() as u32);
                self.ectx
                    .emit
                    .emit_ldr_imm(Width::X64, rd, Reg::SP, off)
                    .expect("ldr slot");
                if rd == Reg::X16 {
                    self.store_to_vreg(*dst, Reg::X16);
                }
            }

            LinearOp::ErrorExit { code } => {
                self.ectx.emit_error_with_ctx_reg(*code, self.ctx_reg);
            }

            LinearOp::CallIntrinsic { func, args, dst } => {
                self.emit_call_intrinsic(*func, args, *dst);
            }

            LinearOp::CallPure { func, args, dst } | LinearOp::CallEffect { func, args, dst } => {
                self.emit_call_pure(*func, args, *dst);
            }

            LinearOp::CallLambda { .. } => {
                // TODO: multi-function support
                self.ectx.emit.emit_nop().expect("nop");
            }

            LinearOp::FuncStart { .. }
            | LinearOp::FuncEnd
            | LinearOp::Label(_)
            | LinearOp::Branch { .. }
            | LinearOp::BranchIf { .. }
            | LinearOp::BranchIfZero { .. }
            | LinearOp::JumpTable { .. } => {
                panic!(
                    "structural op {:?} should not appear as a CFG-MIR instruction",
                    inst.op
                );
            }
        }
        self.current_inst = None;
    }

    /// Emit a binary operation.
    pub(super) fn emit_binop(
        &mut self,
        kind: BinOpKind,
        dst: kajit_ir::VReg,
        lhs: kajit_ir::VReg,
        rhs: kajit_ir::VReg,
    ) {
        // Comparisons: cmp + cset (with immediate folding)
        if matches!(
            kind,
            BinOpKind::CmpEq
                | BinOpKind::CmpNe
                | BinOpKind::CmpLt
                | BinOpKind::CmpLe
                | BinOpKind::CmpGt
                | BinOpKind::CmpGe
        ) {
            // Fold rhs constant into cmp immediate.
            // If lhs is a small const and rhs is not, swap operands and invert
            // the condition to use cmp imm (e.g., CmpGt(1, x) → cmp x, #1 + Lo).
            let (cmp_lhs, cmp_rhs, swapped) = if self.small_const(rhs).is_some() {
                (lhs, rhs, false)
            } else if self.small_const(lhs).is_some() {
                (rhs, lhs, true)
            } else {
                (lhs, rhs, false)
            };

            let cmp_lhs_reg = self.reg_for_vreg_with_temp(cmp_lhs, Reg::X16);
            if let Some(imm) = self.small_const(cmp_rhs) {
                self.ectx
                    .emit
                    .emit_cmp_imm(Width::X64, cmp_lhs_reg, imm)
                    .expect("cmp imm");
            } else {
                let cmp_rhs_reg = self.reg_for_vreg_with_temp(cmp_rhs, Reg::X10);
                self.ectx
                    .emit
                    .emit_cmp_reg(Width::X64, cmp_lhs_reg, cmp_rhs_reg)
                    .expect("cmp");
            }
            // Skip cset if this comparison is fused with its branch terminator
            if self.fused_cmps.contains_key(&dst) {
                // Update the fused condition if we swapped operands
                if swapped {
                    let cc = self.fused_cmps.get_mut(&dst).unwrap();
                    *cc = cc.swap_operands();
                }
                return;
            }
            let condition = match kind {
                BinOpKind::CmpEq => Condition::Eq,
                BinOpKind::CmpNe => Condition::Ne,
                BinOpKind::CmpLt => Condition::Lo,
                BinOpKind::CmpLe => Condition::Ls,
                BinOpKind::CmpGt => Condition::Hi,
                BinOpKind::CmpGe => Condition::Hs,
                _ => unreachable!(),
            };
            // Swap the condition if we swapped operands
            let condition = if swapped {
                condition.swap_operands()
            } else {
                condition
            };
            // Emit cset directly into dst register if possible
            let cset_dst = if let Some(preg) = self.preg_for_vreg(dst) {
                self.preg_to_reg(preg)
            } else {
                Reg::X16
            };
            self.ectx
                .emit
                .emit_cset(Width::X64, cset_dst, condition)
                .expect("cset");
            if cset_dst == Reg::X16 {
                self.store_to_vreg(dst, Reg::X16);
            }
            return;
        }

        // Try to fold a small constant rhs into an immediate-form instruction
        if let Some(imm) = self.small_const(rhs)
            && matches!(kind, BinOpKind::Add | BinOpKind::Sub)
        {
            let lhs_reg = self.reg_for_vreg_with_temp(lhs, Reg::X16);
            let result_reg = if let Some(preg) = self.preg_for_vreg(dst) {
                self.preg_to_reg(preg)
            } else {
                Reg::X11
            };
            match kind {
                BinOpKind::Add => {
                    self.ectx
                        .emit
                        .emit_add_imm(Width::X64, result_reg, lhs_reg, imm, false)
                        .expect("add imm");
                }
                BinOpKind::Sub => {
                    self.ectx
                        .emit
                        .emit_sub_imm(Width::X64, result_reg, lhs_reg, imm, false)
                        .expect("sub imm");
                }
                _ => unreachable!(),
            }
            if result_reg == Reg::X11 {
                self.store_to_vreg(dst, result_reg);
            }
            return;
        }

        // Arithmetic/logic: load operands, compute, store
        let lhs_reg = self.reg_for_vreg_with_temp(lhs, Reg::X16);
        let rhs_reg = self.reg_for_vreg_with_temp(rhs, Reg::X10);
        // Compute directly into dst register when possible.
        // On aarch64, ALU instructions read inputs before writing the result,
        // so rd = rn or rd = rm is safe for single-instruction ops.
        let result_reg = if let Some(preg) = self.preg_for_vreg(dst) {
            self.preg_to_reg(preg)
        } else {
            Reg::X11
        };

        match kind {
            BinOpKind::Add => {
                self.ectx
                    .emit
                    .emit_add_reg(Width::X64, result_reg, lhs_reg, rhs_reg)
                    .expect("add");
            }
            BinOpKind::Sub => {
                self.ectx
                    .emit
                    .emit_sub_reg(Width::X64, result_reg, lhs_reg, rhs_reg)
                    .expect("sub");
            }
            BinOpKind::Mul => {
                self.ectx
                    .emit
                    .emit_mul_reg(Width::X64, result_reg, lhs_reg, rhs_reg)
                    .expect("mul");
            }
            BinOpKind::And => {
                // Try logical immediate encoding
                if let Some(&val) = self.const_values.get(&rhs)
                    && self
                        .ectx
                        .emit
                        .emit_and_imm(Width::X64, result_reg, lhs_reg, val)
                        .is_ok()
                {
                    if result_reg == Reg::X11 {
                        self.store_to_vreg(dst, result_reg);
                    }
                    return;
                }
                self.ectx
                    .emit
                    .emit_and_reg(Width::X64, result_reg, lhs_reg, rhs_reg)
                    .expect("and");
            }
            BinOpKind::Or => {
                // Check for bfi fusion: Or(accum, shifted) → bfi(accum, byte_src, lsb, width)
                let bfi_info = self
                    .fused_bfi
                    .get(&dst)
                    .map(|b| (b.byte_src, b.accum, b.lsb, b.width));
                if let Some((byte_src, accum, bfi_lsb, bfi_width)) = bfi_info {
                    let byte_reg = self.reg_for_vreg_with_temp(byte_src, Reg::X16);
                    let accum_reg = self.reg_for_vreg_with_temp(accum, Reg::X10);
                    // bfi modifies Rd in place, so ensure result_reg == accum_reg
                    if result_reg != accum_reg {
                        self.ectx
                            .emit
                            .emit_mov_reg(Width::X64, result_reg, accum_reg)
                            .expect("mov for bfi");
                    }
                    self.ectx
                        .emit
                        .emit_bfi(Width::X64, result_reg, byte_reg, bfi_lsb, bfi_width)
                        .expect("bfi");
                } else {
                    self.ectx
                        .emit
                        .emit_orr_reg(Width::X64, result_reg, lhs_reg, rhs_reg)
                        .expect("orr");
                }
            }
            BinOpKind::Xor => {
                self.ectx
                    .emit
                    .emit_eor_reg(Width::X64, result_reg, lhs_reg, rhs_reg)
                    .expect("eor");
            }
            BinOpKind::Shl => {
                // Try immediate encoding for constant shift amounts
                if let Some(&val) = self.const_values.get(&rhs)
                    && val < 64
                {
                    self.ectx
                        .emit
                        .emit_lsl_imm(Width::X64, result_reg, lhs_reg, val as u8)
                        .expect("lsl imm");
                    if result_reg == Reg::X11 {
                        self.store_to_vreg(dst, result_reg);
                    }
                    return;
                }
                self.ectx
                    .emit
                    .emit_lsl_reg(Width::X64, result_reg, lhs_reg, rhs_reg)
                    .expect("lsl");
            }
            BinOpKind::Shr => {
                // Try immediate encoding for constant shift amounts
                if let Some(&val) = self.const_values.get(&rhs)
                    && val < 64
                {
                    self.ectx
                        .emit
                        .emit_lsr_imm(Width::X64, result_reg, lhs_reg, val as u8)
                        .expect("lsr imm");
                    if result_reg == Reg::X11 {
                        self.store_to_vreg(dst, result_reg);
                    }
                    return;
                }
                self.ectx
                    .emit
                    .emit_lsr_reg(Width::X64, result_reg, lhs_reg, rhs_reg)
                    .expect("lsr");
            }
            BinOpKind::Sar => {
                self.ectx
                    .emit
                    .emit_asr_reg(Width::X64, result_reg, lhs_reg, rhs_reg)
                    .expect("asr");
            }
            _ => unreachable!("comparison ops handled above"),
        }

        // Only store if we didn't already compute into the target
        if result_reg == Reg::X11 {
            self.store_to_vreg(dst, result_reg);
        }
    }

    /// Emit a unary operation.
    pub(super) fn emit_unary(
        &mut self,
        kind: UnaryOpKind,
        dst: kajit_ir::VReg,
        src: kajit_ir::VReg,
    ) {
        let src_reg = self.reg_for_vreg_with_temp(src, Reg::X16);
        match kind {
            UnaryOpKind::ZigzagDecode { wide } => {
                // zigzag_decode(n) = (n >> 1) ^ -(n & 1)
                // x9 = src
                // x10 = src >> 1
                // x11 = -(src & 1) = neg(src & 1)
                // result = x10 ^ x11
                let w = if wide { Width::X64 } else { Width::W32 };
                self.ectx
                    .emit
                    .emit_lsr_imm(w, Reg::X10, src_reg, 1)
                    .expect("lsr");
                // src & 1
                self.ectx
                    .emit
                    .emit_and_imm(w, Reg::X11, src_reg, 1)
                    .expect("and");
                // neg
                self.ectx
                    .emit
                    .emit_neg_reg(w, Reg::X11, Reg::X11)
                    .expect("neg");
                // xor
                self.ectx
                    .emit
                    .emit_eor_reg(w, Reg::X16, Reg::X10, Reg::X11)
                    .expect("eor");
                self.store_to_vreg(dst, Reg::X16);
            }
            UnaryOpKind::SignExtend { from_width } => {
                match from_width {
                    kajit_ir::Width::W1 => {
                        self.ectx
                            .emit
                            .emit_sxtb(Width::X64, Reg::X16, src_reg)
                            .expect("sxtb");
                    }
                    kajit_ir::Width::W2 => {
                        self.ectx
                            .emit
                            .emit_sxth(Width::X64, Reg::X16, src_reg)
                            .expect("sxth");
                    }
                    kajit_ir::Width::W4 => {
                        self.ectx.emit.emit_sxtw(Reg::X16, src_reg).expect("sxtw");
                    }
                    kajit_ir::Width::W8 => {
                        // 64-bit to 64-bit sign extend is a no-op
                        self.ectx
                            .emit
                            .emit_mov_reg(Width::X64, Reg::X16, src_reg)
                            .expect("mov");
                    }
                }
                self.store_to_vreg(dst, Reg::X16);
            }
        }
    }
}
