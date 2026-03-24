//! aarch64 backend for regalloc3 (native types, no regalloc2 conversion).

use kajit_emit::aarch64::{Condition, LabelId, Reg, Width};
use kajit_mir::cfg_mir::{self, Function, Inst, Terminator};
use kajit_mir::regalloc3::machine_inst::PReg;
use kajit_mir::regalloc3_result::{AllocatedCfgFunctionRa3, AllocatedCfgProgramRa3};

use crate::arch::EmitCtx;
use crate::ir_backend::{BackendDebugInfo, LinearBackendResult};
use kajit_lir::{BinOpKind, LinearOp, UnaryOpKind};
use std::collections::HashMap;

/// Context for emitting a single function.
struct EmitContext<'a> {
    ectx: &'a mut EmitCtx,
    func: &'a Function,
    alloc_func: &'a AllocatedCfgFunctionRa3,
    block_labels: HashMap<cfg_mir::BlockId, LabelId>,
    success_exit: LabelId,
    /// Slot offset base: base_frame + spill_slots * 8 gives the start of user slots.
    slot_base: u32,
    /// VReg → constant value (for immediate folding in BinOps)
    const_values: HashMap<kajit_ir::VReg, u64>,
}

impl<'a> EmitContext<'a> {
    /// Get physical register for a vreg, or None if spilled/dead.
    fn preg_for_vreg(&self, vreg: kajit_ir::VReg) -> Option<PReg> {
        self.alloc_func.preg_for_vreg(vreg)
    }

    /// Convert regalloc3 PReg to kajit_emit Reg.
    fn preg_to_reg(&self, preg: PReg) -> Reg {
        Reg::from_raw(preg.0)
    }

    /// Get hardware register for a vreg, or use a temp register and load from spill slot.
    /// For spilled constants, rematerializes with movz instead of loading from stack.
    fn reg_for_vreg_with_temp(&mut self, vreg: kajit_ir::VReg, temp: Reg) -> Reg {
        if let Some(preg) = self.preg_for_vreg(vreg) {
            return self.preg_to_reg(preg);
        }

        // Rematerializable constant - emit movz instead of stack load
        if let Some(&value) = self.alloc_func.rematerializable.get(&vreg) {
            self.emit_load_u64(temp, value);
            return temp;
        }

        // Spilled - load from spill slot
        if let Some(slot) = self.alloc_func.spill_slot_for_vreg(vreg) {
            let offset = self.ectx.base_frame + (slot.0 * 8);
            self.ectx
                .emit
                .emit_ldr_imm(Width::X64, temp, Reg::SP, offset)
                .expect("ldr spill");
            return temp;
        }

        // Dead vreg - use temp with dummy value
        self.ectx
            .emit
            .emit_movz_imm(Width::X64, temp, 0, 0)
            .expect("movz dead");
        temp
    }

    /// Store a value from a register to a vreg (handling spills).
    fn store_to_vreg(&mut self, vreg: kajit_ir::VReg, from_reg: Reg) {
        if let Some(preg) = self.preg_for_vreg(vreg) {
            let dst_reg = self.preg_to_reg(preg);
            if dst_reg != from_reg {
                self.ectx
                    .emit
                    .emit_mov_reg(Width::X64, dst_reg, from_reg)
                    .expect("mov");
            }
        } else if let Some(slot) = self.alloc_func.spill_slot_for_vreg(vreg) {
            let offset = self.ectx.base_frame + (slot.0 * 8);
            self.ectx
                .emit
                .emit_str_imm(Width::X64, from_reg, Reg::SP, offset)
                .expect("str spill");
        }
        // If dead, do nothing
    }

    /// Load a 64-bit constant into a register.
    fn emit_load_u64(&mut self, rd: Reg, value: u64) {
        let p0 = (value & 0xFFFF) as u16;
        let p1 = ((value >> 16) & 0xFFFF) as u16;
        let p2 = ((value >> 32) & 0xFFFF) as u16;
        let p3 = ((value >> 48) & 0xFFFF) as u16;
        self.ectx
            .emit
            .emit_movz_imm(Width::X64, rd, p0, 0)
            .expect("movz");
        if p1 != 0 {
            self.ectx
                .emit
                .emit_movk_imm(Width::X64, rd, p1, 16)
                .expect("movk");
        }
        if p2 != 0 {
            self.ectx
                .emit
                .emit_movk_imm(Width::X64, rd, p2, 32)
                .expect("movk");
        }
        if p3 != 0 {
            self.ectx
                .emit
                .emit_movk_imm(Width::X64, rd, p3, 48)
                .expect("movk");
        }
    }

    /// If vreg is a known constant that fits in a 12-bit immediate, return its value.
    fn small_const(&self, vreg: kajit_ir::VReg) -> Option<u16> {
        let value = self.const_values.get(&vreg)?;
        if *value <= 0xFFF {
            Some(*value as u16)
        } else {
            None
        }
    }

    /// Offset of a user slot on the stack.
    fn slot_off(&self, slot: u32) -> u32 {
        self.slot_base + slot * 8
    }

    /// Emit a single instruction.
    fn emit_inst(&mut self, inst: &Inst) {
        match &inst.op {
            LinearOp::Copy { dst, src } => {
                // Elide copy when src and dst are in the same register
                if let (Some(sp), Some(dp)) = (self.preg_for_vreg(*src), self.preg_for_vreg(*dst)) {
                    if sp == dp {
                        return; // nop
                    }
                }
                let src_reg = self.reg_for_vreg_with_temp(*src, Reg::X9);
                self.store_to_vreg(*dst, src_reg);
            }

            LinearOp::Const { dst, value } => {
                if let Some(preg) = self.preg_for_vreg(*dst) {
                    self.emit_load_u64(self.preg_to_reg(preg), *value);
                } else if self.alloc_func.rematerializable.contains_key(dst) {
                    // Rematerializable constant: skip store to spill slot.
                    // All reads of this vreg will re-emit movz instead.
                } else {
                    // Spilled - load into x9, store to spill slot
                    self.emit_load_u64(Reg::X9, *value);
                    self.store_to_vreg(*dst, Reg::X9);
                }
            }

            LinearOp::BinOp { op, dst, lhs, rhs } => {
                self.emit_binop(*op, *dst, *lhs, *rhs);
            }

            LinearOp::UnaryOp { op, dst, src } => {
                self.emit_unary(*op, *dst, *src);
            }

            LinearOp::SaveCursor { dst } => {
                if let Some(preg) = self.preg_for_vreg(*dst) {
                    let dst_reg = self.preg_to_reg(preg);
                    self.ectx
                        .emit
                        .emit_mov_reg(Width::X64, dst_reg, Reg::X19)
                        .expect("mov");
                } else {
                    self.ectx
                        .emit
                        .emit_mov_reg(Width::X64, Reg::X9, Reg::X19)
                        .expect("mov");
                    self.store_to_vreg(*dst, Reg::X9);
                }
            }

            LinearOp::SaveInputEnd { dst } => {
                if let Some(preg) = self.preg_for_vreg(*dst) {
                    let dst_reg = self.preg_to_reg(preg);
                    self.ectx
                        .emit
                        .emit_mov_reg(Width::X64, dst_reg, Reg::X20)
                        .expect("mov");
                } else {
                    self.ectx
                        .emit
                        .emit_mov_reg(Width::X64, Reg::X9, Reg::X20)
                        .expect("mov");
                    self.store_to_vreg(*dst, Reg::X9);
                }
            }

            LinearOp::RestoreCursor { src } => {
                let src_reg = self.reg_for_vreg_with_temp(*src, Reg::X9);
                self.ectx
                    .emit
                    .emit_mov_reg(Width::X64, Reg::X19, src_reg)
                    .expect("mov");
            }

            LinearOp::BoundsCheck { count } => {
                self.ectx.emit_bounds_check(*count);
            }

            LinearOp::ReadBytes { dst, count } => {
                // Read from cursor (x19), store to dst
                match count {
                    1 => {
                        self.ectx
                            .emit
                            .emit_ldrb_imm(Reg::X9, Reg::X19, 0)
                            .expect("ldrb");
                    }
                    2 => {
                        self.ectx
                            .emit
                            .emit_ldrh_imm(Reg::X9, Reg::X19, 0)
                            .expect("ldrh");
                    }
                    4 => {
                        self.ectx
                            .emit
                            .emit_ldr_imm(Width::W32, Reg::X9, Reg::X19, 0)
                            .expect("ldr");
                    }
                    8 => {
                        self.ectx
                            .emit
                            .emit_ldr_imm(Width::X64, Reg::X9, Reg::X19, 0)
                            .expect("ldr");
                    }
                    _ => {
                        self.ectx.emit.emit_nop().expect("nop");
                        return;
                    }
                }
                self.store_to_vreg(*dst, Reg::X9);
            }

            LinearOp::PeekByte { dst } => {
                // Read byte at cursor without advancing
                self.ectx
                    .emit
                    .emit_ldrb_imm(Reg::X9, Reg::X19, 0)
                    .expect("ldrb");
                self.store_to_vreg(*dst, Reg::X9);
            }

            LinearOp::AdvanceCursor { count } => {
                self.ectx.emit_advance_cursor_by(*count);
            }

            LinearOp::AdvanceCursorBy { src } => {
                let src_reg = self.reg_for_vreg_with_temp(*src, Reg::X9);
                self.ectx
                    .emit
                    .emit_add_reg(Width::X64, Reg::X19, Reg::X19, src_reg)
                    .expect("add");
            }

            LinearOp::WriteToField { src, offset, width } => {
                let src_reg = self.reg_for_vreg_with_temp(*src, Reg::X9);
                // Out pointer is in x21
                match width {
                    kajit_ir::Width::W1 => {
                        self.ectx
                            .emit
                            .emit_strb_imm(src_reg, Reg::X21, *offset)
                            .expect("strb");
                    }
                    kajit_ir::Width::W2 => {
                        self.ectx
                            .emit
                            .emit_strh_imm(src_reg, Reg::X21, *offset)
                            .expect("strh");
                    }
                    kajit_ir::Width::W4 => {
                        self.ectx
                            .emit
                            .emit_str_imm(Width::W32, src_reg, Reg::X21, *offset)
                            .expect("str");
                    }
                    kajit_ir::Width::W8 => {
                        self.ectx
                            .emit
                            .emit_str_imm(Width::X64, src_reg, Reg::X21, *offset)
                            .expect("str");
                    }
                }
            }

            LinearOp::ReadFromField { dst, offset, width } => {
                // Out pointer is in x21
                match width {
                    kajit_ir::Width::W1 => {
                        self.ectx
                            .emit
                            .emit_ldrb_imm(Reg::X9, Reg::X21, *offset)
                            .expect("ldrb");
                    }
                    kajit_ir::Width::W2 => {
                        self.ectx
                            .emit
                            .emit_ldrh_imm(Reg::X9, Reg::X21, *offset)
                            .expect("ldrh");
                    }
                    kajit_ir::Width::W4 => {
                        self.ectx
                            .emit
                            .emit_ldr_imm(Width::W32, Reg::X9, Reg::X21, *offset)
                            .expect("ldr");
                    }
                    kajit_ir::Width::W8 => {
                        self.ectx
                            .emit
                            .emit_ldr_imm(Width::X64, Reg::X9, Reg::X21, *offset)
                            .expect("ldr");
                    }
                }
                self.store_to_vreg(*dst, Reg::X9);
            }

            LinearOp::SaveOutPtr { dst } => {
                // Out pointer is in x21
                self.ectx
                    .emit
                    .emit_mov_reg(Width::X64, Reg::X9, Reg::X21)
                    .expect("mov");
                self.store_to_vreg(*dst, Reg::X9);
            }

            LinearOp::SetOutPtr { src } => {
                let src_reg = self.reg_for_vreg_with_temp(*src, Reg::X9);
                self.ectx
                    .emit
                    .emit_mov_reg(Width::X64, Reg::X21, src_reg)
                    .expect("mov");
            }

            LinearOp::SlotAddr { dst, slot } => {
                // Compute address of stack slot
                let off = self.slot_off(slot.index() as u32);
                self.ectx
                    .emit
                    .emit_add_imm(Width::X64, Reg::X9, Reg::SP, off as u16, false)
                    .expect("add");
                self.store_to_vreg(*dst, Reg::X9);
            }

            LinearOp::StoreToAddr { addr, src, width } => {
                let addr_reg = self.reg_for_vreg_with_temp(*addr, Reg::X9);
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
                let addr_reg = self.reg_for_vreg_with_temp(*addr, Reg::X10);
                match width {
                    kajit_ir::Width::W1 => {
                        self.ectx
                            .emit
                            .emit_ldrb_imm(Reg::X9, addr_reg, 0)
                            .expect("ldrb");
                    }
                    kajit_ir::Width::W2 => {
                        self.ectx
                            .emit
                            .emit_ldrh_imm(Reg::X9, addr_reg, 0)
                            .expect("ldrh");
                    }
                    kajit_ir::Width::W4 => {
                        self.ectx
                            .emit
                            .emit_ldr_imm(Width::W32, Reg::X9, addr_reg, 0)
                            .expect("ldr");
                    }
                    kajit_ir::Width::W8 => {
                        self.ectx
                            .emit
                            .emit_ldr_imm(Width::X64, Reg::X9, addr_reg, 0)
                            .expect("ldr");
                    }
                }
                self.store_to_vreg(*dst, Reg::X9);
            }

            LinearOp::WriteToSlot { slot, src } => {
                let src_reg = self.reg_for_vreg_with_temp(*src, Reg::X9);
                let off = self.slot_off(slot.index() as u32);
                self.ectx
                    .emit
                    .emit_str_imm(Width::X64, src_reg, Reg::SP, off)
                    .expect("str slot");
            }

            LinearOp::ReadFromSlot { dst, slot } => {
                let off = self.slot_off(slot.index() as u32);
                self.ectx
                    .emit
                    .emit_ldr_imm(Width::X64, Reg::X9, Reg::SP, off)
                    .expect("ldr slot");
                self.store_to_vreg(*dst, Reg::X9);
            }

            LinearOp::ErrorExit { code } => {
                self.ectx.emit_error(*code);
            }

            LinearOp::CallIntrinsic {
                func,
                args,
                dst,
                field_offset,
            } => {
                self.emit_call_intrinsic(*func, args, *dst, Some(*field_offset));
            }

            LinearOp::CallPure { func, args, dst } => {
                self.emit_call_pure(*func, args, *dst);
            }

            LinearOp::CallLambda { .. } => {
                // TODO: multi-function support
                self.ectx.emit.emit_nop().expect("nop");
            }

            LinearOp::SimdStringScan { .. } | LinearOp::SimdWhitespaceSkip => {
                panic!("unsupported SIMD op in regalloc3 aarch64 backend");
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
    }

    /// Emit a binary operation.
    fn emit_binop(
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
            let lhs_reg = self.reg_for_vreg_with_temp(lhs, Reg::X9);
            // Fold rhs constant into cmp immediate
            if let Some(imm) = self.small_const(rhs) {
                self.ectx
                    .emit
                    .emit_cmp_imm(Width::X64, lhs_reg, imm)
                    .expect("cmp imm");
            } else {
                let rhs_reg = self.reg_for_vreg_with_temp(rhs, Reg::X10);
                self.ectx
                    .emit
                    .emit_cmp_reg(Width::X64, lhs_reg, rhs_reg)
                    .expect("cmp");
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
            // Emit cset directly into dst register if possible
            let cset_dst = if let Some(preg) = self.preg_for_vreg(dst) {
                self.preg_to_reg(preg)
            } else {
                Reg::X9
            };
            self.ectx
                .emit
                .emit_cset(Width::X64, cset_dst, condition)
                .expect("cset");
            if cset_dst == Reg::X9 {
                self.store_to_vreg(dst, Reg::X9);
            }
            return;
        }

        // Try to fold a small constant rhs into an immediate-form instruction
        if let Some(imm) = self.small_const(rhs) {
            if matches!(kind, BinOpKind::Add | BinOpKind::Sub) {
                let lhs_reg = self.reg_for_vreg_with_temp(lhs, Reg::X9);
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
        }

        // Arithmetic/logic: load operands, compute, store
        let lhs_reg = self.reg_for_vreg_with_temp(lhs, Reg::X9);
        let rhs_reg = self.reg_for_vreg_with_temp(rhs, Reg::X10);
        // Compute directly into dst register when possible
        let result_reg = if let Some(preg) = self.preg_for_vreg(dst) {
            let r = self.preg_to_reg(preg);
            // Don't clobber an input register
            if r != lhs_reg && r != rhs_reg {
                r
            } else {
                Reg::X11
            }
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
                self.ectx
                    .emit
                    .emit_and_reg(Width::X64, result_reg, lhs_reg, rhs_reg)
                    .expect("and");
            }
            BinOpKind::Or => {
                self.ectx
                    .emit
                    .emit_orr_reg(Width::X64, result_reg, lhs_reg, rhs_reg)
                    .expect("orr");
            }
            BinOpKind::Xor => {
                self.ectx
                    .emit
                    .emit_eor_reg(Width::X64, result_reg, lhs_reg, rhs_reg)
                    .expect("eor");
            }
            BinOpKind::Shl => {
                self.ectx
                    .emit
                    .emit_lsl_reg(Width::X64, result_reg, lhs_reg, rhs_reg)
                    .expect("lsl");
            }
            BinOpKind::Shr => {
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
    fn emit_unary(&mut self, kind: UnaryOpKind, dst: kajit_ir::VReg, src: kajit_ir::VReg) {
        let src_reg = self.reg_for_vreg_with_temp(src, Reg::X9);
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
                    .emit_eor_reg(w, Reg::X9, Reg::X10, Reg::X11)
                    .expect("eor");
                self.store_to_vreg(dst, Reg::X9);
            }
            UnaryOpKind::SignExtend { from_width } => {
                match from_width {
                    kajit_ir::Width::W1 => {
                        self.ectx
                            .emit
                            .emit_sxtb(Width::X64, Reg::X9, src_reg)
                            .expect("sxtb");
                    }
                    kajit_ir::Width::W2 => {
                        self.ectx
                            .emit
                            .emit_sxth(Width::X64, Reg::X9, src_reg)
                            .expect("sxth");
                    }
                    kajit_ir::Width::W4 => {
                        self.ectx.emit.emit_sxtw(Reg::X9, src_reg).expect("sxtw");
                    }
                    kajit_ir::Width::W8 => {
                        // 64-bit to 64-bit sign extend is a no-op
                        self.ectx
                            .emit
                            .emit_mov_reg(Width::X64, Reg::X9, src_reg)
                            .expect("mov");
                    }
                }
                self.store_to_vreg(dst, Reg::X9);
            }
        }
    }

    /// Emit a call to an intrinsic function.
    /// ABI: x0=ctx, x1+=args, with out_field offset adjustment on x21.
    fn emit_call_intrinsic(
        &mut self,
        func: kajit_ir::IntrinsicFn,
        args: &[kajit_ir::VReg],
        dst: Option<kajit_ir::VReg>,
        field_offset: Option<u32>,
    ) {
        use crate::context::{CTX_ERROR_CODE, CTX_INPUT_PTR};

        let error_exit = self.ectx.error_exit;

        // Flush cursor to ctx before call
        self.ectx
            .emit
            .emit_str_imm(Width::X64, Reg::X19, Reg::X22, CTX_INPUT_PTR)
            .expect("str cursor");

        // Adjust out_ptr for field offset if needed
        if let Some(off) = field_offset {
            if off > 0 {
                self.ectx
                    .emit
                    .emit_add_imm(Width::X64, Reg::X21, Reg::X21, off as u16, false)
                    .expect("add out_ptr");
            }
        }

        // Load args into x1+ (x0=ctx)
        for (i, &arg) in args.iter().enumerate() {
            let target_reg = Reg::from_raw((i + 1) as u8);
            let src_reg = self.reg_for_vreg_with_temp(arg, Reg::X9);
            if src_reg != target_reg {
                self.ectx
                    .emit
                    .emit_mov_reg(Width::X64, target_reg, src_reg)
                    .expect("mov arg");
            }
        }

        // If no dst but field_offset, add out_field arg
        if dst.is_none() {
            if let Some(off) = field_offset {
                let arg_idx = args.len() + 1;
                self.ectx
                    .emit
                    .emit_add_imm(
                        Width::X64,
                        Reg::from_raw(arg_idx as u8),
                        Reg::X21,
                        off as u16,
                        false,
                    )
                    .expect("add out_field");
            }
        }

        // x0 = ctx
        self.ectx
            .emit
            .emit_mov_reg(Width::X64, Reg::X0, Reg::X22)
            .expect("mov ctx");

        // Load function pointer and call
        self.emit_load_u64(Reg::X16, func.0 as u64);
        self.ectx.emit.emit_blr(Reg::X16).expect("blr");

        // Restore out_ptr
        if let Some(off) = field_offset {
            if off > 0 {
                self.ectx
                    .emit
                    .emit_sub_imm(Width::X64, Reg::X21, Reg::X21, off as u16, false)
                    .expect("sub out_ptr");
            }
        }

        // Reload cursor from ctx after call
        self.ectx
            .emit
            .emit_ldr_imm(Width::X64, Reg::X19, Reg::X22, CTX_INPUT_PTR)
            .expect("ldr cursor");

        // Check error after call
        self.ectx
            .emit
            .emit_ldr_imm(Width::W32, Reg::X9, Reg::X22, CTX_ERROR_CODE)
            .expect("ldr error");
        self.ectx
            .emit
            .emit_cbnz_label(Width::W32, Reg::X9, error_exit)
            .expect("cbnz error");

        // Store result if needed (return value is in x0)
        if let Some(dst) = dst {
            self.store_to_vreg(dst, Reg::X0);
        }
    }

    /// Emit a call to a pure function (no ctx, no cursor flush).
    fn emit_call_pure(
        &mut self,
        func: kajit_ir::IntrinsicFn,
        args: &[kajit_ir::VReg],
        dst: kajit_ir::VReg,
    ) {
        // Load args into x0+
        for (i, &arg) in args.iter().enumerate() {
            let target_reg = Reg::from_raw(i as u8);
            let src_reg = self.reg_for_vreg_with_temp(arg, Reg::X9);
            if src_reg != target_reg {
                self.ectx
                    .emit
                    .emit_mov_reg(Width::X64, target_reg, src_reg)
                    .expect("mov arg");
            }
        }

        // Load function pointer and call
        self.emit_load_u64(Reg::X16, func.0 as u64);
        self.ectx.emit.emit_blr(Reg::X16).expect("blr");

        // Store result (return value is in x0)
        self.store_to_vreg(dst, Reg::X0);
    }

    /// Emit a terminator. `next_block` is the block that follows in emission order (for fallthrough elision).
    fn emit_terminator(&mut self, term: &Terminator, next_block: Option<cfg_mir::BlockId>) {
        match term {
            Terminator::Return => {
                let success_exit = self.success_exit;
                self.ectx
                    .emit
                    .emit_b_label(success_exit)
                    .expect("b success");
            }

            Terminator::Branch { edge } => {
                let target_block = self.func.edges[edge.index()].to;
                // Skip branch if target is the next block in emission order
                if Some(target_block) != next_block {
                    let label = self.block_labels[&target_block];
                    self.ectx.emit.emit_b_label(label).expect("branch");
                }
            }

            Terminator::BranchIf {
                cond,
                taken,
                fallthrough,
            } => {
                let cond_reg = self.reg_for_vreg_with_temp(*cond, Reg::X9);
                let taken_block = self.func.edges[taken.index()].to;
                let fallthrough_block = self.func.edges[fallthrough.index()].to;
                let taken_label = self.block_labels[&taken_block];
                self.ectx
                    .emit
                    .emit_cbnz_label(Width::X64, cond_reg, taken_label)
                    .expect("cbnz");
                // Only emit explicit branch if fallthrough isn't the next block
                if Some(fallthrough_block) != next_block {
                    let fallthrough_label = self.block_labels[&fallthrough_block];
                    self.ectx
                        .emit
                        .emit_b_label(fallthrough_label)
                        .expect("b fallthrough");
                }
            }

            Terminator::BranchIfZero {
                cond,
                taken,
                fallthrough,
            } => {
                let cond_reg = self.reg_for_vreg_with_temp(*cond, Reg::X9);
                let taken_block = self.func.edges[taken.index()].to;
                let fallthrough_block = self.func.edges[fallthrough.index()].to;
                let taken_label = self.block_labels[&taken_block];
                self.ectx
                    .emit
                    .emit_cbz_label(Width::X64, cond_reg, taken_label)
                    .expect("cbz");
                // Only emit explicit branch if fallthrough isn't the next block
                if Some(fallthrough_block) != next_block {
                    let fallthrough_label = self.block_labels[&fallthrough_block];
                    self.ectx
                        .emit
                        .emit_b_label(fallthrough_label)
                        .expect("b fallthrough");
                }
            }

            Terminator::ErrorExit { code } => {
                self.ectx.emit_error(*code);
            }

            Terminator::JumpTable { .. } => {
                panic!("JumpTable not yet supported in regalloc3 backend");
            }
        }
    }

    /// Emit all blocks for this function.
    fn emit_function(&mut self) {
        // Create labels for all blocks
        for block in &self.func.blocks {
            let label = self.ectx.new_label();
            self.block_labels.insert(block.id, label);
        }

        // Emit each block
        for block_idx in 0..self.func.blocks.len() {
            let block = &self.func.blocks[block_idx];
            if block.dead {
                continue;
            }

            // Bind label for this block (except entry which comes after prologue)
            if block.id.0 != 0 {
                let label = self.block_labels[&block.id];
                self.ectx.bind_label(label);
            }

            // Emit instructions
            for &inst_id in &block.insts {
                let inst = &self.func.insts[inst_id.index()];
                self.emit_inst(inst);
            }

            // Find next non-dead block in emission order
            let next_block_id = self.func.blocks[block_idx + 1..]
                .iter()
                .find(|b| !b.dead)
                .map(|b| b.id);

            // Emit terminator
            let term = &self.func.terms[block.term.0 as usize];
            self.emit_terminator(term, next_block_id);
        }
    }
}

/// Compile CFG-MIR with regalloc3 allocations to aarch64 machine code.
pub fn compile_regalloc3(alloc: &AllocatedCfgProgramRa3) -> LinearBackendResult {
    let program = &alloc.cfg_program;

    // Calculate max spillslots and extra callee-saved pairs needed
    let max_spillslots = alloc
        .functions
        .iter()
        .map(|f| f.num_spillslots)
        .max()
        .unwrap_or(0);

    // Check which callee-saved registers are used
    let extra_saved_pairs = regalloc3_extra_saved_pairs(alloc);

    // Detect leaf functions (no bl instructions needed)
    let is_leaf = program.funcs.iter().all(|func| {
        func.insts.iter().all(|inst| {
            !matches!(
                inst.op,
                LinearOp::CallIntrinsic { .. }
                    | LinearOp::CallPure { .. }
                    | LinearOp::CallLambda { .. }
            )
        })
    });

    // Create emission context with stack space for spills + user slots
    let extra_stack = ((max_spillslots + program.slot_count as usize) * 8) as u32;
    let mut ectx = EmitCtx::new_regalloc(extra_stack, extra_saved_pairs, is_leaf);
    let slot_base = ectx.base_frame + (max_spillslots * 8) as u32;

    // Emit function prologue
    let (entry, error_exit) = ectx.begin_func();

    // Create success exit label
    let success_exit = ectx.new_label();

    // Compile first function
    if let (Some(func), Some(alloc_func)) = (program.funcs.first(), alloc.functions.first()) {
        // Build constant value map for immediate folding
        let mut const_values = HashMap::new();
        for inst in &func.insts {
            if let LinearOp::Const { dst, value } = &inst.op {
                const_values.insert(*dst, *value);
            }
        }

        let mut ctx = EmitContext {
            ectx: &mut ectx,
            func,
            alloc_func,
            block_labels: HashMap::new(),
            success_exit,
            slot_base,
            const_values,
        };

        ctx.emit_function();
    }

    // Bind success exit and emit epilogue
    ectx.bind_label(success_exit);
    ectx.end_func(error_exit);

    // Finalize
    let (buf, asm_program) = ectx.finalize();

    LinearBackendResult {
        buf,
        entry,
        source_map: None,
        backend_debug_info: None,
        asm_program,
    }
}

/// Count how many callee-saved register pairs (x23/x24, x25/x26, x27/x28) are used.
fn regalloc3_extra_saved_pairs(alloc: &AllocatedCfgProgramRa3) -> u32 {
    use kajit_mir::regalloc3::linear_scan;

    let mut max_pair = None::<u32>;
    let mut observe = |a: &linear_scan::Allocation| {
        if let linear_scan::Allocation::Reg(preg) = a {
            let pair = match preg.0 {
                23 | 24 => Some(0),
                25 | 26 => Some(1),
                27 | 28 => Some(2),
                _ => None,
            };
            if let Some(pair) = pair {
                max_pair = Some(max_pair.map_or(pair, |cur| cur.max(pair)));
            }
        }
    };

    for func in &alloc.functions {
        for (_, a) in &func.allocations {
            observe(a);
        }
    }

    max_pair.map_or(0, |p| p + 1)
}
