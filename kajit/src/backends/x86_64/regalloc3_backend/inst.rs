//! Instruction emission for x86_64 regalloc3 backend.

use kajit_emit::x64::{self, Mem, Operand};
use kajit_lir::{BinOpKind, LinearOp, UnaryOpKind};

use crate::context::{CTX_INPUT_END, CTX_INPUT_PTR};

use super::context::EmitContext;

// Scratch register encodings (reserved by SCRATCH_POLICY, never allocated).
const R10: u8 = 10;
const R11: u8 = 11;
// RSP encoding for stack-relative addressing.
const RSP: u8 = 4;
// RCX encoding for shift amounts.
const RCX: u8 = 1;

impl<'a> EmitContext<'a> {
    /// Emit a single instruction.
    pub fn emit_inst(&mut self, inst: &kajit_mir::cfg_mir::Inst) {
        self.current_inst = Some(inst.id);
        // Skip instructions whose outputs were fused.
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
                if let (Some(sp), Some(dp)) = (self.preg_for_vreg(*src), self.preg_for_vreg(*dst)) {
                    if sp == dp {
                        self.current_inst = None;
                        return;
                    }
                }
                let src_enc = self.reg_for_vreg_with_temp(*src, R10);
                self.store_to_vreg(*dst, src_enc);
            }

            LinearOp::Const { dst, value } => {
                if inst.operands.is_empty() {
                    self.current_inst = None;
                    return;
                }
                if let Some(preg) = self.preg_for_vreg(*dst) {
                    self.emit_load_u64(self.preg_to_enc(preg), *value);
                } else if self.alloc_func.rematerializable.contains_key(dst) {
                    // Rematerializable: skip, reads will re-emit.
                } else {
                    self.emit_load_u64(R10, *value);
                    self.store_to_vreg(*dst, R10);
                }
            }

            LinearOp::DataAddr { dst, blob_id } => {
                let code_offset = self.ectx.emit.code_len();
                let dest_enc = self.dst_enc_or_temp(*dst, R10);
                self.emit_load_u64_fixed(dest_enc, 0);
                self.data_relocs.push(super::DataRelocInfo {
                    code_offset,
                    blob_id: *blob_id,
                });
                if self.preg_for_vreg(*dst).is_none() {
                    self.store_to_vreg(*dst, R10);
                }
            }

            LinearOp::BinOp { op, dst, lhs, rhs } => {
                self.emit_binop(*op, *dst, *lhs, *rhs);
            }

            LinearOp::UnaryOp { op, dst, src } => {
                self.emit_unary(*op, *dst, *src);
            }

            LinearOp::SaveCursor { dst } => {
                if self.ectx.is_leaf {
                    let dst_enc = self.dst_enc_or_temp(*dst, R10);
                    self.ectx
                        .emit
                        .emit_mov_r64_m(
                            dst_enc,
                            Mem {
                                base: self.ctx_enc,
                                disp: CTX_INPUT_PTR as i32,
                            },
                        )
                        .expect("mov cursor");
                    if dst_enc == R10 {
                        self.store_to_vreg(*dst, R10);
                    }
                } else {
                    let cursor_enc = self.cursor_writeback_enc;
                    let dst_enc = self.dst_enc_or_temp(*dst, R10);
                    self.ectx
                        .emit
                        .emit_mov_r64_r64(dst_enc, cursor_enc)
                        .expect("mov cursor");
                    if dst_enc == R10 {
                        self.store_to_vreg(*dst, R10);
                    }
                }
            }

            LinearOp::SaveInputEnd { dst } => {
                if self.ectx.is_leaf {
                    let dst_enc = self.dst_enc_or_temp(*dst, R10);
                    self.ectx
                        .emit
                        .emit_mov_r64_m(
                            dst_enc,
                            Mem {
                                base: self.ctx_enc,
                                disp: CTX_INPUT_END as i32,
                            },
                        )
                        .expect("mov input_end");
                    if dst_enc == R10 {
                        self.store_to_vreg(*dst, R10);
                    }
                } else {
                    let end_enc = self.ectx.end_enc;
                    let dst_enc = self.dst_enc_or_temp(*dst, R10);
                    self.ectx
                        .emit
                        .emit_mov_r64_r64(dst_enc, end_enc)
                        .expect("mov input_end");
                    if dst_enc == R10 {
                        self.store_to_vreg(*dst, R10);
                    }
                }
            }

            LinearOp::RestoreCursor { src } => {
                let cursor_enc = self.cursor_writeback_enc;
                if let Some(&(base_vreg, offset)) = self.fused_addr_offsets.get(src) {
                    let base_enc = self.reg_for_vreg_with_temp(base_vreg, R10);
                    self.ectx
                        .emit
                        .emit_lea_r64_m(
                            cursor_enc,
                            Mem {
                                base: base_enc,
                                disp: offset as i32,
                            },
                        )
                        .expect("lea restore_cursor");
                } else {
                    let src_enc = self.reg_for_vreg_with_temp(*src, R10);
                    if cursor_enc != src_enc {
                        self.ectx
                            .emit
                            .emit_mov_r64_r64(cursor_enc, src_enc)
                            .expect("mov restore_cursor");
                    }
                }
            }

            LinearOp::BoundsCheck { count } => {
                self.ectx.emit_bounds_check(*count);
            }

            LinearOp::ReadBytes { dst, count } => {
                let cursor_enc = self.ectx.cursor_enc;
                let rd = self.dst_enc_or_temp(*dst, R10);
                match count {
                    1 => self
                        .ectx
                        .emit
                        .emit_movzx_r32_rm8(
                            rd,
                            Operand::Mem(Mem {
                                base: cursor_enc,
                                disp: 0,
                            }),
                        )
                        .expect("movzx"),
                    2 => self
                        .ectx
                        .emit
                        .emit_movzx_r32_rm16(
                            rd,
                            Operand::Mem(Mem {
                                base: cursor_enc,
                                disp: 0,
                            }),
                        )
                        .expect("movzx"),
                    4 => self
                        .ectx
                        .emit
                        .emit_mov_r32_m(
                            rd,
                            Mem {
                                base: cursor_enc,
                                disp: 0,
                            },
                        )
                        .expect("mov"),
                    8 => self
                        .ectx
                        .emit
                        .emit_mov_r64_m(
                            rd,
                            Mem {
                                base: cursor_enc,
                                disp: 0,
                            },
                        )
                        .expect("mov"),
                    _ => panic!("unsupported ReadBytes count: {count}"),
                }
                if rd == R10 {
                    self.store_to_vreg(*dst, R10);
                }
            }

            LinearOp::PeekByte { dst } => {
                let cursor_enc = self.ectx.cursor_enc;
                let rd = self.dst_enc_or_temp(*dst, R10);
                self.ectx
                    .emit
                    .emit_movzx_r32_rm8(
                        rd,
                        Operand::Mem(Mem {
                            base: cursor_enc,
                            disp: 0,
                        }),
                    )
                    .expect("movzx");
                if rd == R10 {
                    self.store_to_vreg(*dst, R10);
                }
            }

            LinearOp::AdvanceCursor { count } => {
                self.ectx.emit_advance_cursor_by(*count);
            }

            LinearOp::AdvanceCursorBy { src } => {
                let cursor_enc = self.ectx.cursor_enc;
                let src_enc = self.reg_for_vreg_with_temp(*src, R10);
                self.ectx
                    .emit
                    .emit_add_r64_r64(cursor_enc, src_enc)
                    .expect("add cursor");
            }

            LinearOp::WriteToField { src, offset, width } => {
                let src_enc = self.reg_for_vreg_with_temp(*src, R10);
                let out = self.output_enc;
                let off = *offset as i32;
                let mem = Mem { base: out, disp: off };
                match width {
                    kajit_ir::Width::W1 => self
                        .ectx
                        .emit
                        .emit_mov_m_r8(mem, src_enc)
                        .expect("mov"),
                    kajit_ir::Width::W2 => self
                        .ectx
                        .emit
                        .emit_mov_m_r16(mem, src_enc)
                        .expect("mov"),
                    kajit_ir::Width::W4 => self
                        .ectx
                        .emit
                        .emit_mov_m_r32(mem, src_enc)
                        .expect("mov"),
                    kajit_ir::Width::W8 => self
                        .ectx
                        .emit
                        .emit_mov_m_r64(mem, src_enc)
                        .expect("mov"),
                }
            }

            LinearOp::ReadFromField { dst, offset, width } => {
                let out = self.output_enc;
                let off = *offset as i32;
                let rd = self.dst_enc_or_temp(*dst, R10);
                let mem = Mem { base: out, disp: off };
                match width {
                    kajit_ir::Width::W1 => self
                        .ectx
                        .emit
                        .emit_movzx_r32_rm8(rd, Operand::Mem(mem))
                        .expect("movzx"),
                    kajit_ir::Width::W2 => self
                        .ectx
                        .emit
                        .emit_movzx_r32_rm16(rd, Operand::Mem(mem))
                        .expect("movzx"),
                    kajit_ir::Width::W4 => self
                        .ectx
                        .emit
                        .emit_mov_r32_m(rd, mem)
                        .expect("mov"),
                    kajit_ir::Width::W8 => self
                        .ectx
                        .emit
                        .emit_mov_r64_m(rd, mem)
                        .expect("mov"),
                }
                if rd == R10 {
                    self.store_to_vreg(*dst, R10);
                }
            }

            LinearOp::SaveOutPtr { dst } => {
                let rd = self.dst_enc_or_temp(*dst, R10);
                self.ectx
                    .emit
                    .emit_mov_r64_r64(rd, self.output_enc)
                    .expect("mov");
                if rd == R10 {
                    self.store_to_vreg(*dst, R10);
                }
            }

            LinearOp::SetOutPtr { src } => {
                let src_enc = self.reg_for_vreg_with_temp(*src, R10);
                self.ectx
                    .emit
                    .emit_mov_r64_r64(self.output_enc, src_enc)
                    .expect("mov");
            }

            LinearOp::SlotAddr { dst, slot } => {
                let rd = self.dst_enc_or_temp(*dst, R10);
                let off = self.slot_off(slot.index() as u32) as i32;
                self.ectx
                    .emit
                    .emit_lea_r64_m(
                        rd,
                        Mem {
                            base: RSP,
                            disp: off,
                        },
                    )
                    .expect("lea");
                if rd == R10 {
                    self.store_to_vreg(*dst, R10);
                }
            }

            LinearOp::StoreToAddr { addr, src, width } => {
                let addr_enc = self.reg_for_vreg_with_temp(*addr, R11);
                let src_enc = self.reg_for_vreg_with_temp(*src, R10);
                let mem = Mem { base: addr_enc, disp: 0 };
                match width {
                    kajit_ir::Width::W1 => self
                        .ectx
                        .emit
                        .emit_mov_m_r8(mem, src_enc)
                        .expect("mov"),
                    kajit_ir::Width::W2 => self
                        .ectx
                        .emit
                        .emit_mov_m_r16(mem, src_enc)
                        .expect("mov"),
                    kajit_ir::Width::W4 => self
                        .ectx
                        .emit
                        .emit_mov_m_r32(mem, src_enc)
                        .expect("mov"),
                    kajit_ir::Width::W8 => self
                        .ectx
                        .emit
                        .emit_mov_m_r64(mem, src_enc)
                        .expect("mov"),
                }
            }

            LinearOp::LoadFromAddr { dst, addr, width } => {
                let (base_enc, offset) =
                    if let Some(&(base_vreg, off)) = self.fused_addr_offsets.get(addr) {
                        (self.reg_for_vreg_with_temp(base_vreg, R11), off as i32)
                    } else {
                        (self.reg_for_vreg_with_temp(*addr, R11), 0i32)
                    };
                let mut used_scratch = false;
                let assigned = self.dst_enc_or_temp(*dst, R10);
                let rd = if assigned == base_enc {
                    used_scratch = true;
                    if base_enc != R10 { R10 } else { R11 }
                } else {
                    assigned
                };
                let mem = Mem { base: base_enc, disp: offset };
                match width {
                    kajit_ir::Width::W1 => self
                        .ectx
                        .emit
                        .emit_movzx_r32_rm8(rd, Operand::Mem(mem))
                        .expect("movzx"),
                    kajit_ir::Width::W2 => self
                        .ectx
                        .emit
                        .emit_movzx_r32_rm16(rd, Operand::Mem(mem))
                        .expect("movzx"),
                    kajit_ir::Width::W4 => self
                        .ectx
                        .emit
                        .emit_mov_r32_m(rd, mem)
                        .expect("mov"),
                    kajit_ir::Width::W8 => self
                        .ectx
                        .emit
                        .emit_mov_r64_m(rd, mem)
                        .expect("mov"),
                }
                if used_scratch || rd == R10 {
                    self.store_to_vreg(*dst, rd);
                }
            }

            LinearOp::WriteToSlot { slot, src } => {
                let src_enc = self.reg_for_vreg_with_temp(*src, R10);
                let off = self.slot_off(slot.index() as u32) as i32;
                self.ectx
                    .emit
                    .emit_mov_m_r64(Mem { base: RSP, disp: off }, src_enc)
                    .expect("mov slot store");
            }

            LinearOp::ReadFromSlot { dst, slot } => {
                let rd = self.dst_enc_or_temp(*dst, R10);
                let off = self.slot_off(slot.index() as u32) as i32;
                self.ectx
                    .emit
                    .emit_mov_r64_m(rd, Mem { base: RSP, disp: off })
                    .expect("mov slot load");
                if rd == R10 {
                    self.store_to_vreg(*dst, R10);
                }
            }

            LinearOp::ErrorExit { code } => {
                self.ectx.emit_error_with_ctx(*code, self.ctx_enc);
            }

            LinearOp::CallIntrinsic {
                func,
                args,
                dst,
                field_offset,
            } => {
                self.emit_call_intrinsic(*func, args, *dst, Some(*field_offset));
            }

            LinearOp::CallPure { func, args, dst } | LinearOp::CallEffect { func, args, dst } => {
                self.emit_call_pure(*func, args, *dst);
            }

            LinearOp::CallLambda { .. } => {
                self.ectx.emit.emit_nop().expect("nop");
            }

            LinearOp::SimdStringScan { .. } | LinearOp::SimdWhitespaceSkip => {
                panic!("unsupported SIMD op in regalloc3 x86_64 backend");
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
    /// x86_64 uses 2-operand destructive form: `op dst, src` clobbers dst.
    /// Pattern: load lhs → r10, apply op with rhs → r10, store r10 → dst.
    pub fn emit_binop(
        &mut self,
        kind: BinOpKind,
        dst: kajit_ir::VReg,
        lhs: kajit_ir::VReg,
        rhs: kajit_ir::VReg,
    ) {
        // Comparisons: cmp + setcc (or fused cmp + jcc)
        if matches!(
            kind,
            BinOpKind::CmpEq
                | BinOpKind::CmpNe
                | BinOpKind::CmpLt
                | BinOpKind::CmpLe
                | BinOpKind::CmpGt
                | BinOpKind::CmpGe
        ) {
            // Try to fold: if lhs is a small const and rhs is not, swap and invert.
            let (cmp_lhs, cmp_rhs, swapped) = if self.imm32_const(rhs).is_some() {
                (lhs, rhs, false)
            } else if self.imm32_const(lhs).is_some() {
                (rhs, lhs, true)
            } else {
                (lhs, rhs, false)
            };

            let cmp_lhs_enc = self.reg_for_vreg_with_temp(cmp_lhs, R10);
            if let Some(imm) = self.imm32_const(cmp_rhs) {
                self.ectx
                    .emit
                    .emit_cmp_r64_imm32(cmp_lhs_enc, imm)
                    .expect("cmp imm");
            } else {
                let cmp_rhs_enc = self.reg_for_vreg_with_temp(cmp_rhs, R11);
                self.ectx
                    .emit
                    .emit_cmp_r64_r64(cmp_lhs_enc, cmp_rhs_enc)
                    .expect("cmp");
            }

            // Skip setcc if this comparison is fused with its branch terminator.
            if self.fused_cmps.contains_key(&dst) {
                if swapped {
                    let cc = self.fused_cmps.get_mut(&dst).unwrap();
                    *cc = swap_condition(*cc);
                }
                return;
            }

            let condition = match kind {
                BinOpKind::CmpEq => x64::Condition::Eq,
                BinOpKind::CmpNe => x64::Condition::Ne,
                BinOpKind::CmpLt => x64::Condition::Lo,
                BinOpKind::CmpLe => x64::Condition::Ls,
                BinOpKind::CmpGt => x64::Condition::Hi,
                BinOpKind::CmpGe => x64::Condition::Hs,
                _ => unreachable!(),
            };
            let condition = if swapped {
                swap_condition(condition)
            } else {
                condition
            };

            let rd = self.dst_enc_or_temp(dst, R10);
            self.ectx
                .emit
                .emit_setcc_r8(condition, rd)
                .expect("setcc");
            self.ectx
                .emit
                .emit_movzx_r64_rm8(rd, Operand::Reg(rd))
                .expect("movzx");
            if rd == R10 {
                self.store_to_vreg(dst, R10);
            }
            return;
        }

        // Shifts: x86_64 variable shifts require the shift amount in CL (rcx low byte).
        if matches!(kind, BinOpKind::Shl | BinOpKind::Shr | BinOpKind::Sar) {
            // Try immediate shift for constant shift amounts.
            if let Some(&shift_val) = self.const_values.get(&rhs) {
                if shift_val < 64 {
                    let lhs_enc = self.reg_for_vreg_with_temp(lhs, R10);
                    // For 2-operand destructive: load into result, then shift in-place.
                    let rd = self.dst_enc_or_temp(dst, R10);
                    if rd != lhs_enc {
                        self.ectx
                            .emit
                            .emit_mov_r64_r64(rd, lhs_enc)
                            .expect("mov");
                    }
                    match kind {
                        BinOpKind::Shl => self
                            .ectx
                            .emit
                            .emit_shl_r64_imm8(rd, shift_val as u8)
                            .expect("shl imm"),
                        BinOpKind::Shr => self
                            .ectx
                            .emit
                            .emit_shr_r64_imm8(rd, shift_val as u8)
                            .expect("shr imm"),
                        BinOpKind::Sar => self
                            .ectx
                            .emit
                            .emit_sar_r64_imm8(rd, shift_val as u8)
                            .expect("sar imm"),
                        _ => unreachable!(),
                    }
                    if rd == R10 {
                        self.store_to_vreg(dst, R10);
                    }
                    return;
                }
            }

            // Variable shift: move shift amount into rcx, lhs into r10, shift r10.
            let rhs_enc = self.reg_for_vreg_with_temp(rhs, R11);
            self.ectx
                .emit
                .emit_mov_r64_r64(RCX, rhs_enc)
                .expect("mov rcx");
            let lhs_enc = self.reg_for_vreg_with_temp(lhs, R10);
            if lhs_enc != R10 {
                self.ectx
                    .emit
                    .emit_mov_r64_r64(R10, lhs_enc)
                    .expect("mov r10");
            }
            match kind {
                BinOpKind::Shl => self
                    .ectx
                    .emit
                    .emit_shl_r64_cl(R10)
                    .expect("shl cl"),
                BinOpKind::Shr => self
                    .ectx
                    .emit
                    .emit_shr_r64_cl(R10)
                    .expect("shr cl"),
                BinOpKind::Sar => self
                    .ectx
                    .emit
                    .emit_sar_r64_cl(R10)
                    .expect("sar cl"),
                _ => unreachable!(),
            }
            self.store_to_vreg(dst, R10);
            return;
        }

        // Arithmetic/logic: 2-operand destructive form.
        // x86_64 encodes `op rd, rhs` which computes rd = op(rd, rhs).
        // We need: rd = op(lhs, rhs). So: mov rd, lhs; op rd, rhs.
        //
        // Hazard: if rd == rhs_enc, the mov clobbers rhs before the op reads it.
        // In that case: save rhs to R11 first.
        let lhs_enc = self.reg_for_vreg_with_temp(lhs, R10);
        let rd = self.dst_enc_or_temp(dst, R10);

        // Try immediate form for add/sub with 32-bit const (no rhs clobber risk).
        if matches!(kind, BinOpKind::Add | BinOpKind::Sub) {
            if let Some(imm) = self.imm32_const(rhs) {
                if rd != lhs_enc {
                    self.ectx
                        .emit
                        .emit_mov_r64_r64(rd, lhs_enc)
                        .expect("mov lhs→rd");
                }
                match kind {
                    BinOpKind::Add => self
                        .ectx
                        .emit
                        .emit_add_r64_imm32(rd, imm)
                        .expect("add imm"),
                    BinOpKind::Sub => self
                        .ectx
                        .emit
                        .emit_sub_r64_imm32(rd, imm)
                        .expect("sub imm"),
                    _ => unreachable!(),
                }
                if rd == R10 {
                    self.store_to_vreg(dst, R10);
                }
                return;
            }
        }

        let rhs_enc = self.reg_for_vreg_with_temp(rhs, R11);

        // Handle the clobber hazard: if rd == rhs_enc and rd != lhs_enc,
        // the mov lhs→rd would destroy rhs before the op can read it.
        if rd == rhs_enc && rd != lhs_enc {
            // Save rhs to R11, mov lhs into rd, operate with R11.
            self.ectx
                .emit
                .emit_mov_r64_r64(R11, rhs_enc)
                .expect("save rhs to temp");
            self.ectx
                .emit
                .emit_mov_r64_r64(rd, lhs_enc)
                .expect("mov lhs→rd");
            match kind {
                BinOpKind::Add => self.ectx.emit.emit_add_r64_r64(rd, R11).expect("add"),
                BinOpKind::Sub => self.ectx.emit.emit_sub_r64_r64(rd, R11).expect("sub"),
                BinOpKind::Mul => self.ectx.emit.emit_imul_r64_r64(rd, R11).expect("imul"),
                BinOpKind::And => self.ectx.emit.emit_and_r64_r64(rd, R11).expect("and"),
                BinOpKind::Or => self.ectx.emit.emit_or_r64_r64(rd, R11).expect("or"),
                BinOpKind::Xor => self.ectx.emit.emit_xor_r64_r64(rd, R11).expect("xor"),
                _ => unreachable!("comparison/shift ops handled above"),
            }
        } else {
            // Normal case: mov lhs into rd (if needed), then operate with rhs.
            if rd != lhs_enc {
                self.ectx
                    .emit
                    .emit_mov_r64_r64(rd, lhs_enc)
                    .expect("mov lhs→rd");
            }
            match kind {
                BinOpKind::Add => self.ectx.emit.emit_add_r64_r64(rd, rhs_enc).expect("add"),
                BinOpKind::Sub => self.ectx.emit.emit_sub_r64_r64(rd, rhs_enc).expect("sub"),
                BinOpKind::Mul => self.ectx.emit.emit_imul_r64_r64(rd, rhs_enc).expect("imul"),
                BinOpKind::And => self.ectx.emit.emit_and_r64_r64(rd, rhs_enc).expect("and"),
                BinOpKind::Or => self.ectx.emit.emit_or_r64_r64(rd, rhs_enc).expect("or"),
                BinOpKind::Xor => self.ectx.emit.emit_xor_r64_r64(rd, rhs_enc).expect("xor"),
                _ => unreachable!("comparison/shift ops handled above"),
            }
        }

        if rd == R10 {
            self.store_to_vreg(dst, R10);
        }
    }

    /// Emit a unary operation.
    pub fn emit_unary(&mut self, kind: UnaryOpKind, dst: kajit_ir::VReg, src: kajit_ir::VReg) {
        let src_enc = self.reg_for_vreg_with_temp(src, R10);
        match kind {
            UnaryOpKind::ZigzagDecode { wide: true } => {
                // zigzag_decode(n) = (n >> 1) ^ -(n & 1)
                if src_enc != R10 {
                    self.ectx.emit.emit_mov_r64_r64(R10, src_enc).expect("mov");
                }
                // r11 = n >> 1
                self.ectx.emit.emit_mov_r64_r64(R11, R10).expect("mov");
                self.ectx.emit.emit_shr_r64_imm8(R11, 1).expect("shr");
                // shl r10, 63; sar r10, 63 gives sign-extend of bit 0
                // which is exactly -(n & 1)!
                self.ectx.emit.emit_shl_r64_imm8(R10, 63).expect("shl");
                self.ectx.emit.emit_sar_r64_imm8(R10, 63).expect("sar");
                // r10 = -(n & 1), r11 = n >> 1
                self.ectx.emit.emit_xor_r64_r64(R10, R11).expect("xor");
                self.store_to_vreg(dst, R10);
            }
            UnaryOpKind::ZigzagDecode { wide: false } => {
                if src_enc != R10 {
                    self.ectx.emit.emit_mov_r64_r64(R10, src_enc).expect("mov");
                }
                self.ectx.emit.emit_mov_r32_r32(R10, R10).expect("mov32"); // zero-extend to 32
                // r11 = n >> 1
                self.ectx.emit.emit_mov_r64_r64(R11, R10).expect("mov");
                self.ectx.emit.emit_shr_r64_imm8(R11, 1).expect("shr");
                // r10 = -(n & 1) via shl 63; sar 63
                self.ectx.emit.emit_shl_r64_imm8(R10, 63).expect("shl");
                self.ectx.emit.emit_sar_r64_imm8(R10, 63).expect("sar");
                self.ectx.emit.emit_xor_r64_r64(R10, R11).expect("xor");
                self.ectx.emit.emit_mov_r32_r32(R10, R10).expect("mov32"); // truncate to 32
                self.store_to_vreg(dst, R10);
            }
            UnaryOpKind::SignExtend { from_width } => {
                if src_enc != R10 {
                    self.ectx.emit.emit_mov_r64_r64(R10, src_enc).expect("mov");
                }
                match from_width {
                    kajit_ir::Width::W1 => self
                        .ectx
                        .emit
                        .emit_movsx_r64_rm8(R10, Operand::Reg(R10))
                        .expect("movsx"),
                    kajit_ir::Width::W2 => self
                        .ectx
                        .emit
                        .emit_movsx_r64_rm16(R10, Operand::Reg(R10))
                        .expect("movsx"),
                    kajit_ir::Width::W4 => self
                        .ectx
                        .emit
                        .emit_movsxd_r64_rm32(R10, Operand::Reg(R10))
                        .expect("movsxd"),
                    kajit_ir::Width::W8 => {} // no-op
                }
                self.store_to_vreg(dst, R10);
            }
        }
    }
}

/// Swap a comparison condition as if operands were swapped.
/// e.g. CmpLt(a,b) with swapped operands → CmpGt(b,a)
fn swap_condition(cc: x64::Condition) -> x64::Condition {
    match cc {
        x64::Condition::Eq => x64::Condition::Eq,
        x64::Condition::Ne => x64::Condition::Ne,
        x64::Condition::Lo => x64::Condition::Hi,
        x64::Condition::Hi => x64::Condition::Lo,
        x64::Condition::Ls => x64::Condition::Hs,
        x64::Condition::Hs => x64::Condition::Ls,
        x64::Condition::Lt => x64::Condition::Gt,
        x64::Condition::Gt => x64::Condition::Lt,
        x64::Condition::Le => x64::Condition::Ge,
        x64::Condition::Ge => x64::Condition::Le,
        other => other,
    }
}

/// Invert a condition (true ↔ false).
pub(super) fn invert_condition(cc: x64::Condition) -> x64::Condition {
    match cc {
        x64::Condition::Eq => x64::Condition::Ne,
        x64::Condition::Ne => x64::Condition::Eq,
        x64::Condition::Lo => x64::Condition::Hs,
        x64::Condition::Hs => x64::Condition::Lo,
        x64::Condition::Hi => x64::Condition::Ls,
        x64::Condition::Ls => x64::Condition::Hi,
        x64::Condition::Lt => x64::Condition::Ge,
        x64::Condition::Ge => x64::Condition::Lt,
        x64::Condition::Gt => x64::Condition::Le,
        x64::Condition::Le => x64::Condition::Gt,
        x64::Condition::O => x64::Condition::No,
        x64::Condition::No => x64::Condition::O,
        x64::Condition::S => x64::Condition::Ns,
        x64::Condition::Ns => x64::Condition::S,
        x64::Condition::P => x64::Condition::Np,
        x64::Condition::Np => x64::Condition::P,
    }
}
