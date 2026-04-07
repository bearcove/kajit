//! Instruction emission for x86_64 regalloc3 backend.

use kajit_emit::x64::{self, Mem, Operand};
use kajit_lir::{BinOpKind, LinearOp, UnaryOpKind};

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

            LinearOp::WriteToField { src, offset, width } => {
                let src_enc = self.reg_for_vreg_with_temp(*src, R10);
                let out = self.output_enc;
                let off = *offset as i32;
                match width {
                    kajit_ir::Width::W1 => self
                        .ectx
                        .emit
                        .emit_with(|buf| x64::encode_mov_m_r8(out, off, src_enc, buf))
                        .expect("mov"),
                    kajit_ir::Width::W2 => self
                        .ectx
                        .emit
                        .emit_with(|buf| x64::encode_mov_m_r16(out, off, src_enc, buf))
                        .expect("mov"),
                    kajit_ir::Width::W4 => self
                        .ectx
                        .emit
                        .emit_with(|buf| {
                            x64::encode_mov_m_r32(
                                Mem {
                                    base: out,
                                    disp: off,
                                },
                                src_enc,
                                buf,
                            )
                        })
                        .expect("mov"),
                    kajit_ir::Width::W8 => self
                        .ectx
                        .emit
                        .emit_with(|buf| {
                            x64::encode_mov_m_r64(
                                Mem {
                                    base: out,
                                    disp: off,
                                },
                                src_enc,
                                buf,
                            )
                        })
                        .expect("mov"),
                }
            }

            LinearOp::ReadFromField { dst, offset, width } => {
                let out = self.output_enc;
                let off = *offset as i32;
                let rd = self.dst_enc_or_temp(*dst, R10);
                match width {
                    kajit_ir::Width::W1 => self
                        .ectx
                        .emit
                        .emit_with(|buf| {
                            x64::encode_movzx_r32_rm8(
                                rd,
                                Operand::Mem(Mem {
                                    base: out,
                                    disp: off,
                                }),
                                buf,
                            )
                        })
                        .expect("movzx"),
                    kajit_ir::Width::W2 => self
                        .ectx
                        .emit
                        .emit_with(|buf| {
                            x64::encode_movzx_r32_rm16(
                                rd,
                                Operand::Mem(Mem {
                                    base: out,
                                    disp: off,
                                }),
                                buf,
                            )
                        })
                        .expect("movzx"),
                    kajit_ir::Width::W4 => self
                        .ectx
                        .emit
                        .emit_with(|buf| {
                            x64::encode_mov_r32_m(
                                rd,
                                Mem {
                                    base: out,
                                    disp: off,
                                },
                                buf,
                            )
                        })
                        .expect("mov"),
                    kajit_ir::Width::W8 => self
                        .ectx
                        .emit
                        .emit_with(|buf| {
                            x64::encode_mov_r64_m(
                                rd,
                                Mem {
                                    base: out,
                                    disp: off,
                                },
                                buf,
                            )
                        })
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
                    .emit_with(|buf| x64::encode_mov_r64_r64(rd, self.output_enc, buf))
                    .expect("mov");
                if rd == R10 {
                    self.store_to_vreg(*dst, R10);
                }
            }

            LinearOp::SetOutPtr { src } => {
                let src_enc = self.reg_for_vreg_with_temp(*src, R10);
                self.ectx
                    .emit
                    .emit_with(|buf| x64::encode_mov_r64_r64(self.output_enc, src_enc, buf))
                    .expect("mov");
            }

            LinearOp::SlotAddr { dst, slot } => {
                let rd = self.dst_enc_or_temp(*dst, R10);
                let off = self.slot_off(slot.index() as u32) as i32;
                self.ectx
                    .emit
                    .emit_with(|buf| {
                        x64::encode_lea_r64_m(
                            rd,
                            Mem {
                                base: RSP,
                                disp: off,
                            },
                            buf,
                        )
                    })
                    .expect("lea");
                if rd == R10 {
                    self.store_to_vreg(*dst, R10);
                }
            }

            LinearOp::StoreToAddr { addr, src, width } => {
                let addr_enc = self.reg_for_vreg_with_temp(*addr, R11);
                let src_enc = self.reg_for_vreg_with_temp(*src, R10);
                match width {
                    kajit_ir::Width::W1 => self
                        .ectx
                        .emit
                        .emit_with(|buf| x64::encode_mov_m_r8(addr_enc, 0, src_enc, buf))
                        .expect("mov"),
                    kajit_ir::Width::W2 => self
                        .ectx
                        .emit
                        .emit_with(|buf| x64::encode_mov_m_r16(addr_enc, 0, src_enc, buf))
                        .expect("mov"),
                    kajit_ir::Width::W4 => self
                        .ectx
                        .emit
                        .emit_with(|buf| {
                            x64::encode_mov_m_r32(
                                Mem {
                                    base: addr_enc,
                                    disp: 0,
                                },
                                src_enc,
                                buf,
                            )
                        })
                        .expect("mov"),
                    kajit_ir::Width::W8 => self
                        .ectx
                        .emit
                        .emit_with(|buf| {
                            x64::encode_mov_m_r64(
                                Mem {
                                    base: addr_enc,
                                    disp: 0,
                                },
                                src_enc,
                                buf,
                            )
                        })
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
                match width {
                    kajit_ir::Width::W1 => self
                        .ectx
                        .emit
                        .emit_with(|buf| {
                            x64::encode_movzx_r32_rm8(
                                rd,
                                Operand::Mem(Mem {
                                    base: base_enc,
                                    disp: offset,
                                }),
                                buf,
                            )
                        })
                        .expect("movzx"),
                    kajit_ir::Width::W2 => self
                        .ectx
                        .emit
                        .emit_with(|buf| {
                            x64::encode_movzx_r32_rm16(
                                rd,
                                Operand::Mem(Mem {
                                    base: base_enc,
                                    disp: offset,
                                }),
                                buf,
                            )
                        })
                        .expect("movzx"),
                    kajit_ir::Width::W4 => self
                        .ectx
                        .emit
                        .emit_with(|buf| {
                            x64::encode_mov_r32_m(
                                rd,
                                Mem {
                                    base: base_enc,
                                    disp: offset,
                                },
                                buf,
                            )
                        })
                        .expect("mov"),
                    kajit_ir::Width::W8 => self
                        .ectx
                        .emit
                        .emit_with(|buf| {
                            x64::encode_mov_r64_m(
                                rd,
                                Mem {
                                    base: base_enc,
                                    disp: offset,
                                },
                                buf,
                            )
                        })
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
                    .emit_with(|buf| {
                        x64::encode_mov_m_r64(
                            Mem {
                                base: RSP,
                                disp: off,
                            },
                            src_enc,
                            buf,
                        )
                    })
                    .expect("mov slot store");
            }

            LinearOp::ReadFromSlot { dst, slot } => {
                let rd = self.dst_enc_or_temp(*dst, R10);
                let off = self.slot_off(slot.index() as u32) as i32;
                self.ectx
                    .emit
                    .emit_with(|buf| {
                        x64::encode_mov_r64_m(
                            rd,
                            Mem {
                                base: RSP,
                                disp: off,
                            },
                            buf,
                        )
                    })
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
                self.ectx
                    .emit
                    .emit_with(|buf| x64::encode_nop(buf))
                    .expect("nop");
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
                    .emit_with(|buf| x64::encode_cmp_r64_imm32(cmp_lhs_enc, imm, buf))
                    .expect("cmp imm");
            } else {
                let cmp_rhs_enc = self.reg_for_vreg_with_temp(cmp_rhs, R11);
                self.ectx
                    .emit
                    .emit_with(|buf| x64::encode_cmp_r64_r64(cmp_lhs_enc, cmp_rhs_enc, buf))
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
                .emit_with(|buf| {
                    x64::encode_setcc_r8(condition, rd, buf)?;
                    x64::encode_movzx_r64_rm8(rd, Operand::Reg(rd), buf)
                })
                .expect("setcc+movzx");
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
                            .emit_with(|buf| x64::encode_mov_r64_r64(rd, lhs_enc, buf))
                            .expect("mov");
                    }
                    match kind {
                        BinOpKind::Shl => self
                            .ectx
                            .emit
                            .emit_with(|buf| x64::encode_shl_r64_imm8(rd, shift_val as u8, buf))
                            .expect("shl imm"),
                        BinOpKind::Shr => self
                            .ectx
                            .emit
                            .emit_with(|buf| x64::encode_shr_r64_imm8(rd, shift_val as u8, buf))
                            .expect("shr imm"),
                        BinOpKind::Sar => self
                            .ectx
                            .emit
                            .emit_with(|buf| x64::encode_sar_r64_imm8(rd, shift_val as u8, buf))
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
                .emit_with(|buf| x64::encode_mov_r64_r64(RCX, rhs_enc, buf))
                .expect("mov rcx");
            let lhs_enc = self.reg_for_vreg_with_temp(lhs, R10);
            if lhs_enc != R10 {
                self.ectx
                    .emit
                    .emit_with(|buf| x64::encode_mov_r64_r64(R10, lhs_enc, buf))
                    .expect("mov r10");
            }
            match kind {
                BinOpKind::Shl => self
                    .ectx
                    .emit
                    .emit_with(|buf| x64::encode_shl_r64_cl(R10, buf))
                    .expect("shl cl"),
                BinOpKind::Shr => self
                    .ectx
                    .emit
                    .emit_with(|buf| x64::encode_shr_r64_cl(R10, buf))
                    .expect("shr cl"),
                BinOpKind::Sar => self
                    .ectx
                    .emit
                    .emit_with(|buf| x64::encode_sar_r64_cl(R10, buf))
                    .expect("sar cl"),
                _ => unreachable!(),
            }
            self.store_to_vreg(dst, R10);
            return;
        }

        // Arithmetic/logic: 2-operand destructive form.
        // Load lhs into result register, apply op with rhs.
        let lhs_enc = self.reg_for_vreg_with_temp(lhs, R10);
        let rd = self.dst_enc_or_temp(dst, R10);

        // Move lhs into rd if needed (destructive 2-operand).
        if rd != lhs_enc {
            self.ectx
                .emit
                .emit_with(|buf| x64::encode_mov_r64_r64(rd, lhs_enc, buf))
                .expect("mov lhs→rd");
        }

        // Try immediate form for add/sub with 32-bit const.
        if matches!(kind, BinOpKind::Add | BinOpKind::Sub) {
            if let Some(imm) = self.imm32_const(rhs) {
                match kind {
                    BinOpKind::Add => self
                        .ectx
                        .emit
                        .emit_with(|buf| x64::encode_add_r64_imm32(rd, imm, buf))
                        .expect("add imm"),
                    BinOpKind::Sub => self
                        .ectx
                        .emit
                        .emit_with(|buf| x64::encode_sub_r64_imm32(rd, imm, buf))
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

        match kind {
            BinOpKind::Add => self
                .ectx
                .emit
                .emit_with(|buf| x64::encode_add_r64_r64(rd, rhs_enc, buf))
                .expect("add"),
            BinOpKind::Sub => self
                .ectx
                .emit
                .emit_with(|buf| x64::encode_sub_r64_r64(rd, rhs_enc, buf))
                .expect("sub"),
            BinOpKind::Mul => self
                .ectx
                .emit
                .emit_with(|buf| x64::encode_imul_r64_r64(rd, rhs_enc, buf))
                .expect("imul"),
            BinOpKind::And => self
                .ectx
                .emit
                .emit_with(|buf| x64::encode_and_r64_r64(rd, rhs_enc, buf))
                .expect("and"),
            BinOpKind::Or => self
                .ectx
                .emit
                .emit_with(|buf| x64::encode_or_r64_r64(rd, rhs_enc, buf))
                .expect("or"),
            BinOpKind::Xor => self
                .ectx
                .emit
                .emit_with(|buf| x64::encode_xor_r64_r64(rd, rhs_enc, buf))
                .expect("xor"),
            _ => unreachable!("comparison/shift ops handled above"),
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
                    self.ectx
                        .emit
                        .emit_with(|buf| x64::encode_mov_r64_r64(R10, src_enc, buf))
                        .expect("mov");
                }
                self.ectx
                    .emit
                    .emit_with(|buf| {
                        // r11 = n >> 1
                        x64::encode_mov_r64_r64(R11, R10, buf)?;
                        x64::encode_shr_r64_imm8(R11, 1, buf)?;
                        // r10 = n & 1 (use mov_r64_imm32_sext into r10 won't work since r10 has n)
                        // Instead: and r10 with immediate 1 via test+set? No.
                        // Simplest: mov scratch2 to hold 1, but we only have r10/r11.
                        // Use: and r10, r10 keeping only bit 0 by shifting.
                        // Actually simplest: shl r10, 63; sar r10, 63 gives sign-extend of bit 0
                        // which is exactly -(n & 1)!
                        x64::encode_shl_r64_imm8(R10, 63, buf)?;
                        x64::encode_sar_r64_imm8(R10, 63, buf)?;
                        // r10 = -(n & 1), r11 = n >> 1
                        x64::encode_xor_r64_r64(R10, R11, buf)
                    })
                    .expect("zigzag wide");
                self.store_to_vreg(dst, R10);
            }
            UnaryOpKind::ZigzagDecode { wide: false } => {
                if src_enc != R10 {
                    self.ectx
                        .emit
                        .emit_with(|buf| x64::encode_mov_r64_r64(R10, src_enc, buf))
                        .expect("mov");
                }
                self.ectx
                    .emit
                    .emit_with(|buf| {
                        x64::encode_mov_r32_r32(R10, R10, buf)?; // zero-extend to 32
                        // r11 = n >> 1
                        x64::encode_mov_r64_r64(R11, R10, buf)?;
                        x64::encode_shr_r64_imm8(R11, 1, buf)?;
                        // r10 = -(n & 1) via shl 63; sar 63
                        x64::encode_shl_r64_imm8(R10, 63, buf)?;
                        x64::encode_sar_r64_imm8(R10, 63, buf)?;
                        x64::encode_xor_r64_r64(R10, R11, buf)?;
                        x64::encode_mov_r32_r32(R10, R10, buf) // truncate to 32
                    })
                    .expect("zigzag narrow");
                self.store_to_vreg(dst, R10);
            }
            UnaryOpKind::SignExtend { from_width } => {
                if src_enc != R10 {
                    self.ectx
                        .emit
                        .emit_with(|buf| x64::encode_mov_r64_r64(R10, src_enc, buf))
                        .expect("mov");
                }
                match from_width {
                    kajit_ir::Width::W1 => self
                        .ectx
                        .emit
                        .emit_with(|buf| x64::encode_movsx_r64_rm8(R10, Operand::Reg(R10), buf))
                        .expect("movsx"),
                    kajit_ir::Width::W2 => self
                        .ectx
                        .emit
                        .emit_with(|buf| x64::encode_movsx_r64_rm16(R10, Operand::Reg(R10), buf))
                        .expect("movsx"),
                    kajit_ir::Width::W4 => self
                        .ectx
                        .emit
                        .emit_with(|buf| x64::encode_movsxd_r64_rm32(R10, Operand::Reg(R10), buf))
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
