//! Call emission for aarch64 regalloc3 backend.

use kajit_emit::aarch64::{Reg, Width};

use crate::context::CTX_ERROR_CODE;
use crate::ir_backend::IntrinsicCallSiteInfo;

use super::context::EmitContext;

impl<'a> EmitContext<'a> {
    /// Emit a call to an intrinsic function.
    /// ABI: x0=ctx, x1+=args, with out_field offset adjustment on x21.
    pub(super) fn emit_call_intrinsic(
        &mut self,
        func: kajit_ir::IntrinsicFn,
        args: &[kajit_ir::VReg],
        dst: Option<kajit_ir::VReg>,
        field_offset: Option<u32>,
    ) {
        let error_exit = self.ectx.error_exit;

        // Adjust out_ptr for field offset if needed
        if let Some(off) = field_offset
            && off > 0
        {
            self.ectx
                .emit
                .emit_add_imm(
                    Width::X64,
                    self.output_reg,
                    self.output_reg,
                    off as u16,
                    false,
                )
                .expect("add out_ptr");
        }

        // Move register-resident args with parallel-copy semantics first, then
        // materialize spilled/rematerialized args into their ABI homes.
        let mut reg_moves = Vec::new();
        let mut deferred_args = Vec::new();
        for (i, &arg) in args.iter().enumerate() {
            let target_reg = Reg::from_raw((i + 1) as u8);
            if let Some(preg) = self.preg_for_vreg(arg) {
                let src_reg = self.preg_to_reg(preg);
                if src_reg != target_reg {
                    reg_moves.push((target_reg, src_reg));
                }
            } else {
                deferred_args.push((arg, target_reg));
            }
        }
        if !reg_moves.is_empty() {
            self.emit_parallel_moves(&reg_moves, Reg::X16);
        }
        for (arg, target_reg) in deferred_args {
            let src_reg = self.reg_for_vreg_with_temp(arg, target_reg);
            if src_reg != target_reg {
                self.ectx
                    .emit
                    .emit_mov_reg(Width::X64, target_reg, src_reg)
                    .expect("mov arg");
            }
        }

        // If no dst but field_offset, pass output_reg (already adjusted) as out_field arg
        if dst.is_none() && field_offset.is_some() {
            let arg_idx = args.len() + 1;
            self.ectx
                .emit
                .emit_mov_reg(Width::X64, Reg::from_raw(arg_idx as u8), self.output_reg)
                .expect("mov out_field");
        }

        // x0 = ctx
        self.ectx
            .emit
            .emit_mov_reg(Width::X64, Reg::X0, self.ctx_reg)
            .expect("mov ctx");

        // Load function pointer and call
        let call_site_offset = self.ectx.emit.code_len();
        self.emit_load_u64(Reg::X16, func.0 as u64);
        self.ectx.emit.emit_blr(Reg::X16).expect("blr");
        self.intrinsic_call_sites.push(IntrinsicCallSiteInfo {
            code_offset: call_site_offset,
            func,
        });

        // Restore out_ptr
        if let Some(off) = field_offset
            && off > 0
        {
            self.ectx
                .emit
                .emit_sub_imm(
                    Width::X64,
                    self.output_reg,
                    self.output_reg,
                    off as u16,
                    false,
                )
                .expect("sub out_ptr");
        }

        // Check error after call
        self.ectx
            .emit
            .emit_ldr_imm(Width::W32, Reg::X16, self.ctx_reg, CTX_ERROR_CODE)
            .expect("ldr error");
        self.ectx
            .emit
            .emit_cbnz_label(Width::W32, Reg::X16, error_exit)
            .expect("cbnz error");

        // Store result if needed (return value is in x0)
        if let Some(dst) = dst {
            self.store_to_vreg(dst, Reg::X0);
        }
    }

    /// Emit a call to a pure/effect function (no ctx, no cursor flush).
    /// The RA has colored args, and edits move them to ABI registers.
    pub(super) fn emit_call_pure(
        &mut self,
        func: kajit_ir::IntrinsicFn,
        args: &[kajit_ir::VReg],
        dst: kajit_ir::VReg,
    ) {
        // In-register args are already in their ABI registers thanks to RA
        // coloring + OperandEdit moves emitted before this instruction.
        //
        // Spilled/rematerializable args need explicit materialization here:
        // the native regalloc3 path has no separate spill/reload pass.
        for (i, &arg) in args.iter().enumerate() {
            if self.preg_for_vreg(arg).is_none() {
                let abi_reg = Reg::from_raw(i as u8);
                let _ = self.reg_for_vreg_with_temp(arg, abi_reg);
            }
        }

        let call_site_offset = self.ectx.emit.code_len();
        self.emit_load_u64(Reg::X16, func.0 as u64);
        self.ectx.emit.emit_blr(Reg::X16).expect("blr");
        self.intrinsic_call_sites.push(IntrinsicCallSiteInfo {
            code_offset: call_site_offset,
            func,
        });

        self.store_to_vreg(dst, Reg::X0);
    }
}
