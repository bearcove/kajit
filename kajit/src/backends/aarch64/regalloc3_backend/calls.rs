//! Call emission for aarch64 regalloc3 backend.

use kajit_emit::aarch64::{Reg, Width};

use crate::context::CTX_ERROR_CODE;
use crate::ir_backend::IntrinsicCallSiteInfo;

use super::context::EmitContext;

impl<'a> EmitContext<'a> {
    /// Emit a call to an intrinsic function.
    /// Args (including ctx_ptr) are already in ABI registers via RA coloring.
    pub(super) fn emit_call_intrinsic(
        &mut self,
        func: kajit_ir::IntrinsicFn,
        args: &[kajit_ir::VReg],
        dst: Option<kajit_ir::VReg>,
    ) {
        // Materialize any spilled args into their ABI registers.
        for (i, &arg) in args.iter().enumerate() {
            if self.preg_for_vreg(arg).is_none() {
                let abi_reg = Reg::from_raw(i as u8);
                let _ = self.reg_for_vreg_with_temp(arg, abi_reg);
            }
        }

        // Load function pointer and call
        let call_site_offset = self.ectx.emit.code_len();
        self.emit_load_u64(Reg::X16, func.0 as u64);
        self.ectx.emit.emit_blr(Reg::X16).expect("blr");
        self.intrinsic_call_sites.push(IntrinsicCallSiteInfo {
            code_offset: call_site_offset,
            func,
        });

        // Check error after call (TODO: will become explicit IR in 011-3)
        let error_exit = self.ectx.error_exit;
        // ctx_ptr is args[0], which is in x0. But x0 is also the return value
        // register, so after the call x0 holds the return value. We need to
        // read error_code from ctx — but ctx was in x0 before the call and
        // may have been clobbered. We need the ctx vreg's location.
        // For now, reload ctx from args[0]'s vreg allocation.
        let ctx_vreg = args[0];
        let ctx_reg = self.reg_for_vreg_with_temp(ctx_vreg, Reg::X16);
        self.ectx
            .emit
            .emit_ldr_imm(Width::W32, Reg::X16, ctx_reg, CTX_ERROR_CODE)
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
