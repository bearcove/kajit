//! Call emission for aarch64 regalloc3 backend.

use kajit_emit::aarch64::Reg;

use crate::ir_backend::IntrinsicCallSiteInfo;

use super::context::EmitContext;

impl<'a> EmitContext<'a> {
    /// Emit a call to an intrinsic function.
    /// Args (including ctx_ptr) are already in ABI registers via RA coloring.
    pub(super) fn emit_call_intrinsic(
        &mut self,
        func: kajit_ir::FnPtr,
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

        // Store result if needed (return value is in x0)
        if let Some(dst) = dst {
            self.store_to_vreg(dst, Reg::X0);
        }
    }

    /// Emit a call to a pure/effect function (no ctx, no cursor flush).
    /// The RA has colored args, and edits move them to ABI registers.
    pub(super) fn emit_call_pure(
        &mut self,
        func: kajit_ir::FnPtr,
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
