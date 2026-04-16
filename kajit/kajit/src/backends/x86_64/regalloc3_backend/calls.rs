//! Call emission for x86_64 regalloc3 backend.

use kajit_asm::x64;

use super::context::EmitContext;
use crate::ir_backend::IntrinsicCallSiteInfo;

const R10: u8 = 10;
const RAX: u8 = 0;

impl<'a> EmitContext<'a> {
    /// Emit a call to an intrinsic function.
    /// Args (including ctx_ptr as first arg) are placed into ABI registers by RA.
    pub fn emit_call_intrinsic(
        &mut self,
        func: kajit_ir::FnPtr,
        args: &[kajit_ir::VReg],
        dst: Option<kajit_ir::VReg>,
    ) {
        // SysV: rdi, rsi, rdx, rcx, r8, r9
        // Win64: rcx, rdx, r8, r9
        #[cfg(not(windows))]
        let abi_regs: &[u8] = &[7, 6, 2, 1, 8, 9];
        #[cfg(windows)]
        let abi_regs: &[u8] = &[1, 2, 8, 9];

        assert!(
            args.len() <= abi_regs.len(),
            "unsupported CallIntrinsic arity: {} args",
            args.len()
        );

        // Push all args, then pop into ABI registers.
        for &arg in args.iter() {
            let enc = self.reg_for_vreg_with_temp(arg, R10);
            self.ectx
                .emit
                .emit_with(|buf| x64::encode_push_r64(enc, buf))
                .expect("push arg");
        }
        for i in (0..args.len()).rev() {
            let enc = abi_regs[i];
            self.ectx
                .emit
                .emit_with(|buf| x64::encode_pop_r64(enc, buf))
                .expect("pop arg");
        }

        // Load function pointer and call.
        let call_site_offset = self.ectx.emit.code_len();
        let fn_ptr = func.0 as u64;
        self.ectx
            .emit
            .emit_with(|buf| {
                x64::encode_mov_r64_imm64(RAX, fn_ptr, buf)?;
                x64::encode_call_r64(RAX, buf)
            })
            .expect("call intrinsic");
        self.intrinsic_call_sites.push(IntrinsicCallSiteInfo {
            code_offset: call_site_offset,
            func,
        });

        // Store result if needed (return value is in rax).
        if let Some(dst) = dst {
            self.store_to_vreg(dst, RAX);
        }
    }

    /// Emit a call to a pure/effect function (no ctx, no cursor flush).
    pub fn emit_call_pure(
        &mut self,
        func: kajit_ir::FnPtr,
        args: &[kajit_ir::VReg],
        dst: kajit_ir::VReg,
    ) {
        // SysV: rdi, rsi, rdx, rcx, r8, r9
        // Win64: rcx, rdx, r8, r9
        #[cfg(not(windows))]
        let abi_regs: &[u8] = &[7, 6, 2, 1, 8, 9];
        #[cfg(windows)]
        let abi_regs: &[u8] = &[1, 2, 8, 9];

        assert!(
            args.len() <= abi_regs.len(),
            "unsupported CallPure arity: {} args",
            args.len()
        );

        // Push all args, then pop into ABI registers.
        for &arg in args.iter() {
            let enc = self.reg_for_vreg_with_temp(arg, R10);
            self.ectx
                .emit
                .emit_with(|buf| x64::encode_push_r64(enc, buf))
                .expect("push");
        }
        for i in (0..args.len()).rev() {
            let enc = abi_regs[i];
            self.ectx
                .emit
                .emit_with(|buf| x64::encode_pop_r64(enc, buf))
                .expect("pop");
        }

        let call_site_offset = self.ectx.emit.code_len();
        let fn_ptr = func.0 as u64;
        self.ectx
            .emit
            .emit_with(|buf| {
                x64::encode_mov_r64_imm64(RAX, fn_ptr, buf)?;
                x64::encode_call_r64(RAX, buf)
            })
            .expect("call pure");
        self.intrinsic_call_sites.push(IntrinsicCallSiteInfo {
            code_offset: call_site_offset,
            func,
        });

        self.store_to_vreg(dst, RAX);
    }
}
