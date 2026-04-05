//! Call emission for x86_64 regalloc3 backend.

use kajit_emit::x64::{self, Mem};

use crate::context::{CTX_ERROR_CODE, CTX_INPUT_PTR};

use super::IntrinsicCallSiteInfo;
use super::context::EmitContext;

const R10: u8 = 10;
const RAX: u8 = 0;

impl<'a> EmitContext<'a> {
    /// Emit a call to an intrinsic function.
    /// SysV ABI: rdi=ctx, rsi..r9=args. Win64: rcx=ctx, rdx..r9=args.
    pub fn emit_call_intrinsic(
        &mut self,
        func: kajit_ir::IntrinsicFn,
        args: &[kajit_ir::VReg],
        dst: Option<kajit_ir::VReg>,
        field_offset: Option<u32>,
    ) {
        let error_exit = self.ectx.error_exit;
        let ctx_enc = self.ctx_enc;
        let output_enc = self.output_enc;

        if self.sync_ctx_cursor_around_calls {
            let cursor_enc = self.cursor_writeback_enc;
            self.ectx
                .emit
                .emit_with(|buf| {
                    x64::encode_mov_m_r64(
                        Mem {
                            base: ctx_enc,
                            disp: CTX_INPUT_PTR as i32,
                        },
                        cursor_enc,
                        buf,
                    )
                })
                .expect("sync cursor to ctx");
        }

        // Adjust out_ptr for field offset if needed.
        if let Some(off) = field_offset {
            if off > 0 {
                self.ectx
                    .emit
                    .emit_with(|buf| x64::encode_add_r64_imm32(output_enc, off, buf))
                    .expect("add out_ptr offset");
            }
        }

        // Push args onto the stack, then pop into ABI registers.
        // This avoids clobbering ABI registers that may contain vreg values.
        #[cfg(not(windows))]
        let abi_data_regs: &[u8] = &[6, 2, 1, 8, 9]; // rsi, rdx, rcx, r8, r9
        #[cfg(windows)]
        let abi_data_regs: &[u8] = &[2, 8, 9]; // rdx, r8, r9

        assert!(
            args.len() <= abi_data_regs.len(),
            "unsupported CallIntrinsic arity: {} args (+ctx)",
            args.len()
        );

        for &arg in args.iter() {
            let enc = self.reg_for_vreg_with_temp(arg, R10);
            self.ectx
                .emit
                .emit_with(|buf| x64::encode_push_r64(enc, buf))
                .expect("push arg");
        }

        // ctx → first ABI register
        #[cfg(not(windows))]
        self.ectx
            .emit
            .emit_with(|buf| x64::encode_mov_r64_r64(7, ctx_enc, buf)) // rdi = ctx
            .expect("mov ctx→rdi");
        #[cfg(windows)]
        self.ectx
            .emit
            .emit_with(|buf| x64::encode_mov_r64_r64(1, ctx_enc, buf)) // rcx = ctx
            .expect("mov ctx→rcx");

        // If no dst but field_offset, pass output_reg as out_field arg (after data args).
        if dst.is_none() {
            if field_offset.is_some() {
                let out_arg_enc = abi_data_regs[args.len()];
                self.ectx
                    .emit
                    .emit_with(|buf| x64::encode_mov_r64_r64(out_arg_enc, output_enc, buf))
                    .expect("mov out_field arg");
            }
        }

        // Pop args from stack into ABI registers (reverse order).
        for i in (0..args.len()).rev() {
            let enc = abi_data_regs[i];
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

        // Restore out_ptr.
        if let Some(off) = field_offset {
            if off > 0 {
                self.ectx
                    .emit
                    .emit_with(|buf| x64::encode_sub_r64_imm32(output_enc, off, buf))
                    .expect("sub out_ptr offset");
            }
        }

        if self.sync_ctx_cursor_around_calls {
            let cursor_enc = self.cursor_writeback_enc;
            self.ectx
                .emit
                .emit_with(|buf| {
                    x64::encode_mov_r64_m(
                        cursor_enc,
                        Mem {
                            base: ctx_enc,
                            disp: CTX_INPUT_PTR as i32,
                        },
                        buf,
                    )
                })
                .expect("reload cursor from ctx");
        }

        // Check error after call.
        self.ectx
            .emit
            .emit_with(|buf| {
                x64::encode_mov_r32_m(
                    R10,
                    Mem {
                        base: ctx_enc,
                        disp: CTX_ERROR_CODE as i32,
                    },
                    buf,
                )?;
                x64::encode_test_r32_r32(R10, R10, buf)
            })
            .expect("check error");
        self.ectx
            .emit
            .emit_jnz_label(error_exit)
            .expect("jnz error");

        // Store result if needed (return value is in rax).
        if let Some(dst) = dst {
            self.store_to_vreg(dst, RAX);
        }
    }

    /// Emit a call to a pure/effect function (no ctx, no cursor flush).
    pub fn emit_call_pure(
        &mut self,
        func: kajit_ir::IntrinsicFn,
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
