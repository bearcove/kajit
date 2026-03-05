use kajit_emit::x64::{self, Emitter, FinalizedEmission, LabelId, Mem};

use crate::context::{
    CTX_ERROR_CODE, CTX_INPUT_END, CTX_INPUT_PTR, ENC_ERROR_CODE, ENC_OUTPUT_END, ENC_OUTPUT_PTR,
    ErrorCode,
};
use crate::recipe::{ErrorTarget, Op, Recipe, Slot, Width};

/// Base frame size.
///
/// - System V AMD64: `rbp + pad + r12 + r13 + r14 + r15` = 48 bytes
/// - Windows x64: `shadow(32) + rbp + pad + r12 + r13 + r14 + r15` = 80 bytes
///
/// The Windows layout reserves 32 bytes at [rsp+0..31] as the callee home
/// (shadow) space required by the Windows x64 ABI before every call.
#[cfg(not(windows))]
pub const BASE_FRAME: u32 = 48;
#[cfg(windows)]
pub const BASE_FRAME: u32 = 80;

/// Emission context — wraps the assembler plus bookkeeping labels.
pub struct EmitCtx {
    pub emit: Emitter,
    pub error_exit: LabelId,
    pub entry: u32,
    /// Total frame size (base + extra, 16-byte aligned).
    pub frame_size: u32,
}

// Register assignments (callee-saved across all platforms):
//   r12 = cached input_ptr
//   r13 = cached input_end
//   r14 = out pointer
//   r15 = ctx pointer
//
// Scratch (caller-saved):
//   rax = fn ptr loads, return values
//   r10, r11 = temporaries
//
// Argument registers for calls to intrinsics:
//   System V AMD64:  arg0=rdi, arg1=rsi, arg2=rdx, arg3=rcx, arg4=r8, arg5=r9
//   Windows x64:     arg0=rcx, arg1=rdx, arg2=r8,  arg3=r9   (4 register args only)

impl EmitCtx {
    // ── Call helpers ──────────────────────────────────────────────────
    //
    // These small helpers factor out the repeated patterns around function
    // calls in the JIT: flushing/reloading the cached cursor, loading a
    // function pointer, and checking the error flag.

    /// Load a function pointer into rax and call it.
    fn emit_call_fn_ptr(&mut self, ptr: *const u8) {
        let ptr_val = ptr as i64;
        self.emit
            .emit_with(|buf| {
                x64::encode_mov_r64_imm64(0, ptr_val as u64, buf)?;
                x64::encode_call_r64(0, buf)
            })
            .expect("call fn ptr");
    }

    /// Flush the cached input cursor (r12) back to ctx.input_ptr.
    fn emit_flush_input_cursor(&mut self) {
        self.emit
            .emit_with(|buf| {
                x64::encode_mov_m_r64(
                    Mem {
                        base: 15,
                        disp: CTX_INPUT_PTR as i32,
                    },
                    12,
                    buf,
                )
            })
            .expect("flush input cursor");
    }

    /// Reload the cached input cursor from ctx and check the error flag.
    /// Branches to `error_exit` if `ctx.error.code != 0`.
    fn emit_reload_cursor_and_check_error(&mut self) {
        let error_exit = self.error_exit;
        self.emit
            .emit_with(|buf| {
                x64::encode_mov_r64_m(
                    12,
                    Mem {
                        base: 15,
                        disp: CTX_INPUT_PTR as i32,
                    },
                    buf,
                )?;
                x64::encode_mov_r32_m(
                    10,
                    Mem {
                        base: 15,
                        disp: CTX_ERROR_CODE as i32,
                    },
                    buf,
                )?;
                x64::encode_test_r32_r32(10, 10, buf)
            })
            .expect("reload cursor and test");
        self.emit
            .emit_jnz_label(error_exit)
            .expect("reload cursor error");
    }

    /// Check the error flag without reloading the cursor.
    /// Branches to `error_exit` if `ctx.error.code != 0`.
    fn emit_check_error(&mut self) {
        let error_exit = self.error_exit;
        self.emit
            .emit_with(|buf| {
                x64::encode_mov_r32_m(
                    10,
                    Mem {
                        base: 15,
                        disp: CTX_ERROR_CODE as i32,
                    },
                    buf,
                )?;
                x64::encode_test_r32_r32(10, 10, buf)
            })
            .expect("check error");
        self.emit.emit_jnz_label(error_exit).expect("check error");
    }

    /// Flush the cached output cursor (r12) back to ctx for encoding.
    fn emit_enc_flush_output_cursor(&mut self) {
        self.emit
            .emit_with(|buf| {
                x64::encode_mov_m_r64(
                    Mem {
                        base: 15,
                        disp: ENC_OUTPUT_PTR as i32,
                    },
                    12,
                    buf,
                )
            })
            .expect("flush output cursor");
    }

    /// Reload output_ptr and output_end from ctx and check the error flag.
    fn emit_enc_reload_and_check_error(&mut self) {
        let error_exit = self.error_exit;
        self.emit
            .emit_with(|buf| {
                x64::encode_mov_r64_m(
                    12,
                    Mem {
                        base: 15,
                        disp: ENC_OUTPUT_PTR as i32,
                    },
                    buf,
                )?;
                x64::encode_mov_r64_m(
                    13,
                    Mem {
                        base: 15,
                        disp: ENC_OUTPUT_END as i32,
                    },
                    buf,
                )?;
                x64::encode_mov_r32_m(
                    10,
                    Mem {
                        base: 15,
                        disp: ENC_ERROR_CODE as i32,
                    },
                    buf,
                )?;
                x64::encode_test_r32_r32(10, 10, buf)
            })
            .expect("reload output and test");
        self.emit
            .emit_jnz_label(error_exit)
            .expect("reload output error");
    }

    /// Create a new EmitCtx. Does not emit any code.
    ///
    /// `extra_stack` is the number of additional bytes the format needs on the
    /// stack (e.g. 32 for JSON's bitset + key_ptr + key_len + peek_byte).
    /// The total frame is rounded up to 16-byte alignment.
    ///
    /// Call `begin_func()` to emit a function prologue.
    pub fn new(extra_stack: u32) -> Self {
        let frame_size = (BASE_FRAME + extra_stack + 15) & !15;
        let mut emit = Emitter::new();
        let error_exit = emit.new_label();
        let entry = 0;

        EmitCtx {
            emit,
            error_exit,
            entry,
            frame_size,
        }
    }

    /// Emit a function prologue. Returns the entry offset and a fresh error_exit label.
    ///
    /// The returned error_exit label must be passed to `end_func` when done emitting
    /// this function's body.
    ///
    /// # Register assignments after prologue
    /// - r12 = cached input_ptr
    /// - r13 = cached input_end
    /// - r14 = out pointer
    /// - r15 = ctx pointer
    pub fn begin_func(&mut self) -> (u32, LabelId) {
        let error_exit = self.emit.new_label();
        let entry = self.emit.current_offset();
        let frame_size = self.frame_size;

        // On entry: rsp is 8-mod-16 (return address was pushed by `call`).
        // push rbp → rsp is now 16-byte aligned.
        // sub rsp, frame_size → stays 16-byte aligned (frame_size is multiple of 16).
        //
        // System V AMD64 frame layout (BASE_FRAME = 48):
        //   [rsp+0]:  saved rbp      [rsp+8]:  saved rbx
        //   [rsp+16]: saved r12      [rsp+24]: saved r13
        //   [rsp+32]: saved r14      [rsp+40]: saved r15
        //   [rsp+48..]: extra stack  (args arrive in rdi=out, rsi=ctx)
        //
        // Windows x64 frame layout (BASE_FRAME = 80):
        //   [rsp+0..31]: shadow/home space (32 bytes, callee may write here)
        //   [rsp+32]: saved rbp      [rsp+40]: saved rbx
        //   [rsp+48]: saved r12      [rsp+56]: saved r13
        //   [rsp+64]: saved r14      [rsp+72]: saved r15
        //   [rsp+80..]: extra stack  (args arrive in rcx=out, rdx=ctx)
        #[cfg(not(windows))]
        self.emit
            .emit_with(|buf| {
                x64::encode_push_r64(5, buf)?;
                x64::encode_sub_r64_imm32(4, frame_size, buf)?;
                x64::encode_mov_m_r64(Mem { base: 4, disp: 0 }, 5, buf)?;
                x64::encode_mov_m_r64(Mem { base: 4, disp: 8 }, 3, buf)?;
                x64::encode_mov_m_r64(Mem { base: 4, disp: 16 }, 12, buf)?;
                x64::encode_mov_m_r64(Mem { base: 4, disp: 24 }, 13, buf)?;
                x64::encode_mov_m_r64(Mem { base: 4, disp: 32 }, 14, buf)?;
                x64::encode_mov_m_r64(Mem { base: 4, disp: 40 }, 15, buf)?;
                x64::encode_mov_r64_r64(14, 7, buf)?;
                x64::encode_mov_r64_r64(15, 6, buf)?;
                x64::encode_mov_r64_m(
                    12,
                    Mem {
                        base: 15,
                        disp: CTX_INPUT_PTR as i32,
                    },
                    buf,
                )?;
                x64::encode_mov_r64_m(
                    13,
                    Mem {
                        base: 15,
                        disp: CTX_INPUT_END as i32,
                    },
                    buf,
                )
            })
            .expect("begin prologue");
        #[cfg(windows)]
        self.emit
            .emit_with(|buf| {
                x64::encode_push_r64(5, buf)?;
                x64::encode_sub_r64_imm32(4, frame_size, buf)?;
                x64::encode_mov_m_r64(Mem { base: 4, disp: 32 }, 5, buf)?;
                x64::encode_mov_m_r64(Mem { base: 4, disp: 40 }, 3, buf)?;
                x64::encode_mov_m_r64(Mem { base: 4, disp: 48 }, 12, buf)?;
                x64::encode_mov_m_r64(Mem { base: 4, disp: 56 }, 13, buf)?;
                x64::encode_mov_m_r64(Mem { base: 4, disp: 64 }, 14, buf)?;
                x64::encode_mov_m_r64(Mem { base: 4, disp: 72 }, 15, buf)?;
                x64::encode_mov_r64_r64(14, 1, buf)?;
                x64::encode_mov_r64_r64(15, 2, buf)?;
                x64::encode_mov_r64_m(
                    12,
                    Mem {
                        base: 15,
                        disp: CTX_INPUT_PTR as i32,
                    },
                    buf,
                )?;
                x64::encode_mov_r64_m(
                    13,
                    Mem {
                        base: 15,
                        disp: CTX_INPUT_END as i32,
                    },
                    buf,
                )
            })
            .expect("begin prologue windows");

        self.error_exit = error_exit;
        (entry, error_exit)
    }

    /// Emit the success epilogue and error exit for the current function.
    ///
    /// `error_exit` must be the label returned by the corresponding `begin_func` call.
    pub fn end_func(&mut self, error_exit: LabelId) {
        let frame_size = self.frame_size as i32;

        #[cfg(not(windows))]
        self.emit
            .emit_with(|buf| {
                x64::encode_mov_m_r64(
                    Mem {
                        base: 15,
                        disp: CTX_INPUT_PTR as i32,
                    },
                    12,
                    buf,
                )?;
                x64::encode_mov_r64_m(15, Mem { base: 4, disp: 40 }, buf)?;
                x64::encode_mov_r64_m(14, Mem { base: 4, disp: 32 }, buf)?;
                x64::encode_mov_r64_m(13, Mem { base: 4, disp: 24 }, buf)?;
                x64::encode_mov_r64_m(12, Mem { base: 4, disp: 16 }, buf)?;
                x64::encode_mov_r64_m(3, Mem { base: 4, disp: 8 }, buf)?;
                x64::encode_mov_r64_m(5, Mem { base: 4, disp: 0 }, buf)?;
                x64::encode_add_r64_imm32(4, frame_size as u32, buf)?;
                x64::encode_pop_r64(5, buf)?;
                x64::encode_ret(buf)
            })
            .expect("end success");
        #[cfg(windows)]
        self.emit
            .emit_with(|buf| {
                x64::encode_mov_m_r64(
                    Mem {
                        base: 15,
                        disp: CTX_INPUT_PTR as i32,
                    },
                    12,
                    buf,
                )?;
                x64::encode_mov_r64_m(15, Mem { base: 4, disp: 72 }, buf)?;
                x64::encode_mov_r64_m(14, Mem { base: 4, disp: 64 }, buf)?;
                x64::encode_mov_r64_m(13, Mem { base: 4, disp: 56 }, buf)?;
                x64::encode_mov_r64_m(12, Mem { base: 4, disp: 48 }, buf)?;
                x64::encode_mov_r64_m(3, Mem { base: 4, disp: 40 }, buf)?;
                x64::encode_mov_r64_m(5, Mem { base: 4, disp: 32 }, buf)?;
                x64::encode_add_r64_imm32(4, frame_size as u32, buf)?;
                x64::encode_pop_r64(5, buf)?;
                x64::encode_ret(buf)
            })
            .expect("end success windows");

        self.emit.bind_label(error_exit).expect("bind error_exit");
        #[cfg(not(windows))]
        self.emit
            .emit_with(|buf| {
                x64::encode_mov_r64_m(15, Mem { base: 4, disp: 40 }, buf)?;
                x64::encode_mov_r64_m(14, Mem { base: 4, disp: 32 }, buf)?;
                x64::encode_mov_r64_m(13, Mem { base: 4, disp: 24 }, buf)?;
                x64::encode_mov_r64_m(12, Mem { base: 4, disp: 16 }, buf)?;
                x64::encode_mov_r64_m(3, Mem { base: 4, disp: 8 }, buf)?;
                x64::encode_mov_r64_m(5, Mem { base: 4, disp: 0 }, buf)?;
                x64::encode_add_r64_imm32(4, frame_size as u32, buf)?;
                x64::encode_pop_r64(5, buf)?;
                x64::encode_ret(buf)
            })
            .expect("end error");
        #[cfg(windows)]
        self.emit
            .emit_with(|buf| {
                x64::encode_mov_r64_m(15, Mem { base: 4, disp: 72 }, buf)?;
                x64::encode_mov_r64_m(14, Mem { base: 4, disp: 64 }, buf)?;
                x64::encode_mov_r64_m(13, Mem { base: 4, disp: 56 }, buf)?;
                x64::encode_mov_r64_m(12, Mem { base: 4, disp: 48 }, buf)?;
                x64::encode_mov_r64_m(3, Mem { base: 4, disp: 40 }, buf)?;
                x64::encode_mov_r64_m(5, Mem { base: 4, disp: 32 }, buf)?;
                x64::encode_add_r64_imm32(4, frame_size as u32, buf)?;
                x64::encode_pop_r64(5, buf)?;
                x64::encode_ret(buf)
            })
            .expect("end error windows");
    }

    /// Emit a call to another emitted function.
    ///
    /// Convention: rdi = out + field_offset, rsi = ctx (same as our entry convention).
    /// Flushes cursor before call, reloads after, checks error.
    ///
    /// r[impl callconv.inter-function]
    pub fn emit_call_emitted_func(&mut self, label: LabelId, field_offset: u32) {
        self.emit_flush_input_cursor();
        #[cfg(not(windows))]
        self.emit
            .emit_with(|buf| {
                x64::encode_lea_r64_m(
                    7,
                    Mem {
                        base: 14,
                        disp: field_offset as i32,
                    },
                    buf,
                )?;
                x64::encode_mov_r64_r64(6, 15, buf)
            })
            .expect("lea rdi/rsi");
        #[cfg(windows)]
        self.emit
            .emit_with(|buf| {
                x64::encode_lea_r64_m(
                    1,
                    Mem {
                        base: 14,
                        disp: field_offset as i32,
                    },
                    buf,
                )?;
                x64::encode_mov_r64_r64(2, 15, buf)
            })
            .expect("lea rcx/rdx");
        self.emit.emit_call_label(label).expect("call emitted fn");
        self.emit_reload_cursor_and_check_error();
    }

    /// Emit a call to an intrinsic function.
    ///
    /// Before the call: flushes the cached input_ptr back to ctx.
    /// Sets up args: rdi = ctx, rsi = out + field_offset.
    /// After the call: reloads input_ptr from ctx, checks error slot.
    pub fn emit_call_intrinsic(&mut self, fn_ptr: *const u8, field_offset: u32) {
        self.emit_flush_input_cursor();
        #[cfg(not(windows))]
        self.emit
            .emit_with(|buf| {
                x64::encode_mov_r64_r64(7, 15, buf)?;
                x64::encode_lea_r64_m(
                    6,
                    Mem {
                        base: 14,
                        disp: field_offset as i32,
                    },
                    buf,
                )
            })
            .expect("mov rdi/rsi");
        #[cfg(windows)]
        self.emit
            .emit_with(|buf| {
                x64::encode_mov_r64_r64(1, 15, buf)?;
                x64::encode_lea_r64_m(
                    2,
                    Mem {
                        base: 14,
                        disp: field_offset as i32,
                    },
                    buf,
                )
            })
            .expect("mov rcx/rdx");
        self.emit_call_fn_ptr(fn_ptr);
        self.emit_reload_cursor_and_check_error();
    }

    /// Emit a call to an intrinsic that takes only ctx as argument.
    /// Flushes cursor, calls, reloads cursor, checks error.
    pub fn emit_call_intrinsic_ctx_only(&mut self, fn_ptr: *const u8) {
        self.emit_flush_input_cursor();
        #[cfg(not(windows))]
        self.emit
            .emit_with(|buf| x64::encode_mov_r64_r64(7, 15, buf))
            .expect("mov rdi");
        #[cfg(windows)]
        self.emit
            .emit_with(|buf| x64::encode_mov_r64_r64(1, 15, buf))
            .expect("mov rcx");
        self.emit_call_fn_ptr(fn_ptr);
        self.emit_reload_cursor_and_check_error();
    }

    /// Emit a call to an intrinsic that takes (ctx, &mut stack_slot).
    /// arg0 = ctx, arg1 = rsp + sp_offset. Flushes/reloads cursor, checks error.
    pub fn emit_call_intrinsic_ctx_and_stack_out(&mut self, fn_ptr: *const u8, sp_offset: u32) {
        self.emit_flush_input_cursor();
        #[cfg(not(windows))]
        self.emit
            .emit_with(|buf| {
                x64::encode_mov_r64_r64(7, 15, buf)?;
                x64::encode_lea_r64_m(
                    6,
                    Mem {
                        base: 4,
                        disp: sp_offset as i32,
                    },
                    buf,
                )
            })
            .expect("mov rdi/lea rsi");
        #[cfg(windows)]
        self.emit
            .emit_with(|buf| {
                x64::encode_mov_r64_r64(1, 15, buf)?;
                x64::encode_lea_r64_m(
                    2,
                    Mem {
                        base: 4,
                        disp: sp_offset as i32,
                    },
                    buf,
                )
            })
            .expect("mov rcx/lea rdx");
        self.emit_call_fn_ptr(fn_ptr);
        self.emit_reload_cursor_and_check_error();
    }

    /// Emit a call to an intrinsic that takes (ctx, out_ptr1, out_ptr2).
    /// arg0 = ctx, arg1 = rsp + sp_offset1, arg2 = rsp + sp_offset2.
    pub fn emit_call_intrinsic_ctx_and_two_stack_outs(
        &mut self,
        fn_ptr: *const u8,
        sp_offset1: u32,
        sp_offset2: u32,
    ) {
        self.emit_flush_input_cursor();
        #[cfg(not(windows))]
        self.emit
            .emit_with(|buf| {
                x64::encode_mov_r64_r64(7, 15, buf)?;
                x64::encode_lea_r64_m(
                    6,
                    Mem {
                        base: 4,
                        disp: sp_offset1 as i32,
                    },
                    buf,
                )?;
                x64::encode_lea_r64_m(
                    2,
                    Mem {
                        base: 4,
                        disp: sp_offset2 as i32,
                    },
                    buf,
                )
            })
            .expect("mov ctx+args");
        #[cfg(windows)]
        self.emit
            .emit_with(|buf| {
                x64::encode_mov_r64_r64(1, 15, buf)?;
                x64::encode_lea_r64_m(
                    2,
                    Mem {
                        base: 4,
                        disp: sp_offset1 as i32,
                    },
                    buf,
                )?;
                x64::encode_lea_r64_m(
                    8,
                    Mem {
                        base: 4,
                        disp: sp_offset2 as i32,
                    },
                    buf,
                )
            })
            .expect("mov ctx+args");
        self.emit_call_fn_ptr(fn_ptr);
        self.emit_reload_cursor_and_check_error();
    }

    /// Emit a call to a pure function (no ctx, no flush/reload/error-check).
    /// Used for key_equals: arg0=key_ptr, arg1=key_len, arg2=expected_ptr, arg3=expected_len.
    /// Return value is in rax.
    pub fn emit_call_pure_4arg(
        &mut self,
        fn_ptr: *const u8,
        arg0_sp_offset: u32,
        arg1_sp_offset: u32,
        expected_ptr: *const u8,
        expected_len: u32,
    ) {
        let expected_addr = expected_ptr as i64;

        #[cfg(not(windows))]
        self.emit
            .emit_with(|buf| {
                x64::encode_mov_r64_m(
                    7,
                    Mem {
                        base: 4,
                        disp: arg0_sp_offset as i32,
                    },
                    buf,
                )?;
                x64::encode_mov_r64_m(
                    6,
                    Mem {
                        base: 4,
                        disp: arg1_sp_offset as i32,
                    },
                    buf,
                )?;
                x64::encode_mov_r64_imm64(2, expected_addr as u64, buf)?;
                x64::encode_mov_r32_imm32(1, expected_len, buf)
            })
            .expect("mov args");
        #[cfg(windows)]
        self.emit
            .emit_with(|buf| {
                x64::encode_mov_r64_m(
                    1,
                    Mem {
                        base: 4,
                        disp: arg0_sp_offset as i32,
                    },
                    buf,
                )?;
                x64::encode_mov_r64_m(
                    2,
                    Mem {
                        base: 4,
                        disp: arg1_sp_offset as i32,
                    },
                    buf,
                )?;
                x64::encode_mov_r64_imm64(8, expected_addr as u64, buf)?;
                x64::encode_mov_r32_imm32(9, expected_len, buf)
            })
            .expect("mov args");
        self.emit_call_fn_ptr(fn_ptr);
    }

    /// Allocate a new dynamic label.
    pub fn new_label(&mut self) -> LabelId {
        self.emit.new_label()
    }

    /// Bind a dynamic label at the current position.
    pub fn bind_label(&mut self, label: LabelId) {
        self.emit.bind_label(label).expect("bind label");
    }

    /// Set source location metadata for subsequent emitted instructions.
    pub fn set_source_location(&mut self, location: kajit_emit::SourceLocation) {
        self.emit.set_source_location(location);
    }

    /// Emit an unconditional branch to the given label.
    pub fn emit_branch(&mut self, label: LabelId) {
        self.emit.emit_jmp_label(label).expect("jmp");
    }

    /// Write an error code to ctx and branch to error_exit.
    pub fn emit_set_error(&mut self, code: ErrorCode) {
        let error_exit = self.error_exit;
        let error_code = code as i32;
        self.emit
            .emit_with(|buf| {
                x64::encode_mov_r32_imm32(10, error_code as u32, buf)?;
                x64::encode_mov_m_r32(
                    Mem {
                        base: 15,
                        disp: CTX_ERROR_CODE as i32,
                    },
                    10,
                    buf,
                )
            })
            .expect("store error code");
        self.emit
            .emit_jmp_label(error_exit)
            .expect("jmp error_exit");
    }

    /// Compute len = cursor - [rsp+start_slot], store to [rsp+len_slot], advance cursor past `"`.
    pub fn emit_compute_key_len_and_advance(&mut self, start_sp_offset: u32, len_sp_offset: u32) {
        self.emit
            .emit_with(|buf| {
                x64::encode_mov_r64_r64(0, 12, buf)?;
                x64::encode_sub_r64_m(
                    0,
                    Mem {
                        base: 4,
                        disp: start_sp_offset as i32,
                    },
                    buf,
                )?;
                x64::encode_mov_m_r64(
                    Mem {
                        base: 4,
                        disp: len_sp_offset as i32,
                    },
                    0,
                    buf,
                )?;
                x64::encode_add_r64_imm32(12, 1, buf)?;
                Ok(())
            })
            .expect("compute key len");
    }

    /// Emit `test rax, rax; jnz label` — branch if rax is nonzero.
    pub fn emit_cbnz_x0(&mut self, label: LabelId) {
        self.emit
            .emit_with(|buf| x64::encode_test_r64_r64(0, 0, buf))
            .expect("test rax");
        self.emit.emit_jnz_label(label).expect("jnz");
    }

    /// Zero a 64-bit stack slot at rsp + offset.
    pub fn emit_zero_stack_slot(&mut self, sp_offset: u32) {
        self.emit
            .emit_with(|buf| {
                x64::encode_mov_m_r64(
                    Mem {
                        base: 4,
                        disp: sp_offset as i32,
                    },
                    0,
                    buf,
                )
            })
            .expect("zero slot");
    }

    /// Load a byte from rsp + sp_offset, compare with byte_val, branch if equal.
    pub fn emit_stack_byte_cmp_branch(&mut self, sp_offset: u32, byte_val: u8, label: LabelId) {
        let byte_val = byte_val as i32;
        self.emit
            .emit_with(|buf| {
                x64::encode_movzx_r32_rm8(
                    10,
                    x64::Operand::Mem(Mem {
                        base: 4,
                        disp: sp_offset as i32,
                    }),
                    buf,
                )?;
                x64::encode_cmp_r64_imm32(10, byte_val as u32, buf)?;
                Ok(())
            })
            .expect("cmp byte");
        self.emit.emit_je_label(label).expect("je");
    }

    /// Set bit `bit_index` in a 64-bit stack slot at rsp + sp_offset.
    pub fn emit_set_bit_on_stack(&mut self, sp_offset: u32, bit_index: u32) {
        let mask = (1u64 << bit_index) as i64;

        self.emit
            .emit_with(|buf| {
                x64::encode_mov_r64_m(
                    0,
                    Mem {
                        base: 4,
                        disp: sp_offset as i32,
                    },
                    buf,
                )?;
                x64::encode_mov_r64_imm64(10, mask as u64, buf)?;
                x64::encode_or_r64_r64(0, 10, buf)?;
                x64::encode_mov_m_r64(
                    Mem {
                        base: 4,
                        disp: sp_offset as i32,
                    },
                    0,
                    buf,
                )
            })
            .expect("set bit on stack");
    }

    /// Check that the 64-bit stack slot at rsp + sp_offset equals expected_mask.
    /// If not, set MissingRequiredField error and branch to error_exit.
    pub fn emit_check_bitset(&mut self, sp_offset: u32, expected_mask: u64) {
        let error_exit = self.error_exit;
        let ok_label = self.emit.new_label();
        let expected_mask = expected_mask as i64;
        let error_code = crate::context::ErrorCode::MissingRequiredField as i32;

        self.emit
            .emit_with(|buf| {
                x64::encode_mov_r64_m(
                    0,
                    Mem {
                        base: 4,
                        disp: sp_offset as i32,
                    },
                    buf,
                )?;
                x64::encode_mov_r64_imm64(10, expected_mask as u64, buf)?;
                x64::encode_and_r64_r64(0, 10, buf)?;
                x64::encode_cmp_r64_r64(0, 10, buf)?;
                Ok(())
            })
            .expect("check bitset");
        self.emit.emit_je_label(ok_label).expect("je");
        self.emit
            .emit_with(|buf| {
                x64::encode_mov_r32_imm32(10, error_code as u32, buf)?;
                x64::encode_mov_m_r32(
                    Mem {
                        base: 15,
                        disp: CTX_ERROR_CODE as i32,
                    },
                    10,
                    buf,
                )
            })
            .expect("set error");
        self.emit.emit_jmp_label(error_exit).expect("jmp");
        self.emit.bind_label(ok_label).expect("bind ok");
    }

    // ── Inline scalar reads (recipe-based) ─────────────────────────────

    pub fn emit_inline_read_byte(&mut self, offset: u32) {
        self.emit_recipe(&crate::recipe::read_byte(offset));
    }

    pub fn emit_inline_read_byte_to_stack(&mut self, sp_offset: u32) {
        self.emit_recipe(&crate::recipe::read_byte_to_stack(sp_offset));
    }

    pub fn emit_inline_read_bool(&mut self, offset: u32) {
        self.emit_recipe(&crate::recipe::read_bool(offset));
    }

    pub fn emit_inline_read_f32(&mut self, offset: u32) {
        self.emit_recipe(&crate::recipe::read_f32(offset));
    }

    pub fn emit_inline_read_f64(&mut self, offset: u32) {
        self.emit_recipe(&crate::recipe::read_f64(offset));
    }

    pub fn emit_inline_varint_fast_path(
        &mut self,
        offset: u32,
        store_width: u32,
        zigzag: bool,
        intrinsic_fn_ptr: *const u8,
    ) {
        let width = match store_width {
            2 => Width::W2,
            4 => Width::W4,
            8 => Width::W8,
            _ => panic!("unsupported varint store width: {store_width}"),
        };
        self.emit_recipe(&crate::recipe::varint_fast_path(
            offset,
            width,
            zigzag,
            intrinsic_fn_ptr,
        ));
    }

    // ── Inline string reads (recipe-based) ──────────────────────────────

    pub fn emit_inline_postcard_string_malum(
        &mut self,
        offset: u32,
        string_offsets: &crate::malum::StringOffsets,
        slow_varint_intrinsic: *const u8,
        validate_alloc_copy_intrinsic: *const u8,
    ) {
        self.emit_recipe(&crate::recipe::postcard_string_malum(
            offset,
            string_offsets,
            slow_varint_intrinsic,
            validate_alloc_copy_intrinsic,
        ));
    }

    // ── JSON string scanning (SSE2 vectorized) ────────────────────────

    /// Emit a vectorized scan loop that searches for `"` or `\` in the input.
    ///
    /// **Precondition**: r12 (cached input_ptr) points just after the opening `"`.
    /// **Postcondition**: r12 points at the `"` or `\` byte found, then branches
    /// to the corresponding label.
    ///
    /// Uses SSE2 pcmpeqb + pmovmskb to process 16 bytes per iteration.
    /// Falls back to scalar for the last < 16 bytes.
    pub fn emit_json_string_scan(
        &mut self,
        found_quote: LabelId,
        found_escape: LabelId,
        unterminated: LabelId,
    ) {
        let vector_loop = self.emit.new_label();
        let scalar_tail = self.emit.new_label();
        let advance_16 = self.emit.new_label();

        // Broadcast '"' (0x22) and '\' (0x5C) into xmm1 and xmm2
        self.emit
            .emit_with(|buf| {
                x64::encode_mov_r32_imm32(0, 0x2222_2222, buf)?;
                x64::encode_movd_xmm_r64(1, 0, buf)?;
                x64::encode_pshufd_xmm_xmm_imm8(1, 1, 0, buf)?;
                x64::encode_mov_r32_imm32(0, 0x5c5c_5c5c, buf)?;
                x64::encode_movd_xmm_r64(2, 0, buf)?;
                x64::encode_pshufd_xmm_xmm_imm8(2, 2, 0, buf)?;
                Ok(())
            })
            .expect("load scan vectors");
        self.emit.bind_label(vector_loop).expect("bind vector_loop");
        self.emit
            .emit_with(|buf| {
                x64::encode_mov_r64_r64(0, 13, buf)?;
                x64::encode_sub_r64_r64(0, 12, buf)?;
                x64::encode_cmp_r64_imm32(0, 16, buf)?;
                Ok(())
            })
            .expect("test vector chunk");
        self.emit
            .emit_jcc_label(scalar_tail, x64::Condition::Lo)
            .expect("jb scalar_tail");
        self.emit
            .emit_with(|buf| {
                x64::encode_movdqu_xmm_m(0, 12, 0, buf)?;
                x64::encode_movdqa_xmm_xmm(3, 1, buf)?;
                x64::encode_pcmpeqb_xmm_xmm(3, 0, buf)?;
                x64::encode_movdqa_xmm_xmm(4, 2, buf)?;
                x64::encode_pcmpeqb_xmm_xmm(4, 0, buf)?;
                x64::encode_por_xmm_xmm(3, 4, buf)?;
                x64::encode_pmovmskb_r32_xmm(0, 3, buf)?;
                x64::encode_test_r32_r32(0, 0, buf)
            })
            .expect("vector compare");
        self.emit.emit_jz_label(advance_16).expect("jz advance_16");
        self.emit
            .emit_with(|buf| {
                x64::encode_tzcnt_r32_r32(0, 0, buf)?;
                x64::encode_add_r64_r64(12, 0, buf)?;
                x64::encode_movzx_r32_rm8(10, x64::Operand::Mem(Mem { base: 12, disp: 0 }), buf)?;
                x64::encode_cmp_r64_imm32(10, 0x22, buf)?;
                Ok(())
            })
            .expect("check quote");
        self.emit.emit_je_label(found_quote).expect("found quote");
        self.emit
            .emit_jmp_label(found_escape)
            .expect("found escape");
        self.emit.bind_label(advance_16).expect("bind advance_16");
        self.emit
            .emit_with(|buf| x64::encode_add_r64_imm32(12, 16, buf))
            .expect("advance 16");
        self.emit.emit_jmp_label(vector_loop).expect("loop vector");
        self.emit.bind_label(scalar_tail).expect("bind scalar_tail");
        self.emit
            .emit_with(|buf| {
                x64::encode_cmp_r64_r64(12, 13, buf)?;
                x64::encode_movzx_r32_rm8(10, x64::Operand::Mem(Mem { base: 12, disp: 0 }), buf)?;
                x64::encode_cmp_r64_imm32(10, 0x22, buf)?;
                Ok(())
            })
            .expect("scalar compare quote");
        self.emit.emit_je_label(found_quote).expect("found quote");
        self.emit
            .emit_with(|buf| x64::encode_cmp_r64_imm32(10, 0x5c, buf))
            .expect("compare escape");
        self.emit.emit_je_label(found_escape).expect("found escape");
        self.emit
            .emit_with(|buf| x64::encode_add_r64_imm32(12, 1, buf))
            .expect("advance scalar");
        self.emit.emit_jmp_label(scalar_tail).expect("loop scalar");
        self.emit
            .emit_with(|buf| x64::encode_cmp_r64_r64(12, 13, buf))
            .expect("scalar bounds");
        self.emit
            .emit_jcc_label(unterminated, x64::Condition::Hs)
            .expect("unterminated");
    }

    /// Inline skip-whitespace: loop over space/tab/newline/cr, advancing r12.
    /// No function call, no ctx flush.
    pub fn emit_inline_skip_ws(&mut self) {
        let ws_loop = self.emit.new_label();
        let ws_done = self.emit.new_label();
        let advance = self.emit.new_label();

        self.emit.bind_label(ws_loop).expect("bind ws_loop");
        self.emit
            .emit_with(|buf| {
                x64::encode_cmp_r64_r64(12, 13, buf)?;
                x64::encode_jcc_rel32(buf, x64::Condition::Hs, 0)?;
                x64::encode_movzx_r32_rm8(10, x64::Operand::Mem(Mem { base: 12, disp: 0 }), buf)?;
                x64::encode_cmp_r64_imm32(10, b' ' as u32, buf)?;
                Ok(())
            })
            .expect("skip ws space");
        self.emit.emit_je_label(advance).expect("space");
        self.emit
            .emit_with(|buf| x64::encode_cmp_r64_imm32(10, b'\n' as u32, buf))
            .expect("skip ws lf");
        self.emit.emit_je_label(advance).expect("lf");
        self.emit
            .emit_with(|buf| x64::encode_cmp_r64_imm32(10, b'\r' as u32, buf))
            .expect("skip ws cr");
        self.emit.emit_je_label(advance).expect("cr");
        self.emit
            .emit_with(|buf| x64::encode_cmp_r64_imm32(10, b'\t' as u32, buf))
            .expect("skip ws tab");
        self.emit.emit_jne_label(ws_done).expect("done");
        self.emit.bind_label(advance).expect("bind advance");
        self.emit
            .emit_with(|buf| x64::encode_add_r64_imm32(12, 1, buf))
            .expect("advance ws");
        self.emit.emit_jmp_label(ws_loop).expect("loop ws");
        self.emit.bind_label(ws_done).expect("bind ws_done");
    }

    /// Inline comma-or-end-array: skip whitespace, then check for ',' or ']'.
    /// Writes 0 (comma) or 1 (']') to stack at sp_offset. Errors on anything else.
    pub fn emit_inline_comma_or_end_array(&mut self, sp_offset: u32) {
        let error_exit = self.error_exit;
        let got_comma = self.emit.new_label();
        let got_end = self.emit.new_label();
        let done = self.emit.new_label();
        let error_code = ErrorCode::UnexpectedCharacter as i32;

        self.emit_inline_skip_ws();

        self.emit
            .emit_with(|buf| {
                x64::encode_cmp_r64_r64(12, 13, buf)?;
                x64::encode_movzx_r32_rm8(10, x64::Operand::Mem(Mem { base: 12, disp: 0 }), buf)?;
                x64::encode_add_r64_imm32(12, 1, buf)?;
                x64::encode_cmp_r64_imm32(10, b',' as u32, buf)?;
                Ok(())
            })
            .expect("check comma/end");
        self.emit.emit_je_label(got_comma).expect("comma");
        self.emit
            .emit_with(|buf| x64::encode_cmp_r64_imm32(10, b']' as u32, buf))
            .expect("check array end");
        self.emit.emit_je_label(got_end).expect("array end");
        self.emit
            .emit_with(|buf| {
                x64::encode_mov_r32_imm32(10, error_code as u32, buf)?;
                x64::encode_mov_m_r32(
                    Mem {
                        base: 15,
                        disp: CTX_ERROR_CODE as i32,
                    },
                    10,
                    buf,
                )
            })
            .expect("write array error");
        self.emit.emit_jmp_label(error_exit).expect("array err");
        self.emit.bind_label(got_comma).expect("bind comma");
        self.emit
            .emit_with(|buf| {
                x64::encode_mov_r32_imm32(11, 0, buf)?;
                x64::encode_mov_m_r8(4, sp_offset as i32, 11, buf)
            })
            .expect("write comma result");
        self.emit.emit_jmp_label(done).expect("comma_or_end done");
        self.emit.bind_label(got_end).expect("bind end");
        self.emit
            .emit_with(|buf| {
                x64::encode_mov_r32_imm32(11, 1, buf)?;
                x64::encode_mov_m_r8(4, sp_offset as i32, 11, buf)
            })
            .expect("write end result");
        self.emit.bind_label(done).expect("bind done");
    }

    /// Inline comma-or-end-object: skip whitespace, then check for ',' or '}'.
    /// Writes 0 (comma) or 1 ('}') to stack at sp_offset. Errors on anything else.
    pub fn emit_inline_comma_or_end_object(&mut self, sp_offset: u32) {
        let error_exit = self.error_exit;
        let got_comma = self.emit.new_label();
        let got_end = self.emit.new_label();
        let done = self.emit.new_label();
        let error_code = ErrorCode::UnexpectedCharacter as i32;

        self.emit_inline_skip_ws();

        self.emit
            .emit_with(|buf| {
                x64::encode_cmp_r64_r64(12, 13, buf)?;
                x64::encode_movzx_r32_rm8(10, x64::Operand::Mem(Mem { base: 12, disp: 0 }), buf)?;
                x64::encode_add_r64_imm32(12, 1, buf)?;
                x64::encode_cmp_r64_imm32(10, b',' as u32, buf)?;
                Ok(())
            })
            .expect("check comma/object end");
        self.emit.emit_je_label(got_comma).expect("comma");
        self.emit
            .emit_with(|buf| x64::encode_cmp_r64_imm32(10, b'}' as u32, buf))
            .expect("check object end");
        self.emit.emit_je_label(got_end).expect("object end");
        self.emit
            .emit_with(|buf| {
                x64::encode_mov_r32_imm32(10, error_code as u32, buf)?;
                x64::encode_mov_m_r32(
                    Mem {
                        base: 15,
                        disp: CTX_ERROR_CODE as i32,
                    },
                    10,
                    buf,
                )
            })
            .expect("write object error");
        self.emit.emit_jmp_label(error_exit).expect("obj err");
        self.emit.bind_label(got_comma).expect("bind comma");
        self.emit
            .emit_with(|buf| {
                x64::encode_mov_r32_imm32(11, 0, buf)?;
                x64::encode_mov_m_r8(4, sp_offset as i32, 11, buf)
            })
            .expect("write object comma");
        self.emit
            .emit_jmp_label(done)
            .expect("obj comma_or_end done");
        self.emit.bind_label(got_end).expect("bind end");
        self.emit
            .emit_with(|buf| {
                x64::encode_mov_r32_imm32(11, 1, buf)?;
                x64::encode_mov_m_r8(4, sp_offset as i32, 11, buf)
            })
            .expect("write obj end");
        self.emit.bind_label(done).expect("bind obj done");
    }

    /// Store the cached cursor (r12) to a stack slot.
    pub fn emit_save_cursor_to_stack(&mut self, sp_offset: u32) {
        self.emit
            .emit_with(|buf| {
                x64::encode_mov_m_r64(
                    Mem {
                        base: 4,
                        disp: sp_offset as i32,
                    },
                    12,
                    buf,
                )
            })
            .expect("save cursor");
    }

    /// Emit: skip whitespace, expect and consume `"`, branch to error_exit if not found.
    /// After this, r12 points just after the opening `"`.
    pub fn emit_json_expect_quote_after_ws(&mut self, _ws_intrinsic: *const u8) {
        let error_exit = self.error_exit;
        let not_quote = self.emit.new_label();
        let ok = self.emit.new_label();
        let error_code = ErrorCode::ExpectedStringKey as i32;

        self.emit_inline_skip_ws();

        self.emit
            .emit_with(|buf| x64::encode_cmp_r64_r64(12, 13, buf))
            .expect("check input bounds");
        self.emit
            .emit_jcc_label(not_quote, x64::Condition::Hi)
            .expect("not_quote bounds");
        self.emit
            .emit_with(|buf| {
                x64::encode_movzx_r32_rm8(10, x64::Operand::Mem(Mem { base: 12, disp: 0 }), buf)
            })
            .expect("load quote byte");
        self.emit
            .emit_with(|buf| x64::encode_cmp_r64_imm32(10, b'"' as u32, buf))
            .expect("compare quote");
        self.emit.emit_jne_label(not_quote).expect("not_quote");
        self.emit
            .emit_with(|buf| x64::encode_add_r64_imm32(12, 1, buf))
            .expect("advance quote");
        self.emit.emit_jmp_label(ok).expect("ok");
        self.emit.bind_label(not_quote).expect("bind not_quote");
        self.emit
            .emit_with(|buf| {
                x64::encode_mov_r32_imm32(10, error_code as u32, buf)?;
                x64::encode_mov_m_r32(
                    Mem {
                        base: 15,
                        disp: CTX_ERROR_CODE as i32,
                    },
                    10,
                    buf,
                )
            })
            .expect("write quote error");
        self.emit.emit_jmp_label(error_exit).expect("error");
        self.emit.bind_label(ok).expect("bind ok");
    }

    /// Call a post-scan intrinsic: fn(ctx, out+field_offset, start, len).
    /// `start_sp_offset` is the stack slot holding the start pointer.
    /// `len` is computed as r12 - start. Advances r12 past closing `"`.
    /// Flushes/reloads cursor, checks error.
    pub fn emit_call_string_finish(
        &mut self,
        fn_ptr: *const u8,
        field_offset: u32,
        start_sp_offset: u32,
    ) {
        // Compute len = r12 - start, flush cursor advanced past closing '"'
        self.emit
            .emit_with(|buf| {
                x64::encode_mov_r64_m(
                    10,
                    Mem {
                        base: 4,
                        disp: start_sp_offset as i32,
                    },
                    buf,
                )?;
                x64::encode_mov_r64_r64(0, 12, buf)?;
                x64::encode_sub_r64_r64(0, 10, buf)?;
                x64::encode_lea_r64_m(11, Mem { base: 12, disp: 1 }, buf)?;
                x64::encode_mov_m_r64(
                    Mem {
                        base: 15,
                        disp: CTX_INPUT_PTR as i32,
                    },
                    11,
                    buf,
                )
            })
            .expect("compute string finish args");
        // Args: ctx, out+offset, start, len
        #[cfg(not(windows))]
        self.emit
            .emit_with(|buf| {
                x64::encode_mov_r64_r64(7, 15, buf)?;
                x64::encode_lea_r64_m(
                    6,
                    Mem {
                        base: 14,
                        disp: field_offset as i32,
                    },
                    buf,
                )?;
                x64::encode_mov_r64_r64(2, 10, buf)?;
                x64::encode_mov_r64_r64(1, 0, buf)
            })
            .expect("setup string_finish args");
        #[cfg(windows)]
        self.emit
            .emit_with(|buf| {
                x64::encode_mov_r64_r64(1, 15, buf)?;
                x64::encode_lea_r64_m(
                    2,
                    Mem {
                        base: 14,
                        disp: field_offset as i32,
                    },
                    buf,
                )?;
                x64::encode_mov_r64_r64(8, 10, buf)?;
                x64::encode_mov_r64_r64(9, 0, buf)
            })
            .expect("setup string_finish args");
        self.emit_call_fn_ptr(fn_ptr);
        self.emit_reload_cursor_and_check_error();
    }

    /// Call a post-scan escape intrinsic: fn(ctx, out+field_offset, start, prefix_len).
    /// `start_sp_offset` is the stack slot holding the start pointer.
    /// `prefix_len` is computed as r12 - start. r12 is at the `\` byte.
    /// Flushes cursor, reloads after call, checks error.
    pub fn emit_call_string_escape(
        &mut self,
        fn_ptr: *const u8,
        field_offset: u32,
        start_sp_offset: u32,
    ) {
        self.emit
            .emit_with(|buf| {
                x64::encode_mov_r64_m(
                    10,
                    Mem {
                        base: 4,
                        disp: start_sp_offset as i32,
                    },
                    buf,
                )?;
                x64::encode_mov_r64_r64(0, 12, buf)?;
                x64::encode_sub_r64_r64(0, 10, buf)?;
                Ok(())
            })
            .expect("compute string_escape args");
        self.emit_flush_input_cursor();
        #[cfg(not(windows))]
        self.emit
            .emit_with(|buf| {
                x64::encode_mov_r64_r64(7, 15, buf)?;
                x64::encode_lea_r64_m(
                    6,
                    Mem {
                        base: 14,
                        disp: field_offset as i32,
                    },
                    buf,
                )?;
                x64::encode_mov_r64_r64(2, 10, buf)?;
                x64::encode_mov_r64_r64(1, 0, buf)?;
                Ok(())
            })
            .expect("setup string_escape args");
        #[cfg(windows)]
        self.emit
            .emit_with(|buf| {
                x64::encode_mov_r64_r64(1, 15, buf)?;
                x64::encode_lea_r64_m(
                    2,
                    Mem {
                        base: 14,
                        disp: field_offset as i32,
                    },
                    buf,
                )?;
                x64::encode_mov_r64_r64(8, 10, buf)?;
                x64::encode_mov_r64_r64(9, 0, buf)?;
                Ok(())
            })
            .expect("setup string_escape args");
        self.emit_call_fn_ptr(fn_ptr);
        self.emit_reload_cursor_and_check_error();
    }

    /// Call kajit_string_validate_alloc_copy(ctx, start, len) for the String malum path.
    /// `start_sp_offset` holds the start pointer. len = r12 - start.
    /// Returns buf pointer in rax. Does NOT advance cursor — caller does that.
    /// Saves len to `len_save_sp_offset` for WriteMalumString.
    pub fn emit_call_validate_alloc_copy_from_scan(
        &mut self,
        fn_ptr: *const u8,
        start_sp_offset: u32,
        len_save_sp_offset: u32,
    ) {
        self.emit
            .emit_with(|buf| {
                x64::encode_mov_r64_m(
                    10,
                    Mem {
                        base: 4,
                        disp: start_sp_offset as i32,
                    },
                    buf,
                )?;
                x64::encode_mov_r64_r64(11, 12, buf)?;
                x64::encode_sub_r64_r64(11, 10, buf)?;
                x64::encode_mov_m_r64(
                    Mem {
                        base: 4,
                        disp: len_save_sp_offset as i32,
                    },
                    11,
                    buf,
                )?;
                Ok(())
            })
            .expect("compute alloc_copy len");
        self.emit_flush_input_cursor();
        #[cfg(not(windows))]
        self.emit
            .emit_with(|buf| {
                x64::encode_mov_r64_r64(7, 15, buf)?;
                x64::encode_mov_r64_r64(6, 10, buf)?;
                x64::encode_mov_r64_r64(2, 11, buf)?;
                Ok(())
            })
            .expect("setup validate_alloc args");
        #[cfg(windows)]
        self.emit
            .emit_with(|buf| {
                x64::encode_mov_r64_r64(1, 15, buf)?;
                x64::encode_mov_r64_r64(2, 10, buf)?;
                x64::encode_mov_r64_r64(8, 11, buf)?;
                Ok(())
            })
            .expect("setup validate_alloc args");
        self.emit_call_fn_ptr(fn_ptr);
        // rax = buf pointer — no cursor reload needed
        self.emit_check_error();
    }

    /// Write String fields (ptr, len, cap) using malum offsets after validate_alloc_copy.
    /// rax = buf pointer from previous call return.
    /// Reads len from `len_sp_offset`. Advances cursor past closing `"`.
    pub fn emit_write_malum_string_and_advance(
        &mut self,
        field_offset: u32,
        string_offsets: &crate::malum::StringOffsets,
        len_sp_offset: u32,
    ) {
        let ptr_off = (field_offset + string_offsets.ptr_offset) as i32;
        let len_off = (field_offset + string_offsets.len_offset) as i32;
        let cap_off = (field_offset + string_offsets.cap_offset) as i32;

        self.emit
            .emit_with(|buf| {
                x64::encode_mov_m_r64(
                    Mem {
                        base: 14,
                        disp: ptr_off,
                    },
                    0,
                    buf,
                )?;
                x64::encode_mov_r64_m(
                    10,
                    Mem {
                        base: 4,
                        disp: len_sp_offset as i32,
                    },
                    buf,
                )?;
                x64::encode_mov_m_r64(
                    Mem {
                        base: 14,
                        disp: len_off,
                    },
                    10,
                    buf,
                )?;
                x64::encode_mov_m_r64(
                    Mem {
                        base: 14,
                        disp: cap_off,
                    },
                    10,
                    buf,
                )?;
                x64::encode_add_r64_imm32(12, 1, buf)
            })
            .expect("write malum string fields");
    }

    /// Emit inline key-reading slow path call: kajit_json_key_slow_from_jit(ctx, start, prefix_len, &key_ptr, &key_len).
    /// start is in `start_sp_offset`, prefix_len = r12 - start.
    /// key_ptr/key_len written to `key_ptr_sp_offset` and `key_len_sp_offset`.
    pub fn emit_call_key_slow_from_jit(
        &mut self,
        fn_ptr: *const u8,
        start_sp_offset: u32,
        key_ptr_sp_offset: u32,
        key_len_sp_offset: u32,
    ) {
        // Compute prefix_len = r12 - start
        self.emit
            .emit_with(|buf| {
                x64::encode_mov_r64_m(
                    10,
                    Mem {
                        base: 4,
                        disp: start_sp_offset as i32,
                    },
                    buf,
                )?;
                x64::encode_mov_r64_r64(0, 12, buf)?;
                x64::encode_sub_r64_r64(0, 10, buf)?;
                Ok(())
            })
            .expect("compute key_slow args");

        // Flush cursor (at '\' byte)
        self.emit_flush_input_cursor();

        // Call fn(ctx, start, prefix_len, &key_ptr, &key_len)
        // System V: rdi=ctx, rsi=start, rdx=prefix_len, rcx=&key_ptr, r8=&key_len
        #[cfg(not(windows))]
        self.emit
            .emit_with(|buf| {
                x64::encode_mov_r64_r64(7, 15, buf)?;
                x64::encode_mov_r64_r64(6, 10, buf)?;
                x64::encode_mov_r64_r64(2, 0, buf)?;
                x64::encode_lea_r64_m(
                    1,
                    Mem {
                        base: 4,
                        disp: key_ptr_sp_offset as i32,
                    },
                    buf,
                )?;
                x64::encode_lea_r64_m(
                    8,
                    Mem {
                        base: 4,
                        disp: key_len_sp_offset as i32,
                    },
                    buf,
                )
            })
            .expect("setup key_slow args");
        #[cfg(windows)]
        self.emit
            .emit_with(|buf| {
                x64::encode_mov_r64_r64(1, 15, buf)?;
                x64::encode_mov_r64_r64(2, 10, buf)?;
                x64::encode_mov_r64_r64(8, 0, buf)?;
                x64::encode_lea_r64_m(
                    9,
                    Mem {
                        base: 4,
                        disp: key_ptr_sp_offset as i32,
                    },
                    buf,
                )?;
                x64::encode_lea_r64_m(
                    0,
                    Mem {
                        base: 4,
                        disp: key_len_sp_offset as i32,
                    },
                    buf,
                )?;
                x64::encode_mov_m_r64(
                    Mem {
                        base: 4,
                        disp: 0x20,
                    },
                    0,
                    buf,
                )
            })
            .expect("setup key_slow args windows");

        self.emit_call_fn_ptr(fn_ptr);
        self.emit_reload_cursor_and_check_error();
    }

    // ── Enum support ──────────────────────────────────────────────────

    // r[impl deser.enum.set-variant]

    /// Write a discriminant value to [out + 0].
    /// `size` is 1, 2, 4, or 8 bytes (from EnumRepr).
    pub fn emit_write_discriminant(&mut self, value: i64, size: u32) {
        match size {
            1 => {
                let imm = value as u8;
                self.emit
                    .emit_with(|buf| {
                        x64::encode_mov_r32_imm32(0, imm as u32, buf)?;
                        x64::encode_mov_m_r8(14, 0, 0, buf)
                    })
                    .expect("write discr 1 byte");
            }
            2 => {
                let imm = value as u16;
                self.emit
                    .emit_with(|buf| {
                        x64::encode_mov_r32_imm32(0, imm as u32, buf)?;
                        x64::encode_mov_m_r16(14, 0, 0, buf)
                    })
                    .expect("write discr 2 bytes");
            }
            4 => {
                let imm = value as u32;
                self.emit
                    .emit_with(|buf| {
                        x64::encode_mov_r32_imm32(0, imm, buf)?;
                        x64::encode_mov_m_r32(Mem { base: 14, disp: 0 }, 0, buf)
                    })
                    .expect("write discr 4 bytes");
            }
            8 => {
                let val = value;
                self.emit
                    .emit_with(|buf| {
                        x64::encode_mov_r64_imm64(0, val as u64, buf)?;
                        x64::encode_mov_m_r64(Mem { base: 14, disp: 0 }, 0, buf)
                    })
                    .expect("write discr 8 bytes");
            }
            _ => panic!("unsupported discriminant size: {size}"),
        }
    }

    // r[impl deser.postcard.enum.dispatch]

    /// Read a postcard varint discriminant into r10d (kept in register for
    /// dispatch, not stored to memory).
    ///
    /// After this, the caller emits `emit_cmp_imm_branch_eq` for each variant.
    /// The discriminant value is in r10d.
    pub fn emit_read_postcard_discriminant(&mut self, slow_intrinsic: *const u8) {
        let eof_label = self.emit.new_label();
        let slow_path = self.emit.new_label();
        let done_label = self.emit.new_label();

        self.emit
            .emit_with(|buf| {
                x64::encode_cmp_r64_r64(12, 13, buf)?;
                Ok(())
            })
            .expect("compare input bounds");
        self.emit.emit_jae_label(eof_label).expect("eof");
        self.emit
            .emit_with(|buf| {
                x64::encode_movzx_r32_rm8(10, x64::Operand::Mem(Mem { base: 12, disp: 0 }), buf)?;
                x64::encode_mov_r64_imm64(11, 0x80, buf)?;
                x64::encode_test_r64_r64(10, 11, buf)?;
                Ok(())
            })
            .expect("test discr bit");
        self.emit.emit_jnz_label(slow_path).expect("slow path");
        self.emit
            .emit_with(|buf| x64::encode_add_r64_imm32(12, 1, buf))
            .expect("advance one byte");
        self.emit.emit_jmp_label(done_label).expect("done");
        self.emit.bind_label(slow_path).expect("bind slow");
        self.emit_flush_input_cursor();
        // arg0 = ctx, arg1 = pointer to temp u32 at BASE_FRAME (start of extra area)
        #[cfg(not(windows))]
        self.emit
            .emit_with(|buf| {
                x64::encode_mov_r64_r64(7, 15, buf)?;
                x64::encode_lea_r64_m(
                    6,
                    Mem {
                        base: 4,
                        disp: BASE_FRAME as i32,
                    },
                    buf,
                )
            })
            .expect("setup postcard discr args");
        #[cfg(windows)]
        self.emit
            .emit_with(|buf| {
                x64::encode_mov_r64_r64(1, 15, buf)?;
                x64::encode_lea_r64_m(
                    2,
                    Mem {
                        base: 4,
                        disp: BASE_FRAME as i32,
                    },
                    buf,
                )
            })
            .expect("setup postcard discr args");
        self.emit_call_fn_ptr(slow_intrinsic);
        let error_exit = self.error_exit;
        self.emit
            .emit_with(|buf| {
                x64::encode_mov_r64_m(
                    12,
                    Mem {
                        base: 15,
                        disp: CTX_INPUT_PTR as i32,
                    },
                    buf,
                )?;
                x64::encode_mov_r64_m(
                    11,
                    Mem {
                        base: 15,
                        disp: CTX_ERROR_CODE as i32,
                    },
                    buf,
                )?;
                x64::encode_cmp_r64_imm32(11, 0, buf)?;
                Ok(())
            })
            .expect("load postcard discr result");
        self.emit
            .emit_jne_label(error_exit)
            .expect("postcard discr error");
        self.emit
            .emit_with(|buf| {
                x64::encode_mov_r32_m(
                    10,
                    Mem {
                        base: 4,
                        disp: BASE_FRAME as i32,
                    },
                    buf,
                )
            })
            .expect("load discr");
        self.emit.emit_jmp_label(done_label).expect("done label");
        self.emit.bind_label(eof_label).expect("bind eof");
        self.emit
            .emit_with(|buf| {
                x64::encode_mov_r32_imm32(
                    10,
                    crate::context::ErrorCode::UnexpectedEof as u32,
                    buf,
                )?;
                x64::encode_mov_m_r32(
                    Mem {
                        base: 15,
                        disp: CTX_ERROR_CODE as i32,
                    },
                    10,
                    buf,
                )
            })
            .expect("write eof error");
        self.emit
            .emit_jmp_label(error_exit)
            .expect("branch eof error");
        self.emit.bind_label(done_label).expect("bind done");
    }

    /// Compare r10d (discriminant) with immediate `imm` and branch to `label`
    /// if equal.
    pub fn emit_cmp_imm_branch_eq(&mut self, imm: u32, label: LabelId) {
        self.emit
            .emit_with(|buf| x64::encode_cmp_r64_imm32(10, imm, buf))
            .expect("cmp imm branch");
        self.emit.emit_je_label(label).expect("branch eq");
    }

    /// Emit a branch-to-error for unknown variant (sets UnknownVariant error code).
    pub fn emit_unknown_variant_error(&mut self) {
        let error_exit = self.error_exit;
        let error_code = crate::context::ErrorCode::UnknownVariant as i32;

        self.emit
            .emit_with(|buf| {
                x64::encode_mov_r32_imm32(10, error_code as u32, buf)?;
                x64::encode_mov_m_r32(
                    Mem {
                        base: 15,
                        disp: CTX_ERROR_CODE as i32,
                    },
                    10,
                    buf,
                )
            })
            .expect("write unknown variant");
        self.emit
            .emit_jmp_label(error_exit)
            .expect("jump unknown variant");
    }

    /// Save the cached input_ptr (r12) to a stack slot.
    pub fn emit_save_input_ptr(&mut self, stack_offset: u32) {
        self.emit
            .emit_with(|buf| {
                x64::encode_mov_m_r64(
                    Mem {
                        base: 4,
                        disp: stack_offset as i32,
                    },
                    12,
                    buf,
                )
            })
            .expect("save input ptr");
    }

    /// Restore the cached input_ptr (r12) from a stack slot.
    pub fn emit_restore_input_ptr(&mut self, stack_offset: u32) {
        self.emit
            .emit_with(|buf| {
                x64::encode_mov_r64_m(
                    12,
                    Mem {
                        base: 4,
                        disp: stack_offset as i32,
                    },
                    buf,
                )
            })
            .expect("restore input ptr");
    }

    /// Store a 64-bit immediate into a stack slot at rsp + offset.
    pub fn emit_store_imm64_to_stack(&mut self, stack_offset: u32, value: u64) {
        self.emit
            .emit_with(|buf| {
                x64::encode_mov_r64_imm64(0, value, buf)?;
                x64::encode_mov_m_r64(
                    Mem {
                        base: 4,
                        disp: stack_offset as i32,
                    },
                    0,
                    buf,
                )
            })
            .expect("store imm64 stack");
    }

    /// AND a 64-bit immediate into a stack slot at rsp + offset.
    /// Loads the slot, ANDs with the immediate, stores back.
    pub fn emit_and_imm64_on_stack(&mut self, stack_offset: u32, mask: u64) {
        self.emit
            .emit_with(|buf| {
                x64::encode_mov_r64_m(
                    0,
                    Mem {
                        base: 4,
                        disp: stack_offset as i32,
                    },
                    buf,
                )?;
                x64::encode_mov_r64_imm64(10, mask, buf)?;
                x64::encode_and_r64_r64(0, 10, buf)?;
                x64::encode_mov_m_r64(
                    Mem {
                        base: 4,
                        disp: stack_offset as i32,
                    },
                    0,
                    buf,
                )
            })
            .expect("and imm64 stack");
    }

    /// Check if the stack slot at rsp + offset has exactly one bit set (popcount == 1).
    /// If so, branch to `label`.
    pub fn emit_popcount_eq1_branch(&mut self, stack_offset: u32, label: LabelId) {
        let skip = self.emit.new_label();

        self.emit
            .emit_with(|buf| {
                x64::encode_mov_r64_m(
                    0,
                    Mem {
                        base: 4,
                        disp: stack_offset as i32,
                    },
                    buf,
                )?;
                x64::encode_mov_r64_r64(10, 0, buf)?;
                x64::encode_sub_r64_imm32(10, 1, buf)?;
                x64::encode_mov_r64_r64(11, 0, buf)?;
                x64::encode_and_r64_r64(11, 10, buf)?;
                x64::encode_test_r64_r64(11, 11, buf)
            })
            .expect("popcount test");
        self.emit.emit_jne_label(skip).expect("skip non-popcount1");
        self.emit
            .emit_with(|buf| x64::encode_test_r64_r64(0, 0, buf))
            .expect("check nonzero");
        self.emit.emit_jne_label(label).expect("branch popcount1");
        self.emit.bind_label(skip).expect("bind skip");
    }

    /// Check if the stack slot at rsp + offset is zero. If so, branch to `label`.
    pub fn emit_stack_zero_branch(&mut self, stack_offset: u32, label: LabelId) {
        self.emit
            .emit_with(|buf| {
                x64::encode_mov_r64_m(
                    0,
                    Mem {
                        base: 4,
                        disp: stack_offset as i32,
                    },
                    buf,
                )
            })
            .expect("load slot for zero check");
        self.emit
            .emit_with(|buf| x64::encode_cmp_r64_imm32(0, 0, buf))
            .expect("compare zero");
        self.emit.emit_jz_label(label).expect("branch zero");
    }

    /// Load the stack slot at rsp + offset, test if bit `bit_index` is set,
    /// and branch to `label` if so.
    pub fn emit_test_bit_branch(&mut self, stack_offset: u32, bit_index: u32, label: LabelId) {
        let mask = (1u64 << bit_index) as i64;
        self.emit
            .emit_with(|buf| {
                x64::encode_mov_r64_m(
                    0,
                    Mem {
                        base: 4,
                        disp: stack_offset as i32,
                    },
                    buf,
                )?;
                x64::encode_mov_r64_imm64(10, mask as u64, buf)?;
                x64::encode_and_r64_r64(0, 10, buf)?;
                Ok(())
            })
            .expect("test bit");
        self.emit.emit_jnz_label(label).expect("branch bit set");
    }

    /// Test a single bit at `bit_index` in the u64 at `[rsp + stack_offset]`.
    /// Branch to `label` if the bit is CLEAR (zero) — i.e., the field was NOT seen.
    pub fn emit_test_bit_branch_zero(&mut self, stack_offset: u32, bit_index: u32, label: LabelId) {
        let mask = (1u64 << bit_index) as i64;
        self.emit
            .emit_with(|buf| {
                x64::encode_mov_r64_m(
                    0,
                    Mem {
                        base: 4,
                        disp: stack_offset as i32,
                    },
                    buf,
                )?;
                x64::encode_mov_r64_imm64(10, mask as u64, buf)?;
                x64::encode_and_r64_r64(0, 10, buf)?;
                Ok(())
            })
            .expect("test bit zero");
        self.emit.emit_jz_label(label).expect("branch bit clear");
    }

    /// Emit an error (write error code to ctx, jump to error_exit).
    pub fn emit_error(&mut self, code: crate::context::ErrorCode) {
        let error_exit = self.error_exit;
        let error_code = code as i32;
        self.emit
            .emit_with(|buf| {
                x64::encode_mov_r32_imm32(10, error_code as u32, buf)?;
                x64::encode_mov_m_r32(
                    Mem {
                        base: 15,
                        disp: CTX_ERROR_CODE as i32,
                    },
                    10,
                    buf,
                )
            })
            .expect("write error code");
        self.emit.emit_jmp_label(error_exit).expect("jump error");
    }

    /// Advance the cached cursor by n bytes (inline, no function call).
    pub fn emit_advance_cursor_by(&mut self, n: u32) {
        self.emit
            .emit_with(|buf| x64::encode_add_r64_imm32(12, n, buf))
            .expect("advance cursor");
    }

    // r[impl deser.json.struct]
    // r[impl deser.json.map]
    /// Inline: skip JSON whitespace (space/tab/LF/CR) then consume the expected byte.
    /// Sets `error_code` and branches to the current error_exit if EOF or wrong byte.
    ///
    /// Register contract: r12 = cached input_ptr, r13 = cached input_end, r15 = ctx.
    /// Scratch: r10 (raw byte), r11 (loop temp — clobbered).
    pub fn emit_expect_byte_after_ws(&mut self, expected: u8, error_code: ErrorCode) {
        let error_exit = self.error_exit;
        let err_code = error_code as i32;
        let expected = expected as i8;

        let ws_loop = self.emit.new_label();
        let ws_next = self.emit.new_label();
        let non_ws = self.emit.new_label();
        let done = self.emit.new_label();
        let err_lbl = self.emit.new_label();

        self.emit.bind_label(ws_loop).expect("bind ws_loop");
        self.emit
            .emit_with(|buf| x64::encode_cmp_r64_r64(12, 13, buf))
            .expect("expect ws bounds");
        self.emit.emit_jge_label(err_lbl).expect("error eof");
        self.emit
            .emit_with(|buf| {
                x64::encode_movzx_r32_rm8(10, x64::Operand::Mem(Mem { base: 12, disp: 0 }), buf)?;
                x64::encode_cmp_r64_imm32(10, b' ' as u32, buf)?;
                Ok(())
            })
            .expect("check ws space");
        self.emit.emit_je_label(ws_next).expect("space");
        self.emit
            .emit_with(|buf| x64::encode_cmp_r64_imm32(10, b'\t' as u32, buf))
            .expect("check ws tab");
        self.emit.emit_je_label(ws_next).expect("tab");
        self.emit
            .emit_with(|buf| x64::encode_cmp_r64_imm32(10, b'\n' as u32, buf))
            .expect("check ws lf");
        self.emit.emit_je_label(ws_next).expect("lf");
        self.emit
            .emit_with(|buf| x64::encode_cmp_r64_imm32(10, b'\r' as u32, buf))
            .expect("check ws cr");
        self.emit.emit_jne_label(non_ws).expect("non_ws");
        self.emit.bind_label(ws_next).expect("bind ws_next");
        self.emit
            .emit_with(|buf| x64::encode_add_r64_imm32(12, 1, buf))
            .expect("consume ws");
        self.emit.emit_jmp_label(ws_loop).expect("loop ws");
        self.emit.bind_label(non_ws).expect("bind non_ws");
        self.emit
            .emit_with(|buf| x64::encode_cmp_r64_imm32(10, expected as u32, buf))
            .expect("check expected");
        self.emit
            .emit_jne_label(err_lbl)
            .expect("expected mismatch");
        self.emit
            .emit_with(|buf| x64::encode_add_r64_imm32(12, 1, buf))
            .expect("consume expected");
        self.emit.emit_jmp_label(done).expect("done");
        self.emit.bind_label(err_lbl).expect("bind err_lbl");
        self.emit
            .emit_with(|buf| {
                x64::encode_mov_r32_imm32(10, err_code as u32, buf)?;
                x64::encode_mov_m_r32(
                    Mem {
                        base: 15,
                        disp: CTX_ERROR_CODE as i32,
                    },
                    10,
                    buf,
                )
            })
            .expect("write parse error");
        self.emit.emit_jmp_label(error_exit).expect("error branch");
        self.emit.bind_label(done).expect("bind done");
    }

    // ── Option support ────────────────────────────────────────────────

    /// Save the current `out` pointer (r14) and redirect it to a stack scratch area.
    pub fn emit_redirect_out_to_stack(&mut self, scratch_offset: u32) {
        self.emit
            .emit_with(|buf| {
                x64::encode_mov_m_r64(
                    Mem {
                        base: 4,
                        disp: scratch_offset as i32 - 8,
                    },
                    14,
                    buf,
                )?;
                x64::encode_lea_r64_m(
                    14,
                    Mem {
                        base: 4,
                        disp: scratch_offset as i32,
                    },
                    buf,
                )
            })
            .expect("redirect output");
    }

    /// Restore the `out` pointer (r14) from the saved slot.
    pub fn emit_restore_out(&mut self, scratch_offset: u32) {
        self.emit
            .emit_with(|buf| {
                x64::encode_mov_r64_m(
                    14,
                    Mem {
                        base: 4,
                        disp: scratch_offset as i32 - 8,
                    },
                    buf,
                )
            })
            .expect("restore output");
    }

    /// Call kajit_option_init_none(init_none_fn, out + offset).
    /// Does not touch ctx or the cursor.
    pub fn emit_call_option_init_none(
        &mut self,
        wrapper_fn: *const u8,
        init_none_fn: *const u8,
        offset: u32,
    ) {
        let init_none_val = init_none_fn as i64;

        // arg0: init_none_fn pointer, arg1: out + offset
        #[cfg(not(windows))]
        self.emit
            .emit_with(|buf| {
                x64::encode_mov_r64_imm64(7, init_none_val as u64, buf)?;
                x64::encode_lea_r64_m(
                    6,
                    Mem {
                        base: 14,
                        disp: offset as i32,
                    },
                    buf,
                )
            })
            .expect("setup init none");
        #[cfg(windows)]
        self.emit
            .emit_with(|buf| {
                x64::encode_mov_r64_imm64(1, init_none_val as u64, buf)?;
                x64::encode_lea_r64_m(
                    2,
                    Mem {
                        base: 14,
                        disp: offset as i32,
                    },
                    buf,
                )
            })
            .expect("setup init none");
        self.emit_call_fn_ptr(wrapper_fn);
    }

    /// Call wrapper(fn_ptr, out + offset, extra_ptr).
    /// Used for `kajit_field_default_indirect(default_fn, out, shape)`.
    pub fn emit_call_trampoline_3(
        &mut self,
        wrapper_fn: *const u8,
        fn_ptr: *const u8,
        offset: u32,
        extra_ptr: *const u8,
    ) {
        let fn_val = fn_ptr as i64;
        let extra_val = extra_ptr as i64;

        // arg0: fn_ptr, arg1: out + offset, arg2: extra_ptr
        #[cfg(not(windows))]
        self.emit
            .emit_with(|buf| {
                x64::encode_mov_r64_imm64(7, fn_val as u64, buf)?;
                x64::encode_lea_r64_m(
                    6,
                    Mem {
                        base: 14,
                        disp: offset as i32,
                    },
                    buf,
                )?;
                x64::encode_mov_r64_imm64(2, extra_val as u64, buf)
            })
            .expect("setup trampoline3");
        #[cfg(windows)]
        self.emit
            .emit_with(|buf| {
                x64::encode_mov_r64_imm64(1, fn_val as u64, buf)?;
                x64::encode_lea_r64_m(
                    2,
                    Mem {
                        base: 14,
                        disp: offset as i32,
                    },
                    buf,
                )?;
                x64::encode_mov_r64_imm64(8, extra_val as u64, buf)
            })
            .expect("setup trampoline3");
        self.emit_call_fn_ptr(wrapper_fn);
    }

    /// Call kajit_option_init_some(init_some_fn, out + offset, sp + scratch_offset).
    /// Does not touch ctx or the cursor.
    pub fn emit_call_option_init_some(
        &mut self,
        wrapper_fn: *const u8,
        init_some_fn: *const u8,
        offset: u32,
        scratch_offset: u32,
    ) {
        let init_some_val = init_some_fn as i64;

        // arg0: init_some_fn pointer, arg1: out + offset, arg2: scratch area on stack
        #[cfg(not(windows))]
        self.emit
            .emit_with(|buf| {
                x64::encode_mov_r64_imm64(7, init_some_val as u64, buf)?;
                x64::encode_lea_r64_m(
                    6,
                    Mem {
                        base: 14,
                        disp: offset as i32,
                    },
                    buf,
                )?;
                x64::encode_lea_r64_m(
                    2,
                    Mem {
                        base: 4,
                        disp: scratch_offset as i32,
                    },
                    buf,
                )
            })
            .expect("setup init some");
        #[cfg(windows)]
        self.emit
            .emit_with(|buf| {
                x64::encode_mov_r64_imm64(1, init_some_val as u64, buf)?;
                x64::encode_lea_r64_m(
                    2,
                    Mem {
                        base: 14,
                        disp: offset as i32,
                    },
                    buf,
                )?;
                x64::encode_lea_r64_m(
                    8,
                    Mem {
                        base: 4,
                        disp: scratch_offset as i32,
                    },
                    buf,
                )
            })
            .expect("setup init some");
        self.emit_call_fn_ptr(wrapper_fn);
    }

    // ── Vec support ──────────────────────────────────────────────────

    /// Call kajit_vec_alloc with a constant count (for JSON initial allocation).
    ///
    /// Result (buf pointer) is in rax.
    pub fn emit_call_json_vec_initial_alloc(
        &mut self,
        alloc_fn: *const u8,
        count: u32,
        elem_size: u32,
        elem_align: u32,
    ) {
        self.emit_flush_input_cursor();
        // args: ctx, count, elem_size, elem_align
        #[cfg(not(windows))]
        self.emit
            .emit_with(|buf| {
                x64::encode_mov_r64_r64(7, 15, buf)?;
                x64::encode_mov_r32_imm32(6, count as u32, buf)?;
                x64::encode_mov_r32_imm32(2, elem_size as u32, buf)?;
                x64::encode_mov_r32_imm32(1, elem_align as u32, buf)
            })
            .expect("setup json vec alloc");
        #[cfg(windows)]
        self.emit
            .emit_with(|buf| {
                x64::encode_mov_r64_r64(1, 15, buf)?;
                x64::encode_mov_r32_imm32(2, count as u32, buf)?;
                x64::encode_mov_r32_imm32(8, elem_size as u32, buf)?;
                x64::encode_mov_r32_imm32(9, elem_align as u32, buf)
            })
            .expect("setup json vec alloc");
        self.emit_call_fn_ptr(alloc_fn);
        self.emit_reload_cursor_and_check_error();
    }

    /// Initialize JSON Vec loop state: buf from rax, len=0, cap=initial_cap.
    pub fn emit_json_vec_loop_init(
        &mut self,
        saved_out_slot: u32,
        buf_slot: u32,
        len_slot: u32,
        cap_slot: u32,
        initial_cap: u32,
    ) {
        self.emit
            .emit_with(|buf| {
                x64::encode_mov_m_r64(
                    Mem {
                        base: 4,
                        disp: saved_out_slot as i32,
                    },
                    14,
                    buf,
                )?;
                x64::encode_mov_m_r64(
                    Mem {
                        base: 4,
                        disp: buf_slot as i32,
                    },
                    0,
                    buf,
                )?;
                x64::encode_mov_r64_imm64(10, 0, buf)?;
                x64::encode_mov_m_r64(
                    Mem {
                        base: 4,
                        disp: len_slot as i32,
                    },
                    10,
                    buf,
                )?;
                x64::encode_mov_r32_imm32(11, initial_cap as u32, buf)?;
                x64::encode_mov_m_r32(
                    Mem {
                        base: 4,
                        disp: cap_slot as i32,
                    },
                    11,
                    buf,
                )?;
                x64::encode_mov_r32_imm32(11, 0, buf)?;
                x64::encode_mov_m_r32(
                    Mem {
                        base: 4,
                        disp: cap_slot as i32 + 4,
                    },
                    11,
                    buf,
                )
            })
            .expect("init json vec loop");
    }

    /// Check ctx.error.code and branch to label if nonzero.
    pub fn emit_check_error_branch(&mut self, label: LabelId) {
        self.emit
            .emit_with(|buf| {
                x64::encode_mov_r32_m(
                    10,
                    Mem {
                        base: 15,
                        disp: CTX_ERROR_CODE as i32,
                    },
                    buf,
                )?;
                x64::encode_test_r32_r32(10, 10, buf)
            })
            .expect("check error branch");
        self.emit.emit_jnz_label(label).expect("check error branch");
    }

    /// Save the count register (w9 on aarch64, r10d on x64) to a stack slot.
    ///
    /// Used to preserve the count across a function call (r10 is caller-saved).
    pub fn emit_save_count_to_stack(&mut self, slot: u32) {
        self.emit
            .emit_with(|buf| {
                x64::encode_mov_m_r64(
                    Mem {
                        base: 4,
                        disp: slot as i32,
                    },
                    10,
                    buf,
                )
            })
            .expect("save count to stack");
    }

    /// Call kajit_vec_alloc(ctx, count, elem_size, elem_align).
    ///
    /// count is in r10d (from emit_read_postcard_discriminant or JSON parse).
    /// Result (buf pointer) is in rax.
    ///
    /// **Important**: r10 is caller-saved and will be clobbered by the call.
    /// The count must be saved to a stack slot before calling this.
    pub fn emit_call_vec_alloc(&mut self, alloc_fn: *const u8, elem_size: u32, elem_align: u32) {
        self.emit_flush_input_cursor();
        // args: ctx, count(r10), elem_size, elem_align
        #[cfg(not(windows))]
        self.emit
            .emit_with(|buf| {
                x64::encode_mov_r64_r64(7, 15, buf)?;
                x64::encode_mov_r64_r64(6, 10, buf)?;
                x64::encode_mov_r32_imm32(2, elem_size as u32, buf)?;
                x64::encode_mov_r32_imm32(1, elem_align as u32, buf)
            })
            .expect("setup vec alloc");
        #[cfg(windows)]
        self.emit
            .emit_with(|buf| {
                x64::encode_mov_r64_r64(1, 15, buf)?;
                x64::encode_mov_r64_r64(2, 10, buf)?;
                x64::encode_mov_r32_imm32(8, elem_size as u32, buf)?;
                x64::encode_mov_r32_imm32(9, elem_align as u32, buf)
            })
            .expect("setup vec alloc");
        self.emit_call_fn_ptr(alloc_fn);
        self.emit_reload_cursor_and_check_error();
    }

    /// Call kajit_vec_grow(ctx, old_buf, len, old_cap, new_cap, elem_size, elem_align).
    ///
    /// Reads old_buf, len, old_cap from stack slots. Computes new_cap = old_cap * 2.
    /// After call: new buf pointer is in rax.
    pub fn emit_call_vec_grow(
        &mut self,
        grow_fn: *const u8,
        buf_slot: u32,
        len_slot: u32,
        cap_slot: u32,
        elem_size: u32,
        elem_align: u32,
    ) {
        self.emit_flush_input_cursor();

        #[cfg(not(windows))]
        {
            // System V AMD64: 6 register args + 7th on the stack via push.
            // Args: rdi=ctx, rsi=old_buf, rdx=len, rcx=old_cap, r8=new_cap, r9=elem_size
            // 7th arg (elem_align) pushed before call.
            self.emit
                .emit_with(|buf| {
                    x64::encode_mov_r64_m(
                        10,
                        Mem {
                            base: 4,
                            disp: cap_slot as i32,
                        },
                        buf,
                    )?;
                    x64::encode_shl_r64_imm8(10, 1, buf)?;
                    x64::encode_mov_r64_r64(7, 15, buf)?;
                    x64::encode_mov_r64_m(
                        6,
                        Mem {
                            base: 4,
                            disp: buf_slot as i32,
                        },
                        buf,
                    )?;
                    x64::encode_mov_r64_m(
                        2,
                        Mem {
                            base: 4,
                            disp: len_slot as i32,
                        },
                        buf,
                    )?;
                    x64::encode_mov_r64_m(
                        1,
                        Mem {
                            base: 4,
                            disp: cap_slot as i32,
                        },
                        buf,
                    )?;
                    x64::encode_mov_r64_r64(8, 10, buf)?;
                    x64::encode_mov_r32_imm32(9, elem_size as u32, buf)?;
                    x64::encode_mov_r32_imm32(11, elem_align as u32, buf)?;
                    x64::encode_push_r64(11, buf)
                })
                .expect("setup vec grow");
            self.emit_call_fn_ptr(grow_fn);
            self.emit
                .emit_with(|buf| x64::encode_add_r64_imm32(4, 8, buf))
                .expect("restore vec grow stack");
        }

        #[cfg(windows)]
        {
            // Windows x64: 4 register args + args 5-7 on the stack.
            // The shadow space (32 bytes) and 3 stack args (24 bytes) = 56 bytes.
            // Round up to 64 for 16-byte alignment before `call`.
            //
            // Load all values into registers BEFORE sub rsp (frame offsets change after).
            // Args: rcx=ctx, rdx=old_buf, r8=len, r9=old_cap
            // Stack: [rsp+32]=new_cap, [rsp+40]=elem_size, [rsp+48]=elem_align
            self.emit
                .emit_with(|buf| {
                    x64::encode_mov_r64_m(
                        10,
                        Mem {
                            base: 4,
                            disp: cap_slot as i32,
                        },
                        buf,
                    )?;
                    x64::encode_shl_r64_imm8(10, 1, buf)?;
                    x64::encode_mov_r64_r64(1, 15, buf)?;
                    x64::encode_mov_r64_m(
                        2,
                        Mem {
                            base: 4,
                            disp: buf_slot as i32,
                        },
                        buf,
                    )?;
                    x64::encode_mov_r64_m(
                        8,
                        Mem {
                            base: 4,
                            disp: len_slot as i32,
                        },
                        buf,
                    )?;
                    x64::encode_mov_r64_m(
                        9,
                        Mem {
                            base: 4,
                            disp: cap_slot as i32,
                        },
                        buf,
                    )?;
                    x64::encode_sub_r64_imm32(4, 64, buf)?;
                    x64::encode_mov_m_r64(Mem { base: 4, disp: 32 }, 10, buf)?;
                    x64::encode_mov_r32_imm32(11, elem_size as u32, buf)?;
                    x64::encode_mov_m_r64(Mem { base: 4, disp: 40 }, 11, buf)?;
                    x64::encode_mov_r32_imm32(11, elem_align as u32, buf)?;
                    x64::encode_mov_m_r64(Mem { base: 4, disp: 48 }, 11, buf)
                })
                .expect("setup vec grow");
            self.emit_call_fn_ptr(grow_fn);
            self.emit
                .emit_with(|buf| x64::encode_add_r64_imm32(4, 64, buf))
                .expect("restore vec grow stack");
        }

        self.emit_reload_cursor_and_check_error();
        self.emit
            .emit_with(|buf| {
                x64::encode_mov_m_r64(
                    Mem {
                        base: 4,
                        disp: buf_slot as i32,
                    },
                    0,
                    buf,
                )?;
                x64::encode_mov_r64_m(
                    10,
                    Mem {
                        base: 4,
                        disp: cap_slot as i32,
                    },
                    buf,
                )?;
                x64::encode_shl_r64_imm8(10, 1, buf)?;
                x64::encode_mov_m_r64(
                    Mem {
                        base: 4,
                        disp: cap_slot as i32,
                    },
                    10,
                    buf,
                )
            })
            .expect("update vec cap");
    }

    /// Initialize Vec loop state after allocation.
    ///
    /// Saves: out pointer, buf (alloc result in rax), counter=0.
    /// Count must already be stored at count_slot (via emit_save_r10_to_stack).
    /// Initialize Vec loop state with cursor in rbx (callee-saved), end on stack.
    ///
    /// Saves rbx to stack, then sets rbx = buf (cursor), end_slot = buf + count * elem_size.
    /// Also saves out pointer and buf for final Vec store.
    pub fn emit_vec_loop_init_cursor(
        &mut self,
        saved_out_slot: u32,
        buf_slot: u32,
        count_slot: u32,
        save_rbx_slot: u32,
        end_slot: u32,
        elem_size: u32,
    ) {
        self.emit
            .emit_with(|buf| {
                x64::encode_mov_m_r64(
                    Mem {
                        base: 4,
                        disp: saved_out_slot as i32,
                    },
                    14,
                    buf,
                )?;
                x64::encode_mov_m_r64(
                    Mem {
                        base: 4,
                        disp: buf_slot as i32,
                    },
                    0,
                    buf,
                )?;
                x64::encode_mov_m_r64(
                    Mem {
                        base: 4,
                        disp: save_rbx_slot as i32,
                    },
                    3,
                    buf,
                )?;
                x64::encode_mov_r64_r64(3, 0, buf)?;
                x64::encode_mov_r64_m(
                    10,
                    Mem {
                        base: 4,
                        disp: count_slot as i32,
                    },
                    buf,
                )?;
                x64::encode_imul_r64_r64_imm32(10, 10, elem_size as u32, buf)?;
                x64::encode_add_r64_r64(10, 0, buf)?;
                x64::encode_mov_m_r64(
                    Mem {
                        base: 4,
                        disp: end_slot as i32,
                    },
                    10,
                    buf,
                )
            })
            .expect("init vec loop cursor");
    }

    /// Set out = cursor (r14 = rbx). Single register move.
    pub fn emit_vec_loop_load_cursor(&mut self, _save_rbx_slot: u32) {
        self.emit
            .emit_with(|buf| x64::encode_mov_r64_r64(14, 3, buf))
            .expect("set out cursor");
    }

    /// Advance cursor register, check error, branch back if cursor < end.
    /// Cursor advance is register-only; end compare reads from stack (x64 can cmp reg, [mem]).
    pub fn emit_vec_loop_advance_cursor(
        &mut self,
        _save_rbx_slot: u32,
        end_slot: u32,
        elem_size: u32,
        loop_label: LabelId,
        error_cleanup_label: LabelId,
    ) {
        self.emit
            .emit_with(|buf| {
                x64::encode_mov_r32_m(
                    10,
                    Mem {
                        base: 15,
                        disp: CTX_ERROR_CODE as i32,
                    },
                    buf,
                )?;
                x64::encode_test_r32_r32(10, 10, buf)
            })
            .expect("check vec loop error");
        self.emit
            .emit_jnz_label(error_cleanup_label)
            .expect("vec loop error");
        self.emit
            .emit_with(|buf| {
                x64::encode_add_r64_imm32(3, elem_size as u32, buf)?;
                x64::encode_cmp_r64_m(
                    3,
                    Mem {
                        base: 4,
                        disp: end_slot as i32,
                    },
                    buf,
                )
            })
            .expect("advance vec loop cursor");
        self.emit
            .emit_jbe_label(loop_label)
            .expect("vec loop check");
    }

    /// Advance the cursor register and loop back, without checking the error flag.
    ///
    /// Use this when all error paths within the loop body branch directly to
    /// the error cleanup label (e.g. via redirected `error_exit`), making
    /// the per-iteration error check redundant.
    pub fn emit_vec_loop_advance_no_error_check(
        &mut self,
        end_slot: u32,
        elem_size: u32,
        loop_label: LabelId,
    ) {
        self.emit
            .emit_with(|buf| {
                x64::encode_add_r64_imm32(3, elem_size as u32, buf)?;
                x64::encode_cmp_r64_m(
                    3,
                    Mem {
                        base: 4,
                        disp: end_slot as i32,
                    },
                    buf,
                )
            })
            .expect("advance vec loop cursor");
        self.emit
            .emit_jbe_label(loop_label)
            .expect("vec loop check");
    }

    /// Emit a tight varint Vec loop body for x64. Writes directly to `[rbx]`
    /// (the cursor register). The slow path and EOF are placed out-of-line.
    pub fn emit_vec_varint_loop(
        &mut self,
        store_width: u32,
        zigzag: bool,
        intrinsic_fn_ptr: *const u8,
        elem_size: u32,
        end_slot: u32,
        loop_label: LabelId,
        done_label: LabelId,
        error_cleanup: LabelId,
    ) {
        let slow_path = self.emit.new_label();
        let eof_label = self.emit.new_label();

        // === Hot loop ===
        self.emit
            .emit_with(|buf| {
                x64::encode_cmp_r64_r64(12, 13, buf)?;
                x64::encode_movzx_r32_rm8(10, x64::Operand::Mem(Mem { base: 12, disp: 0 }), buf)?;
                x64::encode_add_r64_imm32(12, 1, buf)?;
                x64::encode_mov_r32_imm32(11, 0x80, buf)?;
                x64::encode_test_r32_r32(10, 11, buf)
            })
            .expect("vec varint hot-loop");
        self.emit
            .emit_jne_label(slow_path)
            .expect("vec varint slow path");
        self.emit.emit_jae_label(eof_label).expect("vec varint eof");

        if zigzag {
            self.emit
                .emit_with(|buf| {
                    x64::encode_mov_r32_r32(11, 10, buf)?;
                    x64::encode_shr_r64_imm8(11, 1, buf)?;
                    x64::encode_mov_r32_imm32(2, 1, buf)?;
                    x64::encode_and_r64_r64(10, 2, buf)?;
                    x64::encode_neg_r64(10, buf)?;
                    x64::encode_xor_r64_r64(10, 11, buf)
                })
                .expect("vec varint zigzag");
        }

        // Store directly to cursor (rbx)
        match store_width {
            2 => self
                .emit
                .emit_with(|buf| x64::encode_mov_m_r16(3, 0, 10, buf))
                .expect("store varint width2"),
            4 => self
                .emit
                .emit_with(|buf| x64::encode_mov_m_r32(Mem { base: 3, disp: 0 }, 10, buf))
                .expect("store varint width4"),
            8 => self
                .emit
                .emit_with(|buf| x64::encode_mov_m_r64(Mem { base: 3, disp: 0 }, 10, buf))
                .expect("store varint width8"),
            _ => panic!("unsupported varint store width: {store_width}"),
        }

        // Advance cursor, loop back
        self.emit
            .emit_with(|buf| {
                x64::encode_add_r64_imm32(3, elem_size as u32, buf)?;
                x64::encode_cmp_r64_m(
                    3,
                    Mem {
                        base: 4,
                        disp: end_slot as i32,
                    },
                    buf,
                )
            })
            .expect("advance vec loop cursor");
        self.emit
            .emit_jbe_label(loop_label)
            .expect("vec varint loop");
        self.emit
            .emit_jmp_label(done_label)
            .expect("vec varint done");

        // === Slow path (out-of-line) ===
        self.emit
            .bind_label(slow_path)
            .expect("bind vec varint slow");
        self.emit
            .emit_with(|buf| x64::encode_sub_r64_imm32(12, 1, buf))
            .expect("undo varint increment");
        self.emit_flush_input_cursor();
        // arg0 = ctx, arg1 = out (cursor in rbx)
        #[cfg(not(windows))]
        self.emit
            .emit_with(|buf| {
                x64::encode_mov_r64_r64(7, 15, buf)?;
                x64::encode_mov_r64_r64(6, 3, buf)
            })
            .expect("setup vec varint slow call");
        #[cfg(windows)]
        self.emit
            .emit_with(|buf| {
                x64::encode_mov_r64_r64(1, 15, buf)?;
                x64::encode_mov_r64_r64(2, 3, buf)
            })
            .expect("setup vec varint slow call");
        self.emit_call_fn_ptr(intrinsic_fn_ptr);
        // Reload input pointer and check error (branches to error_cleanup, not error_exit)
        self.emit
            .emit_with(|buf| {
                x64::encode_mov_r64_m(
                    12,
                    Mem {
                        base: 15,
                        disp: CTX_INPUT_PTR as i32,
                    },
                    buf,
                )?;
                x64::encode_mov_r32_m(
                    10,
                    Mem {
                        base: 15,
                        disp: CTX_ERROR_CODE as i32,
                    },
                    buf,
                )?;
                x64::encode_test_r32_r32(10, 10, buf)
            })
            .expect("vec varint slow reload");
        self.emit
            .emit_jnz_label(error_cleanup)
            .expect("vec varint slow error");
        self.emit
            .emit_with(|buf| {
                x64::encode_add_r64_imm32(3, elem_size as u32, buf)?;
                x64::encode_cmp_r64_m(
                    3,
                    Mem {
                        base: 4,
                        disp: end_slot as i32,
                    },
                    buf,
                )
            })
            .expect("vec varint slow advance");
        self.emit
            .emit_jbe_label(loop_label)
            .expect("vec varint continue");
        self.emit
            .emit_jmp_label(done_label)
            .expect("vec varint slow done");

        // === EOF (cold) ===
        self.emit
            .bind_label(eof_label)
            .expect("bind vec varint eof");
        self.emit
            .emit_with(|buf| {
                x64::encode_mov_r32_imm32(
                    10,
                    crate::context::ErrorCode::UnexpectedEof as u32,
                    buf,
                )?;
                x64::encode_mov_m_r32(
                    Mem {
                        base: 15,
                        disp: CTX_ERROR_CODE as i32,
                    },
                    10,
                    buf,
                )
            })
            .expect("set vec varint eof error");
        self.emit
            .emit_jmp_label(error_cleanup)
            .expect("vec varint eof cleanup");
    }

    /// Restore rbx from stack. Must be called on every exit path from a Vec loop.
    pub fn emit_vec_restore_callee_saved(&mut self, save_rbx_slot: u32, _end_slot: u32) {
        self.emit
            .emit_with(|buf| {
                x64::encode_mov_r64_m(
                    3,
                    Mem {
                        base: 4,
                        disp: save_rbx_slot as i32,
                    },
                    buf,
                )
            })
            .expect("restore vec callee");
    }

    /// Emit Vec loop header: compute slot = buf + i * elem_size, set out = slot.
    ///
    /// Used by JSON where buf can change on growth and index-based access is needed.
    pub fn emit_vec_loop_slot(&mut self, buf_slot: u32, counter_slot: u32, elem_size: u32) {
        self.emit
            .emit_with(|buf| {
                x64::encode_mov_r64_m(
                    14,
                    Mem {
                        base: 4,
                        disp: buf_slot as i32,
                    },
                    buf,
                )?;
                x64::encode_mov_r64_m(
                    10,
                    Mem {
                        base: 4,
                        disp: counter_slot as i32,
                    },
                    buf,
                )?;
                x64::encode_imul_r64_r64_imm32(10, 10, elem_size as u32, buf)?;
                x64::encode_add_r64_r64(14, 10, buf)
            })
            .expect("vec loop slot");
    }

    /// Write Vec fields (ptr, len, cap) to out + base_offset using discovered offsets.
    /// Reads buf and count from stack slots.
    pub fn emit_vec_store(
        &mut self,
        base_offset: u32,
        saved_out_slot: u32,
        buf_slot: u32,
        len_slot: u32,
        cap_slot: u32,
        offsets: &crate::malum::VecOffsets,
    ) {
        let ptr_off = (base_offset + offsets.ptr_offset) as i32;
        let len_off = (base_offset + offsets.len_offset) as i32;
        let cap_off = (base_offset + offsets.cap_offset) as i32;

        self.emit
            .emit_with(|buf| {
                x64::encode_mov_r64_m(
                    14,
                    Mem {
                        base: 4,
                        disp: saved_out_slot as i32,
                    },
                    buf,
                )?;
                x64::encode_mov_r64_m(
                    0,
                    Mem {
                        base: 4,
                        disp: buf_slot as i32,
                    },
                    buf,
                )?;
                x64::encode_mov_m_r64(
                    Mem {
                        base: 14,
                        disp: ptr_off,
                    },
                    0,
                    buf,
                )?;
                x64::encode_mov_r64_m(
                    0,
                    Mem {
                        base: 4,
                        disp: len_slot as i32,
                    },
                    buf,
                )?;
                x64::encode_mov_m_r64(
                    Mem {
                        base: 14,
                        disp: len_off,
                    },
                    0,
                    buf,
                )?;
                x64::encode_mov_r64_m(
                    0,
                    Mem {
                        base: 4,
                        disp: cap_slot as i32,
                    },
                    buf,
                )?;
                x64::encode_mov_m_r64(
                    Mem {
                        base: 14,
                        disp: cap_off,
                    },
                    0,
                    buf,
                )
            })
            .expect("store vec fields");
    }

    /// Write an empty Vec to out + base_offset with proper dangling pointer.
    pub fn emit_vec_store_empty_with_align(
        &mut self,
        base_offset: u32,
        elem_align: u32,
        offsets: &crate::malum::VecOffsets,
    ) {
        let ptr_off = (base_offset + offsets.ptr_offset) as i32;
        let len_off = (base_offset + offsets.len_offset) as i32;
        let cap_off = (base_offset + offsets.cap_offset) as i32;

        // Vec::new() writes: ptr = NonNull::dangling() = elem_align as *mut T, len = 0, cap = 0.
        self.emit
            .emit_with(|buf| {
                x64::encode_mov_r64_imm64(10, elem_align as u64, buf)?;
                x64::encode_mov_m_r64(
                    Mem {
                        base: 14,
                        disp: ptr_off,
                    },
                    10,
                    buf,
                )?;
                x64::encode_mov_r64_imm64(10, 0, buf)?;
                x64::encode_mov_m_r64(
                    Mem {
                        base: 14,
                        disp: len_off,
                    },
                    10,
                    buf,
                )?;
                x64::encode_mov_m_r64(
                    Mem {
                        base: 14,
                        disp: cap_off,
                    },
                    10,
                    buf,
                )
            })
            .expect("store empty vec");
    }

    /// Emit error cleanup for Vec: free the buffer and branch to error exit.
    /// Called when element deserialization fails mid-loop.
    pub fn emit_vec_error_cleanup(
        &mut self,
        free_fn: *const u8,
        saved_out_slot: u32,
        buf_slot: u32,
        cap_slot: u32,
        elem_size: u32,
        elem_align: u32,
    ) {
        let error_exit = self.error_exit;

        self.emit
            .emit_with(|buf| {
                x64::encode_mov_r64_m(
                    14,
                    Mem {
                        base: 4,
                        disp: saved_out_slot as i32,
                    },
                    buf,
                )
            })
            .expect("load saved out");
        // args: buf, cap, elem_size, elem_align
        #[cfg(not(windows))]
        self.emit
            .emit_with(|buf| {
                x64::encode_mov_r64_m(
                    7,
                    Mem {
                        base: 4,
                        disp: buf_slot as i32,
                    },
                    buf,
                )?;
                x64::encode_mov_r64_m(
                    6,
                    Mem {
                        base: 4,
                        disp: cap_slot as i32,
                    },
                    buf,
                )?;
                x64::encode_mov_r32_imm32(2, elem_size as u32, buf)?;
                x64::encode_mov_r32_imm32(1, elem_align as u32, buf)
            })
            .expect("setup vec free args");
        #[cfg(windows)]
        self.emit
            .emit_with(|buf| {
                x64::encode_mov_r64_m(
                    1,
                    Mem {
                        base: 4,
                        disp: buf_slot as i32,
                    },
                    buf,
                )?;
                x64::encode_mov_r64_m(
                    2,
                    Mem {
                        base: 4,
                        disp: cap_slot as i32,
                    },
                    buf,
                )?;
                x64::encode_mov_r32_imm32(8, elem_size as u32, buf)?;
                x64::encode_mov_r32_imm32(9, elem_align as u32, buf)
            })
            .expect("setup vec free args");
        self.emit_call_fn_ptr(free_fn);
        self.emit
            .emit_jmp_label(error_exit)
            .expect("vec error cleanup");
    }

    /// Compare the count register (w9 on aarch64, r10d on x64) with zero
    /// and branch to label if equal.
    pub fn emit_cbz_count(&mut self, label: LabelId) {
        self.emit
            .emit_with(|buf| x64::encode_test_r32_r32(10, 10, buf))
            .expect("check count zero");
        self.emit.emit_jz_label(label).expect("count zero");
    }

    /// Compare two stack slot values and branch if equal (len == cap for growth check).
    pub fn emit_cmp_stack_slots_branch_eq(&mut self, slot_a: u32, slot_b: u32, label: LabelId) {
        self.emit
            .emit_with(|buf| {
                x64::encode_mov_r64_m(
                    0,
                    Mem {
                        base: 4,
                        disp: slot_a as i32,
                    },
                    buf,
                )?;
                x64::encode_cmp_r64_m(
                    0,
                    Mem {
                        base: 4,
                        disp: slot_b as i32,
                    },
                    buf,
                )
            })
            .expect("compare stack slots");
        self.emit.emit_je_label(label).expect("stack slots equal");
    }

    /// Increment a stack slot value by 1.
    pub fn emit_inc_stack_slot(&mut self, slot: u32) {
        self.emit
            .emit_with(|buf| {
                x64::encode_mov_r64_m(
                    0,
                    Mem {
                        base: 4,
                        disp: slot as i32,
                    },
                    buf,
                )?;
                x64::encode_add_r64_imm32(0, 1, buf)?;
                x64::encode_mov_m_r64(
                    Mem {
                        base: 4,
                        disp: slot as i32,
                    },
                    0,
                    buf,
                )
            })
            .expect("inc stack slot");
    }

    // ── Map support ─────────────────────────────────────────────────

    /// Advance the out register (r14) by a constant offset.
    ///
    /// Used in map loops to move from the key slot to the value slot within a pair.
    pub fn emit_advance_out_by(&mut self, offset: u32) {
        self.emit
            .emit_with(|buf| x64::encode_add_r64_imm32(14, offset, buf))
            .expect("add");
    }

    /// Call `kajit_map_build(from_pair_slice_fn, saved_out, pairs_buf, count)`.
    ///
    /// We cannot call `from_pair_slice` directly from JIT code because its
    /// first arg `PtrUninit` is a 16-byte struct — passed in 2 registers on
    /// Linux x64 but by hidden pointer on Windows x64.  `kajit_map_build` is a
    /// plain-C trampoline that takes four pointer-/usize-sized args and
    /// constructs `PtrUninit` internally.
    pub fn emit_call_map_from_pairs(
        &mut self,
        from_pair_slice_fn: *const u8,
        saved_out_slot: u32,
        buf_slot: u32,
        count_slot: u32,
    ) {
        let fn_val = from_pair_slice_fn as i64;

        #[cfg(not(windows))]
        self.emit
            .emit_with(|buf| {
                x64::encode_mov_r64_imm64(7, fn_val as u64, buf)?;
                x64::encode_mov_r64_m(
                    6,
                    Mem {
                        base: 4,
                        disp: saved_out_slot as i32,
                    },
                    buf,
                )?;
                x64::encode_mov_r64_m(
                    2,
                    Mem {
                        base: 4,
                        disp: buf_slot as i32,
                    },
                    buf,
                )?;
                x64::encode_mov_r64_m(
                    1,
                    Mem {
                        base: 4,
                        disp: count_slot as i32,
                    },
                    buf,
                )
            })
            .expect("map from pairs");
        #[cfg(windows)]
        self.emit
            .emit_with(|buf| {
                x64::encode_mov_r64_imm64(1, fn_val as u64, buf)?;
                x64::encode_mov_r64_m(
                    2,
                    Mem {
                        base: 4,
                        disp: saved_out_slot as i32,
                    },
                    buf,
                )?;
                x64::encode_mov_r64_m(
                    8,
                    Mem {
                        base: 4,
                        disp: buf_slot as i32,
                    },
                    buf,
                )?;
                x64::encode_mov_r64_m(
                    9,
                    Mem {
                        base: 4,
                        disp: count_slot as i32,
                    },
                    buf,
                )
            })
            .expect("map from pairs");
        self.emit_call_fn_ptr(crate::intrinsics::kajit_map_build as *const () as *const u8);
    }

    /// Call `kajit_map_build(from_pair_slice_fn, r14, null, 0)` — empty map.
    ///
    /// Same trampoline pattern as `emit_call_map_from_pairs`.
    pub fn emit_call_map_from_pairs_empty(&mut self, from_pair_slice_fn: *const u8) {
        let fn_val = from_pair_slice_fn as i64;

        #[cfg(not(windows))]
        self.emit
            .emit_with(|buf| {
                x64::encode_mov_r64_imm64(7, fn_val as u64, buf)?;
                x64::encode_mov_r64_r64(6, 14, buf)?;
                x64::encode_xor_r64_r64(2, 2, buf)?;
                x64::encode_xor_r64_r64(1, 1, buf)
            })
            .expect("map from pairs empty");
        #[cfg(windows)]
        self.emit
            .emit_with(|buf| {
                x64::encode_mov_r64_imm64(1, fn_val as u64, buf)?;
                x64::encode_mov_r64_r64(2, 14, buf)?;
                x64::encode_xor_r64_r64(8, 8, buf)?;
                x64::encode_xor_r64_r64(9, 9, buf)
            })
            .expect("map from pairs empty");
        self.emit_call_fn_ptr(crate::intrinsics::kajit_map_build as *const () as *const u8);
    }

    /// Call `kajit_vec_free(buf, cap, pair_stride, pair_align)` to free the pairs buffer.
    ///
    /// Used on the success path after `from_pair_slice` has moved the pairs into the map.
    /// Does NOT branch to error exit (pairs free on success path, not error).
    pub fn emit_call_pairs_free(
        &mut self,
        free_fn: *const u8,
        buf_slot: u32,
        cap_slot: u32,
        pair_stride: u32,
        pair_align: u32,
    ) {
        #[cfg(not(windows))]
        self.emit
            .emit_with(|buf| {
                x64::encode_mov_r64_m(
                    7,
                    Mem {
                        base: 4,
                        disp: buf_slot as i32,
                    },
                    buf,
                )?;
                x64::encode_mov_r64_m(
                    6,
                    Mem {
                        base: 4,
                        disp: cap_slot as i32,
                    },
                    buf,
                )?;
                x64::encode_mov_r32_imm32(2, pair_stride as u32, buf)?;
                x64::encode_mov_r32_imm32(1, pair_align as u32, buf)
            })
            .expect("pairs free");
        #[cfg(windows)]
        self.emit
            .emit_with(|buf| {
                x64::encode_mov_r64_m(
                    1,
                    Mem {
                        base: 4,
                        disp: buf_slot as i32,
                    },
                    buf,
                )?;
                x64::encode_mov_r64_m(
                    2,
                    Mem {
                        base: 4,
                        disp: cap_slot as i32,
                    },
                    buf,
                )?;
                x64::encode_mov_r32_imm32(8, pair_stride as u32, buf)?;
                x64::encode_mov_r32_imm32(9, pair_align as u32, buf)
            })
            .expect("pairs free");
        self.emit_call_fn_ptr(free_fn);
    }

    // ── Recipe emission ─────────────────────────────────────────────

    /// Emit a recipe — interpret a sequence of micro-ops into x86_64 instructions.
    pub fn emit_recipe(&mut self, recipe: &Recipe) {
        let error_exit = self.error_exit;

        // Allocate dynamic labels for the recipe
        let labels: Vec<LabelId> = (0..recipe.label_count)
            .map(|_| self.emit.new_label())
            .collect();

        // Shared EOF error label
        let eof_label = self.emit.new_label();

        for op in &recipe.ops {
            match op {
                Op::BoundsCheck { count } => {
                    if *count == 1 {
                        self.emit
                            .emit_with(|buf| x64::encode_cmp_r64_r64(12, 13, buf))
                            .expect("bounds check count=1");
                        self.emit.emit_jae_label(eof_label).expect("bounds check eof");
                    } else {
                        let count = *count as i32;
                        self.emit
                            .emit_with(|buf| {
                                x64::encode_mov_r64_r64(10, 13, buf)?;
                                x64::encode_sub_r64_r64(10, 12, buf)?;
                                x64::encode_cmp_r64_imm32(10, count as u32, buf)
                            })
                            .expect("bounds check");
                        self.emit.emit_jbe_label(eof_label).expect("bounds check eof");
                    }
                }
                Op::LoadByte { dst } => match dst {
                    Slot::A => self
                        .emit
                        .emit_with(|buf| {
                            x64::encode_movzx_r32_rm8(
                                10,
                                x64::Operand::Mem(Mem { base: 12, disp: 0 }),
                                buf,
                            )
                        })
                        .expect("load byte slot a"),
                    Slot::B => self
                        .emit
                        .emit_with(|buf| {
                            x64::encode_movzx_r32_rm8(
                                11,
                                x64::Operand::Mem(Mem { base: 12, disp: 0 }),
                                buf,
                            )
                        })
                        .expect("load byte slot b"),
                },
                Op::LoadFromCursor { dst, width } => match (dst, width) {
                    (Slot::A, Width::W4) => self
                        .emit
                        .emit_with(|buf| x64::encode_mov_r32_m(10, Mem { base: 12, disp: 0 }, buf))
                        .expect("load slot a w4"),
                    (Slot::A, Width::W8) => self
                        .emit
                        .emit_with(|buf| x64::encode_mov_r64_m(10, Mem { base: 12, disp: 0 }, buf))
                        .expect("load slot a w8"),
                    (Slot::B, Width::W4) => self
                        .emit
                        .emit_with(|buf| x64::encode_mov_r32_m(11, Mem { base: 12, disp: 0 }, buf))
                        .expect("load slot b w4"),
                    (Slot::B, Width::W8) => self
                        .emit
                        .emit_with(|buf| x64::encode_mov_r64_m(11, Mem { base: 12, disp: 0 }, buf))
                        .expect("load slot b w8"),
                    _ => panic!("unsupported LoadFromCursor width"),
                },
                Op::StoreToOut { src, offset, width } => {
                    let offset = *offset as i32;
                    match (src, width) {
                        (Slot::A, Width::W1) => self
                            .emit
                            .emit_with(|buf| x64::encode_mov_m_r8(14, offset, 10, buf))
                            .expect("store to out w1"),
                        (Slot::A, Width::W2) => self
                            .emit
                            .emit_with(|buf| x64::encode_mov_m_r16(14, offset, 10, buf))
                            .expect("store to out w2"),
                        (Slot::A, Width::W4) => self
                            .emit
                            .emit_with(|buf| {
                                x64::encode_mov_m_r32(
                                    Mem {
                                        base: 14,
                                        disp: offset,
                                    },
                                    10,
                                    buf,
                                )
                            })
                            .expect("store to out w4"),
                        (Slot::A, Width::W8) => self
                            .emit
                            .emit_with(|buf| {
                                x64::encode_mov_m_r64(
                                    Mem {
                                        base: 14,
                                        disp: offset,
                                    },
                                    10,
                                    buf,
                                )
                            })
                            .expect("store to out w8"),
                        (Slot::B, Width::W1) => self
                            .emit
                            .emit_with(|buf| x64::encode_mov_m_r8(14, offset, 11, buf))
                            .expect("store to out w1"),
                        (Slot::B, Width::W2) => self
                            .emit
                            .emit_with(|buf| x64::encode_mov_m_r16(14, offset, 11, buf))
                            .expect("store to out w2"),
                        (Slot::B, Width::W4) => self
                            .emit
                            .emit_with(|buf| {
                                x64::encode_mov_m_r32(
                                    Mem {
                                        base: 14,
                                        disp: offset,
                                    },
                                    11,
                                    buf,
                                )
                            })
                            .expect("store to out w4"),
                        (Slot::B, Width::W8) => self
                            .emit
                            .emit_with(|buf| {
                                x64::encode_mov_m_r64(
                                    Mem {
                                        base: 14,
                                        disp: offset,
                                    },
                                    11,
                                    buf,
                                )
                            })
                            .expect("store to out w8"),
                    }
                }
                Op::StoreByteToStack { src, sp_offset } => {
                    let sp_offset = *sp_offset as i32;
                    match src {
                        Slot::A => self
                            .emit
                            .emit_with(|buf| x64::encode_mov_m_r8(4, sp_offset, 10, buf))
                            .expect("store byte stack a"),
                        Slot::B => self
                            .emit
                            .emit_with(|buf| x64::encode_mov_m_r8(4, sp_offset, 11, buf))
                            .expect("store byte stack b"),
                    }
                }
                Op::StoreToStack {
                    src,
                    sp_offset,
                    width,
                } => {
                    let sp_offset = *sp_offset as i32;
                    match (src, width) {
                        (Slot::A, Width::W4) => self
                            .emit
                            .emit_with(|buf| {
                                x64::encode_mov_m_r32(
                                    Mem {
                                        base: 4,
                                        disp: sp_offset,
                                    },
                                    10,
                                    buf,
                                )
                            })
                            .expect("store stack a w4"),
                        (Slot::A, Width::W8) => self
                            .emit
                            .emit_with(|buf| {
                                x64::encode_mov_m_r64(
                                    Mem {
                                        base: 4,
                                        disp: sp_offset,
                                    },
                                    10,
                                    buf,
                                )
                            })
                            .expect("store stack a w8"),
                        (Slot::B, Width::W4) => self
                            .emit
                            .emit_with(|buf| {
                                x64::encode_mov_m_r32(
                                    Mem {
                                        base: 4,
                                        disp: sp_offset,
                                    },
                                    11,
                                    buf,
                                )
                            })
                            .expect("store stack b w4"),
                        (Slot::B, Width::W8) => self
                            .emit
                            .emit_with(|buf| {
                                x64::encode_mov_m_r64(
                                    Mem {
                                        base: 4,
                                        disp: sp_offset,
                                    },
                                    11,
                                    buf,
                                )
                            })
                            .expect("store stack b w8"),
                        _ => panic!("unsupported StoreToStack width"),
                    }
                }
                Op::LoadFromStack {
                    dst,
                    sp_offset,
                    width,
                } => {
                    let sp_offset = *sp_offset as i32;
                    match (dst, width) {
                        (Slot::A, Width::W4) => self
                            .emit
                            .emit_with(|buf| {
                                x64::encode_mov_r32_m(
                                    10,
                                    Mem {
                                        base: 4,
                                        disp: sp_offset,
                                    },
                                    buf,
                                )
                            })
                            .expect("load stack a w4"),
                        (Slot::A, Width::W8) => self
                            .emit
                            .emit_with(|buf| {
                                x64::encode_mov_r64_m(
                                    10,
                                    Mem {
                                        base: 4,
                                        disp: sp_offset,
                                    },
                                    buf,
                                )
                            })
                            .expect("load stack a w8"),
                        (Slot::B, Width::W4) => self
                            .emit
                            .emit_with(|buf| {
                                x64::encode_mov_r32_m(
                                    11,
                                    Mem {
                                        base: 4,
                                        disp: sp_offset,
                                    },
                                    buf,
                                )
                            })
                            .expect("load stack b w4"),
                        (Slot::B, Width::W8) => self
                            .emit
                            .emit_with(|buf| {
                                x64::encode_mov_r64_m(
                                    11,
                                    Mem {
                                        base: 4,
                                        disp: sp_offset,
                                    },
                                    buf,
                                )
                            })
                            .expect("load stack b w8"),
                        _ => panic!("unsupported LoadFromStack width"),
                    }
                }
                Op::AdvanceCursor { count } => {
                    let count = *count as i32;
                    self.emit
                        .emit_with(|buf| x64::encode_add_r64_imm32(12, count as u32, buf))
                        .expect("advance cursor")
                }
                Op::AdvanceCursorBySlot { slot } => match slot {
                    Slot::A => self
                        .emit
                        .emit_with(|buf| x64::encode_add_r64_r64(12, 10, buf))
                        .expect("advance cursor by a"),
                    Slot::B => self
                        .emit
                        .emit_with(|buf| x64::encode_add_r64_r64(12, 11, buf))
                        .expect("advance cursor by b"),
                },
                Op::ZigzagDecode { slot } => match slot {
                    Slot::A => self
                        .emit
                        .emit_with(|buf| {
                            x64::encode_mov_r32_r32(11, 10, buf)?;
                            x64::encode_shr_r64_imm8(11, 1, buf)?;
                            x64::encode_mov_r64_imm64(9, 1, buf)?;
                            x64::encode_and_r64_r64(10, 9, buf)?;
                            x64::encode_neg_r64(10, buf)?;
                            x64::encode_xor_r64_r64(10, 11, buf)
                        })
                        .expect("zigzag decode a"),
                    Slot::B => self
                        .emit
                        .emit_with(|buf| {
                            x64::encode_mov_r32_r32(10, 11, buf)?;
                            x64::encode_shr_r64_imm8(10, 1, buf)?;
                            x64::encode_mov_r64_imm64(9, 1, buf)?;
                            x64::encode_and_r64_r64(11, 9, buf)?;
                            x64::encode_neg_r64(11, buf)?;
                            x64::encode_xor_r64_r64(11, 10, buf)
                        })
                        .expect("zigzag decode b"),
                },
                Op::ValidateMax {
                    slot,
                    max_val,
                    error,
                } => {
                    let max_val = *max_val as i32;
                    let error_code = *error as i32;
                    let invalid_label = self.emit.new_label();
                    let ok_label = self.emit.new_label();
                    match slot {
                        Slot::A => self
                            .emit
                            .emit_with(|buf| x64::encode_cmp_r64_imm32(10, max_val as u32, buf))
                            .expect("validate max a"),
                        Slot::B => self
                            .emit
                            .emit_with(|buf| x64::encode_cmp_r64_imm32(11, max_val as u32, buf))
                            .expect("validate max b"),
                    }
                    self.emit
                        .emit_ja_label(invalid_label)
                        .expect("validate max branch");
                    self.emit
                        .emit_jmp_label(ok_label)
                        .expect("validate max ok jump");
                    self.emit
                        .bind_label(invalid_label)
                        .expect("bind invalid label");
                    self.emit
                        .emit_with(|buf| {
                            x64::encode_mov_r32_imm32(9, error_code as u32, buf)?;
                            x64::encode_mov_m_r32(
                                Mem {
                                    base: 15,
                                    disp: CTX_ERROR_CODE as i32,
                                },
                                9,
                                buf,
                            )
                        })
                        .expect("write error code");
                    self.emit
                        .emit_jmp_label(error_exit)
                        .expect("validate error_exit");
                    self.emit.bind_label(ok_label).expect("bind ok label");
                }
                Op::TestBit7Branch { slot, target } => {
                    let label = labels[*target];
                    match slot {
                        Slot::A => {
                            self.emit
                                .emit_with(|buf| {
                                    x64::encode_mov_r64_imm64(9, 0x80, buf)?;
                                    x64::encode_test_r64_r64(10, 9, buf)
                                })
                                .expect("test bit7 a");
                            self.emit.emit_jnz_label(label).expect("bit7 a branch")
                        }
                        Slot::B => {
                            self.emit
                                .emit_with(|buf| {
                                    x64::encode_mov_r64_imm64(9, 0x80, buf)?;
                                    x64::encode_test_r64_r64(11, 9, buf)
                                })
                                .expect("test bit7 b");
                            self.emit.emit_jnz_label(label).expect("bit7 b branch")
                        }
                    }
                }
                Op::Branch { target } => {
                    let label = labels[*target];
                    self.emit.emit_jmp_label(label).expect("branch")
                }
                Op::BindLabel { index } => {
                    let label = labels[*index];
                    self.emit.bind_label(label).expect("bind")
                }
                Op::CallIntrinsic {
                    fn_ptr,
                    field_offset,
                } => {
                    let field_offset = *field_offset as i32;
                    self.emit_flush_input_cursor();
                    #[cfg(not(windows))]
                    self.emit
                        .emit_with(|buf| {
                            x64::encode_mov_r64_r64(7, 15, buf)?;
                            x64::encode_lea_r64_m(
                                6,
                                Mem {
                                    base: 14,
                                    disp: field_offset,
                                },
                                buf,
                            )
                        })
                        .expect("call intrinsic args");
                    #[cfg(windows)]
                    self.emit
                        .emit_with(|buf| {
                            x64::encode_mov_r64_r64(1, 15, buf)?;
                            x64::encode_lea_r64_m(
                                2,
                                Mem {
                                    base: 14,
                                    disp: field_offset,
                                },
                                buf,
                            )
                        })
                        .expect("call intrinsic args");
                    self.emit_call_fn_ptr(*fn_ptr);
                    self.emit_reload_cursor_and_check_error();
                }
                Op::CallIntrinsicStackOut { fn_ptr, sp_offset } => {
                    let sp_offset = *sp_offset as i32;
                    self.emit_flush_input_cursor();
                    #[cfg(not(windows))]
                    self.emit
                        .emit_with(|buf| {
                            x64::encode_mov_r64_r64(7, 15, buf)?;
                            x64::encode_lea_r64_m(
                                6,
                                Mem {
                                    base: 4,
                                    disp: sp_offset,
                                },
                                buf,
                            )
                        })
                        .expect("call intrinsic stack out args");
                    #[cfg(windows)]
                    self.emit
                        .emit_with(|buf| {
                            x64::encode_mov_r64_r64(1, 15, buf)?;
                            x64::encode_lea_r64_m(
                                2,
                                Mem {
                                    base: 4,
                                    disp: sp_offset,
                                },
                                buf,
                            )
                        })
                        .expect("call intrinsic stack out args");
                    self.emit_call_fn_ptr(*fn_ptr);
                    self.emit_reload_cursor_and_check_error();
                }
                Op::ComputeRemaining { dst } => match dst {
                    Slot::A => self
                        .emit
                        .emit_with(|buf| {
                            x64::encode_mov_r64_r64(10, 13, buf)?;
                            x64::encode_sub_r64_r64(10, 12, buf)
                        })
                        .expect("compute remaining a"),
                    Slot::B => self
                        .emit
                        .emit_with(|buf| {
                            x64::encode_mov_r64_r64(11, 13, buf)?;
                            x64::encode_sub_r64_r64(11, 12, buf)
                        })
                        .expect("compute remaining b"),
                },
                Op::CmpBranchLo { lhs, rhs, on_fail } => {
                    let target = match on_fail {
                        ErrorTarget::Eof => eof_label,
                        ErrorTarget::ErrorExit => error_exit,
                    };
                    match (lhs, rhs) {
                        (Slot::A, Slot::B) => {
                            self.emit
                                .emit_with(|buf| x64::encode_cmp_r64_r64(10, 11, buf))
                                .expect("cmp branch lo a,b");
                            self.emit
                                .emit_jcc_label(target, x64::Condition::Lo)
                                .expect("cmp branch lo a,b target")
                        }
                        (Slot::B, Slot::A) => {
                            self.emit
                                .emit_with(|buf| x64::encode_cmp_r64_r64(11, 10, buf))
                                .expect("cmp branch lo b,a");
                            self.emit
                                .emit_jcc_label(target, x64::Condition::Lo)
                                .expect("cmp branch lo b,a target")
                        }
                        _ => panic!("CmpBranchLo requires different slots"),
                    }
                }
                Op::SaveCursor { dst } => match dst {
                    Slot::A => self
                        .emit
                        .emit_with(|buf| x64::encode_mov_r64_r64(10, 12, buf))
                        .expect("save cursor a"),
                    Slot::B => self
                        .emit
                        .emit_with(|buf| x64::encode_mov_r64_r64(11, 12, buf))
                        .expect("save cursor b"),
                },
                Op::CallValidateAllocCopy {
                    fn_ptr,
                    data_src,
                    len_src,
                } => {
                    // Call kajit_string_validate_alloc_copy(ctx, data_ptr, data_len)
                    // Returns buf pointer in rax
                    // arg0 = ctx, arg1 = data_ptr (slot), arg2 = data_len (slot)
                    self.emit_flush_input_cursor();
                    #[cfg(not(windows))]
                    {
                        self.emit
                            .emit_with(|buf| x64::encode_mov_r64_r64(7, 15, buf))
                            .expect("validate copy ctx");
                        match data_src {
                            Slot::A => self
                                .emit
                                .emit_with(|buf| x64::encode_mov_r64_r64(6, 10, buf))
                                .expect("validate copy data a"),
                            Slot::B => self
                                .emit
                                .emit_with(|buf| x64::encode_mov_r64_r64(6, 11, buf))
                                .expect("validate copy data b"),
                        }
                        match len_src {
                            Slot::A => self
                                .emit
                                .emit_with(|buf| x64::encode_mov_r32_r32(2, 10, buf))
                                .expect("validate copy len a"),
                            Slot::B => self
                                .emit
                                .emit_with(|buf| x64::encode_mov_r32_r32(2, 11, buf))
                                .expect("validate copy len b"),
                        }
                    }
                    #[cfg(windows)]
                    {
                        self.emit
                            .emit_with(|buf| x64::encode_mov_r64_r64(1, 15, buf))
                            .expect("validate copy ctx windows");
                        match data_src {
                            Slot::A => self
                                .emit
                                .emit_with(|buf| x64::encode_mov_r64_r64(2, 10, buf))
                                .expect("validate copy data a windows"),
                            Slot::B => self
                                .emit
                                .emit_with(|buf| x64::encode_mov_r64_r64(2, 11, buf))
                                .expect("validate copy data b windows"),
                        }
                        match len_src {
                            Slot::A => self
                                .emit
                                .emit_with(|buf| x64::encode_mov_r32_r32(8, 10, buf))
                                .expect("validate copy len a windows"),
                            Slot::B => self
                                .emit
                                .emit_with(|buf| x64::encode_mov_r32_r32(8, 11, buf))
                                .expect("validate copy len b windows"),
                        }
                    }
                    self.emit_call_fn_ptr(*fn_ptr);
                    // Use r11d for error check to preserve r10 (Slot::A) for WriteMalumString
                    self.emit
                        .emit_with(|buf| {
                            x64::encode_mov_r32_m(
                                11,
                                Mem {
                                    base: 15,
                                    disp: CTX_ERROR_CODE as i32,
                                },
                                buf,
                            )?;
                            x64::encode_test_r32_r32(11, 11, buf)
                        })
                        .expect("validate copy error check");
                    self.emit
                        .emit_jnz_label(error_exit)
                        .expect("validate copy branch to error")
                }
                Op::WriteMalumString {
                    base_offset,
                    ptr_off,
                    len_off,
                    cap_off,
                    len_slot,
                } => {
                    let ptr_offset = (*base_offset + *ptr_off) as i32;
                    let len_offset = (*base_offset + *len_off) as i32;
                    let cap_offset = (*base_offset + *cap_off) as i32;
                    // rax = buf pointer from previous call return
                    self.emit
                        .emit_with(|buf| {
                            x64::encode_mov_m_r64(
                                Mem {
                                    base: 14,
                                    disp: ptr_offset,
                                },
                                0,
                                buf,
                            )
                        })
                        .expect("write malum ptr");
                    match len_slot {
                        Slot::A => {
                            self.emit
                                .emit_with(|buf| {
                                    x64::encode_mov_m_r64(
                                        Mem {
                                            base: 14,
                                            disp: len_offset,
                                        },
                                        10,
                                        buf,
                                    )?;
                                    x64::encode_mov_m_r64(
                                        Mem {
                                            base: 14,
                                            disp: cap_offset,
                                        },
                                        10,
                                        buf,
                                    )
                                })
                                .expect("write malum len a");
                        }
                        Slot::B => {
                            self.emit
                                .emit_with(|buf| {
                                    x64::encode_mov_m_r64(
                                        Mem {
                                            base: 14,
                                            disp: len_offset,
                                        },
                                        11,
                                        buf,
                                    )?;
                                    x64::encode_mov_m_r64(
                                        Mem {
                                            base: 14,
                                            disp: cap_offset,
                                        },
                                        11,
                                        buf,
                                    )
                                })
                                .expect("write malum len b");
                        }
                    }
                }

                // ── Encode-direction ops ──────────────────────────────────
                //
                // In encode mode the register assignments are:
                //   r12 = output_ptr (write cursor)
                //   r13 = output_end
                //   r14 = input struct pointer
                //   r15 = EncodeContext pointer
                //
                // Slot::A → r10/r10d, Slot::B → r11/r11d (same as decode)
                Op::LoadFromInput { dst, offset, width } => {
                    let offset = *offset as i32;
                    match (dst, width) {
                        (Slot::A, Width::W1) => self
                            .emit
                            .emit_with(|buf| {
                                x64::encode_movzx_r32_rm8(
                                    10,
                                    x64::Operand::Mem(Mem {
                                        base: 14,
                                        disp: offset,
                                    }),
                                    buf,
                                )
                            })
                            .expect("load input a w1"),
                        (Slot::A, Width::W2) => self
                            .emit
                            .emit_with(|buf| {
                                x64::encode_movzx_r32_rm16(
                                    10,
                                    x64::Operand::Mem(Mem {
                                        base: 14,
                                        disp: offset,
                                    }),
                                    buf,
                                )
                            })
                            .expect("load input a w2"),
                        (Slot::A, Width::W4) => self
                            .emit
                            .emit_with(|buf| {
                                x64::encode_mov_r32_m(
                                    10,
                                    Mem {
                                        base: 14,
                                        disp: offset,
                                    },
                                    buf,
                                )
                            })
                            .expect("load input a w4"),
                        (Slot::A, Width::W8) => self
                            .emit
                            .emit_with(|buf| {
                                x64::encode_mov_r64_m(
                                    10,
                                    Mem {
                                        base: 14,
                                        disp: offset,
                                    },
                                    buf,
                                )
                            })
                            .expect("load input a w8"),
                        (Slot::B, Width::W1) => self
                            .emit
                            .emit_with(|buf| {
                                x64::encode_movzx_r32_rm8(
                                    11,
                                    x64::Operand::Mem(Mem {
                                        base: 14,
                                        disp: offset,
                                    }),
                                    buf,
                                )
                            })
                            .expect("load input b w1"),
                        (Slot::B, Width::W2) => self
                            .emit
                            .emit_with(|buf| {
                                x64::encode_movzx_r32_rm16(
                                    11,
                                    x64::Operand::Mem(Mem {
                                        base: 14,
                                        disp: offset,
                                    }),
                                    buf,
                                )
                            })
                            .expect("load input b w2"),
                        (Slot::B, Width::W4) => self
                            .emit
                            .emit_with(|buf| {
                                x64::encode_mov_r32_m(
                                    11,
                                    Mem {
                                        base: 14,
                                        disp: offset,
                                    },
                                    buf,
                                )
                            })
                            .expect("load input b w4"),
                        (Slot::B, Width::W8) => self
                            .emit
                            .emit_with(|buf| {
                                x64::encode_mov_r64_m(
                                    11,
                                    Mem {
                                        base: 14,
                                        disp: offset,
                                    },
                                    buf,
                                )
                            })
                            .expect("load input b w8"),
                    }
                }
                Op::StoreToOutput { src, width } => match (src, width) {
                    (Slot::A, Width::W1) => self
                        .emit
                        .emit_with(|buf| {
                            x64::encode_mov_m_r8(12, 0, 10, buf)?;
                            x64::encode_add_r64_imm32(12, 1, buf)
                        })
                        .expect("store output a w1"),
                    (Slot::A, Width::W2) => self
                        .emit
                        .emit_with(|buf| {
                            x64::encode_mov_m_r16(12, 0, 10, buf)?;
                            x64::encode_add_r64_imm32(12, 2, buf)
                        })
                        .expect("store output a w2"),
                    (Slot::A, Width::W4) => self
                        .emit
                        .emit_with(|buf| {
                            x64::encode_mov_m_r32(Mem { base: 12, disp: 0 }, 10, buf)?;
                            x64::encode_add_r64_imm32(12, 4, buf)
                        })
                        .expect("store output a w4"),
                    (Slot::A, Width::W8) => self
                        .emit
                        .emit_with(|buf| {
                            x64::encode_mov_m_r64(Mem { base: 12, disp: 0 }, 10, buf)?;
                            x64::encode_add_r64_imm32(12, 8, buf)
                        })
                        .expect("store output a w8"),
                    (Slot::B, Width::W1) => self
                        .emit
                        .emit_with(|buf| {
                            x64::encode_mov_m_r8(12, 0, 11, buf)?;
                            x64::encode_add_r64_imm32(12, 1, buf)
                        })
                        .expect("store output b w1"),
                    (Slot::B, Width::W2) => self
                        .emit
                        .emit_with(|buf| {
                            x64::encode_mov_m_r16(12, 0, 11, buf)?;
                            x64::encode_add_r64_imm32(12, 2, buf)
                        })
                        .expect("store output b w2"),
                    (Slot::B, Width::W4) => self
                        .emit
                        .emit_with(|buf| {
                            x64::encode_mov_m_r32(Mem { base: 12, disp: 0 }, 11, buf)?;
                            x64::encode_add_r64_imm32(12, 4, buf)
                        })
                        .expect("store output b w4"),
                    (Slot::B, Width::W8) => self
                        .emit
                        .emit_with(|buf| {
                            x64::encode_mov_m_r64(Mem { base: 12, disp: 0 }, 11, buf)?;
                            x64::encode_add_r64_imm32(12, 8, buf)
                        })
                        .expect("store output b w8"),
                },
                Op::WriteByte { value } => {
                    let value = *value as i8;
                    self.emit
                        .emit_with(|buf| {
                            x64::encode_mov_r64_imm64(9, value as u8 as u64, buf)?;
                            x64::encode_mov_m_r8(12, 0, 9, buf)?;
                            x64::encode_add_r64_imm32(12, 1, buf)
                        })
                        .expect("write byte")
                }
                Op::AdvanceOutput { count } => {
                    let count = *count;
                    if count > 0 {
                        self.emit
                            .emit_with(|buf| x64::encode_add_r64_imm32(12, count as u32, buf))
                            .expect("advance output")
                    }
                }
                Op::AdvanceOutputBySlot { slot } => match slot {
                    Slot::A => self
                        .emit
                        .emit_with(|buf| x64::encode_add_r64_r64(12, 10, buf))
                        .expect("advance output by a"),
                    Slot::B => self
                        .emit
                        .emit_with(|buf| x64::encode_add_r64_r64(12, 11, buf))
                        .expect("advance output by b"),
                },
                Op::OutputBoundsCheck { count } => {
                    let count = *count as i32;
                    let have_space = self.emit.new_label();

                    self.emit
                        .emit_with(|buf| {
                            x64::encode_mov_r64_r64(0, 13, buf)?;
                            x64::encode_sub_r64_r64(0, 12, buf)?;
                            x64::encode_cmp_r64_imm32(0, count as u32, buf)
                        })
                        .expect("output bounds check");
                    self.emit
                        .emit_jcc_label(have_space, x64::Condition::Ge)
                        .expect("output bounds check ge");

                    // Not enough space — call kajit_output_grow(ctx, needed)
                    self.emit_enc_flush_output_cursor();
                    #[cfg(not(windows))]
                    self.emit
                        .emit_with(|buf| {
                            x64::encode_mov_r64_r64(7, 15, buf)?;
                            x64::encode_mov_r64_imm64(6, count as u64, buf)
                        })
                        .expect("output grow args");
                    #[cfg(windows)]
                    self.emit
                        .emit_with(|buf| {
                            x64::encode_mov_r64_r64(1, 15, buf)?;
                            x64::encode_mov_r64_imm64(2, count as u64, buf)
                        })
                        .expect("output grow args windows");
                    self.emit_call_fn_ptr(crate::intrinsics::kajit_output_grow as *const u8);
                    self.emit_enc_reload_and_check_error();

                    self.emit
                        .bind_label(have_space)
                        .expect("bind output bounds done");
                }
                Op::SignExtend { slot, from } => match (slot, from) {
                    (Slot::A, Width::W1) => self
                        .emit
                        .emit_with(|buf| x64::encode_movsx_r64_rm8(10, x64::Operand::Reg(10), buf))
                        .expect("sign extend a w1"),
                    (Slot::A, Width::W2) => self
                        .emit
                        .emit_with(|buf| x64::encode_movsx_r64_rm16(10, x64::Operand::Reg(10), buf))
                        .expect("sign extend a w2"),
                    (Slot::B, Width::W1) => self
                        .emit
                        .emit_with(|buf| x64::encode_movsx_r64_rm8(11, x64::Operand::Reg(11), buf))
                        .expect("sign extend b w1"),
                    (Slot::B, Width::W2) => self
                        .emit
                        .emit_with(|buf| x64::encode_movsx_r64_rm16(11, x64::Operand::Reg(11), buf))
                        .expect("sign extend b w2"),
                    (_, Width::W4 | Width::W8) => {} // already at natural width
                },
                Op::ZigzagEncode { slot, wide } => match (slot, wide) {
                    // zigzag encode 32-bit: (n << 1) ^ (n >> 31)
                    (Slot::A, false) => self
                        .emit
                        .emit_with(|buf| {
                            x64::encode_mov_r64_r64(11, 10, buf)?;
                            x64::encode_shl_r64_imm8(11, 1, buf)?;
                            x64::encode_sar_r64_imm8(10, 31, buf)?;
                            x64::encode_xor_r64_r64(10, 11, buf)
                        })
                        .expect("zigzag encode a"),
                    (Slot::B, false) => self
                        .emit
                        .emit_with(|buf| {
                            x64::encode_mov_r64_r64(10, 11, buf)?;
                            x64::encode_shl_r64_imm8(10, 1, buf)?;
                            x64::encode_sar_r64_imm8(11, 31, buf)?;
                            x64::encode_xor_r64_r64(11, 10, buf)
                        })
                        .expect("zigzag encode b"),
                    // zigzag encode 64-bit: (n << 1) ^ (n >> 63)
                    (Slot::A, true) => self
                        .emit
                        .emit_with(|buf| {
                            x64::encode_mov_r64_r64(11, 10, buf)?;
                            x64::encode_shl_r64_imm8(11, 1, buf)?;
                            x64::encode_sar_r64_imm8(10, 63, buf)?;
                            x64::encode_xor_r64_r64(10, 11, buf)
                        })
                        .expect("zigzag encode wide a"),
                    (Slot::B, true) => self
                        .emit
                        .emit_with(|buf| {
                            x64::encode_mov_r64_r64(10, 11, buf)?;
                            x64::encode_shl_r64_imm8(10, 1, buf)?;
                            x64::encode_sar_r64_imm8(11, 63, buf)?;
                            x64::encode_xor_r64_r64(11, 10, buf)
                        })
                        .expect("zigzag encode wide b"),
                },
                Op::EncodeVarint { slot, wide } => {
                    // Inline varint encoding loop.
                    // While value >= 0x80: write (byte | 0x80), shift right 7.
                    // Then write final byte.
                    let loop_label = self.emit.new_label();
                    let done_label = self.emit.new_label();
                    let val_reg = match slot {
                        Slot::A => 10,
                        Slot::B => 11,
                    };

                    if *wide {
                        self.emit
                            .emit_with(|buf| x64::encode_mov_r64_r64(9, val_reg, buf))
                            .expect("varint copy");
                    } else {
                        self.emit
                            .emit_with(|buf| x64::encode_mov_r64_r64(9, val_reg, buf))
                            .expect("varint copy");
                    }

                    self.emit.bind_label(loop_label).expect("bind varint loop");
                    self.emit
                        .emit_with(|buf| x64::encode_cmp_r64_imm32(9, 0x80, buf))
                        .expect("varint cmp");
                    self.emit.emit_jbe_label(done_label).expect("varint done");
                    self.emit
                        .emit_with(|buf| {
                            x64::encode_mov_r64_imm64(0, 0x80, buf)?;
                            x64::encode_or_r64_r64(9, 0, buf)?;
                            x64::encode_mov_m_r8(12, 0, 9, buf)?;
                            x64::encode_add_r64_imm32(12, 1, buf)?;
                            x64::encode_shr_r64_imm8(9, 7, buf)
                        })
                        .expect("write varint byte");
                    self.emit.emit_jmp_label(loop_label).expect("varint loop");
                    self.emit.bind_label(done_label).expect("bind varint done");
                    self.emit
                        .emit_with(|buf| {
                            x64::encode_mov_m_r8(12, 0, 9, buf)?;
                            x64::encode_add_r64_imm32(12, 1, buf)
                        })
                        .expect("write final varint byte");
                }
            }
        }

        // Jump over cold path, then emit shared EOF error
        let done_label = self.emit.new_label();
        let eof_code = crate::context::ErrorCode::UnexpectedEof as u32;
        self.emit
            .emit_jmp_label(done_label)
            .expect("skip eof block");
        self.emit.bind_label(eof_label).expect("bind eof");
        self.emit
            .emit_with(|buf| {
                x64::encode_mov_r32_imm32(10, eof_code, buf)?;
                x64::encode_mov_m_r32(
                    Mem {
                        base: 15,
                        disp: CTX_ERROR_CODE as i32,
                    },
                    10,
                    buf,
                )
            })
            .expect("eof store");
        self.emit.emit_jmp_label(error_exit).expect("jmp eof error");
        self.emit.bind_label(done_label).expect("bind done");
    }

    // =====================================================================
    // Encode-direction methods
    // =====================================================================
    //
    // Register assignments for encode (same callee-saved regs, different semantics):
    //   r12 = cached output_ptr (write cursor)
    //   r13 = cached output_end
    //   r14 = input struct pointer (the typed value being serialized)
    //   r15 = EncodeContext pointer
    //
    // Entry convention: arg0 = input struct ptr, arg1 = EncodeContext ptr
    //   System V AMD64:  rdi = input, rsi = ctx
    //   Windows x64:     rcx = input, rdx = ctx

    /// Emit an encode function prologue.
    ///
    /// Caches output_ptr and output_end from EncodeContext into r12/r13.
    /// Returns the entry offset and a fresh error_exit label.
    pub fn begin_encode_func(&mut self) -> (u32, LabelId) {
        let error_exit = self.emit.new_label();
        let entry = self.emit.current_offset();
        let frame_size = self.frame_size;

        #[cfg(not(windows))]
        self.emit
            .emit_with(|buf| {
                x64::encode_push_r64(5, buf)?;
                x64::encode_sub_r64_imm32(4, frame_size, buf)?;
                x64::encode_mov_m_r64(Mem { base: 4, disp: 0 }, 5, buf)?;
                x64::encode_mov_m_r64(Mem { base: 4, disp: 16 }, 12, buf)?;
                x64::encode_mov_m_r64(Mem { base: 4, disp: 24 }, 13, buf)?;
                x64::encode_mov_m_r64(Mem { base: 4, disp: 32 }, 14, buf)?;
                x64::encode_mov_m_r64(Mem { base: 4, disp: 40 }, 15, buf)?;
                x64::encode_mov_r64_r64(14, 7, buf)?;
                x64::encode_mov_r64_r64(15, 6, buf)?;
                x64::encode_mov_r64_m(
                    12,
                    Mem {
                        base: 15,
                        disp: ENC_OUTPUT_PTR as i32,
                    },
                    buf,
                )?;
                x64::encode_mov_r64_m(
                    13,
                    Mem {
                        base: 15,
                        disp: ENC_OUTPUT_END as i32,
                    },
                    buf,
                )
            })
            .expect("begin encode prologue");
        #[cfg(windows)]
        self.emit
            .emit_with(|buf| {
                x64::encode_push_r64(5, buf)?;
                x64::encode_sub_r64_imm32(4, frame_size, buf)?;
                x64::encode_mov_m_r64(Mem { base: 4, disp: 32 }, 5, buf)?;
                x64::encode_mov_m_r64(Mem { base: 4, disp: 48 }, 12, buf)?;
                x64::encode_mov_m_r64(Mem { base: 4, disp: 56 }, 13, buf)?;
                x64::encode_mov_m_r64(Mem { base: 4, disp: 64 }, 14, buf)?;
                x64::encode_mov_m_r64(Mem { base: 4, disp: 72 }, 15, buf)?;
                x64::encode_mov_r64_r64(14, 1, buf)?;
                x64::encode_mov_r64_r64(15, 2, buf)?;
                x64::encode_mov_r64_m(
                    12,
                    Mem {
                        base: 15,
                        disp: ENC_OUTPUT_PTR as i32,
                    },
                    buf,
                )?;
                x64::encode_mov_r64_m(
                    13,
                    Mem {
                        base: 15,
                        disp: ENC_OUTPUT_END as i32,
                    },
                    buf,
                )
            })
            .expect("begin encode prologue windows");

        self.error_exit = error_exit;
        (entry, error_exit)
    }

    /// Emit the success epilogue and error exit for an encode function.
    ///
    /// Flushes output_ptr back to ctx on success.
    pub fn end_encode_func(&mut self, error_exit: LabelId) {
        let frame_size = self.frame_size;

        #[cfg(not(windows))]
        self.emit
            .emit_with(|buf| {
                x64::encode_mov_m_r64(
                    Mem {
                        base: 15,
                        disp: ENC_OUTPUT_PTR as i32,
                    },
                    12,
                    buf,
                )?;
                x64::encode_mov_r64_m(15, Mem { base: 4, disp: 40 }, buf)?;
                x64::encode_mov_r64_m(14, Mem { base: 4, disp: 32 }, buf)?;
                x64::encode_mov_r64_m(13, Mem { base: 4, disp: 24 }, buf)?;
                x64::encode_mov_r64_m(12, Mem { base: 4, disp: 16 }, buf)?;
                x64::encode_mov_r64_m(5, Mem { base: 4, disp: 0 }, buf)?;
                x64::encode_add_r64_imm32(4, frame_size as u32, buf)?;
                x64::encode_pop_r64(5, buf)?;
                x64::encode_ret(buf)
            })
            .expect("end encode");
        self.emit.bind_label(error_exit).expect("bind error_exit");
        self.emit
            .emit_with(|buf| {
                x64::encode_mov_r64_m(15, Mem { base: 4, disp: 40 }, buf)?;
                x64::encode_mov_r64_m(14, Mem { base: 4, disp: 32 }, buf)?;
                x64::encode_mov_r64_m(13, Mem { base: 4, disp: 24 }, buf)?;
                x64::encode_mov_r64_m(12, Mem { base: 4, disp: 16 }, buf)?;
                x64::encode_mov_r64_m(5, Mem { base: 4, disp: 0 }, buf)?;
                x64::encode_add_r64_imm32(4, frame_size, buf)?;
                x64::encode_pop_r64(5, buf)?;
                x64::encode_ret(buf)
            })
            .expect("end encode error");
        #[cfg(windows)]
        self.emit
            .emit_with(|buf| {
                x64::encode_mov_m_r64(
                    Mem {
                        base: 15,
                        disp: ENC_OUTPUT_PTR as i32,
                    },
                    12,
                    buf,
                )?;
                x64::encode_mov_r64_m(15, Mem { base: 4, disp: 72 }, buf)?;
                x64::encode_mov_r64_m(14, Mem { base: 4, disp: 64 }, buf)?;
                x64::encode_mov_r64_m(13, Mem { base: 4, disp: 56 }, buf)?;
                x64::encode_mov_r64_m(12, Mem { base: 4, disp: 48 }, buf)?;
                x64::encode_mov_r64_m(5, Mem { base: 4, disp: 32 }, buf)?;
                x64::encode_add_r64_imm32(4, frame_size, buf)?;
                x64::encode_pop_r64(5, buf)?;
                x64::encode_ret(buf)
            })
            .expect("end encode windows");
        self.emit
            .bind_label(error_exit)
            .expect("bind error_exit windows");
        self.emit
            .emit_with(|buf| {
                x64::encode_mov_r64_m(15, Mem { base: 4, disp: 72 }, buf)?;
                x64::encode_mov_r64_m(14, Mem { base: 4, disp: 64 }, buf)?;
                x64::encode_mov_r64_m(13, Mem { base: 4, disp: 56 }, buf)?;
                x64::encode_mov_r64_m(12, Mem { base: 4, disp: 48 }, buf)?;
                x64::encode_mov_r64_m(5, Mem { base: 4, disp: 32 }, buf)?;
                x64::encode_add_r64_imm32(4, frame_size, buf)?;
                x64::encode_pop_r64(5, buf)?;
                x64::encode_ret(buf)
            })
            .expect("end encode windows error");
    }

    /// Emit a call to another emitted encode function.
    ///
    /// Convention: arg0 = input + field_offset, arg1 = ctx.
    /// Flushes output cursor before call, reloads both output_ptr and output_end
    /// after (buffer may have grown), checks error.
    pub fn emit_enc_call_emitted_func(&mut self, label: LabelId, field_offset: u32) {
        self.emit_enc_flush_output_cursor();
        // Set up arguments: arg0 = input + field_offset, arg1 = ctx
        #[cfg(not(windows))]
        self.emit
            .emit_with(|buf| {
                x64::encode_lea_r64_m(
                    7,
                    Mem {
                        base: 14,
                        disp: field_offset as i32,
                    },
                    buf,
                )?;
                x64::encode_mov_r64_r64(6, 15, buf)
            })
            .expect("call emitted func args");
        #[cfg(windows)]
        self.emit
            .emit_with(|buf| {
                x64::encode_lea_r64_m(
                    1,
                    Mem {
                        base: 14,
                        disp: field_offset as i32,
                    },
                    buf,
                )?;
                x64::encode_mov_r64_r64(2, 15, buf)
            })
            .expect("call emitted func args win");
        self.emit.emit_call_label(label).expect("call emitted");
        self.emit_enc_reload_and_check_error();
    }

    /// Emit a call to an encode intrinsic that takes only ctx as argument.
    /// Flushes output cursor, calls, reloads both ptrs, checks error.
    pub fn emit_enc_call_intrinsic_ctx_only(&mut self, fn_ptr: *const u8) {
        self.emit_enc_flush_output_cursor();
        #[cfg(not(windows))]
        self.emit
            .emit_with(|buf| x64::encode_mov_r64_r64(7, 15, buf))
            .expect("enc intrinsic ctx only args");
        #[cfg(windows)]
        self.emit
            .emit_with(|buf| x64::encode_mov_r64_r64(1, 2, buf))
            .expect("enc intrinsic ctx only args");
        self.emit_call_fn_ptr(fn_ptr);
        self.emit_enc_reload_and_check_error();
    }

    /// Emit a call to an encode intrinsic: fn(ctx, arg1).
    /// Flushes output cursor, calls, reloads both ptrs, checks error.
    pub fn emit_enc_call_intrinsic(&mut self, fn_ptr: *const u8, arg1: u64) {
        let arg1_val = arg1 as i64;

        self.emit_enc_flush_output_cursor();
        #[cfg(not(windows))]
        self.emit
            .emit_with(|buf| {
                x64::encode_mov_r64_r64(7, 15, buf)?;
                x64::encode_mov_r64_imm64(6, arg1_val as u64, buf)
            })
            .expect("enc intrinsic args");
        #[cfg(windows)]
        self.emit
            .emit_with(|buf| {
                x64::encode_mov_r64_r64(1, 2, buf)?;
                x64::encode_mov_r64_imm64(3, arg1_val as u64, buf)
            })
            .expect("enc intrinsic args");
        self.emit_call_fn_ptr(fn_ptr);
        self.emit_enc_reload_and_check_error();
    }

    /// Emit a call to an encode intrinsic: fn(ctx, input + field_offset).
    /// Passes the input data pointer offset by `field_offset` as the second argument.
    /// Flushes output, calls, reloads output_ptr + output_end, checks error.
    pub fn emit_enc_call_intrinsic_with_input(&mut self, fn_ptr: *const u8, field_offset: u32) {
        self.emit_enc_flush_output_cursor();
        #[cfg(not(windows))]
        self.emit
            .emit_with(|buf| {
                x64::encode_mov_r64_r64(7, 15, buf)?;
                x64::encode_lea_r64_m(
                    6,
                    Mem {
                        base: 14,
                        disp: field_offset as i32,
                    },
                    buf,
                )
            })
            .expect("enc intrinsic with input args");
        #[cfg(windows)]
        self.emit
            .emit_with(|buf| {
                x64::encode_mov_r64_r64(1, 2, buf)?;
                x64::encode_lea_r64_m(
                    2,
                    Mem {
                        base: 14,
                        disp: field_offset as i32,
                    },
                    buf,
                )
            })
            .expect("enc intrinsic with input args windows");
        self.emit_call_fn_ptr(fn_ptr);
        self.emit_enc_reload_and_check_error();
    }

    /// Ensure the output buffer has at least `count` bytes of capacity.
    /// Inlines the comparison; calls kajit_output_grow only when needed.
    pub fn emit_enc_ensure_capacity(&mut self, count: u32) {
        let have_space = self.emit.new_label();

        self.emit
            .emit_with(|buf| {
                x64::encode_mov_r64_r64(0, 13, buf)?;
                x64::encode_sub_r64_r64(0, 12, buf)?;
                x64::encode_cmp_r64_imm32(0, count as u32, buf)
            })
            .expect("enc output capacity check");
        self.emit
            .emit_jge_label(have_space)
            .expect("enc have space");

        // Not enough space — call kajit_output_grow(ctx, needed)
        self.emit_enc_flush_output_cursor();
        #[cfg(not(windows))]
        self.emit
            .emit_with(|buf| {
                x64::encode_mov_r64_r64(7, 15, buf)?;
                x64::encode_mov_r64_imm64(6, count as u64, buf)
            })
            .expect("enc ensure grow args");
        #[cfg(windows)]
        self.emit
            .emit_with(|buf| {
                x64::encode_mov_r64_r64(1, 2, buf)?;
                x64::encode_mov_r64_imm64(3, count as u64, buf)
            })
            .expect("enc ensure grow args");
        self.emit_call_fn_ptr(crate::intrinsics::kajit_output_grow as *const u8);
        self.emit_enc_reload_and_check_error();

        self.emit.bind_label(have_space).expect("have_space encode");
    }

    /// Write a single immediate byte to the output buffer and advance.
    pub fn emit_enc_write_byte(&mut self, value: u8) {
        self.emit
            .emit_with(|buf| {
                x64::encode_mov_r32_imm32(10, value as u32, buf)?;
                x64::encode_mov_m_r8(12, 0, 10, buf)?;
                x64::encode_add_r64_imm32(12, 1, buf)
            })
            .expect("enc write byte");
    }

    /// Copy `count` bytes from a static pointer to the output buffer.
    /// For small counts, uses inline stores. For larger, loads from memory.
    pub fn emit_enc_write_static_bytes(&mut self, ptr: *const u8, count: u32) {
        if count == 0 {
            return;
        }

        // For very small counts, read the bytes at codegen time and emit immediate stores.
        if count <= 8 {
            // Read the actual bytes at codegen time
            let bytes: &[u8] = unsafe { core::slice::from_raw_parts(ptr, count as usize) };

            // Emit 8/4/2/1-byte stores from the end to cover the count
            let mut offset = 0u32;
            let mut remaining = count;
            while remaining >= 8 {
                let val = i64::from_le_bytes(
                    bytes[offset as usize..offset as usize + 8]
                        .try_into()
                        .unwrap(),
                );
                self.emit
                    .emit_with(|buf| {
                        x64::encode_mov_r64_imm64(0, val as u64, buf)?;
                        x64::encode_mov_m_r64(
                            Mem {
                                base: 12,
                                disp: offset as i32,
                            },
                            0,
                            buf,
                        )
                    })
                    .expect("write static bytes <=8");
                offset += 8;
                remaining -= 8;
            }
            if remaining >= 4 {
                let val = i32::from_le_bytes(
                    bytes[offset as usize..offset as usize + 4]
                        .try_into()
                        .unwrap(),
                );
                self.emit
                    .emit_with(|buf| {
                        x64::encode_mov_r32_imm32(10, val as u32, buf)?;
                        x64::encode_mov_m_r32(
                            Mem {
                                base: 12,
                                disp: offset as i32,
                            },
                            10,
                            buf,
                        )
                    })
                    .expect("write static 4");
                offset += 4;
                remaining -= 4;
            }
            if remaining >= 2 {
                let val = i16::from_le_bytes(
                    bytes[offset as usize..offset as usize + 2]
                        .try_into()
                        .unwrap(),
                );
                self.emit
                    .emit_with(|buf| {
                        x64::encode_mov_r32_imm32(10, val as u32, buf)?;
                        x64::encode_mov_m_r16(12, offset as i32, 10, buf)
                    })
                    .expect("write static 2");
                offset += 2;
                remaining -= 2;
            }
            if remaining >= 1 {
                self.emit
                    .emit_with(|buf| {
                        x64::encode_mov_r32_imm32(10, bytes[offset as usize] as u32, buf)?;
                        x64::encode_mov_m_r8(12, offset as i32, 10, buf)
                    })
                    .expect("write static 1");
            }
        } else {
            // For larger copies, load pointer and use rep movsb
            let ptr_val = ptr as i64;
            self.emit
                .emit_with(|buf| {
                    x64::encode_mov_r64_imm64(0, ptr_val as u64, buf)?;
                    x64::encode_mov_r64_r64(10, 7, buf)?;
                    x64::encode_mov_r64_r64(11, 6, buf)?;
                    x64::encode_mov_r64_r64(6, 0, buf)?;
                    x64::encode_mov_r64_r64(7, 12, buf)?;
                    x64::encode_mov_r32_imm32(2, count as u32, buf)?;
                    x64::encode_cld(buf)?;
                    x64::encode_rep_movsb(buf)?;
                    x64::encode_mov_r64_r64(7, 10, buf)?;
                    x64::encode_mov_r64_r64(6, 11, buf)
                })
                .expect("rep movsb");
        }

        self.emit
            .emit_with(|buf| x64::encode_add_r64_imm32(12, count as u32, buf))
            .expect("advance after static bytes");
    }

    /// Advance the output cursor by `count` bytes (no write).
    pub fn emit_enc_advance_output(&mut self, count: u32) {
        if count > 0 {
            self.emit
                .emit_with(|buf| x64::encode_add_r64_imm32(12, count as u32, buf))
                .expect("advance output");
        }
    }

    /// Load a value from the input struct at the given byte offset into a scratch register.
    /// The value is left in eax/rax (32-bit for W1/W2/W4, 64-bit for W8).
    pub fn emit_enc_load_from_input(&mut self, offset: u32, width: Width) {
        match width {
            Width::W1 => self
                .emit
                .emit_with(|buf| {
                    x64::encode_movzx_r32_rm8(
                        0,
                        x64::Operand::Mem(Mem {
                            base: 14,
                            disp: offset as i32,
                        }),
                        buf,
                    )
                })
                .expect("enc load 1"),
            Width::W2 => self
                .emit
                .emit_with(|buf| {
                    x64::encode_movzx_r32_rm16(
                        0,
                        x64::Operand::Mem(Mem {
                            base: 14,
                            disp: offset as i32,
                        }),
                        buf,
                    )
                })
                .expect("enc load 2"),
            Width::W4 => self
                .emit
                .emit_with(|buf| {
                    x64::encode_mov_r32_m(
                        0,
                        Mem {
                            base: 14,
                            disp: offset as i32,
                        },
                        buf,
                    )
                })
                .expect("enc load 4"),
            Width::W8 => self
                .emit
                .emit_with(|buf| {
                    x64::encode_mov_r64_m(
                        0,
                        Mem {
                            base: 14,
                            disp: offset as i32,
                        },
                        buf,
                    )
                })
                .expect("enc load 8"),
        }
    }

    /// Store a value from eax/rax to the output buffer and advance the cursor.
    pub fn emit_enc_store_to_output(&mut self, width: Width) {
        match width {
            Width::W1 => self
                .emit
                .emit_with(|buf| {
                    x64::encode_mov_m_r8(12, 0, 0, buf)?;
                    x64::encode_add_r64_imm32(12, 1, buf)
                })
                .expect("enc store 1"),
            Width::W2 => self
                .emit
                .emit_with(|buf| {
                    x64::encode_mov_m_r16(12, 0, 0, buf)?;
                    x64::encode_add_r64_imm32(12, 2, buf)
                })
                .expect("enc store 2"),
            Width::W4 => self
                .emit
                .emit_with(|buf| {
                    x64::encode_mov_m_r32(Mem { base: 12, disp: 0 }, 0, buf)?;
                    x64::encode_add_r64_imm32(12, 4, buf)
                })
                .expect("enc store 4"),
            Width::W8 => self
                .emit
                .emit_with(|buf| {
                    x64::encode_mov_m_r64(Mem { base: 12, disp: 0 }, 0, buf)?;
                    x64::encode_add_r64_imm32(12, 8, buf)
                })
                .expect("enc store 8"),
        }
    }

    /// Commit and finalize the assembler, returning the executable buffer.
    ///
    /// All functions must have been completed with `end_func` before calling this.
    pub fn finalize(self) -> FinalizedEmission {
        self.emit.finalize().expect("failed to finalize assembly")
    }
}
