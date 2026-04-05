use kajit_emit::x64::{self, Emitter, FinalizedEmission, LabelId, Mem};

use crate::context::{CTX_ERROR_CODE, CTX_INPUT_END, CTX_INPUT_PTR, ErrorCode};

/// Base frame size (non-leaf).
///
/// - System V AMD64: `rbp + rbx + r12 + r13 + r14 + r15` = 48 bytes
/// - Windows x64: `shadow(32) + rbp + rbx + r12 + r13 + r14 + r15` = 80 bytes
///
/// The Windows layout reserves 32 bytes at [rsp+0..31] as the callee home
/// (shadow) space required by the Windows x64 ABI before every call.
#[cfg(not(windows))]
pub const BASE_FRAME: u32 = 48;
#[cfg(windows)]
pub const BASE_FRAME: u32 = 80;

/// Base frame size for leaf functions (skip rbp save, minimal callee-saved).
/// Only rbx is saved (as potential callee-saved for regalloc).
#[cfg(not(windows))]
pub const LEAF_BASE_FRAME: u32 = 16;
#[cfg(windows)]
pub const LEAF_BASE_FRAME: u32 = 48;

/// Emission context — wraps the assembler plus bookkeeping labels.
pub struct EmitCtx {
    pub emit: Emitter,
    pub error_exit: LabelId,
    /// Offset from rsp where spill slots begin.
    pub base_frame: u32,
    /// Total frame size (base + extra, 16-byte aligned).
    pub frame_size: u32,
    /// Whether this is a leaf function (no call instructions).
    pub is_leaf: bool,
    /// Register encoding for the cached cursor (input_ptr).
    /// 12 (r12) for legacy, configurable for regalloc3.
    pub cursor_enc: u8,
    /// Register encoding for the cached input_end.
    /// 13 (r13) for legacy, configurable for regalloc3.
    pub end_enc: u8,
    /// Register encoding for the output pointer.
    /// 14 (r14) for legacy, configurable for regalloc3.
    pub output_enc: u8,
    /// Register encoding for the context pointer.
    /// 15 (r15) for legacy, configurable for regalloc3.
    pub ctx_enc: u8,
}

// Legacy register assignments (callee-saved across all platforms):
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
    /// Create a legacy EmitCtx with fixed register assignments.
    /// Used by the regalloc2 backend.
    pub fn new(extra_stack: u32) -> Self {
        let frame_size = (BASE_FRAME + extra_stack + 15) & !15;
        let mut emit = Emitter::new();
        let error_exit = emit.new_label();

        EmitCtx {
            emit,
            error_exit,
            base_frame: BASE_FRAME,
            frame_size,
            is_leaf: false,
            cursor_enc: 12,
            end_enc: 13,
            output_enc: 14,
            ctx_enc: 15,
        }
    }

    /// Create an EmitCtx for regalloc3-driven lowering.
    ///
    /// `extra_stack` is the number of bytes needed for spill slots + user slots + edge temps.
    /// `base_frame` is the offset from rsp where spill slots begin (past callee-saved saves).
    /// `is_leaf` controls whether the function needs callee-save overhead.
    pub fn new_regalloc(extra_stack: u32, base_frame: u32, is_leaf: bool) -> Self {
        let frame_size = (base_frame + extra_stack + 15) & !15;
        let mut emit = Emitter::new();
        let error_exit = emit.new_label();

        EmitCtx {
            emit,
            error_exit,
            base_frame,
            frame_size,
            is_leaf,
            // These will be set by the regalloc3 backend based on leaf/non-leaf
            cursor_enc: 12,
            end_enc: 13,
            output_enc: 14,
            ctx_enc: 15,
        }
    }

    /// Emit a function prologue. Returns the entry offset and a fresh error_exit label.
    ///
    /// The returned error_exit label must be passed to `end_func` when done emitting
    /// this function's body.
    ///
    /// # Register assignments after prologue (legacy)
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
        let ctx_enc = self.ctx_enc;
        let cursor_enc = self.cursor_enc;

        #[cfg(not(windows))]
        self.emit
            .emit_with(|buf| {
                x64::encode_mov_m_r64(
                    Mem {
                        base: ctx_enc,
                        disp: CTX_INPUT_PTR as i32,
                    },
                    cursor_enc,
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
                        base: ctx_enc,
                        disp: CTX_INPUT_PTR as i32,
                    },
                    cursor_enc,
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

    pub fn current_source_location(&self) -> kajit_emit::SourceLocation {
        self.emit.current_source_location()
    }

    /// Emit an unconditional branch to the given label.
    pub fn emit_branch(&mut self, label: LabelId) {
        self.emit.emit_jmp_label(label).expect("jmp");
    }

    /// Emit a bounds check: verify that at least `count` bytes remain in the
    /// input buffer. Branches to the error exit with UnexpectedEof on failure.
    pub fn emit_bounds_check(&mut self, count: u32) {
        let cursor = self.cursor_enc;
        let end = self.end_enc;
        let eof_label = self.emit.new_label();
        if count == 1 {
            self.emit
                .emit_with(|buf| x64::encode_cmp_r64_r64(cursor, end, buf))
                .expect("bounds check count=1");
            self.emit
                .emit_jae_label(eof_label)
                .expect("bounds check eof");
        } else {
            let count = count as i32;
            self.emit
                .emit_with(|buf| {
                    x64::encode_mov_r64_r64(10, end, buf)?;
                    x64::encode_sub_r64_r64(10, cursor, buf)?;
                    x64::encode_cmp_r64_imm32(10, count as u32, buf)
                })
                .expect("bounds check");
            self.emit
                .emit_jbe_label(eof_label)
                .expect("bounds check eof");
        }
        // EOF error path
        let past_eof = self.emit.new_label();
        self.emit.emit_jmp_label(past_eof).expect("jmp");
        self.emit.bind_label(eof_label).expect("bind eof");
        let eof_code = ErrorCode::UnexpectedEof as u32;
        let ctx = self.ctx_enc;
        self.emit
            .emit_with(|buf| {
                x64::encode_mov_r32_imm32(10, eof_code, buf)?;
                x64::encode_mov_m_r32(
                    x64::Mem {
                        base: ctx,
                        disp: CTX_ERROR_CODE as i32,
                    },
                    10,
                    buf,
                )
            })
            .expect("mov error code");
        let error_exit = self.error_exit;
        self.emit.emit_jmp_label(error_exit).expect("jmp");
        self.emit.bind_label(past_eof).expect("bind past_eof");
    }

    /// Emit an error (write error code to ctx, jump to error_exit).
    pub fn emit_error(&mut self, code: crate::context::ErrorCode) {
        let error_exit = self.error_exit;
        let error_code = code as i32;
        let ctx = self.ctx_enc;
        self.emit
            .emit_with(|buf| {
                x64::encode_mov_r32_imm32(10, error_code as u32, buf)?;
                x64::encode_mov_m_r32(
                    Mem {
                        base: ctx,
                        disp: CTX_ERROR_CODE as i32,
                    },
                    10,
                    buf,
                )
            })
            .expect("write error code");
        self.emit.emit_jmp_label(error_exit).expect("jump error");
    }

    /// Emit an error with an explicit ctx register encoding.
    pub fn emit_error_with_ctx(&mut self, code: crate::context::ErrorCode, ctx_enc: u8) {
        let error_exit = self.error_exit;
        let error_code = code as i32;
        self.emit
            .emit_with(|buf| {
                x64::encode_mov_r32_imm32(10, error_code as u32, buf)?;
                x64::encode_mov_m_r32(
                    Mem {
                        base: ctx_enc,
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
        let cursor = self.cursor_enc;
        self.emit
            .emit_with(|buf| x64::encode_add_r64_imm32(cursor, n, buf))
            .expect("advance cursor");
    }

    pub fn finalize(self) -> FinalizedEmission {
        self.emit.finalize().expect("failed to finalize assembly")
    }
}
