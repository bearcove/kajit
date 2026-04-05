use kajit_emit::x64::{self, Emitter, FinalizedEmission, LabelId, Mem};

use crate::context::{CTX_ERROR_CODE, ErrorCode};

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
    pub cursor_enc: u8,
    /// Register encoding for the cached input_end.
    pub end_enc: u8,
    /// Register encoding for the output pointer.
    pub output_enc: u8,
    /// Register encoding for the context pointer.
    pub ctx_enc: u8,
}

impl EmitCtx {
    /// Create an EmitCtx for regalloc3-driven lowering.
    ///
    /// `extra_stack` is the number of bytes needed for spill slots + user slots + edge temps.
    /// `base_frame` is the offset from rsp where spill slots begin (past callee-saved saves).
    /// `is_leaf` controls whether the function needs callee-save overhead.
    pub fn new_regalloc(extra_stack: u32, base_frame: u32, is_leaf: bool) -> Self {
        let frame_size = (base_frame + extra_stack + 15) & !15;
        let mut emit = Emitter::new();
        emit.enable_capture();
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
    #[allow(dead_code)]
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
    #[allow(dead_code)]
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
            .emit_add_r64_imm32(cursor, n)
            .expect("advance cursor");
    }

    pub fn finalize(mut self) -> (FinalizedEmission, Option<kajit_emit::x64_asm::Program>) {
        let asm_program = self.emit.take_captured_program();
        let buf = self.emit.finalize().expect("failed to finalize assembly");
        (buf, asm_program)
    }
}
