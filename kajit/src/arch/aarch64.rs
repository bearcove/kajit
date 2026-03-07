use kajit_emit::aarch64::{self, Emitter, LabelId, Reg};

use crate::context::{
    CTX_ERROR_CODE, CTX_INPUT_END, CTX_INPUT_PTR, ENC_ERROR_CODE, ENC_OUTPUT_END, ENC_OUTPUT_PTR,
    ErrorCode,
};
use crate::jit_f64;
use crate::recipe::{ErrorTarget, Op, Recipe, Slot, Width};

/// Base frame size: 3 pairs of callee-saved registers = 48 bytes.
pub const BASE_FRAME: u32 = 48;
/// Maximum base frame size for regalloc-aware lowering when saving x23..x28.
pub const REGALLOC_BASE_FRAME: u32 = 96;

/// Emission context — wraps the assembler plus bookkeeping labels.
pub struct EmitCtx {
    pub emit: Emitter,
    pub error_exit: LabelId,
    pub entry: u32,
    pub base_frame: u32,
    /// Total frame size (base + extra, 16-byte aligned).
    pub frame_size: u32,
}

// Register assignments (all callee-saved):
//   x19 = cached input_ptr
//   x20 = cached input_end
//   x21 = out pointer
//   x22 = ctx pointer

impl EmitCtx {
    /// Create a new EmitCtx. Does not emit any code.
    ///
    /// `extra_stack` is the number of additional bytes the format needs on the
    /// stack (e.g. 32 for JSON's bitset + key_ptr + key_len + peek_byte).
    /// The total frame is rounded up to 16-byte alignment.
    ///
    /// Call `begin_func()` to emit a function prologue.
    pub fn new(extra_stack: u32) -> Self {
        Self::new_with_base(extra_stack, BASE_FRAME)
    }

    /// Create an EmitCtx for regalloc-driven lowering that saves extra
    /// callee-saved register pairs (x23..x28) as needed.
    pub fn new_regalloc(extra_stack: u32, extra_saved_pairs: u32) -> Self {
        assert!(
            extra_saved_pairs <= 3,
            "aarch64 regalloc supports at most 3 extra callee-saved pairs, got {extra_saved_pairs}"
        );
        Self::new_with_base(extra_stack, BASE_FRAME + extra_saved_pairs * 16)
    }

    fn new_with_base(extra_stack: u32, base_frame: u32) -> Self {
        let frame_size = (base_frame + extra_stack + 15) & !15;
        let mut emit = Emitter::new();
        let error_exit = emit.new_label();
        let entry = 0u32;

        EmitCtx {
            emit,
            error_exit,
            entry,
            base_frame,
            frame_size,
        }
    }

    // ── Call helpers ──────────────────────────────────────────────────
    //
    // These small helpers factor out the repeated patterns around function
    // calls in the JIT: flushing/reloading the cached cursor, loading a
    // function pointer, and checking the error flag.

    /// Load a function pointer into x8 and call it via `blr x8`.
    fn emit_call_fn_ptr(&mut self, ptr: *const u8) {
        self.emit_load_imm64(Reg::X8, ptr as u64);
        self.emit
            .emit_word(aarch64::encode_blr(Reg::X8).expect("blr"));
    }

    /// Load a 64-bit immediate into a register using movz + 3×movk.
    pub(crate) fn emit_load_imm64(&mut self, rd: Reg, val: u64) {
        self.emit.emit_word(
            aarch64::encode_movz(aarch64::Width::X64, rd, (val & 0xFFFF) as u16, 0).expect("movz"),
        );
        self.emit.emit_word(
            aarch64::encode_movk(aarch64::Width::X64, rd, ((val >> 16) & 0xFFFF) as u16, 16)
                .expect("movk"),
        );
        self.emit.emit_word(
            aarch64::encode_movk(aarch64::Width::X64, rd, ((val >> 32) & 0xFFFF) as u16, 32)
                .expect("movk"),
        );
        self.emit.emit_word(
            aarch64::encode_movk(aarch64::Width::X64, rd, ((val >> 48) & 0xFFFF) as u16, 48)
                .expect("movk"),
        );
    }

    fn emit_add_imm_any(&mut self, rd: Reg, rn: Reg, imm: u32) {
        if imm <= 0x0fff {
            self.emit.emit_word(
                aarch64::encode_add_imm(aarch64::Width::X64, rd, rn, imm as u16, false)
                    .expect("add"),
            );
            return;
        }
        if (imm & 0x0fff) == 0 {
            let shifted = imm >> 12;
            if shifted <= 0x0fff {
                self.emit.emit_word(
                    aarch64::encode_add_imm(aarch64::Width::X64, rd, rn, shifted as u16, true)
                        .expect("add"),
                );
                return;
            }
        }

        self.emit_load_imm64(Reg::X9, imm as u64);
        self.emit
            .emit_word(aarch64::encode_add_reg(aarch64::Width::X64, rd, rn, Reg::X9).expect("add"));
    }

    fn emit_sub_imm_any(&mut self, rd: Reg, rn: Reg, imm: u32) {
        if imm <= 0x0fff {
            self.emit.emit_word(
                aarch64::encode_sub_imm(aarch64::Width::X64, rd, rn, imm as u16, false)
                    .expect("sub"),
            );
            return;
        }
        if (imm & 0x0fff) == 0 {
            let shifted = imm >> 12;
            if shifted <= 0x0fff {
                self.emit.emit_word(
                    aarch64::encode_sub_imm(aarch64::Width::X64, rd, rn, shifted as u16, true)
                        .expect("sub"),
                );
                return;
            }
        }

        self.emit_load_imm64(Reg::X9, imm as u64);
        self.emit
            .emit_word(aarch64::encode_sub_reg(aarch64::Width::X64, rd, rn, Reg::X9).expect("sub"));
    }

    /// Flush the cached input cursor (x19) back to ctx.input_ptr.
    fn emit_flush_input_cursor(&mut self) {
        self.emit.emit_word(
            aarch64::encode_str_imm(
                aarch64::Width::X64,
                Reg::X19,
                Reg::X22,
                CTX_INPUT_PTR as u32,
            )
            .expect("str"),
        );
    }

    /// Reload the cached input cursor from ctx and check the error flag.
    /// Branches to `error_exit` if `ctx.error.code != 0`.
    fn emit_reload_cursor_and_check_error(&mut self) {
        let error_exit = self.error_exit;
        self.emit.emit_word(
            aarch64::encode_ldr_imm(
                aarch64::Width::X64,
                Reg::X19,
                Reg::X22,
                CTX_INPUT_PTR as u32,
            )
            .expect("ldr"),
        );
        self.emit.emit_word(
            aarch64::encode_ldr_imm(
                aarch64::Width::W32,
                Reg::X9,
                Reg::X22,
                CTX_ERROR_CODE as u32,
            )
            .expect("ldr"),
        );
        self.emit
            .emit_cbnz_label(aarch64::Width::W32, Reg::X9, error_exit)
            .expect("cbnz");
    }

    /// Check the error flag without reloading the cursor.
    /// Branches to `error_exit` if `ctx.error.code != 0`.
    fn emit_check_error(&mut self) {
        let error_exit = self.error_exit;
        self.emit.emit_word(
            aarch64::encode_ldr_imm(
                aarch64::Width::W32,
                Reg::X9,
                Reg::X22,
                CTX_ERROR_CODE as u32,
            )
            .expect("ldr"),
        );
        self.emit
            .emit_cbnz_label(aarch64::Width::W32, Reg::X9, error_exit)
            .expect("cbnz");
    }

    /// Flush the cached output cursor (x19) back to ctx for encoding.
    fn emit_enc_flush_output_cursor(&mut self) {
        self.emit.emit_word(
            aarch64::encode_str_imm(
                aarch64::Width::X64,
                Reg::X19,
                Reg::X22,
                ENC_OUTPUT_PTR as u32,
            )
            .expect("str"),
        );
    }

    /// Reload output_ptr and output_end from ctx and check the error flag.
    fn emit_enc_reload_and_check_error(&mut self) {
        let error_exit = self.error_exit;
        self.emit.emit_word(
            aarch64::encode_ldr_imm(
                aarch64::Width::X64,
                Reg::X19,
                Reg::X22,
                ENC_OUTPUT_PTR as u32,
            )
            .expect("ldr"),
        );
        self.emit.emit_word(
            aarch64::encode_ldr_imm(
                aarch64::Width::X64,
                Reg::X20,
                Reg::X22,
                ENC_OUTPUT_END as u32,
            )
            .expect("ldr"),
        );
        self.emit.emit_word(
            aarch64::encode_ldr_imm(
                aarch64::Width::W32,
                Reg::X9,
                Reg::X22,
                ENC_ERROR_CODE as u32,
            )
            .expect("ldr"),
        );
        self.emit
            .emit_cbnz_label(aarch64::Width::W32, Reg::X9, error_exit)
            .expect("cbnz");
    }

    /// Emit a function prologue. Returns the entry offset and a fresh error_exit label.
    ///
    /// The returned error_exit label must be passed to `end_func` when done emitting
    /// this function's body.
    ///
    /// # Register assignments after prologue
    /// - x19 = cached input_ptr
    /// - x20 = cached input_end
    /// - x21 = out pointer
    /// - x22 = ctx pointer
    pub fn begin_func(&mut self) -> (u32, LabelId) {
        let error_exit = self.emit.new_label();
        let entry = self.emit.current_offset();
        let frame_size = self.frame_size;

        self.emit_sub_imm_any(Reg::SP, Reg::SP, frame_size);
        self.emit.emit_word(
            aarch64::encode_stp(aarch64::Width::X64, Reg::X29, Reg::X30, Reg::SP, 0).expect("stp"),
        );
        self.emit.emit_word(
            aarch64::encode_stp(aarch64::Width::X64, Reg::X19, Reg::X20, Reg::SP, 16).expect("stp"),
        );
        self.emit.emit_word(
            aarch64::encode_stp(aarch64::Width::X64, Reg::X21, Reg::X22, Reg::SP, 32).expect("stp"),
        );
        let extra_pairs = ((self.base_frame - BASE_FRAME) / 16) as usize;
        assert!(
            extra_pairs <= 3,
            "unsupported extra callee-saved pair count"
        );
        if extra_pairs >= 1 {
            self.emit.emit_word(
                aarch64::encode_stp(aarch64::Width::X64, Reg::X23, Reg::X24, Reg::SP, 48)
                    .expect("stp"),
            );
        }
        if extra_pairs >= 2 {
            self.emit.emit_word(
                aarch64::encode_stp(aarch64::Width::X64, Reg::X25, Reg::X26, Reg::SP, 64)
                    .expect("stp"),
            );
        }
        if extra_pairs >= 3 {
            self.emit.emit_word(
                aarch64::encode_stp(aarch64::Width::X64, Reg::X27, Reg::X28, Reg::SP, 80)
                    .expect("stp"),
            );
        }
        self.emit.emit_word(
            aarch64::encode_add_imm(aarch64::Width::X64, Reg::X29, Reg::SP, 0, false).expect("add"),
        );
        self.emit.emit_word(
            aarch64::encode_mov_reg(aarch64::Width::X64, Reg::X21, Reg::X0).expect("mov"),
        );
        self.emit.emit_word(
            aarch64::encode_mov_reg(aarch64::Width::X64, Reg::X22, Reg::X1).expect("mov"),
        );
        self.emit.emit_word(
            aarch64::encode_ldr_imm(
                aarch64::Width::X64,
                Reg::X19,
                Reg::X22,
                CTX_INPUT_PTR as u32,
            )
            .expect("ldr"),
        );
        self.emit.emit_word(
            aarch64::encode_ldr_imm(
                aarch64::Width::X64,
                Reg::X20,
                Reg::X22,
                CTX_INPUT_END as u32,
            )
            .expect("ldr"),
        );

        self.error_exit = error_exit;
        (entry, error_exit)
    }

    /// Emit the success epilogue and error exit for the current function.
    ///
    /// `error_exit` must be the label returned by the corresponding `begin_func` call.
    pub fn end_func(&mut self, error_exit: LabelId) {
        let frame_size = self.frame_size;

        let extra_pairs = ((self.base_frame - BASE_FRAME) / 16) as usize;
        assert!(
            extra_pairs <= 3,
            "unsupported extra callee-saved pair count"
        );

        self.emit.emit_word(
            aarch64::encode_str_imm(
                aarch64::Width::X64,
                Reg::X19,
                Reg::X22,
                CTX_INPUT_PTR as u32,
            )
            .expect("str"),
        );
        if extra_pairs >= 3 {
            self.emit.emit_word(
                aarch64::encode_ldp(aarch64::Width::X64, Reg::X27, Reg::X28, Reg::SP, 80)
                    .expect("ldp"),
            );
        }
        if extra_pairs >= 2 {
            self.emit.emit_word(
                aarch64::encode_ldp(aarch64::Width::X64, Reg::X25, Reg::X26, Reg::SP, 64)
                    .expect("ldp"),
            );
        }
        if extra_pairs >= 1 {
            self.emit.emit_word(
                aarch64::encode_ldp(aarch64::Width::X64, Reg::X23, Reg::X24, Reg::SP, 48)
                    .expect("ldp"),
            );
        }
        self.emit.emit_word(
            aarch64::encode_ldp(aarch64::Width::X64, Reg::X21, Reg::X22, Reg::SP, 32).expect("ldp"),
        );
        self.emit.emit_word(
            aarch64::encode_ldp(aarch64::Width::X64, Reg::X19, Reg::X20, Reg::SP, 16).expect("ldp"),
        );
        self.emit.emit_word(
            aarch64::encode_ldp(aarch64::Width::X64, Reg::X29, Reg::X30, Reg::SP, 0).expect("ldp"),
        );
        self.emit_add_imm_any(Reg::SP, Reg::SP, frame_size);
        self.emit
            .emit_word(aarch64::encode_ret(Reg::X30).expect("ret"));
        self.emit.bind_label(error_exit).expect("bind");
        if extra_pairs >= 3 {
            self.emit.emit_word(
                aarch64::encode_ldp(aarch64::Width::X64, Reg::X27, Reg::X28, Reg::SP, 80)
                    .expect("ldp"),
            );
        }
        if extra_pairs >= 2 {
            self.emit.emit_word(
                aarch64::encode_ldp(aarch64::Width::X64, Reg::X25, Reg::X26, Reg::SP, 64)
                    .expect("ldp"),
            );
        }
        if extra_pairs >= 1 {
            self.emit.emit_word(
                aarch64::encode_ldp(aarch64::Width::X64, Reg::X23, Reg::X24, Reg::SP, 48)
                    .expect("ldp"),
            );
        }
        self.emit.emit_word(
            aarch64::encode_ldp(aarch64::Width::X64, Reg::X21, Reg::X22, Reg::SP, 32).expect("ldp"),
        );
        self.emit.emit_word(
            aarch64::encode_ldp(aarch64::Width::X64, Reg::X19, Reg::X20, Reg::SP, 16).expect("ldp"),
        );
        self.emit.emit_word(
            aarch64::encode_ldp(aarch64::Width::X64, Reg::X29, Reg::X30, Reg::SP, 0).expect("ldp"),
        );
        self.emit_add_imm_any(Reg::SP, Reg::SP, frame_size);
        self.emit
            .emit_word(aarch64::encode_ret(Reg::X30).expect("ret"));
    }

    /// Emit a call to another emitted function.
    ///
    /// Convention: x0 = out + field_offset, x1 = ctx (same as our entry convention).
    /// Flushes cursor before call, reloads after, checks error.
    ///
    /// r[impl callconv.inter-function]
    pub fn emit_call_emitted_func(&mut self, label: LabelId, field_offset: u32) {
        self.emit_flush_input_cursor();
        self.emit.emit_word(
            aarch64::encode_add_imm(
                aarch64::Width::X64,
                Reg::X0,
                Reg::X21,
                field_offset as u16,
                false,
            )
            .expect("add"),
        );
        self.emit.emit_word(
            aarch64::encode_mov_reg(aarch64::Width::X64, Reg::X1, Reg::X22).expect("mov"),
        );
        self.emit.emit_bl_label(label).expect("bl");
        self.emit_reload_cursor_and_check_error();
    }

    /// Emit a call to an intrinsic function.
    ///
    /// Before the call: flushes the cached input_ptr back to ctx.
    /// Sets up args: x0 = ctx, x1 = out + field_offset.
    /// After the call: reloads input_ptr from ctx, checks error slot.
    pub fn emit_call_intrinsic(&mut self, fn_ptr: *const u8, field_offset: u32) {
        self.emit_flush_input_cursor();
        self.emit.emit_word(
            aarch64::encode_mov_reg(aarch64::Width::X64, Reg::X0, Reg::X22).expect("mov"),
        );
        self.emit.emit_word(
            aarch64::encode_add_imm(
                aarch64::Width::X64,
                Reg::X1,
                Reg::X21,
                field_offset as u16,
                false,
            )
            .expect("add"),
        );
        self.emit_call_fn_ptr(fn_ptr);
        self.emit_reload_cursor_and_check_error();
    }

    /// Emit a call to an intrinsic that takes only ctx as argument.
    /// Flushes cursor, calls, reloads cursor, checks error.
    pub fn emit_call_intrinsic_ctx_only(&mut self, fn_ptr: *const u8) {
        self.emit_flush_input_cursor();
        self.emit.emit_word(
            aarch64::encode_mov_reg(aarch64::Width::X64, Reg::X0, Reg::X22).expect("mov"),
        );
        self.emit_call_fn_ptr(fn_ptr);
        self.emit_reload_cursor_and_check_error();
    }

    /// Emit a call to an intrinsic that takes (ctx, &mut stack_slot).
    /// x0 = ctx, x1 = sp + sp_offset. Flushes/reloads cursor, checks error.
    pub fn emit_call_intrinsic_ctx_and_stack_out(&mut self, fn_ptr: *const u8, sp_offset: u32) {
        self.emit_flush_input_cursor();
        self.emit.emit_word(
            aarch64::encode_mov_reg(aarch64::Width::X64, Reg::X0, Reg::X22).expect("mov"),
        );
        self.emit.emit_word(
            aarch64::encode_add_imm(
                aarch64::Width::X64,
                Reg::X1,
                Reg::SP,
                sp_offset as u16,
                false,
            )
            .expect("add"),
        );
        self.emit_call_fn_ptr(fn_ptr);
        self.emit_reload_cursor_and_check_error();
    }

    /// Emit a call to an intrinsic that takes (ctx, out_ptr1, out_ptr2).
    /// x0 = ctx, x1 = sp + sp_offset1, x2 = sp + sp_offset2.
    pub fn emit_call_intrinsic_ctx_and_two_stack_outs(
        &mut self,
        fn_ptr: *const u8,
        sp_offset1: u32,
        sp_offset2: u32,
    ) {
        self.emit_flush_input_cursor();
        self.emit.emit_word(
            aarch64::encode_mov_reg(aarch64::Width::X64, Reg::X0, Reg::X22).expect("mov"),
        );
        self.emit.emit_word(
            aarch64::encode_add_imm(
                aarch64::Width::X64,
                Reg::X1,
                Reg::SP,
                sp_offset1 as u16,
                false,
            )
            .expect("add"),
        );
        self.emit.emit_word(
            aarch64::encode_add_imm(
                aarch64::Width::X64,
                Reg::X2,
                Reg::SP,
                sp_offset2 as u16,
                false,
            )
            .expect("add"),
        );
        self.emit_call_fn_ptr(fn_ptr);
        self.emit_reload_cursor_and_check_error();
    }

    /// Emit a call to a pure function (no ctx, no flush/reload/error-check).
    /// Used for key_equals: x0=key_ptr, x1=key_len, x2=expected_ptr, x3=expected_len.
    /// Return value is in x0.
    pub fn emit_call_pure_4arg(
        &mut self,
        fn_ptr: *const u8,
        arg0_sp_offset: u32,
        arg1_sp_offset: u32,
        expected_ptr: *const u8,
        expected_len: u32,
    ) {
        self.emit.emit_word(
            aarch64::encode_ldr_imm(aarch64::Width::X64, Reg::X0, Reg::SP, arg0_sp_offset)
                .expect("ldr"),
        );
        self.emit.emit_word(
            aarch64::encode_ldr_imm(aarch64::Width::X64, Reg::X1, Reg::SP, arg1_sp_offset)
                .expect("ldr"),
        );
        let val = expected_ptr as u64;
        self.emit.emit_word(
            aarch64::encode_movz(aarch64::Width::X64, Reg::X2, (val & 0xFFFF) as u16, 0)
                .expect("movz"),
        );
        self.emit.emit_word(
            aarch64::encode_movk(
                aarch64::Width::X64,
                Reg::X2,
                ((val >> 16) & 0xFFFF) as u16,
                16,
            )
            .expect("movk"),
        );
        self.emit.emit_word(
            aarch64::encode_movk(
                aarch64::Width::X64,
                Reg::X2,
                ((val >> 32) & 0xFFFF) as u16,
                32,
            )
            .expect("movk"),
        );
        self.emit.emit_word(
            aarch64::encode_movk(
                aarch64::Width::X64,
                Reg::X2,
                ((val >> 48) & 0xFFFF) as u16,
                48,
            )
            .expect("movk"),
        );
        self.emit.emit_word(
            aarch64::encode_movz(aarch64::Width::X64, Reg::X3, expected_len as u16, 0)
                .expect("movz"),
        );
        self.emit_call_fn_ptr(fn_ptr);
    }

    /// Allocate a new dynamic label.
    pub fn new_label(&mut self) -> LabelId {
        self.emit.new_label()
    }

    /// Bind a dynamic label at the current position.
    pub fn bind_label(&mut self, label: LabelId) {
        self.emit.bind_label(label).expect("bind_label failed");
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
        self.emit.emit_b_label(label).expect("emit_branch failed");
    }

    /// Write an error code to ctx and branch to error_exit.
    pub fn emit_set_error(&mut self, code: ErrorCode) {
        let error_exit = self.error_exit;
        let error_code = code as u32;
        // movz w9, #error_code
        self.emit.emit_word(
            aarch64::encode_movz(aarch64::Width::W32, Reg::X9, error_code as u16, 0)
                .expect("encode_movz"),
        );
        // str w9, [x22, #CTX_ERROR_CODE]
        self.emit.emit_word(
            aarch64::encode_str_imm(
                aarch64::Width::W32,
                Reg::X9,
                Reg::X22,
                CTX_ERROR_CODE as u32,
            )
            .expect("encode_str_imm"),
        );
        // b =>error_exit
        self.emit.emit_b_label(error_exit).expect("emit_b_label");
    }

    /// Compute len = cursor - [sp+start_slot], store to [sp+len_slot], advance cursor past `"`.
    pub fn emit_compute_key_len_and_advance(&mut self, start_sp_offset: u32, len_sp_offset: u32) {
        self.emit.emit_word(
            aarch64::encode_ldr_imm(aarch64::Width::X64, Reg::X9, Reg::SP, start_sp_offset)
                .expect("ldr"),
        );
        self.emit.emit_word(
            aarch64::encode_sub_reg(aarch64::Width::X64, Reg::X10, Reg::X19, Reg::X9).expect("sub"),
        );
        self.emit.emit_word(
            aarch64::encode_str_imm(aarch64::Width::X64, Reg::X10, Reg::SP, len_sp_offset)
                .expect("str"),
        );
        self.emit.emit_word(
            aarch64::encode_add_imm(aarch64::Width::X64, Reg::X19, Reg::X19, 1, false)
                .expect("add"),
        );
    }

    /// Emit `cbnz x0, label` — branch if x0 is nonzero.
    pub fn emit_cbnz_x0(&mut self, label: LabelId) {
        self.emit
            .emit_cbnz_label(aarch64::Width::X64, Reg::X0, label)
            .expect("emit_cbnz_label");
    }

    /// Zero a 64-bit stack slot at sp + offset.
    pub fn emit_zero_stack_slot(&mut self, sp_offset: u32) {
        self.emit.emit_word(
            aarch64::encode_str_imm(aarch64::Width::X64, Reg::XZR, Reg::SP, sp_offset)
                .expect("str"),
        );
    }

    /// Load a byte from sp + sp_offset, compare with byte_val, branch if equal.
    pub fn emit_stack_byte_cmp_branch(&mut self, sp_offset: u32, byte_val: u8, label: LabelId) {
        let byte_val = byte_val as u32;
        self.emit
            .emit_word(aarch64::encode_ldrb_imm(Reg::X9, Reg::SP, sp_offset).expect("ldrb"));
        self.emit.emit_word(
            aarch64::encode_cmp_imm(aarch64::Width::W32, Reg::X9, byte_val as u16, false)
                .expect("cmp"),
        );
        self.emit
            .emit_b_cond_label(aarch64::Condition::Eq, label)
            .expect("b.eq");
    }

    /// Set bit `bit_index` in a 64-bit stack slot at sp + sp_offset.
    pub fn emit_set_bit_on_stack(&mut self, sp_offset: u32, bit_index: u32) {
        let mask = 1u64 << bit_index;
        let mask_lo = (mask & 0xFFFF) as u16;
        let mask_hi = ((mask >> 16) & 0xFFFF) as u16;

        self.emit.emit_word(
            aarch64::encode_ldr_imm(aarch64::Width::X64, Reg::X9, Reg::SP, sp_offset).expect("ldr"),
        );

        if mask <= 0xFFFF {
            self.emit.emit_word(
                aarch64::encode_movz(aarch64::Width::X64, Reg::X10, mask_lo, 0).expect("movz"),
            );
        } else {
            self.emit.emit_word(
                aarch64::encode_movz(aarch64::Width::X64, Reg::X10, mask_lo, 0).expect("movz"),
            );
            self.emit.emit_word(
                aarch64::encode_movk(aarch64::Width::X64, Reg::X10, mask_hi, 16).expect("movk"),
            );
        }

        self.emit.emit_word(
            aarch64::encode_orr_reg(
                aarch64::Width::X64,
                Reg::X9,
                Reg::X9,
                Reg::X10,
                aarch64::Shift::Lsl,
                0,
            )
            .expect("orr"),
        );
        self.emit.emit_word(
            aarch64::encode_str_imm(aarch64::Width::X64, Reg::X9, Reg::SP, sp_offset).expect("str"),
        );
    }

    /// Check that the 64-bit stack slot at sp + sp_offset equals expected_mask.
    /// If not, set MissingRequiredField error and branch to error_exit.
    pub fn emit_check_bitset(&mut self, sp_offset: u32, expected_mask: u64) {
        let error_exit = self.error_exit;
        let ok_label = self.emit.new_label();
        let mask_lo = (expected_mask & 0xFFFF) as u16;
        let mask_hi = ((expected_mask >> 16) & 0xFFFF) as u16;
        let error_code = crate::context::ErrorCode::MissingRequiredField as u32;

        self.emit.emit_word(
            aarch64::encode_ldr_imm(aarch64::Width::X64, Reg::X9, Reg::SP, sp_offset).expect("ldr"),
        );

        if expected_mask <= 0xFFFF {
            self.emit.emit_word(
                aarch64::encode_movz(aarch64::Width::X64, Reg::X10, mask_lo, 0).expect("movz"),
            );
        } else {
            self.emit.emit_word(
                aarch64::encode_movz(aarch64::Width::X64, Reg::X10, mask_lo, 0).expect("movz"),
            );
            self.emit.emit_word(
                aarch64::encode_movk(aarch64::Width::X64, Reg::X10, mask_hi, 16).expect("movk"),
            );
        }

        // Check that (bitset & mask) == mask — all required bits are set.
        // Extra bits (from optional/default fields) are ignored.
        self.emit.emit_word(
            aarch64::encode_and_reg(
                aarch64::Width::X64,
                Reg::X11,
                Reg::X9,
                Reg::X10,
                aarch64::Shift::Lsl,
                0,
            )
            .expect("and"),
        );
        self.emit.emit_word(
            aarch64::encode_cmp_reg(aarch64::Width::X64, Reg::X11, Reg::X10).expect("cmp"),
        );
        self.emit
            .emit_b_cond_label(aarch64::Condition::Eq, ok_label)
            .expect("b.eq");

        // Not all required fields were seen — write error and bail
        self.emit.emit_word(
            aarch64::encode_movz(aarch64::Width::W32, Reg::X9, error_code as u16, 0).expect("movz"),
        );
        self.emit.emit_word(
            aarch64::encode_str_imm(
                aarch64::Width::W32,
                Reg::X9,
                Reg::X22,
                CTX_ERROR_CODE as u32,
            )
            .expect("str"),
        );
        self.emit.emit_b_label(error_exit).expect("b");
        self.emit.bind_label(ok_label).expect("bind");
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

    // ── JSON string scanning (NEON vectorized) ────────────────────────

    /// Emit a vectorized scan loop that searches for `"` or `\` in the input.
    ///
    /// **Precondition**: x19 (cached input_ptr) points just after the opening `"`.
    /// **Postcondition**: x19 points at the `"` or `\` byte found, then branches
    /// to the corresponding label.
    ///
    /// Uses NEON cmeq + umaxv to process 16 bytes per iteration.
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
        let find_pos = self.emit.new_label();
        let found_at_offset = self.emit.new_label();

        self.emit
            .emit_word(aarch64::encode_movi_b16(1, 0x22).expect("movi"));
        self.emit
            .emit_word(aarch64::encode_movi_b16(2, 0x5C).expect("movi"));

        self.emit.bind_label(vector_loop).expect("bind");
        self.emit.emit_word(
            aarch64::encode_sub_reg(aarch64::Width::X64, Reg::X9, Reg::X20, Reg::X19).expect("sub"),
        );
        self.emit.emit_word(
            aarch64::encode_cmp_imm(aarch64::Width::X64, Reg::X9, 16, false).expect("cmp"),
        );
        self.emit
            .emit_b_cond_label(aarch64::Condition::Lo, scalar_tail)
            .expect("b.lo");

        self.emit
            .emit_word(aarch64::encode_ld1_b16(0, Reg::X19).expect("ld1"));
        self.emit
            .emit_word(aarch64::encode_cmeq_b16(3, 0, 1).expect("cmeq"));
        self.emit
            .emit_word(aarch64::encode_cmeq_b16(4, 0, 2).expect("cmeq"));
        self.emit
            .emit_word(aarch64::encode_orr_b16(3, 3, 4).expect("orr"));
        self.emit
            .emit_word(aarch64::encode_umaxv_b16(5, 3).expect("umaxv"));
        self.emit
            .emit_word(aarch64::encode_umov_b(Reg::X9, 5, 0).expect("umov"));
        self.emit
            .emit_cbz_label(aarch64::Width::W32, Reg::X9, advance_16)
            .expect("cbz");

        self.emit
            .emit_word(aarch64::encode_movz(aarch64::Width::X64, Reg::X10, 0, 0).expect("movz"));
        self.emit.bind_label(find_pos).expect("bind");
        self.emit
            .emit_word(aarch64::encode_ldrb_reg(Reg::X9, Reg::X19, Reg::X10).expect("ldrb"));
        self.emit.emit_word(
            aarch64::encode_cmp_imm(aarch64::Width::W32, Reg::X9, 0x22, false).expect("cmp"),
        );
        self.emit
            .emit_b_cond_label(aarch64::Condition::Eq, found_at_offset)
            .expect("b.eq");
        self.emit.emit_word(
            aarch64::encode_cmp_imm(aarch64::Width::W32, Reg::X9, 0x5C, false).expect("cmp"),
        );
        self.emit
            .emit_b_cond_label(aarch64::Condition::Eq, found_at_offset)
            .expect("b.eq");
        self.emit.emit_word(
            aarch64::encode_add_imm(aarch64::Width::X64, Reg::X10, Reg::X10, 1, false)
                .expect("add"),
        );
        self.emit.emit_b_label(find_pos).expect("b");

        self.emit.bind_label(found_at_offset).expect("bind");
        self.emit.emit_word(
            aarch64::encode_add_reg(aarch64::Width::X64, Reg::X19, Reg::X19, Reg::X10)
                .expect("add"),
        );
        self.emit.emit_word(
            aarch64::encode_cmp_imm(aarch64::Width::W32, Reg::X9, 0x22, false).expect("cmp"),
        );
        self.emit
            .emit_b_cond_label(aarch64::Condition::Eq, found_quote)
            .expect("b.eq");
        self.emit.emit_b_label(found_escape).expect("b");

        self.emit.bind_label(advance_16).expect("bind");
        self.emit.emit_word(
            aarch64::encode_add_imm(aarch64::Width::X64, Reg::X19, Reg::X19, 16, false)
                .expect("add"),
        );
        self.emit.emit_b_label(vector_loop).expect("b");

        self.emit.bind_label(scalar_tail).expect("bind");
        self.emit.emit_word(
            aarch64::encode_cmp_reg(aarch64::Width::X64, Reg::X19, Reg::X20).expect("cmp"),
        );
        self.emit
            .emit_b_cond_label(aarch64::Condition::Hs, unterminated)
            .expect("b.hs");
        self.emit
            .emit_word(aarch64::encode_ldrb_imm(Reg::X9, Reg::X19, 0).expect("ldrb"));
        self.emit.emit_word(
            aarch64::encode_cmp_imm(aarch64::Width::W32, Reg::X9, 0x22, false).expect("cmp"),
        );
        self.emit
            .emit_b_cond_label(aarch64::Condition::Eq, found_quote)
            .expect("b.eq");
        self.emit.emit_word(
            aarch64::encode_cmp_imm(aarch64::Width::W32, Reg::X9, 0x5C, false).expect("cmp"),
        );
        self.emit
            .emit_b_cond_label(aarch64::Condition::Eq, found_escape)
            .expect("b.eq");
        self.emit.emit_word(
            aarch64::encode_add_imm(aarch64::Width::X64, Reg::X19, Reg::X19, 1, false)
                .expect("add"),
        );
        self.emit.emit_b_label(scalar_tail).expect("b");
    }

    /// Inline skip-whitespace: loop over space/tab/newline/cr, advancing x19.
    /// No function call, no ctx flush.
    pub fn emit_inline_skip_ws(&mut self) {
        let ws_loop = self.emit.new_label();
        let ws_advance = self.emit.new_label();
        let ws_done = self.emit.new_label();

        self.emit.bind_label(ws_loop).expect("bind");
        self.emit.emit_word(
            aarch64::encode_cmp_reg(aarch64::Width::X64, Reg::X19, Reg::X20).expect("cmp"),
        );
        self.emit
            .emit_b_cond_label(aarch64::Condition::Hs, ws_done)
            .expect("b.hs");
        self.emit
            .emit_word(aarch64::encode_ldrb_imm(Reg::X9, Reg::X19, 0).expect("ldrb"));
        self.emit.emit_word(
            aarch64::encode_cmp_imm(aarch64::Width::W32, Reg::X9, b' ' as u16, false).expect("cmp"),
        );
        self.emit
            .emit_b_cond_label(aarch64::Condition::Eq, ws_advance)
            .expect("b.eq");
        self.emit.emit_word(
            aarch64::encode_cmp_imm(aarch64::Width::W32, Reg::X9, b'\n' as u16, false)
                .expect("cmp"),
        );
        self.emit
            .emit_b_cond_label(aarch64::Condition::Eq, ws_advance)
            .expect("b.eq");
        self.emit.emit_word(
            aarch64::encode_cmp_imm(aarch64::Width::W32, Reg::X9, b'\r' as u16, false)
                .expect("cmp"),
        );
        self.emit
            .emit_b_cond_label(aarch64::Condition::Eq, ws_advance)
            .expect("b.eq");
        self.emit.emit_word(
            aarch64::encode_cmp_imm(aarch64::Width::W32, Reg::X9, b'\t' as u16, false)
                .expect("cmp"),
        );
        self.emit
            .emit_b_cond_label(aarch64::Condition::Ne, ws_done)
            .expect("b.ne");
        self.emit.bind_label(ws_advance).expect("bind");
        self.emit.emit_word(
            aarch64::encode_add_imm(aarch64::Width::X64, Reg::X19, Reg::X19, 1, false)
                .expect("add"),
        );
        self.emit.emit_b_label(ws_loop).expect("b");
        self.emit.bind_label(ws_done).expect("bind");
    }

    /// Inline comma-or-end-array: skip whitespace, then check for ',' or ']'.
    /// Writes 0 (comma) or 1 (']') to stack at sp_offset. Errors on anything else.
    pub fn emit_inline_comma_or_end_array(&mut self, sp_offset: u32) {
        let error_exit = self.error_exit;
        let got_comma = self.emit.new_label();
        let got_end = self.emit.new_label();
        let done = self.emit.new_label();
        let error_code = ErrorCode::UnexpectedCharacter as u32;

        self.emit_inline_skip_ws();

        self.emit.emit_word(
            aarch64::encode_cmp_reg(aarch64::Width::X64, Reg::X19, Reg::X20).expect("cmp"),
        );
        self.emit
            .emit_b_cond_label(aarch64::Condition::Hs, error_exit)
            .expect("b.hs");
        self.emit
            .emit_word(aarch64::encode_ldrb_imm(Reg::X9, Reg::X19, 0).expect("ldrb"));
        self.emit.emit_word(
            aarch64::encode_add_imm(aarch64::Width::X64, Reg::X19, Reg::X19, 1, false)
                .expect("add"),
        );
        self.emit.emit_word(
            aarch64::encode_cmp_imm(aarch64::Width::W32, Reg::X9, b',' as u16, false).expect("cmp"),
        );
        self.emit
            .emit_b_cond_label(aarch64::Condition::Eq, got_comma)
            .expect("b.eq");
        self.emit.emit_word(
            aarch64::encode_cmp_imm(aarch64::Width::W32, Reg::X9, b']' as u16, false).expect("cmp"),
        );
        self.emit
            .emit_b_cond_label(aarch64::Condition::Eq, got_end)
            .expect("b.eq");
        self.emit.emit_word(
            aarch64::encode_movz(aarch64::Width::W32, Reg::X9, error_code as u16, 0).expect("movz"),
        );
        self.emit.emit_word(
            aarch64::encode_str_imm(aarch64::Width::W32, Reg::X9, Reg::X22, CTX_ERROR_CODE)
                .expect("str"),
        );
        self.emit.emit_b_label(error_exit).expect("b");
        self.emit.bind_label(got_comma).expect("bind");
        self.emit
            .emit_word(aarch64::encode_strb_imm(Reg::XZR, Reg::SP, sp_offset).expect("strb"));
        self.emit.emit_b_label(done).expect("b");
        self.emit.bind_label(got_end).expect("bind");
        self.emit
            .emit_word(aarch64::encode_movz(aarch64::Width::W32, Reg::X9, 1, 0).expect("movz"));
        self.emit
            .emit_word(aarch64::encode_strb_imm(Reg::X9, Reg::SP, sp_offset).expect("strb"));
        self.emit.bind_label(done).expect("bind");
    }

    /// Inline comma-or-end-object: skip whitespace, then check for ',' or '}'.
    /// Writes 0 (comma) or 1 ('}') to stack at sp_offset. Errors on anything else.
    pub fn emit_inline_comma_or_end_object(&mut self, sp_offset: u32) {
        let error_exit = self.error_exit;
        let got_comma = self.emit.new_label();
        let got_end = self.emit.new_label();
        let done = self.emit.new_label();
        let error_code = ErrorCode::UnexpectedCharacter as u32;

        self.emit_inline_skip_ws();

        self.emit.emit_word(
            aarch64::encode_cmp_reg(aarch64::Width::X64, Reg::X19, Reg::X20).expect("cmp"),
        );
        self.emit
            .emit_b_cond_label(aarch64::Condition::Hs, error_exit)
            .expect("b.hs");
        self.emit
            .emit_word(aarch64::encode_ldrb_imm(Reg::X9, Reg::X19, 0).expect("ldrb"));
        self.emit.emit_word(
            aarch64::encode_add_imm(aarch64::Width::X64, Reg::X19, Reg::X19, 1, false)
                .expect("add"),
        );
        self.emit.emit_word(
            aarch64::encode_cmp_imm(aarch64::Width::W32, Reg::X9, b',' as u16, false).expect("cmp"),
        );
        self.emit
            .emit_b_cond_label(aarch64::Condition::Eq, got_comma)
            .expect("b.eq");
        self.emit.emit_word(
            aarch64::encode_cmp_imm(aarch64::Width::W32, Reg::X9, b'}' as u16, false).expect("cmp"),
        );
        self.emit
            .emit_b_cond_label(aarch64::Condition::Eq, got_end)
            .expect("b.eq");
        self.emit.emit_word(
            aarch64::encode_movz(aarch64::Width::W32, Reg::X9, error_code as u16, 0).expect("movz"),
        );
        self.emit.emit_word(
            aarch64::encode_str_imm(aarch64::Width::W32, Reg::X9, Reg::X22, CTX_ERROR_CODE)
                .expect("str"),
        );
        self.emit.emit_b_label(error_exit).expect("b");
        self.emit.bind_label(got_comma).expect("bind");
        self.emit
            .emit_word(aarch64::encode_strb_imm(Reg::XZR, Reg::SP, sp_offset).expect("strb"));
        self.emit.emit_b_label(done).expect("b");
        self.emit.bind_label(got_end).expect("bind");
        self.emit
            .emit_word(aarch64::encode_movz(aarch64::Width::W32, Reg::X9, 1, 0).expect("movz"));
        self.emit
            .emit_word(aarch64::encode_strb_imm(Reg::X9, Reg::SP, sp_offset).expect("strb"));
        self.emit.bind_label(done).expect("bind");
    }

    /// Store the cached cursor (x19) to a stack slot.
    pub fn emit_save_cursor_to_stack(&mut self, sp_offset: u32) {
        self.emit.emit_word(
            aarch64::encode_str_imm(aarch64::Width::X64, Reg::X19, Reg::SP, sp_offset)
                .expect("str"),
        );
    }

    /// Emit: skip whitespace, expect and consume `"`, branch to error_exit if not found.
    /// After this, x19 points just after the opening `"`.
    pub fn emit_json_expect_quote_after_ws(&mut self, _ws_intrinsic: *const u8) {
        let error_exit = self.error_exit;
        let not_quote = self.emit.new_label();
        let ok = self.emit.new_label();
        let error_code = ErrorCode::ExpectedStringKey as u32;

        self.emit_inline_skip_ws();

        // Check bounds + opening '"'
        self.emit.emit_word(
            aarch64::encode_cmp_reg(aarch64::Width::X64, Reg::X19, Reg::X20).expect("cmp"),
        );
        self.emit
            .emit_b_cond_label(aarch64::Condition::Hs, not_quote)
            .expect("b.hs");
        self.emit
            .emit_word(aarch64::encode_ldrb_imm(Reg::X9, Reg::X19, 0).expect("ldrb"));
        self.emit.emit_word(
            aarch64::encode_cmp_imm(aarch64::Width::W32, Reg::X9, 0x22, false).expect("cmp"),
        );
        self.emit
            .emit_b_cond_label(aarch64::Condition::Ne, not_quote)
            .expect("b.ne");
        self.emit.emit_word(
            aarch64::encode_add_imm(aarch64::Width::X64, Reg::X19, Reg::X19, 1, false)
                .expect("add"),
        );
        self.emit.emit_b_label(ok).expect("b");

        self.emit.bind_label(not_quote).expect("bind");
        self.emit.emit_word(
            aarch64::encode_movz(aarch64::Width::W32, Reg::X9, error_code as u16, 0).expect("movz"),
        );
        self.emit.emit_word(
            aarch64::encode_str_imm(aarch64::Width::W32, Reg::X9, Reg::X22, CTX_ERROR_CODE)
                .expect("str"),
        );
        self.emit.emit_b_label(error_exit).expect("b");
        self.emit.bind_label(ok).expect("bind");
    }

    /// Call a post-scan intrinsic: fn(ctx, out+field_offset, start, len).
    /// start is in `start_sp_offset`. len = x19 - start. Advances x19 past closing `"`.
    pub fn emit_call_string_finish(
        &mut self,
        fn_ptr: *const u8,
        field_offset: u32,
        start_sp_offset: u32,
    ) {
        // x9 = start, x10 = len, flush cursor advanced past closing '"'
        self.emit.emit_word(
            aarch64::encode_ldr_imm(aarch64::Width::X64, Reg::X9, Reg::SP, start_sp_offset)
                .expect("ldr"),
        );
        self.emit.emit_word(
            aarch64::encode_sub_reg(aarch64::Width::X64, Reg::X10, Reg::X19, Reg::X9).expect("sub"),
        );
        self.emit.emit_word(
            aarch64::encode_add_imm(aarch64::Width::X64, Reg::X11, Reg::X19, 1, false)
                .expect("add"),
        );
        self.emit.emit_word(
            aarch64::encode_str_imm(
                aarch64::Width::X64,
                Reg::X11,
                Reg::X22,
                CTX_INPUT_PTR as u32,
            )
            .expect("str"),
        );
        self.emit.emit_word(
            aarch64::encode_mov_reg(aarch64::Width::X64, Reg::X0, Reg::X22).expect("mov"),
        );
        self.emit.emit_word(
            aarch64::encode_add_imm(
                aarch64::Width::X64,
                Reg::X1,
                Reg::X21,
                field_offset as u16,
                false,
            )
            .expect("add"),
        );
        self.emit.emit_word(
            aarch64::encode_mov_reg(aarch64::Width::X64, Reg::X2, Reg::X9).expect("mov"),
        );
        self.emit.emit_word(
            aarch64::encode_mov_reg(aarch64::Width::X64, Reg::X3, Reg::X10).expect("mov"),
        );
        self.emit_call_fn_ptr(fn_ptr);
        self.emit_reload_cursor_and_check_error();
    }

    /// Call a post-scan escape intrinsic: fn(ctx, out+field_offset, start, prefix_len).
    /// start is in `start_sp_offset`. prefix_len = x19 - start. x19 is at `\`.
    pub fn emit_call_string_escape(
        &mut self,
        fn_ptr: *const u8,
        field_offset: u32,
        start_sp_offset: u32,
    ) {
        self.emit_flush_input_cursor();
        self.emit.emit_word(
            aarch64::encode_ldr_imm(aarch64::Width::X64, Reg::X9, Reg::SP, start_sp_offset)
                .expect("ldr"),
        );
        self.emit.emit_word(
            aarch64::encode_sub_reg(aarch64::Width::X64, Reg::X10, Reg::X19, Reg::X9).expect("sub"),
        );
        self.emit.emit_word(
            aarch64::encode_mov_reg(aarch64::Width::X64, Reg::X0, Reg::X22).expect("mov"),
        );
        self.emit.emit_word(
            aarch64::encode_add_imm(
                aarch64::Width::X64,
                Reg::X1,
                Reg::X21,
                field_offset as u16,
                false,
            )
            .expect("add"),
        );
        self.emit.emit_word(
            aarch64::encode_mov_reg(aarch64::Width::X64, Reg::X2, Reg::X9).expect("mov"),
        );
        self.emit.emit_word(
            aarch64::encode_mov_reg(aarch64::Width::X64, Reg::X3, Reg::X10).expect("mov"),
        );
        self.emit_call_fn_ptr(fn_ptr);
        self.emit_reload_cursor_and_check_error();
    }

    /// Call kajit_string_validate_alloc_copy(ctx, start, len) for String malum path.
    /// start is in `start_sp_offset`. len = x19 - start.
    /// Returns buf pointer in x0. Saves len to `len_save_sp_offset`.
    pub fn emit_call_validate_alloc_copy_from_scan(
        &mut self,
        fn_ptr: *const u8,
        start_sp_offset: u32,
        len_save_sp_offset: u32,
    ) {
        self.emit_flush_input_cursor();
        self.emit.emit_word(
            aarch64::encode_ldr_imm(aarch64::Width::X64, Reg::X9, Reg::SP, start_sp_offset)
                .expect("ldr"),
        );
        self.emit.emit_word(
            aarch64::encode_sub_reg(aarch64::Width::X64, Reg::X10, Reg::X19, Reg::X9).expect("sub"),
        );
        self.emit.emit_word(
            aarch64::encode_str_imm(aarch64::Width::X64, Reg::X10, Reg::SP, len_save_sp_offset)
                .expect("str"),
        );
        self.emit.emit_word(
            aarch64::encode_mov_reg(aarch64::Width::X64, Reg::X0, Reg::X22).expect("mov"),
        );
        self.emit.emit_word(
            aarch64::encode_mov_reg(aarch64::Width::X64, Reg::X1, Reg::X9).expect("mov"),
        );
        self.emit.emit_word(
            aarch64::encode_mov_reg(aarch64::Width::W32, Reg::X2, Reg::X10).expect("mov"),
        );
        self.emit_call_fn_ptr(fn_ptr);
        // x0 = buf pointer (or null on error) — no cursor reload needed
        self.emit_check_error();
    }

    /// Write String fields (ptr, len, cap) using malum offsets after validate_alloc_copy.
    /// x0 = buf pointer. Reads len from `len_sp_offset`. Advances cursor past `"`.
    pub fn emit_write_malum_string_and_advance(
        &mut self,
        field_offset: u32,
        string_offsets: &crate::malum::StringOffsets,
        len_sp_offset: u32,
    ) {
        let ptr_off = field_offset + string_offsets.ptr_offset;
        let len_off = field_offset + string_offsets.len_offset;
        let cap_off = field_offset + string_offsets.cap_offset;

        self.emit.emit_word(
            aarch64::encode_str_imm(aarch64::Width::X64, Reg::X0, Reg::X21, ptr_off).expect("str"),
        );
        self.emit.emit_word(
            aarch64::encode_ldr_imm(aarch64::Width::X64, Reg::X9, Reg::SP, len_sp_offset)
                .expect("ldr"),
        );
        self.emit.emit_word(
            aarch64::encode_str_imm(aarch64::Width::X64, Reg::X9, Reg::X21, len_off).expect("str"),
        );
        self.emit.emit_word(
            aarch64::encode_str_imm(aarch64::Width::X64, Reg::X9, Reg::X21, cap_off).expect("str"),
        );
        self.emit.emit_word(
            aarch64::encode_add_imm(aarch64::Width::X64, Reg::X19, Reg::X19, 1, false)
                .expect("add"),
        );
    }

    /// Emit inline key-reading slow path call: fn(ctx, start, prefix_len, &key_ptr, &key_len).
    pub fn emit_call_key_slow_from_jit(
        &mut self,
        fn_ptr: *const u8,
        start_sp_offset: u32,
        key_ptr_sp_offset: u32,
        key_len_sp_offset: u32,
    ) {
        self.emit_flush_input_cursor();
        self.emit.emit_word(
            aarch64::encode_ldr_imm(aarch64::Width::X64, Reg::X9, Reg::SP, start_sp_offset)
                .expect("ldr"),
        );
        self.emit.emit_word(
            aarch64::encode_sub_reg(aarch64::Width::X64, Reg::X10, Reg::X19, Reg::X9).expect("sub"),
        );
        self.emit.emit_word(
            aarch64::encode_mov_reg(aarch64::Width::X64, Reg::X0, Reg::X22).expect("mov"),
        );
        self.emit.emit_word(
            aarch64::encode_mov_reg(aarch64::Width::X64, Reg::X1, Reg::X9).expect("mov"),
        );
        self.emit.emit_word(
            aarch64::encode_mov_reg(aarch64::Width::X64, Reg::X2, Reg::X10).expect("mov"),
        );
        self.emit.emit_word(
            aarch64::encode_add_imm(
                aarch64::Width::X64,
                Reg::X3,
                Reg::SP,
                key_ptr_sp_offset as u16,
                false,
            )
            .expect("add"),
        );
        self.emit.emit_word(
            aarch64::encode_add_imm(
                aarch64::Width::X64,
                Reg::X4,
                Reg::SP,
                key_len_sp_offset as u16,
                false,
            )
            .expect("add"),
        );
        self.emit_call_fn_ptr(fn_ptr);
        self.emit_reload_cursor_and_check_error();
    }

    // ── Enum support ──────────────────────────────────────────────────

    // r[impl deser.enum.set-variant]

    /// Write a discriminant value to [out + 0].
    /// `size` is 1, 2, 4, or 8 bytes (from EnumRepr).
    pub fn emit_write_discriminant(&mut self, value: i64, size: u32) {
        let val = value as u64;
        // Load immediate into w9/x9
        if size <= 4 {
            let val32 = val as u32;
            if val32 <= 0xFFFF {
                self.emit.emit_word(
                    aarch64::encode_movz(aarch64::Width::W32, Reg::X9, val32 as u16, 0)
                        .expect("movz"),
                );
            } else {
                let lo = val32 & 0xFFFF;
                let hi = (val32 >> 16) & 0xFFFF;
                self.emit.emit_word(
                    aarch64::encode_movz(aarch64::Width::W32, Reg::X9, lo as u16, 0).expect("movz"),
                );
                self.emit.emit_word(
                    aarch64::encode_movk(aarch64::Width::W32, Reg::X9, hi as u16, 16)
                        .expect("movk"),
                );
            }
        } else {
            self.emit.emit_word(
                aarch64::encode_movz(aarch64::Width::X64, Reg::X9, (val & 0xFFFF) as u16, 0)
                    .expect("movz"),
            );
            self.emit.emit_word(
                aarch64::encode_movk(
                    aarch64::Width::X64,
                    Reg::X9,
                    ((val >> 16) & 0xFFFF) as u16,
                    16,
                )
                .expect("movk"),
            );
            self.emit.emit_word(
                aarch64::encode_movk(
                    aarch64::Width::X64,
                    Reg::X9,
                    ((val >> 32) & 0xFFFF) as u16,
                    32,
                )
                .expect("movk"),
            );
            self.emit.emit_word(
                aarch64::encode_movk(
                    aarch64::Width::X64,
                    Reg::X9,
                    ((val >> 48) & 0xFFFF) as u16,
                    48,
                )
                .expect("movk"),
            );
        }
        // Store to [out + 0]
        match size {
            1 => self
                .emit
                .emit_word(aarch64::encode_strb_imm(Reg::X9, Reg::X21, 0).expect("strb")),
            2 => self
                .emit
                .emit_word(aarch64::encode_strh_imm(Reg::X9, Reg::X21, 0).expect("strh")),
            4 => self.emit.emit_word(
                aarch64::encode_str_imm(aarch64::Width::W32, Reg::X9, Reg::X21, 0).expect("str"),
            ),
            8 => self.emit.emit_word(
                aarch64::encode_str_imm(aarch64::Width::X64, Reg::X9, Reg::X21, 0).expect("str"),
            ),
            _ => panic!("unsupported discriminant size: {size}"),
        }
    }

    // r[impl deser.postcard.enum.dispatch]

    /// Read a postcard varint discriminant into w9 (kept in register for
    /// dispatch, not stored to memory).
    ///
    /// On the fast path (single-byte, value < 128): 4 instructions.
    /// On the slow path: calls the intrinsic, which writes to a temporary
    /// on the stack, then loads the result into w9.
    ///
    /// After this, the caller emits `emit_cmp_imm_branch_eq` for each variant.
    pub fn emit_read_postcard_discriminant(&mut self, slow_intrinsic: *const u8) {
        let error_exit = self.error_exit;
        let eof_label = self.emit.new_label();
        let slow_path = self.emit.new_label();
        let done_label = self.emit.new_label();

        self.emit.emit_word(
            aarch64::encode_cmp_reg(aarch64::Width::X64, Reg::X19, Reg::X20).expect("cmp"),
        );
        self.emit
            .emit_b_cond_label(aarch64::Condition::Hs, eof_label)
            .expect("b.hs");
        self.emit
            .emit_word(aarch64::encode_ldrb_imm(Reg::X9, Reg::X19, 0).expect("ldrb"));
        self.emit
            .emit_tbnz_label(Reg::X9, 7, slow_path)
            .expect("tbnz");
        self.emit.emit_word(
            aarch64::encode_add_imm(aarch64::Width::X64, Reg::X19, Reg::X19, 1, false)
                .expect("add"),
        );
        self.emit.emit_b_label(done_label).expect("b");
        self.emit.bind_label(slow_path).expect("bind");

        // Slow path: call full varint decode intrinsic into temp on stack
        self.emit_flush_input_cursor();
        self.emit.emit_word(
            aarch64::encode_mov_reg(aarch64::Width::X64, Reg::X0, Reg::X22).expect("mov"),
        );
        self.emit.emit_word(
            aarch64::encode_add_imm(aarch64::Width::X64, Reg::X1, Reg::SP, 48, false).expect("add"),
        );
        self.emit_call_fn_ptr(slow_intrinsic);
        self.emit_reload_cursor_and_check_error();
        self.emit.emit_word(
            aarch64::encode_ldr_imm(aarch64::Width::W32, Reg::X9, Reg::SP, 48).expect("ldr"),
        );
        self.emit.emit_b_label(done_label).expect("b");
        self.emit.bind_label(eof_label).expect("bind");
        self.emit.emit_word(
            aarch64::encode_movz(
                aarch64::Width::W32,
                Reg::X9,
                crate::context::ErrorCode::UnexpectedEof as u16,
                0,
            )
            .expect("movz"),
        );
        self.emit.emit_word(
            aarch64::encode_str_imm(
                aarch64::Width::W32,
                Reg::X9,
                Reg::X22,
                CTX_ERROR_CODE as u32,
            )
            .expect("str"),
        );
        self.emit.emit_b_label(error_exit).expect("b");
        self.emit.bind_label(done_label).expect("bind");
    }

    /// Compare w9 (discriminant) with immediate `imm` and branch to `label`
    /// if equal.
    pub fn emit_cmp_imm_branch_eq(&mut self, imm: u32, label: LabelId) {
        self.emit.emit_word(
            aarch64::encode_cmp_imm(aarch64::Width::W32, Reg::X9, imm as u16, false).expect("cmp"),
        );
        self.emit
            .emit_b_cond_label(aarch64::Condition::Eq, label)
            .expect("b.eq");
    }

    /// Emit a branch-to-error for unknown variant (sets UnknownVariant error code).
    pub fn emit_unknown_variant_error(&mut self) {
        let error_exit = self.error_exit;
        let error_code = crate::context::ErrorCode::UnknownVariant as u32;

        self.emit.emit_word(
            aarch64::encode_movz(aarch64::Width::W32, Reg::X9, error_code as u16, 0).expect("movz"),
        );
        self.emit.emit_word(
            aarch64::encode_str_imm(
                aarch64::Width::W32,
                Reg::X9,
                Reg::X22,
                CTX_ERROR_CODE as u32,
            )
            .expect("str"),
        );
        self.emit.emit_b_label(error_exit).expect("b");
    }

    /// Save the cached input_ptr (x19) to a stack slot.
    pub fn emit_save_input_ptr(&mut self, stack_offset: u32) {
        self.emit.emit_word(
            aarch64::encode_str_imm(aarch64::Width::X64, Reg::X19, Reg::SP, stack_offset)
                .expect("str"),
        );
    }

    /// Restore the cached input_ptr (x19) from a stack slot.
    pub fn emit_restore_input_ptr(&mut self, stack_offset: u32) {
        self.emit.emit_word(
            aarch64::encode_ldr_imm(aarch64::Width::X64, Reg::X19, Reg::SP, stack_offset)
                .expect("ldr"),
        );
    }

    /// Store a 64-bit immediate into a stack slot at sp + offset.
    pub fn emit_store_imm64_to_stack(&mut self, stack_offset: u32, value: u64) {
        if value == 0 {
            self.emit.emit_word(
                aarch64::encode_str_imm(aarch64::Width::X64, Reg::XZR, Reg::SP, stack_offset)
                    .expect("str"),
            );
        } else {
            // Load immediate into x9, then store
            self.emit.emit_word(
                aarch64::encode_movz(aarch64::Width::X64, Reg::X9, (value & 0xFFFF) as u16, 0)
                    .expect("movz"),
            );
            if value > 0xFFFF {
                self.emit.emit_word(
                    aarch64::encode_movk(
                        aarch64::Width::X64,
                        Reg::X9,
                        ((value >> 16) & 0xFFFF) as u16,
                        16,
                    )
                    .expect("movk"),
                );
            }
            if value > 0xFFFF_FFFF {
                self.emit.emit_word(
                    aarch64::encode_movk(
                        aarch64::Width::X64,
                        Reg::X9,
                        ((value >> 32) & 0xFFFF) as u16,
                        32,
                    )
                    .expect("movk"),
                );
            }
            if value > 0xFFFF_FFFF_FFFF {
                self.emit.emit_word(
                    aarch64::encode_movk(
                        aarch64::Width::X64,
                        Reg::X9,
                        ((value >> 48) & 0xFFFF) as u16,
                        48,
                    )
                    .expect("movk"),
                );
            }
            self.emit.emit_word(
                aarch64::encode_str_imm(aarch64::Width::X64, Reg::X9, Reg::SP, stack_offset)
                    .expect("str"),
            );
        }
    }

    /// AND a 64-bit immediate into a stack slot at sp + offset.
    /// Loads the slot, ANDs with the immediate, stores back.
    pub fn emit_and_imm64_on_stack(&mut self, stack_offset: u32, mask: u64) {
        self.emit.emit_word(
            aarch64::encode_ldr_imm(aarch64::Width::X64, Reg::X9, Reg::SP, stack_offset)
                .expect("ldr"),
        );
        // Load mask into x10
        self.emit.emit_word(
            aarch64::encode_movz(aarch64::Width::X64, Reg::X10, (mask & 0xFFFF) as u16, 0)
                .expect("movz"),
        );
        if mask > 0xFFFF {
            self.emit.emit_word(
                aarch64::encode_movk(
                    aarch64::Width::X64,
                    Reg::X10,
                    ((mask >> 16) & 0xFFFF) as u16,
                    16,
                )
                .expect("movk"),
            );
        }
        if mask > 0xFFFF_FFFF {
            self.emit.emit_word(
                aarch64::encode_movk(
                    aarch64::Width::X64,
                    Reg::X10,
                    ((mask >> 32) & 0xFFFF) as u16,
                    32,
                )
                .expect("movk"),
            );
        }
        if mask > 0xFFFF_FFFF_FFFF {
            self.emit.emit_word(
                aarch64::encode_movk(
                    aarch64::Width::X64,
                    Reg::X10,
                    ((mask >> 48) & 0xFFFF) as u16,
                    48,
                )
                .expect("movk"),
            );
        }
        self.emit.emit_word(
            aarch64::encode_and_reg(
                aarch64::Width::X64,
                Reg::X9,
                Reg::X9,
                Reg::X10,
                aarch64::Shift::Lsl,
                0,
            )
            .expect("and"),
        );
        self.emit.emit_word(
            aarch64::encode_str_imm(aarch64::Width::X64, Reg::X9, Reg::SP, stack_offset)
                .expect("str"),
        );
    }

    /// Check if the stack slot at sp + offset has exactly one bit set (popcount == 1).
    /// If so, branch to `label`.
    pub fn emit_popcount_eq1_branch(&mut self, stack_offset: u32, label: LabelId) {
        self.emit.emit_word(
            aarch64::encode_ldr_imm(aarch64::Width::X64, Reg::X9, Reg::SP, stack_offset)
                .expect("ldr"),
        );
        // popcount == 1 iff (x & (x-1)) == 0 && x != 0
        self.emit.emit_word(
            aarch64::encode_sub_imm(aarch64::Width::X64, Reg::X10, Reg::X9, 1, false).expect("sub"),
        );
        self.emit.emit_word(
            aarch64::encode_tst_reg(aarch64::Width::X64, Reg::X9, Reg::X10).expect("tst"),
        );
        self.emit
            .emit_word(aarch64::encode_b_cond(aarch64::Condition::Ne, 3).expect("b.ne"));
        self.emit
            .emit_cbnz_label(aarch64::Width::X64, Reg::X9, label)
            .expect("cbnz");
    }

    /// Check if the stack slot at sp + offset is zero. If so, branch to `label`.
    pub fn emit_stack_zero_branch(&mut self, stack_offset: u32, label: LabelId) {
        self.emit.emit_word(
            aarch64::encode_ldr_imm(aarch64::Width::X64, Reg::X9, Reg::SP, stack_offset)
                .expect("ldr"),
        );
        self.emit
            .emit_cbz_label(aarch64::Width::X64, Reg::X9, label)
            .expect("cbz");
    }

    /// Load the stack slot at sp + offset into x9, then branch to `label` if
    /// bit `bit_index` is set.
    pub fn emit_test_bit_branch(&mut self, stack_offset: u32, bit_index: u32, label: LabelId) {
        self.emit.emit_word(
            aarch64::encode_ldr_imm(aarch64::Width::X64, Reg::X9, Reg::SP, stack_offset)
                .expect("ldr"),
        );
        self.emit
            .emit_tbnz_label(Reg::X9, bit_index as u8, label)
            .expect("tbnz");
    }

    /// Test a single bit at `bit_index` in the u64 at `[sp + stack_offset]`.
    /// Branch to `label` if the bit is CLEAR (zero) — i.e., the field was NOT seen.
    pub fn emit_test_bit_branch_zero(&mut self, stack_offset: u32, bit_index: u32, label: LabelId) {
        self.emit.emit_word(
            aarch64::encode_ldr_imm(aarch64::Width::X64, Reg::X9, Reg::SP, stack_offset)
                .expect("ldr"),
        );
        self.emit
            .emit_tbz_label(Reg::X9, bit_index as u8, label)
            .expect("tbz");
    }

    /// Emit an error (write error code to ctx, branch to error_exit).
    pub fn emit_error(&mut self, code: crate::context::ErrorCode) {
        let error_exit = self.error_exit;
        let error_code = code as u32;
        self.emit.emit_word(
            aarch64::encode_movz(aarch64::Width::W32, Reg::X9, error_code as u16, 0).expect("movz"),
        );
        self.emit.emit_word(
            aarch64::encode_str_imm(
                aarch64::Width::W32,
                Reg::X9,
                Reg::X22,
                CTX_ERROR_CODE as u32,
            )
            .expect("str"),
        );
        self.emit.emit_b_label(error_exit).expect("b");
    }

    /// Advance the cached cursor by n bytes (inline, no function call).
    pub fn emit_advance_cursor_by(&mut self, n: u32) {
        self.emit.emit_word(
            aarch64::encode_add_imm(aarch64::Width::X64, Reg::X19, Reg::X19, n as u16, false)
                .expect("add"),
        );
    }

    // r[impl deser.json.struct]
    // r[impl deser.json.map]
    /// Inline: skip JSON whitespace (space/tab/LF/CR) then consume the expected byte.
    /// Sets `error_code` and branches to the current error_exit if EOF or wrong byte.
    ///
    /// Register contract: x19 = cached input_ptr, x20 = cached input_end, x22 = ctx.
    /// Scratch: w9 (adjusted byte), w10 (raw byte), x8 (WS bitmask — clobbered).
    ///
    /// WS check: map byte→bit via `b - 9`. SP(0x20)→23, HT(0x09)→0, LF(0x0a)→1,
    /// CR(0x0d)→4. Bitmask 0x0080_0013 has those bits set. If bit `(b-9)` is 1 → WS.
    pub fn emit_expect_byte_after_ws(&mut self, expected: u8, error_code: ErrorCode) {
        let error_exit = self.error_exit;
        let err_code = error_code as u32;
        let expected = expected as u32;

        let ws_loop = self.emit.new_label();
        let non_ws = self.emit.new_label();
        let done = self.emit.new_label();
        let err_lbl = self.emit.new_label();

        self.emit.bind_label(ws_loop).expect("bind");
        self.emit.emit_word(
            aarch64::encode_cmp_reg(aarch64::Width::X64, Reg::X19, Reg::X20).expect("cmp"),
        );
        self.emit
            .emit_b_cond_label(aarch64::Condition::Hs, err_lbl)
            .expect("b.hs");
        self.emit
            .emit_word(aarch64::encode_ldrb_imm(Reg::X10, Reg::X19, 0).expect("ldrb"));
        self.emit.emit_word(
            aarch64::encode_sub_imm(aarch64::Width::W32, Reg::X9, Reg::X10, 9, false).expect("sub"),
        );
        self.emit.emit_word(
            aarch64::encode_cmp_imm(aarch64::Width::W32, Reg::X9, 23, false).expect("cmp"),
        );
        self.emit
            .emit_b_cond_label(aarch64::Condition::Hi, non_ws)
            .expect("b.hi");
        self.emit.emit_word(
            aarch64::encode_movz(aarch64::Width::X64, Reg::X8, 0x0013, 0).expect("movz"),
        );
        self.emit.emit_word(
            aarch64::encode_movk(aarch64::Width::X64, Reg::X8, 0x0080, 16).expect("movk"),
        );
        self.emit.emit_word(
            aarch64::encode_lsr_reg(aarch64::Width::X64, Reg::X8, Reg::X8, Reg::X9).expect("lsr"),
        );
        self.emit.emit_tbz_label(Reg::X8, 0, non_ws).expect("tbz");
        self.emit.emit_word(
            aarch64::encode_add_imm(aarch64::Width::X64, Reg::X19, Reg::X19, 1, false)
                .expect("add"),
        );
        self.emit.emit_b_label(ws_loop).expect("b");

        self.emit.bind_label(non_ws).expect("bind");
        self.emit.emit_word(
            aarch64::encode_cmp_imm(aarch64::Width::W32, Reg::X10, expected as u16, false)
                .expect("cmp"),
        );
        self.emit
            .emit_b_cond_label(aarch64::Condition::Ne, err_lbl)
            .expect("b.ne");
        self.emit.emit_word(
            aarch64::encode_add_imm(aarch64::Width::X64, Reg::X19, Reg::X19, 1, false)
                .expect("add"),
        );
        self.emit.emit_b_label(done).expect("b");

        self.emit.bind_label(err_lbl).expect("bind");
        self.emit.emit_word(
            aarch64::encode_movz(aarch64::Width::W32, Reg::X9, err_code as u16, 0).expect("movz"),
        );
        self.emit.emit_word(
            aarch64::encode_str_imm(
                aarch64::Width::W32,
                Reg::X9,
                Reg::X22,
                CTX_ERROR_CODE as u32,
            )
            .expect("str"),
        );
        self.emit.emit_b_label(error_exit).expect("b");

        self.emit.bind_label(done).expect("bind");
    }

    // ── Option support ────────────────────────────────────────────────

    /// Save the current `out` pointer (x21) and redirect it to a stack scratch area.
    /// The save slot is at `scratch_offset - 8`.
    pub fn emit_redirect_out_to_stack(&mut self, scratch_offset: u32) {
        let save_slot = scratch_offset - 8;
        self.emit.emit_word(
            aarch64::encode_str_imm(aarch64::Width::X64, Reg::X21, Reg::SP, save_slot)
                .expect("str"),
        );
        self.emit.emit_word(
            aarch64::encode_add_imm(
                aarch64::Width::X64,
                Reg::X21,
                Reg::SP,
                scratch_offset as u16,
                false,
            )
            .expect("add"),
        );
    }

    /// Restore the `out` pointer (x21) from the saved slot.
    pub fn emit_restore_out(&mut self, scratch_offset: u32) {
        let save_slot = scratch_offset - 8;
        self.emit.emit_word(
            aarch64::encode_ldr_imm(aarch64::Width::X64, Reg::X21, Reg::SP, save_slot)
                .expect("ldr"),
        );
    }

    /// Call kajit_option_init_none(init_none_fn, out + offset).
    /// Does not touch ctx or the cursor.
    pub fn emit_call_option_init_none(
        &mut self,
        wrapper_fn: *const u8,
        init_none_fn: *const u8,
        offset: u32,
    ) {
        let val = init_none_fn as u64;
        self.emit.emit_word(
            aarch64::encode_movz(aarch64::Width::X64, Reg::X0, (val & 0xFFFF) as u16, 0)
                .expect("movz"),
        );
        self.emit.emit_word(
            aarch64::encode_movk(
                aarch64::Width::X64,
                Reg::X0,
                ((val >> 16) & 0xFFFF) as u16,
                16,
            )
            .expect("movk"),
        );
        self.emit.emit_word(
            aarch64::encode_movk(
                aarch64::Width::X64,
                Reg::X0,
                ((val >> 32) & 0xFFFF) as u16,
                32,
            )
            .expect("movk"),
        );
        self.emit.emit_word(
            aarch64::encode_movk(
                aarch64::Width::X64,
                Reg::X0,
                ((val >> 48) & 0xFFFF) as u16,
                48,
            )
            .expect("movk"),
        );
        self.emit.emit_word(
            aarch64::encode_add_imm(aarch64::Width::X64, Reg::X1, Reg::X21, offset as u16, false)
                .expect("add"),
        );
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
        let fn_val = fn_ptr as u64;
        self.emit.emit_word(
            aarch64::encode_movz(aarch64::Width::X64, Reg::X0, (fn_val & 0xFFFF) as u16, 0)
                .expect("movz"),
        );
        self.emit.emit_word(
            aarch64::encode_movk(
                aarch64::Width::X64,
                Reg::X0,
                ((fn_val >> 16) & 0xFFFF) as u16,
                16,
            )
            .expect("movk"),
        );
        self.emit.emit_word(
            aarch64::encode_movk(
                aarch64::Width::X64,
                Reg::X0,
                ((fn_val >> 32) & 0xFFFF) as u16,
                32,
            )
            .expect("movk"),
        );
        self.emit.emit_word(
            aarch64::encode_movk(
                aarch64::Width::X64,
                Reg::X0,
                ((fn_val >> 48) & 0xFFFF) as u16,
                48,
            )
            .expect("movk"),
        );
        self.emit.emit_word(
            aarch64::encode_add_imm(aarch64::Width::X64, Reg::X1, Reg::X21, offset as u16, false)
                .expect("add"),
        );
        let extra_val = extra_ptr as u64;
        self.emit.emit_word(
            aarch64::encode_movz(aarch64::Width::X64, Reg::X2, (extra_val & 0xFFFF) as u16, 0)
                .expect("movz"),
        );
        self.emit.emit_word(
            aarch64::encode_movk(
                aarch64::Width::X64,
                Reg::X2,
                ((extra_val >> 16) & 0xFFFF) as u16,
                16,
            )
            .expect("movk"),
        );
        self.emit.emit_word(
            aarch64::encode_movk(
                aarch64::Width::X64,
                Reg::X2,
                ((extra_val >> 32) & 0xFFFF) as u16,
                32,
            )
            .expect("movk"),
        );
        self.emit.emit_word(
            aarch64::encode_movk(
                aarch64::Width::X64,
                Reg::X2,
                ((extra_val >> 48) & 0xFFFF) as u16,
                48,
            )
            .expect("movk"),
        );
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
        let val = init_some_fn as u64;
        self.emit.emit_word(
            aarch64::encode_movz(aarch64::Width::X64, Reg::X0, (val & 0xFFFF) as u16, 0)
                .expect("movz"),
        );
        self.emit.emit_word(
            aarch64::encode_movk(
                aarch64::Width::X64,
                Reg::X0,
                ((val >> 16) & 0xFFFF) as u16,
                16,
            )
            .expect("movk"),
        );
        self.emit.emit_word(
            aarch64::encode_movk(
                aarch64::Width::X64,
                Reg::X0,
                ((val >> 32) & 0xFFFF) as u16,
                32,
            )
            .expect("movk"),
        );
        self.emit.emit_word(
            aarch64::encode_movk(
                aarch64::Width::X64,
                Reg::X0,
                ((val >> 48) & 0xFFFF) as u16,
                48,
            )
            .expect("movk"),
        );
        self.emit.emit_word(
            aarch64::encode_add_imm(aarch64::Width::X64, Reg::X1, Reg::X21, offset as u16, false)
                .expect("add"),
        );
        self.emit.emit_word(
            aarch64::encode_add_imm(
                aarch64::Width::X64,
                Reg::X2,
                Reg::SP,
                scratch_offset as u16,
                false,
            )
            .expect("add"),
        );
        self.emit_call_fn_ptr(wrapper_fn);
    }

    // =====================================================================
    // Vec deserialization support
    // =====================================================================

    /// Call kajit_vec_alloc(ctx, count, elem_size, elem_align).
    ///
    /// Before call: count is in w9 (from emit_read_postcard_discriminant).
    /// After call: result (buf pointer) is in x0.
    /// Flushes cursor, reloads after, checks error.
    /// Call kajit_vec_alloc(ctx, count, elem_size, elem_align).
    ///
    /// count is in w9 (from emit_read_postcard_discriminant or JSON parse).
    /// Result (buf pointer) is in x0.
    ///
    /// **Important**: w9 is caller-saved and will be clobbered by the call.
    /// The count is saved to `count_slot` on the stack before the call so
    /// it survives across the function call boundary.
    pub fn emit_call_vec_alloc(&mut self, alloc_fn: *const u8, elem_size: u32, elem_align: u32) {
        self.emit_flush_input_cursor();
        self.emit.emit_word(
            aarch64::encode_mov_reg(aarch64::Width::X64, Reg::X0, Reg::X22).expect("mov"),
        );
        self.emit.emit_word(
            aarch64::encode_mov_reg(aarch64::Width::X64, Reg::X1, Reg::X9).expect("mov"),
        );
        self.emit.emit_word(
            aarch64::encode_movz(aarch64::Width::X64, Reg::X2, (elem_size as u16), 0)
                .expect("movz"),
        );
        self.emit.emit_word(
            aarch64::encode_movz(aarch64::Width::X64, Reg::X3, (elem_align as u16), 0)
                .expect("movz"),
        );
        self.emit_call_fn_ptr(alloc_fn);
        self.emit_reload_cursor_and_check_error();
    }

    /// Call kajit_vec_grow(ctx, old_buf, len, old_cap, new_cap, elem_size, elem_align).
    ///
    /// Reads old_buf, len, old_cap from stack slots. Computes new_cap = old_cap * 2.
    /// After call: new buf pointer is in x0.
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
        self.emit.emit_word(
            aarch64::encode_ldr_imm(aarch64::Width::X64, Reg::X10, Reg::SP, cap_slot).expect("ldr"),
        );
        self.emit.emit_word(
            aarch64::encode_add_reg(aarch64::Width::X64, Reg::X10, Reg::X10, Reg::X10)
                .expect("add"),
        );
        self.emit.emit_word(
            aarch64::encode_mov_reg(aarch64::Width::X64, Reg::X0, Reg::X22).expect("mov"),
        );
        self.emit.emit_word(
            aarch64::encode_ldr_imm(aarch64::Width::X64, Reg::X1, Reg::SP, buf_slot).expect("ldr"),
        );
        self.emit.emit_word(
            aarch64::encode_ldr_imm(aarch64::Width::X64, Reg::X2, Reg::SP, len_slot).expect("ldr"),
        );
        self.emit.emit_word(
            aarch64::encode_ldr_imm(aarch64::Width::X64, Reg::X3, Reg::SP, cap_slot).expect("ldr"),
        );
        self.emit.emit_word(
            aarch64::encode_mov_reg(aarch64::Width::X64, Reg::X4, Reg::X10).expect("mov"),
        );
        self.emit.emit_word(
            aarch64::encode_movz(aarch64::Width::X64, Reg::X5, elem_size as u16, 0).expect("movz"),
        );
        self.emit.emit_word(
            aarch64::encode_movz(aarch64::Width::X64, Reg::X6, elem_align as u16, 0).expect("movz"),
        );
        self.emit_call_fn_ptr(grow_fn);
        self.emit_reload_cursor_and_check_error();
        self.emit.emit_word(
            aarch64::encode_str_imm(aarch64::Width::X64, Reg::X0, Reg::SP, buf_slot).expect("str"),
        );
        self.emit.emit_word(
            aarch64::encode_ldr_imm(aarch64::Width::X64, Reg::X10, Reg::SP, cap_slot).expect("ldr"),
        );
        self.emit.emit_word(
            aarch64::encode_add_reg(aarch64::Width::X64, Reg::X10, Reg::X10, Reg::X10)
                .expect("add"),
        );
        self.emit.emit_word(
            aarch64::encode_str_imm(aarch64::Width::X64, Reg::X10, Reg::SP, cap_slot).expect("str"),
        );
    }

    /// Call kajit_vec_alloc with a constant count (for JSON initial allocation).
    ///
    /// Result (buf pointer) is in x0.
    pub fn emit_call_json_vec_initial_alloc(
        &mut self,
        alloc_fn: *const u8,
        count: u32,
        elem_size: u32,
        elem_align: u32,
    ) {
        self.emit_flush_input_cursor();
        self.emit.emit_word(
            aarch64::encode_mov_reg(aarch64::Width::X64, Reg::X0, Reg::X22).expect("mov"),
        );
        self.emit.emit_word(
            aarch64::encode_movz(aarch64::Width::X64, Reg::X1, count as u16, 0).expect("movz"),
        );
        self.emit.emit_word(
            aarch64::encode_movz(aarch64::Width::X64, Reg::X2, elem_size as u16, 0).expect("movz"),
        );
        self.emit.emit_word(
            aarch64::encode_movz(aarch64::Width::X64, Reg::X3, elem_align as u16, 0).expect("movz"),
        );
        self.emit_call_fn_ptr(alloc_fn);
        self.emit_reload_cursor_and_check_error();
    }

    /// Initialize JSON Vec loop state: buf from x0, len=0, cap=initial_cap.
    pub fn emit_json_vec_loop_init(
        &mut self,
        saved_out_slot: u32,
        buf_slot: u32,
        len_slot: u32,
        cap_slot: u32,
        initial_cap: u32,
    ) {
        self.emit.emit_word(
            aarch64::encode_str_imm(aarch64::Width::X64, Reg::X21, Reg::SP, saved_out_slot)
                .expect("str"),
        );
        self.emit.emit_word(
            aarch64::encode_str_imm(aarch64::Width::X64, Reg::X0, Reg::SP, buf_slot).expect("str"),
        );
        self.emit.emit_word(
            aarch64::encode_str_imm(aarch64::Width::X64, Reg::XZR, Reg::SP, len_slot).expect("str"),
        );
        self.emit.emit_word(
            aarch64::encode_movz(aarch64::Width::X64, Reg::X10, initial_cap as u16, 0)
                .expect("movz"),
        );
        self.emit.emit_word(
            aarch64::encode_str_imm(aarch64::Width::X64, Reg::X10, Reg::SP, cap_slot).expect("str"),
        );
    }

    /// Check ctx.error.code and branch to label if nonzero.
    pub fn emit_check_error_branch(&mut self, label: LabelId) {
        self.emit.emit_word(
            aarch64::encode_ldr_imm(
                aarch64::Width::W32,
                Reg::X10,
                Reg::X22,
                CTX_ERROR_CODE as u32,
            )
            .expect("ldr"),
        );
        self.emit
            .emit_cbnz_label(aarch64::Width::W32, Reg::X10, label)
            .expect("cbnz");
    }

    /// Save the count register (w9 on aarch64, r10d on x64) to a stack slot.
    ///
    /// Used to preserve the count across a function call (w9/r10 are caller-saved).
    pub fn emit_save_count_to_stack(&mut self, slot: u32) {
        self.emit.emit_word(
            aarch64::encode_str_imm(aarch64::Width::X64, Reg::X9, Reg::SP, slot).expect("str"),
        );
    }

    /// Initialize Vec loop state with cursor in x23, end in x24 (register-based).
    ///
    /// Saves x23/x24 to stack (callee-saved), then sets x23 = buf, x24 = end.
    /// Also saves out pointer and buf for final Vec store.
    /// Count must already be stored at count_slot (via emit_save_count_to_stack).
    pub fn emit_vec_loop_init_cursor(
        &mut self,
        saved_out_slot: u32,
        buf_slot: u32,
        count_slot: u32,
        save_x23_slot: u32,
        save_x24_slot: u32,
        elem_size: u32,
    ) {
        self.emit.emit_word(
            aarch64::encode_str_imm(aarch64::Width::X64, Reg::X21, Reg::SP, saved_out_slot)
                .expect("str"),
        );
        self.emit.emit_word(
            aarch64::encode_str_imm(aarch64::Width::X64, Reg::X0, Reg::SP, buf_slot).expect("str"),
        );
        self.emit.emit_word(
            aarch64::encode_str_imm(aarch64::Width::X64, Reg::X23, Reg::SP, save_x23_slot)
                .expect("str"),
        );
        self.emit.emit_word(
            aarch64::encode_str_imm(aarch64::Width::X64, Reg::X24, Reg::SP, save_x24_slot)
                .expect("str"),
        );
        self.emit.emit_word(
            aarch64::encode_mov_reg(aarch64::Width::X64, Reg::X23, Reg::X0).expect("mov"),
        );
        self.emit.emit_word(
            aarch64::encode_ldr_imm(aarch64::Width::X64, Reg::X10, Reg::SP, count_slot)
                .expect("ldr"),
        );
        self.emit.emit_word(
            aarch64::encode_movz(aarch64::Width::X64, Reg::X11, elem_size as u16, 0).expect("movz"),
        );
        self.emit.emit_word(
            aarch64::encode_madd(aarch64::Width::X64, Reg::X24, Reg::X10, Reg::X11, Reg::X0)
                .expect("madd"),
        );
    }

    /// Set out = cursor (x21 = x23). Single register move, no memory access.
    pub fn emit_vec_loop_load_cursor(&mut self, _cursor_slot: u32) {
        self.emit.emit_word(
            aarch64::encode_mov_reg(aarch64::Width::X64, Reg::X21, Reg::X23).expect("mov"),
        );
    }

    /// Advance cursor register, check error, branch back if cursor < end.
    /// All register ops — no memory access in the hot loop.
    pub fn emit_vec_loop_advance_cursor(
        &mut self,
        _cursor_slot: u32,
        _end_slot: u32,
        elem_size: u32,
        loop_label: LabelId,
        error_cleanup_label: LabelId,
    ) {
        self.emit.emit_word(
            aarch64::encode_ldr_imm(
                aarch64::Width::W32,
                Reg::X10,
                Reg::X22,
                CTX_ERROR_CODE as u32,
            )
            .expect("ldr"),
        );
        self.emit
            .emit_cbnz_label(aarch64::Width::W32, Reg::X10, error_cleanup_label)
            .expect("cbnz");
        self.emit.emit_word(
            aarch64::encode_add_imm(
                aarch64::Width::X64,
                Reg::X23,
                Reg::X23,
                elem_size as u16,
                false,
            )
            .expect("add"),
        );
        self.emit.emit_word(
            aarch64::encode_cmp_reg(aarch64::Width::X64, Reg::X23, Reg::X24).expect("cmp"),
        );
        self.emit
            .emit_b_cond_label(aarch64::Condition::Lo, loop_label)
            .expect("b.lo");
    }

    /// Advance the cursor register and loop back, without checking the error flag.
    ///
    /// Use this when all error paths within the loop body branch directly to
    /// the error cleanup label (e.g. via redirected `error_exit`), making
    /// the per-iteration error check redundant.
    pub fn emit_vec_loop_advance_no_error_check(
        &mut self,
        _end_slot: u32,
        elem_size: u32,
        loop_label: LabelId,
    ) {
        self.emit.emit_word(
            aarch64::encode_add_imm(
                aarch64::Width::X64,
                Reg::X23,
                Reg::X23,
                elem_size as u16,
                false,
            )
            .expect("add"),
        );
        self.emit.emit_word(
            aarch64::encode_cmp_reg(aarch64::Width::X64, Reg::X23, Reg::X24).expect("cmp"),
        );
        self.emit
            .emit_b_cond_label(aarch64::Condition::Lo, loop_label)
            .expect("b.lo");
    }

    /// Emit a tight varint Vec loop body: read a varint scalar, store to cursor,
    /// advance, loop. Writes directly to `[x23]` (the cursor register) and uses
    /// post-indexed addressing to combine load+advance. The slow path and EOF
    /// are placed out-of-line after the loop.
    ///
    /// `store_width`: 2/4/8 bytes to write at the cursor.
    /// `zigzag`: true for signed integers (zigzag decode before store).
    /// `intrinsic_fn_ptr`: slow-path multi-byte varint reader.
    /// `loop_label`: label at the top of the loop (already bound by caller).
    /// `done_label`: label to branch to after the loop completes.
    /// `error_cleanup`: label for error/cleanup path.
    #[allow(clippy::too_many_arguments)]
    pub fn emit_vec_varint_loop(
        &mut self,
        store_width: u32,
        zigzag: bool,
        intrinsic_fn_ptr: *const u8,
        elem_size: u32,
        _end_slot: u32,
        loop_label: LabelId,
        done_label: LabelId,
        error_cleanup: LabelId,
    ) {
        let slow_path = self.emit.new_label();
        let eof_label = self.emit.new_label();

        self.emit.emit_word(
            aarch64::encode_cmp_reg(aarch64::Width::X64, Reg::X19, Reg::X20).expect("cmp"),
        );
        self.emit
            .emit_b_cond_label(aarch64::Condition::Hs, eof_label)
            .expect("b.hs");
        self.emit
            .emit_word(aarch64::encode_ldrb_imm(Reg::X9, Reg::X19, 0).expect("ldrb"));
        self.emit.emit_word(
            aarch64::encode_add_imm(aarch64::Width::X64, Reg::X19, Reg::X19, 1, false)
                .expect("add"),
        );
        self.emit
            .emit_tbnz_label(Reg::X9, 7, slow_path)
            .expect("tbnz");

        if zigzag {
            self.emit.emit_word(
                aarch64::encode_lsr_imm(aarch64::Width::W32, Reg::X10, Reg::X9, 1).expect("lsr"),
            );
            self.emit.emit_word(
                aarch64::encode_and_imm(aarch64::Width::W32, Reg::X11, Reg::X9, 1).expect("and"),
            );
            self.emit.emit_word(
                aarch64::encode_neg(aarch64::Width::W32, Reg::X11, Reg::X11).expect("neg"),
            );
            self.emit.emit_word(
                aarch64::encode_eor_reg(
                    aarch64::Width::W32,
                    Reg::X9,
                    Reg::X10,
                    Reg::X11,
                    aarch64::Shift::Lsl,
                    0,
                )
                .expect("eor"),
            );
        }

        // Store directly to cursor (x23), no mov x21 needed
        match store_width {
            2 => self
                .emit
                .emit_word(aarch64::encode_strh_imm(Reg::X9, Reg::X23, 0).expect("strh")),
            4 => self.emit.emit_word(
                aarch64::encode_str_imm(aarch64::Width::W32, Reg::X9, Reg::X23, 0).expect("str"),
            ),
            8 => self.emit.emit_word(
                aarch64::encode_str_imm(aarch64::Width::X64, Reg::X9, Reg::X23, 0).expect("str"),
            ),
            _ => panic!("unsupported varint store width: {store_width}"),
        }

        // Advance cursor, loop back
        self.emit.emit_word(
            aarch64::encode_add_imm(
                aarch64::Width::X64,
                Reg::X23,
                Reg::X23,
                elem_size as u16,
                false,
            )
            .expect("add"),
        );
        self.emit.emit_word(
            aarch64::encode_cmp_reg(aarch64::Width::X64, Reg::X23, Reg::X24).expect("cmp"),
        );
        self.emit
            .emit_b_cond_label(aarch64::Condition::Lo, loop_label)
            .expect("b.lo");
        self.emit.emit_b_label(done_label).expect("b");

        // === Slow path (out-of-line) ===
        self.emit.bind_label(slow_path).expect("bind");
        self.emit.emit_word(
            aarch64::encode_sub_imm(aarch64::Width::X64, Reg::X19, Reg::X19, 1, false)
                .expect("sub"),
        );
        self.emit_flush_input_cursor();
        self.emit.emit_word(
            aarch64::encode_mov_reg(aarch64::Width::X64, Reg::X0, Reg::X22).expect("mov"),
        );
        self.emit.emit_word(
            aarch64::encode_mov_reg(aarch64::Width::X64, Reg::X1, Reg::X23).expect("mov"),
        );
        self.emit_call_fn_ptr(intrinsic_fn_ptr);
        self.emit.emit_word(
            aarch64::encode_ldr_imm(aarch64::Width::X64, Reg::X19, Reg::X22, CTX_INPUT_PTR)
                .expect("ldr"),
        );
        self.emit.emit_word(
            aarch64::encode_ldr_imm(aarch64::Width::W32, Reg::X9, Reg::X22, CTX_ERROR_CODE)
                .expect("ldr"),
        );
        self.emit
            .emit_cbnz_label(aarch64::Width::W32, Reg::X9, error_cleanup)
            .expect("cbnz");
        self.emit.emit_word(
            aarch64::encode_add_imm(
                aarch64::Width::X64,
                Reg::X23,
                Reg::X23,
                elem_size as u16,
                false,
            )
            .expect("add"),
        );
        self.emit.emit_word(
            aarch64::encode_cmp_reg(aarch64::Width::X64, Reg::X23, Reg::X24).expect("cmp"),
        );
        self.emit
            .emit_b_cond_label(aarch64::Condition::Lo, loop_label)
            .expect("b.lo");
        self.emit.emit_b_label(done_label).expect("b");

        // === EOF (cold) ===
        self.emit.bind_label(eof_label).expect("bind");
        self.emit.emit_word(
            aarch64::encode_movz(
                aarch64::Width::W32,
                Reg::X9,
                crate::context::ErrorCode::UnexpectedEof as u16,
                0,
            )
            .expect("movz"),
        );
        self.emit.emit_word(
            aarch64::encode_str_imm(aarch64::Width::W32, Reg::X9, Reg::X22, CTX_ERROR_CODE)
                .expect("str"),
        );
        self.emit.emit_b_label(error_cleanup).expect("b");
    }

    /// Restore x23/x24 from stack. Must be called on every exit path from a Vec loop.
    pub fn emit_vec_restore_callee_saved(&mut self, save_x23_slot: u32, save_x24_slot: u32) {
        self.emit.emit_word(
            aarch64::encode_ldr_imm(aarch64::Width::X64, Reg::X23, Reg::SP, save_x23_slot)
                .expect("ldr"),
        );
        self.emit.emit_word(
            aarch64::encode_ldr_imm(aarch64::Width::X64, Reg::X24, Reg::SP, save_x24_slot)
                .expect("ldr"),
        );
    }

    /// Emit Vec loop header: compute slot = buf + i * elem_size, set out = slot.
    ///
    /// Used by JSON where buf can change on growth and index-based access is needed.
    pub fn emit_vec_loop_slot(&mut self, buf_slot: u32, counter_slot: u32, elem_size: u32) {
        self.emit.emit_word(
            aarch64::encode_ldr_imm(aarch64::Width::X64, Reg::X21, Reg::SP, buf_slot).expect("ldr"),
        );
        self.emit.emit_word(
            aarch64::encode_ldr_imm(aarch64::Width::X64, Reg::X10, Reg::SP, counter_slot)
                .expect("ldr"),
        );
        self.emit.emit_word(
            aarch64::encode_movz(aarch64::Width::X64, Reg::X11, elem_size as u16, 0).expect("movz"),
        );
        self.emit.emit_word(
            aarch64::encode_madd(aarch64::Width::X64, Reg::X21, Reg::X10, Reg::X11, Reg::X21)
                .expect("madd"),
        );
    }

    /// Write Vec fields (ptr, len, cap) to out + base_offset using discovered offsets.
    /// Reads buf and count from stack slots.
    pub fn emit_vec_store(
        &mut self,
        base_offset: u32,
        saved_out_slot: u32,
        buf_slot: u32,
        len_slot: u32, // stack slot holding len (or count for postcard)
        cap_slot: u32, // stack slot holding cap (or count for postcard)
        offsets: &crate::malum::VecOffsets,
    ) {
        let ptr_off = base_offset + offsets.ptr_offset;
        let len_off = base_offset + offsets.len_offset;
        let cap_off = base_offset + offsets.cap_offset;

        self.emit.emit_word(
            aarch64::encode_ldr_imm(aarch64::Width::X64, Reg::X21, Reg::SP, saved_out_slot)
                .expect("ldr"),
        );
        self.emit.emit_word(
            aarch64::encode_ldr_imm(aarch64::Width::X64, Reg::X10, Reg::SP, buf_slot).expect("ldr"),
        );
        self.emit.emit_word(
            aarch64::encode_ldr_imm(aarch64::Width::X64, Reg::X11, Reg::SP, len_slot).expect("ldr"),
        );
        self.emit.emit_word(
            aarch64::encode_ldr_imm(aarch64::Width::X64, Reg::X12, Reg::SP, cap_slot).expect("ldr"),
        );
        self.emit.emit_word(
            aarch64::encode_str_imm(aarch64::Width::X64, Reg::X10, Reg::X21, ptr_off).expect("str"),
        );
        self.emit.emit_word(
            aarch64::encode_str_imm(aarch64::Width::X64, Reg::X11, Reg::X21, len_off).expect("str"),
        );
        self.emit.emit_word(
            aarch64::encode_str_imm(aarch64::Width::X64, Reg::X12, Reg::X21, cap_off).expect("str"),
        );
    }

    /// Write an empty Vec to out + base_offset with proper dangling pointer.
    pub fn emit_vec_store_empty_with_align(
        &mut self,
        base_offset: u32,
        elem_align: u32,
        offsets: &crate::malum::VecOffsets,
    ) {
        let ptr_off = base_offset + offsets.ptr_offset;
        let len_off = base_offset + offsets.len_offset;
        let cap_off = base_offset + offsets.cap_offset;

        // Vec::new() writes: ptr = NonNull::dangling() = elem_align as *mut T, len = 0, cap = 0.
        self.emit.emit_word(
            aarch64::encode_movz(aarch64::Width::X64, Reg::X10, elem_align as u16, 0)
                .expect("movz"),
        );
        self.emit.emit_word(
            aarch64::encode_str_imm(aarch64::Width::X64, Reg::X10, Reg::X21, ptr_off).expect("str"),
        );
        self.emit.emit_word(
            aarch64::encode_str_imm(aarch64::Width::X64, Reg::XZR, Reg::X21, len_off).expect("str"),
        );
        self.emit.emit_word(
            aarch64::encode_str_imm(aarch64::Width::X64, Reg::XZR, Reg::X21, cap_off).expect("str"),
        );
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
        self.emit.emit_word(
            aarch64::encode_ldr_imm(aarch64::Width::X64, Reg::X21, Reg::SP, saved_out_slot)
                .expect("ldr"),
        );
        self.emit.emit_word(
            aarch64::encode_ldr_imm(aarch64::Width::X64, Reg::X0, Reg::SP, buf_slot).expect("ldr"),
        );
        self.emit.emit_word(
            aarch64::encode_ldr_imm(aarch64::Width::X64, Reg::X1, Reg::SP, cap_slot).expect("ldr"),
        );
        self.emit.emit_word(
            aarch64::encode_movz(aarch64::Width::X64, Reg::X2, elem_size as u16, 0).expect("movz"),
        );
        self.emit.emit_word(
            aarch64::encode_movz(aarch64::Width::X64, Reg::X3, elem_align as u16, 0).expect("movz"),
        );
        self.emit_call_fn_ptr(free_fn);
        self.emit.emit_b_label(error_exit).expect("b");
    }

    /// Compare the count register (w9 on aarch64, r10d on x64) with zero
    /// and branch to label if equal.
    pub fn emit_cbz_count(&mut self, label: LabelId) {
        self.emit
            .emit_cbz_label(aarch64::Width::W32, Reg::X9, label)
            .expect("cbz");
    }

    /// Compare two stack slot values and branch if equal (len == cap for growth check).
    pub fn emit_cmp_stack_slots_branch_eq(&mut self, slot_a: u32, slot_b: u32, label: LabelId) {
        self.emit.emit_word(
            aarch64::encode_ldr_imm(aarch64::Width::X64, Reg::X10, Reg::SP, slot_a).expect("ldr"),
        );
        self.emit.emit_word(
            aarch64::encode_ldr_imm(aarch64::Width::X64, Reg::X11, Reg::SP, slot_b).expect("ldr"),
        );
        self.emit.emit_word(
            aarch64::encode_cmp_reg(aarch64::Width::X64, Reg::X10, Reg::X11).expect("cmp"),
        );
        self.emit
            .emit_b_cond_label(aarch64::Condition::Eq, label)
            .expect("b.eq");
    }

    /// Increment a stack slot value by 1.
    pub fn emit_inc_stack_slot(&mut self, slot: u32) {
        self.emit.emit_word(
            aarch64::encode_ldr_imm(aarch64::Width::X64, Reg::X10, Reg::SP, slot).expect("ldr"),
        );
        self.emit.emit_word(
            aarch64::encode_add_imm(aarch64::Width::X64, Reg::X10, Reg::X10, 1, false)
                .expect("add"),
        );
        self.emit.emit_word(
            aarch64::encode_str_imm(aarch64::Width::X64, Reg::X10, Reg::SP, slot).expect("str"),
        );
    }

    // ── Map support ─────────────────────────────────────────────────

    /// Advance the out register (x21) by a constant offset.
    ///
    /// Used in map loops to move from the key slot to the value slot within a pair.
    pub fn emit_advance_out_by(&mut self, offset: u32) {
        self.emit.emit_word(
            aarch64::encode_add_imm(
                aarch64::Width::X64,
                Reg::X21,
                Reg::X21,
                offset as u16,
                false,
            )
            .expect("add"),
        );
    }

    /// Call `kajit_map_build(from_pair_slice_fn, saved_out, pairs_buf, count)`.
    ///
    /// We cannot call `from_pair_slice` directly from JIT code because its
    /// first arg `PtrUninit` is a 16-byte struct (passed in 2 registers on
    /// aarch64).  `kajit_map_build` is a plain-C trampoline that takes four
    /// pointer-/usize-sized args and constructs `PtrUninit` internally.
    pub fn emit_call_map_from_pairs(
        &mut self,
        from_pair_slice_fn: *const u8,
        saved_out_slot: u32,
        buf_slot: u32,
        count_slot: u32,
    ) {
        let trampoline = crate::intrinsics::kajit_map_build as *const u8;
        // Load from_pair_slice_fn via x8, then mov to x0 (can't load directly
        // into x0 because emit_call_fn_ptr will clobber x8).
        self.emit_load_imm64(Reg::X8, from_pair_slice_fn as u64);
        self.emit.emit_word(
            aarch64::encode_mov_reg(aarch64::Width::X64, Reg::X0, Reg::X8).expect("mov"),
        );
        self.emit.emit_word(
            aarch64::encode_ldr_imm(aarch64::Width::X64, Reg::X1, Reg::SP, saved_out_slot)
                .expect("ldr"),
        );
        self.emit.emit_word(
            aarch64::encode_ldr_imm(aarch64::Width::X64, Reg::X2, Reg::SP, buf_slot).expect("ldr"),
        );
        self.emit.emit_word(
            aarch64::encode_ldr_imm(aarch64::Width::X64, Reg::X3, Reg::SP, count_slot)
                .expect("ldr"),
        );
        self.emit_call_fn_ptr(trampoline);
    }

    /// Call `kajit_map_build(from_pair_slice_fn, x21, null, 0)` — empty map.
    ///
    /// Same trampoline pattern as `emit_call_map_from_pairs`.
    pub fn emit_call_map_from_pairs_empty(&mut self, from_pair_slice_fn: *const u8) {
        let trampoline = crate::intrinsics::kajit_map_build as *const u8;
        self.emit_load_imm64(Reg::X8, from_pair_slice_fn as u64);
        self.emit.emit_word(
            aarch64::encode_mov_reg(aarch64::Width::X64, Reg::X0, Reg::X8).expect("mov"),
        );
        self.emit.emit_word(
            aarch64::encode_mov_reg(aarch64::Width::X64, Reg::X1, Reg::X21).expect("mov"),
        );
        self.emit.emit_word(
            aarch64::encode_mov_reg(aarch64::Width::X64, Reg::X2, Reg::XZR).expect("mov"),
        );
        self.emit.emit_word(
            aarch64::encode_mov_reg(aarch64::Width::X64, Reg::X3, Reg::XZR).expect("mov"),
        );
        self.emit_call_fn_ptr(trampoline);
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
        self.emit.emit_word(
            aarch64::encode_ldr_imm(aarch64::Width::X64, Reg::X0, Reg::SP, buf_slot).expect("ldr"),
        );
        self.emit.emit_word(
            aarch64::encode_ldr_imm(aarch64::Width::X64, Reg::X1, Reg::SP, cap_slot).expect("ldr"),
        );
        self.emit.emit_word(
            aarch64::encode_movz(aarch64::Width::X64, Reg::X2, pair_stride as u16, 0)
                .expect("movz"),
        );
        self.emit.emit_word(
            aarch64::encode_movz(aarch64::Width::X64, Reg::X3, pair_align as u16, 0).expect("movz"),
        );
        self.emit_call_fn_ptr(free_fn);
    }

    // ── Recipe emission ─────────────────────────────────────────────
    /// Emit a recipe — interpret a sequence of micro-ops into aarch64 instructions.
    pub fn emit_recipe(&mut self, recipe: &Recipe) {
        let error_exit = self.error_exit;

        // Allocate dynamic labels for the recipe
        let labels: Vec<LabelId> = (0..recipe.label_count)
            .map(|_| self.emit.new_label())
            .collect();

        // Shared EOF error label (lazily bound on first use)
        let eof_label = self.emit.new_label();

        for op in &recipe.ops {
            match op {
                Op::BoundsCheck { count } => {
                    if *count == 1 {
                        self.emit.emit_word(
                            aarch64::encode_cmp_reg(aarch64::Width::X64, Reg::X19, Reg::X20)
                                .expect("cmp"),
                        );
                        self.emit
                            .emit_b_cond_label(aarch64::Condition::Hs, eof_label)
                            .expect("b.hs");
                    } else {
                        let count = *count;
                        self.emit.emit_word(
                            aarch64::encode_sub_reg(
                                aarch64::Width::X64,
                                Reg::X9,
                                Reg::X20,
                                Reg::X19,
                            )
                            .expect("sub"),
                        );
                        self.emit.emit_word(
                            aarch64::encode_cmp_imm(
                                aarch64::Width::X64,
                                Reg::X9,
                                count as u16,
                                false,
                            )
                            .expect("cmp"),
                        );
                        self.emit
                            .emit_b_cond_label(aarch64::Condition::Lo, eof_label)
                            .expect("b.lo");
                    }
                }
                Op::LoadByte { dst } => match dst {
                    Slot::A => self
                        .emit
                        .emit_word(aarch64::encode_ldrb_imm(Reg::X9, Reg::X19, 0).expect("ldrb")),
                    Slot::B => self
                        .emit
                        .emit_word(aarch64::encode_ldrb_imm(Reg::X10, Reg::X19, 0).expect("ldrb")),
                },
                Op::LoadFromCursor { dst, width } => match (dst, width) {
                    (Slot::A, Width::W4) => self.emit.emit_word(
                        aarch64::encode_ldr_imm(aarch64::Width::W32, Reg::X9, Reg::X19, 0)
                            .expect("ldr"),
                    ),
                    (Slot::A, Width::W8) => self.emit.emit_word(
                        aarch64::encode_ldr_imm(aarch64::Width::X64, Reg::X9, Reg::X19, 0)
                            .expect("ldr"),
                    ),
                    (Slot::B, Width::W4) => self.emit.emit_word(
                        aarch64::encode_ldr_imm(aarch64::Width::W32, Reg::X10, Reg::X19, 0)
                            .expect("ldr"),
                    ),
                    (Slot::B, Width::W8) => self.emit.emit_word(
                        aarch64::encode_ldr_imm(aarch64::Width::X64, Reg::X10, Reg::X19, 0)
                            .expect("ldr"),
                    ),
                    _ => panic!("unsupported LoadFromCursor width"),
                },
                Op::StoreToOut { src, offset, width } => {
                    let offset = *offset;
                    match (src, width) {
                        (Slot::A, Width::W1) => self.emit.emit_word(
                            aarch64::encode_strb_imm(Reg::X9, Reg::X21, offset).expect("strb"),
                        ),
                        (Slot::A, Width::W2) => self.emit.emit_word(
                            aarch64::encode_strh_imm(Reg::X9, Reg::X21, offset).expect("strh"),
                        ),
                        (Slot::A, Width::W4) => self.emit.emit_word(
                            aarch64::encode_str_imm(aarch64::Width::W32, Reg::X9, Reg::X21, offset)
                                .expect("str"),
                        ),
                        (Slot::A, Width::W8) => self.emit.emit_word(
                            aarch64::encode_str_imm(aarch64::Width::X64, Reg::X9, Reg::X21, offset)
                                .expect("str"),
                        ),
                        (Slot::B, Width::W1) => self.emit.emit_word(
                            aarch64::encode_strb_imm(Reg::X10, Reg::X21, offset).expect("strb"),
                        ),
                        (Slot::B, Width::W2) => self.emit.emit_word(
                            aarch64::encode_strh_imm(Reg::X10, Reg::X21, offset).expect("strh"),
                        ),
                        (Slot::B, Width::W4) => self.emit.emit_word(
                            aarch64::encode_str_imm(
                                aarch64::Width::W32,
                                Reg::X10,
                                Reg::X21,
                                offset,
                            )
                            .expect("str"),
                        ),
                        (Slot::B, Width::W8) => self.emit.emit_word(
                            aarch64::encode_str_imm(
                                aarch64::Width::X64,
                                Reg::X10,
                                Reg::X21,
                                offset,
                            )
                            .expect("str"),
                        ),
                    }
                }
                Op::StoreByteToStack { src, sp_offset } => {
                    let sp_offset = *sp_offset;
                    match src {
                        Slot::A => self.emit.emit_word(
                            aarch64::encode_strb_imm(Reg::X9, Reg::SP, sp_offset).expect("strb"),
                        ),
                        Slot::B => self.emit.emit_word(
                            aarch64::encode_strb_imm(Reg::X10, Reg::SP, sp_offset).expect("strb"),
                        ),
                    }
                }
                Op::StoreToStack {
                    src,
                    sp_offset,
                    width,
                } => {
                    let sp_offset = *sp_offset;
                    match (src, width) {
                        (Slot::A, Width::W4) => self.emit.emit_word(
                            aarch64::encode_str_imm(
                                aarch64::Width::W32,
                                Reg::X9,
                                Reg::SP,
                                sp_offset,
                            )
                            .expect("str"),
                        ),
                        (Slot::A, Width::W8) => self.emit.emit_word(
                            aarch64::encode_str_imm(
                                aarch64::Width::X64,
                                Reg::X9,
                                Reg::SP,
                                sp_offset,
                            )
                            .expect("str"),
                        ),
                        (Slot::B, Width::W4) => self.emit.emit_word(
                            aarch64::encode_str_imm(
                                aarch64::Width::W32,
                                Reg::X10,
                                Reg::SP,
                                sp_offset,
                            )
                            .expect("str"),
                        ),
                        (Slot::B, Width::W8) => self.emit.emit_word(
                            aarch64::encode_str_imm(
                                aarch64::Width::X64,
                                Reg::X10,
                                Reg::SP,
                                sp_offset,
                            )
                            .expect("str"),
                        ),
                        _ => panic!("unsupported StoreToStack width"),
                    }
                }
                Op::LoadFromStack {
                    dst,
                    sp_offset,
                    width,
                } => {
                    let sp_offset = *sp_offset;
                    match (dst, width) {
                        (Slot::A, Width::W4) => self.emit.emit_word(
                            aarch64::encode_ldr_imm(
                                aarch64::Width::W32,
                                Reg::X9,
                                Reg::SP,
                                sp_offset,
                            )
                            .expect("ldr"),
                        ),
                        (Slot::A, Width::W8) => self.emit.emit_word(
                            aarch64::encode_ldr_imm(
                                aarch64::Width::X64,
                                Reg::X9,
                                Reg::SP,
                                sp_offset,
                            )
                            .expect("ldr"),
                        ),
                        (Slot::B, Width::W4) => self.emit.emit_word(
                            aarch64::encode_ldr_imm(
                                aarch64::Width::W32,
                                Reg::X10,
                                Reg::SP,
                                sp_offset,
                            )
                            .expect("ldr"),
                        ),
                        (Slot::B, Width::W8) => self.emit.emit_word(
                            aarch64::encode_ldr_imm(
                                aarch64::Width::X64,
                                Reg::X10,
                                Reg::SP,
                                sp_offset,
                            )
                            .expect("ldr"),
                        ),
                        _ => panic!("unsupported LoadFromStack width"),
                    }
                }
                Op::AdvanceCursor { count } => {
                    let count = *count;
                    self.emit.emit_word(
                        aarch64::encode_add_imm(
                            aarch64::Width::X64,
                            Reg::X19,
                            Reg::X19,
                            count as u16,
                            false,
                        )
                        .expect("add"),
                    );
                }
                Op::AdvanceCursorBySlot { slot } => match slot {
                    Slot::A => self.emit.emit_word(
                        aarch64::encode_add_reg(aarch64::Width::X64, Reg::X19, Reg::X19, Reg::X9)
                            .expect("add"),
                    ),
                    Slot::B => self.emit.emit_word(
                        aarch64::encode_add_reg(aarch64::Width::X64, Reg::X19, Reg::X19, Reg::X10)
                            .expect("add"),
                    ),
                },
                Op::ZigzagDecode { slot } => match slot {
                    Slot::A => {
                        self.emit.emit_word(
                            aarch64::encode_lsr_imm(aarch64::Width::W32, Reg::X10, Reg::X9, 1)
                                .expect("lsr"),
                        );
                        self.emit.emit_word(
                            aarch64::encode_and_imm(aarch64::Width::W32, Reg::X11, Reg::X9, 1)
                                .expect("and"),
                        );
                        self.emit.emit_word(
                            aarch64::encode_neg(aarch64::Width::W32, Reg::X11, Reg::X11)
                                .expect("neg"),
                        );
                        self.emit.emit_word(
                            aarch64::encode_eor_reg(
                                aarch64::Width::W32,
                                Reg::X9,
                                Reg::X10,
                                Reg::X11,
                                aarch64::Shift::Lsl,
                                0,
                            )
                            .expect("eor"),
                        );
                    }
                    Slot::B => {
                        self.emit.emit_word(
                            aarch64::encode_lsr_imm(aarch64::Width::W32, Reg::X11, Reg::X10, 1)
                                .expect("lsr"),
                        );
                        self.emit.emit_word(
                            aarch64::encode_and_imm(aarch64::Width::W32, Reg::X9, Reg::X10, 1)
                                .expect("and"),
                        );
                        self.emit.emit_word(
                            aarch64::encode_neg(aarch64::Width::W32, Reg::X9, Reg::X9)
                                .expect("neg"),
                        );
                        self.emit.emit_word(
                            aarch64::encode_eor_reg(
                                aarch64::Width::W32,
                                Reg::X10,
                                Reg::X11,
                                Reg::X9,
                                aarch64::Shift::Lsl,
                                0,
                            )
                            .expect("eor"),
                        );
                    }
                },
                Op::ValidateMax {
                    slot,
                    max_val,
                    error,
                } => {
                    let max_val = *max_val;
                    let error_code = *error as u32;
                    let invalid_label = self.emit.new_label();
                    let ok_label = self.emit.new_label();
                    match slot {
                        Slot::A => self.emit.emit_word(
                            aarch64::encode_cmp_imm(
                                aarch64::Width::W32,
                                Reg::X9,
                                max_val as u16,
                                false,
                            )
                            .expect("cmp"),
                        ),
                        Slot::B => self.emit.emit_word(
                            aarch64::encode_cmp_imm(
                                aarch64::Width::W32,
                                Reg::X10,
                                max_val as u16,
                                false,
                            )
                            .expect("cmp"),
                        ),
                    }
                    self.emit
                        .emit_b_cond_label(aarch64::Condition::Hi, invalid_label)
                        .expect("b.hi");
                    self.emit.emit_b_label(ok_label).expect("b");
                    self.emit.bind_label(invalid_label).expect("bind");
                    self.emit.emit_word(
                        aarch64::encode_movz(aarch64::Width::W32, Reg::X9, error_code as u16, 0)
                            .expect("movz"),
                    );
                    self.emit.emit_word(
                        aarch64::encode_str_imm(
                            aarch64::Width::W32,
                            Reg::X9,
                            Reg::X22,
                            CTX_ERROR_CODE as u32,
                        )
                        .expect("str"),
                    );
                    self.emit.emit_b_label(error_exit).expect("b");
                    self.emit.bind_label(ok_label).expect("bind");
                }
                Op::TestBit7Branch { slot, target } => {
                    let label = labels[*target];
                    match slot {
                        Slot::A => self.emit.emit_tbnz_label(Reg::X9, 7, label).expect("tbnz"),
                        Slot::B => self.emit.emit_tbnz_label(Reg::X10, 7, label).expect("tbnz"),
                    }
                }
                Op::Branch { target } => {
                    let label = labels[*target];
                    self.emit.emit_b_label(label).expect("b");
                }
                Op::BindLabel { index } => {
                    let label = labels[*index];
                    self.emit.bind_label(label).expect("bind");
                }
                Op::CallIntrinsic {
                    fn_ptr,
                    field_offset,
                } => {
                    let field_offset = *field_offset;
                    self.emit_flush_input_cursor();
                    self.emit.emit_word(
                        aarch64::encode_mov_reg(aarch64::Width::X64, Reg::X0, Reg::X22)
                            .expect("mov"),
                    );
                    self.emit.emit_word(
                        aarch64::encode_add_imm(
                            aarch64::Width::X64,
                            Reg::X1,
                            Reg::X21,
                            field_offset as u16,
                            false,
                        )
                        .expect("add"),
                    );
                    self.emit_call_fn_ptr(*fn_ptr);
                    self.emit_reload_cursor_and_check_error();
                }
                Op::CallIntrinsicStackOut { fn_ptr, sp_offset } => {
                    let sp_offset = *sp_offset;
                    self.emit_flush_input_cursor();
                    self.emit.emit_word(
                        aarch64::encode_mov_reg(aarch64::Width::X64, Reg::X0, Reg::X22)
                            .expect("mov"),
                    );
                    self.emit.emit_word(
                        aarch64::encode_add_imm(
                            aarch64::Width::X64,
                            Reg::X1,
                            Reg::SP,
                            sp_offset as u16,
                            false,
                        )
                        .expect("add"),
                    );
                    self.emit_call_fn_ptr(*fn_ptr);
                    self.emit_reload_cursor_and_check_error();
                }
                Op::ComputeRemaining { dst } => match dst {
                    Slot::A => self.emit.emit_word(
                        aarch64::encode_sub_reg(aarch64::Width::X64, Reg::X9, Reg::X20, Reg::X19)
                            .expect("sub"),
                    ),
                    Slot::B => self.emit.emit_word(
                        aarch64::encode_sub_reg(aarch64::Width::X64, Reg::X10, Reg::X20, Reg::X19)
                            .expect("sub"),
                    ),
                },
                Op::CmpBranchLo { lhs, rhs, on_fail } => {
                    let target = match on_fail {
                        ErrorTarget::Eof => eof_label,
                        ErrorTarget::ErrorExit => error_exit,
                    };
                    match (lhs, rhs) {
                        (Slot::A, Slot::B) => {
                            self.emit.emit_word(
                                aarch64::encode_cmp_reg(aarch64::Width::X64, Reg::X9, Reg::X10)
                                    .expect("cmp"),
                            );
                            self.emit
                                .emit_b_cond_label(aarch64::Condition::Lo, target)
                                .expect("b.lo");
                        }
                        (Slot::B, Slot::A) => {
                            self.emit.emit_word(
                                aarch64::encode_cmp_reg(aarch64::Width::X64, Reg::X10, Reg::X9)
                                    .expect("cmp"),
                            );
                            self.emit
                                .emit_b_cond_label(aarch64::Condition::Lo, target)
                                .expect("b.lo");
                        }
                        _ => panic!("CmpBranchLo requires different slots"),
                    }
                }
                Op::SaveCursor { dst } => match dst {
                    Slot::A => self.emit.emit_word(
                        aarch64::encode_mov_reg(aarch64::Width::X64, Reg::X9, Reg::X19)
                            .expect("mov"),
                    ),
                    Slot::B => self.emit.emit_word(
                        aarch64::encode_mov_reg(aarch64::Width::X64, Reg::X10, Reg::X19)
                            .expect("mov"),
                    ),
                },
                Op::CallValidateAllocCopy {
                    fn_ptr,
                    data_src,
                    len_src,
                } => {
                    self.emit_flush_input_cursor();
                    self.emit.emit_word(
                        aarch64::encode_mov_reg(aarch64::Width::X64, Reg::X0, Reg::X22)
                            .expect("mov"),
                    );
                    match data_src {
                        Slot::A => self.emit.emit_word(
                            aarch64::encode_mov_reg(aarch64::Width::X64, Reg::X1, Reg::X9)
                                .expect("mov"),
                        ),
                        Slot::B => self.emit.emit_word(
                            aarch64::encode_mov_reg(aarch64::Width::X64, Reg::X1, Reg::X10)
                                .expect("mov"),
                        ),
                    }
                    match len_src {
                        Slot::A => self.emit.emit_word(
                            aarch64::encode_mov_reg(aarch64::Width::W32, Reg::X2, Reg::X9)
                                .expect("mov"),
                        ),
                        Slot::B => self.emit.emit_word(
                            aarch64::encode_mov_reg(aarch64::Width::W32, Reg::X2, Reg::X10)
                                .expect("mov"),
                        ),
                    }
                    self.emit_call_fn_ptr(*fn_ptr);
                    self.emit_check_error();
                }
                Op::WriteMalumString {
                    base_offset,
                    ptr_off,
                    len_off,
                    cap_off,
                    len_slot,
                } => {
                    let ptr_offset = *base_offset + *ptr_off;
                    let len_offset = *base_offset + *len_off;
                    let cap_offset = *base_offset + *cap_off;
                    // x0 = buf pointer from previous call return
                    self.emit.emit_word(
                        aarch64::encode_str_imm(aarch64::Width::X64, Reg::X0, Reg::X21, ptr_offset)
                            .expect("str"),
                    );
                    match len_slot {
                        Slot::A => {
                            self.emit.emit_word(
                                aarch64::encode_str_imm(
                                    aarch64::Width::X64,
                                    Reg::X9,
                                    Reg::X21,
                                    len_offset,
                                )
                                .expect("str"),
                            );
                            self.emit.emit_word(
                                aarch64::encode_str_imm(
                                    aarch64::Width::X64,
                                    Reg::X9,
                                    Reg::X21,
                                    cap_offset,
                                )
                                .expect("str"),
                            );
                        }
                        Slot::B => {
                            self.emit.emit_word(
                                aarch64::encode_str_imm(
                                    aarch64::Width::X64,
                                    Reg::X10,
                                    Reg::X21,
                                    len_offset,
                                )
                                .expect("str"),
                            );
                            self.emit.emit_word(
                                aarch64::encode_str_imm(
                                    aarch64::Width::X64,
                                    Reg::X10,
                                    Reg::X21,
                                    cap_offset,
                                )
                                .expect("str"),
                            );
                        }
                    }
                }

                // ── Encode-direction ops ──────────────────────────────────
                //
                // In encode mode the register assignments are:
                //   x19 = output_ptr (write cursor)
                //   x20 = output_end
                //   x21 = input struct pointer
                //   x22 = EncodeContext pointer
                Op::LoadFromInput { dst, offset, width } => {
                    let offset = *offset;
                    match (dst, width) {
                        (Slot::A, Width::W1) => self.emit.emit_word(
                            aarch64::encode_ldrb_imm(Reg::X9, Reg::X21, offset).expect("ldrb"),
                        ),
                        (Slot::A, Width::W2) => self.emit.emit_word(
                            aarch64::encode_ldrh_imm(Reg::X9, Reg::X21, offset).expect("ldrh"),
                        ),
                        (Slot::A, Width::W4) => self.emit.emit_word(
                            aarch64::encode_ldr_imm(aarch64::Width::W32, Reg::X9, Reg::X21, offset)
                                .expect("ldr"),
                        ),
                        (Slot::A, Width::W8) => self.emit.emit_word(
                            aarch64::encode_ldr_imm(aarch64::Width::X64, Reg::X9, Reg::X21, offset)
                                .expect("ldr"),
                        ),
                        (Slot::B, Width::W1) => self.emit.emit_word(
                            aarch64::encode_ldrb_imm(Reg::X10, Reg::X21, offset).expect("ldrb"),
                        ),
                        (Slot::B, Width::W2) => self.emit.emit_word(
                            aarch64::encode_ldrh_imm(Reg::X10, Reg::X21, offset).expect("ldrh"),
                        ),
                        (Slot::B, Width::W4) => self.emit.emit_word(
                            aarch64::encode_ldr_imm(
                                aarch64::Width::W32,
                                Reg::X10,
                                Reg::X21,
                                offset,
                            )
                            .expect("ldr"),
                        ),
                        (Slot::B, Width::W8) => self.emit.emit_word(
                            aarch64::encode_ldr_imm(
                                aarch64::Width::X64,
                                Reg::X10,
                                Reg::X21,
                                offset,
                            )
                            .expect("ldr"),
                        ),
                    }
                }
                Op::StoreToOutput { src, width } => match (src, width) {
                    (Slot::A, Width::W1) => {
                        self.emit.emit_word(
                            aarch64::encode_strb_imm(Reg::X9, Reg::X19, 0).expect("strb"),
                        );
                        self.emit.emit_word(
                            aarch64::encode_add_imm(
                                aarch64::Width::X64,
                                Reg::X19,
                                Reg::X19,
                                1,
                                false,
                            )
                            .expect("add"),
                        );
                    }
                    (Slot::A, Width::W2) => {
                        self.emit.emit_word(
                            aarch64::encode_strh_imm(Reg::X9, Reg::X19, 0).expect("strh"),
                        );
                        self.emit.emit_word(
                            aarch64::encode_add_imm(
                                aarch64::Width::X64,
                                Reg::X19,
                                Reg::X19,
                                2,
                                false,
                            )
                            .expect("add"),
                        );
                    }
                    (Slot::A, Width::W4) => {
                        self.emit.emit_word(
                            aarch64::encode_str_imm(aarch64::Width::W32, Reg::X9, Reg::X19, 0)
                                .expect("str"),
                        );
                        self.emit.emit_word(
                            aarch64::encode_add_imm(
                                aarch64::Width::X64,
                                Reg::X19,
                                Reg::X19,
                                4,
                                false,
                            )
                            .expect("add"),
                        );
                    }
                    (Slot::A, Width::W8) => {
                        self.emit.emit_word(
                            aarch64::encode_str_imm(aarch64::Width::X64, Reg::X9, Reg::X19, 0)
                                .expect("str"),
                        );
                        self.emit.emit_word(
                            aarch64::encode_add_imm(
                                aarch64::Width::X64,
                                Reg::X19,
                                Reg::X19,
                                8,
                                false,
                            )
                            .expect("add"),
                        );
                    }
                    (Slot::B, Width::W1) => {
                        self.emit.emit_word(
                            aarch64::encode_strb_imm(Reg::X10, Reg::X19, 0).expect("strb"),
                        );
                        self.emit.emit_word(
                            aarch64::encode_add_imm(
                                aarch64::Width::X64,
                                Reg::X19,
                                Reg::X19,
                                1,
                                false,
                            )
                            .expect("add"),
                        );
                    }
                    (Slot::B, Width::W2) => {
                        self.emit.emit_word(
                            aarch64::encode_strh_imm(Reg::X10, Reg::X19, 0).expect("strh"),
                        );
                        self.emit.emit_word(
                            aarch64::encode_add_imm(
                                aarch64::Width::X64,
                                Reg::X19,
                                Reg::X19,
                                2,
                                false,
                            )
                            .expect("add"),
                        );
                    }
                    (Slot::B, Width::W4) => {
                        self.emit.emit_word(
                            aarch64::encode_str_imm(aarch64::Width::W32, Reg::X10, Reg::X19, 0)
                                .expect("str"),
                        );
                        self.emit.emit_word(
                            aarch64::encode_add_imm(
                                aarch64::Width::X64,
                                Reg::X19,
                                Reg::X19,
                                4,
                                false,
                            )
                            .expect("add"),
                        );
                    }
                    (Slot::B, Width::W8) => {
                        self.emit.emit_word(
                            aarch64::encode_str_imm(aarch64::Width::X64, Reg::X10, Reg::X19, 0)
                                .expect("str"),
                        );
                        self.emit.emit_word(
                            aarch64::encode_add_imm(
                                aarch64::Width::X64,
                                Reg::X19,
                                Reg::X19,
                                8,
                                false,
                            )
                            .expect("add"),
                        );
                    }
                },
                Op::WriteByte { value } => {
                    let value = *value as u32;
                    self.emit.emit_word(
                        aarch64::encode_movz(aarch64::Width::W32, Reg::X9, value as u16, 0)
                            .expect("movz"),
                    );
                    self.emit
                        .emit_word(aarch64::encode_strb_imm(Reg::X9, Reg::X19, 0).expect("strb"));
                    self.emit.emit_word(
                        aarch64::encode_add_imm(aarch64::Width::X64, Reg::X19, Reg::X19, 1, false)
                            .expect("add"),
                    );
                }
                Op::AdvanceOutput { count } => {
                    let count = *count;
                    if count > 0 {
                        self.emit.emit_word(
                            aarch64::encode_add_imm(
                                aarch64::Width::X64,
                                Reg::X19,
                                Reg::X19,
                                count as u16,
                                false,
                            )
                            .expect("add"),
                        );
                    }
                }
                Op::AdvanceOutputBySlot { slot } => match slot {
                    Slot::A => self.emit.emit_word(
                        aarch64::encode_add_reg(aarch64::Width::X64, Reg::X19, Reg::X19, Reg::X9)
                            .expect("add"),
                    ),
                    Slot::B => self.emit.emit_word(
                        aarch64::encode_add_reg(aarch64::Width::X64, Reg::X19, Reg::X19, Reg::X10)
                            .expect("add"),
                    ),
                },
                Op::OutputBoundsCheck { count } => {
                    let count = *count;
                    let have_space = self.emit.new_label();
                    let grow_fn = crate::intrinsics::kajit_output_grow as *const u8;

                    self.emit.emit_word(
                        aarch64::encode_sub_reg(aarch64::Width::X64, Reg::X11, Reg::X20, Reg::X19)
                            .expect("sub"),
                    );
                    self.emit.emit_word(
                        aarch64::encode_cmp_imm(aarch64::Width::X64, Reg::X11, count as u16, false)
                            .expect("cmp"),
                    );
                    self.emit
                        .emit_b_cond_label(aarch64::Condition::Hs, have_space)
                        .expect("b.hs");
                    self.emit_enc_flush_output_cursor();
                    self.emit.emit_word(
                        aarch64::encode_mov_reg(aarch64::Width::X64, Reg::X0, Reg::X22)
                            .expect("mov"),
                    );
                    self.emit.emit_word(
                        aarch64::encode_movz(aarch64::Width::X64, Reg::X1, count as u16, 0)
                            .expect("movz"),
                    );
                    self.emit_call_fn_ptr(grow_fn);
                    self.emit_enc_reload_and_check_error();
                    self.emit.bind_label(have_space).expect("bind");
                }
                Op::SignExtend { slot, from } => match (slot, from) {
                    (Slot::A, Width::W1) => self
                        .emit
                        .emit_word(aarch64::encode_sxtb(Reg::X9, Reg::X9).expect("sxtb")),
                    (Slot::A, Width::W2) => self
                        .emit
                        .emit_word(aarch64::encode_sxth(Reg::X9, Reg::X9).expect("sxth")),
                    (Slot::B, Width::W1) => self
                        .emit
                        .emit_word(aarch64::encode_sxtb(Reg::X10, Reg::X10).expect("sxtb")),
                    (Slot::B, Width::W2) => self
                        .emit
                        .emit_word(aarch64::encode_sxth(Reg::X10, Reg::X10).expect("sxth")),
                    (_, Width::W4 | Width::W8) => {} // already at natural width
                },
                Op::ZigzagEncode { slot, wide } => match (slot, wide) {
                    // zigzag encode 32-bit: (n << 1) ^ (n >> 31)
                    (Slot::A, false) => {
                        self.emit.emit_word(
                            aarch64::encode_lsl_imm(aarch64::Width::W32, Reg::X10, Reg::X9, 1)
                                .expect("lsl"),
                        );
                        self.emit.emit_word(
                            aarch64::encode_asr_imm(aarch64::Width::W32, Reg::X11, Reg::X9, 31)
                                .expect("asr"),
                        );
                        self.emit.emit_word(
                            aarch64::encode_eor_reg(
                                aarch64::Width::W32,
                                Reg::X9,
                                Reg::X10,
                                Reg::X11,
                                aarch64::Shift::Lsl,
                                0,
                            )
                            .expect("eor"),
                        );
                    }
                    (Slot::B, false) => {
                        self.emit.emit_word(
                            aarch64::encode_lsl_imm(aarch64::Width::W32, Reg::X11, Reg::X10, 1)
                                .expect("lsl"),
                        );
                        self.emit.emit_word(
                            aarch64::encode_asr_imm(aarch64::Width::W32, Reg::X9, Reg::X10, 31)
                                .expect("asr"),
                        );
                        self.emit.emit_word(
                            aarch64::encode_eor_reg(
                                aarch64::Width::W32,
                                Reg::X10,
                                Reg::X11,
                                Reg::X9,
                                aarch64::Shift::Lsl,
                                0,
                            )
                            .expect("eor"),
                        );
                    }
                    // zigzag encode 64-bit: (n << 1) ^ (n >> 63)
                    (Slot::A, true) => {
                        self.emit.emit_word(
                            aarch64::encode_lsl_imm(aarch64::Width::X64, Reg::X10, Reg::X9, 1)
                                .expect("lsl"),
                        );
                        self.emit.emit_word(
                            aarch64::encode_asr_imm(aarch64::Width::X64, Reg::X11, Reg::X9, 63)
                                .expect("asr"),
                        );
                        self.emit.emit_word(
                            aarch64::encode_eor_reg(
                                aarch64::Width::X64,
                                Reg::X9,
                                Reg::X10,
                                Reg::X11,
                                aarch64::Shift::Lsl,
                                0,
                            )
                            .expect("eor"),
                        );
                    }
                    (Slot::B, true) => {
                        self.emit.emit_word(
                            aarch64::encode_lsl_imm(aarch64::Width::X64, Reg::X11, Reg::X10, 1)
                                .expect("lsl"),
                        );
                        self.emit.emit_word(
                            aarch64::encode_asr_imm(aarch64::Width::X64, Reg::X9, Reg::X10, 63)
                                .expect("asr"),
                        );
                        self.emit.emit_word(
                            aarch64::encode_eor_reg(
                                aarch64::Width::X64,
                                Reg::X10,
                                Reg::X11,
                                Reg::X9,
                                aarch64::Shift::Lsl,
                                0,
                            )
                            .expect("eor"),
                        );
                    }
                },
                Op::EncodeVarint { slot, wide } => {
                    // Inline varint encoding loop.
                    // While value >= 0x80: write (byte | 0x80), shift right 7.
                    // Then write final byte.
                    let loop_label = self.emit.new_label();
                    let done_label = self.emit.new_label();

                    if *wide {
                        // 64-bit varint: use x11 register
                        match slot {
                            Slot::A => self.emit.emit_word(
                                aarch64::encode_mov_reg(aarch64::Width::X64, Reg::X11, Reg::X9)
                                    .expect("mov"),
                            ),
                            Slot::B => self.emit.emit_word(
                                aarch64::encode_mov_reg(aarch64::Width::X64, Reg::X11, Reg::X10)
                                    .expect("mov"),
                            ),
                        }
                        self.emit.bind_label(loop_label).expect("bind");
                        self.emit.emit_word(
                            aarch64::encode_cmp_imm(aarch64::Width::X64, Reg::X11, 0x80, false)
                                .expect("cmp"),
                        );
                        self.emit
                            .emit_b_cond_label(aarch64::Condition::Lo, done_label)
                            .expect("b.lo");
                        self.emit.emit_word(
                            aarch64::encode_orr_imm(aarch64::Width::W32, Reg::X9, Reg::X11, 0x80)
                                .expect("orr"),
                        );
                        self.emit.emit_word(
                            aarch64::encode_strb_imm(Reg::X9, Reg::X19, 0).expect("strb"),
                        );
                        self.emit.emit_word(
                            aarch64::encode_add_imm(
                                aarch64::Width::X64,
                                Reg::X19,
                                Reg::X19,
                                1,
                                false,
                            )
                            .expect("add"),
                        );
                        self.emit.emit_word(
                            aarch64::encode_lsr_imm(aarch64::Width::X64, Reg::X11, Reg::X11, 7)
                                .expect("lsr"),
                        );
                        self.emit.emit_b_label(loop_label).expect("b");
                        self.emit.bind_label(done_label).expect("bind");
                        self.emit.emit_word(
                            aarch64::encode_strb_imm(Reg::X11, Reg::X19, 0).expect("strb"),
                        );
                        self.emit.emit_word(
                            aarch64::encode_add_imm(
                                aarch64::Width::X64,
                                Reg::X19,
                                Reg::X19,
                                1,
                                false,
                            )
                            .expect("add"),
                        );
                    } else {
                        // 32-bit varint: use w11 register
                        match slot {
                            Slot::A => self.emit.emit_word(
                                aarch64::encode_mov_reg(aarch64::Width::W32, Reg::X11, Reg::X9)
                                    .expect("mov"),
                            ),
                            Slot::B => self.emit.emit_word(
                                aarch64::encode_mov_reg(aarch64::Width::W32, Reg::X11, Reg::X10)
                                    .expect("mov"),
                            ),
                        }
                        self.emit.bind_label(loop_label).expect("bind");
                        self.emit.emit_word(
                            aarch64::encode_cmp_imm(aarch64::Width::W32, Reg::X11, 0x80, false)
                                .expect("cmp"),
                        );
                        self.emit
                            .emit_b_cond_label(aarch64::Condition::Lo, done_label)
                            .expect("b.lo");
                        self.emit.emit_word(
                            aarch64::encode_orr_imm(aarch64::Width::W32, Reg::X9, Reg::X11, 0x80)
                                .expect("orr"),
                        );
                        self.emit.emit_word(
                            aarch64::encode_strb_imm(Reg::X9, Reg::X19, 0).expect("strb"),
                        );
                        self.emit.emit_word(
                            aarch64::encode_add_imm(
                                aarch64::Width::X64,
                                Reg::X19,
                                Reg::X19,
                                1,
                                false,
                            )
                            .expect("add"),
                        );
                        self.emit.emit_word(
                            aarch64::encode_lsr_imm(aarch64::Width::W32, Reg::X11, Reg::X11, 7)
                                .expect("lsr"),
                        );
                        self.emit.emit_b_label(loop_label).expect("b");
                        self.emit.bind_label(done_label).expect("bind");
                        self.emit.emit_word(
                            aarch64::encode_strb_imm(Reg::X11, Reg::X19, 0).expect("strb"),
                        );
                        self.emit.emit_word(
                            aarch64::encode_add_imm(
                                aarch64::Width::X64,
                                Reg::X19,
                                Reg::X19,
                                1,
                                false,
                            )
                            .expect("add"),
                        );
                    }
                }
            }
        }

        // Jump over cold path, then emit shared EOF error
        let done_label = self.emit.new_label();
        let eof_code = crate::context::ErrorCode::UnexpectedEof as u32;
        self.emit.emit_b_label(done_label).expect("b");
        self.emit.bind_label(eof_label).expect("bind");
        self.emit.emit_word(
            aarch64::encode_movz(aarch64::Width::W32, Reg::X9, eof_code as u16, 0).expect("movz"),
        );
        self.emit.emit_word(
            aarch64::encode_str_imm(
                aarch64::Width::W32,
                Reg::X9,
                Reg::X22,
                CTX_ERROR_CODE as u32,
            )
            .expect("str"),
        );
        self.emit.emit_b_label(error_exit).expect("b");
        self.emit.bind_label(done_label).expect("bind");
    }
    // ── JIT f64 parser (uscale algorithm) ───────────────────────────────
    //
    // r[impl deser.json.scalar.float]
    // r[impl deser.json.scalar.float.ws]
    // r[impl deser.json.scalar.float.sign]
    // r[impl deser.json.scalar.float.digits]
    // r[impl deser.json.scalar.float.overflow-digits]
    // r[impl deser.json.scalar.float.dot]
    // r[impl deser.json.scalar.float.exponent]
    // r[impl deser.json.scalar.float.validation]
    // r[impl deser.json.scalar.float.zero]
    // r[impl deser.json.scalar.float.exact-int]
    // r[impl deser.json.scalar.float.uscale]
    // r[impl deser.json.scalar.float.uscale.table]
    // r[impl deser.json.scalar.float.uscale.mul128]
    // r[impl deser.json.scalar.float.uscale.clz]
    // r[impl deser.json.scalar.float.pack]
    // r[impl deser.json.scalar.float.pack.subnormal]
    // r[impl deser.json.scalar.float.pack.overflow]
    //
    // Register map:
    //   x19 = cursor    (callee-saved, persistent)
    //   x20 = end       (callee-saved, persistent)
    //   x21 = out       (callee-saved, persistent)
    //   x22 = ctx       (callee-saved, persistent)
    //   x0  = sign (0/1), then reused as scratch in pack
    //   x1  = scratch / pm_hi
    //   x2  = scratch / pm_lo / constant 10
    //   x3  = shift count s
    //   x4  = clz → mul-hi
    //   x5  = left-justified x → mul-mid scratch
    //   x6  = e (binary exponent)
    //   x7  = result bits
    //   x8  = scratch
    //   x9  = mantissa d
    //   x10 = nd (significant digits, ≤ 19)
    //   x11 = frac_digits (signed)
    //   x12 = dropped → p (power-of-10)
    //   x13 = flags (bit 0 = has_dot, bit 1 = saw_digit)
    //   x14 = exp_val → lp → scratch in uscale
    //   x15 = exp_neg → scratch in uscale
    pub fn emit_jit_f64_parse(&mut self, offset: u32) {
        let error_exit = self.error_exit;

        let l_ws = self.emit.new_label();
        let l_ws_done = self.emit.new_label();
        let l_no_sign = self.emit.new_label();
        let l_skip_lz = self.emit.new_label();
        let l_skip_lz_end = self.emit.new_label();
        let l_int_loop = self.emit.new_label();
        let l_int_done = self.emit.new_label();
        let l_int_ovf = self.emit.new_label();
        let l_int_ovf_end = self.emit.new_label();
        let l_no_dot = self.emit.new_label();
        let l_frac_lz = self.emit.new_label();
        let l_frac_lz_end = self.emit.new_label();
        let l_frac_loop = self.emit.new_label();
        let l_frac_done = self.emit.new_label();
        let l_frac_ovf = self.emit.new_label();
        let l_frac_ovf_end = self.emit.new_label();
        let l_no_exp = self.emit.new_label();
        let l_exp_pos = self.emit.new_label();
        let l_exp_loop = self.emit.new_label();
        let l_exp_done = self.emit.new_label();
        let l_zero = self.emit.new_label();
        let l_exact_int = self.emit.new_label();
        let l_uscale = self.emit.new_label();
        let l_pos_overflow = self.emit.new_label();
        let l_neg_underflow = self.emit.new_label();
        let l_need_lo_mul = self.emit.new_label();
        let l_after_lo_mul = self.emit.new_label();
        let l_pack_normal = self.emit.new_label();
        let l_pack_inf = self.emit.new_label();
        let l_apply_sign = self.emit.new_label();
        let l_done = self.emit.new_label();
        let l_skip_cold = self.emit.new_label();
        let l_err_num = self.emit.new_label();
        let l_err_eof = self.emit.new_label();

        let error_code_invalid = ErrorCode::InvalidJsonNumber as u32;
        let error_code_eof = ErrorCode::UnexpectedEof as u32;
        let tab_ptr = jit_f64::pow10_tab_ptr();
        let tab_lo = (tab_ptr & 0xFFFF) as u32;
        let tab_hi16 = ((tab_ptr >> 16) & 0xFFFF) as u32;
        let tab_hi32 = ((tab_ptr >> 32) & 0xFFFF) as u32;
        let tab_hi48 = ((tab_ptr >> 48) & 0xFFFF) as u32;
        #[allow(non_snake_case)]
        let LOG2_10_LO = jit_f64::LOG2_10_NUM as u32 & 0xFFFF;
        #[allow(non_snake_case)]
        let LOG2_10_HI = (jit_f64::LOG2_10_NUM as u32) >> 16;

        // ── Whitespace skip ──
        self.emit.bind_label(l_ws).expect("bind");
        self.emit.emit_word(
            aarch64::encode_cmp_reg(aarch64::Width::X64, Reg::X19, Reg::X20).expect("cmp"),
        );
        self.emit
            .emit_b_cond_label(aarch64::Condition::Hs, l_ws_done)
            .expect("b.hs");
        self.emit
            .emit_word(aarch64::encode_ldrb_imm(Reg::X8, Reg::X19, 0).expect("ldrb"));
        self.emit.emit_word(
            aarch64::encode_cmp_imm(aarch64::Width::W32, Reg::X8, 0x20, false).expect("cmp"),
        );
        self.emit
            .emit_word(aarch64::encode_b_cond(aarch64::Condition::Eq, 5).expect("b.eq"));
        self.emit.emit_word(
            aarch64::encode_cmp_imm(aarch64::Width::W32, Reg::X8, 0x09, false).expect("cmp"),
        );
        self.emit
            .emit_b_cond_label(aarch64::Condition::Lo, l_ws_done)
            .expect("b.lo");
        self.emit.emit_word(
            aarch64::encode_cmp_imm(aarch64::Width::W32, Reg::X8, 0x0d, false).expect("cmp"),
        );
        self.emit
            .emit_b_cond_label(aarch64::Condition::Hi, l_ws_done)
            .expect("b.hi");
        self.emit.emit_word(
            aarch64::encode_add_imm(aarch64::Width::X64, Reg::X19, Reg::X19, 1, false)
                .expect("add"),
        );
        self.emit.emit_b_label(l_ws).expect("b");
        self.emit.bind_label(l_ws_done).expect("bind");

        // ── Sign ──
        self.emit.emit_word(
            aarch64::encode_cmp_reg(aarch64::Width::X64, Reg::X19, Reg::X20).expect("cmp"),
        );
        self.emit
            .emit_b_cond_label(aarch64::Condition::Hs, l_err_eof)
            .expect("b.hs");
        self.emit
            .emit_word(aarch64::encode_movz(aarch64::Width::X64, Reg::X0, 0, 0).expect("movz"));
        self.emit
            .emit_word(aarch64::encode_ldrb_imm(Reg::X8, Reg::X19, 0).expect("ldrb"));
        self.emit.emit_word(
            aarch64::encode_cmp_imm(aarch64::Width::W32, Reg::X8, 0x2d, false).expect("cmp"),
        );
        self.emit
            .emit_b_cond_label(aarch64::Condition::Ne, l_no_sign)
            .expect("b.ne");
        self.emit
            .emit_word(aarch64::encode_movz(aarch64::Width::X64, Reg::X0, 1, 0).expect("movz"));
        self.emit.emit_word(
            aarch64::encode_add_imm(aarch64::Width::X64, Reg::X19, Reg::X19, 1, false)
                .expect("add"),
        );
        self.emit.emit_word(
            aarch64::encode_cmp_reg(aarch64::Width::X64, Reg::X19, Reg::X20).expect("cmp"),
        );
        self.emit
            .emit_b_cond_label(aarch64::Condition::Hs, l_err_eof)
            .expect("b.hs");
        self.emit
            .emit_word(aarch64::encode_ldrb_imm(Reg::X8, Reg::X19, 0).expect("ldrb"));
        self.emit.bind_label(l_no_sign).expect("bind");

        // ── Digit extraction ──
        self.emit
            .emit_word(aarch64::encode_movz(aarch64::Width::X64, Reg::X9, 0, 0).expect("mov"));
        self.emit
            .emit_word(aarch64::encode_movz(aarch64::Width::X64, Reg::X10, 0, 0).expect("mov"));
        self.emit
            .emit_word(aarch64::encode_movz(aarch64::Width::X64, Reg::X11, 0, 0).expect("mov"));
        self.emit
            .emit_word(aarch64::encode_movz(aarch64::Width::X64, Reg::X12, 0, 0).expect("mov"));
        self.emit
            .emit_word(aarch64::encode_movz(aarch64::Width::X64, Reg::X13, 0, 0).expect("mov"));

        // Leading integer zeros
        self.emit.bind_label(l_skip_lz).expect("bind");
        self.emit.emit_word(
            aarch64::encode_cmp_imm(aarch64::Width::W32, Reg::X8, 0x30, false).expect("cmp"),
        );
        self.emit
            .emit_b_cond_label(aarch64::Condition::Ne, l_skip_lz_end)
            .expect("b.ne");
        self.emit.emit_word(
            aarch64::encode_orr_imm(aarch64::Width::X64, Reg::X13, Reg::X13, 2).expect("orr"),
        );
        self.emit.emit_word(
            aarch64::encode_add_imm(aarch64::Width::X64, Reg::X19, Reg::X19, 1, false)
                .expect("add"),
        );
        self.emit.emit_word(
            aarch64::encode_cmp_reg(aarch64::Width::X64, Reg::X19, Reg::X20).expect("cmp"),
        );
        self.emit
            .emit_b_cond_label(aarch64::Condition::Hs, l_skip_lz_end)
            .expect("b.hs");
        self.emit
            .emit_word(aarch64::encode_ldrb_imm(Reg::X8, Reg::X19, 0).expect("ldrb"));
        self.emit.emit_b_label(l_skip_lz).expect("b");
        self.emit.bind_label(l_skip_lz_end).expect("bind");

        // Integer digit loop
        self.emit.bind_label(l_int_loop).expect("bind");
        self.emit.emit_word(
            aarch64::encode_sub_imm(aarch64::Width::W32, Reg::X1, Reg::X8, 0x30, false)
                .expect("sub"),
        );
        self.emit.emit_word(
            aarch64::encode_cmp_imm(aarch64::Width::W32, Reg::X1, 9, false).expect("cmp"),
        );
        self.emit
            .emit_b_cond_label(aarch64::Condition::Hi, l_int_done)
            .expect("b.hi");
        self.emit.emit_word(
            aarch64::encode_orr_imm(aarch64::Width::X64, Reg::X13, Reg::X13, 2).expect("orr"),
        );
        self.emit.emit_word(
            aarch64::encode_cmp_imm(aarch64::Width::W32, Reg::X10, 19, false).expect("cmp"),
        );
        self.emit
            .emit_b_cond_label(aarch64::Condition::Hs, l_int_ovf)
            .expect("b.hs");
        self.emit
            .emit_word(aarch64::encode_movz(aarch64::Width::X64, Reg::X2, 10, 0).expect("mov"));
        self.emit.emit_word(
            aarch64::encode_madd(aarch64::Width::X64, Reg::X9, Reg::X9, Reg::X2, Reg::X1)
                .expect("madd"),
        );
        self.emit.emit_word(
            aarch64::encode_add_imm(aarch64::Width::X64, Reg::X10, Reg::X10, 1, false)
                .expect("add"),
        );
        self.emit.emit_word(
            aarch64::encode_add_imm(aarch64::Width::X64, Reg::X19, Reg::X19, 1, false)
                .expect("add"),
        );
        self.emit.emit_word(
            aarch64::encode_cmp_reg(aarch64::Width::X64, Reg::X19, Reg::X20).expect("cmp"),
        );
        self.emit
            .emit_b_cond_label(aarch64::Condition::Hs, l_int_done)
            .expect("b.hs");
        self.emit
            .emit_word(aarch64::encode_ldrb_imm(Reg::X8, Reg::X19, 0).expect("ldrb"));
        self.emit.emit_b_label(l_int_loop).expect("b");

        self.emit.bind_label(l_int_ovf).expect("bind");
        self.emit.emit_word(
            aarch64::encode_add_imm(aarch64::Width::X64, Reg::X12, Reg::X12, 1, false)
                .expect("add"),
        );
        self.emit.emit_word(
            aarch64::encode_add_imm(aarch64::Width::X64, Reg::X19, Reg::X19, 1, false)
                .expect("add"),
        );
        self.emit.emit_word(
            aarch64::encode_cmp_reg(aarch64::Width::X64, Reg::X19, Reg::X20).expect("cmp"),
        );
        self.emit
            .emit_b_cond_label(aarch64::Condition::Hs, l_int_ovf_end)
            .expect("b.hs");
        self.emit
            .emit_word(aarch64::encode_ldrb_imm(Reg::X8, Reg::X19, 0).expect("ldrb"));
        self.emit.emit_word(
            aarch64::encode_sub_imm(aarch64::Width::W32, Reg::X1, Reg::X8, 0x30, false)
                .expect("sub"),
        );
        self.emit.emit_word(
            aarch64::encode_cmp_imm(aarch64::Width::W32, Reg::X1, 9, false).expect("cmp"),
        );
        self.emit
            .emit_b_cond_label(aarch64::Condition::Ls, l_int_ovf)
            .expect("b.ls");
        self.emit.emit_b_label(l_int_ovf_end).expect("b");
        self.emit.bind_label(l_int_ovf_end).expect("bind");
        self.emit.bind_label(l_int_done).expect("bind");

        // ── Decimal point ──
        self.emit.emit_word(
            aarch64::encode_cmp_reg(aarch64::Width::X64, Reg::X19, Reg::X20).expect("cmp"),
        );
        self.emit
            .emit_b_cond_label(aarch64::Condition::Hs, l_no_dot)
            .expect("b.hs");
        self.emit.emit_word(
            aarch64::encode_cmp_imm(aarch64::Width::W32, Reg::X8, 0x2e, false).expect("cmp"),
        );
        self.emit
            .emit_b_cond_label(aarch64::Condition::Ne, l_no_dot)
            .expect("b.ne");
        self.emit.emit_word(
            aarch64::encode_orr_imm(aarch64::Width::X64, Reg::X13, Reg::X13, 1).expect("orr"),
        );
        self.emit.emit_word(
            aarch64::encode_add_imm(aarch64::Width::X64, Reg::X19, Reg::X19, 1, false)
                .expect("add"),
        );
        self.emit.emit_word(
            aarch64::encode_cmp_reg(aarch64::Width::X64, Reg::X19, Reg::X20).expect("cmp"),
        );
        self.emit
            .emit_b_cond_label(aarch64::Condition::Hs, l_no_dot)
            .expect("b.hs");
        self.emit
            .emit_word(aarch64::encode_ldrb_imm(Reg::X8, Reg::X19, 0).expect("ldrb"));

        self.emit
            .emit_cbnz_label(aarch64::Width::X64, Reg::X10, l_frac_lz_end)
            .expect("cbnz");
        self.emit.bind_label(l_frac_lz).expect("bind");
        self.emit.emit_word(
            aarch64::encode_cmp_imm(aarch64::Width::W32, Reg::X8, 0x30, false).expect("cmp"),
        );
        self.emit
            .emit_b_cond_label(aarch64::Condition::Ne, l_frac_lz_end)
            .expect("b.ne");
        self.emit.emit_word(
            aarch64::encode_orr_imm(aarch64::Width::X64, Reg::X13, Reg::X13, 2).expect("orr"),
        );
        self.emit.emit_word(
            aarch64::encode_add_imm(aarch64::Width::X64, Reg::X11, Reg::X11, 1, false)
                .expect("add"),
        );
        self.emit.emit_word(
            aarch64::encode_add_imm(aarch64::Width::X64, Reg::X19, Reg::X19, 1, false)
                .expect("add"),
        );
        self.emit.emit_word(
            aarch64::encode_cmp_reg(aarch64::Width::X64, Reg::X19, Reg::X20).expect("cmp"),
        );
        self.emit
            .emit_b_cond_label(aarch64::Condition::Hs, l_frac_lz_end)
            .expect("b.hs");
        self.emit
            .emit_word(aarch64::encode_ldrb_imm(Reg::X8, Reg::X19, 0).expect("ldrb"));
        self.emit.emit_b_label(l_frac_lz).expect("b");
        self.emit.bind_label(l_frac_lz_end).expect("bind");

        self.emit.bind_label(l_frac_loop).expect("bind");
        self.emit.emit_word(
            aarch64::encode_sub_imm(aarch64::Width::W32, Reg::X1, Reg::X8, 0x30, false)
                .expect("sub"),
        );
        self.emit.emit_word(
            aarch64::encode_cmp_imm(aarch64::Width::W32, Reg::X1, 9, false).expect("cmp"),
        );
        self.emit
            .emit_b_cond_label(aarch64::Condition::Hi, l_frac_done)
            .expect("b.hi");
        self.emit.emit_word(
            aarch64::encode_orr_imm(aarch64::Width::X64, Reg::X13, Reg::X13, 2).expect("orr"),
        );
        self.emit.emit_word(
            aarch64::encode_cmp_imm(aarch64::Width::W32, Reg::X10, 19, false).expect("cmp"),
        );
        self.emit
            .emit_b_cond_label(aarch64::Condition::Hs, l_frac_ovf)
            .expect("b.hs");
        self.emit
            .emit_word(aarch64::encode_movz(aarch64::Width::X64, Reg::X2, 10, 0).expect("mov"));
        self.emit.emit_word(
            aarch64::encode_madd(aarch64::Width::X64, Reg::X9, Reg::X9, Reg::X2, Reg::X1)
                .expect("madd"),
        );
        self.emit.emit_word(
            aarch64::encode_add_imm(aarch64::Width::X64, Reg::X10, Reg::X10, 1, false)
                .expect("add"),
        );
        self.emit.emit_word(
            aarch64::encode_add_imm(aarch64::Width::X64, Reg::X11, Reg::X11, 1, false)
                .expect("add"),
        );
        self.emit.emit_word(
            aarch64::encode_add_imm(aarch64::Width::X64, Reg::X19, Reg::X19, 1, false)
                .expect("add"),
        );
        self.emit.emit_word(
            aarch64::encode_cmp_reg(aarch64::Width::X64, Reg::X19, Reg::X20).expect("cmp"),
        );
        self.emit
            .emit_b_cond_label(aarch64::Condition::Hs, l_frac_done)
            .expect("b.hs");
        self.emit
            .emit_word(aarch64::encode_ldrb_imm(Reg::X8, Reg::X19, 0).expect("ldrb"));
        self.emit.emit_b_label(l_frac_loop).expect("b");

        self.emit.bind_label(l_frac_ovf).expect("bind");
        self.emit.emit_word(
            aarch64::encode_add_imm(aarch64::Width::X64, Reg::X19, Reg::X19, 1, false)
                .expect("add"),
        );
        self.emit.emit_word(
            aarch64::encode_cmp_reg(aarch64::Width::X64, Reg::X19, Reg::X20).expect("cmp"),
        );
        self.emit
            .emit_b_cond_label(aarch64::Condition::Hs, l_frac_ovf_end)
            .expect("b.hs");
        self.emit
            .emit_word(aarch64::encode_ldrb_imm(Reg::X8, Reg::X19, 0).expect("ldrb"));
        self.emit.emit_word(
            aarch64::encode_sub_imm(aarch64::Width::W32, Reg::X1, Reg::X8, 0x30, false)
                .expect("sub"),
        );
        self.emit.emit_word(
            aarch64::encode_cmp_imm(aarch64::Width::W32, Reg::X1, 9, false).expect("cmp"),
        );
        self.emit
            .emit_b_cond_label(aarch64::Condition::Ls, l_frac_ovf)
            .expect("b.ls");
        self.emit.emit_b_label(l_frac_ovf_end).expect("b");
        self.emit.bind_label(l_frac_ovf_end).expect("bind");
        self.emit.bind_label(l_frac_done).expect("bind");
        self.emit.emit_b_label(l_no_dot).expect("b");

        // ── Validation ──
        self.emit
            .emit_word(aarch64::encode_tst_imm(aarch64::Width::X64, Reg::X13, 2).expect("tst"));
        self.emit
            .emit_b_cond_label(aarch64::Condition::Eq, l_err_num)
            .expect("b.eq");

        // ── Exponent ──
        self.emit
            .emit_word(aarch64::encode_movz(aarch64::Width::X64, Reg::X14, 0, 0).expect("mov"));
        self.emit.emit_word(
            aarch64::encode_cmp_reg(aarch64::Width::X64, Reg::X19, Reg::X20).expect("cmp"),
        );
        self.emit
            .emit_b_cond_label(aarch64::Condition::Hs, l_no_exp)
            .expect("b.hs");
        self.emit.emit_word(
            aarch64::encode_cmp_imm(aarch64::Width::W32, Reg::X8, 0x65, false).expect("cmp"),
        );
        self.emit
            .emit_word(aarch64::encode_b_cond(aarch64::Condition::Eq, 2).expect("b.eq"));
        self.emit.emit_word(
            aarch64::encode_cmp_imm(aarch64::Width::W32, Reg::X8, 0x45, false).expect("cmp"),
        );
        self.emit
            .emit_b_cond_label(aarch64::Condition::Ne, l_no_exp)
            .expect("b.ne");
        self.emit.emit_word(
            aarch64::encode_add_imm(aarch64::Width::X64, Reg::X19, Reg::X19, 1, false)
                .expect("add"),
        );
        self.emit.emit_word(
            aarch64::encode_cmp_reg(aarch64::Width::X64, Reg::X19, Reg::X20).expect("cmp"),
        );
        self.emit
            .emit_b_cond_label(aarch64::Condition::Hs, l_err_num)
            .expect("b.hs");
        self.emit
            .emit_word(aarch64::encode_ldrb_imm(Reg::X8, Reg::X19, 0).expect("ldrb"));
        self.emit
            .emit_word(aarch64::encode_movz(aarch64::Width::X64, Reg::X15, 0, 0).expect("mov"));
        self.emit.emit_word(
            aarch64::encode_cmp_imm(aarch64::Width::W32, Reg::X8, 0x2d, false).expect("cmp"),
        );
        self.emit
            .emit_b_cond_label(aarch64::Condition::Ne, l_exp_pos)
            .expect("b.ne");
        self.emit
            .emit_word(aarch64::encode_movz(aarch64::Width::X64, Reg::X15, 1, 0).expect("mov"));
        self.emit.emit_word(
            aarch64::encode_add_imm(aarch64::Width::X64, Reg::X19, Reg::X19, 1, false)
                .expect("add"),
        );
        self.emit.emit_word(
            aarch64::encode_cmp_reg(aarch64::Width::X64, Reg::X19, Reg::X20).expect("cmp"),
        );
        self.emit
            .emit_b_cond_label(aarch64::Condition::Hs, l_err_num)
            .expect("b.hs");
        self.emit
            .emit_word(aarch64::encode_ldrb_imm(Reg::X8, Reg::X19, 0).expect("ldrb"));
        self.emit.emit_b_label(l_exp_loop).expect("b");

        self.emit.bind_label(l_exp_pos).expect("bind");
        self.emit.emit_word(
            aarch64::encode_cmp_imm(aarch64::Width::W32, Reg::X8, 0x2b, false).expect("cmp"),
        );
        self.emit
            .emit_b_cond_label(aarch64::Condition::Ne, l_exp_loop)
            .expect("b.ne");
        self.emit.emit_word(
            aarch64::encode_add_imm(aarch64::Width::X64, Reg::X19, Reg::X19, 1, false)
                .expect("add"),
        );
        self.emit.emit_word(
            aarch64::encode_cmp_reg(aarch64::Width::X64, Reg::X19, Reg::X20).expect("cmp"),
        );
        self.emit
            .emit_b_cond_label(aarch64::Condition::Hs, l_err_num)
            .expect("b.hs");
        self.emit
            .emit_word(aarch64::encode_ldrb_imm(Reg::X8, Reg::X19, 0).expect("ldrb"));

        // Validate first exponent digit
        self.emit.emit_word(
            aarch64::encode_sub_imm(aarch64::Width::W32, Reg::X1, Reg::X8, 0x30, false)
                .expect("sub"),
        );
        self.emit.emit_word(
            aarch64::encode_cmp_imm(aarch64::Width::W32, Reg::X1, 9, false).expect("cmp"),
        );
        self.emit
            .emit_b_cond_label(aarch64::Condition::Hi, l_err_num)
            .expect("b.hi");

        self.emit.bind_label(l_exp_loop).expect("bind");
        self.emit.emit_word(
            aarch64::encode_sub_imm(aarch64::Width::W32, Reg::X1, Reg::X8, 0x30, false)
                .expect("sub"),
        );
        self.emit.emit_word(
            aarch64::encode_cmp_imm(aarch64::Width::W32, Reg::X1, 9, false).expect("cmp"),
        );
        self.emit
            .emit_b_cond_label(aarch64::Condition::Hi, l_exp_done)
            .expect("b.hi");
        self.emit
            .emit_word(aarch64::encode_movz(aarch64::Width::X64, Reg::X2, 10, 0).expect("mov"));
        self.emit.emit_word(
            aarch64::encode_madd(aarch64::Width::X64, Reg::X14, Reg::X14, Reg::X2, Reg::X1)
                .expect("madd"),
        );
        self.emit
            .emit_word(aarch64::encode_movz(aarch64::Width::X64, Reg::X2, 9999, 0).expect("mov"));
        self.emit.emit_word(
            aarch64::encode_cmp_imm(aarch64::Width::W32, Reg::X14, 9999, false).expect("cmp"),
        );
        self.emit.emit_word(
            aarch64::encode_csel(
                aarch64::Width::X64,
                Reg::X14,
                Reg::X2,
                Reg::X14,
                aarch64::Condition::Hi,
            )
            .expect("csel"),
        );
        self.emit.emit_word(
            aarch64::encode_add_imm(aarch64::Width::X64, Reg::X19, Reg::X19, 1, false)
                .expect("add"),
        );
        self.emit.emit_word(
            aarch64::encode_cmp_reg(aarch64::Width::X64, Reg::X19, Reg::X20).expect("cmp"),
        );
        self.emit
            .emit_b_cond_label(aarch64::Condition::Hs, l_exp_done)
            .expect("b.hs");
        self.emit
            .emit_word(aarch64::encode_ldrb_imm(Reg::X8, Reg::X19, 0).expect("ldrb"));
        self.emit.emit_b_label(l_exp_loop).expect("b");

        self.emit.bind_label(l_exp_done).expect("bind");
        self.emit.emit_word(
            aarch64::encode_cmp_imm(aarch64::Width::X64, Reg::X15, 0, false).expect("cmp"),
        );
        self.emit
            .emit_word(aarch64::encode_b_cond(aarch64::Condition::Eq, 2).expect("b.eq"));
        self.emit
            .emit_word(aarch64::encode_neg(aarch64::Width::X64, Reg::X14, Reg::X14).expect("neg"));
        self.emit.bind_label(l_no_exp).expect("bind");

        // ── Compute p, dispatch ──
        self.emit.emit_word(
            aarch64::encode_add_reg(aarch64::Width::X64, Reg::X12, Reg::X14, Reg::X12)
                .expect("add"),
        ); // p = exp + dropped
        self.emit.emit_word(
            aarch64::encode_sub_reg(aarch64::Width::X64, Reg::X12, Reg::X12, Reg::X11)
                .expect("sub"),
        ); // p -= frac_digits

        self.emit.emit_word(
            aarch64::encode_cmp_imm(aarch64::Width::X64, Reg::X9, 0, false).expect("cmp"),
        );
        self.emit
            .emit_b_cond_label(aarch64::Condition::Eq, l_zero)
            .expect("b.eq");

        self.emit
            .emit_word(aarch64::encode_tst_imm(aarch64::Width::X64, Reg::X13, 1).expect("tst"));
        self.emit
            .emit_b_cond_label(aarch64::Condition::Ne, l_uscale)
            .expect("b.ne");
        self.emit.emit_word(
            aarch64::encode_cmp_imm(aarch64::Width::X64, Reg::X14, 0, false).expect("cmp"),
        );
        self.emit
            .emit_b_cond_label(aarch64::Condition::Ne, l_uscale)
            .expect("b.ne");
        self.emit
            .emit_word(aarch64::encode_movz(aarch64::Width::X64, Reg::X1, 1, 0).expect("mov"));
        self.emit.emit_word(
            aarch64::encode_lsl_imm(aarch64::Width::X64, Reg::X1, Reg::X1, 53).expect("lsl"),
        );
        self.emit.emit_word(
            aarch64::encode_cmp_reg(aarch64::Width::X64, Reg::X9, Reg::X1).expect("cmp"),
        );
        self.emit
            .emit_b_cond_label(aarch64::Condition::Hs, l_uscale)
            .expect("b.hs");

        self.emit.bind_label(l_exact_int).expect("bind");
        self.emit
            .emit_word(aarch64::encode_ucvtf_d_x(0, Reg::X9).expect("ucvtf"));
        self.emit
            .emit_word(aarch64::encode_fmov_x_d(Reg::X7, 0).expect("fmov"));
        self.emit.emit_b_label(l_apply_sign).expect("b");

        self.emit.bind_label(l_zero).expect("bind");
        self.emit
            .emit_word(aarch64::encode_movz(aarch64::Width::X64, Reg::X7, 0, 0).expect("mov"));
        self.emit.emit_b_label(l_apply_sign).expect("b");

        // ── uscale ──
        self.emit.bind_label(l_uscale).expect("bind");
        self.emit.emit_word(
            aarch64::encode_cmp_imm(aarch64::Width::W32, Reg::X12, 347, false).expect("cmp"),
        );
        self.emit
            .emit_b_cond_label(aarch64::Condition::Gt, l_pos_overflow)
            .expect("b.gt");
        self.emit.emit_word(
            aarch64::encode_cmn_imm(aarch64::Width::W32, Reg::X12, 348, false).expect("cmn"),
        );
        self.emit
            .emit_b_cond_label(aarch64::Condition::Lt, l_neg_underflow)
            .expect("b.lt");

        // lp = (p * 108853) >> 15  → x14
        self.emit.emit_word(
            aarch64::encode_movz(aarch64::Width::W32, Reg::X1, LOG2_10_LO as u16, 0).expect("movz"),
        );
        self.emit.emit_word(
            aarch64::encode_movk(aarch64::Width::W32, Reg::X1, LOG2_10_HI as u16, 16)
                .expect("movk"),
        );
        self.emit
            .emit_word(aarch64::encode_smull(Reg::X14, Reg::X12, Reg::X1).expect("smull"));
        self.emit.emit_word(
            aarch64::encode_asr_imm(aarch64::Width::X64, Reg::X14, Reg::X14, 15).expect("asr"),
        );

        // clz → x4
        self.emit
            .emit_word(aarch64::encode_clz(aarch64::Width::X64, Reg::X4, Reg::X9).expect("clz"));

        // e = min(1074, clz - 11 - lp) → x6
        self.emit.emit_word(
            aarch64::encode_sub_imm(aarch64::Width::X64, Reg::X6, Reg::X4, 11, false).expect("sub"),
        );
        self.emit.emit_word(
            aarch64::encode_sub_reg(aarch64::Width::X64, Reg::X6, Reg::X6, Reg::X14).expect("sub"),
        );
        self.emit
            .emit_word(aarch64::encode_movz(aarch64::Width::X64, Reg::X1, 1074, 0).expect("mov"));
        self.emit.emit_word(
            aarch64::encode_cmp_reg(aarch64::Width::X64, Reg::X6, Reg::X1).expect("cmp"),
        );
        self.emit.emit_word(
            aarch64::encode_csel(
                aarch64::Width::X64,
                Reg::X6,
                Reg::X1,
                Reg::X6,
                aarch64::Condition::Gt,
            )
            .expect("csel"),
        );

        // s = clz - e - lp - 3 → x3
        self.emit.emit_word(
            aarch64::encode_sub_reg(aarch64::Width::X64, Reg::X3, Reg::X4, Reg::X6).expect("sub"),
        );
        self.emit.emit_word(
            aarch64::encode_sub_reg(aarch64::Width::X64, Reg::X3, Reg::X3, Reg::X14).expect("sub"),
        );
        self.emit.emit_word(
            aarch64::encode_sub_imm(aarch64::Width::X64, Reg::X3, Reg::X3, 3, false).expect("sub"),
        );

        // left-justify: x5 = d << clz
        self.emit.emit_word(
            aarch64::encode_lsl_reg(aarch64::Width::X64, Reg::X5, Reg::X9, Reg::X4).expect("lsl"),
        );

        // Table lookup: index = p + 348
        self.emit.emit_word(
            aarch64::encode_add_imm(aarch64::Width::W32, Reg::X8, Reg::X12, 348, false)
                .expect("add"),
        );
        self.emit.emit_word(
            aarch64::encode_movz(aarch64::Width::X64, Reg::X1, tab_lo as u16, 0).expect("movz"),
        );
        self.emit.emit_word(
            aarch64::encode_movk(aarch64::Width::X64, Reg::X1, tab_hi16 as u16, 16).expect("movk"),
        );
        self.emit.emit_word(
            aarch64::encode_movk(aarch64::Width::X64, Reg::X1, tab_hi32 as u16, 32).expect("movk"),
        );
        self.emit.emit_word(
            aarch64::encode_movk(aarch64::Width::X64, Reg::X1, tab_hi48 as u16, 48).expect("movk"),
        );
        self.emit.emit_word(
            aarch64::encode_lsl_imm(aarch64::Width::X64, Reg::X8, Reg::X8, 4).expect("lsl"),
        );
        self.emit.emit_word(
            aarch64::encode_add_reg(aarch64::Width::X64, Reg::X1, Reg::X1, Reg::X8).expect("add"),
        );
        self.emit.emit_word(
            aarch64::encode_ldp(aarch64::Width::X64, Reg::X1, Reg::X2, Reg::X1, 0).expect("ldp"),
        );

        // mul128(x_left, pm_hi): hi=umulh, mid=mul
        self.emit
            .emit_word(aarch64::encode_umulh(Reg::X4, Reg::X5, Reg::X1).expect("umulh"));
        self.emit.emit_word(
            aarch64::encode_mul(aarch64::Width::X64, Reg::X8, Reg::X5, Reg::X1).expect("mul"),
        );

        // mask = (1 << (s & 63)) - 1
        self.emit
            .emit_word(aarch64::encode_movz(aarch64::Width::X64, Reg::X15, 63, 0).expect("movz"));
        self.emit.emit_word(
            aarch64::encode_and_reg(
                aarch64::Width::X64,
                Reg::X14,
                Reg::X3,
                Reg::X15,
                aarch64::Shift::Lsl,
                0,
            )
            .expect("and"),
        );
        self.emit
            .emit_word(aarch64::encode_movz(aarch64::Width::X64, Reg::X15, 1, 0).expect("movz"));
        self.emit.emit_word(
            aarch64::encode_lsl_reg(aarch64::Width::X64, Reg::X15, Reg::X15, Reg::X14)
                .expect("lsl"),
        );
        self.emit.emit_word(
            aarch64::encode_sub_imm(aarch64::Width::X64, Reg::X15, Reg::X15, 1, false)
                .expect("sub"),
        );

        // if hi & mask == 0: need second multiply
        self.emit.emit_word(
            aarch64::encode_and_reg(
                aarch64::Width::X64,
                Reg::X15,
                Reg::X4,
                Reg::X15,
                aarch64::Shift::Lsl,
                0,
            )
            .expect("and"),
        );
        self.emit.emit_word(
            aarch64::encode_cmp_imm(aarch64::Width::X64, Reg::X15, 0, false).expect("cmp"),
        );
        self.emit
            .emit_b_cond_label(aarch64::Condition::Eq, l_need_lo_mul)
            .expect("b.eq");
        self.emit
            .emit_word(aarch64::encode_movz(aarch64::Width::X64, Reg::X7, 1, 0).expect("mov"));
        self.emit.emit_b_label(l_after_lo_mul).expect("b");

        self.emit.bind_label(l_need_lo_mul).expect("bind");
        self.emit
            .emit_word(aarch64::encode_umulh(Reg::X14, Reg::X5, Reg::X2).expect("umulh"));
        // sticky = (mid - mid2 > 1) ? 1 : 0
        self.emit.emit_word(
            aarch64::encode_subs_reg(aarch64::Width::X64, Reg::X15, Reg::X8, Reg::X14)
                .expect("subs"),
        );
        self.emit.emit_word(
            aarch64::encode_cmp_imm(aarch64::Width::X64, Reg::X15, 1, false).expect("cmp"),
        );
        self.emit.emit_word(
            aarch64::encode_cset(aarch64::Width::X64, Reg::X7, aarch64::Condition::Hi)
                .expect("cset"),
        );
        // if mid < mid2: hi -= 1
        self.emit.emit_word(
            aarch64::encode_cmp_reg(aarch64::Width::X64, Reg::X8, Reg::X14).expect("cmp"),
        );
        self.emit
            .emit_word(aarch64::encode_b_cond(aarch64::Condition::Hs, 2).expect("b.hs"));
        self.emit.emit_word(
            aarch64::encode_sub_imm(aarch64::Width::X64, Reg::X4, Reg::X4, 1, false).expect("sub"),
        );

        self.emit.bind_label(l_after_lo_mul).expect("bind");
        // top = (s >= 64) ? 0 : (hi >> s)
        self.emit.emit_word(
            aarch64::encode_lsr_reg(aarch64::Width::X64, Reg::X14, Reg::X4, Reg::X3).expect("lsr"),
        );
        self.emit.emit_word(
            aarch64::encode_cmp_imm(aarch64::Width::X64, Reg::X3, 64, false).expect("cmp"),
        );
        self.emit.emit_word(
            aarch64::encode_csel(
                aarch64::Width::X64,
                Reg::X14,
                Reg::XZR,
                Reg::X14,
                aarch64::Condition::Hs,
            )
            .expect("csel"),
        );
        // u = top | sticky
        self.emit.emit_word(
            aarch64::encode_orr_reg(
                aarch64::Width::X64,
                Reg::X7,
                Reg::X14,
                Reg::X7,
                aarch64::Shift::Lsl,
                0,
            )
            .expect("orr"),
        );

        // ── Overflow check + round + pack ──
        // unmin(2^53) = (1<<55) - 2 = 0x007FFFFFFFFFFFFFFE
        self.emit.emit_word(
            aarch64::encode_movz(aarch64::Width::X64, Reg::X1, 0xFFFE, 0).expect("movz"),
        );
        self.emit.emit_word(
            aarch64::encode_movk(aarch64::Width::X64, Reg::X1, 0xFFFF, 16).expect("movk"),
        );
        self.emit.emit_word(
            aarch64::encode_movk(aarch64::Width::X64, Reg::X1, 0xFFFF, 32).expect("movk"),
        );
        self.emit.emit_word(
            aarch64::encode_movk(aarch64::Width::X64, Reg::X1, 0x007F, 48).expect("movk"),
        );
        self.emit.emit_word(
            aarch64::encode_cmp_reg(aarch64::Width::X64, Reg::X7, Reg::X1).expect("cmp"),
        );
        self.emit
            .emit_word(aarch64::encode_b_cond(aarch64::Condition::Lo, 5).expect("b.lo"));
        self.emit.emit_word(
            aarch64::encode_lsr_imm(aarch64::Width::X64, Reg::X14, Reg::X7, 1).expect("lsr"),
        );
        self.emit.emit_word(
            aarch64::encode_and_imm(aarch64::Width::X64, Reg::X15, Reg::X7, 1).expect("and"),
        );
        self.emit.emit_word(
            aarch64::encode_orr_reg(
                aarch64::Width::X64,
                Reg::X7,
                Reg::X14,
                Reg::X15,
                aarch64::Shift::Lsl,
                0,
            )
            .expect("orr"),
        );
        self.emit.emit_word(
            aarch64::encode_sub_imm(aarch64::Width::X64, Reg::X6, Reg::X6, 1, false).expect("sub"),
        );

        // Round: (u + 1 + ((u >> 2) & 1)) >> 2
        self.emit.emit_word(
            aarch64::encode_lsr_imm(aarch64::Width::X64, Reg::X1, Reg::X7, 2).expect("lsr"),
        );
        self.emit.emit_word(
            aarch64::encode_and_imm(aarch64::Width::X64, Reg::X1, Reg::X1, 1).expect("and"),
        );
        self.emit.emit_word(
            aarch64::encode_add_imm(aarch64::Width::X64, Reg::X7, Reg::X7, 1, false).expect("add"),
        );
        self.emit.emit_word(
            aarch64::encode_add_reg(aarch64::Width::X64, Reg::X7, Reg::X7, Reg::X1).expect("add"),
        );
        self.emit.emit_word(
            aarch64::encode_lsr_imm(aarch64::Width::X64, Reg::X7, Reg::X7, 2).expect("lsr"),
        );

        // Pack: check bit 52
        self.emit
            .emit_word(aarch64::encode_movz(aarch64::Width::X64, Reg::X1, 1, 0).expect("movz"));
        self.emit.emit_word(
            aarch64::encode_lsl_imm(aarch64::Width::X64, Reg::X1, Reg::X1, 52).expect("lsl"),
        );
        self.emit.emit_word(
            aarch64::encode_tst_imm(aarch64::Width::X64, Reg::X7, 1u64 << 52).expect("tst"),
        );
        self.emit
            .emit_b_cond_label(aarch64::Condition::Eq, l_apply_sign)
            .expect("b.eq");

        // Normal: biased = 1075 - e
        self.emit.bind_label(l_pack_normal).expect("bind");
        self.emit
            .emit_word(aarch64::encode_movz(aarch64::Width::X64, Reg::X1, 1075, 0).expect("mov"));
        self.emit.emit_word(
            aarch64::encode_sub_reg(aarch64::Width::X64, Reg::X1, Reg::X1, Reg::X6).expect("sub"),
        );
        self.emit.emit_word(
            aarch64::encode_cmp_imm(aarch64::Width::X64, Reg::X1, 2047, false).expect("cmp"),
        );
        self.emit
            .emit_b_cond_label(aarch64::Condition::Hs, l_pack_inf)
            .expect("b.hs");

        // bits = (x7 & ~(1<<52)) | (biased << 52)
        self.emit
            .emit_word(aarch64::encode_movz(aarch64::Width::X64, Reg::X8, 1, 0).expect("mov"));
        self.emit.emit_word(
            aarch64::encode_lsl_imm(aarch64::Width::X64, Reg::X8, Reg::X8, 52).expect("lsl"),
        );
        self.emit.emit_word(
            aarch64::encode_bic(
                aarch64::Width::X64,
                Reg::X7,
                Reg::X7,
                Reg::X8,
                aarch64::Shift::Lsl,
                0,
            )
            .expect("bic"),
        );
        self.emit.emit_word(
            aarch64::encode_lsl_imm(aarch64::Width::X64, Reg::X1, Reg::X1, 52).expect("lsl"),
        );
        self.emit.emit_word(
            aarch64::encode_orr_reg(
                aarch64::Width::X64,
                Reg::X7,
                Reg::X7,
                Reg::X1,
                aarch64::Shift::Lsl,
                0,
            )
            .expect("orr"),
        );
        self.emit.emit_b_label(l_apply_sign).expect("b");

        self.emit.bind_label(l_pack_inf).expect("bind");
        self.emit.emit_word(
            aarch64::encode_movz(aarch64::Width::X64, Reg::X7, 0x7FF0, 48).expect("movz"),
        );
        self.emit.emit_b_label(l_apply_sign).expect("b");

        // ── Apply sign + store + update cursor ──
        self.emit.bind_label(l_pos_overflow).expect("bind");
        self.emit.emit_word(
            aarch64::encode_movz(aarch64::Width::X64, Reg::X7, 0x7FF0, 48).expect("movz"),
        );
        self.emit.emit_b_label(l_apply_sign).expect("b");

        self.emit.bind_label(l_neg_underflow).expect("bind");
        self.emit
            .emit_word(aarch64::encode_movz(aarch64::Width::X64, Reg::X7, 0, 0).expect("movz"));
        // fall through to l_apply_sign

        self.emit.bind_label(l_apply_sign).expect("bind");
        self.emit
            .emit_cbz_label(aarch64::Width::X64, Reg::X0, l_done)
            .expect("cbz");
        self.emit
            .emit_word(aarch64::encode_movz(aarch64::Width::X64, Reg::X1, 1, 0).expect("movz"));
        self.emit.emit_word(
            aarch64::encode_lsl_imm(aarch64::Width::X64, Reg::X1, Reg::X1, 63).expect("lsl"),
        );
        self.emit.emit_word(
            aarch64::encode_orr_reg(
                aarch64::Width::X64,
                Reg::X7,
                Reg::X7,
                Reg::X1,
                aarch64::Shift::Lsl,
                0,
            )
            .expect("orr"),
        );
        self.emit.bind_label(l_done).expect("bind");
        self.emit.emit_word(
            aarch64::encode_str_imm(aarch64::Width::X64, Reg::X7, Reg::X21, offset).expect("str"),
        );
        self.emit.emit_word(
            aarch64::encode_str_imm(
                aarch64::Width::X64,
                Reg::X19,
                Reg::X22,
                CTX_INPUT_PTR as u32,
            )
            .expect("str"),
        );

        // ── Cold: error paths ──
        // Jump over cold paths from the hot path — but we use b =>l_apply_sign
        // to skip, so no skip branch needed here (cold paths are placed after l_done).
        self.emit.emit_b_label(l_skip_cold).expect("b");

        self.emit.bind_label(l_err_num).expect("bind");
        self.emit.emit_word(
            aarch64::encode_movz(aarch64::Width::W32, Reg::X8, error_code_invalid as u16, 0)
                .expect("movz"),
        );
        self.emit.emit_word(
            aarch64::encode_str_imm(
                aarch64::Width::W32,
                Reg::X8,
                Reg::X22,
                CTX_ERROR_CODE as u32,
            )
            .expect("str"),
        );
        self.emit.emit_b_label(error_exit).expect("b");

        self.emit.bind_label(l_err_eof).expect("bind");
        self.emit.emit_word(
            aarch64::encode_movz(aarch64::Width::W32, Reg::X8, error_code_eof as u16, 0)
                .expect("movz"),
        );
        self.emit.emit_word(
            aarch64::encode_str_imm(
                aarch64::Width::W32,
                Reg::X8,
                Reg::X22,
                CTX_ERROR_CODE as u32,
            )
            .expect("str"),
        );
        self.emit.emit_b_label(error_exit).expect("b");

        self.emit.bind_label(l_skip_cold).expect("bind");
    }

    /// Commit and finalize the assembler, returning the executable buffer.
    ///
    /// All functions must have been completed with `end_func` before calling this.
    pub fn finalize(self) -> aarch64::FinalizedEmission {
        self.emit.finalize().expect("failed to finalize assembly")
    }

    // =========================================================================
    // Encode-direction methods
    // =========================================================================
    //
    // Register assignments for encode (same registers, different semantics):
    //   x19 = cached output_ptr (write cursor)
    //   x20 = cached output_end
    //   x21 = input struct pointer (the value being serialized)
    //   x22 = EncodeContext pointer

    /// Emit an encode function prologue. Returns the entry offset and error_exit label.
    ///
    /// # Register assignments after prologue
    /// - x19 = cached output_ptr (from ctx.output_ptr)
    /// - x20 = cached output_end (from ctx.output_end)
    /// - x21 = input struct pointer (arg0)
    /// - x22 = EncodeContext pointer (arg1)
    pub fn begin_encode_func(&mut self) -> (u32, LabelId) {
        let error_exit = self.emit.new_label();
        let entry = self.emit.current_offset();
        let frame_size = self.frame_size;

        self.emit.emit_word(
            aarch64::encode_sub_imm(
                aarch64::Width::X64,
                Reg::SP,
                Reg::SP,
                frame_size as u16,
                false,
            )
            .expect("sub"),
        );
        self.emit.emit_word(
            aarch64::encode_stp(aarch64::Width::X64, Reg::X29, Reg::X30, Reg::SP, 0).expect("stp"),
        );
        self.emit.emit_word(
            aarch64::encode_stp(aarch64::Width::X64, Reg::X19, Reg::X20, Reg::SP, 16).expect("stp"),
        );
        self.emit.emit_word(
            aarch64::encode_stp(aarch64::Width::X64, Reg::X21, Reg::X22, Reg::SP, 32).expect("stp"),
        );
        self.emit.emit_word(
            aarch64::encode_add_imm(aarch64::Width::X64, Reg::X29, Reg::SP, 0, false).expect("add"),
        );
        self.emit.emit_word(
            aarch64::encode_mov_reg(aarch64::Width::X64, Reg::X21, Reg::X0).expect("mov"),
        );
        self.emit.emit_word(
            aarch64::encode_mov_reg(aarch64::Width::X64, Reg::X22, Reg::X1).expect("mov"),
        );
        self.emit.emit_word(
            aarch64::encode_ldr_imm(aarch64::Width::X64, Reg::X19, Reg::X22, ENC_OUTPUT_PTR)
                .expect("ldr"),
        );
        self.emit.emit_word(
            aarch64::encode_ldr_imm(aarch64::Width::X64, Reg::X20, Reg::X22, ENC_OUTPUT_END)
                .expect("ldr"),
        );

        self.error_exit = error_exit;
        (entry, error_exit)
    }

    /// Emit the success epilogue and error exit for an encode function.
    pub fn end_encode_func(&mut self, error_exit: LabelId) {
        let frame_size = self.frame_size;

        self.emit.emit_word(
            aarch64::encode_str_imm(aarch64::Width::X64, Reg::X19, Reg::X22, ENC_OUTPUT_PTR)
                .expect("str"),
        );
        self.emit.emit_word(
            aarch64::encode_ldp(aarch64::Width::X64, Reg::X21, Reg::X22, Reg::SP, 32).expect("ldp"),
        );
        self.emit.emit_word(
            aarch64::encode_ldp(aarch64::Width::X64, Reg::X19, Reg::X20, Reg::SP, 16).expect("ldp"),
        );
        self.emit.emit_word(
            aarch64::encode_ldp(aarch64::Width::X64, Reg::X29, Reg::X30, Reg::SP, 0).expect("ldp"),
        );
        self.emit.emit_word(
            aarch64::encode_add_imm(
                aarch64::Width::X64,
                Reg::SP,
                Reg::SP,
                frame_size as u16,
                false,
            )
            .expect("add"),
        );
        self.emit
            .emit_word(aarch64::encode_ret(Reg::X30).expect("ret"));

        self.emit.bind_label(error_exit).expect("bind");
        self.emit.emit_word(
            aarch64::encode_ldp(aarch64::Width::X64, Reg::X21, Reg::X22, Reg::SP, 32).expect("ldp"),
        );
        self.emit.emit_word(
            aarch64::encode_ldp(aarch64::Width::X64, Reg::X19, Reg::X20, Reg::SP, 16).expect("ldp"),
        );
        self.emit.emit_word(
            aarch64::encode_ldp(aarch64::Width::X64, Reg::X29, Reg::X30, Reg::SP, 0).expect("ldp"),
        );
        self.emit.emit_word(
            aarch64::encode_add_imm(
                aarch64::Width::X64,
                Reg::SP,
                Reg::SP,
                frame_size as u16,
                false,
            )
            .expect("add"),
        );
        self.emit
            .emit_word(aarch64::encode_ret(Reg::X30).expect("ret"));
    }

    /// Emit a call to another emitted encode function.
    ///
    /// Convention: x0 = input + field_offset, x1 = ctx.
    /// Flushes output cursor before call, reloads after, checks error.
    pub fn emit_enc_call_emitted_func(&mut self, label: LabelId, field_offset: u32) {
        self.emit_enc_flush_output_cursor();
        self.emit.emit_word(
            aarch64::encode_add_imm(
                aarch64::Width::X64,
                Reg::X0,
                Reg::X21,
                field_offset as u16,
                false,
            )
            .expect("add"),
        );
        self.emit.emit_word(
            aarch64::encode_mov_reg(aarch64::Width::X64, Reg::X1, Reg::X22).expect("mov"),
        );
        self.emit.emit_bl_label(label).expect("bl");
        self.emit_enc_reload_and_check_error();
    }

    /// Emit a call to an encode intrinsic: fn(ctx: *mut EncodeContext, ...).
    /// Flushes output cursor, calls, reloads output_ptr + output_end, checks error.
    pub fn emit_enc_call_intrinsic_ctx_only(&mut self, fn_ptr: *const u8) {
        self.emit_enc_flush_output_cursor();
        self.emit.emit_word(
            aarch64::encode_mov_reg(aarch64::Width::X64, Reg::X0, Reg::X22).expect("mov"),
        );
        self.emit_call_fn_ptr(fn_ptr);
        self.emit_enc_reload_and_check_error();
    }

    /// Emit a call to an encode intrinsic: fn(ctx, arg1).
    /// Flushes output, calls, reloads output_ptr + output_end, checks error.
    pub fn emit_enc_call_intrinsic(&mut self, fn_ptr: *const u8, arg1: u64) {
        self.emit_enc_flush_output_cursor();
        self.emit.emit_word(
            aarch64::encode_mov_reg(aarch64::Width::X64, Reg::X0, Reg::X22).expect("mov"),
        );
        self.emit_load_imm64(Reg::X1, arg1);
        self.emit_call_fn_ptr(fn_ptr);
        self.emit_enc_reload_and_check_error();
    }

    /// Emit a call to an encode intrinsic: fn(ctx, input + field_offset).
    /// Passes the input data pointer offset by `field_offset` as the second argument.
    /// Flushes output, calls, reloads output_ptr + output_end, checks error.
    pub fn emit_enc_call_intrinsic_with_input(&mut self, fn_ptr: *const u8, field_offset: u32) {
        self.emit_enc_flush_output_cursor();
        self.emit.emit_word(
            aarch64::encode_mov_reg(aarch64::Width::X64, Reg::X0, Reg::X22).expect("mov"),
        );
        self.emit.emit_word(
            aarch64::encode_add_imm(
                aarch64::Width::X64,
                Reg::X1,
                Reg::X21,
                field_offset as u16,
                false,
            )
            .expect("add"),
        );
        self.emit_call_fn_ptr(fn_ptr);
        self.emit_enc_reload_and_check_error();
    }

    /// Emit a bounds check: ensure at least `count` bytes available in output.
    /// If not enough space, calls kajit_output_grow intrinsic.
    pub fn emit_enc_ensure_capacity(&mut self, count: u32) {
        let have_space = self.emit.new_label();
        let grow_fn = crate::intrinsics::kajit_output_grow as *const u8;

        self.emit.emit_word(
            aarch64::encode_sub_reg(aarch64::Width::X64, Reg::X9, Reg::X20, Reg::X19).expect("sub"),
        );
        self.emit.emit_word(
            aarch64::encode_cmp_imm(aarch64::Width::X64, Reg::X9, count as u16, false)
                .expect("cmp"),
        );
        self.emit
            .emit_b_cond_label(aarch64::Condition::Hs, have_space)
            .expect("b.hs");
        self.emit_enc_flush_output_cursor();
        self.emit.emit_word(
            aarch64::encode_mov_reg(aarch64::Width::X64, Reg::X0, Reg::X22).expect("mov"),
        );
        self.emit.emit_word(
            aarch64::encode_movz(aarch64::Width::X64, Reg::X1, count as u16, 0).expect("movz"),
        );
        self.emit_call_fn_ptr(grow_fn);
        self.emit_enc_reload_and_check_error();
        self.emit.bind_label(have_space).expect("bind");
    }

    /// Write a single immediate byte to the output buffer and advance.
    /// Caller must ensure capacity first.
    pub fn emit_enc_write_byte(&mut self, value: u8) {
        self.emit.emit_word(
            aarch64::encode_movz(aarch64::Width::W32, Reg::X9, value as u16, 0).expect("movz"),
        );
        self.emit
            .emit_word(aarch64::encode_strb_imm(Reg::X9, Reg::X19, 0).expect("strb"));
        self.emit.emit_word(
            aarch64::encode_add_imm(aarch64::Width::X64, Reg::X19, Reg::X19, 1, false)
                .expect("add"),
        );
    }

    /// Write `count` bytes from a static pointer to the output buffer and advance.
    /// Caller must ensure capacity first. Uses a byte-by-byte loop for small counts,
    /// or memcpy intrinsic for larger ones.
    pub fn emit_enc_write_static_bytes(&mut self, ptr: *const u8, count: u32) {
        if count == 0 {
            return;
        }
        // For small counts (up to 8), inline the copy.
        // For larger, we'd use a memcpy intrinsic — but that's a future optimization.
        // For now, load pointer into x9 and copy byte-by-byte.
        let ptr_val = ptr as u64;
        self.emit_load_imm64(Reg::X9, ptr_val);
        // Copy count bytes from [x9] to [x19]
        for i in 0..count {
            self.emit
                .emit_word(aarch64::encode_ldrb_imm(Reg::X10, Reg::X9, i).expect("ldrb"));
            self.emit
                .emit_word(aarch64::encode_strb_imm(Reg::X10, Reg::X19, i).expect("strb"));
        }
        self.emit.emit_word(
            aarch64::encode_add_imm(aarch64::Width::X64, Reg::X19, Reg::X19, count as u16, false)
                .expect("add"),
        );
    }

    /// Advance the output cursor by `count` bytes.
    pub fn emit_enc_advance_output(&mut self, count: u32) {
        if count > 0 {
            self.emit.emit_word(
                aarch64::encode_add_imm(
                    aarch64::Width::X64,
                    Reg::X19,
                    Reg::X19,
                    count as u16,
                    false,
                )
                .expect("add"),
            );
        }
    }

    /// Load a value from the input struct at `offset` into a scratch register.
    /// Returns which scratch register was used (w9/x9).
    pub fn emit_enc_load_from_input(&mut self, offset: u32, width: Width) {
        match width {
            Width::W1 => self
                .emit
                .emit_word(aarch64::encode_ldrb_imm(Reg::X9, Reg::X21, offset).expect("ldrb")),
            Width::W2 => self
                .emit
                .emit_word(aarch64::encode_ldrh_imm(Reg::X9, Reg::X21, offset).expect("ldrh")),
            Width::W4 => self.emit.emit_word(
                aarch64::encode_ldr_imm(aarch64::Width::W32, Reg::X9, Reg::X21, offset)
                    .expect("ldr"),
            ),
            Width::W8 => self.emit.emit_word(
                aarch64::encode_ldr_imm(aarch64::Width::X64, Reg::X9, Reg::X21, offset)
                    .expect("ldr"),
            ),
        }
    }

    /// Store a value from scratch w9/x9 to the output buffer and advance.
    /// Caller must ensure capacity first.
    pub fn emit_enc_store_to_output(&mut self, width: Width) {
        match width {
            Width::W1 => {
                self.emit
                    .emit_word(aarch64::encode_strb_imm(Reg::X9, Reg::X19, 0).expect("strb"));
                self.emit.emit_word(
                    aarch64::encode_add_imm(aarch64::Width::X64, Reg::X19, Reg::X19, 1, false)
                        .expect("add"),
                );
            }
            Width::W2 => {
                self.emit
                    .emit_word(aarch64::encode_strh_imm(Reg::X9, Reg::X19, 0).expect("strh"));
                self.emit.emit_word(
                    aarch64::encode_add_imm(aarch64::Width::X64, Reg::X19, Reg::X19, 2, false)
                        .expect("add"),
                );
            }
            Width::W4 => {
                self.emit.emit_word(
                    aarch64::encode_str_imm(aarch64::Width::W32, Reg::X9, Reg::X19, 0)
                        .expect("str"),
                );
                self.emit.emit_word(
                    aarch64::encode_add_imm(aarch64::Width::X64, Reg::X19, Reg::X19, 4, false)
                        .expect("add"),
                );
            }
            Width::W8 => {
                self.emit.emit_word(
                    aarch64::encode_str_imm(aarch64::Width::X64, Reg::X9, Reg::X19, 0)
                        .expect("str"),
                );
                self.emit.emit_word(
                    aarch64::encode_add_imm(aarch64::Width::X64, Reg::X19, Reg::X19, 8, false)
                        .expect("add"),
                );
            }
        }
    }
}
