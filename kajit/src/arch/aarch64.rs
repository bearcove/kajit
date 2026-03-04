use dynasmrt::{DynasmApi, DynasmLabelApi, dynasm};
use kajit_emit::aarch64::{self, Emitter, LabelId, Reg};

use crate::context::{
    CTX_ERROR_CODE, CTX_INPUT_END, CTX_INPUT_PTR, ENC_ERROR_CODE, ENC_OUTPUT_END, ENC_OUTPUT_PTR,
    ErrorCode,
};
use crate::jit_f64;
use crate::recipe::{ErrorTarget, Op, Recipe, Slot, Width};

/// Load a 64-bit immediate into an aarch64 register using movz + 3×movk.
macro_rules! load_imm64 {
    ($ops:expr, $reg:tt, $val:expr) => {{
        let val: u64 = $val;
        dynasm!($ops
            ; .arch aarch64
            ; movz $reg, #((val) & 0xFFFF) as u32
            ; movk $reg, #((val >> 16) & 0xFFFF) as u32, LSL #16
            ; movk $reg, #((val >> 32) & 0xFFFF) as u32, LSL #32
            ; movk $reg, #((val >> 48) & 0xFFFF) as u32, LSL #48
        );
    }};
}

/// Base frame size: 3 pairs of callee-saved registers = 48 bytes.
pub const BASE_FRAME: u32 = 48;
/// Maximum base frame size for regalloc-aware lowering when saving x23..x28.
pub const REGALLOC_BASE_FRAME: u32 = 96;

/// Emission context — wraps the assembler plus bookkeeping labels.
pub struct EmitCtx {
    pub ops: dynasmrt::aarch64::Assembler,
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
        let ops = dynasmrt::aarch64::Assembler::new().expect("failed to create assembler");
        let mut emit = Emitter::new();
        let error_exit = emit.new_label();
        let entry = 0u32;

        EmitCtx {
            ops,
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
        let val = ptr as u64;
        self.emit.emit_word(aarch64::encode_movz(aarch64::Width::X64, Reg::X8, (val & 0xFFFF) as u16, 0).expect("movz"));
        self.emit.emit_word(aarch64::encode_movk(aarch64::Width::X64, Reg::X8, ((val >> 16) & 0xFFFF) as u16, 16).expect("movk"));
        self.emit.emit_word(aarch64::encode_movk(aarch64::Width::X64, Reg::X8, ((val >> 32) & 0xFFFF) as u16, 32).expect("movk"));
        self.emit.emit_word(aarch64::encode_movk(aarch64::Width::X64, Reg::X8, ((val >> 48) & 0xFFFF) as u16, 48).expect("movk"));
        self.emit.emit_word(aarch64::encode_blr(Reg::X8).expect("blr"));
    }

    /// Flush the cached input cursor (x19) back to ctx.input_ptr.
    fn emit_flush_input_cursor(&mut self) {
        self
            .emit
            .emit_word(aarch64::encode_str_imm(
                aarch64::Width::X64,
                Reg::X19,
                Reg::X22,
                CTX_INPUT_PTR as u32,
            ).expect("str"));
    }

    /// Reload the cached input cursor from ctx and check the error flag.
    /// Branches to `error_exit` if `ctx.error.code != 0`.
    fn emit_reload_cursor_and_check_error(&mut self) {
        let error_exit = self.error_exit;
        self.emit.emit_word(aarch64::encode_ldr_imm(
            aarch64::Width::X64,
            Reg::X19,
            Reg::X22,
            CTX_INPUT_PTR as u32,
        ).expect("ldr"));
        self.emit.emit_word(aarch64::encode_ldr_imm(
            aarch64::Width::W32,
            Reg::X9,
            Reg::X22,
            CTX_ERROR_CODE as u32,
        ).expect("ldr"));
        self.emit.emit_cbnz_label(aarch64::Width::W32, Reg::X9, error_exit).expect("cbnz");
    }

    /// Check the error flag without reloading the cursor.
    /// Branches to `error_exit` if `ctx.error.code != 0`.
    fn emit_check_error(&mut self) {
        let error_exit = self.error_exit;
        self.emit.emit_word(aarch64::encode_ldr_imm(
            aarch64::Width::W32,
            Reg::X9,
            Reg::X22,
            CTX_ERROR_CODE as u32,
        ).expect("ldr"));
        self.emit.emit_cbnz_label(aarch64::Width::W32, Reg::X9, error_exit).expect("cbnz");
    }

    /// Flush the cached output cursor (x19) back to ctx for encoding.
    fn emit_enc_flush_output_cursor(&mut self) {
        self
            .emit
            .emit_word(aarch64::encode_str_imm(
                aarch64::Width::X64,
                Reg::X19,
                Reg::X22,
                ENC_OUTPUT_PTR as u32,
            ).expect("str"));
    }

    /// Reload output_ptr and output_end from ctx and check the error flag.
    fn emit_enc_reload_and_check_error(&mut self) {
        let error_exit = self.error_exit;
        self.emit.emit_word(aarch64::encode_ldr_imm(
            aarch64::Width::X64,
            Reg::X19,
            Reg::X22,
            ENC_OUTPUT_PTR as u32,
        ).expect("ldr"));
        self.emit.emit_word(aarch64::encode_ldr_imm(
            aarch64::Width::X64,
            Reg::X20,
            Reg::X22,
            ENC_OUTPUT_END as u32,
        ).expect("ldr"));
        self.emit.emit_word(aarch64::encode_ldr_imm(
            aarch64::Width::W32,
            Reg::X9,
            Reg::X22,
            ENC_ERROR_CODE as u32,
        ).expect("ldr"));
        self.emit.emit_cbnz_label(aarch64::Width::W32, Reg::X9, error_exit).expect("cbnz");
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

        self.emit.emit_word(
            aarch64::encode_sub_imm(aarch64::Width::X64, Reg::SP, Reg::SP, frame_size as u16, false)
                .expect("sub"),
        );
        self.emit.emit_word(
            aarch64::encode_stp(
                aarch64::Width::X64,
                Reg::X29,
                Reg::X30,
                Reg::SP,
                0,
            )
            .expect("stp"),
        );
        self.emit.emit_word(
            aarch64::encode_stp(aarch64::Width::X64, Reg::X19, Reg::X20, Reg::SP, 16).expect(
                "stp",
            ),
        );
        self.emit.emit_word(
            aarch64::encode_stp(aarch64::Width::X64, Reg::X21, Reg::X22, Reg::SP, 32).expect(
                "stp",
            ),
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
            aarch64::encode_add_imm(aarch64::Width::X64, Reg::X29, Reg::SP, 0, false).expect(
                "add",
            ),
        );
        self.emit
            .emit_word(aarch64::encode_mov_reg(aarch64::Width::X64, Reg::X21, Reg::X0).expect("mov"));
        self.emit
            .emit_word(aarch64::encode_mov_reg(aarch64::Width::X64, Reg::X22, Reg::X1).expect("mov"));
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
            aarch64::encode_ldp(aarch64::Width::X64, Reg::X21, Reg::X22, Reg::SP, 32)
                .expect("ldp"),
        );
        self.emit.emit_word(
            aarch64::encode_ldp(aarch64::Width::X64, Reg::X19, Reg::X20, Reg::SP, 16)
                .expect("ldp"),
        );
        self.emit.emit_word(
            aarch64::encode_ldp(aarch64::Width::X64, Reg::X29, Reg::X30, Reg::SP, 0)
                .expect("ldp"),
        );
        self.emit.emit_word(
            aarch64::encode_add_imm(aarch64::Width::X64, Reg::SP, Reg::SP, frame_size as u16, false)
                .expect("add"),
        );
        self.emit.emit_word(aarch64::encode_ret(Reg::X30).expect("ret"));
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
            aarch64::encode_ldp(aarch64::Width::X64, Reg::X21, Reg::X22, Reg::SP, 32)
                .expect("ldp"),
        );
        self.emit.emit_word(
            aarch64::encode_ldp(aarch64::Width::X64, Reg::X19, Reg::X20, Reg::SP, 16)
                .expect("ldp"),
        );
        self.emit.emit_word(
            aarch64::encode_ldp(aarch64::Width::X64, Reg::X29, Reg::X30, Reg::SP, 0)
                .expect("ldp"),
        );
        self.emit.emit_word(
            aarch64::encode_add_imm(aarch64::Width::X64, Reg::SP, Reg::SP, frame_size as u16, false)
                .expect("add"),
        );
        self.emit.emit_word(aarch64::encode_ret(Reg::X30).expect("ret"));
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
        self.emit
            .emit_word(aarch64::encode_mov_reg(aarch64::Width::X64, Reg::X1, Reg::X22).expect("mov"));
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
        self.emit.emit_word(aarch64::encode_mov_reg(aarch64::Width::X64, Reg::X0, Reg::X22).expect("mov"));
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
        self.emit.emit_word(aarch64::encode_mov_reg(aarch64::Width::X64, Reg::X0, Reg::X22).expect("mov"));
        self.emit_call_fn_ptr(fn_ptr);
        self.emit_reload_cursor_and_check_error();
    }

    /// Emit a call to an intrinsic that takes (ctx, &mut stack_slot).
    /// x0 = ctx, x1 = sp + sp_offset. Flushes/reloads cursor, checks error.
    pub fn emit_call_intrinsic_ctx_and_stack_out(&mut self, fn_ptr: *const u8, sp_offset: u32) {
        self.emit_flush_input_cursor();
        self.emit.emit_word(aarch64::encode_mov_reg(aarch64::Width::X64, Reg::X0, Reg::X22).expect("mov"));
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
        self.emit.emit_word(aarch64::encode_mov_reg(aarch64::Width::X64, Reg::X0, Reg::X22).expect("mov"));
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
            aarch64::encode_ldr_imm(
                aarch64::Width::X64,
                Reg::X0,
                Reg::SP,
                arg0_sp_offset,
            )
            .expect("ldr"),
        );
        self.emit.emit_word(
            aarch64::encode_ldr_imm(
                aarch64::Width::X64,
                Reg::X1,
                Reg::SP,
                arg1_sp_offset,
            )
            .expect("ldr"),
        );
        let val = expected_ptr as u64;
        self.emit.emit_word(aarch64::encode_movz(aarch64::Width::X64, Reg::X2, (val & 0xFFFF) as u16, 0).expect("movz"));
        self.emit.emit_word(aarch64::encode_movk(aarch64::Width::X64, Reg::X2, ((val >> 16) & 0xFFFF) as u16, 16).expect("movk"));
        self.emit.emit_word(aarch64::encode_movk(aarch64::Width::X64, Reg::X2, ((val >> 32) & 0xFFFF) as u16, 32).expect("movk"));
        self.emit.emit_word(aarch64::encode_movk(aarch64::Width::X64, Reg::X2, ((val >> 48) & 0xFFFF) as u16, 48).expect("movk"));
        self.emit
            .emit_word(aarch64::encode_movz(aarch64::Width::X64, Reg::X3, expected_len as u16, 0).expect("movz"));
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

    /// Emit an unconditional branch to the given label.
    pub fn emit_branch(&mut self, label: LabelId) {
        self.emit.b_label(label).expect("emit_branch failed");
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
            aarch64::encode_str_imm(aarch64::Width::W32, Reg::X9, Reg::X22, CTX_ERROR_CODE as u32)
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
        self.emit
            .emit_word(aarch64::encode_sub_reg(aarch64::Width::X64, Reg::X10, Reg::X19, Reg::X9).expect("sub"));
        self.emit.emit_word(
            aarch64::encode_str_imm(aarch64::Width::X64, Reg::X10, Reg::SP, len_sp_offset)
                .expect("str"),
        );
        self.emit.emit_word(aarch64::encode_add_imm(aarch64::Width::X64, Reg::X19, Reg::X19, 1, false).expect("add"));
    }

    /// Emit `cbnz x0, label` — branch if x0 is nonzero.
    pub fn emit_cbnz_x0(&mut self, label: LabelId) {
        self.emit
            .emit_cbnz_label(aarch64::Width::X64, Reg::X0, label)
            .expect("emit_cbnz_label");
    }

    /// Zero a 64-bit stack slot at sp + offset.
    pub fn emit_zero_stack_slot(&mut self, sp_offset: u32) {
        self.emit.emit_word(aarch64::encode_str_imm(aarch64::Width::X64, Reg::XZR, Reg::SP, sp_offset).expect("str"));
    }

    /// Load a byte from sp + sp_offset, compare with byte_val, branch if equal.
    pub fn emit_stack_byte_cmp_branch(
        &mut self,
        sp_offset: u32,
        byte_val: u8,
        label: DynamicLabel,
    ) {
        let byte_val = byte_val as u32;
        dynasm!(self.ops
            ; .arch aarch64
            ; ldrb w9, [sp, #sp_offset]
            ; cmp w9, #byte_val
            ; b.eq =>label
        );
    }

    /// Set bit `bit_index` in a 64-bit stack slot at sp + sp_offset.
    pub fn emit_set_bit_on_stack(&mut self, sp_offset: u32, bit_index: u32) {
        let mask = 1u64 << bit_index;
        let mask_lo = (mask & 0xFFFF) as u32;
        let mask_hi = ((mask >> 16) & 0xFFFF) as u32;

        dynasm!(self.ops
            ; .arch aarch64
            ; ldr x9, [sp, #sp_offset]
        );

        if mask <= 0xFFFF {
            dynasm!(self.ops
                ; .arch aarch64
                ; movz x10, #mask_lo
            );
        } else {
            dynasm!(self.ops
                ; .arch aarch64
                ; movz x10, #mask_lo
                ; movk x10, #mask_hi, LSL #16
            );
        }

        dynasm!(self.ops
            ; .arch aarch64
            ; orr x9, x9, x10
            ; str x9, [sp, #sp_offset]
        );
    }

    /// Check that the 64-bit stack slot at sp + sp_offset equals expected_mask.
    /// If not, set MissingRequiredField error and branch to error_exit.
    pub fn emit_check_bitset(&mut self, sp_offset: u32, expected_mask: u64) {
        let error_exit = self.error_exit;
        let ok_label = self.ops.new_dynamic_label();
        let mask_lo = (expected_mask & 0xFFFF) as u32;
        let mask_hi = ((expected_mask >> 16) & 0xFFFF) as u32;
        let error_code = crate::context::ErrorCode::MissingRequiredField as u32;

        dynasm!(self.ops
            ; .arch aarch64
            ; ldr x9, [sp, #sp_offset]
        );

        if expected_mask <= 0xFFFF {
            dynasm!(self.ops
                ; .arch aarch64
                ; movz x10, #mask_lo
            );
        } else {
            dynasm!(self.ops
                ; .arch aarch64
                ; movz x10, #mask_lo
                ; movk x10, #mask_hi, LSL #16
            );
        }

        dynasm!(self.ops
            ; .arch aarch64
            // Check that (bitset & mask) == mask — all required bits are set.
            // Extra bits (from optional/default fields) are ignored.
            ; and x11, x9, x10
            ; cmp x11, x10
            ; b.eq =>ok_label
            // Not all required fields were seen — write error and bail
            ; movz w9, #error_code
            ; str w9, [x22, #CTX_ERROR_CODE]
            ; b =>error_exit
            ; =>ok_label
        );
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
        found_quote: DynamicLabel,
        found_escape: DynamicLabel,
        unterminated: DynamicLabel,
    ) {
        let vector_loop = self.ops.new_dynamic_label();
        let scalar_tail = self.ops.new_dynamic_label();
        let advance_16 = self.ops.new_dynamic_label();
        let find_pos = self.ops.new_dynamic_label();
        let found_at_offset = self.ops.new_dynamic_label();

        // Broadcast '"' (0x22) and '\' (0x5C) into v1.16b and v2.16b
        dynasm!(self.ops
            ; .arch aarch64
            ; movi v1.b16, 0x22
            ; movi v2.b16, 0x5C

            // ── Vector loop: 16 bytes per iteration ──
            ; =>vector_loop
            ; sub x9, x20, x19
            ; cmp x9, 16
            ; b.lo =>scalar_tail

            // Load 16 bytes
            ; ld1 {v0.b16}, [x19]
            // Compare for '"' and '\'
            ; cmeq v3.b16, v0.b16, v1.b16
            ; cmeq v4.b16, v0.b16, v2.b16
            ; orr v3.b16, v3.b16, v4.b16
            // Horizontal max: if any lane is 0xFF, b5 is 0xFF
            ; umaxv b5, v3.b16
            ; umov w9, v5.b[0]
            ; cbz w9, =>advance_16

            // Found a match in this 16-byte window — find exact position
            ; mov x10, 0
            ; =>find_pos
            ; ldrb w9, [x19, x10]
            ; cmp w9, 0x22
            ; b.eq =>found_at_offset
            ; cmp w9, 0x5C
            ; b.eq =>found_at_offset
            ; add x10, x10, 1
            ; b =>find_pos

            ; =>found_at_offset
            ; add x19, x19, x10
            ; cmp w9, 0x22
            ; b.eq =>found_quote
            ; b =>found_escape

            ; =>advance_16
            ; add x19, x19, 16
            ; b =>vector_loop

            // ── Scalar tail: remaining < 16 bytes ──
            ; =>scalar_tail
            ; cmp x19, x20
            ; b.hs =>unterminated
            ; ldrb w9, [x19]
            ; cmp w9, 0x22
            ; b.eq =>found_quote
            ; cmp w9, 0x5C
            ; b.eq =>found_escape
            ; add x19, x19, 1
            ; b =>scalar_tail
        );
    }

    /// Inline skip-whitespace: loop over space/tab/newline/cr, advancing x19.
    /// No function call, no ctx flush.
    pub fn emit_inline_skip_ws(&mut self) {
        let ws_loop = self.ops.new_dynamic_label();
        let ws_advance = self.ops.new_dynamic_label();
        let ws_done = self.ops.new_dynamic_label();

        dynasm!(self.ops
            ; .arch aarch64
            ; =>ws_loop
            ; cmp x19, x20
            ; b.hs =>ws_done
            ; ldrb w9, [x19]
            ; cmp w9, b' ' as u32
            ; b.eq =>ws_advance
            ; cmp w9, b'\n' as u32
            ; b.eq =>ws_advance
            ; cmp w9, b'\r' as u32
            ; b.eq =>ws_advance
            ; cmp w9, b'\t' as u32
            ; b.ne =>ws_done
            ; =>ws_advance
            ; add x19, x19, 1
            ; b =>ws_loop
            ; =>ws_done
        );
    }

    /// Inline comma-or-end-array: skip whitespace, then check for ',' or ']'.
    /// Writes 0 (comma) or 1 (']') to stack at sp_offset. Errors on anything else.
    pub fn emit_inline_comma_or_end_array(&mut self, sp_offset: u32) {
        let error_exit = self.error_exit;
        let got_comma = self.ops.new_dynamic_label();
        let got_end = self.ops.new_dynamic_label();
        let done = self.ops.new_dynamic_label();
        let error_code = ErrorCode::UnexpectedCharacter as u32;

        self.emit_inline_skip_ws();

        dynasm!(self.ops
            ; .arch aarch64
            // bounds check
            ; cmp x19, x20
            ; b.hs =>error_exit  // UnexpectedEof — reuse error_exit
            ; ldrb w9, [x19]
            ; add x19, x19, 1
            ; cmp w9, b',' as u32
            ; b.eq =>got_comma
            ; cmp w9, b']' as u32
            ; b.eq =>got_end
            // unexpected character
            ; movz w9, #error_code
            ; str w9, [x22, #CTX_ERROR_CODE]
            ; b =>error_exit
            ; =>got_comma
            ; strb wzr, [sp, #sp_offset]
            ; b =>done
            ; =>got_end
            ; movz w9, 1
            ; strb w9, [sp, #sp_offset]
            ; =>done
        );
    }

    /// Inline comma-or-end-object: skip whitespace, then check for ',' or '}'.
    /// Writes 0 (comma) or 1 ('}') to stack at sp_offset. Errors on anything else.
    pub fn emit_inline_comma_or_end_object(&mut self, sp_offset: u32) {
        let error_exit = self.error_exit;
        let got_comma = self.ops.new_dynamic_label();
        let got_end = self.ops.new_dynamic_label();
        let done = self.ops.new_dynamic_label();
        let error_code = ErrorCode::UnexpectedCharacter as u32;

        self.emit_inline_skip_ws();

        dynasm!(self.ops
            ; .arch aarch64
            ; cmp x19, x20
            ; b.hs =>error_exit
            ; ldrb w9, [x19]
            ; add x19, x19, 1
            ; cmp w9, b',' as u32
            ; b.eq =>got_comma
            ; cmp w9, b'}' as u32
            ; b.eq =>got_end
            ; movz w9, #error_code
            ; str w9, [x22, #CTX_ERROR_CODE]
            ; b =>error_exit
            ; =>got_comma
            ; strb wzr, [sp, #sp_offset]
            ; b =>done
            ; =>got_end
            ; movz w9, 1
            ; strb w9, [sp, #sp_offset]
            ; =>done
        );
    }

    /// Store the cached cursor (x19) to a stack slot.
    pub fn emit_save_cursor_to_stack(&mut self, sp_offset: u32) {
        dynasm!(self.ops
            ; .arch aarch64
            ; str x19, [sp, #sp_offset]
        );
    }

    /// Emit: skip whitespace, expect and consume `"`, branch to error_exit if not found.
    /// After this, x19 points just after the opening `"`.
    pub fn emit_json_expect_quote_after_ws(&mut self, _ws_intrinsic: *const u8) {
        let error_exit = self.error_exit;
        let not_quote = self.ops.new_dynamic_label();
        let ok = self.ops.new_dynamic_label();
        let error_code = ErrorCode::ExpectedStringKey as u32;

        self.emit_inline_skip_ws();

        // Check bounds + opening '"'
        dynasm!(self.ops
            ; .arch aarch64
            ; cmp x19, x20
            ; b.hs =>not_quote
            ; ldrb w9, [x19]
            ; cmp w9, 0x22
            ; b.ne =>not_quote
            ; add x19, x19, 1
            ; b =>ok

            // Error: set ExpectedStringKey and bail
            ; =>not_quote
            ; movz w9, #error_code
            ; str w9, [x22, #CTX_ERROR_CODE]
            ; b =>error_exit
            ; =>ok
        );
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
        dynasm!(self.ops
            ; .arch aarch64
            ; ldr x9, [sp, #start_sp_offset]
            ; sub x10, x19, x9
            ; add x11, x19, 1
            ; str x11, [x22, #CTX_INPUT_PTR]
            // Args: x0=ctx, x1=out+offset, x2=start, x3=len
            ; mov x0, x22
            ; add x1, x21, #field_offset
            ; mov x2, x9
            ; mov x3, x10
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
        dynasm!(self.ops
            ; .arch aarch64
            ; ldr x9, [sp, #start_sp_offset]
            ; sub x10, x19, x9
            // Args: x0=ctx, x1=out+offset, x2=start, x3=prefix_len
            ; mov x0, x22
            ; add x1, x21, #field_offset
            ; mov x2, x9
            ; mov x3, x10
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
        dynasm!(self.ops
            ; .arch aarch64
            ; ldr x9, [sp, #start_sp_offset]
            ; sub x10, x19, x9
            ; str x10, [sp, #len_save_sp_offset]
            // fn(ctx, data_ptr, data_len_u32)
            ; mov x0, x22
            ; mov x1, x9
            ; mov w2, w10
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

        dynasm!(self.ops
            ; .arch aarch64
            // Write ptr
            ; str x0, [x21, #ptr_off]
            // Write len and cap
            ; ldr x9, [sp, #len_sp_offset]
            ; str x9, [x21, #len_off]
            ; str x9, [x21, #cap_off]
            // Advance cursor past closing '"'
            ; add x19, x19, 1
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
        dynasm!(self.ops
            ; .arch aarch64
            ; ldr x9, [sp, #start_sp_offset]
            ; sub x10, x19, x9
            ; mov x0, x22
            ; mov x1, x9
            ; mov x2, x10
            ; add x3, sp, #key_ptr_sp_offset
            ; add x4, sp, #key_len_sp_offset
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
                dynasm!(self.ops ; .arch aarch64 ; movz w9, val32);
            } else {
                let lo = val32 & 0xFFFF;
                let hi = (val32 >> 16) & 0xFFFF;
                dynasm!(self.ops ; .arch aarch64
                    ; movz w9, lo
                    ; movk w9, hi, LSL #16
                );
            }
        } else {
            dynasm!(self.ops ; .arch aarch64
                ; movz x9, #((val) & 0xFFFF) as u32
                ; movk x9, #((val >> 16) & 0xFFFF) as u32, LSL #16
                ; movk x9, #((val >> 32) & 0xFFFF) as u32, LSL #32
                ; movk x9, #((val >> 48) & 0xFFFF) as u32, LSL #48
            );
        }
        // Store to [out + 0]
        match size {
            1 => dynasm!(self.ops ; .arch aarch64 ; strb w9, [x21]),
            2 => dynasm!(self.ops ; .arch aarch64 ; strh w9, [x21]),
            4 => dynasm!(self.ops ; .arch aarch64 ; str w9, [x21]),
            8 => dynasm!(self.ops ; .arch aarch64 ; str x9, [x21]),
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
        let eof_label = self.ops.new_dynamic_label();
        let slow_path = self.ops.new_dynamic_label();
        let done_label = self.ops.new_dynamic_label();

        dynasm!(self.ops
            ; .arch aarch64
            ; cmp x19, x20
            ; b.hs =>eof_label
            ; ldrb w9, [x19]
            ; tbnz w9, #7, =>slow_path
            ; add x19, x19, #1
            ; b =>done_label
            ; =>slow_path
        );
        // Slow path: call full varint decode intrinsic into temp on stack
        self.emit_flush_input_cursor();
        dynasm!(self.ops
            ; .arch aarch64
            ; mov x0, x22
            ; add x1, sp, #48
        );
        self.emit_call_fn_ptr(slow_intrinsic);
        self.emit_reload_cursor_and_check_error();
        dynasm!(self.ops
            ; .arch aarch64
            ; ldr w9, [sp, #48]
            ; b =>done_label

            ; =>eof_label
            ; movz w9, crate::context::ErrorCode::UnexpectedEof as u32
            ; str w9, [x22, #CTX_ERROR_CODE]
            ; b =>error_exit

            ; =>done_label
        );
    }

    /// Compare w9 (discriminant) with immediate `imm` and branch to `label`
    /// if equal.
    pub fn emit_cmp_imm_branch_eq(&mut self, imm: u32, label: DynamicLabel) {
        dynasm!(self.ops
            ; .arch aarch64
            ; cmp w9, #imm
            ; b.eq =>label
        );
    }

    /// Emit a branch-to-error for unknown variant (sets UnknownVariant error code).
    pub fn emit_unknown_variant_error(&mut self) {
        let error_exit = self.error_exit;
        let error_code = crate::context::ErrorCode::UnknownVariant as u32;

        dynasm!(self.ops
            ; .arch aarch64
            ; movz w9, #error_code
            ; str w9, [x22, #CTX_ERROR_CODE]
            ; b =>error_exit
        );
    }

    /// Save the cached input_ptr (x19) to a stack slot.
    pub fn emit_save_input_ptr(&mut self, stack_offset: u32) {
        dynasm!(self.ops
            ; .arch aarch64
            ; str x19, [sp, #stack_offset]
        );
    }

    /// Restore the cached input_ptr (x19) from a stack slot.
    pub fn emit_restore_input_ptr(&mut self, stack_offset: u32) {
        dynasm!(self.ops
            ; .arch aarch64
            ; ldr x19, [sp, #stack_offset]
        );
    }

    /// Store a 64-bit immediate into a stack slot at sp + offset.
    pub fn emit_store_imm64_to_stack(&mut self, stack_offset: u32, value: u64) {
        if value == 0 {
            dynasm!(self.ops
                ; .arch aarch64
                ; str xzr, [sp, #stack_offset]
            );
        } else {
            // Load immediate into x9, then store
            dynasm!(self.ops
                ; .arch aarch64
                ; movz x9, #((value) & 0xFFFF) as u32
            );
            if value > 0xFFFF {
                dynasm!(self.ops ; .arch aarch64
                    ; movk x9, #((value >> 16) & 0xFFFF) as u32, LSL #16
                );
            }
            if value > 0xFFFF_FFFF {
                dynasm!(self.ops ; .arch aarch64
                    ; movk x9, #((value >> 32) & 0xFFFF) as u32, LSL #32
                );
            }
            if value > 0xFFFF_FFFF_FFFF {
                dynasm!(self.ops ; .arch aarch64
                    ; movk x9, #((value >> 48) & 0xFFFF) as u32, LSL #48
                );
            }
            dynasm!(self.ops
                ; .arch aarch64
                ; str x9, [sp, #stack_offset]
            );
        }
    }

    /// AND a 64-bit immediate into a stack slot at sp + offset.
    /// Loads the slot, ANDs with the immediate, stores back.
    pub fn emit_and_imm64_on_stack(&mut self, stack_offset: u32, mask: u64) {
        dynasm!(self.ops
            ; .arch aarch64
            ; ldr x9, [sp, #stack_offset]
        );
        // Load mask into x10
        dynasm!(self.ops
            ; .arch aarch64
            ; movz x10, #((mask) & 0xFFFF) as u32
        );
        if mask > 0xFFFF {
            dynasm!(self.ops ; .arch aarch64
                ; movk x10, #((mask >> 16) & 0xFFFF) as u32, LSL #16
            );
        }
        if mask > 0xFFFF_FFFF {
            dynasm!(self.ops ; .arch aarch64
                ; movk x10, #((mask >> 32) & 0xFFFF) as u32, LSL #32
            );
        }
        if mask > 0xFFFF_FFFF_FFFF {
            dynasm!(self.ops ; .arch aarch64
                ; movk x10, #((mask >> 48) & 0xFFFF) as u32, LSL #48
            );
        }
        dynasm!(self.ops
            ; .arch aarch64
            ; and x9, x9, x10
            ; str x9, [sp, #stack_offset]
        );
    }

    /// Check if the stack slot at sp + offset has exactly one bit set (popcount == 1).
    /// If so, branch to `label`.
    pub fn emit_popcount_eq1_branch(&mut self, stack_offset: u32, label: DynamicLabel) {
        dynasm!(self.ops
            ; .arch aarch64
            ; ldr x9, [sp, #stack_offset]
            // popcount == 1 iff (x & (x-1)) == 0 && x != 0
            ; sub x10, x9, #1
            ; tst x9, x10           // sets Z if (x & (x-1)) == 0
            ; b.ne #12              // skip if more than 1 bit
            ; cbnz x9, =>label      // branch if nonzero (exactly 1 bit)
        );
    }

    /// Check if the stack slot at sp + offset is zero. If so, branch to `label`.
    pub fn emit_stack_zero_branch(&mut self, stack_offset: u32, label: DynamicLabel) {
        dynasm!(self.ops
            ; .arch aarch64
            ; ldr x9, [sp, #stack_offset]
            ; cbz x9, =>label
        );
    }

    /// Load the stack slot at sp + offset into x9, then branch to `label` if
    /// bit `bit_index` is set.
    pub fn emit_test_bit_branch(&mut self, stack_offset: u32, bit_index: u32, label: DynamicLabel) {
        dynasm!(self.ops
            ; .arch aarch64
            ; ldr x9, [sp, #stack_offset]
            ; tbnz x9, #bit_index, =>label
        );
    }

    /// Test a single bit at `bit_index` in the u64 at `[sp + stack_offset]`.
    /// Branch to `label` if the bit is CLEAR (zero) — i.e., the field was NOT seen.
    pub fn emit_test_bit_branch_zero(
        &mut self,
        stack_offset: u32,
        bit_index: u32,
        label: DynamicLabel,
    ) {
        dynasm!(self.ops
            ; .arch aarch64
            ; ldr x9, [sp, #stack_offset]
            ; tbz x9, #bit_index, =>label
        );
    }

    /// Emit an error (write error code to ctx, branch to error_exit).
    pub fn emit_error(&mut self, code: crate::context::ErrorCode) {
        let error_exit = self.error_exit;
        let error_code = code as u32;
        dynasm!(self.ops
            ; .arch aarch64
            ; movz w9, #error_code
            ; str w9, [x22, #CTX_ERROR_CODE]
            ; b =>error_exit
        );
    }

    /// Advance the cached cursor by n bytes (inline, no function call).
    pub fn emit_advance_cursor_by(&mut self, n: u32) {
        dynasm!(self.ops
            ; .arch aarch64
            ; add x19, x19, #n
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

        let ws_loop = self.ops.new_dynamic_label();
        let non_ws = self.ops.new_dynamic_label();
        let done = self.ops.new_dynamic_label();
        let err_lbl = self.ops.new_dynamic_label();

        dynasm!(self.ops ; .arch aarch64
            // ── whitespace-skip loop ────────────────────────────────────
            ; =>ws_loop
            ; cmp  x19, x20
            ; b.hs =>err_lbl          // EOF → error
            ; ldrb w10, [x19]         // raw byte → w10
            ; sub  w9, w10, #9        // adjusted: HT→0, LF→1, CR→4, SP→23
            ; cmp  w9, #23
            ; b.hi =>non_ws           // adjusted > 23 → definitely not WS
            // within range [0,23]: test bitmask bit at position w9
            ; mov  x8, #0x0013        // bits 0,1,4 = HT,LF,CR
            ; movk x8, #0x0080, lsl #16 // bit 23 = SP  → mask = 0x0080_0013
            ; lsr  x8, x8, x9         // x9 = w9 zero-extended; shift by adjusted value
            ; tbz  x8, #0, =>non_ws   // bit 0 clear → not WS
            ; add  x19, x19, #1       // it's WS → advance
            ; b    =>ws_loop

            // ── byte check ─────────────────────────────────────────────
            ; =>non_ws
            ; cmp  w10, #expected
            ; b.ne =>err_lbl
            ; add  x19, x19, #1       // consume expected byte
            ; b    =>done

            // ── error ───────────────────────────────────────────────────
            ; =>err_lbl
            ; mov  w9, #err_code
            ; str  w9, [x22, #CTX_ERROR_CODE]
            ; b    =>error_exit

            ; =>done
        );
    }

    // ── Option support ────────────────────────────────────────────────

    /// Save the current `out` pointer (x21) and redirect it to a stack scratch area.
    /// The save slot is at `scratch_offset - 8`.
    pub fn emit_redirect_out_to_stack(&mut self, scratch_offset: u32) {
        let save_slot = scratch_offset - 8;
        dynasm!(self.ops
            ; .arch aarch64
            ; str x21, [sp, #save_slot]
            ; add x21, sp, #scratch_offset
        );
    }

    /// Restore the `out` pointer (x21) from the saved slot.
    pub fn emit_restore_out(&mut self, scratch_offset: u32) {
        let save_slot = scratch_offset - 8;
        dynasm!(self.ops
            ; .arch aarch64
            ; ldr x21, [sp, #save_slot]
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
        load_imm64!(self.ops, x0, init_none_fn as u64);
        dynasm!(self.ops ; .arch aarch64 ; add x1, x21, #offset);
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
        load_imm64!(self.ops, x0, fn_ptr as u64);
        dynasm!(self.ops ; .arch aarch64 ; add x1, x21, #offset);
        load_imm64!(self.ops, x2, extra_ptr as u64);
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
        load_imm64!(self.ops, x0, init_some_fn as u64);
        dynasm!(self.ops
            ; .arch aarch64
            ; add x1, x21, #offset
            ; add x2, sp, #scratch_offset
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
        dynasm!(self.ops
            ; .arch aarch64
            ; mov x0, x22
            ; mov x1, x9
            ; movz x2, elem_size
            ; movz x3, elem_align
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
        dynasm!(self.ops
            ; .arch aarch64
            ; ldr x10, [sp, #cap_slot]
            ; lsl x10, x10, #1
            ; mov x0, x22
            ; ldr x1, [sp, #buf_slot]
            ; ldr x2, [sp, #len_slot]
            ; ldr x3, [sp, #cap_slot]
            ; mov x4, x10
            ; movz x5, elem_size
            ; movz x6, elem_align
        );
        self.emit_call_fn_ptr(grow_fn);
        self.emit_reload_cursor_and_check_error();
        dynasm!(self.ops
            ; .arch aarch64
            ; str x0, [sp, #buf_slot]
            ; ldr x10, [sp, #cap_slot]
            ; lsl x10, x10, #1
            ; str x10, [sp, #cap_slot]
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
        dynasm!(self.ops
            ; .arch aarch64
            ; mov x0, x22
            ; movz x1, count
            ; movz x2, elem_size
            ; movz x3, elem_align
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
        dynasm!(self.ops
            ; .arch aarch64
            ; str x21, [sp, #saved_out_slot]        // save out pointer
            ; str x0, [sp, #buf_slot]               // buf = alloc result
            ; str xzr, [sp, #len_slot]              // len = 0
            ; movz x10, initial_cap
            ; str x10, [sp, #cap_slot]              // cap = initial_cap
        );
    }

    /// Check ctx.error.code and branch to label if nonzero.
    pub fn emit_check_error_branch(&mut self, label: DynamicLabel) {
        dynasm!(self.ops
            ; .arch aarch64
            ; ldr w10, [x22, #CTX_ERROR_CODE]
            ; cbnz w10, =>label
        );
    }

    /// Save the count register (w9 on aarch64, r10d on x64) to a stack slot.
    ///
    /// Used to preserve the count across a function call (w9/r10 are caller-saved).
    pub fn emit_save_count_to_stack(&mut self, slot: u32) {
        dynasm!(self.ops
            ; .arch aarch64
            ; str x9, [sp, #slot]
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
        dynasm!(self.ops
            ; .arch aarch64
            ; str x21, [sp, #saved_out_slot]   // save out pointer
            ; str x0, [sp, #buf_slot]          // buf = alloc result
            // Save callee-saved x23/x24
            ; str x23, [sp, #save_x23_slot]
            ; str x24, [sp, #save_x24_slot]
            // x23 = cursor = buf
            ; mov x23, x0
            // x24 = end = buf + count * elem_size
            ; ldr x10, [sp, #count_slot]
            ; movz x11, elem_size
            ; madd x24, x10, x11, x0
        );
    }

    /// Set out = cursor (x21 = x23). Single register move, no memory access.
    pub fn emit_vec_loop_load_cursor(&mut self, _cursor_slot: u32) {
        dynasm!(self.ops
            ; .arch aarch64
            ; mov x21, x23
        );
    }

    /// Advance cursor register, check error, branch back if cursor < end.
    /// All register ops — no memory access in the hot loop.
    pub fn emit_vec_loop_advance_cursor(
        &mut self,
        _cursor_slot: u32,
        _end_slot: u32,
        elem_size: u32,
        loop_label: DynamicLabel,
        error_cleanup_label: DynamicLabel,
    ) {
        dynasm!(self.ops
            ; .arch aarch64
            // Check error from element deserialization
            ; ldr w10, [x22, #CTX_ERROR_CODE]
            ; cbnz w10, =>error_cleanup_label
            // cursor += elem_size (register only)
            ; add x23, x23, elem_size
            // Compare with end (register only)
            ; cmp x23, x24
            ; b.lo =>loop_label
        );
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
        loop_label: DynamicLabel,
    ) {
        dynasm!(self.ops
            ; .arch aarch64
            ; add x23, x23, elem_size
            ; cmp x23, x24
            ; b.lo =>loop_label
        );
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
        loop_label: DynamicLabel,
        done_label: DynamicLabel,
        error_cleanup: DynamicLabel,
    ) {
        let slow_path = self.ops.new_dynamic_label();
        let eof_label = self.ops.new_dynamic_label();

        // === Hot loop (all fast-path instructions) ===
        dynasm!(self.ops
            ; .arch aarch64
            // Bounds check
            ; cmp x19, x20
            ; b.hs =>eof_label
            // Load byte + advance input pointer (post-indexed)
            ; ldrb w9, [x19], #1
            // Test continuation bit
            ; tbnz w9, #7, =>slow_path
        );

        if zigzag {
            dynasm!(self.ops
                ; .arch aarch64
                ; lsr w10, w9, #1
                ; and w11, w9, #1
                ; neg w11, w11
                ; eor w9, w10, w11
            );
        }

        // Store directly to cursor (x23), no mov x21 needed
        match store_width {
            2 => dynasm!(self.ops ; .arch aarch64 ; strh w9, [x23]),
            4 => dynasm!(self.ops ; .arch aarch64 ; str w9, [x23]),
            8 => dynasm!(self.ops ; .arch aarch64 ; str x9, [x23]),
            _ => panic!("unsupported varint store width: {store_width}"),
        }

        // Advance cursor, loop back
        dynasm!(self.ops
            ; .arch aarch64
            ; add x23, x23, elem_size
            ; cmp x23, x24
            ; b.lo =>loop_label
        );
        // Fall through = loop done
        dynasm!(self.ops
            ; .arch aarch64
            ; b =>done_label
        );

        // === Slow path (out-of-line) ===
        dynasm!(self.ops
            ; .arch aarch64
            ; =>slow_path
            ; sub x19, x19, #1
        );
        self.emit_flush_input_cursor();
        dynasm!(self.ops
            ; .arch aarch64
            ; mov x0, x22
            ; mov x1, x23
        );
        self.emit_call_fn_ptr(intrinsic_fn_ptr);
        dynasm!(self.ops
            ; .arch aarch64
            ; ldr x19, [x22, #CTX_INPUT_PTR]
            ; ldr w9, [x22, #CTX_ERROR_CODE]
            ; cbnz w9, =>error_cleanup
            ; add x23, x23, elem_size
            ; cmp x23, x24
            ; b.lo =>loop_label
            ; b =>done_label
        );

        // === EOF (cold) ===
        dynasm!(self.ops
            ; .arch aarch64
            ; =>eof_label
            ; movz w9, crate::context::ErrorCode::UnexpectedEof as u32
            ; str w9, [x22, #CTX_ERROR_CODE]
            ; b =>error_cleanup
        );
    }

    /// Restore x23/x24 from stack. Must be called on every exit path from a Vec loop.
    pub fn emit_vec_restore_callee_saved(&mut self, save_x23_slot: u32, save_x24_slot: u32) {
        dynasm!(self.ops
            ; .arch aarch64
            ; ldr x23, [sp, #save_x23_slot]
            ; ldr x24, [sp, #save_x24_slot]
        );
    }

    /// Emit Vec loop header: compute slot = buf + i * elem_size, set out = slot.
    ///
    /// Used by JSON where buf can change on growth and index-based access is needed.
    pub fn emit_vec_loop_slot(&mut self, buf_slot: u32, counter_slot: u32, elem_size: u32) {
        dynasm!(self.ops
            ; .arch aarch64
            ; ldr x21, [sp, #buf_slot]         // buf
            ; ldr x10, [sp, #counter_slot]     // i
            ; movz x11, elem_size
            ; madd x21, x10, x11, x21          // out = buf + i * elem_size
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

        dynasm!(self.ops
            ; .arch aarch64
            // Restore out pointer
            ; ldr x21, [sp, #saved_out_slot]
            // Load values
            ; ldr x10, [sp, #buf_slot]     // ptr
            ; ldr x11, [sp, #len_slot]     // len
            ; ldr x12, [sp, #cap_slot]     // cap
            // Store at out + discovered offsets
            ; str x10, [x21, #ptr_off]
            ; str x11, [x21, #len_off]
            ; str x12, [x21, #cap_off]
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
        dynasm!(self.ops
            ; .arch aarch64
            ; movz x10, elem_align    // dangling pointer = alignment
            ; str x10, [x21, #ptr_off]
            ; str xzr, [x21, #len_off]
            ; str xzr, [x21, #cap_off]
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
        dynasm!(self.ops
            ; .arch aarch64
            ; ldr x21, [sp, #saved_out_slot]
            ; ldr x0, [sp, #buf_slot]
            ; ldr x1, [sp, #cap_slot]
            ; movz x2, elem_size
            ; movz x3, elem_align
        );
        self.emit_call_fn_ptr(free_fn);
        dynasm!(self.ops ; .arch aarch64 ; b =>error_exit);
    }

    /// Compare the count register (w9 on aarch64, r10d on x64) with zero
    /// and branch to label if equal.
    pub fn emit_cbz_count(&mut self, label: DynamicLabel) {
        dynasm!(self.ops
            ; .arch aarch64
            ; cbz w9, =>label
        );
    }

    /// Compare two stack slot values and branch if equal (len == cap for growth check).
    pub fn emit_cmp_stack_slots_branch_eq(
        &mut self,
        slot_a: u32,
        slot_b: u32,
        label: DynamicLabel,
    ) {
        dynasm!(self.ops
            ; .arch aarch64
            ; ldr x10, [sp, #slot_a]
            ; ldr x11, [sp, #slot_b]
            ; cmp x10, x11
            ; b.eq =>label
        );
    }

    /// Increment a stack slot value by 1.
    pub fn emit_inc_stack_slot(&mut self, slot: u32) {
        dynasm!(self.ops
            ; .arch aarch64
            ; ldr x10, [sp, #slot]
            ; add x10, x10, #1
            ; str x10, [sp, #slot]
        );
    }

    // ── Map support ─────────────────────────────────────────────────

    /// Advance the out register (x21) by a constant offset.
    ///
    /// Used in map loops to move from the key slot to the value slot within a pair.
    pub fn emit_advance_out_by(&mut self, offset: u32) {
        dynasm!(self.ops
            ; .arch aarch64
            ; add x21, x21, #offset
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
        load_imm64!(self.ops, x8, from_pair_slice_fn as u64);
        dynasm!(self.ops
            ; .arch aarch64
            ; mov x0, x8
            ; ldr x1, [sp, #saved_out_slot]
            ; ldr x2, [sp, #buf_slot]
            ; ldr x3, [sp, #count_slot]
        );
        self.emit_call_fn_ptr(trampoline);
    }

    /// Call `kajit_map_build(from_pair_slice_fn, x21, null, 0)` — empty map.
    ///
    /// Same trampoline pattern as `emit_call_map_from_pairs`.
    pub fn emit_call_map_from_pairs_empty(&mut self, from_pair_slice_fn: *const u8) {
        let trampoline = crate::intrinsics::kajit_map_build as *const u8;
        load_imm64!(self.ops, x8, from_pair_slice_fn as u64);
        dynasm!(self.ops
            ; .arch aarch64
            ; mov x0, x8
            ; mov x1, x21
            ; mov x2, xzr
            ; mov x3, xzr
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
        dynasm!(self.ops
            ; .arch aarch64
            ; ldr x0, [sp, #buf_slot]
            ; ldr x1, [sp, #cap_slot]
            ; movz x2, pair_stride
            ; movz x3, pair_align
        );
        self.emit_call_fn_ptr(free_fn);
    }

    // ── Recipe emission ─────────────────────────────────────────────

    /// Emit a recipe — interpret a sequence of micro-ops into aarch64 instructions.
    pub fn emit_recipe(&mut self, recipe: &Recipe) {
        let error_exit = self.error_exit;

        // Allocate dynamic labels for the recipe
        let labels: Vec<DynamicLabel> = (0..recipe.label_count)
            .map(|_| self.ops.new_dynamic_label())
            .collect();

        // Shared EOF error label (lazily bound on first use)
        let eof_label = self.ops.new_dynamic_label();

        for op in &recipe.ops {
            match op {
                Op::BoundsCheck { count } => {
                    if *count == 1 {
                        dynasm!(self.ops
                            ; .arch aarch64
                            ; cmp x19, x20
                            ; b.hs =>eof_label
                        );
                    } else {
                        let count = *count;
                        dynasm!(self.ops
                            ; .arch aarch64
                            ; sub x9, x20, x19
                            ; cmp x9, #count
                            ; b.lo =>eof_label
                        );
                    }
                }
                Op::LoadByte { dst } => match dst {
                    Slot::A => dynasm!(self.ops ; .arch aarch64 ; ldrb w9, [x19]),
                    Slot::B => dynasm!(self.ops ; .arch aarch64 ; ldrb w10, [x19]),
                },
                Op::LoadFromCursor { dst, width } => match (dst, width) {
                    (Slot::A, Width::W4) => dynasm!(self.ops ; .arch aarch64 ; ldr w9, [x19]),
                    (Slot::A, Width::W8) => dynasm!(self.ops ; .arch aarch64 ; ldr x9, [x19]),
                    (Slot::B, Width::W4) => dynasm!(self.ops ; .arch aarch64 ; ldr w10, [x19]),
                    (Slot::B, Width::W8) => dynasm!(self.ops ; .arch aarch64 ; ldr x10, [x19]),
                    _ => panic!("unsupported LoadFromCursor width"),
                },
                Op::StoreToOut { src, offset, width } => {
                    let offset = *offset;
                    match (src, width) {
                        (Slot::A, Width::W1) => {
                            dynasm!(self.ops ; .arch aarch64 ; strb w9, [x21, #offset])
                        }
                        (Slot::A, Width::W2) => {
                            dynasm!(self.ops ; .arch aarch64 ; strh w9, [x21, #offset])
                        }
                        (Slot::A, Width::W4) => {
                            dynasm!(self.ops ; .arch aarch64 ; str w9, [x21, #offset])
                        }
                        (Slot::A, Width::W8) => {
                            dynasm!(self.ops ; .arch aarch64 ; str x9, [x21, #offset])
                        }
                        (Slot::B, Width::W1) => {
                            dynasm!(self.ops ; .arch aarch64 ; strb w10, [x21, #offset])
                        }
                        (Slot::B, Width::W2) => {
                            dynasm!(self.ops ; .arch aarch64 ; strh w10, [x21, #offset])
                        }
                        (Slot::B, Width::W4) => {
                            dynasm!(self.ops ; .arch aarch64 ; str w10, [x21, #offset])
                        }
                        (Slot::B, Width::W8) => {
                            dynasm!(self.ops ; .arch aarch64 ; str x10, [x21, #offset])
                        }
                    }
                }
                Op::StoreByteToStack { src, sp_offset } => {
                    let sp_offset = *sp_offset;
                    match src {
                        Slot::A => dynasm!(self.ops ; .arch aarch64 ; strb w9, [sp, #sp_offset]),
                        Slot::B => dynasm!(self.ops ; .arch aarch64 ; strb w10, [sp, #sp_offset]),
                    }
                }
                Op::StoreToStack {
                    src,
                    sp_offset,
                    width,
                } => {
                    let sp_offset = *sp_offset;
                    match (src, width) {
                        (Slot::A, Width::W4) => {
                            dynasm!(self.ops ; .arch aarch64 ; str w9, [sp, #sp_offset])
                        }
                        (Slot::A, Width::W8) => {
                            dynasm!(self.ops ; .arch aarch64 ; str x9, [sp, #sp_offset])
                        }
                        (Slot::B, Width::W4) => {
                            dynasm!(self.ops ; .arch aarch64 ; str w10, [sp, #sp_offset])
                        }
                        (Slot::B, Width::W8) => {
                            dynasm!(self.ops ; .arch aarch64 ; str x10, [sp, #sp_offset])
                        }
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
                        (Slot::A, Width::W4) => {
                            dynasm!(self.ops ; .arch aarch64 ; ldr w9, [sp, #sp_offset])
                        }
                        (Slot::A, Width::W8) => {
                            dynasm!(self.ops ; .arch aarch64 ; ldr x9, [sp, #sp_offset])
                        }
                        (Slot::B, Width::W4) => {
                            dynasm!(self.ops ; .arch aarch64 ; ldr w10, [sp, #sp_offset])
                        }
                        (Slot::B, Width::W8) => {
                            dynasm!(self.ops ; .arch aarch64 ; ldr x10, [sp, #sp_offset])
                        }
                        _ => panic!("unsupported LoadFromStack width"),
                    }
                }
                Op::AdvanceCursor { count } => {
                    let count = *count;
                    dynasm!(self.ops ; .arch aarch64 ; add x19, x19, #count);
                }
                Op::AdvanceCursorBySlot { slot } => match slot {
                    Slot::A => dynasm!(self.ops ; .arch aarch64 ; add x19, x19, x9),
                    Slot::B => dynasm!(self.ops ; .arch aarch64 ; add x19, x19, x10),
                },
                Op::ZigzagDecode { slot } => match slot {
                    Slot::A => dynasm!(self.ops
                        ; .arch aarch64
                        ; lsr w10, w9, #1
                        ; and w11, w9, #1
                        ; neg w11, w11
                        ; eor w9, w10, w11
                    ),
                    Slot::B => dynasm!(self.ops
                        ; .arch aarch64
                        ; lsr w11, w10, #1
                        ; and w9, w10, #1
                        ; neg w9, w9
                        ; eor w10, w11, w9
                    ),
                },
                Op::ValidateMax {
                    slot,
                    max_val,
                    error,
                } => {
                    let max_val = *max_val;
                    let error_code = *error as u32;
                    let invalid_label = self.ops.new_dynamic_label();
                    let ok_label = self.ops.new_dynamic_label();
                    match slot {
                        Slot::A => dynasm!(self.ops
                            ; .arch aarch64
                            ; cmp w9, #max_val
                            ; b.hi =>invalid_label
                        ),
                        Slot::B => dynasm!(self.ops
                            ; .arch aarch64
                            ; cmp w10, #max_val
                            ; b.hi =>invalid_label
                        ),
                    }
                    dynasm!(self.ops
                        ; .arch aarch64
                        ; b =>ok_label
                        ; =>invalid_label
                        ; movz w9, #error_code
                        ; str w9, [x22, #CTX_ERROR_CODE]
                        ; b =>error_exit
                        ; =>ok_label
                    );
                }
                Op::TestBit7Branch { slot, target } => {
                    let label = labels[*target];
                    match slot {
                        Slot::A => dynasm!(self.ops ; .arch aarch64 ; tbnz w9, #7, =>label),
                        Slot::B => dynasm!(self.ops ; .arch aarch64 ; tbnz w10, #7, =>label),
                    }
                }
                Op::Branch { target } => {
                    let label = labels[*target];
                    dynasm!(self.ops ; .arch aarch64 ; b =>label);
                }
                Op::BindLabel { index } => {
                    let label = labels[*index];
                    dynasm!(self.ops ; .arch aarch64 ; =>label);
                }
                Op::CallIntrinsic {
                    fn_ptr,
                    field_offset,
                } => {
                    let field_offset = *field_offset;
                    self.emit_flush_input_cursor();
                    dynasm!(self.ops
                        ; .arch aarch64
                        ; mov x0, x22
                        ; add x1, x21, #field_offset
                    );
                    self.emit_call_fn_ptr(*fn_ptr);
                    self.emit_reload_cursor_and_check_error();
                }
                Op::CallIntrinsicStackOut { fn_ptr, sp_offset } => {
                    let sp_offset = *sp_offset;
                    self.emit_flush_input_cursor();
                    dynasm!(self.ops
                        ; .arch aarch64
                        ; mov x0, x22
                        ; add x1, sp, #sp_offset
                    );
                    self.emit_call_fn_ptr(*fn_ptr);
                    self.emit_reload_cursor_and_check_error();
                }
                Op::ComputeRemaining { dst } => match dst {
                    Slot::A => dynasm!(self.ops ; .arch aarch64 ; sub x9, x20, x19),
                    Slot::B => dynasm!(self.ops ; .arch aarch64 ; sub x10, x20, x19),
                },
                Op::CmpBranchLo { lhs, rhs, on_fail } => {
                    let target = match on_fail {
                        ErrorTarget::Eof => eof_label,
                        ErrorTarget::ErrorExit => error_exit,
                    };
                    match (lhs, rhs) {
                        (Slot::A, Slot::B) => dynasm!(self.ops
                            ; .arch aarch64
                            ; cmp x9, x10
                            ; b.lo =>target
                        ),
                        (Slot::B, Slot::A) => dynasm!(self.ops
                            ; .arch aarch64
                            ; cmp x10, x9
                            ; b.lo =>target
                        ),
                        _ => panic!("CmpBranchLo requires different slots"),
                    }
                }
                Op::SaveCursor { dst } => match dst {
                    Slot::A => dynasm!(self.ops ; .arch aarch64 ; mov x9, x19),
                    Slot::B => dynasm!(self.ops ; .arch aarch64 ; mov x10, x19),
                },
                Op::CallValidateAllocCopy {
                    fn_ptr,
                    data_src,
                    len_src,
                } => {
                    self.emit_flush_input_cursor();
                    dynasm!(self.ops ; .arch aarch64 ; mov x0, x22);
                    match data_src {
                        Slot::A => dynasm!(self.ops ; .arch aarch64 ; mov x1, x9),
                        Slot::B => dynasm!(self.ops ; .arch aarch64 ; mov x1, x10),
                    }
                    match len_src {
                        Slot::A => dynasm!(self.ops ; .arch aarch64 ; mov w2, w9),
                        Slot::B => dynasm!(self.ops ; .arch aarch64 ; mov w2, w10),
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
                    dynasm!(self.ops ; .arch aarch64 ; str x0, [x21, #ptr_offset]);
                    match len_slot {
                        Slot::A => {
                            dynasm!(self.ops
                                ; .arch aarch64
                                ; str x9, [x21, #len_offset]
                                ; str x9, [x21, #cap_offset]
                            );
                        }
                        Slot::B => {
                            dynasm!(self.ops
                                ; .arch aarch64
                                ; str x10, [x21, #len_offset]
                                ; str x10, [x21, #cap_offset]
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
                        (Slot::A, Width::W1) => {
                            dynasm!(self.ops ; .arch aarch64 ; ldrb w9, [x21, #offset])
                        }
                        (Slot::A, Width::W2) => {
                            dynasm!(self.ops ; .arch aarch64 ; ldrh w9, [x21, #offset])
                        }
                        (Slot::A, Width::W4) => {
                            dynasm!(self.ops ; .arch aarch64 ; ldr w9, [x21, #offset])
                        }
                        (Slot::A, Width::W8) => {
                            dynasm!(self.ops ; .arch aarch64 ; ldr x9, [x21, #offset])
                        }
                        (Slot::B, Width::W1) => {
                            dynasm!(self.ops ; .arch aarch64 ; ldrb w10, [x21, #offset])
                        }
                        (Slot::B, Width::W2) => {
                            dynasm!(self.ops ; .arch aarch64 ; ldrh w10, [x21, #offset])
                        }
                        (Slot::B, Width::W4) => {
                            dynasm!(self.ops ; .arch aarch64 ; ldr w10, [x21, #offset])
                        }
                        (Slot::B, Width::W8) => {
                            dynasm!(self.ops ; .arch aarch64 ; ldr x10, [x21, #offset])
                        }
                    }
                }
                Op::StoreToOutput { src, width } => match (src, width) {
                    (Slot::A, Width::W1) => {
                        dynasm!(self.ops ; .arch aarch64 ; strb w9, [x19] ; add x19, x19, #1)
                    }
                    (Slot::A, Width::W2) => {
                        dynasm!(self.ops ; .arch aarch64 ; strh w9, [x19] ; add x19, x19, #2)
                    }
                    (Slot::A, Width::W4) => {
                        dynasm!(self.ops ; .arch aarch64 ; str w9, [x19] ; add x19, x19, #4)
                    }
                    (Slot::A, Width::W8) => {
                        dynasm!(self.ops ; .arch aarch64 ; str x9, [x19] ; add x19, x19, #8)
                    }
                    (Slot::B, Width::W1) => {
                        dynasm!(self.ops ; .arch aarch64 ; strb w10, [x19] ; add x19, x19, #1)
                    }
                    (Slot::B, Width::W2) => {
                        dynasm!(self.ops ; .arch aarch64 ; strh w10, [x19] ; add x19, x19, #2)
                    }
                    (Slot::B, Width::W4) => {
                        dynasm!(self.ops ; .arch aarch64 ; str w10, [x19] ; add x19, x19, #4)
                    }
                    (Slot::B, Width::W8) => {
                        dynasm!(self.ops ; .arch aarch64 ; str x10, [x19] ; add x19, x19, #8)
                    }
                },
                Op::WriteByte { value } => {
                    let value = *value as u32;
                    dynasm!(self.ops
                        ; .arch aarch64
                        ; mov w9, #value
                        ; strb w9, [x19]
                        ; add x19, x19, #1
                    );
                }
                Op::AdvanceOutput { count } => {
                    let count = *count;
                    if count > 0 {
                        dynasm!(self.ops ; .arch aarch64 ; add x19, x19, #count);
                    }
                }
                Op::AdvanceOutputBySlot { slot } => match slot {
                    Slot::A => dynasm!(self.ops ; .arch aarch64 ; add x19, x19, x9),
                    Slot::B => dynasm!(self.ops ; .arch aarch64 ; add x19, x19, x10),
                },
                Op::OutputBoundsCheck { count } => {
                    let count = *count;
                    let have_space = self.ops.new_dynamic_label();
                    let grow_fn = crate::intrinsics::kajit_output_grow as *const u8;

                    dynasm!(self.ops
                        ; .arch aarch64
                        ; sub x11, x20, x19
                        ; cmp x11, #count
                        ; b.hs =>have_space
                    );
                    self.emit_enc_flush_output_cursor();
                    dynasm!(self.ops
                        ; .arch aarch64
                        ; mov x0, x22
                        ; movz x1, #count, LSL #0
                    );
                    self.emit_call_fn_ptr(grow_fn);
                    self.emit_enc_reload_and_check_error();
                    dynasm!(self.ops ; .arch aarch64 ; =>have_space);
                }
                Op::SignExtend { slot, from } => match (slot, from) {
                    (Slot::A, Width::W1) => dynasm!(self.ops ; .arch aarch64 ; sxtb w9, w9),
                    (Slot::A, Width::W2) => dynasm!(self.ops ; .arch aarch64 ; sxth w9, w9),
                    (Slot::B, Width::W1) => dynasm!(self.ops ; .arch aarch64 ; sxtb w10, w10),
                    (Slot::B, Width::W2) => dynasm!(self.ops ; .arch aarch64 ; sxth w10, w10),
                    (_, Width::W4 | Width::W8) => {} // already at natural width
                },
                Op::ZigzagEncode { slot, wide } => match (slot, wide) {
                    // zigzag encode 32-bit: (n << 1) ^ (n >> 31)
                    (Slot::A, false) => dynasm!(self.ops
                        ; .arch aarch64
                        ; lsl w10, w9, #1
                        ; asr w11, w9, #31
                        ; eor w9, w10, w11
                    ),
                    (Slot::B, false) => dynasm!(self.ops
                        ; .arch aarch64
                        ; lsl w11, w10, #1
                        ; asr w9, w10, #31
                        ; eor w10, w11, w9
                    ),
                    // zigzag encode 64-bit: (n << 1) ^ (n >> 63)
                    (Slot::A, true) => dynasm!(self.ops
                        ; .arch aarch64
                        ; lsl x10, x9, #1
                        ; asr x11, x9, #63
                        ; eor x9, x10, x11
                    ),
                    (Slot::B, true) => dynasm!(self.ops
                        ; .arch aarch64
                        ; lsl x11, x10, #1
                        ; asr x9, x10, #63
                        ; eor x10, x11, x9
                    ),
                },
                Op::EncodeVarint { slot, wide } => {
                    // Inline varint encoding loop.
                    // While value >= 0x80: write (byte | 0x80), shift right 7.
                    // Then write final byte.
                    let loop_label = self.ops.new_dynamic_label();
                    let done_label = self.ops.new_dynamic_label();

                    if *wide {
                        // 64-bit varint: use x11 register
                        match slot {
                            Slot::A => dynasm!(self.ops ; .arch aarch64 ; mov x11, x9),
                            Slot::B => dynasm!(self.ops ; .arch aarch64 ; mov x11, x10),
                        }
                        dynasm!(self.ops
                            ; .arch aarch64
                            ; =>loop_label
                            ; cmp x11, #0x80
                            ; b.lo =>done_label
                            ; orr w9, w11, #0x80
                            ; strb w9, [x19]
                            ; add x19, x19, #1
                            ; lsr x11, x11, #7
                            ; b =>loop_label
                            ; =>done_label
                            ; strb w11, [x19]
                            ; add x19, x19, #1
                        );
                    } else {
                        // 32-bit varint: use w11 register
                        match slot {
                            Slot::A => dynasm!(self.ops ; .arch aarch64 ; mov w11, w9),
                            Slot::B => dynasm!(self.ops ; .arch aarch64 ; mov w11, w10),
                        }
                        dynasm!(self.ops
                            ; .arch aarch64
                            ; =>loop_label
                            ; cmp w11, #0x80
                            ; b.lo =>done_label
                            ; orr w9, w11, #0x80
                            ; strb w9, [x19]
                            ; add x19, x19, #1
                            ; lsr w11, w11, #7
                            ; b =>loop_label
                            ; =>done_label
                            ; strb w11, [x19]
                            ; add x19, x19, #1
                        );
                    }
                }
            }
        }

        // Jump over cold path, then emit shared EOF error
        let done_label = self.ops.new_dynamic_label();
        let eof_code = crate::context::ErrorCode::UnexpectedEof as u32;
        dynasm!(self.ops
            ; .arch aarch64
            ; b =>done_label
            ; =>eof_label
            ; movz w9, #eof_code
            ; str w9, [x22, #CTX_ERROR_CODE]
            ; b =>error_exit
            ; =>done_label
        );
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

        let l_ws = self.ops.new_dynamic_label();
        let l_ws_done = self.ops.new_dynamic_label();
        let l_no_sign = self.ops.new_dynamic_label();
        let l_skip_lz = self.ops.new_dynamic_label();
        let l_skip_lz_end = self.ops.new_dynamic_label();
        let l_int_loop = self.ops.new_dynamic_label();
        let l_int_done = self.ops.new_dynamic_label();
        let l_int_ovf = self.ops.new_dynamic_label();
        let l_int_ovf_end = self.ops.new_dynamic_label();
        let l_no_dot = self.ops.new_dynamic_label();
        let l_frac_lz = self.ops.new_dynamic_label();
        let l_frac_lz_end = self.ops.new_dynamic_label();
        let l_frac_loop = self.ops.new_dynamic_label();
        let l_frac_done = self.ops.new_dynamic_label();
        let l_frac_ovf = self.ops.new_dynamic_label();
        let l_frac_ovf_end = self.ops.new_dynamic_label();
        let l_no_exp = self.ops.new_dynamic_label();
        let l_exp_pos = self.ops.new_dynamic_label();
        let l_exp_loop = self.ops.new_dynamic_label();
        let l_exp_done = self.ops.new_dynamic_label();
        let l_zero = self.ops.new_dynamic_label();
        let l_exact_int = self.ops.new_dynamic_label();
        let l_uscale = self.ops.new_dynamic_label();
        let l_pos_overflow = self.ops.new_dynamic_label();
        let l_neg_underflow = self.ops.new_dynamic_label();
        let l_need_lo_mul = self.ops.new_dynamic_label();
        let l_after_lo_mul = self.ops.new_dynamic_label();
        let l_pack_normal = self.ops.new_dynamic_label();
        let l_pack_inf = self.ops.new_dynamic_label();
        let l_apply_sign = self.ops.new_dynamic_label();
        let l_done = self.ops.new_dynamic_label();
        let l_skip_cold = self.ops.new_dynamic_label();
        let l_err_num = self.ops.new_dynamic_label();
        let l_err_eof = self.ops.new_dynamic_label();

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
        dynasm!(self.ops
            ; .arch aarch64
            ; =>l_ws
            ; cmp x19, x20
            ; b.hs =>l_ws_done
            ; ldrb w8, [x19]
            ; cmp w8, #0x20
            ; b.eq #20
            ; cmp w8, #0x09
            ; b.lo =>l_ws_done
            ; cmp w8, #0x0d
            ; b.hi =>l_ws_done
            ; add x19, x19, #1
            ; b =>l_ws
            ; =>l_ws_done
        );

        // ── Sign ──
        dynasm!(self.ops
            ; .arch aarch64
            ; cmp x19, x20
            ; b.hs =>l_err_eof
            ; mov x0, #0
            ; ldrb w8, [x19]
            ; cmp w8, #0x2d
            ; b.ne =>l_no_sign
            ; mov x0, #1
            ; add x19, x19, #1
            ; cmp x19, x20
            ; b.hs =>l_err_eof
            ; ldrb w8, [x19]
            ; =>l_no_sign
        );

        // ── Digit extraction ──
        dynasm!(self.ops
            ; .arch aarch64
            ; mov x9, #0
            ; mov x10, #0
            ; mov x11, #0
            ; mov x12, #0
            ; mov x13, #0

            // Leading integer zeros
            ; =>l_skip_lz
            ; cmp w8, #0x30
            ; b.ne =>l_skip_lz_end
            ; orr x13, x13, #2
            ; add x19, x19, #1
            ; cmp x19, x20
            ; b.hs =>l_skip_lz_end
            ; ldrb w8, [x19]
            ; b =>l_skip_lz
            ; =>l_skip_lz_end

            // Integer digit loop
            ; =>l_int_loop
            ; sub w1, w8, #0x30
            ; cmp w1, #9
            ; b.hi =>l_int_done
            ; orr x13, x13, #2
            ; cmp x10, #19
            ; b.hs =>l_int_ovf
            ; mov x2, #10
            ; madd x9, x9, x2, x1
            ; add x10, x10, #1
            ; add x19, x19, #1
            ; cmp x19, x20
            ; b.hs =>l_int_done
            ; ldrb w8, [x19]
            ; b =>l_int_loop

            ; =>l_int_ovf
            ; add x12, x12, #1
            ; add x19, x19, #1
            ; cmp x19, x20
            ; b.hs =>l_int_ovf_end
            ; ldrb w8, [x19]
            ; sub w1, w8, #0x30
            ; cmp w1, #9
            ; b.ls =>l_int_ovf
            ; =>l_int_ovf_end
            ; =>l_int_done
        );

        // ── Decimal point ──
        dynasm!(self.ops
            ; .arch aarch64
            ; cmp x19, x20
            ; b.hs =>l_no_dot
            ; cmp w8, #0x2e
            ; b.ne =>l_no_dot
            ; orr x13, x13, #1
            ; add x19, x19, #1
            ; cmp x19, x20
            ; b.hs =>l_no_dot
            ; ldrb w8, [x19]

            ; cbnz x10, =>l_frac_lz_end
            ; =>l_frac_lz
            ; cmp w8, #0x30
            ; b.ne =>l_frac_lz_end
            ; orr x13, x13, #2
            ; add x11, x11, #1
            ; add x19, x19, #1
            ; cmp x19, x20
            ; b.hs =>l_frac_lz_end
            ; ldrb w8, [x19]
            ; b =>l_frac_lz
            ; =>l_frac_lz_end

            ; =>l_frac_loop
            ; sub w1, w8, #0x30
            ; cmp w1, #9
            ; b.hi =>l_frac_done
            ; orr x13, x13, #2
            ; cmp x10, #19
            ; b.hs =>l_frac_ovf
            ; mov x2, #10
            ; madd x9, x9, x2, x1
            ; add x10, x10, #1
            ; add x11, x11, #1
            ; add x19, x19, #1
            ; cmp x19, x20
            ; b.hs =>l_frac_done
            ; ldrb w8, [x19]
            ; b =>l_frac_loop

            ; =>l_frac_ovf
            ; add x19, x19, #1
            ; cmp x19, x20
            ; b.hs =>l_frac_ovf_end
            ; ldrb w8, [x19]
            ; sub w1, w8, #0x30
            ; cmp w1, #9
            ; b.ls =>l_frac_ovf
            ; =>l_frac_ovf_end
            ; =>l_frac_done
            ; =>l_no_dot
        );

        // ── Validation ──
        dynasm!(self.ops
            ; .arch aarch64
            ; tst x13, #2
            ; b.eq =>l_err_num
        );

        // ── Exponent ──
        dynasm!(self.ops
            ; .arch aarch64
            ; mov x14, #0
            ; cmp x19, x20
            ; b.hs =>l_no_exp
            ; cmp w8, #0x65
            ; b.eq #8
            ; cmp w8, #0x45
            ; b.ne =>l_no_exp
            ; add x19, x19, #1
            ; cmp x19, x20
            ; b.hs =>l_err_num
            ; ldrb w8, [x19]
            ; mov x15, #0
            ; cmp w8, #0x2d
            ; b.ne =>l_exp_pos
            ; mov x15, #1
            ; add x19, x19, #1
            ; cmp x19, x20
            ; b.hs =>l_err_num
            ; ldrb w8, [x19]
            ; b =>l_exp_loop
            ; =>l_exp_pos
            ; cmp w8, #0x2b
            ; b.ne =>l_exp_loop
            ; add x19, x19, #1
            ; cmp x19, x20
            ; b.hs =>l_err_num
            ; ldrb w8, [x19]

            // Validate first exponent digit
            ; sub w1, w8, #0x30
            ; cmp w1, #9
            ; b.hi =>l_err_num       // bare 'e'/'e+'/etc

            ; =>l_exp_loop
            ; sub w1, w8, #0x30
            ; cmp w1, #9
            ; b.hi =>l_exp_done
            ; mov x2, #10
            ; madd x14, x14, x2, x1
            ; mov x2, #9999
            ; cmp x14, x2
            ; csel x14, x2, x14, hi
            ; add x19, x19, #1
            ; cmp x19, x20
            ; b.hs =>l_exp_done
            ; ldrb w8, [x19]
            ; b =>l_exp_loop

            ; =>l_exp_done
            ; cbz x15, #8
            ; neg x14, x14
            ; =>l_no_exp
        );

        // ── Compute p, dispatch ──
        dynasm!(self.ops
            ; .arch aarch64
            ; add x12, x14, x12     // p = exp + dropped
            ; sub x12, x12, x11     // p -= frac_digits

            ; cbz x9, =>l_zero

            ; tst x13, #1           // has_dot?
            ; b.ne =>l_uscale
            ; cbnz x14, =>l_uscale  // exp != 0?
            ; mov x1, #1
            ; lsl x1, x1, #53
            ; cmp x9, x1
            ; b.hs =>l_uscale

            ; =>l_exact_int
            ; ucvtf d0, x9
            ; fmov x7, d0
            ; b =>l_apply_sign

            ; =>l_zero
            ; mov x7, #0
            ; b =>l_apply_sign
        );

        // ── uscale ──
        dynasm!(self.ops
            ; .arch aarch64
            ; =>l_uscale

            // Range check p (signed comparison)
            ; cmp w12, #347
            ; b.gt =>l_pos_overflow
            ; cmn w12, #348
            ; b.lt =>l_neg_underflow

            // lp = (p * 108853) >> 15  → x14
            ; movz w1, LOG2_10_LO
            ; movk w1, LOG2_10_HI, lsl #16
            ; smull x14, w12, w1
            ; asr x14, x14, #15         // x14 = lp

            // clz → x4
            ; clz x4, x9

            // e = min(1074, clz - 11 - lp) → x6
            ; sub x6, x4, #11
            ; sub x6, x6, x14
            ; mov x1, #1074
            ; cmp x6, x1
            ; csel x6, x1, x6, gt

            // s = clz - e - lp - 3 → x3
            ; sub x3, x4, x6            // clz - e
            ; sub x3, x3, x14           // clz - e - lp
            ; sub x3, x3, #3            // s

            // left-justify: x5 = d << clz
            ; lsl x5, x9, x4

            // Table lookup: index = p + 348
            ; add w8, w12, #348
            ; movz x1, #tab_lo
            ; movk x1, #tab_hi16, lsl #16
            ; movk x1, #tab_hi32, lsl #32
            ; movk x1, #tab_hi48, lsl #48
            ; add x1, x1, x8, lsl #4    // &table[idx]
            ; ldp x1, x2, [x1]          // x1=pm_hi, x2=pm_lo

            // mul128(x_left, pm_hi): hi=umulh, mid=mul
            ; umulh x4, x5, x1          // x4 = hi
            ; mul x8, x5, x1            // x8 = mid

            // mask = (1 << (s & 63)) - 1
            ; and x14, x3, #63
            ; mov x15, #1
            ; lsl x15, x15, x14
            ; sub x15, x15, #1          // x15 = mask

            // if hi & mask == 0: need second multiply
            ; tst x4, x15
            ; b.eq =>l_need_lo_mul
            ; mov x7, #1                // sticky = 1
            ; b =>l_after_lo_mul

            ; =>l_need_lo_mul
            ; umulh x14, x5, x2         // x14 = mid2
            // sticky = (mid - mid2 > 1) ? 1 : 0
            ; subs x15, x8, x14         // x15 = mid - mid2
            ; cmp x15, #1
            ; cset x7, hi               // x7 = sticky
            // if mid < mid2: hi -= 1
            ; cmp x8, x14
            ; b.hs #8                   // skip if mid >= mid2
            ; sub x4, x4, #1

            ; =>l_after_lo_mul

            // top = (s >= 64) ? 0 : (hi >> s)
            ; lsr x14, x4, x3           // x14 = hi >> s
            ; cmp x3, #64
            ; csel x14, xzr, x14, hs    // x14 = 0 if s >= 64

            // u = top | sticky
            ; orr x7, x14, x7
        );

        // ── Overflow check + round + pack ──
        dynasm!(self.ops
            ; .arch aarch64

            // unmin(2^53) = (1<<55) - 2 = 0x007FFFFFFFFFFFFFFE
            ; movz x1, #0xFFFE
            ; movk x1, #0xFFFF, lsl #16
            ; movk x1, #0xFFFF, lsl #32
            ; movk x1, #0x007F, lsl #48
            ; cmp x7, x1
            ; b.lo #20                   // skip overflow adjust (4 insns = 16 bytes + 4)
            ; lsr x14, x7, #1
            ; and x15, x7, #1
            ; orr x7, x14, x15          // u = (u>>1)|(u&1)
            ; sub x6, x6, #1            // e -= 1

            // Round: (u + 1 + ((u >> 2) & 1)) >> 2
            ; lsr x1, x7, #2
            ; and x1, x1, #1
            ; add x7, x7, #1
            ; add x7, x7, x1
            ; lsr x7, x7, #2            // x7 = rounded mantissa

            // Pack: check bit 52
            ; mov x1, #1
            ; lsl x1, x1, #52
            ; tst x7, x1
            ; b.eq =>l_apply_sign        // subnormal: bits = x7

            // Normal: biased = 1075 - e
            ; =>l_pack_normal
            ; mov x1, #1075
            ; sub x1, x1, x6            // x1 = biased
            ; cmp x1, #2047
            ; b.hs =>l_pack_inf

            // bits = (x7 & ~(1<<52)) | (biased << 52)
            ; mov x8, #1
            ; lsl x8, x8, #52
            ; bic x7, x7, x8            // clear bit 52
            ; lsl x1, x1, #52           // biased << 52
            ; orr x7, x7, x1
            ; b =>l_apply_sign

            ; =>l_pack_inf
            ; movz x7, #0x7FF0, lsl #48
            ; b =>l_apply_sign
        );

        // ── Apply sign + store + update cursor ──
        dynasm!(self.ops
            ; .arch aarch64

            ; =>l_pos_overflow
            ; movz x7, #0x7FF0, lsl #48  // +infinity
            ; b =>l_apply_sign

            ; =>l_neg_underflow
            ; mov x7, #0                  // +0.0
            // fall through to l_apply_sign
        );

        dynasm!(self.ops
            ; .arch aarch64
            ; =>l_apply_sign
            ; cbz x0, =>l_done
            ; mov x1, #1
            ; lsl x1, x1, #63
            ; orr x7, x7, x1
            ; =>l_done
            ; str x7, [x21, #offset]
            ; str x19, [x22, #CTX_INPUT_PTR]
        );

        // ── Cold: error paths ──
        // Jump over cold paths from the hot path — but we use b =>l_apply_sign
        // to skip, so no skip branch needed here (cold paths are placed after l_done).
        dynasm!(self.ops
            ; .arch aarch64
            ; b =>l_skip_cold

            ; =>l_err_num
            ; movz w8, #error_code_invalid
            ; str w8, [x22, #CTX_ERROR_CODE]
            ; b =>error_exit

            ; =>l_err_eof
            ; movz w8, #error_code_eof
            ; str w8, [x22, #CTX_ERROR_CODE]
            ; b =>error_exit

            ; =>l_skip_cold
        );
    }

    /// Commit and finalize the assembler, returning the executable buffer.
    ///
    /// All functions must have been completed with `end_func` before calling this.
    pub fn finalize(mut self) -> dynasmrt::ExecutableBuffer {
        self.ops.commit().expect("failed to commit assembly");
        self.ops.finalize().expect("failed to finalize assembly")
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
    pub fn begin_encode_func(&mut self) -> (AssemblyOffset, DynamicLabel) {
        let error_exit = self.ops.new_dynamic_label();
        let entry = self.ops.offset();
        let frame_size = self.frame_size;

        dynasm!(self.ops
            ; .arch aarch64
            ; sub sp, sp, #frame_size
            ; stp x29, x30, [sp]
            ; stp x19, x20, [sp, #16]
            ; stp x21, x22, [sp, #32]
            ; add x29, sp, #0

            // Save arguments to callee-saved registers
            ; mov x21, x0              // x21 = input struct pointer
            ; mov x22, x1              // x22 = EncodeContext pointer

            // Cache output cursor from ctx
            ; ldr x19, [x22, #ENC_OUTPUT_PTR]  // x19 = ctx.output_ptr
            ; ldr x20, [x22, #ENC_OUTPUT_END]  // x20 = ctx.output_end
        );

        self.error_exit = error_exit;
        (entry, error_exit)
    }

    /// Emit the success epilogue and error exit for an encode function.
    pub fn end_encode_func(&mut self, error_exit: DynamicLabel) {
        let frame_size = self.frame_size;

        dynasm!(self.ops
            ; .arch aarch64
            // Success path: flush output cursor, restore registers, return
            ; str x19, [x22, #ENC_OUTPUT_PTR]
            ; ldp x21, x22, [sp, #32]
            ; ldp x19, x20, [sp, #16]
            ; ldp x29, x30, [sp]
            ; add sp, sp, #frame_size
            ; ret

            // Error exit: just restore and return (error is already in ctx.error)
            ; =>error_exit
            ; ldp x21, x22, [sp, #32]
            ; ldp x19, x20, [sp, #16]
            ; ldp x29, x30, [sp]
            ; add sp, sp, #frame_size
            ; ret
        );
    }

    /// Emit a call to another emitted encode function.
    ///
    /// Convention: x0 = input + field_offset, x1 = ctx.
    /// Flushes output cursor before call, reloads after, checks error.
    pub fn emit_enc_call_emitted_func(&mut self, label: DynamicLabel, field_offset: u32) {
        self.emit_enc_flush_output_cursor();
        dynasm!(self.ops
            ; .arch aarch64
            ; add x0, x21, #field_offset
            ; mov x1, x22
            ; bl =>label
        );
        self.emit_enc_reload_and_check_error();
    }

    /// Emit a call to an encode intrinsic: fn(ctx: *mut EncodeContext, ...).
    /// Flushes output cursor, calls, reloads output_ptr + output_end, checks error.
    pub fn emit_enc_call_intrinsic_ctx_only(&mut self, fn_ptr: *const u8) {
        self.emit_enc_flush_output_cursor();
        dynasm!(self.ops ; .arch aarch64 ; mov x0, x22);
        self.emit_call_fn_ptr(fn_ptr);
        self.emit_enc_reload_and_check_error();
    }

    /// Emit a call to an encode intrinsic: fn(ctx, arg1).
    /// Flushes output, calls, reloads output_ptr + output_end, checks error.
    pub fn emit_enc_call_intrinsic(&mut self, fn_ptr: *const u8, arg1: u64) {
        self.emit_enc_flush_output_cursor();
        dynasm!(self.ops ; .arch aarch64 ; mov x0, x22);
        load_imm64!(self.ops, x1, arg1);
        self.emit_call_fn_ptr(fn_ptr);
        self.emit_enc_reload_and_check_error();
    }

    /// Emit a call to an encode intrinsic: fn(ctx, input + field_offset).
    /// Passes the input data pointer offset by `field_offset` as the second argument.
    /// Flushes output, calls, reloads output_ptr + output_end, checks error.
    pub fn emit_enc_call_intrinsic_with_input(&mut self, fn_ptr: *const u8, field_offset: u32) {
        self.emit_enc_flush_output_cursor();
        dynasm!(self.ops
            ; .arch aarch64
            ; mov x0, x22
            ; add x1, x21, #field_offset
        );
        self.emit_call_fn_ptr(fn_ptr);
        self.emit_enc_reload_and_check_error();
    }

    /// Emit a bounds check: ensure at least `count` bytes available in output.
    /// If not enough space, calls kajit_output_grow intrinsic.
    pub fn emit_enc_ensure_capacity(&mut self, count: u32) {
        let have_space = self.ops.new_dynamic_label();
        let grow_fn = crate::intrinsics::kajit_output_grow as *const u8;

        dynasm!(self.ops
            ; .arch aarch64
            ; sub x9, x20, x19
            ; cmp x9, #count
            ; b.hs =>have_space
        );
        self.emit_enc_flush_output_cursor();
        dynasm!(self.ops
            ; .arch aarch64
            ; mov x0, x22
            ; movz x1, #count, LSL #0
        );
        self.emit_call_fn_ptr(grow_fn);
        self.emit_enc_reload_and_check_error();
        dynasm!(self.ops ; .arch aarch64 ; =>have_space);
    }

    /// Write a single immediate byte to the output buffer and advance.
    /// Caller must ensure capacity first.
    pub fn emit_enc_write_byte(&mut self, value: u8) {
        dynasm!(self.ops
            ; .arch aarch64
            ; mov w9, #value as u32
            ; strb w9, [x19]
            ; add x19, x19, #1
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
        dynasm!(self.ops
            ; .arch aarch64
            ; movz x9, #((ptr_val) & 0xFFFF) as u32
            ; movk x9, #((ptr_val >> 16) & 0xFFFF) as u32, LSL #16
            ; movk x9, #((ptr_val >> 32) & 0xFFFF) as u32, LSL #32
            ; movk x9, #((ptr_val >> 48) & 0xFFFF) as u32, LSL #48
        );
        // Copy count bytes from [x9] to [x19]
        for i in 0..count {
            dynasm!(self.ops
                ; .arch aarch64
                ; ldrb w10, [x9, #i]
                ; strb w10, [x19, #i]
            );
        }
        dynasm!(self.ops
            ; .arch aarch64
            ; add x19, x19, #count
        );
    }

    /// Advance the output cursor by `count` bytes.
    pub fn emit_enc_advance_output(&mut self, count: u32) {
        if count > 0 {
            dynasm!(self.ops
                ; .arch aarch64
                ; add x19, x19, #count
            );
        }
    }

    /// Load a value from the input struct at `offset` into a scratch register.
    /// Returns which scratch register was used (w9/x9).
    pub fn emit_enc_load_from_input(&mut self, offset: u32, width: Width) {
        match width {
            Width::W1 => dynasm!(self.ops ; .arch aarch64 ; ldrb w9, [x21, #offset]),
            Width::W2 => dynasm!(self.ops ; .arch aarch64 ; ldrh w9, [x21, #offset]),
            Width::W4 => dynasm!(self.ops ; .arch aarch64 ; ldr w9, [x21, #offset]),
            Width::W8 => dynasm!(self.ops ; .arch aarch64 ; ldr x9, [x21, #offset]),
        }
    }

    /// Store a value from scratch w9/x9 to the output buffer and advance.
    /// Caller must ensure capacity first.
    pub fn emit_enc_store_to_output(&mut self, width: Width) {
        match width {
            Width::W1 => {
                dynasm!(self.ops ; .arch aarch64 ; strb w9, [x19] ; add x19, x19, #1);
            }
            Width::W2 => {
                dynasm!(self.ops ; .arch aarch64 ; strh w9, [x19] ; add x19, x19, #2);
            }
            Width::W4 => {
                dynasm!(self.ops ; .arch aarch64 ; str w9, [x19] ; add x19, x19, #4);
            }
            Width::W8 => {
                dynasm!(self.ops ; .arch aarch64 ; str x9, [x19] ; add x19, x19, #8);
            }
        }
    }
}
