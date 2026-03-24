use kajit_emit::aarch64::{self, Emitter, LabelId, Reg};

use crate::context::{CTX_ERROR_CODE, CTX_INPUT_END, CTX_INPUT_PTR, ErrorCode};

/// Base frame size: 3 pairs of callee-saved registers = 48 bytes.
pub const BASE_FRAME: u32 = 48;

/// Base frame size for leaf functions (no x29/x30 save): 2 pairs = 32 bytes.
pub const LEAF_BASE_FRAME: u32 = 32;

/// Emission context — wraps the assembler plus bookkeeping labels.
pub struct EmitCtx {
    pub emit: Emitter,
    pub error_exit: LabelId,
    pub base_frame: u32,
    /// Total frame size (base + extra, 16-byte aligned).
    pub frame_size: u32,
    /// Whether this is a leaf function (no bl instructions).
    /// Leaf functions skip saving x29/x30 and frame pointer setup.
    pub is_leaf: bool,
}

// Register assignments (all callee-saved):
//   x19 = cached input_ptr
//   x20 = cached input_end
//   x21 = out pointer
//   x22 = ctx pointer

impl EmitCtx {
    /// Create an EmitCtx for regalloc-driven lowering that saves extra
    /// callee-saved register pairs (x23..x28) as needed.
    pub fn new_regalloc(extra_stack: u32, extra_saved_pairs: u32, is_leaf: bool) -> Self {
        assert!(
            extra_saved_pairs <= 3,
            "aarch64 regalloc supports at most 3 extra callee-saved pairs, got {extra_saved_pairs}"
        );
        let base = if is_leaf { LEAF_BASE_FRAME } else { BASE_FRAME };
        Self::new_with_base(extra_stack, base + extra_saved_pairs * 16, is_leaf)
    }

    fn new_with_base(extra_stack: u32, base_frame: u32, is_leaf: bool) -> Self {
        let frame_size = (base_frame + extra_stack + 15) & !15;
        let mut emit = Emitter::new();
        emit.enable_capture(); // Capture assembly instructions for dump/parse workflow
        let error_exit = emit.new_label();

        EmitCtx {
            emit,
            error_exit,
            base_frame,
            frame_size,
            is_leaf,
        }
    }

    // ── Call helpers ──────────────────────────────────────────────────
    //
    // These small helpers factor out the repeated patterns around function
    // calls in the JIT: flushing/reloading the cached cursor, loading a
    // function pointer, and checking the error flag.

    fn emit_add_imm_any(&mut self, rd: Reg, rn: Reg, imm: u32) {
        self.emit_add_sub_imm_chunks(rd, rn, imm, false);
    }

    fn emit_sub_imm_any(&mut self, rd: Reg, rn: Reg, imm: u32) {
        self.emit_add_sub_imm_chunks(rd, rn, imm, true);
    }

    fn emit_add_sub_imm_chunks(&mut self, rd: Reg, rn: Reg, mut imm: u32, subtract: bool) {
        if imm == 0 {
            if rd != rn {
                let opcode = if subtract { "sub" } else { "add" };
                if subtract {
                    self.emit
                        .emit_sub_imm(aarch64::Width::X64, rd, rn, 0, false)
                        .unwrap_or_else(|_| panic!("{opcode}"));
                } else {
                    self.emit
                        .emit_add_imm(aarch64::Width::X64, rd, rn, 0, false)
                        .unwrap_or_else(|_| panic!("{opcode}"));
                }
            }
            return;
        }

        let mut base = rn;
        while imm != 0 {
            let (imm12, shift, chunk) = if imm > 0x0fff {
                let shifted = (imm >> 12).min(0x0fff);
                if shifted != 0 {
                    (shifted as u16, true, shifted << 12)
                } else {
                    let low = imm.min(0x0fff);
                    (low as u16, false, low)
                }
            } else {
                (imm as u16, false, imm)
            };

            if subtract {
                self.emit
                    .emit_sub_imm(aarch64::Width::X64, rd, base, imm12, shift)
                    .expect("sub");
            } else {
                self.emit
                    .emit_add_imm(aarch64::Width::X64, rd, base, imm12, shift)
                    .expect("add");
            }

            imm -= chunk;
            base = rd;
        }
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
        let base = if self.is_leaf {
            LEAF_BASE_FRAME
        } else {
            BASE_FRAME
        };
        let extra_pairs = ((self.base_frame - base) / 16) as usize;
        assert!(
            extra_pairs <= 3,
            "unsupported extra callee-saved pair count"
        );

        self.emit_sub_imm_any(Reg::SP, Reg::SP, frame_size);

        // Slot layout (leaf):    [x19,x20] [x21,x22] [extra...] [spills...]
        // Slot layout (non-leaf): [x29,x30] [x19,x20] [x21,x22] [extra...] [spills...]
        let mut offset: i16 = 0;

        if !self.is_leaf {
            self.emit
                .emit_stp(aarch64::Width::X64, Reg::X29, Reg::X30, Reg::SP, offset)
                .expect("stp");
            offset += 16;
        }
        self.emit
            .emit_stp(aarch64::Width::X64, Reg::X19, Reg::X20, Reg::SP, offset)
            .expect("stp");
        offset += 16;
        self.emit
            .emit_stp(aarch64::Width::X64, Reg::X21, Reg::X22, Reg::SP, offset)
            .expect("stp");
        offset += 16;
        if extra_pairs >= 1 {
            self.emit
                .emit_stp(aarch64::Width::X64, Reg::X23, Reg::X24, Reg::SP, offset)
                .expect("stp");
            offset += 16;
        }
        if extra_pairs >= 2 {
            self.emit
                .emit_stp(aarch64::Width::X64, Reg::X25, Reg::X26, Reg::SP, offset)
                .expect("stp");
            offset += 16;
        }
        if extra_pairs >= 3 {
            self.emit
                .emit_stp(aarch64::Width::X64, Reg::X27, Reg::X28, Reg::SP, offset)
                .expect("stp");
        }
        if !self.is_leaf {
            self.emit
                .emit_add_imm(aarch64::Width::X64, Reg::X29, Reg::SP, 0, false)
                .expect("add");
        }
        self.emit
            .emit_mov_reg(aarch64::Width::X64, Reg::X21, Reg::X0)
            .expect("mov");
        self.emit
            .emit_mov_reg(aarch64::Width::X64, Reg::X22, Reg::X1)
            .expect("mov");
        self.emit
            .emit_ldr_imm(aarch64::Width::X64, Reg::X19, Reg::X22, CTX_INPUT_PTR)
            .expect("ldr");
        self.emit
            .emit_ldr_imm(aarch64::Width::X64, Reg::X20, Reg::X22, CTX_INPUT_END)
            .expect("ldr");

        self.error_exit = error_exit;
        (entry, error_exit)
    }

    /// Emit the success epilogue and error exit for the current function.
    ///
    /// `error_exit` must be the label returned by the corresponding `begin_func` call.
    pub fn end_func(&mut self, error_exit: LabelId) {
        let frame_size = self.frame_size;
        let base = if self.is_leaf {
            LEAF_BASE_FRAME
        } else {
            BASE_FRAME
        };
        let extra_pairs = ((self.base_frame - base) / 16) as usize;
        assert!(
            extra_pairs <= 3,
            "unsupported extra callee-saved pair count"
        );

        // Emit epilogue (success path), then error exit with same epilogue
        for is_error in [false, true] {
            if is_error {
                self.emit.bind_label(error_exit).expect("bind");
            } else {
                // Write back cursor before returning on success
                self.emit
                    .emit_str_imm(aarch64::Width::X64, Reg::X19, Reg::X22, CTX_INPUT_PTR)
                    .expect("str");
            }

            // Restore callee-saved registers in reverse order
            let mut offset: i16 = base as i16 + (extra_pairs as i16 - 1) * 16;
            if extra_pairs >= 3 {
                self.emit
                    .emit_ldp(aarch64::Width::X64, Reg::X27, Reg::X28, Reg::SP, offset)
                    .expect("ldp");
                offset -= 16;
            }
            if extra_pairs >= 2 {
                self.emit
                    .emit_ldp(aarch64::Width::X64, Reg::X25, Reg::X26, Reg::SP, offset)
                    .expect("ldp");
                offset -= 16;
            }
            if extra_pairs >= 1 {
                self.emit
                    .emit_ldp(aarch64::Width::X64, Reg::X23, Reg::X24, Reg::SP, offset)
                    .expect("ldp");
            }
            // x21/x22 and x19/x20 are always saved
            let x21_offset: i16 = if self.is_leaf { 16 } else { 32 };
            let x19_offset: i16 = if self.is_leaf { 0 } else { 16 };
            self.emit
                .emit_ldp(aarch64::Width::X64, Reg::X21, Reg::X22, Reg::SP, x21_offset)
                .expect("ldp");
            self.emit
                .emit_ldp(aarch64::Width::X64, Reg::X19, Reg::X20, Reg::SP, x19_offset)
                .expect("ldp");
            if !self.is_leaf {
                self.emit
                    .emit_ldp(aarch64::Width::X64, Reg::X29, Reg::X30, Reg::SP, 0)
                    .expect("ldp");
            }
            self.emit_add_imm_any(Reg::SP, Reg::SP, frame_size);
            self.emit.emit_ret().expect("ret");
        }
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

    /// Emit a bounds check: verify that at least `count` bytes remain in the
    /// input buffer. Branches to the error exit with UnexpectedEof on failure.
    pub fn emit_bounds_check(&mut self, count: u32) {
        let eof_label = self.emit.new_label();
        if count == 1 {
            self.emit
                .emit_cmp_reg(aarch64::Width::X64, Reg::X19, Reg::X20)
                .expect("cmp");
            self.emit
                .emit_b_cond_label(aarch64::Condition::Hs, eof_label)
                .expect("b.hs");
        } else {
            self.emit
                .emit_sub_reg(aarch64::Width::X64, Reg::X9, Reg::X20, Reg::X19)
                .expect("sub");
            self.emit
                .emit_cmp_imm(aarch64::Width::X64, Reg::X9, count as u16)
                .expect("cmp");
            self.emit
                .emit_b_cond_label(aarch64::Condition::Lo, eof_label)
                .expect("b.lo");
        }
        // EOF error path
        let past_eof = self.emit.new_label();
        self.emit.emit_b_label(past_eof).expect("b");
        self.emit.bind_label(eof_label).expect("bind eof");
        self.emit
            .emit_movz_imm(
                aarch64::Width::X64,
                Reg::X9,
                ErrorCode::UnexpectedEof as u16,
                0,
            )
            .expect("movz");
        self.emit
            .emit_str_imm(aarch64::Width::W32, Reg::X9, Reg::X22, CTX_ERROR_CODE)
            .expect("str");
        let error_exit = self.error_exit;
        self.emit.emit_b_label(error_exit).expect("b");
        self.emit.bind_label(past_eof).expect("bind past_eof");
    }

    /// Emit an error (write error code to ctx, branch to error_exit).
    pub fn emit_error(&mut self, code: crate::context::ErrorCode) {
        let error_exit = self.error_exit;
        let error_code = code as u32;
        self.emit
            .emit_movz_imm(aarch64::Width::W32, Reg::X9, error_code as u16, 0)
            .expect("movz");
        self.emit
            .emit_str_imm(aarch64::Width::W32, Reg::X9, Reg::X22, CTX_ERROR_CODE)
            .expect("str");
        self.emit.emit_b_label(error_exit).expect("b");
    }

    /// Advance the cached cursor by n bytes (inline, no function call).
    pub fn emit_advance_cursor_by(&mut self, n: u32) {
        self.emit
            .emit_add_imm(aarch64::Width::X64, Reg::X19, Reg::X19, n as u16, false)
            .expect("add");
    }

    /// Commit and finalize the assembler, returning the executable buffer.
    ///
    /// All functions must have been completed with `end_func` before calling this.
    pub fn finalize(
        mut self,
    ) -> (
        aarch64::FinalizedEmission,
        Option<kajit_emit::aarch64_asm::Program>,
    ) {
        let asm_program = self.emit.take_captured_program();
        let buf = self.emit.finalize().expect("failed to finalize assembly");
        (buf, asm_program)
    }
}
