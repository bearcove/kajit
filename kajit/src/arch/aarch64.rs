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
    /// The register holding the current cursor position (input_ptr).
    /// x19 for non-leaf, x19 for leaf when prologue loads it, or
    /// whatever register the leaf SaveCursor loaded into.
    pub cursor_reg: Reg,
    /// The register holding the input end pointer.
    /// x20 for non-leaf, x20 for leaf when prologue loads it.
    pub end_reg: Reg,
    /// Shared error trampoline labels: error code → label.
    /// Each error site emits `b .Lerr_<code>` instead of the full 3-instruction
    /// sequence. The shared blocks are emitted near the epilogue.
    error_trampolines: std::collections::HashMap<u32, LabelId>,
    /// The ctx register used for error trampolines (set during func emission).
    error_ctx_reg: Option<Reg>,
}

// Register assignments (default, all callee-saved):
//   x19 = cached input_ptr
//   x20 = cached input_end
//   x21 = out pointer (non-leaf) or x0 (leaf)
//   x22 = ctx pointer (non-leaf) or x1 (leaf)

/// Controls which registers the prologue/epilogue use for fixed ABI roles.
#[derive(Debug, Clone)]
pub struct PrologueConfig {
    /// Whether to save/restore x21/x22 and move args.
    /// When false (leaf optimization), output_ptr stays in x0 and ctx_ptr in x1.
    pub save_x21_x22: bool,
    /// Whether to save/restore x19/x20. When false, the prologue/epilogue skip
    /// the stp/ldp for these registers (leaf functions that don't modify them).
    pub save_x19_x20: bool,
    /// Whether the prologue loads cursor/end into x19/x20 and the epilogue
    /// writes x19 back. When false (leaf + regalloc3), SaveCursor/SaveInputEnd
    /// load from the context struct directly, and the epilogue uses `cursor_writeback_reg`.
    pub load_cursor_x19_x20: bool,
    /// Register to read the cursor from for the success-path writeback.
    /// Only used when `load_cursor_x19_x20` is false.
    pub cursor_writeback_reg: Option<Reg>,
    /// Whether the success epilogue should write the cursor back to ctx.input_ptr.
    pub writeback_cursor_to_ctx: bool,
}

impl Default for PrologueConfig {
    fn default() -> Self {
        Self {
            save_x21_x22: true,
            save_x19_x20: true,
            load_cursor_x19_x20: true,
            cursor_writeback_reg: None,
            writeback_cursor_to_ctx: true,
        }
    }
}

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
            cursor_reg: Reg::X19,
            end_reg: Reg::X20,
            error_trampolines: std::collections::HashMap::new(),
            error_ctx_reg: None,
        }
    }

    // ── Call helpers ──────────────────────────────────────────────────
    //
    // These small helpers factor out the repeated patterns around function
    // calls in the JIT: flushing/reloading the cached cursor, loading a
    // function pointer, and checking the error flag.

    pub(crate) fn emit_add_imm_any(&mut self, rd: Reg, rn: Reg, imm: u32) {
        self.emit_add_sub_imm_chunks(rd, rn, imm, false);
    }

    pub(crate) fn emit_sub_imm_any(&mut self, rd: Reg, rn: Reg, imm: u32) {
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
        self.begin_func_with_config(&PrologueConfig::default())
    }

    pub fn begin_func_with_config(&mut self, config: &PrologueConfig) -> (u32, LabelId) {
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

        if frame_size > 0 {
            self.emit_sub_imm_any(Reg::SP, Reg::SP, frame_size);
        }

        let mut offset: i16 = 0;

        if !self.is_leaf {
            self.emit
                .emit_stp(aarch64::Width::X64, Reg::X29, Reg::X30, Reg::SP, offset)
                .expect("stp");
            offset += 16;
        }
        if config.save_x19_x20 {
            self.emit
                .emit_stp(aarch64::Width::X64, Reg::X19, Reg::X20, Reg::SP, offset)
                .expect("stp");
            offset += 16;
        }
        if config.save_x21_x22 {
            self.emit
                .emit_stp(aarch64::Width::X64, Reg::X21, Reg::X22, Reg::SP, offset)
                .expect("stp");
            offset += 16;
        }
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

        // ctx_reg: x1 (leaf without x21/x22 save) or x22 (default)
        let ctx_reg = if config.save_x21_x22 {
            self.emit
                .emit_mov_reg(aarch64::Width::X64, Reg::X21, Reg::X0)
                .expect("mov");
            self.emit
                .emit_mov_reg(aarch64::Width::X64, Reg::X22, Reg::X1)
                .expect("mov");
            Reg::X22
        } else {
            // Keep args in x0/x1 — no moves needed
            Reg::X1
        };

        if config.load_cursor_x19_x20 {
            self.emit
                .emit_ldr_imm(aarch64::Width::X64, Reg::X19, ctx_reg, CTX_INPUT_PTR)
                .expect("ldr");
            self.emit
                .emit_ldr_imm(aarch64::Width::X64, Reg::X20, ctx_reg, CTX_INPUT_END)
                .expect("ldr");
        }

        self.error_exit = error_exit;
        (entry, error_exit)
    }

    /// Emit the success epilogue and error exit for the current function.
    ///
    /// `error_exit` must be the label returned by the corresponding `begin_func` call.
    pub fn end_func(&mut self, error_exit: LabelId) {
        self.end_func_with_config(error_exit, &PrologueConfig::default());
    }

    pub fn end_func_with_config(&mut self, error_exit: LabelId, config: &PrologueConfig) {
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

        // ctx_reg for cursor writeback: x1 if leaf without x21/x22 save, x22 otherwise
        let ctx_reg = if config.save_x21_x22 {
            Reg::X22
        } else {
            Reg::X1
        };

        // Emit epilogue (success path), then error exit with same epilogue
        for is_error in [false, true] {
            if is_error {
                self.emit.bind_label(error_exit).expect("bind");
            } else if config.writeback_cursor_to_ctx {
                // Write back cursor before returning on success
                let cursor_reg = config.cursor_writeback_reg.unwrap_or(Reg::X19);
                self.emit
                    .emit_str_imm(aarch64::Width::X64, cursor_reg, ctx_reg, CTX_INPUT_PTR)
                    .expect("str");
            }

            // Restore callee-saved registers in reverse order
            let mut restore_base = base;
            if !config.save_x21_x22 {
                restore_base -= 16;
            }
            if !config.save_x19_x20 {
                restore_base -= 16;
            }
            let mut offset: i16 = restore_base as i16 + (extra_pairs as i16 - 1) * 16;
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
            if config.save_x21_x22 {
                let x21_offset: i16 = if self.is_leaf { 16 } else { 32 };
                self.emit
                    .emit_ldp(aarch64::Width::X64, Reg::X21, Reg::X22, Reg::SP, x21_offset)
                    .expect("ldp");
            }
            if config.save_x19_x20 {
                let x19_offset: i16 = if self.is_leaf { 0 } else { 16 };
                self.emit
                    .emit_ldp(aarch64::Width::X64, Reg::X19, Reg::X20, Reg::SP, x19_offset)
                    .expect("ldp");
            }
            if !self.is_leaf {
                self.emit
                    .emit_ldp(aarch64::Width::X64, Reg::X29, Reg::X30, Reg::SP, 0)
                    .expect("ldp");
            }
            if frame_size > 0 {
                self.emit_add_imm_any(Reg::SP, Reg::SP, frame_size);
            }
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
        let cursor = self.cursor_reg;
        let end = self.end_reg;
        if count == 1 {
            self.emit
                .emit_cmp_reg(aarch64::Width::X64, cursor, end)
                .expect("cmp");
            self.emit
                .emit_b_cond_label(aarch64::Condition::Hs, eof_label)
                .expect("b.hs");
        } else {
            self.emit
                .emit_sub_reg(aarch64::Width::X64, Reg::X9, end, cursor)
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
        let ctx_reg = if self.is_leaf { Reg::X1 } else { Reg::X22 };
        self.emit
            .emit_str_imm(aarch64::Width::W32, Reg::X9, ctx_reg, CTX_ERROR_CODE)
            .expect("str");
        let error_exit = self.error_exit;
        self.emit.emit_b_label(error_exit).expect("b");
        self.emit.bind_label(past_eof).expect("bind past_eof");
    }

    /// Emit an error (write error code to ctx, branch to error_exit).
    /// Uses X22 as ctx_reg for non-leaf, X1 for leaf.
    pub fn emit_error(&mut self, code: crate::context::ErrorCode) {
        let ctx_reg = if self.is_leaf { Reg::X1 } else { Reg::X22 };
        self.emit_error_with_ctx_reg(code, ctx_reg);
    }

    /// Emit an error with an explicit ctx register.
    /// Uses shared trampolines: each error site emits just `b .Lerr_<code>`,
    /// and the shared trampoline block is emitted near the epilogue.
    pub fn emit_error_with_ctx_reg(&mut self, code: crate::context::ErrorCode, ctx_reg: Reg) {
        let error_code = code as u32;
        self.error_ctx_reg = Some(ctx_reg);

        // Get or create the trampoline label for this error code
        let trampoline_label = if let Some(&label) = self.error_trampolines.get(&error_code) {
            label
        } else {
            let label = self.emit.new_label();
            self.error_trampolines.insert(error_code, label);
            label
        };

        self.emit.emit_b_label(trampoline_label).expect("b");
    }

    /// Emit shared error trampoline blocks. Called just before the epilogue.
    /// Each trampoline: `movz w9, #code; str w9, [ctx, #error_code]; b .Lexit`
    pub fn emit_error_trampolines(&mut self) {
        let error_exit = self.error_exit;
        let ctx_reg = self.error_ctx_reg.unwrap_or(Reg::X1);

        // Sort by error code for deterministic output
        let mut codes: Vec<(u32, LabelId)> = self.error_trampolines.drain().collect();
        codes.sort_by_key(|(code, _)| *code);

        for (code, label) in codes {
            self.emit.bind_label(label).expect("bind error trampoline");
            self.emit
                .emit_movz_imm(aarch64::Width::W32, Reg::X9, code as u16, 0)
                .expect("movz");
            self.emit
                .emit_str_imm(aarch64::Width::W32, Reg::X9, ctx_reg, CTX_ERROR_CODE)
                .expect("str");
            self.emit.emit_b_label(error_exit).expect("b");
        }
    }

    /// Advance the cached cursor by n bytes (inline, no function call).
    pub fn emit_advance_cursor_by(&mut self, n: u32) {
        let cursor = self.cursor_reg;
        self.emit
            .emit_add_imm(aarch64::Width::X64, cursor, cursor, n as u16, false)
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
