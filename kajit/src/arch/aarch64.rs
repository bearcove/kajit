use kajit_emit::aarch64::{self, Emitter, LabelId, Reg};

use crate::context::CTX_ERROR_CODE;

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

impl EmitCtx {
    /// Create an EmitCtx for regalloc-driven lowering that saves extra
    /// callee-saved register pairs (x23..x28) as needed.
    pub fn new_regalloc(extra_stack: u32, extra_saved_pairs: u32, is_leaf: bool) -> Self {
        assert!(
            extra_saved_pairs <= 3,
            "aarch64 regalloc supports at most 3 extra callee-saved pairs, got {extra_saved_pairs}"
        );
        let base = if is_leaf { LEAF_BASE_FRAME } else { BASE_FRAME };
        Self::new_with_base(extra_stack, base + extra_saved_pairs * 16)
    }

    fn new_with_base(extra_stack: u32, base_frame: u32) -> Self {
        let frame_size = (base_frame + extra_stack + 15) & !15;
        let mut emit = Emitter::new();
        emit.enable_capture(); // Capture assembly instructions for dump/parse workflow
        let error_exit = emit.new_label();

        EmitCtx {
            emit,
            error_exit,
            base_frame,
            frame_size,
            error_trampolines: std::collections::HashMap::new(),
            error_ctx_reg: None,
        }
    }

    // ── Call helpers ──────────────────────────────────────────────────

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
