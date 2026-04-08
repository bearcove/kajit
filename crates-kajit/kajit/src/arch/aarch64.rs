use kajit_asm::aarch64::{self, Emitter, LabelId, Reg};

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
    pub fn set_source_location(&mut self, location: kajit_asm::SourceLocation) {
        self.emit.set_source_location(location);
    }

    pub fn current_source_location(&self) -> kajit_asm::SourceLocation {
        self.emit.current_source_location()
    }

    /// Commit and finalize the assembler, returning the executable buffer.
    ///
    /// All functions must have been completed with `end_func` before calling this.
    pub fn finalize(
        mut self,
    ) -> (
        aarch64::FinalizedEmission,
        Option<kajit_asm::aarch64_asm::Program>,
    ) {
        let asm_program = self.emit.take_captured_program();
        let buf = self.emit.finalize().expect("failed to finalize assembly");
        (buf, asm_program)
    }
}
