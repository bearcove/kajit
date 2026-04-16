use kajit_asm::x64::{Emitter, FinalizedEmission, LabelId};

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
}

impl EmitCtx {
    /// Create an EmitCtx for regalloc3-driven lowering.
    ///
    /// `extra_stack` is the number of bytes needed for spill slots + user slots + edge temps.
    /// `base_frame` is the offset from rsp where spill slots begin (past callee-saved saves).
    pub fn new_regalloc(extra_stack: u32, base_frame: u32, _is_leaf: bool) -> Self {
        let frame_size = (base_frame + extra_stack + 15) & !15;
        let mut emit = Emitter::new();
        let error_exit = emit.new_label();

        EmitCtx {
            emit,
            error_exit,
            base_frame,
            frame_size,
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
    pub fn set_source_location(&mut self, location: kajit_asm::SourceLocation) {
        self.emit.set_source_location(location);
    }

    pub fn current_source_location(&self) -> kajit_asm::SourceLocation {
        self.emit.current_source_location()
    }

    pub fn finalize(self) -> FinalizedEmission {
        self.emit.finalize().expect("failed to finalize assembly")
    }
}
