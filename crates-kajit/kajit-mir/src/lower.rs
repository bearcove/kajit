//! MIR lowering entrypoints.
//!
//! This will take schema-owned MIR (`kajit_reprs::mir`) plus its `Graph` storage
//! and produce schema-owned ASM (`kajit_reprs::asm`) or another backend-friendly
//! representation.

use kajit_reprs::{asm, mir};

#[derive(Debug)]
pub enum LowerError {
    Unsupported(&'static str),
}

impl std::fmt::Display for LowerError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Unsupported(msg) => write!(f, "unsupported: {msg}"),
        }
    }
}

impl std::error::Error for LowerError {}

/// Lower one MIR program into a single architecture-specific ASM repr program.
///
/// Placeholder API: we can refine once we agree on the minimal 10% surface.
pub fn lower_program_to_asm(
    _mir_graph: &mir::Graph,
    _mir_program: &mir::Program,
) -> Result<asm::Program, LowerError> {
    Err(LowerError::Unsupported(
        "MIR→ASM lowering not implemented yet",
    ))
}
