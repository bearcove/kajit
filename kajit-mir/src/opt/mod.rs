//! Optimization passes for CFG-MIR.
//!
//! These passes transform the CFG to improve code quality:
//! - `loop_phi_elim`: eliminate loop-invariant phi parameters
//! - `block_merge`: merge empty forwarding blocks
//! - `validate`: CFG validation to ensure consistency
//! - `validate_ssa`: SSA validation to catch def/use violations

pub mod block_merge;
pub mod loop_phi_elim;
pub mod validate;
pub mod validate_ssa;
