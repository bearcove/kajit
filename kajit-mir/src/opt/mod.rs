//! Optimization passes for CFG-MIR.
//!
//! These passes transform the CFG to improve code quality:
//! - `loop_phi_elim`: eliminate loop-invariant phi parameters
//! - `block_merge`: merge empty forwarding blocks

pub mod block_merge;
pub mod loop_phi_elim;
