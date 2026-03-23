//! Optimization passes for CFG-MIR.
//!
//! These passes transform the CFG to improve code quality:
//! - `loop_phi_elim`: eliminate loop-invariant phi parameters

pub mod loop_phi_elim;
