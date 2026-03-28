//! Optimization passes for CFG-MIR.
//!
//! These passes transform the CFG to improve code quality:
//! - `constant_phi_elim`: eliminate redundant block parameters (constant phis)
//! - `loop_phi_elim`: (DEPRECATED) eliminate loop-invariant phi parameters
//! - `block_merge`: merge empty forwarding blocks
//! - `validate`: CFG validation to ensure consistency
//! - `validate_ssa`: SSA validation to catch def/use violations

pub mod block_merge;
pub mod const_branch_fold;
pub mod constant_phi_elim;
pub mod control_thread;
pub mod dce;
pub mod loop_phi_elim;
pub mod reduce;
pub mod simplify_cfg;
pub mod validate;
pub mod validate_ssa;
