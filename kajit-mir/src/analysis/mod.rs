//! Analysis passes for CFG-MIR optimization.
//!
//! These modules compute program properties used by optimization passes:
//! - `dominance`: dominance relationships between blocks
//! - `loops`: loop detection and loop nesting
//! - `defuse`: def-use chains for value flow analysis

pub mod dominance;
pub mod loops;
