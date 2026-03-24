//! regalloc3: Native register allocator for CFG-MIR
//!
//! This module implements a simple, correct register allocator that works
//! directly on our CFG-MIR without an adapter layer.
//!
//! ## Design Principles
//!
//! 1. **Simple over sophisticated** - Linear scan, not graph coloring
//! 2. **Correct over optimal** - First goal is removing adapter bugs
//! 3. **One canonical operand model** - No decomposed trait methods
//! 4. **Whole-interval allocation** - One home per vreg (no splitting in v1)
//! 5. **Explicit contracts** - ABI, scratch registers, entry/exit semantics
//!
//! ## Phase 1 Scope (v1)
//!
//! - GPR only (no SIMD)
//! - Whole-interval allocation (no live-range splitting)
//! - No coalescing (copies are fine)
//! - No rematerialization (spill everything)
//! - No RVSDG hints (allocate dumbly)
//!
//! ## Architecture
//!
//! ```text
//! CFG-MIR (stable IDs)
//!   ↓
//! Program points (dense ordering derived per-function)
//!   ↓
//! Liveness analysis (with holes, includes block params)
//!   ↓
//! Linear scan allocation (one register class)
//!   ↓
//! Spill/reload insertion (rewrite-once with scratch temps)
//!   ↓
//! Symbolic verification (location flow)
//!   ↓
//! Backend emission (with physical registers)
//! ```

pub mod critical_edge;
pub mod hints;
pub mod linear_scan;
pub mod liveness;
pub mod machine_inst;
pub mod parallel_copy;
pub mod phi_resolution;
pub mod progpoint;
pub mod spill_rewrite;
pub mod verify;

#[cfg(test)]
mod integration_tests;
