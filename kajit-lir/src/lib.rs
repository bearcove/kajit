//! LIR (Linear IR (Intermediate Representation))
//!
//! The RVSDG is a tree of regions and nodes. The linearizer walks this tree,
//! topologically sorts each region's nodes, and emits a flat `Vec<LinearOp>`
//! with explicit labels and branches for control flow (gamma/theta).

mod ir;
pub use ir::*;

mod fmt;
pub use fmt::*;

mod linearizer;
pub use linearizer::*;

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests;
