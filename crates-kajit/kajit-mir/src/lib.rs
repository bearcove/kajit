//! New MIR crate (fresh rewrite).
//!
//! Responsibilities (intended):
//! - Own the MIR→ASM lowering pipeline for schema-owned CFG-MIR (`kajit-reprs::mir`)
//! - Keep the lowering and optimization logic out of `kajit-reprs` (reprs stay structural)
//!
//! The legacy implementation lives in `kajit-mir-legacy` for reference.

pub mod lower;
