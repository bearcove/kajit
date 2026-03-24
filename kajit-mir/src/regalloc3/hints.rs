//! Allocation hints from RVSDG structure.
//!
//! ## Phase 2 Scope: Spill Costs Only
//!
//! We extract simple spill cost hints from RVSDG structure:
//! - Loop-carried values (theta entry ports) → high spill cost
//! - Non-loop-carried values → low spill cost
//!
//! ## Why This Matters
//!
//! Loop-carried values are used in every iteration. Spilling them means:
//! - Load at loop entry (once per iteration)
//! - Store at loop exit (once per iteration)
//! - Massive performance cost in hot loops
//!
//! Better to spill non-loop-carried values (used once, then dead).
//!
//! ## Threading Through Pipeline
//!
//! RVSDG analysis → VRegMetadata → CFG-MIR → Allocator
//!
//! Each vreg carries metadata through lowering. Allocator uses it for spill victim selection.

use kajit_ir::VReg;
use std::collections::HashMap;

/// Spill cost hint (simple classification)
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum SpillCost {
    /// Rematerializable constant (essentially free to spill — just re-emit movz)
    Rematerializable,
    /// Not loop-carried (cheap to spill)
    Low,
    /// Loop-carried in single loop (expensive to spill)
    Medium,
    /// Loop-carried in nested loop (very expensive to spill)
    High,
}

impl SpillCost {
    /// Weight for spill victim selection (lower = prefer to spill)
    pub fn weight(self) -> u32 {
        match self {
            SpillCost::Rematerializable => 1,
            SpillCost::Low => 2,
            SpillCost::Medium => 20,
            SpillCost::High => 200,
        }
    }
}

impl Default for SpillCost {
    fn default() -> Self {
        SpillCost::Low
    }
}

/// Allocation hint (Phase 2: spill cost only)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct AllocationHint {
    /// Spill cost (from RVSDG analysis)
    pub spill_cost: SpillCost,
}

/// Vreg metadata (threaded through lowering)
#[derive(Debug, Clone, Default)]
pub struct VRegMetadata {
    /// Spill cost hint
    pub spill_cost: SpillCost,

    /// Debug name (for debugging)
    pub debug_name: Option<String>,
}

/// Hint map (vreg -> metadata)
pub type HintMap = HashMap<VReg, VRegMetadata>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_spill_cost_ordering() {
        // Lower cost = prefer to spill
        assert!(SpillCost::Low < SpillCost::Medium);
        assert!(SpillCost::Medium < SpillCost::High);
        assert!(SpillCost::Low.weight() < SpillCost::Medium.weight());
        assert!(SpillCost::Medium.weight() < SpillCost::High.weight());
    }

    #[test]
    fn test_default_hint() {
        let hint = AllocationHint::default();
        assert_eq!(hint.spill_cost, SpillCost::Low);
    }

    #[test]
    fn test_vreg_metadata() {
        let meta = VRegMetadata {
            spill_cost: SpillCost::High,
            debug_name: Some("loop_counter".to_string()),
        };
        assert_eq!(meta.spill_cost, SpillCost::High);
        assert_eq!(meta.debug_name, Some("loop_counter".to_string()));
    }
}
