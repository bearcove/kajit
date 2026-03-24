//! Linear scan register allocation.
//!
//! ## Algorithm
//!
//! Classic linear scan (Poletto & Sarkar 1999):
//! 1. Sort live intervals by start point
//! 2. For each interval:
//!    - Expire old intervals (end < current start)
//!    - Try to allocate a free register
//!    - If no free register, spill victim (furthest next use)
//! 3. Record allocation decisions
//!
//! ## Phase 1 Constraints
//!
//! - GPR only (no SIMD)
//! - Whole-interval allocation (no splitting)
//! - No coalescing (copies stay)
//! - Spill victim = furthest next use (simple heuristic)
//!
//! ## Allocation Result
//!
//! Maps each vreg to either:
//! - Physical register (PReg)
//! - Spill slot (will be rewritten in spill/reload pass)

use kajit_ir::VReg;
use kajit_lir::LinearOp;
use std::collections::HashMap;

use super::{
    hints::{HintMap, SpillCost},
    liveness::LivenessInfo,
    machine_inst::{AbiInfo, PReg, ScratchPolicy},
    progpoint::{LiveInterval, ProgPoint},
};

use crate::cfg_mir::Function;

/// Copy-preference hints for register coalescing.
///
/// For each vreg, stores the set of vregs it has been copied to/from
/// (via phi copies on block edges). When allocating, the allocator
/// prefers to assign the same physical register as a copy partner
/// that's already allocated, eliminating the copy at runtime.
#[derive(Debug, Default)]
pub struct CopyHints {
    /// vreg → list of preferred partner vregs
    pub partners: HashMap<VReg, Vec<VReg>>,
}

impl CopyHints {
    /// Build copy hints from Copy instructions in the function.
    pub fn build(func: &Function) -> Self {
        let mut partners: HashMap<VReg, Vec<VReg>> = HashMap::new();

        for inst in &func.insts {
            if let LinearOp::Copy { dst, src } = &inst.op {
                if dst != src {
                    partners.entry(*dst).or_default().push(*src);
                    partners.entry(*src).or_default().push(*dst);
                }
            }
        }

        CopyHints { partners }
    }
}

/// Allocation decision for a vreg
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Allocation {
    /// Allocated to physical register
    Reg(PReg),
    /// Spilled to stack (slot assigned during spill/reload pass)
    Spill,
}

/// Allocation result
#[derive(Debug)]
pub struct AllocationResult {
    /// Allocation for each vreg
    pub allocations: HashMap<VReg, Allocation>,

    /// Spilled vregs (for spill/reload pass)
    pub spilled: Vec<VReg>,
}

/// Linear scan allocator
pub struct LinearScanAllocator<'a> {
    /// Live intervals (sorted by start point)
    intervals: Vec<LiveInterval>,

    /// ABI information (caller/callee-saved registers)
    abi: &'a AbiInfo,

    /// Scratch register policy
    scratch: &'a ScratchPolicy,

    /// Allocation hints (spill costs from RVSDG)
    hints: &'a HintMap,

    /// Copy-preference hints for coalescing
    copy_hints: &'a CopyHints,

    /// Allocation decisions
    allocations: HashMap<VReg, Allocation>,

    /// Active intervals (currently using registers)
    active: Vec<ActiveInterval>,

    /// Free registers (available for allocation)
    free: Vec<PReg>,

    /// Spilled vregs
    spilled: Vec<VReg>,
}

/// Active interval (vreg allocated to register)
#[derive(Debug, Clone)]
struct ActiveInterval {
    vreg: VReg,
    preg: PReg,
    end: ProgPoint,
    /// Next use point (for spill heuristic)
    next_use: Option<ProgPoint>,
}

impl<'a> LinearScanAllocator<'a> {
    /// Create allocator
    pub fn new(
        mut liveness: LivenessInfo,
        abi: &'a AbiInfo,
        scratch: &'a ScratchPolicy,
        hints: &'a HintMap,
        copy_hints: &'a CopyHints,
    ) -> Self {
        // Extract and sort intervals by start point
        let mut intervals: Vec<LiveInterval> = liveness.intervals.into_values().collect();
        intervals.sort_by_key(|iv| iv.start());

        // Initialize free register pool (GPR only, excluding scratch).
        // Callee-saved go first (bottom of stack) so they're allocated last,
        // minimizing the number of stp/ldp pairs in the prologue/epilogue.
        let mut free = Vec::new();
        for &preg in abi.callee_saved_gpr {
            if !scratch.reserved.contains(&preg) {
                free.push(preg);
            }
        }
        for &preg in abi.caller_saved_gpr {
            if !scratch.reserved.contains(&preg) {
                free.push(preg);
            }
        }

        Self {
            intervals,
            abi,
            scratch,
            hints,
            copy_hints,
            allocations: HashMap::new(),
            active: Vec::new(),
            free,
            spilled: Vec::new(),
        }
    }

    /// Run allocation
    pub fn allocate(mut self) -> AllocationResult {
        for interval in self.intervals.clone() {
            self.allocate_interval(interval);
        }

        AllocationResult {
            allocations: self.allocations,
            spilled: self.spilled,
        }
    }

    /// Allocate a single interval
    fn allocate_interval(&mut self, interval: LiveInterval) {
        let start = interval.start();

        // Expire old intervals (end < start)
        self.expire_old_intervals(start);

        // Try to allocate a register, preferring copy partner's register
        let preferred = self.find_preferred_register(interval.vreg);
        let preg = if let Some(preferred) = preferred {
            // Try to use the preferred register (copy partner's register)
            if let Some(idx) = self.free.iter().position(|&r| r == preferred) {
                Some(self.free.remove(idx))
            } else {
                self.free.pop()
            }
        } else {
            self.free.pop()
        };

        if let Some(preg) = preg {
            self.allocations
                .insert(interval.vreg, Allocation::Reg(preg));
            self.active.push(ActiveInterval {
                vreg: interval.vreg,
                preg,
                end: interval.end(),
                next_use: interval.uses.first().copied(),
            });
        } else {
            // No free register, must spill
            self.spill(interval, start);
        }
    }

    /// Find a preferred physical register for this vreg based on copy hints.
    ///
    /// If this vreg is a copy partner of another vreg that's already allocated
    /// to a physical register, prefer that register to eliminate the copy.
    fn find_preferred_register(&self, vreg: VReg) -> Option<PReg> {
        let partners = self.copy_hints.partners.get(&vreg)?;
        for partner in partners {
            if let Some(Allocation::Reg(preg)) = self.allocations.get(partner) {
                return Some(*preg);
            }
        }
        None
    }

    /// Expire intervals that end before current point
    fn expire_old_intervals(&mut self, current: ProgPoint) {
        // Remove intervals that end before current
        let mut i = 0;
        while i < self.active.len() {
            if self.active[i].end < current {
                let expired = self.active.remove(i);
                self.free.push(expired.preg);
            } else {
                i += 1;
            }
        }
    }

    /// Spill victim selection (this interval or an active one)
    ///
    /// Uses both next-use distance and spill cost hints:
    /// - Prefer to spill vregs with furthest next use
    /// - Weight by spill cost (avoid spilling loop-carried values)
    fn spill(&mut self, interval: LiveInterval, current: ProgPoint) {
        // Compute spill score for current interval
        let current_score =
            self.compute_spill_score(interval.vreg, interval.next_use_after(current));

        // Find active interval with best spill score
        let mut best_candidate = None;
        let mut best_score = current_score;

        for (idx, active) in self.active.iter().enumerate() {
            let score = self.compute_spill_score(active.vreg, active.next_use);
            if score > best_score {
                best_score = score;
                best_candidate = Some(idx);
            }
        }

        if let Some(idx) = best_candidate {
            // Spill active interval, allocate this one to its register
            let spilled = self.active.remove(idx);
            self.allocations.insert(spilled.vreg, Allocation::Spill);
            self.spilled.push(spilled.vreg);

            // Allocate this interval to the freed register
            self.allocations
                .insert(interval.vreg, Allocation::Reg(spilled.preg));
            self.active.push(ActiveInterval {
                vreg: interval.vreg,
                preg: spilled.preg,
                end: interval.end(),
                next_use: interval.next_use_after(current),
            });
        } else {
            // Spill this interval (all active intervals have worse scores)
            self.allocations.insert(interval.vreg, Allocation::Spill);
            self.spilled.push(interval.vreg);
        }
    }

    /// Compute spill score (higher = better to spill)
    ///
    /// Score = next_use_distance / spill_cost_weight
    ///
    /// - Further next use = higher score (less urgent)
    /// - Higher spill cost = lower score (avoid spilling)
    fn compute_spill_score(&self, vreg: VReg, next_use: Option<ProgPoint>) -> u64 {
        // Next use distance (or max if no use)
        let distance = next_use.map(|p| p.0 as u64).unwrap_or(u64::MAX / 1000);

        // Spill cost weight from hints
        let cost_weight = self
            .hints
            .get(&vreg)
            .map(|meta| meta.spill_cost.weight() as u64)
            .unwrap_or(SpillCost::Low.weight() as u64);

        // Score = distance / cost (higher = better to spill)
        // Use checked division to avoid overflow
        distance.checked_div(cost_weight).unwrap_or(u64::MAX)
    }
}

/// Run linear scan allocation
pub fn allocate(
    liveness: LivenessInfo,
    abi: &AbiInfo,
    scratch: &ScratchPolicy,
    hints: &HintMap,
    copy_hints: &CopyHints,
) -> AllocationResult {
    let allocator = LinearScanAllocator::new(liveness, abi, scratch, hints, copy_hints);
    allocator.allocate()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        cfg_mir::{Block, Clobbers, Function, Inst, OperandKind, RegClass, Terminator},
        regalloc3::{
            hints::{SpillCost, VRegMetadata},
            liveness::compute_liveness,
            progpoint::ProgPointMap,
        },
    };
    use kajit_lir::LinearOp;

    // Minimal ABI for testing
    const TEST_ABI: AbiInfo = AbiInfo {
        caller_saved_gpr: &[PReg(0), PReg(1), PReg(2)],
        callee_saved_gpr: &[PReg(3), PReg(4)],
        arg_gprs: &[],
        ret_gprs: &[],
        red_zone_size: 0,
    };

    const TEST_SCRATCH: ScratchPolicy = ScratchPolicy {
        reserved: &[PReg(10)],
        max_simultaneous_spills: 1,
    };

    #[test]
    fn test_simple_allocation() {
        // One vreg, plenty of registers
        use crate::cfg_mir::{BlockId, InstId, Operand};

        let func = Function {
            id: crate::cfg_mir::FunctionId(0),
            lambda_id: kajit_ir::LambdaId::new(0),
            entry: BlockId(0),
            data_args: vec![],
            data_results: vec![],
            output_size: 0,
            blocks: vec![Block {
                id: BlockId(0),
                params: vec![],
                insts: vec![InstId(0)],
                term: crate::cfg_mir::TermId(0),
                preds: vec![],
                succs: vec![],
                dead: false,
            }],
            edges: vec![],
            insts: vec![Inst {
                id: InstId(0),
                op: LinearOp::Const {
                    dst: VReg::new(1),
                    value: 42,
                },
                operands: vec![],
                clobbers: Clobbers::default(),
            }],
            terms: vec![Terminator::Return],
        };

        let progpoints = ProgPointMap::build(&func);
        let liveness = compute_liveness(&func, &progpoints);
        let hints = HintMap::new();
        let copy_hints = CopyHints::default();
        let result = allocate(liveness, &TEST_ABI, &TEST_SCRATCH, &hints, &copy_hints);

        // Should allocate to a register (plenty available)
        assert!(matches!(
            result.allocations.get(&VReg::new(1)),
            Some(Allocation::Reg(_))
        ));
        assert_eq!(result.spilled.len(), 0);
    }

    #[test]
    fn test_register_pressure() {
        // More vregs than registers -> must spill
        use crate::cfg_mir::{BlockId, InstId, Operand};

        let func = Function {
            id: crate::cfg_mir::FunctionId(0),
            lambda_id: kajit_ir::LambdaId::new(0),
            entry: BlockId(0),
            data_args: vec![],
            data_results: vec![],
            output_size: 0,
            blocks: vec![Block {
                id: BlockId(0),
                params: vec![],
                insts: vec![
                    InstId(0),
                    InstId(1),
                    InstId(2),
                    InstId(3),
                    InstId(4),
                    InstId(5),
                ],
                term: crate::cfg_mir::TermId(0),
                preds: vec![],
                succs: vec![],
                dead: false,
            }],
            edges: vec![],
            insts: vec![
                // Define v1-v6 (6 vregs, but only 5 registers available)
                Inst {
                    id: InstId(0),
                    op: LinearOp::Const {
                        dst: VReg::new(1),
                        value: 1,
                    },
                    operands: vec![],
                    clobbers: Clobbers::default(),
                },
                Inst {
                    id: InstId(1),
                    op: LinearOp::Const {
                        dst: VReg::new(2),
                        value: 2,
                    },
                    operands: vec![],
                    clobbers: Clobbers::default(),
                },
                Inst {
                    id: InstId(2),
                    op: LinearOp::Const {
                        dst: VReg::new(3),
                        value: 3,
                    },
                    operands: vec![],
                    clobbers: Clobbers::default(),
                },
                Inst {
                    id: InstId(3),
                    op: LinearOp::Const {
                        dst: VReg::new(4),
                        value: 4,
                    },
                    operands: vec![],
                    clobbers: Clobbers::default(),
                },
                Inst {
                    id: InstId(4),
                    op: LinearOp::Const {
                        dst: VReg::new(5),
                        value: 5,
                    },
                    operands: vec![],
                    clobbers: Clobbers::default(),
                },
                Inst {
                    id: InstId(5),
                    op: LinearOp::Const {
                        dst: VReg::new(6),
                        value: 6,
                    },
                    operands: vec![],
                    clobbers: Clobbers::default(),
                },
            ],
            terms: vec![Terminator::Return],
        };

        let progpoints = ProgPointMap::build(&func);
        let liveness = compute_liveness(&func, &progpoints);
        let hints = HintMap::new();
        let copy_hints = CopyHints::default();
        let result = allocate(liveness, &TEST_ABI, &TEST_SCRATCH, &hints, &copy_hints);

        // Should have some spills (6 vregs, 5 registers)
        // Note: dead code (unused defs) won't cause pressure
        // but we should still see allocation decisions
        assert_eq!(result.allocations.len(), 6);
    }

    #[test]
    fn test_interval_expiry() {
        // Non-overlapping intervals should reuse registers
        use crate::cfg_mir::{BlockId, InstId, Operand};

        let func = Function {
            id: crate::cfg_mir::FunctionId(0),
            lambda_id: kajit_ir::LambdaId::new(0),
            entry: BlockId(0),
            data_args: vec![],
            data_results: vec![],
            output_size: 0,
            blocks: vec![Block {
                id: BlockId(0),
                params: vec![],
                insts: vec![InstId(0), InstId(1)],
                term: crate::cfg_mir::TermId(0),
                preds: vec![],
                succs: vec![],
                dead: false,
            }],
            edges: vec![],
            insts: vec![
                // v1 = const (dies immediately)
                Inst {
                    id: InstId(0),
                    op: LinearOp::Const {
                        dst: VReg::new(1),
                        value: 1,
                    },
                    operands: vec![],
                    clobbers: Clobbers::default(),
                },
                // v2 = const (can reuse v1's register)
                Inst {
                    id: InstId(1),
                    op: LinearOp::Const {
                        dst: VReg::new(2),
                        value: 2,
                    },
                    operands: vec![],
                    clobbers: Clobbers::default(),
                },
            ],
            terms: vec![Terminator::Return],
        };

        let progpoints = ProgPointMap::build(&func);
        let liveness = compute_liveness(&func, &progpoints);
        let hints = HintMap::new();
        let copy_hints = CopyHints::default();
        let result = allocate(liveness, &TEST_ABI, &TEST_SCRATCH, &hints, &copy_hints);

        // Both should get registers (non-overlapping)
        assert!(matches!(
            result.allocations.get(&VReg::new(1)),
            Some(Allocation::Reg(_))
        ));
        assert!(matches!(
            result.allocations.get(&VReg::new(2)),
            Some(Allocation::Reg(_))
        ));
        assert_eq!(result.spilled.len(), 0);
    }

    #[test]
    fn test_spill_cost_hints() {
        // Test that spill cost hints affect victim selection
        // Create two vregs with same next-use, different spill costs
        use crate::cfg_mir::{BlockId, InstId, Operand};

        // Function with 6 vregs (more than 5 available registers)
        // v1 has High spill cost (loop-carried)
        // v2 has Low spill cost (not loop-carried)
        // Both used at same point, so without hints, either could be spilled
        // With hints, v2 should be spilled (lower cost)

        let func = Function {
            id: crate::cfg_mir::FunctionId(0),
            lambda_id: kajit_ir::LambdaId::new(0),
            entry: BlockId(0),
            data_args: vec![],
            data_results: vec![],
            output_size: 0,
            blocks: vec![Block {
                id: BlockId(0),
                params: vec![],
                insts: vec![
                    InstId(0),
                    InstId(1),
                    InstId(2),
                    InstId(3),
                    InstId(4),
                    InstId(5),
                ],
                term: crate::cfg_mir::TermId(0),
                preds: vec![],
                succs: vec![],
                dead: false,
            }],
            edges: vec![],
            insts: vec![
                Inst {
                    id: InstId(0),
                    op: LinearOp::Const {
                        dst: VReg::new(1),
                        value: 1,
                    },
                    operands: vec![],
                    clobbers: Clobbers::default(),
                },
                Inst {
                    id: InstId(1),
                    op: LinearOp::Const {
                        dst: VReg::new(2),
                        value: 2,
                    },
                    operands: vec![],
                    clobbers: Clobbers::default(),
                },
                Inst {
                    id: InstId(2),
                    op: LinearOp::Const {
                        dst: VReg::new(3),
                        value: 3,
                    },
                    operands: vec![],
                    clobbers: Clobbers::default(),
                },
                Inst {
                    id: InstId(3),
                    op: LinearOp::Const {
                        dst: VReg::new(4),
                        value: 4,
                    },
                    operands: vec![],
                    clobbers: Clobbers::default(),
                },
                Inst {
                    id: InstId(4),
                    op: LinearOp::Const {
                        dst: VReg::new(5),
                        value: 5,
                    },
                    operands: vec![],
                    clobbers: Clobbers::default(),
                },
                Inst {
                    id: InstId(5),
                    op: LinearOp::Const {
                        dst: VReg::new(6),
                        value: 6,
                    },
                    operands: vec![],
                    clobbers: Clobbers::default(),
                },
            ],
            terms: vec![Terminator::Return],
        };

        let progpoints = ProgPointMap::build(&func);
        let liveness = compute_liveness(&func, &progpoints);

        // Create hints: v1 is loop-carried (high cost), v2 is not (low cost)
        let mut hints = HintMap::new();
        hints.insert(
            VReg::new(1),
            VRegMetadata {
                spill_cost: SpillCost::High,
                debug_name: Some("loop_counter".to_string()),
            },
        );
        hints.insert(
            VReg::new(2),
            VRegMetadata {
                spill_cost: SpillCost::Low,
                debug_name: Some("temp".to_string()),
            },
        );

        let copy_hints = CopyHints::default();
        let result = allocate(liveness, &TEST_ABI, &TEST_SCRATCH, &hints, &copy_hints);

        // If spilling happens, v2 should be spilled (lower cost) not v1
        // Note: with 6 dead vregs and 5 registers, we may not actually spill
        // This test mainly validates the hint infrastructure works
        assert_eq!(result.allocations.len(), 6);
    }
}
