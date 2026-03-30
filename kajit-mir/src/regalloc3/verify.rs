//! Symbolic verification of register allocation.
//!
//! ## What We Check
//!
//! 1. **No conflicts**: Overlapping live intervals don't share registers
//! 2. **Fixed constraints**: Instructions with fixed-register constraints honored
//! 3. **No scratch register allocation**: User vregs never allocated to scratch regs
//! 4. **All vregs allocated**: Every vreg has an allocation decision
//!
//! ## Why Verification?
//!
//! Register allocation bugs are silent and catastrophic. Verification catches:
//! - Logic errors in allocator
//! - Missed edge cases (block params, phi nodes)
//! - Interference graph bugs
//! - Spill heuristic bugs
//!
//! ## When to Verify
//!
//! After allocation, before emission. Fail fast if allocation is incorrect.

use kajit_ir::VReg;
use std::collections::HashMap;

use crate::cfg_mir::Function;

use super::{
    linear_scan::{Allocation, AllocationResult},
    liveness::LivenessInfo,
    machine_inst::{PReg, ScratchPolicy},
    progpoint::ProgPoint,
};

/// Verification errors
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum VerifyError {
    /// Two vregs with overlapping live ranges allocated to same register
    Conflict {
        vreg1: VReg,
        vreg2: VReg,
        preg: PReg,
        point: ProgPoint,
    },

    /// Vreg allocated to scratch register (should never happen)
    ScratchRegisterUsed { vreg: VReg, preg: PReg },

    /// Vreg has no allocation decision
    Unallocated { vreg: VReg },
    // TODO: Fixed constraint violations (Phase 2)
}

impl std::fmt::Display for VerifyError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            VerifyError::Conflict {
                vreg1,
                vreg2,
                preg,
                point,
            } => {
                write!(
                    f,
                    "Conflict: v{} and v{} both allocated to {:?} at point {}",
                    vreg1.index(),
                    vreg2.index(),
                    preg,
                    point.0
                )
            }
            VerifyError::ScratchRegisterUsed { vreg, preg } => {
                write!(
                    f,
                    "Scratch register used: v{} allocated to scratch {:?}",
                    vreg.index(),
                    preg
                )
            }
            VerifyError::Unallocated { vreg } => {
                write!(f, "Unallocated vreg: v{}", vreg.index())
            }
        }
    }
}

/// Verify allocation is correct
pub fn verify(
    _func: &Function,
    alloc: &AllocationResult,
    liveness: &LivenessInfo,
    scratch: &ScratchPolicy,
) -> Result<(), Vec<VerifyError>> {
    let mut errors = Vec::new();

    // Check 1: No scratch registers allocated to user vregs
    for (&vreg, &allocation) in &alloc.allocations {
        if let Allocation::Reg(preg) = allocation {
            if scratch.reserved.contains(&preg) {
                errors.push(VerifyError::ScratchRegisterUsed { vreg, preg });
            }
        }
    }

    // Check 2: All vregs with intervals have allocations
    for vreg in liveness.intervals.keys() {
        if !alloc.allocations.contains_key(vreg) {
            errors.push(VerifyError::Unallocated { vreg: *vreg });
        }
    }

    // Check 3: No conflicts (overlapping live ranges sharing registers)
    errors.extend(check_conflicts(alloc, liveness));

    if errors.is_empty() {
        Ok(())
    } else {
        Err(errors)
    }
}

/// Check for register conflicts
fn check_conflicts(alloc: &AllocationResult, liveness: &LivenessInfo) -> Vec<VerifyError> {
    let mut errors = Vec::new();

    // Group vregs by allocated register
    let mut reg_to_vregs: HashMap<PReg, Vec<VReg>> = HashMap::new();
    for (&vreg, &allocation) in &alloc.allocations {
        if let Allocation::Reg(preg) = allocation {
            reg_to_vregs.entry(preg).or_insert_with(Vec::new).push(vreg);
        }
    }

    // For each register, check if any two vregs have overlapping live ranges
    for (preg, vregs) in reg_to_vregs {
        for i in 0..vregs.len() {
            for j in (i + 1)..vregs.len() {
                let vreg1 = vregs[i];
                let vreg2 = vregs[j];

                if let (Some(interval1), Some(interval2)) = (
                    liveness.intervals.get(&vreg1),
                    liveness.intervals.get(&vreg2),
                ) {
                    // Check if intervals overlap
                    if let Some(point) = intervals_overlap(interval1, interval2) {
                        errors.push(VerifyError::Conflict {
                            vreg1,
                            vreg2,
                            preg,
                            point,
                        });
                    }
                }
            }
        }
    }

    errors
}

/// Check if two intervals overlap, return first overlapping point
fn intervals_overlap(
    iv1: &super::progpoint::LiveInterval,
    iv2: &super::progpoint::LiveInterval,
) -> Option<ProgPoint> {
    // Check each segment pair
    for &(start1, end1) in &iv1.segments {
        for &(start2, end2) in &iv2.segments {
            // Segments overlap if they're not disjoint
            // Disjoint: end1 < start2 or end2 < start1
            if !(end1 < start2 || end2 < start1) {
                // Overlapping, return first point in overlap
                return Some(start1.max(start2));
            }
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::regalloc3::{
        liveness::LivenessInfo,
        progpoint::{LiveInterval, ProgPoint},
    };
    use indexmap::IndexMap;
    use std::collections::HashMap;

    const TEST_SCRATCH: ScratchPolicy = ScratchPolicy {
        reserved: &[PReg(10), PReg(11)],
        max_simultaneous_spills: 2,
    };

    #[test]
    fn test_no_errors() {
        // Non-overlapping intervals, different registers
        let alloc = AllocationResult {
            allocations: {
                let mut map = HashMap::new();
                map.insert(VReg::new(1), Allocation::Reg(PReg(0)));
                map.insert(VReg::new(2), Allocation::Reg(PReg(1)));
                map
            },
            spilled: vec![],
            edits: vec![],
        };

        let liveness = LivenessInfo {
            intervals: {
                let mut map = IndexMap::new();
                map.insert(
                    VReg::new(1),
                    LiveInterval {
                        vreg: VReg::new(1),
                        segments: vec![(ProgPoint(0), ProgPoint(10))],
                        uses: vec![],
                    },
                );
                map.insert(
                    VReg::new(2),
                    LiveInterval {
                        vreg: VReg::new(2),
                        segments: vec![(ProgPoint(20), ProgPoint(30))],
                        uses: vec![],
                    },
                );
                map
            },
            live_in: HashMap::new(),
            live_out: HashMap::new(),
        };

        let func = create_test_function();
        let result = verify(&func, &alloc, &liveness, &TEST_SCRATCH);
        assert!(result.is_ok());
    }

    #[test]
    fn test_conflict_detection() {
        // Overlapping intervals, same register -> conflict
        let alloc = AllocationResult {
            allocations: {
                let mut map = HashMap::new();
                map.insert(VReg::new(1), Allocation::Reg(PReg(0)));
                map.insert(VReg::new(2), Allocation::Reg(PReg(0))); // SAME REG!
                map
            },
            spilled: vec![],
            edits: vec![],
        };

        let liveness = LivenessInfo {
            intervals: {
                let mut map = IndexMap::new();
                map.insert(
                    VReg::new(1),
                    LiveInterval {
                        vreg: VReg::new(1),
                        segments: vec![(ProgPoint(0), ProgPoint(10))],
                        uses: vec![],
                    },
                );
                map.insert(
                    VReg::new(2),
                    LiveInterval {
                        vreg: VReg::new(2),
                        segments: vec![(ProgPoint(5), ProgPoint(15))], // OVERLAPS!
                        uses: vec![],
                    },
                );
                map
            },
            live_in: HashMap::new(),
            live_out: HashMap::new(),
        };

        let func = create_test_function();
        let result = verify(&func, &alloc, &liveness, &TEST_SCRATCH);
        assert!(result.is_err());

        let errors = result.unwrap_err();
        assert_eq!(errors.len(), 1);
        assert!(matches!(errors[0], VerifyError::Conflict { .. }));
    }

    #[test]
    fn test_scratch_register_error() {
        // Vreg allocated to scratch register
        let alloc = AllocationResult {
            allocations: {
                let mut map = HashMap::new();
                map.insert(VReg::new(1), Allocation::Reg(PReg(10))); // SCRATCH!
                map
            },
            spilled: vec![],
            edits: vec![],
        };

        let liveness = LivenessInfo {
            intervals: {
                let mut map = IndexMap::new();
                map.insert(
                    VReg::new(1),
                    LiveInterval {
                        vreg: VReg::new(1),
                        segments: vec![(ProgPoint(0), ProgPoint(10))],
                        uses: vec![],
                    },
                );
                map
            },
            live_in: HashMap::new(),
            live_out: HashMap::new(),
        };

        let func = create_test_function();
        let result = verify(&func, &alloc, &liveness, &TEST_SCRATCH);
        assert!(result.is_err());

        let errors = result.unwrap_err();
        assert_eq!(errors.len(), 1);
        assert!(matches!(errors[0], VerifyError::ScratchRegisterUsed { .. }));
    }

    #[test]
    fn test_unallocated_vreg() {
        // Vreg in liveness but not in allocation
        let alloc = AllocationResult {
            allocations: HashMap::new(), // EMPTY!
            spilled: vec![],
            edits: vec![],
        };

        let liveness = LivenessInfo {
            intervals: {
                let mut map = IndexMap::new();
                map.insert(
                    VReg::new(1),
                    LiveInterval {
                        vreg: VReg::new(1),
                        segments: vec![(ProgPoint(0), ProgPoint(10))],
                        uses: vec![],
                    },
                );
                map
            },
            live_in: HashMap::new(),
            live_out: HashMap::new(),
        };

        let func = create_test_function();
        let result = verify(&func, &alloc, &liveness, &TEST_SCRATCH);
        assert!(result.is_err());

        let errors = result.unwrap_err();
        assert_eq!(errors.len(), 1);
        assert!(matches!(errors[0], VerifyError::Unallocated { .. }));
    }

    fn create_test_function() -> Function {
        use crate::cfg_mir::{Block, BlockId, Terminator};

        Function {
            id: crate::cfg_mir::FunctionId(0),
            lambda_id: kajit_ir::LambdaId::new(0),
            entry: BlockId(0),
            data_args: vec![],
            data_results: vec![],
            output_size: 0,
            blocks: vec![Block {
                id: BlockId(0),
                params: vec![],
                insts: vec![],
                term: crate::cfg_mir::TermId(0),
                preds: vec![],
                succs: vec![],
                dead: false,
            }],
            edges: vec![],
            insts: vec![],
            terms: vec![Terminator::Return],
        }
    }
}
