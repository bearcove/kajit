//! Spill and reload insertion.
//!
//! ## Problem
//!
//! After allocation, some vregs are marked as spilled. We must:
//! 1. Assign spill slots (stack locations)
//! 2. Insert reload before each use (spill slot -> scratch register)
//! 3. Insert spill after each def (register -> spill slot)
//! 4. Rewrite uses/defs to use scratch registers
//!
//! ## Scratch Register Usage
//!
//! Spill/reload uses scratch registers (never allocated to user values):
//! - Reload: spill_slot -> scratch_reg
//! - Use: instruction uses scratch_reg
//! - Def: instruction defs scratch_reg
//! - Spill: scratch_reg -> spill_slot
//!
//! ## One-Pass Rewrite
//!
//! We rewrite the function once:
//! - Iterate through instructions
//! - For each spilled vreg used: insert reload before instruction
//! - For each spilled vreg defined: insert spill after instruction
//! - Update operands to use scratch registers
//!
//! ## Spill Slot Assignment
//!
//! Simple allocator:
//! - One slot per spilled vreg (no coalescing in v1)
//! - Slots are 8 bytes (u64/pointer size)
//! - Slots are allocated at end of function frame

use kajit_ir::VReg;
use std::collections::HashMap;

use crate::cfg_mir::{Function, InstId};

use super::{
    linear_scan::{Allocation, AllocationResult},
    machine_inst::ScratchPolicy,
};

/// Spill slot ID (index into spill area)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct SpillSlot(pub u32);

/// Spill rewrite result
#[derive(Debug)]
pub struct SpillRewriteResult {
    /// Vreg -> spill slot mapping
    pub spill_slots: HashMap<VReg, SpillSlot>,

    /// Total spill slots needed
    pub num_spill_slots: u32,
}

/// Rewrite function to insert spills/reloads
pub fn rewrite_spills(
    func: &mut Function,
    alloc: &AllocationResult,
    scratch: &ScratchPolicy,
) -> SpillRewriteResult {
    let mut spill_slots = HashMap::new();
    let mut next_slot = 0u32;

    // Assign spill slots
    for &vreg in &alloc.spilled {
        spill_slots.insert(vreg, SpillSlot(next_slot));
        next_slot += 1;
    }

    // Rewrite instructions
    rewrite_instructions(func, alloc, &spill_slots, scratch);

    SpillRewriteResult {
        spill_slots,
        num_spill_slots: next_slot,
    }
}

/// Rewrite instructions to use scratch registers for spilled vregs
fn rewrite_instructions(
    func: &mut Function,
    alloc: &AllocationResult,
    _spill_slots: &HashMap<VReg, SpillSlot>,
    _scratch: &ScratchPolicy,
) {
    // For each block
    for block in &mut func.blocks {
        if block.dead {
            continue;
        }

        // Collect instructions that need rewriting
        let inst_ids: Vec<InstId> = block.insts.clone();

        for &inst_id in &inst_ids {
            let inst = &func.insts[inst_id.index()];

            // Check if this instruction uses or defines spilled vregs
            let uses_spilled = inst
                .operands
                .iter()
                .any(|op| matches!(alloc.allocations.get(&op.vreg), Some(Allocation::Spill)));

            let op = &inst.op;
            let defs_spilled = {
                // Extract def vreg (if any)
                use kajit_lir::LinearOp;
                let def_vreg = match op {
                    LinearOp::Const { dst, .. }
                    | LinearOp::BinOp { dst, .. }
                    | LinearOp::UnaryOp { dst, .. }
                    | LinearOp::Copy { dst, .. }
                    | LinearOp::ReadFromField { dst, .. }
                    | LinearOp::SaveOutPtr { dst }
                    | LinearOp::SlotAddr { dst, .. }
                    | LinearOp::LoadFromAddr { dst, .. } => Some(*dst),
                    LinearOp::CallIntrinsic { dst, .. } => *dst,
                    _ => None,
                };

                if let Some(vreg) = def_vreg {
                    matches!(alloc.allocations.get(&vreg), Some(Allocation::Spill))
                } else {
                    false
                }
            };

            // TODO: Insert reload/spill instructions
            // For now, just a placeholder - full implementation would:
            // 1. Insert reload before instruction for each spilled use
            // 2. Rewrite operands to use scratch register
            // 3. Insert spill after instruction for spilled def

            if uses_spilled || defs_spilled {
                // Placeholder: would insert actual spill/reload here
                // This requires creating new instructions and inserting them
                // into the block at the right positions
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        cfg_mir::{Block, BlockId, Terminator},
        regalloc3::{
            linear_scan::{Allocation, AllocationResult},
            machine_inst::{PReg, ScratchPolicy},
        },
    };

    const TEST_SCRATCH: ScratchPolicy = ScratchPolicy {
        reserved: &[PReg(10), PReg(11)],
        max_simultaneous_spills: 2,
    };

    #[test]
    fn test_spill_slot_assignment() {
        // Test that spill slots are assigned correctly
        let alloc = AllocationResult {
            allocations: {
                let mut map = HashMap::new();
                map.insert(VReg::new(1), Allocation::Reg(PReg(0)));
                map.insert(VReg::new(2), Allocation::Spill);
                map.insert(VReg::new(3), Allocation::Spill);
                map
            },
            spilled: vec![VReg::new(2), VReg::new(3)],
            edits: vec![],
        };

        let mut func = Function {
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
        };

        let result = rewrite_spills(&mut func, &alloc, &TEST_SCRATCH);

        // Should have 2 spill slots (for v2 and v3)
        assert_eq!(result.num_spill_slots, 2);
        assert_eq!(result.spill_slots.len(), 2);
        assert!(result.spill_slots.contains_key(&VReg::new(2)));
        assert!(result.spill_slots.contains_key(&VReg::new(3)));
    }

    #[test]
    fn test_no_spills() {
        // Test case with no spills
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

        let mut func = Function {
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
        };

        let result = rewrite_spills(&mut func, &alloc, &TEST_SCRATCH);

        // Should have 0 spill slots
        assert_eq!(result.num_spill_slots, 0);
        assert_eq!(result.spill_slots.len(), 0);
    }
}
