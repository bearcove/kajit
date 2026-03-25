//! Constant branch folding for CFG-MIR.
//!
//! Replaces conditional branches on known-constant values with unconditional branches:
//! - `branch_if_zero <const(0)>` → branch to taken target
//! - `branch_if_zero <const(N)>` (N≠0) → branch to fallthrough target
//! - `branch_if <const(0)>` → branch to fallthrough target
//! - `branch_if <const(N)>` (N≠0) → branch to taken target
//!
//! Also updates block succs/preds and marks dead edges.

use std::collections::HashMap;

use crate::cfg_mir::{EdgeId, Function, Terminator};
use kajit_ir::VReg;
use kajit_lir::LinearOp;

/// Fold constant conditional branches into unconditional branches.
/// Returns the number of branches folded.
pub fn fold_const_branches(func: &mut Function) -> usize {
    // Build map of vreg → constant value
    let mut const_vals: HashMap<VReg, u64> = HashMap::new();
    for inst in &func.insts {
        if let LinearOp::Const { dst, value } = &inst.op {
            const_vals.insert(*dst, *value);
        }
    }

    // Collect folds: (term_idx, kept_edge, dead_edge)
    let mut folds: Vec<(usize, EdgeId, EdgeId)> = Vec::new();

    for term_idx in 0..func.terms.len() {
        let fold = match &func.terms[term_idx] {
            Terminator::BranchIfZero {
                cond,
                taken,
                fallthrough,
            } => const_vals.get(cond).map(|&val| {
                if val == 0 {
                    (term_idx, *taken, *fallthrough)
                } else {
                    (term_idx, *fallthrough, *taken)
                }
            }),
            Terminator::BranchIf {
                cond,
                taken,
                fallthrough,
            } => const_vals.get(cond).map(|&val| {
                if val != 0 {
                    (term_idx, *taken, *fallthrough)
                } else {
                    (term_idx, *fallthrough, *taken)
                }
            }),
            _ => None,
        };

        if let Some(f) = fold {
            folds.push(f);
        }
    }

    let folded = folds.len();

    // Build term_id → block_id map
    let term_to_block: HashMap<usize, usize> = func
        .blocks
        .iter()
        .enumerate()
        .map(|(bi, b)| (b.term.0 as usize, bi))
        .collect();

    for (term_idx, kept_edge, dead_edge) in folds {
        // Replace terminator
        func.terms[term_idx] = Terminator::Branch { edge: kept_edge };

        // Update block succs: remove the dead edge
        if let Some(&block_idx) = term_to_block.get(&term_idx) {
            func.blocks[block_idx].succs.retain(|e| *e != dead_edge);
        }

        // Remove this block from the dead edge target's preds
        let dead_target = func.edges[dead_edge.index()].to;
        let _dead_source = func.edges[dead_edge.index()].from;
        func.blocks[dead_target.index()]
            .preds
            .retain(|e| *e != dead_edge);
    }

    // Transitively mark unreachable blocks as dead
    let mut changed = true;
    while changed {
        changed = false;
        // Collect blocks to mark dead (can't mutate while iterating)
        let to_kill: Vec<usize> = func
            .blocks
            .iter()
            .enumerate()
            .filter(|(_, b)| {
                !b.dead
                    && b.id != func.entry
                    && b.preds.iter().all(|e| {
                        let from = func.edges[e.index()].from;
                        func.blocks[from.index()].dead
                    })
            })
            .map(|(i, _)| i)
            .collect();

        for bi in to_kill {
            func.blocks[bi].dead = true;
            // Remove this block from successors' pred lists
            let succs: Vec<_> = func.blocks[bi].succs.clone();
            let dead_block_id = func.blocks[bi].id;
            for edge_id in succs {
                let target = func.edges[edge_id.index()].to;
                func.blocks[target.index()]
                    .preds
                    .retain(|e| func.edges[e.index()].from != dead_block_id);
            }
            changed = true;
        }
    }

    folded
}
