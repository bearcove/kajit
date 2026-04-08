//! Constant branch folding for CFG-MIR.
//!
//! Three transforms:
//! 1. **Direct fold**: Replace conditional branches on known-constant values
//!    with unconditional branches.
//! 2. **Single-predecessor propagation**: When a block has one live predecessor,
//!    propagate constant edge args into the const map.
//! 3. **Branch threading**: When a block has no instructions and branches on a
//!    phi, thread predecessor edges that carry a known constant for the phi
//!    directly to the correct successor, bypassing the branch block.

use std::collections::HashMap;

use crate::ir::{EdgeId, Function, Terminator};
use kajit_ir::VReg;
use kajit_lir::LinearOp;

/// Fold constant conditional branches into unconditional branches.
/// Returns the number of branches folded.
pub fn fold_const_branches(func: &mut Function) -> usize {
    let mut total_folded = 0;

    // Build map of vreg → constant value
    let mut const_vals: HashMap<VReg, u64> = HashMap::new();
    for inst in &func.insts {
        if let LinearOp::Const { dst, value } = &inst.op {
            const_vals.insert(*dst, *value);
        }
        if let LinearOp::Copy { dst, src } = &inst.op
            && let Some(&val) = const_vals.get(src)
        {
            const_vals.insert(*dst, val);
        }
    }

    // Propagate constants through single-predecessor edges.
    let mut propagated = true;
    while propagated {
        propagated = false;
        for block in &func.blocks {
            if block.dead {
                continue;
            }
            let live_preds: Vec<_> = block
                .preds
                .iter()
                .filter(|e| !func.blocks[func.edges[e.index()].from.index()].dead)
                .collect();
            if live_preds.len() != 1 {
                continue;
            }
            let edge = &func.edges[live_preds[0].index()];
            for arg in &edge.args {
                if let Some(&val) = const_vals.get(&arg.source)
                    && let std::collections::hash_map::Entry::Vacant(e) =
                        const_vals.entry(arg.target)
                {
                    e.insert(val);
                    propagated = true;
                }
            }
        }
    }

    // Iterate: direct fold may expose new single-predecessor blocks, which
    // enables more propagation, which enables more folds/threads.
    loop {
        let folded = fold_direct_const_branches(func, &const_vals);
        total_folded += folded;
        kill_unreachable(func);

        // Re-propagate after folds may have reduced predecessor counts.
        // Iterate to fixpoint since chains of single-pred blocks may need
        // multiple passes (b4 → b5 → b6).
        let mut new_props = false;
        let mut prop_changed = true;
        while prop_changed {
            prop_changed = false;
            for block in &func.blocks {
                if block.dead {
                    continue;
                }
                let live_preds: Vec<_> = block
                    .preds
                    .iter()
                    .filter(|e| !func.blocks[func.edges[e.index()].from.index()].dead)
                    .collect();
                if live_preds.len() != 1 {
                    continue;
                }
                let edge = &func.edges[live_preds[0].index()];
                for arg in &edge.args {
                    if let Some(&val) = const_vals.get(&arg.source)
                        && let std::collections::hash_map::Entry::Vacant(e) =
                            const_vals.entry(arg.target)
                    {
                        e.insert(val);
                        prop_changed = true;
                        new_props = true;
                    }
                }
            }
        }

        if folded == 0 && !new_props {
            break;
        }
    }

    total_folded
}

/// Phase 1: Replace conditional branches on known constants with unconditional.
fn fold_direct_const_branches(func: &mut Function, const_vals: &HashMap<VReg, u64>) -> usize {
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

    let term_to_block: HashMap<usize, usize> = func
        .blocks
        .iter()
        .enumerate()
        .map(|(bi, b)| (b.term.0 as usize, bi))
        .collect();

    for (term_idx, kept_edge, dead_edge) in folds {
        func.terms[term_idx] = Terminator::Branch { edge: kept_edge };

        if let Some(&block_idx) = term_to_block.get(&term_idx) {
            func.blocks[block_idx].succs.retain(|e| *e != dead_edge);
        }

        let dead_target = func.edges[dead_edge.index()].to;
        func.blocks[dead_target.index()]
            .preds
            .retain(|e| *e != dead_edge);
    }

    folded
}

/// Transitively mark unreachable blocks as dead.
fn kill_unreachable(func: &mut Function) {
    let mut changed = true;
    while changed {
        changed = false;
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
}
