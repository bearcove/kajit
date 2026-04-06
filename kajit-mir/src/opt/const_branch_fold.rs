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

use crate::cfg_mir::{EdgeArg, EdgeId, Function, Terminator};
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

/// Phase 2: Thread predecessor edges past instruction-free branch blocks.
///
/// When block B has:
/// - No instructions (only a branch terminator)
/// - A conditional branch on a block param v_cond
/// - A predecessor edge E where the source for v_cond is a known constant
///
/// Retarget E to bypass B, going directly to the correct successor of B.
fn thread_const_phi_branches(func: &mut Function, const_vals: &HashMap<VReg, u64>) -> usize {
    let mut threaded = 0;

    // Iterate blocks looking for threadable patterns.
    // We need to be careful about mutation, so collect targets first.
    let mut threads: Vec<ThreadAction> = Vec::new();

    for (block_idx, block) in func.blocks.iter().enumerate() {
        if block.dead {
            continue;
        }
        // Block must have no instructions.
        if !block.insts.is_empty() {
            continue;
        }

        let term = &func.terms[block.term.0 as usize];
        let (cond_vreg, taken_edge, fallthrough_edge) = match term {
            Terminator::BranchIfZero {
                cond,
                taken,
                fallthrough,
            } => (*cond, *taken, *fallthrough),
            Terminator::BranchIf {
                cond,
                taken,
                fallthrough,
            } => (*cond, *taken, *fallthrough),
            _ => continue,
        };

        // cond_vreg must be a block param of this block.
        if !block.params.contains(&cond_vreg) {
            continue;
        }

        // Check each predecessor edge.
        for &pred_edge_id in &block.preds {
            let pred_edge = &func.edges[pred_edge_id.index()];
            if func.blocks[pred_edge.from.index()].dead {
                continue;
            }

            // Find the edge arg that targets cond_vreg.
            let cond_arg = pred_edge.args.iter().find(|a| a.target == cond_vreg);
            let Some(cond_arg) = cond_arg else {
                continue;
            };

            // Check if the source is a known constant.
            let Some(&val) = const_vals.get(&cond_arg.source) else {
                continue;
            };

            // Determine which successor edge is taken.
            let is_branch_if_zero = matches!(term, Terminator::BranchIfZero { .. });
            let target_edge = if is_branch_if_zero {
                if val == 0 {
                    taken_edge
                } else {
                    fallthrough_edge
                }
            } else {
                // BranchIf: taken when non-zero
                if val != 0 {
                    taken_edge
                } else {
                    fallthrough_edge
                }
            };

            // Verify all successor edge arg sources are block params of B.
            let succ_edge = &func.edges[target_edge.index()];
            let all_sources_are_params = succ_edge
                .args
                .iter()
                .all(|arg| block.params.contains(&arg.source));
            if !all_sources_are_params {
                continue;
            }

            threads.push(ThreadAction {
                pred_edge_id,
                branch_block_idx: block_idx,
                target_succ_edge: target_edge,
            });
        }
    }

    // Apply threading actions.
    for action in threads {
        let succ_edge = &func.edges[action.target_succ_edge.index()];
        let new_target_block = succ_edge.to;
        let succ_args = succ_edge.args.clone();

        let branch_block = &func.blocks[action.branch_block_idx];
        let _branch_params = branch_block.params.clone();

        let pred_edge = &func.edges[action.pred_edge_id.index()];
        let pred_args = pred_edge.args.clone();

        // Build param → source mapping from the predecessor edge.
        let param_to_source: HashMap<VReg, VReg> =
            pred_args.iter().map(|a| (a.target, a.source)).collect();

        // Compose edge args: for each successor edge arg, substitute the
        // source vreg through the branch block's param mapping.
        // If any source can't be resolved, skip this thread (safety check).
        let mut new_args: Vec<EdgeArg> = Vec::new();
        let mut safe = true;
        for arg in &succ_args {
            if let Some(&from_pred) = param_to_source.get(&arg.source) {
                new_args.push(EdgeArg {
                    target: arg.target,
                    source: from_pred,
                });
            } else {
                // Source is not a block param of B — can't thread safely.
                safe = false;
                break;
            }
        }
        if !safe {
            continue;
        }

        // Retarget the predecessor edge to the successor's target.
        let pred_edge_mut = &mut func.edges[action.pred_edge_id.index()];
        pred_edge_mut.to = new_target_block;
        pred_edge_mut.args = new_args;

        // Remove the predecessor edge from the branch block's preds.
        func.blocks[action.branch_block_idx]
            .preds
            .retain(|e| *e != action.pred_edge_id);

        // Add the predecessor edge to the new target block's preds.
        func.blocks[new_target_block.index()]
            .preds
            .push(action.pred_edge_id);

        // Update the predecessor block's succs: the edge ID stays the same,
        // so no change needed there.

        threaded += 1;
    }

    threaded
}

struct ThreadAction {
    pred_edge_id: EdgeId,
    branch_block_idx: usize,
    target_succ_edge: EdgeId,
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
