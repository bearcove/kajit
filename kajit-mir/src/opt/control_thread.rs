//! Control threading: forward trampolines and thread constant-carrying edges
//! past conditional branches on block params.
//!
//! Targets the "break-as-value" pattern from RVSDG lowering where a loop break
//! materializes a const(0) stop flag that flows through a join block to one or
//! more conditional branches. This pass:
//!
//! 1. **Forwards trampolines**: empty blocks (no instructions) with an unconditional
//!    branch are collapsed — predecessors are redirected to the final target with
//!    composed edge args. Safe when block params are either used in outgoing edge
//!    arg sources (composable) or not used outside the block (dead).
//!    Non-param sources in outgoing edges are fine: they dominate the trampoline
//!    and hence all its predecessors.
//!
//! 2. **Threads break-direction constant edges**: when a predecessor edge provides
//!    const(0) for a conditional branch's condition (a block param), that edge is
//!    retargeted directly to the correct successor, bypassing the branch block.
//!
//! # Design constraints
//!
//! **Break-only threading.** Only the break direction (val=0) is threaded. This
//! is sufficient because after threading, the branch block becomes single-
//! predecessor (only the continue edge remains). The post-simplify `const_branch_fold`
//! pass then propagates the constant through the single-pred edge and folds the
//! branch into an unconditional jump — eliminating the continue-side test for free.
//!
//! **Source block must remain alive.** Threading only one edge ensures the branch
//! block keeps at least one predecessor, preserving its block params as valid SSA
//! definitions. This is critical because param lifting adds `source: leaking_param`
//! to other predecessor edges — if the source block died, those references would
//! dangle. Threading both edges would kill the block and require full SSA
//! reconstruction (iterated phi insertion), which is out of scope.
//!
//! **Param lifting.** When a branch block has params that reach downstream blocks
//! through dominance rather than explicit edge args ("leaking params"), the pass
//! lifts them into successor edge args before threading. This makes the values
//! composable through the threaded edge. Lifting is only attempted when all
//! leaking params are exclusively used in the block's direct successor blocks
//! (instructions, terminators, or edge args) — multi-hop dominance leaks are
//! rejected to avoid requiring a global SSA updater.

use std::collections::{HashMap, HashSet};

use crate::cfg_mir::{BlockId, EdgeArg, EdgeId, Function, Terminator};
use kajit_ir::VReg;
use kajit_lir::LinearOp;

/// Run control threading. Returns true if any changes were made.
pub fn control_thread(func: &mut Function, vreg_count: &mut u32) -> bool {
    let debug = std::env::var("KAJIT_DEBUG_CONTROL_THREAD").is_ok();
    let mut changed = false;
    let mut threaded_blocks: HashSet<BlockId> = HashSet::new();

    if debug {
        eprintln!(
            "[control_thread] starting: {} blocks, {} edges",
            func.blocks.len(),
            func.edges.len()
        );
        // Dump edges with args for the inner loop region (b7-b19 area)
        for edge in &func.edges {
            let args_str: Vec<String> = edge
                .args
                .iter()
                .map(|a| {
                    if a.target == a.source {
                        format!("v{}", a.source.index())
                    } else {
                        format!("v{}=>v{}", a.target.index(), a.source.index())
                    }
                })
                .collect();
            eprintln!(
                "  e{}: b{} -> b{} [{}]",
                edge.id.index(),
                edge.from.index(),
                edge.to.index(),
                args_str.join(", ")
            );
        }
    }

    loop {
        let mut round_changed = false;

        // Phase 1: Forward trampolines to expose constant edges at branch blocks.
        loop {
            let forwarded = forward_one_trampoline(func, debug);
            if !forwarded {
                break;
            }
            round_changed = true;
        }

        // Phase 2: Thread ONE constant-carrying edge past a conditional branch.
        // Skip blocks already threaded to preserve the remaining edge path
        // that keeps the block alive (and its params defined).
        let const_vals = build_const_map(func);
        if let Some(block_id) =
            thread_one_const_edge(func, &const_vals, &threaded_blocks, vreg_count, debug)
        {
            threaded_blocks.insert(block_id);
            round_changed = true;
        }

        if !round_changed {
            break;
        }
        changed = true;
    }

    if changed {
        kill_unreachable(func);
        crate::opt::block_merge::remove_unreachable_blocks(func);
        func.gc_edges();
    }

    changed
}

/// Build a map of vreg → known constant value from Const and Copy instructions.
fn build_const_map(func: &Function) -> HashMap<VReg, u64> {
    let mut const_vals = HashMap::new();
    for inst in &func.insts {
        if let LinearOp::Const { dst, value } = &inst.op {
            const_vals.insert(*dst, *value);
        }
        if let LinearOp::Copy { dst, src } = &inst.op {
            if let Some(&val) = const_vals.get(src) {
                const_vals.insert(*dst, val);
            }
        }
    }
    const_vals
}

/// Forward one trampoline block (empty, unconditional branch, all outgoing edge
/// arg sources are block params). Returns true if a trampoline was forwarded.
fn forward_one_trampoline(func: &mut Function, debug: bool) -> bool {
    let candidate = func.blocks.iter().find_map(|b| {
        if b.dead || b.preds.is_empty() || b.id == func.entry {
            return None;
        }
        if !b.insts.is_empty() {
            return None;
        }
        let term = &func.terms[b.term.index()];
        let Terminator::Branch { edge } = term else {
            return None;
        };

        // Safety: all block params must be either used in outgoing edge args
        // (so they get composed through) or not used outside the block.
        // Non-param sources in outgoing edge args are fine — they dominate
        // the trampoline and hence all its predecessors.
        let out_args = &func.edges[edge.index()].args;
        if !b.params.is_empty() {
            let out_arg_sources: HashSet<VReg> = out_args.iter().map(|a| a.source).collect();
            let safe = b.params.iter().all(|p| {
                out_arg_sources.contains(p) || !is_vreg_used_outside_block(func, *p, b.id)
            });
            if !safe {
                return None;
            }
        }

        Some((b.id, *edge))
    });

    let Some((trampoline_id, out_edge_id)) = candidate else {
        return false;
    };

    let trampoline = &func.blocks[trampoline_id.index()];
    let trampoline_params = trampoline.params.clone();
    let pred_edges: Vec<EdgeId> = trampoline.preds.clone();
    let final_target = func.edges[out_edge_id.index()].to;
    let out_args = func.edges[out_edge_id.index()].args.clone();

    if debug {
        eprintln!(
            "[control_thread] forwarding trampoline b{} -> b{} ({} preds)",
            trampoline_id.index(),
            final_target.index(),
            pred_edges.len()
        );
    }

    // Retarget each predecessor edge to the final target with composed args.
    for &pred_edge_id in &pred_edges {
        let pred_incoming = func.edges[pred_edge_id.index()].args.clone();

        // Build param → source substitution from this predecessor's edge args.
        let subst: HashMap<VReg, VReg> = trampoline_params
            .iter()
            .zip(pred_incoming.iter())
            .map(|(param, arg)| (*param, arg.source))
            .collect();

        // Compose outgoing edge args through substitution.
        let mut composed_args = out_args.clone();
        for arg in &mut composed_args {
            if let Some(&replacement) = subst.get(&arg.source) {
                arg.source = replacement;
            }
        }

        func.edges[pred_edge_id.index()].to = final_target;
        func.edges[pred_edge_id.index()].args = composed_args;
        func.blocks[final_target.index()].preds.push(pred_edge_id);
    }

    // Remove the trampoline's outgoing edge from the final target's preds.
    func.blocks[final_target.index()]
        .preds
        .retain(|e| *e != out_edge_id);

    // Mark trampoline as dead (keep succs intact for terminator consistency).
    func.blocks[trampoline_id.index()].preds.clear();
    func.blocks[trampoline_id.index()].dead = true;

    true
}

/// Collected info for one threading action.
struct ThreadAction {
    pred_edge_id: EdgeId,
    block_id: BlockId,
    target_edge: EdgeId,
    leaking_params: Vec<VReg>,
}

/// Thread one constant-carrying predecessor edge past a conditional branch block.
/// The branch block must have no instructions and branch on a block param.
/// Returns true if an edge was threaded.
fn thread_one_const_edge(
    func: &mut Function,
    const_vals: &HashMap<VReg, u64>,
    skip_blocks: &HashSet<BlockId>,
    vreg_count: &mut u32,
    debug: bool,
) -> Option<BlockId> {
    // Phase 1: Find a threadable action (read-only scan)
    let action = find_thread_action(func, const_vals, skip_blocks);
    let Some(action) = action else {
        return None;
    };

    if debug {
        let succ_to = func.edges[action.target_edge.index()].to;
        eprintln!(
            "[control_thread] threading e{} past b{} -> b{}, leaking={:?}",
            action.pred_edge_id.index(),
            action.block_id.index(),
            succ_to.index(),
            action
                .leaking_params
                .iter()
                .map(|v| v.index())
                .collect::<Vec<_>>(),
        );
    }

    // Phase 2: Lift leaking params (mutates func)
    if !action.leaking_params.is_empty() {
        lift_leaking_params(
            func,
            action.block_id,
            &action.leaking_params,
            vreg_count,
            debug,
        );
    }

    // Phase 3: Thread the edge (mutates func)
    let block_id = action.block_id;
    thread_edge(
        func,
        action.pred_edge_id,
        action.block_id,
        action.target_edge,
    );
    Some(block_id)
}

/// Find a threadable action (read-only).
fn find_thread_action(
    func: &Function,
    const_vals: &HashMap<VReg, u64>,
    skip_blocks: &HashSet<BlockId>,
) -> Option<ThreadAction> {
    for block in &func.blocks {
        if block.dead || !block.insts.is_empty() || skip_blocks.contains(&block.id) {
            continue;
        }

        let term = &func.terms[block.term.index()];
        let (cond, taken_edge, fallthrough_edge, is_zero_test) = match term {
            Terminator::BranchIfZero {
                cond,
                taken,
                fallthrough,
            } => (*cond, *taken, *fallthrough, true),
            Terminator::BranchIf {
                cond,
                taken,
                fallthrough,
            } => (*cond, *taken, *fallthrough, false),
            _ => continue,
        };

        if !block.params.contains(&cond) {
            continue;
        }

        // Find leaking params: a param leaks if ANY successor block uses it
        // without receiving it through that edge's args.
        let leaking_params: Vec<VReg> = block
            .params
            .iter()
            .filter(|p| {
                // Check each successor edge individually: if any successor
                // doesn't carry this param via edge args but uses it, it leaks.
                for &edge_id in &block.succs {
                    let edge_sources: HashSet<VReg> = func.edges[edge_id.index()]
                        .args
                        .iter()
                        .map(|a| a.source)
                        .collect();
                    if !edge_sources.contains(p) {
                        let target = func.edges[edge_id.index()].to;
                        if is_vreg_used_outside_block(func, **p, block.id) {
                            return true;
                        }
                        let _ = target;
                    }
                }
                false
            })
            .copied()
            .collect();

        // Only allow threading if ALL leaking params are exclusively used
        // in direct successor blocks (where lifting can handle them).
        let succ_block_ids: HashSet<BlockId> = block
            .succs
            .iter()
            .map(|e| func.edges[e.index()].to)
            .collect();
        if !leaking_params
            .iter()
            .all(|p| is_only_used_in_blocks(func, *p, block.id, &succ_block_ids))
        {
            continue;
        }

        // Only thread the "break" direction: for branch_if_zero, thread the
        // edge carrying val=0 (which takes the "taken" path, skipping work).
        // This keeps the "continue" edge alive, preserving b13 and its params.
        for &pred_edge_id in &block.preds {
            let pred_edge = &func.edges[pred_edge_id.index()];
            if func.blocks[pred_edge.from.index()].dead {
                continue;
            }

            let cond_arg = pred_edge.args.iter().find(|a| a.target == cond);
            let Some(cond_arg) = cond_arg else {
                continue;
            };

            let Some(&val) = const_vals.get(&cond_arg.source) else {
                continue;
            };

            // Only thread the "break" direction: val=0 for branch_if_zero,
            // val=0 for branch_if. Skip the "continue" direction.
            let is_break = val == 0;
            if !is_break {
                continue;
            }

            let target_edge = if is_zero_test {
                taken_edge // val=0, branch_if_zero → taken
            } else {
                fallthrough_edge // val=0, branch_if → fallthrough
            };

            return Some(ThreadAction {
                pred_edge_id,
                block_id: block.id,
                target_edge,
                leaking_params,
            });
        }
    }

    None
}

/// Lift leaking params from a block into its successor edges' target blocks.
///
/// For each leaking param P of block B and each successor edge E → T:
/// 1. Create a new vreg V_new
/// 2. Add {target: V_new, source: P} to E and all other pred edges of T
/// 3. Add V_new as a param of T
/// 4. Replace uses of P in T's outgoing edge args with V_new (local only)
fn lift_leaking_params(
    func: &mut Function,
    block_id: BlockId,
    leaking_params: &[VReg],
    vreg_count: &mut u32,
    debug: bool,
) {
    let succ_edges: Vec<EdgeId> = func.blocks[block_id.index()].succs.clone();

    for &leaking_param in leaking_params {
        for &edge_id in &succ_edges {
            let target_block_id = func.edges[edge_id.index()].to;
            let new_vreg = VReg::new(*vreg_count);
            *vreg_count += 1;

            if debug {
                eprintln!(
                    "[control_thread]   lifting v{} -> v{} for b{} (via e{})",
                    leaking_param.index(),
                    new_vreg.index(),
                    target_block_id.index(),
                    edge_id.index(),
                );
            }

            // Add new param to target block.
            func.blocks[target_block_id.index()].params.push(new_vreg);

            // Add edge arg to this successor edge.
            func.edges[edge_id.index()].args.push(EdgeArg {
                target: new_vreg,
                source: leaking_param,
            });

            // Add edge arg to ALL OTHER predecessor edges of the target block.
            let other_preds: Vec<EdgeId> = func.blocks[target_block_id.index()]
                .preds
                .iter()
                .filter(|e| **e != edge_id)
                .copied()
                .collect();
            for &other_edge_id in &other_preds {
                func.edges[other_edge_id.index()].args.push(EdgeArg {
                    target: new_vreg,
                    source: leaking_param,
                });
            }

            // Replace leaking_param with new_vreg LOCALLY in the target block:
            // only in the target block's outgoing edge arg sources.
            // (Instruction-free blocks have no instructions to update.)
            let target_succs: Vec<EdgeId> = func.blocks[target_block_id.index()].succs.clone();
            for &out_edge_id in &target_succs {
                for arg in &mut func.edges[out_edge_id.index()].args {
                    if arg.source == leaking_param {
                        arg.source = new_vreg;
                    }
                }
            }
            // Replace in terminator if used as condition.
            let term_id = func.blocks[target_block_id.index()].term;
            let term = &mut func.terms[term_id.index()];
            match term {
                Terminator::BranchIf { cond, .. } | Terminator::BranchIfZero { cond, .. } => {
                    if *cond == leaking_param {
                        *cond = new_vreg;
                    }
                }
                Terminator::JumpTable { predicate, .. } => {
                    if *predicate == leaking_param {
                        *predicate = new_vreg;
                    }
                }
                _ => {}
            }
        }
    }
}

/// Retarget a predecessor edge to bypass a block, going directly to a successor
/// edge's target with composed edge args.
fn thread_edge(
    func: &mut Function,
    pred_edge_id: EdgeId,
    block_id: BlockId,
    target_edge: EdgeId,
) -> bool {
    let block = &func.blocks[block_id.index()];
    let block_params = block.params.clone();

    let succ_edge = &func.edges[target_edge.index()];
    let new_target = succ_edge.to;
    let succ_args = succ_edge.args.clone();

    let pred_args = func.edges[pred_edge_id.index()].args.clone();

    // Build param → source substitution from the predecessor's edge args.
    let subst: HashMap<VReg, VReg> = block_params
        .iter()
        .zip(pred_args.iter())
        .map(|(param, arg)| (*param, arg.source))
        .collect();

    // Compose successor edge args through substitution.
    // Non-param sources dominate the block and hence the predecessor — keep as-is.
    let mut composed_args = succ_args;
    for arg in &mut composed_args {
        if let Some(&replacement) = subst.get(&arg.source) {
            arg.source = replacement;
        }
        // Non-param sources are left unchanged — they dominate this block
        // and therefore also dominate the predecessor.
    }

    // Retarget the predecessor edge.
    func.edges[pred_edge_id.index()].to = new_target;
    func.edges[pred_edge_id.index()].args = composed_args;

    // Update pred lists.
    func.blocks[block_id.index()]
        .preds
        .retain(|e| *e != pred_edge_id);
    func.blocks[new_target.index()].preds.push(pred_edge_id);

    true
}

/// Check if a vreg's external uses are ONLY in specific allowed blocks
/// (the source block itself and its direct successor blocks). Uses in
/// other blocks would prevent safe lifting.
fn is_only_used_in_blocks(
    func: &Function,
    vreg: VReg,
    block_id: BlockId,
    allowed_blocks: &HashSet<BlockId>,
) -> bool {
    let block_succ_edges: HashSet<EdgeId> = func.blocks[block_id.index()]
        .succs
        .iter()
        .copied()
        .collect();

    for block in &func.blocks {
        if block.dead || block.id == block_id || allowed_blocks.contains(&block.id) {
            continue;
        }
        // Check instructions
        for &inst_id in &block.insts {
            let inst = &func.insts[inst_id.index()];
            let mut found = false;
            inst.op.for_each_use(|v| {
                if *v == vreg {
                    found = true;
                }
            });
            if found {
                return false;
            }
        }
        // Check terminators
        let term = &func.terms[block.term.index()];
        match term {
            Terminator::BranchIf { cond, .. } | Terminator::BranchIfZero { cond, .. } => {
                if *cond == vreg {
                    return false;
                }
            }
            Terminator::JumpTable { predicate, .. } => {
                if *predicate == vreg {
                    return false;
                }
            }
            _ => {}
        }
        // Check edge arg sources from non-allowed, non-successor edges
        for &edge_id in &block.succs {
            if block_succ_edges.contains(&edge_id) {
                continue;
            }
            for arg in &func.edges[edge_id.index()].args {
                if arg.source == vreg {
                    return false;
                }
            }
        }
    }
    !func.data_results.contains(&vreg)
}

/// Check if a vreg is used outside a specific block (in instructions,
/// terminators, or edge args of other blocks).
fn is_vreg_used_outside_block(func: &Function, vreg: VReg, block_id: BlockId) -> bool {
    let block_succ_edges: HashSet<EdgeId> = func.blocks[block_id.index()]
        .succs
        .iter()
        .copied()
        .collect();

    for block in &func.blocks {
        if block.dead || block.id == block_id {
            continue;
        }
        for &inst_id in &block.insts {
            let inst = &func.insts[inst_id.index()];
            let mut found = false;
            inst.op.for_each_use(|v| {
                if *v == vreg {
                    found = true;
                }
            });
            if found {
                return true;
            }
        }
        let term = &func.terms[block.term.index()];
        match term {
            Terminator::BranchIf { cond, .. } | Terminator::BranchIfZero { cond, .. } => {
                if *cond == vreg {
                    return true;
                }
            }
            Terminator::JumpTable { predicate, .. } => {
                if *predicate == vreg {
                    return true;
                }
            }
            _ => {}
        }
        // Check edge args from other blocks (not our successor edges).
        for &edge_id in &block.succs {
            if block_succ_edges.contains(&edge_id) {
                continue;
            }
            let edge = &func.edges[edge_id.index()];
            for arg in &edge.args {
                if arg.source == vreg {
                    return true;
                }
            }
        }
    }
    func.data_results.contains(&vreg)
}

/// Transitively mark unreachable blocks as dead.
fn kill_unreachable(func: &mut Function) {
    let mut changed = true;
    while changed {
        changed = false;
        for bi in 0..func.blocks.len() {
            let block = &func.blocks[bi];
            if block.dead || block.id == func.entry {
                continue;
            }
            let all_preds_dead = block
                .preds
                .iter()
                .all(|e| func.blocks[func.edges[e.index()].from.index()].dead);
            if block.preds.is_empty() || all_preds_dead {
                let succs: Vec<_> = func.blocks[bi].succs.clone();
                let dead_block_id = func.blocks[bi].id;
                func.blocks[bi].dead = true;
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
}
