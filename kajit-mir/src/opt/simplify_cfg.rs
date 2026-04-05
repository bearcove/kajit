//! CFG simplification: trampoline forwarding and jump-threading through phi-of-constants.
//!
//! Two transformations:
//! 1. **Trampoline forwarding**: empty blocks with an unconditional branch are forwarded
//!    through — all predecessors are redirected to the final target. When this would break
//!    SSA dominance (non-param vregs losing their dominator path), those vregs are threaded
//!    through the redirected edges as new block params.
//! 2. **Jump-threading through phi-of-constants**: empty blocks with a conditional branch
//!    on a block param, where all predecessors supply a known constant for that param,
//!    are threaded — each predecessor is redirected to the correct successor.

use std::collections::{HashMap, HashSet};

use crate::analysis::dominance::DominanceInfo;
use crate::cfg_mir::{BlockId, EdgeArg, EdgeId, Function, OperandKind, Terminator};
use kajit_ir::VReg;
use kajit_lir::LinearOp;

/// Run CFG simplification: trampoline forwarding + jump-threading.
/// Returns true if any changes were made.
pub fn simplify_cfg(func: &mut Function, vreg_count: &mut u32) -> bool {
    let mut changed = false;
    while forward_trampolines(func, vreg_count) {
        func.gc_edges();
        #[cfg(debug_assertions)]
        verify_edge_arg_defs(func, "after forward_trampolines + gc_edges");
        changed = true;
    }
    #[cfg(debug_assertions)]
    verify_edge_arg_defs(func, "after all forward_trampolines done");
    while thread_phi_branches(func, vreg_count) {
        func.gc_edges();
        changed = true;
    }
    changed
}

#[cfg(debug_assertions)]
fn verify_edge_arg_defs(func: &Function, context: &str) {
    let def_map = build_def_map(func);
    for block in func.live_blocks() {
        for &eid in &block.succs {
            let edge = &func.edges[eid.index()];
            for arg in &edge.args {
                assert!(
                    def_map.contains_key(&arg.source),
                    "simplify_cfg [{}]: edge e{} (b{} -> b{}) has source vreg v{} which is not defined anywhere",
                    context, eid.index(), edge.from.index(), edge.to.index(), arg.source.index()
                );
            }
        }
    }
}

/// Collect all vregs used in a block (instructions, terminator, and outgoing edge args).
fn collect_block_uses(func: &Function, block_id: BlockId) -> HashSet<VReg> {
    let block = &func.blocks[block_id.index()];
    let mut uses = HashSet::new();

    for &inst_id in &block.insts {
        let inst = &func.insts[inst_id.index()];
        inst.op.for_each_use(|v| {
            uses.insert(*v);
        });
    }

    let term = &func.terms[block.term.index()];
    match term {
        Terminator::BranchIf { cond, .. } | Terminator::BranchIfZero { cond, .. } => {
            uses.insert(*cond);
        }
        Terminator::JumpTable { predicate, .. } => {
            uses.insert(*predicate);
        }
        _ => {}
    }

    for &edge_id in &block.succs {
        for arg in &func.edges[edge_id.index()].args {
            uses.insert(arg.source);
        }
    }

    uses
}

/// Collect all vregs defined in a block (params + instruction defs).
fn collect_block_defs(func: &Function, block_id: BlockId) -> HashSet<VReg> {
    let block = &func.blocks[block_id.index()];
    let mut defs: HashSet<VReg> = block.params.iter().copied().collect();
    for &inst_id in &block.insts {
        let inst = &func.insts[inst_id.index()];
        for op in &inst.operands {
            if op.kind.is_def() {
                defs.insert(op.vreg);
            }
        }
    }
    defs
}

/// Build a map from vreg to defining block for the entire function.
fn build_def_map(func: &Function) -> HashMap<VReg, BlockId> {
    let mut def_map = HashMap::new();
    for &arg in &func.data_args {
        def_map.insert(arg, func.entry);
    }
    for block in func.live_blocks() {
        for &param in &block.params {
            def_map.insert(param, block.id);
        }
        for &inst_id in &block.insts {
            let inst = &func.insts[inst_id.index()];
            for op in &inst.operands {
                if op.kind.is_def() {
                    def_map.insert(op.vreg, block.id);
                }
            }
        }
    }
    def_map
}

/// Collect all blocks dominated by `root` in the dominance tree (inclusive).
fn dominated_subtree(dom: &DominanceInfo, root: BlockId, all_blocks: &[BlockId]) -> Vec<BlockId> {
    let mut result = Vec::new();
    let mut stack = vec![root];
    while let Some(block) = stack.pop() {
        result.push(block);
        for &child in dom.dominator_tree_children(block) {
            if all_blocks.contains(&child) {
                stack.push(child);
            }
        }
    }
    result
}

/// Determine which vregs need to be threaded through as new block params when
/// a trampoline is removed.
///
/// Returns `None` if the transformation is unsafe (a vreg needs threading but
/// can't be provided by all edges into the target).
///
/// A vreg needs threading if:
/// 1. It's used in the target block (or blocks dominated by target)
/// 2. It's NOT defined in the target's dominance subtree
/// 3. Its def block won't dominate target after the trampoline is removed
fn find_vregs_needing_threading(
    func: &Function,
    dom: &DominanceInfo,
    def_map: &HashMap<VReg, BlockId>,
    trampoline_id: BlockId,
    target_id: BlockId,
    new_pred_blocks: &[BlockId],
) -> Option<Vec<VReg>> {
    let all_live: Vec<BlockId> = func.live_blocks().map(|b| b.id).collect();
    let target_subtree = dominated_subtree(dom, target_id, &all_live);

    // Collect all vregs used in target's dominance subtree that aren't locally defined
    let mut external_uses: HashSet<VReg> = HashSet::new();
    for &bid in &target_subtree {
        let uses = collect_block_uses(func, bid);
        let defs = collect_block_defs(func, bid);
        for v in uses {
            if !defs.contains(&v) {
                external_uses.insert(v);
            }
        }
    }

    // Check if target has existing predecessors (not through the trampoline)
    let trampoline_edge_ids: HashSet<EdgeId> = func.blocks[trampoline_id.index()]
        .succs
        .iter()
        .copied()
        .collect();
    let existing_pred_blocks: Vec<BlockId> = func.blocks[target_id.index()]
        .preds
        .iter()
        .filter(|&&eid| !trampoline_edge_ids.contains(&eid))
        .map(|&eid| func.edges[eid.index()].from)
        .collect();

    let mut needs_threading = Vec::new();
    for vreg in external_uses {
        let Some(&def_block) = def_map.get(&vreg) else {
            continue;
        };

        // If defined in the target subtree, it's not external
        if target_subtree.contains(&def_block) {
            continue;
        }

        // Check: does the def block currently dominate target?
        if !dom.dominates(def_block, target_id) {
            continue;
        }

        // Check: would the def block still dominate target after the trampoline is gone?
        let dominates_new_preds = new_pred_blocks
            .iter()
            .all(|&pred| dom.dominates(def_block, pred));
        let dominates_existing_preds = existing_pred_blocks
            .iter()
            .all(|&pred| dom.dominates(def_block, pred));

        if dominates_new_preds && dominates_existing_preds {
            // Still dominates — no threading needed
            continue;
        }

        // Needs threading. Check if we can actually provide it on all edges.
        // For redirected edges: we can compose through the trampoline's params.
        // For existing edges: the def must dominate the existing pred blocks,
        // AND the def must be in a "stable" block that won't be killed by future
        // forwarding/threading passes.
        if !dominates_existing_preds {
            // Can't provide this vreg on existing edges — transformation is unsafe
            return None;
        }

        // Check if the def block is stable (has instructions or is entry).
        // An empty block might be forwarded/threaded later, killing the definition.
        if !existing_pred_blocks.is_empty() {
            let def_block_data = &func.blocks[def_block.index()];
            let is_stable = def_block == func.entry
                || !def_block_data.insts.is_empty()
                || def_block_data.dead;
            if !is_stable {
                // Def is in an empty block that might be killed — unsafe
                return None;
            }
        }

        needs_threading.push(vreg);
    }

    needs_threading.sort_by_key(|v| v.index());
    Some(needs_threading)
}

/// Rewrite uses of `old_vreg` to `new_vreg` in a block's instructions, terminator, and
/// outgoing edge args.
fn rewrite_vreg_in_block(func: &mut Function, block_id: BlockId, old_vreg: VReg, new_vreg: VReg) {
    let block = &func.blocks[block_id.index()];
    let inst_ids: Vec<_> = block.insts.clone();
    let term_id = block.term;
    let succ_edges: Vec<_> = block.succs.clone();

    for inst_id in inst_ids {
        let inst = &mut func.insts[inst_id.index()];
        inst.op.for_each_use_mut(|v| {
            if *v == old_vreg {
                *v = new_vreg;
            }
        });
        for operand in &mut inst.operands {
            if operand.vreg == old_vreg {
                operand.vreg = new_vreg;
            }
        }
    }

    let term = &mut func.terms[term_id.index()];
    match term {
        Terminator::BranchIf { cond, .. } | Terminator::BranchIfZero { cond, .. } => {
            if *cond == old_vreg {
                *cond = new_vreg;
            }
        }
        Terminator::JumpTable { predicate, .. } => {
            if *predicate == old_vreg {
                *predicate = new_vreg;
            }
        }
        _ => {}
    }

    for edge_id in succ_edges {
        for arg in &mut func.edges[edge_id.index()].args {
            if arg.source == old_vreg {
                arg.source = new_vreg;
            }
        }
    }
}

/// Forward through empty trampoline blocks (unconditional Branch, no instructions).
/// Processes one block per call; caller iterates to fixpoint.
///
/// When forwarding would break SSA dominance, threads the affected vregs through
/// as new block params on the target.
fn forward_trampolines(func: &mut Function, vreg_count: &mut u32) -> bool {
    let dom = DominanceInfo::compute(func);
    let def_map = build_def_map(func);

    // Collect all trampoline candidates
    let candidates: Vec<(BlockId, EdgeId)> = func
        .live_blocks()
        .filter(|b| b.insts.is_empty() && b.id != func.entry && !b.preds.is_empty())
        .filter_map(|b| {
            let term = &func.terms[b.term.index()];
            if let Terminator::Branch { edge } = term {
                Some((b.id, *edge))
            } else {
                None
            }
        })
        .collect();

    // Try each candidate, skip unsafe ones
    for (trampoline_id, out_edge_id) in candidates {
        let trampoline_block = &func.blocks[trampoline_id.index()];
        let trampoline_params = trampoline_block.params.clone();
        let pred_edges: Vec<EdgeId> = trampoline_block.preds.clone();
        let final_target = func.edges[out_edge_id.index()].to;
        let out_args = func.edges[out_edge_id.index()].args.clone();

        let new_pred_blocks: Vec<BlockId> = pred_edges
            .iter()
            .map(|&eid| func.edges[eid.index()].from)
            .collect();

        // Find vregs that need threading to preserve SSA dominance
        let Some(vregs_to_thread) = find_vregs_needing_threading(
            func,
            &dom,
            &def_map,
            trampoline_id,
            final_target,
            &new_pred_blocks,
        ) else {
            // Transformation is unsafe for this trampoline — skip to next
            continue;
        };

        // Allocate fresh vregs and set up the threading
        let mut vreg_remap: HashMap<VReg, VReg> = HashMap::new();
        for &old_vreg in &vregs_to_thread {
            let new_vreg = VReg::new(*vreg_count);
            *vreg_count += 1;
            vreg_remap.insert(old_vreg, new_vreg);
        }

        // Add new params to the target block
        for &old_vreg in &vregs_to_thread {
            let new_vreg = vreg_remap[&old_vreg];
            func.blocks[final_target.index()].params.push(new_vreg);
        }

        // Add edge args to ALL existing edges into the target (not from the trampoline)
        let target_pred_edges: Vec<EdgeId> = func.blocks[final_target.index()].preds.clone();
        let trampoline_succ_edge_set: HashSet<EdgeId> = func.blocks[trampoline_id.index()]
            .succs
            .iter()
            .copied()
            .collect();
        for &eid in &target_pred_edges {
            if trampoline_succ_edge_set.contains(&eid) {
                continue;
            }
            for &old_vreg in &vregs_to_thread {
                let new_vreg = vreg_remap[&old_vreg];
                func.edges[eid.index()].args.push(EdgeArg {
                    target: new_vreg,
                    source: old_vreg,
                });
            }
        }

        // Redirect every predecessor edge to the final target, composing
        // edge args through the trampoline's params + adding threaded vregs.
        for pred_edge_id in &pred_edges {
            let pred_incoming_args = func.edges[pred_edge_id.index()].args.clone();
            let mut subst: HashMap<VReg, VReg> = HashMap::new();
            for (param, arg) in trampoline_params.iter().zip(pred_incoming_args.iter()) {
                subst.insert(*param, arg.source);
            }
            let mut composed_args = out_args.clone();
            for arg in &mut composed_args {
                if let Some(&replacement) = subst.get(&arg.source) {
                    arg.source = replacement;
                }
            }

            // Add the threaded vregs. If the vreg is a trampoline param, compose
            // through to the predecessor's source for that param.
            for &old_vreg in &vregs_to_thread {
                let new_vreg = vreg_remap[&old_vreg];
                let source = subst.get(&old_vreg).copied().unwrap_or(old_vreg);
                composed_args.push(EdgeArg {
                    target: new_vreg,
                    source,
                });
            }

            func.edges[pred_edge_id.index()].to = final_target;
            func.edges[pred_edge_id.index()].args = composed_args;
        }

        // Rewrite uses in target and all blocks dominated by target
        if !vreg_remap.is_empty() {
            let all_live: Vec<BlockId> = func.live_blocks().map(|b| b.id).collect();
            let subtree = dominated_subtree(&dom, final_target, &all_live);
            for &bid in &subtree {
                for (&old_vreg, &new_vreg) in &vreg_remap {
                    rewrite_vreg_in_block(func, bid, old_vreg, new_vreg);
                }
            }
        }

        // Mark trampoline dead. gc_edges() will rebuild preds/succs.
        func.blocks[trampoline_id.index()].dead = true;

        // Verify: no edge args reference undefined vregs
        #[cfg(debug_assertions)]
        {
            let live_def_map = build_def_map(func);
            for block in func.live_blocks() {
                for &eid in &block.succs {
                    let edge = &func.edges[eid.index()];
                    for arg in &edge.args {
                        assert!(
                            live_def_map.contains_key(&arg.source),
                            "simplify_cfg: edge e{} (b{} -> b{}) has source vreg v{} which is not defined anywhere (after forwarding trampoline b{})",
                            eid.index(), edge.from.index(), edge.to.index(), arg.source.index(), trampoline_id.index()
                        );
                    }
                }
            }
        }

        // Process one trampoline per call, then return for gc_edges + re-iteration
        return true;
    }

    false
}

/// Thread conditional branches through block params when all predecessors
/// supply known zero/nonzero values for the branch condition.
fn thread_phi_branches(func: &mut Function, vreg_count: &mut u32) -> bool {
    let dom = DominanceInfo::compute(func);
    let def_map = build_def_map(func);

    let mut const_vals: HashMap<VReg, u64> = HashMap::new();
    for inst in &func.insts {
        if let LinearOp::Const { dst, value } = &inst.op {
            const_vals.insert(*dst, *value);
        }
    }

    let candidate = func
        .live_blocks()
        .filter(|b| b.insts.is_empty() && !b.params.is_empty())
        .find(|b| {
            let term = &func.terms[b.term.index()];
            matches!(
                term,
                Terminator::BranchIf { .. } | Terminator::BranchIfZero { .. }
            )
        })
        .map(|b| b.id);

    let Some(block_id) = candidate else {
        return false;
    };

    let block = &func.blocks[block_id.index()];
    let term = func.terms[block.term.index()].clone();

    let (cond, taken_edge, fallthrough_edge, is_zero_test) = match &term {
        Terminator::BranchIf {
            cond,
            taken,
            fallthrough,
        } => (*cond, *taken, *fallthrough, false),
        Terminator::BranchIfZero {
            cond,
            taken,
            fallthrough,
        } => (*cond, *taken, *fallthrough, true),
        _ => return false,
    };

    let param_idx = match block.params.iter().position(|p| *p == cond) {
        Some(idx) => idx,
        None => return false,
    };

    let pred_edges: Vec<EdgeId> = block.preds.clone();
    let mut pred_targets: Vec<(EdgeId, EdgeId)> = Vec::new();

    for &pred_edge_id in &pred_edges {
        let pred_edge = &func.edges[pred_edge_id.index()];
        if pred_edge.args.len() <= param_idx {
            return false;
        }
        let source_vreg = pred_edge.args[param_idx].source;
        let known_nonzero = const_vals.get(&source_vreg).map(|&val| val != 0);

        let Some(is_nonzero) = known_nonzero else {
            return false;
        };

        let chosen_edge = if is_zero_test {
            if !is_nonzero {
                taken_edge
            } else {
                fallthrough_edge
            }
        } else if is_nonzero {
            taken_edge
        } else {
            fallthrough_edge
        };
        pred_targets.push((pred_edge_id, chosen_edge));
    }

    if pred_targets.is_empty() {
        return false;
    }

    let block = &func.blocks[block_id.index()];
    let block_params = block.params.clone();

    let taken_target = func.edges[taken_edge.index()].to;
    let taken_args = func.edges[taken_edge.index()].args.clone();
    let fallthrough_target = func.edges[fallthrough_edge.index()].to;
    let fallthrough_args = func.edges[fallthrough_edge.index()].args.clone();

    // For each target, find vregs needing threading
    let targets: HashSet<BlockId> = [taken_target, fallthrough_target].into_iter().collect();
    let pred_blocks: Vec<BlockId> = pred_edges
        .iter()
        .map(|&eid| func.edges[eid.index()].from)
        .collect();

    let mut target_threading: HashMap<BlockId, (Vec<VReg>, HashMap<VReg, VReg>)> = HashMap::new();
    for &target_id in &targets {
        let Some(vregs_to_thread) = find_vregs_needing_threading(
            func,
            &dom,
            &def_map,
            block_id,
            target_id,
            &pred_blocks,
        ) else {
            // Unsafe — can't thread this target
            return false;
        };
        let mut remap = HashMap::new();
        for &old_vreg in &vregs_to_thread {
            let new_vreg = VReg::new(*vreg_count);
            *vreg_count += 1;
            remap.insert(old_vreg, new_vreg);
        }
        // Add params to target
        for &old_vreg in &vregs_to_thread {
            let new_vreg = remap[&old_vreg];
            func.blocks[target_id.index()].params.push(new_vreg);
        }
        // Add edge args to existing edges into target (not from the threading block)
        let target_preds: Vec<EdgeId> = func.blocks[target_id.index()].preds.clone();
        let block_succs: HashSet<EdgeId> = func.blocks[block_id.index()]
            .succs
            .iter()
            .copied()
            .collect();
        for &eid in &target_preds {
            if block_succs.contains(&eid) {
                continue;
            }
            for &old_vreg in &vregs_to_thread {
                let new_vreg = remap[&old_vreg];
                func.edges[eid.index()].args.push(EdgeArg {
                    target: new_vreg,
                    source: old_vreg,
                });
            }
        }
        // Rewrite uses in target subtree
        if !remap.is_empty() {
            let all_live: Vec<BlockId> = func.live_blocks().map(|b| b.id).collect();
            let subtree = dominated_subtree(&dom, target_id, &all_live);
            for &bid in &subtree {
                for (&old_vreg, &new_vreg) in &remap {
                    rewrite_vreg_in_block(func, bid, old_vreg, new_vreg);
                }
            }
        }
        target_threading.insert(target_id, (vregs_to_thread, remap));
    }

    // Redirect predecessor edges
    for (pred_edge_id, chosen_edge) in &pred_targets {
        let pred_incoming = func.edges[pred_edge_id.index()].args.clone();
        let mut subst: HashMap<VReg, VReg> = HashMap::new();
        for (param, arg) in block_params.iter().zip(pred_incoming.iter()) {
            subst.insert(*param, arg.source);
        }

        let (out_args, final_target) = if *chosen_edge == taken_edge {
            (&taken_args, taken_target)
        } else {
            (&fallthrough_args, fallthrough_target)
        };

        let mut composed_args = out_args.clone();
        for arg in &mut composed_args {
            if let Some(&replacement) = subst.get(&arg.source) {
                arg.source = replacement;
            }
        }

        // Add threaded vregs, composing through the block's param mapping
        if let Some((vregs_to_thread, remap)) = target_threading.get(&final_target) {
            for &old_vreg in vregs_to_thread {
                let new_vreg = remap[&old_vreg];
                let source = subst.get(&old_vreg).copied().unwrap_or(old_vreg);
                composed_args.push(EdgeArg {
                    target: new_vreg,
                    source,
                });
            }
        }

        func.edges[pred_edge_id.index()].to = final_target;
        func.edges[pred_edge_id.index()].args = composed_args;
    }

    // Mark block dead
    func.blocks[block_id.index()].dead = true;

    true
}
