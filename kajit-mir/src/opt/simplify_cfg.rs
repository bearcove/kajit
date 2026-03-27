//! CFG simplification: trampoline forwarding and jump-threading through phi-of-constants.
//!
//! Two transformations:
//! 1. **Trampoline forwarding**: empty blocks with an unconditional branch are forwarded
//!    through — all predecessors are redirected to the final target. Handles multi-pred
//!    blocks (unlike merge_empty_blocks which requires single pred + single succ).
//! 2. **Jump-threading through phi-of-constants**: empty blocks with a conditional branch
//!    on a block param, where all predecessors supply a known constant (or provably
//!    zero/nonzero value) for that param, are threaded — each predecessor is redirected
//!    to the correct successor.

use std::collections::HashMap;

use crate::cfg_mir::{BlockId, EdgeId, Function, Terminator};
use kajit_ir::VReg;
use kajit_lir::LinearOp;

/// Run CFG simplification: trampoline forwarding + jump-threading.
/// Returns true if any changes were made.
pub fn simplify_cfg(func: &mut Function, _vreg_count: &mut u32) -> bool {
    let mut changed = false;
    loop {
        let a = forward_trampolines(func);
        let b = thread_phi_branches(func);
        if !a && !b {
            break;
        }
        changed = true;
    }
    if changed {
        crate::opt::block_merge::remove_unreachable_blocks(func);
    }
    changed
}

/// Forward through empty trampoline blocks (unconditional Branch, no instructions).
/// Processes one block per call; caller iterates to fixpoint.
fn forward_trampolines(func: &mut Function) -> bool {
    let trampoline = func
        .blocks
        .iter()
        .filter(|b| !b.dead && b.insts.is_empty() && b.id != func.entry && !b.preds.is_empty())
        .find_map(|b| {
            let term = &func.terms[b.term.index()];
            if let Terminator::Branch { edge } = term {
                if b.preds.len() > 1 && !b.params.is_empty() {
                    let out_args = &func.edges[edge.index()].args;
                    let out_arg_sources: std::collections::HashSet<VReg> =
                        out_args.iter().map(|a| a.source).collect();
                    let safe = b.params.iter().all(|p| {
                        out_arg_sources.contains(p) || !is_vreg_used_outside_block(func, *p, b.id)
                    });
                    if !safe {
                        return None;
                    }
                }
                Some((b.id, *edge))
            } else {
                None
            }
        });

    let Some((trampoline_id, out_edge_id)) = trampoline else {
        return false;
    };

    let trampoline = &func.blocks[trampoline_id.index()];
    let trampoline_params = trampoline.params.clone();
    let pred_edges: Vec<EdgeId> = trampoline.preds.clone();
    let final_target = func.edges[out_edge_id.index()].to;
    let out_args = func.edges[out_edge_id.index()].args.clone();

    if pred_edges.len() == 1 && !trampoline_params.is_empty() {
        let pred_incoming_args = func.edges[pred_edges[0].index()].args.clone();
        let mut global_subst: HashMap<VReg, VReg> = HashMap::new();
        for (param, arg) in trampoline_params.iter().zip(pred_incoming_args.iter()) {
            global_subst.insert(*param, arg.source);
        }
        crate::opt::constant_phi_elim::replace_vregs_in_function(func, &global_subst);
    }

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

        func.edges[pred_edge_id.index()].to = final_target;
        func.edges[pred_edge_id.index()].args = composed_args;
        func.blocks[final_target.index()].preds.push(*pred_edge_id);
    }

    func.blocks[trampoline_id.index()].preds.clear();
    func.blocks[final_target.index()]
        .preds
        .retain(|e| *e != out_edge_id);
    func.blocks[trampoline_id.index()].dead = true;
    func.blocks[trampoline_id.index()].succs.clear();

    true
}

/// Check if a vreg is used outside a specific block.
fn is_vreg_used_outside_block(func: &Function, vreg: VReg, block_id: BlockId) -> bool {
    let block_succ_edges: std::collections::HashSet<EdgeId> = func.blocks[block_id.index()]
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

/// Thread conditional branches through block params when all predecessors
/// supply known zero/nonzero values for the branch condition.
fn thread_phi_branches(func: &mut Function) -> bool {
    let mut const_vals: HashMap<VReg, u64> = HashMap::new();
    for inst in &func.insts {
        if let LinearOp::Const { dst, value } = &inst.op {
            const_vals.insert(*dst, *value);
        }
    }

    let candidate = func
        .blocks
        .iter()
        .filter(|b| !b.dead && b.insts.is_empty() && !b.params.is_empty())
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

        let known_nonzero = if let Some(&val) = const_vals.get(&source_vreg) {
            Some(val != 0)
        } else {
            infer_nonzero_from_predecessor(func, pred_edge_id, source_vreg)
        };

        let Some(is_nonzero) = known_nonzero else {
            return false;
        };

        let chosen_edge = if is_zero_test {
            if !is_nonzero {
                taken_edge
            } else {
                fallthrough_edge
            }
        } else {
            if is_nonzero {
                taken_edge
            } else {
                fallthrough_edge
            }
        };
        pred_targets.push((pred_edge_id, chosen_edge));
    }

    if pred_targets.is_empty() {
        return false;
    }

    // Pre-processing: for non-condition params where ALL predecessors supply the
    // same source vreg, globally replace the param with that source.
    let block = &func.blocks[block_id.index()];
    let block_params = block.params.clone();
    let pred_edges: Vec<EdgeId> = block.preds.clone();
    {
        let mut global_subst: HashMap<VReg, VReg> = HashMap::new();
        for (pi, param) in block_params.iter().enumerate() {
            if *param == cond {
                continue;
            }
            let mut common_source: Option<VReg> = None;
            let mut all_same = true;
            for &pred_edge_id in &pred_edges {
                let pred_edge = &func.edges[pred_edge_id.index()];
                if pred_edge.args.len() <= pi {
                    all_same = false;
                    break;
                }
                let src = pred_edge.args[pi].source;
                match common_source {
                    None => common_source = Some(src),
                    Some(prev) if prev == src => {}
                    _ => {
                        all_same = false;
                        break;
                    }
                }
            }
            if all_same {
                if let Some(src) = common_source {
                    if src != *param {
                        global_subst.insert(*param, src);
                    }
                }
            }
        }
        if !global_subst.is_empty() {
            crate::opt::constant_phi_elim::replace_vregs_in_function(func, &global_subst);
        }
    }

    let block = &func.blocks[block_id.index()];
    let block_params = block.params.clone();
    let succ_edges: Vec<EdgeId> = block.succs.clone();
    let mut out_arg_sources: std::collections::HashSet<VReg> = std::collections::HashSet::new();
    for &edge_id in &succ_edges {
        for arg in &func.edges[edge_id.index()].args {
            out_arg_sources.insert(arg.source);
        }
    }
    for param in &block_params {
        if *param == cond || out_arg_sources.contains(param) {
            continue;
        }
        if is_vreg_used_outside_block(func, *param, block_id) {
            return false;
        }
    }

    let taken_target = func.edges[taken_edge.index()].to;
    let taken_args = func.edges[taken_edge.index()].args.clone();
    let fallthrough_target = func.edges[fallthrough_edge.index()].to;
    let fallthrough_args = func.edges[fallthrough_edge.index()].args.clone();

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

        func.edges[pred_edge_id.index()].to = final_target;
        func.edges[pred_edge_id.index()].args = composed_args;
        func.blocks[final_target.index()].preds.push(*pred_edge_id);
    }

    func.blocks[block_id.index()].preds.clear();
    func.blocks[taken_target.index()]
        .preds
        .retain(|e| *e != taken_edge);
    func.blocks[fallthrough_target.index()]
        .preds
        .retain(|e| *e != fallthrough_edge);
    func.blocks[block_id.index()].dead = true;
    func.blocks[block_id.index()].succs.clear();

    true
}

/// Infer whether `source_vreg` is provably nonzero or zero based on the
/// predecessor block's terminator.
fn infer_nonzero_from_predecessor(
    func: &Function,
    pred_edge_id: EdgeId,
    source_vreg: VReg,
) -> Option<bool> {
    let pred_block_id = func.edges[pred_edge_id.index()].from;
    let pred_block = &func.blocks[pred_block_id.index()];
    let pred_term = &func.terms[pred_block.term.index()];

    match pred_term {
        Terminator::BranchIf {
            cond,
            taken,
            fallthrough,
        } => {
            if *cond == source_vreg {
                if pred_edge_id == *taken {
                    Some(true)
                } else if pred_edge_id == *fallthrough {
                    Some(false)
                } else {
                    None
                }
            } else {
                None
            }
        }
        Terminator::BranchIfZero {
            cond,
            taken,
            fallthrough,
        } => {
            if *cond == source_vreg {
                if pred_edge_id == *taken {
                    Some(false)
                } else if pred_edge_id == *fallthrough {
                    Some(true)
                } else {
                    None
                }
            } else {
                None
            }
        }
        _ => None,
    }
}
