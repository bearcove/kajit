//! RVSDG constant folding pass.
//!
//! After bounded-theta unrolling, each iteration body contains arithmetic on
//! the iteration index — but the index is a known constant for each copy.
//! This pass evaluates pure operations with all-constant inputs and replaces
//! them with `Const` nodes.
//!
//! Combined with gamma simplification (which folds constant-predicate gammas),
//! this propagates iteration constants through the unrolled gamma cascade.
//!
//! The walk is **outside-in** (parent regions before child regions) so that
//! constants at the gamma-input level are visible when processing inner
//! branches via `resolve_to_constant`.

use kajit_reprs::ir::{
    DebugScopeId, IrFunc, IrOp, Node, NodeId, NodeKind, OutputPort, OutputRef, PortKind,
    PortSource, RegionArgRef, RegionId,
};

/// Create a Const node in the given region and return its output as a PortSource.
pub fn create_const_in_region(
    func: &mut IrFunc,
    region: RegionId,
    debug_scope: DebugScopeId,
    value: u64,
) -> PortSource {
    let node_id = func.nodes.push(Node {
        region,
        debug_scope,
        debug_value: None,
        inputs: Vec::new(),
        outputs: vec![OutputPort {
            kind: PortKind::Data,
            vreg: None,
            debug_scope,
        }],
        kind: NodeKind::Simple(IrOp::Const { value }),
    });
    func.regions[region].nodes.push(node_id);
    PortSource::Node(OutputRef {
        node: node_id,
        index: 0,
    })
}

/// Run constant folding to fixpoint. Returns true if any changes were made.
pub fn const_fold(func: &mut IrFunc) -> bool {
    let debug = std::env::var("KAJIT_DEBUG_CONSTFOLD").is_ok();
    let mut any_changed = false;
    let mut total_folded = 0u32;
    loop {
        let (changed, folded) = const_fold_round(func, debug);
        total_folded += folded;
        if !changed {
            break;
        }
        any_changed = true;
    }
    if debug && total_folded > 0 {
        eprintln!("[const_fold] folded {} nodes total", total_folded);
    }
    any_changed
}

/// One round of constant folding across all regions (outside-in).
/// Returns (changed, fold_count).
fn const_fold_round(func: &mut IrFunc, debug: bool) -> (bool, u32) {
    let mut changed = false;
    let mut folded = 0u32;

    // Collect all regions in pre-order (parent before children).
    let regions_preorder = collect_regions_preorder(func);

    for rid in regions_preorder {
        let nodes = func.regions[rid].nodes.clone();
        for &nid in &nodes {
            if try_fold_node(func, nid, debug) || try_simplify_node(func, nid) {
                changed = true;
                folded += 1;
            }
        }
    }

    (changed, folded)
}

/// Collect all regions reachable from the root lambda, in pre-order
/// (parent regions before their children).
fn collect_regions_preorder(func: &IrFunc) -> Vec<RegionId> {
    let root_body = func.root_body();
    let mut result = Vec::new();
    let mut stack = vec![root_body];

    while let Some(rid) = stack.pop() {
        result.push(rid);
        // Push children in reverse order so first child is visited first.
        let nodes = &func.regions[rid].nodes;
        for &nid in nodes.iter().rev() {
            match &func.nodes[nid].kind {
                NodeKind::Gamma { regions } => {
                    for &sub in regions.iter().rev() {
                        stack.push(sub);
                    }
                }
                NodeKind::Theta { body, .. } | NodeKind::Lambda { body, .. } => {
                    stack.push(*body);
                }
                _ => {}
            }
        }
    }

    result
}

/// Try to evaluate a single node and replace it with a Const.
fn try_fold_node(func: &mut IrFunc, node_id: NodeId, debug: bool) -> bool {
    let NodeKind::Simple(op) = &func.nodes[node_id].kind else {
        return false;
    };
    // Already a constant — nothing to fold.
    if matches!(op, IrOp::Const { .. }) {
        return false;
    }
    let op = op.clone();

    // Resolve all inputs to constant values.
    let inputs = func.nodes[node_id].inputs.clone();
    let mut const_values: Vec<u64> = Vec::with_capacity(inputs.len());
    for (i, input) in inputs.iter().enumerate() {
        match resolve_to_constant(func, &input.source) {
            Some(v) => const_values.push(v),
            None => {
                if debug {
                    let trace = crate::provenance::trace_provenance(func, &input.source);
                    eprintln!(
                        "[const_fold] n{}: {:?} input[{}] not constant:\n{}",
                        node_id.index(),
                        op,
                        i,
                        trace,
                    );
                }
                return false;
            }
        }
    }

    // Evaluate the operation.
    let Some(result) = evaluate_op(&op, &const_values) else {
        return false;
    };

    if debug {
        eprintln!(
            "[const_fold] n{}: {:?}({:?}) → {}",
            node_id.index(),
            op,
            const_values,
            result
        );
    }

    // Replace this node with a Const (same single output, no inputs).
    func.nodes[node_id].kind = NodeKind::Simple(IrOp::Const { value: result });
    func.nodes[node_id].inputs.clear();
    true
}

/// Try algebraic simplification when one input is a known constant.
///
/// Identity/absorbing rules:
///   Shl(x, 0) → x,  Shr(x, 0) → x,  Sar(x, 0) → x
///   Add(x, 0) → x,  Sub(x, 0) → x
///   Mul(x, 1) → x,  Mul(x, 0) → 0
///   Or(x, 0)  → x,  And(x, ~0) → x
///   And(x, 0) → 0,  Or(x, ~0)  → ~0
fn try_simplify_node(func: &mut IrFunc, node_id: NodeId) -> bool {
    let NodeKind::Simple(op) = &func.nodes[node_id].kind else {
        return false;
    };
    if matches!(op, IrOp::Const { .. }) {
        return false;
    }
    let op = op.clone();

    // Only handle binary ops with exactly 2 data inputs.
    let data_inputs: Vec<(usize, Option<u64>)> = func.nodes[node_id]
        .inputs
        .iter()
        .enumerate()
        .filter(|(_, inp)| inp.kind == PortKind::Data)
        .map(|(i, inp)| (i, resolve_to_constant(func, &inp.source)))
        .collect();

    if data_inputs.len() != 2 {
        return false;
    }

    let (idx_a, val_a) = data_inputs[0];
    let (idx_b, val_b) = data_inputs[1];

    // Try to simplify based on which input(s) are constant.
    let replacement: Option<SimplifyResult> = match (&op, val_a, val_b) {
        // Shift by zero → identity
        (IrOp::Shl | IrOp::Shr | IrOp::Sar, _, Some(0)) => Some(SimplifyResult::PassInput(idx_a)),

        // Add/Sub zero → identity
        (IrOp::Add, Some(0), _) => Some(SimplifyResult::PassInput(idx_b)),
        (IrOp::Add | IrOp::Sub, _, Some(0)) => Some(SimplifyResult::PassInput(idx_a)),

        // Mul by 1 → identity; Mul by 0 → zero
        (IrOp::Mul, Some(1), _) => Some(SimplifyResult::PassInput(idx_b)),
        (IrOp::Mul, _, Some(1)) => Some(SimplifyResult::PassInput(idx_a)),
        (IrOp::Mul, Some(0), _) | (IrOp::Mul, _, Some(0)) => Some(SimplifyResult::Const(0)),

        // Or with 0 → identity
        (IrOp::Or, Some(0), _) => Some(SimplifyResult::PassInput(idx_b)),
        (IrOp::Or, _, Some(0)) => Some(SimplifyResult::PassInput(idx_a)),

        // And with 0 → zero
        (IrOp::And, Some(0), _) | (IrOp::And, _, Some(0)) => Some(SimplifyResult::Const(0)),

        // Or with all-ones → all-ones
        (IrOp::Or, Some(u64::MAX), _) | (IrOp::Or, _, Some(u64::MAX)) => {
            Some(SimplifyResult::Const(u64::MAX))
        }

        // And with all-ones → identity
        (IrOp::And, Some(u64::MAX), _) => Some(SimplifyResult::PassInput(idx_b)),
        (IrOp::And, _, Some(u64::MAX)) => Some(SimplifyResult::PassInput(idx_a)),

        // Xor with 0 → identity
        (IrOp::Xor, Some(0), _) => Some(SimplifyResult::PassInput(idx_b)),
        (IrOp::Xor, _, Some(0)) => Some(SimplifyResult::PassInput(idx_a)),

        _ => None,
    };

    let Some(replacement) = replacement else {
        return false;
    };

    match replacement {
        SimplifyResult::Const(value) => {
            func.nodes[node_id].kind = NodeKind::Simple(IrOp::Const { value });
            func.nodes[node_id].inputs.clear();
        }
        SimplifyResult::PassInput(input_idx) => {
            // Replace this node with an Identity forwarding the specified input.
            let source = func.nodes[node_id].inputs[input_idx].source;
            func.nodes[node_id].kind = NodeKind::Simple(IrOp::Identity);
            func.nodes[node_id].inputs = vec![crate::InputPort {
                kind: PortKind::Data,
                source,
            }];
        }
    }
    true
}

enum SimplifyResult {
    Const(u64),
    PassInput(usize),
}

/// Fold nodes in a single region (non-recursive).
/// Used by the unroller to evaluate cloned body nodes after each iteration.
/// The `resolve_to_constant` function handles tracing through gamma branches,
/// so we don't need to recurse into sub-regions here.
pub fn fold_nodes_in_region(func: &mut IrFunc, region: RegionId) {
    loop {
        let nodes = func.regions[region].nodes.clone();
        let mut changed = false;
        for &nid in &nodes {
            if try_fold_node(func, nid, false) || try_simplify_node(func, nid) {
                changed = true;
            }
        }
        if !changed {
            break;
        }
    }
}

/// Resolve a PortSource to a constant value, tracing through
/// gamma branch args and theta loop-invariant args.
pub fn resolve_to_constant(func: &IrFunc, source: &PortSource) -> Option<u64> {
    resolve_to_constant_inner(func, source, 0)
}

fn resolve_to_constant_inner(func: &IrFunc, source: &PortSource, depth: usize) -> Option<u64> {
    if depth > 32 {
        return None;
    }
    match source {
        PortSource::Node(out_ref) => match &func.nodes[out_ref.node].kind {
            NodeKind::Simple(IrOp::Const { value }) => Some(*value),
            NodeKind::Gamma { regions } => {
                // If all branches produce the same constant at this output index,
                // the gamma output is that constant regardless of which branch is taken.
                let output_index = out_ref.index as usize;
                let mut common: Option<u64> = None;
                let mut all_agree = true;
                for &region_id in regions {
                    let result_id = *func.regions[region_id].results.get(output_index)?;
                    let result_source = &func.region_results[result_id].source;
                    match resolve_to_constant_inner(func, result_source, depth + 1) {
                        Some(branch_val) => match common {
                            None => common = Some(branch_val),
                            Some(v) if v == branch_val => {}
                            _ => {
                                all_agree = false;
                                break;
                            }
                        },
                        None => {
                            all_agree = false;
                            break;
                        }
                    }
                }
                if all_agree {
                    return common;
                }
                // Branches disagree or some can't be resolved — try evaluating
                // the gamma predicate. If it resolves to a constant, select the
                // appropriate branch.
                let node = &func.nodes[out_ref.node];
                let pred_source = &node.inputs[0].source;
                let pred_val = resolve_to_constant_inner(func, pred_source, depth + 1)?;
                let branch_idx = (pred_val as usize).min(regions.len() - 1);
                let result_id = *func.regions[regions[branch_idx]]
                    .results
                    .get(output_index)?;
                let result_source = &func.region_results[result_id].source;
                resolve_to_constant_inner(func, result_source, depth + 1)
            }
            _ => None,
        },
        PortSource::RegionArg(arg_ref) => {
            let arg_index = func.regions[arg_ref.region]
                .args
                .iter()
                .position(|a| *a == arg_ref.arg)?;

            // Find the node that owns this region.
            let owner = func.find_region_owner(arg_ref.region)?;

            match &func.nodes[owner].kind {
                NodeKind::Gamma { .. } => {
                    // Do not treat gamma branch args as globally constant.
                    // They can be constant within a branch as data while still
                    // being unsafe to reuse for control-sensitive folding in a
                    // downstream gamma or comparison.
                    None
                }
                NodeKind::Theta { body, .. } => {
                    // Theta body arg[K] corresponds to theta input[K].
                    // Only resolve if it's loop-invariant: body result[K+1]
                    // (skip predicate at result[0]) passes back arg[K].
                    let input = func.nodes[owner].inputs.get(arg_index)?;
                    let body_region = *body;
                    let result_index = arg_index + 1; // skip predicate
                    let result_id = func.regions[body_region].results.get(result_index)?;
                    let result_source = &func.region_results[*result_id].source;
                    match result_source {
                        PortSource::RegionArg(RegionArgRef { region, arg })
                            if *region == body_region && *arg == arg_ref.arg =>
                        {
                            resolve_to_constant_inner(func, &input.source, depth + 1)
                        }
                        _ => None, // Not loop-invariant — can't resolve.
                    }
                }
                NodeKind::Lambda { .. } => None, // Lambda args come from apply sites.
                _ => None,
            }
        }
    }
}

/// Evaluate a pure arithmetic/comparison operation on constant inputs.
pub fn evaluate_op(op: &IrOp, inputs: &[u64]) -> Option<u64> {
    match (op, inputs) {
        (IrOp::Const { value }, []) => Some(*value),
        (IrOp::Add, [a, b]) => Some(a.wrapping_add(*b)),
        (IrOp::Sub, [a, b]) => Some(a.wrapping_sub(*b)),
        (IrOp::Mul, [a, b]) => Some(a.wrapping_mul(*b)),
        (IrOp::And, [a, b]) => Some(a & b),
        (IrOp::Or, [a, b]) => Some(a | b),
        (IrOp::Xor, [a, b]) => Some(a ^ b),
        (IrOp::Shl, [a, b]) => {
            if *b >= 64 {
                Some(0)
            } else {
                Some(a.wrapping_shl(*b as u32))
            }
        }
        (IrOp::Shr, [a, b]) => {
            if *b >= 64 {
                Some(0)
            } else {
                Some(a.wrapping_shr(*b as u32))
            }
        }
        (IrOp::Sar, [a, b]) => {
            if *b >= 64 {
                Some((*a as i64 >> 63) as u64)
            } else {
                Some((*a as i64).wrapping_shr(*b as u32) as u64)
            }
        }
        (IrOp::CmpEq, [a, b]) => Some(u64::from(a == b)),
        (IrOp::CmpNe, [a, b]) => Some(u64::from(a != b)),
        (IrOp::CmpLt, [a, b]) => Some(u64::from(a < b)),
        (IrOp::CmpLe, [a, b]) => Some(u64::from(a <= b)),
        (IrOp::CmpGt, [a, b]) => Some(u64::from(a > b)),
        (IrOp::CmpGe, [a, b]) => Some(u64::from(a >= b)),
        (IrOp::SignExtend { from_width }, [v]) => {
            let bits = (from_width.bytes() * 8) as u64;
            if bits >= 64 {
                Some(*v)
            } else {
                let mask = 1u64 << (bits - 1);
                Some((*v ^ mask).wrapping_sub(mask))
            }
        }
        _ => None, // Non-pure ops or wrong arity.
    }
}
