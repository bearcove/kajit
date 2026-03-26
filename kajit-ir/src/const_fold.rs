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

use crate::{
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
            if try_fold_node(func, nid, debug) {
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

/// Fold nodes in a single region (non-recursive).
/// Used by the unroller to evaluate cloned body nodes after each iteration.
/// The `resolve_to_constant` function handles tracing through gamma branches,
/// so we don't need to recurse into sub-regions here.
pub fn fold_nodes_in_region(func: &mut IrFunc, region: RegionId) {
    loop {
        let nodes = func.regions[region].nodes.clone();
        let mut changed = false;
        for &nid in &nodes {
            if try_fold_node(func, nid, false) {
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
            NodeKind::Simple(IrOp::ReadFromSlot { slot }) if out_ref.index == 0 => {
                // Trace backwards along the cursor state chain to find the most
                // recent WriteToSlot for this slot.
                resolve_slot_value(func, out_ref.node, slot.index(), depth)
            }
            NodeKind::Gamma { regions } => {
                // If all branches produce the same constant at this output index,
                // the gamma output is that constant regardless of which branch is taken.
                let output_index = out_ref.index as usize;
                let mut common: Option<u64> = None;
                for &region_id in regions {
                    let result_id = *func.regions[region_id].results.get(output_index)?;
                    let result_source = &func.region_results[result_id].source;
                    let branch_val = resolve_to_constant_inner(func, result_source, depth + 1)?;
                    match common {
                        None => common = Some(branch_val),
                        Some(v) if v == branch_val => {}
                        _ => return None, // Branches disagree
                    }
                }
                common
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
                    // Gamma branch arg[K] corresponds to gamma input[K+1] (skip predicate).
                    let input = func.nodes[owner].inputs.get(arg_index + 1)?;
                    resolve_to_constant_inner(func, &input.source, depth + 1)
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

/// Walk backwards along the cursor state chain from a ReadFromSlot node
/// to find the most recent WriteToSlot for the same slot index.
/// Returns the constant value if the write's data input is a known constant.
fn resolve_slot_value(func: &IrFunc, read_node: NodeId, slot: usize, depth: usize) -> Option<u64> {
    if depth > 64 {
        return None;
    }

    // ReadFromSlot's state input (the cursor chain) tells us what happened before.
    let node = &func.nodes[read_node];
    let state_input = node.inputs.iter().find(|inp| inp.kind != PortKind::Data)?;
    let mut current = state_input.source;

    // Walk back through the state chain, looking for WriteToSlot(slot).
    for _ in 0..256 {
        match current {
            PortSource::Node(ref out_ref) => {
                let prev_node = &func.nodes[out_ref.node];
                match &prev_node.kind {
                    NodeKind::Simple(IrOp::WriteToSlot { slot: s }) if s.index() == slot => {
                        // Found the matching write. Resolve its data input.
                        let data_input = prev_node
                            .inputs
                            .iter()
                            .find(|inp| inp.kind == PortKind::Data)?;
                        return resolve_to_constant_inner(func, &data_input.source, depth + 1);
                    }
                    _ => {
                        // Not our slot write — keep walking back along the state chain.
                        let prev_state_input = prev_node
                            .inputs
                            .iter()
                            .find(|inp| inp.kind != PortKind::Data);
                        match prev_state_input {
                            Some(inp) => current = inp.source,
                            None => return None,
                        }
                    }
                }
            }
            PortSource::RegionArg(_) => {
                // Reached the beginning of the region — try to resolve through parent.
                return resolve_to_constant_inner(func, &current, depth + 1);
            }
        }
    }

    None
}

/// Like `resolve_to_constant` but also resolves gamma outputs where the non-error
/// branch produces a constant. Used by the unroller to compute iteration values
/// when the error-exit branch produces a meaningless value.
pub fn resolve_to_constant_skip_errors(func: &IrFunc, source: &PortSource) -> Option<u64> {
    resolve_skip_errors_inner(func, source, 0)
}

fn resolve_skip_errors_inner(func: &IrFunc, source: &PortSource, depth: usize) -> Option<u64> {
    if depth > 32 {
        return None;
    }
    match source {
        PortSource::Node(out_ref) => match &func.nodes[out_ref.node].kind {
            NodeKind::Simple(IrOp::Const { value }) => Some(*value),
            NodeKind::Gamma { regions } => {
                // Skip branches with ErrorExit, require remaining branches to agree.
                let output_index = out_ref.index as usize;
                let mut common: Option<u64> = None;
                let mut any_non_error = false;
                for &region_id in regions {
                    if region_has_error_exit(func, region_id) {
                        continue;
                    }
                    any_non_error = true;
                    let result_id = *func.regions[region_id].results.get(output_index)?;
                    let result_source = &func.region_results[result_id].source;
                    let branch_val = resolve_skip_errors_inner(func, result_source, depth + 1)?;
                    match common {
                        None => common = Some(branch_val),
                        Some(v) if v == branch_val => {}
                        _ => return None,
                    }
                }
                if any_non_error { common } else { None }
            }
            _ => None,
        },
        PortSource::RegionArg(arg_ref) => {
            let arg_index = func.regions[arg_ref.region]
                .args
                .iter()
                .position(|a| *a == arg_ref.arg)?;
            let owner = func.find_region_owner(arg_ref.region)?;
            match &func.nodes[owner].kind {
                NodeKind::Gamma { .. } => {
                    let input = func.nodes[owner].inputs.get(arg_index + 1)?;
                    resolve_skip_errors_inner(func, &input.source, depth + 1)
                }
                NodeKind::Theta { body, .. } => {
                    let input = func.nodes[owner].inputs.get(arg_index)?;
                    let body_region = *body;
                    let result_index = arg_index + 1;
                    let result_id = func.regions[body_region].results.get(result_index)?;
                    let result_source = &func.region_results[*result_id].source;
                    match result_source {
                        PortSource::RegionArg(RegionArgRef { region, arg })
                            if *region == body_region && *arg == arg_ref.arg =>
                        {
                            resolve_skip_errors_inner(func, &input.source, depth + 1)
                        }
                        _ => None,
                    }
                }
                _ => None,
            }
        }
    }
}

/// Check if a region (or any of its sub-regions) contains an ErrorExit node.
pub fn region_has_error_exit(func: &IrFunc, region: RegionId) -> bool {
    for &nid in &func.regions[region].nodes {
        match &func.nodes[nid].kind {
            NodeKind::Simple(IrOp::ErrorExit { .. }) => return true,
            NodeKind::Gamma { regions } => {
                // If ALL gamma branches have ErrorExit, this gamma always errors.
                // But if only some branches error, the gamma itself doesn't always error.
                // For our purposes, even a single ErrorExit in the region counts.
                for &sub in regions {
                    if region_has_error_exit(func, sub) {
                        return true;
                    }
                }
            }
            NodeKind::Theta { body, .. } => {
                if region_has_error_exit(func, *body) {
                    return true;
                }
            }
            _ => {}
        }
    }
    false
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
        (IrOp::ZigzagDecode { wide: false }, [v]) => {
            let v32 = *v as u32;
            Some(((v32 >> 1) ^ (v32.wrapping_neg() >> 31).wrapping_mul(v32 & 1)) as u64)
        }
        (IrOp::ZigzagDecode { wide: true }, [v]) => {
            let s = *v as i64;
            Some(((s >> 1) ^ -(s & 1)) as u64)
        }
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
