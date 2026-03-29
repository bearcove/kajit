//! Simplify gamma nodes.
//!
//! Three optimizations:
//! 1. **All-passthrough elimination**: If every branch produces the same
//!    passthrough results, the gamma is a no-op and can be removed.
//! 2. **Constant predicate folding**: If the gamma's predicate is a known
//!    constant, inline the selected branch and discard the others.
//! 3. **Empty branch coalescing**: If a branch is empty (no nodes, all results
//!    are passthroughs), try to simplify the control flow.

use std::collections::HashMap;

use crate::{
    ArgId, IrFunc, IrOp, NodeId, NodeKind, OutputRef, PortSource, RegionArgRef, RegionId, ResultId,
};

/// Try to resolve a PortSource to a constant value.
///
/// Handles:
/// 1. Direct Const node references
/// 2. Region args that trace back to a theta input which is a constant
///    AND the theta result passes the same arg through (loop-invariant constant)
fn resolve_constant_source(func: &IrFunc, source: &PortSource) -> Option<usize> {
    match source {
        PortSource::Node(output_ref) => match &func.nodes[output_ref.node].kind {
            NodeKind::Simple(IrOp::Const { value }) => Some(*value as usize),
            _ => None,
        },
        PortSource::RegionArg(RegionArgRef { region, arg }) => {
            // Find the arg's index in the region's arg list
            let arg_index = func.regions[*region].args.iter().position(|a| a == arg)?;

            // Find the node that owns this region (must be a theta or gamma)
            let owner_node = func.find_region_owner(*region)?;

            let node = &func.nodes[owner_node];
            match &node.kind {
                NodeKind::Theta { body, .. } => {
                    // Theta: arg[K] corresponds to input[K]
                    let input = node.inputs.get(arg_index)?;
                    // Check that the theta result at this position passes the same arg through
                    // (loop-invariant: result[K+1] == RegionArg(body, arg[K]))
                    // +1 to skip predicate: body.results[0] is the predicate,
                    // data results start at body.results[1].
                    let result_id = func.regions[*body].results.get(arg_index + 1)?;
                    let result_source = &func.region_results[*result_id].source;
                    let is_loop_invariant = matches!(
                        result_source,
                        PortSource::RegionArg(RegionArgRef { region: r, arg: a })
                        if *r == *body && *a == *arg
                    );
                    if !is_loop_invariant {
                        return None;
                    }
                    // Recurse: check if the theta input is a constant
                    resolve_constant_source(func, &input.source)
                }
                NodeKind::Gamma { regions: _ } => {
                    // Gamma branch: arg[K] corresponds to gamma input[K+1] (skip predicate)
                    let input = node.inputs.get(arg_index + 1)?;
                    resolve_constant_source(func, &input.source)
                }
                _ => None,
            }
        }
    }
}

/// Entry point: simplify gamma nodes until fixed point.
pub fn simplify_trivial_gammas(func: &mut IrFunc) {
    let debug = std::env::var("KAJIT_DEBUG_GAMMA").is_ok();
    let mut total_eliminated = 0;

    loop {
        let region_ids: Vec<RegionId> = func.regions.iter().map(|(rid, _)| rid).collect();
        let mut changed = false;

        for rid in region_ids {
            if simplify_in_region(func, rid, debug) {
                changed = true;
                total_eliminated += 1;
                break; // restart after mutation
            }
        }

        if !changed {
            break;
        }
    }

    if debug && total_eliminated > 0 {
        eprintln!("[simplify_gamma] eliminated {} gammas", total_eliminated);
    }
}

/// Try to simplify one gamma in `region_id`. Returns true if a change was made.
fn simplify_in_region(func: &mut IrFunc, region_id: RegionId, debug: bool) -> bool {
    let nodes = func.regions[region_id].nodes.clone();

    for &node_id in &nodes {
        // Skip nodes that have been moved to a different region by a previous fold.
        if func.nodes[node_id].region != region_id {
            continue;
        }
        let NodeKind::Gamma { regions } = &func.nodes[node_id].kind else {
            continue;
        };
        let regions = regions.clone();

        // Try constant predicate folding first (higher impact).
        if fold_constant_predicate(func, region_id, node_id, &regions) {
            if debug {
                eprintln!(
                    "[simplify_gamma] folded constant predicate in gamma n{}",
                    node_id.index()
                );
            }
            return true;
        }

        // Try all-passthrough elimination.
        if eliminate_all_passthrough(func, region_id, node_id, &regions) {
            if debug {
                eprintln!(
                    "[simplify_gamma] eliminated all-passthrough gamma n{}",
                    node_id.index()
                );
            }
            return true;
        }

        // Try empty branch coalescing.
        if coalesce_empty_branch(func, region_id, node_id, &regions, debug) {
            return true;
        }
    }

    false
}

/// If the gamma's predicate is a known constant, inline the selected branch.
fn fold_constant_predicate(
    func: &mut IrFunc,
    parent_region: RegionId,
    gamma_node: NodeId,
    branch_regions: &[RegionId],
) -> bool {
    // Check if predicate is a constant (direct or through theta arg).
    let pred_source = func.nodes[gamma_node].inputs[0].source;
    let const_value = resolve_constant_source(func, &pred_source);

    let Some(const_value) = const_value else {
        return false;
    };

    // Determine which branch to keep (clamp to last branch for out-of-range values).
    let branch_index = const_value.min(branch_regions.len() - 1);
    let selected_region = branch_regions[branch_index];

    // Build RegionArg → gamma input source mapping.
    // Region args[k] corresponds to gamma inputs[k+1] (skip predicate).
    let branch_args: Vec<ArgId> = func.regions[selected_region].args.clone();
    let mut arg_to_source: HashMap<ArgId, PortSource> = HashMap::new();
    for (k, &arg_id) in branch_args.iter().enumerate() {
        arg_to_source.insert(arg_id, func.nodes[gamma_node].inputs[k + 1].source);
    }

    // Move branch nodes to parent region, rewriting their RegionArg inputs.
    let branch_nodes: Vec<NodeId> = func.regions[selected_region].nodes.clone();
    for &nid in &branch_nodes {
        rewrite_region_arg_inputs(func, nid, &arg_to_source);
        func.nodes[nid].region = parent_region;
    }

    // Insert branch nodes into parent region just before the gamma.
    let gamma_pos = func.regions[parent_region]
        .nodes
        .iter()
        .position(|&nid| nid == gamma_node)
        .unwrap();
    for (offset, &nid) in branch_nodes.iter().enumerate() {
        func.regions[parent_region]
            .nodes
            .insert(gamma_pos + offset, nid);
    }

    // Replace gamma outputs with the selected branch's result sources
    // (rewritten to resolve RegionArgs to gamma inputs).
    let branch_results: Vec<ResultId> = func.regions[selected_region].results.clone();
    for (output_idx, &result_id) in branch_results.iter().enumerate() {
        let result_source = func.region_results[result_id].source;
        let replacement = rewrite_source(&result_source, &arg_to_source);
        let output_ref = OutputRef {
            node: gamma_node,
            index: output_idx as u16,
        };
        replace_all_uses(func, output_ref, replacement);
    }

    // Remove gamma node from parent region.
    func.regions[parent_region]
        .nodes
        .retain(|&nid| nid != gamma_node);

    true
}

/// If all branches produce the same passthrough results (same arg position
/// for every result across all branches), and no branch has side effects,
/// remove the gamma entirely.
fn eliminate_all_passthrough(
    func: &mut IrFunc,
    parent_region: RegionId,
    gamma_node: NodeId,
    branch_regions: &[RegionId],
) -> bool {
    let first_region = branch_regions[0];
    let total_results = func.regions[first_region].results.len();

    // For each result index, check if ALL branches reference the same arg position.
    let mut result_to_arg_pos: Vec<usize> = Vec::with_capacity(total_results);

    for result_idx in 0..total_results {
        let mut common_arg_pos: Option<usize> = None;

        for &branch_region in branch_regions {
            let result_id = func.regions[branch_region].results[result_idx];
            let result = &func.region_results[result_id];

            match result.source {
                PortSource::RegionArg(arg_ref) => {
                    let arg_pos = func.regions[branch_region]
                        .args
                        .iter()
                        .position(|&a| a == arg_ref.arg);

                    match (arg_pos, common_arg_pos) {
                        (Some(pos), None) => common_arg_pos = Some(pos),
                        (Some(pos), Some(prev)) if prev == pos => {}
                        _ => return false,
                    }
                }
                PortSource::Node(_) => return false,
            }
        }

        result_to_arg_pos.push(common_arg_pos.unwrap());
    }

    // Check that no branch contains side-effecting nodes.
    if branches_have_side_effects(func, branch_regions) {
        return false;
    }

    // Replace each gamma output with the corresponding gamma input.
    for (result_idx, &arg_pos) in result_to_arg_pos.iter().enumerate() {
        let gamma_input_idx = arg_pos + 1; // skip predicate
        let replacement = func.nodes[gamma_node].inputs[gamma_input_idx].source;
        let output_ref = OutputRef {
            node: gamma_node,
            index: result_idx as u16,
        };
        replace_all_uses(func, output_ref, replacement);
    }

    // Remove gamma from parent region.
    func.regions[parent_region]
        .nodes
        .retain(|&nid| nid != gamma_node);

    true
}

/// Rewrite a PortSource, replacing RegionArg references using the mapping.
fn rewrite_source(source: &PortSource, arg_map: &HashMap<ArgId, PortSource>) -> PortSource {
    match source {
        PortSource::RegionArg(arg_ref) => {
            if let Some(&replacement) = arg_map.get(&arg_ref.arg) {
                replacement
            } else {
                *source
            }
        }
        PortSource::Node(_) => *source,
    }
}

/// Rewrite all RegionArg input sources on a node using the mapping.
fn rewrite_region_arg_inputs(
    func: &mut IrFunc,
    node_id: NodeId,
    arg_map: &HashMap<ArgId, PortSource>,
) {
    for input in &mut func.nodes[node_id].inputs {
        input.source = rewrite_source(&input.source, arg_map);
    }
}

/// Check whether any branch region (or its nested sub-regions) contains
/// nodes with side effects.
fn branches_have_side_effects(func: &IrFunc, regions: &[RegionId]) -> bool {
    for &region_id in regions {
        for &nid in &func.regions[region_id].nodes {
            match &func.nodes[nid].kind {
                NodeKind::Simple(op) if op.has_side_effects() => return true,
                NodeKind::Gamma { regions } => {
                    if branches_have_side_effects(func, regions) {
                        return true;
                    }
                }
                NodeKind::Theta { body, .. } => {
                    if branches_have_side_effects(func, &[*body]) {
                        return true;
                    }
                }
                NodeKind::Lambda { body, .. } => {
                    if branches_have_side_effects(func, &[*body]) {
                        return true;
                    }
                }
                NodeKind::Apply { .. } => return true,
                _ => {}
            }
        }
    }
    false
}

/// Try to simplify gammas where one or more branches are empty (no nodes).
/// Returns true if a simplification was made.
fn coalesce_empty_branch(
    func: &IrFunc,
    _parent_region: RegionId,
    gamma_node: NodeId,
    branch_regions: &[RegionId],
    debug: bool,
) -> bool {
    // Check which branches are empty (no nodes)
    let mut empty_branches = Vec::new();
    let mut non_empty_branches = Vec::new();

    for (idx, &branch_region) in branch_regions.iter().enumerate() {
        if func.regions[branch_region].nodes.is_empty() {
            empty_branches.push(idx);
        } else {
            non_empty_branches.push(idx);
        }
    }

    // Debug logging to understand patterns
    if debug && !empty_branches.is_empty() {
        eprintln!(
            "[simplify_gamma] gamma n{}: {} branches, {} empty, {} non-empty",
            gamma_node.index(),
            branch_regions.len(),
            empty_branches.len(),
            non_empty_branches.len()
        );

        // Check if empty branches are passthrough
        for &idx in &empty_branches {
            let region = branch_regions[idx];
            let args = &func.regions[region].args;
            let results = &func.regions[region].results;
            let all_passthrough = results.iter().enumerate().all(|(result_idx, &result_id)| {
                match func.region_results[result_id].source {
                    PortSource::RegionArg(arg_ref) => args.get(result_idx) == Some(&arg_ref.arg),
                    _ => false,
                }
            });
            eprintln!("  branch {} empty, passthrough={}", idx, all_passthrough);
        }
    }

    // TODO: Actually implement optimizations for empty branches
    // For now, just report patterns so we can see what's happening
    false
}

/// Replace all uses of `from` output with `to` source, across all node inputs
/// and region results in the function.
fn replace_all_uses(func: &mut IrFunc, from: OutputRef, to: PortSource) {
    let from_source = PortSource::Node(from);
    if from_source == to {
        return;
    }

    let node_ids: Vec<NodeId> = func.nodes.iter().map(|(nid, _)| nid).collect();
    for nid in node_ids {
        for input in &mut func.nodes[nid].inputs {
            if input.source == from_source {
                input.source = to;
            }
        }
    }

    let result_ids: Vec<ResultId> = func.region_results.iter().map(|(rid, _)| rid).collect();
    for rid in result_ids {
        if func.region_results[rid].source == from_source {
            func.region_results[rid].source = to;
        }
    }
}
