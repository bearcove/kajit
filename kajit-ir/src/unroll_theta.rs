//! Bounded-theta unrolling pass.
//!
//! Converts theta nodes with `max_iterations: Some(n)` into a gamma cascade
//! (nested if/else chain). Each iteration is a copy of the body, with the
//! predicate check becoming a gamma branch.
//!
//! Before:
//! ```text
//! theta(init_vars) max_iterations=3 {
//!   body: [pred, new_vars...] = f(args)
//! }
//! ```
//!
//! After:
//! ```text
//! [pred1, vars1] = f(init_vars)
//! gamma(pred1) {
//!   0: result = vars1
//!   1: [pred2, vars2] = f(vars1)
//!      gamma(pred2) {
//!        0: result = vars2
//!        1: [pred3, vars3] = f(vars2)
//!           result = vars3  // max reached, unconditional exit
//!      }
//! }
//! ```

use crate::ir_passes::{CloneCtx, clone_region_into, remap_source};
use crate::{IrFunc, NodeId, NodeKind, OutputRef, PortSource, Region, RegionArg};

fn region_has_error_exit(func: &IrFunc, region: crate::RegionId) -> bool {
    crate::const_fold::region_has_error_exit(func, region)
}

/// Maximum iterations we're willing to unroll.
const MAX_UNROLL: u32 = 10;

/// Diagnostic info about a bounded theta considered for unrolling.
#[derive(Debug)]
pub struct UnrollCandidate {
    pub node_id: NodeId,
    pub max_iterations: u32,
    pub nesting_depth: u32,
    pub body_node_count: usize,
    pub body_gamma_count: usize,
    pub carried_values: usize,
    pub has_calls: bool,
    pub has_inner_bounded_theta: bool,
    pub estimated_node_growth: usize,
    pub estimated_carried_cost: usize,
    pub decision: UnrollDecision,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum UnrollDecision {
    Unroll,
    Skip,
}

/// Analyze a theta node to produce unroll diagnostics.
fn analyze_theta(func: &IrFunc, node_id: NodeId) -> Option<UnrollCandidate> {
    let NodeKind::Theta {
        body,
        max_iterations: Some(max_iter),
        ..
    } = &func.nodes[node_id].kind
    else {
        return None;
    };
    if *max_iter > MAX_UNROLL {
        return None;
    }

    let body_region = &func.regions[*body];
    let body_node_count = count_subtree_nodes(func, *body);
    let body_gamma_count = count_subtree_gammas(func, *body);
    let carried_values = body_region.args.len();

    let has_calls = subtree_has_calls(func, *body);
    let has_inner_bounded_theta = subtree_has_bounded_theta(func, *body);

    // Nesting depth: count how many thetas are above this node
    let nesting_depth = compute_nesting_depth(func, node_id);

    let estimated_node_growth = body_node_count * (*max_iter as usize);
    let estimated_carried_cost = carried_values * (*max_iter as usize);

    // Policy: only unroll top-level thetas (depth=0).
    // Inner thetas (depth>0) are varint decoders inside outer loops
    // that cause code growth, register pressure, and stack spills
    // when unrolled.
    let decision = if nesting_depth == 0 {
        UnrollDecision::Unroll
    } else {
        UnrollDecision::Skip
    };

    Some(UnrollCandidate {
        node_id,
        max_iterations: *max_iter,
        nesting_depth,
        body_node_count,
        body_gamma_count,
        carried_values,
        has_calls,
        has_inner_bounded_theta,
        estimated_node_growth,
        estimated_carried_cost,
        decision,
    })
}

fn count_subtree_nodes(func: &IrFunc, region_id: crate::RegionId) -> usize {
    let region = &func.regions[region_id];
    let mut count = region.nodes.len();
    for &nid in &region.nodes {
        match &func.nodes[nid].kind {
            NodeKind::Gamma { regions } => {
                for &rid in regions {
                    count += count_subtree_nodes(func, rid);
                }
            }
            NodeKind::Theta { body, .. } => {
                count += count_subtree_nodes(func, *body);
            }
            _ => {}
        }
    }
    count
}

fn count_subtree_gammas(func: &IrFunc, region_id: crate::RegionId) -> usize {
    let region = &func.regions[region_id];
    let mut count = 0;
    for &nid in &region.nodes {
        match &func.nodes[nid].kind {
            NodeKind::Gamma { regions } => {
                count += 1;
                for &rid in regions {
                    count += count_subtree_gammas(func, rid);
                }
            }
            NodeKind::Theta { body, .. } => {
                count += count_subtree_gammas(func, *body);
            }
            _ => {}
        }
    }
    count
}

fn subtree_has_calls(func: &IrFunc, region_id: crate::RegionId) -> bool {
    let region = &func.regions[region_id];
    for &nid in &region.nodes {
        match &func.nodes[nid].kind {
            NodeKind::Apply { .. } => return true,
            NodeKind::Gamma { regions } => {
                for &rid in regions {
                    if subtree_has_calls(func, rid) {
                        return true;
                    }
                }
            }
            NodeKind::Theta { body, .. } => {
                if subtree_has_calls(func, *body) {
                    return true;
                }
            }
            _ => {}
        }
    }
    false
}

fn subtree_has_bounded_theta(func: &IrFunc, region_id: crate::RegionId) -> bool {
    let region = &func.regions[region_id];
    for &nid in &region.nodes {
        match &func.nodes[nid].kind {
            NodeKind::Theta {
                max_iterations: Some(_),
                ..
            } => return true,
            NodeKind::Gamma { regions } => {
                for &rid in regions {
                    if subtree_has_bounded_theta(func, rid) {
                        return true;
                    }
                }
            }
            NodeKind::Theta { body, .. } => {
                if subtree_has_bounded_theta(func, *body) {
                    return true;
                }
            }
            _ => {}
        }
    }
    false
}

fn compute_nesting_depth(func: &IrFunc, node_id: NodeId) -> u32 {
    let mut depth = 0u32;
    let mut region = func.nodes[node_id].region;
    // Walk up from the node's region to the root, counting thetas
    loop {
        // Find the node that owns this region
        let _owner = func.regions.iter().find_map(|(_, _r)| {
            // Skip: we're looking for a node whose body/branch IS this region
            None::<NodeId>
        });
        // Simpler: check if any node has this region as a body
        let mut found_owner = false;
        for (_, r) in func.regions.iter() {
            for &nid in &r.nodes {
                match &func.nodes[nid].kind {
                    NodeKind::Theta { body, .. } if *body == region => {
                        depth += 1;
                        region = func.nodes[nid].region;
                        found_owner = true;
                    }
                    NodeKind::Gamma { regions } => {
                        if regions.contains(&region) {
                            region = func.nodes[nid].region;
                            found_owner = true;
                        }
                    }
                    NodeKind::Lambda { body, .. } if *body == region => {
                        // Reached the lambda root — done
                        return depth;
                    }
                    _ => {}
                }
                if found_owner {
                    break;
                }
            }
            if found_owner {
                break;
            }
        }
        if !found_owner {
            return depth;
        }
    }
}

/// Run the bounded-theta unrolling pass.
///
/// Returns true if any theta was unrolled.
pub fn unroll_bounded_thetas(func: &mut IrFunc) -> bool {
    let mut changed = false;

    // Collect and analyze all unroll candidates
    let mut candidates: Vec<UnrollCandidate> = Vec::new();
    for (_region_id, region) in func.regions.iter() {
        for &node_id in &region.nodes {
            if let Some(candidate) = analyze_theta(func, node_id) {
                candidates.push(candidate);
            }
        }
    }

    // Print diagnostics
    if std::env::var("KAJIT_UNROLL_DIAG").is_ok() || !candidates.is_empty() {
        for c in &candidates {
            eprintln!(
                "[unroll-diag] theta #{}: max_iter={}, depth={}, body_nodes={}, \
                 gammas={}, carried={}, calls={}, inner_theta={}, \
                 est_growth={}, est_carried_cost={}, decision={:?}",
                c.node_id.index(),
                c.max_iterations,
                c.nesting_depth,
                c.body_node_count,
                c.body_gamma_count,
                c.carried_values,
                c.has_calls,
                c.has_inner_bounded_theta,
                c.estimated_node_growth,
                c.estimated_carried_cost,
                c.decision,
            );
        }
    }

    // Unroll candidates that passed the policy
    let targets: Vec<(NodeId, u32)> = candidates
        .iter()
        .filter(|c| c.decision == UnrollDecision::Unroll)
        .map(|c| (c.node_id, c.max_iterations))
        .collect();

    for (theta_node_id, max_iter) in targets {
        unroll_one_theta(func, theta_node_id, max_iter);
        changed = true;
    }

    changed
}

fn unroll_one_theta(func: &mut IrFunc, theta_node_id: NodeId, max_iter: u32) {
    eprintln!(
        "[unroll] theta node #{}, max_iter={}",
        theta_node_id.index(),
        max_iter
    );
    let NodeKind::Theta { body, .. } = func.nodes[theta_node_id].kind else {
        return;
    };

    let parent_region = func.nodes[theta_node_id].region;
    let theta_inputs: Vec<_> = func.nodes[theta_node_id]
        .inputs
        .iter()
        .map(|ip| ip.source)
        .collect();
    let theta_output_count = func.nodes[theta_node_id].outputs.len();
    let body_result_count = func.regions[body].results.len();

    // Body region: results[0] = predicate, results[1..] = updated values
    // Body args correspond 1:1 to theta inputs
    let body_arg_count = func.regions[body].args.len();
    assert_eq!(
        body_arg_count,
        theta_inputs.len(),
        "theta body args must match theta inputs"
    );

    // The theta outputs correspond to body results[1..] (skipping predicate)
    assert_eq!(
        theta_output_count,
        body_result_count - 1,
        "theta outputs = body results minus predicate"
    );

    // Strategy: we'll inline N copies of the body into the parent region.
    // After each copy, we build a gamma to check the predicate.
    //
    // The first iteration uses the theta's inputs as the body's args.
    // Each subsequent iteration uses the previous body's results[1..].
    //
    // We replace the theta node with the gamma cascade.

    // Step 1: Inline first iteration's body into parent region
    let current_sources = theta_inputs.clone();

    // Seed the known environment from the theta's initial inputs.
    let mut known_env: Vec<Option<u64>> = current_sources
        .iter()
        .map(|src| crate::const_fold::resolve_to_constant(func, src))
        .collect();

    // We'll build nested gammas. The outermost result sources will replace
    // the theta node's outputs.
    let final_sources = build_unrolled_cascade(
        func,
        body,
        parent_region,
        &current_sources,
        max_iter,
        &mut known_env,
    );

    // Step 2: Replace theta outputs with the cascade results
    // We need to rewrite all uses of the theta's outputs to use the cascade's outputs.
    let _theta_outputs: Vec<OutputRef> = (0..theta_output_count as u16)
        .map(|i| OutputRef {
            node: theta_node_id,
            index: i,
        })
        .collect();

    // Find all uses of the theta's outputs and rewrite them
    for (_region_id, region) in func.regions.iter() {
        for &node_id in &region.nodes {
            if node_id == theta_node_id {
                continue;
            }
            for _input in &func.nodes[node_id].inputs {
                // Check later — need to collect first
            }
        }
    }

    // Actually, let's collect uses first, then rewrite
    let mut uses_to_rewrite: Vec<(NodeId, usize, PortSource)> = Vec::new();
    let mut result_uses_to_rewrite: Vec<(crate::ResultId, PortSource)> = Vec::new();

    for (_region_id, region) in func.regions.iter() {
        for &node_id in &region.nodes {
            if node_id == theta_node_id {
                continue;
            }
            for (input_idx, input) in func.nodes[node_id].inputs.iter().enumerate() {
                if let PortSource::Node(out_ref) = input.source
                    && out_ref.node == theta_node_id
                {
                    let new_source = final_sources[out_ref.index as usize];
                    uses_to_rewrite.push((node_id, input_idx, new_source));
                }
            }
        }
        for &result_id in &region.results {
            let result = &func.region_results[result_id];
            if let PortSource::Node(out_ref) = result.source
                && out_ref.node == theta_node_id
            {
                let new_source = final_sources[out_ref.index as usize];
                result_uses_to_rewrite.push((result_id, new_source));
            }
        }
    }

    // Apply rewrites
    for (node_id, input_idx, new_source) in uses_to_rewrite {
        func.nodes[node_id].inputs[input_idx].source = new_source;
    }
    for (result_id, new_source) in result_uses_to_rewrite {
        func.region_results[result_id].source = new_source;
    }

    // Remove theta from parent region's node list
    let before_count = func.regions[parent_region].nodes.len();
    func.regions[parent_region]
        .nodes
        .retain(|&nid| nid != theta_node_id);
    let after_count = func.regions[parent_region].nodes.len();
    eprintln!(
        "[unroll] removed theta from region: {} -> {} nodes. Has thetas: {}",
        before_count,
        after_count,
        func.regions[parent_region]
            .nodes
            .iter()
            .any(|&nid| matches!(func.nodes[nid].kind, NodeKind::Theta { .. }))
    );

    // Topologically sort the parent region's nodes.
    topological_sort_region(func, parent_region);

    // Debug: check for out-of-scope references
    for (region_id, region) in func.regions.iter() {
        let node_set: std::collections::HashSet<NodeId> = region.nodes.iter().copied().collect();
        for &nid in &region.nodes {
            for (i, input) in func.nodes[nid].inputs.iter().enumerate() {
                if let PortSource::Node(out) = input.source
                    && out.node != nid
                    && !node_set.contains(&out.node)
                {
                    // Source is not in this region — check if it's in a parent
                    let source_region = func.nodes[out.node].region;
                    if source_region != region_id {
                        eprintln!(
                            "[unroll] SCOPE: n{} (region {:?}) input[{}] refs n{} (region {:?})",
                            nid.index(),
                            region_id,
                            i,
                            out.node.index(),
                            source_region
                        );
                    }
                }
            }
        }
    }
}

/// Sort a region's nodes in topological order (dependencies before dependents).
fn topological_sort_region(func: &mut IrFunc, region_id: crate::RegionId) {
    use std::collections::HashSet;

    let nodes = func.regions[region_id].nodes.clone();
    let node_set: HashSet<NodeId> = nodes.iter().copied().collect();

    // Build adjacency: for each node, which other nodes in this region does it depend on?
    let mut deps: std::collections::HashMap<NodeId, Vec<NodeId>> = std::collections::HashMap::new();
    for &nid in &nodes {
        let mut node_deps = Vec::new();
        for input in &func.nodes[nid].inputs {
            if let PortSource::Node(out) = input.source
                && node_set.contains(&out.node)
                && out.node != nid
            {
                node_deps.push(out.node);
            }
        }
        deps.insert(nid, node_deps);
    }

    // Kahn's algorithm
    let mut in_degree: std::collections::HashMap<NodeId, usize> = std::collections::HashMap::new();
    for &nid in &nodes {
        in_degree.entry(nid).or_insert(0);
        for &_dep in &deps[&nid] {
            // dep is used by nid, so nid depends on dep
            // But we need: for each node, how many nodes depend on it
        }
    }

    // Recompute: in_degree[n] = how many of n's dependencies are in this region
    for &nid in &nodes {
        in_degree.insert(nid, deps[&nid].len());
    }

    let mut queue: std::collections::VecDeque<NodeId> = std::collections::VecDeque::new();
    for &nid in &nodes {
        if in_degree[&nid] == 0 {
            queue.push_back(nid);
        }
    }

    // Build reverse deps: for each node, which nodes depend on it?
    let mut rdeps: std::collections::HashMap<NodeId, Vec<NodeId>> =
        std::collections::HashMap::new();
    for &nid in &nodes {
        for &dep in &deps[&nid] {
            rdeps.entry(dep).or_default().push(nid);
        }
    }

    let mut sorted = Vec::with_capacity(nodes.len());
    while let Some(nid) = queue.pop_front() {
        sorted.push(nid);
        for &dependent in rdeps.get(&nid).unwrap_or(&Vec::new()) {
            let deg = in_degree.get_mut(&dependent).unwrap();
            *deg -= 1;
            if *deg == 0 {
                queue.push_back(dependent);
            }
        }
    }

    // If sorted has fewer nodes than the original, there's a cycle (shouldn't happen)
    if sorted.len() == nodes.len() {
        func.regions[region_id].nodes = sorted;
    }
}

/// Build the unrolled gamma cascade, returning the final output sources.
///
/// This recursively builds: execute body → gamma(pred) { 0: exit, 1: recurse }
///
/// `known_env[i]` = Some(v) if input_sources[i] is known to be constant v,
/// even if the RVSDG structure doesn't reflect this (e.g., after flowing
/// through gammas where branches disagree). Updated in-place for the next
/// iteration.
fn build_unrolled_cascade(
    func: &mut IrFunc,
    body_region: crate::RegionId,
    target_region: crate::RegionId,
    input_sources: &[PortSource],
    remaining_iterations: u32,
    known_env: &mut Vec<Option<u64>>,
) -> Vec<PortSource> {
    if remaining_iterations == 0 {
        // Max iterations reached — just return the current values
        // (skip predicate, return loop vars + state domains)
        return input_sources.to_vec();
    }

    // For known-constant loop variables, inject Const nodes in the target region
    // BEFORE cloning the body. This ensures the body sees constants directly,
    // bypassing the phase-order problem where values flow through gammas.
    let debug_scope = func.regions[target_region].debug_scope;
    let specialized_inputs: Vec<PortSource> = input_sources
        .iter()
        .enumerate()
        .map(|(i, src)| {
            if let Some(val) = known_env.get(i).copied().flatten() {
                // Check if the source is already a matching Const.
                let already_const = matches!(src,
                    PortSource::Node(out) if matches!(
                        &func.nodes[out.node].kind,
                        NodeKind::Simple(crate::IrOp::Const { value }) if *value == val
                    )
                );
                if already_const {
                    *src
                } else {
                    crate::const_fold::create_const_in_region(func, target_region, debug_scope, val)
                }
            } else {
                *src
            }
        })
        .collect();

    // Clone the body into target_region, substituting body args with specialized inputs
    let cloned_results =
        clone_body_into_region(func, body_region, target_region, &specialized_inputs);

    // Run const fold on the cloned nodes to evaluate constant expressions.
    crate::const_fold::fold_nodes_in_region(func, target_region);

    // cloned_results[0] = predicate, cloned_results[1..] = updated values
    let predicate = cloned_results[0];
    let updated_values: Vec<PortSource> = cloned_results[1..].to_vec();

    // Update known_env for the next iteration.
    // First try structural resolution. For values that can't be resolved
    // structurally (e.g., counter flowing through gammas where branches
    // disagree), evaluate the original body symbolically with the known_env
    // to compute what each loop variable becomes.
    let body_results = &func.regions[body_region].results.clone();
    let body_args = &func.regions[body_region].args.clone();
    let next_known: Vec<Option<u64>> = updated_values
        .iter()
        .enumerate()
        .map(|(i, src)| {
            // Try structural resolution on the cloned result first.
            if let Some(v) = crate::const_fold::resolve_to_constant(func, src) {
                return Some(v);
            }
            // Fall back to symbolic evaluation of the original body's result
            // expression under the current known_env.
            // Body result index is i+1 (skip predicate at index 0).
            let result_id = body_results[i + 1];
            let result_source = &func.region_results[result_id].source;
            eval_source_with_env(func, body_region, body_args, known_env, result_source)
        })
        .collect();
    *known_env = next_known;

    if remaining_iterations == 1 {
        // Last iteration — no need for a gamma, just return the values
        return updated_values;
    }

    let _exit_values = updated_values.clone();

    // Create gamma with 2 branches:
    // Branch 0 (pred==0, exit): pass through updated_values
    // Branch 1 (pred!=0, continue): recurse with remaining-1

    // Create branch 0 region (exit)
    let mut branch0_args = Vec::new();
    for source in &updated_values {
        let kind = match source {
            PortSource::Node(out) => func.nodes[out.node].outputs[out.index as usize].kind,
            PortSource::RegionArg(arg_ref) => func.region_args[arg_ref.arg].kind,
        };
        let arg_id = func.region_args.push(RegionArg {
            kind,
            vreg: None,
            debug_value: None,
        });
        branch0_args.push(arg_id);
    }
    let branch0_region = func.regions.push(Region {
        debug_scope,
        args: branch0_args.clone(),
        results: Vec::new(),
        nodes: Vec::new(),
    });
    // Branch 0 results: just pass through the args
    for &arg_id in branch0_args.iter() {
        let result_id = func.region_results.push(crate::RegionResult {
            kind: func.region_args[arg_id].kind,
            source: PortSource::RegionArg(crate::RegionArgRef {
                region: branch0_region,
                arg: arg_id,
            }),
        });
        func.regions[branch0_region].results.push(result_id);
    }

    // Create branch 1 region (continue)
    let mut branch1_args = Vec::new();
    for source in &updated_values {
        let kind = match source {
            PortSource::Node(out) => func.nodes[out.node].outputs[out.index as usize].kind,
            PortSource::RegionArg(arg_ref) => func.region_args[arg_ref.arg].kind,
        };
        let arg_id = func.region_args.push(RegionArg {
            kind,
            vreg: None,
            debug_value: None,
        });
        branch1_args.push(arg_id);
    }
    let branch1_region = func.regions.push(Region {
        debug_scope,
        args: branch1_args.clone(),
        results: Vec::new(),
        nodes: Vec::new(),
    });

    // Recursively build the remaining iterations inside branch 1.
    // The branch1 region args correspond to the gamma inputs (= updated_values).
    // The known_env already has the next iteration's known constants.
    let branch1_input_sources: Vec<PortSource> = branch1_args
        .iter()
        .map(|&arg_id| {
            PortSource::RegionArg(crate::RegionArgRef {
                region: branch1_region,
                arg: arg_id,
            })
        })
        .collect();
    let recursive_results = build_unrolled_cascade(
        func,
        body_region,
        branch1_region,
        &branch1_input_sources,
        remaining_iterations - 1,
        known_env,
    );

    // Set branch 1 results
    for source in &recursive_results {
        let kind = match source {
            PortSource::Node(out) => func.nodes[out.node].outputs[out.index as usize].kind,
            PortSource::RegionArg(arg_ref) => func.region_args[arg_ref.arg].kind,
        };
        let result_id = func.region_results.push(crate::RegionResult {
            kind,
            source: *source,
        });
        func.regions[branch1_region].results.push(result_id);
    }

    // Create the gamma node
    let regions = vec![branch0_region, branch1_region];

    // Gamma inputs: predicate + values to pass to branches
    let mut gamma_inputs = Vec::new();
    // First input = predicate
    gamma_inputs.push(crate::InputPort {
        kind: crate::PortKind::Data,
        source: predicate,
    });
    // Remaining inputs = updated values (passed to both branches as args)
    for &source in &updated_values {
        let kind = match source {
            PortSource::Node(out) => func.nodes[out.node].outputs[out.index as usize].kind,
            PortSource::RegionArg(arg_ref) => func.region_args[arg_ref.arg].kind,
        };
        gamma_inputs.push(crate::InputPort { kind, source });
    }

    // Gamma outputs: one per result from each branch (they must match)
    let gamma_output_count = updated_values.len();
    let mut gamma_outputs = Vec::new();
    for i in 0..gamma_output_count {
        let kind = match &updated_values[i] {
            PortSource::Node(out) => func.nodes[out.node].outputs[out.index as usize].kind,
            PortSource::RegionArg(arg_ref) => func.region_args[arg_ref.arg].kind,
        };
        gamma_outputs.push(crate::OutputPort {
            kind,
            vreg: None,
            debug_scope,
        });
    }

    let gamma_node_id = func.nodes.push(crate::Node {
        region: target_region,
        debug_scope,
        debug_value: None,
        inputs: gamma_inputs,
        outputs: gamma_outputs,
        kind: NodeKind::Gamma { regions },
    });
    func.regions[target_region].nodes.push(gamma_node_id);

    // Return gamma outputs as sources
    (0..gamma_output_count as u16)
        .map(|i| {
            PortSource::Node(OutputRef {
                node: gamma_node_id,
                index: i,
            })
        })
        .collect()
}

/// Clone the body region's computation into the target region,
/// substituting body args with the given input sources.
///
/// Returns the remapped body result sources.
fn clone_body_into_region(
    func: &mut IrFunc,
    body_region: crate::RegionId,
    target_region: crate::RegionId,
    input_sources: &[PortSource],
) -> Vec<PortSource> {
    let mut ctx = CloneCtx {
        top_old_region: body_region,
        top_arg_sources: input_sources.to_vec(),
        node_map: std::collections::HashMap::new(),
        region_map: std::collections::HashMap::new(),
        arg_map: std::collections::HashMap::new(),
        top_new_nodes: Vec::new(),
    };

    clone_region_into(func, body_region, target_region, &mut ctx);

    // Add cloned top-level nodes to the target region
    // (clone_region_into puts top-level nodes in ctx.top_new_nodes, not in the region)
    for &node_id in &ctx.top_new_nodes {
        func.regions[target_region].nodes.push(node_id);
    }

    // Return remapped result sources
    let results = func.regions[body_region].results.clone();
    results
        .iter()
        .map(|&result_id| {
            let source = func.region_results[result_id].source;
            remap_source(source, &ctx, func)
        })
        .collect()
}

/// Symbolically evaluate a source expression in the original body region
/// under a known environment for the body's args.
///
/// This is a mini partial evaluator: it walks the expression tree in the body,
/// and for each node:
/// - Const → return its value
/// - Pure op with all-known inputs → evaluate
/// - RegionArg → look up in known_env (by arg index in body)
/// - Gamma → evaluate predicate; if known, recurse into selected branch
/// - Otherwise → None (unknown)
fn eval_source_with_env(
    func: &IrFunc,
    body_region: crate::RegionId,
    body_args: &[crate::ArgId],
    known_env: &[Option<u64>],
    source: &PortSource,
) -> Option<u64> {
    eval_with_env_inner(func, body_region, body_args, known_env, source, 0)
}

fn eval_with_env_inner(
    func: &IrFunc,
    body_region: crate::RegionId,
    body_args: &[crate::ArgId],
    known_env: &[Option<u64>],
    source: &PortSource,
    depth: usize,
) -> Option<u64> {
    if depth > 64 {
        return None;
    }

    match source {
        PortSource::RegionArg(arg_ref) => {
            // If this arg belongs to the body region, look it up in known_env.
            if arg_ref.region == body_region {
                let arg_idx = body_args.iter().position(|a| *a == arg_ref.arg)?;
                return known_env.get(arg_idx).copied().flatten();
            }
            // Otherwise, try to resolve through parent (gamma/theta arg).
            let region = arg_ref.region;
            let arg_idx = func.regions[region]
                .args
                .iter()
                .position(|a| *a == arg_ref.arg)?;
            let owner = func.find_region_owner(region)?;
            match &func.nodes[owner].kind {
                NodeKind::Gamma { .. } => {
                    // Gamma branch arg[K] → gamma input[K+1]
                    let input = func.nodes[owner].inputs.get(arg_idx + 1)?;
                    eval_with_env_inner(
                        func,
                        body_region,
                        body_args,
                        known_env,
                        &input.source,
                        depth + 1,
                    )
                }
                NodeKind::Theta { .. } => {
                    // Theta body arg[K] → theta input[K]
                    let input = func.nodes[owner].inputs.get(arg_idx)?;
                    eval_with_env_inner(
                        func,
                        body_region,
                        body_args,
                        known_env,
                        &input.source,
                        depth + 1,
                    )
                }
                _ => None,
            }
        }
        PortSource::Node(out_ref) => {
            let node = &func.nodes[out_ref.node];
            match &node.kind {
                NodeKind::Simple(crate::IrOp::Const { value }) => Some(*value),
                NodeKind::Simple(op) if !op.has_side_effects() => {
                    // Pure op: evaluate if all data inputs are known.
                    let data_inputs: Vec<Option<u64>> = node
                        .inputs
                        .iter()
                        .filter(|inp| inp.kind == crate::PortKind::Data)
                        .map(|inp| {
                            eval_with_env_inner(
                                func,
                                body_region,
                                body_args,
                                known_env,
                                &inp.source,
                                depth + 1,
                            )
                        })
                        .collect();
                    let all_known: Vec<u64> =
                        data_inputs.into_iter().collect::<Option<Vec<_>>>()?;
                    crate::const_fold::evaluate_op(op, &all_known)
                }
                NodeKind::Gamma { regions } => {
                    // If predicate is known, evaluate the selected branch.
                    let pred_source = &node.inputs[0].source;
                    let pred_val = eval_with_env_inner(
                        func,
                        body_region,
                        body_args,
                        known_env,
                        pred_source,
                        depth + 1,
                    );
                    if let Some(pv) = pred_val {
                        let branch_idx = (pv as usize).min(regions.len() - 1);
                        let branch_region = regions[branch_idx];
                        let result_id = *func.regions[branch_region]
                            .results
                            .get(out_ref.index as usize)?;
                        let result_source = &func.region_results[result_id].source;
                        return eval_with_env_inner(
                            func,
                            body_region,
                            body_args,
                            known_env,
                            result_source,
                            depth + 1,
                        );
                    }

                    // Predicate unknown: try all non-error branches. If they
                    // agree on a value, use it. Error branches are semantically
                    // unreachable on the success path — the program aborts if
                    // they're taken, so their output values don't matter.
                    let output_index = out_ref.index as usize;
                    let mut common: Option<u64> = None;
                    for &region_id in regions {
                        if region_has_error_exit(func, region_id) {
                            continue;
                        }
                        let result_id = *func.regions[region_id].results.get(output_index)?;
                        let result_source = &func.region_results[result_id].source;
                        let branch_val = eval_with_env_inner(
                            func,
                            body_region,
                            body_args,
                            known_env,
                            result_source,
                            depth + 1,
                        )?;
                        match common {
                            None => common = Some(branch_val),
                            Some(v) if v == branch_val => {}
                            _ => return None,
                        }
                    }
                    common
                }
                _ => None,
            }
        }
    }
}
