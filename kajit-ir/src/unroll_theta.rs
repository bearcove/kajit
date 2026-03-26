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

/// Maximum iterations we're willing to unroll.
const MAX_UNROLL: u32 = 10;

/// Run the bounded-theta unrolling pass.
///
/// Returns true if any theta was unrolled.
pub fn unroll_bounded_thetas(func: &mut IrFunc) -> bool {
    let mut changed = false;

    // Collect thetas to unroll (can't mutate while iterating)
    let mut targets: Vec<(NodeId, u32)> = Vec::new();
    for (_region_id, region) in func.regions.iter() {
        for &node_id in &region.nodes {
            if let NodeKind::Theta {
                max_iterations: Some(n),
                ..
            } = &func.nodes[node_id].kind
            {
                if *n <= MAX_UNROLL {
                    targets.push((node_id, *n));
                }
            }
        }
    }

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

    // We'll build nested gammas. The outermost result sources will replace
    // the theta node's outputs.
    let final_sources =
        build_unrolled_cascade(func, body, parent_region, &current_sources, max_iter);

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
                if let PortSource::Node(out_ref) = input.source {
                    if out_ref.node == theta_node_id {
                        let new_source = final_sources[out_ref.index as usize];
                        uses_to_rewrite.push((node_id, input_idx, new_source));
                    }
                }
            }
        }
        for &result_id in &region.results {
            let result = &func.region_results[result_id];
            if let PortSource::Node(out_ref) = result.source {
                if out_ref.node == theta_node_id {
                    let new_source = final_sources[out_ref.index as usize];
                    result_uses_to_rewrite.push((result_id, new_source));
                }
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
                if let PortSource::Node(out) = input.source {
                    if out.node != nid && !node_set.contains(&out.node) {
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
            if let PortSource::Node(out) = input.source {
                if node_set.contains(&out.node) && out.node != nid {
                    node_deps.push(out.node);
                }
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
fn build_unrolled_cascade(
    func: &mut IrFunc,
    body_region: crate::RegionId,
    target_region: crate::RegionId,
    input_sources: &[PortSource],
    remaining_iterations: u32,
) -> Vec<PortSource> {
    if remaining_iterations == 0 {
        // Max iterations reached — just return the current values
        // (skip predicate, return loop vars + state domains)
        return input_sources.to_vec();
    }

    // Clone the body into target_region, substituting body args with input_sources
    let cloned_results = clone_body_into_region(func, body_region, target_region, input_sources);

    // Run const fold on the cloned nodes to evaluate constant expressions
    // (e.g., Mul(Const(0), Const(7)) → Const(0) for iteration 0).
    crate::const_fold::fold_nodes_in_region(func, target_region);

    // cloned_results[0] = predicate, cloned_results[1..] = updated values
    let predicate = cloned_results[0];
    let updated_values: Vec<PortSource> = cloned_results[1..].to_vec();

    if remaining_iterations == 1 {
        // Last iteration — no need for a gamma, just return the values
        return updated_values;
    }

    // Build gamma: check predicate, either exit or continue
    let debug_scope = func.regions[target_region].debug_scope;
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
    for (_i, &arg_id) in branch0_args.iter().enumerate() {
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
    // For values that are evaluable constants (e.g., iteration counter),
    // inject Const nodes directly into branch1_region instead of using
    // gamma region args. This enables const_fold to propagate iteration
    // indices through the cascade.
    let branch1_input_sources: Vec<PortSource> = updated_values
        .iter()
        .zip(branch1_args.iter())
        .map(|(src, &arg_id)| {
            if let Some(val) = crate::const_fold::resolve_to_constant_skip_errors(func, src) {
                crate::const_fold::create_const_in_region(func, branch1_region, debug_scope, val)
            } else {
                PortSource::RegionArg(crate::RegionArgRef {
                    region: branch1_region,
                    arg: arg_id,
                })
            }
        })
        .collect();
    let recursive_results = build_unrolled_cascade(
        func,
        body_region,
        branch1_region,
        &branch1_input_sources,
        remaining_iterations - 1,
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
