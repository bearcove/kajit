//! Dead loop-carried theta-port elimination.
//!
//! After slot_to_reg promotes stack slots to explicit theta ports, many ports
//! carry values that are re-initialized to the same constant at the start of
//! every iteration (e.g., varint acc/shift/byte temps set to 0). These ports
//! inflate the loop-carried live set without conveying useful information across
//! iterations.
//!
//! This pass detects theta data ports where:
//! 1. The theta input is `Const(C)`
//! 2. The body result that feeds back is always `Const(C)` (proven by tracing
//!    through gamma branches: every branch produces either `Const(C)` or passes
//!    through the body arg, which is `C` by induction)
//!
//! For each such port, the pass:
//! - Replaces uses of the body arg inside the theta body with a rematerialized
//!   `Const(C)` node
//! - Replaces uses of the theta output in the parent region with the original
//!   `Const(C)` source
//! - Removes the port from the theta (input, body arg, body result, output)

use crate::{
    ArgId, IrFunc, IrOp, Node, NodeId, NodeKind, OutputPort, OutputRef, PortKind, PortSource,
    RegionArgRef, RegionId, ResultId,
};

/// Eliminate dead loop-carried theta ports across all theta nodes in the function.
/// Returns the total number of ports removed.
pub fn eliminate_dead_theta_ports(func: &mut IrFunc) -> usize {
    let theta_ids: Vec<NodeId> = func
        .nodes
        .iter()
        .filter_map(|(nid, node)| {
            if matches!(node.kind, NodeKind::Theta { .. }) {
                Some(nid)
            } else {
                None
            }
        })
        .collect();

    let mut total_removed = 0;
    for theta_id in theta_ids {
        total_removed += eliminate_dead_ports_for_theta(func, theta_id);
    }
    total_removed
}

fn eliminate_dead_ports_for_theta(func: &mut IrFunc, theta_id: NodeId) -> usize {
    let body = match &func.nodes[theta_id].kind {
        NodeKind::Theta {
            body,
            max_iterations,
        } => {
            // Only target unbounded thetas (outer loops). Bounded thetas
            // (varint decoders) will be unrolled later, so their ports
            // become gamma outputs, not loop-carried values.
            if max_iterations.is_some() {
                return 0;
            }
            *body
        }
        _ => return 0,
    };

    let state_count = func.nodes[theta_id]
        .inputs
        .iter()
        .filter(|ip| matches!(ip.kind, PortKind::State(_)))
        .count();
    let loop_var_count = func.nodes[theta_id].inputs.len() - state_count;

    let debug = std::env::var("KAJIT_DEBUG_DEAD_THETA").is_ok();

    if debug {
        eprintln!(
            "[dead_theta_ports] theta {:?}: {} inputs ({} data, {} state)",
            theta_id,
            func.nodes[theta_id].inputs.len(),
            loop_var_count,
            state_count,
        );
    }

    // Build a use-set for theta outputs: which output indices have consumers?
    let parent_region = func.nodes[theta_id].region;
    let mut output_has_consumer = vec![false; loop_var_count];
    // Check node inputs in the parent region
    for &nid in &func.regions[parent_region].nodes {
        if nid == theta_id {
            continue;
        }
        for input in &func.nodes[nid].inputs {
            if let PortSource::Node(oref) = &input.source {
                if oref.node == theta_id && (oref.index as usize) < loop_var_count {
                    output_has_consumer[oref.index as usize] = true;
                }
            }
        }
    }
    // Check region results in the parent region
    for &rid in &func.regions[parent_region].results {
        if let PortSource::Node(oref) = &func.region_results[rid].source {
            if oref.node == theta_id && (oref.index as usize) < loop_var_count {
                output_has_consumer[oref.index as usize] = true;
            }
        }
    }

    // Phase 1: Find dead ports — input is Const(C) AND output has no consumers.
    // When both hold: the port carries a constant that enters the loop, circulates
    // through the backedge, and exits without anyone reading the exit value.
    // We can rematerialize Const(C) inside the body and remove the port.
    let mut dead_ports: Vec<(usize, u64)> = Vec::new();

    for p in 0..loop_var_count {
        let input_source = func.nodes[theta_id].inputs[p].source;
        let Some(const_val) = resolve_to_const(func, &input_source) else {
            continue;
        };

        if output_has_consumer[p] {
            if debug {
                eprintln!(
                    "  port {}: Const({}) but output HAS consumers",
                    p, const_val
                );
            }
            continue;
        }

        if debug {
            eprintln!(
                "  port {}: Const({}) with NO output consumers → DEAD",
                p, const_val
            );
        }
        // Verify the body arg is ONLY used as a gamma input within the body
        // (not in direct instructions). If used directly, the backedge value
        // matters and we can't safely replace with Const(0).
        let body_arg_id = func.regions[body].args[p];
        let body_arg_source = PortSource::RegionArg(RegionArgRef {
            region: body,
            arg: body_arg_id,
        });
        let mut only_gamma_uses = true;
        for &nid in &func.regions[body].nodes {
            let is_gamma = matches!(func.nodes[nid].kind, NodeKind::Gamma { .. });
            for input in &func.nodes[nid].inputs {
                if input.source == body_arg_source && !is_gamma {
                    only_gamma_uses = false;
                    if debug {
                        eprintln!("    body arg used by non-gamma node {:?}", nid);
                    }
                }
            }
        }
        // Also check body region results
        for &rid in &func.regions[body].results {
            if func.region_results[rid].source == body_arg_source {
                // Used in body result → that's the backedge, which we'll handle
            }
        }
        if !only_gamma_uses {
            if debug {
                eprintln!("  port {}: SKIPPED — body arg has non-gamma uses", p);
            }
            continue;
        }

        dead_ports.push((p, const_val));
        if dead_ports.len() >= 1 {
            break;
        } // TEMP: one at a time
    }

    if dead_ports.is_empty() {
        return 0;
    }

    let debug = std::env::var("KAJIT_DEBUG_DEAD_THETA").is_ok();
    if debug {
        eprintln!(
            "[dead_theta_ports] theta {:?}: removing {} dead ports (of {} data ports)",
            theta_id,
            dead_ports.len(),
            loop_var_count
        );
    }

    // Phase 2: For each dead port, replace uses and prepare for removal.
    // Collect replacement info before mutating.
    let parent_region = func.nodes[theta_id].region;

    for &(p, const_val) in &dead_ports {
        let body_arg_id = func.regions[body].args[p];

        // Create a Const(C) node inside the theta body to replace body arg uses.
        let debug_scope = func.regions[body].debug_scope;
        let remat_node = func.nodes.push(Node {
            region: body,
            debug_scope,
            debug_value: None,
            inputs: vec![],
            outputs: vec![OutputPort {
                kind: PortKind::Data,
                vreg: None,
                debug_scope,
            }],
            kind: NodeKind::Simple(IrOp::Const { value: const_val }),
        });
        // Insert at the beginning so it's topologically before all uses.
        func.regions[body].nodes.insert(0, remat_node);
        let remat_source = PortSource::Node(OutputRef {
            node: remat_node,
            index: 0,
        });

        // Replace all uses of body arg P inside the theta body with the remat Const.
        replace_region_arg_uses(func, body_arg_id, body, remat_source);

        // Replace all uses of theta output P in the parent region with the
        // original Const(C) source (which is the theta input source).
        let input_source = func.nodes[theta_id].inputs[p].source;
        let theta_output_ref = OutputRef {
            node: theta_id,
            index: p as u16,
        };
        replace_output_ref_uses(func, theta_output_ref, input_source);
    }

    // Phase 3: Remove ports in reverse order to preserve indices.
    let mut removed = 0;
    for &(p, _) in dead_ports.iter().rev() {
        let adjusted_p = p; // already correct since we go in reverse
        remove_theta_data_port(func, theta_id, body, adjusted_p);
        removed += 1;
    }

    removed
}

/// Resolve a PortSource to a constant value, if it directly references a Const node.
fn resolve_to_const(func: &IrFunc, source: &PortSource) -> Option<u64> {
    match source {
        PortSource::Node(oref) => match &func.nodes[oref.node].kind {
            NodeKind::Simple(IrOp::Const { value: val }) => Some(*val),
            _ => None,
        },
        _ => None,
    }
}

/// Check if a value is always equal to `expected`, given that the theta body arg
/// `theta_arg` is always `expected` (by induction).
///
/// Traces through gamma outputs: all branches must produce `expected`.
fn is_always_const(
    func: &IrFunc,
    source: &PortSource,
    expected: u64,
    theta_arg: ArgId,
    theta_body: RegionId,
    depth: usize,
) -> bool {
    let debug = depth < 4 && std::env::var("KAJIT_DEBUG_DEAD_THETA").is_ok();
    if depth > 20 {
        return false;
    }
    match source {
        PortSource::RegionArg(rref) => {
            // Direct match: this IS the theta body arg.
            if rref.arg == theta_arg {
                return true;
            }
            // Not a direct match — if this is a gamma branch arg, trace through
            // the gamma's corresponding input to see if it reaches the theta arg.
            if let Some((gamma_node, _branch_idx)) = find_gamma_for_branch(func, rref.region) {
                let arg_idx = func.regions[rref.region]
                    .args
                    .iter()
                    .position(|&a| a == rref.arg);
                if let Some(idx) = arg_idx {
                    // Gamma input at 1 + idx (skip predicate at input 0).
                    let gamma_input_idx = 1 + idx;
                    if gamma_input_idx < func.nodes[gamma_node].inputs.len() {
                        let gamma_input_source =
                            func.nodes[gamma_node].inputs[gamma_input_idx].source;
                        if debug {
                            let indent = "  ".repeat(depth + 2);
                            eprintln!(
                                "{}resolving RegionArg via gamma {:?} input[{}] = {:?}",
                                indent, gamma_node, gamma_input_idx, gamma_input_source
                            );
                        }
                        return is_always_const(
                            func,
                            &gamma_input_source,
                            expected,
                            theta_arg,
                            theta_body,
                            depth + 1,
                        );
                    }
                }
            }
            if debug {
                let indent = "  ".repeat(depth + 2);
                eprintln!(
                    "{}RegionArg({:?}) region={:?}: could not resolve",
                    indent, rref.arg, rref.region
                );
            }
            false
        }
        PortSource::Node(oref) => {
            let node = &func.nodes[oref.node];
            match &node.kind {
                NodeKind::Simple(IrOp::Const { value: val }) => *val == expected,
                NodeKind::Gamma { regions, .. } => {
                    for (bi, &region_id) in regions.iter().enumerate() {
                        let region = &func.regions[region_id];
                        if (oref.index as usize) >= region.results.len() {
                            if debug {
                                let indent = "  ".repeat(depth + 2);
                                eprintln!(
                                    "{}gamma {:?} branch {}: index {} >= results.len() {}",
                                    indent,
                                    oref.node,
                                    bi,
                                    oref.index,
                                    region.results.len()
                                );
                            }
                            return false;
                        }
                        let result_id = region.results[oref.index as usize];
                        let result = &func.region_results[result_id];
                        if !is_always_const(
                            func,
                            &result.source,
                            expected,
                            theta_arg,
                            theta_body,
                            depth + 1,
                        ) {
                            if debug {
                                let indent = "  ".repeat(depth + 2);
                                eprintln!(
                                    "{}gamma {:?} branch {}: result {:?} = {:?} is NOT const({})",
                                    indent, oref.node, bi, result_id, result.source, expected
                                );
                            }
                            return false;
                        }
                    }
                    true
                }
                _ => {
                    if debug {
                        let indent = "  ".repeat(depth + 2);
                        eprintln!(
                            "{}node {:?}: kind={:?} is NOT const/gamma",
                            indent,
                            oref.node,
                            std::mem::discriminant(&node.kind)
                        );
                    }
                    false
                }
            }
        }
    }
}

/// Find the gamma node that contains a given region as one of its branches.
/// Returns (gamma_node_id, branch_index).
fn find_gamma_for_branch(func: &IrFunc, region_id: RegionId) -> Option<(NodeId, usize)> {
    for (nid, node) in func.nodes.iter() {
        if let NodeKind::Gamma { regions, .. } = &node.kind {
            for (branch_idx, &rid) in regions.iter().enumerate() {
                if rid == region_id {
                    return Some((nid, branch_idx));
                }
            }
        }
    }
    None
}

/// Replace all uses of a region arg (by ArgId) with a new PortSource,
/// within the region and its sub-regions.
fn replace_region_arg_uses(
    func: &mut IrFunc,
    arg_id: ArgId,
    region_id: RegionId,
    replacement: PortSource,
) {
    let target = PortSource::RegionArg(RegionArgRef {
        region: region_id,
        arg: arg_id,
    });

    // Replace in node inputs within this region and all sub-regions.
    let all_nodes: Vec<NodeId> = func.nodes.iter().map(|(id, _)| id).collect();
    for nid in all_nodes {
        for input in &mut func.nodes[nid].inputs {
            if input.source == target {
                input.source = replacement;
            }
        }
    }

    // Replace in region results.
    let all_results: Vec<ResultId> = func.region_results.iter().map(|(id, _)| id).collect();
    for rid in all_results {
        if func.region_results[rid].source == target {
            func.region_results[rid].source = replacement;
        }
    }
}

/// Replace all uses of a specific OutputRef with a new PortSource.
fn replace_output_ref_uses(func: &mut IrFunc, from: OutputRef, to: PortSource) {
    let from_source = PortSource::Node(from);

    let all_nodes: Vec<NodeId> = func.nodes.iter().map(|(id, _)| id).collect();
    for nid in all_nodes {
        for input in &mut func.nodes[nid].inputs {
            if input.source == from_source {
                input.source = to;
            }
        }
    }

    let all_results: Vec<ResultId> = func.region_results.iter().map(|(id, _)| id).collect();
    for rid in all_results {
        if func.region_results[rid].source == from_source {
            func.region_results[rid].source = to;
        }
    }
}

/// Remove data port P from a theta node. Updates:
/// 1. Theta inputs (remove index P)
/// 2. Body args (remove index P)
/// 3. Body results (remove index 1+P, accounting for predicate)
/// 4. Theta outputs (remove index P)
/// 5. Fix all OutputRef indices > P for this theta in the parent region
fn remove_theta_data_port(func: &mut IrFunc, theta_id: NodeId, body: RegionId, p: usize) {
    // 1. Remove theta input
    func.nodes[theta_id].inputs.remove(p);

    // 2. Remove body arg
    func.regions[body].args.remove(p);

    // 3. Remove body result (at 1+p because index 0 is the predicate)
    func.regions[body].results.remove(1 + p);

    // 4. Remove theta output
    func.nodes[theta_id].outputs.remove(p);

    // 5. Fix all OutputRef indices > p for this theta.
    // Any reference to theta output index >= p+1 needs to be decremented by 1.
    // (References to index p were already replaced in phase 2.)
    let from = (p + 1) as u16;
    shift_output_refs_down(func, theta_id, from);
}

/// Decrement by 1 all OutputRef indices >= `from` that reference `node_id`.
fn shift_output_refs_down(func: &mut IrFunc, node_id: NodeId, from: u16) {
    let all_node_ids: Vec<NodeId> = func.nodes.iter().map(|(id, _)| id).collect();
    for nid in all_node_ids {
        for input in &mut func.nodes[nid].inputs {
            if let PortSource::Node(ref mut oref) = input.source {
                if oref.node == node_id && oref.index >= from {
                    oref.index -= 1;
                }
            }
        }
    }

    let all_result_ids: Vec<ResultId> = func.region_results.iter().map(|(id, _)| id).collect();
    for rid in all_result_ids {
        if let PortSource::Node(ref mut oref) = func.region_results[rid].source {
            if oref.node == node_id && oref.index >= from {
                oref.index -= 1;
            }
        }
    }
}
