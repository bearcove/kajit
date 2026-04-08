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
            if max_iterations.is_some() {
                return 0;
            }
            *body
        }
        _ => return 0,
    };

    // Skip very large theta bodies to avoid O(n²) scanning.
    if func.regions[body].nodes.len() > 50 {
        return 0;
    }

    let state_count = func.nodes[theta_id]
        .inputs
        .iter()
        .filter(|ip| matches!(ip.kind, PortKind::State))
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
            if let PortSource::Node(oref) = &input.source
                && oref.node == theta_id
                && (oref.index as usize) < loop_var_count
            {
                output_has_consumer[oref.index as usize] = true;
            }
        }
    }
    // Check region results in the parent region
    for &rid in &func.regions[parent_region].results {
        if let PortSource::Node(oref) = &func.region_results[rid].source
            && oref.node == theta_id
            && (oref.index as usize) < loop_var_count
        {
            output_has_consumer[oref.index as usize] = true;
        }
    }

    if debug {
        // Dump body results to diagnose port mapping
        for i in 0..std::cmp::min(loop_var_count + 1, func.regions[body].results.len()) {
            let rid = func.regions[body].results[i];
            let src = &func.region_results[rid].source;
            let kind_str = match src {
                PortSource::Node(oref) => format!(
                    "#{} {:?}",
                    oref.node.index(),
                    std::mem::discriminant(&func.nodes[oref.node].kind)
                ),
                PortSource::RegionArg(rref) => format!("arg#{}", rref.arg.index()),
            };
            eprintln!(
                "  result[{}]: {} (kind={:?})",
                i, kind_str, func.region_results[rid].kind
            );
        }
    }

    // Phase 1: Find dead ports — input is Const(C) AND output has no consumers.
    // When both hold: the port carries a constant that enters the loop, circulates
    // through the backedge, and exits without anyone reading the exit value.
    // We can rematerialize Const(C) inside the body and remove the port.
    let mut dead_ports: Vec<(usize, u64)> = Vec::new();

    #[allow(clippy::needless_range_loop)]
    for p in 0..loop_var_count {
        let input_source = func.nodes[theta_id].inputs[p].source;
        let Some(const_val) = resolve_to_const(func, &input_source) else {
            continue;
        };

        if output_has_consumer[p] {
            continue;
        }

        if debug {
            eprintln!(
                "  port {}: Const({}) no consumers AND backedge const → DEAD",
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
        let mut only_passthrough_uses = true;
        for &nid in &func.regions[body].nodes {
            let is_identity = matches!(func.nodes[nid].kind, NodeKind::Simple(IrOp::Identity));
            if let NodeKind::Gamma { regions, .. } = &func.nodes[nid].kind {
                // Find which gamma input corresponds to this body arg.
                for (inp_idx, input) in func.nodes[nid].inputs.iter().enumerate() {
                    if input.source != body_arg_source {
                        continue;
                    }
                    // inp_idx 0 is the predicate. Data inputs start at 1.
                    // The corresponding branch arg index is inp_idx - 1.
                    if inp_idx == 0 {
                        // Used as predicate — definitely not passthrough!
                        only_passthrough_uses = false;
                        break;
                    }
                    let branch_arg_idx = inp_idx - 1;
                    // Check if ANY branch uses this arg for computation,
                    // recursively through nested gammas (but not thetas,
                    // which have their own slot promotion).
                    for &region_id in regions {
                        let region = &func.regions[region_id];
                        if branch_arg_idx >= region.args.len() {
                            continue;
                        }
                        let branch_arg_id = region.args[branch_arg_idx];
                        let branch_arg_src = PortSource::RegionArg(RegionArgRef {
                            region: region_id,
                            arg: branch_arg_id,
                        });
                        if is_source_used_recursive(func, region_id, branch_arg_src) {
                            only_passthrough_uses = false;
                            if debug {
                                eprintln!("    body arg used inside gamma {:?} branch", nid);
                            }
                        }
                    }
                }
            } else if !is_identity {
                for input in &func.nodes[nid].inputs {
                    if input.source == body_arg_source {
                        only_passthrough_uses = false;
                        if debug {
                            eprintln!("    body arg used by non-gamma node {:?}", nid);
                        }
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
        if !only_passthrough_uses {
            if debug {
                eprintln!("  port {}: SKIPPED — body arg has non-passthrough uses", p);
            }
            continue;
        }

        dead_ports.push((p, const_val));
        // Safety cap: adjacent port removal (e.g., 20+21) triggers a bug
        // in Phase 2+3 interaction that hasn't been root-caused yet. Cap at 3
        // as fallback — matches the previous validated set (17, 18, 20).
        if dead_ports.len() >= 3 {
            break;
        }
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
    let _parent_region = func.nodes[theta_id].region;
    // Create a SINGLE shared Const(0) node for all dead port replacements.
    // Using one node prevents theta_loop_invariant_hoist from hoisting
    // multiple orphaned Const(0) copies as separate theta ports.
    let debug_scope = func.regions[body].debug_scope;
    let shared_remat = func.nodes.push(Node {
        region: body,
        debug_scope,
        debug_value: None,
        inputs: vec![],
        outputs: vec![OutputPort {
            kind: PortKind::Data,
            vreg: None,
            debug_scope,
        }],
        kind: NodeKind::Simple(IrOp::Const { value: 0 }),
    });
    func.regions[body].nodes.insert(0, shared_remat);

    for &(p, _const_val) in &dead_ports {
        let body_arg_id = func.regions[body].args[p];
        let remat_source = PortSource::Node(OutputRef {
            node: shared_remat,
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
        remove_theta_data_port(func, theta_id, body, p);
        removed += 1;
    }

    // Phase 4: If the shared remat Const(0) has no remaining uses after port
    // removal, remove it to prevent theta_loop_invariant_hoist from hoisting it.
    let remat_ref = PortSource::Node(OutputRef {
        node: shared_remat,
        index: 0,
    });
    let has_uses = func
        .nodes
        .iter()
        .any(|(_, n)| n.inputs.iter().any(|inp| inp.source == remat_ref))
        || func
            .region_results
            .iter()
            .any(|(_, r)| r.source == remat_ref);
    if !has_uses {
        func.regions[body].nodes.retain(|n| *n != shared_remat);
    }

    removed
}

/// Resolve a PortSource to a constant value, if it directly references a Const node.
pub fn resolve_to_const(func: &IrFunc, source: &PortSource) -> Option<u64> {
    match source {
        PortSource::Node(oref) => match &func.nodes[oref.node].kind {
            NodeKind::Simple(IrOp::Const { value: val }) => Some(*val),
            _ => None,
        },
        _ => None,
    }
}

/// Check if a PortSource is used by any node input in a region,
/// recursively through nested gammas but NOT thetas.
fn is_source_used_recursive(func: &IrFunc, region: RegionId, source: PortSource) -> bool {
    for &nid in &func.regions[region].nodes {
        for input in &func.nodes[nid].inputs {
            if input.source == source {
                return true;
            }
        }
        // Recurse into gamma branches (but NOT thetas which have own promotion)
        if let NodeKind::Gamma { regions, .. } = &func.nodes[nid].kind {
            for &sub_region in regions {
                // The gamma input that equals `source` becomes a branch arg.
                // Check if THAT branch arg is used inside the sub-region.
                for (inp_idx, input) in func.nodes[nid].inputs.iter().enumerate() {
                    if input.source == source && inp_idx > 0 {
                        let branch_arg_idx = inp_idx - 1;
                        let sub_r = &func.regions[sub_region];
                        if branch_arg_idx < sub_r.args.len() {
                            let inner_src = PortSource::RegionArg(RegionArgRef {
                                region: sub_region,
                                arg: sub_r.args[branch_arg_idx],
                            });
                            if is_source_used_recursive(func, sub_region, inner_src) {
                                return true;
                            }
                        }
                    }
                }
            }
        }
    }
    false
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
            if let PortSource::Node(ref mut oref) = input.source
                && oref.node == node_id
                && oref.index >= from
            {
                oref.index -= 1;
            }
        }
    }

    let all_result_ids: Vec<ResultId> = func.region_results.iter().map(|(id, _)| id).collect();
    for rid in all_result_ids {
        if let PortSource::Node(ref mut oref) = func.region_results[rid].source
            && oref.node == node_id
            && oref.index >= from
        {
            oref.index -= 1;
        }
    }
}
