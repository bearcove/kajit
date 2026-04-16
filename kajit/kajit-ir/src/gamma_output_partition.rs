//! Gamma output partitioning: eliminate gamma outputs that produce the same
//! value on all branches ("same on both branches").
//!
//! When a gamma output is a passthrough of the same gamma input on every branch,
//! consumers within the theta body can use the gamma input directly.
//!
//! Scope: bounded theta bodies only (looped varint kernels). The unrolled
//! scalar path and unbounded element loops use linearizer patterns that depend
//! on exact gamma output chain structure.
//!
//! Limitation: does not compact gamma output tuples. Compaction changes vreg
//! assignments that flow into regalloc hints, triggering a regalloc bug that
//! produces wrong code for vec types. The output rewriting alone creates a
//! cleaner IR for potential future optimization.

use kajit_reprs::ir::{IrFunc, NodeId, NodeKind, OutputRef, PortSource, RegionId};

/// Run gamma output partitioning (single round).
pub fn gamma_output_partition(func: &mut IrFunc) -> bool {
    // Collect body regions of live bounded thetas.
    let mut theta_body_regions: Vec<RegionId> = Vec::new();
    for (nid, node) in func.nodes.iter() {
        if let NodeKind::Theta {
            body,
            max_iterations,
        } = &node.kind
        {
            if max_iterations.is_none() {
                continue;
            }
            if func.regions[node.region].nodes.contains(&nid) {
                theta_body_regions.push(*body);
            }
        }
    }

    let mut gamma_nodes: Vec<(NodeId, RegionId)> = Vec::new();
    for &rid in &theta_body_regions {
        for &nid in &func.regions[rid].nodes {
            if matches!(func.nodes[nid].kind, NodeKind::Gamma { .. }) {
                gamma_nodes.push((nid, rid));
            }
        }
    }

    let mut changed = false;
    for (gamma_id, parent_region) in gamma_nodes {
        if rewrite_same_on_both_branches(func, gamma_id, parent_region) {
            changed = true;
        }
    }
    changed
}

/// For a single gamma, find data outputs that are SameOnBothBranches and
/// redirect consumers in the parent region to use the gamma input source.
fn rewrite_same_on_both_branches(
    func: &mut IrFunc,
    gamma_id: NodeId,
    parent_region: RegionId,
) -> bool {
    let regions = match &func.nodes[gamma_id].kind {
        NodeKind::Gamma { regions } => regions.clone(),
        _ => return false,
    };
    if regions.is_empty() {
        return false;
    }

    let num_outputs = func.regions[regions[0]].results.len();
    let mut rewrites: Vec<(OutputRef, PortSource)> = Vec::new();

    for output_idx in 0..num_outputs {
        let first_result_id = func.regions[regions[0]].results[output_idx];
        if matches!(
            func.region_results[first_result_id].kind,
            crate::PortKind::State
        ) {
            continue;
        }

        let mut common_arg_pos: Option<usize> = None;
        let mut is_same = true;

        for &branch_region in &regions {
            let results = &func.regions[branch_region].results;
            if output_idx >= results.len() {
                is_same = false;
                break;
            }
            let result_id = results[output_idx];
            match func.region_results[result_id].source {
                PortSource::RegionArg(arg_ref) => {
                    let arg_pos = func.regions[branch_region]
                        .args
                        .iter()
                        .position(|&a| a == arg_ref.arg);
                    match (arg_pos, common_arg_pos) {
                        (Some(pos), None) => common_arg_pos = Some(pos),
                        (Some(pos), Some(prev)) if pos == prev => {}
                        _ => {
                            is_same = false;
                            break;
                        }
                    }
                }
                PortSource::Node(_) => {
                    is_same = false;
                    break;
                }
            }
        }

        if is_same && let Some(arg_pos) = common_arg_pos {
            let gamma_input_idx = arg_pos + 1;
            let replacement = func.nodes[gamma_id].inputs[gamma_input_idx].source;
            let output_ref = OutputRef {
                node: gamma_id,
                index: output_idx as u16,
            };
            rewrites.push((output_ref, replacement));
        }
    }

    if rewrites.is_empty() {
        return false;
    }

    // Rewrite node inputs in the parent region only.
    let node_ids: Vec<NodeId> = func.regions[parent_region].nodes.clone();
    for (from, to) in &rewrites {
        let from_source = PortSource::Node(*from);
        if from_source == *to {
            continue;
        }
        for &nid in &node_ids {
            for input in &mut func.nodes[nid].inputs {
                if input.source == from_source {
                    input.source = *to;
                }
            }
        }
    }

    true
}
