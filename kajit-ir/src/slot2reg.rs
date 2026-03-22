//! Slot-to-register promotion pass.
//!
//! Converts `WriteToSlot`/`ReadFromSlot` operations into proper RVSDG data flow
//! (gamma passthrough values and theta loop-carried variables), enabling the
//! register allocator to keep these values in physical registers instead of
//! spilling them to the stack.
//!
//! This is analogous to LLVM's mem2reg pass (alloca → SSA promotion).

use std::collections::{BTreeMap, BTreeSet};

use crate::{
    InputPort, IrFunc, IrOp, NodeId, NodeKind, OutputPort, OutputRef, PortKind, PortSource,
    RegionArg, RegionArgRef, RegionId, RegionResult, SlotId,
};

/// Run the slot-to-register promotion pass.
pub fn slot_to_reg(func: &mut IrFunc) {
    // TODO: disabled — passes verification and simple scalars, but complex
    // cases produce wrong output due to over-aggressive slot threading.
    // The fix: thread only slots that are ACCESSED in sub-regions (not all known slots),
    // but ensure proper scoping by replacing parent-scope values with region args.
    let _ = func;
    return;

    #[allow(unreachable_code)]
    let all_slots = collect_all_slots(func);
    if all_slots.is_empty() {
        return;
    }

    let lambda_body = find_lambda_body(func);
    let initial_values: BTreeMap<SlotId, PortSource> = BTreeMap::new();
    promote_region(func, lambda_body, &all_slots, initial_values);
}

fn collect_all_slots(func: &IrFunc) -> BTreeSet<SlotId> {
    let mut slots = BTreeSet::new();
    for (_nid, node) in func.nodes.iter() {
        match &node.kind {
            NodeKind::Simple(IrOp::WriteToSlot { slot })
            | NodeKind::Simple(IrOp::ReadFromSlot { slot }) => {
                slots.insert(*slot);
            }
            _ => {}
        }
    }
    slots
}

fn find_lambda_body(func: &IrFunc) -> RegionId {
    for (_nid, node) in func.nodes.iter() {
        if let NodeKind::Lambda { body, .. } = &node.kind {
            return *body;
        }
    }
    panic!("no lambda found in IR");
}

/// Process a region: walk nodes in order, tracking slot values.
/// Returns the final slot values at the end of the region.
fn promote_region(
    func: &mut IrFunc,
    region: RegionId,
    all_slots: &BTreeSet<SlotId>,
    mut slot_values: BTreeMap<SlotId, PortSource>,
) -> BTreeMap<SlotId, PortSource> {
    let nodes: Vec<NodeId> = func.regions[region].nodes.clone();
    let mut nodes_to_remove: Vec<NodeId> = Vec::new();

    for &node_id in &nodes {
        match &func.nodes[node_id].kind {
            NodeKind::Simple(IrOp::ReadFromSlot { slot }) => {
                let slot = *slot;
                if let Some(&current_val) = slot_values.get(&slot) {
                    let data_output = OutputRef {
                        node: node_id,
                        index: 0,
                    };
                    let state_output = OutputRef {
                        node: node_id,
                        index: 1,
                    };
                    let incoming_state = func.nodes[node_id].inputs[0].source;

                    replace_uses_in_region(func, region, data_output, current_val);
                    replace_uses_in_region(func, region, state_output, incoming_state);
                    nodes_to_remove.push(node_id);
                }
            }
            NodeKind::Simple(IrOp::WriteToSlot { slot }) => {
                let slot = *slot;
                let written_value = func.nodes[node_id].inputs[0].source;
                let incoming_state = func.nodes[node_id].inputs[1].source;
                let state_output = OutputRef {
                    node: node_id,
                    index: 0,
                };

                slot_values.insert(slot, written_value);
                replace_uses_in_region(func, region, state_output, incoming_state);
                nodes_to_remove.push(node_id);
            }
            NodeKind::Gamma { regions } => {
                let regions_clone = regions.clone();
                // Thread ALL known slots through every gamma to maintain correct scoping.
                let slots_to_thread: Vec<SlotId> = slot_values.keys().copied().collect();

                if !slots_to_thread.is_empty() {
                    promote_gamma(
                        func,
                        node_id,
                        &regions_clone,
                        &slots_to_thread,
                        all_slots,
                        &mut slot_values,
                    );
                } else {
                    for &branch_region in &regions_clone {
                        promote_region(func, branch_region, all_slots, slot_values.clone());
                    }
                }
            }
            NodeKind::Theta { body } => {
                let body = *body;
                // Thread ALL known slots through the theta.
                let slots_to_thread: Vec<SlotId> = slot_values.keys().copied().collect();

                if !slots_to_thread.is_empty() {
                    promote_theta(
                        func,
                        node_id,
                        body,
                        &slots_to_thread,
                        all_slots,
                        &mut slot_values,
                    );
                } else {
                    promote_region(func, body, all_slots, slot_values.clone());
                }
            }
            _ => {}
        }
    }

    let remove_set: BTreeSet<NodeId> = nodes_to_remove.into_iter().collect();
    func.regions[region]
        .nodes
        .retain(|nid| !remove_set.contains(nid));

    slot_values
}

/// For any slot in `needed` that doesn't yet have a value in `slot_values`,
/// create a const(0) node in `region` before `before_node` and set it as the initial value.
fn ensure_slot_defaults(
    func: &mut IrFunc,
    region: RegionId,
    needed: &[SlotId],
    slot_values: &mut BTreeMap<SlotId, PortSource>,
    before_node: NodeId,
) {
    let debug_scope = func.regions[region].debug_scope;
    let insert_pos = func.regions[region]
        .nodes
        .iter()
        .position(|&n| n == before_node)
        .expect("before_node must be in region");

    let mut offset = 0;
    for &slot in needed {
        if slot_values.contains_key(&slot) {
            continue;
        }
        let vreg = func.fresh_vreg();
        let node = func.nodes.push(crate::Node {
            region,
            debug_scope,
            debug_value: None,
            inputs: vec![],
            outputs: vec![OutputPort {
                kind: PortKind::Data,
                vreg: Some(vreg),
                debug_scope,
            }],
            kind: NodeKind::Simple(IrOp::Const { value: 0 }),
        });
        func.regions[region].nodes.insert(insert_pos + offset, node);
        offset += 1;
        slot_values.insert(slot, PortSource::Node(OutputRef { node, index: 0 }));
    }
}

fn slots_accessed_recursively(func: &IrFunc, regions: &[RegionId]) -> BTreeSet<SlotId> {
    let mut slots = BTreeSet::new();
    let mut stack: Vec<RegionId> = regions.to_vec();
    while let Some(region) = stack.pop() {
        for &node_id in &func.regions[region].nodes {
            match &func.nodes[node_id].kind {
                NodeKind::Simple(IrOp::WriteToSlot { slot })
                | NodeKind::Simple(IrOp::ReadFromSlot { slot }) => {
                    slots.insert(*slot);
                }
                NodeKind::Gamma {
                    regions: sub_regions,
                } => stack.extend(sub_regions),
                NodeKind::Theta { body } => stack.push(*body),
                NodeKind::Lambda { body, .. } => stack.push(*body),
                _ => {}
            }
        }
    }
    slots
}

/// Insert a data output on `node_id` before the state outputs, and update all
/// existing references to the shifted state outputs.
fn insert_data_output(func: &mut IrFunc, node_id: NodeId, debug_scope: crate::DebugScopeId) -> u16 {
    let state_count = func.state_domains.len();
    let insert_idx = func.nodes[node_id].outputs.len() - state_count;
    let vreg = func.fresh_vreg();
    func.nodes[node_id].outputs.insert(
        insert_idx,
        OutputPort {
            kind: PortKind::Data,
            vreg: Some(vreg),
            debug_scope,
        },
    );

    // All existing references to outputs at index >= insert_idx need to be shifted by +1.
    shift_output_refs(func, node_id, insert_idx as u16);

    insert_idx as u16
}

/// Shift all references to node outputs at index >= `from` by +1.
fn shift_output_refs(func: &mut IrFunc, node_id: NodeId, from: u16) {
    let all_node_ids: Vec<NodeId> = func.nodes.iter().map(|(id, _)| id).collect();
    for &nid in &all_node_ids {
        for input in &mut func.nodes[nid].inputs {
            if let PortSource::Node(ref mut oref) = input.source {
                if oref.node == node_id && oref.index >= from {
                    oref.index += 1;
                }
            }
        }
    }

    let all_result_ids: Vec<crate::ResultId> =
        func.region_results.iter().map(|(id, _)| id).collect();
    for &rid in &all_result_ids {
        if let PortSource::Node(ref mut oref) = func.region_results[rid].source {
            if oref.node == node_id && oref.index >= from {
                oref.index += 1;
            }
        }
    }
}

/// Insert a data input on `node_id` before the state inputs.
fn insert_data_input(func: &mut IrFunc, node_id: NodeId, source: PortSource) {
    let state_count = func.state_domains.len();
    let insert_idx = func.nodes[node_id].inputs.len() - state_count;
    func.nodes[node_id].inputs.insert(
        insert_idx,
        InputPort {
            kind: PortKind::Data,
            source,
        },
    );
}

/// Insert a data region arg before the state args.
fn insert_data_region_arg(func: &mut IrFunc, region: RegionId) -> PortSource {
    let state_count = func.state_domains.len();
    let insert_idx = func.regions[region].args.len() - state_count;
    let arg_id = func.region_args.push(RegionArg {
        kind: PortKind::Data,
        vreg: None,
        debug_value: None,
    });
    func.regions[region].args.insert(insert_idx, arg_id);
    PortSource::RegionArg(RegionArgRef {
        region,
        arg: arg_id,
    })
}

/// Insert a data region result before the state results.
fn insert_data_region_result(func: &mut IrFunc, region: RegionId, source: PortSource) {
    let state_count = func.state_domains.len();
    let insert_idx = func.regions[region].results.len() - state_count;
    let result_id = func.region_results.push(RegionResult {
        kind: PortKind::Data,
        source,
    });
    func.regions[region].results.insert(insert_idx, result_id);
}

fn promote_gamma(
    func: &mut IrFunc,
    node_id: NodeId,
    branch_regions: &[RegionId],
    slots_to_thread: &[SlotId],
    all_slots: &BTreeSet<SlotId>,
    slot_values: &mut BTreeMap<SlotId, PortSource>,
) {
    let debug_scope = func.nodes[node_id].debug_scope;

    // Phase 1: Add inputs, region args for all slots.
    for &slot in slots_to_thread {
        let incoming_value = slot_values[&slot];
        insert_data_input(func, node_id, incoming_value);
        for &branch_region in branch_regions {
            insert_data_region_arg(func, branch_region);
        }
    }

    // Phase 2: Recurse into each branch with slot values from region args.
    // Track the region arg sources for each branch and slot.
    let mut branch_arg_sources: Vec<BTreeMap<SlotId, PortSource>> = Vec::new();
    let mut branch_final_values: Vec<BTreeMap<SlotId, PortSource>> = Vec::new();
    let state_count = func.state_domains.len();

    for &branch_region in branch_regions {
        // Only carry slot values that we set up as region args for this branch.
        // Parent-scope values are NOT valid inside child regions.
        let mut branch_slots = BTreeMap::new();
        let mut arg_sources = BTreeMap::new();
        let data_arg_count = func.regions[branch_region].args.len() - state_count;
        for (i, &slot) in slots_to_thread.iter().enumerate() {
            let arg_idx = data_arg_count - slots_to_thread.len() + i;
            let arg_id = func.regions[branch_region].args[arg_idx];
            let arg_source = PortSource::RegionArg(RegionArgRef {
                region: branch_region,
                arg: arg_id,
            });
            branch_slots.insert(slot, arg_source);
            arg_sources.insert(slot, arg_source);
        }
        let final_vals = promote_region(func, branch_region, all_slots, branch_slots);
        branch_arg_sources.push(arg_sources);
        branch_final_values.push(final_vals);
    }

    // Phase 3: Add results and outputs for each slot.
    for &slot in slots_to_thread {
        for (branch_idx, &branch_region) in branch_regions.iter().enumerate() {
            // Use the branch's final value, or fall back to the branch's region arg
            // (NOT the parent scope value — that would be invalid cross-scope reference).
            let final_value = branch_final_values[branch_idx]
                .get(&slot)
                .copied()
                .unwrap_or_else(|| branch_arg_sources[branch_idx][&slot]);
            insert_data_region_result(func, branch_region, final_value);
        }

        let output_idx = insert_data_output(func, node_id, debug_scope);
        slot_values.insert(
            slot,
            PortSource::Node(OutputRef {
                node: node_id,
                index: output_idx,
            }),
        );
    }
}

fn promote_theta(
    func: &mut IrFunc,
    node_id: NodeId,
    body: RegionId,
    slots_to_thread: &[SlotId],
    all_slots: &BTreeSet<SlotId>,
    slot_values: &mut BTreeMap<SlotId, PortSource>,
) {
    let debug_scope = func.nodes[node_id].debug_scope;
    let state_count = func.state_domains.len();

    // Phase 1: Add inputs and region args for all slots.
    // Only carry slot values we explicitly set up as region args.
    let mut body_slot_values = BTreeMap::new();
    let mut body_arg_sources: BTreeMap<SlotId, PortSource> = BTreeMap::new();
    for &slot in slots_to_thread {
        let incoming_value = slot_values[&slot];
        insert_data_input(func, node_id, incoming_value);
        let arg_source = insert_data_region_arg(func, body);
        body_slot_values.insert(slot, arg_source);
        body_arg_sources.insert(slot, arg_source);
    }

    // Phase 2: Recurse into body.
    let final_body_values = promote_region(func, body, all_slots, body_slot_values);

    // Phase 3: Add results and outputs for each slot.
    for &slot in slots_to_thread {
        // Use the body's final value, or fall back to the body's region arg
        // (NOT the parent scope value).
        let final_value = final_body_values
            .get(&slot)
            .copied()
            .unwrap_or_else(|| body_arg_sources[&slot]);
        insert_data_region_result(func, body, final_value);

        let output_idx = insert_data_output(func, node_id, debug_scope);
        slot_values.insert(
            slot,
            PortSource::Node(OutputRef {
                node: node_id,
                index: output_idx,
            }),
        );
    }
}

/// Replace all uses of a node output within a specific region (not recursive).
fn replace_uses_in_region(func: &mut IrFunc, region: RegionId, from: OutputRef, to: PortSource) {
    let from_source = PortSource::Node(from);
    if from_source == to {
        return;
    }

    let nodes: Vec<NodeId> = func.regions[region].nodes.clone();
    for &nid in &nodes {
        for input in &mut func.nodes[nid].inputs {
            if input.source == from_source {
                input.source = to;
            }
        }
    }

    for &result_id in &func.regions[region].results {
        if func.region_results[result_id].source == from_source {
            func.region_results[result_id].source = to;
        }
    }
}
