//! Slot-to-register promotion pass.
//!
//! Converts `WriteToSlot`/`ReadFromSlot` operations into proper RVSDG data flow
//! (gamma passthrough values and theta loop-carried variables), enabling the
//! register allocator to keep these values in physical registers instead of
//! spilling them to the stack.
//!
//! This is analogous to LLVM's mem2reg pass (alloca → SSA promotion).

use std::collections::{BTreeMap, BTreeSet, HashMap};

use crate::{
    InputPort, IrFunc, IrOp, NodeId, NodeKind, OutputPort, OutputRef, PortKind, PortSource,
    RegionArg, RegionArgRef, RegionId, RegionResult, SlotId,
};

/// Pre-computed slot access information for each region.
/// Maps region → set of slots accessed anywhere in that region's sub-tree.
type SlotAccessMap = HashMap<RegionId, BTreeSet<SlotId>>;

/// Run the slot-to-register promotion pass.
pub fn slot_to_reg(func: &mut IrFunc) {
    let all_slots = collect_all_slots(func);
    if all_slots.is_empty() {
        return;
    }

    // Pre-compute slot accesses for every region BEFORE we start modifying the IR.
    let slot_access = precompute_slot_access(func);

    let lambda_body = find_lambda_body(func);
    let initial_values: BTreeMap<SlotId, PortSource> = BTreeMap::new();
    promote_region(func, lambda_body, &all_slots, &slot_access, initial_values);
}

fn collect_all_slots(func: &IrFunc) -> BTreeSet<SlotId> {
    let mut slots = BTreeSet::new();
    let mut address_taken = BTreeSet::new();
    for (_nid, node) in func.nodes.iter() {
        match &node.kind {
            NodeKind::Simple(IrOp::WriteToSlot { slot })
            | NodeKind::Simple(IrOp::ReadFromSlot { slot }) => {
                slots.insert(*slot);
            }
            NodeKind::Simple(IrOp::SlotAddr { slot }) => {
                // This slot's address escapes — it must stay in memory.
                address_taken.insert(*slot);
            }
            _ => {}
        }
    }
    // Exclude address-taken slots from promotion.
    for slot in &address_taken {
        slots.remove(slot);
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

/// Pre-compute, for each region, the set of slots accessed anywhere in its sub-tree.
/// This is done once before any modifications so it's stable.
fn precompute_slot_access(func: &IrFunc) -> SlotAccessMap {
    let mut map: SlotAccessMap = HashMap::new();
    // Process all regions. For each, collect direct slot ops + recurse into sub-regions.
    let region_ids: Vec<RegionId> = func.regions.iter().map(|(id, _)| id).collect();
    for &region_id in &region_ids {
        if !map.contains_key(&region_id) {
            compute_region_slots(func, region_id, &mut map);
        }
    }
    map
}

fn compute_region_slots(
    func: &IrFunc,
    region_id: RegionId,
    map: &mut SlotAccessMap,
) -> BTreeSet<SlotId> {
    let mut slots = BTreeSet::new();
    for &node_id in &func.regions[region_id].nodes {
        match &func.nodes[node_id].kind {
            NodeKind::Simple(IrOp::WriteToSlot { slot })
            | NodeKind::Simple(IrOp::ReadFromSlot { slot }) => {
                slots.insert(*slot);
            }
            NodeKind::Gamma { regions } => {
                for &sub_region in regions {
                    if !map.contains_key(&sub_region) {
                        compute_region_slots(func, sub_region, map);
                    }
                    slots.extend(map[&sub_region].iter());
                }
            }
            NodeKind::Theta { body, .. } => {
                let body = *body;
                if !map.contains_key(&body) {
                    compute_region_slots(func, body, map);
                }
                slots.extend(map[&body].iter());
            }
            NodeKind::Lambda { body, .. } => {
                let body = *body;
                if !map.contains_key(&body) {
                    compute_region_slots(func, body, map);
                }
                slots.extend(map[&body].iter());
            }
            _ => {}
        }
    }
    map.insert(region_id, slots.clone());
    slots
}

/// Process a region: walk nodes in order, tracking slot values.
/// Returns the final slot values at the end of the region.
fn promote_region(
    func: &mut IrFunc,
    region: RegionId,
    all_slots: &BTreeSet<SlotId>,
    slot_access: &SlotAccessMap,
    mut slot_values: BTreeMap<SlotId, PortSource>,
) -> BTreeMap<SlotId, PortSource> {
    let nodes: Vec<NodeId> = func.regions[region].nodes.clone();
    let mut nodes_to_remove: Vec<NodeId> = Vec::new();

    for &node_id in &nodes {
        match &func.nodes[node_id].kind {
            NodeKind::Simple(IrOp::ReadFromSlot { slot }) => {
                let slot = *slot;
                if !all_slots.contains(&slot) {
                    // Slot is not promotable (e.g. address-taken) — skip.
                    continue;
                }
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
                if !all_slots.contains(&slot) {
                    // Slot is not promotable (e.g. address-taken) — skip.
                    continue;
                }
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

                // Use PRE-COMPUTED slot accesses — not live node lists.
                let mut needed_slots = BTreeSet::new();
                for &branch_region in &regions_clone {
                    if let Some(accessed) = slot_access.get(&branch_region) {
                        needed_slots.extend(accessed.iter());
                    }
                }
                let slots_to_thread: Vec<SlotId> =
                    needed_slots.intersection(all_slots).copied().collect();

                // Ensure all needed slots have initial values.
                ensure_slot_defaults(func, region, &slots_to_thread, &mut slot_values, node_id);

                if !slots_to_thread.is_empty() {
                    promote_gamma(
                        func,
                        node_id,
                        &regions_clone,
                        &slots_to_thread,
                        all_slots,
                        slot_access,
                        &mut slot_values,
                    );
                } else {
                    for &branch_region in &regions_clone {
                        promote_region(
                            func,
                            branch_region,
                            all_slots,
                            slot_access,
                            slot_values.clone(),
                        );
                    }
                }
            }
            NodeKind::Theta { body, .. } => {
                let body = *body;

                // Use PRE-COMPUTED slot accesses.
                let needed_slots = slot_access.get(&body).cloned().unwrap_or_default();
                let slots_to_thread: Vec<SlotId> =
                    needed_slots.intersection(all_slots).copied().collect();

                // Ensure all needed slots have initial values.
                ensure_slot_defaults(func, region, &slots_to_thread, &mut slot_values, node_id);

                if !slots_to_thread.is_empty() {
                    promote_theta(
                        func,
                        node_id,
                        body,
                        &slots_to_thread,
                        all_slots,
                        slot_access,
                        &mut slot_values,
                    );
                } else {
                    promote_region(func, body, all_slots, slot_access, slot_values.clone());
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
/// create a const(0) node in `region` before `before_node`.
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
    slot_access: &SlotAccessMap,
    slot_values: &mut BTreeMap<SlotId, PortSource>,
) {
    let debug_scope = func.nodes[node_id].debug_scope;
    let state_count = func.state_domains.len();

    // Phase 1: Add inputs and region args for all slots to thread.
    for &slot in slots_to_thread {
        let incoming_value = slot_values[&slot];
        insert_data_input(func, node_id, incoming_value);
        for &branch_region in branch_regions {
            insert_data_region_arg(func, branch_region);
        }
    }

    // Phase 2: Recurse into each branch with strict scoping.
    let mut branch_arg_sources: Vec<BTreeMap<SlotId, PortSource>> = Vec::new();
    let mut branch_final_values: Vec<BTreeMap<SlotId, PortSource>> = Vec::new();

    for &branch_region in branch_regions {
        // Strict scoping: only region args are valid inside the child region.
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
        let final_vals = promote_region(func, branch_region, all_slots, slot_access, branch_slots);
        branch_arg_sources.push(arg_sources);
        branch_final_values.push(final_vals);
    }

    // Phase 3: Add results and outputs for each slot.
    for &slot in slots_to_thread {
        for (branch_idx, &branch_region) in branch_regions.iter().enumerate() {
            // Fall back to the branch's region arg (not parent scope).
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
    slot_access: &SlotAccessMap,
    slot_values: &mut BTreeMap<SlotId, PortSource>,
) {
    let debug_scope = func.nodes[node_id].debug_scope;

    // Phase 1: Add inputs and region args for all slots.
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
    let final_body_values = promote_region(func, body, all_slots, slot_access, body_slot_values);

    // Phase 3: Add results and outputs for each slot.
    let body_debug_scope = func.regions[body].debug_scope;
    for &slot in slots_to_thread {
        let body_arg = body_arg_sources[&slot];
        let mut final_value = final_body_values.get(&slot).copied().unwrap_or(body_arg);

        // If the slot wasn't modified in the body, the result would reference the
        // same region arg — creating a self-copy (v_N from v_N) that regalloc2
        // can't handle. Insert an Identity node to break the cycle.
        if final_value == body_arg {
            let vreg = func.fresh_vreg();
            let identity_node = func.nodes.push(crate::Node {
                region: body,
                debug_scope: body_debug_scope,
                debug_value: None,
                inputs: vec![InputPort {
                    kind: PortKind::Data,
                    source: body_arg,
                }],
                outputs: vec![OutputPort {
                    kind: PortKind::Data,
                    vreg: Some(vreg),
                    debug_scope: body_debug_scope,
                }],
                kind: NodeKind::Simple(IrOp::Identity),
            });
            func.regions[body].nodes.push(identity_node);
            final_value = PortSource::Node(OutputRef {
                node: identity_node,
                index: 0,
            });
        }

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

/// Replace all uses of a node output within a region and all descendant regions.
/// This is needed because gamma/theta sub-regions can directly reference
/// parent-scope node outputs (not just through passthrough/loop-vars).
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
        // Recurse into sub-regions of structural nodes.
        match &func.nodes[nid].kind {
            NodeKind::Gamma { regions } => {
                let sub_regions: Vec<RegionId> = regions.clone();
                for sub_region in sub_regions {
                    replace_uses_in_region(func, sub_region, from, to);
                }
            }
            NodeKind::Theta { body, .. } => {
                let body = *body;
                replace_uses_in_region(func, body, from, to);
            }
            _ => {}
        }
    }

    for &result_id in &func.regions[region].results {
        if func.region_results[result_id].source == from_source {
            func.region_results[result_id].source = to;
        }
    }
}
