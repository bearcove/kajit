//! Slot-to-register promotion pass.
//!
//! Converts `WriteToSlot`/`ReadFromSlot` operations into proper RVSDG data flow
//! (gamma passthrough values and theta loop-carried variables), enabling the
//! register allocator to keep these values in physical registers instead of
//! spilling them to the stack.
//!
//! This is analogous to LLVM's mem2reg pass (alloca → SSA promotion).
//!
//! Design:
//! - Bottom-up: child regions are promoted before parent wiring is computed
//! - Batch port insertion: all new ports for a gamma/theta are added in one shot
//!   with a single index remap, avoiding stale cached PortSources

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
            NodeKind::Simple(IrOp::SlotAddr { slot, num_slots }) => {
                for i in 0..*num_slots {
                    address_taken.insert(SlotId::new(slot.index() as u32 + i));
                }
            }
            _ => {}
        }
    }
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

fn precompute_slot_access(func: &IrFunc) -> SlotAccessMap {
    let mut map: SlotAccessMap = HashMap::new();
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

                let mut needed_slots = BTreeSet::new();
                for &branch_region in &regions_clone {
                    if let Some(accessed) = slot_access.get(&branch_region) {
                        needed_slots.extend(accessed.iter());
                    }
                }
                let slots_to_thread: Vec<SlotId> =
                    needed_slots.intersection(all_slots).copied().collect();

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

                let needed_slots = slot_access.get(&body).cloned().unwrap_or_default();
                let slots_to_thread: Vec<SlotId> =
                    needed_slots.intersection(all_slots).copied().collect();

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

// ─── Batch port insertion helpers ────────────────────────────────────────────

/// Add N data outputs to a node in one shot, shifting existing state output
/// references once. Returns the indices of the new outputs.
fn batch_insert_data_outputs(
    func: &mut IrFunc,
    node_id: NodeId,
    count: usize,
    debug_scope: crate::DebugScopeId,
) -> Vec<u16> {
    if count == 0 {
        return vec![];
    }

    let state_count = func.state_domains.len();
    let insert_base = func.nodes[node_id].outputs.len() - state_count;

    // Insert all new outputs at once
    let mut new_indices = Vec::with_capacity(count);
    for i in 0..count {
        let vreg = func.fresh_vreg();
        func.nodes[node_id].outputs.insert(
            insert_base + i,
            OutputPort {
                kind: PortKind::Data,
                vreg: Some(vreg),
                debug_scope,
            },
        );
        new_indices.push((insert_base + i) as u16);
    }

    // Single remap: shift all existing references at index >= insert_base by +count
    batch_shift_output_refs(func, node_id, insert_base as u16, count as u16);

    new_indices
}

/// Shift all references to node outputs at index >= `from` by `+delta`.
fn batch_shift_output_refs(func: &mut IrFunc, node_id: NodeId, from: u16, delta: u16) {
    if delta == 0 {
        return;
    }
    let all_node_ids: Vec<NodeId> = func.nodes.iter().map(|(id, _)| id).collect();
    for &nid in &all_node_ids {
        for input in &mut func.nodes[nid].inputs {
            if let PortSource::Node(ref mut oref) = input.source {
                if oref.node == node_id && oref.index >= from {
                    oref.index += delta;
                }
            }
        }
    }

    let all_result_ids: Vec<crate::ResultId> =
        func.region_results.iter().map(|(id, _)| id).collect();
    for &rid in &all_result_ids {
        if let PortSource::Node(ref mut oref) = func.region_results[rid].source {
            if oref.node == node_id && oref.index >= from {
                oref.index += delta;
            }
        }
    }
}

/// Also remap any PortSources in a pass-local map that reference the shifted node.
fn remap_slot_values(
    slot_values: &mut BTreeMap<SlotId, PortSource>,
    node_id: NodeId,
    from: u16,
    delta: u16,
) {
    for val in slot_values.values_mut() {
        if let PortSource::Node(oref) = val {
            if oref.node == node_id && oref.index >= from {
                oref.index += delta;
            }
        }
    }
}

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

fn insert_data_region_result(func: &mut IrFunc, region: RegionId, source: PortSource) {
    let state_count = func.state_domains.len();
    let insert_idx = func.regions[region].results.len() - state_count;
    let result_id = func.region_results.push(RegionResult {
        kind: PortKind::Data,
        source,
    });
    func.regions[region].results.insert(insert_idx, result_id);
}

// ─── Gamma promotion ─────────────────────────────────────────────────────────

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

    // Phase 1: Add inputs and region args for all slots.
    for &slot in slots_to_thread {
        let incoming_value = slot_values[&slot];
        insert_data_input(func, node_id, incoming_value);
        for &branch_region in branch_regions {
            insert_data_region_arg(func, branch_region);
        }
    }

    // Phase 2: Recurse into each branch (bottom-up: children first).
    let mut branch_arg_sources: Vec<BTreeMap<SlotId, PortSource>> = Vec::new();
    let mut branch_final_values: Vec<BTreeMap<SlotId, PortSource>> = Vec::new();

    for &branch_region in branch_regions {
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

    // Phase 3: Batch-add results and outputs for all slots at once.
    for &slot in slots_to_thread {
        for (branch_idx, &branch_region) in branch_regions.iter().enumerate() {
            let final_value = branch_final_values[branch_idx]
                .get(&slot)
                .copied()
                .unwrap_or_else(|| branch_arg_sources[branch_idx][&slot]);
            insert_data_region_result(func, branch_region, final_value);
        }
    }

    // Batch insert all data outputs at once — single shift.
    let new_output_indices =
        batch_insert_data_outputs(func, node_id, slots_to_thread.len(), debug_scope);

    // Remap parent slot_values through the same shift.
    let insert_base = new_output_indices.first().copied().unwrap_or(0);
    remap_slot_values(
        slot_values,
        node_id,
        insert_base + slots_to_thread.len() as u16,
        slots_to_thread.len() as u16,
    );

    // Record new output PortSources.
    for (i, &slot) in slots_to_thread.iter().enumerate() {
        slot_values.insert(
            slot,
            PortSource::Node(OutputRef {
                node: node_id,
                index: new_output_indices[i],
            }),
        );
    }

    // Assertion: no branch region args escaped into the parent's slot_values.
    for (&slot, val) in slot_values.iter() {
        if let PortSource::RegionArg(aref) = val {
            for &branch_region in branch_regions {
                if aref.region == branch_region {
                    panic!(
                        "[slot2reg] SCOPING BUG: after promoting gamma {:?}, \
                         parent slot_values[slot {}] still references branch region {:?} arg {:?}. \
                         Should be a gamma output.",
                        node_id,
                        slot.index(),
                        branch_region,
                        aref.arg,
                    );
                }
            }
        }
    }
}

// ─── Theta promotion ─────────────────────────────────────────────────────────

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

    // Phase 1: Add inputs and body args for all slots (before recursion).
    let mut body_slot_values = BTreeMap::new();
    let mut body_arg_sources: BTreeMap<SlotId, PortSource> = BTreeMap::new();
    for &slot in slots_to_thread {
        let incoming_value = slot_values[&slot];
        insert_data_input(func, node_id, incoming_value);
        let arg_source = insert_data_region_arg(func, body);
        body_slot_values.insert(slot, arg_source);
        body_arg_sources.insert(slot, arg_source);
    }

    // Phase 2: Recurse into body (bottom-up: body is fully promoted before
    // we compute the parent theta's output wiring).
    let final_body_values = promote_region(func, body, all_slots, slot_access, body_slot_values);

    // Phase 3: Add body results for all slots.
    let body_debug_scope = func.regions[body].debug_scope;
    for &slot in slots_to_thread {
        let body_arg = body_arg_sources[&slot];
        let mut final_value = final_body_values.get(&slot).copied().unwrap_or(body_arg);

        // If the slot wasn't modified in the body, insert an Identity node
        // to break the self-reference cycle.
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
    }

    // Phase 4: Batch insert all data outputs at once — single shift.
    let new_output_indices =
        batch_insert_data_outputs(func, node_id, slots_to_thread.len(), debug_scope);

    // Remap parent slot_values through the same shift.
    let insert_base = new_output_indices.first().copied().unwrap_or(0);
    remap_slot_values(
        slot_values,
        node_id,
        insert_base + slots_to_thread.len() as u16,
        slots_to_thread.len() as u16,
    );

    // Record new output PortSources.
    for (i, &slot) in slots_to_thread.iter().enumerate() {
        slot_values.insert(
            slot,
            PortSource::Node(OutputRef {
                node: node_id,
                index: new_output_indices[i],
            }),
        );
    }

    // Assertion: no body args escaped into the parent's slot_values.
    for (&slot, val) in slot_values.iter() {
        if let PortSource::RegionArg(aref) = val {
            if aref.region == body {
                panic!(
                    "[slot2reg] SCOPING BUG: after promoting theta {:?}, \
                     parent slot_values[slot {}] still references body region arg {:?}. \
                     Should be a theta output.",
                    node_id,
                    slot.index(),
                    aref.arg,
                );
            }
        }
    }
}

// ─── Replace uses ────────────────────────────────────────────────────────────

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
