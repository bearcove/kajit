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

use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};
use std::sync::atomic::{AtomicBool, Ordering};

use crate::{
    InputPort, IrFunc, IrOp, NodeId, NodeKind, OutputPort, OutputRef, PortKind, PortSource,
    RegionArg, RegionArgRef, RegionId, RegionResult, SlotId,
};

static DEBUG_S2R_INIT: AtomicBool = AtomicBool::new(false);
static DEBUG_S2R: AtomicBool = AtomicBool::new(false);

fn debug_s2r() -> bool {
    if !DEBUG_S2R_INIT.load(Ordering::Relaxed) {
        let val = std::env::var("KAJIT_DEBUG_S2R").is_ok();
        DEBUG_S2R.store(val, Ordering::Relaxed);
        DEBUG_S2R_INIT.store(true, Ordering::Relaxed);
    }
    DEBUG_S2R.load(Ordering::Relaxed)
}

fn fmt_port_source(func: &IrFunc, ps: PortSource) -> String {
    match ps {
        PortSource::Node(oref) => {
            let kind = &func.nodes[oref.node].kind;
            format!("{:?}[{}] ({:?})", oref.node, oref.index, kind)
        }
        PortSource::RegionArg(aref) => {
            format!("arg({:?}, {:?})", aref.region, aref.arg)
        }
    }
}

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
    let _ = promote_region(func, lambda_body, &all_slots, &slot_access, initial_values);
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
/// Result of promote_region: final slot values + set of initial values that
/// were consumed (read before overwrite).
struct PromoteResult {
    final_values: BTreeMap<SlotId, PortSource>,
    /// Initial slot values (body_args / branch_args) that were read by a
    /// ReadFromSlot before being overwritten by a WriteToSlot. These are
    /// "consumed" — the initial value matters for correct execution.
    consumed_initial_values: HashSet<PortSource>,
}

fn promote_region(
    func: &mut IrFunc,
    region: RegionId,
    all_slots: &BTreeSet<SlotId>,
    slot_access: &SlotAccessMap,
    mut slot_values: BTreeMap<SlotId, PortSource>,
) -> PromoteResult {
    // Track which initial slot values are consumed (read before overwrite).
    let initial_values: BTreeMap<SlotId, PortSource> = slot_values.clone();
    let mut consumed: HashSet<PortSource> = HashSet::new();

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
                    // If the current value is STILL the initial value (body_arg),
                    // mark it as consumed — the initial value matters.
                    if initial_values.get(&slot) == Some(&current_val) {
                        consumed.insert(current_val);
                    }

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

                if debug_s2r() {
                    eprintln!(
                        "[s2r] region {:?}: WriteToSlot {} => {}",
                        region,
                        slot.index(),
                        fmt_port_source(func, written_value)
                    );
                }

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
                    let gamma_consumed = promote_gamma(
                        func,
                        node_id,
                        &regions_clone,
                        &slots_to_thread,
                        all_slots,
                        slot_access,
                        &mut slot_values,
                    );
                    consumed.extend(gamma_consumed);
                } else {
                    for &branch_region in &regions_clone {
                        let r = promote_region(
                            func,
                            branch_region,
                            all_slots,
                            slot_access,
                            slot_values.clone(),
                        );
                        consumed.extend(r.consumed_initial_values);
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
                    let r = promote_region(func, body, all_slots, slot_access, slot_values.clone());
                    consumed.extend(r.consumed_initial_values);
                }
            }
            _ => {}
        }
    }

    let remove_set: BTreeSet<NodeId> = nodes_to_remove.into_iter().collect();
    func.regions[region]
        .nodes
        .retain(|nid| !remove_set.contains(nid));

    PromoteResult {
        final_values: slot_values,
        consumed_initial_values: consumed,
    }
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
            if let PortSource::Node(ref mut oref) = input.source
                && oref.node == node_id
                && oref.index >= from
            {
                oref.index += delta;
            }
        }
    }

    let all_result_ids: Vec<crate::ResultId> =
        func.region_results.iter().map(|(id, _)| id).collect();
    for &rid in &all_result_ids {
        if let PortSource::Node(ref mut oref) = func.region_results[rid].source
            && oref.node == node_id
            && oref.index >= from
        {
            oref.index += delta;
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
        if let PortSource::Node(oref) = val
            && oref.node == node_id
            && oref.index >= from
        {
            oref.index += delta;
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

/// Returns the set of slot_values PortSources that were consumed
/// (read before overwrite) inside any gamma branch.
fn promote_gamma(
    func: &mut IrFunc,
    node_id: NodeId,
    branch_regions: &[RegionId],
    slots_to_thread: &[SlotId],
    all_slots: &BTreeSet<SlotId>,
    slot_access: &SlotAccessMap,
    slot_values: &mut BTreeMap<SlotId, PortSource>,
) -> HashSet<PortSource> {
    let mut consumed_parent_sources: HashSet<PortSource> = HashSet::new();
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
        let result = promote_region(func, branch_region, all_slots, slot_access, branch_slots);
        // Map consumed branch args back to parent slot_values.
        for (&slot, &arg_src) in &arg_sources {
            if result.consumed_initial_values.contains(&arg_src)
                && let Some(&parent_src) = slot_values.get(&slot)
            {
                consumed_parent_sources.insert(parent_src);
            }
        }
        let final_vals = result.final_values;
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

    consumed_parent_sources
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

    if debug_s2r() {
        eprintln!("[s2r] promote_theta {:?} body={:?}", node_id, body);
        eprintln!("[s2r]   slots_to_thread: {:?}", slots_to_thread);
        for &slot in slots_to_thread {
            eprintln!(
                "[s2r]   incoming slot {} = {}",
                slot.index(),
                fmt_port_source(func, slot_values[&slot])
            );
        }
    }

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

    if debug_s2r() {
        eprintln!("[s2r]   body_slot_values (entry):");
        for (&slot, &val) in &body_slot_values {
            eprintln!(
                "[s2r]     slot {} = {}",
                slot.index(),
                fmt_port_source(func, val)
            );
        }
    }

    // Pre-scan: for unbounded thetas (outer element loops), identify slots
    // whose first access in the body is a write (re-initialized slots).
    // Bounded thetas (varint decoders) are unrolled later; their slots become
    // gamma outputs and don't benefit from this optimization.
    let max_iterations = match &func.nodes[node_id].kind {
        NodeKind::Theta { max_iterations, .. } => *max_iterations,
        _ => None,
    };
    // Only scan for reinit slots in unbounded thetas with reasonable body size.
    // Deep nesting (e.g., deep_struct) can cause exponential scan time.
    let body_node_count = func.regions[body].nodes.len();
    let reinit_slots = if max_iterations.is_none() && body_node_count < 200 {
        find_reinit_slots(func, body, slots_to_thread)
    } else {
        BTreeSet::new()
    };

    // Phase 2: Recurse into body (bottom-up: body is fully promoted before
    // we compute the parent theta's output wiring).
    let body_result = promote_region(func, body, all_slots, slot_access, body_slot_values);
    let final_body_values = body_result.final_values;

    if debug_s2r() {
        eprintln!("[s2r]   final_body_values (exit):");
        for (&slot, &val) in &final_body_values {
            eprintln!(
                "[s2r]     slot {} = {}",
                slot.index(),
                fmt_port_source(func, val)
            );
        }
    }

    // Post-promotion safety analysis: use the consumed_initial_values set
    // from promote_region to identify body_args that were NEVER consumed
    // (never read by a ReadFromSlot before being overwritten by WriteToSlot).
    // These body_args are truly dead — replacing them with Const(0) is safe.
    let mut dead_body_args: BTreeSet<SlotId> = BTreeSet::new();
    if max_iterations.is_none() && body_node_count < 200 {
        for &slot in slots_to_thread {
            let body_arg_src = body_arg_sources[&slot];
            // The body_arg is safe to eliminate if BOTH:
            // 1. NOT consumed via ReadFromSlot in any gamma branch
            // 2. NOT referenced by non-gamma nodes (catches cursor/state-domain
            //    access invisible to slot-level tracking)
            let not_consumed = !body_result.consumed_initial_values.contains(&body_arg_src);
            let no_non_gamma_refs = !is_port_source_referenced_non_gamma(func, body_arg_src);
            let is_scalar_temp = func.scalar_temp_slots.contains(&slot);
            let is_not_multi = !func.multi_slot_group.contains(&slot);
            // Also require: slot must be in the reinit set (first access
            // in gamma branch is WriteToSlot(Const(0)), no non-zero writes
            // at the gamma level). This catches l12/l13/l14 which are
            // assigned from varint results — NOT pure zero-reinit.
            let is_reinit = reinit_slots.contains(&slot);
            if not_consumed && no_non_gamma_refs && is_scalar_temp && is_not_multi && is_reinit {
                dead_body_args.insert(slot);
                if debug_s2r() {
                    eprintln!(
                        "[s2r]     slot {} dead_body_arg: consumed={}, non_gamma_refs={}, scalar={}, multi={}",
                        slot.index(),
                        !not_consumed,
                        !no_non_gamma_refs,
                        is_scalar_temp,
                        !is_not_multi,
                    );
                }
            }
        }
        if !dead_body_args.is_empty() {
            func.theta_reinit_slots
                .insert(node_id, dead_body_args.clone());
            if debug_s2r() {
                eprintln!(
                    "[s2r]   dead_body_args (promotion-time): {:?}",
                    dead_body_args.iter().map(|s| s.index()).collect::<Vec<_>>()
                );
            }
        }
    }

    // Phase 3: Add body results for all slots.
    let body_debug_scope = func.regions[body].debug_scope;
    for &slot in slots_to_thread {
        let body_arg = body_arg_sources[&slot];
        let mut final_value = final_body_values.get(&slot).copied().unwrap_or(body_arg);
        let _ = &reinit_slots; // legacy, superseded by dead_body_args
        if debug_s2r() {
            let from_map = final_body_values.contains_key(&slot);
            eprintln!(
                "[s2r]   theta {:?} slot {} feedback: {} (from_map={}, eq_body_arg={})",
                node_id,
                slot.index(),
                fmt_port_source(func, final_value),
                from_map,
                final_value == body_arg
            );
        }

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

    // Record slot → port mapping for dead_theta_ports.
    func.theta_port_slots
        .insert(node_id, slots_to_thread.to_vec());
    // theta_reinit_slots is set above from dead_body_args (promotion-time truth).

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
        if let PortSource::RegionArg(aref) = val
            && aref.region == body
        {
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

/// Check if a PortSource is used by any NON-GAMMA node input in a region.
/// Uses in gamma inputs (for break-path pass-through) don't count — those
/// become dead when the theta output is unused.
/// Check if a region (direct nodes only) contains a WriteToSlot(slot, Const(0)).
fn has_write_const0_in_region(func: &IrFunc, region: RegionId, slot: SlotId) -> bool {
    for &nid in &func.regions[region].nodes {
        if let NodeKind::Simple(IrOp::WriteToSlot { slot: s, .. }) = &func.nodes[nid].kind
            && *s == slot
        {
            let write_source = &func.nodes[nid].inputs[0].source;
            if crate::dead_theta_ports::resolve_to_const(func, &write_source) == Some(0) {
                return true;
            }
        }
    }
    false
}

/// Check if a region (direct nodes only) contains a WriteToSlot for the slot with
/// a non-zero value (indicating struct field assignment, not pure re-initialization).
/// Check if a PortSource is referenced by any NON-GAMMA node in the IR.
/// Gamma inputs are handled separately via the consumed-set propagation.
fn is_port_source_referenced_non_gamma(func: &IrFunc, source: PortSource) -> bool {
    for (_nid, node) in func.nodes.iter() {
        if matches!(node.kind, NodeKind::Gamma { .. }) {
            continue;
        }
        for input in &node.inputs {
            if input.source == source {
                return true;
            }
        }
    }
    for (_rid, result) in func.region_results.iter() {
        if result.source == source {
            return true;
        }
    }
    false
}

/// Check for non-zero writes to a slot in a region and one level of nested gammas.
/// Does NOT recurse into thetas (they have their own slot promotion).
fn has_nonzero_write_shallow(func: &IrFunc, region: RegionId, slot: SlotId) -> bool {
    for &nid in &func.regions[region].nodes {
        match &func.nodes[nid].kind {
            NodeKind::Simple(IrOp::WriteToSlot { slot: s, .. }) if *s == slot => {
                let src = &func.nodes[nid].inputs[0].source;
                if crate::dead_theta_ports::resolve_to_const(func, &src) != Some(0) {
                    return true;
                }
            }
            NodeKind::Gamma { regions, .. } => {
                for &sub in regions {
                    if has_nonzero_write_shallow(func, sub, slot) {
                        return true;
                    }
                }
            }
            // Don't recurse into Theta — those are inner loops with own promotion
            _ => {}
        }
    }
    false
}

fn has_nonzero_write_in_region(func: &IrFunc, region: RegionId, slot: SlotId) -> bool {
    for &nid in &func.regions[region].nodes {
        match &func.nodes[nid].kind {
            NodeKind::Simple(IrOp::WriteToSlot { slot: s, .. }) if *s == slot => {
                let write_source = &func.nodes[nid].inputs[0].source;
                if crate::dead_theta_ports::resolve_to_const(func, &write_source) != Some(0) {
                    return true;
                }
            }
            NodeKind::Gamma { regions, .. } => {
                for &sub in regions {
                    if has_nonzero_write_in_region(func, sub, slot) {
                        return true;
                    }
                }
            }
            NodeKind::Theta { body, .. } => {
                if has_nonzero_write_in_region(func, *body, slot) {
                    return true;
                }
            }
            _ => {}
        }
    }
    false
}

/// Check if a region (or any nested sub-region) contains a ReadFromSlot for the slot.
fn has_read_in_region_recursive(func: &IrFunc, region: RegionId, slot: SlotId) -> bool {
    for &nid in &func.regions[region].nodes {
        match &func.nodes[nid].kind {
            NodeKind::Simple(IrOp::ReadFromSlot { slot: s, .. }) if *s == slot => {
                return true;
            }
            NodeKind::Gamma { regions, .. } => {
                for &sub in regions {
                    if has_read_in_region_recursive(func, sub, slot) {
                        return true;
                    }
                }
            }
            NodeKind::Theta { body, .. } => {
                if has_read_in_region_recursive(func, *body, slot) {
                    return true;
                }
            }
            _ => {}
        }
    }
    false
}

/// Find slots whose first access inside the theta body is a WriteToSlot.
/// These slots are "re-initialized" — their body arg value is dead.
///
/// We scan the body's nodes looking for ReadFromSlot/WriteToSlot operations.
/// For each slot, we track whether its first appearance is a Read or Write.
/// If Write: the slot is re-initialized before any read → body arg is dead.
fn find_reinit_slots(
    func: &IrFunc,
    body: RegionId,
    slots_to_thread: &[SlotId],
) -> BTreeSet<SlotId> {
    use crate::IrOp;
    let slot_set: BTreeSet<SlotId> = slots_to_thread.iter().copied().collect();
    let mut first_access_is_write: BTreeSet<SlotId> = BTreeSet::new();
    let mut seen: BTreeSet<SlotId> = BTreeSet::new();

    // Scan nodes in the body region. They're in insertion order, which
    // follows the HIR statement order within the loop body.
    for &nid in &func.regions[body].nodes {
        match &func.nodes[nid].kind {
            NodeKind::Simple(IrOp::WriteToSlot { slot, .. }) => {
                if slot_set.contains(slot) && !seen.contains(slot) {
                    first_access_is_write.insert(*slot);
                    seen.insert(*slot);
                }
            }
            NodeKind::Simple(IrOp::ReadFromSlot { slot, .. }) => {
                if slot_set.contains(slot) && !seen.contains(slot) {
                    // First access is a read → NOT re-initialized
                    seen.insert(*slot);
                }
            }
            NodeKind::Gamma { regions, .. } => {
                // For slots accessed inside a gamma: only mark as reinit if
                // the slot has a WriteToSlot(Const(0)) AND NO ReadFromSlot
                // ANYWHERE in the gamma branch (including nested sub-regions).
                // This prevents false positives for struct sub-fields that
                // are written then read within the same gamma branch.
                for &slot in slots_to_thread {
                    if seen.contains(&slot) || !slot_set.contains(&slot) {
                        continue;
                    }
                    let mut accessed_in_gamma = false;
                    let mut all_branches_safe = true;
                    for &sub_region in regions {
                        let has_write_0 = has_write_const0_in_region(func, sub_region, slot);
                        let has_nonzero_write = has_nonzero_write_in_region(func, sub_region, slot);
                        let has_read = has_read_in_region_recursive(func, sub_region, slot);
                        if has_write_0 || has_nonzero_write || has_read {
                            accessed_in_gamma = true;
                            // Safe only if: writes Const(0), no non-zero writes,
                            // and no reads at this gamma level.
                            if !has_write_0 || has_nonzero_write || has_read {
                                all_branches_safe = false;
                            }
                        }
                    }
                    if accessed_in_gamma {
                        seen.insert(slot);
                        if all_branches_safe {
                            first_access_is_write.insert(slot);
                        }
                    }
                }
            }
            _ => {}
        }
    }

    // Sanity filter: only include slots that have BOTH WriteToSlot AND
    // ReadFromSlot somewhere in the theta body. Slots that are only written
    // (but read via field-based access) are struct sub-fields with complex
    // access patterns we can't safely optimize.
    let mut has_read_anywhere: BTreeSet<SlotId> = BTreeSet::new();
    for &slot in slots_to_thread {
        if has_read_in_region_recursive(func, body, slot) {
            has_read_anywhere.insert(slot);
        }
    }
    // Exclude slots that have ReadFromSlot at the gamma branch level
    // (depth 1). Reads inside nested thetas (depth 2+) are fine — those
    // thetas get their own slot promotion and don't depend on the outer
    // theta's body arg.
    let mut read_at_gamma_level: BTreeSet<SlotId> = BTreeSet::new();
    for &nid in &func.regions[body].nodes {
        if let NodeKind::Gamma { regions, .. } = &func.nodes[nid].kind {
            for &sub_region in regions {
                for &sub_nid in &func.regions[sub_region].nodes {
                    // Only check direct ReadFromSlot at the gamma branch level,
                    // NOT recursively into nested gammas/thetas.
                    if let NodeKind::Simple(IrOp::ReadFromSlot { slot, .. }) =
                        &func.nodes[sub_nid].kind
                    {
                        read_at_gamma_level.insert(*slot);
                    }
                }
            }
        }
    }

    // Also exclude slots with non-zero writes at the gamma branch level.
    // These are struct sub-fields that get StoreToAddr-based reads we can't
    // track via ReadFromSlot. Check only at depth 1 (not recursive into
    // nested thetas, which have their own slot promotion).
    let mut has_nonzero_write_at_gamma: BTreeSet<SlotId> = BTreeSet::new();
    for &nid in &func.regions[body].nodes {
        if let NodeKind::Gamma { regions, .. } = &func.nodes[nid].kind {
            for &sub_region in regions {
                for &slot in slots_to_thread {
                    // Check direct nodes + one level of nested gammas
                    // (but not into thetas which have their own promotion).
                    if has_nonzero_write_shallow(func, sub_region, slot) {
                        has_nonzero_write_at_gamma.insert(slot);
                    }
                }
            }
        }
    }

    // Final filter: only retain reinit slots where ALL reads are inside
    // nested thetas (varint decoder loops with their own slot promotion).
    // Struct sub-fields have reads inside nested gammas (error checks) and
    // hidden StoreToAddr writes that our slot-level checks can't detect.
    let mut has_read_outside_theta: BTreeSet<SlotId> = BTreeSet::new();
    for &nid in &func.regions[body].nodes {
        if let NodeKind::Gamma { regions, .. } = &func.nodes[nid].kind {
            for &sub_region in regions {
                for &sub_nid in &func.regions[sub_region].nodes {
                    match &func.nodes[sub_nid].kind {
                        // ReadFromSlot directly at gamma branch level → not safe
                        NodeKind::Simple(IrOp::ReadFromSlot { slot, .. }) => {
                            has_read_outside_theta.insert(*slot);
                        }
                        // ReadFromSlot inside nested gamma → not safe
                        NodeKind::Gamma { regions: inner, .. } => {
                            for &inner_region in inner {
                                for &slot in slots_to_thread {
                                    if has_read_in_region_recursive(func, inner_region, slot) {
                                        has_read_outside_theta.insert(slot);
                                    }
                                }
                            }
                        }
                        // Nested theta: reads are fine (own promotion)
                        NodeKind::Theta { .. } => {}
                        _ => {}
                    }
                }
            }
        }
    }

    if debug_s2r() {
        for &s in &first_access_is_write {
            let read_any = has_read_anywhere.contains(&s);
            let read_out = has_read_outside_theta.contains(&s);
            eprintln!(
                "[s2r]   slot {} pre-retain: read_anywhere={} read_outside_theta={}",
                s.index(),
                read_any,
                read_out
            );
        }
    }
    first_access_is_write
        .retain(|s| has_read_anywhere.contains(s) && !has_read_outside_theta.contains(s));

    if debug_s2r() && !first_access_is_write.is_empty() {
        eprintln!(
            "[s2r]   reinit_slots: {:?} (of {} checked, {} seen)",
            first_access_is_write
                .iter()
                .map(|s| s.index())
                .collect::<Vec<_>>(),
            slots_to_thread.len(),
            seen.len()
        );
    }

    first_access_is_write
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
