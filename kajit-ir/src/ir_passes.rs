use std::collections::{HashMap, HashSet};

use crate::{
    InputPort, IrFunc, IrOp, LambdaId, Node, NodeId, NodeKind, OutputRef, PortKind, PortSource,
    Region, RegionArgRef, RegionId, RegionResult, verify,
};

const MAX_INLINE_NODES_SINGLE_USE: usize = 256;
const MAX_INLINE_NODES_MULTI_USE: usize = 64;
const MAX_INLINE_CALL_SITES_MULTI_USE: usize = 4;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum UseSite {
    NodeInput { node: NodeId, input_index: u16 },
    RegionResult { region: RegionId, result_index: u16 },
}

#[derive(Default)]
struct UseLists {
    by_output: HashMap<OutputRef, Vec<UseSite>>,
}

impl UseLists {
    fn build(func: &IrFunc) -> Self {
        let mut out = Self::default();

        for (region_id, region) in func.regions.iter() {
            for &node_id in &region.nodes {
                for (input_index, input) in func.nodes[node_id].inputs.iter().enumerate() {
                    let PortSource::Node(source) = input.source else {
                        continue;
                    };
                    out.by_output
                        .entry(source)
                        .or_default()
                        .push(UseSite::NodeInput {
                            node: node_id,
                            input_index: input_index as u16,
                        });
                }
            }

            for (result_index, result) in region.results.iter().enumerate() {
                let PortSource::Node(source) = result.source else {
                    continue;
                };
                out.by_output
                    .entry(source)
                    .or_default()
                    .push(UseSite::RegionResult {
                        region: region_id,
                        result_index: result_index as u16,
                    });
            }
        }

        out
    }

    fn output_has_uses(&self, out: OutputRef) -> bool {
        self.by_output
            .get(&out)
            .is_some_and(|uses| !uses.is_empty())
    }

    fn replace_output_use(&mut self, func: &mut IrFunc, from: OutputRef, to: PortSource) {
        let from_source = PortSource::Node(from);
        if from_source == to {
            return;
        }
        let Some(use_sites) = self.by_output.remove(&from) else {
            return;
        };

        let mut rewritten_sites = Vec::new();
        for use_site in use_sites {
            let rewritten = match use_site {
                UseSite::NodeInput { node, input_index } => {
                    let Some(input) = func.nodes[node].inputs.get_mut(input_index as usize) else {
                        continue;
                    };
                    if input.source == from_source {
                        input.source = to;
                        true
                    } else {
                        false
                    }
                }
                UseSite::RegionResult {
                    region,
                    result_index,
                } => {
                    let Some(result) = func.regions[region].results.get_mut(result_index as usize)
                    else {
                        continue;
                    };
                    if result.source == from_source {
                        result.source = to;
                        true
                    } else {
                        false
                    }
                }
            };
            if rewritten {
                rewritten_sites.push(use_site);
            }
        }

        if let PortSource::Node(to_output) = to {
            self.by_output
                .entry(to_output)
                .or_default()
                .extend(rewritten_sites);
        }
    }
}

// r[impl ir.passes]
pub fn run_default_passes(func: &mut IrFunc) {
    bounds_check_coalescing_pass(func);
    debug_verify(func, "bounds_check_coalescing_pass");
    hoist_theta_loop_invariant_setup_pass(func);
    debug_verify(func, "hoist_theta_loop_invariant_setup_pass");
    inline_apply_pass(func);
    debug_verify(func, "inline_apply_pass");
    dead_code_elimination_pass(func);
    debug_verify(func, "dead_code_elimination_pass");
}

#[cfg(debug_assertions)]
fn debug_verify(func: &IrFunc, pass_name: &str) {
    if let Err(err) = verify(func) {
        panic!("IR verification failed after {pass_name}: {err}");
    }
}

#[cfg(not(debug_assertions))]
fn debug_verify(_func: &IrFunc, _pass_name: &str) {}

// r[impl ir.passes.planned]
fn bounds_check_coalescing_pass(func: &mut IrFunc) {
    loop {
        let mut use_lists = UseLists::build(func);
        let region_ids: Vec<RegionId> = func.regions.iter().map(|(rid, _)| rid).collect();
        let mut changed = false;
        for rid in region_ids {
            if coalesce_bounds_checks_in_region(func, rid, &mut use_lists) {
                changed = true;
                break;
            }
        }
        if !changed {
            break;
        }
    }
}

fn coalesce_bounds_checks_in_region(
    func: &mut IrFunc,
    region_id: RegionId,
    use_lists: &mut UseLists,
) -> bool {
    let nodes = func.regions[region_id].nodes.clone();
    for (i, first) in nodes.iter().copied().enumerate() {
        let Some(first_count) = bounds_check_count(func, first) else {
            continue;
        };
        let Some(first_cursor_out) = state_cursor_output_ref(func, first) else {
            continue;
        };
        let mut chain_source = PortSource::Node(first_cursor_out);
        let mut consumed_bytes = 0u32;

        for second in nodes.iter().copied().skip(i + 1) {
            let Some(second_cursor_in) = state_cursor_input_source(func, second) else {
                break;
            };
            if second_cursor_in != chain_source {
                break;
            }

            if let Some(second_count) = bounds_check_count(func, second) {
                let combined = first_count.max(consumed_bytes.saturating_add(second_count));
                if let NodeKind::Simple(IrOp::BoundsCheck { count }) = &mut func.nodes[first].kind {
                    *count = combined;
                }

                let Some(second_cursor_out) = state_cursor_output_ref(func, second) else {
                    break;
                };
                replace_output_use(func, use_lists, second_cursor_out, second_cursor_in);
                if let Some(pos) = func.regions[region_id]
                    .nodes
                    .iter()
                    .position(|&nid| nid == second)
                {
                    func.regions[region_id].nodes.remove(pos);
                    return true;
                }
                break;
            }

            let Some((advance, next_source)) = cursor_chain_step(func, second, second_cursor_in)
            else {
                break;
            };
            consumed_bytes = consumed_bytes.saturating_add(advance);
            chain_source = next_source;
        }
    }
    false
}

fn bounds_check_count(func: &IrFunc, node_id: NodeId) -> Option<u32> {
    match &func.nodes[node_id].kind {
        NodeKind::Simple(IrOp::BoundsCheck { count }) => Some(*count),
        _ => None,
    }
}

fn state_cursor_input_source(func: &IrFunc, node_id: NodeId) -> Option<PortSource> {
    func.nodes[node_id]
        .inputs
        .iter()
        .find(|inp| inp.kind == PortKind::StateCursor)
        .map(|inp| inp.source)
}

fn state_cursor_output_ref(func: &IrFunc, node_id: NodeId) -> Option<OutputRef> {
    func.nodes[node_id]
        .outputs
        .iter()
        .position(|out| out.kind == PortKind::StateCursor)
        .map(|idx| OutputRef {
            node: node_id,
            index: idx as u16,
        })
}

fn cursor_chain_step(
    func: &IrFunc,
    node_id: NodeId,
    state_in: PortSource,
) -> Option<(u32, PortSource)> {
    let node = &func.nodes[node_id];
    let input = state_cursor_input_source(func, node_id)?;
    if input != state_in {
        return None;
    }
    let out = PortSource::Node(state_cursor_output_ref(func, node_id)?);
    let NodeKind::Simple(op) = &node.kind else {
        return None;
    };

    let advance = op.cursor_advance()?;
    Some((advance, out))
}

// r[impl ir.passes.pre-regalloc.loop-invariants]
fn hoist_theta_loop_invariant_setup_pass(func: &mut IrFunc) {
    loop {
        let live_nodes = collect_live_nodes(func);
        let theta_nodes: Vec<NodeId> = func
            .nodes
            .iter()
            .filter_map(|(nid, node)| {
                if live_nodes.contains(&nid) && matches!(node.kind, NodeKind::Theta { .. }) {
                    Some(nid)
                } else {
                    None
                }
            })
            .collect();

        let mut changed = false;
        for theta in theta_nodes {
            let mut use_lists = UseLists::build(func);
            if hoist_theta_loop_invariants_for_node(func, theta, &mut use_lists) {
                changed = true;
                break;
            }
        }

        if !changed {
            break;
        }
    }
}

fn hoist_theta_loop_invariants_for_node(
    func: &mut IrFunc,
    theta: NodeId,
    use_lists: &mut UseLists,
) -> bool {
    let (parent_region, body_region) = {
        let node = &func.nodes[theta];
        let NodeKind::Theta { body } = &node.kind else {
            return false;
        };
        (node.region, *body)
    };

    let body_nodes = func.regions[body_region].nodes.clone();
    let body_node_set: HashSet<NodeId> = body_nodes.iter().copied().collect();
    let mut hoistable_set = HashSet::new();
    let mut hoistable_nodes = Vec::new();

    for node_id in body_nodes.iter().copied() {
        let NodeKind::Simple(op) = &func.nodes[node_id].kind else {
            continue;
        };
        if !is_hoistable_theta_setup_op(op) {
            continue;
        }
        if func.nodes[node_id].inputs.iter().all(|inp| {
            source_is_theta_invariant(inp.source, body_region, &body_node_set, &hoistable_set)
        }) {
            hoistable_set.insert(node_id);
            hoistable_nodes.push(node_id);
        }
    }

    if hoistable_nodes.is_empty() {
        return false;
    }

    let Some(theta_pos) = func.regions[parent_region]
        .nodes
        .iter()
        .position(|&n| n == theta)
    else {
        return false;
    };

    let mut old_to_new = HashMap::new();
    let mut inserted = Vec::new();

    for old_node in hoistable_nodes.iter().copied() {
        let old_inputs = func.nodes[old_node].inputs.clone();
        let old_outputs = func.nodes[old_node].outputs.clone();
        let NodeKind::Simple(op) = &func.nodes[old_node].kind else {
            continue;
        };

        let new_inputs: Vec<InputPort> = old_inputs
            .into_iter()
            .map(|inp| InputPort {
                kind: inp.kind,
                source: remap_hoisted_source(inp.source, &old_to_new),
            })
            .collect();

        let new_node = func.nodes.push(Node {
            region: parent_region,
            inputs: new_inputs,
            outputs: old_outputs,
            kind: NodeKind::Simple(op.clone()),
        });

        old_to_new.insert(old_node, new_node);
        inserted.push(new_node);
    }

    for (idx, node_id) in inserted.iter().copied().enumerate() {
        func.regions[parent_region]
            .nodes
            .insert(theta_pos + idx, node_id);
    }

    for old_node in hoistable_nodes.iter().copied() {
        let Some(new_node) = old_to_new.get(&old_node).copied() else {
            continue;
        };
        let output_len = func.nodes[old_node].outputs.len();
        for out_idx in 0..output_len {
            let from = OutputRef {
                node: old_node,
                index: out_idx as u16,
            };
            let to = PortSource::Node(OutputRef {
                node: new_node,
                index: out_idx as u16,
            });
            replace_output_use(func, use_lists, from, to);
        }
    }

    let hoisted: HashSet<NodeId> = hoistable_nodes.into_iter().collect();
    func.regions[body_region]
        .nodes
        .retain(|nid| !hoisted.contains(nid));

    true
}

fn remap_hoisted_source(source: PortSource, old_to_new: &HashMap<NodeId, NodeId>) -> PortSource {
    match source {
        PortSource::Node(out) => {
            if let Some(new_node) = old_to_new.get(&out.node).copied() {
                PortSource::Node(OutputRef {
                    node: new_node,
                    index: out.index,
                })
            } else {
                PortSource::Node(out)
            }
        }
        other => other,
    }
}

fn source_is_theta_invariant(
    source: PortSource,
    body_region: RegionId,
    body_node_set: &HashSet<NodeId>,
    hoistable_set: &HashSet<NodeId>,
) -> bool {
    match source {
        PortSource::Node(out) => {
            !body_node_set.contains(&out.node) || hoistable_set.contains(&out.node)
        }
        PortSource::RegionArg(arg) => arg.region != body_region,
    }
}

fn is_hoistable_theta_setup_op(op: &IrOp) -> bool {
    matches!(
        op,
        IrOp::Const { .. }
            | IrOp::Add
            | IrOp::Sub
            | IrOp::And
            | IrOp::Or
            | IrOp::Xor
            | IrOp::Shl
            | IrOp::Shr
            | IrOp::CmpNe
            | IrOp::ZigzagDecode { .. }
            | IrOp::SignExtend { .. }
    )
}

fn dead_code_elimination_pass(func: &mut IrFunc) {
    loop {
        let use_lists = UseLists::build(func);
        let mut dead_node = None;

        let region_ids: Vec<RegionId> = func.regions.iter().map(|(rid, _)| rid).collect();
        'search: for region_id in region_ids {
            for &node_id in &func.regions[region_id].nodes {
                if is_dead_pure_node(func, &use_lists, node_id) {
                    dead_node = Some((region_id, node_id));
                    break 'search;
                }
            }
        }

        let Some((region_id, node_id)) = dead_node else {
            break;
        };

        let Some(position) = func.regions[region_id]
            .nodes
            .iter()
            .position(|&nid| nid == node_id)
        else {
            break;
        };
        func.regions[region_id].nodes.remove(position);
    }
}

fn is_dead_pure_node(func: &IrFunc, use_lists: &UseLists, node_id: NodeId) -> bool {
    let NodeKind::Simple(op) = &func.nodes[node_id].kind else {
        return false;
    };
    if op.has_side_effects() {
        return false;
    }

    for output_index in 0..func.nodes[node_id].outputs.len() {
        if use_lists.output_has_uses(OutputRef {
            node: node_id,
            index: output_index as u16,
        }) {
            return false;
        }
    }
    true
}

fn inline_apply_pass(func: &mut IrFunc) {
    loop {
        let mut use_lists = UseLists::build(func);
        let live_nodes = collect_live_nodes(func);
        let lambda_owner = build_region_owner_map(func);
        let call_sites = count_apply_call_sites(func, &live_nodes);
        let lambda_sizes = lambda_sizes(func);
        let lambda_has_control_flow = lambda_has_control_flow(func);
        let mut changed = false;

        let candidates: Vec<NodeId> = func
            .nodes
            .iter()
            .filter_map(|(nid, node)| {
                if !live_nodes.contains(&nid) {
                    return None;
                }
                let NodeKind::Apply { target } = node.kind else {
                    return None;
                };
                let caller_lambda = *lambda_owner
                    .get(&node.region)
                    .expect("every region should belong to exactly one lambda");
                if should_inline(
                    caller_lambda,
                    target,
                    &call_sites,
                    &lambda_sizes,
                    &lambda_has_control_flow,
                ) {
                    Some(nid)
                } else {
                    None
                }
            })
            .collect();

        for apply in candidates {
            if inline_one_apply(func, &mut use_lists, apply) {
                changed = true;
                break;
            }
        }

        if !changed {
            break;
        }
    }
}

fn should_inline(
    caller_lambda: LambdaId,
    target: LambdaId,
    call_sites: &HashMap<LambdaId, usize>,
    lambda_sizes: &HashMap<LambdaId, usize>,
    lambda_has_control_flow: &HashMap<LambdaId, bool>,
) -> bool {
    if caller_lambda == target {
        return false;
    }
    if lambda_has_control_flow
        .get(&target)
        .copied()
        .unwrap_or(false)
    {
        return false;
    }
    let size = lambda_sizes.get(&target).copied().unwrap_or(0);
    let uses = call_sites.get(&target).copied().unwrap_or(0);

    if uses <= 1 {
        size <= MAX_INLINE_NODES_SINGLE_USE
    } else {
        size <= MAX_INLINE_NODES_MULTI_USE && uses <= MAX_INLINE_CALL_SITES_MULTI_USE
    }
}

fn lambda_has_control_flow(func: &IrFunc) -> HashMap<LambdaId, bool> {
    let mut out = HashMap::new();
    for (idx, node_id) in func.lambdas.iter().copied().enumerate() {
        let lambda = LambdaId::new(idx as u32);
        let body = match &func.nodes[node_id].kind {
            NodeKind::Lambda { body, .. } => body,
            _ => unreachable!("lambda registry must only contain lambda nodes"),
        };
        out.insert(lambda, region_has_control_flow(func, *body));
    }
    out
}

fn region_has_control_flow(func: &IrFunc, region: RegionId) -> bool {
    let mut stack = vec![region];
    while let Some(rid) = stack.pop() {
        for &nid in &func.regions[rid].nodes {
            match &func.nodes[nid].kind {
                NodeKind::Gamma { regions } => {
                    for &sub in regions {
                        stack.push(sub);
                    }
                    return true;
                }
                NodeKind::Theta { body } => {
                    stack.push(*body);
                    return true;
                }
                _ => {}
            }
        }
    }
    false
}

fn count_apply_call_sites(func: &IrFunc, live_nodes: &HashSet<NodeId>) -> HashMap<LambdaId, usize> {
    let mut out = HashMap::new();
    for (nid, node) in func.nodes.iter() {
        if !live_nodes.contains(&nid) {
            continue;
        }
        if let NodeKind::Apply { target } = node.kind {
            *out.entry(target).or_insert(0) += 1;
        }
    }
    out
}

fn lambda_sizes(func: &IrFunc) -> HashMap<LambdaId, usize> {
    let mut out = HashMap::new();
    for (idx, node_id) in func.lambdas.iter().copied().enumerate() {
        let lambda = LambdaId::new(idx as u32);
        let body = match &func.nodes[node_id].kind {
            NodeKind::Lambda { body, .. } => body,
            _ => unreachable!("lambda registry must only contain lambda nodes"),
        };
        out.insert(lambda, count_region_nodes_recursive(func, *body));
    }
    out
}

fn count_region_nodes_recursive(func: &IrFunc, region: RegionId) -> usize {
    let mut total = 0usize;
    let mut stack = vec![region];
    while let Some(rid) = stack.pop() {
        let reg = &func.regions[rid];
        total += reg.nodes.len();
        for &nid in &reg.nodes {
            match &func.nodes[nid].kind {
                NodeKind::Gamma { regions } => {
                    for &sub in regions {
                        stack.push(sub);
                    }
                }
                NodeKind::Theta { body } => stack.push(*body),
                _ => {}
            }
        }
    }
    total
}

fn build_region_owner_map(func: &IrFunc) -> HashMap<RegionId, LambdaId> {
    let mut out = HashMap::new();
    for (idx, node_id) in func.lambdas.iter().copied().enumerate() {
        let lambda = LambdaId::new(idx as u32);
        let body = match &func.nodes[node_id].kind {
            NodeKind::Lambda { body, .. } => body,
            _ => unreachable!("lambda registry must only contain lambda nodes"),
        };
        collect_region_owners(func, *body, lambda, &mut out);
    }
    out
}

fn collect_live_nodes(func: &IrFunc) -> HashSet<NodeId> {
    let mut live_regions = HashSet::new();
    let mut stack = Vec::new();
    for node_id in func.lambdas.iter().copied() {
        if let NodeKind::Lambda { body, .. } = &func.nodes[node_id].kind {
            stack.push(*body);
        }
    }

    let mut live_nodes = HashSet::new();
    while let Some(region) = stack.pop() {
        if !live_regions.insert(region) {
            continue;
        }
        for &nid in &func.regions[region].nodes {
            live_nodes.insert(nid);
            match &func.nodes[nid].kind {
                NodeKind::Gamma { regions } => {
                    for &sub in regions {
                        stack.push(sub);
                    }
                }
                NodeKind::Theta { body } => stack.push(*body),
                _ => {}
            }
        }
    }
    live_nodes
}

fn collect_region_owners(
    func: &IrFunc,
    start: RegionId,
    owner: LambdaId,
    out: &mut HashMap<RegionId, LambdaId>,
) {
    let mut stack = vec![start];
    while let Some(region) = stack.pop() {
        if out.insert(region, owner).is_some() {
            continue;
        }
        for &nid in &func.regions[region].nodes {
            match &func.nodes[nid].kind {
                NodeKind::Gamma { regions } => {
                    for &sub in regions {
                        stack.push(sub);
                    }
                }
                NodeKind::Theta { body } => stack.push(*body),
                _ => {}
            }
        }
    }
}

struct CloneCtx {
    top_old_region: RegionId,
    top_arg_sources: Vec<PortSource>,
    node_map: HashMap<NodeId, NodeId>,
    region_map: HashMap<RegionId, RegionId>,
    top_new_nodes: Vec<NodeId>,
}

fn remap_source(source: PortSource, ctx: &CloneCtx) -> PortSource {
    match source {
        PortSource::Node(out) => {
            let new_node = *ctx
                .node_map
                .get(&out.node)
                .expect("cloned source node should be available");
            PortSource::Node(OutputRef {
                node: new_node,
                index: out.index,
            })
        }
        PortSource::RegionArg(arg) => {
            if arg.region == ctx.top_old_region {
                ctx.top_arg_sources[arg.index as usize]
            } else {
                let region = *ctx
                    .region_map
                    .get(&arg.region)
                    .expect("cloned source region should be available");
                PortSource::RegionArg(RegionArgRef {
                    region,
                    index: arg.index,
                })
            }
        }
    }
}

fn clone_region_into(
    func: &mut IrFunc,
    old_region: RegionId,
    new_region: RegionId,
    ctx: &mut CloneCtx,
) {
    let is_top = old_region == ctx.top_old_region;
    if !is_top {
        ctx.region_map.insert(old_region, new_region);
    }

    let old_nodes = func.regions[old_region].nodes.clone();

    for old_node in old_nodes {
        let old_inputs = func.nodes[old_node].inputs.clone();
        let old_outputs = func.nodes[old_node].outputs.clone();
        let old_kind = match &func.nodes[old_node].kind {
            NodeKind::Simple(op) => NodeKind::Simple(op.clone()),
            NodeKind::Apply { target } => NodeKind::Apply { target: *target },
            NodeKind::Gamma { regions } => NodeKind::Gamma {
                regions: regions.clone(),
            },
            NodeKind::Theta { body } => NodeKind::Theta { body: *body },
            NodeKind::Lambda { .. } => {
                panic!("lambda nodes cannot appear inside lambda body regions");
            }
        };

        let mut remapped_subregions = Vec::new();
        let kind = match old_kind {
            NodeKind::Gamma { regions } => {
                for old_sub in regions {
                    let old_reg = &func.regions[old_sub];
                    let new_sub = func.regions.push(Region {
                        args: old_reg.args.clone(),
                        results: Vec::new(),
                        nodes: Vec::new(),
                    });
                    clone_region_into(func, old_sub, new_sub, ctx);
                    remapped_subregions.push(new_sub);
                }
                NodeKind::Gamma {
                    regions: remapped_subregions,
                }
            }
            NodeKind::Theta { body } => {
                let old_reg = &func.regions[body];
                let new_body = func.regions.push(Region {
                    args: old_reg.args.clone(),
                    results: Vec::new(),
                    nodes: Vec::new(),
                });
                clone_region_into(func, body, new_body, ctx);
                NodeKind::Theta { body: new_body }
            }
            other => other,
        };

        let inputs: Vec<InputPort> = old_inputs
            .into_iter()
            .map(|inp| InputPort {
                kind: inp.kind,
                source: remap_source(inp.source, ctx),
            })
            .collect();

        let new_node = func.nodes.push(Node {
            region: new_region,
            inputs,
            outputs: old_outputs,
            kind,
        });
        ctx.node_map.insert(old_node, new_node);

        if is_top {
            ctx.top_new_nodes.push(new_node);
        } else {
            func.regions[new_region].nodes.push(new_node);
        }
    }

    if !is_top {
        let old_results = func.regions[old_region].results.clone();
        func.regions[new_region].results = old_results
            .into_iter()
            .map(|result| RegionResult {
                kind: result.kind,
                source: remap_source(result.source, ctx),
            })
            .collect();
    }
}

fn replace_output_use(
    func: &mut IrFunc,
    use_lists: &mut UseLists,
    from: OutputRef,
    to: PortSource,
) {
    use_lists.replace_output_use(func, from, to);
}

fn inline_one_apply(func: &mut IrFunc, use_lists: &mut UseLists, apply: NodeId) -> bool {
    let (caller_region, target, top_arg_sources, output_count) = {
        let node = &func.nodes[apply];
        let NodeKind::Apply { target } = node.kind else {
            return false;
        };
        let args: Vec<PortSource> = node.inputs.iter().map(|inp| inp.source).collect();
        (node.region, target, args, node.outputs.len())
    };

    let callee_body = func.lambda_body(target);
    if top_arg_sources.len() != func.regions[callee_body].args.len() {
        panic!(
            "apply argument count mismatch for lambda @{}: got {}, expected {}",
            target.index(),
            top_arg_sources.len(),
            func.regions[callee_body].args.len()
        );
    }

    let mut ctx = CloneCtx {
        top_old_region: callee_body,
        top_arg_sources,
        node_map: HashMap::new(),
        region_map: HashMap::new(),
        top_new_nodes: Vec::new(),
    };
    clone_region_into(func, callee_body, caller_region, &mut ctx);

    let mapped_results: Vec<PortSource> = func.regions[callee_body]
        .results
        .iter()
        .map(|result| remap_source(result.source, &ctx))
        .collect();
    if mapped_results.len() != output_count {
        panic!(
            "apply output count mismatch for lambda @{}: got {}, expected {}",
            target.index(),
            output_count,
            mapped_results.len()
        );
    }

    for (idx, source) in mapped_results.into_iter().enumerate() {
        replace_output_use(
            func,
            use_lists,
            OutputRef {
                node: apply,
                index: idx as u16,
            },
            source,
        );
    }

    let Some(pos) = func.regions[caller_region]
        .nodes
        .iter()
        .position(|&n| n == apply)
    else {
        return false;
    };
    let mut new_nodes = Vec::with_capacity(
        func.regions[caller_region].nodes.len() + ctx.top_new_nodes.len().saturating_sub(1),
    );
    new_nodes.extend_from_slice(&func.regions[caller_region].nodes[..pos]);
    new_nodes.extend(ctx.top_new_nodes);
    new_nodes.extend_from_slice(&func.regions[caller_region].nodes[pos + 1..]);
    func.regions[caller_region].nodes = new_nodes;

    true
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{IrBuilder, IrOp, NodeKind, Width};

    #[test]
    fn inlines_simple_apply() {
        let mut builder = IrBuilder::new(<u32 as facet::Facet>::SHAPE);
        let child = builder.create_lambda(<u8 as facet::Facet>::SHAPE);
        {
            let mut rb = builder.lambda_region(child);
            rb.bounds_check(1);
            let b = rb.read_bytes(1);
            rb.write_to_field(b, 0, Width::W1);
            rb.set_results(&[]);
        }
        {
            let mut rb = builder.root_region();
            let _ = rb.apply(child, &[], 0);
            rb.set_results(&[]);
        }
        let mut func = builder.finish();

        let before_live = collect_live_nodes(&func);
        let before_apply = func
            .nodes
            .iter()
            .filter(|(id, n)| before_live.contains(id) && matches!(n.kind, NodeKind::Apply { .. }))
            .count();
        assert_eq!(before_apply, 1);

        run_default_passes(&mut func);

        let after_live = collect_live_nodes(&func);
        let after_apply = func
            .nodes
            .iter()
            .filter(|(id, n)| after_live.contains(id) && matches!(n.kind, NodeKind::Apply { .. }))
            .count();
        assert_eq!(after_apply, 0);
    }

    #[test]
    fn does_not_inline_recursive_backedge() {
        let mut builder = IrBuilder::new(<u8 as facet::Facet>::SHAPE);
        {
            let mut rb = builder.root_region();
            let _ = rb.apply(LambdaId::new(0), &[], 0);
            rb.set_results(&[]);
        }
        let mut func = builder.finish();

        run_default_passes(&mut func);

        let live = collect_live_nodes(&func);
        let apply_count = func
            .nodes
            .iter()
            .filter(|(id, n)| live.contains(id) && matches!(n.kind, NodeKind::Apply { .. }))
            .count();
        assert_eq!(apply_count, 1);
    }

    #[test]
    fn replace_output_use_rewrites_region_results() {
        let mut builder = IrBuilder::new(<u8 as facet::Facet>::SHAPE);
        let expected_output = {
            let mut rb = builder.root_region();
            let old = rb.const_val(1);
            let new = rb.const_val(2);
            rb.set_results(&[old]);
            match new {
                PortSource::Node(out) => out,
                PortSource::RegionArg(_) => panic!("const output should be node output"),
            }
        };
        let mut func = builder.finish();
        let root = func.root_body();
        let old_result = match func.regions[root].results[0].source {
            PortSource::Node(out) => out,
            PortSource::RegionArg(_) => panic!("root result should be node output"),
        };
        let new_output = expected_output;

        let mut use_lists = UseLists::build(&func);
        replace_output_use(
            &mut func,
            &mut use_lists,
            old_result,
            PortSource::Node(new_output),
        );

        assert_eq!(
            func.regions[root].results[0].source,
            PortSource::Node(new_output)
        );
    }

    #[test]
    fn dce_removes_dead_pure_nodes_after_inlining() {
        let mut builder = IrBuilder::new(<u8 as facet::Facet>::SHAPE);
        let child = builder.create_lambda(<u8 as facet::Facet>::SHAPE);
        {
            let mut rb = builder.lambda_region(child);
            let one = rb.const_val(1);
            let two = rb.const_val(2);
            let sum = rb.binop(IrOp::Add, one, two);
            rb.set_results(&[sum]);
        }
        {
            let mut rb = builder.root_region();
            let _unused = rb.apply(child, &[], 1);
            rb.set_results(&[]);
        }
        let mut func = builder.finish();
        let root = func.root_body();

        run_default_passes(&mut func);

        assert!(
            func.regions[root].nodes.is_empty(),
            "dead pure inline residue should be removed from root region"
        );
    }

    // r[verify ir.passes.pre-regalloc.loop-invariants]
    #[test]
    fn hoists_theta_invariant_setup_consts_out_of_body() {
        let mut builder = IrBuilder::new(<u8 as facet::Facet>::SHAPE);
        {
            let mut rb = builder.root_region();
            let init_count = rb.const_val(3);
            let one = rb.const_val(1);
            let _ = rb.theta(&[init_count, one], |bb| {
                let args = bb.region_args(2);
                let counter = args[0];
                let one = args[1];
                let invariant_setup = bb.const_val(7);
                let _ = bb.binop(IrOp::Add, invariant_setup, invariant_setup);
                let next = bb.binop(IrOp::Sub, counter, one);
                bb.set_results(&[next, next, one]);
            });
            rb.set_results(&[]);
        }
        let mut func = builder.finish();
        let root = func.root_body();
        let theta_before = func.regions[root]
            .nodes
            .iter()
            .copied()
            .find(|&nid| matches!(func.nodes[nid].kind, NodeKind::Theta { .. }))
            .expect("expected theta node");
        let body_before = match &func.nodes[theta_before].kind {
            NodeKind::Theta { body } => *body,
            _ => unreachable!("expected theta node"),
        };
        let body_const_before = func.regions[body_before]
            .nodes
            .iter()
            .filter(|&&nid| matches!(func.nodes[nid].kind, NodeKind::Simple(IrOp::Const { .. })))
            .count();
        assert!(
            body_const_before >= 1,
            "expected invariant setup const in theta body before pass"
        );

        hoist_theta_loop_invariant_setup_pass(&mut func);

        let theta_after = func.regions[root]
            .nodes
            .iter()
            .copied()
            .find(|&nid| matches!(func.nodes[nid].kind, NodeKind::Theta { .. }))
            .expect("expected theta node");
        let body_after = match &func.nodes[theta_after].kind {
            NodeKind::Theta { body } => *body,
            _ => unreachable!("expected theta node"),
        };
        let body_const_after = func.regions[body_after]
            .nodes
            .iter()
            .filter(|&&nid| matches!(func.nodes[nid].kind, NodeKind::Simple(IrOp::Const { .. })))
            .count();
        assert_eq!(
            body_const_after, 0,
            "expected invariant setup consts to be hoisted out of theta body"
        );

        let theta_pos = func.regions[root]
            .nodes
            .iter()
            .position(|&nid| nid == theta_after)
            .expect("theta should stay in root region");
        let hoisted_consts = func.regions[root].nodes[..theta_pos]
            .iter()
            .filter(|&&nid| matches!(func.nodes[nid].kind, NodeKind::Simple(IrOp::Const { .. })))
            .count();
        assert!(
            hoisted_consts >= 1,
            "expected invariant setup const hoisted before theta"
        );
    }

    // r[verify ir.passes.pre-regalloc.loop-invariants]
    #[test]
    fn hoists_theta_invariant_expression_tree_out_of_body() {
        let is_invariant_tree_node = |func: &IrFunc, nid: NodeId| match &func.nodes[nid].kind {
            NodeKind::Simple(IrOp::Const { .. }) | NodeKind::Simple(IrOp::Xor) => true,
            NodeKind::Simple(IrOp::Add) => {
                func.nodes[nid].inputs.iter().all(|inp| match inp.source {
                    PortSource::Node(out) => matches!(
                        func.nodes[out.node].kind,
                        NodeKind::Simple(IrOp::Const { .. })
                            | NodeKind::Simple(IrOp::Add)
                            | NodeKind::Simple(IrOp::Xor)
                    ),
                    PortSource::RegionArg(_) => false,
                })
            }
            _ => false,
        };

        let mut builder = IrBuilder::new(<u8 as facet::Facet>::SHAPE);
        {
            let mut rb = builder.root_region();
            let init_count = rb.const_val(4);
            let one = rb.const_val(1);
            let _ = rb.theta(&[init_count, one], |bb| {
                let args = bb.region_args(2);
                let counter = args[0];
                let one = args[1];

                let c7 = bb.const_val(7);
                let c3 = bb.const_val(3);
                let s = bb.binop(IrOp::Add, c7, c3);
                let x = bb.binop(IrOp::Xor, s, c3);
                let next = bb.binop(IrOp::Sub, counter, one);
                let _ = bb.binop(IrOp::Add, x, next);

                bb.set_results(&[next, next, one]);
            });
            rb.set_results(&[]);
        }
        let mut func = builder.finish();
        let root = func.root_body();
        let theta_before = func.regions[root]
            .nodes
            .iter()
            .copied()
            .find(|&nid| matches!(func.nodes[nid].kind, NodeKind::Theta { .. }))
            .expect("expected theta node");
        let body_before = match &func.nodes[theta_before].kind {
            NodeKind::Theta { body } => *body,
            _ => unreachable!("expected theta node"),
        };
        let invariant_tree_nodes_before = func.regions[body_before]
            .nodes
            .iter()
            .filter(|&&nid| is_invariant_tree_node(&func, nid))
            .count();
        assert!(
            invariant_tree_nodes_before >= 4,
            "expected invariant expression tree in theta body before pass"
        );

        run_default_passes(&mut func);

        let theta_after = func.regions[root]
            .nodes
            .iter()
            .copied()
            .find(|&nid| matches!(func.nodes[nid].kind, NodeKind::Theta { .. }))
            .expect("expected theta node");
        let body_after = match &func.nodes[theta_after].kind {
            NodeKind::Theta { body } => *body,
            _ => unreachable!("expected theta node"),
        };
        let invariant_tree_nodes_after = func.regions[body_after]
            .nodes
            .iter()
            .filter(|&&nid| is_invariant_tree_node(&func, nid))
            .count();
        assert_eq!(
            invariant_tree_nodes_after, 0,
            "expected invariant expression tree nodes to be hoisted out of theta body"
        );
    }

    // r[verify ir.passes.pre-regalloc.loop-invariants]
    #[test]
    fn hoists_theta_invariants_from_builder_fixture() {
        let is_invariant_tree_node = |func: &IrFunc, nid: NodeId| match &func.nodes[nid].kind {
            NodeKind::Simple(IrOp::Const { .. }) | NodeKind::Simple(IrOp::Xor) => true,
            NodeKind::Simple(IrOp::Add) => {
                func.nodes[nid].inputs.iter().all(|inp| match inp.source {
                    PortSource::Node(out) => matches!(
                        func.nodes[out.node].kind,
                        NodeKind::Simple(IrOp::Const { .. })
                            | NodeKind::Simple(IrOp::Add)
                            | NodeKind::Simple(IrOp::Xor)
                    ),
                    PortSource::RegionArg(_) => false,
                })
            }
            _ => false,
        };

        let mut builder = IrBuilder::new(<u8 as facet::Facet>::SHAPE);
        {
            let mut rb = builder.root_region();
            let init_count = rb.const_val(4);
            let one = rb.const_val(1);
            let _ = rb.theta(&[init_count, one], |bb| {
                let args = bb.region_args(2);
                let counter = args[0];
                let one = args[1];
                let c7 = bb.const_val(7);
                let c3 = bb.const_val(3);
                let s = bb.binop(IrOp::Add, c7, c3);
                let x = bb.binop(IrOp::Xor, s, c3);
                let next = bb.binop(IrOp::Sub, counter, one);
                let _ = bb.binop(IrOp::Add, x, next);
                bb.set_results(&[next, next, one]);
            });
            rb.set_results(&[]);
        }
        let mut func = builder.finish();
        let root = func.root_body();
        let theta_before = func.regions[root]
            .nodes
            .iter()
            .copied()
            .find(|&nid| matches!(func.nodes[nid].kind, NodeKind::Theta { .. }))
            .expect("expected theta node");
        let body_before = match &func.nodes[theta_before].kind {
            NodeKind::Theta { body } => *body,
            _ => unreachable!("expected theta node"),
        };
        let invariant_tree_nodes_before = func.regions[body_before]
            .nodes
            .iter()
            .filter(|&&nid| is_invariant_tree_node(&func, nid))
            .count();

        run_default_passes(&mut func);

        let theta_after = func.regions[root]
            .nodes
            .iter()
            .copied()
            .find(|&nid| matches!(func.nodes[nid].kind, NodeKind::Theta { .. }))
            .expect("expected theta node");
        let body_after = match &func.nodes[theta_after].kind {
            NodeKind::Theta { body } => *body,
            _ => unreachable!("expected theta node"),
        };
        let invariant_tree_nodes_after = func.regions[body_after]
            .nodes
            .iter()
            .filter(|&&nid| is_invariant_tree_node(&func, nid))
            .count();
        assert!(
            invariant_tree_nodes_after < invariant_tree_nodes_before,
            "expected fewer invariant setup tree nodes in theta body after pass (before={}, after={})",
            invariant_tree_nodes_before,
            invariant_tree_nodes_after
        );
    }
}
