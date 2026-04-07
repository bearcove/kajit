use std::collections::{HashMap, HashSet};

use crate::{
    InputPort, IrFunc, IrOp, LambdaId, Node, NodeId, NodeKind, OutputPort, OutputRef, PortKind,
    PortSource, Region, RegionArg, RegionArgRef, RegionId, RegionResult, ResultId, verify,
};

const MAX_INLINE_NODES_SINGLE_USE: usize = 256;
const MAX_INLINE_NODES_MULTI_USE: usize = 64;
const MAX_INLINE_CALL_SITES_MULTI_USE: usize = 4;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum UseSite {
    NodeInput { node: NodeId, input_index: u16 },
    RegionResult { result: ResultId },
}

#[derive(Default)]
struct UseLists {
    by_output: HashMap<OutputRef, Vec<UseSite>>,
}

impl UseLists {
    fn build(func: &IrFunc) -> Self {
        let mut out = Self::default();

        for (_region_id, region) in func.regions.iter() {
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

            for &result_id in &region.results {
                let result = &func.region_results[result_id];
                let PortSource::Node(source) = result.source else {
                    continue;
                };
                out.by_output
                    .entry(source)
                    .or_default()
                    .push(UseSite::RegionResult { result: result_id });
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
                UseSite::RegionResult { result: result_id } => {
                    let result = &mut func.region_results[result_id];
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
    run_default_passes_with_filter(func, |_| true);
}

pub fn run_default_passes_with_filter<F>(func: &mut IrFunc, mut enabled: F)
where
    F: FnMut(DefaultPassSpec) -> bool,
{
    for pass in default_pass_registry() {
        if enabled(*pass) {
            pass.run(func);
        }
    }
}

#[cfg(debug_assertions)]
fn debug_verify(func: &IrFunc, pass_name: &str) {
    if let Err(err) = verify(func) {
        panic!("IR verification failed after {pass_name}: {err}");
    }
}

#[cfg(not(debug_assertions))]
fn debug_verify(_func: &IrFunc, _pass_name: &str) {}

#[derive(Debug, Clone, Copy)]
pub struct DefaultPassSpec {
    pub name: &'static str,
    pub description: &'static str,
    run: fn(&mut IrFunc),
}

impl DefaultPassSpec {
    pub fn run(self, func: &mut IrFunc) {
        (self.run)(func);
    }
}

fn run_hoist_theta_loop_invariant_setup_pass(func: &mut IrFunc) {
    hoist_theta_loop_invariant_setup_pass(func);
    debug_verify(func, "hoist_theta_loop_invariant_setup_pass");
}

fn run_inline_apply_pass(func: &mut IrFunc) {
    inline_apply_pass(func);
    debug_verify(func, "inline_apply_pass");
}

fn run_dead_code_elimination_pass(func: &mut IrFunc) {
    dead_code_elimination_pass(func);
    debug_verify(func, "dead_code_elimination_pass");
}

fn run_unroll_bounded_thetas_pass(func: &mut IrFunc) {
    crate::unroll_theta::unroll_bounded_thetas(func);
    debug_verify(func, "unroll_bounded_thetas");
}

fn run_post_unroll_canonicalize_pass(func: &mut IrFunc) {
    crate::post_unroll_canonicalize::post_unroll_canonicalize(func);
    debug_verify(func, "post_unroll_canonicalize");
}

fn run_const_fold_pass(func: &mut IrFunc) {
    crate::const_fold::const_fold(func);
    debug_verify(func, "const_fold");
}

fn run_post_unroll_simplify_pass(func: &mut IrFunc) {
    crate::simplify_gamma::simplify_trivial_gammas(func);
    debug_verify(func, "post_unroll_simplify");
}

fn run_gamma_output_partition_pass(func: &mut IrFunc) {
    crate::gamma_output_partition::gamma_output_partition(func);
    debug_verify(func, "gamma_output_partition");
}

fn run_simplify_trivial_gammas_pass(func: &mut IrFunc) {
    crate::simplify_gamma::simplify_trivial_gammas(func);
    debug_verify(func, "simplify_trivial_gammas");
}

fn run_dead_theta_ports_pass(func: &mut IrFunc) {
    let removed = crate::dead_theta_ports::eliminate_dead_theta_ports(func);
    if removed > 0 && std::env::var("KAJIT_DEBUG_DEAD_THETA").is_ok() {
        eprintln!("[dead_theta_ports] removed {} dead ports total", removed);
    }
    debug_verify(func, "dead_theta_ports");
}

fn run_slot_to_reg_pass(func: &mut IrFunc) {
    if std::env::var("DUMP_SLOT2REG").is_ok() {
        let registry = crate::IntrinsicRegistry::empty();
        eprintln!(
            "=== BEFORE slot2reg ===\n{}\n=== END BEFORE ===",
            func.display_with_registry(&registry)
        );
    }
    crate::slot2reg::slot_to_reg(func);
    if std::env::var("DUMP_SLOT2REG").is_ok() {
        let registry = crate::IntrinsicRegistry::empty();
        eprintln!(
            "=== AFTER slot2reg ===\n{}\n=== END AFTER ===",
            func.display_with_registry(&registry)
        );
    }
    debug_verify(func, "slot_to_reg");
}

const DEFAULT_PASS_REGISTRY: [DefaultPassSpec; 11] = [
    DefaultPassSpec {
        name: "slot_to_reg",
        description: "Promote stack slots to RVSDG data flow (passthrough/loop-vars).",
        run: run_slot_to_reg_pass,
    },
    DefaultPassSpec {
        name: "dead_theta_ports",
        description: "Eliminate loop-carried theta ports that are always re-initialized to a constant.",
        run: run_dead_theta_ports_pass,
    },
    DefaultPassSpec {
        name: "simplify_trivial_gammas",
        description: "Remove gamma nodes where all branches produce identical passthrough results.",
        run: run_simplify_trivial_gammas_pass,
    },
    DefaultPassSpec {
        name: "theta_loop_invariant_hoist",
        description: "Hoist loop-invariant setup out of theta loop bodies when safe.",
        run: run_hoist_theta_loop_invariant_setup_pass,
    },
    DefaultPassSpec {
        name: "unroll_bounded_thetas",
        description: "Unroll theta loops with known max iteration count into gamma cascades.",
        run: run_unroll_bounded_thetas_pass,
    },
    DefaultPassSpec {
        name: "post_unroll_canonicalize",
        description: "Environment-driven constant folding and gamma simplification after unrolling.",
        run: run_post_unroll_canonicalize_pass,
    },
    DefaultPassSpec {
        name: "const_fold",
        description: "Evaluate pure operations with all-constant inputs (propagates unrolled iteration indices).",
        run: run_const_fold_pass,
    },
    DefaultPassSpec {
        name: "post_unroll_simplify",
        description: "Re-run gamma simplification after constant folding to collapse constant-predicate gammas.",
        run: run_post_unroll_simplify_pass,
    },
    DefaultPassSpec {
        name: "gamma_output_partition",
        description: "Eliminate gamma outputs that are identical on all branches.",
        run: run_gamma_output_partition_pass,
    },
    DefaultPassSpec {
        name: "inline_apply",
        description: "Inline apply/lambda calls under size and use-site thresholds.",
        run: run_inline_apply_pass,
    },
    DefaultPassSpec {
        name: "dead_code_elimination",
        description: "Remove dead nodes and unreachable regions after shaping passes.",
        run: run_dead_code_elimination_pass,
    },
];

pub fn default_pass_registry() -> &'static [DefaultPassSpec] {
    &DEFAULT_PASS_REGISTRY
}

// r[impl ir.passes.pre-regalloc.loop-invariants]
fn hoist_theta_loop_invariant_setup_pass(func: &mut IrFunc) {
    // First pass: hoist complete subtrees (existing logic)
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
                // Verify after each change
                #[cfg(debug_assertions)]
                if let Err(err) = verify(func) {
                    panic!(
                        "IR verification failed after hoist_theta_loop_invariants_for_node: {err}"
                    );
                }
                break;
            }
        }

        if !changed {
            break;
        }
    }

    // Second pass: hoist remaining constants by threading as theta arguments
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
            if hoist_theta_constants_as_args(func, theta) {
                changed = true;
                #[cfg(debug_assertions)]
                if let Err(err) = verify(func) {
                    panic!("IR verification failed after hoist_theta_constants_as_args: {err}");
                }
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
        let NodeKind::Theta { body, .. } = &node.kind else {
            return false;
        };
        (node.region, *body)
    };

    let body_nodes = func.regions[body_region].nodes.clone();
    let body_node_set: HashSet<NodeId> = body_nodes.iter().copied().collect();
    let mut hoistable_set = HashSet::new();

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
        }
    }

    if hoistable_set.is_empty() {
        return false;
    }

    loop {
        let mut removed_any = false;
        let current = hoistable_set.clone();
        for node_id in current.iter().copied() {
            let inputs_invariant = func.nodes[node_id].inputs.iter().all(|inp| {
                source_is_theta_invariant(inp.source, body_region, &body_node_set, &hoistable_set)
            });
            let uses_compatible = theta_hoist_uses_stay_within_body_subset(
                func,
                use_lists,
                body_region,
                node_id,
                &hoistable_set,
            );
            if !inputs_invariant || !uses_compatible {
                hoistable_set.remove(&node_id);
                removed_any = true;
            }
        }
        if !removed_any {
            break;
        }
    }

    if hoistable_set.is_empty() {
        return false;
    }

    let hoistable_nodes: Vec<NodeId> = body_nodes
        .iter()
        .copied()
        .filter(|nid| hoistable_set.contains(nid))
        .collect();

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
            debug_scope: func.nodes[old_node].debug_scope,
            debug_value: func.nodes[old_node].debug_value,
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

/// Hoist constants from a theta body by threading them as arguments.
///
/// Unlike the main hoisting logic, this doesn't require all uses to be
/// hoistable. Instead, it creates the constant outside the theta, passes it
/// in as an argument, and rewrites uses inside the body to reference the
/// new region arg.
fn hoist_theta_constants_as_args(func: &mut IrFunc, theta: NodeId) -> bool {
    let (parent_region, body_region, theta_debug_scope) = {
        let node = &func.nodes[theta];
        let NodeKind::Theta { body, .. } = &node.kind else {
            return false;
        };
        (node.region, *body, node.debug_scope)
    };

    // Find constants in the body
    let body_nodes = func.regions[body_region].nodes.clone();
    let const_nodes: Vec<NodeId> = body_nodes
        .iter()
        .copied()
        .filter(|&nid| matches!(func.nodes[nid].kind, NodeKind::Simple(IrOp::Const { .. })))
        .collect();

    if const_nodes.is_empty() {
        return false;
    }

    // Find the theta's position in the parent region
    let Some(theta_pos) = func.regions[parent_region]
        .nodes
        .iter()
        .position(|&n| n == theta)
    else {
        return false;
    };

    // Hoist one constant at a time (loop will iterate until all are done)
    let old_const = const_nodes[0];
    let const_value = {
        let NodeKind::Simple(IrOp::Const { value }) = &func.nodes[old_const].kind else {
            return false;
        };
        *value
    };

    // Create the constant in the parent region (before the theta)
    let new_const = func.nodes.push(Node {
        region: parent_region,
        debug_scope: func.nodes[old_const].debug_scope,
        debug_value: func.nodes[old_const].debug_value,
        inputs: vec![],
        outputs: vec![OutputPort {
            kind: PortKind::Data,
            vreg: None,
            debug_scope: func.nodes[old_const].debug_scope,
        }],
        kind: NodeKind::Simple(IrOp::Const { value: const_value }),
    });
    func.regions[parent_region]
        .nodes
        .insert(theta_pos, new_const);

    // All theta-related structures have layout: [loop_vars..., state_domains...]
    // (Body results also have a predicate prefix: [predicate, loop_vars..., state_domains...])
    // We must insert the new hoisted constant as a loop_var, BEFORE the state domains.
    let state_count = 1;

    // Insert theta input before state domains
    let input_insert_pos = func.nodes[theta].inputs.len() - state_count;
    let new_input = InputPort {
        kind: PortKind::Data,
        source: PortSource::Node(OutputRef {
            node: new_const,
            index: 0,
        }),
    };
    func.nodes[theta].inputs.insert(input_insert_pos, new_input);

    // Insert body arg before state domains
    let arg_insert_pos = func.regions[body_region].args.len() - state_count;
    let new_arg_id = func.region_args.push(RegionArg {
        kind: PortKind::Data,
        vreg: None,
        debug_value: func.nodes[old_const].debug_value,
    });
    func.regions[body_region]
        .args
        .insert(arg_insert_pos, new_arg_id);

    // With stable ArgIds, no shifting of RegionArgRef is needed!
    // The new ArgId is a stable handle that won't be affected by insertions.

    // Rewrite all uses of the old constant to use the new region arg
    let new_source = PortSource::RegionArg(RegionArgRef {
        region: body_region,
        arg: new_arg_id,
    });
    let use_lists = UseLists::build(func);
    let old_output = OutputRef {
        node: old_const,
        index: 0,
    };

    let mut use_lists = use_lists;
    replace_output_use(func, &mut use_lists, old_output, new_source);

    // Remove the old constant from the body
    func.regions[body_region]
        .nodes
        .retain(|&nid| nid != old_const);

    // Insert body result before state domains (accounting for predicate prefix)
    // Body results: [predicate, loop_vars..., state_domains...]
    let result_insert_pos = func.regions[body_region].results.len() - state_count;
    let new_result_id = func.region_results.push(RegionResult {
        kind: PortKind::Data,
        source: new_source,
    });
    func.regions[body_region]
        .results
        .insert(result_insert_pos, new_result_id);

    // Insert theta output before state domains
    let output_insert_pos = func.nodes[theta].outputs.len() - state_count;
    func.nodes[theta].outputs.insert(
        output_insert_pos,
        OutputPort {
            kind: PortKind::Data,
            vreg: None,
            debug_scope: theta_debug_scope,
        },
    );

    // After inserting a new theta output, we need to shift OutputRef indices
    // in the parent region's results that reference the theta's outputs.
    // Any result referencing theta output at index >= output_insert_pos needs
    // its index incremented by 1.
    for &result_id in &func.regions[parent_region].results {
        let result = &mut func.region_results[result_id];
        if let PortSource::Node(ref mut out) = result.source
            && out.node == theta
            && out.index >= output_insert_pos as u16
        {
            out.index += 1;
        }
    }
    // Also update any node inputs in the parent region that reference the theta
    for &node_id in &func.regions[parent_region].nodes {
        for input in func.nodes[node_id].inputs.iter_mut() {
            if let PortSource::Node(ref mut out) = input.source
                && out.node == theta
                && out.index >= output_insert_pos as u16
            {
                out.index += 1;
            }
        }
    }

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

fn theta_hoist_uses_stay_within_body_subset(
    func: &IrFunc,
    use_lists: &UseLists,
    body_region: RegionId,
    node_id: NodeId,
    hoistable_set: &HashSet<NodeId>,
) -> bool {
    let output_len = func.nodes[node_id].outputs.len();
    (0..output_len).all(|out_idx| {
        let out = OutputRef {
            node: node_id,
            index: out_idx as u16,
        };
        let Some(use_sites) = use_lists.by_output.get(&out) else {
            return true;
        };
        use_sites.iter().all(|use_site| match *use_site {
            UseSite::NodeInput { node, .. } => {
                func.nodes[node].region != body_region || hoistable_set.contains(&node)
            }
            UseSite::RegionResult { result } => {
                // Check if this result belongs to the body region
                !func.regions[body_region].results.contains(&result)
            }
        })
    })
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
                NodeKind::Theta { body, .. } => {
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
                NodeKind::Theta { body, .. } => stack.push(*body),
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
                NodeKind::Theta { body, .. } => stack.push(*body),
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
                NodeKind::Theta { body, .. } => stack.push(*body),
                _ => {}
            }
        }
    }
}

pub(crate) struct CloneCtx {
    pub(crate) top_old_region: RegionId,
    pub(crate) top_arg_sources: Vec<PortSource>,
    pub(crate) node_map: HashMap<NodeId, NodeId>,
    pub(crate) region_map: HashMap<RegionId, RegionId>,
    pub(crate) arg_map: HashMap<crate::ArgId, crate::ArgId>,
    pub(crate) top_new_nodes: Vec<NodeId>,
}

pub(crate) fn remap_source(source: PortSource, ctx: &CloneCtx, func: &IrFunc) -> PortSource {
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
                // Find position of this arg in the old region's args
                let pos = func.regions[ctx.top_old_region]
                    .args
                    .iter()
                    .position(|&id| id == arg.arg)
                    .expect("arg should exist in region");
                ctx.top_arg_sources[pos]
            } else {
                let region = *ctx
                    .region_map
                    .get(&arg.region)
                    .expect("cloned source region should be available");
                let new_arg = *ctx
                    .arg_map
                    .get(&arg.arg)
                    .expect("cloned arg should be available");
                PortSource::RegionArg(RegionArgRef {
                    region,
                    arg: new_arg,
                })
            }
        }
    }
}

pub(crate) fn clone_region_into(
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
            NodeKind::Theta {
                body,
                max_iterations,
                ..
            } => NodeKind::Theta {
                body: *body,
                max_iterations: *max_iterations,
            },
            NodeKind::Lambda { .. } => {
                panic!("lambda nodes cannot appear inside lambda body regions");
            }
        };

        let mut remapped_subregions = Vec::new();
        let kind = match old_kind {
            NodeKind::Gamma { regions } => {
                for old_sub in regions {
                    let old_reg = &func.regions[old_sub];
                    // Clone args, creating new ArgIds
                    let mut new_args = Vec::with_capacity(old_reg.args.len());
                    for &old_arg_id in &old_reg.args {
                        let old_arg = &func.region_args[old_arg_id];
                        let mut cloned_arg = old_arg.clone();
                        cloned_arg.vreg = None; // fresh vreg from assign_vregs
                        let new_arg_id = func.region_args.push(cloned_arg);
                        ctx.arg_map.insert(old_arg_id, new_arg_id);
                        new_args.push(new_arg_id);
                    }
                    let new_sub = func.regions.push(Region {
                        debug_scope: old_reg.debug_scope,
                        args: new_args,
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
            NodeKind::Theta {
                body,
                max_iterations,
                ..
            } => {
                let old_reg = &func.regions[body];
                // Clone args, creating new ArgIds
                let mut new_args = Vec::with_capacity(old_reg.args.len());
                for &old_arg_id in &old_reg.args {
                    let old_arg = &func.region_args[old_arg_id];
                    let new_arg_id = func.region_args.push(old_arg.clone());
                    ctx.arg_map.insert(old_arg_id, new_arg_id);
                    new_args.push(new_arg_id);
                }
                let new_body = func.regions.push(Region {
                    debug_scope: old_reg.debug_scope,
                    args: new_args,
                    results: Vec::new(),
                    nodes: Vec::new(),
                });
                clone_region_into(func, body, new_body, ctx);
                NodeKind::Theta {
                    body: new_body,
                    max_iterations,
                }
            }
            other => other,
        };

        let inputs: Vec<InputPort> = old_inputs
            .into_iter()
            .map(|inp| InputPort {
                kind: inp.kind,
                source: remap_source(inp.source, ctx, func),
            })
            .collect();

        // Clear vreg assignments on cloned outputs — assign_vregs will
        // give them fresh vregs during linearization. Without this, cloned
        // nodes from theta unrolling would reuse the original's vregs,
        // violating SSA (multiple defs for the same vreg).
        let outputs: Vec<_> = old_outputs
            .into_iter()
            .map(|mut out| {
                out.vreg = None;
                out
            })
            .collect();

        let new_node = func.nodes.push(Node {
            region: new_region,
            debug_scope: func.nodes[old_node].debug_scope,
            debug_value: func.nodes[old_node].debug_value,
            inputs,
            outputs,
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
        let new_results: Vec<_> = old_results
            .into_iter()
            .map(|result_id| {
                let old_result = &func.region_results[result_id];

                func.region_results.push(RegionResult {
                    kind: old_result.kind,
                    source: remap_source(old_result.source, ctx, func),
                })
            })
            .collect();
        func.regions[new_region].results = new_results;
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
        arg_map: HashMap::new(),
        top_new_nodes: Vec::new(),
    };
    clone_region_into(func, callee_body, caller_region, &mut ctx);

    let mapped_results: Vec<PortSource> = func.regions[callee_body]
        .results
        .iter()
        .map(|&result_id| {
            let result = &func.region_results[result_id];
            remap_source(result.source, &ctx, func)
        })
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
        let mut builder = IrBuilder::new("u32", 0);
        let child = builder.create_lambda("u8", 0);
        {
            let mut rb = builder.lambda_region(child);
            let b = rb.const_val(0);
            let addr = rb.const_val(0u64);
            rb.store_to_addr(addr, b, Width::W1);
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
        let mut builder = IrBuilder::new("u8", 0);
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
        let mut builder = IrBuilder::new("u8", 0);
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
        let result_id = func.regions[root].results[0];
        let old_result = match func.region_results[result_id].source {
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

        let result_id = func.regions[root].results[0];
        assert_eq!(
            func.region_results[result_id].source,
            PortSource::Node(new_output)
        );
    }

    #[test]
    fn dce_removes_dead_pure_nodes_after_inlining() {
        let mut builder = IrBuilder::new("u8", 0);
        let child = builder.create_lambda("u8", 0);
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
        let mut builder = IrBuilder::new("u8", 0);
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
            NodeKind::Theta { body, .. } => *body,
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
            NodeKind::Theta { body, .. } => *body,
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

        let mut builder = IrBuilder::new("u8", 0);
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
            NodeKind::Theta { body, .. } => *body,
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
            NodeKind::Theta { body, .. } => *body,
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

        let mut builder = IrBuilder::new("u8", 0);
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
            NodeKind::Theta { body, .. } => *body,
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
            NodeKind::Theta { body, .. } => *body,
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

    #[test]
    fn theta_hoist_threads_consts_as_args_when_used_by_non_hoistable_ops() {
        // This test verifies that constants used by non-hoistable ops are still
        // hoisted, but via threading as theta arguments rather than being left
        // inside the body.
        let mut builder = IrBuilder::new("u8", 0);
        {
            let mut rb = builder.root_region();
            let init_acc = rb.const_val(0);
            let init_shift = rb.const_val(7);
            let one = rb.const_val(1);
            let mask = rb.const_val(0x7f);
            let _ = rb.theta(&[init_acc, init_shift, one], |bb| {
                let args = bb.region_args(3);
                let acc = args[0];
                let shift = args[1];
                let one = args[2];
                let value = bb.const_val(3);
                let masked = bb.binop(IrOp::And, value, mask);
                let shifted = bb.binop(IrOp::Shl, masked, shift);
                let next_acc = bb.binop(IrOp::Or, acc, shifted);
                let next_shift = bb.binop(IrOp::Add, shift, one);
                bb.set_results(&[next_acc, next_acc, next_shift, one]);
            });
            rb.set_results(&[]);
        }
        let mut func = builder.finish();
        let root = func.root_body();
        let theta = func.regions[root]
            .nodes
            .iter()
            .copied()
            .find(|&nid| matches!(func.nodes[nid].kind, NodeKind::Theta { .. }))
            .expect("expected theta node");
        let body = match &func.nodes[theta].kind {
            NodeKind::Theta { body, .. } => *body,
            _ => unreachable!("expected theta node"),
        };

        let body_consts_before = func.regions[body]
            .nodes
            .iter()
            .filter(|&&nid| matches!(func.nodes[nid].kind, NodeKind::Simple(IrOp::Const { .. })))
            .count();
        let theta_inputs_before = func.nodes[theta].inputs.len();
        assert!(
            body_consts_before >= 1,
            "fixture should include theta-body constants before pass"
        );

        hoist_theta_loop_invariant_setup_pass(&mut func);

        // After hoisting, constants are threaded as theta inputs, so:
        // - The theta should have more inputs than before
        // - The body should have NO const nodes (they're now accessed via args)
        let theta_after = func.regions[root]
            .nodes
            .iter()
            .copied()
            .find(|&nid| matches!(func.nodes[nid].kind, NodeKind::Theta { .. }))
            .expect("theta should still exist");
        let body_after = match &func.nodes[theta_after].kind {
            NodeKind::Theta { body, .. } => *body,
            _ => unreachable!(),
        };
        let body_consts_after = func.regions[body_after]
            .nodes
            .iter()
            .filter(|&&nid| matches!(func.nodes[nid].kind, NodeKind::Simple(IrOp::Const { .. })))
            .count();
        let theta_inputs_after = func.nodes[theta_after].inputs.len();

        assert_eq!(
            body_consts_after, 0,
            "all body constants should be hoisted as theta arguments"
        );
        assert!(
            theta_inputs_after > theta_inputs_before,
            "theta should have more inputs after hoisting constants as args (before={}, after={})",
            theta_inputs_before,
            theta_inputs_after
        );
    }

    #[test]
    fn theta_hoist_preserves_original_debug_scope_provenance() {
        let mut builder = IrBuilder::new("u8", 0);
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
        let root_scope = func.region_debug_scope(root);
        let theta = func.regions[root]
            .nodes
            .iter()
            .copied()
            .find(|&nid| matches!(func.nodes[nid].kind, NodeKind::Theta { .. }))
            .expect("expected theta node");
        let body = match &func.nodes[theta].kind {
            NodeKind::Theta { body, .. } => *body,
            _ => unreachable!("expected theta node"),
        };
        let original_const = func.regions[body]
            .nodes
            .iter()
            .copied()
            .find(|&nid| {
                matches!(
                    func.nodes[nid].kind,
                    NodeKind::Simple(IrOp::Const { value: 7 })
                )
            })
            .expect("expected theta-body const to hoist");
        let original_scope = func.node_debug_scope(original_const);
        assert_ne!(
            original_scope, root_scope,
            "theta-body nodes should have a distinct body scope"
        );

        hoist_theta_loop_invariant_setup_pass(&mut func);

        let theta_after = func.regions[root]
            .nodes
            .iter()
            .copied()
            .find(|&nid| matches!(func.nodes[nid].kind, NodeKind::Theta { .. }))
            .expect("expected theta node after hoist");
        let theta_pos = func.regions[root]
            .nodes
            .iter()
            .position(|&nid| nid == theta_after)
            .expect("theta should remain in root region");
        let hoisted_const = func.regions[root].nodes[..theta_pos]
            .iter()
            .copied()
            .find(|&nid| {
                matches!(
                    func.nodes[nid].kind,
                    NodeKind::Simple(IrOp::Const { value: 7 })
                ) && func.node_debug_scope(nid) == original_scope
            })
            .expect("expected hoisted const to keep original debug scope provenance");
        assert_eq!(func.node_debug_scope(hoisted_const), original_scope);
    }
}
