//! Post-unroll RVSDG canonicalizer.
//!
//! After bounded-theta unrolling, the RVSDG contains a nested gamma cascade
//! where many values are implicitly constant (e.g., iteration indices, shift
//! amounts) but are not structurally visible as `Const` nodes — they flow
//! through gamma region args from values that were only known at unroll time.
//!
//! This pass carries an **environment** that tracks known constant values
//! through gamma branches and performs two transforms:
//!
//! 1. **Predicate materialization**: When a gamma predicate resolves to a
//!    constant via the env, materialize a structural Const node so that the
//!    existing `simplify_trivial_gammas` can fold the gamma.
//!
//! 2. **Pure-op constant folding**: When a pure arithmetic/comparison node has
//!    all inputs resolvable to constants via the env, replace it with a Const.
//!
//! ## Safety boundary: Role::Control vs Role::Data
//!
//! Values that feed gamma predicates or control-only outputs use the
//! **conservative** resolver (env + direct Const + predicate-known branch
//! traversal only). Values that feed pure data computation use the **strong**
//! resolver (adds "all branches same constant" fallback).
//!
//! This prevents the class of miscompiles where a bounds-check gamma always
//! produces 1 (meaning "check passed") and that 1 leaks into a downstream
//! predicate where it means "overflow".

use std::collections::HashMap;

use crate::{ArgId, IrFunc, IrOp, NodeId, NodeKind, OutputRef, PortKind, PortSource, RegionId};

// ---------------------------------------------------------------------------
// Data structures
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum KnownValue {
    Unknown,
    Const(u64),
}

/// Local value-role classification. NOT an IR concept — only used inside this
/// pass to decide which resolver strength is safe.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Role {
    /// Feeds a gamma predicate or control-only chain. Conservative resolver only.
    Control,
    /// Feeds pure data computation. Strong resolver allowed.
    Data,
}

/// Hashable key for PortSource.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum PortSourceKey {
    NodeOutput { node: NodeId, index: u16 },
    RegionArg { region: RegionId, arg: ArgId },
}

impl From<&PortSource> for PortSourceKey {
    fn from(source: &PortSource) -> Self {
        match source {
            PortSource::Node(out_ref) => PortSourceKey::NodeOutput {
                node: out_ref.node,
                index: out_ref.index,
            },
            PortSource::RegionArg(arg_ref) => PortSourceKey::RegionArg {
                region: arg_ref.region,
                arg: arg_ref.arg,
            },
        }
    }
}

struct StateEnv {
    known: HashMap<PortSourceKey, KnownValue>,
}

impl StateEnv {
    fn new() -> Self {
        Self {
            known: HashMap::new(),
        }
    }

    /// Resolve with the strong resolver (all-branches-same allowed).
    /// Safe for Role::Data inputs to pure nodes.
    fn resolve_data(&self, func: &IrFunc, source: &PortSource) -> KnownValue {
        self.resolve_inner(func, source, 0, true)
    }

    /// Resolve with the conservative resolver (no all-branches-same).
    /// Required for Role::Control (gamma predicates, control-only outputs).
    fn resolve_control(&self, func: &IrFunc, source: &PortSource) -> KnownValue {
        self.resolve_inner(func, source, 0, false)
    }

    fn resolve_inner(
        &self,
        func: &IrFunc,
        source: &PortSource,
        depth: usize,
        allow_all_branches_same: bool,
    ) -> KnownValue {
        if depth > 16 {
            return KnownValue::Unknown;
        }

        // 1. Direct env lookup
        let key = PortSourceKey::from(source);
        if let Some(&val) = self.known.get(&key) {
            return val;
        }

        match source {
            PortSource::Node(out_ref) => {
                let node = &func.nodes[out_ref.node];
                match &node.kind {
                    // 2. Direct Const
                    NodeKind::Simple(IrOp::Const { value }) => KnownValue::Const(*value),

                    // 3-4. Gamma output resolution
                    NodeKind::Gamma { regions } => {
                        let output_index = out_ref.index as usize;

                        // 3. Predicate-known path: select the taken branch.
                        let pred_source = &node.inputs[0].source;
                        if let KnownValue::Const(pred_val) = self.resolve_inner(
                            func,
                            pred_source,
                            depth + 1,
                            allow_all_branches_same,
                        ) {
                            let branch_idx =
                                (pred_val as usize).min(regions.len().saturating_sub(1));
                            let branch_region = regions[branch_idx];
                            if let Some(&result_id) =
                                func.regions[branch_region].results.get(output_index)
                            {
                                let result_source = &func.region_results[result_id].source;
                                let branch_env =
                                    self.build_branch_env(func, out_ref.node, branch_idx);
                                return branch_env.resolve_inner(
                                    func,
                                    result_source,
                                    depth + 1,
                                    allow_all_branches_same,
                                );
                            }
                        }

                        // 4. All branches same constant (only for Role::Data).
                        if allow_all_branches_same {
                            if let Some(v) = self.all_branches_same(
                                func,
                                out_ref.node,
                                regions,
                                output_index,
                                depth,
                            ) {
                                return KnownValue::Const(v);
                            }
                        }

                        KnownValue::Unknown
                    }

                    _ => KnownValue::Unknown,
                }
            }
            PortSource::RegionArg(arg_ref) => {
                // 5. Trace through parent gamma/theta arg mapping.
                let arg_index = func.regions[arg_ref.region]
                    .args
                    .iter()
                    .position(|a| *a == arg_ref.arg);
                let Some(arg_index) = arg_index else {
                    return KnownValue::Unknown;
                };
                let Some(owner) = func.find_region_owner(arg_ref.region) else {
                    return KnownValue::Unknown;
                };

                match &func.nodes[owner].kind {
                    NodeKind::Gamma { .. } => {
                        if let Some(input) = func.nodes[owner].inputs.get(arg_index + 1) {
                            return self.resolve_inner(
                                func,
                                &input.source,
                                depth + 1,
                                allow_all_branches_same,
                            );
                        }
                        KnownValue::Unknown
                    }
                    NodeKind::Theta { body, .. } => {
                        let body_region = *body;
                        let result_index = arg_index + 1;
                        if let Some(&result_id) =
                            func.regions[body_region].results.get(result_index)
                        {
                            let result_source = &func.region_results[result_id].source;
                            if let PortSource::RegionArg(r) = result_source {
                                if r.region == body_region && r.arg == arg_ref.arg {
                                    if let Some(input) = func.nodes[owner].inputs.get(arg_index) {
                                        return self.resolve_inner(
                                            func,
                                            &input.source,
                                            depth + 1,
                                            allow_all_branches_same,
                                        );
                                    }
                                }
                            }
                        }
                        KnownValue::Unknown
                    }
                    _ => KnownValue::Unknown,
                }
            }
        }
    }

    /// Lightweight "all branches same constant" check.
    /// Only checks direct Const results and passthrough args.
    fn all_branches_same(
        &self,
        func: &IrFunc,
        gamma_node: NodeId,
        regions: &[RegionId],
        output_index: usize,
        depth: usize,
    ) -> Option<u64> {
        let mut common: Option<u64> = None;
        for &region_id in regions {
            let result_id = *func.regions[region_id].results.get(output_index)?;
            let result_source = &func.region_results[result_id].source;
            let branch_val = match result_source {
                PortSource::Node(r) => match &func.nodes[r.node].kind {
                    NodeKind::Simple(IrOp::Const { value }) => Some(*value),
                    _ => None,
                },
                PortSource::RegionArg(arg_ref) => {
                    let arg_idx = func.regions[region_id]
                        .args
                        .iter()
                        .position(|a| *a == arg_ref.arg)?;
                    let input = func.nodes[gamma_node].inputs.get(arg_idx + 1)?;
                    match self.resolve_inner(func, &input.source, depth + 1, true) {
                        KnownValue::Const(v) => Some(v),
                        _ => None,
                    }
                }
            }?;
            match common {
                None => common = Some(branch_val),
                Some(prev) if prev == branch_val => {}
                _ => return None,
            }
        }
        common
    }

    fn build_branch_env(&self, func: &IrFunc, gamma_node: NodeId, branch_idx: usize) -> StateEnv {
        let NodeKind::Gamma { ref regions } = func.nodes[gamma_node].kind else {
            return StateEnv::new();
        };
        let branch_region = regions[branch_idx];
        let branch_args = &func.regions[branch_region].args;
        let gamma_inputs = &func.nodes[gamma_node].inputs;

        let mut child = StateEnv {
            known: self.known.clone(),
        };

        for (k, &arg_id) in branch_args.iter().enumerate() {
            if let Some(input) = gamma_inputs.get(k + 1) {
                // Use strong resolver for branch arg propagation — these are
                // data values flowing into the branch, not predicates.
                let val = self.resolve_data(func, &input.source);
                let key = PortSourceKey::RegionArg {
                    region: branch_region,
                    arg: arg_id,
                };
                child.known.insert(key, val);
            }
        }

        child
    }
}

// ---------------------------------------------------------------------------
// Op whitelist
// ---------------------------------------------------------------------------

/// Returns true if the op is a pure computation on the whitelist.
fn is_foldable_pure(op: &IrOp) -> bool {
    matches!(
        op,
        IrOp::Const { .. }
            | IrOp::Identity
            | IrOp::Add
            | IrOp::Sub
            | IrOp::Mul
            | IrOp::And
            | IrOp::Or
            | IrOp::Xor
            | IrOp::Shl
            | IrOp::Shr
            | IrOp::Sar
            | IrOp::CmpEq
            | IrOp::CmpNe
            | IrOp::CmpLt
            | IrOp::CmpLe
            | IrOp::CmpGt
            | IrOp::CmpGe
            | IrOp::ZigzagDecode { .. }
            | IrOp::SignExtend { .. }
    )
}

// ---------------------------------------------------------------------------
// Core transforms
// ---------------------------------------------------------------------------

/// Try to fold a pure node to a constant using the environment.
/// Uses resolve_data (strong resolver) for all data inputs.
fn try_fold_with_env(func: &mut IrFunc, node_id: NodeId, env: &StateEnv) -> bool {
    let NodeKind::Simple(ref op) = func.nodes[node_id].kind else {
        return false;
    };
    if matches!(op, IrOp::Const { .. }) {
        return false;
    }
    if !is_foldable_pure(op) {
        return false;
    }
    let op = op.clone();

    // Resolve all data inputs to constants via the strong (Data) resolver.
    let inputs = func.nodes[node_id].inputs.clone();
    let mut const_values: Vec<u64> = Vec::new();
    for input in &inputs {
        if input.kind != PortKind::Data {
            continue;
        }
        match env.resolve_data(func, &input.source) {
            KnownValue::Const(v) => const_values.push(v),
            KnownValue::Unknown => return false,
        }
    }

    // Handle Identity specially: just propagate the input value.
    if matches!(op, IrOp::Identity) {
        if const_values.len() == 1 {
            func.nodes[node_id].kind = NodeKind::Simple(IrOp::Const {
                value: const_values[0],
            });
            func.nodes[node_id].inputs.clear();
            return true;
        }
        return false;
    }

    let Some(result) = crate::const_fold::evaluate_op(&op, &const_values) else {
        return false;
    };

    // Replace node with Const.
    func.nodes[node_id].kind = NodeKind::Simple(IrOp::Const { value: result });
    func.nodes[node_id].inputs.clear();
    true
}

/// Materialize a gamma predicate as a structural Const node.
/// Uses resolve_control (conservative resolver, Role::Control).
fn materialize_known_predicate(func: &mut IrFunc, gamma_node: NodeId, env: &StateEnv) -> bool {
    let pred_source = func.nodes[gamma_node].inputs[0].source;

    // If already a Const node, nothing to do.
    if let PortSource::Node(out) = pred_source {
        if matches!(
            func.nodes[out.node].kind,
            NodeKind::Simple(IrOp::Const { .. })
        ) {
            return false;
        }
    }

    // Role::Control — conservative resolver only.
    let KnownValue::Const(v) = env.resolve_control(func, &pred_source) else {
        return false;
    };

    let parent_region = func.nodes[gamma_node].region;
    let debug_scope = func.nodes[gamma_node].debug_scope;

    // Reuse an existing Const(v) in the same region before the gamma.
    let existing_const = func.regions[parent_region]
        .nodes
        .iter()
        .take_while(|&&nid| nid != gamma_node)
        .find(|&&nid| {
            matches!(func.nodes[nid].kind, NodeKind::Simple(IrOp::Const { value }) if value == v)
        })
        .copied();

    let const_source = if let Some(existing_nid) = existing_const {
        PortSource::Node(OutputRef {
            node: existing_nid,
            index: 0,
        })
    } else {
        let src = crate::const_fold::create_const_in_region(func, parent_region, debug_scope, v);
        let PortSource::Node(const_ref) = src else {
            unreachable!()
        };
        // Move the new Const from the end of the region to just before the gamma.
        let region_nodes = &mut func.regions[parent_region].nodes;
        let popped = region_nodes.pop().unwrap();
        debug_assert_eq!(popped, const_ref.node);
        if let Some(gamma_pos) = region_nodes.iter().position(|&nid| nid == gamma_node) {
            region_nodes.insert(gamma_pos, const_ref.node);
        } else {
            region_nodes.push(const_ref.node);
        }
        src
    };

    func.nodes[gamma_node].inputs[0].source = const_source;
    true
}

// ---------------------------------------------------------------------------
// Recursive region simplifier
// ---------------------------------------------------------------------------

struct Stats {
    materialized: u32,
    folded: u32,
}

fn simplify_region(
    func: &mut IrFunc,
    region: RegionId,
    env: &StateEnv,
    depth: usize,
    stats: &mut Stats,
) -> bool {
    if depth > 20 {
        return false;
    }

    let mut changed = false;
    let nodes: Vec<NodeId> = func.regions[region].nodes.clone();

    for &node_id in &nodes {
        if !func.regions[region].nodes.contains(&node_id) {
            continue;
        }

        match &func.nodes[node_id].kind {
            NodeKind::Simple(op) if !matches!(op, IrOp::Const { .. }) && is_foldable_pure(op) => {
                if try_fold_with_env(func, node_id, env) {
                    stats.folded += 1;
                    changed = true;
                }
            }
            NodeKind::Gamma { regions } => {
                let regions = regions.clone();

                if materialize_known_predicate(func, node_id, env) {
                    stats.materialized += 1;
                    changed = true;
                }

                for (branch_idx, &branch_region) in regions.iter().enumerate() {
                    let branch_env = env.build_branch_env(func, node_id, branch_idx);
                    if simplify_region(func, branch_region, &branch_env, depth + 1, stats) {
                        changed = true;
                    }
                }
            }
            _ => {}
        }
    }

    changed
}

// ---------------------------------------------------------------------------
// Entry point
// ---------------------------------------------------------------------------

/// Run the post-unroll canonicalizer to fixpoint.
pub fn post_unroll_canonicalize(func: &mut IrFunc) -> bool {
    // Skip for large functions to avoid compile-time regression.
    let total_nodes: usize = func.regions.iter().map(|(_, r)| r.nodes.len()).sum();
    if total_nodes > 800 {
        return false;
    }

    let debug = std::env::var("KAJIT_DEBUG_CANONICALIZE").is_ok();
    let mut any_changed = false;
    let mut total_stats = Stats {
        materialized: 0,
        folded: 0,
    };

    for round in 0..4 {
        let env = StateEnv::new();
        let root_body = func.root_body();
        let mut stats = Stats {
            materialized: 0,
            folded: 0,
        };
        let changed = simplify_region(func, root_body, &env, 0, &mut stats);
        total_stats.materialized += stats.materialized;
        total_stats.folded += stats.folded;
        if !changed {
            break;
        }
        any_changed = true;

        if debug {
            let mut node_count = 0usize;
            let mut gamma_count = 0usize;
            for (_, region) in func.regions.iter() {
                for &nid in &region.nodes {
                    node_count += 1;
                    if matches!(func.nodes[nid].kind, NodeKind::Gamma { .. }) {
                        gamma_count += 1;
                    }
                }
            }
            eprintln!(
                "[canonicalize] round {}: {} nodes, {} gammas, {} preds, {} folds",
                round, node_count, gamma_count, stats.materialized, stats.folded
            );
        }
    }

    if debug && (total_stats.materialized > 0 || total_stats.folded > 0) {
        eprintln!(
            "[canonicalize] total: {} predicates materialized, {} pure ops folded",
            total_stats.materialized, total_stats.folded
        );
    }

    any_changed
}
