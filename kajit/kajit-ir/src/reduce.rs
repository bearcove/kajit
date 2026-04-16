//! RVSDG IR reducer (test case minimizer).
//!
//! Given an `IrFunc` and a predicate ("is this program interesting?"), reduce
//! the program to the smallest version that still satisfies the predicate.
//!
//! Reduction strategies (applied in layers, cycled to fixpoint):
//!
//! 1. **Replace node with Const(0)**: For each Simple node that isn't already
//!    Const, try replacing its output with Const(0).
//!
//! 2. **Collapse gamma to branch**: For each gamma node, try replacing it with
//!    one of its branches (inlining that branch into the parent region).
//!
//! 3. **Reduce theta max_iterations**: Lower the bound, eventually to 1.
//!
//! `IrFunc` implements `Clone`, so each candidate is a cheap clone + mutate.

use kajit_reprs::ir::{IrFunc, IrOp, NodeId, NodeKind, RegionId};

/// Simple xorshift64 PRNG (no external dependency needed).
fn shuffle<T>(items: &mut [T], seed: u64) {
    let mut rng = seed;
    for i in (1..items.len()).rev() {
        rng ^= rng << 13;
        rng ^= rng >> 7;
        rng ^= rng << 17;
        let j = (rng as usize) % (i + 1);
        items.swap(i, j);
    }
}

/// Statistics from a reduction run.
pub struct ReduceResult {
    pub initial_nodes: usize,
    pub final_nodes: usize,
    pub candidates_tested: usize,
    pub reductions_applied: usize,
}

/// Count nodes that are actually referenced in regions (not orphaned arena entries).
fn count_nodes(func: &IrFunc) -> usize {
    func.regions.iter().map(|(_, r)| r.nodes.len()).sum()
}

/// Run the RVSDG reducer.
///
/// - `is_interesting`: returns true if the program still exhibits the bug.
/// - `compact_fn`: optional function to compact arena (e.g., text round-trip).
///   Called between rounds to flush orphaned arena entries.
pub fn reduce_ir(
    func: &IrFunc,
    is_interesting: &dyn Fn(&IrFunc) -> bool,
    compact_fn: Option<&dyn Fn(&IrFunc) -> IrFunc>,
) -> (IrFunc, ReduceResult) {
    let initial_nodes = count_nodes(func);
    let mut current = func.clone();
    let mut total_candidates = 0usize;
    let mut total_reductions = 0usize;

    assert!(
        is_interesting(&current),
        "initial program must be interesting"
    );

    // Fixed-point with patience: keep trying even after a "no progress" round,
    // because shuffled ordering may unlock new reductions.
    let mut stale_rounds = 0;
    let max_stale_rounds = 3; // give up after 3 consecutive no-progress rounds
    let mut round_number = 0u64;

    loop {
        round_number += 1;
        let before = count_nodes(&current);
        let mut round_reductions = 0;

        // Strategy 1: Replace Simple nodes with Const(0) (shuffled order).
        let (next, c, r) = pass_replace_with_const(&current, is_interesting, round_number);
        current = next;
        total_candidates += c;
        total_reductions += r;
        round_reductions += r;

        // Strategy 2: Collapse gammas to one branch.
        let (next, c, r) = pass_collapse_gamma(&current, is_interesting);
        current = next;
        total_candidates += c;
        total_reductions += r;
        round_reductions += r;

        // Strategy 3: Reduce theta max_iterations.
        let (next, c, r) = pass_reduce_theta(&current, is_interesting);
        current = next;
        total_candidates += c;
        total_reductions += r;
        round_reductions += r;

        // Strategy 4: Delete individual nodes (if removing them still verifies).
        let (next, c, r) = pass_delete_nodes(&current, is_interesting, round_number);
        current = next;
        total_candidates += c;
        total_reductions += r;
        round_reductions += r;

        // Strategy 5: Replace theta with passthrough (execute body zero times).
        let (next, c, r) = pass_bypass_theta(&current, is_interesting);
        current = next;
        total_candidates += c;
        total_reductions += r;
        round_reductions += r;

        // Compact arena between rounds to prevent unbounded growth.
        if round_reductions > 0
            && let Some(compact) = compact_fn
        {
            current = compact(&current);
        }

        let after = count_nodes(&current);
        eprintln!(
            "[reduce] round: {before} → {after} nodes ({round_reductions} reductions, {total_candidates} total candidates)"
        );

        // A round with "reductions" that didn't actually shrink the program is stale.
        if after >= before {
            stale_rounds += 1;
            if stale_rounds >= max_stale_rounds {
                break;
            }
        } else {
            stale_rounds = 0;
        }
    }

    let final_nodes = count_nodes(&current);
    (
        current,
        ReduceResult {
            initial_nodes,
            final_nodes,
            candidates_tested: total_candidates,
            reductions_applied: total_reductions,
        },
    )
}

/// Strategy 1: Try replacing each non-Const Simple node with Const(0).
fn pass_replace_with_const(
    func: &IrFunc,
    is_interesting: &dyn Fn(&IrFunc) -> bool,
    seed: u64,
) -> (IrFunc, usize, usize) {
    let mut current = func.clone();
    let mut candidates = 0usize;
    let mut reductions = 0usize;

    let mut targets: Vec<NodeId> = current
        .regions
        .iter()
        .flat_map(|(_, region)| region.nodes.iter().copied())
        .filter(|&nid| match &current.nodes[nid].kind {
            NodeKind::Simple(op) if !matches!(op, IrOp::Const { .. }) => {
                !current.nodes[nid].outputs.is_empty()
            }
            _ => false,
        })
        .collect();

    shuffle(&mut targets, seed);

    for target in targets {
        candidates += 1;

        // Clone, mutate, test.
        let mut candidate = current.clone();

        // Check ID is still valid after clone.
        if target.index() >= candidate.nodes.len() {
            continue;
        }

        // Replace with Const(0).
        candidate.nodes[target].kind = NodeKind::Simple(IrOp::Const { value: 0 });
        candidate.nodes[target].inputs.clear();

        if crate::verify(&candidate).is_err() {
            continue;
        }

        if is_interesting(&candidate) {
            eprintln!("[reduce] replaced n{} with Const(0)", target.index());
            current = candidate;
            reductions += 1;
        }
    }

    (current, candidates, reductions)
}

/// Strategy 2: Try collapsing each gamma to one of its branches.
fn pass_collapse_gamma(
    func: &IrFunc,
    is_interesting: &dyn Fn(&IrFunc) -> bool,
) -> (IrFunc, usize, usize) {
    let mut current = func.clone();
    let mut candidates = 0usize;
    let mut reductions = 0usize;

    loop {
        // Scan gammas from region node lists (not arena) to avoid finding orphans.
        let gammas: Vec<(NodeId, Vec<RegionId>)> = current
            .regions
            .iter()
            .flat_map(|(_, region)| region.nodes.iter().copied())
            .filter_map(|nid| match &current.nodes[nid].kind {
                NodeKind::Gamma { regions } => Some((nid, regions.clone())),
                _ => None,
            })
            .collect();

        if gammas.is_empty() {
            break;
        }

        let mut any_collapsed = false;

        for (gamma_id, regions) in &gammas {
            for branch_idx in 0..regions.len() {
                candidates += 1;

                let mut candidate = match Some(current.clone()) {
                    Some(c) => c,
                    None => continue,
                };

                // Check the gamma still exists in the clone (same NodeId).
                if gamma_id.index() >= candidate.nodes.len() {
                    continue;
                }
                if !matches!(&candidate.nodes[*gamma_id].kind, NodeKind::Gamma { .. }) {
                    continue;
                }

                // Extract region and scope before mutating.
                let gamma_region = candidate.nodes[*gamma_id].region;
                let scope = candidate.regions[gamma_region].debug_scope;

                // Replace the predicate with Const(branch_idx) to enable folding.
                let pred_const = crate::const_fold::create_const_in_region(
                    &mut candidate,
                    gamma_region,
                    scope,
                    branch_idx as u64,
                );
                candidate.nodes[*gamma_id].inputs[0].source = pred_const;

                // Fold the constant predicate.
                crate::simplify_gamma::simplify_trivial_gammas(&mut candidate);

                if crate::verify(&candidate).is_err() {
                    continue;
                }

                // Check the gamma was actually removed.
                let gamma_still_exists = candidate
                    .regions
                    .iter()
                    .flat_map(|(_, r)| r.nodes.iter().copied())
                    .any(|nid| nid == *gamma_id);
                if gamma_still_exists {
                    // simplify_trivial_gammas didn't fold it — skip.
                    continue;
                }

                if is_interesting(&candidate) {
                    eprintln!(
                        "[reduce] collapsed gamma n{} to branch {branch_idx}",
                        gamma_id.index()
                    );
                    current = candidate;
                    reductions += 1;
                    any_collapsed = true;
                    break;
                }
            }
            if any_collapsed {
                break;
            }
        }

        if !any_collapsed {
            break;
        }
    }

    (current, candidates, reductions)
}

/// Strategy 3: Try reducing theta max_iterations.
fn pass_reduce_theta(
    func: &IrFunc,
    is_interesting: &dyn Fn(&IrFunc) -> bool,
) -> (IrFunc, usize, usize) {
    let mut current = func.clone();
    let mut candidates = 0usize;
    let mut reductions = 0usize;

    let thetas: Vec<(NodeId, u32)> = current
        .regions
        .iter()
        .flat_map(|(_, region)| region.nodes.iter().copied())
        .filter_map(|nid| match &current.nodes[nid].kind {
            NodeKind::Theta {
                max_iterations: Some(n),
                ..
            } => Some((nid, *n)),
            _ => None,
        })
        .collect();

    for (theta_id, max_iter) in thetas {
        if max_iter <= 1 {
            continue;
        }

        for &new_max in &[max_iter / 2, max_iter - 1, 1] {
            if new_max >= max_iter || new_max == 0 {
                continue;
            }
            candidates += 1;

            let mut candidate = match Some(current.clone()) {
                Some(c) => c,
                None => continue,
            };

            if let NodeKind::Theta {
                max_iterations: ref mut mi,
                ..
            } = candidate.nodes[theta_id].kind
            {
                *mi = Some(new_max);
            }

            if crate::verify(&candidate).is_err() {
                continue;
            }

            if is_interesting(&candidate) {
                eprintln!(
                    "[reduce] theta n{}: max_iterations {} → {new_max}",
                    theta_id.index(),
                    max_iter
                );
                current = candidate;
                reductions += 1;
                break;
            }
        }
    }

    (current, candidates, reductions)
}

/// Strategy 4: Try deleting individual nodes entirely.
///
/// For each node, remove it from its region and replace output references
/// with Const(0). More aggressive than Strategy 1.
fn pass_delete_nodes(
    func: &IrFunc,
    is_interesting: &dyn Fn(&IrFunc) -> bool,
    seed: u64,
) -> (IrFunc, usize, usize) {
    let mut current = func.clone();
    let mut candidates = 0usize;
    let mut reductions = 0usize;

    let mut targets: Vec<(NodeId, RegionId)> = current
        .regions
        .iter()
        .flat_map(|(rid, region)| region.nodes.iter().map(move |&nid| (nid, rid)))
        .collect();

    shuffle(&mut targets, seed.wrapping_add(0x12345));

    for (target, region) in targets {
        if target == current.root {
            continue;
        }
        candidates += 1;

        let mut candidate = current.clone();
        if target.index() >= candidate.nodes.len() {
            continue;
        }

        // Create Const(0) replacement if node has outputs.
        let has_outputs = !candidate.nodes[target].outputs.is_empty();
        let replacement = if has_outputs {
            let scope = candidate.regions[region].debug_scope;
            Some(crate::const_fold::create_const_in_region(
                &mut candidate,
                region,
                scope,
                0,
            ))
        } else {
            None
        };

        // Rewrite all uses of this node's outputs.
        if let Some(repl) = replacement {
            let output_count = candidate.nodes[target].outputs.len();
            for out_idx in 0..output_count {
                let from_source = crate::PortSource::Node(crate::OutputRef {
                    node: target,
                    index: out_idx as u16,
                });
                let nids: Vec<NodeId> = candidate.nodes.iter().map(|(nid, _)| nid).collect();
                for nid in nids {
                    for inp in &mut candidate.nodes[nid].inputs {
                        if inp.source == from_source {
                            inp.source = repl;
                        }
                    }
                }
                let rids: Vec<crate::ResultId> = candidate
                    .region_results
                    .iter()
                    .map(|(rid, _)| rid)
                    .collect();
                for rid in rids {
                    if candidate.region_results[rid].source == from_source {
                        candidate.region_results[rid].source = repl;
                    }
                }
            }
        }

        candidate.regions[region].nodes.retain(|&nid| nid != target);

        if crate::verify(&candidate).is_err() {
            continue;
        }
        if is_interesting(&candidate) {
            eprintln!("[reduce] deleted n{}", target.index());
            // Re-clone to flush orphaned arena entries before continuing.
            current = candidate;
            reductions += 1;
        }
    }

    (current, candidates, reductions)
}

/// Strategy 5: Bypass a theta entirely (replace with input passthrough).
fn pass_bypass_theta(
    func: &IrFunc,
    is_interesting: &dyn Fn(&IrFunc) -> bool,
) -> (IrFunc, usize, usize) {
    let mut current = func.clone();
    let mut candidates = 0usize;
    let mut reductions = 0usize;

    let thetas: Vec<NodeId> = current
        .regions
        .iter()
        .flat_map(|(_, region)| region.nodes.iter().copied())
        .filter(|&nid| matches!(&current.nodes[nid].kind, NodeKind::Theta { .. }))
        .collect();

    for theta_id in thetas {
        candidates += 1;

        let mut candidate = current.clone();
        if theta_id.index() >= candidate.nodes.len() {
            continue;
        }
        if !matches!(&candidate.nodes[theta_id].kind, NodeKind::Theta { .. }) {
            continue;
        }

        let output_count = candidate.nodes[theta_id].outputs.len();
        let input_count = candidate.nodes[theta_id].inputs.len();

        for out_idx in 0..output_count.min(input_count) {
            let from_source = crate::PortSource::Node(crate::OutputRef {
                node: theta_id,
                index: out_idx as u16,
            });
            let to = candidate.nodes[theta_id].inputs[out_idx].source;
            if from_source == to {
                continue;
            }

            let nids: Vec<NodeId> = candidate.nodes.iter().map(|(nid, _)| nid).collect();
            for nid in nids {
                for inp in &mut candidate.nodes[nid].inputs {
                    if inp.source == from_source {
                        inp.source = to;
                    }
                }
            }
            let rids: Vec<crate::ResultId> = candidate
                .region_results
                .iter()
                .map(|(rid, _)| rid)
                .collect();
            for rid in rids {
                if candidate.region_results[rid].source == from_source {
                    candidate.region_results[rid].source = to;
                }
            }
        }

        let region = candidate.nodes[theta_id].region;
        candidate.regions[region]
            .nodes
            .retain(|&nid| nid != theta_id);

        if crate::verify(&candidate).is_err() {
            continue;
        }
        if is_interesting(&candidate) {
            eprintln!("[reduce] bypassed theta n{}", theta_id.index());
            current = candidate;
            reductions += 1;
        }
    }

    (current, candidates, reductions)
}
