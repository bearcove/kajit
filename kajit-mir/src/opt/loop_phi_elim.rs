//! Loop-invariant phi elimination.
//!
//! Eliminates phi parameters that are loop-invariant (always get the same value).
//!
//! Example: A loop header with 11 parameters where 6 always get const 0:
//! ```text
//! block b1 params=[v86, v87, v88, v89, v90, v91, v92, v93, v94, v95, v96]
//!   edge e0: [v86=>v0, v87=>v2, v88=>v3, v89=>v3, v90=>v3, v91=>v3, v92=>v3, v93=>v3, ...]
//!   edge e39: [v86=>v63, v87=>v87, v88=>v3, v89=>v3, v90=>v3, v91=>v3, v92=>v3, v93=>v3, ...]
//! ```
//!
//! All edges pass v3 to v88-v93, so those parameters can be eliminated:
//! - Replace all uses of v88-v93 with v3
//! - Remove v88-v93 from block params
//! - Remove corresponding args from all edges

use std::collections::{HashMap, HashSet};

use kajit_ir::VReg;

use crate::{
    analysis::{dominance::DominanceInfo, loops::LoopInfo},
    cfg_mir::{BlockId, EdgeArg, EdgeId, Function},
};

/// Eliminate loop-invariant phi parameters from loop headers.
///
/// Returns true if any changes were made.
pub fn eliminate_loop_invariant_phis(
    func: &mut Function,
    dom: &DominanceInfo,
    loops: &LoopInfo,
) -> bool {
    let debug = std::env::var("KAJIT_DEBUG_LOOP_PHI").is_ok();
    let mut changed = false;

    // Process each loop header
    for header in loops.loop_headers() {
        if eliminate_invariant_phis_in_header(func, dom, loops, header, debug) {
            changed = true;
        }
    }

    if debug && changed {
        eprintln!("[loop_phi_elim] eliminated loop-invariant parameters");
    }

    changed
}

/// Eliminate loop-invariant phi parameters for a single loop header.
fn eliminate_invariant_phis_in_header(
    func: &mut Function,
    dom: &DominanceInfo,
    loops: &LoopInfo,
    header: BlockId,
    debug: bool,
) -> bool {
    let header_block = &func.blocks[header.index()];
    let params = header_block.params.clone();

    if params.is_empty() {
        return false;
    }

    // Get loop data for this header
    let loop_data = loops.loop_for_header(header).unwrap();

    // Separate backedges from entry edges
    let backedge_set: HashSet<EdgeId> = loop_data.backedges.iter().copied().collect();
    let entry_edges: Vec<EdgeId> = header_block
        .preds
        .iter()
        .copied()
        .filter(|eid| !backedge_set.contains(eid))
        .collect();
    let backedges: Vec<EdgeId> = loop_data.backedges.clone();

    if entry_edges.is_empty() {
        return false; // No entry edges (unreachable loop?)
    }

    // For each parameter, check if it's loop-invariant
    let mut invariant_params: HashMap<usize, VReg> = HashMap::new();

    if debug {
        eprintln!("[loop_phi_elim] analyzing {} parameters:", params.len());
        for (idx, &vreg) in params.iter().enumerate() {
            eprintln!("  param {} = v{}", idx, vreg.index());
        }
        eprintln!(
            "[loop_phi_elim] {} entry edges, {} backedges:",
            entry_edges.len(),
            backedges.len()
        );
        for &edge_id in &entry_edges {
            let edge = &func.edges[edge_id.index()];
            eprintln!(
                "  entry e{} (b{} -> b{}): {} args",
                edge_id.index(),
                edge.from.index(),
                edge.to.index(),
                edge.args.len()
            );
        }
        for &edge_id in &backedges {
            let edge = &func.edges[edge_id.index()];
            eprintln!(
                "  backedge e{} (b{} -> b{}): {} args",
                edge_id.index(),
                edge.from.index(),
                edge.to.index(),
                edge.args.len()
            );
        }
    }

    for (param_idx, &param_vreg) in params.iter().enumerate() {
        // Check entry edges: all must pass the same value
        let mut entry_value: Option<VReg> = None;
        let mut is_loop_invariant = true;

        for &edge_id in &entry_edges {
            let edge = &func.edges[edge_id.index()];
            if param_idx >= edge.args.len() {
                // Edge doesn't provide this parameter - not invariant
                if debug {
                    eprintln!(
                        "  param {} (v{}): entry edge e{} missing args (has {}, need {})",
                        param_idx,
                        param_vreg.index(),
                        edge_id.index(),
                        edge.args.len(),
                        param_idx + 1
                    );
                }
                is_loop_invariant = false;
                break;
            }

            let value = edge.args[param_idx].source;
            if let Some(expected) = entry_value {
                if value != expected {
                    // Entry edges don't agree on value - not invariant
                    is_loop_invariant = false;
                    break;
                }
            } else {
                entry_value = Some(value);
            }
        }

        if !is_loop_invariant {
            continue;
        }

        let first_value = entry_value.unwrap();

        // Check backedges: must pass either first_value or param_vreg (loop-carried)
        for &edge_id in &backedges {
            let edge = &func.edges[edge_id.index()];
            if param_idx >= edge.args.len() {
                is_loop_invariant = false;
                break;
            }

            let value = edge.args[param_idx].source;
            if value != first_value && value != param_vreg {
                // Backedge passes something other than invariant value or self
                is_loop_invariant = false;
                break;
            }
        }

        if !is_loop_invariant {
            continue;
        }

        if debug {
            eprintln!(
                "  param {} (v{}): loop-invariant, first = v{}",
                param_idx,
                param_vreg.index(),
                first_value.index()
            );
        }

        if first_value != param_vreg {
            // This parameter is loop-invariant!
            // Verify that the invariant value definition dominates ALL uses of the parameter
            if let Some(def_block) = find_def_block(func, first_value) {
                // Find all blocks that use this parameter
                let use_blocks = find_vreg_use_blocks(func, param_vreg);

                // Check if def_block dominates all use blocks
                let dominates_all = use_blocks
                    .iter()
                    .all(|&use_block| dom.dominates(def_block, use_block));

                if debug {
                    eprintln!(
                        "    -> invariant! v{} defined in b{}, dominates header={}, dominates all {} uses={}",
                        first_value.index(),
                        def_block.index(),
                        dom.dominates(def_block, header),
                        use_blocks.len(),
                        dominates_all
                    );
                }
                if dominates_all {
                    invariant_params.insert(param_idx, first_value);
                }
            } else if func.data_args.contains(&first_value) {
                // Function argument - always dominates everything
                if debug {
                    eprintln!(
                        "    -> invariant! v{} is function arg, dominates all",
                        first_value.index()
                    );
                }
                invariant_params.insert(param_idx, first_value);
            } else {
                // No definition found and not a function arg - skip this parameter
                // (vreg might be undefined, which would violate SSA)
                if debug {
                    eprintln!(
                        "    -> skipping: v{} has no definition (not a param, inst def, or func arg)",
                        first_value.index()
                    );
                }
            }
        } else {
            // first_value == param_vreg, so it's a self-reference (loop-carried)
            if debug {
                eprintln!("    -> self-reference (param == first_value), not eliminated");
            }
        }
    }

    if invariant_params.is_empty() {
        return false;
    }

    if debug {
        eprintln!(
            "[loop_phi_elim] header b{}: eliminating {} invariant params (of {})",
            header.index(),
            invariant_params.len(),
            params.len()
        );
        for (&idx, &value) in &invariant_params {
            eprintln!(
                "  param {} (v{}) always gets v{}",
                idx,
                params[idx].index(),
                value.index()
            );
        }
    }

    // Replace all uses of invariant parameters with their invariant values
    replace_vregs_in_function(func, &invariant_params, &params);

    // Remove invariant parameters from header block
    let new_params: Vec<VReg> = params
        .iter()
        .enumerate()
        .filter_map(|(idx, &vreg)| {
            if invariant_params.contains_key(&idx) {
                None
            } else {
                Some(vreg)
            }
        })
        .collect();

    func.blocks[header.index()].params = new_params;

    // Update all incoming edges (both entry and backedges) to remove corresponding arguments
    let all_incoming: Vec<EdgeId> = entry_edges
        .iter()
        .chain(backedges.iter())
        .copied()
        .collect();
    for &edge_id in &all_incoming {
        let edge = &mut func.edges[edge_id.index()];
        let new_args: Vec<EdgeArg> = edge
            .args
            .iter()
            .enumerate()
            .filter_map(|(idx, &arg)| {
                if invariant_params.contains_key(&idx) {
                    None
                } else {
                    Some(arg)
                }
            })
            .collect();
        edge.args = new_args;
    }

    true
}

/// Find all blocks that use a vreg (in instructions, terminators, or edge arguments).
fn find_vreg_use_blocks(func: &Function, vreg: VReg) -> Vec<BlockId> {
    let mut use_blocks = Vec::new();

    for block in &func.blocks {
        if block.dead {
            continue;
        }

        let mut found_in_block = false;

        // Check instructions
        for &inst_id in &block.insts {
            let inst = &func.insts[inst_id.index()];
            for operand in &inst.operands {
                if operand.vreg == vreg && operand.kind == crate::cfg_mir::OperandKind::Use {
                    use_blocks.push(block.id);
                    found_in_block = true;
                    break;
                }
            }
            if found_in_block {
                break;
            }
        }

        if found_in_block {
            continue;
        }

        // Check terminator conditions
        let term = &func.terms[block.term.index()];
        match term {
            crate::cfg_mir::Terminator::BranchIf { cond, .. }
            | crate::cfg_mir::Terminator::BranchIfZero { cond, .. } => {
                if *cond == vreg {
                    use_blocks.push(block.id);
                    found_in_block = true;
                }
            }
            crate::cfg_mir::Terminator::JumpTable { predicate, .. } => {
                if *predicate == vreg {
                    use_blocks.push(block.id);
                    found_in_block = true;
                }
            }
            _ => {}
        }

        if found_in_block {
            continue;
        }

        // Check edge arguments (source vregs passed to successors)
        for &succ_edge_id in &block.succs {
            let edge = &func.edges[succ_edge_id.index()];
            for arg in &edge.args {
                if arg.source == vreg {
                    use_blocks.push(block.id);
                    found_in_block = true;
                    break;
                }
            }
            if found_in_block {
                break;
            }
        }
    }

    use_blocks
}

/// Find the block that defines a vreg, if it's defined by an instruction.
fn find_def_block(func: &Function, vreg: VReg) -> Option<BlockId> {
    for block in &func.blocks {
        // Check if vreg is a block parameter
        if block.params.contains(&vreg) {
            return Some(block.id);
        }

        // Check if vreg is defined by an instruction in this block
        for &inst_id in &block.insts {
            let inst = &func.insts[inst_id.index()];
            for operand in &inst.operands {
                if operand.vreg == vreg && operand.kind.is_def() {
                    return Some(block.id);
                }
            }
        }
    }
    None
}

/// Replace uses of invariant parameters with their invariant values.
fn replace_vregs_in_function(
    func: &mut Function,
    invariant_params: &HashMap<usize, VReg>,
    params: &[VReg],
) {
    // Build replacement map: param vreg -> invariant value
    let replacements: HashMap<VReg, VReg> = invariant_params
        .iter()
        .map(|(&idx, &value)| (params[idx], value))
        .collect();

    if replacements.is_empty() {
        return;
    }

    // Replace in instructions (both op field and operands)
    for block in &func.blocks {
        for &inst_id in &block.insts {
            let inst = &mut func.insts[inst_id.index()];

            // Replace vregs in the LinearOp itself
            inst.op.for_each_vreg_mut(|v| {
                if let Some(&repl) = replacements.get(v) {
                    *v = repl;
                }
            });

            // Replace vregs in operands array
            for operand in &mut inst.operands {
                if let Some(&replacement) = replacements.get(&operand.vreg) {
                    operand.vreg = replacement;
                }
            }
        }
    }

    // Replace in terminator conditions
    for term in &mut func.terms {
        match term {
            crate::cfg_mir::Terminator::BranchIf { cond, .. }
            | crate::cfg_mir::Terminator::BranchIfZero { cond, .. } => {
                if let Some(&replacement) = replacements.get(cond) {
                    *cond = replacement;
                }
            }
            crate::cfg_mir::Terminator::JumpTable { predicate, .. } => {
                if let Some(&replacement) = replacements.get(predicate) {
                    *predicate = replacement;
                }
            }
            _ => {}
        }
    }

    // Replace in edge arguments
    for edge in &mut func.edges {
        for arg in &mut edge.args {
            if let Some(&replacement) = replacements.get(&arg.source) {
                arg.source = replacement;
            }
            if let Some(&replacement) = replacements.get(&arg.target) {
                arg.target = replacement;
            }
        }
    }

    // Replace in function results
    for result in &mut func.data_results {
        if let Some(&replacement) = replacements.get(result) {
            *result = replacement;
        }
    }
}

#[cfg(test)]
mod tests {
    // TODO: Add tests once we have a CFG-MIR builder helper.
    // Test cases:
    // 1. Loop header with 6 const params -> eliminated
    // 2. Loop header with 2 duplicate params -> eliminated
    // 3. Loop header with variant params -> not eliminated
    // 4. Verify dominance check (invariant value must dominate header)
}
