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

use std::collections::HashMap;

use kajit_ir::VReg;
use kajit_lir::LinearOp;

use crate::{
    analysis::{dominance::DominanceInfo, loops::LoopInfo},
    cfg_mir::{BlockId, EdgeArg, Function},
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
        if eliminate_invariant_phis_in_header(func, dom, header, debug) {
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
    header: BlockId,
    debug: bool,
) -> bool {
    let header_block = &func.blocks[header.index()];
    let params = header_block.params.clone();

    if params.is_empty() {
        return false;
    }

    // Collect incoming edges to this header
    let incoming_edges: Vec<_> = header_block.preds.iter().map(|&eid| eid).collect();

    if incoming_edges.is_empty() {
        return false;
    }

    // For each parameter, check if it's loop-invariant
    let mut invariant_params: HashMap<usize, VReg> = HashMap::new();

    for (param_idx, &param_vreg) in params.iter().enumerate() {
        // Collect all values assigned to this parameter from incoming edges
        let mut incoming_values: Vec<VReg> = Vec::new();

        for &edge_id in &incoming_edges {
            let edge = &func.edges[edge_id.index()];
            if param_idx < edge.args.len() {
                incoming_values.push(edge.args[param_idx].source);
            } else {
                // Edge doesn't provide this parameter - not invariant
                incoming_values.clear();
                break;
            }
        }

        if incoming_values.is_empty() {
            continue;
        }

        // Check if all incoming values are identical (or equal to param itself)
        let first_value = incoming_values[0];
        let all_same = incoming_values.iter().all(|&v| {
            v == first_value || v == param_vreg // Self-reference is OK
        });

        if all_same && first_value != param_vreg {
            // This parameter is loop-invariant!
            // Verify that the invariant value dominates the header
            if let Some(def_block) = find_def_block(func, first_value) {
                if dom.dominates(def_block, header) {
                    invariant_params.insert(param_idx, first_value);
                }
            } else {
                // Function argument or unknown def - assume it dominates
                invariant_params.insert(param_idx, first_value);
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

    // Update all incoming edges to remove corresponding arguments
    for &edge_id in &incoming_edges {
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
                if operand.vreg == vreg && operand.kind == crate::cfg_mir::OperandKind::Def {
                    return Some(block.id);
                }
            }
        }
    }
    None
}

/// Replace vregs inside a LinearOp.
fn replace_vregs_in_linear_op(op: &mut LinearOp, replacements: &HashMap<VReg, VReg>) {
    // Helper to replace a single vreg
    let replace = |v: &mut VReg| {
        if let Some(&repl) = replacements.get(v) {
            *v = repl;
        }
    };

    match op {
        LinearOp::Copy { dst, src } => {
            replace(dst);
            replace(src);
        }
        LinearOp::Const { dst, .. } => {
            replace(dst);
        }
        LinearOp::BinOp { dst, lhs, rhs, .. } => {
            replace(dst);
            replace(lhs);
            replace(rhs);
        }
        LinearOp::UnaryOp { dst, src, .. } => {
            replace(dst);
            replace(src);
        }
        LinearOp::LoadFromAddr { dst, addr, .. } => {
            replace(dst);
            replace(addr);
        }
        LinearOp::StoreToAddr { addr, src, .. } => {
            replace(addr);
            replace(src);
        }
        LinearOp::ReadBytes { dst, .. }
        | LinearOp::PeekByte { dst }
        | LinearOp::SaveCursor { dst }
        | LinearOp::SaveInputEnd { dst }
        | LinearOp::ReadFromField { dst, .. }
        | LinearOp::SlotAddr { dst, .. }
        | LinearOp::SaveOutPtr { dst }
        | LinearOp::ReadFromSlot { dst, .. } => {
            replace(dst);
        }
        LinearOp::RestoreCursor { src }
        | LinearOp::AdvanceCursorBy { src }
        | LinearOp::SetOutPtr { src }
        | LinearOp::WriteToSlot { src, .. }
        | LinearOp::WriteToField { src, .. } => {
            replace(src);
        }
        LinearOp::CallIntrinsic { args, dst, .. } => {
            for arg in args {
                replace(arg);
            }
            if let Some(d) = dst {
                replace(d);
            }
        }
        LinearOp::CallPure { args, dst, .. } => {
            for arg in args {
                replace(arg);
            }
            replace(dst);
        }
        LinearOp::CallLambda { args, results, .. } => {
            for arg in args {
                replace(arg);
            }
            for result in results {
                replace(result);
            }
        }
        LinearOp::BranchIf {
            cond,
            phi_args,
            fallthrough_phi_args,
            ..
        }
        | LinearOp::BranchIfZero {
            cond,
            phi_args,
            fallthrough_phi_args,
            ..
        } => {
            replace(cond);
            for (src, tgt) in phi_args {
                replace(src);
                replace(tgt);
            }
            for (src, tgt) in fallthrough_phi_args {
                replace(src);
                replace(tgt);
            }
        }
        LinearOp::Branch { phi_args, .. } => {
            for (src, tgt) in phi_args {
                replace(src);
                replace(tgt);
            }
        }
        LinearOp::JumpTable { predicate, .. } => {
            replace(predicate);
        }
        LinearOp::SimdStringScan { pos, kind } => {
            replace(pos);
            replace(kind);
        }
        LinearOp::FuncStart {
            data_args,
            data_results,
            ..
        } => {
            for arg in data_args {
                replace(arg);
            }
            for result in data_results {
                replace(result);
            }
        }
        // No vregs to replace
        LinearOp::Label(_)
        | LinearOp::BoundsCheck { .. }
        | LinearOp::AdvanceCursor { .. }
        | LinearOp::SimdWhitespaceSkip
        | LinearOp::ErrorExit { .. }
        | LinearOp::FuncEnd => {}
    }
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
            replace_vregs_in_linear_op(&mut inst.op, &replacements);

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
