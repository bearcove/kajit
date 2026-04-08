//! Constant phi / redundant block parameter elimination.
//!
//! Based on Cranelift's `remove_constant_phis` pass.
//!
//! Uses iterative dataflow analysis to find block parameters that always
//! receive the same value from all predecessors. If a parameter converges
//! to exactly one non-param value, it's redundant and can be eliminated.
//!
//! Algorithm:
//! 1. Initialize lattice: None (⊥) → One(V) → Many (⊤)
//! 2. For each block param, compute fixed point by looking at all incoming edges
//! 3. If all edges pass same value V (not a param) → One(V)
//! 4. If edges disagree or recursive → Many
//! 5. Eliminate params that converge to One(V)

use std::collections::HashMap;

use kajit_ir::VReg;

use kajit_reprs::mir::{BlockId, Function};

/// Value lattice for dataflow analysis.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ParamValue {
    /// Haven't analyzed yet (⊥)
    None,
    /// Converged to single value (not a block param)
    One(VReg),
    /// Multiple different values (⊤)
    Many,
}

impl ParamValue {
    /// Merge two values in the lattice.
    fn merge(self, other: Self) -> Self {
        match (self, other) {
            (ParamValue::None, x) | (x, ParamValue::None) => x,
            (ParamValue::Many, _) | (_, ParamValue::Many) => ParamValue::Many,
            (ParamValue::One(v1), ParamValue::One(v2)) => {
                if v1 == v2 {
                    ParamValue::One(v1)
                } else {
                    ParamValue::Many
                }
            }
        }
    }
}

/// Eliminate redundant block parameters.
///
/// A block parameter is redundant if all incoming edges pass the same
/// non-param value. Uses iterative dataflow to handle mutually-dependent params.
///
/// Returns true if any parameters were eliminated.
pub fn eliminate_constant_phis(func: &mut Function) -> bool {
    let debug = std::env::var("KAJIT_DEBUG_CONST_PHI").is_ok();

    if debug {
        eprintln!("[const_phi] Starting constant phi elimination");
    }

    // Compute dominance for checking if replacements dominate blocks
    let dom = crate::analysis::dominance::DominanceInfo::compute(func);

    // Build map from (block, param_index) to analysis value
    let param_values = analyze_block_params(func, debug);

    // Find params to eliminate
    let mut to_eliminate: HashMap<BlockId, Vec<(usize, VReg)>> = HashMap::new();

    for block in &func.blocks {
        if block.dead || block.id == func.entry {
            continue; // Skip entry block and dead blocks
        }

        for (param_idx, &param_vreg) in block.params.iter().enumerate() {
            let key = (block.id, param_idx);
            if let Some(&ParamValue::One(replacement)) = param_values.get(&key) {
                // Verify replacement is actually defined
                if !is_vreg_defined(func, replacement) {
                    if debug {
                        eprintln!(
                            "[const_phi] b{} param {} (v{}) → v{} SKIPPED (replacement not defined)",
                            block.id.index(),
                            param_idx,
                            param_vreg.index(),
                            replacement.index()
                        );
                    }
                    continue;
                }

                // Verify replacement dominates this block (if it's an instruction result)
                if let Some(def_block) = find_def_block_for_vreg(func, replacement)
                    && !dom.dominates(def_block, block.id)
                {
                    if debug {
                        eprintln!(
                            "[const_phi] b{} param {} (v{}) → v{} SKIPPED (def in b{} doesn't dominate)",
                            block.id.index(),
                            param_idx,
                            param_vreg.index(),
                            replacement.index(),
                            def_block.index()
                        );
                    }
                    continue;
                }
                // If no def_block, it's a function arg which dominates everything

                // This param always gets the same value - eliminate it
                if debug {
                    eprintln!(
                        "[const_phi] b{} param {} (v{}) → v{}",
                        block.id.index(),
                        param_idx,
                        param_vreg.index(),
                        replacement.index()
                    );
                }
                to_eliminate
                    .entry(block.id)
                    .or_default()
                    .push((param_idx, replacement));
            }
        }
    }

    if to_eliminate.is_empty() {
        if debug {
            eprintln!("[const_phi] No redundant params found");
        }
        return false;
    }

    // Apply eliminations
    for (block_id, eliminations) in to_eliminate {
        apply_eliminations(func, block_id, eliminations, debug);
    }

    if debug {
        eprintln!("[const_phi] Eliminated redundant params");
    }

    true
}

/// Analyze all block parameters using iterative dataflow.
fn analyze_block_params(func: &Function, debug: bool) -> HashMap<(BlockId, usize), ParamValue> {
    let mut values: HashMap<(BlockId, usize), ParamValue> = HashMap::new();

    // Initialize: all params start at None
    for block in &func.blocks {
        if block.dead || block.id == func.entry {
            continue;
        }
        for (param_idx, _) in block.params.iter().enumerate() {
            values.insert((block.id, param_idx), ParamValue::None);
        }
    }

    // Iterate to fixed point
    let mut changed = true;
    let mut iteration = 0;
    while changed {
        changed = false;
        iteration += 1;

        if debug && iteration > 1 {
            eprintln!("[const_phi] Iteration {}", iteration);
        }

        for block in &func.blocks {
            if block.dead || block.id == func.entry {
                continue;
            }

            for (param_idx, &param_vreg) in block.params.iter().enumerate() {
                let key = (block.id, param_idx);
                let old_value = values[&key];

                // Compute new value by looking at all incoming edges
                let new_value = analyze_param(func, &values, block.id, param_idx, param_vreg);

                if new_value != old_value {
                    values.insert(key, new_value);
                    changed = true;

                    if debug {
                        eprintln!(
                            "[const_phi]   b{} param {}: {:?} → {:?}",
                            block.id.index(),
                            param_idx,
                            old_value,
                            new_value
                        );
                    }
                }
            }
        }
    }

    if debug {
        eprintln!("[const_phi] Converged after {} iterations", iteration);
    }

    values
}

/// Analyze a single block parameter.
fn analyze_param(
    func: &Function,
    values: &HashMap<(BlockId, usize), ParamValue>,
    block_id: BlockId,
    param_idx: usize,
    _param_vreg: VReg,
) -> ParamValue {
    let block = &func.blocks[block_id.index()];
    let mut result = ParamValue::None;

    // Look at all incoming edges
    for &pred_edge_id in &block.preds {
        let edge = &func.edges[pred_edge_id.index()];

        if param_idx >= edge.args.len() {
            // Edge doesn't provide this param - treat as Many
            return ParamValue::Many;
        }

        let source_vreg = edge.args[param_idx].source;

        // Is this source a block parameter of the predecessor?
        let source_value = if let Some(value) = resolve_vreg_value(func, values, source_vreg) {
            value
        } else {
            // Source is not a param, or is a param we haven't resolved yet
            ParamValue::One(source_vreg)
        };

        result = result.merge(source_value);

        if result == ParamValue::Many {
            break; // Short-circuit
        }
    }

    result
}

/// Resolve a vreg to its dataflow value.
///
/// If vreg is a block parameter, return its analyzed value.
/// Otherwise, return One(vreg) since it's a concrete value.
fn resolve_vreg_value(
    func: &Function,
    values: &HashMap<(BlockId, usize), ParamValue>,
    vreg: VReg,
) -> Option<ParamValue> {
    // Check if this vreg is a block parameter
    for block in &func.blocks {
        if block.dead {
            continue;
        }
        for (param_idx, &param) in block.params.iter().enumerate() {
            if param == vreg {
                let key = (block.id, param_idx);
                return Some(*values.get(&key).unwrap_or(&ParamValue::None));
            }
        }
    }

    None // Not a block param
}

/// Check if a vreg is defined (as function arg, block param, or instruction result).
fn is_vreg_defined(func: &Function, vreg: VReg) -> bool {
    // Check function arguments
    if func.data_args.contains(&vreg) {
        return true;
    }

    // Check block parameters
    for block in &func.blocks {
        if block.dead {
            continue;
        }
        if block.params.contains(&vreg) {
            return true;
        }
    }

    // Check instruction definitions
    for inst in &func.insts {
        for operand in &inst.operands {
            if operand.vreg == vreg && operand.kind == kajit_reprs::mir::OperandKind::Def {
                return true;
            }
        }
    }

    false
}

/// Find the block that defines a vreg (as param or instruction).
/// Returns None if vreg is a function arg (dominates everything).
fn find_def_block_for_vreg(func: &Function, vreg: VReg) -> Option<BlockId> {
    // Function args dominate everything
    if func.data_args.contains(&vreg) {
        return None;
    }

    // Check block parameters
    for block in &func.blocks {
        if block.dead {
            continue;
        }
        if block.params.contains(&vreg) {
            return Some(block.id);
        }
    }

    // Check instruction definitions
    for block in &func.blocks {
        if block.dead {
            continue;
        }
        for &inst_id in &block.insts {
            let inst = &func.insts[inst_id.index()];
            for operand in &inst.operands {
                if operand.vreg == vreg && operand.kind == kajit_reprs::mir::OperandKind::Def {
                    return Some(block.id);
                }
            }
        }
    }

    None
}

/// Apply parameter eliminations to a block.
fn apply_eliminations(
    func: &mut Function,
    block_id: BlockId,
    mut eliminations: Vec<(usize, VReg)>,
    debug: bool,
) {
    // Sort by param_idx descending so removals don't invalidate indices
    eliminations.sort_by(|a, b| b.0.cmp(&a.0));

    if debug {
        eprintln!(
            "[const_phi] Applying {} eliminations to b{}",
            eliminations.len(),
            block_id.index()
        );
    }

    // Build replacement map for this block
    let mut replacements: HashMap<VReg, VReg> = HashMap::new();
    for &(param_idx, replacement) in &eliminations {
        let param_vreg = func.blocks[block_id.index()].params[param_idx];
        replacements.insert(param_vreg, replacement);
    }

    // Replace uses of eliminated params in this function
    replace_vregs_in_function(func, &replacements);

    // Remove params from block
    let block = &mut func.blocks[block_id.index()];
    for &(param_idx, _replacement) in &eliminations {
        block.params.remove(param_idx);
    }

    // Remove corresponding args from all incoming edges
    for &pred_edge_id in &func.blocks[block_id.index()].preds.clone() {
        let edge = &mut func.edges[pred_edge_id.index()];
        for &(param_idx, _replacement) in &eliminations {
            if param_idx < edge.args.len() {
                edge.args.remove(param_idx);
            }
        }
    }
}

/// Replace vregs throughout the function.
pub fn replace_vregs_in_function(func: &mut Function, replacements: &HashMap<VReg, VReg>) {
    if replacements.is_empty() {
        return;
    }

    // Replace in instructions (both operands and LinearOp fields)
    for inst in &mut func.insts {
        for operand in &mut inst.operands {
            if let Some(&replacement) = replacements.get(&operand.vreg) {
                operand.vreg = replacement;
            }
        }
        inst.op.for_each_vreg_mut(|v| {
            if let Some(&replacement) = replacements.get(v) {
                *v = replacement;
            }
        });
    }

    // Replace in terminators
    for term in &mut func.terms {
        match term {
            kajit_reprs::mir::Terminator::BranchIf { cond, .. }
            | kajit_reprs::mir::Terminator::BranchIfZero { cond, .. } => {
                if let Some(&replacement) = replacements.get(cond) {
                    *cond = replacement;
                }
            }
            kajit_reprs::mir::Terminator::JumpTable { predicate, .. } => {
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
        }
    }
}
