//! Dead code elimination for CFG-MIR.
//!
//! Currently focuses on dead block param elimination: removes block params
//! (phi nodes) that have no downstream uses.

use std::collections::BTreeSet;

use crate::cfg_mir::Function;
use kajit_ir::VReg;

/// Eliminate dead block params and their corresponding edge args.
///
/// A block param is dead if no instruction, terminator, or edge arg source
/// uses its vreg. When a dead param is removed, all edge args targeting it
/// are also removed.
pub fn eliminate_dead_block_params(func: &mut Function) {
    // Collect all used vregs
    let mut used: BTreeSet<VReg> = BTreeSet::new();

    for block in &func.blocks {
        if block.dead {
            continue;
        }
        // Instruction uses
        for &inst_id in &block.insts {
            let inst = &func.insts[inst_id.0 as usize];
            inst.op.for_each_use(|src| {
                used.insert(*src);
            });
        }
        // Terminator uses
        let term = &func.terms[block.term.0 as usize];
        match term {
            crate::cfg_mir::Terminator::BranchIf { cond, .. }
            | crate::cfg_mir::Terminator::BranchIfZero { cond, .. } => {
                used.insert(*cond);
            }
            crate::cfg_mir::Terminator::JumpTable { predicate, .. } => {
                used.insert(*predicate);
            }
            _ => {}
        }
        // Edge arg sources
        for &edge_id in &block.succs {
            let edge = &func.edges[edge_id.index()];
            if edge.from.0 == u32::MAX {
                continue;
            }
            for arg in &edge.args {
                used.insert(arg.source);
            }
        }
    }
    // Data results (function outputs)
    for &vreg in &func.data_results {
        used.insert(vreg);
    }

    // Find dead block params
    let mut changed = true;
    while changed {
        changed = false;
        for block_idx in 0..func.blocks.len() {
            if func.blocks[block_idx].dead {
                continue;
            }
            let block_id = func.blocks[block_idx].id;

            // Find dead params (not in used set)
            let dead_param_indices: Vec<usize> = func.blocks[block_idx]
                .params
                .iter()
                .enumerate()
                .filter(|(_, param)| !used.contains(param))
                .map(|(i, _)| i)
                .collect();

            if dead_param_indices.is_empty() {
                continue;
            }

            // Remove dead params from the block
            let dead_set: BTreeSet<usize> = dead_param_indices.iter().copied().collect();
            let mut new_params = Vec::new();
            for (i, &param) in func.blocks[block_idx].params.iter().enumerate() {
                if !dead_set.contains(&i) {
                    new_params.push(param);
                }
            }
            func.blocks[block_idx].params = new_params;

            // Remove corresponding edge args from all incoming edges
            for pred_block in &func.blocks {
                if pred_block.dead {
                    continue;
                }
                for &edge_id in &pred_block.succs {
                    let edge = &mut func.edges[edge_id.index()];
                    if edge.to != block_id || edge.from.0 == u32::MAX {
                        continue;
                    }
                    let mut new_args = Vec::new();
                    for (i, arg) in edge.args.iter().enumerate() {
                        if !dead_set.contains(&i) {
                            new_args.push(*arg);
                        }
                    }
                    edge.args = new_args;
                }
            }

            changed = true;

            // Recompute used set (removing edge arg sources for dead params
            // may make other vregs unused)
            used.clear();
            for block in &func.blocks {
                if block.dead {
                    continue;
                }
                for &inst_id in &block.insts {
                    let inst = &func.insts[inst_id.0 as usize];
                    inst.op.for_each_use(|src| {
                        used.insert(*src);
                    });
                }
                let term = &func.terms[block.term.0 as usize];
                match term {
                    crate::cfg_mir::Terminator::BranchIf { cond, .. }
                    | crate::cfg_mir::Terminator::BranchIfZero { cond, .. } => {
                        used.insert(*cond);
                    }
                    crate::cfg_mir::Terminator::JumpTable { predicate, .. } => {
                        used.insert(*predicate);
                    }
                    _ => {}
                }
                for &edge_id in &block.succs {
                    let edge = &func.edges[edge_id.index()];
                    if edge.from.0 == u32::MAX {
                        continue;
                    }
                    for arg in &edge.args {
                        used.insert(arg.source);
                    }
                }
            }
            for &vreg in &func.data_results {
                used.insert(vreg);
            }

            break; // restart after modification
        }
    }
}
