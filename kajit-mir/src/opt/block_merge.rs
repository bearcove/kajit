//! Empty block merging.
//!
//! Merges empty forwarding blocks into their predecessors to simplify CFG.
//!
//! An empty block is one with:
//! - No instructions
//! - Single predecessor
//! - Predecessor has single successor
//!
//! Example:
//! ```text
//! block b1:
//!   ...
//!   branch -> b2
//! block b2 (empty):
//!   branch -> b3
//! ```
//! Becomes:
//! ```text
//! block b1:
//!   ...
//!   branch -> b3
//! ```

use std::collections::{HashMap, HashSet};

use crate::cfg_mir::{BlockId, EdgeId, Function, TermId, Terminator};

/// Merge empty forwarding blocks into their predecessors.
///
/// Returns true if any changes were made.
pub fn merge_empty_blocks(func: &mut Function) -> bool {
    let debug = std::env::var("KAJIT_DEBUG_BLOCK_MERGE").is_ok();
    let mut total_merged = 0;

    loop {
        let merge_candidate = find_merge_candidate(func);

        if let Some((empty_block, pred_block)) = merge_candidate {
            if debug {
                eprintln!(
                    "[block_merge] merging empty block b{} into b{}",
                    empty_block.index(),
                    pred_block.index()
                );
            }

            merge_blocks(func, empty_block, pred_block);
            total_merged += 1;
        } else {
            break;
        }
    }

    if debug && total_merged > 0 {
        eprintln!("[block_merge] merged {} empty blocks", total_merged);
    }

    // Physically remove dead blocks immediately to avoid index remapping issues
    if total_merged > 0 {
        remove_unreachable_blocks(func);

        // Validate CFG consistency after removal
        if let Err(errors) = crate::opt::validate::validate_cfg(func) {
            eprintln!("CFG VALIDATION FAILED after block merging:");
            for error in &errors {
                eprintln!("  - {}", error);
            }
            panic!("CFG validation failed with {} errors", errors.len());
        }
    }

    total_merged > 0
}

/// Find a candidate pair of blocks to merge.
///
/// Returns (empty_block, predecessor) if found.
fn find_merge_candidate(func: &Function) -> Option<(BlockId, BlockId)> {
    for block in &func.blocks {
        // Skip dead blocks (tombstones)
        if block.dead {
            continue;
        }

        // Skip blocks that are already dead (no predecessors, not entry)
        if block.preds.is_empty() && block.id != func.entry {
            continue;
        }

        // Skip non-empty blocks
        if !block.insts.is_empty() {
            continue;
        }

        // Skip blocks with parameters (complex to merge)
        if !block.params.is_empty() {
            continue;
        }

        // Must have exactly one predecessor
        if block.preds.len() != 1 {
            continue;
        }

        // Must have exactly one successor (simple forwarding)
        if block.succs.len() != 1 {
            continue;
        }

        let pred_edge_id = block.preds[0];
        let pred_edge = &func.edges[pred_edge_id.index()];
        let pred_block = pred_edge.from;

        // Don't merge if edge has arguments (would need to handle phi parameters)
        if !pred_edge.args.is_empty() {
            continue;
        }

        // Don't merge if successor edge has arguments
        let succ_edge_id = block.succs[0];
        let succ_edge = &func.edges[succ_edge_id.index()];
        if !succ_edge.args.is_empty() {
            continue;
        }

        // Found a candidate!
        return Some((block.id, pred_block));
    }

    None
}

/// Merge an empty block into its predecessor.
///
/// Assumes:
/// - empty_block has no instructions
/// - empty_block has no parameters
/// - empty_block has exactly one predecessor (pred_block)
/// - pred_block has exactly one successor (empty_block)
/// - The edge between them has no arguments
fn merge_blocks(func: &mut Function, empty_block: BlockId, _pred_block: BlockId) {
    let debug = std::env::var("KAJIT_DEBUG_BLOCK_MERGE").is_ok();

    // Get data from empty block BEFORE modifying anything
    let empty = &func.blocks[empty_block.index()];
    let edge_into_empty = empty.preds[0]; // pred → empty
    let edge_out_of_empty = empty.succs[0]; // empty → succ

    let final_target = func.edges[edge_out_of_empty.index()].to;

    // Get pred block info before modifying
    let pred_block_idx = func.edges[edge_into_empty.index()].from;

    if debug {
        let pred_term = &func.terms[func.blocks[pred_block_idx.index()].term.index()];
        eprintln!(
            "  retargeting: edge e{} (b{} -> b{}) to point to b{} instead",
            edge_into_empty.index(),
            pred_block_idx.index(),
            empty_block.index(),
            final_target.index()
        );
        eprintln!("    pred terminator: {:?}", pred_term);
        eprintln!(
            "    edge args: {:?}",
            func.edges[edge_into_empty.index()].args
        );
        eprintln!(
            "    final_target params: {:?}",
            func.blocks[final_target.index()].params
        );
    }

    // Strategy: Retarget edge_into_empty to point to final_target
    // This way pred's terminator doesn't need to change (it already references edge_into_empty)

    // Step 1: Update the edge to point to the final target
    func.edges[edge_into_empty.index()].to = final_target;

    // Step 2: Remove edge_into_empty from the old target (empty block)'s preds
    let empty_mut = &mut func.blocks[empty_block.index()];
    empty_mut.preds.retain(|&e| e != edge_into_empty);

    // Step 3: Update final_target's predecessor list
    // Replace edge_out_of_empty with edge_into_empty
    let succ_block = &mut func.blocks[final_target.index()];
    let mut found = false;
    for pred_edge in &mut succ_block.preds {
        if *pred_edge == edge_out_of_empty {
            if debug {
                eprintln!(
                    "    updating b{} preds: e{} → e{}",
                    final_target.index(),
                    edge_out_of_empty.index(),
                    edge_into_empty.index()
                );
            }
            *pred_edge = edge_into_empty;
            found = true;
            break;
        }
    }
    if debug && !found {
        eprintln!(
            "    WARNING: edge e{} not found in b{}'s preds: {:?}",
            edge_out_of_empty.index(),
            final_target.index(),
            succ_block
                .preds
                .iter()
                .map(|e| e.index())
                .collect::<Vec<_>>()
        );
    }

    // Step 4: Update pred's successor list
    // If pred had edge_into_empty in its succs, it's already there (no change needed)
    // The edge ID stays the same, just points to a different block now

    // Step 5: Mark empty block as dead and clear preds so remove_unreachable_blocks finds it
    let empty_block_mut = &mut func.blocks[empty_block.index()];
    empty_block_mut.dead = true;
    // Preds already removed in step 2, but clear succs too for consistency
    empty_block_mut.succs.clear();
}

/// Remove unreachable blocks (blocks with no predecessors, except entry).
pub fn remove_unreachable_blocks(func: &mut Function) -> bool {
    let debug = std::env::var("KAJIT_DEBUG_BLOCK_MERGE").is_ok();

    // Collect unreachable blocks (no predecessors, and not the entry block)
    let mut unreachable: Vec<BlockId> = func
        .blocks
        .iter()
        .filter(|b| b.preds.is_empty() && b.id != func.entry)
        .map(|b| b.id)
        .collect();

    if unreachable.is_empty() {
        return false;
    }

    if debug {
        eprintln!(
            "[block_merge] removing {} unreachable blocks: {:?}",
            unreachable.len(),
            unreachable.iter().map(|b| b.index()).collect::<Vec<_>>()
        );
        eprintln!(
            "[block_merge] entry block before removal: b{}",
            func.entry.index()
        );
    }

    // Build a mapping from old BlockId to new BlockId
    let unreachable_set: HashSet<BlockId> = unreachable.iter().copied().collect();

    let mut old_to_new: HashMap<BlockId, BlockId> = HashMap::new();
    let mut new_idx = 0;

    for block in &func.blocks {
        if !unreachable_set.contains(&block.id) {
            old_to_new.insert(block.id, BlockId::new(new_idx));
            new_idx += 1;
        }
    }

    // Filter out unreachable blocks
    let new_blocks: Vec<_> = func
        .blocks
        .iter()
        .filter(|b| !unreachable_set.contains(&b.id))
        .cloned()
        .collect();

    // Update block IDs in the remaining blocks
    let mut updated_blocks = Vec::new();
    for mut block in new_blocks {
        block.id = old_to_new[&block.id];
        updated_blocks.push(block);
    }

    func.blocks = updated_blocks;

    // Update entry block ID
    func.entry = old_to_new[&func.entry];

    // Update edge from/to references and build edge ID remapping
    // Note: edges referencing unreachable blocks will be removed by filtering
    let mut valid_edges = Vec::new();
    let mut old_edge_to_new: HashMap<EdgeId, EdgeId> = HashMap::new();

    for (old_idx, edge) in func.edges.iter().enumerate() {
        if let (Some(&new_from), Some(&new_to)) =
            (old_to_new.get(&edge.from), old_to_new.get(&edge.to))
        {
            let new_edge_idx = valid_edges.len();
            let new_edge_id = EdgeId::new(new_edge_idx as u32);
            old_edge_to_new.insert(EdgeId::new(old_idx as u32), new_edge_id);

            let mut updated_edge = edge.clone();
            updated_edge.id = new_edge_id; // Update the edge's ID to match its new position
            updated_edge.from = new_from;
            updated_edge.to = new_to;
            valid_edges.push(updated_edge);
        }
    }
    func.edges = valid_edges;

    // Update edge IDs in blocks using the remapping
    for block in &mut func.blocks {
        block.preds = block
            .preds
            .iter()
            .filter_map(|&eid| old_edge_to_new.get(&eid).copied())
            .collect();
        block.succs = block
            .succs
            .iter()
            .filter_map(|&eid| old_edge_to_new.get(&eid).copied())
            .collect();
    }

    // Update edge IDs in terminators
    for term in &mut func.terms {
        match term {
            crate::cfg_mir::Terminator::Branch { edge } => {
                if let Some(&new_edge) = old_edge_to_new.get(edge) {
                    *edge = new_edge;
                }
            }
            crate::cfg_mir::Terminator::BranchIf {
                taken, fallthrough, ..
            }
            | crate::cfg_mir::Terminator::BranchIfZero {
                taken, fallthrough, ..
            } => {
                if let Some(&new_edge) = old_edge_to_new.get(taken) {
                    *taken = new_edge;
                }
                if let Some(&new_edge) = old_edge_to_new.get(fallthrough) {
                    *fallthrough = new_edge;
                }
            }
            crate::cfg_mir::Terminator::JumpTable {
                targets, default, ..
            } => {
                for edge in targets {
                    if let Some(&new_edge) = old_edge_to_new.get(edge) {
                        *edge = new_edge;
                    }
                }
                if let Some(&new_edge) = old_edge_to_new.get(default) {
                    *default = new_edge;
                }
            }
            crate::cfg_mir::Terminator::Return | crate::cfg_mir::Terminator::ErrorExit { .. } => {}
        }
    }

    true
}

#[cfg(test)]
mod tests {
    // TODO: Add tests once CFG-MIR builder is available
    // Test cases:
    // 1. Simple chain: b0 -> b1(empty) -> b2 should become b0 -> b2
    // 2. Diamond with empty blocks
    // 3. Don't merge blocks with parameters
    // 4. Don't merge blocks with multiple predecessors
}
