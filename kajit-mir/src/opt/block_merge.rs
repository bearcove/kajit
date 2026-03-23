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

    total_merged > 0
}

/// Find a candidate pair of blocks to merge.
///
/// Returns (empty_block, predecessor) if found.
fn find_merge_candidate(func: &Function) -> Option<(BlockId, BlockId)> {
    for block in &func.blocks {
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

        let pred_edge_id = block.preds[0];
        let pred_edge = &func.edges[pred_edge_id.index()];
        let pred_block = pred_edge.from;

        // Predecessor must have this block as its only successor
        let pred = &func.blocks[pred_block.index()];
        if pred.succs.len() != 1 {
            continue;
        }

        // Don't merge if predecessor has edge arguments (would need to handle phi parameters)
        if !pred_edge.args.is_empty() {
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
fn merge_blocks(func: &mut Function, empty_block: BlockId, pred_block: BlockId) {
    // Get data from empty block BEFORE modifying anything
    let empty = &func.blocks[empty_block.index()];
    let empty_term_id = empty.term;
    let empty_succs = empty.succs.clone();
    let empty_pred_edge = empty.preds[0]; // We know it has exactly one pred

    // Update edges: change empty block's outgoing edges to come from pred instead
    for &succ_edge_id in &empty_succs {
        let edge = &mut func.edges[succ_edge_id.index()];
        edge.from = pred_block;
    }

    // Replace predecessor's terminator and successors with empty block's
    let pred = &mut func.blocks[pred_block.index()];
    pred.term = empty_term_id;
    pred.succs = empty_succs.clone();

    // Mark empty block as dead by clearing all its data
    // Create a new dead terminator for it
    let dead_term_idx = func.terms.len();
    func.terms.push(Terminator::Return);
    let dead_term_id = TermId::new(dead_term_idx as u32);

    let empty_block_mut = &mut func.blocks[empty_block.index()];
    empty_block_mut.preds.clear();
    empty_block_mut.succs.clear();
    empty_block_mut.params.clear();
    empty_block_mut.insts.clear();
    empty_block_mut.term = dead_term_id;
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

    // Update edge from/to references
    // Note: edges referencing unreachable blocks will be removed by filtering
    let mut valid_edges = Vec::new();
    for edge in &func.edges {
        if let (Some(&new_from), Some(&new_to)) =
            (old_to_new.get(&edge.from), old_to_new.get(&edge.to))
        {
            let mut updated_edge = edge.clone();
            updated_edge.from = new_from;
            updated_edge.to = new_to;
            valid_edges.push(updated_edge);
        }
    }
    func.edges = valid_edges;

    // Update edge IDs in blocks
    for block in &mut func.blocks {
        block.preds.retain(|eid| eid.index() < func.edges.len());
        block.succs.retain(|eid| eid.index() < func.edges.len());
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
