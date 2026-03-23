//! Loop detection and analysis for CFG-MIR.
//!
//! Identifies natural loops using dominance analysis:
//! - A backedge is an edge B -> H where H dominates B
//! - A loop is the set of blocks that can reach the backedge without going through H

use std::collections::{HashMap, HashSet, VecDeque};

use crate::cfg_mir::{BlockId, EdgeId, Function};

use super::dominance::DominanceInfo;

/// Loop information for a function.
#[derive(Debug, Clone)]
pub struct LoopInfo {
    /// Map from loop header to loop data
    loops: HashMap<BlockId, LoopData>,
    /// Map from block to innermost loop containing it
    block_to_loop: HashMap<BlockId, BlockId>,
}

/// Information about a single loop.
#[derive(Debug, Clone)]
pub struct LoopData {
    /// The loop header (target of backedge)
    pub header: BlockId,
    /// All blocks in the loop body (including header)
    pub body: HashSet<BlockId>,
    /// Backedges into the header (edges B -> header where header dominates B)
    pub backedges: Vec<EdgeId>,
    /// Exit edges (edges from loop body to outside)
    pub exits: Vec<EdgeId>,
    /// Nesting depth (0 = outermost)
    pub depth: usize,
}

impl LoopInfo {
    /// Compute loop information for a function.
    ///
    /// Requires dominance information to identify backedges.
    pub fn compute(func: &Function, dom: &DominanceInfo) -> Self {
        let mut loops: HashMap<BlockId, LoopData> = HashMap::new();

        // Step 1: Find backedges (edges B -> H where H dominates B)
        let mut backedges: HashMap<BlockId, Vec<EdgeId>> = HashMap::new();
        for edge in &func.edges {
            if dom.dominates(edge.to, edge.from) {
                // This is a backedge: edge.to is the loop header
                backedges.entry(edge.to).or_default().push(edge.id);
            }
        }

        // Step 2: For each loop header, compute the loop body
        for (&header, &ref edges) in &backedges {
            let body = compute_loop_body(func, header, edges);
            let exits = compute_loop_exits(func, &body);

            loops.insert(
                header,
                LoopData {
                    header,
                    body,
                    backedges: edges.clone(),
                    exits,
                    depth: 0, // will be computed below
                },
            );
        }

        // Step 3: Compute nesting depth
        compute_loop_depths(&mut loops);

        // Step 4: Build block -> innermost loop map
        let block_to_loop = build_block_to_loop_map(&loops);

        Self {
            loops,
            block_to_loop,
        }
    }

    /// Returns true if the given block is a loop header.
    pub fn is_loop_header(&self, block: BlockId) -> bool {
        self.loops.contains_key(&block)
    }

    /// Returns the loop data for the given header, if it's a loop header.
    pub fn loop_for_header(&self, header: BlockId) -> Option<&LoopData> {
        self.loops.get(&header)
    }

    /// Returns the innermost loop containing the given block, if any.
    pub fn loop_for_block(&self, block: BlockId) -> Option<&LoopData> {
        self.block_to_loop
            .get(&block)
            .and_then(|&header| self.loops.get(&header))
    }

    /// Returns all loop headers in the function.
    pub fn loop_headers(&self) -> impl Iterator<Item = BlockId> + '_ {
        self.loops.keys().copied()
    }
}

/// Compute the loop body for a loop with the given header and backedges.
///
/// The loop body is the set of blocks that can reach a backedge source
/// without going through the header (except for the header itself).
fn compute_loop_body(func: &Function, header: BlockId, backedges: &[EdgeId]) -> HashSet<BlockId> {
    let mut body = HashSet::new();
    body.insert(header);

    // Start from backedge sources and work backwards
    let mut worklist: VecDeque<BlockId> = VecDeque::new();
    for &edge_id in backedges {
        let edge = &func.edges[edge_id.index()];
        let source = edge.from;
        if source != header && body.insert(source) {
            worklist.push_back(source);
        }
    }

    // Backwards reachability from backedge sources
    while let Some(block_id) = worklist.pop_front() {
        let block = &func.blocks[block_id.index()];
        for &pred_edge_id in &block.preds {
            let pred_block = func.edges[pred_edge_id.index()].from;
            if pred_block != header && body.insert(pred_block) {
                worklist.push_back(pred_block);
            }
        }
    }

    body
}

/// Compute exit edges for a loop body.
///
/// An exit edge is an edge from a block in the loop to a block outside the loop.
fn compute_loop_exits(func: &Function, body: &HashSet<BlockId>) -> Vec<EdgeId> {
    let mut exits = Vec::new();

    for &block_id in body {
        let block = &func.blocks[block_id.index()];
        for &succ_edge_id in &block.succs {
            let edge = &func.edges[succ_edge_id.index()];
            if !body.contains(&edge.to) {
                exits.push(edge.id);
            }
        }
    }

    exits
}

/// Compute nesting depths for all loops.
///
/// A loop's depth is 1 + the maximum depth of any loop nested inside it.
fn compute_loop_depths(loops: &mut HashMap<BlockId, LoopData>) {
    // Build parent relationship: child header -> parent header
    let mut parents: HashMap<BlockId, BlockId> = HashMap::new();

    for (&child_header, child_data) in loops.iter() {
        // Find the innermost loop that contains this loop's header (excluding itself)
        let mut parent = None;
        let mut min_body_size = usize::MAX;

        for (&candidate_header, candidate_data) in loops.iter() {
            if candidate_header == child_header {
                continue;
            }
            // If candidate contains child_header, it's a potential parent
            if candidate_data.body.contains(&child_header) {
                // Pick the one with smallest body (innermost)
                if candidate_data.body.len() < min_body_size {
                    min_body_size = candidate_data.body.len();
                    parent = Some(candidate_header);
                }
            }
        }

        if let Some(parent_header) = parent {
            parents.insert(child_header, parent_header);
        }
    }

    // Compute depths bottom-up
    fn compute_depth(
        header: BlockId,
        loops: &mut HashMap<BlockId, LoopData>,
        parents: &HashMap<BlockId, BlockId>,
        computed: &mut HashMap<BlockId, usize>,
    ) -> usize {
        if let Some(&depth) = computed.get(&header) {
            return depth;
        }

        let depth = if let Some(&parent) = parents.get(&header) {
            compute_depth(parent, loops, parents, computed) + 1
        } else {
            0 // Top-level loop
        };

        if let Some(loop_data) = loops.get_mut(&header) {
            loop_data.depth = depth;
        }
        computed.insert(header, depth);
        depth
    }

    let headers: Vec<BlockId> = loops.keys().copied().collect();
    let mut computed = HashMap::new();
    for header in headers {
        compute_depth(header, loops, &parents, &mut computed);
    }
}

/// Build a map from each block to the header of the innermost loop containing it.
fn build_block_to_loop_map(loops: &HashMap<BlockId, LoopData>) -> HashMap<BlockId, BlockId> {
    let mut map = HashMap::new();

    // For each block that's in any loop, find the innermost one
    for (&header, loop_data) in loops {
        for &block in &loop_data.body {
            // Only update if this loop is deeper (more nested) than current assignment
            let current_depth = map
                .get(&block)
                .and_then(|&h| loops.get(&h))
                .map(|d| d.depth)
                .unwrap_or(0);

            if loop_data.depth >= current_depth {
                map.insert(block, header);
            }
        }
    }

    map
}

// TODO: Add tests once CFG-MIR construction API is more stable.
// Tests should verify:
// 1. Simple loop detection
// 2. Nested loop depths
// 3. Loop body computation
