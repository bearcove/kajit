//! Dominance analysis for CFG-MIR.
//!
//! Computes dominance relationships between basic blocks:
//! - Block A dominates block B if every path from entry to B goes through A
//! - Used for safe phi elimination and SSA maintenance

use std::collections::{HashMap, HashSet};

use crate::cfg_mir::{BlockId, Function};

/// Dominance information for a function.
#[derive(Debug, Clone)]
pub struct DominanceInfo {
    /// Immediate dominator for each block (idom[B] = A means A immediately dominates B)
    idom: HashMap<BlockId, BlockId>,
    /// Dominance tree: children of each block in the dominator tree
    dom_tree: HashMap<BlockId, Vec<BlockId>>,
    /// Dominance frontier: blocks where dominance of X ends
    /// (used for SSA construction, not needed for our optimizations yet)
    dom_frontier: HashMap<BlockId, HashSet<BlockId>>,
}

impl DominanceInfo {
    /// Compute dominance information for a function.
    ///
    /// Uses the Cooper-Harvey-Kennedy algorithm (simple dataflow).
    pub fn compute(func: &Function) -> Self {
        let entry = func.entry;
        let blocks: Vec<BlockId> = func.blocks.iter().map(|b| b.id).collect();

        if blocks.is_empty() {
            return Self {
                idom: HashMap::new(),
                dom_tree: HashMap::new(),
                dom_frontier: HashMap::new(),
            };
        }

        // Compute reverse postorder for better convergence
        let rpo = reverse_postorder(func, entry);
        let rpo_index: HashMap<BlockId, usize> = rpo
            .iter()
            .enumerate()
            .map(|(idx, &bid)| (bid, idx))
            .collect();

        // Initialize: idom[entry] = entry, idom[others] = undefined
        let mut idom: HashMap<BlockId, BlockId> = HashMap::new();
        idom.insert(entry, entry);

        // Iterate to fixed point
        let mut changed = true;
        while changed {
            changed = false;

            for &block_id in &rpo {
                if block_id == entry {
                    continue;
                }

                // Find predecessors that have been processed
                let block = &func.blocks[block_id.index()];
                let processed_preds: Vec<BlockId> = block
                    .preds
                    .iter()
                    .map(|&edge_id| func.edges[edge_id.index()].from)
                    .filter(|&pred| idom.contains_key(&pred))
                    .collect();

                if processed_preds.is_empty() {
                    continue; // Unreachable block
                }

                // Compute new idom: intersection of all predecessors' dominator paths
                let mut new_idom = processed_preds[0];
                for &pred in &processed_preds[1..] {
                    new_idom = intersect(&idom, &rpo_index, new_idom, pred);
                }

                if idom.get(&block_id) != Some(&new_idom) {
                    idom.insert(block_id, new_idom);
                    changed = true;
                }
            }
        }

        // Build dominance tree
        let mut dom_tree: HashMap<BlockId, Vec<BlockId>> = HashMap::new();
        for &block_id in &blocks {
            if block_id == entry {
                continue;
            }
            if let Some(&parent) = idom.get(&block_id)
                && parent != block_id
            {
                dom_tree.entry(parent).or_default().push(block_id);
            }
        }

        // Compute dominance frontiers (for completeness, though not used yet)
        let dom_frontier = compute_dominance_frontiers(func, &idom);

        Self {
            idom,
            dom_tree,
            dom_frontier,
        }
    }

    /// Returns true if block A dominates block B.
    ///
    /// A dominates B if every path from entry to B goes through A.
    /// Note: A dominates itself.
    pub fn dominates(&self, a: BlockId, b: BlockId) -> bool {
        if a == b {
            return true;
        }

        let mut current = b;
        while let Some(&idom) = self.idom.get(&current) {
            if idom == current {
                // Reached entry or cycle
                break;
            }
            if idom == a {
                return true;
            }
            current = idom;
        }
        false
    }

    /// Returns true if block A strictly dominates block B.
    ///
    /// A strictly dominates B if A dominates B and A != B.
    pub fn strictly_dominates(&self, a: BlockId, b: BlockId) -> bool {
        a != b && self.dominates(a, b)
    }

    /// Returns the immediate dominator of a block, if any.
    pub fn immediate_dominator(&self, block: BlockId) -> Option<BlockId> {
        self.idom.get(&block).copied().filter(|&idom| idom != block)
    }

    /// Returns the children of a block in the dominance tree.
    pub fn dominator_tree_children(&self, block: BlockId) -> &[BlockId] {
        self.dom_tree
            .get(&block)
            .map(|v| v.as_slice())
            .unwrap_or(&[])
    }

    /// Returns the dominance frontier of a block.
    ///
    /// The dominance frontier of X is the set of blocks Y where:
    /// - X dominates a predecessor of Y
    /// - X does not strictly dominate Y
    pub fn dominance_frontier(&self, block: BlockId) -> impl Iterator<Item = BlockId> + '_ {
        self.dom_frontier
            .get(&block)
            .map(|s| s.iter().copied())
            .into_iter()
            .flatten()
    }
}

/// Compute reverse postorder traversal.
///
/// Visiting blocks in RPO gives better convergence for dataflow.
fn reverse_postorder(func: &Function, entry: BlockId) -> Vec<BlockId> {
    let mut visited = HashSet::new();
    let mut postorder = Vec::new();

    fn dfs(
        func: &Function,
        block: BlockId,
        visited: &mut HashSet<BlockId>,
        postorder: &mut Vec<BlockId>,
    ) {
        if !visited.insert(block) {
            return;
        }

        let block_data = &func.blocks[block.index()];
        for &edge_id in &block_data.succs {
            let edge = &func.edges[edge_id.index()];
            dfs(func, edge.to, visited, postorder);
        }

        postorder.push(block);
    }

    dfs(func, entry, &mut visited, &mut postorder);
    postorder.reverse();
    postorder
}

/// Find the common dominator of two blocks.
///
/// Walks up the dominator tree from both blocks until they meet.
fn intersect(
    idom: &HashMap<BlockId, BlockId>,
    rpo_index: &HashMap<BlockId, usize>,
    mut b1: BlockId,
    mut b2: BlockId,
) -> BlockId {
    while b1 != b2 {
        while rpo_index[&b1] > rpo_index[&b2] {
            b1 = idom[&b1];
        }
        while rpo_index[&b2] > rpo_index[&b1] {
            b2 = idom[&b2];
        }
    }
    b1
}

/// Compute dominance frontiers for all blocks.
fn compute_dominance_frontiers(
    func: &Function,
    idom: &HashMap<BlockId, BlockId>,
) -> HashMap<BlockId, HashSet<BlockId>> {
    let mut df: HashMap<BlockId, HashSet<BlockId>> = HashMap::new();

    for block in &func.blocks {
        let preds: Vec<BlockId> = block
            .preds
            .iter()
            .map(|&eid| func.edges[eid.index()].from)
            .collect();

        if preds.len() >= 2 {
            // Block is a join point
            for &pred in &preds {
                let mut runner = pred;
                while runner != idom.get(&block.id).copied().unwrap_or(block.id) {
                    df.entry(runner).or_default().insert(block.id);
                    if let Some(&next) = idom.get(&runner) {
                        if next == runner {
                            break; // Entry node
                        }
                        runner = next;
                    } else {
                        break;
                    }
                }
            }
        }
    }

    df
}

// TODO: Add tests once CFG-MIR construction API is more stable.
// Tests should verify:
// 1. Basic dominance in diamond CFG
// 2. Strict dominance
// 3. Dominance frontiers
