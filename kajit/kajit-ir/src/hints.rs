//! RVSDG hint analysis for register allocation.
//!
//! ## Phase 2: Spill Cost from Loop Structure
//!
//! Analyzes RVSDG to identify loop-carried values (theta entry ports).
//! These are expensive to spill because they're used in every iteration.
//!
//! ## Algorithm
//!
//! 1. Traverse RVSDG, find all theta nodes
//! 2. For each theta:
//!    - Compute nesting depth (how many parent thetas)
//!    - Mark entry ports as loop-carried
//!    - Assign spill cost: nested → High, single → Medium
//! 3. Build hint map: output port → spill cost
//!
//! ## Threading Through Lowering
//!
//! RVSDG output ports → LIR vregs → CFG-MIR vregs
//! Each lowering stage must preserve hint metadata.

use std::collections::HashMap;

use kajit_reprs::ir::{IrFunc, NodeId, NodeKind, RegionId};

/// Spill cost hint (matches `kajit-mir-legacy::regalloc3::hints::SpillCost`)
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum SpillCost {
    Low,    // Not loop-carried
    Medium, // Single loop
    High,   // Nested loop
}

/// Hint for an output port
#[derive(Debug, Clone, Copy)]
pub struct PortHint {
    pub spill_cost: SpillCost,
}

/// Output identifier: (node_index, output_index)
pub type OutputId = (usize, usize);

/// Hint map: output ID → hint
pub type PortHintMap = HashMap<OutputId, PortHint>;

/// Analyze RVSDG to compute spill cost hints
///
/// TODO: Proper implementation when integrating with lowering pipeline.
/// For now, returns empty hints (all vregs get default Low cost).
pub fn analyze_spill_costs(_func: &IrFunc) -> PortHintMap {
    // TODO: Traverse func starting from func.root
    // - Visit each node recursively
    // - Track nesting depth
    // - Assign SpillCost::Medium/High to theta outputs based on nesting
    PortHintMap::new()
}

// Stub implementation - will be completed with lowering integration
#[allow(dead_code)]
struct SpillCostAnalyzer<'a> {
    func: &'a IrFunc,
    hints: &'a mut PortHintMap,
    nesting_depth: u32,
}

#[allow(dead_code)]
impl<'a> SpillCostAnalyzer<'a> {
    /// Analyze a region (recursively analyzes nodes)
    fn analyze_region(&mut self, region: RegionId) {
        for node_id in &self.func.regions[region].nodes {
            self.analyze_node(*node_id);
        }
    }

    /// Analyze a node
    fn analyze_node(&mut self, node_id: NodeId) {
        let node = &self.func.nodes[node_id];

        match &node.kind {
            NodeKind::Theta { body, .. } => {
                // Theta entry ports are loop-carried
                let spill_cost = if self.nesting_depth > 0 {
                    SpillCost::High // Nested loop
                } else {
                    SpillCost::Medium // Single loop
                };

                // Mark all theta outputs as loop-carried
                for (output_idx, _) in node.outputs.iter().enumerate() {
                    self.hints
                        .insert((node_id.index(), output_idx), PortHint { spill_cost });
                }

                // Recursively analyze body (with increased nesting)
                self.nesting_depth += 1;
                self.analyze_region(*body);
                self.nesting_depth -= 1;
            }

            NodeKind::Gamma { regions, .. } => {
                // Gamma outputs are not loop-carried (branches, not loops)
                // Recursively analyze regions
                for region in regions {
                    self.analyze_region(*region);
                }
            }

            NodeKind::Lambda { body, .. } => {
                // Recursively analyze lambda body
                self.analyze_region(*body);
            }

            NodeKind::Apply { .. } => {
                // Apply node outputs are not loop-carried
                // Mark as Low cost (default)
                for (output_idx, _) in node.outputs.iter().enumerate() {
                    self.hints
                        .entry((node_id.index(), output_idx))
                        .or_insert(PortHint {
                            spill_cost: SpillCost::Low,
                        });
                }
            }

            // All other nodes: default to Low cost
            _ => {
                for (output_idx, _) in node.outputs.iter().enumerate() {
                    self.hints
                        .entry((node_id.index(), output_idx))
                        .or_insert(PortHint {
                            spill_cost: SpillCost::Low,
                        });
                }
            }
        }
    }
}

// TODO: Add integration tests when connecting to lowering pipeline
// Tests require proper RVSDG construction API
