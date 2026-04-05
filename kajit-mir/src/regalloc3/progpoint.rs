//! Program point model for liveness analysis.
//!
//! ## Problem
//!
//! `InstId` is a stable identity, not a program order. Live intervals need:
//! - Total ordering over instructions
//! - Ability to represent holes (dead regions)
//! - Entry/exit points for blocks
//!
//! ## Solution
//!
//! Derive dense `ProgPoint` numbering per-function, with gaps between points
//! to allow spill insertion without renumbering.

use std::collections::HashMap;

use kajit_ir::VReg;

use crate::cfg_mir::{BlockId, Function, InstId};

/// Program point (derived fresh per function, NOT InstId)
///
/// This is a dense ordering over instructions within a function.
/// Program points are derived in reverse postorder (RPO) traversal.
#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ProgPoint(pub u32);

impl ProgPoint {
    pub fn new(n: u32) -> Self {
        Self(n)
    }

    pub fn index(self) -> u32 {
        self.0
    }

    /// Next program point (for iteration)
    pub fn next(self) -> Self {
        Self(self.0 + 1)
    }

    /// Previous program point (for iteration)
    pub fn prev(self) -> Self {
        Self(self.0.saturating_sub(1))
    }
}

/// Live interval (can have holes!)
///
/// Unlike a simple (start, end) pair, intervals can have multiple segments
/// to represent holes where the value is dead.
#[derive(Debug, Clone)]
pub struct LiveInterval {
    /// Virtual register
    pub vreg: VReg,

    /// Live segments (can have gaps)
    /// Each segment is (start, end) inclusive
    /// Sorted by start point, non-overlapping
    pub segments: Vec<(ProgPoint, ProgPoint)>,

    /// All use points (for spill heuristics)
    pub uses: Vec<ProgPoint>,
}

impl LiveInterval {
    /// First program point where this interval is live
    pub fn start(&self) -> ProgPoint {
        self.segments[0].0
    }

    /// Last program point where this interval is live
    pub fn end(&self) -> ProgPoint {
        self.segments.last().unwrap().1
    }

    /// Is this interval live at given point?
    pub fn covers(&self, point: ProgPoint) -> bool {
        self.segments
            .iter()
            .any(|&(start, end)| point >= start && point <= end)
    }

    /// Next use point after given point (for spill victim selection)
    pub fn next_use_after(&self, point: ProgPoint) -> Option<ProgPoint> {
        self.uses
            .iter()
            .copied()
            .find(|&use_point| use_point > point)
    }
}

/// Map from program order to stable IDs (for reconstruction)
///
/// This allows us to derive a dense ordering for liveness analysis
/// while still working with stable IDs in the rest of the allocator.
#[derive(Debug)]
pub struct ProgPointMap {
    /// ProgPoint -> InstId (for reconstruction)
    pub point_to_inst: Vec<InstId>,

    /// InstId -> ProgPoint (for lookup)
    pub inst_to_point: HashMap<InstId, ProgPoint>,

    /// Block entry points (before first instruction)
    pub block_entry: HashMap<BlockId, ProgPoint>,

    /// Block exit points (after last instruction)
    pub block_exit: HashMap<BlockId, ProgPoint>,

    /// Total number of program points
    pub num_points: u32,
}

impl ProgPointMap {
    /// Build program points from CFG-MIR function
    ///
    /// Algorithm:
    /// 1. Traverse blocks in reverse postorder (RPO)
    /// 2. Assign program points with gaps (for spill insertion)
    /// 3. Record entry/exit points for each block
    ///
    /// Gaps: We leave 2-point gaps between instructions to allow
    /// spill/reload insertion without renumbering.
    pub fn build(func: &Function) -> Self {
        let mut point = 0u32;
        let mut point_to_inst = Vec::new();
        let mut inst_to_point = HashMap::new();
        let mut block_entry = HashMap::new();
        let mut block_exit = HashMap::new();

        // Compute reverse postorder (RPO)
        let rpo = compute_rpo(func);

        for &block_id in &rpo {
            let block = &func.blocks[block_id.index()];

            if block.dead {
                continue; // skip dead blocks
            }

            // Block entry point (before first instruction)
            block_entry.insert(block_id, ProgPoint(point));
            point += 2; // gap for potential spills

            // Instructions in block
            for &inst_id in &block.insts {
                point_to_inst.push(inst_id);
                inst_to_point.insert(inst_id, ProgPoint(point));
                point += 2; // gap between instructions
            }

            // Block exit point (after last instruction)
            block_exit.insert(block_id, ProgPoint(point));
            point += 2; // gap before next block
        }

        ProgPointMap {
            point_to_inst,
            inst_to_point,
            block_entry,
            block_exit,
            num_points: point,
        }
    }

    /// Get instruction ID for program point (if it exists)
    pub fn inst_at(&self, point: ProgPoint) -> Option<InstId> {
        // Program points are sparse (we have gaps)
        // Need to find the actual instruction at or before this point
        let idx = (point.0 / 2) as usize; // divide by 2 since we have gaps
        self.point_to_inst.get(idx).copied()
    }

    /// Iterate over all program points with their instructions
    pub fn iter(&self) -> impl Iterator<Item = (ProgPoint, InstId)> + '_ {
        self.point_to_inst
            .iter()
            .enumerate()
            .map(|(idx, &inst_id)| {
                let point = ProgPoint((idx as u32) * 2 + 2); // account for gaps
                (point, inst_id)
            })
    }
}

/// Compute reverse postorder (RPO) of blocks
///
/// RPO is the standard ordering for forward dataflow analysis.
/// It ensures we visit dominators before dominated blocks.
fn compute_rpo(func: &Function) -> Vec<BlockId> {
    let mut rpo = Vec::new();
    let mut visited = vec![false; func.blocks.len()];

    fn postorder(func: &Function, block_id: BlockId, visited: &mut [bool], rpo: &mut Vec<BlockId>) {
        if visited[block_id.index()] {
            return;
        }
        visited[block_id.index()] = true;

        // Visit successors first (postorder)
        for &edge_id in &func.blocks[block_id.index()].succs {
            let succ = func.edges[edge_id.index()].to;
            postorder(func, succ, visited, rpo);
        }

        // Then add this block
        rpo.push(block_id);
    }

    // Start from entry block
    postorder(func, func.entry, &mut visited, &mut rpo);

    // Reverse to get reverse postorder
    rpo.reverse();
    rpo
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cfg_mir::{Block, Function, Inst, Terminator};

    #[test]
    fn test_progpoint_ordering() {
        let p1 = ProgPoint(0);
        let p2 = ProgPoint(10);
        let p3 = ProgPoint(20);

        assert!(p1 < p2);
        assert!(p2 < p3);
        assert!(p1 < p3);
    }

    #[test]
    fn test_live_interval_covers() {
        let interval = LiveInterval {
            vreg: VReg::new(0),
            segments: vec![
                (ProgPoint(10), ProgPoint(20)),
                (ProgPoint(30), ProgPoint(40)),
            ],
            uses: vec![ProgPoint(15), ProgPoint(35)],
        };

        // Inside first segment
        assert!(interval.covers(ProgPoint(15)));

        // Inside second segment
        assert!(interval.covers(ProgPoint(35)));

        // In hole between segments
        assert!(!interval.covers(ProgPoint(25)));

        // Before interval
        assert!(!interval.covers(ProgPoint(5)));

        // After interval
        assert!(!interval.covers(ProgPoint(45)));
    }

    #[test]
    fn test_live_interval_next_use() {
        let interval = LiveInterval {
            vreg: VReg::new(0),
            segments: vec![(ProgPoint(0), ProgPoint(30))],
            uses: vec![ProgPoint(5), ProgPoint(15), ProgPoint(25)],
        };

        assert_eq!(interval.next_use_after(ProgPoint(0)), Some(ProgPoint(5)));
        assert_eq!(interval.next_use_after(ProgPoint(5)), Some(ProgPoint(15)));
        assert_eq!(interval.next_use_after(ProgPoint(15)), Some(ProgPoint(25)));
        assert_eq!(interval.next_use_after(ProgPoint(25)), None);
    }

    #[test]
    fn test_progpoint_map_simple() {
        // Build a simple function with one block, two instructions
        use kajit_ir::VReg;
        use kajit_lir::LinearOp;

        let func = Function {
            id: crate::cfg_mir::FunctionId(0),
            lambda_id: kajit_ir::LambdaId::new(0),
            entry: BlockId(0),
            data_args: vec![],
            data_results: vec![],
            output_size: 0,
            blocks: vec![Block {
                id: BlockId(0),
                params: vec![],
                insts: vec![InstId(0), InstId(1)],
                term: crate::cfg_mir::TermId(0),
                preds: vec![],
                succs: vec![],
                dead: false,
            }],
            edges: vec![],
            insts: vec![
                Inst {
                    id: InstId(0),
                    op: LinearOp::Const {
                        dst: VReg::new(0),
                        value: 42,
                    },
                    operands: vec![],
                    clobbers: crate::cfg_mir::Clobbers::default(),
                },
                Inst {
                    id: InstId(1),
                    op: LinearOp::Const {
                        dst: VReg::new(1),
                        value: 99,
                    },
                    operands: vec![],
                    clobbers: crate::cfg_mir::Clobbers::default(),
                },
            ],
            terms: vec![Terminator::Return],
        };

        let progpoints = ProgPointMap::build(&func);

        // Check that we have points for both instructions
        assert_eq!(progpoints.point_to_inst.len(), 2);

        // Check entry/exit points
        assert!(progpoints.block_entry.contains_key(&BlockId(0)));
        assert!(progpoints.block_exit.contains_key(&BlockId(0)));

        // Check instruction mapping
        assert!(progpoints.inst_to_point.contains_key(&InstId(0)));
        assert!(progpoints.inst_to_point.contains_key(&InstId(1)));

        // Check ordering: entry < inst0 < inst1 < exit
        let entry = progpoints.block_entry[&BlockId(0)];
        let inst0 = progpoints.inst_to_point[&InstId(0)];
        let inst1 = progpoints.inst_to_point[&InstId(1)];
        let exit = progpoints.block_exit[&BlockId(0)];

        assert!(entry < inst0);
        assert!(inst0 < inst1);
        assert!(inst1 < exit);
    }
}
