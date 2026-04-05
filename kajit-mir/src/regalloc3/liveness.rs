//! Liveness analysis for register allocation.
//!
//! ## Problem
//!
//! Register allocation needs to know:
//! - Which vregs are live at each program point
//! - Live intervals (can have holes!)
//! - Use points (for spill heuristics)
//!
//! ## Challenges with Block Parameters
//!
//! Block params are PHI nodes in SSA form:
//! - Block param = DEF at block entry
//! - Edge arg = USE on predecessor edge (not at terminator!)
//!
//! Example:
//! ```text
//! b0:
//!   br b2(v1)  ← v1 is USE on edge b0→b2
//!
//! b1:
//!   br b2(v2)  ← v2 is USE on edge b1→b2
//!
//! b2(v3):      ← v3 is DEF at b2 entry
//!   ...
//! ```
//!
//! Standard liveness (live-out at terminator) is WRONG for edge args.
//! We need edge-specific liveness.
//!
//! ## Algorithm (Backward Dataflow)
//!
//! 1. Build use/def sets per instruction and per block
//! 2. Compute live-in/live-out for each block (fixpoint)
//! 3. Build live intervals with program points
//! 4. Handle holes (dead regions within interval)
//!
//! ## Edge Liveness
//!
//! For each edge (pred → succ):
//! - Edge args are live at END of pred block
//! - But NOT necessarily live throughout pred
//! - Must track edge-specific liveness separately

use indexmap::IndexMap;
use kajit_ir::VReg;
use std::collections::{BTreeSet, HashMap};

use crate::cfg_mir::{BlockId, Function, InstId};

use super::progpoint::{LiveInterval, ProgPoint, ProgPointMap};

/// Liveness information for a function
#[derive(Debug)]
pub struct LivenessInfo {
    /// Live intervals for each vreg (IndexMap for deterministic iteration)
    pub intervals: IndexMap<VReg, LiveInterval>,

    /// Live-in set for each block (BTreeSet for deterministic iteration)
    pub live_in: HashMap<BlockId, BTreeSet<VReg>>,

    /// Live-out set for each block (BTreeSet for deterministic iteration)
    pub live_out: HashMap<BlockId, BTreeSet<VReg>>,
}

/// Compute liveness information
pub fn compute_liveness(func: &Function, progpoints: &ProgPointMap) -> LivenessInfo {
    let mut analyzer = LivenessAnalyzer::new(func, progpoints);
    analyzer.analyze();
    analyzer.build_intervals()
}

/// Liveness analysis state
struct LivenessAnalyzer<'a> {
    func: &'a Function,
    progpoints: &'a ProgPointMap,

    /// Use set per instruction (vregs read)
    inst_uses: HashMap<InstId, Vec<VReg>>,

    /// Def set per instruction (vregs written)
    inst_defs: HashMap<InstId, Vec<VReg>>,

    /// Live-in set per block
    live_in: HashMap<BlockId, BTreeSet<VReg>>,

    /// Live-out set per block
    live_out: HashMap<BlockId, BTreeSet<VReg>>,
}

impl<'a> LivenessAnalyzer<'a> {
    fn new(func: &'a Function, progpoints: &'a ProgPointMap) -> Self {
        let mut inst_uses = HashMap::new();
        let mut inst_defs = HashMap::new();

        // Collect use/def for each instruction from LinearOp fields
        // (operands may be incomplete for some instructions)
        for inst in &func.insts {
            use kajit_lir::LinearOp;

            let defs = match &inst.op {
                LinearOp::Const { dst, .. }
                | LinearOp::BinOp { dst, .. }
                | LinearOp::UnaryOp { dst, .. }
                | LinearOp::Copy { dst, .. }
                | LinearOp::ReadBytes { dst, .. }
                | LinearOp::PeekByte { dst }
                | LinearOp::SaveCursor { dst }
                | LinearOp::SaveInputEnd { dst }
                | LinearOp::ReadFromField { dst, .. }
                | LinearOp::SaveOutPtr { dst }
                | LinearOp::SlotAddr { dst, .. }
                | LinearOp::LoadFromAddr { dst, .. }
                | LinearOp::ReadFromSlot { dst, .. }
                | LinearOp::CallPure { dst, .. }
                | LinearOp::CallEffect { dst, .. } => vec![*dst],
                LinearOp::CallIntrinsic { dst, .. } => {
                    if let Some(dst) = dst {
                        vec![*dst]
                    } else {
                        vec![]
                    }
                }
                _ => vec![],
            };

            let uses = match &inst.op {
                LinearOp::BinOp { lhs, rhs, .. } => vec![*lhs, *rhs],
                LinearOp::UnaryOp { src, .. }
                | LinearOp::Copy { src, .. }
                | LinearOp::AdvanceCursorBy { src }
                | LinearOp::RestoreCursor { src }
                | LinearOp::WriteToField { src, .. }
                | LinearOp::SetOutPtr { src }
                | LinearOp::WriteToSlot { src, .. } => vec![*src],
                LinearOp::StoreToAddr { addr, src, .. } => vec![*addr, *src],
                LinearOp::LoadFromAddr { addr, .. } => vec![*addr],
                LinearOp::CallIntrinsic { args, .. }
                | LinearOp::CallPure { args, .. }
                | LinearOp::CallEffect { args, .. }
                | LinearOp::CallLambda { args, .. } => args.clone(),
                _ => vec![],
            };

            inst_uses.insert(inst.id, uses);
            inst_defs.insert(inst.id, defs);
        }

        Self {
            func,
            progpoints,
            inst_uses,
            inst_defs,
            live_in: HashMap::new(),
            live_out: HashMap::new(),
        }
    }

    /// Run backward dataflow to fixpoint
    fn analyze(&mut self) {
        // Initialize all sets to empty
        for block in self.func.live_blocks() {
            self.live_in.insert(block.id, BTreeSet::new());
            self.live_out.insert(block.id, BTreeSet::new());
        }

        // Iterate to fixpoint
        let mut changed = true;
        while changed {
            changed = false;

            // Process blocks in reverse order (better convergence)
            for block in self.func.live_blocks().rev() {

                let old_live_in = self.live_in[&block.id].clone();

                // Compute live-out from successors
                let mut live_out = BTreeSet::new();
                for &edge_id in &block.succs {
                    let edge = &self.func.edges[edge_id.index()];
                    let succ = &self.func.blocks[edge.to.index()];

                    // Successor's live-in (minus block params, plus edge args)
                    let mut succ_live = self.live_in[&succ.id].clone();

                    // Remove block params (defined at succ entry)
                    for param in &succ.params {
                        succ_live.remove(param);
                    }

                    // Add edge args (used on this edge)
                    for arg in &edge.args {
                        succ_live.insert(arg.source);
                    }

                    live_out.extend(succ_live);
                }

                // Add terminator uses (condition vregs in branches)
                let term = &self.func.terms[block.term.0 as usize];
                match term {
                    crate::cfg_mir::Terminator::BranchIf { cond, .. }
                    | crate::cfg_mir::Terminator::BranchIfZero { cond, .. } => {
                        live_out.insert(*cond);
                    }
                    crate::cfg_mir::Terminator::Return => {
                        // data_results are live at the function return.
                        for &result in &self.func.data_results {
                            live_out.insert(result);
                        }
                    }
                    _ => {}
                }

                // Compute live-in from live-out
                let mut live_in = live_out.clone();

                // Process instructions in reverse order
                for &inst_id in block.insts.iter().rev() {
                    // live = (live - defs) ∪ uses
                    for &def in &self.inst_defs[&inst_id] {
                        live_in.remove(&def);
                    }
                    for &use_vreg in &self.inst_uses[&inst_id] {
                        live_in.insert(use_vreg);
                    }
                }

                // Remove block params (defined at block entry)
                for param in &block.params {
                    live_in.remove(param);
                }

                // Remove data_args (defined at function entry, like block params)
                if block.id == self.func.entry {
                    for &arg in &self.func.data_args {
                        live_in.remove(&arg);
                    }
                }

                // Update sets
                self.live_out.insert(block.id, live_out);
                if live_in != old_live_in {
                    self.live_in.insert(block.id, live_in);
                    changed = true;
                }
            }
        }
    }

    /// Build live intervals from liveness sets
    fn build_intervals(self) -> LivenessInfo {
        let mut intervals: IndexMap<VReg, Vec<(ProgPoint, ProgPoint)>> = IndexMap::new();
        let mut use_points: IndexMap<VReg, Vec<ProgPoint>> = IndexMap::new();

        // For each block, track live ranges
        for block in self.func.live_blocks() {
            let block_entry = self.progpoints.block_entry[&block.id];
            let block_exit = self.progpoints.block_exit[&block.id];

            // Vregs live at block exit
            let mut live = self.live_out[&block.id].clone();

            // Process instructions in reverse
            for &inst_id in block.insts.iter().rev() {
                let inst_point = self.progpoints.inst_to_point[&inst_id];

                // Defs kill liveness
                for &def in &self.inst_defs[&inst_id] {
                    if live.contains(&def) {
                        // Live after def, add segment
                        intervals
                            .entry(def)
                            .or_insert_with(Vec::new)
                            .push((inst_point, block_exit));
                        live.remove(&def);
                    } else {
                        // Def not live after - still need interval for the def point
                        // (dead code, but still needs a register)
                        intervals
                            .entry(def)
                            .or_insert_with(Vec::new)
                            .push((inst_point, inst_point));
                    }
                }

                // Uses add liveness
                for &use_vreg in &self.inst_uses[&inst_id] {
                    live.insert(use_vreg);
                    use_points
                        .entry(use_vreg)
                        .or_insert_with(Vec::new)
                        .push(inst_point);
                }
            }

            // Vregs still live at block entry
            for vreg in live {
                intervals
                    .entry(vreg)
                    .or_insert_with(Vec::new)
                    .push((block_entry, block_exit));
            }

            // Block params are defined at entry
            for param in &block.params {
                intervals
                    .entry(*param)
                    .or_insert_with(Vec::new)
                    .push((block_entry, block_exit));
            }
        }

        // Convert to LiveInterval structs — sort by vreg index for determinism
        intervals.sort_by(|a, _, b, _| a.index().cmp(&b.index()));
        let mut live_intervals = IndexMap::new();
        for (vreg, mut segments) in intervals {
            // Sort and merge overlapping segments
            segments.sort_by_key(|&(start, _)| start);

            // Merge all segments into one contiguous interval (min start, max end).
            // In SSA form without phi copies, vregs live across block boundaries
            // have segments in non-adjacent prog point ranges (especially across
            // back-edges where the successor has lower prog points than the
            // predecessor). A single contiguous interval is conservative but correct.
            let min_start = segments.iter().map(|s| s.0).min().unwrap();
            let max_end = segments.iter().map(|s| s.1).max().unwrap();
            let merged = vec![(min_start, max_end)];

            let uses = use_points.get(&vreg).cloned().unwrap_or_default();
            live_intervals.insert(
                vreg,
                LiveInterval {
                    vreg,
                    segments: merged,
                    uses,
                },
            );
        }

        LivenessInfo {
            intervals: live_intervals,
            live_in: self.live_in,
            live_out: self.live_out,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cfg_mir::{
        Block, Clobbers, Edge, EdgeArg, EdgeId, Function, Inst, Operand, OperandKind, RegClass,
        Terminator,
    };
    use kajit_lir::LinearOp;

    #[test]
    fn test_simple_liveness() {
        // Block with one instruction: v1 = const 42
        // v1 should not be live-in (defined in block)
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
                insts: vec![InstId(0)],
                term: crate::cfg_mir::TermId(0),
                preds: vec![],
                succs: vec![],
                dead: false,
            }],
            edges: vec![],
            insts: vec![Inst {
                id: InstId(0),
                op: LinearOp::Const {
                    dst: VReg::new(1),
                    value: 42,
                },
                operands: vec![],
                clobbers: Clobbers::default(),
            }],
            terms: vec![Terminator::Return],
        };

        let progpoints = ProgPointMap::build(&func);
        let liveness = compute_liveness(&func, &progpoints);

        // v1 should have a live interval
        assert!(liveness.intervals.contains_key(&VReg::new(1)));

        // v1 should not be live-in (defined in block)
        assert!(!liveness.live_in[&BlockId(0)].contains(&VReg::new(1)));
    }

    #[test]
    fn test_use_liveness() {
        // v0 is used by instruction, should be live-in
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
                insts: vec![InstId(0)],
                term: crate::cfg_mir::TermId(0),
                preds: vec![],
                succs: vec![],
                dead: false,
            }],
            edges: vec![],
            insts: vec![Inst {
                id: InstId(0),
                op: LinearOp::Copy {
                    dst: VReg::new(1),
                    src: VReg::new(0),
                },
                operands: vec![Operand {
                    vreg: VReg::new(0),
                    kind: OperandKind::Use,
                    class: RegClass::Gpr,
                    fixed: None,
                }],
                clobbers: Clobbers::default(),
            }],
            terms: vec![Terminator::Return],
        };

        let progpoints = ProgPointMap::build(&func);
        let liveness = compute_liveness(&func, &progpoints);

        // v0 should be live-in (used but not defined)
        assert!(liveness.live_in[&BlockId(0)].contains(&VReg::new(0)));
    }

    #[test]
    fn test_block_param_liveness() {
        // b0 -> b1(v0)
        // b1(v1): ...
        // v0 should be live-out of b0 (edge arg)
        // v1 should be defined in b1 (block param)
        let func = Function {
            id: crate::cfg_mir::FunctionId(0),
            lambda_id: kajit_ir::LambdaId::new(0),
            entry: BlockId(0),
            data_args: vec![],
            data_results: vec![],
            output_size: 0,
            blocks: vec![
                Block {
                    id: BlockId(0),
                    params: vec![],
                    insts: vec![],
                    term: crate::cfg_mir::TermId(0),
                    preds: vec![],
                    succs: vec![EdgeId(0)],
                    dead: false,
                },
                Block {
                    id: BlockId(1),
                    params: vec![VReg::new(1)],
                    insts: vec![],
                    term: crate::cfg_mir::TermId(1),
                    preds: vec![EdgeId(0)],
                    succs: vec![],
                    dead: false,
                },
            ],
            edges: vec![Edge {
                id: EdgeId(0),
                from: BlockId(0),
                to: BlockId(1),
                args: vec![EdgeArg {
                    target: VReg::new(1),
                    source: VReg::new(0),
                }],
            }],
            insts: vec![],
            terms: vec![Terminator::Branch { edge: EdgeId(0) }, Terminator::Return],
        };

        let progpoints = ProgPointMap::build(&func);
        let liveness = compute_liveness(&func, &progpoints);

        // v0 should be live-out of b0 (edge arg)
        assert!(liveness.live_out[&BlockId(0)].contains(&VReg::new(0)));

        // v1 should NOT be live-in of b1 (defined by block param)
        assert!(!liveness.live_in[&BlockId(1)].contains(&VReg::new(1)));
    }
}
