//! Def-use chain analysis for CFG-MIR.
//!
//! Tracks where each vreg is defined and where it's used:
//! - Definitions: instructions, block parameters, function arguments
//! - Uses: instruction operands, terminator operands, edge arguments

use std::collections::HashMap;

use kajit_ir::VReg;
use kajit_lir::LinearOp;

use kajit_reprs::mir::{BlockId, EdgeId, Function, InstId, OperandKind};

/// Def-use information for a function.
#[derive(Debug, Clone)]
pub struct DefUseInfo {
    /// For each vreg, where it's defined
    defs: HashMap<VReg, DefSite>,
    /// For each vreg, all use sites
    uses: HashMap<VReg, Vec<UseSite>>,
}

/// Where a vreg is defined.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DefSite {
    /// Defined by an instruction in a block
    Inst(BlockId, InstId),
    /// Defined as a block parameter
    BlockParam(BlockId, usize),
    /// Defined as a function argument
    FuncArg(usize),
}

/// Where a vreg is used.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct UseSite {
    /// Which block contains the use
    pub block: BlockId,
    /// Which instruction (None for terminator or edge argument)
    pub inst: Option<InstId>,
    /// Which edge (if used in edge argument)
    pub edge: Option<EdgeId>,
    /// Operand index (for instructions) or argument index (for edges)
    pub index: usize,
}

impl DefUseInfo {
    /// Compute def-use chains for a function.
    pub fn compute(func: &Function) -> Self {
        let mut defs: HashMap<VReg, DefSite> = HashMap::new();
        let mut uses: HashMap<VReg, Vec<UseSite>> = HashMap::new();

        // Collect function argument defs
        for (idx, &vreg) in func.data_args.iter().enumerate() {
            defs.insert(vreg, DefSite::FuncArg(idx));
        }

        // Collect block parameter defs and instruction defs/uses
        for block in &func.blocks {
            // Block parameters are defs
            for (idx, &vreg) in block.params.iter().enumerate() {
                defs.insert(vreg, DefSite::BlockParam(block.id, idx));
            }

            // Process instructions
            for &inst_id in &block.insts {
                let inst = &func.insts[inst_id.index()];

                // Record def from this instruction
                if let Some(dst) = inst.op.dst() {
                    defs.insert(dst, DefSite::Inst(block.id, inst_id));
                }

                // Record uses in this instruction
                for (idx, operand) in inst.operands.iter().enumerate() {
                    if operand.kind == OperandKind::Use {
                        uses.entry(operand.vreg).or_default().push(UseSite {
                            block: block.id,
                            inst: Some(inst_id),
                            edge: None,
                            index: idx,
                        });
                    }
                }
            }

            // Process terminator uses
            let term = &func.terms[block.term.index()];
            if let Some(vreg) = term.condition_vreg() {
                uses.entry(vreg).or_default().push(UseSite {
                    block: block.id,
                    inst: None,
                    edge: None,
                    index: 0,
                });
            }
        }

        // Collect edge argument uses
        for edge in &func.edges {
            for (idx, arg) in edge.args.iter().enumerate() {
                uses.entry(arg.source).or_default().push(UseSite {
                    block: edge.from,
                    inst: None,
                    edge: Some(edge.id),
                    index: idx,
                });
            }
        }

        Self { defs, uses }
    }

    /// Returns where the given vreg is defined, if known.
    pub fn def_of(&self, vreg: VReg) -> Option<DefSite> {
        self.defs.get(&vreg).copied()
    }

    /// Returns all use sites of the given vreg.
    pub fn uses_of(&self, vreg: VReg) -> &[UseSite] {
        self.uses.get(&vreg).map(|v| v.as_slice()).unwrap_or(&[])
    }

    /// Returns the single use of a vreg, if it has exactly one use.
    pub fn single_use(&self, vreg: VReg) -> Option<UseSite> {
        match self.uses.get(&vreg)?.as_slice() {
            [single] => Some(*single),
            _ => None,
        }
    }

    /// Returns true if the given vreg has no uses.
    pub fn is_unused(&self, vreg: VReg) -> bool {
        self.uses.get(&vreg).is_none_or(|v| v.is_empty())
    }

    /// Returns all vregs that are defined but never used.
    pub fn dead_vregs(&self) -> Vec<VReg> {
        self.defs
            .keys()
            .filter(|&&vreg| self.is_unused(vreg))
            .copied()
            .collect()
    }
}

/// Helper trait to extract destination vreg from LinearOp.
trait LinearOpDst {
    fn dst(&self) -> Option<VReg>;
}

impl LinearOpDst for LinearOp {
    fn dst(&self) -> Option<VReg> {
        match self {
            Self::Copy { dst, .. }
            | Self::Const { dst, .. }
            | Self::DataAddr { dst, .. }
            | Self::ExternAddr { dst, .. }
            | Self::BinOp { dst, .. }
            | Self::UnaryOp { dst, .. }
            | Self::LoadFromAddr { dst, .. }
            | Self::SlotAddr { dst, .. }
            | Self::StackAlloc { dst, .. }
            | Self::ReadFromSlot { dst, .. }
            | Self::CallPure { dst, .. }
            | Self::CallEffect { dst, .. } => Some(*dst),
            Self::CallIntrinsic { dst, .. } => *dst,
            // CallLambda has multiple results, not a single dst
            Self::CallLambda { .. } => None,
            // Control flow operations (shouldn't appear in block.insts, but handle for completeness)
            Self::Label(_)
            | Self::Branch { .. }
            | Self::BranchIf { .. }
            | Self::BranchIfZero { .. }
            | Self::JumpTable { .. } => None,
            // Function structure markers
            Self::FuncStart { .. } | Self::FuncEnd => None,
            // Operations with no destination
            Self::StoreToAddr { .. } | Self::WriteToSlot { .. } => None,
        }
    }
}

/// Helper trait to extract condition vreg from Terminator.
trait TerminatorCondition {
    fn condition_vreg(&self) -> Option<VReg>;
}

impl TerminatorCondition for kajit_reprs::mir::Terminator {
    fn condition_vreg(&self) -> Option<VReg> {
        match self {
            Self::BranchIf { cond, .. } | Self::BranchIfZero { cond, .. } => Some(*cond),
            Self::JumpTable { predicate, .. } => Some(*predicate),
            Self::Branch { .. } | Self::Return => None,
        }
    }
}

// TODO: Add tests once CFG-MIR construction API is more stable.
// Tests should verify:
// 1. Def sites are correctly identified (inst, block param, func arg)
// 2. Use sites are correctly identified (inst operand, terminator, edge arg)
// 3. Unused vreg detection
