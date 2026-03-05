//! Canonical post-linearization CFG MIR.
//!
//! This module defines an explicit control-flow representation with typed IDs
//! for blocks/edges/operations. It is intended to be the source-of-truth IR for
//! post-linearization stages (regalloc, backends, simulation, and debug views).

use std::collections::{BTreeSet, HashMap};
use std::fmt;
use std::ops::Range;

use kajit_ir::{ErrorCode, LambdaId, VReg};
use kajit_lir::LinearOp;

macro_rules! define_id {
    ($name:ident) => {
        #[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
        pub struct $name(pub u32);

        impl $name {
            pub const fn new(index: u32) -> Self {
                Self(index)
            }

            pub const fn index(self) -> usize {
                self.0 as usize
            }
        }
    };
}

define_id!(FunctionId);
define_id!(BlockId);
define_id!(EdgeId);
define_id!(InstId);
define_id!(TermId);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum OpId {
    Inst(InstId),
    Term(TermId),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OperandKind {
    Use,
    Def,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RegClass {
    Gpr,
    Simd,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FixedReg {
    AbiArg(u8),
    AbiRet(u8),
    HwReg(u8),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Operand {
    pub vreg: VReg,
    pub kind: OperandKind,
    pub class: RegClass,
    pub fixed: Option<FixedReg>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Clobbers {
    pub caller_saved_gpr: bool,
    pub caller_saved_simd: bool,
}

#[derive(Debug, Clone)]
pub struct Inst {
    pub id: InstId,
    pub op: LinearOp,
    pub operands: Vec<Operand>,
    pub clobbers: Clobbers,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct EdgeArg {
    pub target: VReg,
    pub source: VReg,
}

#[derive(Debug, Clone)]
pub struct Edge {
    pub id: EdgeId,
    pub from: BlockId,
    pub to: BlockId,
    pub args: Vec<EdgeArg>,
}

#[derive(Debug, Clone)]
pub enum Terminator {
    Return,
    ErrorExit {
        code: ErrorCode,
    },
    Branch {
        edge: EdgeId,
    },
    BranchIf {
        cond: VReg,
        taken: EdgeId,
        fallthrough: EdgeId,
    },
    BranchIfZero {
        cond: VReg,
        taken: EdgeId,
        fallthrough: EdgeId,
    },
    JumpTable {
        predicate: VReg,
        targets: Vec<EdgeId>,
        default: EdgeId,
    },
}

impl Terminator {
    pub fn successor_edges(&self) -> Vec<EdgeId> {
        match self {
            Self::Return | Self::ErrorExit { .. } => Vec::new(),
            Self::Branch { edge } => vec![*edge],
            Self::BranchIf {
                taken, fallthrough, ..
            }
            | Self::BranchIfZero {
                taken, fallthrough, ..
            } => vec![*taken, *fallthrough],
            Self::JumpTable {
                targets, default, ..
            } => {
                let mut out = targets.clone();
                out.push(*default);
                out
            }
        }
    }
}

#[derive(Debug, Clone)]
pub struct Block {
    pub id: BlockId,
    pub params: Vec<VReg>,
    pub insts: Vec<InstId>,
    pub term: TermId,
    pub preds: Vec<EdgeId>,
    pub succs: Vec<EdgeId>,
}

#[derive(Debug, Clone)]
pub struct Function {
    pub id: FunctionId,
    pub lambda_id: LambdaId,
    pub entry: BlockId,
    pub data_args: Vec<VReg>,
    pub data_results: Vec<VReg>,
    pub blocks: Vec<Block>,
    pub edges: Vec<Edge>,
    pub insts: Vec<Inst>,
    pub terms: Vec<Terminator>,
}

#[derive(Debug, Clone)]
pub struct Program {
    pub funcs: Vec<Function>,
    pub vreg_count: u32,
    pub slot_count: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ProgPoint {
    Before(OpId),
    After(OpId),
    Edge(EdgeId),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Schedule {
    pub op_order: Vec<OpId>,
    pub op_to_index: HashMap<OpId, u32>,
    pub block_ranges: HashMap<BlockId, Range<u32>>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CfgMirError {
    message: String,
}

impl CfgMirError {
    fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
        }
    }
}

impl fmt::Display for CfgMirError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.message)
    }
}

impl std::error::Error for CfgMirError {}

impl Function {
    pub fn block(&self, id: BlockId) -> Option<&Block> {
        self.blocks.get(id.index())
    }

    pub fn edge(&self, id: EdgeId) -> Option<&Edge> {
        self.edges.get(id.index())
    }

    pub fn inst(&self, id: InstId) -> Option<&Inst> {
        self.insts.get(id.index())
    }

    pub fn term(&self, id: TermId) -> Option<&Terminator> {
        self.terms.get(id.index())
    }

    pub fn derive_schedule(&self) -> Result<Schedule, CfgMirError> {
        self.validate()?;

        let mut op_order = Vec::new();
        let mut op_to_index = HashMap::<OpId, u32>::new();
        let mut block_ranges = HashMap::<BlockId, Range<u32>>::new();

        for block in &self.blocks {
            let start = op_order.len() as u32;
            for inst_id in &block.insts {
                let op = OpId::Inst(*inst_id);
                op_to_index.insert(op, op_order.len() as u32);
                op_order.push(op);
            }
            let term_op = OpId::Term(block.term);
            op_to_index.insert(term_op, op_order.len() as u32);
            op_order.push(term_op);
            let end = op_order.len() as u32;
            block_ranges.insert(block.id, start..end);
        }

        Ok(Schedule {
            op_order,
            op_to_index,
            block_ranges,
        })
    }

    pub fn validate(&self) -> Result<(), CfgMirError> {
        if self.blocks.is_empty() {
            return Err(CfgMirError::new(format!(
                "func @{} has no blocks",
                self.lambda_id.index()
            )));
        }

        if self.block(self.entry).is_none() {
            return Err(CfgMirError::new(format!(
                "func @{} entry block b{} is out of range",
                self.lambda_id.index(),
                self.entry.0
            )));
        }

        for (idx, block) in self.blocks.iter().enumerate() {
            if block.id.index() != idx {
                return Err(CfgMirError::new(format!(
                    "func @{} block index mismatch: position {} has id b{}",
                    self.lambda_id.index(),
                    idx,
                    block.id.0
                )));
            }
        }
        for (idx, edge) in self.edges.iter().enumerate() {
            if edge.id.index() != idx {
                return Err(CfgMirError::new(format!(
                    "func @{} edge index mismatch: position {} has id e{}",
                    self.lambda_id.index(),
                    idx,
                    edge.id.0
                )));
            }
        }
        for (idx, inst) in self.insts.iter().enumerate() {
            if inst.id.index() != idx {
                return Err(CfgMirError::new(format!(
                    "func @{} inst index mismatch: position {} has id i{}",
                    self.lambda_id.index(),
                    idx,
                    inst.id.0
                )));
            }
        }

        let mut used_terms = BTreeSet::<TermId>::new();
        let mut used_insts = BTreeSet::<InstId>::new();

        for block in &self.blocks {
            if self.term(block.term).is_none() {
                return Err(CfgMirError::new(format!(
                    "func @{} block b{} references missing term t{}",
                    self.lambda_id.index(),
                    block.id.0,
                    block.term.0
                )));
            }
            used_terms.insert(block.term);

            for inst_id in &block.insts {
                if self.inst(*inst_id).is_none() {
                    return Err(CfgMirError::new(format!(
                        "func @{} block b{} references missing inst i{}",
                        self.lambda_id.index(),
                        block.id.0,
                        inst_id.0
                    )));
                }
                used_insts.insert(*inst_id);
            }

            for succ in &block.succs {
                let edge = self.edge(*succ).ok_or_else(|| {
                    CfgMirError::new(format!(
                        "func @{} block b{} has missing succ edge e{}",
                        self.lambda_id.index(),
                        block.id.0,
                        succ.0
                    ))
                })?;
                if edge.from != block.id {
                    return Err(CfgMirError::new(format!(
                        "func @{} block b{} lists succ e{} but edge.from is b{}",
                        self.lambda_id.index(),
                        block.id.0,
                        succ.0,
                        edge.from.0
                    )));
                }
            }

            for pred in &block.preds {
                let edge = self.edge(*pred).ok_or_else(|| {
                    CfgMirError::new(format!(
                        "func @{} block b{} has missing pred edge e{}",
                        self.lambda_id.index(),
                        block.id.0,
                        pred.0
                    ))
                })?;
                if edge.to != block.id {
                    return Err(CfgMirError::new(format!(
                        "func @{} block b{} lists pred e{} but edge.to is b{}",
                        self.lambda_id.index(),
                        block.id.0,
                        pred.0,
                        edge.to.0
                    )));
                }
            }

            let term = self.term(block.term).expect("validated above");
            let term_succs = term.successor_edges();
            if term_succs != block.succs {
                return Err(CfgMirError::new(format!(
                    "func @{} block b{} terminator successors {:?} != block succs {:?}",
                    self.lambda_id.index(),
                    block.id.0,
                    term_succs,
                    block.succs
                )));
            }
        }

        let entry = self.block(self.entry).expect("validated above");
        if !entry.preds.is_empty() {
            return Err(CfgMirError::new(format!(
                "func @{} entry block b{} has predecessors {:?}",
                self.lambda_id.index(),
                self.entry.0,
                entry.preds
            )));
        }

        if used_terms.len() != self.blocks.len() {
            return Err(CfgMirError::new(format!(
                "func @{} term ownership mismatch: {} blocks reference {} unique terms",
                self.lambda_id.index(),
                self.blocks.len(),
                used_terms.len()
            )));
        }

        if used_insts.len() != self.insts.len() {
            return Err(CfgMirError::new(format!(
                "func @{} instruction ownership mismatch: {} unique inst refs for {} insts",
                self.lambda_id.index(),
                used_insts.len(),
                self.insts.len()
            )));
        }

        for edge in &self.edges {
            let to_block = self.block(edge.to).ok_or_else(|| {
                CfgMirError::new(format!(
                    "func @{} edge e{} targets missing block b{}",
                    self.lambda_id.index(),
                    edge.id.0,
                    edge.to.0
                ))
            })?;
            if edge.args.len() != to_block.params.len() {
                return Err(CfgMirError::new(format!(
                    "func @{} edge e{} arg count {} != dest block b{} param count {}",
                    self.lambda_id.index(),
                    edge.id.0,
                    edge.args.len(),
                    edge.to.0,
                    to_block.params.len()
                )));
            }
        }

        Ok(())
    }
}

impl Program {
    pub fn validate(&self) -> Result<(), CfgMirError> {
        for (idx, func) in self.funcs.iter().enumerate() {
            if func.id.index() != idx {
                return Err(CfgMirError::new(format!(
                    "function index mismatch: position {} has id f{}",
                    idx, func.id.0
                )));
            }
            func.validate()?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn v(index: u32) -> VReg {
        VReg::new(index)
    }

    fn simple_cfg_function() -> Function {
        Function {
            id: FunctionId(0),
            lambda_id: LambdaId::new(0),
            entry: BlockId(0),
            data_args: Vec::new(),
            data_results: Vec::new(),
            blocks: vec![Block {
                id: BlockId(0),
                params: Vec::new(),
                insts: vec![InstId(0)],
                term: TermId(0),
                preds: Vec::new(),
                succs: Vec::new(),
            }],
            edges: Vec::new(),
            insts: vec![Inst {
                id: InstId(0),
                op: LinearOp::Const {
                    dst: v(0),
                    value: 42,
                },
                operands: vec![Operand {
                    vreg: v(0),
                    kind: OperandKind::Def,
                    class: RegClass::Gpr,
                    fixed: None,
                }],
                clobbers: Clobbers::default(),
            }],
            terms: vec![Terminator::Return],
        }
    }

    #[test]
    fn validate_accepts_minimal_well_formed_cfg() {
        let f = simple_cfg_function();
        f.validate().expect("minimal cfg must validate");
    }

    #[test]
    fn derive_schedule_includes_terminator_after_insts() {
        let f = simple_cfg_function();
        let schedule = f.derive_schedule().expect("schedule should derive");
        assert_eq!(
            schedule.op_order,
            vec![OpId::Inst(InstId(0)), OpId::Term(TermId(0))]
        );
        assert_eq!(schedule.block_ranges[&BlockId(0)], 0..2);
    }

    #[test]
    fn validate_rejects_entry_block_with_predecessor() {
        let mut f = simple_cfg_function();
        f.edges.push(Edge {
            id: EdgeId(0),
            from: BlockId(0),
            to: BlockId(0),
            args: Vec::new(),
        });
        f.blocks[0].preds = vec![EdgeId(0)];
        f.blocks[0].succs = vec![EdgeId(0)];
        f.terms[0] = Terminator::Branch { edge: EdgeId(0) };

        let err = f.validate().expect_err("entry preds should fail");
        assert!(
            err.to_string().contains("entry block"),
            "unexpected error: {err}"
        );
    }
}
