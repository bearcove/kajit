//! Canonical post-linearization CFG MIR.
//!
//! This module defines an explicit control-flow representation with typed IDs
//! for blocks/edges/operations. It is intended to be the source-of-truth IR for
//! post-linearization stages (regalloc, backends, simulation, and debug views).

pub mod display;
pub mod parse;

use std::collections::{BTreeSet, HashMap, HashSet};
use std::fmt;
use std::ops::Range;

pub(crate) use crate::ir as kajit_ir;
pub(crate) use crate::lir as kajit_lir;
use kajit_ir::{
    Arena, DebugScope, DebugScopeId, DebugValue, DebugValueId, FnPtr, IntrinsicRegistry, LambdaId,
    VReg,
};
use kajit_lir::{LabelId, LinearIr, LinearOp};

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

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum OpId {
    Inst(InstId),
    Term(TermId),
}

/// Whether an operand reads an existing value or defines a new one.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OperandKind {
    Use,
    Def,
}

/// Register bank required by an operand.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RegClass {
    Gpr,
    Simd,
}

/// Fixed-register constraint attached to an operand.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FixedReg {
    AbiArg(u8),
    AbiRet(u8),
    HwReg(u8),
}

/// Register-allocation operand metadata for an instruction.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Operand {
    /// Virtual register referenced by the operand.
    pub vreg: VReg,
    /// Use/def role of the operand.
    pub kind: OperandKind,
    /// Required register class.
    pub class: RegClass,
    /// Optional fixed physical register.
    pub fixed: Option<FixedReg>,
}

/// Implicit register clobbers not represented as explicit defs.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Clobbers {
    pub caller_saved_gpr: bool,
    pub caller_saved_simd: bool,
}

/// A single non-terminator instruction in CFG-MIR.
#[derive(Debug, Clone)]
pub struct Inst {
    pub id: InstId,
    pub op: LinearOp,
    pub operands: Vec<Operand>,
    pub clobbers: Clobbers,
}

/// Phi-like block-parameter binding carried on a CFG edge.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct EdgeArg {
    /// Destination block parameter.
    pub target: VReg,
    /// Source value provided by the predecessor.
    pub source: VReg,
}

/// A control-flow edge between two blocks.
#[derive(Debug, Clone)]
pub struct Edge {
    pub id: EdgeId,
    pub from: BlockId,
    pub to: BlockId,
    pub args: Vec<EdgeArg>,
}

/// Terminator for a CFG-MIR block.
#[derive(Debug, Clone)]
pub enum Terminator {
    Return,
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
    /// Return the edge IDs that this terminator may transfer control to.
    pub fn successor_edges(&self) -> Vec<EdgeId> {
        match self {
            Self::Return => Vec::new(),
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

/// A basic block in CFG-MIR.
#[derive(Debug, Clone)]
pub struct Block {
    /// Stable block ID.
    pub id: BlockId,
    /// SSA block parameters filled from incoming `EdgeArg`s.
    pub params: Vec<VReg>,
    /// Instructions executed before the terminator.
    pub insts: Vec<InstId>,
    /// Block terminator.
    pub term: TermId,
    /// Incoming edges, rebuildable from the edge list.
    pub preds: Vec<EdgeId>,
    /// Outgoing edges, rebuildable from the edge list.
    pub succs: Vec<EdgeId>,
    pub dead: bool, // Tombstone for merged blocks - backend should skip when computing offsets
}

/// A single lowered lambda/function in canonical CFG-MIR form.
#[derive(Debug, Clone)]
pub struct Function {
    pub id: FunctionId,
    pub lambda_id: LambdaId,
    pub entry: BlockId,
    pub data_args: Vec<VReg>,
    pub data_results: Vec<VReg>,
    /// Minimum output buffer size in bytes.
    pub output_size: usize,
    pub blocks: Vec<Block>,
    pub edges: Vec<Edge>,
    pub insts: Vec<Inst>,
    pub terms: Vec<Terminator>,
}

/// Whole-program CFG-MIR container shared by optimization, regalloc, and backends.
#[derive(Debug, Clone, Default)]
pub struct Program {
    pub funcs: Vec<Function>,
    pub vreg_count: u32,
    pub slot_count: u32,
    /// Number of data args passed in calling-convention registers.
    pub param_slot_count: u32,
    pub debug: ProgramDebugProvenance,
    /// Embedded constant data blobs (string literals, etc.).
    pub data_blobs: Vec<Vec<u8>>,
    /// Stack allocations (variable-size frame regions).
    pub stack_allocs: Vec<kajit_ir::StackAllocInfo>,
    /// Type layouts for data_args (debug info for pointer tracking).
    /// Each entry describes the pointee type of the corresponding data_arg.
    pub data_arg_layouts: Vec<kajit_types::TypeLayout>,
}

/// Debug provenance copied onto CFG-MIR operations and vregs.
#[derive(Debug, Clone, Default)]
pub struct ProgramDebugProvenance {
    /// Scope arena inherited from upstream IR.
    pub scopes: Arena<DebugScope>,
    /// Semantic value labels inherited from upstream IR.
    pub values: Arena<DebugValue>,
    /// Root scope of the program, if known.
    pub root_scope: Option<DebugScopeId>,
    /// Per-op scope provenance keyed by `(lambda_id, op_id)`.
    pub op_scopes: HashMap<(LambdaId, OpId), DebugScopeId>,
    /// Per-op semantic value provenance keyed by `(lambda_id, op_id)`.
    pub op_values: HashMap<(LambdaId, OpId), DebugValueId>,
    /// Scope provenance by raw vreg index.
    pub vreg_scopes: Vec<Option<DebugScopeId>>,
    /// Semantic value provenance by raw vreg index.
    pub vreg_values: Vec<Option<DebugValueId>>,
}

/// Program point used by liveness and scheduling analyses.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ProgPoint {
    Before(OpId),
    After(OpId),
    Edge(EdgeId),
}

/// Linear execution schedule derived from the current block/instruction order.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Schedule {
    /// Instructions/terminators in block-local order.
    pub op_order: Vec<OpId>,
    /// Reverse index into `op_order`.
    pub op_to_index: HashMap<OpId, u32>,
    /// Half-open op ranges for each block in `op_order`.
    pub block_ranges: HashMap<BlockId, Range<u32>>,
}

/// Structural validation error for CFG-MIR.
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
    /// Iterate over live (non-tombstoned) blocks.
    pub fn live_blocks(&self) -> impl DoubleEndedIterator<Item = &Block> {
        self.blocks.iter().filter(|b| !b.dead)
    }

    /// Iterate mutably over live (non-tombstoned) blocks.
    pub fn live_blocks_mut(&mut self) -> impl DoubleEndedIterator<Item = &mut Block> {
        self.blocks.iter_mut().filter(|b| !b.dead)
    }

    /// Rebuild all blocks' preds and succs lists from the edge list.
    /// Edges are the source of truth — call this after any structural CFG rewrite.
    pub fn rebuild_preds_succs(&mut self) {
        for block in &mut self.blocks {
            block.preds.clear();
            block.succs.clear();
        }
        for edge in &self.edges {
            if edge.from.index() < self.blocks.len() {
                self.blocks[edge.from.index()].succs.push(edge.id);
            }
            if edge.to.index() < self.blocks.len() {
                self.blocks[edge.to.index()].preds.push(edge.id);
            }
        }
    }

    /// Remove edges that no terminator references, then rebuild preds/succs
    /// and renumber edges to be contiguous. This fixes orphaned edges left by
    /// block removal or merge operations.
    pub fn gc_edges(&mut self) {
        use std::collections::{HashMap, HashSet};

        // Collect all edges referenced by terminators
        let mut referenced: HashSet<EdgeId> = HashSet::new();
        for block in &self.blocks {
            if block.dead {
                continue;
            }
            let term = &self.terms[block.term.index()];
            for eid in term.successor_edges() {
                referenced.insert(eid);
            }
        }

        // Build new edge list with only referenced edges, renumbered
        let mut old_to_new: HashMap<EdgeId, EdgeId> = HashMap::new();
        let mut new_edges = Vec::new();
        for edge in &self.edges {
            if referenced.contains(&edge.id) {
                let new_id = EdgeId::new(new_edges.len() as u32);
                old_to_new.insert(edge.id, new_id);
                let mut e = edge.clone();
                e.id = new_id;
                new_edges.push(e);
            }
        }
        self.edges = new_edges;

        // Update edge IDs in terminators
        for term in &mut self.terms {
            match term {
                Terminator::Branch { edge } => {
                    if let Some(&new) = old_to_new.get(edge) {
                        *edge = new;
                    }
                }
                Terminator::BranchIf {
                    taken, fallthrough, ..
                }
                | Terminator::BranchIfZero {
                    taken, fallthrough, ..
                } => {
                    if let Some(&new) = old_to_new.get(taken) {
                        *taken = new;
                    }
                    if let Some(&new) = old_to_new.get(fallthrough) {
                        *fallthrough = new;
                    }
                }
                Terminator::JumpTable {
                    targets, default, ..
                } => {
                    for t in targets.iter_mut() {
                        if let Some(&new) = old_to_new.get(t) {
                            *t = new;
                        }
                    }
                    if let Some(&new) = old_to_new.get(default) {
                        *default = new;
                    }
                }
                Terminator::Return => {}
            }
        }

        // Rebuild preds/succs from the cleaned edge list
        self.rebuild_preds_succs();
    }

    pub fn block(&self, id: BlockId) -> Option<&Block> {
        self.blocks.get(id.index())
    }

    /// Look up an edge by ID.
    pub fn edge(&self, id: EdgeId) -> Option<&Edge> {
        self.edges.get(id.index())
    }

    /// Look up an instruction by ID.
    pub fn inst(&self, id: InstId) -> Option<&Inst> {
        self.insts.get(id.index())
    }

    /// Look up a terminator by ID.
    pub fn term(&self, id: TermId) -> Option<&Terminator> {
        self.terms.get(id.index())
    }

    /// Derive a stable operation schedule from the function's current block order.
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

    /// Validate CFG-MIR structural invariants for this function.
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

        for block in self.live_blocks() {
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

        let live_block_count = self.live_blocks().count();
        if used_terms.len() != live_block_count {
            return Err(CfgMirError::new(format!(
                "func @{} term ownership mismatch: {} live blocks reference {} unique terms",
                self.lambda_id.index(),
                live_block_count,
                used_terms.len()
            )));
        }

        // Note: We allow used_insts.len() < self.insts.len() because DCE may leave
        // orphaned instructions in the arena to keep InstIds stable. However, we still
        // check that blocks don't reference more instructions than exist.
        if used_insts.len() > self.insts.len() {
            return Err(CfgMirError::new(format!(
                "func @{} instruction refs exceed arena: {} unique refs for {} insts",
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

        // Validate that every vreg use has a reaching definition
        self.validate_def_use()?;

        Ok(())
    }

    /// Check that every vreg use is satisfied by a definition that reaches it.
    /// This catches issues like DCE removing too much.
    fn validate_def_use(&self) -> Result<(), CfgMirError> {
        for block in &self.blocks {
            // Track definitions available at each point in the block
            let mut defs_available: HashSet<VReg> = HashSet::new();

            // Function data_args are implicitly defined at entry
            for &arg in &self.data_args {
                defs_available.insert(arg);
            }

            // Block params are defined at entry
            for &param in &block.params {
                defs_available.insert(param);
            }

            // In SSA, a def in any dominating block is available. For a sound
            // but simple check, collect ALL defs across the entire function —
            // this won't catch use-before-def within a single block or
            // non-dominating def issues, but it prevents false positives from
            // the old entry-block-only heuristic.
            for other_block in &self.blocks {
                for &param in &other_block.params {
                    defs_available.insert(param);
                }
                for &inst_id in &other_block.insts {
                    let inst = &self.insts[inst_id.index()];
                    for op in &inst.operands {
                        if op.kind == OperandKind::Def {
                            defs_available.insert(op.vreg);
                        }
                    }
                }
            }

            // Walk instructions in order
            for &inst_id in &block.insts {
                let inst = &self.insts[inst_id.index()];

                // Check uses come before the instruction's defs
                for op in &inst.operands {
                    if op.kind == OperandKind::Use && !defs_available.contains(&op.vreg) {
                        return Err(CfgMirError::new(format!(
                            "func @{} block b{} inst i{}: use of v{} has no reaching definition (inst op: {:?})",
                            self.lambda_id.index(),
                            block.id.0,
                            inst_id.0,
                            op.vreg.index(),
                            inst.op
                        )));
                    }
                }

                // Add this instruction's defs
                for op in &inst.operands {
                    if op.kind == OperandKind::Def {
                        defs_available.insert(op.vreg);
                    }
                }
            }
        }
        Ok(())
    }
}

impl Program {
    /// Return the debug scope attached to a specific op, if any.
    pub fn op_debug_scope(&self, lambda_id: LambdaId, op_id: OpId) -> Option<DebugScopeId> {
        self.debug.op_scopes.get(&(lambda_id, op_id)).copied()
    }

    /// Return the semantic debug value attached to a specific op, if any.
    pub fn op_debug_value(&self, lambda_id: LambdaId, op_id: OpId) -> Option<DebugValueId> {
        self.debug.op_values.get(&(lambda_id, op_id)).copied()
    }

    /// Return the debug scope attached to a vreg, if any.
    pub fn vreg_debug_scope(&self, vreg: VReg) -> Option<DebugScopeId> {
        self.debug.vreg_scopes.get(vreg.index()).copied().flatten()
    }

    /// Return the semantic debug value attached to a vreg, if any.
    pub fn vreg_debug_value(&self, vreg: VReg) -> Option<DebugValueId> {
        self.debug.vreg_values.get(vreg.index()).copied().flatten()
    }

    /// Validate every function in the program.
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

/// Display wrapper that optionally resolves function pointers and constants
/// through an [`IntrinsicRegistry`].
pub struct ProgramDisplay<'a> {
    program: &'a Program,
    registry: Option<&'a IntrinsicRegistry>,
}

impl fmt::Display for Program {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let display = ProgramDisplay {
            program: self,
            registry: None,
        };
        fmt::Display::fmt(&display, f)
    }
}

impl<'a> fmt::Display for ProgramDisplay<'a> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(
            f,
            "cfg_program vregs={} slots={} {{",
            self.program.vreg_count, self.program.slot_count
        )?;
        for func in &self.program.funcs {
            fmt_cfg_function(f, func, self.registry)?;
        }
        writeln!(f, "}}")
    }
}

impl Program {
    /// Render this program using names from `registry` where available.
    pub fn display_with_registry<'a>(
        &'a self,
        registry: &'a IntrinsicRegistry,
    ) -> ProgramDisplay<'a> {
        ProgramDisplay {
            program: self,
            registry: Some(registry),
        }
    }

    /// Build a debug-friendly textual listing for all functions in the program.
    pub fn debug_line_listing_with_registry(
        &self,
        registry: Option<&IntrinsicRegistry>,
    ) -> Vec<String> {
        let mut lines = Vec::new();
        for func in &self.funcs {
            let lambda = func.lambda_id.index();
            for block in &func.blocks {
                let block_id = block.id.0;
                for inst_id in &block.insts {
                    let inst = func
                        .inst(*inst_id)
                        .expect("block instruction should exist for debug listing");
                    let op_id = OpId::Inst(*inst_id);
                    let mut line = format!("f{lambda} b{block_id} op={op_id:?} :: ");
                    fmt_cfg_inst_to_string(&mut line, inst, registry);
                    lines.push(line);
                }
                let term = func
                    .term(block.term)
                    .expect("block terminator should exist for debug listing");
                let op_id = OpId::Term(block.term);
                let mut line = format!("f{lambda} b{block_id} op={op_id:?} :: ");
                fmt_terminator_to_string(&mut line, term);
                lines.push(line);
            }
        }
        lines
    }
}

fn fmt_cfg_function(
    f: &mut fmt::Formatter<'_>,
    func: &Function,
    registry: Option<&IntrinsicRegistry>,
) -> fmt::Result {
    writeln!(
        f,
        "  cfg_func @{} f{} entry=b{} {{",
        func.lambda_id.index(),
        func.id.0,
        func.entry.0
    )?;
    writeln!(
        f,
        "    data_args: {}",
        fmt_vreg_list_bracketed(&func.data_args)
    )?;
    writeln!(
        f,
        "    data_results: {}",
        fmt_vreg_list_bracketed(&func.data_results)
    )?;

    for block in &func.blocks {
        writeln!(
            f,
            "    block b{} params={} insts={} term=t{} preds={} succs={}",
            block.id.0,
            fmt_vreg_list_bracketed(&block.params),
            fmt_inst_id_list_bracketed(&block.insts),
            block.term.0,
            fmt_edge_id_list_bracketed(&block.preds),
            fmt_edge_id_list_bracketed(&block.succs)
        )?;
    }

    // Collect inst IDs referenced by blocks (skip dead/unreferenced insts)
    let mut referenced_insts = std::collections::HashSet::new();
    for block in &func.blocks {
        for &inst_id in &block.insts {
            referenced_insts.insert(inst_id);
        }
    }
    for inst in &func.insts {
        if !referenced_insts.contains(&inst.id) {
            continue;
        }
        write!(f, "    inst i{}: ", inst.id.0)?;
        fmt_cfg_inst(f, inst, registry)?;
        writeln!(f)?;
    }

    for (idx, term) in func.terms.iter().enumerate() {
        write!(f, "    term t{}: ", idx)?;
        fmt_terminator(f, term)?;
        writeln!(f)?;
    }

    for edge in &func.edges {
        writeln!(
            f,
            "    edge e{}: b{} -> b{} {}",
            edge.id.0,
            edge.from.0,
            edge.to.0,
            fmt_edge_arg_list_bracketed(&edge.args)
        )?;
    }

    writeln!(f, "  }}")
}

fn fmt_vreg_list_bracketed(vregs: &[VReg]) -> String {
    let mut out = String::from("[");
    for (idx, vreg) in vregs.iter().enumerate() {
        if idx > 0 {
            out.push_str(", ");
        }
        out.push('v');
        out.push_str(&vreg.index().to_string());
    }
    out.push(']');
    out
}

fn fmt_inst_id_list_bracketed(insts: &[InstId]) -> String {
    let mut out = String::from("[");
    for (idx, inst) in insts.iter().enumerate() {
        if idx > 0 {
            out.push_str(", ");
        }
        out.push('i');
        out.push_str(&inst.0.to_string());
    }
    out.push(']');
    out
}

fn fmt_edge_id_list_bracketed(edges: &[EdgeId]) -> String {
    let mut out = String::from("[");
    for (idx, edge) in edges.iter().enumerate() {
        if idx > 0 {
            out.push_str(", ");
        }
        out.push('e');
        out.push_str(&edge.0.to_string());
    }
    out.push(']');
    out
}

fn fmt_edge_arg_list_bracketed(args: &[EdgeArg]) -> String {
    let mut out = String::from("[");
    for (idx, arg) in args.iter().enumerate() {
        if idx > 0 {
            out.push_str(", ");
        }
        if arg.target == arg.source {
            out.push('v');
            out.push_str(&arg.source.index().to_string());
        } else {
            out.push('v');
            out.push_str(&arg.target.index().to_string());
            out.push_str("=>");
            out.push('v');
            out.push_str(&arg.source.index().to_string());
        }
    }
    out.push(']');
    out
}

fn _fmt_cfg_operand(f: &mut fmt::Formatter<'_>, operand: &Operand) -> fmt::Result {
    write!(f, "v{}", operand.vreg.index())?;
    write!(
        f,
        ":{}",
        match operand.class {
            RegClass::Gpr => "gpr",
            RegClass::Simd => "simd",
        }
    )?;
    if let Some(fixed) = operand.fixed {
        match fixed {
            FixedReg::AbiArg(i) => write!(f, "/arg{i}")?,
            FixedReg::AbiRet(i) => write!(f, "/ret{i}")?,
            FixedReg::HwReg(enc) => write!(f, "/hw{enc}")?,
        }
    }
    Ok(())
}

fn fmt_cfg_op_name(
    f: &mut fmt::Formatter<'_>,
    op: &LinearOp,
    registry: Option<&IntrinsicRegistry>,
) -> fmt::Result {
    match op {
        LinearOp::Const { value, .. } => {
            write!(f, "const(")?;
            fmt_const(f, *value, registry)?;
            write!(f, ")")
        }
        LinearOp::DataAddr { blob_id, .. } => write!(f, "data_addr({blob_id})"),
        LinearOp::ExternAddr { symbol, .. } => write!(f, "extern_addr(@{symbol})"),
        LinearOp::BinOp { op, .. } => write!(f, "{op:?}"),
        LinearOp::UnaryOp { op, .. } => write!(f, "{op:?}"),
        LinearOp::Copy { .. } => write!(f, "copy"),
        LinearOp::SlotAddr { slot, .. } => write!(f, "slot_addr({})", slot.index()),
        LinearOp::StackAlloc { id, .. } => write!(f, "stack_alloc({})", id.index()),
        LinearOp::StoreToAddr { width, .. } => write!(f, "store_addr([{width}])"),
        LinearOp::LoadFromAddr { width, .. } => write!(f, "load_addr([{width}])"),
        LinearOp::WriteToSlot { slot, .. } => write!(f, "write_slot({})", slot.index()),
        LinearOp::ReadFromSlot { slot, .. } => write!(f, "read_slot({})", slot.index()),
        LinearOp::CallIntrinsic { func, .. } => {
            write!(f, "call_intrinsic(")?;
            fmt_intrinsic(f, *func, registry)?;
            write!(f, ")")
        }
        LinearOp::CallPure { func, .. } => {
            write!(f, "call_pure(")?;
            fmt_intrinsic(f, *func, registry)?;
            write!(f, ")")
        }
        LinearOp::CallEffect { func, .. } => {
            write!(f, "call_effect(")?;
            fmt_intrinsic(f, *func, registry)?;
            write!(f, ")")
        }
        LinearOp::CallLambda { target, .. } => write!(f, "call_lambda(@{})", target.index()),
        other => write!(f, "<?op:{other:?}>"),
    }
}

/// Extract dst vreg from a LinearOp (fallback when operands are empty).
fn linearop_dst(op: &LinearOp) -> Option<VReg> {
    match op {
        LinearOp::Const { dst, .. }
        | LinearOp::DataAddr { dst, .. }
        | LinearOp::ExternAddr { dst, .. }
        | LinearOp::BinOp { dst, .. }
        | LinearOp::UnaryOp { dst, .. }
        | LinearOp::Copy { dst, .. }
        | LinearOp::SlotAddr { dst, .. }
        | LinearOp::StackAlloc { dst, .. }
        | LinearOp::LoadFromAddr { dst, .. }
        | LinearOp::ReadFromSlot { dst, .. }
        | LinearOp::CallPure { dst, .. }
        | LinearOp::CallEffect { dst, .. } => Some(*dst),
        LinearOp::CallIntrinsic { dst, .. } => *dst,
        _ => None,
    }
}

/// Extract use vregs from a LinearOp (fallback when operands are empty).
fn linearop_uses(op: &LinearOp) -> Vec<VReg> {
    match op {
        LinearOp::BinOp { lhs, rhs, .. } => vec![*lhs, *rhs],
        LinearOp::UnaryOp { src, .. }
        | LinearOp::Copy { src, .. }
        | LinearOp::WriteToSlot { src, .. } => vec![*src],
        LinearOp::StoreToAddr { addr, src, .. } => vec![*addr, *src],
        LinearOp::LoadFromAddr { addr, .. } => vec![*addr],
        LinearOp::CallIntrinsic { args, .. }
        | LinearOp::CallPure { args, .. }
        | LinearOp::CallEffect { args, .. }
        | LinearOp::CallLambda { args, .. } => args.clone(),
        _ => vec![],
    }
}

fn fmt_cfg_inst(
    f: &mut fmt::Formatter<'_>,
    inst: &Inst,
    registry: Option<&IntrinsicRegistry>,
) -> fmt::Result {
    let _defs: Vec<_> = inst
        .operands
        .iter()
        .filter(|op| op.kind == OperandKind::Def)
        .collect();
    let _uses: Vec<_> = inst
        .operands
        .iter()
        .filter(|op| op.kind == OperandKind::Use)
        .collect();

    // Show def: always use linearop_dst() for consistency (inst.operands may
    // have been modified by elim_imm, but the LinearOp always has the real dst).
    if let Some(dst) = linearop_dst(&inst.op) {
        write!(f, "v{}:gpr = ", dst.index())?;
    }

    fmt_cfg_op_name(f, &inst.op, registry)?;

    // Show uses: always use linearop_uses() for completeness (inst.operands
    // may have operands removed by elim_imm pass, but the LinearOp still has
    // all source vregs and the text format must be round-trippable).
    let canonical_uses = linearop_uses(&inst.op);
    if !canonical_uses.is_empty() {
        write!(f, " ")?;
        for (idx, vreg) in canonical_uses.iter().enumerate() {
            if idx > 0 {
                write!(f, ", ")?;
            }
            write!(f, "v{}:gpr", vreg.index())?;
        }
    }

    if inst.clobbers.caller_saved_gpr || inst.clobbers.caller_saved_simd {
        write!(f, " !")?;
        if inst.clobbers.caller_saved_gpr {
            write!(f, "gpr")?;
        }
        if inst.clobbers.caller_saved_simd {
            if inst.clobbers.caller_saved_gpr {
                write!(f, ",")?;
            }
            write!(f, "simd")?;
        }
    }

    Ok(())
}

fn fmt_cfg_inst_to_string(out: &mut String, inst: &Inst, registry: Option<&IntrinsicRegistry>) {
    use std::fmt::Write as _;
    write!(out, "{}", InstDisplay { inst, registry }).expect("writing to String should not fail");
}

fn fmt_terminator_to_string(out: &mut String, term: &Terminator) {
    use std::fmt::Write as _;
    write!(out, "{}", TerminatorDisplay(term)).expect("writing to String should not fail");
}

struct InstDisplay<'a> {
    inst: &'a Inst,
    registry: Option<&'a IntrinsicRegistry>,
}

impl fmt::Display for InstDisplay<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt_cfg_inst(f, self.inst, self.registry)
    }
}

struct TerminatorDisplay<'a>(&'a Terminator);

impl fmt::Display for TerminatorDisplay<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt_terminator(f, self.0)
    }
}

fn fmt_intrinsic(
    f: &mut fmt::Formatter<'_>,
    func: FnPtr,
    registry: Option<&IntrinsicRegistry>,
) -> fmt::Result {
    if let Some(registry) = registry
        && let Some(name) = registry.name_of(func)
    {
        return write!(f, "@{name}");
    }
    write!(f, "{func}")
}

fn fmt_const(
    f: &mut fmt::Formatter<'_>,
    value: u64,
    registry: Option<&IntrinsicRegistry>,
) -> fmt::Result {
    if let Some(registry) = registry
        && let Some(name) = registry.const_name_of(value)
    {
        return write!(f, "@{name}");
    }
    write!(f, "{value:#x}")
}

fn fmt_terminator(f: &mut fmt::Formatter<'_>, term: &Terminator) -> fmt::Result {
    match term {
        Terminator::Return => write!(f, "return"),
        Terminator::Branch { edge } => write!(f, "branch e{}", edge.0),
        Terminator::BranchIf {
            cond,
            taken,
            fallthrough,
        } => write!(
            f,
            "branch_if v{} -> e{}, fallthrough e{}",
            cond.index(),
            taken.0,
            fallthrough.0
        ),
        Terminator::BranchIfZero {
            cond,
            taken,
            fallthrough,
        } => write!(
            f,
            "branch_if_zero v{} -> e{}, fallthrough e{}",
            cond.index(),
            taken.0,
            fallthrough.0
        ),
        Terminator::JumpTable {
            predicate,
            targets,
            default,
        } => {
            write!(f, "jump_table v{} [", predicate.index())?;
            for (idx, edge) in targets.iter().enumerate() {
                if idx > 0 {
                    write!(f, ", ")?;
                }
                write!(f, "e{}", edge.0)?;
            }
            write!(f, "], default e{}", default.0)
        }
    }
}

#[derive(Debug, Clone)]
enum TempTermLabel {
    Return,
    Branch {
        target: LabelId,
        phi_args: Vec<(VReg, VReg)>,
    },
    BranchIf {
        cond: VReg,
        target: LabelId,
        phi_args: Vec<(VReg, VReg)>,
        fallthrough_phi_args: Vec<(VReg, VReg)>,
    },
    BranchIfZero {
        cond: VReg,
        target: LabelId,
        phi_args: Vec<(VReg, VReg)>,
        fallthrough_phi_args: Vec<(VReg, VReg)>,
    },
    JumpTable {
        predicate: VReg,
        labels: Vec<LabelId>,
        default: LabelId,
    },
    Fallthrough(usize),
}

#[derive(Debug, Clone)]
enum TempTermBlock {
    Return,
    Branch {
        target: BlockId,
        phi_args: Vec<(VReg, VReg)>,
    },
    BranchIf {
        cond: VReg,
        target: BlockId,
        fallthrough: BlockId,
        phi_args: Vec<(VReg, VReg)>,
        fallthrough_phi_args: Vec<(VReg, VReg)>,
    },
    BranchIfZero {
        cond: VReg,
        target: BlockId,
        fallthrough: BlockId,
        phi_args: Vec<(VReg, VReg)>,
        fallthrough_phi_args: Vec<(VReg, VReg)>,
    },
    JumpTable {
        predicate: VReg,
        targets: Vec<BlockId>,
        default: BlockId,
    },
}

impl TempTermBlock {
    fn _uses(&self) -> Vec<VReg> {
        let mut out = Vec::new();
        match self {
            Self::Branch { phi_args, .. } => {
                out.extend(phi_args.iter().map(|(src, _)| *src));
            }
            Self::BranchIf {
                cond,
                phi_args,
                fallthrough_phi_args,
                ..
            }
            | Self::BranchIfZero {
                cond,
                phi_args,
                fallthrough_phi_args,
                ..
            } => {
                out.push(*cond);
                out.extend(phi_args.iter().map(|(src, _)| *src));
                out.extend(fallthrough_phi_args.iter().map(|(src, _)| *src));
            }
            Self::JumpTable { predicate, .. } => out.push(*predicate),
            Self::Return => {}
        }
        out
    }

    /// Get phi args for a specific successor (by index in successors() order).
    fn phi_args_for_successor(&self, succ_idx: usize) -> &[(VReg, VReg)] {
        match self {
            Self::Branch { phi_args, .. } => {
                assert_eq!(succ_idx, 0);
                phi_args
            }
            Self::BranchIf {
                phi_args,
                fallthrough_phi_args,
                ..
            }
            | Self::BranchIfZero {
                phi_args,
                fallthrough_phi_args,
                ..
            } => match succ_idx {
                0 => phi_args,
                1 => fallthrough_phi_args,
                _ => panic!("BranchIf has only 2 successors"),
            },
            _ => &[],
        }
    }

    fn successors(&self) -> Vec<BlockId> {
        match self {
            Self::Return => Vec::new(),
            Self::Branch { target, .. } => vec![*target],
            Self::BranchIf {
                target,
                fallthrough,
                ..
            }
            | Self::BranchIfZero {
                target,
                fallthrough,
                ..
            } => vec![*target, *fallthrough],
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

fn is_terminator(op: &LinearOp) -> bool {
    matches!(
        op,
        LinearOp::Branch { .. }
            | LinearOp::BranchIf { .. }
            | LinearOp::BranchIfZero { .. }
            | LinearOp::JumpTable { .. }
    )
}

fn push_use(out: &mut Vec<Operand>, v: VReg, fixed: Option<FixedReg>) {
    out.push(Operand {
        vreg: v,
        kind: OperandKind::Use,
        class: RegClass::Gpr,
        fixed,
    });
}

fn push_def(out: &mut Vec<Operand>, v: VReg, fixed: Option<FixedReg>) {
    out.push(Operand {
        vreg: v,
        kind: OperandKind::Def,
        class: RegClass::Gpr,
        fixed,
    });
}

// Lowering, optimization, and regalloc integration live in `kajit-mir`.
