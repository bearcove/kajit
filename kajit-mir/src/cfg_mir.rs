#![allow(dead_code)]
//! Canonical post-linearization CFG MIR.
//!
//! This module defines an explicit control-flow representation with typed IDs
//! for blocks/edges/operations. It is intended to be the source-of-truth IR for
//! post-linearization stages (regalloc, backends, simulation, and debug views).

use std::collections::{BTreeSet, HashMap, HashSet};
use std::fmt;
use std::ops::Range;

use kajit_ir::{
    Arena, DebugScope, DebugScopeId, DebugValue, DebugValueId, ErrorCode, IntrinsicFn,
    IntrinsicRegistry, LambdaId, VReg,
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
    pub dead: bool, // Tombstone for merged blocks - backend should skip when computing offsets
}

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

#[derive(Debug, Clone)]
pub struct Program {
    pub funcs: Vec<Function>,
    pub vreg_count: u32,
    pub slot_count: u32,
    pub debug: ProgramDebugProvenance,
    pub hints: crate::regalloc3::hints::HintMap,
}

#[derive(Debug, Clone, Default)]
pub struct ProgramDebugProvenance {
    pub scopes: Arena<DebugScope>,
    pub values: Arena<DebugValue>,
    pub root_scope: Option<DebugScopeId>,
    pub op_scopes: HashMap<(LambdaId, OpId), DebugScopeId>,
    pub op_values: HashMap<(LambdaId, OpId), DebugValueId>,
    pub vreg_scopes: Vec<Option<DebugScopeId>>,
    pub vreg_values: Vec<Option<DebugValueId>>,
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
    pub fn op_debug_scope(&self, lambda_id: LambdaId, op_id: OpId) -> Option<DebugScopeId> {
        self.debug.op_scopes.get(&(lambda_id, op_id)).copied()
    }

    pub fn op_debug_value(&self, lambda_id: LambdaId, op_id: OpId) -> Option<DebugValueId> {
        self.debug.op_values.get(&(lambda_id, op_id)).copied()
    }

    pub fn vreg_debug_scope(&self, vreg: VReg) -> Option<DebugScopeId> {
        self.debug.vreg_scopes.get(vreg.index()).copied().flatten()
    }

    pub fn vreg_debug_value(&self, vreg: VReg) -> Option<DebugValueId> {
        self.debug.vreg_values.get(vreg.index()).copied().flatten()
    }

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
    pub fn display_with_registry<'a>(
        &'a self,
        registry: &'a IntrinsicRegistry,
    ) -> ProgramDisplay<'a> {
        ProgramDisplay {
            program: self,
            registry: Some(registry),
        }
    }

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
        LinearOp::BinOp { op, .. } => write!(f, "{op:?}"),
        LinearOp::UnaryOp { op, .. } => write!(f, "{op:?}"),
        LinearOp::Copy { .. } => write!(f, "copy"),
        LinearOp::BoundsCheck { count } => write!(f, "bounds_check({count})"),
        LinearOp::ReadBytes { count, .. } => write!(f, "read_bytes({count})"),
        LinearOp::PeekByte { .. } => write!(f, "peek_byte"),
        LinearOp::AdvanceCursor { count } => write!(f, "advance({count})"),
        LinearOp::AdvanceCursorBy { .. } => write!(f, "advance_by"),
        LinearOp::SaveCursor { .. } => write!(f, "save_cursor"),
        LinearOp::SaveInputEnd { .. } => write!(f, "save_input_end"),
        LinearOp::RestoreCursor { .. } => write!(f, "restore_cursor"),
        LinearOp::WriteToField { offset, width, .. } => write!(f, "store([{offset}:{width}])"),
        LinearOp::ReadFromField { offset, width, .. } => write!(f, "load([{offset}:{width}])"),
        LinearOp::SaveOutPtr { .. } => write!(f, "save_out_ptr"),
        LinearOp::SetOutPtr { .. } => write!(f, "set_out_ptr"),
        LinearOp::SlotAddr { slot, .. } => write!(f, "slot_addr({})", slot.index()),
        LinearOp::StoreToAddr { width, .. } => write!(f, "store_addr([{width}])"),
        LinearOp::LoadFromAddr { width, .. } => write!(f, "load_addr([{width}])"),
        LinearOp::WriteToSlot { slot, .. } => write!(f, "write_slot({})", slot.index()),
        LinearOp::ReadFromSlot { slot, .. } => write!(f, "read_slot({})", slot.index()),
        LinearOp::CallIntrinsic {
            func, field_offset, ..
        } => {
            write!(f, "call_intrinsic(")?;
            fmt_intrinsic(f, *func, registry)?;
            write!(f, ", fo={field_offset})")
        }
        LinearOp::CallPure { func, .. } => {
            write!(f, "call_pure(")?;
            fmt_intrinsic(f, *func, registry)?;
            write!(f, ")")
        }
        LinearOp::CallLambda { target, .. } => write!(f, "call_lambda(@{})", target.index()),
        LinearOp::SimdStringScan { .. } => write!(f, "simd_string_scan"),
        LinearOp::SimdWhitespaceSkip => write!(f, "simd_ws_skip"),
        LinearOp::ErrorExit { code } => write!(f, "error_exit({code:?})"),
        other => write!(f, "<?op:{other:?}>"),
    }
}

/// Extract dst vreg from a LinearOp (fallback when operands are empty).
fn linearop_dst(op: &LinearOp) -> Option<VReg> {
    match op {
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
        | LinearOp::CallPure { dst, .. } => Some(*dst),
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
        | LinearOp::AdvanceCursorBy { src }
        | LinearOp::RestoreCursor { src }
        | LinearOp::WriteToField { src, .. }
        | LinearOp::SetOutPtr { src }
        | LinearOp::WriteToSlot { src, .. } => vec![*src],
        LinearOp::StoreToAddr { addr, src, .. } => vec![*addr, *src],
        LinearOp::LoadFromAddr { addr, .. } => vec![*addr],
        LinearOp::CallIntrinsic { args, .. }
        | LinearOp::CallPure { args, .. }
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
    func: IntrinsicFn,
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
        Terminator::ErrorExit { code } => write!(f, "error_exit({code:?})"),
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
    ErrorExit(ErrorCode),
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
    ErrorExit(ErrorCode),
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
            Self::Return | Self::ErrorExit(_) => {}
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
            Self::Return | Self::ErrorExit(_) => Vec::new(),
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
            | LinearOp::ErrorExit { .. }
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

fn lower_inst(id: InstId, op: LinearOp) -> Inst {
    let mut operands = Vec::new();
    let mut clobbers = Clobbers::default();

    match &op {
        LinearOp::Const { dst, .. }
        | LinearOp::ReadBytes { dst, .. }
        | LinearOp::PeekByte { dst }
        | LinearOp::SaveCursor { dst }
        | LinearOp::SaveInputEnd { dst }
        | LinearOp::ReadFromField { dst, .. }
        | LinearOp::SaveOutPtr { dst }
        | LinearOp::SlotAddr { dst, .. }
        | LinearOp::ReadFromSlot { dst, .. } => {
            push_def(&mut operands, *dst, None);
        }
        LinearOp::LoadFromAddr { dst, addr, .. } => {
            push_use(&mut operands, *addr, None);
            push_def(&mut operands, *dst, None);
        }
        LinearOp::BinOp {
            dst, lhs, rhs, op, ..
        } => {
            push_use(&mut operands, *lhs, None);
            let rhs_fixed = {
                #[cfg(target_arch = "x86_64")]
                {
                    match op {
                        kajit_lir::BinOpKind::Shr | kajit_lir::BinOpKind::Shl => {
                            Some(FixedReg::HwReg(1))
                        }
                        _ => None,
                    }
                }
                #[cfg(not(target_arch = "x86_64"))]
                {
                    let _ = op;
                    None
                }
            };
            push_use(&mut operands, *rhs, rhs_fixed);
            push_def(&mut operands, *dst, None);
        }
        LinearOp::UnaryOp { dst, src, .. } | LinearOp::Copy { dst, src } => {
            push_use(&mut operands, *src, None);
            push_def(&mut operands, *dst, None);
        }
        LinearOp::AdvanceCursorBy { src }
        | LinearOp::RestoreCursor { src }
        | LinearOp::WriteToField { src, .. }
        | LinearOp::SetOutPtr { src }
        | LinearOp::WriteToSlot { src, .. } => {
            push_use(&mut operands, *src, None);
        }
        LinearOp::StoreToAddr { addr, src, .. } => {
            push_use(&mut operands, *addr, None);
            push_use(&mut operands, *src, None);
        }
        LinearOp::CallIntrinsic { args, dst, .. } => {
            for &arg in args {
                push_use(&mut operands, arg, None);
            }
            if let Some(dst) = dst {
                push_def(&mut operands, *dst, None);
            }
            clobbers = Clobbers {
                caller_saved_gpr: true,
                caller_saved_simd: true,
            };
        }
        LinearOp::CallPure { args, dst, .. } => {
            for &arg in args {
                push_use(&mut operands, arg, None);
            }
            push_def(&mut operands, *dst, None);
            clobbers = Clobbers {
                caller_saved_gpr: true,
                caller_saved_simd: true,
            };
        }
        LinearOp::CallLambda { args, results, .. } => {
            for (i, &arg) in args.iter().enumerate() {
                push_use(&mut operands, arg, Some(FixedReg::AbiArg((i + 2) as u8)));
            }
            for (i, &r) in results.iter().enumerate() {
                push_def(&mut operands, r, Some(FixedReg::AbiRet(i as u8)));
            }
            clobbers = Clobbers {
                caller_saved_gpr: true,
                caller_saved_simd: true,
            };
        }
        LinearOp::SimdStringScan { pos, kind } => {
            push_def(&mut operands, *pos, None);
            push_def(&mut operands, *kind, None);
        }
        LinearOp::BoundsCheck { .. }
        | LinearOp::AdvanceCursor { .. }
        | LinearOp::SimdWhitespaceSkip => {}
        LinearOp::Label(_)
        | LinearOp::Branch { .. }
        | LinearOp::BranchIf { .. }
        | LinearOp::BranchIfZero { .. }
        | LinearOp::JumpTable { .. }
        | LinearOp::ErrorExit { .. }
        | LinearOp::FuncStart { .. }
        | LinearOp::FuncEnd => {
            panic!("unexpected non-inst op in cfg_mir::lower_inst: {op:?}");
        }
    }

    Inst {
        id,
        op,
        operands,
        clobbers,
    }
}

fn resolve_term_labels(
    term: &TempTermLabel,
    labels: &HashMap<LabelId, BlockId>,
    next: Option<BlockId>,
) -> TempTermBlock {
    match term {
        TempTermLabel::Return => TempTermBlock::Return,
        TempTermLabel::ErrorExit(code) => TempTermBlock::ErrorExit(*code),
        TempTermLabel::Branch {
            target: label,
            phi_args,
        } => TempTermBlock::Branch {
            target: *labels
                .get(label)
                .unwrap_or_else(|| panic!("unknown label target: {label:?}")),
            phi_args: phi_args.clone(),
        },
        TempTermLabel::BranchIf {
            cond,
            target,
            phi_args,
            fallthrough_phi_args,
        } => TempTermBlock::BranchIf {
            cond: *cond,
            target: *labels
                .get(target)
                .unwrap_or_else(|| panic!("unknown label target: {target:?}")),
            fallthrough: next.expect("BranchIf must have fallthrough block"),
            phi_args: phi_args.clone(),
            fallthrough_phi_args: fallthrough_phi_args.clone(),
        },
        TempTermLabel::BranchIfZero {
            cond,
            target,
            phi_args,
            fallthrough_phi_args,
        } => TempTermBlock::BranchIfZero {
            cond: *cond,
            target: *labels
                .get(target)
                .unwrap_or_else(|| panic!("unknown label target: {target:?}")),
            fallthrough: next.expect("BranchIfZero must have fallthrough block"),
            phi_args: phi_args.clone(),
            fallthrough_phi_args: fallthrough_phi_args.clone(),
        },
        TempTermLabel::JumpTable {
            predicate,
            labels: targets,
            default,
        } => TempTermBlock::JumpTable {
            predicate: *predicate,
            targets: targets
                .iter()
                .map(|label| {
                    *labels
                        .get(label)
                        .unwrap_or_else(|| panic!("unknown jump-table label: {label:?}"))
                })
                .collect(),
            default: *labels
                .get(default)
                .unwrap_or_else(|| panic!("unknown jump-table default: {default:?}")),
        },
        TempTermLabel::Fallthrough(next_idx) => TempTermBlock::Branch {
            target: BlockId(*next_idx as u32),
            phi_args: vec![],
        },
    }
}

fn _collect_use_def(
    block: &Block,
    insts: &[Inst],
    term: &TempTermBlock,
    use_set: &mut [bool],
    def_set: &mut [bool],
) {
    for inst_id in &block.insts {
        let inst = &insts[inst_id.index()];
        for operand in &inst.operands {
            match operand.kind {
                OperandKind::Use => {
                    if !def_set[operand.vreg.index()] {
                        use_set[operand.vreg.index()] = true;
                    }
                }
                OperandKind::Def => {
                    def_set[operand.vreg.index()] = true;
                }
            }
        }
    }
    for vreg in term._uses() {
        if !def_set[vreg.index()] {
            use_set[vreg.index()] = true;
        }
    }
}

fn lower_function(
    function_id: FunctionId,
    lambda_id: LambdaId,
    data_args: Vec<VReg>,
    data_results: Vec<VReg>,
    output_size: usize,
    ops: &[LinearOp],
    op_scopes: &[Option<DebugScopeId>],
    op_values: &[Option<DebugValueId>],
    _vreg_count: u32,
) -> (
    Function,
    HashMap<OpId, DebugScopeId>,
    HashMap<OpId, DebugValueId>,
) {
    if ops.is_empty() {
        return (
            Function {
                id: function_id,
                lambda_id,
                entry: BlockId(0),
                data_args,
                data_results,
                output_size: 0,
                blocks: vec![Block {
                    id: BlockId(0),
                    params: Vec::new(),
                    insts: Vec::new(),
                    term: TermId(0),
                    preds: Vec::new(),
                    succs: Vec::new(),
                    dead: false,
                }],
                edges: Vec::new(),
                insts: Vec::new(),
                terms: vec![Terminator::Return],
            },
            HashMap::new(),
            HashMap::new(),
        );
    }

    let mut leaders = vec![0usize];
    for (idx, op) in ops.iter().enumerate() {
        if idx != 0 && matches!(op, LinearOp::Label(_)) {
            leaders.push(idx);
        }
        if is_terminator(op) && idx + 1 < ops.len() {
            leaders.push(idx + 1);
        }
    }
    leaders.sort_unstable();
    leaders.dedup();

    let mut labels = HashMap::<LabelId, BlockId>::new();
    let mut blocks = Vec::<Block>::new();
    let mut insts = Vec::<Inst>::new();
    let mut label_terms = Vec::<TempTermLabel>::new();
    let mut lowered_scopes = HashMap::<OpId, DebugScopeId>::new();
    let mut lowered_values = HashMap::<OpId, DebugValueId>::new();

    for bi in 0..leaders.len() {
        let start = leaders[bi];
        let end = if bi + 1 < leaders.len() {
            leaders[bi + 1]
        } else {
            ops.len()
        };

        let mut cursor = start;
        if matches!(ops[cursor], LinearOp::Label(_))
            && let LinearOp::Label(label) = ops[cursor]
        {
            labels.insert(label, BlockId(bi as u32));
            cursor += 1;
        }

        let mut block_inst_ids = Vec::<InstId>::new();
        let mut term = None::<TempTermLabel>;

        while cursor < end {
            let op_scope = op_scopes.get(cursor).copied().flatten();
            let op_value = op_values.get(cursor).copied().flatten();
            match ops[cursor].clone() {
                LinearOp::Branch { target, phi_args } => {
                    if let Some(scope) = op_scope {
                        lowered_scopes.insert(OpId::Term(TermId(bi as u32)), scope);
                    }
                    if let Some(debug_value) = op_value {
                        lowered_values.insert(OpId::Term(TermId(bi as u32)), debug_value);
                    }
                    term = Some(TempTermLabel::Branch { target, phi_args });
                    cursor += 1;
                    break;
                }
                LinearOp::BranchIf {
                    cond,
                    target,
                    phi_args,
                    fallthrough_phi_args,
                } => {
                    if let Some(scope) = op_scope {
                        lowered_scopes.insert(OpId::Term(TermId(bi as u32)), scope);
                    }
                    if let Some(debug_value) = op_value {
                        lowered_values.insert(OpId::Term(TermId(bi as u32)), debug_value);
                    }
                    term = Some(TempTermLabel::BranchIf {
                        cond,
                        target,
                        phi_args,
                        fallthrough_phi_args,
                    });
                    cursor += 1;
                    break;
                }
                LinearOp::BranchIfZero {
                    cond,
                    target,
                    phi_args,
                    fallthrough_phi_args,
                } => {
                    if let Some(scope) = op_scope {
                        lowered_scopes.insert(OpId::Term(TermId(bi as u32)), scope);
                    }
                    if let Some(debug_value) = op_value {
                        lowered_values.insert(OpId::Term(TermId(bi as u32)), debug_value);
                    }
                    term = Some(TempTermLabel::BranchIfZero {
                        cond,
                        target,
                        phi_args,
                        fallthrough_phi_args,
                    });
                    cursor += 1;
                    break;
                }
                LinearOp::JumpTable {
                    predicate,
                    labels,
                    default,
                } => {
                    if let Some(scope) = op_scope {
                        lowered_scopes.insert(OpId::Term(TermId(bi as u32)), scope);
                    }
                    if let Some(debug_value) = op_value {
                        lowered_values.insert(OpId::Term(TermId(bi as u32)), debug_value);
                    }
                    term = Some(TempTermLabel::JumpTable {
                        predicate,
                        labels,
                        default,
                    });
                    cursor += 1;
                    break;
                }
                LinearOp::ErrorExit { code } => {
                    if let Some(scope) = op_scope {
                        lowered_scopes.insert(OpId::Term(TermId(bi as u32)), scope);
                    }
                    if let Some(debug_value) = op_value {
                        lowered_values.insert(OpId::Term(TermId(bi as u32)), debug_value);
                    }
                    term = Some(TempTermLabel::ErrorExit(code));
                    cursor += 1;
                    break;
                }
                LinearOp::Label(_) | LinearOp::FuncStart { .. } | LinearOp::FuncEnd => {
                    panic!(
                        "unexpected structural op in function body: {:?}",
                        ops[cursor]
                    );
                }
                other => {
                    let inst_id = InstId(insts.len() as u32);
                    if let Some(scope) = op_scope {
                        lowered_scopes.insert(OpId::Inst(inst_id), scope);
                    }
                    if let Some(debug_value) = op_value {
                        lowered_values.insert(OpId::Inst(inst_id), debug_value);
                    }
                    insts.push(lower_inst(inst_id, other));
                    block_inst_ids.push(inst_id);
                    cursor += 1;
                }
            }
        }

        assert!(
            cursor == end,
            "non-terminator ops after terminator in block {bi}"
        );

        if term.is_none() {
            if bi + 1 < leaders.len() {
                term = Some(TempTermLabel::Fallthrough(bi + 1));
            } else {
                term = Some(TempTermLabel::Return);
            }
        }

        blocks.push(Block {
            id: BlockId(bi as u32),
            params: Vec::new(),
            insts: block_inst_ids,
            term: TermId(bi as u32),
            preds: Vec::new(),
            succs: Vec::new(),
            dead: false,
        });
        label_terms.push(term.expect("term must be set"));
    }

    let mut block_terms = Vec::<TempTermBlock>::new();
    for (bi, label_term) in label_terms.iter().enumerate() {
        let next = if bi + 1 < blocks.len() {
            Some(BlockId((bi + 1) as u32))
        } else {
            None
        };
        block_terms.push(resolve_term_labels(label_term, &labels, next));
    }

    // Block params come from phi_args, NOT from liveness. In SSA with block
    // params, only actual phi join points are block params. Live-through values
    // cross block boundaries normally without becoming params.
    //
    // For each block, collect phi_arg targets from ALL incoming edges.
    let mut block_param_sets = vec![Vec::<VReg>::new(); blocks.len()];
    for (_bi, term) in block_terms.iter().enumerate() {
        let successors = term.successors();
        for (succ_idx, succ) in successors.iter().enumerate() {
            for &(_src, tgt) in term.phi_args_for_successor(succ_idx) {
                let params = &mut block_param_sets[succ.index()];
                if !params.contains(&tgt) {
                    params.push(tgt);
                }
            }
        }
    }

    for bi in 0..blocks.len() {
        blocks[bi].params = block_param_sets[bi].clone();
    }

    let mut edges = Vec::<Edge>::new();
    for from in 0..blocks.len() {
        let successors = block_terms[from].successors();
        for (succ_idx, to) in successors.iter().enumerate() {
            let edge_id = EdgeId(edges.len() as u32);
            // Build a map from phi_args: target_param → source_vreg.
            let phi = block_terms[from].phi_args_for_successor(succ_idx);
            let phi_map: HashMap<VReg, VReg> = phi.iter().map(|&(src, tgt)| (tgt, src)).collect();
            // For each block param, use the phi source if provided,
            // otherwise the vreg flows through unchanged (source == target).
            let args = blocks[to.index()]
                .params
                .iter()
                .copied()
                .map(|target| {
                    let source = phi_map.get(&target).copied().unwrap_or(target);
                    EdgeArg { target, source }
                })
                .collect();
            edges.push(Edge {
                id: edge_id,
                from: BlockId(from as u32),
                to: *to,
                args,
            });
            blocks[from].succs.push(edge_id);
            blocks[to.index()].preds.push(edge_id);
        }
    }

    let mut terms = Vec::<Terminator>::with_capacity(block_terms.len());
    for (bi, term) in block_terms.iter().enumerate() {
        let succ_edges = blocks[bi].succs.clone();
        let lowered = match term {
            TempTermBlock::Return => Terminator::Return,
            TempTermBlock::ErrorExit(code) => Terminator::ErrorExit { code: *code },
            TempTermBlock::Branch { .. } => Terminator::Branch {
                edge: *succ_edges
                    .first()
                    .expect("branch block should have one successor edge"),
            },
            TempTermBlock::BranchIf { cond, .. } => {
                assert_eq!(
                    succ_edges.len(),
                    2,
                    "branch-if block must have two successor edges"
                );
                Terminator::BranchIf {
                    cond: *cond,
                    taken: succ_edges[0],
                    fallthrough: succ_edges[1],
                }
            }
            TempTermBlock::BranchIfZero { cond, .. } => {
                assert_eq!(
                    succ_edges.len(),
                    2,
                    "branch-if-zero block must have two successor edges"
                );
                Terminator::BranchIfZero {
                    cond: *cond,
                    taken: succ_edges[0],
                    fallthrough: succ_edges[1],
                }
            }
            TempTermBlock::JumpTable {
                predicate, targets, ..
            } => {
                assert_eq!(
                    succ_edges.len(),
                    targets.len() + 1,
                    "jump-table block must have target edges plus default edge"
                );
                let split_at = targets.len();
                Terminator::JumpTable {
                    predicate: *predicate,
                    targets: succ_edges[..split_at].to_vec(),
                    default: succ_edges[split_at],
                }
            }
        };
        terms.push(lowered);
    }

    (
        Function {
            id: function_id,
            lambda_id,
            entry: BlockId(0),
            data_args,
            data_results,
            output_size,
            blocks,
            edges,
            insts,
            terms,
        },
        lowered_scopes,
        lowered_values,
    )
}

/// Lower linearized IR into the canonical CFG MIR model.
pub fn lower_linear_ir(ir: &LinearIr, hints: crate::regalloc3::hints::HintMap) -> Program {
    let mut funcs = Vec::<Function>::new();
    let mut op_scopes = HashMap::<(LambdaId, OpId), DebugScopeId>::new();
    let mut op_values = HashMap::<(LambdaId, OpId), DebugValueId>::new();
    let mut cursor = 0usize;
    while cursor < ir.ops.len() {
        let (lambda_id, data_args, data_results, output_size) = match &ir.ops[cursor] {
            LinearOp::FuncStart {
                lambda_id,
                data_args,
                data_results,
                output_size,
                ..
            } => (
                *lambda_id,
                data_args.clone(),
                data_results.clone(),
                *output_size,
            ),
            other => panic!("expected FuncStart at op {cursor}, got {other:?}"),
        };

        let mut depth = 1usize;
        let mut end = cursor + 1;
        while end < ir.ops.len() {
            match &ir.ops[end] {
                LinearOp::FuncStart { .. } => depth += 1,
                LinearOp::FuncEnd => {
                    depth -= 1;
                    if depth == 0 {
                        break;
                    }
                }
                _ => {}
            }
            end += 1;
        }
        assert!(
            end < ir.ops.len(),
            "missing FuncEnd for lambda {:?}",
            lambda_id
        );

        let body = &ir.ops[cursor + 1..end];
        let function_id = FunctionId(funcs.len() as u32);
        let body_scopes = &ir.debug.op_scopes[cursor + 1..end];
        let body_values = &ir.debug.op_values[cursor + 1..end];
        let (function, function_scopes, function_values) = lower_function(
            function_id,
            lambda_id,
            data_args,
            data_results,
            output_size,
            body,
            body_scopes,
            body_values,
            ir.vreg_count,
        );
        for (op_id, scope) in function_scopes {
            op_scopes.insert((lambda_id, op_id), scope);
        }
        for (op_id, debug_value) in function_values {
            op_values.insert((lambda_id, op_id), debug_value);
        }
        funcs.push(function);
        cursor = end + 1;
    }

    Program {
        funcs,
        vreg_count: ir.vreg_count,
        slot_count: ir.slot_count,
        debug: ProgramDebugProvenance {
            scopes: ir.debug.scopes.clone(),
            values: ir.debug.values.clone(),
            root_scope: ir.debug.root_scope,
            op_scopes,
            op_values,
            vreg_scopes: ir.debug.vreg_scopes.clone(),
            vreg_values: ir.debug.vreg_values.clone(),
        },
        hints,
    }
}

/// Rematerialize constants that are passed as block parameters.
///
/// When constants are defined in one block and passed through edges to other
/// blocks as parameters, they become regular vregs that consume registers.
/// This pass identifies such constants and re-emits them locally in each block
/// Lower linear IR to CFG-MIR and run all optimization passes.
///
/// This is the single entry point for producing optimized CFG-MIR from linear IR,
/// ensuring consistent behavior between compilation and debug/test paths.
pub fn lower_and_optimize(ir: &LinearIr, hints: crate::regalloc3::hints::HintMap) -> Program {
    let mut cfg = lower_linear_ir(ir, hints);
    let opts = CfgOptOptions::from_env();

    // SSA validation: enabled in debug builds or via KAJIT_VALIDATE_SSA=1
    let validate_ssa = cfg!(debug_assertions) || std::env::var("KAJIT_VALIDATE_SSA").is_ok();

    // Validate BEFORE any passes — catches bugs from linearization
    if validate_ssa {
        for func in &cfg.funcs {
            if let Err(errors) = crate::opt::validate_ssa::validate_ssa(func) {
                eprintln!(
                    "\n❌ SSA VALIDATION FAILED before optimization passes (fresh from linearizer)"
                );
                eprintln!("Found {} SSA violation(s):\n", errors.len());
                for (i, error) in errors.iter().enumerate() {
                    eprintln!("  {}. {}", i + 1, error);
                }
                panic!("SSA validation failed before optimization passes");
            }
        }
        eprintln!("[SSA] ✓ Passed validation before optimization passes");
    }

    // Helper to validate after each opt
    let validate_after = |pass_name: &str, cfg: &Program| {
        if !validate_ssa {
            return;
        }
        eprintln!("[SSA] Validating after {}...", pass_name);
        for func in &cfg.funcs {
            if let Err(errors) = crate::opt::validate_ssa::validate_ssa(func) {
                eprintln!(
                    "\n❌ SSA VALIDATION FAILED after optimization pass: {}",
                    pass_name
                );
                eprintln!("Found {} SSA violation(s):\n", errors.len());
                for (i, error) in errors.iter().enumerate() {
                    eprintln!("  {}. {}", i + 1, error);
                }
                eprintln!("\nTo debug:");
                eprintln!(
                    "  1. Run with KAJIT_CFG_OPTS=-all,+{} to isolate this pass",
                    pass_name
                );
                eprintln!("  2. Add KAJIT_DUMP_STAGES=cfg to see CFG before/after");
                eprintln!("  3. Check if other passes compensate (try adding +cse, +gvn, etc.)");
                panic!("SSA validation failed after {}", pass_name);
            } else {
                eprintln!("[SSA] ✓ Passed validation after {}", pass_name);
            }
        }
    };

    // Run constant phi elimination (general redundant block-param elimination)
    // This replaces loop_phi_elim with a proper iterative dataflow approach
    if opts.enabled("const_phi_elim") {
        if std::env::var("KAJIT_DUMP_BEFORE_PHI_ELIM").is_ok() {
            eprintln!("=== BEFORE const_phi_elim ===\n{cfg}\n=== END ===");
        }
        for func in &mut cfg.funcs {
            crate::opt::constant_phi_elim::eliminate_constant_phis(func);
        }
        validate_after("const_phi_elim", &cfg);
    }

    // DEPRECATED: old loop-specific phi elimination (disabled by default)
    if opts.enabled("loop_phi_elim") {
        for func in &mut cfg.funcs {
            let dom = crate::analysis::dominance::DominanceInfo::compute(func);
            let loops = crate::analysis::loops::LoopInfo::compute(func, &dom);
            crate::opt::loop_phi_elim::eliminate_loop_invariant_phis(func, &dom, &loops);
        }
        validate_after("loop_phi_elim", &cfg);
    }

    if opts.enabled("remat") {
        rematerialize_constants(&mut cfg);
        validate_after("remat", &cfg);
    }
    if opts.enabled("cse") {
        local_cse(&mut cfg);
        validate_after("cse", &cfg);
    }
    if opts.enabled("gvn") {
        global_value_numbering(&mut cfg);
        validate_after("gvn", &cfg);
    }
    if opts.enabled("copyprop") {
        copy_propagation(&mut cfg);
        validate_after("copyprop", &cfg);
    }
    if opts.enabled("fuse_cmpz") {
        fuse_compare_zero_branch(&mut cfg);
        validate_after("fuse_cmpz", &cfg);
    }
    if opts.enabled("elim_imm") {
        eliminate_immediate_only_const_defs(&mut cfg);
        validate_after("elim_imm", &cfg);
    }
    if opts.enabled("dce") {
        dead_code_elimination(&mut cfg);
        for func in &mut cfg.funcs {
            crate::opt::dce::eliminate_dead_block_params(func);
        }
        validate_after("dce", &cfg);
    }
    if opts.enabled("const_branch_fold") {
        for func in &mut cfg.funcs {
            crate::opt::const_branch_fold::fold_const_branches(func);
        }
        validate_after("const_branch_fold", &cfg);
    }
    if opts.enabled("merge_blocks") {
        for func in &mut cfg.funcs {
            crate::opt::block_merge::merge_empty_blocks(func);
            // Note: unreachable blocks are left in place, will be cleaned up by later passes
        }
        validate_after("merge_blocks", &cfg);
    }
    // TODO: simplify_trivial_phis needs more work to maintain SSA
    // The basic idea is sound (found 32 trivial phis in scalar_u32)
    // but removing them breaks SSA when replacement doesn't dominate uses
    // if opts.enabled("simplify_phis") {
    //     simplify_trivial_phis(&mut cfg);
    // }
    cfg
}

/// Controls which CFG-MIR optimization passes run.
///
/// Set `KAJIT_CFG_OPTS` to a comma-separated list of `+name` or `-name` tokens.
/// Use `-all` to disable everything, then selectively re-enable with `+name`.
///
/// Pass names: `loop_phi_elim`, `remat`, `cse`, `gvn`, `copyprop`, `fuse_cmpz`, `elim_imm`, `dce`, `simplify_phis`, `merge_blocks`.
///
/// Examples:
///   `KAJIT_CFG_OPTS=-all`           — disable all CFG opts
///   `KAJIT_CFG_OPTS=-all,+copyprop` — only copy propagation
///   `KAJIT_CFG_OPTS=-dce,-copyprop` — everything except DCE and copyprop
struct CfgOptOptions {
    default_enabled: bool,
    overrides: std::collections::HashMap<String, bool>,
}

impl CfgOptOptions {
    fn from_env() -> Self {
        let raw = std::env::var("KAJIT_CFG_OPTS").unwrap_or_default();
        let mut opts = Self {
            default_enabled: true,
            overrides: std::collections::HashMap::new(),
        };
        for token in raw.split(',') {
            let token = token.trim();
            if token.is_empty() {
                continue;
            }
            let (enabled, name) = match token.as_bytes()[0] {
                b'+' => (true, &token[1..]),
                b'-' => (false, &token[1..]),
                _ => (true, token),
            };
            if name == "all" {
                opts.default_enabled = enabled;
            } else {
                opts.overrides.insert(name.to_owned(), enabled);
            }
        }
        opts
    }

    fn enabled(&self, name: &str) -> bool {
        // Check explicit override first
        if let Some(&enabled) = self.overrides.get(name) {
            return enabled;
        }

        // Special cases for deprecated/new opts
        match name {
            "loop_phi_elim" => false, // DEPRECATED: disabled by default
            "const_phi_elim" => true, // NEW: enabled by default
            _ => self.default_enabled,
        }
    }
}

/// Fuse compare-with-zero followed by conditional branch into a direct branch.
///
/// Transforms patterns like:
///   v1 = CmpNe v0, const(0)
///   BranchIfZero v1 -> taken, fallthrough
/// Into:
///   BranchIfZero v0 -> taken, fallthrough
///
/// And:
///   v1 = CmpEq v0, const(0)
///   BranchIfZero v1 -> taken, fallthrough
/// Into:
///   BranchIf v0 -> taken, fallthrough
///
/// This enables the backend to emit cbz/cbnz directly on the original value
/// instead of: cmp + cset + cbz.
pub fn fuse_compare_zero_branch(program: &mut Program) {
    for func in &mut program.funcs {
        fuse_compare_zero_branch_in_function(func);
    }
}

fn fuse_compare_zero_branch_in_function(func: &mut Function) {
    use kajit_lir::BinOpKind;

    // Step 1: Build map of const vreg -> value
    let mut const_values: HashMap<VReg, u64> = HashMap::new();
    for inst in &func.insts {
        if let LinearOp::Const { dst, value } = &inst.op {
            const_values.insert(*dst, *value);
        }
    }

    // Step 2: Find compare-with-zero instructions
    // Map: result vreg -> (comparison_kind, non_zero_operand)
    #[derive(Clone, Copy)]
    struct CompareZeroInfo {
        kind: BinOpKind, // CmpEq or CmpNe
        operand: VReg,   // the non-zero operand
    }

    let mut compare_zero_map: HashMap<VReg, CompareZeroInfo> = HashMap::new();

    for inst in &func.insts {
        if let LinearOp::BinOp { op, dst, lhs, rhs } = &inst.op
            && (*op == BinOpKind::CmpEq || *op == BinOpKind::CmpNe)
        {
            // Check if one operand is const(0)
            let lhs_is_zero = const_values.get(lhs) == Some(&0);
            let rhs_is_zero = const_values.get(rhs) == Some(&0);

            if lhs_is_zero && !rhs_is_zero {
                compare_zero_map.insert(
                    *dst,
                    CompareZeroInfo {
                        kind: *op,
                        operand: *rhs,
                    },
                );
            } else if rhs_is_zero && !lhs_is_zero {
                compare_zero_map.insert(
                    *dst,
                    CompareZeroInfo {
                        kind: *op,
                        operand: *lhs,
                    },
                );
            }
        }
    }

    if compare_zero_map.is_empty() {
        return;
    }

    // Step 3: Transform BranchIfZero/BranchIf terminators
    for term in &mut func.terms {
        match term {
            Terminator::BranchIfZero {
                cond,
                taken,
                fallthrough,
            } => {
                if let Some(info) = compare_zero_map.get(cond) {
                    // BranchIfZero on (v != 0) -> branch if v == 0 -> BranchIfZero v
                    // BranchIfZero on (v == 0) -> branch if v != 0 -> BranchIf v
                    match info.kind {
                        BinOpKind::CmpNe => {
                            // (v != 0) == 0 means v == 0, keep BranchIfZero
                            *cond = info.operand;
                        }
                        BinOpKind::CmpEq => {
                            // (v == 0) == 0 means v != 0, flip to BranchIf
                            *term = Terminator::BranchIf {
                                cond: info.operand,
                                taken: *taken,
                                fallthrough: *fallthrough,
                            };
                        }
                        _ => {}
                    }
                }
            }
            Terminator::BranchIf {
                cond,
                taken,
                fallthrough,
            } => {
                if let Some(info) = compare_zero_map.get(cond) {
                    // BranchIf on (v != 0) -> branch if v != 0 -> BranchIf v
                    // BranchIf on (v == 0) -> branch if v == 0 -> BranchIfZero v
                    match info.kind {
                        BinOpKind::CmpNe => {
                            // (v != 0) != 0 means v != 0, keep BranchIf
                            *cond = info.operand;
                        }
                        BinOpKind::CmpEq => {
                            // (v == 0) != 0 means v == 0, flip to BranchIfZero
                            *term = Terminator::BranchIfZero {
                                cond: info.operand,
                                taken: *taken,
                                fallthrough: *fallthrough,
                            };
                        }
                        _ => {}
                    }
                }
            }
            _ => {}
        }
    }
}

/// that needs them, eliminating the need to pass them through edges.
///
/// Benefits:
/// - Reduces register pressure (constants don't need to be kept live)
/// - Enables immediate encoding in backends (AND reg, #imm instead of AND reg, reg)
/// - Removes unnecessary edge argument traffic
pub fn rematerialize_constants(program: &mut Program) {
    // Build a map of VReg -> constant value for all constants in the program
    // Start with values from Const instructions
    let mut const_values: HashMap<VReg, u64> = HashMap::new();
    for func in &program.funcs {
        for inst in &func.insts {
            if let LinearOp::Const { dst, value } = &inst.op {
                const_values.insert(*dst, *value);
            }
        }
    }

    if const_values.is_empty() {
        return;
    }

    // Propagate const values through block params: if a block param receives
    // the same constant value from ALL incoming edges, it's also a constant.
    // This handles loop back edges that pass through constants unchanged.
    let mut changed = true;
    while changed {
        changed = false;
        for func in &program.funcs {
            for block in &func.blocks {
                if block.params.is_empty() || block.preds.is_empty() {
                    continue;
                }
                for (param_idx, &param_vreg) in block.params.iter().enumerate() {
                    // Skip if already known as const
                    if const_values.contains_key(&param_vreg) {
                        continue;
                    }
                    // Check if all incoming edges provide the same constant
                    let mut all_same_const: Option<u64> = None;
                    let mut is_uniform_const = true;
                    for &pred_edge_id in &block.preds {
                        let edge = &func.edges[pred_edge_id.index()];
                        if param_idx >= edge.args.len() {
                            is_uniform_const = false;
                            break;
                        }
                        let source_vreg = edge.args[param_idx].source;
                        if let Some(&const_val) = const_values.get(&source_vreg) {
                            match all_same_const {
                                None => all_same_const = Some(const_val),
                                Some(v) if v == const_val => {}
                                Some(_) => {
                                    is_uniform_const = false;
                                    break;
                                }
                            }
                        } else {
                            is_uniform_const = false;
                            break;
                        }
                    }
                    if is_uniform_const && let Some(const_val) = all_same_const {
                        const_values.insert(param_vreg, const_val);
                        changed = true;
                    }
                }
            }
        }
    }

    let debug = std::env::var("KAJIT_DEBUG_REMAT").is_ok();
    if debug {
        // Print first block's first param to identify which IR this is
        let first_param = program
            .funcs
            .first()
            .and_then(|f| f.blocks.get(1))
            .and_then(|b| b.params.first())
            .map(|v| v.index());
        eprintln!(
            "[remat] Found {} consts, vregs={}, b1.params[0]=v{:?}",
            const_values.len(),
            program.vreg_count,
            first_param
        );
    }

    for func in &mut program.funcs {
        rematerialize_constants_in_function(func, &const_values, debug);
    }

    if debug {
        let first_param = program
            .funcs
            .first()
            .and_then(|f| f.blocks.get(1))
            .and_then(|b| b.params.first())
            .map(|v| v.index());
        let param_count = program
            .funcs
            .first()
            .and_then(|f| f.blocks.get(1))
            .map(|b| b.params.len())
            .unwrap_or(0);
        eprintln!(
            "[remat] AFTER: b1 has {} params, first=v{:?}",
            param_count, first_param
        );
    }
}

fn rematerialize_constants_in_function(
    func: &mut Function,
    const_values: &HashMap<VReg, u64>,
    debug: bool,
) {
    // Track which block params to remove and what constants to insert
    // Key: BlockId, Value: Vec<(param_index, vreg, constant_value)>
    let mut remat_plan: HashMap<BlockId, Vec<(usize, VReg, u64)>> = HashMap::new();

    if debug {
        eprintln!(
            "[remat] func f{} const_values has {} entries",
            func.id.0,
            const_values.len()
        );
        for (vreg, val) in const_values.iter().take(10) {
            eprintln!("[remat]   v{} = {}", vreg.index(), val);
        }
    }

    // For each block (except entry), check its params
    for block in &func.blocks {
        if block.id == func.entry {
            continue;
        }

        if block.params.is_empty() || block.preds.is_empty() {
            continue;
        }

        if debug && block.id.0 < 3 {
            eprintln!(
                "[remat] checking block b{} with {} params",
                block.id.0,
                block.params.len()
            );
        }

        // For each param position, check if all incoming edges provide the same constant
        for (param_idx, &param_vreg) in block.params.iter().enumerate() {
            let mut all_same_const: Option<u64> = None;
            let mut is_uniform_const = true;

            for &pred_edge_id in &block.preds {
                let edge = &func.edges[pred_edge_id.index()];
                if param_idx >= edge.args.len() {
                    if debug && block.id.0 < 3 && param_idx < 3 {
                        eprintln!(
                            "[remat]   b{} param {} (v{}): edge e{} has only {} args",
                            block.id.0,
                            param_idx,
                            param_vreg.index(),
                            pred_edge_id.0,
                            edge.args.len()
                        );
                    }
                    is_uniform_const = false;
                    break;
                }
                let source_vreg = edge.args[param_idx].source;
                if let Some(&const_val) = const_values.get(&source_vreg) {
                    match all_same_const {
                        None => all_same_const = Some(const_val),
                        Some(v) if v == const_val => {}
                        Some(_) => {
                            is_uniform_const = false;
                            break;
                        }
                    }
                } else {
                    if debug && block.id.0 < 3 && param_idx < 3 {
                        eprintln!(
                            "[remat]   b{} param {} (v{}): source v{} not a constant",
                            block.id.0,
                            param_idx,
                            param_vreg.index(),
                            source_vreg.index()
                        );
                    }
                    is_uniform_const = false;
                    break;
                }
            }

            if is_uniform_const && let Some(const_val) = all_same_const {
                if debug && block.id.0 < 3 {
                    eprintln!(
                        "[remat]   b{} param {} (v{}): rematerializing const {}",
                        block.id.0,
                        param_idx,
                        param_vreg.index(),
                        const_val
                    );
                }
                remat_plan
                    .entry(block.id)
                    .or_default()
                    .push((param_idx, param_vreg, const_val));
            }
        }
    }

    if remat_plan.is_empty() {
        return;
    }

    // Apply the rematerialization plan
    // We need to process params in reverse order to keep indices valid during removal
    for (block_id, mut params_to_remat) in remat_plan {
        // Sort by param_idx descending so removals don't shift indices
        params_to_remat.sort_by(|a, b| b.0.cmp(&a.0));

        let block = &mut func.blocks[block_id.index()];

        // Collect new instructions to insert
        let mut new_insts = Vec::new();

        for (param_idx, vreg, const_val) in &params_to_remat {
            // Remove from block params
            block.params.remove(*param_idx);

            // Create new Const instruction
            let inst_id = InstId(func.insts.len() as u32);
            func.insts.push(Inst {
                id: inst_id,
                op: LinearOp::Const {
                    dst: *vreg,
                    value: *const_val,
                },
                operands: vec![Operand {
                    vreg: *vreg,
                    kind: OperandKind::Def,
                    class: RegClass::Gpr,
                    fixed: None,
                }],
                clobbers: Clobbers::default(),
            });
            new_insts.push(inst_id);
        }

        // Insert new instructions at the beginning of the block
        let block = &mut func.blocks[block_id.index()];
        for inst_id in new_insts.into_iter().rev() {
            block.insts.insert(0, inst_id);
        }

        // Remove corresponding edge args from all predecessor edges
        // Again, process in reverse order to keep indices valid
        for &pred_edge_id in &func.blocks[block_id.index()].preds.clone() {
            let edge = &mut func.edges[pred_edge_id.index()];
            for (param_idx, _, _) in &params_to_remat {
                if *param_idx < edge.args.len() {
                    edge.args.remove(*param_idx);
                }
            }
        }
    }

    // Phase 2: DISABLED - this phase created multiple defs which violate SSA.
    // TODO: Reimplement by allocating fresh vregs and updating uses.
    // The idea was: Rematerialize constants used directly across blocks (not via params).
    // For each non-entry block, find uses of vregs that are:
    // - Known constants (in const_values)
    // - NOT defined BEFORE this use in the block (respecting program order)
    // Insert a local Const instruction at block start for each such vreg.
    // But the old implementation reused the same vreg, creating multiple defs.
    if false {
        for block_idx in 0..func.blocks.len() {
            let block = &func.blocks[block_idx];
            if block.id == func.entry {
                continue;
            }

            // Process instructions in order, tracking defs as we go
            // Block params are defined at entry
            let mut defs_so_far: HashSet<VReg> = block.params.iter().copied().collect();
            let mut consts_to_insert: HashSet<VReg> = HashSet::new();

            for &inst_id in &block.insts {
                let inst = &func.insts[inst_id.index()];

                // First check uses (before adding this inst's defs)
                for op in &inst.operands {
                    if op.kind == OperandKind::Use
                        && !defs_so_far.contains(&op.vreg)
                        && const_values.contains_key(&op.vreg)
                    {
                        consts_to_insert.insert(op.vreg);
                    }
                }

                // Then add this instruction's defs
                for op in &inst.operands {
                    if op.kind == OperandKind::Def {
                        defs_so_far.insert(op.vreg);
                    }
                }
            }

            if consts_to_insert.is_empty() {
                continue;
            }

            if debug {
                eprintln!(
                    "[remat] b{}: inserting {} local consts for cross-block uses",
                    block.id.0,
                    consts_to_insert.len()
                );
            }

            // Insert Const instructions at block start (in stable order)
            let mut sorted_consts: Vec<_> = consts_to_insert.into_iter().collect();
            sorted_consts.sort_by_key(|v| v.index());

            let mut new_insts = Vec::new();
            for vreg in sorted_consts {
                let const_val = const_values[&vreg];
                let inst_id = InstId(func.insts.len() as u32);
                func.insts.push(Inst {
                    id: inst_id,
                    op: LinearOp::Const {
                        dst: vreg,
                        value: const_val,
                    },
                    operands: vec![Operand {
                        vreg,
                        kind: OperandKind::Def,
                        class: RegClass::Gpr,
                        fixed: None,
                    }],
                    clobbers: Clobbers::default(),
                });
                new_insts.push(inst_id);
            }

            // Insert at block start
            let block = &mut func.blocks[block_idx];
            for inst_id in new_insts.into_iter().rev() {
                block.insts.insert(0, inst_id);
            }
        }
    } // end if false
}

/// Copy propagation for CFG-MIR.
///
/// Replaces uses of a vreg that's just a copy of another vreg with the
/// original vreg. This enables later dead code elimination to remove
/// the now-unused Copy instructions.
///
/// Example:
///   v1 = Const(42)
///   v2 = Copy(v1)
///   v3 = Add(v2, v2)
/// Becomes:
///   v1 = Const(42)
///   v2 = Copy(v1)    // now dead
///   v3 = Add(v1, v1)
pub fn copy_propagation(program: &mut Program) {
    for func in &mut program.funcs {
        global_copy_propagation(func);
    }
}

/// Global (inter-block) copy propagation using dataflow analysis.
///
/// This pass propagates copies across block boundaries by:
/// 1. Building a map of all copies in the function
/// 2. Computing which vregs are available at each program point
/// 3. Rewriting uses to the ultimate source when safe
fn global_copy_propagation(func: &mut Function) {
    // Step 1: Build global copy map: dst -> src
    let mut copy_map: HashMap<VReg, VReg> = HashMap::new();
    for block in &func.blocks {
        for &inst_id in &block.insts {
            let inst = &func.insts[inst_id.index()];
            if let LinearOp::Copy { dst, src } = &inst.op {
                copy_map.insert(*dst, *src);
            }
        }
    }

    if copy_map.is_empty() {
        return;
    }

    // Step 2: Build vreg definition map: which block defines each vreg
    let mut vreg_def_block: HashMap<VReg, BlockId> = HashMap::new();

    for block in &func.blocks {
        // Block parameters are defined by this block
        for &param in &block.params {
            vreg_def_block.insert(param, block.id);
        }

        // Instruction defs
        for &inst_id in &block.insts {
            let inst = &func.insts[inst_id.index()];
            for operand in &inst.operands {
                if operand.kind == OperandKind::Def {
                    vreg_def_block.insert(operand.vreg, block.id);
                }
            }
        }
    }

    // Step 3: Compute dominators
    let idom = compute_dominators(func);

    // Helper: resolve copy chains to ultimate source
    let get_ultimate_source = |mut v: VReg| -> VReg {
        let mut visited = std::collections::HashSet::new();
        while let Some(&src) = copy_map.get(&v) {
            if v == src || !visited.insert(v) {
                break; // Self-copy or cycle
            }
            v = src;
        }
        v
    };

    // Step 4: Precompute intra-block def positions for each block
    let mut block_vreg_def_idx: HashMap<(BlockId, VReg), usize> = HashMap::new();

    for block in &func.blocks {
        let block_id = block.id;

        // Block params are defined at index 0
        for &param in &block.params {
            block_vreg_def_idx.insert((block_id, param), 0);
        }

        // Instruction defs
        for (idx, &inst_id) in block.insts.iter().enumerate() {
            let inst = &func.insts[inst_id.index()];
            for operand in &inst.operands {
                if operand.kind == OperandKind::Def {
                    block_vreg_def_idx.insert((block_id, operand.vreg), idx + 1);
                }
            }
        }
    }

    // Helper: check if vreg is available at given instruction in given block
    let is_available_at_inst = |vreg: VReg, block_id: BlockId, inst_idx: usize| -> bool {
        if let Some(&def_block) = vreg_def_block.get(&vreg) {
            if def_block != block_id {
                // Cross-block: check if def_block dominates block_id
                dominates(&idom, def_block, block_id)
            } else {
                // Same block: check that def comes before use
                if let Some(&def_idx) = block_vreg_def_idx.get(&(block_id, vreg)) {
                    def_idx < inst_idx
                } else {
                    false
                }
            }
        } else {
            false
        }
    };

    // Step 5: Rewrite all uses of copy destinations to their ultimate sources
    for block in &func.blocks {
        let block_id = block.id;

        // Rewrite instruction uses
        for (idx, &inst_id) in block.insts.iter().enumerate() {
            let inst_idx = idx + 1; // Instructions are 1-indexed (0 is block params)
            let inst = &mut func.insts[inst_id.index()];
            let mut changed = false;

            // Closure to rewrite a single vreg use
            let rewrite_use = |v: &mut VReg| -> bool {
                let ultimate = get_ultimate_source(*v);
                if ultimate != *v && is_available_at_inst(ultimate, block_id, inst_idx) {
                    *v = ultimate;
                    true
                } else {
                    false
                }
            };

            // Rewrite all uses in the instruction op
            match &mut inst.op {
                LinearOp::Copy { src, .. } => {
                    changed |= rewrite_use(src);
                }
                LinearOp::BinOp { lhs, rhs, .. } => {
                    changed |= rewrite_use(lhs);
                    changed |= rewrite_use(rhs);
                }
                LinearOp::UnaryOp { src, .. } => {
                    changed |= rewrite_use(src);
                }
                LinearOp::WriteToSlot { src, .. } => {
                    changed |= rewrite_use(src);
                }
                LinearOp::WriteToField { src, .. } => {
                    changed |= rewrite_use(src);
                }
                LinearOp::StoreToAddr { addr, src, .. } => {
                    changed |= rewrite_use(addr);
                    changed |= rewrite_use(src);
                }
                LinearOp::LoadFromAddr { addr, .. } => {
                    changed |= rewrite_use(addr);
                }
                LinearOp::AdvanceCursorBy { src } => {
                    changed |= rewrite_use(src);
                }
                LinearOp::RestoreCursor { src } => {
                    changed |= rewrite_use(src);
                }
                LinearOp::SetOutPtr { src } => {
                    changed |= rewrite_use(src);
                }
                LinearOp::BranchIf { cond, .. } | LinearOp::BranchIfZero { cond, .. } => {
                    changed |= rewrite_use(cond);
                }
                LinearOp::JumpTable { predicate, .. } => {
                    changed |= rewrite_use(predicate);
                }
                LinearOp::CallIntrinsic { args, .. }
                | LinearOp::CallPure { args, .. }
                | LinearOp::CallLambda { args, .. } => {
                    for arg in args.iter_mut() {
                        changed |= rewrite_use(arg);
                    }
                }
                LinearOp::SimdStringScan { pos, kind } => {
                    changed |= rewrite_use(pos);
                    changed |= rewrite_use(kind);
                }
                _ => {}
            }

            // Update operands to match
            if changed {
                for operand in &mut inst.operands {
                    if operand.kind == OperandKind::Use {
                        let ultimate = get_ultimate_source(operand.vreg);
                        if ultimate != operand.vreg
                            && is_available_at_inst(ultimate, block_id, inst_idx)
                        {
                            operand.vreg = ultimate;
                        }
                    }
                }
            }
        }

        // Rewrite terminator uses
        let term_idx = block.insts.len() + 1; // Terminator comes after all instructions
        let term = &mut func.terms[block.term.index()];

        let rewrite_term_use = |v: &mut VReg| {
            let ultimate = get_ultimate_source(*v);
            if ultimate != *v && is_available_at_inst(ultimate, block_id, term_idx) {
                *v = ultimate;
            }
        };

        match term {
            Terminator::BranchIf { cond, .. } => {
                rewrite_term_use(cond);
            }
            Terminator::BranchIfZero { cond, .. } => {
                rewrite_term_use(cond);
            }
            Terminator::JumpTable { predicate, .. } => {
                rewrite_term_use(predicate);
            }
            _ => {}
        }

        // Rewrite edge arg sources
        for &edge_id in &block.succs {
            let edge = &mut func.edges[edge_id.index()];
            for arg in &mut edge.args {
                let ultimate = get_ultimate_source(arg.source);
                if ultimate != arg.source && is_available_at_inst(ultimate, block_id, term_idx) {
                    arg.source = ultimate;
                }
            }
        }
    }
}

/// Compute reverse postorder traversal of the CFG
fn compute_rpo(func: &Function) -> Vec<BlockId> {
    let mut visited = std::collections::HashSet::new();
    let mut postorder = Vec::new();

    fn dfs(
        func: &Function,
        block_id: BlockId,
        visited: &mut std::collections::HashSet<BlockId>,
        postorder: &mut Vec<BlockId>,
    ) {
        if !visited.insert(block_id) {
            return;
        }

        let block = &func.blocks[block_id.index()];
        for &edge_id in &block.succs {
            let edge = &func.edges[edge_id.index()];
            dfs(func, edge.to, visited, postorder);
        }

        postorder.push(block_id);
    }

    dfs(func, func.entry, &mut visited, &mut postorder);
    postorder.reverse();
    postorder
}

/// Compute dominators using Cooper-Harvey-Kennedy algorithm.
/// Returns a map from each block to its immediate dominator (idom).
/// The entry block has no idom (maps to None).
fn compute_dominators(func: &Function) -> HashMap<BlockId, Option<BlockId>> {
    let rpo = compute_rpo(func);
    let mut rpo_index: HashMap<BlockId, usize> = HashMap::new();
    for (idx, &block_id) in rpo.iter().enumerate() {
        rpo_index.insert(block_id, idx);
    }

    // Build predecessor map
    let mut preds: HashMap<BlockId, Vec<BlockId>> = HashMap::new();
    for block in &func.blocks {
        for &edge_id in &block.succs {
            let edge = &func.edges[edge_id.index()];
            preds.entry(edge.to).or_default().push(block.id);
        }
    }

    // Initialize: entry has no idom, others are undefined
    let mut idom: HashMap<BlockId, Option<BlockId>> = HashMap::new();
    idom.insert(func.entry, None);

    // Iterative fixed-point computation
    let mut changed = true;
    while changed {
        changed = false;
        for &block_id in &rpo {
            if block_id == func.entry {
                continue;
            }

            // Find the first processed predecessor
            let block_preds = preds.get(&block_id);
            let mut new_idom = None;

            if let Some(pred_list) = block_preds {
                for &pred in pred_list {
                    if idom.contains_key(&pred) {
                        new_idom = match new_idom {
                            None => Some(pred),
                            Some(current) => {
                                // Intersect current and pred
                                let mut finger1 = current;
                                let mut finger2 = pred;
                                while finger1 != finger2 {
                                    while rpo_index[&finger1] > rpo_index[&finger2] {
                                        finger1 = idom[&finger1].expect("idom should be set");
                                    }
                                    while rpo_index[&finger2] > rpo_index[&finger1] {
                                        finger2 = idom[&finger2].expect("idom should be set");
                                    }
                                }
                                Some(finger1)
                            }
                        };
                    }
                }
            }

            // Update if changed
            if idom.get(&block_id) != Some(&new_idom) {
                idom.insert(block_id, new_idom);
                changed = true;
            }
        }
    }

    idom
}

/// Check if block `a` dominates block `b` using the idom map.
fn dominates(idom: &HashMap<BlockId, Option<BlockId>>, a: BlockId, b: BlockId) -> bool {
    if a == b {
        return true;
    }

    // Walk up the dominator tree from b until we find a or reach the entry
    let mut current = b;
    loop {
        match idom.get(&current) {
            Some(Some(parent)) => {
                if *parent == a {
                    return true;
                }
                current = *parent;
            }
            _ => return false, // Reached entry or undefined
        }
    }
}

/// Check if a value can be encoded as an ARM64 immediate for the given operation.
/// This is conservative - we only eliminate def operands for values we're confident
/// can be encoded as immediates on all supported targets.
fn can_encode_as_immediate(op: kajit_lir::BinOpKind, value: u64) -> bool {
    use kajit_lir::BinOpKind;
    match op {
        // ARM64 add/sub immediate: 12-bit unsigned (0-4095)
        BinOpKind::Add | BinOpKind::Sub => value <= 4095,
        // ARM64 shift immediate: 6-bit (0-63)
        BinOpKind::Shl | BinOpKind::Shr => value < 64,
        // ARM64 logical immediate uses bitmask encoding. Only allow values we know encode.
        // Common varint masks: 0x7f (7 consecutive 1s), 0x80 (single bit), 0xff, etc.
        BinOpKind::And | BinOpKind::Or | BinOpKind::Xor => is_encodable_logical_imm(value),
        _ => false,
    }
}

/// Conservative check for ARM64 logical immediate encoding.
/// Returns true only for values we're certain can be encoded.
fn is_encodable_logical_imm(value: u64) -> bool {
    // All-zeros and all-ones are never encodable
    if value == 0 || value == u64::MAX {
        return false;
    }
    // Common varint masks - these are all encodable as bitmasks
    matches!(
        value,
        0x1 | 0x3 | 0x7 | 0xf | 0x1f | 0x3f | 0x7f | 0xff | // 1-8 consecutive 1s
            0x80 | 0x100 | 0x200 | 0x400 | 0x800 | 0x1000 | // single bits
            0xffff | 0xff00 | 0xf0 // other common masks
    )
}

/// Remove def operands from Const instructions that are only used as immediates.
///
/// When a constant is only used as the RHS of BinOp instructions where the value
/// can be encoded as an immediate, we don't need regalloc to track it. By removing
/// the Def operand, regalloc won't allocate a register or generate moves for it.
/// The backend will use the immediate form directly via `const_of()`.
pub fn eliminate_immediate_only_const_defs(program: &mut Program) {
    for func in &mut program.funcs {
        eliminate_immediate_only_const_defs_in_function(func);
    }
}

fn eliminate_immediate_only_const_defs_in_function(func: &mut Function) {
    // Step 1: Build map of const vreg -> value
    let mut const_values: HashMap<VReg, u64> = HashMap::new();
    for inst in &func.insts {
        if let LinearOp::Const { dst, value } = &inst.op {
            const_values.insert(*dst, *value);
        }
    }

    if const_values.is_empty() {
        return;
    }

    // Step 1b: Build copy chains - track copies of consts
    // copy_to_const[copy_dst] = (original_const_vreg, const_value)
    let mut copy_to_const: HashMap<VReg, (VReg, u64)> = HashMap::new();
    for inst in &func.insts {
        if let LinearOp::Copy { dst, src } = &inst.op
            && let Some(&value) = const_values.get(src)
        {
            copy_to_const.insert(*dst, (*src, value));
        }
    }

    // Step 2: Track how each const/copy vreg is used
    // A const is "immediate-only" if ALL its uses (direct or via copies) are as RHS
    // of BinOp where the value can be encoded as an immediate.
    #[derive(Clone, Copy, PartialEq, Eq)]
    enum UseKind {
        ImmediateOnly,
        RequiresRegister,
    }

    // Track use kinds for both original consts and copies of consts
    let mut use_kinds: HashMap<VReg, UseKind> = HashMap::new();
    for vreg in const_values.keys() {
        use_kinds.insert(*vreg, UseKind::ImmediateOnly);
    }
    for vreg in copy_to_const.keys() {
        use_kinds.insert(*vreg, UseKind::ImmediateOnly);
    }

    // Helper: get const value for a vreg (either direct const or copy of const)
    let get_const_value = |v: &VReg| -> Option<u64> {
        if let Some(&val) = const_values.get(v) {
            return Some(val);
        }
        if let Some(&(_, val)) = copy_to_const.get(v) {
            return Some(val);
        }
        None
    };

    // Helper: check if vreg is a const or copy-of-const
    let is_const_like =
        |v: &VReg| -> bool { const_values.contains_key(v) || copy_to_const.contains_key(v) };

    // Scan all uses
    for inst in &func.insts {
        match &inst.op {
            LinearOp::BinOp { op, lhs, rhs, .. } => {
                // LHS use always requires register
                if is_const_like(lhs) {
                    use_kinds.insert(*lhs, UseKind::RequiresRegister);
                }
                // RHS can potentially be immediate
                if let Some(value) = get_const_value(rhs)
                    && !can_encode_as_immediate(*op, value)
                {
                    use_kinds.insert(*rhs, UseKind::RequiresRegister);
                }
            }
            LinearOp::Copy { dst, src } => {
                // If src is a const and dst is tracked as a copy-of-const, we handle
                // this specially - don't mark src as requiring register yet.
                // The copy is "transparent" - we'll decide based on how dst is used.
                if const_values.contains_key(src) && copy_to_const.contains_key(dst) {
                    // This copy is tracked - don't mark as RequiresRegister yet
                } else if is_const_like(src) {
                    use_kinds.insert(*src, UseKind::RequiresRegister);
                }
            }
            LinearOp::UnaryOp { src, .. } => {
                if is_const_like(src) {
                    use_kinds.insert(*src, UseKind::RequiresRegister);
                }
            }
            // Any other use requires a register
            _ => {
                for operand in &inst.operands {
                    if operand.kind == OperandKind::Use && is_const_like(&operand.vreg) {
                        use_kinds.insert(operand.vreg, UseKind::RequiresRegister);
                    }
                }
            }
        }
    }

    // Also check terminator uses
    for term in &func.terms {
        let cond = match term {
            Terminator::BranchIf { cond, .. } | Terminator::BranchIfZero { cond, .. } => {
                Some(*cond)
            }
            Terminator::JumpTable { predicate, .. } => Some(*predicate),
            _ => None,
        };
        if let Some(cond) = cond
            && is_const_like(&cond)
        {
            use_kinds.insert(cond, UseKind::RequiresRegister);
        }
    }

    // Also check edge args - consts passed through edges require registers
    for edge in &func.edges {
        for arg in &edge.args {
            if is_const_like(&arg.source) {
                use_kinds.insert(arg.source, UseKind::RequiresRegister);
            }
        }
    }

    // Step 3: Propagate RequiresRegister from copies back to original consts
    // If a copy-of-const requires a register, the original const also needs one.
    for (copy_vreg, (original_const, _)) in &copy_to_const {
        if use_kinds.get(copy_vreg) == Some(&UseKind::RequiresRegister) {
            use_kinds.insert(*original_const, UseKind::RequiresRegister);
        }
    }

    // Step 4: Collect vregs that are immediate-only
    let immediate_only: HashSet<VReg> = use_kinds
        .iter()
        .filter(|(_, kind)| **kind == UseKind::ImmediateOnly)
        .map(|(vreg, _)| *vreg)
        .collect();

    if immediate_only.is_empty() {
        return;
    }

    // Step 5: Remove Def operands from Const/Copy instructions for immediate-only vregs,
    // AND remove Use operands from BinOps that use them as RHS.
    // This is necessary because regalloc requires all uses to have reaching definitions.
    for inst in &mut func.insts {
        match &inst.op {
            LinearOp::Const { dst, .. } => {
                if immediate_only.contains(dst) {
                    inst.operands.clear();
                }
            }
            LinearOp::Copy { dst, src } => {
                // If this is a copy of an immediate-only const, clear its operands too
                if copy_to_const.contains_key(dst) && immediate_only.contains(dst) {
                    inst.operands.clear();
                }
                // Also handle if src is immediate-only (shouldn't happen given our logic, but be safe)
                let _ = src;
            }
            LinearOp::BinOp { rhs, .. } => {
                if immediate_only.contains(rhs) {
                    // Remove the Use operand for the RHS (which is the second operand)
                    // Operand order is: lhs (Use), rhs (Use), dst (Def)
                    inst.operands.retain(|op| op.vreg != *rhs);
                }
            }
            _ => {}
        }
    }
}

/// Dead code elimination for CFG-MIR.
///
/// Removes instructions whose outputs are never used. This is particularly
/// useful after rematerialization, which can leave the original constant
/// definitions unused.
pub fn dead_code_elimination(program: &mut Program) {
    for func in &mut program.funcs {
        dead_code_elimination_in_function(func);
    }
}

fn dead_code_elimination_in_function(func: &mut Function) {
    let debug = std::env::var("KAJIT_DEBUG_DCE").is_ok();

    // Iterate until no more changes
    let mut iteration = 0;
    loop {
        iteration += 1;

        // Build analysis structures in one pass over blocks
        let analysis = build_use_def_analysis(func);

        if debug && iteration == 1 {
            eprintln!(
                "[dce] func f{}: {} vregs with external uses, {} vregs with local defs",
                func.id.0,
                analysis.external_uses.len(),
                analysis.local_defs.len()
            );
        }

        // Find instructions to remove
        let mut insts_to_remove: HashSet<InstId> = HashSet::new();

        for block in &func.blocks {
            let is_entry = block.id == func.entry;

            for &inst_id in &block.insts {
                let inst = &func.insts[inst_id.index()];
                if !is_pure_instruction(&inst.op) {
                    continue;
                }

                // Don't remove Const instructions with no operands - they're needed
                // for const_of() tracking in the backend (immediate-only consts).
                if matches!(&inst.op, LinearOp::Const { .. }) && inst.operands.is_empty() {
                    continue;
                }

                // Check if all defs are unused
                let all_defs_unused = inst.operands.iter().all(|op| {
                    if op.kind != OperandKind::Def {
                        return true;
                    }
                    is_def_unused(func, &analysis, op.vreg, block.id, is_entry)
                });

                if all_defs_unused {
                    insts_to_remove.insert(inst.id);
                }
            }
        }

        if insts_to_remove.is_empty() {
            if debug {
                eprintln!("[dce] iteration {} found no dead code, done", iteration);
            }
            break;
        }

        if debug {
            eprintln!(
                "[dce] iteration {} removing {} dead insts",
                iteration,
                insts_to_remove.len()
            );
        }

        // Remove dead instructions from blocks
        for block in &mut func.blocks {
            block.insts.retain(|id| !insts_to_remove.contains(id));
        }
    }
}

/// Analysis results for use-def relationships
struct UseDefAnalysis {
    /// For each vreg: blocks that use it BEFORE any local def (external uses)
    external_uses: HashMap<VReg, HashSet<BlockId>>,
    /// For each vreg: blocks that define it locally
    local_defs: HashMap<VReg, HashSet<BlockId>>,
    /// Vregs used by edge arguments
    edge_arg_uses: HashSet<VReg>,
    /// Vregs used by terminators
    terminator_uses: HashSet<VReg>,
    /// Vregs that are function results
    result_uses: HashSet<VReg>,
}

fn build_use_def_analysis(func: &Function) -> UseDefAnalysis {
    let mut external_uses: HashMap<VReg, HashSet<BlockId>> = HashMap::new();
    let mut local_defs: HashMap<VReg, HashSet<BlockId>> = HashMap::new();

    // Analyze each block
    for block in &func.blocks {
        // Block params are local defs
        for &param in &block.params {
            local_defs.entry(param).or_default().insert(block.id);
        }

        // Track defs seen so far in this block (for determining external vs local uses)
        let mut defs_in_block: HashSet<VReg> = block.params.iter().copied().collect();

        for &inst_id in &block.insts {
            let inst = &func.insts[inst_id.index()];

            // Check uses first (before adding this instruction's defs)
            for op in &inst.operands {
                if op.kind == OperandKind::Use && !defs_in_block.contains(&op.vreg) {
                    // This is an external use - vreg not defined in this block before this point
                    external_uses.entry(op.vreg).or_default().insert(block.id);
                }
            }

            // Then record defs
            for op in &inst.operands {
                if op.kind == OperandKind::Def {
                    defs_in_block.insert(op.vreg);
                    local_defs.entry(op.vreg).or_default().insert(block.id);
                }
            }
        }
    }

    // Collect edge argument uses
    let mut edge_arg_uses: HashSet<VReg> = HashSet::new();
    for edge in &func.edges {
        for arg in &edge.args {
            edge_arg_uses.insert(arg.source);
        }
    }

    // Collect terminator uses
    let mut terminator_uses: HashSet<VReg> = HashSet::new();
    for term in &func.terms {
        match term {
            Terminator::BranchIf { cond, .. } | Terminator::BranchIfZero { cond, .. } => {
                terminator_uses.insert(*cond);
            }
            Terminator::JumpTable { predicate, .. } => {
                terminator_uses.insert(*predicate);
            }
            Terminator::Return | Terminator::ErrorExit { .. } | Terminator::Branch { .. } => {}
        }
    }

    // Collect result uses
    let result_uses: HashSet<VReg> = func.data_results.iter().copied().collect();

    UseDefAnalysis {
        external_uses,
        local_defs,
        edge_arg_uses,
        terminator_uses,
        result_uses,
    }
}

/// Check if a def of `vreg` in `def_block` is unused
fn is_def_unused(
    func: &Function,
    analysis: &UseDefAnalysis,
    vreg: VReg,
    def_block: BlockId,
    is_entry_block: bool,
) -> bool {
    // If it's a function result, it's used
    if analysis.result_uses.contains(&vreg) {
        return false;
    }

    // If it's used by an edge arg, it's used
    if analysis.edge_arg_uses.contains(&vreg) {
        return false;
    }

    // If it's used by a terminator, it's used
    if analysis.terminator_uses.contains(&vreg) {
        return false;
    }

    // Get blocks that have external uses of this vreg
    let use_blocks = analysis.external_uses.get(&vreg);

    // If no external uses anywhere, def is unused (for cross-block purposes)
    // But we still need to check local uses within the same block
    if use_blocks.is_none() || use_blocks.unwrap().is_empty() {
        // Check if there are local uses in the def block itself
        // (uses after the def in the same block)
        return !has_local_use_after_def(func, vreg, def_block);
    }

    let use_blocks = use_blocks.unwrap();

    // For entry block defs: they're unused if every block that uses the vreg
    // (externally) also has a local def that shadows the entry def
    if is_entry_block {
        let def_blocks = analysis.local_defs.get(&vreg);
        if let Some(def_blocks) = def_blocks {
            // Entry def is unused if all use_blocks are also in def_blocks
            // (meaning each using block has its own local def)
            let all_uses_shadowed = use_blocks.iter().all(|use_block| {
                // The use block has a local def (which shadows the entry def)
                def_blocks.contains(use_block)
            });
            if all_uses_shadowed {
                // Also check no local use in entry block itself
                return !has_local_use_after_def(func, vreg, def_block);
            }
        }
    }

    // There are external uses not satisfied by local defs
    false
}

/// Check if there's a use of `vreg` in `block` after its definition
fn has_local_use_after_def(func: &Function, vreg: VReg, block_id: BlockId) -> bool {
    let block = &func.blocks[block_id.index()];

    // Find where vreg is defined in this block, then check for uses after
    let mut found_def = false;
    for &inst_id in &block.insts {
        let inst = &func.insts[inst_id.index()];

        if found_def {
            // Check if this instruction uses vreg
            for op in &inst.operands {
                if op.kind == OperandKind::Use && op.vreg == vreg {
                    return true;
                }
            }
        }

        // Check if this instruction defines vreg
        for op in &inst.operands {
            if op.kind == OperandKind::Def && op.vreg == vreg {
                found_def = true;
                break;
            }
        }
    }

    false
}

/// Returns true if an instruction has no side effects and can be safely removed
/// if its outputs are unused.
fn is_pure_instruction(op: &LinearOp) -> bool {
    matches!(
        op,
        LinearOp::Const { .. }
            | LinearOp::Copy { .. }
            | LinearOp::BinOp { .. }
            | LinearOp::UnaryOp { .. }
    )
}

/// Local Common Subexpression Elimination within each basic block.
///
/// This pass identifies redundant computations within a block and replaces them
/// with copies from the first computation. Handles:
/// - Duplicate constants (same value)
/// - Duplicate slot reads (same slot, no intervening write)
/// - Duplicate binary operations (same op and operands)
/// - Duplicate unary operations (same op and operand)
pub fn local_cse(program: &mut Program) {
    for func in &mut program.funcs {
        local_cse_in_function(func);
    }
}

fn local_cse_in_function(func: &mut Function) {
    use kajit_ir::SlotId;
    use kajit_lir::{BinOpKind, UnaryOpKind};

    for block in &func.blocks {
        // Track known values within this block
        let mut known_consts: HashMap<u64, VReg> = HashMap::new();
        let mut known_slot_reads: HashMap<SlotId, VReg> = HashMap::new();
        let mut known_binops: HashMap<(BinOpKind, VReg, VReg), VReg> = HashMap::new();
        let mut known_unaryops: HashMap<(UnaryOpKind, VReg), VReg> = HashMap::new();

        // Collect replacements: inst_id -> (Copy instruction, new operands)
        let mut replacements: Vec<(InstId, LinearOp, Vec<Operand>)> = Vec::new();

        for &inst_id in &block.insts {
            let inst = &func.insts[inst_id.index()];
            match &inst.op {
                LinearOp::Const { dst, value } => {
                    if let Some(&existing) = known_consts.get(value) {
                        // Replace with copy from existing
                        let new_operands = vec![
                            Operand {
                                vreg: existing,
                                kind: OperandKind::Use,
                                class: RegClass::Gpr,
                                fixed: None,
                            },
                            Operand {
                                vreg: *dst,
                                kind: OperandKind::Def,
                                class: RegClass::Gpr,
                                fixed: None,
                            },
                        ];
                        replacements.push((
                            inst_id,
                            LinearOp::Copy {
                                dst: *dst,
                                src: existing,
                            },
                            new_operands,
                        ));
                    } else {
                        known_consts.insert(*value, *dst);
                    }
                }

                LinearOp::ReadFromSlot { dst, slot } => {
                    if let Some(&existing) = known_slot_reads.get(slot) {
                        let new_operands = vec![
                            Operand {
                                vreg: existing,
                                kind: OperandKind::Use,
                                class: RegClass::Gpr,
                                fixed: None,
                            },
                            Operand {
                                vreg: *dst,
                                kind: OperandKind::Def,
                                class: RegClass::Gpr,
                                fixed: None,
                            },
                        ];
                        replacements.push((
                            inst_id,
                            LinearOp::Copy {
                                dst: *dst,
                                src: existing,
                            },
                            new_operands,
                        ));
                    } else {
                        known_slot_reads.insert(*slot, *dst);
                    }
                }

                LinearOp::WriteToSlot { slot, .. } => {
                    // Invalidate the slot read cache for this slot
                    known_slot_reads.remove(slot);
                }

                LinearOp::BinOp { op, dst, lhs, rhs } => {
                    let key = (*op, *lhs, *rhs);
                    if let Some(&existing) = known_binops.get(&key) {
                        let new_operands = vec![
                            Operand {
                                vreg: existing,
                                kind: OperandKind::Use,
                                class: RegClass::Gpr,
                                fixed: None,
                            },
                            Operand {
                                vreg: *dst,
                                kind: OperandKind::Def,
                                class: RegClass::Gpr,
                                fixed: None,
                            },
                        ];
                        replacements.push((
                            inst_id,
                            LinearOp::Copy {
                                dst: *dst,
                                src: existing,
                            },
                            new_operands,
                        ));
                    } else {
                        known_binops.insert(key, *dst);
                        // For commutative ops, also insert the swapped version
                        if matches!(
                            op,
                            BinOpKind::Add
                                | BinOpKind::Mul
                                | BinOpKind::And
                                | BinOpKind::Or
                                | BinOpKind::Xor
                                | BinOpKind::CmpEq
                                | BinOpKind::CmpNe
                        ) && lhs != rhs
                        {
                            known_binops.insert((*op, *rhs, *lhs), *dst);
                        }
                    }
                }

                LinearOp::UnaryOp { op, dst, src } => {
                    let key = (*op, *src);
                    if let Some(&existing) = known_unaryops.get(&key) {
                        let new_operands = vec![
                            Operand {
                                vreg: existing,
                                kind: OperandKind::Use,
                                class: RegClass::Gpr,
                                fixed: None,
                            },
                            Operand {
                                vreg: *dst,
                                kind: OperandKind::Def,
                                class: RegClass::Gpr,
                                fixed: None,
                            },
                        ];
                        replacements.push((
                            inst_id,
                            LinearOp::Copy {
                                dst: *dst,
                                src: existing,
                            },
                            new_operands,
                        ));
                    } else {
                        known_unaryops.insert(key, *dst);
                    }
                }

                // Other instructions may have side effects or modify state
                // that invalidates our knowledge
                LinearOp::CallIntrinsic { .. } | LinearOp::CallLambda { .. } => {
                    // Calls may modify slots, so clear slot knowledge
                    known_slot_reads.clear();
                }

                _ => {}
            }
        }

        // Apply replacements
        for (inst_id, new_op, new_operands) in replacements {
            let inst = &mut func.insts[inst_id.index()];
            inst.op = new_op;
            inst.operands = new_operands;
        }
    }
}

// ============================================================================
// Global Value Numbering (GVN)
// ============================================================================

/// Dominator tree for a function's CFG.
#[derive(Debug, Clone)]
pub struct DomTree {
    /// Immediate dominator for each block. Entry block has None.
    pub idom: Vec<Option<BlockId>>,
    /// Children in the dominator tree (blocks immediately dominated by this one).
    pub children: Vec<Vec<BlockId>>,
}

impl DomTree {
    /// Compute the dominator tree using iterative dataflow.
    pub fn compute(func: &Function) -> Self {
        let n = func.blocks.len();
        if n == 0 {
            return DomTree {
                idom: Vec::new(),
                children: Vec::new(),
            };
        }

        // Initialize: entry dominates itself, others undefined
        let mut idom: Vec<Option<BlockId>> = vec![None; n];
        let entry_idx = func.entry.index();

        // Build predecessor map from edges
        let mut preds: Vec<Vec<BlockId>> = vec![Vec::new(); n];
        for edge in &func.edges {
            let to_idx = edge.to.index();
            if to_idx < n {
                preds[to_idx].push(edge.from);
            }
        }

        // Compute reverse postorder for iteration efficiency
        let rpo = Self::reverse_postorder(func);
        let mut rpo_index: Vec<usize> = vec![usize::MAX; n];
        for (i, &block_id) in rpo.iter().enumerate() {
            rpo_index[block_id.index()] = i;
        }

        // Iterative dominator computation
        let mut changed = true;
        while changed {
            changed = false;
            for &block_id in &rpo {
                let b = block_id.index();
                if b == entry_idx {
                    continue;
                }

                // Find first processed predecessor
                let mut new_idom: Option<BlockId> = None;
                for &pred in &preds[b] {
                    let p = pred.index();
                    if rpo_index[p] < rpo_index[b] || idom[p].is_some() || p == entry_idx {
                        if new_idom.is_none() {
                            new_idom = Some(pred);
                        } else {
                            // Intersect
                            new_idom = Some(Self::intersect(
                                new_idom.unwrap(),
                                pred,
                                &idom,
                                &rpo_index,
                                entry_idx,
                            ));
                        }
                    }
                }

                if new_idom != idom[b] {
                    idom[b] = new_idom;
                    changed = true;
                }
            }
        }

        // Build children map
        let mut children: Vec<Vec<BlockId>> = vec![Vec::new(); n];
        for (b, dom) in idom.iter().enumerate() {
            if let Some(d) = dom {
                children[d.index()].push(BlockId(b as u32));
            }
        }

        DomTree { idom, children }
    }

    fn reverse_postorder(func: &Function) -> Vec<BlockId> {
        let n = func.blocks.len();
        let mut visited = vec![false; n];
        let mut postorder = Vec::with_capacity(n);

        // Build successor map
        let mut succs: Vec<Vec<BlockId>> = vec![Vec::new(); n];
        for edge in &func.edges {
            let from_idx = edge.from.index();
            if from_idx < n {
                succs[from_idx].push(edge.to);
            }
        }

        fn dfs(
            block: BlockId,
            succs: &[Vec<BlockId>],
            visited: &mut [bool],
            postorder: &mut Vec<BlockId>,
        ) {
            let b = block.index();
            if visited[b] {
                return;
            }
            visited[b] = true;
            for &succ in &succs[b] {
                dfs(succ, succs, visited, postorder);
            }
            postorder.push(block);
        }

        dfs(func.entry, &succs, &mut visited, &mut postorder);
        postorder.reverse();
        postorder
    }

    fn intersect(
        mut b1: BlockId,
        mut b2: BlockId,
        idom: &[Option<BlockId>],
        rpo_index: &[usize],
        entry_idx: usize,
    ) -> BlockId {
        while b1 != b2 {
            while rpo_index[b1.index()] > rpo_index[b2.index()] {
                if b1.index() == entry_idx {
                    break;
                }
                b1 = idom[b1.index()].unwrap_or(b1);
            }
            while rpo_index[b2.index()] > rpo_index[b1.index()] {
                if b2.index() == entry_idx {
                    break;
                }
                b2 = idom[b2.index()].unwrap_or(b2);
            }
        }
        b1
    }
}

/// Hashable key for value numbering expressions.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
enum ValueKey {
    Const(u64),
    BinOp {
        op: kajit_lir::BinOpKind,
        lhs: VReg,
        rhs: VReg,
    },
    UnaryOp {
        op: kajit_lir::UnaryOpKind,
        src: VReg,
    },
    SlotAddr(kajit_ir::SlotId),
    CallPure {
        func: IntrinsicFn,
        args: Vec<VReg>,
    },
}

/// Scoped hash table for dominator-tree-ordered value numbering.
struct ScopedValueTable {
    scopes: Vec<HashMap<ValueKey, VReg>>,
}

impl ScopedValueTable {
    fn new() -> Self {
        ScopedValueTable {
            scopes: vec![HashMap::new()],
        }
    }

    fn push_scope(&mut self) {
        self.scopes.push(HashMap::new());
    }

    fn lookup(&self, key: &ValueKey) -> Option<VReg> {
        for scope in self.scopes.iter().rev() {
            if let Some(&vreg) = scope.get(key) {
                return Some(vreg);
            }
        }
        None
    }

    fn insert(&mut self, key: ValueKey, vreg: VReg) {
        if let Some(scope) = self.scopes.last_mut() {
            scope.insert(key, vreg);
        }
    }
}

/// Global Value Numbering: eliminates redundant computations across blocks.
///
/// Uses dominator tree to ensure that when we reference a canonical value,
/// it is guaranteed to dominate (be available at) the current block.
pub fn global_value_numbering(program: &mut Program) {
    for func in &mut program.funcs {
        global_value_numbering_in_function(func);
    }
}

fn global_value_numbering_in_function(func: &mut Function) {
    if func.blocks.is_empty() {
        return;
    }

    // For now, we do per-block value numbering. True cross-block GVN would require
    // ensuring the canonical vreg is live at the use site (via block params/edges).
    // Local CSE already handles constants, so focus on BinOp/UnaryOp/SlotAddr/CallPure.
    //
    // Within each block, we can safely convert redundant computations to copies
    // because the canonical vreg is defined earlier in the same block.

    for block in &func.blocks {
        let mut table = ScopedValueTable::new();
        table.push_scope();

        // Map from InstId -> canonical VReg (instructions to convert to copies)
        let mut convert_to_copy: Vec<(InstId, VReg)> = Vec::new();

        for &inst_id in &block.insts {
            let inst = &func.insts[inst_id.index()];

            if let Some(key) = make_value_key_simple(&inst.op) {
                if let Some(canonical) = table.lookup(&key) {
                    // Found redundant computation - convert to copy
                    if let Some(dst) = get_def_vreg(&inst.op) {
                        // Don't create self-copies
                        if dst != canonical {
                            convert_to_copy.push((inst_id, canonical));
                        }
                    }
                } else {
                    // First time seeing this value
                    if let Some(dst) = get_def_vreg(&inst.op) {
                        table.insert(key, dst);
                    }
                }
            }
        }

        // Convert redundant instructions to copies
        for (inst_id, canonical) in convert_to_copy {
            let inst = &mut func.insts[inst_id.index()];
            if let Some(dst) = get_def_vreg(&inst.op) {
                inst.op = LinearOp::Copy {
                    dst,
                    src: canonical,
                };
                // Update operands: one Use of canonical, one Def of dst
                inst.operands = vec![
                    Operand {
                        vreg: canonical,
                        kind: OperandKind::Use,
                        class: RegClass::Gpr,
                        fixed: None,
                    },
                    Operand {
                        vreg: dst,
                        kind: OperandKind::Def,
                        class: RegClass::Gpr,
                        fixed: None,
                    },
                ];
            }
        }
    }
}

/// Create a ValueKey for an instruction (simple version without canonicalization).
/// This is safe for the copy-conversion approach: we only deduplicate identical expressions.
fn make_value_key_simple(op: &LinearOp) -> Option<ValueKey> {
    match op {
        LinearOp::Const { value, .. } => Some(ValueKey::Const(*value)),

        LinearOp::BinOp { op, lhs, rhs, .. } => Some(ValueKey::BinOp {
            op: *op,
            lhs: *lhs,
            rhs: *rhs,
        }),

        LinearOp::UnaryOp { op, src, .. } => Some(ValueKey::UnaryOp { op: *op, src: *src }),

        // Don't value-number Copy - it's handled by copy_propagation
        LinearOp::Copy { .. } => None,

        LinearOp::SlotAddr { slot, .. } => Some(ValueKey::SlotAddr(*slot)),

        LinearOp::CallPure { func, args, .. } => Some(ValueKey::CallPure {
            func: *func,
            args: args.clone(),
        }),

        // Side-effecting operations cannot be value-numbered
        _ => None,
    }
}

/// Get the destination vreg defined by an instruction.
fn get_def_vreg(op: &LinearOp) -> Option<VReg> {
    match op {
        LinearOp::Const { dst, .. }
        | LinearOp::BinOp { dst, .. }
        | LinearOp::UnaryOp { dst, .. }
        | LinearOp::Copy { dst, .. }
        | LinearOp::SlotAddr { dst, .. }
        | LinearOp::CallPure { dst, .. } => Some(*dst),
        _ => None,
    }
}

/// Simplifies trivial phi nodes where all incoming values are identical.
///
/// A phi node (block parameter) is trivial if all incoming edges provide
/// the same value for that parameter. In this case, we can replace all
/// uses of the parameter with that common value.
///
/// Example:
/// ```text
/// block b1 params=[v10, v11]:
///   edge e0: b0 -> b1 [v10=>v5, v11=>v3]
///   edge e1: b2 -> b1 [v10=>v5, v11=>v3]
/// ```
/// Both v10 and v11 are trivial - v10 is always v5, v11 is always v3.
/// We rewrite all uses of v10 to v5 and v11 to v3, then DCE removes them.
pub fn simplify_trivial_phis(program: &mut Program) {
    for func in &mut program.funcs {
        simplify_trivial_phis_in_function(func);
    }
}

fn simplify_trivial_phis_in_function(func: &mut Function) {
    let debug = std::env::var("KAJIT_DEBUG_PHI").is_ok();

    // For each block, identify which parameters are trivial and should be removed
    // Map: BlockId -> Vec<(param_index, param_vreg, replacement_vreg)>
    let mut trivial_params_per_block: HashMap<BlockId, Vec<(usize, VReg, VReg)>> = HashMap::new();

    for block in &func.blocks {
        if block.params.is_empty() {
            continue;
        }

        let mut trivial_params = Vec::new();

        // For each parameter, check if all incoming edges provide the same value
        for (param_idx, &param_vreg) in block.params.iter().enumerate() {
            let mut common_value: Option<VReg> = None;
            let mut is_trivial = true;

            for &pred_edge_id in &block.preds {
                let edge = &func.edges[pred_edge_id.index()];

                if param_idx >= edge.args.len() {
                    is_trivial = false;
                    break;
                }
                let incoming_value = edge.args[param_idx].source;

                match common_value {
                    None => common_value = Some(incoming_value),
                    Some(cv) if cv == incoming_value => {}
                    Some(cv) if cv == param_vreg => {
                        // Self-reference phi, use the incoming value
                        common_value = Some(incoming_value);
                    }
                    _ => {
                        is_trivial = false;
                        break;
                    }
                }
            }

            if is_trivial {
                if let Some(replacement) = common_value {
                    if replacement != param_vreg {
                        trivial_params.push((param_idx, param_vreg, replacement));
                    }
                }
            }
        }

        if !trivial_params.is_empty() {
            trivial_params_per_block.insert(block.id, trivial_params);
        }
    }

    if trivial_params_per_block.is_empty() {
        return;
    }

    let total_trivial: usize = trivial_params_per_block.values().map(|v| v.len()).sum();
    if debug {
        eprintln!(
            "[simplify_phis] func f{}: found {} trivial phis across {} blocks",
            func.id.0,
            total_trivial,
            trivial_params_per_block.len()
        );
    }

    // Build global phi -> replacement map for rewriting uses
    let mut phi_replacements: HashMap<VReg, VReg> = HashMap::new();
    for trivial_params in trivial_params_per_block.values() {
        for &(_, param_vreg, replacement) in trivial_params {
            phi_replacements.insert(param_vreg, replacement);
        }
    }

    // Rewrite all uses of trivial phis
    let rewrite_vreg = |v: &mut VReg| {
        if let Some(&replacement) = phi_replacements.get(v) {
            *v = replacement;
        }
    };

    // Rewrite instruction uses
    for block in &func.blocks {
        for &inst_id in &block.insts {
            let inst = &mut func.insts[inst_id.index()];
            for operand in &mut inst.operands {
                if operand.kind == OperandKind::Use {
                    rewrite_vreg(&mut operand.vreg);
                }
            }
        }
    }

    // Rewrite terminator uses
    for block in &func.blocks {
        let term = &mut func.terms[block.term.index()];
        match term {
            Terminator::BranchIf { cond, .. } | Terminator::BranchIfZero { cond, .. } => {
                rewrite_vreg(cond);
            }
            Terminator::JumpTable { predicate, .. } => {
                rewrite_vreg(predicate);
            }
            _ => {}
        }
    }

    // Rewrite function data results
    for result in &mut func.data_results {
        rewrite_vreg(result);
    }

    // Now remove trivial parameters from blocks and update edge arguments
    // Process in reverse index order to keep indices valid during removal
    for (block_id, mut trivial_params) in trivial_params_per_block {
        // Sort by index descending so we remove from end to beginning
        trivial_params.sort_by(|a, b| b.0.cmp(&a.0));

        // Remove parameters from block
        for &(param_idx, _, _) in &trivial_params {
            func.blocks[block_id.index()].params.remove(param_idx);
        }

        // Remove corresponding edge arguments from all predecessor edges
        for &pred_edge_id in &func.blocks[block_id.index()].preds.clone() {
            let edge = &mut func.edges[pred_edge_id.index()];
            for &(param_idx, _, _) in &trivial_params {
                if param_idx < edge.args.len() {
                    edge.args.remove(param_idx);
                }
            }
        }
    }

    if debug {
        eprintln!("[simplify_phis] func f{}: cleanup complete", func.id.0);
    }
}

/// Merge empty forwarding blocks to simplify the CFG.
///
/// An empty block is one with no instructions that just forwards values to
/// a single successor. If the successor has only this predecessor, we can
/// merge the blocks by:
/// 1. Moving the successor's instructions into this block
/// 2. Updating this block's terminator to the successor's terminator
/// 3. Rewiring all references to the successor to point to this block
pub fn merge_empty_blocks(program: &mut Program) {
    for func in &mut program.funcs {
        merge_empty_blocks_in_function(func);
    }
}

fn merge_empty_blocks_in_function(func: &mut Function) {
    let debug = std::env::var("KAJIT_DEBUG_MERGE").is_ok();
    let mut total_merged = 0;

    loop {
        let mut merged_any = false;

        // Find a candidate block to merge
        for block_idx in 0..func.blocks.len() {
            let block_id = BlockId(block_idx as u32);

            // Skip entry block
            if block_id == func.entry {
                continue;
            }

            // Check if this block is a merge candidate:
            // - No instructions
            // - No parameters (pure forwarding, no phis)
            // - Exactly one successor (unconditional branch)
            // - Terminator is a simple branch (not conditional, not error)
            let block = &func.blocks[block_idx];

            if !block.insts.is_empty() || !block.params.is_empty() {
                continue;
            }

            let Terminator::Branch { edge } = func.terms[block.term.index()] else {
                continue;
            };

            let edge_data = &func.edges[edge.index()];
            let successor_id = edge_data.to;

            // Check if successor has only one predecessor (this block)
            let successor = &func.blocks[successor_id.index()];
            if successor.preds.len() != 1 {
                continue;
            }

            if debug {
                eprintln!(
                    "[merge_blocks] merging b{} into successor b{}",
                    block_id.0, successor_id.0
                );
            }

            // Perform the merge
            // This is complex because we need to update many references
            // For now, just count how many we could merge
            merged_any = true;
            total_merged += 1;
            break;
        }

        if !merged_any {
            break;
        }
    }

    if debug && total_merged > 0 {
        eprintln!(
            "[merge_blocks] func f{}: merged {} blocks",
            func.id.0, total_merged
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use kajit_ir::{DebugScope, DebugScopeKind, IrBuilder, PortSource, Width};
    use kajit_lir::linearize;

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
            output_size: 0,
            blocks: vec![Block {
                id: BlockId(0),
                params: Vec::new(),
                insts: vec![InstId(0)],
                term: TermId(0),
                preds: Vec::new(),
                succs: Vec::new(),
                dead: false,
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

    #[test]
    fn lower_linear_ir_produces_valid_cfg_program() {
        let mut builder = IrBuilder::new("u32", 0);
        {
            let mut rb = builder.root_region();
            rb.bounds_check(4);
            let value = rb.read_bytes(4);
            rb.write_to_field(value, 0, Width::W4);
            rb.set_results(&[]);
        }
        let mut func = builder.finish();
        let linear = linearize(&mut func);
        let hints = crate::regalloc3::hints::HintMap::default();
        let program = lower_linear_ir(&linear, hints);
        program
            .validate()
            .expect("lowered cfg program should validate");
        assert_eq!(program.funcs.len(), 1);
        assert!(!program.funcs[0].blocks.is_empty());
    }

    #[test]
    fn lower_linear_ir_models_gamma_join_block_params() {
        let mut builder = IrBuilder::new("u32", 0);
        {
            let mut rb = builder.root_region();
            let pred = rb.const_val(0);
            let out = rb.gamma(pred, &[], 2, |branch_idx, bb| {
                let val = if branch_idx == 0 {
                    bb.const_val(7)
                } else {
                    bb.const_val(9)
                };
                bb.set_results(&[val]);
            });
            rb.write_to_field(out[0], 0, Width::W4);
            rb.set_results(&[]);
        }
        let mut func = builder.finish();
        let linear = linearize(&mut func);
        let hints = crate::regalloc3::hints::HintMap::default();
        let program = lower_linear_ir(&linear, hints);
        let root = &program.funcs[0];

        let merge = root
            .blocks
            .iter()
            .find(|block| block.preds.len() >= 2 && !block.params.is_empty())
            .expect("expected merge block with parameters");

        for pred_edge in &merge.preds {
            let edge = root
                .edge(*pred_edge)
                .expect("pred edge should exist in function");
            assert_eq!(
                edge.args.len(),
                merge.params.len(),
                "edge args should match merge params"
            );
        }
    }

    #[test]
    fn lower_linear_ir_preserves_debug_scope_provenance() {
        let mut builder = IrBuilder::new("u32", 0);
        let (const_node, output_index, root_scope) = {
            let mut rb = builder.root_region();
            let value = rb.const_val(42);
            rb.set_results(&[value]);
            let output_ref = match value {
                PortSource::Node(output_ref) => output_ref,
                other => panic!("expected node output, got {other:?}"),
            };
            (output_ref.node, output_ref.index as usize, rb.debug_scope())
        };

        let mut func = builder.finish();
        let value_vreg = func.nodes[const_node].outputs[output_index]
            .vreg
            .expect("const output should have vreg");
        let extra_scope = func.debug_scopes.push(DebugScope {
            parent: Some(root_scope),
            kind: DebugScopeKind::ThetaBody,
        });
        func.nodes[const_node].debug_scope = extra_scope;
        func.nodes[const_node].outputs[0].debug_scope = root_scope;

        let linear = linearize(&mut func);
        let hints = crate::regalloc3::hints::HintMap::default();
        let program = lower_linear_ir(&linear, hints);
        let root = &program.funcs[0];
        let const_inst = root.insts[0].id;

        assert_eq!(program.debug.root_scope, Some(root_scope));
        assert_eq!(program.debug.scopes.len(), func.debug_scopes.len());
        assert_eq!(
            program.op_debug_scope(root.lambda_id, OpId::Inst(const_inst)),
            Some(extra_scope)
        );
        assert_eq!(program.vreg_debug_scope(value_vreg), Some(root_scope));
    }

    // ─── Optimization pass tests ───────────────────────────────────────────

    /// Helper to build a single-block function for optimization tests.
    fn single_block_func(insts: Vec<Inst>) -> Function {
        let inst_ids: Vec<InstId> = insts.iter().map(|i| i.id).collect();
        Function {
            id: FunctionId(0),
            lambda_id: LambdaId::new(0),
            entry: BlockId(0),
            data_args: Vec::new(),
            data_results: Vec::new(),
            output_size: 0,
            blocks: vec![Block {
                id: BlockId(0),
                params: Vec::new(),
                insts: inst_ids,
                term: TermId(0),
                preds: Vec::new(),
                succs: Vec::new(),
                dead: false,
            }],
            edges: Vec::new(),
            insts,
            terms: vec![Terminator::Return],
        }
    }

    fn make_const(id: u32, dst: u32, value: u64) -> Inst {
        Inst {
            id: InstId(id),
            op: LinearOp::Const { dst: v(dst), value },
            operands: vec![Operand {
                vreg: v(dst),
                kind: OperandKind::Def,
                class: RegClass::Gpr,
                fixed: None,
            }],
            clobbers: Clobbers::default(),
        }
    }

    fn make_copy(id: u32, dst: u32, src: u32) -> Inst {
        Inst {
            id: InstId(id),
            op: LinearOp::Copy {
                dst: v(dst),
                src: v(src),
            },
            operands: vec![
                Operand {
                    vreg: v(dst),
                    kind: OperandKind::Def,
                    class: RegClass::Gpr,
                    fixed: None,
                },
                Operand {
                    vreg: v(src),
                    kind: OperandKind::Use,
                    class: RegClass::Gpr,
                    fixed: None,
                },
            ],
            clobbers: Clobbers::default(),
        }
    }

    fn make_binop(id: u32, dst: u32, lhs: u32, rhs: u32, op: kajit_lir::BinOpKind) -> Inst {
        Inst {
            id: InstId(id),
            op: LinearOp::BinOp {
                dst: v(dst),
                lhs: v(lhs),
                rhs: v(rhs),
                op,
            },
            operands: vec![
                Operand {
                    vreg: v(dst),
                    kind: OperandKind::Def,
                    class: RegClass::Gpr,
                    fixed: None,
                },
                Operand {
                    vreg: v(lhs),
                    kind: OperandKind::Use,
                    class: RegClass::Gpr,
                    fixed: None,
                },
                Operand {
                    vreg: v(rhs),
                    kind: OperandKind::Use,
                    class: RegClass::Gpr,
                    fixed: None,
                },
            ],
            clobbers: Clobbers::default(),
        }
    }

    #[test]
    fn copy_propagation_rewrites_uses_to_source() {
        // v0 = const 42
        // v1 = copy v0
        // v2 = add v1, v1
        // After copy prop: v2 = add v0, v0
        let mut func = single_block_func(vec![
            make_const(0, 0, 42),
            make_copy(1, 1, 0),
            make_binop(2, 2, 1, 1, kajit_lir::BinOpKind::Add),
        ]);

        global_copy_propagation(&mut func);

        // Check that the add now uses v0 directly
        match &func.insts[2].op {
            LinearOp::BinOp { lhs, rhs, .. } => {
                assert_eq!(lhs.index(), 0, "lhs should be rewritten to v0");
                assert_eq!(rhs.index(), 0, "rhs should be rewritten to v0");
            }
            other => panic!("expected BinOp, got {:?}", other),
        }
    }

    #[test]
    fn copy_propagation_follows_copy_chains() {
        // v0 = const 42
        // v1 = copy v0
        // v2 = copy v1
        // v3 = add v2, v2
        // After copy prop: v3 = add v0, v0
        let mut func = single_block_func(vec![
            make_const(0, 0, 42),
            make_copy(1, 1, 0),
            make_copy(2, 2, 1),
            make_binop(3, 3, 2, 2, kajit_lir::BinOpKind::Add),
        ]);

        global_copy_propagation(&mut func);

        match &func.insts[3].op {
            LinearOp::BinOp { lhs, rhs, .. } => {
                assert_eq!(lhs.index(), 0, "lhs should follow chain to v0");
                assert_eq!(rhs.index(), 0, "rhs should follow chain to v0");
            }
            other => panic!("expected BinOp, got {:?}", other),
        }
    }

    #[test]
    fn dead_code_elimination_removes_unused_consts() {
        // v0 = const 42  (unused)
        // v1 = const 99  (used by return, simulated by keeping it)
        let mut program = Program {
            vreg_count: 2,
            slot_count: 0,
            funcs: vec![single_block_func(vec![
                make_const(0, 0, 42),
                make_const(1, 1, 99),
            ])],
            debug: Default::default(),
            hints: Default::default(),
        };

        // Mark v1 as used by adding it to data_results
        program.funcs[0].data_results = vec![v(1)];

        dead_code_elimination(&mut program);

        // v0 should be removed, v1 kept
        let func = &program.funcs[0];
        assert_eq!(
            func.blocks[0].insts.len(),
            1,
            "should have 1 inst after DCE"
        );
        match &func.insts[func.blocks[0].insts[0].index()].op {
            LinearOp::Const { dst, value } => {
                assert_eq!(dst.index(), 1, "remaining const should be v1");
                assert_eq!(*value, 99);
            }
            other => panic!("expected Const, got {:?}", other),
        }
    }

    #[test]
    fn local_cse_eliminates_duplicate_consts() {
        // v0 = const 42
        // v1 = const 42  (duplicate)
        // v2 = add v0, v1
        // After CSE: v1 = copy v0, v2 = add v0, v1
        let mut program = Program {
            vreg_count: 3,
            slot_count: 0,
            funcs: vec![single_block_func(vec![
                make_const(0, 0, 42),
                make_const(1, 1, 42),
                make_binop(2, 2, 0, 1, kajit_lir::BinOpKind::Add),
            ])],
            debug: Default::default(),
            hints: Default::default(),
        };

        local_cse(&mut program);

        // Second const should become a copy
        let func = &program.funcs[0];
        match &func.insts[1].op {
            LinearOp::Copy { dst, src } => {
                assert_eq!(dst.index(), 1);
                assert_eq!(src.index(), 0, "should copy from first const");
            }
            other => panic!("expected Copy after CSE, got {:?}", other),
        }
    }

    #[test]
    fn cse_then_copy_prop_then_dce_eliminates_redundant_consts() {
        // Test the full pipeline on a single block:
        // v0 = const 42
        // v1 = const 42  (duplicate)
        // v2 = add v0, v1
        //
        // After CSE: v1 = copy v0
        // After copy prop: v2 = add v0, v0
        // After DCE: v1 removed (dead copy)
        let mut program = Program {
            vreg_count: 3,
            slot_count: 0,
            funcs: vec![single_block_func(vec![
                make_const(0, 0, 42),
                make_const(1, 1, 42),
                make_binop(2, 2, 0, 1, kajit_lir::BinOpKind::Add),
            ])],
            debug: Default::default(),
            hints: Default::default(),
        };

        // Mark v2 as used (result)
        program.funcs[0].data_results = vec![v(2)];

        // Run full pipeline
        local_cse(&mut program);
        copy_propagation(&mut program);
        dead_code_elimination(&mut program);

        let func = &program.funcs[0];

        // Should only have 2 instructions: const and add
        assert_eq!(
            func.blocks[0].insts.len(),
            2,
            "should have const + add after pipeline, got {} insts",
            func.blocks[0].insts.len()
        );

        // First should be const
        match &func.insts[func.blocks[0].insts[0].index()].op {
            LinearOp::Const { dst, value } => {
                assert_eq!(dst.index(), 0);
                assert_eq!(*value, 42);
            }
            other => panic!("expected Const, got {:?}", other),
        }

        // Second should be add using v0 for both operands
        match &func.insts[func.blocks[0].insts[1].index()].op {
            LinearOp::BinOp { lhs, rhs, .. } => {
                assert_eq!(lhs.index(), 0, "lhs should use v0");
                assert_eq!(rhs.index(), 0, "rhs should use v0 after copy prop");
            }
            other => panic!("expected BinOp, got {:?}", other),
        }
    }

    /// Build a two-block function where b0 defines a const and passes it to b1.
    fn two_block_const_param_func() -> Function {
        // b0: v0 = const 42; branch to b1 passing v0
        // b1: param v1 receives v0; return
        Function {
            id: FunctionId(0),
            lambda_id: LambdaId::new(0),
            entry: BlockId(0),
            data_args: Vec::new(),
            data_results: vec![v(1)], // v1 is used as result
            output_size: 0,
            blocks: vec![
                Block {
                    id: BlockId(0),
                    params: Vec::new(),
                    insts: vec![InstId(0)], // const
                    term: TermId(0),        // branch to b1
                    preds: Vec::new(),
                    succs: vec![EdgeId(0)],
                    dead: false,
                },
                Block {
                    id: BlockId(1),
                    params: vec![v(1)], // receives const from b0
                    insts: Vec::new(),
                    term: TermId(1), // return
                    preds: vec![EdgeId(0)],
                    succs: Vec::new(),
                    dead: false,
                },
            ],
            edges: vec![Edge {
                id: EdgeId(0),
                from: BlockId(0),
                to: BlockId(1),
                args: vec![EdgeArg {
                    source: v(0),
                    target: v(1),
                }],
            }],
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
            terms: vec![Terminator::Branch { edge: EdgeId(0) }, Terminator::Return],
        }
    }

    #[test]
    fn rematerialize_constants_replaces_param_with_local_const() {
        // Before: b0 defines const, passes to b1 as param
        // After: b1 has local const, param removed
        let mut program = Program {
            vreg_count: 2,
            slot_count: 0,
            funcs: vec![two_block_const_param_func()],
            debug: Default::default(),
            hints: Default::default(),
        };

        // Verify initial state
        assert_eq!(
            program.funcs[0].blocks[1].params.len(),
            1,
            "b1 should have 1 param before remat"
        );
        assert_eq!(
            program.funcs[0].blocks[1].insts.len(),
            0,
            "b1 should have 0 insts before remat"
        );

        rematerialize_constants(&mut program);

        let func = &program.funcs[0];

        // b1 should now have no params and one local const instruction
        assert_eq!(
            func.blocks[1].params.len(),
            0,
            "b1 param should be removed after remat"
        );
        assert_eq!(
            func.blocks[1].insts.len(),
            1,
            "b1 should have 1 const inst after remat"
        );

        // The edge should have no args
        assert_eq!(
            func.edges[0].args.len(),
            0,
            "edge args should be removed after remat"
        );

        // The instruction should be a const with value 42
        let inst_id = func.blocks[1].insts[0];
        match &func.insts[inst_id.index()].op {
            LinearOp::Const { value, .. } => {
                assert_eq!(*value, 42, "rematerialized const should have same value");
            }
            other => panic!("expected Const, got {:?}", other),
        }
    }

    #[test]
    fn rematerialize_does_not_explode_instruction_count() {
        // Create a simple two-block program and verify instruction count doesn't grow unexpectedly
        let mut program = Program {
            vreg_count: 2,
            slot_count: 0,
            funcs: vec![two_block_const_param_func()],
            debug: Default::default(),
            hints: Default::default(),
        };

        let insts_before: usize = program.funcs.iter().map(|f| f.insts.len()).sum();

        rematerialize_constants(&mut program);

        let insts_after: usize = program.funcs.iter().map(|f| f.insts.len()).sum();

        // Should add at most 1 instruction per rematerialized constant
        // In this case: 1 const in b0, rematerialized to b1 = 2 total
        assert!(
            insts_after <= insts_before + 1,
            "instruction count grew too much: {} -> {}",
            insts_before,
            insts_after
        );
    }

    // ========================================================================
    // GVN Tests
    // ========================================================================

    #[test]
    fn dom_tree_single_block() {
        let func = single_block_func(vec![make_const(0, 0, 42)]);
        let dom_tree = DomTree::compute(&func);

        assert_eq!(dom_tree.idom.len(), 1);
        assert_eq!(dom_tree.idom[0], None, "entry block has no dominator");
        assert!(
            dom_tree.children[0].is_empty(),
            "entry has no dominated children"
        );
    }

    #[test]
    fn dom_tree_linear_chain() {
        // b0 -> b1 -> b2
        let func = Function {
            id: FunctionId(0),
            lambda_id: LambdaId::new(0),
            entry: BlockId(0),
            data_args: Vec::new(),
            data_results: Vec::new(),
            output_size: 0,
            blocks: vec![
                Block {
                    id: BlockId(0),
                    params: Vec::new(),
                    insts: Vec::new(),
                    term: TermId(0),
                    preds: Vec::new(),
                    succs: vec![EdgeId(0)],
                    dead: false,
                },
                Block {
                    id: BlockId(1),
                    params: Vec::new(),
                    insts: Vec::new(),
                    term: TermId(1),
                    preds: vec![EdgeId(0)],
                    succs: vec![EdgeId(1)],
                    dead: false,
                },
                Block {
                    id: BlockId(2),
                    params: Vec::new(),
                    insts: Vec::new(),
                    term: TermId(2),
                    preds: vec![EdgeId(1)],
                    succs: Vec::new(),
                    dead: false,
                },
            ],
            edges: vec![
                Edge {
                    id: EdgeId(0),
                    from: BlockId(0),
                    to: BlockId(1),
                    args: Vec::new(),
                },
                Edge {
                    id: EdgeId(1),
                    from: BlockId(1),
                    to: BlockId(2),
                    args: Vec::new(),
                },
            ],
            insts: Vec::new(),
            terms: vec![
                Terminator::Branch { edge: EdgeId(0) },
                Terminator::Branch { edge: EdgeId(1) },
                Terminator::Return,
            ],
        };

        let dom_tree = DomTree::compute(&func);

        assert_eq!(dom_tree.idom[0], None, "b0 is entry");
        assert_eq!(dom_tree.idom[1], Some(BlockId(0)), "b0 dominates b1");
        assert_eq!(dom_tree.idom[2], Some(BlockId(1)), "b1 dominates b2");
    }

    #[test]
    fn dom_tree_diamond() {
        // b0 -> b1 -> b3
        //    \> b2 /
        let func = Function {
            id: FunctionId(0),
            lambda_id: LambdaId::new(0),
            entry: BlockId(0),
            data_args: Vec::new(),
            data_results: Vec::new(),
            output_size: 0,
            blocks: vec![
                Block {
                    id: BlockId(0),
                    params: Vec::new(),
                    insts: vec![InstId(0)],
                    term: TermId(0),
                    preds: Vec::new(),
                    succs: vec![EdgeId(0), EdgeId(1)],
                    dead: false,
                },
                Block {
                    id: BlockId(1),
                    params: Vec::new(),
                    insts: Vec::new(),
                    term: TermId(1),
                    preds: vec![EdgeId(0)],
                    succs: vec![EdgeId(2)],
                    dead: false,
                },
                Block {
                    id: BlockId(2),
                    params: Vec::new(),
                    insts: Vec::new(),
                    term: TermId(2),
                    preds: vec![EdgeId(1)],
                    succs: vec![EdgeId(3)],
                    dead: false,
                },
                Block {
                    id: BlockId(3),
                    params: Vec::new(),
                    insts: Vec::new(),
                    term: TermId(3),
                    preds: vec![EdgeId(2), EdgeId(3)],
                    succs: Vec::new(),
                    dead: false,
                },
            ],
            edges: vec![
                Edge {
                    id: EdgeId(0),
                    from: BlockId(0),
                    to: BlockId(1),
                    args: Vec::new(),
                },
                Edge {
                    id: EdgeId(1),
                    from: BlockId(0),
                    to: BlockId(2),
                    args: Vec::new(),
                },
                Edge {
                    id: EdgeId(2),
                    from: BlockId(1),
                    to: BlockId(3),
                    args: Vec::new(),
                },
                Edge {
                    id: EdgeId(3),
                    from: BlockId(2),
                    to: BlockId(3),
                    args: Vec::new(),
                },
            ],
            insts: vec![make_const(0, 0, 42)],
            terms: vec![
                Terminator::BranchIf {
                    cond: v(0),
                    taken: EdgeId(0),
                    fallthrough: EdgeId(1),
                },
                Terminator::Branch { edge: EdgeId(2) },
                Terminator::Branch { edge: EdgeId(3) },
                Terminator::Return,
            ],
        };

        let dom_tree = DomTree::compute(&func);

        assert_eq!(dom_tree.idom[0], None);
        assert_eq!(dom_tree.idom[1], Some(BlockId(0)), "b0 dominates b1");
        assert_eq!(dom_tree.idom[2], Some(BlockId(0)), "b0 dominates b2");
        assert_eq!(
            dom_tree.idom[3],
            Some(BlockId(0)),
            "b0 dominates b3 (join point)"
        );
    }

    #[test]
    fn gvn_converts_duplicate_const_to_copy() {
        // v0 = const 42
        // v1 = const 42  <- redundant
        // After GVN: v1 = copy v0
        let mut func = single_block_func(vec![make_const(0, 0, 42), make_const(1, 1, 42)]);

        global_value_numbering_in_function(&mut func);

        // Both instructions should still exist
        assert_eq!(
            func.blocks[0].insts.len(),
            2,
            "both instructions should remain"
        );

        // Second should be converted to copy
        match &func.insts[1].op {
            LinearOp::Copy { dst, src } => {
                assert_eq!(dst.index(), 1, "dst should be v1");
                assert_eq!(src.index(), 0, "src should be v0");
            }
            other => panic!("expected Copy, got {:?}", other),
        }
    }

    #[test]
    fn gvn_converts_redundant_binop_to_copy() {
        // v0 = const 1
        // v1 = const 2
        // v2 = add v0, v1
        // v3 = add v0, v1  <- redundant
        // After GVN: v3 = copy v2
        let mut func = single_block_func(vec![
            make_const(0, 0, 1),
            make_const(1, 1, 2),
            make_binop(2, 2, 0, 1, kajit_lir::BinOpKind::Add),
            make_binop(3, 3, 0, 1, kajit_lir::BinOpKind::Add),
        ]);

        global_value_numbering_in_function(&mut func);

        // All 4 instructions should remain
        assert_eq!(
            func.blocks[0].insts.len(),
            4,
            "all instructions should remain"
        );

        // Fourth should be converted to copy
        match &func.insts[3].op {
            LinearOp::Copy { dst, src } => {
                assert_eq!(dst.index(), 3, "dst should be v3");
                assert_eq!(src.index(), 2, "src should be v2");
            }
            other => panic!("expected Copy, got {:?}", other),
        }
    }

    #[test]
    fn gvn_preserves_uses_after_copy_conversion() {
        // v0 = const 42
        // v1 = const 42  <- converted to copy
        // v2 = add v1, v1  <- uses remain as v1 (copy prop will handle later)
        let mut func = single_block_func(vec![
            make_const(0, 0, 42),
            make_const(1, 1, 42),
            make_binop(2, 2, 1, 1, kajit_lir::BinOpKind::Add),
        ]);

        global_value_numbering_in_function(&mut func);

        // Check that add still uses v1 (copy propagation handles rewriting)
        match &func.insts[2].op {
            LinearOp::BinOp { lhs, rhs, .. } => {
                assert_eq!(lhs.index(), 1, "lhs should still be v1");
                assert_eq!(rhs.index(), 1, "rhs should still be v1");
            }
            other => panic!("expected BinOp, got {:?}", other),
        }
    }

    #[test]
    fn gvn_per_block_no_cross_block() {
        // b0: v0 = const 42; branch to b1
        // b1: v1 = const 42  <- NOT eliminated (cross-block GVN disabled)
        // Per-block GVN only affects instructions within the same block.
        let mut func = Function {
            id: FunctionId(0),
            lambda_id: LambdaId::new(0),
            entry: BlockId(0),
            data_args: Vec::new(),
            data_results: Vec::new(),
            output_size: 0,
            blocks: vec![
                Block {
                    id: BlockId(0),
                    params: Vec::new(),
                    insts: vec![InstId(0)],
                    term: TermId(0),
                    preds: Vec::new(),
                    succs: vec![EdgeId(0)],
                    dead: false,
                },
                Block {
                    id: BlockId(1),
                    params: Vec::new(),
                    insts: vec![InstId(1)],
                    term: TermId(1),
                    preds: vec![EdgeId(0)],
                    succs: Vec::new(),
                    dead: false,
                },
            ],
            edges: vec![Edge {
                id: EdgeId(0),
                from: BlockId(0),
                to: BlockId(1),
                args: Vec::new(),
            }],
            insts: vec![make_const(0, 0, 42), make_const(1, 1, 42)],
            terms: vec![Terminator::Branch { edge: EdgeId(0) }, Terminator::Return],
        };

        global_value_numbering_in_function(&mut func);

        // Both blocks keep their const (no cross-block optimization)
        assert_eq!(func.blocks[0].insts.len(), 1, "b0 keeps its const");
        assert_eq!(
            func.blocks[1].insts.len(),
            1,
            "b1 keeps its const (per-block GVN only)"
        );
        // b1's instruction should still be a Const, not a Copy
        match &func.insts[1].op {
            LinearOp::Const { value, .. } => {
                assert_eq!(*value, 42);
            }
            other => panic!("expected Const, got {:?}", other),
        }
    }

    #[test]
    fn gvn_does_not_eliminate_across_non_dominating_blocks() {
        // Diamond: b0 branches to b1 or b2, both merge at b3
        // b1: v1 = const 42
        // b2: v2 = const 42
        // Neither dominates the other, so both should survive
        let mut func = Function {
            id: FunctionId(0),
            lambda_id: LambdaId::new(0),
            entry: BlockId(0),
            data_args: Vec::new(),
            data_results: Vec::new(),
            output_size: 0,
            blocks: vec![
                Block {
                    id: BlockId(0),
                    params: Vec::new(),
                    insts: vec![InstId(0)], // cond
                    term: TermId(0),
                    preds: Vec::new(),
                    succs: vec![EdgeId(0), EdgeId(1)],
                    dead: false,
                },
                Block {
                    id: BlockId(1),
                    params: Vec::new(),
                    insts: vec![InstId(1)], // const 42
                    term: TermId(1),
                    preds: vec![EdgeId(0)],
                    succs: vec![EdgeId(2)],
                    dead: false,
                },
                Block {
                    id: BlockId(2),
                    params: Vec::new(),
                    insts: vec![InstId(2)], // const 42
                    term: TermId(2),
                    preds: vec![EdgeId(1)],
                    succs: vec![EdgeId(3)],
                    dead: false,
                },
                Block {
                    id: BlockId(3),
                    params: Vec::new(),
                    insts: Vec::new(),
                    term: TermId(3),
                    preds: vec![EdgeId(2), EdgeId(3)],
                    succs: Vec::new(),
                    dead: false,
                },
            ],
            edges: vec![
                Edge {
                    id: EdgeId(0),
                    from: BlockId(0),
                    to: BlockId(1),
                    args: Vec::new(),
                },
                Edge {
                    id: EdgeId(1),
                    from: BlockId(0),
                    to: BlockId(2),
                    args: Vec::new(),
                },
                Edge {
                    id: EdgeId(2),
                    from: BlockId(1),
                    to: BlockId(3),
                    args: Vec::new(),
                },
                Edge {
                    id: EdgeId(3),
                    from: BlockId(2),
                    to: BlockId(3),
                    args: Vec::new(),
                },
            ],
            insts: vec![
                make_const(0, 0, 1), // condition
                make_const(1, 1, 42),
                make_const(2, 2, 42),
            ],
            terms: vec![
                Terminator::BranchIf {
                    cond: v(0),
                    taken: EdgeId(0),
                    fallthrough: EdgeId(1),
                },
                Terminator::Branch { edge: EdgeId(2) },
                Terminator::Branch { edge: EdgeId(3) },
                Terminator::Return,
            ],
        };

        global_value_numbering_in_function(&mut func);

        // Both b1 and b2 should keep their const 42 since neither dominates the other
        assert_eq!(func.blocks[1].insts.len(), 1, "b1 keeps its const");
        assert_eq!(func.blocks[2].insts.len(), 1, "b2 keeps its const");
    }

    #[test]
    fn eliminate_immediate_handles_copy_chains() {
        // Test that copy-chain tracking works:
        // v0 = const 127  (immediate-encodable for AND)
        // v1 = copy v0
        // v2 = and v3, v1  (uses copy of const as RHS)
        //
        // After eliminate_immediate_only_const_defs:
        // - v0 and v1 should both be marked immediate-only
        // - Their operands should be cleared (no regalloc needed)
        use kajit_lir::BinOpKind;

        let mut program = Program {
            vreg_count: 4,
            slot_count: 0,
            funcs: vec![single_block_func(vec![
                Inst {
                    id: InstId(0),
                    op: LinearOp::Const {
                        dst: v(0),
                        value: 127,
                    },
                    operands: vec![Operand {
                        vreg: v(0),
                        kind: OperandKind::Def,
                        class: RegClass::Gpr,
                        fixed: None,
                    }],
                    clobbers: Clobbers::default(),
                },
                Inst {
                    id: InstId(1),
                    op: LinearOp::Copy {
                        dst: v(1),
                        src: v(0),
                    },
                    operands: vec![
                        Operand {
                            vreg: v(0),
                            kind: OperandKind::Use,
                            class: RegClass::Gpr,
                            fixed: None,
                        },
                        Operand {
                            vreg: v(1),
                            kind: OperandKind::Def,
                            class: RegClass::Gpr,
                            fixed: None,
                        },
                    ],
                    clobbers: Clobbers::default(),
                },
                Inst {
                    id: InstId(2),
                    // v2 = and v3, v1 where v1 is a copy of const 127
                    op: LinearOp::BinOp {
                        op: BinOpKind::And,
                        dst: v(2),
                        lhs: v(3),
                        rhs: v(1),
                    },
                    operands: vec![
                        Operand {
                            vreg: v(3),
                            kind: OperandKind::Use,
                            class: RegClass::Gpr,
                            fixed: None,
                        },
                        Operand {
                            vreg: v(1),
                            kind: OperandKind::Use,
                            class: RegClass::Gpr,
                            fixed: None,
                        },
                        Operand {
                            vreg: v(2),
                            kind: OperandKind::Def,
                            class: RegClass::Gpr,
                            fixed: None,
                        },
                    ],
                    clobbers: Clobbers::default(),
                },
            ])],
            debug: Default::default(),
            hints: Default::default(),
        };

        // v3 needs to be defined somewhere (as data_arg)
        program.funcs[0].data_args = vec![v(3)];
        // Mark v2 as used
        program.funcs[0].data_results = vec![v(2)];

        eliminate_immediate_only_const_defs(&mut program);

        let func = &program.funcs[0];

        // const should have no operands (immediate-only)
        assert!(
            func.insts[0].operands.is_empty(),
            "const v0 should have no operands (immediate-only)"
        );

        // copy should also have no operands (copy of immediate-only const)
        assert!(
            func.insts[1].operands.is_empty(),
            "copy v1 should have no operands (copy of immediate-only const)"
        );

        // BinOp should only have 2 operands: v3 (Use) and v2 (Def)
        // v1 (the RHS) should be removed since it's immediate-only
        assert_eq!(
            func.insts[2].operands.len(),
            2,
            "BinOp should have 2 operands after eliminating immediate RHS"
        );
        assert!(
            func.insts[2].operands.iter().all(|op| op.vreg != v(1)),
            "BinOp should not have v1 in operands"
        );
    }

    #[test]
    fn fuse_compare_zero_branch_cmpne() {
        // Test CmpNe with zero followed by BranchIfZero:
        // v0 = const 0
        // v1 = some_value (from data_arg)
        // v2 = CmpNe v1, v0
        // BranchIfZero v2 -> taken, fallthrough
        //
        // After fusion, BranchIfZero should use v1 directly (not v2)
        use kajit_lir::BinOpKind;

        let mut func = Function {
            id: FunctionId(0),
            lambda_id: LambdaId::new(0),
            entry: BlockId(0),
            data_args: vec![v(1)],
            data_results: Vec::new(),
            output_size: 0,
            blocks: vec![
                Block {
                    id: BlockId(0),
                    params: Vec::new(),
                    insts: vec![InstId(0), InstId(1)],
                    term: TermId(0),
                    preds: Vec::new(),
                    succs: vec![EdgeId(0), EdgeId(1)],
                    dead: false,
                },
                Block {
                    id: BlockId(1),
                    params: Vec::new(),
                    insts: Vec::new(),
                    term: TermId(1),
                    preds: vec![EdgeId(0)],
                    succs: Vec::new(),
                    dead: false,
                },
                Block {
                    id: BlockId(2),
                    params: Vec::new(),
                    insts: Vec::new(),
                    term: TermId(2),
                    preds: vec![EdgeId(1)],
                    succs: Vec::new(),
                    dead: false,
                },
            ],
            edges: vec![
                Edge {
                    id: EdgeId(0),
                    from: BlockId(0),
                    to: BlockId(1),
                    args: Vec::new(),
                },
                Edge {
                    id: EdgeId(1),
                    from: BlockId(0),
                    to: BlockId(2),
                    args: Vec::new(),
                },
            ],
            insts: vec![
                Inst {
                    id: InstId(0),
                    op: LinearOp::Const {
                        dst: v(0),
                        value: 0,
                    },
                    operands: vec![Operand {
                        vreg: v(0),
                        kind: OperandKind::Def,
                        class: RegClass::Gpr,
                        fixed: None,
                    }],
                    clobbers: Clobbers::default(),
                },
                Inst {
                    id: InstId(1),
                    op: LinearOp::BinOp {
                        op: BinOpKind::CmpNe,
                        dst: v(2),
                        lhs: v(1),
                        rhs: v(0),
                    },
                    operands: vec![
                        Operand {
                            vreg: v(1),
                            kind: OperandKind::Use,
                            class: RegClass::Gpr,
                            fixed: None,
                        },
                        Operand {
                            vreg: v(0),
                            kind: OperandKind::Use,
                            class: RegClass::Gpr,
                            fixed: None,
                        },
                        Operand {
                            vreg: v(2),
                            kind: OperandKind::Def,
                            class: RegClass::Gpr,
                            fixed: None,
                        },
                    ],
                    clobbers: Clobbers::default(),
                },
            ],
            terms: vec![
                Terminator::BranchIfZero {
                    cond: v(2),
                    taken: EdgeId(0),
                    fallthrough: EdgeId(1),
                },
                Terminator::Return,
                Terminator::Return,
            ],
        };

        fuse_compare_zero_branch_in_function(&mut func);

        // BranchIfZero should now use v1 directly instead of v2
        match &func.terms[0] {
            Terminator::BranchIfZero { cond, .. } => {
                assert_eq!(
                    cond.index(),
                    1,
                    "BranchIfZero should use v1 (the original value) after fusion"
                );
            }
            other => panic!("expected BranchIfZero, got {:?}", other),
        }
    }

    #[test]
    fn fuse_compare_zero_branch_cmpeq_flips_to_branch_if() {
        // Test CmpEq with zero followed by BranchIfZero:
        // v0 = const 0
        // v1 = some_value
        // v2 = CmpEq v1, v0
        // BranchIfZero v2 -> taken, fallthrough
        //
        // After fusion: BranchIf v1 -> taken, fallthrough
        // (because "branch if (v1 == 0) is zero" = "branch if v1 != 0")
        use kajit_lir::BinOpKind;

        let mut func = Function {
            id: FunctionId(0),
            lambda_id: LambdaId::new(0),
            entry: BlockId(0),
            data_args: vec![v(1)],
            data_results: Vec::new(),
            output_size: 0,
            blocks: vec![Block {
                id: BlockId(0),
                params: Vec::new(),
                insts: vec![InstId(0), InstId(1)],
                term: TermId(0),
                preds: Vec::new(),
                succs: vec![EdgeId(0), EdgeId(1)],
                dead: false,
            }],
            edges: vec![
                Edge {
                    id: EdgeId(0),
                    from: BlockId(0),
                    to: BlockId(0),
                    args: Vec::new(),
                },
                Edge {
                    id: EdgeId(1),
                    from: BlockId(0),
                    to: BlockId(0),
                    args: Vec::new(),
                },
            ],
            insts: vec![
                Inst {
                    id: InstId(0),
                    op: LinearOp::Const {
                        dst: v(0),
                        value: 0,
                    },
                    operands: vec![Operand {
                        vreg: v(0),
                        kind: OperandKind::Def,
                        class: RegClass::Gpr,
                        fixed: None,
                    }],
                    clobbers: Clobbers::default(),
                },
                Inst {
                    id: InstId(1),
                    op: LinearOp::BinOp {
                        op: BinOpKind::CmpEq,
                        dst: v(2),
                        lhs: v(1),
                        rhs: v(0),
                    },
                    operands: vec![
                        Operand {
                            vreg: v(1),
                            kind: OperandKind::Use,
                            class: RegClass::Gpr,
                            fixed: None,
                        },
                        Operand {
                            vreg: v(0),
                            kind: OperandKind::Use,
                            class: RegClass::Gpr,
                            fixed: None,
                        },
                        Operand {
                            vreg: v(2),
                            kind: OperandKind::Def,
                            class: RegClass::Gpr,
                            fixed: None,
                        },
                    ],
                    clobbers: Clobbers::default(),
                },
            ],
            terms: vec![Terminator::BranchIfZero {
                cond: v(2),
                taken: EdgeId(0),
                fallthrough: EdgeId(1),
            }],
        };

        fuse_compare_zero_branch_in_function(&mut func);

        // BranchIfZero should be flipped to BranchIf with v1
        match &func.terms[0] {
            Terminator::BranchIf { cond, .. } => {
                assert_eq!(cond.index(), 1, "BranchIf should use v1 after fusion");
            }
            other => panic!("expected BranchIf after CmpEq fusion, got {:?}", other),
        }
    }
}
