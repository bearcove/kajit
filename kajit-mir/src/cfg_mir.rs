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
    pub debug: ProgramDebugProvenance,
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

            // Also, vregs defined in dominating blocks are available
            // For simplicity, we assume entry block defs are available everywhere
            // (a more precise analysis would compute dominators)
            if block.id != self.entry {
                let entry_block = &self.blocks[self.entry.index()];
                for &param in &entry_block.params {
                    defs_available.insert(param);
                }
                for &inst_id in &entry_block.insts {
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

    for inst in &func.insts {
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

fn fmt_cfg_operand(f: &mut fmt::Formatter<'_>, operand: &Operand) -> fmt::Result {
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

fn fmt_cfg_inst(
    f: &mut fmt::Formatter<'_>,
    inst: &Inst,
    registry: Option<&IntrinsicRegistry>,
) -> fmt::Result {
    let defs: Vec<_> = inst
        .operands
        .iter()
        .filter(|op| op.kind == OperandKind::Def)
        .collect();
    let uses: Vec<_> = inst
        .operands
        .iter()
        .filter(|op| op.kind == OperandKind::Use)
        .collect();

    if !defs.is_empty() {
        for (idx, op) in defs.iter().enumerate() {
            if idx > 0 {
                write!(f, ", ")?;
            }
            fmt_cfg_operand(f, op)?;
        }
        write!(f, " = ")?;
    }

    fmt_cfg_op_name(f, &inst.op, registry)?;

    if !uses.is_empty() {
        write!(f, " ")?;
        for (idx, op) in uses.iter().enumerate() {
            if idx > 0 {
                write!(f, ", ")?;
            }
            fmt_cfg_operand(f, op)?;
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
    Branch(LabelId),
    BranchIf {
        cond: VReg,
        target: LabelId,
    },
    BranchIfZero {
        cond: VReg,
        target: LabelId,
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
    },
    BranchIf {
        cond: VReg,
        target: BlockId,
        fallthrough: BlockId,
    },
    BranchIfZero {
        cond: VReg,
        target: BlockId,
        fallthrough: BlockId,
    },
    JumpTable {
        predicate: VReg,
        targets: Vec<BlockId>,
        default: BlockId,
    },
}

impl TempTermBlock {
    fn uses(&self) -> Vec<VReg> {
        match self {
            Self::BranchIf { cond, .. } | Self::BranchIfZero { cond, .. } => vec![*cond],
            Self::JumpTable { predicate, .. } => vec![*predicate],
            Self::Return | Self::ErrorExit(_) | Self::Branch { .. } => Vec::new(),
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
        LinearOp::Branch(_)
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
        | LinearOp::Branch(_)
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
        TempTermLabel::Branch(label) => TempTermBlock::Branch {
            target: *labels
                .get(label)
                .unwrap_or_else(|| panic!("unknown label target: {label:?}")),
        },
        TempTermLabel::BranchIf { cond, target } => TempTermBlock::BranchIf {
            cond: *cond,
            target: *labels
                .get(target)
                .unwrap_or_else(|| panic!("unknown label target: {target:?}")),
            fallthrough: next.expect("BranchIf must have fallthrough block"),
        },
        TempTermLabel::BranchIfZero { cond, target } => TempTermBlock::BranchIfZero {
            cond: *cond,
            target: *labels
                .get(target)
                .unwrap_or_else(|| panic!("unknown label target: {target:?}")),
            fallthrough: next.expect("BranchIfZero must have fallthrough block"),
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
        },
    }
}

fn collect_use_def(
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
    for vreg in term.uses() {
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
    ops: &[LinearOp],
    op_scopes: &[Option<DebugScopeId>],
    op_values: &[Option<DebugValueId>],
    vreg_count: u32,
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
                blocks: vec![Block {
                    id: BlockId(0),
                    params: Vec::new(),
                    insts: Vec::new(),
                    term: TermId(0),
                    preds: Vec::new(),
                    succs: Vec::new(),
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
                LinearOp::Branch(target) => {
                    if let Some(scope) = op_scope {
                        lowered_scopes.insert(OpId::Term(TermId(bi as u32)), scope);
                    }
                    if let Some(debug_value) = op_value {
                        lowered_values.insert(OpId::Term(TermId(bi as u32)), debug_value);
                    }
                    term = Some(TempTermLabel::Branch(target));
                    cursor += 1;
                    break;
                }
                LinearOp::BranchIf { cond, target } => {
                    if let Some(scope) = op_scope {
                        lowered_scopes.insert(OpId::Term(TermId(bi as u32)), scope);
                    }
                    if let Some(debug_value) = op_value {
                        lowered_values.insert(OpId::Term(TermId(bi as u32)), debug_value);
                    }
                    term = Some(TempTermLabel::BranchIf { cond, target });
                    cursor += 1;
                    break;
                }
                LinearOp::BranchIfZero { cond, target } => {
                    if let Some(scope) = op_scope {
                        lowered_scopes.insert(OpId::Term(TermId(bi as u32)), scope);
                    }
                    if let Some(debug_value) = op_value {
                        lowered_values.insert(OpId::Term(TermId(bi as u32)), debug_value);
                    }
                    term = Some(TempTermLabel::BranchIfZero { cond, target });
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

    let mut use_sets = vec![vec![false; vreg_count as usize]; blocks.len()];
    let mut def_sets = vec![vec![false; vreg_count as usize]; blocks.len()];
    for (idx, block) in blocks.iter().enumerate() {
        collect_use_def(
            block,
            &insts,
            &block_terms[idx],
            &mut use_sets[idx],
            &mut def_sets[idx],
        );
    }

    if let Some(entry_defs) = def_sets.first_mut() {
        for &arg in &data_args {
            entry_defs[arg.index()] = true;
        }
    }

    let mut live_in = vec![vec![false; vreg_count as usize]; blocks.len()];
    let mut live_out = vec![vec![false; vreg_count as usize]; blocks.len()];
    loop {
        let mut changed = false;
        for bi in (0..blocks.len()).rev() {
            let mut out = vec![false; vreg_count as usize];
            for succ in block_terms[bi].successors() {
                for (idx, live) in live_in[succ.index()].iter().enumerate() {
                    out[idx] |= *live;
                }
            }

            let mut inn = use_sets[bi].clone();
            for idx in 0..inn.len() {
                inn[idx] |= out[idx] && !def_sets[bi][idx];
            }

            if out != live_out[bi] || inn != live_in[bi] {
                live_out[bi] = out;
                live_in[bi] = inn;
                changed = true;
            }
        }
        if !changed {
            break;
        }
    }

    for bi in 0..blocks.len() {
        if bi == 0 {
            blocks[bi].params.clear();
            continue;
        }
        blocks[bi].params = live_in[bi]
            .iter()
            .enumerate()
            .filter_map(|(idx, live)| live.then_some(VReg::new(idx as u32)))
            .collect();
    }

    let mut edges = Vec::<Edge>::new();
    for from in 0..blocks.len() {
        for to in block_terms[from].successors() {
            let edge_id = EdgeId(edges.len() as u32);
            let args = blocks[to.index()]
                .params
                .iter()
                .copied()
                .map(|target| EdgeArg {
                    target,
                    source: target,
                })
                .collect();
            edges.push(Edge {
                id: edge_id,
                from: BlockId(from as u32),
                to,
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
pub fn lower_linear_ir(ir: &LinearIr) -> Program {
    let mut funcs = Vec::<Function>::new();
    let mut op_scopes = HashMap::<(LambdaId, OpId), DebugScopeId>::new();
    let mut op_values = HashMap::<(LambdaId, OpId), DebugValueId>::new();
    let mut cursor = 0usize;
    while cursor < ir.ops.len() {
        let (lambda_id, data_args, data_results) = match &ir.ops[cursor] {
            LinearOp::FuncStart {
                lambda_id,
                data_args,
                data_results,
                ..
            } => (*lambda_id, data_args.clone(), data_results.clone()),
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
pub fn lower_and_optimize(ir: &LinearIr) -> Program {
    let debug = std::env::var("KAJIT_DEBUG_REMAT").is_ok();
    if debug {
        eprintln!("[lower_and_optimize] called with {} ops", ir.ops.len());
    }
    let mut cfg = lower_linear_ir(ir);
    rematerialize_constants(&mut cfg);
    dead_code_elimination(&mut cfg);
    cfg
}

/// that needs them, eliminating the need to pass them through edges.
///
/// Benefits:
/// - Reduces register pressure (constants don't need to be kept live)
/// - Enables immediate encoding in backends (AND reg, #imm instead of AND reg, reg)
/// - Removes unnecessary edge argument traffic
pub fn rematerialize_constants(program: &mut Program) {
    // Build a map of VReg -> constant value for all constants in the program
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

            if is_uniform_const {
                if let Some(const_val) = all_same_const {
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
    // Iterate until no more changes (some dead code may only become visible
    // after other dead code is removed)
    let mut iteration = 0;
    loop {
        iteration += 1;
        // Collect which instructions are actually live (referenced by blocks)
        let live_insts: HashSet<InstId> = func
            .blocks
            .iter()
            .flat_map(|b| b.insts.iter().copied())
            .collect();
        if debug && iteration == 1 {
            eprintln!(
                "[dce] func f{} has {} live insts",
                func.id.0,
                live_insts.len()
            );
        }

        // Collect all used vregs
        let mut used_vregs: HashSet<VReg> = HashSet::new();

        // Uses from instruction operands (inputs) - only from live instructions
        // Track which vregs have been defined so far in each block (order-aware)
        // We need to track two things:
        // 1. external_uses: vregs used that come from outside this block (for cross-block DCE)
        // 2. all uses: any vreg used (to keep local definitions alive)
        let mut external_uses: HashSet<VReg> = HashSet::new();

        for block in &func.blocks {
            // Block params are defined at entry
            let mut defs_so_far: HashSet<VReg> = block.params.iter().copied().collect();

            for &inst_id in &block.insts {
                if !live_insts.contains(&inst_id) {
                    continue;
                }
                let inst = &func.insts[inst_id.index()];

                // First, check uses - they see defs from BEFORE this instruction
                for operand in &inst.operands {
                    if operand.kind == OperandKind::Use {
                        // ALL uses keep definitions alive
                        used_vregs.insert(operand.vreg);

                        // But only external uses matter for cross-block analysis
                        if !defs_so_far.contains(&operand.vreg) {
                            if debug && operand.vreg.index() == 37 {
                                eprintln!("[dce] v37 used by inst i{} (external)", inst_id.0);
                            }
                            external_uses.insert(operand.vreg);
                        }
                    }
                }

                // Then, add this instruction's defs
                for operand in &inst.operands {
                    if operand.kind == OperandKind::Def {
                        defs_so_far.insert(operand.vreg);
                    }
                }
            }
        }

        // Collect which instructions are in the entry block
        let entry_block_insts: HashSet<InstId> = func
            .blocks
            .get(func.entry.index())
            .map(|b| b.insts.iter().copied().collect())
            .unwrap_or_default();

        // Uses from terminators
        for term in &func.terms {
            match term {
                Terminator::BranchIf { cond, .. } | Terminator::BranchIfZero { cond, .. } => {
                    used_vregs.insert(*cond);
                }
                Terminator::JumpTable { predicate, .. } => {
                    used_vregs.insert(*predicate);
                }
                Terminator::Return | Terminator::ErrorExit { .. } | Terminator::Branch { .. } => {}
            }
        }

        // Uses from edge arguments (source vregs that flow into target blocks)
        for edge in &func.edges {
            for arg in &edge.args {
                used_vregs.insert(arg.source);
            }
        }

        // Uses from data_results (function return values)
        for &vreg in &func.data_results {
            used_vregs.insert(vreg);
        }

        // Find instructions to remove: those that define vregs that are never used
        // and have no side effects (only consider live instructions)
        let mut insts_to_remove: HashSet<InstId> = HashSet::new();
        for inst in &func.insts {
            if !live_insts.contains(&inst.id) {
                continue;
            }
            if is_pure_instruction(&inst.op) {
                // Check if all defs are unused
                let all_defs_unused = inst.operands.iter().all(|op| {
                    if op.kind != OperandKind::Def {
                        return true;
                    }
                    // A def is unused if:
                    // 1. It's not in used_vregs at all, OR
                    // 2. It's in entry block AND not in external_uses (meaning all uses
                    //    are satisfied by local redefinitions in other blocks)
                    if !used_vregs.contains(&op.vreg) {
                        return true;
                    }
                    // TODO: For entry block defs, we could check if all uses are satisfied
                    // by local redefinitions in other blocks. For now, keep it simple.
                    false
                });
                if all_defs_unused {
                    insts_to_remove.insert(inst.id);
                } else if debug
                    && matches!(&inst.op, LinearOp::Const { dst, .. } if dst.index() == 37)
                {
                    // Debug: show why v37 const is considered used
                    let is_external = external_uses.contains(&VReg::new(37));
                    eprintln!(
                        "[dce] const i{} v37 is used (external={})",
                        inst.id.0, is_external
                    );
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
            // Show first few removed consts
            let mut shown = 0;
            for &inst_id in &insts_to_remove {
                let inst = &func.insts[inst_id.index()];
                if let LinearOp::Const { dst, .. } = &inst.op {
                    if shown < 5 {
                        eprintln!("[dce]   removing const i{} v{}", inst_id.0, dst.index());
                        shown += 1;
                    }
                }
            }
        }

        // Remove dead instructions from blocks
        for block in &mut func.blocks {
            block.insts.retain(|id| !insts_to_remove.contains(id));
        }

        // Note: We leave func.insts intact to keep InstIds stable.
        // Orphaned instructions in the arena are harmless.
    }
}

/// Check if a vreg is used by any instruction after `def_inst` in the entry block,
/// OR by any edge arg flowing out of the entry block.
fn is_used_later_in_entry_block(func: &Function, def_inst: InstId, vreg: VReg) -> bool {
    let entry_block = &func.blocks[func.entry.index()];
    let mut seen_def = false;
    for &inst_id in &entry_block.insts {
        if inst_id == def_inst {
            seen_def = true;
            continue;
        }
        if seen_def {
            let inst = &func.insts[inst_id.index()];
            for op in &inst.operands {
                if op.kind == OperandKind::Use && op.vreg == vreg {
                    return true;
                }
            }
        }
    }
    // Also check edge args from successor edges
    for &edge_id in &entry_block.succs {
        let edge = &func.edges[edge_id.index()];
        for arg in &edge.args {
            if arg.source == vreg {
                return true;
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

    #[test]
    fn lower_linear_ir_produces_valid_cfg_program() {
        let mut builder = IrBuilder::new(<u32 as facet::Facet>::SHAPE);
        {
            let mut rb = builder.root_region();
            rb.bounds_check(4);
            let value = rb.read_bytes(4);
            rb.write_to_field(value, 0, Width::W4);
            rb.set_results(&[]);
        }
        let mut func = builder.finish();
        let linear = linearize(&mut func);
        let program = lower_linear_ir(&linear);
        program
            .validate()
            .expect("lowered cfg program should validate");
        assert_eq!(program.funcs.len(), 1);
        assert!(!program.funcs[0].blocks.is_empty());
    }

    #[test]
    fn lower_linear_ir_models_gamma_join_block_params() {
        let mut builder = IrBuilder::new(<u32 as facet::Facet>::SHAPE);
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
        let program = lower_linear_ir(&linear);
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
        let mut builder = IrBuilder::new(<u32 as facet::Facet>::SHAPE);
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
        let program = lower_linear_ir(&linear);
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
}
