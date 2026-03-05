//! RA-MIR lowering from linear IR.
//!
//! This module builds a CFG-oriented machine IR used as allocator input.

use kajit_ir::ErrorCode;
use kajit_ir::{LambdaId, VReg};
use kajit_lir::{LabelId, LinearIr, LinearOp};
use std::collections::HashMap;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct BlockId(pub u32);

impl BlockId {
    fn index(self) -> usize {
        self.0 as usize
    }
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
    /// Pin operand to a specific hardware register encoding (e.g., rcx=1 for x64 shifts).
    HwReg(u8),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RaOperand {
    pub vreg: VReg,
    pub kind: OperandKind,
    pub class: RegClass,
    pub fixed: Option<FixedReg>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct RaClobbers {
    pub caller_saved_gpr: bool,
    pub caller_saved_simd: bool,
}

#[derive(Debug, Clone)]
pub struct RaInst {
    pub linear_op_index: usize,
    pub op: LinearOp,
    pub operands: Vec<RaOperand>,
    pub clobbers: RaClobbers,
}

#[derive(Debug, Clone)]
pub struct RaEdge {
    pub to: BlockId,
    pub args: Vec<RaEdgeArg>,
}

#[derive(Debug, Clone, Copy)]
pub struct RaEdgeArg {
    pub target: VReg,
    pub source: VReg,
}

#[derive(Debug, Clone)]
pub enum RaTerminator {
    Return,
    ErrorExit {
        code: ErrorCode,
    },
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

impl RaTerminator {
    fn uses(&self) -> Vec<VReg> {
        match self {
            Self::BranchIf { cond, .. } | Self::BranchIfZero { cond, .. } => vec![*cond],
            Self::JumpTable { predicate, .. } => vec![*predicate],
            Self::Return | Self::ErrorExit { .. } | Self::Branch { .. } => Vec::new(),
        }
    }

    fn successors(&self) -> Vec<BlockId> {
        match self {
            Self::Return | Self::ErrorExit { .. } => Vec::new(),
            Self::Branch { target } => vec![*target],
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

#[derive(Debug, Clone)]
pub struct RaBlock {
    pub id: BlockId,
    pub label: Option<LabelId>,
    pub params: Vec<VReg>,
    pub insts: Vec<RaInst>,
    pub term_linear_op_index: usize,
    pub term: RaTerminator,
    pub preds: Vec<BlockId>,
    pub succs: Vec<RaEdge>,
}

#[derive(Debug, Clone)]
pub struct RaFunction {
    pub lambda_id: LambdaId,
    pub entry: BlockId,
    pub data_args: Vec<VReg>,
    pub data_results: Vec<VReg>,
    pub blocks: Vec<RaBlock>,
}

#[derive(Debug, Clone)]
pub struct RaProgram {
    pub funcs: Vec<RaFunction>,
    pub vreg_count: u32,
    pub slot_count: u32,
}

// ─── Display ────────────────────────────────────────────────────────────────

#[derive(Clone, Copy, PartialEq, Eq)]
enum DisplayStyle {
    Canonical,
    Human,
}

pub struct HumanRaProgram<'a> {
    program: &'a RaProgram,
}

impl<'a> std::fmt::Display for HumanRaProgram<'a> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        fmt_program(self.program, f, DisplayStyle::Human)
    }
}

impl RaProgram {
    pub fn human(&self) -> HumanRaProgram<'_> {
        HumanRaProgram { program: self }
    }
}

impl std::fmt::Display for RaProgram {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let style = if f.alternate() {
            DisplayStyle::Human
        } else {
            DisplayStyle::Canonical
        };
        fmt_program(self, f, style)
    }
}

impl std::fmt::Display for RaFunction {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let style = if f.alternate() {
            DisplayStyle::Human
        } else {
            DisplayStyle::Canonical
        };
        fmt_function(self, f, style)
    }
}

fn fmt_program(
    program: &RaProgram,
    f: &mut std::fmt::Formatter<'_>,
    style: DisplayStyle,
) -> std::fmt::Result {
    for func in &program.funcs {
        fmt_function(func, f, style)?;
    }
    Ok(())
}

fn fmt_function(
    func: &RaFunction,
    f: &mut std::fmt::Formatter<'_>,
    style: DisplayStyle,
) -> std::fmt::Result {
    match style {
        DisplayStyle::Canonical => fmt_function_canonical(func, f),
        DisplayStyle::Human => fmt_function_human(func, f),
    }
}

fn fmt_function_canonical(func: &RaFunction, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    writeln!(
        f,
        "ra_func @{} {{ ; entry: b{}",
        func.lambda_id.index(),
        func.entry.0,
    )?;
    for block in &func.blocks {
        fmt_block_canonical(f, block)?;
    }
    writeln!(f, "}}")
}

fn fmt_function_human(func: &RaFunction, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    let meta = build_display_meta(func);
    write!(
        f,
        "ra_func @{} {{ ; entry: b{}",
        func.lambda_id.index(),
        func.entry.0,
    )?;
    if !meta.const_alias_list.is_empty() {
        write!(
            f,
            " ; consts: {}",
            fmt_const_alias_table(&meta.const_alias_list)
        )?;
    }
    writeln!(f)?;
    for block in &func.blocks {
        fmt_block_human(f, block, &meta)?;
        writeln!(f)?;
    }
    writeln!(f, "}}")
}

fn fmt_block_canonical(f: &mut std::fmt::Formatter<'_>, block: &RaBlock) -> std::fmt::Result {
    write!(f, "  block b{}", block.id.0)?;
    if !block.params.is_empty() {
        write!(f, " [params:")?;
        for (i, p) in block.params.iter().enumerate() {
            if i > 0 {
                write!(f, ",")?;
            }
            write!(f, " v{}", p.index())?;
        }
        write!(f, "]")?;
    }
    if !block.preds.is_empty() {
        write!(f, " (preds:")?;
        for (i, p) in block.preds.iter().enumerate() {
            if i > 0 {
                write!(f, ",")?;
            }
            write!(f, " b{}", p.0)?;
        }
        write!(f, ")")?;
    }
    writeln!(f, ": ; insts: {}", block.insts.len())?;

    for inst in &block.insts {
        write!(f, "    ")?;
        fmt_ra_inst(f, inst)?;
        writeln!(f)?;
    }

    write!(f, "    term: ")?;
    fmt_terminator(f, &block.term)?;
    writeln!(f, " ; uses: {}", fmt_vregs(&block.term.uses()))?;

    if block.succs.is_empty() {
        writeln!(f, "    succs: (none)")?;
    } else {
        write!(f, "    succs:")?;
        for edge in &block.succs {
            write!(f, " b{}", edge.to.0)?;
            if !edge.args.is_empty() {
                write!(f, " [")?;
                for (i, arg) in edge.args.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    if arg.target == arg.source {
                        write!(f, "v{}", arg.source.index())?;
                    } else {
                        write!(f, "v{}=>v{}", arg.target.index(), arg.source.index())?;
                    }
                }
                write!(f, "]")?;
            }
        }
        writeln!(f, " ; count: {}", block.succs.len())?;
    }

    Ok(())
}

fn fmt_block_human(
    f: &mut std::fmt::Formatter<'_>,
    block: &RaBlock,
    meta: &DisplayMeta,
) -> std::fmt::Result {
    // Block header: block b0 [params: v0, v1] (preds: b2, b3):
    write!(f, "  block b{}", block.id.0)?;
    if !block.params.is_empty() {
        write!(f, " [params:")?;
        for (i, p) in block.params.iter().enumerate() {
            if i > 0 {
                write!(f, ",")?;
            }
            write!(f, " v{}", p.index())?;
        }
        write!(f, "]")?;
    }
    if !block.preds.is_empty() {
        write!(f, " (preds:")?;
        for (i, p) in block.preds.iter().enumerate() {
            if i > 0 {
                write!(f, ",")?;
            }
            write!(f, " b{}", p.0)?;
        }
        write!(f, ")")?;
    }
    write!(f, ": ; insts: {}", block.insts.len())?;
    if !block.params.is_empty() {
        write!(f, " ; params_dbg: {}", fmt_param_debug_names(&block.params))?;
    }
    writeln!(f)?;

    // Instructions.
    let block_lir_base = *meta
        .block_lir_base
        .get(&block.id)
        .expect("every block should have a display lir base index");
    for (inst_idx, inst) in block.insts.iter().enumerate() {
        let defs: Vec<VReg> = inst
            .operands
            .iter()
            .filter(|o| o.kind == OperandKind::Def)
            .map(|o| o.vreg)
            .collect();

        write!(f, "    ")?;
        fmt_ra_inst(f, inst)?;
        write!(f, " ; orig: lir#{}", block_lir_base + inst_idx)?;
        if let Some(hint) = ra_inst_semantic_hint(&inst.op) {
            write!(f, " ; sem: {hint}")?;
        }
        if let Some(alias) = const_alias_for_inst(&inst.op, meta) {
            write!(f, " ; const_alias: {alias}")?;
        }
        if !defs.is_empty() {
            write!(f, " ; defs_dbg: {}", fmt_temp_debug_names(&defs))?;
        }
        writeln!(f)?;
    }

    // Terminator.
    write!(f, "    term: ")?;
    fmt_terminator(f, &block.term)?;
    write!(f, " ; uses: {}", fmt_vregs(&block.term.uses()))?;
    write!(f, " ; orig: lir#{}", block_lir_base + block.insts.len())?;
    if let Some(hint) = ra_term_semantic_hint(&block.term) {
        write!(f, " ; sem: {hint}")?;
    }
    writeln!(f)?;

    // Successors with edge args.
    if block.succs.is_empty() {
        writeln!(f, "    succs: (none)")?;
    } else {
        write!(f, "    succs:")?;
        for edge in &block.succs {
            write!(f, " b{}", edge.to.0)?;
            if !edge.args.is_empty() {
                write!(f, " [")?;
                for (i, arg) in edge.args.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    if arg.target == arg.source {
                        write!(f, "v{}", arg.source.index())?;
                    } else {
                        write!(f, "v{}=>v{}", arg.target.index(), arg.source.index())?;
                    }
                }
                write!(f, "]")?;
            }
        }
        writeln!(f, " ; count: {}", block.succs.len())?;
    }

    Ok(())
}

fn fmt_vregs(vregs: &[VReg]) -> String {
    let mut out = String::from("[");
    for (i, v) in vregs.iter().enumerate() {
        if i > 0 {
            out.push_str(", ");
        }
        out.push('v');
        out.push_str(&v.index().to_string());
    }
    out.push(']');
    out
}

fn fmt_temp_debug_names(vregs: &[VReg]) -> String {
    let mut out = String::new();
    for (i, v) in vregs.iter().enumerate() {
        if i > 0 {
            out.push_str(", ");
        }
        out.push_str(&format!("t{}=v{}", v.index(), v.index()));
    }
    out
}

fn fmt_param_debug_names(params: &[VReg]) -> String {
    let mut out = String::new();
    for (i, p) in params.iter().enumerate() {
        if i > 0 {
            out.push_str(", ");
        }
        out.push_str(&format!("p{i}=v{}", p.index()));
    }
    out
}

struct DisplayMeta {
    const_aliases: HashMap<u64, String>,
    const_alias_list: Vec<(String, u64)>,
    block_lir_base: HashMap<BlockId, usize>,
}

fn build_display_meta(func: &RaFunction) -> DisplayMeta {
    let mut const_aliases = HashMap::new();
    let mut const_alias_list = Vec::new();
    let mut block_lir_base = HashMap::new();
    let mut next_lir_index = 0usize;

    for block in &func.blocks {
        block_lir_base.insert(block.id, next_lir_index);
        next_lir_index += block.insts.len() + 1; // +1 for terminator

        for inst in &block.insts {
            if let LinearOp::Const { value, .. } = inst.op
                && !const_aliases.contains_key(&value)
            {
                let alias = format!("k{}", const_alias_list.len());
                const_aliases.insert(value, alias.clone());
                const_alias_list.push((alias, value));
            }
        }
    }

    DisplayMeta {
        const_aliases,
        const_alias_list,
        block_lir_base,
    }
}

fn const_alias_for_inst<'a>(op: &LinearOp, meta: &'a DisplayMeta) -> Option<&'a str> {
    let value = match op {
        LinearOp::Const { value, .. } => *value,
        _ => return None,
    };
    meta.const_aliases.get(&value).map(String::as_str)
}

fn fmt_const_alias_table(aliases: &[(String, u64)]) -> String {
    let mut out = String::new();
    for (i, (alias, value)) in aliases.iter().enumerate() {
        if i > 0 {
            out.push_str(", ");
        }
        out.push_str(&format!("{alias}={value:#x}"));
    }
    out
}

fn fmt_ra_inst(f: &mut std::fmt::Formatter<'_>, inst: &RaInst) -> std::fmt::Result {
    // Show operands in a compact format.
    // Defs first, then the op name, then uses.
    let defs: Vec<_> = inst
        .operands
        .iter()
        .filter(|o| o.kind == OperandKind::Def)
        .collect();
    let uses: Vec<_> = inst
        .operands
        .iter()
        .filter(|o| o.kind == OperandKind::Use)
        .collect();

    if !defs.is_empty() {
        for (i, d) in defs.iter().enumerate() {
            if i > 0 {
                write!(f, ", ")?;
            }
            fmt_operand(f, d)?;
        }
        write!(f, " = ")?;
    }

    // Op name (simplified from LinearOp).
    fmt_ra_op_name(f, &inst.op)?;

    if !uses.is_empty() {
        write!(f, " ")?;
        for (i, u) in uses.iter().enumerate() {
            if i > 0 {
                write!(f, ", ")?;
            }
            fmt_operand(f, u)?;
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

fn fmt_operand(f: &mut std::fmt::Formatter<'_>, op: &RaOperand) -> std::fmt::Result {
    write!(f, "v{}", op.vreg.index())?;
    write!(
        f,
        ":{}",
        match op.class {
            RegClass::Gpr => "gpr",
            RegClass::Simd => "simd",
        }
    )?;
    if let Some(fixed) = op.fixed {
        match fixed {
            FixedReg::AbiArg(i) => write!(f, "/arg{i}")?,
            FixedReg::AbiRet(i) => write!(f, "/ret{i}")?,
            FixedReg::HwReg(enc) => write!(f, "/hw{enc}")?,
        }
    }
    Ok(())
}

fn fmt_ra_op_name(f: &mut std::fmt::Formatter<'_>, op: &LinearOp) -> std::fmt::Result {
    match op {
        LinearOp::Const { value, .. } => write!(f, "const({value:#x})"),
        LinearOp::BinOp { op, .. } => write!(f, "{op:?}"),
        LinearOp::UnaryOp { op, .. } => write!(f, "{op:?}"),
        LinearOp::Copy { .. } => write!(f, "copy"),
        LinearOp::BoundsCheck { count } => write!(f, "bounds_check({count})"),
        LinearOp::ReadBytes { count, .. } => write!(f, "read_bytes({count})"),
        LinearOp::PeekByte { .. } => write!(f, "peek_byte"),
        LinearOp::AdvanceCursor { count } => write!(f, "advance({count})"),
        LinearOp::AdvanceCursorBy { .. } => write!(f, "advance_by"),
        LinearOp::SaveCursor { .. } => write!(f, "save_cursor"),
        LinearOp::RestoreCursor { .. } => write!(f, "restore_cursor"),
        LinearOp::WriteToField { offset, width, .. } => {
            write!(f, "store([{offset}:{width}])")
        }
        LinearOp::ReadFromField { offset, width, .. } => {
            write!(f, "load([{offset}:{width}])")
        }
        LinearOp::SaveOutPtr { .. } => write!(f, "save_out_ptr"),
        LinearOp::SetOutPtr { .. } => write!(f, "set_out_ptr"),
        LinearOp::SlotAddr { slot, .. } => write!(f, "slot_addr({})", slot.index()),
        LinearOp::WriteToSlot { slot, .. } => write!(f, "write_slot({})", slot.index()),
        LinearOp::ReadFromSlot { slot, .. } => write!(f, "read_slot({})", slot.index()),
        LinearOp::CallIntrinsic {
            func, field_offset, ..
        } => write!(f, "call_intrinsic({func}, fo={field_offset})"),
        LinearOp::CallPure { func, .. } => write!(f, "call_pure({func})"),
        LinearOp::CallLambda { target, .. } => write!(f, "call_lambda(@{})", target.index()),
        LinearOp::SimdStringScan { .. } => write!(f, "simd_string_scan"),
        LinearOp::SimdWhitespaceSkip => write!(f, "simd_ws_skip"),
        LinearOp::ErrorExit { code } => write!(f, "error_exit({code:?})"),
        _ => write!(f, "<?op>"),
    }
}

fn ra_inst_semantic_hint(op: &LinearOp) -> Option<String> {
    match op {
        LinearOp::Const { value, .. } => Some(format!("materialize constant {value:#x}")),
        LinearOp::BinOp { op, .. } => Some(match op {
            kajit_lir::BinOpKind::Add => "integer addition".to_string(),
            kajit_lir::BinOpKind::Sub => "integer subtraction".to_string(),
            kajit_lir::BinOpKind::And => "bitwise and".to_string(),
            kajit_lir::BinOpKind::Or => "bitwise or".to_string(),
            kajit_lir::BinOpKind::Shr => "logical shift right".to_string(),
            kajit_lir::BinOpKind::Shl => "logical shift left".to_string(),
            kajit_lir::BinOpKind::Xor => "bitwise xor".to_string(),
            kajit_lir::BinOpKind::CmpNe => "compare not-equal (produces 0/1)".to_string(),
        }),
        LinearOp::UnaryOp { .. } => Some("unary transform".to_string()),
        LinearOp::Copy { .. } => Some("ssa copy/move".to_string()),
        LinearOp::BoundsCheck { count } => Some(format!("ensure {count} input byte(s) available")),
        LinearOp::ReadBytes { count, .. } => Some(format!("consume {count} input byte(s)")),
        LinearOp::PeekByte { .. } => Some("peek next input byte without advancing".to_string()),
        LinearOp::AdvanceCursor { count } => Some(format!("advance input cursor by {count}")),
        LinearOp::AdvanceCursorBy { .. } => {
            Some("advance input cursor by runtime amount".to_string())
        }
        LinearOp::SaveCursor { .. } => Some("save current input cursor".to_string()),
        LinearOp::RestoreCursor { .. } => Some("restore previously saved cursor".to_string()),
        LinearOp::WriteToField { offset, width, .. } => {
            Some(format!("write output field at +{offset} ({width:?})"))
        }
        LinearOp::ReadFromField { offset, width, .. } => {
            Some(format!("read output field at +{offset} ({width:?})"))
        }
        LinearOp::SaveOutPtr { .. } => Some("save output base pointer".to_string()),
        LinearOp::SetOutPtr { .. } => Some("set output pointer".to_string()),
        LinearOp::SlotAddr { slot, .. } => {
            Some(format!("compute spill slot {} address", slot.index()))
        }
        LinearOp::WriteToSlot { slot, .. } => Some(format!("write spill slot {}", slot.index())),
        LinearOp::ReadFromSlot { slot, .. } => Some(format!("read spill slot {}", slot.index())),
        LinearOp::CallIntrinsic { .. } => Some("call runtime intrinsic".to_string()),
        LinearOp::CallPure { .. } => Some("call pure helper".to_string()),
        LinearOp::CallLambda { target, .. } => Some(format!("call lambda @{}", target.index())),
        LinearOp::SimdStringScan { .. } => Some("simd accelerated string scan".to_string()),
        LinearOp::SimdWhitespaceSkip => Some("simd accelerated whitespace skip".to_string()),
        LinearOp::ErrorExit { code } => Some(format!("raise decode error {code:?}")),
        _ => None,
    }
}

fn fmt_terminator(f: &mut std::fmt::Formatter<'_>, term: &RaTerminator) -> std::fmt::Result {
    match term {
        RaTerminator::Return => write!(f, "return"),
        RaTerminator::ErrorExit { code } => write!(f, "error_exit({code:?})"),
        RaTerminator::Branch { target } => write!(f, "branch b{}", target.0),
        RaTerminator::BranchIf {
            cond,
            target,
            fallthrough,
        } => write!(
            f,
            "branch_if v{} -> b{}, fallthrough b{}",
            cond.index(),
            target.0,
            fallthrough.0
        ),
        RaTerminator::BranchIfZero {
            cond,
            target,
            fallthrough,
        } => write!(
            f,
            "branch_if_zero v{} -> b{}, fallthrough b{}",
            cond.index(),
            target.0,
            fallthrough.0
        ),
        RaTerminator::JumpTable {
            predicate,
            targets,
            default,
        } => {
            write!(f, "jump_table v{} [", predicate.index())?;
            for (i, t) in targets.iter().enumerate() {
                if i > 0 {
                    write!(f, ", ")?;
                }
                write!(f, "b{}", t.0)?;
            }
            write!(f, "], default b{}", default.0)
        }
    }
}

fn ra_term_semantic_hint(term: &RaTerminator) -> Option<&'static str> {
    match term {
        RaTerminator::Return => Some("finish function"),
        RaTerminator::ErrorExit { .. } => Some("terminate with error"),
        RaTerminator::Branch { .. } => Some("unconditional jump"),
        RaTerminator::BranchIf { .. } => Some("branch when condition is non-zero"),
        RaTerminator::BranchIfZero { .. } => Some("branch when condition is zero"),
        RaTerminator::JumpTable { .. } => Some("indexed branch table dispatch"),
    }
}

// r[impl ir.regalloc.ra-mir]
// r[impl ir.regalloc.ra-mir.block-params]
// r[impl ir.regalloc.ra-mir.operands]
// r[impl ir.regalloc.ra-mir.calls]
pub fn lower_linear_ir(ir: &LinearIr) -> RaProgram {
    let mut funcs = Vec::new();
    let mut i = 0usize;

    while i < ir.ops.len() {
        let (lambda_id, data_args, data_results) = match &ir.ops[i] {
            LinearOp::FuncStart {
                lambda_id,
                data_args,
                data_results,
                ..
            } => (*lambda_id, data_args.clone(), data_results.clone()),
            other => panic!("expected FuncStart at op {i}, got {other:?}"),
        };

        let mut depth = 1usize;
        let mut end = i + 1;
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
        let body = &ir.ops[i + 1..end];
        funcs.push(lower_function(
            lambda_id,
            data_args,
            data_results,
            body,
            ir.vreg_count,
        ));
        i = end + 1;
    }

    RaProgram {
        funcs,
        vreg_count: ir.vreg_count,
        slot_count: ir.slot_count,
    }
}

#[derive(Debug, Clone)]
enum TempTerm {
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

fn lower_function(
    lambda_id: LambdaId,
    data_args: Vec<VReg>,
    data_results: Vec<VReg>,
    ops: &[LinearOp],
    vreg_count: u32,
) -> RaFunction {
    if ops.is_empty() {
        return RaFunction {
            lambda_id,
            entry: BlockId(0),
            data_args,
            data_results,
            blocks: vec![RaBlock {
                id: BlockId(0),
                label: None,
                params: Vec::new(),
                insts: Vec::new(),
                term_linear_op_index: 0,
                term: RaTerminator::Return,
                preds: Vec::new(),
                succs: Vec::new(),
            }],
        };
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

    let mut block_for_op = vec![usize::MAX; ops.len()];
    for bi in 0..leaders.len() {
        let start = leaders[bi];
        let end = if bi + 1 < leaders.len() {
            leaders[bi + 1]
        } else {
            ops.len()
        };
        for slot in &mut block_for_op[start..end] {
            *slot = bi;
        }
    }

    let mut labels = std::collections::HashMap::new();
    let mut blocks = Vec::new();
    let mut temp_terms = Vec::new();

    for bi in 0..leaders.len() {
        let start = leaders[bi];
        let end = if bi + 1 < leaders.len() {
            leaders[bi + 1]
        } else {
            ops.len()
        };

        let mut cursor = start;
        let mut label = None;
        if matches!(ops[cursor], LinearOp::Label(_))
            && let LinearOp::Label(l) = ops[cursor]
        {
            label = Some(l);
            labels.insert(l, BlockId(bi as u32));
            cursor += 1;
        }

        let mut insts = Vec::new();
        let mut term: Option<TempTerm> = None;
        let mut term_linear_op_index = usize::MAX;
        while cursor < end {
            let op = ops[cursor].clone();
            match op {
                LinearOp::Branch(target) => {
                    term = Some(TempTerm::Branch(target));
                    term_linear_op_index = cursor;
                    cursor += 1;
                    break;
                }
                LinearOp::BranchIf { cond, target } => {
                    term = Some(TempTerm::BranchIf { cond, target });
                    term_linear_op_index = cursor;
                    cursor += 1;
                    break;
                }
                LinearOp::BranchIfZero { cond, target } => {
                    term = Some(TempTerm::BranchIfZero { cond, target });
                    term_linear_op_index = cursor;
                    cursor += 1;
                    break;
                }
                LinearOp::JumpTable {
                    predicate,
                    labels,
                    default,
                } => {
                    term = Some(TempTerm::JumpTable {
                        predicate,
                        labels,
                        default,
                    });
                    term_linear_op_index = cursor;
                    cursor += 1;
                    break;
                }
                LinearOp::ErrorExit { code } => {
                    term = Some(TempTerm::ErrorExit(code));
                    term_linear_op_index = cursor;
                    cursor += 1;
                    break;
                }
                LinearOp::Label(_) | LinearOp::FuncStart { .. } | LinearOp::FuncEnd => {
                    panic!("unexpected structural op in function body: {:?}", op);
                }
                other => {
                    insts.push(lower_inst(cursor, other));
                    cursor += 1;
                }
            }
        }
        assert!(
            cursor == end,
            "non-terminator ops after terminator in block {bi}"
        );

        if term.is_none() {
            term_linear_op_index = insts
                .last()
                .map(|inst| inst.linear_op_index)
                .unwrap_or_else(|| cursor.saturating_sub(1));
            if bi + 1 < leaders.len() {
                term = Some(TempTerm::Fallthrough(bi + 1));
            } else {
                term = Some(TempTerm::Return);
            }
        }
        assert_ne!(
            term_linear_op_index,
            usize::MAX,
            "block {bi} missing term linear op index"
        );

        blocks.push(RaBlock {
            id: BlockId(bi as u32),
            label,
            params: Vec::new(),
            insts,
            term_linear_op_index,
            term: RaTerminator::Return,
            preds: Vec::new(),
            succs: Vec::new(),
        });
        temp_terms.push(term.expect("term just set"));
    }

    for (bi, tt) in temp_terms.iter().enumerate() {
        let next = if bi + 1 < blocks.len() {
            Some(BlockId((bi + 1) as u32))
        } else {
            None
        };
        blocks[bi].term = resolve_term(tt, &labels, next);
    }

    let mut use_sets = vec![vec![false; vreg_count as usize]; blocks.len()];
    let mut def_sets = vec![vec![false; vreg_count as usize]; blocks.len()];
    for (i, b) in blocks.iter().enumerate() {
        collect_use_def(b, &mut use_sets[i], &mut def_sets[i]);
    }

    if let Some(first) = blocks.first() {
        let defs = &mut def_sets[first.id.index()];
        for &arg in &data_args {
            defs[arg.index()] = true;
        }
    }

    let mut live_in = vec![vec![false; vreg_count as usize]; blocks.len()];
    let mut live_out = vec![vec![false; vreg_count as usize]; blocks.len()];
    loop {
        let mut changed = false;
        for bi in (0..blocks.len()).rev() {
            let mut out = vec![false; vreg_count as usize];
            for succ in blocks[bi].term.successors() {
                for (idx, v) in live_in[succ.index()].iter().enumerate() {
                    out[idx] |= *v;
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
            blocks[bi].params = Vec::new();
            continue;
        }
        let mut params = Vec::new();
        for (idx, live) in live_in[bi].iter().enumerate() {
            if *live {
                params.push(VReg::new(idx as u32));
            }
        }
        blocks[bi].params = params;
    }

    let succ_lists: Vec<Vec<BlockId>> = blocks.iter().map(|b| b.term.successors()).collect();
    for bi in 0..blocks.len() {
        blocks[bi].succs = succ_lists[bi]
            .iter()
            .map(|to| {
                let args = blocks[to.index()]
                    .params
                    .iter()
                    .copied()
                    .map(|target| RaEdgeArg {
                        target,
                        source: target,
                    })
                    .collect();
                RaEdge { to: *to, args }
            })
            .collect();
    }
    for block in &mut blocks {
        coalesce_uncond_branch_tail_copies(block);
    }
    for (from, succs) in succ_lists.iter().enumerate() {
        for succ in succs {
            blocks[succ.index()].preds.push(BlockId(from as u32));
        }
    }

    RaFunction {
        lambda_id,
        entry: BlockId(0),
        data_args,
        data_results,
        blocks,
    }
}

fn resolve_term(
    t: &TempTerm,
    labels: &std::collections::HashMap<LabelId, BlockId>,
    next: Option<BlockId>,
) -> RaTerminator {
    match t {
        TempTerm::Return => RaTerminator::Return,
        TempTerm::ErrorExit(code) => RaTerminator::ErrorExit { code: *code },
        TempTerm::Branch(label) => RaTerminator::Branch {
            target: *labels
                .get(label)
                .unwrap_or_else(|| panic!("unknown label target: {:?}", label)),
        },
        TempTerm::BranchIf { cond, target } => RaTerminator::BranchIf {
            cond: *cond,
            target: *labels
                .get(target)
                .unwrap_or_else(|| panic!("unknown label target: {:?}", target)),
            fallthrough: next.expect("BranchIf must have fallthrough block"),
        },
        TempTerm::BranchIfZero { cond, target } => RaTerminator::BranchIfZero {
            cond: *cond,
            target: *labels
                .get(target)
                .unwrap_or_else(|| panic!("unknown label target: {:?}", target)),
            fallthrough: next.expect("BranchIfZero must have fallthrough block"),
        },
        TempTerm::JumpTable {
            predicate,
            labels: ls,
            default,
        } => RaTerminator::JumpTable {
            predicate: *predicate,
            targets: ls
                .iter()
                .map(|l| {
                    *labels
                        .get(l)
                        .unwrap_or_else(|| panic!("unknown jump-table label: {:?}", l))
                })
                .collect(),
            default: *labels
                .get(default)
                .unwrap_or_else(|| panic!("unknown jump-table default: {:?}", default)),
        },
        TempTerm::Fallthrough(next_idx) => RaTerminator::Branch {
            target: BlockId(*next_idx as u32),
        },
    }
}

// r[impl ir.passes.pre-regalloc.coalescing]
fn coalesce_uncond_branch_tail_copies(block: &mut RaBlock) {
    let is_likely_synthetic_fallthrough = block
        .insts
        .last()
        .is_none_or(|last| block.term_linear_op_index <= last.linear_op_index);
    if is_likely_synthetic_fallthrough {
        // Skip synthetic fallthrough edges; keep this optimization on explicit
        // branch ops only where edge/value intent is unambiguous.
        return;
    }
    let RaTerminator::Branch { target } = block.term else {
        return;
    };
    if target.index() == 0 || target.index() <= block.id.index() || block.succs.len() != 1 {
        // Keep entry and backward-edge behavior conservative for now.
        return;
    }

    let mut tail_start = block.insts.len();
    while tail_start > 0 && matches!(block.insts[tail_start - 1].op, LinearOp::Copy { .. }) {
        tail_start -= 1;
    }
    if tail_start == block.insts.len() {
        return;
    }
    let last_idx = block.insts.len() - 1;
    let LinearOp::Copy { dst, src } = block.insts[last_idx].op else {
        unreachable!("tail range should only contain copy ops");
    };
    if dst == src {
        return;
    }
    let edge_args_before = block.succs[0].args.clone();
    let has_dst_arg = edge_args_before.iter().any(|arg| arg.source == dst);
    let has_other_src_arg = edge_args_before
        .iter()
        .any(|arg| arg.source == src && arg.source != dst);
    if has_dst_arg && has_other_src_arg {
        // Avoid introducing duplicate source args on the same edge; this can
        // destabilize parallel block-parameter moves.
        return;
    }

    let tail = &block.insts[tail_start..];
    let mut original_sym = std::collections::HashMap::<VReg, VReg>::new();
    for inst in tail {
        let LinearOp::Copy { dst, src } = inst.op else {
            unreachable!("tail range should only contain copy ops");
        };
        let resolved = original_sym.get(&src).copied().unwrap_or(src);
        original_sym.insert(dst, resolved);
    }
    let original_edge_values: Vec<VReg> = edge_args_before
        .iter()
        .map(|arg| original_sym.get(&arg.source).copied().unwrap_or(arg.source))
        .collect();

    let mut rewritten_sym = std::collections::HashMap::<VReg, VReg>::new();
    for inst in &tail[..tail.len() - 1] {
        let LinearOp::Copy { dst, src } = inst.op else {
            unreachable!("tail range should only contain copy ops");
        };
        let resolved = rewritten_sym.get(&src).copied().unwrap_or(src);
        rewritten_sym.insert(dst, resolved);
    }
    let rewritten_edge_values: Vec<VReg> = edge_args_before
        .iter()
        .map(|arg| if arg.source == dst { src } else { arg.source })
        .map(|arg| rewritten_sym.get(&arg).copied().unwrap_or(arg))
        .collect();
    if original_edge_values != rewritten_edge_values {
        return;
    }

    let edge = &mut block.succs[0];
    let mut rewrote = false;
    for arg in &mut edge.args {
        if arg.source == dst {
            arg.source = src;
            rewrote = true;
        }
    }
    if !rewrote {}
    // Keep instruction/operand indexing fully stable for backend/regalloc.
    // We coalesce by rewriting edge args only; the original tail copy remains
    // in place (possibly dead) to preserve all mapping invariants.
}

fn push_use(out: &mut Vec<RaOperand>, v: VReg, fixed: Option<FixedReg>) {
    out.push(RaOperand {
        vreg: v,
        kind: OperandKind::Use,
        class: RegClass::Gpr,
        fixed,
    });
}

fn push_def(out: &mut Vec<RaOperand>, v: VReg, fixed: Option<FixedReg>) {
    out.push(RaOperand {
        vreg: v,
        kind: OperandKind::Def,
        class: RegClass::Gpr,
        fixed,
    });
}

fn lower_inst(linear_op_index: usize, op: LinearOp) -> RaInst {
    let mut operands = Vec::new();
    let mut clobbers = RaClobbers::default();

    match &op {
        LinearOp::Const { dst, .. }
        | LinearOp::ReadBytes { dst, .. }
        | LinearOp::PeekByte { dst }
        | LinearOp::SaveCursor { dst }
        | LinearOp::ReadFromField { dst, .. }
        | LinearOp::SaveOutPtr { dst }
        | LinearOp::SlotAddr { dst, .. }
        | LinearOp::ReadFromSlot { dst, .. } => {
            push_def(&mut operands, *dst, None);
        }
        LinearOp::BinOp {
            dst, lhs, rhs, op, ..
        } => {
            push_use(&mut operands, *lhs, None);
            // On x86_64, variable shifts require the count in cl (rcx hw_enc=1).
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
        LinearOp::CallIntrinsic { args, dst, .. } => {
            for (i, &arg) in args.iter().enumerate() {
                let _ = i;
                push_use(&mut operands, arg, None);
            }
            if let Some(dst) = dst {
                push_def(&mut operands, *dst, None);
            }
            clobbers = RaClobbers {
                caller_saved_gpr: true,
                caller_saved_simd: true,
            };
        }
        LinearOp::CallPure { args, dst, .. } => {
            for &arg in args {
                push_use(&mut operands, arg, None);
            }
            push_def(&mut operands, *dst, None);
            clobbers = RaClobbers {
                caller_saved_gpr: true,
                caller_saved_simd: true,
            };
        }
        LinearOp::CallLambda { args, results, .. } => {
            for (i, &arg) in args.iter().enumerate() {
                // x0/x1 are reserved for out_ptr + ctx in lambda calls; data args start at x2.
                push_use(&mut operands, arg, Some(FixedReg::AbiArg((i + 2) as u8)));
            }
            for (i, &r) in results.iter().enumerate() {
                push_def(&mut operands, r, Some(FixedReg::AbiRet(i as u8)));
            }
            clobbers = RaClobbers {
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
            panic!("unexpected non-inst op in lower_inst: {:?}", op);
        }
    }

    RaInst {
        linear_op_index,
        op,
        operands,
        clobbers,
    }
}

fn collect_use_def(block: &RaBlock, use_set: &mut [bool], def_set: &mut [bool]) {
    for inst in &block.insts {
        for op in &inst.operands {
            match op.kind {
                OperandKind::Use => {
                    if !def_set[op.vreg.index()] {
                        use_set[op.vreg.index()] = true;
                    }
                }
                OperandKind::Def => {
                    def_set[op.vreg.index()] = true;
                }
            }
        }
    }
    for v in block.term.uses() {
        if !def_set[v.index()] {
            use_set[v.index()] = true;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use kajit_ir::{IntrinsicFn, IrBuilder, IrOp, Width};
    use kajit_lir::linearize;

    fn edge_arg_values(args: &[RaEdgeArg]) -> Vec<VReg> {
        args.iter().map(|arg| arg.source).collect()
    }

    // r[verify ir.regalloc.ra-mir.block-params]
    #[test]
    fn ra_mir_has_block_params_for_gamma_join() {
        let mut builder = IrBuilder::new(<u32 as facet::Facet>::SHAPE);
        {
            let mut rb = builder.root_region();
            let pred = rb.const_val(0);
            let out = rb.gamma(pred, &[], 2, |branch_idx, bb| {
                let val = if branch_idx == 0 {
                    bb.const_val(42)
                } else {
                    bb.const_val(99)
                };
                bb.set_results(&[val]);
            });
            rb.write_to_field(out[0], 0, Width::W4);
            rb.set_results(&[]);
        }
        let mut func = builder.finish();
        let lin = linearize(&mut func);
        let ra = lower_linear_ir(&lin);
        let root = &ra.funcs[0];

        assert!(
            root.blocks
                .iter()
                .any(|b| b.preds.len() >= 2 && !b.params.is_empty()),
            "expected merge block with params; got: {:#?}",
            root.blocks
        );
    }

    // r[verify ir.regalloc.ra-mir.block-params]
    #[test]
    fn ra_mir_has_block_params_for_theta_loop_header() {
        let mut builder = IrBuilder::new(<u32 as facet::Facet>::SHAPE);
        {
            let mut rb = builder.root_region();
            let init_count = rb.const_val(5);
            let one = rb.const_val(1);
            let _ = rb.theta(&[init_count, one], |bb| {
                let args = bb.region_args(2);
                let counter = args[0];
                let one = args[1];
                let next = bb.binop(IrOp::Sub, counter, one);
                bb.set_results(&[next, next, one]);
            });
            rb.set_results(&[]);
        }
        let mut func = builder.finish();
        let lin = linearize(&mut func);
        let ra = lower_linear_ir(&lin);
        let root = &ra.funcs[0];

        assert!(
            root.blocks
                .iter()
                .any(|b| b.preds.len() >= 2 && !b.params.is_empty()),
            "expected loop header with params; got: {:#?}",
            root.blocks
        );
    }

    // r[verify ir.passes.pre-regalloc.coalescing]
    #[test]
    fn ra_mir_coalesces_tail_copies_on_uncond_branch_edges() {
        let mut builder = IrBuilder::new(<u32 as facet::Facet>::SHAPE);
        {
            let mut rb = builder.root_region();
            let pred = rb.const_val(0);
            let out = rb.gamma(pred, &[], 2, |branch_idx, bb| {
                let val = if branch_idx == 0 {
                    bb.const_val(42)
                } else {
                    bb.const_val(99)
                };
                bb.set_results(&[val]);
            });
            rb.write_to_field(out[0], 0, Width::W4);
            rb.set_results(&[]);
        }
        let mut func = builder.finish();
        let lin = linearize(&mut func);
        assert!(
            lin.ops.iter().any(|op| matches!(op, LinearOp::Copy { .. })),
            "expected linearized copies"
        );

        let ra = lower_linear_ir(&lin);
        let root = &ra.funcs[0];
        let merge = root
            .blocks
            .iter()
            .find(|b| b.preds.len() >= 2 && !b.params.is_empty())
            .expect("missing merge block with params");
        let mut saw_non_identity_edge = false;
        for pred in &merge.preds {
            let pred_block = &root.blocks[pred.index()];
            let edge = pred_block
                .succs
                .iter()
                .find(|e| e.to == merge.id)
                .expect("pred should have edge to merge");
            let edge_values = edge_arg_values(&edge.args);
            assert_eq!(edge_values.len(), merge.params.len());
            if edge_values != merge.params {
                saw_non_identity_edge = true;
            }
        }
        assert!(
            saw_non_identity_edge,
            "expected at least one coalesced edge argument to differ from merge params"
        );
    }

    // r[verify ir.passes.pre-regalloc.coalescing]
    #[test]
    fn ra_mir_coalesces_multiple_tail_copies_on_uncond_branch_edges() {
        let mut builder = IrBuilder::new(<u32 as facet::Facet>::SHAPE);
        {
            let mut rb = builder.root_region();
            let pred = rb.const_val(0);
            let out = rb.gamma(pred, &[], 2, |branch_idx, bb| {
                let lhs = if branch_idx == 0 {
                    bb.const_val(11)
                } else {
                    bb.const_val(33)
                };
                let rhs = if branch_idx == 0 {
                    bb.const_val(22)
                } else {
                    bb.const_val(44)
                };
                bb.set_results(&[lhs, rhs]);
            });
            rb.write_to_field(out[0], 0, Width::W4);
            rb.write_to_field(out[1], 4, Width::W4);
            rb.set_results(&[]);
        }
        let mut func = builder.finish();
        let lin = linearize(&mut func);
        let ra = lower_linear_ir(&lin);
        let root = &ra.funcs[0];
        let merge = root
            .blocks
            .iter()
            .find(|b| b.preds.len() >= 2 && b.params.len() >= 2)
            .expect("missing merge block with multiple params");

        let mut saw_multi_tail_block = false;
        let mut saw_rewrite_in_multi_tail_block = false;
        for pred in &merge.preds {
            let pred_block = &root.blocks[pred.index()];
            let edge = pred_block
                .succs
                .iter()
                .find(|e| e.to == merge.id)
                .expect("pred should have edge to merge");
            let edge_values = edge_arg_values(&edge.args);
            assert_eq!(edge_values.len(), merge.params.len());

            let mut tail_copies = Vec::new();
            for inst in pred_block.insts.iter().rev() {
                let LinearOp::Copy { dst, src } = inst.op else {
                    break;
                };
                tail_copies.push((dst, src));
            }
            if tail_copies.len() >= 2 {
                saw_multi_tail_block = true;
                if edge_values != merge.params {
                    saw_rewrite_in_multi_tail_block = true;
                }
            }
        }
        assert!(
            saw_multi_tail_block,
            "expected at least one predecessor with multiple tail copies"
        );
        assert!(
            saw_rewrite_in_multi_tail_block,
            "expected at least one rewritten edge arg in multi-tail-copy predecessor"
        );
    }

    // r[verify ir.regalloc.ra-mir.calls]
    #[test]
    fn ra_mir_call_operands_and_clobbers_are_modeled() {
        unsafe extern "C" fn add3(_ctx: *mut core::ffi::c_void, a: u64, b: u64, c: u64) -> u64 {
            a + b + c
        }

        let mut builder = IrBuilder::new(<u64 as facet::Facet>::SHAPE);
        {
            let mut rb = builder.root_region();
            let a = rb.const_val(1);
            let b = rb.const_val(2);
            let c = rb.const_val(3);
            let out = rb
                .call_intrinsic(IntrinsicFn(add3 as *const () as usize), &[a, b, c], 0, true)
                .expect("call should return output");
            rb.write_to_field(out, 0, Width::W8);
            rb.set_results(&[]);
        }
        let mut func = builder.finish();
        let lin = linearize(&mut func);
        let ra = lower_linear_ir(&lin);
        let root = &ra.funcs[0];
        let call = root
            .blocks
            .iter()
            .flat_map(|b| b.insts.iter())
            .find(|i| matches!(i.op, LinearOp::CallIntrinsic { .. }))
            .expect("missing call inst");

        assert!(call.clobbers.caller_saved_gpr);
        assert!(call.clobbers.caller_saved_simd);
        assert_eq!(call.operands.len(), 4);
        assert_eq!(call.operands[0].fixed, None);
        assert_eq!(call.operands[1].fixed, None);
        assert_eq!(call.operands[2].fixed, None);
        assert_eq!(call.operands[3].fixed, None);
    }

    #[test]
    fn ra_mir_canonical_and_human_display_modes() {
        let mut builder = IrBuilder::new(<u32 as facet::Facet>::SHAPE);
        {
            let mut rb = builder.root_region();
            let pred = rb.const_val(0);
            let out = rb.gamma(pred, &[], 2, |branch_idx, bb| {
                let val = if branch_idx == 0 {
                    bb.const_val(42)
                } else {
                    bb.const_val(99)
                };
                bb.set_results(&[val]);
            });
            rb.write_to_field(out[0], 0, Width::W4);
            rb.set_results(&[]);
        }
        let mut func = builder.finish();
        let lin = linearize(&mut func);
        let ra = lower_linear_ir(&lin);

        let canonical = format!("{ra}");
        let human_alt = format!("{ra:#}");
        let human_wrapper = format!("{}", ra.human());

        assert!(canonical.starts_with("ra_func @0 { ; entry: b0"));
        assert!(canonical.contains(" ; insts: "));
        assert!(canonical.contains(" ; uses: "));
        assert!(canonical.contains("term:"));
        assert!(!canonical.contains(" ; sem: "));
        assert!(!canonical.contains(" ; const_alias: "));

        assert!(human_alt.starts_with("ra_func @0 { ; entry: b0"));
        assert!(human_alt.contains(" ; consts: "));
        assert!(human_alt.contains(" ; sem: "));
        assert!(human_alt.contains(" ; const_alias: "));
        assert!(human_alt.contains(" ; defs_dbg: "));
        assert_eq!(human_alt, human_wrapper);
    }
}
