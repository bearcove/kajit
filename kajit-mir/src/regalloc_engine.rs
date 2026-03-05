//! regalloc2 integration over canonical CFG MIR.

use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};
use std::fmt;

use kajit_ir::ErrorCode;
use kajit_lir::{BinOpKind, LinearOp, UnaryOpKind};
use regalloc2::{
    Allocation, Block, Edit, Inst, InstRange, MachineEnv, Operand, OperandConstraint, OperandKind,
    OperandPos, Output, PReg, PRegSet, RegAllocError, RegClass, RegallocOptions, VReg,
};

use crate::cfg_mir::{self, OpId};
use crate::{FixedReg, OperandKind as RaOperandKind};
use kajit_ir::LambdaId;

/// Materialized allocation result for a full canonical CFG MIR program.
#[derive(Debug, Clone)]
pub struct AllocatedCfgProgram {
    pub cfg_program: cfg_mir::Program,
    pub functions: Vec<AllocatedCfgFunction>,
}

#[derive(Debug, Clone)]
pub struct CfgEdgeEdit {
    pub edge: cfg_mir::EdgeId,
    pub pos: regalloc2::InstPosition,
    pub from: Allocation,
    pub to: Allocation,
}

#[derive(Debug, Clone)]
pub struct AllocatedCfgFunction {
    pub lambda_id: LambdaId,
    pub num_spillslots: usize,
    pub edits: Vec<(cfg_mir::ProgPoint, Edit)>,
    pub op_allocs: HashMap<cfg_mir::OpId, Vec<Allocation>>,
    pub op_operands: HashMap<cfg_mir::OpId, Vec<(kajit_ir::VReg, RaOperandKind)>>,
    pub edge_edits: Vec<CfgEdgeEdit>,
    pub return_result_allocs: Vec<Allocation>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ExecutionPosition {
    pub block: crate::BlockId,
    pub next_inst_index: usize,
    pub at_terminator: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExecutionState {
    pub physical_registers: HashMap<PReg, u64>,
    pub spillslots: HashMap<usize, u64>,
    pub position: ExecutionPosition,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExecutionResult {
    pub state: ExecutionState,
    pub output: Vec<u8>,
    pub cursor: usize,
    pub trap: Option<crate::InterpreterTrap>,
    pub returned: bool,
}

#[derive(Debug, Clone)]
pub enum RegallocEngineError {
    Regalloc(RegAllocError),
    Checker(String),
    StaticVerifier(String),
    Simulation(String),
}

impl fmt::Display for RegallocEngineError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Regalloc(err) => write!(f, "{err}"),
            Self::Checker(msg) => write!(f, "{msg}"),
            Self::StaticVerifier(msg) => write!(f, "{msg}"),
            Self::Simulation(msg) => write!(f, "{msg}"),
        }
    }
}

impl std::error::Error for RegallocEngineError {}

impl From<RegAllocError> for RegallocEngineError {
    fn from(value: RegAllocError) -> Self {
        Self::Regalloc(value)
    }
}

#[derive(Debug, Clone)]
struct AdapterInst {
    operands: Vec<Operand>,
    operand_vregs: Vec<(kajit_ir::VReg, RaOperandKind)>,
    clobbers: PRegSet,
    is_branch: bool,
    is_ret: bool,
    ret_value_operand_start: usize,
    ret_value_operand_count: usize,
}

#[derive(Debug, Clone)]
struct AdapterBlock {
    inst_range: InstRange,
    succs: Vec<Block>,
    preds: Vec<Block>,
    params: Vec<VReg>,
    succ_args: Vec<Vec<VReg>>,
}

#[derive(Debug, Clone)]
enum BlockTermKind {
    Ret,
    BranchLike,
}

#[derive(Debug, Clone)]
struct WorkBlock {
    raw_insts: Vec<crate::RaInst>,
    raw_inst_op_ids: Vec<Option<cfg_mir::OpId>>,
    term_kind: BlockTermKind,
    term_returns_data_results: bool,
    term_linear_op_index: usize,
    term_op_id: Option<cfg_mir::OpId>,
    term_uses: Vec<kajit_ir::VReg>,
    succs: Vec<usize>,
    preds: Vec<usize>,
    params: Vec<kajit_ir::VReg>,
    succ_args: Vec<Vec<kajit_ir::VReg>>,
}

fn align_succ_arg_sources_to_target(
    target_params: &[kajit_ir::VReg],
    args: &[crate::RaEdgeArg],
) -> Vec<kajit_ir::VReg> {
    let source_by_target: HashMap<_, _> = args.iter().map(|arg| (arg.target, arg.source)).collect();

    target_params
        .iter()
        .map(|target| source_by_target.get(target).copied().unwrap_or(*target))
        .collect()
}

#[derive(Debug, Clone, Copy)]
struct EdgeBlockInfo {
    cfg_edge_id: Option<cfg_mir::EdgeId>,
}

#[derive(Debug, Clone)]
struct AdapterFunction {
    blocks: Vec<AdapterBlock>,
    insts: Vec<AdapterInst>,
    inst_op_ids: Vec<Option<cfg_mir::OpId>>,
    inst_edge_infos: Vec<Option<EdgeBlockInfo>>,
    num_vregs: usize,
    empty_vregs: Vec<VReg>,
}

fn int_vreg(v: kajit_ir::VReg) -> VReg {
    VReg::new(v.index(), RegClass::Int)
}

#[cfg(target_arch = "aarch64")]
fn preg_int(n: usize) -> PReg {
    PReg::new(n, RegClass::Int)
}

#[cfg(target_arch = "aarch64")]
fn preg_vec(n: usize) -> PReg {
    PReg::new(n, RegClass::Vector)
}

#[cfg(target_arch = "x86_64")]
fn preg_int(n: usize) -> PReg {
    PReg::new(n, RegClass::Int)
}

#[cfg(target_arch = "x86_64")]
fn preg_vec(n: usize) -> PReg {
    PReg::new(n, RegClass::Vector)
}

fn set_from_regs(regs: &[PReg]) -> PRegSet {
    let mut out = PRegSet::empty();
    for &reg in regs {
        out.add(reg);
    }
    out
}

#[cfg(target_arch = "aarch64")]
fn abi_arg_int(idx: usize) -> Option<PReg> {
    (idx <= 7).then(|| preg_int(idx))
}

#[cfg(target_arch = "aarch64")]
fn abi_ret_int(idx: usize) -> Option<PReg> {
    (idx <= 1).then(|| preg_int(idx))
}

#[cfg(all(target_arch = "x86_64", not(windows)))]
fn abi_arg_int(idx: usize) -> Option<PReg> {
    // SysV: rdi, rsi, rdx, rcx, r8, r9.
    const ORDER: [usize; 6] = [7, 6, 2, 1, 8, 9];
    ORDER.get(idx).copied().map(preg_int)
}

#[cfg(all(target_arch = "x86_64", windows))]
fn abi_arg_int(idx: usize) -> Option<PReg> {
    // Win64: rcx, rdx, r8, r9.
    const ORDER: [usize; 4] = [1, 2, 8, 9];
    ORDER.get(idx).copied().map(preg_int)
}

#[cfg(target_arch = "x86_64")]
fn abi_ret_int(idx: usize) -> Option<PReg> {
    // rax, rdx
    const ORDER: [usize; 2] = [0, 2];
    ORDER.get(idx).copied().map(preg_int)
}

fn fixed_preg(fixed: FixedReg) -> Option<PReg> {
    match fixed {
        FixedReg::AbiArg(i) => abi_arg_int(i as usize),
        FixedReg::AbiRet(i) => abi_ret_int(i as usize),
        FixedReg::HwReg(enc) => Some(PReg::new(enc as usize, RegClass::Int)),
    }
}

#[cfg(target_arch = "aarch64")]
fn caller_saved_gprs() -> PRegSet {
    let mut regs = Vec::new();
    for n in 0..=17 {
        regs.push(preg_int(n));
    }
    set_from_regs(&regs)
}

#[cfg(target_arch = "x86_64")]
fn caller_saved_gprs() -> PRegSet {
    #[cfg(not(windows))]
    let regs = [
        preg_int(0),  // rax
        preg_int(1),  // rcx
        preg_int(2),  // rdx
        preg_int(6),  // rsi
        preg_int(7),  // rdi
        preg_int(8),  // r8
        preg_int(9),  // r9
        preg_int(10), // r10
        preg_int(11), // r11
    ];
    #[cfg(windows)]
    let regs = [
        preg_int(0),  // rax
        preg_int(1),  // rcx
        preg_int(2),  // rdx
        preg_int(8),  // r8
        preg_int(9),  // r9
        preg_int(10), // r10
        preg_int(11), // r11
    ];
    set_from_regs(&regs)
}

#[cfg(target_arch = "aarch64")]
fn caller_saved_simd() -> PRegSet {
    let mut regs = Vec::new();
    for n in 0..=31 {
        regs.push(preg_vec(n));
    }
    set_from_regs(&regs)
}

#[cfg(target_arch = "x86_64")]
fn caller_saved_simd() -> PRegSet {
    let mut regs = Vec::new();
    for n in 0..=15 {
        regs.push(preg_vec(n));
    }
    set_from_regs(&regs)
}

#[cfg(target_arch = "aarch64")]
fn machine_env() -> MachineEnv {
    let preferred_int = set_from_regs(&[
        preg_int(23),
        preg_int(24),
        preg_int(25),
        preg_int(26),
        preg_int(11),
        preg_int(12),
        preg_int(13),
        preg_int(14),
        preg_int(15),
    ]);

    let mut non_pref_int_regs = Vec::new();
    for n in 0..=8 {
        non_pref_int_regs.push(preg_int(n));
    }
    // x10/x16 are used as fixed backend temporaries during lowering.
    // Keep them out of allocatable sets to avoid silent clobbers.
    non_pref_int_regs.push(preg_int(17));

    let non_preferred_int = set_from_regs(&non_pref_int_regs);

    let preferred_vec = set_from_regs(&[]);
    let non_preferred_vec = set_from_regs(&[]);

    MachineEnv {
        preferred_regs_by_class: [preferred_int, set_from_regs(&[]), preferred_vec],
        non_preferred_regs_by_class: [non_preferred_int, set_from_regs(&[]), non_preferred_vec],
        scratch_by_class: [Some(preg_int(9)), None, None],
        fixed_stack_slots: Vec::new(),
    }
}

// x64 allocatable registers:
//   Reserved: r10 (backend temp + regalloc2 scratch, like x9 on aarch64),
//             r11 (backend temp for ZigzagDecode),
//             r12 (cached cursor), r13 (cached end), r14 (out ptr), r15 (ctx ptr),
//             rbp/r5 (frame pointer), rsp/r4 (stack pointer)
//   rbx(3) is callee-saved and goes into preferred (saved/restored in prologue/epilogue).
//   All remaining caller-saved GPRs are allocatable in non_preferred.
#[cfg(all(target_arch = "x86_64", not(windows)))]
fn machine_env() -> MachineEnv {
    let preferred_int = set_from_regs(&[
        preg_int(3), // rbx (callee-saved)
    ]);
    // System V AMD64 caller-saved: rax(0), rcx(1), rdx(2), rsi(6), rdi(7), r8(8), r9(9)
    // r11 excluded: used as backend temp for ZigzagDecode
    let non_preferred_int = set_from_regs(&[
        preg_int(0), // rax
        preg_int(1), // rcx
        preg_int(2), // rdx
        preg_int(6), // rsi
        preg_int(7), // rdi
        preg_int(8), // r8
        preg_int(9), // r9
    ]);
    MachineEnv {
        preferred_regs_by_class: [preferred_int, set_from_regs(&[]), set_from_regs(&[])],
        non_preferred_regs_by_class: [non_preferred_int, set_from_regs(&[]), set_from_regs(&[])],
        scratch_by_class: [Some(preg_int(10)), None, None],
        fixed_stack_slots: Vec::new(),
    }
}

#[cfg(all(target_arch = "x86_64", windows))]
fn machine_env() -> MachineEnv {
    let preferred_int = set_from_regs(&[
        preg_int(3), // rbx (callee-saved)
    ]);
    // Windows x64 caller-saved: rax(0), rcx(1), rdx(2), r8(8), r9(9)
    // r11 excluded: used as backend temp for ZigzagDecode
    // rsi(6) and rdi(7) are callee-saved on Windows but we don't use them.
    let non_preferred_int = set_from_regs(&[
        preg_int(0), // rax
        preg_int(1), // rcx
        preg_int(2), // rdx
        preg_int(8), // r8
        preg_int(9), // r9
    ]);
    MachineEnv {
        preferred_regs_by_class: [preferred_int, set_from_regs(&[]), set_from_regs(&[])],
        non_preferred_regs_by_class: [non_preferred_int, set_from_regs(&[]), set_from_regs(&[])],
        scratch_by_class: [Some(preg_int(10)), None, None],
        fixed_stack_slots: Vec::new(),
    }
}

fn lower_operand(op: crate::regalloc_mir::RaOperand) -> Operand {
    let class = match op.class {
        crate::regalloc_mir::RegClass::Gpr => RegClass::Int,
        crate::regalloc_mir::RegClass::Simd => RegClass::Vector,
    };
    let vreg = VReg::new(op.vreg.index(), class);
    let constraint = match op.fixed.and_then(fixed_preg) {
        Some(preg) => OperandConstraint::FixedReg(preg),
        None => OperandConstraint::Reg,
    };
    let kind = match op.kind {
        RaOperandKind::Use => OperandKind::Use,
        RaOperandKind::Def => OperandKind::Def,
    };
    let pos = match kind {
        OperandKind::Use => OperandPos::Early,
        OperandKind::Def => OperandPos::Late,
    };
    Operand::new(vreg, constraint, kind, pos)
}

fn lower_term_uses_cfg(term: &cfg_mir::Terminator) -> Vec<kajit_ir::VReg> {
    match term {
        cfg_mir::Terminator::BranchIf { cond, .. }
        | cfg_mir::Terminator::BranchIfZero { cond, .. } => vec![*cond],
        cfg_mir::Terminator::JumpTable { predicate, .. } => vec![*predicate],
        cfg_mir::Terminator::Return
        | cfg_mir::Terminator::ErrorExit { .. }
        | cfg_mir::Terminator::Branch { .. } => Vec::new(),
    }
}

fn map_cfg_operand(op: cfg_mir::Operand) -> crate::RaOperand {
    crate::RaOperand {
        vreg: op.vreg,
        kind: match op.kind {
            cfg_mir::OperandKind::Use => RaOperandKind::Use,
            cfg_mir::OperandKind::Def => RaOperandKind::Def,
        },
        class: match op.class {
            cfg_mir::RegClass::Gpr => crate::RegClass::Gpr,
            cfg_mir::RegClass::Simd => crate::RegClass::Simd,
        },
        fixed: op.fixed.map(|fixed| match fixed {
            cfg_mir::FixedReg::AbiArg(i) => FixedReg::AbiArg(i),
            cfg_mir::FixedReg::AbiRet(i) => FixedReg::AbiRet(i),
            cfg_mir::FixedReg::HwReg(enc) => FixedReg::HwReg(enc),
        }),
    }
}

fn map_cfg_clobbers(clobbers: cfg_mir::Clobbers) -> crate::RaClobbers {
    crate::RaClobbers {
        caller_saved_gpr: clobbers.caller_saved_gpr,
        caller_saved_simd: clobbers.caller_saved_simd,
    }
}

fn split_critical_edges_cfg(
    func: &cfg_mir::Function,
) -> Result<(Vec<WorkBlock>, Vec<Option<EdgeBlockInfo>>), RegallocEngineError> {
    let schedule = func
        .derive_schedule()
        .map_err(|err| RegallocEngineError::Checker(err.to_string()))?;

    let mut blocks: Vec<WorkBlock> = func
        .blocks
        .iter()
        .map(|b| {
            let term = &func.terms[b.term.index()];
            let term_kind = match term {
                cfg_mir::Terminator::Return | cfg_mir::Terminator::ErrorExit { .. } => {
                    BlockTermKind::Ret
                }
                _ => BlockTermKind::BranchLike,
            };
            let term_returns_data_results = matches!(term, cfg_mir::Terminator::Return);
            let mut raw_inst_op_ids = Vec::with_capacity(b.insts.len());
            let raw_insts = b
                .insts
                .iter()
                .map(|inst_id| {
                    let inst = &func.insts[inst_id.index()];
                    raw_inst_op_ids.push(Some(OpId::Inst(*inst_id)));
                    let linear_op_index = schedule
                        .op_to_index
                        .get(&OpId::Inst(*inst_id))
                        .copied()
                        .expect("schedule should include every instruction")
                        as usize;
                    crate::RaInst {
                        linear_op_index,
                        op: inst.op.clone(),
                        operands: inst.operands.iter().copied().map(map_cfg_operand).collect(),
                        clobbers: map_cfg_clobbers(inst.clobbers),
                    }
                })
                .collect();
            let term_linear_op_index = schedule
                .op_to_index
                .get(&OpId::Term(b.term))
                .copied()
                .expect("schedule should include every terminator")
                as usize;
            WorkBlock {
                raw_insts,
                raw_inst_op_ids,
                term_kind,
                term_returns_data_results,
                term_linear_op_index,
                term_op_id: Some(OpId::Term(b.term)),
                term_uses: lower_term_uses_cfg(term),
                succs: b
                    .succs
                    .iter()
                    .map(|edge_id| func.edges[edge_id.index()].to.index())
                    .collect(),
                preds: b
                    .preds
                    .iter()
                    .map(|edge_id| func.edges[edge_id.index()].from.index())
                    .collect(),
                params: b.params.clone(),
                succ_args: b
                    .succs
                    .iter()
                    .map(|edge_id| {
                        let edge = &func.edges[edge_id.index()];
                        let aligned_args: Vec<crate::RaEdgeArg> = edge
                            .args
                            .iter()
                            .copied()
                            .map(|arg| crate::RaEdgeArg {
                                source: arg.source,
                                target: arg.target,
                            })
                            .collect();
                        align_succ_arg_sources_to_target(
                            &func.blocks[edge.to.index()].params,
                            &aligned_args,
                        )
                    })
                    .collect(),
            }
        })
        .collect();
    let mut edge_infos = vec![None; blocks.len()];

    let mut from = 0usize;
    while from < blocks.len() {
        let succ_count = blocks[from].succs.len();
        for succ_idx in 0..succ_count {
            let to = blocks[from].succs[succ_idx];
            if succ_count <= 1 || blocks[to].preds.len() <= 1 {
                continue;
            }

            let to_params = blocks[to].params.clone();
            let edge_block_term_linear_index = blocks[from].term_linear_op_index;
            let cfg_edge_id = func.blocks[from].succs[succ_idx];
            let edge_succ_args = to_params.clone();
            let edge_block = WorkBlock {
                raw_insts: Vec::new(),
                raw_inst_op_ids: Vec::new(),
                term_kind: BlockTermKind::BranchLike,
                term_returns_data_results: false,
                term_linear_op_index: edge_block_term_linear_index,
                term_op_id: None,
                term_uses: Vec::new(),
                succs: vec![to],
                preds: vec![from],
                params: to_params,
                succ_args: vec![edge_succ_args],
            };
            let edge_id = blocks.len();
            blocks.push(edge_block);
            edge_infos.push(Some(EdgeBlockInfo {
                cfg_edge_id: Some(cfg_edge_id),
            }));

            blocks[from].succs[succ_idx] = edge_id;

            let pred_slot = blocks[to]
                .preds
                .iter()
                .position(|&p| p == from)
                .expect("target block should list source as predecessor");
            blocks[to].preds[pred_slot] = edge_id;
        }
        from += 1;
    }

    Ok((blocks, edge_infos))
}

impl AdapterFunction {
    fn from_work_blocks(
        data_args: &[kajit_ir::VReg],
        data_results: &[kajit_ir::VReg],
        num_vregs: usize,
        mut blocks: Vec<WorkBlock>,
        edge_infos: Vec<Option<EdgeBlockInfo>>,
    ) -> Self {
        let trace_self_loop = std::env::var_os("KAJIT_TRACE_SELF_LOOP").is_some();
        let mut adapter_insts = Vec::<AdapterInst>::new();
        let mut inst_op_ids = Vec::<Option<cfg_mir::OpId>>::new();
        let mut inst_edge_infos = Vec::<Option<EdgeBlockInfo>>::new();
        let mut adapter_blocks = Vec::<AdapterBlock>::new();

        for (block_index, b) in blocks.iter_mut().enumerate() {
            let start = Inst::new(adapter_insts.len());

            if block_index == 0 {
                let mut entry_operands = Vec::new();
                for (arg_idx, &arg) in data_args.iter().enumerate() {
                    let fixed = fixed_preg(FixedReg::AbiArg((arg_idx + 2) as u8))
                        .expect("entry data arg has unsupported ABI register index");
                    entry_operands.push(Operand::new(
                        int_vreg(arg),
                        OperandConstraint::FixedReg(fixed),
                        OperandKind::Def,
                        OperandPos::Late,
                    ));
                }
                if !entry_operands.is_empty() {
                    adapter_insts.push(AdapterInst {
                        operands: entry_operands,
                        operand_vregs: data_args
                            .iter()
                            .copied()
                            .map(|arg| (arg, RaOperandKind::Def))
                            .collect(),
                        clobbers: PRegSet::empty(),
                        is_branch: false,
                        is_ret: false,
                        ret_value_operand_start: 0,
                        ret_value_operand_count: 0,
                    });
                    inst_op_ids.push(None);
                    inst_edge_infos.push(None);
                }
            }

            for (inst_i, inst) in b.raw_insts.iter().enumerate() {
                let operands: Vec<Operand> =
                    inst.operands.iter().copied().map(lower_operand).collect();
                let operand_vregs: Vec<(kajit_ir::VReg, RaOperandKind)> =
                    inst.operands.iter().map(|op| (op.vreg, op.kind)).collect();
                let mut clobbers = PRegSet::empty();
                if inst.clobbers.caller_saved_gpr {
                    clobbers.union_from(caller_saved_gprs());
                }
                if inst.clobbers.caller_saved_simd {
                    clobbers.union_from(caller_saved_simd());
                }
                for operand in &operands {
                    if let OperandConstraint::FixedReg(preg) = operand.constraint()
                        && operand.kind() == OperandKind::Def
                    {
                        clobbers.remove(preg);
                    }
                }
                adapter_insts.push(AdapterInst {
                    operands,
                    operand_vregs,
                    clobbers,
                    is_branch: false,
                    is_ret: false,
                    ret_value_operand_start: 0,
                    ret_value_operand_count: 0,
                });
                inst_op_ids.push(b.raw_inst_op_ids.get(inst_i).and_then(|id| *id));
                inst_edge_infos.push(None);
            }

            let mut term_operands = b
                .term_uses
                .iter()
                .copied()
                .map(int_vreg)
                .map(Operand::reg_use)
                .collect::<Vec<_>>();
            let mut ret_value_operand_start = 0usize;
            let mut ret_value_operand_count = 0usize;
            if b.term_returns_data_results {
                ret_value_operand_start = term_operands.len();
                ret_value_operand_count = data_results.len();
                for (i, &result) in data_results.iter().enumerate() {
                    let fixed = fixed_preg(FixedReg::AbiRet(i as u8))
                        .expect("return data result has unsupported ABI register index");
                    term_operands.push(Operand::new(
                        int_vreg(result),
                        OperandConstraint::FixedReg(fixed),
                        OperandKind::Use,
                        OperandPos::Early,
                    ));
                }
            }
            let is_branch = matches!(b.term_kind, BlockTermKind::BranchLike);
            let is_ret = matches!(b.term_kind, BlockTermKind::Ret);
            let mut term_operand_vregs = b
                .term_uses
                .iter()
                .copied()
                .map(|v| (v, RaOperandKind::Use))
                .collect::<Vec<_>>();
            if b.term_returns_data_results {
                term_operand_vregs.extend(
                    data_results
                        .iter()
                        .copied()
                        .map(|v| (v, RaOperandKind::Use)),
                );
            }
            adapter_insts.push(AdapterInst {
                operands: term_operands,
                operand_vregs: term_operand_vregs,
                clobbers: PRegSet::empty(),
                is_branch,
                is_ret,
                ret_value_operand_start,
                ret_value_operand_count,
            });
            inst_op_ids.push(b.term_op_id);
            inst_edge_infos.push(edge_infos[block_index]);

            let end = Inst::new(adapter_insts.len());
            adapter_blocks.push(AdapterBlock {
                inst_range: InstRange::new(start, end),
                succs: b.succs.iter().copied().map(Block::new).collect(),
                preds: b.preds.iter().copied().map(Block::new).collect(),
                params: b.params.iter().copied().map(int_vreg).collect(),
                succ_args: b
                    .succ_args
                    .iter()
                    .map(|args| args.iter().copied().map(int_vreg).collect())
                    .collect(),
            });
            if trace_self_loop {
                let block = &adapter_blocks[block_index];
                let is_self_loop = block.succs.iter().any(|s| s.index() == block_index);
                let is_edge_of_self_loop = edge_infos[block_index].is_some()
                    && block.succs.iter().any(|s| {
                        adapter_blocks.get(s.index()).map_or(false, |target| {
                            target.succs.iter().any(|t| t.index() == s.index())
                        })
                    });
                if is_self_loop || is_edge_of_self_loop {
                    eprintln!(
                        "adapter block[{block_index}] inst_range={:?} preds={:?} succs={:?}",
                        block.inst_range, block.preds, block.succs
                    );
                    let params: Vec<usize> = block.params.iter().map(|v| v.vreg()).collect();
                    eprintln!("  params(vreg idx): {:?}", params);
                    for (succ_i, succ_args) in block.succ_args.iter().enumerate() {
                        let succ_arg_ids: Vec<usize> = succ_args.iter().map(|v| v.vreg()).collect();
                        eprintln!("  succ_args[{succ_i}] (vreg idx): {:?}", succ_arg_ids);
                    }
                }
            }
        }

        Self {
            blocks: adapter_blocks,
            insts: adapter_insts,
            inst_op_ids,
            inst_edge_infos,
            num_vregs,
            empty_vregs: Vec::new(),
        }
    }

    fn from_cfg(func: &cfg_mir::Function, num_vregs: usize) -> Result<Self, RegallocEngineError> {
        let (blocks, edge_infos) = split_critical_edges_cfg(func)?;
        Ok(Self::from_work_blocks(
            &func.data_args,
            &func.data_results,
            num_vregs,
            blocks,
            edge_infos,
        ))
    }
}

impl regalloc2::Function for AdapterFunction {
    fn num_insts(&self) -> usize {
        self.insts.len()
    }

    fn num_blocks(&self) -> usize {
        self.blocks.len()
    }

    fn entry_block(&self) -> Block {
        Block::new(0)
    }

    fn block_insns(&self, block: Block) -> InstRange {
        self.blocks[block.index()].inst_range
    }

    fn block_succs(&self, block: Block) -> &[Block] {
        &self.blocks[block.index()].succs
    }

    fn block_preds(&self, block: Block) -> &[Block] {
        &self.blocks[block.index()].preds
    }

    fn block_params(&self, block: Block) -> &[VReg] {
        &self.blocks[block.index()].params
    }

    fn is_ret(&self, insn: Inst) -> bool {
        self.insts[insn.index()].is_ret
    }

    fn is_branch(&self, insn: Inst) -> bool {
        self.insts[insn.index()].is_branch
    }

    fn branch_blockparams(&self, block: Block, _insn: Inst, succ_idx: usize) -> &[VReg] {
        self.blocks[block.index()]
            .succ_args
            .get(succ_idx)
            .map(Vec::as_slice)
            .unwrap_or(self.empty_vregs.as_slice())
    }

    fn inst_operands(&self, insn: Inst) -> &[Operand] {
        &self.insts[insn.index()].operands
    }

    fn inst_clobbers(&self, insn: Inst) -> PRegSet {
        self.insts[insn.index()].clobbers
    }

    fn num_vregs(&self) -> usize {
        self.num_vregs
    }

    fn spillslot_size(&self, regclass: RegClass) -> usize {
        match regclass {
            RegClass::Int | RegClass::Float => 1,
            RegClass::Vector => 2,
        }
    }

    fn allow_multiple_vreg_defs(&self) -> bool {
        true
    }
}

fn allocate_cfg_function(
    func: &cfg_mir::Function,
    vreg_count: usize,
    env: &MachineEnv,
    options: &RegallocOptions,
) -> Result<AllocatedCfgFunction, RegallocEngineError> {
    let adapter = AdapterFunction::from_cfg(func, vreg_count)?;
    let out = regalloc2::run(&adapter, env, options)?;

    #[cfg(debug_assertions)]
    {
        let mut checker = regalloc2::checker::Checker::new(&adapter, env);
        checker.prepare(&out);
        checker
            .run()
            .map_err(|errs| RegallocEngineError::Checker(format!("{errs:?}")))?;
    }

    Ok(materialize_cfg_output(&out, &adapter, func.lambda_id))
}

fn materialize_cfg_output(
    out: &Output,
    adapter: &AdapterFunction,
    lambda_id: LambdaId,
) -> AllocatedCfgFunction {
    let mut return_result_allocs = Vec::<Allocation>::new();
    let mut op_allocs = HashMap::<cfg_mir::OpId, Vec<Allocation>>::new();
    let mut op_operands = HashMap::<cfg_mir::OpId, Vec<(kajit_ir::VReg, RaOperandKind)>>::new();

    for i in 0..adapter.insts.len() {
        let Some(op_id) = adapter.inst_op_ids.get(i).and_then(|op| *op) else {
            continue;
        };
        let allocs = out.inst_allocs(Inst::new(i)).to_vec();
        op_allocs.insert(op_id, allocs.clone());
        op_operands.insert(op_id, adapter.insts[i].operand_vregs.clone());

        let inst = &adapter.insts[i];
        if inst.ret_value_operand_count == 0 {
            continue;
        }
        let start = inst.ret_value_operand_start;
        let end = start + inst.ret_value_operand_count;
        assert!(
            end <= allocs.len(),
            "missing return alloc operands for lambda {:?}: expected range {start}..{end}, got {}",
            lambda_id,
            allocs.len()
        );
        let candidate = allocs[start..end].to_vec();
        if return_result_allocs.is_empty() {
            return_result_allocs = candidate;
        } else if return_result_allocs != candidate {
            panic!(
                "inconsistent return allocs for lambda {:?}: {:?} vs {:?}",
                lambda_id, return_result_allocs, candidate
            );
        }
    }

    let mut edits = Vec::<(cfg_mir::ProgPoint, Edit)>::new();
    let mut edge_edits = Vec::<CfgEdgeEdit>::new();
    for (prog_point, edit) in &out.edits {
        let inst_index = prog_point.inst().index();
        let Some(edge_info) = adapter
            .inst_edge_infos
            .get(inst_index)
            .and_then(|info| *info)
        else {
            if let Some(op_id) = adapter.inst_op_ids.get(inst_index).and_then(|op| *op) {
                let point = match prog_point.pos() {
                    regalloc2::InstPosition::Before => cfg_mir::ProgPoint::Before(op_id),
                    regalloc2::InstPosition::After => cfg_mir::ProgPoint::After(op_id),
                };
                edits.push((point, edit.clone()));
            }
            continue;
        };

        if let Some(edge_id) = edge_info.cfg_edge_id {
            let Edit::Move { from, to } = edit;
            edge_edits.push(CfgEdgeEdit {
                edge: edge_id,
                pos: prog_point.pos(),
                from: *from,
                to: *to,
            });
        }
    }

    AllocatedCfgFunction {
        lambda_id,
        num_spillslots: out.num_spillslots,
        edits,
        op_allocs,
        op_operands,
        edge_edits,
        return_result_allocs,
    }
}

fn verify_allocation_is_valid(
    alloc: Allocation,
    num_spillslots: usize,
    field: &str,
    edge_label: &str,
) -> Result<(), RegallocEngineError> {
    if alloc.is_reg() {
        return Ok(());
    }
    if let Some(slot) = alloc.as_stack() {
        if slot.index() < num_spillslots {
            return Ok(());
        }
        return Err(RegallocEngineError::StaticVerifier(format!(
            "{edge_label}: invalid {field} allocation {alloc:?} (spillslot {} out of range, num_spillslots={num_spillslots})",
            slot.index()
        )));
    }
    Err(RegallocEngineError::StaticVerifier(format!(
        "{edge_label}: invalid {field} allocation kind {alloc:?}"
    )))
}

fn find_cfg_alloc_for_vreg_in_op(
    func: &AllocatedCfgFunction,
    op_id: cfg_mir::OpId,
    vreg: kajit_ir::VReg,
    preferred_kind: Option<RaOperandKind>,
) -> Option<Allocation> {
    let operands = func.op_operands.get(&op_id)?;
    let allocs = func.op_allocs.get(&op_id)?;
    for ((operand_vreg, operand_kind), alloc) in operands.iter().zip(allocs.iter().copied()) {
        if *operand_vreg != vreg {
            continue;
        }
        if preferred_kind.is_none_or(|k| *operand_kind == k) {
            return Some(alloc);
        }
    }
    None
}

fn infer_cfg_block_param_entry_alloc(
    cfg_func: &cfg_mir::Function,
    func: &AllocatedCfgFunction,
    block: &cfg_mir::Block,
    param: kajit_ir::VReg,
) -> Option<Allocation> {
    for inst_id in &block.insts {
        let op_id = cfg_mir::OpId::Inst(*inst_id);
        if let Some(a) = find_cfg_alloc_for_vreg_in_op(func, op_id, param, Some(RaOperandKind::Use))
        {
            return Some(a);
        }
        if let Some(a) = find_cfg_alloc_for_vreg_in_op(func, op_id, param, Some(RaOperandKind::Def))
        {
            return Some(a);
        }
    }
    let term_op = cfg_mir::OpId::Term(block.term);
    if cfg_func.term(block.term).is_some() {
        if let Some(a) =
            find_cfg_alloc_for_vreg_in_op(func, term_op, param, Some(RaOperandKind::Use))
        {
            return Some(a);
        }
    }
    None
}

fn infer_cfg_vreg_alloc_at_block_end(
    cfg_func: &cfg_mir::Function,
    func: &AllocatedCfgFunction,
    block: &cfg_mir::Block,
    vreg: kajit_ir::VReg,
) -> Option<Allocation> {
    let term_op = cfg_mir::OpId::Term(block.term);
    if cfg_func.term(block.term).is_some()
        && let Some(a) =
            find_cfg_alloc_for_vreg_in_op(func, term_op, vreg, Some(RaOperandKind::Use))
    {
        return Some(a);
    }
    None
}

#[derive(Clone)]
struct CfgExpectedEdgeTransfer {
    source_vreg: kajit_ir::VReg,
    target_vreg: kajit_ir::VReg,
    source_alloc: Option<Allocation>,
    target_alloc: Option<Allocation>,
}

fn cfg_expected_transfers_for_edge(
    cfg_func: &cfg_mir::Function,
    func: &AllocatedCfgFunction,
    edge_id: cfg_mir::EdgeId,
) -> Vec<CfgExpectedEdgeTransfer> {
    let mut out = Vec::new();
    let Some(edge) = cfg_func.edge(edge_id) else {
        return out;
    };
    let Some(pred_block) = cfg_func.block(edge.from) else {
        return out;
    };
    let Some(succ_block) = cfg_func.block(edge.to) else {
        return out;
    };

    for arg in &edge.args {
        let source_alloc =
            infer_cfg_vreg_alloc_at_block_end(cfg_func, func, pred_block, arg.source);
        let target_alloc =
            infer_cfg_block_param_entry_alloc(cfg_func, func, succ_block, arg.target);
        out.push(CfgExpectedEdgeTransfer {
            source_vreg: arg.source,
            target_vreg: arg.target,
            source_alloc,
            target_alloc,
        });
    }

    out
}

fn verify_function_static_edge_edits_cfg(
    cfg_func: Option<&cfg_mir::Function>,
    func: &AllocatedCfgFunction,
) -> Result<(), RegallocEngineError> {
    let scratch_regs: HashSet<PReg> = machine_env()
        .scratch_by_class
        .into_iter()
        .flatten()
        .collect();

    let mut by_edge = BTreeMap::<cfg_mir::EdgeId, Vec<&CfgEdgeEdit>>::new();
    for edge in &func.edge_edits {
        by_edge.entry(edge.edge).or_default().push(edge);
    }

    let mut edge_ids = BTreeSet::<cfg_mir::EdgeId>::new();
    edge_ids.extend(by_edge.keys().copied());
    if let Some(cfg_func) = cfg_func {
        for edge in &cfg_func.edges {
            edge_ids.insert(edge.id);
        }
    }

    for edge_id in edge_ids {
        let edits = by_edge.get(&edge_id).cloned().unwrap_or_default();
        let edge_label = format!("lambda @{} edge e{}", func.lambda_id.index(), edge_id.0);
        let expected = if let Some(cfg_func) = cfg_func {
            cfg_expected_transfers_for_edge(cfg_func, func, edge_id)
        } else {
            Vec::new()
        };

        let mut last_write_by_destination = HashMap::<Allocation, (Allocation, Allocation)>::new();
        for edit in &edits {
            verify_allocation_is_valid(edit.from, func.num_spillslots, "source", &edge_label)?;
            verify_allocation_is_valid(edit.to, func.num_spillslots, "target", &edge_label)?;
            last_write_by_destination.insert(edit.to, (edit.from, edit.to));
        }

        let actual_targets: HashSet<Allocation> =
            last_write_by_destination.keys().copied().collect();
        let mut tracked_transfers = Vec::<(usize, Allocation, Allocation)>::new();
        let mut expected_complete = true;
        for transfer in &expected {
            let (Some(source_alloc), Some(target_alloc)) =
                (transfer.source_alloc, transfer.target_alloc)
            else {
                expected_complete = false;
                continue;
            };
            if source_alloc == target_alloc {
                continue;
            }
            tracked_transfers.push((tracked_transfers.len(), source_alloc, target_alloc));
            if !actual_targets.contains(&target_alloc) {
                return Err(RegallocEngineError::StaticVerifier(format!(
                    "{edge_label}: missing edge edit to cover block param v{} from v{} (expected target {:?})",
                    transfer.target_vreg.index(),
                    transfer.source_vreg.index(),
                    target_alloc
                )));
            }
        }

        if !tracked_transfers.is_empty() {
            let mut symbols = HashMap::<Allocation, HashSet<usize>>::new();
            for (transfer_id, source_alloc, _) in &tracked_transfers {
                symbols
                    .entry(*source_alloc)
                    .or_default()
                    .insert(*transfer_id);
            }

            let mut suspicious_extra = None;
            for edit in edits
                .iter()
                .copied()
                .filter(|e| e.pos == regalloc2::InstPosition::Before)
                .chain(
                    edits
                        .iter()
                        .copied()
                        .filter(|e| e.pos == regalloc2::InstPosition::After),
                )
            {
                let moved = symbols.get(&edit.from).cloned().unwrap_or_default();
                if moved.is_empty()
                    && expected_complete
                    && edit
                        .to
                        .as_reg()
                        .is_none_or(|reg| !scratch_regs.contains(&reg))
                {
                    suspicious_extra = Some((edit.from, edit.to));
                }
                if !moved.is_empty() {
                    symbols.insert(edit.to, moved);
                }
            }

            for (transfer_id, source_alloc, target_alloc) in &tracked_transfers {
                let delivered = symbols
                    .get(target_alloc)
                    .is_some_and(|set| set.contains(transfer_id));
                if !delivered {
                    return Err(RegallocEngineError::StaticVerifier(format!(
                        "{edge_label}: edge edits do not deliver source {:?} to expected target {:?}",
                        source_alloc, target_alloc
                    )));
                }
            }
            if let Some((from, to)) = suspicious_extra {
                return Err(RegallocEngineError::StaticVerifier(format!(
                    "{edge_label}: suspicious extra edge edit {:?} -> {:?} (not connected to any expected block-param transfer)",
                    from, to
                )));
            }
        }
    }

    Ok(())
}

pub fn verify_static_edge_edits_cfg(
    program: &AllocatedCfgProgram,
) -> Result<(), RegallocEngineError> {
    for func in &program.functions {
        let cfg_func = program
            .cfg_program
            .funcs
            .iter()
            .find(|candidate| candidate.lambda_id == func.lambda_id);
        verify_function_static_edge_edits_cfg(cfg_func, func)?;
    }
    Ok(())
}

/// Run regalloc2 over canonical CFG MIR.
pub fn allocate_cfg_program(
    program: &cfg_mir::Program,
) -> Result<AllocatedCfgProgram, RegallocEngineError> {
    program
        .validate()
        .map_err(|err| RegallocEngineError::Checker(err.to_string()))?;

    let env = machine_env();
    let options = RegallocOptions {
        verbose_log: false,
        validate_ssa: false,
        algorithm: regalloc2::Algorithm::Ion,
    };

    let mut functions = Vec::with_capacity(program.funcs.len());
    for func in &program.funcs {
        functions.push(allocate_cfg_function(
            func,
            program.vreg_count as usize,
            &env,
            &options,
        )?);
    }

    let allocated = AllocatedCfgProgram {
        cfg_program: program.clone(),
        functions,
    };

    #[cfg(debug_assertions)]
    verify_static_edge_edits_cfg(&allocated)?;

    Ok(allocated)
}

const MAX_SIM_STEPS: usize = 1_000_000;
const SLOT_ADDR_STRIDE: usize = 16;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExecutionTraceEntry {
    pub step_index: usize,
    pub state: ExecutionState,
    pub output: Vec<u8>,
    pub cursor: usize,
    pub trap: Option<crate::InterpreterTrap>,
    pub returned: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DifferentialState {
    pub step_index: usize,
    pub position: ExecutionPosition,
    pub cursor: usize,
    pub output: Vec<u8>,
    pub trap: Option<crate::InterpreterTrap>,
    pub returned: bool,
    pub vregs: Option<Vec<u64>>,
    pub physical_registers: Option<HashMap<PReg, u64>>,
    pub spillslots: Option<HashMap<usize, u64>>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DifferentialDivergence {
    pub step_index: usize,
    pub field: String,
    pub ideal: DifferentialState,
    pub post_regalloc: DifferentialState,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DifferentialCheckResult {
    Match {
        steps: usize,
        ideal_final: DifferentialState,
        post_regalloc_final: DifferentialState,
    },
    Diverged(DifferentialDivergence),
    Error(String),
}

fn read_allocation(
    regs: &HashMap<PReg, u64>,
    spills: &HashMap<usize, u64>,
    alloc: Allocation,
) -> u64 {
    if let Some(reg) = alloc.as_reg() {
        return regs.get(&reg).copied().unwrap_or(0);
    }
    if let Some(slot) = alloc.as_stack() {
        return spills.get(&slot.index()).copied().unwrap_or(0);
    }
    0
}

fn write_allocation(
    regs: &mut HashMap<PReg, u64>,
    spills: &mut HashMap<usize, u64>,
    alloc: Allocation,
    value: u64,
) {
    if let Some(reg) = alloc.as_reg() {
        regs.insert(reg, value);
    } else if let Some(slot) = alloc.as_stack() {
        spills.insert(slot.index(), value);
    }
}

fn apply_moves(
    regs: &mut HashMap<PReg, u64>,
    spills: &mut HashMap<usize, u64>,
    edits: &[(Allocation, Allocation)],
) {
    for (from, to) in edits {
        let value = read_allocation(regs, spills, *from);
        write_allocation(regs, spills, *to, value);
    }
}

fn exec_binop(op: BinOpKind, lhs: u64, rhs: u64) -> u64 {
    match op {
        BinOpKind::Add => lhs.wrapping_add(rhs),
        BinOpKind::Sub => lhs.wrapping_sub(rhs),
        BinOpKind::And => lhs & rhs,
        BinOpKind::Or => lhs | rhs,
        BinOpKind::Xor => lhs ^ rhs,
        BinOpKind::Shl => {
            if rhs >= 64 {
                0
            } else {
                lhs.wrapping_shl(rhs as u32)
            }
        }
        BinOpKind::Shr => {
            if rhs >= 64 {
                0
            } else {
                lhs.wrapping_shr(rhs as u32)
            }
        }
        BinOpKind::CmpNe => u64::from(lhs != rhs),
    }
}

fn exec_unaryop(op: UnaryOpKind, src: u64) -> u64 {
    match op {
        UnaryOpKind::ZigzagDecode { wide } => {
            if wide {
                ((src >> 1) as i64 ^ -((src & 1) as i64)) as u64
            } else {
                let s = src as u32;
                let v = ((s >> 1) as i32) ^ -((s & 1) as i32);
                v as i64 as u64
            }
        }
        UnaryOpKind::SignExtend { from_width } => match from_width {
            kajit_ir::Width::W1 => (src as u8 as i8 as i64) as u64,
            kajit_ir::Width::W2 => (src as u16 as i16 as i64) as u64,
            kajit_ir::Width::W4 => (src as u32 as i32 as i64) as u64,
            kajit_ir::Width::W8 => src,
        },
    }
}

#[repr(C)]
struct RuntimeErrorSlot {
    code: u32,
    offset: u32,
}

#[repr(C)]
struct RuntimeDeserContext {
    input_ptr: *const u8,
    input_end: *const u8,
    error: RuntimeErrorSlot,
    key_scratch_ptr: *mut u8,
    key_scratch_cap: usize,
    trusted_utf8: bool,
}

impl RuntimeDeserContext {
    fn new(input: &[u8]) -> Self {
        let ptr = input.as_ptr();
        Self {
            input_ptr: ptr,
            input_end: unsafe { ptr.add(input.len()) },
            error: RuntimeErrorSlot { code: 0, offset: 0 },
            key_scratch_ptr: core::ptr::null_mut(),
            key_scratch_cap: 0,
            trusted_utf8: false,
        }
    }
}

impl Drop for RuntimeDeserContext {
    fn drop(&mut self) {
        if self.key_scratch_cap > 0 {
            unsafe {
                let layout = std::alloc::Layout::from_size_align_unchecked(self.key_scratch_cap, 1);
                std::alloc::dealloc(self.key_scratch_ptr, layout);
            }
        }
    }
}

fn error_code_from_u32(code: u32) -> ErrorCode {
    match code {
        0 => ErrorCode::Ok,
        1 => ErrorCode::UnexpectedEof,
        2 => ErrorCode::InvalidVarint,
        3 => ErrorCode::InvalidUtf8,
        4 => ErrorCode::UnsupportedShape,
        5 => ErrorCode::ExpectedObjectStart,
        6 => ErrorCode::ExpectedColon,
        7 => ErrorCode::ExpectedStringKey,
        8 => ErrorCode::UnterminatedString,
        9 => ErrorCode::InvalidJsonNumber,
        10 => ErrorCode::MissingRequiredField,
        11 => ErrorCode::UnexpectedCharacter,
        12 => ErrorCode::NumberOutOfRange,
        13 => ErrorCode::InvalidBool,
        14 => ErrorCode::UnknownVariant,
        15 => ErrorCode::ExpectedTagKey,
        16 => ErrorCode::AmbiguousVariant,
        17 => ErrorCode::AllocError,
        18 => ErrorCode::InvalidEscapeSequence,
        19 => ErrorCode::UnknownField,
        _ => ErrorCode::UnexpectedCharacter,
    }
}

struct SimulatorRuntime {
    input_base: *const u8,
    input_len: usize,
    out_ptr: *mut u8,
    slots: Vec<u64>,
    slot_mem: Vec<u8>,
    ctx: RuntimeDeserContext,
}

impl SimulatorRuntime {
    fn new(input: &[u8], slot_count: usize) -> Self {
        Self {
            input_base: input.as_ptr(),
            input_len: input.len(),
            out_ptr: core::ptr::null_mut(),
            slots: vec![0; slot_count],
            slot_mem: vec![0; slot_count.saturating_mul(SLOT_ADDR_STRIDE)],
            ctx: RuntimeDeserContext::new(input),
        }
    }

    fn init_out_ptr(&mut self, output: &mut Vec<u8>) {
        self.out_ptr = output.as_mut_ptr();
    }

    fn out_ptr_in_output(&self, output: &[u8]) -> Option<usize> {
        let base = output.as_ptr() as usize;
        let end = base + output.len();
        let ptr = self.out_ptr as usize;
        if ptr >= base && ptr <= end {
            Some(ptr - base)
        } else {
            None
        }
    }

    fn ensure_output_range_for_out_ptr(
        &mut self,
        output: &mut Vec<u8>,
        offset: usize,
        width: usize,
    ) {
        let Some(ptr_offset) = self.out_ptr_in_output(output) else {
            return;
        };
        let needed = ptr_offset.saturating_add(offset).saturating_add(width);
        if output.len() < needed {
            output.resize(needed, 0);
        }
        self.out_ptr = unsafe { output.as_mut_ptr().add(ptr_offset) };
    }

    fn ensure_slot(&mut self, slot: usize) {
        if slot >= self.slots.len() {
            self.slots.resize(slot + 1, 0);
        }
        let needed_bytes = (slot + 1) * SLOT_ADDR_STRIDE;
        if self.slot_mem.len() < needed_bytes {
            self.slot_mem.resize(needed_bytes, 0);
        }
    }

    fn slot_addr_value(&mut self, slot: usize) -> u64 {
        self.ensure_slot(slot);
        let ptr = self.slot_mem.as_mut_ptr();
        unsafe { ptr.add(slot * SLOT_ADDR_STRIDE) as u64 }
    }
}

fn set_trap_if_none(trap: &mut Option<crate::InterpreterTrap>, cursor: usize, code: ErrorCode) {
    if trap.is_none() {
        *trap = Some(crate::InterpreterTrap {
            code,
            offset: cursor as u32,
        });
    }
}

fn run_call_intrinsic(
    sim: &mut SimulatorRuntime,
    cursor: &mut usize,
    trap: &mut Option<crate::InterpreterTrap>,
    func: usize,
    args: &[u64],
    out_ptr: Option<*mut u8>,
) -> u64 {
    sim.ctx.input_ptr = unsafe { sim.input_base.add(*cursor) };
    sim.ctx.input_end = unsafe { sim.input_base.add(sim.input_len) };
    sim.ctx.error.code = 0;
    sim.ctx.error.offset = 0;

    let ctx_ptr = &mut sim.ctx as *mut RuntimeDeserContext;
    let ret = unsafe {
        match (out_ptr, args.len()) {
            (None, 0) => {
                let f: unsafe extern "C" fn(*mut RuntimeDeserContext) -> u64 =
                    core::mem::transmute(func);
                f(ctx_ptr)
            }
            (None, 1) => {
                let f: unsafe extern "C" fn(*mut RuntimeDeserContext, u64) -> u64 =
                    core::mem::transmute(func);
                f(ctx_ptr, args[0])
            }
            (None, 2) => {
                let f: unsafe extern "C" fn(*mut RuntimeDeserContext, u64, u64) -> u64 =
                    core::mem::transmute(func);
                f(ctx_ptr, args[0], args[1])
            }
            (None, 3) => {
                let f: unsafe extern "C" fn(*mut RuntimeDeserContext, u64, u64, u64) -> u64 =
                    core::mem::transmute(func);
                f(ctx_ptr, args[0], args[1], args[2])
            }
            (None, 4) => {
                let f: unsafe extern "C" fn(*mut RuntimeDeserContext, u64, u64, u64, u64) -> u64 =
                    core::mem::transmute(func);
                f(ctx_ptr, args[0], args[1], args[2], args[3])
            }
            (None, 5) => {
                let f: unsafe extern "C" fn(
                    *mut RuntimeDeserContext,
                    u64,
                    u64,
                    u64,
                    u64,
                    u64,
                ) -> u64 = core::mem::transmute(func);
                f(ctx_ptr, args[0], args[1], args[2], args[3], args[4])
            }
            (Some(out), 0) => {
                let f: unsafe extern "C" fn(*mut RuntimeDeserContext, *mut u8) =
                    core::mem::transmute(func);
                f(ctx_ptr, out);
                0
            }
            (Some(out), 1) => {
                let f: unsafe extern "C" fn(*mut RuntimeDeserContext, u64, *mut u8) =
                    core::mem::transmute(func);
                f(ctx_ptr, args[0], out);
                0
            }
            (Some(out), 2) => {
                let f: unsafe extern "C" fn(*mut RuntimeDeserContext, u64, u64, *mut u8) =
                    core::mem::transmute(func);
                f(ctx_ptr, args[0], args[1], out);
                0
            }
            (Some(out), 3) => {
                let f: unsafe extern "C" fn(*mut RuntimeDeserContext, u64, u64, u64, *mut u8) =
                    core::mem::transmute(func);
                f(ctx_ptr, args[0], args[1], args[2], out);
                0
            }
            _ => 0,
        }
    };

    *cursor = unsafe { sim.ctx.input_ptr.offset_from(sim.input_base) as usize };
    if sim.ctx.error.code != 0 {
        *trap = Some(crate::InterpreterTrap {
            code: error_code_from_u32(sim.ctx.error.code),
            offset: sim.ctx.error.offset,
        });
    }
    ret
}

fn run_call_pure(func: usize, args: &[u64]) -> u64 {
    unsafe {
        match args.len() {
            0 => {
                let f: unsafe extern "C" fn() -> u64 = core::mem::transmute(func);
                f()
            }
            1 => {
                let f: unsafe extern "C" fn(u64) -> u64 = core::mem::transmute(func);
                f(args[0])
            }
            2 => {
                let f: unsafe extern "C" fn(u64, u64) -> u64 = core::mem::transmute(func);
                f(args[0], args[1])
            }
            3 => {
                let f: unsafe extern "C" fn(u64, u64, u64) -> u64 = core::mem::transmute(func);
                f(args[0], args[1], args[2])
            }
            4 => {
                let f: unsafe extern "C" fn(u64, u64, u64, u64) -> u64 = core::mem::transmute(func);
                f(args[0], args[1], args[2], args[3])
            }
            5 => {
                let f: unsafe extern "C" fn(u64, u64, u64, u64, u64) -> u64 =
                    core::mem::transmute(func);
                f(args[0], args[1], args[2], args[3], args[4])
            }
            6 => {
                let f: unsafe extern "C" fn(u64, u64, u64, u64, u64, u64) -> u64 =
                    core::mem::transmute(func);
                f(args[0], args[1], args[2], args[3], args[4], args[5])
            }
            _ => 0,
        }
    }
}

fn find_operand_alloc(
    inst: &crate::RaInst,
    inst_allocs: &[Allocation],
    vreg: kajit_ir::VReg,
    kind: RaOperandKind,
) -> Result<Allocation, RegallocEngineError> {
    for (op, alloc) in inst.operands.iter().zip(inst_allocs.iter().copied()) {
        if op.vreg == vreg && op.kind == kind {
            return Ok(alloc);
        }
    }
    Err(RegallocEngineError::Simulation(format!(
        "missing allocation for {:?} operand v{} in linear op {}",
        kind,
        vreg.index(),
        inst.linear_op_index
    )))
}

#[allow(clippy::too_many_arguments)]
fn execute_sim_linear_op(
    op: &LinearOp,
    block_id: crate::BlockId,
    inst: &crate::RaInst,
    inst_allocs: &[Allocation],
    regs: &mut HashMap<PReg, u64>,
    spills: &mut HashMap<usize, u64>,
    sim: &mut SimulatorRuntime,
    cursor: &mut usize,
    output: &mut Vec<u8>,
    trap: &mut Option<crate::InterpreterTrap>,
) -> Result<(), RegallocEngineError> {
    match op {
        LinearOp::Const { dst, value } => {
            let dst_alloc = find_operand_alloc(inst, inst_allocs, *dst, RaOperandKind::Def)?;
            write_allocation(regs, spills, dst_alloc, *value);
        }
        LinearOp::Copy { dst, src } => {
            let src_alloc = find_operand_alloc(inst, inst_allocs, *src, RaOperandKind::Use)?;
            let dst_alloc = find_operand_alloc(inst, inst_allocs, *dst, RaOperandKind::Def)?;
            let value = read_allocation(regs, spills, src_alloc);
            write_allocation(regs, spills, dst_alloc, value);
        }
        LinearOp::BinOp { op, dst, lhs, rhs } => {
            let lhs_alloc = find_operand_alloc(inst, inst_allocs, *lhs, RaOperandKind::Use)?;
            let rhs_alloc = find_operand_alloc(inst, inst_allocs, *rhs, RaOperandKind::Use)?;
            let dst_alloc = find_operand_alloc(inst, inst_allocs, *dst, RaOperandKind::Def)?;
            let lhs_val = read_allocation(regs, spills, lhs_alloc);
            let rhs_val = read_allocation(regs, spills, rhs_alloc);
            write_allocation(regs, spills, dst_alloc, exec_binop(*op, lhs_val, rhs_val));
        }
        LinearOp::UnaryOp { op, dst, src } => {
            let src_alloc = find_operand_alloc(inst, inst_allocs, *src, RaOperandKind::Use)?;
            let dst_alloc = find_operand_alloc(inst, inst_allocs, *dst, RaOperandKind::Def)?;
            let src_val = read_allocation(regs, spills, src_alloc);
            write_allocation(regs, spills, dst_alloc, exec_unaryop(*op, src_val));
        }
        LinearOp::BoundsCheck { count } => {
            if *cursor + (*count as usize) > sim.input_len {
                set_trap_if_none(trap, *cursor, ErrorCode::UnexpectedEof);
            }
        }
        LinearOp::ReadBytes { dst, count } => {
            let count = *count as usize;
            if *cursor + count > sim.input_len {
                set_trap_if_none(trap, *cursor, ErrorCode::UnexpectedEof);
            } else {
                let input = unsafe { std::slice::from_raw_parts(sim.input_base, sim.input_len) };
                let mut value = 0u64;
                for i in 0..count {
                    value |= (input[*cursor + i] as u64) << (i * 8);
                }
                *cursor += count;
                let dst_alloc = find_operand_alloc(inst, inst_allocs, *dst, RaOperandKind::Def)?;
                write_allocation(regs, spills, dst_alloc, value);
            }
        }
        LinearOp::PeekByte { dst } => {
            if *cursor >= sim.input_len {
                set_trap_if_none(trap, *cursor, ErrorCode::UnexpectedEof);
            } else {
                let input = unsafe { std::slice::from_raw_parts(sim.input_base, sim.input_len) };
                let dst_alloc = find_operand_alloc(inst, inst_allocs, *dst, RaOperandKind::Def)?;
                write_allocation(regs, spills, dst_alloc, input[*cursor] as u64);
            }
        }
        LinearOp::AdvanceCursor { count } => {
            let count = *count as usize;
            if *cursor + count > sim.input_len {
                set_trap_if_none(trap, *cursor, ErrorCode::UnexpectedEof);
            } else {
                *cursor += count;
            }
        }
        LinearOp::AdvanceCursorBy { src } => {
            let src_alloc = find_operand_alloc(inst, inst_allocs, *src, RaOperandKind::Use)?;
            let count = read_allocation(regs, spills, src_alloc) as usize;
            if *cursor + count > sim.input_len {
                set_trap_if_none(trap, *cursor, ErrorCode::UnexpectedEof);
            } else {
                *cursor += count;
            }
        }
        LinearOp::SaveCursor { dst } => {
            let dst_alloc = find_operand_alloc(inst, inst_allocs, *dst, RaOperandKind::Def)?;
            write_allocation(regs, spills, dst_alloc, *cursor as u64);
        }
        LinearOp::RestoreCursor { src } => {
            let src_alloc = find_operand_alloc(inst, inst_allocs, *src, RaOperandKind::Use)?;
            let next = read_allocation(regs, spills, src_alloc) as usize;
            if next > sim.input_len {
                set_trap_if_none(trap, *cursor, ErrorCode::UnexpectedEof);
            } else {
                *cursor = next;
            }
        }
        LinearOp::WriteToField { src, offset, width } => {
            let src_alloc = find_operand_alloc(inst, inst_allocs, *src, RaOperandKind::Use)?;
            let value = read_allocation(regs, spills, src_alloc);
            let width_bytes = width.bytes() as usize;
            let base = *offset as usize;
            sim.ensure_output_range_for_out_ptr(output, base, width_bytes);
            unsafe {
                let dst = sim.out_ptr.add(base);
                for i in 0..width_bytes {
                    *dst.add(i) = ((value >> (i * 8)) & 0xff) as u8;
                }
            }
        }
        LinearOp::ReadFromField { dst, offset, width } => {
            let width_bytes = width.bytes() as usize;
            let base = *offset as usize;
            sim.ensure_output_range_for_out_ptr(output, base, width_bytes);
            let mut value = 0u64;
            unsafe {
                let src = sim.out_ptr.add(base);
                for i in 0..width_bytes {
                    value |= (*src.add(i) as u64) << (i * 8);
                }
            }
            let dst_alloc = find_operand_alloc(inst, inst_allocs, *dst, RaOperandKind::Def)?;
            write_allocation(regs, spills, dst_alloc, value);
        }
        LinearOp::SaveOutPtr { dst } => {
            let dst_alloc = find_operand_alloc(inst, inst_allocs, *dst, RaOperandKind::Def)?;
            write_allocation(regs, spills, dst_alloc, sim.out_ptr as u64);
        }
        LinearOp::SetOutPtr { src } => {
            let src_alloc = find_operand_alloc(inst, inst_allocs, *src, RaOperandKind::Use)?;
            sim.out_ptr = read_allocation(regs, spills, src_alloc) as *mut u8;
        }
        LinearOp::SlotAddr { dst, slot } => {
            let dst_alloc = find_operand_alloc(inst, inst_allocs, *dst, RaOperandKind::Def)?;
            let addr = sim.slot_addr_value(slot.index());
            write_allocation(regs, spills, dst_alloc, addr);
        }
        LinearOp::WriteToSlot { slot, src } => {
            let src_alloc = find_operand_alloc(inst, inst_allocs, *src, RaOperandKind::Use)?;
            let value = read_allocation(regs, spills, src_alloc);
            let slot = slot.index();
            sim.ensure_slot(slot);
            sim.slots[slot] = value;
        }
        LinearOp::ReadFromSlot { dst, slot } => {
            let slot = slot.index();
            sim.ensure_slot(slot);
            let value = sim.slots[slot];
            let dst_alloc = find_operand_alloc(inst, inst_allocs, *dst, RaOperandKind::Def)?;
            write_allocation(regs, spills, dst_alloc, value);
        }
        LinearOp::CallIntrinsic {
            func,
            args,
            dst,
            field_offset,
        } => {
            let mut args_values = Vec::with_capacity(args.len());
            for arg in args {
                let arg_alloc = find_operand_alloc(inst, inst_allocs, *arg, RaOperandKind::Use)?;
                args_values.push(read_allocation(regs, spills, arg_alloc));
            }
            let out_ptr = if dst.is_none() {
                let offset = *field_offset as usize;
                sim.ensure_output_range_for_out_ptr(output, offset, 64);
                Some(unsafe { sim.out_ptr.add(offset) })
            } else {
                None
            };
            let ret = run_call_intrinsic(sim, cursor, trap, func.0, &args_values, out_ptr);
            if let Some(dst) = dst {
                let dst_alloc = find_operand_alloc(inst, inst_allocs, *dst, RaOperandKind::Def)?;
                write_allocation(regs, spills, dst_alloc, ret);
            }
        }
        LinearOp::CallPure { func, args, dst } => {
            let mut args_values = Vec::with_capacity(args.len());
            for arg in args {
                let arg_alloc = find_operand_alloc(inst, inst_allocs, *arg, RaOperandKind::Use)?;
                args_values.push(read_allocation(regs, spills, arg_alloc));
            }
            let ret = run_call_pure(func.0, &args_values);
            let dst_alloc = find_operand_alloc(inst, inst_allocs, *dst, RaOperandKind::Def)?;
            write_allocation(regs, spills, dst_alloc, ret);
        }
        LinearOp::ErrorExit { code } => {
            set_trap_if_none(trap, *cursor, *code);
        }
        other => {
            return Err(RegallocEngineError::Simulation(format!(
                "unsupported op in simulation at b{}: {other:?}",
                block_id.0
            )));
        }
    }
    Ok(())
}

fn infer_output_size_for_cfg_function(func: &cfg_mir::Function) -> usize {
    func.insts
        .iter()
        .filter_map(|inst| match &inst.op {
            LinearOp::WriteToField { offset, width, .. }
            | LinearOp::ReadFromField { offset, width, .. } => {
                Some(*offset as usize + width.bytes() as usize)
            }
            _ => None,
        })
        .max()
        .unwrap_or(0)
}

fn infer_output_size_for_cfg_program(program: &cfg_mir::Program) -> usize {
    program
        .funcs
        .iter()
        .map(infer_output_size_for_cfg_function)
        .max()
        .unwrap_or(0)
}

fn edit_pos_key(pos: regalloc2::InstPosition) -> u8 {
    match pos {
        regalloc2::InstPosition::Before => 0,
        regalloc2::InstPosition::After => 1,
    }
}

fn find_cfg_operand_alloc_for_kind(
    alloc_func: &AllocatedCfgFunction,
    op_id: cfg_mir::OpId,
    vreg: kajit_ir::VReg,
    kind: RaOperandKind,
) -> Result<Allocation, RegallocEngineError> {
    let Some(operands) = alloc_func.op_operands.get(&op_id) else {
        return Err(RegallocEngineError::Simulation(format!(
            "missing operand metadata for cfg op {:?}",
            op_id
        )));
    };
    let Some(allocs) = alloc_func.op_allocs.get(&op_id) else {
        return Err(RegallocEngineError::Simulation(format!(
            "missing allocation metadata for cfg op {:?}",
            op_id
        )));
    };
    for ((operand_vreg, operand_kind), alloc) in operands.iter().zip(allocs.iter().copied()) {
        if *operand_vreg == vreg && *operand_kind == kind {
            return Ok(alloc);
        }
    }
    Err(RegallocEngineError::Simulation(format!(
        "missing allocation for {:?} operand v{} in cfg op {:?}",
        kind,
        vreg.index(),
        op_id
    )))
}

pub fn simulate_execution_cfg(
    allocated: &AllocatedCfgProgram,
    input: &[u8],
) -> Result<ExecutionResult, RegallocEngineError> {
    let func = allocated.cfg_program.funcs.first().ok_or_else(|| {
        RegallocEngineError::Simulation("CFG MIR program has no functions".to_string())
    })?;
    let alloc_func = allocated
        .functions
        .iter()
        .find(|f| f.lambda_id == func.lambda_id)
        .or_else(|| allocated.functions.first())
        .ok_or_else(|| {
            RegallocEngineError::Simulation("allocated cfg program has no functions".to_string())
        })?;
    let schedule = func
        .derive_schedule()
        .map_err(|err| RegallocEngineError::Simulation(err.to_string()))?;

    let mut block_indices = HashMap::with_capacity(func.blocks.len());
    for (idx, block) in func.blocks.iter().enumerate() {
        block_indices.insert(block.id, idx);
    }

    let mut edits_by_progpoint =
        HashMap::<(cfg_mir::OpId, u8), Vec<(Allocation, Allocation)>>::new();
    for (point, edit) in &alloc_func.edits {
        let Edit::Move { from, to } = edit;
        let op_id = match point {
            cfg_mir::ProgPoint::Before(op) | cfg_mir::ProgPoint::After(op) => *op,
            cfg_mir::ProgPoint::Edge(_) => continue,
        };
        edits_by_progpoint
            .entry((
                op_id,
                edit_pos_key(match point {
                    cfg_mir::ProgPoint::Before(_) => regalloc2::InstPosition::Before,
                    cfg_mir::ProgPoint::After(_) => regalloc2::InstPosition::After,
                    cfg_mir::ProgPoint::Edge(_) => regalloc2::InstPosition::Before,
                }),
            ))
            .or_default()
            .push((*from, *to));
    }

    let mut edge_edits = HashMap::<(cfg_mir::EdgeId, u8), Vec<(Allocation, Allocation)>>::new();
    for edge in &alloc_func.edge_edits {
        edge_edits
            .entry((edge.edge, edit_pos_key(edge.pos)))
            .or_default()
            .push((edge.from, edge.to));
    }

    let mut regs = HashMap::<PReg, u64>::new();
    let mut spills = HashMap::<usize, u64>::new();
    let mut output = vec![0u8; infer_output_size_for_cfg_program(&allocated.cfg_program)];
    let mut sim = SimulatorRuntime::new(input, allocated.cfg_program.slot_count as usize);
    sim.init_out_ptr(&mut output);
    let mut cursor = 0usize;
    let mut trap: Option<crate::InterpreterTrap> = None;
    let mut returned = false;
    let mut current = func.entry;
    let mut next_inst = 0usize;
    let mut steps = 0usize;

    while trap.is_none() && !returned {
        if steps >= MAX_SIM_STEPS {
            return Err(RegallocEngineError::Simulation(format!(
                "cfg simulation exceeded step limit ({MAX_SIM_STEPS})"
            )));
        }
        steps += 1;

        let block_idx = *block_indices.get(&current).ok_or_else(|| {
            RegallocEngineError::Simulation(format!("unknown cfg block b{}", current.0))
        })?;
        let block = &func.blocks[block_idx];

        if next_inst < block.insts.len() {
            let inst_id = block.insts[next_inst];
            let inst = &func.insts[inst_id.index()];
            let op_id = cfg_mir::OpId::Inst(inst_id);
            if let Some(edits) =
                edits_by_progpoint.get(&(op_id, edit_pos_key(regalloc2::InstPosition::Before)))
            {
                apply_moves(&mut regs, &mut spills, edits);
            }

            let linear_op_index = *schedule.op_to_index.get(&op_id).ok_or_else(|| {
                RegallocEngineError::Simulation(format!("schedule missing cfg inst {:?}", op_id))
            })? as usize;
            let inst_view = crate::RaInst {
                linear_op_index,
                op: inst.op.clone(),
                operands: inst.operands.iter().copied().map(map_cfg_operand).collect(),
                clobbers: map_cfg_clobbers(inst.clobbers),
            };
            let inst_allocs = alloc_func.op_allocs.get(&op_id).ok_or_else(|| {
                RegallocEngineError::Simulation(format!(
                    "missing allocation metadata for cfg op {:?}",
                    op_id
                ))
            })?;

            execute_sim_linear_op(
                &inst.op,
                crate::BlockId(block.id.0),
                &inst_view,
                inst_allocs,
                &mut regs,
                &mut spills,
                &mut sim,
                &mut cursor,
                &mut output,
                &mut trap,
            )?;

            if let Some(edits) =
                edits_by_progpoint.get(&(op_id, edit_pos_key(regalloc2::InstPosition::After)))
            {
                apply_moves(&mut regs, &mut spills, edits);
            }

            next_inst += 1;
            continue;
        }

        let term = func.term(block.term).ok_or_else(|| {
            RegallocEngineError::Simulation(format!("missing cfg term t{}", block.term.0))
        })?;
        let term_op = cfg_mir::OpId::Term(block.term);

        if let Some(edits) =
            edits_by_progpoint.get(&(term_op, edit_pos_key(regalloc2::InstPosition::Before)))
        {
            apply_moves(&mut regs, &mut spills, edits);
        }

        match term {
            cfg_mir::Terminator::Return => {
                returned = true;
            }
            cfg_mir::Terminator::ErrorExit { code } => {
                trap = Some(crate::InterpreterTrap {
                    code: *code,
                    offset: cursor as u32,
                });
            }
            cfg_mir::Terminator::Branch { edge } => {
                if let Some(edits) =
                    edge_edits.get(&(*edge, edit_pos_key(regalloc2::InstPosition::Before)))
                {
                    apply_moves(&mut regs, &mut spills, edits);
                }
                if let Some(edits) =
                    edge_edits.get(&(*edge, edit_pos_key(regalloc2::InstPosition::After)))
                {
                    apply_moves(&mut regs, &mut spills, edits);
                }
                current = func.edges[edge.index()].to;
                next_inst = 0;
            }
            cfg_mir::Terminator::BranchIf {
                cond,
                taken,
                fallthrough,
            } => {
                let cond_alloc = find_cfg_operand_alloc_for_kind(
                    alloc_func,
                    term_op,
                    *cond,
                    RaOperandKind::Use,
                )?;
                let cond_value = read_allocation(&regs, &spills, cond_alloc);
                let chosen = if cond_value != 0 {
                    *taken
                } else {
                    *fallthrough
                };
                if let Some(edits) =
                    edge_edits.get(&(chosen, edit_pos_key(regalloc2::InstPosition::Before)))
                {
                    apply_moves(&mut regs, &mut spills, edits);
                }
                if let Some(edits) =
                    edge_edits.get(&(chosen, edit_pos_key(regalloc2::InstPosition::After)))
                {
                    apply_moves(&mut regs, &mut spills, edits);
                }
                current = func.edges[chosen.index()].to;
                next_inst = 0;
            }
            cfg_mir::Terminator::BranchIfZero {
                cond,
                taken,
                fallthrough,
            } => {
                let cond_alloc = find_cfg_operand_alloc_for_kind(
                    alloc_func,
                    term_op,
                    *cond,
                    RaOperandKind::Use,
                )?;
                let cond_value = read_allocation(&regs, &spills, cond_alloc);
                let chosen = if cond_value == 0 {
                    *taken
                } else {
                    *fallthrough
                };
                if let Some(edits) =
                    edge_edits.get(&(chosen, edit_pos_key(regalloc2::InstPosition::Before)))
                {
                    apply_moves(&mut regs, &mut spills, edits);
                }
                if let Some(edits) =
                    edge_edits.get(&(chosen, edit_pos_key(regalloc2::InstPosition::After)))
                {
                    apply_moves(&mut regs, &mut spills, edits);
                }
                current = func.edges[chosen.index()].to;
                next_inst = 0;
            }
            cfg_mir::Terminator::JumpTable {
                predicate,
                targets,
                default,
            } => {
                let pred_alloc = find_cfg_operand_alloc_for_kind(
                    alloc_func,
                    term_op,
                    *predicate,
                    RaOperandKind::Use,
                )?;
                let pred = read_allocation(&regs, &spills, pred_alloc) as usize;
                let chosen = targets.get(pred).copied().unwrap_or(*default);
                if let Some(edits) =
                    edge_edits.get(&(chosen, edit_pos_key(regalloc2::InstPosition::Before)))
                {
                    apply_moves(&mut regs, &mut spills, edits);
                }
                if let Some(edits) =
                    edge_edits.get(&(chosen, edit_pos_key(regalloc2::InstPosition::After)))
                {
                    apply_moves(&mut regs, &mut spills, edits);
                }
                current = func.edges[chosen.index()].to;
                next_inst = 0;
            }
        }

        if let Some(edits) =
            edits_by_progpoint.get(&(term_op, edit_pos_key(regalloc2::InstPosition::After)))
        {
            apply_moves(&mut regs, &mut spills, edits);
        }
    }

    let current_block_idx = *block_indices.get(&current).ok_or_else(|| {
        RegallocEngineError::Simulation(format!("unknown final cfg block b{}", current.0))
    })?;
    let current_block = &func.blocks[current_block_idx];

    Ok(ExecutionResult {
        state: ExecutionState {
            physical_registers: regs,
            spillslots: spills,
            position: ExecutionPosition {
                block: crate::BlockId(current_block.id.0),
                next_inst_index: next_inst,
                at_terminator: next_inst >= current_block.insts.len(),
            },
        },
        output,
        cursor,
        trap,
        returned,
    })
}

#[allow(clippy::too_many_arguments)]
fn simulate_execution_trace_cfg_function(
    allocated: &AllocatedCfgProgram,
    lambda_id: LambdaId,
    args: &[u64],
    sim: &mut SimulatorRuntime,
    cursor: &mut usize,
    output: &mut Vec<u8>,
    trap: &mut Option<crate::InterpreterTrap>,
    trace: &mut Vec<ExecutionTraceEntry>,
    steps: &mut usize,
) -> Result<Vec<u64>, RegallocEngineError> {
    let func = allocated
        .cfg_program
        .funcs
        .iter()
        .find(|f| f.lambda_id == lambda_id)
        .ok_or_else(|| {
            RegallocEngineError::Simulation(format!(
                "CFG MIR function for lambda @{} not found",
                lambda_id.index()
            ))
        })?;
    let alloc_func = allocated
        .functions
        .iter()
        .find(|f| f.lambda_id == lambda_id)
        .ok_or_else(|| {
            RegallocEngineError::Simulation(format!(
                "allocated cfg function for lambda @{} not found",
                lambda_id.index()
            ))
        })?;
    let schedule = func
        .derive_schedule()
        .map_err(|err| RegallocEngineError::Simulation(err.to_string()))?;

    let mut block_indices = HashMap::with_capacity(func.blocks.len());
    for (idx, block) in func.blocks.iter().enumerate() {
        block_indices.insert(block.id, idx);
    }

    let mut edits_by_progpoint =
        HashMap::<(cfg_mir::OpId, u8), Vec<(Allocation, Allocation)>>::new();
    for (point, edit) in &alloc_func.edits {
        let Edit::Move { from, to } = edit;
        let op_id = match point {
            cfg_mir::ProgPoint::Before(op) | cfg_mir::ProgPoint::After(op) => *op,
            cfg_mir::ProgPoint::Edge(_) => continue,
        };
        edits_by_progpoint
            .entry((
                op_id,
                edit_pos_key(match point {
                    cfg_mir::ProgPoint::Before(_) => regalloc2::InstPosition::Before,
                    cfg_mir::ProgPoint::After(_) => regalloc2::InstPosition::After,
                    cfg_mir::ProgPoint::Edge(_) => regalloc2::InstPosition::Before,
                }),
            ))
            .or_default()
            .push((*from, *to));
    }

    let mut edge_edits = HashMap::<(cfg_mir::EdgeId, u8), Vec<(Allocation, Allocation)>>::new();
    for edge in &alloc_func.edge_edits {
        edge_edits
            .entry((edge.edge, edit_pos_key(edge.pos)))
            .or_default()
            .push((edge.from, edge.to));
    }

    let mut regs = HashMap::<PReg, u64>::new();
    let mut spills = HashMap::<usize, u64>::new();
    for (arg_index, arg_value) in args.iter().copied().enumerate() {
        let preg = abi_arg_int(arg_index + 2).ok_or_else(|| {
            RegallocEngineError::Simulation(format!(
                "lambda @{} arg {} exceeds ABI register capacity",
                lambda_id.index(),
                arg_index
            ))
        })?;
        regs.insert(preg, arg_value);
    }

    let mut returned = false;
    let mut current = func.entry;
    let mut next_inst = 0usize;

    while trap.is_none() && !returned {
        if *steps >= MAX_SIM_STEPS {
            return Err(RegallocEngineError::Simulation(format!(
                "cfg simulation exceeded step limit ({MAX_SIM_STEPS})"
            )));
        }
        *steps += 1;
        let step_index = *steps;

        let block_idx = *block_indices.get(&current).ok_or_else(|| {
            RegallocEngineError::Simulation(format!("unknown cfg block b{}", current.0))
        })?;
        let block = &func.blocks[block_idx];

        if next_inst < block.insts.len() {
            let inst_id = block.insts[next_inst];
            let inst = &func.insts[inst_id.index()];
            let op_id = cfg_mir::OpId::Inst(inst_id);
            if let Some(edits) =
                edits_by_progpoint.get(&(op_id, edit_pos_key(regalloc2::InstPosition::Before)))
            {
                apply_moves(&mut regs, &mut spills, edits);
            }

            let linear_op_index = *schedule.op_to_index.get(&op_id).ok_or_else(|| {
                RegallocEngineError::Simulation(format!("schedule missing cfg inst {:?}", op_id))
            })? as usize;
            let inst_view = crate::RaInst {
                linear_op_index,
                op: inst.op.clone(),
                operands: inst.operands.iter().copied().map(map_cfg_operand).collect(),
                clobbers: map_cfg_clobbers(inst.clobbers),
            };
            let inst_allocs = alloc_func.op_allocs.get(&op_id).ok_or_else(|| {
                RegallocEngineError::Simulation(format!(
                    "missing allocation metadata for cfg op {:?}",
                    op_id
                ))
            })?;

            if let LinearOp::CallLambda {
                target,
                args,
                results,
            } = &inst.op
            {
                let mut call_args = Vec::with_capacity(args.len());
                for arg in args {
                    let arg_alloc =
                        find_operand_alloc(&inst_view, inst_allocs, *arg, RaOperandKind::Use)?;
                    call_args.push(read_allocation(&regs, &spills, arg_alloc));
                }

                let caller_out_ptr = sim.out_ptr;
                let call_results = simulate_execution_trace_cfg_function(
                    allocated, *target, &call_args, sim, cursor, output, trap, trace, steps,
                )?;
                sim.out_ptr = caller_out_ptr;

                if trap.is_none() {
                    if call_results.len() != results.len() {
                        return Err(RegallocEngineError::Simulation(format!(
                            "CallLambda @{} result arity mismatch: expected {}, got {}",
                            target.index(),
                            results.len(),
                            call_results.len()
                        )));
                    }
                    for (dst, value) in results.iter().zip(call_results.into_iter()) {
                        let dst_alloc =
                            find_operand_alloc(&inst_view, inst_allocs, *dst, RaOperandKind::Def)?;
                        write_allocation(&mut regs, &mut spills, dst_alloc, value);
                    }
                }
            } else {
                execute_sim_linear_op(
                    &inst.op,
                    crate::BlockId(block.id.0),
                    &inst_view,
                    inst_allocs,
                    &mut regs,
                    &mut spills,
                    sim,
                    cursor,
                    output,
                    trap,
                )?;
            }

            if let Some(edits) =
                edits_by_progpoint.get(&(op_id, edit_pos_key(regalloc2::InstPosition::After)))
            {
                apply_moves(&mut regs, &mut spills, edits);
            }

            next_inst += 1;
            trace.push(ExecutionTraceEntry {
                step_index,
                state: ExecutionState {
                    physical_registers: regs.clone(),
                    spillslots: spills.clone(),
                    position: ExecutionPosition {
                        block: crate::BlockId(block.id.0),
                        next_inst_index: next_inst,
                        at_terminator: next_inst >= block.insts.len(),
                    },
                },
                output: output.clone(),
                cursor: *cursor,
                trap: *trap,
                returned,
            });
            continue;
        }

        let term = func.term(block.term).ok_or_else(|| {
            RegallocEngineError::Simulation(format!("missing cfg term t{}", block.term.0))
        })?;
        let term_op = cfg_mir::OpId::Term(block.term);

        if let Some(edits) =
            edits_by_progpoint.get(&(term_op, edit_pos_key(regalloc2::InstPosition::Before)))
        {
            apply_moves(&mut regs, &mut spills, edits);
        }

        match term {
            cfg_mir::Terminator::Return => {
                returned = true;
            }
            cfg_mir::Terminator::ErrorExit { code } => {
                *trap = Some(crate::InterpreterTrap {
                    code: *code,
                    offset: *cursor as u32,
                });
            }
            cfg_mir::Terminator::Branch { edge } => {
                if let Some(edits) =
                    edge_edits.get(&(*edge, edit_pos_key(regalloc2::InstPosition::Before)))
                {
                    apply_moves(&mut regs, &mut spills, edits);
                }
                if let Some(edits) =
                    edge_edits.get(&(*edge, edit_pos_key(regalloc2::InstPosition::After)))
                {
                    apply_moves(&mut regs, &mut spills, edits);
                }
                current = func.edges[edge.index()].to;
                next_inst = 0;
            }
            cfg_mir::Terminator::BranchIf {
                cond,
                taken,
                fallthrough,
            } => {
                let cond_alloc = find_cfg_operand_alloc_for_kind(
                    alloc_func,
                    term_op,
                    *cond,
                    RaOperandKind::Use,
                )?;
                let cond_value = read_allocation(&regs, &spills, cond_alloc);
                let chosen = if cond_value != 0 {
                    *taken
                } else {
                    *fallthrough
                };
                if let Some(edits) =
                    edge_edits.get(&(chosen, edit_pos_key(regalloc2::InstPosition::Before)))
                {
                    apply_moves(&mut regs, &mut spills, edits);
                }
                if let Some(edits) =
                    edge_edits.get(&(chosen, edit_pos_key(regalloc2::InstPosition::After)))
                {
                    apply_moves(&mut regs, &mut spills, edits);
                }
                current = func.edges[chosen.index()].to;
                next_inst = 0;
            }
            cfg_mir::Terminator::BranchIfZero {
                cond,
                taken,
                fallthrough,
            } => {
                let cond_alloc = find_cfg_operand_alloc_for_kind(
                    alloc_func,
                    term_op,
                    *cond,
                    RaOperandKind::Use,
                )?;
                let cond_value = read_allocation(&regs, &spills, cond_alloc);
                let chosen = if cond_value == 0 {
                    *taken
                } else {
                    *fallthrough
                };
                if let Some(edits) =
                    edge_edits.get(&(chosen, edit_pos_key(regalloc2::InstPosition::Before)))
                {
                    apply_moves(&mut regs, &mut spills, edits);
                }
                if let Some(edits) =
                    edge_edits.get(&(chosen, edit_pos_key(regalloc2::InstPosition::After)))
                {
                    apply_moves(&mut regs, &mut spills, edits);
                }
                current = func.edges[chosen.index()].to;
                next_inst = 0;
            }
            cfg_mir::Terminator::JumpTable {
                predicate,
                targets,
                default,
            } => {
                let pred_alloc = find_cfg_operand_alloc_for_kind(
                    alloc_func,
                    term_op,
                    *predicate,
                    RaOperandKind::Use,
                )?;
                let pred = read_allocation(&regs, &spills, pred_alloc) as usize;
                let chosen = targets.get(pred).copied().unwrap_or(*default);
                if let Some(edits) =
                    edge_edits.get(&(chosen, edit_pos_key(regalloc2::InstPosition::Before)))
                {
                    apply_moves(&mut regs, &mut spills, edits);
                }
                if let Some(edits) =
                    edge_edits.get(&(chosen, edit_pos_key(regalloc2::InstPosition::After)))
                {
                    apply_moves(&mut regs, &mut spills, edits);
                }
                current = func.edges[chosen.index()].to;
                next_inst = 0;
            }
        }

        if let Some(edits) =
            edits_by_progpoint.get(&(term_op, edit_pos_key(regalloc2::InstPosition::After)))
        {
            apply_moves(&mut regs, &mut spills, edits);
        }

        let current_block_idx = *block_indices.get(&current).ok_or_else(|| {
            RegallocEngineError::Simulation(format!("unknown cfg block b{}", current.0))
        })?;
        let current_block = &func.blocks[current_block_idx];
        trace.push(ExecutionTraceEntry {
            step_index,
            state: ExecutionState {
                physical_registers: regs.clone(),
                spillslots: spills.clone(),
                position: ExecutionPosition {
                    block: crate::BlockId(current_block.id.0),
                    next_inst_index: next_inst,
                    at_terminator: next_inst >= current_block.insts.len(),
                },
            },
            output: output.clone(),
            cursor: *cursor,
            trap: *trap,
            returned,
        });
    }

    if trap.is_some() {
        return Ok(Vec::new());
    }

    Ok(alloc_func
        .return_result_allocs
        .iter()
        .map(|alloc| read_allocation(&regs, &spills, *alloc))
        .collect())
}

pub fn simulate_execution_trace_cfg(
    allocated: &AllocatedCfgProgram,
    input: &[u8],
) -> Result<Vec<ExecutionTraceEntry>, RegallocEngineError> {
    let root = allocated.cfg_program.funcs.first().ok_or_else(|| {
        RegallocEngineError::Simulation("CFG MIR program has no functions".to_string())
    })?;
    let mut output = vec![0u8; infer_output_size_for_cfg_program(&allocated.cfg_program)];
    let mut sim = SimulatorRuntime::new(input, allocated.cfg_program.slot_count as usize);
    sim.init_out_ptr(&mut output);
    let mut cursor = 0usize;
    let mut trap: Option<crate::InterpreterTrap> = None;
    let mut trace = Vec::<ExecutionTraceEntry>::new();
    let mut steps = 0usize;

    let _ = simulate_execution_trace_cfg_function(
        allocated,
        root.lambda_id,
        &[],
        &mut sim,
        &mut cursor,
        &mut output,
        &mut trap,
        &mut trace,
        &mut steps,
    )?;

    Ok(trace)
}

fn ideal_trace(
    program: &cfg_mir::Program,
    input: &[u8],
) -> Result<Vec<DifferentialState>, RegallocEngineError> {
    let (_, entries) = crate::execute_program_with_trace(program, input).map_err(|err| {
        RegallocEngineError::Simulation(format!("ideal interpreter failed: {err}"))
    })?;

    Ok(entries
        .into_iter()
        .map(|entry| DifferentialState {
            step_index: entry.step_index,
            position: ExecutionPosition {
                block: crate::BlockId(entry.block.0),
                next_inst_index: entry.next_inst_index,
                at_terminator: entry.at_terminator,
            },
            cursor: entry.cursor,
            output: entry.output,
            trap: entry.trap,
            returned: entry.returned,
            vregs: Some(entry.vregs),
            physical_registers: None,
            spillslots: None,
        })
        .collect())
}

fn first_differing_field(
    ideal: &DifferentialState,
    post: &DifferentialState,
) -> Option<&'static str> {
    if ideal.position != post.position {
        return Some("position");
    }
    if ideal.cursor != post.cursor {
        return Some("cursor");
    }
    if ideal.trap != post.trap {
        return Some("trap");
    }
    if ideal.returned != post.returned {
        return Some("returned");
    }
    if std::env::var_os("KAJIT_DIFF_IGNORE_OUTPUT").is_none() && ideal.output != post.output {
        return Some("output");
    }
    None
}

fn differential_state_from_trace_entry(entry: ExecutionTraceEntry) -> DifferentialState {
    DifferentialState {
        step_index: entry.step_index,
        position: entry.state.position,
        cursor: entry.cursor,
        output: entry.output,
        trap: entry.trap,
        returned: entry.returned,
        vregs: None,
        physical_registers: Some(entry.state.physical_registers),
        spillslots: Some(entry.state.spillslots),
    }
}

fn compare_differential_states(
    ideal: Vec<DifferentialState>,
    post: Vec<DifferentialState>,
) -> DifferentialCheckResult {
    let shared = ideal.len().min(post.len());
    for i in 0..shared {
        if let Some(field) = first_differing_field(&ideal[i], &post[i]) {
            return DifferentialCheckResult::Diverged(DifferentialDivergence {
                step_index: ideal[i].step_index.min(post[i].step_index),
                field: field.to_string(),
                ideal: ideal[i].clone(),
                post_regalloc: post[i].clone(),
            });
        }
    }

    if ideal.len() != post.len() {
        let ideal_final = ideal.last().cloned().unwrap_or(DifferentialState {
            step_index: 0,
            position: ExecutionPosition {
                block: crate::BlockId(0),
                next_inst_index: 0,
                at_terminator: true,
            },
            cursor: 0,
            output: Vec::new(),
            trap: None,
            returned: false,
            vregs: Some(Vec::new()),
            physical_registers: None,
            spillslots: None,
        });
        let post_final = post.last().cloned().unwrap_or(DifferentialState {
            step_index: 0,
            position: ExecutionPosition {
                block: crate::BlockId(0),
                next_inst_index: 0,
                at_terminator: true,
            },
            cursor: 0,
            output: Vec::new(),
            trap: None,
            returned: false,
            vregs: None,
            physical_registers: Some(HashMap::new()),
            spillslots: Some(HashMap::new()),
        });
        return DifferentialCheckResult::Diverged(DifferentialDivergence {
            step_index: shared,
            field: "step_count".to_string(),
            ideal: ideal_final,
            post_regalloc: post_final,
        });
    }

    let ideal_final = ideal.last().cloned().unwrap_or(DifferentialState {
        step_index: 0,
        position: ExecutionPosition {
            block: crate::BlockId(0),
            next_inst_index: 0,
            at_terminator: true,
        },
        cursor: 0,
        output: Vec::new(),
        trap: None,
        returned: false,
        vregs: Some(Vec::new()),
        physical_registers: None,
        spillslots: None,
    });
    let post_final = post.last().cloned().unwrap_or(DifferentialState {
        step_index: 0,
        position: ExecutionPosition {
            block: crate::BlockId(0),
            next_inst_index: 0,
            at_terminator: true,
        },
        cursor: 0,
        output: Vec::new(),
        trap: None,
        returned: false,
        vregs: None,
        physical_registers: Some(HashMap::new()),
        spillslots: Some(HashMap::new()),
    });

    if let Some(field) = first_differing_field(&ideal_final, &post_final) {
        return DifferentialCheckResult::Diverged(DifferentialDivergence {
            step_index: ideal_final.step_index.min(post_final.step_index),
            field: field.to_string(),
            ideal: ideal_final,
            post_regalloc: post_final,
        });
    }

    DifferentialCheckResult::Match {
        steps: shared,
        ideal_final,
        post_regalloc_final: post_final,
    }
}

pub fn differential_check_cfg(
    ideal_program: &cfg_mir::Program,
    allocated: &AllocatedCfgProgram,
    input: &[u8],
) -> DifferentialCheckResult {
    let ideal = match ideal_trace(ideal_program, input) {
        Ok(v) => v,
        Err(err) => return DifferentialCheckResult::Error(err.to_string()),
    };
    let post_trace = match simulate_execution_trace_cfg(allocated, input) {
        Ok(v) => v,
        Err(err) => return DifferentialCheckResult::Error(err.to_string()),
    };
    let post = post_trace
        .into_iter()
        .map(differential_state_from_trace_entry)
        .collect::<Vec<_>>();
    compare_differential_states(ideal, post)
}

#[cfg(test)]
mod tests {
    use super::*;
    use facet::Facet;
    use kajit_ir::{IntrinsicFn, IntrinsicRegistry, IrBuilder, IrOp, Width, run_default_passes};
    use kajit_ir_text::parse_ir;
    use kajit_lir::linearize;

    fn build_stress_ir() -> kajit_ir::IrFunc {
        let mut builder = IrBuilder::new(<u32 as facet::Facet>::SHAPE);
        {
            let mut rb = builder.root_region();
            let mut acc = rb.const_val(0_u64);
            for i in 1_u64..=128_u64 {
                let c = rb.const_val(i);
                acc = rb.binop(IrOp::Add, acc, c);
            }
            rb.write_to_field(acc, 0, Width::W4);
            rb.set_results(&[]);
        }
        builder.finish()
    }

    #[test]
    fn regalloc2_allocates_cfg_mir_program() {
        let mut func = build_stress_ir();
        let lin = linearize(&mut func);
        let cfg = crate::cfg_mir::lower_linear_ir(&lin);
        let alloc = allocate_cfg_program(&cfg).expect("regalloc2 should allocate cfg_mir");

        assert_eq!(alloc.functions.len(), cfg.funcs.len());
        assert!(
            alloc.functions.iter().all(|f| !f.op_allocs.is_empty()),
            "expected op allocation maps for cfg_mir functions"
        );
        assert!(
            alloc.functions.iter().all(|f| !f.op_operands.is_empty()),
            "expected op operand maps for cfg_mir functions"
        );
    }

    #[test]
    fn regalloc2_allocates_cfg_mir_gamma_theta_program() {
        let mut builder = IrBuilder::new(<u32 as facet::Facet>::SHAPE);
        {
            let mut rb = builder.root_region();
            let pred = rb.const_val(1);
            let init_count = rb.const_val(5);
            let one = rb.const_val(1);

            let gamma_out = rb.gamma(pred, &[init_count], 2, |branch_idx, bb| {
                let args = bb.region_args(1);
                let val = if branch_idx == 0 {
                    args[0]
                } else {
                    bb.binop(IrOp::Add, args[0], one)
                };
                bb.set_results(&[val]);
            });
            let _ = rb.theta(&[gamma_out[0], one], |bb| {
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
        let cfg = crate::cfg_mir::lower_linear_ir(&lin);
        let alloc =
            allocate_cfg_program(&cfg).expect("regalloc2 should allocate cfg_mir gamma/theta");

        assert_eq!(alloc.functions.len(), 1);
        assert!(!alloc.functions[0].op_allocs.is_empty());
    }

    #[test]
    fn cfg_mir_typed_edits_reference_valid_program_ids() {
        let mut func = build_stress_ir();
        let lin = linearize(&mut func);
        let cfg = crate::cfg_mir::lower_linear_ir(&lin);
        let alloc = allocate_cfg_program(&cfg).expect("regalloc2 should allocate cfg_mir");

        for (func_alloc, cfg_func) in alloc.functions.iter().zip(cfg.funcs.iter()) {
            for (point, _) in &func_alloc.edits {
                let op_id = match point {
                    crate::cfg_mir::ProgPoint::Before(op)
                    | crate::cfg_mir::ProgPoint::After(op) => *op,
                    crate::cfg_mir::ProgPoint::Edge(_) => {
                        panic!("point edits must not use edge program points")
                    }
                };
                assert!(
                    func_alloc.op_allocs.contains_key(&op_id),
                    "typed edit references op_id not present in op_allocs: {:?}",
                    op_id
                );
            }

            for edge_edit in &func_alloc.edge_edits {
                assert!(
                    cfg_func.edges.get(edge_edit.edge.index()).is_some(),
                    "typed edge edit references missing edge: e{}",
                    edge_edit.edge.0
                );
            }
        }
    }

    // r[verify ir.regalloc.engine]
    #[test]
    fn regalloc2_allocates_gamma_and_theta_programs() {
        let mut builder = IrBuilder::new(<u32 as facet::Facet>::SHAPE);
        {
            let mut rb = builder.root_region();
            let pred = rb.const_val(1);
            let init_count = rb.const_val(5);
            let one = rb.const_val(1);

            let gamma_out = rb.gamma(pred, &[init_count], 2, |branch_idx, bb| {
                let args = bb.region_args(1);
                let val = if branch_idx == 0 {
                    args[0]
                } else {
                    bb.binop(IrOp::Add, args[0], one)
                };
                bb.set_results(&[val]);
            });
            let _ = rb.theta(&[gamma_out[0], one], |bb| {
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
        let cfg = crate::cfg_mir::lower_linear_ir(&lin);
        let alloc = allocate_cfg_program(&cfg).expect("regalloc2 should allocate gamma/theta");
        assert!(!alloc.functions.is_empty());
    }

    #[test]
    fn regalloc2_allocates_textual_theta_invariant_fixture() {
        let input = r#"
lambda @0 (shape: "u8") {
  region {
    args: [%cs, %os]
    n0 = Const(0x4) [] -> [v0]
    n1 = Const(0x1) [] -> [v1]
    n2 = theta [v0, v1, %cs:arg, %os:arg] {
      region {
        args: [arg0, arg1, %cs, %os]
        n3 = Const(0x7) [] -> [v2]
        n4 = Const(0x3) [] -> [v3]
        n5 = Add [v2, v3] -> [v4]
        n6 = Xor [v4, v3] -> [v5]
        n7 = Sub [arg0, arg1] -> [v6]
        n8 = Add [v5, v6] -> [v7]
        results: [v6, v6, arg1, %cs:arg, %os:arg]
      }
    } -> [v8, v9, %cs, %os]
    n9 = WriteToField(offset=0, W1) [v8, %os:n2] -> [%os]
    results: [%cs:n2, %os:n9]
  }
}
"#;

        let registry = IntrinsicRegistry::empty();
        let mut func =
            parse_ir(input, <u8 as Facet>::SHAPE, &registry).expect("fixture should parse");
        let lin = linearize(&mut func);
        let cfg = crate::cfg_mir::lower_linear_ir(&lin);
        let _alloc = allocate_cfg_program(&cfg).unwrap_or_else(|e| {
            panic!(
                "regalloc should allocate textual theta fixture: {e}\n--- linear ---\n{lin}\n--- cfg-mir ---\n{cfg}"
            )
        });
    }

    #[test]
    fn regalloc2_allocates_textual_theta_invariant_fixture_after_passes() {
        let input = r#"
lambda @0 (shape: "u8") {
  region {
    args: [%cs, %os]
    n0 = Const(0x4) [] -> [v0]
    n1 = Const(0x1) [] -> [v1]
    n2 = theta [v0, v1, %cs:arg, %os:arg] {
      region {
        args: [arg0, arg1, %cs, %os]
        n3 = Const(0x7) [] -> [v2]
        n4 = Const(0x3) [] -> [v3]
        n5 = Add [v2, v3] -> [v4]
        n6 = Xor [v4, v3] -> [v5]
        n7 = Sub [arg0, arg1] -> [v6]
        n8 = Add [v5, v6] -> [v7]
        results: [v6, v6, arg1, %cs:arg, %os:arg]
      }
    } -> [v8, v9, %cs, %os]
    n9 = WriteToField(offset=0, W1) [v8, %os:n2] -> [%os]
    results: [%cs:n2, %os:n9]
  }
}
"#;

        let registry = IntrinsicRegistry::empty();
        let mut func =
            parse_ir(input, <u8 as Facet>::SHAPE, &registry).expect("fixture should parse");
        run_default_passes(&mut func);
        let lin = linearize(&mut func);
        let cfg = crate::cfg_mir::lower_linear_ir(&lin);
        let _alloc = allocate_cfg_program(&cfg).unwrap_or_else(|e| {
            panic!(
                "regalloc should allocate textual theta fixture after passes: {e}\n--- linear ---\n{lin}\n--- cfg-mir ---\n{cfg}"
            )
        });
    }

    // r[verify ir.regalloc.engine]
    #[test]
    fn regalloc2_allocates_postcard_vec_decoder() {
        let mut func = build_stress_ir();
        run_default_passes(&mut func);
        let lin = linearize(&mut func);
        let cfg = crate::cfg_mir::lower_linear_ir(&lin);
        let alloc = allocate_cfg_program(&cfg).expect("regalloc2 should allocate stress IR");
        assert!(
            alloc.functions.iter().any(|f| f.lambda_id.index() == 0),
            "expected allocation output to include root lambda"
        );
        let total_inst_allocs: usize = alloc.functions.iter().map(|f| f.op_allocs.len()).sum();
        assert!(
            total_inst_allocs > 100,
            "expected sizeable stress IR lowering, got {} insts",
            total_inst_allocs
        );
    }

    // r[verify ir.regalloc.engine]
    #[test]
    fn regalloc2_covers_call_operands_and_clobbers() {
        unsafe extern "C" fn add3(_ctx: *mut core::ffi::c_void, a: u64, b: u64, c: u64) -> u64 {
            a + b + c
        }

        let mut builder = IrBuilder::new(<u64 as facet::Facet>::SHAPE);
        {
            let mut rb = builder.root_region();
            let a = rb.const_val(11);
            let b = rb.const_val(7);
            let c = rb.const_val(5);
            let out = rb
                .call_intrinsic(IntrinsicFn(add3 as *const () as usize), &[a, b, c], 0, true)
                .expect("intrinsic call should produce output");
            rb.write_to_field(out, 0, Width::W8);
            rb.set_results(&[]);
        }
        let mut func = builder.finish();
        let lin = linearize(&mut func);
        let cfg = crate::cfg_mir::lower_linear_ir(&lin);
        let alloc = allocate_cfg_program(&cfg).expect("regalloc2 should allocate call-heavy IR");
        assert!(!alloc.functions[0].op_allocs.is_empty());
    }

    #[test]
    fn regalloc2_postcard_vec_edit_budget_on_aarch64() {
        if !cfg!(target_arch = "aarch64") {
            return;
        }
        let mut func = build_stress_ir();
        run_default_passes(&mut func);
        let lin = linearize(&mut func);
        let cfg = crate::cfg_mir::lower_linear_ir(&lin);
        let alloc = allocate_cfg_program(&cfg).expect("regalloc2 should allocate stress IR");

        let total_edits: usize = alloc.functions.iter().map(|f| f.edits.len()).sum();
        assert!(
            total_edits <= 256,
            "expected stress IR path to stay within edit budget (<=256), got {}",
            total_edits
        );
    }

    // r[verify ir.regalloc.checker]
    #[test]
    fn regalloc_checker_runs_on_corrupted_mapping() {
        let mut func = build_stress_ir();
        run_default_passes(&mut func);
        let lin = linearize(&mut func);
        let cfg = crate::cfg_mir::lower_linear_ir(&lin);
        let env = machine_env();
        let options = RegallocOptions {
            verbose_log: false,
            validate_ssa: false,
            algorithm: regalloc2::Algorithm::Ion,
        };

        let adapter =
            AdapterFunction::from_cfg(&cfg.funcs[0], cfg.vreg_count as usize).expect("valid cfg");
        let mut out = regalloc2::run(&adapter, &env, &options).expect("allocation should succeed");
        if !out.allocs.is_empty() {
            // Corrupt the first allocation with the wrong register class on purpose.
            out.allocs[0] = Allocation::reg(preg_vec(0));
        }

        let mut checker = regalloc2::checker::Checker::new(&adapter, &env);
        checker.prepare(&out);
        let _ = checker.run();
    }

    fn synthetic_cfg_edge_fixture() -> AllocatedCfgProgram {
        let source_vreg = kajit_ir::VReg::new(10);
        let target_vreg = kajit_ir::VReg::new(11);
        let aux_vreg = kajit_ir::VReg::new(12);
        let inst_id = crate::cfg_mir::InstId::new(0);
        let term_branch = crate::cfg_mir::TermId::new(0);
        let term_ret = crate::cfg_mir::TermId::new(1);
        let edge_id = crate::cfg_mir::EdgeId::new(0);

        let cfg_func = crate::cfg_mir::Function {
            id: crate::cfg_mir::FunctionId::new(0),
            lambda_id: LambdaId::new(0),
            entry: crate::cfg_mir::BlockId::new(0),
            data_args: Vec::new(),
            data_results: Vec::new(),
            blocks: vec![
                crate::cfg_mir::Block {
                    id: crate::cfg_mir::BlockId::new(0),
                    params: Vec::new(),
                    insts: Vec::new(),
                    term: term_branch,
                    preds: Vec::new(),
                    succs: vec![edge_id],
                },
                crate::cfg_mir::Block {
                    id: crate::cfg_mir::BlockId::new(1),
                    params: vec![target_vreg],
                    insts: vec![inst_id],
                    term: term_ret,
                    preds: vec![edge_id],
                    succs: Vec::new(),
                },
            ],
            edges: vec![crate::cfg_mir::Edge {
                id: edge_id,
                from: crate::cfg_mir::BlockId::new(0),
                to: crate::cfg_mir::BlockId::new(1),
                args: vec![crate::cfg_mir::EdgeArg {
                    target: target_vreg,
                    source: source_vreg,
                }],
            }],
            insts: vec![crate::cfg_mir::Inst {
                id: inst_id,
                op: LinearOp::Const {
                    dst: aux_vreg,
                    value: 1,
                },
                operands: vec![crate::cfg_mir::Operand {
                    vreg: target_vreg,
                    kind: crate::cfg_mir::OperandKind::Use,
                    class: crate::cfg_mir::RegClass::Gpr,
                    fixed: None,
                }],
                clobbers: crate::cfg_mir::Clobbers::default(),
            }],
            terms: vec![
                crate::cfg_mir::Terminator::Branch { edge: edge_id },
                crate::cfg_mir::Terminator::Return,
            ],
        };

        let mut op_allocs = HashMap::<crate::cfg_mir::OpId, Vec<Allocation>>::new();
        let mut op_operands =
            HashMap::<crate::cfg_mir::OpId, Vec<(kajit_ir::VReg, crate::OperandKind)>>::new();
        op_allocs.insert(
            crate::cfg_mir::OpId::Term(term_branch),
            vec![Allocation::reg(preg_int(0))],
        );
        op_operands.insert(
            crate::cfg_mir::OpId::Term(term_branch),
            vec![(source_vreg, crate::OperandKind::Use)],
        );
        op_allocs.insert(
            crate::cfg_mir::OpId::Inst(inst_id),
            vec![Allocation::reg(preg_int(1))],
        );
        op_operands.insert(
            crate::cfg_mir::OpId::Inst(inst_id),
            vec![(target_vreg, crate::OperandKind::Use)],
        );

        AllocatedCfgProgram {
            cfg_program: crate::cfg_mir::Program {
                funcs: vec![cfg_func],
                vreg_count: 32,
                slot_count: 0,
            },
            functions: vec![AllocatedCfgFunction {
                lambda_id: LambdaId::new(0),
                num_spillslots: 0,
                edits: Vec::new(),
                op_allocs,
                op_operands,
                edge_edits: vec![CfgEdgeEdit {
                    edge: edge_id,
                    pos: regalloc2::InstPosition::After,
                    from: Allocation::reg(preg_int(0)),
                    to: Allocation::reg(preg_int(1)),
                }],
                return_result_allocs: Vec::new(),
            }],
        }
    }

    #[test]
    fn cfg_static_edge_edit_verifier_accepts_expected_edge_edit_set() {
        let alloc = synthetic_cfg_edge_fixture();
        verify_static_edge_edits_cfg(&alloc).expect("expected cfg edge edit set should pass");
    }

    #[test]
    fn cfg_static_edge_edit_verifier_rejects_missing_required_edge_edit() {
        let mut alloc = synthetic_cfg_edge_fixture();
        let alloc_func = alloc
            .functions
            .first_mut()
            .expect("allocated cfg program should have one function");
        alloc_func.edge_edits.clear();

        let err =
            verify_static_edge_edits_cfg(&alloc).expect_err("missing cfg edge edit must fail");
        let msg = err.to_string();
        assert!(
            msg.contains("missing edge edit to cover block param"),
            "unexpected verifier error: {msg}"
        );
    }

    #[test]
    fn cfg_static_edge_edit_verifier_rejects_wrong_source_target_edit() {
        let mut alloc = synthetic_cfg_edge_fixture();
        let alloc_func = alloc
            .functions
            .first_mut()
            .expect("allocated cfg program should have one function");

        let edit = alloc_func
            .edge_edits
            .first_mut()
            .expect("first cfg edge edit should exist");
        edit.from = Allocation::reg(preg_int(2));

        let err = verify_static_edge_edits_cfg(&alloc)
            .expect_err("wrong cfg edge edit mapping must fail");
        let msg = err.to_string();
        assert!(
            msg.contains("do not deliver source"),
            "unexpected verifier error: {msg}"
        );
    }

    #[test]
    fn post_regalloc_simulation_matches_interpreter_for_simple_decoder() {
        let mut builder = IrBuilder::new(<u32 as facet::Facet>::SHAPE);
        {
            let mut rb = builder.root_region();
            rb.bounds_check(4);
            let value = rb.read_bytes(4);
            rb.write_to_field(value, 0, Width::W4);
            rb.set_results(&[]);
        }
        let mut func = builder.finish();
        let lin = linearize(&mut func);
        let cfg = crate::cfg_mir::lower_linear_ir(&lin);
        let alloc = allocate_cfg_program(&cfg).expect("allocation should succeed");

        let input = [0x78_u8, 0x56, 0x34, 0x12];
        let sim = simulate_execution_cfg(&alloc, &input).expect("simulation should succeed");
        let interp = crate::execute_program(&cfg, &input).expect("interpreter should succeed");

        assert_eq!(sim.output, interp.output);
        assert_eq!(sim.cursor, interp.cursor);
        assert_eq!(sim.trap, interp.trap);
        assert!(
            sim.returned,
            "expected simulator to finish with a return terminator"
        );
    }

    #[test]
    fn differential_checker_matches_for_simple_decoder() {
        let mut builder = IrBuilder::new(<u32 as facet::Facet>::SHAPE);
        {
            let mut rb = builder.root_region();
            rb.bounds_check(4);
            let value = rb.read_bytes(4);
            rb.write_to_field(value, 0, Width::W4);
            rb.set_results(&[]);
        }
        let mut func = builder.finish();
        let lin = linearize(&mut func);
        let cfg = crate::cfg_mir::lower_linear_ir(&lin);
        let alloc = allocate_cfg_program(&cfg).expect("cfg allocation should succeed");

        let input = [0x78_u8, 0x56, 0x34, 0x12];
        let result = differential_check_cfg(&cfg, &alloc, &input);
        match result {
            DifferentialCheckResult::Match { .. } => {}
            other => panic!("expected differential check to match, got: {other:?}"),
        }
    }

    #[test]
    fn differential_checker_matches_for_call_lambda_data_result() {
        let mut builder = IrBuilder::new(<u64 as facet::Facet>::SHAPE);
        let child = builder.create_lambda_with_data_args(<u64 as facet::Facet>::SHAPE, 1);
        {
            let mut rb = builder.lambda_region(child);
            let arg = rb.region_args(1)[0];
            let one = rb.const_val(1);
            let sum = rb.binop(IrOp::Add, arg, one);
            rb.set_results(&[sum]);
        }
        {
            let mut rb = builder.root_region();
            let x = rb.const_val(41);
            let out = rb.apply(child, &[x], 1);
            rb.write_to_field(out[0], 0, Width::W8);
            rb.set_results(&[]);
        }
        let mut func = builder.finish();
        let lin = linearize(&mut func);
        let cfg = crate::cfg_mir::lower_linear_ir(&lin);
        let alloc = allocate_cfg_program(&cfg).expect("cfg allocation should succeed");

        let result = differential_check_cfg(&cfg, &alloc, &[]);
        match result {
            DifferentialCheckResult::Match { .. } => {}
            other => panic!("expected differential check to match, got: {other:?}"),
        }
    }

    #[test]
    fn allocate_cfg_program_materializes_lambda_return_result_allocs() {
        let mut builder = IrBuilder::new(<u64 as facet::Facet>::SHAPE);
        let child = builder.create_lambda_with_data_args(<u64 as facet::Facet>::SHAPE, 1);
        {
            let mut rb = builder.lambda_region(child);
            let arg = rb.region_args(1)[0];
            let one = rb.const_val(1);
            let sum = rb.binop(IrOp::Add, arg, one);
            rb.set_results(&[sum]);
        }
        {
            let mut rb = builder.root_region();
            let x = rb.const_val(41);
            let out = rb.apply(child, &[x], 1);
            rb.write_to_field(out[0], 0, Width::W8);
            rb.set_results(&[]);
        }
        let mut func = builder.finish();
        let lin = linearize(&mut func);
        let cfg = crate::cfg_mir::lower_linear_ir(&lin);
        let alloc = allocate_cfg_program(&cfg).expect("cfg allocation should succeed");
        let child_alloc = alloc
            .functions
            .iter()
            .find(|f| f.lambda_id == child)
            .expect("child lambda allocation should exist");
        assert_eq!(
            child_alloc.return_result_allocs.len(),
            1,
            "child lambda should have one return-result allocation"
        );
    }

    #[test]
    fn allocate_cfg_program_assigns_call_lambda_abi_operands() {
        let mut builder = IrBuilder::new(<u64 as facet::Facet>::SHAPE);
        let child = builder.create_lambda_with_data_args(<u64 as facet::Facet>::SHAPE, 1);
        {
            let mut rb = builder.lambda_region(child);
            let arg = rb.region_args(1)[0];
            let one = rb.const_val(1);
            let sum = rb.binop(IrOp::Add, arg, one);
            rb.set_results(&[sum]);
        }
        {
            let mut rb = builder.root_region();
            let x = rb.const_val(41);
            let out = rb.apply(child, &[x], 1);
            rb.write_to_field(out[0], 0, Width::W8);
            rb.set_results(&[]);
        }
        let mut func = builder.finish();
        let lin = linearize(&mut func);
        let cfg = crate::cfg_mir::lower_linear_ir(&lin);
        let alloc = allocate_cfg_program(&cfg).expect("cfg allocation should succeed");

        let root_alloc = alloc
            .functions
            .iter()
            .find(|f| f.lambda_id.index() == 0)
            .expect("root lambda allocation should exist");
        let root_func = cfg
            .funcs
            .iter()
            .find(|f| f.lambda_id.index() == 0)
            .expect("root lambda function should exist");
        let call_inst = root_func
            .insts
            .iter()
            .find(|inst| matches!(inst.op, LinearOp::CallLambda { .. }))
            .expect("root should contain CallLambda");
        let call_allocs = root_alloc
            .op_allocs
            .get(&OpId::Inst(call_inst.id))
            .expect("call op should have allocation metadata");

        assert!(
            call_allocs
                .first()
                .and_then(|a| a.as_reg())
                .is_some_and(|r| r.hw_enc() == 2),
            "first call operand should use ABI arg register 2, got {:?}",
            call_allocs.first()
        );
        assert!(
            call_allocs
                .get(1)
                .and_then(|a| a.as_reg())
                .is_some_and(|r| r.hw_enc() == 0),
            "second call operand should define ABI return register 0, got {:?}",
            call_allocs.get(1)
        );
    }

    #[test]
    fn allocate_cfg_program_wires_lambda_data_arg_from_abi_register() {
        let mut builder = IrBuilder::new(<u64 as facet::Facet>::SHAPE);
        let child = builder.create_lambda_with_data_args(<u64 as facet::Facet>::SHAPE, 1);
        {
            let mut rb = builder.lambda_region(child);
            let arg = rb.region_args(1)[0];
            let one = rb.const_val(1);
            let sum = rb.binop(IrOp::Add, arg, one);
            rb.set_results(&[sum]);
        }
        {
            let mut rb = builder.root_region();
            let x = rb.const_val(41);
            let out = rb.apply(child, &[x], 1);
            rb.write_to_field(out[0], 0, Width::W8);
            rb.set_results(&[]);
        }
        let mut func = builder.finish();
        let lin = linearize(&mut func);
        let cfg = crate::cfg_mir::lower_linear_ir(&lin);
        let alloc = allocate_cfg_program(&cfg).expect("cfg allocation should succeed");

        let child_func = cfg
            .funcs
            .iter()
            .find(|f| f.lambda_id == child)
            .expect("child lambda function should exist");
        let child_alloc = alloc
            .functions
            .iter()
            .find(|f| f.lambda_id == child)
            .expect("child lambda allocation should exist");
        let data_arg = child_func.data_args[0];
        let add_inst = child_func
            .insts
            .iter()
            .find(|inst| {
                matches!(
                    inst.op,
                    LinearOp::BinOp {
                        op: BinOpKind::Add,
                        ..
                    }
                )
            })
            .expect("child should contain add");
        let add_op = OpId::Inst(add_inst.id);
        let add_operands = child_alloc
            .op_operands
            .get(&add_op)
            .expect("add op should have operand metadata");
        let data_arg_operand_index = add_operands
            .iter()
            .position(|(vreg, kind)| *vreg == data_arg && *kind == RaOperandKind::Use)
            .expect("add should use child data arg");
        let data_arg_alloc = child_alloc
            .op_allocs
            .get(&add_op)
            .expect("add op should have allocation metadata")[data_arg_operand_index];

        let direct_abi = data_arg_alloc.as_reg().is_some_and(|reg| reg.hw_enc() == 2);
        if !direct_abi {
            let moved_from_abi = child_alloc.edits.iter().any(|(point, edit)| {
                let Edit::Move { from, to } = edit;
                let from_abi_arg2 = from.as_reg().is_some_and(|reg| reg.hw_enc() == 2);
                let to_matches = *to == data_arg_alloc;
                let before_add = *point == cfg_mir::ProgPoint::Before(add_op);
                from_abi_arg2 && to_matches && before_add
            });
            assert!(
                moved_from_abi,
                "child data arg alloc {:?} is not direct x2 and has no pre-add move from x2",
                data_arg_alloc
            );
        }
    }
}
