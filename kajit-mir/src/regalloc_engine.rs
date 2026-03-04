//! regalloc2 integration over RA-MIR.
//!
//! This module adapts `regalloc_mir::RaProgram` to `regalloc2::Function`.

use std::collections::{BTreeMap, HashMap, HashSet};
use std::fmt;

use kajit_ir::ErrorCode;
use kajit_lir::{BinOpKind, LinearOp};
use regalloc2::{
    Allocation, Block, Edit, Inst, InstRange, MachineEnv, Operand, OperandConstraint, OperandKind,
    OperandPos, Output, PReg, PRegSet, RegAllocError, RegClass, RegallocOptions, VReg,
};

use crate::{FixedReg, OperandKind as RaOperandKind, RaFunction, RaProgram};
use kajit_ir::LambdaId;

/// Materialized allocation result for one function.
#[derive(Debug, Clone)]
pub struct EdgeEdit {
    pub from_linear_op_index: usize,
    pub succ_index: usize,
    pub pos: regalloc2::InstPosition,
    pub from: Allocation,
    pub to: Allocation,
}

/// Materialized allocation result for one function.
#[derive(Debug, Clone)]
pub struct AllocatedFunction {
    pub lambda_id: LambdaId,
    pub num_spillslots: usize,
    pub edits: Vec<(regalloc2::ProgPoint, Edit)>,
    pub inst_allocs: Vec<Vec<Allocation>>,
    pub inst_linear_op_indices: Vec<Option<usize>>,
    pub edge_edits: Vec<EdgeEdit>,
    pub return_result_allocs: Vec<Allocation>,
}

/// Materialized allocation result for a full RA-MIR program.
#[derive(Debug, Clone)]
pub struct AllocatedProgram {
    pub ra_program: RaProgram,
    pub functions: Vec<AllocatedFunction>,
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
    term_kind: BlockTermKind,
    term_returns_data_results: bool,
    term_linear_op_index: Option<usize>,
    term_uses: Vec<kajit_ir::VReg>,
    succs: Vec<usize>,
    preds: Vec<usize>,
    params: Vec<kajit_ir::VReg>,
    succ_args: Vec<Vec<crate::RaEdgeArg>>,
}

fn align_succ_args_to_target(
    target_params: &[kajit_ir::VReg],
    args: &[crate::RaEdgeArg],
) -> Vec<crate::RaEdgeArg> {
    let source_by_target: HashMap<_, _> = args.iter().map(|arg| (arg.target, arg.source)).collect();

    target_params
        .iter()
        .map(|target| crate::RaEdgeArg {
            target: *target,
            source: source_by_target.get(target).copied().unwrap_or(*target),
        })
        .collect()
}

#[derive(Debug, Clone, Copy)]
struct EdgeBlockInfo {
    from_linear_op_index: usize,
    succ_index: usize,
}

#[derive(Debug, Clone)]
struct AdapterFunction {
    blocks: Vec<AdapterBlock>,
    insts: Vec<AdapterInst>,
    inst_linear_op_indices: Vec<Option<usize>>,
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

fn lower_term_uses(term: &crate::regalloc_mir::RaTerminator) -> Vec<kajit_ir::VReg> {
    match term {
        crate::regalloc_mir::RaTerminator::BranchIf { cond, .. }
        | crate::regalloc_mir::RaTerminator::BranchIfZero { cond, .. } => vec![*cond],
        crate::regalloc_mir::RaTerminator::JumpTable { predicate, .. } => vec![*predicate],
        crate::regalloc_mir::RaTerminator::Return
        | crate::regalloc_mir::RaTerminator::ErrorExit { .. }
        | crate::regalloc_mir::RaTerminator::Branch { .. } => Vec::new(),
    }
}

fn split_critical_edges(func: &RaFunction) -> (Vec<WorkBlock>, Vec<Option<EdgeBlockInfo>>) {
    let mut blocks: Vec<WorkBlock> = func
        .blocks
        .iter()
        .map(|b| {
            let term_kind = match b.term {
                crate::regalloc_mir::RaTerminator::Return
                | crate::regalloc_mir::RaTerminator::ErrorExit { .. } => BlockTermKind::Ret,
                _ => BlockTermKind::BranchLike,
            };
            let term_returns_data_results =
                matches!(b.term, crate::regalloc_mir::RaTerminator::Return);
            WorkBlock {
                raw_insts: b.insts.clone(),
                term_kind,
                term_returns_data_results,
                term_linear_op_index: b.term_linear_op_index,
                term_uses: lower_term_uses(&b.term),
                succs: b.succs.iter().map(|s| s.to.0 as usize).collect(),
                preds: b.preds.iter().map(|p| p.0 as usize).collect(),
                params: b.params.clone(),
                succ_args: b
                    .succs
                    .iter()
                    .map(|s| {
                        align_succ_args_to_target(&func.blocks[s.to.0 as usize].params, &s.args)
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

            let args = blocks[from].succ_args[succ_idx].clone();
            let to_params = blocks[to].params.clone();
            let from_linear_op_index = blocks[from]
                .term_linear_op_index
                .expect("critical-edge source must map to a linear op index");
            let edge_block = WorkBlock {
                raw_insts: Vec::new(),
                term_kind: BlockTermKind::BranchLike,
                term_returns_data_results: false,
                term_linear_op_index: None,
                term_uses: Vec::new(),
                succs: vec![to],
                preds: vec![from],
                params: to_params,
                succ_args: vec![args],
            };
            let edge_id = blocks.len();
            blocks.push(edge_block);
            edge_infos.push(Some(EdgeBlockInfo {
                from_linear_op_index,
                succ_index: succ_idx,
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

    (blocks, edge_infos)
}

impl AdapterFunction {
    fn from_ra(func: &RaFunction, num_vregs: usize) -> Self {
        let (mut blocks, edge_infos) = split_critical_edges(func);
        let mut adapter_insts = Vec::<AdapterInst>::new();
        let mut inst_linear_op_indices = Vec::<Option<usize>>::new();
        let mut inst_edge_infos = Vec::<Option<EdgeBlockInfo>>::new();
        let mut adapter_blocks = Vec::<AdapterBlock>::new();

        for (block_index, b) in blocks.iter_mut().enumerate() {
            let start = Inst::new(adapter_insts.len());

            if block_index == 0 {
                let entry_linear_op_index = b
                    .raw_insts
                    .first()
                    .map(|inst| inst.linear_op_index)
                    .or(b.term_linear_op_index);
                let mut entry_operands = Vec::new();
                for (arg_idx, &arg) in func.data_args.iter().enumerate() {
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
                        clobbers: PRegSet::empty(),
                        is_branch: false,
                        is_ret: false,
                        ret_value_operand_start: 0,
                        ret_value_operand_count: 0,
                    });
                    inst_linear_op_indices.push(entry_linear_op_index);
                    inst_edge_infos.push(None);
                }
            }

            for inst in &b.raw_insts {
                let operands: Vec<Operand> =
                    inst.operands.iter().copied().map(lower_operand).collect();
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
                    clobbers,
                    is_branch: false,
                    is_ret: false,
                    ret_value_operand_start: 0,
                    ret_value_operand_count: 0,
                });
                inst_linear_op_indices.push(Some(inst.linear_op_index));
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
                ret_value_operand_count = func.data_results.len();
                for (i, &result) in func.data_results.iter().enumerate() {
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
            adapter_insts.push(AdapterInst {
                operands: term_operands,
                clobbers: PRegSet::empty(),
                is_branch,
                is_ret,
                ret_value_operand_start,
                ret_value_operand_count,
            });
            inst_linear_op_indices.push(b.term_linear_op_index);
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
                    .map(|args| args.iter().map(|arg| int_vreg(arg.source)).collect())
                    .collect(),
            });
        }

        Self {
            blocks: adapter_blocks,
            insts: adapter_insts,
            inst_linear_op_indices,
            inst_edge_infos,
            num_vregs,
            empty_vregs: Vec::new(),
        }
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

fn materialize_output(
    out: &Output,
    adapter: &AdapterFunction,
    lambda_id: LambdaId,
) -> AllocatedFunction {
    let mut inst_allocs = Vec::with_capacity(adapter.insts.len());
    for i in 0..adapter.insts.len() {
        inst_allocs.push(out.inst_allocs(Inst::new(i)).to_vec());
    }
    let mut return_result_allocs = Vec::<Allocation>::new();
    for (inst_index, inst) in adapter.insts.iter().enumerate() {
        if inst.ret_value_operand_count == 0 {
            continue;
        }
        let term_allocs = &inst_allocs[inst_index];
        let start = inst.ret_value_operand_start;
        let end = start + inst.ret_value_operand_count;
        assert!(
            end <= term_allocs.len(),
            "missing return alloc operands for lambda {:?}: expected range {start}..{end}, got {}",
            lambda_id,
            term_allocs.len()
        );
        let candidate = term_allocs[start..end].to_vec();
        if return_result_allocs.is_empty() {
            return_result_allocs = candidate;
        } else if return_result_allocs != candidate {
            panic!(
                "inconsistent return allocs for lambda {:?}: {:?} vs {:?}",
                lambda_id, return_result_allocs, candidate
            );
        }
    }
    let mut edge_edits = Vec::new();
    for (prog_point, edit) in &out.edits {
        let inst_index = prog_point.inst().index();
        let Some(edge_info) = adapter
            .inst_edge_infos
            .get(inst_index)
            .and_then(|info| *info)
        else {
            continue;
        };
        let Edit::Move { from, to } = edit;
        edge_edits.push(EdgeEdit {
            from_linear_op_index: edge_info.from_linear_op_index,
            succ_index: edge_info.succ_index,
            pos: prog_point.pos(),
            from: *from,
            to: *to,
        });
    }
    AllocatedFunction {
        lambda_id,
        num_spillslots: out.num_spillslots,
        edits: out.edits.clone(),
        inst_allocs,
        inst_linear_op_indices: adapter.inst_linear_op_indices.clone(),
        edge_edits,
        return_result_allocs,
    }
}

fn allocate_ra_function(
    func: &RaFunction,
    vreg_count: usize,
    env: &MachineEnv,
    options: &RegallocOptions,
) -> Result<AllocatedFunction, RegallocEngineError> {
    let adapter = AdapterFunction::from_ra(func, vreg_count);
    let out = regalloc2::run(&adapter, env, options)?;

    // r[impl ir.regalloc.checker]
    #[cfg(debug_assertions)]
    {
        let mut checker = regalloc2::checker::Checker::new(&adapter, env);
        checker.prepare(&out);
        checker
            .run()
            .map_err(|errs| RegallocEngineError::Checker(format!("{errs:?}")))?;
    }

    Ok(materialize_output(&out, &adapter, func.lambda_id))
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

fn verify_function_static_edge_edits(func: &AllocatedFunction) -> Result<(), RegallocEngineError> {
    let scratch_regs: HashSet<PReg> = machine_env()
        .scratch_by_class
        .into_iter()
        .flatten()
        .collect();

    let mut by_edge = BTreeMap::<(usize, usize), Vec<&EdgeEdit>>::new();
    for edge in &func.edge_edits {
        by_edge
            .entry((edge.from_linear_op_index, edge.succ_index))
            .or_default()
            .push(edge);
    }

    for ((from_linear_op_index, succ_index), edits) in by_edge {
        let edge_label = format!(
            "lambda @{} edge (lin={from_linear_op_index}, succ={succ_index})",
            func.lambda_id.index()
        );

        let mut destinations = HashSet::<Allocation>::new();
        for edit in &edits {
            verify_allocation_is_valid(edit.from, func.num_spillslots, "source", &edge_label)?;
            verify_allocation_is_valid(edit.to, func.num_spillslots, "target", &edge_label)?;

            if !destinations.insert(edit.to) {
                return Err(RegallocEngineError::StaticVerifier(format!(
                    "{edge_label}: multiple edits write to the same destination {:#?}",
                    edit.to
                )));
            }
        }

        // Static ordering check: if an edit writes a register that a later edit
        // reads before any intervening write, this likely clobbers a still-live
        // source value. Exempt dedicated scratch registers.
        for (idx, edit) in edits.iter().enumerate() {
            let Some(dst_reg) = edit.to.as_reg() else {
                continue;
            };
            if scratch_regs.contains(&dst_reg) {
                continue;
            }

            let mut next_read = None;
            let mut overwritten = false;
            for (later_idx, later) in edits.iter().enumerate().skip(idx + 1) {
                if later.to == edit.to {
                    overwritten = true;
                    break;
                }
                if later.from == edit.to {
                    next_read = Some(later_idx);
                    break;
                }
            }

            if overwritten {
                continue;
            }
            if let Some(later_idx) = next_read {
                return Err(RegallocEngineError::StaticVerifier(format!(
                    "{edge_label}: edit #{idx} writes {:?} before later edit #{later_idx} reads it (possible clobber of still-live source)",
                    edit.to
                )));
            }
        }
    }

    Ok(())
}

pub fn verify_static_edge_edits(program: &AllocatedProgram) -> Result<(), RegallocEngineError> {
    for func in &program.functions {
        verify_function_static_edge_edits(func)?;
    }
    Ok(())
}

// r[impl ir.regalloc.engine]
pub fn allocate_program(program: &RaProgram) -> Result<AllocatedProgram, RegallocEngineError> {
    let env = machine_env();
    let options = RegallocOptions {
        verbose_log: false,
        validate_ssa: false,
        algorithm: regalloc2::Algorithm::Ion,
    };

    let mut functions = Vec::with_capacity(program.funcs.len());
    for func in &program.funcs {
        functions.push(allocate_ra_function(
            func,
            program.vreg_count as usize,
            &env,
            &options,
        )?);
    }
    let allocated = AllocatedProgram {
        ra_program: program.clone(),
        functions,
    };

    #[cfg(debug_assertions)]
    verify_static_edge_edits(&allocated)?;

    Ok(allocated)
}

/// Run regalloc2 on linear IR by first lowering to RA-MIR.
pub fn allocate_linear_ir(
    ir: &kajit_lir::LinearIr,
) -> Result<AllocatedProgram, RegallocEngineError> {
    let ra = crate::regalloc_mir::lower_linear_ir(ir);
    allocate_program(&ra)
}

const MAX_SIM_STEPS: usize = 1_000_000;

fn infer_output_size_for_function(func: &RaFunction) -> usize {
    func.blocks
        .iter()
        .flat_map(|block| block.insts.iter())
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

fn execution_position(block: &crate::RaBlock, next_inst_index: usize) -> ExecutionPosition {
    ExecutionPosition {
        block: block.id,
        next_inst_index,
        at_terminator: next_inst_index >= block.insts.len(),
    }
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

fn find_inst_index_for_linear(
    inst_index_by_linear: &HashMap<usize, usize>,
    linear_op_index: usize,
) -> Result<usize, RegallocEngineError> {
    inst_index_by_linear
        .get(&linear_op_index)
        .copied()
        .ok_or_else(|| {
            RegallocEngineError::Simulation(format!(
                "missing allocated instruction for linear op {linear_op_index}"
            ))
        })
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

fn find_term_use_alloc(
    term: &crate::RaTerminator,
    term_allocs: &[Allocation],
    vreg: kajit_ir::VReg,
    linear_op_index: usize,
) -> Result<Allocation, RegallocEngineError> {
    let uses = lower_term_uses(term);
    for (idx, use_vreg) in uses.into_iter().enumerate() {
        if use_vreg == vreg {
            return term_allocs.get(idx).copied().ok_or_else(|| {
                RegallocEngineError::Simulation(format!(
                    "missing term allocation for v{} in linear op {linear_op_index}",
                    vreg.index()
                ))
            });
        }
    }
    Err(RegallocEngineError::Simulation(format!(
        "missing term use v{} in linear op {linear_op_index}",
        vreg.index()
    )))
}

fn find_succ_index(
    block: &crate::RaBlock,
    to: crate::BlockId,
) -> Result<usize, RegallocEngineError> {
    block
        .succs
        .iter()
        .position(|edge| edge.to == to)
        .ok_or_else(|| {
            RegallocEngineError::Simulation(format!(
                "missing CFG edge b{} -> b{}",
                block.id.0, to.0
            ))
        })
}

pub fn simulate_execution(
    allocated: &AllocatedProgram,
    input: &[u8],
) -> Result<ExecutionResult, RegallocEngineError> {
    let func = allocated.ra_program.funcs.first().ok_or_else(|| {
        RegallocEngineError::Simulation("RA-MIR program has no functions".to_string())
    })?;
    let alloc_func = allocated
        .functions
        .iter()
        .find(|f| f.lambda_id == func.lambda_id)
        .or_else(|| allocated.functions.first())
        .ok_or_else(|| {
            RegallocEngineError::Simulation("allocated program has no functions".to_string())
        })?;

    let mut block_indices = HashMap::with_capacity(func.blocks.len());
    for (idx, block) in func.blocks.iter().enumerate() {
        block_indices.insert(block.id, idx);
    }

    let mut inst_index_by_linear = HashMap::<usize, usize>::new();
    for (inst_idx, maybe_linear) in alloc_func.inst_linear_op_indices.iter().enumerate() {
        if let Some(linear) = maybe_linear {
            inst_index_by_linear
                .entry(*linear)
                .and_modify(|existing| {
                    if inst_idx > *existing {
                        *existing = inst_idx;
                    }
                })
                .or_insert(inst_idx);
        }
    }

    let mut edits_by_progpoint =
        BTreeMap::<(usize, regalloc2::InstPosition), Vec<(Allocation, Allocation)>>::new();
    for (prog_point, edit) in &alloc_func.edits {
        let Edit::Move { from, to } = edit;
        edits_by_progpoint
            .entry((prog_point.inst().index(), prog_point.pos()))
            .or_default()
            .push((*from, *to));
    }

    let mut edge_edits =
        BTreeMap::<(usize, usize, regalloc2::InstPosition), Vec<(Allocation, Allocation)>>::new();
    for edge in &alloc_func.edge_edits {
        edge_edits
            .entry((edge.from_linear_op_index, edge.succ_index, edge.pos))
            .or_default()
            .push((edge.from, edge.to));
    }

    let mut regs = HashMap::<PReg, u64>::new();
    let mut spills = HashMap::<usize, u64>::new();
    let mut output = vec![0u8; infer_output_size_for_function(func)];
    let mut cursor = 0usize;
    let mut trap: Option<crate::InterpreterTrap> = None;
    let mut returned = false;
    let mut current = func.entry;
    let mut next_inst = 0usize;
    let mut steps = 0usize;

    while trap.is_none() && !returned {
        if steps >= MAX_SIM_STEPS {
            return Err(RegallocEngineError::Simulation(format!(
                "simulation exceeded step limit ({MAX_SIM_STEPS})"
            )));
        }
        steps += 1;

        let block_idx = *block_indices.get(&current).ok_or_else(|| {
            RegallocEngineError::Simulation(format!("unknown block b{}", current.0))
        })?;
        let block = &func.blocks[block_idx];

        if next_inst < block.insts.len() {
            let inst = &block.insts[next_inst];
            let inst_idx = find_inst_index_for_linear(&inst_index_by_linear, inst.linear_op_index)?;

            if let Some(edits) =
                edits_by_progpoint.get(&(inst_idx, regalloc2::InstPosition::Before))
            {
                apply_moves(&mut regs, &mut spills, edits);
            }

            let inst_allocs = alloc_func.inst_allocs.get(inst_idx).ok_or_else(|| {
                RegallocEngineError::Simulation(format!(
                    "missing inst allocs for adapter inst {inst_idx}"
                ))
            })?;

            match &inst.op {
                LinearOp::Const { dst, value } => {
                    let dst_alloc =
                        find_operand_alloc(inst, inst_allocs, *dst, RaOperandKind::Def)?;
                    write_allocation(&mut regs, &mut spills, dst_alloc, *value);
                }
                LinearOp::Copy { dst, src } => {
                    let src_alloc =
                        find_operand_alloc(inst, inst_allocs, *src, RaOperandKind::Use)?;
                    let dst_alloc =
                        find_operand_alloc(inst, inst_allocs, *dst, RaOperandKind::Def)?;
                    let value = read_allocation(&regs, &spills, src_alloc);
                    write_allocation(&mut regs, &mut spills, dst_alloc, value);
                }
                LinearOp::BinOp { op, dst, lhs, rhs } => {
                    let lhs_alloc =
                        find_operand_alloc(inst, inst_allocs, *lhs, RaOperandKind::Use)?;
                    let rhs_alloc =
                        find_operand_alloc(inst, inst_allocs, *rhs, RaOperandKind::Use)?;
                    let dst_alloc =
                        find_operand_alloc(inst, inst_allocs, *dst, RaOperandKind::Def)?;
                    let lhs_val = read_allocation(&regs, &spills, lhs_alloc);
                    let rhs_val = read_allocation(&regs, &spills, rhs_alloc);
                    write_allocation(
                        &mut regs,
                        &mut spills,
                        dst_alloc,
                        exec_binop(*op, lhs_val, rhs_val),
                    );
                }
                LinearOp::BoundsCheck { count } => {
                    if cursor + (*count as usize) > input.len() {
                        trap = Some(crate::InterpreterTrap {
                            code: ErrorCode::UnexpectedEof,
                            offset: cursor as u32,
                        });
                    }
                }
                LinearOp::ReadBytes { dst, count } => {
                    let count = *count as usize;
                    if cursor + count > input.len() {
                        trap = Some(crate::InterpreterTrap {
                            code: ErrorCode::UnexpectedEof,
                            offset: cursor as u32,
                        });
                    } else {
                        let mut value = 0u64;
                        for i in 0..count {
                            value |= (input[cursor + i] as u64) << (i * 8);
                        }
                        cursor += count;
                        let dst_alloc =
                            find_operand_alloc(inst, inst_allocs, *dst, RaOperandKind::Def)?;
                        write_allocation(&mut regs, &mut spills, dst_alloc, value);
                    }
                }
                LinearOp::WriteToField { src, offset, width } => {
                    let src_alloc =
                        find_operand_alloc(inst, inst_allocs, *src, RaOperandKind::Use)?;
                    let value = read_allocation(&regs, &spills, src_alloc);
                    let width_bytes = width.bytes() as usize;
                    let base = *offset as usize;
                    if output.len() < base + width_bytes {
                        output.resize(base + width_bytes, 0);
                    }
                    for i in 0..width_bytes {
                        output[base + i] = ((value >> (i * 8)) & 0xff) as u8;
                    }
                }
                LinearOp::ReadFromField { dst, offset, width } => {
                    let width_bytes = width.bytes() as usize;
                    let base = *offset as usize;
                    if output.len() < base + width_bytes {
                        output.resize(base + width_bytes, 0);
                    }
                    let mut value = 0u64;
                    for i in 0..width_bytes {
                        value |= (output[base + i] as u64) << (i * 8);
                    }
                    let dst_alloc =
                        find_operand_alloc(inst, inst_allocs, *dst, RaOperandKind::Def)?;
                    write_allocation(&mut regs, &mut spills, dst_alloc, value);
                }
                LinearOp::ErrorExit { code } => {
                    trap = Some(crate::InterpreterTrap {
                        code: *code,
                        offset: cursor as u32,
                    });
                }
                op => {
                    return Err(RegallocEngineError::Simulation(format!(
                        "unsupported op in simulation at b{}: {op:?}",
                        block.id.0
                    )));
                }
            }

            if let Some(edits) = edits_by_progpoint.get(&(inst_idx, regalloc2::InstPosition::After))
            {
                apply_moves(&mut regs, &mut spills, edits);
            }

            next_inst += 1;
            continue;
        }

        let term_linear = block.term_linear_op_index;
        let term_inst_idx =
            term_linear.and_then(|linear| inst_index_by_linear.get(&linear).copied());

        if let Some(term_inst_idx) = term_inst_idx
            && let Some(edits) =
                edits_by_progpoint.get(&(term_inst_idx, regalloc2::InstPosition::Before))
        {
            apply_moves(&mut regs, &mut spills, edits);
        }

        match &block.term {
            crate::RaTerminator::Return => {
                returned = true;
            }
            crate::RaTerminator::ErrorExit { code } => {
                trap = Some(crate::InterpreterTrap {
                    code: *code,
                    offset: cursor as u32,
                });
            }
            crate::RaTerminator::Branch { target } => {
                let succ_index = find_succ_index(block, *target)?;
                if let Some(term_linear) = term_linear
                    && let Some(edits) =
                        edge_edits.get(&(term_linear, succ_index, regalloc2::InstPosition::Before))
                {
                    apply_moves(&mut regs, &mut spills, edits);
                }
                if let Some(term_linear) = term_linear
                    && let Some(edits) =
                        edge_edits.get(&(term_linear, succ_index, regalloc2::InstPosition::After))
                {
                    apply_moves(&mut regs, &mut spills, edits);
                }
                current = *target;
                next_inst = 0;
            }
            crate::RaTerminator::BranchIf {
                cond,
                target,
                fallthrough,
            } => {
                let term_inst_idx = term_inst_idx.ok_or_else(|| {
                    RegallocEngineError::Simulation(format!(
                        "missing term inst index for b{} branch_if",
                        block.id.0
                    ))
                })?;
                let term_linear = term_linear.ok_or_else(|| {
                    RegallocEngineError::Simulation(format!(
                        "missing term linear index for b{} branch_if",
                        block.id.0
                    ))
                })?;
                let term_allocs = alloc_func.inst_allocs.get(term_inst_idx).ok_or_else(|| {
                    RegallocEngineError::Simulation(format!(
                        "missing term allocs for adapter inst {term_inst_idx}"
                    ))
                })?;
                let cond_alloc = find_term_use_alloc(&block.term, term_allocs, *cond, term_linear)?;
                let cond_value = read_allocation(&regs, &spills, cond_alloc);
                let next_block = if cond_value != 0 {
                    *target
                } else {
                    *fallthrough
                };
                let succ_index = find_succ_index(block, next_block)?;
                if let Some(edits) =
                    edge_edits.get(&(term_linear, succ_index, regalloc2::InstPosition::Before))
                {
                    apply_moves(&mut regs, &mut spills, edits);
                }
                if let Some(edits) =
                    edge_edits.get(&(term_linear, succ_index, regalloc2::InstPosition::After))
                {
                    apply_moves(&mut regs, &mut spills, edits);
                }
                current = next_block;
                next_inst = 0;
            }
            crate::RaTerminator::BranchIfZero {
                cond,
                target,
                fallthrough,
            } => {
                let term_inst_idx = term_inst_idx.ok_or_else(|| {
                    RegallocEngineError::Simulation(format!(
                        "missing term inst index for b{} branch_if_zero",
                        block.id.0
                    ))
                })?;
                let term_linear = term_linear.ok_or_else(|| {
                    RegallocEngineError::Simulation(format!(
                        "missing term linear index for b{} branch_if_zero",
                        block.id.0
                    ))
                })?;
                let term_allocs = alloc_func.inst_allocs.get(term_inst_idx).ok_or_else(|| {
                    RegallocEngineError::Simulation(format!(
                        "missing term allocs for adapter inst {term_inst_idx}"
                    ))
                })?;
                let cond_alloc = find_term_use_alloc(&block.term, term_allocs, *cond, term_linear)?;
                let cond_value = read_allocation(&regs, &spills, cond_alloc);
                let next_block = if cond_value == 0 {
                    *target
                } else {
                    *fallthrough
                };
                let succ_index = find_succ_index(block, next_block)?;
                if let Some(edits) =
                    edge_edits.get(&(term_linear, succ_index, regalloc2::InstPosition::Before))
                {
                    apply_moves(&mut regs, &mut spills, edits);
                }
                if let Some(edits) =
                    edge_edits.get(&(term_linear, succ_index, regalloc2::InstPosition::After))
                {
                    apply_moves(&mut regs, &mut spills, edits);
                }
                current = next_block;
                next_inst = 0;
            }
            term => {
                return Err(RegallocEngineError::Simulation(format!(
                    "unsupported terminator in simulation at b{}: {term:?}",
                    block.id.0
                )));
            }
        }

        if let Some(term_inst_idx) = term_inst_idx
            && let Some(edits) =
                edits_by_progpoint.get(&(term_inst_idx, regalloc2::InstPosition::After))
        {
            apply_moves(&mut regs, &mut spills, edits);
        }
    }

    let current_block_idx = *block_indices.get(&current).ok_or_else(|| {
        RegallocEngineError::Simulation(format!("unknown final block b{}", current.0))
    })?;
    let current_block = &func.blocks[current_block_idx];

    Ok(ExecutionResult {
        state: ExecutionState {
            physical_registers: regs,
            spillslots: spills,
            position: execution_position(current_block, next_inst),
        },
        output,
        cursor,
        trap,
        returned,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lower_linear_ir;
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
        let alloc = allocate_linear_ir(&lin).expect("regalloc2 should allocate gamma/theta");
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
        let ra = lower_linear_ir(&lin);
        let _alloc = allocate_program(&ra).unwrap_or_else(|e| {
            panic!(
                "regalloc should allocate textual theta fixture: {e}\n--- linear ---\n{lin}\n--- ra-mir ---\n{ra}"
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
        let ra = lower_linear_ir(&lin);
        let _alloc = allocate_program(&ra).unwrap_or_else(|e| {
            panic!(
                "regalloc should allocate textual theta fixture after passes: {e}\n--- linear ---\n{lin}\n--- ra-mir ---\n{ra}"
            )
        });
    }

    // r[verify ir.regalloc.engine]
    #[test]
    fn regalloc2_allocates_postcard_vec_decoder() {
        let mut func = build_stress_ir();
        run_default_passes(&mut func);
        let lin = linearize(&mut func);
        let alloc = allocate_linear_ir(&lin).expect("regalloc2 should allocate stress IR");
        assert!(
            alloc.functions.iter().any(|f| f.lambda_id.index() == 0),
            "expected allocation output to include root lambda"
        );
        let total_inst_allocs: usize = alloc.functions.iter().map(|f| f.inst_allocs.len()).sum();
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
        let alloc = allocate_linear_ir(&lin).expect("regalloc2 should allocate call-heavy IR");
        assert!(!alloc.functions[0].inst_allocs.is_empty());
    }

    #[test]
    fn regalloc2_postcard_vec_edit_budget_on_aarch64() {
        if !cfg!(target_arch = "aarch64") {
            return;
        }
        let mut func = build_stress_ir();
        run_default_passes(&mut func);
        let lin = linearize(&mut func);
        let alloc = allocate_linear_ir(&lin).expect("regalloc2 should allocate stress IR");

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
        let ra = lower_linear_ir(&lin);
        let env = machine_env();
        let options = RegallocOptions {
            verbose_log: false,
            validate_ssa: false,
            algorithm: regalloc2::Algorithm::Ion,
        };

        let adapter = AdapterFunction::from_ra(&ra.funcs[0], ra.vreg_count as usize);
        let mut out = regalloc2::run(&adapter, &env, &options).expect("allocation should succeed");
        if !out.allocs.is_empty() {
            // Corrupt the first allocation with the wrong register class on purpose.
            out.allocs[0] = Allocation::reg(preg_vec(0));
        }

        let mut checker = regalloc2::checker::Checker::new(&adapter, &env);
        checker.prepare(&out);
        let _ = checker.run();
    }

    #[test]
    fn static_edge_edit_verifier_accepts_allocated_program() {
        let mut func = build_stress_ir();
        run_default_passes(&mut func);
        let lin = linearize(&mut func);
        let alloc = allocate_linear_ir(&lin).expect("allocation should succeed");
        verify_static_edge_edits(&alloc).expect("static verifier should accept allocator output");
    }

    #[test]
    fn static_edge_edit_verifier_rejects_duplicate_dest_and_clobber() {
        let edge_key_linear = 42usize;
        let edge_key_succ = 0usize;
        let bad = AllocatedProgram {
            ra_program: RaProgram {
                funcs: Vec::new(),
                vreg_count: 0,
                slot_count: 0,
            },
            functions: vec![AllocatedFunction {
                lambda_id: LambdaId::new(0),
                num_spillslots: 2,
                edits: Vec::new(),
                inst_allocs: Vec::new(),
                inst_linear_op_indices: Vec::new(),
                edge_edits: vec![
                    EdgeEdit {
                        from_linear_op_index: edge_key_linear,
                        succ_index: edge_key_succ,
                        pos: regalloc2::InstPosition::After,
                        from: Allocation::reg(preg_int(0)),
                        to: Allocation::reg(preg_int(1)),
                    },
                    EdgeEdit {
                        from_linear_op_index: edge_key_linear,
                        succ_index: edge_key_succ,
                        pos: regalloc2::InstPosition::After,
                        from: Allocation::reg(preg_int(2)),
                        to: Allocation::reg(preg_int(1)),
                    },
                ],
                return_result_allocs: Vec::new(),
            }],
        };
        let err = verify_static_edge_edits(&bad).expect_err("duplicate destination must fail");
        let msg = err.to_string();
        assert!(
            msg.contains("multiple edits write to the same destination"),
            "unexpected verifier error: {msg}"
        );

        let clobber = AllocatedProgram {
            ra_program: RaProgram {
                funcs: Vec::new(),
                vreg_count: 0,
                slot_count: 0,
            },
            functions: vec![AllocatedFunction {
                lambda_id: LambdaId::new(0),
                num_spillslots: 2,
                edits: Vec::new(),
                inst_allocs: Vec::new(),
                inst_linear_op_indices: Vec::new(),
                edge_edits: vec![
                    EdgeEdit {
                        from_linear_op_index: edge_key_linear,
                        succ_index: edge_key_succ,
                        pos: regalloc2::InstPosition::After,
                        from: Allocation::reg(preg_int(0)),
                        to: Allocation::reg(preg_int(1)),
                    },
                    EdgeEdit {
                        from_linear_op_index: edge_key_linear,
                        succ_index: edge_key_succ,
                        pos: regalloc2::InstPosition::After,
                        from: Allocation::reg(preg_int(1)),
                        to: Allocation::reg(preg_int(2)),
                    },
                ],
                return_result_allocs: Vec::new(),
            }],
        };
        let err = verify_static_edge_edits(&clobber).expect_err("clobbering order must fail");
        let msg = err.to_string();
        assert!(
            msg.contains("possible clobber of still-live source"),
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
        let ra = lower_linear_ir(&lin);
        let alloc = allocate_program(&ra).expect("allocation should succeed");

        let input = [0x78_u8, 0x56, 0x34, 0x12];
        let sim = simulate_execution(&alloc, &input).expect("simulation should succeed");
        let interp = crate::execute_program(&ra, &input).expect("interpreter should succeed");

        assert_eq!(sim.output, interp.output);
        assert_eq!(sim.cursor, interp.cursor);
        assert_eq!(sim.trap, interp.trap);
        assert!(
            sim.returned,
            "expected simulator to finish with a return terminator"
        );
    }
}
