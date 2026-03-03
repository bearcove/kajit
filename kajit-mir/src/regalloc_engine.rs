//! regalloc2 integration over RA-MIR.
//!
//! This module adapts `regalloc_mir::RaProgram` to `regalloc2::Function`.

use std::fmt;

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
    pub functions: Vec<AllocatedFunction>,
}

#[derive(Debug, Clone)]
pub enum RegallocEngineError {
    Regalloc(RegAllocError),
    Checker(String),
}

impl fmt::Display for RegallocEngineError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Regalloc(err) => write!(f, "{err}"),
            Self::Checker(msg) => write!(f, "{msg}"),
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
    succ_args: Vec<Vec<kajit_ir::VReg>>,
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

#[cfg(all(target_arch = "x86_64", not(windows)))]
fn machine_env() -> MachineEnv {
    // r12-r15 first, then caller-saved GPRs.
    let preferred_int = set_from_regs(&[preg_int(12), preg_int(13), preg_int(14), preg_int(15)]);
    let non_preferred_int = set_from_regs(&[
        preg_int(0),
        preg_int(1),
        preg_int(2),
        preg_int(6),
        preg_int(7),
        preg_int(8),
        preg_int(9),
        preg_int(10),
        preg_int(11),
    ]);
    MachineEnv {
        preferred_regs_by_class: [preferred_int, set_from_regs(&[]), set_from_regs(&[])],
        non_preferred_regs_by_class: [non_preferred_int, set_from_regs(&[]), set_from_regs(&[])],
        scratch_by_class: [Some(preg_int(3)), None, None],
        fixed_stack_slots: Vec::new(),
    }
}

#[cfg(all(target_arch = "x86_64", windows))]
fn machine_env() -> MachineEnv {
    let preferred_int = set_from_regs(&[preg_int(12), preg_int(13), preg_int(14), preg_int(15)]);
    let non_preferred_int = set_from_regs(&[
        preg_int(0),
        preg_int(1),
        preg_int(2),
        preg_int(8),
        preg_int(9),
        preg_int(10),
        preg_int(11),
    ]);
    MachineEnv {
        preferred_regs_by_class: [preferred_int, set_from_regs(&[]), set_from_regs(&[])],
        non_preferred_regs_by_class: [non_preferred_int, set_from_regs(&[]), set_from_regs(&[])],
        scratch_by_class: [Some(preg_int(3)), None, None],
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
                succ_args: b.succs.iter().map(|s| s.args.clone()).collect(),
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
                params: args.clone(),
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
                    .map(|args| args.iter().copied().map(int_vreg).collect())
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
    Ok(AllocatedProgram { functions })
}

/// Run regalloc2 on linear IR by first lowering to RA-MIR.
pub fn allocate_linear_ir(
    ir: &kajit_lir::LinearIr,
) -> Result<AllocatedProgram, RegallocEngineError> {
    let ra = crate::regalloc_mir::lower_linear_ir(ir);
    allocate_program(&ra)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compiler;
    use crate::ir::IntrinsicRegistry;
    use crate::ir_parse::parse_ir;
    use crate::linearize::linearize;
    use facet::Facet;

    #[derive(Debug, Clone, serde::Serialize, serde::Deserialize, Facet)]
    struct ScalarVec {
        values: Vec<u32>,
    }

    // r[verify ir.regalloc.engine]
    #[test]
    fn regalloc2_allocates_gamma_and_theta_programs() {
        let mut builder = crate::ir::IrBuilder::new(<u32 as facet::Facet>::SHAPE);
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
                    bb.binop(crate::ir::IrOp::Add, args[0], one)
                };
                bb.set_results(&[val]);
            });
            let _ = rb.theta(&[gamma_out[0], one], |bb| {
                let args = bb.region_args(2);
                let counter = args[0];
                let one = args[1];
                let next = bb.binop(crate::ir::IrOp::Sub, counter, one);
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
        let ra = crate::regalloc_mir::lower_linear_ir(&lin);
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
        crate::ir_passes::run_default_passes(&mut func);
        let lin = linearize(&mut func);
        let ra = crate::regalloc_mir::lower_linear_ir(&lin);
        let _alloc = allocate_program(&ra).unwrap_or_else(|e| {
            panic!(
                "regalloc should allocate textual theta fixture after passes: {e}\n--- linear ---\n{lin}\n--- ra-mir ---\n{ra}"
            )
        });
    }

    // r[verify ir.regalloc.engine]
    #[test]
    fn regalloc2_allocates_postcard_vec_decoder() {
        let mut func =
            compiler::build_decoder_ir(ScalarVec::SHAPE, &crate::postcard::KajitPostcard);
        crate::ir_passes::run_default_passes(&mut func);
        let lin = linearize(&mut func);
        let alloc = allocate_linear_ir(&lin).expect("regalloc2 should allocate postcard vec path");
        assert!(
            alloc.functions.iter().any(|f| f.lambda_id.index() == 0),
            "expected allocation output to include root lambda"
        );
        let total_inst_allocs: usize = alloc.functions.iter().map(|f| f.inst_allocs.len()).sum();
        assert!(
            total_inst_allocs > 100,
            "expected sizable vec lowering, got {} insts",
            total_inst_allocs
        );
    }

    // r[verify ir.regalloc.engine]
    #[test]
    fn regalloc2_covers_call_operands_and_clobbers() {
        unsafe extern "C" fn add3(
            _ctx: *mut crate::context::DeserContext,
            a: u64,
            b: u64,
            c: u64,
        ) -> u64 {
            a + b + c
        }

        let mut builder = crate::ir::IrBuilder::new(<u64 as facet::Facet>::SHAPE);
        {
            let mut rb = builder.root_region();
            let a = rb.const_val(11);
            let b = rb.const_val(7);
            let c = rb.const_val(5);
            let out = rb
                .call_intrinsic(
                    crate::ir::IntrinsicFn(add3 as *const () as usize),
                    &[a, b, c],
                    0,
                    true,
                )
                .expect("intrinsic call should produce output");
            rb.write_to_field(out, 0, crate::ir::Width::W8);
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
        let mut func =
            compiler::build_decoder_ir(ScalarVec::SHAPE, &crate::postcard::KajitPostcard);
        crate::ir_passes::run_default_passes(&mut func);
        let lin = linearize(&mut func);
        let alloc = allocate_linear_ir(&lin).expect("regalloc2 should allocate postcard vec path");

        let total_edits: usize = alloc.functions.iter().map(|f| f.edits.len()).sum();
        assert!(
            total_edits <= 64,
            "expected postcard vec path to stay within edit budget (<=64), got {}",
            total_edits
        );
    }

    // r[verify ir.regalloc.checker]
    #[test]
    fn regalloc_checker_runs_on_corrupted_mapping() {
        let mut func =
            compiler::build_decoder_ir(ScalarVec::SHAPE, &crate::postcard::KajitPostcard);
        crate::ir_passes::run_default_passes(&mut func);
        let lin = linearize(&mut func);
        let ra = crate::regalloc_mir::lower_linear_ir(&lin);
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
}
