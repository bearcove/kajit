use std::collections::{BTreeMap, BTreeSet};

use crate::cfg_mir;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ProgramSize {
    pub funcs: usize,
    pub blocks: usize,
    pub insts: usize,
    pub terms: usize,
    pub edges: usize,
    pub edge_args: usize,
    pub block_params: usize,
}

impl ProgramSize {
    pub fn of(program: &cfg_mir::Program) -> Self {
        let mut size = Self {
            funcs: program.funcs.len(),
            blocks: 0,
            insts: 0,
            terms: 0,
            edges: 0,
            edge_args: 0,
            block_params: 0,
        };
        for func in &program.funcs {
            size.blocks += func.blocks.len();
            size.insts += func.insts.len();
            size.terms += func.terms.len();
            size.edges += func.edges.len();
            size.edge_args += func.edges.iter().map(|edge| edge.args.len()).sum::<usize>();
            size.block_params += func
                .blocks
                .iter()
                .map(|block| block.params.len())
                .sum::<usize>();
        }
        size
    }

    fn weight(self) -> (usize, usize, usize, usize, usize, usize, usize) {
        (
            self.blocks,
            self.insts,
            self.edges,
            self.edge_args,
            self.terms,
            self.block_params,
            self.funcs,
        )
    }

    fn is_smaller_than(self, other: Self) -> bool {
        self.weight() < other.weight()
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ReductionStep {
    pub strategy: &'static str,
    pub before: ProgramSize,
    pub after: ProgramSize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MinimizeStats {
    pub initial_size: ProgramSize,
    pub final_size: ProgramSize,
    pub attempts: usize,
    pub accepted: usize,
    pub steps: Vec<ReductionStep>,
}

#[derive(Debug)]
pub enum MinimizeError<E> {
    NotInteresting,
    Predicate(E),
}

#[derive(Debug, Clone)]
struct Candidate {
    strategy: &'static str,
    program: cfg_mir::Program,
}

pub fn minimize_cfg_program<F, E>(
    program: &cfg_mir::Program,
    mut is_interesting: F,
) -> Result<(cfg_mir::Program, MinimizeStats), MinimizeError<E>>
where
    F: FnMut(&cfg_mir::Program) -> Result<bool, E>,
{
    if !is_interesting(program).map_err(MinimizeError::Predicate)? {
        return Err(MinimizeError::NotInteresting);
    }

    let mut current = program.clone();
    let initial_size = ProgramSize::of(&current);
    let mut attempts = 0usize;
    let mut accepted = 0usize;
    let mut steps = Vec::new();

    loop {
        let before = ProgramSize::of(&current);
        let mut best: Option<Candidate> = None;
        let mut best_size = before;

        for candidate in generate_candidates(&current) {
            let candidate_size = ProgramSize::of(&candidate.program);
            if !candidate_size.is_smaller_than(before) {
                continue;
            }

            attempts += 1;
            if !is_interesting(&candidate.program).map_err(MinimizeError::Predicate)? {
                continue;
            }

            if best.is_none() || candidate_size.is_smaller_than(best_size) {
                best_size = candidate_size;
                best = Some(candidate);
            }
        }

        let Some(best) = best else {
            break;
        };

        accepted += 1;
        steps.push(ReductionStep {
            strategy: best.strategy,
            before,
            after: best_size,
        });
        current = best.program;
    }

    let final_size = ProgramSize::of(&current);
    Ok((
        current,
        MinimizeStats {
            initial_size,
            final_size,
            attempts,
            accepted,
            steps,
        },
    ))
}

fn generate_candidates(program: &cfg_mir::Program) -> Vec<Candidate> {
    let mut out = Vec::new();
    for (func_index, func) in program.funcs.iter().enumerate() {
        for block in &func.blocks {
            let unused_params = unused_block_params(func, block);
            for param in unused_params {
                if let Some(candidate_program) =
                    remove_block_param(program, func_index, block.id, param)
                {
                    out.push(Candidate {
                        strategy: "remove_unused_block_param",
                        program: candidate_program,
                    });
                }
            }
        }

        for block in &func.blocks {
            if let Some(candidate_program) =
                collapse_trampoline_block(program, func_index, block.id)
            {
                out.push(Candidate {
                    strategy: "collapse_trampoline_block",
                    program: candidate_program,
                });
            }
        }

        let reachable = reachable_blocks(func);
        for block in &func.blocks {
            if block.id == func.entry || reachable.contains(&block.id) {
                continue;
            }
            if let Some(candidate_program) = remove_block(program, func_index, block.id) {
                out.push(Candidate {
                    strategy: "remove_unreachable_block",
                    program: candidate_program,
                });
            }
        }
    }
    out
}

fn unused_block_params(func: &cfg_mir::Function, block: &cfg_mir::Block) -> Vec<kajit_ir::VReg> {
    let used = used_vregs(func);
    block
        .params
        .iter()
        .copied()
        .filter(|param| !used.contains(param))
        .collect()
}

fn used_vregs(func: &cfg_mir::Function) -> BTreeSet<kajit_ir::VReg> {
    let mut used = BTreeSet::new();
    for &vreg in &func.data_args {
        used.insert(vreg);
    }
    for &vreg in &func.data_results {
        used.insert(vreg);
    }
    for inst in &func.insts {
        for operand in &inst.operands {
            if operand.kind == cfg_mir::OperandKind::Use {
                used.insert(operand.vreg);
            }
        }
    }
    for term in &func.terms {
        match term {
            cfg_mir::Terminator::BranchIf { cond, .. }
            | cfg_mir::Terminator::BranchIfZero { cond, .. } => {
                used.insert(*cond);
            }
            cfg_mir::Terminator::JumpTable { predicate, .. } => {
                used.insert(*predicate);
            }
            cfg_mir::Terminator::Return
            | cfg_mir::Terminator::ErrorExit { .. }
            | cfg_mir::Terminator::Branch { .. } => {}
        }
    }
    for edge in &func.edges {
        for arg in &edge.args {
            used.insert(arg.source);
        }
    }
    used
}

fn reachable_blocks(func: &cfg_mir::Function) -> BTreeSet<cfg_mir::BlockId> {
    let mut visited = BTreeSet::new();
    let mut stack = vec![func.entry];
    while let Some(block_id) = stack.pop() {
        if !visited.insert(block_id) {
            continue;
        }
        let Some(block) = func.block(block_id) else {
            continue;
        };
        for edge_id in &block.succs {
            if let Some(edge) = func.edge(*edge_id) {
                stack.push(edge.to);
            }
        }
    }
    visited
}

fn remove_block(
    program: &cfg_mir::Program,
    func_index: usize,
    block_id: cfg_mir::BlockId,
) -> Option<cfg_mir::Program> {
    let func = program.funcs.get(func_index)?;
    if block_id == func.entry || reachable_blocks(func).contains(&block_id) {
        return None;
    }

    let rebuilt = rebuild_function_without_block(func, block_id)?;
    rebuilt.validate().ok()?;

    let mut out = program.clone();
    out.funcs[func_index] = rebuilt;
    out.validate().ok()?;
    Some(out)
}

fn remove_block_param(
    program: &cfg_mir::Program,
    func_index: usize,
    block_id: cfg_mir::BlockId,
    param: kajit_ir::VReg,
) -> Option<cfg_mir::Program> {
    let mut out = program.clone();
    let func = out.funcs.get_mut(func_index)?;
    let block = func.blocks.get_mut(block_id.index())?;
    let before_len = block.params.len();
    block.params.retain(|candidate| *candidate != param);
    if block.params.len() == before_len {
        return None;
    }

    for edge_id in block.preds.clone() {
        let edge = func.edges.get_mut(edge_id.index())?;
        edge.args.retain(|arg| arg.target != param);
    }

    func.validate().ok()?;
    out.validate().ok()?;
    Some(out)
}

fn collapse_trampoline_block(
    program: &cfg_mir::Program,
    func_index: usize,
    block_id: cfg_mir::BlockId,
) -> Option<cfg_mir::Program> {
    let mut out = program.clone();
    let func = out.funcs.get_mut(func_index)?;
    let block = func.block(block_id)?.clone();
    if block.id == func.entry || !block.insts.is_empty() || block.preds.is_empty() {
        return None;
    }

    let cfg_mir::Terminator::Branch { edge: out_edge_id } = func.term(block.term)?.clone() else {
        return None;
    };
    let out_edge = func.edge(out_edge_id)?.clone();
    let successor_id = out_edge.to;

    let successor = func.block(successor_id)?.clone();
    let mut successor_preds = successor
        .preds
        .iter()
        .copied()
        .filter(|edge_id| *edge_id != out_edge_id)
        .collect::<Vec<_>>();

    for pred_edge_id in &block.preds {
        let pred_edge = func.edge(*pred_edge_id)?.clone();
        let composed_args = compose_edge_args(&block.params, &pred_edge.args, &out_edge.args)?;
        let edge = func.edges.get_mut(pred_edge_id.index())?;
        edge.to = successor_id;
        edge.args = composed_args;
        successor_preds.push(*pred_edge_id);
    }

    func.blocks.get_mut(successor_id.index())?.preds = successor_preds;
    let rebuilt = rebuild_function_without_block(func, block_id)?;
    rebuilt.validate().ok()?;
    out.funcs[func_index] = rebuilt;
    out.validate().ok()?;
    Some(out)
}

fn compose_edge_args(
    params: &[kajit_ir::VReg],
    pred_args: &[cfg_mir::EdgeArg],
    succ_args: &[cfg_mir::EdgeArg],
) -> Option<Vec<cfg_mir::EdgeArg>> {
    let param_set = params.iter().copied().collect::<BTreeSet<_>>();
    let pred_source_by_target = pred_args
        .iter()
        .map(|arg| (arg.target, arg.source))
        .collect::<BTreeMap<_, _>>();
    let mut out = Vec::with_capacity(succ_args.len());
    for arg in succ_args {
        if !param_set.contains(&arg.source) {
            return None;
        }
        let source = pred_source_by_target.get(&arg.source).copied()?;
        out.push(cfg_mir::EdgeArg {
            target: arg.target,
            source,
        });
    }
    Some(out)
}

fn rebuild_function_without_block(
    func: &cfg_mir::Function,
    removed_block: cfg_mir::BlockId,
) -> Option<cfg_mir::Function> {
    let mut block_map = vec![None; func.blocks.len()];
    let mut next_block = 0u32;
    for block in &func.blocks {
        if block.id == removed_block {
            continue;
        }
        block_map[block.id.index()] = Some(cfg_mir::BlockId::new(next_block));
        next_block += 1;
    }

    let kept_block_ids = func
        .blocks
        .iter()
        .filter(|block| block.id != removed_block)
        .map(|block| block.id)
        .collect::<BTreeSet<_>>();

    let mut inst_map = vec![None; func.insts.len()];
    let mut next_inst = 0u32;
    for block in &func.blocks {
        if block.id == removed_block {
            continue;
        }
        for inst_id in &block.insts {
            inst_map[inst_id.index()] = Some(cfg_mir::InstId::new(next_inst));
            next_inst += 1;
        }
    }

    let mut term_map = vec![None; func.terms.len()];
    let mut next_term = 0u32;
    for block in &func.blocks {
        if block.id == removed_block {
            continue;
        }
        term_map[block.term.index()] = Some(cfg_mir::TermId::new(next_term));
        next_term += 1;
    }

    let mut edge_map = vec![None; func.edges.len()];
    let mut next_edge = 0u32;
    for edge in &func.edges {
        if !kept_block_ids.contains(&edge.from) || !kept_block_ids.contains(&edge.to) {
            continue;
        }
        edge_map[edge.id.index()] = Some(cfg_mir::EdgeId::new(next_edge));
        next_edge += 1;
    }

    let mut new_blocks = Vec::with_capacity(func.blocks.len().saturating_sub(1));
    for block in &func.blocks {
        if block.id == removed_block {
            continue;
        }
        let new_id = block_map[block.id.index()]?;
        let new_term = term_map[block.term.index()]?;
        let insts = block
            .insts
            .iter()
            .map(|inst_id| inst_map[inst_id.index()])
            .collect::<Option<Vec<_>>>()?;
        let preds = block
            .preds
            .iter()
            .filter_map(|edge_id| edge_map[edge_id.index()])
            .collect::<Vec<_>>();
        let succs = block
            .succs
            .iter()
            .filter_map(|edge_id| edge_map[edge_id.index()])
            .collect::<Vec<_>>();
        new_blocks.push(cfg_mir::Block {
            id: new_id,
            params: block.params.clone(),
            insts,
            term: new_term,
            preds,
            succs,
        });
    }

    let mut new_insts = Vec::new();
    for inst in &func.insts {
        let Some(new_id) = inst_map[inst.id.index()] else {
            continue;
        };
        let mut cloned = inst.clone();
        cloned.id = new_id;
        new_insts.push(cloned);
    }

    let mut new_terms = Vec::new();
    for block in &func.blocks {
        if block.id == removed_block {
            continue;
        }
        let term = func.term(block.term)?.clone();
        new_terms.push(remap_terminator(&term, &edge_map)?);
    }

    let mut new_edges = Vec::new();
    for edge in &func.edges {
        let Some(new_id) = edge_map[edge.id.index()] else {
            continue;
        };
        new_edges.push(cfg_mir::Edge {
            id: new_id,
            from: block_map[edge.from.index()]?,
            to: block_map[edge.to.index()]?,
            args: edge.args.clone(),
        });
    }

    Some(cfg_mir::Function {
        id: func.id,
        lambda_id: func.lambda_id,
        entry: block_map[func.entry.index()]?,
        data_args: func.data_args.clone(),
        data_results: func.data_results.clone(),
        blocks: new_blocks,
        edges: new_edges,
        insts: new_insts,
        terms: new_terms,
    })
}

fn remap_terminator(
    term: &cfg_mir::Terminator,
    edge_map: &[Option<cfg_mir::EdgeId>],
) -> Option<cfg_mir::Terminator> {
    match term {
        cfg_mir::Terminator::Return => Some(cfg_mir::Terminator::Return),
        cfg_mir::Terminator::ErrorExit { code } => {
            Some(cfg_mir::Terminator::ErrorExit { code: *code })
        }
        cfg_mir::Terminator::Branch { edge } => Some(cfg_mir::Terminator::Branch {
            edge: edge_map[edge.index()]?,
        }),
        cfg_mir::Terminator::BranchIf {
            cond,
            taken,
            fallthrough,
        } => Some(cfg_mir::Terminator::BranchIf {
            cond: *cond,
            taken: edge_map[taken.index()]?,
            fallthrough: edge_map[fallthrough.index()]?,
        }),
        cfg_mir::Terminator::BranchIfZero {
            cond,
            taken,
            fallthrough,
        } => Some(cfg_mir::Terminator::BranchIfZero {
            cond: *cond,
            taken: edge_map[taken.index()]?,
            fallthrough: edge_map[fallthrough.index()]?,
        }),
        cfg_mir::Terminator::JumpTable {
            predicate,
            targets,
            default,
        } => Some(cfg_mir::Terminator::JumpTable {
            predicate: *predicate,
            targets: targets
                .iter()
                .map(|edge| edge_map[edge.index()])
                .collect::<Option<Vec<_>>>()?,
            default: edge_map[default.index()]?,
        }),
    }
}

#[cfg(test)]
mod tests {
    use kajit_ir::{ErrorCode, LambdaId, VReg};
    use kajit_lir::LinearOp;

    use super::*;

    fn vreg(index: u32) -> VReg {
        VReg::new(index)
    }

    fn dead_block_program() -> cfg_mir::Program {
        cfg_mir::Program {
            funcs: vec![cfg_mir::Function {
                id: cfg_mir::FunctionId::new(0),
                lambda_id: LambdaId::new(0),
                entry: cfg_mir::BlockId::new(0),
                data_args: Vec::new(),
                data_results: Vec::new(),
                blocks: vec![
                    cfg_mir::Block {
                        id: cfg_mir::BlockId::new(0),
                        params: Vec::new(),
                        insts: vec![cfg_mir::InstId::new(0)],
                        term: cfg_mir::TermId::new(0),
                        preds: Vec::new(),
                        succs: Vec::new(),
                    },
                    cfg_mir::Block {
                        id: cfg_mir::BlockId::new(1),
                        params: Vec::new(),
                        insts: vec![cfg_mir::InstId::new(1)],
                        term: cfg_mir::TermId::new(1),
                        preds: Vec::new(),
                        succs: Vec::new(),
                    },
                ],
                edges: Vec::new(),
                insts: vec![
                    cfg_mir::Inst {
                        id: cfg_mir::InstId::new(0),
                        op: LinearOp::Const {
                            dst: vreg(0),
                            value: 1,
                        },
                        operands: vec![cfg_mir::Operand {
                            vreg: vreg(0),
                            kind: cfg_mir::OperandKind::Def,
                            class: cfg_mir::RegClass::Gpr,
                            fixed: None,
                        }],
                        clobbers: cfg_mir::Clobbers::default(),
                    },
                    cfg_mir::Inst {
                        id: cfg_mir::InstId::new(1),
                        op: LinearOp::ErrorExit {
                            code: ErrorCode::InvalidVarint,
                        },
                        operands: Vec::new(),
                        clobbers: cfg_mir::Clobbers::default(),
                    },
                ],
                terms: vec![
                    cfg_mir::Terminator::Return,
                    cfg_mir::Terminator::ErrorExit {
                        code: ErrorCode::InvalidVarint,
                    },
                ],
            }],
            vreg_count: 1,
            slot_count: 0,
        }
    }

    fn unused_block_param_program() -> cfg_mir::Program {
        cfg_mir::Program {
            funcs: vec![cfg_mir::Function {
                id: cfg_mir::FunctionId::new(0),
                lambda_id: LambdaId::new(0),
                entry: cfg_mir::BlockId::new(0),
                data_args: Vec::new(),
                data_results: Vec::new(),
                blocks: vec![
                    cfg_mir::Block {
                        id: cfg_mir::BlockId::new(0),
                        params: Vec::new(),
                        insts: vec![cfg_mir::InstId::new(0)],
                        term: cfg_mir::TermId::new(0),
                        preds: Vec::new(),
                        succs: vec![cfg_mir::EdgeId::new(0)],
                    },
                    cfg_mir::Block {
                        id: cfg_mir::BlockId::new(1),
                        params: vec![vreg(1)],
                        insts: Vec::new(),
                        term: cfg_mir::TermId::new(1),
                        preds: vec![cfg_mir::EdgeId::new(0)],
                        succs: Vec::new(),
                    },
                ],
                edges: vec![cfg_mir::Edge {
                    id: cfg_mir::EdgeId::new(0),
                    from: cfg_mir::BlockId::new(0),
                    to: cfg_mir::BlockId::new(1),
                    args: vec![cfg_mir::EdgeArg {
                        target: vreg(1),
                        source: vreg(0),
                    }],
                }],
                insts: vec![cfg_mir::Inst {
                    id: cfg_mir::InstId::new(0),
                    op: LinearOp::Const {
                        dst: vreg(0),
                        value: 7,
                    },
                    operands: vec![cfg_mir::Operand {
                        vreg: vreg(0),
                        kind: cfg_mir::OperandKind::Def,
                        class: cfg_mir::RegClass::Gpr,
                        fixed: None,
                    }],
                    clobbers: cfg_mir::Clobbers::default(),
                }],
                terms: vec![
                    cfg_mir::Terminator::Branch {
                        edge: cfg_mir::EdgeId::new(0),
                    },
                    cfg_mir::Terminator::Return,
                ],
            }],
            vreg_count: 2,
            slot_count: 0,
        }
    }

    fn trampoline_program() -> cfg_mir::Program {
        cfg_mir::Program {
            funcs: vec![cfg_mir::Function {
                id: cfg_mir::FunctionId::new(0),
                lambda_id: LambdaId::new(0),
                entry: cfg_mir::BlockId::new(0),
                data_args: Vec::new(),
                data_results: Vec::new(),
                blocks: vec![
                    cfg_mir::Block {
                        id: cfg_mir::BlockId::new(0),
                        params: Vec::new(),
                        insts: vec![cfg_mir::InstId::new(0)],
                        term: cfg_mir::TermId::new(0),
                        preds: Vec::new(),
                        succs: vec![cfg_mir::EdgeId::new(0)],
                    },
                    cfg_mir::Block {
                        id: cfg_mir::BlockId::new(1),
                        params: vec![vreg(1)],
                        insts: Vec::new(),
                        term: cfg_mir::TermId::new(1),
                        preds: vec![cfg_mir::EdgeId::new(0)],
                        succs: vec![cfg_mir::EdgeId::new(1)],
                    },
                    cfg_mir::Block {
                        id: cfg_mir::BlockId::new(2),
                        params: vec![vreg(2)],
                        insts: vec![cfg_mir::InstId::new(1)],
                        term: cfg_mir::TermId::new(2),
                        preds: vec![cfg_mir::EdgeId::new(1)],
                        succs: Vec::new(),
                    },
                ],
                edges: vec![
                    cfg_mir::Edge {
                        id: cfg_mir::EdgeId::new(0),
                        from: cfg_mir::BlockId::new(0),
                        to: cfg_mir::BlockId::new(1),
                        args: vec![cfg_mir::EdgeArg {
                            target: vreg(1),
                            source: vreg(0),
                        }],
                    },
                    cfg_mir::Edge {
                        id: cfg_mir::EdgeId::new(1),
                        from: cfg_mir::BlockId::new(1),
                        to: cfg_mir::BlockId::new(2),
                        args: vec![cfg_mir::EdgeArg {
                            target: vreg(2),
                            source: vreg(1),
                        }],
                    },
                ],
                insts: vec![
                    cfg_mir::Inst {
                        id: cfg_mir::InstId::new(0),
                        op: LinearOp::Const {
                            dst: vreg(0),
                            value: 9,
                        },
                        operands: vec![cfg_mir::Operand {
                            vreg: vreg(0),
                            kind: cfg_mir::OperandKind::Def,
                            class: cfg_mir::RegClass::Gpr,
                            fixed: None,
                        }],
                        clobbers: cfg_mir::Clobbers::default(),
                    },
                    cfg_mir::Inst {
                        id: cfg_mir::InstId::new(1),
                        op: LinearOp::Copy {
                            dst: vreg(3),
                            src: vreg(2),
                        },
                        operands: vec![
                            cfg_mir::Operand {
                                vreg: vreg(3),
                                kind: cfg_mir::OperandKind::Def,
                                class: cfg_mir::RegClass::Gpr,
                                fixed: None,
                            },
                            cfg_mir::Operand {
                                vreg: vreg(2),
                                kind: cfg_mir::OperandKind::Use,
                                class: cfg_mir::RegClass::Gpr,
                                fixed: None,
                            },
                        ],
                        clobbers: cfg_mir::Clobbers::default(),
                    },
                ],
                terms: vec![
                    cfg_mir::Terminator::Branch {
                        edge: cfg_mir::EdgeId::new(0),
                    },
                    cfg_mir::Terminator::Branch {
                        edge: cfg_mir::EdgeId::new(1),
                    },
                    cfg_mir::Terminator::Return,
                ],
            }],
            vreg_count: 4,
            slot_count: 0,
        }
    }

    #[test]
    fn minimizer_rejects_uninteresting_seed() {
        let program = dead_block_program();
        let result = minimize_cfg_program(&program, |_| Ok::<_, ()>(false));
        assert!(matches!(result, Err(MinimizeError::NotInteresting)));
    }

    #[test]
    fn minimizer_removes_unreachable_block() {
        let program = dead_block_program();
        let (reduced, stats) =
            minimize_cfg_program(&program, |_| Ok::<_, ()>(true)).expect("seed should minimize");

        reduced.validate().expect("reduced program should validate");
        assert_eq!(reduced.funcs[0].blocks.len(), 1);
        assert_eq!(reduced.funcs[0].insts.len(), 1);
        assert_eq!(stats.accepted, 1);
        assert_eq!(stats.steps[0].strategy, "remove_unreachable_block");
        assert!(stats.final_size.is_smaller_than(stats.initial_size));
    }

    #[test]
    fn minimizer_removes_unused_block_param() {
        let program = unused_block_param_program();
        let (reduced, stats) =
            minimize_cfg_program(&program, |_| Ok::<_, ()>(true)).expect("seed should minimize");

        reduced.validate().expect("reduced program should validate");
        assert!(reduced.funcs[0].blocks[1].params.is_empty());
        assert!(reduced.funcs[0].edges[0].args.is_empty());
        assert_eq!(stats.accepted, 1);
        assert_eq!(stats.steps[0].strategy, "remove_unused_block_param");
        assert!(stats.final_size.is_smaller_than(stats.initial_size));
    }

    #[test]
    fn minimizer_collapses_trampoline_block() {
        let program = trampoline_program();
        let (reduced, stats) =
            minimize_cfg_program(&program, |_| Ok::<_, ()>(true)).expect("seed should minimize");

        reduced.validate().expect("reduced program should validate");
        assert_eq!(reduced.funcs[0].blocks.len(), 2);
        assert_eq!(reduced.funcs[0].edges.len(), 1);
        assert_eq!(
            reduced.funcs[0].blocks[1].preds,
            vec![cfg_mir::EdgeId::new(0)]
        );
        assert_eq!(
            reduced.funcs[0].edges[0].args,
            vec![cfg_mir::EdgeArg {
                target: vreg(2),
                source: vreg(0),
            }]
        );
        assert!(
            stats
                .steps
                .iter()
                .any(|step| step.strategy == "collapse_trampoline_block")
        );
    }

    #[test]
    fn minimizer_respects_predicate() {
        let program = dead_block_program();
        let (reduced, stats) = minimize_cfg_program(&program, |candidate| {
            Ok::<_, ()>(candidate.funcs[0].blocks.len() >= 2)
        })
        .expect("seed should be interesting");

        assert_eq!(ProgramSize::of(&reduced), ProgramSize::of(&program));
        assert_eq!(stats.accepted, 0);
        assert_eq!(stats.attempts, 1);
    }
}
