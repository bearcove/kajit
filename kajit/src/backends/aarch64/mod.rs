#![allow(clippy::useless_conversion)]

use kajit_emit::aarch64::{self, LabelId, Reg};
use regalloc2::{Allocation, Edit, InstPosition, PReg, RegClass, SpillSlot};
use std::collections::BTreeMap;

use crate::arch::{BASE_FRAME, EmitCtx};
use crate::ir::Width;
use crate::ir_backend::LinearBackendResult;
use crate::linearize::{BinOpKind, LinearOp, UnaryOpKind};
use crate::recipe::{Op, Recipe};
use crate::regalloc_engine::{AllocatedCfgProgram, cfg_mir};

mod alloc;
mod calls;
mod edits;
mod emit;

pub(crate) struct FunctionCtx {
    pub(crate) error_exit: LabelId,
    pub(crate) data_results: Vec<crate::ir::VReg>,
    pub(crate) lambda_id: crate::ir::LambdaId,
}

#[derive(Default)]
pub(crate) struct LambdaEditMap {
    pub(crate) before: BTreeMap<cfg_mir::OpId, Vec<(Allocation, Allocation)>>,
    pub(crate) after: BTreeMap<cfg_mir::OpId, Vec<(Allocation, Allocation)>>,
}

#[derive(Default)]
pub(crate) struct LambdaEdgeEditMap {
    pub(crate) before: BTreeMap<cfg_mir::EdgeId, Vec<(Allocation, Allocation)>>,
    pub(crate) after: BTreeMap<cfg_mir::EdgeId, Vec<(Allocation, Allocation)>>,
}

pub(crate) struct EdgeTrampoline {
    pub(crate) label: LabelId,
    pub(crate) target: LabelId,
    pub(crate) moves: Vec<(Allocation, Allocation)>,
}

pub(crate) struct Lowerer {
    pub(crate) ectx: EmitCtx,
    pub(crate) block_labels: BTreeMap<(u32, u32), LabelId>,
    pub(crate) lambda_labels: Vec<LabelId>,
    pub(crate) slot_base: u32,
    pub(crate) spill_base: u32,
    pub(crate) entry: Option<u32>,
    pub(crate) current_func: Option<FunctionCtx>,
    pub(crate) const_vregs: Vec<Option<u64>>,
    pub(crate) edits_by_lambda: BTreeMap<u32, LambdaEditMap>,
    pub(crate) edge_edits_by_lambda: BTreeMap<u32, LambdaEdgeEditMap>,
    pub(crate) forward_branch_blocks: BTreeMap<(u32, u32), u32>,
    pub(crate) allocs_by_lambda: BTreeMap<u32, BTreeMap<cfg_mir::OpId, Vec<Allocation>>>,
    pub(crate) return_result_allocs_by_lambda: BTreeMap<u32, Vec<Allocation>>,
    pub(crate) edge_trampoline_labels: BTreeMap<(u32, cfg_mir::EdgeId), LabelId>,
    pub(crate) edge_trampolines: Vec<EdgeTrampoline>,
    pub(crate) current_inst_allocs: Option<Vec<Allocation>>,
    pub(crate) current_op_id: Option<cfg_mir::OpId>,
    pub(crate) apply_regalloc_edits: bool,
    pub(crate) no_edit_edge_tmp_base: u32,
    pub(crate) edge_args_by_lambda: BTreeMap<u32, BTreeMap<cfg_mir::EdgeId, Vec<cfg_mir::EdgeArg>>>,
}

#[derive(Clone, Copy)]
pub(crate) enum IntrinsicArg {
    VReg { operand_index: usize },
    OutField(u32),
}

// r[impl ir.backends.post-regalloc.branch-test]
// r[impl ir.backends.post-regalloc.shuffle]
pub fn compile(
    program: &cfg_mir::Program,
    alloc: &AllocatedCfgProgram,
    apply_regalloc_edits: bool,
) -> LinearBackendResult {
    let max_spillslots = alloc
        .functions
        .iter()
        .map(|f| f.num_spillslots)
        .max()
        .unwrap_or(0);

    Lowerer::new(program, max_spillslots, alloc, apply_regalloc_edits).run(program)
}

impl Lowerer {
    fn max_edge_move_count(
        program: &cfg_mir::Program,
        alloc: &AllocatedCfgProgram,
        no_edit_mode: bool,
    ) -> u32 {
        if no_edit_mode {
            return program
                .funcs
                .iter()
                .flat_map(|func| func.edges.iter())
                .map(|edge| edge.args.len())
                .max()
                .unwrap_or(0) as u32;
        }

        let mut max_moves = 0usize;
        for func in &alloc.functions {
            let mut by_edge = BTreeMap::<cfg_mir::EdgeId, usize>::new();
            for edge in &func.edge_edits {
                *by_edge.entry(edge.edge).or_default() += 1;
            }
            if let Some(local_max) = by_edge.values().copied().max() {
                max_moves = max_moves.max(local_max);
            }
        }

        max_moves as u32
    }

    fn max_progpoint_move_count(alloc: &AllocatedCfgProgram) -> u32 {
        let mut max_moves = 0usize;
        for func in &alloc.functions {
            let mut before_by_op = BTreeMap::<cfg_mir::OpId, usize>::new();
            let mut after_by_op = BTreeMap::<cfg_mir::OpId, usize>::new();
            for (prog_point, edit) in &func.edits {
                let Edit::Move { from, to } = edit;
                let Some(_) = Self::normalize_edit_move(*from, *to) else {
                    continue;
                };
                match prog_point {
                    cfg_mir::ProgPoint::Before(op) => {
                        *before_by_op.entry(*op).or_default() += 1;
                    }
                    cfg_mir::ProgPoint::After(op) => {
                        *after_by_op.entry(*op).or_default() += 1;
                    }
                    cfg_mir::ProgPoint::Edge(_) => {
                        // Counted by max_edge_move_count.
                    }
                }
            }
            let local_max = before_by_op
                .values()
                .copied()
                .chain(after_by_op.values().copied())
                .max();
            if let Some(local_max) = local_max {
                max_moves = max_moves.max(local_max);
            }
        }
        max_moves as u32
    }

    pub(super) fn new_label_id(&mut self) -> LabelId {
        self.ectx.new_label()
    }

    fn normalize_edit_move(from: Allocation, to: Allocation) -> Option<(Allocation, Allocation)> {
        if from == to || from.is_none() || to.is_none() {
            return None;
        }
        Some((from, to))
    }

    fn regalloc_extra_saved_pairs(alloc: &AllocatedCfgProgram) -> u32 {
        let mut max_pair = None::<u32>;
        let mut observe = |a: Allocation| {
            let Some(reg) = a.as_reg() else {
                return;
            };
            if reg.class() != RegClass::Int {
                return;
            }
            let enc = reg.hw_enc() as u8;
            let pair = match enc {
                23 | 24 => Some(0),
                25 | 26 => Some(1),
                27 | 28 => Some(2),
                _ => None,
            };
            if let Some(pair) = pair {
                max_pair = Some(max_pair.map_or(pair, |cur| cur.max(pair)));
            }
        };

        for func in &alloc.functions {
            for inst_allocs in func.op_allocs.values() {
                for &a in inst_allocs {
                    observe(a);
                }
            }
            for (_, edit) in &func.edits {
                let Edit::Move { from, to } = edit;
                observe(*from);
                observe(*to);
            }
            for edge in &func.edge_edits {
                observe(edge.from);
                observe(edge.to);
            }
            for &a in &func.return_result_allocs {
                observe(a);
            }
        }

        max_pair.map_or(0, |p| p + 1)
    }

    fn new(
        program: &cfg_mir::Program,
        max_spillslots: usize,
        alloc: &AllocatedCfgProgram,
        apply_regalloc_edits: bool,
    ) -> Self {
        let no_edit_mode = !apply_regalloc_edits;
        let required_spillslots = if no_edit_mode {
            max_spillslots.max(program.vreg_count as usize)
        } else {
            max_spillslots
        };
        let max_edge_args = Self::max_edge_move_count(program, alloc, no_edit_mode);
        let max_progpoint_moves = Self::max_progpoint_move_count(alloc);
        let max_parallel_moves = max_edge_args.max(max_progpoint_moves);
        let extra_saved_pairs = Self::regalloc_extra_saved_pairs(alloc);
        let slot_base = BASE_FRAME + extra_saved_pairs * 16;
        let slot_bytes = program.slot_count * 8;
        let spill_base = slot_base + slot_bytes;
        let spill_bytes = required_spillslots as u32 * 8;
        let no_edit_edge_tmp_base = spill_base + spill_bytes;
        let edge_tmp_bytes = max_parallel_moves * 8;
        let extra_stack = slot_bytes + spill_bytes + edge_tmp_bytes + 8;

        let mut ectx = EmitCtx::new_regalloc(extra_stack, extra_saved_pairs);

        let mut block_labels = BTreeMap::new();
        let mut lambda_max = 0usize;
        for func in &program.funcs {
            lambda_max = lambda_max.max(func.lambda_id.index());
            for inst in &func.insts {
                if let LinearOp::CallLambda { target, .. } = &inst.op {
                    lambda_max = lambda_max.max(target.index());
                }
            }
            for block in &func.blocks {
                let key = (func.lambda_id.index() as u32, block.id.0);
                block_labels.insert(key, ectx.new_label());
            }
        }
        let lambda_labels: Vec<LabelId> = (0..=lambda_max).map(|_| ectx.new_label()).collect();

        let mut forward_branch_blocks = BTreeMap::<(u32, u32), u32>::new();
        let mut edge_args_by_lambda =
            BTreeMap::<u32, BTreeMap<cfg_mir::EdgeId, Vec<cfg_mir::EdgeArg>>>::new();
        for func in &program.funcs {
            let lid = func.lambda_id.index() as u32;
            let mut by_edge = BTreeMap::new();
            for edge in &func.edges {
                by_edge.insert(edge.id, edge.args.clone());
            }
            edge_args_by_lambda.insert(lid, by_edge);

            for block in &func.blocks {
                if block.insts.is_empty()
                    && let Some(cfg_mir::Terminator::Branch { edge }) = func.term(block.term)
                {
                    let target = func.edge(*edge).expect("branch edge should exist").to;
                    forward_branch_blocks.insert((lid, block.id.0), target.0);
                }
            }
        }

        let mut edits_by_lambda = BTreeMap::<u32, LambdaEditMap>::new();
        let mut edge_edits_by_lambda = BTreeMap::<u32, LambdaEdgeEditMap>::new();
        let mut allocs_by_lambda = BTreeMap::<u32, BTreeMap<cfg_mir::OpId, Vec<Allocation>>>::new();
        let mut return_result_allocs_by_lambda = BTreeMap::<u32, Vec<Allocation>>::new();

        for func in &alloc.functions {
            let lambda_id = func.lambda_id.index() as u32;
            let lambda_entry = edits_by_lambda.entry(lambda_id).or_default();
            let lambda_edge_entry = edge_edits_by_lambda.entry(lambda_id).or_default();
            let allocs_entry = allocs_by_lambda.entry(lambda_id).or_default();

            return_result_allocs_by_lambda
                .entry(lambda_id)
                .or_insert_with(|| func.return_result_allocs.clone());

            for (prog_point, edit) in &func.edits {
                let Edit::Move { from, to } = edit;
                let Some((from, to)) = Self::normalize_edit_move(*from, *to) else {
                    continue;
                };
                match prog_point {
                    cfg_mir::ProgPoint::Before(op) => {
                        lambda_entry.before.entry(*op).or_default().push((from, to));
                    }
                    cfg_mir::ProgPoint::After(op) => {
                        lambda_entry.after.entry(*op).or_default().push((from, to));
                    }
                    cfg_mir::ProgPoint::Edge(_) => {}
                }
            }

            for edge_edit in &func.edge_edits {
                let Some((from, to)) = Self::normalize_edit_move(edge_edit.from, edge_edit.to)
                else {
                    continue;
                };
                let bucket = match edge_edit.pos {
                    InstPosition::Before => &mut lambda_edge_entry.before,
                    InstPosition::After => &mut lambda_edge_entry.after,
                };
                bucket.entry(edge_edit.edge).or_default().push((from, to));
            }

            for (op_id, inst_allocs) in &func.op_allocs {
                allocs_entry.insert(*op_id, inst_allocs.clone());
            }
        }
        Self {
            ectx,
            block_labels,
            lambda_labels,
            slot_base,
            spill_base,
            entry: None,
            current_func: None,
            const_vregs: vec![None; program.vreg_count as usize],
            edits_by_lambda,
            edge_edits_by_lambda,
            forward_branch_blocks,
            allocs_by_lambda,
            return_result_allocs_by_lambda,
            edge_trampoline_labels: BTreeMap::new(),
            edge_trampolines: Vec::new(),
            current_inst_allocs: None,
            current_op_id: None,
            apply_regalloc_edits,
            no_edit_edge_tmp_base,
            edge_args_by_lambda,
        }
    }

    pub(super) fn slot_off(&self, s: crate::ir::SlotId) -> u32 {
        self.slot_base + (s.index() as u32) * 8
    }

    pub(super) fn block_label(&self, lambda_id: u32, block_id: cfg_mir::BlockId) -> LabelId {
        self.block_labels[&(lambda_id, block_id.0)]
    }

    pub(super) fn spill_off(&self, slot: regalloc2::SpillSlot) -> u32 {
        self.spill_base + (slot.index() as u32) * 8
    }

    // r[impl ir.regalloc.no-boundary-flush]
    pub(super) fn flush_all_vregs(&mut self) {
        self.const_vregs.fill(None);
    }

    pub(super) fn emit_recipe_ops(&mut self, ops: Vec<Op>) {
        self.flush_all_vregs();
        self.ectx.emit_recipe(&Recipe {
            ops,
            label_count: 0,
        });
    }

    pub(super) fn current_alloc(&self, operand_index: usize) -> Allocation {
        self.current_inst_allocs
            .as_ref()
            .and_then(|allocs| allocs.get(operand_index).copied())
            .unwrap_or_else(|| {
                let lambda = self.current_func.as_ref().map(|f| f.lambda_id.index());
                let op_id = self.current_op_id;
                let alloc_len = self
                    .current_inst_allocs
                    .as_ref()
                    .map(|allocs| allocs.len())
                    .unwrap_or(0);
                panic!(
                    "missing regalloc allocation for operand index {operand_index} (lambda={lambda:?}, op_id={op_id:?}, alloc_len={alloc_len})"
                )
            })
    }

    pub(super) fn no_edit_mode(&self) -> bool {
        !self.apply_regalloc_edits
    }

    pub(super) fn canonical_alloc_for_vreg(&self, v: crate::ir::VReg) -> Allocation {
        Allocation::stack(SpillSlot::new(v.index()))
    }

    pub(super) fn canonical_allocs_for_operands(
        &self,
        operands: &[cfg_mir::Operand],
    ) -> Vec<Allocation> {
        operands
            .iter()
            .map(|operand| self.canonical_alloc_for_vreg(operand.vreg))
            .collect()
    }

    fn canonical_allocs_for_terminator(&self, term: &cfg_mir::Terminator) -> Vec<Allocation> {
        match term {
            cfg_mir::Terminator::BranchIf { cond, .. }
            | cfg_mir::Terminator::BranchIfZero { cond, .. } => {
                vec![self.canonical_alloc_for_vreg(*cond)]
            }
            cfg_mir::Terminator::JumpTable { predicate, .. } => {
                vec![self.canonical_alloc_for_vreg(*predicate)]
            }
            cfg_mir::Terminator::Return
            | cfg_mir::Terminator::ErrorExit { .. }
            | cfg_mir::Terminator::Branch { .. } => Vec::new(),
        }
    }

    fn allocs_for_inst(&self, lambda_id: u32, op_id: cfg_mir::OpId) -> Option<Vec<Allocation>> {
        self.allocs_by_lambda
            .get(&lambda_id)
            .and_then(|by_lambda| by_lambda.get(&op_id))
            .cloned()
    }

    fn allocs_for_terminator(
        &self,
        lambda_id: u32,
        op_id: cfg_mir::OpId,
    ) -> Option<Vec<Allocation>> {
        self.allocs_by_lambda
            .get(&lambda_id)
            .and_then(|by_lambda| by_lambda.get(&op_id))
            .cloned()
    }

    fn edge_edit_moves(&self, edge_id: cfg_mir::EdgeId) -> Vec<(Allocation, Allocation)> {
        let Some(lambda_id) = self
            .current_func
            .as_ref()
            .map(|f| f.lambda_id.index() as u32)
        else {
            return Vec::new();
        };

        if self.no_edit_mode() {
            let Some(by_edge) = self.edge_args_by_lambda.get(&lambda_id) else {
                return Vec::new();
            };
            let Some(edge_args) = by_edge.get(&edge_id) else {
                return Vec::new();
            };
            return edge_args
                .iter()
                .filter_map(|arg| {
                    let from = self.canonical_alloc_for_vreg(arg.source);
                    let to = self.canonical_alloc_for_vreg(arg.target);
                    (from != to).then_some((from, to))
                })
                .collect();
        }

        let Some(by_lambda) = self.edge_edits_by_lambda.get(&lambda_id) else {
            return Vec::new();
        };
        let mut moves = Vec::new();
        if let Some(before) = by_lambda.before.get(&edge_id) {
            moves.extend(before.iter().copied());
        }
        if let Some(after) = by_lambda.after.get(&edge_id) {
            moves.extend(after.iter().copied());
        }
        moves
    }

    pub(super) fn const_of(&self, v: crate::ir::VReg) -> Option<u64> {
        self.const_vregs[v.index()]
    }

    pub(super) fn set_const(&mut self, v: crate::ir::VReg, value: Option<u64>) {
        self.const_vregs[v.index()] = value;
    }

    fn emit_inst(&mut self, op: &LinearOp) {
        match op {
            LinearOp::Const { dst, value } => {
                self.emit_load_u64_x9(*value);
                self.emit_store_def_x9(*dst, 0);
                self.set_const(*dst, Some(*value));
            }
            LinearOp::Copy { dst, src } => {
                let from = self.current_alloc(0);
                let to = self.current_alloc(1);
                self.emit_edit_move(from, to);
                self.set_const(*dst, self.const_of(*src));
            }
            LinearOp::BinOp { op, dst, lhs, rhs } => {
                self.emit_binop(*op, *dst, *lhs, *rhs);
            }
            LinearOp::UnaryOp { op, dst, src } => self.emit_unary(*op, *dst, *src),

            LinearOp::BoundsCheck { count } => {
                self.emit_recipe_ops(vec![Op::BoundsCheck { count: *count }]);
            }
            LinearOp::ReadBytes { dst, count } => self.emit_read_bytes(*dst, *count),
            LinearOp::PeekByte { dst } => self.emit_peek_byte(*dst),
            LinearOp::AdvanceCursor { count } => self.ectx.emit_advance_cursor_by(*count),
            LinearOp::AdvanceCursorBy { src } => {
                self.emit_load_use_x9(*src, 0);
                self.ectx.emit.emit_word(
                    aarch64::encode_add_reg(aarch64::Width::X64, Reg::X19, Reg::X19, Reg::X9)
                        .expect("add"),
                );
            }
            LinearOp::SaveCursor { dst } => {
                self.ectx.emit.emit_word(
                    aarch64::encode_mov_reg(aarch64::Width::X64, Reg::X9, Reg::X19).expect("mov"),
                );
                self.emit_store_def_x9(*dst, 0);
                self.set_const(*dst, None);
            }
            LinearOp::RestoreCursor { src } => {
                self.emit_load_use_x9(*src, 0);
                self.ectx.emit.emit_word(
                    aarch64::encode_mov_reg(aarch64::Width::X64, Reg::X19, Reg::X9).expect("mov"),
                );
            }

            LinearOp::WriteToField { src, offset, width } => {
                self.emit_write_to_field(*src, *offset, *width);
            }
            LinearOp::ReadFromField { dst, offset, width } => {
                self.emit_read_from_field(*dst, *offset, *width);
                self.set_const(*dst, None);
            }
            LinearOp::SaveOutPtr { dst } => {
                self.emit_save_out_ptr(*dst);
                self.set_const(*dst, None);
            }
            LinearOp::SetOutPtr { src } => {
                self.emit_set_out_ptr(*src);
            }
            LinearOp::SlotAddr { dst, slot } => {
                self.emit_slot_addr(*dst, *slot);
                self.set_const(*dst, None);
            }
            LinearOp::WriteToSlot { slot, src } => {
                self.emit_load_use_x9(*src, 0);
                let off = self.slot_off(*slot);
                self.ectx.emit.emit_word(
                    aarch64::encode_str_imm(aarch64::Width::X64, Reg::X9, Reg::SP, off)
                        .expect("str"),
                );
            }
            LinearOp::ReadFromSlot { dst, slot } => {
                let off = self.slot_off(*slot);
                self.ectx.emit.emit_word(
                    aarch64::encode_ldr_imm(aarch64::Width::X64, Reg::X9, Reg::SP, off)
                        .expect("ldr"),
                );
                self.emit_store_def_x9(*dst, 0);
                self.set_const(*dst, None);
            }

            LinearOp::CallIntrinsic {
                func,
                args,
                dst,
                field_offset,
            } => {
                self.emit_call_intrinsic(*func, args, *dst, *field_offset);
                if let Some(dst) = dst {
                    self.set_const(*dst, None);
                }
            }
            LinearOp::CallPure { func, args, dst } => {
                self.emit_call_pure(*func, args, *dst);
                self.set_const(*dst, None);
            }

            LinearOp::ErrorExit { code } => {
                self.flush_all_vregs();
                self.ectx.emit_error(*code);
            }

            LinearOp::SimdStringScan { .. } | LinearOp::SimdWhitespaceSkip => {
                panic!("unsupported SIMD op in aarch64 backend");
            }

            LinearOp::CallLambda {
                target,
                args,
                results,
            } => {
                let label = self.lambda_labels[target.index()];
                self.emit_call_lambda(label, args, results);
                for &r in results {
                    self.set_const(r, None);
                }
            }

            LinearOp::FuncStart { .. }
            | LinearOp::FuncEnd
            | LinearOp::Label(_)
            | LinearOp::Branch(_)
            | LinearOp::BranchIf { .. }
            | LinearOp::BranchIfZero { .. }
            | LinearOp::JumpTable { .. } => {
                panic!("structural op {op:?} should not appear as RaInst");
            }
        }
    }

    fn emit_terminator(
        &mut self,
        func: &cfg_mir::Function,
        block: &cfg_mir::Block,
        next_block: Option<&cfg_mir::Block>,
        term_op: cfg_mir::OpId,
    ) {
        let lambda_id = func.lambda_id.index() as u32;
        self.current_op_id = Some(term_op);
        let term = func
            .term(block.term)
            .expect("block terminator should exist in function");

        self.current_inst_allocs = if self.no_edit_mode() {
            Some(self.canonical_allocs_for_terminator(term))
        } else {
            self.allocs_for_terminator(lambda_id, term_op)
        };
        if self.apply_regalloc_edits {
            self.apply_regalloc_edits(term_op, InstPosition::Before);
        }

        match term {
            cfg_mir::Terminator::Return => {}
            cfg_mir::Terminator::ErrorExit { code } => {
                self.flush_all_vregs();
                self.ectx.emit_error(*code);
            }
            cfg_mir::Terminator::Branch { edge } => {
                let target = func
                    .edge(*edge)
                    .expect("branch edge should exist in function")
                    .to;
                let resolved = self.resolve_forwarded_block(lambda_id, target);
                let target_label =
                    self.edge_target_label(*edge, self.block_label(lambda_id, resolved));
                let is_redundant_fallthrough = if self.has_edge_edits(*edge) {
                    false
                } else if let Some(next) = next_block {
                    let resolved_next = self.resolve_forwarded_block(lambda_id, next.id);
                    resolved == resolved_next
                } else {
                    false
                };
                if !is_redundant_fallthrough {
                    self.ectx.emit_branch(target_label);
                }
            }
            cfg_mir::Terminator::BranchIf {
                cond,
                taken,
                fallthrough,
            } => {
                let taken_target_block = func
                    .edge(*taken)
                    .expect("taken edge should exist in function")
                    .to;
                let resolved = self.resolve_forwarded_block(lambda_id, taken_target_block);
                let taken_target =
                    self.edge_target_label(*taken, self.block_label(lambda_id, resolved));
                if let Some(cond_const) = self.const_of(*cond) {
                    if cond_const != 0 {
                        self.ectx.emit_branch(taken_target);
                    } else {
                        self.apply_fallthrough_edge_edits(*fallthrough);
                    }
                } else {
                    self.emit_branch_if(*cond, taken_target, false);
                    self.apply_fallthrough_edge_edits(*fallthrough);
                }
            }
            cfg_mir::Terminator::BranchIfZero {
                cond,
                taken,
                fallthrough,
            } => {
                let taken_target_block = func
                    .edge(*taken)
                    .expect("taken edge should exist in function")
                    .to;
                let resolved = self.resolve_forwarded_block(lambda_id, taken_target_block);
                let taken_target =
                    self.edge_target_label(*taken, self.block_label(lambda_id, resolved));
                if let Some(cond_const) = self.const_of(*cond) {
                    if cond_const == 0 {
                        self.ectx.emit_branch(taken_target);
                    } else {
                        self.apply_fallthrough_edge_edits(*fallthrough);
                    }
                } else {
                    self.emit_branch_if(*cond, taken_target, true);
                    self.apply_fallthrough_edge_edits(*fallthrough);
                }
            }
            cfg_mir::Terminator::JumpTable {
                predicate,
                targets,
                default,
            } => {
                self.emit_jump_table(lambda_id, *predicate, targets, *default, func);
            }
        }

        if self.apply_regalloc_edits {
            self.apply_regalloc_edits(term_op, InstPosition::After);
        }
        self.current_inst_allocs = None;
        self.current_op_id = None;
    }

    fn run(mut self, program: &cfg_mir::Program) -> LinearBackendResult {
        let mut source_line_base = 0u32;
        for func in &program.funcs {
            let lambda_id = func.lambda_id.index() as u32;
            let schedule = func
                .derive_schedule()
                .expect("cfg_mir function should have a valid schedule");

            self.flush_all_vregs();
            let label = self.lambda_labels[func.lambda_id.index()];
            self.ectx.bind_label(label);
            self.ectx.set_source_location(kajit_emit::SourceLocation {
                file: 1,
                line: 1,
                column: 0,
            });
            let (entry_offset, error_exit) = self.ectx.begin_func();
            if func.lambda_id.index() == 0 {
                self.entry = Some(entry_offset);
            }
            self.current_func = Some(FunctionCtx {
                error_exit,
                data_results: func.data_results.clone(),
                lambda_id: func.lambda_id,
            });
            self.emit_store_incoming_lambda_args(&func.data_args);

            for (block_index, block) in func.blocks.iter().enumerate() {
                self.flush_all_vregs();
                let block_label = self.block_label(lambda_id, block.id);
                self.ectx.bind_label(block_label);

                for inst_id in &block.insts {
                    let op_id = cfg_mir::OpId::Inst(*inst_id);
                    let inst = func
                        .inst(*inst_id)
                        .expect("block instruction should exist in function");
                    let source_index = schedule
                        .op_to_index
                        .get(&op_id)
                        .copied()
                        .expect("instruction op should exist in derived schedule");
                    self.ectx.set_source_location(kajit_emit::SourceLocation {
                        file: 1,
                        line: source_line_base + source_index + 1,
                        column: 0,
                    });
                    self.current_op_id = Some(op_id);
                    self.current_inst_allocs = self.allocs_for_inst(lambda_id, op_id);
                    if self.no_edit_mode() {
                        self.current_inst_allocs =
                            Some(self.canonical_allocs_for_operands(&inst.operands));
                    }
                    if self.apply_regalloc_edits {
                        self.apply_regalloc_edits(op_id, InstPosition::Before);
                    }
                    self.emit_inst(&inst.op);
                    if self.apply_regalloc_edits {
                        self.apply_regalloc_edits(op_id, InstPosition::After);
                    }
                    self.current_inst_allocs = None;
                    self.current_op_id = None;
                }

                let term_op = cfg_mir::OpId::Term(block.term);
                let source_index = schedule
                    .op_to_index
                    .get(&term_op)
                    .copied()
                    .expect("terminator op should exist in derived schedule");
                self.ectx.set_source_location(kajit_emit::SourceLocation {
                    file: 1,
                    line: source_line_base + source_index + 1,
                    column: 0,
                });
                let next_block = func.blocks.get(block_index + 1);
                self.emit_terminator(func, block, next_block, term_op);
            }

            self.flush_all_vregs();
            let func_ctx = self
                .current_func
                .take()
                .expect("FuncEnd without active function");
            self.emit_load_lambda_results_to_ret_regs(func_ctx.lambda_id, &func_ctx.data_results);
            self.ectx.end_func(func_ctx.error_exit);
            source_line_base += schedule.op_order.len() as u32;
        }

        self.emit_edge_trampolines();

        let entry = self.entry.expect("missing root FuncStart for lambda 0");
        let buf = self.ectx.finalize();
        let source_map = Some(buf.source_map.clone());
        LinearBackendResult {
            buf,
            entry,
            source_map,
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::compiler;
    use crate::context::{DeserContext, ErrorCode};
    use crate::ir::{IntrinsicFn, IrBuilder, IrOp, Width};
    use crate::linearize::{LinearIr, linearize};
    use facet::Facet;

    #[derive(Debug, Clone, serde::Serialize, serde::Deserialize, Facet, PartialEq)]
    struct ScalarVec {
        values: Vec<u32>,
    }

    fn run_u32_decoder(ir: &LinearIr, input: &[u8]) -> (u32, DeserContext) {
        let deser = compiler::compile_linear_ir_decoder(ir, false);
        let mut out = core::mem::MaybeUninit::<u32>::uninit();
        let mut ctx = DeserContext::from_bytes(input);
        unsafe {
            (deser.func())(out.as_mut_ptr() as *mut u8, &mut ctx);
            (out.assume_init(), ctx)
        }
    }

    fn run_u64_decoder(ir: &LinearIr, input: &[u8]) -> (u64, DeserContext) {
        let deser = compiler::compile_linear_ir_decoder(ir, false);
        let mut out = core::mem::MaybeUninit::<u64>::uninit();
        let mut ctx = DeserContext::from_bytes(input);
        unsafe {
            (deser.func())(out.as_mut_ptr() as *mut u8, &mut ctx);
            (out.assume_init(), ctx)
        }
    }

    fn run_decoder<'input, T: facet::Facet<'input>>(
        ir: &LinearIr,
        input: &'input [u8],
    ) -> (T, DeserContext) {
        let deser = compiler::compile_linear_ir_decoder(ir, false);
        let mut out = core::mem::MaybeUninit::<T>::uninit();
        let mut ctx = DeserContext::from_bytes(input);
        unsafe {
            (deser.func())(out.as_mut_ptr() as *mut u8, &mut ctx);
            (out.assume_init(), ctx)
        }
    }

    fn disasm_bytes(code: &[u8], marker_offset: Option<usize>) -> String {
        let mut out = String::new();

        #[cfg(target_arch = "aarch64")]
        {
            use std::fmt::Write;
            use yaxpeax_arch::{Decoder, U8Reader};
            use yaxpeax_arm::armv8::a64::InstDecoder;

            let decoder = InstDecoder::default();
            let mut reader = U8Reader::new(code);
            let mut offset = 0usize;
            let mut ret_count = 0u32;

            while offset + 4 <= code.len() {
                let prefix = if marker_offset == Some(offset) {
                    "> "
                } else {
                    "  "
                };
                match decoder.decode(&mut reader) {
                    Ok(inst) => {
                        let text = crate::disasm_normalize::normalize_inst(&format!("{inst}"));
                        writeln!(&mut out, "{prefix}{text}").unwrap();
                        if text.trim() == "ret" {
                            ret_count += 1;
                            if ret_count >= 2 {
                                break;
                            }
                        }
                    }
                    Err(e) => {
                        let word = u32::from_le_bytes(code[offset..offset + 4].try_into().unwrap());
                        writeln!(&mut out, "{prefix}<{e}> (0x{word:08x})").unwrap();
                    }
                }
                offset += 4;
            }
        }

        #[cfg(target_arch = "x86_64")]
        {
            use std::fmt::Write;
            use yaxpeax_arch::LengthedInstruction;
            use yaxpeax_arch::{Decoder, U8Reader};
            use yaxpeax_x86::amd64::InstDecoder;

            let decoder = InstDecoder::default();
            let mut reader = U8Reader::new(code);
            let mut offset = 0usize;
            let mut ret_count = 0u32;

            while offset < code.len() {
                let prefix = if marker_offset == Some(offset) {
                    "> "
                } else {
                    "  "
                };
                match decoder.decode(&mut reader) {
                    Ok(inst) => {
                        let len = inst.len().to_const() as usize;
                        let text = crate::disasm_normalize::normalize_inst(&format!("{inst}"));
                        writeln!(&mut out, "{prefix}{text}").unwrap();
                        if text.trim() == "ret" {
                            ret_count += 1;
                            if ret_count >= 2 {
                                break;
                            }
                        }
                        offset += len;
                    }
                    Err(_) => {
                        writeln!(&mut out, "{prefix}<decode error> (0x{:02x})", code[offset])
                            .unwrap();
                        offset += 1;
                    }
                }
            }
        }

        out
    }

    macro_rules! ir_micro_cases {
        (
            $(
                $name:ident => {
                    output: $out_ty:ty,
                    input: $input:expr,
                    expected: $expected:expr,
                    build: |$rb:ident| $build:block
                }
            ),+ $(,)?
        ) => {
            $(
                #[test]
                fn $name() {
                    let mut builder = IrBuilder::new(<$out_ty as facet::Facet>::SHAPE);
                    {
                        let mut $rb = builder.root_region();
                        $build
                    }
                    let mut func = builder.finish();
                crate::ir_passes::run_default_passes(&mut func);
                let lin = linearize(&mut func);
                    let _ = &lin;

                    let (value, ctx): ($out_ty, DeserContext) = run_decoder(&lin, $input);
                    assert_eq!(ctx.error.code, 0);
                    assert_eq!(value, $expected);
                }
            )+
        };
    }

    unsafe extern "C" fn add3_intrinsic(
        _ctx: *mut crate::context::DeserContext,
        a: u64,
        b: u64,
        c: u64,
    ) -> u64 {
        a + b + c
    }

    ir_micro_cases! {
        linear_ir_micro_const_u32 => {
            output: u32,
            input: &[],
            expected: 42u32,
            build: |rb| {
                let v = rb.const_val(42);
                rb.write_to_field(v, 0, Width::W4);
                rb.set_results(&[]);
            }
        },
        linear_ir_micro_read_u32 => {
            output: u32,
            input: &[0x78, 0x56, 0x34, 0x12],
            expected: 0x1234_5678u32,
            build: |rb| {
                rb.bounds_check(4);
                let v = rb.read_bytes(4);
                rb.write_to_field(v, 0, Width::W4);
                rb.set_results(&[]);
            }
        },
        linear_ir_micro_gamma_u32 => {
            output: u32,
            input: &[],
            expected: 20u32,
            build: |rb| {
                let pred = rb.const_val(1);
                rb.gamma(pred, &[], 2, |branch_idx, br| {
                    let v = br.const_val(if branch_idx == 0 { 10 } else { 20 });
                    br.write_to_field(v, 0, Width::W4);
                    br.set_results(&[]);
                });
                rb.set_results(&[]);
            }
        },
        linear_ir_micro_intrinsic_u64 => {
            output: u64,
            input: &[],
            expected: 23u64,
            build: |rb| {
                let a = rb.const_val(11);
                let b = rb.const_val(7);
                let c = rb.const_val(5);
                let out = rb
                    .call_intrinsic(IntrinsicFn(add3_intrinsic as *const () as usize), &[a, b, c], 0, true)
                    .expect("return-value intrinsic should produce output");
                rb.write_to_field(out, 0, Width::W8);
                rb.set_results(&[]);
            }
        }
    }

    #[test]
    fn linear_backend_reads_u32_from_cursor() {
        let mut builder = IrBuilder::new(<u32 as facet::Facet>::SHAPE);
        {
            let mut rb = builder.root_region();
            rb.bounds_check(4);
            let v = rb.read_bytes(4);
            rb.write_to_field(v, 0, Width::W4);
            rb.set_results(&[]);
        }

        let mut func = builder.finish();
        let lin = linearize(&mut func);
        let (value, ctx) = run_u32_decoder(&lin, &[0x78, 0x56, 0x34, 0x12]);

        assert_eq!(ctx.error.code, 0);
        assert_eq!(value, 0x1234_5678);
    }

    #[test]
    fn linear_backend_call_intrinsic_zero_arg_return_value() {
        unsafe extern "C" fn return_300(_ctx: *mut crate::context::DeserContext) -> u64 {
            300
        }

        let mut builder = IrBuilder::new(<u32 as facet::Facet>::SHAPE);
        {
            let mut rb = builder.root_region();
            let v = rb
                .call_intrinsic(IntrinsicFn(return_300 as *const () as usize), &[], 0, true)
                .expect("intrinsic should produce output");
            rb.write_to_field(v, 0, Width::W4);
            rb.set_results(&[]);
        }

        let mut func = builder.finish();
        let lin = linearize(&mut func);
        let (value, ctx) = run_u32_decoder(&lin, &[]);

        assert_eq!(ctx.error.code, 0);
        assert_eq!(value, 300);
    }

    #[test]
    fn linear_backend_bounds_check_sets_eof() {
        let mut builder = IrBuilder::new(<u32 as facet::Facet>::SHAPE);
        {
            let mut rb = builder.root_region();
            rb.bounds_check(4);
            let v = rb.read_bytes(4);
            rb.write_to_field(v, 0, Width::W4);
            rb.set_results(&[]);
        }

        let mut func = builder.finish();
        let lin = linearize(&mut func);
        let (_value, ctx) = run_u32_decoder(&lin, &[0x01, 0x02]);

        assert_eq!(ctx.error.code, ErrorCode::UnexpectedEof as u32);
    }

    #[test]
    fn linear_backend_two_way_gamma_branch() {
        let mut builder = IrBuilder::new(<u32 as facet::Facet>::SHAPE);
        {
            let mut rb = builder.root_region();
            let pred = rb.const_val(1);
            rb.gamma(pred, &[], 2, |branch_idx, br| {
                let value = if branch_idx == 0 { 10 } else { 20 };
                let v = br.const_val(value);
                br.write_to_field(v, 0, Width::W4);
                br.set_results(&[]);
            });
            rb.set_results(&[]);
        }

        let mut func = builder.finish();
        let lin = linearize(&mut func);
        let (value, ctx) = run_u32_decoder(&lin, &[]);

        assert_eq!(ctx.error.code, 0);
        assert_eq!(value, 20);
    }

    #[test]
    fn linear_backend_jump_table_gamma_branch() {
        let mut builder = IrBuilder::new(<u32 as facet::Facet>::SHAPE);
        {
            let mut rb = builder.root_region();
            let pred = rb.const_val(2);
            rb.gamma(pred, &[], 3, |branch_idx, br| {
                let value = match branch_idx {
                    0 => 111,
                    1 => 222,
                    2 => 333,
                    _ => unreachable!(),
                };
                let v = br.const_val(value);
                br.write_to_field(v, 0, Width::W4);
                br.set_results(&[]);
            });
            rb.set_results(&[]);
        }

        let mut func = builder.finish();
        let lin = linearize(&mut func);
        let (value, ctx) = run_u32_decoder(&lin, &[]);

        assert_eq!(ctx.error.code, 0);
        assert_eq!(value, 333);
    }

    #[test]
    fn linear_backend_call_intrinsic_with_args_return_value() {
        unsafe extern "C" fn add3(
            _ctx: *mut crate::context::DeserContext,
            a: u64,
            b: u64,
            c: u64,
        ) -> u64 {
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
                .expect("return-value intrinsic should produce output");
            rb.write_to_field(out, 0, Width::W8);
            rb.set_results(&[]);
        }

        let mut func = builder.finish();
        let lin = linearize(&mut func);
        let (value, ctx) = run_u64_decoder(&lin, &[]);

        assert_eq!(ctx.error.code, 0);
        assert_eq!(value, 23);
    }

    #[test]
    fn linear_backend_call_intrinsic_with_args_and_out_ptr() {
        unsafe extern "C" fn write_scaled_sum(
            _ctx: *mut crate::context::DeserContext,
            x: u64,
            y: u64,
            out: *mut u64,
        ) {
            unsafe { *out = x * 10 + y };
        }

        let mut builder = IrBuilder::new(<u64 as facet::Facet>::SHAPE);
        {
            let mut rb = builder.root_region();
            let x = rb.const_val(9);
            let y = rb.const_val(4);
            rb.call_intrinsic(
                IntrinsicFn(write_scaled_sum as *const () as usize),
                &[x, y],
                0,
                false,
            );
            rb.set_results(&[]);
        }

        let mut func = builder.finish();
        let lin = linearize(&mut func);
        let (value, ctx) = run_u64_decoder(&lin, &[]);

        assert_eq!(ctx.error.code, 0);
        assert_eq!(value, 94);
    }

    #[test]
    fn linear_backend_call_lambda_with_data_args_and_results() {
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
        let (value, ctx) = run_u64_decoder(&lin, &[]);

        assert_eq!(ctx.error.code, 0);
        assert_eq!(value, 42);
    }

    #[test]
    fn linear_backend_vec_u32_matches_serde() {
        let expected = ScalarVec {
            values: (0..2048).map(|i| i as u32).collect(),
        };
        let bytes = postcard::to_allocvec(&expected).expect("serialize vec");

        let ir = crate::compile_decoder(ScalarVec::SHAPE, &crate::postcard::KajitPostcard);

        let ir_out = crate::deserialize::<ScalarVec>(&ir, &bytes).expect("ir decode");
        let serde_out = postcard::from_bytes::<ScalarVec>(&bytes).expect("serde decode");

        assert_eq!(ir_out, expected);
        assert_eq!(serde_out, expected);
    }
}
