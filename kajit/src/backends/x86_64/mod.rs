#![allow(clippy::useless_conversion)]

use kajit_emit::x64::{self, LabelId, Mem};
use regalloc2::{Allocation, Edit, InstPosition, PReg, RegClass};
use std::collections::BTreeMap;

use crate::arch::{BASE_FRAME, EmitCtx};
use crate::ir::Width;
use crate::ir_backend::{
    BackendCodeRange, BackendDebugInfo, BackendOpDebugInfo, LinearBackendResult,
};
use crate::linearize::{BinOpKind, LinearOp, UnaryOpKind};
use crate::regalloc_engine::{AllocatedCfgProgram, cfg_mir};

mod alloc;
mod calls;
mod edits;
mod emit;

pub mod regalloc3_backend;

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
    pub(crate) source_location: kajit_emit::SourceLocation,
}

pub(crate) struct Lowerer {
    pub(crate) ectx: EmitCtx,
    /// LabelId for each block, indexed by (lambda_index, block_id).
    pub(crate) block_labels: BTreeMap<(u32, u32), LabelId>,
    pub(crate) lambda_labels: Vec<LabelId>,
    pub(crate) slot_base: u32,
    pub(crate) spill_base: u32,
    pub(crate) entry: Option<u32>,
    pub(crate) current_func: Option<FunctionCtx>,
    pub(crate) const_vregs: Vec<Option<u64>>,
    pub(crate) edits_by_lambda: BTreeMap<u32, LambdaEditMap>,
    pub(crate) edge_edits_by_lambda: BTreeMap<u32, LambdaEdgeEditMap>,
    /// Forward-branch blocks: blocks with no instructions whose terminator is a
    /// plain Branch. Maps (lambda_id, block_id) → target block_id.
    pub(crate) forward_branch_blocks: BTreeMap<(u32, u32), u32>,
    pub(crate) allocs_by_lambda: BTreeMap<u32, BTreeMap<cfg_mir::OpId, Vec<Allocation>>>,
    pub(crate) return_result_allocs_by_lambda: BTreeMap<u32, Vec<Allocation>>,
    pub(crate) edge_trampoline_labels: BTreeMap<(u32, cfg_mir::EdgeId), LabelId>,
    pub(crate) edge_trampolines: Vec<EdgeTrampoline>,
    pub(crate) current_inst_allocs: Option<Vec<Allocation>>,
    pub(crate) parallel_move_tmp_base: u32,
    pub(crate) backend_debug_info: BackendDebugInfo,
}

#[derive(Clone, Copy)]
pub(crate) enum IntrinsicArg {
    VReg { operand_index: usize },
    OutField(u32),
}

pub fn compile(program: &cfg_mir::Program, alloc: &AllocatedCfgProgram) -> LinearBackendResult {
    let max_spillslots = alloc
        .functions
        .iter()
        .map(|f| f.num_spillslots)
        .max()
        .unwrap_or(0);

    Lowerer::new(program, max_spillslots, alloc).run(program)
}

fn build_debug_line_maps(
    program: &cfg_mir::Program,
) -> (BTreeMap<(u32, cfg_mir::OpId), u32>, BTreeMap<u32, u32>) {
    let mut line_by_lambda_op = BTreeMap::<(u32, cfg_mir::OpId), u32>::new();
    let mut first_line_by_lambda = BTreeMap::<u32, u32>::new();
    let mut next_line = 1u32;
    for func in &program.funcs {
        let lambda_id = func.lambda_id.index() as u32;
        let mut first_line = None::<u32>;
        for block in func.live_blocks() {
            for inst_id in &block.insts {
                let op_id = cfg_mir::OpId::Inst(*inst_id);
                line_by_lambda_op.insert((lambda_id, op_id), next_line);
                if first_line.is_none() {
                    first_line = Some(next_line);
                }
                next_line += 1;
            }
            let term_op = cfg_mir::OpId::Term(block.term);
            line_by_lambda_op.insert((lambda_id, term_op), next_line);
            if first_line.is_none() {
                first_line = Some(next_line);
            }
            next_line += 1;
        }
        first_line_by_lambda.insert(lambda_id, first_line.unwrap_or(1));
    }
    (line_by_lambda_op, first_line_by_lambda)
}

impl Lowerer {
    fn max_parallel_move_count(alloc: &AllocatedCfgProgram) -> u32 {
        let mut max_moves = 0usize;
        for func in &alloc.functions {
            let mut by_progpoint = BTreeMap::<(cfg_mir::OpId, u8), usize>::new();
            let mut by_edge = BTreeMap::<cfg_mir::EdgeId, usize>::new();
            for (prog_point, edit) in &func.edits {
                let Edit::Move { from, to } = edit;
                let Some(_) = Self::normalize_edit_move(*from, *to) else {
                    continue;
                };
                let (op_id, pos) = match prog_point {
                    cfg_mir::ProgPoint::Before(op) => (*op, 0u8),
                    cfg_mir::ProgPoint::After(op) => (*op, 1u8),
                    cfg_mir::ProgPoint::Edge(_) => continue,
                };
                *by_progpoint.entry((op_id, pos)).or_default() += 1;
            }
            for edge in &func.edge_edits {
                let Some(_) = Self::normalize_edit_move(edge.from, edge.to) else {
                    continue;
                };
                *by_edge.entry(edge.edge).or_default() += 1;
            }
            if let Some(local_max) = by_progpoint.values().copied().max() {
                max_moves = max_moves.max(local_max);
            }
            if let Some(local_max) = by_edge.values().copied().max() {
                max_moves = max_moves.max(local_max);
            }
        }
        max_moves as u32
    }

    fn normalize_edit_move(from: Allocation, to: Allocation) -> Option<(Allocation, Allocation)> {
        if from == to || from.is_none() || to.is_none() {
            return None;
        }
        Some((from, to))
    }

    fn new(program: &cfg_mir::Program, max_spillslots: usize, alloc: &AllocatedCfgProgram) -> Self {
        let slot_base = BASE_FRAME;
        let slot_bytes = program.slot_count * 8;
        let spill_base = slot_base + slot_bytes;
        let spill_bytes = max_spillslots as u32 * 8;
        let parallel_move_tmp_base = spill_base + spill_bytes;
        let parallel_move_tmp_bytes = Self::max_parallel_move_count(alloc) * 8;
        let extra_stack = slot_bytes + spill_bytes + parallel_move_tmp_bytes + 8;

        let mut ectx = EmitCtx::new(extra_stack);

        let mut block_labels = BTreeMap::new();
        let mut lambda_max = 0usize;
        for func in &program.funcs {
            lambda_max = lambda_max.max(func.lambda_id.index());
            for inst in &func.insts {
                if let LinearOp::CallLambda { target, .. } = &inst.op {
                    lambda_max = lambda_max.max(target.index());
                }
            }
            for block in func.live_blocks() {
                let key = (func.lambda_id.index() as u32, block.id.0);
                block_labels.insert(key, ectx.new_label());
            }
        }
        let lambda_labels: Vec<LabelId> = (0..=lambda_max).map(|_| ectx.new_label()).collect();

        let mut forward_branch_blocks = BTreeMap::<(u32, u32), u32>::new();
        for func in &program.funcs {
            let lid = func.lambda_id.index() as u32;
            for block in func.live_blocks() {
                if block.insts.is_empty()
                    && block.params.is_empty()
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
                    cfg_mir::ProgPoint::Edge(_) => {
                        // materialized in edge_edits below.
                    }
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
            parallel_move_tmp_base,
            backend_debug_info: BackendDebugInfo::default(),
        }
    }

    fn record_debug_op_range(
        &mut self,
        lambda_id: u32,
        op_id: cfg_mir::OpId,
        line: u32,
        start: u32,
        end: u32,
    ) {
        if end <= start {
            return;
        }
        self.backend_debug_info.op_infos.push(BackendOpDebugInfo {
            lambda_id,
            op_id,
            line,
            code_ranges: vec![BackendCodeRange {
                start_offset: start,
                end_offset: end,
            }],
        });
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

    pub(super) fn flush_all_vregs(&mut self) {
        self.const_vregs.fill(None);
    }

    pub(super) fn emit_bounds_check(&mut self, count: u32) {
        self.flush_all_vregs();
        self.ectx.emit_bounds_check(count);
    }

    pub(super) fn current_alloc(&self, operand_index: usize) -> Allocation {
        self.current_inst_allocs
            .as_ref()
            .and_then(|allocs| allocs.get(operand_index).copied())
            .unwrap_or_else(|| {
                panic!("missing regalloc allocation for operand index {operand_index}")
            })
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
                self.ectx
                    .emit
                    .emit_with(|buf| x64::encode_mov_r64_imm64(10, *value, buf))
                    .expect("mov");
                self.emit_store_def_r10(*dst, 0);
                self.set_const(*dst, Some(*value));
            }
            LinearOp::DataAddr { dst, blob_id } => {
                let sentinel = 0xDEAD_DA7A_0000_0000u64 | *blob_id as u64;
                self.ectx
                    .emit
                    .emit_with(|buf| x64::encode_mov_r64_imm64(10, sentinel, buf))
                    .expect("mov");
                self.emit_store_def_r10(*dst, 0);
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
                self.emit_bounds_check(*count);
            }
            LinearOp::ReadBytes { dst, count } => self.emit_read_bytes(*dst, *count),
            LinearOp::PeekByte { dst } => self.emit_peek_byte(*dst),
            LinearOp::AdvanceCursor { count } => {
                self.ectx.emit_advance_cursor_by(*count);
            }
            LinearOp::AdvanceCursorBy { src } => {
                self.emit_load_use_r10(*src, 0);
                self.ectx
                    .emit
                    .emit_with(|buf| x64::encode_add_r64_r64(12, 10, buf))
                    .expect("add");
            }
            LinearOp::SaveCursor { dst } => {
                self.ectx
                    .emit
                    .emit_with(|buf| x64::encode_mov_r64_r64(10, 12, buf))
                    .expect("mov");
                self.emit_store_def_r10(*dst, 0);
                self.set_const(*dst, None);
            }
            LinearOp::SaveInputEnd { dst } => {
                self.ectx
                    .emit
                    .emit_with(|buf| x64::encode_mov_r64_r64(10, 13, buf))
                    .expect("mov");
                self.emit_store_def_r10(*dst, 0);
                self.set_const(*dst, None);
            }
            LinearOp::RestoreCursor { src } => {
                self.emit_load_use_r10(*src, 0);
                self.ectx
                    .emit
                    .emit_with(|buf| x64::encode_mov_r64_r64(12, 10, buf))
                    .expect("mov");
            }

            LinearOp::WriteToField { src, offset, width } => {
                self.emit_write_to_field(*src, *offset, *width);
            }
            LinearOp::ReadFromField { dst, offset, width } => {
                self.emit_read_from_field(*dst, *offset, *width);
            }
            LinearOp::SaveOutPtr { dst } => {
                self.emit_save_out_ptr(*dst);
            }
            LinearOp::SetOutPtr { src } => {
                self.emit_set_out_ptr(*src);
            }
            LinearOp::SlotAddr { dst, slot } => {
                self.emit_slot_addr(*dst, *slot);
            }
            LinearOp::StoreToAddr { addr, src, width } => {
                self.emit_store_to_addr(*addr, *src, *width);
            }
            LinearOp::LoadFromAddr { dst, addr, width } => {
                self.emit_load_from_addr(*dst, *addr, *width);
            }
            LinearOp::WriteToSlot { slot, src } => {
                self.emit_load_use_r10(*src, 0);
                let off = self.slot_off(*slot) as i32;
                self.ectx
                    .emit
                    .emit_with(|buf| x64::encode_mov_m_r64(Mem { base: 4, disp: off }, 10, buf))
                    .expect("mov");
            }
            LinearOp::ReadFromSlot { dst, slot } => {
                let off = self.slot_off(*slot) as i32;
                self.ectx
                    .emit
                    .emit_with(|buf| x64::encode_mov_r64_m(10, Mem { base: 4, disp: off }, buf))
                    .expect("mov");
                self.emit_store_def_r10(*dst, 0);
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
            LinearOp::CallPure { func, args, dst } | LinearOp::CallEffect { func, args, dst } => {
                self.emit_call_pure(*func, args, *dst);
                self.set_const(*dst, None);
            }

            LinearOp::ErrorExit { code } => {
                self.flush_all_vregs();
                self.ectx.emit_error(*code);
            }

            LinearOp::SimdStringScan { .. } | LinearOp::SimdWhitespaceSkip => {
                panic!("unsupported SIMD op in linear backend adapter");
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
            | LinearOp::Branch { .. }
            | LinearOp::BranchIf { .. }
            | LinearOp::BranchIfZero { .. }
            | LinearOp::JumpTable { .. } => {
                panic!("structural op {op:?} should not appear as a CFG-MIR instruction");
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
        let term = func
            .term(block.term)
            .expect("block terminator should exist in function");

        self.current_inst_allocs = self
            .allocs_by_lambda
            .get(&lambda_id)
            .and_then(|by_lambda| by_lambda.get(&term_op))
            .cloned();
        self.apply_regalloc_edits(term_op, InstPosition::Before);

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
                let target = func
                    .edge(*taken)
                    .expect("taken edge should exist in function")
                    .to;
                let resolved = self.resolve_forwarded_block(lambda_id, target);
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
                let target = func
                    .edge(*taken)
                    .expect("taken edge should exist in function")
                    .to;
                let resolved = self.resolve_forwarded_block(lambda_id, target);
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

        self.apply_regalloc_edits(term_op, InstPosition::After);
        self.current_inst_allocs = None;
    }

    // r[impl ir.backends.post-regalloc.branch-test]
    // r[impl ir.backends.post-regalloc.shuffle]
    fn run(mut self, program: &cfg_mir::Program) -> LinearBackendResult {
        let (line_by_lambda_op, first_line_by_lambda) = build_debug_line_maps(program);
        for func in &program.funcs {
            let lambda_id = func.lambda_id.index() as u32;

            self.flush_all_vregs();
            let label = self.lambda_labels[func.lambda_id.index()];
            self.ectx.bind_label(label);
            self.ectx.set_source_location(kajit_emit::SourceLocation {
                file: 1,
                line: first_line_by_lambda.get(&lambda_id).copied().unwrap_or(1),
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
                // Skip dead blocks (tombstones from block merging)
                if block.dead {
                    continue;
                }

                self.flush_all_vregs();
                let block_label = self.block_label(lambda_id, block.id);
                self.ectx.bind_label(block_label);

                for inst_id in &block.insts {
                    let op_id = cfg_mir::OpId::Inst(*inst_id);
                    let inst = func
                        .inst(*inst_id)
                        .expect("block instruction should exist in function");
                    let line = line_by_lambda_op
                        .get(&(lambda_id, op_id))
                        .copied()
                        .expect("instruction op should exist in debug line map");
                    self.ectx.set_source_location(kajit_emit::SourceLocation {
                        file: 1,
                        line,
                        column: 0,
                    });
                    self.current_inst_allocs = self
                        .allocs_by_lambda
                        .get(&lambda_id)
                        .and_then(|by_lambda| by_lambda.get(&op_id))
                        .cloned();
                    let start_offset = self.ectx.emit.current_offset();
                    self.apply_regalloc_edits(op_id, InstPosition::Before);
                    self.emit_inst(&inst.op);
                    self.apply_regalloc_edits(op_id, InstPosition::After);
                    let end_offset = self.ectx.emit.current_offset();
                    self.record_debug_op_range(lambda_id, op_id, line, start_offset, end_offset);
                    self.current_inst_allocs = None;
                }

                let term_op = cfg_mir::OpId::Term(block.term);
                let line = line_by_lambda_op
                    .get(&(lambda_id, term_op))
                    .copied()
                    .expect("terminator op should exist in debug line map");
                self.ectx.set_source_location(kajit_emit::SourceLocation {
                    file: 1,
                    line,
                    column: 0,
                });
                let next_block = func.blocks.get(block_index + 1);
                let start_offset = self.ectx.emit.current_offset();
                self.emit_terminator(func, block, next_block, term_op);
                let end_offset = self.ectx.emit.current_offset();
                self.record_debug_op_range(lambda_id, term_op, line, start_offset, end_offset);
            }

            self.flush_all_vregs();
            let func_ctx = self
                .current_func
                .take()
                .expect("FuncEnd without active function");
            self.emit_load_lambda_results_to_ret_regs(func_ctx.lambda_id, &func_ctx.data_results);
            self.ectx.end_func(func_ctx.error_exit);
        }

        self.emit_edge_trampolines();

        let entry = self.entry.expect("missing root FuncStart for lambda 0");
        let buf = self.ectx.finalize();
        let source_map = Some(buf.source_map.clone());
        LinearBackendResult {
            buf,
            entry,
            source_map,
            backend_debug_info: Some(self.backend_debug_info),
            intrinsic_call_sites: Vec::new(),
            data_relocs: Vec::new(),
        }
    }
}
