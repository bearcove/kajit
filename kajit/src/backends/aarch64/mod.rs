#![allow(clippy::useless_conversion)]

use kajit_emit::aarch64::{self, LabelId, Reg};
use regalloc2::{Allocation, Edit, InstPosition, PReg, RegClass, SpillSlot};
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

/// Tracks a bit test pattern: `(src & (1 << bit)) == 0` or `!= 0`
#[derive(Clone, Copy, Debug)]
#[allow(dead_code)]
pub(crate) struct BitTestInfo {
    /// The source vreg being tested
    pub(crate) src: crate::ir::VReg,
    /// The bit position (0-63)
    pub(crate) bit: u8,
    /// true if this is `== 0` (use tbz), false if `!= 0` (use tbnz)
    pub(crate) test_zero: bool,
}

/// Tracks a masked value: `src & mask` where mask is a power of 2
#[derive(Clone, Copy, Debug)]
pub(crate) struct MaskedValueInfo {
    /// The source vreg
    pub(crate) src: crate::ir::VReg,
    /// The bit position of the single set bit in the mask
    pub(crate) bit: u8,
}

pub(crate) struct Lowerer<'a> {
    pub(crate) ectx: EmitCtx,
    pub(crate) block_labels: BTreeMap<(u32, u32), LabelId>,
    pub(crate) lambda_labels: Vec<LabelId>,
    pub(crate) slot_base: u32,
    pub(crate) spill_base: u32,
    pub(crate) entry: Option<u32>,
    pub(crate) current_func: Option<FunctionCtx>,
    pub(crate) const_vregs: Vec<Option<u64>>,
    /// Tracks vregs that are the result of `src & (1 << bit)`
    pub(crate) masked_vregs: Vec<Option<MaskedValueInfo>>,
    /// Tracks vregs that are the result of bit test comparisons
    pub(crate) bit_test_vregs: Vec<Option<BitTestInfo>>,
    pub(crate) edits_by_lambda: BTreeMap<u32, LambdaEditMap>,
    pub(crate) edge_edits_by_lambda: BTreeMap<u32, LambdaEdgeEditMap>,
    pub(crate) forward_branch_blocks: BTreeMap<(u32, u32), u32>,
    pub(crate) allocs_by_lambda: BTreeMap<u32, BTreeMap<cfg_mir::OpId, Vec<Allocation>>>,
    pub(crate) return_result_allocs_by_lambda: BTreeMap<u32, Vec<Allocation>>,
    pub(crate) edge_trampoline_labels: BTreeMap<(u32, cfg_mir::EdgeId), LabelId>,
    pub(crate) edge_trampolines: Vec<EdgeTrampoline>,
    pub(crate) current_inst_allocs: Option<Vec<Allocation>>,
    pub(crate) current_inst_operands: Option<Vec<cfg_mir::Operand>>,
    pub(crate) current_op_id: Option<cfg_mir::OpId>,
    pub(crate) apply_regalloc_edits: bool,
    pub(crate) no_edit_edge_tmp_base: u32,
    pub(crate) edge_args_by_lambda: BTreeMap<u32, BTreeMap<cfg_mir::EdgeId, Vec<cfg_mir::EdgeArg>>>,
    pub(crate) backend_debug_info: BackendDebugInfo,
    pub(crate) intrinsic_registry: Option<&'a crate::ir::IntrinsicRegistry>,
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
    intrinsic_registry: Option<&crate::ir::IntrinsicRegistry>,
) -> LinearBackendResult {
    let max_spillslots = alloc
        .functions
        .iter()
        .map(|f| f.num_spillslots)
        .max()
        .unwrap_or(0);

    Lowerer::new(
        program,
        max_spillslots,
        alloc,
        apply_regalloc_edits,
        intrinsic_registry,
    )
    .run(program)
}

pub(crate) fn build_debug_line_maps(
    program: &cfg_mir::Program,
) -> (BTreeMap<(u32, cfg_mir::OpId), u32>, BTreeMap<u32, u32>) {
    let mut line_by_lambda_op = BTreeMap::<(u32, cfg_mir::OpId), u32>::new();
    let mut first_line_by_lambda = BTreeMap::<u32, u32>::new();
    let mut next_line = 1u32;
    for func in &program.funcs {
        let lambda_id = func.lambda_id.index() as u32;
        let mut first_line = None::<u32>;
        for block in &func.blocks {
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

impl Lowerer<'_> {
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

    fn max_call_move_count(program: &cfg_mir::Program) -> u32 {
        let mut max_moves = 0usize;
        for func in &program.funcs {
            for inst in &func.insts {
                let args_len = match &inst.op {
                    LinearOp::CallIntrinsic { args, .. } => args.len(),
                    LinearOp::CallPure { args, .. } => args.len(),
                    LinearOp::CallLambda { args, .. } => args.len(),
                    _ => 0,
                };
                max_moves = max_moves.max(args_len);
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

    fn new<'a>(
        program: &cfg_mir::Program,
        max_spillslots: usize,
        alloc: &AllocatedCfgProgram,
        apply_regalloc_edits: bool,
        intrinsic_registry: Option<&'a crate::ir::IntrinsicRegistry>,
    ) -> Lowerer<'a> {
        let no_edit_mode = !apply_regalloc_edits;
        let required_spillslots = if no_edit_mode {
            max_spillslots.max(program.vreg_count as usize)
        } else {
            max_spillslots
        };
        let max_edge_args = Self::max_edge_move_count(program, alloc, no_edit_mode);
        let max_progpoint_moves = Self::max_progpoint_move_count(alloc);
        let max_call_moves = Self::max_call_move_count(program);
        let max_parallel_moves = max_edge_args.max(max_progpoint_moves).max(max_call_moves);
        let extra_saved_pairs = Self::regalloc_extra_saved_pairs(alloc);
        let slot_base = BASE_FRAME + extra_saved_pairs * 16;
        let slot_bytes = program.slot_count * 8;
        let spill_base = slot_base + slot_bytes;
        let spill_bytes = required_spillslots as u32 * 8;
        let no_edit_edge_tmp_base = spill_base + spill_bytes;
        let edge_tmp_bytes = max_parallel_moves * 8;
        let extra_stack = slot_bytes + spill_bytes + edge_tmp_bytes + 8;

        let mut ectx = EmitCtx::new_regalloc(extra_stack, extra_saved_pairs, false);

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
        Lowerer {
            ectx,
            block_labels,
            lambda_labels,
            slot_base,
            spill_base,
            entry: None,
            current_func: None,
            const_vregs: vec![None; program.vreg_count as usize],
            masked_vregs: vec![None; program.vreg_count as usize],
            bit_test_vregs: vec![None; program.vreg_count as usize],
            edits_by_lambda,
            edge_edits_by_lambda,
            forward_branch_blocks,
            allocs_by_lambda,
            return_result_allocs_by_lambda,
            edge_trampoline_labels: BTreeMap::new(),
            edge_trampolines: Vec::new(),
            current_inst_allocs: None,
            current_inst_operands: None,
            current_op_id: None,
            apply_regalloc_edits,
            no_edit_edge_tmp_base,
            edge_args_by_lambda,
            backend_debug_info: BackendDebugInfo::default(),
            intrinsic_registry,
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

    // r[impl ir.regalloc.no-boundary-flush]
    pub(super) fn flush_all_vregs(&mut self) {
        // Note: We intentionally do NOT clear const_vregs here.
        // Const values are static and must persist across block boundaries
        // for immediate-only const optimization to work.
        // const_vregs is cleared at function start via flush_all_vregs_and_consts.
    }

    pub(super) fn flush_all_vregs_and_consts(&mut self) {
        self.const_vregs.fill(None);
        self.masked_vregs.fill(None);
        self.bit_test_vregs.fill(None);
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

    /// Look up allocation by vreg. Returns None if the vreg has no operand
    /// (e.g., immediate-only consts that were optimized away).
    pub(super) fn alloc_for_vreg(&self, v: crate::ir::VReg) -> Option<Allocation> {
        let operands = self.current_inst_operands.as_ref()?;
        let allocs = self.current_inst_allocs.as_ref()?;

        for (i, operand) in operands.iter().enumerate() {
            if operand.vreg == v {
                return allocs.get(i).copied();
            }
        }
        None
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

    /// Build operands list for a terminator (for vreg-based allocation lookup).
    fn operands_for_terminator(&self, term: &cfg_mir::Terminator) -> Vec<cfg_mir::Operand> {
        match term {
            cfg_mir::Terminator::BranchIf { cond, .. }
            | cfg_mir::Terminator::BranchIfZero { cond, .. } => {
                vec![cfg_mir::Operand {
                    vreg: *cond,
                    kind: cfg_mir::OperandKind::Use,
                    class: cfg_mir::RegClass::Gpr,
                    fixed: None,
                }]
            }
            cfg_mir::Terminator::JumpTable { predicate, .. } => {
                vec![cfg_mir::Operand {
                    vreg: *predicate,
                    kind: cfg_mir::OperandKind::Use,
                    class: cfg_mir::RegClass::Gpr,
                    fixed: None,
                }]
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

    pub(super) fn masked_value_of(&self, v: crate::ir::VReg) -> Option<MaskedValueInfo> {
        self.masked_vregs[v.index()]
    }

    pub(super) fn set_masked_value(&mut self, v: crate::ir::VReg, info: Option<MaskedValueInfo>) {
        self.masked_vregs[v.index()] = info;
    }

    #[allow(dead_code)]
    pub(super) fn bit_test_of(&self, v: crate::ir::VReg) -> Option<BitTestInfo> {
        self.bit_test_vregs[v.index()]
    }

    #[allow(dead_code)]
    pub(super) fn set_bit_test(&mut self, v: crate::ir::VReg, info: Option<BitTestInfo>) {
        self.bit_test_vregs[v.index()] = info;
    }

    fn emit_inst(&mut self, inst: &cfg_mir::Inst) {
        match &inst.op {
            LinearOp::Const { dst, value } => {
                // If the Const has no operands, it's an immediate-only const.
                // Just record the value for const_of() lookups, no code emission needed.
                if inst.operands.is_empty() {
                    if std::env::var("KAJIT_DEBUG_CONST").is_ok() {
                        eprintln!(
                            "[const] immediate-only const: dst={:?}, value={}, inst_id={:?}",
                            dst, value, inst.id
                        );
                    }
                    self.set_const(*dst, Some(*value));
                    return;
                }
                // Load directly into allocated register when possible
                let dst_alloc = self.alloc_for_vreg(*dst);
                if let Some(alloc) = dst_alloc
                    && let Some(reg) = alloc.as_reg()
                    && reg.class() == regalloc2::RegClass::Int
                {
                    self.emit_load_u64_reg(Reg::from_raw(reg.hw_enc() as u8), *value);
                } else {
                    self.emit_load_u64_x9(*value);
                    self.emit_store_def_x9(*dst, 0);
                }
                self.set_const(*dst, Some(*value));
            }
            LinearOp::Copy { dst, src } => {
                let from = self
                    .alloc_for_vreg(*src)
                    .expect("copy src should have alloc");
                let to = self
                    .alloc_for_vreg(*dst)
                    .expect("copy dst should have alloc");
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
            LinearOp::AdvanceCursor { count } => self.ectx.emit_advance_cursor_by(*count),
            LinearOp::AdvanceCursorBy { src } => {
                self.emit_load_use_x9(*src, 0);
                self.ectx
                    .emit
                    .emit_add_reg(aarch64::Width::X64, Reg::X19, Reg::X19, Reg::X9)
                    .expect("add");
            }
            LinearOp::SaveCursor { dst } => {
                self.ectx
                    .emit
                    .emit_mov_reg(aarch64::Width::X64, Reg::X9, Reg::X19)
                    .expect("mov");
                self.emit_store_def_x9(*dst, 0);
                self.set_const(*dst, None);
            }
            LinearOp::SaveInputEnd { dst } => {
                self.ectx
                    .emit
                    .emit_mov_reg(aarch64::Width::X64, Reg::X9, Reg::X20)
                    .expect("mov");
                self.emit_store_def_x9(*dst, 0);
                self.set_const(*dst, None);
            }
            LinearOp::RestoreCursor { src } => {
                // Use allocated register directly if available
                if let Some(alloc) = self.alloc_for_vreg(*src)
                    && let Some(reg) = alloc.as_reg()
                    && reg.class() == regalloc2::RegClass::Int
                {
                    self.ectx
                        .emit
                        .emit_mov_reg(
                            aarch64::Width::X64,
                            Reg::X19,
                            Reg::from_raw(reg.hw_enc() as u8),
                        )
                        .expect("mov");
                } else {
                    self.emit_load_use_x9(*src, 0);
                    self.ectx
                        .emit
                        .emit_mov_reg(aarch64::Width::X64, Reg::X19, Reg::X9)
                        .expect("mov");
                }
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
            LinearOp::StoreToAddr { addr, src, width } => {
                self.emit_store_to_addr(*addr, *src, *width);
            }
            LinearOp::LoadFromAddr { dst, addr, width } => {
                self.emit_load_from_addr(*dst, *addr, *width);
            }
            LinearOp::WriteToSlot { slot, src } => {
                self.emit_load_use_x9(*src, 0);
                let off = self.slot_off(*slot);
                self.emit_stack_store(aarch64::Width::X64, Reg::X9, off);
            }
            LinearOp::ReadFromSlot { dst, slot } => {
                let off = self.slot_off(*slot);
                self.emit_stack_load(aarch64::Width::X64, Reg::X9, off);
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
            | LinearOp::Branch { .. }
            | LinearOp::BranchIf { .. }
            | LinearOp::BranchIfZero { .. }
            | LinearOp::JumpTable { .. } => {
                panic!(
                    "structural op {:?} should not appear as a CFG-MIR instruction",
                    inst.op
                );
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
        self.current_inst_operands = Some(self.operands_for_terminator(term));
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
        self.current_inst_operands = None;
        self.current_op_id = None;
    }

    fn run(mut self, program: &cfg_mir::Program) -> LinearBackendResult {
        let (line_by_lambda_op, first_line_by_lambda) = build_debug_line_maps(program);
        for func in &program.funcs {
            let lambda_id = func.lambda_id.index() as u32;

            // Clear all vreg tracking including consts at function start
            self.flush_all_vregs_and_consts();

            // Pre-populate const_vregs with all Const values from this function.
            // This is needed because immediate-only consts may be used in blocks
            // that are emitted before the block containing their Const instruction.
            for inst in &func.insts {
                if let kajit_lir::LinearOp::Const { dst, value } = &inst.op {
                    self.const_vregs[dst.index()] = Some(*value);
                }
            }

            // Also track copies of consts (needed for immediate-only const copies)
            for inst in &func.insts {
                if let kajit_lir::LinearOp::Copy { dst, src } = &inst.op {
                    if let Some(value) = self.const_vregs[src.index()] {
                        self.const_vregs[dst.index()] = Some(value);
                    }
                }
            }

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
                    self.current_op_id = Some(op_id);
                    self.current_inst_allocs = self.allocs_for_inst(lambda_id, op_id);
                    self.current_inst_operands = Some(inst.operands.clone());
                    if self.no_edit_mode() {
                        self.current_inst_allocs =
                            Some(self.canonical_allocs_for_operands(&inst.operands));
                    }
                    let start_offset = self.ectx.emit.current_offset();
                    if self.apply_regalloc_edits {
                        self.apply_regalloc_edits(op_id, InstPosition::Before);
                    }
                    self.emit_inst(inst);
                    if self.apply_regalloc_edits {
                        self.apply_regalloc_edits(op_id, InstPosition::After);
                    }
                    let end_offset = self.ectx.emit.current_offset();
                    self.record_debug_op_range(lambda_id, op_id, line, start_offset, end_offset);
                    self.current_inst_allocs = None;
                    self.current_op_id = None;
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
        let (buf, asm_program) = self.ectx.finalize();
        let source_map = Some(buf.source_map.clone());
        LinearBackendResult {
            buf,
            entry,
            source_map,
            backend_debug_info: Some(self.backend_debug_info),
            asm_program,
            intrinsic_call_sites: Vec::new(),
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
                $(#[$meta:meta])*
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
                $(#[$meta])*
                fn $name() {
                    let mut builder = IrBuilder::new(stringify!($out_ty), 0);
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
        #[ignore] // regalloc conflict: v0 and v3 both colored to p0
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
        let mut builder = IrBuilder::new("u32", 0);
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

        let mut builder = IrBuilder::new("u32", 0);
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
        let mut builder = IrBuilder::new("u32", 0);
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
        let mut builder = IrBuilder::new("u32", 0);
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
        let mut builder = IrBuilder::new("u32", 0);
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

        let mut builder = IrBuilder::new("u64", 0);
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

        let mut builder = IrBuilder::new("u64", 0);
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
    #[ignore = "non-HIR path disabled"]
    fn linear_backend_call_lambda_with_data_args_and_results() {
        let mut builder = IrBuilder::new("u64", 0);
        let child = builder.create_lambda_with_data_args("u64", 0, 1);
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

        let ir = crate::compile_decoder(ScalarVec::SHAPE, crate::DecoderKind::Postcard);

        let ir_out = crate::deserialize::<ScalarVec>(&ir, &bytes).expect("ir decode");
        let serde_out = postcard::from_bytes::<ScalarVec>(&bytes).expect("serde decode");

        assert_eq!(ir_out, expected);
        assert_eq!(serde_out, expected);
    }
}
