#![allow(clippy::useless_conversion)]

use dynasmrt::{DynamicLabel, DynasmApi, DynasmLabelApi, dynasm};
use regalloc2::{Allocation, InstPosition, PReg, RegClass};
use std::collections::BTreeMap;

use crate::arch::{BASE_FRAME, EmitCtx};
use crate::ir::Width;
use crate::ir_backend::LinearBackendResult;
use crate::linearize::{BinOpKind, LinearOp, UnaryOpKind};
use crate::recipe::{Op, Recipe};
use crate::regalloc_engine::AllocatedProgram;
use crate::regalloc_mir::{BlockId, RaBlock, RaFunction, RaProgram, RaTerminator};

struct FunctionCtx {
    error_exit: DynamicLabel,
    data_results: Vec<crate::ir::VReg>,
    lambda_id: crate::ir::LambdaId,
}

struct EdgeTrampoline {
    label: DynamicLabel,
    target: DynamicLabel,
    moves: Vec<(Allocation, Allocation)>,
}

struct Lowerer<'a> {
    ectx: EmitCtx,
    /// DynamicLabel for each block, indexed by (lambda_index, block_id).
    block_labels: BTreeMap<(u32, u32), DynamicLabel>,
    lambda_labels: Vec<DynamicLabel>,
    slot_base: u32,
    spill_base: u32,
    entry: Option<dynasmrt::AssemblyOffset>,
    current_func: Option<FunctionCtx>,
    const_vregs: Vec<Option<u64>>,
    postreg_plan: crate::regalloc_engine::PostRegAllocPlan<'a>,
    /// Forward-branch blocks: blocks with no instructions whose terminator is a
    /// plain Branch. Maps (lambda_id, block_id) → target block_id.
    forward_branch_blocks: BTreeMap<(u32, u32), u32>,
    edge_trampoline_labels: BTreeMap<(u32, usize, usize), DynamicLabel>,
    edge_trampolines: Vec<EdgeTrampoline>,
    current_inst_allocs: Option<&'a [Allocation]>,
}

#[derive(Clone, Copy)]
enum IntrinsicArg {
    VReg { operand_index: usize },
    OutField(u32),
}

pub fn compile(program: &RaProgram, alloc: &AllocatedProgram) -> LinearBackendResult {
    let postreg_plan = crate::regalloc_engine::PostRegAllocPlan::build(alloc);
    let max_spillslots = postreg_plan.max_spillslots;
    Lowerer::new(program, max_spillslots, postreg_plan).run(program)
}

impl<'a> Lowerer<'a> {
    fn new(
        program: &RaProgram,
        max_spillslots: usize,
        postreg_plan: crate::regalloc_engine::PostRegAllocPlan<'a>,
    ) -> Self {
        let slot_base = BASE_FRAME;
        let slot_bytes = program.slot_count * 8;
        let spill_base = slot_base + slot_bytes;
        let spill_bytes = max_spillslots as u32 * 8;
        let extra_stack = slot_bytes + spill_bytes + 8;

        let mut ectx = EmitCtx::new(extra_stack);

        // Allocate a DynamicLabel for every block in every function.
        let mut block_labels = BTreeMap::new();
        let mut lambda_max = 0usize;
        for func in &program.funcs {
            lambda_max = lambda_max.max(func.lambda_id.index());
            // Scan instructions for CallLambda targets.
            for block in &func.blocks {
                for inst in &block.insts {
                    if let LinearOp::CallLambda { target, .. } = &inst.op {
                        lambda_max = lambda_max.max(target.index());
                    }
                }
            }
            for block in &func.blocks {
                let key = (func.lambda_id.index() as u32, block.id.0);
                block_labels.insert(key, ectx.new_label());
            }
        }
        let lambda_labels: Vec<DynamicLabel> = (0..=lambda_max).map(|_| ectx.new_label()).collect();

        // Pre-compute forward-branch blocks: blocks with no instructions
        // whose terminator is a plain Branch. These can be skipped and their
        // label resolved to the branch target (transitively).
        let mut forward_branch_blocks = BTreeMap::<(u32, u32), u32>::new();
        for func in &program.funcs {
            let lid = func.lambda_id.index() as u32;
            for block in &func.blocks {
                if block.insts.is_empty() {
                    if let RaTerminator::Branch { target } = &block.term {
                        forward_branch_blocks.insert((lid, block.id.0), target.0);
                    }
                }
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
            postreg_plan,
            forward_branch_blocks,
            edge_trampoline_labels: BTreeMap::new(),
            edge_trampolines: Vec::new(),
            current_inst_allocs: None,
        }
    }

    fn slot_off(&self, s: crate::ir::SlotId) -> u32 {
        self.slot_base + (s.index() as u32) * 8
    }

    fn block_label(&self, lambda_id: u32, block_id: BlockId) -> DynamicLabel {
        self.block_labels[&(lambda_id, block_id.0)]
    }

    fn spill_off(&self, slot: regalloc2::SpillSlot) -> u32 {
        self.spill_base + (slot.index() as u32) * 8
    }

    fn flush_all_vregs(&mut self) {
        self.const_vregs.fill(None);
    }

    fn emit_recipe_ops(&mut self, ops: Vec<Op>) {
        self.flush_all_vregs();
        self.ectx.emit_recipe(&Recipe {
            ops,
            label_count: 0,
        });
    }

    fn current_alloc(&self, operand_index: usize) -> Allocation {
        self.current_inst_allocs
            .and_then(|allocs| allocs.get(operand_index).copied())
            .unwrap_or_else(|| {
                panic!("missing regalloc allocation for operand index {operand_index}")
            })
    }

    fn const_of(&self, v: crate::ir::VReg) -> Option<u64> {
        self.const_vregs[v.index()]
    }

    fn set_const(&mut self, v: crate::ir::VReg, value: Option<u64>) {
        self.const_vregs[v.index()] = value;
    }

    // ── Allocation ↔ r10 helpers ──────────────────────────────────

    fn emit_load_r10_from_allocation(&mut self, alloc: Allocation) {
        if let Some(reg) = alloc.as_reg() {
            assert!(
                reg.class() == RegClass::Int,
                "unsupported register class {:?} for r10 load",
                reg.class()
            );
            let enc = reg.hw_enc() as u8;
            if enc != 10 {
                dynasm!(self.ectx.ops ; .arch x64 ; mov r10, Rq(enc));
            }
            return;
        }
        if let Some(slot) = alloc.as_stack() {
            let off = self.spill_off(slot) as i32;
            dynasm!(self.ectx.ops ; .arch x64 ; mov r10, [rsp + off]);
            return;
        }
        panic!("unexpected none allocation for r10 load");
    }

    fn emit_store_r10_to_allocation(&mut self, alloc: Allocation) {
        if let Some(reg) = alloc.as_reg() {
            assert!(
                reg.class() == RegClass::Int,
                "unsupported register class {:?} for r10 store",
                reg.class()
            );
            let enc = reg.hw_enc() as u8;
            if enc != 10 {
                dynasm!(self.ectx.ops ; .arch x64 ; mov Rq(enc), r10);
            }
            return;
        }
        if let Some(slot) = alloc.as_stack() {
            let off = self.spill_off(slot) as i32;
            dynasm!(self.ectx.ops ; .arch x64 ; mov [rsp + off], r10);
            return;
        }
        panic!("unexpected none allocation for r10 store");
    }

    fn emit_load_use_r10(&mut self, _v: crate::ir::VReg, operand_index: usize) {
        let alloc = self.current_alloc(operand_index);
        self.emit_load_r10_from_allocation(alloc);
    }

    fn emit_store_def_r10(&mut self, _v: crate::ir::VReg, operand_index: usize) {
        let alloc = self.current_alloc(operand_index);
        self.emit_store_r10_to_allocation(alloc);
    }

    // ── Allocation ↔ ABI register helpers ─────────────────────────

    fn emit_mov_preg_to_preg(&mut self, from: PReg, to: PReg) {
        if from == to {
            return;
        }
        assert!(
            from.class() == RegClass::Int && to.class() == RegClass::Int,
            "preg-to-preg move requires Int class"
        );
        let from_enc = from.hw_enc() as u8;
        let to_enc = to.hw_enc() as u8;
        dynasm!(self.ectx.ops ; .arch x64 ; mov Rq(to_enc), Rq(from_enc));
    }

    fn emit_capture_abi_ret_to_allocation(&mut self, abi_reg: PReg, operand_index: usize) {
        let alloc = self.current_alloc(operand_index);
        if let Some(reg) = alloc.as_reg() {
            self.emit_mov_preg_to_preg(abi_reg, reg);
            return;
        }
        if let Some(slot) = alloc.as_stack() {
            let off = self.spill_off(slot) as i32;
            let enc = abi_reg.hw_enc() as u8;
            dynasm!(self.ectx.ops ; .arch x64 ; mov [rsp + off], Rq(enc));
            return;
        }
        panic!("unexpected none allocation for ABI ret capture");
    }

    // ── Push/pop arg helpers (avoid ABI register clobbering) ─────

    fn push_allocation_operand(&mut self, operand_index: usize, push_count: usize) {
        let alloc = self.current_alloc(operand_index);
        let rsp_adjust = (push_count as i32) * 8;
        if let Some(reg) = alloc.as_reg() {
            let enc = reg.hw_enc() as u8;
            dynasm!(self.ectx.ops ; .arch x64 ; push Rq(enc));
        } else if let Some(slot) = alloc.as_stack() {
            let off = self.spill_off(slot) as i32 + rsp_adjust;
            dynasm!(self.ectx.ops ; .arch x64
                ; mov r10, [rsp + off]
                ; push r10
            );
        } else {
            panic!("unexpected none allocation for call arg");
        }
    }

    fn push_intrinsic_arg(&mut self, arg: IntrinsicArg, push_count: usize) {
        match arg {
            IntrinsicArg::VReg { operand_index } => {
                self.push_allocation_operand(operand_index, push_count);
            }
            IntrinsicArg::OutField(offset) => {
                let off = offset as i32;
                dynasm!(self.ectx.ops ; .arch x64
                    ; lea r10, [r14 + off]
                    ; push r10
                );
            }
        }
    }

    // ── Edit moves ────────────────────────────────────────────────

    fn emit_edit_move(&mut self, from: Allocation, to: Allocation) {
        if from == to || from.is_none() || to.is_none() {
            return;
        }
        match (from.as_reg(), from.as_stack(), to.as_reg(), to.as_stack()) {
            (Some(from_reg), None, Some(to_reg), None) => {
                self.emit_mov_preg_to_preg(from_reg, to_reg);
            }
            (Some(from_reg), None, None, Some(to_stack)) => {
                let off = self.spill_off(to_stack) as i32;
                let enc = from_reg.hw_enc() as u8;
                dynasm!(self.ectx.ops ; .arch x64 ; mov [rsp + off], Rq(enc));
            }
            (None, Some(from_stack), Some(to_reg), None) => {
                let off = self.spill_off(from_stack) as i32;
                let enc = to_reg.hw_enc() as u8;
                dynasm!(self.ectx.ops ; .arch x64 ; mov Rq(enc), [rsp + off]);
            }
            (None, Some(from_stack), None, Some(to_stack)) => {
                if from_stack == to_stack {
                    return;
                }
                let from_off = self.spill_off(from_stack) as i32;
                let to_off = self.spill_off(to_stack) as i32;
                dynasm!(self.ectx.ops
                    ; .arch x64
                    ; mov r10, [rsp + from_off]
                    ; mov [rsp + to_off], r10
                );
            }
            _ => {}
        }
    }

    // r[impl ir.regalloc.edits]
    fn apply_regalloc_edits(&mut self, linear_op_index: usize, pos: InstPosition) {
        let lambda_id = match self.current_func.as_ref() {
            Some(func) => func.lambda_id.index() as u32,
            None => return,
        };

        let edits = self
            .postreg_plan
            .lambda_for_id(lambda_id)
            .and_then(|by_lambda| match pos {
                InstPosition::Before => by_lambda.edits_before.get(&linear_op_index),
                InstPosition::After => by_lambda.edits_after.get(&linear_op_index),
            })
            .cloned()
            .unwrap_or_default();

        if edits.is_empty() {
            return;
        }

        self.flush_all_vregs();
        for (from, to) in edits {
            self.emit_edit_move(from, to);
        }
    }

    fn resolve_forwarded_block(&self, lambda_id: u32, block_id: BlockId) -> BlockId {
        let mut resolved = block_id;
        let mut hops = 0usize;
        while hops < 64 {
            let Some(&next_block) = self.forward_branch_blocks.get(&(lambda_id, resolved.0)) else {
                break;
            };
            let next_id = BlockId(next_block);
            // Check that the forwarding block itself has no edits that would
            // prevent skipping it.
            // For forward-branch blocks, the term_linear_op_index would need to
            // be checked, but since these blocks have no insts and we already
            // filtered to Branch-only blocks, we check edge edits.
            if next_id == resolved {
                break;
            }
            resolved = next_id;
            hops += 1;
        }
        resolved
    }

    fn edge_edit_moves(
        &self,
        linear_op_index: usize,
        succ_index: usize,
    ) -> Vec<(Allocation, Allocation)> {
        let Some(lambda_id) = self
            .current_func
            .as_ref()
            .map(|f| f.lambda_id.index() as u32)
        else {
            return Vec::new();
        };
        let Some(by_lambda) = self.postreg_plan.lambda_for_id(lambda_id) else {
            return Vec::new();
        };
        let key = (linear_op_index, succ_index);
        let mut moves = Vec::new();
        if let Some(before) = by_lambda.edge_edits_before.get(&key) {
            moves.extend(before.iter().copied());
        }
        if let Some(after) = by_lambda.edge_edits_after.get(&key) {
            moves.extend(after.iter().copied());
        }
        moves
    }

    fn apply_fallthrough_edge_edits(&mut self, linear_op_index: usize, succ_index: usize) {
        let moves = self.edge_edit_moves(linear_op_index, succ_index);
        if moves.is_empty() {
            return;
        }
        self.flush_all_vregs();
        for (from, to) in moves {
            self.emit_edit_move(from, to);
        }
    }

    fn has_edge_edits(&self, linear_op_index: usize, succ_index: usize) -> bool {
        !self.edge_edit_moves(linear_op_index, succ_index).is_empty()
    }

    fn edge_target_label(
        &mut self,
        linear_op_index: usize,
        succ_index: usize,
        actual_target: DynamicLabel,
    ) -> DynamicLabel {
        let Some(lambda_id) = self
            .current_func
            .as_ref()
            .map(|f| f.lambda_id.index() as u32)
        else {
            return actual_target;
        };
        let Some(by_lambda) = self.postreg_plan.lambda_for_id(lambda_id) else {
            return actual_target;
        };
        let key = (linear_op_index, succ_index);
        let has_edits = by_lambda.edge_edits_before.contains_key(&key)
            || by_lambda.edge_edits_after.contains_key(&key);
        if !has_edits {
            return actual_target;
        }

        let cache_key = (lambda_id, linear_op_index, succ_index);
        if let Some(label) = self.edge_trampoline_labels.get(&cache_key).copied() {
            return label;
        }

        let moves = self.edge_edit_moves(linear_op_index, succ_index);

        let label = self.ectx.new_label();
        self.edge_trampoline_labels.insert(cache_key, label);
        self.edge_trampolines.push(EdgeTrampoline {
            label,
            target: actual_target,
            moves,
        });
        label
    }

    fn emit_edge_trampolines(&mut self) {
        let trampolines = std::mem::take(&mut self.edge_trampolines);
        for trampoline in trampolines {
            self.ectx.bind_label(trampoline.label);
            if !trampoline.moves.is_empty() {
                self.flush_all_vregs();
                for (from, to) in trampoline.moves {
                    self.emit_edit_move(from, to);
                }
            }
            self.ectx.emit_branch(trampoline.target);
        }
    }

    // ── Per-op lowering helpers ───────────────────────────────────

    fn emit_read_from_field(&mut self, dst: crate::ir::VReg, offset: u32, width: Width) {
        let off = offset as i32;
        match width {
            Width::W1 => dynasm!(self.ectx.ops ; .arch x64 ; movzx r10d, BYTE [r14 + off]),
            Width::W2 => dynasm!(self.ectx.ops ; .arch x64 ; movzx r10d, WORD [r14 + off]),
            Width::W4 => dynasm!(self.ectx.ops ; .arch x64 ; mov r10d, DWORD [r14 + off]),
            Width::W8 => dynasm!(self.ectx.ops ; .arch x64 ; mov r10, QWORD [r14 + off]),
        }
        self.emit_store_def_r10(dst, 0);
        self.set_const(dst, None);
    }

    fn emit_write_to_field(&mut self, src: crate::ir::VReg, offset: u32, width: Width) {
        self.emit_load_use_r10(src, 0);
        let off = offset as i32;
        match width {
            Width::W1 => dynasm!(self.ectx.ops ; .arch x64 ; mov BYTE [r14 + off], r10b),
            Width::W2 => dynasm!(self.ectx.ops ; .arch x64 ; mov WORD [r14 + off], r10w),
            Width::W4 => dynasm!(self.ectx.ops ; .arch x64 ; mov DWORD [r14 + off], r10d),
            Width::W8 => dynasm!(self.ectx.ops ; .arch x64 ; mov QWORD [r14 + off], r10),
        }
    }

    fn emit_save_out_ptr(&mut self, dst: crate::ir::VReg) {
        dynasm!(self.ectx.ops ; .arch x64 ; mov r10, r14);
        self.emit_store_def_r10(dst, 0);
        self.set_const(dst, None);
    }

    fn emit_set_out_ptr(&mut self, src: crate::ir::VReg) {
        self.emit_load_use_r10(src, 0);
        dynasm!(self.ectx.ops ; .arch x64 ; mov r14, r10);
    }

    fn emit_slot_addr(&mut self, dst: crate::ir::VReg, slot: crate::ir::SlotId) {
        let slot_off = self.slot_off(slot) as i32;
        dynasm!(self.ectx.ops ; .arch x64 ; lea r10, [rsp + slot_off]);
        self.emit_store_def_r10(dst, 0);
        self.set_const(dst, None);
    }

    fn emit_read_bytes(&mut self, dst: crate::ir::VReg, count: u32) {
        self.emit_recipe_ops(vec![Op::BoundsCheck { count }]);
        match count {
            1 => dynasm!(self.ectx.ops ; .arch x64 ; movzx r10d, BYTE [r12]),
            2 => dynasm!(self.ectx.ops ; .arch x64 ; movzx r10d, WORD [r12]),
            4 => dynasm!(self.ectx.ops ; .arch x64 ; mov r10d, DWORD [r12]),
            8 => dynasm!(self.ectx.ops ; .arch x64 ; mov r10, QWORD [r12]),
            _ => panic!("unsupported ReadBytes count: {count}"),
        }
        self.emit_store_def_r10(dst, 0);
        self.set_const(dst, None);
        self.ectx.emit_advance_cursor_by(count);
    }

    fn emit_peek_byte(&mut self, dst: crate::ir::VReg) {
        self.emit_recipe_ops(vec![Op::BoundsCheck { count: 1 }]);
        dynasm!(self.ectx.ops ; .arch x64 ; movzx r10d, BYTE [r12]);
        self.emit_store_def_r10(dst, 0);
        self.set_const(dst, None);
    }

    fn emit_binop(
        &mut self,
        kind: BinOpKind,
        dst: crate::ir::VReg,
        lhs: crate::ir::VReg,
        _rhs: crate::ir::VReg,
    ) {
        self.emit_load_use_r10(lhs, 0);
        match kind {
            BinOpKind::Add => {
                let rhs_alloc = self.current_alloc(1);
                if let Some(reg) = rhs_alloc.as_reg() {
                    let enc = reg.hw_enc() as u8;
                    dynasm!(self.ectx.ops ; .arch x64 ; add r10, Rq(enc));
                } else if let Some(slot) = rhs_alloc.as_stack() {
                    let off = self.spill_off(slot) as i32;
                    dynasm!(self.ectx.ops ; .arch x64 ; add r10, [rsp + off]);
                }
            }
            BinOpKind::Sub => {
                let rhs_alloc = self.current_alloc(1);
                if let Some(reg) = rhs_alloc.as_reg() {
                    let enc = reg.hw_enc() as u8;
                    dynasm!(self.ectx.ops ; .arch x64 ; sub r10, Rq(enc));
                } else if let Some(slot) = rhs_alloc.as_stack() {
                    let off = self.spill_off(slot) as i32;
                    dynasm!(self.ectx.ops ; .arch x64 ; sub r10, [rsp + off]);
                }
            }
            BinOpKind::And => {
                let rhs_alloc = self.current_alloc(1);
                if let Some(reg) = rhs_alloc.as_reg() {
                    let enc = reg.hw_enc() as u8;
                    dynasm!(self.ectx.ops ; .arch x64 ; and r10, Rq(enc));
                } else if let Some(slot) = rhs_alloc.as_stack() {
                    let off = self.spill_off(slot) as i32;
                    dynasm!(self.ectx.ops ; .arch x64 ; and r10, [rsp + off]);
                }
            }
            BinOpKind::Or => {
                let rhs_alloc = self.current_alloc(1);
                if let Some(reg) = rhs_alloc.as_reg() {
                    let enc = reg.hw_enc() as u8;
                    dynasm!(self.ectx.ops ; .arch x64 ; or r10, Rq(enc));
                } else if let Some(slot) = rhs_alloc.as_stack() {
                    let off = self.spill_off(slot) as i32;
                    dynasm!(self.ectx.ops ; .arch x64 ; or r10, [rsp + off]);
                }
            }
            BinOpKind::Xor => {
                let rhs_alloc = self.current_alloc(1);
                if let Some(reg) = rhs_alloc.as_reg() {
                    let enc = reg.hw_enc() as u8;
                    dynasm!(self.ectx.ops ; .arch x64 ; xor r10, Rq(enc));
                } else if let Some(slot) = rhs_alloc.as_stack() {
                    let off = self.spill_off(slot) as i32;
                    dynasm!(self.ectx.ops ; .arch x64 ; xor r10, [rsp + off]);
                }
            }
            BinOpKind::CmpNe => {
                let rhs_alloc = self.current_alloc(1);
                if let Some(reg) = rhs_alloc.as_reg() {
                    let enc = reg.hw_enc() as u8;
                    dynasm!(self.ectx.ops ; .arch x64 ; cmp r10, Rq(enc));
                } else if let Some(slot) = rhs_alloc.as_stack() {
                    let off = self.spill_off(slot) as i32;
                    dynasm!(self.ectx.ops ; .arch x64 ; cmp r10, [rsp + off]);
                }
                dynasm!(self.ectx.ops
                    ; .arch x64
                    ; setne r10b
                    ; movzx r10, r10b
                );
            }
            BinOpKind::Shr => {
                dynasm!(self.ectx.ops ; .arch x64 ; shr r10, cl);
            }
            BinOpKind::Shl => {
                dynasm!(self.ectx.ops ; .arch x64 ; shl r10, cl);
            }
        }
        self.emit_store_def_r10(dst, 2);
        self.set_const(dst, None);
    }

    fn emit_unary(&mut self, kind: UnaryOpKind, dst: crate::ir::VReg, src: crate::ir::VReg) {
        self.emit_load_use_r10(src, 0);
        match kind {
            UnaryOpKind::ZigzagDecode { wide: true } => {
                dynasm!(self.ectx.ops
                    ; .arch x64
                    ; mov r11, r10
                    ; shr r11, 1
                    ; and r10, 1
                    ; neg r10
                    ; xor r10, r11
                );
            }
            UnaryOpKind::ZigzagDecode { wide: false } => {
                dynasm!(self.ectx.ops
                    ; .arch x64
                    ; mov r11d, r10d
                    ; shr r11d, 1
                    ; and r10d, 1
                    ; neg r10d
                    ; xor r10d, r11d
                );
            }
            UnaryOpKind::SignExtend { from_width } => match from_width {
                Width::W1 => dynasm!(self.ectx.ops ; .arch x64 ; movsx r10, r10b),
                Width::W2 => dynasm!(self.ectx.ops ; .arch x64 ; movsx r10, r10w),
                Width::W4 => dynasm!(self.ectx.ops ; .arch x64 ; movsxd r10, r10d),
                Width::W8 => {}
            },
        }
        self.emit_store_def_r10(dst, 1);
        self.set_const(dst, None);
    }

    fn emit_branch_if(&mut self, cond: crate::ir::VReg, target: DynamicLabel, invert: bool) {
        let _ = cond;
        let alloc = self.current_alloc(0);
        self.emit_branch_if_allocation(alloc, target, invert);
    }

    fn emit_branch_if_allocation(&mut self, alloc: Allocation, target: DynamicLabel, invert: bool) {
        if let Some(reg) = alloc.as_reg() {
            assert!(
                reg.class() == RegClass::Int,
                "unsupported register class {:?} for branch condition",
                reg.class()
            );
            let enc = reg.hw_enc() as u8;
            dynasm!(self.ectx.ops ; .arch x64 ; test Rq(enc), Rq(enc));
            if invert {
                dynasm!(self.ectx.ops ; .arch x64 ; jz =>target);
            } else {
                dynasm!(self.ectx.ops ; .arch x64 ; jnz =>target);
            }
            return;
        }
        if let Some(slot) = alloc.as_stack() {
            let off = self.spill_off(slot) as i32;
            dynasm!(self.ectx.ops ; .arch x64 ; mov r10, [rsp + off]);
            dynasm!(self.ectx.ops ; .arch x64 ; test r10, r10);
            if invert {
                dynasm!(self.ectx.ops ; .arch x64 ; jz =>target);
            } else {
                dynasm!(self.ectx.ops ; .arch x64 ; jnz =>target);
            }
            return;
        }
        panic!("unexpected none allocation for branch condition");
    }

    fn emit_jump_table(
        &mut self,
        lambda_id: u32,
        predicate: crate::ir::VReg,
        term: &RaTerminator,
        linear_op_index: usize,
    ) {
        let RaTerminator::JumpTable {
            targets, default, ..
        } = term
        else {
            unreachable!();
        };
        let _ = predicate;
        let alloc = self.current_alloc(0);
        if let Some(reg) = alloc.as_reg() {
            let enc = reg.hw_enc() as u8;
            dynasm!(self.ectx.ops ; .arch x64 ; mov r10, Rq(enc));
        } else if let Some(slot) = alloc.as_stack() {
            let off = self.spill_off(slot) as i32;
            dynasm!(self.ectx.ops ; .arch x64 ; mov r10, [rsp + off]);
        } else {
            panic!("unexpected none allocation for jumptable predicate");
        }
        for (index, target_block) in targets.iter().enumerate() {
            let resolved = self.resolve_forwarded_block(lambda_id, *target_block);
            let target_label = self.edge_target_label(
                linear_op_index,
                index,
                self.block_label(lambda_id, resolved),
            );
            dynasm!(self.ectx.ops
                ; .arch x64
                ; cmp r10d, index as i32
                ; je =>target_label
            );
        }
        let default_succ_index = targets.len();
        let resolved_default = self.resolve_forwarded_block(lambda_id, *default);
        let default_target = self.edge_target_label(
            linear_op_index,
            default_succ_index,
            self.block_label(lambda_id, resolved_default),
        );
        self.ectx.emit_branch(default_target);
    }

    // ── Intrinsic / call helpers ──────────────────────────────────

    fn emit_call_intrinsic_with_args(&mut self, fn_ptr: *const u8, args: &[IntrinsicArg]) {
        use crate::context::{CTX_ERROR_CODE, CTX_INPUT_PTR};

        let error_exit = self
            .current_func
            .as_ref()
            .expect("CallIntrinsic outside function")
            .error_exit;

        self.flush_all_vregs();

        dynasm!(self.ectx.ops
            ; .arch x64
            ; mov [r15 + CTX_INPUT_PTR as i32], r12
        );

        #[cfg(not(windows))]
        {
            if args.len() > 5 {
                panic!(
                    "unsupported CallIntrinsic arity in linear backend adapter: {} args (+ctx)",
                    args.len()
                );
            }
            let abi_regs: &[PReg] = &[
                PReg::new(6, RegClass::Int), // rsi
                PReg::new(2, RegClass::Int), // rdx
                PReg::new(1, RegClass::Int), // rcx
                PReg::new(8, RegClass::Int), // r8
                PReg::new(9, RegClass::Int), // r9
            ];
            for (i, arg) in args.iter().copied().enumerate() {
                self.push_intrinsic_arg(arg, i);
            }
            dynasm!(self.ectx.ops ; .arch x64 ; mov rdi, r15);
            for i in (0..args.len()).rev() {
                let enc = abi_regs[i].hw_enc() as u8;
                dynasm!(self.ectx.ops ; .arch x64 ; pop Rq(enc));
            }
        }

        #[cfg(windows)]
        {
            if args.len() > 3 {
                panic!(
                    "unsupported CallIntrinsic arity in linear backend adapter: {} args (+ctx)",
                    args.len()
                );
            }
            let abi_regs: &[PReg] = &[
                PReg::new(2, RegClass::Int), // rdx
                PReg::new(8, RegClass::Int), // r8
                PReg::new(9, RegClass::Int), // r9
            ];
            for (i, arg) in args.iter().copied().enumerate() {
                self.push_intrinsic_arg(arg, i);
            }
            dynasm!(self.ectx.ops ; .arch x64 ; mov rcx, r15);
            for i in (0..args.len()).rev() {
                let enc = abi_regs[i].hw_enc() as u8;
                dynasm!(self.ectx.ops ; .arch x64 ; pop Rq(enc));
            }
        }

        let ptr_val = fn_ptr as i64;
        dynasm!(self.ectx.ops
            ; .arch x64
            ; mov rax, QWORD ptr_val
            ; call rax
        );

        dynasm!(self.ectx.ops
            ; .arch x64
            ; mov r12, [r15 + CTX_INPUT_PTR as i32]
            ; mov r10d, [r15 + CTX_ERROR_CODE as i32]
            ; test r10d, r10d
            ; jnz =>error_exit
        );
    }

    // r[impl ir.intrinsics]
    fn emit_call_intrinsic(
        &mut self,
        func: crate::ir::IntrinsicFn,
        args: &[crate::ir::VReg],
        dst: Option<crate::ir::VReg>,
        field_offset: u32,
    ) {
        let fn_ptr = func.0 as *const u8;
        match dst {
            Some(dst) => {
                let dst_operand_index = args.len();
                let call_args: Vec<IntrinsicArg> = args
                    .iter()
                    .copied()
                    .enumerate()
                    .map(|(i, _vreg)| IntrinsicArg::VReg { operand_index: i })
                    .collect();
                self.emit_call_intrinsic_with_args(fn_ptr, &call_args);
                let rax = PReg::new(0, RegClass::Int);
                self.emit_capture_abi_ret_to_allocation(rax, dst_operand_index);
                let _ = dst;
            }
            None => {
                let mut call_args: Vec<IntrinsicArg> = args
                    .iter()
                    .copied()
                    .enumerate()
                    .map(|(i, _vreg)| IntrinsicArg::VReg { operand_index: i })
                    .collect();
                call_args.push(IntrinsicArg::OutField(field_offset));
                self.emit_call_intrinsic_with_args(fn_ptr, &call_args);
            }
        }
    }

    fn emit_call_pure_with_args(&mut self, fn_ptr: *const u8, args: &[IntrinsicArg]) {
        self.flush_all_vregs();

        #[cfg(not(windows))]
        {
            if args.len() > 6 {
                panic!(
                    "unsupported CallPure arity in linear backend adapter: {} args",
                    args.len()
                );
            }
            let abi_regs: &[PReg] = &[
                PReg::new(7, RegClass::Int), // rdi
                PReg::new(6, RegClass::Int), // rsi
                PReg::new(2, RegClass::Int), // rdx
                PReg::new(1, RegClass::Int), // rcx
                PReg::new(8, RegClass::Int), // r8
                PReg::new(9, RegClass::Int), // r9
            ];
            for (i, arg) in args.iter().copied().enumerate() {
                self.push_intrinsic_arg(arg, i);
            }
            for i in (0..args.len()).rev() {
                let enc = abi_regs[i].hw_enc() as u8;
                dynasm!(self.ectx.ops ; .arch x64 ; pop Rq(enc));
            }
        }

        #[cfg(windows)]
        {
            if args.len() > 4 {
                panic!(
                    "unsupported CallPure arity in linear backend adapter: {} args",
                    args.len()
                );
            }
            let abi_regs: &[PReg] = &[
                PReg::new(1, RegClass::Int), // rcx
                PReg::new(2, RegClass::Int), // rdx
                PReg::new(8, RegClass::Int), // r8
                PReg::new(9, RegClass::Int), // r9
            ];
            for (i, arg) in args.iter().copied().enumerate() {
                self.push_intrinsic_arg(arg, i);
            }
            for i in (0..args.len()).rev() {
                let enc = abi_regs[i].hw_enc() as u8;
                dynasm!(self.ectx.ops ; .arch x64 ; pop Rq(enc));
            }
        }

        let ptr_val = fn_ptr as i64;
        dynasm!(self.ectx.ops
            ; .arch x64
            ; mov rax, QWORD ptr_val
            ; call rax
        );
    }

    fn emit_call_pure(
        &mut self,
        func: crate::ir::IntrinsicFn,
        args: &[crate::ir::VReg],
        dst: crate::ir::VReg,
    ) {
        let fn_ptr = func.0 as *const u8;
        let dst_operand_index = args.len();
        let call_args: Vec<IntrinsicArg> = args
            .iter()
            .copied()
            .enumerate()
            .map(|(i, _)| IntrinsicArg::VReg { operand_index: i })
            .collect();
        self.emit_call_pure_with_args(fn_ptr, &call_args);
        let rax = PReg::new(0, RegClass::Int);
        self.emit_capture_abi_ret_to_allocation(rax, dst_operand_index);
        let _ = dst;
    }

    fn emit_store_incoming_lambda_args(&mut self, data_args: &[crate::ir::VReg]) {
        #[cfg(not(windows))]
        const MAX_LAMBDA_DATA_ARGS: usize = 4;
        #[cfg(windows)]
        const MAX_LAMBDA_DATA_ARGS: usize = 2;

        if data_args.len() > MAX_LAMBDA_DATA_ARGS {
            panic!(
                "x64 CallLambda supports at most {MAX_LAMBDA_DATA_ARGS} data args, got {}",
                data_args.len()
            );
        }
    }

    fn emit_load_lambda_results_to_ret_regs(
        &mut self,
        lambda_id: crate::ir::LambdaId,
        data_results: &[crate::ir::VReg],
    ) {
        if data_results.len() > 2 {
            panic!(
                "x64 CallLambda supports at most 2 data results, got {}",
                data_results.len()
            );
        }

        let result_allocs = self
            .postreg_plan
            .lambda_for(lambda_id)
            .map(|lambda| lambda.return_result_allocs)
            .unwrap_or(&[]);
        assert!(
            result_allocs.len() >= data_results.len(),
            "missing return allocation mapping for lambda {:?}: need {}, got {}",
            lambda_id,
            data_results.len(),
            result_allocs.len()
        );

        let rax = PReg::new(0, RegClass::Int);
        let rdx = PReg::new(2, RegClass::Int);
        if let Some(&alloc) = result_allocs.first() {
            self.emit_load_r10_from_allocation(alloc);
            self.emit_mov_preg_to_preg(PReg::new(10, RegClass::Int), rax);
        }
        if let Some(&alloc) = result_allocs.get(1) {
            self.emit_load_r10_from_allocation(alloc);
            self.emit_mov_preg_to_preg(PReg::new(10, RegClass::Int), rdx);
        }
    }

    fn emit_call_lambda(
        &mut self,
        label: DynamicLabel,
        args: &[crate::ir::VReg],
        results: &[crate::ir::VReg],
    ) {
        use crate::context::{CTX_ERROR_CODE, CTX_INPUT_PTR};

        #[cfg(not(windows))]
        const MAX_LAMBDA_DATA_ARGS: usize = 4;
        #[cfg(windows)]
        const MAX_LAMBDA_DATA_ARGS: usize = 2;
        if args.len() > MAX_LAMBDA_DATA_ARGS {
            panic!(
                "x64 CallLambda supports at most {MAX_LAMBDA_DATA_ARGS} data args, got {}",
                args.len()
            );
        }
        if results.len() > 2 {
            panic!(
                "x64 CallLambda supports at most 2 data results, got {}",
                results.len()
            );
        }

        let error_exit = self
            .current_func
            .as_ref()
            .expect("CallLambda outside function")
            .error_exit;
        self.flush_all_vregs();
        dynasm!(self.ectx.ops
            ; .arch x64
            ; mov [r15 + CTX_INPUT_PTR as i32], r12
        );

        #[cfg(not(windows))]
        {
            let abi_data_regs: &[PReg] = &[
                PReg::new(2, RegClass::Int), // rdx
                PReg::new(1, RegClass::Int), // rcx
                PReg::new(8, RegClass::Int), // r8
                PReg::new(9, RegClass::Int), // r9
            ];
            for (i, &_arg) in args.iter().enumerate() {
                self.push_allocation_operand(i, i);
            }
            dynasm!(self.ectx.ops ; .arch x64 ; lea rdi, [r14] ; mov rsi, r15);
            for i in (0..args.len()).rev() {
                let enc = abi_data_regs[i].hw_enc() as u8;
                dynasm!(self.ectx.ops ; .arch x64 ; pop Rq(enc));
            }
        }
        #[cfg(windows)]
        {
            let abi_data_regs: &[PReg] = &[
                PReg::new(8, RegClass::Int), // r8
                PReg::new(9, RegClass::Int), // r9
            ];
            for (i, &_arg) in args.iter().enumerate() {
                self.push_allocation_operand(i, i);
            }
            dynasm!(self.ectx.ops ; .arch x64 ; lea rcx, [r14] ; mov rdx, r15);
            for i in (0..args.len()).rev() {
                let enc = abi_data_regs[i].hw_enc() as u8;
                dynasm!(self.ectx.ops ; .arch x64 ; pop Rq(enc));
            }
        }

        dynasm!(self.ectx.ops ; .arch x64 ; call =>label);
        dynasm!(self.ectx.ops
            ; .arch x64
            ; mov r12, [r15 + CTX_INPUT_PTR as i32]
            ; mov r10d, [r15 + CTX_ERROR_CODE as i32]
            ; test r10d, r10d
            ; jnz =>error_exit
        );

        let rax = PReg::new(0, RegClass::Int);
        let rdx = PReg::new(2, RegClass::Int);
        if let Some(&_dst) = results.first() {
            self.emit_capture_abi_ret_to_allocation(rax, args.len());
        }
        if let Some(&_dst) = results.get(1) {
            self.emit_capture_abi_ret_to_allocation(rdx, args.len() + 1);
        }
    }

    // ── Instruction emission ──────────────────────────────────────

    fn emit_inst(&mut self, op: &LinearOp) {
        match op {
            LinearOp::Const { dst, value } => {
                dynasm!(self.ectx.ops
                    ; .arch x64
                    ; mov r10, QWORD *value as i64
                );
                self.emit_store_def_r10(*dst, 0);
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
            LinearOp::AdvanceCursor { count } => {
                self.ectx.emit_advance_cursor_by(*count);
            }
            LinearOp::AdvanceCursorBy { src } => {
                self.emit_load_use_r10(*src, 0);
                dynasm!(self.ectx.ops ; .arch x64 ; add r12, r10);
            }
            LinearOp::SaveCursor { dst } => {
                dynasm!(self.ectx.ops ; .arch x64 ; mov r10, r12);
                self.emit_store_def_r10(*dst, 0);
                self.set_const(*dst, None);
            }
            LinearOp::RestoreCursor { src } => {
                self.emit_load_use_r10(*src, 0);
                dynasm!(self.ectx.ops ; .arch x64 ; mov r12, r10);
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
            LinearOp::WriteToSlot { slot, src } => {
                self.emit_load_use_r10(*src, 0);
                let off = self.slot_off(*slot) as i32;
                dynasm!(self.ectx.ops ; .arch x64 ; mov [rsp + off], r10);
            }
            LinearOp::ReadFromSlot { dst, slot } => {
                let off = self.slot_off(*slot) as i32;
                dynasm!(self.ectx.ops ; .arch x64 ; mov r10, [rsp + off]);
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
            LinearOp::CallPure { func, args, dst } => {
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

            // These are structural ops handled at function/block level, not as instructions.
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

    // ── Terminator emission ───────────────────────────────────────

    fn emit_terminator(
        &mut self,
        func: &RaFunction,
        block: &RaBlock,
        next_block: Option<&RaBlock>,
    ) {
        let lambda_id = func.lambda_id.index() as u32;
        let linear_op_index = block.term_linear_op_index;

        // Load allocations for the terminator if it has a linear_op_index.
        if let Some(lin_idx) = linear_op_index {
            self.current_inst_allocs = self
                .postreg_plan
                .lambda_for_id(lambda_id)
                .and_then(|by_lambda| by_lambda.allocs_by_linear_op.get(&lin_idx).copied());
            self.apply_regalloc_edits(lin_idx, InstPosition::Before);
        }

        match &block.term {
            RaTerminator::Return => {
                // Return is handled by end_func in the function epilogue.
            }
            RaTerminator::ErrorExit { code } => {
                self.flush_all_vregs();
                self.ectx.emit_error(*code);
            }
            RaTerminator::Branch { target } => {
                let lin_idx = linear_op_index;
                let resolved = self.resolve_forwarded_block(lambda_id, *target);
                let target_label = if let Some(lin_idx) = lin_idx {
                    self.edge_target_label(lin_idx, 0, self.block_label(lambda_id, resolved))
                } else {
                    self.block_label(lambda_id, resolved)
                };
                // Check if this is a redundant fallthrough.
                let is_redundant_fallthrough = if let Some(lin_idx) = lin_idx {
                    if self.has_edge_edits(lin_idx, 0) {
                        false
                    } else if let Some(next) = next_block {
                        let resolved_next = self.resolve_forwarded_block(lambda_id, next.id);
                        resolved == resolved_next
                    } else {
                        false
                    }
                } else {
                    false
                };
                if !is_redundant_fallthrough {
                    self.ectx.emit_branch(target_label);
                }
            }
            RaTerminator::BranchIf {
                cond,
                target,
                fallthrough: _,
            } => {
                let lin_idx = linear_op_index.expect("BranchIf should have linear op index");
                let resolved = self.resolve_forwarded_block(lambda_id, *target);
                let taken_target =
                    self.edge_target_label(lin_idx, 0, self.block_label(lambda_id, resolved));
                if let Some(cond_const) = self.const_of(*cond) {
                    if cond_const != 0 {
                        self.ectx.emit_branch(taken_target);
                    } else {
                        self.apply_fallthrough_edge_edits(lin_idx, 1);
                    }
                } else {
                    self.emit_branch_if(*cond, taken_target, false);
                    self.apply_fallthrough_edge_edits(lin_idx, 1);
                }
            }
            RaTerminator::BranchIfZero {
                cond,
                target,
                fallthrough: _,
            } => {
                let lin_idx = linear_op_index.expect("BranchIfZero should have linear op index");
                let resolved = self.resolve_forwarded_block(lambda_id, *target);
                let taken_target =
                    self.edge_target_label(lin_idx, 0, self.block_label(lambda_id, resolved));
                if let Some(cond_const) = self.const_of(*cond) {
                    if cond_const == 0 {
                        self.ectx.emit_branch(taken_target);
                    } else {
                        self.apply_fallthrough_edge_edits(lin_idx, 1);
                    }
                } else {
                    self.emit_branch_if(*cond, taken_target, true);
                    self.apply_fallthrough_edge_edits(lin_idx, 1);
                }
            }
            RaTerminator::JumpTable { predicate, .. } => {
                let lin_idx = linear_op_index.expect("JumpTable should have linear op index");
                self.emit_jump_table(lambda_id, *predicate, &block.term, lin_idx);
            }
        }

        if let Some(lin_idx) = linear_op_index {
            self.apply_regalloc_edits(lin_idx, InstPosition::After);
        }
        self.current_inst_allocs = None;
    }

    // ── Main emission loop ────────────────────────────────────────

    // r[impl ir.backends.post-regalloc.branch-test]
    // r[impl ir.backends.post-regalloc.shuffle]
    fn run(mut self, program: &RaProgram) -> LinearBackendResult {
        for func in &program.funcs {
            let lambda_id = func.lambda_id.index() as u32;

            // Begin function (equivalent to FuncStart).
            self.flush_all_vregs();
            let label = self.lambda_labels[func.lambda_id.index()];
            self.ectx.bind_label(label);
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

            // Emit blocks.
            for (block_index, block) in func.blocks.iter().enumerate() {
                self.flush_all_vregs();
                let block_label = self.block_label(lambda_id, block.id);
                self.ectx.bind_label(block_label);

                // Emit instructions.
                for inst in &block.insts {
                    let lin_idx = inst.linear_op_index;
                    self.current_inst_allocs = self
                        .postreg_plan
                        .lambda_for_id(lambda_id)
                        .and_then(|by_lambda| by_lambda.allocs_by_linear_op.get(&lin_idx).copied());
                    self.apply_regalloc_edits(lin_idx, InstPosition::Before);
                    self.emit_inst(&inst.op);
                    self.apply_regalloc_edits(lin_idx, InstPosition::After);
                    self.current_inst_allocs = None;
                }

                // Emit terminator.
                let next_block = func.blocks.get(block_index + 1);
                self.emit_terminator(func, block, next_block);
            }

            // End function (equivalent to FuncEnd).
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
        LinearBackendResult { buf, entry }
    }
}
