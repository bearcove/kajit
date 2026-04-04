//! aarch64 backend for regalloc3 (native types, no regalloc2 conversion).

use kajit_emit::aarch64::{Condition, LabelId, Reg, Width};
use kajit_mir::cfg_mir::{self, Function, Inst, Terminator};
use kajit_mir::regalloc3::machine_inst::PReg;
use kajit_mir::regalloc3_result::{AllocatedCfgFunctionRa3, AllocatedCfgProgramRa3};

use crate::arch::EmitCtx;
use crate::context::{CTX_INPUT_END, CTX_INPUT_PTR};
use crate::ir_backend::LinearBackendResult;
use kajit_lir::{BinOpKind, LinearOp, UnaryOpKind};
use std::collections::HashMap;

/// Recorded intrinsic call site for harness relocation.
#[derive(Debug, Clone)]
pub struct IntrinsicCallSiteInfo {
    /// Offset in the code buffer of the first `movz` instruction.
    pub code_offset: usize,
    /// The intrinsic function pointer (for looking up the symbol name).
    pub func: kajit_ir::IntrinsicFn,
}

/// Recorded data blob address site for relocation.
#[derive(Debug, Clone)]
pub struct DataRelocInfo {
    /// Offset in the code buffer of the first `movz` instruction (4-instruction fixed sequence).
    pub code_offset: usize,
    /// Index into data_blobs.
    pub blob_id: u32,
}

/// Context for emitting a single function.
struct EmitContext<'a> {
    ectx: &'a mut EmitCtx,
    func: &'a Function,
    alloc_func: &'a AllocatedCfgFunctionRa3,
    block_labels: HashMap<cfg_mir::BlockId, LabelId>,
    success_exit: LabelId,
    /// Slot offset base: base_frame + spill_slots * 8 gives the start of user slots.
    slot_base: u32,
    /// Scratch stack area used to snapshot edge arguments before delivering them.
    edge_tmp_base: u32,
    /// VReg → constant value (for immediate folding in BinOps)
    const_values: HashMap<kajit_ir::VReg, u64>,
    /// OpId → DWARF line number (for source-level debugging)
    line_map: HashMap<cfg_mir::OpId, u32>,
    /// Recorded intrinsic call sites for harness relocation.
    intrinsic_call_sites: Vec<IntrinsicCallSiteInfo>,
    /// Recorded data blob address sites for relocation.
    data_relocs: Vec<DataRelocInfo>,
    /// VRegs whose CmpXx can be fused with the terminator branch (skip cset, emit b.cc).
    fused_cmps: HashMap<kajit_ir::VReg, Condition>,
    /// Or vregs that should be emitted as bfi. Maps Or's dst → (byte_src, accum, lsb, width).
    fused_bfi: HashMap<kajit_ir::VReg, BfiInfo>,
    /// Output pointer register (x0 for leaf, x21 for non-leaf).
    output_reg: Reg,
    /// Context pointer register (x1 for leaf, x22 for non-leaf).
    ctx_reg: Reg,
    /// Whether intrinsic/lambda calls should sync the fixed cursor register through ctx.input_ptr.
    sync_ctx_cursor_around_calls: bool,
    /// Intermediate vregs (And/Shl results) whose instructions should be skipped.
    fused_skip: std::collections::HashSet<kajit_ir::VReg>,
    /// Fused base+offset info for LoadFromAddr/RestoreCursor.
    /// Maps addr vreg → (base_vreg, offset). When a LoadFromAddr/RestoreCursor
    /// consumes an addr that was defined by `Add(base, const_offset)`, the Add
    /// is skipped and the load/restore uses `[base_reg, #offset]` directly.
    fused_addr_offsets: HashMap<kajit_ir::VReg, (kajit_ir::VReg, u64)>,
    /// Register used by RestoreCursor and the epilogue for cursor writeback.
    /// x19 for non-leaf, a caller-saved register for leaf.
    cursor_writeback_reg: Reg,
    /// Set to true when emitting the last block before the success epilogue.
    /// Allows Return terminator to fall through instead of branching.
    is_last_emitted_block: bool,
    /// Per-edge trampoline labels for edges that need value delivery before control transfer.
    edge_trampoline_labels: HashMap<cfg_mir::EdgeId, LabelId>,
}

struct BfiInfo {
    byte_src: kajit_ir::VReg,
    accum: kajit_ir::VReg,
    lsb: u8,
    width: u8,
}

fn emit_parallel_reg_moves(ectx: &mut EmitCtx, moves: &[(Reg, Reg)], temp: Reg) {
    // Build dependency map: dst -> src.
    let mut deps: HashMap<Reg, Reg> = HashMap::new();
    for &(dst, src) in moves {
        if dst != src {
            deps.insert(dst, src);
        }
    }

    while !deps.is_empty() {
        let ready = deps
            .iter()
            .find(|(dst, _)| !deps.values().any(|src| src == *dst))
            .map(|(&dst, &src)| (dst, src));

        if let Some((dst, src)) = ready {
            ectx.emit.emit_mov_reg(Width::X64, dst, src).expect("mov");
            deps.remove(&dst);
            continue;
        }

        let (&cycle_dst, &cycle_src) = deps.iter().next().unwrap();
        ectx.emit
            .emit_mov_reg(Width::X64, temp, cycle_dst)
            .expect("mov to temp");
        deps.remove(&cycle_dst);
        for (_, src) in deps.iter_mut() {
            if *src == cycle_dst {
                *src = temp;
            }
        }
        ectx.emit
            .emit_mov_reg(Width::X64, cycle_dst, cycle_src)
            .expect("mov cycle edge");
    }
}

impl<'a> EmitContext<'a> {
    /// Get physical register for a vreg, or None if spilled/dead.
    fn preg_for_vreg(&self, vreg: kajit_ir::VReg) -> Option<PReg> {
        self.alloc_func.preg_for_vreg(vreg)
    }

    /// Convert regalloc3 PReg to kajit_emit Reg.
    fn preg_to_reg(&self, preg: PReg) -> Reg {
        Reg::from_raw(preg.0)
    }

    /// Get hardware register for a vreg, or use a temp register and load from spill slot.
    /// For spilled constants, rematerializes with movz instead of loading from stack.
    fn reg_for_vreg_with_temp(&mut self, vreg: kajit_ir::VReg, temp: Reg) -> Reg {
        if let Some(preg) = self.preg_for_vreg(vreg) {
            return self.preg_to_reg(preg);
        }

        // Rematerializable constant - emit movz instead of stack load
        if let Some(&value) = self.alloc_func.rematerializable.get(&vreg) {
            self.emit_load_u64(temp, value);
            return temp;
        }

        // Spilled - load from spill slot
        if let Some(slot) = self.alloc_func.spill_slot_for_vreg(vreg) {
            let offset = self.ectx.base_frame + (slot.0 * 8);
            self.ectx
                .emit
                .emit_ldr_imm(Width::X64, temp, Reg::SP, offset)
                .expect("ldr spill");
            return temp;
        }

        // Dead vreg - use temp with dummy value
        self.ectx
            .emit
            .emit_movz_imm(Width::X64, temp, 0, 0)
            .expect("movz dead");
        temp
    }

    /// Store a value from a register to a vreg (handling spills).
    fn store_to_vreg(&mut self, vreg: kajit_ir::VReg, from_reg: Reg) {
        if let Some(preg) = self.preg_for_vreg(vreg) {
            let dst_reg = self.preg_to_reg(preg);
            if dst_reg != from_reg {
                self.ectx
                    .emit
                    .emit_mov_reg(Width::X64, dst_reg, from_reg)
                    .expect("mov");
            }
        } else if let Some(slot) = self.alloc_func.spill_slot_for_vreg(vreg) {
            let offset = self.ectx.base_frame + (slot.0 * 8);
            self.ectx
                .emit
                .emit_str_imm(Width::X64, from_reg, Reg::SP, offset)
                .expect("str spill");
        }
        // If dead, do nothing
    }

    /// Get the destination register for a vreg def.
    /// Returns the allocated register if available, or the fallback temp.
    /// After emitting, call `store_to_vreg` only if this returned the fallback.
    fn dst_reg_or_temp(&self, vreg: kajit_ir::VReg, fallback: Reg) -> Reg {
        if let Some(preg) = self.preg_for_vreg(vreg) {
            self.preg_to_reg(preg)
        } else {
            fallback
        }
    }

    /// Emit a set of parallel register moves using Briggs-style resolution.
    /// `moves` is a list of (dst, src) pairs. `temp` is a scratch register
    /// used to break cycles.
    fn emit_parallel_moves(&mut self, moves: &[(Reg, Reg)], temp: Reg) {
        emit_parallel_reg_moves(self.ectx, moves, temp);
    }

    /// Load a 64-bit constant into a register.
    fn emit_load_u64(&mut self, rd: Reg, value: u64) {
        let p0 = (value & 0xFFFF) as u16;
        let p1 = ((value >> 16) & 0xFFFF) as u16;
        let p2 = ((value >> 32) & 0xFFFF) as u16;
        let p3 = ((value >> 48) & 0xFFFF) as u16;
        self.ectx
            .emit
            .emit_movz_imm(Width::X64, rd, p0, 0)
            .expect("movz");
        if p1 != 0 {
            self.ectx
                .emit
                .emit_movk_imm(Width::X64, rd, p1, 16)
                .expect("movk");
        }
        if p2 != 0 {
            self.ectx
                .emit
                .emit_movk_imm(Width::X64, rd, p2, 32)
                .expect("movk");
        }
        if p3 != 0 {
            self.ectx
                .emit
                .emit_movk_imm(Width::X64, rd, p3, 48)
                .expect("movk");
        }
    }

    /// Load a 64-bit value into a register using exactly 4 instructions
    /// (movz + 3 movk). The fixed 16-byte size makes the sequence relocatable.
    fn emit_load_u64_fixed(&mut self, rd: Reg, value: u64) {
        let p0 = (value & 0xFFFF) as u16;
        let p1 = ((value >> 16) & 0xFFFF) as u16;
        let p2 = ((value >> 32) & 0xFFFF) as u16;
        let p3 = ((value >> 48) & 0xFFFF) as u16;
        self.ectx
            .emit
            .emit_movz_imm(Width::X64, rd, p0, 0)
            .expect("movz");
        self.ectx
            .emit
            .emit_movk_imm(Width::X64, rd, p1, 16)
            .expect("movk");
        self.ectx
            .emit
            .emit_movk_imm(Width::X64, rd, p2, 32)
            .expect("movk");
        self.ectx
            .emit
            .emit_movk_imm(Width::X64, rd, p3, 48)
            .expect("movk");
    }

    /// If vreg is a known constant that fits in a 12-bit immediate, return its value.
    fn small_const(&self, vreg: kajit_ir::VReg) -> Option<u16> {
        let value = self.const_values.get(&vreg)?;
        if *value <= 0xFFF {
            Some(*value as u16)
        } else {
            None
        }
    }

    /// Offset of a user slot on the stack.
    fn slot_off(&self, slot: u32) -> u32 {
        self.slot_base + slot * 8
    }

    fn edge_tmp_off(&self, index: usize) -> u32 {
        self.edge_tmp_base + (index as u32) * 8
    }

    fn edge_has_moves(&self, edge_id: cfg_mir::EdgeId) -> bool {
        let edge = &self.func.edges[edge_id.index()];
        edge.args.iter().any(|arg| arg.source != arg.target)
    }

    fn edge_target_label(&mut self, edge_id: cfg_mir::EdgeId, target_label: LabelId) -> LabelId {
        if !self.edge_has_moves(edge_id) {
            return target_label;
        }
        *self
            .edge_trampoline_labels
            .entry(edge_id)
            .or_insert_with(|| self.ectx.new_label())
    }

    fn emit_edge_moves(&mut self, edge_id: cfg_mir::EdgeId) {
        let edge = &self.func.edges[edge_id.index()];
        if edge.args.is_empty() {
            return;
        }

        for (index, arg) in edge.args.iter().enumerate() {
            let src_reg = self.reg_for_vreg_with_temp(arg.source, Reg::X9);
            let off = self.edge_tmp_off(index);
            self.ectx
                .emit
                .emit_str_imm(Width::X64, src_reg, Reg::SP, off)
                .expect("str edge tmp");
        }

        for (index, arg) in edge.args.iter().enumerate() {
            let off = self.edge_tmp_off(index);
            self.ectx
                .emit
                .emit_ldr_imm(Width::X64, Reg::X9, Reg::SP, off)
                .expect("ldr edge tmp");
            self.store_to_vreg(arg.target, Reg::X9);
        }
    }

    fn emit_edge_trampolines(&mut self) {
        let trampolines: Vec<(cfg_mir::EdgeId, LabelId)> = self
            .edge_trampoline_labels
            .iter()
            .map(|(&edge_id, &label)| (edge_id, label))
            .collect();
        for (edge_id, trampoline_label) in trampolines {
            let edge = &self.func.edges[edge_id.index()];
            let target_label = self.block_labels[&edge.to];
            self.ectx.bind_label(trampoline_label);
            self.emit_edge_moves(edge_id);
            self.ectx
                .emit
                .emit_b_label(target_label)
                .expect("b edge target");
        }
    }

    /// Emit a single instruction.
    fn emit_inst(&mut self, inst: &Inst) {
        // Skip instructions whose outputs were fused (bfi, bit-test, etc.)
        match &inst.op {
            LinearOp::BinOp { dst, .. }
            | LinearOp::Const { dst, .. }
            | LinearOp::DataAddr { dst, .. } => {
                if self.fused_skip.contains(dst) {
                    return;
                }
            }
            _ => {}
        }
        match &inst.op {
            LinearOp::Copy { dst, src } => {
                // Elide copy when src and dst are in the same register
                if let (Some(sp), Some(dp)) = (self.preg_for_vreg(*src), self.preg_for_vreg(*dst)) {
                    if sp == dp {
                        return; // nop
                    }
                }
                let src_reg = self.reg_for_vreg_with_temp(*src, Reg::X9);
                self.store_to_vreg(*dst, src_reg);
            }

            LinearOp::Const { dst, value } => {
                // Skip immediate-only consts (operands cleared by elim_imm).
                if inst.operands.is_empty() {
                    return;
                }
                if let Some(preg) = self.preg_for_vreg(*dst) {
                    self.emit_load_u64(self.preg_to_reg(preg), *value);
                } else if self.alloc_func.rematerializable.contains_key(dst) {
                    // Rematerializable constant: skip store to spill slot.
                    // All reads of this vreg will re-emit movz instead.
                } else {
                    // Spilled - load into x9, store to spill slot
                    self.emit_load_u64(Reg::X9, *value);
                    self.store_to_vreg(*dst, Reg::X9);
                }
            }

            LinearOp::DataAddr { dst, blob_id } => {
                // Emit a fixed 4-instruction movz/movk sequence with placeholder 0.
                // The actual address will be patched after JIT finalization.
                let code_offset = self.ectx.emit.code_len();
                let dest_reg = if let Some(preg) = self.preg_for_vreg(*dst) {
                    self.preg_to_reg(preg)
                } else {
                    Reg::X9
                };
                self.emit_load_u64_fixed(dest_reg, 0);
                self.data_relocs.push(DataRelocInfo {
                    code_offset,
                    blob_id: *blob_id,
                });
                if self.preg_for_vreg(*dst).is_none() {
                    self.store_to_vreg(*dst, Reg::X9);
                }
            }

            LinearOp::BinOp { op, dst, lhs, rhs } => {
                self.emit_binop(*op, *dst, *lhs, *rhs);
            }

            LinearOp::UnaryOp { op, dst, src } => {
                self.emit_unary(*op, *dst, *src);
            }

            LinearOp::SaveCursor { dst } => {
                // For leaf functions, load cursor directly from context struct
                // into the regalloc'd register (avoids prologue → x19 → copy).
                if self.ectx.is_leaf {
                    let dst_reg = self.dst_reg_or_temp(*dst, Reg::X9);
                    self.ectx
                        .emit
                        .emit_ldr_imm(Width::X64, dst_reg, self.ctx_reg, CTX_INPUT_PTR)
                        .expect("ldr cursor");
                    if dst_reg == Reg::X9 {
                        self.store_to_vreg(*dst, Reg::X9);
                    }
                } else if let Some(preg) = self.preg_for_vreg(*dst) {
                    let dst_reg = self.preg_to_reg(preg);
                    self.ectx
                        .emit
                        .emit_mov_reg(Width::X64, dst_reg, Reg::X19)
                        .expect("mov");
                } else {
                    self.ectx
                        .emit
                        .emit_mov_reg(Width::X64, Reg::X9, Reg::X19)
                        .expect("mov");
                    self.store_to_vreg(*dst, Reg::X9);
                }
            }

            LinearOp::SaveInputEnd { dst } => {
                // For leaf functions, load input_end directly from context struct.
                if self.ectx.is_leaf {
                    let dst_reg = self.dst_reg_or_temp(*dst, Reg::X9);
                    self.ectx
                        .emit
                        .emit_ldr_imm(Width::X64, dst_reg, self.ctx_reg, CTX_INPUT_END)
                        .expect("ldr input_end");
                    if dst_reg == Reg::X9 {
                        self.store_to_vreg(*dst, Reg::X9);
                    }
                } else if let Some(preg) = self.preg_for_vreg(*dst) {
                    let dst_reg = self.preg_to_reg(preg);
                    self.ectx
                        .emit
                        .emit_mov_reg(Width::X64, dst_reg, Reg::X20)
                        .expect("mov");
                } else {
                    self.ectx
                        .emit
                        .emit_mov_reg(Width::X64, Reg::X9, Reg::X20)
                        .expect("mov");
                    self.store_to_vreg(*dst, Reg::X9);
                }
            }

            LinearOp::RestoreCursor { src } => {
                let cursor_reg = self.cursor_writeback_reg;
                // Check for fused base+offset: emit `add cursor, base, #offset`
                if let Some(&(base_vreg, offset)) = self.fused_addr_offsets.get(src) {
                    let base_reg = self.reg_for_vreg_with_temp(base_vreg, Reg::X9);
                    self.ectx
                        .emit
                        .emit_add_imm(Width::X64, cursor_reg, base_reg, offset as u16, false)
                        .expect("add imm for restore_cursor");
                } else {
                    let src_reg = self.reg_for_vreg_with_temp(*src, Reg::X9);
                    if cursor_reg != src_reg {
                        self.ectx
                            .emit
                            .emit_mov_reg(Width::X64, cursor_reg, src_reg)
                            .expect("mov");
                    }
                }
            }

            LinearOp::BoundsCheck { count } => {
                self.ectx.emit_bounds_check(*count);
            }

            LinearOp::ReadBytes { dst, count } => {
                let rd = self.dst_reg_or_temp(*dst, Reg::X9);
                let cursor = self.ectx.cursor_reg;
                match count {
                    1 => {
                        self.ectx.emit.emit_ldrb_imm(rd, cursor, 0).expect("ldrb");
                    }
                    2 => {
                        self.ectx.emit.emit_ldrh_imm(rd, cursor, 0).expect("ldrh");
                    }
                    4 => {
                        self.ectx
                            .emit
                            .emit_ldr_imm(Width::W32, rd, cursor, 0)
                            .expect("ldr");
                    }
                    8 => {
                        self.ectx
                            .emit
                            .emit_ldr_imm(Width::X64, rd, cursor, 0)
                            .expect("ldr");
                    }
                    _ => {
                        self.ectx.emit.emit_nop().expect("nop");
                        return;
                    }
                }
                if rd == Reg::X9 {
                    self.store_to_vreg(*dst, Reg::X9);
                }
            }

            LinearOp::PeekByte { dst } => {
                let rd = self.dst_reg_or_temp(*dst, Reg::X9);
                let cursor = self.ectx.cursor_reg;
                self.ectx.emit.emit_ldrb_imm(rd, cursor, 0).expect("ldrb");
                if rd == Reg::X9 {
                    self.store_to_vreg(*dst, Reg::X9);
                }
            }

            LinearOp::AdvanceCursor { count } => {
                self.ectx.emit_advance_cursor_by(*count);
            }

            LinearOp::AdvanceCursorBy { src } => {
                let cursor = self.ectx.cursor_reg;
                let src_reg = self.reg_for_vreg_with_temp(*src, Reg::X9);
                self.ectx
                    .emit
                    .emit_add_reg(Width::X64, cursor, cursor, src_reg)
                    .expect("add");
            }

            LinearOp::WriteToField { src, offset, width } => {
                let src_reg = self.reg_for_vreg_with_temp(*src, Reg::X9);
                // Out pointer is in x21
                match width {
                    kajit_ir::Width::W1 => {
                        self.ectx
                            .emit
                            .emit_strb_imm(src_reg, self.output_reg, *offset)
                            .expect("strb");
                    }
                    kajit_ir::Width::W2 => {
                        self.ectx
                            .emit
                            .emit_strh_imm(src_reg, self.output_reg, *offset)
                            .expect("strh");
                    }
                    kajit_ir::Width::W4 => {
                        self.ectx
                            .emit
                            .emit_str_imm(Width::W32, src_reg, self.output_reg, *offset)
                            .expect("str");
                    }
                    kajit_ir::Width::W8 => {
                        self.ectx
                            .emit
                            .emit_str_imm(Width::X64, src_reg, self.output_reg, *offset)
                            .expect("str");
                    }
                }
            }

            LinearOp::ReadFromField { dst, offset, width } => {
                let rd = self.dst_reg_or_temp(*dst, Reg::X9);
                match width {
                    kajit_ir::Width::W1 => {
                        self.ectx
                            .emit
                            .emit_ldrb_imm(rd, self.output_reg, *offset)
                            .expect("ldrb");
                    }
                    kajit_ir::Width::W2 => {
                        self.ectx
                            .emit
                            .emit_ldrh_imm(rd, self.output_reg, *offset)
                            .expect("ldrh");
                    }
                    kajit_ir::Width::W4 => {
                        self.ectx
                            .emit
                            .emit_ldr_imm(Width::W32, rd, self.output_reg, *offset)
                            .expect("ldr");
                    }
                    kajit_ir::Width::W8 => {
                        self.ectx
                            .emit
                            .emit_ldr_imm(Width::X64, rd, self.output_reg, *offset)
                            .expect("ldr");
                    }
                }
                if rd == Reg::X9 {
                    self.store_to_vreg(*dst, Reg::X9);
                }
            }

            LinearOp::SaveOutPtr { dst } => {
                let rd = self.dst_reg_or_temp(*dst, Reg::X9);
                self.ectx
                    .emit
                    .emit_mov_reg(Width::X64, rd, self.output_reg)
                    .expect("mov");
                if rd == Reg::X9 {
                    self.store_to_vreg(*dst, Reg::X9);
                }
            }

            LinearOp::SetOutPtr { src } => {
                let src_reg = self.reg_for_vreg_with_temp(*src, Reg::X9);
                self.ectx
                    .emit
                    .emit_mov_reg(Width::X64, self.output_reg, src_reg)
                    .expect("mov");
            }

            LinearOp::SlotAddr { dst, slot } => {
                let rd = self.dst_reg_or_temp(*dst, Reg::X9);
                let off = self.slot_off(slot.index() as u32);
                self.ectx
                    .emit
                    .emit_add_imm(Width::X64, rd, Reg::SP, off as u16, false)
                    .expect("add");
                if rd == Reg::X9 {
                    self.store_to_vreg(*dst, Reg::X9);
                }
            }

            LinearOp::StoreToAddr { addr, src, width } => {
                let addr_reg = self.reg_for_vreg_with_temp(*addr, Reg::X9);
                let src_reg = self.reg_for_vreg_with_temp(*src, Reg::X10);
                match width {
                    kajit_ir::Width::W1 => {
                        self.ectx
                            .emit
                            .emit_strb_imm(src_reg, addr_reg, 0)
                            .expect("strb");
                    }
                    kajit_ir::Width::W2 => {
                        self.ectx
                            .emit
                            .emit_strh_imm(src_reg, addr_reg, 0)
                            .expect("strh");
                    }
                    kajit_ir::Width::W4 => {
                        self.ectx
                            .emit
                            .emit_str_imm(Width::W32, src_reg, addr_reg, 0)
                            .expect("str");
                    }
                    kajit_ir::Width::W8 => {
                        self.ectx
                            .emit
                            .emit_str_imm(Width::X64, src_reg, addr_reg, 0)
                            .expect("str");
                    }
                }
            }

            LinearOp::LoadFromAddr { dst, addr, width } => {
                // Check for fused base+offset: skip the Add, use [base, #offset]
                let (base_reg, offset) =
                    if let Some(&(base_vreg, off)) = self.fused_addr_offsets.get(addr) {
                        (self.reg_for_vreg_with_temp(base_vreg, Reg::X10), off as u32)
                    } else {
                        (self.reg_for_vreg_with_temp(*addr, Reg::X10), 0)
                    };
                let mut used_scratch = false;
                let assigned = self.dst_reg_or_temp(*dst, Reg::X9);
                let rd = if assigned == base_reg {
                    used_scratch = true;
                    if base_reg != Reg::X9 {
                        Reg::X9
                    } else if base_reg != Reg::X10 {
                        Reg::X10
                    } else {
                        Reg::X11
                    }
                } else {
                    assigned
                };
                match width {
                    kajit_ir::Width::W1 => {
                        self.ectx
                            .emit
                            .emit_ldrb_imm(rd, base_reg, offset)
                            .expect("ldrb");
                    }
                    kajit_ir::Width::W2 => {
                        self.ectx
                            .emit
                            .emit_ldrh_imm(rd, base_reg, offset)
                            .expect("ldrh");
                    }
                    kajit_ir::Width::W4 => {
                        self.ectx
                            .emit
                            .emit_ldr_imm(Width::W32, rd, base_reg, offset)
                            .expect("ldr");
                    }
                    kajit_ir::Width::W8 => {
                        self.ectx
                            .emit
                            .emit_ldr_imm(Width::X64, rd, base_reg, offset)
                            .expect("ldr");
                    }
                }
                if used_scratch || rd == Reg::X9 {
                    self.store_to_vreg(*dst, rd);
                }
            }

            LinearOp::WriteToSlot { slot, src } => {
                let src_reg = self.reg_for_vreg_with_temp(*src, Reg::X9);
                let off = self.slot_off(slot.index() as u32);
                self.ectx
                    .emit
                    .emit_str_imm(Width::X64, src_reg, Reg::SP, off)
                    .expect("str slot");
            }

            LinearOp::ReadFromSlot { dst, slot } => {
                let rd = self.dst_reg_or_temp(*dst, Reg::X9);
                let off = self.slot_off(slot.index() as u32);
                self.ectx
                    .emit
                    .emit_ldr_imm(Width::X64, rd, Reg::SP, off)
                    .expect("ldr slot");
                if rd == Reg::X9 {
                    self.store_to_vreg(*dst, Reg::X9);
                }
            }

            LinearOp::ErrorExit { code } => {
                self.ectx.emit_error_with_ctx_reg(*code, self.ctx_reg);
            }

            LinearOp::CallIntrinsic {
                func,
                args,
                dst,
                field_offset,
            } => {
                self.emit_call_intrinsic(*func, args, *dst, Some(*field_offset));
            }

            LinearOp::CallPure { func, args, dst } | LinearOp::CallEffect { func, args, dst } => {
                self.emit_call_pure(*func, args, *dst);
            }

            LinearOp::CallLambda { .. } => {
                // TODO: multi-function support
                self.ectx.emit.emit_nop().expect("nop");
            }

            LinearOp::SimdStringScan { .. } | LinearOp::SimdWhitespaceSkip => {
                panic!("unsupported SIMD op in regalloc3 aarch64 backend");
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

    /// Emit a binary operation.
    fn emit_binop(
        &mut self,
        kind: BinOpKind,
        dst: kajit_ir::VReg,
        lhs: kajit_ir::VReg,
        rhs: kajit_ir::VReg,
    ) {
        // Comparisons: cmp + cset (with immediate folding)
        if matches!(
            kind,
            BinOpKind::CmpEq
                | BinOpKind::CmpNe
                | BinOpKind::CmpLt
                | BinOpKind::CmpLe
                | BinOpKind::CmpGt
                | BinOpKind::CmpGe
        ) {
            // Fold rhs constant into cmp immediate.
            // If lhs is a small const and rhs is not, swap operands and invert
            // the condition to use cmp imm (e.g., CmpGt(1, x) → cmp x, #1 + Lo).
            let (cmp_lhs, cmp_rhs, swapped) = if self.small_const(rhs).is_some() {
                (lhs, rhs, false)
            } else if self.small_const(lhs).is_some() {
                (rhs, lhs, true)
            } else {
                (lhs, rhs, false)
            };

            let cmp_lhs_reg = self.reg_for_vreg_with_temp(cmp_lhs, Reg::X9);
            if let Some(imm) = self.small_const(cmp_rhs) {
                self.ectx
                    .emit
                    .emit_cmp_imm(Width::X64, cmp_lhs_reg, imm)
                    .expect("cmp imm");
            } else {
                let cmp_rhs_reg = self.reg_for_vreg_with_temp(cmp_rhs, Reg::X10);
                self.ectx
                    .emit
                    .emit_cmp_reg(Width::X64, cmp_lhs_reg, cmp_rhs_reg)
                    .expect("cmp");
            }
            // Skip cset if this comparison is fused with its branch terminator
            if self.fused_cmps.contains_key(&dst) {
                // Update the fused condition if we swapped operands
                if swapped {
                    let cc = self.fused_cmps.get_mut(&dst).unwrap();
                    *cc = cc.swap_operands();
                }
                return;
            }
            let condition = match kind {
                BinOpKind::CmpEq => Condition::Eq,
                BinOpKind::CmpNe => Condition::Ne,
                BinOpKind::CmpLt => Condition::Lo,
                BinOpKind::CmpLe => Condition::Ls,
                BinOpKind::CmpGt => Condition::Hi,
                BinOpKind::CmpGe => Condition::Hs,
                _ => unreachable!(),
            };
            // Swap the condition if we swapped operands
            let condition = if swapped {
                condition.swap_operands()
            } else {
                condition
            };
            // Emit cset directly into dst register if possible
            let cset_dst = if let Some(preg) = self.preg_for_vreg(dst) {
                self.preg_to_reg(preg)
            } else {
                Reg::X9
            };
            self.ectx
                .emit
                .emit_cset(Width::X64, cset_dst, condition)
                .expect("cset");
            if cset_dst == Reg::X9 {
                self.store_to_vreg(dst, Reg::X9);
            }
            return;
        }

        // Try to fold a small constant rhs into an immediate-form instruction
        if let Some(imm) = self.small_const(rhs) {
            if matches!(kind, BinOpKind::Add | BinOpKind::Sub) {
                let lhs_reg = self.reg_for_vreg_with_temp(lhs, Reg::X9);
                let result_reg = if let Some(preg) = self.preg_for_vreg(dst) {
                    self.preg_to_reg(preg)
                } else {
                    Reg::X11
                };
                match kind {
                    BinOpKind::Add => {
                        self.ectx
                            .emit
                            .emit_add_imm(Width::X64, result_reg, lhs_reg, imm, false)
                            .expect("add imm");
                    }
                    BinOpKind::Sub => {
                        self.ectx
                            .emit
                            .emit_sub_imm(Width::X64, result_reg, lhs_reg, imm, false)
                            .expect("sub imm");
                    }
                    _ => unreachable!(),
                }
                if result_reg == Reg::X11 {
                    self.store_to_vreg(dst, result_reg);
                }
                return;
            }
        }

        // Arithmetic/logic: load operands, compute, store
        let lhs_reg = self.reg_for_vreg_with_temp(lhs, Reg::X9);
        let rhs_reg = self.reg_for_vreg_with_temp(rhs, Reg::X10);
        // Compute directly into dst register when possible.
        // On aarch64, ALU instructions read inputs before writing the result,
        // so rd = rn or rd = rm is safe for single-instruction ops.
        let result_reg = if let Some(preg) = self.preg_for_vreg(dst) {
            self.preg_to_reg(preg)
        } else {
            Reg::X11
        };

        match kind {
            BinOpKind::Add => {
                self.ectx
                    .emit
                    .emit_add_reg(Width::X64, result_reg, lhs_reg, rhs_reg)
                    .expect("add");
            }
            BinOpKind::Sub => {
                self.ectx
                    .emit
                    .emit_sub_reg(Width::X64, result_reg, lhs_reg, rhs_reg)
                    .expect("sub");
            }
            BinOpKind::Mul => {
                self.ectx
                    .emit
                    .emit_mul_reg(Width::X64, result_reg, lhs_reg, rhs_reg)
                    .expect("mul");
            }
            BinOpKind::And => {
                // Try logical immediate encoding
                if let Some(&val) = self.const_values.get(&rhs) {
                    if self
                        .ectx
                        .emit
                        .emit_and_imm(Width::X64, result_reg, lhs_reg, val)
                        .is_ok()
                    {
                        if result_reg == Reg::X11 {
                            self.store_to_vreg(dst, result_reg);
                        }
                        return;
                    }
                }
                self.ectx
                    .emit
                    .emit_and_reg(Width::X64, result_reg, lhs_reg, rhs_reg)
                    .expect("and");
            }
            BinOpKind::Or => {
                // Check for bfi fusion: Or(accum, shifted) → bfi(accum, byte_src, lsb, width)
                let bfi_info = self
                    .fused_bfi
                    .get(&dst)
                    .map(|b| (b.byte_src, b.accum, b.lsb, b.width));
                if let Some((byte_src, accum, bfi_lsb, bfi_width)) = bfi_info {
                    let byte_reg = self.reg_for_vreg_with_temp(byte_src, Reg::X9);
                    let accum_reg = self.reg_for_vreg_with_temp(accum, Reg::X10);
                    // bfi modifies Rd in place, so ensure result_reg == accum_reg
                    if result_reg != accum_reg {
                        self.ectx
                            .emit
                            .emit_mov_reg(Width::X64, result_reg, accum_reg)
                            .expect("mov for bfi");
                    }
                    self.ectx
                        .emit
                        .emit_bfi(Width::X64, result_reg, byte_reg, bfi_lsb, bfi_width)
                        .expect("bfi");
                } else {
                    self.ectx
                        .emit
                        .emit_orr_reg(Width::X64, result_reg, lhs_reg, rhs_reg)
                        .expect("orr");
                }
            }
            BinOpKind::Xor => {
                self.ectx
                    .emit
                    .emit_eor_reg(Width::X64, result_reg, lhs_reg, rhs_reg)
                    .expect("eor");
            }
            BinOpKind::Shl => {
                // Try immediate encoding for constant shift amounts
                if let Some(&val) = self.const_values.get(&rhs) {
                    if val < 64 {
                        self.ectx
                            .emit
                            .emit_lsl_imm(Width::X64, result_reg, lhs_reg, val as u8)
                            .expect("lsl imm");
                        if result_reg == Reg::X11 {
                            self.store_to_vreg(dst, result_reg);
                        }
                        return;
                    }
                }
                self.ectx
                    .emit
                    .emit_lsl_reg(Width::X64, result_reg, lhs_reg, rhs_reg)
                    .expect("lsl");
            }
            BinOpKind::Shr => {
                // Try immediate encoding for constant shift amounts
                if let Some(&val) = self.const_values.get(&rhs) {
                    if val < 64 {
                        self.ectx
                            .emit
                            .emit_lsr_imm(Width::X64, result_reg, lhs_reg, val as u8)
                            .expect("lsr imm");
                        if result_reg == Reg::X11 {
                            self.store_to_vreg(dst, result_reg);
                        }
                        return;
                    }
                }
                self.ectx
                    .emit
                    .emit_lsr_reg(Width::X64, result_reg, lhs_reg, rhs_reg)
                    .expect("lsr");
            }
            BinOpKind::Sar => {
                self.ectx
                    .emit
                    .emit_asr_reg(Width::X64, result_reg, lhs_reg, rhs_reg)
                    .expect("asr");
            }
            _ => unreachable!("comparison ops handled above"),
        }

        // Only store if we didn't already compute into the target
        if result_reg == Reg::X11 {
            self.store_to_vreg(dst, result_reg);
        }
    }

    /// Emit a unary operation.
    fn emit_unary(&mut self, kind: UnaryOpKind, dst: kajit_ir::VReg, src: kajit_ir::VReg) {
        let src_reg = self.reg_for_vreg_with_temp(src, Reg::X9);
        match kind {
            UnaryOpKind::ZigzagDecode { wide } => {
                // zigzag_decode(n) = (n >> 1) ^ -(n & 1)
                // x9 = src
                // x10 = src >> 1
                // x11 = -(src & 1) = neg(src & 1)
                // result = x10 ^ x11
                let w = if wide { Width::X64 } else { Width::W32 };
                self.ectx
                    .emit
                    .emit_lsr_imm(w, Reg::X10, src_reg, 1)
                    .expect("lsr");
                // src & 1
                self.ectx
                    .emit
                    .emit_and_imm(w, Reg::X11, src_reg, 1)
                    .expect("and");
                // neg
                self.ectx
                    .emit
                    .emit_neg_reg(w, Reg::X11, Reg::X11)
                    .expect("neg");
                // xor
                self.ectx
                    .emit
                    .emit_eor_reg(w, Reg::X9, Reg::X10, Reg::X11)
                    .expect("eor");
                self.store_to_vreg(dst, Reg::X9);
            }
            UnaryOpKind::SignExtend { from_width } => {
                match from_width {
                    kajit_ir::Width::W1 => {
                        self.ectx
                            .emit
                            .emit_sxtb(Width::X64, Reg::X9, src_reg)
                            .expect("sxtb");
                    }
                    kajit_ir::Width::W2 => {
                        self.ectx
                            .emit
                            .emit_sxth(Width::X64, Reg::X9, src_reg)
                            .expect("sxth");
                    }
                    kajit_ir::Width::W4 => {
                        self.ectx.emit.emit_sxtw(Reg::X9, src_reg).expect("sxtw");
                    }
                    kajit_ir::Width::W8 => {
                        // 64-bit to 64-bit sign extend is a no-op
                        self.ectx
                            .emit
                            .emit_mov_reg(Width::X64, Reg::X9, src_reg)
                            .expect("mov");
                    }
                }
                self.store_to_vreg(dst, Reg::X9);
            }
        }
    }

    /// Emit a call to an intrinsic function.
    /// ABI: x0=ctx, x1+=args, with out_field offset adjustment on x21.
    fn emit_call_intrinsic(
        &mut self,
        func: kajit_ir::IntrinsicFn,
        args: &[kajit_ir::VReg],
        dst: Option<kajit_ir::VReg>,
        field_offset: Option<u32>,
    ) {
        use crate::context::{CTX_ERROR_CODE, CTX_INPUT_PTR};

        let error_exit = self.ectx.error_exit;

        if self.sync_ctx_cursor_around_calls {
            // Legacy cursor-ABI path: host calls observe the cursor through ctx.input_ptr.
            self.ectx
                .emit
                .emit_str_imm(Width::X64, Reg::X19, self.ctx_reg, CTX_INPUT_PTR)
                .expect("str cursor");
        }

        // Adjust out_ptr for field offset if needed
        if let Some(off) = field_offset {
            if off > 0 {
                self.ectx
                    .emit
                    .emit_add_imm(
                        Width::X64,
                        self.output_reg,
                        self.output_reg,
                        off as u16,
                        false,
                    )
                    .expect("add out_ptr");
            }
        }

        // Move register-resident args with parallel-copy semantics first, then
        // materialize spilled/rematerialized args into their ABI homes.
        let mut reg_moves = Vec::new();
        let mut deferred_args = Vec::new();
        for (i, &arg) in args.iter().enumerate() {
            let target_reg = Reg::from_raw((i + 1) as u8);
            if let Some(preg) = self.preg_for_vreg(arg) {
                let src_reg = self.preg_to_reg(preg);
                if src_reg != target_reg {
                    reg_moves.push((target_reg, src_reg));
                }
            } else {
                deferred_args.push((arg, target_reg));
            }
        }
        if !reg_moves.is_empty() {
            self.emit_parallel_moves(&reg_moves, Reg::X16);
        }
        for (arg, target_reg) in deferred_args {
            let src_reg = self.reg_for_vreg_with_temp(arg, target_reg);
            if src_reg != target_reg {
                self.ectx
                    .emit
                    .emit_mov_reg(Width::X64, target_reg, src_reg)
                    .expect("mov arg");
            }
        }

        // If no dst but field_offset, pass output_reg (already adjusted) as out_field arg
        if dst.is_none() {
            if field_offset.is_some() {
                let arg_idx = args.len() + 1;
                self.ectx
                    .emit
                    .emit_mov_reg(Width::X64, Reg::from_raw(arg_idx as u8), self.output_reg)
                    .expect("mov out_field");
            }
        }

        // x0 = ctx
        self.ectx
            .emit
            .emit_mov_reg(Width::X64, Reg::X0, self.ctx_reg)
            .expect("mov ctx");

        // Load function pointer and call
        let call_site_offset = self.ectx.emit.code_len();
        self.emit_load_u64(Reg::X16, func.0 as u64);
        self.ectx.emit.emit_blr(Reg::X16).expect("blr");
        self.intrinsic_call_sites.push(IntrinsicCallSiteInfo {
            code_offset: call_site_offset,
            func,
        });

        // Restore out_ptr
        if let Some(off) = field_offset {
            if off > 0 {
                self.ectx
                    .emit
                    .emit_sub_imm(
                        Width::X64,
                        self.output_reg,
                        self.output_reg,
                        off as u16,
                        false,
                    )
                    .expect("sub out_ptr");
            }
        }

        if self.sync_ctx_cursor_around_calls {
            self.ectx
                .emit
                .emit_ldr_imm(Width::X64, Reg::X19, self.ctx_reg, CTX_INPUT_PTR)
                .expect("ldr cursor");
        }

        // Check error after call
        self.ectx
            .emit
            .emit_ldr_imm(Width::W32, Reg::X9, self.ctx_reg, CTX_ERROR_CODE)
            .expect("ldr error");
        self.ectx
            .emit
            .emit_cbnz_label(Width::W32, Reg::X9, error_exit)
            .expect("cbnz error");

        // Store result if needed (return value is in x0)
        if let Some(dst) = dst {
            self.store_to_vreg(dst, Reg::X0);
        }
    }

    /// Emit a call to a pure/effect function (no ctx, no cursor flush).
    /// The RA has colored args, and edits move them to ABI registers.
    fn emit_call_pure(
        &mut self,
        func: kajit_ir::IntrinsicFn,
        args: &[kajit_ir::VReg],
        dst: kajit_ir::VReg,
    ) {
        // In-register args are already in their ABI registers thanks to RA
        // coloring + OperandEdit moves emitted before this instruction.
        //
        // Spilled/rematerializable args need explicit materialization here:
        // the native regalloc3 path has no separate spill/reload pass.
        for (i, &arg) in args.iter().enumerate() {
            if self.preg_for_vreg(arg).is_none() {
                let abi_reg = Reg::from_raw(i as u8);
                let _ = self.reg_for_vreg_with_temp(arg, abi_reg);
            }
        }

        let call_site_offset = self.ectx.emit.code_len();
        self.emit_load_u64(Reg::X16, func.0 as u64);
        self.ectx.emit.emit_blr(Reg::X16).expect("blr");
        self.intrinsic_call_sites.push(IntrinsicCallSiteInfo {
            code_offset: call_site_offset,
            func,
        });

        self.store_to_vreg(dst, Reg::X0);
    }

    /// Check if a vreg is the result of `And(x, power_of_2)`.
    /// Returns (source_vreg, bit_position) if so.
    fn is_and_bit_test(&self, vreg: kajit_ir::VReg) -> Option<(kajit_ir::VReg, u8)> {
        for inst in &self.func.insts {
            if let LinearOp::BinOp {
                op: BinOpKind::And,
                dst,
                lhs,
                rhs,
            } = &inst.op
            {
                if *dst == vreg {
                    // Check if rhs is a power of 2 constant
                    if let Some(&val) = self.const_values.get(rhs) {
                        if val.is_power_of_two() {
                            return Some((*lhs, val.trailing_zeros() as u8));
                        }
                    }
                    // Check if lhs is a power of 2 constant
                    if let Some(&val) = self.const_values.get(lhs) {
                        if val.is_power_of_two() {
                            return Some((*rhs, val.trailing_zeros() as u8));
                        }
                    }
                }
            }
        }
        None
    }

    /// Emit a conditional branch. When `invert` is false, branches if cond != 0.
    /// When `invert` is true, branches if cond == 0.
    fn emit_branch_cond(&mut self, cond: kajit_ir::VReg, target: LabelId, invert: bool) {
        if let Some(&cc) = self.fused_cmps.get(&cond) {
            let cc = if invert { cc.invert() } else { cc };
            self.ectx.emit.emit_b_cond_label(cc, target).expect("b.cc");
        } else if let Some((src, bit)) = self.is_and_bit_test(cond) {
            let src_reg = self.reg_for_vreg_with_temp(src, Reg::X9);
            if invert {
                self.ectx
                    .emit
                    .emit_tbz_label(src_reg, bit, target)
                    .expect("tbz");
            } else {
                self.ectx
                    .emit
                    .emit_tbnz_label(src_reg, bit, target)
                    .expect("tbnz");
            }
        } else {
            let cond_reg = self.reg_for_vreg_with_temp(cond, Reg::X9);
            if invert {
                self.ectx
                    .emit
                    .emit_cbz_label(Width::X64, cond_reg, target)
                    .expect("cbz");
            } else {
                self.ectx
                    .emit
                    .emit_cbnz_label(Width::X64, cond_reg, target)
                    .expect("cbnz");
            }
        }
    }

    /// Compute which CmpXx vregs can be fused with their branch terminator.
    /// A cmp is fusable if its result vreg is only used by the block's BranchIf/BranchIfZero.
    fn compute_fusable_cmps(func: &Function) -> HashMap<kajit_ir::VReg, Condition> {
        // Count uses of each vreg across the entire function
        let mut use_counts: HashMap<kajit_ir::VReg, usize> = HashMap::new();

        for block in &func.blocks {
            if block.dead {
                continue;
            }
            for &inst_id in &block.insts {
                let inst = &func.insts[inst_id.index()];
                inst.op.for_each_use(|src| {
                    *use_counts.entry(*src).or_default() += 1;
                });
            }
            let term = &func.terms[block.term.0 as usize];
            match term {
                Terminator::BranchIf { cond, .. } | Terminator::BranchIfZero { cond, .. } => {
                    *use_counts.entry(*cond).or_default() += 1;
                }
                Terminator::JumpTable { predicate, .. } => {
                    *use_counts.entry(*predicate).or_default() += 1;
                }
                _ => {}
            }
            for &edge_id in &block.succs {
                let edge = &func.edges[edge_id.index()];
                for arg in &edge.args {
                    *use_counts.entry(arg.source).or_default() += 1;
                }
            }
        }
        for &vreg in &func.data_results {
            *use_counts.entry(vreg).or_default() += 1;
        }

        let mut fusable = HashMap::new();

        for block in &func.blocks {
            if block.dead {
                continue;
            }
            let term = &func.terms[block.term.0 as usize];
            let cond = match term {
                Terminator::BranchIf { cond, .. } | Terminator::BranchIfZero { cond, .. } => *cond,
                _ => continue,
            };

            // Only fuse if the cmp result has exactly 1 use (the terminator)
            if use_counts.get(&cond).copied().unwrap_or(0) != 1 {
                continue;
            }

            // Find the defining CmpXx instruction in this block
            for &inst_id in block.insts.iter().rev() {
                let inst = &func.insts[inst_id.index()];
                if let LinearOp::BinOp { op, dst, .. } = &inst.op {
                    if *dst == cond {
                        let condition = match op {
                            BinOpKind::CmpEq => Some(Condition::Eq),
                            BinOpKind::CmpNe => Some(Condition::Ne),
                            BinOpKind::CmpLt => Some(Condition::Lo),
                            BinOpKind::CmpLe => Some(Condition::Ls),
                            BinOpKind::CmpGt => Some(Condition::Hi),
                            BinOpKind::CmpGe => Some(Condition::Hs),
                            _ => None,
                        };
                        if let Some(cc) = condition {
                            fusable.insert(cond, cc);
                        }
                        break;
                    }
                }
            }
        }

        fusable
    }

    /// Compute which Or instructions can be replaced with bfi.
    /// Pattern: Or(accum, Shl(And(byte, mask), shift)) where mask has consecutive low bits.
    fn compute_fusable_bfis(
        func: &Function,
        const_values: &HashMap<kajit_ir::VReg, u64>,
    ) -> (
        HashMap<kajit_ir::VReg, BfiInfo>,
        std::collections::HashSet<kajit_ir::VReg>,
    ) {
        use std::collections::HashSet;

        // Count uses of each vreg
        let mut use_counts: HashMap<kajit_ir::VReg, usize> = HashMap::new();
        for block in &func.blocks {
            if block.dead {
                continue;
            }
            for &inst_id in &block.insts {
                let inst = &func.insts[inst_id.index()];
                inst.op.for_each_use(|src| {
                    *use_counts.entry(*src).or_default() += 1;
                });
            }
            let term = &func.terms[block.term.0 as usize];
            match term {
                Terminator::BranchIf { cond, .. } | Terminator::BranchIfZero { cond, .. } => {
                    *use_counts.entry(*cond).or_default() += 1;
                }
                Terminator::JumpTable { predicate, .. } => {
                    *use_counts.entry(*predicate).or_default() += 1;
                }
                _ => {}
            }
            for &edge_id in &block.succs {
                let edge = &func.edges[edge_id.index()];
                for arg in &edge.args {
                    *use_counts.entry(arg.source).or_default() += 1;
                }
            }
        }
        for &vreg in &func.data_results {
            *use_counts.entry(vreg).or_default() += 1;
        }

        // Build def map: vreg → defining BinOp instruction
        let mut def_map: HashMap<kajit_ir::VReg, &LinearOp> = HashMap::new();
        for inst in &func.insts {
            if let LinearOp::BinOp { dst, .. } = &inst.op {
                def_map.insert(*dst, &inst.op);
            }
        }

        let mut bfi_map = HashMap::new();
        let mut skip_set = HashSet::new();

        for inst in &func.insts {
            // Look for Or(dst, accum, shifted)
            if let LinearOp::BinOp {
                op: BinOpKind::Or,
                dst,
                lhs: accum,
                rhs: shifted,
            } = &inst.op
            {
                // Check: shifted = Shl(masked, shift_const) where shift_const is known
                let shl_info = if let Some(LinearOp::BinOp {
                    op: BinOpKind::Shl,
                    dst: shl_dst,
                    lhs: masked,
                    rhs: shift_vreg,
                }) = def_map.get(shifted).copied()
                {
                    if let Some(&shift_val) = const_values.get(shift_vreg) {
                        if shift_val <= 63 {
                            Some((*shl_dst, *masked, shift_val as u8))
                        } else {
                            None
                        }
                    } else {
                        None
                    }
                } else {
                    None
                };

                let Some((shl_dst, masked, lsb)) = shl_info else {
                    continue;
                };

                // Check: masked = And(byte, mask_const) where mask_const is (1<<N)-1
                let and_info = if let Some(LinearOp::BinOp {
                    op: BinOpKind::And,
                    dst: and_dst,
                    lhs: and_lhs,
                    rhs: and_rhs,
                }) = def_map.get(&masked).copied()
                {
                    // Try rhs as mask constant
                    if let Some(&mask_val) = const_values.get(and_rhs) {
                        let width = mask_val.count_ones();
                        if width > 0 && width <= 32 && mask_val == (1u64 << width) - 1 {
                            Some((*and_dst, *and_lhs, width as u8))
                        } else {
                            None
                        }
                    }
                    // Try lhs as mask constant
                    else if let Some(&mask_val) = const_values.get(and_lhs) {
                        let width = mask_val.count_ones();
                        if width > 0 && width <= 32 && mask_val == (1u64 << width) - 1 {
                            Some((*and_dst, *and_rhs, width as u8))
                        } else {
                            None
                        }
                    } else {
                        None
                    }
                } else {
                    None
                };

                let Some((and_dst, byte_src, width)) = and_info else {
                    continue;
                };

                // Check that intermediates have single use (consumed only by the chain)
                let and_uses = use_counts.get(&and_dst).copied().unwrap_or(0);
                let shl_uses = use_counts.get(&shl_dst).copied().unwrap_or(0);
                if and_uses != 1 || shl_uses != 1 {
                    continue;
                }

                // bfi requires lsb + width <= 64 (for X64)
                if (lsb as u32) + (width as u32) > 64 {
                    continue;
                }

                bfi_map.insert(
                    *dst,
                    BfiInfo {
                        byte_src,
                        accum: *accum,
                        lsb,
                        width,
                    },
                );
                skip_set.insert(and_dst);
                skip_set.insert(shl_dst);
            }
        }

        (bfi_map, skip_set)
    }

    /// Detect And-bit-test patterns whose results are only used by terminators.
    /// Add the And vreg and its power-of-2 mask const vreg to skip_set so they
    /// don't get emitted as separate instructions (the branch uses tbnz/tbz directly).
    fn compute_fusable_bit_tests(
        func: &Function,
        const_values: &HashMap<kajit_ir::VReg, u64>,
        skip_set: &mut std::collections::HashSet<kajit_ir::VReg>,
    ) {
        // Count uses of each vreg across the entire function
        let mut use_counts: HashMap<kajit_ir::VReg, usize> = HashMap::new();
        for block in &func.blocks {
            if block.dead {
                continue;
            }
            for &inst_id in &block.insts {
                let inst = &func.insts[inst_id.index()];
                inst.op.for_each_use(|src| {
                    *use_counts.entry(*src).or_default() += 1;
                });
            }
            let term = &func.terms[block.term.0 as usize];
            match term {
                Terminator::BranchIf { cond, .. } | Terminator::BranchIfZero { cond, .. } => {
                    *use_counts.entry(*cond).or_default() += 1;
                }
                Terminator::JumpTable { predicate, .. } => {
                    *use_counts.entry(*predicate).or_default() += 1;
                }
                _ => {}
            }
            for &edge_id in &block.succs {
                let edge = &func.edges[edge_id.index()];
                for arg in &edge.args {
                    *use_counts.entry(arg.source).or_default() += 1;
                }
            }
        }
        for &vreg in &func.data_results {
            *use_counts.entry(vreg).or_default() += 1;
        }

        // Find And(x, power_of_2) patterns whose result vreg has exactly 1 use
        // (the terminator) and is not already in skip_set.
        for inst in &func.insts {
            if let LinearOp::BinOp {
                op: BinOpKind::And,
                dst,
                lhs,
                rhs,
            } = &inst.op
            {
                if skip_set.contains(dst) {
                    continue;
                }
                let and_use_count = use_counts.get(dst).copied().unwrap_or(0);
                if and_use_count != 1 {
                    continue;
                }
                // Check if rhs or lhs is a power-of-2 constant
                let mask_vreg = if let Some(&val) = const_values.get(rhs) {
                    if val.is_power_of_two() {
                        Some(*rhs)
                    } else {
                        None
                    }
                } else if let Some(&val) = const_values.get(lhs) {
                    if val.is_power_of_two() {
                        Some(*lhs)
                    } else {
                        None
                    }
                } else {
                    None
                };

                if let Some(mask_vreg) = mask_vreg {
                    // Check that the mask const is also only used by this And
                    let mask_use_count = use_counts.get(&mask_vreg).copied().unwrap_or(0);
                    if mask_use_count == 1 {
                        skip_set.insert(*dst);
                        skip_set.insert(mask_vreg);
                    }
                }
            }
        }
    }

    /// Pre-compute base+offset fusions for LoadFromAddr and RestoreCursor.
    /// When an Add(base, const) result is consumed ONLY by LoadFromAddr or
    /// RestoreCursor, we can skip the Add and use `[base_reg, #offset]` directly.
    fn compute_fusable_addr_offsets(
        func: &Function,
        alloc_func: &AllocatedCfgFunctionRa3,
        const_values: &HashMap<kajit_ir::VReg, u64>,
        skip_set: &mut std::collections::HashSet<kajit_ir::VReg>,
    ) -> HashMap<kajit_ir::VReg, (kajit_ir::VReg, u64)> {
        use kajit_lir::BinOpKind;

        // Count uses of each vreg across all instructions and edge args
        let mut use_counts: HashMap<kajit_ir::VReg, usize> = HashMap::new();
        for inst in &func.insts {
            for op in &inst.operands {
                if op.kind == cfg_mir::OperandKind::Use {
                    *use_counts.entry(op.vreg).or_insert(0) += 1;
                }
            }
        }
        for block in &func.blocks {
            let term = &func.terms[block.term.index()];
            let edge_ids: Vec<cfg_mir::EdgeId> = match term {
                cfg_mir::Terminator::Branch { edge } => vec![*edge],
                cfg_mir::Terminator::BranchIf {
                    taken, fallthrough, ..
                }
                | cfg_mir::Terminator::BranchIfZero {
                    taken, fallthrough, ..
                } => vec![*taken, *fallthrough],
                cfg_mir::Terminator::JumpTable { targets, .. } => targets.clone(),
                _ => vec![],
            };
            for eid in edge_ids {
                let edge = &func.edges[eid.index()];
                for arg in &edge.args {
                    *use_counts.entry(arg.source).or_insert(0) += 1;
                }
            }
        }

        // Build a map: vreg → defining Add(base, const) info
        let mut add_defs: HashMap<kajit_ir::VReg, (kajit_ir::VReg, kajit_ir::VReg)> =
            HashMap::new();
        for inst in &func.insts {
            if let LinearOp::BinOp {
                op: BinOpKind::Add,
                dst,
                lhs,
                rhs,
            } = &inst.op
            {
                add_defs.insert(*dst, (*lhs, *rhs));
            }
        }

        let mut result = HashMap::new();

        // Find LoadFromAddr/RestoreCursor whose addr is defined by Add(base, const)
        for inst in &func.insts {
            let addr_vreg = match &inst.op {
                LinearOp::LoadFromAddr { addr, .. } => *addr,
                LinearOp::RestoreCursor { src } => *src,
                _ => continue,
            };

            // addr must have exactly 1 use (this instruction)
            let addr_uses = use_counts.get(&addr_vreg).copied().unwrap_or(0);
            if addr_uses != 1 {
                if std::env::var("KAJIT_DEBUG_ADDR_FUSION").is_ok() {
                    eprintln!(
                        "[addr-fusion] v{} has {} uses, skip",
                        addr_vreg.index(),
                        addr_uses
                    );
                }
                continue;
            }
            // addr must be defined by an Add
            let Some(&(base, rhs)) = add_defs.get(&addr_vreg) else {
                if std::env::var("KAJIT_DEBUG_ADDR_FUSION").is_ok() {
                    eprintln!("[addr-fusion] v{} not from Add, skip", addr_vreg.index());
                }
                continue;
            };
            // rhs must be a constant ≤ 4095
            let Some(&offset) = const_values.get(&rhs) else {
                if std::env::var("KAJIT_DEBUG_ADDR_FUSION").is_ok() {
                    eprintln!(
                        "[addr-fusion] v{} Add rhs v{} not const, skip",
                        addr_vreg.index(),
                        rhs.index()
                    );
                }
                continue;
            };
            if offset > 4095 {
                continue;
            }
            // The const vreg must be used only by this Add (0 or 1 uses).
            // 0 uses happens when elim_imm already cleared the const operand.
            let rhs_uses = use_counts.get(&rhs).copied().unwrap_or(0);
            if rhs_uses > 1 {
                if std::env::var("KAJIT_DEBUG_ADDR_FUSION").is_ok() {
                    eprintln!(
                        "[addr-fusion] v{} const v{} has {} uses, skip",
                        addr_vreg.index(),
                        rhs.index(),
                        rhs_uses
                    );
                }
                continue;
            }

            // Only fuse when regalloc assigned the temporary address to the
            // same physical register as the base. Otherwise the address vreg
            // has its own live range/home, and reviving the base here can read
            // from a register that has been legitimately reused.
            let Some(addr_preg) = alloc_func.preg_for_vreg(addr_vreg) else {
                continue;
            };
            let Some(base_preg) = alloc_func.preg_for_vreg(base) else {
                continue;
            };
            if addr_preg != base_preg {
                if std::env::var("KAJIT_DEBUG_ADDR_FUSION").is_ok() {
                    eprintln!(
                        "[addr-fusion] v{} != base v{} reg homes (p{} vs p{}), skip",
                        addr_vreg.index(),
                        base.index(),
                        addr_preg.0,
                        base_preg.0
                    );
                }
                continue;
            }

            if std::env::var("KAJIT_DEBUG_ADDR_FUSION").is_ok() {
                eprintln!(
                    "[addr-fusion] FUSE: v{} = v{} + {} → skip Add+Const",
                    addr_vreg.index(),
                    base.index(),
                    offset
                );
            }
            result.insert(addr_vreg, (base, offset));
            skip_set.insert(addr_vreg); // skip the Add
            skip_set.insert(rhs); // skip the Const
        }

        result
    }

    /// Resolve a block ID through trampoline aliases.
    /// If `block_id` is a trampoline (no insts, Branch terminator), follow
    /// the chain to the final non-trampoline target.
    fn resolve_trampoline(&self, mut block_id: cfg_mir::BlockId) -> cfg_mir::BlockId {
        for _ in 0..16 {
            let block = &self.func.blocks[block_id.index()];
            if !block.insts.is_empty() {
                break;
            }
            let term = &self.func.terms[block.term.0 as usize];
            if let Terminator::Branch { edge } = term {
                block_id = self.func.edges[edge.index()].to;
            } else {
                break;
            }
        }
        block_id
    }

    /// Emit a terminator. `next_block` is the block that follows in emission order (for fallthrough elision).
    fn emit_terminator(&mut self, term: &Terminator, next_block: Option<cfg_mir::BlockId>) {
        match term {
            Terminator::Return => {
                // Elide the branch only when the success epilogue is the next
                // emitted code. Edge trampolines are emitted after the block
                // stream, so a "last" return block cannot safely fall through
                // when any trampoline labels were materialized.
                if !self.is_last_emitted_block || !self.edge_trampoline_labels.is_empty() {
                    let success_exit = self.success_exit;
                    self.ectx
                        .emit
                        .emit_b_label(success_exit)
                        .expect("b success");
                }
            }

            Terminator::Branch { edge } => {
                let target_block = self.func.edges[edge.index()].to;
                // Resolve through trampolines for fallthrough elision.
                let resolved = self.resolve_trampoline(target_block);
                if self.edge_has_moves(*edge) {
                    let trampoline =
                        self.edge_target_label(*edge, self.block_labels[&target_block]);
                    self.ectx.emit.emit_b_label(trampoline).expect("branch");
                } else if Some(resolved) != next_block {
                    let label = self.block_labels[&target_block];
                    self.ectx.emit.emit_b_label(label).expect("branch");
                }
            }

            Terminator::BranchIf {
                cond,
                taken,
                fallthrough,
            } => {
                let taken_block = self.func.edges[taken.index()].to;
                let fallthrough_block = self.func.edges[fallthrough.index()].to;
                let taken_label = self.block_labels[&taken_block];
                let fallthrough_label = self.block_labels[&fallthrough_block];

                // Resolve through trampolines for fallthrough elision.
                let resolved_taken = self.resolve_trampoline(taken_block);
                let resolved_fall = self.resolve_trampoline(fallthrough_block);
                let invert =
                    Some(resolved_taken) == next_block && Some(resolved_fall) != next_block;
                let taken_label = self.edge_target_label(*taken, taken_label);
                let fallthrough_label = self.edge_target_label(*fallthrough, fallthrough_label);

                if invert {
                    self.emit_branch_cond(*cond, fallthrough_label, true);
                    self.emit_edge_moves(*taken);
                } else {
                    self.emit_branch_cond(*cond, taken_label, false);
                    if self.edge_has_moves(*fallthrough) {
                        self.emit_edge_moves(*fallthrough);
                        if Some(resolved_fall) != next_block {
                            self.ectx
                                .emit
                                .emit_b_label(fallthrough_label)
                                .expect("b fallthrough");
                        }
                    } else if Some(resolved_fall) != next_block {
                        self.ectx
                            .emit
                            .emit_b_label(fallthrough_label)
                            .expect("b fallthrough");
                    }
                }
            }

            Terminator::BranchIfZero {
                cond,
                taken,
                fallthrough,
            } => {
                let taken_block = self.func.edges[taken.index()].to;
                let fallthrough_block = self.func.edges[fallthrough.index()].to;
                let taken_label = self.block_labels[&taken_block];
                let fallthrough_label = self.block_labels[&fallthrough_block];

                let resolved_taken = self.resolve_trampoline(taken_block);
                let resolved_fall = self.resolve_trampoline(fallthrough_block);
                let invert =
                    Some(resolved_taken) == next_block && Some(resolved_fall) != next_block;
                let taken_label = self.edge_target_label(*taken, taken_label);
                let fallthrough_label = self.edge_target_label(*fallthrough, fallthrough_label);

                if invert {
                    // BranchIfZero inverted = BranchIf → emit non-inverted branch to fallthrough
                    self.emit_branch_cond(*cond, fallthrough_label, false);
                    self.emit_edge_moves(*taken);
                } else {
                    // Normal: BranchIfZero = branch to taken when zero
                    self.emit_branch_cond(*cond, taken_label, true);
                    if self.edge_has_moves(*fallthrough) {
                        self.emit_edge_moves(*fallthrough);
                        if Some(resolved_fall) != next_block {
                            self.ectx
                                .emit
                                .emit_b_label(fallthrough_label)
                                .expect("b fallthrough");
                        }
                    } else if Some(resolved_fall) != next_block {
                        self.ectx
                            .emit
                            .emit_b_label(fallthrough_label)
                            .expect("b fallthrough");
                    }
                }
            }

            Terminator::ErrorExit { code } => {
                self.ectx.emit_error_with_ctx_reg(*code, self.ctx_reg);
            }

            Terminator::JumpTable { .. } => {
                panic!("JumpTable not yet supported in regalloc3 backend");
            }
        }
    }

    /// Emit all blocks for this function.
    fn emit_function(&mut self) {
        // Create labels for all blocks
        for block in &self.func.blocks {
            let label = self.ectx.new_label();
            self.block_labels.insert(block.id, label);
        }

        // Alias trampoline blocks: blocks with no instructions and an
        // unconditional Branch terminator become label aliases for their target.
        for block in &self.func.blocks {
            if block.dead || !block.insts.is_empty() {
                continue;
            }
            let term = &self.func.terms[block.term.0 as usize];
            if let Terminator::Branch { edge } = term {
                let target_block = self.func.edges[edge.index()].to;
                let from_label = self.block_labels[&block.id];
                let to_label = self.block_labels[&target_block];
                self.ectx.emit.alias_label(from_label, to_label);
            }
        }

        // Build emission order: all non-Return blocks first, then Return blocks.
        // This allows the last Return block to fall through into the success epilogue.
        let mut emit_order: Vec<usize> = Vec::new();
        let mut return_blocks: Vec<usize> = Vec::new();
        for block_idx in 0..self.func.blocks.len() {
            let block = &self.func.blocks[block_idx];
            if block.dead {
                continue;
            }
            // Skip trampoline blocks (aliased above, no code to emit).
            if block.insts.is_empty() {
                let term = &self.func.terms[block.term.0 as usize];
                if let Terminator::Branch { .. } = term {
                    continue;
                }
            }
            let term = &self.func.terms[block.term.0 as usize];
            if matches!(term, Terminator::Return) {
                return_blocks.push(block_idx);
            } else {
                emit_order.push(block_idx);
            }
        }
        emit_order.extend(return_blocks);

        // Emit each block in the computed order
        for (emit_idx, &block_idx) in emit_order.iter().enumerate() {
            let block = &self.func.blocks[block_idx];

            // Detect if this is the last block in emission order.
            self.is_last_emitted_block = emit_idx == emit_order.len() - 1;

            // Bind label for this block (except entry which comes after prologue)
            if block.id.0 != 0 {
                let label = self.block_labels[&block.id];
                self.ectx.bind_label(label);
            }

            // Emit instructions with source location tracking
            for &inst_id in &block.insts {
                // Emit OperandEdits (register moves) required before this instruction
                // to satisfy fixed-register operand constraints.
                // Collect all edits for this instruction and emit as a parallel move.
                let edits_here: Vec<(Reg, Reg)> = self
                    .alloc_func
                    .edits
                    .iter()
                    .filter(|e| e.before_inst == inst_id)
                    .map(|e| (Reg::from_raw(e.to.0), Reg::from_raw(e.from.0)))
                    .collect();
                if !edits_here.is_empty() {
                    self.emit_parallel_moves(&edits_here, Reg::X16);
                }

                let op_id = kajit_mir::cfg_mir::OpId::Inst(inst_id);
                if let Some(&line) = self.line_map.get(&op_id) {
                    self.ectx.set_source_location(kajit_emit::SourceLocation {
                        file: 1,
                        line,
                        column: 0,
                    });
                }
                let inst = &self.func.insts[inst_id.index()];
                self.emit_inst(inst);
            }

            // Find next block in emission order (for fallthrough elision)
            let next_block_id = emit_order
                .get(emit_idx + 1)
                .map(|&idx| self.func.blocks[idx].id);

            // Emit terminator with source location
            let term_op = kajit_mir::cfg_mir::OpId::Term(block.term);
            if let Some(&line) = self.line_map.get(&term_op) {
                self.ectx.set_source_location(kajit_emit::SourceLocation {
                    file: 1,
                    line,
                    column: 0,
                });
            }
            let term = &self.func.terms[block.term.0 as usize];
            self.emit_terminator(term, next_block_id);
        }

        self.emit_edge_trampolines();
    }
}

/// Compute the base frame offset for spill slots (past callee-saved register save area).
/// Used by the lockstep debugger to read spilled vregs from the JIT's stack.
pub fn compute_base_frame(alloc: &AllocatedCfgProgramRa3) -> u32 {
    let extra_saved_pairs = regalloc3_extra_saved_pairs(alloc);
    let is_leaf = alloc.cfg_program.funcs.iter().all(|func| {
        func.insts.iter().all(|inst| {
            !matches!(
                inst.op,
                LinearOp::CallIntrinsic { .. }
                    | LinearOp::CallPure { .. }
                    | LinearOp::CallEffect { .. }
                    | LinearOp::CallLambda { .. }
            )
        })
    });
    let base = if is_leaf {
        crate::arch::LEAF_BASE_FRAME
    } else {
        crate::arch::BASE_FRAME
    };
    base + extra_saved_pairs * 16
}

/// Compile CFG-MIR with regalloc3 allocations to aarch64 machine code.
pub fn compile_regalloc3(alloc: &AllocatedCfgProgramRa3) -> LinearBackendResult {
    compile_regalloc3_with_root_data_abi(alloc, crate::compiler::RootDecoderDataAbi::None)
}

pub fn compile_regalloc3_with_root_data_abi(
    alloc: &AllocatedCfgProgramRa3,
    root_data_abi: crate::compiler::RootDecoderDataAbi,
) -> LinearBackendResult {
    let program = &alloc.cfg_program;

    // Calculate max spillslots and extra callee-saved pairs needed
    let max_spillslots = alloc
        .functions
        .iter()
        .map(|f| f.num_spillslots)
        .max()
        .unwrap_or(0);

    // Check which callee-saved registers are used
    let extra_saved_pairs = regalloc3_extra_saved_pairs(alloc);

    // Detect leaf functions (no bl instructions needed)
    let is_leaf = program.funcs.iter().all(|func| {
        func.insts.iter().all(|inst| {
            !matches!(
                inst.op,
                LinearOp::CallIntrinsic { .. }
                    | LinearOp::CallPure { .. }
                    | LinearOp::CallEffect { .. }
                    | LinearOp::CallLambda { .. }
            )
        })
    });

    // Count actually-used slots (slot_count may be stale after slot_to_reg promotion).
    let actual_slot_count = {
        let mut max_slot: Option<u32> = None;
        for func in &program.funcs {
            for inst in &func.insts {
                match &inst.op {
                    LinearOp::WriteToSlot { slot, .. } | LinearOp::ReadFromSlot { slot, .. } => {
                        let s = slot.index() as u32;
                        max_slot = Some(max_slot.map_or(s, |m: u32| m.max(s)));
                    }
                    _ => {}
                }
            }
        }
        max_slot.map_or(0, |m| m + 1)
    };

    let max_edge_args = program
        .funcs
        .iter()
        .flat_map(|func| func.edges.iter().map(|edge| edge.args.len()))
        .max()
        .unwrap_or(0);

    // Create emission context with stack space for spills + actual slots
    let extra_stack = ((max_spillslots + actual_slot_count as usize + max_edge_args) * 8) as u32;
    let mut ectx = EmitCtx::new_regalloc(extra_stack, extra_saved_pairs, is_leaf);
    let slot_base = ectx.base_frame + (max_spillslots * 8) as u32;
    let edge_tmp_base = slot_base + (actual_slot_count as u32 * 8);

    // Check if the program uses cursor operations (BoundsCheck, ReadBytes, etc.)
    // that require the cursor to live in a fixed register.
    let ctx_cursor_abi = matches!(root_data_abi, crate::compiler::RootDecoderDataAbi::None);
    let uses_cursor_ops = ctx_cursor_abi
        && program.funcs.iter().any(|func| {
            func.insts.iter().any(|inst| {
                matches!(
                    inst.op,
                    kajit_lir::LinearOp::BoundsCheck { .. }
                        | kajit_lir::LinearOp::ReadBytes { .. }
                        | kajit_lir::LinearOp::PeekByte { .. }
                        | kajit_lir::LinearOp::AdvanceCursor { .. }
                        | kajit_lir::LinearOp::AdvanceCursorBy { .. }
                )
            })
        });

    // For leaf functions with cursor ops, we still need x19/x20 for the cursor
    // and input end — just skip the callee-save overhead since we're a leaf.
    let leaf_needs_cursor = is_leaf && uses_cursor_ops;
    let cursor_writeback_reg = if is_leaf && !leaf_needs_cursor {
        Reg::X15
    } else {
        Reg::X19
    };

    // Check if regalloc actually uses x19 or x20 for anything.
    let uses_x19_x20 = alloc.functions.iter().any(|f| {
        f.allocations.values().any(|a| {
            matches!(a, kajit_mir::regalloc3::linear_scan::Allocation::Reg(p) if p.0 == 19 || p.0 == 20)
        })
    });
    let need_save_x19_x20 = if is_leaf {
        // Leaf: save x19/x20 if regalloc uses them, or if we load cursor into them.
        uses_x19_x20 || leaf_needs_cursor
    } else {
        // Non-leaf: always save (prologue modifies x19/x20).
        true
    };

    let prologue_config = crate::arch::PrologueConfig {
        save_x21_x22: !is_leaf,
        save_x19_x20: need_save_x19_x20,
        load_cursor_x19_x20: ctx_cursor_abi && (!is_leaf || leaf_needs_cursor),
        cursor_writeback_reg: if is_leaf && !leaf_needs_cursor {
            Some(cursor_writeback_reg)
        } else {
            None
        },
        writeback_cursor_to_ctx: ctx_cursor_abi,
    };

    let is_scalar_function = program.is_scalar;

    // Emit function prologue
    let (entry, error_exit) = if is_scalar_function {
        // Scalar function prologue: frame setup, callee-saved register
        // save, and data_arg moves from ABI registers to RA-assigned registers.
        let entry = ectx.emit.current_offset() as u32;
        let error_exit = ectx.emit.new_label();
        let frame_size = ectx.frame_size;

        let saved_pairs: [(Reg, Reg); 3] = [
            (Reg::X23, Reg::X24),
            (Reg::X25, Reg::X26),
            (Reg::X27, Reg::X28),
        ];
        let pairs_to_save = extra_saved_pairs as usize;

        // Allocate frame: sub sp, sp, total_size
        // Frame layout (low to high):
        //   [sp+0]:  FP/LR save (16 bytes, if non-leaf)
        //   [sp+16]: callee-saved pairs (pairs_to_save * 16 bytes)
        //   [sp+16+pairs*16]: spill slots + user slots (frame_size already accounts for these)
        if frame_size > 0 {
            ectx.emit_sub_imm_any(Reg::SP, Reg::SP, frame_size);
        }

        // Save FP/LR (needed for calls)
        let mut offset: i16 = 0;
        ectx.emit
            .emit_stp(Width::X64, Reg::X29, Reg::X30, Reg::SP, offset)
            .expect("stp fp,lr");
        offset += 16;

        // Save callee-saved pairs
        for i in 0..pairs_to_save {
            ectx.emit
                .emit_stp(
                    Width::X64,
                    saved_pairs[i].0,
                    saved_pairs[i].1,
                    Reg::SP,
                    offset,
                )
                .expect("stp callee-saved");
            offset += 16;
        }

        // Materialize scalar data_args from ABI registers into their assigned homes.
        // Spilled args must be stored before any register shuffles so later moves
        // cannot clobber their ABI source registers.
        if let Some(alloc_func) = alloc.functions.first() {
            if let Some(func) = program.funcs.first() {
                for (i, &arg) in func.data_args.iter().enumerate() {
                    let abi_reg = Reg::from_raw(i as u8);
                    if let Some(slot) = alloc_func.spill_slot_for_vreg(arg) {
                        let offset = ectx.base_frame + (slot.0 * 8);
                        ectx.emit
                            .emit_str_imm(Width::X64, abi_reg, Reg::SP, offset)
                            .expect("str spilled data_arg");
                    }
                }

                let mut arg_moves = Vec::new();
                for (i, &arg) in func.data_args.iter().enumerate() {
                    let abi_reg = Reg::from_raw(i as u8);
                    if let Some(preg) = alloc_func.preg_for_vreg(arg) {
                        let assigned = Reg::from_raw(preg.0);
                        if assigned != abi_reg {
                            arg_moves.push((assigned, abi_reg));
                        }
                    }
                }
                if !arg_moves.is_empty() {
                    emit_parallel_reg_moves(&mut ectx, &arg_moves, Reg::X16);
                }
            }
        }

        ectx.error_exit = error_exit;
        (entry, error_exit)
    } else {
        ectx.begin_func_with_config(&prologue_config)
    };

    if !is_scalar_function {
        if let Some(alloc_func) = alloc.functions.first() {
            if let Some(func) = program.funcs.first() {
                for (i, &arg) in func.data_args.iter().enumerate() {
                    let abi_reg = Reg::from_raw(i as u8 + 2);
                    if let Some(slot) = alloc_func.spill_slot_for_vreg(arg) {
                        let offset = ectx.base_frame + (slot.0 * 8);
                        ectx.emit
                            .emit_str_imm(Width::X64, abi_reg, Reg::SP, offset)
                            .expect("str spilled decoder data_arg");
                    }
                }

                let mut arg_moves = Vec::new();
                for (i, &arg) in func.data_args.iter().enumerate() {
                    let abi_reg = Reg::from_raw(i as u8 + 2);
                    if let Some(preg) = alloc_func.preg_for_vreg(arg) {
                        let assigned = Reg::from_raw(preg.0);
                        if assigned != abi_reg {
                            arg_moves.push((assigned, abi_reg));
                        }
                    }
                }
                if !arg_moves.is_empty() {
                    emit_parallel_reg_moves(&mut ectx, &arg_moves, Reg::X16);
                }
            }
        }
    }

    // Create success exit label
    let success_exit = ectx.new_label();

    // Compile first function
    let mut intrinsic_call_sites = Vec::new();
    let mut data_relocs = Vec::<DataRelocInfo>::new();
    if let (Some(func), Some(alloc_func)) = (program.funcs.first(), alloc.functions.first()) {
        // Build constant value map for immediate folding
        let mut const_values = HashMap::new();
        for inst in &func.insts {
            if let LinearOp::Const { dst, value } = &inst.op {
                const_values.insert(*dst, *value);
            }
        }

        // Build debug line map for source location tracking
        let (line_by_op, _) = super::build_debug_line_maps(program);
        let lambda_id = func.lambda_id.index() as u32;
        let line_map: HashMap<cfg_mir::OpId, u32> = line_by_op
            .iter()
            .filter(|((lid, _), _)| *lid == lambda_id)
            .map(|((_, op_id), &line)| (*op_id, line))
            .collect();

        let fused_cmps = EmitContext::compute_fusable_cmps(func);
        let (fused_bfi, mut fused_skip) = EmitContext::compute_fusable_bfis(func, &const_values);
        EmitContext::compute_fusable_bit_tests(func, &const_values, &mut fused_skip);
        let fused_addr_offsets = EmitContext::compute_fusable_addr_offsets(
            func,
            alloc_func,
            &const_values,
            &mut fused_skip,
        );

        // For leaf functions: keep output_ptr in x0 and ctx_ptr in x1
        // (avoids saving/restoring x21/x22 and the arg moves).
        let (output_reg, ctx_reg) = if is_leaf {
            (Reg::X0, Reg::X1)
        } else {
            (Reg::X21, Reg::X22)
        };

        let mut ctx = EmitContext {
            ectx: &mut ectx,
            func,
            alloc_func,
            block_labels: HashMap::new(),
            success_exit,
            slot_base,
            edge_tmp_base,
            const_values,
            line_map,
            intrinsic_call_sites: Vec::new(),
            data_relocs: Vec::new(),
            fused_cmps,
            fused_bfi,
            fused_skip,
            fused_addr_offsets,
            output_reg,
            ctx_reg,
            sync_ctx_cursor_around_calls: ctx_cursor_abi,
            cursor_writeback_reg,
            is_last_emitted_block: false,
            edge_trampoline_labels: HashMap::new(),
        };

        ctx.emit_function();
        intrinsic_call_sites = ctx.intrinsic_call_sites.clone();
        data_relocs = ctx.data_relocs.clone();
    }

    // Bind success exit and emit epilogue
    ectx.bind_label(success_exit);
    if is_scalar_function {
        // Scalar function epilogue: move data_results to x0, x1, ..., restore frame, ret.
        if let Some(func) = program.funcs.first() {
            if let Some(alloc_func) = alloc.functions.first() {
                // Resolve each result vreg to its physical location.
                let result_regs: Vec<Option<Reg>> = func
                    .data_results
                    .iter()
                    .map(|&vreg| {
                        if let Some(preg) = alloc_func.preg_for_vreg(vreg) {
                            Some(Reg::from_raw(preg.0))
                        } else if let Some(slot) = alloc_func.spill_slot_for_vreg(vreg) {
                            // Load spilled values into scratch first.
                            let offset = ectx.base_frame + (slot.0 * 8);
                            ectx.emit
                                .emit_ldr_imm(Width::X64, Reg::X9, Reg::SP, offset)
                                .expect("ldr result from spill");
                            Some(Reg::X9)
                        } else {
                            None
                        }
                    })
                    .collect();

                // Emit parallel move: check if any target is a source for a
                // later move and use x9 as scratch to break cycles.
                let n = result_regs.len();
                let mut done = vec![false; n];
                for round in 0..n + 1 {
                    let mut progress = false;
                    for i in 0..n {
                        if done[i] {
                            continue;
                        }
                        let target = Reg::from_raw(i as u8);
                        let Some(src) = result_regs[i] else {
                            done[i] = true;
                            continue;
                        };
                        if src == target {
                            done[i] = true;
                            progress = true;
                            continue;
                        }
                        // Check if target is needed as source by an undone move.
                        let blocked =
                            (0..n).any(|j| !done[j] && j != i && result_regs[j] == Some(target));
                        if !blocked || round == n {
                            // If blocked on last round, use scratch to break cycle.
                            if blocked {
                                // Save the blocking value through scratch.
                                let blocker = (0..n)
                                    .find(|&j| !done[j] && j != i && result_regs[j] == Some(target))
                                    .unwrap();
                                ectx.emit
                                    .emit_mov_reg(Width::X64, Reg::X9, target)
                                    .expect("mov scratch");
                                ectx.emit
                                    .emit_mov_reg(Width::X64, target, src)
                                    .expect("mov result");
                                ectx.emit
                                    .emit_mov_reg(Width::X64, Reg::from_raw(blocker as u8), Reg::X9)
                                    .expect("mov from scratch");
                                done[i] = true;
                                done[blocker] = true;
                            } else {
                                ectx.emit
                                    .emit_mov_reg(Width::X64, target, src)
                                    .expect("mov result to return reg");
                                done[i] = true;
                            }
                            progress = true;
                        }
                    }
                    if done.iter().all(|&d| d) || !progress {
                        break;
                    }
                }
            }
        }
        // Restore callee-saved registers and tear down frame.
        let saved_pairs: [(Reg, Reg); 3] = [
            (Reg::X23, Reg::X24),
            (Reg::X25, Reg::X26),
            (Reg::X27, Reg::X28),
        ];
        let pairs_to_save = extra_saved_pairs as usize;
        let frame_size = ectx.frame_size;

        let emit_scalar_epilogue = |ectx: &mut EmitCtx| {
            let mut offset: i16 = 0;
            ectx.emit
                .emit_ldp(Width::X64, Reg::X29, Reg::X30, Reg::SP, offset)
                .expect("ldp fp,lr");
            offset += 16;
            for i in 0..pairs_to_save {
                ectx.emit
                    .emit_ldp(
                        Width::X64,
                        saved_pairs[i].0,
                        saved_pairs[i].1,
                        Reg::SP,
                        offset,
                    )
                    .expect("ldp callee-saved");
                offset += 16;
            }
            if frame_size > 0 {
                ectx.emit_add_imm_any(Reg::SP, Reg::SP, frame_size);
            }
            ectx.emit.emit_ret().expect("ret");
        };

        emit_scalar_epilogue(&mut ectx);

        // Bind error exit (just returns 0 for now).
        ectx.emit.bind_label(error_exit).expect("bind error_exit");
        let zero = Reg::XZR;
        ectx.emit
            .emit_mov_reg(Width::X64, Reg::X0, zero)
            .expect("mov x0, xzr");
        emit_scalar_epilogue(&mut ectx);
    } else {
        ectx.end_func_with_config(error_exit, &prologue_config);
        // Emit shared error trampolines after the epilogue (cold, unreachable
        // from the success/error return paths — only reached via error-site branches)
        ectx.emit_error_trampolines();
    }

    // Append data section to the code buffer (before finalization so it's
    // included in the mmap'd executable buffer).
    let mut data_blob_offsets = Vec::new();
    if !program.data_blobs.is_empty() {
        // Align data section start to 8 bytes.
        let code_end = ectx.emit.code_len();
        let padding = (8 - (code_end % 8)) % 8;
        if padding > 0 {
            ectx.emit.emit_raw_bytes(&vec![0u8; padding]);
        }
        for blob in &program.data_blobs {
            let offset = ectx.emit.code_len();
            data_blob_offsets.push(offset);
            ectx.emit.emit_raw_bytes(blob);
            // Align each blob to 8 bytes.
            let blob_padding = (8 - (blob.len() % 8)) % 8;
            if blob_padding > 0 {
                ectx.emit.emit_raw_bytes(&vec![0u8; blob_padding]);
            }
        }
    }

    // Finalize (resolves branch fixups, creates executable buffer)
    let (buf, asm_program) = ectx.finalize();

    // Patch data address relocations with actual runtime addresses.
    if !data_relocs.is_empty() {
        let base = buf.exec.as_ptr() as u64;
        for reloc in &data_relocs {
            let blob_offset = data_blob_offsets[reloc.blob_id as usize];
            let addr = base + blob_offset as u64;
            unsafe {
                buf.exec.patch_u64_load(reloc.code_offset, addr);
            }
        }
    }

    let source_map = buf.source_map.clone();
    LinearBackendResult {
        buf,
        entry,
        source_map: if source_map.is_empty() {
            None
        } else {
            Some(source_map)
        },
        backend_debug_info: None,
        asm_program,
        intrinsic_call_sites,
        data_relocs,
    }
}

/// Count how many callee-saved register pairs (x23/x24, x25/x26, x27/x28) are used.
fn regalloc3_extra_saved_pairs(alloc: &AllocatedCfgProgramRa3) -> u32 {
    use kajit_mir::regalloc3::linear_scan;

    let mut max_pair = None::<u32>;
    let mut observe = |a: &linear_scan::Allocation| {
        if let linear_scan::Allocation::Reg(preg) = a {
            let pair = match preg.0 {
                23 | 24 => Some(0),
                25 | 26 => Some(1),
                27 | 28 => Some(2),
                _ => None,
            };
            if let Some(pair) = pair {
                max_pair = Some(max_pair.map_or(pair, |cur| cur.max(pair)));
            }
        }
    };

    for func in &alloc.functions {
        for (_, a) in &func.allocations {
            observe(a);
        }
    }

    max_pair.map_or(0, |p| p + 1)
}
