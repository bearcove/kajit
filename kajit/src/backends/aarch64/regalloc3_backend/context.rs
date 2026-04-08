//! EmitContext struct and helper methods for aarch64 regalloc3 backend.

use kajit_emit::aarch64::{Reg, Width};
use kajit_mir::cfg_mir;
use kajit_mir::regalloc3::machine_inst::PReg;
use kajit_mir::regalloc3_result::AllocatedCfgFunctionRa3;

use crate::arch::aarch64::EmitCtx;
use crate::harness::VRegLocation;
use crate::ir_backend::{DataRelocInfo, ExternAddrRelocInfo, IntrinsicCallSiteInfo};
use kajit_emit::aarch64::{Condition, LabelId};
use std::collections::HashMap;

use super::BfiInfo;

/// Physical location for edge move resolution.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub(super) enum EdgeLoc {
    Reg(Reg),
    Stack(u32),
}

/// Context for emitting a single function.
pub(super) struct EmitContext<'a> {
    pub(super) ectx: &'a mut EmitCtx,
    pub(super) func: &'a cfg_mir::Function,
    pub(super) alloc_func: &'a AllocatedCfgFunctionRa3,
    pub(super) block_labels: HashMap<cfg_mir::BlockId, LabelId>,
    pub(super) success_exit: LabelId,
    /// Slot offset base: base_frame + spill_slots * 8 gives the start of user slots.
    pub(super) slot_base: u32,
    /// Scratch stack area used to snapshot edge arguments before delivering them.
    pub(super) edge_tmp_base: u32,
    /// VReg → constant value (for immediate folding in BinOps)
    pub(super) const_values: HashMap<kajit_ir::VReg, u64>,
    /// OpId → DWARF line number (for source-level debugging)
    pub(super) line_map: HashMap<cfg_mir::OpId, u32>,
    /// Recorded intrinsic call sites for harness relocation.
    pub(super) intrinsic_call_sites: Vec<IntrinsicCallSiteInfo>,
    /// Recorded data blob address sites for relocation.
    pub(super) data_relocs: Vec<DataRelocInfo>,
    /// Recorded external symbol address sites for relocation.
    pub(super) extern_addr_relocs: Vec<ExternAddrRelocInfo>,
    /// VRegs whose CmpXx can be fused with the terminator branch (skip cset, emit b.cc).
    pub(super) fused_cmps: HashMap<kajit_ir::VReg, Condition>,
    /// Or vregs that should be emitted as bfi. Maps Or's dst → (byte_src, accum, lsb, width).
    pub(super) fused_bfi: HashMap<kajit_ir::VReg, BfiInfo>,
    /// Intermediate vregs (And/Shl results) whose instructions should be skipped.
    pub(super) fused_skip: std::collections::HashSet<kajit_ir::VReg>,
    /// Fused base+offset info for LoadFromAddr.
    /// Maps addr vreg → (base_vreg, offset). When a LoadFromAddr
    /// consumes an addr that was defined by `Add(base, const_offset)`, the Add
    /// is skipped and the load uses `[base_reg, #offset]` directly.
    pub(super) fused_addr_offsets: HashMap<kajit_ir::VReg, (kajit_ir::VReg, u64)>,
    /// Set to true when emitting the last block before the success epilogue.
    /// Allows Return terminator to fall through instead of branching.
    pub(super) is_last_emitted_block: bool,
    /// Per-edge trampoline labels for edges that need value delivery before control transfer.
    pub(super) edge_trampoline_labels:
        HashMap<cfg_mir::EdgeId, (LabelId, kajit_emit::SourceLocation)>,
    /// Actual source homes for edge arguments at predecessor exit.
    pub(super) edge_source_locations: HashMap<(cfg_mir::EdgeId, u32), VRegLocation>,
    /// Actual source homes for instruction use operands at instruction entry.
    pub(super) inst_source_locations: HashMap<(cfg_mir::InstId, u32), VRegLocation>,
    pub(super) current_inst: Option<cfg_mir::InstId>,
    /// External symbol resolution table.
    pub(super) symbol_table: &'a kajit_types::SymbolTable,
    /// Whether we're emitting for JIT or object file.
    pub(super) compile_target: crate::pipeline_opts::CompileTarget,
    /// Frame offsets for each StackAllocId.
    pub(super) stack_alloc_offsets: Vec<u32>,
}

pub(super) fn emit_parallel_reg_moves(ectx: &mut EmitCtx, moves: &[(Reg, Reg)], temp: Reg) {
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
    pub(super) fn preg_for_vreg(&self, vreg: kajit_ir::VReg) -> Option<PReg> {
        self.alloc_func.preg_for_vreg(vreg)
    }

    /// Convert regalloc3 PReg to kajit_emit Reg.
    pub(super) fn preg_to_reg(&self, preg: PReg) -> Reg {
        Reg::from_raw(preg.0)
    }

    /// Get hardware register for a vreg, or use a temp register and load from spill slot.
    /// For spilled constants, rematerializes with movz instead of loading from stack.
    pub(super) fn reg_for_vreg_with_temp(&mut self, vreg: kajit_ir::VReg, temp: Reg) -> Reg {
        if let Some(inst_id) = self.current_inst
            && let Some(loc) = self
                .inst_source_locations
                .get(&(inst_id, vreg.index() as u32))
                .cloned()
        {
            return self.reg_for_location_with_temp(&loc, temp);
        }

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
    pub(super) fn store_to_vreg(&mut self, vreg: kajit_ir::VReg, from_reg: Reg) {
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
    pub(super) fn dst_reg_or_temp(&self, vreg: kajit_ir::VReg, fallback: Reg) -> Reg {
        if let Some(preg) = self.preg_for_vreg(vreg) {
            self.preg_to_reg(preg)
        } else {
            fallback
        }
    }

    /// Emit a set of parallel register moves using Briggs-style resolution.
    /// `moves` is a list of (dst, src) pairs. `temp` is a scratch register
    /// used to break cycles.
    pub(super) fn emit_parallel_moves(&mut self, moves: &[(Reg, Reg)], temp: Reg) {
        emit_parallel_reg_moves(self.ectx, moves, temp);
    }

    /// Load a 64-bit constant into a register.
    pub(super) fn emit_load_u64(&mut self, rd: Reg, value: u64) {
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
    pub(super) fn emit_load_u64_fixed(&mut self, rd: Reg, value: u64) {
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
    pub(super) fn small_const(&self, vreg: kajit_ir::VReg) -> Option<u16> {
        let value = self.const_values.get(&vreg)?;
        if *value <= 0xFFF {
            Some(*value as u16)
        } else {
            None
        }
    }

    /// Offset of a user slot on the stack.
    pub(super) fn slot_off(&self, slot: u32) -> u32 {
        self.slot_base + slot * 8
    }

    pub(super) fn edge_tmp_off(&self, index: usize) -> u32 {
        self.edge_tmp_base + (index as u32) * 8
    }

    pub(super) fn edge_has_moves(&self, edge_id: cfg_mir::EdgeId) -> bool {
        let edge = &self.func.edges[edge_id.index()];
        edge.args.iter().any(|arg| arg.source != arg.target)
    }

    pub(super) fn edge_target_label(
        &mut self,
        edge_id: cfg_mir::EdgeId,
        target_label: LabelId,
    ) -> LabelId {
        if !self.edge_has_moves(edge_id) {
            return target_label;
        }
        self.edge_trampoline_labels
            .entry(edge_id)
            .or_insert_with(|| (self.ectx.new_label(), self.ectx.current_source_location()))
            .0
    }

    pub(super) fn reg_for_location_with_temp(&mut self, loc: &VRegLocation, temp: Reg) -> Reg {
        match loc {
            VRegLocation::Register(preg) => Reg::from_raw(*preg),
            VRegLocation::StackSlot(offset) => {
                self.ectx
                    .emit
                    .emit_ldr_imm(Width::X64, temp, Reg::SP, *offset)
                    .expect("ldr edge source");
                temp
            }
            VRegLocation::Constant(value) => {
                self.emit_load_u64(temp, *value);
                temp
            }
        }
    }

    pub(super) fn emit_edge_moves(&mut self, edge_id: cfg_mir::EdgeId) {
        let edge = &self.func.edges[edge_id.index()];
        if edge.args.is_empty() {
            return;
        }

        // Resolve each edge arg to physical source and destination locations.
        // X16 and X17 are reserved scratch (never allocated to vregs).

        // Build dependency map: dst → src (keyed by destination location).
        // Constants are emitted immediately since they have no source to conflict.
        let mut deps: HashMap<EdgeLoc, EdgeLoc> = HashMap::new();
        let mut tmp_count = 0usize;

        for arg in &edge.args {
            // Resolve source location
            let src = if let Some(loc) = self
                .edge_source_locations
                .get(&(edge_id, arg.source.index() as u32))
            {
                match loc {
                    VRegLocation::Register(preg) => Some(EdgeLoc::Reg(Reg::from_raw(*preg))),
                    VRegLocation::StackSlot(offset) => Some(EdgeLoc::Stack(*offset)),
                    VRegLocation::Constant(value) => {
                        // Constants: emit immediately, they read no location.
                        let dst = self.resolve_vreg_to_loc(arg.target);
                        if let Some(dst) = dst {
                            self.emit_constant_to_loc(*value, dst);
                        }
                        continue;
                    }
                }
            } else if let Some(preg) = self.preg_for_vreg(arg.source) {
                Some(EdgeLoc::Reg(self.preg_to_reg(preg)))
            } else if let Some(&value) = self.alloc_func.rematerializable.get(&arg.source) {
                let dst = self.resolve_vreg_to_loc(arg.target);
                if let Some(dst) = dst {
                    self.emit_constant_to_loc(value, dst);
                }
                continue;
            } else if let Some(slot) = self.alloc_func.spill_slot_for_vreg(arg.source) {
                Some(EdgeLoc::Stack(self.ectx.base_frame + slot.0 * 8))
            } else {
                None // dead source
            };

            let dst = self.resolve_vreg_to_loc(arg.target);

            match (src, dst) {
                (Some(s), Some(d)) if s != d => {
                    deps.insert(d, s);
                }
                _ => {} // identity or dead
            }
        }

        // Briggs-style parallel copy resolution over physical locations.
        // Same algorithm as emit_parallel_reg_moves but generalized to Loc.
        while !deps.is_empty() {
            // Find a move whose destination is not used as any other move's source.
            let ready = deps
                .iter()
                .find(|(dst, _)| !deps.values().any(|src| src == *dst))
                .map(|(&dst, &src)| (dst, src));

            if let Some((dst, src)) = ready {
                Self::emit_loc_move(self.ectx, src, dst);
                deps.remove(&dst);
                continue;
            }

            // Cycle detected. Break it by saving one destination to a temp.
            let (&cycle_dst, &cycle_src) = deps.iter().next().unwrap();

            // Save cycle_dst's current value (it will be overwritten).
            let saved = match cycle_dst {
                EdgeLoc::Reg(rd) => {
                    // Save register to X17 (reserved scratch)
                    self.ectx
                        .emit
                        .emit_mov_reg(Width::X64, Reg::X17, rd)
                        .expect("mov save");
                    EdgeLoc::Reg(Reg::X17)
                }
                EdgeLoc::Stack(off) => {
                    // Save stack slot to an edge temp slot via X16
                    let tmp_off = self.edge_tmp_off(tmp_count);
                    tmp_count += 1;
                    self.ectx
                        .emit
                        .emit_ldr_imm(Width::X64, Reg::X16, Reg::SP, off)
                        .expect("ldr");
                    self.ectx
                        .emit
                        .emit_str_imm(Width::X64, Reg::X16, Reg::SP, tmp_off)
                        .expect("str");
                    EdgeLoc::Stack(tmp_off)
                }
            };

            // Remove this edge and emit it.
            deps.remove(&cycle_dst);

            // Redirect any other move that reads from cycle_dst to read from saved.
            for (_, src) in deps.iter_mut() {
                if *src == cycle_dst {
                    *src = saved;
                }
            }

            Self::emit_loc_move(self.ectx, cycle_src, cycle_dst);
        }
    }

    /// Resolve a vreg to its physical location (register or stack offset).
    pub(super) fn resolve_vreg_to_loc(&self, vreg: kajit_ir::VReg) -> Option<EdgeLoc> {
        if let Some(preg) = self.preg_for_vreg(vreg) {
            Some(EdgeLoc::Reg(self.preg_to_reg(preg)))
        } else {
            self.alloc_func
                .spill_slot_for_vreg(vreg)
                .map(|slot| EdgeLoc::Stack(self.ectx.base_frame + slot.0 * 8))
        }
    }

    /// Emit a constant value to a location. X16 is used as scratch for stack destinations.
    pub(super) fn emit_constant_to_loc(&mut self, value: u64, dst: EdgeLoc) {
        match dst {
            EdgeLoc::Reg(rd) => {
                self.emit_load_u64(rd, value);
            }
            EdgeLoc::Stack(off) => {
                self.emit_load_u64(Reg::X16, value);
                self.ectx
                    .emit
                    .emit_str_imm(Width::X64, Reg::X16, Reg::SP, off)
                    .expect("str");
            }
        }
    }

    /// Emit a single move between physical locations. X16 is used as scratch
    /// for stack↔stack transfers. Both X16 and X17 are reserved (never allocated).
    pub(super) fn emit_loc_move(ectx: &mut EmitCtx, src: EdgeLoc, dst: EdgeLoc) {
        match (src, dst) {
            (EdgeLoc::Reg(rs), EdgeLoc::Reg(rd)) => {
                ectx.emit.emit_mov_reg(Width::X64, rd, rs).expect("mov");
            }
            (EdgeLoc::Reg(rs), EdgeLoc::Stack(off)) => {
                ectx.emit
                    .emit_str_imm(Width::X64, rs, Reg::SP, off)
                    .expect("str");
            }
            (EdgeLoc::Stack(off), EdgeLoc::Reg(rd)) => {
                ectx.emit
                    .emit_ldr_imm(Width::X64, rd, Reg::SP, off)
                    .expect("ldr");
            }
            (EdgeLoc::Stack(src_off), EdgeLoc::Stack(dst_off)) => {
                ectx.emit
                    .emit_ldr_imm(Width::X64, Reg::X16, Reg::SP, src_off)
                    .expect("ldr");
                ectx.emit
                    .emit_str_imm(Width::X64, Reg::X16, Reg::SP, dst_off)
                    .expect("str");
            }
        }
    }

    pub(super) fn emit_edge_trampolines(&mut self) {
        let trampolines: Vec<(cfg_mir::EdgeId, LabelId, kajit_emit::SourceLocation)> = self
            .edge_trampoline_labels
            .iter()
            .map(|(&edge_id, &(label, source_location))| (edge_id, label, source_location))
            .collect();
        for (edge_id, trampoline_label, source_location) in trampolines {
            let edge = &self.func.edges[edge_id.index()];
            let target_label = self.block_labels[&edge.to];
            self.ectx.set_source_location(source_location);
            self.ectx.bind_label(trampoline_label);
            self.emit_edge_moves(edge_id);
            self.ectx
                .emit
                .emit_b_label(target_label)
                .expect("b edge target");
        }
    }
}
