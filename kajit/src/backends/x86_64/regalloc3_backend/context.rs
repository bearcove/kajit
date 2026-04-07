//! EmitContext — core helpers for the x86_64 regalloc3 backend.

use kajit_emit::x64::{self, LabelId, Mem};
use kajit_mir::cfg_mir::{self, Function};
use kajit_mir::regalloc3::machine_inst::PReg;
use kajit_mir::regalloc3_result::AllocatedCfgFunctionRa3;
use std::collections::HashMap;

use crate::arch::x64::EmitCtx;
use crate::harness::VRegLocation;

use crate::ir_backend::{DataRelocInfo, ExternAddrRelocInfo, IntrinsicCallSiteInfo};

/// Context for emitting a single function.
pub(super) struct EmitContext<'a> {
    pub ectx: &'a mut EmitCtx,
    pub func: &'a Function,
    pub alloc_func: &'a AllocatedCfgFunctionRa3,
    pub block_labels: HashMap<cfg_mir::BlockId, LabelId>,
    pub success_exit: LabelId,
    /// Slot offset base: base_frame + spill_slots * 8 gives the start of user slots.
    pub slot_base: u32,
    /// Scratch stack area used to snapshot edge arguments before delivering them.
    pub edge_tmp_base: u32,
    /// VReg → constant value (for immediate folding in BinOps)
    pub const_values: HashMap<kajit_ir::VReg, u64>,
    /// OpId → DWARF line number (for source-level debugging)
    pub line_map: HashMap<cfg_mir::OpId, u32>,
    /// Recorded intrinsic call sites for harness relocation.
    pub intrinsic_call_sites: Vec<IntrinsicCallSiteInfo>,
    /// Recorded data blob address sites for relocation.
    pub data_relocs: Vec<DataRelocInfo>,
    /// Recorded external symbol address sites for relocation.
    pub extern_addr_relocs: Vec<ExternAddrRelocInfo>,
    /// VRegs whose CmpXx can be fused with the terminator branch (skip setcc, emit jcc directly).
    pub fused_cmps: HashMap<kajit_ir::VReg, x64::Condition>,
    /// Pre-computed base+offset info for LoadFromAddr fusion.
    pub fused_addr_offsets: HashMap<kajit_ir::VReg, (kajit_ir::VReg, u64)>,
    /// Intermediate vregs whose instructions should be skipped (fused away).
    pub fused_skip: std::collections::HashSet<kajit_ir::VReg>,
    /// Context pointer register encoding — derived from data_args[1] RA allocation.
    /// Used by ErrorExit (will be removed in 011-3).
    pub ctx_enc: u8,
    /// Set to true when emitting the last block before the success epilogue.
    pub is_last_emitted_block: bool,
    /// Per-edge trampoline labels for edges that need value delivery before control transfer.
    pub edge_trampoline_labels: HashMap<cfg_mir::EdgeId, (LabelId, kajit_emit::SourceLocation)>,
    /// Actual source homes for edge arguments at predecessor exit.
    pub edge_source_locations: HashMap<(cfg_mir::EdgeId, u32), VRegLocation>,
    /// Actual source homes for instruction use operands at instruction entry.
    pub inst_source_locations: HashMap<(cfg_mir::InstId, u32), VRegLocation>,
    pub current_inst: Option<cfg_mir::InstId>,
}

/// Emit a set of parallel register moves using Briggs-style cycle resolution.
/// `moves` is a list of (dst_enc, src_enc) pairs. `temp_enc` is a scratch register.
pub(super) fn emit_parallel_reg_moves(ectx: &mut EmitCtx, moves: &[(u8, u8)], temp_enc: u8) {
    let mut deps: HashMap<u8, u8> = HashMap::new();
    for &(dst, src) in moves {
        debug_assert!(
            dst < 16,
            "invalid dst register {dst} in parallel move (src={src})"
        );
        debug_assert!(
            src < 16,
            "invalid src register {src} in parallel move (dst={dst})"
        );
        if dst != src {
            deps.insert(dst, src);
        }
    }

    while !deps.is_empty() {
        // Find a ready move (dst is not used as a source by any other pending move).
        let ready = deps
            .iter()
            .find(|(dst, _)| !deps.values().any(|src| src == *dst))
            .map(|(&dst, &src)| (dst, src));

        if let Some((dst, src)) = ready {
            ectx.emit
                .emit_with(|buf| x64::encode_mov_r64_r64(dst, src, buf))
                .expect("mov");
            deps.remove(&dst);
            continue;
        }

        // Cycle: break it with the temp register.
        let (&cycle_dst, &cycle_src) = deps.iter().next().unwrap();
        ectx.emit
            .emit_with(|buf| x64::encode_mov_r64_r64(temp_enc, cycle_dst, buf))
            .expect("mov to temp");
        deps.remove(&cycle_dst);
        for (_, src) in deps.iter_mut() {
            if *src == cycle_dst {
                *src = temp_enc;
            }
        }
        ectx.emit
            .emit_with(|buf| x64::encode_mov_r64_r64(cycle_dst, cycle_src, buf))
            .expect("mov cycle edge");
    }
}

impl<'a> EmitContext<'a> {
    /// Get physical register for a vreg, or None if spilled/dead.
    pub fn preg_for_vreg(&self, vreg: kajit_ir::VReg) -> Option<PReg> {
        self.alloc_func.preg_for_vreg(vreg)
    }

    /// Convert regalloc3 PReg to hardware register encoding.
    pub fn preg_to_enc(&self, preg: PReg) -> u8 {
        preg.0
    }

    /// Get hardware register encoding for a vreg, or use a temp register and load from spill slot.
    /// For spilled constants, rematerializes with mov imm instead of loading from stack.
    pub fn reg_for_vreg_with_temp(&mut self, vreg: kajit_ir::VReg, temp_enc: u8) -> u8 {
        // Check inst-local source locations (from edge/operand edit resolution).
        if let Some(inst_id) = self.current_inst {
            if let Some(loc) = self
                .inst_source_locations
                .get(&(inst_id, vreg.index() as u32))
                .cloned()
            {
                return self.enc_for_location_with_temp(&loc, temp_enc);
            }
        }

        if let Some(preg) = self.preg_for_vreg(vreg) {
            return self.preg_to_enc(preg);
        }

        // Rematerializable constant — emit mov imm instead of stack load
        if let Some(&value) = self.alloc_func.rematerializable.get(&vreg) {
            self.emit_load_u64(temp_enc, value);
            return temp_enc;
        }

        // Spilled — load from spill slot
        if let Some(slot) = self.alloc_func.spill_slot_for_vreg(vreg) {
            let offset = self.ectx.base_frame + (slot.0 * 8);
            self.ectx
                .emit
                .emit_with(|buf| {
                    x64::encode_mov_r64_m(
                        temp_enc,
                        Mem {
                            base: 4,
                            disp: offset as i32,
                        },
                        buf,
                    )
                })
                .expect("mov spill load");
            return temp_enc;
        }

        // Dead vreg — use temp with dummy value
        self.ectx
            .emit
            .emit_with(|buf| x64::encode_xor_r32_r32(temp_enc, temp_enc, buf))
            .expect("xor dead");
        temp_enc
    }

    /// Store a value from a register to a vreg (handling spills).
    pub fn store_to_vreg(&mut self, vreg: kajit_ir::VReg, from_enc: u8) {
        if let Some(preg) = self.preg_for_vreg(vreg) {
            let dst_enc = self.preg_to_enc(preg);
            if dst_enc != from_enc {
                self.ectx
                    .emit
                    .emit_with(|buf| x64::encode_mov_r64_r64(dst_enc, from_enc, buf))
                    .expect("mov");
            }
        } else if let Some(slot) = self.alloc_func.spill_slot_for_vreg(vreg) {
            let offset = self.ectx.base_frame + (slot.0 * 8);
            self.ectx
                .emit
                .emit_with(|buf| {
                    x64::encode_mov_m_r64(
                        Mem {
                            base: 4,
                            disp: offset as i32,
                        },
                        from_enc,
                        buf,
                    )
                })
                .expect("mov spill store");
        }
        // If dead, do nothing
    }

    /// Get the destination register encoding for a vreg def.
    /// Returns the allocated register if available, or the fallback temp.
    pub fn dst_enc_or_temp(&self, vreg: kajit_ir::VReg, fallback: u8) -> u8 {
        if let Some(preg) = self.preg_for_vreg(vreg) {
            self.preg_to_enc(preg)
        } else {
            fallback
        }
    }

    /// Emit a set of parallel register moves.
    pub fn emit_parallel_moves(&mut self, moves: &[(u8, u8)], temp_enc: u8) {
        emit_parallel_reg_moves(self.ectx, moves, temp_enc);
    }

    /// Load a 64-bit constant into a register.
    pub fn emit_load_u64(&mut self, enc: u8, value: u64) {
        if value == 0 {
            self.ectx
                .emit
                .emit_with(|buf| x64::encode_xor_r32_r32(enc, enc, buf))
                .expect("xor zero");
        } else if value <= 0xFFFF_FFFF {
            self.ectx
                .emit
                .emit_with(|buf| x64::encode_mov_r32_imm32(enc, value as u32, buf))
                .expect("mov imm32");
        } else {
            self.ectx
                .emit
                .emit_with(|buf| x64::encode_mov_r64_imm64(enc, value, buf))
                .expect("mov imm64");
        }
    }

    /// Load a 64-bit value into a register using a fixed 10-byte `mov r64, imm64`.
    /// The fixed size makes the sequence relocatable.
    pub fn emit_load_u64_fixed(&mut self, enc: u8, value: u64) {
        self.ectx
            .emit
            .emit_with(|buf| x64::encode_mov_r64_imm64(enc, value, buf))
            .expect("mov imm64 fixed");
    }

    /// If vreg is a known constant that fits in a 32-bit immediate, return its value.
    pub fn imm32_const(&self, vreg: kajit_ir::VReg) -> Option<u32> {
        let value = self.const_values.get(&vreg)?;
        if *value <= 0xFFFF_FFFF {
            Some(*value as u32)
        } else {
            None
        }
    }

    /// Offset of a user slot on the stack.
    pub fn slot_off(&self, slot: u32) -> u32 {
        self.slot_base + slot * 8
    }

    pub fn edge_tmp_off(&self, index: usize) -> u32 {
        self.edge_tmp_base + (index as u32) * 8
    }

    pub fn edge_has_moves(&self, edge_id: cfg_mir::EdgeId) -> bool {
        let edge = &self.func.edges[edge_id.index()];
        edge.args.iter().any(|arg| arg.source != arg.target)
    }

    pub fn edge_target_label(
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

    pub fn enc_for_location_with_temp(&mut self, loc: &VRegLocation, temp_enc: u8) -> u8 {
        match loc {
            VRegLocation::Register(preg) => *preg,
            VRegLocation::StackSlot(offset) => {
                self.ectx
                    .emit
                    .emit_with(|buf| {
                        x64::encode_mov_r64_m(
                            temp_enc,
                            Mem {
                                base: 4,
                                disp: *offset as i32,
                            },
                            buf,
                        )
                    })
                    .expect("mov edge source");
                temp_enc
            }
            VRegLocation::Constant(value) => {
                self.emit_load_u64(temp_enc, *value);
                temp_enc
            }
        }
    }

    pub fn emit_edge_moves(&mut self, edge_id: cfg_mir::EdgeId) {
        let edge = &self.func.edges[edge_id.index()];
        if edge.args.is_empty() {
            return;
        }

        if std::env::var("KAJIT_EDGE_DEBUG").is_ok() {
            eprintln!(
                "[edge_moves] edge {:?} ({:?} -> {:?})",
                edge_id, edge.from, edge.to
            );
            for arg in &edge.args {
                let src_preg = self.preg_for_vreg(arg.source);
                let tgt_preg = self.preg_for_vreg(arg.target);
                let src_loc = self
                    .edge_source_locations
                    .get(&(edge_id, arg.source.index() as u32));
                eprintln!(
                    "  {:?} => {:?}: src_preg={:?}, tgt_preg={:?}, src_loc={:?}",
                    arg.source, arg.target, src_preg, tgt_preg, src_loc
                );
            }
        }

        // Snapshot all sources to temp stack slots.
        for (index, arg) in edge.args.iter().enumerate() {
            let src_enc = self
                .edge_source_locations
                .get(&(edge_id, arg.source.index() as u32))
                .cloned()
                .map(|loc| self.enc_for_location_with_temp(&loc, 10))
                .unwrap_or_else(|| self.reg_for_vreg_with_temp(arg.source, 10));
            let off = self.edge_tmp_off(index) as i32;
            self.ectx
                .emit
                .emit_with(|buf| x64::encode_mov_m_r64(Mem { base: 4, disp: off }, src_enc, buf))
                .expect("mov edge tmp store");
        }

        // Deliver from temp slots to targets.
        for (index, arg) in edge.args.iter().enumerate() {
            let off = self.edge_tmp_off(index) as i32;
            self.ectx
                .emit
                .emit_with(|buf| x64::encode_mov_r64_m(10, Mem { base: 4, disp: off }, buf))
                .expect("mov edge tmp load");
            self.store_to_vreg(arg.target, 10);
        }
    }

    pub fn emit_edge_trampolines(&mut self) {
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
                .emit_jmp_label(target_label)
                .expect("jmp edge target");
        }
    }

    /// Resolve a block ID through trampoline aliases.
    pub fn resolve_trampoline(&self, mut block_id: cfg_mir::BlockId) -> cfg_mir::BlockId {
        for _ in 0..16 {
            let block = &self.func.blocks[block_id.index()];
            if !block.insts.is_empty() {
                break;
            }
            let term = &self.func.terms[block.term.0 as usize];
            if let cfg_mir::Terminator::Branch { edge } = term {
                if self.edge_has_moves(*edge) {
                    break;
                }
                block_id = self.func.edges[edge.index()].to;
            } else {
                break;
            }
        }
        block_id
    }
}
