#![allow(dead_code)]
//! Standalone test harness generator.
//!
//! Generates a native executable containing:
//! - The JIT-compiled decoder code in `.text`
//! - DWARF debug sections (`.debug_line`, `.debug_info`, `.debug_abbrev`)
//! - A C harness wrapper that sets up input/output and calls the decoder
//!
//! Usage: `kajit compile postcard u32 -s harness`
//! Produces: `harness_postcard_u32` executable + source listing

use std::collections::{BTreeSet, HashMap};
use std::path::Path;

/// Where a vreg lives after register allocation.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum VRegLocation {
    /// Physical register (aarch64 GPR index: 0=x0, 1=x1, ..., 28=x28)
    Register(u8),
    /// Stack slot (offset from frame pointer in bytes)
    StackSlot(u32),
    /// Rematerializable constant (re-emitted as movz/movk, not loaded from stack)
    Constant(u64),
}

/// Maps vreg index → physical location. Used by the lockstep debugger
/// to read JIT register/stack state and compare with interpreter vreg values.
#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize)]
pub struct AllocationMap {
    /// vreg_index → location
    pub locations: HashMap<u32, VRegLocation>,
    /// Number of spill slots allocated (for computing frame size)
    pub num_spill_slots: usize,
}

impl AllocationMap {
    /// Build from a regalloc3 allocation result.
    ///
    /// `base_frame` is the stack offset past the callee-saved register save area.
    /// Spill slots are at `[sp + base_frame + slot * 8]`.
    pub fn from_regalloc3(
        alloc: &kajit_mir::regalloc3_result::AllocatedCfgFunctionRa3,
        base_frame: u32,
    ) -> Self {
        let mut locations = HashMap::new();

        for (&vreg, allocation) in &alloc.allocations {
            match allocation {
                kajit_mir::regalloc3::linear_scan::Allocation::Reg(preg) => {
                    locations.insert(vreg.index() as u32, VRegLocation::Register(preg.0));
                }
                kajit_mir::regalloc3::linear_scan::Allocation::Spill => {
                    // Check if it's rematerializable first
                    if let Some(&value) = alloc.rematerializable.get(&vreg) {
                        locations.insert(vreg.index() as u32, VRegLocation::Constant(value));
                    } else if let Some(slot) = alloc.spill_slots.get(&vreg) {
                        locations.insert(
                            vreg.index() as u32,
                            VRegLocation::StackSlot(base_frame + slot.0 * 8),
                        );
                    }
                }
            }
        }

        Self {
            locations,
            num_spill_slots: alloc.num_spillslots,
        }
    }

    /// Write as JSON to a file.
    pub fn write_json(&self, path: &Path) -> Result<(), std::io::Error> {
        let json = serde_json::to_string_pretty(self).map_err(std::io::Error::other)?;
        std::fs::write(path, json)
    }

    /// Get the aarch64 register name for a physical register index.
    pub fn reg_name(preg: u8) -> &'static str {
        match preg {
            0 => "x0",
            1 => "x1",
            2 => "x2",
            3 => "x3",
            4 => "x4",
            5 => "x5",
            6 => "x6",
            7 => "x7",
            8 => "x8",
            9 => "x9",
            10 => "x10",
            11 => "x11",
            12 => "x12",
            13 => "x13",
            14 => "x14",
            15 => "x15",
            16 => "x16",
            17 => "x17",
            18 => "x18",
            19 => "x19",
            20 => "x20",
            21 => "x21",
            22 => "x22",
            23 => "x23",
            24 => "x24",
            25 => "x25",
            26 => "x26",
            27 => "x27",
            28 => "x28",
            29 => "fp",
            30 => "lr",
            31 => "sp",
            _ => "???",
        }
    }
}

/// Per-program-point vreg location map for the lockstep debugger.
///
/// Extends the static AllocationMap with call-clobber awareness: at DWARF lines
/// that contain call instructions, caller-saved registers (x0-x18) are clobbered
/// by the ABI. The only valid caller-saved register after a call is the return
/// value's register (if any).
///
/// This replaces reading from a static vreg→register map, which gives false
/// divergences when the lockstep reads a clobbered register after a call.
#[derive(Debug, Clone, Default)]
pub struct LocationMap {
    /// Static allocation for each vreg (same data as AllocationMap.locations).
    pub static_locations: HashMap<u32, VRegLocation>,
    /// DWARF lines that contain call instructions (CallIntrinsic, CallPure, CallEffect, CallLambda).
    /// At these lines, caller-saved registers are clobbered after execution.
    pub call_lines: std::collections::HashSet<u32>,
    /// For each call line, the return value vreg index (if any).
    /// This vreg IS valid in its allocated register after the call.
    pub call_return_vregs: HashMap<u32, u32>,
    /// DWARF lines with pre-op regalloc edits that overwrite fixed registers.
    /// These locations are invalidated before the op result is written back.
    pub edit_clobbers: HashMap<u32, Vec<VRegLocation>>,
    pub num_spill_slots: usize,
}

impl LocationMap {
    /// Look up the location of a vreg after a specific DWARF line has executed.
    ///
    /// Returns `None` if the vreg is in a clobbered register at a call site
    /// (meaning its value cannot be reliably read from the JIT at this point).
    pub fn location_at(&self, dwarf_line: u32, vreg_idx: u32) -> Option<&VRegLocation> {
        let loc = self.static_locations.get(&vreg_idx)?;

        if self.call_lines.contains(&dwarf_line)
            && let VRegLocation::Register(preg) = loc
        {
            // Caller-saved registers on aarch64: x0-x18
            // (x19-x28 are callee-saved, x29=fp, x30=lr)
            if *preg <= 18 {
                // The return value vreg is valid — the backend stores x0 → dst register
                if self.call_return_vregs.get(&dwarf_line) == Some(&vreg_idx) {
                    return Some(loc);
                }
                // All other vregs in caller-saved registers: clobbered by the call
                return None;
            }
        }

        Some(loc)
    }

    fn key_for(loc: &VRegLocation) -> Option<LocationKey> {
        match loc {
            VRegLocation::Register(preg) => Some(LocationKey::Register(*preg)),
            VRegLocation::StackSlot(offset) => Some(LocationKey::StackSlot(*offset)),
            VRegLocation::Constant(_) => None,
        }
    }

    fn assign_owner(
        owners: &mut HashMap<LocationKey, u32>,
        static_locations: &HashMap<u32, VRegLocation>,
        vreg_idx: u32,
    ) {
        let Some(loc) = static_locations.get(&vreg_idx) else {
            return;
        };
        let Some(key) = Self::key_for(loc) else {
            return;
        };
        owners.insert(key, vreg_idx);
    }

    /// Build from an AllocationMap and CFG program.
    ///
    /// Walks the CFG in the same order as `build_debug_line_maps` to assign
    /// DWARF line numbers, then identifies call sites and their return vregs.
    pub fn from_alloc_map_and_cfg(
        alloc_map: &AllocationMap,
        program: &kajit_mir::ir::Program,
        alloc: &kajit_mir::regalloc3_result::AllocatedCfgProgramRa3,
    ) -> Self {
        use kajit_lir::LinearOp;
        use std::collections::HashSet;

        let mut call_lines = HashSet::new();
        let mut call_return_vregs = HashMap::new();
        let mut inst_lines = HashMap::new();

        for func in &program.funcs {
            let lambda_id = func.lambda_id.index() as u32;
            let mut next_line = 1u32;
            for block in func.live_blocks() {
                for inst_id in &block.insts {
                    let inst = &func.insts[inst_id.index()];
                    inst_lines.insert((lambda_id, *inst_id), next_line);
                    match &inst.op {
                        LinearOp::CallIntrinsic { dst, .. } => {
                            call_lines.insert(next_line);
                            if let Some(dst) = dst {
                                call_return_vregs.insert(next_line, dst.index() as u32);
                            }
                        }
                        LinearOp::CallPure { dst, .. } | LinearOp::CallEffect { dst, .. } => {
                            call_lines.insert(next_line);
                            call_return_vregs.insert(next_line, dst.index() as u32);
                        }
                        LinearOp::CallLambda { results, .. } => {
                            call_lines.insert(next_line);
                            // First result is the primary return value
                            if let Some(first) = results.first() {
                                call_return_vregs.insert(next_line, first.index() as u32);
                            }
                        }
                        _ => {}
                    }
                    next_line += 1;
                }
                // Terminator line
                next_line += 1;
            }
        }

        let mut edit_clobbers = HashMap::<u32, Vec<VRegLocation>>::new();

        for alloc_func in &alloc.functions {
            let lambda_id = alloc_func.lambda_id.index() as u32;
            for edit in &alloc_func.edits {
                let Some(line) = inst_lines.get(&(lambda_id, edit.before_inst)).copied() else {
                    continue;
                };
                edit_clobbers
                    .entry(line)
                    .or_default()
                    .push(VRegLocation::Register(edit.to.0));
            }
        }

        Self {
            static_locations: alloc_map.locations.clone(),
            call_lines,
            call_return_vregs,
            edit_clobbers,
            num_spill_slots: alloc_map.num_spill_slots,
        }
    }

    /// Get the aarch64 register name for a physical register index.
    pub fn reg_name(preg: u8) -> &'static str {
        AllocationMap::reg_name(preg)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum LocationKey {
    Register(u8),
    StackSlot(u32),
}

/// Tracks which vreg currently owns each architectural location along the
/// executed program path. This makes lockstep comparisons path-sensitive:
/// once an op or edge writes a register/stack slot, previous occupants are no
/// longer readable from that location.
#[derive(Debug, Clone, Default)]
pub struct LocationTracker {
    owners: HashMap<LocationKey, BTreeSet<u32>>,
}

impl LocationTracker {
    pub fn new(map: &LocationMap, program: &kajit_mir::ir::Program) -> Self {
        let mut owners = HashMap::new();

        if let Some(func) = program.funcs.first() {
            for &vreg in &func.data_args {
                assign_aliases(
                    &mut owners,
                    &map.static_locations,
                    vreg.index() as u32,
                    BTreeSet::from([vreg.index() as u32]),
                );
            }
            if let Some(entry) = func.blocks.get(func.entry.index()) {
                for &vreg in &entry.params {
                    assign_aliases(
                        &mut owners,
                        &map.static_locations,
                        vreg.index() as u32,
                        BTreeSet::from([vreg.index() as u32]),
                    );
                }
            }
        }

        Self { owners }
    }

    pub fn location_for(&self, map: &LocationMap, vreg_idx: u32) -> Option<VRegLocation> {
        current_location_from_owners(&self.owners, map, vreg_idx)
    }

    pub fn owner_of_vreg_location(&self, map: &LocationMap, vreg_idx: u32) -> Option<u32> {
        let loc = self.location_for(map, vreg_idx)?;
        self.owner_of_location(&loc)
    }

    pub fn owner_of_location(&self, loc: &VRegLocation) -> Option<u32> {
        let key = LocationMap::key_for(loc)?;
        self.owners.get(&key)?.iter().next().copied()
    }

    pub fn observe_step(
        &mut self,
        map: &LocationMap,
        func: &kajit_mir::ir::Function,
        executed_line: u32,
        loc_before: &kajit_mir::ProgramLocation,
        loc_after: &kajit_mir::ProgramLocation,
    ) {
        if let Some(clobbers) = map.edit_clobbers.get(&executed_line) {
            for loc in clobbers {
                self.invalidate_location(loc);
            }
        }

        if map.call_lines.contains(&executed_line) {
            self.owners
                .retain(|key, _| !matches!(key, LocationKey::Register(preg) if *preg <= 18));
        }

        if loc_before.at_terminator {
            if let Some(edge) = chosen_edge(func, loc_before, loc_after) {
                for arg in &edge.args {
                    let aliases = aliases_for_vreg(&self.owners, arg.source.index() as u32)
                        .unwrap_or_else(|| BTreeSet::from([arg.source.index() as u32]));
                    assign_aliases(
                        &mut self.owners,
                        &map.static_locations,
                        arg.target.index() as u32,
                        aliases,
                    );
                }
            }
            return;
        }

        if let Some(block) = func.blocks.get(loc_before.block.index())
            && let Some(&inst_id) = block.insts.get(loc_before.next_inst_index)
        {
            let inst = &func.insts[inst_id.index()];
            if let Some(def_vreg) = op_def_vreg(func, loc_before) {
                let aliases = match inst.op {
                    kajit_lir::LinearOp::Copy { src, .. } => {
                        aliases_for_vreg(&self.owners, src.index() as u32)
                            .unwrap_or_else(|| BTreeSet::from([src.index() as u32]))
                    }
                    _ => BTreeSet::from([def_vreg.index() as u32]),
                };
                assign_aliases(
                    &mut self.owners,
                    &map.static_locations,
                    def_vreg.index() as u32,
                    aliases,
                );
            }
        }
    }

    fn invalidate_location(&mut self, loc: &VRegLocation) {
        if let Some(key) = LocationMap::key_for(loc) {
            self.owners.remove(&key);
        }
    }
}

fn current_location_from_owners(
    owners: &HashMap<LocationKey, BTreeSet<u32>>,
    map: &LocationMap,
    vreg_idx: u32,
) -> Option<VRegLocation> {
    let loc = map.static_locations.get(&vreg_idx)?;
    if let VRegLocation::Constant(value) = loc {
        return Some(VRegLocation::Constant(*value));
    }

    match loc {
        VRegLocation::Register(preg)
            if owners
                .get(&LocationKey::Register(*preg))
                .is_some_and(|aliases| aliases.contains(&vreg_idx)) =>
        {
            return Some(VRegLocation::Register(*preg));
        }
        VRegLocation::StackSlot(offset)
            if owners
                .get(&LocationKey::StackSlot(*offset))
                .is_some_and(|aliases| aliases.contains(&vreg_idx)) =>
        {
            return Some(VRegLocation::StackSlot(*offset));
        }
        _ => {}
    }

    for (key, aliases) in owners {
        if !aliases.contains(&vreg_idx) {
            continue;
        }
        match key {
            LocationKey::Register(preg) => return Some(VRegLocation::Register(*preg)),
            LocationKey::StackSlot(offset) => return Some(VRegLocation::StackSlot(*offset)),
        }
    }

    None
}

fn merge_owner_maps(
    dst: &mut HashMap<LocationKey, BTreeSet<u32>>,
    src: &HashMap<LocationKey, BTreeSet<u32>>,
) -> bool {
    if dst.is_empty() {
        *dst = src.clone();
        return !dst.is_empty();
    }

    let before = dst.clone();
    dst.retain(|key, aliases| {
        let Some(src_aliases) = src.get(key) else {
            return false;
        };
        aliases.retain(|vreg| src_aliases.contains(vreg));
        !aliases.is_empty()
    });
    *dst != before
}

fn aliases_for_vreg(
    owners: &HashMap<LocationKey, BTreeSet<u32>>,
    vreg_idx: u32,
) -> Option<BTreeSet<u32>> {
    let mut aliases = BTreeSet::new();
    for members in owners.values() {
        if members.contains(&vreg_idx) {
            aliases.extend(members.iter().copied());
        }
    }
    (!aliases.is_empty()).then_some(aliases)
}

fn remove_vreg_from_all_locations(owners: &mut HashMap<LocationKey, BTreeSet<u32>>, vreg_idx: u32) {
    owners.retain(|_, aliases| {
        aliases.remove(&vreg_idx);
        !aliases.is_empty()
    });
}

fn assign_aliases(
    owners: &mut HashMap<LocationKey, BTreeSet<u32>>,
    static_locations: &HashMap<u32, VRegLocation>,
    vreg_idx: u32,
    mut aliases: BTreeSet<u32>,
) {
    remove_vreg_from_all_locations(owners, vreg_idx);
    aliases.insert(vreg_idx);
    let Some(loc) = static_locations.get(&vreg_idx) else {
        return;
    };
    let Some(key) = LocationMap::key_for(loc) else {
        return;
    };
    owners.insert(key, aliases);
}

fn apply_block_transfer(
    owners: &mut HashMap<LocationKey, BTreeSet<u32>>,
    map: &LocationMap,
    func: &kajit_mir::ir::Function,
    block: &kajit_mir::ir::Block,
    inst_lines: &HashMap<kajit_mir::ir::InstId, u32>,
) {
    for &inst_id in &block.insts {
        let Some(&line) = inst_lines.get(&inst_id) else {
            continue;
        };

        if let Some(clobbers) = map.edit_clobbers.get(&line) {
            for loc in clobbers {
                if let Some(key) = LocationMap::key_for(loc) {
                    owners.remove(&key);
                }
            }
        }

        if map.call_lines.contains(&line) {
            owners.retain(|key, _| !matches!(key, LocationKey::Register(preg) if *preg <= 18));
        }

        let inst = &func.insts[inst_id.index()];
        if let Some(def) = inst
            .operands
            .iter()
            .find(|operand| operand.kind == kajit_mir::ir::OperandKind::Def)
            .map(|operand| operand.vreg)
        {
            let aliases = match inst.op {
                kajit_lir::LinearOp::Copy { src, .. } => {
                    aliases_for_vreg(owners, src.index() as u32)
                        .unwrap_or_else(|| BTreeSet::from([src.index() as u32]))
                }
                _ => BTreeSet::from([def.index() as u32]),
            };
            assign_aliases(owners, &map.static_locations, def.index() as u32, aliases);
        }
    }
}

pub fn compute_edge_source_locations(
    map: &LocationMap,
    program: &kajit_mir::ir::Program,
) -> HashMap<(kajit_mir::ir::EdgeId, u32), VRegLocation> {
    use std::collections::{HashMap, VecDeque};

    let Some(func) = program.funcs.first() else {
        return HashMap::new();
    };

    let mut inst_lines = HashMap::<kajit_mir::ir::InstId, u32>::new();
    let mut next_line = 1u32;
    for block in func.live_blocks() {
        for &inst_id in &block.insts {
            inst_lines.insert(inst_id, next_line);
            next_line += 1;
        }
        next_line += 1;
    }

    let mut entry_owners =
        HashMap::<kajit_mir::ir::BlockId, HashMap<LocationKey, BTreeSet<u32>>>::new();
    let mut worklist = VecDeque::new();

    let seed = LocationTracker::new(map, program).owners;
    entry_owners.insert(func.entry, seed);
    worklist.push_back(func.entry);

    while let Some(block_id) = worklist.pop_front() {
        let Some(block) = func.blocks.get(block_id.index()) else {
            continue;
        };
        let Some(mut owners) = entry_owners.get(&block_id).cloned() else {
            continue;
        };

        apply_block_transfer(&mut owners, map, func, block, &inst_lines);

        for &edge_id in &block.succs {
            let edge = &func.edges[edge_id.index()];
            let mut edge_owners = owners.clone();
            for arg in &edge.args {
                let aliases = aliases_for_vreg(&owners, arg.source.index() as u32)
                    .unwrap_or_else(|| BTreeSet::from([arg.source.index() as u32]));
                assign_aliases(
                    &mut edge_owners,
                    &map.static_locations,
                    arg.target.index() as u32,
                    aliases,
                );
            }

            let succ_entry = entry_owners.entry(edge.to).or_default();
            if merge_owner_maps(succ_entry, &edge_owners) {
                worklist.push_back(edge.to);
            }
        }
    }

    let mut edge_source_locations = HashMap::new();
    for block in func.live_blocks() {
        let Some(mut owners) = entry_owners.get(&block.id).cloned() else {
            continue;
        };

        apply_block_transfer(&mut owners, map, func, block, &inst_lines);

        for &edge_id in &block.succs {
            let edge = &func.edges[edge_id.index()];
            for arg in &edge.args {
                if let Some(loc) =
                    current_location_from_owners(&owners, map, arg.source.index() as u32)
                {
                    edge_source_locations.insert((edge_id, arg.source.index() as u32), loc);
                }
            }
        }
    }

    edge_source_locations
}

pub fn compute_inst_source_locations(
    map: &LocationMap,
    program: &kajit_mir::ir::Program,
) -> HashMap<(kajit_mir::ir::InstId, u32), VRegLocation> {
    use std::collections::VecDeque;

    let Some(func) = program.funcs.first() else {
        return HashMap::new();
    };

    let mut inst_lines = HashMap::<kajit_mir::ir::InstId, u32>::new();
    let mut next_line = 1u32;
    for block in func.live_blocks() {
        for &inst_id in &block.insts {
            inst_lines.insert(inst_id, next_line);
            next_line += 1;
        }
        next_line += 1;
    }

    let mut entry_owners =
        HashMap::<kajit_mir::ir::BlockId, HashMap<LocationKey, BTreeSet<u32>>>::new();
    let mut worklist = VecDeque::new();

    let seed = LocationTracker::new(map, program).owners;
    entry_owners.insert(func.entry, seed);
    worklist.push_back(func.entry);

    while let Some(block_id) = worklist.pop_front() {
        let Some(block) = func.blocks.get(block_id.index()) else {
            continue;
        };
        let Some(mut owners) = entry_owners.get(&block_id).cloned() else {
            continue;
        };

        apply_block_transfer(&mut owners, map, func, block, &inst_lines);

        for &edge_id in &block.succs {
            let edge = &func.edges[edge_id.index()];
            let mut edge_owners = owners.clone();
            for arg in &edge.args {
                let aliases = aliases_for_vreg(&owners, arg.source.index() as u32)
                    .unwrap_or_else(|| BTreeSet::from([arg.source.index() as u32]));
                assign_aliases(
                    &mut edge_owners,
                    &map.static_locations,
                    arg.target.index() as u32,
                    aliases,
                );
            }

            let succ_entry = entry_owners.entry(edge.to).or_default();
            if merge_owner_maps(succ_entry, &edge_owners) {
                worklist.push_back(edge.to);
            }
        }
    }

    let mut inst_source_locations = HashMap::new();
    for block in func.live_blocks() {
        let Some(mut owners) = entry_owners.get(&block.id).cloned() else {
            continue;
        };

        for &inst_id in &block.insts {
            let Some(&line) = inst_lines.get(&inst_id) else {
                continue;
            };
            let inst = &func.insts[inst_id.index()];
            for operand in &inst.operands {
                if operand.kind != kajit_mir::ir::OperandKind::Use {
                    continue;
                }
                if let Some(loc) =
                    current_location_from_owners(&owners, map, operand.vreg.index() as u32)
                {
                    inst_source_locations.insert((inst_id, operand.vreg.index() as u32), loc);
                }
            }

            if let Some(clobbers) = map.edit_clobbers.get(&line) {
                for loc in clobbers {
                    if let Some(key) = LocationMap::key_for(loc) {
                        owners.remove(&key);
                    }
                }
            }

            if map.call_lines.contains(&line) {
                owners.retain(|key, _| !matches!(key, LocationKey::Register(preg) if *preg <= 18));
            }

            if let Some(def) = inst
                .operands
                .iter()
                .find(|operand| operand.kind == kajit_mir::ir::OperandKind::Def)
                .map(|operand| operand.vreg)
            {
                let aliases = match inst.op {
                    kajit_lir::LinearOp::Copy { src, .. } => {
                        aliases_for_vreg(&owners, src.index() as u32)
                            .unwrap_or_else(|| BTreeSet::from([src.index() as u32]))
                    }
                    _ => BTreeSet::from([def.index() as u32]),
                };
                assign_aliases(
                    &mut owners,
                    &map.static_locations,
                    def.index() as u32,
                    aliases,
                );
            }
        }
    }

    inst_source_locations
}

fn chosen_edge<'a>(
    func: &'a kajit_mir::ir::Function,
    loc_before: &kajit_mir::ProgramLocation,
    loc_after: &kajit_mir::ProgramLocation,
) -> Option<&'a kajit_mir::ir::Edge> {
    let block = func.blocks.get(loc_before.block.index())?;
    let term = func.terms.get(block.term.index())?;
    let edge_id = match term {
        kajit_mir::ir::Terminator::Branch { edge } => Some(*edge),
        kajit_mir::ir::Terminator::BranchIf {
            taken, fallthrough, ..
        }
        | kajit_mir::ir::Terminator::BranchIfZero {
            taken, fallthrough, ..
        } => {
            let taken_edge = func.edges.get(taken.index())?;
            if taken_edge.to == loc_after.block {
                Some(*taken)
            } else {
                Some(*fallthrough)
            }
        }
        kajit_mir::ir::Terminator::JumpTable {
            targets, default, ..
        } => targets
            .iter()
            .copied()
            .find(|edge_id| func.edges[edge_id.index()].to == loc_after.block)
            .or(Some(*default).filter(|edge_id| func.edges[edge_id.index()].to == loc_after.block)),
        _ => None,
    }?;
    func.edges.get(edge_id.index())
}

fn op_def_vreg(
    func: &kajit_mir::ir::Function,
    loc: &kajit_mir::ProgramLocation,
) -> Option<kajit_ir::VReg> {
    use kajit_mir::ir::OperandKind;

    if loc.at_terminator {
        return None;
    }
    let block = func.blocks.get(loc.block.index())?;
    let inst_id = *block.insts.get(loc.next_inst_index)?;
    let inst = func.insts.get(inst_id.index())?;
    inst.operands
        .iter()
        .find(|operand| operand.kind == OperandKind::Def)
        .map(|operand| operand.vreg)
}

#[cfg(test)]
mod tests {
    use super::*;
    use kajit_ir::{LambdaId, VReg};
    use kajit_lir::LinearOp;
    use kajit_mir::ir;

    fn v(index: u32) -> VReg {
        VReg::new(index)
    }

    fn def_inst(id: u32, dst: VReg) -> ir::Inst {
        ir::Inst {
            id: ir::InstId(id),
            op: LinearOp::Const {
                dst,
                value: id as u64,
            },
            operands: vec![ir::Operand {
                vreg: dst,
                kind: ir::OperandKind::Def,
                class: ir::RegClass::Gpr,
                fixed: None,
            }],
            clobbers: ir::Clobbers::default(),
        }
    }

    fn branch_program() -> ir::Program {
        ir::Program {
            funcs: vec![ir::Function {
                id: ir::FunctionId(0),
                lambda_id: LambdaId::new(0),
                entry: ir::BlockId(0),
                data_args: vec![v(2)],
                data_results: Vec::new(),
                output_size: 0,
                blocks: vec![
                    ir::Block {
                        id: ir::BlockId(0),
                        params: Vec::new(),
                        insts: vec![ir::InstId(0)],
                        term: ir::TermId(0),
                        preds: Vec::new(),
                        succs: vec![ir::EdgeId(0)],
                        dead: false,
                    },
                    ir::Block {
                        id: ir::BlockId(1),
                        params: vec![v(1)],
                        insts: Vec::new(),
                        term: ir::TermId(1),
                        preds: vec![ir::EdgeId(0)],
                        succs: Vec::new(),
                        dead: false,
                    },
                ],
                edges: vec![ir::Edge {
                    id: ir::EdgeId(0),
                    from: ir::BlockId(0),
                    to: ir::BlockId(1),
                    args: vec![ir::EdgeArg {
                        target: v(1),
                        source: v(2),
                    }],
                }],
                insts: vec![def_inst(0, v(3))],
                terms: vec![
                    ir::Terminator::Branch {
                        edge: ir::EdgeId(0),
                    },
                    ir::Terminator::Return,
                ],
            }],
            vreg_count: 4,
            slot_count: 0,
            param_slot_count: 0,
            debug: Default::default(),
            hints: Default::default(),
            extra_excluded_regs: vec![],
            data_blobs: vec![],
            stack_allocs: vec![],
            data_arg_layouts: vec![],
        }
    }

    #[test]
    fn location_tracker_reassigns_stack_slot_on_taken_edge() {
        let program = branch_program();
        let func = &program.funcs[0];
        let mut map = LocationMap::default();
        map.static_locations.insert(1, VRegLocation::StackSlot(16));
        map.static_locations.insert(2, VRegLocation::Register(5));
        map.static_locations.insert(3, VRegLocation::StackSlot(16));

        let mut tracker = LocationTracker::new(&map, &program);
        tracker.observe_step(
            &map,
            func,
            1,
            &kajit_mir::ProgramLocation {
                block: ir::BlockId(0),
                next_inst_index: 0,
                at_terminator: false,
            },
            &kajit_mir::ProgramLocation {
                block: ir::BlockId(0),
                next_inst_index: 1,
                at_terminator: true,
            },
        );
        assert_eq!(
            tracker.location_for(&map, 3),
            Some(VRegLocation::StackSlot(16))
        );
        assert_eq!(tracker.location_for(&map, 1), None);

        tracker.observe_step(
            &map,
            func,
            2,
            &kajit_mir::ProgramLocation {
                block: ir::BlockId(0),
                next_inst_index: 0,
                at_terminator: true,
            },
            &kajit_mir::ProgramLocation {
                block: ir::BlockId(1),
                next_inst_index: 0,
                at_terminator: true,
            },
        );
        assert_eq!(
            tracker.location_for(&map, 1),
            Some(VRegLocation::StackSlot(16))
        );
        assert_eq!(tracker.location_for(&map, 3), None);
    }

    #[test]
    fn location_tracker_invalidates_preop_edit_clobber() {
        let program = ir::Program {
            funcs: vec![ir::Function {
                id: ir::FunctionId(0),
                lambda_id: LambdaId::new(0),
                entry: ir::BlockId(0),
                data_args: vec![v(1)],
                data_results: Vec::new(),
                output_size: 0,
                blocks: vec![ir::Block {
                    id: ir::BlockId(0),
                    params: Vec::new(),
                    insts: vec![ir::InstId(0)],
                    term: ir::TermId(0),
                    preds: Vec::new(),
                    succs: Vec::new(),
                    dead: false,
                }],
                edges: Vec::new(),
                insts: vec![ir::Inst {
                    id: ir::InstId(0),
                    op: LinearOp::WriteToSlot {
                        src: v(1),
                        slot: kajit_ir::SlotId::new(0),
                    },
                    operands: vec![ir::Operand {
                        vreg: v(1),
                        kind: ir::OperandKind::Use,
                        class: ir::RegClass::Gpr,
                        fixed: None,
                    }],
                    clobbers: ir::Clobbers::default(),
                }],
                terms: vec![ir::Terminator::Return],
            }],
            vreg_count: 2,
            slot_count: 1,
            param_slot_count: 0,
            debug: Default::default(),
            hints: Default::default(),
            extra_excluded_regs: vec![],
            data_blobs: vec![],
            stack_allocs: vec![],
            data_arg_layouts: vec![],
        };

        let mut map = LocationMap::default();
        map.static_locations.insert(1, VRegLocation::Register(5));
        map.edit_clobbers.insert(1, vec![VRegLocation::Register(5)]);

        let mut tracker = LocationTracker::new(&map, &program);
        assert_eq!(
            tracker.location_for(&map, 1),
            Some(VRegLocation::Register(5))
        );

        tracker.observe_step(
            &map,
            &program.funcs[0],
            1,
            &kajit_mir::ProgramLocation {
                block: ir::BlockId(0),
                next_inst_index: 0,
                at_terminator: false,
            },
            &kajit_mir::ProgramLocation {
                block: ir::BlockId(0),
                next_inst_index: 0,
                at_terminator: true,
            },
        );

        assert_eq!(tracker.location_for(&map, 1), None);
    }

    #[test]
    fn copy_equivalent_live_home_survives_source_clobber() {
        let program = ir::Program {
            funcs: vec![ir::Function {
                id: ir::FunctionId(0),
                lambda_id: LambdaId::new(0),
                entry: ir::BlockId(0),
                data_args: vec![v(2)],
                data_results: Vec::new(),
                output_size: 0,
                blocks: vec![
                    ir::Block {
                        id: ir::BlockId(0),
                        params: Vec::new(),
                        insts: vec![ir::InstId(0), ir::InstId(1)],
                        term: ir::TermId(0),
                        preds: Vec::new(),
                        succs: vec![ir::EdgeId(0)],
                        dead: false,
                    },
                    ir::Block {
                        id: ir::BlockId(1),
                        params: vec![v(1)],
                        insts: Vec::new(),
                        term: ir::TermId(1),
                        preds: vec![ir::EdgeId(0)],
                        succs: Vec::new(),
                        dead: false,
                    },
                ],
                edges: vec![ir::Edge {
                    id: ir::EdgeId(0),
                    from: ir::BlockId(0),
                    to: ir::BlockId(1),
                    args: vec![ir::EdgeArg {
                        target: v(1),
                        source: v(2),
                    }],
                }],
                insts: vec![
                    ir::Inst {
                        id: ir::InstId(0),
                        op: LinearOp::Copy {
                            dst: v(1),
                            src: v(2),
                        },
                        operands: vec![
                            ir::Operand {
                                vreg: v(1),
                                kind: ir::OperandKind::Def,
                                class: ir::RegClass::Gpr,
                                fixed: None,
                            },
                            ir::Operand {
                                vreg: v(2),
                                kind: ir::OperandKind::Use,
                                class: ir::RegClass::Gpr,
                                fixed: None,
                            },
                        ],
                        clobbers: ir::Clobbers::default(),
                    },
                    def_inst(1, v(3)),
                ],
                terms: vec![
                    ir::Terminator::Branch {
                        edge: ir::EdgeId(0),
                    },
                    ir::Terminator::Return,
                ],
            }],
            vreg_count: 4,
            slot_count: 0,
            param_slot_count: 0,
            debug: Default::default(),
            hints: Default::default(),
            extra_excluded_regs: vec![],
            data_blobs: vec![],
            stack_allocs: vec![],
            data_arg_layouts: vec![],
        };

        let mut map = LocationMap::default();
        map.static_locations.insert(1, VRegLocation::Register(0));
        map.static_locations.insert(2, VRegLocation::Register(23));
        map.static_locations.insert(3, VRegLocation::Register(23));
        let func = &program.funcs[0];
        let mut tracker = LocationTracker::new(&map, &program);

        tracker.observe_step(
            &map,
            func,
            1,
            &kajit_mir::ProgramLocation {
                block: ir::BlockId(0),
                next_inst_index: 0,
                at_terminator: false,
            },
            &kajit_mir::ProgramLocation {
                block: ir::BlockId(0),
                next_inst_index: 1,
                at_terminator: false,
            },
        );
        tracker.observe_step(
            &map,
            func,
            2,
            &kajit_mir::ProgramLocation {
                block: ir::BlockId(0),
                next_inst_index: 1,
                at_terminator: false,
            },
            &kajit_mir::ProgramLocation {
                block: ir::BlockId(0),
                next_inst_index: 2,
                at_terminator: true,
            },
        );

        assert_eq!(
            tracker.location_for(&map, 2),
            Some(VRegLocation::Register(0))
        );

        let edge_sources = compute_edge_source_locations(&map, &program);
        assert_eq!(
            edge_sources.get(&(ir::EdgeId(0), 2)),
            Some(&VRegLocation::Register(0))
        );
    }
}

/// An intrinsic call site in the JIT code that needs patching for the standalone harness.
#[derive(Debug, Clone)]
pub struct IntrinsicCallSite {
    /// Offset in the code buffer of the first `movz` instruction (3-instruction sequence).
    pub code_offset: usize,
    /// The baked-in function pointer (from the compiler process).
    pub baked_addr: u64,
    /// Symbol name to resolve (e.g. "_kajit_alloc_persistent").
    pub symbol_name: String,
}

/// Information needed to generate a standalone harness.
pub struct HarnessInput<'a> {
    /// Raw JIT machine code bytes.
    pub code: &'a [u8],
    /// Entry point offset within the code buffer.
    pub entry_offset: usize,
    /// Output buffer size in bytes (sizeof the target type).
    pub output_size: usize,
    /// DWARF sections (if available).
    pub dwarf: Option<crate::jit_dwarf::JitDwarfSections>,
    /// CFG-MIR listing lines (for the source file).
    pub cfg_mir_lines: &'a [String],
    /// Name for the generated function symbol.
    pub function_name: &'a str,
    /// Allocation map (vreg → physical location).
    pub alloc_map: Option<&'a AllocationMap>,
    /// Intrinsic call sites that need address patching.
    pub intrinsic_calls: Vec<IntrinsicCallSite>,
    /// External address relocations (vtable function pointers).
    pub extern_addr_relocs: Vec<crate::ir_backend::ExternAddrRelocInfo>,
}

/// Generate a standalone test harness.
///
/// Returns the path to the generated executable.
pub fn generate_harness(
    input: &HarnessInput,
    output_dir: &Path,
    base_name: &str,
) -> Result<std::path::PathBuf, HarnessError> {
    std::fs::create_dir_all(output_dir).map_err(|e| HarnessError::Io("create output dir", e))?;

    // Write the CFG-MIR listing file (DWARF source)
    let listing_path = output_dir.join(format!("{base_name}.cfg-mir"));
    let listing_text = input.cfg_mir_lines.join("\n");
    std::fs::write(&listing_path, &listing_text)
        .map_err(|e| HarnessError::Io("write listing", e))?;

    // Build the object file with JIT code
    let obj_path = output_dir.join(format!("{base_name}.o"));
    build_object_file(input, &obj_path)?;

    // Write the C harness
    let c_path = output_dir.join(format!("{base_name}_main.c"));
    write_c_harness(input, &c_path)?;

    // Write allocation map (for lockstep debugger)
    if let Some(alloc_map) = input.alloc_map {
        let map_path = output_dir.join(format!("{base_name}.alloc.json"));
        alloc_map
            .write_json(&map_path)
            .map_err(|e| HarnessError::Io("write alloc map", e))?;
    }

    // Link: cc -o harness harness_main.c jit.o -lSystem
    let exe_path = output_dir.join(base_name);
    link_harness(&c_path, &obj_path, &exe_path)?;

    maybe_build_debug_bundle(&exe_path, input);

    eprintln!("[harness] generated: {}", exe_path.display());
    eprintln!("[harness] listing:   {}", listing_path.display());
    eprintln!("[harness] usage:     {} <input-hex>", exe_path.display());
    #[cfg(target_os = "macos")]
    eprintln!(
        "[harness] debug:     lldb {} -- <input-hex>",
        exe_path.display()
    );
    #[cfg(target_os = "linux")]
    eprintln!(
        "[harness] debug:     gdb --args {} <input-hex>",
        exe_path.display()
    );

    Ok(exe_path)
}

pub fn build_object_file(input: &HarnessInput, path: &Path) -> Result<(), HarnessError> {
    use object::write::{Object, Relocation, Symbol, SymbolSection};
    use object::{
        Architecture, BinaryFormat, Endianness, RelocationFlags, SectionKind, SymbolFlags,
        SymbolKind, SymbolScope,
    };

    let mut obj = Object::new(
        BinaryFormat::native_object(),
        Architecture::Aarch64,
        Endianness::Little,
    );

    #[cfg(target_os = "macos")]
    {
        // Set macOS platform version (prevents "no platform load command" warning)
        let mut build_ver = object::write::MachOBuildVersion::default();
        build_ver.platform = object::macho::PLATFORM_MACOS;
        build_ver.minos = 14 << 16; // macOS 14.0
        build_ver.sdk = 14 << 16;
        obj.set_macho_build_version(build_ver);
    }

    // Patch intrinsic call sites: replace movz/movk/movk with adrp/add/nop
    // so the linker can resolve intrinsic symbols.
    let mut code = input.code.to_vec();
    let mut intrinsic_relocs: Vec<(usize, object::write::SymbolId)> = Vec::new();

    for site in &input.intrinsic_calls {
        // Add an undefined symbol for this intrinsic
        let sym_id = obj.add_symbol(Symbol {
            name: site.symbol_name.as_bytes().to_vec(),
            value: 0,
            size: 0,
            kind: SymbolKind::Text,
            scope: SymbolScope::Dynamic,
            weak: false,
            section: SymbolSection::Undefined,
            flags: SymbolFlags::None,
        });

        // Rewrite movz/movk/movk (12 bytes) to adrp/add/nop (12 bytes).
        let off = site.code_offset;
        let adrp = 0x90000010u32; // adrp x16, #0 (imm filled by linker)
        let add = 0x91000210u32; // add x16, x16, #0 (imm filled by linker)
        let nop = 0xD503201Fu32; // nop
        code[off..off + 4].copy_from_slice(&adrp.to_le_bytes());
        code[off + 4..off + 8].copy_from_slice(&add.to_le_bytes());
        code[off + 8..off + 12].copy_from_slice(&nop.to_le_bytes());
        intrinsic_relocs.push((off, sym_id));
    }

    // Patch extern addr sites: replace movz/movk/movk/movk (16 bytes) with adrp/add/nop/nop
    // so the linker can resolve vtable function pointer symbols.
    let mut extern_addr_reloc_entries: Vec<(usize, object::write::SymbolId)> = Vec::new();

    for reloc in &input.extern_addr_relocs {
        let sym_id = obj.add_symbol(Symbol {
            name: reloc.symbol.as_str().as_bytes().to_vec(),
            value: 0,
            size: 0,
            kind: SymbolKind::Data,
            scope: SymbolScope::Dynamic,
            weak: false,
            section: SymbolSection::Undefined,
            flags: SymbolFlags::None,
        });

        // Rewrite movz/movk/movk/movk (16 bytes) to adrp/add/nop/nop (16 bytes).
        let off = reloc.code_offset;
        let adrp = 0x90000010u32; // adrp x16, #0 (imm filled by linker)
        let add = 0x91000210u32; // add x16, x16, #0 (imm filled by linker)
        let nop = 0xD503201Fu32; // nop
        code[off..off + 4].copy_from_slice(&adrp.to_le_bytes());
        code[off + 4..off + 8].copy_from_slice(&add.to_le_bytes());
        code[off + 8..off + 12].copy_from_slice(&nop.to_le_bytes());
        code[off + 12..off + 16].copy_from_slice(&nop.to_le_bytes());
        extern_addr_reloc_entries.push((off, sym_id));
    }

    // Add .text section with (possibly patched) JIT code
    let text_section = obj.section_id(object::write::StandardSection::Text);
    obj.append_section_data(text_section, &code, 16);

    // Add relocations for intrinsic call sites
    for &(off, sym_id) in &intrinsic_relocs {
        #[cfg(target_os = "macos")]
        {
            obj.add_relocation(
                text_section,
                Relocation {
                    offset: off as u64,
                    symbol: sym_id,
                    flags: RelocationFlags::MachO {
                        r_type: object::macho::ARM64_RELOC_PAGE21,
                        r_pcrel: true,
                        r_length: 2,
                    },
                    addend: 0,
                },
            )
            .expect("adrp relocation");

            obj.add_relocation(
                text_section,
                Relocation {
                    offset: (off + 4) as u64,
                    symbol: sym_id,
                    flags: RelocationFlags::MachO {
                        r_type: object::macho::ARM64_RELOC_PAGEOFF12,
                        r_pcrel: false,
                        r_length: 2,
                    },
                    addend: 0,
                },
            )
            .expect("add relocation");
        }

        #[cfg(target_os = "linux")]
        {
            obj.add_relocation(
                text_section,
                Relocation {
                    offset: off as u64,
                    symbol: sym_id,
                    flags: RelocationFlags::Elf {
                        r_type: object::elf::R_AARCH64_ADR_PREL_PG_HI21,
                    },
                    addend: 0,
                },
            )
            .expect("adrp relocation");

            obj.add_relocation(
                text_section,
                Relocation {
                    offset: (off + 4) as u64,
                    symbol: sym_id,
                    flags: RelocationFlags::Elf {
                        r_type: object::elf::R_AARCH64_ADD_ABS_LO12_NC,
                    },
                    addend: 0,
                },
            )
            .expect("add relocation");
        }
    }

    // Add relocations for extern addr (vtable pointer) sites — same pattern as intrinsics
    for &(off, sym_id) in &extern_addr_reloc_entries {
        #[cfg(target_os = "macos")]
        {
            obj.add_relocation(
                text_section,
                Relocation {
                    offset: off as u64,
                    symbol: sym_id,
                    flags: RelocationFlags::MachO {
                        r_type: object::macho::ARM64_RELOC_PAGE21,
                        r_pcrel: true,
                        r_length: 2,
                    },
                    addend: 0,
                },
            )
            .expect("extern addr adrp relocation");

            obj.add_relocation(
                text_section,
                Relocation {
                    offset: (off + 4) as u64,
                    symbol: sym_id,
                    flags: RelocationFlags::MachO {
                        r_type: object::macho::ARM64_RELOC_PAGEOFF12,
                        r_pcrel: false,
                        r_length: 2,
                    },
                    addend: 0,
                },
            )
            .expect("extern addr add relocation");
        }

        #[cfg(target_os = "linux")]
        {
            obj.add_relocation(
                text_section,
                Relocation {
                    offset: off as u64,
                    symbol: sym_id,
                    flags: RelocationFlags::Elf {
                        r_type: object::elf::R_AARCH64_ADR_PREL_PG_HI21,
                    },
                    addend: 0,
                },
            )
            .expect("extern addr adrp relocation");

            obj.add_relocation(
                text_section,
                Relocation {
                    offset: (off + 4) as u64,
                    symbol: sym_id,
                    flags: RelocationFlags::Elf {
                        r_type: object::elf::R_AARCH64_ADD_ABS_LO12_NC,
                    },
                    addend: 0,
                },
            )
            .expect("extern addr add relocation");
        }
    }

    // Add the entry point symbol (global, so the C harness can call it)
    let symbol_name = input.function_name.to_string();
    let text_symbol = obj.add_symbol(Symbol {
        name: symbol_name.into_bytes(),
        value: input.entry_offset as u64,
        size: (input.code.len() - input.entry_offset) as u64,
        kind: SymbolKind::Text,
        scope: SymbolScope::Dynamic,
        weak: false,
        section: SymbolSection::Section(text_section),
        flags: SymbolFlags::None,
    });

    // Add DWARF sections with relocations
    if let Some(dwarf) = &input.dwarf {
        let mut debug_info_section_id = None;
        let mut debug_line_section_id = None;

        if !dwarf.debug_line.is_empty() {
            let sid = obj.add_section(
                dwarf_segment_name(),
                dwarf_debug_section_name("debug_line"),
                SectionKind::Debug,
            );
            obj.append_section_data(sid, &dwarf.debug_line, 1);
            debug_line_section_id = Some(sid);
        }
        if !dwarf.debug_info.is_empty() {
            let sid = obj.add_section(
                dwarf_segment_name(),
                dwarf_debug_section_name("debug_info"),
                SectionKind::Debug,
            );
            obj.append_section_data(sid, &dwarf.debug_info, 1);
            debug_info_section_id = Some(sid);
        }
        if !dwarf.debug_abbrev.is_empty() {
            let sid = obj.add_section(
                dwarf_segment_name(),
                dwarf_debug_section_name("debug_abbrev"),
                SectionKind::Debug,
            );
            obj.append_section_data(sid, &dwarf.debug_abbrev, 1);
        }
        let mut debug_aranges_section_id = None;
        if !dwarf.debug_aranges.is_empty() {
            let sid = obj.add_section(
                dwarf_segment_name(),
                dwarf_debug_section_name("debug_aranges"),
                SectionKind::Debug,
            );
            obj.append_section_data(sid, &dwarf.debug_aranges, 1);
            debug_aranges_section_id = Some(sid);
        }

        // Add relocations so the linker/dsymutil fixes up DWARF addresses
        for (section, reloc) in &dwarf.relocations {
            let target_section = match section {
                crate::jit_dwarf::DwarfSection::DebugInfo => debug_info_section_id,
                crate::jit_dwarf::DwarfSection::DebugLine => debug_line_section_id,
                crate::jit_dwarf::DwarfSection::DebugAranges => debug_aranges_section_id,
            };
            if let Some(sid) = target_section {
                obj.add_relocation(
                    sid,
                    dwarf_text_relocation(text_symbol, reloc, input.entry_offset),
                )
                .map_err(HarnessError::ObjectWrite)?;
            }
        }
    }

    let data = obj.write().map_err(HarnessError::ObjectWrite)?;
    std::fs::write(path, data).map_err(|e| HarnessError::Io("write object", e))?;

    Ok(())
}

fn dwarf_segment_name() -> Vec<u8> {
    #[cfg(target_os = "macos")]
    {
        b"__DWARF".to_vec()
    }
    #[cfg(target_os = "linux")]
    {
        Vec::new()
    }
}

fn dwarf_debug_section_name(name: &str) -> Vec<u8> {
    #[cfg(target_os = "macos")]
    {
        format!("__{name}").into_bytes()
    }
    #[cfg(target_os = "linux")]
    {
        format!(".{name}").into_bytes()
    }
}

fn dwarf_text_relocation(
    text_symbol: object::write::SymbolId,
    reloc: &crate::jit_dwarf::DwarfRelocation,
    entry_offset: usize,
) -> object::write::Relocation {
    #[cfg(target_os = "macos")]
    {
        object::write::Relocation {
            offset: reloc.offset as u64,
            symbol: text_symbol,
            addend: reloc.addend + entry_offset as i64,
            flags: object::RelocationFlags::MachO {
                r_type: object::macho::ARM64_RELOC_UNSIGNED,
                r_pcrel: false,
                r_length: 3,
            },
        }
    }
    #[cfg(target_os = "linux")]
    {
        object::write::Relocation {
            offset: reloc.offset as u64,
            symbol: text_symbol,
            addend: reloc.addend + entry_offset as i64,
            flags: object::RelocationFlags::Generic {
                kind: object::RelocationKind::Absolute,
                encoding: object::RelocationEncoding::Generic,
                size: 64,
            },
        }
    }
}

fn write_c_harness(input: &HarnessInput, path: &Path) -> Result<(), HarnessError> {
    let output_size = input.output_size;
    let func_name = input.function_name;
    let call_decl = format!(
        r#"
typedef struct {{
    const uint8_t *ptr;
    size_t len;
}} RuntimeSliceU8;

typedef struct {{
    RuntimeSliceU8 bytes;
    uint64_t pos;
}} RuntimeCursorArg;

extern void {func_name}(RuntimeCursorArg *cursor, uint8_t *output, DeserContext *ctx);
"#
    );
    let call_site = format!(
        r#"
    RuntimeCursorArg cursor;
    memset(&cursor, 0, sizeof(cursor));
    cursor.bytes.ptr = input;
    cursor.bytes.len = input_len;
    {func_name}(&cursor, output, &ctx);
    ctx.cursor = cursor.bytes.ptr + cursor.pos;
"#
    );

    let c_code = format!(
        r#"// Auto-generated test harness for kajit JIT code.
// Usage: ./{func_name} <input-hex>
// Example: ./{func_name} 8001

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

// DeserContext — must match kajit::context::DeserContext layout
typedef struct {{
    const uint8_t *cursor;   // input_ptr
    const uint8_t *end;      // input_end
    struct {{                 // error (ErrorSlot)
        uint32_t code;
        uint32_t offset;
    }} error;
    uint8_t *key_scratch_ptr;
    size_t key_scratch_cap;
    uint8_t trusted_utf8;    // bool
}} DeserContext;

// The JIT-compiled decoder function (linked from the .o file)
{call_decl}

static int hex_digit(char c) {{
    if (c >= '0' && c <= '9') return c - '0';
    if (c >= 'a' && c <= 'f') return c - 'a' + 10;
    if (c >= 'A' && c <= 'F') return c - 'A' + 10;
    return -1;
}}

static size_t parse_hex(const char *s, uint8_t *buf, size_t max) {{
    size_t len = 0;
    while (*s && len < max) {{
        // Skip spaces and other non-hex characters
        while (*s && hex_digit(*s) < 0) s++;
        if (!*s || !*(s+1)) break;
        int hi = hex_digit(*s);
        int lo = hex_digit(*(s+1));
        if (hi < 0 || lo < 0) break;
        buf[len++] = (uint8_t)((hi << 4) | lo);
        s += 2;
    }}
    return len;
}}

int main(int argc, char **argv) {{
    if (argc < 2) {{
        fprintf(stderr, "usage: %s <input-hex>\n", argv[0]);
        return 1;
    }}

    // Parse hex input
    uint8_t input[4096];
    size_t input_len = parse_hex(argv[1], input, sizeof(input));

    // Allocate output buffer
    uint8_t output[{output_size}];
    memset(output, 0, sizeof(output));

    // Set up context
    DeserContext ctx;
    memset(&ctx, 0, sizeof(ctx));
    ctx.cursor = input;
    ctx.end = input + input_len;

    // Call the JIT decoder
    {call_site}

    // Check for errors
    if (ctx.error.code != 0) {{
        fprintf(stderr, "error: code=%u offset=%u\n", ctx.error.code, ctx.error.offset);
        return 1;
    }}

    // Print output as hex
    for (size_t i = 0; i < {output_size}; i++) {{
        printf("%02x", output[i]);
    }}
    printf("\n");

    return 0;
}}
"#,
        call_decl = call_decl,
        call_site = call_site,
    );

    std::fs::write(path, c_code).map_err(|e| HarnessError::Io("write C harness", e))?;
    Ok(())
}

/// Build a dSYM bundle by hand: read UUID from exe, patch DWARF addresses,
/// write a Mach-O with DWARF into the dSYM directory structure.
fn build_dsym(
    exe_path: &Path,
    dwarf: &crate::jit_dwarf::JitDwarfSections,
    function_name: &str,
    _entry_offset: usize,
) -> Result<(), HarnessError> {
    use object::read::{Object, ObjectSegment, ObjectSymbol};

    let exe_data = std::fs::read(exe_path).map_err(|e| HarnessError::Io("read exe for dSYM", e))?;
    let exe_obj = object::read::File::parse(&*exe_data)
        .map_err(|e| HarnessError::Link(format!("parse exe: {e}")))?;

    // Get UUID
    let uuid = exe_obj.mach_uuid().ok().flatten().unwrap_or([0u8; 16]);

    // Get symbol address
    let mangled = format!("_{function_name}");
    let symbol_addr = exe_obj
        .symbols()
        .find(|s| s.name() == Ok(&mangled))
        .map(|s| s.address())
        .ok_or_else(|| HarnessError::Link(format!("symbol {mangled} not found for dSYM")))?;

    // Get __TEXT segment address range (needed for LLDB address resolution)
    let (text_vmaddr, text_vmsize) = exe_obj
        .segments()
        .find(|s| s.name() == Ok(Some("__TEXT")))
        .map(|s| (s.address(), s.size()))
        .unwrap_or((symbol_addr, 0x10000));

    // Get symbol size
    let code_size = exe_obj
        .symbols()
        .find(|s| s.name() == Ok(&mangled))
        .map(|s| s.size())
        .unwrap_or(0x1000);

    drop(exe_obj);

    eprintln!(
        "[harness] building dSYM: {} @ 0x{:x}, uuid={}",
        mangled,
        symbol_addr,
        uuid.iter().map(|b| format!("{b:02X}")).collect::<String>()
    );

    // Patch DWARF: copy sections and fix addresses at relocation offsets
    let addr_bytes = symbol_addr.to_le_bytes();

    let mut debug_info = dwarf.debug_info.clone();
    let mut debug_line = dwarf.debug_line.clone();
    let mut debug_aranges = dwarf.debug_aranges.clone();

    for (section, reloc) in &dwarf.relocations {
        let data = match section {
            crate::jit_dwarf::DwarfSection::DebugInfo => &mut debug_info,
            crate::jit_dwarf::DwarfSection::DebugLine => &mut debug_line,
            crate::jit_dwarf::DwarfSection::DebugAranges => &mut debug_aranges,
        };
        let offset = reloc.offset as usize;
        if offset + 8 <= data.len() {
            data[offset..offset + 8].copy_from_slice(&addr_bytes);
        }
    }

    // Build the dSYM Mach-O by hand (need LC_UUID which the object crate can't emit)
    let dsym_data = build_dsym_macho(
        &uuid,
        &debug_info,
        &debug_line,
        &dwarf.debug_abbrev,
        &debug_aranges,
        text_vmaddr,
        text_vmsize,
        symbol_addr,
        code_size,
        &mangled,
    );

    // Write dSYM bundle
    let dsym_dir = exe_path.with_extension("dSYM");
    let dwarf_dir = dsym_dir.join("Contents/Resources/DWARF");
    std::fs::create_dir_all(&dwarf_dir).map_err(|e| HarnessError::Io("create dSYM dir", e))?;

    let dsym_file = dwarf_dir.join(exe_path.file_name().unwrap());
    std::fs::write(&dsym_file, &dsym_data).map_err(|e| HarnessError::Io("write dSYM Mach-O", e))?;

    // Write Info.plist
    let plist = format!(
        r#"<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>CFBundleDevelopmentRegion</key>
    <string>English</string>
    <key>CFBundleIdentifier</key>
    <string>com.kajit.harness.{}</string>
    <key>CFBundleInfoDictionaryVersion</key>
    <string>6.0</string>
    <key>CFBundlePackageType</key>
    <string>dSYM</string>
    <key>CFBundleVersion</key>
    <string>1</string>
</dict>
</plist>"#,
        exe_path.file_stem().unwrap().to_str().unwrap()
    );
    let plist_path = dsym_dir.join("Contents/Info.plist");
    std::fs::write(&plist_path, plist).map_err(|e| HarnessError::Io("write Info.plist", e))?;

    eprintln!("[harness] created dSYM: {}", dsym_dir.display());
    Ok(())
}

/// Build a dSYM Mach-O with: LC_UUID + LC_SYMTAB + LC_SEGMENT_64(__TEXT) +
/// LC_SEGMENT_64(__LINKEDIT) + LC_SEGMENT_64(__DWARF).
fn build_dsym_macho(
    uuid: &[u8; 16],
    debug_info: &[u8],
    debug_line: &[u8],
    debug_abbrev: &[u8],
    debug_aranges: &[u8],
    text_vmaddr: u64,
    text_vmsize: u64,
    symbol_addr: u64,
    _symbol_size: u64,
    symbol_name: &str, // mangled, e.g. "_kajit_decode"
) -> Vec<u8> {
    // Count DWARF sections
    let mut dwarf_nsects = 0u32;
    if !debug_info.is_empty() {
        dwarf_nsects += 1;
    }
    if !debug_line.is_empty() {
        dwarf_nsects += 1;
    }
    if !debug_abbrev.is_empty() {
        dwarf_nsects += 1;
    }
    if !debug_aranges.is_empty() {
        dwarf_nsects += 1;
    }

    // Sizes
    let header_size = 32u32; // mach_header_64
    let uuid_cmd_size = 24u32; // LC_UUID
    let segment_cmd_size = 72u32; // LC_SEGMENT_64 (without sections)
    let section_size = 80u32; // section_64 per section
    let text_nsects = 1u32; // __text section stub
    let text_segment_size = segment_cmd_size + text_nsects * section_size;
    let linkedit_segment_size = segment_cmd_size; // no sections
    let symtab_cmd_size = 24u32; // LC_SYMTAB
    let dwarf_segment_size = segment_cmd_size + dwarf_nsects * section_size;
    let ncmds = 5u32; // LC_UUID + LC_SYMTAB + __TEXT + __LINKEDIT + __DWARF
    let load_cmds_size = uuid_cmd_size
        + symtab_cmd_size
        + text_segment_size
        + linkedit_segment_size
        + dwarf_segment_size;
    let header_and_cmds = header_size + load_cmds_size;

    // Build symtab + strtab for __LINKEDIT
    // strtab: \0 + symbol_name + \0
    let mut strtab = vec![0u8]; // index 0 = empty string
    let sym_name_offset = strtab.len() as u32;
    strtab.extend_from_slice(symbol_name.as_bytes());
    strtab.push(0);

    // nlist_64: 16 bytes per symbol
    // { n_strx: u32, n_type: u8, n_sect: u8, n_desc: u16, n_value: u64 }
    let mut symtab_data = Vec::new();
    symtab_data.extend_from_slice(&sym_name_offset.to_le_bytes()); // n_strx
    symtab_data.push(0x0F); // n_type: N_SECT | N_EXT
    symtab_data.push(1); // n_sect: 1 (__text)
    symtab_data.extend_from_slice(&0u16.to_le_bytes()); // n_desc
    symtab_data.extend_from_slice(&symbol_addr.to_le_bytes()); // n_value

    // Align data start to page boundary (4096) for __LINKEDIT
    let data_start = header_and_cmds.div_ceil(4096) * 4096;
    let _padding = data_start - header_and_cmds;

    // Layout: [headers + padding] [LINKEDIT: symtab + strtab] [DWARF sections]
    let linkedit_fileoff = data_start;
    let linkedit_size = (symtab_data.len() + strtab.len()) as u32;
    let linkedit_size_aligned = (linkedit_size as u32).div_ceil(4096) * 4096;
    let linkedit_vmaddr = text_vmaddr + text_vmsize;

    let dwarf_fileoff = linkedit_fileoff + linkedit_size_aligned;
    let mut section_offsets = Vec::new();
    let mut offset = dwarf_fileoff;
    for data in [debug_info, debug_line, debug_abbrev, debug_aranges] {
        if !data.is_empty() {
            section_offsets.push((offset, data.len() as u32));
            offset += data.len() as u32;
        }
    }
    let dwarf_total_data = offset - dwarf_fileoff;
    let dwarf_vmaddr = linkedit_vmaddr + linkedit_size_aligned as u64;

    let mut out = Vec::with_capacity(offset as usize);

    // --- Mach-O header (mach_header_64) ---
    out.extend_from_slice(&0xFEEDFACFu32.to_le_bytes()); // magic (MH_MAGIC_64)
    out.extend_from_slice(&(12u32 | 0x01000000).to_le_bytes()); // cputype: CPU_TYPE_ARM64
    out.extend_from_slice(&0u32.to_le_bytes()); // cpusubtype: ALL
    out.extend_from_slice(&0x0Au32.to_le_bytes()); // filetype: MH_DSYM
    out.extend_from_slice(&ncmds.to_le_bytes()); // ncmds
    out.extend_from_slice(&load_cmds_size.to_le_bytes()); // sizeofcmds
    out.extend_from_slice(&0u32.to_le_bytes()); // flags
    out.extend_from_slice(&0u32.to_le_bytes()); // reserved

    // --- LC_UUID ---
    out.extend_from_slice(&0x1Bu32.to_le_bytes()); // cmd: LC_UUID
    out.extend_from_slice(&uuid_cmd_size.to_le_bytes()); // cmdsize
    out.extend_from_slice(uuid); // 16 bytes UUID

    // --- LC_SYMTAB ---
    out.extend_from_slice(&0x02u32.to_le_bytes()); // cmd: LC_SYMTAB
    out.extend_from_slice(&symtab_cmd_size.to_le_bytes()); // cmdsize
    out.extend_from_slice(&linkedit_fileoff.to_le_bytes()); // symoff
    out.extend_from_slice(&1u32.to_le_bytes()); // nsyms
    out.extend_from_slice(&(linkedit_fileoff + symtab_data.len() as u32).to_le_bytes()); // stroff
    out.extend_from_slice(&(strtab.len() as u32).to_le_bytes()); // strsize

    // --- LC_SEGMENT_64 (__TEXT) — stub for address resolution ---
    out.extend_from_slice(&0x19u32.to_le_bytes()); // cmd: LC_SEGMENT_64
    out.extend_from_slice(&text_segment_size.to_le_bytes()); // cmdsize
    let mut text_segname = [0u8; 16];
    text_segname[..6].copy_from_slice(b"__TEXT");
    out.extend_from_slice(&text_segname); // segname
    out.extend_from_slice(&text_vmaddr.to_le_bytes()); // vmaddr
    out.extend_from_slice(&text_vmsize.to_le_bytes()); // vmsize
    out.extend_from_slice(&0u64.to_le_bytes()); // fileoff (no file data)
    out.extend_from_slice(&0u64.to_le_bytes()); // filesize (no file data)
    out.extend_from_slice(&5i32.to_le_bytes()); // maxprot: VM_PROT_READ | VM_PROT_EXECUTE
    out.extend_from_slice(&5i32.to_le_bytes()); // initprot: VM_PROT_READ | VM_PROT_EXECUTE
    out.extend_from_slice(&text_nsects.to_le_bytes()); // nsects
    out.extend_from_slice(&0u32.to_le_bytes()); // flags

    // __text section stub (no file data, just address range)
    let mut text_sectname = [0u8; 16];
    text_sectname[..6].copy_from_slice(b"__text");
    out.extend_from_slice(&text_sectname); // sectname
    out.extend_from_slice(&text_segname); // segname
    out.extend_from_slice(&text_vmaddr.to_le_bytes()); // addr
    out.extend_from_slice(&text_vmsize.to_le_bytes()); // size
    out.extend_from_slice(&0u32.to_le_bytes()); // offset (no file data)
    out.extend_from_slice(&0u32.to_le_bytes()); // align
    out.extend_from_slice(&0u32.to_le_bytes()); // reloff
    out.extend_from_slice(&0u32.to_le_bytes()); // nreloc
    out.extend_from_slice(&0x80000400u32.to_le_bytes()); // flags: S_REGULAR | S_ATTR_PURE_INSTRUCTIONS | S_ATTR_SOME_INSTRUCTIONS
    out.extend_from_slice(&0u32.to_le_bytes()); // reserved1
    out.extend_from_slice(&0u32.to_le_bytes()); // reserved2
    out.extend_from_slice(&0u32.to_le_bytes()); // reserved3

    // --- LC_SEGMENT_64 (__LINKEDIT) ---
    out.extend_from_slice(&0x19u32.to_le_bytes()); // cmd: LC_SEGMENT_64
    out.extend_from_slice(&linkedit_segment_size.to_le_bytes());
    let mut linkedit_segname = [0u8; 16];
    linkedit_segname[..10].copy_from_slice(b"__LINKEDIT");
    out.extend_from_slice(&linkedit_segname);
    out.extend_from_slice(&linkedit_vmaddr.to_le_bytes()); // vmaddr
    out.extend_from_slice(&(linkedit_size_aligned as u64).to_le_bytes()); // vmsize
    out.extend_from_slice(&(linkedit_fileoff as u64).to_le_bytes()); // fileoff
    out.extend_from_slice(&(linkedit_size as u64).to_le_bytes()); // filesize
    out.extend_from_slice(&1i32.to_le_bytes()); // maxprot: VM_PROT_READ
    out.extend_from_slice(&1i32.to_le_bytes()); // initprot: VM_PROT_READ
    out.extend_from_slice(&0u32.to_le_bytes()); // nsects
    out.extend_from_slice(&0u32.to_le_bytes()); // flags

    // --- LC_SEGMENT_64 (__DWARF) ---
    out.extend_from_slice(&0x19u32.to_le_bytes()); // cmd: LC_SEGMENT_64
    out.extend_from_slice(&dwarf_segment_size.to_le_bytes());
    let mut segname = [0u8; 16];
    segname[..7].copy_from_slice(b"__DWARF");
    out.extend_from_slice(&segname);
    out.extend_from_slice(&dwarf_vmaddr.to_le_bytes()); // vmaddr
    out.extend_from_slice(&(dwarf_total_data as u64).to_le_bytes()); // vmsize
    out.extend_from_slice(&(dwarf_fileoff as u64).to_le_bytes()); // fileoff
    out.extend_from_slice(&(dwarf_total_data as u64).to_le_bytes()); // filesize
    out.extend_from_slice(&0i32.to_le_bytes()); // maxprot
    out.extend_from_slice(&0i32.to_le_bytes()); // initprot
    out.extend_from_slice(&dwarf_nsects.to_le_bytes()); // nsects
    out.extend_from_slice(&0u32.to_le_bytes()); // flags

    // --- section_64 entries ---
    let section_names: Vec<&[u8]> = {
        let mut names = Vec::new();
        if !debug_info.is_empty() {
            names.push(b"__debug_info" as &[u8]);
        }
        if !debug_line.is_empty() {
            names.push(b"__debug_line" as &[u8]);
        }
        if !debug_abbrev.is_empty() {
            names.push(b"__debug_abbrev" as &[u8]);
        }
        if !debug_aranges.is_empty() {
            names.push(b"__debug_aranges" as &[u8]);
        }
        names
    };

    for (name, &(off, size)) in section_names.iter().zip(section_offsets.iter()) {
        let mut sectname = [0u8; 16];
        let len = name.len().min(16);
        sectname[..len].copy_from_slice(&name[..len]);
        out.extend_from_slice(&sectname);
        out.extend_from_slice(&segname); // segname: __DWARF
        let sect_vmaddr = dwarf_vmaddr + (off - dwarf_fileoff) as u64;
        out.extend_from_slice(&sect_vmaddr.to_le_bytes()); // addr
        out.extend_from_slice(&(size as u64).to_le_bytes()); // size
        out.extend_from_slice(&off.to_le_bytes()); // offset
        out.extend_from_slice(&0u32.to_le_bytes()); // align
        out.extend_from_slice(&0u32.to_le_bytes()); // reloff
        out.extend_from_slice(&0u32.to_le_bytes()); // nreloc
        out.extend_from_slice(&0x02000000u32.to_le_bytes()); // flags: S_REGULAR | S_ATTR_DEBUG
        out.extend_from_slice(&0u32.to_le_bytes()); // reserved1
        out.extend_from_slice(&0u32.to_le_bytes()); // reserved2
        out.extend_from_slice(&0u32.to_le_bytes()); // reserved3 (padding for 64-bit)
    }

    // Padding to page boundary
    out.resize(data_start as usize, 0);

    // __LINKEDIT data: symtab + strtab
    out.extend_from_slice(&symtab_data);
    out.extend_from_slice(&strtab);
    // Pad __LINKEDIT to page boundary
    out.resize((linkedit_fileoff + linkedit_size_aligned) as usize, 0);

    // __DWARF section data
    if !debug_info.is_empty() {
        out.extend_from_slice(debug_info);
    }
    if !debug_line.is_empty() {
        out.extend_from_slice(debug_line);
    }
    if !debug_abbrev.is_empty() {
        out.extend_from_slice(debug_abbrev);
    }
    if !debug_aranges.is_empty() {
        out.extend_from_slice(debug_aranges);
    }

    out
}

/// Patch the LC_UUID in a Mach-O binary.
/// LC_UUID has cmd=0x1B, cmdsize=24, followed by 16 bytes of UUID.
fn patch_macho_uuid(data: &mut [u8], uuid: &[u8; 16]) {
    const LC_UUID: u32 = 0x1b;
    // Walk load commands to find LC_UUID
    // Mach-O 64 header: 32 bytes, then load commands
    if data.len() < 32 {
        return;
    }
    let ncmds = u32::from_le_bytes(data[16..20].try_into().unwrap()) as usize;
    let mut offset = 32; // past mach_header_64

    for _ in 0..ncmds {
        if offset + 8 > data.len() {
            break;
        }
        let cmd = u32::from_le_bytes(data[offset..offset + 4].try_into().unwrap());
        let cmdsize = u32::from_le_bytes(data[offset + 4..offset + 8].try_into().unwrap()) as usize;

        if cmd == LC_UUID && cmdsize >= 24 && offset + 24 <= data.len() {
            data[offset + 8..offset + 24].copy_from_slice(uuid);
            return;
        }

        offset += cmdsize;
    }
}

/// Find the address of a symbol in a linked binary.
fn find_symbol_address(exe_path: &Path, function_name: &str) -> Result<u64, HarnessError> {
    use object::read::{Object, ObjectSymbol};

    let binary = std::fs::read(exe_path).map_err(|e| HarnessError::Io("read binary", e))?;
    let obj = object::read::File::parse(&*binary)
        .map_err(|e| HarnessError::Link(format!("parse linked binary: {e}")))?;

    let mangled = format!("_{function_name}");
    obj.symbols()
        .find(|s| s.name() == Ok(&mangled))
        .map(|s| s.address())
        .ok_or_else(|| HarnessError::Link(format!("symbol {mangled} not found")))
}

/// Patch DWARF addresses in a .o file so dsymutil picks them up.
fn patch_object_dwarf(
    obj_path: &Path,
    dwarf: &crate::jit_dwarf::JitDwarfSections,
    symbol_addr: u64,
) -> Result<(), HarnessError> {
    use object::read::{Object, ObjectSection};

    let mut data = std::fs::read(obj_path).map_err(|e| HarnessError::Io("read .o for patch", e))?;

    let obj = object::read::File::parse(&*data)
        .map_err(|e| HarnessError::Link(format!("parse .o: {e}")))?;

    let debug_info_offset = obj
        .section_by_name("__debug_info")
        .and_then(|s| s.file_range())
        .map(|(off, _)| off);
    let debug_line_offset = obj
        .section_by_name("__debug_line")
        .and_then(|s| s.file_range())
        .map(|(off, _)| off);

    drop(obj); // release borrow on data

    eprintln!("[harness] patching .o DWARF: addr=0x{:x}", symbol_addr);
    let addr_bytes = symbol_addr.to_le_bytes();

    for (section, reloc) in &dwarf.relocations {
        let base = match section {
            crate::jit_dwarf::DwarfSection::DebugInfo => debug_info_offset,
            crate::jit_dwarf::DwarfSection::DebugLine => debug_line_offset,
            crate::jit_dwarf::DwarfSection::DebugAranges => None, // aranges handled by dSYM builder
        };
        let Some(base) = base else { continue };
        let offset = base as usize + reloc.offset as usize;
        if offset + 8 <= data.len() {
            data[offset..offset + 8].copy_from_slice(&addr_bytes);
            eprintln!(
                "[harness]   patched {:?} @ 0x{:x} (section+{})",
                section, offset, reloc.offset
            );
        }
    }

    std::fs::write(obj_path, &data).map_err(|e| HarnessError::Io("write patched .o", e))?;
    Ok(())
}

fn link_harness(c_path: &Path, obj_path: &Path, exe_path: &Path) -> Result<(), HarnessError> {
    // Find the kajit staticlib for intrinsic resolution.
    // Build it if needed: `cargo rustc -p kajit --crate-type=staticlib`
    let staticlib = find_or_build_staticlib("kajit", "libkajit.a")?;
    let vtables_lib = find_or_build_staticlib("kajit-vtables", "libkajit_vtables.a")?;

    let mut command = std::process::Command::new("cc");
    command.arg("-O0");

    #[cfg(target_os = "macos")]
    {
        command.arg("-g");
    }

    #[cfg(target_os = "linux")]
    {
        // Keep the generated decoder's DWARF as the only debug contribution.
        // Our standalone .debug_info currently hard-codes section offsets like
        // abbrev_offset=0 and stmt_list=0, which only remain valid when the
        // linker does not merge in an additional C TU's DWARF contribution.
        command.arg("-g0");
    }

    command.arg("-o").arg(exe_path).arg(c_path).arg(obj_path);

    command.arg(&staticlib);

    // Force-load the vtables staticlib so its #[no_mangle] symbols are
    // available for the JIT object file's ExternAddr relocations.
    #[cfg(target_os = "macos")]
    {
        command.arg("-Wl,-force_load").arg(&vtables_lib);
    }
    #[cfg(target_os = "linux")]
    {
        command
            .arg("-Wl,--whole-archive")
            .arg(&vtables_lib)
            .arg("-Wl,--no-whole-archive");
    }

    #[cfg(target_os = "macos")]
    {
        command
            .arg("-Wl,-no_deduplicate")
            .arg("-lSystem")
            .arg("-lc++")
            .arg("-framework")
            .arg("Security");
    }

    #[cfg(target_os = "linux")]
    {
        for lib in linux_native_static_libs()? {
            command.arg(lib);
        }
    }

    let output = command
        .output()
        .map_err(|e| HarnessError::Io("invoke cc", e))?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Err(HarnessError::Link(stderr.into_owned()));
    }

    Ok(())
}

#[cfg(target_os = "linux")]
fn linux_native_static_libs() -> Result<Vec<String>, HarnessError> {
    let output = std::process::Command::new("cargo")
        .args([
            "rustc",
            "-p",
            "kajit",
            "--crate-type=staticlib",
            "--",
            "--print=native-static-libs",
        ])
        .output()
        .map_err(|e| HarnessError::Io("query native static libs", e))?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Err(HarnessError::Link(format!(
            "failed to query native static libs: {stderr}"
        )));
    }

    let stderr = String::from_utf8_lossy(&output.stderr);
    let line = stderr
        .lines()
        .find_map(|line| {
            line.split_once("native-static-libs: ")
                .map(|(_, libs)| libs)
        })
        .ok_or_else(|| HarnessError::Link("native-static-libs output missing".into()))?;

    Ok(line.split_whitespace().map(str::to_owned).collect())
}

#[cfg(target_os = "macos")]
fn maybe_build_debug_bundle(exe_path: &Path, input: &HarnessInput) {
    if let Some(dwarf) = &input.dwarf
        && let Err(e) = build_dsym(exe_path, dwarf, input.function_name, input.entry_offset)
    {
        eprintln!("[harness] warning: dSYM creation failed: {e}");
    }
}

#[cfg(target_os = "linux")]
fn maybe_build_debug_bundle(_exe_path: &Path, _input: &HarnessInput) {}

/// Find or build the kajit staticlib for linking intrinsics into standalone harnesses.
fn find_or_build_staticlib(
    package: &str,
    lib_filename: &str,
) -> Result<std::path::PathBuf, HarnessError> {
    // Check common locations for an existing staticlib
    let candidates = [
        format!("target/debug/{lib_filename}"),
        format!("target/release/{lib_filename}"),
        format!("../target/debug/{lib_filename}"),
    ];
    for path in &candidates {
        let p = std::path::PathBuf::from(path);
        if p.exists() {
            return Ok(p);
        }
    }

    // Try to build it
    eprintln!("[harness] building {package} staticlib...");
    let output = std::process::Command::new("cargo")
        .args(["rustc", "-p", package, "--crate-type=staticlib"])
        .output()
        .map_err(|e| HarnessError::Io("build staticlib", e))?;
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Err(HarnessError::Link(format!(
            "failed to build {package} staticlib: {stderr}"
        )));
    }

    // Find the built lib
    for path in &candidates {
        let p = std::path::PathBuf::from(path);
        if p.exists() {
            return Ok(p);
        }
    }

    Err(HarnessError::Link(format!(
        "{package} staticlib not found after build"
    )))
}

#[derive(Debug)]
pub enum HarnessError {
    Io(&'static str, std::io::Error),
    ObjectWrite(object::write::Error),
    Link(String),
}

impl std::fmt::Display for HarnessError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            HarnessError::Io(ctx, e) => write!(f, "{ctx}: {e}"),
            HarnessError::ObjectWrite(e) => write!(f, "object write: {e}"),
            HarnessError::Link(msg) => write!(f, "link failed: {msg}"),
        }
    }
}

impl std::error::Error for HarnessError {}
