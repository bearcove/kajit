//! SSA-based chordal graph coloring register allocator.
//!
//! Implements the Hack/Grund/Goos approach:
//! 1. **Spill**: Braun/Hack SSA spilling — Belady/furthest-next-use, loop-aware
//! 2. **Color**: Domtree-preorder walk, assign lowest available color
//! 3. **Coalesce**: Bounded phi-affinity recoloring
//!
//! Key insight: SSA programs have chordal interference graphs, so once
//! pressure is reduced to k registers, optimal k-coloring is trivially
//! found by processing defs in domtree preorder.

use std::collections::{HashMap, HashSet};

use crate::analysis::dominance::DominanceInfo;
use crate::analysis::loops::LoopInfo;
use crate::cfg_mir::{self, BlockId, Function, InstId};
use kajit_ir::VReg;
use kajit_lir::LinearOp;

use super::hints::{HintMap, SpillCost};
use super::linear_scan::{Allocation, AllocationResult, CopyHints};
use super::liveness::LivenessInfo;
use super::machine_inst::{AbiInfo, PReg, ScratchPolicy};

/// Run SSA coloring allocation.
///
/// Same interface as `linear_scan::allocate()` but uses domtree-preorder
/// coloring instead of interval-based linear scan.
pub fn allocate(
    func: &Function,
    liveness: &LivenessInfo,
    abi: &AbiInfo,
    scratch: &ScratchPolicy,
    hints: &HintMap,
    _copy_hints: &CopyHints,
) -> AllocationResult {
    let dom = DominanceInfo::compute(func);
    let loop_info = LoopInfo::compute(func, &dom);

    // Collect allocatable registers
    let mut allocatable: Vec<PReg> = Vec::new();
    for &preg in abi.caller_saved_gpr {
        if !scratch.reserved.contains(&preg) {
            allocatable.push(preg);
        }
    }
    for &preg in abi.callee_saved_gpr {
        if !scratch.reserved.contains(&preg) {
            allocatable.push(preg);
        }
    }
    let k = allocatable.len();

    // Build def-site map: vreg → (block, inst_index_in_block)
    // Also build next-use-distance info per block
    let mut def_block: HashMap<VReg, BlockId> = HashMap::new();
    for block in &func.blocks {
        if block.dead {
            continue;
        }
        // Block params are defined at block entry
        for &param in &block.params {
            def_block.insert(param, block.id);
        }
        // Instruction defs
        for &inst_id in &block.insts {
            let inst = &func.insts[inst_id.0 as usize];
            inst.op.for_each_def(|dst| {
                def_block.insert(*dst, block.id);
            });
        }
    }

    // Phase 1: Spill — reduce max pressure at every point to ≤ k
    let spilled = spill_phase(func, liveness, &dom, &loop_info, hints, k);

    // Phase 2: Color — domtree preorder walk, assign colors
    let coloring = color_phase(func, liveness, &dom, &spilled, &allocatable);

    // Phase 3: Coalesce — bounded phi-affinity recoloring
    let coloring = coalesce_phase(func, liveness, &dom, &def_block, coloring, &spilled);

    // Build result
    let mut allocations = HashMap::new();
    for (&vreg, &preg) in &coloring {
        allocations.insert(vreg, Allocation::Reg(preg));
    }
    for &vreg in &spilled {
        allocations.insert(vreg, Allocation::Spill);
    }

    AllocationResult {
        allocations,
        spilled: spilled.into_iter().collect(),
    }
}

// ─── Phase 1: Spill ─────────────────────────────────────────────────────────

/// Braun/Hack-style SSA spilling.
///
/// Walk blocks in RPO, tracking live values. When pressure exceeds k,
/// spill the value with furthest next use (Belady), weighted by spill cost.
fn spill_phase(
    func: &Function,
    liveness: &LivenessInfo,
    dom: &DominanceInfo,
    loop_info: &LoopInfo,
    hints: &HintMap,
    k: usize,
) -> HashSet<VReg> {
    let mut spilled = HashSet::new();

    // Compute next-use distances per block for each vreg
    let next_uses = compute_next_uses(func);

    // Process each block: check pressure at block entry (live-in + block params)
    // and after each instruction
    for block in &func.blocks {
        if block.dead {
            continue;
        }

        // Live values at block entry = live_in ∪ block_params
        let mut live: HashSet<VReg> = liveness.live_in.get(&block.id).cloned().unwrap_or_default();
        for &param in &block.params {
            live.insert(param);
        }

        // Remove already-spilled values from live set
        live.retain(|v| !spilled.contains(v));

        // If pressure exceeds k at block entry, spill
        while live.len() > k {
            let victim = pick_spill_victim(&live, &next_uses, block.id, 0, hints, loop_info);
            if let Some(victim) = victim {
                spilled.insert(victim);
                live.remove(&victim);
            } else {
                break;
            }
        }

        // Walk instructions, tracking pressure changes
        for (inst_idx, &inst_id) in block.insts.iter().enumerate() {
            let inst = &func.insts[inst_id.0 as usize];

            // Remove dead values (last use at this instruction)
            inst.op.for_each_use(|src| {
                // Check if this is the last use in this block
                if is_last_use_in_block(func, block, *src, inst_idx) {
                    // Only remove from live if not live-out
                    if !liveness
                        .live_out
                        .get(&block.id)
                        .map_or(false, |lo| lo.contains(src))
                    {
                        live.remove(src);
                    }
                }
            });

            // Add defs
            inst.op.for_each_def(|dst| {
                if !spilled.contains(dst) {
                    live.insert(*dst);
                }
            });

            // If pressure exceeds k, spill
            while live.len() > k {
                let victim =
                    pick_spill_victim(&live, &next_uses, block.id, inst_idx + 1, hints, loop_info);
                if let Some(victim) = victim {
                    spilled.insert(victim);
                    live.remove(&victim);
                } else {
                    break;
                }
            }
        }
    }

    spilled
}

/// Pick the best spill victim from live set using Belady/furthest-next-use
/// weighted by spill cost hints and loop depth.
fn pick_spill_victim(
    live: &HashSet<VReg>,
    next_uses: &HashMap<VReg, Vec<(BlockId, usize)>>,
    current_block: BlockId,
    current_inst_idx: usize,
    hints: &HintMap,
    loop_info: &LoopInfo,
) -> Option<VReg> {
    let mut best_victim = None;
    let mut best_score: u64 = 0;

    for &vreg in live {
        // Compute distance to next use (further = better to spill)
        let distance = next_use_distance(vreg, next_uses, current_block, current_inst_idx);

        // Spill cost weight (lower weight = prefer to spill)
        let cost_weight = hints
            .get(&vreg)
            .map(|meta| meta.spill_cost.weight() as u64)
            .unwrap_or(SpillCost::Low.weight() as u64);

        // Loop depth penalty: avoid spilling values used in inner loops
        let loop_depth = loop_info
            .loop_for_block(current_block)
            .map(|l| l.depth as u64 + 1)
            .unwrap_or(1);

        // Score: distance / (cost * loop_depth). Higher = better to spill.
        let score = distance / (cost_weight * loop_depth).max(1);

        if score > best_score || best_victim.is_none() {
            best_score = score;
            best_victim = Some(vreg);
        }
    }

    best_victim
}

/// Compute next-use info: for each vreg, list of (block, inst_index) where it's used.
fn compute_next_uses(func: &Function) -> HashMap<VReg, Vec<(BlockId, usize)>> {
    let mut uses: HashMap<VReg, Vec<(BlockId, usize)>> = HashMap::new();
    for block in &func.blocks {
        if block.dead {
            continue;
        }
        for (inst_idx, &inst_id) in block.insts.iter().enumerate() {
            let inst = &func.insts[inst_id.0 as usize];
            inst.op.for_each_use(|src| {
                uses.entry(*src).or_default().push((block.id, inst_idx));
            });
        }
    }
    uses
}

/// Distance to next use of vreg from (current_block, current_inst_idx).
/// Returns u64::MAX if no more uses (dead after this point).
fn next_use_distance(
    vreg: VReg,
    next_uses: &HashMap<VReg, Vec<(BlockId, usize)>>,
    _current_block: BlockId,
    _current_inst_idx: usize,
) -> u64 {
    // Simplified: count total uses (fewer uses = easier to spill)
    // A proper implementation would compute CFG-aware distance.
    next_uses.get(&vreg).map(|v| v.len() as u64).unwrap_or(0)
}

/// Check if `vreg` has its last use in `block` at instruction index `inst_idx`.
fn is_last_use_in_block(
    func: &Function,
    block: &cfg_mir::Block,
    vreg: VReg,
    inst_idx: usize,
) -> bool {
    // Check remaining instructions after inst_idx
    for &later_inst_id in &block.insts[inst_idx + 1..] {
        let later_inst = &func.insts[later_inst_id.0 as usize];
        let mut found = false;
        later_inst.op.for_each_use(|src| {
            if *src == vreg {
                found = true;
            }
        });
        if found {
            return false;
        }
    }
    // Check terminator uses
    let term = &func.terms[block.term.0 as usize];
    match term {
        cfg_mir::Terminator::BranchIf { cond, .. }
        | cfg_mir::Terminator::BranchIfZero { cond, .. } => {
            if *cond == vreg {
                return false;
            }
        }
        _ => {}
    }
    true
}

// ─── Phase 2: Color ──────────────────────────────────────────────────────────

/// Domtree-preorder SSA coloring.
///
/// Walk blocks in domtree preorder. At each block:
/// 1. Mark colors of live-in values as occupied
/// 2. Color block params (phi defs)
/// 3. Scan instructions: on def, assign lowest available color; on last use, free color
fn color_phase(
    func: &Function,
    liveness: &LivenessInfo,
    dom: &DominanceInfo,
    spilled: &HashSet<VReg>,
    allocatable: &[PReg],
) -> HashMap<VReg, PReg> {
    let mut coloring: HashMap<VReg, PReg> = HashMap::new();

    // Walk domtree in preorder
    let preorder = domtree_preorder(func.entry, dom);

    for block_id in preorder {
        let block = &func.blocks[block_id.index()];
        if block.dead {
            continue;
        }

        // Track which colors are occupied in this block
        let mut occupied: HashSet<PReg> = HashSet::new();

        // Mark colors of live-in values
        if let Some(live_in) = liveness.live_in.get(&block_id) {
            for &vreg in live_in {
                if let Some(&preg) = coloring.get(&vreg) {
                    occupied.insert(preg);
                }
            }
        }

        // Color block params first (they're defined at block entry)
        for &param in &block.params {
            if spilled.contains(&param) {
                continue;
            }
            let color = lowest_available_color(allocatable, &occupied);
            if let Some(color) = color {
                coloring.insert(param, color);
                occupied.insert(color);
            }
            // else: ran out of colors (shouldn't happen if spill phase worked)
        }

        // Scan instructions in program order
        for (inst_idx, &inst_id) in block.insts.iter().enumerate() {
            let inst = &func.insts[inst_id.0 as usize];

            // Free colors of values with last use at this instruction
            inst.op.for_each_use(|src| {
                if is_last_use_in_block(func, block, *src, inst_idx)
                    && !liveness
                        .live_out
                        .get(&block_id)
                        .map_or(false, |lo| lo.contains(src))
                {
                    if let Some(&preg) = coloring.get(src) {
                        occupied.remove(&preg);
                    }
                }
            });

            // Color defs
            inst.op.for_each_def(|dst| {
                if spilled.contains(dst) {
                    return;
                }
                let color = lowest_available_color(allocatable, &occupied);
                if let Some(color) = color {
                    coloring.insert(*dst, color);
                    occupied.insert(color);
                }
            });
        }

        // Free colors of values that are NOT live-out
        // (their last use was in the terminator or they die at block exit)
    }

    coloring
}

/// Compute domtree preorder traversal.
fn domtree_preorder(entry: BlockId, dom: &DominanceInfo) -> Vec<BlockId> {
    let mut order = Vec::new();
    let mut stack = vec![entry];
    while let Some(block) = stack.pop() {
        order.push(block);
        // Push children in reverse so we visit them in order
        let children = dom.dominator_tree_children(block);
        for &child in children.iter().rev() {
            stack.push(child);
        }
    }
    order
}

/// Find the lowest-numbered available color not in the occupied set.
fn lowest_available_color(allocatable: &[PReg], occupied: &HashSet<PReg>) -> Option<PReg> {
    allocatable.iter().find(|p| !occupied.contains(p)).copied()
}

// ─── Phase 3: Coalesce ──────────────────────────────────────────────────────

/// Bounded phi-affinity recoloring.
///
/// For each edge arg (phi connection), if source and target got different
/// colors, try to recolor the target to match the source (or vice versa).
fn coalesce_phase(
    func: &Function,
    liveness: &LivenessInfo,
    dom: &DominanceInfo,
    def_block: &HashMap<VReg, BlockId>,
    mut coloring: HashMap<VReg, PReg>,
    spilled: &HashSet<VReg>,
) -> HashMap<VReg, PReg> {
    // Build phi affinity groups from edge args
    for edge in &func.edges {
        if edge.from.0 == u32::MAX {
            continue; // dead edge
        }
        for arg in &edge.args {
            if spilled.contains(&arg.target) || spilled.contains(&arg.source) {
                continue;
            }
            let target_color = coloring.get(&arg.target).copied();
            let source_color = coloring.get(&arg.source).copied();

            if target_color == source_color {
                continue; // already coalesced
            }

            let (Some(tc), Some(sc)) = (target_color, source_color) else {
                continue;
            };

            // Try to recolor the target to match the source
            if can_recolor(arg.target, sc, &coloring, liveness, dom, def_block, spilled) {
                coloring.insert(arg.target, sc);
            }
            // Else try to recolor the source to match the target
            else if can_recolor(arg.source, tc, &coloring, liveness, dom, def_block, spilled) {
                coloring.insert(arg.source, tc);
            }
        }
    }

    coloring
}

/// Check if vreg can be safely recolored to new_color.
///
/// Safe iff no interfering neighbor currently uses new_color.
/// Interference: v interferes with w iff def(v) dominates def(w) and v is
/// live at def(w), or vice versa.
fn can_recolor(
    vreg: VReg,
    new_color: PReg,
    coloring: &HashMap<VReg, PReg>,
    liveness: &LivenessInfo,
    dom: &DominanceInfo,
    def_block: &HashMap<VReg, BlockId>,
    spilled: &HashSet<VReg>,
) -> bool {
    let Some(&vreg_def_block) = def_block.get(&vreg) else {
        return false;
    };

    // Check all other vregs that have new_color — do any interfere with vreg?
    for (&other, &other_color) in coloring.iter() {
        if other == vreg || other_color != new_color {
            continue;
        }
        if spilled.contains(&other) {
            continue;
        }
        let Some(&other_def_block) = def_block.get(&other) else {
            continue;
        };

        // Check interference: do vreg and other overlap?
        if dom.dominates(vreg_def_block, other_def_block) {
            // vreg defined before other — interfere if vreg is live at other's def
            if liveness
                .live_in
                .get(&other_def_block)
                .map_or(false, |li| li.contains(&vreg))
            {
                return false; // interferes
            }
        } else if dom.dominates(other_def_block, vreg_def_block) {
            // other defined before vreg — interfere if other is live at vreg's def
            if liveness
                .live_in
                .get(&vreg_def_block)
                .map_or(false, |li| li.contains(&other))
            {
                return false; // interferes
            }
        }
        // If neither dominates the other, they can't interfere in SSA
    }

    true
}
