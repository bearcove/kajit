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

    if std::env::var("KAJIT_RA_DEBUG").is_ok() {
        eprintln!(
            "[ssa_coloring] k={k}, spilled={}, colored={}",
            spilled.len(),
            coloring.len()
        );
        // Check for conflicts: any two colored vregs that interfere
        // and got the same color
        let mut conflicts = 0;
        for (&v1, &c1) in &coloring {
            for (&v2, &c2) in &coloring {
                if v1 >= v2 || c1 != c2 {
                    continue;
                }
                // Check if they interfere: both live at some common point
                // Simple check: same block live-in
                for (block_id, live_in) in &liveness.live_in {
                    if live_in.contains(&v1) && live_in.contains(&v2) {
                        eprintln!(
                            "  CONFLICT: v{} and v{} both colored p{}, both live-in at b{}",
                            v1.index(),
                            v2.index(),
                            c1.0,
                            block_id.0
                        );
                        conflicts += 1;
                        break;
                    }
                }
            }
        }
        if conflicts > 0 {
            eprintln!("  {} conflicts found!", conflicts);
        }
    }

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
/// 3. Scan instructions: free dead values before each def, assign lowest color to defs
///
/// Key correctness invariant: a color is "occupied" iff the vreg holding it
/// is still live (will be used again in this block or is live-out).
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

    // Pre-compute: for each (block, vreg), the index of the last instruction
    // that uses it in that block (including terminator and edge args).
    // u32::MAX means "used at/after terminator" (edge arg or branch cond).
    let mut last_use_in_block: HashMap<(BlockId, VReg), u32> = HashMap::new();
    for block in &func.blocks {
        if block.dead {
            continue;
        }
        // Instruction uses
        for (inst_idx, &inst_id) in block.insts.iter().enumerate() {
            let inst = &func.insts[inst_id.0 as usize];
            inst.op.for_each_use(|src| {
                last_use_in_block.insert((block.id, *src), inst_idx as u32);
            });
        }
        // Terminator condition uses
        let term = &func.terms[block.term.0 as usize];
        match term {
            cfg_mir::Terminator::BranchIf { cond, .. }
            | cfg_mir::Terminator::BranchIfZero { cond, .. } => {
                last_use_in_block.insert((block.id, *cond), u32::MAX);
            }
            _ => {}
        }
        // Edge arg source uses (at terminator)
        for &edge_id in &block.succs {
            let edge = &func.edges[edge_id.index()];
            if edge.from.0 == u32::MAX {
                continue;
            }
            for arg in &edge.args {
                last_use_in_block.insert((block.id, arg.source), u32::MAX);
            }
        }
    }

    for block_id in preorder {
        let block = &func.blocks[block_id.index()];
        if block.dead {
            continue;
        }

        let live_out = liveness.live_out.get(&block_id);

        // Track which colors are occupied in this block
        let mut occupied: HashSet<PReg> = HashSet::new();

        // Mark colors of live-in values (not spilled)
        if let Some(live_in) = liveness.live_in.get(&block_id) {
            for &vreg in live_in {
                if spilled.contains(&vreg) {
                    continue;
                }
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
        }

        // Scan instructions in program order
        for (inst_idx, &inst_id) in block.insts.iter().enumerate() {
            let inst = &func.insts[inst_id.0 as usize];

            // Before processing this instruction's defs, free colors of values
            // whose last use was at a PREVIOUS instruction and are NOT live-out.
            // (We free after last use, not at last use, to avoid freeing a color
            // that this instruction reads and another def wants.)
            let mut to_free = Vec::new();
            for (&preg, _) in occupied.iter().map(|p| (p, ())) {
                // Find which vreg holds this color
                // (linear scan through coloring — could be optimized with reverse map)
            }
            // Simpler approach: collect all uses of this instruction, then free
            // any previously-used vregs that are dead after this point.
            let mut uses_here: Vec<VReg> = Vec::new();
            inst.op.for_each_use(|src| {
                uses_here.push(*src);
            });

            // Free colors of vregs whose last use in this block is at inst_idx
            // and that are NOT live-out
            for &vreg in &uses_here {
                if spilled.contains(&vreg) {
                    continue;
                }
                let last = last_use_in_block.get(&(block_id, vreg)).copied();
                if last == Some(inst_idx as u32) && !live_out.map_or(false, |lo| lo.contains(&vreg))
                {
                    if let Some(&preg) = coloring.get(&vreg) {
                        to_free.push(preg);
                    }
                }
            }
            for preg in to_free {
                occupied.remove(&preg);
            }

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
/// SSA interference: v and w interfere iff def(v) dominates def(w) and v is
/// live at def(w), or vice versa.
///
/// "Live at def(w)" means:
/// - v is in live_in of w's block, OR
/// - v is defined in the SAME block as w, before w, and v is live-out or
///   used after w's definition point.
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

        // Check interference using SSA dominance property
        if interferes(vreg, vreg_def_block, other, other_def_block, liveness, dom) {
            return false;
        }
    }

    true
}

/// Check if two vregs interfere.
///
/// In SSA: v and w interfere iff one's definition dominates the other's
/// AND the dominator is live at the dominated definition point.
fn interferes(
    v: VReg,
    v_block: BlockId,
    w: VReg,
    w_block: BlockId,
    liveness: &LivenessInfo,
    dom: &DominanceInfo,
) -> bool {
    if v_block == w_block {
        // Same block: they interfere if their live ranges overlap within the block.
        // Conservative: if both are defined in the same block and either is live-out,
        // assume they interfere (since we don't track intra-block ordering here).
        let v_live_out = liveness
            .live_out
            .get(&v_block)
            .map_or(false, |lo| lo.contains(&v));
        let w_live_out = liveness
            .live_out
            .get(&w_block)
            .map_or(false, |lo| lo.contains(&w));
        // If either is live-out, they potentially overlap
        if v_live_out || w_live_out {
            return true;
        }
        // Both die in the same block — conservatively assume they interfere
        // (proper fix: check instruction ordering within the block)
        return true;
    }

    if dom.dominates(v_block, w_block) {
        // v defined before w — interfere if v is live at w's block entry
        // (live-in means v is alive when w gets defined)
        return liveness
            .live_in
            .get(&w_block)
            .map_or(false, |li| li.contains(&v));
    }

    if dom.dominates(w_block, v_block) {
        // w defined before v — interfere if w is live at v's block entry
        return liveness
            .live_in
            .get(&v_block)
            .map_or(false, |li| li.contains(&w));
    }

    // Neither dominates the other — in SSA, they can't interfere
    false
}
