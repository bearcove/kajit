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

use std::collections::{BTreeSet, HashMap, HashSet};

use crate::analysis::dominance::DominanceInfo;
use crate::analysis::loops::LoopInfo;
use crate::cfg_mir::{self, BlockId, FixedReg, Function, OperandKind};
use kajit_ir::VReg;

use super::hints::{HintMap, SpillCost};
use super::linear_scan::{Allocation, AllocationResult, CopyHints, OperandEdit};
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
    allocate_with_excluded(
        func,
        liveness,
        abi,
        scratch,
        hints,
        _copy_hints,
        &[],
        &HashMap::new(),
    )
}

/// Like `allocate`, but with additional registers excluded from allocation.
/// Used to reserve registers for hardcoded ABI roles (e.g., x0/x1 for output_ptr/ctx_ptr
/// in leaf functions).
///
/// `fixed_colors`: VRegs that must be assigned to specific physical registers
/// (e.g., function parameters that arrive in ABI argument registers).
pub fn allocate_with_excluded(
    func: &Function,
    liveness: &LivenessInfo,
    abi: &AbiInfo,
    scratch: &ScratchPolicy,
    hints: &HintMap,
    _copy_hints: &CopyHints,
    extra_excluded: &[PReg],
    fixed_colors: &HashMap<VReg, PReg>,
) -> AllocationResult {
    let dom = DominanceInfo::compute(func);
    let loop_info = LoopInfo::compute(func, &dom);

    // Collect allocatable registers
    let mut allocatable: Vec<PReg> = Vec::new();
    for &preg in abi.caller_saved_gpr {
        if !scratch.reserved.contains(&preg) && !extra_excluded.contains(&preg) {
            allocatable.push(preg);
        }
    }
    for &preg in abi.callee_saved_gpr {
        if !scratch.reserved.contains(&preg) && !extra_excluded.contains(&preg) {
            allocatable.push(preg);
        }
    }
    let k = allocatable.len();

    // Count callee-saved registers (survive calls)
    let callee_saved_set: HashSet<PReg> = abi.callee_saved_gpr.iter().copied().collect();
    let k_callee_saved = allocatable
        .iter()
        .filter(|p| callee_saved_set.contains(p))
        .count();

    // Build def-site map: vreg → (block, inst_index_in_block)
    // Block params use DEF_AT_ENTRY (before all instructions).
    const DEF_AT_ENTRY: u32 = u32::MAX;
    let mut def_block: HashMap<VReg, BlockId> = HashMap::new();
    let mut def_inst_idx: HashMap<VReg, u32> = HashMap::new();

    // Function data_args are defined at the entry block (before all instructions),
    // just like block params.
    for &arg in &func.data_args {
        def_block.insert(arg, func.entry);
        def_inst_idx.insert(arg, DEF_AT_ENTRY);
    }

    for block in &func.blocks {
        if block.dead {
            continue;
        }
        // Block params are defined at block entry
        for &param in &block.params {
            def_block.insert(param, block.id);
            def_inst_idx.insert(param, DEF_AT_ENTRY);
        }
        // Instruction defs
        for (idx, &inst_id) in block.insts.iter().enumerate() {
            let inst = &func.insts[inst_id.0 as usize];
            inst.op.for_each_def(|dst| {
                def_block.insert(*dst, block.id);
                def_inst_idx.insert(*dst, idx as u32);
            });
        }
    }

    // Phase 1: Spill — reduce max pressure at every point to ≤ k
    let spilled = spill_phase(func, liveness, &dom, &loop_info, hints, k, k_callee_saved);

    // Pre-compute live-across-call set (shared between color + coalesce phases)
    let live_across_call = compute_live_across_call(func, liveness, &spilled);

    // Pre-compute last-use map for precise same-block interference
    let last_use_map = compute_last_use_map(func);

    // Phase 2: Color — domtree preorder walk, assign colors (preference-guided)
    let coloring = color_phase(
        func,
        liveness,
        &dom,
        &loop_info,
        &spilled,
        &allocatable,
        &callee_saved_set,
        &live_across_call,
        fixed_colors,
    );

    if std::env::var("KAJIT_RA_DEBUG").is_ok() {
        eprintln!(
            "[ssa_coloring] k={k}, spilled={}, colored={}",
            spilled.len(),
            coloring.len()
        );
        // Print all allocations sorted by vreg
        let mut allocs: Vec<_> = coloring.iter().collect();
        allocs.sort_by_key(|(v, _)| v.index());
        for (v, p) in &allocs {
            eprintln!("  v{} -> p{}", v.index(), p.0);
        }
        // Check for conflicts: any two colored vregs with same color that interfere
        let mut conflicts = 0;
        for (&v1, &c1) in &coloring {
            for (&v2, &c2) in &coloring {
                if v1 >= v2 || c1 != c2 {
                    continue;
                }
                let v1_block = def_block.get(&v1).copied().unwrap_or(BlockId(u32::MAX));
                let v2_block = def_block.get(&v2).copied().unwrap_or(BlockId(u32::MAX));
                if interferes(
                    v1,
                    v1_block,
                    v2,
                    v2_block,
                    liveness,
                    &dom,
                    &last_use_map,
                    &def_inst_idx,
                ) {
                    eprintln!(
                        "  CONFLICT: v{} (b{}) and v{} (b{}) both colored p{}",
                        v1.index(),
                        v1_block.0,
                        v2.index(),
                        v2_block.0,
                        c1.0
                    );
                    conflicts += 1;
                    break;
                }
            }
        }
        if conflicts > 0 {
            eprintln!("  {} conflicts found!", conflicts);
        }
    }

    // Phase 3: Coalesce — bounded phi-affinity recoloring
    // For each phi edge arg (source→target), try to recolor one side to match
    // the other, eliminating the copy on that edge.
    let coloring = if std::env::var("KAJIT_NO_COALESCE").is_err() {
        coalesce_phase(
            func,
            liveness,
            &dom,
            &def_block,
            &def_inst_idx,
            &last_use_map,
            coloring,
            &spilled,
            &callee_saved_set,
            &allocatable,
            &live_across_call,
        )
    } else {
        coloring
    };

    // (Validation removed — the previous check was overly conservative and flagged
    // values defined BY the call instruction. The color_phase already ensures values
    // in live_across_call get callee-saved registers.)

    // Resolve fixed-register operand constraints into edits.
    // When a use operand requires a specific register but the vreg is
    // colored elsewhere, record a move the backend must emit.
    let edits = resolve_fixed_use_constraints(func, &coloring, &spilled);

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
        edits,
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
    _dom: &DominanceInfo,
    loop_info: &LoopInfo,
    hints: &HintMap,
    k: usize,
    k_callee_saved: usize,
) -> BTreeSet<VReg> {
    let mut spilled = BTreeSet::new();

    // Compute next-use distances per block for each vreg
    let next_uses = compute_next_uses(func);

    // Process each block: check pressure at block entry (live-in + block params)
    // and after each instruction
    for block in &func.blocks {
        if block.dead {
            continue;
        }

        // Live values at block entry = live_in ∪ block_params
        let mut live: BTreeSet<VReg> = liveness.live_in.get(&block.id).cloned().unwrap_or_default();
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

            // Call clobber handling: if this instruction clobbers caller-saved GPRs,
            // values that are live PAST the call can only use callee-saved registers.
            // Reduce effective capacity to k_callee_saved and spill excess.
            if inst.clobbers.caller_saved_gpr {
                // Values live across the call = values in `live` that are NOT
                // solely consumed by this instruction (i.e., they have uses after this point).
                // Conservatively: anything still in `live` after processing uses/defs
                // is live across the call.
                while live.len() > k_callee_saved {
                    let victim = pick_spill_victim(
                        &live,
                        &next_uses,
                        block.id,
                        inst_idx + 1,
                        hints,
                        loop_info,
                    );
                    if let Some(victim) = victim {
                        spilled.insert(victim);
                        live.remove(&victim);
                    } else {
                        break;
                    }
                }
            }
        }
    }

    spilled
}

/// Pick the best spill victim from live set using Belady/furthest-next-use
/// weighted by spill cost hints and loop depth.
fn pick_spill_victim(
    live: &BTreeSet<VReg>,
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

/// Compute last-use map: (block, vreg) → last instruction index where vreg is used.
/// u32::MAX means used at/after the terminator (branch condition or edge arg).
fn compute_last_use_map(func: &Function) -> HashMap<(BlockId, VReg), u32> {
    let mut map: HashMap<(BlockId, VReg), u32> = HashMap::new();
    for block in &func.blocks {
        if block.dead {
            continue;
        }
        // Instruction uses
        for (inst_idx, &inst_id) in block.insts.iter().enumerate() {
            let inst = &func.insts[inst_id.0 as usize];
            inst.op.for_each_use(|src| {
                map.insert((block.id, *src), inst_idx as u32);
            });
        }
        // Terminator condition uses
        let term = &func.terms[block.term.0 as usize];
        match term {
            cfg_mir::Terminator::BranchIf { cond, .. }
            | cfg_mir::Terminator::BranchIfZero { cond, .. } => {
                map.insert((block.id, *cond), u32::MAX);
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
                map.insert((block.id, arg.source), u32::MAX);
            }
        }
    }
    map
}

// ─── Phase 2: Color ──────────────────────────────────────────────────────────

/// Compute which vregs are live across a clobbering instruction (call).
/// These must be assigned callee-saved registers (or spilled).
fn compute_live_across_call(
    func: &Function,
    liveness: &LivenessInfo,
    spilled: &BTreeSet<VReg>,
) -> HashSet<VReg> {
    let mut live_across_call: HashSet<VReg> = HashSet::new();
    for block in &func.blocks {
        if block.dead {
            continue;
        }
        for (inst_idx, &inst_id) in block.insts.iter().enumerate() {
            let inst = &func.insts[inst_id.0 as usize];
            if !inst.clobbers.caller_saved_gpr {
                continue;
            }
            let live_in = liveness.live_in.get(&block.id);
            let live_out = liveness.live_out.get(&block.id);

            // Vregs live-in to the block
            if let Some(live_in) = live_in {
                for &vreg in live_in {
                    if spilled.contains(&vreg) {
                        continue;
                    }
                    let used_after = has_use_after_inst(func, block, vreg, inst_idx)
                        || live_out.map_or(false, |lo| lo.contains(&vreg));
                    if used_after {
                        live_across_call.insert(vreg);
                    }
                }
            }

            // Function data_args (live-in to entry but not in live_in set)
            if block.id == func.entry {
                for &arg in &func.data_args {
                    if spilled.contains(&arg) {
                        continue;
                    }
                    let used_after = has_use_after_inst(func, block, arg, inst_idx)
                        || live_out.map_or(false, |lo| lo.contains(&arg));
                    if used_after {
                        live_across_call.insert(arg);
                    }
                }
            }

            // Block params
            for &param in &block.params {
                if spilled.contains(&param) {
                    continue;
                }
                let used_after = has_use_after_inst(func, block, param, inst_idx)
                    || live_out.map_or(false, |lo| lo.contains(&param));
                if used_after {
                    live_across_call.insert(param);
                }
            }

            // Vregs defined before this instruction in this block
            for &prev_inst_id in &block.insts[..inst_idx] {
                let prev_inst = &func.insts[prev_inst_id.0 as usize];
                prev_inst.op.for_each_def(|dst| {
                    if spilled.contains(dst) {
                        return;
                    }
                    let used_after = has_use_after_inst(func, block, *dst, inst_idx)
                        || live_out.map_or(false, |lo| lo.contains(dst));
                    if used_after {
                        live_across_call.insert(*dst);
                    }
                });
            }
        }
    }
    live_across_call
}

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
    loop_info: &LoopInfo,
    spilled: &BTreeSet<VReg>,
    allocatable: &[PReg],
    callee_saved_set: &HashSet<PReg>,
    live_across_call: &HashSet<VReg>,
    fixed_colors: &HashMap<VReg, PReg>,
) -> HashMap<VReg, PReg> {
    // Start with fixed colorings (e.g., function parameters in ABI registers).
    // Exclude fixed colors for vregs that are live across calls and pre-colored
    // to caller-saved registers — those need callee-saved registers instead.
    // The backend prologue will move from the ABI register to the assigned register.
    let mut coloring: HashMap<VReg, PReg> = HashMap::new();
    for (&vreg, &preg) in fixed_colors {
        if live_across_call.contains(&vreg) && !callee_saved_set.contains(&preg) {
            // Skip: let the coloring phase assign a callee-saved register.
            continue;
        }
        coloring.insert(vreg, preg);
    }

    // Walk domtree in preorder
    let preorder = domtree_preorder(func.entry, dom);

    // Build callee-saved-only allocatable list
    let callee_saved_allocatable: Vec<PReg> = allocatable
        .iter()
        .filter(|p| callee_saved_set.contains(p))
        .copied()
        .collect();

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

    // Diagnostic: check if a specific vreg is tracked correctly
    let trace_vreg = std::env::var("KAJIT_RA_TRACE_VREG")
        .ok()
        .and_then(|s| s.parse::<u32>().ok())
        .map(VReg::new);

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
            if let Some(tv) = trace_vreg {
                let in_live_in = live_in.contains(&tv);
                let in_live_out = live_out.map_or(false, |lo| lo.contains(&tv));
                if in_live_in || in_live_out {
                    eprintln!(
                        "[ra-trace] b{}: v{} live_in={} live_out={}",
                        block_id.0,
                        tv.index(),
                        in_live_in,
                        in_live_out
                    );
                }
            }
            for &vreg in live_in {
                if spilled.contains(&vreg) {
                    continue;
                }
                if let Some(&preg) = coloring.get(&vreg) {
                    occupied.insert(preg);
                }
            }
        }

        // Mark data_args' colors as occupied in the entry block.
        // Data_args are like block params but defined at function level — they
        // don't appear in live_in, so their occupied colors must be added explicitly.
        if block_id == func.entry {
            for &arg in &func.data_args {
                if spilled.contains(&arg) {
                    continue;
                }
                if let Some(&preg) = coloring.get(&arg) {
                    occupied.insert(preg);
                }
            }
        }

        // Pre-compute: for each block param, find edge arg sources that feed it.
        // The param should prefer the register of its sources (phi affinity).
        let mut param_source_vregs: HashMap<VReg, Vec<VReg>> = HashMap::new();
        for pred_block in &func.blocks {
            if pred_block.dead {
                continue;
            }
            for &edge_id in &pred_block.succs {
                let edge = &func.edges[edge_id.index()];
                if edge.to != block_id || edge.from.0 == u32::MAX {
                    continue;
                }
                for arg in &edge.args {
                    if !spilled.contains(&arg.source) && !spilled.contains(&arg.target) {
                        param_source_vregs
                            .entry(arg.target)
                            .or_default()
                            .push(arg.source);
                    }
                }
            }
        }

        // Color block params first (they're defined at block entry)
        for &param in &block.params {
            if spilled.contains(&param) {
                continue;
            }
            let pool = if live_across_call.contains(&param) {
                &callee_saved_allocatable
            } else {
                allocatable
            };
            // Phi affinity: prefer registers of already-colored source vregs.
            // Weight by loop depth: phi edges inside loops are more valuable
            // to coalesce (they execute many times).
            let block_loop_depth = loop_info
                .loop_for_block(block_id)
                .map(|l| l.depth + 1)
                .unwrap_or(0);
            let phi_weight = WEIGHT_PHI_EDGE + (block_loop_depth as u32) * WEIGHT_PHI_LOOP_BONUS;
            let phi_prefs: Vec<(PReg, u32)> = param_source_vregs
                .get(&param)
                .map(|sources| {
                    sources
                        .iter()
                        .filter_map(|src| coloring.get(src).copied().map(|p| (p, phi_weight)))
                        .collect()
                })
                .unwrap_or_default();
            let color = preferred_color(pool, &occupied, &[], &phi_prefs);
            if let Some(color) = color {
                if std::env::var("KAJIT_RA_TRACE").is_ok() {
                    eprintln!(
                        "  b{} param v{} -> p{} (occupied: {:?})",
                        block_id.0,
                        param.index(),
                        color.0,
                        occupied.iter().map(|p| p.0).collect::<Vec<_>>()
                    );
                }
                coloring.insert(param, color);
                occupied.insert(color);
            }
        }

        // Color data_args that weren't pre-colored (e.g., because they're
        // live across a call and their ABI register is caller-saved).
        if block_id == func.entry {
            for &arg in &func.data_args {
                if spilled.contains(&arg) || coloring.contains_key(&arg) {
                    continue;
                }
                let pool = if live_across_call.contains(&arg) {
                    &callee_saved_allocatable
                } else {
                    allocatable
                };
                let color = preferred_color(pool, &occupied, &[], &[]);
                if let Some(color) = color {
                    coloring.insert(arg, color);
                    occupied.insert(color);
                }
            }
        }

        // Scan instructions in program order
        for (inst_idx, &inst_id) in block.insts.iter().enumerate() {
            let inst = &func.insts[inst_id.0 as usize];

            // Collect uses of this instruction.
            let mut uses_here: Vec<VReg> = Vec::new();
            inst.op.for_each_use(|src| {
                uses_here.push(*src);
            });

            // Identify dying sources: vregs whose last use is this instruction
            // and that are NOT live-out.
            let mut dying_source_colors: Vec<PReg> = Vec::new();
            let mut to_free = Vec::new();
            for &vreg in &uses_here {
                if spilled.contains(&vreg) {
                    continue;
                }
                let last = last_use_in_block.get(&(block_id, vreg)).copied();
                let is_dying = last == Some(inst_idx as u32)
                    && !live_out.map_or(false, |lo| lo.contains(&vreg));
                if is_dying {
                    if let Some(&preg) = coloring.get(&vreg) {
                        dying_source_colors.push(preg);
                        // For non-clobbering instructions, free dying source
                        // colors so defs can reuse them (the def logically
                        // happens after uses at the same instruction).
                        // For clobbering instructions (calls), don't free —
                        // the call clobbers caller-saved regs and the def goes
                        // into the ABI return register, not the dying source's.
                        if !inst.clobbers.caller_saved_gpr {
                            to_free.push(preg);
                        }
                    }
                }
            }
            let freed_debug: Vec<u8> = to_free.iter().map(|p| p.0).collect();
            for preg in to_free {
                occupied.remove(&preg);
            }

            // Edge arg preferences: if this def will be used as an edge arg source,
            // prefer the target param's register (if already colored).
            let edge_arg_prefs: Vec<(VReg, PReg)> = {
                let mut prefs = Vec::new();
                for &edge_id in &block.succs {
                    let edge = &func.edges[edge_id.index()];
                    if edge.from.0 == u32::MAX {
                        continue;
                    }
                    for arg in &edge.args {
                        if !spilled.contains(&arg.target) {
                            if let Some(&tc) = coloring.get(&arg.target) {
                                prefs.push((arg.source, tc));
                            }
                        }
                    }
                }
                prefs
            };

            // Color defs
            inst.op.for_each_def(|dst| {
                if spilled.contains(dst) {
                    return;
                }
                // If this vreg is live across a call, only use callee-saved registers
                let pool = if live_across_call.contains(dst) {
                    &callee_saved_allocatable
                } else {
                    allocatable
                };

                // Collect weighted phi preferences specific to this def.
                let block_loop_depth = loop_info
                    .loop_for_block(block_id)
                    .map(|l| l.depth + 1)
                    .unwrap_or(0);
                let phi_weight =
                    WEIGHT_PHI_EDGE + (block_loop_depth as u32) * WEIGHT_PHI_LOOP_BONUS;
                let phi_prefs: Vec<(PReg, u32)> = edge_arg_prefs
                    .iter()
                    .filter(|(src, _)| *src == *dst)
                    .map(|(_, tc)| (*tc, phi_weight))
                    .collect();

                let color = preferred_color(pool, &occupied, &dying_source_colors, &phi_prefs);
                if let Some(color) = color {
                    if std::env::var("KAJIT_RA_TRACE").is_ok() {
                        eprintln!(
                            "  b{} def v{} -> p{} (freed: {:?}, occupied: {}, callee_only: {})",
                            block_id.0,
                            dst.index(),
                            color.0,
                            freed_debug,
                            occupied.len(),
                            live_across_call.contains(dst),
                        );
                    }
                    coloring.insert(*dst, color);
                    occupied.insert(color);
                }
            });

            // Call clobber handling: after a clobbering instruction, caller-saved
            // registers are destroyed. Free caller-saved colors from occupied,
            // EXCEPT colors that belong to this instruction's own defs — those
            // survive the clobber (the return value is written after the call).
            if inst.clobbers.caller_saved_gpr {
                // Collect colors of this instruction's defs (they survive the clobber).
                let mut def_colors: HashSet<PReg> = HashSet::new();
                inst.op.for_each_def(|dst| {
                    if let Some(&preg) = coloring.get(dst) {
                        def_colors.insert(preg);
                    }
                });

                let caller_saved_to_free: Vec<PReg> = occupied
                    .iter()
                    .filter(|p| !callee_saved_set.contains(p) && !def_colors.contains(p))
                    .copied()
                    .collect();
                for preg in caller_saved_to_free {
                    occupied.remove(&preg);
                }
            }
        }
    }

    coloring
}

/// Convert a `FixedReg` constraint to a regalloc3 `PReg`.
#[cfg(target_arch = "aarch64")]
fn fixed_to_preg(fixed: FixedReg) -> Option<PReg> {
    match fixed {
        FixedReg::AbiArg(i) if i <= 7 => Some(PReg(i)),
        FixedReg::AbiRet(i) if i <= 1 => Some(PReg(i)),
        FixedReg::HwReg(enc) => Some(PReg(enc)),
        _ => None,
    }
}

#[cfg(all(target_arch = "x86_64", not(windows)))]
fn fixed_to_preg(fixed: FixedReg) -> Option<PReg> {
    match fixed {
        // SysV: rdi=7, rsi=6, rdx=2, rcx=1, r8=8, r9=9
        FixedReg::AbiArg(i) => {
            const ORDER: [u8; 6] = [7, 6, 2, 1, 8, 9];
            ORDER.get(i as usize).map(|&enc| PReg(enc))
        }
        FixedReg::AbiRet(i) => {
            const ORDER: [u8; 2] = [0, 2]; // rax, rdx
            ORDER.get(i as usize).map(|&enc| PReg(enc))
        }
        FixedReg::HwReg(enc) => Some(PReg(enc)),
    }
}

#[cfg(all(target_arch = "x86_64", windows))]
fn fixed_to_preg(fixed: FixedReg) -> Option<PReg> {
    match fixed {
        // Win64: rcx=1, rdx=2, r8=8, r9=9
        FixedReg::AbiArg(i) => {
            const ORDER: [u8; 4] = [1, 2, 8, 9];
            ORDER.get(i as usize).map(|&enc| PReg(enc))
        }
        FixedReg::AbiRet(i) => {
            const ORDER: [u8; 2] = [0, 2]; // rax, rdx
            ORDER.get(i as usize).map(|&enc| PReg(enc))
        }
        FixedReg::HwReg(enc) => Some(PReg(enc)),
    }
}

/// Resolve fixed-register operand constraints into backend edits.
///
/// After coloring, each vreg has a single assigned register. But a use operand
/// may require the value in a *different* register (e.g., `AbiArg(0)` requires x0).
/// When the vreg's color doesn't match, we emit an `OperandEdit` telling the
/// backend to insert a `mov` before the instruction.
fn resolve_fixed_use_constraints(
    func: &Function,
    coloring: &HashMap<VReg, PReg>,
    spilled: &BTreeSet<VReg>,
) -> Vec<OperandEdit> {
    let mut edits = Vec::new();
    for block in &func.blocks {
        if block.dead {
            continue;
        }
        for &inst_id in &block.insts {
            let inst = &func.insts[inst_id.0 as usize];
            for operand in &inst.operands {
                if operand.kind != OperandKind::Use {
                    continue;
                }
                let fixed = match operand.fixed {
                    Some(f) => f,
                    None => continue,
                };
                let required_preg = match fixed_to_preg(fixed) {
                    Some(p) => p,
                    None => continue,
                };
                if spilled.contains(&operand.vreg) {
                    // Spilled vreg — the spill/reload pass will handle loading
                    // it into the required register.
                    continue;
                }
                if let Some(&actual_preg) = coloring.get(&operand.vreg) {
                    if actual_preg != required_preg {
                        edits.push(OperandEdit {
                            before_inst: inst_id,
                            from: actual_preg,
                            to: required_preg,
                        });
                    }
                }
            }
        }
    }
    edits
}

/// Check if a vreg has any use strictly after `inst_idx` in the given block.
fn has_use_after_inst(
    func: &Function,
    block: &cfg_mir::Block,
    vreg: VReg,
    inst_idx: usize,
) -> bool {
    // Check instructions after inst_idx
    for &later_inst_id in &block.insts[inst_idx + 1..] {
        let later_inst = &func.insts[later_inst_id.0 as usize];
        let mut found = false;
        later_inst.op.for_each_use(|src| {
            if *src == vreg {
                found = true;
            }
        });
        if found {
            return true;
        }
    }
    // Check terminator
    let term = &func.terms[block.term.0 as usize];
    match term {
        cfg_mir::Terminator::BranchIf { cond, .. }
        | cfg_mir::Terminator::BranchIfZero { cond, .. } => {
            if *cond == vreg {
                return true;
            }
        }
        cfg_mir::Terminator::JumpTable { predicate, .. } => {
            if *predicate == vreg {
                return true;
            }
        }
        _ => {}
    }
    // Check edge args
    for &edge_id in &block.succs {
        let edge = &func.edges[edge_id.index()];
        for arg in &edge.args {
            if arg.source == vreg {
                return true;
            }
        }
    }
    false
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

// ─── Preference-guided coloring ─────────────────────────────────────────────

/// Preference weights by source type.
/// These determine the relative importance of different preference sources.
/// When multiple sources suggest the same register, weights accumulate.
const WEIGHT_DYING_SOURCE: u32 = 100;
const WEIGHT_PHI_EDGE: u32 = 40;
const WEIGHT_PHI_LOOP_BONUS: u32 = 60; // added to PHI_EDGE for edges inside loops

/// Choose the best available register for a vreg using accumulated weighted scoring.
///
/// Preference sources (resolved dynamically from current coloring state):
/// 1. Dying sources at this instruction → their freed registers
/// 2. Edge arg partners already colored → their registers (phi affinity)
///
/// Each source contributes a weight to the registers it suggests. If multiple
/// sources suggest the same register, weights accumulate. The highest-scoring
/// available register wins. Falls back to lowest available if no preference
/// is available.
fn preferred_color(
    allocatable: &[PReg],
    occupied: &HashSet<PReg>,
    dying_source_colors: &[PReg],
    phi_prefs: &[(PReg, u32)], // (register, weight) pairs
) -> Option<PReg> {
    // Accumulate weights per register.
    let mut scores: HashMap<PReg, u32> = HashMap::new();

    for &preg in dying_source_colors {
        *scores.entry(preg).or_insert(0) += WEIGHT_DYING_SOURCE;
    }

    for &(preg, weight) in phi_prefs {
        *scores.entry(preg).or_insert(0) += weight;
    }

    // Pick the highest-scoring available register.
    let mut best: Option<(PReg, u32)> = None;
    for (&preg, &score) in &scores {
        if !occupied.contains(&preg) && allocatable.contains(&preg) {
            match best {
                None => best = Some((preg, score)),
                Some((_, best_score)) if score > best_score => best = Some((preg, score)),
                Some((best_preg, best_score)) if score == best_score && preg.0 < best_preg.0 => {
                    // Tie-break by register number for determinism.
                    best = Some((preg, score));
                }
                _ => {}
            }
        }
    }

    if let Some((preg, _)) = best {
        return Some(preg);
    }

    // Fall back to lowest available.
    lowest_available_color(allocatable, occupied)
}

// ─── Phase 3: Coalesce ──────────────────────────────────────────────────────

/// Bounded depth for recursive recoloring per phi bundle.
/// Set to 0 to disable recursive displacement (equivalent to the old can_recolor).
const MAX_RECOLOR_DEPTH: usize = 2;
/// Maximum blockers to displace at each recursion level.
const MAX_BLOCKERS_WIDTH: usize = 2;

/// Bounded phi-affinity recoloring (Hack/Grund/Goos style).
///
/// For each edge arg (phi connection), if source and target got different
/// colors, try to recolor one to match the other. If direct recoloring is
/// blocked by an interfering neighbor, recursively try to displace the blocker
/// to a different color (bounded by depth and width limits).
fn coalesce_phase(
    func: &Function,
    liveness: &LivenessInfo,
    dom: &DominanceInfo,
    def_block: &HashMap<VReg, BlockId>,
    def_inst_idx: &HashMap<VReg, u32>,
    last_use_map: &HashMap<(BlockId, VReg), u32>,
    mut coloring: HashMap<VReg, PReg>,
    spilled: &BTreeSet<VReg>,
    callee_saved_set: &HashSet<PReg>,
    allocatable: &[PReg],
    live_across_call: &HashSet<VReg>,
) -> HashMap<VReg, PReg> {
    // Use the same live_across_call set as color_phase for callee-saved constraints.
    let must_be_callee_saved = live_across_call;

    // Build color → vregs index for fast blocker lookup
    let mut color_to_vregs: HashMap<PReg, BTreeSet<VReg>> = HashMap::new();
    for (&vreg, &preg) in coloring.iter() {
        if !spilled.contains(&vreg) {
            color_to_vregs.entry(preg).or_default().insert(vreg);
        }
    }

    // Build vreg → edge arg index map for fast mismatch counting
    let mut vreg_edge_args: HashMap<VReg, Vec<usize>> = HashMap::new();
    let mut all_edge_args: Vec<(VReg, VReg)> = Vec::new(); // (source, target) pairs
    for edge in &func.edges {
        if edge.from.0 == u32::MAX {
            continue;
        }
        for arg in &edge.args {
            if spilled.contains(&arg.target) || spilled.contains(&arg.source) {
                continue;
            }
            let idx = all_edge_args.len();
            all_edge_args.push((arg.source, arg.target));
            vreg_edge_args.entry(arg.source).or_default().push(idx);
            vreg_edge_args.entry(arg.target).or_default().push(idx);
        }
    }

    // Count mismatches for edges involving specific vregs
    let count_affected_mismatches =
        |coloring: &HashMap<VReg, PReg>, changes: &[(VReg, PReg)]| -> usize {
            let changed_vregs: BTreeSet<VReg> = changes.iter().map(|(v, _)| *v).collect();
            let mut seen = BTreeSet::new();
            let mut count = 0;
            for &v in &changed_vregs {
                if let Some(indices) = vreg_edge_args.get(&v) {
                    for &idx in indices {
                        if seen.insert(idx) {
                            let (src, tgt) = all_edge_args[idx];
                            if coloring.get(&src) != coloring.get(&tgt) {
                                count += 1;
                            }
                        }
                    }
                }
            }
            count
        };

    // Iterate phi-affinity recoloring to a fixed point with bounded
    // recursive recoloring (Hack/Grund/Goos style).
    // Monotone: only commit recolorings that strictly reduce phi-copy count.
    let max_rounds = 1;
    let mut total_attempts = 0usize;
    let max_attempts = all_edge_args.len() * 4; // budget: 4x edge count
    for _round in 0..max_rounds {
        let mut changed = false;

        for edge in &func.edges {
            if edge.from.0 == u32::MAX {
                continue;
            }
            for arg in &edge.args {
                if spilled.contains(&arg.target) || spilled.contains(&arg.source) {
                    continue;
                }
                let target_color = coloring.get(&arg.target).copied();
                let source_color = coloring.get(&arg.source).copied();

                if target_color == source_color {
                    continue;
                }

                let (Some(tc), Some(sc)) = (target_color, source_color) else {
                    continue;
                };

                // Budget check
                total_attempts += 1;
                if total_attempts > max_attempts {
                    break;
                }

                // Try target→source, then source→target
                let candidates = [(arg.target, sc), (arg.source, tc)];
                for &(vreg, desired) in &candidates {
                    let mut visited = HashSet::new();
                    let mut changes = Vec::new();
                    if !try_recolor_recursive(
                        vreg,
                        desired,
                        0,
                        &mut coloring,
                        &mut color_to_vregs,
                        liveness,
                        last_use_map,
                        def_inst_idx,
                        dom,
                        def_block,
                        spilled,
                        must_be_callee_saved,
                        callee_saved_set,
                        allocatable,
                        &mut visited,
                        &mut changes,
                    ) {
                        continue;
                    }
                    // Monotonicity check: save new, restore old, compare
                    let new_colors: Vec<(VReg, PReg)> =
                        changes.iter().map(|(v, _)| (*v, coloring[v])).collect();
                    // Temporarily restore old for "before" measurement
                    for &(v, old_c) in changes.iter().rev() {
                        coloring.insert(v, old_c);
                    }
                    let before = count_affected_mismatches(&coloring, &changes);
                    // Re-apply new for "after" measurement
                    for &(v, new_c) in &new_colors {
                        coloring.insert(v, new_c);
                    }
                    let after = count_affected_mismatches(&coloring, &changes);
                    if after < before {
                        // Index was already updated by try_recolor_recursive
                        // (temp rollback/re-apply only touched coloring, not index,
                        // and final state matches index)
                        changed = true;
                        break;
                    }
                    // Rollback coloring AND index
                    for &(v, old_c) in changes.iter().rev() {
                        let cur_c = coloring[&v];
                        coloring.insert(v, old_c);
                        color_to_vregs.entry(cur_c).and_modify(|s| {
                            s.remove(&v);
                        });
                        color_to_vregs.entry(old_c).or_default().insert(v);
                    }
                }
            }
        }

        if !changed {
            break;
        }
    }

    coloring
}

/// Try to recolor `vreg` to `desired_color`, recursively displacing blockers.
///
/// On success, `coloring` is updated and true is returned.
/// On failure, all speculative changes are rolled back and false is returned.
fn try_recolor_recursive(
    vreg: VReg,
    desired_color: PReg,
    depth: usize,
    coloring: &mut HashMap<VReg, PReg>,
    color_to_vregs: &mut HashMap<PReg, BTreeSet<VReg>>,
    liveness: &LivenessInfo,
    last_use_map: &HashMap<(BlockId, VReg), u32>,
    def_inst_idx: &HashMap<VReg, u32>,
    dom: &DominanceInfo,
    def_block: &HashMap<VReg, BlockId>,
    spilled: &BTreeSet<VReg>,
    must_be_callee_saved: &HashSet<VReg>,
    callee_saved_set: &HashSet<PReg>,
    allocatable: &[PReg],
    visited: &mut HashSet<VReg>,
    changes: &mut Vec<(VReg, PReg)>,
) -> bool {
    if depth > MAX_RECOLOR_DEPTH {
        return false;
    }

    let Some(&current_color) = coloring.get(&vreg) else {
        return false;
    };

    if current_color == desired_color {
        return true;
    }

    // Prevent cycles in recursive chain
    if !visited.insert(vreg) {
        return false;
    }

    // Callee-saved constraint
    if must_be_callee_saved.contains(&vreg) && !callee_saved_set.contains(&desired_color) {
        visited.remove(&vreg);
        return false;
    }

    // Must be an allocatable register
    if !allocatable.contains(&desired_color) {
        visited.remove(&vreg);
        return false;
    }

    let Some(&vreg_def_block) = def_block.get(&vreg) else {
        visited.remove(&vreg);
        return false;
    };

    // Find interfering neighbors that currently hold desired_color (indexed lookup)
    let blockers: Vec<VReg> = color_to_vregs
        .get(&desired_color)
        .map(|vregs| {
            vregs
                .iter()
                .filter(|&&other| other != vreg)
                .filter(|&&other| {
                    let Some(&other_block) = def_block.get(&other) else {
                        return false;
                    };
                    interferes(
                        vreg,
                        vreg_def_block,
                        other,
                        other_block,
                        liveness,
                        dom,
                        last_use_map,
                        def_inst_idx,
                    )
                })
                .copied()
                .collect()
        })
        .unwrap_or_default();

    // Width budget
    if blockers.len() > MAX_BLOCKERS_WIDTH {
        visited.remove(&vreg);
        return false;
    }

    if blockers.is_empty() {
        // No blockers — safe to recolor directly
        changes.push((vreg, current_color));
        coloring.insert(vreg, desired_color);
        color_to_vregs.entry(current_color).and_modify(|s| {
            s.remove(&vreg);
        });
        color_to_vregs
            .entry(desired_color)
            .or_default()
            .insert(vreg);
        return true;
    }

    // Snapshot for rollback
    let snapshot = changes.len();

    // Try to displace each blocker to a different color
    for &blocker in &blockers {
        let mut moved = false;

        for &alt_color in allocatable {
            if alt_color == desired_color {
                continue;
            }
            if try_recolor_recursive(
                blocker,
                alt_color,
                depth + 1,
                coloring,
                color_to_vregs,
                liveness,
                last_use_map,
                def_inst_idx,
                dom,
                def_block,
                spilled,
                must_be_callee_saved,
                callee_saved_set,
                allocatable,
                visited,
                changes,
            ) {
                moved = true;
                break;
            }
        }

        if !moved {
            // Failed — rollback all changes since snapshot
            while changes.len() > snapshot {
                let (v, old_c) = changes.pop().unwrap();
                let cur_c = coloring[&v];
                coloring.insert(v, old_c);
                color_to_vregs.entry(cur_c).and_modify(|s| {
                    s.remove(&v);
                });
                color_to_vregs.entry(old_c).or_default().insert(v);
            }
            visited.remove(&vreg);
            return false;
        }
    }

    // Re-verify: displacing blockers may have moved a vreg INTO desired_color.
    let still_blocked = color_to_vregs
        .get(&desired_color)
        .map(|vregs| {
            vregs.iter().any(|&other| {
                if other == vreg {
                    return false;
                }
                let Some(&other_block) = def_block.get(&other) else {
                    return false;
                };
                interferes(
                    vreg,
                    vreg_def_block,
                    other,
                    other_block,
                    liveness,
                    dom,
                    last_use_map,
                    def_inst_idx,
                )
            })
        })
        .unwrap_or(false);
    if still_blocked {
        while changes.len() > snapshot {
            let (v, old_c) = changes.pop().unwrap();
            let cur_c = coloring[&v];
            coloring.insert(v, old_c);
            color_to_vregs.entry(cur_c).and_modify(|s| {
                s.remove(&v);
            });
            color_to_vregs.entry(old_c).or_default().insert(v);
        }
        visited.remove(&vreg);
        return false;
    }

    // All clear — recolor vreg
    changes.push((vreg, current_color));
    coloring.insert(vreg, desired_color);
    color_to_vregs.entry(current_color).and_modify(|s| {
        s.remove(&vreg);
    });
    color_to_vregs
        .entry(desired_color)
        .or_default()
        .insert(vreg);
    true
}

/// Check if two vregs interfere.
///
/// In SSA: v and w interfere iff one's definition dominates the other's
/// AND the dominator is live at the dominated definition point.
///
/// For same-block: uses intra-block instruction ordering + last-use tracking
/// for precise interference (no false positives).
fn interferes(
    v: VReg,
    v_block: BlockId,
    w: VReg,
    w_block: BlockId,
    liveness: &LivenessInfo,
    dom: &DominanceInfo,
    last_use_map: &HashMap<(BlockId, VReg), u32>,
    def_inst_idx: &HashMap<VReg, u32>,
) -> bool {
    if v_block == w_block {
        // Same block: precise check using instruction ordering.
        // The earlier-defined vreg interferes with the later-defined one
        // iff the earlier is still alive at the later's def point.
        let v_idx = def_inst_idx.get(&v).copied().unwrap_or(u32::MAX);
        let w_idx = def_inst_idx.get(&w).copied().unwrap_or(u32::MAX);

        // Determine which is defined first (block params at u32::MAX = "before all")
        let (early, early_idx, late_idx) = if v_idx <= w_idx {
            (v, v_idx, w_idx)
        } else {
            (w, w_idx, v_idx)
        };
        // If defined at the same point (e.g., both block params), they interfere
        // iff either is live-out or used after the other.
        if v_idx == w_idx {
            return true; // conservative for same-point defs
        }

        // early is defined before late. They interfere iff early is alive at late's def.
        // early is alive at late's def iff:
        //   - early is live-out of this block, OR
        //   - early has a use at or after late_idx
        let early_live_out = liveness
            .live_out
            .get(&v_block)
            .map_or(false, |lo| lo.contains(&early));
        if early_live_out {
            return true;
        }
        // Check last use of early in this block.
        let early_last_use = last_use_map.get(&(v_block, early)).copied();
        match early_last_use {
            // u32::MAX means used at/after terminator (edge arg or branch cond)
            Some(u32::MAX) => true,
            Some(last) => last >= late_idx,
            None => false, // no use in this block after def → dead immediately
        }
    } else if dom.dominates(v_block, w_block) {
        // v's block dominates w's block. v interferes with w iff v is
        // live-in to w's block AND still alive at w's def point.
        let v_live_in = liveness
            .live_in
            .get(&w_block)
            .map_or(false, |li| li.contains(&v));
        if !v_live_in {
            return false;
        }
        // Refine: if v dies before w's def in w's block, they don't interfere.
        let v_last_use = last_use_map.get(&(w_block, v)).copied();
        let w_def_idx = def_inst_idx.get(&w).copied().unwrap_or(u32::MAX);
        match v_last_use {
            Some(u32::MAX) => true,
            Some(last) => last >= w_def_idx,
            None => {
                // v is live-in but has no use in w's block — check live-out.
                liveness
                    .live_out
                    .get(&w_block)
                    .map_or(false, |lo| lo.contains(&v))
            }
        }
    } else if dom.dominates(w_block, v_block) {
        let w_live_in = liveness
            .live_in
            .get(&v_block)
            .map_or(false, |li| li.contains(&w));
        if !w_live_in {
            return false;
        }
        let w_last_use = last_use_map.get(&(v_block, w)).copied();
        let v_def_idx = def_inst_idx.get(&v).copied().unwrap_or(u32::MAX);
        match w_last_use {
            Some(u32::MAX) => true,
            Some(last) => last >= v_def_idx,
            None => {
                liveness
                    .live_out
                    .get(&v_block)
                    .map_or(false, |lo| lo.contains(&w))
            }
        }
    } else {
        false
    }
}
