//! Structure-aware hierarchical CFG-MIR reducer.
//!
//! Given a CFG-MIR program and a predicate ("is this interesting?"), produces
//! the smallest program that still satisfies the predicate. Inspired by
//! mlir-reduce / C-Reduce: many domain-specific transforms run to a fixed point.
//!
//! The reduction follows a hierarchical pass schedule:
//! 1. CFG shape: simplify terminators, remove unreachable regions
//! 2. Block sets: ddmin over groups of blocks
//! 3. Block signatures: reduce params/edge args
//! 4. Instructions: ddmin within blocks
//! 5. Operands: replace uses with constants/nearby dominators
//! 6. Fixed-point: repeat until no pass reduces further

use std::collections::HashMap;

use kajit_ir::VReg;
use kajit_lir::LinearOp;

use crate::cfg_mir::{
    BlockId, EdgeId, Function, Inst, InstId, Operand, OperandKind, Program, RegClass, TermId,
    Terminator,
};

/// Result of a reduction run.
#[derive(Debug)]
pub struct ReduceResult {
    /// The minimized program.
    pub program: Program,
    /// How many reduction steps were applied.
    pub steps_applied: usize,
    /// How many candidates were tested.
    pub candidates_tested: usize,
}

/// A predicate that tests whether a program is "interesting" (triggers the bug).
pub type Predicate = Box<dyn Fn(&Program) -> bool>;

/// Run the full hierarchical reducer on a program.
pub fn reduce(program: &Program, is_interesting: &dyn Fn(&Program) -> bool) -> ReduceResult {
    let mut current = program.clone();
    let mut total_steps = 0;
    let mut total_candidates = 0;

    // Fixed-point loop: keep cycling passes until nothing reduces
    loop {
        let size_before = program_size(&current);
        let mut any_changed = false;

        // Layer 1: CFG shape — simplify terminators, remove unreachable
        let (changed, candidates) = pass_simplify_terminators(&mut current, is_interesting);
        total_candidates += candidates;
        if changed {
            total_steps += 1;
            any_changed = true;
        }

        // Layer 2: Block deletion with ddmin
        let (changed, candidates) = pass_delete_blocks(&mut current, is_interesting);
        total_candidates += candidates;
        if changed {
            total_steps += 1;
            any_changed = true;
        }

        // Layer 3: Block signature reduction (remove params)
        let (changed, candidates) = pass_reduce_block_params(&mut current, is_interesting);
        total_candidates += candidates;
        if changed {
            total_steps += 1;
            any_changed = true;
        }

        // Layer 4: Instruction deletion
        let (changed, candidates) = pass_delete_instructions(&mut current, is_interesting);
        total_candidates += candidates;
        if changed {
            total_steps += 1;
            any_changed = true;
        }

        // Layer 5: Operand simplification
        let (changed, candidates) = pass_simplify_operands(&mut current, is_interesting);
        total_candidates += candidates;
        if changed {
            total_steps += 1;
            any_changed = true;
        }

        let size_after = program_size(&current);
        eprintln!(
            "[reduce] round complete: {} → {} (Δ={}), changed={}",
            size_before,
            size_after,
            size_before as i64 - size_after as i64,
            any_changed
        );

        if !any_changed {
            break;
        }
    }

    ReduceResult {
        program: current,
        steps_applied: total_steps,
        candidates_tested: total_candidates,
    }
}

/// Rough size metric: blocks + instructions + edges + params.
fn program_size(p: &Program) -> usize {
    let mut n = 0;
    for func in &p.funcs {
        for block in &func.blocks {
            if block.dead {
                continue;
            }
            n += 1; // block
            n += block.insts.len(); // instructions
            n += block.params.len(); // params
        }
        n += func.edges.len(); // edges
    }
    n
}

// ─── Layer 1: Simplify terminators ──────────────────────────────────────────

/// Try converting BranchIf/BranchIfZero → Branch (pick each arm).
/// After each simplification, remove unreachable blocks.
fn pass_simplify_terminators(
    program: &mut Program,
    is_interesting: &dyn Fn(&Program) -> bool,
) -> (bool, usize) {
    let mut any_changed = false;
    let mut candidates = 0;

    for func_idx in 0..program.funcs.len() {
        // Collect blocks with conditional terminators
        let conditional_blocks: Vec<(BlockId, TermId)> = program.funcs[func_idx]
            .blocks
            .iter()
            .filter(|b| !b.dead)
            .filter_map(|b| {
                let term = &program.funcs[func_idx].terms[b.term.index()];
                match term {
                    Terminator::BranchIf { .. }
                    | Terminator::BranchIfZero { .. }
                    | Terminator::JumpTable { .. } => Some((b.id, b.term)),
                    _ => None,
                }
            })
            .collect();

        for (block_id, term_id) in conditional_blocks {
            let term = &program.funcs[func_idx].terms[term_id.index()];
            let edges_to_try: Vec<EdgeId> = match term {
                Terminator::BranchIf {
                    taken, fallthrough, ..
                }
                | Terminator::BranchIfZero {
                    taken, fallthrough, ..
                } => vec![*taken, *fallthrough],
                Terminator::JumpTable {
                    targets, default, ..
                } => {
                    let mut edges = targets.clone();
                    edges.push(*default);
                    edges
                }
                _ => continue,
            };

            for &kept_edge in &edges_to_try {
                let mut candidate = program.clone();
                let func = &mut candidate.funcs[func_idx];

                // Replace terminator with unconditional branch
                func.terms[term_id.index()] = Terminator::Branch { edge: kept_edge };

                // Update block succs: keep only the kept edge
                let block = &mut func.blocks[block_id.index()];
                let dead_edges: Vec<EdgeId> = block
                    .succs
                    .iter()
                    .copied()
                    .filter(|e| *e != kept_edge)
                    .collect();
                block.succs.retain(|e| *e == kept_edge);

                // Remove dead edges from successor preds
                for dead_edge in &dead_edges {
                    let target = func.edges[dead_edge.index()].to;
                    func.blocks[target.index()].preds.retain(|e| e != dead_edge);
                }

                // Repair: remove unreachable blocks
                repair_remove_unreachable(func);

                candidates += 1;
                if is_valid_and_interesting(&candidate, is_interesting) {
                    eprintln!(
                        "[reduce] simplified terminator in b{} → branch to e{}",
                        block_id.index(),
                        kept_edge.index()
                    );
                    *program = candidate;
                    any_changed = true;
                    break; // restart from this block
                }
            }
        }
    }

    (any_changed, candidates)
}

// ─── Layer 2: Delete blocks (ddmin-style) ───────────────────────────────────

/// Try removing sets of blocks using delta debugging.
fn pass_delete_blocks(
    program: &mut Program,
    is_interesting: &dyn Fn(&Program) -> bool,
) -> (bool, usize) {
    let mut any_changed = false;
    let mut candidates = 0;

    for func_idx in 0..program.funcs.len() {
        let live_blocks: Vec<BlockId> = program.funcs[func_idx]
            .blocks
            .iter()
            .filter(|b| !b.dead && b.id != program.funcs[func_idx].entry)
            .map(|b| b.id)
            .collect();

        // Try removing individual blocks (simplest ddmin: granularity 1)
        for &block_id in &live_blocks {
            let mut candidate = program.clone();
            let func = &mut candidate.funcs[func_idx];

            // Redirect all predecessors of this block to its single successor (if any)
            let block = &func.blocks[block_id.index()];
            if block.succs.len() != 1 {
                // Can only redirect if single successor
                // Try just marking it dead and cleaning up
                let block_mut = &mut func.blocks[block_id.index()];
                block_mut.dead = true;

                // Remove this block from successors' pred lists
                let succs: Vec<EdgeId> = block_mut.succs.clone();
                block_mut.succs.clear();
                for edge_id in &succs {
                    let target = func.edges[edge_id.index()].to;
                    func.blocks[target.index()]
                        .preds
                        .retain(|e| func.edges[e.index()].from != block_id);
                }

                // Remove this block from predecessors' succ lists
                let preds: Vec<EdgeId> = func.blocks[block_id.index()].preds.clone();
                func.blocks[block_id.index()].preds.clear();
                for edge_id in &preds {
                    let source = func.edges[edge_id.index()].from;
                    func.blocks[source.index()]
                        .succs
                        .retain(|e| func.edges[e.index()].to != block_id);
                }

                repair_remove_unreachable(func);
            } else {
                // Single successor: redirect predecessors
                let succ_edge = block.succs[0];
                let succ_block = func.edges[succ_edge.index()].to;

                // For each predecessor edge, retarget to the successor
                let pred_edges: Vec<EdgeId> = block.preds.clone();
                for &pred_edge_id in &pred_edges {
                    func.edges[pred_edge_id.index()].to = succ_block;
                    // Clear edge args (can't compose params through, just drop them)
                    func.edges[pred_edge_id.index()].args.clear();

                    // Update successor's preds: add this edge
                    if !func.blocks[succ_block.index()]
                        .preds
                        .contains(&pred_edge_id)
                    {
                        func.blocks[succ_block.index()].preds.push(pred_edge_id);
                    }
                }

                // Remove original edge from successor's preds
                func.blocks[succ_block.index()]
                    .preds
                    .retain(|e| *e != succ_edge);

                // Mark block dead
                func.blocks[block_id.index()].dead = true;
                func.blocks[block_id.index()].preds.clear();
                func.blocks[block_id.index()].succs.clear();

                repair_remove_unreachable(func);
            }

            // Clear block params on successor if we dropped edge args
            repair_dangling_params(func);

            candidates += 1;
            if is_valid_and_interesting(&candidate, is_interesting) {
                eprintln!("[reduce] deleted block b{}", block_id.index());
                *program = candidate;
                any_changed = true;
                // Don't break — keep trying more blocks
            }
        }
    }

    (any_changed, candidates)
}

// ─── Layer 3: Reduce block params ───────────────────────────────────────────

/// Try removing individual block parameters (and corresponding edge args).
fn pass_reduce_block_params(
    program: &mut Program,
    is_interesting: &dyn Fn(&Program) -> bool,
) -> (bool, usize) {
    let mut any_changed = false;
    let mut candidates = 0;

    for func_idx in 0..program.funcs.len() {
        // Collect all (block, param_index) pairs
        let params_to_try: Vec<(BlockId, usize)> = program.funcs[func_idx]
            .blocks
            .iter()
            .filter(|b| !b.dead)
            .flat_map(|b| (0..b.params.len()).rev().map(move |i| (b.id, i)))
            .collect();

        for (block_id, param_idx) in params_to_try {
            let func = &program.funcs[func_idx];
            // Skip if param_idx is now out of bounds (earlier reductions may have shrunk it)
            if param_idx >= func.blocks[block_id.index()].params.len() {
                continue;
            }

            let param_vreg = func.blocks[block_id.index()].params[param_idx];

            let mut candidate = program.clone();
            let func = &mut candidate.funcs[func_idx];

            // Find a replacement for uses of this param: try entry-defined const 0
            let replacement = find_or_create_const_zero(func);

            // Replace all uses of the param vreg with the replacement
            let mut replacements = HashMap::new();
            replacements.insert(param_vreg, replacement);
            crate::opt::constant_phi_elim::replace_vregs_in_function(func, &replacements);

            // Remove param from block
            func.blocks[block_id.index()].params.remove(param_idx);

            // Remove corresponding arg from all incoming edges
            let pred_edges: Vec<EdgeId> = func.blocks[block_id.index()].preds.clone();
            for &edge_id in &pred_edges {
                let edge = &mut func.edges[edge_id.index()];
                if param_idx < edge.args.len() {
                    edge.args.remove(param_idx);
                }
            }

            candidates += 1;
            if is_valid_and_interesting(&candidate, is_interesting) {
                eprintln!(
                    "[reduce] removed param {} (v{}) from b{}",
                    param_idx,
                    param_vreg.index(),
                    block_id.index()
                );
                *program = candidate;
                any_changed = true;
            }
        }
    }

    (any_changed, candidates)
}

// ─── Layer 4: Delete instructions ───────────────────────────────────────────

/// Try deleting individual instructions. If an instruction defines a vreg,
/// replace uses with const 0.
fn pass_delete_instructions(
    program: &mut Program,
    is_interesting: &dyn Fn(&Program) -> bool,
) -> (bool, usize) {
    let mut any_changed = false;
    let mut candidates = 0;

    for func_idx in 0..program.funcs.len() {
        // Collect all (block, inst_position) pairs
        let insts_to_try: Vec<(BlockId, usize)> = program.funcs[func_idx]
            .blocks
            .iter()
            .filter(|b| !b.dead)
            .flat_map(|b| (0..b.insts.len()).rev().map(move |i| (b.id, i)))
            .collect();

        for (block_id, inst_pos) in insts_to_try {
            let func = &program.funcs[func_idx];
            if inst_pos >= func.blocks[block_id.index()].insts.len() {
                continue;
            }

            let inst_id = func.blocks[block_id.index()].insts[inst_pos];
            let inst = &func.insts[inst_id.index()];

            // Find dst vreg if any
            let dst_vreg = inst.op.dst();

            let mut candidate = program.clone();
            let func = &mut candidate.funcs[func_idx];

            // If instruction defines a vreg, replace all its uses with const 0
            if let Some(dst) = dst_vreg {
                let replacement = find_or_create_const_zero(func);
                let mut replacements = HashMap::new();
                replacements.insert(dst, replacement);
                crate::opt::constant_phi_elim::replace_vregs_in_function(func, &replacements);
            }

            // Remove instruction from block
            func.blocks[block_id.index()].insts.remove(inst_pos);

            candidates += 1;
            if is_valid_and_interesting(&candidate, is_interesting) {
                eprintln!(
                    "[reduce] deleted inst i{} from b{}",
                    inst_id.index(),
                    block_id.index()
                );
                *program = candidate;
                any_changed = true;
            }
        }
    }

    (any_changed, candidates)
}

// ─── Layer 5: Simplify operands ─────────────────────────────────────────────

/// Try replacing edge arg sources with const 0.
fn pass_simplify_operands(
    program: &mut Program,
    is_interesting: &dyn Fn(&Program) -> bool,
) -> (bool, usize) {
    let mut any_changed = false;
    let mut candidates = 0;

    for func_idx in 0..program.funcs.len() {
        // Try simplifying edge args: replace source with const zero
        let edges_and_args: Vec<(EdgeId, usize)> = program.funcs[func_idx]
            .edges
            .iter()
            .flat_map(|e| (0..e.args.len()).map(move |i| (e.id, i)))
            .collect();

        for (edge_id, arg_idx) in edges_and_args {
            if arg_idx >= program.funcs[func_idx].edges[edge_id.index()].args.len() {
                continue;
            }

            let mut candidate = program.clone();
            let func = &mut candidate.funcs[func_idx];
            let const_zero = find_or_create_const_zero(func);

            let edge = &mut func.edges[edge_id.index()];
            let old_source = edge.args[arg_idx].source;
            if old_source == const_zero {
                continue; // already zero
            }
            edge.args[arg_idx].source = const_zero;

            candidates += 1;
            if is_valid_and_interesting(&candidate, is_interesting) {
                eprintln!(
                    "[reduce] simplified edge e{} arg {} source v{} → v{}",
                    edge_id.index(),
                    arg_idx,
                    old_source.index(),
                    const_zero.index()
                );
                *program = candidate;
                any_changed = true;
            }
        }
    }

    (any_changed, candidates)
}

// ─── Repair utilities ───────────────────────────────────────────────────────

/// Remove blocks that are unreachable from entry (no predecessors, not entry).
fn repair_remove_unreachable(func: &mut Function) {
    let mut changed = true;
    while changed {
        changed = false;
        let unreachable: Vec<BlockId> = func
            .blocks
            .iter()
            .filter(|b| !b.dead && b.id != func.entry && b.preds.is_empty())
            .map(|b| b.id)
            .collect();

        for block_id in unreachable {
            let block = &mut func.blocks[block_id.index()];
            block.dead = true;
            let succs: Vec<EdgeId> = block.succs.clone();
            block.succs.clear();
            block.preds.clear();

            for edge_id in succs {
                let target = func.edges[edge_id.index()].to;
                if target.index() < func.blocks.len() {
                    func.blocks[target.index()].preds.retain(|e| *e != edge_id);
                }
            }
            changed = true;
        }
    }
}

/// Remove block params that have no incoming edge args (because edges were cleared).
fn repair_dangling_params(func: &mut Function) {
    for block_idx in 0..func.blocks.len() {
        if func.blocks[block_idx].dead {
            continue;
        }
        let _block_id = func.blocks[block_idx].id;

        // Check if all incoming edges have the right arg count
        let expected_params = func.blocks[block_idx].params.len();
        if expected_params == 0 {
            continue;
        }

        let preds: Vec<EdgeId> = func.blocks[block_idx].preds.clone();
        let all_edges_match = preds
            .iter()
            .all(|&e| func.edges[e.index()].args.len() == expected_params);

        if !all_edges_match {
            // Some edges don't provide all args — clear all params
            func.blocks[block_idx].params.clear();
            for &e in &preds {
                func.edges[e.index()].args.clear();
            }
        }
    }
}

/// Find or create a vreg holding const 0 in the entry block.
fn find_or_create_const_zero(func: &mut Function) -> VReg {
    let entry = func.entry;

    // Look for an existing const 0 in the entry block
    for &inst_id in &func.blocks[entry.index()].insts {
        let inst = &func.insts[inst_id.index()];
        if let LinearOp::Const { dst, value: 0 } = &inst.op {
            return *dst;
        }
    }

    // Create a new const 0 instruction in the entry block
    let new_vreg = VReg::new(func.insts.len() as u32 + 10000); // high number to avoid collisions
    let new_inst_id = InstId::new(func.insts.len() as u32);
    func.insts.push(Inst {
        id: new_inst_id,
        op: LinearOp::Const {
            dst: new_vreg,
            value: 0,
        },
        operands: vec![Operand {
            vreg: new_vreg,
            kind: OperandKind::Def,
            class: RegClass::Gpr,
            fixed: None,
        }],
        clobbers: crate::cfg_mir::Clobbers::default(),
    });

    // Insert at the beginning of entry block
    func.blocks[entry.index()].insts.insert(0, new_inst_id);
    new_vreg
}

/// Check that a program is structurally valid (CFG + SSA) and interesting.
fn is_valid_and_interesting(program: &Program, is_interesting: &dyn Fn(&Program) -> bool) -> bool {
    // First check structural validity
    for func in &program.funcs {
        if crate::opt::validate::validate_cfg(func).is_err() {
            return false;
        }
        if crate::opt::validate_ssa::validate_ssa(func).is_err() {
            return false;
        }
    }

    // Then check the predicate
    is_interesting(program)
}

/// Run a named optimization pass on a program.
pub fn run_named_pass(prog: &mut Program, pass_name: &str) -> bool {
    match pass_name {
        "const_phi_elim" => {
            let mut changed = false;
            for func in &mut prog.funcs {
                changed |= crate::opt::constant_phi_elim::eliminate_constant_phis(func);
            }
            changed
        }
        "const_branch_fold" => {
            let mut changed = false;
            for func in &mut prog.funcs {
                changed |= crate::opt::const_branch_fold::fold_const_branches(func) > 0;
            }
            changed
        }
        "merge_blocks" => {
            let mut changed = false;
            for func in &mut prog.funcs {
                changed |= crate::opt::block_merge::merge_empty_blocks(func);
            }
            changed
        }
        "remat" => {
            crate::cfg_mir::rematerialize_constants(prog);
            true
        }
        "cse" => {
            crate::cfg_mir::local_cse(prog);
            true
        }
        "local_cse" => {
            crate::cfg_mir::local_common_subexpr_elim(prog);
            true
        }
        "copyprop" => {
            crate::cfg_mir::copy_propagation(prog);
            true
        }
        "fuse_cmpz" => {
            crate::cfg_mir::fuse_compare_zero_branch(prog);
            true
        }
        "elim_imm" => {
            crate::cfg_mir::eliminate_immediate_only_const_defs(prog);
            true
        }
        "dce" => {
            crate::cfg_mir::dead_code_elimination(prog);
            true
        }
        other => {
            eprintln!("[reduce] unknown pass: {}", other);
            false
        }
    }
}

/// List of all known pass names.
pub const ALL_PASS_NAMES: &[&str] = &[
    "const_phi_elim",
    "const_branch_fold",
    "merge_blocks",
    "remat",
    "cse",
    "local_cse",
    "copyprop",
    "fuse_cmpz",
    "elim_imm",
    "dce",
];

/// Run a sequence of passes, validating SSA after each one.
/// Returns the name of the first pass that breaks SSA, or None.
pub fn find_breaking_pass(program: &Program, passes: &[&str]) -> Option<String> {
    let mut prog = program.clone();
    for &pass in passes {
        run_named_pass(&mut prog, pass);
        for func in &prog.funcs {
            if crate::opt::validate_ssa::validate_ssa(func).is_err() {
                return Some(pass.to_string());
            }
        }
    }
    None
}

/// Check if a pass sequence breaks SSA (any pass in the sequence, not just the last).
pub fn sequence_breaks_ssa(program: &Program, passes: &[&str]) -> bool {
    find_breaking_pass(program, passes).is_some()
}

/// Bisect a pass sequence to find the minimal ordered subsequence that breaks SSA.
///
/// Returns the minimal set of passes (in order) such that running them
/// sequentially on the program produces an SSA violation.
pub fn bisect_pass_sequence(program: &Program, all_passes: &[&str]) -> Vec<String> {
    // First verify the full sequence breaks SSA
    if !sequence_breaks_ssa(program, all_passes) {
        return Vec::new();
    }

    eprintln!(
        "[bisect] full sequence ({} passes) breaks SSA, bisecting...",
        all_passes.len()
    );

    // Use delta debugging: try removing passes and see if it still breaks
    let mut required: Vec<&str> = all_passes.to_vec();
    let mut granularity = required.len() / 2;

    while granularity >= 1 {
        let mut i = 0;
        let mut any_removed = false;

        while i < required.len() {
            let end = (i + granularity).min(required.len());
            // Try without this chunk
            let mut candidate: Vec<&str> = Vec::new();
            candidate.extend_from_slice(&required[..i]);
            candidate.extend_from_slice(&required[end..]);

            if !candidate.is_empty() && sequence_breaks_ssa(program, &candidate) {
                eprintln!(
                    "[bisect] removed passes [{}..{}]: {:?}",
                    i,
                    end,
                    &required[i..end]
                );
                required = candidate;
                any_removed = true;
                // Don't advance i — the next chunk is now at position i
            } else {
                i += granularity;
            }
        }

        if !any_removed {
            granularity /= 2;
        }
    }

    // Find which pass in the sequence actually produces the SSA error
    let breaking = find_breaking_pass(program, &required);

    eprintln!(
        "[bisect] minimal sequence: {:?} (breaks at: {:?})",
        required, breaking
    );

    required.into_iter().map(|s| s.to_string()).collect()
}

/// Create a predicate that checks if a pass sequence breaks SSA.
pub fn predicate_sequence_breaks_ssa(passes: &[String]) -> Box<dyn Fn(&Program) -> bool> {
    let passes: Vec<String> = passes.to_vec();
    Box::new(move |program: &Program| {
        let pass_refs: Vec<&str> = passes.iter().map(|s| s.as_str()).collect();
        sequence_breaks_ssa(program, &pass_refs)
    })
}

/// Helper: get the specific SSA errors after running a pass sequence.
pub fn get_sequence_ssa_errors(
    program: &Program,
    passes: &[&str],
) -> Option<(String, Vec<crate::opt::validate_ssa::SsaError>)> {
    let mut prog = program.clone();
    for &pass in passes {
        run_named_pass(&mut prog, pass);
        for func in &prog.funcs {
            if let Err(errors) = crate::opt::validate_ssa::validate_ssa(func) {
                return Some((pass.to_string(), errors));
            }
        }
    }
    None
}

/// Outcome of running the ideal interpreter.
#[derive(Debug, Clone)]
pub enum InterpOutcome {
    /// Program returned normally.
    Returned(Vec<u8>),
    /// Program hit a trap (bounds check, unexpected EOF, etc.)
    Trapped(crate::InterpreterTrap),
    /// Interpreter did not finish within the step budget.
    TimedOut,
}

impl InterpOutcome {
    /// Returns the output bytes if the program returned normally.
    pub fn returned(&self) -> Option<&Vec<u8>> {
        match self {
            Self::Returned(out) => Some(out),
            _ => None,
        }
    }
}

/// Run a program through the ideal interpreter.
pub fn interpret(program: &Program, input: &[u8]) -> InterpOutcome {
    let mut session = match crate::DebuggerSession::new(program, input) {
        Ok(s) => s,
        Err(_) => return InterpOutcome::TimedOut,
    };

    // If the program has intrinsic calls, set real base addresses so pointer
    // values passed to intrinsics are dereferenceable.
    let has_intrinsics = program.funcs.iter().any(|f| {
        f.insts.iter().any(|inst| {
            matches!(
                inst.op,
                kajit_lir::LinearOp::CallIntrinsic { .. }
                    | kajit_lir::LinearOp::CallPure { .. }
                    | kajit_lir::LinearOp::CallEffect { .. }
            )
        })
    });
    if has_intrinsics {
        session.set_real_addresses();
    }

    let max_steps = std::env::var("KAJIT_INTERP_STEPS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(1_000_000);
    let trace = std::env::var("KAJIT_INTERP_TRACE").is_ok();

    for step in 0..max_steps {
        let Ok(event) = session.step_forward() else {
            return InterpOutcome::TimedOut;
        };
        if trace {
            let st = session.state();
            eprintln!(
                "[interp] step {step}: block b{} inst#{} term={} event={event:?}",
                st.location.block.0, st.location.next_inst_index, st.location.at_terminator
            );
        }
        if session.state().returned || session.state().trap.is_some() {
            break;
        }
    }

    let state = session.state();
    if trace && !state.returned && state.trap.is_none() {
        eprintln!("[interp] did NOT return after {max_steps} steps");
    }
    if state.returned {
        InterpOutcome::Returned(state.output.clone())
    } else if let Some(trap) = &state.trap {
        InterpOutcome::Trapped(trap.clone())
    } else {
        InterpOutcome::TimedOut
    }
}

/// Run a program through the ideal interpreter after applying a pass sequence.
pub fn interpret_after_passes(program: &Program, passes: &[&str], input: &[u8]) -> InterpOutcome {
    let mut prog = program.clone();
    for &pass in passes {
        run_named_pass(&mut prog, pass);
    }
    interpret(&prog, input)
}

/// Create a predicate: "does applying these passes change the interpreter output?"
///
/// The predicate captures the "correct" output by interpreting the unoptimized
/// program, then checks if the optimized program produces different output.
pub fn predicate_wrong_output(
    passes: Vec<String>,
    input: Vec<u8>,
    expected_output: Vec<u8>,
) -> Box<dyn Fn(&Program) -> bool> {
    Box::new(move |program: &Program| {
        // First: the unoptimized program must produce the expected output
        // (otherwise the reduction made the base program wrong, not interesting)
        let base_output = interpret(program, &input);
        if base_output.returned() != Some(&expected_output) {
            return false; // base program is itself broken, skip
        }

        // Then: apply passes and check if output changes
        let pass_refs: Vec<&str> = passes.iter().map(|s| s.as_str()).collect();
        let opt_output = interpret_after_passes(program, &pass_refs, &input);
        opt_output.returned() != Some(&expected_output)
    })
}

/// Bisect which passes cause wrong output (not just SSA breakage).
pub fn bisect_wrong_output(
    program: &Program,
    all_passes: &[&str],
    input: &[u8],
    expected_output: &[u8],
) -> Vec<String> {
    // Verify the full sequence produces wrong output
    let opt_output = interpret_after_passes(program, all_passes, input);
    if opt_output.returned() == Some(&expected_output.to_vec()) {
        return Vec::new(); // all passes produce correct output
    }

    eprintln!(
        "[bisect] full sequence ({} passes) produces wrong output, bisecting...",
        all_passes.len()
    );
    match &opt_output {
        InterpOutcome::Returned(out) => {
            eprintln!(
                "[bisect] expected {} bytes, got {} bytes",
                expected_output.len(),
                out.len()
            );
        }
        InterpOutcome::Trapped(trap) => {
            eprintln!(
                "[bisect] optimized program trapped: {:?} at offset {}",
                trap.code, trap.offset
            );
        }
        InterpOutcome::TimedOut => {
            eprintln!("[bisect] optimized program timed out");
        }
    }

    // Delta debugging on the pass list
    let mut required: Vec<&str> = all_passes.to_vec();
    let mut granularity = required.len() / 2;

    while granularity >= 1 {
        let mut i = 0;
        let mut any_removed = false;

        while i < required.len() {
            let end = (i + granularity).min(required.len());
            let mut candidate: Vec<&str> = Vec::new();
            candidate.extend_from_slice(&required[..i]);
            candidate.extend_from_slice(&required[end..]);

            if !candidate.is_empty() {
                let out = interpret_after_passes(program, &candidate, input);
                if out.returned() != Some(&expected_output.to_vec()) {
                    eprintln!(
                        "[bisect] removed passes [{}..{}]: {:?}",
                        i,
                        end,
                        &required[i..end]
                    );
                    required = candidate;
                    any_removed = true;
                } else {
                    i += granularity;
                }
            } else {
                i += granularity;
            }
        }

        if !any_removed {
            granularity /= 2;
        }
    }

    eprintln!("[bisect] minimal sequence: {:?}", required);
    required.into_iter().map(|s| s.to_string()).collect()
}

/// Trait for LinearOp destination extraction (needed for instruction deletion).
trait LinearOpDst {
    fn dst(&self) -> Option<VReg>;
}

impl LinearOpDst for LinearOp {
    fn dst(&self) -> Option<VReg> {
        match self {
            Self::Copy { dst, .. }
            | Self::Const { dst, .. }
            | Self::DataAddr { dst, .. }
            | Self::BinOp { dst, .. }
            | Self::UnaryOp { dst, .. }
            | Self::LoadFromAddr { dst, .. }
            | Self::SaveCursor { dst, .. }
            | Self::SaveInputEnd { dst, .. }
            | Self::PeekByte { dst, .. }
            | Self::ReadBytes { dst, .. }
            | Self::ReadFromField { dst, .. }
            | Self::SlotAddr { dst, .. }
            | Self::SaveOutPtr { dst, .. }
            | Self::ReadFromSlot { dst, .. }
            | Self::CallPure { dst, .. }
            | Self::CallEffect { dst, .. } => Some(*dst),
            Self::CallIntrinsic { dst, .. } => *dst,
            Self::CallLambda { .. } => None,
            Self::SimdStringScan { .. } => None,
            Self::Label(_)
            | Self::Branch { .. }
            | Self::BranchIf { .. }
            | Self::BranchIfZero { .. }
            | Self::JumpTable { .. }
            | Self::ErrorExit { .. } => None,
            Self::FuncStart { .. } | Self::FuncEnd => None,
            Self::StoreToAddr { .. }
            | Self::WriteToSlot { .. }
            | Self::WriteToField { .. }
            | Self::RestoreCursor { .. }
            | Self::AdvanceCursorBy { .. }
            | Self::SetOutPtr { .. }
            | Self::BoundsCheck { .. }
            | Self::AdvanceCursor { .. }
            | Self::SimdWhitespaceSkip => None,
        }
    }
}
