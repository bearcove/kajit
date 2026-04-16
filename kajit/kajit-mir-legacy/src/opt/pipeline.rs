//! MIR optimization helpers that operate over whole programs.
//!
//! These passes were previously embedded in the canonical MIR definition.
//! They belong in the (new) `kajit-mir`, not in `kajit-reprs`, even if some of
//! them still need finer-grained re-homing later.
//!
//! For now they live in `kajit-mir-legacy` as a reference implementation.

use std::collections::{HashMap, HashSet};

use kajit_ir::{FnPtr, SlotId, VReg};
use kajit_lir::LinearOp;
use kajit_reprs::mir::{
    BlockId, Clobbers, FixedReg, Function, Inst, InstId, Operand, OperandKind, Program, RegClass,
    Terminator,
};

pub fn fuse_compare_zero_branch(program: &mut Program) {
    for func in &mut program.funcs {
        fuse_compare_zero_branch_in_function(func);
    }
}

fn fuse_compare_zero_branch_in_function(func: &mut Function) {
    use kajit_lir::BinOpKind;

    // Step 1: Build map of const vreg -> value
    let mut const_values: HashMap<VReg, u64> = HashMap::new();
    for inst in &func.insts {
        if let LinearOp::Const { dst, value } = &inst.op {
            const_values.insert(*dst, *value);
        }
    }

    // Step 2: Find compare-with-zero instructions
    // Map: result vreg -> (comparison_kind, non_zero_operand)
    #[derive(Clone, Copy)]
    struct CompareZeroInfo {
        kind: BinOpKind, // CmpEq or CmpNe
        operand: VReg,   // the non-zero operand
    }

    let mut compare_zero_map: HashMap<VReg, CompareZeroInfo> = HashMap::new();

    for inst in &func.insts {
        if let LinearOp::BinOp { op, dst, lhs, rhs } = &inst.op
            && (*op == BinOpKind::CmpEq || *op == BinOpKind::CmpNe)
        {
            // Check if one operand is const(0)
            let lhs_is_zero = const_values.get(lhs) == Some(&0);
            let rhs_is_zero = const_values.get(rhs) == Some(&0);

            if lhs_is_zero && !rhs_is_zero {
                compare_zero_map.insert(
                    *dst,
                    CompareZeroInfo {
                        kind: *op,
                        operand: *rhs,
                    },
                );
            } else if rhs_is_zero && !lhs_is_zero {
                compare_zero_map.insert(
                    *dst,
                    CompareZeroInfo {
                        kind: *op,
                        operand: *lhs,
                    },
                );
            }
        }
    }

    if compare_zero_map.is_empty() {
        return;
    }

    // Step 3: Transform BranchIfZero/BranchIf terminators
    for term in &mut func.terms {
        match term {
            Terminator::BranchIfZero {
                cond,
                taken,
                fallthrough,
            } => {
                if let Some(info) = compare_zero_map.get(cond) {
                    // BranchIfZero on (v != 0) -> branch if v == 0 -> BranchIfZero v
                    // BranchIfZero on (v == 0) -> branch if v != 0 -> BranchIf v
                    match info.kind {
                        BinOpKind::CmpNe => {
                            // (v != 0) == 0 means v == 0, keep BranchIfZero
                            *cond = info.operand;
                        }
                        BinOpKind::CmpEq => {
                            // (v == 0) == 0 means v != 0, flip to BranchIf
                            *term = Terminator::BranchIf {
                                cond: info.operand,
                                taken: *taken,
                                fallthrough: *fallthrough,
                            };
                        }
                        _ => {}
                    }
                }
            }
            Terminator::BranchIf {
                cond,
                taken,
                fallthrough,
            } => {
                if let Some(info) = compare_zero_map.get(cond) {
                    // BranchIf on (v != 0) -> branch if v != 0 -> BranchIf v
                    // BranchIf on (v == 0) -> branch if v == 0 -> BranchIfZero v
                    match info.kind {
                        BinOpKind::CmpNe => {
                            // (v != 0) != 0 means v != 0, keep BranchIf
                            *cond = info.operand;
                        }
                        BinOpKind::CmpEq => {
                            // (v == 0) != 0 means v == 0, flip to BranchIfZero
                            *term = Terminator::BranchIfZero {
                                cond: info.operand,
                                taken: *taken,
                                fallthrough: *fallthrough,
                            };
                        }
                        _ => {}
                    }
                }
            }
            _ => {}
        }
    }
}

/// that needs them, eliminating the need to pass them through edges.
///
/// Benefits:
/// - Reduces register pressure (constants don't need to be kept live)
/// - Enables immediate encoding in backends (AND reg, #imm instead of AND reg, reg)
/// - Removes unnecessary edge argument traffic
/// - Keeps constants near their use sites for later lowering stages
pub fn rematerialize_constants(program: &mut Program) {
    // Build a map of VReg -> constant value for all constants in the program
    // Start with values from Const instructions
    let mut const_values: HashMap<VReg, u64> = HashMap::new();
    for func in &program.funcs {
        for inst in &func.insts {
            if let LinearOp::Const { dst, value } = &inst.op {
                const_values.insert(*dst, *value);
            }
        }
    }

    if const_values.is_empty() {
        return;
    }

    // Propagate const values through block params: if a block param receives
    // the same constant value from ALL incoming edges, it's also a constant.
    // This handles loop back edges that pass through constants unchanged.
    let mut changed = true;
    while changed {
        changed = false;
        for func in &program.funcs {
            for block in &func.blocks {
                if block.params.is_empty() || block.preds.is_empty() {
                    continue;
                }
                for (param_idx, &param_vreg) in block.params.iter().enumerate() {
                    // Skip if already known as const
                    if const_values.contains_key(&param_vreg) {
                        continue;
                    }
                    // Check if all incoming edges provide the same constant
                    let mut all_same_const: Option<u64> = None;
                    let mut is_uniform_const = true;
                    for &pred_edge_id in &block.preds {
                        let edge = &func.edges[pred_edge_id.index()];
                        if param_idx >= edge.args.len() {
                            is_uniform_const = false;
                            break;
                        }
                        let source_vreg = edge.args[param_idx].source;
                        if let Some(&const_val) = const_values.get(&source_vreg) {
                            match all_same_const {
                                None => all_same_const = Some(const_val),
                                Some(v) if v == const_val => {}
                                Some(_) => {
                                    is_uniform_const = false;
                                    break;
                                }
                            }
                        } else {
                            is_uniform_const = false;
                            break;
                        }
                    }
                    if is_uniform_const && let Some(const_val) = all_same_const {
                        const_values.insert(param_vreg, const_val);
                        changed = true;
                    }
                }
            }
        }
    }

    let debug = std::env::var("KAJIT_DEBUG_REMAT").is_ok();
    if debug {
        // Print first block's first param to identify which IR this is
        let first_param = program
            .funcs
            .first()
            .and_then(|f| f.blocks.get(1))
            .and_then(|b| b.params.first())
            .map(|v| v.index());
        eprintln!(
            "[remat] Found {} consts, vregs={}, b1.params[0]=v{:?}",
            const_values.len(),
            program.vreg_count,
            first_param
        );
    }

    for func in &mut program.funcs {
        rematerialize_constants_in_function(func, &const_values, debug);
    }

    if debug {
        let first_param = program
            .funcs
            .first()
            .and_then(|f| f.blocks.get(1))
            .and_then(|b| b.params.first())
            .map(|v| v.index());
        let param_count = program
            .funcs
            .first()
            .and_then(|f| f.blocks.get(1))
            .map(|b| b.params.len())
            .unwrap_or(0);
        eprintln!(
            "[remat] AFTER: b1 has {} params, first=v{:?}",
            param_count, first_param
        );
    }
}

fn rematerialize_constants_in_function(
    func: &mut Function,
    const_values: &HashMap<VReg, u64>,
    debug: bool,
) {
    // Track which block params to remove and what constants to insert
    // Key: BlockId, Value: Vec<(param_index, vreg, constant_value)>
    let mut remat_plan: HashMap<BlockId, Vec<(usize, VReg, u64)>> = HashMap::new();

    if debug {
        eprintln!(
            "[remat] func f{} const_values has {} entries",
            func.id.0,
            const_values.len()
        );
        for (vreg, val) in const_values.iter().take(10) {
            eprintln!("[remat]   v{} = {}", vreg.index(), val);
        }
    }

    // For each block (except entry), check its params
    for block in &func.blocks {
        if block.id == func.entry {
            continue;
        }

        if block.params.is_empty() || block.preds.is_empty() {
            continue;
        }

        if debug && block.id.0 < 3 {
            eprintln!(
                "[remat] checking block b{} with {} params",
                block.id.0,
                block.params.len()
            );
        }

        // For each param position, check if all incoming edges provide the same constant
        for (param_idx, &param_vreg) in block.params.iter().enumerate() {
            let mut all_same_const: Option<u64> = None;
            let mut is_uniform_const = true;

            for &pred_edge_id in &block.preds {
                let edge = &func.edges[pred_edge_id.index()];
                if param_idx >= edge.args.len() {
                    if debug && block.id.0 < 3 && param_idx < 3 {
                        eprintln!(
                            "[remat]   b{} param {} (v{}): edge e{} has only {} args",
                            block.id.0,
                            param_idx,
                            param_vreg.index(),
                            pred_edge_id.0,
                            edge.args.len()
                        );
                    }
                    is_uniform_const = false;
                    break;
                }
                let source_vreg = edge.args[param_idx].source;
                if let Some(&const_val) = const_values.get(&source_vreg) {
                    match all_same_const {
                        None => all_same_const = Some(const_val),
                        Some(v) if v == const_val => {}
                        Some(_) => {
                            is_uniform_const = false;
                            break;
                        }
                    }
                } else {
                    if debug && block.id.0 < 3 && param_idx < 3 {
                        eprintln!(
                            "[remat]   b{} param {} (v{}): source v{} not a constant",
                            block.id.0,
                            param_idx,
                            param_vreg.index(),
                            source_vreg.index()
                        );
                    }
                    is_uniform_const = false;
                    break;
                }
            }

            if is_uniform_const && let Some(const_val) = all_same_const {
                if debug && block.id.0 < 3 {
                    eprintln!(
                        "[remat]   b{} param {} (v{}): rematerializing const {}",
                        block.id.0,
                        param_idx,
                        param_vreg.index(),
                        const_val
                    );
                }
                remat_plan
                    .entry(block.id)
                    .or_default()
                    .push((param_idx, param_vreg, const_val));
            }
        }
    }

    if remat_plan.is_empty() {
        return;
    }

    // Apply the rematerialization plan
    // We need to process params in reverse order to keep indices valid during removal
    for (block_id, mut params_to_remat) in remat_plan {
        // Sort by param_idx descending so removals don't shift indices
        params_to_remat.sort_by(|a, b| b.0.cmp(&a.0));

        let block = &mut func.blocks[block_id.index()];

        // Collect new instructions to insert
        let mut new_insts = Vec::new();

        for (param_idx, vreg, const_val) in &params_to_remat {
            // Remove from block params
            block.params.remove(*param_idx);

            // Create new Const instruction
            let inst_id = InstId(func.insts.len() as u32);
            func.insts.push(Inst {
                id: inst_id,
                op: LinearOp::Const {
                    dst: *vreg,
                    value: *const_val,
                },
                operands: vec![Operand {
                    vreg: *vreg,
                    kind: OperandKind::Def,
                    class: RegClass::Gpr,
                    fixed: None,
                }],
                clobbers: Clobbers::default(),
            });
            new_insts.push(inst_id);
        }

        // Insert new instructions at the beginning of the block
        let block = &mut func.blocks[block_id.index()];
        for inst_id in new_insts.into_iter().rev() {
            block.insts.insert(0, inst_id);
        }

        // Remove corresponding edge args from all predecessor edges
        // Again, process in reverse order to keep indices valid
        for &pred_edge_id in &func.blocks[block_id.index()].preds.clone() {
            let edge = &mut func.edges[pred_edge_id.index()];
            for (param_idx, _, _) in &params_to_remat {
                if *param_idx < edge.args.len() {
                    edge.args.remove(*param_idx);
                }
            }
        }
    }
}

/// Copy propagation for CFG-MIR.
///
/// Replaces uses of a vreg that's just a copy of another vreg with the
/// original vreg. This enables later dead code elimination to remove
/// Propagate copies through uses across the CFG, leaving dead copies for later DCE.
///
/// Example:
///   v1 = Const(42)
///   v2 = Copy(v1)
///   v3 = Add(v2, v2)
/// Becomes:
///   v1 = Const(42)
///   v2 = Copy(v1)    // now dead
///   v3 = Add(v1, v1)
pub fn copy_propagation(program: &mut Program) {
    for func in &mut program.funcs {
        global_copy_propagation(func);
    }
}

/// Global (inter-block) copy propagation using dataflow analysis.
///
/// This pass propagates copies across block boundaries by:
/// 1. Building a map of all copies in the function
/// 2. Computing which vregs are available at each program point
/// 3. Rewriting uses to the ultimate source when safe
fn global_copy_propagation(func: &mut Function) {
    // Step 1: Build global copy map: dst -> src
    let mut copy_map: HashMap<VReg, VReg> = HashMap::new();
    for block in &func.blocks {
        for &inst_id in &block.insts {
            let inst = &func.insts[inst_id.index()];
            if let LinearOp::Copy { dst, src } = &inst.op {
                copy_map.insert(*dst, *src);
            }
        }
    }

    if copy_map.is_empty() {
        return;
    }

    // Step 2: Build vreg definition map: which block defines each vreg
    let mut vreg_def_block: HashMap<VReg, BlockId> = HashMap::new();

    for block in &func.blocks {
        // Block parameters are defined by this block
        for &param in &block.params {
            vreg_def_block.insert(param, block.id);
        }

        // Instruction defs
        for &inst_id in &block.insts {
            let inst = &func.insts[inst_id.index()];
            for operand in &inst.operands {
                if operand.kind == OperandKind::Def {
                    vreg_def_block.insert(operand.vreg, block.id);
                }
            }
        }
    }

    // Step 3: Compute dominators
    let idom = compute_dominators(func);

    // Helper: resolve copy chains to ultimate source
    let get_ultimate_source = |mut v: VReg| -> VReg {
        let mut visited = std::collections::HashSet::new();
        while let Some(&src) = copy_map.get(&v) {
            if v == src || !visited.insert(v) {
                break; // Self-copy or cycle
            }
            v = src;
        }
        v
    };

    // Step 4: Precompute intra-block def positions for each block
    let mut block_vreg_def_idx: HashMap<(BlockId, VReg), usize> = HashMap::new();

    for block in &func.blocks {
        let block_id = block.id;

        // Block params are defined at index 0
        for &param in &block.params {
            block_vreg_def_idx.insert((block_id, param), 0);
        }

        // Instruction defs
        for (idx, &inst_id) in block.insts.iter().enumerate() {
            let inst = &func.insts[inst_id.index()];
            for operand in &inst.operands {
                if operand.kind == OperandKind::Def {
                    block_vreg_def_idx.insert((block_id, operand.vreg), idx + 1);
                }
            }
        }
    }

    // Helper: check if vreg is available at given instruction in given block
    let is_available_at_inst = |vreg: VReg, block_id: BlockId, inst_idx: usize| -> bool {
        if let Some(&def_block) = vreg_def_block.get(&vreg) {
            if def_block != block_id {
                // Cross-block: check if def_block dominates block_id
                dominates(&idom, def_block, block_id)
            } else {
                // Same block: check that def comes before use
                if let Some(&def_idx) = block_vreg_def_idx.get(&(block_id, vreg)) {
                    def_idx < inst_idx
                } else {
                    false
                }
            }
        } else {
            false
        }
    };

    // Step 5: Rewrite all uses of copy destinations to their ultimate sources
    for block in &func.blocks {
        let block_id = block.id;

        // Rewrite instruction uses
        for (idx, &inst_id) in block.insts.iter().enumerate() {
            let inst_idx = idx + 1; // Instructions are 1-indexed (0 is block params)
            let inst = &mut func.insts[inst_id.index()];
            let mut changed = false;

            // Closure to rewrite a single vreg use
            let rewrite_use = |v: &mut VReg| -> bool {
                let ultimate = get_ultimate_source(*v);
                if ultimate != *v && is_available_at_inst(ultimate, block_id, inst_idx) {
                    *v = ultimate;
                    true
                } else {
                    false
                }
            };

            // Rewrite all uses in the instruction op
            match &mut inst.op {
                LinearOp::Copy { src, .. } => {
                    changed |= rewrite_use(src);
                }
                LinearOp::BinOp { lhs, rhs, .. } => {
                    changed |= rewrite_use(lhs);
                    changed |= rewrite_use(rhs);
                }
                LinearOp::UnaryOp { src, .. } => {
                    changed |= rewrite_use(src);
                }
                LinearOp::WriteToSlot { src, .. } => {
                    changed |= rewrite_use(src);
                }
                LinearOp::StoreToAddr { addr, src, .. } => {
                    changed |= rewrite_use(addr);
                    changed |= rewrite_use(src);
                }
                LinearOp::LoadFromAddr { addr, .. } => {
                    changed |= rewrite_use(addr);
                }
                LinearOp::BranchIf { cond, .. } | LinearOp::BranchIfZero { cond, .. } => {
                    changed |= rewrite_use(cond);
                }
                LinearOp::JumpTable { predicate, .. } => {
                    changed |= rewrite_use(predicate);
                }
                LinearOp::CallIntrinsic { args, .. }
                | LinearOp::CallPure { args, .. }
                | LinearOp::CallEffect { args, .. }
                | LinearOp::CallLambda { args, .. } => {
                    for arg in args.iter_mut() {
                        changed |= rewrite_use(arg);
                    }
                }
                _ => {}
            }

            // Update operands to match
            if changed {
                for operand in &mut inst.operands {
                    if operand.kind == OperandKind::Use {
                        let ultimate = get_ultimate_source(operand.vreg);
                        if ultimate != operand.vreg
                            && is_available_at_inst(ultimate, block_id, inst_idx)
                        {
                            operand.vreg = ultimate;
                        }
                    }
                }
            }
        }

        // Rewrite terminator uses
        let term_idx = block.insts.len() + 1; // Terminator comes after all instructions
        let term = &mut func.terms[block.term.index()];

        let rewrite_term_use = |v: &mut VReg| {
            let ultimate = get_ultimate_source(*v);
            if ultimate != *v && is_available_at_inst(ultimate, block_id, term_idx) {
                *v = ultimate;
            }
        };

        match term {
            Terminator::BranchIf { cond, .. } => {
                rewrite_term_use(cond);
            }
            Terminator::BranchIfZero { cond, .. } => {
                rewrite_term_use(cond);
            }
            Terminator::JumpTable { predicate, .. } => {
                rewrite_term_use(predicate);
            }
            _ => {}
        }

        // Rewrite edge arg sources
        for &edge_id in &block.succs {
            let edge = &mut func.edges[edge_id.index()];
            for arg in &mut edge.args {
                let ultimate = get_ultimate_source(arg.source);
                if ultimate != arg.source && is_available_at_inst(ultimate, block_id, term_idx) {
                    arg.source = ultimate;
                }
            }
        }
    }
}

/// Compute reverse postorder traversal of the CFG
fn compute_rpo(func: &Function) -> Vec<BlockId> {
    let mut visited = std::collections::HashSet::new();
    let mut postorder = Vec::new();

    fn dfs(
        func: &Function,
        block_id: BlockId,
        visited: &mut std::collections::HashSet<BlockId>,
        postorder: &mut Vec<BlockId>,
    ) {
        if !visited.insert(block_id) {
            return;
        }

        let block = &func.blocks[block_id.index()];
        for &edge_id in &block.succs {
            let edge = &func.edges[edge_id.index()];
            dfs(func, edge.to, visited, postorder);
        }

        postorder.push(block_id);
    }

    dfs(func, func.entry, &mut visited, &mut postorder);
    postorder.reverse();
    postorder
}

/// Compute dominators using Cooper-Harvey-Kennedy algorithm.
/// Returns a map from each block to its immediate dominator (idom).
/// The entry block has no idom (maps to None).
fn compute_dominators(func: &Function) -> HashMap<BlockId, Option<BlockId>> {
    let rpo = compute_rpo(func);
    let mut rpo_index: HashMap<BlockId, usize> = HashMap::new();
    for (idx, &block_id) in rpo.iter().enumerate() {
        rpo_index.insert(block_id, idx);
    }

    // Build predecessor map
    let mut preds: HashMap<BlockId, Vec<BlockId>> = HashMap::new();
    for block in &func.blocks {
        for &edge_id in &block.succs {
            let edge = &func.edges[edge_id.index()];
            preds.entry(edge.to).or_default().push(block.id);
        }
    }

    // Initialize: entry has no idom, others are undefined
    let mut idom: HashMap<BlockId, Option<BlockId>> = HashMap::new();
    idom.insert(func.entry, None);

    // Iterative fixed-point computation
    let mut changed = true;
    while changed {
        changed = false;
        for &block_id in &rpo {
            if block_id == func.entry {
                continue;
            }

            // Find the first processed predecessor
            let block_preds = preds.get(&block_id);
            let mut new_idom = None;

            if let Some(pred_list) = block_preds {
                for &pred in pred_list {
                    if idom.contains_key(&pred) {
                        new_idom = match new_idom {
                            None => Some(pred),
                            Some(current) => {
                                // Intersect current and pred
                                let mut finger1 = current;
                                let mut finger2 = pred;
                                while finger1 != finger2 {
                                    while rpo_index[&finger1] > rpo_index[&finger2] {
                                        finger1 = idom[&finger1].expect("idom should be set");
                                    }
                                    while rpo_index[&finger2] > rpo_index[&finger1] {
                                        finger2 = idom[&finger2].expect("idom should be set");
                                    }
                                }
                                Some(finger1)
                            }
                        };
                    }
                }
            }

            // Update if changed
            if idom.get(&block_id) != Some(&new_idom) {
                idom.insert(block_id, new_idom);
                changed = true;
            }
        }
    }

    idom
}

/// Check if block `a` dominates block `b` using the idom map.
fn dominates(idom: &HashMap<BlockId, Option<BlockId>>, a: BlockId, b: BlockId) -> bool {
    if a == b {
        return true;
    }

    // Walk up the dominator tree from b until we find a or reach the entry
    let mut current = b;
    loop {
        match idom.get(&current) {
            Some(Some(parent)) => {
                if *parent == a {
                    return true;
                }
                current = *parent;
            }
            _ => return false, // Reached entry or undefined
        }
    }
}

/// Check if a value can be encoded as an ARM64 immediate for the given operation.
/// This is conservative - we only eliminate def operands for values we're confident
/// can be encoded as immediates on all supported targets.
fn can_encode_as_immediate(op: kajit_lir::BinOpKind, value: u64) -> bool {
    use kajit_lir::BinOpKind;
    match op {
        // ARM64 add/sub immediate: 12-bit unsigned (0-4095)
        BinOpKind::Add | BinOpKind::Sub => value <= 4095,
        // ARM64 shift immediate: 6-bit (0-63)
        BinOpKind::Shl | BinOpKind::Shr => value < 64,
        // ARM64 logical immediate uses bitmask encoding. Only allow values we know encode.
        // Common varint masks: 0x7f (7 consecutive 1s), 0x80 (single bit), 0xff, etc.
        BinOpKind::And | BinOpKind::Or | BinOpKind::Xor => is_encodable_logical_imm(value),
        _ => false,
    }
}

/// Conservative check for ARM64 logical immediate encoding.
/// Returns true only for values we're certain can be encoded.
fn is_encodable_logical_imm(value: u64) -> bool {
    // All-zeros and all-ones are never encodable
    if value == 0 || value == u64::MAX {
        return false;
    }
    // Common varint masks - these are all encodable as bitmasks
    matches!(
        value,
        0x1 | 0x3 | 0x7 | 0xf | 0x1f | 0x3f | 0x7f | 0xff | // 1-8 consecutive 1s
            0x80 | 0x100 | 0x200 | 0x400 | 0x800 | 0x1000 | // single bits
            0xffff | 0xff00 | 0xf0 // other common masks
    )
}

/// Remove def operands from Const instructions that are only used as immediates.
///
/// When a constant is only used as the RHS of BinOp instructions where the value
/// can be encoded as an immediate, we don't need regalloc to track it. By removing
/// the Def operand, regalloc won't allocate a register or generate moves for it.
/// The backend will use the immediate form directly via `const_of()`.
/// Remove constant-def operands that are only needed as backend immediates.
pub fn eliminate_immediate_only_const_defs(program: &mut Program) {
    for func in &mut program.funcs {
        eliminate_immediate_only_const_defs_in_function(func);
    }
}

fn eliminate_immediate_only_const_defs_in_function(func: &mut Function) {
    use kajit_lir::BinOpKind;

    // Step 1: Build map of const vreg -> value
    let mut const_values: HashMap<VReg, u64> = HashMap::new();
    for inst in &func.insts {
        if let LinearOp::Const { dst, value } = &inst.op {
            const_values.insert(*dst, *value);
        }
    }

    if const_values.is_empty() {
        return;
    }

    // Step 1b: Build copy chains - track copies of consts
    // copy_to_const[copy_dst] = (original_const_vreg, const_value)
    let mut copy_to_const: HashMap<VReg, (VReg, u64)> = HashMap::new();
    for inst in &func.insts {
        if let LinearOp::Copy { dst, src } = &inst.op
            && let Some(&value) = const_values.get(src)
        {
            copy_to_const.insert(*dst, (*src, value));
        }
    }

    // Step 2: Track how each const/copy vreg is used
    // A const is "immediate-only" if ALL its uses (direct or via copies) are as RHS
    // of BinOp where the value can be encoded as an immediate.
    #[derive(Clone, Copy, PartialEq, Eq)]
    enum UseKind {
        ImmediateOnly,
        RequiresRegister,
    }

    // Track use kinds for both original consts and copies of consts
    let mut use_kinds: HashMap<VReg, UseKind> = HashMap::new();
    for vreg in const_values.keys() {
        use_kinds.insert(*vreg, UseKind::ImmediateOnly);
    }
    for vreg in copy_to_const.keys() {
        use_kinds.insert(*vreg, UseKind::ImmediateOnly);
    }

    // Helper: get const value for a vreg (either direct const or copy of const)
    let get_const_value = |v: &VReg| -> Option<u64> {
        if let Some(&val) = const_values.get(v) {
            return Some(val);
        }
        if let Some(&(_, val)) = copy_to_const.get(v) {
            return Some(val);
        }
        None
    };

    // Helper: check if vreg is a const or copy-of-const
    let is_const_like =
        |v: &VReg| -> bool { const_values.contains_key(v) || copy_to_const.contains_key(v) };

    // Scan all uses
    for inst in &func.insts {
        match &inst.op {
            LinearOp::BinOp { op, lhs, rhs, .. } => {
                let is_compare = matches!(
                    op,
                    BinOpKind::CmpEq
                        | BinOpKind::CmpNe
                        | BinOpKind::CmpLt
                        | BinOpKind::CmpLe
                        | BinOpKind::CmpGt
                        | BinOpKind::CmpGe
                );
                if is_compare {
                    // For compares, EITHER operand can be the immediate —
                    // the backend swaps operands to put the const on the RHS.
                    // ARM64 cmp immediate: 12-bit unsigned (0-4095).
                    if let Some(value) = get_const_value(lhs)
                        && value > 4095
                    {
                        use_kinds.insert(*lhs, UseKind::RequiresRegister);
                    }
                    if let Some(value) = get_const_value(rhs)
                        && value > 4095
                    {
                        use_kinds.insert(*rhs, UseKind::RequiresRegister);
                    }
                } else {
                    // Non-compare: LHS always requires register
                    if is_const_like(lhs) {
                        use_kinds.insert(*lhs, UseKind::RequiresRegister);
                    }
                    // RHS can potentially be immediate
                    if let Some(value) = get_const_value(rhs)
                        && !can_encode_as_immediate(*op, value)
                    {
                        use_kinds.insert(*rhs, UseKind::RequiresRegister);
                    }
                }
            }
            LinearOp::Copy { dst, src } => {
                // If src is a const and dst is tracked as a copy-of-const, we handle
                // this specially - don't mark src as requiring register yet.
                // The copy is "transparent" - we'll decide based on how dst is used.
                if const_values.contains_key(src) && copy_to_const.contains_key(dst) {
                    // This copy is tracked - don't mark as RequiresRegister yet
                } else if is_const_like(src) {
                    use_kinds.insert(*src, UseKind::RequiresRegister);
                }
            }
            LinearOp::UnaryOp { src, .. } => {
                if is_const_like(src) {
                    use_kinds.insert(*src, UseKind::RequiresRegister);
                }
            }
            // Any other use requires a register
            _ => {
                for operand in &inst.operands {
                    if operand.kind == OperandKind::Use && is_const_like(&operand.vreg) {
                        use_kinds.insert(operand.vreg, UseKind::RequiresRegister);
                    }
                }
            }
        }
    }

    // Also check terminator uses
    for term in &func.terms {
        let cond = match term {
            Terminator::BranchIf { cond, .. } | Terminator::BranchIfZero { cond, .. } => {
                Some(*cond)
            }
            Terminator::JumpTable { predicate, .. } => Some(*predicate),
            _ => None,
        };
        if let Some(cond) = cond
            && is_const_like(&cond)
        {
            use_kinds.insert(cond, UseKind::RequiresRegister);
        }
    }

    // Also check edge args - consts passed through edges require registers
    for edge in &func.edges {
        for arg in &edge.args {
            if is_const_like(&arg.source) {
                use_kinds.insert(arg.source, UseKind::RequiresRegister);
            }
        }
    }

    // Also check data_results - consts used as function results require registers
    for vreg in &func.data_results {
        if is_const_like(vreg) {
            use_kinds.insert(*vreg, UseKind::RequiresRegister);
        }
    }

    // Step 3: Propagate RequiresRegister from copies back to original consts
    // If a copy-of-const requires a register, the original const also needs one.
    for (copy_vreg, (original_const, _)) in &copy_to_const {
        if use_kinds.get(copy_vreg) == Some(&UseKind::RequiresRegister) {
            use_kinds.insert(*original_const, UseKind::RequiresRegister);
        }
    }

    // Step 4: Collect vregs that are immediate-only
    let immediate_only: HashSet<VReg> = use_kinds
        .iter()
        .filter(|(_, kind)| **kind == UseKind::ImmediateOnly)
        .map(|(vreg, _)| *vreg)
        .collect();

    if immediate_only.is_empty() {
        return;
    }

    // Step 5: Remove Def operands from Const/Copy instructions for immediate-only vregs,
    // AND remove Use operands from BinOps that use them as RHS.
    // This is necessary because regalloc requires all uses to have reaching definitions.
    for inst in &mut func.insts {
        match &inst.op {
            LinearOp::Const { dst, .. } => {
                if immediate_only.contains(dst) {
                    inst.operands.clear();
                }
            }
            LinearOp::Copy { dst, src } => {
                // If this is a copy of an immediate-only const, clear its operands too
                if copy_to_const.contains_key(dst) && immediate_only.contains(dst) {
                    inst.operands.clear();
                }
                // Also handle if src is immediate-only (shouldn't happen given our logic, but be safe)
                let _ = src;
            }
            LinearOp::BinOp { lhs, rhs, .. } => {
                // Remove Use operands for immediate-only consts (LHS for compares, RHS for all)
                if immediate_only.contains(rhs) || immediate_only.contains(lhs) {
                    inst.operands.retain(|op| {
                        !immediate_only.contains(&op.vreg) || op.kind != OperandKind::Use
                    });
                }
            }
            _ => {}
        }
    }
}

/// Dead code elimination for CFG-MIR.
///
/// Removes instructions whose outputs are never used. This is particularly
/// useful after rematerialization, which can leave the original constant
/// definitions unused.
/// Remove instructions whose results are unused and which have no side effects.
pub fn dead_code_elimination(program: &mut Program) {
    for func in &mut program.funcs {
        dead_code_elimination_in_function(func);
    }
}

fn dead_code_elimination_in_function(func: &mut Function) {
    let debug = std::env::var("KAJIT_DEBUG_DCE").is_ok();

    // Iterate until no more changes
    let mut iteration = 0;
    loop {
        iteration += 1;

        // Build analysis structures in one pass over blocks
        let analysis = build_use_def_analysis(func);

        if debug && iteration == 1 {
            eprintln!(
                "[dce] func f{}: {} vregs with external uses, {} vregs with local defs",
                func.id.0,
                analysis.external_uses.len(),
                analysis.local_defs.len()
            );
        }

        // Find instructions to remove
        let mut insts_to_remove: HashSet<InstId> = HashSet::new();

        for block in &func.blocks {
            let is_entry = block.id == func.entry;

            for &inst_id in &block.insts {
                let inst = &func.insts[inst_id.index()];
                if !is_pure_instruction(&inst.op) {
                    continue;
                }

                // Don't remove Const instructions with no operands - they're needed
                // for const_of() tracking in the backend (immediate-only consts).
                if matches!(&inst.op, LinearOp::Const { .. }) && inst.operands.is_empty() {
                    continue;
                }

                // Check if all defs are unused
                let all_defs_unused = inst.operands.iter().all(|op| {
                    if op.kind != OperandKind::Def {
                        return true;
                    }
                    is_def_unused(func, &analysis, op.vreg, block.id, is_entry)
                });

                if all_defs_unused {
                    insts_to_remove.insert(inst.id);
                }
            }
        }

        if insts_to_remove.is_empty() {
            if debug {
                eprintln!("[dce] iteration {} found no dead code, done", iteration);
            }
            break;
        }

        if debug {
            eprintln!(
                "[dce] iteration {} removing {} dead insts",
                iteration,
                insts_to_remove.len()
            );
        }

        // Remove dead instructions from blocks
        for block in &mut func.blocks {
            block.insts.retain(|id| !insts_to_remove.contains(id));
        }
    }
}

/// Analysis results for use-def relationships
struct UseDefAnalysis {
    /// For each vreg: blocks that use it BEFORE any local def (external uses)
    external_uses: HashMap<VReg, HashSet<BlockId>>,
    /// For each vreg: blocks that define it locally
    local_defs: HashMap<VReg, HashSet<BlockId>>,
    /// Vregs used by edge arguments
    edge_arg_uses: HashSet<VReg>,
    /// Vregs used by terminators
    terminator_uses: HashSet<VReg>,
    /// Vregs that are function results
    result_uses: HashSet<VReg>,
}

fn build_use_def_analysis(func: &Function) -> UseDefAnalysis {
    let mut external_uses: HashMap<VReg, HashSet<BlockId>> = HashMap::new();
    let mut local_defs: HashMap<VReg, HashSet<BlockId>> = HashMap::new();

    // Analyze each block
    for block in &func.blocks {
        // Block params are local defs
        for &param in &block.params {
            local_defs.entry(param).or_default().insert(block.id);
        }

        // Track defs seen so far in this block (for determining external vs local uses)
        let mut defs_in_block: HashSet<VReg> = block.params.iter().copied().collect();

        for &inst_id in &block.insts {
            let inst = &func.insts[inst_id.index()];

            // Check uses first (before adding this instruction's defs)
            for op in &inst.operands {
                if op.kind == OperandKind::Use && !defs_in_block.contains(&op.vreg) {
                    // This is an external use - vreg not defined in this block before this point
                    external_uses.entry(op.vreg).or_default().insert(block.id);
                }
            }

            // Then record defs
            for op in &inst.operands {
                if op.kind == OperandKind::Def {
                    defs_in_block.insert(op.vreg);
                    local_defs.entry(op.vreg).or_default().insert(block.id);
                }
            }
        }
    }

    // Collect edge argument uses
    let mut edge_arg_uses: HashSet<VReg> = HashSet::new();
    for edge in &func.edges {
        for arg in &edge.args {
            edge_arg_uses.insert(arg.source);
        }
    }

    // Collect terminator uses
    let mut terminator_uses: HashSet<VReg> = HashSet::new();
    for term in &func.terms {
        match term {
            Terminator::BranchIf { cond, .. } | Terminator::BranchIfZero { cond, .. } => {
                terminator_uses.insert(*cond);
            }
            Terminator::JumpTable { predicate, .. } => {
                terminator_uses.insert(*predicate);
            }
            Terminator::Return | Terminator::Branch { .. } => {}
        }
    }

    // Collect result uses
    let result_uses: HashSet<VReg> = func.data_results.iter().copied().collect();

    UseDefAnalysis {
        external_uses,
        local_defs,
        edge_arg_uses,
        terminator_uses,
        result_uses,
    }
}

/// Check if a def of `vreg` in `def_block` is unused
fn is_def_unused(
    func: &Function,
    analysis: &UseDefAnalysis,
    vreg: VReg,
    def_block: BlockId,
    is_entry_block: bool,
) -> bool {
    // If it's a function result, it's used
    if analysis.result_uses.contains(&vreg) {
        return false;
    }

    // If it's used by an edge arg, it's used
    if analysis.edge_arg_uses.contains(&vreg) {
        return false;
    }

    // If it's used by a terminator, it's used
    if analysis.terminator_uses.contains(&vreg) {
        return false;
    }

    // Get blocks that have external uses of this vreg
    let use_blocks = analysis.external_uses.get(&vreg);

    // If no external uses anywhere, def is unused (for cross-block purposes)
    // But we still need to check local uses within the same block
    if use_blocks.is_none() || use_blocks.unwrap().is_empty() {
        // Check if there are local uses in the def block itself
        // (uses after the def in the same block)
        return !has_local_use_after_def(func, vreg, def_block);
    }

    let use_blocks = use_blocks.unwrap();

    // For entry block defs: they're unused if every block that uses the vreg
    // (externally) also has a local def that shadows the entry def
    if is_entry_block {
        let def_blocks = analysis.local_defs.get(&vreg);
        if let Some(def_blocks) = def_blocks {
            // Entry def is unused if all use_blocks are also in def_blocks
            // (meaning each using block has its own local def)
            let all_uses_shadowed = use_blocks.iter().all(|use_block| {
                // The use block has a local def (which shadows the entry def)
                def_blocks.contains(use_block)
            });
            if all_uses_shadowed {
                // Also check no local use in entry block itself
                return !has_local_use_after_def(func, vreg, def_block);
            }
        }
    }

    // There are external uses not satisfied by local defs
    false
}

/// Check if there's a use of `vreg` in `block` after its definition
fn has_local_use_after_def(func: &Function, vreg: VReg, block_id: BlockId) -> bool {
    let block = &func.blocks[block_id.index()];

    // Find where vreg is defined in this block, then check for uses after
    let mut found_def = false;
    for &inst_id in &block.insts {
        let inst = &func.insts[inst_id.index()];

        if found_def {
            // Check if this instruction uses vreg
            for op in &inst.operands {
                if op.kind == OperandKind::Use && op.vreg == vreg {
                    return true;
                }
            }
        }

        // Check if this instruction defines vreg
        for op in &inst.operands {
            if op.kind == OperandKind::Def && op.vreg == vreg {
                found_def = true;
                break;
            }
        }
    }

    false
}

/// Returns true if an instruction has no side effects and can be safely removed
/// if its outputs are unused.
fn is_pure_instruction(op: &LinearOp) -> bool {
    matches!(
        op,
        LinearOp::Const { .. }
            | LinearOp::Copy { .. }
            | LinearOp::BinOp { .. }
            | LinearOp::UnaryOp { .. }
    )
}

/// Local Common Subexpression Elimination within each basic block.
///
/// This pass identifies redundant computations within a block and replaces them
/// with copies from the first computation. Handles:
/// - Duplicate constants (same value)
/// - Duplicate slot reads (same slot, no intervening write)
/// - Duplicate binary operations (same op and operands)
/// - Duplicate unary operations (same op and operand)
pub fn local_cse(program: &mut Program) {
    for func in &mut program.funcs {
        local_cse_in_function(func);
    }
}

fn local_cse_in_function(func: &mut Function) {
    use kajit_ir::SlotId;
    use kajit_lir::{BinOpKind, UnaryOpKind};

    for block in &func.blocks {
        // Track known values within this block
        let mut known_consts: HashMap<u64, VReg> = HashMap::new();
        let mut known_slot_reads: HashMap<SlotId, VReg> = HashMap::new();
        let mut known_binops: HashMap<(BinOpKind, VReg, VReg), VReg> = HashMap::new();
        let mut known_unaryops: HashMap<(UnaryOpKind, VReg), VReg> = HashMap::new();

        // Collect replacements: inst_id -> (Copy instruction, new operands)
        let mut replacements: Vec<(InstId, LinearOp, Vec<Operand>)> = Vec::new();

        for &inst_id in &block.insts {
            let inst = &func.insts[inst_id.index()];
            match &inst.op {
                LinearOp::Const { dst, value } => {
                    if let Some(&existing) = known_consts.get(value) {
                        // Replace with copy from existing
                        let new_operands = vec![
                            Operand {
                                vreg: existing,
                                kind: OperandKind::Use,
                                class: RegClass::Gpr,
                                fixed: None,
                            },
                            Operand {
                                vreg: *dst,
                                kind: OperandKind::Def,
                                class: RegClass::Gpr,
                                fixed: None,
                            },
                        ];
                        replacements.push((
                            inst_id,
                            LinearOp::Copy {
                                dst: *dst,
                                src: existing,
                            },
                            new_operands,
                        ));
                    } else {
                        known_consts.insert(*value, *dst);
                    }
                }

                LinearOp::ReadFromSlot { dst, slot } => {
                    if let Some(&existing) = known_slot_reads.get(slot) {
                        let new_operands = vec![
                            Operand {
                                vreg: existing,
                                kind: OperandKind::Use,
                                class: RegClass::Gpr,
                                fixed: None,
                            },
                            Operand {
                                vreg: *dst,
                                kind: OperandKind::Def,
                                class: RegClass::Gpr,
                                fixed: None,
                            },
                        ];
                        replacements.push((
                            inst_id,
                            LinearOp::Copy {
                                dst: *dst,
                                src: existing,
                            },
                            new_operands,
                        ));
                    } else {
                        known_slot_reads.insert(*slot, *dst);
                    }
                }

                LinearOp::WriteToSlot { slot, .. } => {
                    // Invalidate the slot read cache for this slot
                    known_slot_reads.remove(slot);
                }

                LinearOp::BinOp { op, dst, lhs, rhs } => {
                    let key = (*op, *lhs, *rhs);
                    if let Some(&existing) = known_binops.get(&key) {
                        let new_operands = vec![
                            Operand {
                                vreg: existing,
                                kind: OperandKind::Use,
                                class: RegClass::Gpr,
                                fixed: None,
                            },
                            Operand {
                                vreg: *dst,
                                kind: OperandKind::Def,
                                class: RegClass::Gpr,
                                fixed: None,
                            },
                        ];
                        replacements.push((
                            inst_id,
                            LinearOp::Copy {
                                dst: *dst,
                                src: existing,
                            },
                            new_operands,
                        ));
                    } else {
                        known_binops.insert(key, *dst);
                        // For commutative ops, also insert the swapped version
                        if matches!(
                            op,
                            BinOpKind::Add
                                | BinOpKind::Mul
                                | BinOpKind::And
                                | BinOpKind::Or
                                | BinOpKind::Xor
                                | BinOpKind::CmpEq
                                | BinOpKind::CmpNe
                        ) && lhs != rhs
                        {
                            known_binops.insert((*op, *rhs, *lhs), *dst);
                        }
                    }
                }

                LinearOp::UnaryOp { op, dst, src } => {
                    let key = (*op, *src);
                    if let Some(&existing) = known_unaryops.get(&key) {
                        let new_operands = vec![
                            Operand {
                                vreg: existing,
                                kind: OperandKind::Use,
                                class: RegClass::Gpr,
                                fixed: None,
                            },
                            Operand {
                                vreg: *dst,
                                kind: OperandKind::Def,
                                class: RegClass::Gpr,
                                fixed: None,
                            },
                        ];
                        replacements.push((
                            inst_id,
                            LinearOp::Copy {
                                dst: *dst,
                                src: existing,
                            },
                            new_operands,
                        ));
                    } else {
                        known_unaryops.insert(key, *dst);
                    }
                }

                // Other instructions may have side effects or modify state
                // that invalidates our knowledge
                LinearOp::CallIntrinsic { .. } | LinearOp::CallLambda { .. } => {
                    // Calls may modify slots, so clear slot knowledge
                    known_slot_reads.clear();
                }

                _ => {}
            }
        }

        // Apply replacements
        for (inst_id, new_op, new_operands) in replacements {
            let inst = &mut func.insts[inst_id.index()];
            inst.op = new_op;
            inst.operands = new_operands;
        }
    }
}

// ============================================================================
// Local Common Subexpression Elimination
// ============================================================================

/// Dominator tree for a function's CFG.
#[derive(Debug, Clone)]
pub struct DomTree {
    /// Immediate dominator for each block. Entry block has None.
    pub idom: Vec<Option<BlockId>>,
    /// Children in the dominator tree (blocks immediately dominated by this one).
    pub children: Vec<Vec<BlockId>>,
}

impl DomTree {
    /// Compute the dominator tree using iterative dataflow.
    pub fn compute(func: &Function) -> Self {
        let n = func.blocks.len();
        if n == 0 {
            return DomTree {
                idom: Vec::new(),
                children: Vec::new(),
            };
        }

        // Initialize: entry dominates itself, others undefined
        let mut idom: Vec<Option<BlockId>> = vec![None; n];
        let entry_idx = func.entry.index();

        // Build predecessor map from edges
        let mut preds: Vec<Vec<BlockId>> = vec![Vec::new(); n];
        for edge in &func.edges {
            let to_idx = edge.to.index();
            if to_idx < n {
                preds[to_idx].push(edge.from);
            }
        }

        // Compute reverse postorder for iteration efficiency
        let rpo = Self::reverse_postorder(func);
        let mut rpo_index: Vec<usize> = vec![usize::MAX; n];
        for (i, &block_id) in rpo.iter().enumerate() {
            rpo_index[block_id.index()] = i;
        }

        // Iterative dominator computation
        let mut changed = true;
        while changed {
            changed = false;
            for &block_id in &rpo {
                let b = block_id.index();
                if b == entry_idx {
                    continue;
                }

                // Find first processed predecessor
                let mut new_idom: Option<BlockId> = None;
                for &pred in &preds[b] {
                    let p = pred.index();
                    if rpo_index[p] < rpo_index[b] || idom[p].is_some() || p == entry_idx {
                        if new_idom.is_none() {
                            new_idom = Some(pred);
                        } else {
                            // Intersect
                            new_idom = Some(Self::intersect(
                                new_idom.unwrap(),
                                pred,
                                &idom,
                                &rpo_index,
                                entry_idx,
                            ));
                        }
                    }
                }

                if new_idom != idom[b] {
                    idom[b] = new_idom;
                    changed = true;
                }
            }
        }

        // Build children map
        let mut children: Vec<Vec<BlockId>> = vec![Vec::new(); n];
        for (b, dom) in idom.iter().enumerate() {
            if let Some(d) = dom {
                children[d.index()].push(BlockId(b as u32));
            }
        }

        DomTree { idom, children }
    }

    fn reverse_postorder(func: &Function) -> Vec<BlockId> {
        let n = func.blocks.len();
        let mut visited = vec![false; n];
        let mut postorder = Vec::with_capacity(n);

        // Build successor map
        let mut succs: Vec<Vec<BlockId>> = vec![Vec::new(); n];
        for edge in &func.edges {
            let from_idx = edge.from.index();
            if from_idx < n {
                succs[from_idx].push(edge.to);
            }
        }

        fn dfs(
            block: BlockId,
            succs: &[Vec<BlockId>],
            visited: &mut [bool],
            postorder: &mut Vec<BlockId>,
        ) {
            let b = block.index();
            if visited[b] {
                return;
            }
            visited[b] = true;
            for &succ in &succs[b] {
                dfs(succ, succs, visited, postorder);
            }
            postorder.push(block);
        }

        dfs(func.entry, &succs, &mut visited, &mut postorder);
        postorder.reverse();
        postorder
    }

    fn intersect(
        mut b1: BlockId,
        mut b2: BlockId,
        idom: &[Option<BlockId>],
        rpo_index: &[usize],
        entry_idx: usize,
    ) -> BlockId {
        while b1 != b2 {
            while rpo_index[b1.index()] > rpo_index[b2.index()] {
                if b1.index() == entry_idx {
                    break;
                }
                b1 = idom[b1.index()].unwrap_or(b1);
            }
            while rpo_index[b2.index()] > rpo_index[b1.index()] {
                if b2.index() == entry_idx {
                    break;
                }
                b2 = idom[b2.index()].unwrap_or(b2);
            }
        }
        b1
    }
}

/// Hashable key for value numbering expressions.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
enum ValueKey {
    Const(u64),
    BinOp {
        op: kajit_lir::BinOpKind,
        lhs: VReg,
        rhs: VReg,
    },
    UnaryOp {
        op: kajit_lir::UnaryOpKind,
        src: VReg,
    },
    SlotAddr(kajit_ir::SlotId),
    CallPure {
        func: FnPtr,
        args: Vec<VReg>,
    },
}

/// Scoped hash table for dominator-tree-ordered value numbering.
struct ScopedValueTable {
    scopes: Vec<HashMap<ValueKey, VReg>>,
}

impl ScopedValueTable {
    fn new() -> Self {
        ScopedValueTable {
            scopes: vec![HashMap::new()],
        }
    }

    fn push_scope(&mut self) {
        self.scopes.push(HashMap::new());
    }

    fn lookup(&self, key: &ValueKey) -> Option<VReg> {
        for scope in self.scopes.iter().rev() {
            if let Some(&vreg) = scope.get(key) {
                return Some(vreg);
            }
        }
        None
    }

    fn insert(&mut self, key: ValueKey, vreg: VReg) {
        if let Some(scope) = self.scopes.last_mut() {
            scope.insert(key, vreg);
        }
    }
}

/// Local common subexpression elimination: eliminates redundant computations within blocks.
///
/// Uses dominator tree to ensure that when we reference a canonical value,
/// it is guaranteed to dominate (be available at) the current block.
/// Eliminate redundant expressions within blocks by converting them to copies.
pub fn local_common_subexpr_elim(program: &mut Program) {
    for func in &mut program.funcs {
        local_common_subexpr_elim_in_function(func);
    }
}

fn local_common_subexpr_elim_in_function(func: &mut Function) {
    if func.blocks.is_empty() {
        return;
    }

    // For now, we do per-block value numbering. True cross-block CSE would require
    // ensuring the canonical vreg is live at the use site (via block params/edges).
    // Local CSE already handles constants, so focus on BinOp/UnaryOp/SlotAddr/CallPure.
    //
    // Within each block, we can safely convert redundant computations to copies
    // because the canonical vreg is defined earlier in the same block.

    for block in &func.blocks {
        let mut table = ScopedValueTable::new();
        table.push_scope();

        // Map from InstId -> canonical VReg (instructions to convert to copies)
        let mut convert_to_copy: Vec<(InstId, VReg)> = Vec::new();

        for &inst_id in &block.insts {
            let inst = &func.insts[inst_id.index()];

            if let Some(key) = make_value_key_simple(&inst.op) {
                if let Some(canonical) = table.lookup(&key) {
                    // Found redundant computation - convert to copy
                    if let Some(dst) = get_def_vreg(&inst.op) {
                        // Don't create self-copies
                        if dst != canonical {
                            convert_to_copy.push((inst_id, canonical));
                        }
                    }
                } else {
                    // First time seeing this value
                    if let Some(dst) = get_def_vreg(&inst.op) {
                        table.insert(key, dst);
                    }
                }
            }
        }

        // Convert redundant instructions to copies
        for (inst_id, canonical) in convert_to_copy {
            let inst = &mut func.insts[inst_id.index()];
            if let Some(dst) = get_def_vreg(&inst.op) {
                inst.op = LinearOp::Copy {
                    dst,
                    src: canonical,
                };
                // Update operands: one Use of canonical, one Def of dst
                inst.operands = vec![
                    Operand {
                        vreg: canonical,
                        kind: OperandKind::Use,
                        class: RegClass::Gpr,
                        fixed: None,
                    },
                    Operand {
                        vreg: dst,
                        kind: OperandKind::Def,
                        class: RegClass::Gpr,
                        fixed: None,
                    },
                ];
            }
        }
    }
}

/// Create a ValueKey for an instruction (simple version without canonicalization).
/// This is safe for the copy-conversion approach: we only deduplicate identical expressions.
fn make_value_key_simple(op: &LinearOp) -> Option<ValueKey> {
    match op {
        LinearOp::Const { value, .. } => Some(ValueKey::Const(*value)),

        LinearOp::BinOp { op, lhs, rhs, .. } => Some(ValueKey::BinOp {
            op: *op,
            lhs: *lhs,
            rhs: *rhs,
        }),

        LinearOp::UnaryOp { op, src, .. } => Some(ValueKey::UnaryOp { op: *op, src: *src }),

        // Don't value-number Copy - it's handled by copy_propagation
        LinearOp::Copy { .. } => None,

        LinearOp::SlotAddr { slot, .. } => Some(ValueKey::SlotAddr(*slot)),

        // Each StackAlloc is unique — don't CSE them.
        LinearOp::StackAlloc { .. } => None,

        LinearOp::CallPure { func, args, .. } => Some(ValueKey::CallPure {
            func: *func,
            args: args.clone(),
        }),

        // Side-effecting operations cannot be value-numbered
        _ => None,
    }
}

/// Get the destination vreg defined by an instruction.
fn get_def_vreg(op: &LinearOp) -> Option<VReg> {
    match op {
        LinearOp::Const { dst, .. }
        | LinearOp::ExternAddr { dst, .. }
        | LinearOp::BinOp { dst, .. }
        | LinearOp::UnaryOp { dst, .. }
        | LinearOp::Copy { dst, .. }
        | LinearOp::SlotAddr { dst, .. }
        | LinearOp::StackAlloc { dst, .. }
        | LinearOp::CallPure { dst, .. }
        | LinearOp::CallEffect { dst, .. } => Some(*dst),
        _ => None,
    }
}
