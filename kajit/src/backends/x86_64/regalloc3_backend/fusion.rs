//! Instruction fusion analysis for x86_64 regalloc3 backend.
//!
//! x86_64 fusions:
//! - CmpXx + BranchIf/BranchIfZero → cmp + jcc (skip setcc)
//! - Add(base, const) + LoadFromAddr → [base + offset] addressing

use kajit_emit::x64;
use kajit_lir::{BinOpKind, LinearOp};
use kajit_mir::ir::{self, Function, Terminator};
use kajit_mir::regalloc3_result::AllocatedCfgFunctionRa3;
use std::collections::HashMap;

/// Compute which CmpXx vregs can be fused with their branch terminator.
/// A cmp is fusable if its result vreg is only used by the block's BranchIf/BranchIfZero.
pub(super) fn compute_fusable_cmps(func: &Function) -> HashMap<kajit_ir::VReg, x64::Condition> {
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

        if use_counts.get(&cond).copied().unwrap_or(0) != 1 {
            continue;
        }

        // Find the cmp instruction index and check no flag-clobbering ops follow it.
        let mut cmp_idx = None;
        let mut cmp_condition = None;
        for (i, &inst_id) in block.insts.iter().enumerate() {
            let inst = &func.insts[inst_id.index()];
            if let LinearOp::BinOp { op, dst, .. } = &inst.op
                && *dst == cond
            {
                cmp_condition = match op {
                    BinOpKind::CmpEq => Some(x64::Condition::Eq),
                    BinOpKind::CmpNe => Some(x64::Condition::Ne),
                    BinOpKind::CmpLt => Some(x64::Condition::Lo),
                    BinOpKind::CmpLe => Some(x64::Condition::Ls),
                    BinOpKind::CmpGt => Some(x64::Condition::Hi),
                    BinOpKind::CmpGe => Some(x64::Condition::Hs),
                    _ => None,
                };
                if cmp_condition.is_some() {
                    cmp_idx = Some(i);
                }
                break;
            }
        }

        let (Some(idx), Some(cc)) = (cmp_idx, cmp_condition) else {
            continue;
        };

        // Check that all instructions after the cmp don't clobber flags.
        let flags_safe = block.insts[idx + 1..].iter().all(|&inst_id| {
            let inst = &func.insts[inst_id.index()];
            matches!(
                &inst.op,
                LinearOp::LoadFromAddr { .. }
                    | LinearOp::StoreToAddr { .. }
                    | LinearOp::WriteToSlot { .. }
                    | LinearOp::ReadFromSlot { .. }
                    | LinearOp::Const { .. }
                    | LinearOp::SlotAddr { .. }
                    | LinearOp::Copy { .. }
            )
        });

        if flags_safe {
            fusable.insert(cond, cc);
        }
    }

    fusable
}

/// Pre-compute base+offset fusions for LoadFromAddr.
/// When an Add(base, const) result is consumed ONLY by LoadFromAddr,
/// we can skip the Add and use `[base + offset]` directly.
pub(super) fn compute_fusable_addr_offsets(
    func: &Function,
    alloc_func: &AllocatedCfgFunctionRa3,
    const_values: &HashMap<kajit_ir::VReg, u64>,
    skip_set: &mut std::collections::HashSet<kajit_ir::VReg>,
) -> HashMap<kajit_ir::VReg, (kajit_ir::VReg, u64)> {
    // Count uses of each vreg
    let mut use_counts: HashMap<kajit_ir::VReg, usize> = HashMap::new();
    for inst in &func.insts {
        for op in &inst.operands {
            if op.kind == ir::OperandKind::Use {
                *use_counts.entry(op.vreg).or_insert(0) += 1;
            }
        }
    }
    for block in &func.blocks {
        let term = &func.terms[block.term.index()];
        let edge_ids: Vec<ir::EdgeId> = match term {
            ir::Terminator::Branch { edge } => vec![*edge],
            ir::Terminator::BranchIf {
                taken, fallthrough, ..
            }
            | ir::Terminator::BranchIfZero {
                taken, fallthrough, ..
            } => vec![*taken, *fallthrough],
            ir::Terminator::JumpTable { targets, .. } => targets.clone(),
            _ => vec![],
        };
        for eid in edge_ids {
            let edge = &func.edges[eid.index()];
            for arg in &edge.args {
                *use_counts.entry(arg.source).or_insert(0) += 1;
            }
        }
    }

    // Build def map: vreg → defining Add info
    let mut add_defs: HashMap<kajit_ir::VReg, (kajit_ir::VReg, kajit_ir::VReg)> = HashMap::new();
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

    for inst in &func.insts {
        let addr_vreg = match &inst.op {
            LinearOp::LoadFromAddr { addr, .. } => *addr,
            _ => continue,
        };

        let addr_uses = use_counts.get(&addr_vreg).copied().unwrap_or(0);
        if addr_uses != 1 {
            continue;
        }
        let Some(&(base, rhs)) = add_defs.get(&addr_vreg) else {
            continue;
        };
        // x86_64 displacement fits in i32, but limit to reasonable range
        let Some(&offset) = const_values.get(&rhs) else {
            continue;
        };
        if offset > 0x7FFF_FFFF {
            continue;
        }
        let rhs_uses = use_counts.get(&rhs).copied().unwrap_or(0);
        if rhs_uses > 1 {
            continue;
        }

        // Only fuse when addr and base are in the same physical register.
        let Some(addr_preg) = alloc_func.preg_for_vreg(addr_vreg) else {
            continue;
        };
        let Some(base_preg) = alloc_func.preg_for_vreg(base) else {
            continue;
        };
        if addr_preg != base_preg {
            continue;
        }

        result.insert(addr_vreg, (base, offset));
        skip_set.insert(addr_vreg);
        skip_set.insert(rhs);
    }

    result
}
