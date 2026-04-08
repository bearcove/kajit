//! Instruction fusion analysis for aarch64 regalloc3 backend.
//! Computes which instructions can be fused: cmp+branch, bfi, bit tests, addr offsets.

use kajit_asm::aarch64::{Condition, Reg, Width};
use kajit_mir::ir::{self, Function, Terminator};
use kajit_mir::regalloc3_result::AllocatedCfgFunctionRa3;

use kajit_lir::{BinOpKind, LinearOp};
use std::collections::HashMap;

use super::BfiInfo;
use super::context::EmitContext;

impl<'a> EmitContext<'a> {
    /// Check if a vreg is the result of `And(x, power_of_2)`.
    /// Returns (source_vreg, bit_position) if so.
    pub(super) fn is_and_bit_test(&self, vreg: kajit_ir::VReg) -> Option<(kajit_ir::VReg, u8)> {
        for inst in &self.func.insts {
            if let LinearOp::BinOp {
                op: BinOpKind::And,
                dst,
                lhs,
                rhs,
            } = &inst.op
                && *dst == vreg
            {
                // Check if rhs is a power of 2 constant
                if let Some(&val) = self.const_values.get(rhs)
                    && val.is_power_of_two()
                {
                    return Some((*lhs, val.trailing_zeros() as u8));
                }
                // Check if lhs is a power of 2 constant
                if let Some(&val) = self.const_values.get(lhs)
                    && val.is_power_of_two()
                {
                    return Some((*rhs, val.trailing_zeros() as u8));
                }
            }
        }
        None
    }

    /// Emit a conditional branch. When `invert` is false, branches if cond != 0.
    /// When `invert` is true, branches if cond == 0.
    pub(super) fn emit_branch_cond(
        &mut self,
        cond: kajit_ir::VReg,
        target: kajit_asm::aarch64::LabelId,
        invert: bool,
    ) {
        if let Some(&cc) = self.fused_cmps.get(&cond) {
            let cc = if invert { cc.invert() } else { cc };
            self.ectx.emit.emit_b_cond_label(cc, target).expect("b.cc");
        } else if let Some((src, bit)) = self.is_and_bit_test(cond) {
            let src_reg = self.reg_for_vreg_with_temp(src, Reg::X16);
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
            let cond_reg = self.reg_for_vreg_with_temp(cond, Reg::X16);
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
    pub(super) fn compute_fusable_cmps(func: &Function) -> HashMap<kajit_ir::VReg, Condition> {
        // Count uses of each vreg across the entire function
        let mut use_counts: HashMap<kajit_ir::VReg, usize> = HashMap::new();

        for block in func.live_blocks() {
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

        for block in func.live_blocks() {
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
                if let LinearOp::BinOp { op, dst, .. } = &inst.op
                    && *dst == cond
                {
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

        fusable
    }

    /// Compute which Or instructions can be replaced with bfi.
    /// Pattern: Or(accum, Shl(And(byte, mask), shift)) where mask has consecutive low bits.
    pub(super) fn compute_fusable_bfis(
        func: &Function,
        const_values: &HashMap<kajit_ir::VReg, u64>,
    ) -> (
        HashMap<kajit_ir::VReg, BfiInfo>,
        std::collections::HashSet<kajit_ir::VReg>,
    ) {
        use std::collections::HashSet;

        // Count uses of each vreg
        let mut use_counts: HashMap<kajit_ir::VReg, usize> = HashMap::new();
        for block in func.live_blocks() {
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
    pub(super) fn compute_fusable_bit_tests(
        func: &Function,
        const_values: &HashMap<kajit_ir::VReg, u64>,
        skip_set: &mut std::collections::HashSet<kajit_ir::VReg>,
    ) {
        // Count uses of each vreg across the entire function
        let mut use_counts: HashMap<kajit_ir::VReg, usize> = HashMap::new();
        for block in func.live_blocks() {
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

    /// Pre-compute base+offset fusions for LoadFromAddr.
    /// When an Add(base, const) result is consumed ONLY by LoadFromAddr,
    /// we can skip the Add and use `[base_reg, #offset]` directly.
    pub(super) fn compute_fusable_addr_offsets(
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
                if op.kind == ir::OperandKind::Use {
                    *use_counts.entry(op.vreg).or_insert(0) += 1;
                }
            }
        }
        for block in func.live_blocks() {
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

        // Find LoadFromAddr whose addr is defined by Add(base, const)
        for inst in &func.insts {
            let addr_vreg = match &inst.op {
                LinearOp::LoadFromAddr { addr, .. } => *addr,
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
}
