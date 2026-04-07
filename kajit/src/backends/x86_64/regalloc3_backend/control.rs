//! Control flow emission for x86_64 regalloc3 backend.

use kajit_emit::x64;
use kajit_mir::cfg_mir::{self, Terminator};

use super::context::EmitContext;
use super::inst::invert_condition;

const R10: u8 = 10;

impl<'a> EmitContext<'a> {
    /// Emit a conditional branch.
    /// When `invert` is false, branches if cond != 0.
    /// When `invert` is true, branches if cond == 0.
    pub fn emit_branch_cond(
        &mut self,
        cond: kajit_ir::VReg,
        target: kajit_emit::x64::LabelId,
        invert: bool,
    ) {
        // Fused cmp: the cmp was already emitted, just emit jcc.
        if let Some(&cc) = self.fused_cmps.get(&cond) {
            let cc = if invert { invert_condition(cc) } else { cc };
            self.ectx.emit.emit_jcc_label(target, cc).expect("jcc");
            return;
        }

        // General case: test + jz/jnz.
        let cond_enc = self.reg_for_vreg_with_temp(cond, R10);
        self.ectx
            .emit
            .emit_with(|buf| x64::encode_test_r64_r64(cond_enc, cond_enc, buf))
            .expect("test");
        if invert {
            self.ectx.emit.emit_jz_label(target).expect("jz");
        } else {
            self.ectx.emit.emit_jnz_label(target).expect("jnz");
        }
    }

    /// Emit a terminator. `next_block` is the block that follows in emission order.
    pub fn emit_terminator(&mut self, term: &Terminator, next_block: Option<cfg_mir::BlockId>) {
        match term {
            Terminator::Return => {
                if !self.is_last_emitted_block || !self.edge_trampoline_labels.is_empty() {
                    let success_exit = self.success_exit;
                    self.ectx
                        .emit
                        .emit_jmp_label(success_exit)
                        .expect("jmp success");
                }
            }

            Terminator::Branch { edge } => {
                let target_block = self.func.edges[edge.index()].to;
                let resolved = self.resolve_trampoline(target_block);
                if self.edge_has_moves(*edge) {
                    let trampoline =
                        self.edge_target_label(*edge, self.block_labels[&target_block]);
                    self.ectx.emit.emit_jmp_label(trampoline).expect("jmp");
                } else if Some(resolved) != next_block {
                    let label = self.block_labels[&target_block];
                    self.ectx.emit.emit_jmp_label(label).expect("jmp");
                }
            }

            Terminator::BranchIf {
                cond,
                taken,
                fallthrough,
            } => {
                let taken_block = self.func.edges[taken.index()].to;
                let fallthrough_block = self.func.edges[fallthrough.index()].to;
                let taken_label = self.block_labels[&taken_block];
                let fallthrough_label = self.block_labels[&fallthrough_block];

                let resolved_taken = self.resolve_trampoline(taken_block);
                let resolved_fall = self.resolve_trampoline(fallthrough_block);
                let invert =
                    Some(resolved_taken) == next_block && Some(resolved_fall) != next_block;
                let taken_label = self.edge_target_label(*taken, taken_label);
                let fallthrough_label = self.edge_target_label(*fallthrough, fallthrough_label);

                if invert {
                    self.emit_branch_cond(*cond, fallthrough_label, true);
                    self.emit_edge_moves(*taken);
                } else {
                    self.emit_branch_cond(*cond, taken_label, false);
                    if self.edge_has_moves(*fallthrough) {
                        self.emit_edge_moves(*fallthrough);
                        if Some(resolved_fall) != next_block {
                            self.ectx
                                .emit
                                .emit_jmp_label(fallthrough_label)
                                .expect("jmp fallthrough");
                        }
                    } else if Some(resolved_fall) != next_block {
                        self.ectx
                            .emit
                            .emit_jmp_label(fallthrough_label)
                            .expect("jmp fallthrough");
                    }
                }
            }

            Terminator::BranchIfZero {
                cond,
                taken,
                fallthrough,
            } => {
                let taken_block = self.func.edges[taken.index()].to;
                let fallthrough_block = self.func.edges[fallthrough.index()].to;
                let taken_label = self.block_labels[&taken_block];
                let fallthrough_label = self.block_labels[&fallthrough_block];

                let resolved_taken = self.resolve_trampoline(taken_block);
                let resolved_fall = self.resolve_trampoline(fallthrough_block);
                let invert =
                    Some(resolved_taken) == next_block && Some(resolved_fall) != next_block;
                let taken_label = self.edge_target_label(*taken, taken_label);
                let fallthrough_label = self.edge_target_label(*fallthrough, fallthrough_label);

                if invert {
                    // BranchIfZero inverted → branch to fallthrough when non-zero.
                    self.emit_branch_cond(*cond, fallthrough_label, false);
                    self.emit_edge_moves(*taken);
                } else {
                    // Normal: branch to taken when zero.
                    self.emit_branch_cond(*cond, taken_label, true);
                    if self.edge_has_moves(*fallthrough) {
                        self.emit_edge_moves(*fallthrough);
                        if Some(resolved_fall) != next_block {
                            self.ectx
                                .emit
                                .emit_jmp_label(fallthrough_label)
                                .expect("jmp fallthrough");
                        }
                    } else if Some(resolved_fall) != next_block {
                        self.ectx
                            .emit
                            .emit_jmp_label(fallthrough_label)
                            .expect("jmp fallthrough");
                    }
                }
            }

            Terminator::JumpTable {
                predicate, targets, ..
            } => {
                let pred_enc = self.reg_for_vreg_with_temp(*predicate, R10);
                if pred_enc != R10 {
                    self.ectx
                        .emit
                        .emit_with(|buf| x64::encode_mov_r64_r64(R10, pred_enc, buf))
                        .expect("mov");
                }
                // Zero-extend to 32-bit.
                self.ectx
                    .emit
                    .emit_with(|buf| x64::encode_mov_r32_r32(R10, R10, buf))
                    .expect("mov32");

                // Emit cmp + je chain. Last target is default (fallthrough jmp).
                let last = targets.len() - 1;
                for (index, edge_id) in targets.iter().enumerate() {
                    let edge = &self.func.edges[edge_id.index()];
                    let target_block = edge.to;
                    let resolved = self.resolve_trampoline(target_block);
                    let label = self.edge_target_label(*edge_id, self.block_labels[&resolved]);

                    if index < last {
                        self.ectx
                            .emit
                            .emit_with(|buf| x64::encode_cmp_r64_imm32(R10, index as u32, buf))
                            .expect("cmp");
                        self.ectx.emit.emit_je_label(label).expect("je");
                    } else {
                        self.ectx.emit.emit_jmp_label(label).expect("jmp default");
                    }
                }
            }
        }
    }

    /// Emit all blocks for this function.
    pub fn emit_function(&mut self) {
        // Create labels for all blocks.
        for block in &self.func.blocks {
            let label = self.ectx.new_label();
            self.block_labels.insert(block.id, label);
        }

        // Build emission order: non-Return blocks first, then Return blocks.
        let mut emit_order: Vec<usize> = Vec::new();
        let mut return_blocks: Vec<usize> = Vec::new();
        for block_idx in 0..self.func.blocks.len() {
            let block = &self.func.blocks[block_idx];
            if block.dead {
                continue;
            }
            // Skip pure trampoline blocks (no insts, plain Branch, no edge moves).
            if block.insts.is_empty() {
                let term = &self.func.terms[block.term.0 as usize];
                if let Terminator::Branch { edge } = term
                    && !self.edge_has_moves(*edge)
                {
                    continue;
                }
            }
            let term = &self.func.terms[block.term.0 as usize];
            if matches!(term, Terminator::Return) {
                return_blocks.push(block_idx);
            } else {
                emit_order.push(block_idx);
            }
        }
        emit_order.extend(return_blocks);

        // Emit each block.
        for (emit_idx, &block_idx) in emit_order.iter().enumerate() {
            let block = &self.func.blocks[block_idx];

            self.is_last_emitted_block = emit_idx == emit_order.len() - 1;

            // Bind label (except entry block which comes after prologue).
            if block.id.0 != 0 {
                let label = self.block_labels[&block.id];
                self.ectx.bind_label(label);
            }

            // Emit instructions with source location tracking.
            for &inst_id in &block.insts {
                // Emit OperandEdits (register moves) before this instruction.
                let edits_here: Vec<(u8, u8)> = self
                    .alloc_func
                    .edits
                    .iter()
                    .filter(|e| e.before_inst == inst_id)
                    .map(|e| (e.to.0, e.from.0))
                    .collect();
                if !edits_here.is_empty() {
                    self.emit_parallel_moves(&edits_here, R10);
                }

                let op_id = kajit_mir::cfg_mir::OpId::Inst(inst_id);
                if let Some(&line) = self.line_map.get(&op_id) {
                    self.ectx.set_source_location(kajit_emit::SourceLocation {
                        file: 1,
                        line,
                        column: 0,
                    });
                }
                let inst = &self.func.insts[inst_id.index()];
                self.emit_inst(inst);
            }

            // Emit terminator.
            let next_block_id = emit_order
                .get(emit_idx + 1)
                .map(|&idx| self.func.blocks[idx].id);

            let term_op = kajit_mir::cfg_mir::OpId::Term(block.term);
            if let Some(&line) = self.line_map.get(&term_op) {
                self.ectx.set_source_location(kajit_emit::SourceLocation {
                    file: 1,
                    line,
                    column: 0,
                });
            }
            let term = &self.func.terms[block.term.0 as usize];
            self.emit_terminator(term, next_block_id);
        }

        self.emit_edge_trampolines();
    }
}
