//! Control flow emission for aarch64 regalloc3 backend.

use kajit_emit::aarch64::Reg;
use kajit_mir::cfg_mir::{self, Terminator};

use super::context::EmitContext;

impl<'a> EmitContext<'a> {
    /// Resolve a block ID through trampoline aliases.
    /// If `block_id` is a trampoline (no insts, Branch terminator) whose
    /// outgoing edge carries no value moves, follow the chain to the final
    /// non-trampoline target.
    pub(super) fn resolve_trampoline(&self, mut block_id: cfg_mir::BlockId) -> cfg_mir::BlockId {
        for _ in 0..16 {
            let block = &self.func.blocks[block_id.index()];
            if !block.insts.is_empty() {
                break;
            }
            let term = &self.func.terms[block.term.0 as usize];
            if let Terminator::Branch { edge } = term {
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

    /// Emit a terminator. `next_block` is the block that follows in emission order (for fallthrough elision).
    pub(super) fn emit_terminator(
        &mut self,
        term: &Terminator,
        next_block: Option<cfg_mir::BlockId>,
    ) {
        match term {
            Terminator::Return => {
                // Elide the branch only when the success epilogue is the next
                // emitted code. Edge trampolines are emitted after the block
                // stream, so a "last" return block cannot safely fall through
                // when any trampoline labels were materialized.
                if !self.is_last_emitted_block || !self.edge_trampoline_labels.is_empty() {
                    let success_exit = self.success_exit;
                    self.ectx
                        .emit
                        .emit_b_label(success_exit)
                        .expect("b success");
                }
            }

            Terminator::Branch { edge } => {
                let target_block = self.func.edges[edge.index()].to;
                // Resolve through trampolines for fallthrough elision.
                let resolved = self.resolve_trampoline(target_block);
                if self.edge_has_moves(*edge) {
                    let trampoline =
                        self.edge_target_label(*edge, self.block_labels[&target_block]);
                    self.ectx.emit.emit_b_label(trampoline).expect("branch");
                } else if Some(resolved) != next_block {
                    let label = self.block_labels[&target_block];
                    self.ectx.emit.emit_b_label(label).expect("branch");
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

                // Resolve through trampolines for fallthrough elision.
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
                                .emit_b_label(fallthrough_label)
                                .expect("b fallthrough");
                        }
                    } else if Some(resolved_fall) != next_block {
                        self.ectx
                            .emit
                            .emit_b_label(fallthrough_label)
                            .expect("b fallthrough");
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
                    // BranchIfZero inverted = BranchIf → emit non-inverted branch to fallthrough
                    self.emit_branch_cond(*cond, fallthrough_label, false);
                    self.emit_edge_moves(*taken);
                } else {
                    // Normal: BranchIfZero = branch to taken when zero
                    self.emit_branch_cond(*cond, taken_label, true);
                    if self.edge_has_moves(*fallthrough) {
                        self.emit_edge_moves(*fallthrough);
                        if Some(resolved_fall) != next_block {
                            self.ectx
                                .emit
                                .emit_b_label(fallthrough_label)
                                .expect("b fallthrough");
                        }
                    } else if Some(resolved_fall) != next_block {
                        self.ectx
                            .emit
                            .emit_b_label(fallthrough_label)
                            .expect("b fallthrough");
                    }
                }
            }

            Terminator::ErrorExit { code } => {
                self.ectx.emit_error_with_ctx_reg(*code, self.ctx_reg);
            }

            Terminator::JumpTable { .. } => {
                panic!("JumpTable not yet supported in regalloc3 backend");
            }
        }
    }

    /// Emit all blocks for this function.
    pub(super) fn emit_function(&mut self) {
        // Create labels for all blocks
        for block in self.func.live_blocks() {
            let label = self.ectx.new_label();
            self.block_labels.insert(block.id, label);
        }

        // Alias trampoline blocks only when they are pure control-flow aliases.
        // If the outgoing edge carries block-param moves, the block must remain
        // materialized so its branch can target the edge trampoline.
        for block in self.func.live_blocks() {
            if block.dead || !block.insts.is_empty() {
                continue;
            }
            let term = &self.func.terms[block.term.0 as usize];
            if let Terminator::Branch { edge } = term {
                if self.edge_has_moves(*edge) {
                    continue;
                }
                let target_block = self.func.edges[edge.index()].to;
                let from_label = self.block_labels[&block.id];
                let to_label = self.block_labels[&target_block];
                self.ectx.emit.alias_label(from_label, to_label);
            }
        }

        // Build emission order: all non-Return blocks first, then Return blocks.
        // This allows the last Return block to fall through into the success epilogue.
        let mut emit_order: Vec<usize> = Vec::new();
        let mut return_blocks: Vec<usize> = Vec::new();
        for block_idx in 0..self.func.blocks.len() {
            let block = &self.func.blocks[block_idx];
            if block.dead {
                continue;
            }
            // Skip only pure alias trampoline blocks (aliased above, no code to emit).
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

        // Emit each block in the computed order
        for (emit_idx, &block_idx) in emit_order.iter().enumerate() {
            let block = &self.func.blocks[block_idx];

            // Detect if this is the last block in emission order.
            self.is_last_emitted_block = emit_idx == emit_order.len() - 1;

            // Bind label for this block (except entry which comes after prologue)
            if block.id.0 != 0 {
                let label = self.block_labels[&block.id];
                self.ectx.bind_label(label);
            }

            // Emit instructions with source location tracking
            for &inst_id in &block.insts {
                // Emit OperandEdits (register moves) required before this instruction
                // to satisfy fixed-register operand constraints.
                // Collect all edits for this instruction and emit as a parallel move.
                let edits_here: Vec<(Reg, Reg)> = self
                    .alloc_func
                    .edits
                    .iter()
                    .filter(|e| e.before_inst == inst_id)
                    .map(|e| (Reg::from_raw(e.to.0), Reg::from_raw(e.from.0)))
                    .collect();
                if !edits_here.is_empty() {
                    self.emit_parallel_moves(&edits_here, Reg::X16);
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

            // Find next block in emission order (for fallthrough elision)
            let next_block_id = emit_order
                .get(emit_idx + 1)
                .map(|&idx| self.func.blocks[idx].id);

            // Emit terminator with source location
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
