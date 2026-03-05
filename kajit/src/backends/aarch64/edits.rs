//! Regalloc edit infrastructure: edit moves, edge trampolines, block resolution.

use super::*;
use crate::backends::parallel_moves::{MoveEmitter, emit_parallel_moves};
use kajit_emit::aarch64::{self, Reg};

impl Lowerer {
    pub(super) fn emit_edit_move(&mut self, from: Allocation, to: Allocation) {
        if from == to || from.is_none() || to.is_none() {
            return;
        }

        match (from.as_reg(), from.as_stack(), to.as_reg(), to.as_stack()) {
            (Some(from_reg), None, Some(to_reg), None) => {
                if from_reg == to_reg {
                    return;
                }
                if from_reg.class() != regalloc2::RegClass::Int
                    || to_reg.class() != regalloc2::RegClass::Int
                {
                    return;
                }
                let from_r = from_reg.hw_enc() as u8;
                let to_r = to_reg.hw_enc() as u8;
                self.ectx.emit.emit_word(
                    aarch64::encode_mov_reg(
                        aarch64::Width::X64,
                        Reg::from_raw(to_r),
                        Reg::from_raw(from_r),
                    )
                    .expect("mov"),
                );
            }
            (Some(from_reg), None, None, Some(to_stack)) => {
                let off = self.spill_off(to_stack);
                if self.emit_store_stack_from_preg(from_reg, off) {
                    return;
                }
                if !self.emit_mov_x9_from_preg(from_reg) {
                    return;
                }
                self.ectx.emit.emit_word(
                    aarch64::encode_str_imm(aarch64::Width::X64, Reg::X9, Reg::SP, off)
                        .expect("str"),
                );
            }
            (None, Some(from_stack), Some(to_reg), None) => {
                let off = self.spill_off(from_stack);
                if self.emit_load_preg_from_stack(to_reg, off) {
                    return;
                }
                self.ectx.emit.emit_word(
                    aarch64::encode_ldr_imm(aarch64::Width::X64, Reg::X9, Reg::SP, off)
                        .expect("ldr"),
                );
                let _ = self.emit_mov_preg_from_x9(to_reg);
            }
            (None, Some(from_stack), None, Some(to_stack)) => {
                if from_stack == to_stack {
                    return;
                }
                let from_off = self.spill_off(from_stack);
                let to_off = self.spill_off(to_stack);
                self.ectx.emit.emit_word(
                    aarch64::encode_ldr_imm(aarch64::Width::X64, Reg::X9, Reg::SP, from_off)
                        .expect("ldr"),
                );
                self.ectx.emit.emit_word(
                    aarch64::encode_str_imm(aarch64::Width::X64, Reg::X9, Reg::SP, to_off)
                        .expect("str"),
                );
            }
            _ => {}
        }
    }

    // r[impl ir.regalloc.edits]
    pub(super) fn apply_regalloc_edits(&mut self, linear_op_index: usize, pos: InstPosition) {
        let lambda_id = match self.current_func.as_ref() {
            Some(func) => func.lambda_id.index() as u32,
            None => return,
        };

        let edits = self
            .edits_by_lambda
            .get(&lambda_id)
            .and_then(|by_lambda| match pos {
                InstPosition::Before => by_lambda.before.get(&linear_op_index),
                InstPosition::After => by_lambda.after.get(&linear_op_index),
            })
            .cloned()
            .unwrap_or_default();

        if edits.is_empty() {
            return;
        }
        emit_parallel_moves(self, &edits);
    }

    pub(super) fn resolve_forwarded_block(&self, lambda_id: u32, block_id: BlockId) -> BlockId {
        let mut resolved = block_id;
        let mut hops = 0usize;
        while hops < 64 {
            let Some(&next_block) = self.forward_branch_blocks.get(&(lambda_id, resolved.0)) else {
                break;
            };
            let next_id = BlockId(next_block);
            if next_id == resolved {
                break;
            }
            resolved = next_id;
            hops += 1;
        }
        resolved
    }

    pub(super) fn apply_fallthrough_edge_edits(&mut self, from_block_id: u32, succ_index: usize) {
        let moves = self.edge_edit_moves(from_block_id, succ_index);
        if moves.is_empty() {
            return;
        }
        emit_parallel_moves(self, &moves);
    }

    pub(super) fn has_edge_edits(&self, from_block_id: u32, succ_index: usize) -> bool {
        !self.edge_edit_moves(from_block_id, succ_index).is_empty()
    }

    pub(super) fn edge_target_label(
        &mut self,
        from_block_id: u32,
        succ_index: usize,
        actual_target: LabelId,
    ) -> LabelId {
        let Some(lambda_id) = self
            .current_func
            .as_ref()
            .map(|f| f.lambda_id.index() as u32)
        else {
            return actual_target;
        };
        let Some(by_lambda) = self.edge_edits_by_lambda.get(&lambda_id) else {
            return actual_target;
        };
        let key = (from_block_id, succ_index);
        let has_edits = by_lambda.before.contains_key(&key) || by_lambda.after.contains_key(&key);
        if !has_edits {
            return actual_target;
        }

        let cache_key = (lambda_id, from_block_id, succ_index);
        if let Some(label) = self.edge_trampoline_labels.get(&cache_key).copied() {
            return label;
        }

        let moves = self.edge_edit_moves(from_block_id, succ_index);

        let label = self.new_label_id();
        self.edge_trampoline_labels.insert(cache_key, label);
        self.edge_trampolines.push(EdgeTrampoline {
            label,
            target: actual_target,
            moves,
        });
        label
    }

    pub(super) fn emit_edge_trampolines(&mut self) {
        let trampolines = std::mem::take(&mut self.edge_trampolines);
        for trampoline in trampolines {
            self.ectx.bind_label(trampoline.label);
            emit_parallel_moves(self, &trampoline.moves);
            self.ectx.emit_branch(trampoline.target);
        }
    }
}

impl MoveEmitter for Lowerer {
    fn flush_all_vregs(&mut self) {
        Lowerer::flush_all_vregs(self);
    }

    fn emit_move(&mut self, from: Allocation, to: Allocation) {
        self.emit_edit_move(from, to);
    }

    fn save_move_src_to_tmp(&mut self, tmp_index: usize, from: Allocation) {
        self.emit_load_x9_from_allocation(from);
        let tmp_off = self.no_edit_edge_tmp_base + (tmp_index as u32) * 8;
        self.ectx.emit.emit_word(
            aarch64::encode_str_imm(aarch64::Width::X64, Reg::X9, Reg::SP, tmp_off).expect("str"),
        );
    }

    fn restore_move_tmp_to_dst(&mut self, tmp_index: usize, to: Allocation) {
        let tmp_off = self.no_edit_edge_tmp_base + (tmp_index as u32) * 8;
        self.ectx.emit.emit_word(
            aarch64::encode_ldr_imm(aarch64::Width::X64, Reg::X9, Reg::SP, tmp_off).expect("ldr"),
        );
        let _ = self.emit_store_x9_to_allocation(to);
    }
}
