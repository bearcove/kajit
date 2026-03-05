use super::*;
use crate::backends::parallel_moves::{MoveEmitter, emit_parallel_moves};
use kajit_emit::x64::{self, LabelId, Mem};

impl Lowerer {
    pub(super) fn emit_edit_move(&mut self, from: Allocation, to: Allocation) {
        if from == to || from.is_none() || to.is_none() {
            return;
        }
        match (from.as_reg(), from.as_stack(), to.as_reg(), to.as_stack()) {
            (Some(from_reg), None, Some(to_reg), None) => {
                self.emit_mov_preg_to_preg(from_reg, to_reg);
            }
            (Some(from_reg), None, None, Some(to_stack)) => {
                let off = self.spill_off(to_stack) as i32;
                let enc = from_reg.hw_enc() as u8;
                self.ectx
                    .emit
                    .emit_with(|buf| x64::encode_mov_m_r64(Mem { base: 4, disp: off }, enc, buf))
                    .expect("mov");
            }
            (None, Some(from_stack), Some(to_reg), None) => {
                let off = self.spill_off(from_stack) as i32;
                let enc = to_reg.hw_enc() as u8;
                self.ectx
                    .emit
                    .emit_with(|buf| x64::encode_mov_r64_m(enc, Mem { base: 4, disp: off }, buf))
                    .expect("mov");
            }
            (None, Some(from_stack), None, Some(to_stack)) => {
                if from_stack == to_stack {
                    return;
                }
                let from_off = self.spill_off(from_stack) as i32;
                let to_off = self.spill_off(to_stack) as i32;
                self.ectx
                    .emit
                    .emit_with(|buf| {
                        x64::encode_mov_r64_m(
                            10,
                            Mem {
                                base: 4,
                                disp: from_off,
                            },
                            buf,
                        )?;
                        x64::encode_mov_m_r64(
                            Mem {
                                base: 4,
                                disp: to_off,
                            },
                            10,
                            buf,
                        )
                    })
                    .expect("mov");
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

    pub(super) fn edge_edit_moves(
        &self,
        linear_op_index: usize,
        succ_index: usize,
    ) -> Vec<(Allocation, Allocation)> {
        let Some(lambda_id) = self
            .current_func
            .as_ref()
            .map(|f| f.lambda_id.index() as u32)
        else {
            return Vec::new();
        };
        let Some(by_lambda) = self.edge_edits_by_lambda.get(&lambda_id) else {
            return Vec::new();
        };
        let key = (linear_op_index, succ_index);
        let mut moves = Vec::new();
        if let Some(before) = by_lambda.before.get(&key) {
            moves.extend(before.iter().copied());
        }
        if let Some(after) = by_lambda.after.get(&key) {
            moves.extend(after.iter().copied());
        }
        moves
    }

    pub(super) fn apply_fallthrough_edge_edits(
        &mut self,
        linear_op_index: usize,
        succ_index: usize,
    ) {
        let moves = self.edge_edit_moves(linear_op_index, succ_index);
        if moves.is_empty() {
            return;
        }
        emit_parallel_moves(self, &moves);
    }

    pub(super) fn has_edge_edits(&self, linear_op_index: usize, succ_index: usize) -> bool {
        !self.edge_edit_moves(linear_op_index, succ_index).is_empty()
    }

    pub(super) fn edge_target_label(
        &mut self,
        linear_op_index: usize,
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
        let key = (linear_op_index, succ_index);
        let has_edits = by_lambda.before.contains_key(&key) || by_lambda.after.contains_key(&key);
        if !has_edits {
            return actual_target;
        }

        let cache_key = (lambda_id, linear_op_index, succ_index);
        if let Some(label) = self.edge_trampoline_labels.get(&cache_key).copied() {
            return label;
        }

        let moves = self.edge_edit_moves(linear_op_index, succ_index);

        let label = self.ectx.new_label();
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
        self.emit_load_r10_from_allocation(from);
        let tmp_off = self.parallel_move_tmp_base + (tmp_index as u32) * 8;
        self.ectx
            .emit
            .emit_with(|buf| {
                x64::encode_mov_m_r64(
                    Mem {
                        base: 4,
                        disp: tmp_off as i32,
                    },
                    10,
                    buf,
                )
            })
            .expect("mov");
    }

    fn restore_move_tmp_to_dst(&mut self, tmp_index: usize, to: Allocation) {
        let tmp_off = self.parallel_move_tmp_base + (tmp_index as u32) * 8;
        self.ectx
            .emit
            .emit_with(|buf| {
                x64::encode_mov_r64_m(
                    10,
                    Mem {
                        base: 4,
                        disp: tmp_off as i32,
                    },
                    buf,
                )
            })
            .expect("mov");
        self.emit_store_r10_to_allocation(to);
    }
}
