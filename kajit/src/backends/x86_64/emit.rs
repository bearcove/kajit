use super::*;

impl Lowerer {
    pub(super) fn emit_read_from_field(&mut self, dst: crate::ir::VReg, offset: u32, width: Width) {
        let off = offset as i32;
        match width {
            Width::W1 => dynasm!(self.ectx.ops ; .arch x64 ; movzx r10d, BYTE [r14 + off]),
            Width::W2 => dynasm!(self.ectx.ops ; .arch x64 ; movzx r10d, WORD [r14 + off]),
            Width::W4 => dynasm!(self.ectx.ops ; .arch x64 ; mov r10d, DWORD [r14 + off]),
            Width::W8 => dynasm!(self.ectx.ops ; .arch x64 ; mov r10, QWORD [r14 + off]),
        }
        self.emit_store_def_r10(dst, 0);
        self.set_const(dst, None);
    }

    pub(super) fn emit_write_to_field(&mut self, src: crate::ir::VReg, offset: u32, width: Width) {
        self.emit_load_use_r10(src, 0);
        let off = offset as i32;
        match width {
            Width::W1 => dynasm!(self.ectx.ops ; .arch x64 ; mov BYTE [r14 + off], r10b),
            Width::W2 => dynasm!(self.ectx.ops ; .arch x64 ; mov WORD [r14 + off], r10w),
            Width::W4 => dynasm!(self.ectx.ops ; .arch x64 ; mov DWORD [r14 + off], r10d),
            Width::W8 => dynasm!(self.ectx.ops ; .arch x64 ; mov QWORD [r14 + off], r10),
        }
    }

    pub(super) fn emit_save_out_ptr(&mut self, dst: crate::ir::VReg) {
        dynasm!(self.ectx.ops ; .arch x64 ; mov r10, r14);
        self.emit_store_def_r10(dst, 0);
        self.set_const(dst, None);
    }

    pub(super) fn emit_set_out_ptr(&mut self, src: crate::ir::VReg) {
        self.emit_load_use_r10(src, 0);
        dynasm!(self.ectx.ops ; .arch x64 ; mov r14, r10);
    }

    pub(super) fn emit_slot_addr(&mut self, dst: crate::ir::VReg, slot: crate::ir::SlotId) {
        let slot_off = self.slot_off(slot) as i32;
        dynasm!(self.ectx.ops ; .arch x64 ; lea r10, [rsp + slot_off]);
        self.emit_store_def_r10(dst, 0);
        self.set_const(dst, None);
    }

    pub(super) fn emit_read_bytes(&mut self, dst: crate::ir::VReg, count: u32) {
        self.emit_recipe_ops(vec![Op::BoundsCheck { count }]);
        match count {
            1 => dynasm!(self.ectx.ops ; .arch x64 ; movzx r10d, BYTE [r12]),
            2 => dynasm!(self.ectx.ops ; .arch x64 ; movzx r10d, WORD [r12]),
            4 => dynasm!(self.ectx.ops ; .arch x64 ; mov r10d, DWORD [r12]),
            8 => dynasm!(self.ectx.ops ; .arch x64 ; mov r10, QWORD [r12]),
            _ => panic!("unsupported ReadBytes count: {count}"),
        }
        self.emit_store_def_r10(dst, 0);
        self.set_const(dst, None);
        self.ectx.emit_advance_cursor_by(count);
    }

    pub(super) fn emit_peek_byte(&mut self, dst: crate::ir::VReg) {
        self.emit_recipe_ops(vec![Op::BoundsCheck { count: 1 }]);
        dynasm!(self.ectx.ops ; .arch x64 ; movzx r10d, BYTE [r12]);
        self.emit_store_def_r10(dst, 0);
        self.set_const(dst, None);
    }

    pub(super) fn emit_binop(
        &mut self,
        kind: BinOpKind,
        dst: crate::ir::VReg,
        lhs: crate::ir::VReg,
        _rhs: crate::ir::VReg,
    ) {
        self.emit_load_use_r10(lhs, 0);
        match kind {
            BinOpKind::Add => {
                let rhs_alloc = self.current_alloc(1);
                if let Some(reg) = rhs_alloc.as_reg() {
                    let enc = reg.hw_enc() as u8;
                    dynasm!(self.ectx.ops ; .arch x64 ; add r10, Rq(enc));
                } else if let Some(slot) = rhs_alloc.as_stack() {
                    let off = self.spill_off(slot) as i32;
                    dynasm!(self.ectx.ops ; .arch x64 ; add r10, [rsp + off]);
                }
            }
            BinOpKind::Sub => {
                let rhs_alloc = self.current_alloc(1);
                if let Some(reg) = rhs_alloc.as_reg() {
                    let enc = reg.hw_enc() as u8;
                    dynasm!(self.ectx.ops ; .arch x64 ; sub r10, Rq(enc));
                } else if let Some(slot) = rhs_alloc.as_stack() {
                    let off = self.spill_off(slot) as i32;
                    dynasm!(self.ectx.ops ; .arch x64 ; sub r10, [rsp + off]);
                }
            }
            BinOpKind::And => {
                let rhs_alloc = self.current_alloc(1);
                if let Some(reg) = rhs_alloc.as_reg() {
                    let enc = reg.hw_enc() as u8;
                    dynasm!(self.ectx.ops ; .arch x64 ; and r10, Rq(enc));
                } else if let Some(slot) = rhs_alloc.as_stack() {
                    let off = self.spill_off(slot) as i32;
                    dynasm!(self.ectx.ops ; .arch x64 ; and r10, [rsp + off]);
                }
            }
            BinOpKind::Or => {
                let rhs_alloc = self.current_alloc(1);
                if let Some(reg) = rhs_alloc.as_reg() {
                    let enc = reg.hw_enc() as u8;
                    dynasm!(self.ectx.ops ; .arch x64 ; or r10, Rq(enc));
                } else if let Some(slot) = rhs_alloc.as_stack() {
                    let off = self.spill_off(slot) as i32;
                    dynasm!(self.ectx.ops ; .arch x64 ; or r10, [rsp + off]);
                }
            }
            BinOpKind::Xor => {
                let rhs_alloc = self.current_alloc(1);
                if let Some(reg) = rhs_alloc.as_reg() {
                    let enc = reg.hw_enc() as u8;
                    dynasm!(self.ectx.ops ; .arch x64 ; xor r10, Rq(enc));
                } else if let Some(slot) = rhs_alloc.as_stack() {
                    let off = self.spill_off(slot) as i32;
                    dynasm!(self.ectx.ops ; .arch x64 ; xor r10, [rsp + off]);
                }
            }
            BinOpKind::CmpNe => {
                let rhs_alloc = self.current_alloc(1);
                if let Some(reg) = rhs_alloc.as_reg() {
                    let enc = reg.hw_enc() as u8;
                    dynasm!(self.ectx.ops ; .arch x64 ; cmp r10, Rq(enc));
                } else if let Some(slot) = rhs_alloc.as_stack() {
                    let off = self.spill_off(slot) as i32;
                    dynasm!(self.ectx.ops ; .arch x64 ; cmp r10, [rsp + off]);
                }
                dynasm!(self.ectx.ops
                    ; .arch x64
                    ; setne r10b
                    ; movzx r10, r10b
                );
            }
            BinOpKind::Shr => {
                dynasm!(self.ectx.ops ; .arch x64 ; shr r10, cl);
            }
            BinOpKind::Shl => {
                dynasm!(self.ectx.ops ; .arch x64 ; shl r10, cl);
            }
        }
        self.emit_store_def_r10(dst, 2);
        self.set_const(dst, None);
    }

    pub(super) fn emit_unary(
        &mut self,
        kind: UnaryOpKind,
        dst: crate::ir::VReg,
        src: crate::ir::VReg,
    ) {
        self.emit_load_use_r10(src, 0);
        match kind {
            UnaryOpKind::ZigzagDecode { wide: true } => {
                dynasm!(self.ectx.ops
                    ; .arch x64
                    ; mov r11, r10
                    ; shr r11, 1
                    ; and r10, 1
                    ; neg r10
                    ; xor r10, r11
                );
            }
            UnaryOpKind::ZigzagDecode { wide: false } => {
                dynasm!(self.ectx.ops
                    ; .arch x64
                    ; mov r11d, r10d
                    ; shr r11d, 1
                    ; and r10d, 1
                    ; neg r10d
                    ; xor r10d, r11d
                );
            }
            UnaryOpKind::SignExtend { from_width } => match from_width {
                Width::W1 => dynasm!(self.ectx.ops ; .arch x64 ; movsx r10, r10b),
                Width::W2 => dynasm!(self.ectx.ops ; .arch x64 ; movsx r10, r10w),
                Width::W4 => dynasm!(self.ectx.ops ; .arch x64 ; movsxd r10, r10d),
                Width::W8 => {}
            },
        }
        self.emit_store_def_r10(dst, 1);
        self.set_const(dst, None);
    }

    pub(super) fn emit_branch_if(
        &mut self,
        cond: crate::ir::VReg,
        target: DynamicLabel,
        invert: bool,
    ) {
        let _ = cond;
        let alloc = self.current_alloc(0);
        self.emit_branch_if_allocation(alloc, target, invert);
    }

    pub(super) fn emit_branch_if_allocation(
        &mut self,
        alloc: Allocation,
        target: DynamicLabel,
        invert: bool,
    ) {
        if let Some(reg) = alloc.as_reg() {
            assert!(
                reg.class() == RegClass::Int,
                "unsupported register class {:?} for branch condition",
                reg.class()
            );
            let enc = reg.hw_enc() as u8;
            dynasm!(self.ectx.ops ; .arch x64 ; test Rq(enc), Rq(enc));
            if invert {
                dynasm!(self.ectx.ops ; .arch x64 ; jz =>target);
            } else {
                dynasm!(self.ectx.ops ; .arch x64 ; jnz =>target);
            }
            return;
        }
        if let Some(slot) = alloc.as_stack() {
            let off = self.spill_off(slot) as i32;
            dynasm!(self.ectx.ops ; .arch x64 ; mov r10, [rsp + off]);
            dynasm!(self.ectx.ops ; .arch x64 ; test r10, r10);
            if invert {
                dynasm!(self.ectx.ops ; .arch x64 ; jz =>target);
            } else {
                dynasm!(self.ectx.ops ; .arch x64 ; jnz =>target);
            }
            return;
        }
        panic!("unexpected none allocation for branch condition");
    }

    pub(super) fn emit_jump_table(
        &mut self,
        lambda_id: u32,
        predicate: crate::ir::VReg,
        term: &RaTerminator,
        linear_op_index: usize,
    ) {
        let RaTerminator::JumpTable {
            targets, default, ..
        } = term
        else {
            unreachable!();
        };
        let _ = predicate;
        let alloc = self.current_alloc(0);
        if let Some(reg) = alloc.as_reg() {
            let enc = reg.hw_enc() as u8;
            dynasm!(self.ectx.ops ; .arch x64 ; mov r10, Rq(enc));
        } else if let Some(slot) = alloc.as_stack() {
            let off = self.spill_off(slot) as i32;
            dynasm!(self.ectx.ops ; .arch x64 ; mov r10, [rsp + off]);
        } else {
            panic!("unexpected none allocation for jumptable predicate");
        }
        for (index, target_block) in targets.iter().enumerate() {
            let resolved = self.resolve_forwarded_block(lambda_id, *target_block);
            let target_label = self.edge_target_label(
                linear_op_index,
                index,
                self.block_label(lambda_id, resolved),
            );
            dynasm!(self.ectx.ops
                ; .arch x64
                ; cmp r10d, index as i32
                ; je =>target_label
            );
        }
        let default_succ_index = targets.len();
        let resolved_default = self.resolve_forwarded_block(lambda_id, *default);
        let default_target = self.edge_target_label(
            linear_op_index,
            default_succ_index,
            self.block_label(lambda_id, resolved_default),
        );
        self.ectx.emit_branch(default_target);
    }
}
