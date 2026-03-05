use super::*;
use kajit_emit::x64::{self, Mem};

impl Lowerer {
    pub(super) fn emit_load_r10_from_allocation(&mut self, alloc: Allocation) {
        if let Some(reg) = alloc.as_reg() {
            assert!(
                reg.class() == RegClass::Int,
                "unsupported register class {:?} for r10 load",
                reg.class()
            );
            let enc = reg.hw_enc() as u8;
            if enc != 10 {
                self.ectx
                    .emit
                    .emit_with(|buf| x64::encode_mov_r64_r64(10, enc, buf))
                    .expect("mov");
            }
            return;
        }
        if let Some(slot) = alloc.as_stack() {
            let off = self.spill_off(slot) as i32;
            self.ectx
                .emit
                .emit_with(|buf| x64::encode_mov_r64_m(10, Mem { base: 4, disp: off }, buf))
                .expect("mov");
            return;
        }
        panic!("unexpected none allocation for r10 load");
    }

    pub(super) fn emit_store_r10_to_allocation(&mut self, alloc: Allocation) {
        if let Some(reg) = alloc.as_reg() {
            assert!(
                reg.class() == RegClass::Int,
                "unsupported register class {:?} for r10 store",
                reg.class()
            );
            let enc = reg.hw_enc() as u8;
            if enc != 10 {
                self.ectx
                    .emit
                    .emit_with(|buf| x64::encode_mov_r64_r64(enc, 10, buf))
                    .expect("mov");
            }
            return;
        }
        if let Some(slot) = alloc.as_stack() {
            let off = self.spill_off(slot) as i32;
            self.ectx
                .emit
                .emit_with(|buf| x64::encode_mov_m_r64(Mem { base: 4, disp: off }, 10, buf))
                .expect("mov");
            return;
        }
        panic!("unexpected none allocation for r10 store");
    }

    pub(super) fn emit_load_use_r10(&mut self, _v: crate::ir::VReg, operand_index: usize) {
        let alloc = self.current_alloc(operand_index);
        self.emit_load_r10_from_allocation(alloc);
    }

    pub(super) fn emit_store_def_r10(&mut self, _v: crate::ir::VReg, operand_index: usize) {
        let alloc = self.current_alloc(operand_index);
        self.emit_store_r10_to_allocation(alloc);
    }

    pub(super) fn emit_mov_preg_to_preg(&mut self, from: PReg, to: PReg) {
        if from == to {
            return;
        }
        assert!(
            from.class() == RegClass::Int && to.class() == RegClass::Int,
            "preg-to-preg move requires Int class"
        );
        let from_enc = from.hw_enc() as u8;
        let to_enc = to.hw_enc() as u8;
        self.ectx
            .emit
            .emit_with(|buf| x64::encode_mov_r64_r64(to_enc, from_enc, buf))
            .expect("mov");
    }

    pub(super) fn emit_capture_abi_ret_to_allocation(
        &mut self,
        abi_reg: PReg,
        operand_index: usize,
    ) {
        let alloc = self.current_alloc(operand_index);
        if let Some(reg) = alloc.as_reg() {
            self.emit_mov_preg_to_preg(abi_reg, reg);
            return;
        }
        if let Some(slot) = alloc.as_stack() {
            let off = self.spill_off(slot) as i32;
            let enc = abi_reg.hw_enc() as u8;
            self.ectx
                .emit
                .emit_with(|buf| x64::encode_mov_m_r64(Mem { base: 4, disp: off }, enc, buf))
                .expect("mov");
            return;
        }
        panic!("unexpected none allocation for ABI ret capture");
    }

    pub(super) fn push_allocation_operand(&mut self, operand_index: usize, push_count: usize) {
        let alloc = self.current_alloc(operand_index);
        let rsp_adjust = (push_count as i32) * 8;
        if let Some(reg) = alloc.as_reg() {
            let enc = reg.hw_enc() as u8;
            self.ectx
                .emit
                .emit_with(|buf| x64::encode_push_r64(enc, buf))
                .expect("push");
        } else if let Some(slot) = alloc.as_stack() {
            let off = self.spill_off(slot) as i32 + rsp_adjust;
            self.ectx
                .emit
                .emit_with(|buf| {
                    x64::encode_mov_r64_m(10, Mem { base: 4, disp: off }, buf)?;
                    x64::encode_push_r64(10, buf)
                })
                .expect("push arg");
        } else {
            panic!("unexpected none allocation for call arg");
        }
    }

    pub(super) fn push_intrinsic_arg(&mut self, arg: IntrinsicArg, push_count: usize) {
        match arg {
            IntrinsicArg::VReg { operand_index } => {
                self.push_allocation_operand(operand_index, push_count);
            }
            IntrinsicArg::OutField(offset) => {
                let off = offset as i32;
                self.ectx
                    .emit
                    .emit_with(|buf| {
                        x64::encode_lea_r64_m(
                            10,
                            Mem {
                                base: 14,
                                disp: off,
                            },
                            buf,
                        )?;
                        x64::encode_push_r64(10, buf)
                    })
                    .expect("outfield");
            }
        }
    }
}
