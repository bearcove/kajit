use super::*;

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
                dynasm!(self.ectx.ops ; .arch x64 ; mov r10, Rq(enc));
            }
            return;
        }
        if let Some(slot) = alloc.as_stack() {
            let off = self.spill_off(slot) as i32;
            dynasm!(self.ectx.ops ; .arch x64 ; mov r10, [rsp + off]);
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
                dynasm!(self.ectx.ops ; .arch x64 ; mov Rq(enc), r10);
            }
            return;
        }
        if let Some(slot) = alloc.as_stack() {
            let off = self.spill_off(slot) as i32;
            dynasm!(self.ectx.ops ; .arch x64 ; mov [rsp + off], r10);
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
        dynasm!(self.ectx.ops ; .arch x64 ; mov Rq(to_enc), Rq(from_enc));
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
            dynasm!(self.ectx.ops ; .arch x64 ; mov [rsp + off], Rq(enc));
            return;
        }
        panic!("unexpected none allocation for ABI ret capture");
    }

    pub(super) fn push_allocation_operand(&mut self, operand_index: usize, push_count: usize) {
        let alloc = self.current_alloc(operand_index);
        let rsp_adjust = (push_count as i32) * 8;
        if let Some(reg) = alloc.as_reg() {
            let enc = reg.hw_enc() as u8;
            dynasm!(self.ectx.ops ; .arch x64 ; push Rq(enc));
        } else if let Some(slot) = alloc.as_stack() {
            let off = self.spill_off(slot) as i32 + rsp_adjust;
            dynasm!(self.ectx.ops ; .arch x64
                ; mov r10, [rsp + off]
                ; push r10
            );
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
                dynasm!(self.ectx.ops ; .arch x64
                    ; lea r10, [r14 + off]
                    ; push r10
                );
            }
        }
    }
}
