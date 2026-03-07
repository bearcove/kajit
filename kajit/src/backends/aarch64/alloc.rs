//! Register/allocation load, store, and move helpers for aarch64.

use super::*;
use kajit_emit::aarch64::{self, Reg};

impl Lowerer {
    fn scratch_reg_avoiding(&self, a: Reg, b: Option<Reg>) -> Reg {
        for reg in [Reg::X9, Reg::X10, Reg::X16] {
            if reg != a && Some(reg) != b {
                return reg;
            }
        }
        panic!("no scratch register available");
    }

    pub(super) fn emit_load_u64_reg(&mut self, rd: Reg, value: u64) {
        let p0 = (value & 0xFFFF) as u32;
        let p1 = ((value >> 16) & 0xFFFF) as u32;
        let p2 = ((value >> 32) & 0xFFFF) as u32;
        let p3 = ((value >> 48) & 0xFFFF) as u32;
        self.ectx
            .emit
            .emit_word(aarch64::encode_movz(aarch64::Width::X64, rd, p0 as u16, 0).expect("movz"));
        if p1 != 0 {
            self.ectx.emit.emit_word(
                aarch64::encode_movk(aarch64::Width::X64, rd, p1 as u16, 16).expect("movk"),
            );
        }
        if p2 != 0 {
            self.ectx.emit.emit_word(
                aarch64::encode_movk(aarch64::Width::X64, rd, p2 as u16, 32).expect("movk"),
            );
        }
        if p3 != 0 {
            self.ectx.emit.emit_word(
                aarch64::encode_movk(aarch64::Width::X64, rd, p3 as u16, 48).expect("movk"),
            );
        }
    }

    fn emit_add_imm_any(&mut self, rd: Reg, rn: Reg, imm: u32) {
        if imm <= 0x0fff {
            self.ectx.emit.emit_word(
                aarch64::encode_add_imm(aarch64::Width::X64, rd, rn, imm as u16, false)
                    .expect("add"),
            );
            return;
        }
        if (imm & 0x0fff) == 0 {
            let shifted = imm >> 12;
            if shifted <= 0x0fff {
                self.ectx.emit.emit_word(
                    aarch64::encode_add_imm(aarch64::Width::X64, rd, rn, shifted as u16, true)
                        .expect("add"),
                );
                return;
            }
        }
        let scratch = self.scratch_reg_avoiding(rd, Some(rn));
        self.emit_load_u64_reg(scratch, imm as u64);
        self.ectx
            .emit
            .emit_word(aarch64::encode_add_reg(aarch64::Width::X64, rd, rn, scratch).expect("add"));
    }

    pub(super) fn emit_stack_addr(&mut self, rd: Reg, off: u32) {
        self.emit_add_imm_any(rd, Reg::SP, off);
    }

    pub(super) fn emit_stack_load(&mut self, width: aarch64::Width, rd: Reg, off: u32) {
        if let Ok(word) = aarch64::encode_ldr_imm(width, rd, Reg::SP, off) {
            self.ectx.emit.emit_word(word);
            return;
        }
        let addr_reg = self.scratch_reg_avoiding(rd, None);
        self.emit_stack_addr(addr_reg, off);
        self.ectx
            .emit
            .emit_word(aarch64::encode_ldr_imm(width, rd, addr_reg, 0).expect("ldr"));
    }

    pub(super) fn emit_stack_store(&mut self, width: aarch64::Width, rs: Reg, off: u32) {
        if let Ok(word) = aarch64::encode_str_imm(width, rs, Reg::SP, off) {
            self.ectx.emit.emit_word(word);
            return;
        }
        let addr_reg = self.scratch_reg_avoiding(rs, None);
        self.emit_stack_addr(addr_reg, off);
        self.ectx
            .emit
            .emit_word(aarch64::encode_str_imm(width, rs, addr_reg, 0).expect("str"));
    }

    pub(super) fn emit_mov_x9_from_preg(&mut self, preg: regalloc2::PReg) -> bool {
        if preg.class() != regalloc2::RegClass::Int {
            return false;
        }
        let r = preg.hw_enc() as u8;
        if r == 9 {
            return true;
        }
        self.ectx.emit.emit_word(
            aarch64::encode_mov_reg(aarch64::Width::X64, Reg::X9, Reg::from_raw(r)).expect("mov"),
        );
        true
    }

    pub(super) fn emit_mov_preg_from_x9(&mut self, preg: regalloc2::PReg) -> bool {
        if preg.class() != regalloc2::RegClass::Int {
            return false;
        }
        let r = preg.hw_enc() as u8;
        if r == 9 {
            return true;
        }
        self.ectx.emit.emit_word(
            aarch64::encode_mov_reg(aarch64::Width::X64, Reg::from_raw(r), Reg::X9).expect("mov"),
        );
        true
    }

    pub(super) fn emit_mov_preg_to_preg(&mut self, from: PReg, to: PReg) -> bool {
        if from == to {
            return true;
        }
        if from.class() != RegClass::Int || to.class() != RegClass::Int {
            return false;
        }
        let from_r = from.hw_enc() as u8;
        let to_r = to.hw_enc() as u8;
        self.ectx.emit.emit_word(
            aarch64::encode_mov_reg(
                aarch64::Width::X64,
                Reg::from_raw(to_r),
                Reg::from_raw(from_r),
            )
            .expect("mov"),
        );
        true
    }

    pub(super) fn emit_store_stack_from_preg(&mut self, preg: regalloc2::PReg, off: u32) -> bool {
        if preg.class() != regalloc2::RegClass::Int {
            return false;
        }
        let r = preg.hw_enc() as u8;
        self.emit_stack_store(aarch64::Width::X64, Reg::from_raw(r), off);
        true
    }

    pub(super) fn emit_load_preg_from_stack(&mut self, preg: regalloc2::PReg, off: u32) -> bool {
        if preg.class() != regalloc2::RegClass::Int {
            return false;
        }
        let r = preg.hw_enc() as u8;
        self.emit_stack_load(aarch64::Width::X64, Reg::from_raw(r), off);
        true
    }

    pub(super) fn emit_load_x9_from_allocation(&mut self, alloc: Allocation) {
        if let Some(reg) = alloc.as_reg() {
            assert!(
                self.emit_mov_x9_from_preg(reg),
                "unsupported register allocation class {:?} for x9 load",
                reg.class()
            );
            return;
        }
        if let Some(slot) = alloc.as_stack() {
            let off = self.spill_off(slot);
            self.emit_stack_load(aarch64::Width::X64, Reg::X9, off);
            return;
        }
        panic!("unexpected none allocation for x9 load");
    }

    pub(super) fn emit_load_x10_from_allocation(&mut self, alloc: Allocation) {
        if let Some(reg) = alloc.as_reg() {
            assert!(
                reg.class() == regalloc2::RegClass::Int,
                "unsupported register allocation class {:?} for x10 load",
                reg.class()
            );
            let r = reg.hw_enc() as u8;
            if r != 10 {
                self.ectx.emit.emit_word(
                    aarch64::encode_mov_reg(aarch64::Width::X64, Reg::X10, Reg::from_raw(r))
                        .expect("mov"),
                );
            }
            return;
        }
        if let Some(slot) = alloc.as_stack() {
            let off = self.spill_off(slot);
            self.emit_stack_load(aarch64::Width::X64, Reg::X10, off);
            return;
        }
        panic!("unexpected none allocation for x10 load");
    }

    pub(super) fn emit_store_x9_to_allocation(&mut self, alloc: Allocation) -> bool {
        if let Some(reg) = alloc.as_reg() {
            return self.emit_mov_preg_from_x9(reg);
        }
        if let Some(slot) = alloc.as_stack() {
            let off = self.spill_off(slot);
            self.emit_stack_store(aarch64::Width::X64, Reg::X9, off);
            return true;
        }
        false
    }

    pub(super) fn emit_load_use_x9(&mut self, v: crate::ir::VReg, operand_index: usize) {
        let _ = v;
        let alloc = self.current_alloc(operand_index);
        self.emit_load_x9_from_allocation(alloc);
    }

    pub(super) fn emit_load_use_x10(&mut self, v: crate::ir::VReg, operand_index: usize) {
        let _ = v;
        let alloc = self.current_alloc(operand_index);
        self.emit_load_x10_from_allocation(alloc);
    }

    pub(super) fn emit_store_def_x9(&mut self, _v: crate::ir::VReg, operand_index: usize) {
        let alloc = self.current_alloc(operand_index);
        let _ = self.emit_store_x9_to_allocation(alloc);
    }

    pub(super) fn emit_set_abi_arg_from_allocation(&mut self, abi_arg: u8, operand_index: usize) {
        let alloc = self.current_alloc(operand_index);
        let target = PReg::new(abi_arg as usize, RegClass::Int);
        if let Some(reg) = alloc.as_reg() {
            assert!(
                self.emit_mov_preg_to_preg(reg, target),
                "unsupported register allocation class {:?} for CallLambda arg",
                reg.class()
            );
            return;
        }
        if let Some(slot) = alloc.as_stack() {
            let off = self.spill_off(slot);
            let target_r = target.hw_enc() as u8;
            self.emit_stack_load(aarch64::Width::X64, Reg::from_raw(target_r), off);
            return;
        }
        panic!("unexpected none allocation for CallLambda arg");
    }

    pub(super) fn emit_capture_abi_ret_to_allocation(&mut self, abi_ret: u8, operand_index: usize) {
        let alloc = self.current_alloc(operand_index);
        let source = PReg::new(abi_ret as usize, RegClass::Int);
        if let Some(reg) = alloc.as_reg() {
            assert!(
                self.emit_mov_preg_to_preg(source, reg),
                "unsupported register allocation class {:?} for CallLambda result",
                reg.class()
            );
            return;
        }
        if let Some(slot) = alloc.as_stack() {
            let off = self.spill_off(slot);
            let _ = self.emit_store_stack_from_preg(source, off);
            return;
        }
        panic!("unexpected none allocation for CallLambda result");
    }

    pub(super) fn emit_load_u32_w10(&mut self, value: u32) {
        let lo = value & 0xFFFF;
        let hi = (value >> 16) & 0xFFFF;
        self.ectx.emit.emit_word(
            aarch64::encode_movz(aarch64::Width::W32, Reg::X10, lo as u16, 0).expect("movz"),
        );
        if value > 0xFFFF {
            self.ectx.emit.emit_word(
                aarch64::encode_movk(aarch64::Width::W32, Reg::X10, hi as u16, 16).expect("movk"),
            );
        }
    }

    pub(super) fn emit_load_u64_x10(&mut self, value: u64) {
        self.emit_load_u64_reg(Reg::X10, value);
    }

    pub(super) fn emit_load_u64_x9(&mut self, value: u64) {
        self.emit_load_u64_reg(Reg::X9, value);
    }
}
