//! Intrinsic and lambda call emission for aarch64.

use super::*;
use kajit_emit::aarch64::{self, Reg};

impl Lowerer {
    pub(super) fn emit_set_abi_reg_from_intrinsic_arg(&mut self, abi_arg: u8, arg: IntrinsicArg) {
        match arg {
            IntrinsicArg::VReg { operand_index } => {
                self.emit_set_abi_arg_from_allocation(abi_arg, operand_index)
            }
            IntrinsicArg::OutField(offset) => match abi_arg {
                1 => self
                    .ectx
                    .emit
                    .emit_word(
                        aarch64::encode_add_imm(
                            aarch64::Width::X64,
                            Reg::X1,
                            Reg::X21,
                            offset as u16,
                            false,
                        )
                        .expect("add")),
                2 => self
                    .ectx
                    .emit
                    .emit_word(
                        aarch64::encode_add_imm(
                            aarch64::Width::X64,
                            Reg::X2,
                            Reg::X21,
                            offset as u16,
                            false,
                        )
                        .expect("add")),
                3 => self
                    .ectx
                    .emit
                    .emit_word(
                        aarch64::encode_add_imm(
                            aarch64::Width::X64,
                            Reg::X3,
                            Reg::X21,
                            offset as u16,
                            false,
                        )
                        .expect("add")),
                4 => self
                    .ectx
                    .emit
                    .emit_word(
                        aarch64::encode_add_imm(
                            aarch64::Width::X64,
                            Reg::X4,
                            Reg::X21,
                            offset as u16,
                            false,
                        )
                        .expect("add")),
                5 => self
                    .ectx
                    .emit
                    .emit_word(
                        aarch64::encode_add_imm(
                            aarch64::Width::X64,
                            Reg::X5,
                            Reg::X21,
                            offset as u16,
                            false,
                        )
                        .expect("add")),
                6 => self
                    .ectx
                    .emit
                    .emit_word(
                        aarch64::encode_add_imm(
                            aarch64::Width::X64,
                            Reg::X6,
                            Reg::X21,
                            offset as u16,
                            false,
                        )
                        .expect("add")),
                7 => self
                    .ectx
                    .emit
                    .emit_word(
                        aarch64::encode_add_imm(
                            aarch64::Width::X64,
                            Reg::X7,
                            Reg::X21,
                            offset as u16,
                            false,
                        )
                        .expect("add")),
                _ => unreachable!("unsupported intrinsic ABI arg register x{abi_arg}"),
            },
        }
    }

    pub(super) fn emit_call_intrinsic_with_args(
        &mut self,
        fn_ptr: *const u8,
        args: &[IntrinsicArg],
    ) {
        use crate::context::{CTX_ERROR_CODE, CTX_INPUT_PTR};

        if args.len() > 7 {
            panic!(
                "unsupported CallIntrinsic arity in linear backend adapter: {} args (+ctx)",
                args.len()
            );
        }

        let error_exit = self
            .current_func
            .as_ref()
            .expect("CallIntrinsic outside function")
            .error_exit;

        self.flush_all_vregs();

        self.ectx.emit.emit_word(
            aarch64::encode_str_imm(
                aarch64::Width::X64,
                Reg::X19,
                Reg::X22,
                CTX_INPUT_PTR as u32,
            )
            .expect("str"),
        );
        self.ectx.emit.emit_word(
            aarch64::encode_mov_reg(aarch64::Width::X64, Reg::X0, Reg::X22).expect("mov"),
        );

        for (i, arg) in args.iter().copied().enumerate() {
            self.emit_set_abi_reg_from_intrinsic_arg((i + 1) as u8, arg);
        }

        let ptr = fn_ptr as u64;
        let p0 = (ptr & 0xFFFF) as u32;
        let p1 = ((ptr >> 16) & 0xFFFF) as u32;
        let p2 = ((ptr >> 32) & 0xFFFF) as u32;
        let p3 = ((ptr >> 48) & 0xFFFF) as u32;
        self.ectx
            .emit
            .emit_word(aarch64::encode_movz(aarch64::Width::X64, Reg::X16, p0 as u16, 0).expect("movz"));
        if p1 != 0 {
            self.ectx.emit.emit_word(
                aarch64::encode_movk(aarch64::Width::X64, Reg::X16, p1 as u16, 16).expect("movk"),
            );
        }
        if p2 != 0 {
            self.ectx.emit.emit_word(
                aarch64::encode_movk(aarch64::Width::X64, Reg::X16, p2 as u16, 32).expect("movk"),
            );
        }
        if p3 != 0 {
            self.ectx.emit.emit_word(
                aarch64::encode_movk(aarch64::Width::X64, Reg::X16, p3 as u16, 48).expect("movk"),
            );
        }
        self.ectx
            .emit
            .emit_word(aarch64::encode_blr(Reg::X16).expect("blr"));

        self.ectx.emit.emit_word(
            aarch64::encode_ldr_imm(
                aarch64::Width::X64,
                Reg::X19,
                Reg::X22,
                CTX_INPUT_PTR as u32,
            )
            .expect("ldr"),
        );
        self.ectx.emit.emit_word(
            aarch64::encode_ldr_imm(
                aarch64::Width::W32,
                Reg::X9,
                Reg::X22,
                CTX_ERROR_CODE as u32,
            )
            .expect("ldr"),
        );
        self.ectx
            .emit
            .emit_cbnz_label(aarch64::Width::W32, Reg::X9, error_exit)
            .expect("cbnz");
    }

    // r[impl ir.intrinsics]
    pub(super) fn emit_call_intrinsic(
        &mut self,
        func: crate::ir::IntrinsicFn,
        args: &[crate::ir::VReg],
        dst: Option<crate::ir::VReg>,
        field_offset: u32,
    ) {
        let fn_ptr = func.0 as *const u8;
        match dst {
            Some(dst) => {
                let dst_operand_index = args.len();
                let call_args: Vec<IntrinsicArg> = args
                    .iter()
                    .copied()
                    .enumerate()
                    .map(|(i, _vreg)| IntrinsicArg::VReg { operand_index: i })
                    .collect();
                self.emit_call_intrinsic_with_args(fn_ptr, &call_args);
                self.ectx.emit.emit_word(
                    aarch64::encode_mov_reg(aarch64::Width::X64, Reg::X9, Reg::X0).expect("mov"),
                );
                self.emit_store_def_x9(dst, dst_operand_index);
            }
            None => {
                let mut call_args: Vec<IntrinsicArg> = args
                    .iter()
                    .copied()
                    .enumerate()
                    .map(|(i, _vreg)| IntrinsicArg::VReg { operand_index: i })
                    .collect();
                call_args.push(IntrinsicArg::OutField(field_offset));
                self.emit_call_intrinsic_with_args(fn_ptr, &call_args);
            }
        }
    }

    pub(super) fn emit_call_pure_with_args(&mut self, fn_ptr: *const u8, args: &[IntrinsicArg]) {
        if args.len() > 8 {
            panic!(
                "unsupported CallPure arity in linear backend adapter: {} args",
                args.len()
            );
        }

        self.flush_all_vregs();

        for (i, arg) in args.iter().copied().enumerate() {
            self.emit_set_abi_reg_from_intrinsic_arg(i as u8, arg);
        }

        let ptr = fn_ptr as u64;
        let p0 = (ptr & 0xFFFF) as u32;
        let p1 = ((ptr >> 16) & 0xFFFF) as u32;
        let p2 = ((ptr >> 32) & 0xFFFF) as u32;
        let p3 = ((ptr >> 48) & 0xFFFF) as u32;
        self.ectx
            .emit
            .emit_word(aarch64::encode_movz(aarch64::Width::X64, Reg::X16, p0 as u16, 0).expect("movz"));
        if p1 != 0 {
            self.ectx.emit.emit_word(
                aarch64::encode_movk(aarch64::Width::X64, Reg::X16, p1 as u16, 16).expect("movk"),
            );
        }
        if p2 != 0 {
            self.ectx.emit.emit_word(
                aarch64::encode_movk(aarch64::Width::X64, Reg::X16, p2 as u16, 32).expect("movk"),
            );
        }
        if p3 != 0 {
            self.ectx.emit.emit_word(
                aarch64::encode_movk(aarch64::Width::X64, Reg::X16, p3 as u16, 48).expect("movk"),
            );
        }
        self.ectx
            .emit
            .emit_word(aarch64::encode_blr(Reg::X16).expect("blr"));
    }

    pub(super) fn emit_call_pure(
        &mut self,
        func: crate::ir::IntrinsicFn,
        args: &[crate::ir::VReg],
        dst: crate::ir::VReg,
    ) {
        let fn_ptr = func.0 as *const u8;
        let dst_operand_index = args.len();
        let call_args: Vec<IntrinsicArg> = args
            .iter()
            .copied()
            .enumerate()
            .map(|(i, _)| IntrinsicArg::VReg { operand_index: i })
            .collect();
        self.emit_call_pure_with_args(fn_ptr, &call_args);
        self.emit_capture_abi_ret_to_allocation(0, dst_operand_index);
        let _ = dst;
    }

    pub(super) fn emit_store_incoming_lambda_args(&mut self, data_args: &[crate::ir::VReg]) {
        const MAX_LAMBDA_DATA_ARGS: usize = 6;
        if data_args.len() > MAX_LAMBDA_DATA_ARGS {
            panic!(
                "aarch64 CallLambda supports at most {MAX_LAMBDA_DATA_ARGS} data args, got {}",
                data_args.len()
            );
        }
    }

    pub(super) fn emit_load_lambda_results_to_ret_regs(
        &mut self,
        lambda_id: crate::ir::LambdaId,
        data_results: &[crate::ir::VReg],
    ) {
        if data_results.len() > 2 {
            panic!(
                "aarch64 CallLambda supports at most 2 data results, got {}",
                data_results.len()
            );
        }

        let result_allocs = self
            .return_result_allocs_by_lambda
            .get(&(lambda_id.index() as u32))
            .cloned()
            .unwrap_or_default();
        assert!(
            result_allocs.len() >= data_results.len(),
            "missing return allocation mapping for lambda {:?}: need {}, got {}",
            lambda_id,
            data_results.len(),
            result_allocs.len()
        );

        if let Some(&alloc) = result_allocs.first() {
            self.emit_load_x9_from_allocation(alloc);
            self.ectx.emit.emit_word(
                aarch64::encode_mov_reg(aarch64::Width::X64, Reg::X0, Reg::X9).expect("mov"),
            );
        }
        if let Some(&alloc) = result_allocs.get(1) {
            self.emit_load_x9_from_allocation(alloc);
            self.ectx.emit.emit_word(
                aarch64::encode_mov_reg(aarch64::Width::X64, Reg::X1, Reg::X9).expect("mov"),
            );
        }
    }

    pub(super) fn emit_call_lambda(
        &mut self,
        label: LabelId,
        args: &[crate::ir::VReg],
        results: &[crate::ir::VReg],
    ) {
        use crate::context::{CTX_ERROR_CODE, CTX_INPUT_PTR};

        const MAX_LAMBDA_DATA_ARGS: usize = 6;
        if args.len() > MAX_LAMBDA_DATA_ARGS {
            panic!(
                "aarch64 CallLambda supports at most {MAX_LAMBDA_DATA_ARGS} data args, got {}",
                args.len()
            );
        }
        if results.len() > 2 {
            panic!(
                "aarch64 CallLambda supports at most 2 data results, got {}",
                results.len()
            );
        }

        let error_exit = self
            .current_func
            .as_ref()
            .expect("CallLambda outside function")
            .error_exit;
        self.flush_all_vregs();
        self.ectx.emit.emit_word(
            aarch64::encode_str_imm(
                aarch64::Width::X64,
                Reg::X19,
                Reg::X22,
                CTX_INPUT_PTR as u32,
            )
            .expect("str"),
        );
        self.ectx.emit.emit_word(
            aarch64::encode_mov_reg(aarch64::Width::X64, Reg::X0, Reg::X21).expect("mov"),
        );
        self.ectx.emit.emit_word(
            aarch64::encode_mov_reg(aarch64::Width::X64, Reg::X1, Reg::X22).expect("mov"),
        );

        for (i, &arg) in args.iter().enumerate() {
            let _ = arg;
            self.emit_set_abi_arg_from_allocation((i + 2) as u8, i);
        }

        self.ectx.emit.emit_bl_label(label).expect("bl");
        self.ectx.emit.emit_word(
            aarch64::encode_ldr_imm(
                aarch64::Width::X64,
                Reg::X19,
                Reg::X22,
                CTX_INPUT_PTR as u32,
            )
            .expect("ldr"),
        );
        self.ectx.emit.emit_word(
            aarch64::encode_ldr_imm(
                aarch64::Width::W32,
                Reg::X9,
                Reg::X22,
                CTX_ERROR_CODE as u32,
            )
            .expect("ldr"),
        );
        self.ectx
            .emit
            .emit_cbnz_label(aarch64::Width::W32, Reg::X9, error_exit)
            .expect("cbnz");

        if let Some(&dst) = results.first() {
            let _ = dst;
            self.emit_capture_abi_ret_to_allocation(0, args.len());
        }
        if let Some(&dst) = results.get(1) {
            let _ = dst;
            self.emit_capture_abi_ret_to_allocation(1, args.len() + 1);
        }
    }
}
