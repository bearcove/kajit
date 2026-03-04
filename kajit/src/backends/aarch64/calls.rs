//! Intrinsic and lambda call emission for aarch64.

use super::*;

impl Lowerer {
    pub(super) fn emit_set_abi_reg_from_intrinsic_arg(&mut self, abi_arg: u8, arg: IntrinsicArg) {
        match arg {
            IntrinsicArg::VReg { operand_index } => {
                self.emit_set_abi_arg_from_allocation(abi_arg, operand_index)
            }
            IntrinsicArg::OutField(offset) => match abi_arg {
                1 => dynasm!(self.ectx.ops ; .arch aarch64 ; add x1, x21, #offset),
                2 => dynasm!(self.ectx.ops ; .arch aarch64 ; add x2, x21, #offset),
                3 => dynasm!(self.ectx.ops ; .arch aarch64 ; add x3, x21, #offset),
                4 => dynasm!(self.ectx.ops ; .arch aarch64 ; add x4, x21, #offset),
                5 => dynasm!(self.ectx.ops ; .arch aarch64 ; add x5, x21, #offset),
                6 => dynasm!(self.ectx.ops ; .arch aarch64 ; add x6, x21, #offset),
                7 => dynasm!(self.ectx.ops ; .arch aarch64 ; add x7, x21, #offset),
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

        dynasm!(self.ectx.ops
            ; .arch aarch64
            ; str x19, [x22, #CTX_INPUT_PTR]
            ; mov x0, x22
        );

        for (i, arg) in args.iter().copied().enumerate() {
            self.emit_set_abi_reg_from_intrinsic_arg((i + 1) as u8, arg);
        }

        let ptr = fn_ptr as u64;
        let p0 = (ptr & 0xFFFF) as u32;
        let p1 = ((ptr >> 16) & 0xFFFF) as u32;
        let p2 = ((ptr >> 32) & 0xFFFF) as u32;
        let p3 = ((ptr >> 48) & 0xFFFF) as u32;
        dynasm!(self.ectx.ops ; .arch aarch64 ; movz x16, #p0);
        if p1 != 0 {
            dynasm!(self.ectx.ops ; .arch aarch64 ; movk x16, #p1, LSL #16);
        }
        if p2 != 0 {
            dynasm!(self.ectx.ops ; .arch aarch64 ; movk x16, #p2, LSL #32);
        }
        if p3 != 0 {
            dynasm!(self.ectx.ops ; .arch aarch64 ; movk x16, #p3, LSL #48);
        }
        dynasm!(self.ectx.ops ; .arch aarch64 ; blr x16);

        dynasm!(self.ectx.ops
            ; .arch aarch64
            ; ldr x19, [x22, #CTX_INPUT_PTR]
            ; ldr w9, [x22, #CTX_ERROR_CODE]
            ; cbnz w9, =>error_exit
        );
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
                dynasm!(self.ectx.ops ; .arch aarch64 ; mov x9, x0);
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
        dynasm!(self.ectx.ops ; .arch aarch64 ; movz x16, #p0);
        if p1 != 0 {
            dynasm!(self.ectx.ops ; .arch aarch64 ; movk x16, #p1, LSL #16);
        }
        if p2 != 0 {
            dynasm!(self.ectx.ops ; .arch aarch64 ; movk x16, #p2, LSL #32);
        }
        if p3 != 0 {
            dynasm!(self.ectx.ops ; .arch aarch64 ; movk x16, #p3, LSL #48);
        }
        dynasm!(self.ectx.ops ; .arch aarch64 ; blr x16);
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
            dynasm!(self.ectx.ops ; .arch aarch64 ; mov x0, x9);
        }
        if let Some(&alloc) = result_allocs.get(1) {
            self.emit_load_x9_from_allocation(alloc);
            dynasm!(self.ectx.ops ; .arch aarch64 ; mov x1, x9);
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
        dynasm!(self.ectx.ops
            ; .arch aarch64
            ; str x19, [x22, #CTX_INPUT_PTR]
        );
        dynasm!(self.ectx.ops
            ; .arch aarch64
            ; mov x0, x21
            ; mov x1, x22
        );

        for (i, &arg) in args.iter().enumerate() {
            let _ = arg;
            self.emit_set_abi_arg_from_allocation((i + 2) as u8, i);
        }

        dynasm!(self.ectx.ops ; .arch aarch64 ; bl =>label);
        dynasm!(self.ectx.ops
            ; .arch aarch64
            ; ldr x19, [x22, #CTX_INPUT_PTR]
            ; ldr w9, [x22, #CTX_ERROR_CODE]
            ; cbnz w9, =>error_exit
        );

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
