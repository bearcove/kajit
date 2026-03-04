use super::*;
use kajit_emit::x64::{self, LabelId, Mem};

impl Lowerer {
    pub(super) fn emit_call_intrinsic_with_args(
        &mut self,
        fn_ptr: *const u8,
        args: &[IntrinsicArg],
    ) {
        use crate::context::{CTX_ERROR_CODE, CTX_INPUT_PTR};

        let error_exit = self
            .current_func
            .as_ref()
            .expect("CallIntrinsic outside function")
            .error_exit;

        self.flush_all_vregs();

        self.ectx
            .emit
            .emit_with(|buf| {
                x64::encode_mov_m_r64(
                    Mem {
                        base: 15,
                        disp: CTX_INPUT_PTR as i32,
                    },
                    12,
                    buf,
                )
            })
            .expect("mov");

        #[cfg(not(windows))]
        {
            if args.len() > 5 {
                panic!(
                    "unsupported CallIntrinsic arity in linear backend adapter: {} args (+ctx)",
                    args.len()
                );
            }
            let abi_regs: &[PReg] = &[
                PReg::new(6, RegClass::Int), // rsi
                PReg::new(2, RegClass::Int), // rdx
                PReg::new(1, RegClass::Int), // rcx
                PReg::new(8, RegClass::Int), // r8
                PReg::new(9, RegClass::Int), // r9
            ];
            for (i, arg) in args.iter().copied().enumerate() {
                self.push_intrinsic_arg(arg, i);
            }
            self.ectx
                .emit
                .emit_with(|buf| x64::encode_mov_r64_r64(7, 15, buf))
                .expect("mov");
            for i in (0..args.len()).rev() {
                let enc = abi_regs[i].hw_enc() as u8;
                self.ectx
                    .emit
                    .emit_with(|buf| x64::encode_pop_r64(enc, buf))
                    .expect("pop");
            }
        }

        #[cfg(windows)]
        {
            if args.len() > 3 {
                panic!(
                    "unsupported CallIntrinsic arity in linear backend adapter: {} args (+ctx)",
                    args.len()
                );
            }
            let abi_regs: &[PReg] = &[
                PReg::new(2, RegClass::Int), // rdx
                PReg::new(8, RegClass::Int), // r8
                PReg::new(9, RegClass::Int), // r9
            ];
            for (i, arg) in args.iter().copied().enumerate() {
                self.push_intrinsic_arg(arg, i);
            }
            self.ectx
                .emit
                .emit_with(|buf| x64::encode_mov_r64_r64(1, 15, buf))
                .expect("mov");
            for i in (0..args.len()).rev() {
                let enc = abi_regs[i].hw_enc() as u8;
                self.ectx
                    .emit
                    .emit_with(|buf| x64::encode_pop_r64(enc, buf))
                    .expect("pop");
            }
        }

        let ptr_val = fn_ptr as u64;
        self.ectx
            .emit
            .emit_with(|buf| {
                x64::encode_mov_r64_imm64(0, ptr_val, buf)?;
                x64::encode_call_r64(0, buf)
            })
            .expect("call");

        self.ectx
            .emit
            .emit_with(|buf| {
                x64::encode_mov_r64_m(
                    12,
                    Mem {
                        base: 15,
                        disp: CTX_INPUT_PTR as i32,
                    },
                    buf,
                )?;
                x64::encode_mov_r32_m(
                    10,
                    Mem {
                        base: 15,
                        disp: CTX_ERROR_CODE as i32,
                    },
                    buf,
                )?;
                x64::encode_test_r32_r32(10, 10, buf)
            })
            .expect("reload");
        self.ectx
            .emit
            .emit_jnz_label(error_exit)
            .expect("jnz error_exit");
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
                let rax = PReg::new(0, RegClass::Int);
                self.emit_capture_abi_ret_to_allocation(rax, dst_operand_index);
                let _ = dst;
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
        self.flush_all_vregs();

        #[cfg(not(windows))]
        {
            if args.len() > 6 {
                panic!(
                    "unsupported CallPure arity in linear backend adapter: {} args",
                    args.len()
                );
            }
            let abi_regs: &[PReg] = &[
                PReg::new(7, RegClass::Int), // rdi
                PReg::new(6, RegClass::Int), // rsi
                PReg::new(2, RegClass::Int), // rdx
                PReg::new(1, RegClass::Int), // rcx
                PReg::new(8, RegClass::Int), // r8
                PReg::new(9, RegClass::Int), // r9
            ];
            for (i, arg) in args.iter().copied().enumerate() {
                self.push_intrinsic_arg(arg, i);
            }
            for i in (0..args.len()).rev() {
                let enc = abi_regs[i].hw_enc() as u8;
                self.ectx
                    .emit
                    .emit_with(|buf| x64::encode_pop_r64(enc, buf))
                    .expect("pop");
            }
        }

        #[cfg(windows)]
        {
            if args.len() > 4 {
                panic!(
                    "unsupported CallPure arity in linear backend adapter: {} args",
                    args.len()
                );
            }
            let abi_regs: &[PReg] = &[
                PReg::new(1, RegClass::Int), // rcx
                PReg::new(2, RegClass::Int), // rdx
                PReg::new(8, RegClass::Int), // r8
                PReg::new(9, RegClass::Int), // r9
            ];
            for (i, arg) in args.iter().copied().enumerate() {
                self.push_intrinsic_arg(arg, i);
            }
            for i in (0..args.len()).rev() {
                let enc = abi_regs[i].hw_enc() as u8;
                self.ectx
                    .emit
                    .emit_with(|buf| x64::encode_pop_r64(enc, buf))
                    .expect("pop");
            }
        }

        let ptr_val = fn_ptr as u64;
        self.ectx
            .emit
            .emit_with(|buf| {
                x64::encode_mov_r64_imm64(0, ptr_val, buf)?;
                x64::encode_call_r64(0, buf)
            })
            .expect("call");
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
        let rax = PReg::new(0, RegClass::Int);
        self.emit_capture_abi_ret_to_allocation(rax, dst_operand_index);
        let _ = dst;
    }

    pub(super) fn emit_store_incoming_lambda_args(&mut self, data_args: &[crate::ir::VReg]) {
        #[cfg(not(windows))]
        const MAX_LAMBDA_DATA_ARGS: usize = 4;
        #[cfg(windows)]
        const MAX_LAMBDA_DATA_ARGS: usize = 2;

        if data_args.len() > MAX_LAMBDA_DATA_ARGS {
            panic!(
                "x64 CallLambda supports at most {MAX_LAMBDA_DATA_ARGS} data args, got {}",
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
                "x64 CallLambda supports at most 2 data results, got {}",
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

        let rax = PReg::new(0, RegClass::Int);
        let rdx = PReg::new(2, RegClass::Int);
        if let Some(&alloc) = result_allocs.first() {
            self.emit_load_r10_from_allocation(alloc);
            self.emit_mov_preg_to_preg(PReg::new(10, RegClass::Int), rax);
        }
        if let Some(&alloc) = result_allocs.get(1) {
            self.emit_load_r10_from_allocation(alloc);
            self.emit_mov_preg_to_preg(PReg::new(10, RegClass::Int), rdx);
        }
    }

    pub(super) fn emit_call_lambda(
        &mut self,
        label: LabelId,
        args: &[crate::ir::VReg],
        results: &[crate::ir::VReg],
    ) {
        use crate::context::{CTX_ERROR_CODE, CTX_INPUT_PTR};

        #[cfg(not(windows))]
        const MAX_LAMBDA_DATA_ARGS: usize = 4;
        #[cfg(windows)]
        const MAX_LAMBDA_DATA_ARGS: usize = 2;
        if args.len() > MAX_LAMBDA_DATA_ARGS {
            panic!(
                "x64 CallLambda supports at most {MAX_LAMBDA_DATA_ARGS} data args, got {}",
                args.len()
            );
        }
        if results.len() > 2 {
            panic!(
                "x64 CallLambda supports at most 2 data results, got {}",
                results.len()
            );
        }

        let error_exit = self
            .current_func
            .as_ref()
            .expect("CallLambda outside function")
            .error_exit;
        self.flush_all_vregs();
        self.ectx
            .emit
            .emit_with(|buf| {
                x64::encode_mov_m_r64(
                    Mem {
                        base: 15,
                        disp: CTX_INPUT_PTR as i32,
                    },
                    12,
                    buf,
                )
            })
            .expect("mov");

        #[cfg(not(windows))]
        {
            let abi_data_regs: &[PReg] = &[
                PReg::new(2, RegClass::Int), // rdx
                PReg::new(1, RegClass::Int), // rcx
                PReg::new(8, RegClass::Int), // r8
                PReg::new(9, RegClass::Int), // r9
            ];
            for (i, &_arg) in args.iter().enumerate() {
                self.push_allocation_operand(i, i);
            }
            self.ectx
                .emit
                .emit_with(|buf| {
                    x64::encode_lea_r64_m(7, Mem { base: 14, disp: 0 }, buf)?;
                    x64::encode_mov_r64_r64(6, 15, buf)
                })
                .expect("arg setup");
            for i in (0..args.len()).rev() {
                let enc = abi_data_regs[i].hw_enc() as u8;
                self.ectx
                    .emit
                    .emit_with(|buf| x64::encode_pop_r64(enc, buf))
                    .expect("pop");
            }
        }
        #[cfg(windows)]
        {
            let abi_data_regs: &[PReg] = &[
                PReg::new(8, RegClass::Int), // r8
                PReg::new(9, RegClass::Int), // r9
            ];
            for (i, &_arg) in args.iter().enumerate() {
                self.push_allocation_operand(i, i);
            }
            self.ectx
                .emit
                .emit_with(|buf| {
                    x64::encode_lea_r64_m(1, Mem { base: 14, disp: 0 }, buf)?;
                    x64::encode_mov_r64_r64(2, 15, buf)
                })
                .expect("arg setup");
            for i in (0..args.len()).rev() {
                let enc = abi_data_regs[i].hw_enc() as u8;
                self.ectx
                    .emit
                    .emit_with(|buf| x64::encode_pop_r64(enc, buf))
                    .expect("pop");
            }
        }

        self.ectx
            .emit
            .emit_call_label(label)
            .expect("call label");
        self.ectx
            .emit
            .emit_with(|buf| {
                x64::encode_mov_r64_m(
                    12,
                    Mem {
                        base: 15,
                        disp: CTX_INPUT_PTR as i32,
                    },
                    buf,
                )?;
                x64::encode_mov_r32_m(
                    10,
                    Mem {
                        base: 15,
                        disp: CTX_ERROR_CODE as i32,
                    },
                    buf,
                )?;
                x64::encode_test_r32_r32(10, 10, buf)
            })
            .expect("reload");
        self.ectx
            .emit
            .emit_jnz_label(error_exit)
            .expect("jnz error_exit");

        let rax = PReg::new(0, RegClass::Int);
        let rdx = PReg::new(2, RegClass::Int);
        if let Some(&_dst) = results.first() {
            self.emit_capture_abi_ret_to_allocation(rax, args.len());
        }
        if let Some(&_dst) = results.get(1) {
            self.emit_capture_abi_ret_to_allocation(rdx, args.len() + 1);
        }
    }
}
