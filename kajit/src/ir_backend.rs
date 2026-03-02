#![allow(clippy::useless_conversion)]

use dynasmrt::AssemblyOffset;

use crate::linearize::LinearIr;

pub struct LinearBackendResult {
    pub buf: dynasmrt::ExecutableBuffer,
    pub entry: AssemblyOffset,
}

pub fn compile_linear_ir(ir: &LinearIr) -> LinearBackendResult {
    let alloc = crate::regalloc_engine::allocate_linear_ir(ir)
        .unwrap_or_else(|err| panic!("regalloc2 allocation failed: {err}"));
    compile_linear_ir_with_alloc(ir, &alloc)
}

pub fn compile_linear_ir_with_alloc(
    ir: &LinearIr,
    alloc: &crate::regalloc_engine::AllocatedProgram,
) -> LinearBackendResult {
    let max_spillslots = alloc
        .functions
        .iter()
        .map(|f| f.num_spillslots)
        .max()
        .unwrap_or(0);

    #[cfg(target_arch = "x86_64")]
    {
        compile_linear_ir_x64(ir, max_spillslots)
    }

    #[cfg(target_arch = "aarch64")]
    {
        compile_linear_ir_aarch64(ir, max_spillslots, alloc)
    }
}

#[cfg(target_arch = "x86_64")]
fn compile_linear_ir_x64(ir: &LinearIr, _max_spillslots: usize) -> LinearBackendResult {
    use dynasmrt::{DynamicLabel, DynasmApi, DynasmLabelApi, dynasm};

    use crate::arch::{BASE_FRAME, EmitCtx};
    use crate::ir::Width;
    use crate::linearize::{BinOpKind, LabelId, LinearOp, UnaryOpKind};
    use crate::recipe::{Op, Recipe, Slot};

    struct FunctionCtx {
        error_exit: DynamicLabel,
        data_results: Vec<crate::ir::VReg>,
    }

    struct Lowerer {
        ectx: EmitCtx,
        labels: Vec<DynamicLabel>,
        lambda_labels: Vec<DynamicLabel>,
        slot_base: u32,
        vreg_base: u32,
        entry: Option<AssemblyOffset>,
        current_func: Option<FunctionCtx>,
        x9_cached_vreg: Option<crate::ir::VReg>,
        const_vregs: Vec<Option<u64>>,
    }

    #[derive(Clone, Copy)]
    enum IntrinsicArg {
        VReg(crate::ir::VReg),
        OutField(u32),
        OutStack(u32),
    }

    impl Lowerer {
        fn new(ir: &LinearIr) -> Self {
            let slot_base = BASE_FRAME;
            let slot_bytes = ir.slot_count * 8;
            let vreg_base = slot_base + slot_bytes;
            let vreg_bytes = ir.vreg_count * 8;
            let extra_stack = slot_bytes + vreg_bytes + 8;

            let mut ectx = EmitCtx::new(extra_stack);

            let labels: Vec<DynamicLabel> = (0..ir.label_count).map(|_| ectx.new_label()).collect();

            let mut lambda_max = 0usize;
            for op in &ir.ops {
                match op {
                    LinearOp::FuncStart { lambda_id, .. } => {
                        lambda_max = lambda_max.max(lambda_id.index() as usize);
                    }
                    LinearOp::CallLambda { target, .. } => {
                        lambda_max = lambda_max.max(target.index() as usize);
                    }
                    _ => {}
                }
            }
            let lambda_labels: Vec<DynamicLabel> =
                (0..=lambda_max).map(|_| ectx.new_label()).collect();

            Self {
                ectx,
                labels,
                lambda_labels,
                slot_base,
                vreg_base,
                entry: None,
                current_func: None,
                x9_cached_vreg: None,
                const_vregs: vec![None; ir.vreg_count as usize],
            }
        }

        fn vreg_off(&self, v: crate::ir::VReg) -> u32 {
            self.vreg_base + (v.index() as u32) * 8
        }

        fn slot_off(&self, s: crate::ir::SlotId) -> u32 {
            self.slot_base + (s.index() as u32) * 8
        }

        fn label(&self, label: LabelId) -> DynamicLabel {
            self.labels[label.index() as usize]
        }

        fn emit_recipe_ops(&mut self, ops: Vec<Op>) {
            self.ectx.emit_recipe(&Recipe {
                ops,
                label_count: 0,
            });
        }

        fn emit_load_vreg_r10(&mut self, v: crate::ir::VReg) {
            let off = self.vreg_off(v) as i32;
            dynasm!(self.ectx.ops
                ; .arch x64
                ; mov r10, [rsp + off]
            );
        }

        fn emit_store_r10_to_vreg(&mut self, v: crate::ir::VReg) {
            let off = self.vreg_off(v) as i32;
            dynasm!(self.ectx.ops
                ; .arch x64
                ; mov [rsp + off], r10
            );
        }

        fn emit_read_from_field(&mut self, dst: crate::ir::VReg, offset: u32, width: Width) {
            let off = offset as i32;
            match width {
                Width::W1 => dynasm!(self.ectx.ops ; .arch x64 ; movzx r10d, BYTE [r14 + off]),
                Width::W2 => dynasm!(self.ectx.ops ; .arch x64 ; movzx r10d, WORD [r14 + off]),
                Width::W4 => dynasm!(self.ectx.ops ; .arch x64 ; mov r10d, DWORD [r14 + off]),
                Width::W8 => dynasm!(self.ectx.ops ; .arch x64 ; mov r10, QWORD [r14 + off]),
            }
            self.emit_store_r10_to_vreg(dst);
        }

        fn emit_write_to_field(&mut self, src: crate::ir::VReg, offset: u32, width: Width) {
            self.emit_load_vreg_r10(src);
            let off = offset as i32;
            match width {
                Width::W1 => dynasm!(self.ectx.ops ; .arch x64 ; mov BYTE [r14 + off], r10b),
                Width::W2 => dynasm!(self.ectx.ops ; .arch x64 ; mov WORD [r14 + off], r10w),
                Width::W4 => dynasm!(self.ectx.ops ; .arch x64 ; mov DWORD [r14 + off], r10d),
                Width::W8 => dynasm!(self.ectx.ops ; .arch x64 ; mov QWORD [r14 + off], r10),
            }
        }

        fn emit_save_out_ptr(&mut self, dst: crate::ir::VReg) {
            let off = self.vreg_off(dst) as i32;
            dynasm!(self.ectx.ops
                ; .arch x64
                ; mov [rsp + off], r14
            );
        }

        fn emit_set_out_ptr(&mut self, src: crate::ir::VReg) {
            let off = self.vreg_off(src) as i32;
            dynasm!(self.ectx.ops
                ; .arch x64
                ; mov r14, [rsp + off]
            );
        }

        fn emit_slot_addr(&mut self, dst: crate::ir::VReg, slot: crate::ir::SlotId) {
            let dst_off = self.vreg_off(dst) as i32;
            let slot_off = self.slot_off(slot) as i32;
            dynasm!(self.ectx.ops
                ; .arch x64
                ; lea r10, [rsp + slot_off]
                ; mov [rsp + dst_off], r10
            );
        }

        fn emit_read_bytes(&mut self, dst: crate::ir::VReg, count: u32) {
            self.emit_recipe_ops(vec![Op::BoundsCheck { count }]);
            match count {
                1 => dynasm!(self.ectx.ops ; .arch x64 ; movzx r10d, BYTE [r12]),
                2 => dynasm!(self.ectx.ops ; .arch x64 ; movzx r10d, WORD [r12]),
                4 => dynasm!(self.ectx.ops ; .arch x64 ; mov r10d, DWORD [r12]),
                8 => dynasm!(self.ectx.ops ; .arch x64 ; mov r10, QWORD [r12]),
                _ => panic!("unsupported ReadBytes count: {count}"),
            }
            self.emit_store_r10_to_vreg(dst);
            self.ectx.emit_advance_cursor_by(count);
        }

        fn emit_peek_byte(&mut self, dst: crate::ir::VReg) {
            self.emit_recipe_ops(vec![Op::BoundsCheck { count: 1 }]);
            dynasm!(self.ectx.ops ; .arch x64 ; movzx r10d, BYTE [r12]);
            self.emit_store_r10_to_vreg(dst);
        }

        fn emit_binop(
            &mut self,
            kind: BinOpKind,
            dst: crate::ir::VReg,
            lhs: crate::ir::VReg,
            rhs: crate::ir::VReg,
        ) {
            self.emit_load_vreg_r10(lhs);
            let rhs_off = self.vreg_off(rhs) as i32;
            match kind {
                BinOpKind::Add => dynasm!(self.ectx.ops ; .arch x64 ; add r10, [rsp + rhs_off]),
                BinOpKind::Sub => dynasm!(self.ectx.ops ; .arch x64 ; sub r10, [rsp + rhs_off]),
                BinOpKind::And => dynasm!(self.ectx.ops ; .arch x64 ; and r10, [rsp + rhs_off]),
                BinOpKind::Or => dynasm!(self.ectx.ops ; .arch x64 ; or r10, [rsp + rhs_off]),
                BinOpKind::Xor => dynasm!(self.ectx.ops ; .arch x64 ; xor r10, [rsp + rhs_off]),
                BinOpKind::CmpNe => dynasm!(self.ectx.ops
                    ; .arch x64
                    ; cmp r10, [rsp + rhs_off]
                    ; setne r10b
                    ; movzx r10, r10b
                ),
                BinOpKind::Shr => dynasm!(self.ectx.ops
                    ; .arch x64
                    ; mov rcx, [rsp + rhs_off]
                    ; shr r10, cl
                ),
                BinOpKind::Shl => dynasm!(self.ectx.ops
                    ; .arch x64
                    ; mov rcx, [rsp + rhs_off]
                    ; shl r10, cl
                ),
            }
            self.emit_store_r10_to_vreg(dst);
        }

        fn emit_unary(&mut self, kind: UnaryOpKind, dst: crate::ir::VReg, src: crate::ir::VReg) {
            self.emit_load_vreg_r10(src);
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
            self.emit_store_r10_to_vreg(dst);
        }

        fn emit_branch_if(&mut self, cond: crate::ir::VReg, target: DynamicLabel, invert: bool) {
            self.emit_load_vreg_r10(cond);
            if invert {
                dynasm!(self.ectx.ops
                    ; .arch x64
                    ; test r10, r10
                    ; jz =>target
                );
            } else {
                dynasm!(self.ectx.ops
                    ; .arch x64
                    ; test r10, r10
                    ; jnz =>target
                );
            }
        }

        fn emit_jump_table(
            &mut self,
            predicate: crate::ir::VReg,
            labels: &[LabelId],
            default: LabelId,
        ) {
            self.emit_load_vreg_r10(predicate);
            for (index, label) in labels.iter().enumerate() {
                let target = self.label(*label);
                dynasm!(self.ectx.ops
                    ; .arch x64
                    ; cmp r10d, index as i32
                    ; je =>target
                );
            }
            self.ectx.emit_branch(self.label(default));
        }

        fn emit_load_intrinsic_arg_rsi(&mut self, arg: IntrinsicArg) {
            match arg {
                IntrinsicArg::VReg(v) => {
                    let off = self.vreg_off(v) as i32;
                    dynasm!(self.ectx.ops ; .arch x64 ; mov rsi, [rsp + off]);
                }
                IntrinsicArg::OutField(offset) => {
                    let off = offset as i32;
                    dynasm!(self.ectx.ops ; .arch x64 ; lea rsi, [r14 + off]);
                }
                IntrinsicArg::OutStack(offset) => {
                    let off = offset as i32;
                    dynasm!(self.ectx.ops ; .arch x64 ; lea rsi, [rsp + off]);
                }
            }
        }

        fn emit_load_intrinsic_arg_rdi(&mut self, arg: IntrinsicArg) {
            match arg {
                IntrinsicArg::VReg(v) => {
                    let off = self.vreg_off(v) as i32;
                    dynasm!(self.ectx.ops ; .arch x64 ; mov rdi, [rsp + off]);
                }
                IntrinsicArg::OutField(offset) => {
                    let off = offset as i32;
                    dynasm!(self.ectx.ops ; .arch x64 ; lea rdi, [r14 + off]);
                }
                IntrinsicArg::OutStack(offset) => {
                    let off = offset as i32;
                    dynasm!(self.ectx.ops ; .arch x64 ; lea rdi, [rsp + off]);
                }
            }
        }

        fn emit_load_intrinsic_arg_rdx(&mut self, arg: IntrinsicArg) {
            match arg {
                IntrinsicArg::VReg(v) => {
                    let off = self.vreg_off(v) as i32;
                    dynasm!(self.ectx.ops ; .arch x64 ; mov rdx, [rsp + off]);
                }
                IntrinsicArg::OutField(offset) => {
                    let off = offset as i32;
                    dynasm!(self.ectx.ops ; .arch x64 ; lea rdx, [r14 + off]);
                }
                IntrinsicArg::OutStack(offset) => {
                    let off = offset as i32;
                    dynasm!(self.ectx.ops ; .arch x64 ; lea rdx, [rsp + off]);
                }
            }
        }

        fn emit_load_intrinsic_arg_rcx(&mut self, arg: IntrinsicArg) {
            match arg {
                IntrinsicArg::VReg(v) => {
                    let off = self.vreg_off(v) as i32;
                    dynasm!(self.ectx.ops ; .arch x64 ; mov rcx, [rsp + off]);
                }
                IntrinsicArg::OutField(offset) => {
                    let off = offset as i32;
                    dynasm!(self.ectx.ops ; .arch x64 ; lea rcx, [r14 + off]);
                }
                IntrinsicArg::OutStack(offset) => {
                    let off = offset as i32;
                    dynasm!(self.ectx.ops ; .arch x64 ; lea rcx, [rsp + off]);
                }
            }
        }

        fn emit_load_intrinsic_arg_r8(&mut self, arg: IntrinsicArg) {
            match arg {
                IntrinsicArg::VReg(v) => {
                    let off = self.vreg_off(v) as i32;
                    dynasm!(self.ectx.ops ; .arch x64 ; mov r8, [rsp + off]);
                }
                IntrinsicArg::OutField(offset) => {
                    let off = offset as i32;
                    dynasm!(self.ectx.ops ; .arch x64 ; lea r8, [r14 + off]);
                }
                IntrinsicArg::OutStack(offset) => {
                    let off = offset as i32;
                    dynasm!(self.ectx.ops ; .arch x64 ; lea r8, [rsp + off]);
                }
            }
        }

        fn emit_load_intrinsic_arg_r9(&mut self, arg: IntrinsicArg) {
            match arg {
                IntrinsicArg::VReg(v) => {
                    let off = self.vreg_off(v) as i32;
                    dynasm!(self.ectx.ops ; .arch x64 ; mov r9, [rsp + off]);
                }
                IntrinsicArg::OutField(offset) => {
                    let off = offset as i32;
                    dynasm!(self.ectx.ops ; .arch x64 ; lea r9, [r14 + off]);
                }
                IntrinsicArg::OutStack(offset) => {
                    let off = offset as i32;
                    dynasm!(self.ectx.ops ; .arch x64 ; lea r9, [rsp + off]);
                }
            }
        }

        fn emit_load_intrinsic_arg_r10(&mut self, arg: IntrinsicArg) {
            match arg {
                IntrinsicArg::VReg(v) => {
                    let off = self.vreg_off(v) as i32;
                    dynasm!(self.ectx.ops ; .arch x64 ; mov r10, [rsp + off]);
                }
                IntrinsicArg::OutField(offset) => {
                    let off = offset as i32;
                    dynasm!(self.ectx.ops ; .arch x64 ; lea r10, [r14 + off]);
                }
                IntrinsicArg::OutStack(offset) => {
                    let off = offset as i32;
                    dynasm!(self.ectx.ops ; .arch x64 ; lea r10, [rsp + off]);
                }
            }
        }

        fn emit_call_intrinsic_with_args(&mut self, fn_ptr: *const u8, args: &[IntrinsicArg]) {
            use crate::context::{CTX_ERROR_CODE, CTX_INPUT_PTR};

            let error_exit = self
                .current_func
                .as_ref()
                .expect("CallIntrinsic outside function")
                .error_exit;

            // Flush cursor before call.
            dynasm!(self.ectx.ops
                ; .arch x64
                ; mov [r15 + CTX_INPUT_PTR as i32], r12
            );

            #[cfg(not(windows))]
            {
                if args.len() > 5 {
                    panic!(
                        "unsupported CallIntrinsic arity in linear backend adapter: {} args (+ctx)",
                        args.len()
                    );
                }
                dynasm!(self.ectx.ops ; .arch x64 ; mov rdi, r15);
                if let Some(arg) = args.first().copied() {
                    self.emit_load_intrinsic_arg_rsi(arg);
                }
                if let Some(arg) = args.get(1).copied() {
                    self.emit_load_intrinsic_arg_rdx(arg);
                }
                if let Some(arg) = args.get(2).copied() {
                    self.emit_load_intrinsic_arg_rcx(arg);
                }
                if let Some(arg) = args.get(3).copied() {
                    self.emit_load_intrinsic_arg_r8(arg);
                }
                if let Some(arg) = args.get(4).copied() {
                    self.emit_load_intrinsic_arg_r9(arg);
                }
            }

            #[cfg(windows)]
            {
                if args.len() > 5 {
                    panic!(
                        "unsupported CallIntrinsic arity in linear backend adapter: {} args (+ctx)",
                        args.len()
                    );
                }
                dynasm!(self.ectx.ops ; .arch x64 ; mov rcx, r15);
                if let Some(arg) = args.first().copied() {
                    self.emit_load_intrinsic_arg_rdx(arg);
                }
                if let Some(arg) = args.get(1).copied() {
                    self.emit_load_intrinsic_arg_r8(arg);
                }
                if let Some(arg) = args.get(2).copied() {
                    self.emit_load_intrinsic_arg_r9(arg);
                }
                if let Some(arg) = args.get(3).copied() {
                    self.emit_load_intrinsic_arg_r10(arg);
                    dynasm!(self.ectx.ops ; .arch x64 ; mov [rsp + 32], r10);
                }
                if let Some(arg) = args.get(4).copied() {
                    self.emit_load_intrinsic_arg_r10(arg);
                    dynasm!(self.ectx.ops ; .arch x64 ; mov [rsp + 40], r10);
                }
            }

            let ptr_val = fn_ptr as i64;
            dynasm!(self.ectx.ops
                ; .arch x64
                ; mov rax, QWORD ptr_val
                ; call rax
            );

            // Reload cursor and branch to error exit if needed.
            dynasm!(self.ectx.ops
                ; .arch x64
                ; mov r12, [r15 + CTX_INPUT_PTR as i32]
                ; mov r10d, [r15 + CTX_ERROR_CODE as i32]
                ; test r10d, r10d
                ; jnz =>error_exit
            );
        }

        // r[impl ir.intrinsics]
        fn emit_call_intrinsic(
            &mut self,
            func: crate::ir::IntrinsicFn,
            args: &[crate::ir::VReg],
            dst: Option<crate::ir::VReg>,
            field_offset: u32,
        ) {
            let fn_ptr = func.0 as *const u8;
            match dst {
                Some(dst) => {
                    if args.is_empty() {
                        // Legacy stack-out intrinsic ABI: fn(ctx, &mut out)
                        let out_offset = self.vreg_off(dst);
                        self.ectx.emit_store_imm64_to_stack(out_offset, 0);
                        self.emit_call_intrinsic_with_args(
                            fn_ptr,
                            &[IntrinsicArg::OutStack(out_offset)],
                        );
                    } else {
                        // Return-value intrinsic ABI: fn(ctx, args...) -> value
                        let call_args: Vec<IntrinsicArg> =
                            args.iter().copied().map(IntrinsicArg::VReg).collect();
                        self.emit_call_intrinsic_with_args(fn_ptr, &call_args);
                        let out_offset = self.vreg_off(dst) as i32;
                        dynasm!(self.ectx.ops
                            ; .arch x64
                            ; mov [rsp + out_offset], rax
                        );
                    }
                }
                None => {
                    let mut call_args: Vec<IntrinsicArg> =
                        args.iter().copied().map(IntrinsicArg::VReg).collect();
                    // Side-effect intrinsic ABI: fn(ctx, args..., out+field_offset)
                    call_args.push(IntrinsicArg::OutField(field_offset));
                    self.emit_call_intrinsic_with_args(fn_ptr, &call_args);
                }
            }
        }

        fn emit_call_pure(
            &mut self,
            func: crate::ir::IntrinsicFn,
            args: &[crate::ir::VReg],
            dst: crate::ir::VReg,
        ) {
            let fn_ptr = func.0 as i64;

            #[cfg(not(windows))]
            {
                if args.len() > 6 {
                    panic!(
                        "unsupported CallPure arity in linear backend adapter: {} args",
                        args.len()
                    );
                }
                if let Some(&arg) = args.first() {
                    self.emit_load_intrinsic_arg_rdi(IntrinsicArg::VReg(arg));
                }
                if let Some(&arg) = args.get(1) {
                    self.emit_load_intrinsic_arg_rsi(IntrinsicArg::VReg(arg));
                }
                if let Some(&arg) = args.get(2) {
                    self.emit_load_intrinsic_arg_rdx(IntrinsicArg::VReg(arg));
                }
                if let Some(&arg) = args.get(3) {
                    self.emit_load_intrinsic_arg_rcx(IntrinsicArg::VReg(arg));
                }
                if let Some(&arg) = args.get(4) {
                    self.emit_load_intrinsic_arg_r8(IntrinsicArg::VReg(arg));
                }
                if let Some(&arg) = args.get(5) {
                    self.emit_load_intrinsic_arg_r9(IntrinsicArg::VReg(arg));
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
                if let Some(&arg) = args.first() {
                    self.emit_load_intrinsic_arg_rcx(IntrinsicArg::VReg(arg));
                }
                if let Some(&arg) = args.get(1) {
                    self.emit_load_intrinsic_arg_rdx(IntrinsicArg::VReg(arg));
                }
                if let Some(&arg) = args.get(2) {
                    self.emit_load_intrinsic_arg_r8(IntrinsicArg::VReg(arg));
                }
                if let Some(&arg) = args.get(3) {
                    self.emit_load_intrinsic_arg_r9(IntrinsicArg::VReg(arg));
                }
            }

            dynasm!(self.ectx.ops
                ; .arch x64
                ; mov rax, QWORD fn_ptr
                ; call rax
            );

            let out_offset = self.vreg_off(dst) as i32;
            dynasm!(self.ectx.ops
                ; .arch x64
                ; mov [rsp + out_offset], rax
            );
        }

        fn emit_store_incoming_lambda_args(&mut self, data_args: &[crate::ir::VReg]) {
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

            for (i, &arg) in data_args.iter().enumerate() {
                let off = self.vreg_off(arg) as i32;
                #[cfg(not(windows))]
                match i {
                    0 => dynasm!(self.ectx.ops ; .arch x64 ; mov [rsp + off], rdx),
                    1 => dynasm!(self.ectx.ops ; .arch x64 ; mov [rsp + off], rcx),
                    2 => dynasm!(self.ectx.ops ; .arch x64 ; mov [rsp + off], r8),
                    3 => dynasm!(self.ectx.ops ; .arch x64 ; mov [rsp + off], r9),
                    _ => unreachable!(),
                }
                #[cfg(windows)]
                match i {
                    0 => dynasm!(self.ectx.ops ; .arch x64 ; mov [rsp + off], r8),
                    1 => dynasm!(self.ectx.ops ; .arch x64 ; mov [rsp + off], r9),
                    _ => unreachable!(),
                }
            }
        }

        fn emit_load_lambda_results_to_ret_regs(&mut self, data_results: &[crate::ir::VReg]) {
            if data_results.len() > 2 {
                panic!(
                    "x64 CallLambda supports at most 2 data results, got {}",
                    data_results.len()
                );
            }

            if let Some(&v) = data_results.first() {
                let off = self.vreg_off(v) as i32;
                dynasm!(self.ectx.ops ; .arch x64 ; mov rax, [rsp + off]);
            }
            if let Some(&v) = data_results.get(1) {
                let off = self.vreg_off(v) as i32;
                dynasm!(self.ectx.ops ; .arch x64 ; mov rdx, [rsp + off]);
            }
        }

        fn emit_call_lambda(
            &mut self,
            label: DynamicLabel,
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
            dynasm!(self.ectx.ops
                ; .arch x64
                ; mov [r15 + CTX_INPUT_PTR as i32], r12
            );
            #[cfg(not(windows))]
            dynasm!(self.ectx.ops ; .arch x64 ; lea rdi, [r14] ; mov rsi, r15);
            #[cfg(windows)]
            dynasm!(self.ectx.ops ; .arch x64 ; lea rcx, [r14] ; mov rdx, r15);

            for (i, &arg) in args.iter().enumerate() {
                let off = self.vreg_off(arg) as i32;
                #[cfg(not(windows))]
                match i {
                    0 => dynasm!(self.ectx.ops ; .arch x64 ; mov rdx, [rsp + off]),
                    1 => dynasm!(self.ectx.ops ; .arch x64 ; mov rcx, [rsp + off]),
                    2 => dynasm!(self.ectx.ops ; .arch x64 ; mov r8, [rsp + off]),
                    3 => dynasm!(self.ectx.ops ; .arch x64 ; mov r9, [rsp + off]),
                    _ => unreachable!(),
                }
                #[cfg(windows)]
                match i {
                    0 => dynasm!(self.ectx.ops ; .arch x64 ; mov r8, [rsp + off]),
                    1 => dynasm!(self.ectx.ops ; .arch x64 ; mov r9, [rsp + off]),
                    _ => unreachable!(),
                }
            }

            dynasm!(self.ectx.ops ; .arch x64 ; call =>label);
            dynasm!(self.ectx.ops
                ; .arch x64
                ; mov r12, [r15 + CTX_INPUT_PTR as i32]
                ; mov r10d, [r15 + CTX_ERROR_CODE as i32]
                ; test r10d, r10d
                ; jnz =>error_exit
            );

            if let Some(&dst) = results.first() {
                let off = self.vreg_off(dst) as i32;
                dynasm!(self.ectx.ops ; .arch x64 ; mov [rsp + off], rax);
            }
            if let Some(&dst) = results.get(1) {
                let off = self.vreg_off(dst) as i32;
                dynasm!(self.ectx.ops ; .arch x64 ; mov [rsp + off], rdx);
            }
        }

        fn run(mut self, ir: &LinearIr) -> LinearBackendResult {
            for op in &ir.ops {
                match op {
                    LinearOp::FuncStart {
                        lambda_id,
                        data_args,
                        data_results,
                        ..
                    } => {
                        let label = self.lambda_labels[lambda_id.index() as usize];
                        self.ectx.bind_label(label);
                        let (entry_offset, error_exit) = self.ectx.begin_func();
                        if lambda_id.index() == 0 {
                            self.entry = Some(entry_offset);
                        }
                        self.current_func = Some(FunctionCtx {
                            error_exit,
                            data_results: data_results.clone(),
                        });
                        self.const_vregs.fill(None);
                        self.x9_cached_vreg = None;
                        self.emit_store_incoming_lambda_args(data_args);
                    }
                    LinearOp::FuncEnd => {
                        let func = self
                            .current_func
                            .take()
                            .expect("FuncEnd without active function");
                        self.emit_load_lambda_results_to_ret_regs(&func.data_results);
                        self.ectx.end_func(func.error_exit);
                    }
                    LinearOp::Label(label) => self.ectx.bind_label(self.label(*label)),
                    LinearOp::Branch(target) => self.ectx.emit_branch(self.label(*target)),
                    LinearOp::BranchIf { cond, target } => {
                        self.emit_branch_if(*cond, self.label(*target), false);
                    }
                    LinearOp::BranchIfZero { cond, target } => {
                        self.emit_branch_if(*cond, self.label(*target), true);
                    }
                    LinearOp::JumpTable {
                        predicate,
                        labels,
                        default,
                    } => {
                        self.emit_jump_table(*predicate, labels, *default);
                    }

                    LinearOp::Const { dst, value } => {
                        self.ectx
                            .emit_store_imm64_to_stack(self.vreg_off(*dst), *value);
                    }
                    LinearOp::Copy { dst, src } => {
                        self.emit_recipe_ops(vec![
                            Op::LoadFromStack {
                                dst: Slot::A,
                                sp_offset: self.vreg_off(*src),
                                width: crate::recipe::Width::W8,
                            },
                            Op::StoreToStack {
                                src: Slot::A,
                                sp_offset: self.vreg_off(*dst),
                                width: crate::recipe::Width::W8,
                            },
                        ]);
                    }
                    LinearOp::BinOp { op, dst, lhs, rhs } => self.emit_binop(*op, *dst, *lhs, *rhs),
                    LinearOp::UnaryOp { op, dst, src } => self.emit_unary(*op, *dst, *src),

                    LinearOp::BoundsCheck { count } => {
                        self.emit_recipe_ops(vec![Op::BoundsCheck { count: *count }]);
                    }
                    LinearOp::ReadBytes { dst, count } => self.emit_read_bytes(*dst, *count),
                    LinearOp::PeekByte { dst } => self.emit_peek_byte(*dst),
                    LinearOp::AdvanceCursor { count } => self.ectx.emit_advance_cursor_by(*count),
                    LinearOp::AdvanceCursorBy { src } => {
                        self.emit_recipe_ops(vec![
                            Op::LoadFromStack {
                                dst: Slot::A,
                                sp_offset: self.vreg_off(*src),
                                width: crate::recipe::Width::W8,
                            },
                            Op::AdvanceCursorBySlot { slot: Slot::A },
                        ]);
                    }
                    LinearOp::SaveCursor { dst } => {
                        self.ectx.emit_save_cursor_to_stack(self.vreg_off(*dst));
                    }
                    LinearOp::RestoreCursor { src } => {
                        self.ectx.emit_restore_input_ptr(self.vreg_off(*src));
                    }

                    LinearOp::WriteToField { src, offset, width } => {
                        self.emit_write_to_field(*src, *offset, *width);
                    }
                    LinearOp::ReadFromField { dst, offset, width } => {
                        self.emit_read_from_field(*dst, *offset, *width);
                    }
                    LinearOp::SaveOutPtr { dst } => {
                        self.emit_save_out_ptr(*dst);
                    }
                    LinearOp::SetOutPtr { src } => {
                        self.emit_set_out_ptr(*src);
                    }
                    LinearOp::SlotAddr { dst, slot } => {
                        self.emit_slot_addr(*dst, *slot);
                    }
                    LinearOp::WriteToSlot { slot, src } => {
                        self.emit_recipe_ops(vec![
                            Op::LoadFromStack {
                                dst: Slot::A,
                                sp_offset: self.vreg_off(*src),
                                width: crate::recipe::Width::W8,
                            },
                            Op::StoreToStack {
                                src: Slot::A,
                                sp_offset: self.slot_off(*slot),
                                width: crate::recipe::Width::W8,
                            },
                        ]);
                    }
                    LinearOp::ReadFromSlot { dst, slot } => {
                        self.emit_recipe_ops(vec![
                            Op::LoadFromStack {
                                dst: Slot::A,
                                sp_offset: self.slot_off(*slot),
                                width: crate::recipe::Width::W8,
                            },
                            Op::StoreToStack {
                                src: Slot::A,
                                sp_offset: self.vreg_off(*dst),
                                width: crate::recipe::Width::W8,
                            },
                        ]);
                    }

                    LinearOp::CallIntrinsic {
                        func,
                        args,
                        dst,
                        field_offset,
                    } => {
                        self.emit_call_intrinsic(*func, args, *dst, *field_offset);
                    }
                    LinearOp::CallPure { func, args, dst } => {
                        self.emit_call_pure(*func, args, *dst);
                    }

                    LinearOp::ErrorExit { code } => {
                        self.ectx.emit_error(*code);
                    }

                    LinearOp::SimdStringScan { .. } | LinearOp::SimdWhitespaceSkip => {
                        panic!("unsupported SIMD op in linear backend adapter");
                    }

                    LinearOp::CallLambda {
                        target,
                        args,
                        results,
                    } => {
                        let label = self.lambda_labels[target.index() as usize];
                        self.emit_call_lambda(label, args, results);
                    }
                }
            }

            if self.current_func.is_some() {
                panic!("unterminated function: missing FuncEnd");
            }

            let entry = self.entry.expect("missing root FuncStart for lambda 0");
            let buf = self.ectx.finalize();
            LinearBackendResult { buf, entry }
        }
    }

    Lowerer::new(ir).run(ir)
}

#[cfg(target_arch = "aarch64")]
// r[impl ir.backends.post-regalloc.branch-test]
// r[impl ir.backends.post-regalloc.shuffle]
fn compile_linear_ir_aarch64(
    ir: &LinearIr,
    max_spillslots: usize,
    alloc: &crate::regalloc_engine::AllocatedProgram,
) -> LinearBackendResult {
    use dynasmrt::{DynamicLabel, DynasmApi, DynasmLabelApi, dynasm};
    use regalloc2::{Allocation, Edit, InstPosition, PReg, RegClass};
    use std::collections::{BTreeMap, BTreeSet};

    use crate::arch::{BASE_FRAME, EmitCtx};
    use crate::ir::Width;
    use crate::linearize::{BinOpKind, LabelId, LinearOp, UnaryOpKind};
    use crate::recipe::{Op, Recipe};

    struct FunctionCtx {
        error_exit: DynamicLabel,
        data_results: Vec<crate::ir::VReg>,
        lambda_id: crate::ir::LambdaId,
    }

    #[derive(Default)]
    struct LambdaEditMap {
        before: BTreeMap<usize, Vec<(Allocation, Allocation)>>,
        after: BTreeMap<usize, Vec<(Allocation, Allocation)>>,
    }

    #[derive(Default)]
    struct LambdaEdgeEditMap {
        before: BTreeMap<(usize, usize), Vec<(Allocation, Allocation)>>,
        after: BTreeMap<(usize, usize), Vec<(Allocation, Allocation)>>,
    }

    struct EdgeTrampoline {
        label: DynamicLabel,
        target: DynamicLabel,
        moves: Vec<(Allocation, Allocation)>,
    }

    struct Lowerer {
        ectx: EmitCtx,
        labels: Vec<DynamicLabel>,
        lambda_labels: Vec<DynamicLabel>,
        slot_base: u32,
        spill_base: u32,
        entry: Option<AssemblyOffset>,
        current_func: Option<FunctionCtx>,
        const_vregs: Vec<Option<u64>>,
        edits_by_lambda: BTreeMap<u32, LambdaEditMap>,
        edge_edits_by_lambda: BTreeMap<u32, LambdaEdgeEditMap>,
        forward_branch_labels_by_lambda: BTreeMap<u32, BTreeMap<LabelId, (usize, LabelId)>>,
        allocs_by_lambda: BTreeMap<u32, BTreeMap<usize, Vec<Allocation>>>,
        return_result_allocs_by_lambda: BTreeMap<u32, Vec<Allocation>>,
        edge_trampoline_labels: BTreeMap<(u32, usize, usize), DynamicLabel>,
        edge_trampolines: Vec<EdgeTrampoline>,
        current_inst_allocs: Option<Vec<Allocation>>,
        current_lambda_linear_op_index: usize,
    }

    #[derive(Clone, Copy)]
    enum IntrinsicArg {
        VReg { operand_index: usize },
        OutField(u32),
    }

    impl Lowerer {
        // r[impl ir.regalloc.edits.minimize]
        fn normalize_edit_move(
            from: Allocation,
            to: Allocation,
        ) -> Option<(Allocation, Allocation)> {
            if from == to || from.is_none() || to.is_none() {
                return None;
            }
            Some((from, to))
        }

        fn linear_op_uses_vreg(op: &LinearOp, v: crate::ir::VReg) -> bool {
            match op {
                LinearOp::Const { .. }
                | LinearOp::BoundsCheck { .. }
                | LinearOp::AdvanceCursor { .. }
                | LinearOp::SlotAddr { .. }
                | LinearOp::Label(_)
                | LinearOp::Branch(_)
                | LinearOp::ErrorExit { .. }
                | LinearOp::SimdWhitespaceSkip
                | LinearOp::FuncStart { .. }
                | LinearOp::FuncEnd => false,
                LinearOp::BinOp { lhs, rhs, .. } => *lhs == v || *rhs == v,
                LinearOp::UnaryOp { src, .. }
                | LinearOp::AdvanceCursorBy { src }
                | LinearOp::RestoreCursor { src }
                | LinearOp::SetOutPtr { src }
                | LinearOp::WriteToSlot { src, .. } => *src == v,
                LinearOp::Copy { src, .. } => *src == v,
                LinearOp::ReadBytes { .. }
                | LinearOp::PeekByte { .. }
                | LinearOp::SaveCursor { .. }
                | LinearOp::ReadFromField { .. }
                | LinearOp::SaveOutPtr { .. }
                | LinearOp::ReadFromSlot { .. } => false,
                LinearOp::WriteToField { src, .. } => *src == v,
                LinearOp::CallIntrinsic { args, .. } | LinearOp::CallPure { args, .. } => {
                    args.contains(&v)
                }
                LinearOp::BranchIf { cond, .. } | LinearOp::BranchIfZero { cond, .. } => *cond == v,
                LinearOp::JumpTable { predicate, .. } => *predicate == v,
                LinearOp::SimdStringScan { pos, kind } => *pos == v || *kind == v,
                LinearOp::CallLambda { args, .. } => args.contains(&v),
            }
        }

        fn linear_op_defs_vreg(op: &LinearOp, v: crate::ir::VReg) -> bool {
            match op {
                LinearOp::Const { dst, .. }
                | LinearOp::UnaryOp { dst, .. }
                | LinearOp::Copy { dst, .. }
                | LinearOp::ReadBytes { dst, .. }
                | LinearOp::PeekByte { dst }
                | LinearOp::SaveCursor { dst }
                | LinearOp::ReadFromField { dst, .. }
                | LinearOp::SaveOutPtr { dst }
                | LinearOp::SlotAddr { dst, .. }
                | LinearOp::ReadFromSlot { dst, .. } => *dst == v,
                LinearOp::BinOp { dst, .. } | LinearOp::CallPure { dst, .. } => *dst == v,
                LinearOp::CallIntrinsic { dst, .. } => dst.is_some_and(|dst| dst == v),
                LinearOp::CallLambda { results, .. } => results.contains(&v),
                LinearOp::BoundsCheck { .. }
                | LinearOp::AdvanceCursor { .. }
                | LinearOp::AdvanceCursorBy { .. }
                | LinearOp::RestoreCursor { .. }
                | LinearOp::WriteToField { .. }
                | LinearOp::SetOutPtr { .. }
                | LinearOp::WriteToSlot { .. }
                | LinearOp::Label(_)
                | LinearOp::Branch(_)
                | LinearOp::BranchIf { .. }
                | LinearOp::BranchIfZero { .. }
                | LinearOp::JumpTable { .. }
                | LinearOp::ErrorExit { .. }
                | LinearOp::SimdStringScan { .. }
                | LinearOp::SimdWhitespaceSkip
                | LinearOp::FuncStart { .. }
                | LinearOp::FuncEnd => false,
            }
        }

        fn linear_op_preserves_cmp_flags(op: &LinearOp) -> bool {
            #[allow(clippy::match_like_matches_macro)]
            match op {
                LinearOp::BinOp {
                    op: BinOpKind::CmpNe,
                    ..
                } => false,
                LinearOp::BoundsCheck { .. }
                | LinearOp::ReadBytes { .. }
                | LinearOp::PeekByte { .. }
                | LinearOp::CallIntrinsic { .. }
                | LinearOp::CallPure { .. }
                | LinearOp::JumpTable { .. }
                | LinearOp::ErrorExit { .. }
                | LinearOp::CallLambda { .. }
                | LinearOp::SimdStringScan { .. }
                | LinearOp::SimdWhitespaceSkip => false,
                _ => true,
            }
        }

        fn find_cmpne_branch_use(
            &self,
            ir: &LinearIr,
            cmp_op_index: usize,
            cond_vreg: crate::ir::VReg,
            cmp_linear_op_index: usize,
        ) -> Option<usize> {
            let mut scan_linear_op_index = cmp_linear_op_index + 1;
            for scan_index in cmp_op_index + 1..ir.ops.len() {
                let op = &ir.ops[scan_index];
                match op {
                    LinearOp::BranchIf { cond, .. } | LinearOp::BranchIfZero { cond, .. }
                        if *cond == cond_vreg =>
                    {
                        for lin in cmp_linear_op_index..=scan_linear_op_index {
                            if self.has_inst_edits(lin) {
                                return None;
                            }
                        }
                        return Some(scan_index);
                    }
                    LinearOp::Label(_) => {
                        scan_linear_op_index += 1;
                        continue;
                    }
                    LinearOp::Branch(_)
                    | LinearOp::BranchIf { .. }
                    | LinearOp::BranchIfZero { .. }
                    | LinearOp::JumpTable { .. }
                    | LinearOp::ErrorExit { .. }
                    | LinearOp::FuncStart { .. }
                    | LinearOp::FuncEnd => return None,
                    _ => {
                        if Self::linear_op_uses_vreg(op, cond_vreg)
                            || Self::linear_op_defs_vreg(op, cond_vreg)
                            || !Self::linear_op_preserves_cmp_flags(op)
                        {
                            return None;
                        }
                    }
                }
                scan_linear_op_index += 1;
            }
            None
        }

        fn find_and_branch_use(
            &self,
            ir: &LinearIr,
            and_op_index: usize,
            and_dst: crate::ir::VReg,
            and_lhs: crate::ir::VReg,
            and_rhs: crate::ir::VReg,
            and_linear_op_index: usize,
        ) -> Option<usize> {
            let mut scan_linear_op_index = and_linear_op_index + 1;
            for scan_index in and_op_index + 1..ir.ops.len() {
                let op = &ir.ops[scan_index];
                match op {
                    LinearOp::BranchIf { cond, .. } | LinearOp::BranchIfZero { cond, .. }
                        if *cond == and_dst =>
                    {
                        for lin in and_linear_op_index..=scan_linear_op_index {
                            if self.has_inst_edits(lin) {
                                return None;
                            }
                        }
                        return Some(scan_index);
                    }
                    LinearOp::Label(_) => {
                        scan_linear_op_index += 1;
                        continue;
                    }
                    LinearOp::Branch(_)
                    | LinearOp::BranchIf { .. }
                    | LinearOp::BranchIfZero { .. }
                    | LinearOp::JumpTable { .. }
                    | LinearOp::ErrorExit { .. }
                    | LinearOp::FuncStart { .. }
                    | LinearOp::FuncEnd => return None,
                    _ => {
                        if Self::linear_op_uses_vreg(op, and_dst)
                            || Self::linear_op_defs_vreg(op, and_dst)
                            || Self::linear_op_defs_vreg(op, and_lhs)
                            || Self::linear_op_defs_vreg(op, and_rhs)
                        {
                            return None;
                        }
                    }
                }
                scan_linear_op_index += 1;
            }
            None
        }

        fn linear_ir_use_count_in_function(
            ir: &LinearIr,
            op_index: usize,
            v: crate::ir::VReg,
        ) -> usize {
            let mut start = op_index;
            while start > 0 {
                if matches!(ir.ops[start], LinearOp::FuncStart { .. }) {
                    break;
                }
                start -= 1;
            }
            let mut end = op_index;
            while end + 1 < ir.ops.len() {
                if matches!(ir.ops[end], LinearOp::FuncEnd) {
                    break;
                }
                end += 1;
            }
            ir.ops[start..=end]
                .iter()
                .filter(|op| Self::linear_op_uses_vreg(op, v))
                .count()
        }

        fn allocs_for_linear_op(&self, linear_op_index: usize) -> Option<Vec<Allocation>> {
            let lambda_id = self.current_func.as_ref()?.lambda_id.index() as u32;
            self.allocs_by_lambda
                .get(&lambda_id)
                .and_then(|by_lambda| by_lambda.get(&linear_op_index))
                .cloned()
        }

        fn regalloc_extra_saved_pairs(alloc: &crate::regalloc_engine::AllocatedProgram) -> u32 {
            let mut max_pair = None::<u32>;
            let mut observe = |a: Allocation| {
                let Some(reg) = a.as_reg() else {
                    return;
                };
                if reg.class() != RegClass::Int {
                    return;
                }
                let enc = reg.hw_enc() as u32;
                let pair = match enc {
                    23 | 24 => Some(0),
                    25 | 26 => Some(1),
                    27 | 28 => Some(2),
                    _ => None,
                };
                if let Some(pair) = pair {
                    max_pair = Some(max_pair.map_or(pair, |cur| cur.max(pair)));
                }
            };

            for func in &alloc.functions {
                for inst_allocs in &func.inst_allocs {
                    for &a in inst_allocs {
                        observe(a);
                    }
                }
                for (_, edit) in &func.edits {
                    let Edit::Move { from, to } = edit;
                    observe(*from);
                    observe(*to);
                }
                for edge in &func.edge_edits {
                    observe(edge.from);
                    observe(edge.to);
                }
                for &a in &func.return_result_allocs {
                    observe(a);
                }
            }

            max_pair.map_or(0, |p| p + 1)
        }

        fn new(
            ir: &LinearIr,
            max_spillslots: usize,
            alloc: &crate::regalloc_engine::AllocatedProgram,
        ) -> Self {
            let extra_saved_pairs = Self::regalloc_extra_saved_pairs(alloc);
            let slot_base = BASE_FRAME + extra_saved_pairs * 16;
            let slot_bytes = ir.slot_count * 8;
            let spill_base = slot_base + slot_bytes;
            let spill_bytes = max_spillslots as u32 * 8;
            let extra_stack = slot_bytes + spill_bytes + 8;

            let mut ectx = EmitCtx::new_regalloc(extra_stack, extra_saved_pairs);
            let labels: Vec<DynamicLabel> = (0..ir.label_count).map(|_| ectx.new_label()).collect();

            let mut lambda_max = 0usize;
            for op in &ir.ops {
                match op {
                    LinearOp::FuncStart { lambda_id, .. } => {
                        lambda_max = lambda_max.max(lambda_id.index());
                    }
                    LinearOp::CallLambda { target, .. } => {
                        lambda_max = lambda_max.max(target.index());
                    }
                    _ => {}
                }
            }
            let lambda_labels: Vec<DynamicLabel> =
                (0..=lambda_max).map(|_| ectx.new_label()).collect();

            let mut forward_branch_labels_by_lambda =
                BTreeMap::<u32, BTreeMap<LabelId, (usize, LabelId)>>::new();
            let mut current_lambda_id = None::<u32>;
            let mut current_linear_op_index = 0usize;
            let mut pending_labels = Vec::<LabelId>::new();
            for op in &ir.ops {
                match op {
                    LinearOp::FuncStart { lambda_id, .. } => {
                        current_lambda_id = Some(lambda_id.index() as u32);
                        current_linear_op_index = 0;
                        pending_labels.clear();
                    }
                    LinearOp::FuncEnd => {
                        current_lambda_id = None;
                        pending_labels.clear();
                    }
                    _ => {
                        let Some(lambda_id) = current_lambda_id else {
                            continue;
                        };
                        match op {
                            LinearOp::Label(label) => {
                                pending_labels.push(*label);
                            }
                            LinearOp::Branch(target) => {
                                if !pending_labels.is_empty() {
                                    let by_lambda = forward_branch_labels_by_lambda
                                        .entry(lambda_id)
                                        .or_default();
                                    for label in pending_labels.drain(..) {
                                        by_lambda.insert(label, (current_linear_op_index, *target));
                                    }
                                }
                            }
                            _ => {
                                pending_labels.clear();
                            }
                        }
                        current_linear_op_index += 1;
                    }
                }
            }

            let mut edits_by_lambda = BTreeMap::<u32, LambdaEditMap>::new();
            let mut edge_edits_by_lambda = BTreeMap::<u32, LambdaEdgeEditMap>::new();
            let mut allocs_by_lambda = BTreeMap::<u32, BTreeMap<usize, Vec<Allocation>>>::new();
            let mut return_result_allocs_by_lambda = BTreeMap::<u32, Vec<Allocation>>::new();
            for func in &alloc.functions {
                let lambda_id = func.lambda_id.index() as u32;
                let lambda_entry = edits_by_lambda.entry(lambda_id).or_default();
                let lambda_edge_entry = edge_edits_by_lambda.entry(lambda_id).or_default();
                let allocs_entry = allocs_by_lambda.entry(lambda_id).or_default();
                return_result_allocs_by_lambda
                    .entry(lambda_id)
                    .or_insert_with(|| func.return_result_allocs.clone());
                for (prog_point, edit) in &func.edits {
                    let Some(Some(linear_op_index)) =
                        func.inst_linear_op_indices.get(prog_point.inst().index())
                    else {
                        continue;
                    };
                    let Edit::Move { from, to } = edit;
                    let Some((from, to)) = Self::normalize_edit_move(*from, *to) else {
                        continue;
                    };
                    let bucket = match prog_point.pos() {
                        InstPosition::Before => &mut lambda_entry.before,
                        InstPosition::After => &mut lambda_entry.after,
                    };
                    bucket.entry(*linear_op_index).or_default().push((from, to));
                }
                for edge_edit in &func.edge_edits {
                    let Some((from, to)) = Self::normalize_edit_move(edge_edit.from, edge_edit.to)
                    else {
                        continue;
                    };
                    let key = (edge_edit.from_linear_op_index, edge_edit.succ_index);
                    let bucket = match edge_edit.pos {
                        InstPosition::Before => &mut lambda_edge_entry.before,
                        InstPosition::After => &mut lambda_edge_entry.after,
                    };
                    bucket.entry(key).or_default().push((from, to));
                }
                for (inst_index, maybe_linear_op_index) in
                    func.inst_linear_op_indices.iter().copied().enumerate()
                {
                    let Some(linear_op_index) = maybe_linear_op_index else {
                        continue;
                    };
                    let Some(inst_allocs) = func.inst_allocs.get(inst_index) else {
                        continue;
                    };
                    allocs_entry.insert(linear_op_index, inst_allocs.clone());
                }
            }
            Self {
                ectx,
                labels,
                lambda_labels,
                slot_base,
                spill_base,
                entry: None,
                current_func: None,
                const_vregs: vec![None; ir.vreg_count as usize],
                edits_by_lambda,
                edge_edits_by_lambda,
                forward_branch_labels_by_lambda,
                allocs_by_lambda,
                return_result_allocs_by_lambda,
                edge_trampoline_labels: BTreeMap::new(),
                edge_trampolines: Vec::new(),
                current_inst_allocs: None,
                current_lambda_linear_op_index: 0,
            }
        }

        fn slot_off(&self, s: crate::ir::SlotId) -> u32 {
            self.slot_base + (s.index() as u32) * 8
        }

        fn label(&self, label: LabelId) -> DynamicLabel {
            self.labels[label.index()]
        }

        fn emit_mov_x9_from_preg(&mut self, preg: regalloc2::PReg) -> bool {
            if preg.class() != regalloc2::RegClass::Int {
                return false;
            }
            let r = preg.hw_enc() as u8;
            if r == 9 {
                return true;
            }
            dynasm!(self.ectx.ops ; .arch aarch64 ; mov x9, X(r));
            true
        }

        fn emit_mov_preg_from_x9(&mut self, preg: regalloc2::PReg) -> bool {
            if preg.class() != regalloc2::RegClass::Int {
                return false;
            }
            let r = preg.hw_enc() as u8;
            if r == 9 {
                return true;
            }
            dynasm!(self.ectx.ops ; .arch aarch64 ; mov X(r), x9);
            true
        }

        fn emit_mov_preg_to_preg(&mut self, from: PReg, to: PReg) -> bool {
            if from == to {
                return true;
            }
            if from.class() != RegClass::Int || to.class() != RegClass::Int {
                return false;
            }
            let from_r = from.hw_enc() as u8;
            let to_r = to.hw_enc() as u8;
            dynasm!(self.ectx.ops ; .arch aarch64 ; mov X(to_r), X(from_r));
            true
        }

        fn emit_store_stack_from_preg(&mut self, preg: regalloc2::PReg, off: u32) -> bool {
            if preg.class() != regalloc2::RegClass::Int {
                return false;
            }
            let r = preg.hw_enc() as u8;
            dynasm!(self.ectx.ops ; .arch aarch64 ; str X(r), [sp, #off]);
            true
        }

        fn emit_load_preg_from_stack(&mut self, preg: regalloc2::PReg, off: u32) -> bool {
            if preg.class() != regalloc2::RegClass::Int {
                return false;
            }
            let r = preg.hw_enc() as u8;
            dynasm!(self.ectx.ops ; .arch aarch64 ; ldr X(r), [sp, #off]);
            true
        }

        fn spill_off(&self, slot: regalloc2::SpillSlot) -> u32 {
            self.spill_base + (slot.index() as u32) * 8
        }

        fn emit_edit_move(&mut self, from: Allocation, to: Allocation) {
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
                    dynasm!(self.ectx.ops ; .arch aarch64 ; mov X(to_r), X(from_r));
                }
                (Some(from_reg), None, None, Some(to_stack)) => {
                    let off = self.spill_off(to_stack);
                    if self.emit_store_stack_from_preg(from_reg, off) {
                        return;
                    }
                    if !self.emit_mov_x9_from_preg(from_reg) {
                        return;
                    }
                    dynasm!(self.ectx.ops ; .arch aarch64 ; str x9, [sp, #off]);
                }
                (None, Some(from_stack), Some(to_reg), None) => {
                    let off = self.spill_off(from_stack);
                    if self.emit_load_preg_from_stack(to_reg, off) {
                        return;
                    }
                    dynasm!(self.ectx.ops ; .arch aarch64 ; ldr x9, [sp, #off]);
                    let _ = self.emit_mov_preg_from_x9(to_reg);
                }
                (None, Some(from_stack), None, Some(to_stack)) => {
                    if from_stack == to_stack {
                        return;
                    }
                    let from_off = self.spill_off(from_stack);
                    let to_off = self.spill_off(to_stack);
                    dynasm!(self.ectx.ops
                        ; .arch aarch64
                        ; ldr x9, [sp, #from_off]
                        ; str x9, [sp, #to_off]
                    );
                }
                _ => {}
            }
        }

        // r[impl ir.regalloc.edits]
        fn apply_regalloc_edits(&mut self, linear_op_index: usize, pos: InstPosition) {
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

            self.flush_all_vregs();
            for (from, to) in edits {
                self.emit_edit_move(from, to);
            }
        }

        fn has_inst_edits(&self, linear_op_index: usize) -> bool {
            let Some(lambda_id) = self
                .current_func
                .as_ref()
                .map(|f| f.lambda_id.index() as u32)
            else {
                return false;
            };
            let Some(by_lambda) = self.edits_by_lambda.get(&lambda_id) else {
                return false;
            };
            by_lambda.before.contains_key(&linear_op_index)
                || by_lambda.after.contains_key(&linear_op_index)
        }

        fn resolve_forwarded_label(&self, label: LabelId) -> LabelId {
            let Some(lambda_id) = self
                .current_func
                .as_ref()
                .map(|f| f.lambda_id.index() as u32)
            else {
                return label;
            };
            let Some(by_lambda) = self.forward_branch_labels_by_lambda.get(&lambda_id) else {
                return label;
            };
            let mut resolved = label;
            let mut hops = 0usize;
            while hops < 64 {
                let Some((branch_linear_op_index, next_label)) = by_lambda.get(&resolved).copied()
                else {
                    break;
                };
                if self.has_inst_edits(branch_linear_op_index)
                    || self.has_edge_edits(branch_linear_op_index, 0)
                    || next_label == resolved
                {
                    break;
                }
                resolved = next_label;
                hops += 1;
            }
            resolved
        }

        fn edge_edit_moves(
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
                // Edge blocks only contain an unconditional branch; both before/after edits
                // must execute before branching to the real CFG target.
                moves.extend(after.iter().copied());
            }
            moves
        }

        fn apply_fallthrough_edge_edits(&mut self, linear_op_index: usize, succ_index: usize) {
            let moves = self.edge_edit_moves(linear_op_index, succ_index);
            if moves.is_empty() {
                return;
            }
            self.flush_all_vregs();
            for (from, to) in moves {
                self.emit_edit_move(from, to);
            }
        }

        fn has_edge_edits(&self, linear_op_index: usize, succ_index: usize) -> bool {
            !self.edge_edit_moves(linear_op_index, succ_index).is_empty()
        }

        fn edge_target_label(
            &mut self,
            linear_op_index: usize,
            succ_index: usize,
            actual_target: DynamicLabel,
        ) -> DynamicLabel {
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
            let has_edits =
                by_lambda.before.contains_key(&key) || by_lambda.after.contains_key(&key);
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

        fn emit_edge_trampolines(&mut self) {
            let trampolines = std::mem::take(&mut self.edge_trampolines);
            for trampoline in trampolines {
                self.ectx.bind_label(trampoline.label);
                if !trampoline.moves.is_empty() {
                    self.flush_all_vregs();
                    for (from, to) in trampoline.moves {
                        self.emit_edit_move(from, to);
                    }
                }
                self.ectx.emit_branch(trampoline.target);
            }
        }

        // r[impl ir.regalloc.no-boundary-flush]
        fn flush_all_vregs(&mut self) {
            self.const_vregs.fill(None);
        }

        fn emit_recipe_ops(&mut self, ops: Vec<Op>) {
            self.flush_all_vregs();
            self.ectx.emit_recipe(&Recipe {
                ops,
                label_count: 0,
            });
        }

        fn current_alloc(&self, operand_index: usize) -> Allocation {
            self.current_inst_allocs
                .as_ref()
                .and_then(|allocs| allocs.get(operand_index).copied())
                .unwrap_or_else(|| {
                    panic!("missing regalloc allocation for operand index {operand_index}")
                })
        }

        fn emit_load_x9_from_allocation(&mut self, alloc: Allocation) {
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
                dynasm!(self.ectx.ops ; .arch aarch64 ; ldr x9, [sp, #off]);
                return;
            }
            panic!("unexpected none allocation for x9 load");
        }

        fn emit_load_x10_from_allocation(&mut self, alloc: Allocation) {
            if let Some(reg) = alloc.as_reg() {
                assert!(
                    reg.class() == regalloc2::RegClass::Int,
                    "unsupported register allocation class {:?} for x10 load",
                    reg.class()
                );
                let r = reg.hw_enc() as u8;
                if r != 10 {
                    dynasm!(self.ectx.ops ; .arch aarch64 ; mov x10, X(r));
                }
                return;
            }
            if let Some(slot) = alloc.as_stack() {
                let off = self.spill_off(slot);
                dynasm!(self.ectx.ops ; .arch aarch64 ; ldr x10, [sp, #off]);
                return;
            }
            panic!("unexpected none allocation for x10 load");
        }

        fn emit_store_x9_to_allocation(&mut self, alloc: Allocation) -> bool {
            if let Some(reg) = alloc.as_reg() {
                return self.emit_mov_preg_from_x9(reg);
            }
            if let Some(slot) = alloc.as_stack() {
                let off = self.spill_off(slot);
                dynasm!(self.ectx.ops ; .arch aarch64 ; str x9, [sp, #off]);
                return true;
            }
            false
        }

        fn emit_load_use_x9(&mut self, v: crate::ir::VReg, operand_index: usize) {
            let _ = v;
            let alloc = self.current_alloc(operand_index);
            self.emit_load_x9_from_allocation(alloc);
        }

        fn emit_load_use_x10(&mut self, v: crate::ir::VReg, operand_index: usize) {
            let _ = v;
            let alloc = self.current_alloc(operand_index);
            self.emit_load_x10_from_allocation(alloc);
        }

        fn emit_store_def_x9(&mut self, _v: crate::ir::VReg, operand_index: usize) {
            let alloc = self.current_alloc(operand_index);
            let _ = self.emit_store_x9_to_allocation(alloc);
        }

        fn emit_set_abi_arg_from_allocation(&mut self, abi_arg: u8, operand_index: usize) {
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
                dynasm!(self.ectx.ops ; .arch aarch64 ; ldr X(target_r), [sp, #off]);
                return;
            }
            panic!("unexpected none allocation for CallLambda arg");
        }

        fn emit_capture_abi_ret_to_allocation(&mut self, abi_ret: u8, operand_index: usize) {
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

        fn emit_load_u32_w10(&mut self, value: u32) {
            let lo = value & 0xFFFF;
            let hi = (value >> 16) & 0xFFFF;
            dynasm!(self.ectx.ops ; .arch aarch64 ; movz w10, #lo);
            if value > 0xFFFF {
                dynasm!(self.ectx.ops ; .arch aarch64 ; movk w10, #hi, LSL #16);
            }
        }

        fn emit_load_u64_x10(&mut self, value: u64) {
            let p0 = (value & 0xFFFF) as u32;
            let p1 = ((value >> 16) & 0xFFFF) as u32;
            let p2 = ((value >> 32) & 0xFFFF) as u32;
            let p3 = ((value >> 48) & 0xFFFF) as u32;
            dynasm!(self.ectx.ops ; .arch aarch64 ; movz x10, #p0);
            if p1 != 0 {
                dynasm!(self.ectx.ops ; .arch aarch64 ; movk x10, #p1, LSL #16);
            }
            if p2 != 0 {
                dynasm!(self.ectx.ops ; .arch aarch64 ; movk x10, #p2, LSL #32);
            }
            if p3 != 0 {
                dynasm!(self.ectx.ops ; .arch aarch64 ; movk x10, #p3, LSL #48);
            }
        }

        fn emit_load_u64_x9(&mut self, value: u64) {
            let p0 = (value & 0xFFFF) as u32;
            let p1 = ((value >> 16) & 0xFFFF) as u32;
            let p2 = ((value >> 32) & 0xFFFF) as u32;
            let p3 = ((value >> 48) & 0xFFFF) as u32;
            dynasm!(self.ectx.ops ; .arch aarch64 ; movz x9, #p0);
            if p1 != 0 {
                dynasm!(self.ectx.ops ; .arch aarch64 ; movk x9, #p1, LSL #16);
            }
            if p2 != 0 {
                dynasm!(self.ectx.ops ; .arch aarch64 ; movk x9, #p2, LSL #32);
            }
            if p3 != 0 {
                dynasm!(self.ectx.ops ; .arch aarch64 ; movk x9, #p3, LSL #48);
            }
        }

        fn const_of(&self, v: crate::ir::VReg) -> Option<u64> {
            self.const_vregs[v.index()]
        }

        fn set_const(&mut self, v: crate::ir::VReg, value: Option<u64>) {
            self.const_vregs[v.index()] = value;
        }

        fn emit_read_from_field(&mut self, dst: crate::ir::VReg, offset: u32, width: Width) {
            match width {
                Width::W1 => dynasm!(self.ectx.ops ; .arch aarch64 ; ldrb w9, [x21, #offset]),
                Width::W2 => dynasm!(self.ectx.ops ; .arch aarch64 ; ldrh w9, [x21, #offset]),
                Width::W4 => dynasm!(self.ectx.ops ; .arch aarch64 ; ldr w9, [x21, #offset]),
                Width::W8 => dynasm!(self.ectx.ops ; .arch aarch64 ; ldr x9, [x21, #offset]),
            }
            self.emit_store_def_x9(dst, 0);
            self.set_const(dst, None);
        }

        fn emit_write_to_field(&mut self, src: crate::ir::VReg, offset: u32, width: Width) {
            self.emit_load_use_x9(src, 0);
            match width {
                Width::W1 => dynasm!(self.ectx.ops ; .arch aarch64 ; strb w9, [x21, #offset]),
                Width::W2 => dynasm!(self.ectx.ops ; .arch aarch64 ; strh w9, [x21, #offset]),
                Width::W4 => dynasm!(self.ectx.ops ; .arch aarch64 ; str w9, [x21, #offset]),
                Width::W8 => dynasm!(self.ectx.ops ; .arch aarch64 ; str x9, [x21, #offset]),
            }
        }

        fn emit_save_out_ptr(&mut self, dst: crate::ir::VReg) {
            dynasm!(self.ectx.ops ; .arch aarch64 ; mov x9, x21);
            self.emit_store_def_x9(dst, 0);
            self.set_const(dst, None);
        }

        fn emit_set_out_ptr(&mut self, src: crate::ir::VReg) {
            self.emit_load_use_x9(src, 0);
            dynasm!(self.ectx.ops ; .arch aarch64 ; mov x21, x9);
        }

        fn emit_slot_addr(&mut self, dst: crate::ir::VReg, slot: crate::ir::SlotId) {
            let slot_off = self.slot_off(slot);
            dynasm!(self.ectx.ops
                ; .arch aarch64
                ; add x9, sp, #slot_off
            );
            self.emit_store_def_x9(dst, 0);
            self.set_const(dst, None);
        }

        fn emit_read_bytes(&mut self, dst: crate::ir::VReg, count: u32) {
            self.emit_recipe_ops(vec![Op::BoundsCheck { count }]);
            match count {
                1 => dynasm!(self.ectx.ops ; .arch aarch64 ; ldrb w9, [x19]),
                2 => dynasm!(self.ectx.ops ; .arch aarch64 ; ldrh w9, [x19]),
                4 => dynasm!(self.ectx.ops ; .arch aarch64 ; ldr w9, [x19]),
                8 => dynasm!(self.ectx.ops ; .arch aarch64 ; ldr x9, [x19]),
                _ => panic!("unsupported ReadBytes count: {count}"),
            }
            self.emit_store_def_x9(dst, 0);
            self.set_const(dst, None);
            self.ectx.emit_advance_cursor_by(count);
        }

        fn emit_peek_byte(&mut self, dst: crate::ir::VReg) {
            self.emit_recipe_ops(vec![Op::BoundsCheck { count: 1 }]);
            dynasm!(self.ectx.ops ; .arch aarch64 ; ldrb w9, [x19]);
            self.emit_store_def_x9(dst, 0);
            self.set_const(dst, None);
        }

        fn emit_binop(
            &mut self,
            kind: BinOpKind,
            dst: crate::ir::VReg,
            lhs: crate::ir::VReg,
            rhs: crate::ir::VReg,
            materialize_cmpne_result: bool,
        ) {
            if kind == BinOpKind::CmpNe {
                let lhs_alloc = self.current_alloc(0);
                let rhs_alloc = self.current_alloc(1);
                let rhs_const = self.const_of(rhs);

                if let Some(reg) = lhs_alloc.as_reg() {
                    assert!(
                        reg.class() == regalloc2::RegClass::Int,
                        "unsupported register allocation class {:?} for CmpNe lhs",
                        reg.class()
                    );
                }
                if let Some(reg) = rhs_alloc.as_reg() {
                    assert!(
                        reg.class() == regalloc2::RegClass::Int,
                        "unsupported register allocation class {:?} for CmpNe rhs",
                        reg.class()
                    );
                }

                match (
                    lhs_alloc.as_reg(),
                    lhs_alloc.as_stack(),
                    rhs_const,
                    rhs_alloc.as_reg(),
                    rhs_alloc.as_stack(),
                ) {
                    (Some(lhs_reg), None, Some(c), _, _) => {
                        let lhs_r = lhs_reg.hw_enc() as u8;
                        self.emit_load_u64_x10(c);
                        dynasm!(self.ectx.ops ; .arch aarch64 ; cmp X(lhs_r), x10);
                    }
                    (None, Some(lhs_stack), Some(c), _, _) => {
                        let lhs_off = self.spill_off(lhs_stack);
                        dynasm!(self.ectx.ops ; .arch aarch64 ; ldr x9, [sp, #lhs_off]);
                        if c <= 4095 {
                            dynasm!(self.ectx.ops ; .arch aarch64 ; cmp x9, c as u32);
                        } else {
                            self.emit_load_u64_x10(c);
                            dynasm!(self.ectx.ops ; .arch aarch64 ; cmp x9, x10);
                        }
                    }
                    (Some(lhs_reg), None, None, Some(rhs_reg), None) => {
                        let lhs_r = lhs_reg.hw_enc() as u8;
                        let rhs_r = rhs_reg.hw_enc() as u8;
                        dynasm!(self.ectx.ops ; .arch aarch64 ; cmp X(lhs_r), X(rhs_r));
                    }
                    (Some(lhs_reg), None, None, None, Some(rhs_stack)) => {
                        let lhs_r = lhs_reg.hw_enc() as u8;
                        let rhs_off = self.spill_off(rhs_stack);
                        dynasm!(self.ectx.ops
                            ; .arch aarch64
                            ; ldr x10, [sp, #rhs_off]
                            ; cmp X(lhs_r), x10
                        );
                    }
                    (None, Some(lhs_stack), None, Some(rhs_reg), None) => {
                        let lhs_off = self.spill_off(lhs_stack);
                        let rhs_r = rhs_reg.hw_enc() as u8;
                        dynasm!(self.ectx.ops
                            ; .arch aarch64
                            ; ldr x9, [sp, #lhs_off]
                            ; cmp x9, X(rhs_r)
                        );
                    }
                    (None, Some(lhs_stack), None, None, Some(rhs_stack)) => {
                        let lhs_off = self.spill_off(lhs_stack);
                        let rhs_off = self.spill_off(rhs_stack);
                        dynasm!(self.ectx.ops
                            ; .arch aarch64
                            ; ldr x9, [sp, #lhs_off]
                            ; ldr x10, [sp, #rhs_off]
                            ; cmp x9, x10
                        );
                    }
                    _ => panic!("unexpected none allocation for CmpNe operands"),
                }

                if !materialize_cmpne_result {
                    self.set_const(dst, None);
                    return;
                }

                let dst_alloc = self.current_alloc(2);
                if let Some(dst_reg) = dst_alloc.as_reg() {
                    assert!(
                        dst_reg.class() == regalloc2::RegClass::Int,
                        "unsupported register allocation class {:?} for CmpNe dst",
                        dst_reg.class()
                    );
                    let dst_r = dst_reg.hw_enc() as u8;
                    dynasm!(self.ectx.ops ; .arch aarch64 ; cset X(dst_r), ne);
                } else if let Some(dst_stack) = dst_alloc.as_stack() {
                    let dst_off = self.spill_off(dst_stack);
                    dynasm!(self.ectx.ops
                        ; .arch aarch64
                        ; cset x9, ne
                        ; str x9, [sp, #dst_off]
                    );
                } else {
                    panic!("unexpected none allocation for CmpNe dst");
                }
                self.set_const(dst, None);
                return;
            }

            let lhs_alloc = self.current_alloc(0);
            let rhs_alloc = self.current_alloc(1);
            let dst_alloc = self.current_alloc(2);
            if let (Some(lhs_reg), Some(rhs_reg), Some(dst_reg)) =
                (lhs_alloc.as_reg(), rhs_alloc.as_reg(), dst_alloc.as_reg())
            {
                assert!(
                    lhs_reg.class() == regalloc2::RegClass::Int,
                    "unsupported register allocation class {:?} for binop lhs",
                    lhs_reg.class()
                );
                assert!(
                    rhs_reg.class() == regalloc2::RegClass::Int,
                    "unsupported register allocation class {:?} for binop rhs",
                    rhs_reg.class()
                );
                assert!(
                    dst_reg.class() == regalloc2::RegClass::Int,
                    "unsupported register allocation class {:?} for binop dst",
                    dst_reg.class()
                );

                let lhs_r = lhs_reg.hw_enc() as u8;
                let rhs_r = rhs_reg.hw_enc() as u8;
                let dst_r = dst_reg.hw_enc() as u8;

                let handled = match kind {
                    BinOpKind::Add => {
                        if dst_r == lhs_r {
                            dynasm!(self.ectx.ops ; .arch aarch64 ; add X(dst_r), X(dst_r), X(rhs_r));
                        } else if dst_r == rhs_r {
                            dynasm!(self.ectx.ops ; .arch aarch64 ; add X(dst_r), X(dst_r), X(lhs_r));
                        } else {
                            dynasm!(self.ectx.ops
                                ; .arch aarch64
                                ; mov X(dst_r), X(lhs_r)
                                ; add X(dst_r), X(dst_r), X(rhs_r)
                            );
                        }
                        true
                    }
                    BinOpKind::Sub => {
                        if dst_r == lhs_r {
                            dynasm!(self.ectx.ops ; .arch aarch64 ; sub X(dst_r), X(dst_r), X(rhs_r));
                            true
                        } else if dst_r != rhs_r {
                            dynasm!(self.ectx.ops
                                ; .arch aarch64
                                ; mov X(dst_r), X(lhs_r)
                                ; sub X(dst_r), X(dst_r), X(rhs_r)
                            );
                            true
                        } else {
                            false
                        }
                    }
                    BinOpKind::And => {
                        if dst_r == lhs_r {
                            dynasm!(self.ectx.ops ; .arch aarch64 ; and X(dst_r), X(dst_r), X(rhs_r));
                        } else if dst_r == rhs_r {
                            dynasm!(self.ectx.ops ; .arch aarch64 ; and X(dst_r), X(dst_r), X(lhs_r));
                        } else {
                            dynasm!(self.ectx.ops
                                ; .arch aarch64
                                ; mov X(dst_r), X(lhs_r)
                                ; and X(dst_r), X(dst_r), X(rhs_r)
                            );
                        }
                        true
                    }
                    BinOpKind::Or => {
                        if dst_r == lhs_r {
                            dynasm!(self.ectx.ops ; .arch aarch64 ; orr X(dst_r), X(dst_r), X(rhs_r));
                        } else if dst_r == rhs_r {
                            dynasm!(self.ectx.ops ; .arch aarch64 ; orr X(dst_r), X(dst_r), X(lhs_r));
                        } else {
                            dynasm!(self.ectx.ops
                                ; .arch aarch64
                                ; mov X(dst_r), X(lhs_r)
                                ; orr X(dst_r), X(dst_r), X(rhs_r)
                            );
                        }
                        true
                    }
                    BinOpKind::Xor => {
                        if dst_r == lhs_r {
                            dynasm!(self.ectx.ops ; .arch aarch64 ; eor X(dst_r), X(dst_r), X(rhs_r));
                        } else if dst_r == rhs_r {
                            dynasm!(self.ectx.ops ; .arch aarch64 ; eor X(dst_r), X(dst_r), X(lhs_r));
                        } else {
                            dynasm!(self.ectx.ops
                                ; .arch aarch64
                                ; mov X(dst_r), X(lhs_r)
                                ; eor X(dst_r), X(dst_r), X(rhs_r)
                            );
                        }
                        true
                    }
                    BinOpKind::Shr => {
                        if dst_r == lhs_r {
                            dynasm!(self.ectx.ops ; .arch aarch64 ; lsr X(dst_r), X(dst_r), X(rhs_r));
                            true
                        } else if dst_r != rhs_r {
                            dynasm!(self.ectx.ops
                                ; .arch aarch64
                                ; mov X(dst_r), X(lhs_r)
                                ; lsr X(dst_r), X(dst_r), X(rhs_r)
                            );
                            true
                        } else {
                            false
                        }
                    }
                    BinOpKind::Shl => {
                        if dst_r == lhs_r {
                            dynasm!(self.ectx.ops ; .arch aarch64 ; lsl X(dst_r), X(dst_r), X(rhs_r));
                            true
                        } else if dst_r != rhs_r {
                            dynasm!(self.ectx.ops
                                ; .arch aarch64
                                ; mov X(dst_r), X(lhs_r)
                                ; lsl X(dst_r), X(dst_r), X(rhs_r)
                            );
                            true
                        } else {
                            false
                        }
                    }
                    BinOpKind::CmpNe => unreachable!("CmpNe handled above"),
                };
                if handled {
                    self.set_const(dst, None);
                    return;
                }
            }

            self.emit_load_use_x9(lhs, 0);
            let rhs_const = self.const_of(rhs);
            match kind {
                BinOpKind::Add => {
                    if let Some(c) = rhs_const
                        && c <= 4095
                    {
                        dynasm!(self.ectx.ops ; .arch aarch64 ; add x9, x9, c as u32);
                    } else {
                        self.emit_load_use_x10(rhs, 1);
                        dynasm!(self.ectx.ops ; .arch aarch64 ; add x9, x9, x10);
                    }
                }
                BinOpKind::Sub => {
                    if let Some(c) = rhs_const
                        && c <= 4095
                    {
                        dynasm!(self.ectx.ops ; .arch aarch64 ; sub x9, x9, c as u32);
                    } else {
                        self.emit_load_use_x10(rhs, 1);
                        dynasm!(self.ectx.ops ; .arch aarch64 ; sub x9, x9, x10);
                    }
                }
                BinOpKind::And => {
                    if matches!(rhs_const, Some(0x7f | 0x7e | 0x80 | 0x1)) {
                        let c = rhs_const.expect("just matched Some");
                        dynasm!(self.ectx.ops ; .arch aarch64 ; and x9, x9, c);
                    } else {
                        self.emit_load_use_x10(rhs, 1);
                        dynasm!(self.ectx.ops ; .arch aarch64 ; and x9, x9, x10);
                    }
                }
                BinOpKind::Or => {
                    self.emit_load_use_x10(rhs, 1);
                    dynasm!(self.ectx.ops ; .arch aarch64 ; orr x9, x9, x10);
                }
                BinOpKind::Xor => {
                    self.emit_load_use_x10(rhs, 1);
                    dynasm!(self.ectx.ops ; .arch aarch64 ; eor x9, x9, x10);
                }
                BinOpKind::CmpNe => unreachable!("CmpNe handled above"),
                BinOpKind::Shr => {
                    if let Some(c) = rhs_const
                        && c <= 63
                    {
                        dynasm!(self.ectx.ops ; .arch aarch64 ; lsr x9, x9, c as u32);
                    } else {
                        self.emit_load_use_x10(rhs, 1);
                        dynasm!(self.ectx.ops ; .arch aarch64 ; lsr x9, x9, x10);
                    }
                }
                BinOpKind::Shl => {
                    if let Some(c) = rhs_const
                        && c <= 63
                    {
                        dynasm!(self.ectx.ops ; .arch aarch64 ; lsl x9, x9, c as u32);
                    } else {
                        self.emit_load_use_x10(rhs, 1);
                        dynasm!(self.ectx.ops ; .arch aarch64 ; lsl x9, x9, x10);
                    }
                }
            }
            self.emit_store_def_x9(dst, 2);
            self.set_const(dst, None);
        }

        fn emit_unary(&mut self, kind: UnaryOpKind, dst: crate::ir::VReg, src: crate::ir::VReg) {
            self.emit_load_use_x9(src, 0);
            match kind {
                UnaryOpKind::ZigzagDecode { wide: true } => {
                    dynasm!(self.ectx.ops
                        ; .arch aarch64
                        ; lsr x10, x9, #1
                        ; and x16, x9, #1
                        ; neg x16, x16
                        ; eor x9, x10, x16
                    );
                }
                UnaryOpKind::ZigzagDecode { wide: false } => {
                    dynasm!(self.ectx.ops
                        ; .arch aarch64
                        ; lsr w10, w9, #1
                        ; and w16, w9, #1
                        ; neg w16, w16
                        ; eor w9, w10, w16
                    );
                }
                UnaryOpKind::SignExtend { from_width } => match from_width {
                    Width::W1 => dynasm!(self.ectx.ops ; .arch aarch64 ; sxtb x9, w9),
                    Width::W2 => dynasm!(self.ectx.ops ; .arch aarch64 ; sxth x9, w9),
                    Width::W4 => dynasm!(self.ectx.ops ; .arch aarch64 ; sxtw x9, w9),
                    Width::W8 => {}
                },
            }
            self.emit_store_def_x9(dst, 1);
            self.set_const(dst, None);
        }

        fn emit_branch_if(&mut self, cond: crate::ir::VReg, target: DynamicLabel, invert: bool) {
            let _ = cond;
            let alloc = self.current_alloc(0);
            self.emit_branch_if_allocation(alloc, target, invert);
        }

        fn emit_branch_if_allocation(
            &mut self,
            alloc: Allocation,
            target: DynamicLabel,
            invert: bool,
        ) {
            if let Some(reg) = alloc.as_reg() {
                assert!(
                    reg.class() == regalloc2::RegClass::Int,
                    "unsupported register allocation class {:?} for branch condition",
                    reg.class()
                );
                let r = reg.hw_enc() as u8;
                if invert {
                    dynasm!(self.ectx.ops ; .arch aarch64 ; cbz X(r), =>target);
                } else {
                    dynasm!(self.ectx.ops ; .arch aarch64 ; cbnz X(r), =>target);
                }
                return;
            }
            if let Some(slot) = alloc.as_stack() {
                let off = self.spill_off(slot);
                dynasm!(self.ectx.ops ; .arch aarch64 ; ldr x9, [sp, #off]);
                if invert {
                    dynasm!(self.ectx.ops ; .arch aarch64 ; cbz x9, =>target);
                } else {
                    dynasm!(self.ectx.ops ; .arch aarch64 ; cbnz x9, =>target);
                }
                return;
            }
            panic!("unexpected none allocation for branch condition");
        }

        fn emit_branch_on_last_cmp_ne(&mut self, target: DynamicLabel, invert: bool) {
            if invert {
                dynasm!(self.ectx.ops ; .arch aarch64 ; b.eq =>target);
            } else {
                dynasm!(self.ectx.ops ; .arch aarch64 ; b.ne =>target);
            }
        }

        fn emit_jump_table(
            &mut self,
            predicate: crate::ir::VReg,
            labels: &[LabelId],
            default: LabelId,
            linear_op_index: usize,
        ) {
            let _ = predicate;
            let alloc = self.current_alloc(0);
            let pred_reg = alloc.as_reg().map(|r| {
                assert!(
                    r.class() == regalloc2::RegClass::Int,
                    "unsupported register allocation class {:?} for jumptable predicate",
                    r.class()
                );
                r.hw_enc() as u8
            });
            if let Some(slot) = alloc.as_stack() {
                let off = self.spill_off(slot);
                dynasm!(self.ectx.ops ; .arch aarch64 ; ldr x9, [sp, #off]);
            } else if pred_reg.is_none() {
                panic!("unexpected none allocation for jumptable predicate");
            }
            for (index, label) in labels.iter().enumerate() {
                let resolved = self.resolve_forwarded_label(*label);
                let target = self.edge_target_label(linear_op_index, index, self.label(resolved));
                let idx = index as u32;
                if let Some(r) = pred_reg {
                    self.emit_load_u32_w10(idx);
                    dynasm!(self.ectx.ops
                        ; .arch aarch64
                        ; cmp X(r), x10
                        ; b.eq =>target
                    );
                } else if idx <= 4095 {
                    dynasm!(self.ectx.ops
                        ; .arch aarch64
                        ; cmp w9, idx
                        ; b.eq =>target
                    );
                } else {
                    self.emit_load_u32_w10(idx);
                    dynasm!(self.ectx.ops
                        ; .arch aarch64
                        ; cmp w9, w10
                        ; b.eq =>target
                    );
                }
            }
            let default_succ_index = labels.len();
            let resolved_default = self.resolve_forwarded_label(default);
            let default_target = self.edge_target_label(
                linear_op_index,
                default_succ_index,
                self.label(resolved_default),
            );
            self.ectx.emit_branch(default_target);
        }

        fn emit_set_abi_reg_from_intrinsic_arg(&mut self, abi_arg: u8, arg: IntrinsicArg) {
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

        fn emit_call_intrinsic_with_args(&mut self, fn_ptr: *const u8, args: &[IntrinsicArg]) {
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

            // Flush cursor before call.
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

            // Reload cursor and branch to error exit if needed.
            dynasm!(self.ectx.ops
                ; .arch aarch64
                ; ldr x19, [x22, #CTX_INPUT_PTR]
                ; ldr w9, [x22, #CTX_ERROR_CODE]
                ; cbnz w9, =>error_exit
            );
        }

        // r[impl ir.intrinsics]
        fn emit_call_intrinsic(
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
                    // Return-value intrinsic ABI: fn(ctx, args...) -> value
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
                    // Side-effect intrinsic ABI: fn(ctx, args..., out+field_offset)
                    call_args.push(IntrinsicArg::OutField(field_offset));
                    self.emit_call_intrinsic_with_args(fn_ptr, &call_args);
                }
            }
        }

        fn emit_call_pure_with_args(&mut self, fn_ptr: *const u8, args: &[IntrinsicArg]) {
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

        fn emit_call_pure(
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

        fn emit_store_incoming_lambda_args(&mut self, data_args: &[crate::ir::VReg]) {
            const MAX_LAMBDA_DATA_ARGS: usize = 6;
            if data_args.len() > MAX_LAMBDA_DATA_ARGS {
                panic!(
                    "aarch64 CallLambda supports at most {MAX_LAMBDA_DATA_ARGS} data args, got {}",
                    data_args.len()
                );
            }
            // Incoming lambda args already arrive in ABI arg regs x2..x7.
            // Regalloc edits around the first mapped linear op place them as needed.
        }

        fn emit_load_lambda_results_to_ret_regs(
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

        fn emit_call_lambda(
            &mut self,
            label: DynamicLabel,
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

        fn run(mut self, ir: &LinearIr) -> LinearBackendResult {
            let mut fused_cmpne_cond = None::<(usize, crate::ir::VReg)>;
            let mut skipped_ops = BTreeSet::<usize>::new();
            let mut pending_and_branch = None::<(usize, crate::ir::VReg, Allocation, Allocation)>;
            for (op_index, op) in ir.ops.iter().enumerate() {
                if let Some((expected_branch_op_index, expected_cond)) = fused_cmpne_cond {
                    if op_index > expected_branch_op_index {
                        panic!(
                            "fused CmpNe for vreg {:?} expected branch op index {}, reached {}",
                            expected_cond, expected_branch_op_index, op_index
                        );
                    }
                    if op_index == expected_branch_op_index
                        && !matches!(
                            op,
                            LinearOp::BranchIf { cond, .. } | LinearOp::BranchIfZero { cond, .. }
                                if *cond == expected_cond
                        )
                    {
                        panic!(
                            "fused CmpNe for vreg {:?} must target BranchIf/BranchIfZero",
                            expected_cond
                        );
                    }
                }
                if let Some((expected_branch_op_index, _, _, _)) = pending_and_branch
                    && op_index > expected_branch_op_index
                {
                    panic!(
                        "pending and-branch peephole expected branch op index {}, reached {}",
                        expected_branch_op_index, op_index
                    );
                }
                let linear_op_index = if self.current_func.is_some()
                    && !matches!(op, LinearOp::FuncStart { .. } | LinearOp::FuncEnd)
                {
                    Some(self.current_lambda_linear_op_index)
                } else {
                    None
                };
                let fuse_cmpne_to_branch_op_index = match op {
                    LinearOp::BinOp {
                        op: BinOpKind::CmpNe,
                        dst,
                        ..
                    } => linear_op_index.and_then(|lin_idx| {
                        self.find_cmpne_branch_use(ir, op_index, *dst, lin_idx)
                    }),
                    _ => None,
                };
                self.current_inst_allocs = linear_op_index.and_then(|lin_idx| {
                    let lambda_id = self.current_func.as_ref()?.lambda_id.index() as u32;
                    self.allocs_by_lambda
                        .get(&lambda_id)
                        .and_then(|by_lambda| by_lambda.get(&lin_idx))
                        .cloned()
                });
                if let Some(linear_op_index) = linear_op_index {
                    self.apply_regalloc_edits(linear_op_index, InstPosition::Before);
                }

                if skipped_ops.contains(&op_index) {
                    // Branch was folded into the previous conditional branch.
                    // Keep linear-op accounting intact, but emit no machine code.
                } else {
                    match op {
                        LinearOp::FuncStart {
                            lambda_id,
                            data_args,
                            data_results,
                            ..
                        } => {
                            self.flush_all_vregs();
                            let label = self.lambda_labels[lambda_id.index()];
                            self.ectx.bind_label(label);
                            let (entry_offset, error_exit) = self.ectx.begin_func();
                            if lambda_id.index() == 0 {
                                self.entry = Some(entry_offset);
                            }
                            self.current_func = Some(FunctionCtx {
                                error_exit,
                                data_results: data_results.clone(),
                                lambda_id: *lambda_id,
                            });
                            self.current_lambda_linear_op_index = 0;
                            self.emit_store_incoming_lambda_args(data_args);
                        }
                        LinearOp::FuncEnd => {
                            self.flush_all_vregs();
                            let func = self
                                .current_func
                                .take()
                                .expect("FuncEnd without active function");
                            self.emit_load_lambda_results_to_ret_regs(
                                func.lambda_id,
                                &func.data_results,
                            );
                            self.ectx.end_func(func.error_exit);
                        }
                        LinearOp::Label(label) => {
                            self.flush_all_vregs();
                            self.ectx.bind_label(self.label(*label));
                        }
                        LinearOp::Branch(target) => {
                            let resolved_target = self.resolve_forwarded_label(*target);
                            let target_label = if let Some(lin_idx) = linear_op_index {
                                self.edge_target_label(lin_idx, 0, self.label(resolved_target))
                            } else {
                                self.label(resolved_target)
                            };
                            let is_redundant_fallthrough = if let Some(lin_idx) = linear_op_index {
                                if self.has_edge_edits(lin_idx, 0) {
                                    false
                                } else if let Some(LinearOp::Label(next_label)) =
                                    ir.ops.get(op_index + 1)
                                {
                                    let resolved_next = self.resolve_forwarded_label(*next_label);
                                    resolved_target == resolved_next
                                } else {
                                    false
                                }
                            } else {
                                false
                            };
                            if !is_redundant_fallthrough {
                                self.ectx.emit_branch(target_label);
                            }
                        }
                        LinearOp::BranchIf { cond, target } => {
                            let use_cmp_flags = match fused_cmpne_cond {
                                Some((expected_op_index, expected))
                                    if expected_op_index == op_index && expected == *cond =>
                                {
                                    fused_cmpne_cond = None;
                                    true
                                }
                                Some((expected_op_index, expected))
                                    if expected_op_index == op_index =>
                                {
                                    panic!(
                                        "fused CmpNe expected branch on {:?}, got {:?}",
                                        expected, cond
                                    )
                                }
                                None => false,
                                Some(_) => false,
                            };
                            let and_branch_allocs = match pending_and_branch {
                                Some((expected_op_index, expected_cond, lhs_alloc, rhs_alloc))
                                    if expected_op_index == op_index && expected_cond == *cond =>
                                {
                                    pending_and_branch = None;
                                    Some((lhs_alloc, rhs_alloc))
                                }
                                Some((expected_op_index, expected_cond, _, _))
                                    if expected_op_index == op_index =>
                                {
                                    panic!(
                                        "pending and-branch expected condition {:?}, got {:?}",
                                        expected_cond, cond
                                    );
                                }
                                Some(_) => None,
                                None => None,
                            };
                            let lin_idx = linear_op_index
                                .expect("BranchIf should have linear op index inside function");
                            let resolved_target = self.resolve_forwarded_label(*target);
                            let taken_target =
                                self.edge_target_label(lin_idx, 0, self.label(resolved_target));
                            let mut emitted_cond_branch = false;
                            if let Some((lhs_alloc, rhs_alloc)) = and_branch_allocs {
                                let fallthrough_cont = self.ectx.new_label();
                                self.emit_branch_if_allocation(lhs_alloc, fallthrough_cont, true);
                                self.emit_branch_if_allocation(rhs_alloc, taken_target, false);
                                self.ectx.bind_label(fallthrough_cont);
                                self.apply_fallthrough_edge_edits(lin_idx, 1);
                                emitted_cond_branch = true;
                            }
                            if !self.has_edge_edits(lin_idx, 1)
                                && !use_cmp_flags
                                && and_branch_allocs.is_none()
                                && let (
                                    Some(LinearOp::Branch(next_target)),
                                    Some(LinearOp::Label(next_label)),
                                ) = (ir.ops.get(op_index + 1), ir.ops.get(op_index + 2))
                            {
                                let next_lin_idx = lin_idx + 1;
                                if !self.has_inst_edits(next_lin_idx)
                                    && !self.has_edge_edits(next_lin_idx, 0)
                                {
                                    let resolved_false = self.resolve_forwarded_label(*next_target);
                                    let resolved_fallthrough =
                                        self.resolve_forwarded_label(*next_label);
                                    if resolved_fallthrough == resolved_false {
                                        self.emit_branch_if(*cond, taken_target, false);
                                        skipped_ops.insert(op_index + 1);
                                        emitted_cond_branch = true;
                                    } else if resolved_fallthrough == resolved_target {
                                        self.emit_branch_if(
                                            *cond,
                                            self.label(resolved_false),
                                            true,
                                        );
                                        skipped_ops.insert(op_index + 1);
                                        emitted_cond_branch = true;
                                    }
                                }
                            } else if !self.has_edge_edits(lin_idx, 1)
                                && use_cmp_flags
                                && and_branch_allocs.is_none()
                                && let (
                                    Some(LinearOp::Branch(next_target)),
                                    Some(LinearOp::Label(next_label)),
                                ) = (ir.ops.get(op_index + 1), ir.ops.get(op_index + 2))
                            {
                                let next_lin_idx = lin_idx + 1;
                                if !self.has_inst_edits(next_lin_idx)
                                    && !self.has_edge_edits(next_lin_idx, 0)
                                {
                                    let resolved_false = self.resolve_forwarded_label(*next_target);
                                    let resolved_fallthrough =
                                        self.resolve_forwarded_label(*next_label);
                                    if resolved_fallthrough == resolved_false {
                                        self.emit_branch_on_last_cmp_ne(taken_target, false);
                                        skipped_ops.insert(op_index + 1);
                                        emitted_cond_branch = true;
                                    } else if resolved_fallthrough == resolved_target {
                                        self.emit_branch_on_last_cmp_ne(
                                            self.label(resolved_false),
                                            true,
                                        );
                                        skipped_ops.insert(op_index + 1);
                                        emitted_cond_branch = true;
                                    }
                                }
                            }
                            if emitted_cond_branch {
                                // keep existing per-op edit handling below
                            } else if !use_cmp_flags {
                                if let Some(cond_const) = self.const_of(*cond) {
                                    if cond_const != 0 {
                                        self.ectx.emit_branch(taken_target);
                                    } else {
                                        self.apply_fallthrough_edge_edits(lin_idx, 1);
                                    }
                                } else {
                                    self.emit_branch_if(*cond, taken_target, false);
                                    self.apply_fallthrough_edge_edits(lin_idx, 1);
                                }
                            } else if self.has_edge_edits(lin_idx, 1) {
                                self.emit_branch_on_last_cmp_ne(taken_target, false);
                                self.apply_fallthrough_edge_edits(lin_idx, 1);
                            } else {
                                self.emit_branch_on_last_cmp_ne(taken_target, false);
                            }
                        }
                        LinearOp::BranchIfZero { cond, target } => {
                            let use_cmp_flags = match fused_cmpne_cond {
                                Some((expected_op_index, expected))
                                    if expected_op_index == op_index && expected == *cond =>
                                {
                                    fused_cmpne_cond = None;
                                    true
                                }
                                Some((expected_op_index, expected))
                                    if expected_op_index == op_index =>
                                {
                                    panic!(
                                        "fused CmpNe expected branch on {:?}, got {:?}",
                                        expected, cond
                                    )
                                }
                                None => false,
                                Some(_) => false,
                            };
                            let and_branch_allocs = match pending_and_branch {
                                Some((expected_op_index, expected_cond, lhs_alloc, rhs_alloc))
                                    if expected_op_index == op_index && expected_cond == *cond =>
                                {
                                    pending_and_branch = None;
                                    Some((lhs_alloc, rhs_alloc))
                                }
                                Some((expected_op_index, expected_cond, _, _))
                                    if expected_op_index == op_index =>
                                {
                                    panic!(
                                        "pending and-branch expected condition {:?}, got {:?}",
                                        expected_cond, cond
                                    );
                                }
                                Some(_) => None,
                                None => None,
                            };
                            let lin_idx = linear_op_index
                                .expect("BranchIfZero should have linear op index inside function");
                            let resolved_target = self.resolve_forwarded_label(*target);
                            let taken_target =
                                self.edge_target_label(lin_idx, 0, self.label(resolved_target));
                            let mut emitted_cond_branch = false;
                            if let Some((lhs_alloc, rhs_alloc)) = and_branch_allocs {
                                self.emit_branch_if_allocation(lhs_alloc, taken_target, true);
                                self.emit_branch_if_allocation(rhs_alloc, taken_target, true);
                                self.apply_fallthrough_edge_edits(lin_idx, 1);
                                emitted_cond_branch = true;
                            }
                            if !self.has_edge_edits(lin_idx, 1)
                                && !use_cmp_flags
                                && and_branch_allocs.is_none()
                                && let (
                                    Some(LinearOp::Branch(next_target)),
                                    Some(LinearOp::Label(next_label)),
                                ) = (ir.ops.get(op_index + 1), ir.ops.get(op_index + 2))
                            {
                                let next_lin_idx = lin_idx + 1;
                                if !self.has_inst_edits(next_lin_idx)
                                    && !self.has_edge_edits(next_lin_idx, 0)
                                {
                                    let resolved_false = self.resolve_forwarded_label(*next_target);
                                    let resolved_fallthrough =
                                        self.resolve_forwarded_label(*next_label);
                                    if resolved_fallthrough == resolved_false {
                                        self.emit_branch_if(*cond, taken_target, true);
                                        skipped_ops.insert(op_index + 1);
                                        emitted_cond_branch = true;
                                    } else if resolved_fallthrough == resolved_target {
                                        self.emit_branch_if(
                                            *cond,
                                            self.label(resolved_false),
                                            false,
                                        );
                                        skipped_ops.insert(op_index + 1);
                                        emitted_cond_branch = true;
                                    }
                                }
                            } else if !self.has_edge_edits(lin_idx, 1)
                                && use_cmp_flags
                                && and_branch_allocs.is_none()
                                && let (
                                    Some(LinearOp::Branch(next_target)),
                                    Some(LinearOp::Label(next_label)),
                                ) = (ir.ops.get(op_index + 1), ir.ops.get(op_index + 2))
                            {
                                let next_lin_idx = lin_idx + 1;
                                if !self.has_inst_edits(next_lin_idx)
                                    && !self.has_edge_edits(next_lin_idx, 0)
                                {
                                    let resolved_false = self.resolve_forwarded_label(*next_target);
                                    let resolved_fallthrough =
                                        self.resolve_forwarded_label(*next_label);
                                    if resolved_fallthrough == resolved_false {
                                        self.emit_branch_on_last_cmp_ne(taken_target, true);
                                        skipped_ops.insert(op_index + 1);
                                        emitted_cond_branch = true;
                                    } else if resolved_fallthrough == resolved_target {
                                        self.emit_branch_on_last_cmp_ne(
                                            self.label(resolved_false),
                                            false,
                                        );
                                        skipped_ops.insert(op_index + 1);
                                        emitted_cond_branch = true;
                                    }
                                }
                            }
                            if emitted_cond_branch {
                                // keep existing per-op edit handling below
                            } else if !use_cmp_flags {
                                if let Some(cond_const) = self.const_of(*cond) {
                                    if cond_const == 0 {
                                        self.ectx.emit_branch(taken_target);
                                    } else {
                                        self.apply_fallthrough_edge_edits(lin_idx, 1);
                                    }
                                } else {
                                    self.emit_branch_if(*cond, taken_target, true);
                                    self.apply_fallthrough_edge_edits(lin_idx, 1);
                                }
                            } else if self.has_edge_edits(lin_idx, 1) {
                                self.emit_branch_on_last_cmp_ne(taken_target, true);
                                self.apply_fallthrough_edge_edits(lin_idx, 1);
                            } else {
                                self.emit_branch_on_last_cmp_ne(taken_target, true);
                            }
                        }
                        LinearOp::JumpTable {
                            predicate,
                            labels,
                            default,
                        } => {
                            let lin_idx = linear_op_index
                                .expect("JumpTable should have linear op index inside function");
                            self.emit_jump_table(*predicate, labels, *default, lin_idx);
                        }

                        LinearOp::Const { dst, value } => {
                            self.emit_load_u64_x9(*value);
                            self.emit_store_def_x9(*dst, 0);
                            self.set_const(*dst, Some(*value));
                        }
                        LinearOp::Copy { dst, src } => {
                            let from = self.current_alloc(0);
                            let to = self.current_alloc(1);
                            self.emit_edit_move(from, to);
                            self.set_const(*dst, self.const_of(*src));
                        }
                        LinearOp::BinOp { op, dst, lhs, rhs } => {
                            let mut emitted_short_circuit_and_branch = false;
                            if *op == BinOpKind::CmpNe
                                && let Some(lin_idx) = linear_op_index
                                && !self.has_inst_edits(lin_idx)
                                && let (
                                    Some(LinearOp::BinOp {
                                        op: BinOpKind::CmpNe,
                                        dst: cmp1_dst,
                                        lhs: cmp1_lhs,
                                        rhs: cmp1_rhs,
                                    }),
                                    Some(LinearOp::BinOp {
                                        op: BinOpKind::And,
                                        dst: and_dst,
                                        lhs: and_lhs,
                                        rhs: and_rhs,
                                    }),
                                    Some(LinearOp::BranchIf { cond, target }),
                                ) = (
                                    ir.ops.get(op_index + 1),
                                    ir.ops.get(op_index + 2),
                                    ir.ops.get(op_index + 3),
                                )
                            {
                                let cmp_pair_used_by_and = (*and_lhs == *dst
                                    && *and_rhs == *cmp1_dst)
                                    || (*and_rhs == *dst && *and_lhs == *cmp1_dst);
                                if cmp_pair_used_by_and
                                    && *cond == *and_dst
                                    && Self::linear_ir_use_count_in_function(ir, op_index, *dst)
                                        == 1
                                    && Self::linear_ir_use_count_in_function(ir, op_index, *and_dst)
                                        == 1
                                    && !self.has_inst_edits(lin_idx + 1)
                                    && !self.has_inst_edits(lin_idx + 2)
                                    && !self.has_inst_edits(lin_idx + 3)
                                {
                                    let cmp1_allocs = self
                                        .allocs_for_linear_op(lin_idx + 1)
                                        .expect("missing allocs for second cmp in peephole");
                                    let cmp1_dst_alloc = cmp1_allocs
                                        .get(2)
                                        .copied()
                                        .expect("missing cmp1 dst alloc");

                                    let saved_allocs = self.current_inst_allocs.clone();
                                    self.current_inst_allocs = Some(cmp1_allocs);
                                    // Keep cmp1 boolean materialized, because it may be consumed
                                    // after the branch; this still lets us eliminate cmp0 cset/and.
                                    self.emit_binop(
                                        BinOpKind::CmpNe,
                                        *cmp1_dst,
                                        *cmp1_lhs,
                                        *cmp1_rhs,
                                        true,
                                    );
                                    self.current_inst_allocs = saved_allocs;

                                    self.emit_binop(*op, *dst, *lhs, *rhs, false);
                                    let fallthrough_cont = self.ectx.new_label();
                                    self.emit_branch_on_last_cmp_ne(fallthrough_cont, true);

                                    let resolved_target = self.resolve_forwarded_label(*target);
                                    let taken_target = self.edge_target_label(
                                        lin_idx + 3,
                                        0,
                                        self.label(resolved_target),
                                    );
                                    self.emit_branch_if_allocation(
                                        cmp1_dst_alloc,
                                        taken_target,
                                        false,
                                    );
                                    self.ectx.bind_label(fallthrough_cont);
                                    self.apply_fallthrough_edge_edits(lin_idx + 3, 1);
                                    self.set_const(*and_dst, None);
                                    skipped_ops.insert(op_index + 1);
                                    skipped_ops.insert(op_index + 2);
                                    skipped_ops.insert(op_index + 3);
                                    emitted_short_circuit_and_branch = true;
                                }
                            }

                            if !emitted_short_circuit_and_branch {
                                let mut skipped_and_materialization = false;
                                if *op == BinOpKind::And
                                    && let Some(lin_idx) = linear_op_index
                                    && !self.has_inst_edits(lin_idx)
                                    && Self::linear_ir_use_count_in_function(ir, op_index, *dst)
                                        == 1
                                    && let Some(branch_op_index) = self.find_and_branch_use(
                                        ir, op_index, *dst, *lhs, *rhs, lin_idx,
                                    )
                                {
                                    let lhs_alloc = self.current_alloc(0);
                                    let rhs_alloc = self.current_alloc(1);
                                    pending_and_branch =
                                        Some((branch_op_index, *dst, lhs_alloc, rhs_alloc));
                                    self.set_const(*dst, None);
                                    skipped_and_materialization = true;
                                }
                                if !skipped_and_materialization {
                                    self.emit_binop(
                                        *op,
                                        *dst,
                                        *lhs,
                                        *rhs,
                                        fuse_cmpne_to_branch_op_index.is_none(),
                                    );
                                    if let Some(branch_op_index) = fuse_cmpne_to_branch_op_index {
                                        fused_cmpne_cond = Some((branch_op_index, *dst));
                                    }
                                }
                            }
                        }
                        LinearOp::UnaryOp { op, dst, src } => self.emit_unary(*op, *dst, *src),

                        LinearOp::BoundsCheck { count } => {
                            self.emit_recipe_ops(vec![Op::BoundsCheck { count: *count }]);
                        }
                        LinearOp::ReadBytes { dst, count } => self.emit_read_bytes(*dst, *count),
                        LinearOp::PeekByte { dst } => self.emit_peek_byte(*dst),
                        LinearOp::AdvanceCursor { count } => {
                            self.ectx.emit_advance_cursor_by(*count)
                        }
                        LinearOp::AdvanceCursorBy { src } => {
                            self.emit_load_use_x9(*src, 0);
                            dynasm!(self.ectx.ops ; .arch aarch64 ; add x19, x19, x9);
                        }
                        LinearOp::SaveCursor { dst } => {
                            dynasm!(self.ectx.ops ; .arch aarch64 ; mov x9, x19);
                            self.emit_store_def_x9(*dst, 0);
                            self.set_const(*dst, None);
                        }
                        LinearOp::RestoreCursor { src } => {
                            self.emit_load_use_x9(*src, 0);
                            dynasm!(self.ectx.ops ; .arch aarch64 ; mov x19, x9);
                        }

                        LinearOp::WriteToField { src, offset, width } => {
                            self.emit_write_to_field(*src, *offset, *width);
                        }
                        LinearOp::ReadFromField { dst, offset, width } => {
                            self.emit_read_from_field(*dst, *offset, *width);
                            self.set_const(*dst, None);
                        }
                        LinearOp::SaveOutPtr { dst } => {
                            self.emit_save_out_ptr(*dst);
                            self.set_const(*dst, None);
                        }
                        LinearOp::SetOutPtr { src } => {
                            self.emit_set_out_ptr(*src);
                        }
                        LinearOp::SlotAddr { dst, slot } => {
                            self.emit_slot_addr(*dst, *slot);
                            self.set_const(*dst, None);
                        }
                        LinearOp::WriteToSlot { slot, src } => {
                            self.emit_load_use_x9(*src, 0);
                            let off = self.slot_off(*slot);
                            dynasm!(self.ectx.ops ; .arch aarch64 ; str x9, [sp, #off]);
                        }
                        LinearOp::ReadFromSlot { dst, slot } => {
                            let off = self.slot_off(*slot);
                            dynasm!(self.ectx.ops ; .arch aarch64 ; ldr x9, [sp, #off]);
                            self.emit_store_def_x9(*dst, 0);
                            self.set_const(*dst, None);
                        }

                        LinearOp::CallIntrinsic {
                            func,
                            args,
                            dst,
                            field_offset,
                        } => {
                            self.emit_call_intrinsic(*func, args, *dst, *field_offset);
                            if let Some(dst) = dst {
                                self.set_const(*dst, None);
                            }
                        }
                        LinearOp::CallPure { func, args, dst } => {
                            self.emit_call_pure(*func, args, *dst);
                            self.set_const(*dst, None);
                        }

                        LinearOp::ErrorExit { code } => {
                            self.flush_all_vregs();
                            self.ectx.emit_error(*code);
                        }

                        LinearOp::SimdStringScan { .. } | LinearOp::SimdWhitespaceSkip => {
                            panic!("unsupported SIMD op in linear backend adapter");
                        }

                        LinearOp::CallLambda {
                            target,
                            args,
                            results,
                        } => {
                            let label = self.lambda_labels[target.index()];
                            self.emit_call_lambda(label, args, results);
                            for &r in results {
                                self.set_const(r, None);
                            }
                        }
                    }
                }
                if let Some(linear_op_index) = linear_op_index {
                    self.apply_regalloc_edits(linear_op_index, InstPosition::After);
                    self.current_lambda_linear_op_index += 1;
                }
                self.current_inst_allocs = None;
            }

            if self.current_func.is_some() {
                panic!("unterminated function: missing FuncEnd");
            }
            if fused_cmpne_cond.is_some() {
                panic!("unterminated fused CmpNe/BranchIf pair");
            }

            self.emit_edge_trampolines();

            let entry = self.entry.expect("missing root FuncStart for lambda 0");
            let buf = self.ectx.finalize();
            LinearBackendResult { buf, entry }
        }
    }

    Lowerer::new(ir, max_spillslots, alloc).run(ir)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compiler;
    use crate::context::{DeserContext, ErrorCode};
    use crate::ir::{IntrinsicFn, IrBuilder, IrOp, Width};
    use crate::linearize::linearize;
    use facet::Facet;

    #[derive(Debug, Clone, serde::Serialize, serde::Deserialize, Facet, PartialEq)]
    struct ScalarVec {
        values: Vec<u32>,
    }

    fn run_u32_decoder(ir: &LinearIr, input: &[u8]) -> (u32, DeserContext) {
        let deser = compiler::compile_linear_ir_decoder(ir, false);
        let mut out = core::mem::MaybeUninit::<u32>::uninit();
        let mut ctx = DeserContext::from_bytes(input);
        unsafe {
            (deser.func())(out.as_mut_ptr() as *mut u8, &mut ctx);
            (out.assume_init(), ctx)
        }
    }

    fn run_u64_decoder(ir: &LinearIr, input: &[u8]) -> (u64, DeserContext) {
        let deser = compiler::compile_linear_ir_decoder(ir, false);
        let mut out = core::mem::MaybeUninit::<u64>::uninit();
        let mut ctx = DeserContext::from_bytes(input);
        unsafe {
            (deser.func())(out.as_mut_ptr() as *mut u8, &mut ctx);
            (out.assume_init(), ctx)
        }
    }

    fn run_decoder<'input, T: facet::Facet<'input>>(
        ir: &LinearIr,
        input: &'input [u8],
    ) -> (T, DeserContext) {
        let deser = compiler::compile_linear_ir_decoder(ir, false);
        let mut out = core::mem::MaybeUninit::<T>::uninit();
        let mut ctx = DeserContext::from_bytes(input);
        unsafe {
            (deser.func())(out.as_mut_ptr() as *mut u8, &mut ctx);
            (out.assume_init(), ctx)
        }
    }

    fn assert_ir_micro_snapshot(case_label: &str, ir: &LinearIr) {
        let deser = compiler::compile_linear_ir_decoder(ir, false);
        let mut out = String::new();
        out.push_str(&disasm_bytes(deser.code(), Some(deser.entry_offset())));
        insta::assert_snapshot!(
            format!("linear_ir_micro_{}_{}", case_label, std::env::consts::ARCH),
            out
        );
    }

    fn disasm_bytes(code: &[u8], marker_offset: Option<usize>) -> String {
        let mut out = String::new();

        #[cfg(target_arch = "aarch64")]
        {
            use std::fmt::Write;
            use yaxpeax_arch::{Decoder, U8Reader};
            use yaxpeax_arm::armv8::a64::InstDecoder;

            let decoder = InstDecoder::default();
            let mut reader = U8Reader::new(code);
            let mut offset = 0usize;
            let mut ret_count = 0u32;

            while offset + 4 <= code.len() {
                let prefix = if marker_offset == Some(offset) {
                    "> "
                } else {
                    "  "
                };
                match decoder.decode(&mut reader) {
                    Ok(inst) => {
                        let text = crate::disasm_normalize::normalize_inst(&format!("{inst}"));
                        writeln!(&mut out, "{prefix}{text}").unwrap();
                        if text.trim() == "ret" {
                            ret_count += 1;
                            if ret_count >= 2 {
                                break;
                            }
                        }
                    }
                    Err(e) => {
                        let word = u32::from_le_bytes(code[offset..offset + 4].try_into().unwrap());
                        writeln!(&mut out, "{prefix}<{e}> (0x{word:08x})").unwrap();
                    }
                }
                offset += 4;
            }
        }

        #[cfg(target_arch = "x86_64")]
        {
            use std::fmt::Write;
            use yaxpeax_arch::LengthedInstruction;
            use yaxpeax_arch::{Decoder, U8Reader};
            use yaxpeax_x86::amd64::InstDecoder;

            let decoder = InstDecoder::default();
            let mut reader = U8Reader::new(code);
            let mut offset = 0usize;
            let mut ret_count = 0u32;

            while offset < code.len() {
                let prefix = if marker_offset == Some(offset) {
                    "> "
                } else {
                    "  "
                };
                match decoder.decode(&mut reader) {
                    Ok(inst) => {
                        let len = inst.len().to_const() as usize;
                        let text = crate::disasm_normalize::normalize_inst(&format!("{inst}"));
                        writeln!(&mut out, "{prefix}{text}").unwrap();
                        if text.trim() == "ret" {
                            ret_count += 1;
                            if ret_count >= 2 {
                                break;
                            }
                        }
                        offset += len;
                    }
                    Err(_) => {
                        writeln!(&mut out, "{prefix}<decode error> (0x{:02x})", code[offset])
                            .unwrap();
                        offset += 1;
                    }
                }
            }
        }

        out
    }

    macro_rules! ir_micro_cases {
        (
            $(
                $name:ident => {
                    output: $out_ty:ty,
                    input: $input:expr,
                    expected: $expected:expr,
                    build: |$rb:ident| $build:block
                }
            ),+ $(,)?
        ) => {
            $(
                #[test]
                fn $name() {
                    let mut builder = IrBuilder::new(<$out_ty as facet::Facet>::SHAPE);
                    {
                        let mut $rb = builder.root_region();
                        $build
                    }
                    let mut func = builder.finish();
                    crate::ir_passes::run_default_passes(&mut func);
                    let lin = linearize(&mut func);
                    assert_ir_micro_snapshot(stringify!($name), &lin);

                    let (value, ctx): ($out_ty, DeserContext) = run_decoder(&lin, $input);
                    assert_eq!(ctx.error.code, 0);
                    assert_eq!(value, $expected);
                }
            )+
        };
    }

    unsafe extern "C" fn add3_intrinsic(
        _ctx: *mut crate::context::DeserContext,
        a: u64,
        b: u64,
        c: u64,
    ) -> u64 {
        a + b + c
    }

    ir_micro_cases! {
        linear_ir_micro_const_u32 => {
            output: u32,
            input: &[],
            expected: 42u32,
            build: |rb| {
                let v = rb.const_val(42);
                rb.write_to_field(v, 0, Width::W4);
                rb.set_results(&[]);
            }
        },
        linear_ir_micro_read_u32 => {
            output: u32,
            input: &[0x78, 0x56, 0x34, 0x12],
            expected: 0x1234_5678u32,
            build: |rb| {
                rb.bounds_check(4);
                let v = rb.read_bytes(4);
                rb.write_to_field(v, 0, Width::W4);
                rb.set_results(&[]);
            }
        },
        linear_ir_micro_gamma_u32 => {
            output: u32,
            input: &[],
            expected: 20u32,
            build: |rb| {
                let pred = rb.const_val(1);
                rb.gamma(pred, &[], 2, |branch_idx, br| {
                    let v = br.const_val(if branch_idx == 0 { 10 } else { 20 });
                    br.write_to_field(v, 0, Width::W4);
                    br.set_results(&[]);
                });
                rb.set_results(&[]);
            }
        },
        linear_ir_micro_intrinsic_u64 => {
            output: u64,
            input: &[],
            expected: 23u64,
            build: |rb| {
                let a = rb.const_val(11);
                let b = rb.const_val(7);
                let c = rb.const_val(5);
                let out = rb
                    .call_intrinsic(IntrinsicFn(add3_intrinsic as *const () as usize), &[a, b, c], 0, true)
                    .expect("return-value intrinsic should produce output");
                rb.write_to_field(out, 0, Width::W8);
                rb.set_results(&[]);
            }
        }
    }

    #[test]
    fn linear_backend_reads_u32_from_cursor() {
        let mut builder = IrBuilder::new(<u32 as facet::Facet>::SHAPE);
        {
            let mut rb = builder.root_region();
            rb.bounds_check(4);
            let v = rb.read_bytes(4);
            rb.write_to_field(v, 0, Width::W4);
            rb.set_results(&[]);
        }

        let mut func = builder.finish();
        let lin = linearize(&mut func);
        let (value, ctx) = run_u32_decoder(&lin, &[0x78, 0x56, 0x34, 0x12]);

        assert_eq!(ctx.error.code, 0);
        assert_eq!(value, 0x1234_5678);
    }

    #[test]
    fn linear_backend_call_intrinsic_zero_arg_return_value() {
        unsafe extern "C" fn return_300(_ctx: *mut crate::context::DeserContext) -> u64 {
            300
        }

        let mut builder = IrBuilder::new(<u32 as facet::Facet>::SHAPE);
        {
            let mut rb = builder.root_region();
            let v = rb
                .call_intrinsic(IntrinsicFn(return_300 as *const () as usize), &[], 0, true)
                .expect("intrinsic should produce output");
            rb.write_to_field(v, 0, Width::W4);
            rb.set_results(&[]);
        }

        let mut func = builder.finish();
        let lin = linearize(&mut func);
        let (value, ctx) = run_u32_decoder(&lin, &[]);

        assert_eq!(ctx.error.code, 0);
        assert_eq!(value, 300);
    }

    #[test]
    fn linear_backend_bounds_check_sets_eof() {
        let mut builder = IrBuilder::new(<u32 as facet::Facet>::SHAPE);
        {
            let mut rb = builder.root_region();
            rb.bounds_check(4);
            let v = rb.read_bytes(4);
            rb.write_to_field(v, 0, Width::W4);
            rb.set_results(&[]);
        }

        let mut func = builder.finish();
        let lin = linearize(&mut func);
        let (_value, ctx) = run_u32_decoder(&lin, &[0x01, 0x02]);

        assert_eq!(ctx.error.code, ErrorCode::UnexpectedEof as u32);
    }

    #[test]
    fn linear_backend_two_way_gamma_branch() {
        let mut builder = IrBuilder::new(<u32 as facet::Facet>::SHAPE);
        {
            let mut rb = builder.root_region();
            let pred = rb.const_val(1);
            rb.gamma(pred, &[], 2, |branch_idx, br| {
                let value = if branch_idx == 0 { 10 } else { 20 };
                let v = br.const_val(value);
                br.write_to_field(v, 0, Width::W4);
                br.set_results(&[]);
            });
            rb.set_results(&[]);
        }

        let mut func = builder.finish();
        let lin = linearize(&mut func);
        let (value, ctx) = run_u32_decoder(&lin, &[]);

        assert_eq!(ctx.error.code, 0);
        assert_eq!(value, 20);
    }

    #[test]
    fn linear_backend_jump_table_gamma_branch() {
        let mut builder = IrBuilder::new(<u32 as facet::Facet>::SHAPE);
        {
            let mut rb = builder.root_region();
            let pred = rb.const_val(2);
            rb.gamma(pred, &[], 3, |branch_idx, br| {
                let value = match branch_idx {
                    0 => 111,
                    1 => 222,
                    2 => 333,
                    _ => unreachable!(),
                };
                let v = br.const_val(value);
                br.write_to_field(v, 0, Width::W4);
                br.set_results(&[]);
            });
            rb.set_results(&[]);
        }

        let mut func = builder.finish();
        let lin = linearize(&mut func);
        let (value, ctx) = run_u32_decoder(&lin, &[]);

        assert_eq!(ctx.error.code, 0);
        assert_eq!(value, 333);
    }

    #[test]
    fn linear_backend_call_intrinsic_with_args_return_value() {
        unsafe extern "C" fn add3(
            _ctx: *mut crate::context::DeserContext,
            a: u64,
            b: u64,
            c: u64,
        ) -> u64 {
            a + b + c
        }

        let mut builder = IrBuilder::new(<u64 as facet::Facet>::SHAPE);
        {
            let mut rb = builder.root_region();
            let a = rb.const_val(11);
            let b = rb.const_val(7);
            let c = rb.const_val(5);
            let out = rb
                .call_intrinsic(IntrinsicFn(add3 as *const () as usize), &[a, b, c], 0, true)
                .expect("return-value intrinsic should produce output");
            rb.write_to_field(out, 0, Width::W8);
            rb.set_results(&[]);
        }

        let mut func = builder.finish();
        let lin = linearize(&mut func);
        let (value, ctx) = run_u64_decoder(&lin, &[]);

        assert_eq!(ctx.error.code, 0);
        assert_eq!(value, 23);
    }

    #[test]
    fn linear_backend_call_intrinsic_with_args_and_out_ptr() {
        unsafe extern "C" fn write_scaled_sum(
            _ctx: *mut crate::context::DeserContext,
            x: u64,
            y: u64,
            out: *mut u64,
        ) {
            unsafe { *out = x * 10 + y };
        }

        let mut builder = IrBuilder::new(<u64 as facet::Facet>::SHAPE);
        {
            let mut rb = builder.root_region();
            let x = rb.const_val(9);
            let y = rb.const_val(4);
            rb.call_intrinsic(
                IntrinsicFn(write_scaled_sum as *const () as usize),
                &[x, y],
                0,
                false,
            );
            rb.set_results(&[]);
        }

        let mut func = builder.finish();
        let lin = linearize(&mut func);
        let (value, ctx) = run_u64_decoder(&lin, &[]);

        assert_eq!(ctx.error.code, 0);
        assert_eq!(value, 94);
    }

    #[test]
    fn linear_backend_call_lambda_with_data_args_and_results() {
        let mut builder = IrBuilder::new(<u64 as facet::Facet>::SHAPE);
        let child = builder.create_lambda_with_data_args(<u64 as facet::Facet>::SHAPE, 1);
        {
            let mut rb = builder.lambda_region(child);
            let arg = rb.region_args(1)[0];
            let one = rb.const_val(1);
            let sum = rb.binop(IrOp::Add, arg, one);
            rb.set_results(&[sum]);
        }
        {
            let mut rb = builder.root_region();
            let x = rb.const_val(41);
            let out = rb.apply(child, &[x], 1);
            rb.write_to_field(out[0], 0, Width::W8);
            rb.set_results(&[]);
        }

        let mut func = builder.finish();
        let lin = linearize(&mut func);
        let (value, ctx) = run_u64_decoder(&lin, &[]);

        assert_eq!(ctx.error.code, 0);
        assert_eq!(value, 42);
    }

    #[test]
    fn linear_backend_vec_u32_matches_serde() {
        let expected = ScalarVec {
            values: (0..2048).map(|i| i as u32).collect(),
        };
        let bytes = postcard::to_allocvec(&expected).expect("serialize vec");

        let ir = crate::compile_decoder(ScalarVec::SHAPE, &crate::postcard::KajitPostcard);

        let ir_out = crate::deserialize::<ScalarVec>(&ir, &bytes).expect("ir decode");
        let serde_out = postcard::from_bytes::<ScalarVec>(&bytes).expect("serde decode");

        assert_eq!(ir_out, expected);
        assert_eq!(serde_out, expected);
    }
}
