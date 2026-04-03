#![allow(dead_code)]
//! HIR→IR lowering — converts hir::Module into RVSDG (ir::IrFunc).

use super::*;

#[derive(Clone, Copy)]
struct StructuralLocalStorage {
    base_slot: crate::ir::SlotId,
}

struct StructuralHirIrLowerer<'a> {
    module: &'a hir::Module,
    local_slots: std::collections::HashMap<hir::LocalId, StructuralLocalStorage>,
    local_types: std::collections::HashMap<hir::LocalId, &'a hir::Type>,
    _marker: std::marker::PhantomData<&'a hir::Module>,
}

enum ResolvedStructuralPlace<'a> {
    Destination {
        /// The HIR type at this destination position.
        ty: &'a hir::Type,
        /// Byte offset from the output base pointer.
        offset: usize,
    },
    Local {
        ty: &'a hir::Type,
        storage: StructuralLocalStorage,
        slot_offset: usize,
    },
}

enum ResolvedDynamicIndex<'a> {
    Destination {
        /// The HIR type at this destination position.
        ty: &'a hir::Type,
        addr: crate::ir::PortSource,
    },
    Local {
        ty: &'a hir::Type,
        addr: crate::ir::PortSource,
    },
}

impl<'a> StructuralHirIrLowerer<'a> {
    fn new(
        rb: &mut RegionBuilder<'_>,
        module: &'a hir::Module,
        function: &'a hir::Function,
    ) -> Self {
        let mut local_slots = std::collections::HashMap::new();
        let mut local_types = std::collections::HashMap::new();
        for param in &function.params {
            if !param.is_destination() {
                local_slots.insert(
                    param.local,
                    Self::alloc_local_storage(rb, module, &param.ty),
                );
            }
            local_types.insert(param.local, &param.ty);
        }
        for local in &function.locals {
            local_slots.insert(
                local.local,
                Self::alloc_local_storage(rb, module, &local.ty),
            );
            local_types.insert(local.local, &local.ty);
        }
        Self {
            module,
            local_slots,
            local_types,
            _marker: std::marker::PhantomData,
        }
    }

    fn alloc_local_storage(
        rb: &mut RegionBuilder<'_>,
        module: &'a hir::Module,
        ty: &hir::Type,
    ) -> StructuralLocalStorage {
        let slot_count = Self::slot_count_for_type(module, ty);
        let base_slot = rb.alloc_slot();
        if slot_count > 1 {
            rb.func().multi_slot_group.insert(base_slot);
            for _ in 1..slot_count {
                let sub_slot = rb.alloc_slot();
                rb.func().multi_slot_group.insert(sub_slot);
            }
        } else {
            // Single-slot scalar local — eligible for dead-port elimination.
            rb.func().scalar_temp_slots.insert(base_slot);
            for _ in 1..slot_count {
                let _ = rb.alloc_slot();
            }
        }
        StructuralLocalStorage { base_slot }
    }

    fn slot_count_for_type(module: &'a hir::Module, ty: &hir::Type) -> usize {
        match ty {
            hir::Type::Unit
            | hir::Type::Bool
            | hir::Type::Integer(_)
            | hir::Type::Address { .. } => 1,
            hir::Type::Array { element, len } => Self::slot_count_for_type(module, element)
                .saturating_mul(*len)
                .max(1),
            hir::Type::Str { .. } | hir::Type::Slice { .. } => 2,
            hir::Type::Handle { .. } => 1,
            hir::Type::Named { def, .. } => match &module.type_defs[*def].kind {
                hir::TypeDefKind::Struct { fields } => fields
                    .iter()
                    .map(|field| Self::slot_count_for_type(module, &field.ty))
                    .sum::<usize>()
                    .max(1),
                hir::TypeDefKind::Enum { variants, .. } => {
                    let payload_slots = variants
                        .iter()
                        .map(|variant| {
                            variant
                                .fields
                                .iter()
                                .map(|field| Self::slot_count_for_type(module, &field.ty))
                                .sum::<usize>()
                        })
                        .max()
                        .unwrap_or(0);
                    (1 + payload_slots).max(1)
                }
            },
        }
    }

    fn callable_spec(&self, call: &hir::CallExpr) -> &'a hir::CallableSpec {
        match call.target {
            hir::CallTarget::Callable(callable) => &self.module.callables[callable],
        }
    }

    fn callable_name(&self, call: &hir::CallExpr) -> &str {
        &self.callable_spec(call).name
    }

    fn callable_intrinsic(&self, call: &hir::CallExpr) -> Option<hir::RuntimeIntrinsic> {
        self.callable_spec(call).intrinsic
    }

    fn slot_at(storage: StructuralLocalStorage, slot_offset: usize) -> crate::ir::SlotId {
        crate::ir::SlotId::new(storage.base_slot.index() as u32 + slot_offset as u32)
    }

    fn lower_block(
        &self,
        rb: &mut RegionBuilder<'_>,
        statements: &[hir::Stmt],
        dest_local: hir::LocalId,
        dest_ty: &'a hir::Type,
    ) {
        for stmt in statements {
            self.lower_stmt(rb, stmt, dest_local, dest_ty);
        }
    }

    fn lower_stmt(
        &self,
        rb: &mut RegionBuilder<'_>,
        stmt: &hir::Stmt,
        dest_local: hir::LocalId,
        dest_ty: &'a hir::Type,
    ) {
        match &stmt.kind {
            hir::StmtKind::Init { place, value } | hir::StmtKind::Assign { place, value } => {
                self.lower_assign_like(rb, place, value, dest_local, dest_ty);
            }
            hir::StmtKind::Expr(expr) => self.lower_effect_expr(rb, expr, dest_local, dest_ty),
            hir::StmtKind::Fail { code } => rb.error_exit(*code),
            hir::StmtKind::Store { addr, width, value } => {
                let addr = self.lower_scalar_expr(rb, addr, dest_local, dest_ty);
                let value = self.lower_scalar_expr(rb, value, dest_local, dest_ty);
                rb.store_to_addr(addr, value, self.ir_width_for_memory_width(*width));
            }
            hir::StmtKind::If {
                condition,
                then_block,
                else_block,
            } => {
                let predicate = self.lower_scalar_expr(rb, condition, dest_local, dest_ty);
                let else_block = else_block
                    .as_ref()
                    .expect("structural HIR subset requires else");
                let _ = rb.gamma(predicate, &[], 2, |branch_idx, branch| {
                    match branch_idx {
                        0 => self.lower_block(branch, &else_block.statements, dest_local, dest_ty),
                        1 => self.lower_block(branch, &then_block.statements, dest_local, dest_ty),
                        _ => unreachable!(),
                    }
                    branch.set_results(&[]);
                });
            }
            hir::StmtKind::Match { scrutinee, arms } => {
                let predicate = self.lower_scalar_expr(rb, scrutinee, dest_local, dest_ty);
                for (expected, arm) in arms.iter().enumerate() {
                    let hir::Pattern::Integer(value) = arm.pattern else {
                        panic!("structural HIR subset only supports integer match patterns");
                    };
                    assert_eq!(
                        value, expected as u64,
                        "structural HIR subset requires contiguous integer match arms starting at 0"
                    );
                }
                let _ = rb.gamma(predicate, &[], arms.len(), |branch_idx, branch| {
                    self.lower_block(
                        branch,
                        &arms[branch_idx].body.statements,
                        dest_local,
                        dest_ty,
                    );
                    branch.set_results(&[]);
                });
            }
            hir::StmtKind::Loop {
                body,
                max_iterations,
                ..
            } => {
                let active_slot = rb.alloc_slot();
                let continue_slot = rb.alloc_slot();
                let build_body = |body_rb: &mut kajit_ir::RegionBuilder<'_>| {
                    let one = body_rb.const_val(1);
                    body_rb.write_to_slot(active_slot, one);
                    body_rb.write_to_slot(continue_slot, one);
                    self.lower_loop_block(
                        body_rb,
                        &body.statements,
                        dest_local,
                        dest_ty,
                        active_slot,
                        continue_slot,
                    );
                    let predicate = body_rb.read_from_slot(continue_slot);
                    body_rb.set_results(&[predicate]);
                };
                if let Some(max_iter) = max_iterations {
                    let _ = rb.theta_bounded(&[], *max_iter, build_body);
                } else {
                    let _ = rb.theta(&[], build_body);
                }
            }
            hir::StmtKind::Return(None) => {}
            other => panic!("unsupported structural HIR statement: {other:?}"),
        }
    }

    /// Lower a bounded loop by unrolling it into a gamma cascade.
    ///
    /// Instead of generating a theta, we generate N iterations as straight-line
    /// code with gamma branches (continue or exit) after each.
    fn lower_unrolled_loop(
        &self,
        rb: &mut RegionBuilder<'_>,
        statements: &[hir::Stmt],
        dest_local: hir::LocalId,
        dest_ty: &'a hir::Type,
        active_slot: kajit_ir::SlotId,
        continue_slot: kajit_ir::SlotId,
        max_iterations: usize,
    ) {
        self.lower_unrolled_iteration(
            rb,
            statements,
            dest_local,
            dest_ty,
            active_slot,
            continue_slot,
            max_iterations,
            0,
        );
    }

    /// Lower one iteration of an unrolled loop, then recurse for remaining.
    fn lower_unrolled_iteration(
        &self,
        rb: &mut RegionBuilder<'_>,
        statements: &[hir::Stmt],
        dest_local: hir::LocalId,
        dest_ty: &'a hir::Type,
        active_slot: kajit_ir::SlotId,
        continue_slot: kajit_ir::SlotId,
        remaining: usize,
        iteration: usize,
    ) {
        if remaining == 0 {
            return;
        }

        // Initialize active/continue slots for each iteration.
        // In the original theta, these are re-initialized at body entry.
        // In the unrolled form, each iteration starts fresh.
        let one = rb.const_val(1);
        rb.write_to_slot(active_slot, one);
        rb.write_to_slot(continue_slot, one);

        // Execute one iteration of the loop body
        self.lower_loop_block(
            rb,
            statements,
            dest_local,
            dest_ty,
            active_slot,
            continue_slot,
        );

        if remaining == 1 {
            // Last iteration — no need for a gamma, just stop
            return;
        }

        // Read predicate: should we continue?
        let predicate = rb.read_from_slot(continue_slot);

        // Gamma: predicate != 0 → branch 1 (continue), predicate == 0 → branch 0 (exit)
        let _ = rb.gamma(predicate, &[], 2, |branch_idx, branch_rb| {
            if branch_idx == 0 {
                // Exit branch: do nothing (just pass through state)
                branch_rb.set_results(&[]);
            } else {
                // Continue branch: recurse with remaining - 1
                self.lower_unrolled_iteration(
                    branch_rb,
                    statements,
                    dest_local,
                    dest_ty,
                    active_slot,
                    continue_slot,
                    remaining - 1,
                    iteration + 1,
                );
                branch_rb.set_results(&[]);
            }
        });
    }

    fn lower_loop_block(
        &self,
        rb: &mut RegionBuilder<'_>,
        statements: &[hir::Stmt],
        dest_local: hir::LocalId,
        dest_ty: &'a hir::Type,
        active_slot: crate::ir::SlotId,
        continue_slot: crate::ir::SlotId,
    ) {
        // Batch consecutive non-control-flow statements into single guards.
        // Control-flow statements (Break, Continue, If, Match, Loop) need
        // individual guards because they can change `active_slot`.
        let mut i = 0;
        while i < statements.len() {
            // Find run of non-control-flow statements
            let start = i;
            while i < statements.len() && !Self::is_loop_control_flow(&statements[i]) {
                i += 1;
            }

            // Lower the batch (if any) in one guard
            if start < i {
                self.with_active_guard(rb, active_slot, |guard_rb| {
                    for stmt in &statements[start..i] {
                        self.lower_stmt(guard_rb, stmt, dest_local, dest_ty);
                    }
                });
            }

            // Lower the control-flow statement (if any) in its own guard
            if i < statements.len() {
                self.lower_loop_control_flow_stmt(
                    rb,
                    &statements[i],
                    dest_local,
                    dest_ty,
                    active_slot,
                    continue_slot,
                );
                i += 1;
            }
        }
    }

    /// Returns true if this statement is control flow that can change `active_slot`.
    fn is_loop_control_flow(stmt: &hir::Stmt) -> bool {
        matches!(
            stmt.kind,
            hir::StmtKind::Break
                | hir::StmtKind::Continue
                | hir::StmtKind::If { .. }
                | hir::StmtKind::Match { .. }
                | hir::StmtKind::Loop { .. }
        )
    }

    /// Lower a control-flow statement inside a loop body.
    /// These need individual guards because they can change `active_slot`.
    fn lower_loop_control_flow_stmt(
        &self,
        rb: &mut RegionBuilder<'_>,
        stmt: &hir::Stmt,
        dest_local: hir::LocalId,
        dest_ty: &'a hir::Type,
        active_slot: crate::ir::SlotId,
        continue_slot: crate::ir::SlotId,
    ) {
        self.with_active_guard(rb, active_slot, |guard_rb| match &stmt.kind {
            hir::StmtKind::Break => {
                let zero = guard_rb.const_val(0);
                guard_rb.write_to_slot(active_slot, zero);
                guard_rb.write_to_slot(continue_slot, zero);
            }
            hir::StmtKind::Continue => {
                let zero = guard_rb.const_val(0);
                let one = guard_rb.const_val(1);
                guard_rb.write_to_slot(active_slot, zero);
                guard_rb.write_to_slot(continue_slot, one);
            }
            hir::StmtKind::If {
                condition,
                then_block,
                else_block,
            } => {
                let predicate = self.lower_scalar_expr(guard_rb, condition, dest_local, dest_ty);
                let else_block = else_block
                    .as_ref()
                    .expect("structural HIR loop subset requires else");
                let _ = guard_rb.gamma(predicate, &[], 2, |branch_idx, branch| {
                    match branch_idx {
                        0 => self.lower_loop_block(
                            branch,
                            &else_block.statements,
                            dest_local,
                            dest_ty,
                            active_slot,
                            continue_slot,
                        ),
                        1 => self.lower_loop_block(
                            branch,
                            &then_block.statements,
                            dest_local,
                            dest_ty,
                            active_slot,
                            continue_slot,
                        ),
                        _ => unreachable!(),
                    }
                    branch.set_results(&[]);
                });
            }
            hir::StmtKind::Match { scrutinee, arms } => {
                let predicate = self.lower_scalar_expr(guard_rb, scrutinee, dest_local, dest_ty);
                for (expected, arm) in arms.iter().enumerate() {
                    let hir::Pattern::Integer(value) = arm.pattern else {
                        panic!("structural HIR loop subset only supports integer match patterns");
                    };
                    assert_eq!(
                        value, expected as u64,
                        "structural HIR loop subset requires contiguous integer match arms starting at 0"
                    );
                }
                let _ = guard_rb.gamma(predicate, &[], arms.len(), |branch_idx, branch| {
                    self.lower_loop_block(
                        branch,
                        &arms[branch_idx].body.statements,
                        dest_local,
                        dest_ty,
                        active_slot,
                        continue_slot,
                    );
                    branch.set_results(&[]);
                });
            }
            hir::StmtKind::Loop { body, max_iterations, .. } => {
                let nested_active_slot = guard_rb.alloc_slot();
                let nested_continue_slot = guard_rb.alloc_slot();
                let build_body = |body_rb: &mut kajit_ir::RegionBuilder<'_>| {
                    let one = body_rb.const_val(1);
                    body_rb.write_to_slot(nested_active_slot, one);
                    body_rb.write_to_slot(nested_continue_slot, one);
                    self.lower_loop_block(
                        body_rb,
                        &body.statements,
                        dest_local,
                        dest_ty,
                        nested_active_slot,
                        nested_continue_slot,
                    );
                    let predicate = body_rb.read_from_slot(nested_continue_slot);
                    body_rb.set_results(&[predicate]);
                };
                if let Some(max_iter) = max_iterations {
                    let _ = guard_rb.theta_bounded(&[], *max_iter, build_body);
                } else {
                    let _ = guard_rb.theta(&[], build_body);
                }
            }
            other => panic!("is_loop_control_flow returned true for non-control-flow: {other:?}"),
        });
    }

    fn with_active_guard(
        &self,
        rb: &mut RegionBuilder<'_>,
        active_slot: crate::ir::SlotId,
        f: impl FnOnce(&mut RegionBuilder<'_>),
    ) {
        let active = rb.read_from_slot(active_slot);
        let mut f = Some(f);
        let _ = rb.gamma(active, &[], 2, |branch_idx, branch| {
            if branch_idx == 1 {
                f.take().expect("active branch should lower exactly once")(branch);
            }
            branch.set_results(&[]);
        });
    }

    fn lower_effect_expr(
        &self,
        rb: &mut RegionBuilder<'_>,
        expr: &hir::Expr,
        dest_local: hir::LocalId,
        dest_ty: &'a hir::Type,
    ) {
        match expr {
            hir::Expr::Call(call) => self.lower_effect_call(rb, call, dest_local, dest_ty),
            other => panic!("unsupported structural HIR effect expression: {other:?}"),
        }
    }

    fn lower_assign_like(
        &self,
        rb: &mut RegionBuilder<'_>,
        place: &hir::Place,
        value: &hir::Expr,
        dest_local: hir::LocalId,
        dest_ty: &'a hir::Type,
    ) {
        self.lower_place_write(rb, place, value, dest_local, dest_ty);
    }

    fn lower_place_write(
        &self,
        rb: &mut RegionBuilder<'_>,
        place: &hir::Place,
        value: &hir::Expr,
        dest_local: hir::LocalId,
        dest_ty: &'a hir::Type,
    ) {
        if let hir::Place::Index { base, index } = place
            && !matches!(**index, hir::Expr::Literal(hir::Literal::Integer(_)))
        {
            if let hir::Expr::Local(local) = value
                && Self::slot_count_for_type(self.module, self.local_types[local]) > 1
            {
                self.lower_dynamic_index_write_from_local(
                    rb, base, index, *local, dest_local, dest_ty,
                );
            } else if let hir::Expr::Str { data, len } = value {
                // Str is two 8-byte values: (data_ptr, len) — write both at the dynamic address
                let resolved = self.lower_dynamic_index_addr(rb, base, index, dest_local, dest_ty);
                let base_addr = match resolved {
                    ResolvedDynamicIndex::Destination { addr, .. }
                    | ResolvedDynamicIndex::Local { addr, .. } => addr,
                };
                let data = self.lower_scalar_expr(rb, data, dest_local, dest_ty);
                let len = self.lower_scalar_expr(rb, len, dest_local, dest_ty);
                rb.store_to_addr(base_addr, data, crate::ir::Width::W8);
                let len_addr =
                    self.add_byte_offset(rb, base_addr, crate::ir::SLOT_ADDR_STRIDE_BYTES);
                rb.store_to_addr(len_addr, len, crate::ir::Width::W8);
            } else {
                self.lower_dynamic_index_write(rb, base, index, value, dest_local, dest_ty);
            }
            return;
        }
        let resolved = self.resolve_place(place, dest_local, dest_ty);
        match value {
            hir::Expr::Local(local) => match resolved {
                ResolvedStructuralPlace::Destination { ty, offset } => {
                    self.copy_local_into_dest_offset(rb, *local, ty, offset);
                }
                ResolvedStructuralPlace::Local {
                    ty,
                    storage,
                    slot_offset,
                } => {
                    self.copy_local_into_local(rb, *local, ty, storage, slot_offset);
                }
            },
            hir::Expr::Call(_) => {
                let scalar = self.lower_scalar_expr(rb, value, dest_local, dest_ty);
                match resolved {
                    ResolvedStructuralPlace::Destination { ty, offset } => {
                        let width = self.scalar_width_for_hir_type(ty);
                        rb.write_to_field(scalar, offset as u32, width);
                    }
                    ResolvedStructuralPlace::Local {
                        ty,
                        storage,
                        slot_offset,
                    } => {
                        assert_eq!(
                            Self::slot_count_for_type(self.module, ty),
                            1,
                            "structural local scalar write requires single-slot type"
                        );
                        rb.write_to_slot(Self::slot_at(storage, slot_offset), scalar);
                    }
                }
            }
            hir::Expr::Str { data, len } => match resolved {
                ResolvedStructuralPlace::Destination { ty, offset } => {
                    assert!(
                        matches!(ty, hir::Type::Str { .. }),
                        "str materialization requires a str destination, got {ty:?}"
                    );
                    let data = self.lower_scalar_expr(rb, data, dest_local, dest_ty);
                    let len = self.lower_scalar_expr(rb, len, dest_local, dest_ty);
                    rb.write_to_field(data, offset as u32, crate::ir::Width::W8);
                    rb.write_to_field(len, (offset + 8) as u32, crate::ir::Width::W8);
                }
                ResolvedStructuralPlace::Local {
                    ty,
                    storage,
                    slot_offset,
                } => {
                    assert!(
                        matches!(ty, hir::Type::Str { .. }),
                        "str materialization requires a local str type, got {ty:?}"
                    );
                    let data = self.lower_scalar_expr(rb, data, dest_local, dest_ty);
                    let len = self.lower_scalar_expr(rb, len, dest_local, dest_ty);
                    rb.write_to_slot(Self::slot_at(storage, slot_offset), data);
                    rb.write_to_slot(Self::slot_at(storage, slot_offset + 1), len);
                }
            },
            hir::Expr::Variant {
                variant, fields, ..
            } => match resolved {
                ResolvedStructuralPlace::Destination { ty, offset } => {
                    let hir::Type::Named { def, .. } = ty else {
                        panic!("variant init must target a named enum type, got {ty:?}");
                    };
                    let type_def = &self.module.type_defs[*def];
                    let hir::TypeDefKind::Enum {
                        variants: hir_variants,
                        discriminant_width,
                    } = &type_def.kind
                    else {
                        panic!(
                            "variant init must target an enum type def, got {}",
                            type_def.name
                        );
                    };
                    let hir_variant = hir_variants
                        .iter()
                        .find(|v| v.name == variant.as_str())
                        .unwrap_or_else(|| {
                            panic!("missing enum variant {variant} in {}", type_def.name)
                        });
                    let disc_width_val = discriminant_width.unwrap_or(1);
                    let disc_ir_width = ir_width_from_disc_size(disc_width_val);
                    // Get discriminant: use annotation, or fall back to variant index
                    let disc: u64 = hir_variant
                        .discriminant
                        .map(|d| d.try_into().expect("enum discriminant must fit in u64"))
                        .unwrap_or_else(|| {
                            hir_variants
                                .iter()
                                .position(|v| v.name == variant.as_str())
                                .expect("variant must exist") as u64
                        });
                    let value = rb.const_val(disc);
                    rb.write_to_field(value, offset as u32, disc_ir_width);
                    for hir_field in &hir_variant.fields {
                        let (_, expr) = fields
                            .iter()
                            .find(|(name, _)| name == &hir_field.name)
                            .unwrap_or_else(|| {
                                panic!(
                                    "missing enum payload field {} for variant {variant}",
                                    hir_field.name
                                )
                            });
                        let field_offset =
                            hir_field.offset.map(|o| o as usize).unwrap_or_else(|| {
                                // Compute from discriminant width + preceding field sizes
                                let disc_bytes = disc_width_val as usize;
                                let mut running = disc_bytes;
                                for f in &hir_variant.fields {
                                    if f.name == hir_field.name {
                                        return running;
                                    }
                                    running += self.hir_type_size(&f.ty);
                                }
                                disc_bytes
                            });
                        self.lower_value_into_dest_offset(
                            rb,
                            &hir_field.ty,
                            offset + field_offset,
                            expr,
                            dest_local,
                            dest_ty,
                        );
                    }
                }
                ResolvedStructuralPlace::Local { .. } => {
                    panic!("local enum writes are not supported yet");
                }
            },
            hir::Expr::Index { .. } => match resolved {
                ResolvedStructuralPlace::Destination { ty, offset } => {
                    self.lower_value_into_dest_offset(rb, ty, offset, value, dest_local, dest_ty);
                }
                ResolvedStructuralPlace::Local {
                    ty,
                    storage,
                    slot_offset,
                } => {
                    let scalar = self.lower_scalar_expr(rb, value, dest_local, dest_ty);
                    assert_eq!(
                        Self::slot_count_for_type(self.module, ty),
                        1,
                        "structural local indexed write requires single-slot type"
                    );
                    rb.write_to_slot(Self::slot_at(storage, slot_offset), scalar);
                }
            },
            hir::Expr::Literal(hir::Literal::Unit) => {}
            _ => {
                let scalar = self.lower_scalar_expr(rb, value, dest_local, dest_ty);
                match resolved {
                    ResolvedStructuralPlace::Destination { ty, offset } => {
                        let width = self.scalar_width_for_hir_type(ty);
                        rb.write_to_field(scalar, offset as u32, width);
                    }
                    ResolvedStructuralPlace::Local {
                        ty,
                        storage,
                        slot_offset,
                    } => {
                        assert_eq!(
                            Self::slot_count_for_type(self.module, ty),
                            1,
                            "structural local scalar write requires single-slot type"
                        );
                        rb.write_to_slot(Self::slot_at(storage, slot_offset), scalar);
                    }
                }
            }
        }
    }

    fn copy_local_into_local(
        &self,
        rb: &mut RegionBuilder<'_>,
        source_local: hir::LocalId,
        dest_ty: &hir::Type,
        dest_storage: StructuralLocalStorage,
        dest_slot_offset: usize,
    ) {
        let source_storage = self.local_slots[&source_local];
        let source_ty = self.local_types[&source_local];
        let source_slots = Self::slot_count_for_type(self.module, source_ty);
        let dest_slots = Self::slot_count_for_type(self.module, dest_ty);
        assert_eq!(
            source_slots, dest_slots,
            "structural local copy requires matching slot counts"
        );
        for slot_index in 0..source_slots {
            let value = rb.read_from_slot(Self::slot_at(source_storage, slot_index));
            rb.write_to_slot(
                Self::slot_at(dest_storage, dest_slot_offset + slot_index),
                value,
            );
        }
    }

    fn lower_value_into_dest_offset(
        &self,
        rb: &mut RegionBuilder<'_>,
        ty: &'a hir::Type,
        offset: usize,
        expr: &hir::Expr,
        dest_local: hir::LocalId,
        dest_ty: &'a hir::Type,
    ) {
        match expr {
            hir::Expr::Call(_) => {
                let scalar = self.lower_scalar_expr(rb, expr, dest_local, dest_ty);
                let width = self.scalar_width_for_hir_type(ty);
                rb.write_to_field(scalar, offset as u32, width);
            }
            hir::Expr::Local(local) => self.copy_local_into_dest_offset(rb, *local, ty, offset),
            hir::Expr::Index { base, index } => {
                let base = self.expr_to_place(base);
                if matches!(&**index, hir::Expr::Literal(hir::Literal::Integer(_))) {
                    let place = hir::Place::Index {
                        base: Box::new(base),
                        index: index.clone(),
                    };
                    match self.resolve_place(&place, dest_local, dest_ty) {
                        ResolvedStructuralPlace::Destination {
                            ty: source_ty,
                            offset: source_offset,
                        } => {
                            self.copy_dest_bytes_to_dest_offset(
                                rb,
                                source_ty,
                                source_offset,
                                offset,
                            );
                        }
                        ResolvedStructuralPlace::Local {
                            ty: source_ty,
                            storage,
                            slot_offset,
                        } => {
                            let slot_count = Self::slot_count_for_type(self.module, source_ty);
                            for slot_index in 0..slot_count {
                                let slot = Self::slot_at(storage, slot_offset + slot_index);
                                let value = rb.read_from_slot(slot);
                                rb.write_to_field(
                                    value,
                                    (offset + slot_index * 8) as u32,
                                    crate::ir::Width::W8,
                                );
                            }
                        }
                    }
                } else {
                    self.copy_dynamic_index_into_shape_offset(
                        rb, &base, index, offset, dest_local, dest_ty,
                    );
                }
            }
            hir::Expr::Literal(hir::Literal::Unit) => {}
            hir::Expr::Literal(hir::Literal::Bool(value)) => {
                let value = rb.const_val(u64::from(*value));
                rb.write_to_field(value, offset as u32, crate::ir::Width::W1);
            }
            hir::Expr::Literal(hir::Literal::Integer(value)) => {
                let value = rb.const_val(*value);
                let width = self.scalar_width_for_hir_type(ty);
                rb.write_to_field(value, offset as u32, width);
            }
            hir::Expr::Str { data, len } => {
                assert!(
                    matches!(ty, hir::Type::Str { .. }),
                    "str materialization requires a str destination, got {ty:?}"
                );
                let data = self.lower_scalar_expr(rb, data, dest_local, dest_ty);
                let len = self.lower_scalar_expr(rb, len, dest_local, dest_ty);
                rb.write_to_field(data, offset as u32, crate::ir::Width::W8);
                rb.write_to_field(len, (offset + 8) as u32, crate::ir::Width::W8);
            }
            hir::Expr::Variant {
                variant, fields, ..
            } => {
                self.lower_nested_variant_write(
                    rb, offset, ty, variant, fields, dest_local, dest_ty,
                );
            }
            other => panic!("unsupported structural payload expression: {other:?}"),
        }
    }

    fn copy_dest_bytes_to_dest_offset(
        &self,
        rb: &mut RegionBuilder<'_>,
        source_ty: &hir::Type,
        source_offset: usize,
        target_offset: usize,
    ) {
        let size = self.hir_type_size(source_ty);
        let full_words = size / 8;
        let remainder = size % 8;
        for word_index in 0..full_words {
            let value = rb.read_from_field(
                (source_offset + word_index * 8) as u32,
                crate::ir::Width::W8,
            );
            rb.write_to_field(
                value,
                (target_offset + word_index * 8) as u32,
                crate::ir::Width::W8,
            );
        }
        if remainder != 0 {
            let width = match remainder {
                1 => crate::ir::Width::W1,
                2 => crate::ir::Width::W2,
                4 => crate::ir::Width::W4,
                _ => panic!("unsupported indexed destination remainder width {remainder}"),
            };
            let value = rb.read_from_field((source_offset + full_words * 8) as u32, width);
            rb.write_to_field(value, (target_offset + full_words * 8) as u32, width);
        }
    }

    fn copy_local_into_dest_offset(
        &self,
        rb: &mut RegionBuilder<'_>,
        local: hir::LocalId,
        ty: &hir::Type,
        offset: usize,
    ) {
        let storage = self.local_slots[&local];
        // For simple scalar types (non-string-like), just copy one slot
        if self.is_simple_scalar_hir_type(ty) {
            let value = rb.read_from_slot(storage.base_slot);
            let width = self.scalar_width_for_hir_type(ty);
            rb.write_to_field(value, offset as u32, width);
            return;
        }

        let size = self.hir_type_size(ty);
        let full_slots = size / 8;
        let remainder = size % 8;

        for slot_index in 0..full_slots {
            let slot = crate::ir::SlotId::new(storage.base_slot.index() as u32 + slot_index as u32);
            let value = rb.read_from_slot(slot);
            rb.write_to_field(
                value,
                (offset + slot_index * 8) as u32,
                crate::ir::Width::W8,
            );
        }

        if remainder != 0 {
            let slot = crate::ir::SlotId::new(storage.base_slot.index() as u32 + full_slots as u32);
            let value = rb.read_from_slot(slot);
            let width = match remainder {
                1 => crate::ir::Width::W1,
                2 => crate::ir::Width::W2,
                4 => crate::ir::Width::W4,
                _ => panic!("unsupported remainder width {remainder}"),
            };
            rb.write_to_field(value, (offset + full_slots * 8) as u32, width);
        }
    }

    fn lower_scalar_expr(
        &self,
        rb: &mut RegionBuilder<'_>,
        expr: &hir::Expr,
        dest_local: hir::LocalId,
        dest_ty: &'a hir::Type,
    ) -> crate::ir::PortSource {
        match expr {
            hir::Expr::Literal(hir::Literal::Bool(value)) => rb.const_val(u64::from(*value)),
            hir::Expr::Literal(hir::Literal::Integer(value)) => rb.const_val(*value),
            hir::Expr::Local(local) => {
                let slot = self.local_slots[local].base_slot;
                rb.read_from_slot(slot)
            }
            hir::Expr::AddrOf(place) => self.lower_place_addr(rb, place, dest_local, dest_ty),
            hir::Expr::Load { addr, width } => {
                let addr = self.lower_scalar_expr(rb, addr, dest_local, dest_ty);
                rb.load_from_addr(addr, self.ir_width_for_memory_width(*width))
            }
            hir::Expr::SliceData { value } => {
                self.lower_view_component(rb, value, 0, dest_local, dest_ty)
            }
            hir::Expr::SliceLen { value } => {
                self.lower_view_component(rb, value, 1, dest_local, dest_ty)
            }
            hir::Expr::Field { .. } | hir::Expr::Index { .. } => {
                let place = self.expr_to_place(expr);
                if let hir::Place::Index { base, index } = &place
                    && !matches!(**index, hir::Expr::Literal(hir::Literal::Integer(_)))
                {
                    return self.lower_dynamic_index_read(rb, base, index, dest_local, dest_ty);
                }
                match self.resolve_place(&place, dest_local, dest_ty) {
                    ResolvedStructuralPlace::Destination { ty, offset } => {
                        let width = self.scalar_width_for_hir_type(ty);
                        rb.read_from_field(offset as u32, width)
                    }
                    ResolvedStructuralPlace::Local {
                        ty,
                        storage,
                        slot_offset,
                    } => {
                        assert_eq!(
                            Self::slot_count_for_type(self.module, ty),
                            1,
                            "structural local scalar read requires single-slot type"
                        );
                        rb.read_from_slot(Self::slot_at(storage, slot_offset))
                    }
                }
            }
            hir::Expr::Binary { op, lhs, rhs } => {
                let lhs = self.lower_scalar_expr(rb, lhs, dest_local, dest_ty);
                let rhs = self.lower_scalar_expr(rb, rhs, dest_local, dest_ty);
                let ir_op = match op {
                    hir::BinaryOp::Add => crate::ir::IrOp::Add,
                    hir::BinaryOp::Sub => crate::ir::IrOp::Sub,
                    hir::BinaryOp::Mul => crate::ir::IrOp::Mul,
                    hir::BinaryOp::BitAnd => crate::ir::IrOp::And,
                    hir::BinaryOp::BitOr => crate::ir::IrOp::Or,
                    hir::BinaryOp::Xor => crate::ir::IrOp::Xor,
                    hir::BinaryOp::Shl => crate::ir::IrOp::Shl,
                    hir::BinaryOp::Shr => crate::ir::IrOp::Shr,
                    hir::BinaryOp::Sar => crate::ir::IrOp::Sar,
                    hir::BinaryOp::Eq => crate::ir::IrOp::CmpEq,
                    hir::BinaryOp::And => crate::ir::IrOp::And,
                    hir::BinaryOp::Or => crate::ir::IrOp::Or,
                    hir::BinaryOp::Ne => crate::ir::IrOp::CmpNe,
                    hir::BinaryOp::Lt => crate::ir::IrOp::CmpLt,
                    hir::BinaryOp::Le => crate::ir::IrOp::CmpLe,
                    hir::BinaryOp::Gt => crate::ir::IrOp::CmpGt,
                    hir::BinaryOp::Ge => crate::ir::IrOp::CmpGe,
                    other => panic!("unsupported structural HIR binary op: {other:?}"),
                };
                rb.binop(ir_op, lhs, rhs)
            }
            hir::Expr::Call(call) => self.lower_scalar_call_expr(rb, call, dest_local, dest_ty),
            other => panic!("unsupported structural HIR scalar expression: {other:?}"),
        }
    }

    fn lower_scalar_call_expr(
        &self,
        rb: &mut RegionBuilder<'_>,
        call: &hir::CallExpr,
        dest_local: hir::LocalId,
        dest_ty: &'a hir::Type,
    ) -> crate::ir::PortSource {
        let args = call
            .args
            .iter()
            .map(|arg| self.lower_scalar_expr(rb, arg, dest_local, dest_ty))
            .collect::<Vec<_>>();
        let func = match self.callable_intrinsic(call) {
            Some(hir::RuntimeIntrinsic::SaveCursor) => return rb.save_cursor(),
            Some(hir::RuntimeIntrinsic::SaveInputEnd) => return rb.save_input_end(),
            Some(hir::RuntimeIntrinsic::AllocPersistent) => {
                crate::ir::IntrinsicFn(intrinsics::kajit_alloc_persistent as *const () as usize)
            }
            Some(hir::RuntimeIntrinsic::StringValidateAllocCopy) => crate::ir::IntrinsicFn(
                intrinsics::kajit_string_validate_alloc_copy as *const () as usize,
            ),
            other => panic!(
                "unsupported structural HIR scalar call {} ({other:?})",
                self.callable_name(call)
            ),
        };
        rb.call_intrinsic(func, &args, 0, true)
            .expect("scalar intrinsic call should return a value")
    }

    fn lower_effect_call(
        &self,
        rb: &mut RegionBuilder<'_>,
        call: &hir::CallExpr,
        dest_local: hir::LocalId,
        dest_ty: &'a hir::Type,
    ) {
        match self.callable_intrinsic(call) {
            Some(hir::RuntimeIntrinsic::OptionInitNone) => {
                self.lower_option_init_none_call(rb, call, dest_local, dest_ty);
                return;
            }
            Some(hir::RuntimeIntrinsic::OptionInitSome) => {
                self.lower_option_init_some_call(rb, call, dest_local, dest_ty);
                return;
            }
            Some(hir::RuntimeIntrinsic::ValidateUtf8Range) => {
                let args = call
                    .args
                    .iter()
                    .map(|arg| self.lower_scalar_expr(rb, arg, dest_local, dest_ty))
                    .collect::<Vec<_>>();
                rb.call_intrinsic(
                    crate::ir::IntrinsicFn(
                        intrinsics::kajit_validate_utf8_range as *const () as usize,
                    ),
                    &args,
                    0,
                    false,
                );
                return;
            }
            Some(hir::RuntimeIntrinsic::CursorRestore) => {
                let args = call
                    .args
                    .iter()
                    .map(|arg| self.lower_scalar_expr(rb, arg, dest_local, dest_ty))
                    .collect::<Vec<_>>();
                assert_eq!(
                    args.len(),
                    1,
                    "runtime.cursor_restore expects one absolute cursor address"
                );
                rb.restore_cursor(args[0]);
                return;
            }
            other => panic!(
                "unsupported structural HIR effect call {} ({other:?})",
                self.callable_name(call)
            ),
        }
    }

    fn lower_option_init_none_call(
        &self,
        rb: &mut RegionBuilder<'_>,
        call: &hir::CallExpr,
        dest_local: hir::LocalId,
        dest_ty: &'a hir::Type,
    ) {
        assert_eq!(
            call.args.len(),
            2,
            "runtime.option_init_none expects init_fn and out addr"
        );
        let init_fn = self.lower_scalar_expr(rb, &call.args[0], dest_local, dest_ty);
        if let hir::Expr::AddrOf(place) = &call.args[1] {
            match self.resolve_place(place, dest_local, dest_ty) {
                ResolvedStructuralPlace::Destination { offset, .. } => {
                    rb.call_intrinsic(
                        crate::ir::IntrinsicFn(
                            intrinsics::kajit_option_init_none_ctx as *const () as usize,
                        ),
                        &[init_fn],
                        offset as u32,
                        false,
                    );
                    return;
                }
                ResolvedStructuralPlace::Local { .. } => {}
            }
        }
        let out_addr = self.lower_scalar_expr(rb, &call.args[1], dest_local, dest_ty);
        let _ = rb.call_effect(
            crate::ir::IntrinsicFn(intrinsics::kajit_option_init_none as *const () as usize),
            &[init_fn, out_addr],
        );
    }

    fn lower_option_init_some_call(
        &self,
        rb: &mut RegionBuilder<'_>,
        call: &hir::CallExpr,
        dest_local: hir::LocalId,
        dest_ty: &'a hir::Type,
    ) {
        assert_eq!(
            call.args.len(),
            3,
            "runtime.option_init_some expects init_fn, out addr, and payload addr"
        );
        let init_fn = self.lower_scalar_expr(rb, &call.args[0], dest_local, dest_ty);
        let payload_addr = self.lower_scalar_expr(rb, &call.args[2], dest_local, dest_ty);
        if let hir::Expr::AddrOf(place) = &call.args[1] {
            match self.resolve_place(place, dest_local, dest_ty) {
                ResolvedStructuralPlace::Destination { offset, .. } => {
                    rb.call_intrinsic(
                        crate::ir::IntrinsicFn(
                            intrinsics::kajit_option_init_some_ctx as *const () as usize,
                        ),
                        &[init_fn, payload_addr],
                        offset as u32,
                        false,
                    );
                    return;
                }
                ResolvedStructuralPlace::Local { .. } => {}
            }
        }
        let out_addr = self.lower_scalar_expr(rb, &call.args[1], dest_local, dest_ty);
        let _ = rb.call_effect(
            crate::ir::IntrinsicFn(intrinsics::kajit_option_init_some as *const () as usize),
            &[init_fn, out_addr, payload_addr],
        );
    }

    fn expr_to_place(&self, expr: &hir::Expr) -> hir::Place {
        match expr {
            hir::Expr::Local(local) => hir::Place::Local(*local),
            hir::Expr::Field { base, field } => hir::Place::Field {
                base: Box::new(self.expr_to_place(base)),
                field: field.clone(),
            },
            hir::Expr::Index { base, index } => hir::Place::Index {
                base: Box::new(self.expr_to_place(base)),
                index: index.clone(),
            },
            other => panic!("unsupported structural HIR place expression: {other:?}"),
        }
    }

    fn lower_view_component(
        &self,
        rb: &mut RegionBuilder<'_>,
        value: &hir::Expr,
        word_index: usize,
        dest_local: hir::LocalId,
        dest_ty: &'a hir::Type,
    ) -> crate::ir::PortSource {
        let place = self.expr_to_place(value);
        match self.resolve_place(&place, dest_local, dest_ty) {
            ResolvedStructuralPlace::Local {
                ty,
                storage,
                slot_offset,
            } => {
                assert!(
                    matches!(ty, hir::Type::Slice { .. } | hir::Type::Str { .. }),
                    "slice_data/slice_len require a local Slice/str, got {ty:?}"
                );
                rb.read_from_slot(Self::slot_at(storage, slot_offset + word_index))
            }
            ResolvedStructuralPlace::Destination { ty, offset } => {
                assert!(
                    matches!(ty, hir::Type::Str { .. } | hir::Type::Slice { .. }),
                    "slice_data/slice_len require a slice-like destination, got {ty:?}"
                );
                rb.read_from_field((offset + word_index * 8) as u32, crate::ir::Width::W8)
            }
        }
    }

    fn add_scaled_index(
        &self,
        rb: &mut RegionBuilder<'_>,
        base_addr: crate::ir::PortSource,
        index: crate::ir::PortSource,
        stride_bytes: usize,
    ) -> crate::ir::PortSource {
        if stride_bytes == 1 {
            rb.binop(crate::ir::IrOp::Add, base_addr, index)
        } else {
            let stride = rb.const_val(stride_bytes as u64);
            let scaled = rb.binop(crate::ir::IrOp::Mul, index, stride);
            rb.binop(crate::ir::IrOp::Add, base_addr, scaled)
        }
    }

    fn add_byte_offset(
        &self,
        rb: &mut RegionBuilder<'_>,
        base_addr: crate::ir::PortSource,
        offset: usize,
    ) -> crate::ir::PortSource {
        if offset == 0 {
            base_addr
        } else {
            let offset = rb.const_val(offset as u64);
            rb.binop(crate::ir::IrOp::Add, base_addr, offset)
        }
    }

    fn lower_dynamic_index_read(
        &self,
        rb: &mut RegionBuilder<'_>,
        base: &hir::Place,
        index: &hir::Expr,
        dest_local: hir::LocalId,
        dest_ty: &'a hir::Type,
    ) -> crate::ir::PortSource {
        let resolved = self.lower_dynamic_index_addr(rb, base, index, dest_local, dest_ty);
        match resolved {
            ResolvedDynamicIndex::Destination { ty, addr } => {
                let width = self.scalar_width_for_hir_type(ty);
                rb.load_from_addr(addr, width)
            }
            ResolvedDynamicIndex::Local { ty, addr } => {
                let width = self.scalar_width_for_hir_type(ty);
                rb.load_from_addr(addr, width)
            }
        }
    }

    fn lower_place_addr(
        &self,
        rb: &mut RegionBuilder<'_>,
        place: &hir::Place,
        dest_local: hir::LocalId,
        dest_ty: &'a hir::Type,
    ) -> crate::ir::PortSource {
        if let hir::Place::Index { base, index } = place
            && !matches!(**index, hir::Expr::Literal(hir::Literal::Integer(_)))
        {
            let resolved = self.lower_dynamic_index_addr(rb, base, index, dest_local, dest_ty);
            return match resolved {
                ResolvedDynamicIndex::Destination { addr, .. }
                | ResolvedDynamicIndex::Local { addr, .. } => addr,
            };
        }

        match self.resolve_place(place, dest_local, dest_ty) {
            ResolvedStructuralPlace::Destination { offset, .. } => {
                let base = rb.save_out_ptr();
                self.add_byte_offset(rb, base, offset)
            }
            ResolvedStructuralPlace::Local {
                ty,
                storage,
                slot_offset,
            } => {
                let num_slots = Self::slot_count_for_type(self.module, ty) as u32;
                rb.slot_addr(Self::slot_at(storage, slot_offset), num_slots)
            }
        }
    }

    fn lower_dynamic_index_write(
        &self,
        rb: &mut RegionBuilder<'_>,
        base: &hir::Place,
        index: &hir::Expr,
        value: &hir::Expr,
        dest_local: hir::LocalId,
        dest_ty: &'a hir::Type,
    ) {
        let resolved = self.lower_dynamic_index_addr(rb, base, index, dest_local, dest_ty);
        let value = self.lower_scalar_expr(rb, value, dest_local, dest_ty);
        match resolved {
            ResolvedDynamicIndex::Destination { ty, addr } => {
                let width = self.scalar_width_for_hir_type(ty);
                rb.store_to_addr(addr, value, width);
            }
            ResolvedDynamicIndex::Local { ty, addr } => {
                let width = self.scalar_width_for_hir_type(ty);
                rb.store_to_addr(addr, value, width);
            }
        }
    }

    fn lower_dynamic_index_write_from_local(
        &self,
        rb: &mut RegionBuilder<'_>,
        base: &hir::Place,
        index: &hir::Expr,
        local: hir::LocalId,
        dest_local: hir::LocalId,
        dest_ty: &'a hir::Type,
    ) {
        let resolved = self.lower_dynamic_index_addr(rb, base, index, dest_local, dest_ty);
        let storage = self.local_slots[&local];
        let slot_count = Self::slot_count_for_type(self.module, self.local_types[&local]);
        let base_addr = match resolved {
            ResolvedDynamicIndex::Destination { addr, .. }
            | ResolvedDynamicIndex::Local { addr, .. } => addr,
        };
        for slot_index in 0..slot_count {
            let slot = Self::slot_at(storage, slot_index);
            let value = rb.read_from_slot(slot);
            let dst_addr = self.add_byte_offset(
                rb,
                base_addr,
                slot_index * crate::ir::SLOT_ADDR_STRIDE_BYTES,
            );
            rb.store_to_addr(dst_addr, value, crate::ir::Width::W8);
        }
    }

    fn copy_dynamic_index_into_shape_offset(
        &self,
        rb: &mut RegionBuilder<'_>,
        base: &hir::Place,
        index: &hir::Expr,
        target_offset: usize,
        dest_local: hir::LocalId,
        dest_ty: &'a hir::Type,
    ) {
        let resolved = self.lower_dynamic_index_addr(rb, base, index, dest_local, dest_ty);
        match resolved {
            ResolvedDynamicIndex::Destination { ty, addr } => {
                let size = self.hir_type_size(ty);
                let full_words = size / 8;
                let remainder = size % 8;
                for word_index in 0..full_words {
                    let src_addr = self.add_byte_offset(rb, addr, word_index * 8);
                    let value = rb.load_from_addr(src_addr, crate::ir::Width::W8);
                    rb.write_to_field(
                        value,
                        (target_offset + word_index * 8) as u32,
                        crate::ir::Width::W8,
                    );
                }
                if remainder != 0 {
                    let src_addr = self.add_byte_offset(rb, addr, full_words * 8);
                    let width = match remainder {
                        1 => crate::ir::Width::W1,
                        2 => crate::ir::Width::W2,
                        4 => crate::ir::Width::W4,
                        _ => panic!(
                            "unsupported dynamic indexed destination remainder width {remainder}"
                        ),
                    };
                    let value = rb.load_from_addr(src_addr, width);
                    rb.write_to_field(value, (target_offset + full_words * 8) as u32, width);
                }
            }
            ResolvedDynamicIndex::Local { ty, addr } => {
                let slot_count = Self::slot_count_for_type(self.module, ty);
                for slot_index in 0..slot_count {
                    let src_addr = self.add_byte_offset(
                        rb,
                        addr,
                        slot_index * crate::ir::SLOT_ADDR_STRIDE_BYTES,
                    );
                    let value = rb.load_from_addr(src_addr, crate::ir::Width::W8);
                    rb.write_to_field(
                        value,
                        (target_offset + slot_index * 8) as u32,
                        crate::ir::Width::W8,
                    );
                }
            }
        }
    }

    fn lower_dynamic_index_addr(
        &self,
        rb: &mut RegionBuilder<'_>,
        base: &hir::Place,
        index: &hir::Expr,
        dest_local: hir::LocalId,
        dest_ty: &'a hir::Type,
    ) -> ResolvedDynamicIndex<'a> {
        let index = self.lower_scalar_expr(rb, index, dest_local, dest_ty);
        match self.resolve_place(base, dest_local, dest_ty) {
            ResolvedStructuralPlace::Destination { ty, offset } => {
                let hir::Type::Array { element, .. } = ty else {
                    panic!(
                        "dynamic indexed structural HIR place requires an array destination, got {ty:?}"
                    );
                };
                let elem_size = self.hir_type_size(element);
                let mut base_addr = rb.save_out_ptr();
                if offset != 0 {
                    let offset_val = rb.const_val(offset as u64);
                    base_addr = rb.binop(crate::ir::IrOp::Add, base_addr, offset_val);
                }
                let addr = self.add_scaled_index(rb, base_addr, index, elem_size);
                ResolvedDynamicIndex::Destination { ty: element, addr }
            }
            ResolvedStructuralPlace::Local {
                ty,
                storage,
                slot_offset,
            } => {
                let hir::Type::Array { element, .. } = ty else {
                    panic!("dynamic indexed local place requires an HIR array type");
                };
                let base_slot = Self::slot_at(storage, slot_offset);
                let elem_slots = Self::slot_count_for_type(self.module, element);
                let total_slots = Self::slot_count_for_type(self.module, ty);
                let base_addr = rb.slot_addr(base_slot, total_slots as u32);
                let addr = self.add_scaled_index(
                    rb,
                    base_addr,
                    index,
                    elem_slots * crate::ir::SLOT_ADDR_STRIDE_BYTES,
                );
                ResolvedDynamicIndex::Local { ty: element, addr }
            }
        }
    }

    fn resolve_place(
        &self,
        place: &hir::Place,
        dest_local: hir::LocalId,
        dest_ty: &'a hir::Type,
    ) -> ResolvedStructuralPlace<'a> {
        match place {
            hir::Place::Local(local) => {
                if *local == dest_local {
                    ResolvedStructuralPlace::Destination {
                        ty: dest_ty,
                        offset: 0,
                    }
                } else {
                    ResolvedStructuralPlace::Local {
                        ty: self.local_types[local],
                        storage: self.local_slots[local],
                        slot_offset: 0,
                    }
                }
            }
            hir::Place::Field { base, field } => {
                match self.resolve_place(base, dest_local, dest_ty) {
                    ResolvedStructuralPlace::Destination { ty, offset } => {
                        let mut ty = ty;
                        let mut offset = offset;
                        // Unwrap transparent newtypes
                        loop {
                            let hir::Type::Named { def, .. } = ty else {
                                break;
                            };
                            let type_def = &self.module.type_defs[*def];
                            if !type_def.transparent {
                                break;
                            }
                            let hir::TypeDefKind::Struct { fields } = &type_def.kind else {
                                break;
                            };
                            assert_eq!(
                                fields.len(),
                                1,
                                "structural HIR subset requires transparent wrappers to lower to one field"
                            );
                            let f = &fields[0];
                            offset += f.offset.map(|o| o as usize).unwrap_or(0);
                            ty = &f.ty;
                        }
                        // For Bits128Raw-like types (lo/hi u64 fields), fall through
                        // to normal struct field resolution — the field's own HIR type
                        // will be used for width.
                        let hir::Type::Named { def, .. } = ty else {
                            panic!(
                                "destination field place requires a named struct type, got {ty:?} for field {field}"
                            );
                        };
                        let type_def = &self.module.type_defs[*def];
                        let hir::TypeDefKind::Struct { fields } = &type_def.kind else {
                            panic!(
                                "destination field place requires a struct type def for {}",
                                type_def.name
                            );
                        };
                        let field_offset = self.find_field_byte_offset(fields, field);
                        let field_def = fields
                            .iter()
                            .find(|candidate| candidate.name == field.as_str())
                            .unwrap_or_else(|| {
                                panic!(
                                    "missing field {field} while lowering structural HIR place for {}",
                                    type_def.name
                                )
                            });
                        ResolvedStructuralPlace::Destination {
                            ty: &field_def.ty,
                            offset: offset + field_offset,
                        }
                    }
                    ResolvedStructuralPlace::Local {
                        ty,
                        storage,
                        slot_offset,
                    } => {
                        let (field_ty, field_offset) = match ty {
                            hir::Type::Named { def, .. } => {
                                let hir::TypeDefKind::Struct { fields } =
                                    &self.module.type_defs[*def].kind
                                else {
                                    panic!("local field place requires a struct type");
                                };
                                let mut running_slots = 0usize;
                                fields
                                    .iter()
                                    .find_map(|candidate| {
                                        let found = (candidate.name == field.as_str())
                                            .then_some((&candidate.ty, running_slots));
                                        running_slots +=
                                            Self::slot_count_for_type(self.module, &candidate.ty);
                                        found
                                    })
                                    .unwrap_or_else(|| {
                                        panic!(
                                            "missing HIR struct field {field} while resolving local place"
                                        )
                                    })
                            }
                            hir::Type::Slice { .. } | hir::Type::Str { .. } => {
                                let (field_offset, field_ty) = match field.as_str() {
                                    "ptr" | "data" => (0, hir::Type::u(64)),
                                    "len" => (1, hir::Type::u(64)),
                                    _ => panic!(
                                        "local field place requires a known slice-like field, got {field}"
                                    ),
                                };
                                (Box::leak(Box::new(field_ty)) as &'a hir::Type, field_offset)
                            }
                            _ => {
                                panic!(
                                    "local field place requires a struct-like type, got {ty:?} for field {field}"
                                );
                            }
                        };
                        ResolvedStructuralPlace::Local {
                            ty: field_ty,
                            storage,
                            slot_offset: slot_offset + field_offset,
                        }
                    }
                }
            }
            hir::Place::Index { base, index } => {
                let hir::Expr::Literal(hir::Literal::Integer(index)) = &**index else {
                    panic!("structural HIR array indices must be integer literals");
                };
                let index = usize::try_from(*index).expect("array index must fit in usize");
                match self.resolve_place(base, dest_local, dest_ty) {
                    ResolvedStructuralPlace::Destination { ty, offset } => {
                        let hir::Type::Array { element, len } = ty else {
                            panic!(
                                "indexed structural HIR destination requires an array type, got {ty:?}"
                            );
                        };
                        assert!(
                            index < *len,
                            "array index {index} out of bounds for array of len {len}"
                        );
                        let elem_size = self.hir_type_size(element);
                        ResolvedStructuralPlace::Destination {
                            ty: element,
                            offset: offset + index * elem_size,
                        }
                    }
                    ResolvedStructuralPlace::Local {
                        ty,
                        storage,
                        slot_offset,
                    } => {
                        let hir::Type::Array { element, len } = ty else {
                            panic!("indexed local place requires an HIR array type");
                        };
                        assert!(
                            index < *len,
                            "local array index {index} out of bounds for {len}"
                        );
                        let elem_slots = Self::slot_count_for_type(self.module, element);
                        ResolvedStructuralPlace::Local {
                            ty: element,
                            storage,
                            slot_offset: slot_offset + index * elem_slots,
                        }
                    }
                }
            }
        }
    }

    fn scalar_width_for_hir_type(&self, ty: &hir::Type) -> crate::ir::Width {
        match ty {
            hir::Type::Bool => crate::ir::Width::W1,
            hir::Type::Integer(kind) => match kind.bits {
                8 => crate::ir::Width::W1,
                16 => crate::ir::Width::W2,
                32 => crate::ir::Width::W4,
                64 => crate::ir::Width::W8,
                other => panic!("unsupported structural HIR integer width: {other}"),
            },
            hir::Type::Address { .. } => crate::ir::Width::W8,
            _ => panic!("unsupported structural HIR scalar local type: {ty:?}"),
        }
    }

    fn ir_width_for_memory_width(&self, width: hir::MemoryWidth) -> crate::ir::Width {
        match width {
            hir::MemoryWidth::W1 => crate::ir::Width::W1,
            hir::MemoryWidth::W2 => crate::ir::Width::W2,
            hir::MemoryWidth::W4 => crate::ir::Width::W4,
            hir::MemoryWidth::W8 => crate::ir::Width::W8,
        }
    }

    /// Returns true if the HIR type is a simple scalar (non-string-like, non-compound).
    fn is_simple_scalar_hir_type(&self, ty: &hir::Type) -> bool {
        matches!(
            ty,
            hir::Type::Bool
                | hir::Type::Integer(_)
                | hir::Type::Address { .. }
                | hir::Type::Handle { .. }
        )
    }

    /// Returns the byte size of an HIR type by looking up type def annotations,
    /// or computing it from the type structure when annotations are absent.
    fn hir_type_size(&self, ty: &hir::Type) -> usize {
        match ty {
            hir::Type::Unit => 0,
            hir::Type::Bool => 1,
            hir::Type::Integer(kind) => (kind.bits as usize) / 8,
            hir::Type::Address { .. } => core::mem::size_of::<usize>(),
            hir::Type::Handle { .. } => core::mem::size_of::<usize>(),
            hir::Type::Str { .. } | hir::Type::Slice { .. } => core::mem::size_of::<usize>() * 2,
            hir::Type::Array { element, len } => self.hir_type_size(element) * len,
            hir::Type::Named { def, .. } => {
                let type_def = &self.module.type_defs[*def];
                if let Some(size) = type_def.size {
                    return size as usize;
                }
                // Fallback: compute size from fields/variants
                self.compute_type_def_size(type_def)
            }
        }
    }

    /// Compute the size of a type def from its fields when no size annotation exists.
    fn compute_type_def_size(&self, type_def: &hir::TypeDef) -> usize {
        match &type_def.kind {
            hir::TypeDefKind::Struct { fields } => {
                // If fields have offsets, use offset + size of last field
                let mut max_end = 0usize;
                for field in fields {
                    if let Some(offset) = field.offset {
                        let field_size = self.hir_type_size(&field.ty);
                        max_end = max_end.max(offset as usize + field_size);
                    } else {
                        // No offset annotation — sum field sizes (no padding)
                        max_end += self.hir_type_size(&field.ty);
                    }
                }
                max_end
            }
            hir::TypeDefKind::Enum {
                variants,
                discriminant_width,
            } => {
                let disc_size = discriminant_width.unwrap_or(1) as usize;
                let max_payload = variants
                    .iter()
                    .map(|variant| {
                        variant
                            .fields
                            .iter()
                            .map(|f| {
                                if let Some(offset) = f.offset {
                                    offset as usize + self.hir_type_size(&f.ty)
                                } else {
                                    self.hir_type_size(&f.ty)
                                }
                            })
                            .max()
                            .unwrap_or(0)
                    })
                    .max()
                    .unwrap_or(0);
                disc_size + max_payload
            }
        }
    }

    /// Find the byte offset of a named field within a struct's fields.
    /// Uses the field's `offset` annotation if present, otherwise computes
    /// from preceding field sizes (packed layout).
    fn find_field_byte_offset(&self, fields: &[hir::FieldDef], field_name: &str) -> usize {
        // If the target field has an explicit offset, use it
        for f in fields {
            if f.name == field_name {
                if let Some(offset) = f.offset {
                    return offset as usize;
                }
            }
        }
        // Fallback: compute offset by summing sizes of preceding fields
        let mut running_offset = 0usize;
        for f in fields {
            if f.name == field_name {
                return running_offset;
            }
            running_offset += self.hir_type_size(&f.ty);
        }
        panic!("field {field_name} not found in struct fields");
    }

    /// Lower a nested variant write (e.g. variant payload containing another variant).
    fn lower_nested_variant_write(
        &self,
        rb: &mut RegionBuilder<'_>,
        offset: usize,
        ty: &'a hir::Type,
        variant: &str,
        fields: &[(String, hir::Expr)],
        dest_local: hir::LocalId,
        dest_ty: &'a hir::Type,
    ) {
        let hir::Type::Named { def, .. } = ty else {
            panic!("nested variant write requires a named enum type, got {ty:?}");
        };
        let type_def = &self.module.type_defs[*def];
        let hir::TypeDefKind::Enum {
            variants,
            discriminant_width,
        } = &type_def.kind
        else {
            panic!(
                "nested variant write requires an enum type def, got {}",
                type_def.name
            );
        };
        let hir_variant = variants
            .iter()
            .find(|v| v.name == variant)
            .unwrap_or_else(|| panic!("missing enum variant {variant} in {}", type_def.name));
        let disc_width_val = discriminant_width.expect("enum must have discriminant_width");
        let disc_ir_width = ir_width_from_disc_size(disc_width_val);
        let disc: u64 = hir_variant
            .discriminant
            .expect("enum variant must have discriminant")
            .try_into()
            .expect("enum discriminant must fit in u64");
        let value = rb.const_val(disc);
        rb.write_to_field(value, offset as u32, disc_ir_width);
        for hir_field in &hir_variant.fields {
            let (_, expr) = fields
                .iter()
                .find(|(name, _)| name == &hir_field.name)
                .unwrap_or_else(|| {
                    panic!(
                        "missing enum payload field {} for variant {variant}",
                        hir_field.name
                    )
                });
            let field_offset = hir_field.offset.expect("variant field must have offset") as usize;
            self.lower_value_into_dest_offset(
                rb,
                &hir_field.ty,
                offset + field_offset,
                expr,
                dest_local,
                dest_ty,
            );
        }
    }
}

pub(crate) fn build_structural_hir_ir(module: &hir::Module) -> crate::ir::IrFunc {
    build_structural_hir_ir_impl(module)
}

/// Lower HIR to IR without any facet Shape. All layout info must come from
/// HIR annotations.
///
/// Dispatches to either the structural (destination-writing) path or the
/// scalar (return-value) path depending on whether the function has a
/// destination parameter.
pub fn lower_hir_module(module: &hir::Module) -> crate::ir::IrFunc {
    let (_, function) = module
        .functions
        .iter()
        .next()
        .expect("HIR module should contain at least one function");
    if function.destination_param().is_some() {
        build_structural_hir_ir_impl(module)
    } else {
        build_scalar_hir_ir(module, function)
    }
}

/// Lower a plain scalar HIR function into IR.
///
/// This is the clean lowering path for functions that take ordinary params
/// and return a value — no cursor, no destination, no decoder-specific
/// machinery. All params get slots, all locals get slots, expressions lower
/// to IR operations, and `return expr` sets the region result.
/// Check if an HIR function body uses effect callables that require the MEMORY state domain.
fn hir_function_uses_effect_calls(module: &hir::Module, function: &hir::Function) -> bool {
    fn expr_uses_effects(module: &hir::Module, expr: &hir::Expr) -> bool {
        match expr {
            hir::Expr::Call(call) => {
                let id = match call.target {
                    hir::CallTarget::Callable(id) => id,
                };
                if matches!(
                    module.callables[id].intrinsic,
                    Some(
                        hir::RuntimeIntrinsic::AllocTransient
                            | hir::RuntimeIntrinsic::Memcpy
                            | hir::RuntimeIntrinsic::FreeTransient
                    )
                ) {
                    return true;
                }
                call.args.iter().any(|a| expr_uses_effects(module, a))
            }
            hir::Expr::Binary { lhs, rhs, .. } => {
                expr_uses_effects(module, lhs) || expr_uses_effects(module, rhs)
            }
            hir::Expr::Unary { value, .. } => expr_uses_effects(module, value),
            hir::Expr::Field { base, .. } => expr_uses_effects(module, base),
            hir::Expr::Index { base, index } => {
                expr_uses_effects(module, base) || expr_uses_effects(module, index)
            }
            hir::Expr::Struct { fields, .. } | hir::Expr::Variant { fields, .. } => {
                fields.iter().any(|(_, e)| expr_uses_effects(module, e))
            }
            _ => false,
        }
    }
    fn block_uses_effects(module: &hir::Module, stmts: &[hir::Stmt]) -> bool {
        stmts.iter().any(|stmt| match &stmt.kind {
            hir::StmtKind::Init { value, .. }
            | hir::StmtKind::Assign { value, .. }
            | hir::StmtKind::Expr(value) => expr_uses_effects(module, value),
            hir::StmtKind::Store { addr, value, .. } => {
                expr_uses_effects(module, addr) || expr_uses_effects(module, value)
            }
            hir::StmtKind::Return(Some(e)) => expr_uses_effects(module, e),
            hir::StmtKind::If {
                condition,
                then_block,
                else_block,
            } => {
                expr_uses_effects(module, condition)
                    || block_uses_effects(module, &then_block.statements)
                    || else_block
                        .as_ref()
                        .map_or(false, |b| block_uses_effects(module, &b.statements))
            }
            hir::StmtKind::Loop { body, .. } => block_uses_effects(module, &body.statements),
            hir::StmtKind::Match { scrutinee, arms } => {
                expr_uses_effects(module, scrutinee)
                    || arms
                        .iter()
                        .any(|arm| block_uses_effects(module, &arm.body.statements))
            }
            _ => false,
        })
    }
    block_uses_effects(module, &function.body.statements)
}

fn build_scalar_hir_ir(module: &hir::Module, function: &hir::Function) -> crate::ir::IrFunc {
    // Count the total number of u64-sized words across all params.
    let param_word_count: usize = function
        .params
        .iter()
        .map(|p| ScalarHirIrLowerer::word_count_for_type(module, &p.ty))
        .sum();

    let (mut builder, data_arg_sources) =
        crate::ir::IrBuilder::new_with_data_args(&function.name, 0, param_word_count);
    // Only add the MEMORY state domain when the function uses effect calls.
    if hir_function_uses_effect_calls(module, function) {
        let _ = builder.add_state_domain(crate::ir::MEMORY_STATE_DOMAIN_NAME);
    }
    {
        let mut rb = builder.root_region();
        let mut lowerer = ScalarHirIrLowerer::new(module, function, &data_arg_sources);
        let ret = lowerer.lower_block(&mut rb, &function.body.statements);
        match ret {
            Some(ret_vals) => rb.set_results(&ret_vals),
            None => rb.set_results(&[]),
        }
    }
    let mut func = builder.finish();
    func.param_slot_count = param_word_count as u32;
    func.is_scalar = true;
    func
}

struct ScalarHirIrLowerer<'a> {
    module: &'a hir::Module,
    /// Port sources for each local (params and locals). Multi-word types
    /// have multiple port sources (one per u64 word).
    local_values: std::collections::HashMap<hir::LocalId, Vec<crate::ir::PortSource>>,
    /// Type of each local, for field resolution.
    local_types: std::collections::HashMap<hir::LocalId, &'a hir::Type>,
}

impl<'a> ScalarHirIrLowerer<'a> {
    fn new(
        module: &'a hir::Module,
        function: &'a hir::Function,
        data_arg_sources: &[crate::ir::PortSource],
    ) -> Self {
        let mut local_values = std::collections::HashMap::new();
        let mut local_types = std::collections::HashMap::new();

        // Params: consume data arg port sources in order.
        let mut arg_cursor = 0;
        for param in &function.params {
            let word_count = Self::word_count_for_type(module, &param.ty);
            let sources = data_arg_sources[arg_cursor..arg_cursor + word_count].to_vec();
            arg_cursor += word_count;
            local_values.insert(param.local, sources);
            local_types.insert(param.local, &param.ty);
        }
        assert_eq!(arg_cursor, data_arg_sources.len());

        // Locals: register types. Values are populated by Let or field writes.
        for local in &function.locals {
            local_types.insert(local.local, &local.ty);
        }

        Self {
            module,
            local_values,
            local_types,
        }
    }

    /// Number of u64-sized words needed to represent a type.
    fn word_count_for_type(module: &hir::Module, ty: &hir::Type) -> usize {
        match ty {
            hir::Type::Unit
            | hir::Type::Bool
            | hir::Type::Integer(_)
            | hir::Type::Address { .. }
            | hir::Type::Handle { .. } => 1,
            hir::Type::Str { .. } | hir::Type::Slice { .. } => 2,
            hir::Type::Array { element, len } => Self::word_count_for_type(module, element)
                .saturating_mul(*len)
                .max(1),
            hir::Type::Named { def, .. } => match &module.type_defs[*def].kind {
                hir::TypeDefKind::Struct { fields } => fields
                    .iter()
                    .map(|f| Self::word_count_for_type(module, &f.ty))
                    .sum::<usize>()
                    .max(1),
                hir::TypeDefKind::Enum { variants, .. } => {
                    let payload_words = variants
                        .iter()
                        .map(|v| {
                            v.fields
                                .iter()
                                .map(|f| Self::word_count_for_type(module, &f.ty))
                                .sum::<usize>()
                        })
                        .max()
                        .unwrap_or(0);
                    (1 + payload_words).max(1)
                }
            },
        }
    }

    /// Resolve a local + field path to the type and word offset within the
    /// local's port source vector.
    fn resolve_local_field(
        &self,
        local: hir::LocalId,
        fields: &[(String, &hir::Type)],
    ) -> (usize, &'a hir::Type) {
        let mut offset = 0usize;
        let mut ty = self.local_types[&local];
        for (field_name, _) in fields {
            let hir::Type::Named { def, .. } = ty else {
                panic!("field access requires a named struct type, got {:?}", ty);
            };
            let hir::TypeDefKind::Struct { fields: field_defs } = &self.module.type_defs[*def].kind
            else {
                panic!("field access requires a struct type def");
            };
            let mut running = 0usize;
            let (field_ty, field_offset) = field_defs
                .iter()
                .find_map(|candidate| {
                    let found =
                        (candidate.name == field_name.as_str()).then_some((&candidate.ty, running));
                    running += Self::word_count_for_type(self.module, &candidate.ty);
                    found
                })
                .unwrap_or_else(|| panic!("missing struct field {field_name}"));
            offset += field_offset;
            ty = field_ty;
        }
        (offset, ty)
    }

    /// Collect the chain of field accesses from an expression.
    /// Returns (base local, list of field names with types).
    fn expr_field_chain(expr: &hir::Expr) -> (hir::LocalId, Vec<(String, &hir::Type)>) {
        match expr {
            hir::Expr::Local(local) => (*local, vec![]),
            hir::Expr::Field { base, field } => {
                let (local, mut chain) = Self::expr_field_chain(base);
                chain.push((field.clone(), &hir::Type::Unit));
                (local, chain)
            }
            other => panic!("expected local or field chain, got {other:?}"),
        }
    }

    /// Non-panicking version of `expr_field_chain`. Returns `None` when the
    /// base is not a local-rooted field chain.
    fn try_expr_field_chain(expr: &hir::Expr) -> Option<(hir::LocalId, Vec<(String, &hir::Type)>)> {
        match expr {
            hir::Expr::Local(local) => Some((*local, vec![])),
            hir::Expr::Field { base, field } => {
                let (local, mut chain) = Self::try_expr_field_chain(base)?;
                chain.push((field.clone(), &hir::Type::Unit));
                Some((local, chain))
            }
            _ => None,
        }
    }

    /// Infer the type of an HIR expression. Used by the type-driven field
    /// projection fallback when the base is not a local-rooted chain.
    fn infer_expr_type(&self, expr: &hir::Expr) -> hir::Type {
        match expr {
            hir::Expr::Local(id) => self.local_types[id].clone(),
            hir::Expr::Literal(hir::Literal::Bool(_)) => hir::Type::Bool,
            hir::Expr::Literal(hir::Literal::Integer(_)) => hir::Type::u(64),
            hir::Expr::Literal(hir::Literal::String(_)) => {
                // String literals produce (ptr, len) — a built-in Str type.
                hir::Type::Str {
                    region: hir::RegionId::new(0),
                }
            }
            hir::Expr::Literal(hir::Literal::Unit) => hir::Type::Unit,
            hir::Expr::Struct { def, .. } => hir::Type::Named {
                def: *def,
                args: vec![],
            },
            hir::Expr::Field { base, field } => {
                let base_ty = self.infer_expr_type(base);
                self.resolve_field_type_in(&base_ty, field)
            }
            hir::Expr::Binary { op, .. } => {
                // Comparison ops produce bool, arithmetic ops produce u64.
                match op {
                    hir::BinaryOp::Eq
                    | hir::BinaryOp::Ne
                    | hir::BinaryOp::Lt
                    | hir::BinaryOp::Le
                    | hir::BinaryOp::Gt
                    | hir::BinaryOp::Ge => hir::Type::Bool,
                    _ => hir::Type::u(64),
                }
            }
            hir::Expr::Unary { .. } => hir::Type::u(64),
            hir::Expr::Call(call) => {
                let callable = &self.module.callables[match call.target {
                    hir::CallTarget::Callable(id) => id,
                }];
                callable
                    .signature
                    .returns
                    .first()
                    .cloned()
                    .unwrap_or(hir::Type::Unit)
            }
            hir::Expr::SliceData { .. } | hir::Expr::SliceLen { .. } => hir::Type::u(64),
            hir::Expr::Load { .. } => hir::Type::u(64),
            other => panic!("infer_expr_type: cannot infer type of expression: {other:?}"),
        }
    }

    /// Resolve the type of a field within a parent type.
    fn resolve_field_type_in(&self, ty: &hir::Type, field: &str) -> hir::Type {
        match ty {
            hir::Type::Named { def, .. } => {
                let hir::TypeDefKind::Struct { fields } = &self.module.type_defs[*def].kind else {
                    panic!("field access on non-struct named type: {:?}", ty);
                };
                fields
                    .iter()
                    .find(|f| f.name == field)
                    .map(|f| f.ty.clone())
                    .unwrap_or_else(|| panic!("field '{field}' not found in struct"))
            }
            hir::Type::Str { .. } => match field {
                "ptr" | "data" => hir::Type::u(64),
                "len" => hir::Type::u(64),
                _ => panic!("unknown Str field: '{field}'"),
            },
            hir::Type::Slice { .. } => match field {
                "data" | "ptr" => hir::Type::u(64),
                "len" => hir::Type::u(64),
                _ => panic!("unknown Slice field: '{field}'"),
            },
            _ => panic!("field access on type that has no fields: {:?}", ty),
        }
    }

    /// Compute the word offset and word count of a field within a type.
    /// Returns `(word_offset, field_word_count)`.
    fn field_offset_in_type(&self, ty: &hir::Type, field_name: &str) -> (usize, usize) {
        match ty {
            hir::Type::Named { def, .. } => {
                let hir::TypeDefKind::Struct { fields } = &self.module.type_defs[*def].kind else {
                    panic!("field_offset_in_type: not a struct: {:?}", ty);
                };
                let mut offset = 0;
                for f in fields {
                    let wc = Self::word_count_for_type(self.module, &f.ty);
                    if f.name == field_name {
                        return (offset, wc);
                    }
                    offset += wc;
                }
                panic!("field '{field_name}' not found in struct {:?}", def);
            }
            hir::Type::Str { .. } => match field_name {
                "ptr" | "data" => (0, 1),
                "len" => (1, 1),
                _ => panic!("unknown Str field: '{field_name}'"),
            },
            hir::Type::Slice { .. } => match field_name {
                "data" | "ptr" => (0, 1),
                "len" => (1, 1),
                _ => panic!("unknown Slice field: '{field_name}'"),
            },
            _ => panic!(
                "field_offset_in_type: type {:?} has no field '{}'",
                ty, field_name
            ),
        }
    }

    /// Collect the chain of field accesses from a place.
    fn place_field_chain(place: &hir::Place) -> (hir::LocalId, Vec<(String, &hir::Type)>) {
        match place {
            hir::Place::Local(local) => (*local, vec![]),
            hir::Place::Field { base, field } => {
                let (local, mut chain) = Self::place_field_chain(base);
                chain.push((field.clone(), &hir::Type::Unit));
                (local, chain)
            }
            other => panic!("expected local or field place, got {other:?}"),
        }
    }

    /// Get the port sources for an expression (one per word).
    fn get_local_values(&self, local: hir::LocalId) -> &[crate::ir::PortSource] {
        self.local_values
            .get(&local)
            .unwrap_or_else(|| panic!("local {local:?} not yet defined"))
    }

    fn lower_block(
        &mut self,
        rb: &mut RegionBuilder<'_>,
        statements: &[hir::Stmt],
    ) -> Option<Vec<crate::ir::PortSource>> {
        for stmt in statements {
            if let Some(ret) = self.lower_stmt(rb, stmt) {
                return Some(ret);
            }
        }
        None
    }

    /// Lower an expression into one or more port sources (one per word).
    fn lower_expr_multi(
        &self,
        rb: &mut RegionBuilder<'_>,
        expr: &hir::Expr,
    ) -> Vec<crate::ir::PortSource> {
        match expr {
            hir::Expr::Local(local) => self.get_local_values(*local).to_vec(),
            hir::Expr::Field { base, field } => {
                // Fast path: local-rooted field chain (most common case).
                if let Some((local, chain)) = Self::try_expr_field_chain(expr) {
                    let values = self.get_local_values(local);
                    let (offset, ty) = self.resolve_local_field(local, &chain);
                    let word_count = Self::word_count_for_type(self.module, ty);
                    return values[offset..offset + word_count].to_vec();
                }
                // Type-driven fallback: evaluate base to word vector, project field.
                let words = self.lower_expr_multi(rb, base);
                let base_ty = self.infer_expr_type(base);
                let (offset, count) = self.field_offset_in_type(&base_ty, field);
                words[offset..offset + count].to_vec()
            }
            hir::Expr::Struct { def, fields } => {
                let hir::TypeDefKind::Struct { fields: field_defs } =
                    &self.module.type_defs[*def].kind
                else {
                    panic!("Expr::Struct requires a struct type def");
                };
                let mut result = Vec::new();
                for field_def in field_defs {
                    let (_, expr) = fields
                        .iter()
                        .find(|(name, _)| name == &field_def.name)
                        .unwrap_or_else(|| panic!("missing struct field {}", field_def.name));
                    result.extend(self.lower_expr_multi(rb, expr));
                }
                result
            }
            hir::Expr::Literal(hir::Literal::String(s)) => {
                // String literal → (ptr, len) where ptr is a DataAddr relocation.
                let blob_id = rb.add_data_blob(s.as_bytes().to_vec());
                let ptr = rb.data_addr(blob_id);
                let len = rb.const_val(s.len() as u64);
                vec![ptr, len]
            }
            _ => vec![self.lower_expr(rb, expr)],
        }
    }

    fn lower_stmt(
        &mut self,
        rb: &mut RegionBuilder<'_>,
        stmt: &hir::Stmt,
    ) -> Option<Vec<crate::ir::PortSource>> {
        match &stmt.kind {
            hir::StmtKind::Init { place, value } | hir::StmtKind::Assign { place, value } => {
                let new_values = self.lower_expr_multi(rb, value);
                match place {
                    hir::Place::Local(local) => {
                        self.local_values.insert(*local, new_values);
                    }
                    hir::Place::Field { .. } => {
                        // Field write: update specific words within the local's vector.
                        let (local, chain) = Self::place_field_chain(place);
                        let (offset, ty) = self.resolve_local_field(local, &chain);
                        let word_count = Self::word_count_for_type(self.module, ty);
                        assert_eq!(new_values.len(), word_count);
                        // Lazily initialize the local with zero constants if needed.
                        if !self.local_values.contains_key(&local) {
                            let local_ty = self.local_types[&local];
                            let total_words = Self::word_count_for_type(self.module, local_ty);
                            let zero = rb.const_val(0);
                            self.local_values.insert(local, vec![zero; total_words]);
                        }
                        let values = self.local_values.get_mut(&local).unwrap();
                        for (i, val) in new_values.into_iter().enumerate() {
                            values[offset + i] = val;
                        }
                    }
                    other => panic!("unsupported scalar HIR place: {other:?}"),
                }
                None
            }
            hir::StmtKind::Expr(hir::Expr::Call(call)) => {
                let args: Vec<_> = call
                    .args
                    .iter()
                    .map(|arg| self.lower_expr(rb, arg))
                    .collect();
                let callable = &self.module.callables[match call.target {
                    hir::CallTarget::Callable(id) => id,
                }];
                let func = match callable.intrinsic {
                    Some(hir::RuntimeIntrinsic::FreeTransient) => crate::ir::IntrinsicFn(
                        intrinsics::kajit_free_transient as *const () as usize,
                    ),
                    other => panic!(
                        "unsupported scalar HIR effect call: {} ({other:?})",
                        callable.name
                    ),
                };
                // Void-returning effectful call: use call_effect but ignore result.
                let _result = rb.call_effect(func, &args);
                None
            }
            hir::StmtKind::Expr(_) => None,
            hir::StmtKind::Fail { code } => {
                rb.error_exit(*code);
                None
            }
            hir::StmtKind::If {
                condition,
                then_block,
                else_block,
            } => {
                let predicate = self.lower_expr(rb, condition);
                let else_block = else_block
                    .as_ref()
                    .expect("scalar HIR if requires else branch");
                let mut branch_ret_count: Option<usize> = None;
                let gamma_outputs = rb.gamma(predicate, &[], 2, |branch_idx, branch| {
                    let block = match branch_idx {
                        0 => &else_block.statements,
                        1 => &then_block.statements,
                        _ => unreachable!(),
                    };
                    if let Some(vals) = self.lower_block(branch, block) {
                        if let Some(expected) = branch_ret_count {
                            assert_eq!(
                                vals.len(),
                                expected,
                                "gamma branches must return same number of values"
                            );
                        }
                        branch_ret_count = Some(vals.len());
                        branch.set_results(&vals);
                    } else {
                        branch.set_results(&[]);
                    }
                });
                if branch_ret_count.is_some() {
                    Some(gamma_outputs)
                } else {
                    None
                }
            }
            hir::StmtKind::Match { scrutinee, arms } => {
                let predicate = self.lower_expr(rb, scrutinee);
                for (expected, arm) in arms.iter().enumerate() {
                    let hir::Pattern::Integer(value) = arm.pattern else {
                        panic!("scalar HIR only supports integer match patterns");
                    };
                    assert_eq!(
                        value, expected as u64,
                        "scalar HIR requires contiguous integer match arms starting at 0"
                    );
                }
                let mut branch_ret_count: Option<usize> = None;
                let gamma_outputs = rb.gamma(predicate, &[], arms.len(), |branch_idx, branch| {
                    if let Some(vals) = self.lower_block(branch, &arms[branch_idx].body.statements)
                    {
                        if let Some(expected) = branch_ret_count {
                            assert_eq!(
                                vals.len(),
                                expected,
                                "gamma branches must return same number of values"
                            );
                        }
                        branch_ret_count = Some(vals.len());
                        branch.set_results(&vals);
                    } else {
                        branch.set_results(&[]);
                    }
                });
                if branch_ret_count.is_some() {
                    Some(gamma_outputs)
                } else {
                    None
                }
            }
            hir::StmtKind::Loop {
                body,
                max_iterations,
                ..
            } => {
                let active_slot = rb.alloc_slot();
                let continue_slot = rb.alloc_slot();
                let build_body = |body_rb: &mut kajit_ir::RegionBuilder<'_>| {
                    let one = body_rb.const_val(1);
                    body_rb.write_to_slot(active_slot, one);
                    body_rb.write_to_slot(continue_slot, one);
                    self.lower_loop_block(body_rb, &body.statements, active_slot, continue_slot);
                    let predicate = body_rb.read_from_slot(continue_slot);
                    body_rb.set_results(&[predicate]);
                };
                if let Some(max_iter) = max_iterations {
                    let _ = rb.theta_bounded(&[], *max_iter, build_body);
                } else {
                    let _ = rb.theta(&[], build_body);
                }
                None
            }
            hir::StmtKind::Return(Some(expr)) => Some(self.lower_expr_multi(rb, expr)),
            hir::StmtKind::Return(None) => None,
            other => panic!("unsupported scalar HIR statement: {other:?}"),
        }
    }

    fn lower_loop_block(
        &mut self,
        rb: &mut RegionBuilder<'_>,
        statements: &[hir::Stmt],
        active_slot: crate::ir::SlotId,
        continue_slot: crate::ir::SlotId,
    ) {
        let mut i = 0;
        while i < statements.len() {
            let start = i;
            while i < statements.len() && !Self::is_loop_control_flow(&statements[i]) {
                i += 1;
            }
            if start < i {
                self.with_active_guard(rb, active_slot, |guard_rb, this| {
                    for stmt in &statements[start..i] {
                        this.lower_stmt(guard_rb, stmt);
                    }
                });
            }
            if i < statements.len() {
                self.lower_loop_control_flow_stmt(rb, &statements[i], active_slot, continue_slot);
                i += 1;
            }
        }
    }

    fn is_loop_control_flow(stmt: &hir::Stmt) -> bool {
        matches!(
            stmt.kind,
            hir::StmtKind::Break
                | hir::StmtKind::Continue
                | hir::StmtKind::If { .. }
                | hir::StmtKind::Match { .. }
                | hir::StmtKind::Loop { .. }
        )
    }

    fn lower_loop_control_flow_stmt(
        &mut self,
        rb: &mut RegionBuilder<'_>,
        stmt: &hir::Stmt,
        active_slot: crate::ir::SlotId,
        continue_slot: crate::ir::SlotId,
    ) {
        self.with_active_guard(rb, active_slot, |guard_rb, this| match &stmt.kind {
            hir::StmtKind::Break => {
                let zero = guard_rb.const_val(0);
                guard_rb.write_to_slot(active_slot, zero);
                guard_rb.write_to_slot(continue_slot, zero);
            }
            hir::StmtKind::Continue => {
                let zero = guard_rb.const_val(0);
                let one = guard_rb.const_val(1);
                guard_rb.write_to_slot(active_slot, zero);
                guard_rb.write_to_slot(continue_slot, one);
            }
            hir::StmtKind::If {
                condition,
                then_block,
                else_block,
            } => {
                let predicate = this.lower_expr(guard_rb, condition);
                let else_block = else_block
                    .as_ref()
                    .expect("scalar HIR loop if requires else branch");
                let _ = guard_rb.gamma(predicate, &[], 2, |branch_idx, branch| {
                    match branch_idx {
                        0 => this.lower_loop_block(
                            branch,
                            &else_block.statements,
                            active_slot,
                            continue_slot,
                        ),
                        1 => this.lower_loop_block(
                            branch,
                            &then_block.statements,
                            active_slot,
                            continue_slot,
                        ),
                        _ => unreachable!(),
                    }
                    branch.set_results(&[]);
                });
            }
            hir::StmtKind::Match { scrutinee, arms } => {
                let predicate = this.lower_expr(guard_rb, scrutinee);
                for (expected, arm) in arms.iter().enumerate() {
                    let hir::Pattern::Integer(value) = arm.pattern else {
                        panic!("scalar HIR loop only supports integer match patterns");
                    };
                    assert_eq!(
                        value, expected as u64,
                        "scalar HIR loop requires contiguous integer match arms starting at 0"
                    );
                }
                let _ = guard_rb.gamma(predicate, &[], arms.len(), |branch_idx, branch| {
                    this.lower_loop_block(
                        branch,
                        &arms[branch_idx].body.statements,
                        active_slot,
                        continue_slot,
                    );
                    branch.set_results(&[]);
                });
            }
            hir::StmtKind::Loop {
                body,
                max_iterations,
                ..
            } => {
                let nested_active = guard_rb.alloc_slot();
                let nested_continue = guard_rb.alloc_slot();
                let build_body = |body_rb: &mut kajit_ir::RegionBuilder<'_>| {
                    let one = body_rb.const_val(1);
                    body_rb.write_to_slot(nested_active, one);
                    body_rb.write_to_slot(nested_continue, one);
                    this.lower_loop_block(
                        body_rb,
                        &body.statements,
                        nested_active,
                        nested_continue,
                    );
                    let predicate = body_rb.read_from_slot(nested_continue);
                    body_rb.set_results(&[predicate]);
                };
                if let Some(max_iter) = max_iterations {
                    let _ = guard_rb.theta_bounded(&[], *max_iter, build_body);
                } else {
                    let _ = guard_rb.theta(&[], build_body);
                }
            }
            other => {
                panic!("is_loop_control_flow returned true for non-control-flow: {other:?}")
            }
        });
    }

    fn with_active_guard(
        &mut self,
        rb: &mut RegionBuilder<'_>,
        active_slot: crate::ir::SlotId,
        f: impl FnOnce(&mut RegionBuilder<'_>, &mut Self),
    ) {
        let active = rb.read_from_slot(active_slot);
        let mut f = Some(f);
        let _ = rb.gamma(active, &[], 2, |branch_idx, branch| {
            if branch_idx == 1 {
                f.take().expect("active branch should lower exactly once")(branch, self);
            }
            branch.set_results(&[]);
        });
    }

    fn lower_call_expr(
        &self,
        rb: &mut RegionBuilder<'_>,
        call: &hir::CallExpr,
    ) -> crate::ir::PortSource {
        let args: Vec<_> = call
            .args
            .iter()
            .map(|arg| self.lower_expr(rb, arg))
            .collect();
        let callable = &self.module.callables[match call.target {
            hir::CallTarget::Callable(id) => id,
        }];
        let func = match callable.intrinsic {
            Some(hir::RuntimeIntrinsic::AllocTransient) => {
                crate::ir::IntrinsicFn(intrinsics::kajit_alloc_transient as *const () as usize)
            }
            Some(hir::RuntimeIntrinsic::Memcpy) => {
                crate::ir::IntrinsicFn(intrinsics::kajit_memcpy as *const () as usize)
            }
            other => panic!(
                "unsupported scalar HIR call target: {} ({other:?})",
                callable.name
            ),
        };
        rb.call_effect(func, &args)
    }

    fn lower_expr(&self, rb: &mut RegionBuilder<'_>, expr: &hir::Expr) -> crate::ir::PortSource {
        match expr {
            hir::Expr::Literal(hir::Literal::Bool(value)) => rb.const_val(u64::from(*value)),
            hir::Expr::Literal(hir::Literal::Integer(value)) => rb.const_val(*value),
            hir::Expr::Local(local) => {
                let values = self.get_local_values(*local);
                assert_eq!(values.len(), 1, "lower_expr on multi-word local");
                values[0]
            }
            hir::Expr::Load { addr, width } => {
                let addr = self.lower_expr(rb, addr);
                rb.load_from_addr(addr, Self::ir_width(*width))
            }
            hir::Expr::SliceData { value } => {
                let values = self.lower_expr_multi(rb, value);
                assert!(values.len() >= 2, "SliceData requires 2-word type");
                values[0]
            }
            hir::Expr::SliceLen { value } => {
                let values = self.lower_expr_multi(rb, value);
                assert!(values.len() >= 2, "SliceLen requires 2-word type");
                values[1]
            }
            hir::Expr::Field { base, field } => {
                // Fast path: local-rooted field chain (most common case).
                if let Some((local, chain)) = Self::try_expr_field_chain(expr) {
                    let values = self.get_local_values(local);
                    let (offset, ty) = self.resolve_local_field(local, &chain);
                    assert_eq!(
                        Self::word_count_for_type(self.module, ty),
                        1,
                        "lower_expr on multi-word field"
                    );
                    return values[offset];
                }
                // Type-driven fallback: evaluate base to word vector, project field.
                let words = self.lower_expr_multi(rb, base);
                let base_ty = self.infer_expr_type(base);
                let (offset, count) = self.field_offset_in_type(&base_ty, field);
                assert_eq!(count, 1, "lower_expr on multi-word field projection");
                words[offset]
            }
            hir::Expr::Binary { op, lhs, rhs } => {
                let lhs = self.lower_expr(rb, lhs);
                let rhs = self.lower_expr(rb, rhs);
                let ir_op = match op {
                    hir::BinaryOp::Add => crate::ir::IrOp::Add,
                    hir::BinaryOp::Sub => crate::ir::IrOp::Sub,
                    hir::BinaryOp::Mul => crate::ir::IrOp::Mul,
                    hir::BinaryOp::BitAnd => crate::ir::IrOp::And,
                    hir::BinaryOp::BitOr => crate::ir::IrOp::Or,
                    hir::BinaryOp::Xor => crate::ir::IrOp::Xor,
                    hir::BinaryOp::Shl => crate::ir::IrOp::Shl,
                    hir::BinaryOp::Shr => crate::ir::IrOp::Shr,
                    hir::BinaryOp::Sar => crate::ir::IrOp::Sar,
                    hir::BinaryOp::Eq => crate::ir::IrOp::CmpEq,
                    hir::BinaryOp::Ne => crate::ir::IrOp::CmpNe,
                    hir::BinaryOp::Lt => crate::ir::IrOp::CmpLt,
                    hir::BinaryOp::Le => crate::ir::IrOp::CmpLe,
                    hir::BinaryOp::Gt => crate::ir::IrOp::CmpGt,
                    hir::BinaryOp::Ge => crate::ir::IrOp::CmpGe,
                    hir::BinaryOp::And => crate::ir::IrOp::And,
                    hir::BinaryOp::Or => crate::ir::IrOp::Or,
                    other => panic!("unsupported scalar HIR binary op: {other:?}"),
                };
                rb.binop(ir_op, lhs, rhs)
            }
            hir::Expr::Call(call) => self.lower_call_expr(rb, call),
            other => panic!("unsupported scalar HIR expression: {other:?}"),
        }
    }

    fn ir_width(width: hir::MemoryWidth) -> crate::ir::Width {
        match width {
            hir::MemoryWidth::W1 => crate::ir::Width::W1,
            hir::MemoryWidth::W2 => crate::ir::Width::W2,
            hir::MemoryWidth::W4 => crate::ir::Width::W4,
            hir::MemoryWidth::W8 => crate::ir::Width::W8,
        }
    }
}

fn build_structural_hir_ir_impl(module: &hir::Module) -> crate::ir::IrFunc {
    let (_, function) = module
        .functions
        .iter()
        .next()
        .expect("structural HIR module should contain one function");
    let dest_local = function
        .destination_param()
        .map(|param| param.local)
        .expect("structural HIR function should have a destination param");

    let dest_ty = &function
        .destination_param()
        .expect("structural HIR function should have a destination param")
        .ty;

    let label = &function.name;
    let output_size = structural_hir_type_size(module, dest_ty);
    let mut builder = crate::ir::IrBuilder::new(label, output_size);
    // Structural path always needs the memory domain — load_from_addr/store_to_addr thread on it.
    let _ = builder.add_state_domain(crate::ir::MEMORY_STATE_DOMAIN_NAME);
    {
        let mut rb = builder.root_region();
        let lowerer = StructuralHirIrLowerer::new(&mut rb, module, function);
        lowerer.lower_block(&mut rb, &function.body.statements, dest_local, dest_ty);
        rb.set_results(&[]);
    }
    builder.finish()
}

fn structural_hir_type_size(module: &hir::Module, ty: &hir::Type) -> usize {
    match ty {
        hir::Type::Unit => 0,
        hir::Type::Bool => 1,
        hir::Type::Integer(kind) => (kind.bits as usize) / 8,
        hir::Type::Address { .. } | hir::Type::Handle { .. } => core::mem::size_of::<usize>(),
        hir::Type::Str { .. } | hir::Type::Slice { .. } => core::mem::size_of::<usize>() * 2,
        hir::Type::Array { element, len } => structural_hir_type_size(module, element) * len,
        hir::Type::Named { def, .. } => {
            let type_def = &module.type_defs[*def];
            if let Some(size) = type_def.size {
                return size as usize;
            }
            structural_hir_type_def_size(module, type_def)
        }
    }
}

fn structural_hir_type_def_size(module: &hir::Module, type_def: &hir::TypeDef) -> usize {
    match &type_def.kind {
        hir::TypeDefKind::Struct { fields } => {
            let mut max_end = 0usize;
            for field in fields {
                let field_size = structural_hir_type_size(module, &field.ty);
                if let Some(offset) = field.offset {
                    max_end = max_end.max(offset as usize + field_size);
                } else {
                    max_end += field_size;
                }
            }
            max_end
        }
        hir::TypeDefKind::Enum {
            variants,
            discriminant_width,
        } => {
            let disc_size = discriminant_width.unwrap_or(1) as usize;
            let max_payload = variants
                .iter()
                .map(|variant| {
                    variant
                        .fields
                        .iter()
                        .map(|field| {
                            let field_size = structural_hir_type_size(module, &field.ty);
                            if let Some(offset) = field.offset {
                                offset as usize + field_size
                            } else {
                                field_size
                            }
                        })
                        .max()
                        .unwrap_or(0)
                })
                .max()
                .unwrap_or(0);
            disc_size + max_payload
        }
    }
}
