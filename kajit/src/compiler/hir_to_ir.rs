#![allow(dead_code)]
//! HIR→IR lowering — converts hir::Module into RVSDG (ir::IrFunc).

use super::*;

pub(crate) fn build_postcard_decoder_ir_via_hir(shape: &'static Shape) -> crate::ir::IrFunc {
    let module = build_postcard_decoder_hir(shape);
    build_structural_hir_ir(shape, &module)
}

#[derive(Clone, Copy)]
struct StructuralLocalStorage {
    base_slot: crate::ir::SlotId,
}

struct StructuralHirIrLowerer<'a> {
    module: &'a hir::Module,
    /// Cached Option vtable defs discovered from the Shape tree.
    /// Each entry pairs the OptionDef with its inner type's Shape for matching.
    option_defs: Vec<(&'static Shape, OptionDef)>,
    /// Cached Vec offsets discovered from the Shape tree, keyed by
    /// the type_identifier of the shape that contained the ListDef.
    vec_offsets: std::collections::HashMap<&'static str, crate::malum::VecOffsets>,
    cursor_local: hir::LocalId,
    local_slots: std::collections::HashMap<hir::LocalId, StructuralLocalStorage>,
    local_types: std::collections::HashMap<hir::LocalId, &'a hir::Type>,
    cursor_bytes_ptr_slot: Option<crate::ir::SlotId>,
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
        root_shape: &'static Shape,
    ) -> Self {
        Self::new_with_optional_shape(rb, module, function, Some(root_shape))
    }

    fn new_with_optional_shape(
        rb: &mut RegionBuilder<'_>,
        module: &'a hir::Module,
        function: &'a hir::Function,
        root_shape: Option<&'static Shape>,
    ) -> Self {
        let cursor_local = function
            .params
            .iter()
            .find(|param| param.kind == hir::LocalKind::Param)
            .map(|param| param.local)
            .expect("structural HIR function should have a cursor param");
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
        let (cursor_bytes_ptr_slot, _cursor_bytes_len_slot, _cursor_pos_slot) =
            Self::initialize_cursor_shadow(rb, module, cursor_local, &local_slots, &local_types);
        let mut option_defs = Vec::new();
        let mut vec_offsets = std::collections::HashMap::new();
        if let Some(shape) = root_shape {
            Self::collect_shape_defs(shape, &mut option_defs, &mut vec_offsets);
        }
        Self {
            module,
            option_defs,
            vec_offsets,
            cursor_local,
            local_slots,
            local_types,
            cursor_bytes_ptr_slot,
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

    fn initialize_cursor_shadow(
        rb: &mut RegionBuilder<'_>,
        module: &'a hir::Module,
        cursor_local: hir::LocalId,
        local_slots: &std::collections::HashMap<hir::LocalId, StructuralLocalStorage>,
        local_types: &std::collections::HashMap<hir::LocalId, &'a hir::Type>,
    ) -> (
        Option<crate::ir::SlotId>,
        Option<crate::ir::SlotId>,
        Option<crate::ir::SlotId>,
    ) {
        let Some(cursor_ty) = local_types.get(&cursor_local).copied() else {
            return (None, None, None);
        };
        let Some(storage) = local_slots.get(&cursor_local).copied() else {
            return (None, None, None);
        };
        let bytes_offset = match Self::struct_field_slot_offset(module, cursor_ty, "bytes") {
            Some(offset) => offset,
            None => return (None, None, None),
        };
        let pos_offset = match Self::struct_field_slot_offset(module, cursor_ty, "pos") {
            Some(offset) => offset,
            None => return (None, None, None),
        };

        let bytes_ptr_slot = Self::slot_at(storage, bytes_offset);
        let bytes_len_slot = Self::slot_at(storage, bytes_offset + 1);
        let pos_slot = Self::slot_at(storage, pos_offset);
        let base = rb.save_cursor();
        let end = rb.save_input_end();
        let len = rb.binop(crate::ir::IrOp::Sub, end, base);
        let zero = rb.const_val(0);
        rb.write_to_slot(bytes_ptr_slot, base);
        rb.write_to_slot(bytes_len_slot, len);
        rb.write_to_slot(pos_slot, zero);
        (Some(bytes_ptr_slot), Some(bytes_len_slot), Some(pos_slot))
    }

    fn struct_field_slot_offset(
        module: &'a hir::Module,
        ty: &hir::Type,
        field_name: &str,
    ) -> Option<usize> {
        let hir::Type::Named { def, .. } = ty else {
            return None;
        };
        let hir::TypeDefKind::Struct { fields } = &module.type_defs[*def].kind else {
            return None;
        };
        let mut slot_offset = 0usize;
        for field in fields {
            if field.name == field_name {
                return Some(slot_offset);
            }
            slot_offset += Self::slot_count_for_type(module, &field.ty);
        }
        None
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

    fn callable_name(&self, call: &hir::CallExpr) -> &str {
        match call.target {
            hir::CallTarget::Callable(callable) => &self.module.callables[callable].name,
        }
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
            hir::Expr::Call(call) => {
                if self.is_vec_from_raw_parts(call) {
                    match resolved {
                        ResolvedStructuralPlace::Destination { ty, offset } => {
                            self.lower_vec_from_raw_parts_at_offset(
                                rb, call, ty, offset, dest_local, dest_ty,
                            );
                        }
                        ResolvedStructuralPlace::Local { .. } => {
                            panic!("local vec materialization is not supported yet");
                        }
                    }
                    return;
                }
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
                        self.maybe_sync_cursor_position(rb, place, scalar);
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
                    // Check if this is an Option-like type (has None and Some variants)
                    if self.is_option_like_enum(hir_variants) {
                        self.lower_option_variant_write_hir(rb, offset, ty, variant, fields);
                        return;
                    }
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
                        self.maybe_sync_cursor_position(rb, place, scalar);
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

    fn maybe_sync_cursor_position(
        &self,
        rb: &mut RegionBuilder<'_>,
        place: &hir::Place,
        value: crate::ir::PortSource,
    ) {
        let Some(bytes_ptr_slot) = self.cursor_bytes_ptr_slot else {
            return;
        };
        if !self.is_cursor_pos_place(place) {
            return;
        }
        let base = rb.read_from_slot(bytes_ptr_slot);
        let absolute = rb.binop(crate::ir::IrOp::Add, base, value);
        rb.restore_cursor(absolute);
    }

    fn is_cursor_pos_place(&self, place: &hir::Place) -> bool {
        matches!(
            place,
            hir::Place::Field { base, field }
                if field == "pos" && matches!(&**base, hir::Place::Local(local) if *local == self.cursor_local)
        )
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
            hir::Expr::Call(call) if self.is_vec_from_raw_parts(call) => {
                self.lower_vec_from_raw_parts_at_offset(rb, call, ty, offset, dest_local, dest_ty);
            }
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

    fn is_vec_from_raw_parts(&self, call: &hir::CallExpr) -> bool {
        self.callable_name(call) == "runtime.vec_from_raw_parts"
    }

    fn lower_vec_from_raw_parts_at_offset(
        &self,
        rb: &mut RegionBuilder<'_>,
        call: &hir::CallExpr,
        ty: &hir::Type,
        offset: usize,
        dest_local: hir::LocalId,
        dest_ty: &'a hir::Type,
    ) {
        // Try to get ptr/len/cap offsets from the HIR type def
        let (ptr_offset, len_offset, cap_offset) = if let hir::Type::Named { def, .. } = ty {
            let type_def = &self.module.type_defs[*def];
            if let hir::TypeDefKind::Struct { fields } = &type_def.kind {
                let find =
                    |name: &str| -> Option<u32> { fields.iter().find(|f| f.name == name)?.offset };
                if let (Some(p), Some(l), Some(c)) = (find("ptr"), find("len"), find("cap")) {
                    (p, l, c)
                } else {
                    self.lookup_vec_offsets(&type_def.name)
                }
            } else {
                self.lookup_vec_offsets_any()
            }
        } else {
            // Not a named type — look up from Shape-based cache
            self.lookup_vec_offsets_any()
        };

        assert_eq!(
            call.args.len(),
            4,
            "runtime.vec_from_raw_parts expects ptr, len, cap, align"
        );
        let ptr = self.lower_scalar_expr(rb, &call.args[0], dest_local, dest_ty);
        let len = self.lower_scalar_expr(rb, &call.args[1], dest_local, dest_ty);
        let cap = self.lower_scalar_expr(rb, &call.args[2], dest_local, dest_ty);
        let align = self.lower_scalar_expr(rb, &call.args[3], dest_local, dest_ty);
        let usize_width = if core::mem::size_of::<usize>() == 8 {
            crate::ir::Width::W8
        } else {
            crate::ir::Width::W4
        };

        let zero = rb.const_val(0);
        let cap_nonzero = rb.binop(crate::ir::IrOp::CmpNe, cap, zero);
        rb.gamma(cap_nonzero, &[], 2, |branch_idx, branch| {
            let ptr_value = match branch_idx {
                0 => align,
                1 => ptr,
                _ => unreachable!(),
            };
            branch.write_to_field(ptr_value, (offset as u32) + ptr_offset, usize_width);
            branch.write_to_field(len, (offset as u32) + len_offset, usize_width);
            branch.write_to_field(cap, (offset as u32) + cap_offset, usize_width);
            branch.set_results(&[]);
        });
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
        let func = match self.callable_name(call) {
            "runtime.alloc_persistent" => {
                crate::ir::IntrinsicFn(intrinsics::kajit_alloc_persistent as *const () as usize)
            }
            "runtime.string_validate_alloc_copy" => crate::ir::IntrinsicFn(
                intrinsics::kajit_string_validate_alloc_copy as *const () as usize,
            ),
            other => panic!("unsupported structural HIR scalar call {other}"),
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
        let args = call
            .args
            .iter()
            .map(|arg| self.lower_scalar_expr(rb, arg, dest_local, dest_ty))
            .collect::<Vec<_>>();
        let func = match self.callable_name(call) {
            "runtime.validate_utf8_range" => {
                crate::ir::IntrinsicFn(intrinsics::kajit_validate_utf8_range as *const () as usize)
            }
            other => panic!("unsupported structural HIR effect call {other}"),
        };
        rb.call_intrinsic(func, &args, 0, false);
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
                        let hir::Type::Named { def, .. } = ty else {
                            panic!(
                                "local field place requires a named struct type, got {ty:?} for field {field}"
                            );
                        };
                        let hir::TypeDefKind::Struct { fields } = &self.module.type_defs[*def].kind
                        else {
                            panic!("local field place requires a struct type");
                        };
                        let mut running_slots = 0usize;
                        let field_info = fields
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
                            });
                        ResolvedStructuralPlace::Local {
                            ty: field_info.0,
                            storage,
                            slot_offset: slot_offset + field_info.1,
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

    /// Look up Vec offsets by matching a type name against cached Shape-based offsets.
    fn lookup_vec_offsets(&self, type_name: &str) -> (u32, u32, u32) {
        for (type_id, offsets) in &self.vec_offsets {
            if type_id.contains(type_name) {
                return (offsets.ptr_offset, offsets.len_offset, offsets.cap_offset);
            }
        }
        // Use the first available vec offsets (there's usually only one Vec type)
        self.lookup_vec_offsets_any()
    }

    /// Return any cached Vec offsets (for when the HIR type doesn't identify the Vec).
    fn lookup_vec_offsets_any(&self) -> (u32, u32, u32) {
        if let Some(offsets) = self.vec_offsets.values().next() {
            (offsets.ptr_offset, offsets.len_offset, offsets.cap_offset)
        } else {
            // Ultimate fallback if no Vec shapes were found
            (0, 8, 16)
        }
    }

    /// Check if an HIR type matches a Shape by comparing structural properties.
    fn hir_type_matches_shape(&self, ty: &hir::Type, shape: &'static Shape) -> bool {
        match ty {
            hir::Type::Bool => shape.scalar_type() == Some(ScalarType::Bool),
            hir::Type::Unit => shape.scalar_type() == Some(ScalarType::Unit),
            hir::Type::Integer(kind) => match (kind.bits, kind.signedness) {
                (8, hir::Signedness::Unsigned) => shape.scalar_type() == Some(ScalarType::U8),
                (8, hir::Signedness::Signed) => shape.scalar_type() == Some(ScalarType::I8),
                (16, hir::Signedness::Unsigned) => shape.scalar_type() == Some(ScalarType::U16),
                (16, hir::Signedness::Signed) => shape.scalar_type() == Some(ScalarType::I16),
                (32, hir::Signedness::Unsigned) => {
                    matches!(
                        shape.scalar_type(),
                        Some(ScalarType::U32 | ScalarType::Char | ScalarType::F32)
                    )
                }
                (32, hir::Signedness::Signed) => shape.scalar_type() == Some(ScalarType::I32),
                (64, hir::Signedness::Unsigned) => {
                    matches!(
                        shape.scalar_type(),
                        Some(ScalarType::U64 | ScalarType::USize | ScalarType::F64)
                    )
                }
                (64, hir::Signedness::Signed) => {
                    matches!(
                        shape.scalar_type(),
                        Some(ScalarType::I64 | ScalarType::ISize)
                    )
                }
                _ => false,
            },
            hir::Type::Str { .. } => matches!(
                shape.scalar_type(),
                Some(ScalarType::Str | ScalarType::CowStr)
            ),
            hir::Type::Named { def, .. } => {
                let type_def = &self.module.type_defs[*def];
                // Direct name match
                if shape.type_identifier == type_def.name {
                    return true;
                }
                // Special case: HostStringRaw maps to String scalar type
                if type_def.name == "HostStringRaw" {
                    return shape.scalar_type() == Some(ScalarType::String);
                }
                // Special case: Bits128Raw maps to U128/I128
                if type_def.name == "Bits128Raw" {
                    return matches!(
                        shape.scalar_type(),
                        Some(ScalarType::U128 | ScalarType::I128)
                    );
                }
                // Match by comparing the byte sizes
                if let Some(size) = type_def.size {
                    if let Ok(layout) = shape.layout.sized_layout() {
                        return layout.size() == size as usize;
                    }
                }
                false
            }
            _ => false,
        }
    }

    /// Check if an enum's variants match the Option pattern (None + Some).
    fn is_option_like_enum(&self, variants: &[hir::VariantDef]) -> bool {
        variants.len() == 2
            && variants.iter().any(|v| v.name == "None")
            && variants.iter().any(|v| v.name == "Some")
    }

    /// Recursively walk a Shape tree collecting OptionDef and VecOffsets entries.
    fn collect_shape_defs(
        shape: &'static Shape,
        option_map: &mut Vec<(&'static Shape, OptionDef)>,
        vec_map: &mut std::collections::HashMap<&'static str, crate::malum::VecOffsets>,
    ) {
        if let Some(opt_def) = get_option_def(shape) {
            option_map.push((shape, *opt_def));
        }
        // Recurse into sub-shapes
        match &shape.def {
            Def::Scalar => {}
            Def::List(list_def) => {
                vec_map
                    .entry(shape.type_identifier)
                    .or_insert_with(|| crate::malum::discover_vec_offsets(list_def, shape));
                Self::collect_shape_defs(list_def.t, option_map, vec_map);
            }
            Def::Array(array_def) => {
                Self::collect_shape_defs(array_def.t, option_map, vec_map);
            }
            Def::Option(opt_def) => {
                Self::collect_shape_defs(opt_def.t, option_map, vec_map);
            }
            _ => {
                if let Type::User(UserType::Struct(struct_type)) = &shape.ty {
                    for field in struct_type.fields {
                        Self::collect_shape_defs(field.shape.get(), option_map, vec_map);
                    }
                } else if let Type::User(UserType::Enum(enum_type)) = &shape.ty {
                    for variant in enum_type.variants {
                        for field in variant.data.fields {
                            Self::collect_shape_defs(field.shape.get(), option_map, vec_map);
                        }
                    }
                }
            }
        }
    }

    /// Look up the OptionDef for an HIR type by matching the inner payload type
    /// against the collected Shape-based option defs.
    fn find_option_def(&self, ty: &hir::Type) -> Option<OptionDef> {
        let hir::Type::Named { def, .. } = ty else {
            return None;
        };
        let type_def = &self.module.type_defs[*def];
        let hir::TypeDefKind::Enum { variants, .. } = &type_def.kind else {
            return None;
        };
        // Get the "Some" variant's payload type to match against Shape inner types
        let some_variant = variants.iter().find(|v| v.name == "Some")?;
        let payload_ty = some_variant.fields.first().map(|f| &f.ty);

        // If there's only one option def, use it
        if self.option_defs.len() == 1 {
            return Some(self.option_defs[0].1);
        }

        // Match by comparing the inner type's Shape type_identifier
        // against the HIR payload type
        if let Some(payload_ty) = payload_ty {
            for (_shape, opt_def) in &self.option_defs {
                if self.hir_type_matches_shape(payload_ty, opt_def.t) {
                    return Some(*opt_def);
                }
            }
        } else {
            // Unit payload (Option<()>) — match by shape name
            for (_shape, opt_def) in &self.option_defs {
                if opt_def.t.type_identifier == "()" {
                    return Some(*opt_def);
                }
            }
        }

        // Last resort: use the first one
        self.option_defs.first().map(|(_, def)| *def)
    }

    /// Lower an Option-like variant write using init_fn from HIR variant annotations.
    fn lower_option_variant_write_hir(
        &self,
        rb: &mut RegionBuilder<'_>,
        offset: usize,
        ty: &hir::Type,
        variant: &str,
        fields: &[(String, hir::Expr)],
    ) {
        let hir::Type::Named { def, .. } = ty else {
            panic!("Option variant write requires a Named type, got {ty:?}");
        };
        let type_def = &self.module.type_defs[*def];
        let hir::TypeDefKind::Enum { variants, .. } = &type_def.kind else {
            panic!("Option variant write requires an Enum type def");
        };
        let variant_def = variants
            .iter()
            .find(|v| v.name == variant)
            .unwrap_or_else(|| panic!("missing Option variant {variant}"));

        // Try to get init_fn from the HIR annotation first, fall back to Shape cache.
        let init_fn_ptr = if let Some(ptr) = variant_def.init_fn {
            ptr
        } else {
            // Fallback: look up from cached OptionDef (Shape-based).
            let opt_def = self.find_option_def(ty).unwrap_or_else(|| {
                panic!(
                    "Option variant '{variant}' on '{}' has no init_fn annotation \
                     and no OptionDef found in Shape cache",
                    type_def.name
                )
            });
            match variant {
                "None" => opt_def.vtable.init_none as *const () as usize as u64,
                "Some" => opt_def.vtable.init_some as *const () as usize as u64,
                _ => panic!("unsupported Option variant {variant}"),
            }
        };

        let offset = offset as u32;
        match variant {
            "None" => {
                assert!(
                    fields.is_empty(),
                    "Option::None should not carry payload fields"
                );
                let init_fn = rb.const_val(init_fn_ptr);
                rb.call_intrinsic(
                    crate::ir::IntrinsicFn(
                        intrinsics::kajit_option_init_none_ctx as *const () as usize,
                    ),
                    &[init_fn],
                    offset,
                    false,
                );
            }
            "Some" => {
                assert_eq!(
                    fields.len(),
                    1,
                    "Option::Some should carry exactly one payload field"
                );
                let payload_ptr = match &fields[0].1 {
                    hir::Expr::Local(local) => {
                        let num_slots =
                            Self::slot_count_for_type(self.module, self.local_types[local]) as u32;
                        rb.slot_addr(self.local_slots[local].base_slot, num_slots)
                    }
                    hir::Expr::Literal(hir::Literal::Unit) => {
                        let slot = rb.alloc_slot();
                        rb.slot_addr(slot, 1)
                    }
                    hir::Expr::Literal(hir::Literal::Bool(value)) => {
                        let slot = rb.alloc_slot();
                        let value = rb.const_val(u64::from(*value));
                        rb.write_to_slot(slot, value);
                        rb.slot_addr(slot, 1)
                    }
                    hir::Expr::Literal(hir::Literal::Integer(value)) => {
                        let slot = rb.alloc_slot();
                        let value = rb.const_val(*value);
                        rb.write_to_slot(slot, value);
                        rb.slot_addr(slot, 1)
                    }
                    other => panic!("unsupported structural Option payload: {other:?}"),
                };
                let init_fn = rb.const_val(init_fn_ptr);
                rb.call_intrinsic(
                    crate::ir::IntrinsicFn(
                        intrinsics::kajit_option_init_some_ctx as *const () as usize,
                    ),
                    &[init_fn, payload_ptr],
                    offset,
                    false,
                );
            }
            other => panic!("unsupported structural Option variant {other}"),
        }
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
        if self.is_option_like_enum(variants) {
            self.lower_option_variant_write_hir(rb, offset, ty, variant, fields);
            return;
        }
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

pub(crate) fn build_structural_hir_ir(
    shape: &'static Shape,
    module: &hir::Module,
) -> crate::ir::IrFunc {
    build_structural_hir_ir_impl(module, Some(shape))
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
        build_structural_hir_ir_impl(module, None)
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
fn build_scalar_hir_ir(module: &hir::Module, function: &hir::Function) -> crate::ir::IrFunc {
    let mut builder = crate::ir::IrBuilder::new(&function.name, 0);
    {
        let mut rb = builder.root_region();
        let lowerer = ScalarHirIrLowerer::new(&mut rb, module, function);
        let ret = lowerer.lower_block(&mut rb, &function.body.statements);
        if let Some(ret_val) = ret {
            rb.set_results(&[ret_val]);
        } else {
            rb.set_results(&[]);
        }
    }
    builder.finish()
}

struct ScalarHirIrLowerer<'a> {
    module: &'a hir::Module,
    /// Base slot for each local (params and locals alike). Multi-field types
    /// occupy consecutive slots starting at the base.
    local_base_slots: std::collections::HashMap<hir::LocalId, crate::ir::SlotId>,
    /// Type of each local, for field resolution.
    local_types: std::collections::HashMap<hir::LocalId, &'a hir::Type>,
}

/// A resolved place in the scalar lowerer — always a local with a slot offset.
struct ResolvedScalarPlace<'a> {
    ty: &'a hir::Type,
    base_slot: crate::ir::SlotId,
    slot_offset: usize,
}

impl<'a> ScalarHirIrLowerer<'a> {
    fn new(
        rb: &mut RegionBuilder<'_>,
        module: &'a hir::Module,
        function: &'a hir::Function,
    ) -> Self {
        let mut local_base_slots = std::collections::HashMap::new();
        let mut local_types = std::collections::HashMap::new();
        for param in &function.params {
            let base_slot = Self::alloc_local(rb, module, &param.ty);
            local_base_slots.insert(param.local, base_slot);
            local_types.insert(param.local, &param.ty);
        }
        for local in &function.locals {
            let base_slot = Self::alloc_local(rb, module, &local.ty);
            local_base_slots.insert(local.local, base_slot);
            local_types.insert(local.local, &local.ty);
        }
        Self {
            module,
            local_base_slots,
            local_types,
        }
    }

    fn alloc_local(
        rb: &mut RegionBuilder<'_>,
        module: &hir::Module,
        ty: &hir::Type,
    ) -> crate::ir::SlotId {
        let slot_count = Self::slot_count_for_type(module, ty);
        let base_slot = rb.alloc_slot();
        if slot_count > 1 {
            rb.func().multi_slot_group.insert(base_slot);
            for _ in 1..slot_count {
                let sub_slot = rb.alloc_slot();
                rb.func().multi_slot_group.insert(sub_slot);
            }
        } else {
            rb.func().scalar_temp_slots.insert(base_slot);
        }
        base_slot
    }

    fn slot_count_for_type(module: &hir::Module, ty: &hir::Type) -> usize {
        match ty {
            hir::Type::Unit
            | hir::Type::Bool
            | hir::Type::Integer(_)
            | hir::Type::Address { .. }
            | hir::Type::Handle { .. } => 1,
            hir::Type::Str { .. } | hir::Type::Slice { .. } => 2,
            hir::Type::Array { element, len } => Self::slot_count_for_type(module, element)
                .saturating_mul(*len)
                .max(1),
            hir::Type::Named { def, .. } => match &module.type_defs[*def].kind {
                hir::TypeDefKind::Struct { fields } => fields
                    .iter()
                    .map(|f| Self::slot_count_for_type(module, &f.ty))
                    .sum::<usize>()
                    .max(1),
                hir::TypeDefKind::Enum { variants, .. } => {
                    let payload_slots = variants
                        .iter()
                        .map(|v| {
                            v.fields
                                .iter()
                                .map(|f| Self::slot_count_for_type(module, &f.ty))
                                .sum::<usize>()
                        })
                        .max()
                        .unwrap_or(0);
                    (1 + payload_slots).max(1)
                }
            },
        }
    }

    fn slot_at(base: crate::ir::SlotId, offset: usize) -> crate::ir::SlotId {
        crate::ir::SlotId::new(base.index() as u32 + offset as u32)
    }

    /// If the outermost place is a dynamic (non-literal) index, return the
    /// base place and index expression. Otherwise return None.
    fn dynamic_index_place<'p>(place: &'p hir::Place) -> Option<(&'p hir::Place, &'p hir::Expr)> {
        if let hir::Place::Index { base, index } = place {
            if !matches!(**index, hir::Expr::Literal(hir::Literal::Integer(_))) {
                return Some((base, index));
            }
        }
        None
    }

    /// Compute the runtime address for a dynamic array index.
    /// Returns (element_type, address).
    fn lower_dynamic_index_addr(
        &self,
        rb: &mut RegionBuilder<'_>,
        base_place: &hir::Place,
        index: &hir::Expr,
    ) -> (&'a hir::Type, crate::ir::PortSource) {
        let resolved = self.resolve_place(base_place);
        let hir::Type::Array { element, .. } = resolved.ty else {
            panic!("dynamic index requires array type, got {:?}", resolved.ty);
        };
        let index = self.lower_expr(rb, index);
        let elem_slots = Self::slot_count_for_type(self.module, element);
        let total_slots = Self::slot_count_for_type(self.module, resolved.ty);
        let base_addr = rb.slot_addr(
            Self::slot_at(resolved.base_slot, resolved.slot_offset),
            total_slots as u32,
        );
        let stride = elem_slots * crate::ir::SLOT_ADDR_STRIDE_BYTES;
        let addr = if stride == 1 {
            rb.binop(crate::ir::IrOp::Add, base_addr, index)
        } else {
            let stride_val = rb.const_val(stride as u64);
            let scaled = rb.binop(crate::ir::IrOp::Mul, index, stride_val);
            rb.binop(crate::ir::IrOp::Add, base_addr, scaled)
        };
        (element, addr)
    }

    fn scalar_width_for_hir_type(ty: &hir::Type) -> crate::ir::Width {
        match ty {
            hir::Type::Bool => crate::ir::Width::W1,
            hir::Type::Integer(kind) => match kind.bits {
                8 => crate::ir::Width::W1,
                16 => crate::ir::Width::W2,
                32 => crate::ir::Width::W4,
                64 => crate::ir::Width::W8,
                other => panic!("unsupported integer width: {other}"),
            },
            hir::Type::Address { .. } | hir::Type::Handle { .. } => crate::ir::Width::W8,
            _ => panic!("unsupported scalar type for width: {ty:?}"),
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

    fn resolve_place(&self, place: &hir::Place) -> ResolvedScalarPlace<'a> {
        match place {
            hir::Place::Local(local) => ResolvedScalarPlace {
                ty: self.local_types[local],
                base_slot: self.local_base_slots[local],
                slot_offset: 0,
            },
            hir::Place::Field { base, field } => {
                let parent = self.resolve_place(base);
                let hir::Type::Named { def, .. } = parent.ty else {
                    panic!(
                        "field access requires a named struct type, got {:?}",
                        parent.ty
                    );
                };
                let hir::TypeDefKind::Struct { fields } = &self.module.type_defs[*def].kind else {
                    panic!("field access requires a struct type def");
                };
                let mut running_slots = 0usize;
                let (field_ty, field_slot_offset) = fields
                    .iter()
                    .find_map(|candidate| {
                        let found = (candidate.name == field.as_str())
                            .then_some((&candidate.ty, running_slots));
                        running_slots += Self::slot_count_for_type(self.module, &candidate.ty);
                        found
                    })
                    .unwrap_or_else(|| panic!("missing struct field {field}"));
                ResolvedScalarPlace {
                    ty: field_ty,
                    base_slot: parent.base_slot,
                    slot_offset: parent.slot_offset + field_slot_offset,
                }
            }
            hir::Place::Index { base, index } => {
                let hir::Expr::Literal(hir::Literal::Integer(index)) = &**index else {
                    panic!("array index must be an integer literal");
                };
                let index = *index as usize;
                let parent = self.resolve_place(base);
                let hir::Type::Array { element, len } = parent.ty else {
                    panic!("index access requires an array type, got {:?}", parent.ty);
                };
                assert!(
                    index < *len,
                    "array index {index} out of bounds for len {len}"
                );
                let elem_slots = Self::slot_count_for_type(self.module, element);
                ResolvedScalarPlace {
                    ty: element,
                    base_slot: parent.base_slot,
                    slot_offset: parent.slot_offset + index * elem_slots,
                }
            }
        }
    }

    fn lower_block(
        &self,
        rb: &mut RegionBuilder<'_>,
        statements: &[hir::Stmt],
    ) -> Option<crate::ir::PortSource> {
        for stmt in statements {
            if let Some(ret) = self.lower_stmt(rb, stmt) {
                return Some(ret);
            }
        }
        None
    }

    fn lower_stmt(
        &self,
        rb: &mut RegionBuilder<'_>,
        stmt: &hir::Stmt,
    ) -> Option<crate::ir::PortSource> {
        match &stmt.kind {
            hir::StmtKind::Init { place, value } | hir::StmtKind::Assign { place, value } => {
                // Check for dynamic index write (non-literal index in the place).
                if let Some((base, index)) = Self::dynamic_index_place(place) {
                    let val = self.lower_expr(rb, value);
                    let (elem_ty, addr) = self.lower_dynamic_index_addr(rb, base, index);
                    let width = Self::scalar_width_for_hir_type(elem_ty);
                    rb.store_to_addr(addr, val, width);
                    return None;
                }
                let resolved = self.resolve_place(place);
                let slot_count = Self::slot_count_for_type(self.module, resolved.ty);
                if slot_count == 1 {
                    let val = self.lower_expr(rb, value);
                    rb.write_to_slot(Self::slot_at(resolved.base_slot, resolved.slot_offset), val);
                } else {
                    // Multi-slot assignment: the value must also be a place (local/field)
                    // so we copy slot-by-slot.
                    let source = self.expr_to_place(value);
                    let src_resolved = self.resolve_place(&source);
                    for i in 0..slot_count {
                        let val = rb.read_from_slot(Self::slot_at(
                            src_resolved.base_slot,
                            src_resolved.slot_offset + i,
                        ));
                        rb.write_to_slot(
                            Self::slot_at(resolved.base_slot, resolved.slot_offset + i),
                            val,
                        );
                    }
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
                let func = match callable.name.as_str() {
                    "runtime.validate_utf8_range" => crate::ir::IntrinsicFn(
                        intrinsics::kajit_validate_utf8_range as *const () as usize,
                    ),
                    name => panic!("unsupported scalar HIR effect call: {name}"),
                };
                rb.call_intrinsic(func, &args, 0, false);
                None
            }
            hir::StmtKind::Expr(_) => None,
            hir::StmtKind::Store { addr, width, value } => {
                let addr = self.lower_expr(rb, addr);
                let value = self.lower_expr(rb, value);
                rb.store_to_addr(addr, value, Self::ir_width(*width));
                None
            }
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
                let _ = rb.gamma(predicate, &[], 2, |branch_idx, branch| {
                    match branch_idx {
                        0 => {
                            self.lower_block(branch, &else_block.statements);
                        }
                        1 => {
                            self.lower_block(branch, &then_block.statements);
                        }
                        _ => unreachable!(),
                    }
                    branch.set_results(&[]);
                });
                None
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
                let _ = rb.gamma(predicate, &[], arms.len(), |branch_idx, branch| {
                    self.lower_block(branch, &arms[branch_idx].body.statements);
                    branch.set_results(&[]);
                });
                None
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
            hir::StmtKind::Return(Some(expr)) => Some(self.lower_expr(rb, expr)),
            hir::StmtKind::Return(None) => None,
            other => panic!("unsupported scalar HIR statement: {other:?}"),
        }
    }

    fn lower_loop_block(
        &self,
        rb: &mut RegionBuilder<'_>,
        statements: &[hir::Stmt],
        active_slot: crate::ir::SlotId,
        continue_slot: crate::ir::SlotId,
    ) {
        let mut i = 0;
        while i < statements.len() {
            // Batch consecutive non-control-flow statements into a single guard.
            let start = i;
            while i < statements.len() && !Self::is_loop_control_flow(&statements[i]) {
                i += 1;
            }
            if start < i {
                self.with_active_guard(rb, active_slot, |guard_rb| {
                    for stmt in &statements[start..i] {
                        self.lower_stmt(guard_rb, stmt);
                    }
                });
            }
            // Lower the control-flow statement in its own guard.
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
        &self,
        rb: &mut RegionBuilder<'_>,
        stmt: &hir::Stmt,
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
                let predicate = self.lower_expr(guard_rb, condition);
                let else_block = else_block
                    .as_ref()
                    .expect("scalar HIR loop if requires else branch");
                let _ = guard_rb.gamma(predicate, &[], 2, |branch_idx, branch| {
                    match branch_idx {
                        0 => self.lower_loop_block(
                            branch,
                            &else_block.statements,
                            active_slot,
                            continue_slot,
                        ),
                        1 => self.lower_loop_block(
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
                let predicate = self.lower_expr(guard_rb, scrutinee);
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
                    self.lower_loop_block(
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
                    self.lower_loop_block(
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

    fn expr_to_place(&self, expr: &hir::Expr) -> hir::Place {
        match expr {
            hir::Expr::Local(local) => hir::Place::Local(*local),
            hir::Expr::Field { base, field } => hir::Place::Field {
                base: Box::new(self.expr_to_place(base)),
                field: field.clone(),
            },
            hir::Expr::Index { base, index } => hir::Place::Index {
                base: Box::new(self.expr_to_place(base)),
                index: Box::new(*index.clone()),
            },
            other => panic!("cannot convert expression to place: {other:?}"),
        }
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
        let func = match callable.name.as_str() {
            "runtime.alloc_persistent" => {
                crate::ir::IntrinsicFn(intrinsics::kajit_alloc_persistent as *const () as usize)
            }
            "runtime.string_validate_alloc_copy" => crate::ir::IntrinsicFn(
                intrinsics::kajit_string_validate_alloc_copy as *const () as usize,
            ),
            name => panic!("unsupported scalar HIR call target: {name}"),
        };
        rb.call_intrinsic(func, &args, 0, true)
            .expect("scalar call should return a value")
    }

    fn lower_expr(&self, rb: &mut RegionBuilder<'_>, expr: &hir::Expr) -> crate::ir::PortSource {
        match expr {
            hir::Expr::Literal(hir::Literal::Bool(value)) => rb.const_val(u64::from(*value)),
            hir::Expr::Literal(hir::Literal::Integer(value)) => rb.const_val(*value),
            hir::Expr::Local(local) => {
                let base_slot = self.local_base_slots[local];
                rb.read_from_slot(base_slot)
            }
            hir::Expr::Load { addr, width } => {
                let addr = self.lower_expr(rb, addr);
                rb.load_from_addr(addr, Self::ir_width(*width))
            }
            hir::Expr::SliceData { value } => {
                let place = self.expr_to_place(value);
                let resolved = self.resolve_place(&place);
                rb.read_from_slot(Self::slot_at(resolved.base_slot, resolved.slot_offset))
            }
            hir::Expr::SliceLen { value } => {
                let place = self.expr_to_place(value);
                let resolved = self.resolve_place(&place);
                rb.read_from_slot(Self::slot_at(resolved.base_slot, resolved.slot_offset + 1))
            }
            hir::Expr::Field { .. } => {
                let place = self.expr_to_place(expr);
                let resolved = self.resolve_place(&place);
                assert_eq!(
                    Self::slot_count_for_type(self.module, resolved.ty),
                    1,
                    "scalar read requires single-slot type"
                );
                rb.read_from_slot(Self::slot_at(resolved.base_slot, resolved.slot_offset))
            }
            hir::Expr::Index { base, index } => {
                if matches!(**index, hir::Expr::Literal(hir::Literal::Integer(_))) {
                    // Static index — resolve to a slot directly.
                    let place = self.expr_to_place(expr);
                    let resolved = self.resolve_place(&place);
                    assert_eq!(
                        Self::slot_count_for_type(self.module, resolved.ty),
                        1,
                        "scalar read requires single-slot type"
                    );
                    rb.read_from_slot(Self::slot_at(resolved.base_slot, resolved.slot_offset))
                } else {
                    // Dynamic index — compute address at runtime.
                    let base_place = self.expr_to_place(base);
                    let (elem_ty, addr) = self.lower_dynamic_index_addr(rb, &base_place, index);
                    let width = Self::scalar_width_for_hir_type(elem_ty);
                    rb.load_from_addr(addr, width)
                }
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
}

fn build_structural_hir_ir_impl(
    module: &hir::Module,
    shape: Option<&'static Shape>,
) -> crate::ir::IrFunc {
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

    let label = shape.map(|s| s.type_identifier).unwrap_or(&function.name);
    let output_size = shape
        .and_then(|s| s.layout.sized_layout().ok())
        .map(|l| l.size())
        .unwrap_or(0);
    let mut builder = crate::ir::IrBuilder::new(label, output_size);
    let _ = builder.add_state_domain(crate::ir::MEMORY_STATE_DOMAIN_NAME);
    {
        let mut rb = builder.root_region();
        let lowerer =
            StructuralHirIrLowerer::new_with_optional_shape(&mut rb, module, function, shape);
        lowerer.lower_block(&mut rb, &function.body.statements, dest_local, dest_ty);
        rb.set_results(&[]);
    }
    builder.finish()
}
