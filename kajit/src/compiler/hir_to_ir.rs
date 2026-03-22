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
    cursor_local: hir::LocalId,
    local_slots: std::collections::HashMap<hir::LocalId, StructuralLocalStorage>,
    local_types: std::collections::HashMap<hir::LocalId, &'a hir::Type>,
    cursor_bytes_ptr_slot: Option<crate::ir::SlotId>,
    _marker: std::marker::PhantomData<&'a hir::Module>,
}

enum ResolvedStructuralPlace<'a> {
    Destination {
        shape: &'static Shape,
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
        shape: &'static Shape,
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
        Self {
            module,
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
        for _ in 1..slot_count {
            let _ = rb.alloc_slot();
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
                hir::TypeDefKind::Enum { variants } => {
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
        dest_shape: &'static Shape,
    ) {
        for stmt in statements {
            self.lower_stmt(rb, stmt, dest_local, dest_shape);
        }
    }

    fn lower_stmt(
        &self,
        rb: &mut RegionBuilder<'_>,
        stmt: &hir::Stmt,
        dest_local: hir::LocalId,
        dest_shape: &'static Shape,
    ) {
        match &stmt.kind {
            hir::StmtKind::Init { place, value } | hir::StmtKind::Assign { place, value } => {
                self.lower_assign_like(rb, place, value, dest_local, dest_shape);
            }
            hir::StmtKind::Expr(expr) => self.lower_effect_expr(rb, expr, dest_local, dest_shape),
            hir::StmtKind::Fail { code } => rb.error_exit(*code),
            hir::StmtKind::Store { addr, width, value } => {
                let addr = self.lower_scalar_expr(rb, addr, dest_local, dest_shape);
                let value = self.lower_scalar_expr(rb, value, dest_local, dest_shape);
                rb.store_to_addr(addr, value, self.ir_width_for_memory_width(*width));
            }
            hir::StmtKind::If {
                condition,
                then_block,
                else_block,
            } => {
                let predicate = self.lower_scalar_expr(rb, condition, dest_local, dest_shape);
                let else_block = else_block
                    .as_ref()
                    .expect("structural HIR subset requires else");
                let _ = rb.gamma(predicate, &[], 2, |branch_idx, branch| {
                    match branch_idx {
                        0 => {
                            self.lower_block(branch, &else_block.statements, dest_local, dest_shape)
                        }
                        1 => {
                            self.lower_block(branch, &then_block.statements, dest_local, dest_shape)
                        }
                        _ => unreachable!(),
                    }
                    branch.set_results(&[]);
                });
            }
            hir::StmtKind::Match { scrutinee, arms } => {
                let predicate = self.lower_scalar_expr(rb, scrutinee, dest_local, dest_shape);
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
                        dest_shape,
                    );
                    branch.set_results(&[]);
                });
            }
            hir::StmtKind::Loop { body } => {
                let active_slot = rb.alloc_slot();
                let continue_slot = rb.alloc_slot();
                let _ = rb.theta(&[], |body_rb| {
                    let one = body_rb.const_val(1);
                    body_rb.write_to_slot(active_slot, one);
                    body_rb.write_to_slot(continue_slot, one);
                    self.lower_loop_block(
                        body_rb,
                        &body.statements,
                        dest_local,
                        dest_shape,
                        active_slot,
                        continue_slot,
                    );
                    let predicate = body_rb.read_from_slot(continue_slot);
                    body_rb.set_results(&[predicate]);
                });
            }
            hir::StmtKind::Return(None) => {}
            other => panic!("unsupported structural HIR statement: {other:?}"),
        }
    }

    fn lower_loop_block(
        &self,
        rb: &mut RegionBuilder<'_>,
        statements: &[hir::Stmt],
        dest_local: hir::LocalId,
        dest_shape: &'static Shape,
        active_slot: crate::ir::SlotId,
        continue_slot: crate::ir::SlotId,
    ) {
        for stmt in statements {
            self.lower_loop_stmt(rb, stmt, dest_local, dest_shape, active_slot, continue_slot);
        }
    }

    fn lower_loop_stmt(
        &self,
        rb: &mut RegionBuilder<'_>,
        stmt: &hir::Stmt,
        dest_local: hir::LocalId,
        dest_shape: &'static Shape,
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
                let predicate = self.lower_scalar_expr(guard_rb, condition, dest_local, dest_shape);
                let else_block = else_block
                    .as_ref()
                    .expect("structural HIR loop subset requires else");
                let _ = guard_rb.gamma(predicate, &[], 2, |branch_idx, branch| {
                    match branch_idx {
                        0 => self.lower_loop_block(
                            branch,
                            &else_block.statements,
                            dest_local,
                            dest_shape,
                            active_slot,
                            continue_slot,
                        ),
                        1 => self.lower_loop_block(
                            branch,
                            &then_block.statements,
                            dest_local,
                            dest_shape,
                            active_slot,
                            continue_slot,
                        ),
                        _ => unreachable!(),
                    }
                    branch.set_results(&[]);
                });
            }
            hir::StmtKind::Match { scrutinee, arms } => {
                let predicate = self.lower_scalar_expr(guard_rb, scrutinee, dest_local, dest_shape);
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
                        dest_shape,
                        active_slot,
                        continue_slot,
                    );
                    branch.set_results(&[]);
                });
            }
            hir::StmtKind::Loop { body } => {
                let nested_active_slot = guard_rb.alloc_slot();
                let nested_continue_slot = guard_rb.alloc_slot();
                let _ = guard_rb.theta(&[], |body_rb| {
                    let one = body_rb.const_val(1);
                    body_rb.write_to_slot(nested_active_slot, one);
                    body_rb.write_to_slot(nested_continue_slot, one);
                    self.lower_loop_block(
                        body_rb,
                        &body.statements,
                        dest_local,
                        dest_shape,
                        nested_active_slot,
                        nested_continue_slot,
                    );
                    let predicate = body_rb.read_from_slot(nested_continue_slot);
                    body_rb.set_results(&[predicate]);
                });
            }
            hir::StmtKind::Return(_) => {
                panic!("structural HIR loops do not support return in loop bodies yet");
            }
            _ => self.lower_stmt(guard_rb, stmt, dest_local, dest_shape),
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
        dest_shape: &'static Shape,
    ) {
        match expr {
            hir::Expr::Call(call) => self.lower_effect_call(rb, call, dest_local, dest_shape),
            other => panic!("unsupported structural HIR effect expression: {other:?}"),
        }
    }

    fn lower_assign_like(
        &self,
        rb: &mut RegionBuilder<'_>,
        place: &hir::Place,
        value: &hir::Expr,
        dest_local: hir::LocalId,
        dest_shape: &'static Shape,
    ) {
        self.lower_place_write(rb, place, value, dest_local, dest_shape);
    }

    fn lower_place_write(
        &self,
        rb: &mut RegionBuilder<'_>,
        place: &hir::Place,
        value: &hir::Expr,
        dest_local: hir::LocalId,
        dest_shape: &'static Shape,
    ) {
        if let hir::Place::Index { base, index } = place
            && !matches!(**index, hir::Expr::Literal(hir::Literal::Integer(_)))
        {
            if let hir::Expr::Local(local) = value
                && Self::slot_count_for_type(self.module, self.local_types[local]) > 1
            {
                self.lower_dynamic_index_write_from_local(
                    rb, base, index, *local, dest_local, dest_shape,
                );
            } else {
                self.lower_dynamic_index_write(rb, base, index, value, dest_local, dest_shape);
            }
            return;
        }
        let resolved = self.resolve_place(place, dest_local, dest_shape);
        match value {
            hir::Expr::Local(local) => match resolved {
                ResolvedStructuralPlace::Destination { shape, offset } => {
                    self.copy_local_into_shape_offset(rb, *local, shape, offset);
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
                        ResolvedStructuralPlace::Destination { shape, offset } => {
                            self.lower_vec_from_raw_parts_at_offset(
                                rb, call, shape, offset, dest_local, dest_shape,
                            );
                        }
                        ResolvedStructuralPlace::Local { .. } => {
                            panic!("local vec materialization is not supported yet");
                        }
                    }
                    return;
                }
                let scalar = self.lower_scalar_expr(rb, value, dest_local, dest_shape);
                match resolved {
                    ResolvedStructuralPlace::Destination { shape, offset } => {
                        let width = self.scalar_width_for_shape(shape);
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
                ResolvedStructuralPlace::Destination { shape, offset } => {
                    assert_eq!(
                        shape.scalar_type(),
                        Some(ScalarType::Str),
                        "str materialization requires a str destination, got {}",
                        shape.type_identifier
                    );
                    let data = self.lower_scalar_expr(rb, data, dest_local, dest_shape);
                    let len = self.lower_scalar_expr(rb, len, dest_local, dest_shape);
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
                    let data = self.lower_scalar_expr(rb, data, dest_local, dest_shape);
                    let len = self.lower_scalar_expr(rb, len, dest_local, dest_shape);
                    rb.write_to_slot(Self::slot_at(storage, slot_offset), data);
                    rb.write_to_slot(Self::slot_at(storage, slot_offset + 1), len);
                }
            },
            hir::Expr::Variant {
                variant, fields, ..
            } => match resolved {
                ResolvedStructuralPlace::Destination { shape, offset } => {
                    if let Some(opt_def) = get_option_def(shape) {
                        self.lower_option_variant_write(rb, offset, *opt_def, variant, fields);
                        return;
                    }
                    let Type::User(UserType::Enum(enum_type)) = &shape.ty else {
                        panic!("variant init must target an enum place");
                    };
                    let variant_info = collect_variants(enum_type)
                        .into_iter()
                        .find(|candidate| candidate.name == variant.as_str())
                        .unwrap_or_else(|| panic!("missing enum variant {variant}"));
                    let disc_width =
                        ir_width_from_disc_size(discriminant_size(enum_type.enum_repr));
                    let disc = variant_info
                        .rust_discriminant
                        .try_into()
                        .expect("enum discriminant must fit in u64");
                    let value = rb.const_val(disc);
                    rb.write_to_field(value, offset as u32, disc_width);
                    for field in &variant_info.fields {
                        let (_, expr) = fields
                            .iter()
                            .find(|(name, _)| name == field.name)
                            .unwrap_or_else(|| {
                                panic!(
                                    "missing enum payload field {} for variant {variant}",
                                    field.name
                                )
                            });
                        self.lower_value_into_shape_offset(
                            rb,
                            field.shape,
                            offset + field.offset,
                            expr,
                            dest_local,
                            dest_shape,
                        );
                    }
                }
                ResolvedStructuralPlace::Local { .. } => {
                    panic!("local enum writes are not supported yet");
                }
            },
            hir::Expr::Index { .. } => match resolved {
                ResolvedStructuralPlace::Destination { shape, offset } => {
                    self.lower_value_into_shape_offset(
                        rb, shape, offset, value, dest_local, dest_shape,
                    );
                }
                ResolvedStructuralPlace::Local {
                    ty,
                    storage,
                    slot_offset,
                } => {
                    let scalar = self.lower_scalar_expr(rb, value, dest_local, dest_shape);
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
                let scalar = self.lower_scalar_expr(rb, value, dest_local, dest_shape);
                match resolved {
                    ResolvedStructuralPlace::Destination { shape, offset } => {
                        let width = self.scalar_width_for_shape(shape);
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

    fn lower_option_variant_write(
        &self,
        rb: &mut RegionBuilder<'_>,
        offset: usize,
        opt_def: OptionDef,
        variant: &str,
        fields: &[(String, hir::Expr)],
    ) {
        let offset = offset as u32;
        match variant {
            "None" => {
                assert!(
                    fields.is_empty(),
                    "Option::None should not carry payload fields"
                );
                let init_fn = rb.const_val(opt_def.vtable.init_none as *const () as usize as u64);
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
                    hir::Expr::Local(local) => rb.slot_addr(self.local_slots[local].base_slot),
                    hir::Expr::Literal(hir::Literal::Unit) => {
                        let slot = rb.alloc_slot();
                        rb.slot_addr(slot)
                    }
                    hir::Expr::Literal(hir::Literal::Bool(value)) => {
                        let slot = rb.alloc_slot();
                        let value = rb.const_val(u64::from(*value));
                        rb.write_to_slot(slot, value);
                        rb.slot_addr(slot)
                    }
                    hir::Expr::Literal(hir::Literal::Integer(value)) => {
                        let slot = rb.alloc_slot();
                        let value = rb.const_val(*value);
                        rb.write_to_slot(slot, value);
                        rb.slot_addr(slot)
                    }
                    other => panic!("unsupported structural Option payload: {other:?}"),
                };
                let init_fn = rb.const_val(opt_def.vtable.init_some as *const () as usize as u64);
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

    fn lower_value_into_shape_offset(
        &self,
        rb: &mut RegionBuilder<'_>,
        shape: &'static Shape,
        offset: usize,
        expr: &hir::Expr,
        dest_local: hir::LocalId,
        dest_shape: &'static Shape,
    ) {
        match expr {
            hir::Expr::Call(call) if self.is_vec_from_raw_parts(call) => {
                self.lower_vec_from_raw_parts_at_offset(
                    rb, call, shape, offset, dest_local, dest_shape,
                );
            }
            hir::Expr::Call(_) => {
                let scalar = self.lower_scalar_expr(rb, expr, dest_local, dest_shape);
                let width = self.scalar_width_for_shape(shape);
                rb.write_to_field(scalar, offset as u32, width);
            }
            hir::Expr::Local(local) => self.copy_local_into_shape_offset(rb, *local, shape, offset),
            hir::Expr::Index { base, index } => {
                let base = self.expr_to_place(base);
                if matches!(&**index, hir::Expr::Literal(hir::Literal::Integer(_))) {
                    let place = hir::Place::Index {
                        base: Box::new(base),
                        index: index.clone(),
                    };
                    match self.resolve_place(&place, dest_local, dest_shape) {
                        ResolvedStructuralPlace::Destination {
                            shape: source_shape,
                            offset: source_offset,
                        } => {
                            self.copy_shape_bytes_to_shape_offset(
                                rb,
                                source_shape,
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
                        rb, &base, index, offset, dest_local, dest_shape,
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
                let width = self.scalar_width_for_shape(shape);
                rb.write_to_field(value, offset as u32, width);
            }
            hir::Expr::Str { data, len } => {
                assert_eq!(
                    shape.scalar_type(),
                    Some(ScalarType::Str),
                    "str materialization requires a str destination, got {}",
                    shape.type_identifier
                );
                let data = self.lower_scalar_expr(rb, data, dest_local, dest_shape);
                let len = self.lower_scalar_expr(rb, len, dest_local, dest_shape);
                rb.write_to_field(data, offset as u32, crate::ir::Width::W8);
                rb.write_to_field(len, (offset + 8) as u32, crate::ir::Width::W8);
            }
            hir::Expr::Variant {
                variant, fields, ..
            } => {
                if let Some(opt_def) = get_option_def(shape) {
                    self.lower_option_variant_write(rb, offset, *opt_def, variant, fields);
                } else {
                    panic!("nested non-Option variant payloads are not supported yet");
                }
            }
            other => panic!("unsupported structural payload expression: {other:?}"),
        }
    }

    fn copy_shape_bytes_to_shape_offset(
        &self,
        rb: &mut RegionBuilder<'_>,
        source_shape: &'static Shape,
        source_offset: usize,
        target_offset: usize,
    ) {
        let size = source_shape
            .layout
            .sized_layout()
            .expect("indexed destination element must be Sized")
            .size();
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

    fn copy_local_into_shape_offset(
        &self,
        rb: &mut RegionBuilder<'_>,
        local: hir::LocalId,
        shape: &'static Shape,
        offset: usize,
    ) {
        let storage = self.local_slots[&local];
        if let Some(st) = shape.scalar_type() {
            if !is_string_like_scalar(st) {
                let value = rb.read_from_slot(storage.base_slot);
                let width = self.scalar_width_for_shape(shape);
                rb.write_to_field(value, offset as u32, width);
                return;
            }
        }

        let size = shape
            .layout
            .sized_layout()
            .expect("structural local copy requires Sized layout")
            .size();
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
        shape: &'static Shape,
        offset: usize,
        dest_local: hir::LocalId,
        dest_shape: &'static Shape,
    ) {
        let Def::List(list_def) = &shape.def else {
            panic!("runtime.vec_from_raw_parts requires a list destination");
        };
        assert_eq!(
            call.args.len(),
            4,
            "runtime.vec_from_raw_parts expects ptr, len, cap, align"
        );
        let ptr = self.lower_scalar_expr(rb, &call.args[0], dest_local, dest_shape);
        let len = self.lower_scalar_expr(rb, &call.args[1], dest_local, dest_shape);
        let cap = self.lower_scalar_expr(rb, &call.args[2], dest_local, dest_shape);
        let align = self.lower_scalar_expr(rb, &call.args[3], dest_local, dest_shape);
        let offsets = crate::malum::discover_vec_offsets(list_def, shape);
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
            branch.write_to_field(ptr_value, (offset as u32) + offsets.ptr_offset, usize_width);
            branch.write_to_field(len, (offset as u32) + offsets.len_offset, usize_width);
            branch.write_to_field(cap, (offset as u32) + offsets.cap_offset, usize_width);
            branch.set_results(&[]);
        });
    }

    fn lower_scalar_expr(
        &self,
        rb: &mut RegionBuilder<'_>,
        expr: &hir::Expr,
        dest_local: hir::LocalId,
        dest_shape: &'static Shape,
    ) -> crate::ir::PortSource {
        match expr {
            hir::Expr::Literal(hir::Literal::Bool(value)) => rb.const_val(u64::from(*value)),
            hir::Expr::Literal(hir::Literal::Integer(value)) => rb.const_val(*value),
            hir::Expr::Local(local) => {
                let slot = self.local_slots[local].base_slot;
                rb.read_from_slot(slot)
            }
            hir::Expr::Load { addr, width } => {
                let addr = self.lower_scalar_expr(rb, addr, dest_local, dest_shape);
                rb.load_from_addr(addr, self.ir_width_for_memory_width(*width))
            }
            hir::Expr::SliceData { value } => {
                self.lower_view_component(rb, value, 0, dest_local, dest_shape)
            }
            hir::Expr::SliceLen { value } => {
                self.lower_view_component(rb, value, 1, dest_local, dest_shape)
            }
            hir::Expr::Field { .. } | hir::Expr::Index { .. } => {
                let place = self.expr_to_place(expr);
                if let hir::Place::Index { base, index } = &place
                    && !matches!(**index, hir::Expr::Literal(hir::Literal::Integer(_)))
                {
                    return self.lower_dynamic_index_read(rb, base, index, dest_local, dest_shape);
                }
                match self.resolve_place(&place, dest_local, dest_shape) {
                    ResolvedStructuralPlace::Destination { shape, offset } => {
                        let width = self.scalar_width_for_shape(shape);
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
                let lhs = self.lower_scalar_expr(rb, lhs, dest_local, dest_shape);
                let rhs = self.lower_scalar_expr(rb, rhs, dest_local, dest_shape);
                let ir_op = match op {
                    hir::BinaryOp::Add => crate::ir::IrOp::Add,
                    hir::BinaryOp::Sub => crate::ir::IrOp::Sub,
                    hir::BinaryOp::Mul => crate::ir::IrOp::Mul,
                    hir::BinaryOp::BitAnd => crate::ir::IrOp::And,
                    hir::BinaryOp::BitOr => crate::ir::IrOp::Or,
                    hir::BinaryOp::Xor => crate::ir::IrOp::Xor,
                    hir::BinaryOp::Shl => crate::ir::IrOp::Shl,
                    hir::BinaryOp::Shr => crate::ir::IrOp::Shr,
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
            hir::Expr::Call(call) => self.lower_scalar_call_expr(rb, call, dest_local, dest_shape),
            other => panic!("unsupported structural HIR scalar expression: {other:?}"),
        }
    }

    fn lower_scalar_call_expr(
        &self,
        rb: &mut RegionBuilder<'_>,
        call: &hir::CallExpr,
        dest_local: hir::LocalId,
        dest_shape: &'static Shape,
    ) -> crate::ir::PortSource {
        let args = call
            .args
            .iter()
            .map(|arg| self.lower_scalar_expr(rb, arg, dest_local, dest_shape))
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
        dest_shape: &'static Shape,
    ) {
        let args = call
            .args
            .iter()
            .map(|arg| self.lower_scalar_expr(rb, arg, dest_local, dest_shape))
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
        dest_shape: &'static Shape,
    ) -> crate::ir::PortSource {
        let place = self.expr_to_place(value);
        match self.resolve_place(&place, dest_local, dest_shape) {
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
            ResolvedStructuralPlace::Destination { shape, offset } => {
                let width = crate::ir::Width::W8;
                match shape.scalar_type() {
                    Some(ScalarType::Str) => {
                        rb.read_from_field((offset + word_index * 8) as u32, width)
                    }
                    _ => panic!(
                        "slice_data/slice_len require a slice-like destination, got {}",
                        shape.type_identifier
                    ),
                }
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
        dest_shape: &'static Shape,
    ) -> crate::ir::PortSource {
        let resolved = self.lower_dynamic_index_addr(rb, base, index, dest_local, dest_shape);
        match resolved {
            ResolvedDynamicIndex::Destination { shape, addr } => {
                let width = self.scalar_width_for_shape(shape);
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
        dest_shape: &'static Shape,
    ) {
        let resolved = self.lower_dynamic_index_addr(rb, base, index, dest_local, dest_shape);
        let value = self.lower_scalar_expr(rb, value, dest_local, dest_shape);
        match resolved {
            ResolvedDynamicIndex::Destination { shape, addr } => {
                let width = self.scalar_width_for_shape(shape);
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
        dest_shape: &'static Shape,
    ) {
        let resolved = self.lower_dynamic_index_addr(rb, base, index, dest_local, dest_shape);
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
        dest_shape: &'static Shape,
    ) {
        let resolved = self.lower_dynamic_index_addr(rb, base, index, dest_local, dest_shape);
        match resolved {
            ResolvedDynamicIndex::Destination { shape, addr } => {
                let size = shape
                    .layout
                    .sized_layout()
                    .expect("dynamic indexed destination element must be Sized")
                    .size();
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
        dest_shape: &'static Shape,
    ) -> ResolvedDynamicIndex<'a> {
        let index = self.lower_scalar_expr(rb, index, dest_local, dest_shape);
        match self.resolve_place(base, dest_local, dest_shape) {
            ResolvedStructuralPlace::Destination { shape, offset } => {
                let Def::Array(array_def) = &shape.def else {
                    panic!(
                        "dynamic indexed structural HIR place requires an array destination, got {}",
                        shape.type_identifier
                    );
                };
                let elem_layout = array_def
                    .t
                    .layout
                    .sized_layout()
                    .expect("array element must be Sized");
                let mut base_addr = rb.save_out_ptr();
                if offset != 0 {
                    let offset_val = rb.const_val(offset as u64);
                    base_addr = rb.binop(crate::ir::IrOp::Add, base_addr, offset_val);
                }
                let addr = self.add_scaled_index(rb, base_addr, index, elem_layout.size());
                ResolvedDynamicIndex::Destination {
                    shape: array_def.t,
                    addr,
                }
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
                let base_addr = rb.slot_addr(base_slot);
                let elem_slots = Self::slot_count_for_type(self.module, element);
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
        dest_shape: &'static Shape,
    ) -> ResolvedStructuralPlace<'a> {
        match place {
            hir::Place::Local(local) => {
                if *local == dest_local {
                    ResolvedStructuralPlace::Destination {
                        shape: dest_shape,
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
                match self.resolve_place(base, dest_local, dest_shape) {
                    ResolvedStructuralPlace::Destination { shape, offset } => {
                        let mut shape = shape;
                        let mut offset = offset;
                        while shape.is_transparent() {
                            let (fields, skipped) = collect_fields(shape);
                            assert!(
                                skipped.is_empty() && fields.len() == 1,
                                "structural HIR subset requires transparent wrappers to lower to one field"
                            );
                            let field_info = &fields[0];
                            shape = field_info.shape;
                            offset += field_info.offset;
                        }
                        if matches!(
                            shape.scalar_type(),
                            Some(ScalarType::U128 | ScalarType::I128)
                        ) {
                            let field_offset = match field.as_str() {
                                "lo" => 0,
                                "hi" => 8,
                                _ => panic!(
                                    "missing raw128 field {field} while lowering structural HIR place for {}",
                                    shape.type_identifier
                                ),
                            };
                            return ResolvedStructuralPlace::Destination {
                                shape: u64::SHAPE,
                                offset: offset + field_offset,
                            };
                        }
                        let (fields, skipped) = collect_fields(shape);
                        assert!(
                            skipped.is_empty(),
                            "structural HIR subset does not support skipped/defaulted fields"
                        );
                        let field_info = fields
                            .into_iter()
                            .find(|candidate| candidate.name == field.as_str())
                            .unwrap_or_else(|| {
                                panic!(
                                    "missing field {field} while lowering structural HIR place for {}",
                                    shape.type_identifier
                                )
                            });
                        ResolvedStructuralPlace::Destination {
                            shape: field_info.shape,
                            offset: offset + field_info.offset,
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
                match self.resolve_place(base, dest_local, dest_shape) {
                    ResolvedStructuralPlace::Destination { shape, offset } => {
                        let Def::Array(array_def) = &shape.def else {
                            panic!(
                                "indexed structural HIR place requires an array base, got {}",
                                shape.type_identifier
                            );
                        };
                        assert!(
                            index < array_def.n,
                            "array index {index} out of bounds for {}",
                            shape.type_identifier
                        );
                        let elem_layout = array_def
                            .t
                            .layout
                            .sized_layout()
                            .expect("array element must be Sized");
                        let stride = elem_layout.size();
                        ResolvedStructuralPlace::Destination {
                            shape: array_def.t,
                            offset: offset + index * stride,
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

    fn scalar_width_for_shape(&self, shape: &'static Shape) -> crate::ir::Width {
        match shape.scalar_type() {
            Some(ScalarType::Bool | ScalarType::U8 | ScalarType::I8) => crate::ir::Width::W1,
            Some(ScalarType::U16 | ScalarType::I16) => crate::ir::Width::W2,
            Some(ScalarType::U32 | ScalarType::I32 | ScalarType::F32 | ScalarType::Char) => {
                crate::ir::Width::W4
            }
            Some(
                ScalarType::U64
                | ScalarType::I64
                | ScalarType::USize
                | ScalarType::ISize
                | ScalarType::U128
                | ScalarType::I128
                | ScalarType::F64,
            ) => crate::ir::Width::W8,
            _ => panic!(
                "unsupported structural HIR scalar width for {}",
                shape.type_identifier
            ),
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
}

pub(crate) fn build_structural_hir_ir(
    shape: &'static Shape,
    module: &hir::Module,
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

    let mut builder = crate::ir::IrBuilder::new(shape);
    let _ = builder.add_state_domain(crate::ir::MEMORY_STATE_DOMAIN_NAME);
    {
        let mut rb = builder.root_region();
        let lowerer = StructuralHirIrLowerer::new(&mut rb, module, function);
        lowerer.lower_block(&mut rb, &function.body.statements, dest_local, shape);
        rb.set_results(&[]);
    }
    builder.finish()
}
