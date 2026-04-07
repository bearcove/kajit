//! Postcard HIR generation — builds `hir::Module` from facet `Shape` for postcard format.

use std::collections::HashMap;

use facet::{Def, EnumRepr, ListDef, ScalarType, Shape, Type, UserType};
use kajit_format::{
    FieldEmitInfo, SkippedFieldInfo, VtableEntry, collect_variants, discriminant_size,
    get_option_def, get_pointer_def, is_unit, vtable_symbol_name,
};
use kajit_hir as hir;

/// Wrapper around `kajit_format::collect_fields` using no-op default resolvers.
///
/// Postcard HIR generation asserts that no fields have defaults or are skipped,
/// so the resolvers are never actually invoked.
fn collect_fields(shape: &'static Shape) -> (Vec<FieldEmitInfo>, Vec<SkippedFieldInfo>) {
    kajit_format::collect_fields(
        shape,
        |_| None,
        |_| panic!("postcard HIR does not support custom defaults"),
    )
}

pub struct PostcardHirLowerer {
    module: hir::Module,
    input_region: hir::RegionId,
    cursor_type: hir::TypeDefId,
    string_raw_type: Option<hir::TypeDefId>,
    bits128_raw_type: Option<hir::TypeDefId>,
    type_defs_by_shape: HashMap<*const Shape, hir::TypeDefId>,
    callables_by_name: HashMap<&'static str, hir::CallableId>,
    pub locals: Vec<hir::LocalDecl>,
    next_local: u32,
    next_stmt: u32,
}

impl Default for PostcardHirLowerer {
    fn default() -> Self {
        Self::new()
    }
}

impl PostcardHirLowerer {
    pub fn new() -> Self {
        let mut module = hir::Module::new();
        let input_region = module.add_region("input");
        let cursor_type = kajit_format::hir_helpers::add_cursor_type(&mut module, input_region);

        Self {
            module,
            input_region,
            cursor_type,
            string_raw_type: None,
            bits128_raw_type: None,
            type_defs_by_shape: HashMap::new(),
            callables_by_name: HashMap::new(),
            locals: Vec::new(),
            next_local: 2,
            next_stmt: 0,
        }
    }

    fn finish(self) -> hir::Module {
        self.module
    }

    fn next_local(&mut self) -> hir::LocalId {
        let id = hir::LocalId::new(self.next_local);
        self.next_local += 1;
        id
    }

    pub fn next_stmt_id(&mut self) -> hir::StmtId {
        let id = hir::StmtId::new(self.next_stmt);
        self.next_stmt += 1;
        id
    }

    fn alloc_local(
        &mut self,
        name: impl Into<String>,
        ty: hir::Type,
        kind: hir::LocalKind,
    ) -> hir::LocalId {
        let local = self.next_local();
        self.locals.push(hir::LocalDecl {
            local,
            name: name.into(),
            ty,
            kind,
        });
        local
    }

    fn push_init(&mut self, statements: &mut Vec<hir::Stmt>, place: hir::Place, value: hir::Expr) {
        statements.push(hir::Stmt {
            id: self.next_stmt_id(),
            kind: hir::StmtKind::Init { place, value },
        });
    }

    fn shape_has_input_borrow(shape: &'static Shape) -> bool {
        match shape.scalar_type() {
            Some(ScalarType::Str | ScalarType::CowStr) => return true,
            Some(_) => return false,
            None => {}
        }

        if shape.is_transparent() {
            let (fields, _) = collect_fields(shape);
            return fields
                .iter()
                .any(|field| Self::shape_has_input_borrow(field.shape));
        }

        if let Some(opt_def) = get_option_def(shape) {
            return Self::shape_has_input_borrow(opt_def.t);
        }

        if let Some(ptr_def) = get_pointer_def(shape) {
            return ptr_def.pointee.is_some_and(Self::shape_has_input_borrow);
        }

        match &shape.def {
            Def::List(list_def) => return Self::shape_has_input_borrow(list_def.t),
            Def::Map(map_def) => {
                return Self::shape_has_input_borrow(map_def.k)
                    || Self::shape_has_input_borrow(map_def.v);
            }
            Def::Array(array_def) => return Self::shape_has_input_borrow(array_def.t),
            _ => {}
        }

        match &shape.ty {
            Type::User(UserType::Struct(_)) => {
                let (fields, _) = collect_fields(shape);
                fields
                    .iter()
                    .any(|field| Self::shape_has_input_borrow(field.shape))
            }
            Type::User(UserType::Enum(enum_type)) => collect_variants(enum_type)
                .iter()
                .flat_map(|variant| variant.fields.iter())
                .any(|field| Self::shape_has_input_borrow(field.shape)),
            _ => false,
        }
    }

    fn ensure_type_def(&mut self, shape: &'static Shape) -> hir::TypeDefId {
        let key = shape as *const Shape;
        if let Some(existing) = self.type_defs_by_shape.get(&key).copied() {
            return existing;
        }

        let generic_params = if Self::shape_has_input_borrow(shape) {
            vec![hir::GenericParam::Region {
                name: "r_input".to_owned(),
            }]
        } else {
            Vec::new()
        };

        let type_def = hir::TypeDef {
            name: shape.type_identifier.to_owned(),
            generic_params,
            kind: hir::TypeDefKind::Struct { fields: Vec::new() },
            size: None,
            transparent: false,
        };
        let type_id = self.module.add_type_def(type_def);
        self.type_defs_by_shape.insert(key, type_id);

        let shape_size = shape.layout.sized_layout().ok().map(|l| l.size() as u32);

        let kind = if let Def::List(list_def) = &shape.def {
            let offsets = kajit_malum::discover_vec_offsets(list_def, shape);
            let mut fields = vec![
                (
                    offsets.ptr_offset,
                    hir::FieldDef {
                        name: "ptr".to_owned(),
                        ty: hir::Type::persistent_addr(),
                        offset: Some(offsets.ptr_offset),
                    },
                ),
                (
                    offsets.len_offset,
                    hir::FieldDef {
                        name: "len".to_owned(),
                        ty: hir::Type::u(64),
                        offset: Some(offsets.len_offset),
                    },
                ),
                (
                    offsets.cap_offset,
                    hir::FieldDef {
                        name: "cap".to_owned(),
                        ty: hir::Type::u(64),
                        offset: Some(offsets.cap_offset),
                    },
                ),
            ];
            fields.sort_by_key(|(offset, _)| *offset);
            hir::TypeDefKind::Struct {
                fields: fields.into_iter().map(|(_, field)| field).collect(),
            }
        } else if shape.is_transparent() {
            let (fields, skipped) = collect_fields(shape);
            assert!(
                skipped.is_empty(),
                "postcard HIR prototype does not support transparent defaults"
            );
            hir::TypeDefKind::Struct {
                fields: fields
                    .into_iter()
                    .map(|field| hir::FieldDef {
                        name: field.name.to_owned(),
                        ty: self.lower_type(field.shape),
                        offset: Some(field.offset as u32),
                    })
                    .collect(),
            }
        } else if let Some(opt_def) = get_option_def(shape) {
            hir::TypeDefKind::Enum {
                variants: vec![
                    hir::VariantDef {
                        name: "None".to_owned(),
                        fields: Vec::new(),
                        discriminant: None,
                        init_fn: Some(opt_def.vtable.init_none as *const () as usize as u64),
                    },
                    hir::VariantDef {
                        name: "Some".to_owned(),
                        fields: vec![hir::FieldDef {
                            name: "value".to_owned(),
                            ty: self.lower_type(opt_def.t),
                            offset: None,
                        }],
                        discriminant: None,
                        init_fn: Some(opt_def.vtable.init_some as *const () as usize as u64),
                    },
                ],
                discriminant_width: None,
            }
        } else {
            match &shape.ty {
                Type::User(UserType::Struct(_)) => {
                    let (fields, skipped) = collect_fields(shape);
                    assert!(
                        skipped.is_empty(),
                        "postcard HIR prototype does not support skipped/defaulted fields"
                    );
                    hir::TypeDefKind::Struct {
                        fields: fields
                            .into_iter()
                            .map(|field| hir::FieldDef {
                                name: field.name.to_owned(),
                                ty: self.lower_type(field.shape),
                                offset: Some(field.offset as u32),
                            })
                            .collect(),
                    }
                }
                Type::User(UserType::Enum(enum_type)) => hir::TypeDefKind::Enum {
                    variants: collect_variants(enum_type)
                        .into_iter()
                        .map(|variant| hir::VariantDef {
                            name: variant.name.to_owned(),
                            fields: variant
                                .fields
                                .into_iter()
                                .map(|field| hir::FieldDef {
                                    name: field.name.to_owned(),
                                    ty: self.lower_type(field.shape),
                                    offset: Some(field.offset as u32),
                                })
                                .collect(),
                            discriminant: Some(variant.rust_discriminant),
                            init_fn: None,
                        })
                        .collect(),
                    discriminant_width: Some(discriminant_size(enum_type.enum_repr)),
                },
                _ => panic!(
                    "postcard HIR prototype only supports struct-like composite roots for now: {}",
                    shape.type_identifier
                ),
            }
        };

        self.module.type_defs[type_id].kind = kind;
        self.module.type_defs[type_id].size = shape_size;
        self.module.type_defs[type_id].transparent = shape.is_transparent();
        type_id
    }

    fn ensure_string_raw_type(&mut self) -> hir::TypeDefId {
        if let Some(existing) = self.string_raw_type {
            return existing;
        }
        let offsets = kajit_malum::discover_string_offsets();
        let mut fields = vec![
            (
                offsets.ptr_offset,
                hir::FieldDef {
                    name: "ptr".to_owned(),
                    ty: hir::Type::persistent_addr(),
                    offset: Some(offsets.ptr_offset),
                },
            ),
            (
                offsets.len_offset,
                hir::FieldDef {
                    name: "len".to_owned(),
                    ty: hir::Type::u(64),
                    offset: Some(offsets.len_offset),
                },
            ),
            (
                offsets.cap_offset,
                hir::FieldDef {
                    name: "cap".to_owned(),
                    ty: hir::Type::u(64),
                    offset: Some(offsets.cap_offset),
                },
            ),
        ];
        fields.sort_by_key(|(offset, _)| *offset);
        let type_id = self.module.add_type_def(hir::TypeDef {
            name: "HostStringRaw".to_owned(),
            generic_params: Vec::new(),
            kind: hir::TypeDefKind::Struct {
                fields: fields.into_iter().map(|(_, field)| field).collect(),
            },
            size: Some(core::mem::size_of::<String>() as u32),
            transparent: false,
        });
        self.string_raw_type = Some(type_id);
        type_id
    }

    fn ensure_bits128_raw_type(&mut self) -> hir::TypeDefId {
        if let Some(existing) = self.bits128_raw_type {
            return existing;
        }
        let type_id = self.module.add_type_def(hir::TypeDef {
            name: "Bits128Raw".to_owned(),
            generic_params: Vec::new(),
            kind: hir::TypeDefKind::Struct {
                fields: vec![
                    hir::FieldDef {
                        name: "lo".to_owned(),
                        ty: hir::Type::u(64),
                        offset: Some(0),
                    },
                    hir::FieldDef {
                        name: "hi".to_owned(),
                        ty: hir::Type::u(64),
                        offset: Some(8),
                    },
                ],
            },
            size: Some(16),
            transparent: false,
        });
        self.bits128_raw_type = Some(type_id);
        type_id
    }

    fn lower_type(&mut self, shape: &'static Shape) -> hir::Type {
        if is_unit(shape) {
            return hir::Type::unit();
        }

        if matches!(shape.def, Def::List(_)) {
            let type_id = self.ensure_type_def(shape);
            let args = if Self::shape_has_input_borrow(shape) {
                vec![hir::GenericArg::Region(self.input_region)]
            } else {
                Vec::new()
            };
            return hir::Type::named(type_id, args);
        }

        if let Def::Array(array_def) = &shape.def {
            return hir::Type::array(self.lower_type(array_def.t), array_def.n);
        }

        if shape.is_transparent() {
            let (fields, skipped) = collect_fields(shape);
            assert!(
                skipped.is_empty() && fields.len() == 1,
                "transparent HIR prototype expects one lowered field"
            );
            return self.lower_type(fields[0].shape);
        }

        if let Some(opt_def) = get_option_def(shape) {
            let type_id = self.ensure_type_def(shape);
            let args = if Self::shape_has_input_borrow(opt_def.t) {
                vec![hir::GenericArg::Region(self.input_region)]
            } else {
                Vec::new()
            };
            return hir::Type::named(type_id, args);
        }

        if let Some(st) = shape.scalar_type() {
            return match st {
                ScalarType::Unit => hir::Type::unit(),
                ScalarType::Bool => hir::Type::bool(),
                ScalarType::U8 => hir::Type::u(8),
                ScalarType::U16 => hir::Type::u(16),
                ScalarType::U32 => hir::Type::u(32),
                ScalarType::U64 => hir::Type::u(64),
                ScalarType::U128 => hir::Type::named(self.ensure_bits128_raw_type(), Vec::new()),
                ScalarType::USize => hir::Type::u(64),
                ScalarType::I8 => hir::Type::i(8),
                ScalarType::I16 => hir::Type::i(16),
                ScalarType::I32 => hir::Type::i(32),
                ScalarType::I64 => hir::Type::i(64),
                ScalarType::I128 => hir::Type::named(self.ensure_bits128_raw_type(), Vec::new()),
                ScalarType::ISize => hir::Type::i(64),
                ScalarType::Str => hir::Type::str(self.input_region),
                ScalarType::String => hir::Type::named(self.ensure_string_raw_type(), Vec::new()),
                ScalarType::Char => hir::Type::u(32),
                ScalarType::F32 => hir::Type::u(32),
                ScalarType::F64 => hir::Type::u(64),
                ScalarType::CowStr => {
                    panic!(
                        "postcard HIR prototype does not support scalar {st:?} yet for {}",
                        shape.type_identifier
                    );
                }
                _ => panic!(
                    "postcard HIR prototype encountered unknown scalar {st:?} for {}",
                    shape.type_identifier
                ),
            };
        }

        let type_id = self.ensure_type_def(shape);
        let args = if Self::shape_has_input_borrow(shape) {
            vec![hir::GenericArg::Region(self.input_region)]
        } else {
            Vec::new()
        };
        hir::Type::named(type_id, args)
    }

    fn ensure_postcard_reader(&mut self, scalar_type: ScalarType) -> hir::CallableId {
        let name = match scalar_type {
            ScalarType::Bool => "postcard.read_bool",
            ScalarType::U8 => "postcard.read_u8",
            ScalarType::U16 => "postcard.read_u16",
            ScalarType::U32 => "postcard.read_u32",
            ScalarType::U64 => "postcard.read_u64",
            ScalarType::U128 => "postcard.read_u128",
            ScalarType::USize => "postcard.read_usize",
            ScalarType::I8 => "postcard.read_i8",
            ScalarType::I16 => "postcard.read_i16",
            ScalarType::I32 => "postcard.read_i32",
            ScalarType::I64 => "postcard.read_i64",
            ScalarType::I128 => "postcard.read_i128",
            ScalarType::ISize => "postcard.read_isize",
            ScalarType::Str => "postcard.read_str",
            other => panic!("unsupported postcard HIR reader for {other:?}"),
        };
        if let Some(existing) = self.callables_by_name.get(name).copied() {
            return existing;
        }

        let returns = vec![match scalar_type {
            ScalarType::Str => hir::Type::str(self.input_region),
            _ => self.lower_type_for_scalar(scalar_type),
        }];
        let callable = hir::CallableSpec {
            kind: hir::CallableKind::Builtin,
            name: name.to_owned(),
            intrinsic: None,
            signature: hir::CallSignature {
                params: vec![hir::Type::mut_ref(hir::Type::named(
                    self.cursor_type,
                    vec![hir::GenericArg::Region(self.input_region)],
                ))],
                returns,
                effect_class: hir::EffectClass::Mutates,
                domain_effects: vec![hir::DomainEffect {
                    domain: "cursor".to_owned(),
                    access: hir::DomainAccess::Mutate,
                }],
                control: hir::ControlTransfer::MayFail,
                capabilities: vec!["deser.postcard".to_owned()],
                safety: hir::CallSafety::SafeCore,
            },
            docs: Some(format!(
                "Read a postcard {:?} value from the input cursor.",
                scalar_type
            )),
        };
        let callable_id = self.module.add_callable(callable);
        self.callables_by_name.insert(name, callable_id);
        callable_id
    }

    fn lower_type_for_scalar(&mut self, scalar_type: ScalarType) -> hir::Type {
        match scalar_type {
            ScalarType::Bool => hir::Type::bool(),
            ScalarType::U8 => hir::Type::u(8),
            ScalarType::U16 => hir::Type::u(16),
            ScalarType::U32 => hir::Type::u(32),
            ScalarType::U64 => hir::Type::u(64),
            ScalarType::U128 => hir::Type::named(self.ensure_bits128_raw_type(), Vec::new()),
            ScalarType::USize => hir::Type::u(64),
            ScalarType::I8 => hir::Type::i(8),
            ScalarType::I16 => hir::Type::i(16),
            ScalarType::I32 => hir::Type::i(32),
            ScalarType::I64 => hir::Type::i(64),
            ScalarType::I128 => hir::Type::named(self.ensure_bits128_raw_type(), Vec::new()),
            ScalarType::ISize => hir::Type::i(64),
            ScalarType::Char => hir::Type::u(32),
            ScalarType::F32 => hir::Type::u(32),
            ScalarType::F64 => hir::Type::u(64),
            other => panic!("unsupported postcard HIR scalar type {other:?}"),
        }
    }

    fn cursor_bytes_expr(&self, cursor_local: hir::LocalId) -> hir::Expr {
        hir::Expr::Field {
            base: Box::new(hir::Expr::Deref(Box::new(hir::Expr::Local(cursor_local)))),
            field: "bytes".to_owned(),
        }
    }

    fn cursor_pos_expr(&self, cursor_local: hir::LocalId) -> hir::Expr {
        hir::Expr::Field {
            base: Box::new(hir::Expr::Deref(Box::new(hir::Expr::Local(cursor_local)))),
            field: "pos".to_owned(),
        }
    }

    fn push_cursor_pos_update(
        &mut self,
        statements: &mut Vec<hir::Stmt>,
        cursor_local: hir::LocalId,
        new_pos: hir::Expr,
    ) {
        statements.push(hir::Stmt {
            id: self.next_stmt_id(),
            kind: hir::StmtKind::Assign {
                place: hir::Place::Field {
                    base: Box::new(hir::Place::Deref {
                        base: Box::new(hir::Expr::Local(cursor_local)),
                    }),
                    field: "pos".to_owned(),
                },
                value: new_pos,
            },
        });
    }

    fn push_cursor_bounds_check(
        &mut self,
        statements: &mut Vec<hir::Stmt>,
        cursor_local: hir::LocalId,
        needed: u64,
        code: hir::ErrorCode,
    ) {
        self.push_cursor_bounds_check_expr(
            statements,
            cursor_local,
            hir::Expr::Literal(hir::Literal::Integer(needed)),
            code,
        );
    }

    fn push_cursor_bounds_check_expr(
        &mut self,
        statements: &mut Vec<hir::Stmt>,
        cursor_local: hir::LocalId,
        needed: hir::Expr,
        code: hir::ErrorCode,
    ) {
        let bytes = self.cursor_bytes_expr(cursor_local);
        let end = hir::Expr::SliceLen {
            value: Box::new(bytes),
        };
        let pos = self.cursor_pos_expr(cursor_local);
        let limit = hir::Expr::Binary {
            op: hir::BinaryOp::Add,
            lhs: Box::new(pos),
            rhs: Box::new(needed),
        };
        let fail_condition = hir::Expr::Binary {
            op: hir::BinaryOp::Gt,
            lhs: Box::new(limit),
            rhs: Box::new(end),
        };
        statements.push(hir::Stmt {
            id: self.next_stmt_id(),
            kind: hir::StmtKind::If {
                condition: fail_condition,
                then_block: hir::Block {
                    scope: hir::ScopeId::new(0),
                    statements: vec![hir::Stmt {
                        id: self.next_stmt_id(),
                        kind: hir::StmtKind::Fail { code },
                    }],
                },
                else_block: Some(hir::Block {
                    scope: hir::ScopeId::new(0),
                    statements: Vec::new(),
                }),
            },
        });
    }

    fn lower_postcard_fixed_width_scalar_into_place(
        &mut self,
        statements: &mut Vec<hir::Stmt>,
        cursor_local: hir::LocalId,
        place: hir::Place,
        scalar_type: ScalarType,
    ) {
        let width = match scalar_type {
            ScalarType::Bool | ScalarType::U8 | ScalarType::I8 => hir::MemoryWidth::W1,
            ScalarType::F32 => hir::MemoryWidth::W4,
            ScalarType::F64 => hir::MemoryWidth::W8,
            other => panic!("unsupported fixed-width postcard HIR scalar {other:?}"),
        };
        self.push_cursor_bounds_check(
            statements,
            cursor_local,
            u64::from(width.bytes()),
            hir::ErrorCode::UnexpectedEof,
        );

        let bytes = self.cursor_bytes_expr(cursor_local);
        let pos = self.cursor_pos_expr(cursor_local);
        let addr = hir::Expr::Binary {
            op: hir::BinaryOp::Add,
            lhs: Box::new(hir::Expr::SliceData {
                value: Box::new(bytes),
            }),
            rhs: Box::new(pos.clone()),
        };

        let raw_local = self.alloc_local(
            format!("fixed_scalar_{}", self.locals.len()),
            match width {
                hir::MemoryWidth::W1 => hir::Type::u(8),
                hir::MemoryWidth::W2 => hir::Type::u(16),
                hir::MemoryWidth::W4 => hir::Type::u(32),
                hir::MemoryWidth::W8 => hir::Type::u(64),
            },
            hir::LocalKind::Temp,
        );
        self.push_init(
            statements,
            hir::Place::Local(raw_local),
            hir::Expr::Load {
                addr: Box::new(addr),
                width,
            },
        );
        self.push_cursor_pos_update(
            statements,
            cursor_local,
            hir::Expr::Binary {
                op: hir::BinaryOp::Add,
                lhs: Box::new(pos),
                rhs: Box::new(hir::Expr::Literal(hir::Literal::Integer(u64::from(
                    width.bytes(),
                )))),
            },
        );

        match scalar_type {
            ScalarType::Bool => {
                let invalid = hir::Expr::Binary {
                    op: hir::BinaryOp::Gt,
                    lhs: Box::new(hir::Expr::Local(raw_local)),
                    rhs: Box::new(hir::Expr::Literal(hir::Literal::Integer(1))),
                };
                statements.push(hir::Stmt {
                    id: self.next_stmt_id(),
                    kind: hir::StmtKind::If {
                        condition: invalid,
                        then_block: hir::Block {
                            scope: hir::ScopeId::new(0),
                            statements: vec![hir::Stmt {
                                id: self.next_stmt_id(),
                                kind: hir::StmtKind::Fail {
                                    code: hir::ErrorCode::InvalidBool,
                                },
                            }],
                        },
                        else_block: Some(hir::Block {
                            scope: hir::ScopeId::new(0),
                            statements: vec![hir::Stmt {
                                id: self.next_stmt_id(),
                                kind: hir::StmtKind::Init {
                                    place,
                                    value: hir::Expr::Binary {
                                        op: hir::BinaryOp::Ne,
                                        lhs: Box::new(hir::Expr::Local(raw_local)),
                                        rhs: Box::new(hir::Expr::Literal(hir::Literal::Integer(0))),
                                    },
                                },
                            }],
                        }),
                    },
                });
            }
            ScalarType::U8 | ScalarType::I8 | ScalarType::F32 | ScalarType::F64 => {
                self.push_init(statements, place, hir::Expr::Local(raw_local));
            }
            _ => unreachable!(),
        }
    }

    fn bits128_field_place(&self, base: hir::Place, field: &str) -> hir::Place {
        hir::Place::Field {
            base: Box::new(base),
            field: field.to_owned(),
        }
    }

    fn field_place(&self, base: hir::Place, field: &str) -> hir::Place {
        hir::Place::Field {
            base: Box::new(base),
            field: field.to_owned(),
        }
    }

    fn postcard_varint128_finish_into_place(
        &mut self,
        statements: &mut Vec<hir::Stmt>,
        place: hir::Place,
        zigzag: bool,
        acc_lo_local: hir::LocalId,
        acc_hi_local: hir::LocalId,
    ) {
        let (lo_value, hi_value) = if zigzag {
            let shifted_lo = hir::Expr::Binary {
                op: hir::BinaryOp::BitOr,
                lhs: Box::new(hir::Expr::Binary {
                    op: hir::BinaryOp::Shr,
                    lhs: Box::new(hir::Expr::Local(acc_lo_local)),
                    rhs: Box::new(hir::Expr::Literal(hir::Literal::Integer(1))),
                }),
                rhs: Box::new(hir::Expr::Binary {
                    op: hir::BinaryOp::Shl,
                    lhs: Box::new(hir::Expr::Local(acc_hi_local)),
                    rhs: Box::new(hir::Expr::Literal(hir::Literal::Integer(63))),
                }),
            };
            let shifted_hi = hir::Expr::Binary {
                op: hir::BinaryOp::Shr,
                lhs: Box::new(hir::Expr::Local(acc_hi_local)),
                rhs: Box::new(hir::Expr::Literal(hir::Literal::Integer(1))),
            };
            let sign = hir::Expr::Binary {
                op: hir::BinaryOp::BitAnd,
                lhs: Box::new(hir::Expr::Local(acc_lo_local)),
                rhs: Box::new(hir::Expr::Literal(hir::Literal::Integer(1))),
            };
            let neg_mask = hir::Expr::Binary {
                op: hir::BinaryOp::Sub,
                lhs: Box::new(hir::Expr::Literal(hir::Literal::Integer(0))),
                rhs: Box::new(sign),
            };
            (
                hir::Expr::Binary {
                    op: hir::BinaryOp::Xor,
                    lhs: Box::new(shifted_lo),
                    rhs: Box::new(neg_mask.clone()),
                },
                hir::Expr::Binary {
                    op: hir::BinaryOp::Xor,
                    lhs: Box::new(shifted_hi),
                    rhs: Box::new(neg_mask),
                },
            )
        } else {
            (
                hir::Expr::Local(acc_lo_local),
                hir::Expr::Local(acc_hi_local),
            )
        };

        self.push_init(
            statements,
            self.bits128_field_place(place.clone(), "lo"),
            lo_value,
        );
        self.push_init(statements, self.bits128_field_place(place, "hi"), hi_value);
    }

    fn postcard_varint128_finish_block(
        &mut self,
        place: hir::Place,
        zigzag: bool,
        acc_lo_local: hir::LocalId,
        acc_hi_local: hir::LocalId,
        byte_index: u64,
        raw_local: hir::LocalId,
    ) -> hir::Block {
        let mut block = hir::Block {
            scope: hir::ScopeId::new(0),
            statements: Vec::new(),
        };

        if byte_index + 1 == Self::postcard_varint_max_bytes(128) {
            let extra_bits = hir::Expr::Binary {
                op: hir::BinaryOp::BitAnd,
                lhs: Box::new(hir::Expr::Local(raw_local)),
                rhs: Box::new(hir::Expr::Literal(hir::Literal::Integer(0x7e))),
            };
            let mut ok_block = hir::Block {
                scope: hir::ScopeId::new(0),
                statements: Vec::new(),
            };
            self.postcard_varint128_finish_into_place(
                &mut ok_block.statements,
                place,
                zigzag,
                acc_lo_local,
                acc_hi_local,
            );
            block.statements.push(hir::Stmt {
                id: self.next_stmt_id(),
                kind: hir::StmtKind::If {
                    condition: hir::Expr::Binary {
                        op: hir::BinaryOp::Ne,
                        lhs: Box::new(extra_bits),
                        rhs: Box::new(hir::Expr::Literal(hir::Literal::Integer(0))),
                    },
                    then_block: hir::Block {
                        scope: hir::ScopeId::new(0),
                        statements: vec![hir::Stmt {
                            id: self.next_stmt_id(),
                            kind: hir::StmtKind::Fail {
                                code: hir::ErrorCode::InvalidVarint,
                            },
                        }],
                    },
                    else_block: Some(ok_block),
                },
            });
            return block;
        }

        self.postcard_varint128_finish_into_place(
            &mut block.statements,
            place,
            zigzag,
            acc_lo_local,
            acc_hi_local,
        );
        block
    }

    fn lower_postcard_varint128_step(
        &mut self,
        statements: &mut Vec<hir::Stmt>,
        cursor_local: hir::LocalId,
        place: hir::Place,
        zigzag: bool,
        acc_lo_local: hir::LocalId,
        acc_hi_local: hir::LocalId,
        byte_index: u64,
    ) {
        self.push_cursor_bounds_check(statements, cursor_local, 1, hir::ErrorCode::UnexpectedEof);

        let pos = self.cursor_pos_expr(cursor_local);
        let addr = hir::Expr::Binary {
            op: hir::BinaryOp::Add,
            lhs: Box::new(hir::Expr::SliceData {
                value: Box::new(self.cursor_bytes_expr(cursor_local)),
            }),
            rhs: Box::new(pos.clone()),
        };
        let raw_local = self.alloc_local(
            format!("varint128_byte_{}", self.locals.len()),
            hir::Type::u(8),
            hir::LocalKind::Temp,
        );
        self.push_init(
            statements,
            hir::Place::Local(raw_local),
            hir::Expr::Load {
                addr: Box::new(addr),
                width: hir::MemoryWidth::W1,
            },
        );
        self.push_cursor_pos_update(
            statements,
            cursor_local,
            hir::Expr::Binary {
                op: hir::BinaryOp::Add,
                lhs: Box::new(pos),
                rhs: Box::new(hir::Expr::Literal(hir::Literal::Integer(1))),
            },
        );

        let low = hir::Expr::Binary {
            op: hir::BinaryOp::BitAnd,
            lhs: Box::new(hir::Expr::Local(raw_local)),
            rhs: Box::new(hir::Expr::Literal(hir::Literal::Integer(0x7f))),
        };
        let shift = byte_index * 7;
        if shift < 64 {
            let lo_part = if shift == 0 {
                low.clone()
            } else {
                hir::Expr::Binary {
                    op: hir::BinaryOp::Shl,
                    lhs: Box::new(low.clone()),
                    rhs: Box::new(hir::Expr::Literal(hir::Literal::Integer(shift))),
                }
            };
            statements.push(hir::Stmt {
                id: self.next_stmt_id(),
                kind: hir::StmtKind::Assign {
                    place: hir::Place::Local(acc_lo_local),
                    value: hir::Expr::Binary {
                        op: hir::BinaryOp::BitOr,
                        lhs: Box::new(hir::Expr::Local(acc_lo_local)),
                        rhs: Box::new(lo_part),
                    },
                },
            });

            if shift > 57 {
                let hi_part = hir::Expr::Binary {
                    op: hir::BinaryOp::Shr,
                    lhs: Box::new(low),
                    rhs: Box::new(hir::Expr::Literal(hir::Literal::Integer(64 - shift))),
                };
                statements.push(hir::Stmt {
                    id: self.next_stmt_id(),
                    kind: hir::StmtKind::Assign {
                        place: hir::Place::Local(acc_hi_local),
                        value: hir::Expr::Binary {
                            op: hir::BinaryOp::BitOr,
                            lhs: Box::new(hir::Expr::Local(acc_hi_local)),
                            rhs: Box::new(hi_part),
                        },
                    },
                });
            }
        } else {
            let hi_part = hir::Expr::Binary {
                op: hir::BinaryOp::Shl,
                lhs: Box::new(low),
                rhs: Box::new(hir::Expr::Literal(hir::Literal::Integer(shift - 64))),
            };
            statements.push(hir::Stmt {
                id: self.next_stmt_id(),
                kind: hir::StmtKind::Assign {
                    place: hir::Place::Local(acc_hi_local),
                    value: hir::Expr::Binary {
                        op: hir::BinaryOp::BitOr,
                        lhs: Box::new(hir::Expr::Local(acc_hi_local)),
                        rhs: Box::new(hi_part),
                    },
                },
            });
        }

        let max_bytes = Self::postcard_varint_max_bytes(128);
        let then_block = if byte_index + 1 == max_bytes {
            hir::Block {
                scope: hir::ScopeId::new(0),
                statements: vec![hir::Stmt {
                    id: self.next_stmt_id(),
                    kind: hir::StmtKind::Fail {
                        code: hir::ErrorCode::InvalidVarint,
                    },
                }],
            }
        } else {
            let mut block = hir::Block {
                scope: hir::ScopeId::new(0),
                statements: Vec::new(),
            };
            self.lower_postcard_varint128_step(
                &mut block.statements,
                cursor_local,
                place.clone(),
                zigzag,
                acc_lo_local,
                acc_hi_local,
                byte_index + 1,
            );
            block
        };
        let else_block = Some(self.postcard_varint128_finish_block(
            place,
            zigzag,
            acc_lo_local,
            acc_hi_local,
            byte_index,
            raw_local,
        ));
        statements.push(hir::Stmt {
            id: self.next_stmt_id(),
            kind: hir::StmtKind::If {
                condition: hir::Expr::Binary {
                    op: hir::BinaryOp::Ne,
                    lhs: Box::new(hir::Expr::Binary {
                        op: hir::BinaryOp::BitAnd,
                        lhs: Box::new(hir::Expr::Local(raw_local)),
                        rhs: Box::new(hir::Expr::Literal(hir::Literal::Integer(0x80))),
                    }),
                    rhs: Box::new(hir::Expr::Literal(hir::Literal::Integer(0))),
                },
                then_block,
                else_block,
            },
        });
    }

    fn lower_postcard_varint128_into_place(
        &mut self,
        statements: &mut Vec<hir::Stmt>,
        cursor_local: hir::LocalId,
        place: hir::Place,
        zigzag: bool,
    ) {
        let acc_lo_local = self.alloc_local(
            format!("varint128_lo_{}", self.locals.len()),
            hir::Type::u(64),
            hir::LocalKind::Temp,
        );
        let acc_hi_local = self.alloc_local(
            format!("varint128_hi_{}", self.locals.len()),
            hir::Type::u(64),
            hir::LocalKind::Temp,
        );
        self.push_init(
            statements,
            hir::Place::Local(acc_lo_local),
            hir::Expr::Literal(hir::Literal::Integer(0)),
        );
        self.push_init(
            statements,
            hir::Place::Local(acc_hi_local),
            hir::Expr::Literal(hir::Literal::Integer(0)),
        );
        self.lower_postcard_varint128_step(
            statements,
            cursor_local,
            place,
            zigzag,
            acc_lo_local,
            acc_hi_local,
            0,
        );
    }

    fn lower_postcard_char_into_place(
        &mut self,
        statements: &mut Vec<hir::Stmt>,
        cursor_local: hir::LocalId,
        place: hir::Place,
    ) {
        let len_local = self.alloc_local(
            format!("char_len_{}", self.locals.len()),
            hir::Type::u(32),
            hir::LocalKind::Temp,
        );
        self.lower_postcard_varint_into_place(
            statements,
            cursor_local,
            hir::Place::Local(len_local),
            32,
            false,
        );

        let invalid_len = hir::Expr::Binary {
            op: hir::BinaryOp::Or,
            lhs: Box::new(hir::Expr::Binary {
                op: hir::BinaryOp::Eq,
                lhs: Box::new(hir::Expr::Local(len_local)),
                rhs: Box::new(hir::Expr::Literal(hir::Literal::Integer(0))),
            }),
            rhs: Box::new(hir::Expr::Binary {
                op: hir::BinaryOp::Gt,
                lhs: Box::new(hir::Expr::Local(len_local)),
                rhs: Box::new(hir::Expr::Literal(hir::Literal::Integer(4))),
            }),
        };
        statements.push(hir::Stmt {
            id: self.next_stmt_id(),
            kind: hir::StmtKind::If {
                condition: invalid_len,
                then_block: hir::Block {
                    scope: hir::ScopeId::new(0),
                    statements: vec![hir::Stmt {
                        id: self.next_stmt_id(),
                        kind: hir::StmtKind::Fail {
                            code: hir::ErrorCode::InvalidUtf8,
                        },
                    }],
                },
                else_block: Some(hir::Block {
                    scope: hir::ScopeId::new(0),
                    statements: Vec::new(),
                }),
            },
        });

        self.push_cursor_bounds_check_expr(
            statements,
            cursor_local,
            hir::Expr::Local(len_local),
            hir::ErrorCode::UnexpectedEof,
        );

        let bytes = self.cursor_bytes_expr(cursor_local);
        let pos = self.cursor_pos_expr(cursor_local);
        let data_local = self.alloc_local(
            format!("char_data_{}", self.locals.len()),
            hir::Type::u(64),
            hir::LocalKind::Temp,
        );
        self.push_init(
            statements,
            hir::Place::Local(data_local),
            hir::Expr::Binary {
                op: hir::BinaryOp::Add,
                lhs: Box::new(hir::Expr::SliceData {
                    value: Box::new(bytes),
                }),
                rhs: Box::new(pos.clone()),
            },
        );

        let validate_utf8 = self.ensure_runtime_validate_utf8_range();
        statements.push(hir::Stmt {
            id: self.next_stmt_id(),
            kind: hir::StmtKind::Expr(hir::Expr::Call(hir::CallExpr {
                target: hir::CallTarget::Callable(validate_utf8),
                args: vec![hir::Expr::Local(data_local), hir::Expr::Local(len_local)],
            })),
        });

        let raw0 = self.alloc_local(
            format!("char_raw0_{}", self.locals.len()),
            hir::Type::u(8),
            hir::LocalKind::Temp,
        );
        let raw1 = self.alloc_local(
            format!("char_raw1_{}", self.locals.len()),
            hir::Type::u(8),
            hir::LocalKind::Temp,
        );
        let raw2 = self.alloc_local(
            format!("char_raw2_{}", self.locals.len()),
            hir::Type::u(8),
            hir::LocalKind::Temp,
        );
        let raw3 = self.alloc_local(
            format!("char_raw3_{}", self.locals.len()),
            hir::Type::u(8),
            hir::LocalKind::Temp,
        );
        let code_local = self.alloc_local(
            format!("char_code_{}", self.locals.len()),
            hir::Type::u(32),
            hir::LocalKind::Temp,
        );

        self.push_init(
            statements,
            hir::Place::Local(raw0),
            hir::Expr::Load {
                addr: Box::new(hir::Expr::Local(data_local)),
                width: hir::MemoryWidth::W1,
            },
        );

        let load_byte = |data_local, offset| hir::Expr::Load {
            addr: Box::new(hir::Expr::Binary {
                op: hir::BinaryOp::Add,
                lhs: Box::new(hir::Expr::Local(data_local)),
                rhs: Box::new(hir::Expr::Literal(hir::Literal::Integer(offset))),
            }),
            width: hir::MemoryWidth::W1,
        };

        let one_byte = hir::Block {
            scope: hir::ScopeId::new(0),
            statements: vec![hir::Stmt {
                id: self.next_stmt_id(),
                kind: hir::StmtKind::Init {
                    place: hir::Place::Local(code_local),
                    value: hir::Expr::Local(raw0),
                },
            }],
        };

        let mut two_byte_statements = Vec::new();
        self.push_init(
            &mut two_byte_statements,
            hir::Place::Local(raw1),
            load_byte(data_local, 1),
        );
        two_byte_statements.push(hir::Stmt {
            id: self.next_stmt_id(),
            kind: hir::StmtKind::Init {
                place: hir::Place::Local(code_local),
                value: hir::Expr::Binary {
                    op: hir::BinaryOp::BitOr,
                    lhs: Box::new(hir::Expr::Binary {
                        op: hir::BinaryOp::Shl,
                        lhs: Box::new(hir::Expr::Binary {
                            op: hir::BinaryOp::BitAnd,
                            lhs: Box::new(hir::Expr::Local(raw0)),
                            rhs: Box::new(hir::Expr::Literal(hir::Literal::Integer(0x1f))),
                        }),
                        rhs: Box::new(hir::Expr::Literal(hir::Literal::Integer(6))),
                    }),
                    rhs: Box::new(hir::Expr::Binary {
                        op: hir::BinaryOp::BitAnd,
                        lhs: Box::new(hir::Expr::Local(raw1)),
                        rhs: Box::new(hir::Expr::Literal(hir::Literal::Integer(0x3f))),
                    }),
                },
            },
        });
        let two_byte = hir::Block {
            scope: hir::ScopeId::new(0),
            statements: two_byte_statements,
        };

        let mut three_byte_statements = Vec::new();
        self.push_init(
            &mut three_byte_statements,
            hir::Place::Local(raw1),
            load_byte(data_local, 1),
        );
        self.push_init(
            &mut three_byte_statements,
            hir::Place::Local(raw2),
            load_byte(data_local, 2),
        );
        three_byte_statements.push(hir::Stmt {
            id: self.next_stmt_id(),
            kind: hir::StmtKind::Init {
                place: hir::Place::Local(code_local),
                value: hir::Expr::Binary {
                    op: hir::BinaryOp::BitOr,
                    lhs: Box::new(hir::Expr::Binary {
                        op: hir::BinaryOp::Shl,
                        lhs: Box::new(hir::Expr::Binary {
                            op: hir::BinaryOp::BitAnd,
                            lhs: Box::new(hir::Expr::Local(raw0)),
                            rhs: Box::new(hir::Expr::Literal(hir::Literal::Integer(0x0f))),
                        }),
                        rhs: Box::new(hir::Expr::Literal(hir::Literal::Integer(12))),
                    }),
                    rhs: Box::new(hir::Expr::Binary {
                        op: hir::BinaryOp::BitOr,
                        lhs: Box::new(hir::Expr::Binary {
                            op: hir::BinaryOp::Shl,
                            lhs: Box::new(hir::Expr::Binary {
                                op: hir::BinaryOp::BitAnd,
                                lhs: Box::new(hir::Expr::Local(raw1)),
                                rhs: Box::new(hir::Expr::Literal(hir::Literal::Integer(0x3f))),
                            }),
                            rhs: Box::new(hir::Expr::Literal(hir::Literal::Integer(6))),
                        }),
                        rhs: Box::new(hir::Expr::Binary {
                            op: hir::BinaryOp::BitAnd,
                            lhs: Box::new(hir::Expr::Local(raw2)),
                            rhs: Box::new(hir::Expr::Literal(hir::Literal::Integer(0x3f))),
                        }),
                    }),
                },
            },
        });
        let three_byte = hir::Block {
            scope: hir::ScopeId::new(0),
            statements: three_byte_statements,
        };

        let mut four_byte_statements = Vec::new();
        self.push_init(
            &mut four_byte_statements,
            hir::Place::Local(raw1),
            load_byte(data_local, 1),
        );
        self.push_init(
            &mut four_byte_statements,
            hir::Place::Local(raw2),
            load_byte(data_local, 2),
        );
        self.push_init(
            &mut four_byte_statements,
            hir::Place::Local(raw3),
            load_byte(data_local, 3),
        );
        four_byte_statements.push(hir::Stmt {
            id: self.next_stmt_id(),
            kind: hir::StmtKind::Init {
                place: hir::Place::Local(code_local),
                value: hir::Expr::Binary {
                    op: hir::BinaryOp::BitOr,
                    lhs: Box::new(hir::Expr::Binary {
                        op: hir::BinaryOp::BitOr,
                        lhs: Box::new(hir::Expr::Binary {
                            op: hir::BinaryOp::Shl,
                            lhs: Box::new(hir::Expr::Binary {
                                op: hir::BinaryOp::BitAnd,
                                lhs: Box::new(hir::Expr::Local(raw0)),
                                rhs: Box::new(hir::Expr::Literal(hir::Literal::Integer(0x07))),
                            }),
                            rhs: Box::new(hir::Expr::Literal(hir::Literal::Integer(18))),
                        }),
                        rhs: Box::new(hir::Expr::Binary {
                            op: hir::BinaryOp::Shl,
                            lhs: Box::new(hir::Expr::Binary {
                                op: hir::BinaryOp::BitAnd,
                                lhs: Box::new(hir::Expr::Local(raw1)),
                                rhs: Box::new(hir::Expr::Literal(hir::Literal::Integer(0x3f))),
                            }),
                            rhs: Box::new(hir::Expr::Literal(hir::Literal::Integer(12))),
                        }),
                    }),
                    rhs: Box::new(hir::Expr::Binary {
                        op: hir::BinaryOp::BitOr,
                        lhs: Box::new(hir::Expr::Binary {
                            op: hir::BinaryOp::Shl,
                            lhs: Box::new(hir::Expr::Binary {
                                op: hir::BinaryOp::BitAnd,
                                lhs: Box::new(hir::Expr::Local(raw2)),
                                rhs: Box::new(hir::Expr::Literal(hir::Literal::Integer(0x3f))),
                            }),
                            rhs: Box::new(hir::Expr::Literal(hir::Literal::Integer(6))),
                        }),
                        rhs: Box::new(hir::Expr::Binary {
                            op: hir::BinaryOp::BitAnd,
                            lhs: Box::new(hir::Expr::Local(raw3)),
                            rhs: Box::new(hir::Expr::Literal(hir::Literal::Integer(0x3f))),
                        }),
                    }),
                },
            },
        });
        let four_byte = hir::Block {
            scope: hir::ScopeId::new(0),
            statements: four_byte_statements,
        };

        statements.push(hir::Stmt {
            id: self.next_stmt_id(),
            kind: hir::StmtKind::If {
                condition: hir::Expr::Binary {
                    op: hir::BinaryOp::Eq,
                    lhs: Box::new(hir::Expr::Local(len_local)),
                    rhs: Box::new(hir::Expr::Literal(hir::Literal::Integer(1))),
                },
                then_block: one_byte,
                else_block: Some(hir::Block {
                    scope: hir::ScopeId::new(0),
                    statements: vec![hir::Stmt {
                        id: self.next_stmt_id(),
                        kind: hir::StmtKind::If {
                            condition: hir::Expr::Binary {
                                op: hir::BinaryOp::Eq,
                                lhs: Box::new(hir::Expr::Local(len_local)),
                                rhs: Box::new(hir::Expr::Literal(hir::Literal::Integer(2))),
                            },
                            then_block: two_byte,
                            else_block: Some(hir::Block {
                                scope: hir::ScopeId::new(0),
                                statements: vec![hir::Stmt {
                                    id: self.next_stmt_id(),
                                    kind: hir::StmtKind::If {
                                        condition: hir::Expr::Binary {
                                            op: hir::BinaryOp::Eq,
                                            lhs: Box::new(hir::Expr::Local(len_local)),
                                            rhs: Box::new(hir::Expr::Literal(
                                                hir::Literal::Integer(3),
                                            )),
                                        },
                                        then_block: three_byte,
                                        else_block: Some(four_byte),
                                    },
                                }],
                            }),
                        },
                    }],
                }),
            },
        });

        self.push_init(statements, place, hir::Expr::Local(code_local));
        self.push_cursor_pos_update(
            statements,
            cursor_local,
            hir::Expr::Binary {
                op: hir::BinaryOp::Add,
                lhs: Box::new(pos),
                rhs: Box::new(hir::Expr::Local(len_local)),
            },
        );
    }

    fn lower_postcard_option_tag_into_local(
        &mut self,
        statements: &mut Vec<hir::Stmt>,
        cursor_local: hir::LocalId,
        tag_local: hir::LocalId,
    ) {
        self.push_cursor_bounds_check(statements, cursor_local, 1, hir::ErrorCode::UnexpectedEof);

        let bytes = self.cursor_bytes_expr(cursor_local);
        let pos = self.cursor_pos_expr(cursor_local);
        let addr = hir::Expr::Binary {
            op: hir::BinaryOp::Add,
            lhs: Box::new(hir::Expr::SliceData {
                value: Box::new(bytes),
            }),
            rhs: Box::new(pos.clone()),
        };
        let raw_local = self.alloc_local(
            format!("option_tag_raw_{}", self.locals.len()),
            hir::Type::u(8),
            hir::LocalKind::Temp,
        );
        self.push_init(
            statements,
            hir::Place::Local(raw_local),
            hir::Expr::Load {
                addr: Box::new(addr),
                width: hir::MemoryWidth::W1,
            },
        );
        self.push_cursor_pos_update(
            statements,
            cursor_local,
            hir::Expr::Binary {
                op: hir::BinaryOp::Add,
                lhs: Box::new(pos),
                rhs: Box::new(hir::Expr::Literal(hir::Literal::Integer(1))),
            },
        );
        let invalid = hir::Expr::Binary {
            op: hir::BinaryOp::Gt,
            lhs: Box::new(hir::Expr::Local(raw_local)),
            rhs: Box::new(hir::Expr::Literal(hir::Literal::Integer(1))),
        };
        statements.push(hir::Stmt {
            id: self.next_stmt_id(),
            kind: hir::StmtKind::If {
                condition: invalid,
                then_block: hir::Block {
                    scope: hir::ScopeId::new(0),
                    statements: vec![hir::Stmt {
                        id: self.next_stmt_id(),
                        kind: hir::StmtKind::Fail {
                            code: hir::ErrorCode::UnknownVariant,
                        },
                    }],
                },
                else_block: Some(hir::Block {
                    scope: hir::ScopeId::new(0),
                    statements: vec![hir::Stmt {
                        id: self.next_stmt_id(),
                        kind: hir::StmtKind::Init {
                            place: hir::Place::Local(tag_local),
                            value: hir::Expr::Binary {
                                op: hir::BinaryOp::Ne,
                                lhs: Box::new(hir::Expr::Local(raw_local)),
                                rhs: Box::new(hir::Expr::Literal(hir::Literal::Integer(0))),
                            },
                        },
                    }],
                }),
            },
        });
    }

    fn ensure_runtime_validate_utf8_range(&mut self) -> hir::CallableId {
        const NAME: &str = "runtime.validate_utf8_range";
        if let Some(existing) = self.callables_by_name.get(NAME).copied() {
            return existing;
        }

        let callable = hir::CallableSpec {
            kind: hir::CallableKind::Host,
            name: NAME.to_owned(),
            intrinsic: Some(hir::RuntimeIntrinsic::ValidateUtf8Range),
            signature: hir::CallSignature {
                params: vec![hir::Type::u(64), hir::Type::u(32)],
                returns: vec![],
                effect_class: hir::EffectClass::Reads,
                domain_effects: vec![hir::DomainEffect {
                    domain: "input".to_owned(),
                    access: hir::DomainAccess::Read,
                }],
                control: hir::ControlTransfer::MayFail,
                capabilities: vec!["runtime.utf8".to_owned()],
                safety: hir::CallSafety::OpaqueHost,
            },
            docs: Some("Validate that a borrowed byte range is UTF-8.".to_owned()),
        };
        let callable_id = self.module.add_callable(callable);
        self.callables_by_name.insert(NAME, callable_id);
        callable_id
    }

    fn ensure_runtime_option_init_none(&mut self) -> hir::CallableId {
        const NAME: &str = "runtime.option_init_none";
        if let Some(existing) = self.callables_by_name.get(NAME).copied() {
            return existing;
        }
        let callable = hir::CallableSpec {
            kind: hir::CallableKind::Host,
            name: NAME.to_owned(),
            intrinsic: Some(hir::RuntimeIntrinsic::OptionInitNone),
            signature: hir::CallSignature {
                params: vec![hir::Type::u(64), hir::Type::u(64)],
                returns: vec![],
                effect_class: hir::EffectClass::Barrier,
                domain_effects: vec![hir::DomainEffect {
                    domain: "output".to_owned(),
                    access: hir::DomainAccess::Mutate,
                }],
                control: hir::ControlTransfer::Returns,
                capabilities: vec!["runtime.option".to_owned()],
                safety: hir::CallSafety::OpaqueHost,
            },
            docs: Some("Initialize an Option destination with None via its vtable.".to_owned()),
        };
        let callable_id = self.module.add_callable(callable);
        self.callables_by_name.insert(NAME, callable_id);
        callable_id
    }

    fn ensure_runtime_option_init_some(&mut self) -> hir::CallableId {
        const NAME: &str = "runtime.option_init_some";
        if let Some(existing) = self.callables_by_name.get(NAME).copied() {
            return existing;
        }
        let callable = hir::CallableSpec {
            kind: hir::CallableKind::Host,
            name: NAME.to_owned(),
            intrinsic: Some(hir::RuntimeIntrinsic::OptionInitSome),
            signature: hir::CallSignature {
                params: vec![hir::Type::u(64), hir::Type::u(64), hir::Type::u(64)],
                returns: vec![],
                effect_class: hir::EffectClass::Barrier,
                domain_effects: vec![hir::DomainEffect {
                    domain: "output".to_owned(),
                    access: hir::DomainAccess::Mutate,
                }],
                control: hir::ControlTransfer::Returns,
                capabilities: vec!["runtime.option".to_owned()],
                safety: hir::CallSafety::OpaqueHost,
            },
            docs: Some(
                "Initialize an Option destination with Some(payload) via its vtable.".to_owned(),
            ),
        };
        let callable_id = self.module.add_callable(callable);
        self.callables_by_name.insert(NAME, callable_id);
        callable_id
    }

    fn ensure_runtime_alloc_persistent(&mut self) -> hir::CallableId {
        const NAME: &str = "runtime.alloc_persistent";
        if let Some(existing) = self.callables_by_name.get(NAME).copied() {
            return existing;
        }
        let callable = hir::CallableSpec {
            kind: hir::CallableKind::Host,
            name: NAME.to_owned(),
            intrinsic: Some(hir::RuntimeIntrinsic::AllocPersistent),
            signature: hir::CallSignature {
                params: vec![hir::Type::u(64), hir::Type::u(64)],
                returns: vec![hir::Type::persistent_addr()],
                effect_class: hir::EffectClass::Mutates,
                domain_effects: vec![hir::DomainEffect {
                    domain: "persistent_heap".to_owned(),
                    access: hir::DomainAccess::Mutate,
                }],
                control: hir::ControlTransfer::MayFail,
                capabilities: vec!["runtime.alloc".to_owned()],
                safety: hir::CallSafety::OpaqueHost,
            },
            docs: Some("Allocate persistent memory that may escape in the result.".to_owned()),
        };
        let callable_id = self.module.add_callable(callable);
        self.callables_by_name.insert(NAME, callable_id);
        callable_id
    }

    fn ensure_runtime_memcpy(&mut self) -> hir::CallableId {
        const NAME: &str = "runtime.memcpy";
        if let Some(existing) = self.callables_by_name.get(NAME).copied() {
            return existing;
        }
        let callable = hir::CallableSpec {
            kind: hir::CallableKind::Host,
            name: NAME.to_owned(),
            intrinsic: Some(hir::RuntimeIntrinsic::Memcpy),
            signature: hir::CallSignature {
                params: vec![hir::Type::u(64), hir::Type::u(64), hir::Type::u(64)],
                returns: vec![hir::Type::u(64)],
                effect_class: hir::EffectClass::Mutates,
                domain_effects: vec![
                    hir::DomainEffect {
                        domain: "persistent_heap".to_owned(),
                        access: hir::DomainAccess::Mutate,
                    },
                    hir::DomainEffect {
                        domain: "input".to_owned(),
                        access: hir::DomainAccess::Read,
                    },
                ],
                control: hir::ControlTransfer::Returns,
                capabilities: vec!["runtime.memcpy".to_owned()],
                safety: hir::CallSafety::OpaqueHost,
            },
            docs: Some("Copy bytes from one address to another.".to_owned()),
        };
        let callable_id = self.module.add_callable(callable);
        self.callables_by_name.insert(NAME, callable_id);
        callable_id
    }

    fn lower_postcard_str_into_place(
        &mut self,
        statements: &mut Vec<hir::Stmt>,
        cursor_local: hir::LocalId,
        place: hir::Place,
    ) {
        let len_local = self.alloc_local(
            format!("str_len_{}", self.locals.len()),
            hir::Type::u(32),
            hir::LocalKind::Temp,
        );
        let data_local = self.alloc_local(
            format!("str_data_{}", self.locals.len()),
            hir::Type::u(64),
            hir::LocalKind::Temp,
        );
        self.lower_postcard_varint_into_place(
            statements,
            cursor_local,
            hir::Place::Local(len_local),
            32,
            false,
        );

        self.push_cursor_bounds_check_expr(
            statements,
            cursor_local,
            hir::Expr::Local(len_local),
            hir::ErrorCode::UnexpectedEof,
        );

        let bytes = self.cursor_bytes_expr(cursor_local);
        let pos = self.cursor_pos_expr(cursor_local);
        let data_expr = hir::Expr::Binary {
            op: hir::BinaryOp::Add,
            lhs: Box::new(hir::Expr::SliceData {
                value: Box::new(bytes),
            }),
            rhs: Box::new(pos.clone()),
        };
        self.push_init(statements, hir::Place::Local(data_local), data_expr);

        let validate_utf8 = self.ensure_runtime_validate_utf8_range();
        statements.push(hir::Stmt {
            id: self.next_stmt_id(),
            kind: hir::StmtKind::Expr(hir::Expr::Call(hir::CallExpr {
                target: hir::CallTarget::Callable(validate_utf8),
                args: vec![hir::Expr::Local(data_local), hir::Expr::Local(len_local)],
            })),
        });

        self.push_cursor_pos_update(
            statements,
            cursor_local,
            hir::Expr::Binary {
                op: hir::BinaryOp::Add,
                lhs: Box::new(pos),
                rhs: Box::new(hir::Expr::Local(len_local)),
            },
        );

        self.push_init(
            statements,
            place,
            hir::Expr::Str {
                data: Box::new(hir::Expr::Local(data_local)),
                len: Box::new(hir::Expr::Local(len_local)),
            },
        );
    }

    fn lower_postcard_owned_string_into_place(
        &mut self,
        statements: &mut Vec<hir::Stmt>,
        cursor_local: hir::LocalId,
        place: hir::Place,
    ) {
        let len_local = self.alloc_local(
            format!("string_len_{}", self.locals.len()),
            hir::Type::u(32),
            hir::LocalKind::Temp,
        );
        let data_local = self.alloc_local(
            format!("string_data_{}", self.locals.len()),
            hir::Type::u(64),
            hir::LocalKind::Temp,
        );
        let ptr_local = self.alloc_local(
            format!("string_ptr_{}", self.locals.len()),
            hir::Type::persistent_addr(),
            hir::LocalKind::Temp,
        );
        let string_raw_type = self.ensure_string_raw_type();
        let raw_string_local = self.alloc_local(
            format!("string_raw_{}", self.locals.len()),
            hir::Type::named(string_raw_type, Vec::new()),
            hir::LocalKind::Temp,
        );

        self.lower_postcard_varint_into_place(
            statements,
            cursor_local,
            hir::Place::Local(len_local),
            32,
            false,
        );
        self.push_cursor_bounds_check_expr(
            statements,
            cursor_local,
            hir::Expr::Local(len_local),
            hir::ErrorCode::UnexpectedEof,
        );

        let bytes = self.cursor_bytes_expr(cursor_local);
        let pos = self.cursor_pos_expr(cursor_local);
        let data_expr = hir::Expr::Binary {
            op: hir::BinaryOp::Add,
            lhs: Box::new(hir::Expr::SliceData {
                value: Box::new(bytes),
            }),
            rhs: Box::new(pos.clone()),
        };
        self.push_init(statements, hir::Place::Local(data_local), data_expr);
        let validate_utf8 = self.ensure_runtime_validate_utf8_range();
        statements.push(hir::Stmt {
            id: self.next_stmt_id(),
            kind: hir::StmtKind::Expr(hir::Expr::Call(hir::CallExpr {
                target: hir::CallTarget::Callable(validate_utf8),
                args: vec![hir::Expr::Local(data_local), hir::Expr::Local(len_local)],
            })),
        });

        let alloc_persistent = self.ensure_runtime_alloc_persistent();
        let memcpy = self.ensure_runtime_memcpy();
        statements.push(hir::Stmt {
            id: self.next_stmt_id(),
            kind: hir::StmtKind::If {
                condition: hir::Expr::Binary {
                    op: hir::BinaryOp::Eq,
                    lhs: Box::new(hir::Expr::Local(len_local)),
                    rhs: Box::new(hir::Expr::Literal(hir::Literal::Integer(0))),
                },
                then_block: hir::Block {
                    scope: hir::ScopeId::new(0),
                    statements: vec![hir::Stmt {
                        id: self.next_stmt_id(),
                        kind: hir::StmtKind::Init {
                            place: hir::Place::Local(ptr_local),
                            value: hir::Expr::Literal(hir::Literal::Integer(1)),
                        },
                    }],
                },
                else_block: Some(hir::Block {
                    scope: hir::ScopeId::new(0),
                    statements: vec![
                        hir::Stmt {
                            id: self.next_stmt_id(),
                            kind: hir::StmtKind::Init {
                                place: hir::Place::Local(ptr_local),
                                value: hir::Expr::Call(hir::CallExpr {
                                    target: hir::CallTarget::Callable(alloc_persistent),
                                    args: vec![
                                        hir::Expr::Local(len_local),
                                        hir::Expr::Literal(hir::Literal::Integer(1)),
                                    ],
                                }),
                            },
                        },
                        hir::Stmt {
                            id: self.next_stmt_id(),
                            kind: hir::StmtKind::Expr(hir::Expr::Call(hir::CallExpr {
                                target: hir::CallTarget::Callable(memcpy),
                                args: vec![
                                    hir::Expr::Local(ptr_local),
                                    hir::Expr::Local(data_local),
                                    hir::Expr::Local(len_local),
                                ],
                            })),
                        },
                    ],
                }),
            },
        });

        self.push_cursor_pos_update(
            statements,
            cursor_local,
            hir::Expr::Binary {
                op: hir::BinaryOp::Add,
                lhs: Box::new(pos),
                rhs: Box::new(hir::Expr::Local(len_local)),
            },
        );

        self.push_init(
            statements,
            hir::Place::Field {
                base: Box::new(hir::Place::Local(raw_string_local)),
                field: "ptr".to_owned(),
            },
            hir::Expr::Local(ptr_local),
        );
        self.push_init(
            statements,
            hir::Place::Field {
                base: Box::new(hir::Place::Local(raw_string_local)),
                field: "len".to_owned(),
            },
            hir::Expr::Local(len_local),
        );
        self.push_init(
            statements,
            hir::Place::Field {
                base: Box::new(hir::Place::Local(raw_string_local)),
                field: "cap".to_owned(),
            },
            hir::Expr::Local(len_local),
        );
        self.push_init(statements, place, hir::Expr::Local(raw_string_local));
    }

    fn push_store(
        &mut self,
        statements: &mut Vec<hir::Stmt>,
        addr: hir::Expr,
        width: hir::MemoryWidth,
        value: hir::Expr,
    ) {
        statements.push(hir::Stmt {
            id: self.next_stmt_id(),
            kind: hir::StmtKind::Store { addr, width, value },
        });
    }

    fn add_addr_offset(base: hir::Expr, offset: usize) -> hir::Expr {
        if offset == 0 {
            base
        } else {
            hir::Expr::Binary {
                op: hir::BinaryOp::Add,
                lhs: Box::new(base),
                rhs: Box::new(hir::Expr::Literal(hir::Literal::Integer(offset as u64))),
            }
        }
    }

    fn store_expr_into_addr(
        &mut self,
        statements: &mut Vec<hir::Stmt>,
        base_addr: hir::Expr,
        shape: &'static Shape,
        value: hir::Expr,
    ) {
        if is_unit(shape) {
            return;
        }

        if shape.is_transparent() {
            let (fields, skipped) = collect_fields(shape);
            assert!(
                skipped.is_empty() && fields.len() == 1,
                "transparent HIR store expects one lowered field"
            );
            self.store_expr_into_addr(
                statements,
                base_addr,
                fields[0].shape,
                hir::Expr::Field {
                    base: Box::new(value),
                    field: fields[0].name.to_owned(),
                },
            );
            return;
        }

        if let Def::Array(array_def) = &shape.def {
            let elem_layout = array_def
                .t
                .layout
                .sized_layout()
                .expect("array element must be Sized");
            for index in 0..array_def.n {
                self.store_expr_into_addr(
                    statements,
                    Self::add_addr_offset(base_addr.clone(), index * elem_layout.size()),
                    array_def.t,
                    hir::Expr::Index {
                        base: Box::new(value.clone()),
                        index: Box::new(hir::Expr::Literal(hir::Literal::Integer(index as u64))),
                    },
                );
            }
            return;
        }

        if let Some(st) = shape.scalar_type() {
            match st {
                ScalarType::Bool | ScalarType::U8 | ScalarType::I8 => {
                    self.push_store(statements, base_addr, hir::MemoryWidth::W1, value);
                    return;
                }
                ScalarType::U16 | ScalarType::I16 => {
                    self.push_store(statements, base_addr, hir::MemoryWidth::W2, value);
                    return;
                }
                ScalarType::U32 | ScalarType::I32 | ScalarType::F32 | ScalarType::Char => {
                    self.push_store(statements, base_addr, hir::MemoryWidth::W4, value);
                    return;
                }
                ScalarType::U64
                | ScalarType::I64
                | ScalarType::USize
                | ScalarType::ISize
                | ScalarType::F64 => {
                    self.push_store(statements, base_addr, hir::MemoryWidth::W8, value);
                    return;
                }
                ScalarType::U128 | ScalarType::I128 => {
                    self.push_store(
                        statements,
                        base_addr.clone(),
                        hir::MemoryWidth::W8,
                        hir::Expr::Field {
                            base: Box::new(value.clone()),
                            field: "lo".to_owned(),
                        },
                    );
                    self.push_store(
                        statements,
                        Self::add_addr_offset(base_addr, 8),
                        hir::MemoryWidth::W8,
                        hir::Expr::Field {
                            base: Box::new(value),
                            field: "hi".to_owned(),
                        },
                    );
                    return;
                }
                ScalarType::Str => {
                    self.push_store(
                        statements,
                        base_addr.clone(),
                        hir::MemoryWidth::W8,
                        hir::Expr::SliceData {
                            value: Box::new(value.clone()),
                        },
                    );
                    self.push_store(
                        statements,
                        Self::add_addr_offset(base_addr, 8),
                        hir::MemoryWidth::W8,
                        hir::Expr::SliceLen {
                            value: Box::new(value),
                        },
                    );
                    return;
                }
                ScalarType::String => {
                    let offsets = kajit_malum::discover_string_offsets();
                    for (offset, field) in [
                        (offsets.ptr_offset as usize, "ptr"),
                        (offsets.len_offset as usize, "len"),
                        (offsets.cap_offset as usize, "cap"),
                    ] {
                        self.push_store(
                            statements,
                            Self::add_addr_offset(base_addr.clone(), offset),
                            hir::MemoryWidth::W8,
                            hir::Expr::Field {
                                base: Box::new(value.clone()),
                                field: field.to_owned(),
                            },
                        );
                    }
                    return;
                }
                _ => panic!(
                    "postcard HIR store into addr does not support scalar {st:?} yet for {}",
                    shape.type_identifier
                ),
            }
        }

        match &shape.ty {
            Type::User(UserType::Struct(_)) => {
                let (fields, skipped) = collect_fields(shape);
                assert!(
                    skipped.is_empty(),
                    "postcard HIR addr-store does not support skipped/defaulted fields"
                );
                for field in fields {
                    self.store_expr_into_addr(
                        statements,
                        Self::add_addr_offset(base_addr.clone(), field.offset),
                        field.shape,
                        hir::Expr::Field {
                            base: Box::new(value.clone()),
                            field: field.name.to_owned(),
                        },
                    );
                }
            }
            _ => panic!(
                "postcard HIR addr-store does not support shape {} yet",
                shape.type_identifier
            ),
        }
    }

    fn lower_postcard_list_into_place(
        &mut self,
        statements: &mut Vec<hir::Stmt>,
        cursor_local: hir::LocalId,
        place: hir::Place,
        list_def: &ListDef,
        shape: &'static Shape,
    ) {
        let elem_layout = list_def
            .t
            .layout
            .sized_layout()
            .expect("postcard HIR list element must be Sized");
        let len_local = self.alloc_local(
            format!("list_len_{}", self.locals.len()),
            hir::Type::u(64),
            hir::LocalKind::Temp,
        );
        let bytes_local = self.alloc_local(
            format!("list_bytes_{}", self.locals.len()),
            hir::Type::u(64),
            hir::LocalKind::Temp,
        );
        let ptr_local = self.alloc_local(
            format!("list_ptr_{}", self.locals.len()),
            hir::Type::persistent_addr(),
            hir::LocalKind::Temp,
        );
        let index_local = self.alloc_local(
            format!("list_index_{}", self.locals.len()),
            hir::Type::u(64),
            hir::LocalKind::Temp,
        );
        let elem_ty = self.lower_type(list_def.t);
        let elem_local = self.alloc_local(
            format!("list_elem_{}", self.locals.len()),
            elem_ty,
            hir::LocalKind::Temp,
        );

        self.lower_postcard_varint_into_place(
            statements,
            cursor_local,
            hir::Place::Local(len_local),
            64,
            false,
        );
        self.push_init(
            statements,
            hir::Place::Local(bytes_local),
            hir::Expr::Binary {
                op: hir::BinaryOp::Mul,
                lhs: Box::new(hir::Expr::Local(len_local)),
                rhs: Box::new(hir::Expr::Literal(hir::Literal::Integer(
                    elem_layout.size() as u64,
                ))),
            },
        );
        let alloc_persistent = self.ensure_runtime_alloc_persistent();
        self.push_init(
            statements,
            hir::Place::Local(ptr_local),
            hir::Expr::Call(hir::CallExpr {
                target: hir::CallTarget::Callable(alloc_persistent),
                args: vec![
                    hir::Expr::Local(bytes_local),
                    hir::Expr::Literal(hir::Literal::Integer(elem_layout.align() as u64)),
                ],
            }),
        );
        self.push_init(
            statements,
            hir::Place::Local(index_local),
            hir::Expr::Literal(hir::Literal::Integer(0)),
        );

        let mut loop_body = Vec::new();
        loop_body.push(hir::Stmt {
            id: self.next_stmt_id(),
            kind: hir::StmtKind::If {
                condition: hir::Expr::Binary {
                    op: hir::BinaryOp::Eq,
                    lhs: Box::new(hir::Expr::Local(index_local)),
                    rhs: Box::new(hir::Expr::Local(len_local)),
                },
                then_block: hir::Block {
                    scope: hir::ScopeId::new(0),
                    statements: vec![hir::Stmt {
                        id: self.next_stmt_id(),
                        kind: hir::StmtKind::Break,
                    }],
                },
                else_block: Some(hir::Block {
                    scope: hir::ScopeId::new(0),
                    statements: Vec::new(),
                }),
            },
        });
        self.lower_shape_into_place(
            &mut loop_body,
            cursor_local,
            hir::Place::Local(elem_local),
            list_def.t,
        );
        let elem_addr = hir::Expr::Binary {
            op: hir::BinaryOp::Add,
            lhs: Box::new(hir::Expr::Local(ptr_local)),
            rhs: Box::new(hir::Expr::Binary {
                op: hir::BinaryOp::Mul,
                lhs: Box::new(hir::Expr::Local(index_local)),
                rhs: Box::new(hir::Expr::Literal(hir::Literal::Integer(
                    elem_layout.size() as u64,
                ))),
            }),
        };
        self.store_expr_into_addr(
            &mut loop_body,
            elem_addr,
            list_def.t,
            hir::Expr::Local(elem_local),
        );
        loop_body.push(hir::Stmt {
            id: self.next_stmt_id(),
            kind: hir::StmtKind::Assign {
                place: hir::Place::Local(index_local),
                value: hir::Expr::Binary {
                    op: hir::BinaryOp::Add,
                    lhs: Box::new(hir::Expr::Local(index_local)),
                    rhs: Box::new(hir::Expr::Literal(hir::Literal::Integer(1))),
                },
            },
        });
        statements.push(hir::Stmt {
            id: self.next_stmt_id(),
            kind: hir::StmtKind::Loop {
                body: hir::Block {
                    scope: hir::ScopeId::new(0),
                    statements: loop_body,
                },
                max_iterations: None,
            },
        });

        let offsets = kajit_malum::discover_vec_offsets(list_def, shape);
        let mut vec_fields = [
            ("ptr", offsets.ptr_offset),
            ("len", offsets.len_offset),
            ("cap", offsets.cap_offset),
        ];
        vec_fields.sort_by_key(|(_, offset)| *offset);

        for (field, _) in vec_fields {
            match field {
                "ptr" => self.push_init(
                    statements,
                    self.field_place(place.clone(), "ptr"),
                    hir::Expr::Binary {
                        op: hir::BinaryOp::Add,
                        lhs: Box::new(hir::Expr::Local(ptr_local)),
                        rhs: Box::new(hir::Expr::Binary {
                            op: hir::BinaryOp::Mul,
                            lhs: Box::new(hir::Expr::Binary {
                                op: hir::BinaryOp::Eq,
                                lhs: Box::new(hir::Expr::Local(len_local)),
                                rhs: Box::new(hir::Expr::Literal(hir::Literal::Integer(0))),
                            }),
                            rhs: Box::new(hir::Expr::Literal(hir::Literal::Integer(
                                elem_layout.align() as u64,
                            ))),
                        }),
                    },
                ),
                "len" | "cap" => self.push_init(
                    statements,
                    self.field_place(place.clone(), field),
                    hir::Expr::Local(len_local),
                ),
                _ => unreachable!(),
            }
        }
    }

    fn postcard_varint_max_bytes(bits: u32) -> u64 {
        match bits {
            16 => 3,
            32 => 5,
            64 => 10,
            128 => 19,
            _ => panic!("unsupported postcard HIR varint width {bits}"),
        }
    }

    fn postcard_varint_finish_expr(&self, acc_local: hir::LocalId, zigzag: bool) -> hir::Expr {
        let acc = hir::Expr::Local(acc_local);
        if !zigzag {
            return acc;
        }
        // ZigZag decoding: (n >> 1) ^ (-(n & 1))
        // Optimized to: (n >> 1) ^ ((n << 63) sar 63)
        // This uses arithmetic shift right to sign-extend the LSB to all 64 bits.
        let shifted = hir::Expr::Binary {
            op: hir::BinaryOp::Shr,
            lhs: Box::new(acc.clone()),
            rhs: Box::new(hir::Expr::Literal(hir::Literal::Integer(1))),
        };
        let sign_extended = hir::Expr::Binary {
            op: hir::BinaryOp::Sar,
            lhs: Box::new(hir::Expr::Binary {
                op: hir::BinaryOp::Shl,
                lhs: Box::new(acc),
                rhs: Box::new(hir::Expr::Literal(hir::Literal::Integer(63))),
            }),
            rhs: Box::new(hir::Expr::Literal(hir::Literal::Integer(63))),
        };
        hir::Expr::Binary {
            op: hir::BinaryOp::Xor,
            lhs: Box::new(shifted),
            rhs: Box::new(sign_extended),
        }
    }

    #[allow(dead_code)]
    fn postcard_varint_finish_block(
        &mut self,
        place: hir::Place,
        bits: u32,
        zigzag: bool,
        acc_local: hir::LocalId,
        byte_index: u64,
        raw_local: hir::LocalId,
    ) -> hir::Block {
        let mut block = hir::Block {
            scope: hir::ScopeId::new(0),
            statements: Vec::new(),
        };

        if bits == 64 && byte_index + 1 == Self::postcard_varint_max_bytes(bits) {
            let extra_bits = hir::Expr::Binary {
                op: hir::BinaryOp::BitAnd,
                lhs: Box::new(hir::Expr::Local(raw_local)),
                rhs: Box::new(hir::Expr::Literal(hir::Literal::Integer(0x7e))),
            };
            block.statements.push(hir::Stmt {
                id: self.next_stmt_id(),
                kind: hir::StmtKind::If {
                    condition: hir::Expr::Binary {
                        op: hir::BinaryOp::Ne,
                        lhs: Box::new(extra_bits),
                        rhs: Box::new(hir::Expr::Literal(hir::Literal::Integer(0))),
                    },
                    then_block: hir::Block {
                        scope: hir::ScopeId::new(0),
                        statements: vec![hir::Stmt {
                            id: self.next_stmt_id(),
                            kind: hir::StmtKind::Fail {
                                code: hir::ErrorCode::InvalidVarint,
                            },
                        }],
                    },
                    else_block: Some(hir::Block {
                        scope: hir::ScopeId::new(0),
                        statements: vec![hir::Stmt {
                            id: self.next_stmt_id(),
                            kind: hir::StmtKind::Init {
                                place,
                                value: self.postcard_varint_finish_expr(acc_local, zigzag),
                            },
                        }],
                    }),
                },
            });
            return block;
        }

        if bits < 64 {
            let upper = hir::Expr::Binary {
                op: hir::BinaryOp::Shr,
                lhs: Box::new(hir::Expr::Local(acc_local)),
                rhs: Box::new(hir::Expr::Literal(hir::Literal::Integer(bits as u64))),
            };
            block.statements.push(hir::Stmt {
                id: self.next_stmt_id(),
                kind: hir::StmtKind::If {
                    condition: hir::Expr::Binary {
                        op: hir::BinaryOp::Ne,
                        lhs: Box::new(upper),
                        rhs: Box::new(hir::Expr::Literal(hir::Literal::Integer(0))),
                    },
                    then_block: hir::Block {
                        scope: hir::ScopeId::new(0),
                        statements: vec![hir::Stmt {
                            id: self.next_stmt_id(),
                            kind: hir::StmtKind::Fail {
                                code: hir::ErrorCode::NumberOutOfRange,
                            },
                        }],
                    },
                    else_block: Some(hir::Block {
                        scope: hir::ScopeId::new(0),
                        statements: vec![hir::Stmt {
                            id: self.next_stmt_id(),
                            kind: hir::StmtKind::Init {
                                place,
                                value: self.postcard_varint_finish_expr(acc_local, zigzag),
                            },
                        }],
                    }),
                },
            });
            return block;
        }

        self.push_init(
            &mut block.statements,
            place,
            self.postcard_varint_finish_expr(acc_local, zigzag),
        );
        block
    }

    fn lower_postcard_varint_step(
        &mut self,
        statements: &mut Vec<hir::Stmt>,
        cursor_local: hir::LocalId,
        place: hir::Place,
        bits: u32,
        zigzag: bool,
        acc_local: hir::LocalId,
    ) {
        let max_bytes = Self::postcard_varint_max_bytes(bits);

        // shift tracks the bit shift amount (0, 7, 14, 21, ...)
        // This replaces the byte_index * 7 multiplication with direct accumulation.
        let shift_local = self.alloc_local(
            format!("varint_shift_{}", self.locals.len()),
            hir::Type::u(64),
            hir::LocalKind::Temp,
        );
        self.push_init(
            statements,
            hir::Place::Local(shift_local),
            hir::Expr::Literal(hir::Literal::Integer(0)),
        );

        // raw_byte holds the current byte being processed
        let raw_local = self.alloc_local(
            format!("varint_byte_{}", self.locals.len()),
            hir::Type::u(8),
            hir::LocalKind::Temp,
        );
        self.push_init(
            statements,
            hir::Place::Local(raw_local),
            hir::Expr::Literal(hir::Literal::Integer(0)),
        );

        let mut loop_body = Vec::new();

        // bounds check: pos + 1 > len → fail
        self.push_cursor_bounds_check(
            &mut loop_body,
            cursor_local,
            1,
            hir::ErrorCode::UnexpectedEof,
        );

        // load byte from cursor
        let pos = self.cursor_pos_expr(cursor_local);
        let addr = hir::Expr::Binary {
            op: hir::BinaryOp::Add,
            lhs: Box::new(hir::Expr::SliceData {
                value: Box::new(self.cursor_bytes_expr(cursor_local)),
            }),
            rhs: Box::new(pos.clone()),
        };
        loop_body.push(hir::Stmt {
            id: self.next_stmt_id(),
            kind: hir::StmtKind::Assign {
                place: hir::Place::Local(raw_local),
                value: hir::Expr::Load {
                    addr: Box::new(addr),
                    width: hir::MemoryWidth::W1,
                },
            },
        });

        // pos++
        self.push_cursor_pos_update(
            &mut loop_body,
            cursor_local,
            hir::Expr::Binary {
                op: hir::BinaryOp::Add,
                lhs: Box::new(pos),
                rhs: Box::new(hir::Expr::Literal(hir::Literal::Integer(1))),
            },
        );

        // acc |= (byte & 0x7f) << shift
        let low = hir::Expr::Binary {
            op: hir::BinaryOp::BitAnd,
            lhs: Box::new(hir::Expr::Local(raw_local)),
            rhs: Box::new(hir::Expr::Literal(hir::Literal::Integer(0x7f))),
        };
        let part = hir::Expr::Binary {
            op: hir::BinaryOp::Shl,
            lhs: Box::new(low),
            rhs: Box::new(hir::Expr::Local(shift_local)),
        };
        loop_body.push(hir::Stmt {
            id: self.next_stmt_id(),
            kind: hir::StmtKind::Assign {
                place: hir::Place::Local(acc_local),
                value: hir::Expr::Binary {
                    op: hir::BinaryOp::BitOr,
                    lhs: Box::new(hir::Expr::Local(acc_local)),
                    rhs: Box::new(part),
                },
            },
        });

        // shift += 7
        loop_body.push(hir::Stmt {
            id: self.next_stmt_id(),
            kind: hir::StmtKind::Assign {
                place: hir::Place::Local(shift_local),
                value: hir::Expr::Binary {
                    op: hir::BinaryOp::Add,
                    lhs: Box::new(hir::Expr::Local(shift_local)),
                    rhs: Box::new(hir::Expr::Literal(hir::Literal::Integer(7))),
                },
            },
        });

        // if !(byte & 0x80) { ... } - no continuation bit means we're done
        // For 64-bit: if byte_index == max_bytes, check for extra bits in last byte
        let no_cont_body = if bits == 64 {
            // if byte_index == max_bytes && (byte & 0x7e) != 0 { fail InvalidVarint }
            // else { break }
            vec![hir::Stmt {
                id: self.next_stmt_id(),
                kind: hir::StmtKind::If {
                    condition: hir::Expr::Binary {
                        op: hir::BinaryOp::Eq,
                        lhs: Box::new(hir::Expr::Local(shift_local)),
                        rhs: Box::new(hir::Expr::Literal(hir::Literal::Integer(max_bytes * 7))),
                    },
                    then_block: hir::Block {
                        scope: hir::ScopeId::new(0),
                        statements: vec![hir::Stmt {
                            id: self.next_stmt_id(),
                            kind: hir::StmtKind::If {
                                condition: hir::Expr::Binary {
                                    op: hir::BinaryOp::Ne,
                                    lhs: Box::new(hir::Expr::Binary {
                                        op: hir::BinaryOp::BitAnd,
                                        lhs: Box::new(hir::Expr::Local(raw_local)),
                                        rhs: Box::new(hir::Expr::Literal(hir::Literal::Integer(
                                            0x7e,
                                        ))),
                                    }),
                                    rhs: Box::new(hir::Expr::Literal(hir::Literal::Integer(0))),
                                },
                                then_block: hir::Block {
                                    scope: hir::ScopeId::new(0),
                                    statements: vec![hir::Stmt {
                                        id: self.next_stmt_id(),
                                        kind: hir::StmtKind::Fail {
                                            code: hir::ErrorCode::InvalidVarint,
                                        },
                                    }],
                                },
                                else_block: Some(hir::Block {
                                    scope: hir::ScopeId::new(0),
                                    statements: vec![hir::Stmt {
                                        id: self.next_stmt_id(),
                                        kind: hir::StmtKind::Break,
                                    }],
                                }),
                            },
                        }],
                    },
                    else_block: Some(hir::Block {
                        scope: hir::ScopeId::new(0),
                        statements: vec![hir::Stmt {
                            id: self.next_stmt_id(),
                            kind: hir::StmtKind::Break,
                        }],
                    }),
                },
            }]
        } else {
            vec![hir::Stmt {
                id: self.next_stmt_id(),
                kind: hir::StmtKind::Break,
            }]
        };

        loop_body.push(hir::Stmt {
            id: self.next_stmt_id(),
            kind: hir::StmtKind::If {
                condition: hir::Expr::Binary {
                    op: hir::BinaryOp::Eq,
                    lhs: Box::new(hir::Expr::Binary {
                        op: hir::BinaryOp::BitAnd,
                        lhs: Box::new(hir::Expr::Local(raw_local)),
                        rhs: Box::new(hir::Expr::Literal(hir::Literal::Integer(0x80))),
                    }),
                    rhs: Box::new(hir::Expr::Literal(hir::Literal::Integer(0))),
                },
                then_block: hir::Block {
                    scope: hir::ScopeId::new(0),
                    statements: no_cont_body,
                },
                else_block: Some(hir::Block {
                    scope: hir::ScopeId::new(0),
                    statements: Vec::new(),
                }),
            },
        });

        // if shift == max_bytes * 7 { fail InvalidVarint } - can't continue past max
        loop_body.push(hir::Stmt {
            id: self.next_stmt_id(),
            kind: hir::StmtKind::If {
                condition: hir::Expr::Binary {
                    op: hir::BinaryOp::Eq,
                    lhs: Box::new(hir::Expr::Local(shift_local)),
                    rhs: Box::new(hir::Expr::Literal(hir::Literal::Integer(max_bytes * 7))),
                },
                then_block: hir::Block {
                    scope: hir::ScopeId::new(0),
                    statements: vec![hir::Stmt {
                        id: self.next_stmt_id(),
                        kind: hir::StmtKind::Fail {
                            code: hir::ErrorCode::InvalidVarint,
                        },
                    }],
                },
                else_block: Some(hir::Block {
                    scope: hir::ScopeId::new(0),
                    statements: Vec::new(),
                }),
            },
        });

        // Push the loop — bounded by max varint byte count for this type
        statements.push(hir::Stmt {
            id: self.next_stmt_id(),
            kind: hir::StmtKind::Loop {
                body: hir::Block {
                    scope: hir::ScopeId::new(0),
                    statements: loop_body,
                },
                max_iterations: Some(max_bytes as u32),
            },
        });

        // After loop: overflow checks for <64-bit types
        // (64-bit extra bits check is done inside the loop when exiting)

        // For <64-bit: if acc >> bits != 0 { fail NumberOutOfRange }
        if bits < 64 {
            statements.push(hir::Stmt {
                id: self.next_stmt_id(),
                kind: hir::StmtKind::If {
                    condition: hir::Expr::Binary {
                        op: hir::BinaryOp::Ne,
                        lhs: Box::new(hir::Expr::Binary {
                            op: hir::BinaryOp::Shr,
                            lhs: Box::new(hir::Expr::Local(acc_local)),
                            rhs: Box::new(hir::Expr::Literal(hir::Literal::Integer(bits as u64))),
                        }),
                        rhs: Box::new(hir::Expr::Literal(hir::Literal::Integer(0))),
                    },
                    then_block: hir::Block {
                        scope: hir::ScopeId::new(0),
                        statements: vec![hir::Stmt {
                            id: self.next_stmt_id(),
                            kind: hir::StmtKind::Fail {
                                code: hir::ErrorCode::NumberOutOfRange,
                            },
                        }],
                    },
                    else_block: Some(hir::Block {
                        scope: hir::ScopeId::new(0),
                        statements: Vec::new(),
                    }),
                },
            });
        }

        // Store result
        self.push_init(
            statements,
            place,
            self.postcard_varint_finish_expr(acc_local, zigzag),
        );
    }

    fn lower_postcard_varint_into_place(
        &mut self,
        statements: &mut Vec<hir::Stmt>,
        cursor_local: hir::LocalId,
        place: hir::Place,
        bits: u32,
        zigzag: bool,
    ) {
        let acc_local = self.alloc_local(
            format!("varint_acc_{}", self.locals.len()),
            hir::Type::u(64),
            hir::LocalKind::Temp,
        );
        self.push_init(
            statements,
            hir::Place::Local(acc_local),
            hir::Expr::Literal(hir::Literal::Integer(0)),
        );
        self.lower_postcard_varint_step(statements, cursor_local, place, bits, zigzag, acc_local);
    }

    pub fn lower_shape_into_place(
        &mut self,
        statements: &mut Vec<hir::Stmt>,
        cursor_local: hir::LocalId,
        place: hir::Place,
        shape: &'static Shape,
    ) {
        if is_unit(shape) {
            return;
        }

        if let Def::List(list_def) = &shape.def {
            self.lower_postcard_list_into_place(statements, cursor_local, place, list_def, shape);
            return;
        }

        if shape.is_transparent() {
            let (fields, skipped) = collect_fields(shape);
            assert!(
                skipped.is_empty() && fields.len() == 1,
                "transparent HIR prototype expects one lowered field"
            );
            self.lower_shape_into_place(statements, cursor_local, place, fields[0].shape);
            return;
        }

        if let Def::Array(array_def) = &shape.def {
            let counter_local = self.alloc_local(
                format!("array_idx_{}", self.locals.len()),
                hir::Type::u(64),
                hir::LocalKind::Temp,
            );
            let scope = hir::ScopeId::new(0);

            // init counter = 0
            statements.push(hir::Stmt {
                id: self.next_stmt_id(),
                kind: hir::StmtKind::Init {
                    place: hir::Place::Local(counter_local),
                    value: hir::Expr::Literal(hir::Literal::Integer(0)),
                },
            });

            // loop { if counter >= N { break } ... counter += 1 }
            let mut body_stmts = Vec::new();

            // break condition: if counter >= array_len { break }
            body_stmts.push(hir::Stmt {
                id: self.next_stmt_id(),
                kind: hir::StmtKind::If {
                    condition: hir::Expr::Binary {
                        op: hir::BinaryOp::Ge,
                        lhs: Box::new(hir::Expr::Local(counter_local)),
                        rhs: Box::new(hir::Expr::Literal(hir::Literal::Integer(
                            array_def.n as u64,
                        ))),
                    },
                    then_block: hir::Block {
                        scope,
                        statements: vec![hir::Stmt {
                            id: self.next_stmt_id(),
                            kind: hir::StmtKind::Break,
                        }],
                    },
                    else_block: Some(hir::Block {
                        scope,
                        statements: Vec::new(),
                    }),
                },
            });

            // decode element at place[counter]
            let elem_place = hir::Place::Index {
                base: Box::new(place.clone()),
                index: Box::new(hir::Expr::Local(counter_local)),
            };
            self.lower_shape_into_place(&mut body_stmts, cursor_local, elem_place, array_def.t);

            // counter += 1
            body_stmts.push(hir::Stmt {
                id: self.next_stmt_id(),
                kind: hir::StmtKind::Assign {
                    place: hir::Place::Local(counter_local),
                    value: hir::Expr::Binary {
                        op: hir::BinaryOp::Add,
                        lhs: Box::new(hir::Expr::Local(counter_local)),
                        rhs: Box::new(hir::Expr::Literal(hir::Literal::Integer(1))),
                    },
                },
            });

            statements.push(hir::Stmt {
                id: self.next_stmt_id(),
                kind: hir::StmtKind::Loop {
                    body: hir::Block {
                        scope,
                        statements: body_stmts,
                    },
                    max_iterations: Some(array_def.n as u32),
                },
            });
            return;
        }

        if let Some(opt_def) = get_option_def(shape) {
            let tag_local = self.alloc_local(
                format!("option_is_some_{}", self.locals.len()),
                hir::Type::bool(),
                hir::LocalKind::Temp,
            );
            let option_init_none = self.ensure_runtime_option_init_none();
            let option_init_some = self.ensure_runtime_option_init_some();
            self.lower_postcard_option_tag_into_local(statements, cursor_local, tag_local);

            let mut then_block = hir::Block {
                scope: hir::ScopeId::new(0),
                statements: Vec::new(),
            };

            if is_unit(opt_def.t) {
                let payload_local = self.alloc_local(
                    format!("option_value_{}", self.locals.len()),
                    hir::Type::unit(),
                    hir::LocalKind::Temp,
                );
                self.push_init(
                    &mut then_block.statements,
                    hir::Place::Local(payload_local),
                    hir::Expr::Literal(hir::Literal::Unit),
                );
                then_block.statements.push(hir::Stmt {
                    id: self.next_stmt_id(),
                    kind: hir::StmtKind::Expr(hir::Expr::Call(hir::CallExpr {
                        target: hir::CallTarget::Callable(option_init_some),
                        args: vec![
                            hir::Expr::Literal(hir::Literal::ExternAddr {
                                symbol: vtable_symbol_name(shape, VtableEntry::OptionInitSome),
                                value: opt_def.vtable.init_some as *const () as usize as u64,
                            }),
                            hir::Expr::AddrOf(Box::new(place.clone())),
                            hir::Expr::AddrOf(Box::new(hir::Place::Local(payload_local))),
                        ],
                    })),
                });
            } else {
                let payload_ty = self.lower_type(opt_def.t);
                let payload_local = self.alloc_local(
                    format!("option_value_{}", self.locals.len()),
                    payload_ty,
                    hir::LocalKind::Temp,
                );
                self.lower_shape_into_place(
                    &mut then_block.statements,
                    cursor_local,
                    hir::Place::Local(payload_local),
                    opt_def.t,
                );
                then_block.statements.push(hir::Stmt {
                    id: self.next_stmt_id(),
                    kind: hir::StmtKind::Expr(hir::Expr::Call(hir::CallExpr {
                        target: hir::CallTarget::Callable(option_init_some),
                        args: vec![
                            hir::Expr::Literal(hir::Literal::ExternAddr {
                                symbol: vtable_symbol_name(shape, VtableEntry::OptionInitSome),
                                value: opt_def.vtable.init_some as *const () as usize as u64,
                            }),
                            hir::Expr::AddrOf(Box::new(place.clone())),
                            hir::Expr::AddrOf(Box::new(hir::Place::Local(payload_local))),
                        ],
                    })),
                });
            }

            let mut else_block = hir::Block {
                scope: hir::ScopeId::new(0),
                statements: Vec::new(),
            };
            else_block.statements.push(hir::Stmt {
                id: self.next_stmt_id(),
                kind: hir::StmtKind::Expr(hir::Expr::Call(hir::CallExpr {
                    target: hir::CallTarget::Callable(option_init_none),
                    args: vec![
                        hir::Expr::Literal(hir::Literal::ExternAddr {
                            symbol: vtable_symbol_name(shape, VtableEntry::OptionInitNone),
                            value: opt_def.vtable.init_none as *const () as usize as u64,
                        }),
                        hir::Expr::AddrOf(Box::new(place)),
                    ],
                })),
            });

            statements.push(hir::Stmt {
                id: self.next_stmt_id(),
                kind: hir::StmtKind::If {
                    condition: hir::Expr::Local(tag_local),
                    then_block,
                    else_block: Some(else_block),
                },
            });
            return;
        }

        if let Some(st) = shape.scalar_type() {
            match st {
                ScalarType::Bool | ScalarType::U8 | ScalarType::I8 => {
                    self.lower_postcard_fixed_width_scalar_into_place(
                        statements,
                        cursor_local,
                        place,
                        st,
                    );
                    return;
                }
                ScalarType::U16 => {
                    self.lower_postcard_varint_into_place(
                        statements,
                        cursor_local,
                        place,
                        16,
                        false,
                    );
                    return;
                }
                ScalarType::U32 => {
                    self.lower_postcard_varint_into_place(
                        statements,
                        cursor_local,
                        place,
                        32,
                        false,
                    );
                    return;
                }
                ScalarType::U64 | ScalarType::USize => {
                    self.lower_postcard_varint_into_place(
                        statements,
                        cursor_local,
                        place,
                        64,
                        false,
                    );
                    return;
                }
                ScalarType::U128 => {
                    self.lower_postcard_varint128_into_place(
                        statements,
                        cursor_local,
                        place,
                        false,
                    );
                    return;
                }
                ScalarType::I16 => {
                    self.lower_postcard_varint_into_place(
                        statements,
                        cursor_local,
                        place,
                        16,
                        true,
                    );
                    return;
                }
                ScalarType::I32 => {
                    self.lower_postcard_varint_into_place(
                        statements,
                        cursor_local,
                        place,
                        32,
                        true,
                    );
                    return;
                }
                ScalarType::I64 | ScalarType::ISize => {
                    self.lower_postcard_varint_into_place(
                        statements,
                        cursor_local,
                        place,
                        64,
                        true,
                    );
                    return;
                }
                ScalarType::I128 => {
                    self.lower_postcard_varint128_into_place(statements, cursor_local, place, true);
                    return;
                }
                ScalarType::F32 | ScalarType::F64 => {
                    self.lower_postcard_fixed_width_scalar_into_place(
                        statements,
                        cursor_local,
                        place,
                        st,
                    );
                    return;
                }
                ScalarType::Char => {
                    self.lower_postcard_char_into_place(statements, cursor_local, place);
                    return;
                }
                ScalarType::Str => {
                    self.lower_postcard_str_into_place(statements, cursor_local, place);
                    return;
                }
                ScalarType::String => {
                    self.lower_postcard_owned_string_into_place(statements, cursor_local, place);
                    return;
                }
                _ => {}
            }
            let callable = self.ensure_postcard_reader(st);
            self.push_init(
                statements,
                place,
                hir::Expr::Call(hir::CallExpr {
                    target: hir::CallTarget::Callable(callable),
                    args: vec![hir::Expr::Local(cursor_local)],
                }),
            );
            return;
        }

        match &shape.ty {
            Type::User(UserType::Struct(_)) => {
                let (fields, skipped) = collect_fields(shape);
                assert!(
                    skipped.is_empty(),
                    "postcard HIR prototype does not support skipped/defaulted fields"
                );
                for field in fields {
                    let field_place = hir::Place::Field {
                        base: Box::new(place.clone()),
                        field: field.name.to_owned(),
                    };
                    self.lower_shape_into_place(statements, cursor_local, field_place, field.shape);
                }
            }
            Type::User(UserType::Enum(enum_type)) => {
                let variants = collect_variants(enum_type);
                let enum_def = self.ensure_type_def(shape);
                let disc_local = self.alloc_local(
                    format!("enum_discriminant_{}", self.locals.len()),
                    hir::Type::u(32),
                    hir::LocalKind::Temp,
                );
                self.lower_postcard_varint_into_place(
                    statements,
                    cursor_local,
                    hir::Place::Local(disc_local),
                    32,
                    false,
                );

                statements.push(hir::Stmt {
                    id: self.next_stmt_id(),
                    kind: hir::StmtKind::Match {
                        scrutinee: hir::Expr::Local(disc_local),
                        arms: variants
                            .into_iter()
                            .map(|variant| hir::MatchArm {
                                pattern: hir::Pattern::Integer(
                                    variant
                                        .rust_discriminant
                                        .try_into()
                                        .expect("enum discriminant must fit in u64"),
                                ),
                                body: hir::Block {
                                    scope: hir::ScopeId::new(0),
                                    statements: {
                                        let mut statements = Vec::new();
                                        let mut variant_fields = Vec::new();
                                        for field in &variant.fields {
                                            if is_unit(field.shape) {
                                                variant_fields.push((
                                                    field.name.to_owned(),
                                                    hir::Expr::Literal(hir::Literal::Unit),
                                                ));
                                                continue;
                                            }
                                            let field_ty = self.lower_type(field.shape);
                                            let field_local = self.alloc_local(
                                                format!(
                                                    "variant_{}_{}_{}",
                                                    variant.name,
                                                    field.name,
                                                    self.locals.len()
                                                ),
                                                field_ty,
                                                hir::LocalKind::Temp,
                                            );
                                            self.lower_shape_into_place(
                                                &mut statements,
                                                cursor_local,
                                                hir::Place::Local(field_local),
                                                field.shape,
                                            );
                                            variant_fields.push((
                                                field.name.to_owned(),
                                                hir::Expr::Local(field_local),
                                            ));
                                        }
                                        statements.push(hir::Stmt {
                                            id: self.next_stmt_id(),
                                            kind: hir::StmtKind::Init {
                                                place: place.clone(),
                                                value: hir::Expr::Variant {
                                                    def: enum_def,
                                                    variant: variant.name.to_owned(),
                                                    fields: variant_fields,
                                                },
                                            },
                                        });
                                        statements
                                    },
                                },
                            })
                            .collect(),
                    },
                });
            }
            _ => panic!(
                "postcard HIR prototype does not support shape {} yet",
                shape.type_identifier
            ),
        }
    }
}

pub fn build_postcard_decoder_hir(shape: &'static Shape) -> hir::Module {
    let mut lowerer = PostcardHirLowerer::new();
    let root_type = lowerer.lower_type(shape);
    let cursor_local = lowerer.next_local();
    let out_local = lowerer.next_local();
    let root_scope = hir::ScopeId::new(0);
    let mut statements = Vec::new();
    lowerer.lower_shape_into_place(
        &mut statements,
        cursor_local,
        hir::Place::Local(out_local),
        shape,
    );
    statements.push(hir::Stmt {
        id: lowerer.next_stmt_id(),
        kind: hir::StmtKind::Return(None),
    });
    let locals = std::mem::take(&mut lowerer.locals);

    lowerer.module.add_function(hir::Function {
        name: format!("decode_{}", shape.type_identifier.replace("::", "_")),
        region_params: vec![lowerer.input_region],
        store_params: Vec::new(),
        params: vec![
            hir::Parameter {
                local: cursor_local,
                name: "cursor".to_owned(),
                ty: hir::Type::mut_ref(hir::Type::named(
                    lowerer.cursor_type,
                    vec![hir::GenericArg::Region(lowerer.input_region)],
                )),
                kind: hir::LocalKind::Param,
            },
            hir::Parameter {
                local: out_local,
                name: "out".to_owned(),
                ty: root_type,
                kind: hir::LocalKind::Destination,
            },
        ],
        locals,
        return_type: hir::Type::unit(),
        scopes: vec![hir::Scope {
            id: root_scope,
            parent: None,
            comment: Some(format!(
                "Postcard prototype HIR for {}",
                shape.type_identifier
            )),
        }],
        body: hir::Block {
            scope: root_scope,
            statements,
        },
    });

    lowerer.finish()
}
pub fn supports_postcard_decoder_hir(shape: &'static Shape) -> bool {
    fn fields_are_supported(shape: &'static Shape) -> bool {
        let (fields, skipped) = collect_fields(shape);
        skipped.is_empty()
            && fields
                .into_iter()
                .all(|field| supports_postcard_decoder_hir(field.shape))
    }

    if is_unit(shape) {
        return true;
    }

    if get_pointer_def(shape).is_some() {
        return false;
    }

    match &shape.def {
        Def::List(list_def) => return supports_postcard_decoder_hir(list_def.t),
        Def::Array(array_def) => return supports_postcard_decoder_hir(array_def.t),
        Def::Option(opt_def) => return supports_postcard_decoder_hir(opt_def.t),
        Def::Map(_) => return false,
        _ => {}
    }

    if shape.is_transparent() {
        let (fields, skipped) = collect_fields(shape);
        return skipped.is_empty()
            && fields.len() == 1
            && supports_postcard_decoder_hir(fields[0].shape);
    }

    if let Some(st) = shape.scalar_type() {
        return !matches!(st, ScalarType::CowStr);
    }

    match &shape.ty {
        Type::User(UserType::Struct(_)) => fields_are_supported(shape),
        Type::User(UserType::Enum(enum_type)) => {
            !matches!(enum_type.enum_repr, EnumRepr::Rust | EnumRepr::RustNPO)
                && collect_variants(enum_type).into_iter().all(|variant| {
                    variant
                        .fields
                        .into_iter()
                        .all(|field| supports_postcard_decoder_hir(field.shape))
                })
        }
        _ => false,
    }
}
