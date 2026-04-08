use crate::hir as kajit_hir;
use crate::hir::Module;

pub fn parse_hir(text: &str) -> Result<Module, String> {
    crate::hir::token_parser::parse_hir_v2(text)
}

#[cfg(test)]
mod tests {
    use super::parse_hir;
    use kajit_hir::{
        BinaryOp, Block, CallExpr, CallSafety, CallSignature, CallTarget, CallableKind,
        CallableSpec, ControlTransfer, DomainAccess, DomainEffect, EffectClass, Expr, FieldDef,
        Function, GenericArg, GenericParam, Literal, LocalDecl, LocalId, LocalKind, MatchArm,
        MemoryWidth, Module, Pattern, PatternField, Place, Scope, ScopeId, Stmt, StmtId, StmtKind,
        Type, TypeDef, TypeDefKind, VariantDef, VixenCallableRef, VixenCoreTypes, VixenTypedExpr,
        VixenTypedFunction, VixenTypedLocal, VixenTypedParam, VixenTypedStmt,
    };

    fn sample_module() -> Module {
        let mut module = Module::new();
        let r_input = module.add_region("input");
        let cursor = module.add_type_def(TypeDef {
            name: "Cursor".to_owned(),
            generic_params: vec![GenericParam::Region {
                name: "r_input".to_owned(),
            }],
            kind: TypeDefKind::Struct {
                fields: vec![
                    FieldDef {
                        name: "bytes".to_owned(),
                        ty: Type::slice(r_input, Type::u(8)),
                        offset: None,
                    },
                    FieldDef {
                        name: "pos".to_owned(),
                        ty: Type::u(64),
                        offset: None,
                    },
                ],
            },
            size: None,
            transparent: false,
        });
        let opt_str = module.add_type_def(TypeDef {
            name: "core::option::Option<&str>".to_owned(),
            generic_params: vec![GenericParam::Region {
                name: "r_input".to_owned(),
            }],
            kind: TypeDefKind::Enum {
                variants: vec![
                    VariantDef {
                        name: "None".to_owned(),
                        fields: vec![],
                        discriminant: None,
                        init_fn: None,
                    },
                    VariantDef {
                        name: "Some".to_owned(),
                        fields: vec![FieldDef {
                            name: "value".to_owned(),
                            ty: Type::str(r_input),
                            offset: None,
                        }],
                        discriminant: None,
                        init_fn: None,
                    },
                ],
                discriminant_width: None,
            },
            size: None,
            transparent: false,
        });
        let record = module.add_type_def(TypeDef {
            name: "MaybeBorrowedName".to_owned(),
            generic_params: vec![GenericParam::Region {
                name: "r_input".to_owned(),
            }],
            kind: TypeDefKind::Struct {
                fields: vec![FieldDef {
                    name: "name".to_owned(),
                    ty: Type::named(opt_str, vec![GenericArg::Region(r_input)]),
                    offset: None,
                }],
            },
            size: None,
            transparent: false,
        });
        let read_tag = module.add_callable(CallableSpec {
            kind: CallableKind::Builtin,
            name: "postcard.read_option_tag".to_owned(),
            intrinsic: None,
            signature: CallSignature {
                params: vec![Type::named(cursor, vec![GenericArg::Region(r_input)])],
                returns: vec![Type::bool()],
                effect_class: EffectClass::Mutates,
                domain_effects: vec![DomainEffect {
                    domain: "cursor".to_owned(),
                    access: DomainAccess::Mutate,
                }],
                control: ControlTransfer::MayFail,
                capabilities: vec!["deser.postcard".to_owned()],
                safety: CallSafety::SafeCore,
            },
            docs: Some("Read and validate a postcard Option tag.".to_owned()),
        });
        let read_str = module.add_callable(CallableSpec {
            kind: CallableKind::Builtin,
            name: "postcard.read_str".to_owned(),
            intrinsic: None,
            signature: CallSignature {
                params: vec![Type::named(cursor, vec![GenericArg::Region(r_input)])],
                returns: vec![Type::str(r_input)],
                effect_class: EffectClass::Mutates,
                domain_effects: vec![DomainEffect {
                    domain: "cursor".to_owned(),
                    access: DomainAccess::Mutate,
                }],
                control: ControlTransfer::MayFail,
                capabilities: vec!["deser.postcard".to_owned()],
                safety: CallSafety::SafeCore,
            },
            docs: Some("Read a borrowed string.".to_owned()),
        });

        module.add_function(Function {
            name: "decode_MaybeBorrowedName".to_owned(),
            region_params: vec![r_input],
            store_params: vec![],
            params: vec![
                kajit_hir::Parameter {
                    local: LocalId::new(0),
                    name: "cursor".to_owned(),
                    ty: Type::named(cursor, vec![GenericArg::Region(r_input)]),
                    kind: LocalKind::Param,
                },
                kajit_hir::Parameter {
                    local: LocalId::new(1),
                    name: "out".to_owned(),
                    ty: Type::named(record, vec![GenericArg::Region(r_input)]),
                    kind: LocalKind::Param,
                },
            ],
            locals: vec![
                LocalDecl {
                    local: LocalId::new(2),
                    name: "option_is_some_0".to_owned(),
                    ty: Type::bool(),
                    kind: LocalKind::Temp,
                },
                LocalDecl {
                    local: LocalId::new(3),
                    name: "option_value_1".to_owned(),
                    ty: Type::str(r_input),
                    kind: LocalKind::Temp,
                },
            ],
            return_type: Type::unit(),
            scopes: vec![Scope {
                id: ScopeId::new(0),
                parent: None,
                comment: Some("sample".to_owned()),
            }],
            body: Block {
                scope: ScopeId::new(0),
                statements: vec![
                    Stmt {
                        id: StmtId::new(0),
                        kind: StmtKind::Init {
                            place: Place::Local(LocalId::new(2)),
                            value: Expr::Call(CallExpr {
                                target: CallTarget::Callable(read_tag),
                                args: vec![Expr::Local(LocalId::new(0))],
                            }),
                        },
                    },
                    Stmt {
                        id: StmtId::new(1),
                        kind: StmtKind::If {
                            condition: Expr::Local(LocalId::new(2)),
                            then_block: Block {
                                scope: ScopeId::new(0),
                                statements: vec![
                                    Stmt {
                                        id: StmtId::new(2),
                                        kind: StmtKind::Init {
                                            place: Place::Local(LocalId::new(3)),
                                            value: Expr::Call(CallExpr {
                                                target: CallTarget::Callable(read_str),
                                                args: vec![Expr::Local(LocalId::new(0))],
                                            }),
                                        },
                                    },
                                    Stmt {
                                        id: StmtId::new(3),
                                        kind: StmtKind::Init {
                                            place: Place::Field {
                                                base: Box::new(Place::Local(LocalId::new(1))),
                                                field: "name".to_owned(),
                                            },
                                            value: Expr::Variant {
                                                def: opt_str,
                                                variant: "Some".to_owned(),
                                                fields: vec![(
                                                    "value".to_owned(),
                                                    Expr::Local(LocalId::new(3)),
                                                )],
                                            },
                                        },
                                    },
                                ],
                            },
                            else_block: Some(Block {
                                scope: ScopeId::new(0),
                                statements: vec![Stmt {
                                    id: StmtId::new(4),
                                    kind: StmtKind::Init {
                                        place: Place::Field {
                                            base: Box::new(Place::Local(LocalId::new(1))),
                                            field: "name".to_owned(),
                                        },
                                        value: Expr::Variant {
                                            def: opt_str,
                                            variant: "None".to_owned(),
                                            fields: vec![],
                                        },
                                    },
                                }],
                            }),
                        },
                    },
                    Stmt {
                        id: StmtId::new(5),
                        kind: StmtKind::Return(None),
                    },
                ],
            },
        });

        module
    }

    fn known_len_persistent_vec_kernel_module() -> Module {
        let mut module = Module::new();
        let callables = module.install_runtime_memory_callables();
        module.add_function(Function {
            name: "build_vec_u32_2".to_owned(),
            region_params: vec![],
            store_params: vec![],
            params: vec![],
            locals: vec![
                LocalDecl {
                    local: LocalId::new(0),
                    name: "len".to_owned(),
                    ty: Type::u(64),
                    kind: LocalKind::Temp,
                },
                LocalDecl {
                    local: LocalId::new(1),
                    name: "bytes".to_owned(),
                    ty: Type::u(64),
                    kind: LocalKind::Temp,
                },
                LocalDecl {
                    local: LocalId::new(2),
                    name: "ptr".to_owned(),
                    ty: Type::persistent_addr(),
                    kind: LocalKind::Temp,
                },
            ],
            return_type: Type::u(64),
            scopes: vec![Scope {
                id: ScopeId::new(0),
                parent: None,
                comment: Some("Known-length persistent vec kernel".to_owned()),
            }],
            body: Block {
                scope: ScopeId::new(0),
                statements: vec![
                    Stmt {
                        id: StmtId::new(0),
                        kind: StmtKind::Init {
                            place: Place::Local(LocalId::new(0)),
                            value: Expr::Literal(Literal::Integer(2)),
                        },
                    },
                    Stmt {
                        id: StmtId::new(1),
                        kind: StmtKind::Init {
                            place: Place::Local(LocalId::new(1)),
                            value: Expr::Binary {
                                op: BinaryOp::Mul,
                                lhs: Box::new(Expr::Local(LocalId::new(0))),
                                rhs: Box::new(Expr::Literal(Literal::Integer(4))),
                            },
                        },
                    },
                    Stmt {
                        id: StmtId::new(2),
                        kind: StmtKind::Init {
                            place: Place::Local(LocalId::new(2)),
                            value: Expr::Call(CallExpr {
                                target: CallTarget::Callable(callables.alloc_persistent),
                                args: vec![
                                    Expr::Local(LocalId::new(1)),
                                    Expr::Literal(Literal::Integer(4)),
                                ],
                            }),
                        },
                    },
                    Stmt {
                        id: StmtId::new(3),
                        kind: StmtKind::Store {
                            addr: Expr::Local(LocalId::new(2)),
                            width: MemoryWidth::W4,
                            value: Expr::Literal(Literal::Integer(10)),
                        },
                    },
                    Stmt {
                        id: StmtId::new(4),
                        kind: StmtKind::Store {
                            addr: Expr::Binary {
                                op: BinaryOp::Add,
                                lhs: Box::new(Expr::Local(LocalId::new(2))),
                                rhs: Box::new(Expr::Literal(Literal::Integer(4))),
                            },
                            width: MemoryWidth::W4,
                            value: Expr::Literal(Literal::Integer(20)),
                        },
                    },
                    Stmt {
                        id: StmtId::new(5),
                        kind: StmtKind::Return(Some(Expr::Call(CallExpr {
                            target: CallTarget::Callable(callables.vec_from_raw_parts),
                            args: vec![
                                Expr::Local(LocalId::new(2)),
                                Expr::Local(LocalId::new(0)),
                                Expr::Local(LocalId::new(0)),
                                Expr::Literal(Literal::Integer(4)),
                            ],
                        }))),
                    },
                ],
            },
        });
        module
    }

    #[test]
    fn round_trip_address_types() {
        let text = r#"
hir_module {
  regions [
  ]
  stores [
  ]
  types [
    type t0 "RawParts" = struct {
      "transient": addr<transient>
      "persistent": addr<persistent>
    }
  ]
  callables [
  ]
  functions [
  ]
}
"#;

        let module = parse_hir(text).expect("address types should parse");
        let round_trip = module.to_string();
        let reparsed = parse_hir(&round_trip).expect("round-tripped address types should parse");
        assert_eq!(module, reparsed);
    }

    #[test]
    fn round_trip_ref_types_and_deref_places() {
        let text = r#"
hir_module {
  regions []
  stores []
  types [
    type t0 "Cursor" size=8 = struct {
      "pos": u64 @0
    }
  ]
  callables []
  functions [
    function f0 "ref_demo" {
      regions []
      stores []
      params [
        l0 param "cursor": &mut t0
      ]
      locals []
      return unit
      scopes [
        scope sc0 parent none comment none
      ]
      body @sc0 {
        stmt0: assign field(deref(l0), "pos") = 0x1
        stmt1: return
      }
    }
  ]
}
"#;

        let module = parse_hir(text).expect("ref types should parse");
        let round_trip = module.to_string();
        let reparsed = parse_hir(&round_trip).expect("round-tripped ref types should parse");
        assert_eq!(module, reparsed);
    }

    #[test]
    fn round_trips_sample_module() {
        let module = sample_module();
        let text = module.to_string();
        let reparsed = parse_hir(&text).expect("HIR text should parse");
        assert_eq!(reparsed, module);
    }

    #[test]
    fn round_trips_known_len_persistent_vec_kernel() {
        let module = known_len_persistent_vec_kernel_module();
        let text = module.to_string();
        let reparsed = parse_hir(&text).expect("kernel HIR text should parse");
        assert_eq!(reparsed, module);
    }

    #[test]
    fn round_trips_load_expressions() {
        let mut module = Module::new();
        module.add_function(Function {
            name: "load_demo".to_owned(),
            region_params: vec![],
            store_params: vec![],
            params: vec![kajit_hir::Parameter {
                local: LocalId::new(0),
                name: "addr".to_owned(),
                ty: Type::persistent_addr(),
                kind: LocalKind::Param,
            }],
            locals: vec![LocalDecl {
                local: LocalId::new(1),
                name: "word".to_owned(),
                ty: Type::u(32),
                kind: LocalKind::Temp,
            }],
            return_type: Type::unit(),
            scopes: vec![Scope {
                id: ScopeId::new(0),
                parent: None,
                comment: None,
            }],
            body: Block {
                scope: ScopeId::new(0),
                statements: vec![
                    Stmt {
                        id: StmtId::new(0),
                        kind: StmtKind::Init {
                            place: Place::Local(LocalId::new(1)),
                            value: Expr::Load {
                                addr: Box::new(Expr::Local(LocalId::new(0))),
                                width: MemoryWidth::W4,
                            },
                        },
                    },
                    Stmt {
                        id: StmtId::new(1),
                        kind: StmtKind::Return(None),
                    },
                ],
            },
        });

        let text = module.to_string();
        let reparsed = parse_hir(&text).expect("round-tripped load expressions should parse");
        assert_eq!(module, reparsed);
    }

    #[test]
    fn round_trips_slice_view_statements() {
        let text = r#"
hir_module {
  regions [
    r0 "input"
  ]
  stores [
  ]
  types [
    type t0 "Cursor" <region "r_input"> = struct {
      "bytes": Slice<r0, u8>
      "pos": u64
    }
  ]
  callables [
  ]
  functions [
    function f0 "slice_demo" {
      regions [r0]
      stores []
      params [
        l0 param "cursor": t0<r0>
      ]
      locals [
      ]
      return unit
      scopes [
        scope sc0 parent none comment none
      ]
      body @sc0 {
        stmt0: expr slice_data(field(l0, "bytes"))
        stmt1: expr slice_len(field(l0, "bytes"))
        stmt2: return
      }
    }
  ]
}
"#;

        let module = parse_hir(text).expect("slice views should parse");
        let round_trip = module.to_string();
        let reparsed = parse_hir(&round_trip).expect("round-tripped slice HIR should parse");
        assert_eq!(module, reparsed);
    }

    #[test]
    fn round_trips_match_and_expr_forms() {
        let mut module = Module::new();
        let r0 = module.add_region("input");
        let enum_id = module.add_type_def(TypeDef {
            name: "Flag".to_owned(),
            generic_params: vec![],
            kind: TypeDefKind::Enum {
                variants: vec![VariantDef {
                    name: "Set".to_owned(),
                    fields: vec![FieldDef {
                        name: "value".to_owned(),
                        ty: Type::u(32),
                        offset: None,
                    }],
                    discriminant: None,
                    init_fn: None,
                }],
                discriminant_width: None,
            },
            size: None,
            transparent: false,
        });
        module.add_function(Function {
            name: "demo".to_owned(),
            region_params: vec![r0],
            store_params: vec![],
            params: vec![kajit_hir::Parameter {
                local: LocalId::new(0),
                name: "x".to_owned(),
                ty: Type::u(32),
                kind: LocalKind::Param,
            }],
            locals: vec![LocalDecl {
                local: LocalId::new(1),
                name: "value".to_owned(),
                ty: Type::u(32),
                kind: LocalKind::Let,
            }],
            return_type: Type::unit(),
            scopes: vec![Scope {
                id: ScopeId::new(0),
                parent: None,
                comment: None,
            }],
            body: Block {
                scope: ScopeId::new(0),
                statements: vec![
                    Stmt {
                        id: StmtId::new(0),
                        kind: StmtKind::Expr(Expr::Binary {
                            op: BinaryOp::BitAnd,
                            lhs: Box::new(Expr::Literal(Literal::Integer(1))),
                            rhs: Box::new(Expr::Binary {
                                op: BinaryOp::Shl,
                                lhs: Box::new(Expr::Literal(Literal::Integer(3))),
                                rhs: Box::new(Expr::Literal(Literal::Integer(1))),
                            }),
                        }),
                    },
                    Stmt {
                        id: StmtId::new(1),
                        kind: StmtKind::Match {
                            scrutinee: Expr::Variant {
                                def: enum_id,
                                variant: "Set".to_owned(),
                                fields: vec![(
                                    "value".to_owned(),
                                    Expr::Literal(Literal::Integer(7)),
                                )],
                            },
                            arms: vec![
                                MatchArm {
                                    pattern: Pattern::Variant {
                                        name: "Set".to_owned(),
                                        fields: vec![PatternField::Bind {
                                            field: "value".to_owned(),
                                            local: LocalId::new(1),
                                        }],
                                    },
                                    body: Block {
                                        scope: ScopeId::new(0),
                                        statements: vec![
                                            Stmt {
                                                id: StmtId::new(2),
                                                kind: StmtKind::Expr(Expr::Local(LocalId::new(1))),
                                            },
                                            Stmt {
                                                id: StmtId::new(3),
                                                kind: StmtKind::Break,
                                            },
                                        ],
                                    },
                                },
                                MatchArm {
                                    pattern: Pattern::Wildcard,
                                    body: Block {
                                        scope: ScopeId::new(0),
                                        statements: vec![Stmt {
                                            id: StmtId::new(4),
                                            kind: StmtKind::Continue,
                                        }],
                                    },
                                },
                            ],
                        },
                    },
                ],
            },
        });

        let text = module.to_string();
        let reparsed = parse_hir(&text).expect("HIR text should parse");
        assert_eq!(reparsed, module);
    }

    #[test]
    fn round_trips_lowered_vixen_function_module() {
        let mut module = Module::new();
        let string = module.add_type_def(TypeDef {
            name: "String".to_owned(),
            generic_params: vec![],
            kind: TypeDefKind::Struct { fields: vec![] },
            size: None,
            transparent: false,
        });
        let node = module.add_type_def(TypeDef {
            name: "Node".to_owned(),
            generic_params: vec![],
            kind: TypeDefKind::Struct {
                fields: vec![FieldDef {
                    name: "label".to_owned(),
                    ty: Type::named(string, Vec::new()),
                    offset: None,
                }],
            },
            size: None,
            transparent: false,
        });
        let edge = module.add_type_def(TypeDef {
            name: "Edge".to_owned(),
            generic_params: vec![],
            kind: TypeDefKind::Struct { fields: vec![] },
            size: None,
            transparent: false,
        });
        let fact = module.add_type_def(TypeDef {
            name: "Fact".to_owned(),
            generic_params: vec![],
            kind: TypeDefKind::Struct { fields: vec![] },
            size: None,
            transparent: false,
        });
        let crate_graph = module.add_type_def(TypeDef {
            name: "CrateGraph".to_owned(),
            generic_params: vec![],
            kind: TypeDefKind::Struct { fields: vec![] },
            size: None,
            transparent: false,
        });
        let crate_node = module.add_type_def(TypeDef {
            name: "CrateNode".to_owned(),
            generic_params: vec![],
            kind: TypeDefKind::Struct { fields: vec![] },
            size: None,
            transparent: false,
        });
        let crate_id = module.add_type_def(TypeDef {
            name: "CrateId".to_owned(),
            generic_params: vec![],
            kind: TypeDefKind::Struct {
                fields: vec![FieldDef {
                    name: "value".to_owned(),
                    ty: Type::named(string, Vec::new()),
                    offset: None,
                }],
            },
            size: None,
            transparent: false,
        });
        let crate_type = module.add_type_def(TypeDef {
            name: "CrateType".to_owned(),
            generic_params: vec![],
            kind: TypeDefKind::Enum {
                variants: vec![
                    VariantDef {
                        name: "Lib".to_owned(),
                        fields: vec![],
                        discriminant: None,
                        init_fn: None,
                    },
                    VariantDef {
                        name: "Bin".to_owned(),
                        fields: vec![],
                        discriminant: None,
                        init_fn: None,
                    },
                ],
                discriminant_width: None,
            },
            size: None,
            transparent: false,
        });

        module.install_vixen_core_callables(&VixenCoreTypes {
            string: Type::named(string, Vec::new()),
            node: Type::named(node, Vec::new()),
            edge: Type::named(edge, Vec::new()),
            fact: Type::named(fact, Vec::new()),
            crate_graph: Type::named(crate_graph, Vec::new()),
            crate_node: Type::named(crate_node, Vec::new()),
            crate_id: Type::named(crate_id, Vec::new()),
            crate_type: Type::named(crate_type, Vec::new()),
        });

        let lowered = module
            .lower_vixen_typed_function_into_module(&VixenTypedFunction {
                name: "plan_compile".to_owned(),
                params: vec![VixenTypedParam {
                    local: LocalId::new(0),
                    name: "graph".to_owned(),
                    ty: Type::named(crate_graph, Vec::new()),
                }],
                locals: vec![
                    VixenTypedLocal {
                        local: LocalId::new(1),
                        name: "node".to_owned(),
                        ty: Type::named(node, Vec::new()),
                    },
                    VixenTypedLocal {
                        local: LocalId::new(2),
                        name: "emit_enabled".to_owned(),
                        ty: Type::bool(),
                    },
                ],
                return_type: Type::unit(),
                body: vec![
                    VixenTypedStmt::Let {
                        local: LocalId::new(2),
                        value: VixenTypedExpr::Literal(Literal::Bool(true)),
                    },
                    VixenTypedStmt::If {
                        condition: VixenTypedExpr::Local(LocalId::new(2)),
                        then_body: vec![
                            VixenTypedStmt::Let {
                                local: LocalId::new(1),
                                value: VixenTypedExpr::Struct {
                                    def: node,
                                    fields: vec![(
                                        "label".to_owned(),
                                        VixenTypedExpr::Literal(Literal::String(
                                            "compile".to_owned(),
                                        )),
                                    )],
                                },
                            },
                            VixenTypedStmt::Expr(VixenTypedExpr::Call {
                                callee: VixenCallableRef::Named("emit.node".to_owned()),
                                args: vec![VixenTypedExpr::Local(LocalId::new(1))],
                            }),
                        ],
                        else_body: vec![],
                    },
                    VixenTypedStmt::Return(None),
                ],
                comment: Some("lowered from typed Vixen stub".to_owned()),
            })
            .expect("typed Vixen function should lower");

        let text = lowered.to_string();
        let reparsed = parse_hir(&text).expect("HIR text should parse");
        assert_eq!(reparsed, lowered);
    }

    #[test]
    fn round_trips_extern_addr_literal() {
        let mut module = Module::new();
        module.add_function(Function {
            name: "use_extern".to_owned(),
            region_params: vec![],
            store_params: vec![],
            params: vec![],
            locals: vec![],
            return_type: Type::u(64),
            scopes: vec![Scope {
                id: ScopeId::new(0),
                parent: None,
                comment: None,
            }],
            body: Block {
                scope: ScopeId::new(0),
                statements: vec![Stmt {
                    id: StmtId::new(0),
                    kind: StmtKind::Return(Some(Expr::Literal(Literal::ExternAddr {
                        symbol: kajit_types::SymbolName::new("postcard.option_init_some"),
                    }))),
                }],
            },
        });
        let text = module.to_string();
        assert!(
            text.contains("@postcard.option_init_some"),
            "display should use @ syntax: {text}"
        );
        let reparsed = parse_hir(&text).expect("round-tripped ExternAddr should parse");
        assert_eq!(reparsed, module);
    }
}
