use super::*;

#[test]
fn structural_hir_ir_path_decodes_constant_output() {
    let module = parse_hir(
        r#"
hir_module {
  regions [
  ]
  stores [
  ]
  types [
    type t0 "ConstantNumber" = struct {
      "value": u32
    }
  ]
  callables [
  ]
  functions [
    function f0 "const_number" {
      regions []
      stores []
      params [
        l0 param "cursor": u64
        l1 destination "out": t0
      ]
      locals [
      ]
      return unit
      scopes [
        scope sc0 parent none comment "constant structural HIR"
      ]
      body @sc0 {
        stmt0: init field(l1, "value") = 0x2a
        stmt1: return
      }
    }
  ]
}
"#,
    )
    .expect("HIR text should parse");

    let ir = build_structural_hir_ir(<ConstantNumber>::SHAPE, &module);
    let registry = symbol_registry_for_shape(<ConstantNumber>::SHAPE);
    insta::assert_snapshot!(format!("{}", ir.display_with_registry(&registry)));
}

#[test]
fn structural_hir_ir_path_preserves_local_scalar_across_empty_else_if() {
    let mut module = hir::Module::new();
    let root_def = module.add_type_def(hir::TypeDef {
        name: <ConstantNumber>::SHAPE.type_identifier.to_owned(),
        generic_params: vec![],
        kind: hir::TypeDefKind::Struct {
            fields: vec![hir::FieldDef {
                name: "value".to_owned(),
                ty: hir::Type::u(32),
                offset: None,
            }],
        },
        size: None,
        transparent: false,
    });

    module.add_function(hir::Function {
        name: "local_across_if".to_owned(),
        region_params: vec![],
        store_params: vec![],
        params: vec![
            hir::Parameter {
                local: hir::LocalId::new(0),
                name: "cursor".to_owned(),
                ty: hir::Type::u(64),
                kind: hir::LocalKind::Param,
            },
            hir::Parameter {
                local: hir::LocalId::new(1),
                name: "out".to_owned(),
                ty: hir::Type::named(root_def, Vec::new()),
                kind: hir::LocalKind::Destination,
            },
        ],
        locals: vec![hir::LocalDecl {
            local: hir::LocalId::new(2),
            name: "tmp".to_owned(),
            ty: hir::Type::u(32),
            kind: hir::LocalKind::Temp,
        }],
        return_type: hir::Type::unit(),
        scopes: vec![hir::Scope {
            id: hir::ScopeId::new(0),
            parent: None,
            comment: Some("local scalar across empty else".to_owned()),
        }],
        body: hir::Block {
            scope: hir::ScopeId::new(0),
            statements: vec![
                hir::Stmt {
                    id: hir::StmtId::new(0),
                    kind: hir::StmtKind::Init {
                        place: hir::Place::Local(hir::LocalId::new(2)),
                        value: hir::Expr::Literal(hir::Literal::Integer(2)),
                    },
                },
                hir::Stmt {
                    id: hir::StmtId::new(1),
                    kind: hir::StmtKind::If {
                        condition: hir::Expr::Literal(hir::Literal::Bool(false)),
                        then_block: hir::Block {
                            scope: hir::ScopeId::new(0),
                            statements: vec![hir::Stmt {
                                id: hir::StmtId::new(2),
                                kind: hir::StmtKind::Fail {
                                    code: hir::ErrorCode::InvalidBool,
                                },
                            }],
                        },
                        else_block: Some(hir::Block {
                            scope: hir::ScopeId::new(0),
                            statements: Vec::new(),
                        }),
                    },
                },
                hir::Stmt {
                    id: hir::StmtId::new(3),
                    kind: hir::StmtKind::Assign {
                        place: hir::Place::Local(hir::LocalId::new(2)),
                        value: hir::Expr::Binary {
                            op: hir::BinaryOp::BitOr,
                            lhs: Box::new(hir::Expr::Local(hir::LocalId::new(2))),
                            rhs: Box::new(hir::Expr::Literal(hir::Literal::Integer(4))),
                        },
                    },
                },
                hir::Stmt {
                    id: hir::StmtId::new(4),
                    kind: hir::StmtKind::Init {
                        place: hir::Place::Field {
                            base: Box::new(hir::Place::Local(hir::LocalId::new(1))),
                            field: "value".to_owned(),
                        },
                        value: hir::Expr::Local(hir::LocalId::new(2)),
                    },
                },
                hir::Stmt {
                    id: hir::StmtId::new(5),
                    kind: hir::StmtKind::Return(None),
                },
            ],
        },
    });

    let decoder = compile_structural_hir_decoder(<ConstantNumber>::SHAPE, &module);
    let value = crate::deserialize::<ConstantNumber>(&decoder, &[])
        .expect("structural HIR decoder should preserve local scalars across if");
    assert_eq!(value, ConstantNumber { value: 6 });
}

#[test]
fn structural_hir_ir_path_executes_unrolled_varint_shape() {
    let mut module = hir::Module::new();
    let root_def = module.add_type_def(hir::TypeDef {
        name: <ScalarArrayHolder>::SHAPE.type_identifier.to_owned(),
        generic_params: vec![],
        kind: hir::TypeDefKind::Struct {
            fields: vec![hir::FieldDef {
                name: "values".to_owned(),
                ty: hir::Type::array(hir::Type::u(32), 4),
                offset: None,
            }],
        },
        size: None,
        transparent: false,
    });

    let mut locals = Vec::new();
    let mut statements = Vec::new();
    let mut next_stmt = 0u32;
    let mut next_local = 2u32;
    let make_local = |locals: &mut Vec<hir::LocalDecl>,
                      next_local: &mut u32,
                      name: String,
                      ty: hir::Type|
     -> hir::LocalId {
        let local = hir::LocalId::new(*next_local);
        *next_local += 1;
        locals.push(hir::LocalDecl {
            local,
            name,
            ty,
            kind: hir::LocalKind::Temp,
        });
        local
    };

    for (index, raw_value) in [1_u64, 2, 3, 4].into_iter().enumerate() {
        let acc_local = make_local(
            &mut locals,
            &mut next_local,
            format!("acc_{index}"),
            hir::Type::u(64),
        );
        let raw_local = make_local(
            &mut locals,
            &mut next_local,
            format!("raw_{index}"),
            hir::Type::u(8),
        );
        statements.push(hir::Stmt {
            id: hir::StmtId::new(next_stmt),
            kind: hir::StmtKind::Init {
                place: hir::Place::Local(acc_local),
                value: hir::Expr::Literal(hir::Literal::Integer(0)),
            },
        });
        next_stmt += 1;
        statements.push(hir::Stmt {
            id: hir::StmtId::new(next_stmt),
            kind: hir::StmtKind::Init {
                place: hir::Place::Local(raw_local),
                value: hir::Expr::Literal(hir::Literal::Integer(raw_value)),
            },
        });
        next_stmt += 1;
        statements.push(hir::Stmt {
            id: hir::StmtId::new(next_stmt),
            kind: hir::StmtKind::Assign {
                place: hir::Place::Local(acc_local),
                value: hir::Expr::Binary {
                    op: hir::BinaryOp::BitOr,
                    lhs: Box::new(hir::Expr::Local(acc_local)),
                    rhs: Box::new(hir::Expr::Binary {
                        op: hir::BinaryOp::BitAnd,
                        lhs: Box::new(hir::Expr::Local(raw_local)),
                        rhs: Box::new(hir::Expr::Literal(hir::Literal::Integer(0x7f))),
                    }),
                },
            },
        });
        next_stmt += 1;
        statements.push(hir::Stmt {
            id: hir::StmtId::new(next_stmt),
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
                then_block: hir::Block {
                    scope: hir::ScopeId::new(0),
                    statements: vec![hir::Stmt {
                        id: hir::StmtId::new(next_stmt + 1),
                        kind: hir::StmtKind::Fail {
                            code: hir::ErrorCode::InvalidVarint,
                        },
                    }],
                },
                else_block: Some(hir::Block {
                    scope: hir::ScopeId::new(0),
                    statements: vec![hir::Stmt {
                        id: hir::StmtId::new(next_stmt + 2),
                        kind: hir::StmtKind::Init {
                            place: hir::Place::Index {
                                base: Box::new(hir::Place::Field {
                                    base: Box::new(hir::Place::Local(hir::LocalId::new(1))),
                                    field: "values".to_owned(),
                                }),
                                index: Box::new(hir::Expr::Literal(hir::Literal::Integer(
                                    index as u64,
                                ))),
                            },
                            value: hir::Expr::Local(acc_local),
                        },
                    }],
                }),
            },
        });
        next_stmt += 3;
    }
    statements.push(hir::Stmt {
        id: hir::StmtId::new(next_stmt),
        kind: hir::StmtKind::Return(None),
    });

    module.add_function(hir::Function {
        name: "unrolled_varint_shape".to_owned(),
        region_params: vec![],
        store_params: vec![],
        params: vec![
            hir::Parameter {
                local: hir::LocalId::new(0),
                name: "cursor".to_owned(),
                ty: hir::Type::u(64),
                kind: hir::LocalKind::Param,
            },
            hir::Parameter {
                local: hir::LocalId::new(1),
                name: "out".to_owned(),
                ty: hir::Type::named(root_def, Vec::new()),
                kind: hir::LocalKind::Destination,
            },
        ],
        locals,
        return_type: hir::Type::unit(),
        scopes: vec![hir::Scope {
            id: hir::ScopeId::new(0),
            parent: None,
            comment: Some("unrolled varint shape".to_owned()),
        }],
        body: hir::Block {
            scope: hir::ScopeId::new(0),
            statements,
        },
    });

    let decoder = compile_structural_hir_decoder(<ScalarArrayHolder>::SHAPE, &module);
    let value = crate::deserialize::<ScalarArrayHolder>(&decoder, &[])
        .expect("structural HIR decoder should execute unrolled varint shape");
    assert_eq!(
        value,
        ScalarArrayHolder {
            values: [1, 2, 3, 4]
        }
    );
}

#[test]
fn structural_hir_ir_path_reads_bytes_via_cursor_shadow() {
    let mut module = hir::Module::new();
    let input_region = module.add_region("input");
    let cursor_def = module.add_type_def(hir::TypeDef {
        name: "Cursor".to_owned(),
        generic_params: vec![hir::GenericParam::Region {
            name: "r_input".to_owned(),
        }],
        kind: hir::TypeDefKind::Struct {
            fields: vec![
                hir::FieldDef {
                    name: "bytes".to_owned(),
                    ty: hir::Type::slice(input_region, hir::Type::u(8)),
                    offset: None,
                },
                hir::FieldDef {
                    name: "pos".to_owned(),
                    ty: hir::Type::u(64),
                    offset: None,
                },
            ],
        },
        size: None,
        transparent: false,
    });
    let root_def = module.add_type_def(hir::TypeDef {
        name: <ScalarArrayHolder>::SHAPE.type_identifier.to_owned(),
        generic_params: vec![],
        kind: hir::TypeDefKind::Struct {
            fields: vec![hir::FieldDef {
                name: "values".to_owned(),
                ty: hir::Type::array(hir::Type::u(32), 4),
                offset: None,
            }],
        },
        size: None,
        transparent: false,
    });

    let mut statements = Vec::new();
    let mut next_stmt = 0u32;
    for index in 0..4_u64 {
        let pos_expr = hir::Expr::Field {
            base: Box::new(hir::Expr::Local(hir::LocalId::new(0))),
            field: "pos".to_owned(),
        };
        statements.push(hir::Stmt {
            id: hir::StmtId::new(next_stmt),
            kind: hir::StmtKind::If {
                condition: hir::Expr::Binary {
                    op: hir::BinaryOp::Gt,
                    lhs: Box::new(hir::Expr::Binary {
                        op: hir::BinaryOp::Add,
                        lhs: Box::new(pos_expr.clone()),
                        rhs: Box::new(hir::Expr::Literal(hir::Literal::Integer(1))),
                    }),
                    rhs: Box::new(hir::Expr::SliceLen {
                        value: Box::new(hir::Expr::Field {
                            base: Box::new(hir::Expr::Local(hir::LocalId::new(0))),
                            field: "bytes".to_owned(),
                        }),
                    }),
                },
                then_block: hir::Block {
                    scope: hir::ScopeId::new(0),
                    statements: vec![hir::Stmt {
                        id: hir::StmtId::new(next_stmt + 1),
                        kind: hir::StmtKind::Fail {
                            code: hir::ErrorCode::UnexpectedEof,
                        },
                    }],
                },
                else_block: Some(hir::Block {
                    scope: hir::ScopeId::new(0),
                    statements: Vec::new(),
                }),
            },
        });
        next_stmt += 2;
        statements.push(hir::Stmt {
            id: hir::StmtId::new(next_stmt),
            kind: hir::StmtKind::Init {
                place: hir::Place::Index {
                    base: Box::new(hir::Place::Field {
                        base: Box::new(hir::Place::Local(hir::LocalId::new(1))),
                        field: "values".to_owned(),
                    }),
                    index: Box::new(hir::Expr::Literal(hir::Literal::Integer(index))),
                },
                value: hir::Expr::Load {
                    addr: Box::new(hir::Expr::Binary {
                        op: hir::BinaryOp::Add,
                        lhs: Box::new(hir::Expr::SliceData {
                            value: Box::new(hir::Expr::Field {
                                base: Box::new(hir::Expr::Local(hir::LocalId::new(0))),
                                field: "bytes".to_owned(),
                            }),
                        }),
                        rhs: Box::new(pos_expr.clone()),
                    }),
                    width: hir::MemoryWidth::W1,
                },
            },
        });
        next_stmt += 1;
        statements.push(hir::Stmt {
            id: hir::StmtId::new(next_stmt),
            kind: hir::StmtKind::Assign {
                place: hir::Place::Field {
                    base: Box::new(hir::Place::Local(hir::LocalId::new(0))),
                    field: "pos".to_owned(),
                },
                value: hir::Expr::Binary {
                    op: hir::BinaryOp::Add,
                    lhs: Box::new(pos_expr),
                    rhs: Box::new(hir::Expr::Literal(hir::Literal::Integer(1))),
                },
            },
        });
        next_stmt += 1;
    }
    statements.push(hir::Stmt {
        id: hir::StmtId::new(next_stmt),
        kind: hir::StmtKind::Return(None),
    });

    module.add_function(hir::Function {
        name: "cursor_shadow_reads".to_owned(),
        region_params: vec![input_region],
        store_params: vec![],
        params: vec![
            hir::Parameter {
                local: hir::LocalId::new(0),
                name: "cursor".to_owned(),
                ty: hir::Type::named(cursor_def, vec![hir::GenericArg::Region(input_region)]),
                kind: hir::LocalKind::Param,
            },
            hir::Parameter {
                local: hir::LocalId::new(1),
                name: "out".to_owned(),
                ty: hir::Type::named(root_def, Vec::new()),
                kind: hir::LocalKind::Destination,
            },
        ],
        locals: vec![],
        return_type: hir::Type::unit(),
        scopes: vec![hir::Scope {
            id: hir::ScopeId::new(0),
            parent: None,
            comment: Some("cursor shadow byte loads".to_owned()),
        }],
        body: hir::Block {
            scope: hir::ScopeId::new(0),
            statements,
        },
    });

    let decoder = compile_structural_hir_decoder(<ScalarArrayHolder>::SHAPE, &module);
    let value = crate::deserialize::<ScalarArrayHolder>(&decoder, &[1, 2, 3, 4])
        .expect("structural HIR decoder should read bytes via the cursor shadow");
    assert_eq!(
        value,
        ScalarArrayHolder {
            values: [1, 2, 3, 4]
        }
    );
}

#[test]
fn structural_hir_ir_path_parses_json_bool_tokens() {
    let module = build_structural_json_bool_module();
    let decoder = compile_structural_hir_decoder(<BoolHeader>::SHAPE, &module);

    let t = crate::deserialize::<BoolHeader>(&decoder, br#"true"#)
        .expect("json bool kernel should parse true");
    assert_eq!(t, BoolHeader { value: true });

    let f = crate::deserialize::<BoolHeader>(&decoder, b" \n\tfalse")
        .expect("json bool kernel should skip leading whitespace and parse false");
    assert_eq!(f, BoolHeader { value: false });
}

#[test]
fn structural_hir_ir_path_rejects_invalid_json_bool_tokens() {
    let module = build_structural_json_bool_module();
    let decoder = compile_structural_hir_decoder(<BoolHeader>::SHAPE, &module);

    let err = crate::deserialize::<BoolHeader>(&decoder, br#"trux"#)
        .expect_err("json bool kernel should reject invalid bool tokens");
    assert_eq!(err.code, crate::context::ErrorCode::InvalidBool);

    let err = crate::deserialize::<BoolHeader>(&decoder, b"   ")
        .expect_err("json bool kernel should reject whitespace-only input");
    assert_eq!(err.code, crate::context::ErrorCode::UnexpectedEof);
}

#[test]
fn json_structural_hir_ir_path_decodes_root_bool() {
    let decoder = compile_json_decoder_via_structural_hir(<bool>::SHAPE);

    let t = crate::deserialize::<bool>(&decoder, br#"true"#)
        .expect("shape-driven JSON HIR should parse true");
    assert!(t);

    let f = crate::deserialize::<bool>(&decoder, b"\r\n false")
        .expect("shape-driven JSON HIR should skip leading whitespace and parse false");
    assert!(!f);
}

#[test]
fn json_structural_hir_ir_path_decodes_root_u32() {
    let decoder = compile_json_decoder_via_structural_hir(<u32>::SHAPE);

    let zero =
        crate::deserialize::<u32>(&decoder, b"0").expect("shape-driven JSON HIR should parse zero");
    assert_eq!(zero, 0);

    let forty_two = crate::deserialize::<u32>(&decoder, b" \n42")
        .expect("shape-driven JSON HIR should skip leading whitespace and parse digits");
    assert_eq!(forty_two, 42);

    let err = crate::deserialize::<u32>(&decoder, b"")
        .expect_err("shape-driven JSON HIR should reject empty input");
    assert_eq!(err.code, crate::context::ErrorCode::UnexpectedEof);

    let err = crate::deserialize::<u32>(&decoder, b"abc")
        .expect_err("shape-driven JSON HIR should reject non-digit input");
    assert_eq!(err.code, crate::context::ErrorCode::InvalidJsonNumber);

    let err = crate::deserialize::<u32>(&decoder, b"4294967296")
        .expect_err("shape-driven JSON HIR should reject out-of-range u32 values");
    assert_eq!(err.code, crate::context::ErrorCode::NumberOutOfRange);
}

#[test]
fn json_structural_hir_ir_path_decodes_root_u64() {
    let decoder = compile_json_decoder_via_structural_hir(<u64>::SHAPE);

    let zero =
        crate::deserialize::<u64>(&decoder, b"0").expect("shape-driven JSON HIR should parse zero");
    assert_eq!(zero, 0);

    let max = crate::deserialize::<u64>(&decoder, b"18446744073709551615")
        .expect("shape-driven JSON HIR should parse u64::MAX");
    assert_eq!(max, u64::MAX);

    let err = crate::deserialize::<u64>(&decoder, b"18446744073709551616")
        .expect_err("shape-driven JSON HIR should reject overflowing u64 values");
    assert_eq!(err.code, crate::context::ErrorCode::NumberOutOfRange);
}

#[test]
fn structural_hir_ir_path_executes_cursor_shadow_varint_array() {
    let mut module = hir::Module::new();
    let input_region = module.add_region("input");
    let cursor_def = module.add_type_def(hir::TypeDef {
        name: "Cursor".to_owned(),
        generic_params: vec![hir::GenericParam::Region {
            name: "r_input".to_owned(),
        }],
        kind: hir::TypeDefKind::Struct {
            fields: vec![
                hir::FieldDef {
                    name: "bytes".to_owned(),
                    ty: hir::Type::slice(input_region, hir::Type::u(8)),
                    offset: None,
                },
                hir::FieldDef {
                    name: "pos".to_owned(),
                    ty: hir::Type::u(64),
                    offset: None,
                },
            ],
        },
        size: None,
        transparent: false,
    });
    let root_def = module.add_type_def(hir::TypeDef {
        name: <ScalarArrayHolder>::SHAPE.type_identifier.to_owned(),
        generic_params: vec![],
        kind: hir::TypeDefKind::Struct {
            fields: vec![hir::FieldDef {
                name: "values".to_owned(),
                ty: hir::Type::array(hir::Type::u(32), 4),
                offset: None,
            }],
        },
        size: None,
        transparent: false,
    });

    let mut locals = Vec::new();
    let mut statements = Vec::new();
    let mut next_stmt = 0u32;
    let mut next_local = 2u32;
    let make_local = |locals: &mut Vec<hir::LocalDecl>,
                      next_local: &mut u32,
                      name: String,
                      ty: hir::Type|
     -> hir::LocalId {
        let local = hir::LocalId::new(*next_local);
        *next_local += 1;
        locals.push(hir::LocalDecl {
            local,
            name,
            ty,
            kind: hir::LocalKind::Temp,
        });
        local
    };

    for index in 0..4_u64 {
        let acc_local = make_local(
            &mut locals,
            &mut next_local,
            format!("acc_{index}"),
            hir::Type::u(64),
        );
        let raw_local = make_local(
            &mut locals,
            &mut next_local,
            format!("raw_{index}"),
            hir::Type::u(8),
        );
        let pos_expr = hir::Expr::Field {
            base: Box::new(hir::Expr::Local(hir::LocalId::new(0))),
            field: "pos".to_owned(),
        };
        statements.push(hir::Stmt {
            id: hir::StmtId::new(next_stmt),
            kind: hir::StmtKind::If {
                condition: hir::Expr::Binary {
                    op: hir::BinaryOp::Gt,
                    lhs: Box::new(hir::Expr::Binary {
                        op: hir::BinaryOp::Add,
                        lhs: Box::new(pos_expr.clone()),
                        rhs: Box::new(hir::Expr::Literal(hir::Literal::Integer(1))),
                    }),
                    rhs: Box::new(hir::Expr::SliceLen {
                        value: Box::new(hir::Expr::Field {
                            base: Box::new(hir::Expr::Local(hir::LocalId::new(0))),
                            field: "bytes".to_owned(),
                        }),
                    }),
                },
                then_block: hir::Block {
                    scope: hir::ScopeId::new(0),
                    statements: vec![hir::Stmt {
                        id: hir::StmtId::new(next_stmt + 1),
                        kind: hir::StmtKind::Fail {
                            code: hir::ErrorCode::UnexpectedEof,
                        },
                    }],
                },
                else_block: Some(hir::Block {
                    scope: hir::ScopeId::new(0),
                    statements: Vec::new(),
                }),
            },
        });
        next_stmt += 2;
        statements.push(hir::Stmt {
            id: hir::StmtId::new(next_stmt),
            kind: hir::StmtKind::Init {
                place: hir::Place::Local(raw_local),
                value: hir::Expr::Load {
                    addr: Box::new(hir::Expr::Binary {
                        op: hir::BinaryOp::Add,
                        lhs: Box::new(hir::Expr::SliceData {
                            value: Box::new(hir::Expr::Field {
                                base: Box::new(hir::Expr::Local(hir::LocalId::new(0))),
                                field: "bytes".to_owned(),
                            }),
                        }),
                        rhs: Box::new(pos_expr.clone()),
                    }),
                    width: hir::MemoryWidth::W1,
                },
            },
        });
        next_stmt += 1;
        statements.push(hir::Stmt {
            id: hir::StmtId::new(next_stmt),
            kind: hir::StmtKind::Assign {
                place: hir::Place::Field {
                    base: Box::new(hir::Place::Local(hir::LocalId::new(0))),
                    field: "pos".to_owned(),
                },
                value: hir::Expr::Binary {
                    op: hir::BinaryOp::Add,
                    lhs: Box::new(pos_expr),
                    rhs: Box::new(hir::Expr::Literal(hir::Literal::Integer(1))),
                },
            },
        });
        next_stmt += 1;
        statements.push(hir::Stmt {
            id: hir::StmtId::new(next_stmt),
            kind: hir::StmtKind::Init {
                place: hir::Place::Local(acc_local),
                value: hir::Expr::Binary {
                    op: hir::BinaryOp::BitAnd,
                    lhs: Box::new(hir::Expr::Local(raw_local)),
                    rhs: Box::new(hir::Expr::Literal(hir::Literal::Integer(0x7f))),
                },
            },
        });
        next_stmt += 1;
        statements.push(hir::Stmt {
            id: hir::StmtId::new(next_stmt),
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
                then_block: hir::Block {
                    scope: hir::ScopeId::new(0),
                    statements: vec![hir::Stmt {
                        id: hir::StmtId::new(next_stmt + 1),
                        kind: hir::StmtKind::Fail {
                            code: hir::ErrorCode::InvalidVarint,
                        },
                    }],
                },
                else_block: Some(hir::Block {
                    scope: hir::ScopeId::new(0),
                    statements: vec![hir::Stmt {
                        id: hir::StmtId::new(next_stmt + 2),
                        kind: hir::StmtKind::Init {
                            place: hir::Place::Index {
                                base: Box::new(hir::Place::Field {
                                    base: Box::new(hir::Place::Local(hir::LocalId::new(1))),
                                    field: "values".to_owned(),
                                }),
                                index: Box::new(hir::Expr::Literal(hir::Literal::Integer(index))),
                            },
                            value: hir::Expr::Local(acc_local),
                        },
                    }],
                }),
            },
        });
        next_stmt += 3;
    }

    statements.push(hir::Stmt {
        id: hir::StmtId::new(next_stmt),
        kind: hir::StmtKind::Return(None),
    });

    module.add_function(hir::Function {
        name: "cursor_shadow_varint_array".to_owned(),
        region_params: vec![input_region],
        store_params: vec![],
        params: vec![
            hir::Parameter {
                local: hir::LocalId::new(0),
                name: "cursor".to_owned(),
                ty: hir::Type::named(cursor_def, vec![hir::GenericArg::Region(input_region)]),
                kind: hir::LocalKind::Param,
            },
            hir::Parameter {
                local: hir::LocalId::new(1),
                name: "out".to_owned(),
                ty: hir::Type::named(root_def, Vec::new()),
                kind: hir::LocalKind::Destination,
            },
        ],
        locals,
        return_type: hir::Type::unit(),
        scopes: vec![hir::Scope {
            id: hir::ScopeId::new(0),
            parent: None,
            comment: Some("cursor shadow varint array".to_owned()),
        }],
        body: hir::Block {
            scope: hir::ScopeId::new(0),
            statements,
        },
    });

    let decoder = compile_structural_hir_decoder(<ScalarArrayHolder>::SHAPE, &module);
    let value = crate::deserialize::<ScalarArrayHolder>(&decoder, &[1, 2, 3, 4])
        .expect("structural HIR decoder should execute cursor shadow varint array");
    assert_eq!(
        value,
        ScalarArrayHolder {
            values: [1, 2, 3, 4]
        }
    );
}

#[test]
fn structural_hir_ir_path_executes_cursor_shadow_range_checked_varint_array() {
    let mut module = hir::Module::new();
    let input_region = module.add_region("input");
    let cursor_def = module.add_type_def(hir::TypeDef {
        name: "Cursor".to_owned(),
        generic_params: vec![hir::GenericParam::Region {
            name: "r_input".to_owned(),
        }],
        kind: hir::TypeDefKind::Struct {
            fields: vec![
                hir::FieldDef {
                    name: "bytes".to_owned(),
                    ty: hir::Type::slice(input_region, hir::Type::u(8)),
                    offset: None,
                },
                hir::FieldDef {
                    name: "pos".to_owned(),
                    ty: hir::Type::u(64),
                    offset: None,
                },
            ],
        },
        size: None,
        transparent: false,
    });
    let root_def = module.add_type_def(hir::TypeDef {
        name: <ScalarArrayHolder>::SHAPE.type_identifier.to_owned(),
        generic_params: vec![],
        kind: hir::TypeDefKind::Struct {
            fields: vec![hir::FieldDef {
                name: "values".to_owned(),
                ty: hir::Type::array(hir::Type::u(32), 4),
                offset: None,
            }],
        },
        size: None,
        transparent: false,
    });

    let mut locals = Vec::new();
    let mut statements = Vec::new();
    let mut next_stmt = 0u32;
    let mut next_local = 2u32;
    let make_local = |locals: &mut Vec<hir::LocalDecl>,
                      next_local: &mut u32,
                      name: String,
                      ty: hir::Type|
     -> hir::LocalId {
        let local = hir::LocalId::new(*next_local);
        *next_local += 1;
        locals.push(hir::LocalDecl {
            local,
            name,
            ty,
            kind: hir::LocalKind::Temp,
        });
        local
    };

    for index in 0..4_u64 {
        let acc_local = make_local(
            &mut locals,
            &mut next_local,
            format!("acc_{index}"),
            hir::Type::u(64),
        );
        let raw_local = make_local(
            &mut locals,
            &mut next_local,
            format!("raw_{index}"),
            hir::Type::u(8),
        );
        let pos_expr = hir::Expr::Field {
            base: Box::new(hir::Expr::Local(hir::LocalId::new(0))),
            field: "pos".to_owned(),
        };
        statements.push(hir::Stmt {
            id: hir::StmtId::new(next_stmt),
            kind: hir::StmtKind::If {
                condition: hir::Expr::Binary {
                    op: hir::BinaryOp::Gt,
                    lhs: Box::new(hir::Expr::Binary {
                        op: hir::BinaryOp::Add,
                        lhs: Box::new(pos_expr.clone()),
                        rhs: Box::new(hir::Expr::Literal(hir::Literal::Integer(1))),
                    }),
                    rhs: Box::new(hir::Expr::SliceLen {
                        value: Box::new(hir::Expr::Field {
                            base: Box::new(hir::Expr::Local(hir::LocalId::new(0))),
                            field: "bytes".to_owned(),
                        }),
                    }),
                },
                then_block: hir::Block {
                    scope: hir::ScopeId::new(0),
                    statements: vec![hir::Stmt {
                        id: hir::StmtId::new(next_stmt + 1),
                        kind: hir::StmtKind::Fail {
                            code: hir::ErrorCode::UnexpectedEof,
                        },
                    }],
                },
                else_block: Some(hir::Block {
                    scope: hir::ScopeId::new(0),
                    statements: Vec::new(),
                }),
            },
        });
        next_stmt += 2;
        statements.push(hir::Stmt {
            id: hir::StmtId::new(next_stmt),
            kind: hir::StmtKind::Init {
                place: hir::Place::Local(raw_local),
                value: hir::Expr::Load {
                    addr: Box::new(hir::Expr::Binary {
                        op: hir::BinaryOp::Add,
                        lhs: Box::new(hir::Expr::SliceData {
                            value: Box::new(hir::Expr::Field {
                                base: Box::new(hir::Expr::Local(hir::LocalId::new(0))),
                                field: "bytes".to_owned(),
                            }),
                        }),
                        rhs: Box::new(pos_expr.clone()),
                    }),
                    width: hir::MemoryWidth::W1,
                },
            },
        });
        next_stmt += 1;
        statements.push(hir::Stmt {
            id: hir::StmtId::new(next_stmt),
            kind: hir::StmtKind::Assign {
                place: hir::Place::Field {
                    base: Box::new(hir::Place::Local(hir::LocalId::new(0))),
                    field: "pos".to_owned(),
                },
                value: hir::Expr::Binary {
                    op: hir::BinaryOp::Add,
                    lhs: Box::new(pos_expr),
                    rhs: Box::new(hir::Expr::Literal(hir::Literal::Integer(1))),
                },
            },
        });
        next_stmt += 1;
        statements.push(hir::Stmt {
            id: hir::StmtId::new(next_stmt),
            kind: hir::StmtKind::Init {
                place: hir::Place::Local(acc_local),
                value: hir::Expr::Binary {
                    op: hir::BinaryOp::BitAnd,
                    lhs: Box::new(hir::Expr::Local(raw_local)),
                    rhs: Box::new(hir::Expr::Literal(hir::Literal::Integer(0x7f))),
                },
            },
        });
        next_stmt += 1;
        statements.push(hir::Stmt {
            id: hir::StmtId::new(next_stmt),
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
                then_block: hir::Block {
                    scope: hir::ScopeId::new(0),
                    statements: vec![hir::Stmt {
                        id: hir::StmtId::new(next_stmt + 1),
                        kind: hir::StmtKind::Fail {
                            code: hir::ErrorCode::InvalidVarint,
                        },
                    }],
                },
                else_block: Some(hir::Block {
                    scope: hir::ScopeId::new(0),
                    statements: vec![hir::Stmt {
                        id: hir::StmtId::new(next_stmt + 2),
                        kind: hir::StmtKind::If {
                            condition: hir::Expr::Binary {
                                op: hir::BinaryOp::Ne,
                                lhs: Box::new(hir::Expr::Binary {
                                    op: hir::BinaryOp::Shr,
                                    lhs: Box::new(hir::Expr::Local(acc_local)),
                                    rhs: Box::new(hir::Expr::Literal(hir::Literal::Integer(32))),
                                }),
                                rhs: Box::new(hir::Expr::Literal(hir::Literal::Integer(0))),
                            },
                            then_block: hir::Block {
                                scope: hir::ScopeId::new(0),
                                statements: vec![hir::Stmt {
                                    id: hir::StmtId::new(next_stmt + 3),
                                    kind: hir::StmtKind::Fail {
                                        code: hir::ErrorCode::NumberOutOfRange,
                                    },
                                }],
                            },
                            else_block: Some(hir::Block {
                                scope: hir::ScopeId::new(0),
                                statements: vec![hir::Stmt {
                                    id: hir::StmtId::new(next_stmt + 4),
                                    kind: hir::StmtKind::Init {
                                        place: hir::Place::Index {
                                            base: Box::new(hir::Place::Field {
                                                base: Box::new(hir::Place::Local(
                                                    hir::LocalId::new(1),
                                                )),
                                                field: "values".to_owned(),
                                            }),
                                            index: Box::new(hir::Expr::Literal(
                                                hir::Literal::Integer(index),
                                            )),
                                        },
                                        value: hir::Expr::Local(acc_local),
                                    },
                                }],
                            }),
                        },
                    }],
                }),
            },
        });
        next_stmt += 5;
    }

    statements.push(hir::Stmt {
        id: hir::StmtId::new(next_stmt),
        kind: hir::StmtKind::Return(None),
    });

    module.add_function(hir::Function {
        name: "cursor_shadow_range_checked_varint_array".to_owned(),
        region_params: vec![input_region],
        store_params: vec![],
        params: vec![
            hir::Parameter {
                local: hir::LocalId::new(0),
                name: "cursor".to_owned(),
                ty: hir::Type::named(cursor_def, vec![hir::GenericArg::Region(input_region)]),
                kind: hir::LocalKind::Param,
            },
            hir::Parameter {
                local: hir::LocalId::new(1),
                name: "out".to_owned(),
                ty: hir::Type::named(root_def, Vec::new()),
                kind: hir::LocalKind::Destination,
            },
        ],
        locals,
        return_type: hir::Type::unit(),
        scopes: vec![hir::Scope {
            id: hir::ScopeId::new(0),
            parent: None,
            comment: Some("cursor shadow range checked varint array".to_owned()),
        }],
        body: hir::Block {
            scope: hir::ScopeId::new(0),
            statements,
        },
    });

    let decoder = compile_structural_hir_decoder(<ScalarArrayHolder>::SHAPE, &module);
    let value = crate::deserialize::<ScalarArrayHolder>(&decoder, &[1, 2, 3, 4])
        .expect("structural HIR decoder should execute range checked cursor shadow varint array");
    assert_eq!(
        value,
        ScalarArrayHolder {
            values: [1, 2, 3, 4]
        }
    );
}

#[test]
fn structural_hir_ir_path_executes_exact_postcard_varint_array_shape() {
    let mut module = hir::Module::new();
    let input_region = module.add_region("input");
    let cursor_def = module.add_type_def(hir::TypeDef {
        name: "Cursor".to_owned(),
        generic_params: vec![hir::GenericParam::Region {
            name: "r_input".to_owned(),
        }],
        kind: hir::TypeDefKind::Struct {
            fields: vec![
                hir::FieldDef {
                    name: "bytes".to_owned(),
                    ty: hir::Type::slice(input_region, hir::Type::u(8)),
                    offset: None,
                },
                hir::FieldDef {
                    name: "pos".to_owned(),
                    ty: hir::Type::u(64),
                    offset: None,
                },
            ],
        },
        size: None,
        transparent: false,
    });
    let root_def = module.add_type_def(hir::TypeDef {
        name: <ScalarArrayHolder>::SHAPE.type_identifier.to_owned(),
        generic_params: vec![],
        kind: hir::TypeDefKind::Struct {
            fields: vec![hir::FieldDef {
                name: "values".to_owned(),
                ty: hir::Type::array(hir::Type::u(32), 4),
                offset: None,
            }],
        },
        size: None,
        transparent: false,
    });

    let mut lowerer = PostcardHirLowerer::new();
    let cursor_local = hir::LocalId::new(0);
    let out_local = hir::LocalId::new(1);
    let mut statements = Vec::new();
    let (fields, _) = collect_fields(<ScalarArrayHolder>::SHAPE);
    lowerer.lower_shape_into_place(
        &mut statements,
        cursor_local,
        hir::Place::Field {
            base: Box::new(hir::Place::Local(out_local)),
            field: "values".to_owned(),
        },
        fields[0].shape,
    );
    statements.push(hir::Stmt {
        id: lowerer.next_stmt_id(),
        kind: hir::StmtKind::Return(None),
    });

    module.add_function(hir::Function {
        name: "exact_postcard_varint_array_shape".to_owned(),
        region_params: vec![input_region],
        store_params: vec![],
        params: vec![
            hir::Parameter {
                local: cursor_local,
                name: "cursor".to_owned(),
                ty: hir::Type::named(cursor_def, vec![hir::GenericArg::Region(input_region)]),
                kind: hir::LocalKind::Param,
            },
            hir::Parameter {
                local: out_local,
                name: "out".to_owned(),
                ty: hir::Type::named(root_def, Vec::new()),
                kind: hir::LocalKind::Destination,
            },
        ],
        locals: lowerer.locals,
        return_type: hir::Type::unit(),
        scopes: vec![hir::Scope {
            id: hir::ScopeId::new(0),
            parent: None,
            comment: Some("exact postcard varint array shape".to_owned()),
        }],
        body: hir::Block {
            scope: hir::ScopeId::new(0),
            statements,
        },
    });

    let decoder = compile_structural_hir_decoder(<ScalarArrayHolder>::SHAPE, &module);
    let value = crate::deserialize::<ScalarArrayHolder>(&decoder, &[1, 2, 3, 4])
        .expect("structural HIR decoder should execute exact postcard array shape");
    assert_eq!(
        value,
        ScalarArrayHolder {
            values: [1, 2, 3, 4]
        }
    );
}

#[test]
fn structural_hir_ir_path_preserves_temp_after_cursor_sync() {
    let mut module = hir::Module::new();
    let input_region = module.add_region("input");
    let cursor_def = module.add_type_def(hir::TypeDef {
        name: "Cursor".to_owned(),
        generic_params: vec![hir::GenericParam::Region {
            name: "r_input".to_owned(),
        }],
        kind: hir::TypeDefKind::Struct {
            fields: vec![
                hir::FieldDef {
                    name: "bytes".to_owned(),
                    ty: hir::Type::slice(input_region, hir::Type::u(8)),
                    offset: None,
                },
                hir::FieldDef {
                    name: "pos".to_owned(),
                    ty: hir::Type::u(64),
                    offset: None,
                },
            ],
        },
        size: None,
        transparent: false,
    });
    let root_def = module.add_type_def(hir::TypeDef {
        name: <ConstantNumber>::SHAPE.type_identifier.to_owned(),
        generic_params: vec![],
        kind: hir::TypeDefKind::Struct {
            fields: vec![hir::FieldDef {
                name: "value".to_owned(),
                ty: hir::Type::u(32),
                offset: None,
            }],
        },
        size: None,
        transparent: false,
    });

    module.add_function(hir::Function {
        name: "temp_after_cursor_sync".to_owned(),
        region_params: vec![input_region],
        store_params: vec![],
        params: vec![
            hir::Parameter {
                local: hir::LocalId::new(0),
                name: "cursor".to_owned(),
                ty: hir::Type::named(cursor_def, vec![hir::GenericArg::Region(input_region)]),
                kind: hir::LocalKind::Param,
            },
            hir::Parameter {
                local: hir::LocalId::new(1),
                name: "out".to_owned(),
                ty: hir::Type::named(root_def, Vec::new()),
                kind: hir::LocalKind::Destination,
            },
        ],
        locals: vec![hir::LocalDecl {
            local: hir::LocalId::new(2),
            name: "raw".to_owned(),
            ty: hir::Type::u(8),
            kind: hir::LocalKind::Temp,
        }],
        return_type: hir::Type::unit(),
        scopes: vec![hir::Scope {
            id: hir::ScopeId::new(0),
            parent: None,
            comment: Some("temp survives cursor sync".to_owned()),
        }],
        body: hir::Block {
            scope: hir::ScopeId::new(0),
            statements: vec![
                hir::Stmt {
                    id: hir::StmtId::new(0),
                    kind: hir::StmtKind::Init {
                        place: hir::Place::Local(hir::LocalId::new(2)),
                        value: hir::Expr::Load {
                            addr: Box::new(hir::Expr::SliceData {
                                value: Box::new(hir::Expr::Field {
                                    base: Box::new(hir::Expr::Local(hir::LocalId::new(0))),
                                    field: "bytes".to_owned(),
                                }),
                            }),
                            width: hir::MemoryWidth::W1,
                        },
                    },
                },
                hir::Stmt {
                    id: hir::StmtId::new(1),
                    kind: hir::StmtKind::Assign {
                        place: hir::Place::Field {
                            base: Box::new(hir::Place::Local(hir::LocalId::new(0))),
                            field: "pos".to_owned(),
                        },
                        value: hir::Expr::Literal(hir::Literal::Integer(1)),
                    },
                },
                hir::Stmt {
                    id: hir::StmtId::new(2),
                    kind: hir::StmtKind::Init {
                        place: hir::Place::Field {
                            base: Box::new(hir::Place::Local(hir::LocalId::new(1))),
                            field: "value".to_owned(),
                        },
                        value: hir::Expr::Local(hir::LocalId::new(2)),
                    },
                },
                hir::Stmt {
                    id: hir::StmtId::new(3),
                    kind: hir::StmtKind::Return(None),
                },
            ],
        },
    });

    let decoder = compile_structural_hir_decoder(<ConstantNumber>::SHAPE, &module);
    let value = crate::deserialize::<ConstantNumber>(&decoder, &[42])
        .expect("temp should survive cursor sync");
    assert_eq!(value, ConstantNumber { value: 42 });
}

#[test]
fn structural_hir_ir_path_builds_persistent_buffer_kernel() {
    let mut module = hir::Module::new();
    let root_def = module.add_type_def(hir::TypeDef {
        name: <PersistentBufferSummary>::SHAPE.type_identifier.to_owned(),
        generic_params: vec![],
        kind: hir::TypeDefKind::Struct {
            fields: vec![
                hir::FieldDef {
                    name: "ptr".to_owned(),
                    ty: hir::Type::u(64),
                    offset: None,
                },
                hir::FieldDef {
                    name: "len".to_owned(),
                    ty: hir::Type::u(64),
                    offset: None,
                },
            ],
        },
        size: None,
        transparent: false,
    });
    let runtime = module.install_runtime_memory_callables();
    module.add_function(hir::Function {
        name: "build_persistent_buffer".to_owned(),
        region_params: vec![],
        store_params: vec![],
        params: vec![
            hir::Parameter {
                local: hir::LocalId::new(0),
                name: "cursor".to_owned(),
                ty: hir::Type::u(64),
                kind: hir::LocalKind::Param,
            },
            hir::Parameter {
                local: hir::LocalId::new(1),
                name: "out".to_owned(),
                ty: hir::Type::named(root_def, Vec::new()),
                kind: hir::LocalKind::Destination,
            },
        ],
        locals: vec![
            hir::LocalDecl {
                local: hir::LocalId::new(2),
                name: "len".to_owned(),
                ty: hir::Type::u(64),
                kind: hir::LocalKind::Temp,
            },
            hir::LocalDecl {
                local: hir::LocalId::new(3),
                name: "bytes".to_owned(),
                ty: hir::Type::u(64),
                kind: hir::LocalKind::Temp,
            },
            hir::LocalDecl {
                local: hir::LocalId::new(4),
                name: "ptr".to_owned(),
                ty: hir::Type::persistent_addr(),
                kind: hir::LocalKind::Temp,
            },
        ],
        return_type: hir::Type::unit(),
        scopes: vec![hir::Scope {
            id: hir::ScopeId::new(0),
            parent: None,
            comment: Some("persistent buffer kernel".to_owned()),
        }],
        body: hir::Block {
            scope: hir::ScopeId::new(0),
            statements: vec![
                hir::Stmt {
                    id: hir::StmtId::new(0),
                    kind: hir::StmtKind::Init {
                        place: hir::Place::Local(hir::LocalId::new(2)),
                        value: hir::Expr::Literal(hir::Literal::Integer(2)),
                    },
                },
                hir::Stmt {
                    id: hir::StmtId::new(1),
                    kind: hir::StmtKind::Init {
                        place: hir::Place::Local(hir::LocalId::new(3)),
                        value: hir::Expr::Binary {
                            op: hir::BinaryOp::Mul,
                            lhs: Box::new(hir::Expr::Local(hir::LocalId::new(2))),
                            rhs: Box::new(hir::Expr::Literal(hir::Literal::Integer(4))),
                        },
                    },
                },
                hir::Stmt {
                    id: hir::StmtId::new(2),
                    kind: hir::StmtKind::Init {
                        place: hir::Place::Local(hir::LocalId::new(4)),
                        value: hir::Expr::Call(hir::CallExpr {
                            target: hir::CallTarget::Callable(runtime.alloc_persistent),
                            args: vec![
                                hir::Expr::Local(hir::LocalId::new(3)),
                                hir::Expr::Literal(hir::Literal::Integer(4)),
                            ],
                        }),
                    },
                },
                hir::Stmt {
                    id: hir::StmtId::new(3),
                    kind: hir::StmtKind::Store {
                        addr: hir::Expr::Local(hir::LocalId::new(4)),
                        width: hir::MemoryWidth::W4,
                        value: hir::Expr::Literal(hir::Literal::Integer(10)),
                    },
                },
                hir::Stmt {
                    id: hir::StmtId::new(4),
                    kind: hir::StmtKind::Store {
                        addr: hir::Expr::Binary {
                            op: hir::BinaryOp::Add,
                            lhs: Box::new(hir::Expr::Local(hir::LocalId::new(4))),
                            rhs: Box::new(hir::Expr::Literal(hir::Literal::Integer(4))),
                        },
                        width: hir::MemoryWidth::W4,
                        value: hir::Expr::Literal(hir::Literal::Integer(20)),
                    },
                },
                hir::Stmt {
                    id: hir::StmtId::new(5),
                    kind: hir::StmtKind::Init {
                        place: hir::Place::Field {
                            base: Box::new(hir::Place::Local(hir::LocalId::new(1))),
                            field: "ptr".to_owned(),
                        },
                        value: hir::Expr::Local(hir::LocalId::new(4)),
                    },
                },
                hir::Stmt {
                    id: hir::StmtId::new(6),
                    kind: hir::StmtKind::Init {
                        place: hir::Place::Field {
                            base: Box::new(hir::Place::Local(hir::LocalId::new(1))),
                            field: "len".to_owned(),
                        },
                        value: hir::Expr::Local(hir::LocalId::new(2)),
                    },
                },
                hir::Stmt {
                    id: hir::StmtId::new(7),
                    kind: hir::StmtKind::Return(None),
                },
            ],
        },
    });

    let decoder = compile_structural_hir_decoder(<PersistentBufferSummary>::SHAPE, &module);
    let value = crate::deserialize::<PersistentBufferSummary>(&decoder, &[])
        .expect("structural HIR decoder should build a persistent buffer kernel");
    assert_eq!(value.len, 2);
    assert_ne!(value.ptr, 0);

    let ptr = value.ptr as *const u32;
    let words = unsafe { std::slice::from_raw_parts(ptr, value.len) };
    assert_eq!(words, &[10, 20]);

    let layout = std::alloc::Layout::from_size_align(8, 4).unwrap();
    unsafe { std::alloc::dealloc(value.ptr as *mut u8, layout) };
}

#[test]
fn structural_hir_ir_path_materializes_vec_from_raw_parts() {
    let mut module = hir::Module::new();
    let root_def = module.add_type_def(hir::TypeDef {
        name: <VecHolder>::SHAPE.type_identifier.to_owned(),
        generic_params: vec![],
        kind: hir::TypeDefKind::Struct {
            fields: vec![hir::FieldDef {
                name: "values".to_owned(),
                ty: hir::Type::u(64),
                offset: None,
            }],
        },
        size: None,
        transparent: false,
    });
    let runtime = module.install_runtime_memory_callables();
    module.add_function(hir::Function {
        name: "build_vec_holder".to_owned(),
        region_params: vec![],
        store_params: vec![],
        params: vec![
            hir::Parameter {
                local: hir::LocalId::new(0),
                name: "cursor".to_owned(),
                ty: hir::Type::u(64),
                kind: hir::LocalKind::Param,
            },
            hir::Parameter {
                local: hir::LocalId::new(1),
                name: "out".to_owned(),
                ty: hir::Type::named(root_def, Vec::new()),
                kind: hir::LocalKind::Destination,
            },
        ],
        locals: vec![
            hir::LocalDecl {
                local: hir::LocalId::new(2),
                name: "len".to_owned(),
                ty: hir::Type::u(64),
                kind: hir::LocalKind::Temp,
            },
            hir::LocalDecl {
                local: hir::LocalId::new(3),
                name: "bytes".to_owned(),
                ty: hir::Type::u(64),
                kind: hir::LocalKind::Temp,
            },
            hir::LocalDecl {
                local: hir::LocalId::new(4),
                name: "ptr".to_owned(),
                ty: hir::Type::persistent_addr(),
                kind: hir::LocalKind::Temp,
            },
        ],
        return_type: hir::Type::unit(),
        scopes: vec![hir::Scope {
            id: hir::ScopeId::new(0),
            parent: None,
            comment: Some("vec materialization kernel".to_owned()),
        }],
        body: hir::Block {
            scope: hir::ScopeId::new(0),
            statements: vec![
                hir::Stmt {
                    id: hir::StmtId::new(0),
                    kind: hir::StmtKind::Init {
                        place: hir::Place::Local(hir::LocalId::new(2)),
                        value: hir::Expr::Literal(hir::Literal::Integer(2)),
                    },
                },
                hir::Stmt {
                    id: hir::StmtId::new(1),
                    kind: hir::StmtKind::Init {
                        place: hir::Place::Local(hir::LocalId::new(3)),
                        value: hir::Expr::Binary {
                            op: hir::BinaryOp::Mul,
                            lhs: Box::new(hir::Expr::Local(hir::LocalId::new(2))),
                            rhs: Box::new(hir::Expr::Literal(hir::Literal::Integer(4))),
                        },
                    },
                },
                hir::Stmt {
                    id: hir::StmtId::new(2),
                    kind: hir::StmtKind::Init {
                        place: hir::Place::Local(hir::LocalId::new(4)),
                        value: hir::Expr::Call(hir::CallExpr {
                            target: hir::CallTarget::Callable(runtime.alloc_persistent),
                            args: vec![
                                hir::Expr::Local(hir::LocalId::new(3)),
                                hir::Expr::Literal(hir::Literal::Integer(4)),
                            ],
                        }),
                    },
                },
                hir::Stmt {
                    id: hir::StmtId::new(3),
                    kind: hir::StmtKind::Store {
                        addr: hir::Expr::Local(hir::LocalId::new(4)),
                        width: hir::MemoryWidth::W4,
                        value: hir::Expr::Literal(hir::Literal::Integer(10)),
                    },
                },
                hir::Stmt {
                    id: hir::StmtId::new(4),
                    kind: hir::StmtKind::Store {
                        addr: hir::Expr::Binary {
                            op: hir::BinaryOp::Add,
                            lhs: Box::new(hir::Expr::Local(hir::LocalId::new(4))),
                            rhs: Box::new(hir::Expr::Literal(hir::Literal::Integer(4))),
                        },
                        width: hir::MemoryWidth::W4,
                        value: hir::Expr::Literal(hir::Literal::Integer(20)),
                    },
                },
                hir::Stmt {
                    id: hir::StmtId::new(5),
                    kind: hir::StmtKind::Init {
                        place: hir::Place::Field {
                            base: Box::new(hir::Place::Local(hir::LocalId::new(1))),
                            field: "values".to_owned(),
                        },
                        value: hir::Expr::Call(hir::CallExpr {
                            target: hir::CallTarget::Callable(runtime.vec_from_raw_parts),
                            args: vec![
                                hir::Expr::Local(hir::LocalId::new(4)),
                                hir::Expr::Local(hir::LocalId::new(2)),
                                hir::Expr::Local(hir::LocalId::new(2)),
                                hir::Expr::Literal(hir::Literal::Integer(4)),
                            ],
                        }),
                    },
                },
                hir::Stmt {
                    id: hir::StmtId::new(6),
                    kind: hir::StmtKind::Return(None),
                },
            ],
        },
    });

    let decoder = compile_structural_hir_decoder(<VecHolder>::SHAPE, &module);
    let value = crate::deserialize::<VecHolder>(&decoder, &[])
        .expect("structural HIR decoder should materialize a Vec from raw parts");
    assert_eq!(value.values, vec![10, 20]);
}

#[test]
fn structural_hir_ir_path_materializes_root_vec_from_raw_parts() {
    let mut module = hir::Module::new();
    let runtime = module.install_runtime_memory_callables();
    module.add_function(hir::Function {
        name: "build_root_vec".to_owned(),
        region_params: vec![],
        store_params: vec![],
        params: vec![
            hir::Parameter {
                local: hir::LocalId::new(0),
                name: "cursor".to_owned(),
                ty: hir::Type::u(64),
                kind: hir::LocalKind::Param,
            },
            hir::Parameter {
                local: hir::LocalId::new(1),
                name: "out".to_owned(),
                ty: hir::Type::u(64),
                kind: hir::LocalKind::Destination,
            },
        ],
        locals: vec![
            hir::LocalDecl {
                local: hir::LocalId::new(2),
                name: "len".to_owned(),
                ty: hir::Type::u(64),
                kind: hir::LocalKind::Temp,
            },
            hir::LocalDecl {
                local: hir::LocalId::new(3),
                name: "bytes".to_owned(),
                ty: hir::Type::u(64),
                kind: hir::LocalKind::Temp,
            },
            hir::LocalDecl {
                local: hir::LocalId::new(4),
                name: "ptr".to_owned(),
                ty: hir::Type::persistent_addr(),
                kind: hir::LocalKind::Temp,
            },
        ],
        return_type: hir::Type::unit(),
        scopes: vec![hir::Scope {
            id: hir::ScopeId::new(0),
            parent: None,
            comment: Some("root vec materialization kernel".to_owned()),
        }],
        body: hir::Block {
            scope: hir::ScopeId::new(0),
            statements: vec![
                hir::Stmt {
                    id: hir::StmtId::new(0),
                    kind: hir::StmtKind::Init {
                        place: hir::Place::Local(hir::LocalId::new(2)),
                        value: hir::Expr::Literal(hir::Literal::Integer(2)),
                    },
                },
                hir::Stmt {
                    id: hir::StmtId::new(1),
                    kind: hir::StmtKind::Init {
                        place: hir::Place::Local(hir::LocalId::new(3)),
                        value: hir::Expr::Binary {
                            op: hir::BinaryOp::Mul,
                            lhs: Box::new(hir::Expr::Local(hir::LocalId::new(2))),
                            rhs: Box::new(hir::Expr::Literal(hir::Literal::Integer(4))),
                        },
                    },
                },
                hir::Stmt {
                    id: hir::StmtId::new(2),
                    kind: hir::StmtKind::Init {
                        place: hir::Place::Local(hir::LocalId::new(4)),
                        value: hir::Expr::Call(hir::CallExpr {
                            target: hir::CallTarget::Callable(runtime.alloc_persistent),
                            args: vec![
                                hir::Expr::Local(hir::LocalId::new(3)),
                                hir::Expr::Literal(hir::Literal::Integer(4)),
                            ],
                        }),
                    },
                },
                hir::Stmt {
                    id: hir::StmtId::new(3),
                    kind: hir::StmtKind::Store {
                        addr: hir::Expr::Local(hir::LocalId::new(4)),
                        width: hir::MemoryWidth::W4,
                        value: hir::Expr::Literal(hir::Literal::Integer(10)),
                    },
                },
                hir::Stmt {
                    id: hir::StmtId::new(4),
                    kind: hir::StmtKind::Store {
                        addr: hir::Expr::Binary {
                            op: hir::BinaryOp::Add,
                            lhs: Box::new(hir::Expr::Local(hir::LocalId::new(4))),
                            rhs: Box::new(hir::Expr::Literal(hir::Literal::Integer(4))),
                        },
                        width: hir::MemoryWidth::W4,
                        value: hir::Expr::Literal(hir::Literal::Integer(20)),
                    },
                },
                hir::Stmt {
                    id: hir::StmtId::new(5),
                    kind: hir::StmtKind::Init {
                        place: hir::Place::Local(hir::LocalId::new(1)),
                        value: hir::Expr::Call(hir::CallExpr {
                            target: hir::CallTarget::Callable(runtime.vec_from_raw_parts),
                            args: vec![
                                hir::Expr::Local(hir::LocalId::new(4)),
                                hir::Expr::Local(hir::LocalId::new(2)),
                                hir::Expr::Local(hir::LocalId::new(2)),
                                hir::Expr::Literal(hir::Literal::Integer(4)),
                            ],
                        }),
                    },
                },
                hir::Stmt {
                    id: hir::StmtId::new(6),
                    kind: hir::StmtKind::Return(None),
                },
            ],
        },
    });

    let decoder = compile_structural_hir_decoder(<Vec<u32>>::SHAPE, &module);
    let value = crate::deserialize::<Vec<u32>>(&decoder, &[])
        .expect("structural HIR decoder should materialize a root Vec from raw parts");
    assert_eq!(value, vec![10, 20]);
}

#[test]
fn structural_hir_ir_path_materializes_empty_vec_from_raw_parts() {
    let mut module = hir::Module::new();
    let root_def = module.add_type_def(hir::TypeDef {
        name: <VecHolder>::SHAPE.type_identifier.to_owned(),
        generic_params: vec![],
        kind: hir::TypeDefKind::Struct {
            fields: vec![hir::FieldDef {
                name: "values".to_owned(),
                ty: hir::Type::u(64),
                offset: None,
            }],
        },
        size: None,
        transparent: false,
    });
    let runtime = module.install_runtime_memory_callables();
    module.add_function(hir::Function {
        name: "build_empty_vec_holder".to_owned(),
        region_params: vec![],
        store_params: vec![],
        params: vec![
            hir::Parameter {
                local: hir::LocalId::new(0),
                name: "cursor".to_owned(),
                ty: hir::Type::u(64),
                kind: hir::LocalKind::Param,
            },
            hir::Parameter {
                local: hir::LocalId::new(1),
                name: "out".to_owned(),
                ty: hir::Type::named(root_def, Vec::new()),
                kind: hir::LocalKind::Destination,
            },
        ],
        locals: vec![],
        return_type: hir::Type::unit(),
        scopes: vec![hir::Scope {
            id: hir::ScopeId::new(0),
            parent: None,
            comment: Some("empty vec materialization kernel".to_owned()),
        }],
        body: hir::Block {
            scope: hir::ScopeId::new(0),
            statements: vec![
                hir::Stmt {
                    id: hir::StmtId::new(0),
                    kind: hir::StmtKind::Init {
                        place: hir::Place::Field {
                            base: Box::new(hir::Place::Local(hir::LocalId::new(1))),
                            field: "values".to_owned(),
                        },
                        value: hir::Expr::Call(hir::CallExpr {
                            target: hir::CallTarget::Callable(runtime.vec_from_raw_parts),
                            args: vec![
                                hir::Expr::Literal(hir::Literal::Integer(0)),
                                hir::Expr::Literal(hir::Literal::Integer(0)),
                                hir::Expr::Literal(hir::Literal::Integer(0)),
                                hir::Expr::Literal(hir::Literal::Integer(4)),
                            ],
                        }),
                    },
                },
                hir::Stmt {
                    id: hir::StmtId::new(1),
                    kind: hir::StmtKind::Return(None),
                },
            ],
        },
    });

    let decoder = compile_structural_hir_decoder(<VecHolder>::SHAPE, &module);
    let value = crate::deserialize::<VecHolder>(&decoder, &[])
        .expect("structural HIR decoder should materialize an empty Vec from raw parts");
    assert!(value.values.is_empty());
}

#[test]
fn structural_hir_ir_path_loads_from_persistent_buffer() {
    let mut module = hir::Module::new();
    let root_def = module.add_type_def(hir::TypeDef {
        name: <ScalarNumber>::SHAPE.type_identifier.to_owned(),
        generic_params: vec![],
        kind: hir::TypeDefKind::Struct {
            fields: vec![hir::FieldDef {
                name: "value".to_owned(),
                ty: hir::Type::u(32),
                offset: None,
            }],
        },
        size: None,
        transparent: false,
    });
    let runtime = module.install_runtime_memory_callables();
    module.add_function(hir::Function {
        name: "load_persistent_word".to_owned(),
        region_params: vec![],
        store_params: vec![],
        params: vec![
            hir::Parameter {
                local: hir::LocalId::new(0),
                name: "cursor".to_owned(),
                ty: hir::Type::u(64),
                kind: hir::LocalKind::Param,
            },
            hir::Parameter {
                local: hir::LocalId::new(1),
                name: "out".to_owned(),
                ty: hir::Type::named(root_def, Vec::new()),
                kind: hir::LocalKind::Destination,
            },
        ],
        locals: vec![hir::LocalDecl {
            local: hir::LocalId::new(2),
            name: "ptr".to_owned(),
            ty: hir::Type::persistent_addr(),
            kind: hir::LocalKind::Temp,
        }],
        return_type: hir::Type::unit(),
        scopes: vec![hir::Scope {
            id: hir::ScopeId::new(0),
            parent: None,
            comment: Some("persistent buffer load kernel".to_owned()),
        }],
        body: hir::Block {
            scope: hir::ScopeId::new(0),
            statements: vec![
                hir::Stmt {
                    id: hir::StmtId::new(0),
                    kind: hir::StmtKind::Init {
                        place: hir::Place::Local(hir::LocalId::new(2)),
                        value: hir::Expr::Call(hir::CallExpr {
                            target: hir::CallTarget::Callable(runtime.alloc_persistent),
                            args: vec![
                                hir::Expr::Literal(hir::Literal::Integer(4)),
                                hir::Expr::Literal(hir::Literal::Integer(4)),
                            ],
                        }),
                    },
                },
                hir::Stmt {
                    id: hir::StmtId::new(1),
                    kind: hir::StmtKind::Store {
                        addr: hir::Expr::Local(hir::LocalId::new(2)),
                        width: hir::MemoryWidth::W4,
                        value: hir::Expr::Literal(hir::Literal::Integer(0x1234)),
                    },
                },
                hir::Stmt {
                    id: hir::StmtId::new(2),
                    kind: hir::StmtKind::Init {
                        place: hir::Place::Field {
                            base: Box::new(hir::Place::Local(hir::LocalId::new(1))),
                            field: "value".to_owned(),
                        },
                        value: hir::Expr::Load {
                            addr: Box::new(hir::Expr::Local(hir::LocalId::new(2))),
                            width: hir::MemoryWidth::W4,
                        },
                    },
                },
                hir::Stmt {
                    id: hir::StmtId::new(3),
                    kind: hir::StmtKind::Return(None),
                },
            ],
        },
    });

    let decoder = compile_structural_hir_decoder(<ScalarNumber>::SHAPE, &module);
    let value = crate::deserialize::<ScalarNumber>(&decoder, &[])
        .expect("structural HIR decoder should load from persistent buffer");
    assert_eq!(value, ScalarNumber { value: 0x1234 });
}

#[test]
fn structural_hir_ir_path_executes_loop_break_and_continue() {
    let mut module = hir::Module::new();
    let root_def = module.add_type_def(hir::TypeDef {
        name: <ConstantNumber>::SHAPE.type_identifier.to_owned(),
        generic_params: vec![],
        kind: hir::TypeDefKind::Struct {
            fields: vec![hir::FieldDef {
                name: "value".to_owned(),
                ty: hir::Type::u(32),
                offset: None,
            }],
        },
        size: None,
        transparent: false,
    });

    module.add_function(hir::Function {
        name: "loop_break_continue".to_owned(),
        region_params: vec![],
        store_params: vec![],
        params: vec![
            hir::Parameter {
                local: hir::LocalId::new(0),
                name: "cursor".to_owned(),
                ty: hir::Type::u(64),
                kind: hir::LocalKind::Param,
            },
            hir::Parameter {
                local: hir::LocalId::new(1),
                name: "out".to_owned(),
                ty: hir::Type::named(root_def, Vec::new()),
                kind: hir::LocalKind::Destination,
            },
        ],
        locals: vec![
            hir::LocalDecl {
                local: hir::LocalId::new(2),
                name: "i".to_owned(),
                ty: hir::Type::u(64),
                kind: hir::LocalKind::Temp,
            },
            hir::LocalDecl {
                local: hir::LocalId::new(3),
                name: "sum".to_owned(),
                ty: hir::Type::u(64),
                kind: hir::LocalKind::Temp,
            },
        ],
        return_type: hir::Type::unit(),
        scopes: vec![hir::Scope {
            id: hir::ScopeId::new(0),
            parent: None,
            comment: Some("loop break/continue kernel".to_owned()),
        }],
        body: hir::Block {
            scope: hir::ScopeId::new(0),
            statements: vec![
                hir::Stmt {
                    id: hir::StmtId::new(0),
                    kind: hir::StmtKind::Init {
                        place: hir::Place::Local(hir::LocalId::new(2)),
                        value: hir::Expr::Literal(hir::Literal::Integer(0)),
                    },
                },
                hir::Stmt {
                    id: hir::StmtId::new(1),
                    kind: hir::StmtKind::Init {
                        place: hir::Place::Local(hir::LocalId::new(3)),
                        value: hir::Expr::Literal(hir::Literal::Integer(0)),
                    },
                },
                hir::Stmt {
                    id: hir::StmtId::new(2),
                    kind: hir::StmtKind::Loop {
                        body: hir::Block {
                            scope: hir::ScopeId::new(0),
                            statements: vec![
                                hir::Stmt {
                                    id: hir::StmtId::new(3),
                                    kind: hir::StmtKind::If {
                                        condition: hir::Expr::Binary {
                                            op: hir::BinaryOp::Eq,
                                            lhs: Box::new(hir::Expr::Local(hir::LocalId::new(2))),
                                            rhs: Box::new(hir::Expr::Literal(
                                                hir::Literal::Integer(5),
                                            )),
                                        },
                                        then_block: hir::Block {
                                            scope: hir::ScopeId::new(0),
                                            statements: vec![hir::Stmt {
                                                id: hir::StmtId::new(4),
                                                kind: hir::StmtKind::Break,
                                            }],
                                        },
                                        else_block: Some(hir::Block {
                                            scope: hir::ScopeId::new(0),
                                            statements: vec![],
                                        }),
                                    },
                                },
                                hir::Stmt {
                                    id: hir::StmtId::new(5),
                                    kind: hir::StmtKind::Assign {
                                        place: hir::Place::Local(hir::LocalId::new(2)),
                                        value: hir::Expr::Binary {
                                            op: hir::BinaryOp::Add,
                                            lhs: Box::new(hir::Expr::Local(hir::LocalId::new(2))),
                                            rhs: Box::new(hir::Expr::Literal(
                                                hir::Literal::Integer(1),
                                            )),
                                        },
                                    },
                                },
                                hir::Stmt {
                                    id: hir::StmtId::new(6),
                                    kind: hir::StmtKind::If {
                                        condition: hir::Expr::Binary {
                                            op: hir::BinaryOp::Eq,
                                            lhs: Box::new(hir::Expr::Binary {
                                                op: hir::BinaryOp::BitAnd,
                                                lhs: Box::new(hir::Expr::Local(hir::LocalId::new(
                                                    2,
                                                ))),
                                                rhs: Box::new(hir::Expr::Literal(
                                                    hir::Literal::Integer(1),
                                                )),
                                            }),
                                            rhs: Box::new(hir::Expr::Literal(
                                                hir::Literal::Integer(0),
                                            )),
                                        },
                                        then_block: hir::Block {
                                            scope: hir::ScopeId::new(0),
                                            statements: vec![hir::Stmt {
                                                id: hir::StmtId::new(7),
                                                kind: hir::StmtKind::Continue,
                                            }],
                                        },
                                        else_block: Some(hir::Block {
                                            scope: hir::ScopeId::new(0),
                                            statements: vec![],
                                        }),
                                    },
                                },
                                hir::Stmt {
                                    id: hir::StmtId::new(8),
                                    kind: hir::StmtKind::Assign {
                                        place: hir::Place::Local(hir::LocalId::new(3)),
                                        value: hir::Expr::Binary {
                                            op: hir::BinaryOp::Add,
                                            lhs: Box::new(hir::Expr::Local(hir::LocalId::new(3))),
                                            rhs: Box::new(hir::Expr::Local(hir::LocalId::new(2))),
                                        },
                                    },
                                },
                            ],
                        },
                    },
                },
                hir::Stmt {
                    id: hir::StmtId::new(9),
                    kind: hir::StmtKind::Init {
                        place: hir::Place::Field {
                            base: Box::new(hir::Place::Local(hir::LocalId::new(1))),
                            field: "value".to_owned(),
                        },
                        value: hir::Expr::Local(hir::LocalId::new(3)),
                    },
                },
                hir::Stmt {
                    id: hir::StmtId::new(10),
                    kind: hir::StmtKind::Return(None),
                },
            ],
        },
    });

    let decoder = compile_structural_hir_decoder(<ConstantNumber>::SHAPE, &module);
    let value = crate::deserialize::<ConstantNumber>(&decoder, &[])
        .expect("structural HIR decoder should execute loops with break/continue");
    assert_eq!(value, ConstantNumber { value: 9 });
}

#[test]
fn structural_hir_ir_path_decodes_if_and_match() {
    let mut module = hir::Module::new();
    let animal_def = module.add_type_def(hir::TypeDef {
        name: <UnitAnimal>::SHAPE.type_identifier.to_owned(),
        generic_params: vec![],
        kind: hir::TypeDefKind::Enum {
            variants: vec![
                hir::VariantDef {
                    name: "Cat".to_owned(),
                    fields: vec![],
                    discriminant: None,
                },
                hir::VariantDef {
                    name: "Dog".to_owned(),
                    fields: vec![],
                    discriminant: None,
                },
                hir::VariantDef {
                    name: "Parrot".to_owned(),
                    fields: vec![],
                    discriminant: None,
                },
            ],
            discriminant_width: None,
        },
        size: None,
        transparent: false,
    });
    let root_def = module.add_type_def(hir::TypeDef {
        name: <BranchyAnimal>::SHAPE.type_identifier.to_owned(),
        generic_params: vec![],
        kind: hir::TypeDefKind::Struct {
            fields: vec![
                hir::FieldDef {
                    name: "animal".to_owned(),
                    ty: hir::Type::named(animal_def, Vec::new()),
                    offset: None,
                },
                hir::FieldDef {
                    name: "value".to_owned(),
                    ty: hir::Type::u(32),
                    offset: None,
                },
            ],
        },
        size: None,
        transparent: false,
    });

    module.add_function(hir::Function {
        name: "branchy_animal".to_owned(),
        region_params: vec![],
        store_params: vec![],
        params: vec![
            hir::Parameter {
                local: hir::LocalId::new(0),
                name: "cursor".to_owned(),
                ty: hir::Type::u(64),
                kind: hir::LocalKind::Param,
            },
            hir::Parameter {
                local: hir::LocalId::new(1),
                name: "out".to_owned(),
                ty: hir::Type::named(root_def, Vec::new()),
                kind: hir::LocalKind::Destination,
            },
        ],
        locals: vec![
            hir::LocalDecl {
                local: hir::LocalId::new(2),
                name: "flag".to_owned(),
                ty: hir::Type::bool(),
                kind: hir::LocalKind::Let,
            },
            hir::LocalDecl {
                local: hir::LocalId::new(3),
                name: "tag".to_owned(),
                ty: hir::Type::u(32),
                kind: hir::LocalKind::Let,
            },
        ],
        return_type: hir::Type::unit(),
        scopes: vec![hir::Scope {
            id: hir::ScopeId::new(0),
            parent: None,
            comment: Some("structural if/match HIR".to_owned()),
        }],
        body: hir::Block {
            scope: hir::ScopeId::new(0),
            statements: vec![
                hir::Stmt {
                    id: hir::StmtId::new(0),
                    kind: hir::StmtKind::Init {
                        place: hir::Place::Local(hir::LocalId::new(2)),
                        value: hir::Expr::Literal(hir::Literal::Bool(true)),
                    },
                },
                hir::Stmt {
                    id: hir::StmtId::new(1),
                    kind: hir::StmtKind::If {
                        condition: hir::Expr::Local(hir::LocalId::new(2)),
                        then_block: hir::Block {
                            scope: hir::ScopeId::new(0),
                            statements: vec![hir::Stmt {
                                id: hir::StmtId::new(2),
                                kind: hir::StmtKind::Init {
                                    place: hir::Place::Field {
                                        base: Box::new(hir::Place::Local(hir::LocalId::new(1))),
                                        field: "animal".to_owned(),
                                    },
                                    value: hir::Expr::Variant {
                                        def: animal_def,
                                        variant: "Dog".to_owned(),
                                        fields: vec![],
                                    },
                                },
                            }],
                        },
                        else_block: Some(hir::Block {
                            scope: hir::ScopeId::new(0),
                            statements: vec![hir::Stmt {
                                id: hir::StmtId::new(3),
                                kind: hir::StmtKind::Init {
                                    place: hir::Place::Field {
                                        base: Box::new(hir::Place::Local(hir::LocalId::new(1))),
                                        field: "animal".to_owned(),
                                    },
                                    value: hir::Expr::Variant {
                                        def: animal_def,
                                        variant: "Cat".to_owned(),
                                        fields: vec![],
                                    },
                                },
                            }],
                        }),
                    },
                },
                hir::Stmt {
                    id: hir::StmtId::new(4),
                    kind: hir::StmtKind::Init {
                        place: hir::Place::Local(hir::LocalId::new(3)),
                        value: hir::Expr::Literal(hir::Literal::Integer(1)),
                    },
                },
                hir::Stmt {
                    id: hir::StmtId::new(5),
                    kind: hir::StmtKind::Match {
                        scrutinee: hir::Expr::Local(hir::LocalId::new(3)),
                        arms: vec![
                            hir::MatchArm {
                                pattern: hir::Pattern::Integer(0),
                                body: hir::Block {
                                    scope: hir::ScopeId::new(0),
                                    statements: vec![hir::Stmt {
                                        id: hir::StmtId::new(6),
                                        kind: hir::StmtKind::Init {
                                            place: hir::Place::Field {
                                                base: Box::new(hir::Place::Local(
                                                    hir::LocalId::new(1),
                                                )),
                                                field: "value".to_owned(),
                                            },
                                            value: hir::Expr::Literal(hir::Literal::Integer(7)),
                                        },
                                    }],
                                },
                            },
                            hir::MatchArm {
                                pattern: hir::Pattern::Integer(1),
                                body: hir::Block {
                                    scope: hir::ScopeId::new(0),
                                    statements: vec![hir::Stmt {
                                        id: hir::StmtId::new(7),
                                        kind: hir::StmtKind::Init {
                                            place: hir::Place::Field {
                                                base: Box::new(hir::Place::Local(
                                                    hir::LocalId::new(1),
                                                )),
                                                field: "value".to_owned(),
                                            },
                                            value: hir::Expr::Literal(hir::Literal::Integer(42)),
                                        },
                                    }],
                                },
                            },
                        ],
                    },
                },
                hir::Stmt {
                    id: hir::StmtId::new(8),
                    kind: hir::StmtKind::Return(None),
                },
            ],
        },
    });

    let decoder = compile_structural_hir_decoder(<BranchyAnimal>::SHAPE, &module);
    let value = crate::deserialize::<BranchyAnimal>(&decoder, &[])
        .expect("structural HIR decoder should lower if+match");
    assert_eq!(
        value,
        BranchyAnimal {
            animal: UnitAnimal::Dog,
            value: 42,
        }
    );
}

#[test]
fn structural_hir_ir_path_computes_bit_masks() {
    let mut module = hir::Module::new();
    let root_def = module.add_type_def(hir::TypeDef {
        name: <MaskSummary>::SHAPE.type_identifier.to_owned(),
        generic_params: vec![],
        kind: hir::TypeDefKind::Struct {
            fields: vec![
                hir::FieldDef {
                    name: "masked".to_owned(),
                    ty: hir::Type::u(32),
                    offset: None,
                },
                hir::FieldDef {
                    name: "shifted".to_owned(),
                    ty: hir::Type::u(32),
                    offset: None,
                },
                hir::FieldDef {
                    name: "toggled".to_owned(),
                    ty: hir::Type::u(32),
                    offset: None,
                },
                hir::FieldDef {
                    name: "combined".to_owned(),
                    ty: hir::Type::u(32),
                    offset: None,
                },
            ],
        },
        size: None,
        transparent: false,
    });

    module.add_function(hir::Function {
        name: "mask_summary".to_owned(),
        region_params: vec![],
        store_params: vec![],
        params: vec![
            hir::Parameter {
                local: hir::LocalId::new(0),
                name: "cursor".to_owned(),
                ty: hir::Type::u(64),
                kind: hir::LocalKind::Param,
            },
            hir::Parameter {
                local: hir::LocalId::new(1),
                name: "out".to_owned(),
                ty: hir::Type::named(root_def, Vec::new()),
                kind: hir::LocalKind::Destination,
            },
        ],
        locals: vec![
            hir::LocalDecl {
                local: hir::LocalId::new(2),
                name: "mask".to_owned(),
                ty: hir::Type::u(32),
                kind: hir::LocalKind::Let,
            },
            hir::LocalDecl {
                local: hir::LocalId::new(3),
                name: "masked".to_owned(),
                ty: hir::Type::u(32),
                kind: hir::LocalKind::Let,
            },
            hir::LocalDecl {
                local: hir::LocalId::new(4),
                name: "shifted".to_owned(),
                ty: hir::Type::u(32),
                kind: hir::LocalKind::Let,
            },
            hir::LocalDecl {
                local: hir::LocalId::new(5),
                name: "toggled".to_owned(),
                ty: hir::Type::u(32),
                kind: hir::LocalKind::Let,
            },
            hir::LocalDecl {
                local: hir::LocalId::new(6),
                name: "combined".to_owned(),
                ty: hir::Type::u(32),
                kind: hir::LocalKind::Let,
            },
        ],
        return_type: hir::Type::unit(),
        scopes: vec![hir::Scope {
            id: hir::ScopeId::new(0),
            parent: None,
            comment: Some("structural bit-mask HIR".to_owned()),
        }],
        body: hir::Block {
            scope: hir::ScopeId::new(0),
            statements: vec![
                hir::Stmt {
                    id: hir::StmtId::new(0),
                    kind: hir::StmtKind::Init {
                        place: hir::Place::Local(hir::LocalId::new(2)),
                        value: hir::Expr::Literal(hir::Literal::Integer(0b1111)),
                    },
                },
                hir::Stmt {
                    id: hir::StmtId::new(1),
                    kind: hir::StmtKind::Init {
                        place: hir::Place::Local(hir::LocalId::new(3)),
                        value: hir::Expr::Binary {
                            op: hir::BinaryOp::BitAnd,
                            lhs: Box::new(hir::Expr::Local(hir::LocalId::new(2))),
                            rhs: Box::new(hir::Expr::Literal(hir::Literal::Integer(0b1011))),
                        },
                    },
                },
                hir::Stmt {
                    id: hir::StmtId::new(2),
                    kind: hir::StmtKind::Init {
                        place: hir::Place::Local(hir::LocalId::new(4)),
                        value: hir::Expr::Binary {
                            op: hir::BinaryOp::Shr,
                            lhs: Box::new(hir::Expr::Local(hir::LocalId::new(3))),
                            rhs: Box::new(hir::Expr::Literal(hir::Literal::Integer(1))),
                        },
                    },
                },
                hir::Stmt {
                    id: hir::StmtId::new(3),
                    kind: hir::StmtKind::Init {
                        place: hir::Place::Local(hir::LocalId::new(5)),
                        value: hir::Expr::Binary {
                            op: hir::BinaryOp::Xor,
                            lhs: Box::new(hir::Expr::Local(hir::LocalId::new(3))),
                            rhs: Box::new(hir::Expr::Literal(hir::Literal::Integer(0b0011))),
                        },
                    },
                },
                hir::Stmt {
                    id: hir::StmtId::new(4),
                    kind: hir::StmtKind::Init {
                        place: hir::Place::Local(hir::LocalId::new(6)),
                        value: hir::Expr::Binary {
                            op: hir::BinaryOp::BitOr,
                            lhs: Box::new(hir::Expr::Binary {
                                op: hir::BinaryOp::Shl,
                                lhs: Box::new(hir::Expr::Literal(hir::Literal::Integer(1))),
                                rhs: Box::new(hir::Expr::Literal(hir::Literal::Integer(3))),
                            }),
                            rhs: Box::new(hir::Expr::Literal(hir::Literal::Integer(1))),
                        },
                    },
                },
                hir::Stmt {
                    id: hir::StmtId::new(5),
                    kind: hir::StmtKind::Init {
                        place: hir::Place::Field {
                            base: Box::new(hir::Place::Local(hir::LocalId::new(1))),
                            field: "masked".to_owned(),
                        },
                        value: hir::Expr::Local(hir::LocalId::new(3)),
                    },
                },
                hir::Stmt {
                    id: hir::StmtId::new(6),
                    kind: hir::StmtKind::Init {
                        place: hir::Place::Field {
                            base: Box::new(hir::Place::Local(hir::LocalId::new(1))),
                            field: "shifted".to_owned(),
                        },
                        value: hir::Expr::Local(hir::LocalId::new(4)),
                    },
                },
                hir::Stmt {
                    id: hir::StmtId::new(7),
                    kind: hir::StmtKind::Init {
                        place: hir::Place::Field {
                            base: Box::new(hir::Place::Local(hir::LocalId::new(1))),
                            field: "toggled".to_owned(),
                        },
                        value: hir::Expr::Local(hir::LocalId::new(5)),
                    },
                },
                hir::Stmt {
                    id: hir::StmtId::new(8),
                    kind: hir::StmtKind::Init {
                        place: hir::Place::Field {
                            base: Box::new(hir::Place::Local(hir::LocalId::new(1))),
                            field: "combined".to_owned(),
                        },
                        value: hir::Expr::Local(hir::LocalId::new(6)),
                    },
                },
                hir::Stmt {
                    id: hir::StmtId::new(9),
                    kind: hir::StmtKind::Return(None),
                },
            ],
        },
    });

    let decoder = compile_structural_hir_decoder(<MaskSummary>::SHAPE, &module);
    let value = crate::deserialize::<MaskSummary>(&decoder, &[])
        .expect("structural HIR decoder should compute bit-mask values");
    assert_eq!(
        value,
        MaskSummary {
            masked: 0b1011,
            shifted: 0b0101,
            toggled: 0b1000,
            combined: 0b1001,
        }
    );
}

#[test]
fn structural_hir_ir_path_updates_local_scratch_struct_fields() {
    let mut module = hir::Module::new();
    let scratch_def = module.add_type_def(hir::TypeDef {
        name: "ScratchState".to_owned(),
        generic_params: vec![],
        kind: hir::TypeDefKind::Struct {
            fields: vec![
                hir::FieldDef {
                    name: "mask".to_owned(),
                    ty: hir::Type::u(32),
                    offset: None,
                },
                hir::FieldDef {
                    name: "done".to_owned(),
                    ty: hir::Type::u(32),
                    offset: None,
                },
            ],
        },
        size: None,
        transparent: false,
    });
    let root_def = module.add_type_def(hir::TypeDef {
        name: <ScratchSummary>::SHAPE.type_identifier.to_owned(),
        generic_params: vec![],
        kind: hir::TypeDefKind::Struct {
            fields: vec![
                hir::FieldDef {
                    name: "mask".to_owned(),
                    ty: hir::Type::u(32),
                    offset: None,
                },
                hir::FieldDef {
                    name: "done".to_owned(),
                    ty: hir::Type::u(32),
                    offset: None,
                },
            ],
        },
        size: None,
        transparent: false,
    });

    module.add_function(hir::Function {
        name: "scratch_summary".to_owned(),
        region_params: vec![],
        store_params: vec![],
        params: vec![
            hir::Parameter {
                local: hir::LocalId::new(0),
                name: "cursor".to_owned(),
                ty: hir::Type::u(64),
                kind: hir::LocalKind::Param,
            },
            hir::Parameter {
                local: hir::LocalId::new(1),
                name: "out".to_owned(),
                ty: hir::Type::named(root_def, Vec::new()),
                kind: hir::LocalKind::Destination,
            },
        ],
        locals: vec![hir::LocalDecl {
            local: hir::LocalId::new(2),
            name: "scratch".to_owned(),
            ty: hir::Type::named(scratch_def, Vec::new()),
            kind: hir::LocalKind::Let,
        }],
        return_type: hir::Type::unit(),
        scopes: vec![hir::Scope {
            id: hir::ScopeId::new(0),
            parent: None,
            comment: Some("structural local scratch-state HIR".to_owned()),
        }],
        body: hir::Block {
            scope: hir::ScopeId::new(0),
            statements: vec![
                hir::Stmt {
                    id: hir::StmtId::new(0),
                    kind: hir::StmtKind::Init {
                        place: hir::Place::Field {
                            base: Box::new(hir::Place::Local(hir::LocalId::new(2))),
                            field: "mask".to_owned(),
                        },
                        value: hir::Expr::Literal(hir::Literal::Integer(0b1111)),
                    },
                },
                hir::Stmt {
                    id: hir::StmtId::new(1),
                    kind: hir::StmtKind::Init {
                        place: hir::Place::Field {
                            base: Box::new(hir::Place::Local(hir::LocalId::new(2))),
                            field: "done".to_owned(),
                        },
                        value: hir::Expr::Binary {
                            op: hir::BinaryOp::BitAnd,
                            lhs: Box::new(hir::Expr::Field {
                                base: Box::new(hir::Expr::Local(hir::LocalId::new(2))),
                                field: "mask".to_owned(),
                            }),
                            rhs: Box::new(hir::Expr::Literal(hir::Literal::Integer(0b0011))),
                        },
                    },
                },
                hir::Stmt {
                    id: hir::StmtId::new(2),
                    kind: hir::StmtKind::Init {
                        place: hir::Place::Field {
                            base: Box::new(hir::Place::Local(hir::LocalId::new(1))),
                            field: "mask".to_owned(),
                        },
                        value: hir::Expr::Field {
                            base: Box::new(hir::Expr::Local(hir::LocalId::new(2))),
                            field: "mask".to_owned(),
                        },
                    },
                },
                hir::Stmt {
                    id: hir::StmtId::new(3),
                    kind: hir::StmtKind::Init {
                        place: hir::Place::Field {
                            base: Box::new(hir::Place::Local(hir::LocalId::new(1))),
                            field: "done".to_owned(),
                        },
                        value: hir::Expr::Field {
                            base: Box::new(hir::Expr::Local(hir::LocalId::new(2))),
                            field: "done".to_owned(),
                        },
                    },
                },
                hir::Stmt {
                    id: hir::StmtId::new(4),
                    kind: hir::StmtKind::Return(None),
                },
            ],
        },
    });

    let decoder = compile_structural_hir_decoder(<ScratchSummary>::SHAPE, &module);
    let value = crate::deserialize::<ScratchSummary>(&decoder, &[])
        .expect("structural HIR decoder should support local scratch-state fields");
    assert_eq!(
        value,
        ScratchSummary {
            mask: 0b1111,
            done: 0b0011,
        }
    );
}

#[test]
fn structural_hir_ir_path_updates_dynamic_local_array_elements() {
    let mut module = hir::Module::new();
    let root_def = module.add_type_def(hir::TypeDef {
        name: <DynamicIndexSummary>::SHAPE.type_identifier.to_owned(),
        generic_params: vec![],
        kind: hir::TypeDefKind::Struct {
            fields: vec![hir::FieldDef {
                name: "selected".to_owned(),
                ty: hir::Type::u(32),
                offset: None,
            }],
        },
        size: None,
        transparent: false,
    });

    module.add_function(hir::Function {
        name: "dynamic_index_summary".to_owned(),
        region_params: vec![],
        store_params: vec![],
        params: vec![
            hir::Parameter {
                local: hir::LocalId::new(0),
                name: "cursor".to_owned(),
                ty: hir::Type::u(64),
                kind: hir::LocalKind::Param,
            },
            hir::Parameter {
                local: hir::LocalId::new(1),
                name: "out".to_owned(),
                ty: hir::Type::named(root_def, Vec::new()),
                kind: hir::LocalKind::Destination,
            },
        ],
        locals: vec![
            hir::LocalDecl {
                local: hir::LocalId::new(2),
                name: "scratch".to_owned(),
                ty: hir::Type::array(hir::Type::u(32), 4),
                kind: hir::LocalKind::Let,
            },
            hir::LocalDecl {
                local: hir::LocalId::new(3),
                name: "idx".to_owned(),
                ty: hir::Type::u(32),
                kind: hir::LocalKind::Let,
            },
        ],
        return_type: hir::Type::unit(),
        scopes: vec![hir::Scope {
            id: hir::ScopeId::new(0),
            parent: None,
            comment: Some("structural dynamic indexed scratch-array HIR".to_owned()),
        }],
        body: hir::Block {
            scope: hir::ScopeId::new(0),
            statements: vec![
                hir::Stmt {
                    id: hir::StmtId::new(0),
                    kind: hir::StmtKind::Init {
                        place: hir::Place::Local(hir::LocalId::new(3)),
                        value: hir::Expr::Literal(hir::Literal::Integer(2)),
                    },
                },
                hir::Stmt {
                    id: hir::StmtId::new(1),
                    kind: hir::StmtKind::Assign {
                        place: hir::Place::Index {
                            base: Box::new(hir::Place::Local(hir::LocalId::new(2))),
                            index: Box::new(hir::Expr::Local(hir::LocalId::new(3))),
                        },
                        value: hir::Expr::Literal(hir::Literal::Integer(42)),
                    },
                },
                hir::Stmt {
                    id: hir::StmtId::new(2),
                    kind: hir::StmtKind::Init {
                        place: hir::Place::Field {
                            base: Box::new(hir::Place::Local(hir::LocalId::new(1))),
                            field: "selected".to_owned(),
                        },
                        value: hir::Expr::Index {
                            base: Box::new(hir::Expr::Local(hir::LocalId::new(2))),
                            index: Box::new(hir::Expr::Local(hir::LocalId::new(3))),
                        },
                    },
                },
                hir::Stmt {
                    id: hir::StmtId::new(3),
                    kind: hir::StmtKind::Return(None),
                },
            ],
        },
    });

    let decoder = compile_structural_hir_decoder(<DynamicIndexSummary>::SHAPE, &module);
    let value = crate::deserialize::<DynamicIndexSummary>(&decoder, &[])
        .expect("structural HIR decoder should support computed local array indexing");
    assert_eq!(value, DynamicIndexSummary { selected: 42 });
}

#[test]
fn structural_hir_ir_path_updates_dynamic_destination_array_elements() {
    let mut module = hir::Module::new();
    let root_def = module.add_type_def(hir::TypeDef {
        name: <DynamicDestinationSummary>::SHAPE
            .type_identifier
            .to_owned(),
        generic_params: vec![],
        kind: hir::TypeDefKind::Struct {
            fields: vec![
                hir::FieldDef {
                    name: "values".to_owned(),
                    ty: hir::Type::array(hir::Type::u(32), 4),
                    offset: None,
                },
                hir::FieldDef {
                    name: "selected".to_owned(),
                    ty: hir::Type::u(32),
                    offset: None,
                },
            ],
        },
        size: None,
        transparent: false,
    });

    module.add_function(hir::Function {
        name: "dynamic_destination_summary".to_owned(),
        region_params: vec![],
        store_params: vec![],
        params: vec![
            hir::Parameter {
                local: hir::LocalId::new(0),
                name: "cursor".to_owned(),
                ty: hir::Type::u(64),
                kind: hir::LocalKind::Param,
            },
            hir::Parameter {
                local: hir::LocalId::new(1),
                name: "out".to_owned(),
                ty: hir::Type::named(root_def, Vec::new()),
                kind: hir::LocalKind::Destination,
            },
        ],
        locals: vec![hir::LocalDecl {
            local: hir::LocalId::new(2),
            name: "idx".to_owned(),
            ty: hir::Type::u(32),
            kind: hir::LocalKind::Let,
        }],
        return_type: hir::Type::unit(),
        scopes: vec![hir::Scope {
            id: hir::ScopeId::new(0),
            parent: None,
            comment: Some("structural dynamic indexed destination-array HIR".to_owned()),
        }],
        body: hir::Block {
            scope: hir::ScopeId::new(0),
            statements: vec![
                hir::Stmt {
                    id: hir::StmtId::new(0),
                    kind: hir::StmtKind::Init {
                        place: hir::Place::Local(hir::LocalId::new(2)),
                        value: hir::Expr::Literal(hir::Literal::Integer(1)),
                    },
                },
                hir::Stmt {
                    id: hir::StmtId::new(1),
                    kind: hir::StmtKind::Init {
                        place: hir::Place::Index {
                            base: Box::new(hir::Place::Field {
                                base: Box::new(hir::Place::Local(hir::LocalId::new(1))),
                                field: "values".to_owned(),
                            }),
                            index: Box::new(hir::Expr::Literal(hir::Literal::Integer(0))),
                        },
                        value: hir::Expr::Literal(hir::Literal::Integer(5)),
                    },
                },
                hir::Stmt {
                    id: hir::StmtId::new(2),
                    kind: hir::StmtKind::Assign {
                        place: hir::Place::Index {
                            base: Box::new(hir::Place::Field {
                                base: Box::new(hir::Place::Local(hir::LocalId::new(1))),
                                field: "values".to_owned(),
                            }),
                            index: Box::new(hir::Expr::Local(hir::LocalId::new(2))),
                        },
                        value: hir::Expr::Literal(hir::Literal::Integer(7)),
                    },
                },
                hir::Stmt {
                    id: hir::StmtId::new(3),
                    kind: hir::StmtKind::Init {
                        place: hir::Place::Index {
                            base: Box::new(hir::Place::Field {
                                base: Box::new(hir::Place::Local(hir::LocalId::new(1))),
                                field: "values".to_owned(),
                            }),
                            index: Box::new(hir::Expr::Literal(hir::Literal::Integer(2))),
                        },
                        value: hir::Expr::Literal(hir::Literal::Integer(11)),
                    },
                },
                hir::Stmt {
                    id: hir::StmtId::new(4),
                    kind: hir::StmtKind::Init {
                        place: hir::Place::Index {
                            base: Box::new(hir::Place::Field {
                                base: Box::new(hir::Place::Local(hir::LocalId::new(1))),
                                field: "values".to_owned(),
                            }),
                            index: Box::new(hir::Expr::Literal(hir::Literal::Integer(3))),
                        },
                        value: hir::Expr::Literal(hir::Literal::Integer(13)),
                    },
                },
                hir::Stmt {
                    id: hir::StmtId::new(5),
                    kind: hir::StmtKind::Init {
                        place: hir::Place::Field {
                            base: Box::new(hir::Place::Local(hir::LocalId::new(1))),
                            field: "selected".to_owned(),
                        },
                        value: hir::Expr::Index {
                            base: Box::new(hir::Expr::Field {
                                base: Box::new(hir::Expr::Local(hir::LocalId::new(1))),
                                field: "values".to_owned(),
                            }),
                            index: Box::new(hir::Expr::Local(hir::LocalId::new(2))),
                        },
                    },
                },
                hir::Stmt {
                    id: hir::StmtId::new(6),
                    kind: hir::StmtKind::Return(None),
                },
            ],
        },
    });

    let decoder = compile_structural_hir_decoder(<DynamicDestinationSummary>::SHAPE, &module);
    let value = crate::deserialize::<DynamicDestinationSummary>(&decoder, &[])
        .expect("structural HIR decoder should support computed destination array indexing");
    assert_eq!(
        value,
        DynamicDestinationSummary {
            values: [5, 7, 11, 13],
            selected: 7,
        }
    );
}

#[test]
fn structural_hir_ir_path_reads_dynamic_local_aggregate_elements() {
    let mut module = hir::Module::new();
    let pair_def = module.add_type_def(hir::TypeDef {
        name: "Pair".to_owned(),
        generic_params: vec![],
        kind: hir::TypeDefKind::Struct {
            fields: vec![
                hir::FieldDef {
                    name: "lo".to_owned(),
                    ty: hir::Type::u(64),
                    offset: None,
                },
                hir::FieldDef {
                    name: "hi".to_owned(),
                    ty: hir::Type::u(64),
                    offset: None,
                },
            ],
        },
        size: None,
        transparent: false,
    });
    let root_def = module.add_type_def(hir::TypeDef {
        name: <DynamicAggregateSummary>::SHAPE.type_identifier.to_owned(),
        generic_params: vec![],
        kind: hir::TypeDefKind::Struct {
            fields: vec![hir::FieldDef {
                name: "pair".to_owned(),
                ty: hir::Type::named(pair_def, Vec::new()),
                offset: None,
            }],
        },
        size: None,
        transparent: false,
    });

    module.add_function(hir::Function {
        name: "dynamic_aggregate_summary".to_owned(),
        region_params: vec![],
        store_params: vec![],
        params: vec![
            hir::Parameter {
                local: hir::LocalId::new(0),
                name: "cursor".to_owned(),
                ty: hir::Type::u(64),
                kind: hir::LocalKind::Param,
            },
            hir::Parameter {
                local: hir::LocalId::new(1),
                name: "out".to_owned(),
                ty: hir::Type::named(root_def, Vec::new()),
                kind: hir::LocalKind::Destination,
            },
        ],
        locals: vec![
            hir::LocalDecl {
                local: hir::LocalId::new(2),
                name: "pairs".to_owned(),
                ty: hir::Type::array(hir::Type::named(pair_def, Vec::new()), 2),
                kind: hir::LocalKind::Let,
            },
            hir::LocalDecl {
                local: hir::LocalId::new(3),
                name: "idx".to_owned(),
                ty: hir::Type::u(32),
                kind: hir::LocalKind::Let,
            },
        ],
        return_type: hir::Type::unit(),
        scopes: vec![hir::Scope {
            id: hir::ScopeId::new(0),
            parent: None,
            comment: Some("structural dynamic indexed aggregate-array HIR".to_owned()),
        }],
        body: hir::Block {
            scope: hir::ScopeId::new(0),
            statements: vec![
                hir::Stmt {
                    id: hir::StmtId::new(0),
                    kind: hir::StmtKind::Init {
                        place: hir::Place::Field {
                            base: Box::new(hir::Place::Index {
                                base: Box::new(hir::Place::Local(hir::LocalId::new(2))),
                                index: Box::new(hir::Expr::Literal(hir::Literal::Integer(0))),
                            }),
                            field: "lo".to_owned(),
                        },
                        value: hir::Expr::Literal(hir::Literal::Integer(1)),
                    },
                },
                hir::Stmt {
                    id: hir::StmtId::new(1),
                    kind: hir::StmtKind::Init {
                        place: hir::Place::Field {
                            base: Box::new(hir::Place::Index {
                                base: Box::new(hir::Place::Local(hir::LocalId::new(2))),
                                index: Box::new(hir::Expr::Literal(hir::Literal::Integer(0))),
                            }),
                            field: "hi".to_owned(),
                        },
                        value: hir::Expr::Literal(hir::Literal::Integer(2)),
                    },
                },
                hir::Stmt {
                    id: hir::StmtId::new(2),
                    kind: hir::StmtKind::Init {
                        place: hir::Place::Field {
                            base: Box::new(hir::Place::Index {
                                base: Box::new(hir::Place::Local(hir::LocalId::new(2))),
                                index: Box::new(hir::Expr::Literal(hir::Literal::Integer(1))),
                            }),
                            field: "lo".to_owned(),
                        },
                        value: hir::Expr::Literal(hir::Literal::Integer(3)),
                    },
                },
                hir::Stmt {
                    id: hir::StmtId::new(3),
                    kind: hir::StmtKind::Init {
                        place: hir::Place::Field {
                            base: Box::new(hir::Place::Index {
                                base: Box::new(hir::Place::Local(hir::LocalId::new(2))),
                                index: Box::new(hir::Expr::Literal(hir::Literal::Integer(1))),
                            }),
                            field: "hi".to_owned(),
                        },
                        value: hir::Expr::Literal(hir::Literal::Integer(4)),
                    },
                },
                hir::Stmt {
                    id: hir::StmtId::new(4),
                    kind: hir::StmtKind::Init {
                        place: hir::Place::Local(hir::LocalId::new(3)),
                        value: hir::Expr::Literal(hir::Literal::Integer(1)),
                    },
                },
                hir::Stmt {
                    id: hir::StmtId::new(5),
                    kind: hir::StmtKind::Init {
                        place: hir::Place::Field {
                            base: Box::new(hir::Place::Local(hir::LocalId::new(1))),
                            field: "pair".to_owned(),
                        },
                        value: hir::Expr::Index {
                            base: Box::new(hir::Expr::Local(hir::LocalId::new(2))),
                            index: Box::new(hir::Expr::Local(hir::LocalId::new(3))),
                        },
                    },
                },
                hir::Stmt {
                    id: hir::StmtId::new(6),
                    kind: hir::StmtKind::Return(None),
                },
            ],
        },
    });

    let decoder = compile_structural_hir_decoder(<DynamicAggregateSummary>::SHAPE, &module);
    let value = crate::deserialize::<DynamicAggregateSummary>(&decoder, &[])
        .expect("structural HIR decoder should support computed aggregate local array indexing");
    assert_eq!(
        value,
        DynamicAggregateSummary {
            pair: Pair { lo: 3, hi: 4 },
        }
    );
}

#[test]
fn structural_hir_ir_path_writes_dynamic_local_aggregate_elements() {
    let mut module = hir::Module::new();
    let pair_def = module.add_type_def(hir::TypeDef {
        name: "Pair".to_owned(),
        generic_params: vec![],
        kind: hir::TypeDefKind::Struct {
            fields: vec![
                hir::FieldDef {
                    name: "lo".to_owned(),
                    ty: hir::Type::u(64),
                    offset: None,
                },
                hir::FieldDef {
                    name: "hi".to_owned(),
                    ty: hir::Type::u(64),
                    offset: None,
                },
            ],
        },
        size: None,
        transparent: false,
    });
    let root_def = module.add_type_def(hir::TypeDef {
        name: <DynamicAggregateSummary>::SHAPE.type_identifier.to_owned(),
        generic_params: vec![],
        kind: hir::TypeDefKind::Struct {
            fields: vec![hir::FieldDef {
                name: "pair".to_owned(),
                ty: hir::Type::named(pair_def, Vec::new()),
                offset: None,
            }],
        },
        size: None,
        transparent: false,
    });

    module.add_function(hir::Function {
        name: "dynamic_aggregate_write_summary".to_owned(),
        region_params: vec![],
        store_params: vec![],
        params: vec![
            hir::Parameter {
                local: hir::LocalId::new(0),
                name: "cursor".to_owned(),
                ty: hir::Type::u(64),
                kind: hir::LocalKind::Param,
            },
            hir::Parameter {
                local: hir::LocalId::new(1),
                name: "out".to_owned(),
                ty: hir::Type::named(root_def, Vec::new()),
                kind: hir::LocalKind::Destination,
            },
        ],
        locals: vec![
            hir::LocalDecl {
                local: hir::LocalId::new(2),
                name: "pairs".to_owned(),
                ty: hir::Type::array(hir::Type::named(pair_def, Vec::new()), 2),
                kind: hir::LocalKind::Let,
            },
            hir::LocalDecl {
                local: hir::LocalId::new(3),
                name: "pair".to_owned(),
                ty: hir::Type::named(pair_def, Vec::new()),
                kind: hir::LocalKind::Let,
            },
            hir::LocalDecl {
                local: hir::LocalId::new(4),
                name: "idx".to_owned(),
                ty: hir::Type::u(32),
                kind: hir::LocalKind::Let,
            },
        ],
        return_type: hir::Type::unit(),
        scopes: vec![hir::Scope {
            id: hir::ScopeId::new(0),
            parent: None,
            comment: Some("structural dynamic indexed aggregate-array write HIR".to_owned()),
        }],
        body: hir::Block {
            scope: hir::ScopeId::new(0),
            statements: vec![
                hir::Stmt {
                    id: hir::StmtId::new(0),
                    kind: hir::StmtKind::Init {
                        place: hir::Place::Field {
                            base: Box::new(hir::Place::Local(hir::LocalId::new(3))),
                            field: "lo".to_owned(),
                        },
                        value: hir::Expr::Literal(hir::Literal::Integer(9)),
                    },
                },
                hir::Stmt {
                    id: hir::StmtId::new(1),
                    kind: hir::StmtKind::Init {
                        place: hir::Place::Field {
                            base: Box::new(hir::Place::Local(hir::LocalId::new(3))),
                            field: "hi".to_owned(),
                        },
                        value: hir::Expr::Literal(hir::Literal::Integer(10)),
                    },
                },
                hir::Stmt {
                    id: hir::StmtId::new(2),
                    kind: hir::StmtKind::Init {
                        place: hir::Place::Local(hir::LocalId::new(4)),
                        value: hir::Expr::Literal(hir::Literal::Integer(1)),
                    },
                },
                hir::Stmt {
                    id: hir::StmtId::new(3),
                    kind: hir::StmtKind::Assign {
                        place: hir::Place::Index {
                            base: Box::new(hir::Place::Local(hir::LocalId::new(2))),
                            index: Box::new(hir::Expr::Local(hir::LocalId::new(4))),
                        },
                        value: hir::Expr::Local(hir::LocalId::new(3)),
                    },
                },
                hir::Stmt {
                    id: hir::StmtId::new(4),
                    kind: hir::StmtKind::Init {
                        place: hir::Place::Field {
                            base: Box::new(hir::Place::Local(hir::LocalId::new(1))),
                            field: "pair".to_owned(),
                        },
                        value: hir::Expr::Index {
                            base: Box::new(hir::Expr::Local(hir::LocalId::new(2))),
                            index: Box::new(hir::Expr::Local(hir::LocalId::new(4))),
                        },
                    },
                },
                hir::Stmt {
                    id: hir::StmtId::new(5),
                    kind: hir::StmtKind::Return(None),
                },
            ],
        },
    });

    let decoder = compile_structural_hir_decoder(<DynamicAggregateSummary>::SHAPE, &module);
    let value = crate::deserialize::<DynamicAggregateSummary>(&decoder, &[])
        .expect("structural HIR decoder should support computed aggregate local array writes");
    assert_eq!(
        value,
        DynamicAggregateSummary {
            pair: Pair { lo: 9, hi: 10 },
        }
    );
}

#[test]
fn structural_hir_ir_path_writes_dynamic_destination_aggregate_elements() {
    let mut module = hir::Module::new();
    let pair_def = module.add_type_def(hir::TypeDef {
        name: "Pair".to_owned(),
        generic_params: vec![],
        kind: hir::TypeDefKind::Struct {
            fields: vec![
                hir::FieldDef {
                    name: "lo".to_owned(),
                    ty: hir::Type::u(64),
                    offset: None,
                },
                hir::FieldDef {
                    name: "hi".to_owned(),
                    ty: hir::Type::u(64),
                    offset: None,
                },
            ],
        },
        size: None,
        transparent: false,
    });
    let root_def = module.add_type_def(hir::TypeDef {
        name: <DynamicAggregateDestinationSummary>::SHAPE
            .type_identifier
            .to_owned(),
        generic_params: vec![],
        kind: hir::TypeDefKind::Struct {
            fields: vec![
                hir::FieldDef {
                    name: "pairs".to_owned(),
                    ty: hir::Type::array(hir::Type::named(pair_def, Vec::new()), 2),
                    offset: None,
                },
                hir::FieldDef {
                    name: "selected".to_owned(),
                    ty: hir::Type::named(pair_def, Vec::new()),
                    offset: None,
                },
            ],
        },
        size: None,
        transparent: false,
    });

    module.add_function(hir::Function {
        name: "dynamic_aggregate_destination_summary".to_owned(),
        region_params: vec![],
        store_params: vec![],
        params: vec![
            hir::Parameter {
                local: hir::LocalId::new(0),
                name: "cursor".to_owned(),
                ty: hir::Type::u(64),
                kind: hir::LocalKind::Param,
            },
            hir::Parameter {
                local: hir::LocalId::new(1),
                name: "out".to_owned(),
                ty: hir::Type::named(root_def, Vec::new()),
                kind: hir::LocalKind::Destination,
            },
        ],
        locals: vec![
            hir::LocalDecl {
                local: hir::LocalId::new(2),
                name: "pair".to_owned(),
                ty: hir::Type::named(pair_def, Vec::new()),
                kind: hir::LocalKind::Let,
            },
            hir::LocalDecl {
                local: hir::LocalId::new(3),
                name: "idx".to_owned(),
                ty: hir::Type::u(32),
                kind: hir::LocalKind::Let,
            },
        ],
        return_type: hir::Type::unit(),
        scopes: vec![hir::Scope {
            id: hir::ScopeId::new(0),
            parent: None,
            comment: Some("structural dynamic indexed destination aggregate-array HIR".to_owned()),
        }],
        body: hir::Block {
            scope: hir::ScopeId::new(0),
            statements: vec![
                hir::Stmt {
                    id: hir::StmtId::new(0),
                    kind: hir::StmtKind::Init {
                        place: hir::Place::Field {
                            base: Box::new(hir::Place::Local(hir::LocalId::new(2))),
                            field: "lo".to_owned(),
                        },
                        value: hir::Expr::Literal(hir::Literal::Integer(21)),
                    },
                },
                hir::Stmt {
                    id: hir::StmtId::new(1),
                    kind: hir::StmtKind::Init {
                        place: hir::Place::Field {
                            base: Box::new(hir::Place::Local(hir::LocalId::new(2))),
                            field: "hi".to_owned(),
                        },
                        value: hir::Expr::Literal(hir::Literal::Integer(22)),
                    },
                },
                hir::Stmt {
                    id: hir::StmtId::new(2),
                    kind: hir::StmtKind::Init {
                        place: hir::Place::Local(hir::LocalId::new(3)),
                        value: hir::Expr::Literal(hir::Literal::Integer(1)),
                    },
                },
                hir::Stmt {
                    id: hir::StmtId::new(3),
                    kind: hir::StmtKind::Assign {
                        place: hir::Place::Index {
                            base: Box::new(hir::Place::Field {
                                base: Box::new(hir::Place::Local(hir::LocalId::new(1))),
                                field: "pairs".to_owned(),
                            }),
                            index: Box::new(hir::Expr::Local(hir::LocalId::new(3))),
                        },
                        value: hir::Expr::Local(hir::LocalId::new(2)),
                    },
                },
                hir::Stmt {
                    id: hir::StmtId::new(4),
                    kind: hir::StmtKind::Init {
                        place: hir::Place::Field {
                            base: Box::new(hir::Place::Index {
                                base: Box::new(hir::Place::Field {
                                    base: Box::new(hir::Place::Local(hir::LocalId::new(1))),
                                    field: "pairs".to_owned(),
                                }),
                                index: Box::new(hir::Expr::Literal(hir::Literal::Integer(0))),
                            }),
                            field: "lo".to_owned(),
                        },
                        value: hir::Expr::Literal(hir::Literal::Integer(1)),
                    },
                },
                hir::Stmt {
                    id: hir::StmtId::new(5),
                    kind: hir::StmtKind::Init {
                        place: hir::Place::Field {
                            base: Box::new(hir::Place::Index {
                                base: Box::new(hir::Place::Field {
                                    base: Box::new(hir::Place::Local(hir::LocalId::new(1))),
                                    field: "pairs".to_owned(),
                                }),
                                index: Box::new(hir::Expr::Literal(hir::Literal::Integer(0))),
                            }),
                            field: "hi".to_owned(),
                        },
                        value: hir::Expr::Literal(hir::Literal::Integer(2)),
                    },
                },
                hir::Stmt {
                    id: hir::StmtId::new(6),
                    kind: hir::StmtKind::Init {
                        place: hir::Place::Field {
                            base: Box::new(hir::Place::Local(hir::LocalId::new(1))),
                            field: "selected".to_owned(),
                        },
                        value: hir::Expr::Index {
                            base: Box::new(hir::Expr::Field {
                                base: Box::new(hir::Expr::Local(hir::LocalId::new(1))),
                                field: "pairs".to_owned(),
                            }),
                            index: Box::new(hir::Expr::Local(hir::LocalId::new(3))),
                        },
                    },
                },
                hir::Stmt {
                    id: hir::StmtId::new(7),
                    kind: hir::StmtKind::Return(None),
                },
            ],
        },
    });

    let decoder =
        compile_structural_hir_decoder(<DynamicAggregateDestinationSummary>::SHAPE, &module);
    let value = crate::deserialize::<DynamicAggregateDestinationSummary>(&decoder, &[])
        .expect("structural HIR decoder should support computed aggregate destination writes");
    assert_eq!(
        value,
        DynamicAggregateDestinationSummary {
            pairs: [Pair { lo: 1, hi: 2 }, Pair { lo: 21, hi: 22 }],
            selected: Pair { lo: 21, hi: 22 },
        }
    );
}
