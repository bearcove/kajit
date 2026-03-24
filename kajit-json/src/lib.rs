//! JSON HIR generation — builds `hir::Module` from facet `Shape` for JSON format.

use facet::{ScalarType, Shape};
use kajit_hir as hir;

pub fn build_json_decoder_hir(shape: &'static Shape) -> hir::Module {
    match shape.scalar_type() {
        Some(ScalarType::Bool) => build_json_root_bool_decoder_hir(shape),
        Some(ScalarType::U32) => build_json_root_u32_decoder_hir(shape),
        Some(ScalarType::U64) => build_json_root_u64_decoder_hir(shape),
        other => panic!("unsupported JSON HIR prototype shape: {other:?}"),
    }
}

pub fn supports_json_decoder_hir(shape: &'static Shape) -> bool {
    matches!(
        shape.scalar_type(),
        Some(ScalarType::Bool | ScalarType::U32 | ScalarType::U64)
    )
}
fn build_json_root_bool_decoder_hir(shape: &'static Shape) -> hir::Module {
    let mut module = hir::Module::new();
    let input_region = module.add_region("input");
    let cursor_type = kajit_format::hir_helpers::add_cursor_type(&mut module, input_region);

    let cursor_local = hir::LocalId::new(0);
    let out_local = hir::LocalId::new(1);
    let byte_local = hir::LocalId::new(2);
    let root_scope = hir::ScopeId::new(0);

    let cursor_bytes = || hir::Expr::Field {
        base: Box::new(hir::Expr::Local(cursor_local)),
        field: "bytes".to_owned(),
    };
    let cursor_pos = || hir::Expr::Field {
        base: Box::new(hir::Expr::Local(cursor_local)),
        field: "pos".to_owned(),
    };
    let byte_at_cursor = || hir::Expr::Load {
        addr: Box::new(hir::Expr::Binary {
            op: hir::BinaryOp::Add,
            lhs: Box::new(hir::Expr::SliceData {
                value: Box::new(cursor_bytes()),
            }),
            rhs: Box::new(cursor_pos()),
        }),
        width: hir::MemoryWidth::W1,
    };
    let advance_cursor_stmt = |stmt_id: u32, delta: u64| -> hir::Stmt {
        hir::Stmt {
            id: hir::StmtId::new(stmt_id),
            kind: hir::StmtKind::Assign {
                place: hir::Place::Field {
                    base: Box::new(hir::Place::Local(cursor_local)),
                    field: "pos".to_owned(),
                },
                value: hir::Expr::Binary {
                    op: hir::BinaryOp::Add,
                    lhs: Box::new(cursor_pos()),
                    rhs: Box::new(hir::Expr::Literal(hir::Literal::Integer(delta))),
                },
            },
        }
    };
    let cursor_bounds_if =
        |stmt_id: u32, need: u64, fail_stmt: u32, error: hir::ErrorCode| -> hir::Stmt {
            hir::Stmt {
                id: hir::StmtId::new(stmt_id),
                kind: hir::StmtKind::If {
                    condition: hir::Expr::Binary {
                        op: hir::BinaryOp::Gt,
                        lhs: Box::new(hir::Expr::Binary {
                            op: hir::BinaryOp::Add,
                            lhs: Box::new(cursor_pos()),
                            rhs: Box::new(hir::Expr::Literal(hir::Literal::Integer(need))),
                        }),
                        rhs: Box::new(hir::Expr::SliceLen {
                            value: Box::new(cursor_bytes()),
                        }),
                    },
                    then_block: hir::Block {
                        scope: root_scope,
                        statements: vec![hir::Stmt {
                            id: hir::StmtId::new(fail_stmt),
                            kind: hir::StmtKind::Fail { code: error },
                        }],
                    },
                    else_block: Some(hir::Block {
                        scope: root_scope,
                        statements: Vec::new(),
                    }),
                },
            }
        };
    let matches_ascii = |text: &[u8], start_stmt: &mut u32| -> Vec<hir::Stmt> {
        let mut statements = Vec::new();
        for (index, expected) in text.iter().copied().enumerate() {
            let mismatch_stmt = *start_stmt;
            *start_stmt += 2;
            statements.push(hir::Stmt {
                id: hir::StmtId::new(mismatch_stmt),
                kind: hir::StmtKind::If {
                    condition: hir::Expr::Binary {
                        op: hir::BinaryOp::Ne,
                        lhs: Box::new(hir::Expr::Load {
                            addr: Box::new(hir::Expr::Binary {
                                op: hir::BinaryOp::Add,
                                lhs: Box::new(hir::Expr::SliceData {
                                    value: Box::new(cursor_bytes()),
                                }),
                                rhs: Box::new(hir::Expr::Binary {
                                    op: hir::BinaryOp::Add,
                                    lhs: Box::new(cursor_pos()),
                                    rhs: Box::new(hir::Expr::Literal(hir::Literal::Integer(
                                        index as u64,
                                    ))),
                                }),
                            }),
                            width: hir::MemoryWidth::W1,
                        }),
                        rhs: Box::new(hir::Expr::Literal(hir::Literal::Integer(expected as u64))),
                    },
                    then_block: hir::Block {
                        scope: root_scope,
                        statements: vec![hir::Stmt {
                            id: hir::StmtId::new(mismatch_stmt + 1),
                            kind: hir::StmtKind::Fail {
                                code: hir::ErrorCode::InvalidBool,
                            },
                        }],
                    },
                    else_block: Some(hir::Block {
                        scope: root_scope,
                        statements: Vec::new(),
                    }),
                },
            });
        }
        statements
    };

    module.add_function(hir::Function {
        name: format!("decode_{}", shape.type_identifier.replace("::", "_")),
        region_params: vec![input_region],
        store_params: Vec::new(),
        params: vec![
            hir::Parameter {
                local: cursor_local,
                name: "cursor".to_owned(),
                ty: hir::Type::named(cursor_type, vec![hir::GenericArg::Region(input_region)]),
                kind: hir::LocalKind::Param,
            },
            hir::Parameter {
                local: out_local,
                name: "out".to_owned(),
                ty: hir::Type::bool(),
                kind: hir::LocalKind::Destination,
            },
        ],
        locals: vec![hir::LocalDecl {
            local: byte_local,
            name: "byte".to_owned(),
            ty: hir::Type::u(8),
            kind: hir::LocalKind::Temp,
        }],
        return_type: hir::Type::unit(),
        scopes: vec![hir::Scope {
            id: root_scope,
            parent: None,
            comment: Some(format!("Prototype JSON HIR for {}", shape.type_identifier)),
        }],
        body: hir::Block {
            scope: root_scope,
            statements: vec![
                hir::Stmt {
                    id: hir::StmtId::new(0),
                    kind: hir::StmtKind::Loop {
                        body: hir::Block {
                            scope: root_scope,
                            statements: vec![
                                cursor_bounds_if(1, 1, 2, hir::ErrorCode::UnexpectedEof),
                                hir::Stmt {
                                    id: hir::StmtId::new(3),
                                    kind: hir::StmtKind::Assign {
                                        place: hir::Place::Local(byte_local),
                                        value: byte_at_cursor(),
                                    },
                                },
                                hir::Stmt {
                                    id: hir::StmtId::new(4),
                                    kind: hir::StmtKind::If {
                                        condition: hir::Expr::Binary {
                                            op: hir::BinaryOp::Or,
                                            lhs: Box::new(hir::Expr::Binary {
                                                op: hir::BinaryOp::Eq,
                                                lhs: Box::new(hir::Expr::Local(byte_local)),
                                                rhs: Box::new(hir::Expr::Literal(
                                                    hir::Literal::Integer(b' ' as u64),
                                                )),
                                            }),
                                            rhs: Box::new(hir::Expr::Binary {
                                                op: hir::BinaryOp::Or,
                                                lhs: Box::new(hir::Expr::Binary {
                                                    op: hir::BinaryOp::Eq,
                                                    lhs: Box::new(hir::Expr::Local(byte_local)),
                                                    rhs: Box::new(hir::Expr::Literal(
                                                        hir::Literal::Integer(b'\n' as u64),
                                                    )),
                                                }),
                                                rhs: Box::new(hir::Expr::Binary {
                                                    op: hir::BinaryOp::Or,
                                                    lhs: Box::new(hir::Expr::Binary {
                                                        op: hir::BinaryOp::Eq,
                                                        lhs: Box::new(hir::Expr::Local(byte_local)),
                                                        rhs: Box::new(hir::Expr::Literal(
                                                            hir::Literal::Integer(b'\r' as u64),
                                                        )),
                                                    }),
                                                    rhs: Box::new(hir::Expr::Binary {
                                                        op: hir::BinaryOp::Eq,
                                                        lhs: Box::new(hir::Expr::Local(byte_local)),
                                                        rhs: Box::new(hir::Expr::Literal(
                                                            hir::Literal::Integer(b'\t' as u64),
                                                        )),
                                                    }),
                                                }),
                                            }),
                                        },
                                        then_block: hir::Block {
                                            scope: root_scope,
                                            statements: vec![
                                                advance_cursor_stmt(5, 1),
                                                hir::Stmt {
                                                    id: hir::StmtId::new(6),
                                                    kind: hir::StmtKind::Continue,
                                                },
                                            ],
                                        },
                                        else_block: Some(hir::Block {
                                            scope: root_scope,
                                            statements: vec![hir::Stmt {
                                                id: hir::StmtId::new(7),
                                                kind: hir::StmtKind::Break,
                                            }],
                                        }),
                                    },
                                },
                            ],
                        },
                        max_iterations: None,
                    },
                },
                hir::Stmt {
                    id: hir::StmtId::new(8),
                    kind: hir::StmtKind::If {
                        condition: hir::Expr::Binary {
                            op: hir::BinaryOp::Eq,
                            lhs: Box::new(byte_at_cursor()),
                            rhs: Box::new(hir::Expr::Literal(hir::Literal::Integer(b't' as u64))),
                        },
                        then_block: {
                            let mut statements =
                                vec![cursor_bounds_if(9, 4, 10, hir::ErrorCode::UnexpectedEof)];
                            let mut next_stmt = 11;
                            statements.extend(matches_ascii(b"true", &mut next_stmt));
                            statements.push(advance_cursor_stmt(next_stmt, 4));
                            statements.push(hir::Stmt {
                                id: hir::StmtId::new(next_stmt + 1),
                                kind: hir::StmtKind::Init {
                                    place: hir::Place::Local(out_local),
                                    value: hir::Expr::Literal(hir::Literal::Bool(true)),
                                },
                            });
                            statements.push(hir::Stmt {
                                id: hir::StmtId::new(next_stmt + 2),
                                kind: hir::StmtKind::Return(None),
                            });
                            hir::Block {
                                scope: root_scope,
                                statements,
                            }
                        },
                        else_block: Some(hir::Block {
                            scope: root_scope,
                            statements: vec![hir::Stmt {
                                id: hir::StmtId::new(17),
                                kind: hir::StmtKind::If {
                                    condition: hir::Expr::Binary {
                                        op: hir::BinaryOp::Eq,
                                        lhs: Box::new(byte_at_cursor()),
                                        rhs: Box::new(hir::Expr::Literal(hir::Literal::Integer(
                                            b'f' as u64,
                                        ))),
                                    },
                                    then_block: {
                                        let mut statements = vec![cursor_bounds_if(
                                            18,
                                            5,
                                            19,
                                            hir::ErrorCode::UnexpectedEof,
                                        )];
                                        let mut next_stmt = 20;
                                        statements.extend(matches_ascii(b"false", &mut next_stmt));
                                        statements.push(advance_cursor_stmt(next_stmt, 5));
                                        statements.push(hir::Stmt {
                                            id: hir::StmtId::new(next_stmt + 1),
                                            kind: hir::StmtKind::Init {
                                                place: hir::Place::Local(out_local),
                                                value: hir::Expr::Literal(hir::Literal::Bool(
                                                    false,
                                                )),
                                            },
                                        });
                                        statements.push(hir::Stmt {
                                            id: hir::StmtId::new(next_stmt + 2),
                                            kind: hir::StmtKind::Return(None),
                                        });
                                        hir::Block {
                                            scope: root_scope,
                                            statements,
                                        }
                                    },
                                    else_block: Some(hir::Block {
                                        scope: root_scope,
                                        statements: vec![hir::Stmt {
                                            id: hir::StmtId::new(28),
                                            kind: hir::StmtKind::Fail {
                                                code: hir::ErrorCode::InvalidBool,
                                            },
                                        }],
                                    }),
                                },
                            }],
                        }),
                    },
                },
            ],
        },
    });

    module
}

fn build_json_root_u32_decoder_hir(shape: &'static Shape) -> hir::Module {
    let mut module = hir::Module::new();
    let input_region = module.add_region("input");
    let cursor_type = kajit_format::hir_helpers::add_cursor_type(&mut module, input_region);

    let cursor_local = hir::LocalId::new(0);
    let out_local = hir::LocalId::new(1);
    let byte_local = hir::LocalId::new(2);
    let acc_local = hir::LocalId::new(3);
    let digit_count_local = hir::LocalId::new(4);
    let root_scope = hir::ScopeId::new(0);

    let cursor_bytes = || hir::Expr::Field {
        base: Box::new(hir::Expr::Local(cursor_local)),
        field: "bytes".to_owned(),
    };
    let cursor_pos = || hir::Expr::Field {
        base: Box::new(hir::Expr::Local(cursor_local)),
        field: "pos".to_owned(),
    };
    let byte_at_cursor = || hir::Expr::Load {
        addr: Box::new(hir::Expr::Binary {
            op: hir::BinaryOp::Add,
            lhs: Box::new(hir::Expr::SliceData {
                value: Box::new(cursor_bytes()),
            }),
            rhs: Box::new(cursor_pos()),
        }),
        width: hir::MemoryWidth::W1,
    };
    let advance_cursor_stmt = |stmt_id: u32, delta: u64| -> hir::Stmt {
        hir::Stmt {
            id: hir::StmtId::new(stmt_id),
            kind: hir::StmtKind::Assign {
                place: hir::Place::Field {
                    base: Box::new(hir::Place::Local(cursor_local)),
                    field: "pos".to_owned(),
                },
                value: hir::Expr::Binary {
                    op: hir::BinaryOp::Add,
                    lhs: Box::new(cursor_pos()),
                    rhs: Box::new(hir::Expr::Literal(hir::Literal::Integer(delta))),
                },
            },
        }
    };
    let cursor_bounds_if =
        |stmt_id: u32, need: u64, fail_stmt: u32, error: hir::ErrorCode| -> hir::Stmt {
            hir::Stmt {
                id: hir::StmtId::new(stmt_id),
                kind: hir::StmtKind::If {
                    condition: hir::Expr::Binary {
                        op: hir::BinaryOp::Gt,
                        lhs: Box::new(hir::Expr::Binary {
                            op: hir::BinaryOp::Add,
                            lhs: Box::new(cursor_pos()),
                            rhs: Box::new(hir::Expr::Literal(hir::Literal::Integer(need))),
                        }),
                        rhs: Box::new(hir::Expr::SliceLen {
                            value: Box::new(cursor_bytes()),
                        }),
                    },
                    then_block: hir::Block {
                        scope: root_scope,
                        statements: vec![hir::Stmt {
                            id: hir::StmtId::new(fail_stmt),
                            kind: hir::StmtKind::Fail { code: error },
                        }],
                    },
                    else_block: Some(hir::Block {
                        scope: root_scope,
                        statements: Vec::new(),
                    }),
                },
            }
        };

    module.add_function(hir::Function {
        name: format!("decode_{}", shape.type_identifier.replace("::", "_")),
        region_params: vec![input_region],
        store_params: Vec::new(),
        params: vec![
            hir::Parameter {
                local: cursor_local,
                name: "cursor".to_owned(),
                ty: hir::Type::named(cursor_type, vec![hir::GenericArg::Region(input_region)]),
                kind: hir::LocalKind::Param,
            },
            hir::Parameter {
                local: out_local,
                name: "out".to_owned(),
                ty: hir::Type::u(32),
                kind: hir::LocalKind::Destination,
            },
        ],
        locals: vec![
            hir::LocalDecl {
                local: byte_local,
                name: "byte".to_owned(),
                ty: hir::Type::u(8),
                kind: hir::LocalKind::Temp,
            },
            hir::LocalDecl {
                local: acc_local,
                name: "acc".to_owned(),
                ty: hir::Type::u(64),
                kind: hir::LocalKind::Temp,
            },
            hir::LocalDecl {
                local: digit_count_local,
                name: "digit_count".to_owned(),
                ty: hir::Type::u(64),
                kind: hir::LocalKind::Temp,
            },
        ],
        return_type: hir::Type::unit(),
        scopes: vec![hir::Scope {
            id: root_scope,
            parent: None,
            comment: Some(format!(
                "Prototype JSON HIR for {}",
                shape.type_identifier
            )),
        }],
        body: hir::Block {
            scope: root_scope,
            statements: vec![
                hir::Stmt {
                    id: hir::StmtId::new(0),
                    kind: hir::StmtKind::Loop {
                        body: hir::Block {
                            scope: root_scope,
                            statements: vec![
                                cursor_bounds_if(1, 1, 2, hir::ErrorCode::UnexpectedEof),
                                hir::Stmt {
                                    id: hir::StmtId::new(3),
                                    kind: hir::StmtKind::Assign {
                                        place: hir::Place::Local(byte_local),
                                        value: byte_at_cursor(),
                                    },
                                },
                                hir::Stmt {
                                    id: hir::StmtId::new(4),
                                    kind: hir::StmtKind::If {
                                        condition: hir::Expr::Binary {
                                            op: hir::BinaryOp::Or,
                                            lhs: Box::new(hir::Expr::Binary {
                                                op: hir::BinaryOp::Eq,
                                                lhs: Box::new(hir::Expr::Local(byte_local)),
                                                rhs: Box::new(hir::Expr::Literal(
                                                    hir::Literal::Integer(b' ' as u64),
                                                )),
                                            }),
                                            rhs: Box::new(hir::Expr::Binary {
                                                op: hir::BinaryOp::Or,
                                                lhs: Box::new(hir::Expr::Binary {
                                                    op: hir::BinaryOp::Eq,
                                                    lhs: Box::new(hir::Expr::Local(byte_local)),
                                                    rhs: Box::new(hir::Expr::Literal(
                                                        hir::Literal::Integer(b'\n' as u64),
                                                    )),
                                                }),
                                                rhs: Box::new(hir::Expr::Binary {
                                                    op: hir::BinaryOp::Or,
                                                    lhs: Box::new(hir::Expr::Binary {
                                                        op: hir::BinaryOp::Eq,
                                                        lhs: Box::new(hir::Expr::Local(byte_local)),
                                                        rhs: Box::new(hir::Expr::Literal(
                                                            hir::Literal::Integer(b'\r' as u64),
                                                        )),
                                                    }),
                                                    rhs: Box::new(hir::Expr::Binary {
                                                        op: hir::BinaryOp::Eq,
                                                        lhs: Box::new(hir::Expr::Local(byte_local)),
                                                        rhs: Box::new(hir::Expr::Literal(
                                                            hir::Literal::Integer(b'\t' as u64),
                                                        )),
                                                    }),
                                                }),
                                            }),
                                        },
                                        then_block: hir::Block {
                                            scope: root_scope,
                                            statements: vec![
                                                advance_cursor_stmt(5, 1),
                                                hir::Stmt {
                                                    id: hir::StmtId::new(6),
                                                    kind: hir::StmtKind::Continue,
                                                },
                                            ],
                                        },
                                        else_block: Some(hir::Block {
                                            scope: root_scope,
                                            statements: vec![hir::Stmt {
                                                id: hir::StmtId::new(7),
                                                kind: hir::StmtKind::Break,
                                            }],
                                        }),
                                    },
                                },
                            ],
                        },
                        max_iterations: None,
                    },
                },
                hir::Stmt {
                    id: hir::StmtId::new(8),
                    kind: hir::StmtKind::Init {
                        place: hir::Place::Local(acc_local),
                        value: hir::Expr::Literal(hir::Literal::Integer(0)),
                    },
                },
                hir::Stmt {
                    id: hir::StmtId::new(9),
                    kind: hir::StmtKind::Init {
                        place: hir::Place::Local(digit_count_local),
                        value: hir::Expr::Literal(hir::Literal::Integer(0)),
                    },
                },
                hir::Stmt {
                    id: hir::StmtId::new(10),
                    kind: hir::StmtKind::Loop {
                        body: hir::Block {
                            scope: root_scope,
                            statements: vec![
                                hir::Stmt {
                                    id: hir::StmtId::new(11),
                                    kind: hir::StmtKind::If {
                                        condition: hir::Expr::Binary {
                                            op: hir::BinaryOp::Ge,
                                            lhs: Box::new(cursor_pos()),
                                            rhs: Box::new(hir::Expr::SliceLen {
                                                value: Box::new(cursor_bytes()),
                                            }),
                                        },
                                        then_block: hir::Block {
                                            scope: root_scope,
                                            statements: vec![hir::Stmt {
                                                id: hir::StmtId::new(12),
                                                kind: hir::StmtKind::Break,
                                            }],
                                        },
                                        else_block: Some(hir::Block {
                                            scope: root_scope,
                                            statements: Vec::new(),
                                        }),
                                    },
                                },
                                hir::Stmt {
                                    id: hir::StmtId::new(13),
                                    kind: hir::StmtKind::Assign {
                                        place: hir::Place::Local(byte_local),
                                        value: byte_at_cursor(),
                                    },
                                },
                                hir::Stmt {
                                    id: hir::StmtId::new(14),
                                    kind: hir::StmtKind::If {
                                        condition: hir::Expr::Binary {
                                            op: hir::BinaryOp::Or,
                                            lhs: Box::new(hir::Expr::Binary {
                                                op: hir::BinaryOp::Lt,
                                                lhs: Box::new(hir::Expr::Local(byte_local)),
                                                rhs: Box::new(hir::Expr::Literal(
                                                    hir::Literal::Integer(b'0' as u64),
                                                )),
                                            }),
                                            rhs: Box::new(hir::Expr::Binary {
                                                op: hir::BinaryOp::Gt,
                                                lhs: Box::new(hir::Expr::Local(byte_local)),
                                                rhs: Box::new(hir::Expr::Literal(
                                                    hir::Literal::Integer(b'9' as u64),
                                                )),
                                            }),
                                        },
                                        then_block: hir::Block {
                                            scope: root_scope,
                                            statements: vec![hir::Stmt {
                                                id: hir::StmtId::new(15),
                                                kind: hir::StmtKind::Break,
                                            }],
                                        },
                                        else_block: Some(hir::Block {
                                            scope: root_scope,
                                            statements: vec![
                                                hir::Stmt {
                                                    id: hir::StmtId::new(16),
                                                    kind: hir::StmtKind::Assign {
                                                        place: hir::Place::Local(acc_local),
                                                        value: hir::Expr::Binary {
                                                            op: hir::BinaryOp::Add,
                                                            lhs: Box::new(hir::Expr::Binary {
                                                                op: hir::BinaryOp::Mul,
                                                                lhs: Box::new(hir::Expr::Local(
                                                                    acc_local,
                                                                )),
                                                                rhs: Box::new(hir::Expr::Literal(
                                                                    hir::Literal::Integer(10),
                                                                )),
                                                            }),
                                                            rhs: Box::new(hir::Expr::Binary {
                                                                op: hir::BinaryOp::Sub,
                                                                lhs: Box::new(hir::Expr::Local(
                                                                    byte_local,
                                                                )),
                                                                rhs: Box::new(hir::Expr::Literal(
                                                                    hir::Literal::Integer(
                                                                        b'0' as u64,
                                                                    ),
                                                                )),
                                                            }),
                                                        },
                                                    },
                                                },
                                                hir::Stmt {
                                                    id: hir::StmtId::new(17),
                                                    kind: hir::StmtKind::If {
                                                        condition: hir::Expr::Binary {
                                                            op: hir::BinaryOp::Gt,
                                                            lhs: Box::new(hir::Expr::Local(
                                                                acc_local,
                                                            )),
                                                            rhs: Box::new(hir::Expr::Literal(
                                                                hir::Literal::Integer(
                                                                    u32::MAX as u64,
                                                                ),
                                                            )),
                                                        },
                                                        then_block: hir::Block {
                                                            scope: root_scope,
                                                            statements: vec![hir::Stmt {
                                                                id: hir::StmtId::new(18),
                                                                kind: hir::StmtKind::Fail {
                                                                    code: hir::ErrorCode::NumberOutOfRange,
                                                                },
                                                            }],
                                                        },
                                                        else_block: Some(hir::Block {
                                                            scope: root_scope,
                                                            statements: Vec::new(),
                                                        }),
                                                    },
                                                },
                                                hir::Stmt {
                                                    id: hir::StmtId::new(19),
                                                    kind: hir::StmtKind::Assign {
                                                        place: hir::Place::Local(digit_count_local),
                                                        value: hir::Expr::Binary {
                                                            op: hir::BinaryOp::Add,
                                                            lhs: Box::new(hir::Expr::Local(
                                                                digit_count_local,
                                                            )),
                                                            rhs: Box::new(hir::Expr::Literal(
                                                                hir::Literal::Integer(1),
                                                            )),
                                                        },
                                                    },
                                                },
                                                advance_cursor_stmt(20, 1),
                                                hir::Stmt {
                                                    id: hir::StmtId::new(21),
                                                    kind: hir::StmtKind::Continue,
                                                },
                                            ],
                                        }),
                                    },
                                },
                            ],
                        },
                        max_iterations: None,
                    },
                },
                hir::Stmt {
                    id: hir::StmtId::new(22),
                    kind: hir::StmtKind::If {
                        condition: hir::Expr::Binary {
                            op: hir::BinaryOp::Eq,
                            lhs: Box::new(hir::Expr::Local(digit_count_local)),
                            rhs: Box::new(hir::Expr::Literal(hir::Literal::Integer(0))),
                        },
                        then_block: hir::Block {
                            scope: root_scope,
                            statements: vec![hir::Stmt {
                                id: hir::StmtId::new(23),
                                kind: hir::StmtKind::Fail {
                                    code: hir::ErrorCode::InvalidJsonNumber,
                                },
                            }],
                        },
                        else_block: Some(hir::Block {
                            scope: root_scope,
                            statements: vec![],
                        }),
                    },
                },
                hir::Stmt {
                    id: hir::StmtId::new(24),
                    kind: hir::StmtKind::Init {
                        place: hir::Place::Local(out_local),
                        value: hir::Expr::Local(acc_local),
                    },
                },
                hir::Stmt {
                    id: hir::StmtId::new(25),
                    kind: hir::StmtKind::Return(None),
                },
            ],
        },
    });

    module
}

fn build_json_root_u64_decoder_hir(shape: &'static Shape) -> hir::Module {
    let mut module = hir::Module::new();
    let input_region = module.add_region("input");
    let cursor_type = kajit_format::hir_helpers::add_cursor_type(&mut module, input_region);

    let cursor_local = hir::LocalId::new(0);
    let out_local = hir::LocalId::new(1);
    let byte_local = hir::LocalId::new(2);
    let acc_local = hir::LocalId::new(3);
    let digit_count_local = hir::LocalId::new(4);
    let root_scope = hir::ScopeId::new(0);

    let cursor_bytes = || hir::Expr::Field {
        base: Box::new(hir::Expr::Local(cursor_local)),
        field: "bytes".to_owned(),
    };
    let cursor_pos = || hir::Expr::Field {
        base: Box::new(hir::Expr::Local(cursor_local)),
        field: "pos".to_owned(),
    };
    let byte_at_cursor = || hir::Expr::Load {
        addr: Box::new(hir::Expr::Binary {
            op: hir::BinaryOp::Add,
            lhs: Box::new(hir::Expr::SliceData {
                value: Box::new(cursor_bytes()),
            }),
            rhs: Box::new(cursor_pos()),
        }),
        width: hir::MemoryWidth::W1,
    };
    let advance_cursor_stmt = |stmt_id: u32, delta: u64| -> hir::Stmt {
        hir::Stmt {
            id: hir::StmtId::new(stmt_id),
            kind: hir::StmtKind::Assign {
                place: hir::Place::Field {
                    base: Box::new(hir::Place::Local(cursor_local)),
                    field: "pos".to_owned(),
                },
                value: hir::Expr::Binary {
                    op: hir::BinaryOp::Add,
                    lhs: Box::new(cursor_pos()),
                    rhs: Box::new(hir::Expr::Literal(hir::Literal::Integer(delta))),
                },
            },
        }
    };
    let cursor_bounds_if =
        |stmt_id: u32, need: u64, fail_stmt: u32, error: hir::ErrorCode| -> hir::Stmt {
            hir::Stmt {
                id: hir::StmtId::new(stmt_id),
                kind: hir::StmtKind::If {
                    condition: hir::Expr::Binary {
                        op: hir::BinaryOp::Gt,
                        lhs: Box::new(hir::Expr::Binary {
                            op: hir::BinaryOp::Add,
                            lhs: Box::new(cursor_pos()),
                            rhs: Box::new(hir::Expr::Literal(hir::Literal::Integer(need))),
                        }),
                        rhs: Box::new(hir::Expr::SliceLen {
                            value: Box::new(cursor_bytes()),
                        }),
                    },
                    then_block: hir::Block {
                        scope: root_scope,
                        statements: vec![hir::Stmt {
                            id: hir::StmtId::new(fail_stmt),
                            kind: hir::StmtKind::Fail { code: error },
                        }],
                    },
                    else_block: Some(hir::Block {
                        scope: root_scope,
                        statements: Vec::new(),
                    }),
                },
            }
        };

    module.add_function(hir::Function {
        name: format!("decode_{}", shape.type_identifier.replace("::", "_")),
        region_params: vec![input_region],
        store_params: Vec::new(),
        params: vec![
            hir::Parameter {
                local: cursor_local,
                name: "cursor".to_owned(),
                ty: hir::Type::named(cursor_type, vec![hir::GenericArg::Region(input_region)]),
                kind: hir::LocalKind::Param,
            },
            hir::Parameter {
                local: out_local,
                name: "out".to_owned(),
                ty: hir::Type::u(64),
                kind: hir::LocalKind::Destination,
            },
        ],
        locals: vec![
            hir::LocalDecl {
                local: byte_local,
                name: "byte".to_owned(),
                ty: hir::Type::u(8),
                kind: hir::LocalKind::Temp,
            },
            hir::LocalDecl {
                local: acc_local,
                name: "acc".to_owned(),
                ty: hir::Type::u(64),
                kind: hir::LocalKind::Temp,
            },
            hir::LocalDecl {
                local: digit_count_local,
                name: "digit_count".to_owned(),
                ty: hir::Type::u(64),
                kind: hir::LocalKind::Temp,
            },
        ],
        return_type: hir::Type::unit(),
        scopes: vec![hir::Scope {
            id: root_scope,
            parent: None,
            comment: Some(format!(
                "Prototype JSON HIR for {}",
                shape.type_identifier
            )),
        }],
        body: hir::Block {
            scope: root_scope,
            statements: vec![
                hir::Stmt {
                    id: hir::StmtId::new(0),
                    kind: hir::StmtKind::Loop {
                        body: hir::Block {
                            scope: root_scope,
                            statements: vec![
                                cursor_bounds_if(1, 1, 2, hir::ErrorCode::UnexpectedEof),
                                hir::Stmt {
                                    id: hir::StmtId::new(3),
                                    kind: hir::StmtKind::Assign {
                                        place: hir::Place::Local(byte_local),
                                        value: byte_at_cursor(),
                                    },
                                },
                                hir::Stmt {
                                    id: hir::StmtId::new(4),
                                    kind: hir::StmtKind::If {
                                        condition: hir::Expr::Binary {
                                            op: hir::BinaryOp::Or,
                                            lhs: Box::new(hir::Expr::Binary {
                                                op: hir::BinaryOp::Eq,
                                                lhs: Box::new(hir::Expr::Local(byte_local)),
                                                rhs: Box::new(hir::Expr::Literal(
                                                    hir::Literal::Integer(b' ' as u64),
                                                )),
                                            }),
                                            rhs: Box::new(hir::Expr::Binary {
                                                op: hir::BinaryOp::Or,
                                                lhs: Box::new(hir::Expr::Binary {
                                                    op: hir::BinaryOp::Eq,
                                                    lhs: Box::new(hir::Expr::Local(byte_local)),
                                                    rhs: Box::new(hir::Expr::Literal(
                                                        hir::Literal::Integer(b'\n' as u64),
                                                    )),
                                                }),
                                                rhs: Box::new(hir::Expr::Binary {
                                                    op: hir::BinaryOp::Or,
                                                    lhs: Box::new(hir::Expr::Binary {
                                                        op: hir::BinaryOp::Eq,
                                                        lhs: Box::new(hir::Expr::Local(byte_local)),
                                                        rhs: Box::new(hir::Expr::Literal(
                                                            hir::Literal::Integer(b'\r' as u64),
                                                        )),
                                                    }),
                                                    rhs: Box::new(hir::Expr::Binary {
                                                        op: hir::BinaryOp::Eq,
                                                        lhs: Box::new(hir::Expr::Local(byte_local)),
                                                        rhs: Box::new(hir::Expr::Literal(
                                                            hir::Literal::Integer(b'\t' as u64),
                                                        )),
                                                    }),
                                                }),
                                            }),
                                        },
                                        then_block: hir::Block {
                                            scope: root_scope,
                                            statements: vec![
                                                advance_cursor_stmt(5, 1),
                                                hir::Stmt {
                                                    id: hir::StmtId::new(6),
                                                    kind: hir::StmtKind::Continue,
                                                },
                                            ],
                                        },
                                        else_block: Some(hir::Block {
                                            scope: root_scope,
                                            statements: vec![hir::Stmt {
                                                id: hir::StmtId::new(7),
                                                kind: hir::StmtKind::Break,
                                            }],
                                        }),
                                    },
                                },
                            ],
                        },
                        max_iterations: None,
                    },
                },
                hir::Stmt {
                    id: hir::StmtId::new(8),
                    kind: hir::StmtKind::Init {
                        place: hir::Place::Local(acc_local),
                        value: hir::Expr::Literal(hir::Literal::Integer(0)),
                    },
                },
                hir::Stmt {
                    id: hir::StmtId::new(9),
                    kind: hir::StmtKind::Init {
                        place: hir::Place::Local(digit_count_local),
                        value: hir::Expr::Literal(hir::Literal::Integer(0)),
                    },
                },
                hir::Stmt {
                    id: hir::StmtId::new(10),
                    kind: hir::StmtKind::Loop {
                        body: hir::Block {
                            scope: root_scope,
                            statements: vec![
                                hir::Stmt {
                                    id: hir::StmtId::new(11),
                                    kind: hir::StmtKind::If {
                                        condition: hir::Expr::Binary {
                                            op: hir::BinaryOp::Ge,
                                            lhs: Box::new(cursor_pos()),
                                            rhs: Box::new(hir::Expr::SliceLen {
                                                value: Box::new(cursor_bytes()),
                                            }),
                                        },
                                        then_block: hir::Block {
                                            scope: root_scope,
                                            statements: vec![hir::Stmt {
                                                id: hir::StmtId::new(12),
                                                kind: hir::StmtKind::Break,
                                            }],
                                        },
                                        else_block: Some(hir::Block {
                                            scope: root_scope,
                                            statements: Vec::new(),
                                        }),
                                    },
                                },
                                hir::Stmt {
                                    id: hir::StmtId::new(13),
                                    kind: hir::StmtKind::Assign {
                                        place: hir::Place::Local(byte_local),
                                        value: byte_at_cursor(),
                                    },
                                },
                                hir::Stmt {
                                    id: hir::StmtId::new(14),
                                    kind: hir::StmtKind::If {
                                        condition: hir::Expr::Binary {
                                            op: hir::BinaryOp::Or,
                                            lhs: Box::new(hir::Expr::Binary {
                                                op: hir::BinaryOp::Lt,
                                                lhs: Box::new(hir::Expr::Local(byte_local)),
                                                rhs: Box::new(hir::Expr::Literal(
                                                    hir::Literal::Integer(b'0' as u64),
                                                )),
                                            }),
                                            rhs: Box::new(hir::Expr::Binary {
                                                op: hir::BinaryOp::Gt,
                                                lhs: Box::new(hir::Expr::Local(byte_local)),
                                                rhs: Box::new(hir::Expr::Literal(
                                                    hir::Literal::Integer(b'9' as u64),
                                                )),
                                            }),
                                        },
                                        then_block: hir::Block {
                                            scope: root_scope,
                                            statements: vec![hir::Stmt {
                                                id: hir::StmtId::new(15),
                                                kind: hir::StmtKind::Break,
                                            }],
                                        },
                                        else_block: Some(hir::Block {
                                            scope: root_scope,
                                            statements: vec![
                                                hir::Stmt {
                                                    id: hir::StmtId::new(16),
                                                    kind: hir::StmtKind::If {
                                                        condition: hir::Expr::Binary {
                                                            op: hir::BinaryOp::Or,
                                                            lhs: Box::new(hir::Expr::Binary {
                                                                op: hir::BinaryOp::Gt,
                                                                lhs: Box::new(hir::Expr::Local(
                                                                    acc_local,
                                                                )),
                                                                rhs: Box::new(hir::Expr::Literal(
                                                                    hir::Literal::Integer(
                                                                        u64::MAX / 10,
                                                                    ),
                                                                )),
                                                            }),
                                                            rhs: Box::new(hir::Expr::Binary {
                                                                op: hir::BinaryOp::And,
                                                                lhs: Box::new(hir::Expr::Binary {
                                                                    op: hir::BinaryOp::Eq,
                                                                    lhs: Box::new(hir::Expr::Local(
                                                                        acc_local,
                                                                    )),
                                                                    rhs: Box::new(hir::Expr::Literal(
                                                                        hir::Literal::Integer(
                                                                            u64::MAX / 10,
                                                                        ),
                                                                    )),
                                                                }),
                                                                rhs: Box::new(hir::Expr::Binary {
                                                                    op: hir::BinaryOp::Gt,
                                                                    lhs: Box::new(hir::Expr::Binary {
                                                                        op: hir::BinaryOp::Sub,
                                                                        lhs: Box::new(hir::Expr::Local(
                                                                            byte_local,
                                                                        )),
                                                                        rhs: Box::new(hir::Expr::Literal(
                                                                            hir::Literal::Integer(
                                                                                b'0' as u64,
                                                                            ),
                                                                        )),
                                                                    }),
                                                                    rhs: Box::new(hir::Expr::Literal(
                                                                        hir::Literal::Integer(
                                                                            u64::MAX % 10,
                                                                        ),
                                                                    )),
                                                                }),
                                                            }),
                                                        },
                                                        then_block: hir::Block {
                                                            scope: root_scope,
                                                            statements: vec![hir::Stmt {
                                                                id: hir::StmtId::new(17),
                                                                kind: hir::StmtKind::Fail {
                                                                    code: hir::ErrorCode::NumberOutOfRange,
                                                                },
                                                            }],
                                                        },
                                                        else_block: Some(hir::Block {
                                                            scope: root_scope,
                                                            statements: vec![],
                                                        }),
                                                    },
                                                },
                                                hir::Stmt {
                                                    id: hir::StmtId::new(18),
                                                    kind: hir::StmtKind::Assign {
                                                        place: hir::Place::Local(acc_local),
                                                        value: hir::Expr::Binary {
                                                            op: hir::BinaryOp::Add,
                                                            lhs: Box::new(hir::Expr::Binary {
                                                                op: hir::BinaryOp::Mul,
                                                                lhs: Box::new(hir::Expr::Local(
                                                                    acc_local,
                                                                )),
                                                                rhs: Box::new(hir::Expr::Literal(
                                                                    hir::Literal::Integer(10),
                                                                )),
                                                            }),
                                                            rhs: Box::new(hir::Expr::Binary {
                                                                op: hir::BinaryOp::Sub,
                                                                lhs: Box::new(hir::Expr::Local(
                                                                    byte_local,
                                                                )),
                                                                rhs: Box::new(hir::Expr::Literal(
                                                                    hir::Literal::Integer(
                                                                        b'0' as u64,
                                                                    ),
                                                                )),
                                                            }),
                                                        },
                                                    },
                                                },
                                                hir::Stmt {
                                                    id: hir::StmtId::new(19),
                                                    kind: hir::StmtKind::Assign {
                                                        place: hir::Place::Local(digit_count_local),
                                                        value: hir::Expr::Binary {
                                                            op: hir::BinaryOp::Add,
                                                            lhs: Box::new(hir::Expr::Local(
                                                                digit_count_local,
                                                            )),
                                                            rhs: Box::new(hir::Expr::Literal(
                                                                hir::Literal::Integer(1),
                                                            )),
                                                        },
                                                    },
                                                },
                                                advance_cursor_stmt(20, 1),
                                                hir::Stmt {
                                                    id: hir::StmtId::new(21),
                                                    kind: hir::StmtKind::Continue,
                                                },
                                            ],
                                        }),
                                    },
                                },
                            ],
                        },
                        max_iterations: None,
                    },
                },
                hir::Stmt {
                    id: hir::StmtId::new(22),
                    kind: hir::StmtKind::If {
                        condition: hir::Expr::Binary {
                            op: hir::BinaryOp::Eq,
                            lhs: Box::new(hir::Expr::Local(digit_count_local)),
                            rhs: Box::new(hir::Expr::Literal(hir::Literal::Integer(0))),
                        },
                        then_block: hir::Block {
                            scope: root_scope,
                            statements: vec![hir::Stmt {
                                id: hir::StmtId::new(23),
                                kind: hir::StmtKind::Fail {
                                    code: hir::ErrorCode::InvalidJsonNumber,
                                },
                            }],
                        },
                        else_block: Some(hir::Block {
                            scope: root_scope,
                            statements: vec![],
                        }),
                    },
                },
                hir::Stmt {
                    id: hir::StmtId::new(24),
                    kind: hir::StmtKind::Init {
                        place: hir::Place::Local(out_local),
                        value: hir::Expr::Local(acc_local),
                    },
                },
                hir::Stmt {
                    id: hir::StmtId::new(25),
                    kind: hir::StmtKind::Return(None),
                },
            ],
        },
    });

    module
}
