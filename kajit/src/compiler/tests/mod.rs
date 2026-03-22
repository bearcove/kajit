use kajit_hir_text::parse_hir;
use serde::Serialize;

use super::*;
use facet::Facet;

#[derive(Facet)]
struct Wrapper<T> {
    inner: T,
}

#[derive(Facet)]
struct ConstWrapper<const N: usize> {
    inner: [u8; N],
}

#[derive(Debug, PartialEq, Eq, Facet)]
struct BorrowedHeader<'a> {
    len: u32,
    name: &'a str,
}

#[derive(Debug, PartialEq, Eq, Facet)]
struct OwnedHeader {
    len: u32,
    name: String,
}

#[derive(Debug, PartialEq, Eq, Facet)]
struct OwnedAddress {
    city: String,
    zip: u32,
}

#[derive(Debug, PartialEq, Facet)]
struct FloatHeader {
    a: f32,
    b: f64,
}

#[derive(Debug, PartialEq, Eq, Facet)]
struct CharHeader {
    ch: char,
}

#[derive(Debug, PartialEq, Eq, Facet, Serialize)]
struct BigUnsigned {
    value: u128,
}

#[derive(Debug, PartialEq, Eq, Facet, Serialize)]
struct BigSigned {
    value: i128,
}

#[derive(Debug, PartialEq, Eq, Facet, Serialize)]
struct MaybeBigUnsigned {
    value: Option<u128>,
}

#[derive(Debug, PartialEq, Eq, Facet)]
struct MaybeBorrowedName<'a> {
    name: Option<&'a str>,
}

#[derive(Debug, PartialEq, Eq, Facet)]
struct MaybeCount {
    count: Option<u32>,
}

#[derive(Debug, PartialEq, Eq, Facet)]
#[repr(u8)]
enum UnitAnimal {
    Cat,
    Dog,
    Parrot,
}

#[derive(Debug, PartialEq, Eq, Facet)]
#[repr(u8)]
enum PayloadAnimal<'a> {
    Cat,
    Count(u32),
    Name(&'a str),
}

#[derive(Debug, PartialEq, Eq, Facet)]
#[repr(u8)]
enum OwnedAnimal {
    Cat,
    Dog { name: String, good_boy: bool },
    Parrot(String),
}

#[derive(Debug, PartialEq, Eq, Facet)]
struct OwnedZoo {
    name: String,
    star: OwnedAnimal,
}

#[derive(Debug, PartialEq, Eq, Facet)]
struct ConstantNumber {
    value: u32,
}

#[derive(Debug, PartialEq, Eq, Facet)]
struct ScalarNumber {
    value: u32,
}

#[derive(Debug, PartialEq, Eq, Facet)]
struct BoolHeader {
    value: bool,
}

#[derive(Debug, PartialEq, Eq, Facet)]
struct ScalarArrayHolder {
    values: [u32; 4],
}

#[derive(Debug, PartialEq, Eq, Facet)]
struct BorrowedArrayHolder<'a> {
    values: [&'a str; 2],
}

#[derive(Debug, PartialEq, Eq, Facet)]
struct BranchyAnimal {
    animal: UnitAnimal,
    value: u32,
}

#[derive(Debug, PartialEq, Eq, Facet)]
struct MaskSummary {
    masked: u32,
    shifted: u32,
    toggled: u32,
    combined: u32,
}

#[derive(Debug, PartialEq, Eq, Facet)]
struct ScratchSummary {
    mask: u32,
    done: u32,
}

#[derive(Debug, PartialEq, Eq, Facet)]
struct DynamicIndexSummary {
    selected: u32,
}

#[derive(Debug, PartialEq, Eq, Facet)]
struct DynamicDestinationSummary {
    values: [u32; 4],
    selected: u32,
}

#[derive(Debug, PartialEq, Eq, Facet)]
struct PersistentBufferSummary {
    ptr: usize,
    len: usize,
}

#[derive(Debug, PartialEq, Eq, Facet)]
struct VecHolder {
    values: Vec<u32>,
}

#[derive(Debug, PartialEq, Eq, Facet)]
struct Pair {
    lo: u64,
    hi: u64,
}

#[derive(Debug, PartialEq, Eq, Facet)]
struct DynamicAggregateSummary {
    pair: Pair,
}

#[derive(Debug, PartialEq, Eq, Facet)]
struct DynamicAggregateDestinationSummary {
    pairs: [Pair; 2],
    selected: Pair,
}

fn compile_structural_hir_decoder(shape: &'static Shape, module: &hir::Module) -> CompiledDecoder {
    let mut func = build_structural_hir_ir(shape, module);
    run_default_passes_from_env(&mut func);
    let linear = crate::linearize::linearize(&mut func);
    compile_linear_ir_decoder(&linear, false)
}

fn compile_postcard_decoder_via_structural_hir(shape: &'static Shape) -> CompiledDecoder {
    let module = build_postcard_decoder_hir(shape);
    compile_structural_hir_decoder(shape, &module)
}

fn compile_json_decoder_via_structural_hir(shape: &'static Shape) -> CompiledDecoder {
    let module = build_json_decoder_hir(shape);
    compile_structural_hir_decoder(shape, &module)
}

fn build_structural_json_bool_module() -> hir::Module {
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
                },
                hir::FieldDef {
                    name: "pos".to_owned(),
                    ty: hir::Type::u(64),
                },
            ],
        },
    });
    let root_def = module.add_type_def(hir::TypeDef {
        name: <BoolHeader>::SHAPE.type_identifier.to_owned(),
        generic_params: vec![],
        kind: hir::TypeDefKind::Struct {
            fields: vec![hir::FieldDef {
                name: "value".to_owned(),
                ty: hir::Type::bool(),
            }],
        },
    });

    let cursor_local = hir::LocalId::new(0);
    let out_local = hir::LocalId::new(1);
    let byte_local = hir::LocalId::new(2);

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
    let advance_cursor_stmt = |stmt_id: u32, delta: u64| hir::Stmt {
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
                        scope: hir::ScopeId::new(0),
                        statements: vec![hir::Stmt {
                            id: hir::StmtId::new(fail_stmt),
                            kind: hir::StmtKind::Fail { code: error },
                        }],
                    },
                    else_block: Some(hir::Block {
                        scope: hir::ScopeId::new(0),
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
                        scope: hir::ScopeId::new(0),
                        statements: vec![hir::Stmt {
                            id: hir::StmtId::new(mismatch_stmt + 1),
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
            });
        }
        statements
    };

    module.add_function(hir::Function {
        name: "json_bool".to_owned(),
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
        locals: vec![hir::LocalDecl {
            local: byte_local,
            name: "byte".to_owned(),
            ty: hir::Type::u(8),
            kind: hir::LocalKind::Temp,
        }],
        return_type: hir::Type::unit(),
        scopes: vec![hir::Scope {
            id: hir::ScopeId::new(0),
            parent: None,
            comment: Some("structural JSON bool parser".to_owned()),
        }],
        body: hir::Block {
            scope: hir::ScopeId::new(0),
            statements: vec![
                hir::Stmt {
                    id: hir::StmtId::new(0),
                    kind: hir::StmtKind::Loop {
                        body: hir::Block {
                            scope: hir::ScopeId::new(0),
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
                                            scope: hir::ScopeId::new(0),
                                            statements: vec![
                                                advance_cursor_stmt(5, 1),
                                                hir::Stmt {
                                                    id: hir::StmtId::new(6),
                                                    kind: hir::StmtKind::Continue,
                                                },
                                            ],
                                        },
                                        else_block: Some(hir::Block {
                                            scope: hir::ScopeId::new(0),
                                            statements: vec![hir::Stmt {
                                                id: hir::StmtId::new(7),
                                                kind: hir::StmtKind::Break,
                                            }],
                                        }),
                                    },
                                },
                            ],
                        },
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
                                    place: hir::Place::Field {
                                        base: Box::new(hir::Place::Local(out_local)),
                                        field: "value".to_owned(),
                                    },
                                    value: hir::Expr::Literal(hir::Literal::Bool(true)),
                                },
                            });
                            statements.push(hir::Stmt {
                                id: hir::StmtId::new(next_stmt + 2),
                                kind: hir::StmtKind::Return(None),
                            });
                            hir::Block {
                                scope: hir::ScopeId::new(0),
                                statements,
                            }
                        },
                        else_block: Some(hir::Block {
                            scope: hir::ScopeId::new(0),
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
                                                place: hir::Place::Field {
                                                    base: Box::new(hir::Place::Local(out_local)),
                                                    field: "value".to_owned(),
                                                },
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
                                            scope: hir::ScopeId::new(0),
                                            statements,
                                        }
                                    },
                                    else_block: Some(hir::Block {
                                        scope: hir::ScopeId::new(0),
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

#[test]
fn instantiated_shape_symbol_key_distinguishes_generic_instantiations() {
    let u32_key = instantiated_shape_symbol_key(<Wrapper<u32>>::SHAPE);
    let string_key = instantiated_shape_symbol_key(<Wrapper<String>>::SHAPE);

    assert_ne!(u32_key, string_key);
    assert!(u32_key.starts_with('d'));
    assert!(u32_key.contains("__t_0_d"));
    assert_eq!(
        <Wrapper<u32>>::SHAPE.decl_id,
        <Wrapper<String>>::SHAPE.decl_id
    );
}

#[test]
fn instantiated_shape_symbol_key_includes_const_params() {
    let n4_key = instantiated_shape_symbol_key(<ConstWrapper<4>>::SHAPE);
    let n8_key = instantiated_shape_symbol_key(<ConstWrapper<8>>::SHAPE);

    assert_ne!(n4_key, n8_key);
    assert!(n4_key.contains("__c_0_u4"));
    assert!(n8_key.contains("__c_0_u8"));
}

#[test]
fn postcard_hir_models_borrowed_output_structs() {
    let module = build_postcard_decoder_hir(<BorrowedHeader<'static>>::SHAPE);
    assert_eq!(module.functions.len(), 1);

    let (_, function) = module.functions.iter().next().unwrap();
    assert_eq!(function.params.len(), 2);
    assert_eq!(function.region_params.len(), 1);

    let destination = function.destination_param().unwrap();
    let hir::Type::Named { def, args } = &destination.ty else {
        panic!("expected named root output type");
    };
    assert_eq!(
        module.type_defs[*def].name,
        <BorrowedHeader<'static>>::SHAPE.type_identifier
    );
    assert_eq!(
        args,
        &vec![hir::GenericArg::Region(function.region_params[0])]
    );

    let statements = &function.body.statements;
    assert!(
        module
            .callable_named("runtime.validate_utf8_range")
            .is_some(),
        "borrowed string lowering should install runtime UTF-8 validation"
    );
    assert!(
        module.callable_named("postcard.read_str").is_none(),
        "borrowed string lowering should not use postcard.read_str"
    );
    assert!(
        module.callable_named("postcard.read_u32").is_none(),
        "borrowed header lowering should not use postcard.read_u32"
    );
    assert!(
        statements.iter().any(|stmt| matches!(
            &stmt.kind,
            hir::StmtKind::Expr(hir::Expr::Call(hir::CallExpr {
                target: hir::CallTarget::Callable(callable_id),
                ..
            })) if module.callables[*callable_id].name == "runtime.validate_utf8_range"
        )),
        "borrowed string lowering should validate UTF-8 explicitly"
    );
    assert!(
        statements.iter().any(|stmt| matches!(
            &stmt.kind,
            hir::StmtKind::Init {
                place: hir::Place::Field { field, .. },
                value: hir::Expr::Str { .. },
            } if field == "name"
        )),
        "borrowed string lowering should materialize a str value directly into the destination"
    );
}

#[test]
fn json_hir_models_root_bool_without_reader_calls() {
    let module = build_json_decoder_hir(<bool>::SHAPE);
    let (_, function) = module.functions.iter().next().unwrap();

    assert!(
        module.callable_named("postcard.read_bool").is_none(),
        "json root bool HIR should not mention postcard bool readers"
    );
    assert!(
        module.callable_named("kajit_json_read_bool").is_none(),
        "json root bool HIR should not call the old JSON bool intrinsic"
    );
    assert!(
        module.callables.is_empty(),
        "root bool HIR should be leaf-free"
    );
    assert!(
        function
            .body
            .statements
            .iter()
            .any(|stmt| matches!(&stmt.kind, hir::StmtKind::Loop { .. })),
        "json root bool HIR should spell out whitespace skipping as control flow"
    );
    assert!(
        function
            .body
            .statements
            .iter()
            .any(|stmt| matches!(&stmt.kind, hir::StmtKind::If { .. })),
        "json root bool HIR should spell out token dispatch as control flow"
    );
}

#[test]
fn json_hir_models_root_u32_without_reader_calls() {
    let module = build_json_decoder_hir(<u32>::SHAPE);
    let (_, function) = module.functions.iter().next().unwrap();

    assert!(
        module.callable_named("postcard.read_u32").is_none(),
        "json root u32 HIR should not mention postcard integer readers"
    );
    assert!(
        module.callable_named("kajit_json_read_u32").is_none(),
        "json root u32 HIR should not call the old JSON u32 intrinsic"
    );
    assert!(
        module.callables.is_empty(),
        "root u32 HIR should be leaf-free"
    );
    assert!(
        function
            .body
            .statements
            .iter()
            .any(|stmt| matches!(&stmt.kind, hir::StmtKind::Loop { .. })),
        "json root u32 HIR should spell out whitespace and digit scanning as control flow"
    );
}

#[test]
fn json_hir_models_root_u64_without_reader_calls() {
    let module = build_json_decoder_hir(<u64>::SHAPE);
    let (_, function) = module.functions.iter().next().unwrap();

    assert!(
        module.callable_named("postcard.read_u64").is_none(),
        "json root u64 HIR should not mention postcard integer readers"
    );
    assert!(
        module.callable_named("kajit_json_read_u64").is_none(),
        "json root u64 HIR should not call the old JSON u64 intrinsic"
    );
    assert!(
        module.callables.is_empty(),
        "root u64 HIR should be leaf-free"
    );
    assert!(
        function
            .body
            .statements
            .iter()
            .any(|stmt| matches!(&stmt.kind, hir::StmtKind::Loop { .. })),
        "json root u64 HIR should spell out whitespace and digit scanning as control flow"
    );
}

#[test]
fn compile_decoder_prefers_hir_for_supported_json_root_u32() {
    let decoder = crate::compile_decoder(<u32>::SHAPE, &crate::json::KajitJson);
    let listing = decoder.cfg_mir_line_text_by_line.join("\n");

    assert!(
        !listing.contains("kajit_json_read_u32"),
        "supported JSON shapes should compile through the HIR path"
    );
    assert!(
        listing.contains("branch_if_zero") || listing.contains("branch "),
        "HIR JSON lowering should still produce explicit control flow"
    );
}

#[test]
fn postcard_hir_models_owned_output_strings() {
    let module = build_postcard_decoder_hir(<OwnedHeader>::SHAPE);
    let (_, function) = module.functions.iter().next().unwrap();

    assert!(
        module
            .callable_named("runtime.string_validate_alloc_copy")
            .is_some(),
        "owned string lowering should install the raw string allocation helper"
    );
    assert!(
        module.callable_named("postcard.read_str").is_none(),
        "owned string lowering should not use postcard.read_str"
    );
    assert!(
        function
            .locals
            .iter()
            .any(|local| matches!(local.ty, hir::Type::Address { .. })),
        "owned string lowering should allocate a persistent data pointer local"
    );
    assert!(
        function.body.statements.iter().any(|stmt| matches!(
            &stmt.kind,
            hir::StmtKind::Init {
                value: hir::Expr::Call(hir::CallExpr {
                    target: hir::CallTarget::Callable(callable_id),
                    ..
                }),
                ..
            } if module.callables[*callable_id].name == "runtime.string_validate_alloc_copy"
        )),
        "owned string lowering should compute string storage through the lean helper"
    );
}

#[test]
fn compile_decoder_prefers_hir_for_supported_postcard_bool_field() {
    let decoder = crate::compile_decoder(<BoolHeader>::SHAPE, &crate::postcard::KajitPostcard);
    let listing = decoder.cfg_mir_line_text_by_line.join("\n");

    assert!(
        !listing.contains("kajit_read_bool"),
        "supported postcard shapes should compile through the HIR path"
    );
}

#[test]
fn postcard_hir_models_float_scalars_without_reader_calls() {
    let module = build_postcard_decoder_hir(<FloatHeader>::SHAPE);

    assert!(
        module.callable_named("postcard.read_f32").is_none(),
        "float lowering should not use postcard.read_f32"
    );
    assert!(
        module.callable_named("postcard.read_f64").is_none(),
        "float lowering should not use postcard.read_f64"
    );
}

#[test]
fn postcard_hir_models_char_without_reader_calls() {
    let module = build_postcard_decoder_hir(<CharHeader>::SHAPE);

    assert!(
        module.callable_named("postcard.read_char").is_none(),
        "char lowering should not use postcard.read_char"
    );
    assert!(
        module
            .callable_named("runtime.validate_utf8_range")
            .is_some(),
        "char lowering should validate UTF-8 explicitly"
    );
}

#[test]
fn postcard_hir_models_128bit_scalars_without_reader_calls() {
    let unsigned = build_postcard_decoder_hir(<BigUnsigned>::SHAPE);
    let signed = build_postcard_decoder_hir(<BigSigned>::SHAPE);
    let optional = build_postcard_decoder_hir(<MaybeBigUnsigned>::SHAPE);

    assert!(
        unsigned.callable_named("postcard.read_u128").is_none(),
        "u128 lowering should not use postcard.read_u128"
    );
    assert!(
        signed.callable_named("postcard.read_i128").is_none(),
        "i128 lowering should not use postcard.read_i128"
    );
    assert!(
        optional.callable_named("postcard.read_u128").is_none(),
        "Option<u128> lowering should not use postcard.read_u128"
    );
}

#[test]
fn postcard_hir_models_option_borrowed_fields() {
    let module = build_postcard_decoder_hir(<MaybeBorrowedName<'static>>::SHAPE);
    let (_, function) = module.functions.iter().next().unwrap();
    let input_region = function.region_params[0];

    assert!(
        module.callable_named("postcard.read_option_tag").is_none(),
        "option lowering should not use postcard.read_option_tag"
    );

    assert!(function.locals.len() >= 4);
    assert!(
        function
            .locals
            .iter()
            .any(|local| local.ty == hir::Type::bool())
    );
    assert!(
        function
            .locals
            .iter()
            .any(|local| local.ty == hir::Type::u(8))
    );
    assert!(
        function
            .locals
            .iter()
            .any(|local| local.ty == hir::Type::str(input_region))
    );

    let (_, then_block, else_block) = function
        .body
        .statements
        .iter()
        .find_map(|stmt| match &stmt.kind {
            hir::StmtKind::If {
                condition,
                then_block,
                else_block,
            } if matches!(condition, hir::Expr::Local(_)) => {
                Some((condition, then_block, else_block))
            }
            _ => None,
        })
        .expect("expected option if statement");

    let Some(else_block) = else_block else {
        panic!("expected explicit option else block");
    };
    assert!(then_block.statements.len() >= 2);
    assert_eq!(else_block.statements.len(), 1);
    assert!(
        then_block.statements.iter().any(|stmt| matches!(
            &stmt.kind,
            hir::StmtKind::Expr(hir::Expr::Call(hir::CallExpr {
                target: hir::CallTarget::Callable(callable_id),
                ..
            })) if module.callables[*callable_id].name == "runtime.validate_utf8_range"
        )),
        "borrowed option payload should validate UTF-8 explicitly"
    );

    let value = then_block
            .statements
            .iter()
            .find_map(|stmt| match &stmt.kind {
                hir::StmtKind::Init { value, .. }
                    if matches!(value, hir::Expr::Variant { variant, .. } if variant == "Some") =>
                {
                    Some(value)
                }
                _ => None,
            })
            .expect("expected Some variant init");
    let hir::Expr::Variant {
        def,
        variant,
        fields,
    } = value
    else {
        panic!("expected Some variant expression");
    };
    assert_eq!(variant, "Some");
    assert_eq!(fields.len(), 1);
    let hir::Expr::Local(payload_local) = &fields[0].1 else {
        panic!("expected Some variant payload local");
    };
    assert_eq!(
        function
            .locals
            .iter()
            .find(|local| local.ty == hir::Type::str(input_region))
            .expect("expected borrowed payload local")
            .local,
        *payload_local
    );

    let hir::TypeDefKind::Enum { variants } = &module.type_defs[*def].kind else {
        panic!("expected Option HIR enum type");
    };
    assert_eq!(variants[0].name, "None");
    assert_eq!(variants[1].name, "Some");
    assert_eq!(variants[1].fields[0].ty, hir::Type::str(input_region));

    let value = else_block
            .statements
            .iter()
            .find_map(|stmt| match &stmt.kind {
                hir::StmtKind::Init { value, .. }
                    if matches!(value, hir::Expr::Variant { variant, .. } if variant == "None") =>
                {
                    Some(value)
                }
                _ => None,
            })
            .expect("expected None variant init");
    let hir::Expr::Variant {
        variant, fields, ..
    } = value
    else {
        panic!("expected None variant expression");
    };
    assert_eq!(variant, "None");
    assert!(fields.is_empty());
}

#[test]
fn postcard_hir_text_round_trips() {
    std::thread::Builder::new()
        .name("postcard_hir_text_round_trips".to_owned())
        .stack_size(32 * 1024 * 1024)
        .spawn(|| {
            let module = build_postcard_decoder_hir(<MaybeBorrowedName<'static>>::SHAPE);
            let text = module.to_string();
            let reparsed = parse_hir(&text).expect("postcard HIR text should parse");

            assert_eq!(reparsed, module);
        })
        .expect("thread should spawn")
        .join()
        .expect("round-trip thread should succeed");
}

#[test]
fn postcard_hir_ir_path_decodes_option_borrowed_fields() {
    let decoder = crate::compile_postcard_decoder_via_hir(<MaybeBorrowedName<'static>>::SHAPE);

    let some = crate::deserialize::<MaybeBorrowedName<'_>>(&decoder, &[1, 2, b'h', b'i'])
        .expect("HIR->RVSDG postcard decoder should decode Some(&str)");
    assert_eq!(some, MaybeBorrowedName { name: Some("hi") });

    let none = crate::deserialize::<MaybeBorrowedName<'_>>(&decoder, &[0])
        .expect("HIR->RVSDG postcard decoder should decode None");
    assert_eq!(none, MaybeBorrowedName { name: None });
}

#[test]
fn postcard_structural_hir_ir_path_decodes_float_fields() {
    let decoder = compile_postcard_decoder_via_structural_hir(<FloatHeader>::SHAPE);

    let mut bytes = Vec::new();
    bytes.extend_from_slice(&3.14f32.to_le_bytes());
    bytes.extend_from_slice(&2.718281828459045f64.to_le_bytes());

    let value = crate::deserialize::<FloatHeader>(&decoder, &bytes)
        .expect("structural HIR postcard decoder should decode float fields");
    assert_eq!(value.a.to_bits(), 3.14f32.to_bits());
    assert_eq!(value.b.to_bits(), 2.718281828459045f64.to_bits());
}

#[test]
fn postcard_structural_hir_ir_path_decodes_char_field() {
    let decoder = compile_postcard_decoder_via_structural_hir(<CharHeader>::SHAPE);

    let value = crate::deserialize::<CharHeader>(&decoder, &[2, 0xC3, 0x9F])
        .expect("structural HIR postcard decoder should decode char fields");
    assert_eq!(value, CharHeader { ch: 'ß' });
}

#[test]
fn postcard_structural_hir_ir_path_decodes_128bit_fields() {
    let unsigned = BigUnsigned {
        value: (1_u128 << 100) | 0x1234_5678_9abc_def0_u128,
    };
    let signed = BigSigned {
        value: -((1_i128 << 97) - 0x1234_5678_9abc_i128),
    };

    let unsigned_decoder = compile_postcard_decoder_via_structural_hir(<BigUnsigned>::SHAPE);
    let unsigned_bytes =
        postcard::to_allocvec(&unsigned).expect("postcard should encode unsigned 128-bit sample");
    let unsigned_value = crate::deserialize::<BigUnsigned>(&unsigned_decoder, &unsigned_bytes)
        .expect("structural HIR postcard decoder should decode u128 fields");
    assert_eq!(unsigned_value, unsigned);

    let signed_decoder = compile_postcard_decoder_via_structural_hir(<BigSigned>::SHAPE);
    let signed_bytes =
        postcard::to_allocvec(&signed).expect("postcard should encode signed 128-bit sample");
    let signed_value = crate::deserialize::<BigSigned>(&signed_decoder, &signed_bytes)
        .expect("structural HIR postcard decoder should decode i128 fields");
    assert_eq!(signed_value, signed);
}

#[test]
fn postcard_structural_hir_ir_path_decodes_option_u128_field() {
    let decoder = compile_postcard_decoder_via_structural_hir(<MaybeBigUnsigned>::SHAPE);
    let sample = MaybeBigUnsigned {
        value: Some((1_u128 << 72) | 0x55aa_33cc_77ee_u128),
    };
    let bytes = postcard::to_allocvec(&sample).expect("postcard should encode Option<u128>");
    let value = crate::deserialize::<MaybeBigUnsigned>(&decoder, &bytes)
        .expect("structural HIR postcard decoder should decode Option<u128>");
    assert_eq!(value, sample);
}

#[test]
fn postcard_hir_models_unit_enums() {
    let module = build_postcard_decoder_hir(<UnitAnimal>::SHAPE);
    let (_, function) = module.functions.iter().next().unwrap();

    assert!(
        module
            .callable_named("postcard.read_discriminant")
            .is_none(),
        "unit enum lowering should not use postcard.read_discriminant"
    );

    assert!(
        function
            .locals
            .iter()
            .any(|local| local.ty == hir::Type::u(32)),
        "unit enum lowering should use a scalar discriminant local"
    );
    let (scrutinee, arms) = function
        .body
        .statements
        .iter()
        .find_map(|stmt| match &stmt.kind {
            hir::StmtKind::Match { scrutinee, arms } => Some((scrutinee, arms)),
            _ => None,
        })
        .expect("expected enum match statement");
    let hir::Expr::Local(disc_local) = scrutinee else {
        panic!("expected discriminant local");
    };
    assert!(
        function
            .locals
            .iter()
            .any(|local| local.local == *disc_local),
        "match should scrutinee the decoded discriminant local"
    );
    assert_eq!(arms.len(), 3);
    assert!(matches!(arms[0].pattern, hir::Pattern::Integer(0)));
    assert!(matches!(arms[1].pattern, hir::Pattern::Integer(1)));
    assert!(matches!(arms[2].pattern, hir::Pattern::Integer(2)));

    let hir::StmtKind::Init { value, .. } = &arms[1].body.statements[0].kind else {
        panic!("expected unit variant init");
    };
    let hir::Expr::Variant {
        variant, fields, ..
    } = value
    else {
        panic!("expected unit variant expression");
    };
    assert_eq!(variant, "Dog");
    assert!(fields.is_empty());
}

#[test]
fn postcard_hir_models_payload_enums() {
    let module = build_postcard_decoder_hir(<PayloadAnimal<'static>>::SHAPE);
    let (_, function) = module.functions.iter().next().unwrap();

    let arms = function
        .body
        .statements
        .iter()
        .find_map(|stmt| match &stmt.kind {
            hir::StmtKind::Match { arms, .. } => Some(arms),
            _ => None,
        })
        .expect("expected enum match statement");
    assert_eq!(arms.len(), 3);

    let count_arm = &arms[1];
    let value = count_arm
            .body
            .statements
            .iter()
            .find_map(|stmt| match &stmt.kind {
                hir::StmtKind::Init { value, .. }
                    if matches!(value, hir::Expr::Variant { variant, .. } if variant == "Count") =>
                {
                    Some(value)
                }
                _ => None,
            })
            .expect("expected Count variant init");
    let hir::Expr::Variant {
        variant, fields, ..
    } = value
    else {
        panic!("expected Count variant expression");
    };
    assert_eq!(variant, "Count");
    assert_eq!(fields.len(), 1);

    let name_arm = &arms[2];
    assert!(name_arm.body.statements.len() >= 2);
    let value = name_arm
            .body
            .statements
            .iter()
            .find_map(|stmt| match &stmt.kind {
                hir::StmtKind::Init { value, .. }
                    if matches!(value, hir::Expr::Variant { variant, .. } if variant == "Name") =>
                {
                    Some(value)
                }
                _ => None,
            })
            .expect("expected Name variant init");
    let hir::Expr::Variant {
        variant, fields, ..
    } = value
    else {
        panic!("expected Name variant expression");
    };
    assert_eq!(variant, "Name");
    assert_eq!(fields.len(), 1);
    assert!(
        name_arm.body.statements.iter().any(|stmt| matches!(
            &stmt.kind,
            hir::StmtKind::Expr(hir::Expr::Call(hir::CallExpr {
                target: hir::CallTarget::Callable(callable_id),
                ..
            })) if module.callables[*callable_id].name == "runtime.validate_utf8_range"
        )),
        "borrowed enum payload should validate UTF-8 explicitly"
    );
}

#[test]
fn postcard_hir_models_arrays() {
    let module = build_postcard_decoder_hir(<BorrowedArrayHolder<'static>>::SHAPE);
    let (_, function) = module.functions.iter().next().unwrap();

    let array_inits = function
        .body
        .statements
        .iter()
        .filter_map(|stmt| match &stmt.kind {
            hir::StmtKind::Init {
                place: hir::Place::Index { .. },
                value,
            } => Some(value),
            _ => None,
        })
        .collect::<Vec<_>>();
    assert_eq!(array_inits.len(), 2);
    for value in array_inits {
        assert!(matches!(value, hir::Expr::Str { .. } | hir::Expr::Local(_)));
    }
    assert!(
        module.callable_named("postcard.read_str").is_none(),
        "borrowed array lowering should not use postcard.read_str"
    );
}

#[test]
fn postcard_hir_ir_path_decodes_unit_enums() {
    let decoder = crate::compile_postcard_decoder_via_hir(<UnitAnimal>::SHAPE);

    let cat = crate::deserialize::<UnitAnimal>(&decoder, &[0])
        .expect("HIR->RVSDG postcard decoder should decode Cat");
    assert_eq!(cat, UnitAnimal::Cat);

    let dog = crate::deserialize::<UnitAnimal>(&decoder, &[1])
        .expect("HIR->RVSDG postcard decoder should decode Dog");
    assert_eq!(dog, UnitAnimal::Dog);

    let parrot = crate::deserialize::<UnitAnimal>(&decoder, &[2])
        .expect("HIR->RVSDG postcard decoder should decode Parrot");
    assert_eq!(parrot, UnitAnimal::Parrot);
}

#[test]
fn postcard_structural_hir_ir_path_decodes_scalar_field() {
    let decoder = compile_postcard_decoder_via_structural_hir(<ScalarNumber>::SHAPE);

    let value = crate::deserialize::<ScalarNumber>(&decoder, &[42])
        .expect("structural HIR postcard decoder should decode a scalar field");
    assert_eq!(value, ScalarNumber { value: 42 });
}

#[test]
fn postcard_structural_hir_ir_path_decodes_borrowed_header() {
    let decoder = compile_postcard_decoder_via_structural_hir(<BorrowedHeader<'static>>::SHAPE);

    let value = crate::deserialize::<BorrowedHeader<'_>>(&decoder, &[7, 2, b'h', b'i'])
        .expect("structural HIR postcard decoder should decode direct borrowed fields");
    assert_eq!(value, BorrowedHeader { len: 7, name: "hi" });
}

#[test]
fn postcard_structural_hir_ir_path_decodes_owned_header() {
    let decoder = compile_postcard_decoder_via_structural_hir(<OwnedHeader>::SHAPE);

    let value = crate::deserialize::<OwnedHeader>(&decoder, &[7, 2, b'h', b'i'])
        .expect("structural HIR postcard decoder should decode direct owned string fields");
    assert_eq!(
        value,
        OwnedHeader {
            len: 7,
            name: "hi".to_owned(),
        }
    );
}

#[test]
fn postcard_structural_hir_ir_path_decodes_root_vec_u32() {
    let decoder = compile_postcard_decoder_via_structural_hir(<Vec<u32>>::SHAPE);

    let value = crate::deserialize::<Vec<u32>>(&decoder, &[3, 1, 2, 3])
        .expect("structural HIR postcard decoder should decode root Vec<u32>");
    assert_eq!(value, vec![1, 2, 3]);
}

#[test]
fn postcard_structural_hir_ir_path_decodes_root_vec_string() {
    let decoder = compile_postcard_decoder_via_structural_hir(<Vec<String>>::SHAPE);

    let value = crate::deserialize::<Vec<String>>(&decoder, &[2, 2, b'h', b'i', 2, b'b', b'y'])
        .expect("structural HIR postcard decoder should decode root Vec<String>");
    assert_eq!(value, vec!["hi".to_owned(), "by".to_owned()]);
}

#[test]
fn postcard_structural_hir_ir_path_decodes_root_vec_structs() {
    let decoder = compile_postcard_decoder_via_structural_hir(<Vec<OwnedAddress>>::SHAPE);

    let value = crate::deserialize::<Vec<OwnedAddress>>(
        &decoder,
        &[2, 2, b'P', b'A', 75, 2, b'L', b'Y', 13],
    )
    .expect("structural HIR postcard decoder should decode root Vec<struct>");
    assert_eq!(
        value,
        vec![
            OwnedAddress {
                city: "PA".to_owned(),
                zip: 75,
            },
            OwnedAddress {
                city: "LY".to_owned(),
                zip: 13,
            },
        ]
    );
}

#[test]
fn postcard_structural_hir_ir_path_decodes_unit_enums() {
    let decoder = compile_postcard_decoder_via_structural_hir(<UnitAnimal>::SHAPE);

    let cat = crate::deserialize::<UnitAnimal>(&decoder, &[0])
        .expect("structural HIR postcard decoder should decode Cat");
    assert_eq!(cat, UnitAnimal::Cat);

    let dog = crate::deserialize::<UnitAnimal>(&decoder, &[1])
        .expect("structural HIR postcard decoder should decode Dog");
    assert_eq!(dog, UnitAnimal::Dog);

    let parrot = crate::deserialize::<UnitAnimal>(&decoder, &[2])
        .expect("structural HIR postcard decoder should decode Parrot");
    assert_eq!(parrot, UnitAnimal::Parrot);
}

#[test]
fn postcard_structural_hir_ir_path_decodes_option_scalar_field() {
    let decoder = compile_postcard_decoder_via_structural_hir(<MaybeCount>::SHAPE);

    let some = crate::deserialize::<MaybeCount>(&decoder, &[1, 42])
        .expect("structural HIR postcard decoder should decode Some(u32)");
    assert_eq!(some, MaybeCount { count: Some(42) });

    let none = crate::deserialize::<MaybeCount>(&decoder, &[0])
        .expect("structural HIR postcard decoder should decode None");
    assert_eq!(none, MaybeCount { count: None });
}

#[test]
fn postcard_structural_hir_ir_path_decodes_option_borrowed_field() {
    let decoder = compile_postcard_decoder_via_structural_hir(<MaybeBorrowedName<'static>>::SHAPE);

    let some = crate::deserialize::<MaybeBorrowedName<'_>>(&decoder, &[1, 2, b'h', b'i'])
        .expect("structural HIR postcard decoder should decode Some(&str)");
    assert_eq!(some, MaybeBorrowedName { name: Some("hi") });

    let none = crate::deserialize::<MaybeBorrowedName<'_>>(&decoder, &[0])
        .expect("structural HIR postcard decoder should decode None");
    assert_eq!(none, MaybeBorrowedName { name: None });
}

#[test]
fn structural_hir_ir_path_decodes_constant_output() {
    let mut module = hir::Module::new();
    let root_def = module.add_type_def(hir::TypeDef {
        name: <ConstantNumber>::SHAPE.type_identifier.to_owned(),
        generic_params: vec![],
        kind: hir::TypeDefKind::Struct {
            fields: vec![hir::FieldDef {
                name: "value".to_owned(),
                ty: hir::Type::u(32),
            }],
        },
    });
    module.add_function(hir::Function {
        name: "const_number".to_owned(),
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
            comment: Some("constant structural HIR".to_owned()),
        }],
        body: hir::Block {
            scope: hir::ScopeId::new(0),
            statements: vec![
                hir::Stmt {
                    id: hir::StmtId::new(0),
                    kind: hir::StmtKind::Init {
                        place: hir::Place::Field {
                            base: Box::new(hir::Place::Local(hir::LocalId::new(1))),
                            field: "value".to_owned(),
                        },
                        value: hir::Expr::Literal(hir::Literal::Integer(42)),
                    },
                },
                hir::Stmt {
                    id: hir::StmtId::new(1),
                    kind: hir::StmtKind::Return(None),
                },
            ],
        },
    });

    let decoder = compile_structural_hir_decoder(<ConstantNumber>::SHAPE, &module);
    let value = crate::deserialize::<ConstantNumber>(&decoder, &[])
        .expect("structural HIR decoder should ignore input and write a constant");
    assert_eq!(value, ConstantNumber { value: 42 });
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
            }],
        },
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
            }],
        },
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
                },
                hir::FieldDef {
                    name: "pos".to_owned(),
                    ty: hir::Type::u(64),
                },
            ],
        },
    });
    let root_def = module.add_type_def(hir::TypeDef {
        name: <ScalarArrayHolder>::SHAPE.type_identifier.to_owned(),
        generic_params: vec![],
        kind: hir::TypeDefKind::Struct {
            fields: vec![hir::FieldDef {
                name: "values".to_owned(),
                ty: hir::Type::array(hir::Type::u(32), 4),
            }],
        },
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
                },
                hir::FieldDef {
                    name: "pos".to_owned(),
                    ty: hir::Type::u(64),
                },
            ],
        },
    });
    let root_def = module.add_type_def(hir::TypeDef {
        name: <ScalarArrayHolder>::SHAPE.type_identifier.to_owned(),
        generic_params: vec![],
        kind: hir::TypeDefKind::Struct {
            fields: vec![hir::FieldDef {
                name: "values".to_owned(),
                ty: hir::Type::array(hir::Type::u(32), 4),
            }],
        },
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
                },
                hir::FieldDef {
                    name: "pos".to_owned(),
                    ty: hir::Type::u(64),
                },
            ],
        },
    });
    let root_def = module.add_type_def(hir::TypeDef {
        name: <ScalarArrayHolder>::SHAPE.type_identifier.to_owned(),
        generic_params: vec![],
        kind: hir::TypeDefKind::Struct {
            fields: vec![hir::FieldDef {
                name: "values".to_owned(),
                ty: hir::Type::array(hir::Type::u(32), 4),
            }],
        },
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
                },
                hir::FieldDef {
                    name: "pos".to_owned(),
                    ty: hir::Type::u(64),
                },
            ],
        },
    });
    let root_def = module.add_type_def(hir::TypeDef {
        name: <ScalarArrayHolder>::SHAPE.type_identifier.to_owned(),
        generic_params: vec![],
        kind: hir::TypeDefKind::Struct {
            fields: vec![hir::FieldDef {
                name: "values".to_owned(),
                ty: hir::Type::array(hir::Type::u(32), 4),
            }],
        },
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
                },
                hir::FieldDef {
                    name: "pos".to_owned(),
                    ty: hir::Type::u(64),
                },
            ],
        },
    });
    let root_def = module.add_type_def(hir::TypeDef {
        name: <ConstantNumber>::SHAPE.type_identifier.to_owned(),
        generic_params: vec![],
        kind: hir::TypeDefKind::Struct {
            fields: vec![hir::FieldDef {
                name: "value".to_owned(),
                ty: hir::Type::u(32),
            }],
        },
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
                },
                hir::FieldDef {
                    name: "len".to_owned(),
                    ty: hir::Type::u(64),
                },
            ],
        },
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
            }],
        },
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
            }],
        },
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
            }],
        },
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
            }],
        },
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
                },
                hir::VariantDef {
                    name: "Dog".to_owned(),
                    fields: vec![],
                },
                hir::VariantDef {
                    name: "Parrot".to_owned(),
                    fields: vec![],
                },
            ],
        },
    });
    let root_def = module.add_type_def(hir::TypeDef {
        name: <BranchyAnimal>::SHAPE.type_identifier.to_owned(),
        generic_params: vec![],
        kind: hir::TypeDefKind::Struct {
            fields: vec![
                hir::FieldDef {
                    name: "animal".to_owned(),
                    ty: hir::Type::named(animal_def, Vec::new()),
                },
                hir::FieldDef {
                    name: "value".to_owned(),
                    ty: hir::Type::u(32),
                },
            ],
        },
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
                },
                hir::FieldDef {
                    name: "shifted".to_owned(),
                    ty: hir::Type::u(32),
                },
                hir::FieldDef {
                    name: "toggled".to_owned(),
                    ty: hir::Type::u(32),
                },
                hir::FieldDef {
                    name: "combined".to_owned(),
                    ty: hir::Type::u(32),
                },
            ],
        },
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
                },
                hir::FieldDef {
                    name: "done".to_owned(),
                    ty: hir::Type::u(32),
                },
            ],
        },
    });
    let root_def = module.add_type_def(hir::TypeDef {
        name: <ScratchSummary>::SHAPE.type_identifier.to_owned(),
        generic_params: vec![],
        kind: hir::TypeDefKind::Struct {
            fields: vec![
                hir::FieldDef {
                    name: "mask".to_owned(),
                    ty: hir::Type::u(32),
                },
                hir::FieldDef {
                    name: "done".to_owned(),
                    ty: hir::Type::u(32),
                },
            ],
        },
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
            }],
        },
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
                },
                hir::FieldDef {
                    name: "selected".to_owned(),
                    ty: hir::Type::u(32),
                },
            ],
        },
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
                },
                hir::FieldDef {
                    name: "hi".to_owned(),
                    ty: hir::Type::u(64),
                },
            ],
        },
    });
    let root_def = module.add_type_def(hir::TypeDef {
        name: <DynamicAggregateSummary>::SHAPE.type_identifier.to_owned(),
        generic_params: vec![],
        kind: hir::TypeDefKind::Struct {
            fields: vec![hir::FieldDef {
                name: "pair".to_owned(),
                ty: hir::Type::named(pair_def, Vec::new()),
            }],
        },
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
                },
                hir::FieldDef {
                    name: "hi".to_owned(),
                    ty: hir::Type::u(64),
                },
            ],
        },
    });
    let root_def = module.add_type_def(hir::TypeDef {
        name: <DynamicAggregateSummary>::SHAPE.type_identifier.to_owned(),
        generic_params: vec![],
        kind: hir::TypeDefKind::Struct {
            fields: vec![hir::FieldDef {
                name: "pair".to_owned(),
                ty: hir::Type::named(pair_def, Vec::new()),
            }],
        },
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
                },
                hir::FieldDef {
                    name: "hi".to_owned(),
                    ty: hir::Type::u(64),
                },
            ],
        },
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
                },
                hir::FieldDef {
                    name: "selected".to_owned(),
                    ty: hir::Type::named(pair_def, Vec::new()),
                },
            ],
        },
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

#[test]
fn postcard_structural_hir_ir_path_decodes_scalar_arrays() {
    let decoder = compile_postcard_decoder_via_structural_hir(<ScalarArrayHolder>::SHAPE);

    let value = crate::deserialize::<ScalarArrayHolder>(&decoder, &[1, 2, 3, 4])
        .expect("structural HIR postcard decoder should decode scalar arrays");
    assert_eq!(
        value,
        ScalarArrayHolder {
            values: [1, 2, 3, 4],
        }
    );
}

#[test]
fn postcard_structural_hir_ir_path_decodes_borrowed_arrays() {
    let decoder =
        compile_postcard_decoder_via_structural_hir(<BorrowedArrayHolder<'static>>::SHAPE);

    let value =
        crate::deserialize::<BorrowedArrayHolder<'_>>(&decoder, &[2, b'h', b'i', 2, b'o', b'k'])
            .expect("structural HIR postcard decoder should decode borrowed arrays");
    assert_eq!(
        value,
        BorrowedArrayHolder {
            values: ["hi", "ok"],
        }
    );
}

#[test]
fn postcard_structural_hir_ir_path_decodes_payload_enums() {
    let decoder = compile_postcard_decoder_via_structural_hir(<PayloadAnimal<'static>>::SHAPE);

    let cat = crate::deserialize::<PayloadAnimal<'_>>(&decoder, &[0])
        .expect("structural HIR postcard decoder should decode unit enum variant");
    assert_eq!(cat, PayloadAnimal::Cat);

    let count = crate::deserialize::<PayloadAnimal<'_>>(&decoder, &[1, 42])
        .expect("structural HIR postcard decoder should decode scalar payload enum variant");
    assert_eq!(count, PayloadAnimal::Count(42));

    let name = crate::deserialize::<PayloadAnimal<'_>>(&decoder, &[2, 2, b'h', b'i'])
        .expect("structural HIR postcard decoder should decode borrowed payload enum variant");
    assert_eq!(name, PayloadAnimal::Name("hi"));
}

#[test]
fn postcard_structural_hir_ir_path_decodes_enum_in_struct_field() {
    let decoder = compile_postcard_decoder_via_structural_hir(<OwnedZoo>::SHAPE);

    let value = crate::deserialize::<OwnedZoo>(
        &decoder,
        &[
            8, b'C', b'i', b't', b'y', b' ', b'Z', b'o', b'o', // zoo name
            1,    // Dog discriminant
            3, b'R', b'e', b'x', // dog name
            1,    // good_boy
        ],
    )
    .expect("structural HIR postcard decoder should decode nested enum payloads");
    assert_eq!(
        value,
        OwnedZoo {
            name: "City Zoo".to_owned(),
            star: OwnedAnimal::Dog {
                name: "Rex".to_owned(),
                good_boy: true,
            },
        }
    );
}

#[test]
fn postcard_structural_hir_array_path_matches_jit_differential_harness() {
    let mut func = build_postcard_decoder_ir_via_hir(<ScalarArrayHolder>::SHAPE);
    let linear = crate::linearize::linearize(&mut func);
    let report = crate::differential_check_linear_ir_vs_jit(&linear, &[1, 2, 3, 4])
        .expect("differential harness should execute structural HIR postcard array decoder");
    assert!(
        report.is_match(),
        "unexpected differential mismatch: {:?}",
        report.mismatch
    );
}

#[test]
fn postcard_structural_hir_array_path_matches_post_regalloc_simulation() {
    let mut func = build_postcard_decoder_ir_via_hir(<ScalarArrayHolder>::SHAPE);
    let linear = crate::linearize::linearize(&mut func);
    let cfg = crate::regalloc_engine::cfg_mir::lower_linear_ir(&linear);
    let alloc = crate::regalloc_engine::allocate_cfg_program(&cfg)
        .expect("regalloc should allocate structural HIR postcard array cfg");
    let result = crate::regalloc_engine::differential_check_cfg(&cfg, &alloc, &[1, 2, 3, 4]);
    assert!(
        matches!(
            result,
            crate::regalloc_engine::DifferentialCheckResult::Match { .. }
        ),
        "unexpected interpreter/post-regalloc mismatch: {result:?}"
    );
}

#[test]
fn postcard_structural_hir_array_path_without_backend_edit_emission() {
    let mut func = build_postcard_decoder_ir_via_hir(<ScalarArrayHolder>::SHAPE);
    let linear = crate::linearize::linearize(&mut func);
    let cfg = crate::regalloc_engine::cfg_mir::lower_linear_ir(&linear);
    let alloc = crate::regalloc_engine::allocate_cfg_program(&cfg)
        .expect("regalloc should allocate structural HIR postcard array cfg");
    let result =
        crate::ir_backend::compile_linear_ir_with_alloc_and_mode(&linear, &cfg, &alloc, false);
    let (buf, entry, _source_map, _backend_debug_info) = materialize_backend_result(result);
    let func: unsafe extern "C" fn(*mut u8, *mut crate::context::DeserContext) =
        unsafe { core::mem::transmute(buf.code_ptr().add(entry)) };
    let decoder = CompiledDecoder {
        buf,
        cfg_mir_line_text_by_line: Default::default(),
        entry,
        func,
        trusted_utf8_input: false,
        _jit_registration: None,
    };

    let value = crate::deserialize::<ScalarArrayHolder>(&decoder, &[1, 2, 3, 4])
        .expect("structural HIR postcard array decoder should execute without backend edits");
    assert_eq!(
        value,
        ScalarArrayHolder {
            values: [1, 2, 3, 4]
        }
    );
}

#[test]
fn debug_scalar_array_regalloc_edits() {
    let mut func = build_postcard_decoder_ir_via_hir(<ScalarArrayHolder>::SHAPE);
    let linear = crate::linearize::linearize(&mut func);
    let cfg = crate::regalloc_engine::cfg_mir::lower_linear_ir(&linear);
    let alloc = crate::regalloc_engine::allocate_cfg_program(&cfg)
        .expect("regalloc should allocate structural HIR postcard array cfg");
    println!("{}", format_allocated_regalloc_edits(&alloc));
}

#[test]
fn debug_scalar_array_emission_trace() {
    let decoder = compile_postcard_decoder_via_structural_hir(<ScalarArrayHolder>::SHAPE);
    println!(
        "{}",
        decoder
            .emission_trace_text()
            .expect("emission trace should render")
    );
}

#[test]
#[ignore = "json struct HIR lowering not implemented yet"]
fn json_bool_true_false_matches_post_regalloc_simulation() {
    #[derive(Debug, PartialEq, Eq, Facet, serde::Serialize, serde::Deserialize)]
    struct Bools {
        a: bool,
        b: bool,
    }

    let value = Bools { a: true, b: false };
    let input = serde_json::to_vec(&value).expect("serialize json input");
    let mut func = build_decoder_ir_via_hir(<Bools>::SHAPE, &crate::json::KajitJson);
    run_default_passes_from_env(&mut func);
    let linear = crate::linearize::linearize(&mut func);
    let cfg = crate::regalloc_engine::cfg_mir::lower_linear_ir(&linear);
    let alloc = crate::regalloc_engine::allocate_cfg_program(&cfg)
        .expect("regalloc should allocate json bool cfg");
    let result = crate::regalloc_engine::differential_check_cfg(&cfg, &alloc, &input);
    assert!(
        matches!(
            result,
            crate::regalloc_engine::DifferentialCheckResult::Match { .. }
        ),
        "unexpected interpreter/post-regalloc mismatch: {result:?}"
    );
}

#[test]
#[ignore = "json struct HIR lowering not implemented yet"]
fn json_bool_true_false_without_backend_edit_emission() {
    #[derive(Debug, PartialEq, Eq, Facet, serde::Serialize, serde::Deserialize)]
    struct Bools {
        a: bool,
        b: bool,
    }

    let value = Bools { a: true, b: false };
    let input = serde_json::to_vec(&value).expect("serialize json input");
    let expected: Bools = serde_json::from_slice(&input).expect("decode reference json");
    let mut func = build_decoder_ir_via_hir(<Bools>::SHAPE, &crate::json::KajitJson);
    run_default_passes_from_env(&mut func);
    let linear = crate::linearize::linearize(&mut func);
    let cfg = crate::regalloc_engine::cfg_mir::lower_linear_ir(&linear);
    let alloc = crate::regalloc_engine::allocate_cfg_program(&cfg)
        .expect("regalloc should allocate json bool cfg");
    let result =
        crate::ir_backend::compile_linear_ir_with_alloc_and_mode(&linear, &cfg, &alloc, false);
    let (buf, entry, _source_map, _backend_debug_info) = materialize_backend_result(result);
    let func: unsafe extern "C" fn(*mut u8, *mut crate::context::DeserContext) =
        unsafe { core::mem::transmute(buf.code_ptr().add(entry)) };
    let decoder = CompiledDecoder {
        buf,
        cfg_mir_line_text_by_line: Default::default(),
        entry,
        func,
        trusted_utf8_input: false,
        _jit_registration: None,
    };

    let got = crate::from_str::<Bools>(&decoder, core::str::from_utf8(&input).unwrap())
        .expect("json bool decoder should execute without backend edits");
    assert_eq!(got, expected);
}

#[test]
#[ignore = "non-HIR path disabled"]
fn json_bool_true_false_with_backend_edit_emission() {
    #[derive(Debug, PartialEq, Eq, Facet, serde::Serialize, serde::Deserialize)]
    struct Bools {
        a: bool,
        b: bool,
    }

    let value = Bools { a: true, b: false };
    let input = serde_json::to_vec(&value).expect("serialize json input");
    let expected: Bools = serde_json::from_slice(&input).expect("decode reference json");
    let decoder = crate::compile_decoder(<Bools>::SHAPE, &crate::json::KajitJson);
    let got = crate::from_str::<Bools>(&decoder, core::str::from_utf8(&input).unwrap())
        .expect("json bool decoder should execute with backend edits");
    assert_eq!(got, expected);
    assert!(
        crate::regalloc_edit_count(<Bools>::SHAPE, &crate::json::KajitJson) > 0,
        "expected this regression test to exercise backend edit emission"
    );
}

#[test]
fn builds_dwarf_sections_from_source_map_lines() {
    let source_map = vec![
        kajit_emit::SourceMapEntry {
            offset: 0,
            location: kajit_emit::SourceLocation {
                file: 0,
                line: 1,
                column: 1,
            },
        },
        kajit_emit::SourceMapEntry {
            offset: 8,
            location: kajit_emit::SourceLocation {
                file: 0,
                line: 2,
                column: 1,
            },
        },
    ];

    let listing_path = std::env::temp_dir()
        .join(format!("kajit-debug-test-{}", std::process::id()))
        .join("sample.cfg-mir");
    std::fs::create_dir_all(listing_path.parent().expect("temp listing dir")).unwrap();
    std::fs::write(&listing_path, "inst0\ninst1\n").unwrap();

    let debug_info = build_jit_debug_info_from_source_map(
        0x1000 as *const u8,
        32,
        Some(&source_map),
        &listing_path,
        crate::jit_dwarf::JitDebugSubprogram {
            name: "kajit::decode::test".to_string(),
            frame_base_expression: crate::jit_dwarf::expr_breg(
                crate::jit_dwarf::frame_base_register(jit_dwarf_target_arch()),
                0,
            ),
            variables: Vec::new(),
            lexical_blocks: Vec::new(),
        },
    )
    .expect("expected debug info");
    let dwarf = crate::jit_dwarf::build_jit_dwarf_sections_from_debug_info(&debug_info)
        .expect("expected dwarf sections");
    assert!(!dwarf.debug_line.is_empty());
}

#[test]
fn deser_dwarf_variables_cover_fixed_runtime_state() {
    let vars = deser_dwarf_variables(jit_dwarf_target_arch());
    let names = vars.iter().map(|var| var.name.as_str()).collect::<Vec<_>>();
    assert_eq!(
        names,
        vec![
            "input_ptr",
            "input_end",
            "out_ptr",
            "ctx",
            "error_code",
            "error_offset",
        ]
    );
    for var in vars {
        match var.location {
            crate::jit_dwarf::DwarfVariableLocation::Expr(expr) => {
                assert!(!expr.is_empty());
            }
            crate::jit_dwarf::DwarfVariableLocation::List(_) => {
                panic!("deserializer runtime-state vars should use inline exprloc")
            }
        }
    }
}

#[test]
fn cfg_value_dwarf_variables_cover_def_vregs() {
    let v0 = crate::ir::VReg::new(0);
    let inst_id = crate::regalloc_engine::cfg_mir::InstId::new(0);
    let inst_id_2 = crate::regalloc_engine::cfg_mir::InstId::new(1);
    let term_id = crate::regalloc_engine::cfg_mir::TermId::new(0);
    let block_id = crate::regalloc_engine::cfg_mir::BlockId::new(0);
    let func = crate::regalloc_engine::cfg_mir::Function {
        id: crate::regalloc_engine::cfg_mir::FunctionId::new(0),
        lambda_id: crate::ir::LambdaId::new(0),
        entry: block_id,
        data_args: Vec::new(),
        data_results: Vec::new(),
        blocks: vec![crate::regalloc_engine::cfg_mir::Block {
            id: block_id,
            params: Vec::new(),
            insts: vec![inst_id, inst_id_2],
            term: term_id,
            preds: Vec::new(),
            succs: Vec::new(),
        }],
        edges: Vec::new(),
        insts: vec![
            crate::regalloc_engine::cfg_mir::Inst {
                id: inst_id,
                op: crate::linearize::LinearOp::Const { dst: v0, value: 7 },
                operands: vec![crate::regalloc_engine::cfg_mir::Operand {
                    vreg: v0,
                    kind: crate::regalloc_engine::cfg_mir::OperandKind::Def,
                    class: crate::regalloc_engine::cfg_mir::RegClass::Gpr,
                    fixed: None,
                }],
                clobbers: crate::regalloc_engine::cfg_mir::Clobbers::default(),
            },
            crate::regalloc_engine::cfg_mir::Inst {
                id: inst_id_2,
                op: crate::linearize::LinearOp::Copy {
                    dst: crate::ir::VReg::new(1),
                    src: v0,
                },
                operands: vec![
                    crate::regalloc_engine::cfg_mir::Operand {
                        vreg: v0,
                        kind: crate::regalloc_engine::cfg_mir::OperandKind::Use,
                        class: crate::regalloc_engine::cfg_mir::RegClass::Gpr,
                        fixed: None,
                    },
                    crate::regalloc_engine::cfg_mir::Operand {
                        vreg: crate::ir::VReg::new(1),
                        kind: crate::regalloc_engine::cfg_mir::OperandKind::Def,
                        class: crate::regalloc_engine::cfg_mir::RegClass::Gpr,
                        fixed: None,
                    },
                ],
                clobbers: crate::regalloc_engine::cfg_mir::Clobbers::default(),
            },
        ],
        terms: vec![crate::regalloc_engine::cfg_mir::Terminator::Return],
    };
    let root_scope = crate::ir::DebugScopeId::new(0);
    let block_scope = crate::ir::DebugScopeId::new(1);
    let mut scopes = crate::ir::Arena::new();
    scopes.push(crate::ir::DebugScope {
        parent: None,
        kind: crate::ir::DebugScopeKind::LambdaBody {
            lambda_id: crate::ir::LambdaId::new(0),
        },
    });
    scopes.push(crate::ir::DebugScope {
        parent: Some(root_scope),
        kind: crate::ir::DebugScopeKind::ThetaBody,
    });
    let op_id = crate::regalloc_engine::cfg_mir::OpId::Inst(inst_id);
    let op_id_2 = crate::regalloc_engine::cfg_mir::OpId::Inst(inst_id_2);
    let program = crate::regalloc_engine::cfg_mir::Program {
        funcs: vec![func],
        vreg_count: 2,
        slot_count: 0,
        debug: crate::regalloc_engine::cfg_mir::ProgramDebugProvenance {
            scopes,
            values: crate::ir::Arena::new(),
            root_scope: Some(root_scope),
            op_scopes: std::collections::HashMap::from([
                ((crate::ir::LambdaId::new(0), op_id), block_scope),
                ((crate::ir::LambdaId::new(0), op_id_2), block_scope),
                (
                    (
                        crate::ir::LambdaId::new(0),
                        crate::regalloc_engine::cfg_mir::OpId::Term(term_id),
                    ),
                    block_scope,
                ),
            ]),
            op_values: std::collections::HashMap::new(),
            vreg_scopes: vec![Some(block_scope), Some(root_scope)],
            vreg_values: vec![None, None],
        },
    };
    #[cfg(target_arch = "aarch64")]
    let reg = regalloc2::PReg::new(19, regalloc2::RegClass::Int);
    #[cfg(target_arch = "x86_64")]
    let reg = regalloc2::PReg::new(12, regalloc2::RegClass::Int);
    #[cfg(target_arch = "aarch64")]
    let reg_2 = regalloc2::PReg::new(20, regalloc2::RegClass::Int);
    #[cfg(target_arch = "x86_64")]
    let reg_2 = regalloc2::PReg::new(13, regalloc2::RegClass::Int);
    let alloc = crate::regalloc_engine::AllocatedCfgProgram {
        cfg_program: program.clone(),
        functions: vec![crate::regalloc_engine::AllocatedCfgFunction {
            lambda_id: crate::ir::LambdaId::new(0),
            num_spillslots: 0,
            edits: Vec::new(),
            op_allocs: std::collections::HashMap::from([
                (op_id, vec![regalloc2::Allocation::reg(reg)]),
                (
                    op_id_2,
                    vec![
                        regalloc2::Allocation::reg(reg),
                        regalloc2::Allocation::reg(reg_2),
                    ],
                ),
            ]),
            op_operands: std::collections::HashMap::from([
                (
                    op_id,
                    vec![(v0, crate::regalloc_engine::cfg_mir::OperandKind::Def)],
                ),
                (
                    op_id_2,
                    vec![
                        (v0, crate::regalloc_engine::cfg_mir::OperandKind::Use),
                        (
                            crate::ir::VReg::new(1),
                            crate::regalloc_engine::cfg_mir::OperandKind::Def,
                        ),
                    ],
                ),
            ]),
            edge_edits: Vec::new(),
            return_result_allocs: Vec::new(),
        }],
    };
    let backend_debug_info = crate::ir_backend::BackendDebugInfo {
        op_infos: vec![
            crate::ir_backend::BackendOpDebugInfo {
                lambda_id: 0,
                op_id,
                line: 1,
                code_ranges: vec![crate::ir_backend::BackendCodeRange {
                    start_offset: 0,
                    end_offset: 4,
                }],
            },
            crate::ir_backend::BackendOpDebugInfo {
                lambda_id: 0,
                op_id: op_id_2,
                line: 2,
                code_ranges: vec![crate::ir_backend::BackendCodeRange {
                    start_offset: 4,
                    end_offset: 8,
                }],
            },
            crate::ir_backend::BackendOpDebugInfo {
                lambda_id: 0,
                op_id: crate::regalloc_engine::cfg_mir::OpId::Term(term_id),
                line: 3,
                code_ranges: vec![crate::ir_backend::BackendCodeRange {
                    start_offset: 8,
                    end_offset: 12,
                }],
            },
        ],
    };

    let vars = cfg_value_dwarf_variables(
        &program,
        &alloc,
        Some(&backend_debug_info),
        0x1000 as *const u8,
        jit_dwarf_target_arch(),
        true,
        false,
    );

    assert_eq!(vars.len(), 1);
    assert_eq!(vars[0].variable.name, "v0");
    assert_eq!(
        vars[0].lexical_ranges,
        vec![crate::jit_dwarf::JitDebugRange {
            low_pc: 0x1000,
            high_pc: 0x1008,
        }]
    );
    match &vars[0].variable.location {
        crate::jit_dwarf::DwarfVariableLocation::List(locations) => {
            assert_eq!(locations.len(), 1);
            assert_eq!(locations[0].start, 0x1004);
            assert_eq!(locations[0].end, 0x1008);
            let dwarf_reg = crate::jit_dwarf::dwarf_register_from_hw_encoding(
                jit_dwarf_target_arch(),
                reg.hw_enc() as u8,
            )
            .unwrap();
            assert_eq!(
                locations[0].expression,
                crate::jit_dwarf::expr_reg(dwarf_reg)
            );
        }
        crate::jit_dwarf::DwarfVariableLocation::Expr(_) => {
            panic!("cfg def vregs should use ranged locations")
        }
    }
}

#[test]
fn cfg_value_dwarf_variables_keep_edge_carried_defs_live() {
    let v0 = crate::ir::VReg::new(0);
    let v1 = crate::ir::VReg::new(1);
    let inst_id = crate::regalloc_engine::cfg_mir::InstId::new(0);
    let inst_id_2 = crate::regalloc_engine::cfg_mir::InstId::new(1);
    let term_id = crate::regalloc_engine::cfg_mir::TermId::new(0);
    let return_term_id = crate::regalloc_engine::cfg_mir::TermId::new(1);
    let entry_block_id = crate::regalloc_engine::cfg_mir::BlockId::new(0);
    let exit_block_id = crate::regalloc_engine::cfg_mir::BlockId::new(1);
    let edge_id = crate::regalloc_engine::cfg_mir::EdgeId::new(0);
    let func = crate::regalloc_engine::cfg_mir::Function {
        id: crate::regalloc_engine::cfg_mir::FunctionId::new(0),
        lambda_id: crate::ir::LambdaId::new(0),
        entry: entry_block_id,
        data_args: Vec::new(),
        data_results: Vec::new(),
        blocks: vec![
            crate::regalloc_engine::cfg_mir::Block {
                id: entry_block_id,
                params: Vec::new(),
                insts: vec![inst_id, inst_id_2],
                term: term_id,
                preds: Vec::new(),
                succs: vec![edge_id],
            },
            crate::regalloc_engine::cfg_mir::Block {
                id: exit_block_id,
                params: vec![v0],
                insts: Vec::new(),
                term: return_term_id,
                preds: vec![edge_id],
                succs: Vec::new(),
            },
        ],
        edges: vec![crate::regalloc_engine::cfg_mir::Edge {
            id: edge_id,
            from: entry_block_id,
            to: exit_block_id,
            args: vec![crate::regalloc_engine::cfg_mir::EdgeArg {
                target: v0,
                source: v0,
            }],
        }],
        insts: vec![
            crate::regalloc_engine::cfg_mir::Inst {
                id: inst_id,
                op: crate::linearize::LinearOp::Const { dst: v0, value: 7 },
                operands: vec![crate::regalloc_engine::cfg_mir::Operand {
                    vreg: v0,
                    kind: crate::regalloc_engine::cfg_mir::OperandKind::Def,
                    class: crate::regalloc_engine::cfg_mir::RegClass::Gpr,
                    fixed: None,
                }],
                clobbers: crate::regalloc_engine::cfg_mir::Clobbers::default(),
            },
            crate::regalloc_engine::cfg_mir::Inst {
                id: inst_id_2,
                op: crate::linearize::LinearOp::Const { dst: v1, value: 9 },
                operands: vec![crate::regalloc_engine::cfg_mir::Operand {
                    vreg: v1,
                    kind: crate::regalloc_engine::cfg_mir::OperandKind::Def,
                    class: crate::regalloc_engine::cfg_mir::RegClass::Gpr,
                    fixed: None,
                }],
                clobbers: crate::regalloc_engine::cfg_mir::Clobbers::default(),
            },
        ],
        terms: vec![
            crate::regalloc_engine::cfg_mir::Terminator::Branch { edge: edge_id },
            crate::regalloc_engine::cfg_mir::Terminator::Return,
        ],
    };
    let op_id = crate::regalloc_engine::cfg_mir::OpId::Inst(inst_id);
    let op_id_2 = crate::regalloc_engine::cfg_mir::OpId::Inst(inst_id_2);
    let root_scope = crate::ir::DebugScopeId::new(0);
    let block_scope = crate::ir::DebugScopeId::new(1);
    let mut scopes = crate::ir::Arena::new();
    scopes.push(crate::ir::DebugScope {
        parent: None,
        kind: crate::ir::DebugScopeKind::LambdaBody {
            lambda_id: crate::ir::LambdaId::new(0),
        },
    });
    scopes.push(crate::ir::DebugScope {
        parent: Some(root_scope),
        kind: crate::ir::DebugScopeKind::ThetaBody,
    });
    let program = crate::regalloc_engine::cfg_mir::Program {
        funcs: vec![func],
        vreg_count: 2,
        slot_count: 0,
        debug: crate::regalloc_engine::cfg_mir::ProgramDebugProvenance {
            scopes,
            values: crate::ir::Arena::new(),
            root_scope: Some(root_scope),
            op_scopes: std::collections::HashMap::from([
                ((crate::ir::LambdaId::new(0), op_id), block_scope),
                ((crate::ir::LambdaId::new(0), op_id_2), block_scope),
                (
                    (
                        crate::ir::LambdaId::new(0),
                        crate::regalloc_engine::cfg_mir::OpId::Term(term_id),
                    ),
                    block_scope,
                ),
            ]),
            op_values: std::collections::HashMap::new(),
            vreg_scopes: vec![Some(block_scope), Some(root_scope)],
            vreg_values: vec![None, None],
        },
    };
    let term_op = crate::regalloc_engine::cfg_mir::OpId::Term(term_id);
    #[cfg(target_arch = "aarch64")]
    let reg = regalloc2::PReg::new(19, regalloc2::RegClass::Int);
    #[cfg(target_arch = "x86_64")]
    let reg = regalloc2::PReg::new(12, regalloc2::RegClass::Int);
    #[cfg(target_arch = "aarch64")]
    let reg_2 = regalloc2::PReg::new(20, regalloc2::RegClass::Int);
    #[cfg(target_arch = "x86_64")]
    let reg_2 = regalloc2::PReg::new(13, regalloc2::RegClass::Int);
    let alloc = crate::regalloc_engine::AllocatedCfgProgram {
        cfg_program: program.clone(),
        functions: vec![crate::regalloc_engine::AllocatedCfgFunction {
            lambda_id: crate::ir::LambdaId::new(0),
            num_spillslots: 0,
            edits: Vec::new(),
            op_allocs: std::collections::HashMap::from([
                (op_id, vec![regalloc2::Allocation::reg(reg)]),
                (op_id_2, vec![regalloc2::Allocation::reg(reg_2)]),
            ]),
            op_operands: std::collections::HashMap::from([
                (
                    op_id,
                    vec![(v0, crate::regalloc_engine::cfg_mir::OperandKind::Def)],
                ),
                (
                    op_id_2,
                    vec![(v1, crate::regalloc_engine::cfg_mir::OperandKind::Def)],
                ),
                (term_op, Vec::new()),
            ]),
            edge_edits: Vec::new(),
            return_result_allocs: Vec::new(),
        }],
    };
    let backend_debug_info = crate::ir_backend::BackendDebugInfo {
        op_infos: vec![
            crate::ir_backend::BackendOpDebugInfo {
                lambda_id: 0,
                op_id,
                line: 1,
                code_ranges: vec![crate::ir_backend::BackendCodeRange {
                    start_offset: 0,
                    end_offset: 4,
                }],
            },
            crate::ir_backend::BackendOpDebugInfo {
                lambda_id: 0,
                op_id: op_id_2,
                line: 2,
                code_ranges: vec![crate::ir_backend::BackendCodeRange {
                    start_offset: 4,
                    end_offset: 8,
                }],
            },
            crate::ir_backend::BackendOpDebugInfo {
                lambda_id: 0,
                op_id: term_op,
                line: 3,
                code_ranges: vec![crate::ir_backend::BackendCodeRange {
                    start_offset: 8,
                    end_offset: 12,
                }],
            },
        ],
    };

    let vars = cfg_value_dwarf_variables(
        &program,
        &alloc,
        Some(&backend_debug_info),
        0x1000 as *const u8,
        jit_dwarf_target_arch(),
        true,
        false,
    );

    assert_eq!(vars.len(), 1);
    assert_eq!(vars[0].variable.name, "v0");
    assert_eq!(
        vars[0].lexical_ranges,
        vec![crate::jit_dwarf::JitDebugRange {
            low_pc: 0x1000,
            high_pc: 0x100c,
        }]
    );
    match &vars[0].variable.location {
        crate::jit_dwarf::DwarfVariableLocation::List(locations) => {
            assert_eq!(locations.len(), 1);
            assert_eq!(locations[0].start, 0x1004);
            assert_eq!(locations[0].end, 0x100c);
            let dwarf_reg = crate::jit_dwarf::dwarf_register_from_hw_encoding(
                jit_dwarf_target_arch(),
                reg.hw_enc() as u8,
            )
            .unwrap();
            assert_eq!(
                locations[0].expression,
                crate::jit_dwarf::expr_reg(dwarf_reg)
            );
        }
        crate::jit_dwarf::DwarfVariableLocation::Expr(_) => {
            panic!("cfg edge-carried vregs should use ranged locations")
        }
    }
}

#[test]
fn cfg_mir_dwarf_variables_place_block_local_vregs_in_lexical_blocks() {
    let v0 = crate::ir::VReg::new(0);
    let inst_id = crate::regalloc_engine::cfg_mir::InstId::new(0);
    let inst_id_2 = crate::regalloc_engine::cfg_mir::InstId::new(1);
    let term_id = crate::regalloc_engine::cfg_mir::TermId::new(0);
    let block_id = crate::regalloc_engine::cfg_mir::BlockId::new(0);
    let func = crate::regalloc_engine::cfg_mir::Function {
        id: crate::regalloc_engine::cfg_mir::FunctionId::new(0),
        lambda_id: crate::ir::LambdaId::new(0),
        entry: block_id,
        data_args: Vec::new(),
        data_results: Vec::new(),
        blocks: vec![crate::regalloc_engine::cfg_mir::Block {
            id: block_id,
            params: Vec::new(),
            insts: vec![inst_id, inst_id_2],
            term: term_id,
            preds: Vec::new(),
            succs: Vec::new(),
        }],
        edges: Vec::new(),
        insts: vec![
            crate::regalloc_engine::cfg_mir::Inst {
                id: inst_id,
                op: crate::linearize::LinearOp::Const { dst: v0, value: 7 },
                operands: vec![crate::regalloc_engine::cfg_mir::Operand {
                    vreg: v0,
                    kind: crate::regalloc_engine::cfg_mir::OperandKind::Def,
                    class: crate::regalloc_engine::cfg_mir::RegClass::Gpr,
                    fixed: None,
                }],
                clobbers: crate::regalloc_engine::cfg_mir::Clobbers::default(),
            },
            crate::regalloc_engine::cfg_mir::Inst {
                id: inst_id_2,
                op: crate::linearize::LinearOp::Copy {
                    dst: crate::ir::VReg::new(1),
                    src: v0,
                },
                operands: vec![
                    crate::regalloc_engine::cfg_mir::Operand {
                        vreg: v0,
                        kind: crate::regalloc_engine::cfg_mir::OperandKind::Use,
                        class: crate::regalloc_engine::cfg_mir::RegClass::Gpr,
                        fixed: None,
                    },
                    crate::regalloc_engine::cfg_mir::Operand {
                        vreg: crate::ir::VReg::new(1),
                        kind: crate::regalloc_engine::cfg_mir::OperandKind::Def,
                        class: crate::regalloc_engine::cfg_mir::RegClass::Gpr,
                        fixed: None,
                    },
                ],
                clobbers: crate::regalloc_engine::cfg_mir::Clobbers::default(),
            },
        ],
        terms: vec![crate::regalloc_engine::cfg_mir::Terminator::Return],
    };
    let op_id = crate::regalloc_engine::cfg_mir::OpId::Inst(inst_id);
    let op_id_2 = crate::regalloc_engine::cfg_mir::OpId::Inst(inst_id_2);
    let root_scope = crate::ir::DebugScopeId::new(0);
    let block_scope = crate::ir::DebugScopeId::new(1);
    let mut scopes = crate::ir::Arena::new();
    scopes.push(crate::ir::DebugScope {
        parent: None,
        kind: crate::ir::DebugScopeKind::LambdaBody {
            lambda_id: crate::ir::LambdaId::new(0),
        },
    });
    scopes.push(crate::ir::DebugScope {
        parent: Some(root_scope),
        kind: crate::ir::DebugScopeKind::ThetaBody,
    });
    let program = crate::regalloc_engine::cfg_mir::Program {
        funcs: vec![func],
        vreg_count: 2,
        slot_count: 0,
        debug: crate::regalloc_engine::cfg_mir::ProgramDebugProvenance {
            scopes,
            values: crate::ir::Arena::new(),
            root_scope: Some(root_scope),
            op_scopes: std::collections::HashMap::from([
                ((crate::ir::LambdaId::new(0), op_id), block_scope),
                ((crate::ir::LambdaId::new(0), op_id_2), block_scope),
                (
                    (
                        crate::ir::LambdaId::new(0),
                        crate::regalloc_engine::cfg_mir::OpId::Term(term_id),
                    ),
                    block_scope,
                ),
            ]),
            op_values: std::collections::HashMap::new(),
            vreg_scopes: vec![Some(block_scope), Some(root_scope)],
            vreg_values: vec![None, None],
        },
    };
    #[cfg(target_arch = "aarch64")]
    let reg = regalloc2::PReg::new(19, regalloc2::RegClass::Int);
    #[cfg(target_arch = "x86_64")]
    let reg = regalloc2::PReg::new(12, regalloc2::RegClass::Int);
    #[cfg(target_arch = "aarch64")]
    let reg_2 = regalloc2::PReg::new(20, regalloc2::RegClass::Int);
    #[cfg(target_arch = "x86_64")]
    let reg_2 = regalloc2::PReg::new(13, regalloc2::RegClass::Int);
    let alloc = crate::regalloc_engine::AllocatedCfgProgram {
        cfg_program: program.clone(),
        functions: vec![crate::regalloc_engine::AllocatedCfgFunction {
            lambda_id: crate::ir::LambdaId::new(0),
            num_spillslots: 0,
            edits: Vec::new(),
            op_allocs: std::collections::HashMap::from([
                (op_id, vec![regalloc2::Allocation::reg(reg)]),
                (
                    op_id_2,
                    vec![
                        regalloc2::Allocation::reg(reg),
                        regalloc2::Allocation::reg(reg_2),
                    ],
                ),
            ]),
            op_operands: std::collections::HashMap::from([
                (
                    op_id,
                    vec![(v0, crate::regalloc_engine::cfg_mir::OperandKind::Def)],
                ),
                (
                    op_id_2,
                    vec![
                        (v0, crate::regalloc_engine::cfg_mir::OperandKind::Use),
                        (
                            crate::ir::VReg::new(1),
                            crate::regalloc_engine::cfg_mir::OperandKind::Def,
                        ),
                    ],
                ),
            ]),
            edge_edits: Vec::new(),
            return_result_allocs: Vec::new(),
        }],
    };
    let backend_debug_info = crate::ir_backend::BackendDebugInfo {
        op_infos: vec![
            crate::ir_backend::BackendOpDebugInfo {
                lambda_id: 0,
                op_id,
                line: 1,
                code_ranges: vec![crate::ir_backend::BackendCodeRange {
                    start_offset: 0,
                    end_offset: 4,
                }],
            },
            crate::ir_backend::BackendOpDebugInfo {
                lambda_id: 0,
                op_id: op_id_2,
                line: 2,
                code_ranges: vec![crate::ir_backend::BackendCodeRange {
                    start_offset: 4,
                    end_offset: 8,
                }],
            },
            crate::ir_backend::BackendOpDebugInfo {
                lambda_id: 0,
                op_id: crate::regalloc_engine::cfg_mir::OpId::Term(term_id),
                line: 3,
                code_ranges: vec![crate::ir_backend::BackendCodeRange {
                    start_offset: 8,
                    end_offset: 12,
                }],
            },
        ],
    };

    let subprogram = cfg_mir_dwarf_variables(
        None,
        &program,
        &alloc,
        Some(&backend_debug_info),
        0x1000 as *const u8,
        jit_dwarf_target_arch(),
        true,
    );

    assert!(
        !subprogram
            .variables
            .iter()
            .any(|variable| variable.name == "v0")
    );
    assert_eq!(subprogram.lexical_blocks.len(), 1);
    assert_eq!(subprogram.lexical_blocks[0].ranges.len(), 1);
    assert_eq!(subprogram.lexical_blocks[0].ranges[0].low_pc, 0x1000);
    assert_eq!(subprogram.lexical_blocks[0].ranges[0].high_pc, 0x100c);
    assert!(subprogram.lexical_blocks[0].variables.is_empty());
    assert_eq!(subprogram.lexical_blocks[0].lexical_blocks.len(), 1);
    assert_eq!(
        subprogram.lexical_blocks[0].lexical_blocks[0].ranges.len(),
        1
    );
    assert_eq!(
        subprogram.lexical_blocks[0].lexical_blocks[0].ranges[0].low_pc,
        0x1000
    );
    assert_eq!(
        subprogram.lexical_blocks[0].lexical_blocks[0].ranges[0].high_pc,
        0x1008
    );
    assert_eq!(
        subprogram.lexical_blocks[0].lexical_blocks[0]
            .variables
            .len(),
        1
    );
    assert_eq!(
        subprogram.lexical_blocks[0].lexical_blocks[0].variables[0].name,
        "v0"
    );
}

#[test]
fn cfg_semantic_field_dwarf_variables_follow_field_debug_values() {
    #[derive(Facet)]
    struct Bools {
        a: bool,
        b: bool,
    }

    let inst_a = crate::regalloc_engine::cfg_mir::InstId::new(0);
    let inst_b = crate::regalloc_engine::cfg_mir::InstId::new(1);
    let term_id = crate::regalloc_engine::cfg_mir::TermId::new(0);
    let block_id = crate::regalloc_engine::cfg_mir::BlockId::new(0);
    let op_a = crate::regalloc_engine::cfg_mir::OpId::Inst(inst_a);
    let op_b = crate::regalloc_engine::cfg_mir::OpId::Inst(inst_b);
    let term_op = crate::regalloc_engine::cfg_mir::OpId::Term(term_id);
    let root_scope = crate::ir::DebugScopeId::new(0);

    let func = crate::regalloc_engine::cfg_mir::Function {
        id: crate::regalloc_engine::cfg_mir::FunctionId::new(0),
        lambda_id: crate::ir::LambdaId::new(0),
        entry: block_id,
        data_args: Vec::new(),
        data_results: Vec::new(),
        blocks: vec![crate::regalloc_engine::cfg_mir::Block {
            id: block_id,
            params: Vec::new(),
            insts: vec![inst_a, inst_b],
            term: term_id,
            preds: Vec::new(),
            succs: Vec::new(),
        }],
        edges: Vec::new(),
        insts: vec![
            crate::regalloc_engine::cfg_mir::Inst {
                id: inst_a,
                op: crate::linearize::LinearOp::CallIntrinsic {
                    func: crate::ir::IntrinsicFn(
                        crate::json_intrinsics::kajit_json_read_bool as *const () as usize,
                    ),
                    args: Vec::new(),
                    dst: None,
                    field_offset: 0,
                },
                operands: Vec::new(),
                clobbers: crate::regalloc_engine::cfg_mir::Clobbers::default(),
            },
            crate::regalloc_engine::cfg_mir::Inst {
                id: inst_b,
                op: crate::linearize::LinearOp::CallIntrinsic {
                    func: crate::ir::IntrinsicFn(
                        crate::json_intrinsics::kajit_json_read_bool as *const () as usize,
                    ),
                    args: Vec::new(),
                    dst: None,
                    field_offset: 1,
                },
                operands: Vec::new(),
                clobbers: crate::regalloc_engine::cfg_mir::Clobbers::default(),
            },
        ],
        terms: vec![crate::regalloc_engine::cfg_mir::Terminator::Return],
    };

    let mut scopes = crate::ir::Arena::new();
    scopes.push(crate::ir::DebugScope {
        parent: None,
        kind: crate::ir::DebugScopeKind::LambdaBody {
            lambda_id: crate::ir::LambdaId::new(0),
        },
    });
    let mut values = crate::ir::Arena::new();
    let debug_a = values.push(crate::ir::DebugValue {
        name: "a".to_string(),
        kind: crate::ir::DebugValueKind::Field { offset: 0 },
    });
    let debug_b = values.push(crate::ir::DebugValue {
        name: "b".to_string(),
        kind: crate::ir::DebugValueKind::Field { offset: 1 },
    });
    let program = crate::regalloc_engine::cfg_mir::Program {
        funcs: vec![func],
        vreg_count: 0,
        slot_count: 0,
        debug: crate::regalloc_engine::cfg_mir::ProgramDebugProvenance {
            scopes,
            values,
            root_scope: Some(root_scope),
            op_scopes: std::collections::HashMap::new(),
            op_values: std::collections::HashMap::from([
                ((crate::ir::LambdaId::new(0), op_a), debug_a),
                ((crate::ir::LambdaId::new(0), op_b), debug_b),
            ]),
            vreg_scopes: Vec::new(),
            vreg_values: Vec::new(),
        },
    };
    let backend_debug_info = crate::ir_backend::BackendDebugInfo {
        op_infos: vec![
            crate::ir_backend::BackendOpDebugInfo {
                lambda_id: 0,
                op_id: op_a,
                line: 10,
                code_ranges: vec![crate::ir_backend::BackendCodeRange {
                    start_offset: 0,
                    end_offset: 8,
                }],
            },
            crate::ir_backend::BackendOpDebugInfo {
                lambda_id: 0,
                op_id: op_b,
                line: 20,
                code_ranges: vec![crate::ir_backend::BackendCodeRange {
                    start_offset: 16,
                    end_offset: 24,
                }],
            },
            crate::ir_backend::BackendOpDebugInfo {
                lambda_id: 0,
                op_id: term_op,
                line: 30,
                code_ranges: vec![crate::ir_backend::BackendCodeRange {
                    start_offset: 32,
                    end_offset: 40,
                }],
            },
        ],
    };

    let vars = cfg_semantic_field_dwarf_variables(
        <Bools as Facet>::SHAPE,
        &program,
        Some(&backend_debug_info),
        0x1000 as *const u8,
        jit_dwarf_target_arch(),
    );

    assert_eq!(vars.len(), 2);
    assert_eq!(vars[0].scope, Some(root_scope));
    assert_eq!(vars[0].variable.name, "a");
    assert_eq!(
        vars[0].lexical_ranges,
        vec![crate::jit_dwarf::JitDebugRange {
            low_pc: 0x1000,
            high_pc: 0x1028,
        }]
    );
    assert_eq!(vars[1].scope, Some(root_scope));
    assert_eq!(vars[1].variable.name, "b");
    assert_eq!(
        vars[1].lexical_ranges,
        vec![crate::jit_dwarf::JitDebugRange {
            low_pc: 0x1010,
            high_pc: 0x1028,
        }]
    );

    let expected_expr_a = dwarf_expr_for_out_field(jit_dwarf_target_arch(), 0, 1);
    let expected_expr_b = dwarf_expr_for_out_field(jit_dwarf_target_arch(), 1, 1);
    match &vars[0].variable.location {
        crate::jit_dwarf::DwarfVariableLocation::List(locations) => {
            assert_eq!(locations.len(), 1);
            assert_eq!(locations[0].start, 0x1008);
            assert_eq!(locations[0].end, 0x1028);
            assert_eq!(locations[0].expression, expected_expr_a);
        }
        crate::jit_dwarf::DwarfVariableLocation::Expr(_) => {
            panic!("semantic field vars should use ranged locations")
        }
    }
    match &vars[1].variable.location {
        crate::jit_dwarf::DwarfVariableLocation::List(locations) => {
            assert_eq!(locations.len(), 1);
            assert_eq!(locations[0].start, 0x1018);
            assert_eq!(locations[0].end, 0x1028);
            assert_eq!(locations[0].expression, expected_expr_b);
        }
        crate::jit_dwarf::DwarfVariableLocation::Expr(_) => {
            panic!("semantic field vars should use ranged locations")
        }
    }
}

#[test]
fn cfg_value_dwarf_variables_can_hide_semantic_owned_vregs() {
    let v0 = crate::ir::VReg::new(0);
    let inst_id = crate::regalloc_engine::cfg_mir::InstId::new(0);
    let term_id = crate::regalloc_engine::cfg_mir::TermId::new(0);
    let block_id = crate::regalloc_engine::cfg_mir::BlockId::new(0);
    let op_id = crate::regalloc_engine::cfg_mir::OpId::Inst(inst_id);
    let term_op = crate::regalloc_engine::cfg_mir::OpId::Term(term_id);
    let root_scope = crate::ir::DebugScopeId::new(0);
    let func = crate::regalloc_engine::cfg_mir::Function {
        id: crate::regalloc_engine::cfg_mir::FunctionId::new(0),
        lambda_id: crate::ir::LambdaId::new(0),
        entry: block_id,
        data_args: Vec::new(),
        data_results: Vec::new(),
        blocks: vec![crate::regalloc_engine::cfg_mir::Block {
            id: block_id,
            params: Vec::new(),
            insts: vec![inst_id],
            term: term_id,
            preds: Vec::new(),
            succs: Vec::new(),
        }],
        edges: Vec::new(),
        insts: vec![crate::regalloc_engine::cfg_mir::Inst {
            id: inst_id,
            op: crate::linearize::LinearOp::Const { dst: v0, value: 1 },
            operands: vec![crate::regalloc_engine::cfg_mir::Operand {
                vreg: v0,
                kind: crate::regalloc_engine::cfg_mir::OperandKind::Def,
                class: crate::regalloc_engine::cfg_mir::RegClass::Gpr,
                fixed: None,
            }],
            clobbers: crate::regalloc_engine::cfg_mir::Clobbers::default(),
        }],
        terms: vec![crate::regalloc_engine::cfg_mir::Terminator::Return],
    };
    let mut scopes = crate::ir::Arena::new();
    scopes.push(crate::ir::DebugScope {
        parent: None,
        kind: crate::ir::DebugScopeKind::LambdaBody {
            lambda_id: crate::ir::LambdaId::new(0),
        },
    });
    let mut values = crate::ir::Arena::new();
    let debug_a = values.push(crate::ir::DebugValue {
        name: "a".to_string(),
        kind: crate::ir::DebugValueKind::Field { offset: 0 },
    });
    let program = crate::regalloc_engine::cfg_mir::Program {
        funcs: vec![func],
        vreg_count: 1,
        slot_count: 0,
        debug: crate::regalloc_engine::cfg_mir::ProgramDebugProvenance {
            scopes,
            values,
            root_scope: Some(root_scope),
            op_scopes: std::collections::HashMap::from([
                ((crate::ir::LambdaId::new(0), op_id), root_scope),
                ((crate::ir::LambdaId::new(0), term_op), root_scope),
            ]),
            op_values: std::collections::HashMap::new(),
            vreg_scopes: vec![Some(root_scope)],
            vreg_values: vec![Some(debug_a)],
        },
    };
    #[cfg(target_arch = "aarch64")]
    let reg = regalloc2::PReg::new(19, regalloc2::RegClass::Int);
    #[cfg(target_arch = "x86_64")]
    let reg = regalloc2::PReg::new(12, regalloc2::RegClass::Int);
    let alloc = crate::regalloc_engine::AllocatedCfgProgram {
        cfg_program: program.clone(),
        functions: vec![crate::regalloc_engine::AllocatedCfgFunction {
            lambda_id: crate::ir::LambdaId::new(0),
            num_spillslots: 0,
            edits: Vec::new(),
            op_allocs: std::collections::HashMap::from([(
                op_id,
                vec![regalloc2::Allocation::reg(reg)],
            )]),
            op_operands: std::collections::HashMap::from([
                (
                    op_id,
                    vec![(v0, crate::regalloc_engine::cfg_mir::OperandKind::Def)],
                ),
                (term_op, Vec::new()),
            ]),
            edge_edits: Vec::new(),
            return_result_allocs: Vec::new(),
        }],
    };
    let backend_debug_info = crate::ir_backend::BackendDebugInfo {
        op_infos: vec![
            crate::ir_backend::BackendOpDebugInfo {
                lambda_id: 0,
                op_id,
                line: 1,
                code_ranges: vec![crate::ir_backend::BackendCodeRange {
                    start_offset: 0,
                    end_offset: 4,
                }],
            },
            crate::ir_backend::BackendOpDebugInfo {
                lambda_id: 0,
                op_id: term_op,
                line: 2,
                code_ranges: vec![crate::ir_backend::BackendCodeRange {
                    start_offset: 4,
                    end_offset: 8,
                }],
            },
        ],
    };

    let vars = cfg_value_dwarf_variables(
        &program,
        &alloc,
        Some(&backend_debug_info),
        0x1000 as *const u8,
        jit_dwarf_target_arch(),
        true,
        true,
    );

    assert!(vars.is_empty(), "semantic-owned vregs should be hidden");
}

#[test]
fn cfg_semantic_named_dwarf_variables_merge_shared_vregs() {
    let v0 = crate::ir::VReg::new(0);
    let v1 = crate::ir::VReg::new(1);
    let inst0 = crate::regalloc_engine::cfg_mir::InstId::new(0);
    let inst1 = crate::regalloc_engine::cfg_mir::InstId::new(1);
    let inst2 = crate::regalloc_engine::cfg_mir::InstId::new(2);
    let term_id = crate::regalloc_engine::cfg_mir::TermId::new(0);
    let block_id = crate::regalloc_engine::cfg_mir::BlockId::new(0);
    let op0 = crate::regalloc_engine::cfg_mir::OpId::Inst(inst0);
    let op1 = crate::regalloc_engine::cfg_mir::OpId::Inst(inst1);
    let op2 = crate::regalloc_engine::cfg_mir::OpId::Inst(inst2);
    let term_op = crate::regalloc_engine::cfg_mir::OpId::Term(term_id);
    let root_scope = crate::ir::DebugScopeId::new(0);
    let func = crate::regalloc_engine::cfg_mir::Function {
        id: crate::regalloc_engine::cfg_mir::FunctionId::new(0),
        lambda_id: crate::ir::LambdaId::new(0),
        entry: block_id,
        data_args: Vec::new(),
        data_results: Vec::new(),
        blocks: vec![crate::regalloc_engine::cfg_mir::Block {
            id: block_id,
            params: Vec::new(),
            insts: vec![inst0, inst1, inst2],
            term: term_id,
            preds: Vec::new(),
            succs: Vec::new(),
        }],
        edges: Vec::new(),
        insts: vec![
            crate::regalloc_engine::cfg_mir::Inst {
                id: inst0,
                op: crate::linearize::LinearOp::Const { dst: v0, value: 1 },
                operands: vec![crate::regalloc_engine::cfg_mir::Operand {
                    vreg: v0,
                    kind: crate::regalloc_engine::cfg_mir::OperandKind::Def,
                    class: crate::regalloc_engine::cfg_mir::RegClass::Gpr,
                    fixed: None,
                }],
                clobbers: crate::regalloc_engine::cfg_mir::Clobbers::default(),
            },
            crate::regalloc_engine::cfg_mir::Inst {
                id: inst1,
                op: crate::linearize::LinearOp::Copy { dst: v1, src: v0 },
                operands: vec![
                    crate::regalloc_engine::cfg_mir::Operand {
                        vreg: v0,
                        kind: crate::regalloc_engine::cfg_mir::OperandKind::Use,
                        class: crate::regalloc_engine::cfg_mir::RegClass::Gpr,
                        fixed: None,
                    },
                    crate::regalloc_engine::cfg_mir::Operand {
                        vreg: v1,
                        kind: crate::regalloc_engine::cfg_mir::OperandKind::Def,
                        class: crate::regalloc_engine::cfg_mir::RegClass::Gpr,
                        fixed: None,
                    },
                ],
                clobbers: crate::regalloc_engine::cfg_mir::Clobbers::default(),
            },
            crate::regalloc_engine::cfg_mir::Inst {
                id: inst2,
                op: crate::linearize::LinearOp::WriteToField {
                    src: v1,
                    offset: 0,
                    width: crate::ir::Width::W1,
                },
                operands: vec![crate::regalloc_engine::cfg_mir::Operand {
                    vreg: v1,
                    kind: crate::regalloc_engine::cfg_mir::OperandKind::Use,
                    class: crate::regalloc_engine::cfg_mir::RegClass::Gpr,
                    fixed: None,
                }],
                clobbers: crate::regalloc_engine::cfg_mir::Clobbers::default(),
            },
        ],
        terms: vec![crate::regalloc_engine::cfg_mir::Terminator::Return],
    };
    let mut scopes = crate::ir::Arena::new();
    scopes.push(crate::ir::DebugScope {
        parent: None,
        kind: crate::ir::DebugScopeKind::LambdaBody {
            lambda_id: crate::ir::LambdaId::new(0),
        },
    });
    let mut values = crate::ir::Arena::new();
    let debug_flag = values.push(crate::ir::DebugValue {
        name: "flag".to_string(),
        kind: crate::ir::DebugValueKind::Named,
    });
    let program = crate::regalloc_engine::cfg_mir::Program {
        funcs: vec![func],
        vreg_count: 2,
        slot_count: 0,
        debug: crate::regalloc_engine::cfg_mir::ProgramDebugProvenance {
            scopes,
            values,
            root_scope: Some(root_scope),
            op_scopes: std::collections::HashMap::from([
                ((crate::ir::LambdaId::new(0), op0), root_scope),
                ((crate::ir::LambdaId::new(0), op1), root_scope),
                ((crate::ir::LambdaId::new(0), op2), root_scope),
                ((crate::ir::LambdaId::new(0), term_op), root_scope),
            ]),
            op_values: std::collections::HashMap::from([
                ((crate::ir::LambdaId::new(0), op0), debug_flag),
                ((crate::ir::LambdaId::new(0), op1), debug_flag),
            ]),
            vreg_scopes: vec![Some(root_scope), Some(root_scope)],
            vreg_values: vec![Some(debug_flag), Some(debug_flag)],
        },
    };
    #[cfg(target_arch = "aarch64")]
    let reg = regalloc2::PReg::new(19, regalloc2::RegClass::Int);
    #[cfg(target_arch = "x86_64")]
    let reg = regalloc2::PReg::new(12, regalloc2::RegClass::Int);
    let alloc = crate::regalloc_engine::AllocatedCfgProgram {
        cfg_program: program.clone(),
        functions: vec![crate::regalloc_engine::AllocatedCfgFunction {
            lambda_id: crate::ir::LambdaId::new(0),
            num_spillslots: 0,
            edits: Vec::new(),
            op_allocs: std::collections::HashMap::from([
                (op0, vec![regalloc2::Allocation::reg(reg)]),
                (
                    op1,
                    vec![
                        regalloc2::Allocation::reg(reg),
                        regalloc2::Allocation::reg(reg),
                    ],
                ),
                (op2, vec![regalloc2::Allocation::reg(reg)]),
            ]),
            op_operands: std::collections::HashMap::from([
                (
                    op0,
                    vec![(v0, crate::regalloc_engine::cfg_mir::OperandKind::Def)],
                ),
                (
                    op1,
                    vec![
                        (v0, crate::regalloc_engine::cfg_mir::OperandKind::Use),
                        (v1, crate::regalloc_engine::cfg_mir::OperandKind::Def),
                    ],
                ),
                (
                    op2,
                    vec![(v1, crate::regalloc_engine::cfg_mir::OperandKind::Use)],
                ),
                (term_op, Vec::new()),
            ]),
            edge_edits: Vec::new(),
            return_result_allocs: Vec::new(),
        }],
    };
    let backend_debug_info = crate::ir_backend::BackendDebugInfo {
        op_infos: vec![
            crate::ir_backend::BackendOpDebugInfo {
                lambda_id: 0,
                op_id: op0,
                line: 1,
                code_ranges: vec![crate::ir_backend::BackendCodeRange {
                    start_offset: 0,
                    end_offset: 4,
                }],
            },
            crate::ir_backend::BackendOpDebugInfo {
                lambda_id: 0,
                op_id: op1,
                line: 2,
                code_ranges: vec![crate::ir_backend::BackendCodeRange {
                    start_offset: 4,
                    end_offset: 8,
                }],
            },
            crate::ir_backend::BackendOpDebugInfo {
                lambda_id: 0,
                op_id: op2,
                line: 3,
                code_ranges: vec![crate::ir_backend::BackendCodeRange {
                    start_offset: 8,
                    end_offset: 12,
                }],
            },
            crate::ir_backend::BackendOpDebugInfo {
                lambda_id: 0,
                op_id: term_op,
                line: 4,
                code_ranges: vec![crate::ir_backend::BackendCodeRange {
                    start_offset: 12,
                    end_offset: 16,
                }],
            },
        ],
    };

    let vars = cfg_semantic_named_dwarf_variables(
        &program,
        &alloc,
        Some(&backend_debug_info),
        0x1000 as *const u8,
        jit_dwarf_target_arch(),
        true,
    );

    assert_eq!(vars.len(), 1);
    assert_eq!(vars[0].scope, Some(root_scope));
    assert_eq!(vars[0].variable.name, "flag");
    assert_eq!(
        vars[0].lexical_ranges,
        vec![crate::jit_dwarf::JitDebugRange {
            low_pc: 0x1000,
            high_pc: 0x100c,
        }]
    );
    match &vars[0].variable.location {
        crate::jit_dwarf::DwarfVariableLocation::List(locations) => {
            assert_eq!(locations.len(), 1);
            assert_eq!(locations[0].start, 0x1004);
            assert_eq!(locations[0].end, 0x100c);
        }
        crate::jit_dwarf::DwarfVariableLocation::Expr(_) => {
            panic!("semantic named vars should use ranged locations")
        }
    }
}
