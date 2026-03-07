use chumsky::prelude::*;

use kajit_hir::{
    BinaryOp, Block, CallExpr, CallSafety, CallSignature, CallTarget, CallableId, CallableKind,
    CallableSpec, ControlTransfer, DomainAccess, DomainEffect, EffectClass, Expr, FieldDef,
    Function, FunctionId, GenericArg, GenericParam, Id, Literal, LocalDecl, LocalId, LocalKind,
    MatchArm, Module, Pattern, PatternField, Place, Scope, ScopeId, Stmt, StmtId, StmtKind,
    StoreId, Type, TypeDef, TypeDefId, TypeDefKind, UnaryOp, VariantDef,
};

type Extra<'src> = extra::Err<Rich<'src, char>>;

#[derive(Debug, Clone)]
struct ParsedRegion {
    id: Id<kajit_hir::RegionParam>,
    name: String,
}

#[derive(Debug, Clone)]
struct ParsedStore {
    id: Id<kajit_hir::StoreParam>,
    name: String,
}

#[derive(Debug, Clone)]
struct ParsedTypeDef {
    id: TypeDefId,
    def: TypeDef,
}

#[derive(Debug, Clone)]
struct ParsedCallable {
    id: CallableId,
    callable: CallableSpec,
}

#[derive(Debug, Clone)]
struct ParsedFunction {
    id: FunctionId,
    function: Function,
}

#[derive(Debug, Clone)]
struct ParsedModule {
    regions: Vec<ParsedRegion>,
    stores: Vec<ParsedStore>,
    types: Vec<ParsedTypeDef>,
    callables: Vec<ParsedCallable>,
    functions: Vec<ParsedFunction>,
}

fn ws<'src>() -> impl Parser<'src, &'src str, (), Extra<'src>> + Clone {
    any()
        .filter(|c: &char| c.is_whitespace())
        .repeated()
        .ignored()
}

fn token<'src>(text: &'static str) -> impl Parser<'src, &'src str, (), Extra<'src>> + Clone {
    just(text).padded_by(ws()).ignored()
}

fn uint32<'src>() -> impl Parser<'src, &'src str, u32, Extra<'src>> + Clone {
    text::int::<_, Extra<'_>>(10)
        .map(|s: &str| s.parse::<u32>().unwrap())
        .padded_by(ws())
}

fn uint64<'src>() -> impl Parser<'src, &'src str, u64, Extra<'src>> + Clone {
    let hex = just("0x")
        .ignore_then(text::int::<_, Extra<'_>>(16))
        .map(|s: &str| u64::from_str_radix(s, 16).unwrap());
    let dec = text::int::<_, Extra<'_>>(10).map(|s: &str| s.parse::<u64>().unwrap());
    hex.or(dec).padded_by(ws())
}

fn quoted_string<'src>() -> impl Parser<'src, &'src str, String, Extra<'src>> + Clone {
    let escape = just('\\').ignore_then(choice((
        just('\\').to('\\'),
        just('"').to('"'),
        just('n').to('\n'),
        just('r').to('\r'),
        just('t').to('\t'),
    )));
    let plain = any().filter(|c: &char| *c != '"' && *c != '\\');
    just('"')
        .ignore_then(choice((escape, plain)).repeated().collect::<String>())
        .then_ignore(just('"'))
        .padded_by(ws())
}

fn region_id<'src>() -> impl Parser<'src, &'src str, kajit_hir::RegionId, Extra<'src>> + Clone {
    just("r")
        .ignore_then(uint32())
        .map(kajit_hir::RegionId::new)
        .padded_by(ws())
}

fn store_id<'src>() -> impl Parser<'src, &'src str, StoreId, Extra<'src>> + Clone {
    just("store")
        .ignore_then(uint32())
        .map(StoreId::new)
        .padded_by(ws())
}

fn type_id<'src>() -> impl Parser<'src, &'src str, TypeDefId, Extra<'src>> + Clone {
    just("t")
        .ignore_then(uint32())
        .map(TypeDefId::new)
        .padded_by(ws())
}

fn callable_id<'src>() -> impl Parser<'src, &'src str, CallableId, Extra<'src>> + Clone {
    just("c")
        .ignore_then(uint32())
        .map(CallableId::new)
        .padded_by(ws())
}

fn function_id<'src>() -> impl Parser<'src, &'src str, FunctionId, Extra<'src>> + Clone {
    just("f")
        .ignore_then(uint32())
        .map(FunctionId::new)
        .padded_by(ws())
}

fn scope_id<'src>() -> impl Parser<'src, &'src str, ScopeId, Extra<'src>> + Clone {
    just("sc")
        .ignore_then(uint32())
        .map(ScopeId::new)
        .padded_by(ws())
}

fn local_id<'src>() -> impl Parser<'src, &'src str, LocalId, Extra<'src>> + Clone {
    just("l")
        .ignore_then(uint32())
        .map(LocalId::new)
        .padded_by(ws())
}

fn stmt_id<'src>() -> impl Parser<'src, &'src str, StmtId, Extra<'src>> + Clone {
    just("stmt")
        .ignore_then(uint32())
        .map(StmtId::new)
        .padded_by(ws())
}

fn generic_params<'src>() -> impl Parser<'src, &'src str, Vec<GenericParam>, Extra<'src>> + Clone {
    let item = choice((
        token("type")
            .ignore_then(quoted_string())
            .map(|name| GenericParam::Type { name }),
        token("region")
            .ignore_then(quoted_string())
            .map(|name| GenericParam::Region { name }),
        token("store")
            .ignore_then(quoted_string())
            .map(|name| GenericParam::Store { name }),
    ));
    token("<")
        .ignore_then(item.separated_by(token(",")).collect::<Vec<_>>())
        .then_ignore(token(">"))
        .or_not()
        .map(|opt| opt.unwrap_or_default())
}

fn ty<'src>() -> impl Parser<'src, &'src str, Type, Extra<'src>> + Clone {
    recursive(|ty| {
        let named = type_id()
            .then(
                token("<")
                    .ignore_then(
                        choice((
                            region_id().map(GenericArg::Region),
                            store_id().map(GenericArg::Store),
                            ty.clone().map(GenericArg::Type),
                        ))
                        .separated_by(token(","))
                        .collect::<Vec<_>>(),
                    )
                    .then_ignore(token(">"))
                    .or_not(),
            )
            .map(|(def, args)| Type::named(def, args.unwrap_or_default()));

        choice((
            token("unit").to(Type::unit()),
            token("bool").to(Type::bool()),
            just("u")
                .ignore_then(uint32())
                .map(|bits| Type::u(bits as u16))
                .padded_by(ws()),
            just("i")
                .ignore_then(uint32())
                .map(|bits| Type::i(bits as u16))
                .padded_by(ws()),
            token("Slice")
                .ignore_then(token("<"))
                .ignore_then(region_id())
                .then_ignore(token(","))
                .then(ty.clone())
                .then_ignore(token(">"))
                .map(|(region, element)| Type::slice(region, element)),
            token("Array")
                .ignore_then(token("<"))
                .ignore_then(ty.clone())
                .then_ignore(token(","))
                .then(uint32())
                .then_ignore(token(">"))
                .map(|(element, len)| Type::array(element, len as usize)),
            token("str")
                .ignore_then(token("<"))
                .ignore_then(region_id())
                .then_ignore(token(">"))
                .map(Type::str),
            token("Place")
                .ignore_then(token("<"))
                .ignore_then(ty.clone())
                .then_ignore(token(">"))
                .map(Type::place),
            token("Handle")
                .ignore_then(token("<"))
                .ignore_then(store_id())
                .then_ignore(token(","))
                .then(ty.clone())
                .then_ignore(token(">"))
                .map(|(store, value)| Type::handle(store, value)),
            named,
        ))
    })
}

fn field_defs<'src>() -> impl Parser<'src, &'src str, Vec<FieldDef>, Extra<'src>> + Clone {
    quoted_string()
        .then_ignore(token(":"))
        .then(ty())
        .map(|(name, ty)| FieldDef { name, ty })
        .repeated()
        .collect()
}

fn type_def<'src>() -> impl Parser<'src, &'src str, ParsedTypeDef, Extra<'src>> + Clone {
    let variant =
        quoted_string().then(token("{").ignore_then(field_defs()).then_ignore(token("}")));

    token("type")
        .ignore_then(type_id())
        .then(quoted_string())
        .then(generic_params())
        .then_ignore(token("="))
        .then(choice((
            token("struct")
                .ignore_then(token("{"))
                .ignore_then(field_defs())
                .then_ignore(token("}"))
                .map(|fields| TypeDefKind::Struct { fields }),
            token("enum")
                .ignore_then(token("{"))
                .ignore_then(variant.repeated().collect::<Vec<_>>().map(|variants| {
                    TypeDefKind::Enum {
                        variants: variants
                            .into_iter()
                            .map(|(name, fields)| VariantDef { name, fields })
                            .collect(),
                    }
                }))
                .then_ignore(token("}")),
        )))
        .map(|(((id, name), generic_params), kind)| ParsedTypeDef {
            id,
            def: TypeDef {
                name,
                generic_params,
                kind,
            },
        })
}

fn effect_class<'src>() -> impl Parser<'src, &'src str, EffectClass, Extra<'src>> + Clone {
    choice((
        token("pure").to(EffectClass::Pure),
        token("reads").to(EffectClass::Reads),
        token("mutates").to(EffectClass::Mutates),
        token("barrier").to(EffectClass::Barrier),
    ))
}

fn domain_access<'src>() -> impl Parser<'src, &'src str, DomainAccess, Extra<'src>> + Clone {
    choice((
        token("read").to(DomainAccess::Read),
        token("mutate").to(DomainAccess::Mutate),
    ))
}

fn callable<'src>() -> impl Parser<'src, &'src str, ParsedCallable, Extra<'src>> + Clone {
    token("callable")
        .ignore_then(callable_id())
        .then(choice((
            token("builtin").to(CallableKind::Builtin),
            token("host").to(CallableKind::Host),
        )))
        .then(quoted_string())
        .then_ignore(token("{"))
        .then(token("params").ignore_then(list_of(ty())))
        .then(token("returns").ignore_then(list_of(ty())))
        .then(token("effect").ignore_then(effect_class()))
        .then(
            token("domains").ignore_then(
                token("[")
                    .ignore_then(
                        quoted_string()
                            .then_ignore(token(":"))
                            .then(domain_access())
                            .map(|(domain, access)| DomainEffect { domain, access })
                            .separated_by(token(","))
                            .collect::<Vec<_>>(),
                    )
                    .then_ignore(token("]")),
            ),
        )
        .then(token("control").ignore_then(choice((
            token("returns").to(ControlTransfer::Returns),
            token("may_fail").to(ControlTransfer::MayFail),
            token("never_returns").to(ControlTransfer::NeverReturns),
        ))))
        .then(token("capabilities").ignore_then(list_of(quoted_string())))
        .then(token("safety").ignore_then(choice((
            token("safe_core").to(CallSafety::SafeCore),
            token("opaque_host").to(CallSafety::OpaqueHost),
            token("unsafe_interop").to(CallSafety::UnsafeInterop),
        ))))
        .then(
            token("docs").ignore_then(choice((token("none").to(None), quoted_string().map(Some)))),
        )
        .then_ignore(token("}"))
        .map(|data| {
            let (data, docs) = data;
            let (data, safety) = data;
            let (data, capabilities) = data;
            let (data, control) = data;
            let (data, domain_effects) = data;
            let (data, effect_class) = data;
            let (data, returns) = data;
            let (data, params) = data;
            let ((id, kind), name) = data;
            ParsedCallable {
                id,
                callable: CallableSpec {
                    kind,
                    name,
                    signature: CallSignature {
                        params,
                        returns,
                        effect_class,
                        domain_effects,
                        control,
                        capabilities,
                        safety,
                    },
                    docs,
                },
            }
        })
}

fn list_of<'src, T: 'src, P>(parser: P) -> impl Parser<'src, &'src str, Vec<T>, Extra<'src>> + Clone
where
    P: Parser<'src, &'src str, T, Extra<'src>> + Clone,
{
    token("[")
        .ignore_then(parser.separated_by(token(",")).collect::<Vec<_>>())
        .then_ignore(token("]"))
}

fn local_kind<'src>() -> impl Parser<'src, &'src str, LocalKind, Extra<'src>> + Clone {
    choice((
        token("param").to(LocalKind::Param),
        token("let").to(LocalKind::Let),
        token("temp").to(LocalKind::Temp),
        token("destination").to(LocalKind::Destination),
    ))
}

fn local_decl<'src>() -> impl Parser<'src, &'src str, LocalDecl, Extra<'src>> + Clone {
    local_id()
        .then(local_kind())
        .then(quoted_string())
        .then_ignore(token(":"))
        .then(ty())
        .map(|(((local, kind), name), ty)| LocalDecl {
            local,
            name,
            ty,
            kind,
        })
}

fn scope<'src>() -> impl Parser<'src, &'src str, Scope, Extra<'src>> + Clone {
    token("scope")
        .ignore_then(scope_id())
        .then_ignore(token("parent"))
        .then(choice((scope_id().map(Some), token("none").to(None))))
        .then_ignore(token("comment"))
        .then(choice((token("none").to(None), quoted_string().map(Some))))
        .map(|((id, parent), comment)| Scope {
            id,
            parent,
            comment,
        })
}

fn pattern<'src>() -> impl Parser<'src, &'src str, Pattern, Extra<'src>> + Clone {
    let pattern_field = quoted_string()
        .then_ignore(token("="))
        .then(choice((
            local_id().map(|local| PatternField::Bind {
                field: String::new(),
                local,
            }),
            token("_").to(PatternField::Wildcard {
                field: String::new(),
            }),
        )))
        .map(|(field, pattern)| match pattern {
            PatternField::Bind { local, .. } => PatternField::Bind { field, local },
            PatternField::Wildcard { .. } => PatternField::Wildcard { field },
        });

    choice((
        token("_").to(Pattern::Wildcard),
        token("true").to(Pattern::Bool(true)),
        token("false").to(Pattern::Bool(false)),
        token("variant")
            .ignore_then(quoted_string())
            .then(
                token("{")
                    .ignore_then(pattern_field.separated_by(token(",")).collect::<Vec<_>>())
                    .then_ignore(token("}")),
            )
            .map(|(name, fields)| Pattern::Variant { name, fields }),
        uint64().map(Pattern::Integer),
    ))
}

fn expr<'src>() -> impl Parser<'src, &'src str, Expr, Extra<'src>> + Clone {
    recursive(|expr| {
        let local = local_id().map(Expr::Local);

        let literal = choice((
            token("()").to(Expr::Literal(Literal::Unit)),
            token("true").to(Expr::Literal(Literal::Bool(true))),
            token("false").to(Expr::Literal(Literal::Bool(false))),
            uint64().map(|value| Expr::Literal(Literal::Integer(value))),
            quoted_string().map(|value| Expr::Literal(Literal::String(value))),
        ));

        let call = token("call")
            .ignore_then(callable_id())
            .then(
                token("(")
                    .ignore_then(expr.clone().separated_by(token(",")).collect::<Vec<_>>())
                    .then_ignore(token(")")),
            )
            .map(|(callable, args)| {
                Expr::Call(CallExpr {
                    target: CallTarget::Callable(callable),
                    args,
                })
            });

        let field_expr = token("field")
            .ignore_then(token("("))
            .ignore_then(expr.clone())
            .then_ignore(token(","))
            .then(quoted_string())
            .then_ignore(token(")"))
            .map(|(base, field)| Expr::Field {
                base: Box::new(base),
                field,
            });

        let index_expr = token("index")
            .ignore_then(token("("))
            .ignore_then(expr.clone())
            .then_ignore(token(","))
            .then(expr.clone())
            .then_ignore(token(")"))
            .map(|(base, index)| Expr::Index {
                base: Box::new(base),
                index: Box::new(index),
            });

        let struct_expr = token("struct")
            .ignore_then(type_id())
            .then(
                token("{")
                    .ignore_then(
                        quoted_string()
                            .then_ignore(token("="))
                            .then(expr.clone())
                            .separated_by(token(","))
                            .collect::<Vec<_>>(),
                    )
                    .then_ignore(token("}")),
            )
            .map(|(def, fields)| Expr::Struct { def, fields });

        let variant_expr = token("variant")
            .ignore_then(type_id())
            .then_ignore(token("::"))
            .then(quoted_string())
            .then(
                token("{")
                    .ignore_then(
                        quoted_string()
                            .then_ignore(token("="))
                            .then(expr.clone())
                            .separated_by(token(","))
                            .collect::<Vec<_>>(),
                    )
                    .then_ignore(token("}")),
            )
            .map(|((def, variant), fields)| Expr::Variant {
                def,
                variant,
                fields,
            });

        let unary_expr = token("unary")
            .ignore_then(choice((
                token("not").to(UnaryOp::Not),
                token("neg").to(UnaryOp::Neg),
            )))
            .then_ignore(token("("))
            .then(expr.clone())
            .then_ignore(token(")"))
            .map(|(op, value)| Expr::Unary {
                op,
                value: Box::new(value),
            });

        let binary_expr = token("binary")
            .ignore_then(choice((
                token("add").to(BinaryOp::Add),
                token("sub").to(BinaryOp::Sub),
                token("mul").to(BinaryOp::Mul),
                token("div").to(BinaryOp::Div),
                token("eq").to(BinaryOp::Eq),
                token("ne").to(BinaryOp::Ne),
                token("lt").to(BinaryOp::Lt),
                token("le").to(BinaryOp::Le),
                token("gt").to(BinaryOp::Gt),
                token("ge").to(BinaryOp::Ge),
                token("and").to(BinaryOp::And),
                token("or").to(BinaryOp::Or),
            )))
            .then_ignore(token("("))
            .then(expr.clone())
            .then_ignore(token(","))
            .then(expr.clone())
            .then_ignore(token(")"))
            .map(|((op, lhs), rhs)| Expr::Binary {
                op,
                lhs: Box::new(lhs),
                rhs: Box::new(rhs),
            });

        choice((
            call,
            field_expr,
            index_expr,
            struct_expr,
            variant_expr,
            unary_expr,
            binary_expr,
            local,
            literal,
            token("(").ignore_then(expr.clone()).then_ignore(token(")")),
        ))
        .boxed()
    })
}

fn place<'src, P>(expr: P) -> impl Parser<'src, &'src str, Place, Extra<'src>> + Clone
where
    P: Parser<'src, &'src str, Expr, Extra<'src>> + Clone + 'src,
{
    recursive(move |place| {
        choice((
            local_id().map(Place::Local),
            token("field")
                .ignore_then(token("("))
                .ignore_then(place.clone())
                .then_ignore(token(","))
                .then(quoted_string())
                .then_ignore(token(")"))
                .map(|(base, field)| Place::Field {
                    base: Box::new(base),
                    field,
                }),
            token("index")
                .ignore_then(token("("))
                .ignore_then(place.clone())
                .then_ignore(token(","))
                .then(expr.clone())
                .then_ignore(token(")"))
                .map(|(base, index)| Place::Index {
                    base: Box::new(base),
                    index: Box::new(index),
                }),
        ))
    })
}

fn stmt<'src>() -> impl Parser<'src, &'src str, Stmt, Extra<'src>> + Clone {
    recursive(|stmt| {
        let expr = expr();
        let place = place(expr.clone());
        let block = scope_id()
            .delimited_by(token("@"), token("{"))
            .then(stmt.repeated().collect::<Vec<_>>())
            .then_ignore(token("}"))
            .map(|(scope, statements)| Block { scope, statements });

        let if_stmt = stmt_id()
            .then_ignore(token(":"))
            .then_ignore(token("if"))
            .then(expr.clone())
            .then(block.clone())
            .then(token("else").ignore_then(block.clone()).or_not())
            .map(|(((id, condition), then_block), else_block)| Stmt {
                id,
                kind: StmtKind::If {
                    condition,
                    then_block,
                    else_block,
                },
            });

        let loop_stmt = stmt_id()
            .then_ignore(token(":"))
            .then_ignore(token("loop"))
            .then(block.clone())
            .map(|(id, body)| Stmt {
                id,
                kind: StmtKind::Loop { body },
            });

        let match_stmt = stmt_id()
            .then_ignore(token(":"))
            .then_ignore(token("match"))
            .then(expr.clone())
            .then_ignore(token("{"))
            .then(
                token("arm")
                    .ignore_then(pattern())
                    .then(block.clone())
                    .map(|(pattern, body)| MatchArm { pattern, body })
                    .repeated()
                    .collect::<Vec<_>>(),
            )
            .then_ignore(token("}"))
            .map(|((id, scrutinee), arms)| Stmt {
                id,
                kind: StmtKind::Match { scrutinee, arms },
            });

        let init_stmt = stmt_id()
            .then_ignore(token(":"))
            .then_ignore(token("init"))
            .then(place.clone())
            .then_ignore(token("="))
            .then(expr.clone())
            .map(|((id, place), value)| Stmt {
                id,
                kind: StmtKind::Init { place, value },
            });

        let assign_stmt = stmt_id()
            .then_ignore(token(":"))
            .then_ignore(token("assign"))
            .then(place.clone())
            .then_ignore(token("="))
            .then(expr.clone())
            .map(|((id, place), value)| Stmt {
                id,
                kind: StmtKind::Assign { place, value },
            });

        let expr_stmt = stmt_id()
            .then_ignore(token(":"))
            .then_ignore(token("expr"))
            .then(expr.clone())
            .map(|(id, expr)| Stmt {
                id,
                kind: StmtKind::Expr(expr),
            });

        let break_stmt = stmt_id()
            .then_ignore(token(":"))
            .then_ignore(token("break"))
            .map(|id| Stmt {
                id,
                kind: StmtKind::Break,
            });

        let continue_stmt = stmt_id()
            .then_ignore(token(":"))
            .then_ignore(token("continue"))
            .map(|id| Stmt {
                id,
                kind: StmtKind::Continue,
            });

        let return_stmt = stmt_id()
            .then_ignore(token(":"))
            .then_ignore(token("return"))
            .then(expr.clone().or_not())
            .map(|(id, expr)| Stmt {
                id,
                kind: StmtKind::Return(expr),
            });

        choice((
            if_stmt,
            loop_stmt,
            match_stmt,
            init_stmt,
            assign_stmt,
            expr_stmt,
            break_stmt,
            continue_stmt,
            return_stmt,
        ))
    })
}

fn block<'src>() -> impl Parser<'src, &'src str, Block, Extra<'src>> + Clone {
    scope_id()
        .delimited_by(token("@"), token("{"))
        .then(stmt().repeated().collect::<Vec<_>>())
        .then_ignore(token("}"))
        .map(|(scope, statements)| Block { scope, statements })
}

fn function<'src>() -> impl Parser<'src, &'src str, ParsedFunction, Extra<'src>> + Clone {
    token("function")
        .ignore_then(function_id())
        .then(quoted_string())
        .then_ignore(token("{"))
        .then(token("regions").ignore_then(list_of(region_id())))
        .then(token("stores").ignore_then(list_of(store_id())))
        .then(
            token("params").ignore_then(
                token("[")
                    .ignore_then(local_decl().repeated().collect::<Vec<_>>())
                    .then_ignore(token("]")),
            ),
        )
        .then(
            token("locals").ignore_then(
                token("[")
                    .ignore_then(local_decl().repeated().collect::<Vec<_>>())
                    .then_ignore(token("]")),
            ),
        )
        .then(token("return").ignore_then(ty()))
        .then(
            token("scopes").ignore_then(
                token("[")
                    .ignore_then(scope().repeated().collect::<Vec<_>>())
                    .then_ignore(token("]")),
            ),
        )
        .then(token("body").ignore_then(block()))
        .then_ignore(token("}"))
        .map(
            |(
                (
                    ((((((id, name), region_params), store_params), params), locals), return_type),
                    scopes,
                ),
                body,
            )| {
                ParsedFunction {
                    id,
                    function: Function {
                        name,
                        region_params,
                        store_params,
                        params: params
                            .into_iter()
                            .map(|local| kajit_hir::Parameter {
                                local: local.local,
                                name: local.name,
                                ty: local.ty,
                                kind: local.kind,
                            })
                            .collect(),
                        locals,
                        return_type,
                        scopes,
                        body,
                    },
                }
            },
        )
}

fn region<'src>() -> impl Parser<'src, &'src str, ParsedRegion, Extra<'src>> + Clone {
    region_id()
        .then(quoted_string())
        .map(|(id, name)| ParsedRegion { id, name })
}

fn store<'src>() -> impl Parser<'src, &'src str, ParsedStore, Extra<'src>> + Clone {
    store_id()
        .then(quoted_string())
        .map(|(id, name)| ParsedStore { id, name })
}

fn module_parser<'src>() -> impl Parser<'src, &'src str, ParsedModule, Extra<'src>> + Clone {
    token("hir_module")
        .ignore_then(token("{"))
        .ignore_then(
            token("regions").ignore_then(
                token("[")
                    .ignore_then(region().repeated().collect::<Vec<_>>())
                    .then_ignore(token("]")),
            ),
        )
        .then(
            token("stores").ignore_then(
                token("[")
                    .ignore_then(store().repeated().collect::<Vec<_>>())
                    .then_ignore(token("]")),
            ),
        )
        .then(
            token("types").ignore_then(
                token("[")
                    .ignore_then(type_def().repeated().collect::<Vec<_>>())
                    .then_ignore(token("]")),
            ),
        )
        .then(
            token("callables").ignore_then(
                token("[")
                    .ignore_then(callable().repeated().collect::<Vec<_>>())
                    .then_ignore(token("]")),
            ),
        )
        .then(
            token("functions").ignore_then(
                token("[")
                    .ignore_then(function().repeated().collect::<Vec<_>>())
                    .then_ignore(token("]")),
            ),
        )
        .then_ignore(token("}"))
        .then_ignore(ws())
        .then_ignore(end())
        .map(
            |((((regions, stores), types), callables), functions)| ParsedModule {
                regions,
                stores,
                types,
                callables,
                functions,
            },
        )
}

fn build_module(parsed: ParsedModule) -> Result<Module, String> {
    let mut module = Module::new();

    for (index, region) in parsed.regions.into_iter().enumerate() {
        if region.id.index() != index {
            return Err(format!(
                "region IDs must be sequential from r0, got r{} at index {}",
                region.id.index(),
                index
            ));
        }
        let inserted = module.add_region(region.name);
        debug_assert_eq!(inserted, region.id);
    }

    for (index, store) in parsed.stores.into_iter().enumerate() {
        if store.id.index() != index {
            return Err(format!(
                "store IDs must be sequential from store0, got store{} at index {}",
                store.id.index(),
                index
            ));
        }
        let inserted = module.add_store(store.name);
        debug_assert_eq!(inserted, store.id);
    }

    for (index, type_def) in parsed.types.into_iter().enumerate() {
        if type_def.id.index() != index {
            return Err(format!(
                "type IDs must be sequential from t0, got t{} at index {}",
                type_def.id.index(),
                index
            ));
        }
        let inserted = module.add_type_def(type_def.def);
        debug_assert_eq!(inserted, type_def.id);
    }

    for (index, callable) in parsed.callables.into_iter().enumerate() {
        if callable.id.index() != index {
            return Err(format!(
                "callable IDs must be sequential from c0, got c{} at index {}",
                callable.id.index(),
                index
            ));
        }
        let inserted = module.add_callable(callable.callable);
        debug_assert_eq!(inserted, callable.id);
    }

    for (index, function) in parsed.functions.into_iter().enumerate() {
        if function.id.index() != index {
            return Err(format!(
                "function IDs must be sequential from f0, got f{} at index {}",
                function.id.index(),
                index
            ));
        }
        let inserted = module.add_function(function.function);
        debug_assert_eq!(inserted, function.id);
    }

    Ok(module)
}

pub fn parse_hir(text: &str) -> Result<Module, String> {
    let parsed = module_parser().parse(text).into_result().map_err(|errs| {
        errs.into_iter()
            .map(|err| err.to_string())
            .collect::<Vec<_>>()
            .join("\n")
    })?;
    build_module(parsed)
}

#[cfg(test)]
mod tests {
    use super::parse_hir;
    use kajit_hir::{
        BinaryOp, Block, CallExpr, CallSafety, CallSignature, CallTarget, CallableKind,
        CallableSpec, ControlTransfer, DomainAccess, DomainEffect, EffectClass, Expr, FieldDef,
        Function, GenericArg, GenericParam, Literal, LocalDecl, LocalId, LocalKind, MatchArm,
        Module, Pattern, PatternField, Place, Scope, ScopeId, Stmt, StmtId, StmtKind, Type,
        TypeDef, TypeDefKind, VariantDef, VixenCallableRef, VixenCoreTypes, VixenTypedExpr,
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
                    },
                    FieldDef {
                        name: "pos".to_owned(),
                        ty: Type::u(64),
                    },
                ],
            },
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
                    },
                    VariantDef {
                        name: "Some".to_owned(),
                        fields: vec![FieldDef {
                            name: "value".to_owned(),
                            ty: Type::str(r_input),
                        }],
                    },
                ],
            },
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
                }],
            },
        });
        let read_tag = module.add_callable(CallableSpec {
            kind: CallableKind::Builtin,
            name: "postcard.read_option_tag".to_owned(),
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
                    ty: Type::place(Type::named(record, vec![GenericArg::Region(r_input)])),
                    kind: LocalKind::Destination,
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

    #[test]
    fn round_trips_sample_module() {
        let module = sample_module();
        let text = module.to_string();
        let reparsed = parse_hir(&text).expect("HIR text should parse");
        assert_eq!(reparsed, module);
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
                    }],
                }],
            },
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
                            op: BinaryOp::Add,
                            lhs: Box::new(Expr::Literal(Literal::Integer(1))),
                            rhs: Box::new(Expr::Literal(Literal::Integer(2))),
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
        });
        let node = module.add_type_def(TypeDef {
            name: "Node".to_owned(),
            generic_params: vec![],
            kind: TypeDefKind::Struct {
                fields: vec![FieldDef {
                    name: "label".to_owned(),
                    ty: Type::named(string, Vec::new()),
                }],
            },
        });
        let edge = module.add_type_def(TypeDef {
            name: "Edge".to_owned(),
            generic_params: vec![],
            kind: TypeDefKind::Struct { fields: vec![] },
        });
        let fact = module.add_type_def(TypeDef {
            name: "Fact".to_owned(),
            generic_params: vec![],
            kind: TypeDefKind::Struct { fields: vec![] },
        });
        let crate_graph = module.add_type_def(TypeDef {
            name: "CrateGraph".to_owned(),
            generic_params: vec![],
            kind: TypeDefKind::Struct { fields: vec![] },
        });
        let crate_node = module.add_type_def(TypeDef {
            name: "CrateNode".to_owned(),
            generic_params: vec![],
            kind: TypeDefKind::Struct { fields: vec![] },
        });
        let crate_id = module.add_type_def(TypeDef {
            name: "CrateId".to_owned(),
            generic_params: vec![],
            kind: TypeDefKind::Struct {
                fields: vec![FieldDef {
                    name: "value".to_owned(),
                    ty: Type::named(string, Vec::new()),
                }],
            },
        });
        let crate_type = module.add_type_def(TypeDef {
            name: "CrateType".to_owned(),
            generic_params: vec![],
            kind: TypeDefKind::Enum {
                variants: vec![
                    VariantDef {
                        name: "Lib".to_owned(),
                        fields: vec![],
                    },
                    VariantDef {
                        name: "Bin".to_owned(),
                        fields: vec![],
                    },
                ],
            },
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
}
