//! Token-based HIR parser. Built incrementally — each sub-parser
//! lives here and is composed in `parse_hir_from_tokens`.

use chumsky::{input::ValueInput, prelude::*};

use crate::lexer::{Span, Token};

pub type ParserExtra<'tokens, 'src> = extra::Err<Rich<'tokens, Token<'src>, Span>>;

// === Parsed intermediate types (same as in hir_parse.rs) ===

#[derive(Debug, Clone)]
pub struct ParsedRegion {
    pub id: kajit_hir::Id<kajit_hir::RegionParam>,
    pub name: String,
}

#[derive(Debug, Clone)]
pub struct ParsedStore {
    pub id: kajit_hir::Id<kajit_hir::StoreParam>,
    pub name: String,
}

#[derive(Debug, Clone)]
pub struct ParsedTypeDef {
    pub id: kajit_hir::TypeDefId,
    pub def: kajit_hir::TypeDef,
}

#[derive(Debug, Clone)]
pub struct ParsedCallable {
    pub id: kajit_hir::CallableId,
    pub callable: kajit_hir::CallableSpec,
}

#[derive(Debug, Clone)]
pub struct ParsedFunction {
    pub id: kajit_hir::FunctionId,
    pub function: kajit_hir::Function,
}

#[derive(Debug, Clone)]
pub struct ParsedModule {
    pub regions: Vec<ParsedRegion>,
    pub stores: Vec<ParsedStore>,
    pub types: Vec<ParsedTypeDef>,
    pub callables: Vec<ParsedCallable>,
    pub functions: Vec<ParsedFunction>,
}

// === Primitive token consumers ===

pub fn uint32<'tokens, 'src: 'tokens, I>()
-> impl Parser<'tokens, I, u32, ParserExtra<'tokens, 'src>> + Clone
where
    I: ValueInput<'tokens, Token = Token<'src>, Span = Span>,
{
    select! {
        Token::Int(n) => n as u32,
    }
    .labelled("integer")
}

pub fn uint64<'tokens, 'src: 'tokens, I>()
-> impl Parser<'tokens, I, u64, ParserExtra<'tokens, 'src>> + Clone
where
    I: ValueInput<'tokens, Token = Token<'src>, Span = Span>,
{
    select! {
        Token::Int(n) => n,
        Token::HexInt(n) => n,
    }
    .labelled("integer")
}

pub fn signed_int64<'tokens, 'src: 'tokens, I>()
-> impl Parser<'tokens, I, i64, ParserExtra<'tokens, 'src>> + Clone
where
    I: ValueInput<'tokens, Token = Token<'src>, Span = Span>,
{
    let positive = select! {
        Token::Int(n) => n as i64,
        Token::HexInt(n) => n as i64,
    };
    let negative = just(Token::Minus).ignore_then(select! { Token::Int(n) => -(n as i64) });
    negative.or(positive).labelled("integer")
}

pub fn quoted_string<'tokens, 'src: 'tokens, I>()
-> impl Parser<'tokens, I, String, ParserExtra<'tokens, 'src>> + Clone
where
    I: ValueInput<'tokens, Token = Token<'src>, Span = Span>,
{
    select! {
        Token::Str(s) => s.to_string(),
    }
    .labelled("quoted string")
}

pub fn ident<'tokens, 'src: 'tokens, I>()
-> impl Parser<'tokens, I, &'src str, ParserExtra<'tokens, 'src>> + Clone
where
    I: ValueInput<'tokens, Token = Token<'src>, Span = Span>,
{
    select! {
        Token::Ident(s) => s,
    }
    .labelled("identifier")
}

// === ID consumers ===

pub fn local_id<'tokens, 'src: 'tokens, I>()
-> impl Parser<'tokens, I, kajit_hir::LocalId, ParserExtra<'tokens, 'src>> + Clone
where
    I: ValueInput<'tokens, Token = Token<'src>, Span = Span>,
{
    select! { Token::LocalId(n) => kajit_hir::LocalId::new(n) }.labelled("local id (l0, l1, ...)")
}

pub fn region_id<'tokens, 'src: 'tokens, I>()
-> impl Parser<'tokens, I, kajit_hir::RegionId, ParserExtra<'tokens, 'src>> + Clone
where
    I: ValueInput<'tokens, Token = Token<'src>, Span = Span>,
{
    select! { Token::RegionId(n) => kajit_hir::RegionId::new(n) }
        .labelled("region id (r0, r1, ...)")
}

pub fn store_id<'tokens, 'src: 'tokens, I>()
-> impl Parser<'tokens, I, kajit_hir::StoreId, ParserExtra<'tokens, 'src>> + Clone
where
    I: ValueInput<'tokens, Token = Token<'src>, Span = Span>,
{
    select! { Token::StoreId(n) => kajit_hir::StoreId::new(n) }
        .labelled("store id (store0, store1, ...)")
}

pub fn type_def_id<'tokens, 'src: 'tokens, I>()
-> impl Parser<'tokens, I, kajit_hir::TypeDefId, ParserExtra<'tokens, 'src>> + Clone
where
    I: ValueInput<'tokens, Token = Token<'src>, Span = Span>,
{
    select! { Token::TypeDefId(n) => kajit_hir::TypeDefId::new(n) }
        .labelled("type id (t0, t1, ...)")
}

pub fn callable_id<'tokens, 'src: 'tokens, I>()
-> impl Parser<'tokens, I, kajit_hir::CallableId, ParserExtra<'tokens, 'src>> + Clone
where
    I: ValueInput<'tokens, Token = Token<'src>, Span = Span>,
{
    select! { Token::CallableId(n) => kajit_hir::CallableId::new(n) }
        .labelled("callable id (c0, c1, ...)")
}

pub fn function_id<'tokens, 'src: 'tokens, I>()
-> impl Parser<'tokens, I, kajit_hir::FunctionId, ParserExtra<'tokens, 'src>> + Clone
where
    I: ValueInput<'tokens, Token = Token<'src>, Span = Span>,
{
    select! { Token::FunctionId(n) => kajit_hir::FunctionId::new(n) }
        .labelled("function id (f0, f1, ...)")
}

pub fn scope_id<'tokens, 'src: 'tokens, I>()
-> impl Parser<'tokens, I, kajit_hir::ScopeId, ParserExtra<'tokens, 'src>> + Clone
where
    I: ValueInput<'tokens, Token = Token<'src>, Span = Span>,
{
    select! { Token::ScopeId(n) => kajit_hir::ScopeId::new(n) }.labelled("scope id (sc0, sc1, ...)")
}

pub fn stmt_id<'tokens, 'src: 'tokens, I>()
-> impl Parser<'tokens, I, kajit_hir::StmtId, ParserExtra<'tokens, 'src>> + Clone
where
    I: ValueInput<'tokens, Token = Token<'src>, Span = Span>,
{
    select! { Token::StmtId(n) => kajit_hir::StmtId::new(n) }
        .labelled("statement id (stmt0, stmt1, ...)")
}

// === Helpers ===

/// Parse a bracketed list: `[ item, item, ... ]`
/// (trailing comma optional, separator optional for compatibility)
pub fn bracketed_list<'tokens, 'src: 'tokens, T: 'tokens, I>(
    item: impl Parser<'tokens, I, T, ParserExtra<'tokens, 'src>> + Clone + 'tokens,
) -> impl Parser<'tokens, I, Vec<T>, ParserExtra<'tokens, 'src>> + Clone
where
    I: ValueInput<'tokens, Token = Token<'src>, Span = Span>,
{
    item.separated_by(just(Token::Comma))
        .allow_trailing()
        .collect::<Vec<_>>()
        .delimited_by(just(Token::LBracket), just(Token::RBracket))
}

// === Memory width ===

pub fn memory_width<'tokens, 'src: 'tokens, I>()
-> impl Parser<'tokens, I, kajit_hir::MemoryWidth, ParserExtra<'tokens, 'src>> + Clone
where
    I: ValueInput<'tokens, Token = Token<'src>, Span = Span>,
{
    use kajit_hir::MemoryWidth;
    choice((
        just(Token::KwW1).to(MemoryWidth::W1),
        just(Token::KwW2).to(MemoryWidth::W2),
        just(Token::KwW4).to(MemoryWidth::W4),
        just(Token::KwW8).to(MemoryWidth::W8),
    ))
    .labelled("memory width (w1, w2, w4, w8)")
}

// === Type parser ===

pub fn ty<'tokens, 'src: 'tokens, I>()
-> impl Parser<'tokens, I, kajit_hir::Type, ParserExtra<'tokens, 'src>> + Clone
where
    I: ValueInput<'tokens, Token = Token<'src>, Span = Span>,
{
    use kajit_hir::{AllocationDomain, GenericArg, Type};

    recursive(|ty| {
        // Named type: t0 or t0<r1, store2, SomeType>
        let named = type_def_id()
            .then(
                choice((
                    region_id().map(GenericArg::Region),
                    store_id().map(GenericArg::Store),
                    ty.clone().map(GenericArg::Type),
                ))
                .separated_by(just(Token::Comma))
                .collect::<Vec<_>>()
                .delimited_by(just(Token::LAngle), just(Token::RAngle))
                .or_not(),
            )
            .map(|(def, args)| Type::named(def, args.unwrap_or_default()));

        // Integer types: u8, u16, u32, u64, i8, i16, i32, i64
        // These are lexed as Ident("u32") etc.
        let int_type = select! {
            Token::Ident(s) if s.starts_with('u') || s.starts_with('i') => s,
        }
        .try_map(|s: &str, span| {
            let (prefix, bits_str) = s.split_at(1);
            let bits: u16 = bits_str
                .parse()
                .map_err(|_| Rich::custom(span, format!("invalid integer type '{s}'")))?;
            match prefix {
                "u" => Ok(Type::u(bits)),
                "i" => Ok(Type::i(bits)),
                _ => Err(Rich::custom(span, format!("invalid integer type '{s}'"))),
            }
        })
        .labelled("integer type (u8, u16, u32, u64, i8, ...)");

        choice((
            // &T or &mut T
            just(Token::Amp)
                .ignore_then(just(Token::KwMut).or_not())
                .then(ty.clone())
                .map(|(mutable, pointee)| {
                    if mutable.is_some() {
                        Type::mut_ref(pointee)
                    } else {
                        Type::r#ref(pointee)
                    }
                }),
            just(Token::KwUnit).to(Type::unit()),
            just(Token::KwBool).to(Type::bool()),
            int_type,
            // addr<transient> or addr<persistent>
            just(Token::KwAddr)
                .ignore_then(
                    choice((
                        just(Token::KwTransient).to(AllocationDomain::Transient),
                        just(Token::KwPersistent).to(AllocationDomain::Persistent),
                    ))
                    .delimited_by(just(Token::LAngle), just(Token::RAngle)),
                )
                .map(Type::address),
            // Slice<r0, T>
            just(Token::KwSlice)
                .ignore_then(
                    region_id()
                        .then_ignore(just(Token::Comma))
                        .then(ty.clone())
                        .delimited_by(just(Token::LAngle), just(Token::RAngle)),
                )
                .map(|(region, element)| Type::slice(region, element)),
            // Array<T, N>
            just(Token::KwArray)
                .ignore_then(
                    ty.clone()
                        .then_ignore(just(Token::Comma))
                        .then(uint32())
                        .delimited_by(just(Token::LAngle), just(Token::RAngle)),
                )
                .map(|(element, len)| Type::array(element, len as usize)),
            // str<r0>
            just(Token::KwStr)
                .ignore_then(region_id().delimited_by(just(Token::LAngle), just(Token::RAngle)))
                .map(Type::str),
            // Handle<store0, T>
            just(Token::KwHandle)
                .ignore_then(
                    store_id()
                        .then_ignore(just(Token::Comma))
                        .then(ty.clone())
                        .delimited_by(just(Token::LAngle), just(Token::RAngle)),
                )
                .map(|(store, value)| Type::handle(store, value)),
            named,
        ))
        .labelled("type")
    })
}

// === Pattern parser ===

pub fn pattern<'tokens, 'src: 'tokens, I>()
-> impl Parser<'tokens, I, kajit_hir::Pattern, ParserExtra<'tokens, 'src>> + Clone
where
    I: ValueInput<'tokens, Token = Token<'src>, Span = Span>,
{
    use kajit_hir::{Pattern, PatternField};

    let pattern_field = quoted_string()
        .then_ignore(just(Token::Eq))
        .then(choice((
            local_id().map(|local| PatternField::Bind {
                field: String::new(),
                local,
            }),
            just(Token::Ident("_")).to(PatternField::Wildcard {
                field: String::new(),
            }),
        )))
        .map(|(field, pf)| match pf {
            PatternField::Bind { local, .. } => PatternField::Bind { field, local },
            PatternField::Wildcard { .. } => PatternField::Wildcard { field },
        });

    choice((
        just(Token::Ident("_")).to(Pattern::Wildcard),
        just(Token::KwTrue).to(Pattern::Bool(true)),
        just(Token::KwFalse).to(Pattern::Bool(false)),
        just(Token::KwVariant)
            .ignore_then(quoted_string())
            .then(
                pattern_field
                    .separated_by(just(Token::Comma))
                    .collect::<Vec<_>>()
                    .delimited_by(just(Token::LBrace), just(Token::RBrace)),
            )
            .map(|(name, fields)| Pattern::Variant { name, fields }),
        uint64().map(Pattern::Integer),
    ))
    .labelled("pattern")
}

// === Place parser ===

// === Expression parser ===

/// Standalone place parser. Uses expr() internally for deref/index sub-expressions.
pub fn place<'tokens, 'src: 'tokens, I>()
-> impl Parser<'tokens, I, kajit_hir::Place, ParserExtra<'tokens, 'src>> + Clone
where
    I: ValueInput<'tokens, Token = Token<'src>, Span = Span>,
{
    use kajit_hir::Place;
    recursive(|place| {
        choice((
            local_id().map(Place::Local),
            just(Token::KwDeref)
                .ignore_then(expr().delimited_by(just(Token::LParen), just(Token::RParen)))
                .map(|base| Place::Deref {
                    base: Box::new(base),
                }),
            just(Token::KwField)
                .ignore_then(
                    place
                        .clone()
                        .then_ignore(just(Token::Comma))
                        .then(quoted_string())
                        .delimited_by(just(Token::LParen), just(Token::RParen)),
                )
                .map(|(base, field)| Place::Field {
                    base: Box::new(base),
                    field,
                }),
            just(Token::KwIndex)
                .ignore_then(
                    place
                        .clone()
                        .then_ignore(just(Token::Comma))
                        .then(expr())
                        .delimited_by(just(Token::LParen), just(Token::RParen)),
                )
                .map(|(base, index)| Place::Index {
                    base: Box::new(base),
                    index: Box::new(index),
                }),
        ))
        .labelled("place")
    })
}

pub fn expr<'tokens, 'src: 'tokens, I>()
-> impl Parser<'tokens, I, kajit_hir::Expr, ParserExtra<'tokens, 'src>> + Clone
where
    I: ValueInput<'tokens, Token = Token<'src>, Span = Span>,
{
    use kajit_hir::{BinaryOp, CallExpr, CallTarget, Expr, Literal, Place, UnaryOp};

    recursive(|expr| {
        // Place parser defined inside expr's recursive scope to avoid mutual recursion
        let place_parser = {
            let expr_clone = expr.clone();
            recursive(
                move |place: Recursive<
                    dyn Parser<'tokens, I, Place, ParserExtra<'tokens, 'src>> + '_,
                >| {
                    choice((
                        local_id().map(Place::Local),
                        just(Token::KwDeref)
                            .ignore_then(
                                expr_clone
                                    .clone()
                                    .delimited_by(just(Token::LParen), just(Token::RParen)),
                            )
                            .map(|base| Place::Deref {
                                base: Box::new(base),
                            }),
                        just(Token::KwField)
                            .ignore_then(
                                place
                                    .clone()
                                    .then_ignore(just(Token::Comma))
                                    .then(quoted_string())
                                    .delimited_by(just(Token::LParen), just(Token::RParen)),
                            )
                            .map(|(base, field)| Place::Field {
                                base: Box::new(base),
                                field,
                            }),
                        just(Token::KwIndex)
                            .ignore_then(
                                place
                                    .clone()
                                    .then_ignore(just(Token::Comma))
                                    .then(expr_clone.clone())
                                    .delimited_by(just(Token::LParen), just(Token::RParen)),
                            )
                            .map(|(base, index)| Place::Index {
                                base: Box::new(base),
                                index: Box::new(index),
                            }),
                    ))
                    .labelled("place")
                },
            )
        };

        let local = local_id().map(Expr::Local);

        let extern_addr = select! {
            Token::ExternSymbol(name) => Expr::Literal(Literal::ExternAddr {
                symbol: kajit_types::SymbolName::new(name.to_string()),
            }),
        };

        let literal = choice((
            // () for unit
            just(Token::LParen)
                .then_ignore(just(Token::RParen))
                .to(Expr::Literal(Literal::Unit)),
            just(Token::KwTrue).to(Expr::Literal(Literal::Bool(true))),
            just(Token::KwFalse).to(Expr::Literal(Literal::Bool(false))),
            // "none" is not a literal — it's used for Option fields in scope/function decls
            uint64().map(|value| Expr::Literal(Literal::Integer(value))),
            quoted_string().map(|value| Expr::Literal(Literal::String(value))),
            extern_addr,
        ));

        let call = just(Token::KwCall)
            .ignore_then(callable_id())
            .then(
                expr.clone()
                    .separated_by(just(Token::Comma))
                    .collect::<Vec<_>>()
                    .delimited_by(just(Token::LParen), just(Token::RParen)),
            )
            .map(|(callable, args)| {
                Expr::Call(CallExpr {
                    target: CallTarget::Callable(callable),
                    args,
                })
            });

        let load_expr = just(Token::KwLoad)
            .ignore_then(memory_width())
            .then(
                expr.clone()
                    .delimited_by(just(Token::LParen), just(Token::RParen)),
            )
            .map(|(width, addr)| Expr::Load {
                addr: Box::new(addr),
                width,
            });

        let slice_data_expr = just(Token::KwSliceData)
            .ignore_then(
                expr.clone()
                    .delimited_by(just(Token::LParen), just(Token::RParen)),
            )
            .map(|value| Expr::SliceData {
                value: Box::new(value),
            });

        let slice_len_expr = just(Token::KwSliceLen)
            .ignore_then(
                expr.clone()
                    .delimited_by(just(Token::LParen), just(Token::RParen)),
            )
            .map(|value| Expr::SliceLen {
                value: Box::new(value),
            });

        let str_expr = just(Token::KwStr)
            .ignore_then(
                expr.clone()
                    .then_ignore(just(Token::Comma))
                    .then(expr.clone())
                    .delimited_by(just(Token::LParen), just(Token::RParen)),
            )
            .map(|(data, len)| Expr::Str {
                data: Box::new(data),
                len: Box::new(len),
            });

        let deref_expr = just(Token::KwDeref)
            .ignore_then(
                expr.clone()
                    .delimited_by(just(Token::LParen), just(Token::RParen)),
            )
            .map(|base| Expr::Deref(Box::new(base)));

        let field_expr = just(Token::KwField)
            .ignore_then(
                expr.clone()
                    .then_ignore(just(Token::Comma))
                    .then(quoted_string())
                    .delimited_by(just(Token::LParen), just(Token::RParen)),
            )
            .map(|(base, field)| Expr::Field {
                base: Box::new(base),
                field,
            });

        let index_expr = just(Token::KwIndex)
            .ignore_then(
                expr.clone()
                    .then_ignore(just(Token::Comma))
                    .then(expr.clone())
                    .delimited_by(just(Token::LParen), just(Token::RParen)),
            )
            .map(|(base, index)| Expr::Index {
                base: Box::new(base),
                index: Box::new(index),
            });

        let addr_of_expr = just(Token::KwAddrOf)
            .ignore_then(place_parser.delimited_by(just(Token::LParen), just(Token::RParen)))
            .map(|place| Expr::AddrOf(Box::new(place)));

        let struct_expr = just(Token::KwStruct)
            .ignore_then(type_def_id())
            .then(
                quoted_string()
                    .then_ignore(just(Token::Eq))
                    .then(expr.clone())
                    .separated_by(just(Token::Comma))
                    .collect::<Vec<_>>()
                    .delimited_by(just(Token::LBrace), just(Token::RBrace)),
            )
            .map(|(def, fields)| Expr::Struct { def, fields });

        let variant_expr = just(Token::KwVariant)
            .ignore_then(type_def_id())
            .then_ignore(just(Token::ColonColon))
            .then(quoted_string())
            .then(
                quoted_string()
                    .then_ignore(just(Token::Eq))
                    .then(expr.clone())
                    .separated_by(just(Token::Comma))
                    .collect::<Vec<_>>()
                    .delimited_by(just(Token::LBrace), just(Token::RBrace)),
            )
            .map(|((def, variant), fields)| Expr::Variant {
                def,
                variant,
                fields,
            });

        let unary_op = choice((
            just(Token::KwNot).to(UnaryOp::Not),
            just(Token::KwNeg).to(UnaryOp::Neg),
        ));
        let unary_expr = just(Token::KwUnary)
            .ignore_then(unary_op)
            .then(
                expr.clone()
                    .delimited_by(just(Token::LParen), just(Token::RParen)),
            )
            .map(|(op, value)| Expr::Unary {
                op,
                value: Box::new(value),
            });

        let binary_op = choice((
            just(Token::KwAdd).to(BinaryOp::Add),
            just(Token::KwSub).to(BinaryOp::Sub),
            just(Token::KwMul).to(BinaryOp::Mul),
            just(Token::KwDiv).to(BinaryOp::Div),
            just(Token::KwBitand).to(BinaryOp::BitAnd),
            just(Token::KwBitor).to(BinaryOp::BitOr),
            just(Token::KwXor).to(BinaryOp::Xor),
            just(Token::KwShl).to(BinaryOp::Shl),
            just(Token::KwShr).to(BinaryOp::Shr),
            just(Token::KwSar).to(BinaryOp::Sar),
            just(Token::KwEq).to(BinaryOp::Eq),
            just(Token::KwNe).to(BinaryOp::Ne),
            just(Token::KwLt).to(BinaryOp::Lt),
            just(Token::KwLe).to(BinaryOp::Le),
            just(Token::KwGt).to(BinaryOp::Gt),
            just(Token::KwGe).to(BinaryOp::Ge),
            just(Token::KwAnd).to(BinaryOp::And),
            just(Token::KwOr).to(BinaryOp::Or),
        ))
        .labelled("binary operator");
        let binary_expr = just(Token::KwBinary)
            .ignore_then(binary_op)
            .then(
                expr.clone()
                    .then_ignore(just(Token::Comma))
                    .then(expr.clone())
                    .delimited_by(just(Token::LParen), just(Token::RParen)),
            )
            .map(|(op, (lhs, rhs))| Expr::Binary {
                op,
                lhs: Box::new(lhs),
                rhs: Box::new(rhs),
            });

        choice((
            slice_data_expr,
            slice_len_expr,
            str_expr,
            deref_expr,
            load_expr,
            call,
            field_expr,
            index_expr,
            addr_of_expr,
            struct_expr,
            variant_expr,
            unary_expr,
            binary_expr,
            local,
            literal,
            // Parenthesized expression
            expr.clone()
                .delimited_by(just(Token::LParen), just(Token::RParen)),
        ))
        .labelled("expression")
        .boxed()
    })
}

// === Statement / Block parser ===

pub fn block<'tokens, 'src: 'tokens, I>()
-> impl Parser<'tokens, I, kajit_hir::Block, ParserExtra<'tokens, 'src>> + Clone
where
    I: ValueInput<'tokens, Token = Token<'src>, Span = Span>,
{
    scope_id()
        .delimited_by(just(Token::At), just(Token::LBrace))
        .then(stmt().repeated().collect::<Vec<_>>())
        .then_ignore(just(Token::RBrace))
        .map(|(scope, statements)| kajit_hir::Block { scope, statements })
}

pub fn stmt<'tokens, 'src: 'tokens, I>()
-> impl Parser<'tokens, I, kajit_hir::Stmt, ParserExtra<'tokens, 'src>> + Clone
where
    I: ValueInput<'tokens, Token = Token<'src>, Span = Span>,
{
    use kajit_hir::{MatchArm, Stmt, StmtKind};

    recursive(|stmt| {
        let expr_p = expr();
        let block_p = scope_id()
            .delimited_by(just(Token::At), just(Token::LBrace))
            .then(stmt.repeated().collect::<Vec<_>>())
            .then_ignore(just(Token::RBrace))
            .map(|(scope, statements)| kajit_hir::Block { scope, statements });

        // Place parser for init/assign — we need it from the expr's recursive scope,
        // but here we define a simple version that uses the standalone place() which
        // delegates to expr(). Since stmt is itself recursive (for blocks), and place/expr
        // are separate recursive parsers, this should be fine.
        let place_p = place();

        let stmt_body = choice((
            just(Token::KwIf)
                .ignore_then(expr_p.clone())
                .then(block_p.clone())
                .then(just(Token::KwElse).ignore_then(block_p.clone()).or_not())
                .map(|((condition, then_block), else_block)| StmtKind::If {
                    condition,
                    then_block,
                    else_block,
                }),
            just(Token::KwLoop)
                .ignore_then(
                    just(Token::KwMaxIterations)
                        .ignore_then(just(Token::Eq))
                        .ignore_then(uint64())
                        .map(|v| Some(v as u32))
                        .or_not()
                        .map(|opt| opt.flatten()),
                )
                .then(block_p.clone())
                .map(|(max_iterations, body)| StmtKind::Loop {
                    body,
                    max_iterations,
                }),
            just(Token::KwMatch)
                .ignore_then(expr_p.clone())
                .then(
                    just(Token::KwArm)
                        .ignore_then(pattern())
                        .then(block_p.clone())
                        .map(|(pattern, body)| MatchArm { pattern, body })
                        .repeated()
                        .collect::<Vec<_>>()
                        .delimited_by(just(Token::LBrace), just(Token::RBrace)),
                )
                .map(|(scrutinee, arms)| StmtKind::Match { scrutinee, arms }),
            just(Token::KwInit)
                .ignore_then(place_p.clone())
                .then_ignore(just(Token::Eq))
                .then(expr_p.clone())
                .map(|(place, value)| StmtKind::Init { place, value }),
            just(Token::KwAssign)
                .ignore_then(place_p.clone())
                .then_ignore(just(Token::Eq))
                .then(expr_p.clone())
                .map(|(place, value)| StmtKind::Assign { place, value }),
            just(Token::KwStore)
                .ignore_then(memory_width())
                .then(expr_p.clone())
                .then_ignore(just(Token::Eq))
                .then(expr_p.clone())
                .map(|((width, addr), value)| StmtKind::Store { addr, width, value }),
            just(Token::KwExpr)
                .ignore_then(expr_p.clone())
                .map(StmtKind::Expr),
            just(Token::KwBreak).to(StmtKind::Break),
            just(Token::KwContinue).to(StmtKind::Continue),
            just(Token::KwReturn)
                .ignore_then(expr_p.clone().or_not())
                .map(StmtKind::Return),
        ))
        .labelled("statement keyword");

        stmt_id()
            .then_ignore(just(Token::Colon))
            .then(stmt_body)
            .map(|(id, kind)| Stmt { id, kind })
    })
}

// === Scope / Local / Parameter parsers ===

pub fn scope<'tokens, 'src: 'tokens, I>()
-> impl Parser<'tokens, I, kajit_hir::Scope, ParserExtra<'tokens, 'src>> + Clone
where
    I: ValueInput<'tokens, Token = Token<'src>, Span = Span>,
{
    just(Token::KwScope)
        .ignore_then(scope_id())
        .then_ignore(just(Token::KwParent))
        .then(choice((scope_id().map(Some), just(Token::KwNone).to(None))))
        .then_ignore(just(Token::KwComment))
        .then(choice((
            just(Token::KwNone).to(None),
            quoted_string().map(Some),
        )))
        .map(|((id, parent), comment)| kajit_hir::Scope {
            id,
            parent,
            comment,
        })
}

pub fn local_kind<'tokens, 'src: 'tokens, I>()
-> impl Parser<'tokens, I, kajit_hir::LocalKind, ParserExtra<'tokens, 'src>> + Clone
where
    I: ValueInput<'tokens, Token = Token<'src>, Span = Span>,
{
    use kajit_hir::LocalKind;
    choice((
        just(Token::KwParam).to(LocalKind::Param),
        just(Token::KwLet).to(LocalKind::Let),
        just(Token::KwTemp).to(LocalKind::Temp),
    ))
    .labelled("local kind")
}

pub fn local_decl<'tokens, 'src: 'tokens, I>()
-> impl Parser<'tokens, I, kajit_hir::LocalDecl, ParserExtra<'tokens, 'src>> + Clone
where
    I: ValueInput<'tokens, Token = Token<'src>, Span = Span>,
{
    local_id()
        .then(local_kind())
        .then(quoted_string())
        .then_ignore(just(Token::Colon))
        .then(ty())
        .map(|(((local, kind), name), ty)| kajit_hir::LocalDecl {
            local,
            name,
            ty,
            kind,
        })
}

pub fn parameter_decl<'tokens, 'src: 'tokens, I>()
-> impl Parser<'tokens, I, kajit_hir::Parameter, ParserExtra<'tokens, 'src>> + Clone
where
    I: ValueInput<'tokens, Token = Token<'src>, Span = Span>,
{
    local_id()
        .then(local_kind())
        .then(quoted_string())
        .then_ignore(just(Token::Colon))
        .then(ty())
        .map(|(((local, kind), name), ty)| kajit_hir::Parameter {
            local,
            name,
            ty,
            kind,
        })
}

// === Effect / Domain parsers ===

pub fn effect_class<'tokens, 'src: 'tokens, I>()
-> impl Parser<'tokens, I, kajit_hir::EffectClass, ParserExtra<'tokens, 'src>> + Clone
where
    I: ValueInput<'tokens, Token = Token<'src>, Span = Span>,
{
    use kajit_hir::EffectClass;
    choice((
        just(Token::KwPure).to(EffectClass::Pure),
        just(Token::KwReads).to(EffectClass::Reads),
        just(Token::KwMutates).to(EffectClass::Mutates),
        just(Token::KwBarrier).to(EffectClass::Barrier),
    ))
}

pub fn domain_access<'tokens, 'src: 'tokens, I>()
-> impl Parser<'tokens, I, kajit_hir::DomainAccess, ParserExtra<'tokens, 'src>> + Clone
where
    I: ValueInput<'tokens, Token = Token<'src>, Span = Span>,
{
    use kajit_hir::DomainAccess;
    choice((
        just(Token::KwRead).to(DomainAccess::Read),
        just(Token::KwMutate).to(DomainAccess::Mutate),
    ))
}

// === Callable parser ===

pub fn callable<'tokens, 'src: 'tokens, I>()
-> impl Parser<'tokens, I, ParsedCallable, ParserExtra<'tokens, 'src>> + Clone
where
    I: ValueInput<'tokens, Token = Token<'src>, Span = Span>,
{
    use kajit_hir::{
        CallSafety, CallSignature, CallableKind, CallableSpec, ControlTransfer, DomainEffect,
    };

    let runtime_intrinsic = ident().try_map(|name, span| {
        use kajit_hir::RuntimeIntrinsic;
        match name {
            "option_init_none" => Ok(RuntimeIntrinsic::OptionInitNone),
            "option_init_some" => Ok(RuntimeIntrinsic::OptionInitSome),
            "alloc_transient" => Ok(RuntimeIntrinsic::AllocTransient),
            "alloc_persistent" => Ok(RuntimeIntrinsic::AllocPersistent),
            "vec_from_raw_parts" => Ok(RuntimeIntrinsic::VecFromRawParts),
            "validate_utf8_range" => Ok(RuntimeIntrinsic::ValidateUtf8Range),
            "string_validate_alloc_copy" => Ok(RuntimeIntrinsic::StringValidateAllocCopy),
            "memcpy" => Ok(RuntimeIntrinsic::Memcpy),
            "free_transient" => Ok(RuntimeIntrinsic::FreeTransient),
            _ => Err(Rich::custom(span, format!("unknown intrinsic '{name}'"))),
        }
    });

    just(Token::KwCallable)
        .ignore_then(callable_id())
        .then(choice((
            just(Token::KwBuiltin).to(CallableKind::Builtin),
            just(Token::KwHost).to(CallableKind::Host),
        )))
        .then(quoted_string())
        .then_ignore(just(Token::LBrace))
        .then(just(Token::KwParams).ignore_then(bracketed_list(ty())))
        .then(
            just(Token::KwIntrinsic)
                .ignore_then(runtime_intrinsic)
                .map(Some)
                .or_not()
                .map(|v| v.flatten()),
        )
        .then(just(Token::KwReturns).ignore_then(bracketed_list(ty())))
        .then(just(Token::KwEffect).ignore_then(effect_class()))
        .then(
            just(Token::KwDomains).ignore_then(
                quoted_string()
                    .then_ignore(just(Token::Colon))
                    .then(domain_access())
                    .map(|(domain, access)| DomainEffect { domain, access })
                    .separated_by(just(Token::Comma))
                    .collect::<Vec<_>>()
                    .delimited_by(just(Token::LBracket), just(Token::RBracket)),
            ),
        )
        .then(just(Token::KwControl).ignore_then(choice((
            just(Token::KwReturns).to(ControlTransfer::Returns),
            just(Token::KwMayFail).to(ControlTransfer::MayFail),
            just(Token::KwNeverReturns).to(ControlTransfer::NeverReturns),
        ))))
        .then(just(Token::KwCapabilities).ignore_then(bracketed_list(quoted_string())))
        .then(just(Token::KwSafety).ignore_then(choice((
            just(Token::KwSafeCore).to(CallSafety::SafeCore),
            just(Token::KwOpaqueHost).to(CallSafety::OpaqueHost),
            just(Token::KwUnsafeInterop).to(CallSafety::UnsafeInterop),
        ))))
        .then(just(Token::KwDocs).ignore_then(choice((
            just(Token::KwNone).to(None),
            quoted_string().map(Some),
        ))))
        .then_ignore(just(Token::RBrace))
        .map(|data| {
            let (data, docs) = data;
            let (data, safety) = data;
            let (data, capabilities) = data;
            let (data, control) = data;
            let (data, domain_effects) = data;
            let (data, effect_class) = data;
            let (data, returns) = data;
            let (data, intrinsic) = data;
            let (data, params) = data;
            let ((id, kind), name) = data;
            ParsedCallable {
                id,
                callable: CallableSpec {
                    kind,
                    name,
                    intrinsic,
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

// === Generic params ===

pub fn generic_params<'tokens, 'src: 'tokens, I>()
-> impl Parser<'tokens, I, Vec<kajit_hir::GenericParam>, ParserExtra<'tokens, 'src>> + Clone
where
    I: ValueInput<'tokens, Token = Token<'src>, Span = Span>,
{
    use kajit_hir::GenericParam;
    let item = choice((
        just(Token::KwType)
            .ignore_then(quoted_string())
            .map(|name| GenericParam::Type { name }),
        just(Token::KwRegion)
            .ignore_then(quoted_string())
            .map(|name| GenericParam::Region { name }),
        just(Token::KwStore)
            .ignore_then(quoted_string())
            .map(|name| GenericParam::Store { name }),
    ));
    item.separated_by(just(Token::Comma))
        .collect::<Vec<_>>()
        .delimited_by(just(Token::LAngle), just(Token::RAngle))
        .or_not()
        .map(|opt| opt.unwrap_or_default())
}

// === Type definition parser ===

pub fn type_def<'tokens, 'src: 'tokens, I>()
-> impl Parser<'tokens, I, ParsedTypeDef, ParserExtra<'tokens, 'src>> + Clone
where
    I: ValueInput<'tokens, Token = Token<'src>, Span = Span>,
{
    use kajit_hir::{FieldDef, TypeDef, TypeDefKind, VariantDef};

    let field_def = quoted_string()
        .then_ignore(just(Token::Colon))
        .then(ty())
        .then(
            just(Token::At)
                .ignore_then(uint64())
                .map(|v| Some(v as u32))
                .or_not()
                .map(|opt| opt.flatten()),
        )
        .map(|((name, ty), offset)| FieldDef { name, ty, offset });

    let size_attr = just(Token::KwSize)
        .ignore_then(just(Token::Eq))
        .ignore_then(uint64())
        .map(|v| Some(v as u32))
        .or_not()
        .map(|opt| opt.flatten());

    let transparent_attr = just(Token::KwTransparent)
        .to(true)
        .or_not()
        .map(|opt| opt.unwrap_or(false));

    let disc_width_attr = just(Token::KwDiscWidth)
        .ignore_then(just(Token::Eq))
        .ignore_then(uint64())
        .map(|v| Some(v as u32))
        .or_not()
        .map(|opt| opt.flatten());

    let variant_discriminant = just(Token::Eq)
        .ignore_then(signed_int64())
        .map(Some)
        .or_not()
        .map(|opt| opt.flatten());

    let variant_init_fn = ident()
        .try_map(|name, span| {
            if name == "init_fn" {
                Ok(())
            } else {
                Err(Rich::custom(span, "expected 'init_fn'"))
            }
        })
        .ignore_then(just(Token::Eq))
        .ignore_then(uint64())
        .map(Some)
        .or_not()
        .map(|opt| opt.flatten());

    let variant = quoted_string()
        .then(variant_discriminant)
        .then(variant_init_fn)
        .then(
            field_def
                .clone()
                .repeated()
                .collect::<Vec<_>>()
                .delimited_by(just(Token::LBrace), just(Token::RBrace)),
        )
        .map(|(((name, discriminant), init_fn), fields)| VariantDef {
            name,
            fields,
            discriminant,
            init_fn,
        });

    just(Token::KwType)
        .ignore_then(type_def_id())
        .then(quoted_string())
        .then(generic_params())
        .then(size_attr)
        .then(transparent_attr)
        .then_ignore(just(Token::Eq))
        .then(choice((
            just(Token::KwStruct)
                .ignore_then(
                    field_def
                        .repeated()
                        .collect::<Vec<_>>()
                        .delimited_by(just(Token::LBrace), just(Token::RBrace)),
                )
                .map(|fields| (TypeDefKind::Struct { fields }, None)),
            just(Token::KwEnum)
                .ignore_then(disc_width_attr)
                .then(
                    variant
                        .repeated()
                        .collect::<Vec<_>>()
                        .delimited_by(just(Token::LBrace), just(Token::RBrace)),
                )
                .map(|(dw, variants)| {
                    (
                        TypeDefKind::Enum {
                            variants,
                            discriminant_width: dw,
                        },
                        dw,
                    )
                }),
        )))
        .map(
            |(((((id, name), generic_params), size), transparent), (kind, _))| ParsedTypeDef {
                id,
                def: TypeDef {
                    name,
                    generic_params,
                    kind,
                    size,
                    transparent,
                },
            },
        )
}

// === Function parser ===

pub fn function<'tokens, 'src: 'tokens, I>()
-> impl Parser<'tokens, I, ParsedFunction, ParserExtra<'tokens, 'src>> + Clone
where
    I: ValueInput<'tokens, Token = Token<'src>, Span = Span>,
{
    just(Token::KwFunction)
        .ignore_then(function_id())
        .then(quoted_string())
        .then_ignore(just(Token::LBrace))
        .then(just(Token::KwRegions).ignore_then(bracketed_list(region_id())))
        .then(just(Token::KwStores).ignore_then(bracketed_list(store_id())))
        .then(
            just(Token::KwParams).ignore_then(
                parameter_decl()
                    .repeated()
                    .collect::<Vec<_>>()
                    .delimited_by(just(Token::LBracket), just(Token::RBracket)),
            ),
        )
        .then(
            just(Token::KwLocals).ignore_then(
                local_decl()
                    .repeated()
                    .collect::<Vec<_>>()
                    .delimited_by(just(Token::LBracket), just(Token::RBracket)),
            ),
        )
        .then(just(Token::KwReturn).ignore_then(ty()))
        .then(
            just(Token::KwScopes).ignore_then(
                scope()
                    .repeated()
                    .collect::<Vec<_>>()
                    .delimited_by(just(Token::LBracket), just(Token::RBracket)),
            ),
        )
        .then(just(Token::KwBody).ignore_then(block()))
        .then_ignore(just(Token::RBrace))
        .map(|data| {
            let (data, body) = data;
            let (data, scopes) = data;
            let (data, return_type) = data;
            let (data, locals) = data;
            let (data, params) = data;
            let (data, store_params) = data;
            let (data, region_params) = data;
            let ((id, name), _) = (data, ());
            ParsedFunction {
                id,
                function: kajit_hir::Function {
                    name,
                    region_params,
                    store_params,
                    params,
                    locals,
                    return_type,
                    scopes,
                    body,
                },
            }
        })
}

// === Region / Store parsers ===

pub fn region<'tokens, 'src: 'tokens, I>()
-> impl Parser<'tokens, I, ParsedRegion, ParserExtra<'tokens, 'src>> + Clone
where
    I: ValueInput<'tokens, Token = Token<'src>, Span = Span>,
{
    region_id()
        .then(quoted_string())
        .map(|(id, name)| ParsedRegion { id, name })
}

pub fn store<'tokens, 'src: 'tokens, I>()
-> impl Parser<'tokens, I, ParsedStore, ParserExtra<'tokens, 'src>> + Clone
where
    I: ValueInput<'tokens, Token = Token<'src>, Span = Span>,
{
    store_id()
        .then(quoted_string())
        .map(|(id, name)| ParsedStore { id, name })
}

// === Module parser ===

pub fn module_parser<'tokens, 'src: 'tokens, I>()
-> impl Parser<'tokens, I, ParsedModule, ParserExtra<'tokens, 'src>> + Clone
where
    I: ValueInput<'tokens, Token = Token<'src>, Span = Span>,
{
    just(Token::KwHirModule)
        .ignore_then(just(Token::LBrace))
        .ignore_then(
            just(Token::KwRegions).ignore_then(
                region()
                    .repeated()
                    .collect::<Vec<_>>()
                    .delimited_by(just(Token::LBracket), just(Token::RBracket)),
            ),
        )
        .then(
            just(Token::KwStores).ignore_then(
                store()
                    .repeated()
                    .collect::<Vec<_>>()
                    .delimited_by(just(Token::LBracket), just(Token::RBracket)),
            ),
        )
        .then(
            just(Token::KwTypes).ignore_then(
                type_def()
                    .repeated()
                    .collect::<Vec<_>>()
                    .delimited_by(just(Token::LBracket), just(Token::RBracket)),
            ),
        )
        .then(
            just(Token::KwCallables).ignore_then(
                callable()
                    .repeated()
                    .collect::<Vec<_>>()
                    .delimited_by(just(Token::LBracket), just(Token::RBracket)),
            ),
        )
        .then(
            just(Token::KwFunctions).ignore_then(
                function()
                    .repeated()
                    .collect::<Vec<_>>()
                    .delimited_by(just(Token::LBracket), just(Token::RBracket)),
            ),
        )
        .then_ignore(just(Token::RBrace))
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

// === Two-pass parse entry point ===

fn build_module(parsed: ParsedModule) -> Result<kajit_hir::Module, String> {
    let mut module = kajit_hir::Module::new();

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

/// Two-pass HIR parser: lex into tokens, then parse token stream.
/// Produces better error messages than the single-pass char-level parser.
pub fn parse_hir_v2(text: &str) -> Result<kajit_hir::Module, String> {
    use chumsky::input::Input as _;

    // Phase 1: Lex
    let (tokens, lex_errs) = crate::lexer::lexer().parse(text).into_output_errors();

    if !lex_errs.is_empty() {
        return Err(kajit_parse_util::format_rich_errors(text, lex_errs));
    }

    let tokens = tokens.ok_or_else(|| "lexer produced no output".to_string())?;

    // Phase 2: Parse tokens
    let eoi: Span = (text.len()..text.len()).into();
    let token_stream = tokens.as_slice().map(eoi, |(t, s)| (t, s));

    let (parsed, parse_errs) = module_parser()
        .then_ignore(end())
        .parse(token_stream)
        .into_output_errors();

    if !parse_errs.is_empty() {
        // Format token-level errors — map spans back to source positions
        let mut buf = Vec::new();
        for e in &parse_errs {
            use ariadne::{Color, Label, Report, ReportKind, Source};
            Report::build(ReportKind::Error, ((), e.span().into_range()))
                .with_config(ariadne::Config::new().with_index_type(ariadne::IndexType::Byte))
                .with_message(e.to_string())
                .with_label(
                    Label::new(((), e.span().into_range()))
                        .with_message(e.reason().to_string())
                        .with_color(Color::Red),
                )
                .finish()
                .write(Source::from(text), &mut buf)
                .unwrap();
        }
        return Err(String::from_utf8(buf)
            .unwrap_or_else(|e| format!("(error formatting parse errors: {e})")));
    }

    let parsed = parsed.ok_or_else(|| "parser produced no output".to_string())?;

    // Phase 3: Build module from parsed AST
    build_module(parsed)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lexer;

    /// Lex then parse a string, returning the parsed value or error.
    macro_rules! parse_tokens {
        ($src:expr, $parser:expr) => {{
            let src: &str = $src;
            let (tokens, lex_errs) = lexer::lexer().parse(src).into_output_errors();
            assert!(lex_errs.is_empty(), "lex errors: {lex_errs:?}");
            let tokens = tokens.unwrap();
            let eoi: Span = (src.len()..src.len()).into();
            $parser
                .parse(tokens.as_slice().map(eoi, |(t, s)| (t, s)))
                .into_result()
                .map_err(|errs| format!("{errs:?}"))
        }};
    }

    #[test]
    fn parse_uint32() {
        assert_eq!(parse_tokens!("42", uint32()).unwrap(), 42);
    }

    #[test]
    fn parse_uint64_hex() {
        assert_eq!(parse_tokens!("0xff", uint64()).unwrap(), 255);
    }

    #[test]
    fn parse_negative_int() {
        assert_eq!(parse_tokens!("-5", signed_int64()).unwrap(), -5);
    }

    #[test]
    fn parse_string() {
        assert_eq!(
            parse_tokens!(r#""hello""#, quoted_string()).unwrap(),
            "hello"
        );
    }

    #[test]
    fn parse_local_id() {
        let id = parse_tokens!("l42", local_id()).unwrap();
        assert_eq!(id, kajit_hir::LocalId::new(42));
    }

    #[test]
    fn parse_bracketed_list_of_ints() {
        let result = parse_tokens!("[1, 2, 3]", bracketed_list(uint32())).unwrap();
        assert_eq!(result, vec![1, 2, 3]);
    }

    #[test]
    fn parse_empty_bracketed_list() {
        let result = parse_tokens!("[]", bracketed_list(uint32())).unwrap();
        assert_eq!(result, vec![]);
    }

    #[test]
    fn parse_type_unit() {
        let t = parse_tokens!("unit", ty()).unwrap();
        assert_eq!(t, kajit_hir::Type::unit());
    }

    #[test]
    fn parse_type_u32() {
        let t = parse_tokens!("u32", ty()).unwrap();
        assert_eq!(t, kajit_hir::Type::u(32));
    }

    #[test]
    fn parse_type_i64() {
        let t = parse_tokens!("i64", ty()).unwrap();
        assert_eq!(t, kajit_hir::Type::i(64));
    }

    #[test]
    fn parse_type_bool() {
        let t = parse_tokens!("bool", ty()).unwrap();
        assert_eq!(t, kajit_hir::Type::bool());
    }

    #[test]
    fn parse_type_addr_transient() {
        let t = parse_tokens!("addr<transient>", ty()).unwrap();
        assert_eq!(
            t,
            kajit_hir::Type::address(kajit_hir::AllocationDomain::Transient)
        );
    }

    #[test]
    fn parse_type_ref() {
        let t = parse_tokens!("&u32", ty()).unwrap();
        assert_eq!(t, kajit_hir::Type::r#ref(kajit_hir::Type::u(32)));
    }

    #[test]
    fn parse_type_mut_ref() {
        let t = parse_tokens!("&mut u64", ty()).unwrap();
        assert_eq!(t, kajit_hir::Type::mut_ref(kajit_hir::Type::u(64)));
    }

    #[test]
    fn parse_type_named() {
        let t = parse_tokens!("t0", ty()).unwrap();
        assert_eq!(
            t,
            kajit_hir::Type::named(kajit_hir::TypeDefId::new(0), vec![])
        );
    }

    #[test]
    fn parse_expr_local() {
        let e = parse_tokens!("l0", expr()).unwrap();
        assert_eq!(e, kajit_hir::Expr::Local(kajit_hir::LocalId::new(0)));
    }

    #[test]
    fn parse_expr_integer() {
        let e = parse_tokens!("42", expr()).unwrap();
        assert_eq!(e, kajit_hir::Expr::Literal(kajit_hir::Literal::Integer(42)));
    }

    #[test]
    fn parse_expr_binary_add() {
        let e = parse_tokens!("binary add(l0, l1)", expr()).unwrap();
        assert_eq!(
            e,
            kajit_hir::Expr::Binary {
                op: kajit_hir::BinaryOp::Add,
                lhs: Box::new(kajit_hir::Expr::Local(kajit_hir::LocalId::new(0))),
                rhs: Box::new(kajit_hir::Expr::Local(kajit_hir::LocalId::new(1))),
            }
        );
    }

    #[test]
    fn parse_expr_call() {
        let e = parse_tokens!("call c0(l0, l1)", expr()).unwrap();
        assert!(matches!(e, kajit_hir::Expr::Call(_)));
    }

    #[test]
    fn parse_expr_load() {
        let e = parse_tokens!("load w4(l0)", expr()).unwrap();
        assert!(matches!(e, kajit_hir::Expr::Load { .. }));
    }

    #[test]
    fn parse_expr_addr_of() {
        let e = parse_tokens!("addr_of(l0)", expr()).unwrap();
        assert!(matches!(e, kajit_hir::Expr::AddrOf(_)));
    }

    #[test]
    fn parse_stmt_init() {
        let s = parse_tokens!("stmt0: init l0 = 42", stmt()).unwrap();
        assert_eq!(s.id, kajit_hir::StmtId::new(0));
        assert!(matches!(s.kind, kajit_hir::StmtKind::Init { .. }));
    }

    #[test]
    fn parse_stmt_return() {
        let s = parse_tokens!("stmt0: return l0", stmt()).unwrap();
        assert!(matches!(s.kind, kajit_hir::StmtKind::Return(Some(_))));
    }

    #[test]
    fn parse_stmt_return_void() {
        let s = parse_tokens!("stmt0: return", stmt()).unwrap();
        assert!(matches!(s.kind, kajit_hir::StmtKind::Return(None)));
    }

    #[test]
    fn parse_stmt_break() {
        let s = parse_tokens!("stmt0: break", stmt()).unwrap();
        assert!(matches!(s.kind, kajit_hir::StmtKind::Break));
    }

    #[test]
    fn round_trip_simple_module() {
        // Build a simple module, print it, parse it back with the new two-pass parser
        let mut module = kajit_hir::Module::new();
        module.add_function(kajit_hir::Function {
            name: "add".to_string(),
            region_params: vec![],
            store_params: vec![],
            params: vec![
                kajit_hir::Parameter {
                    local: kajit_hir::LocalId::new(0),
                    name: "a".to_string(),
                    ty: kajit_hir::Type::u(32),
                    kind: kajit_hir::LocalKind::Param,
                },
                kajit_hir::Parameter {
                    local: kajit_hir::LocalId::new(1),
                    name: "b".to_string(),
                    ty: kajit_hir::Type::u(32),
                    kind: kajit_hir::LocalKind::Param,
                },
            ],
            locals: vec![],
            return_type: kajit_hir::Type::u(32),
            scopes: vec![kajit_hir::Scope {
                id: kajit_hir::ScopeId::new(0),
                parent: None,
                comment: None,
            }],
            body: kajit_hir::Block {
                scope: kajit_hir::ScopeId::new(0),
                statements: vec![kajit_hir::Stmt {
                    id: kajit_hir::StmtId::new(0),
                    kind: kajit_hir::StmtKind::Return(Some(kajit_hir::Expr::Binary {
                        op: kajit_hir::BinaryOp::Add,
                        lhs: Box::new(kajit_hir::Expr::Local(kajit_hir::LocalId::new(0))),
                        rhs: Box::new(kajit_hir::Expr::Local(kajit_hir::LocalId::new(1))),
                    })),
                }],
            },
        });

        let text = module.to_string();
        eprintln!("HIR text:\n{text}");
        let reparsed = super::parse_hir_v2(&text).expect("should parse with new parser");
        assert_eq!(reparsed, module);
    }

    #[test]
    fn parse_expr_addr_of_field() {
        let e = parse_tokens!("addr_of(field(l0, \"x\"))", expr()).unwrap();
        if let kajit_hir::Expr::AddrOf(place) = e {
            assert!(matches!(*place, kajit_hir::Place::Field { .. }));
        } else {
            panic!("expected AddrOf");
        }
    }
}
