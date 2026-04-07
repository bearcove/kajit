//! Token-based HIR parser. Built incrementally — each sub-parser
//! lives here and is composed in `parse_hir_from_tokens`.

use chumsky::{input::ValueInput, prelude::*};

use crate::lexer::{Span, Token};

pub type ParserExtra<'tokens, 'src> = extra::Err<Rich<'tokens, Token<'src>, Span>>;

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

// === Field offset ===

pub fn field_offset<'tokens, 'src: 'tokens, I>()
-> impl Parser<'tokens, I, Option<u32>, ParserExtra<'tokens, 'src>> + Clone
where
    I: ValueInput<'tokens, Token = Token<'src>, Span = Span>,
{
    just(Token::At)
        .ignore_then(uint64())
        .map(|v| Some(v as u32))
        .or_not()
        .map(|opt| opt.flatten())
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
}
