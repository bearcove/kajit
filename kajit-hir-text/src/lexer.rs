use chumsky::prelude::*;
use std::fmt;

pub type Span = SimpleSpan;
pub type Spanned<T> = (T, Span);

#[derive(Clone, Debug, PartialEq)]
pub enum Token<'src> {
    // Punctuation
    LBrace,
    RBrace,
    LBracket,
    RBracket,
    LParen,
    RParen,
    LAngle,
    RAngle,
    Colon,
    ColonColon,
    Comma,
    Eq,
    At,
    Amp,
    Minus,

    // Literals
    Int(u64),
    HexInt(u64),
    Str(&'src str),

    // Identifiers (not a keyword)
    Ident(&'src str),

    // Prefixed IDs
    LocalId(u32),
    RegionId(u32),
    StoreId(u32),
    ScopeId(u32),
    StmtId(u32),
    TypeDefId(u32),
    CallableId(u32),
    FunctionId(u32),

    // Keywords — structure
    KwHirModule,
    KwType,
    KwStruct,
    KwEnum,
    KwCallable,
    KwFunction,
    KwRegions,
    KwStores,
    KwTypes,
    KwCallables,
    KwFunctions,

    // Keywords — local kinds
    KwParam,
    KwLet,
    KwTemp,
    KwDestination,

    // Keywords — types
    KwUnit,
    KwBool,
    KwAddr,
    KwTransient,
    KwPersistent,
    KwSlice,
    KwArray,
    KwStr,
    KwHandle,
    KwMut,
    KwTransparent,

    // Keywords — statements
    KwInit,
    KwAssign,
    KwStore,
    KwExpr,
    KwIf,
    KwElse,
    KwLoop,
    KwMatch,
    KwArm,
    KwBreak,
    KwContinue,
    KwReturn,

    // Keywords — expressions
    KwCall,
    KwLoad,
    KwDeref,
    KwSliceData,
    KwSliceLen,
    KwField,
    KwIndex,
    KwAddrOf,
    KwVariant,
    KwUnary,
    KwBinary,

    // Keywords — unary ops
    KwNot,
    KwNeg,

    // Keywords — binary ops
    KwAdd,
    KwSub,
    KwMul,
    KwDiv,
    KwBitand,
    KwBitor,
    KwXor,
    KwShl,
    KwShr,
    KwSar,
    KwEq,
    KwNe,
    KwLt,
    KwLe,
    KwGt,
    KwGe,
    KwAnd,
    KwOr,

    // Keywords — literals
    KwTrue,
    KwFalse,
    KwNone,

    // Keywords — scopes/structure
    KwParent,
    KwComment,
    KwDocs,
    KwScope,
    KwScopes,
    KwLocals,
    KwParams,
    KwCapabilities,
    KwControl,
    KwDomains,
    KwEffect,
    KwSafety,
    KwIntrinsic,
    KwBody,

    // Keywords — effects
    KwPure,
    KwReads,
    KwMutates,
    KwBarrier,
    KwRead,
    KwMutate,

    // Keywords — callable kinds
    KwBuiltin,
    KwHost,

    // Keywords — control transfer
    KwReturns,
    KwMayFail,
    KwNeverReturns,

    // Keywords — call safety
    KwSafeCore,
    KwOpaqueHost,
    KwUnsafeInterop,

    // Keywords — sizes
    KwSize,
    KwDiscWidth,

    // Keywords — memory widths
    KwW1,
    KwW2,
    KwW4,
    KwW8,

    // Keywords — region/store/type generic params
    KwRegion,
    KwGenericStore,
    // "type" reuses KwType
}

impl fmt::Display for Token<'_> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Token::LBrace => write!(f, "{{"),
            Token::RBrace => write!(f, "}}"),
            Token::LBracket => write!(f, "["),
            Token::RBracket => write!(f, "]"),
            Token::LParen => write!(f, "("),
            Token::RParen => write!(f, ")"),
            Token::LAngle => write!(f, "<"),
            Token::RAngle => write!(f, ">"),
            Token::Colon => write!(f, ":"),
            Token::ColonColon => write!(f, "::"),
            Token::Comma => write!(f, ","),
            Token::Eq => write!(f, "="),
            Token::At => write!(f, "@"),
            Token::Amp => write!(f, "&"),
            Token::Minus => write!(f, "-"),

            Token::Int(n) => write!(f, "{n}"),
            Token::HexInt(n) => write!(f, "0x{n:x}"),
            Token::Str(s) => write!(f, "\"{s}\""),
            Token::Ident(s) => write!(f, "{s}"),

            Token::LocalId(n) => write!(f, "l{n}"),
            Token::RegionId(n) => write!(f, "r{n}"),
            Token::StoreId(n) => write!(f, "store{n}"),
            Token::ScopeId(n) => write!(f, "sc{n}"),
            Token::StmtId(n) => write!(f, "stmt{n}"),
            Token::TypeDefId(n) => write!(f, "t{n}"),
            Token::CallableId(n) => write!(f, "c{n}"),
            Token::FunctionId(n) => write!(f, "f{n}"),

            Token::KwHirModule => write!(f, "hir_module"),
            Token::KwType => write!(f, "type"),
            Token::KwStruct => write!(f, "struct"),
            Token::KwEnum => write!(f, "enum"),
            Token::KwCallable => write!(f, "callable"),
            Token::KwFunction => write!(f, "function"),
            Token::KwRegions => write!(f, "regions"),
            Token::KwStores => write!(f, "stores"),
            Token::KwTypes => write!(f, "types"),
            Token::KwCallables => write!(f, "callables"),
            Token::KwFunctions => write!(f, "functions"),
            Token::KwParam => write!(f, "param"),
            Token::KwLet => write!(f, "let"),
            Token::KwTemp => write!(f, "temp"),
            Token::KwDestination => write!(f, "destination"),
            Token::KwUnit => write!(f, "unit"),
            Token::KwBool => write!(f, "bool"),
            Token::KwAddr => write!(f, "addr"),
            Token::KwTransient => write!(f, "transient"),
            Token::KwPersistent => write!(f, "persistent"),
            Token::KwSlice => write!(f, "Slice"),
            Token::KwArray => write!(f, "Array"),
            Token::KwStr => write!(f, "str"),
            Token::KwHandle => write!(f, "Handle"),
            Token::KwMut => write!(f, "mut"),
            Token::KwTransparent => write!(f, "transparent"),
            Token::KwInit => write!(f, "init"),
            Token::KwAssign => write!(f, "assign"),
            Token::KwStore => write!(f, "store"),
            Token::KwExpr => write!(f, "expr"),
            Token::KwIf => write!(f, "if"),
            Token::KwElse => write!(f, "else"),
            Token::KwLoop => write!(f, "loop"),
            Token::KwMatch => write!(f, "match"),
            Token::KwArm => write!(f, "arm"),
            Token::KwBreak => write!(f, "break"),
            Token::KwContinue => write!(f, "continue"),
            Token::KwReturn => write!(f, "return"),
            Token::KwCall => write!(f, "call"),
            Token::KwLoad => write!(f, "load"),
            Token::KwDeref => write!(f, "deref"),
            Token::KwSliceData => write!(f, "slice_data"),
            Token::KwSliceLen => write!(f, "slice_len"),
            Token::KwField => write!(f, "field"),
            Token::KwIndex => write!(f, "index"),
            Token::KwAddrOf => write!(f, "addr_of"),
            Token::KwVariant => write!(f, "variant"),
            Token::KwUnary => write!(f, "unary"),
            Token::KwBinary => write!(f, "binary"),
            Token::KwNot => write!(f, "not"),
            Token::KwNeg => write!(f, "neg"),
            Token::KwAdd => write!(f, "add"),
            Token::KwSub => write!(f, "sub"),
            Token::KwMul => write!(f, "mul"),
            Token::KwDiv => write!(f, "div"),
            Token::KwBitand => write!(f, "bitand"),
            Token::KwBitor => write!(f, "bitor"),
            Token::KwXor => write!(f, "xor"),
            Token::KwShl => write!(f, "shl"),
            Token::KwShr => write!(f, "shr"),
            Token::KwSar => write!(f, "sar"),
            Token::KwEq => write!(f, "eq"),
            Token::KwNe => write!(f, "ne"),
            Token::KwLt => write!(f, "lt"),
            Token::KwLe => write!(f, "le"),
            Token::KwGt => write!(f, "gt"),
            Token::KwGe => write!(f, "ge"),
            Token::KwAnd => write!(f, "and"),
            Token::KwOr => write!(f, "or"),
            Token::KwTrue => write!(f, "true"),
            Token::KwFalse => write!(f, "false"),
            Token::KwNone => write!(f, "none"),
            Token::KwParent => write!(f, "parent"),
            Token::KwComment => write!(f, "comment"),
            Token::KwDocs => write!(f, "docs"),
            Token::KwScope => write!(f, "scope"),
            Token::KwScopes => write!(f, "scopes"),
            Token::KwLocals => write!(f, "locals"),
            Token::KwParams => write!(f, "params"),
            Token::KwCapabilities => write!(f, "capabilities"),
            Token::KwControl => write!(f, "control"),
            Token::KwDomains => write!(f, "domains"),
            Token::KwEffect => write!(f, "effect"),
            Token::KwSafety => write!(f, "safety"),
            Token::KwIntrinsic => write!(f, "intrinsic"),
            Token::KwBody => write!(f, "body"),
            Token::KwPure => write!(f, "pure"),
            Token::KwReads => write!(f, "reads"),
            Token::KwMutates => write!(f, "mutates"),
            Token::KwBarrier => write!(f, "barrier"),
            Token::KwRead => write!(f, "read"),
            Token::KwMutate => write!(f, "mutate"),
            Token::KwBuiltin => write!(f, "builtin"),
            Token::KwHost => write!(f, "host"),
            Token::KwReturns => write!(f, "returns"),
            Token::KwMayFail => write!(f, "may_fail"),
            Token::KwNeverReturns => write!(f, "never_returns"),
            Token::KwSafeCore => write!(f, "safe_core"),
            Token::KwOpaqueHost => write!(f, "opaque_host"),
            Token::KwUnsafeInterop => write!(f, "unsafe_interop"),
            Token::KwSize => write!(f, "size"),
            Token::KwDiscWidth => write!(f, "disc_width"),
            Token::KwW1 => write!(f, "w1"),
            Token::KwW2 => write!(f, "w2"),
            Token::KwW4 => write!(f, "w4"),
            Token::KwW8 => write!(f, "w8"),
            Token::KwRegion => write!(f, "region"),
            Token::KwGenericStore => write!(f, "store"),
        }
    }
}

fn keyword_or_ident<'src>(s: &'src str) -> Token<'src> {
    match s {
        "hir_module" => Token::KwHirModule,
        "type" => Token::KwType,
        "struct" => Token::KwStruct,
        "enum" => Token::KwEnum,
        "callable" => Token::KwCallable,
        "function" => Token::KwFunction,
        "regions" => Token::KwRegions,
        "stores" => Token::KwStores,
        "types" => Token::KwTypes,
        "callables" => Token::KwCallables,
        "functions" => Token::KwFunctions,
        "param" => Token::KwParam,
        "let" => Token::KwLet,
        "temp" => Token::KwTemp,
        "destination" => Token::KwDestination,
        "unit" => Token::KwUnit,
        "bool" => Token::KwBool,
        "addr" => Token::KwAddr,
        "transient" => Token::KwTransient,
        "persistent" => Token::KwPersistent,
        "Slice" => Token::KwSlice,
        "Array" => Token::KwArray,
        "str" => Token::KwStr,
        "Handle" => Token::KwHandle,
        "mut" => Token::KwMut,
        "transparent" => Token::KwTransparent,
        "init" => Token::KwInit,
        "assign" => Token::KwAssign,
        "store" => Token::KwStore,
        "expr" => Token::KwExpr,
        "if" => Token::KwIf,
        "else" => Token::KwElse,
        "loop" => Token::KwLoop,
        "match" => Token::KwMatch,
        "arm" => Token::KwArm,
        "break" => Token::KwBreak,
        "continue" => Token::KwContinue,
        "return" => Token::KwReturn,
        "call" => Token::KwCall,
        "load" => Token::KwLoad,
        "deref" => Token::KwDeref,
        "slice_data" => Token::KwSliceData,
        "slice_len" => Token::KwSliceLen,
        "field" => Token::KwField,
        "index" => Token::KwIndex,
        "addr_of" => Token::KwAddrOf,
        "variant" => Token::KwVariant,
        "unary" => Token::KwUnary,
        "binary" => Token::KwBinary,
        "not" => Token::KwNot,
        "neg" => Token::KwNeg,
        "add" => Token::KwAdd,
        "sub" => Token::KwSub,
        "mul" => Token::KwMul,
        "div" => Token::KwDiv,
        "bitand" => Token::KwBitand,
        "bitor" => Token::KwBitor,
        "xor" => Token::KwXor,
        "shl" => Token::KwShl,
        "shr" => Token::KwShr,
        "sar" => Token::KwSar,
        "eq" => Token::KwEq,
        "ne" => Token::KwNe,
        "lt" => Token::KwLt,
        "le" => Token::KwLe,
        "gt" => Token::KwGt,
        "ge" => Token::KwGe,
        "and" => Token::KwAnd,
        "or" => Token::KwOr,
        "true" => Token::KwTrue,
        "false" => Token::KwFalse,
        "none" => Token::KwNone,
        "parent" => Token::KwParent,
        "comment" => Token::KwComment,
        "docs" => Token::KwDocs,
        "scope" => Token::KwScope,
        "scopes" => Token::KwScopes,
        "locals" => Token::KwLocals,
        "params" => Token::KwParams,
        "capabilities" => Token::KwCapabilities,
        "control" => Token::KwControl,
        "domains" => Token::KwDomains,
        "effect" => Token::KwEffect,
        "safety" => Token::KwSafety,
        "intrinsic" => Token::KwIntrinsic,
        "body" => Token::KwBody,
        "pure" => Token::KwPure,
        "reads" => Token::KwReads,
        "mutates" => Token::KwMutates,
        "barrier" => Token::KwBarrier,
        "read" => Token::KwRead,
        "mutate" => Token::KwMutate,
        "builtin" => Token::KwBuiltin,
        "host" => Token::KwHost,
        "returns" => Token::KwReturns,
        "may_fail" => Token::KwMayFail,
        "never_returns" => Token::KwNeverReturns,
        "safe_core" => Token::KwSafeCore,
        "opaque_host" => Token::KwOpaqueHost,
        "unsafe_interop" => Token::KwUnsafeInterop,
        "size" => Token::KwSize,
        "disc_width" => Token::KwDiscWidth,
        "w1" => Token::KwW1,
        "w2" => Token::KwW2,
        "w4" => Token::KwW4,
        "w8" => Token::KwW8,
        "region" => Token::KwRegion,
        _ => Token::Ident(s),
    }
}

/// Try to parse a prefixed ID like l0, r1, sc5, stmt42, t0, c1, f3, store0.
/// Returns None if the ident doesn't match any prefix pattern.
fn try_prefixed_id(s: &str) -> Option<Token<'_>> {
    if let Some(rest) = s.strip_prefix("stmt") {
        // Must come before "store" since both start with "st"
        return rest.parse::<u32>().ok().map(Token::StmtId);
    }
    if let Some(rest) = s.strip_prefix("store") {
        return rest.parse::<u32>().ok().map(Token::StoreId);
    }
    if let Some(rest) = s.strip_prefix("sc") {
        return rest.parse::<u32>().ok().map(Token::ScopeId);
    }
    if s.starts_with('l') && s.len() > 1 && s[1..].chars().all(|c| c.is_ascii_digit()) {
        return s[1..].parse::<u32>().ok().map(Token::LocalId);
    }
    if s.starts_with('r') && s.len() > 1 && s[1..].chars().all(|c| c.is_ascii_digit()) {
        return s[1..].parse::<u32>().ok().map(Token::RegionId);
    }
    if s.starts_with('t') && s.len() > 1 && s[1..].chars().all(|c| c.is_ascii_digit()) {
        return s[1..].parse::<u32>().ok().map(Token::TypeDefId);
    }
    if s.starts_with('c') && s.len() > 1 && s[1..].chars().all(|c| c.is_ascii_digit()) {
        return s[1..].parse::<u32>().ok().map(Token::CallableId);
    }
    if s.starts_with('f') && s.len() > 1 && s[1..].chars().all(|c| c.is_ascii_digit()) {
        return s[1..].parse::<u32>().ok().map(Token::FunctionId);
    }
    None
}

type LexExtra<'src> = extra::Err<Rich<'src, char, Span>>;

pub fn lexer<'src>() -> impl Parser<'src, &'src str, Vec<Spanned<Token<'src>>>, LexExtra<'src>> {
    let hex_int = just("0x")
        .ignore_then(text::digits(16).to_slice())
        .map(|s: &str| Token::HexInt(u64::from_str_radix(s, 16).unwrap()));

    let dec_int = text::int::<_, LexExtra<'_>>(10)
        .to_slice()
        .map(|s: &str| Token::Int(s.parse::<u64>().unwrap()));

    // Quoted string with escape sequences
    let escape = just('\\').ignore_then(choice((
        just('\\').to('\\'),
        just('"').to('"'),
        just('n').to('\n'),
        just('r').to('\r'),
        just('t').to('\t'),
    )));
    let string = just('"')
        .ignore_then(none_of("\\\"").or(escape).repeated().to_slice())
        .then_ignore(just('"'))
        .map(Token::Str);

    // Identifiers, keywords, and prefixed IDs
    // Accept underscores in identifiers (for keywords like slice_data, may_fail, etc.)
    let ident_or_kw = text::ident::<_, LexExtra<'_>>().to_slice().map(|s: &str| {
        // First try prefixed IDs
        if let Some(id_token) = try_prefixed_id(s) {
            return id_token;
        }
        // Then keywords
        keyword_or_ident(s)
    });

    // Punctuation — order matters, longer tokens first
    let punct = choice((
        just("::").to(Token::ColonColon),
        just('{').to(Token::LBrace),
        just('}').to(Token::RBrace),
        just('[').to(Token::LBracket),
        just(']').to(Token::RBracket),
        just('(').to(Token::LParen),
        just(')').to(Token::RParen),
        just('<').to(Token::LAngle),
        just('>').to(Token::RAngle),
        just(':').to(Token::Colon),
        just(',').to(Token::Comma),
        just('=').to(Token::Eq),
        just('@').to(Token::At),
        just('&').to(Token::Amp),
        just('-').to(Token::Minus),
    ));

    let token = choice((hex_int, dec_int, string, ident_or_kw, punct));

    token
        .map_with(|tok, e| (tok, e.span()))
        .padded()
        .recover_with(skip_then_retry_until(any().ignored(), end()))
        .repeated()
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lex_simple_hir() {
        let src = r#"hir_module { regions [] }"#;
        let (tokens, errs) = lexer().parse(src).into_output_errors();
        assert!(errs.is_empty(), "lex errors: {errs:?}");
        let tokens = tokens.unwrap();
        let kinds: Vec<_> = tokens.iter().map(|(t, _)| t.clone()).collect();
        assert_eq!(
            kinds,
            vec![
                Token::KwHirModule,
                Token::LBrace,
                Token::KwRegions,
                Token::LBracket,
                Token::RBracket,
                Token::RBrace,
            ]
        );
    }

    #[test]
    fn lex_prefixed_ids() {
        let src = "l0 l42 r1 sc5 stmt99 t0 c1 f3 store2";
        let (tokens, errs) = lexer().parse(src).into_output_errors();
        assert!(errs.is_empty(), "lex errors: {errs:?}");
        let tokens = tokens.unwrap();
        let kinds: Vec<_> = tokens.iter().map(|(t, _)| t.clone()).collect();
        assert_eq!(
            kinds,
            vec![
                Token::LocalId(0),
                Token::LocalId(42),
                Token::RegionId(1),
                Token::ScopeId(5),
                Token::StmtId(99),
                Token::TypeDefId(0),
                Token::CallableId(1),
                Token::FunctionId(3),
                Token::StoreId(2),
            ]
        );
    }

    #[test]
    fn lex_string_with_escapes() {
        let src = r#""hello \"world\"""#;
        let (tokens, errs) = lexer().parse(src).into_output_errors();
        assert!(errs.is_empty(), "lex errors: {errs:?}");
        let tokens = tokens.unwrap();
        assert_eq!(tokens.len(), 1);
        // The Str token contains the raw slice including escape sequences
        matches!(&tokens[0].0, Token::Str(_));
    }

    #[test]
    fn lex_hex_and_decimal() {
        let src = "42 0xff 0x1a";
        let (tokens, errs) = lexer().parse(src).into_output_errors();
        assert!(errs.is_empty(), "lex errors: {errs:?}");
        let tokens = tokens.unwrap();
        let kinds: Vec<_> = tokens.iter().map(|(t, _)| t.clone()).collect();
        assert_eq!(
            kinds,
            vec![Token::Int(42), Token::HexInt(0xff), Token::HexInt(0x1a),]
        );
    }

    #[test]
    fn lex_keywords_vs_idents() {
        let src = "init assign my_custom_name";
        let (tokens, errs) = lexer().parse(src).into_output_errors();
        assert!(errs.is_empty(), "lex errors: {errs:?}");
        let tokens = tokens.unwrap();
        let kinds: Vec<_> = tokens.iter().map(|(t, _)| t.clone()).collect();
        assert_eq!(
            kinds,
            vec![
                Token::KwInit,
                Token::KwAssign,
                Token::Ident("my_custom_name"),
            ]
        );
    }

    #[test]
    fn lex_full_repro_hir() {
        let src = r#"
hir_module {
  regions []
  stores []
  types []
  callables []
  functions [
    function f0 "add" {
      regions []
      stores []
      params [
        l0 param "a": u64
        l1 param "b": u64
        l2 destination "out": u64
      ]
      locals [
        l3 let "sum": u64
      ]
      return unit
      scopes [
        scope sc0 parent none comment "root"
      ]
      body @sc0 {
        stmt0: init l3 = binary add(l0, l1)
        stmt1: init l2 = l3
        stmt2: return
      }
    }
  ]
}
"#;
        let (tokens, errs) = lexer().parse(src).into_output_errors();
        assert!(errs.is_empty(), "lex errors: {errs:?}");
        let tokens = tokens.unwrap();
        // Just check we got a reasonable number of tokens and no errors
        assert!(
            tokens.len() > 50,
            "expected many tokens, got {}",
            tokens.len()
        );
        // Check some specific tokens
        assert_eq!(tokens[0].0, Token::KwHirModule);
        assert_eq!(tokens[1].0, Token::LBrace);
    }
}
