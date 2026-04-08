use std::fmt;
use chumsky::prelude::*;
use kajit_types::SymbolName;
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Module {
    pub functions: Vec<Function>,
}
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Function {
    pub name: String,
    pub params: Vec<Param>,
    pub return_type: Type,
    pub body: Block,
}
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Param {
    pub name: String,
    pub ty: Type,
}
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Block {
    pub statements: Vec<Stmt>,
}
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Stmt {
    Return(Option<Expr>),
    Expr(Expr),
}
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Expr {
    Local(String),
    Literal(u64),
    Call { callee: SymbolName, args: Vec<Expr> },
}
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Type {
    Unit,
    Named(String),
}
impl fmt::Display for Module {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "module {{")?;
        for function in &self.functions {
            writeln!(f, "{}", DisplayIndented(function, 1))?;
        }
        write!(f, "}}")
    }
}
struct DisplayIndented<'a, T>(&'a T, usize);
fn indent(level: usize) -> String {
    "  ".repeat(level)
}
impl fmt::Display for DisplayIndented<'_, Function> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let pad = indent(self.1);
        write!(f, "{pad}fn {}(", self.0.name)?;
        for (index, param) in self.0.params.iter().enumerate() {
            if index > 0 {
                write!(f, ", ")?;
            }
            write!(f, "{param}")?;
        }
        writeln!(f, ") -> {} {{", self.0.return_type)?;
        for stmt in &self.0.body.statements {
            writeln!(f, "{}", DisplayIndented(stmt, self.1 + 1))?;
        }
        write!(f, "{pad}}}")
    }
}
impl fmt::Display for DisplayIndented<'_, Stmt> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let pad = indent(self.1);
        match self.0 {
            Stmt::Return(Some(expr)) => write!(f, "{pad}return {expr}"),
            Stmt::Return(None) => write!(f, "{pad}return"),
            Stmt::Expr(expr) => write!(f, "{pad}{expr}"),
        }
    }
}
impl fmt::Display for Param {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}: {}", self.name, self.ty)
    }
}
impl fmt::Display for Expr {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Expr::Local(name) => write!(f, "{name}"),
            Expr::Literal(value) => write!(f, "{value}"),
            Expr::Call { callee, args } => {
                write!(f, "call @{callee}(")?;
                for (index, arg) in args.iter().enumerate() {
                    if index > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{arg}")?;
                }
                write!(f, ")")
            }
        }
    }
}
impl fmt::Display for Type {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Type::Unit => write!(f, "unit"),
            Type::Named(name) => write!(f, "{name}"),
        }
    }
}
type ParseError<'src> = extra::Err<Rich<'src, char>>;
fn ws<'src>() -> impl Parser<'src, &'src str, (), ParseError<'src>> + Clone {
    text::whitespace().ignored().repeated()
}
fn ident<'src>() -> impl Parser<'src, &'src str, String, ParseError<'src>> + Clone {
    text::ident().map(str::to_owned).padded_by(ws())
}
fn symbol<'src>() -> impl Parser<'src, &'src str, SymbolName, ParseError<'src>> + Clone {
    just('@')
        .ignore_then(
            any()
                .filter(|c: &char| c.is_alphanumeric() || *c == '_' || *c == '.')
                .repeated()
                .at_least(1)
                .to_slice(),
        )
        .map(|name: &str| SymbolName::new(name.to_owned()))
        .padded_by(ws())
}
fn uint<'src>() -> impl Parser<'src, &'src str, u64, ParseError<'src>> + Clone {
    text::int(10).from_str().unwrapped().padded_by(ws())
}
fn ty<'src>() -> impl Parser<'src, &'src str, Type, ParseError<'src>> + Clone {
    ident().map(|name| { if name == "unit" { Type::Unit } else { Type::Named(name) } })
}
fn expr<'src>() -> impl Parser<'src, &'src str, Expr, ParseError<'src>> + Clone {
    recursive(|expr| {
        let call = just("call")
            .padded_by(ws())
            .ignore_then(symbol())
            .then(
                expr
                    .clone()
                    .separated_by(just(',').padded_by(ws()))
                    .collect::<Vec<_>>()
                    .delimited_by(just('(').padded_by(ws()), just(')').padded_by(ws())),
            )
            .map(|(callee, args)| Expr::Call { callee, args });
        choice((call, uint().map(Expr::Literal), ident().map(Expr::Local)))
    })
}
fn stmt<'src>() -> impl Parser<'src, &'src str, Stmt, ParseError<'src>> + Clone {
    let ret = just("return")
        .padded_by(ws())
        .ignore_then(expr().or_not())
        .map(Stmt::Return);
    choice((ret, expr().map(Stmt::Expr))).padded_by(ws())
}
fn block<'src>() -> impl Parser<'src, &'src str, Block, ParseError<'src>> + Clone {
    stmt()
        .repeated()
        .collect::<Vec<_>>()
        .delimited_by(just('{').padded_by(ws()), just('}').padded_by(ws()))
        .map(|statements| Block { statements })
}
fn param<'src>() -> impl Parser<'src, &'src str, Param, ParseError<'src>> + Clone {
    ident()
        .then_ignore(just(':').padded_by(ws()))
        .then(ty())
        .map(|(name, ty)| Param { name, ty })
}
fn function<'src>() -> impl Parser<'src, &'src str, Function, ParseError<'src>> + Clone {
    just("fn")
        .padded_by(ws())
        .ignore_then(ident())
        .then(
            param()
                .separated_by(just(',').padded_by(ws()))
                .collect::<Vec<_>>()
                .delimited_by(just('(').padded_by(ws()), just(')').padded_by(ws())),
        )
        .then_ignore(just("->").padded_by(ws()))
        .then(ty())
        .then(block())
        .map(|(((name, params), return_type), body)| Function {
            name,
            params,
            return_type,
            body,
        })
}
pub fn parser<'src>() -> impl Parser<'src, &'src str, Module, ParseError<'src>> + Clone {
    just("module")
        .padded_by(ws())
        .ignore_then(
            function()
                .repeated()
                .collect::<Vec<_>>()
                .delimited_by(just('{').padded_by(ws()), just('}').padded_by(ws())),
        )
        .then_ignore(end())
        .map(|functions| Module { functions })
}
pub fn parse_module(source: &str) -> Result<Module, String> {
    let (module, errors) = parser().parse(source).into_output_errors();
    if errors.is_empty() {
        module.ok_or_else(|| "parser produced no module".to_owned())
    } else {
        Err(
            errors
                .into_iter()
                .map(|error| error.to_string())
                .collect::<Vec<_>>()
                .join("\n"),
        )
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn round_trips_simple_module() {
        let text = r#"
module {
  fn decode(cursor: Cursor, out: Record) -> unit {
    call @postcard.read_option_tag(cursor)
    return call @postcard.read_str(cursor)
  }
}
"#;
        let module = parse_module(text).expect("pilot HIR should parse");
        let printed = module.to_string();
        let reparsed = parse_module(&printed).expect("printed pilot HIR should parse");
        assert_eq!(module, reparsed);
    }
}
