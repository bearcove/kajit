use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};

use facet::Facet;
use facet_styx::RenderError;

pub fn generate_repr_poc(workspace_root: &Path) -> Result<Vec<PathBuf>, String> {
    let schema_path = workspace_root.join("notes/unified-ast/pilot/hir.repr.styx");
    let schema_source = fs::read_to_string(&schema_path)
        .map_err(|e| format!("failed to read {}: {e}", schema_path.display()))?;
    let schema: PilotSchemaDocument = facet_styx::from_str(&schema_source)
        .map_err(|e| {
            format!(
                "failed to parse {} as Styx\n{}",
                schema_path.display(),
                e.render(&schema_path.display().to_string(), &schema_source)
            )
        })?;
    let repr = validate_hir_pilot_schema(&schema, &schema_path)?;

    let out_dir = workspace_root.join("crates-kajit/kajit-reprs/src/schema_poc");
    fs::create_dir_all(&out_dir)
        .map_err(|e| format!("failed to create {}: {e}", out_dir.display()))?;

    let mod_path = out_dir.join("mod.rs");
    fs::write(&mod_path, "pub mod hir;\n")
        .map_err(|e| format!("failed to write {}: {e}", mod_path.display()))?;

    let hir_path = out_dir.join("hir.rs");
    fs::write(&hir_path, render_hir_poc_module(repr))
        .map_err(|e| format!("failed to write {}: {e}", hir_path.display()))?;

    Ok(vec![mod_path, hir_path])
}

#[derive(Facet, Debug)]
#[allow(dead_code)]
struct PilotSchemaDocument {
    meta: PilotMeta,
    repr: ReprDecl,
}

#[derive(Facet, Debug)]
#[allow(dead_code)]
struct PilotMeta {
    id: String,
    version: u64,
    description: String,
}

#[derive(Facet, Debug)]
#[allow(dead_code)]
#[facet(rename_all = "snake_case")]
#[repr(u8)]
enum ReprDecl {
    Module(ReprBody),
}

#[derive(Facet, Debug)]
#[allow(dead_code)]
struct ReprBody {
    name: String,
    file_ext: String,
    contract: ReprContract,
    syntax: ReprSyntax,
    common: Option<HashMap<String, TypeUse>>,
    nodes: Option<HashMap<String, NodeDecl>>,
}

#[derive(Facet, Debug)]
#[allow(dead_code)]
struct ReprContract {
    purpose: String,
    canonical_identities: Vec<String>,
    round_trip: String,
    provenance: String,
}

#[derive(Facet, Debug)]
#[allow(dead_code)]
struct ReprSyntax {
    tokens: HashMap<String, TokenExpr>,
    rules: HashMap<String, RuleExpr>,
    canonical_print: HashMap<String, String>,
}

#[derive(Facet, Debug)]
#[allow(dead_code)]
#[facet(rename_all = "lowercase")]
#[repr(u8)]
enum TokenExpr {
    Regex(Vec<String>),
    #[facet(other)]
    Other {
        #[facet(tag)]
        name: Option<String>,
        #[facet(content)]
        content: Option<ExprPayload>,
    },
}

#[derive(Facet, Debug)]
#[allow(dead_code)]
#[repr(u8)]
enum RuleExpr {
    #[facet(other)]
    Form {
        #[facet(tag)]
        tag: Option<String>,
        #[facet(content)]
        content: Option<ExprPayload>,
    },
}

#[derive(Facet, Debug)]
#[allow(dead_code)]
#[facet(untagged)]
#[repr(u8)]
enum ExprPayload {
    Scalar(String),
    Seq(Vec<RuleExpr>),
    #[facet(other)]
    Other {
        #[facet(tag)]
        name: Option<String>,
        #[facet(content)]
        content: Option<Box<ExprPayload>>,
    },
}

#[derive(Facet, Debug)]
#[allow(dead_code)]
#[repr(u8)]
enum TypeUse {
    #[facet(other)]
    Form {
        #[facet(tag)]
        tag: Option<String>,
        #[facet(content)]
        content: Option<TypeUsePayload>,
    },
}

#[derive(Facet, Debug)]
#[allow(dead_code)]
#[facet(untagged)]
#[repr(u8)]
enum TypeUsePayload {
    Scalar(String),
    Seq(Vec<TypeUse>),
}

#[derive(Facet, Debug)]
#[allow(dead_code)]
#[facet(rename_all = "lowercase")]
#[repr(u8)]
enum NodeDecl {
    Node(NodeFields),
    Enum(NodeVariants),
    Struct(NodeFields),
    #[facet(other)]
    Other {
        #[facet(tag)]
        tag: Option<String>,
        #[facet(content)]
        content: Option<TypeUsePayload>,
    },
}

#[derive(Facet, Debug)]
#[allow(dead_code)]
struct NodeFields {
    #[facet(flatten)]
    fields: HashMap<String, TypeUse>,
}

#[derive(Facet, Debug)]
#[allow(dead_code)]
struct NodeVariants {
    #[facet(flatten)]
    variants: HashMap<String, NodeDecl>,
}

fn validate_hir_pilot_schema<'a>(schema: &'a PilotSchemaDocument, path: &Path) -> Result<&'a ReprBody, String> {
    if schema.meta.id != "kajit:repr-schema/hir-pilot" {
        return Err(format!(
            "expected {} meta.id to be kajit:repr-schema/hir-pilot, got {:?}",
            path.display(),
            schema.meta.id
        ));
    }

    if schema.meta.version == 0 {
        return Err(format!(
            "expected {} meta.version to be non-zero",
            path.display()
        ));
    }

    if schema.meta.description.trim().is_empty() {
        return Err(format!(
            "expected {} meta.description to be non-empty",
            path.display()
        ));
    }

    let ReprDecl::Module(repr) = &schema.repr;

    if repr.name != "HIR" {
        return Err(format!(
            "expected {} repr name to be HIR, got {:?}",
            path.display(),
            repr.name
        ));
    }

    if repr.file_ext != ".vixen-hir" {
        return Err(format!(
            "expected {} file_ext to be .vixen-hir, got {:?}",
            path.display(),
            repr.file_ext
        ));
    }

    if repr.contract.purpose.trim().is_empty() {
        return Err(format!(
            "expected {} contract.purpose to be non-empty",
            path.display()
        ));
    }

    if repr.contract.round_trip != "canonical-print" {
        return Err(format!(
            "expected {} round_trip to be canonical-print, got {:?}",
            path.display(),
            repr.contract.round_trip
        ));
    }

    if repr.contract.provenance != "required" {
        return Err(format!(
            "expected {} provenance to be required, got {:?}",
            path.display(),
            repr.contract.provenance
        ));
    }

    if repr.contract.canonical_identities.is_empty() {
        return Err(format!(
            "expected {} canonical_identities to be non-empty",
            path.display()
        ));
    }

    for rule_name in ["Module", "Function", "Param", "Block", "Stmt", "Expr"] {
        if !repr.syntax.rules.contains_key(rule_name) {
            return Err(format!(
                "expected {} syntax.rules to contain {:?}",
                path.display(),
                rule_name
            ));
        }
    }

    for print_name in ["Module", "Function", "Stmt.Return", "Expr.Call"] {
        if repr
            .syntax
            .canonical_print
            .get(print_name)
            .is_none_or(|s| s.trim().is_empty())
        {
            return Err(format!(
                "expected {} canonical_print to contain non-empty {:?}",
                path.display(),
                print_name
            ));
        }
    }

    for token_name in ["ident", "symbol", "int"] {
        let Some(token_spec) = repr.syntax.tokens.get(token_name) else {
            return Err(format!(
                "expected {} syntax.tokens to contain {:?}",
                path.display(),
                token_name
            ));
        };

        match token_spec {
            TokenExpr::Regex(patterns) if !patterns.is_empty() && !patterns[0].is_empty() => {}
            TokenExpr::Regex(_) => {
                return Err(format!(
                    "expected {} syntax.tokens.{token_name} regex payload to be non-empty",
                    path.display()
                ));
            }
            TokenExpr::Other { name, .. } => {
                return Err(format!(
                    "expected {} syntax.tokens.{token_name} to be @regex(...), got {:?}",
                    path.display(),
                    name
                ));
            }
        }
    }

    expect_tagged_rule(path, &repr.syntax.rules, "Module", "seq")?;
    expect_tagged_rule(path, &repr.syntax.rules, "Function", "seq")?;
    expect_tagged_rule(path, &repr.syntax.rules, "Param", "seq")?;
    expect_tagged_rule(path, &repr.syntax.rules, "Block", "seq")?;
    expect_tagged_rule(path, &repr.syntax.rules, "Stmt", "choice")?;
    expect_tagged_rule(path, &repr.syntax.rules, "Expr", "choice")?;

    if repr.common.is_none() {
        return Err(format!("expected {} repr.common to be present", path.display()));
    }

    if repr.nodes.is_none() {
        return Err(format!("expected {} repr.nodes to be present", path.display()));
    }

    Ok(repr)
}

fn expect_tagged_rule(
    path: &Path,
    rules: &HashMap<String, RuleExpr>,
    rule_name: &str,
    expected_tag: &str,
) -> Result<(), String> {
    let Some(rule) = rules.get(rule_name) else {
        return Err(format!(
            "expected {} syntax.rules to contain {:?}",
            path.display(),
            rule_name
        ));
    };

    let actual_tag = match rule {
        RuleExpr::Form { tag, .. } => tag.as_deref().unwrap_or("literal"),
    };

    if actual_tag != expected_tag {
        return Err(format!(
            "expected {} syntax.rules.{rule_name} to be @{expected_tag}(...), got {actual_tag:?}",
            path.display()
        ));
    }

    Ok(())
}

fn render_hir_poc_module(repr: &ReprBody) -> String {
    let raw = format!(
        r###"
// @generated by kajit-foundation::generate_repr_poc from {file_ext} schema {name}.
// Do not edit manually.

use std::fmt;

use chumsky::prelude::*;
use kajit_types::SymbolName;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Module {{
    pub functions: Vec<Function>,
}}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Function {{
    pub name: String,
    pub params: Vec<Param>,
    pub return_type: Type,
    pub body: Block,
}}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Param {{
    pub name: String,
    pub ty: Type,
}}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Block {{
    pub statements: Vec<Stmt>,
}}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Stmt {{
    Return(Option<Expr>),
    Expr(Expr),
}}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Expr {{
    Local(String),
    Literal(u64),
    Call {{ callee: SymbolName, args: Vec<Expr> }},
}}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Type {{
    Unit,
    Named(String),
}}

impl fmt::Display for Module {{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {{
        writeln!(f, "module {{{{")?;
        for function in &self.functions {{
            writeln!(f, "{{}}", DisplayIndented(function, 1))?;
        }}
        write!(f, "}}}}")
    }}
}}

struct DisplayIndented<'a, T>(&'a T, usize);

fn indent(level: usize) -> String {{
    "  ".repeat(level)
}}

impl fmt::Display for DisplayIndented<'_, Function> {{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {{
        let pad = indent(self.1);
        write!(f, "{{pad}}fn {{}}(", self.0.name)?;
        for (index, param) in self.0.params.iter().enumerate() {{
            if index > 0 {{
                write!(f, ", ")?;
            }}
            write!(f, "{{param}}")?;
        }}
        writeln!(f, ") -> {{}} {{{{", self.0.return_type)?;
        for stmt in &self.0.body.statements {{
            writeln!(f, "{{}}", DisplayIndented(stmt, self.1 + 1))?;
        }}
        write!(f, "{{pad}}}}}}")
    }}
}}

impl fmt::Display for DisplayIndented<'_, Stmt> {{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {{
        let pad = indent(self.1);
        match self.0 {{
            Stmt::Return(Some(expr)) => write!(f, "{{pad}}return {{expr}}"),
            Stmt::Return(None) => write!(f, "{{pad}}return"),
            Stmt::Expr(expr) => write!(f, "{{pad}}{{expr}}"),
        }}
    }}
}}

impl fmt::Display for Param {{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {{
        write!(f, "{{}}: {{}}", self.name, self.ty)
    }}
}}

impl fmt::Display for Expr {{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {{
        match self {{
            Expr::Local(name) => write!(f, "{{name}}"),
            Expr::Literal(value) => write!(f, "{{value}}"),
            Expr::Call {{ callee, args }} => {{
                write!(f, "call @{{callee}}(")?;
                for (index, arg) in args.iter().enumerate() {{
                    if index > 0 {{
                        write!(f, ", ")?;
                    }}
                    write!(f, "{{arg}}")?;
                }}
                write!(f, ")")
            }}
        }}
    }}
}}

impl fmt::Display for Type {{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {{
        match self {{
            Type::Unit => write!(f, "unit"),
            Type::Named(name) => write!(f, "{{name}}"),
        }}
    }}
}}

type ParseError<'src> = extra::Err<Rich<'src, char>>;

fn ws<'src>() -> impl Parser<'src, &'src str, (), ParseError<'src>> + Clone {{
    text::whitespace().ignored().repeated()
}}

fn ident<'src>() -> impl Parser<'src, &'src str, String, ParseError<'src>> + Clone {{
    text::ident().map(str::to_owned).padded_by(ws())
}}

fn symbol<'src>() -> impl Parser<'src, &'src str, SymbolName, ParseError<'src>> + Clone {{
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
}}

fn uint<'src>() -> impl Parser<'src, &'src str, u64, ParseError<'src>> + Clone {{
    text::int(10).from_str().unwrapped().padded_by(ws())
}}

fn ty<'src>() -> impl Parser<'src, &'src str, Type, ParseError<'src>> + Clone {{
    ident().map(|name| {{
        if name == "unit" {{
            Type::Unit
        }} else {{
            Type::Named(name)
        }}
    }})
}}

fn expr<'src>() -> impl Parser<'src, &'src str, Expr, ParseError<'src>> + Clone {{
    recursive(|expr| {{
        let call = just("call")
            .padded_by(ws())
            .ignore_then(symbol())
            .then(
                expr.clone()
                    .separated_by(just(',').padded_by(ws()))
                    .collect::<Vec<_>>()
                    .delimited_by(just('(').padded_by(ws()), just(')').padded_by(ws())),
            )
            .map(|(callee, args)| Expr::Call {{ callee, args }});

        choice((call, uint().map(Expr::Literal), ident().map(Expr::Local)))
    }})
}}

fn stmt<'src>() -> impl Parser<'src, &'src str, Stmt, ParseError<'src>> + Clone {{
    let ret = just("return")
        .padded_by(ws())
        .ignore_then(expr().or_not())
        .map(Stmt::Return);
    choice((ret, expr().map(Stmt::Expr))).padded_by(ws())
}}

fn block<'src>() -> impl Parser<'src, &'src str, Block, ParseError<'src>> + Clone {{
    stmt()
        .repeated()
        .collect::<Vec<_>>()
        .delimited_by(just('{{').padded_by(ws()), just('}}').padded_by(ws()))
        .map(|statements| Block {{ statements }})
}}

fn param<'src>() -> impl Parser<'src, &'src str, Param, ParseError<'src>> + Clone {{
    ident()
        .then_ignore(just(':').padded_by(ws()))
        .then(ty())
        .map(|(name, ty)| Param {{ name, ty }})
}}

fn function<'src>() -> impl Parser<'src, &'src str, Function, ParseError<'src>> + Clone {{
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
        .map(|(((name, params), return_type), body)| Function {{
            name,
            params,
            return_type,
            body,
        }})
}}

pub fn parser<'src>() -> impl Parser<'src, &'src str, Module, ParseError<'src>> + Clone {{
    just("module")
        .padded_by(ws())
        .ignore_then(
            function()
                .repeated()
                .collect::<Vec<_>>()
                .delimited_by(just('{{').padded_by(ws()), just('}}').padded_by(ws())),
        )
        .then_ignore(end())
        .map(|functions| Module {{ functions }})
}}

pub fn parse_module(source: &str) -> Result<Module, String> {{
    let (module, errors) = parser().parse(source).into_output_errors();
    if errors.is_empty() {{
        module.ok_or_else(|| "parser produced no module".to_owned())
    }} else {{
        Err(errors
            .into_iter()
            .map(|error| error.to_string())
            .collect::<Vec<_>>()
            .join("\n"))
    }}
}}

#[cfg(test)]
mod tests {{
    use super::*;

    #[test]
    fn round_trips_simple_module() {{
        let text = r#"
module {{
  fn decode(cursor: Cursor, out: Record) -> unit {{
    call @postcard.read_option_tag(cursor)
    return call @postcard.read_str(cursor)
  }}
}}
"#;

        let module = parse_module(text).expect("pilot HIR should parse");
        let printed = module.to_string();
        let reparsed = parse_module(&printed).expect("printed pilot HIR should parse");
        assert_eq!(module, reparsed);
    }}
}}
"###,
        file_ext = repr.file_ext,
        name = repr.name,
    );

    let file = syn::parse_file(&raw).expect("generated HIR PoC module should parse");
    prettyplease::unparse(&file)
}
