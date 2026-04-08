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
        content: Option<Vec<String>>,
    },
}

#[derive(Facet, Debug)]
#[allow(dead_code)]
#[facet(rename_all = "lowercase")]
#[repr(u8)]
enum RuleExpr {
    Seq(Vec<RuleExpr>),
    Choice(Vec<RuleExpr>),
    Field(RuleNamed),
    Variant(RuleNamed),
    Ref(Vec<String>),
    Token(Vec<String>),
    Optional(Vec<RuleExpr>),
    Repeat(Vec<RuleExpr>),
    #[facet(other)]
    Literal(Option<String>),
}

#[derive(Facet, Debug)]
#[allow(dead_code)]
#[repr(transparent)]
struct RuleNamed((String, Box<RuleExpr>));

#[derive(Facet, Debug)]
#[allow(dead_code)]
#[facet(rename_all = "lowercase")]
#[repr(u8)]
enum TypeUse {
    Optional(Vec<TypeUse>),
    Seq(Vec<TypeUse>),
    #[facet(other)]
    Ref {
        #[facet(tag)]
        name: Option<String>,
    },
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
        content: Option<Vec<TypeUse>>,
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

    let actual_tag = rule_expr_kind(rule);

    if actual_tag != expected_tag {
        return Err(format!(
            "expected {} syntax.rules.{rule_name} to be @{expected_tag}(...), got {actual_tag:?}",
            path.display()
        ));
    }

    Ok(())
}

fn rule_expr_kind(rule: &RuleExpr) -> &'static str {
    match rule {
        RuleExpr::Seq(_) => "seq",
        RuleExpr::Choice(_) => "choice",
        RuleExpr::Field(_) => "field",
        RuleExpr::Variant(_) => "variant",
        RuleExpr::Ref(_) => "ref",
        RuleExpr::Token(_) => "token",
        RuleExpr::Optional(_) => "optional",
        RuleExpr::Repeat(_) => "repeat",
        RuleExpr::Literal(_) => "literal",
    }
}

fn rust_ident(name: &str) -> String {
    match name {
        "else" | "type" | "struct" | "enum" | "fn" | "mod" | "move" | "ref" | "self"
        | "Self" | "crate" | "super" | "use" | "where" | "loop" | "match" | "return"
        | "pub" | "in" | "let" | "impl" | "trait" | "const" | "static" | "async"
        | "await" | "dyn" => format!("r#{name}"),
        _ => name.to_owned(),
    }
}

fn collect_type_tags(ty: &TypeUse, out: &mut Vec<String>) {
    match ty {
        TypeUse::Ref { name: Some(tag) } => out.push(tag.clone()),
        TypeUse::Ref { name: None } => {}
        TypeUse::Optional(items) | TypeUse::Seq(items) => {
            for item in items {
                collect_type_tags(item, out);
            }
        }
    }
}

fn render_type_use(ty: &TypeUse, node_names: &[String], box_node_refs: bool) -> String {
    match ty {
        TypeUse::Optional(items) if items.len() == 1 => {
            format!("Option<{}>", render_type_use(&items[0], node_names, true))
        }
        TypeUse::Seq(items) if items.len() == 1 => {
            format!("Vec<{}>", render_type_use(&items[0], node_names, false))
        }
        TypeUse::Ref { name: Some(tag) } => {
            if box_node_refs && node_names.iter().any(|name| name == tag) {
                format!("Box<{tag}>")
            } else {
                tag.to_owned()
            }
        }
        TypeUse::Ref { name: None } => "String".to_owned(),
        _ => "UnsupportedTypeUse".to_owned(),
    }
}

fn type_use_tag(ty: &TypeUse) -> Option<&str> {
    match ty {
        TypeUse::Ref { name: Some(tag) } => Some(tag.as_str()),
        _ => None,
    }
}

fn node_fields_have_prov(fields: &NodeFields, provenance_tag: &str) -> bool {
    fields
        .fields
        .get("prov")
        .is_some_and(|ty| type_use_tag(ty) == Some(provenance_tag))
}

fn rule_named_parts(named: &RuleNamed) -> (&str, &RuleExpr) {
    (named.0.0.as_str(), &named.0.1)
}

fn rule_literal_text(rule: &RuleExpr) -> Option<&str> {
    match rule {
        RuleExpr::Literal(Some(text)) => Some(text.as_str()),
        _ => None,
    }
}

fn rule_ref_name(rule: &RuleExpr) -> Option<&str> {
    match rule {
        RuleExpr::Ref(names) if names.len() == 1 => Some(names[0].as_str()),
        _ => None,
    }
}

fn rule_token_name(rule: &RuleExpr) -> Option<&str> {
    match rule {
        RuleExpr::Token(names) if names.len() == 1 => Some(names[0].as_str()),
        _ => None,
    }
}

fn rule_repeat_parts(rule: &RuleExpr) -> Option<(&RuleExpr, Option<&str>)> {
    match rule {
        RuleExpr::Repeat(items) if !items.is_empty() => {
            let sep = if items.len() >= 3 && rule_literal_text(&items[1]) == Some("sep=") {
                rule_literal_text(&items[2])
            } else {
                None
            };
            Some((&items[0], sep))
        }
        _ => None,
    }
}

fn inner_type_use(ty: &TypeUse) -> Option<&TypeUse> {
    match ty {
        TypeUse::Optional(items) | TypeUse::Seq(items) if items.len() == 1 => Some(&items[0]),
        _ => None,
    }
}

fn render_default_value(ty: &TypeUse, provenance_tag: &str) -> String {
    match ty {
        TypeUse::Optional(_) => "None".to_owned(),
        TypeUse::Seq(_) => "Vec::new()".to_owned(),
        TypeUse::Ref { name: Some(tag) } if tag == provenance_tag => {
            format!("{provenance_tag}::default()")
        }
        TypeUse::Ref { name: Some(tag) } => format!("{tag}::default()"),
        TypeUse::Ref { name: None } => "String::new()".to_owned(),
    }
}

fn render_token_value_parser(token_name: &str, ty: &TypeUse) -> Result<String, String> {
    let parser = match (token_name, type_use_tag(ty)) {
        ("ident", Some("Symbol")) => "ident_token().map(Symbol)".to_owned(),
        ("ident", Some("Type")) => "ident_token().map(Type)".to_owned(),
        ("ident", None) => "ident_token()".to_owned(),
        ("symbol", Some("Symbol")) => "symbol_token().map(Symbol)".to_owned(),
        ("symbol", None) => "symbol_token()".to_owned(),
        ("int", Some("Literal")) => "int_token().map(Literal)".to_owned(),
        ("int", None) => "int_token()".to_owned(),
        _ => {
            return Err(format!(
                "unsupported token parser mapping for token {token_name:?} and type {:?}",
                type_use_tag(ty)
            ));
        }
    };
    Ok(parser)
}

fn render_ref_value_parser(
    ref_name: &str,
    ty: &TypeUse,
    rule_names: &[String],
    node_names: &[String],
    box_node_refs: bool,
) -> Result<String, String> {
    let base = if rule_names.iter().any(|name| name == ref_name) {
        format!("{}_parser.clone()", snake_case(ref_name))
    } else {
        match ref_name {
            "Type" => "ident_token().map(Type)".to_owned(),
            _ => return Err(format!("unsupported reference parser for {ref_name:?}")),
        }
    };

    let parser = match ty {
        TypeUse::Ref { name: Some(tag) }
            if box_node_refs && node_names.iter().any(|name| name == tag) =>
        {
            format!("({base}).map(Box::new)")
        }
        TypeUse::Ref { .. } => base,
        _ => base,
    };
    Ok(parser)
}

fn render_value_parser(
    rule: &RuleExpr,
    ty: &TypeUse,
    rule_names: &[String],
    node_names: &[String],
    box_node_refs: bool,
) -> Result<String, String> {
    match rule {
        RuleExpr::Token(_) => render_token_value_parser(
            rule_token_name(rule).ok_or_else(|| "malformed token rule".to_owned())?,
            ty,
        ),
        RuleExpr::Ref(_) => render_ref_value_parser(
            rule_ref_name(rule).ok_or_else(|| "malformed ref rule".to_owned())?,
            ty,
            rule_names,
            node_names,
            box_node_refs,
        ),
        RuleExpr::Optional(items) if items.len() == 1 => {
            let inner_ty =
                inner_type_use(ty).ok_or_else(|| "optional rule without optional type".to_owned())?;
            let inner = render_value_parser(&items[0], inner_ty, rule_names, node_names, true)?;
            Ok(format!("({inner}).or_not()"))
        }
        RuleExpr::Repeat(_) => {
            let (inner_rule, sep) =
                rule_repeat_parts(rule).ok_or_else(|| "malformed repeat rule".to_owned())?;
            let inner_ty =
                inner_type_use(ty).ok_or_else(|| "repeat rule without seq type".to_owned())?;
            let inner =
                render_value_parser(inner_rule, inner_ty, rule_names, node_names, false)?;
            Ok(if let Some(sep) = sep {
                format!(
                    "({inner}).separated_by(just({sep:?}).padded()).allow_trailing().collect::<Vec<_>>()"
                )
            } else {
                format!("({inner}).repeated().collect::<Vec<_>>()")
            })
        }
        _ => Err(format!(
            "unsupported value rule kind for field type {:?}: {:?}",
            type_use_tag(ty),
            rule
        )),
    }
}

enum SeqItem {
    Ignore(String),
    Bind { name: String, parser: String },
}

fn flatten_struct_rule_items(
    rule: &RuleExpr,
    fields: &NodeFields,
    rule_names: &[String],
    node_names: &[String],
) -> Result<Vec<SeqItem>, String> {
    match rule {
        RuleExpr::Seq(items) => {
            let mut out = Vec::new();
            for item in items {
                out.extend(flatten_struct_rule_items(item, fields, rule_names, node_names)?);
            }
            Ok(out)
        }
        RuleExpr::Field(named) => {
            let (field_name, inner) = rule_named_parts(named);
            let ty = fields
                .fields
                .get(field_name)
                .ok_or_else(|| format!("schema node field {field_name:?} not found"))?;
            let parser = render_value_parser(inner, ty, rule_names, node_names, true)?;
            Ok(vec![SeqItem::Bind {
                name: field_name.to_owned(),
                parser,
            }])
        }
        RuleExpr::Literal(Some(text)) => Ok(vec![SeqItem::Ignore(format!(
            "just({text:?}).padded()"
        ))]),
        _ => Err(format!("unsupported struct rule shape: {rule:?}")),
    }
}

fn nested_tuple_pattern(names: &[String]) -> String {
    let mut iter = names.iter();
    let Some(first) = iter.next() else {
        return "()".to_owned();
    };
    let mut out = first.clone();
    for name in iter {
        out = format!("({out}, {name})");
    }
    out
}

fn render_binding_chain(items: &[SeqItem]) -> Result<(String, Vec<String>), String> {
    let mut expr: Option<String> = None;
    let mut pending_ignores = Vec::new();
    let mut names = Vec::new();

    for item in items {
        match item {
            SeqItem::Ignore(ignore) => {
                if let Some(current) = expr.take() {
                    expr = Some(format!("({current}).then_ignore({ignore})"));
                } else {
                    pending_ignores.push(ignore.clone());
                }
            }
            SeqItem::Bind { name, parser } => {
                let mut field_expr = parser.clone();
                for ignore in pending_ignores.drain(..).rev() {
                    field_expr = format!("({ignore}).ignore_then({field_expr})");
                }
                expr = Some(match expr {
                    Some(current) => format!("({current}).then({field_expr})"),
                    None => field_expr,
                });
                names.push(rust_ident(name));
            }
        }
    }

    if !pending_ignores.is_empty() {
        let mut ignored = pending_ignores[0].clone();
        for ignore in pending_ignores.iter().skip(1) {
            ignored = format!("({ignored}).ignore_then({ignore})");
        }
        expr = Some(match expr {
            Some(current) => format!("({current}).then_ignore({ignored})"),
            None => format!("({ignored}).to(())"),
        });
    }

    let expr = expr.ok_or_else(|| "cannot build empty binding chain".to_owned())?;
    Ok((expr, names))
}

fn render_struct_parser_expr(
    type_name: &str,
    fields: &NodeFields,
    rule: &RuleExpr,
    rule_names: &[String],
    node_names: &[String],
    provenance_tag: &str,
) -> Result<String, String> {
    let items = flatten_struct_rule_items(rule, fields, rule_names, node_names)?;
    let (chain, bound_names) = render_binding_chain(&items)?;
    let bound_set = bound_names.iter().cloned().collect::<std::collections::BTreeSet<_>>();

    let mut field_names = fields.fields.keys().cloned().collect::<Vec<_>>();
    field_names.sort();
    let field_rows = field_names
        .iter()
        .map(|field_name| {
            let ident = rust_ident(field_name);
            let value = if bound_set.contains(&ident) {
                ident.clone()
            } else {
                render_default_value(fields.fields.get(field_name).unwrap(), provenance_tag)
            };
            format!("{ident}: {value}")
        })
        .collect::<Vec<_>>()
        .join(", ");

    let map = if bound_names.is_empty() {
        format!(".map(|()| {type_name} {{ {field_rows} }})")
    } else if bound_names.len() == 1 {
        format!(".map(|{}| {type_name} {{ {field_rows} }})", bound_names[0])
    } else {
        format!(
            ".map(|{}| {type_name} {{ {field_rows} }})",
            nested_tuple_pattern(&bound_names)
        )
    };

    Ok(format!("({chain}){map}.boxed()"))
}

fn render_enum_parser_expr(
    enum_name: &str,
    variants: &NodeVariants,
    rule: &RuleExpr,
    rule_names: &[String],
    node_names: &[String],
    provenance_tag: &str,
) -> Result<String, String> {
    let RuleExpr::Choice(items) = rule else {
        return Err(format!("enum rule for {enum_name} must be choice"));
    };

    let mut variant_parsers = Vec::new();
    for item in items {
        let RuleExpr::Variant(named) = item else {
            return Err(format!("enum rule for {enum_name} contains non-variant item"));
        };
        let (variant_name, inner_rule) = rule_named_parts(named);
        let Some(NodeDecl::Struct(fields) | NodeDecl::Node(fields)) = variants.variants.get(variant_name) else {
            return Err(format!(
                "schema enum {enum_name} is missing variant declaration {variant_name:?}"
            ));
        };
        let items = flatten_struct_rule_items(inner_rule, fields, rule_names, node_names)?;
        let (chain, bound_names) = render_binding_chain(&items)?;
        let bound_set = bound_names.iter().cloned().collect::<std::collections::BTreeSet<_>>();
        let mut field_names = fields.fields.keys().cloned().collect::<Vec<_>>();
        field_names.sort();
        let field_rows = field_names
            .iter()
            .map(|field_name| {
                let ident = rust_ident(field_name);
                let value = if bound_set.contains(&ident) {
                    ident.clone()
                } else {
                    render_default_value(fields.fields.get(field_name).unwrap(), provenance_tag)
                };
                format!("{ident}: {value}")
            })
            .collect::<Vec<_>>()
            .join(", ");
        let parser = if bound_names.is_empty() {
            format!("({chain}).map(|()| {enum_name}::{variant_name} {{ {field_rows} }})")
        } else if bound_names.len() == 1 {
            format!(
                "({chain}).map(|{}| {enum_name}::{variant_name} {{ {field_rows} }})",
                bound_names[0]
            )
        } else {
            format!(
                "({chain}).map(|{}| {enum_name}::{variant_name} {{ {field_rows} }})",
                nested_tuple_pattern(&bound_names)
            )
        };
        variant_parsers.push(parser);
    }

    Ok(format!("choice(({})).boxed()", variant_parsers.join(", ")))
}

fn render_rule_parser_expr(
    rule_name: &str,
    rule: &RuleExpr,
    decl: &NodeDecl,
    rule_names: &[String],
    node_names: &[String],
    provenance_tag: &str,
) -> Result<String, String> {
    match decl {
        NodeDecl::Node(fields) | NodeDecl::Struct(fields) => {
            render_struct_parser_expr(rule_name, fields, rule, rule_names, node_names, provenance_tag)
        }
        NodeDecl::Enum(variants) => {
            render_enum_parser_expr(rule_name, variants, rule, rule_names, node_names, provenance_tag)
        }
        NodeDecl::Other { .. } => Err(format!("unsupported node declaration for parser: {rule_name}")),
    }
}

fn render_parser_block(
    repr: &ReprBody,
    node_names: &[String],
    provenance_tag: &str,
) -> Result<String, String> {
    let rule_names = repr.syntax.rules.keys().cloned().collect::<Vec<_>>();
    let parser_order = ["Param", "Expr", "Stmt", "Block", "Function", "Module"];

    let parser_defs = parser_order
        .iter()
        .filter_map(|name| {
            let rule = repr.syntax.rules.get(*name)?;
            let decl = repr.nodes.as_ref()?.get(*name)?;
            Some(render_rule_parser_expr(
                name,
                rule,
                decl,
                &rule_names,
                node_names,
                provenance_tag,
            ))
        })
        .collect::<Result<Vec<_>, _>>()?;

    if parser_defs.len() != parser_order.len() {
        return Err("missing pilot parser rules or nodes".to_owned());
    }

    Ok(format!(
        r#"
type ParseExtra<'src> = extra::Err<Rich<'src, char>>;

fn ws<'src>() -> impl Parser<'src, &'src str, (), ParseExtra<'src>> + Clone {{
    any()
        .filter(|c: &char| c.is_whitespace())
        .repeated()
        .ignored()
}}

fn ident_token<'src>() -> impl Parser<'src, &'src str, String, ParseExtra<'src>> + Clone {{
    text::ident::<_, ParseExtra<'src>>()
        .map(str::to_owned)
        .padded_by(ws())
}}

fn symbol_token<'src>() -> impl Parser<'src, &'src str, String, ParseExtra<'src>> + Clone {{
    just('@')
        .then(text::ident::<_, ParseExtra<'src>>().map(str::to_owned))
        .then(
            just('.')
                .ignore_then(text::ident::<_, ParseExtra<'src>>().map(str::to_owned))
                .repeated()
                .collect::<Vec<_>>()
        )
        .map(|((_, head), tail)| {{
            let mut out = format!("@{{head}}");
            for part in tail {{
                out.push('.');
                out.push_str(&part);
            }}
            out
        }})
        .padded_by(ws())
}}

fn int_token<'src>() -> impl Parser<'src, &'src str, String, ParseExtra<'src>> + Clone {{
    text::int::<_, ParseExtra<'src>>(10)
        .map(str::to_owned)
        .padded_by(ws())
}}

pub fn parse_module_text(source: &str) -> Result<Module, String> {{
    let param_parser = {param_parser};
    let expr_parser = recursive(|expr_parser| {expr_parser_body});
    let stmt_parser = {stmt_parser};
    let block_parser = {block_parser};
    let function_parser = {function_parser};
    let module_parser = {module_parser};

    module_parser
        .then_ignore(end())
        .parse(source)
        .into_result()
        .map_err(|errs| crate::format_rich_errors(source, errs))
}}
"#,
        param_parser = parser_defs[0],
        expr_parser_body = parser_defs[1],
        stmt_parser = parser_defs[2],
        block_parser = parser_defs[3],
        function_parser = parser_defs[4],
        module_parser = parser_defs[5],
    ))
}

fn render_common_placeholder(tag: &str, common_names: &[String], repr: &ReprBody) -> String {
    let common_name = common_names.iter().find(|name| {
        repr.common
            .as_ref()
            .and_then(|common| common.get(*name))
            .and_then(type_use_tag)
            == Some(tag)
    });

    match tag {
        "Prov" => {
            let alias = common_name
                .filter(|name| name.as_str() != "provenance")
                .map(|name| format!("pub type {} = Prov;\n", pascal_case(name)))
                .unwrap_or_default();
            format!(
                "#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]\npub struct Span {{\n    pub start: u32,\n    pub end: u32,\n}}\n\n#[derive(Debug, Clone, PartialEq, Eq, Default)]\npub struct Prov {{\n    pub file_id: Option<u32>,\n    pub span: Option<Span>,\n}}\n{alias}"
            )
        }
        "Symbol" => {
            let alias = common_name
                .filter(|name| name.as_str() != "symbol")
                .map(|name| format!("pub type {} = Symbol;\n", pascal_case(name)))
                .unwrap_or_default();
            format!(
                "#[derive(Debug, Clone, PartialEq, Eq, Default)]\npub struct Symbol(pub String);\n{alias}"
            )
        }
        "DocBlock" => {
            let alias = common_name
                .filter(|name| name.as_str() != "docs")
                .map(|name| format!("pub type {} = DocBlock;\n", pascal_case(name)))
                .unwrap_or_default();
            format!(
                "#[derive(Debug, Clone, PartialEq, Eq, Default)]\npub struct DocBlock(pub Vec<String>);\n{alias}"
            )
        }
        "Type" | "Literal" => {
            format!("#[derive(Debug, Clone, PartialEq, Eq, Default)]\npub struct {tag}(pub String);")
        }
        _ => format!("#[derive(Debug, Clone, PartialEq, Eq, Default)]\npub struct {tag};"),
    }
}

fn pascal_case(name: &str) -> String {
    let mut out = String::new();
    for part in name.split(['-', '_']) {
        if part.is_empty() {
            continue;
        }
        let mut chars = part.chars();
        if let Some(first) = chars.next() {
            out.extend(first.to_uppercase());
            out.push_str(chars.as_str());
        }
    }
    if out.is_empty() { "Unnamed".to_owned() } else { out }
}

fn snake_case(name: &str) -> String {
    let mut out = String::new();
    for (i, ch) in name.chars().enumerate() {
        if ch.is_ascii_uppercase() {
            if i != 0 {
                out.push('_');
            }
            out.push(ch.to_ascii_lowercase());
        } else {
            out.push(ch);
        }
    }
    out
}

fn render_visit_calls(
    ty: &TypeUse,
    expr: &str,
    node_names: &[String],
    mutable: bool,
    borrowed: bool,
) -> Vec<String> {
    match ty {
        TypeUse::Optional(items) if items.len() == 1 => {
            let binding = if mutable {
                if borrowed {
                    format!("if let Some(value) = {expr} {{")
                } else {
                    format!("if let Some(value) = &mut {expr} {{")
                }
            } else if borrowed {
                format!("if let Some(value) = {expr} {{")
            } else {
                format!("if let Some(value) = &{expr} {{")
            };
            let inner = render_visit_calls(&items[0], "value", node_names, mutable, true);
            if inner.is_empty() {
                Vec::new()
            } else {
                vec![format!(
                    "{binding}\n{}\n}}",
                    inner.into_iter().map(|line| format!("    {line}")).collect::<Vec<_>>().join("\n")
                )]
            }
        }
        TypeUse::Seq(items) if items.len() == 1 => {
            let inner = render_visit_calls(&items[0], "value", node_names, mutable, true);
            if inner.is_empty() {
                Vec::new()
            } else {
                let iter = if mutable { "iter_mut()" } else { "iter()" };
                vec![format!(
                    "for value in {expr}.{iter} {{\n{}\n}}",
                    inner.into_iter().map(|line| format!("    {line}")).collect::<Vec<_>>().join("\n")
                )]
            }
        }
        TypeUse::Ref { name: Some(tag) } if node_names.iter().any(|name| name == tag) => {
            let method = snake_case(tag);
            if mutable && borrowed {
                vec![format!("v.visit_{method}_mut({expr});")]
            } else if mutable {
                vec![format!("v.visit_{method}_mut(&mut {expr});")]
            } else if borrowed {
                vec![format!("v.visit_{method}({expr});")]
            } else {
                vec![format!("v.visit_{method}(&{expr});")]
            }
        }
        _ => Vec::new(),
    }
}

fn render_walk_fn(
    name: &str,
    decl: &NodeDecl,
    node_names: &[String],
    mutable: bool,
) -> Option<String> {
    let walk_name = if mutable {
        format!("walk_{}_mut", snake_case(name))
    } else {
        format!("walk_{}", snake_case(name))
    };
    let trait_name = if mutable { "VisitMut" } else { "Visit" };
    let node_ty = if mutable {
        format!("&mut {name}")
    } else {
        format!("&{name}")
    };
    match decl {
        NodeDecl::Node(fields) | NodeDecl::Struct(fields) => {
            let mut field_names = fields.fields.keys().cloned().collect::<Vec<_>>();
            field_names.sort();
            let body_lines = field_names
                .iter()
                .flat_map(|field_name| {
                    let field_expr = format!("node.{}", rust_ident(field_name));
                    render_visit_calls(
                        fields.fields.get(field_name).unwrap(),
                        &field_expr,
                        node_names,
                        mutable,
                        false,
                    )
                })
                .collect::<Vec<_>>();
            let (v_name, node_name) = if body_lines.is_empty() {
                ("_v", "_node")
            } else {
                ("v", "node")
            };
            let body = body_lines.join("\n").replace("node.", &format!("{node_name}."));
            Some(format!(
                "pub fn {walk_name}<V: ?Sized + {trait_name}>({v_name}: &mut V, {node_name}: {node_ty}) {{\n{body}\n}}"
            ))
        }
        NodeDecl::Enum(variants) => {
            let mut variant_names = variants.variants.keys().cloned().collect::<Vec<_>>();
            variant_names.sort();
            let arms = variant_names
                .iter()
                .filter_map(|variant_name| match variants.variants.get(variant_name).unwrap() {
                    NodeDecl::Node(fields) | NodeDecl::Struct(fields) => {
                        let mut field_names = fields.fields.keys().cloned().collect::<Vec<_>>();
                        field_names.sort();
                        let traversed = field_names
                            .iter()
                            .filter_map(|field_name| {
                                let expr = rust_ident(field_name);
                                let calls = render_visit_calls(
                                    fields.fields.get(field_name).unwrap(),
                                    &expr,
                                    node_names,
                                    mutable,
                                    true,
                                );
                                if calls.is_empty() {
                                    None
                                } else {
                                    Some((field_name.clone(), calls))
                                }
                            })
                            .collect::<Vec<_>>();
                        let pattern_fields = if traversed.len() == field_names.len() {
                            traversed
                                .iter()
                                .map(|(field_name, _)| rust_ident(field_name))
                                .collect::<Vec<_>>()
                                .join(", ")
                        } else if traversed.is_empty() {
                            "..".to_owned()
                        } else {
                            let mut parts = traversed
                                .iter()
                                .map(|(field_name, _)| rust_ident(field_name))
                                .collect::<Vec<_>>();
                            parts.push("..".to_owned());
                            parts.join(", ")
                        };
                        let body = traversed
                            .into_iter()
                            .flat_map(|(_, calls)| calls)
                            .map(|line| format!("            {line}"))
                            .collect::<Vec<_>>()
                            .join("\n");
                        Some(format!(
                            "        {name}::{variant_name} {{ {pattern_fields} }} => {{\n{body}\n        }}"
                        ))
                    }
                    _ => None,
                })
                .collect::<Vec<_>>()
                .join(",\n");
            Some(format!(
                "pub fn {walk_name}<V: ?Sized + {trait_name}>(v: &mut V, node: {node_ty}) {{\n    match node {{\n{arms}\n    }}\n}}"
            ))
        }
        NodeDecl::Other { .. } => None,
    }
}

fn render_node_decl(
    name: &str,
    decl: &NodeDecl,
    node_names: &[String],
    provenance_tag: &str,
) -> Option<String> {
    match decl {
        NodeDecl::Node(fields) | NodeDecl::Struct(fields) => {
            let mut field_names = fields.fields.keys().cloned().collect::<Vec<_>>();
            field_names.sort();
            let field_rows = field_names
                .iter()
                .map(|field_name| {
                    let ty = render_type_use(fields.fields.get(field_name).unwrap(), node_names, true);
                    format!("    pub {}: {},", rust_ident(field_name), ty)
                })
                .collect::<Vec<_>>()
                .join("\n");

            Some(format!(
                "#[derive(Debug, Clone, PartialEq, Eq)]\npub struct {name} {{\n{field_rows}\n}}"
            ))
        }
        NodeDecl::Enum(variants) => {
            let mut variant_names = variants.variants.keys().cloned().collect::<Vec<_>>();
            variant_names.sort();
            let variant_rows = variant_names
                .iter()
                .map(|variant_name| match variants.variants.get(variant_name).unwrap() {
                    NodeDecl::Node(fields) | NodeDecl::Struct(fields) => {
                        let mut field_names = fields.fields.keys().cloned().collect::<Vec<_>>();
                        field_names.sort();
                        let rows = field_names
                            .iter()
                            .map(|field_name| {
                                let ty =
                                    render_type_use(fields.fields.get(field_name).unwrap(), node_names, true);
                                format!("        {}: {},", rust_ident(field_name), ty)
                            })
                            .collect::<Vec<_>>()
                            .join("\n");
                        format!("    {variant_name} {{\n{rows}\n    }},")
                    }
                    other => format!("    {variant_name}, // unsupported variant shape: {other:?}"),
                })
                .collect::<Vec<_>>()
                .join("\n");
            let prov_impl = if variants
                .variants
                .values()
                .all(|variant| matches!(variant, NodeDecl::Node(fields) | NodeDecl::Struct(fields) if node_fields_have_prov(fields, provenance_tag)))
            {
                let match_rows = variant_names
                    .iter()
                    .map(|variant_name| format!("            Self::{variant_name} {{ prov, .. }} => Some(prov),"))
                    .collect::<Vec<_>>()
                    .join("\n");
                format!(
                    "\nimpl HasProvenance for {name} {{\n    fn provenance(&self) -> Option<&{provenance_tag}> {{\n        match self {{\n{match_rows}\n        }}\n    }}\n}}"
                )
            } else {
                String::new()
            };
            Some(format!(
                "#[derive(Debug, Clone, PartialEq, Eq)]\npub enum {name} {{\n{variant_rows}\n}}{prov_impl}"
            ))
        }
        NodeDecl::Other { .. } => None,
    }
}

fn render_hir_poc_module(repr: &ReprBody) -> String {
    let mut token_names = repr.syntax.tokens.keys().cloned().collect::<Vec<_>>();
    token_names.sort();

    let mut rule_names = repr.syntax.rules.keys().cloned().collect::<Vec<_>>();
    rule_names.sort();

    let mut print_keys = repr.syntax.canonical_print.keys().cloned().collect::<Vec<_>>();
    print_keys.sort();

    let mut common_names = repr
        .common
        .as_ref()
        .map(|common| common.keys().cloned().collect::<Vec<_>>())
        .unwrap_or_default();
    common_names.sort();

    let mut node_names = repr
        .nodes
        .as_ref()
        .map(|nodes| nodes.keys().cloned().collect::<Vec<_>>())
        .unwrap_or_default();
    node_names.sort();

    let provenance_tag = repr
        .common
        .as_ref()
        .and_then(|common| common.get("provenance"))
        .and_then(type_use_tag)
        .unwrap_or("Prov")
        .to_owned();

    let token_rows = token_names
        .iter()
        .map(|name| {
            let kind = match repr.syntax.tokens.get(name).unwrap() {
                TokenExpr::Regex(_) => "regex",
                TokenExpr::Other { name, .. } => name.as_deref().unwrap_or("<unknown>"),
            };
            format!("    TokenSpec {{ name: {name:?}, kind: {kind:?} }},")
        })
        .collect::<Vec<_>>()
        .join("\n");

    let rule_rows = rule_names
        .iter()
        .map(|name| {
            let kind = rule_expr_kind(repr.syntax.rules.get(name).unwrap());
            format!("    RuleSpec {{ name: {name:?}, kind: {kind:?} }},")
        })
        .collect::<Vec<_>>()
        .join("\n");

    let common_rows = common_names
        .iter()
        .map(|name| {
            let kind = match repr.common.as_ref().and_then(|common| common.get(name)) {
                Some(ty) => type_use_tag(ty).unwrap_or("scalar"),
                None => "<missing>",
            };
            format!("    TypeUseSpec {{ name: {name:?}, kind: {kind:?} }},")
        })
        .collect::<Vec<_>>()
        .join("\n");

    let node_rows = node_names
        .iter()
        .map(|name| {
            let kind = match repr.nodes.as_ref().and_then(|nodes| nodes.get(name)) {
                Some(NodeDecl::Node(_)) => "node",
                Some(NodeDecl::Enum(_)) => "enum",
                Some(NodeDecl::Struct(_)) => "struct",
                Some(NodeDecl::Other { tag, .. }) => tag.as_deref().unwrap_or("<unknown>"),
                None => "<missing>",
            };
            format!("    NodeSpec {{ name: {name:?}, kind: {kind:?} }},")
        })
        .collect::<Vec<_>>()
        .join("\n");

    let mut placeholder_names = Vec::new();
    if let Some(common) = &repr.common {
        for ty in common.values() {
            collect_type_tags(ty, &mut placeholder_names);
        }
    }
    if let Some(nodes) = &repr.nodes {
        for decl in nodes.values() {
            match decl {
                NodeDecl::Node(fields) | NodeDecl::Struct(fields) => {
                    for ty in fields.fields.values() {
                        collect_type_tags(ty, &mut placeholder_names);
                    }
                }
                NodeDecl::Enum(variants) => {
                    for variant in variants.variants.values() {
                        if let NodeDecl::Node(fields) | NodeDecl::Struct(fields) = variant {
                            for ty in fields.fields.values() {
                                collect_type_tags(ty, &mut placeholder_names);
                            }
                        }
                    }
                }
                NodeDecl::Other { .. } => {}
            }
        }
    }
    placeholder_names.sort();
    placeholder_names.dedup();
    placeholder_names.retain(|name| {
        !matches!(name.as_str(), "optional" | "seq") && !node_names.iter().any(|node| node == name)
    });

    let placeholder_rows = placeholder_names
        .iter()
        .map(|name| render_common_placeholder(name, &common_names, repr))
        .collect::<Vec<_>>()
        .join("\n\n");

    let ast_rows = node_names
        .iter()
        .filter_map(|name| {
            repr.nodes
                .as_ref()
                .and_then(|nodes| nodes.get(name))
                .and_then(|decl| render_node_decl(name, decl, &node_names, &provenance_tag))
        })
        .collect::<Vec<_>>()
        .join("\n\n");

    let prov_impl_rows = node_names
        .iter()
        .filter_map(|name| {
            let decl = repr.nodes.as_ref().and_then(|nodes| nodes.get(name))?;
            match decl {
                NodeDecl::Node(fields) | NodeDecl::Struct(fields)
                    if node_fields_have_prov(fields, &provenance_tag) =>
                {
                    Some(format!(
                        "impl HasProvenance for {name} {{\n    fn provenance(&self) -> Option<&{provenance_tag}> {{\n        Some(&self.prov)\n    }}\n}}"
                    ))
                }
                _ => None,
            }
        })
        .collect::<Vec<_>>()
        .join("\n\n");

    let visit_trait_rows = node_names
        .iter()
        .map(|name| {
            let method = snake_case(name);
            format!(
                "    fn visit_{method}(&mut self, node: &{name}) {{\n        walk_{method}(self, node);\n    }}"
            )
        })
        .collect::<Vec<_>>()
        .join("\n");

    let visit_mut_trait_rows = node_names
        .iter()
        .map(|name| {
            let method = snake_case(name);
            format!(
                "    fn visit_{method}_mut(&mut self, node: &mut {name}) {{\n        walk_{method}_mut(self, node);\n    }}"
            )
        })
        .collect::<Vec<_>>()
        .join("\n");

    let walk_rows = node_names
        .iter()
        .filter_map(|name| {
            repr.nodes
                .as_ref()
                .and_then(|nodes| nodes.get(name))
                .and_then(|decl| render_walk_fn(name, decl, &node_names, false))
        })
        .collect::<Vec<_>>()
        .join("\n\n");

    let walk_mut_rows = node_names
        .iter()
        .filter_map(|name| {
            repr.nodes
                .as_ref()
                .and_then(|nodes| nodes.get(name))
                .and_then(|decl| render_walk_fn(name, decl, &node_names, true))
        })
        .collect::<Vec<_>>()
        .join("\n\n");

    let print_rows = print_keys
        .iter()
        .map(|name| {
            let template = repr.syntax.canonical_print.get(name).unwrap();
            format!("    PrintSpec {{ name: {name:?}, template: {template:?} }},")
        })
        .collect::<Vec<_>>()
        .join("\n");

    let parser_rows =
        render_parser_block(repr, &node_names, &provenance_tag).expect("parser block should render");

    let raw = format!(
        r###"
// @generated by kajit-foundation::generate_repr_poc from {file_ext} schema {name}.
// Do not edit manually.
//
// This module is intentionally narrow: it exposes only data that is actually
// derived from the pilot schema.

use chumsky::prelude::*;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TokenSpec {{
    pub name: &'static str,
    pub kind: &'static str,
}}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RuleSpec {{
    pub name: &'static str,
    pub kind: &'static str,
}}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TypeUseSpec {{
    pub name: &'static str,
    pub kind: &'static str,
}}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct NodeSpec {{
    pub name: &'static str,
    pub kind: &'static str,
}}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PrintSpec {{
    pub name: &'static str,
    pub template: &'static str,
}}

pub trait HasProvenance {{
    fn provenance(&self) -> Option<&{provenance_tag}>;
}}

pub trait Visit {{
{visit_trait_rows}
}}

pub trait VisitMut {{
{visit_mut_trait_rows}
}}

pub const REPR_NAME: &str = {name:?};
pub const REPR_FILE_EXT: &str = {file_ext:?};
pub const REPR_PURPOSE: &str = {purpose:?};
pub const REPR_ROUND_TRIP: &str = {round_trip:?};
pub const REPR_PROVENANCE: &str = {provenance:?};

{placeholder_rows}

{ast_rows}

{prov_impl_rows}

{walk_rows}

{walk_mut_rows}

{parser_rows}

#[cfg(test)]
mod tests {{
    use super::*;

    #[test]
    fn parse_module_smoke() {{
        let module = parse_module_text("module {{ fn main() -> Value {{ return }} }}").unwrap();
        assert_eq!(module.functions.len(), 1);
        assert_eq!(module.functions[0].name, Symbol("main".to_owned()));
        assert_eq!(module.functions[0].return_type, Type("Value".to_owned()));
        assert!(matches!(module.functions[0].body.statements.as_slice(), [Stmt::Return {{ value: None, .. }}]));
    }}
}}

pub static TOKENS: &[TokenSpec] = &[
{token_rows}
];

pub static RULES: &[RuleSpec] = &[
{rule_rows}
];

pub static COMMON_TYPES: &[TypeUseSpec] = &[
{common_rows}
];

pub static NODES: &[NodeSpec] = &[
{node_rows}
];

pub static CANONICAL_PRINT: &[PrintSpec] = &[
{print_rows}
];
"###,
        file_ext = repr.file_ext,
        name = repr.name,
        purpose = repr.contract.purpose,
        round_trip = repr.contract.round_trip,
        provenance = repr.contract.provenance,
        placeholder_rows = placeholder_rows,
        ast_rows = ast_rows,
        prov_impl_rows = prov_impl_rows,
        visit_trait_rows = visit_trait_rows,
        visit_mut_trait_rows = visit_mut_trait_rows,
        walk_rows = walk_rows,
        walk_mut_rows = walk_mut_rows,
        parser_rows = parser_rows,
        provenance_tag = provenance_tag,
        token_rows = token_rows,
        rule_rows = rule_rows,
        common_rows = common_rows,
        node_rows = node_rows,
        print_rows = print_rows,
    );

    prettyplease::unparse(&syn::parse_file(&raw).expect("generated HIR POC should parse"))
}
