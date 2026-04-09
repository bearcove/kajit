use std::collections::{BTreeSet, HashMap};

use crate::normalize::{
    DocumentedValue, NormalizedNodeDecl, NormalizedRepr, SyntaxRule, SyntaxTypeUse,
    render_default_value, syntax_type_name,
};
use crate::render_helpers::rust_ident;

enum SeqItem {
    Ignore(String),
    Bind { name: String, parser: String },
}

fn render_token_value_parser(token_name: &str, ty: &SyntaxTypeUse) -> Result<String, String> {
    let parser = match (token_name, syntax_type_name(ty)) {
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
                syntax_type_name(ty)
            ));
        }
    };
    Ok(parser)
}

fn render_ref_value_parser(
    ref_name: &str,
    ty: &SyntaxTypeUse,
    rule_names: &[String],
    node_names: &[String],
    box_node_refs: bool,
) -> Result<String, String> {
    let base = if rule_names.iter().any(|name| name == ref_name) {
        format!(
            "{}_parser.clone()",
            crate::render_helpers::snake_case(ref_name)
        )
    } else {
        match ref_name {
            "Type" => "ident_token().map(Type)".to_owned(),
            _ => return Err(format!("unsupported reference parser for {ref_name:?}")),
        }
    };

    let parser = match ty {
        SyntaxTypeUse::Ref { name }
            if box_node_refs && node_names.iter().any(|node| node == name) =>
        {
            format!("({base}).map(Box::new)")
        }
        SyntaxTypeUse::Ref { .. } => base,
        _ => base,
    };
    Ok(parser)
}

fn render_value_parser(
    rule: &SyntaxRule,
    ty: &SyntaxTypeUse,
    rule_names: &[String],
    node_names: &[String],
    box_node_refs: bool,
) -> Result<String, String> {
    match rule {
        SyntaxRule::Token { name } => render_token_value_parser(name, ty),
        SyntaxRule::Ref { name } => {
            render_ref_value_parser(name, ty, rule_names, node_names, box_node_refs)
        }
        SyntaxRule::Optional { inner } => {
            let SyntaxTypeUse::Optional(inner_ty) = ty else {
                return Err("optional rule without optional type".to_owned());
            };
            let inner = render_value_parser(inner, inner_ty, rule_names, node_names, true)?;
            Ok(format!("({inner}).or_not()"))
        }
        SyntaxRule::Repeat { item, sep } => {
            let SyntaxTypeUse::Seq(inner_ty) = ty else {
                return Err("repeat rule without seq type".to_owned());
            };
            let inner = render_value_parser(item, inner_ty, rule_names, node_names, false)?;
            Ok(if let Some(sep) = sep.as_deref() {
                format!(
                    "({inner}).separated_by(just({sep:?}).padded()).allow_trailing().collect::<Vec<_>>()"
                )
            } else {
                format!("({inner}).repeated().collect::<Vec<_>>()")
            })
        }
        _ => Err(format!(
            "unsupported value rule kind for field type {:?}: {:?}",
            syntax_type_name(ty),
            rule
        )),
    }
}

fn flatten_struct_rule_items(
    rule: &SyntaxRule,
    fields: &HashMap<String, DocumentedValue<SyntaxTypeUse>>,
    rule_names: &[String],
    node_names: &[String],
) -> Result<Vec<SeqItem>, String> {
    match rule {
        SyntaxRule::Seq(items) => {
            let mut out = Vec::new();
            for item in items {
                out.extend(flatten_struct_rule_items(
                    item, fields, rule_names, node_names,
                )?);
            }
            Ok(out)
        }
        SyntaxRule::Field(named) => {
            let field_name = named.name.as_str();
            let ty = &fields
                .get(field_name)
                .ok_or_else(|| format!("schema node field {field_name:?} not found"))?
                .value;
            let parser =
                render_value_parser(named.inner.as_ref(), ty, rule_names, node_names, true)?;
            Ok(vec![SeqItem::Bind {
                name: field_name.to_owned(),
                parser,
            }])
        }
        SyntaxRule::Literal(text) => Ok(vec![SeqItem::Ignore(format!("just({text:?}).padded()"))]),
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
    fields: &HashMap<String, DocumentedValue<SyntaxTypeUse>>,
    rule: &SyntaxRule,
    rule_names: &[String],
    node_names: &[String],
    provenance_tag: &str,
) -> Result<String, String> {
    let items = flatten_struct_rule_items(rule, fields, rule_names, node_names)?;
    let (chain, bound_names) = render_binding_chain(&items)?;
    let bound_set = bound_names.iter().cloned().collect::<BTreeSet<_>>();

    let mut field_names = fields.keys().cloned().collect::<Vec<_>>();
    field_names.sort();
    let field_rows = field_names
        .iter()
        .map(|field_name| {
            let ident = rust_ident(field_name);
            let value = if bound_set.contains(&ident) {
                ident.clone()
            } else if field_name == "prov" {
                "prov_from_span(e.span(), file_id)".to_owned()
            } else {
                render_default_value(&fields.get(field_name).unwrap().value, provenance_tag)
            };
            format!("{ident}: {value}")
        })
        .collect::<Vec<_>>()
        .join(", ");
    let span_ident = if fields.contains_key("prov") {
        "e"
    } else {
        "_e"
    };

    let map = if bound_names.is_empty() {
        format!(".map_with(move |(), {span_ident}| {type_name} {{ {field_rows} }})")
    } else if bound_names.len() == 1 {
        format!(
            ".map_with(move |{}, {span_ident}| {type_name} {{ {field_rows} }})",
            bound_names[0],
        )
    } else {
        format!(
            ".map_with(move |{}, {span_ident}| {type_name} {{ {field_rows} }})",
            nested_tuple_pattern(&bound_names),
        )
    };

    Ok(format!("({chain}){map}.boxed()"))
}

fn render_enum_parser_expr(
    enum_name: &str,
    variants: &HashMap<String, DocumentedValue<NormalizedNodeDecl>>,
    rule: &SyntaxRule,
    rule_names: &[String],
    node_names: &[String],
    provenance_tag: &str,
) -> Result<String, String> {
    let SyntaxRule::Choice(items) = rule else {
        return Err(format!("enum rule for {enum_name} must be choice"));
    };

    let mut variant_parsers = Vec::new();
    for item in items {
        let SyntaxRule::Variant(named) = item else {
            return Err(format!(
                "enum rule for {enum_name} contains non-variant item"
            ));
        };
        let variant_name = named.name.as_str();
        let Some(variant_decl) = variants.get(variant_name) else {
            return Err(format!(
                "schema enum {enum_name} is missing variant declaration {variant_name:?}"
            ));
        };
        let (NormalizedNodeDecl::Struct(fields) | NormalizedNodeDecl::Node(fields)) =
            &variant_decl.value
        else {
            return Err(format!(
                "schema enum {enum_name} variant {variant_name:?} has unsupported declaration"
            ));
        };
        let items =
            flatten_struct_rule_items(named.inner.as_ref(), fields, rule_names, node_names)?;
        let (chain, bound_names) = render_binding_chain(&items)?;
        let bound_set = bound_names.iter().cloned().collect::<BTreeSet<_>>();
        let mut field_names = fields.keys().cloned().collect::<Vec<_>>();
        field_names.sort();
        let field_rows = field_names
            .iter()
            .map(|field_name| {
                let ident = rust_ident(field_name);
                let value = if bound_set.contains(&ident) {
                    ident.clone()
                } else if field_name == "prov" {
                    "prov_from_span(e.span(), file_id)".to_owned()
                } else {
                    render_default_value(&fields.get(field_name).unwrap().value, provenance_tag)
                };
                format!("{ident}: {value}")
            })
            .collect::<Vec<_>>()
            .join(", ");
        let span_ident = if fields.contains_key("prov") {
            "e"
        } else {
            "_e"
        };
        let parser = if bound_names.is_empty() {
            format!(
                "({chain}).map_with(move |(), {span_ident}| {enum_name}::{variant_name} {{ {field_rows} }})"
            )
        } else if bound_names.len() == 1 {
            format!(
                "({chain}).map_with(move |{}, {span_ident}| {enum_name}::{variant_name} {{ {field_rows} }})",
                bound_names[0],
            )
        } else {
            format!(
                "({chain}).map_with(move |{}, {span_ident}| {enum_name}::{variant_name} {{ {field_rows} }})",
                nested_tuple_pattern(&bound_names),
            )
        };
        variant_parsers.push(parser);
    }

    Ok(format!("choice(({})).boxed()", variant_parsers.join(", ")))
}

fn render_rule_parser_expr(
    rule_name: &str,
    rule: &SyntaxRule,
    decl: &NormalizedNodeDecl,
    rule_names: &[String],
    node_names: &[String],
    provenance_tag: &str,
) -> Result<String, String> {
    match decl {
        NormalizedNodeDecl::Node(fields) | NormalizedNodeDecl::Struct(fields) => {
            render_struct_parser_expr(
                rule_name,
                fields,
                rule,
                rule_names,
                node_names,
                provenance_tag,
            )
        }
        NormalizedNodeDecl::Enum(variants) => render_enum_parser_expr(
            rule_name,
            variants,
            rule,
            rule_names,
            node_names,
            provenance_tag,
        ),
    }
}

pub(crate) fn render_parser_block(
    repr: &NormalizedRepr,
    node_names: &[String],
    provenance_tag: &str,
) -> Result<String, String> {
    let rule_names = repr.syntax.rules.keys().cloned().collect::<Vec<_>>();
    let parser_order = ["Param", "Expr", "Stmt", "Block", "Function", "Module"];

    let parser_defs = parser_order
        .iter()
        .filter_map(|name| {
            let rule = repr.syntax.rules.get(*name)?;
            let decl = &repr.nodes.get(*name)?.value;
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

fn prov_from_span(span: chumsky::span::SimpleSpan<usize>, file_id: Option<u32>) -> Prov {{
    Prov {{
        file_id,
        span: Some(Span {{
            start: span.start as u32,
            end: span.end as u32,
        }}),
    }}
}}

pub fn parse_module_text_rich(source: &str, file_id: Option<u32>) -> Result<Module, Vec<Rich<'_, char>>> {{
    let param_parser = {param_parser};
    let expr_parser = recursive(move |expr_parser| {expr_parser_body});
    let stmt_parser = {stmt_parser};
    let block_parser = {block_parser};
    let function_parser = {function_parser};
    let module_parser = {module_parser};

    module_parser
        .then_ignore(end())
        .parse(source)
        .into_result()
}}

pub fn parse_module_text_with_file_id(source: &str, file_id: Option<u32>) -> Result<Module, String> {{
    parse_module_text_rich(source, file_id).map_err(|errs| crate::format_rich_errors(source, errs))
}}

pub fn parse_module_text(source: &str) -> Result<Module, String> {{
    parse_module_text_with_file_id(source, None)
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
