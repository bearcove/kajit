use std::collections::{BTreeSet, HashMap, HashSet};

use crate::normalize::{
    DocumentedValue, NormalizedNodeDecl, NormalizedRepr, NormalizedTokenKind, SyntaxRule,
    SyntaxTypeUse, is_string_scalar_type, render_default_value,
};
use crate::render_helpers::{rust_ident, snake_case};

enum SeqItem {
    Ignore(String),
    Bind { name: String, parser: String },
}

fn token_parser_fn_name(token_name: &str) -> String {
    format!("token_{}", snake_case(token_name))
}

fn render_wrapped_string_parser(
    repr: &NormalizedRepr,
    ty: &SyntaxTypeUse,
    parser_fn_name: &str,
) -> Result<String, String> {
    match ty {
        SyntaxTypeUse::Ref { name } if is_string_scalar_type(repr, name) => {
            Ok(format!("{parser_fn_name}().map({name})"))
        }
        SyntaxTypeUse::Ref { name } => Err(format!(
            "token parser cannot construct non-string scalar type {name:?}"
        )),
        _ => Err(format!(
            "token parser requires a scalar reference type, got {:?}",
            ty
        )),
    }
}

fn render_token_value_parser(
    repr: &NormalizedRepr,
    token_name: &str,
    ty: &SyntaxTypeUse,
) -> Result<String, String> {
    let parser_fn_name = token_parser_fn_name(token_name);
    render_wrapped_string_parser(repr, ty, &parser_fn_name)
}

fn render_ref_value_parser(
    ref_name: &str,
    ty: &SyntaxTypeUse,
    rule_names: &HashSet<String>,
    node_names: &[String],
    box_node_refs: bool,
) -> Result<String, String> {
    if !rule_names.contains(ref_name) {
        return Err(format!(
            "unsupported reference parser target {ref_name:?}; scalar values must come from @token(...)"
        ));
    }

    let base = format!("{}_parser.clone()", snake_case(ref_name));

    Ok(match ty {
        SyntaxTypeUse::Ref { name }
            if box_node_refs && node_names.iter().any(|node| node == name) =>
        {
            format!("({base}).map(Box::new)")
        }
        _ => base,
    })
}

fn render_value_parser(
    repr: &NormalizedRepr,
    rule: &SyntaxRule,
    ty: &SyntaxTypeUse,
    rule_names: &HashSet<String>,
    node_names: &[String],
    box_node_refs: bool,
) -> Result<String, String> {
    match rule {
        SyntaxRule::Token { name } => render_token_value_parser(repr, name, ty),
        SyntaxRule::Ref { name } => {
            render_ref_value_parser(name, ty, rule_names, node_names, box_node_refs)
        }
        SyntaxRule::Optional { inner } => {
            let SyntaxTypeUse::Optional(inner_ty) = ty else {
                return Err("optional rule without optional type".to_owned());
            };
            let inner = render_value_parser(repr, inner, inner_ty, rule_names, node_names, true)?;
            Ok(format!("({inner}).or_not()"))
        }
        SyntaxRule::Repeat { item, sep } => {
            let SyntaxTypeUse::Seq(inner_ty) = ty else {
                return Err("repeat rule without seq type".to_owned());
            };
            let inner = render_value_parser(repr, item, inner_ty, rule_names, node_names, false)?;
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
            ty, rule
        )),
    }
}

fn flatten_struct_rule_items(
    repr: &NormalizedRepr,
    rule: &SyntaxRule,
    fields: &HashMap<String, DocumentedValue<SyntaxTypeUse>>,
    rule_names: &HashSet<String>,
    node_names: &[String],
) -> Result<Vec<SeqItem>, String> {
    match rule {
        SyntaxRule::Seq(items) => {
            let mut out = Vec::new();
            for item in items {
                out.extend(flatten_struct_rule_items(
                    repr, item, fields, rule_names, node_names,
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
                render_value_parser(repr, named.inner.as_ref(), ty, rule_names, node_names, true)?;
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

fn render_unbound_field_value(
    field_name: &str,
    ty: &SyntaxTypeUse,
    provenance_tag: &str,
) -> Result<String, String> {
    if field_name == "prov" && matches!(ty, SyntaxTypeUse::Ref { name } if name == provenance_tag) {
        return Ok("prov_from_span(e.span(), file_id)".to_owned());
    }

    render_default_value(ty, provenance_tag).ok_or_else(|| {
        format!("field {field_name:?} is not bound by syntax and has no implicit synthesis rule")
    })
}

fn render_struct_parser_expr(
    repr: &NormalizedRepr,
    type_name: &str,
    fields: &HashMap<String, DocumentedValue<SyntaxTypeUse>>,
    rule: &SyntaxRule,
    rule_names: &HashSet<String>,
    node_names: &[String],
    provenance_tag: &str,
) -> Result<String, String> {
    let items = flatten_struct_rule_items(repr, rule, fields, rule_names, node_names)?;
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
            } else {
                render_unbound_field_value(
                    field_name,
                    &fields.get(field_name).unwrap().value,
                    provenance_tag,
                )?
            };
            Ok(format!("{ident}: {value}"))
        })
        .collect::<Result<Vec<_>, String>>()?
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
    repr: &NormalizedRepr,
    enum_name: &str,
    variants: &HashMap<String, DocumentedValue<NormalizedNodeDecl>>,
    rule: &SyntaxRule,
    rule_names: &HashSet<String>,
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
            flatten_struct_rule_items(repr, named.inner.as_ref(), fields, rule_names, node_names)?;
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
                } else {
                    render_unbound_field_value(
                        field_name,
                        &fields.get(field_name).unwrap().value,
                        provenance_tag,
                    )?
                };
                Ok(format!("{ident}: {value}"))
            })
            .collect::<Result<Vec<_>, String>>()?
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
    repr: &NormalizedRepr,
    rule_name: &str,
    rule: &SyntaxRule,
    decl: &NormalizedNodeDecl,
    rule_names: &HashSet<String>,
    node_names: &[String],
    provenance_tag: &str,
) -> Result<String, String> {
    match decl {
        NormalizedNodeDecl::Node(fields) | NormalizedNodeDecl::Struct(fields) => {
            render_struct_parser_expr(
                repr,
                rule_name,
                fields,
                rule,
                rule_names,
                node_names,
                provenance_tag,
            )
        }
        NormalizedNodeDecl::Enum(variants) => render_enum_parser_expr(
            repr,
            rule_name,
            variants,
            rule,
            rule_names,
            node_names,
            provenance_tag,
        ),
    }
}

fn collect_rule_dependencies(
    rule: &SyntaxRule,
    rule_names: &HashSet<String>,
    out: &mut BTreeSet<String>,
) {
    match rule {
        SyntaxRule::Seq(items) | SyntaxRule::Choice(items) => {
            for item in items {
                collect_rule_dependencies(item, rule_names, out);
            }
        }
        SyntaxRule::Field(named) | SyntaxRule::Variant(named) => {
            collect_rule_dependencies(named.inner.as_ref(), rule_names, out);
        }
        SyntaxRule::Ref { name } => {
            if rule_names.contains(name) {
                out.insert(name.clone());
            }
        }
        SyntaxRule::Optional { inner } => {
            collect_rule_dependencies(inner.as_ref(), rule_names, out);
        }
        SyntaxRule::Repeat { item, .. } => {
            collect_rule_dependencies(item.as_ref(), rule_names, out);
        }
        SyntaxRule::Token { .. } | SyntaxRule::Literal(_) => {}
    }
}

fn topo_visit(
    name: &str,
    deps: &HashMap<String, BTreeSet<String>>,
    temporary: &mut HashSet<String>,
    permanent: &mut HashSet<String>,
    ordered: &mut Vec<String>,
) -> Result<(), String> {
    if permanent.contains(name) {
        return Ok(());
    }
    if !temporary.insert(name.to_owned()) {
        return Err(format!(
            "mutually recursive syntax rules are not supported yet; cycle includes {name:?}"
        ));
    }

    if let Some(children) = deps.get(name) {
        for dep in children {
            topo_visit(dep, deps, temporary, permanent, ordered)?;
        }
    }

    temporary.remove(name);
    permanent.insert(name.to_owned());
    ordered.push(name.to_owned());
    Ok(())
}

fn derive_parser_order(repr: &NormalizedRepr) -> Result<Vec<String>, String> {
    let rule_names = repr.syntax.rules.keys().cloned().collect::<HashSet<_>>();
    let mut deps = HashMap::new();
    for (name, rule) in &repr.syntax.rules {
        let mut refs = BTreeSet::new();
        collect_rule_dependencies(rule, &rule_names, &mut refs);
        refs.remove(name);
        deps.insert(name.clone(), refs);
    }

    let mut ordered = Vec::new();
    let mut temporary = HashSet::new();
    let mut permanent = HashSet::new();
    let mut names = repr.syntax.rules.keys().cloned().collect::<Vec<_>>();
    names.sort();
    for name in names {
        topo_visit(&name, &deps, &mut temporary, &mut permanent, &mut ordered)?;
    }
    Ok(ordered)
}

fn rule_is_self_recursive(name: &str, rule: &SyntaxRule) -> bool {
    let mut refs = BTreeSet::new();
    collect_rule_dependencies(rule, &HashSet::from([name.to_owned()]), &mut refs);
    refs.contains(name)
}

fn render_token_parser_fn(token_name: &str, kind: NormalizedTokenKind) -> String {
    let fn_name = token_parser_fn_name(token_name);
    let body = match kind {
        NormalizedTokenKind::Ident => {
            "text::ident::<_, ParseExtra<'src>>().map(str::to_owned).padded_by(ws())"
        }
        NormalizedTokenKind::Symbol => {
            "just('@')\n        .then(text::ident::<_, ParseExtra<'src>>().map(str::to_owned))\n        .then(\n            just('.')\n                .ignore_then(text::ident::<_, ParseExtra<'src>>().map(str::to_owned))\n                .repeated()\n                .collect::<Vec<_>>()\n        )\n        .map(|((_, head), tail)| {\n            let mut out = format!(\"@{head}\");\n            for part in tail {\n                out.push('.');\n                out.push_str(&part);\n            }\n            out\n        })\n        .padded_by(ws())"
        }
        NormalizedTokenKind::Int => {
            "text::int::<_, ParseExtra<'src>>(10).map(str::to_owned).padded_by(ws())"
        }
    };

    format!(
        "fn {fn_name}<'src>() -> impl Parser<'src, &'src str, String, ParseExtra<'src>> + Clone {{\n    {body}\n}}"
    )
}

pub(crate) fn render_parser_block(
    repr: &NormalizedRepr,
    node_names: &[String],
    provenance_tag: &str,
) -> Result<String, String> {
    let rule_names = repr.syntax.rules.keys().cloned().collect::<HashSet<_>>();
    let parser_order = derive_parser_order(repr)?;
    let root_name = repr.syntax.root.as_str();
    let root_parser_name = format!("{}_parser", snake_case(root_name));
    let root_fn_suffix = snake_case(root_name);

    let parser_defs = parser_order
        .iter()
        .map(|name| {
            let rule = repr
                .syntax
                .rules
                .get(name)
                .ok_or_else(|| format!("missing syntax rule for {name}"))?;
            let decl = &repr
                .nodes
                .get(name)
                .ok_or_else(|| format!("missing node declaration for {name}"))?
                .value;
            let parser_expr = render_rule_parser_expr(
                repr,
                name,
                rule,
                decl,
                &rule_names,
                node_names,
                provenance_tag,
            )?;
            let parser_name = format!("{}_parser", snake_case(name));
            Ok(if rule_is_self_recursive(name, rule) {
                format!("    let {parser_name} = recursive(move |{parser_name}| {parser_expr});")
            } else {
                format!("    let {parser_name} = {parser_expr};")
            })
        })
        .collect::<Result<Vec<_>, String>>()?
        .join("\n");

    let mut token_names = repr.syntax.token_kinds.keys().cloned().collect::<Vec<_>>();
    token_names.sort();
    let token_rows = token_names
        .iter()
        .map(|name| render_token_parser_fn(name, *repr.syntax.token_kinds.get(name).unwrap()))
        .collect::<Vec<_>>()
        .join("\n\n");

    Ok(format!(
        r#"
type ParseExtra<'src> = extra::Err<Rich<'src, char>>;

fn ws<'src>() -> impl Parser<'src, &'src str, (), ParseExtra<'src>> + Clone {{
    any()
        .filter(|c: &char| c.is_whitespace())
        .repeated()
        .ignored()
}}

{token_rows}

fn prov_from_span(span: chumsky::span::SimpleSpan<usize>, file_id: Option<u32>) -> Prov {{
    Prov {{
        file_id,
        span: Some(Span {{
            start: span.start as u32,
            end: span.end as u32,
        }}),
    }}
}}

pub fn parse_root_text_rich(
    source: &str,
    file_id: Option<u32>,
) -> Result<{root_name}, Vec<Rich<'_, char>>> {{
{parser_defs}

    {root_parser_name}
        .then_ignore(end())
        .parse(source)
        .into_result()
}}

pub fn parse_root_text_with_file_id(source: &str, file_id: Option<u32>) -> Result<{root_name}, String> {{
    parse_root_text_rich(source, file_id).map_err(|errs| crate::format_rich_errors(source, errs))
}}

pub fn parse_root_text(source: &str) -> Result<{root_name}, String> {{
    parse_root_text_with_file_id(source, None)
}}

pub fn parse_{root_fn_suffix}_text_rich(
    source: &str,
    file_id: Option<u32>,
) -> Result<{root_name}, Vec<Rich<'_, char>>> {{
    parse_root_text_rich(source, file_id)
}}

pub fn parse_{root_fn_suffix}_text_with_file_id(
    source: &str,
    file_id: Option<u32>,
) -> Result<{root_name}, String> {{
    parse_root_text_with_file_id(source, file_id)
}}

pub fn parse_{root_fn_suffix}_text(source: &str) -> Result<{root_name}, String> {{
    parse_root_text(source)
}}
"#,
        token_rows = token_rows,
        parser_defs = parser_defs,
        root_name = root_name,
        root_parser_name = root_parser_name,
        root_fn_suffix = root_fn_suffix,
    ))
}
