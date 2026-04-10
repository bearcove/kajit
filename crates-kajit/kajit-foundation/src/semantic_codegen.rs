use std::collections::HashMap;

use crate::normalize::{
    DocumentedValue, NormalizedNodeDecl, NormalizedRepr, SyntaxRule, SyntaxTypeUse,
    classify_ref_type, is_id_type, is_int_scalar_type, is_string_scalar_type,
};
use crate::render_helpers::{is_prov_only_struct, rust_ident, snake_case};

fn parse_target(target: &str) -> Result<Option<(String, String)>, String> {
    if let Some(literal) = target.strip_prefix("literal.") {
        if literal.is_empty() {
            return Err(format!("unsupported semantic token target {target:?}"));
        }
        return Ok(None);
    }
    if let Some(path) = target.strip_prefix("field.") {
        let parts = path.split('.').collect::<Vec<_>>();
        return match parts.as_slice() {
            [_, _] | [_, _, _] => Ok(None),
            _ => Err(format!("unsupported semantic token target {target:?}")),
        };
    }
    if let Some(path) = target.strip_prefix("variant.") {
        let parts = path.split('.').collect::<Vec<_>>();
        return match parts.as_slice() {
            [type_name, variant_name] => {
                Ok(Some(((*type_name).to_owned(), (*variant_name).to_owned())))
            }
            _ => Err(format!("unsupported semantic token target {target:?}")),
        };
    }
    Err(format!("unsupported semantic token target {target:?}"))
}

fn collect_annotated_literals(
    rule: &SyntaxRule,
    semantic_tokens: &HashMap<String, String>,
    out: &mut Vec<(String, String)>,
) {
    match rule {
        SyntaxRule::Literal(text) => {
            let key = format!("literal.{text}");
            if let Some(kind) = semantic_tokens.get(&key) {
                out.push((text.clone(), kind.clone()));
            }
        }
        SyntaxRule::Seq(items) | SyntaxRule::Choice(items) => {
            for item in items {
                collect_annotated_literals(item, semantic_tokens, out);
            }
        }
        SyntaxRule::Field(named) | SyntaxRule::Variant(named) => {
            collect_annotated_literals(named.inner.as_ref(), semantic_tokens, out);
        }
        SyntaxRule::Optional { inner } => {
            collect_annotated_literals(inner, semantic_tokens, out);
        }
        SyntaxRule::Repeat { item, .. } => collect_annotated_literals(item, semantic_tokens, out),
        SyntaxRule::Ref { .. } | SyntaxRule::Token { .. } => {}
    }
}

fn recurse_calls_for_type(
    ty: &SyntaxTypeUse,
    expr: &str,
    node_names: &[String],
    borrowed: bool,
    graph_expr: Option<&str>,
) -> Vec<String> {
    match ty {
        SyntaxTypeUse::Optional(inner) => {
            let inner_calls = recurse_calls_for_type(inner, "value", node_names, true, graph_expr);
            if inner_calls.is_empty() {
                Vec::new()
            } else {
                vec![format!(
                    "if let Some(value) = {expr}.as_ref() {{\n{}\n}}",
                    inner_calls
                        .into_iter()
                        .map(|line| format!("    {line}"))
                        .collect::<Vec<_>>()
                        .join("\n")
                )]
            }
        }
        SyntaxTypeUse::Seq(inner) | SyntaxTypeUse::Order(inner) => {
            let inner_calls = recurse_calls_for_type(inner, "value", node_names, true, graph_expr);
            if inner_calls.is_empty() {
                Vec::new()
            } else {
                let iter_expr = if borrowed {
                    expr.to_owned()
                } else {
                    format!("&{expr}")
                };
                vec![format!(
                    "for value in {iter_expr} {{\n{}\n}}",
                    inner_calls
                        .into_iter()
                        .map(|line| format!("    {line}"))
                        .collect::<Vec<_>>()
                        .join("\n")
                )]
            }
        }
        SyntaxTypeUse::Arena { item, key: Some(_) } => {
            let Some(graph_expr) = graph_expr else {
                return Vec::new();
            };
            let SyntaxTypeUse::Ref { name: item_name } = item.as_ref() else {
                return Vec::new();
            };
            let accessor = snake_case(item_name);
            let collect = snake_case(item_name);
            let iter_expr = if borrowed {
                expr.to_owned()
            } else {
                format!("&{expr}")
            };
            vec![format!(
                "for id in {iter_expr} {{\n    if let Some(value) = {graph_expr}.{accessor}(*id) {{\n        collect_{collect}(source, {graph_expr}, value, out);\n    }}\n}}"
            )]
        }
        SyntaxTypeUse::Arena {
            item: inner,
            key: None,
        } => {
            let inner_calls = recurse_calls_for_type(inner, "value", node_names, true, graph_expr);
            if inner_calls.is_empty() {
                Vec::new()
            } else {
                let iter_expr = if borrowed {
                    expr.to_owned()
                } else {
                    format!("&{expr}")
                };
                vec![format!(
                    "for value in {iter_expr} {{\n{}\n}}",
                    inner_calls
                        .into_iter()
                        .map(|line| format!("    {line}"))
                        .collect::<Vec<_>>()
                        .join("\n")
                )]
            }
        }
        SyntaxTypeUse::Pool { item: inner, .. } => {
            let inner_calls = recurse_calls_for_type(inner, "value", node_names, true, graph_expr);
            if inner_calls.is_empty() {
                Vec::new()
            } else {
                let iter_expr = if borrowed {
                    expr.to_owned()
                } else {
                    format!("&{expr}")
                };
                vec![format!(
                    "for value in {iter_expr} {{\n{}\n}}",
                    inner_calls
                        .into_iter()
                        .map(|line| format!("    {line}"))
                        .collect::<Vec<_>>()
                        .join("\n")
                )]
            }
        }
        SyntaxTypeUse::RefTo { id, .. } => {
            recurse_calls_for_type(id, expr, node_names, borrowed, graph_expr)
        }
        SyntaxTypeUse::Ref { name } if node_names.iter().any(|node| node == name) => {
            let expr = if borrowed {
                expr.to_owned()
            } else {
                format!("&{expr}")
            };
            let method = snake_case(name);
            if let Some(graph_expr) = graph_expr {
                vec![format!(
                    "collect_{method}(source, {graph_expr}, {expr}, out);"
                )]
            } else {
                vec![format!("collect_{method}(source, {expr}, out);")]
            }
        }
        _ => Vec::new(),
    }
}

fn render_field_annotation_line(
    repr: &NormalizedRepr,
    ty: &SyntaxTypeUse,
    expr: &str,
    kind: &str,
    node_names: &[String],
) -> Option<String> {
    match ty {
        SyntaxTypeUse::Optional(_)
        | SyntaxTypeUse::Seq(_)
        | SyntaxTypeUse::Order(_)
        | SyntaxTypeUse::Arena { .. }
        | SyntaxTypeUse::Pool { .. } => None,
        SyntaxTypeUse::RefTo { id, .. } => {
            render_field_annotation_line(repr, id, expr, kind, node_names)
        }
        SyntaxTypeUse::Ref { name }
            if is_string_scalar_type(repr, name) || is_int_scalar_type(repr, name) =>
        {
            Some(format!("emit_prov_token(&{expr}.prov, {kind:?}, out);"))
        }
        SyntaxTypeUse::Ref { name } if is_id_type(repr, name) => None,
        SyntaxTypeUse::Ref { name } if node_names.iter().any(|node| node == name) => Some(format!(
            "if let Some(prov) = {expr}.provenance() {{ emit_prov_token(prov, {kind:?}, out); }}"
        )),
        _ => match classify_ref_type(
            repr,
            match ty {
                SyntaxTypeUse::Ref { name } => name,
                _ => unreachable!(),
            },
        ) {
            crate::normalize::NormalizedRefKind::Enum => Some(format!(
                "if let Some(prov) = {expr}.provenance() {{ emit_prov_token(prov, {kind:?}, out); }}"
            )),
            _ => None,
        },
    }
}

fn render_struct_field_annotations(
    repr: &NormalizedRepr,
    type_name: &str,
    fields: &HashMap<String, DocumentedValue<SyntaxTypeUse>>,
    node_names: &[String],
) -> Vec<String> {
    let mut out = Vec::new();
    let mut field_names = fields.keys().cloned().collect::<Vec<_>>();
    field_names.sort();
    for field_name in field_names {
        let target = format!("field.{type_name}.{field_name}");
        let Some(kind) = repr.syntax.semantic_tokens.get(&target) else {
            continue;
        };
        let Some(line) = render_field_annotation_line(
            repr,
            &fields[&field_name].value,
            &format!("node.{}", rust_ident(&field_name)),
            kind,
            node_names,
        ) else {
            continue;
        };
        out.push(line);
    }
    out
}

fn render_variant_field_annotations(
    repr: &NormalizedRepr,
    type_name: &str,
    variant_name: &str,
    fields: &HashMap<String, DocumentedValue<SyntaxTypeUse>>,
    node_names: &[String],
) -> Vec<String> {
    let mut out = Vec::new();
    let mut field_names = fields.keys().cloned().collect::<Vec<_>>();
    field_names.sort();
    for field_name in field_names {
        let target = format!("field.{type_name}.{variant_name}.{field_name}");
        let Some(kind) = repr.syntax.semantic_tokens.get(&target) else {
            continue;
        };
        let Some(line) = render_field_annotation_line(
            repr,
            &fields[&field_name].value,
            &rust_ident(&field_name),
            kind,
            node_names,
        ) else {
            continue;
        };
        out.push(line);
    }
    out
}

pub(crate) fn render_semantic_block(
    repr: &NormalizedRepr,
    node_names: &[String],
) -> Result<String, String> {
    let graph_backed = has_keyed_arenas(repr);
    let mut variant_targets = std::collections::HashSet::new();
    for target in repr.syntax.semantic_tokens.keys() {
        if let Some((type_name, variant_name)) = parse_target(target)? {
            variant_targets.insert((type_name, variant_name));
        }
    }

    let root_name = &repr.syntax.root;
    let root_method = snake_case(root_name);
    let mut collector_rows = Vec::new();

    for node_name in node_names {
        let decl = &repr.nodes[node_name].value;
        let rule = repr.syntax.rules.get(node_name);
        let method = snake_case(node_name);

        let mut literal_annotations = Vec::new();
        if let Some(rule) = rule {
            collect_annotated_literals(
                rule,
                &repr.syntax.semantic_tokens,
                &mut literal_annotations,
            );
        }
        let literal_rows = literal_annotations
            .into_iter()
            .map(|(text, kind)| {
                format!("emit_literal_token(source, current_prov, {text:?}, {kind:?}, out);")
            })
            .collect::<Vec<_>>();

        let body = match decl {
            NormalizedNodeDecl::Record { fields, .. } => {
                let field_rows =
                    render_struct_field_annotations(repr, node_name, fields, node_names);
                let recurse_rows = {
                    let mut field_names = fields.keys().cloned().collect::<Vec<_>>();
                    field_names.sort();
                    field_names
                        .into_iter()
                        .flat_map(|field_name| {
                            recurse_calls_for_type(
                                &fields[&field_name].value,
                                &format!("node.{}", rust_ident(&field_name)),
                                node_names,
                                false,
                                graph_backed.then_some("graph"),
                            )
                        })
                        .collect::<Vec<_>>()
                };
                [
                    "let current_prov = node.provenance();".to_owned(),
                    literal_rows.join("\n"),
                    field_rows.join("\n"),
                    recurse_rows.join("\n"),
                ]
                .into_iter()
                .filter(|row| !row.is_empty())
                .collect::<Vec<_>>()
                .join("\n")
            }
            NormalizedNodeDecl::Enum(variants) => {
                let mut variant_names = variants.keys().cloned().collect::<Vec<_>>();
                variant_names.sort();
                let arms = variant_names
                    .into_iter()
                    .map(|variant_name| {
                        let variant = &variants[&variant_name].value;
                        let variant_rule_literals = match rule {
                            Some(SyntaxRule::Choice(items)) => items
                                .iter()
                                .find_map(|item| match item {
                                    SyntaxRule::Variant(named) if named.name == variant_name => {
                                        let mut rows = Vec::new();
                                        collect_annotated_literals(
                                            named.inner.as_ref(),
                                            &repr.syntax.semantic_tokens,
                                            &mut rows,
                                        );
                                        Some(rows)
                                    }
                                    _ => None,
                                })
                                .unwrap_or_default(),
                            _ => Vec::new(),
                        };
                        let literal_rows = variant_rule_literals
                            .into_iter()
                            .map(|(text, kind)| {
                                format!(
                                    "emit_literal_token(source, current_prov, {text:?}, {kind:?}, out);"
                                )
                            })
                            .collect::<Vec<_>>();
                        let variant_row = if variant_targets
                            .contains(&(node_name.clone(), variant_name.clone()))
                        {
                            repr
                                .syntax
                                .semantic_tokens
                                .get(&format!("variant.{node_name}.{variant_name}"))
                                .map(|kind| {
                                    format!(
                                        "if let Some(prov) = node.provenance() {{ emit_prov_token(prov, {kind:?}, out); }}"
                                    )
                                })
                                .unwrap_or_default()
                        } else {
                            String::new()
                        };
                        match variant {
                            NormalizedNodeDecl::Record { fields, .. } => {
                                let prov_tag = repr
                                    .common
                                    .get("provenance")
                                    .and_then(|ty| match ty {
                                        SyntaxTypeUse::Ref { name } => Some(name.as_str()),
                                        _ => None,
                                    })
                                    .unwrap_or("Prov");
                                let mut field_names = fields.keys().cloned().collect::<Vec<_>>();
                                field_names.sort();
                                let pattern_fields = if is_prov_only_struct(fields, prov_tag) {
                                    "value".to_owned()
                                } else if field_names.is_empty() {
                                    "..".to_owned()
                                } else {
                                    let mut parts = field_names
                                        .iter()
                                        .map(|field_name| rust_ident(field_name))
                                        .collect::<Vec<_>>();
                                    if !field_names.iter().any(|name| name == "prov") {
                                        parts.push("..".to_owned());
                                    }
                                    parts.join(", ")
                                };
                                let field_rows = render_variant_field_annotations(
                                    repr,
                                    node_name,
                                    &variant_name,
                                    fields,
                                    node_names,
                                );
                                let recurse_rows = field_names
                                    .iter()
                                    .flat_map(|field_name| {
                                    recurse_calls_for_type(
                                        &fields[field_name].value,
                                        &rust_ident(field_name),
                                        node_names,
                                        true,
                                        graph_backed.then_some("graph"),
                                    )
                                })
                                    .collect::<Vec<_>>();
                                let rows = [
                                    "let current_prov = node.provenance();".to_owned(),
                                    variant_row,
                                    literal_rows.join("\n"),
                                    field_rows.join("\n"),
                                    recurse_rows.join("\n"),
                                ]
                                .into_iter()
                                .filter(|row| !row.is_empty())
                                .collect::<Vec<_>>()
                                .join("\n");
                                if is_prov_only_struct(fields, prov_tag) {
                                    format!(
                                        "{node_name}::{variant_name}({pattern_fields}) => {{\n{}\n        }}",
                                        rows.lines()
                                            .map(|line| format!("            {line}"))
                                            .collect::<Vec<_>>()
                                            .join("\n")
                                    )
                                } else {
                                    format!(
                                        "{node_name}::{variant_name} {{ {pattern_fields} }} => {{\n{}\n        }}",
                                        rows.lines()
                                            .map(|line| format!("            {line}"))
                                            .collect::<Vec<_>>()
                                            .join("\n")
                                    )
                                }
                            }
                            NormalizedNodeDecl::Enum(_) => {
                                panic!(
                                    "nested enum variants are unsupported for semantic generation: {node_name}.{variant_name}"
                                )
                            }
                        }
                    })
                    .collect::<Vec<_>>()
                    .join(",\n");
                format!("match node {{\n        {arms}\n    }}")
            }
        };

        if graph_backed {
            collector_rows.push(format!(
                "fn collect_{method}(source: &str, graph: &Graph, node: &{node_name}, out: &mut Vec<SemanticToken>) {{\n    let _ = graph;\n    {body}\n}}"
            ));
        } else {
            collector_rows.push(format!(
                "fn collect_{method}(source: &str, node: &{node_name}, out: &mut Vec<SemanticToken>) {{\n    {body}\n}}"
            ));
        }
    }

    let semantic_head = if graph_backed {
        format!(
            r#"pub fn semantic_tokens(source: &str) -> Vec<SemanticToken> {{
    let mut graph = Graph::new();
    let Ok(handle) = parse_root_into_graph(&mut graph, source) else {{
        return Vec::new();
    }};
    semantic_tokens_in_graph(source, &graph, handle)
}}

pub fn semantic_tokens_in_graph(
    source: &str,
    graph: &Graph,
    handle: {root_name}Handle,
) -> Vec<SemanticToken> {{
    let Some(root) = graph.root(handle) else {{
        return Vec::new();
    }};
    let mut out = Vec::new();
    collect_{root_method}(source, graph, root, &mut out);
    out.sort_by_key(|token| (token.start, token.end, token.kind));
    out.dedup_by(|a, b| a.start == b.start && a.end == b.end && a.kind == b.kind);
    out
}}
"#,
            root_name = root_name,
            root_method = root_method
        )
    } else {
        format!(
            r#"pub fn semantic_tokens(source: &str) -> Vec<SemanticToken> {{
    let Ok(root) = parse_root_text_rich(source, None) else {{
        return Vec::new();
    }};
    let mut out = Vec::new();
    collect_{root_method}(source, &root, &mut out);
    out.sort_by_key(|token| (token.start, token.end, token.kind));
    out.dedup_by(|a, b| a.start == b.start && a.end == b.end && a.kind == b.kind);
    out
}}
"#,
            root_method = root_method
        )
    };

    Ok(format!(
        r#"
{semantic_head}

fn emit_prov_token(prov: &Prov, kind: &'static str, out: &mut Vec<SemanticToken>) {{
    let Some(span) = prov.span.as_ref() else {{
        return;
    }};
    if span.end <= span.start {{
        return;
    }}
    out.push(SemanticToken {{
        start: span.start,
        end: span.end,
        kind,
    }});
}}

fn emit_literal_token(
    source: &str,
    prov: Option<&Prov>,
    text: &str,
    kind: &'static str,
    out: &mut Vec<SemanticToken>,
) {{
    let Some(prov) = prov else {{
        return;
    }};
    let Some(span) = prov.span.as_ref() else {{
        return;
    }};
    let start = span.start as usize;
    let end = span.end as usize;
    if start > end || end > source.len() {{
        return;
    }}
    let Some(offset) = source[start..end].find(text) else {{
        return;
    }};
    out.push(SemanticToken {{
        start: (start + offset) as u32,
        end: (start + offset + text.len()) as u32,
        kind,
    }});
}}

{collector_rows}
"#,
        semantic_head = semantic_head,
        collector_rows = collector_rows.join("\n\n"),
    ))
}

fn has_keyed_arenas(repr: &NormalizedRepr) -> bool {
    repr.nodes.values().any(|decl| match &decl.value {
        NormalizedNodeDecl::Record { fields, .. } => fields
            .values()
            .any(|field| matches!(field.value, SyntaxTypeUse::Arena { key: Some(_), .. })),
        NormalizedNodeDecl::Enum(variants) => variants.values().any(|variant| {
            matches!(
                &variant.value,
                NormalizedNodeDecl::Record { fields, .. }
                    if fields.values().any(
                        |field| matches!(field.value, SyntaxTypeUse::Arena { key: Some(_), .. })
                    )
            )
        }),
    })
}
