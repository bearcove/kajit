use std::collections::HashMap;

use crate::normalize::{
    DocumentedValue, NormalizedNodeDecl, NormalizedRepr, SyntaxTypeUse, classify_ref_type,
    is_id_type, is_int_scalar_type, is_string_scalar_type,
};
use crate::render_helpers::{is_prov_only_struct, rust_ident, snake_case};

fn markdown_literal(lines: &[String]) -> String {
    lines.join("\n")
}

fn combined_markdown(
    parent_doc: Option<&[String]>,
    child_doc: Option<&[String]>,
) -> Option<String> {
    match (parent_doc, child_doc) {
        (None, None) => None,
        (Some(parent), None) => Some(markdown_literal(parent)),
        (None, Some(child)) => Some(markdown_literal(child)),
        (Some(parent), Some(child)) => Some(format!(
            "{}\n\n{}",
            markdown_literal(parent),
            markdown_literal(child)
        )),
    }
}

fn recurse_calls_for_type(ty: &SyntaxTypeUse, expr: &str, node_names: &[String]) -> Vec<String> {
    match ty {
        SyntaxTypeUse::Optional(inner) => {
            let inner_calls = recurse_calls_for_type(inner, "value", node_names);
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
        SyntaxTypeUse::Seq(inner) => {
            let inner_calls = recurse_calls_for_type(inner, "value", node_names);
            if inner_calls.is_empty() {
                Vec::new()
            } else {
                vec![format!(
                    "for value in {expr} {{\n{}\n}}",
                    inner_calls
                        .into_iter()
                        .map(|line| format!("    {line}"))
                        .collect::<Vec<_>>()
                        .join("\n")
                )]
            }
        }
        SyntaxTypeUse::Ref { name } if node_names.iter().any(|node| node == name) => {
            vec![format!("collect_{}({expr}, out);", snake_case(name))]
        }
        _ => Vec::new(),
    }
}

fn render_hover_line(
    repr: &NormalizedRepr,
    ty: &SyntaxTypeUse,
    expr: &str,
    markdown: &str,
    priority: u8,
    node_names: &[String],
) -> Option<String> {
    match ty {
        SyntaxTypeUse::Optional(_) | SyntaxTypeUse::Seq(_) => None,
        SyntaxTypeUse::Ref { name }
            if is_string_scalar_type(repr, name) || is_int_scalar_type(repr, name) =>
        {
            Some(format!(
                "emit_hover(&{expr}.prov, {markdown:?}, {priority}, out);"
            ))
        }
        SyntaxTypeUse::Ref { name } if is_id_type(repr, name) => None,
        SyntaxTypeUse::Ref { name } if node_names.iter().any(|node| node == name) => Some(format!(
            "if let Some(prov) = {expr}.provenance() {{ emit_hover(prov, {markdown:?}, {priority}, out); }}"
        )),
        _ => match classify_ref_type(
            repr,
            match ty {
                SyntaxTypeUse::Ref { name } => name,
                _ => unreachable!(),
            },
        ) {
            _ => None,
        },
    }
}

fn render_struct_field_hovers(
    repr: &NormalizedRepr,
    fields: &HashMap<String, DocumentedValue<SyntaxTypeUse>>,
    parent_doc: Option<&[String]>,
    node_names: &[String],
) -> Vec<String> {
    let mut out = Vec::new();
    let mut field_names = fields.keys().cloned().collect::<Vec<_>>();
    field_names.sort();
    for field_name in field_names {
        let Some(markdown) = combined_markdown(parent_doc, fields[&field_name].doc.as_deref())
        else {
            continue;
        };
        let Some(line) = render_hover_line(
            repr,
            &fields[&field_name].value,
            &format!("node.{}", rust_ident(&field_name)),
            &markdown,
            30,
            node_names,
        ) else {
            continue;
        };
        out.push(line);
    }
    out
}

fn render_variant_field_hovers(
    repr: &NormalizedRepr,
    fields: &HashMap<String, DocumentedValue<SyntaxTypeUse>>,
    parent_doc: Option<&[String]>,
    node_names: &[String],
) -> Vec<String> {
    let mut out = Vec::new();
    let mut field_names = fields.keys().cloned().collect::<Vec<_>>();
    field_names.sort();
    for field_name in field_names {
        let Some(markdown) = combined_markdown(parent_doc, fields[&field_name].doc.as_deref())
        else {
            continue;
        };
        let Some(line) = render_hover_line(
            repr,
            &fields[&field_name].value,
            &rust_ident(&field_name),
            &markdown,
            30,
            node_names,
        ) else {
            continue;
        };
        out.push(line);
    }
    out
}

pub(crate) fn render_hover_block(
    repr: &NormalizedRepr,
    node_names: &[String],
) -> Result<String, String> {
    let root_name = &repr.syntax.root;
    let root_method = snake_case(root_name);
    let mut collector_rows = Vec::new();

    for node_name in node_names {
        let decl_doc = repr.nodes[node_name].doc.as_deref();
        let decl = &repr.nodes[node_name].value;
        let method = snake_case(node_name);

        let body = match decl {
            NormalizedNodeDecl::Record { fields, .. } => {
                let mut rows = Vec::new();
                if let Some(doc) = decl_doc {
                    rows.push(format!(
                        "if let Some(prov) = node.provenance() {{ emit_hover(prov, {:?}, 10, out); }}",
                        markdown_literal(doc)
                    ));
                }
                rows.extend(render_struct_field_hovers(
                    repr, fields, decl_doc, node_names,
                ));
                let mut field_names = fields.keys().cloned().collect::<Vec<_>>();
                field_names.sort();
                rows.extend(field_names.into_iter().flat_map(|field_name| {
                    recurse_calls_for_type(
                        &fields[&field_name].value,
                        &format!("&node.{}", rust_ident(&field_name)),
                        node_names,
                    )
                }));
                rows.join("\n")
            }
            NormalizedNodeDecl::Enum(variants) => {
                let mut variant_names = variants.keys().cloned().collect::<Vec<_>>();
                variant_names.sort();
                let arms = variant_names
                    .into_iter()
                    .map(|variant_name| {
                        let variant_doc = variants[&variant_name].doc.as_deref();
                        let variant = &variants[&variant_name].value;
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

                                let mut rows = Vec::new();
                                if let Some(doc) = variant_doc {
                                    rows.push(format!(
                                        "if let Some(prov) = node.provenance() {{ emit_hover(prov, {:?}, 40, out); }}",
                                        markdown_literal(doc)
                                    ));
                                }
                                rows.extend(render_variant_field_hovers(repr, fields, variant_doc, node_names));
                                rows.extend(field_names.iter().flat_map(|field_name| {
                                    recurse_calls_for_type(
                                        &fields[field_name].value,
                                        &rust_ident(field_name),
                                        node_names,
                                    )
                                }));
                                let rows = rows.join("\n");
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
                                    "nested enum variants are unsupported for hover generation: {node_name}.{variant_name}"
                                )
                            }
                        }
                    })
                    .collect::<Vec<_>>()
                    .join(",\n");
                format!("match node {{\n        {arms}\n    }}")
            }
        };

        collector_rows.push(format!(
            "fn collect_{method}(node: &{node_name}, out: &mut Vec<HoverEntry>) {{\n    {body}\n}}"
        ));
    }

    Ok(format!(
        r#"
pub fn hover_entries(source: &str) -> Vec<HoverEntry> {{
    let Ok(root) = parse_root_text_rich(source, None) else {{
        return Vec::new();
    }};
    let mut out = Vec::new();
    collect_{root_method}(&root, &mut out);
    out.sort_by_key(|entry| (entry.start, entry.end, entry.priority));
    out.dedup_by(|a, b| a.start == b.start && a.end == b.end && a.markdown == b.markdown);
    out
}}

fn emit_hover(prov: &Prov, markdown: &str, priority: u8, out: &mut Vec<HoverEntry>) {{
    let Some(span) = prov.span.as_ref() else {{
        return;
    }};
    if span.end <= span.start {{
        return;
    }}
    out.push(HoverEntry {{
        start: span.start,
        end: span.end,
        markdown: markdown.to_owned(),
        priority,
    }});
}}

{collector_rows}
"#,
        collector_rows = collector_rows.join("\n\n"),
    ))
}
