use std::collections::HashMap;

use crate::normalize::{
    DocumentedValue, NormalizedNodeDecl, NormalizedRepr, SyntaxTypeUse, is_id_type,
    is_int_scalar_type, is_string_scalar_type,
};
use crate::render_helpers::{is_prov_only_struct, rust_ident, snake_case};

#[derive(Debug, Clone)]
enum TemplatePart {
    Literal(String),
    Field {
        name: String,
        joiner: Option<String>,
    },
    Optional {
        name: String,
        parts: Vec<TemplatePart>,
    },
}

pub(crate) fn render_formatter_block(
    repr: &NormalizedRepr,
    node_names: &[String],
) -> Result<String, String> {
    let graph_backed = has_keyed_arenas(repr);
    let mut formatters = Vec::new();
    let root_name = &repr.syntax.root;
    let root_fn = snake_case(root_name);

    for node_name in node_names {
        let decl = &repr
            .nodes
            .get(node_name)
            .ok_or_else(|| format!("missing node declaration for {node_name}"))?
            .value;
        formatters.push(render_node_formatter(node_name, decl, repr, node_names)?);
    }

    let graph_rows = if graph_backed {
        format!(
            r#"
pub fn format_root_in_graph(graph: &Graph, handle: {root_name}Handle) -> Result<String, String> {{
    let Some(node) = graph.root(handle) else {{
        return Err(format!("unknown root handle {{:?}}", handle));
    }};
    Ok(format_{root_fn}_in_graph_node(graph, node))
}}

pub fn format_{root_fn}_in_graph(graph: &Graph, handle: {root_name}Handle) -> Result<String, String> {{
    format_root_in_graph(graph, handle)
}}
"#
        )
    } else {
        String::new()
    };

    Ok(format!(
        r#"
pub fn format_root_text(node: &{root_name}) -> String {{
    format_{root_fn}(node)
}}

pub fn format_{root_fn}_text(node: &{root_name}) -> String {{
    format_root_text(node)
}}

impl std::fmt::Display for {root_name} {{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {{
        f.write_str(&format_root_text(self))
    }}
}}

{graph_rows}

const INDENT_WIDTH: usize = 4;

struct Writer {{
    out: String,
    indent: usize,
    at_line_start: bool,
}}

impl Writer {{
    fn new() -> Self {{
        Self {{
            out: String::new(),
            indent: 0,
            at_line_start: false,
        }}
    }}

    fn text(&mut self, text: &str) {{
        for ch in text.chars() {{
            if self.at_line_start && ch != '\n' {{
                for _ in 0..self.indent {{
                    self.out.push(' ');
                }}
                self.at_line_start = false;
            }}

            self.out.push(ch);
            if ch == '\n' {{
                self.at_line_start = true;
            }}
        }}
    }}

    fn with_indent(&mut self, f: impl FnOnce(&mut Self)) {{
        self.indent += INDENT_WIDTH;
        f(self);
        self.indent -= INDENT_WIDTH;
    }}

    fn finish(self) -> String {{
        self.out
    }}
}}

{formatters}
"#,
        root_name = root_name,
        root_fn = root_fn,
        graph_rows = graph_rows,
        formatters = formatters.join("\n\n"),
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

fn render_node_formatter(
    node_name: &str,
    decl: &NormalizedNodeDecl,
    repr: &NormalizedRepr,
    node_names: &[String],
) -> Result<String, String> {
    let graph_backed = has_keyed_arenas(repr);
    let fn_name = format!("format_{}", snake_case(node_name));
    let write_name = format!("write_{}", snake_case(node_name));
    let fn_name_in_graph = format!("{fn_name}_in_graph_node");
    let write_name_in_graph = format!("{write_name}_in_graph");
    match decl {
        NormalizedNodeDecl::Record { fields, .. } => {
            if let Some(template) = repr.syntax.canonical_print.get(node_name) {
                let parts = parse_template(template)?;
                let body = render_template_lines(&parts, "node", fields, None, repr, node_names)?;
                let base = format!(
                    "pub fn {fn_name}(node: &{node_name}) -> String {{\n    let mut w = Writer::new();\n    {write_name}(&mut w, node);\n    w.finish()\n}}\n\nfn {write_name}(w: &mut Writer, node: &{node_name}) {{\n{body}\n}}"
                );
                if !graph_backed {
                    return Ok(base);
                }

                let body_in_graph = render_template_lines_in_graph(
                    &parts, "graph", "node", fields, None, repr, node_names,
                )?;
                Ok(format!(
                    "{base}\n\npub fn {fn_name_in_graph}(graph: &Graph, node: &{node_name}) -> String {{\n    let mut w = Writer::new();\n    {write_name_in_graph}(&mut w, graph, node);\n    w.finish()\n}}\n\nfn {write_name_in_graph}(w: &mut Writer, graph: &Graph, node: &{node_name}) {{\n{body_in_graph}\n}}",
                ))
            } else {
                let base = format!(
                    "pub fn {fn_name}(node: &{node_name}) -> String {{\n    format!(\"{{:?}}\", node)\n}}\n\nfn {write_name}(w: &mut Writer, node: &{node_name}) {{\n    w.text(&format!(\"{{:?}}\", node));\n}}"
                );
                if !graph_backed {
                    Ok(base)
                } else {
                    Ok(format!(
                        "{base}\n\npub fn {fn_name_in_graph}(graph: &Graph, node: &{node_name}) -> String {{\n    let _ = graph;\n    {fn_name}(node)\n}}\n\nfn {write_name_in_graph}(w: &mut Writer, graph: &Graph, node: &{node_name}) {{\n    let _ = graph;\n    {write_name}(w, node)\n}}"
                    ))
                }
            }
        }
        NormalizedNodeDecl::Enum(variants) => {
            let mut variant_names = variants.keys().cloned().collect::<Vec<_>>();
            variant_names.sort();
            let mut arms = Vec::new();
            let mut arms_in_graph = Vec::new();
            for variant_name in variant_names {
                let variant_decl = variants.get(&variant_name).unwrap();
                let NormalizedNodeDecl::Record { fields, .. } = &variant_decl.value else {
                    return Err(format!(
                        "unsupported nested enum variant {node_name}.{variant_name}"
                    ));
                };
                let template_key = format!("{node_name}.{variant_name}");
                if let Some(template) = repr.syntax.canonical_print.get(&template_key) {
                    let parts = parse_template(template)?;
                    let used = collect_used_fields(&parts);

                    let mut field_names = fields.keys().cloned().collect::<Vec<_>>();
                    field_names.sort();
                    let prov_tag = repr
                        .common
                        .get("provenance")
                        .and_then(|ty| match ty {
                            SyntaxTypeUse::Ref { name } => Some(name.as_str()),
                            _ => None,
                        })
                        .unwrap_or("Prov");
                    let pattern = if is_prov_only_struct(fields, prov_tag) {
                        "_value".to_owned()
                    } else if used.is_empty() {
                        "..".to_owned()
                    } else {
                        let mut binders = used
                            .iter()
                            .map(|field| rust_ident(field))
                            .collect::<Vec<_>>();
                        if used.len() != field_names.len() {
                            binders.push("..".to_owned());
                        }
                        binders.join(", ")
                    };

                    let overrides = used
                        .iter()
                        .map(|field| {
                            (
                                field.clone(),
                                (
                                    rust_ident(field),
                                    fields
                                        .get(field)
                                        .expect("used field should exist")
                                        .value
                                        .clone(),
                                ),
                            )
                        })
                        .collect::<HashMap<_, _>>();
                    let body = render_template_lines(
                        &parts,
                        "node",
                        fields,
                        Some(&overrides),
                        repr,
                        node_names,
                    )?;
                    let body_in_graph = if graph_backed {
                        render_template_lines_in_graph(
                            &parts,
                            "graph",
                            "node",
                            fields,
                            Some(&overrides),
                            repr,
                            node_names,
                        )?
                    } else {
                        String::new()
                    };
                    if is_prov_only_struct(fields, prov_tag) {
                        arms.push(format!(
                            "        {node_name}::{variant_name}({pattern}) => {{\n{body}\n        }}"
                        ));
                        if graph_backed {
                            arms_in_graph.push(format!(
                                "        {node_name}::{variant_name}({pattern}) => {{\n{body_in_graph}\n        }}"
                            ));
                        }
                    } else {
                        arms.push(format!(
                            "        {node_name}::{variant_name} {{ {pattern} }} => {{\n{body}\n        }}"
                        ));
                        if graph_backed {
                            arms_in_graph.push(format!(
                                "        {node_name}::{variant_name} {{ {pattern} }} => {{\n{body_in_graph}\n        }}"
                            ));
                        }
                    }
                } else {
                    let prov_tag = repr
                        .common
                        .get("provenance")
                        .and_then(|ty| match ty {
                            SyntaxTypeUse::Ref { name } => Some(name.as_str()),
                            _ => None,
                        })
                        .unwrap_or("Prov");
                    if is_prov_only_struct(fields, prov_tag) {
                        arms.push(format!(
                            "        other @ {node_name}::{variant_name}(..) => {{ w.text(&format!(\"{{:?}}\", other)); }}"
                        ));
                        if graph_backed {
                            arms_in_graph.push(format!(
                                "        other @ {node_name}::{variant_name}(..) => {{ w.text(&format!(\"{{:?}}\", other)); }}"
                            ));
                        }
                    } else {
                        arms.push(format!(
                            "        other @ {node_name}::{variant_name} {{ .. }} => {{ w.text(&format!(\"{{:?}}\", other)); }}"
                        ));
                        if graph_backed {
                            arms_in_graph.push(format!(
                                "        other @ {node_name}::{variant_name} {{ .. }} => {{ w.text(&format!(\"{{:?}}\", other)); }}"
                            ));
                        }
                    }
                }
            }
            let base = format!(
                "pub fn {fn_name}(node: &{node_name}) -> String {{\n    let mut w = Writer::new();\n    {write_name}(&mut w, node);\n    w.finish()\n}}\n\nfn {write_name}(w: &mut Writer, node: &{node_name}) {{\n    match node {{\n{}\n    }}\n}}",
                arms.join(",\n")
            );
            if !graph_backed {
                return Ok(base);
            }
            Ok(format!(
                "{base}\n\npub fn {fn_name_in_graph}(graph: &Graph, node: &{node_name}) -> String {{\n    let mut w = Writer::new();\n    {write_name_in_graph}(&mut w, graph, node);\n    w.finish()\n}}\n\nfn {write_name_in_graph}(w: &mut Writer, graph: &Graph, node: &{node_name}) {{\n    let _ = graph;\n    match node {{\n{}\n    }}\n}}",
                arms_in_graph.join(",\n")
            ))
        }
    }
}

fn render_template_lines(
    parts: &[TemplatePart],
    node_expr: &str,
    fields: &HashMap<String, DocumentedValue<SyntaxTypeUse>>,
    overrides: Option<&HashMap<String, (String, SyntaxTypeUse)>>,
    repr: &NormalizedRepr,
    node_names: &[String],
) -> Result<String, String> {
    let mut lines = Vec::new();
    let mut at_line_start = false;
    for part in parts {
        match part {
            TemplatePart::Literal(text) => {
                lines.push(format!("    w.text({text:?});"));
                at_line_start = text.ends_with('\n');
            }
            TemplatePart::Field { name, joiner } => {
                let (expr, ty) = lookup_field(name, node_expr, fields, overrides)?;
                lines.extend(render_value_write_lines(
                    repr,
                    "w",
                    &expr,
                    &ty,
                    joiner.as_deref(),
                    node_names,
                    at_line_start,
                )?);
                at_line_start = false;
            }
            TemplatePart::Optional { name, parts } => {
                let (expr, ty) = lookup_field(name, node_expr, fields, overrides)?;
                let SyntaxTypeUse::Optional(inner_ty) = ty else {
                    return Err(format!("optional template field {name:?} is not optional"));
                };
                let mut nested_overrides = overrides.cloned().unwrap_or_default();
                nested_overrides.insert(name.clone(), ("value".to_owned(), (*inner_ty).clone()));
                let nested = render_template_lines(
                    parts,
                    node_expr,
                    fields,
                    Some(&nested_overrides),
                    repr,
                    node_names,
                )?;
                lines.push(format!("    if let Some(value) = {expr}.as_ref() {{"));
                lines.push(nested);
                lines.push("    }".to_owned());
                at_line_start = false;
            }
        }
    }
    Ok(lines.join("\n"))
}

fn render_template_lines_in_graph(
    parts: &[TemplatePart],
    graph_expr: &str,
    node_expr: &str,
    fields: &HashMap<String, DocumentedValue<SyntaxTypeUse>>,
    overrides: Option<&HashMap<String, (String, SyntaxTypeUse)>>,
    repr: &NormalizedRepr,
    node_names: &[String],
) -> Result<String, String> {
    let mut lines = Vec::new();
    let mut at_line_start = false;
    for part in parts {
        match part {
            TemplatePart::Literal(text) => {
                lines.push(format!("    w.text({text:?});"));
                at_line_start = text.ends_with('\n');
            }
            TemplatePart::Field { name, joiner } => {
                let (expr, ty) = lookup_field(name, node_expr, fields, overrides)?;
                lines.extend(render_value_write_lines_in_graph(
                    repr,
                    "w",
                    graph_expr,
                    &expr,
                    &ty,
                    joiner.as_deref(),
                    node_names,
                    at_line_start,
                )?);
                at_line_start = false;
            }
            TemplatePart::Optional { name, parts } => {
                let (expr, ty) = lookup_field(name, node_expr, fields, overrides)?;
                let SyntaxTypeUse::Optional(inner_ty) = ty else {
                    return Err(format!("optional template field {name:?} is not optional"));
                };
                let mut nested_overrides = overrides.cloned().unwrap_or_default();
                nested_overrides.insert(name.clone(), ("value".to_owned(), (*inner_ty).clone()));
                let nested = render_template_lines_in_graph(
                    parts,
                    graph_expr,
                    node_expr,
                    fields,
                    Some(&nested_overrides),
                    repr,
                    node_names,
                )?;
                lines.push(format!("    if let Some(value) = {expr}.as_ref() {{"));
                lines.push(nested);
                lines.push("    }".to_owned());
                at_line_start = false;
            }
        }
    }
    Ok(lines.join("\n"))
}

fn lookup_field(
    field_name: &str,
    node_expr: &str,
    fields: &HashMap<String, DocumentedValue<SyntaxTypeUse>>,
    overrides: Option<&HashMap<String, (String, SyntaxTypeUse)>>,
) -> Result<(String, SyntaxTypeUse), String> {
    if let Some((expr, ty)) = overrides.and_then(|map| map.get(field_name)) {
        return Ok((expr.clone(), ty.clone()));
    }

    let ty = &fields
        .get(field_name)
        .ok_or_else(|| format!("template refers to unknown field {field_name:?}"))?
        .value;
    Ok((
        format!("{node_expr}.{}", rust_ident(field_name)),
        ty.clone(),
    ))
}

fn render_value_write_lines(
    repr: &NormalizedRepr,
    writer_name: &str,
    expr: &str,
    ty: &SyntaxTypeUse,
    joiner: Option<&str>,
    node_names: &[String],
    at_line_start: bool,
) -> Result<Vec<String>, String> {
    match ty {
        SyntaxTypeUse::Optional(_) => {
            Err("optional formatting must be handled by TemplatePart::Optional".to_owned())
        }
        SyntaxTypeUse::Seq(inner) | SyntaxTypeUse::Order(inner) => {
            let sep = joiner.unwrap_or("\n");
            let mut lines = vec![format!(
                "    for (idx, value) in {expr}.iter().enumerate() {{"
            )];
            lines.push("        if idx != 0 {".to_owned());
            lines.push(format!("            {writer_name}.text({sep:?});"));
            lines.push("        }".to_owned());
            if should_indent_value(inner, node_names) && at_line_start {
                lines.push(format!(
                    "        {writer_name}.with_indent(|{writer_name}| {{"
                ));
                lines.extend(
                    render_value_write_lines(
                        repr,
                        writer_name,
                        "value",
                        inner,
                        None,
                        node_names,
                        false,
                    )?
                    .into_iter()
                    .map(|line| format!("    {line}")),
                );
                lines.push("        });".to_owned());
            } else {
                lines.extend(
                    render_value_write_lines(
                        repr,
                        writer_name,
                        "value",
                        inner,
                        None,
                        node_names,
                        false,
                    )?
                    .into_iter()
                    .map(|line| format!("    {line}")),
                );
            }
            lines.push("    }".to_owned());
            Ok(lines)
        }
        SyntaxTypeUse::Arena { key: Some(key), .. } => {
            let sep = joiner.unwrap_or("\n");
            let mut lines = vec![format!(
                "    for (idx, value) in {expr}.iter().enumerate() {{"
            )];
            lines.push("        if idx != 0 {".to_owned());
            lines.push(format!("            {writer_name}.text({sep:?});"));
            lines.push("        }".to_owned());
            lines.push(format!("        {writer_name}.text(&value.0.to_string());"));
            lines.push("    }".to_owned());
            let _ = key;
            Ok(lines)
        }
        SyntaxTypeUse::Arena {
            item: inner,
            key: None,
        } => {
            let sep = joiner.unwrap_or("\n");
            let mut lines = vec![format!(
                "    for (idx, value) in {expr}.iter().enumerate() {{"
            )];
            lines.push("        if idx != 0 {".to_owned());
            lines.push(format!("            {writer_name}.text({sep:?});"));
            lines.push("        }".to_owned());
            lines.extend(
                render_value_write_lines(
                    repr,
                    writer_name,
                    "value",
                    inner,
                    None,
                    node_names,
                    false,
                )?
                .into_iter()
                .map(|line| format!("    {line}")),
            );
            lines.push("    }".to_owned());
            Ok(lines)
        }
        SyntaxTypeUse::Pool { item: inner, .. } => {
            let sep = joiner.unwrap_or("\n");
            let mut lines = vec![format!(
                "    for (idx, value) in {expr}.iter().enumerate() {{"
            )];
            lines.push("        if idx != 0 {".to_owned());
            lines.push(format!("            {writer_name}.text({sep:?});"));
            lines.push("        }".to_owned());
            if should_indent_value(inner, node_names) && at_line_start {
                lines.push(format!(
                    "        {writer_name}.with_indent(|{writer_name}| {{"
                ));
                lines.extend(
                    render_value_write_lines(
                        repr,
                        writer_name,
                        "value",
                        inner,
                        None,
                        node_names,
                        false,
                    )?
                    .into_iter()
                    .map(|line| format!("    {line}")),
                );
                lines.push("        });".to_owned());
            } else {
                lines.extend(
                    render_value_write_lines(
                        repr,
                        writer_name,
                        "value",
                        inner,
                        None,
                        node_names,
                        false,
                    )?
                    .into_iter()
                    .map(|line| format!("    {line}")),
                );
            }
            lines.push("    }".to_owned());
            Ok(lines)
        }
        SyntaxTypeUse::RefTo { id, .. } => render_value_write_lines(
            repr,
            writer_name,
            expr,
            id,
            joiner,
            node_names,
            at_line_start,
        ),
        SyntaxTypeUse::Ref { name } if node_names.iter().any(|node| node == name) => {
            let write_name = format!("write_{}", snake_case(name));
            if at_line_start {
                Ok(vec![format!(
                    "    {writer_name}.with_indent(|{writer_name}| {write_name}({writer_name}, &{expr}));"
                )])
            } else {
                Ok(vec![format!("    {write_name}({writer_name}, &{expr});")])
            }
        }
        SyntaxTypeUse::Ref { name } => Ok(vec![match () {
            _ if is_string_scalar_type(repr, name) => {
                format!("    {writer_name}.text(&{expr}.text);")
            }
            _ if is_int_scalar_type(repr, name) => {
                format!("    {writer_name}.text(&{expr}.value.to_string());")
            }
            _ if is_id_type(repr, name) => {
                format!("    {writer_name}.text(&{expr}.0.to_string());")
            }
            _ => format!("    {writer_name}.text(&format!(\"{{:?}}\", {expr}));"),
        }]),
    }
}

fn render_value_write_lines_in_graph(
    repr: &NormalizedRepr,
    writer_name: &str,
    graph_name: &str,
    expr: &str,
    ty: &SyntaxTypeUse,
    joiner: Option<&str>,
    node_names: &[String],
    at_line_start: bool,
) -> Result<Vec<String>, String> {
    match ty {
        SyntaxTypeUse::Optional(_) => {
            Err("optional formatting must be handled by TemplatePart::Optional".to_owned())
        }
        SyntaxTypeUse::Seq(inner) | SyntaxTypeUse::Order(inner) => {
            let sep = joiner.unwrap_or("\n");
            let mut lines = vec![format!(
                "    for (idx, value) in {expr}.iter().enumerate() {{"
            )];
            lines.push("        if idx != 0 {".to_owned());
            lines.push(format!("            {writer_name}.text({sep:?});"));
            lines.push("        }".to_owned());
            if should_indent_value(inner, node_names) && at_line_start {
                lines.push(format!(
                    "        {writer_name}.with_indent(|{writer_name}| {{"
                ));
                lines.extend(
                    render_value_write_lines_in_graph(
                        repr,
                        writer_name,
                        graph_name,
                        "value",
                        inner,
                        None,
                        node_names,
                        false,
                    )?
                    .into_iter()
                    .map(|line| format!("    {line}")),
                );
                lines.push("        });".to_owned());
            } else {
                lines.extend(
                    render_value_write_lines_in_graph(
                        repr,
                        writer_name,
                        graph_name,
                        "value",
                        inner,
                        None,
                        node_names,
                        false,
                    )?
                    .into_iter()
                    .map(|line| format!("    {line}")),
                );
            }
            lines.push("    }".to_owned());
            Ok(lines)
        }
        SyntaxTypeUse::Arena {
            item: inner,
            key: Some(_key),
        } => {
            // Keyed arenas are stored in `Graph`; the field itself stores ids.
            let sep = joiner.unwrap_or("\n");
            let SyntaxTypeUse::Ref { name: item_name } = inner.as_ref() else {
                return Err(format!(
                    "keyed arena item must be a node ref, got {inner:?}"
                ));
            };
            let accessor = snake_case(item_name);
            let mut lines = vec![format!("    for (idx, id) in {expr}.iter().enumerate() {{")];
            lines.push("        if idx != 0 {".to_owned());
            lines.push(format!("            {writer_name}.text({sep:?});"));
            lines.push("        }".to_owned());
            lines.push(format!(
                "        let Some(value) = {graph_name}.{accessor}(*id) else {{ continue; }};"
            ));
            if should_indent_value(inner, node_names) && at_line_start {
                lines.push(format!(
                    "        {writer_name}.with_indent(|{writer_name}| {{"
                ));
                lines.extend(
                    render_value_write_lines_in_graph(
                        repr,
                        writer_name,
                        graph_name,
                        "value",
                        inner,
                        None,
                        node_names,
                        false,
                    )?
                    .into_iter()
                    .map(|line| format!("    {line}")),
                );
                lines.push("        });".to_owned());
            } else {
                lines.extend(
                    render_value_write_lines_in_graph(
                        repr,
                        writer_name,
                        graph_name,
                        "value",
                        inner,
                        None,
                        node_names,
                        false,
                    )?
                    .into_iter()
                    .map(|line| format!("    {line}")),
                );
            }
            lines.push("    }".to_owned());
            Ok(lines)
        }
        SyntaxTypeUse::Arena {
            item: inner,
            key: None,
        } => {
            let sep = joiner.unwrap_or("\n");
            let mut lines = vec![format!(
                "    for (idx, value) in {expr}.iter().enumerate() {{"
            )];
            lines.push("        if idx != 0 {".to_owned());
            lines.push(format!("            {writer_name}.text({sep:?});"));
            lines.push("        }".to_owned());
            lines.extend(
                render_value_write_lines_in_graph(
                    repr,
                    writer_name,
                    graph_name,
                    "value",
                    inner,
                    None,
                    node_names,
                    false,
                )?
                .into_iter()
                .map(|line| format!("    {line}")),
            );
            lines.push("    }".to_owned());
            Ok(lines)
        }
        SyntaxTypeUse::Pool { item: inner, .. } => {
            let sep = joiner.unwrap_or("\n");
            let mut lines = vec![format!(
                "    for (idx, value) in {expr}.iter().enumerate() {{"
            )];
            lines.push("        if idx != 0 {".to_owned());
            lines.push(format!("            {writer_name}.text({sep:?});"));
            lines.push("        }".to_owned());
            if should_indent_value(inner, node_names) && at_line_start {
                lines.push(format!(
                    "        {writer_name}.with_indent(|{writer_name}| {{"
                ));
                lines.extend(
                    render_value_write_lines_in_graph(
                        repr,
                        writer_name,
                        graph_name,
                        "value",
                        inner,
                        None,
                        node_names,
                        false,
                    )?
                    .into_iter()
                    .map(|line| format!("    {line}")),
                );
                lines.push("        });".to_owned());
            } else {
                lines.extend(
                    render_value_write_lines_in_graph(
                        repr,
                        writer_name,
                        graph_name,
                        "value",
                        inner,
                        None,
                        node_names,
                        false,
                    )?
                    .into_iter()
                    .map(|line| format!("    {line}")),
                );
            }
            lines.push("    }".to_owned());
            Ok(lines)
        }
        SyntaxTypeUse::RefTo { id, .. } => render_value_write_lines_in_graph(
            repr,
            writer_name,
            graph_name,
            expr,
            id,
            joiner,
            node_names,
            at_line_start,
        ),
        SyntaxTypeUse::Ref { name } if node_names.iter().any(|node| node == name) => {
            let write_name = format!("write_{}_in_graph", snake_case(name));
            if at_line_start {
                Ok(vec![format!(
                    "    {writer_name}.with_indent(|{writer_name}| {write_name}({writer_name}, {graph_name}, &{expr}));"
                )])
            } else {
                Ok(vec![format!(
                    "    {write_name}({writer_name}, {graph_name}, &{expr});"
                )])
            }
        }
        SyntaxTypeUse::Ref { name } => Ok(vec![match () {
            _ if is_string_scalar_type(repr, name) => {
                format!("    {writer_name}.text(&{expr}.text);")
            }
            _ if is_int_scalar_type(repr, name) => {
                format!("    {writer_name}.text(&{expr}.value.to_string());")
            }
            _ if is_id_type(repr, name) => {
                format!("    {writer_name}.text(&{expr}.0.to_string());")
            }
            _ => format!("    {writer_name}.text(&format!(\"{{:?}}\", {expr}));"),
        }]),
    }
}

fn should_indent_value(ty: &SyntaxTypeUse, node_names: &[String]) -> bool {
    match ty {
        SyntaxTypeUse::Ref { name } => node_names.iter().any(|node| node == name),
        SyntaxTypeUse::Seq(inner)
        | SyntaxTypeUse::Order(inner)
        | SyntaxTypeUse::Optional(inner) => should_indent_value(inner, node_names),
        SyntaxTypeUse::Arena { item: inner, .. } | SyntaxTypeUse::Pool { item: inner, .. } => {
            should_indent_value(inner, node_names)
        }
        SyntaxTypeUse::RefTo { id, .. } => should_indent_value(id, node_names),
    }
}

fn parse_template(template: &str) -> Result<Vec<TemplatePart>, String> {
    let mut parts = Vec::new();
    let mut cursor = 0usize;
    let bytes = template.as_bytes();

    while cursor < template.len() {
        if bytes[cursor] == b'{' {
            if let Some((end, inner)) = extract_braced(template, cursor)
                && let Ok(part) = parse_template_expr(inner)
            {
                parts.push(part);
                cursor = end + 1;
                continue;
            }
            parts.push(TemplatePart::Literal("{".to_owned()));
            cursor += 1;
        } else {
            let start = cursor;
            while cursor < template.len() && bytes[cursor] != b'{' {
                cursor += 1;
            }
            parts.push(TemplatePart::Literal(template[start..cursor].to_owned()));
        }
    }

    Ok(parts)
}

fn parse_template_expr(inner: &str) -> Result<TemplatePart, String> {
    if let Some((name, rest)) = inner.split_once("? : ") {
        let parts = parse_template(rest)?;
        return Ok(TemplatePart::Optional {
            name: parse_template_name(name)?,
            parts,
        });
    }

    if let Some((name, joiner)) = inner.split_once(':') {
        return Ok(TemplatePart::Field {
            name: parse_template_name(name)?,
            joiner: Some(joiner.to_owned()),
        });
    }

    Ok(TemplatePart::Field {
        name: parse_template_name(inner)?,
        joiner: None,
    })
}

fn parse_template_name(name: &str) -> Result<String, String> {
    let name = name.trim();
    if name.is_empty() {
        return Err("template field name cannot be empty".to_owned());
    }
    if name.contains('{') || name.contains('}') {
        return Err(format!("invalid template field name {name:?}"));
    }
    Ok(name.to_owned())
}

fn extract_braced(input: &str, start: usize) -> Option<(usize, &str)> {
    let mut depth = 0usize;
    for (offset, ch) in input[start..].char_indices() {
        match ch {
            '{' => depth += 1,
            '}' => {
                depth -= 1;
                if depth == 0 {
                    let end = start + offset;
                    return Some((end, &input[start + 1..end]));
                }
            }
            _ => {}
        }
    }
    None
}

fn collect_used_fields(parts: &[TemplatePart]) -> Vec<String> {
    let mut out = Vec::new();
    collect_used_fields_into(parts, &mut out);
    out.sort();
    out.dedup();
    out
}

fn collect_used_fields_into(parts: &[TemplatePart], out: &mut Vec<String>) {
    for part in parts {
        match part {
            TemplatePart::Literal(_) => {}
            TemplatePart::Field { name, .. } | TemplatePart::Optional { name, .. } => {
                out.push(name.clone());
            }
        }
        if let TemplatePart::Optional { parts, .. } = part {
            collect_used_fields_into(parts, out);
        }
    }
}
