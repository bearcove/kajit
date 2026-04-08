use crate::schema::{NodeDecl, NodeFields, ReprBody, TypeUse, type_use_tag};

pub(crate) fn rust_ident(name: &str) -> String {
    match name {
        "else" | "type" | "struct" | "enum" | "fn" | "mod" | "move" | "ref" | "self" | "Self"
        | "crate" | "super" | "use" | "where" | "loop" | "match" | "return" | "pub" | "in"
        | "let" | "impl" | "trait" | "const" | "static" | "async" | "await" | "dyn" => {
            format!("r#{name}")
        }
        _ => name.to_owned(),
    }
}

pub(crate) fn pascal_case(name: &str) -> String {
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
    if out.is_empty() {
        "Unnamed".to_owned()
    } else {
        out
    }
}

pub(crate) fn snake_case(name: &str) -> String {
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

pub(crate) fn collect_type_tags(ty: &TypeUse, out: &mut Vec<String>) {
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

pub(crate) fn render_type_use(ty: &TypeUse, node_names: &[String], box_node_refs: bool) -> String {
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

pub(crate) fn node_fields_have_prov(fields: &NodeFields, provenance_tag: &str) -> bool {
    fields
        .fields
        .get("prov")
        .is_some_and(|ty| type_use_tag(ty) == Some(provenance_tag))
}

pub(crate) fn render_common_placeholder(
    tag: &str,
    common_names: &[String],
    repr: &ReprBody,
) -> String {
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
            format!(
                "#[derive(Debug, Clone, PartialEq, Eq, Default)]\npub struct {tag}(pub String);"
            )
        }
        _ => format!("#[derive(Debug, Clone, PartialEq, Eq, Default)]\npub struct {tag};"),
    }
}

pub(crate) fn render_visit_calls(
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
                    inner
                        .into_iter()
                        .map(|line| format!("    {line}"))
                        .collect::<Vec<_>>()
                        .join("\n")
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
                    inner
                        .into_iter()
                        .map(|line| format!("    {line}"))
                        .collect::<Vec<_>>()
                        .join("\n")
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

pub(crate) fn render_walk_fn(
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
            let body = body_lines
                .join("\n")
                .replace("node.", &format!("{node_name}."));
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

pub(crate) fn render_node_decl(
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
                    let ty =
                        render_type_use(fields.fields.get(field_name).unwrap(), node_names, true);
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
                .map(
                    |variant_name| match variants.variants.get(variant_name).unwrap() {
                        NodeDecl::Node(fields) | NodeDecl::Struct(fields) => {
                            let mut field_names = fields.fields.keys().cloned().collect::<Vec<_>>();
                            field_names.sort();
                            let rows = field_names
                                .iter()
                                .map(|field_name| {
                                    let ty = render_type_use(
                                        fields.fields.get(field_name).unwrap(),
                                        node_names,
                                        true,
                                    );
                                    format!("        {}: {},", rust_ident(field_name), ty)
                                })
                                .collect::<Vec<_>>()
                                .join("\n");
                            format!("    {variant_name} {{\n{rows}\n    }},")
                        }
                        other => {
                            format!("    {variant_name}, // unsupported variant shape: {other:?}")
                        }
                    },
                )
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
