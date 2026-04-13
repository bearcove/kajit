use std::collections::HashMap;

use crate::normalize::{
    DocumentedValue, NormalizedNodeDecl, NormalizedNodeKind, NormalizedSupportDecl, SyntaxTypeUse,
};

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
    let chars = name.chars().collect::<Vec<_>>();
    for (i, ch) in chars.iter().copied().enumerate() {
        let prev = i.checked_sub(1).and_then(|idx| chars.get(idx)).copied();
        let next = chars.get(i + 1).copied();
        if ch.is_ascii_uppercase() {
            let word_boundary = matches!(prev, Some(p) if p.is_ascii_lowercase() || p.is_ascii_digit())
                || matches!((prev, next), (Some(p), Some(n)) if p.is_ascii_uppercase() && n.is_ascii_lowercase());
            if i != 0 && word_boundary {
                out.push('_');
            }
            out.push(ch.to_ascii_lowercase());
        } else {
            out.push(ch);
        }
    }
    out
}

pub(crate) fn collect_syntax_type_tags(ty: &SyntaxTypeUse, out: &mut Vec<String>) {
    match ty {
        SyntaxTypeUse::Ref { name } => out.push(name.clone()),
        SyntaxTypeUse::Optional(inner)
        | SyntaxTypeUse::Seq(inner)
        | SyntaxTypeUse::Order(inner) => {
            collect_syntax_type_tags(inner, out);
        }
        SyntaxTypeUse::Arena { item: inner, key } => {
            collect_syntax_type_tags(inner, out);
            if let Some(key) = key {
                out.push(key.clone());
            }
        }
        SyntaxTypeUse::Pool { item: inner, key } => {
            collect_syntax_type_tags(inner, out);
            if let Some(key) = key {
                out.push(key.clone());
            }
        }
        SyntaxTypeUse::RefTo { id, target } => {
            collect_syntax_type_tags(id, out);
            out.push(target.clone());
        }
    }
}

pub(crate) fn render_syntax_type_use(
    ty: &SyntaxTypeUse,
    node_names: &[String],
    box_node_refs: bool,
) -> String {
    match ty {
        SyntaxTypeUse::Optional(inner) => {
            format!(
                "Option<{}>",
                render_syntax_type_use(inner, node_names, true)
            )
        }
        SyntaxTypeUse::Seq(inner) => {
            format!("Vec<{}>", render_syntax_type_use(inner, node_names, false))
        }
        SyntaxTypeUse::Arena { key: Some(key), .. } => format!("super::super::Order<{key}>"),
        SyntaxTypeUse::Arena {
            item: inner,
            key: None,
        } => {
            format!(
                "super::super::Arena<{}>",
                render_syntax_type_use(inner, node_names, false)
            )
        }
        SyntaxTypeUse::Pool { item: inner, .. } => {
            format!(
                "super::super::Pool<{}>",
                render_syntax_type_use(inner, node_names, false)
            )
        }
        SyntaxTypeUse::Order(inner) => {
            format!(
                "super::super::Order<{}>",
                render_syntax_type_use(inner, node_names, false)
            )
        }
        SyntaxTypeUse::RefTo { id, .. } => render_syntax_type_use(id, node_names, box_node_refs),
        SyntaxTypeUse::Ref { name } => {
            if box_node_refs && node_names.iter().any(|node| node == name) {
                format!("Box<{name}>")
            } else {
                name.clone()
            }
        }
    }
}

pub(crate) fn node_fields_have_prov(
    fields: &HashMap<String, DocumentedValue<SyntaxTypeUse>>,
    provenance_tag: &str,
) -> bool {
    fields.get("prov").is_some_and(
        |ty| matches!(&ty.value, SyntaxTypeUse::Ref { name } if name == provenance_tag),
    )
}

pub(crate) fn is_prov_only_struct(
    fields: &HashMap<String, DocumentedValue<SyntaxTypeUse>>,
    provenance_tag: &str,
) -> bool {
    fields.len() == 1 && node_fields_have_prov(fields, provenance_tag)
}

pub(crate) fn leaf_variant_wrapper_name(_enum_name: &str, variant_name: &str) -> String {
    variant_name.to_owned()
}

pub(crate) fn render_common_placeholder(
    tag: &str,
    common_names: &[String],
    common: &HashMap<String, SyntaxTypeUse>,
) -> String {
    let common_name = common_names
        .iter()
        .find(|name| matches!(common.get(*name), Some(SyntaxTypeUse::Ref { name }) if name == tag));

    match tag {
        "Prov" => {
            let alias = common_name
                .filter(|name| name.as_str() != "provenance")
                .map(|name| format!("pub type {} = Prov;\n", pascal_case(name)))
                .unwrap_or_default();
            format!("pub use kajit_types::{{Prov, Span}};\n{alias}")
        }
        "Symbol" => {
            let alias = common_name
                .filter(|name| name.as_str() != "symbol")
                .map(|name| format!("pub type {} = Symbol;\n", pascal_case(name)))
                .unwrap_or_default();
            format!(
                "#[derive(Debug, Clone, PartialEq, Eq, Default)]\npub struct Symbol {{\n    pub prov: Prov,\n    pub text: String,\n}}\n\nimpl Symbol {{\n    pub fn as_str(&self) -> &str {{\n        &self.text\n    }}\n}}\n{alias}"
            )
        }
        "DocBlock" => {
            let alias = common_name
                .filter(|name| name.as_str() != "docs")
                .map(|name| format!("pub type {} = DocBlock;\n", pascal_case(name)))
                .unwrap_or_default();
            format!(
                "/// Preserved doc-comment lines collected from leading `///` comments.\n#[derive(Debug, Clone, PartialEq, Eq, Default)]\npub struct DocBlock(pub Vec<String>);\n{alias}"
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

pub(crate) fn render_type_use_kind(ty: &SyntaxTypeUse) -> String {
    match ty {
        SyntaxTypeUse::Optional(inner) => format!("optional<{}>", render_type_use_kind(inner)),
        SyntaxTypeUse::Seq(inner) => format!("seq<{}>", render_type_use_kind(inner)),
        SyntaxTypeUse::Arena { item: inner, key } => match key {
            Some(key) => format!("arena<{} key={key}>", render_type_use_kind(inner)),
            None => format!("arena<{}>", render_type_use_kind(inner)),
        },
        SyntaxTypeUse::Pool { item: inner, key } => match key {
            Some(key) => format!("pool<{} key={key}>", render_type_use_kind(inner)),
            None => format!("pool<{}>", render_type_use_kind(inner)),
        },
        SyntaxTypeUse::Order(inner) => format!("order<{}>", render_type_use_kind(inner)),
        SyntaxTypeUse::RefTo { id, target } => {
            format!("ref<{} -> {target}>", render_type_use_kind(id))
        }
        SyntaxTypeUse::Ref { name } => name.clone(),
    }
}

pub(crate) fn render_support_decl(
    name: &str,
    decl: &NormalizedSupportDecl,
    doc: Option<&[String]>,
    node_names: &[String],
) -> String {
    let docs = render_doc_lines(doc, "");
    let body = match decl {
        NormalizedSupportDecl::String => format!(
            "#[derive(Debug, Clone, PartialEq, Eq, Default)]\npub struct {name} {{\n    pub prov: Prov,\n    pub text: String,\n}}\n\nimpl {name} {{\n    pub fn as_str(&self) -> &str {{\n        &self.text\n    }}\n}}"
        ),
        NormalizedSupportDecl::Int => format!(
            "#[derive(Debug, Clone, PartialEq, Eq, Default)]\npub struct {name} {{\n    pub prov: Prov,\n    pub value: u64,\n}}"
        ),
        NormalizedSupportDecl::Id => format!(
            "#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]\npub struct {name}(pub u32);\n\nimpl {name} {{\n    pub const fn new(index: u32) -> Self {{\n        Self(index)\n    }}\n\n    pub const fn index(self) -> usize {{\n        self.0 as usize\n    }}\n}}"
        ),
        NormalizedSupportDecl::StringSeq => format!(
            "#[derive(Debug, Clone, PartialEq, Eq, Default)]\npub struct {name}(pub Vec<String>);"
        ),
        NormalizedSupportDecl::Unit => {
            format!("#[derive(Debug, Clone, PartialEq, Eq, Default)]\npub struct {name};")
        }
        NormalizedSupportDecl::Struct(fields) => {
            let mut field_names = fields.keys().cloned().collect::<Vec<_>>();
            field_names.sort();
            let field_rows = field_names
                .iter()
                .map(|field_name| {
                    let field = fields.get(field_name).unwrap();
                    let ty = render_syntax_type_use(&field.value, node_names, true);
                    let docs = render_doc_lines(field.doc.as_deref(), "    ");
                    if docs.is_empty() {
                        format!("    pub {}: {},", rust_ident(field_name), ty)
                    } else {
                        format!("{docs}\n    pub {}: {},", rust_ident(field_name), ty)
                    }
                })
                .collect::<Vec<_>>()
                .join("\n");
            format!(
                "#[derive(Debug, Clone, PartialEq, Eq)]\npub struct {name} {{\n{field_rows}\n}}"
            )
        }
        NormalizedSupportDecl::Enum(variants) => {
            let rows = variants
                .iter()
                .enumerate()
                .map(|(idx, variant)| {
                    let variant_docs = render_doc_lines(variant.doc.as_deref(), "    ");
                    if idx == 0 {
                        if variant_docs.is_empty() {
                            format!("    #[default]\n    {},", variant.value)
                        } else {
                            format!("{variant_docs}\n    #[default]\n    {},", variant.value)
                        }
                    } else {
                        if variant_docs.is_empty() {
                            format!("    {},", variant.value)
                        } else {
                            format!("{variant_docs}\n    {},", variant.value)
                        }
                    }
                })
                .collect::<Vec<_>>()
                .join("\n\n");
            format!(
                "#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]\npub enum {name} {{\n\n{rows}\n}}"
            )
        }
    };
    if docs.is_empty() {
        body
    } else {
        format!("{docs}\n{body}")
    }
}

fn render_doc_lines(doc: Option<&[String]>, indent: &str) -> String {
    let Some(lines) = doc else {
        return String::new();
    };
    if lines.is_empty() {
        return String::new();
    }
    lines
        .iter()
        .map(|line| format!("{indent}/// {line}"))
        .collect::<Vec<_>>()
        .join("\n")
}

pub(crate) fn render_visit_calls(
    ty: &SyntaxTypeUse,
    expr: &str,
    node_names: &[String],
    mutable: bool,
    borrowed: bool,
) -> Vec<String> {
    match ty {
        SyntaxTypeUse::Optional(inner) => {
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
            let inner = render_visit_calls(inner, "value", node_names, mutable, true);
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
        SyntaxTypeUse::Seq(inner) | SyntaxTypeUse::Order(inner) => {
            let inner = render_visit_calls(inner, "value", node_names, mutable, true);
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
        SyntaxTypeUse::Arena { key: Some(_), .. } => Vec::new(),
        SyntaxTypeUse::Arena {
            item: inner,
            key: None,
        } => {
            let inner = render_visit_calls(inner, "value", node_names, mutable, true);
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
        SyntaxTypeUse::Pool { item: inner, .. } => {
            let inner = render_visit_calls(inner, "value", node_names, mutable, true);
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
        SyntaxTypeUse::RefTo { id, .. } => {
            render_visit_calls(id, expr, node_names, mutable, borrowed)
        }
        SyntaxTypeUse::Ref { name } if node_names.iter().any(|node| node == name) => {
            let method = snake_case(name);
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
    decl: &NormalizedNodeDecl,
    node_names: &[String],
    mutable: bool,
    provenance_tag: &str,
) -> String {
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
        NormalizedNodeDecl::Record { fields, .. } => {
            let mut field_names = fields.keys().cloned().collect::<Vec<_>>();
            field_names.sort();
            let body_lines = field_names
                .iter()
                .flat_map(|field_name| {
                    let field_expr = format!("node.{}", rust_ident(field_name));
                    render_visit_calls(
                        &fields.get(field_name).unwrap().value,
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
            format!(
                "pub fn {walk_name}<V: ?Sized + {trait_name}>({v_name}: &mut V, {node_name}: {node_ty}) {{\n{body}\n}}"
            )
        }
        NormalizedNodeDecl::Enum(variants) => {
            let variant_names = variants.keys().cloned().collect::<Vec<_>>();
            let arms = variant_names
                .iter()
                .filter_map(|variant_name| match &variants.get(variant_name).unwrap().value {
                    NormalizedNodeDecl::Record { fields, .. } => {
                        if is_prov_only_struct(fields, provenance_tag) {
                            return Some(format!(
                                "        {name}::{variant_name}(..) => {{}}"
                            ));
                        }
                        let mut field_names = fields.keys().cloned().collect::<Vec<_>>();
                        field_names.sort();
                        let traversed = field_names
                            .iter()
                            .filter_map(|field_name| {
                                let expr = rust_ident(field_name);
                                let calls = render_visit_calls(
                                    &fields.get(field_name).unwrap().value,
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
                    NormalizedNodeDecl::Enum(_) => None,
                })
                .collect::<Vec<_>>()
                .join(",\n");
            format!(
                "pub fn {walk_name}<V: ?Sized + {trait_name}>(v: &mut V, node: {node_ty}) {{\n    match node {{\n{arms}\n    }}\n}}"
            )
        }
    }
}

pub(crate) fn render_node_decl(
    name: &str,
    decl: &NormalizedNodeDecl,
    node_names: &[String],
    doc: Option<&[String]>,
    provenance_tag: &str,
) -> String {
    match decl {
        NormalizedNodeDecl::Record { kind, fields } => {
            let mut field_names = fields.keys().cloned().collect::<Vec<_>>();
            field_names.sort();
            let field_rows = field_names
                .iter()
                .map(|field_name| {
                    let field = fields.get(field_name).unwrap();
                    let ty = render_syntax_type_use(&field.value, node_names, true);
                    let docs = render_doc_lines(field.doc.as_deref(), "    ");
                    if docs.is_empty() {
                        format!("    pub {}: {},", rust_ident(field_name), ty)
                    } else {
                        format!("{docs}\n    pub {}: {},", rust_ident(field_name), ty)
                    }
                })
                .collect::<Vec<_>>()
                .join("\n");
            let docs = render_doc_lines(doc, "");
            let marker_impl = match kind {
                NormalizedNodeKind::Struct => String::new(),
            };
            let body = format!(
                "#[derive(Debug, Clone, PartialEq, Eq)]\npub struct {name} {{\n{field_rows}\n}}{marker_impl}"
            );
            if docs.is_empty() {
                body
            } else {
                format!("{docs}\n{body}")
            }
        }
        NormalizedNodeDecl::Enum(variants) => {
            let variant_names = variants.keys().cloned().collect::<Vec<_>>();
            let leaf_struct_rows = variant_names
                .iter()
                .filter_map(|variant_name| match &variants.get(variant_name).unwrap().value {
                    NormalizedNodeDecl::Record { fields, .. }
                        if is_prov_only_struct(fields, provenance_tag) =>
                    {
                        let wrapper_name = leaf_variant_wrapper_name(name, variant_name);
                        let variant_docs = render_doc_lines(
                            variants.get(variant_name).unwrap().doc.as_deref(),
                            "",
                        );
                        let body = format!(
                            "#[derive(Debug, Clone, PartialEq, Eq)]\npub struct {wrapper_name} {{\n    pub prov: {provenance_tag},\n}}"
                        );
                        Some(if variant_docs.is_empty() {
                            body
                        } else {
                            format!("{variant_docs}\n{body}")
                        })
                    }
                    _ => None,
                })
                .collect::<Vec<_>>()
                .join("\n\n");
            let variant_rows = variant_names
                .iter()
                .map(
                    |variant_name| match &variants.get(variant_name).unwrap().value {
                        NormalizedNodeDecl::Record { fields, .. }
                            if is_prov_only_struct(fields, provenance_tag) =>
                        {
                            let variant_docs = render_doc_lines(
                                variants.get(variant_name).unwrap().doc.as_deref(),
                                "    ",
                            );
                            let wrapper_name = leaf_variant_wrapper_name(name, variant_name);
                            if variant_docs.is_empty() {
                                format!("    {variant_name}({wrapper_name}),")
                            } else {
                                format!("{variant_docs}\n    {variant_name}({wrapper_name}),")
                            }
                        }
                        NormalizedNodeDecl::Record { fields, .. } => {
                            let mut field_names = fields.keys().cloned().collect::<Vec<_>>();
                            field_names.sort();
                            let rows = field_names
                                .iter()
                                .map(|field_name| {
                                    let field = fields.get(field_name).unwrap();
                                    let ty = render_syntax_type_use(&field.value, node_names, true);
                                    let docs = render_doc_lines(field.doc.as_deref(), "        ");
                                    if docs.is_empty() {
                                        format!("        {}: {},", rust_ident(field_name), ty)
                                    } else {
                                        format!(
                                            "{docs}\n        {}: {},",
                                            rust_ident(field_name),
                                            ty
                                        )
                                    }
                                })
                                .collect::<Vec<_>>()
                                .join("\n\n");
                            let variant_docs = render_doc_lines(
                                variants.get(variant_name).unwrap().doc.as_deref(),
                                "    ",
                            );
                            if variant_docs.is_empty() {
                                format!("    {variant_name} {{\n{rows}\n    }},")
                            } else {
                                format!("{variant_docs}\n    {variant_name} {{\n{rows}\n    }},")
                            }
                        }
                        other => {
                            format!("    {variant_name}, // unsupported variant shape: {other:?}")
                        }
                    },
                )
                .collect::<Vec<_>>()
                .join("\n\n");
            let docs = render_doc_lines(doc, "");
            let body = format!(
                "#[derive(Debug, Clone, PartialEq, Eq)]\npub enum {name} {{\n\n{variant_rows}\n}}"
            );
            let combined = if leaf_struct_rows.is_empty() {
                body
            } else {
                format!("{leaf_struct_rows}\n\n{body}")
            };
            if docs.is_empty() {
                combined
            } else {
                format!("{docs}\n{combined}")
            }
        }
    }
}

pub(crate) fn render_provenance_impl(
    name: &str,
    decl: &NormalizedNodeDecl,
    provenance_tag: &str,
) -> Option<String> {
    match decl {
        NormalizedNodeDecl::Record { fields, .. }
            if node_fields_have_prov(fields, provenance_tag) =>
        {
            Some(format!(
                "impl HasProvenance for {name} {{\n    fn provenance(&self) -> Option<&{provenance_tag}> {{\n        Some(&self.prov)\n    }}\n}}"
            ))
        }
        NormalizedNodeDecl::Enum(variants)
            if variants.values().all(|variant| {
                matches!(
                    &variant.value,
                    NormalizedNodeDecl::Record { fields, .. }
                        if node_fields_have_prov(fields, provenance_tag)
                )
            }) =>
        {
            let variant_names = variants.keys().cloned().collect::<Vec<_>>();
            let match_rows = variant_names
                .iter()
                .map(
                    |variant_name| match &variants.get(variant_name).unwrap().value {
                        NormalizedNodeDecl::Record { fields, .. }
                            if is_prov_only_struct(fields, provenance_tag) =>
                        {
                            format!("            Self::{variant_name}(value) => Some(&value.prov),")
                        }
                        _ => {
                            format!(
                                "            Self::{variant_name} {{ prov, .. }} => Some(prov),"
                            )
                        }
                    },
                )
                .collect::<Vec<_>>()
                .join("\n");
            Some(format!(
                "impl HasProvenance for {name} {{\n    fn provenance(&self) -> Option<&{provenance_tag}> {{\n        match self {{\n{match_rows}\n        }}\n    }}\n}}"
            ))
        }
        _ => None,
    }
}
