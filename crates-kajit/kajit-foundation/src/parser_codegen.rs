use std::collections::{BTreeSet, HashMap, HashSet};

use crate::normalize::{
    DocumentedValue, NormalizedNodeDecl, NormalizedRepr, NormalizedSupportDecl,
    NormalizedTokenSpec, SyntaxRule, SyntaxTypeUse, direct_ref_name, is_docs_type, is_id_type,
    is_int_scalar_type, is_string_scalar_type, render_default_value,
};
use crate::render_helpers::{
    is_prov_only_struct, leaf_variant_wrapper_name, rust_ident, snake_case,
};

enum SeqItem {
    Ignore(String),
    Bind { name: String, parser: String },
}

enum RuleTargetDecl<'a> {
    Node(&'a NormalizedNodeDecl),
    Support(&'a NormalizedSupportDecl),
}

fn token_parser_fn_name(token_name: &str) -> String {
    format!("token_{}", snake_case(token_name))
}

fn render_wrapped_scalar_parser(
    repr: &NormalizedRepr,
    ty: &SyntaxTypeUse,
    parser_fn_name: &str,
) -> Result<String, String> {
    if let SyntaxTypeUse::RefTo { id, .. } = ty {
        return render_wrapped_scalar_parser(repr, id, parser_fn_name);
    }
    match ty {
        SyntaxTypeUse::Ref { name } if is_string_scalar_type(repr, name) => Ok(format!(
            "{parser_fn_name}().map_with(move |text, e| {name} {{ prov: prov_from_span(e.span(), file_id), text }}).then_ignore(ws())"
        )),
        SyntaxTypeUse::Ref { name } if is_int_scalar_type(repr, name) => Ok(format!(
            "{parser_fn_name}().try_map(move |text, span| {{\n            match text.parse::<u64>() {{\n                Ok(value) => Ok({name} {{ prov: prov_from_span(span, file_id), value }}),\n                Err(err) => Err(Rich::custom(span, format!(\"invalid integer literal {{text:?}}: {{err}}\"))),\n            }}\n        }}).then_ignore(ws())"
        )),
        SyntaxTypeUse::Ref { name } if is_id_type(repr, name) => Ok(format!(
            "{parser_fn_name}().try_map(move |text, span| {{\n            match text.parse::<u32>() {{\n                Ok(value) => Ok({name}(value)),\n                Err(err) => Err(Rich::custom(span, format!(\"invalid id literal {{text:?}}: {{err}}\"))),\n            }}\n        }}).then_ignore(ws())"
        )),
        SyntaxTypeUse::Ref { name } => Err(format!(
            "token parser cannot construct non-scalar support type {name:?}"
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
    render_wrapped_scalar_parser(repr, ty, &parser_fn_name)
}

fn render_ref_value_parser(
    repr: &NormalizedRepr,
    ref_name: &str,
    ty: &SyntaxTypeUse,
    rule_names: &HashSet<String>,
    node_names: &[String],
    box_node_refs: bool,
) -> Result<String, String> {
    if let SyntaxTypeUse::RefTo { id, .. } = ty {
        return render_ref_value_parser(repr, ref_name, id, rule_names, node_names, box_node_refs);
    }
    if !rule_names.contains(ref_name) {
        return Err(format!("unsupported reference parser target {ref_name:?}"));
    }

    let base = format!("{}_parser.clone()", snake_case(ref_name));

    Ok(match ty {
        SyntaxTypeUse::Ref { name }
            if box_node_refs && node_names.iter().any(|node| node == name) =>
        {
            format!("({base}).map(Box::new)")
        }
        SyntaxTypeUse::Ref { name }
            if matches!(
                repr.support.get(name).map(|decl| &decl.value),
                Some(NormalizedSupportDecl::Struct(_))
            ) =>
        {
            base
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
    if let SyntaxTypeUse::Optional(inner_ty) = ty
        && !matches!(rule, SyntaxRule::Optional { .. })
    {
        return render_value_parser(repr, rule, inner_ty, rule_names, node_names, box_node_refs);
    }

    if let SyntaxTypeUse::RefTo { id, .. } = ty {
        return render_value_parser(repr, rule, id, rule_names, node_names, box_node_refs);
    }

    match rule {
        SyntaxRule::Token { name } => render_token_value_parser(repr, name, ty),
        SyntaxRule::Ref { name } => {
            render_ref_value_parser(repr, name, ty, rule_names, node_names, box_node_refs)
        }
        SyntaxRule::Optional { inner } => {
            let SyntaxTypeUse::Optional(inner_ty) = ty else {
                return Err("optional rule without optional type".to_owned());
            };
            let inner = render_value_parser(repr, inner, inner_ty, rule_names, node_names, true)?;
            Ok(format!("({inner}).or_not()"))
        }
        SyntaxRule::Repeat { item, sep } => {
            let inner_ty = match ty {
                SyntaxTypeUse::Seq(inner_ty)
                | SyntaxTypeUse::Order(inner_ty)
                | SyntaxTypeUse::Arena { item: inner_ty, .. }
                | SyntaxTypeUse::Pool { item: inner_ty, .. } => inner_ty,
                _ => return Err("repeat rule without repeated type".to_owned()),
            };
            let inner = render_value_parser(repr, item, inner_ty, rule_names, node_names, false)?;
            let collected = if let Some(sep) = sep.as_deref() {
                format!(
                    "({inner}).separated_by(just({sep:?}).padded()).allow_trailing().collect::<Vec<_>>()"
                )
            } else {
                format!("({inner}).repeated().collect::<Vec<_>>()")
            };
            Ok(match ty {
                SyntaxTypeUse::Seq(_) => collected,
                SyntaxTypeUse::Arena { .. } => {
                    format!("({collected}).map(super::super::Arena::from)")
                }
                SyntaxTypeUse::Pool { .. } => {
                    format!("({collected}).map(super::super::Pool::from)")
                }
                SyntaxTypeUse::Order(_) => {
                    format!("({collected}).map(super::super::Order::from)")
                }
                _ => unreachable!("repeat rule without repeated type"),
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
        SyntaxRule::Optional { inner } => {
            let nested = flatten_struct_rule_items(repr, inner, fields, rule_names, node_names)?;
            let (chain, names) = render_binding_chain(&nested)?;
            if names.len() != 1 {
                return Err(format!(
                    "optional struct item must bind exactly one field, got {:?}",
                    names
                ));
            }
            Ok(vec![SeqItem::Bind {
                name: names[0].clone(),
                parser: format!("({chain}).or_not()"),
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

fn render_scalar_support_rule_expr(
    repr: &NormalizedRepr,
    support_name: &str,
    decl: &NormalizedSupportDecl,
    rule: &SyntaxRule,
    rule_names: &HashSet<String>,
    node_names: &[String],
) -> Result<String, String> {
    let support_ty = SyntaxTypeUse::Ref {
        name: support_name.to_owned(),
    };

    match rule {
        SyntaxRule::Token { .. }
        | SyntaxRule::Ref { .. }
        | SyntaxRule::Optional { .. }
        | SyntaxRule::Repeat { .. } => {
            return render_value_parser(repr, rule, &support_ty, rule_names, node_names, false);
        }
        SyntaxRule::Seq(items) => {
            let expected_field = match decl {
                NormalizedSupportDecl::String => "text",
                NormalizedSupportDecl::Int | NormalizedSupportDecl::Id => "value",
                _ => unreachable!(),
            };
            let mut items_out = Vec::new();
            let mut bound = None;
            for item in items {
                match item {
                    SyntaxRule::Literal(text) => {
                        items_out.push(SeqItem::Ignore(format!("just({text:?}).padded()")));
                    }
                    SyntaxRule::Field(named) if named.name == expected_field => {
                        let parser = match named.inner.as_ref() {
                            SyntaxRule::Token { name } => {
                                format!("{}().then_ignore(ws())", token_parser_fn_name(name))
                            }
                            SyntaxRule::Ref { name } => {
                                if !rule_names.contains(name) {
                                    return Err(format!(
                                        "scalar support rule for {support_name} references unknown parser target {name:?}"
                                    ));
                                }
                                format!("{}_parser.clone()", snake_case(name))
                            }
                            other => {
                                return Err(format!(
                                    "scalar support rule for {support_name} field {expected_field:?} must come from @token(...) or @ref(...), got {other:?}"
                                ));
                            }
                        };
                        items_out.push(SeqItem::Bind {
                            name: expected_field.to_owned(),
                            parser,
                        });
                        bound = Some(expected_field.to_owned());
                    }
                    other => {
                        return Err(format!(
                            "scalar support rule for {support_name} has unsupported item {other:?}"
                        ));
                    }
                }
            }
            let Some(bound_name) = bound else {
                return Err(format!(
                    "scalar support rule for {support_name} must bind field {expected_field:?}"
                ));
            };
            let (chain, _bound_names) = render_binding_chain(&items_out)?;
            let body = match decl {
                NormalizedSupportDecl::String => format!(
                    "({chain}).map_with(move |{bound_name}, e| {support_name} {{ prov: prov_from_span(e.span(), file_id), text: {bound_name} }})"
                ),
                NormalizedSupportDecl::Int => format!(
                    "({chain}).try_map(move |{bound_name}, span| match {bound_name}.parse::<u64>() {{ Ok(value) => Ok({support_name} {{ prov: prov_from_span(span, file_id), value }}), Err(err) => Err(Rich::custom(span, format!(\"invalid integer literal {{{bound_name}:?}}: {{err}}\"))) }})"
                ),
                NormalizedSupportDecl::Id => format!(
                    "({chain}).try_map(move |{bound_name}, span| match {bound_name}.parse::<u32>() {{ Ok(value) => Ok({support_name}(value)), Err(err) => Err(Rich::custom(span, format!(\"invalid id literal {{{bound_name}:?}}: {{err}}\"))) }})"
                ),
                _ => unreachable!(),
            };
            Ok(format!("{body}.boxed()"))
        }
        other => Err(format!(
            "scalar support rule for {support_name} has unsupported shape {other:?}"
        )),
    }
}

fn render_implicit_docs_parser(
    repr: &NormalizedRepr,
    field_name: &str,
    ty: &SyntaxTypeUse,
) -> Option<String> {
    if field_name != "docs" {
        return None;
    }

    match ty {
        SyntaxTypeUse::Optional(inner) => match inner.as_ref() {
            ty if direct_ref_name(ty).is_some_and(|name| is_docs_type(repr, name)) => {
                Some("doc_block()".to_owned())
            }
            _ => None,
        },
        ty if direct_ref_name(ty).is_some_and(|name| is_docs_type(repr, name)) => {
            Some("doc_block().map(|docs| docs.unwrap_or_default())".to_owned())
        }
        _ => None,
    }
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
    if is_prov_only_struct(fields, provenance_tag) {
        if let Some(text) = extract_single_literal(rule) {
            return Ok(format!(
                "(just({text:?}).to(())).map_with(move |(), e| {type_name} {{ prov: prov_from_span(e.span(), file_id) }}).then_ignore(ws()).boxed()"
            ));
        }
    }

    let mut items = flatten_struct_rule_items(repr, rule, fields, rule_names, node_names)?;
    if fields.contains_key("docs")
        && !items
            .iter()
            .any(|item| matches!(item, SeqItem::Bind { name, .. } if name == "docs"))
    {
        if let Some(parser) = render_implicit_docs_parser(
            repr,
            "docs",
            &fields.get("docs").expect("docs field should exist").value,
        ) {
            items.insert(
                0,
                SeqItem::Bind {
                    name: "docs".to_owned(),
                    parser,
                },
            );
        }
    }
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
        let NormalizedNodeDecl::Record { fields, .. } = &variant_decl.value else {
            return Err(format!(
                "schema enum {enum_name} variant {variant_name:?} has unsupported declaration"
            ));
        };
        let mut items =
            flatten_struct_rule_items(repr, named.inner.as_ref(), fields, rule_names, node_names)?;
        if fields.contains_key("docs")
            && !items
                .iter()
                .any(|item| matches!(item, SeqItem::Bind { name, .. } if name == "docs"))
        {
            if let Some(parser) = render_implicit_docs_parser(
                repr,
                "docs",
                &fields.get("docs").expect("docs field should exist").value,
            ) {
                items.insert(
                    0,
                    SeqItem::Bind {
                        name: "docs".to_owned(),
                        parser,
                    },
                );
            }
        }
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
        let parser = if is_prov_only_struct(fields, provenance_tag) {
            if let Some(text) = extract_single_literal(named.inner.as_ref()) {
                let wrapper_name = leaf_variant_wrapper_name(enum_name, variant_name);
                format!(
                    "(just({text:?}).to(())).map_with(move |(), {span_ident}| {enum_name}::{variant_name}({wrapper_name} {{ {field_rows} }})).then_ignore(ws())"
                )
            } else {
                let wrapper_name = leaf_variant_wrapper_name(enum_name, variant_name);
                format!(
                    "({chain}).map_with(move |(), {span_ident}| {enum_name}::{variant_name}({wrapper_name} {{ {field_rows} }}))"
                )
            }
        } else if bound_names.is_empty() {
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

    if variant_parsers.len() == 1 {
        return Ok(format!("({}).boxed()", variant_parsers[0]));
    }

    Ok(format!("choice(({})).boxed()", variant_parsers.join(", ")))
}

fn render_rule_parser_expr(
    repr: &NormalizedRepr,
    rule_name: &str,
    rule: &SyntaxRule,
    decl: RuleTargetDecl<'_>,
    rule_names: &HashSet<String>,
    node_names: &[String],
    provenance_tag: &str,
) -> Result<String, String> {
    match decl {
        RuleTargetDecl::Node(NormalizedNodeDecl::Record { fields, .. }) => {
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
        RuleTargetDecl::Node(NormalizedNodeDecl::Enum(variants)) => render_enum_parser_expr(
            repr,
            rule_name,
            variants,
            rule,
            rule_names,
            node_names,
            provenance_tag,
        ),
        RuleTargetDecl::Support(NormalizedSupportDecl::Struct(fields)) => {
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
        RuleTargetDecl::Support(NormalizedSupportDecl::Enum(variants)) => {
            let SyntaxRule::Choice(items) = rule else {
                return Err(format!("enum rule for {rule_name} must be choice"));
            };
            let mut parsers = Vec::new();
            for item in items {
                let SyntaxRule::Variant(named) = item else {
                    return Err(format!(
                        "enum rule for {rule_name} contains non-variant item"
                    ));
                };
                let variant_name = named.name.as_str();
                if !variants.iter().any(|variant| variant.value == variant_name) {
                    return Err(format!(
                        "support enum {rule_name} is missing variant declaration {variant_name:?}"
                    ));
                }
                let Some(text) = extract_single_literal(named.inner.as_ref()) else {
                    return Err(format!(
                        "support enum {rule_name} variant {variant_name:?} must be a literal"
                    ));
                };
                parsers.push(format!("just({text:?}).to({rule_name}::{variant_name})"));
            }
            Ok(if parsers.len() == 1 {
                format!("({}).then_ignore(ws()).boxed()", parsers[0])
            } else {
                format!("choice(({})).then_ignore(ws()).boxed()", parsers.join(", "))
            })
        }
        RuleTargetDecl::Support(
            NormalizedSupportDecl::String | NormalizedSupportDecl::Int | NormalizedSupportDecl::Id,
        ) => render_scalar_support_rule_expr(
            repr,
            rule_name,
            match decl {
                RuleTargetDecl::Support(kind) => kind,
                _ => unreachable!(),
            },
            rule,
            rule_names,
            node_names,
        ),
        RuleTargetDecl::Support(NormalizedSupportDecl::StringSeq | NormalizedSupportDecl::Unit) => {
            Err(format!(
                "support rule target {rule_name:?} has no parser strategy yet"
            ))
        }
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

fn render_token_parser_body(spec: &NormalizedTokenSpec) -> Result<String, String> {
    match spec.regex.as_str() {
        "[A-Za-z_][A-Za-z0-9_]*" => {
            Ok("text::ident::<_, ParseExtra<'src>>().map(str::to_owned)".to_owned())
        }
        "@[A-Za-z_][A-Za-z0-9_.]*" => Ok(
            "just('@')\n        .then(text::ident::<_, ParseExtra<'src>>().map(str::to_owned))\n        .then(\n            just('.')\n                .ignore_then(text::ident::<_, ParseExtra<'src>>().map(str::to_owned))\n                .repeated()\n                .collect::<Vec<_>>()\n        )\n        .map(|((_, head), tail)| {\n            let mut out = format!(\"@{head}\");\n            for part in tail {\n                out.push('.');\n                out.push_str(&part);\n            }\n            out\n        })"
                .to_owned(),
        ),
        "[0-9]+" => Ok("text::int::<_, ParseExtra<'src>>(10).map(str::to_owned)".to_owned()),
        "\"[^\\\"]*\"" => Ok(
            "just('\"')\n        .ignore_then(any().filter(|c: &char| *c != '\"' && *c != '\\n').repeated().collect::<String>())\n        .then_ignore(just('\"'))"
                .to_owned(),
        ),
        other => Err(format!(
            "unsupported token regex pattern {other:?}; token codegen currently supports a small explicit subset"
        )),
    }
}

fn render_token_parser_fn(token_name: &str, spec: &NormalizedTokenSpec) -> Result<String, String> {
    let fn_name = token_parser_fn_name(token_name);
    let body = render_token_parser_body(spec)?;

    Ok(format!(
        "fn {fn_name}<'src>() -> impl Parser<'src, &'src str, String, ParseExtra<'src>> + Clone {{\n    {body}\n}}"
    ))
}

fn extract_single_literal(rule: &SyntaxRule) -> Option<&str> {
    match rule {
        SyntaxRule::Literal(text) => Some(text.as_str()),
        SyntaxRule::Seq(items) if items.len() == 1 => extract_single_literal(&items[0]),
        _ => None,
    }
}

fn collect_rule_refs(rule: &SyntaxRule, out: &mut HashSet<String>) {
    match rule {
        SyntaxRule::Ref { name } => {
            out.insert(name.clone());
        }
        SyntaxRule::Seq(items) | SyntaxRule::Choice(items) => {
            for item in items {
                collect_rule_refs(item, out);
            }
        }
        SyntaxRule::Field(named) | SyntaxRule::Variant(named) => {
            collect_rule_refs(named.inner.as_ref(), out);
        }
        SyntaxRule::Optional { inner } => collect_rule_refs(inner, out),
        SyntaxRule::Repeat { item, .. } => collect_rule_refs(item, out),
        SyntaxRule::Token { .. } | SyntaxRule::Literal(_) => {}
    }
}

pub(crate) fn render_parser_block(
    repr: &NormalizedRepr,
    node_names: &[String],
    provenance_tag: &str,
) -> Result<String, String> {
    let rule_names = repr.syntax.rules.keys().cloned().collect::<HashSet<_>>();
    let mut referenced_rules = HashSet::new();
    for rule in repr.syntax.rules.values() {
        collect_rule_refs(rule, &mut referenced_rules);
    }
    let parser_order = derive_parser_order(repr)?;
    let root_name = repr.syntax.root.as_str();
    let root_parser_name = format!("{}_parser", snake_case(root_name));
    let root_fn_suffix = snake_case(root_name);
    let root_handle_name = format!("{root_name}Handle");

    let parser_defs = parser_order
        .iter()
        .map(|name| {
            let rule = repr
                .syntax
                .rules
                .get(name)
                .ok_or_else(|| format!("missing syntax rule for {name}"))?;
            let decl = repr
                .nodes
                .get(name)
                .map(|decl| RuleTargetDecl::Node(&decl.value))
                .or_else(|| {
                    repr.support
                        .get(name)
                        .map(|decl| RuleTargetDecl::Support(&decl.value))
                })
                .ok_or_else(|| format!("missing type declaration for {name}"))?;
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
            let binding_name = if name == root_name || referenced_rules.contains(name) {
                parser_name.clone()
            } else {
                format!("_{parser_name}")
            };
            Ok(if rule_is_self_recursive(name, rule) {
                format!("    let {binding_name} = recursive(move |{parser_name}| {parser_expr});")
            } else {
                format!("    let {binding_name} = {parser_expr};")
            })
        })
        .collect::<Result<Vec<_>, String>>()?
        .join("\n");

    let mut token_names = repr.syntax.token_specs.keys().cloned().collect::<Vec<_>>();
    token_names.sort();
    let token_rows = token_names
        .iter()
        .map(|name| render_token_parser_fn(name, repr.syntax.token_specs.get(name).unwrap()))
        .collect::<Result<Vec<_>, _>>()?
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

fn line_ws<'src>() -> impl Parser<'src, &'src str, (), ParseExtra<'src>> + Clone {{
    one_of(" \t").repeated().ignored()
}}

fn doc_comment_line<'src>() -> impl Parser<'src, &'src str, String, ParseExtra<'src>> + Clone {{
    line_ws()
        .ignore_then(just("///"))
        .ignore_then(just(' ').or_not())
        .ignore_then(any().filter(|c: &char| *c != '\n').repeated().collect::<String>())
        .then_ignore(just('\n').or_not())
}}

fn doc_block<'src>() -> impl Parser<'src, &'src str, Option<DocBlock>, ParseExtra<'src>> + Clone {{
    doc_comment_line()
        .repeated()
        .collect::<Vec<_>>()
        .map(|lines| if lines.is_empty() {{ None }} else {{ Some(DocBlock(lines)) }})
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

fn parse_root_value_rich(
    source: &str,
    file_id: Option<u32>,
) -> Result<{root_name}, Vec<Rich<'_, char>>> {{
{parser_defs}

    {root_parser_name}
        .then_ignore(end())
        .parse(source)
        .into_result()
}}

pub fn parse_root_text_rich(
    source: &str,
    file_id: Option<u32>,
) -> Result<{root_name}, Vec<Rich<'_, char>>> {{
    parse_root_value_rich(source, file_id)
}}

pub fn parse_root_text_with_file_id(source: &str, file_id: Option<u32>) -> Result<{root_name}, String> {{
    parse_root_value_rich(source, file_id).map_err(|errs| crate::format_rich_errors(source, errs))
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

pub fn parse_root_into_graph_rich<'src>(
    graph: &mut Graph,
    source: &'src str,
    file_id: Option<u32>,
) -> Result<{root_handle_name}, Vec<Rich<'src, char>>> {{
    let root = parse_root_value_rich(source, file_id)?;
    Ok(graph.insert_root(root))
}}

pub fn parse_root_into_graph_with_file_id(
    graph: &mut Graph,
    source: &str,
    file_id: Option<u32>,
) -> Result<{root_handle_name}, String> {{
    parse_root_into_graph_rich(graph, source, file_id)
        .map_err(|errs| crate::format_rich_errors(source, errs))
}}

pub fn parse_root_into_graph(graph: &mut Graph, source: &str) -> Result<{root_handle_name}, String> {{
    parse_root_into_graph_with_file_id(graph, source, None)
}}

pub fn parse_{root_fn_suffix}_into_graph_rich<'src>(
    graph: &mut Graph,
    source: &'src str,
    file_id: Option<u32>,
) -> Result<{root_handle_name}, Vec<Rich<'src, char>>> {{
    parse_root_into_graph_rich(graph, source, file_id)
}}

pub fn parse_{root_fn_suffix}_into_graph_with_file_id(
    graph: &mut Graph,
    source: &str,
    file_id: Option<u32>,
) -> Result<{root_handle_name}, String> {{
    parse_root_into_graph_with_file_id(graph, source, file_id)
}}

pub fn parse_{root_fn_suffix}_into_graph(
    graph: &mut Graph,
    source: &str,
) -> Result<{root_handle_name}, String> {{
    parse_root_into_graph(graph, source)
}}
"#,
        token_rows = token_rows,
        parser_defs = parser_defs,
        root_name = root_name,
        root_handle_name = root_handle_name,
        root_parser_name = root_parser_name,
        root_fn_suffix = root_fn_suffix,
    ))
}
