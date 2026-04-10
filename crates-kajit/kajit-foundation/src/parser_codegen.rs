use std::collections::{BTreeSet, HashMap, HashSet};

use crate::normalize::{
    DocumentedValue, NormalizedNodeDecl, NormalizedRepr, NormalizedSupportDecl,
    NormalizedTokenSpec, SyntaxRepeatSeparator, SyntaxRule, SyntaxTypeUse, direct_ref_name,
    is_docs_type, is_id_type, is_int_scalar_type, is_string_scalar_type, render_default_value,
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

#[derive(Debug, Clone)]
struct ArenaParserInfo {
    item: String,
    key: String,
    capture_var: String,
}

fn arena_key_accessor_expr(
    item_name: &str,
    decl: &NormalizedNodeDecl,
    key_name: &str,
    value_expr: &str,
) -> Result<String, String> {
    match decl {
        NormalizedNodeDecl::Record { fields, .. } => {
            let matching_fields = fields
                .iter()
                .filter_map(|(field_name, ty)| match &ty.value {
                    SyntaxTypeUse::Ref { name } if name == key_name => Some(field_name.clone()),
                    _ => None,
                })
                .collect::<Vec<_>>();
            let key_field = matching_fields
                .iter()
                .find(|field_name| field_name.as_str() == "id")
                .cloned()
                .or_else(|| {
                    if matching_fields.len() == 1 {
                        matching_fields.first().cloned()
                    } else {
                        None
                    }
                })
                .ok_or_else(|| {
                    format!(
                        "keyed arena item {item_name} must expose a unique field of type {key_name}"
                    )
                })?;
            Ok(format!("{value_expr}.{}", rust_ident(&key_field)))
        }
        NormalizedNodeDecl::Enum(variants) => {
            let mut arms = Vec::new();
            let mut variant_names = variants.keys().cloned().collect::<Vec<_>>();
            variant_names.sort();
            for variant_name in variant_names {
                let variant = &variants[&variant_name].value;
                match variant {
                    NormalizedNodeDecl::Record { fields, .. } => {
                        let matching_fields = fields
                            .iter()
                            .filter_map(|(field_name, ty)| match &ty.value {
                                SyntaxTypeUse::Ref { name } if name == key_name => {
                                    Some(field_name.clone())
                                }
                                _ => None,
                            })
                            .collect::<Vec<_>>();
                        let key_field = matching_fields
                            .iter()
                            .find(|field_name| field_name.as_str() == "id")
                            .cloned()
                            .or_else(|| {
                                if matching_fields.len() == 1 {
                                    matching_fields.first().cloned()
                                } else {
                                    None
                                }
                            })
                            .ok_or_else(|| {
                                format!(
                                    "keyed arena item {item_name}.{variant_name} must expose a unique field of type {key_name}"
                                )
                            })?;
                        let key_ident = rust_ident(&key_field);
                        arms.push(format!(
                            "        {item_name}::{variant_name} {{ {key_ident}, .. }} => *{key_ident},"
                        ));
                    }
                    NormalizedNodeDecl::Enum(_) => {
                        return Err(format!(
                            "keyed arena item {item_name}.{variant_name} cannot be nested enum"
                        ));
                    }
                }
            }
            Ok(format!(
                "match {value_expr} {{\n{}\n    }}",
                arms.join("\n")
            ))
        }
    }
}

fn collect_arena_parser_infos(repr: &NormalizedRepr) -> Result<Vec<ArenaParserInfo>, String> {
    let mut by_item = HashMap::<String, ArenaParserInfo>::new();
    let mut collect_from_fields =
        |fields: &HashMap<String, DocumentedValue<SyntaxTypeUse>>| -> Result<(), String> {
            for ty in fields.values() {
                if let SyntaxTypeUse::Arena {
                    item: inner,
                    key: Some(key),
                } = &ty.value
                    && let SyntaxTypeUse::Ref { name: item_name } = inner.as_ref()
                {
                    let decl = repr.nodes.get(item_name).ok_or_else(|| {
                        format!("keyed arena item {item_name:?} must be a declared node")
                    })?;
                    let key_expr = arena_key_accessor_expr(item_name, &decl.value, key, "value")?;
                    let info = ArenaParserInfo {
                        item: item_name.clone(),
                        key: key.clone(),
                        capture_var: format!("arena_{}_values", snake_case(item_name)),
                    };
                    if let Some(existing) = by_item.get(item_name) {
                        if existing.key != *key {
                            return Err(format!(
                                "arena item {item_name} used with conflicting keys: {:?} vs {:?}",
                                existing.key, key
                            ));
                        }
                    } else {
                        let _ = key_expr;
                        by_item.insert(item_name.clone(), info);
                    }
                }
            }
            Ok(())
        };

    for decl in repr.nodes.values() {
        match &decl.value {
            NormalizedNodeDecl::Record { fields, .. } => collect_from_fields(fields)?,
            NormalizedNodeDecl::Enum(variants) => {
                for variant in variants.values() {
                    if let NormalizedNodeDecl::Record { fields, .. } = &variant.value {
                        collect_from_fields(fields)?;
                    }
                }
            }
        }
    }

    let mut out = by_item.into_values().collect::<Vec<_>>();
    out.sort_by(|a, b| a.item.cmp(&b.item));
    Ok(out)
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
    arena_infos: &[ArenaParserInfo],
) -> Result<String, String> {
    if let SyntaxTypeUse::Optional(inner_ty) = ty
        && !matches!(rule, SyntaxRule::Optional { .. })
    {
        return render_value_parser(
            repr,
            rule,
            inner_ty,
            rule_names,
            node_names,
            box_node_refs,
            arena_infos,
        );
    }

    if let SyntaxTypeUse::RefTo { id, .. } = ty {
        return render_value_parser(
            repr,
            rule,
            id,
            rule_names,
            node_names,
            box_node_refs,
            arena_infos,
        );
    }

    match rule {
        SyntaxRule::Semantic { inner, .. } => render_value_parser(
            repr,
            inner,
            ty,
            rule_names,
            node_names,
            box_node_refs,
            arena_infos,
        ),
        SyntaxRule::Tag(text) => Ok(format!("just({text:?}).padded()")),
        SyntaxRule::Token { name } => render_token_value_parser(repr, name, ty),
        SyntaxRule::Ref { name } => {
            render_ref_value_parser(repr, name, ty, rule_names, node_names, box_node_refs)
        }
        SyntaxRule::Optional { inner } => {
            let SyntaxTypeUse::Optional(inner_ty) = ty else {
                return Err("optional rule without optional type".to_owned());
            };
            let inner = render_value_parser(
                repr,
                inner,
                inner_ty,
                rule_names,
                node_names,
                true,
                arena_infos,
            )?;
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
            let inner = render_value_parser(
                repr,
                item,
                inner_ty,
                rule_names,
                node_names,
                false,
                arena_infos,
            )?;
            let collected = if let Some(sep) = sep {
                let sep_parser = match sep {
                    SyntaxRepeatSeparator::Literal(text) => format!("just({text:?}).padded()"),
                    SyntaxRepeatSeparator::RuleRef(name) => {
                        let target = repr.syntax.rules.get(name).ok_or_else(|| {
                            format!("repeat separator ref target {name:?} does not exist")
                        })?;
                        render_separator_parser(repr, target)?
                    }
                };
                format!("({inner}).separated_by({sep_parser}).allow_trailing().collect::<Vec<_>>()")
            } else {
                format!("({inner}).repeated().collect::<Vec<_>>()")
            };
            Ok(match ty {
                SyntaxTypeUse::Seq(_) => collected,
                SyntaxTypeUse::Arena {
                    item: arena_item,
                    key: Some(_),
                } => {
                    let SyntaxTypeUse::Ref { name: item_name } = arena_item.as_ref() else {
                        return Err("keyed arena item must be a node reference".to_owned());
                    };
                    let info = arena_infos
                        .iter()
                        .find(|info| info.item == *item_name)
                        .ok_or_else(|| {
                            format!("missing parser arena info for item {item_name:?}")
                        })?;
                    let decl = repr.nodes.get(item_name).ok_or_else(|| {
                        format!("keyed arena item {item_name:?} must be a declared node")
                    })?;
                    let key_expr =
                        arena_key_accessor_expr(item_name, &decl.value, &info.key, "value")?;
                    format!(
                        "({collected}).map({{
            let {capture_var} = {capture_var}.clone();
            move |values| {{
                let mut ids = Vec::with_capacity(values.len());
                let mut arena = {capture_var}.borrow_mut();
                for value in values {{
                    let id = {key_expr};
                    ids.push(id);
                    arena.push(value);
                }}
                super::super::Order::from(ids)
            }}
        }})",
                        capture_var = info.capture_var,
                        key_expr = key_expr,
                    )
                }
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
    arena_infos: &[ArenaParserInfo],
) -> Result<Vec<SeqItem>, String> {
    match rule {
        SyntaxRule::Semantic { inner, .. } => {
            flatten_struct_rule_items(repr, inner, fields, rule_names, node_names, arena_infos)
        }
        SyntaxRule::Seq(items) => {
            let mut out = Vec::new();
            for item in items {
                out.extend(flatten_struct_rule_items(
                    repr,
                    item,
                    fields,
                    rule_names,
                    node_names,
                    arena_infos,
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
            let parser = render_value_parser(
                repr,
                named.inner.as_ref(),
                ty,
                rule_names,
                node_names,
                true,
                arena_infos,
            )?;
            Ok(vec![SeqItem::Bind {
                name: field_name.to_owned(),
                parser,
            }])
        }
        SyntaxRule::Optional { inner } => {
            let nested = flatten_struct_rule_items(
                repr,
                inner,
                fields,
                rule_names,
                node_names,
                arena_infos,
            )?;
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
        SyntaxRule::Tag(text) => Ok(vec![SeqItem::Ignore(format!("just({text:?}).padded()"))]),
        SyntaxRule::Ref { name } => Ok(vec![SeqItem::Ignore(format!(
            "({}).map(|_| ())",
            format!("{}_parser.clone()", snake_case(name))
        ))]),
        SyntaxRule::Token { name } => Ok(vec![SeqItem::Ignore(format!(
            "{}().then_ignore(ws()).to(())",
            token_parser_fn_name(name)
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
    arena_infos: &[ArenaParserInfo],
) -> Result<String, String> {
    let support_ty = SyntaxTypeUse::Ref {
        name: support_name.to_owned(),
    };

    match rule {
        SyntaxRule::Semantic { inner, .. } => render_scalar_support_rule_expr(
            repr,
            support_name,
            decl,
            inner,
            rule_names,
            node_names,
            arena_infos,
        ),
        SyntaxRule::Token { .. }
        | SyntaxRule::Ref { .. }
        | SyntaxRule::Optional { .. }
        | SyntaxRule::Repeat { .. } => {
            return render_value_parser(
                repr,
                rule,
                &support_ty,
                rule_names,
                node_names,
                false,
                arena_infos,
            );
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
                    SyntaxRule::Tag(text) => {
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
    arena_infos: &[ArenaParserInfo],
) -> Result<String, String> {
    if is_prov_only_struct(fields, provenance_tag) {
        if let Some(text) = extract_single_literal(rule) {
            return Ok(format!(
                "(just({text:?}).to(())).map_with(move |(), e| {type_name} {{ prov: prov_from_span(e.span(), file_id) }}).then_ignore(ws()).boxed()"
            ));
        }
    }

    let mut items =
        flatten_struct_rule_items(repr, rule, fields, rule_names, node_names, arena_infos)?;
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
    arena_infos: &[ArenaParserInfo],
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
        let mut items = flatten_struct_rule_items(
            repr,
            named.inner.as_ref(),
            fields,
            rule_names,
            node_names,
            arena_infos,
        )?;
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
    arena_infos: &[ArenaParserInfo],
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
                arena_infos,
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
            arena_infos,
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
                arena_infos,
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
            arena_infos,
        ),
        RuleTargetDecl::Support(NormalizedSupportDecl::StringSeq | NormalizedSupportDecl::Unit) => {
            Err(format!(
                "support rule target {rule_name:?} has no parser strategy yet"
            ))
        }
    }
}

fn render_syntax_only_rule_parser_expr(
    repr: &NormalizedRepr,
    rule: &SyntaxRule,
) -> Result<String, String> {
    fn render_sep_expr(
        repr: &NormalizedRepr,
        sep: &SyntaxRepeatSeparator,
    ) -> Result<String, String> {
        match sep {
            SyntaxRepeatSeparator::Literal(text) => Ok(format!("just({text:?}).padded()")),
            SyntaxRepeatSeparator::RuleRef(name) => {
                let target = repr
                    .syntax
                    .rules
                    .get(name)
                    .ok_or_else(|| format!("repeat separator rule {name:?} does not exist"))?;
                render_syntax_only_rule_parser_expr(repr, target)
            }
        }
    }

    match rule {
        SyntaxRule::Semantic { inner, .. } => render_syntax_only_rule_parser_expr(repr, inner),
        SyntaxRule::Tag(text) | SyntaxRule::Literal(text) => {
            Ok(format!("just({text:?}).padded().to(()).boxed()"))
        }
        SyntaxRule::Token { name } => Ok(format!(
            "{}().then_ignore(ws()).to(()).boxed()",
            token_parser_fn_name(name)
        )),
        SyntaxRule::Ref { name } => Ok(format!(
            "{}_parser.clone().map(|_| ()).boxed()",
            snake_case(name)
        )),
        SyntaxRule::Optional { inner } => {
            let inner = render_syntax_only_rule_parser_expr(repr, inner)?;
            Ok(format!("({inner}).or_not().to(()).boxed()"))
        }
        SyntaxRule::Repeat { item, sep } => {
            let item_expr = render_syntax_only_rule_parser_expr(repr, item)?;
            let repeated = if let Some(sep) = sep {
                let sep_expr = render_sep_expr(repr, sep)?;
                format!(
                    "({item_expr}).separated_by({sep_expr}).allow_trailing().collect::<Vec<_>>()"
                )
            } else {
                format!("({item_expr}).repeated().collect::<Vec<_>>()")
            };
            Ok(format!("({repeated}).to(()).boxed()"))
        }
        SyntaxRule::Seq(items) => {
            let mut iter = items.iter();
            let Some(first) = iter.next() else {
                return Ok("empty().to(()).boxed()".to_owned());
            };
            let mut chain = render_syntax_only_rule_parser_expr(repr, first)?;
            for item in iter {
                let part = render_syntax_only_rule_parser_expr(repr, item)?;
                chain = format!("({chain}).ignore_then({part})");
            }
            Ok(format!("({chain}).to(()).boxed()"))
        }
        SyntaxRule::Choice(items) => {
            if items.is_empty() {
                return Err("choice rule must have at least one item".to_owned());
            }
            let branches = items
                .iter()
                .map(|item| render_syntax_only_rule_parser_expr(repr, item))
                .collect::<Result<Vec<_>, _>>()?
                .join(", ");
            Ok(format!("choice(({branches})).boxed()"))
        }
        SyntaxRule::Field(_) | SyntaxRule::Variant(_) => {
            Err("syntax-only rules cannot contain @field or @variant".to_owned())
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
        SyntaxRule::Semantic { inner, .. } => {
            collect_rule_dependencies(inner, rule_names, out);
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
        SyntaxRule::Token { .. } | SyntaxRule::Tag(_) | SyntaxRule::Literal(_) => {}
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
        SyntaxRule::Tag(text) => Some(text.as_str()),
        SyntaxRule::Seq(items) if items.len() == 1 => extract_single_literal(&items[0]),
        SyntaxRule::Semantic { inner, .. } => extract_single_literal(inner),
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
        SyntaxRule::Semantic { inner, .. } => collect_rule_refs(inner, out),
        SyntaxRule::Field(named) | SyntaxRule::Variant(named) => {
            collect_rule_refs(named.inner.as_ref(), out);
        }
        SyntaxRule::Optional { inner } => collect_rule_refs(inner, out),
        SyntaxRule::Repeat { item, .. } => collect_rule_refs(item, out),
        SyntaxRule::Token { .. } | SyntaxRule::Tag(_) | SyntaxRule::Literal(_) => {}
    }
}

pub(crate) fn render_parser_block(
    repr: &NormalizedRepr,
    node_names: &[String],
    provenance_tag: &str,
) -> Result<String, String> {
    let arena_infos = collect_arena_parser_infos(repr)?;
    let has_graph_arenas = !arena_infos.is_empty();
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
                });
            let parser_expr = if let Some(decl) = decl {
                render_rule_parser_expr(
                    repr,
                    name,
                    rule,
                    decl,
                    &rule_names,
                    node_names,
                    provenance_tag,
                    &arena_infos,
                )?
            } else {
                render_syntax_only_rule_parser_expr(repr, rule)?
            };
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

    let arena_capture_init_rows = arena_infos
        .iter()
        .map(|info| {
            format!(
                "    let {capture_var} = std::rc::Rc::new(std::cell::RefCell::new(Vec::<{item}>::new()));",
                capture_var = info.capture_var,
                item = info.item,
            )
        })
        .collect::<Vec<_>>()
        .join("\n");

    let arena_capture_struct_rows = arena_infos
        .iter()
        .map(|info| format!("    {}: Vec<{}>,", info.capture_var, info.item))
        .collect::<Vec<_>>()
        .join("\n");
    let arena_capture_build_rows = arena_infos
        .iter()
        .map(|info| {
            format!(
                "        {field}: std::mem::take(&mut *{field}.borrow_mut()),",
                field = info.capture_var
            )
        })
        .collect::<Vec<_>>()
        .join("\n");
    let arena_insert_rows = arena_infos
        .iter()
        .map(|info| {
            format!(
                "    for value in captures.{field} {{\n        graph.insert_{method}(value).map_err(|err| vec![Rich::custom(chumsky::span::SimpleSpan::new(0, 0), err)])?;\n    }}",
                field = info.capture_var,
                method = snake_case(&info.item),
            )
        })
        .collect::<Vec<_>>()
        .join("\n");

    let parse_return_ty = if has_graph_arenas {
        format!("({root_name}, ParseArenaCaptures)")
    } else {
        root_name.to_owned()
    };
    let parse_result_value = if has_graph_arenas {
        format!(
            "    let root = {root_parser_name}\n        .then_ignore(end())\n        .parse(source)\n        .into_result()?;\n    Ok((root, ParseArenaCaptures {{\n{arena_capture_build_rows}\n    }}))"
        )
    } else {
        format!(
            "    {root_parser_name}\n        .then_ignore(end())\n        .parse(source)\n        .into_result()"
        )
    };
    let parse_root_text_from_value = if has_graph_arenas {
        "    parse_root_value_rich(source, file_id).map(|(root, _)| root)"
    } else {
        "    parse_root_value_rich(source, file_id)"
    };
    let parse_root_text_with_file_id_body = if has_graph_arenas {
        "    parse_root_value_rich(source, file_id)\n        .map(|(root, _)| root)\n        .map_err(|errs| crate::format_rich_errors(source, errs))"
    } else {
        "    parse_root_value_rich(source, file_id)\n        .map_err(|errs| crate::format_rich_errors(source, errs))"
    };
    let parse_into_graph_preamble = if has_graph_arenas {
        "    let (root, captures) = parse_root_value_rich(source, file_id)?;"
    } else {
        "    let root = parse_root_value_rich(source, file_id)?;"
    };
    let arena_capture_struct_block = if has_graph_arenas {
        format!(
            "\n#[derive(Debug, Default)]\nstruct ParseArenaCaptures {{\n{arena_capture_struct_rows}\n}}\n"
        )
    } else {
        String::new()
    };

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

{arena_capture_struct_block}

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
) -> Result<{parse_return_ty}, Vec<Rich<'_, char>>> {{
{arena_capture_init_rows}
{parser_defs}
{parse_result_value}
}}

pub fn parse_root_text_rich(
    source: &str,
    file_id: Option<u32>,
) -> Result<{root_name}, Vec<Rich<'_, char>>> {{
{parse_root_text_from_value}
}}

pub fn parse_root_text_with_file_id(source: &str, file_id: Option<u32>) -> Result<{root_name}, String> {{
{parse_root_text_with_file_id_body}
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
{parse_into_graph_preamble}
{arena_insert_rows}
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
        arena_capture_struct_block = arena_capture_struct_block,
        parse_return_ty = parse_return_ty,
        arena_capture_init_rows = arena_capture_init_rows,
        parser_defs = parser_defs,
        parse_result_value = parse_result_value,
        parse_root_text_from_value = parse_root_text_from_value,
        parse_root_text_with_file_id_body = parse_root_text_with_file_id_body,
        parse_into_graph_preamble = parse_into_graph_preamble,
        arena_insert_rows = arena_insert_rows,
        root_name = root_name,
        root_handle_name = root_handle_name,
        root_fn_suffix = root_fn_suffix,
    ))
}
fn render_separator_parser(repr: &NormalizedRepr, rule: &SyntaxRule) -> Result<String, String> {
    match rule {
        SyntaxRule::Tag(text) | SyntaxRule::Literal(text) => Ok(format!("just({text:?}).padded()")),
        SyntaxRule::Semantic { inner, .. } => render_separator_parser(repr, inner),
        SyntaxRule::Ref { name } => {
            let target =
                repr.syntax.rules.get(name).ok_or_else(|| {
                    format!("repeat separator ref target {name:?} does not exist")
                })?;
            render_separator_parser(repr, target)
        }
        other => Err(format!(
            "repeat separator must resolve to @tag(\"...\") via optional @ref/@semantic wrapper, got {other:?}"
        )),
    }
}
