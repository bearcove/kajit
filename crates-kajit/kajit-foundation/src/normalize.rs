use std::collections::HashMap;

use indexmap::IndexMap;

use crate::schema::{
    ModernReprBody, ModernRuleDecl, SyntaxExpr, SyntaxObject, SyntaxPayload, TemplateDecl,
    TemplateSyntaxDecl, documented_doc, documented_name,
};

#[derive(Debug, Clone)]
pub(crate) struct SyntaxRuleNamed {
    pub(crate) name: String,
    pub(crate) inner: Box<SyntaxRule>,
}

#[derive(Debug, Clone)]
pub(crate) enum SyntaxRule {
    Seq(Vec<SyntaxRule>),
    Choice(Vec<SyntaxRule>),
    Semantic {
        kind: String,
        inner: Box<SyntaxRule>,
    },
    Tag(String),
    Field(SyntaxRuleNamed),
    Variant(SyntaxRuleNamed),
    Ref {
        name: String,
    },
    Token {
        name: String,
    },
    Optional {
        inner: Box<SyntaxRule>,
    },
    Repeat {
        item: Box<SyntaxRule>,
        sep: Option<SyntaxRepeatSeparator>,
    },
    Literal(String),
}

#[derive(Debug, Clone)]
pub(crate) enum SyntaxRepeatSeparator {
    Literal(String),
    RuleRef(String),
}

#[derive(Debug, Clone)]
pub(crate) enum SyntaxTypeUse {
    Optional(Box<SyntaxTypeUse>),
    Seq(Box<SyntaxTypeUse>),
    Arena {
        item: Box<SyntaxTypeUse>,
        key: Option<String>,
    },
    Pool {
        item: Box<SyntaxTypeUse>,
        key: Option<String>,
    },
    Order(Box<SyntaxTypeUse>),
    RefTo {
        id: Box<SyntaxTypeUse>,
        target: String,
    },
    Ref {
        name: String,
    },
}

#[derive(Debug, Clone)]
pub(crate) struct NormalizedContract {
    pub(crate) purpose: String,
    pub(crate) canonical_identities: Vec<String>,
    pub(crate) round_trip: String,
    pub(crate) provenance: String,
}

#[derive(Debug, Clone)]
pub(crate) struct NormalizedSyntax {
    pub(crate) root: String,
    pub(crate) token_specs: HashMap<String, NormalizedTokenSpec>,
    pub(crate) rules: HashMap<String, SyntaxRule>,
    pub(crate) canonical_print: HashMap<String, String>,
    pub(crate) semantic_tokens: HashMap<String, String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct NormalizedTokenSpec {
    pub(crate) regex: String,
}

#[derive(Debug, Clone)]
pub(crate) struct DocumentedValue<T> {
    pub(crate) value: T,
    pub(crate) doc: Option<Vec<String>>,
}

#[derive(Debug, Clone)]
pub(crate) enum NormalizedSupportDecl {
    String,
    Int,
    Id,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum NormalizedRefKind {
    Provenance,
    StringScalar,
    IntScalar,
    Id,
    StringSeq,
    Unit,
    Enum,
    Unknown,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum NormalizedNodeKind {
    Struct,
}

#[derive(Debug, Clone)]
pub(crate) enum NormalizedNodeDecl {
    Record {
        kind: NormalizedNodeKind,
        fields: HashMap<String, DocumentedValue<SyntaxTypeUse>>,
    },
    Enum(IndexMap<String, DocumentedValue<NormalizedNodeDecl>>),
}

#[derive(Debug, Clone)]
pub(crate) struct NormalizedRepr {
    pub(crate) doc: Option<Vec<String>>,
    pub(crate) name: String,
    pub(crate) file_ext: String,
    pub(crate) contract: NormalizedContract,
    pub(crate) syntax: NormalizedSyntax,
    pub(crate) common: HashMap<String, SyntaxTypeUse>,
    pub(crate) support: HashMap<String, DocumentedValue<NormalizedSupportDecl>>,
    pub(crate) nodes: HashMap<String, DocumentedValue<NormalizedNodeDecl>>,
}

fn collect_inline_semantic_tokens(
    out: &mut HashMap<String, String>,
    type_name: &str,
    variant_name: Option<&str>,
    rule: &SyntaxRule,
    active_kind: Option<&str>,
) {
    match rule {
        SyntaxRule::Semantic { kind, inner } => {
            collect_inline_semantic_tokens(out, type_name, variant_name, inner, Some(kind));
        }
        SyntaxRule::Tag(text) | SyntaxRule::Literal(text) => {
            if let Some(kind) = active_kind {
                out.insert(format!("literal.{text}"), kind.to_owned());
            }
        }
        SyntaxRule::Field(named) => {
            if let Some(kind) = active_kind {
                let target = match variant_name {
                    Some(variant) => format!("field.{type_name}.{variant}.{}", named.name),
                    None => format!("field.{type_name}.{}", named.name),
                };
                out.insert(target, kind.to_owned());
            }
            collect_inline_semantic_tokens(out, type_name, variant_name, &named.inner, active_kind);
        }
        SyntaxRule::Variant(named) => {
            if let Some(kind) = active_kind {
                out.insert(
                    format!("variant.{type_name}.{}", named.name),
                    kind.to_owned(),
                );
            }
            collect_inline_semantic_tokens(
                out,
                type_name,
                Some(&named.name),
                &named.inner,
                active_kind,
            );
        }
        SyntaxRule::Seq(items) | SyntaxRule::Choice(items) => {
            for item in items {
                collect_inline_semantic_tokens(out, type_name, variant_name, item, active_kind);
            }
        }
        SyntaxRule::Optional { inner } => {
            collect_inline_semantic_tokens(out, type_name, variant_name, inner, active_kind);
        }
        SyntaxRule::Repeat { item, .. } => {
            collect_inline_semantic_tokens(out, type_name, variant_name, item, active_kind);
        }
        SyntaxRule::Ref { .. } | SyntaxRule::Token { .. } => {}
    }
}

#[derive(Debug, Clone)]
#[allow(dead_code)]
enum ModernResolvedRuleKind {
    Literal(String),
    Regex(String),
    Inline {
        syntax: Option<SyntaxExpr>,
        highlight: Option<String>,
    },
    Struct {
        syntax: Option<SyntaxExpr>,
        highlight: Option<String>,
    },
    Enum,
}

#[derive(Debug, Clone)]
struct ModernResolvedRule {
    doc: Option<Vec<String>>,
    kind: ModernResolvedRuleKind,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ModernSupportClass {
    String,
    Int,
    Id,
}

fn modern_scalar_expr(text: &str) -> SyntaxExpr {
    SyntaxExpr::Other {
        tag: None,
        content: Some(SyntaxPayload::Scalar(text.to_owned())),
    }
}

fn modern_extract_template(decl: &TemplateDecl) -> Result<&TemplateSyntaxDecl, String> {
    match &decl.syntax {
        SyntaxExpr::Template(template) => Ok(template),
        other => Err(format!(
            "template declaration must use @template{{...}}, got {other:?}"
        )),
    }
}

fn modern_param_name(expr: &SyntaxExpr) -> Result<String, String> {
    let SyntaxExpr::Other {
        tag: None,
        content: Some(SyntaxPayload::Object(SyntaxObject { fields })),
    } = expr
    else {
        return Err(format!(
            "template param must be a single field binding object, got {expr:?}"
        ));
    };
    let mut iter = fields.keys();
    let Some(name) = iter.next() else {
        return Err("template param object must contain exactly one field".to_owned());
    };
    if iter.next().is_some() {
        return Err("template param object must contain exactly one field".to_owned());
    }
    Ok(documented_name(name).to_owned())
}

fn modern_payload_items(payload: Option<&SyntaxPayload>) -> Vec<SyntaxExpr> {
    match payload {
        None => Vec::new(),
        Some(SyntaxPayload::Scalar(text)) => vec![modern_scalar_expr(text)],
        Some(SyntaxPayload::Seq(items)) => items.clone(),
        Some(SyntaxPayload::Object(object)) => vec![SyntaxExpr::Other {
            tag: None,
            content: Some(SyntaxPayload::Object(object.clone())),
        }],
    }
}

fn modern_substitute_expr(expr: &SyntaxExpr, bindings: &HashMap<String, SyntaxExpr>) -> SyntaxExpr {
    match expr {
        SyntaxExpr::Template(inner) => SyntaxExpr::Template(Box::new(TemplateSyntaxDecl {
            params: inner
                .params
                .iter()
                .map(|item| modern_substitute_expr(item, bindings))
                .collect(),
            body: crate::schema::ModernStructDecl {
                syntax: inner
                    .body
                    .syntax
                    .as_ref()
                    .map(|syntax| modern_substitute_expr(syntax, bindings)),
                highlight: inner.body.highlight.clone(),
            },
        })),
        SyntaxExpr::Regex(items) => SyntaxExpr::Regex(
            items
                .iter()
                .map(|item| modern_substitute_expr(item, bindings))
                .collect(),
        ),
        SyntaxExpr::BraceBlock(items) => SyntaxExpr::BraceBlock(
            items
                .iter()
                .map(|item| modern_substitute_expr(item, bindings))
                .collect(),
        ),
        SyntaxExpr::ParenList(items) => SyntaxExpr::ParenList(
            items
                .iter()
                .map(|item| modern_substitute_expr(item, bindings))
                .collect(),
        ),
        SyntaxExpr::BracketList(items) => SyntaxExpr::BracketList(
            items
                .iter()
                .map(|item| modern_substitute_expr(item, bindings))
                .collect(),
        ),
        SyntaxExpr::Paren(items) => SyntaxExpr::Paren(
            items
                .iter()
                .map(|item| modern_substitute_expr(item, bindings))
                .collect(),
        ),
        SyntaxExpr::Bracket(items) => SyntaxExpr::Bracket(
            items
                .iter()
                .map(|item| modern_substitute_expr(item, bindings))
                .collect(),
        ),
        SyntaxExpr::Optional(items) => SyntaxExpr::Optional(
            items
                .iter()
                .map(|item| modern_substitute_expr(item, bindings))
                .collect(),
        ),
        SyntaxExpr::Many(items) => SyntaxExpr::Many(
            items
                .iter()
                .map(|item| modern_substitute_expr(item, bindings))
                .collect(),
        ),
        SyntaxExpr::SeparatedBy(items) => SyntaxExpr::SeparatedBy(
            items
                .iter()
                .map(|item| modern_substitute_expr(item, bindings))
                .collect(),
        ),
        SyntaxExpr::Indent(items) => SyntaxExpr::Indent(
            items
                .iter()
                .map(|item| modern_substitute_expr(item, bindings))
                .collect(),
        ),
        SyntaxExpr::SoftSpace => SyntaxExpr::SoftSpace,
        SyntaxExpr::Newline => SyntaxExpr::Newline,
        SyntaxExpr::Any => SyntaxExpr::Any,
        SyntaxExpr::Rule => SyntaxExpr::Rule,
        SyntaxExpr::Other {
            tag: None,
            content: Some(SyntaxPayload::Scalar(text)),
        } if bindings.contains_key(text) => bindings[text].clone(),
        SyntaxExpr::Other { tag, content } => SyntaxExpr::Other {
            tag: tag.clone(),
            content: content.as_ref().map(|payload| match payload {
                SyntaxPayload::Scalar(text) => {
                    if let Some(bound) = bindings.get(text) {
                        match bound {
                            SyntaxExpr::Other {
                                tag: None,
                                content: Some(payload),
                            } => payload.clone(),
                            other => SyntaxPayload::Seq(vec![other.clone()]),
                        }
                    } else {
                        SyntaxPayload::Scalar(text.clone())
                    }
                }
                SyntaxPayload::Seq(items) => SyntaxPayload::Seq(
                    items
                        .iter()
                        .map(|item| modern_substitute_expr(item, bindings))
                        .collect(),
                ),
                SyntaxPayload::Object(object) => SyntaxPayload::Object(SyntaxObject {
                    fields: object
                        .fields
                        .iter()
                        .map(|(name, value)| {
                            (name.clone(), modern_substitute_expr(value, bindings))
                        })
                        .collect(),
                }),
            }),
        },
    }
}

fn modern_expand_rule(
    rule: &ModernRuleDecl,
    templates: &IndexMap<String, TemplateSyntaxDecl>,
) -> Result<ModernResolvedRuleKind, String> {
    match rule {
        ModernRuleDecl::Literal(text) => Ok(ModernResolvedRuleKind::Literal(text.clone())),
        ModernRuleDecl::InlineStruct(body) => {
            if let Some(SyntaxExpr::Regex(items)) = &body.syntax
                && body.highlight.is_none()
                && items.len() == 1
                && let SyntaxExpr::Other {
                    tag: None,
                    content: Some(SyntaxPayload::Scalar(pattern)),
                } = &items[0]
            {
                return Ok(ModernResolvedRuleKind::Regex(pattern.clone()));
            }
            Ok(ModernResolvedRuleKind::Inline {
                syntax: body.syntax.clone(),
                highlight: body.highlight.clone(),
            })
        }
        ModernRuleDecl::Struct(body) => Ok(ModernResolvedRuleKind::Struct {
            syntax: body.syntax.clone(),
            highlight: body.highlight.clone(),
        }),
        ModernRuleDecl::Enum => Ok(ModernResolvedRuleKind::Enum),
        ModernRuleDecl::UserTag { tag, content } => {
            if tag.is_empty() {
                let Some(SyntaxPayload::Object(object)) = content else {
                    return Err("anonymous modern rule body must be an object".to_owned());
                };
                let syntax = object.fields.iter().find_map(|(name, value)| {
                    (documented_name(name) == "syntax").then(|| value.clone())
                });
                let highlight = object.fields.iter().find_map(|(name, value)| {
                    if documented_name(name) != "highlight" {
                        return None;
                    }
                    match value {
                        SyntaxExpr::Other {
                            tag: None,
                            content: Some(SyntaxPayload::Scalar(text)),
                        } => Some(text.clone()),
                        _ => None,
                    }
                });
                if let Some(SyntaxExpr::Regex(items)) = &syntax
                    && highlight.is_none()
                    && items.len() == 1
                    && let SyntaxExpr::Other {
                        tag: None,
                        content: Some(SyntaxPayload::Scalar(pattern)),
                    } = &items[0]
                {
                    return Ok(ModernResolvedRuleKind::Regex(pattern.clone()));
                }
                return Ok(ModernResolvedRuleKind::Inline { syntax, highlight });
            }
            let template = templates
                .get(tag)
                .ok_or_else(|| format!("unknown modern rule tag @{tag}"))?;
            let args = modern_payload_items(content.as_ref());
            if args.len() != template.params.len() {
                return Err(format!(
                    "template @{tag} expected {} args, got {}",
                    template.params.len(),
                    args.len()
                ));
            }
            let mut bindings = HashMap::new();
            for (param, arg) in template.params.iter().zip(args.iter()) {
                bindings.insert(modern_param_name(param)?, arg.clone());
            }
            Ok(ModernResolvedRuleKind::Inline {
                syntax: template
                    .body
                    .syntax
                    .as_ref()
                    .map(|syntax| modern_substitute_expr(syntax, &bindings)),
                highlight: template.body.highlight.clone(),
            })
        }
    }
}

fn modern_strip_layout(rule: Option<SyntaxRule>) -> Option<SyntaxRule> {
    match rule {
        None => None,
        Some(SyntaxRule::Seq(items)) => {
            let items = items
                .into_iter()
                .filter_map(|item| modern_strip_layout(Some(item)))
                .collect::<Vec<_>>();
            match items.as_slice() {
                [] => None,
                [only] => Some(only.clone()),
                _ => Some(SyntaxRule::Seq(items)),
            }
        }
        Some(other) => Some(other),
    }
}

fn modern_lower_object_binding(object: &SyntaxObject) -> Result<SyntaxRule, String> {
    let mut iter = object.fields.iter();
    let Some((name, value)) = iter.next() else {
        return Err("syntax object binding must contain exactly one field".to_owned());
    };
    if iter.next().is_some() {
        return Err("syntax object binding must contain exactly one field".to_owned());
    }
    Ok(SyntaxRule::Field(SyntaxRuleNamed {
        name: documented_name(name).to_owned(),
        inner: Box::new(modern_lower_syntax_expr(value)?.ok_or_else(|| {
            format!(
                "field binding {} lowered to empty syntax",
                documented_name(name)
            )
        })?),
    }))
}

fn modern_lower_container(
    open: &str,
    close: &str,
    items: &[SyntaxExpr],
    repeated: bool,
) -> Result<Option<SyntaxRule>, String> {
    let mut seq = vec![SyntaxRule::Literal(open.to_owned())];
    if let Some(first) = items.first() {
        let inner = modern_lower_syntax_expr(first)?
            .ok_or_else(|| "container body lowered to empty syntax".to_owned())?;
        let inner = if repeated {
            let sep = items.get(1).and_then(|item| {
                let Ok(Some(rule)) = modern_lower_syntax_expr(item) else {
                    return None;
                };
                match rule {
                    SyntaxRule::Literal(text) | SyntaxRule::Tag(text) => {
                        Some(SyntaxRepeatSeparator::Literal(text))
                    }
                    SyntaxRule::Ref { name } => Some(SyntaxRepeatSeparator::RuleRef(name)),
                    _ => None,
                }
            });
            SyntaxRule::Repeat {
                item: Box::new(inner),
                sep,
            }
        } else {
            inner
        };
        seq.push(inner);
    }
    seq.push(SyntaxRule::Literal(close.to_owned()));
    Ok(Some(SyntaxRule::Seq(seq)))
}

fn modern_lower_brace_block(items: &[SyntaxExpr]) -> Result<Option<SyntaxRule>, String> {
    let mut prefix = None;
    let mut body = None;
    let mut idx = 0;
    while idx < items.len() {
        if let SyntaxExpr::Other {
            tag: None,
            content: Some(SyntaxPayload::Scalar(name)),
        } = &items[idx]
            && let Some(value) = items.get(idx + 1)
        {
            match name.as_str() {
                "prefix" => prefix = modern_lower_syntax_expr(value)?,
                "body" => body = modern_lower_syntax_expr(value)?,
                _ => {}
            }
            idx += 2;
            continue;
        }
        idx += 1;
    }
    let mut out = Vec::new();
    if let Some(prefix) = modern_strip_layout(prefix) {
        out.push(prefix);
    }
    out.push(SyntaxRule::Literal("{".to_owned()));
    if let Some(body) = modern_strip_layout(body) {
        out.push(body);
    }
    out.push(SyntaxRule::Literal("}".to_owned()));
    Ok(Some(SyntaxRule::Seq(out)))
}

fn modern_lower_syntax_expr(expr: &SyntaxExpr) -> Result<Option<SyntaxRule>, String> {
    match expr {
        SyntaxExpr::Template(_) => {
            Err("@template is only valid inside template declarations".to_owned())
        }
        SyntaxExpr::Regex(_) => Err("@regex is only valid as a leaf rule body".to_owned()),
        SyntaxExpr::BraceBlock(items) => modern_lower_brace_block(items),
        SyntaxExpr::ParenList(items) => modern_lower_container("(", ")", items, true),
        SyntaxExpr::BracketList(items) => modern_lower_container("[", "]", items, true),
        SyntaxExpr::Paren(items) => modern_lower_container("(", ")", items, false),
        SyntaxExpr::Bracket(items) => modern_lower_container("[", "]", items, false),
        SyntaxExpr::Optional(items) => Ok(modern_strip_layout(Some(SyntaxRule::Optional {
            inner: Box::new(
                modern_lower_syntax_expr(&SyntaxExpr::Other {
                    tag: None,
                    content: Some(SyntaxPayload::Seq(items.clone())),
                })?
                .ok_or_else(|| "optional body lowered to empty syntax".to_owned())?,
            ),
        }))),
        SyntaxExpr::Many(items) => {
            let Some(first) = items.first() else {
                return Err("@many requires an item expression".to_owned());
            };
            let item = modern_lower_syntax_expr(first)?
                .ok_or_else(|| "@many item lowered to empty syntax".to_owned())?;
            let sep = items.get(1).and_then(|item| {
                let Ok(Some(rule)) = modern_lower_syntax_expr(item) else {
                    return None;
                };
                match rule {
                    SyntaxRule::Literal(text) | SyntaxRule::Tag(text) => {
                        Some(SyntaxRepeatSeparator::Literal(text))
                    }
                    SyntaxRule::Ref { name } => Some(SyntaxRepeatSeparator::RuleRef(name)),
                    _ => None,
                }
            });
            Ok(Some(SyntaxRule::Repeat {
                item: Box::new(item),
                sep,
            }))
        }
        SyntaxExpr::SeparatedBy(items) => {
            let Some(first) = items.first() else {
                return Err("@separated_by requires an item expression".to_owned());
            };
            let item = modern_lower_syntax_expr(first)?
                .ok_or_else(|| "@separated_by item lowered to empty syntax".to_owned())?;
            let sep = items.get(1).and_then(|item| {
                let Ok(Some(rule)) = modern_lower_syntax_expr(item) else {
                    return None;
                };
                match rule {
                    SyntaxRule::Literal(text) | SyntaxRule::Tag(text) => {
                        Some(SyntaxRepeatSeparator::Literal(text))
                    }
                    SyntaxRule::Ref { name } => Some(SyntaxRepeatSeparator::RuleRef(name)),
                    _ => None,
                }
            });
            Ok(Some(SyntaxRule::Repeat {
                item: Box::new(item),
                sep,
            }))
        }
        SyntaxExpr::Indent(items) => modern_lower_syntax_expr(&SyntaxExpr::Other {
            tag: None,
            content: Some(SyntaxPayload::Seq(items.clone())),
        }),
        SyntaxExpr::SoftSpace | SyntaxExpr::Newline => Ok(None),
        SyntaxExpr::Any => Ok(Some(SyntaxRule::Ref {
            name: "Any".to_owned(),
        })),
        SyntaxExpr::Rule => Ok(Some(SyntaxRule::Ref {
            name: "Rule".to_owned(),
        })),
        SyntaxExpr::Other {
            tag: None,
            content: Some(SyntaxPayload::Scalar(text)),
        } => Ok(Some(SyntaxRule::Literal(text.clone()))),
        SyntaxExpr::Other {
            tag: None,
            content: Some(SyntaxPayload::Seq(items)),
        } => {
            let items = items
                .iter()
                .filter_map(|item| modern_lower_syntax_expr(item).transpose())
                .collect::<Result<Vec<_>, _>>()?;
            match items.as_slice() {
                [] => Ok(None),
                [only] => Ok(Some(only.clone())),
                _ => Ok(Some(SyntaxRule::Seq(items))),
            }
        }
        SyntaxExpr::Other {
            tag: None,
            content: Some(SyntaxPayload::Object(object)),
        } => Ok(Some(modern_lower_object_binding(object)?)),
        SyntaxExpr::Other {
            tag: Some(name),
            content: None,
        } => Ok(Some(SyntaxRule::Ref { name: name.clone() })),
        SyntaxExpr::Other {
            tag: Some(name),
            content: Some(_),
        } => Err(format!(
            "unsupported syntax tag @{name} in lowered modern syntax"
        )),
        SyntaxExpr::Other {
            tag: None,
            content: None,
        } => Ok(None),
    }
}

fn modern_field_ty_from_rule(rule: &SyntaxRule) -> Result<SyntaxTypeUse, String> {
    match rule {
        SyntaxRule::Ref { name } | SyntaxRule::Token { name } => {
            Ok(SyntaxTypeUse::Ref { name: name.clone() })
        }
        SyntaxRule::Optional { inner } => Ok(SyntaxTypeUse::Optional(Box::new(
            modern_field_ty_from_rule(inner)?,
        ))),
        SyntaxRule::Repeat { item, .. } => Ok(SyntaxTypeUse::Seq(Box::new(
            modern_field_ty_from_rule(item)?,
        ))),
        SyntaxRule::Seq(items) => {
            let mut tys = items
                .iter()
                .filter_map(|item| match item {
                    SyntaxRule::Literal(_) | SyntaxRule::Tag(_) => None,
                    other => Some(modern_field_ty_from_rule(other)),
                })
                .collect::<Result<Vec<_>, _>>()?;
            match tys.len() {
                1 => Ok(tys.remove(0)),
                _ => Err(format!(
                    "cannot infer field type from syntax sequence {rule:?}"
                )),
            }
        }
        other => Err(format!(
            "cannot infer field type from syntax rule {other:?}"
        )),
    }
}

fn modern_collect_struct_fields(
    rule: &SyntaxRule,
    out: &mut HashMap<String, DocumentedValue<SyntaxTypeUse>>,
    optional: bool,
    repeated: bool,
) -> Result<(), String> {
    match rule {
        SyntaxRule::Seq(items) | SyntaxRule::Choice(items) => {
            for item in items {
                modern_collect_struct_fields(item, out, optional, repeated)?;
            }
        }
        SyntaxRule::Field(named) => {
            let mut ty = modern_field_ty_from_rule(&named.inner)?;
            if repeated {
                ty = SyntaxTypeUse::Seq(Box::new(ty));
            }
            if optional {
                ty = SyntaxTypeUse::Optional(Box::new(ty));
            }
            out.insert(
                named.name.clone(),
                DocumentedValue {
                    value: ty,
                    doc: None,
                },
            );
        }
        SyntaxRule::Optional { inner } => {
            modern_collect_struct_fields(inner, out, true, repeated)?;
        }
        SyntaxRule::Repeat { item, .. } => {
            modern_collect_struct_fields(item, out, optional, true)?;
        }
        SyntaxRule::Semantic { inner, .. } => {
            modern_collect_struct_fields(inner, out, optional, repeated)?;
        }
        SyntaxRule::Variant(_)
        | SyntaxRule::Ref { .. }
        | SyntaxRule::Token { .. }
        | SyntaxRule::Tag(_)
        | SyntaxRule::Literal(_) => {}
    }
    Ok(())
}

fn modern_support_seed(kind: &ModernResolvedRuleKind, name: &str) -> Option<ModernSupportClass> {
    match kind {
        ModernResolvedRuleKind::Regex(pattern) => {
            if name.ends_with("Int") || name.ends_with("Nat") || pattern.contains("[0-9]") {
                Some(ModernSupportClass::Int)
            } else {
                Some(ModernSupportClass::String)
            }
        }
        ModernResolvedRuleKind::Inline {
            syntax: Some(syntax),
            ..
        } => {
            if let SyntaxExpr::Regex(items) = syntax
                && items.len() == 1
                && let SyntaxExpr::Other {
                    tag: None,
                    content: Some(SyntaxPayload::Scalar(pattern)),
                } = &items[0]
            {
                return if name.ends_with("Int")
                    || name.ends_with("Nat")
                    || pattern.contains("[0-9]")
                {
                    Some(ModernSupportClass::Int)
                } else {
                    Some(ModernSupportClass::String)
                };
            }
            let Ok(Some(rule)) = modern_lower_syntax_expr(syntax) else {
                return None;
            };
            match &rule {
                SyntaxRule::Ref { name: inner } | SyntaxRule::Token { name: inner } => {
                    if name.ends_with("Id") && (inner.ends_with("Int") || inner.ends_with("Nat")) {
                        Some(ModernSupportClass::Id)
                    } else {
                        None
                    }
                }
                _ => None,
            }
        }
        ModernResolvedRuleKind::Struct {
            syntax: Some(syntax),
            ..
        } => {
            if let SyntaxExpr::Regex(items) = syntax
                && items.len() == 1
                && let SyntaxExpr::Other {
                    tag: None,
                    content: Some(SyntaxPayload::Scalar(_)),
                } = &items[0]
            {
                return None;
            }
            let Ok(Some(rule)) = modern_lower_syntax_expr(syntax) else {
                return None;
            };
            let mut fields = HashMap::new();
            if modern_collect_struct_fields(&rule, &mut fields, false, false).is_err() {
                return None;
            }
            if fields.len() == 1 && name.ends_with("Id") {
                Some(ModernSupportClass::Id)
            } else {
                None
            }
        }
        _ => None,
    }
}

fn modern_root_name(rules: &IndexMap<String, ModernResolvedRule>) -> Result<String, String> {
    for (name, rule) in rules {
        if name.contains('.') {
            continue;
        }
        match rule.kind {
            ModernResolvedRuleKind::Struct { .. } | ModernResolvedRuleKind::Enum => {
                return Ok(name.clone());
            }
            ModernResolvedRuleKind::Literal(_)
            | ModernResolvedRuleKind::Regex(_)
            | ModernResolvedRuleKind::Inline { .. } => {}
        }
    }
    Err("modern repr has no struct or enum root rule".to_owned())
}

fn modern_collect_refs(rule: &SyntaxRule, out: &mut Vec<String>) {
    match rule {
        SyntaxRule::Seq(items) | SyntaxRule::Choice(items) => {
            for item in items {
                modern_collect_refs(item, out);
            }
        }
        SyntaxRule::Semantic { inner, .. } | SyntaxRule::Optional { inner } => {
            modern_collect_refs(inner, out)
        }
        SyntaxRule::Field(named) | SyntaxRule::Variant(named) => {
            modern_collect_refs(&named.inner, out);
        }
        SyntaxRule::Repeat { item, .. } => modern_collect_refs(item, out),
        SyntaxRule::Ref { name } | SyntaxRule::Token { name } => out.push(name.clone()),
        SyntaxRule::Tag(_) | SyntaxRule::Literal(_) => {}
    }
}

fn modern_normalize_repr(repr: &ModernReprBody) -> Result<NormalizedRepr, String> {
    let mut templates = IndexMap::new();
    for (name, decl) in &repr.templates {
        templates.insert(
            documented_name(name).to_owned(),
            modern_extract_template(decl)?.clone(),
        );
    }

    let mut resolved_rules = IndexMap::new();
    for (name, decl) in &repr.rules {
        resolved_rules.insert(
            documented_name(name).to_owned(),
            ModernResolvedRule {
                doc: documented_doc(name).map(|lines| lines.to_vec()),
                kind: modern_expand_rule(decl, &templates)
                    .map_err(|err| format!("rule {} ({decl:?}): {err}", documented_name(name)))?,
            },
        );
    }

    let root = modern_root_name(&resolved_rules)?;

    let mut token_specs = HashMap::new();
    let mut syntax_rules = HashMap::new();
    let mut support_scalars = HashMap::new();

    for (name, rule) in &resolved_rules {
        match &rule.kind {
            ModernResolvedRuleKind::Regex(pattern) => {
                token_specs.insert(
                    name.clone(),
                    NormalizedTokenSpec {
                        regex: pattern.clone(),
                    },
                );
                syntax_rules.insert(name.clone(), SyntaxRule::Token { name: name.clone() });
                if let Some(kind) = modern_support_seed(&rule.kind, name) {
                    support_scalars.insert(name.clone(), kind);
                }
            }
            ModernResolvedRuleKind::Literal(text) => {
                syntax_rules.insert(name.clone(), SyntaxRule::Literal(text.clone()));
            }
            ModernResolvedRuleKind::Inline { syntax, .. }
            | ModernResolvedRuleKind::Struct { syntax, .. } => {
                if let Some(syntax) = syntax {
                    if let SyntaxExpr::Regex(items) = syntax
                        && items.len() == 1
                        && let SyntaxExpr::Other {
                            tag: None,
                            content: Some(SyntaxPayload::Scalar(pattern)),
                        } = &items[0]
                    {
                        token_specs.insert(
                            name.clone(),
                            NormalizedTokenSpec {
                                regex: pattern.clone(),
                            },
                        );
                        syntax_rules.insert(name.clone(), SyntaxRule::Token { name: name.clone() });
                        if let Some(kind) = modern_support_seed(&rule.kind, name) {
                            support_scalars.insert(name.clone(), kind);
                        }
                        continue;
                    }
                    if let Some(rule_syntax) = modern_lower_syntax_expr(syntax)? {
                        syntax_rules.insert(name.clone(), rule_syntax);
                    }
                }
                if let Some(kind) = modern_support_seed(&rule.kind, name) {
                    support_scalars.insert(name.clone(), kind);
                }
            }
            ModernResolvedRuleKind::Enum => {}
        }
    }

    let mut support = HashMap::new();
    for (name, class) in &support_scalars {
        let decl = match class {
            ModernSupportClass::String => NormalizedSupportDecl::String,
            ModernSupportClass::Int => NormalizedSupportDecl::Int,
            ModernSupportClass::Id => NormalizedSupportDecl::Id,
        };
        let doc = resolved_rules.get(name).and_then(|rule| rule.doc.clone());
        support.insert(name.clone(), DocumentedValue { value: decl, doc });
    }

    let mut reachable = vec![root.clone()];
    let mut seen = HashMap::<String, ()>::new();
    while let Some(name) = reachable.pop() {
        if seen.insert(name.clone(), ()).is_some() {
            continue;
        }
        if let Some(rule) = syntax_rules.get(&name) {
            let mut refs = Vec::new();
            modern_collect_refs(rule, &mut refs);
            for target in refs {
                if resolved_rules.contains_key(&target) && !seen.contains_key(&target) {
                    reachable.push(target);
                }
            }
        }
        if matches!(
            resolved_rules.get(&name).map(|rule| &rule.kind),
            Some(ModernResolvedRuleKind::Enum)
        ) {
            let prefix = format!("{name}.");
            for (variant_name, _) in &resolved_rules {
                if !variant_name.starts_with(&prefix) {
                    continue;
                }
                if let Some(rule) = syntax_rules.get(variant_name) {
                    let mut refs = Vec::new();
                    modern_collect_refs(rule, &mut refs);
                    for target in refs {
                        if resolved_rules.contains_key(&target) && !seen.contains_key(&target) {
                            reachable.push(target);
                        }
                    }
                }
            }
        }
    }

    let mut nodes = HashMap::new();
    for (name, rule) in &resolved_rules {
        if name.contains('.') || !seen.contains_key(name) || support.contains_key(name) {
            continue;
        }
        match &rule.kind {
            ModernResolvedRuleKind::Struct { .. } => {
                let mut fields = HashMap::new();
                fields.insert(
                    "prov".to_owned(),
                    DocumentedValue {
                        value: SyntaxTypeUse::Ref {
                            name: "Prov".to_owned(),
                        },
                        doc: None,
                    },
                );
                if let Some(rule_syntax) = syntax_rules.get(name) {
                    modern_collect_struct_fields(rule_syntax, &mut fields, false, false)?;
                }
                nodes.insert(
                    name.clone(),
                    DocumentedValue {
                        value: NormalizedNodeDecl::Record {
                            kind: NormalizedNodeKind::Struct,
                            fields,
                        },
                        doc: rule.doc.clone(),
                    },
                );
            }
            ModernResolvedRuleKind::Enum => {
                let mut variants = IndexMap::new();
                let mut choice_items = Vec::new();
                let prefix = format!("{name}.");
                for (variant_name, variant_rule) in &resolved_rules {
                    let Some(short) = variant_name.strip_prefix(&prefix) else {
                        continue;
                    };
                    let ModernResolvedRuleKind::Struct { .. } = &variant_rule.kind else {
                        continue;
                    };
                    let variant_syntax = syntax_rules.get(variant_name).ok_or_else(|| {
                        format!("enum variant rule {variant_name} is missing syntax")
                    })?;
                    choice_items.push(SyntaxRule::Variant(SyntaxRuleNamed {
                        name: short.to_owned(),
                        inner: Box::new(variant_syntax.clone()),
                    }));
                    let mut fields = HashMap::new();
                    fields.insert(
                        "prov".to_owned(),
                        DocumentedValue {
                            value: SyntaxTypeUse::Ref {
                                name: "Prov".to_owned(),
                            },
                            doc: None,
                        },
                    );
                    modern_collect_struct_fields(variant_syntax, &mut fields, false, false)?;
                    variants.insert(
                        short.to_owned(),
                        DocumentedValue {
                            value: NormalizedNodeDecl::Record {
                                kind: NormalizedNodeKind::Struct,
                                fields,
                            },
                            doc: variant_rule.doc.clone(),
                        },
                    );
                }
                syntax_rules.insert(name.clone(), SyntaxRule::Choice(choice_items));
                nodes.insert(
                    name.clone(),
                    DocumentedValue {
                        value: NormalizedNodeDecl::Enum(variants),
                        doc: rule.doc.clone(),
                    },
                );
            }
            ModernResolvedRuleKind::Literal(_)
            | ModernResolvedRuleKind::Regex(_)
            | ModernResolvedRuleKind::Inline { .. } => {}
        }
    }

    syntax_rules.retain(|name, _| !name.contains('.'));

    Ok(NormalizedRepr {
        doc: None,
        name: repr.name.clone(),
        file_ext: repr.file_ext.clone(),
        contract: NormalizedContract {
            purpose: repr.description.clone(),
            canonical_identities: Vec::new(),
            round_trip: "canonical".to_owned(),
            provenance: "required".to_owned(),
        },
        syntax: NormalizedSyntax {
            root,
            token_specs,
            rules: syntax_rules,
            canonical_print: HashMap::new(),
            semantic_tokens: HashMap::new(),
        },
        common: HashMap::from([(
            "provenance".to_owned(),
            SyntaxTypeUse::Ref {
                name: "Prov".to_owned(),
            },
        )]),
        support,
        nodes,
    })
}

pub(crate) fn normalize_repr(repr: &ModernReprBody) -> Result<NormalizedRepr, String> {
    modern_normalize_repr(repr)
}

pub(crate) fn render_default_value(ty: &SyntaxTypeUse, provenance_tag: &str) -> Option<String> {
    match ty {
        SyntaxTypeUse::Optional(_) => Some("None".to_owned()),
        SyntaxTypeUse::Seq(_) => Some("Vec::new()".to_owned()),
        SyntaxTypeUse::Arena { key: Some(_), .. } => {
            Some("super::super::Order::default()".to_owned())
        }
        SyntaxTypeUse::Arena { key: None, .. } => Some("super::super::Arena::default()".to_owned()),
        SyntaxTypeUse::Pool { .. } => Some("super::super::Pool::default()".to_owned()),
        SyntaxTypeUse::Order(_) => Some("super::super::Order::default()".to_owned()),
        SyntaxTypeUse::RefTo { .. } => None,
        SyntaxTypeUse::Ref { name } if name == provenance_tag => {
            Some(format!("{provenance_tag}::default()"))
        }
        SyntaxTypeUse::Ref { .. } => None,
    }
}

pub(crate) fn direct_ref_name(ty: &SyntaxTypeUse) -> Option<&str> {
    match ty {
        SyntaxTypeUse::Ref { name } => Some(name.as_str()),
        SyntaxTypeUse::RefTo { id, .. } => direct_ref_name(id),
        _ => None,
    }
}

fn common_alias_target<'a>(repr: &'a NormalizedRepr, role: &str) -> Option<&'a str> {
    match repr.common.get(role) {
        Some(SyntaxTypeUse::Ref { name }) => Some(name.as_str()),
        _ => None,
    }
}

pub(crate) fn classify_ref_type(repr: &NormalizedRepr, name: &str) -> NormalizedRefKind {
    if common_alias_target(repr, "provenance") == Some(name) {
        return NormalizedRefKind::Provenance;
    }

    if common_alias_target(repr, "symbol") == Some(name) {
        return NormalizedRefKind::StringScalar;
    }

    if common_alias_target(repr, "docs") == Some(name) {
        return NormalizedRefKind::StringSeq;
    }

    match repr.support.get(name).map(|decl| &decl.value) {
        Some(NormalizedSupportDecl::String) => NormalizedRefKind::StringScalar,
        Some(NormalizedSupportDecl::Int) => NormalizedRefKind::IntScalar,
        Some(NormalizedSupportDecl::Id) => NormalizedRefKind::Id,
        None => NormalizedRefKind::Unknown,
    }
}

pub(crate) fn is_string_scalar_type(repr: &NormalizedRepr, name: &str) -> bool {
    classify_ref_type(repr, name) == NormalizedRefKind::StringScalar
}

pub(crate) fn is_int_scalar_type(repr: &NormalizedRepr, name: &str) -> bool {
    classify_ref_type(repr, name) == NormalizedRefKind::IntScalar
}

pub(crate) fn is_id_type(repr: &NormalizedRepr, name: &str) -> bool {
    classify_ref_type(repr, name) == NormalizedRefKind::Id
}

pub(crate) fn is_docs_type(repr: &NormalizedRepr, name: &str) -> bool {
    common_alias_target(repr, "docs") == Some(name)
}
