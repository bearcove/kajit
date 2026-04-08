use std::collections::HashMap;

use crate::schema::{NodeFields, RuleExpr, TypeUse, rule_literal_text, rule_named_parts};

#[derive(Debug, Clone)]
pub(crate) struct SyntaxRuleNamed {
    pub(crate) name: String,
    pub(crate) inner: Box<SyntaxRule>,
}

#[derive(Debug, Clone)]
pub(crate) enum SyntaxRule {
    Seq(Vec<SyntaxRule>),
    Choice(Vec<SyntaxRule>),
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
        sep: Option<String>,
    },
    Literal(String),
}

#[derive(Debug, Clone)]
pub(crate) enum SyntaxTypeUse {
    Optional(Box<SyntaxTypeUse>),
    Seq(Box<SyntaxTypeUse>),
    Ref { name: String },
}

pub(crate) fn normalize_type_use(ty: &TypeUse) -> Result<SyntaxTypeUse, String> {
    match ty {
        TypeUse::Optional(items) if items.len() == 1 => Ok(SyntaxTypeUse::Optional(Box::new(
            normalize_type_use(&items[0])?,
        ))),
        TypeUse::Seq(items) if items.len() == 1 => {
            Ok(SyntaxTypeUse::Seq(Box::new(normalize_type_use(&items[0])?)))
        }
        TypeUse::Ref { name: Some(name) } => Ok(SyntaxTypeUse::Ref { name: name.clone() }),
        TypeUse::Ref { name: None } => Err("type reference missing tag name".to_owned()),
        TypeUse::Optional(_) => Err("optional type must have exactly one item".to_owned()),
        TypeUse::Seq(_) => Err("seq type must have exactly one item".to_owned()),
    }
}

pub(crate) fn normalize_rule(rule: &RuleExpr) -> Result<SyntaxRule, String> {
    match rule {
        RuleExpr::Seq(items) => Ok(SyntaxRule::Seq(
            items
                .iter()
                .map(normalize_rule)
                .collect::<Result<Vec<_>, _>>()?,
        )),
        RuleExpr::Choice(items) => Ok(SyntaxRule::Choice(
            items
                .iter()
                .map(normalize_rule)
                .collect::<Result<Vec<_>, _>>()?,
        )),
        RuleExpr::Field(named) => {
            let (name, inner) = rule_named_parts(named);
            Ok(SyntaxRule::Field(SyntaxRuleNamed {
                name: name.to_owned(),
                inner: Box::new(normalize_rule(inner)?),
            }))
        }
        RuleExpr::Variant(named) => {
            let (name, inner) = rule_named_parts(named);
            Ok(SyntaxRule::Variant(SyntaxRuleNamed {
                name: name.to_owned(),
                inner: Box::new(normalize_rule(inner)?),
            }))
        }
        RuleExpr::Ref(names) if names.len() == 1 => Ok(SyntaxRule::Ref {
            name: names[0].clone(),
        }),
        RuleExpr::Token(names) if names.len() == 1 => Ok(SyntaxRule::Token {
            name: names[0].clone(),
        }),
        RuleExpr::Optional(items) if items.len() == 1 => Ok(SyntaxRule::Optional {
            inner: Box::new(normalize_rule(&items[0])?),
        }),
        RuleExpr::Repeat(items) if !items.is_empty() => {
            let sep = if items.len() >= 2 {
                rule_literal_text(&items[1]).map(str::to_owned)
            } else {
                None
            };
            Ok(SyntaxRule::Repeat {
                item: Box::new(normalize_rule(&items[0])?),
                sep,
            })
        }
        RuleExpr::Literal(Some(text)) => Ok(SyntaxRule::Literal(text.clone())),
        RuleExpr::Ref(_) => Err("ref rule must have exactly one target".to_owned()),
        RuleExpr::Token(_) => Err("token rule must have exactly one token name".to_owned()),
        RuleExpr::Optional(_) => Err("optional rule must have exactly one item".to_owned()),
        RuleExpr::Repeat(_) => Err("repeat rule must have at least one item".to_owned()),
        RuleExpr::Literal(None) => Err("literal rule missing text".to_owned()),
    }
}

pub(crate) fn normalize_node_fields(
    fields: &NodeFields,
) -> Result<HashMap<String, SyntaxTypeUse>, String> {
    fields
        .fields
        .iter()
        .map(|(name, ty)| Ok((name.clone(), normalize_type_use(ty)?)))
        .collect()
}

pub(crate) fn syntax_type_name(ty: &SyntaxTypeUse) -> Option<&str> {
    match ty {
        SyntaxTypeUse::Ref { name } => Some(name.as_str()),
        _ => None,
    }
}

pub(crate) fn render_default_value(ty: &SyntaxTypeUse, provenance_tag: &str) -> String {
    match ty {
        SyntaxTypeUse::Optional(_) => "None".to_owned(),
        SyntaxTypeUse::Seq(_) => "Vec::new()".to_owned(),
        SyntaxTypeUse::Ref { name } if name == provenance_tag => {
            format!("{provenance_tag}::default()")
        }
        SyntaxTypeUse::Ref { name } => format!("{name}::default()"),
    }
}
