use std::collections::HashMap;

use crate::schema::{
    NodeDecl, NodeFields, ReprBody, RuleExpr, SupportDecl, SupportVariantDecl, TypeUse,
    documented_doc, documented_name, rule_literal_text, rule_named_parts,
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

#[derive(Debug, Clone)]
pub(crate) struct NormalizedContract {
    pub(crate) purpose: String,
    pub(crate) canonical_identities: Vec<String>,
    pub(crate) round_trip: String,
    pub(crate) provenance: String,
}

#[derive(Debug, Clone)]
pub(crate) struct NormalizedSyntax {
    pub(crate) token_kinds: HashMap<String, String>,
    pub(crate) rules: HashMap<String, SyntaxRule>,
    pub(crate) canonical_print: HashMap<String, String>,
}

#[derive(Debug, Clone)]
pub(crate) struct DocumentedValue<T> {
    pub(crate) value: T,
    pub(crate) doc: Option<Vec<String>>,
}

#[derive(Debug, Clone)]
pub(crate) enum NormalizedSupportDecl {
    String,
    StringSeq,
    Unit,
    Enum(Vec<DocumentedValue<String>>),
}

#[derive(Debug, Clone)]
pub(crate) enum NormalizedNodeDecl {
    Node(HashMap<String, DocumentedValue<SyntaxTypeUse>>),
    Enum(HashMap<String, DocumentedValue<NormalizedNodeDecl>>),
    Struct(HashMap<String, DocumentedValue<SyntaxTypeUse>>),
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

fn normalize_support_decl(decl: &SupportDecl) -> Result<NormalizedSupportDecl, String> {
    match decl {
        SupportDecl::String => Ok(NormalizedSupportDecl::String),
        SupportDecl::StringSeq => Ok(NormalizedSupportDecl::StringSeq),
        SupportDecl::Unit => Ok(NormalizedSupportDecl::Unit),
        SupportDecl::Enum(variants) if !variants.variants.is_empty() => {
            let mut out = Vec::new();
            for (name, decl) in &variants.variants {
                match decl {
                    SupportVariantDecl::Unit => {
                        out.push(DocumentedValue {
                            value: documented_name(name).to_owned(),
                            doc: documented_doc(name).map(|lines| lines.to_vec()),
                        });
                    }
                }
            }
            Ok(NormalizedSupportDecl::Enum(out))
        }
        SupportDecl::Enum(_) => Err("support enum must have at least one variant".to_owned()),
    }
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
) -> Result<HashMap<String, DocumentedValue<SyntaxTypeUse>>, String> {
    fields
        .fields
        .iter()
        .map(|(name, ty)| {
            Ok((
                documented_name(name).to_owned(),
                DocumentedValue {
                    value: normalize_type_use(ty)?,
                    doc: documented_doc(name).map(|lines| lines.to_vec()),
                },
            ))
        })
        .collect()
}

fn normalize_node_decl(decl: &NodeDecl) -> Result<NormalizedNodeDecl, String> {
    match decl {
        NodeDecl::Node(fields) => Ok(NormalizedNodeDecl::Node(normalize_node_fields(fields)?)),
        NodeDecl::Struct(fields) => Ok(NormalizedNodeDecl::Struct(normalize_node_fields(fields)?)),
        NodeDecl::Enum(variants) => {
            let mut out = HashMap::new();
            for (name, variant) in &variants.variants {
                out.insert(
                    documented_name(name).to_owned(),
                    DocumentedValue {
                        value: normalize_node_decl(variant)?,
                        doc: documented_doc(name).map(|lines| lines.to_vec()),
                    },
                );
            }
            Ok(NormalizedNodeDecl::Enum(out))
        }
        NodeDecl::Other { tag, .. } => Err(format!(
            "unsupported node declaration tag {:?}",
            tag.as_deref().unwrap_or("<unknown>")
        )),
    }
}

pub(crate) fn normalize_repr(repr: &ReprBody) -> Result<NormalizedRepr, String> {
    let common = repr
        .common
        .as_ref()
        .ok_or_else(|| "repr.common missing after validation".to_owned())?
        .iter()
        .map(|(name, ty)| Ok((name.clone(), normalize_type_use(ty)?)))
        .collect::<Result<HashMap<_, _>, String>>()?;

    let nodes = repr
        .nodes
        .as_ref()
        .ok_or_else(|| "repr.nodes missing after validation".to_owned())?
        .iter()
        .map(|(name, decl)| {
            Ok((
                documented_name(name).to_owned(),
                DocumentedValue {
                    value: normalize_node_decl(decl)?,
                    doc: documented_doc(name).map(|lines| lines.to_vec()),
                },
            ))
        })
        .collect::<Result<HashMap<_, _>, String>>()?;

    let support = repr
        .support
        .clone()
        .unwrap_or_default()
        .into_iter()
        .map(|(name, decl)| {
            Ok((
                documented_name(&name).to_owned(),
                DocumentedValue {
                    value: normalize_support_decl(&decl)?,
                    doc: documented_doc(&name).map(|lines| lines.to_vec()),
                },
            ))
        })
        .collect::<Result<HashMap<_, _>, String>>()?;

    let rules = repr
        .syntax
        .rules
        .iter()
        .map(|(name, rule)| Ok((name.clone(), normalize_rule(rule)?)))
        .collect::<Result<HashMap<_, _>, String>>()?;

    let token_kinds = repr
        .syntax
        .tokens
        .iter()
        .map(|(name, token)| {
            let kind = match token {
                crate::schema::TokenExpr::Regex(_) => "regex".to_owned(),
                crate::schema::TokenExpr::Other { name, .. } => {
                    name.clone().unwrap_or_else(|| "<unknown>".to_owned())
                }
            };
            Ok((name.clone(), kind))
        })
        .collect::<Result<HashMap<_, _>, String>>()?;

    Ok(NormalizedRepr {
        doc: None,
        name: repr.name.clone(),
        file_ext: repr.file_ext.clone(),
        contract: NormalizedContract {
            purpose: repr.contract.purpose.clone(),
            canonical_identities: repr.contract.canonical_identities.clone(),
            round_trip: repr.contract.round_trip.clone(),
            provenance: repr.contract.provenance.clone(),
        },
        syntax: NormalizedSyntax {
            token_kinds,
            rules,
            canonical_print: repr.syntax.canonical_print.clone(),
        },
        common,
        support,
        nodes,
    })
}

pub(crate) fn with_module_doc(
    mut repr: NormalizedRepr,
    doc: Option<Vec<String>>,
) -> NormalizedRepr {
    repr.doc = doc;
    repr
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
