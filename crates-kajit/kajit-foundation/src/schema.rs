use std::collections::HashMap;
use std::fs;
use std::path::Path;

use facet::Facet;
use facet_styx::{Documented, RenderError};
use indexmap::IndexMap;

#[derive(Facet, Debug, Clone)]
#[facet(metadata_container)]
pub(crate) struct WithDoc<T> {
    pub(crate) value: T,

    #[facet(metadata = "doc")]
    pub(crate) doc: Option<Vec<String>>,
}

#[derive(Facet, Debug, Clone)]
#[allow(dead_code)]
pub(crate) struct PilotSchemaDocument {
    pub(crate) meta: PilotMeta,
    pub(crate) repr: Documented<ReprDecl>,
}

#[derive(Debug, Clone)]
pub(crate) struct LoadedRepr {
    pub(crate) doc: Option<Vec<String>>,
    pub(crate) body: LoadedReprBody,
}

#[derive(Debug, Clone)]
pub(crate) enum LoadedReprBody {
    Legacy(ReprBody),
    Modern(ModernReprBody),
}

#[derive(Facet, Debug, Clone)]
#[allow(dead_code)]
pub(crate) struct PilotMeta {
    pub(crate) id: String,
    pub(crate) version: u64,
    pub(crate) description: String,
}

#[derive(Facet, Debug, Clone)]
#[allow(dead_code)]
#[facet(rename_all = "snake_case")]
#[repr(u8)]
pub(crate) enum ReprDecl {
    Module(ReprBody),
}

#[derive(Facet, Debug, Clone)]
#[allow(dead_code)]
pub(crate) struct ReprBody {
    pub(crate) name: String,
    pub(crate) file_ext: String,
    pub(crate) contract: ReprContract,
    pub(crate) syntax: ReprSyntax,
    pub(crate) common: Option<HashMap<String, TypeUse>>,
    pub(crate) support: Option<HashMap<Documented<String>, SupportDecl>>,
    pub(crate) nodes: Option<HashMap<Documented<String>, NodeDecl>>,
}

#[derive(Facet, Debug, Clone)]
#[allow(dead_code)]
pub(crate) struct ModernPilotSchemaDocument {
    pub(crate) name: String,
    pub(crate) file_ext: String,
    pub(crate) description: String,
    pub(crate) rules: ModernRulesDecl,
}

#[derive(Facet, Debug, Clone)]
#[allow(dead_code)]
pub(crate) struct ModernRulesDecl {
    pub(crate) templates: Option<IndexMap<Documented<String>, TemplateDecl>>,

    #[facet(flatten)]
    pub(crate) rules: IndexMap<Documented<String>, ModernRuleDecl>,
}

#[derive(Facet, Debug, Clone)]
#[allow(dead_code)]
pub(crate) struct TemplateDecl {
    pub(crate) syntax: SyntaxExpr,
    pub(crate) highlight: Option<String>,
}

#[derive(Facet, Debug, Clone)]
#[allow(dead_code)]
pub(crate) struct TemplateSyntaxDecl {
    pub(crate) params: Vec<SyntaxExpr>,
    pub(crate) body: ModernStructDecl,
}

#[derive(Facet, Debug, Clone)]
#[allow(dead_code)]
pub(crate) struct ModernStructDecl {
    pub(crate) syntax: Option<SyntaxExpr>,
    pub(crate) highlight: Option<String>,
}

#[derive(Facet, Debug, Clone)]
#[allow(dead_code)]
#[facet(rename_all = "lowercase")]
#[repr(u8)]
pub(crate) enum ModernRuleDecl {
    #[facet(untagged)]
    Literal(String),
    #[facet(untagged)]
    InlineStruct(ModernStructDecl),
    Struct(ModernStructDecl),
    Enum,
    #[facet(other)]
    UserTag {
        #[facet(tag)]
        tag: String,
        #[facet(content)]
        content: Option<SyntaxPayload>,
    },
}

#[derive(Facet, Debug, Clone)]
#[allow(dead_code)]
pub(crate) struct ModernReprBody {
    pub(crate) name: String,
    pub(crate) file_ext: String,
    pub(crate) description: String,
    pub(crate) templates: IndexMap<Documented<String>, TemplateDecl>,
    pub(crate) rules: IndexMap<Documented<String>, ModernRuleDecl>,
}

#[derive(Facet, Debug, Clone)]
#[allow(dead_code)]
#[facet(rename_all = "snake_case")]
#[repr(u8)]
pub(crate) enum SyntaxExpr {
    Template(Box<TemplateSyntaxDecl>),
    Regex(Vec<SyntaxExpr>),
    BraceBlock(Vec<SyntaxExpr>),
    ParenList(Vec<SyntaxExpr>),
    BracketList(Vec<SyntaxExpr>),
    Paren(Vec<SyntaxExpr>),
    Bracket(Vec<SyntaxExpr>),
    Optional(Vec<SyntaxExpr>),
    Many(Vec<SyntaxExpr>),
    SeparatedBy(Vec<SyntaxExpr>),
    Indent(Vec<SyntaxExpr>),
    SoftSpace,
    Newline,
    #[facet(rename = "Any")]
    Any,
    #[facet(rename = "Rule")]
    Rule,
    #[facet(other)]
    Other {
        #[facet(tag)]
        tag: Option<String>,
        #[facet(content)]
        content: Option<SyntaxPayload>,
    },
}

#[derive(Facet, Debug, Clone)]
#[allow(dead_code)]
pub(crate) struct SyntaxObject {
    #[facet(flatten)]
    pub(crate) fields: IndexMap<Documented<String>, SyntaxExpr>,
}

#[derive(Facet, Debug, Clone)]
#[allow(dead_code)]
#[repr(u8)]
#[facet(untagged)]
pub(crate) enum SyntaxPayload {
    Scalar(String),
    Seq(Vec<SyntaxExpr>),
    Object(SyntaxObject),
}

#[derive(Facet, Debug, Clone)]
#[allow(dead_code)]
pub(crate) struct ReprContract {
    pub(crate) purpose: String,
    pub(crate) canonical_identities: Vec<String>,
    pub(crate) round_trip: String,
    pub(crate) provenance: String,
}

#[derive(Facet, Debug, Clone)]
#[allow(dead_code)]
pub(crate) struct ReprSyntax {
    pub(crate) root: String,
    pub(crate) tokens: HashMap<String, TokenExpr>,
    pub(crate) rules: HashMap<String, RuleExpr>,
    pub(crate) canonical_print: HashMap<String, String>,
}

#[derive(Facet, Debug, Clone)]
#[allow(dead_code)]
#[facet(rename_all = "lowercase")]
#[repr(u8)]
pub(crate) enum TokenExpr {
    Regex(Vec<String>),
    #[facet(other)]
    Other {
        #[facet(tag)]
        name: Option<String>,
        #[facet(content)]
        content: Option<Vec<String>>,
    },
}

#[derive(Facet, Debug, Clone)]
#[allow(dead_code)]
#[facet(rename_all = "lowercase")]
#[repr(u8)]
pub(crate) enum RuleExpr {
    Seq(Vec<RuleExpr>),
    Choice(Vec<RuleExpr>),
    Semantic(Vec<RuleExpr>),
    Tag(Vec<String>),
    Field(RuleNamed),
    Variant(RuleNamed),
    Ref(Vec<String>),
    Token(Vec<String>),
    Optional(Vec<RuleExpr>),
    Repeat(Vec<RuleExpr>),
    #[facet(other)]
    Literal(Option<String>),
}

#[derive(Facet, Debug, Clone)]
#[allow(dead_code)]
#[repr(transparent)]
pub(crate) struct RuleNamed(pub(crate) (String, Box<RuleExpr>));

#[derive(Facet, Debug, Clone)]
#[allow(dead_code)]
#[facet(rename_all = "snake_case")]
#[repr(u8)]
pub(crate) enum TypeUse {
    Optional(Vec<TypeUse>),
    Seq(Vec<TypeUse>),
    Arena(Vec<TypeUse>),
    Pool(Vec<TypeUse>),
    Order(Vec<TypeUse>),
    Key(Vec<TypeUse>),
    RefTo(Vec<TypeUse>),
    #[facet(other)]
    Ref {
        #[facet(tag)]
        name: Option<String>,
    },
}

#[derive(Facet, Debug, Clone)]
#[allow(dead_code)]
#[facet(rename_all = "lowercase")]
#[repr(u8)]
pub(crate) enum SupportDecl {
    String,
    Int,
    Id,
    StringSeq,
    Unit,
    Struct(NodeFields),
    Enum(SupportVariants),
}

#[derive(Facet, Debug, Clone)]
#[allow(dead_code)]
#[facet(rename_all = "lowercase")]
#[repr(u8)]
pub(crate) enum SupportVariantDecl {
    Unit,
}

#[derive(Facet, Debug, Clone)]
#[allow(dead_code)]
pub(crate) struct SupportVariants {
    #[facet(flatten)]
    pub(crate) variants: HashMap<Documented<String>, SupportVariantDecl>,
}

#[derive(Facet, Debug, Clone)]
#[allow(dead_code)]
#[facet(rename_all = "lowercase")]
#[repr(u8)]
pub(crate) enum NodeDecl {
    Node(NodeFields),
    Enum(NodeVariants),
    Struct(NodeFields),
    Entity(NodeFields),
    Slot(NodeFields),
    #[facet(other)]
    Other {
        #[facet(tag)]
        tag: Option<String>,
        #[facet(content)]
        content: Option<Vec<TypeUse>>,
    },
}

#[derive(Facet, Debug, Clone)]
#[allow(dead_code)]
pub(crate) struct NodeFields {
    #[facet(flatten)]
    pub(crate) fields: HashMap<Documented<String>, TypeUse>,
}

#[derive(Facet, Debug, Clone)]
#[allow(dead_code)]
pub(crate) struct NodeVariants {
    #[facet(flatten)]
    pub(crate) variants: HashMap<Documented<String>, NodeDecl>,
}

pub(crate) fn load_pilot_schema(schema_path: &Path) -> Result<LoadedRepr, String> {
    let schema_source = fs::read_to_string(schema_path)
        .map_err(|e| format!("failed to read {}: {e}", schema_path.display()))?;
    match facet_styx::from_str::<PilotSchemaDocument>(&schema_source) {
        Ok(schema) => validate_repr_schema(&schema, schema_path),
        Err(legacy_err) => {
            match facet_styx::from_str::<ModernPilotSchemaDocument>(&schema_source) {
                Ok(schema) => validate_modern_repr_schema(&schema, schema_path),
                Err(modern_err) => Err(format!(
                    "failed to parse {} as Styx\nlegacy schema:\n{}\n\nmodern schema:\n{}",
                    schema_path.display(),
                    legacy_err.render(&schema_path.display().to_string(), &schema_source),
                    modern_err.render(&schema_path.display().to_string(), &schema_source),
                )),
            }
        }
    }
}

fn validate_repr_schema(schema: &PilotSchemaDocument, path: &Path) -> Result<LoadedRepr, String> {
    if schema.meta.version == 0 {
        return Err(format!(
            "expected {} meta.version to be non-zero",
            path.display()
        ));
    }

    if schema.meta.description.trim().is_empty() {
        return Err(format!(
            "expected {} meta.description to be non-empty",
            path.display()
        ));
    }

    let ReprDecl::Module(repr) = &schema.repr.value;

    if repr.name.trim().is_empty() {
        return Err(format!(
            "expected {} repr.name to be non-empty",
            path.display()
        ));
    }

    if !repr.file_ext.starts_with('.') || repr.file_ext.len() < 2 {
        return Err(format!(
            "expected {} file_ext to start with '.', got {:?}",
            path.display(),
            repr.file_ext
        ));
    }

    if repr.contract.purpose.trim().is_empty() {
        return Err(format!(
            "expected {} contract.purpose to be non-empty",
            path.display()
        ));
    }

    if repr.contract.round_trip != "canonical-print" {
        return Err(format!(
            "expected {} round_trip to be canonical-print, got {:?}",
            path.display(),
            repr.contract.round_trip
        ));
    }

    if repr.contract.provenance != "required" {
        return Err(format!(
            "expected {} provenance to be required, got {:?}",
            path.display(),
            repr.contract.provenance
        ));
    }

    if repr.contract.canonical_identities.is_empty() {
        return Err(format!(
            "expected {} canonical_identities to be non-empty",
            path.display()
        ));
    }

    if repr.common.is_none() {
        return Err(format!(
            "expected {} repr.common to be present",
            path.display()
        ));
    }

    if repr.nodes.is_none() {
        return Err(format!(
            "expected {} repr.nodes to be present",
            path.display()
        ));
    }

    if repr.syntax.root.trim().is_empty() {
        return Err(format!(
            "expected {} syntax.root to be non-empty",
            path.display()
        ));
    }

    if !repr.syntax.rules.contains_key(&repr.syntax.root) {
        return Err(format!(
            "expected {} syntax.root {:?} to name a declared syntax rule",
            path.display(),
            repr.syntax.root
        ));
    }

    if !type_decl_exists(repr, &repr.syntax.root) {
        return Err(format!(
            "expected {} syntax.root {:?} to name a declared type",
            path.display(),
            repr.syntax.root
        ));
    }

    for (token_name, token_spec) in &repr.syntax.tokens {
        match token_spec {
            TokenExpr::Regex(patterns)
                if !patterns.is_empty() && !patterns[0].trim().is_empty() => {}
            TokenExpr::Regex(_) => {
                return Err(format!(
                    "expected {} syntax.tokens.{token_name} regex payload to be non-empty",
                    path.display()
                ));
            }
            TokenExpr::Other { name, .. } => {
                return Err(format!(
                    "expected {} syntax.tokens.{token_name} to be @regex(...), got {:?}",
                    path.display(),
                    name
                ));
            }
        }
    }

    for (rule_name, rule) in &repr.syntax.rules {
        let mut refs = Vec::new();
        collect_rule_refs(rule, &mut refs);
        for target in refs {
            if !repr.syntax.rules.contains_key(target) {
                return Err(format!(
                    "expected {} syntax.rules.{rule_name} ref target {:?} to exist",
                    path.display(),
                    target
                ));
            }
        }

        let mut tokens = Vec::new();
        collect_rule_tokens(rule, &mut tokens);
        for token in tokens {
            if !repr.syntax.tokens.contains_key(token) {
                return Err(format!(
                    "expected {} syntax.rules.{rule_name} token {:?} to exist",
                    path.display(),
                    token
                ));
            }
        }
    }

    for (print_name, template) in &repr.syntax.canonical_print {
        if template.trim().is_empty() {
            return Err(format!(
                "expected {} canonical_print.{print_name} to be non-empty",
                path.display()
            ));
        }

        if !canonical_print_target_exists(repr, print_name) {
            return Err(format!(
                "expected {} canonical_print.{print_name} to target a declared node or variant",
                path.display()
            ));
        }
    }

    Ok(LoadedRepr {
        doc: schema.repr.doc.clone(),
        body: LoadedReprBody::Legacy(ReprBody {
            name: repr.name.clone(),
            file_ext: repr.file_ext.clone(),
            contract: ReprContract {
                purpose: repr.contract.purpose.clone(),
                canonical_identities: repr.contract.canonical_identities.clone(),
                round_trip: repr.contract.round_trip.clone(),
                provenance: repr.contract.provenance.clone(),
            },
            syntax: ReprSyntax {
                root: repr.syntax.root.clone(),
                tokens: repr.syntax.tokens.clone(),
                rules: repr.syntax.rules.clone(),
                canonical_print: repr.syntax.canonical_print.clone(),
            },
            common: repr.common.clone(),
            support: repr.support.clone(),
            nodes: repr.nodes.clone(),
        }),
    })
}

fn validate_modern_repr_schema(
    schema: &ModernPilotSchemaDocument,
    path: &Path,
) -> Result<LoadedRepr, String> {
    if schema.name.trim().is_empty() {
        return Err(format!("expected {} name to be non-empty", path.display()));
    }

    if !schema.file_ext.starts_with('.') || schema.file_ext.len() < 2 {
        return Err(format!(
            "expected {} file_ext to start with '.', got {:?}",
            path.display(),
            schema.file_ext
        ));
    }

    if schema.description.trim().is_empty() {
        return Err(format!(
            "expected {} description to be non-empty",
            path.display()
        ));
    }

    if schema.rules.rules.is_empty() {
        return Err(format!(
            "expected {} rules to contain at least one rule",
            path.display()
        ));
    }

    Ok(LoadedRepr {
        doc: None,
        body: LoadedReprBody::Modern(ModernReprBody {
            name: schema.name.clone(),
            file_ext: schema.file_ext.clone(),
            description: schema.description.clone(),
            templates: schema.rules.templates.clone().unwrap_or_default(),
            rules: schema.rules.rules.clone(),
        }),
    })
}

pub(crate) fn documented_name(key: &Documented<String>) -> &str {
    key.value.as_str()
}

pub(crate) fn documented_doc(key: &Documented<String>) -> Option<&[String]> {
    key.doc.as_deref()
}

pub(crate) fn rule_named_parts(named: &RuleNamed) -> (&str, &RuleExpr) {
    (named.0.0.as_str(), &named.0.1)
}

fn collect_rule_refs<'a>(rule: &'a RuleExpr, out: &mut Vec<&'a str>) {
    match rule {
        RuleExpr::Seq(items)
        | RuleExpr::Choice(items)
        | RuleExpr::Semantic(items)
        | RuleExpr::Optional(items)
        | RuleExpr::Repeat(items) => {
            for item in items {
                collect_rule_refs(item, out);
            }
        }
        RuleExpr::Field(named) | RuleExpr::Variant(named) => {
            collect_rule_refs(&named.0.1, out);
        }
        RuleExpr::Ref(names) => out.extend(names.iter().map(String::as_str)),
        RuleExpr::Token(_) | RuleExpr::Tag(_) | RuleExpr::Literal(_) => {}
    }
}

fn collect_rule_tokens<'a>(rule: &'a RuleExpr, out: &mut Vec<&'a str>) {
    match rule {
        RuleExpr::Seq(items)
        | RuleExpr::Choice(items)
        | RuleExpr::Semantic(items)
        | RuleExpr::Optional(items)
        | RuleExpr::Repeat(items) => {
            for item in items {
                collect_rule_tokens(item, out);
            }
        }
        RuleExpr::Field(named) | RuleExpr::Variant(named) => {
            collect_rule_tokens(&named.0.1, out);
        }
        RuleExpr::Token(names) => out.extend(names.iter().map(String::as_str)),
        RuleExpr::Ref(_) | RuleExpr::Tag(_) | RuleExpr::Literal(_) => {}
    }
}

fn canonical_print_target_exists(repr: &ReprBody, target: &str) -> bool {
    if let Some((node_name, variant_name)) = target.split_once('.') {
        return repr.nodes.as_ref().is_some_and(|nodes| {
            nodes.iter().any(|(name, decl)| {
                documented_name(name) == node_name
                    && matches!(
                        decl,
                        NodeDecl::Enum(variants)
                            if variants
                                .variants
                                .keys()
                                .any(|variant| documented_name(variant) == variant_name)
                    )
            })
        }) || repr.support.as_ref().is_some_and(|support| {
            support.iter().any(|(name, decl)| {
                documented_name(name) == node_name
                    && matches!(
                        decl,
                        SupportDecl::Enum(variants)
                            if variants
                                .variants
                                .keys()
                                .any(|variant| documented_name(variant) == variant_name)
                    )
            })
        });
    }

    repr.nodes
        .as_ref()
        .is_some_and(|nodes| nodes.keys().any(|name| documented_name(name) == target))
        || repr
            .support
            .as_ref()
            .is_some_and(|support| support.keys().any(|name| documented_name(name) == target))
}

fn type_decl_exists(repr: &ReprBody, target: &str) -> bool {
    repr.nodes
        .as_ref()
        .is_some_and(|nodes| nodes.keys().any(|name| documented_name(name) == target))
        || repr
            .support
            .as_ref()
            .is_some_and(|support| support.keys().any(|name| documented_name(name) == target))
}
