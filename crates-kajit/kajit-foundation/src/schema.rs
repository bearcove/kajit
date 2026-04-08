use std::collections::HashMap;
use std::fs;
use std::path::Path;

use facet::Facet;
use facet_styx::RenderError;

#[derive(Facet, Debug, Clone)]
#[allow(dead_code)]
pub(crate) struct PilotSchemaDocument {
    pub(crate) meta: PilotMeta,
    pub(crate) repr: ReprDecl,
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
    pub(crate) nodes: Option<HashMap<String, NodeDecl>>,
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
#[facet(rename_all = "lowercase")]
#[repr(u8)]
pub(crate) enum TypeUse {
    Optional(Vec<TypeUse>),
    Seq(Vec<TypeUse>),
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
pub(crate) enum NodeDecl {
    Node(NodeFields),
    Enum(NodeVariants),
    Struct(NodeFields),
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
    pub(crate) fields: HashMap<String, TypeUse>,
}

#[derive(Facet, Debug, Clone)]
#[allow(dead_code)]
pub(crate) struct NodeVariants {
    #[facet(flatten)]
    pub(crate) variants: HashMap<String, NodeDecl>,
}

pub(crate) fn load_hir_pilot_schema(schema_path: &Path) -> Result<ReprBody, String> {
    let schema_source = fs::read_to_string(schema_path)
        .map_err(|e| format!("failed to read {}: {e}", schema_path.display()))?;
    let schema: PilotSchemaDocument = facet_styx::from_str(&schema_source).map_err(|e| {
        format!(
            "failed to parse {} as Styx\n{}",
            schema_path.display(),
            e.render(&schema_path.display().to_string(), &schema_source)
        )
    })?;
    validate_hir_pilot_schema(&schema, schema_path)
}

fn validate_hir_pilot_schema(
    schema: &PilotSchemaDocument,
    path: &Path,
) -> Result<ReprBody, String> {
    if schema.meta.id != "kajit:repr-schema/hir-pilot" {
        return Err(format!(
            "expected {} meta.id to be kajit:repr-schema/hir-pilot, got {:?}",
            path.display(),
            schema.meta.id
        ));
    }

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

    let ReprDecl::Module(repr) = &schema.repr;

    if repr.name != "HIR" {
        return Err(format!(
            "expected {} repr name to be HIR, got {:?}",
            path.display(),
            repr.name
        ));
    }

    if repr.file_ext != ".vixen-hir" {
        return Err(format!(
            "expected {} file_ext to be .vixen-hir, got {:?}",
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

    for rule_name in ["Module", "Function", "Param", "Block", "Stmt", "Expr"] {
        if !repr.syntax.rules.contains_key(rule_name) {
            return Err(format!(
                "expected {} syntax.rules to contain {:?}",
                path.display(),
                rule_name
            ));
        }
    }

    for print_name in ["Module", "Function", "Stmt.Return", "Expr.Call"] {
        if repr
            .syntax
            .canonical_print
            .get(print_name)
            .is_none_or(|s| s.trim().is_empty())
        {
            return Err(format!(
                "expected {} canonical_print to contain non-empty {:?}",
                path.display(),
                print_name
            ));
        }
    }

    for token_name in ["ident", "symbol", "int"] {
        let Some(token_spec) = repr.syntax.tokens.get(token_name) else {
            return Err(format!(
                "expected {} syntax.tokens to contain {:?}",
                path.display(),
                token_name
            ));
        };

        match token_spec {
            TokenExpr::Regex(patterns) if !patterns.is_empty() && !patterns[0].is_empty() => {}
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

    expect_tagged_rule(path, &repr.syntax.rules, "Module", "seq")?;
    expect_tagged_rule(path, &repr.syntax.rules, "Function", "seq")?;
    expect_tagged_rule(path, &repr.syntax.rules, "Param", "seq")?;
    expect_tagged_rule(path, &repr.syntax.rules, "Block", "seq")?;
    expect_tagged_rule(path, &repr.syntax.rules, "Stmt", "choice")?;
    expect_tagged_rule(path, &repr.syntax.rules, "Expr", "choice")?;

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

    Ok(ReprBody {
        name: repr.name.clone(),
        file_ext: repr.file_ext.clone(),
        contract: ReprContract {
            purpose: repr.contract.purpose.clone(),
            canonical_identities: repr.contract.canonical_identities.clone(),
            round_trip: repr.contract.round_trip.clone(),
            provenance: repr.contract.provenance.clone(),
        },
        syntax: ReprSyntax {
            tokens: repr.syntax.tokens.clone(),
            rules: repr.syntax.rules.clone(),
            canonical_print: repr.syntax.canonical_print.clone(),
        },
        common: repr.common.clone(),
        nodes: repr.nodes.clone(),
    })
}

fn expect_tagged_rule(
    path: &Path,
    rules: &HashMap<String, RuleExpr>,
    rule_name: &str,
    expected_tag: &str,
) -> Result<(), String> {
    let Some(rule) = rules.get(rule_name) else {
        return Err(format!(
            "expected {} syntax.rules to contain {:?}",
            path.display(),
            rule_name
        ));
    };

    let actual_tag = rule_expr_kind(rule);

    if actual_tag != expected_tag {
        return Err(format!(
            "expected {} syntax.rules.{rule_name} to be @{expected_tag}(...), got {actual_tag:?}",
            path.display()
        ));
    }

    Ok(())
}

pub(crate) fn rule_expr_kind(rule: &RuleExpr) -> &'static str {
    match rule {
        RuleExpr::Seq(_) => "seq",
        RuleExpr::Choice(_) => "choice",
        RuleExpr::Field(_) => "field",
        RuleExpr::Variant(_) => "variant",
        RuleExpr::Ref(_) => "ref",
        RuleExpr::Token(_) => "token",
        RuleExpr::Optional(_) => "optional",
        RuleExpr::Repeat(_) => "repeat",
        RuleExpr::Literal(_) => "literal",
    }
}

pub(crate) fn rule_named_parts(named: &RuleNamed) -> (&str, &RuleExpr) {
    (named.0.0.as_str(), &named.0.1)
}

pub(crate) fn rule_literal_text(rule: &RuleExpr) -> Option<&str> {
    match rule {
        RuleExpr::Literal(Some(text)) => Some(text.as_str()),
        _ => None,
    }
}
