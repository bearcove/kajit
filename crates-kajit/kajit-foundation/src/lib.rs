use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};

use facet::Facet;
use facet_styx::RenderError;

pub fn generate_repr_poc(workspace_root: &Path) -> Result<Vec<PathBuf>, String> {
    let schema_path = workspace_root.join("notes/unified-ast/pilot/hir.repr.styx");
    let schema_source = fs::read_to_string(&schema_path)
        .map_err(|e| format!("failed to read {}: {e}", schema_path.display()))?;
    let schema: PilotSchemaDocument = facet_styx::from_str(&schema_source)
        .map_err(|e| {
            format!(
                "failed to parse {} as Styx\n{}",
                schema_path.display(),
                e.render(&schema_path.display().to_string(), &schema_source)
            )
        })?;
    let repr = validate_hir_pilot_schema(&schema, &schema_path)?;

    let out_dir = workspace_root.join("crates-kajit/kajit-reprs/src/schema_poc");
    fs::create_dir_all(&out_dir)
        .map_err(|e| format!("failed to create {}: {e}", out_dir.display()))?;

    let mod_path = out_dir.join("mod.rs");
    fs::write(&mod_path, "pub mod hir;\n")
        .map_err(|e| format!("failed to write {}: {e}", mod_path.display()))?;

    let hir_path = out_dir.join("hir.rs");
    fs::write(&hir_path, render_hir_poc_module(repr))
        .map_err(|e| format!("failed to write {}: {e}", hir_path.display()))?;

    Ok(vec![mod_path, hir_path])
}

#[derive(Facet, Debug)]
#[allow(dead_code)]
struct PilotSchemaDocument {
    meta: PilotMeta,
    repr: ReprDecl,
}

#[derive(Facet, Debug)]
#[allow(dead_code)]
struct PilotMeta {
    id: String,
    version: u64,
    description: String,
}

#[derive(Facet, Debug)]
#[allow(dead_code)]
#[facet(rename_all = "snake_case")]
#[repr(u8)]
enum ReprDecl {
    Module(ReprBody),
}

#[derive(Facet, Debug)]
#[allow(dead_code)]
struct ReprBody {
    name: String,
    file_ext: String,
    contract: ReprContract,
    syntax: ReprSyntax,
    common: Option<HashMap<String, TypeUse>>,
    nodes: Option<HashMap<String, NodeDecl>>,
}

#[derive(Facet, Debug)]
#[allow(dead_code)]
struct ReprContract {
    purpose: String,
    canonical_identities: Vec<String>,
    round_trip: String,
    provenance: String,
}

#[derive(Facet, Debug)]
#[allow(dead_code)]
struct ReprSyntax {
    tokens: HashMap<String, TokenExpr>,
    rules: HashMap<String, RuleExpr>,
    canonical_print: HashMap<String, String>,
}

#[derive(Facet, Debug)]
#[allow(dead_code)]
#[facet(rename_all = "lowercase")]
#[repr(u8)]
enum TokenExpr {
    Regex(Vec<String>),
    #[facet(other)]
    Other {
        #[facet(tag)]
        name: Option<String>,
        #[facet(content)]
        content: Option<ExprPayload>,
    },
}

#[derive(Facet, Debug)]
#[allow(dead_code)]
#[repr(u8)]
enum RuleExpr {
    #[facet(other)]
    Form {
        #[facet(tag)]
        tag: Option<String>,
        #[facet(content)]
        content: Option<ExprPayload>,
    },
}

#[derive(Facet, Debug)]
#[allow(dead_code)]
#[facet(untagged)]
#[repr(u8)]
enum ExprPayload {
    Scalar(String),
    Seq(Vec<RuleExpr>),
    #[facet(other)]
    Other {
        #[facet(tag)]
        name: Option<String>,
        #[facet(content)]
        content: Option<Box<ExprPayload>>,
    },
}

#[derive(Facet, Debug)]
#[allow(dead_code)]
#[repr(u8)]
enum TypeUse {
    #[facet(other)]
    Form {
        #[facet(tag)]
        tag: Option<String>,
        #[facet(content)]
        content: Option<TypeUsePayload>,
    },
}

#[derive(Facet, Debug)]
#[allow(dead_code)]
#[facet(untagged)]
#[repr(u8)]
enum TypeUsePayload {
    Scalar(String),
    Seq(Vec<TypeUse>),
}

#[derive(Facet, Debug)]
#[allow(dead_code)]
#[facet(rename_all = "lowercase")]
#[repr(u8)]
enum NodeDecl {
    Node(NodeFields),
    Enum(NodeVariants),
    Struct(NodeFields),
    #[facet(other)]
    Other {
        #[facet(tag)]
        tag: Option<String>,
        #[facet(content)]
        content: Option<TypeUsePayload>,
    },
}

#[derive(Facet, Debug)]
#[allow(dead_code)]
struct NodeFields {
    #[facet(flatten)]
    fields: HashMap<String, TypeUse>,
}

#[derive(Facet, Debug)]
#[allow(dead_code)]
struct NodeVariants {
    #[facet(flatten)]
    variants: HashMap<String, NodeDecl>,
}

fn validate_hir_pilot_schema<'a>(schema: &'a PilotSchemaDocument, path: &Path) -> Result<&'a ReprBody, String> {
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
        return Err(format!("expected {} repr.common to be present", path.display()));
    }

    if repr.nodes.is_none() {
        return Err(format!("expected {} repr.nodes to be present", path.display()));
    }

    Ok(repr)
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

    let actual_tag = match rule {
        RuleExpr::Form { tag, .. } => tag.as_deref().unwrap_or("literal"),
    };

    if actual_tag != expected_tag {
        return Err(format!(
            "expected {} syntax.rules.{rule_name} to be @{expected_tag}(...), got {actual_tag:?}",
            path.display()
        ));
    }

    Ok(())
}

fn render_hir_poc_module(repr: &ReprBody) -> String {
    let mut token_names = repr.syntax.tokens.keys().cloned().collect::<Vec<_>>();
    token_names.sort();

    let mut rule_names = repr.syntax.rules.keys().cloned().collect::<Vec<_>>();
    rule_names.sort();

    let mut print_keys = repr.syntax.canonical_print.keys().cloned().collect::<Vec<_>>();
    print_keys.sort();

    let mut common_names = repr
        .common
        .as_ref()
        .map(|common| common.keys().cloned().collect::<Vec<_>>())
        .unwrap_or_default();
    common_names.sort();

    let mut node_names = repr
        .nodes
        .as_ref()
        .map(|nodes| nodes.keys().cloned().collect::<Vec<_>>())
        .unwrap_or_default();
    node_names.sort();

    let token_rows = token_names
        .iter()
        .map(|name| {
            let kind = match repr.syntax.tokens.get(name).unwrap() {
                TokenExpr::Regex(_) => "regex",
                TokenExpr::Other { name, .. } => name.as_deref().unwrap_or("<unknown>"),
            };
            format!("    TokenSpec {{ name: {name:?}, kind: {kind:?} }},")
        })
        .collect::<Vec<_>>()
        .join("\n");

    let rule_rows = rule_names
        .iter()
        .map(|name| {
            let kind = match repr.syntax.rules.get(name).unwrap() {
                RuleExpr::Form { tag, .. } => tag.as_deref().unwrap_or("literal"),
            };
            format!("    RuleSpec {{ name: {name:?}, kind: {kind:?} }},")
        })
        .collect::<Vec<_>>()
        .join("\n");

    let common_rows = common_names
        .iter()
        .map(|name| {
            let kind = match repr.common.as_ref().and_then(|common| common.get(name)) {
                Some(TypeUse::Form { tag, .. }) => tag.as_deref().unwrap_or("scalar"),
                None => "<missing>",
            };
            format!("    TypeUseSpec {{ name: {name:?}, kind: {kind:?} }},")
        })
        .collect::<Vec<_>>()
        .join("\n");

    let node_rows = node_names
        .iter()
        .map(|name| {
            let kind = match repr.nodes.as_ref().and_then(|nodes| nodes.get(name)) {
                Some(NodeDecl::Node(_)) => "node",
                Some(NodeDecl::Enum(_)) => "enum",
                Some(NodeDecl::Struct(_)) => "struct",
                Some(NodeDecl::Other { tag, .. }) => tag.as_deref().unwrap_or("<unknown>"),
                None => "<missing>",
            };
            format!("    NodeSpec {{ name: {name:?}, kind: {kind:?} }},")
        })
        .collect::<Vec<_>>()
        .join("\n");

    let print_rows = print_keys
        .iter()
        .map(|name| {
            let template = repr.syntax.canonical_print.get(name).unwrap();
            format!("    PrintSpec {{ name: {name:?}, template: {template:?} }},")
        })
        .collect::<Vec<_>>()
        .join("\n");

    let raw = format!(
        r###"
// @generated by kajit-foundation::generate_repr_poc from {file_ext} schema {name}.
// Do not edit manually.
//
// This module is intentionally narrow: it exposes only data that is actually
// derived from the pilot schema. AST/parser/formatter generation starts later.

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TokenSpec {{
    pub name: &'static str,
    pub kind: &'static str,
}}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RuleSpec {{
    pub name: &'static str,
    pub kind: &'static str,
}}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TypeUseSpec {{
    pub name: &'static str,
    pub kind: &'static str,
}}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct NodeSpec {{
    pub name: &'static str,
    pub kind: &'static str,
}}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PrintSpec {{
    pub name: &'static str,
    pub template: &'static str,
}}

pub const REPR_NAME: &str = {name:?};
pub const REPR_FILE_EXT: &str = {file_ext:?};
pub const REPR_PURPOSE: &str = {purpose:?};
pub const REPR_ROUND_TRIP: &str = {round_trip:?};
pub const REPR_PROVENANCE: &str = {provenance:?};

pub static TOKENS: &[TokenSpec] = &[
{token_rows}
];

pub static RULES: &[RuleSpec] = &[
{rule_rows}
];

pub static COMMON_TYPES: &[TypeUseSpec] = &[
{common_rows}
];

pub static NODES: &[NodeSpec] = &[
{node_rows}
];

pub static CANONICAL_PRINT: &[PrintSpec] = &[
{print_rows}
];
"###,
        file_ext = repr.file_ext,
        name = repr.name,
        purpose = repr.contract.purpose,
        round_trip = repr.contract.round_trip,
        provenance = repr.contract.provenance,
        token_rows = token_rows,
        rule_rows = rule_rows,
        common_rows = common_rows,
        node_rows = node_rows,
        print_rows = print_rows,
    );

    prettyplease::unparse(&syn::parse_file(&raw).expect("generated HIR POC should parse"))
}
