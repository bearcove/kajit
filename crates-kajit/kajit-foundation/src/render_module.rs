use crate::formatter_codegen::render_formatter_block;
use crate::normalize::{NormalizedNodeDecl, NormalizedRepr};
use crate::parser_codegen::render_parser_block;
use crate::render_helpers::{
    collect_syntax_type_tags, node_fields_have_prov, render_common_placeholder, render_node_decl,
    render_support_decl, render_walk_fn, snake_case,
};

pub(crate) struct GeneratedModuleFile {
    pub(crate) relative_path: String,
    pub(crate) contents: String,
}

pub(crate) fn render_hir_poc_files(repr: &NormalizedRepr) -> Vec<GeneratedModuleFile> {
    let parts = render_parts(repr);

    vec![
        GeneratedModuleFile {
            relative_path: "mod.rs".to_owned(),
            contents: render_root_mod_file(),
        },
        GeneratedModuleFile {
            relative_path: "hir/mod.rs".to_owned(),
            contents: render_hir_mod_file(),
        },
        GeneratedModuleFile {
            relative_path: "hir/meta.rs".to_owned(),
            contents: format_generated_file(render_meta_file(&parts)),
        },
        GeneratedModuleFile {
            relative_path: "hir/ast.rs".to_owned(),
            contents: format_generated_file(render_ast_file(&parts)),
        },
        GeneratedModuleFile {
            relative_path: "hir/visit.rs".to_owned(),
            contents: format_generated_file(render_visit_file(&parts)),
        },
        GeneratedModuleFile {
            relative_path: "hir/parse.rs".to_owned(),
            contents: format_generated_file(render_parse_file(&parts)),
        },
        GeneratedModuleFile {
            relative_path: "hir/format.rs".to_owned(),
            contents: format_generated_file(render_format_file(&parts)),
        },
        GeneratedModuleFile {
            relative_path: "hir/tests.rs".to_owned(),
            contents: format_generated_file(render_tests_file()),
        },
    ]
}

struct RenderParts {
    file_ext: String,
    name: String,
    provenance_tag: String,
    purpose: String,
    round_trip: String,
    provenance: String,
    canonical_identity_rows: String,
    token_rows: String,
    rule_rows: String,
    common_rows: String,
    support_rows: String,
    node_rows: String,
    print_rows: String,
    placeholder_rows: String,
    ast_rows: String,
    prov_impl_rows: String,
    visit_trait_rows: String,
    visit_mut_trait_rows: String,
    walk_rows: String,
    walk_mut_rows: String,
    parser_rows: String,
    formatter_rows: String,
}

fn render_parts(repr: &NormalizedRepr) -> RenderParts {
    let mut token_names = repr.syntax.token_kinds.keys().cloned().collect::<Vec<_>>();
    token_names.sort();

    let mut rule_names = repr.syntax.rules.keys().cloned().collect::<Vec<_>>();
    rule_names.sort();

    let mut print_keys = repr
        .syntax
        .canonical_print
        .keys()
        .cloned()
        .collect::<Vec<_>>();
    print_keys.sort();

    let mut common_names = repr.common.keys().cloned().collect::<Vec<_>>();
    common_names.sort();

    let mut node_names = repr.nodes.keys().cloned().collect::<Vec<_>>();
    node_names.sort();

    let provenance_tag = repr
        .common
        .get("provenance")
        .and_then(|ty| match ty {
            crate::normalize::SyntaxTypeUse::Ref { name } => Some(name.as_str()),
            _ => None,
        })
        .unwrap_or("Prov")
        .to_owned();

    let token_rows = token_names
        .iter()
        .map(|name| {
            let kind = repr.syntax.token_kinds.get(name).unwrap().as_str();
            format!("    TokenSpec {{ name: {name:?}, kind: {kind:?} }},")
        })
        .collect::<Vec<_>>()
        .join("\n");

    let rule_rows = rule_names
        .iter()
        .map(|name| {
            let kind = match repr.syntax.rules.get(name).unwrap() {
                crate::normalize::SyntaxRule::Seq(_) => "seq",
                crate::normalize::SyntaxRule::Choice(_) => "choice",
                crate::normalize::SyntaxRule::Field(_) => "field",
                crate::normalize::SyntaxRule::Variant(_) => "variant",
                crate::normalize::SyntaxRule::Ref { .. } => "ref",
                crate::normalize::SyntaxRule::Token { .. } => "token",
                crate::normalize::SyntaxRule::Optional { .. } => "optional",
                crate::normalize::SyntaxRule::Repeat { .. } => "repeat",
                crate::normalize::SyntaxRule::Literal(_) => "literal",
            };
            format!("    RuleSpec {{ name: {name:?}, kind: {kind:?} }},")
        })
        .collect::<Vec<_>>()
        .join("\n");

    let canonical_identity_rows = repr
        .contract
        .canonical_identities
        .iter()
        .map(|name| format!("    {name:?},"))
        .collect::<Vec<_>>()
        .join("\n");

    let common_rows = common_names
        .iter()
        .map(|name| {
            let kind = match repr.common.get(name) {
                Some(crate::normalize::SyntaxTypeUse::Ref { name }) => name.as_str(),
                Some(_) => "scalar",
                None => "<missing>",
            };
            format!("    TypeUseSpec {{ name: {name:?}, kind: {kind:?} }},")
        })
        .collect::<Vec<_>>()
        .join("\n");

    let node_rows = node_names
        .iter()
        .map(|name| {
            let kind = match repr.nodes.get(name) {
                Some(NormalizedNodeDecl::Node(_)) => "node",
                Some(NormalizedNodeDecl::Enum(_)) => "enum",
                Some(NormalizedNodeDecl::Struct(_)) => "struct",
                None => "<missing>",
            };
            format!("    NodeSpec {{ name: {name:?}, kind: {kind:?} }},")
        })
        .collect::<Vec<_>>()
        .join("\n");

    let mut support_names = repr.support.keys().cloned().collect::<Vec<_>>();
    support_names.sort();

    let support_rows = support_names
        .iter()
        .map(|name| render_support_decl(name, repr.support.get(name).unwrap()))
        .collect::<Vec<_>>()
        .join("\n\n");

    let mut placeholder_names = Vec::new();
    for ty in repr.common.values() {
        collect_syntax_type_tags(ty, &mut placeholder_names);
    }
    for decl in repr.nodes.values() {
        match decl {
            NormalizedNodeDecl::Node(fields) | NormalizedNodeDecl::Struct(fields) => {
                for ty in fields.values() {
                    collect_syntax_type_tags(ty, &mut placeholder_names);
                }
            }
            NormalizedNodeDecl::Enum(variants) => {
                for variant in variants.values() {
                    if let NormalizedNodeDecl::Node(fields) | NormalizedNodeDecl::Struct(fields) =
                        variant
                    {
                        for ty in fields.values() {
                            collect_syntax_type_tags(ty, &mut placeholder_names);
                        }
                    }
                }
            }
        }
    }
    placeholder_names.sort();
    placeholder_names.dedup();
    placeholder_names.retain(|name| {
        !matches!(name.as_str(), "optional" | "seq")
            && !node_names.contains(name)
            && !support_names.contains(name)
    });

    let placeholder_rows = placeholder_names
        .iter()
        .map(|name| render_common_placeholder(name, &common_names, &repr.common))
        .collect::<Vec<_>>()
        .join("\n\n");

    let ast_rows = node_names
        .iter()
        .map(|name| {
            render_node_decl(
                name,
                repr.nodes.get(name).unwrap(),
                &node_names,
                &provenance_tag,
            )
        })
        .collect::<Vec<_>>()
        .join("\n\n");

    let prov_impl_rows = node_names
        .iter()
        .filter_map(|name| {
            let decl = repr.nodes.get(name)?;
            match decl {
                NormalizedNodeDecl::Node(fields) | NormalizedNodeDecl::Struct(fields)
                    if node_fields_have_prov(fields, &provenance_tag) =>
                {
                    Some(format!(
                        "impl HasProvenance for {name} {{\n    fn provenance(&self) -> Option<&{provenance_tag}> {{\n        Some(&self.prov)\n    }}\n}}"
                    ))
                }
                _ => None,
            }
        })
        .collect::<Vec<_>>()
        .join("\n\n");

    let visit_trait_rows = node_names
        .iter()
        .map(|name| {
            let method = snake_case(name);
            format!(
                "    fn visit_{method}(&mut self, node: &{name}) {{\n        walk_{method}(self, node);\n    }}"
            )
        })
        .collect::<Vec<_>>()
        .join("\n");

    let visit_mut_trait_rows = node_names
        .iter()
        .map(|name| {
            let method = snake_case(name);
            format!(
                "    fn visit_{method}_mut(&mut self, node: &mut {name}) {{\n        walk_{method}_mut(self, node);\n    }}"
            )
        })
        .collect::<Vec<_>>()
        .join("\n");

    let walk_rows = node_names
        .iter()
        .map(|name| render_walk_fn(name, repr.nodes.get(name).unwrap(), &node_names, false))
        .collect::<Vec<_>>()
        .join("\n\n");

    let walk_mut_rows = node_names
        .iter()
        .map(|name| render_walk_fn(name, repr.nodes.get(name).unwrap(), &node_names, true))
        .collect::<Vec<_>>()
        .join("\n\n");

    let print_rows = print_keys
        .iter()
        .map(|name| {
            let template = repr.syntax.canonical_print.get(name).unwrap();
            format!("    PrintSpec {{ name: {name:?}, template: {template:?} }},")
        })
        .collect::<Vec<_>>()
        .join("\n");

    let parser_rows = render_parser_block(repr, &node_names, &provenance_tag)
        .expect("parser block should render");
    let formatter_rows =
        render_formatter_block(repr, &node_names).expect("formatter block should render");

    RenderParts {
        file_ext: repr.file_ext.clone(),
        name: repr.name.clone(),
        provenance_tag,
        purpose: repr.contract.purpose.clone(),
        round_trip: repr.contract.round_trip.clone(),
        provenance: repr.contract.provenance.clone(),
        canonical_identity_rows,
        token_rows,
        rule_rows,
        common_rows,
        support_rows,
        node_rows,
        print_rows,
        placeholder_rows,
        ast_rows,
        prov_impl_rows,
        visit_trait_rows,
        visit_mut_trait_rows,
        walk_rows,
        walk_mut_rows,
        parser_rows,
        formatter_rows,
    }
}

fn render_root_mod_file() -> String {
    "// @generated by kajit-foundation::generate_repr_poc. Do not edit manually.\npub mod hir;\n"
        .to_owned()
}

fn render_hir_mod_file() -> String {
    format_generated_file(
        r#"
pub mod ast;
pub mod format;
pub mod meta;
pub mod parse;
pub mod visit;

pub use ast::*;
pub use format::*;
pub use meta::*;
pub use parse::*;
pub use visit::*;

#[cfg(test)]
mod tests;
"#
        .to_owned(),
    )
}

fn render_meta_file(parts: &RenderParts) -> String {
    format!(
        r#"
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

pub static REPR_CANONICAL_IDENTITIES: &[&str] = &[
{canonical_identity_rows}
];

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
"#,
        name = parts.name,
        file_ext = parts.file_ext,
        purpose = parts.purpose,
        round_trip = parts.round_trip,
        provenance = parts.provenance,
        canonical_identity_rows = parts.canonical_identity_rows,
        token_rows = parts.token_rows,
        rule_rows = parts.rule_rows,
        common_rows = parts.common_rows,
        node_rows = parts.node_rows,
        print_rows = parts.print_rows,
    )
}

fn render_ast_file(parts: &RenderParts) -> String {
    format!(
        r#"
pub trait HasProvenance {{
    fn provenance(&self) -> Option<&{provenance_tag}>;
}}

{placeholder_rows}

{support_rows}

{ast_rows}

{prov_impl_rows}
"#,
        provenance_tag = parts.provenance_tag,
        placeholder_rows = parts.placeholder_rows,
        support_rows = parts.support_rows,
        ast_rows = parts.ast_rows,
        prov_impl_rows = parts.prov_impl_rows,
    )
}

fn render_visit_file(parts: &RenderParts) -> String {
    format!(
        r#"
use super::*;

pub trait Visit {{
{visit_trait_rows}
}}

pub trait VisitMut {{
{visit_mut_trait_rows}
}}

{walk_rows}

{walk_mut_rows}
"#,
        visit_trait_rows = parts.visit_trait_rows,
        visit_mut_trait_rows = parts.visit_mut_trait_rows,
        walk_rows = parts.walk_rows,
        walk_mut_rows = parts.walk_mut_rows,
    )
}

fn render_parse_file(parts: &RenderParts) -> String {
    format!(
        r#"
use chumsky::prelude::*;

use super::*;

{parser_rows}
"#,
        parser_rows = parts.parser_rows,
    )
}

fn render_format_file(parts: &RenderParts) -> String {
    format!(
        r#"
use super::*;

{formatter_rows}
"#,
        formatter_rows = parts.formatter_rows,
    )
}

fn render_tests_file() -> String {
    r#"
use super::*;

#[test]
fn parse_module_smoke() {
    let module = parse_module_text("module { fn main() -> Value { return } }").unwrap();
    assert_eq!(module.functions.len(), 1);
    assert_eq!(module.functions[0].name, Symbol("main".to_owned()));
    assert_eq!(module.functions[0].return_type, Type("Value".to_owned()));
    assert!(matches!(
        module.functions[0].body.statements.as_slice(),
        [Stmt::Return { value: None, .. }]
    ));
}

#[test]
fn format_module_smoke() {
    let text = "module { fn main() -> Value { return } }";
    let module = parse_module_text(text).unwrap();
    let formatted = format_module_text(&module);
    assert_eq!(formatted, "module {\nfn main() -> Value {\nreturn\n}\n}");

    let reparsed = parse_module_text(&formatted).unwrap();
    assert_eq!(reparsed, module);
}
"#
    .to_owned()
}

fn format_generated_file(raw: String) -> String {
    let body = prettyplease::unparse(&syn::parse_file(&raw).expect("generated file should parse"));
    format!("// @generated by kajit-foundation::generate_repr_poc. Do not edit manually.\n\n{body}")
}
