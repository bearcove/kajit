use crate::normalize::{NormalizedNodeDecl, NormalizedRepr};
use crate::parser_codegen::render_parser_block;
use crate::render_helpers::{
    collect_syntax_type_tags, render_common_placeholder, render_node_decl, render_walk_fn,
    snake_case,
};

pub(crate) fn render_hir_poc_module(repr: &NormalizedRepr) -> String {
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
    placeholder_names
        .retain(|name| !matches!(name.as_str(), "optional" | "seq") && !node_names.contains(name));

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
                    if crate::render_helpers::node_fields_have_prov(fields, &provenance_tag) => {
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

    let raw = format!(
        r###"
// @generated by kajit-foundation::generate_repr_poc from {file_ext} schema {name}.
// Do not edit manually.
//
// This module is intentionally narrow: it exposes only data that is actually
// derived from the pilot schema.

use chumsky::prelude::*;

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

pub trait HasProvenance {{
    fn provenance(&self) -> Option<&{provenance_tag}>;
}}

pub trait Visit {{
{visit_trait_rows}
}}

pub trait VisitMut {{
{visit_mut_trait_rows}
}}

pub const REPR_NAME: &str = {name:?};
pub const REPR_FILE_EXT: &str = {file_ext:?};
pub const REPR_PURPOSE: &str = {purpose:?};
pub const REPR_ROUND_TRIP: &str = {round_trip:?};
pub const REPR_PROVENANCE: &str = {provenance:?};
pub static REPR_CANONICAL_IDENTITIES: &[&str] = &[
{canonical_identity_rows}
];

{placeholder_rows}

{ast_rows}

{prov_impl_rows}

{walk_rows}

{walk_mut_rows}

{parser_rows}

#[cfg(test)]
mod tests {{
    use super::*;

    #[test]
    fn parse_module_smoke() {{
        let module = parse_module_text("module {{ fn main() -> Value {{ return }} }}").unwrap();
        assert_eq!(module.functions.len(), 1);
        assert_eq!(module.functions[0].name, Symbol("main".to_owned()));
        assert_eq!(module.functions[0].return_type, Type("Value".to_owned()));
        assert!(matches!(module.functions[0].body.statements.as_slice(), [Stmt::Return {{ value: None, .. }}]));
    }}
}}

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
        canonical_identity_rows = canonical_identity_rows,
        round_trip = repr.contract.round_trip,
        provenance = repr.contract.provenance,
        placeholder_rows = placeholder_rows,
        ast_rows = ast_rows,
        prov_impl_rows = prov_impl_rows,
        visit_trait_rows = visit_trait_rows,
        visit_mut_trait_rows = visit_mut_trait_rows,
        walk_rows = walk_rows,
        walk_mut_rows = walk_mut_rows,
        parser_rows = parser_rows,
        provenance_tag = provenance_tag,
        token_rows = token_rows,
        rule_rows = rule_rows,
        common_rows = common_rows,
        node_rows = node_rows,
        print_rows = print_rows,
    );

    prettyplease::unparse(&syn::parse_file(&raw).expect("generated HIR POC should parse"))
}
