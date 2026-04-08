use crate::parser_codegen::render_parser_block;
use crate::render_helpers::{
    collect_type_tags, render_common_placeholder, render_node_decl, render_walk_fn, snake_case,
};
use crate::schema::{NodeDecl, ReprBody, rule_expr_kind, type_use_tag};

pub(crate) fn render_hir_poc_module(repr: &ReprBody) -> String {
    let mut token_names = repr.syntax.tokens.keys().cloned().collect::<Vec<_>>();
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

    let provenance_tag = repr
        .common
        .as_ref()
        .and_then(|common| common.get("provenance"))
        .and_then(type_use_tag)
        .unwrap_or("Prov")
        .to_owned();

    let token_rows = token_names
        .iter()
        .map(|name| {
            let kind = match repr.syntax.tokens.get(name).unwrap() {
                crate::schema::TokenExpr::Regex(_) => "regex",
                crate::schema::TokenExpr::Other { name, .. } => {
                    name.as_deref().unwrap_or("<unknown>")
                }
            };
            format!("    TokenSpec {{ name: {name:?}, kind: {kind:?} }},")
        })
        .collect::<Vec<_>>()
        .join("\n");

    let rule_rows = rule_names
        .iter()
        .map(|name| {
            let kind = rule_expr_kind(repr.syntax.rules.get(name).unwrap());
            format!("    RuleSpec {{ name: {name:?}, kind: {kind:?} }},")
        })
        .collect::<Vec<_>>()
        .join("\n");

    let common_rows = common_names
        .iter()
        .map(|name| {
            let kind = match repr.common.as_ref().and_then(|common| common.get(name)) {
                Some(ty) => type_use_tag(ty).unwrap_or("scalar"),
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

    let mut placeholder_names = Vec::new();
    if let Some(common) = &repr.common {
        for ty in common.values() {
            collect_type_tags(ty, &mut placeholder_names);
        }
    }
    if let Some(nodes) = &repr.nodes {
        for decl in nodes.values() {
            match decl {
                NodeDecl::Node(fields) | NodeDecl::Struct(fields) => {
                    for ty in fields.fields.values() {
                        collect_type_tags(ty, &mut placeholder_names);
                    }
                }
                NodeDecl::Enum(variants) => {
                    for variant in variants.variants.values() {
                        if let NodeDecl::Node(fields) | NodeDecl::Struct(fields) = variant {
                            for ty in fields.fields.values() {
                                collect_type_tags(ty, &mut placeholder_names);
                            }
                        }
                    }
                }
                NodeDecl::Other { .. } => {}
            }
        }
    }
    placeholder_names.sort();
    placeholder_names.dedup();
    placeholder_names
        .retain(|name| !matches!(name.as_str(), "optional" | "seq") && !node_names.contains(name));

    let placeholder_rows = placeholder_names
        .iter()
        .map(|name| render_common_placeholder(name, &common_names, repr))
        .collect::<Vec<_>>()
        .join("\n\n");

    let ast_rows = node_names
        .iter()
        .filter_map(|name| {
            repr.nodes
                .as_ref()
                .and_then(|nodes| nodes.get(name))
                .and_then(|decl| render_node_decl(name, decl, &node_names, &provenance_tag))
        })
        .collect::<Vec<_>>()
        .join("\n\n");

    let prov_impl_rows = node_names
        .iter()
        .filter_map(|name| {
            let decl = repr.nodes.as_ref().and_then(|nodes| nodes.get(name))?;
            match decl {
                NodeDecl::Node(fields) | NodeDecl::Struct(fields)
                    if crate::render_helpers::node_fields_have_prov(fields, &provenance_tag) =>
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
        .filter_map(|name| {
            repr.nodes
                .as_ref()
                .and_then(|nodes| nodes.get(name))
                .and_then(|decl| render_walk_fn(name, decl, &node_names, false))
        })
        .collect::<Vec<_>>()
        .join("\n\n");

    let walk_mut_rows = node_names
        .iter()
        .filter_map(|name| {
            repr.nodes
                .as_ref()
                .and_then(|nodes| nodes.get(name))
                .and_then(|decl| render_walk_fn(name, decl, &node_names, true))
        })
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
