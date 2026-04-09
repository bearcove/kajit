use crate::formatter_codegen::render_formatter_block;
use crate::hover_codegen::render_hover_block;
use crate::normalize::{NormalizedNodeDecl, NormalizedNodeKind, NormalizedRepr};
use crate::parser_codegen::render_parser_block;
use crate::render_helpers::{
    collect_syntax_type_tags, render_common_placeholder, render_node_decl, render_provenance_impl,
    render_support_decl, render_walk_fn, snake_case,
};
use crate::semantic_codegen::render_semantic_block;

pub(crate) struct GeneratedModuleFile {
    pub(crate) relative_path: String,
    pub(crate) contents: String,
}

pub(crate) fn render_repr_poc_files(reprs: &[NormalizedRepr]) -> Vec<GeneratedModuleFile> {
    let mut files = Vec::new();
    files.push(GeneratedModuleFile {
        relative_path: "mod.rs".to_owned(),
        contents: render_root_mod_file(reprs),
    });

    for repr in reprs {
        let parts = render_parts(repr);
        let module_dir = snake_case(&repr.name);
        files.extend([
            GeneratedModuleFile {
                relative_path: format!("{module_dir}/mod.rs"),
                contents: render_repr_mod_file(repr.name == "HIR"),
            },
            GeneratedModuleFile {
                relative_path: format!("{module_dir}/meta.rs"),
                contents: format_generated_file(render_meta_file(&parts)),
            },
            GeneratedModuleFile {
                relative_path: format!("{module_dir}/ast.rs"),
                contents: format_generated_file(render_ast_file(&parts)),
            },
            GeneratedModuleFile {
                relative_path: format!("{module_dir}/visit.rs"),
                contents: format_generated_file(render_visit_file(&parts)),
            },
            GeneratedModuleFile {
                relative_path: format!("{module_dir}/provenance.rs"),
                contents: format_generated_file(render_provenance_file(&parts)),
            },
            GeneratedModuleFile {
                relative_path: format!("{module_dir}/parse.rs"),
                contents: format_generated_file(render_parse_file(&parts)),
            },
            GeneratedModuleFile {
                relative_path: format!("{module_dir}/format.rs"),
                contents: format_generated_file(render_format_file(&parts)),
            },
            GeneratedModuleFile {
                relative_path: format!("{module_dir}/semantic.rs"),
                contents: format_generated_file(render_semantic_file(&parts)),
            },
            GeneratedModuleFile {
                relative_path: format!("{module_dir}/hover.rs"),
                contents: format_generated_file(render_hover_file(&parts)),
            },
        ]);
        if repr.name == "HIR" {
            files.push(GeneratedModuleFile {
                relative_path: format!("{module_dir}/tests.rs"),
                contents: format_generated_file(render_tests_file()),
            });
        }
    }

    files
}

struct RenderParts {
    module_doc_rows: String,
    file_ext: String,
    name: String,
    root_name: String,
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
    semantic_rows: String,
    hover_rows: String,
}

fn render_parts(repr: &NormalizedRepr) -> RenderParts {
    let mut token_names = repr.syntax.token_specs.keys().cloned().collect::<Vec<_>>();
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

    let module_doc_rows = repr
        .doc
        .as_deref()
        .map(render_module_doc_lines)
        .unwrap_or_default();

    let token_rows = token_names
        .iter()
        .map(|name| {
            let kind = repr.syntax.token_specs.get(name).unwrap().regex.as_str();
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
            let kind = match repr.nodes.get(name).map(|decl| &decl.value) {
                Some(NormalizedNodeDecl::Record { kind, .. }) => match kind {
                    NormalizedNodeKind::Node => "node",
                    NormalizedNodeKind::Struct => "struct",
                    NormalizedNodeKind::Entity => "entity",
                    NormalizedNodeKind::Slot => "slot",
                },
                Some(NormalizedNodeDecl::Enum(_)) => "enum",
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
        .map(|name| {
            let decl = repr.support.get(name).unwrap();
            render_support_decl(name, &decl.value, decl.doc.as_deref(), &node_names)
        })
        .collect::<Vec<_>>()
        .join("\n\n");

    let mut placeholder_names = Vec::new();
    for ty in repr.common.values() {
        collect_syntax_type_tags(ty, &mut placeholder_names);
    }
    for decl in repr.nodes.values() {
        match &decl.value {
            NormalizedNodeDecl::Record { fields, .. } => {
                for ty in fields.values() {
                    collect_syntax_type_tags(&ty.value, &mut placeholder_names);
                }
            }
            NormalizedNodeDecl::Enum(variants) => {
                for variant in variants.values() {
                    if let NormalizedNodeDecl::Record { fields, .. } = &variant.value {
                        for ty in fields.values() {
                            collect_syntax_type_tags(&ty.value, &mut placeholder_names);
                        }
                    }
                }
            }
        }
    }
    for decl in repr.support.values() {
        if let crate::normalize::NormalizedSupportDecl::Struct(fields) = &decl.value {
            for ty in fields.values() {
                collect_syntax_type_tags(&ty.value, &mut placeholder_names);
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
            let decl = repr.nodes.get(name).unwrap();
            render_node_decl(
                name,
                &decl.value,
                &node_names,
                decl.doc.as_deref(),
                &provenance_tag,
            )
        })
        .collect::<Vec<_>>()
        .join("\n\n");

    let prov_impl_rows = node_names
        .iter()
        .filter_map(|name| {
            render_provenance_impl(name, &repr.nodes.get(name)?.value, &provenance_tag)
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
        .map(|name| {
            render_walk_fn(
                name,
                &repr.nodes.get(name).unwrap().value,
                &node_names,
                false,
                &provenance_tag,
            )
        })
        .collect::<Vec<_>>()
        .join("\n\n");

    let walk_mut_rows = node_names
        .iter()
        .map(|name| {
            render_walk_fn(
                name,
                &repr.nodes.get(name).unwrap().value,
                &node_names,
                true,
                &provenance_tag,
            )
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
    let formatter_rows =
        render_formatter_block(repr, &node_names).expect("formatter block should render");
    let semantic_rows =
        render_semantic_block(repr, &node_names).expect("semantic block should render");
    let hover_rows = render_hover_block(repr, &node_names).expect("hover block should render");

    RenderParts {
        module_doc_rows,
        file_ext: repr.file_ext.clone(),
        name: repr.name.clone(),
        root_name: repr.syntax.root.clone(),
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
        semantic_rows,
        hover_rows,
    }
}

fn render_root_mod_file(reprs: &[NormalizedRepr]) -> String {
    let mut module_names = reprs
        .iter()
        .map(|repr| snake_case(&repr.name))
        .collect::<Vec<_>>();
    module_names.sort();
    let mod_rows = module_names
        .iter()
        .map(|name| format!("pub mod {name};"))
        .collect::<Vec<_>>()
        .join("\n");
    let helper_rows = reprs
        .iter()
        .map(|repr| {
            let module_name = snake_case(&repr.name);
            format!(
                r#"
fn validate_{module_name}(source: &str) -> Result<(), String> {{
    {module_name}::parse_root_text(source).map(|_| ())
}}

fn format_{module_name}(source: &str) -> Result<String, String> {{
    let root = {module_name}::parse_root_text(source)?;
    Ok({module_name}::format_root_text(&root))
}}

fn semantic_tokens_{module_name}(source: &str) -> Vec<SemanticToken> {{
    {module_name}::semantic_tokens(source)
}}

fn hover_entries_{module_name}(source: &str) -> Vec<HoverEntry> {{
    {module_name}::hover_entries(source)
}}

fn resolve_{module_name}(source: &str) -> Result<ResolutionSet, String> {{
    {module_name}::resolve(source)
}}
"#
            )
        })
        .collect::<Vec<_>>()
        .join("\n");
    let repr_rows = reprs
        .iter()
        .map(|repr| {
            let module_name = snake_case(&repr.name);
            format!(
                r#"    ReprSpec {{
        name: {module_name}::REPR_NAME,
        file_ext: {module_name}::REPR_FILE_EXT,
        validate: validate_{module_name},
        format: format_{module_name},
        semantic_tokens: semantic_tokens_{module_name},
        hover_entries: hover_entries_{module_name},
        resolve: resolve_{module_name},
    }}"#
            )
        })
        .collect::<Vec<_>>()
        .join(",\n");

    format_generated_file(format!(
        r#"
{mod_rows}

#[derive(Clone, Copy)]
pub struct ReprSpec {{
    pub name: &'static str,
    pub file_ext: &'static str,
    pub validate: fn(&str) -> Result<(), String>,
    pub format: fn(&str) -> Result<String, String>,
    pub semantic_tokens: fn(&str) -> Vec<SemanticToken>,
    pub hover_entries: fn(&str) -> Vec<HoverEntry>,
    pub resolve: fn(&str) -> Result<ResolutionSet, String>,
}}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SemanticToken {{
    pub start: u32,
    pub end: u32,
    pub kind: &'static str,
}}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HoverEntry {{
    pub start: u32,
    pub end: u32,
    pub markdown: String,
    pub priority: u8,
}}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SymbolKind {{
    Function,
    Type,
    Label,
}}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SymbolDef {{
    pub name: String,
    pub kind: SymbolKind,
    pub start: u32,
    pub end: u32,
    pub detail: Option<String>,
    pub docs: Option<String>,
}}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SymbolRef {{
    pub name: String,
    pub kind: SymbolKind,
    pub start: u32,
    pub end: u32,
}}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResolvedRef {{
    pub reference: SymbolRef,
    pub target: Option<usize>,
}}

#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct ResolutionSet {{
    pub definitions: Vec<SymbolDef>,
    pub references: Vec<ResolvedRef>,
}}

{helper_rows}

pub static REPRS: &[ReprSpec] = &[
{repr_rows}
];
"#
    ))
}

fn render_repr_mod_file(include_tests: bool) -> String {
    let tests_row = if include_tests {
        "\n#[cfg(test)]\nmod tests;\n"
    } else {
        "\n"
    };
    format_generated_file(format!(
        r#"
pub mod ast;
pub mod format;
pub mod hover;
pub mod meta;
pub mod parse;
pub mod provenance;
pub mod resolve;
pub mod semantic;
pub mod visit;

pub use ast::*;
pub use format::*;
pub use hover::*;
pub use meta::*;
pub use parse::*;
pub use provenance::*;
pub use resolve::*;
pub use semantic::*;
pub use visit::*;
{tests_row}"#
    ))
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
pub const REPR_ROOT: &str = {root_name:?};
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
        root_name = parts.root_name,
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
{module_doc_rows}

pub trait EntityNode {{}}
pub trait SlotNode {{}}

{placeholder_rows}

{support_rows}

{ast_rows}
"#,
        module_doc_rows = parts.module_doc_rows,
        placeholder_rows = parts.placeholder_rows,
        support_rows = parts.support_rows,
        ast_rows = parts.ast_rows,
    )
}

fn render_visit_file(parts: &RenderParts) -> String {
    format!(
        r#"
#![allow(unused_variables)]

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

fn render_provenance_file(parts: &RenderParts) -> String {
    format!(
        r#"
use super::ast::*;

pub trait HasProvenance {{
    fn provenance(&self) -> Option<&{provenance_tag}>;
}}

{prov_impl_rows}
"#,
        provenance_tag = parts.provenance_tag,
        prov_impl_rows = parts.prov_impl_rows,
    )
}

fn render_format_file(parts: &RenderParts) -> String {
    format!(
        r#"
#![allow(dead_code, unused_variables)]

use super::*;

{formatter_rows}
"#,
        formatter_rows = parts.formatter_rows,
    )
}

fn render_semantic_file(parts: &RenderParts) -> String {
    format!(
        r#"
#![allow(dead_code, unused_variables)]

use kajit_types::Prov;

use super::*;
use crate::schema_poc::SemanticToken;
use super::provenance::HasProvenance;

{semantic_rows}
"#,
        semantic_rows = parts.semantic_rows,
    )
}

fn render_hover_file(parts: &RenderParts) -> String {
    format!(
        r#"
#![allow(dead_code, unused_variables)]

use kajit_types::Prov;

use super::*;
use crate::schema_poc::HoverEntry;
use super::provenance::HasProvenance;

{hover_rows}
"#,
        hover_rows = parts.hover_rows,
    )
}

fn render_tests_file() -> String {
    r#"
use super::*;

#[test]
fn parse_module_smoke() {
    let module = parse_root_text("module { fn main() -> Value { return } }").unwrap();
    assert_eq!(module.functions.len(), 1);
    assert_eq!(module.functions[0].name.text, "main");
    assert_eq!(module.functions[0].return_type.text, "Value");
    assert!(matches!(
        module.functions[0].body.statements.as_slice(),
        [Stmt::Return { value: None, .. }]
    ));
}

#[test]
fn format_module_smoke() {
    let text = "module { fn main() -> Value { return } }";
    let module = parse_root_text(text).unwrap();
    let formatted = format_root_text(&module);
    assert_eq!(
        formatted,
        "module {\n    fn main() -> Value {\n        return\n    }\n}"
    );

    let reparsed = parse_root_text(&formatted).unwrap();
    assert_eq!(reparsed, module);
}
"#
    .to_owned()
}

pub(crate) fn render_default_resolve_file() -> String {
    format_generated_file(
        r#"
use super::*;
use crate::schema_poc::ResolutionSet;

pub fn resolve(_source: &str) -> Result<ResolutionSet, String> {
    Ok(ResolutionSet::default())
}
"#
        .to_owned(),
    )
}

fn format_generated_file(raw: String) -> String {
    let body = prettyplease::unparse(&syn::parse_file(&raw).expect("generated file should parse"));
    let body = add_breathing_room(&body);
    format!("// @generated by kajit-foundation::generate_repr_poc. Do not edit manually.\n\n{body}")
}

fn render_module_doc_lines(lines: &[String]) -> String {
    lines
        .iter()
        .map(|line| format!("//! {line}"))
        .collect::<Vec<_>>()
        .join("\n")
}

fn add_breathing_room(body: &str) -> String {
    let mut out = String::new();
    let mut prev = "";

    for line in body.lines() {
        let trimmed = line.trim();
        let prev_trimmed = prev.trim();

        let is_top_level = !line.starts_with(' ');
        let starts_top_level_item = is_top_level
            && (trimmed.starts_with("///")
                || trimmed.starts_with("#[derive")
                || trimmed.starts_with("pub struct ")
                || trimmed.starts_with("pub enum ")
                || trimmed.starts_with("pub trait ")
                || trimmed.starts_with("impl ")
                || trimmed.starts_with("pub use ")
                || trimmed.starts_with("pub const ")
                || trimmed.starts_with("pub static "));

        let starts_variant_doc = line.starts_with("    ///");
        let starts_variant = line.starts_with("    ")
            && !line.starts_with("        ")
            && trimmed.ends_with(',')
            && !trimmed.starts_with("#[");
        let starts_field_doc = line.starts_with("        ///");
        let prev_opens_enum_body = prev_trimmed.ends_with('{') && !prev.starts_with("    ");

        let prev_ends_item = prev_trimmed.ends_with('}') || prev_trimmed.ends_with(';');
        let prev_ends_variant = prev_trimmed.ends_with(',') && !prev.starts_with("        ");
        let prev_ends_field = prev_trimmed.ends_with(',') && prev.starts_with("        ");

        if !out.is_empty() {
            if starts_top_level_item && prev_ends_item && !prev_trimmed.is_empty() {
                out.push('\n');
            } else if (starts_variant_doc || starts_variant)
                && prev_opens_enum_body
                && !prev_trimmed.is_empty()
            {
                out.push('\n');
            } else if (starts_variant_doc || starts_variant)
                && prev_ends_variant
                && !prev_trimmed.is_empty()
            {
                out.push('\n');
            } else if starts_field_doc && prev_ends_field && !prev_trimmed.is_empty() {
                out.push('\n');
            }
        }

        out.push_str(line);
        out.push('\n');
        prev = line;
    }

    out
}
