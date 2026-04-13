use crate::formatter_codegen::render_formatter_block;
use crate::hover_codegen::render_hover_block;
use crate::normalize::SyntaxTypeUse;
use crate::normalize::{DocumentedValue, NormalizedNodeDecl, NormalizedNodeKind, NormalizedRepr};
use crate::parser_codegen::render_parser_block;
use crate::render_helpers::{
    collect_syntax_type_tags, render_common_placeholder, render_node_decl, render_provenance_impl,
    render_support_decl, render_type_use_kind, render_walk_fn, rust_ident, snake_case,
};
use crate::semantic_codegen::render_semantic_block;

pub(crate) struct GeneratedModuleFile {
    pub(crate) relative_path: String,
    pub(crate) contents: String,
}

pub(crate) fn render_repr_poc_files(reprs: &[NormalizedRepr]) -> Vec<GeneratedModuleFile> {
    let mut files = Vec::new();
    files.push(GeneratedModuleFile {
        relative_path: "generated.rs".to_owned(),
        contents: render_root_mod_file(reprs),
    });

    for repr in reprs {
        let parts = render_parts(repr);
        let module_dir = snake_case(&repr.name);
        files.extend([
            GeneratedModuleFile {
                relative_path: format!("{module_dir}/mod.rs"),
                contents: render_repr_mod_file(repr.name == "HIR", repr.name == "MIR"),
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
    node_field_rows: String,
    pool_rows: String,
    ref_rows: String,
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
    graph_storage_field_rows: String,
    graph_storage_init_rows: String,
    graph_storage_method_rows: String,
}

#[derive(Clone)]
struct ArenaGraphInfo {
    item: String,
    key: String,
    storage_field: String,
    key_accessor: String,
}

fn field_id_accessor(
    item_name: &str,
    decl: &NormalizedNodeDecl,
    key_name: &str,
) -> Result<String, String> {
    match decl {
        NormalizedNodeDecl::Record { fields, .. } => {
            let matching_fields = fields
                .iter()
                .filter_map(|(field_name, ty)| match &ty.value {
                    SyntaxTypeUse::Ref { name } if name == key_name => Some(field_name.clone()),
                    _ => None,
                })
                .collect::<Vec<_>>();
            let key_field = matching_fields
                .iter()
                .find(|field_name| field_name.as_str() == "id")
                .cloned()
                .or_else(|| {
                    if matching_fields.len() == 1 {
                        matching_fields.first().cloned()
                    } else {
                        None
                    }
                })
                .ok_or_else(|| {
                    format!(
                        "keyed pool item {item_name} must expose a unique field of type {key_name}"
                    )
                })?;
            Ok(format!("value.{}", rust_ident(&key_field)))
        }
        NormalizedNodeDecl::Enum(variants) => {
            let mut arms = Vec::new();
            let variant_names = variants.keys().cloned().collect::<Vec<_>>();
            for variant_name in variant_names {
                let variant = &variants[&variant_name].value;
                match variant {
                    NormalizedNodeDecl::Record { fields, .. } => {
                        let matching_fields = fields
                            .iter()
                            .filter_map(|(field_name, ty)| match &ty.value {
                                SyntaxTypeUse::Ref { name } if name == key_name => {
                                    Some(field_name.clone())
                                }
                                _ => None,
                            })
                            .collect::<Vec<_>>();
                        let key_field = matching_fields
                            .iter()
                            .find(|field_name| field_name.as_str() == "id")
                            .cloned()
                            .or_else(|| {
                                if matching_fields.len() == 1 {
                                    matching_fields.first().cloned()
                                } else {
                                    None
                                }
                            })
                            .ok_or_else(|| {
                                format!(
                                    "keyed pool item {item_name}.{variant_name} must expose a unique field of type {key_name}"
                                )
                            })?;
                        arms.push(format!(
                            "        {item_name}::{variant_name} {{ {}, .. }} => *{},",
                            rust_ident(&key_field),
                            rust_ident(&key_field),
                        ));
                    }
                    NormalizedNodeDecl::Enum(_) => {
                        return Err(format!(
                            "keyed pool item {item_name}.{variant_name} cannot be nested enum"
                        ));
                    }
                }
            }
            Ok(format!("match value {{\n{}\n    }}", arms.join("\n")))
        }
    }
}

fn collect_arena_graph_infos(repr: &NormalizedRepr) -> Result<Vec<ArenaGraphInfo>, String> {
    let mut infos = std::collections::BTreeMap::<String, ArenaGraphInfo>::new();

    let mut collect_from_fields =
        |fields: &std::collections::HashMap<String, DocumentedValue<SyntaxTypeUse>>| {
            for ty in fields.values() {
                if let SyntaxTypeUse::Arena {
                    item: inner,
                    key: Some(key),
                } = &ty.value
                    && let SyntaxTypeUse::Ref { name: item_name } = inner.as_ref()
                {
                    let decl = repr.nodes.get(item_name).ok_or_else(|| {
                        format!("keyed arena item {item_name:?} must be a declared node")
                    })?;
                    let key_accessor = field_id_accessor(item_name, &decl.value, key)?;
                    let storage_field = format!("{}s", snake_case(item_name));
                    if let Some(existing) = infos.get(item_name) {
                        if existing.key != *key {
                            return Err(format!(
                                "arena item {item_name} used with conflicting keys: {:?} vs {:?}",
                                existing.key, key
                            ));
                        }
                    } else {
                        infos.insert(
                            item_name.clone(),
                            ArenaGraphInfo {
                                item: item_name.clone(),
                                key: key.clone(),
                                storage_field,
                                key_accessor,
                            },
                        );
                    }
                }
            }
            Ok::<(), String>(())
        };

    for decl in repr.nodes.values() {
        match &decl.value {
            NormalizedNodeDecl::Record { fields, .. } => collect_from_fields(fields)?,
            NormalizedNodeDecl::Enum(variants) => {
                for variant in variants.values() {
                    if let NormalizedNodeDecl::Record { fields, .. } = &variant.value {
                        collect_from_fields(fields)?;
                    }
                }
            }
        }
    }

    Ok(infos.into_values().collect())
}

fn repr_has_graph_arenas(repr: &NormalizedRepr) -> bool {
    repr.nodes.values().any(|decl| match &decl.value {
        NormalizedNodeDecl::Record { fields, .. } => fields
            .values()
            .any(|field| matches!(field.value, SyntaxTypeUse::Arena { key: Some(_), .. })),
        NormalizedNodeDecl::Enum(variants) => variants.values().any(|variant| {
            matches!(
                &variant.value,
                NormalizedNodeDecl::Record { fields, .. }
                    if fields.values().any(
                        |field| matches!(field.value, SyntaxTypeUse::Arena { key: Some(_), .. })
                    )
            )
        }),
    })
}

fn collect_pool_specs(
    owner: &str,
    field: &str,
    ty: &crate::normalize::SyntaxTypeUse,
) -> Vec<String> {
    match ty {
        crate::normalize::SyntaxTypeUse::Arena {
            item,
            key: Some(key),
        }
        | crate::normalize::SyntaxTypeUse::Pool {
            item,
            key: Some(key),
        } => {
            vec![format!(
                "    PoolSpec {{ owner: {owner:?}, field: {field:?}, item: {:?}, key: {key:?} }},",
                render_type_use_kind(item)
            )]
        }
        _ => Vec::new(),
    }
}

fn collect_ref_specs(
    owner: &str,
    field: &str,
    ty: &crate::normalize::SyntaxTypeUse,
) -> Vec<String> {
    match ty {
        crate::normalize::SyntaxTypeUse::Optional(inner)
        | crate::normalize::SyntaxTypeUse::Seq(inner)
        | crate::normalize::SyntaxTypeUse::Order(inner) => collect_ref_specs(owner, field, inner),
        crate::normalize::SyntaxTypeUse::Arena { item, .. }
        | crate::normalize::SyntaxTypeUse::Pool { item, .. } => {
            collect_ref_specs(owner, field, item)
        }
        crate::normalize::SyntaxTypeUse::RefTo { id, target } => vec![format!(
            "    RefSpec {{ owner: {owner:?}, field: {field:?}, id: {:?}, target: {target:?} }},",
            render_type_use_kind(id)
        )],
        _ => Vec::new(),
    }
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
                crate::normalize::SyntaxRule::Semantic { .. } => "semantic",
                crate::normalize::SyntaxRule::Field(_) => "field",
                crate::normalize::SyntaxRule::Variant(_) => "variant",
                crate::normalize::SyntaxRule::Ref { .. } => "ref",
                crate::normalize::SyntaxRule::Token { .. } => "token",
                crate::normalize::SyntaxRule::Tag(_) => "tag",
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
                Some(ty) => render_type_use_kind(ty),
                None => "<missing>".to_owned(),
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
                    NormalizedNodeKind::Struct => "struct",
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
            render_support_decl(name, &decl.value, decl.doc.as_deref())
        })
        .collect::<Vec<_>>()
        .join("\n\n");

    let node_field_rows = node_names
        .iter()
        .flat_map(|name| {
            let decl = repr.nodes.get(name).unwrap();
            match &decl.value {
                NormalizedNodeDecl::Record { fields, .. } => {
                    let mut field_names = fields.keys().cloned().collect::<Vec<_>>();
                    field_names.sort();
                    field_names
                        .into_iter()
                        .map(|field_name| {
                            let kind = render_type_use_kind(&fields[&field_name].value);
                            format!(
                                "    FieldSpec {{ owner: {name:?}, field: {field_name:?}, kind: {kind:?} }},"
                            )
                        })
                        .collect::<Vec<_>>()
                }
                NormalizedNodeDecl::Enum(variants) => {
                    let variant_names = variants.keys().cloned().collect::<Vec<_>>();
                    variant_names
                        .into_iter()
                        .flat_map(|variant_name| match &variants[&variant_name].value {
                            NormalizedNodeDecl::Record { fields, .. } => {
                                let mut field_names = fields.keys().cloned().collect::<Vec<_>>();
                                field_names.sort();
                                field_names
                                    .into_iter()
                                    .map(|field_name| {
                                        let kind = render_type_use_kind(&fields[&field_name].value);
                                        format!(
                                            "    FieldSpec {{ owner: {:?}, field: {field_name:?}, kind: {kind:?} }},",
                                            format!("{name}.{variant_name}")
                                        )
                                    })
                                    .collect::<Vec<_>>()
                            }
                            NormalizedNodeDecl::Enum(_) => Vec::new(),
                        })
                        .collect::<Vec<_>>()
                }
            }
        })
        .collect::<Vec<_>>()
        .join("\n");

    let pool_rows = node_names
        .iter()
        .flat_map(|name| {
            let decl = repr.nodes.get(name).unwrap();
            match &decl.value {
                NormalizedNodeDecl::Record { fields, .. } => {
                    let mut field_names = fields.keys().cloned().collect::<Vec<_>>();
                    field_names.sort();
                    field_names
                        .into_iter()
                        .flat_map(|field_name| {
                            collect_pool_specs(name, &field_name, &fields[&field_name].value)
                        })
                        .collect::<Vec<_>>()
                }
                NormalizedNodeDecl::Enum(variants) => {
                    let variant_names = variants.keys().cloned().collect::<Vec<_>>();
                    variant_names
                        .into_iter()
                        .flat_map(|variant_name| match &variants[&variant_name].value {
                            NormalizedNodeDecl::Record { fields, .. } => {
                                let owner = format!("{name}.{variant_name}");
                                let mut field_names = fields.keys().cloned().collect::<Vec<_>>();
                                field_names.sort();
                                field_names
                                    .into_iter()
                                    .flat_map(|field_name| {
                                        collect_pool_specs(
                                            &owner,
                                            &field_name,
                                            &fields[&field_name].value,
                                        )
                                    })
                                    .collect::<Vec<_>>()
                            }
                            NormalizedNodeDecl::Enum(_) => Vec::new(),
                        })
                        .collect::<Vec<_>>()
                }
            }
        })
        .collect::<Vec<_>>()
        .join("\n");

    let ref_rows = node_names
        .iter()
        .flat_map(|name| {
            let decl = repr.nodes.get(name).unwrap();
            match &decl.value {
                NormalizedNodeDecl::Record { fields, .. } => {
                    let mut field_names = fields.keys().cloned().collect::<Vec<_>>();
                    field_names.sort();
                    field_names
                        .into_iter()
                        .flat_map(|field_name| {
                            collect_ref_specs(name, &field_name, &fields[&field_name].value)
                        })
                        .collect::<Vec<_>>()
                }
                NormalizedNodeDecl::Enum(variants) => {
                    let variant_names = variants.keys().cloned().collect::<Vec<_>>();
                    variant_names
                        .into_iter()
                        .flat_map(|variant_name| match &variants[&variant_name].value {
                            NormalizedNodeDecl::Record { fields, .. } => {
                                let owner = format!("{name}.{variant_name}");
                                let mut field_names = fields.keys().cloned().collect::<Vec<_>>();
                                field_names.sort();
                                field_names
                                    .into_iter()
                                    .flat_map(|field_name| {
                                        collect_ref_specs(
                                            &owner,
                                            &field_name,
                                            &fields[&field_name].value,
                                        )
                                    })
                                    .collect::<Vec<_>>()
                            }
                            NormalizedNodeDecl::Enum(_) => Vec::new(),
                        })
                        .collect::<Vec<_>>()
                }
            }
        })
        .collect::<Vec<_>>()
        .join("\n");

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
    let arena_graph_infos =
        collect_arena_graph_infos(repr).expect("arena graph infos should collect");
    let graph_storage_field_rows = arena_graph_infos
        .iter()
        .map(|info| {
            format!(
                "    {}: super::super::ArenaStorage<{}, {}>,",
                info.storage_field, info.key, info.item
            )
        })
        .collect::<Vec<_>>()
        .join("\n");
    let graph_storage_init_rows = arena_graph_infos
        .iter()
        .map(|info| {
            format!(
                "            {}: super::super::ArenaStorage::new(),",
                info.storage_field
            )
        })
        .collect::<Vec<_>>()
        .join("\n");
    let graph_storage_method_rows = arena_graph_infos
        .iter()
        .map(|info| {
            let item_snake = snake_case(&info.item);
            format!(
                "    pub fn insert_{item_snake}(&mut self, value: {item}) -> Result<{key}, String> {{
        let key = {{
            let value = &value;
            {key_accessor}
        }};
        self.{storage_field}.insert(key, value)?;
        Ok(key)
    }}

    pub fn {item_snake}(&self, handle: {key}) -> Option<&{item}> {{
        self.{storage_field}.get(handle)
    }}

    pub fn {item_snake}_mut(&mut self, handle: {key}) -> Option<&mut {item}> {{
        self.{storage_field}.get_mut(handle)
    }}

    pub fn all_{item_snake}(&self) -> impl Iterator<Item = &{item}> {{
        self.{storage_field}.values()
    }}",
                item = info.item,
                key = info.key,
                storage_field = info.storage_field,
                key_accessor = info.key_accessor,
            )
        })
        .collect::<Vec<_>>()
        .join("\n\n");

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
        node_field_rows,
        pool_rows,
        ref_rows,
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
        graph_storage_field_rows,
        graph_storage_init_rows,
        graph_storage_method_rows,
    }
}

fn render_root_mod_file(reprs: &[NormalizedRepr]) -> String {
    let mut module_names = reprs
        .iter()
        .map(|repr| snake_case(&repr.name))
        .collect::<Vec<_>>();
    module_names.sort();
    let module_use_rows = if module_names.is_empty() {
        String::new()
    } else {
        format!("use crate::{{{}}};\n", module_names.join(", "))
    };
    let helper_rows = reprs
        .iter()
        .map(|repr| {
            let module_name = snake_case(&repr.name);
            if repr_has_graph_arenas(repr) {
                format!(
                    r#"
fn validate_{module_name}(source: &str) -> Result<(), String> {{
    let mut graph = {module_name}::Graph::new();
    let handle = {module_name}::parse_root_into_graph(&mut graph, source)?;
    {module_name}::validate_root_in_graph(&graph, handle)
}}

fn format_{module_name}(source: &str) -> Result<String, String> {{
    let mut graph = {module_name}::Graph::new();
    let handle = {module_name}::parse_root_into_graph(&mut graph, source)?;
    {module_name}::format_root_in_graph(&graph, handle)
}}

fn semantic_tokens_{module_name}(source: &str) -> Vec<SemanticToken> {{
    let mut graph = {module_name}::Graph::new();
    let Ok(handle) = {module_name}::parse_root_into_graph(&mut graph, source) else {{
        return Vec::new();
    }};
    {module_name}::semantic_tokens_in_graph(source, &graph, handle)
}}

fn hover_entries_{module_name}(source: &str) -> Vec<HoverEntry> {{
    let mut graph = {module_name}::Graph::new();
    let Ok(handle) = {module_name}::parse_root_into_graph(&mut graph, source) else {{
        return Vec::new();
    }};
    {module_name}::hover_entries_in_graph(&graph, handle)
}}

fn resolve_{module_name}(source: &str) -> Result<ResolutionSet, String> {{
    {module_name}::resolve(source)
}}
"#
                )
            } else {
                format!(
                    r#"
fn validate_{module_name}(source: &str) -> Result<(), String> {{
    {module_name}::validate_root_text(source)
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
            }
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
{module_use_rows}

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

#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct Pool<T>(pub Vec<T>);

impl<T> Pool<T> {{
    pub fn new() -> Self {{
        Self(Vec::new())
    }}
}}

impl<T> From<Vec<T>> for Pool<T> {{
    fn from(value: Vec<T>) -> Self {{
        Self(value)
    }}
}}

impl<T> std::ops::Deref for Pool<T> {{
    type Target = Vec<T>;

    fn deref(&self) -> &Self::Target {{
        &self.0
    }}
}}

impl<T> std::ops::DerefMut for Pool<T> {{
    fn deref_mut(&mut self) -> &mut Self::Target {{
        &mut self.0
    }}
}}

impl<T> IntoIterator for Pool<T> {{
    type Item = T;
    type IntoIter = std::vec::IntoIter<T>;

    fn into_iter(self) -> Self::IntoIter {{
        self.0.into_iter()
    }}
}}

impl<'a, T> IntoIterator for &'a Pool<T> {{
    type Item = &'a T;
    type IntoIter = std::slice::Iter<'a, T>;

    fn into_iter(self) -> Self::IntoIter {{
        self.0.iter()
    }}
}}

impl<'a, T> IntoIterator for &'a mut Pool<T> {{
    type Item = &'a mut T;
    type IntoIter = std::slice::IterMut<'a, T>;

    fn into_iter(self) -> Self::IntoIter {{
        self.0.iter_mut()
    }}
}}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ArenaStorage<Id, T>
where
    Id: Ord + Copy + std::fmt::Debug,
{{
    by_id: std::collections::BTreeMap<Id, T>,
}}

impl<Id, T> Default for ArenaStorage<Id, T>
where
    Id: Ord + Copy + std::fmt::Debug,
{{
    fn default() -> Self {{
        Self {{
            by_id: std::collections::BTreeMap::new(),
        }}
    }}
}}

impl<Id, T> ArenaStorage<Id, T>
where
    Id: Ord + Copy + std::fmt::Debug,
{{
    pub fn new() -> Self {{
        Self {{
            by_id: std::collections::BTreeMap::new(),
        }}
    }}

    pub fn insert(&mut self, id: Id, value: T) -> Result<(), String> {{
        if self.by_id.insert(id, value).is_some() {{
            return Err(format!("duplicate arena id {{id:?}}"));
        }}
        Ok(())
    }}

    pub fn get(&self, id: Id) -> Option<&T> {{
        self.by_id.get(&id)
    }}

    pub fn get_mut(&mut self, id: Id) -> Option<&mut T> {{
        self.by_id.get_mut(&id)
    }}

    pub fn values(&self) -> impl Iterator<Item = &T> {{
        self.by_id.values()
    }}
}}

#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct Arena<T>(pub Vec<T>);

impl<T> Arena<T> {{
    pub fn new() -> Self {{
        Self(Vec::new())
    }}
}}

impl<T> From<Vec<T>> for Arena<T> {{
    fn from(value: Vec<T>) -> Self {{
        Self(value)
    }}
}}

impl<T> std::ops::Deref for Arena<T> {{
    type Target = Vec<T>;

    fn deref(&self) -> &Self::Target {{
        &self.0
    }}
}}

impl<T> std::ops::DerefMut for Arena<T> {{
    fn deref_mut(&mut self) -> &mut Self::Target {{
        &mut self.0
    }}
}}

impl<T> IntoIterator for Arena<T> {{
    type Item = T;
    type IntoIter = std::vec::IntoIter<T>;

    fn into_iter(self) -> Self::IntoIter {{
        self.0.into_iter()
    }}
}}

impl<'a, T> IntoIterator for &'a Arena<T> {{
    type Item = &'a T;
    type IntoIter = std::slice::Iter<'a, T>;

    fn into_iter(self) -> Self::IntoIter {{
        self.0.iter()
    }}
}}

impl<'a, T> IntoIterator for &'a mut Arena<T> {{
    type Item = &'a mut T;
    type IntoIter = std::slice::IterMut<'a, T>;

    fn into_iter(self) -> Self::IntoIter {{
        self.0.iter_mut()
    }}
}}

#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct Order<T>(pub Vec<T>);

impl<T> Order<T> {{
    pub fn new() -> Self {{
        Self(Vec::new())
    }}
}}

impl<T> From<Vec<T>> for Order<T> {{
    fn from(value: Vec<T>) -> Self {{
        Self(value)
    }}
}}

impl<T> std::ops::Deref for Order<T> {{
    type Target = Vec<T>;

    fn deref(&self) -> &Self::Target {{
        &self.0
    }}
}}

impl<T> std::ops::DerefMut for Order<T> {{
    fn deref_mut(&mut self) -> &mut Self::Target {{
        &mut self.0
    }}
}}

impl<T> IntoIterator for Order<T> {{
    type Item = T;
    type IntoIter = std::vec::IntoIter<T>;

    fn into_iter(self) -> Self::IntoIter {{
        self.0.into_iter()
    }}
}}

impl<'a, T> IntoIterator for &'a Order<T> {{
    type Item = &'a T;
    type IntoIter = std::slice::Iter<'a, T>;

    fn into_iter(self) -> Self::IntoIter {{
        self.0.iter()
    }}
}}

impl<'a, T> IntoIterator for &'a mut Order<T> {{
    type Item = &'a mut T;
    type IntoIter = std::slice::IterMut<'a, T>;

    fn into_iter(self) -> Self::IntoIter {{
        self.0.iter_mut()
    }}
}}

{helper_rows}

pub static REPRS: &[ReprSpec] = &[
{repr_rows}
];
"#
    ))
}

fn render_repr_mod_file(include_tests: bool, include_storage: bool) -> String {
    let tests_row = if include_tests {
        "\n#[cfg(test)]\nmod tests;\n"
    } else {
        "\n"
    };
    let storage_mod_row = if include_storage {
        "pub mod storage;\n"
    } else {
        ""
    };
    let storage_use_row = if include_storage {
        "pub use storage::*;\n"
    } else {
        ""
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
{storage_mod_row}
pub mod validate;
pub mod visit;

pub use ast::*;
pub use format::*;
pub use hover::*;
pub use meta::*;
pub use parse::*;
pub use provenance::*;
pub use resolve::*;
pub use semantic::*;
{storage_use_row}
pub use validate::*;
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
pub struct FieldSpec {{
    pub owner: &'static str,
    pub field: &'static str,
    pub kind: &'static str,
}}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PoolSpec {{
    pub owner: &'static str,
    pub field: &'static str,
    pub item: &'static str,
    pub key: &'static str,
}}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RefSpec {{
    pub owner: &'static str,
    pub field: &'static str,
    pub id: &'static str,
    pub target: &'static str,
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

pub static NODE_FIELDS: &[FieldSpec] = &[
{node_field_rows}
];

pub static POOLS: &[PoolSpec] = &[
{pool_rows}
];

pub static REFS: &[RefSpec] = &[
{ref_rows}
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
        node_field_rows = parts.node_field_rows,
        pool_rows = parts.pool_rows,
        ref_rows = parts.ref_rows,
        print_rows = parts.print_rows,
    )
}

fn render_ast_file(parts: &RenderParts) -> String {
    let root_handle_name = format!("{}Handle", parts.root_name);
    let graph_new_rows = if parts.graph_storage_init_rows.trim().is_empty() {
        "        Self::default()".to_owned()
    } else {
        format!(
            "        Self {{\n            roots: Vec::new(),\n{}\n        }}",
            parts.graph_storage_init_rows
        )
    };
    format!(
        r#"
{module_doc_rows}

pub trait EntityNode {{}}
pub trait SlotNode {{}}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
pub struct {root_handle_name}(pub u32);

impl {root_handle_name} {{
    pub const fn new(index: u32) -> Self {{
        Self(index)
    }}

    pub const fn index(self) -> usize {{
        self.0 as usize
    }}
}}

#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct Graph {{
    roots: Vec<{root_name}>,
{graph_storage_field_rows}
}}

impl Graph {{
    pub fn new() -> Self {{
{graph_new_rows}
    }}

    pub fn insert_root(&mut self, root: {root_name}) -> {root_handle_name} {{
        let handle = {root_handle_name}::new(self.roots.len() as u32);
        self.roots.push(root);
        handle
    }}

    pub fn root(&self, handle: {root_handle_name}) -> Option<&{root_name}> {{
        self.roots.get(handle.index())
    }}

    pub fn root_mut(&mut self, handle: {root_handle_name}) -> Option<&mut {root_name}> {{
        self.roots.get_mut(handle.index())
    }}

    pub fn roots(&self) -> &[{root_name}] {{
        &self.roots
    }}

    pub fn len(&self) -> usize {{
        self.roots.len()
    }}

    pub fn is_empty(&self) -> bool {{
        self.roots.is_empty()
    }}

{graph_storage_method_rows}
}}

{placeholder_rows}

{support_rows}

{ast_rows}
"#,
        module_doc_rows = parts.module_doc_rows,
        root_handle_name = root_handle_name,
        root_name = parts.root_name,
        graph_storage_field_rows = parts.graph_storage_field_rows,
        graph_new_rows = graph_new_rows,
        graph_storage_method_rows = parts.graph_storage_method_rows,
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
use crate::SemanticToken;
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
use crate::HoverEntry;
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
fn parse_module_into_graph_smoke() {
    let mut graph = Graph::new();
    let handle = parse_root_into_graph(&mut graph, "module { fn main() -> Value { return } }")
        .unwrap();
    let module = graph.root(handle).unwrap();
    assert_eq!(module.functions.len(), 1);
    assert_eq!(module.functions[0].name.text, "main");
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
    assert_eq!(format_root_text(&reparsed), formatted);
    assert_eq!(reparsed.functions[0].name.text, module.functions[0].name.text);
    assert_eq!(
        reparsed.functions[0].return_type.text,
        module.functions[0].return_type.text
    );
}
"#
    .to_owned()
}

pub(crate) fn render_default_resolve_file() -> String {
    format_generated_file(
        r#"
use crate::ResolutionSet;

pub fn resolve(_source: &str) -> Result<ResolutionSet, String> {
    Ok(ResolutionSet::default())
}
"#
        .to_owned(),
    )
}

fn format_generated_file(raw: String) -> String {
    let parsed = syn::parse_file(&raw)
        .unwrap_or_else(|err| panic!("generated file should parse: {err}\n\n{raw}"));
    let body = prettyplease::unparse(&parsed);
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

        let prev_ends_item = prev_trimmed.ends_with('}') || prev_trimmed.ends_with(';');
        let prev_ends_variant = prev_trimmed.ends_with(',') && !prev.starts_with("        ");
        let prev_ends_field = prev_trimmed.ends_with(',') && prev.starts_with("        ");

        if !out.is_empty()
            && ((starts_top_level_item && prev_ends_item && !prev_trimmed.is_empty())
                || ((starts_variant_doc || starts_variant)
                    && prev_ends_variant
                    && !prev_trimmed.is_empty())
                || (starts_field_doc && prev_ends_field && !prev_trimmed.is_empty()))
        {
            out.push('\n');
        }

        out.push_str(line);
        out.push('\n');
        prev = line;
    }

    out
}
