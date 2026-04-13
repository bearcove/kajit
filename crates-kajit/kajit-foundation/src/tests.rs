use std::fs;
use std::path::PathBuf;

use crate::normalize;
use crate::render_module;
use crate::schema;

#[test]
fn modern_schema_template_syntax_deserializes() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("modern.repr.styx");
    fs::write(
        &path,
        r#"
name Demo
file_ext .k-demo
description "modern schema"

rules {
    templates {
        Keyword {
            syntax @template{
                params ({text @Any})
                body {
                    syntax text
                    highlight keyword
                }
            }
        }
    }

    FnKw @Keyword("fn")
}
"#,
    )
    .unwrap();

    let loaded = schema::load_pilot_schema(&path).unwrap();
    assert_eq!(loaded.name, "Demo");
    assert!(
        loaded
            .templates
            .keys()
            .any(|name| schema::documented_name(name) == "Keyword")
    );
    assert!(
        loaded
            .rules
            .keys()
            .any(|name| schema::documented_name(name) == "FnKw")
    );
}

#[test]
fn mir_pilot_schema_loads_and_normalizes() {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../notes/unified-ast/pilot/mir.repr.styx");

    let loaded = schema::load_pilot_schema(&path).unwrap();
    let _repr = normalize::normalize_repr(&loaded).unwrap();
}

#[test]
fn hir_support_enum_variant_order_stays_in_schema_order() {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../notes/unified-ast/pilot/hir.repr.styx");

    let loaded = schema::load_pilot_schema(&path).unwrap();
    let repr = normalize::normalize_repr(&loaded).unwrap();
    let files = render_module::render_repr_poc_files(&[repr]);
    let ast = files
        .iter()
        .find(|file| file.relative_path == "hir/ast.rs")
        .unwrap()
        .contents
        .clone();

    let call_pos = ast.find("\n    Call {").unwrap();
    let local_pos = ast.find("\n    Local {").unwrap();
    let literal_pos = ast.find("\n    Literal {").unwrap();
    assert!(call_pos < local_pos && local_pos < literal_pos);
}

#[test]
fn asm_repro_schema_loads() {
    tracing_subscriber::fmt::init();

    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../notes/unified-ast/pilot/asm-repro.repr.styx");

    let _loaded = schema::load_pilot_schema(&path).unwrap();
}

// Minimal repro for double-free in facet-reflect deferred cleanup
// https://github.com/bearcove/kajit/issues/XXX
#[test]
fn asm_repro_btreemap_string_u64() {
    use std::collections::BTreeMap;

    let input = r#"{
        "hello" 42
        "world" 123
    }"#;

    let _result: BTreeMap<String, u64> = facet_styx::from_str(input).unwrap();
}

#[test]
fn asm_repro_btreemap_u64_string() {
    use std::collections::BTreeMap;

    let input = r#"{
        42 "hello"
        123 "world"
    }"#;

    let _result: BTreeMap<u64, String> = facet_styx::from_str(input).unwrap();
}

// Minimal repro for double-free in facet-reflect deferred cleanup.
// Uses facet_styx::Documented<T> which has #[facet(metadata = "doc")] on the Option field.
// The original bug triggers when a map with Documented<String> keys is deserialized
// and the deferred cleanup path drops the same memory twice.
#[test]
fn asm_repro_documented_string_map_key() {
    use std::collections::HashMap;

    // In styx, bare scalars become strings for Documented<String> map keys
    let input = r#"{
        hello 42
        world 123
    }"#;

    let _result: HashMap<facet_styx::Documented<String>, u64> =
        facet_styx::from_str(input).unwrap();
}

#[test]
fn asm_repro_map_with_struct_value() {
    use std::collections::BTreeMap;

    // Struct with Option field as map value - mirrors the original cleanup path
    #[derive(Debug, PartialEq, Eq, PartialOrd, Ord, facet::Facet)]
    #[repr(C)]
    struct Entry {
        syntax: SyntaxKind,
        doc: Option<String>,
    }

    #[derive(Debug, PartialEq, Eq, PartialOrd, Ord, facet::Facet)]
    #[repr(u8)]
    enum SyntaxKind {
        Template,
        Literal,
    }

    let input = r#"{
        hello {
            syntax @template
        }
    }"#;

    let _result: BTreeMap<String, Entry> = facet_styx::from_str(input).unwrap();
}

// Exact pattern from ModernRulesDecl: Option<IndexMap<Documented<String>, T>>
// where T is a struct containing an enum with a Box variant.
// This reproduces the double-free in facet-reflect deferred cleanup.
#[test]
fn asm_repro_option_indexmap_documented_key() {
    tracing_subscriber::fmt::init();

    use facet_styx::Documented;
    use indexmap::IndexMap;

    #[derive(Debug, PartialEq, Eq, facet::Facet)]
    #[repr(u8)]
    #[facet(rename_all = "snake_case")]
    enum SyntaxExpr {
        Template(Box<TemplateSyntaxDecl>),
    }

    #[derive(Debug, PartialEq, Eq, facet::Facet)]
    #[repr(C)]
    struct TemplateSyntaxDecl {
        params: Option<String>,
    }

    #[derive(Debug, PartialEq, Eq, facet::Facet)]
    #[repr(C)]
    struct TemplateDecl {
        syntax: SyntaxExpr,
    }

    #[derive(Debug, PartialEq, Eq, facet::Facet)]
    #[repr(C)]
    struct RuleDecl {
        description: Option<String>,
    }

    #[derive(Debug, PartialEq, Eq, facet::Facet)]
    #[repr(C)]
    struct RulesDecl {
        templates: Option<IndexMap<Documented<String>, TemplateDecl>>,

        #[facet(flatten)]
        rules: IndexMap<Documented<String>, RuleDecl>,
    }

    // Input that exercises both templates AND the flattened rules map.
    // The flattened field causes deserialize_struct_with_flatten to be used,
    // which enters deferred mode and triggers the cleanup path.
    let input = r#"templates {
        ZeroOperand {
            syntax @template{}
        }
    }
    SomeRule {
        description "a rule"
    }"#;

    let _result: RulesDecl = facet_styx::from_str(input).unwrap();
}

#[test]
fn asm_repro_map_with_option_vec_field() {
    use std::collections::BTreeMap;

    // Struct with Option<Vec<String>> mirrors Documented<String>'s exact layout
    #[derive(Debug, PartialEq, Eq, PartialOrd, Ord, facet::Facet)]
    struct Tagged {
        name: String,
        aliases: Option<Vec<String>>,
    }

    let input = r#"{
        first {
            name "hello"
            aliases ("a" "b")
        }
        second {
            name "world"
        }
    }"#;

    let _result: BTreeMap<String, Tagged> = facet_styx::from_str(input).unwrap();
}

#[test]
fn asm_pilot_schema_loads() {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../notes/unified-ast/pilot/asm.repr.styx");

    let _loaded = schema::load_pilot_schema(&path).unwrap();
}
