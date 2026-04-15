use std::path::PathBuf;

use crate::schema;

#[test]
fn all_pilot_schemas_load() {
    let _ = tracing_subscriber::fmt::try_init();

    let schemas_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../notes/unified-ast/pilot")
        .canonicalize()
        .unwrap();
    for name in ["hir", "ir", "mir", "lir", "asm"] {
        let path = schemas_dir.join(format!("{name}.repr.styx"));
        schema::read_from_file(&path).unwrap_or_else(|e| panic!("failed to load {name}: {e}"));
    }
}

// Exact pattern from ModernRulesDecl: Option<IndexMap<Documented<String>, T>>
// where T is a struct containing an enum with a Box variant.
// This reproduces the double-free in facet-reflect deferred cleanup.
#[test]
fn asm_repro_option_indexmap_documented_key() {
    let _ = tracing_subscriber::fmt::try_init();

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

// Minimal repro of the schema_loads SIGSEGV: nested failure deep inside a
// Template variant triggers cleanup that severs a pending map key in a
// non-flatten `Option<IndexMap<Documented<String>, _>>`. The key String
// backing pointer ends up pointing at 0x4 (discriminant?) and free()ing it
// aborts.
#[test]
fn asm_repro_nested_failure_severs_templates_key() {
    let _ = tracing_subscriber::fmt::try_init();

    use facet_styx::Documented;
    use indexmap::IndexMap;

    #[derive(Debug, PartialEq, Eq, facet::Facet)]
    #[repr(C)]
    struct TemplateSyntaxDecl {
        required: String,
    }

    #[derive(Debug, PartialEq, Eq, facet::Facet)]
    #[repr(u8)]
    #[facet(rename_all = "snake_case")]
    enum SyntaxExpr {
        Template(Box<TemplateSyntaxDecl>),
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

    // `@template{}` is missing `required` → inner struct fails → cleanup
    // walks up through SyntaxExpr → TemplateDecl → templates map entry.
    let input = r#"templates {
        ZeroOperand {
            syntax @template{}
        }
    }"#;

    // Expected: Err (missing field). Currently: SIGABRT / SEGV.
    let _result: Result<RulesDecl, _> = facet_styx::from_str(input);
}

// Minimal double-free repro: #[facet(flatten)] on an IndexMap whose value type
// has a required field. When the input provides an entry with an empty value
// (`ZeroOperand {}`), `require_full_initialization` fails during
// `finish_deferred`. The ensuing `cleanup_stored_frames_on_error` path in
// facet-reflect drops the Map (Field(0)) — but the Map already owns the key
// "ZeroOperand" from `complete_map_key_frame`, AND the cleanup also drops the
// already-moved key again, producing a double-free / SEGV under ASAN.
//
// Stripped back from the full schema repro: no Documented, no Option, no Box,
// no enum, no untagged — just flatten + map + missing required field.
#[test]
fn asm_repro_flatten_map_missing_field() {
    let _ = tracing_subscriber::fmt::try_init();

    use indexmap::IndexMap;

    #[derive(Debug, PartialEq, Eq, facet::Facet)]
    #[repr(C)]
    struct RuleValue {
        required: String,
    }

    #[derive(Debug, PartialEq, Eq, facet::Facet)]
    #[repr(C)]
    struct RulesDecl {
        #[facet(flatten)]
        rules: IndexMap<String, RuleValue>,
    }

    let input = r#"ZeroOperand {}"#;

    // Expected to return Err (missing `required` field) — but currently
    // double-frees before it can return.
    let _result: Result<RulesDecl, _> = facet_styx::from_str(input);
}
