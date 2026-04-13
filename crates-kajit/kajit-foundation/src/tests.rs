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
    match loaded.body {
        schema::LoadedReprBody::Modern(repr) => {
            assert_eq!(repr.name, "Demo");
            assert!(
                repr.templates
                    .keys()
                    .any(|name| schema::documented_name(name) == "Keyword")
            );
            assert!(
                repr.rules
                    .keys()
                    .any(|name| schema::documented_name(name) == "FnKw")
            );
        }
        schema::LoadedReprBody::Legacy(_) => panic!("expected modern repr"),
    }
}

#[test]
fn supports_id_entity_and_slot_shapes() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("mini.repr.styx");
    fs::write(
        &path,
        r#"
meta {
    id kajit:test/entity-slot
    version 1
    description "mini schema"
}

repr @module{
    name Mini
    file_ext .k-mini
    contract {
        purpose "mini"
        canonical_identities (block-id)
        round_trip canonical-print
        provenance required
    }
    syntax {
        root Program
        tokens {
            ident @regex("[A-Za-z_][A-Za-z0-9_]*")
            int @regex("[0-9]+")
        }
        rules {
            Program @seq(
                "program"
                "{"
                @field(blocks @repeat(@ref(Block)))
                "}"
            )
            Block @seq(
                "block"
                @field(id @token(int))
                "{"
                @field(insts @repeat(@ref(Inst)))
                "}"
            )
            Inst @seq(
                "inst"
                @field(name @token(ident))
            )
        }
        canonical_print {
            Program "program {\n{blocks}\n}"
            Block "block {id} {\n{insts}\n}"
            Inst "inst {name}"
        }
    }
    common {
        provenance @Prov
    }
    support {
        BlockId @id
        Name @string
    }
    nodes {
        Program @node{
            prov @Prov
            blocks @seq(@Block)
        }
        Block @entity{
            prov @Prov
            id @BlockId
            insts @seq(@Inst)
        }
        Inst @slot{
            prov @Prov
            name @Name
        }
    }
}
"#,
    )
    .unwrap();

    let loaded = schema::load_pilot_schema(&path).unwrap();
    let repr =
        normalize::with_module_doc(normalize::normalize_repr(&loaded.body).unwrap(), loaded.doc);
    let files = render_module::render_repr_poc_files(&[repr]);

    let ast = files
        .iter()
        .find(|file| file.relative_path == "mini/ast.rs")
        .unwrap()
        .contents
        .clone();
    let meta = files
        .iter()
        .find(|file| file.relative_path == "mini/meta.rs")
        .unwrap()
        .contents
        .clone();

    assert!(ast.contains("pub struct BlockId(pub u32);"));
    assert!(ast.contains("pub trait EntityNode {}"));
    assert!(ast.contains("pub trait SlotNode {}"));
    assert!(ast.contains("impl EntityNode for Block {}"));
    assert!(ast.contains("impl SlotNode for Inst {}"));
    assert!(meta.contains("name: \"Block\""));
    assert!(meta.contains("name: \"Inst\""));
    assert!(meta.contains("kind: \"entity\""));
    assert!(meta.contains("kind: \"slot\""));
}

#[test]
fn supports_structured_support_types_and_support_rules() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("supportful.repr.styx");
    fs::write(
        &path,
        r#"
meta {
    id kajit:test/structured-support
    version 1
    description "structured support schema"
}

repr @module{
    name Supportful
    file_ext .k-support
    contract {
        purpose "supportful"
        canonical_identities (program)
        round_trip canonical-print
        provenance required
    }
    syntax {
        root Program
        tokens {
            int @regex("[0-9]+")
        }
        rules {
            Program @seq(
                "program"
                @field(loc @ref(TokenLoc))
                @field(width @ref(Width))
            )
            Count @token(int)
            TokenLoc @seq(
                "@"
                @field(line @ref(Count))
                ":"
                @field(column @ref(Count))
            )
            Width @choice(
                @variant(W1 "w1")
                @variant(W2 "w2")
            )
        }
        canonical_print {
            Program "program {loc} {width}"
        }
    }
    common {
        provenance @Prov
    }
    support {
        Count @id
        TokenLoc @struct{
            line @Count
            column @Count
        }
        Width @enum{
            W1 @unit
            W2 @unit
        }
    }
    nodes {
        Program @node{
            prov @Prov
            loc @TokenLoc
            width @Width
        }
    }
}
"#,
    )
    .unwrap();

    let loaded = schema::load_pilot_schema(&path).unwrap();
    let repr =
        normalize::with_module_doc(normalize::normalize_repr(&loaded.body).unwrap(), loaded.doc);
    let files = render_module::render_repr_poc_files(&[repr]);

    let ast = files
        .iter()
        .find(|file| file.relative_path == "supportful/ast.rs")
        .unwrap()
        .contents
        .clone();
    let parse = files
        .iter()
        .find(|file| file.relative_path == "supportful/parse.rs")
        .unwrap()
        .contents
        .clone();

    assert!(ast.contains("pub struct Count(pub u32);"));
    assert!(ast.contains("pub struct TokenLoc {"));
    assert!(ast.contains("pub line: Count,"));
    assert!(ast.contains("pub column: Count,"));
    assert!(ast.contains("pub enum Width"));
    assert!(parse.contains("let count_parser ="));
    assert!(parse.contains("let token_loc_parser ="));
    assert!(parse.contains("let width_parser ="));
    assert!(parse.contains("Width::W1"));
    assert!(parse.contains("Width::W2"));
}

#[test]
fn supports_pool_and_order_storage_shapes() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("storage.repr.styx");
    fs::write(
        &path,
        r#"
meta {
    id kajit:test/storage-shapes
    version 1
    description "storage schema"
}

repr @module{
    name Storage
    file_ext .k-storage
    contract {
        purpose "storage"
        canonical_identities (block-id)
        round_trip canonical-print
        provenance required
    }
    syntax {
        root Program
        tokens {
            int @regex("[0-9]+")
        }
        rules {
            Program @seq(
                "program"
                "{"
                @field(blocks @repeat(@ref(Block)))
                "}"
            )
            Block @seq(
                "block"
                @field(id @token(int))
                "["
                @field(succs @repeat(@ref(BlockId) {sep ","}))
                "]"
            )
            BlockId @token(int)
        }
        canonical_print {
            Program "program {\n{blocks}\n}"
            Block "block {id} [{succs:, }]"
        }
    }
    common {
        provenance @Prov
    }
    support {
        BlockId @id
    }
    nodes {
        Program @node{
            prov @Prov
            blocks @arena(@Block @key(@BlockId))
        }
        Block @entity{
            prov @Prov
            id @BlockId
            succs @order(@ref_to(@BlockId @Block))
        }
    }
}
"#,
    )
    .unwrap();

    let loaded = schema::load_pilot_schema(&path).unwrap();
    let repr =
        normalize::with_module_doc(normalize::normalize_repr(&loaded.body).unwrap(), loaded.doc);
    let files = render_module::render_repr_poc_files(&[repr]);

    let ast = files
        .iter()
        .find(|file| file.relative_path == "storage/ast.rs")
        .unwrap()
        .contents
        .clone();
    let meta = files
        .iter()
        .find(|file| file.relative_path == "storage/meta.rs")
        .unwrap()
        .contents
        .clone();

    assert!(ast.contains("pub blocks: super::super::Order<BlockId>,"));
    assert!(ast.contains("blocks: super::super::ArenaStorage<BlockId, Block>,"));
    assert!(ast.contains("pub succs: super::super::Order<BlockId>,"));
    assert!(meta.contains("owner: \"Program\""));
    assert!(meta.contains("field: \"blocks\""));
    assert!(meta.contains("kind: \"arena<Block key=BlockId>\""));
    assert!(meta.contains("owner: \"Block\""));
    assert!(meta.contains("field: \"succs\""));
    assert!(meta.contains("kind: \"order<ref<BlockId -> Block>>\""));
    assert!(meta.contains("pub static POOLS: &[PoolSpec] = &["));
    assert!(meta.contains("field: \"blocks\""));
    assert!(meta.contains("item: \"Block\""));
    assert!(meta.contains("key: \"BlockId\""));
    assert!(meta.contains("pub static REFS: &[RefSpec] = &["));
    assert!(meta.contains("field: \"succs\""));
    assert!(meta.contains("id: \"BlockId\""));
    assert!(meta.contains("target: \"Block\""));
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
    let repr = normalize::with_module_doc(normalize::normalize_repr(&loaded).unwrap(), loaded.doc);
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
