use std::fs;

use crate::normalize;
use crate::render_module;
use crate::schema;

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
