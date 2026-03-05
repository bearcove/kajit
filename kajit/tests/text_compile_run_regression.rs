use facet::Facet;

const POSTCARD_U32_V0_RVSDG_SNAPSHOT: &str =
    include_str!("snapshots/corpus__generated_rvsdg_postcard_scalar_u32__v0_x86_64.snap");

#[derive(Debug, PartialEq, Facet)]
struct JsonFieldStruct {
    x: u32,
}

#[derive(Debug, PartialEq, Facet)]
struct PostcardOptionStruct {
    x: Option<u32>,
}

fn snapshot_body(snapshot: &'static str) -> &'static str {
    let snapshot = snapshot
        .strip_prefix("---\n")
        .expect("insta snapshot should start with frontmatter");
    let (_, body) = snapshot
        .split_once("\n---\n")
        .expect("insta snapshot frontmatter should end with separator");
    body.trim()
}

#[test]
fn compile_and_run_from_ir_text_snapshot_u32() {
    let ir_text = snapshot_body(POSTCARD_U32_V0_RVSDG_SNAPSHOT);
    let registry = kajit::known_intrinsic_registry();
    let decoder =
        kajit::compile_decoder_from_ir_text(ir_text, <u32 as Facet>::SHAPE, &registry, false);
    let out: u32 = kajit::deserialize(&decoder, &[0x2a]).expect("decode should succeed");
    assert_eq!(out, 42);
}

#[test]
fn compile_and_run_from_cfg_mir_text_with_named_intrinsics_u32() {
    let cfg_text = kajit::debug_cfg_mir_text(<u32 as Facet>::SHAPE, &kajit::json::KajitJson);
    assert!(
        cfg_text.contains("@kajit_json_read_u32"),
        "expected named intrinsic in CFG-MIR text, got:\n{cfg_text}"
    );

    let registry = kajit::symbol_registry_for_shape(<u32 as Facet>::SHAPE);
    let decoder = kajit::compile_decoder_from_cfg_mir_text(&cfg_text, &registry, false);
    let out: u32 = kajit::deserialize(&decoder, b"42").expect("decode should succeed");
    assert_eq!(out, 42);
}

#[test]
fn deserialize_from_ir_text_helper_u32() {
    let ir_text = snapshot_body(POSTCARD_U32_V0_RVSDG_SNAPSHOT);
    let registry = kajit::known_intrinsic_registry();
    let out: u32 = kajit::deserialize_from_ir_text(
        ir_text,
        <u32 as Facet>::SHAPE,
        &registry,
        false,
        &[0x80, 0x01],
    )
    .expect("decode should succeed");
    assert_eq!(out, 128);
}

#[test]
fn deserialize_from_cfg_mir_text_helper_u32() {
    let cfg_text =
        kajit::debug_cfg_mir_text(<u32 as Facet>::SHAPE, &kajit::postcard::KajitPostcard);

    let out: u32 = kajit::deserialize_from_cfg_mir_text(&cfg_text, &[0xac, 0x02])
        .expect("decode should succeed");
    assert_eq!(out, 300);
}

#[test]
fn deserialize_from_ir_text_with_named_json_key_ptr_const() {
    let (ir_text, _) =
        kajit::debug_ir_and_cfg_mir_text(JsonFieldStruct::SHAPE, &kajit::json::KajitJson);
    assert!(
        ir_text.contains("@json_key_ptr.78"),
        "expected named JSON key pointer const in IR text, got:\n{ir_text}"
    );

    let registry = kajit::symbol_registry_for_shape(JsonFieldStruct::SHAPE);
    let out: JsonFieldStruct = kajit::deserialize_from_ir_text(
        &ir_text,
        JsonFieldStruct::SHAPE,
        &registry,
        false,
        br#"{"x":42}"#,
    )
    .expect("decode should succeed");
    assert_eq!(out, JsonFieldStruct { x: 42 });
}

#[test]
fn deserialize_from_cfg_mir_text_with_named_option_init_const() {
    let cfg_text =
        kajit::debug_cfg_mir_text(PostcardOptionStruct::SHAPE, &kajit::postcard::KajitPostcard);
    assert!(
        cfg_text.contains("@option_init_none.") || cfg_text.contains("@option_init_some."),
        "expected named option init const in CFG-MIR text, got:\n{cfg_text}"
    );

    let out: PostcardOptionStruct = kajit::deserialize_from_cfg_mir_text(&cfg_text, &[0x01, 0x2a])
        .expect("decode should succeed");
    assert_eq!(out, PostcardOptionStruct { x: Some(42) });
}

#[cfg(target_arch = "aarch64")]
#[test]
fn emission_trace_snapshot_captures_backend_lowering_path() {
    let trace =
        kajit::emission_trace_text(PostcardOptionStruct::SHAPE, &kajit::postcard::KajitPostcard);
    let edits =
        kajit::regalloc_edits_text(PostcardOptionStruct::SHAPE, &kajit::postcard::KajitPostcard);

    assert!(
        trace.contains("branch_if") || trace.contains("branch_if_zero"),
        "expected branch provenance in emission trace, got:\n{trace}"
    );
    assert!(
        !edits.contains("branch_if") && !edits.contains("branch_if_zero"),
        "regalloc edits dump should not already encode backend lowering details:\n{edits}"
    );

    insta::assert_snapshot!(trace);
}
