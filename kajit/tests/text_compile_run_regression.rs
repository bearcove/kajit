use facet::Facet;

const POSTCARD_U32_V0_RVSDG_SNAPSHOT: &str =
    include_str!("snapshots/corpus__generated_rvsdg_postcard_scalar_u32__v0_x86_64.snap");

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

    let decoder = kajit::compile_decoder_from_cfg_mir_text(&cfg_text, false);
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
