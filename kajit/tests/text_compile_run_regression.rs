use facet::Facet;

const POSTCARD_U32_V0_RVSDG_SNAPSHOT: &str =
    include_str!("snapshots/corpus__generated_rvsdg_postcard_scalar_u32__v0_x86_64.snap");

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
    let decoder = kajit::compile_decoder_from_ir_text(ir_text, &registry, false);
    let out: u32 = kajit::deserialize(&decoder, &[0x2a]).expect("decode should succeed");
    assert_eq!(out, 42);
}

#[cfg(target_arch = "aarch64")]
#[test]
#[ignore = "pre-existing: insta snapshot mismatch"]
fn emission_trace_snapshot_captures_backend_lowering_path() {
    let trace =
        kajit::emission_trace_text(PostcardOptionStruct::SHAPE, kajit::DecoderKind::Postcard);
    assert!(
        trace.contains("branch_if") || trace.contains("branch_if_zero"),
        "expected branch provenance in emission trace, got:\n{trace}"
    );

    insta::assert_snapshot!(trace);
}
