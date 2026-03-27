#[test]
#[cfg(target_arch = "aarch64")]
#[ignore] // requires pre-generated dump files
fn test_parse_and_roundtrip_scalar_u32() {
    // Find workspace root by walking up from CARGO_MANIFEST_DIR
    let manifest_dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
    let workspace_root = manifest_dir.parent().expect("workspace root");
    let dump_dir = workspace_root.join("target/kajit-stage-dumps");

    let original_asm = std::fs::read_to_string(
        dump_dir.join("postcard__scalar_u32_v0__aarch64__asm.vixen-asm")
    ).expect("original asm file should exist - run: KAJIT_DUMP_STAGES=asm KAJIT_DUMP_FILTER=scalar_u32_v0 cargo nextest run --test corpus -E 'test(~scalar_u32_v0)'");

    let optimized_asm = std::fs::read_to_string(
        dump_dir.join("postcard__scalar_u32_v0__aarch64__asm.alt.vixen-asm"),
    )
    .expect("optimized asm file should exist");

    // Parse original
    let original_program = kajit_emit_text::parse_asm(&original_asm).expect("parse original");

    // Parse optimized
    let optimized_program = kajit_emit_text::parse_asm(&optimized_asm).expect("parse optimized");

    println!("Original: {} items", original_program.items.len());
    println!("Optimized: {} items", optimized_program.items.len());

    // Round-trip test - display and re-parse
    let original_roundtrip = format!("{}", original_program);
    let original_reparsed =
        kajit_emit_text::parse_asm(&original_roundtrip).expect("reparse original");

    let optimized_roundtrip = format!("{}", optimized_program);
    let optimized_reparsed =
        kajit_emit_text::parse_asm(&optimized_roundtrip).expect("reparse optimized");

    // Verify structure is preserved
    assert_eq!(original_program.items.len(), original_reparsed.items.len());
    assert_eq!(
        optimized_program.items.len(),
        optimized_reparsed.items.len()
    );

    // Expand pseudo-instructions (in case there are load_extern)
    let original_expanded =
        kajit_emit_text::expand_pseudo_instructions(original_program, |_symbol| None);
    let optimized_expanded =
        kajit_emit_text::expand_pseudo_instructions(optimized_program, |_symbol| None);

    println!("✓ Parse and round-trip successful");
    println!("✓ Pseudo-instruction expansion successful");

    // Test assembly
    let original_assembled =
        kajit_emit_text::assemble(&original_expanded).expect("assemble original");
    let optimized_assembled =
        kajit_emit_text::assemble(&optimized_expanded).expect("assemble optimized");

    println!("✓ Assembly successful");
    println!("  Original: {} bytes", original_assembled.code.len());
    println!("  Optimized: {} bytes", optimized_assembled.code.len());

    assert!(
        optimized_assembled.code.len() < original_assembled.code.len(),
        "optimized version should be smaller"
    );
}
