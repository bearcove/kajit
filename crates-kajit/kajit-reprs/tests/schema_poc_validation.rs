use kajit_reprs::mir::validate_root_text;

#[test]
fn mir_validation_accepts_resolved_storage_relationships() {
    let source = r#"cfg_program vregs=0 slots=0 {
cfg_func @0 f0 entry=b0 {
data_args: []
data_results: []
block b0 params=[] insts=[] term=t0 preds=[] succs=[]
term t0: return
}
}"#;

    validate_root_text(source).expect("valid MIR should validate");
}

#[test]
fn mir_validation_rejects_unresolved_block_refs() {
    let source = r#"cfg_program vregs=0 slots=0 {
cfg_func @0 f0 entry=b0 {
data_args: []
data_results: []
block b0 params=[] insts=[] term=t0 preds=[] succs=[1]
term t0: return
edge e1: b0 -> b9 []
}
}"#;

    let error = validate_root_text(source).expect_err("invalid MIR should fail validation");
    assert!(error.contains("Edge.to references"), "{error}");
}
