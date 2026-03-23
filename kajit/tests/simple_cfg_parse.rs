#[test]
fn simple_cfg_parse() {
    let cfg_text = r#"
cfg_program vregs=5 slots=0 {
  cfg_func @0 f0 entry=b0 {
    data_args: []
    data_results: []
    block b0 params=[] insts=[i0, i1] term=t0 preds=[] succs=[e0]
    block b1 params=[v1] insts=[i2] term=t1 preds=[e0] succs=[]
    inst i0: v0:gpr = const(0x42)
    inst i1: v1:gpr = const(0x7)
    inst i2: store([0:W4]) v1:gpr
    term t0: branch e0
    term t1: return
    edge e0: b0 -> b1 [v1=>v0]
  }
}
"#;

    use kajit::ir::IntrinsicRegistry;
    let registry = IntrinsicRegistry::empty();
    let _decoder = kajit::compile_decoder_from_cfg_mir_text(cfg_text, &registry, false);
    println!("Parse succeeded!");
}
