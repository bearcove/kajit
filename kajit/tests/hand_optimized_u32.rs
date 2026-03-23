// Hand-optimized CFG-MIR for u32 varint decoding
// This is an unrolled version with minimal blocks (16 blocks vs 43 in generated version)

#[test]
#[ignore = "WIP: regalloc issue with block parameters"]
fn hand_optimized_u32_varint() {
    let cfg_text = r#"
cfg_program vregs=60 slots=0 {
  cfg_func @0 f0 entry=b0 {
    data_args: []
    data_results: []

    block b0 params=[] insts=[i0, i1, i2, i3, i4] term=t0 preds=[] succs=[e0]
    block b1 params=[] insts=[i5, i6] term=t1 preds=[e0] succs=[e2, e1]
    block b2 params=[] insts=[i7, i8, i9, i10, i11, i12, i13] term=t2 preds=[e1] succs=[e4, e3]
    block b3 params=[] insts=[i14, i15] term=t3 preds=[e3] succs=[e5, e6]
    block b4 params=[] insts=[i16, i17, i18, i19, i20, i21, i22, i23] term=t4 preds=[e6] succs=[e8, e7]
    block b5 params=[] insts=[i24, i25] term=t5 preds=[e7] succs=[e9, e10]
    block b6 params=[] insts=[i26, i27, i28, i29, i30, i31, i32, i33] term=t6 preds=[e10] succs=[e12, e11]
    block b7 params=[] insts=[i34, i35] term=t7 preds=[e11] succs=[e13, e14]
    block b8 params=[] insts=[i36, i37, i38, i39, i40, i41, i42, i43] term=t8 preds=[e14] succs=[e16, e15]
    block b9 params=[] insts=[i44, i45] term=t9 preds=[e15] succs=[e17, e18]
    block b10 params=[] insts=[i46, i47, i48, i49, i50, i51, i52, i53] term=t10 preds=[e18] succs=[e19, e20]
    block b11 params=[] insts=[] term=t11 preds=[e20] succs=[]
    block b12 params=[v53] insts=[i54, i55, i56, i57] term=t12 preds=[e4, e8, e12, e16, e19] succs=[e21, e22]
    block b13 params=[] insts=[] term=t13 preds=[e21] succs=[]
    block b14 params=[] insts=[i58] term=t14 preds=[e22] succs=[]
    block b15 params=[] insts=[] term=t15 preds=[e2, e5, e9, e13, e17] succs=[]

    inst i0: v0:gpr = save_cursor
    inst i1: v1:gpr = save_input_end
    inst i2: v2:gpr = Sub v1:gpr, v0:gpr
    inst i3: v3:gpr = const(0x0)
    inst i4: v4:gpr = const(0x1)

    inst i5: v5:gpr = Add v0:gpr, v4:gpr
    inst i6: v6:gpr = CmpGt v5:gpr, v2:gpr

    inst i7: v7:gpr = load_addr([W1]) v0:gpr
    inst i8: v8:gpr = Add v0:gpr, v4:gpr
    inst i9: restore_cursor v8:gpr
    inst i10: v9:gpr = const(0x7f)
    inst i11: v10:gpr = And v7:gpr, v9:gpr
    inst i12: v11:gpr = const(0x80)
    inst i13: v12:gpr = And v7:gpr, v11:gpr

    inst i14: v13:gpr = Add v8:gpr, v4:gpr
    inst i15: v14:gpr = CmpGt v13:gpr, v2:gpr

    inst i16: v15:gpr = load_addr([W1]) v8:gpr
    inst i17: v16:gpr = Add v8:gpr, v4:gpr
    inst i18: restore_cursor v16:gpr
    inst i19: v17:gpr = And v15:gpr, v9:gpr
    inst i20: v18:gpr = const(0x7)
    inst i21: v19:gpr = Shl v17:gpr, v18:gpr
    inst i22: v20:gpr = Or v10:gpr, v19:gpr
    inst i23: v21:gpr = And v15:gpr, v11:gpr

    inst i24: v22:gpr = Add v16:gpr, v4:gpr
    inst i25: v23:gpr = CmpGt v22:gpr, v2:gpr

    inst i26: v24:gpr = load_addr([W1]) v16:gpr
    inst i27: v25:gpr = Add v16:gpr, v4:gpr
    inst i28: restore_cursor v25:gpr
    inst i29: v26:gpr = And v24:gpr, v9:gpr
    inst i30: v27:gpr = const(0xe)
    inst i31: v28:gpr = Shl v26:gpr, v27:gpr
    inst i32: v29:gpr = Or v20:gpr, v28:gpr
    inst i33: v30:gpr = And v24:gpr, v11:gpr

    inst i34: v31:gpr = Add v25:gpr, v4:gpr
    inst i35: v32:gpr = CmpGt v31:gpr, v2:gpr

    inst i36: v33:gpr = load_addr([W1]) v25:gpr
    inst i37: v34:gpr = Add v25:gpr, v4:gpr
    inst i38: restore_cursor v34:gpr
    inst i39: v35:gpr = And v33:gpr, v9:gpr
    inst i40: v36:gpr = const(0x15)
    inst i41: v37:gpr = Shl v35:gpr, v36:gpr
    inst i42: v38:gpr = Or v29:gpr, v37:gpr
    inst i43: v39:gpr = And v33:gpr, v11:gpr

    inst i44: v40:gpr = Add v34:gpr, v4:gpr
    inst i45: v41:gpr = CmpGt v40:gpr, v2:gpr

    inst i46: v42:gpr = load_addr([W1]) v34:gpr
    inst i47: v43:gpr = Add v34:gpr, v4:gpr
    inst i48: restore_cursor v43:gpr
    inst i49: v44:gpr = And v42:gpr, v9:gpr
    inst i50: v45:gpr = const(0x1c)
    inst i51: v46:gpr = Shl v44:gpr, v45:gpr
    inst i52: v47:gpr = Or v38:gpr, v46:gpr
    inst i53: v48:gpr = And v42:gpr, v11:gpr

    inst i54: v49:gpr = const(0x20)
    inst i55: v50:gpr = Shr v53:gpr, v49:gpr
    inst i56: v51:gpr = const(0x0)
    inst i57: v52:gpr = CmpNe v50:gpr, v51:gpr

    inst i58: store([0:W4]) v53:gpr

    term t0: branch e0
    term t1: branch_if v6 -> e2, fallthrough e1
    term t2: branch_if_zero v12 -> e4, fallthrough e3
    term t3: branch_if v14 -> e5, fallthrough e6
    term t4: branch_if_zero v21 -> e8, fallthrough e7
    term t5: branch_if v23 -> e9, fallthrough e10
    term t6: branch_if_zero v30 -> e12, fallthrough e11
    term t7: branch_if v32 -> e13, fallthrough e14
    term t8: branch_if_zero v39 -> e16, fallthrough e15
    term t9: branch_if v41 -> e17, fallthrough e18
    term t10: branch_if_zero v48 -> e19, fallthrough e20
    term t11: error_exit(InvalidVarint)
    term t12: branch_if v52 -> e21, fallthrough e22
    term t13: error_exit(NumberOutOfRange)
    term t14: return
    term t15: error_exit(UnexpectedEof)

    edge e0: b0 -> b1 []
    edge e1: b1 -> b2 []
    edge e2: b1 -> b15 []
    edge e3: b2 -> b3 []
    edge e4: b2 -> b12 [v53=>v10]
    edge e5: b3 -> b15 []
    edge e6: b3 -> b4 []
    edge e7: b4 -> b5 []
    edge e8: b4 -> b12 [v53=>v20]
    edge e9: b5 -> b15 []
    edge e10: b5 -> b6 []
    edge e11: b6 -> b7 []
    edge e12: b6 -> b12 [v53=>v29]
    edge e13: b7 -> b15 []
    edge e14: b7 -> b8 []
    edge e15: b8 -> b9 []
    edge e16: b8 -> b12 [v53=>v38]
    edge e17: b9 -> b15 []
    edge e18: b9 -> b10 []
    edge e19: b10 -> b12 [v53=>v47]
    edge e20: b10 -> b11 []
    edge e21: b12 -> b13 []
    edge e22: b12 -> b14 []
  }
}
"#;

    use kajit::ir::IntrinsicRegistry;
    let registry = IntrinsicRegistry::empty();
    let decoder = kajit::compile_decoder_from_cfg_mir_text(cfg_text, &registry, false);

    // Test that it works correctly
    let test_cases = vec![
        (vec![0x00], 0u32),
        (vec![0x01], 1u32),
        (vec![0x7f], 127u32),
        (vec![0x80, 0x01], 128u32),
        (vec![0xff, 0xff, 0xff, 0xff, 0x0f], u32::MAX),
    ];

    for (input, expected) in test_cases {
        let result: u32 = kajit::deserialize(&decoder, &input).unwrap();
        assert_eq!(result, expected, "Failed for input {:?}", input);
    }
}
