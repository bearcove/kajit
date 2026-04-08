use super::*;
use kajit_ir::{DebugScope, DebugScopeKind, IrBuilder, IrOp, LambdaId, PortSource, VReg, Width};

#[test]
fn linearize_simple_chain() {
    // Const(42) → StoreToAddr
    let mut builder = IrBuilder::new("u32", 0);
    {
        let mut rb = builder.root_region();
        let data = rb.const_val(42);
        let addr = rb.const_val(0);
        rb.store_to_addr(addr, data, Width::W4);
        rb.set_results(&[]);
    }
    let mut func = builder.finish();
    let ir = linearize(&mut func);

    // Expected: FuncStart, Const(42), Const(0), StoreToAddr, FuncEnd
    assert!(matches!(ir.ops[0], LinearOp::FuncStart { .. }));
    assert!(matches!(ir.ops[1], LinearOp::Const { .. }));
    assert!(matches!(ir.ops[2], LinearOp::Const { .. }));
    assert!(matches!(
        ir.ops[3],
        LinearOp::StoreToAddr {
            width: Width::W4,
            ..
        }
    ));
    assert!(matches!(ir.ops[4], LinearOp::FuncEnd));
    assert_eq!(ir.ops.len(), 5);
}

#[test]
fn linearize_preserves_debug_scope_provenance() {
    let mut builder = IrBuilder::new("u32", 0);
    let (const_node, output_index, root_scope) = {
        let mut rb = builder.root_region();
        let value = rb.const_val(42);
        rb.set_results(&[value]);
        let output_ref = match value {
            PortSource::Node(output_ref) => output_ref,
            other => panic!("expected node output, got {other:?}"),
        };
        (output_ref.node, output_ref.index as usize, rb.debug_scope())
    };

    let mut func = builder.finish();
    let value_vreg = func.nodes[const_node].outputs[output_index]
        .vreg
        .expect("expected vreg on const output");
    let extra_scope = func.debug_scopes.push(DebugScope {
        parent: Some(root_scope),
        kind: DebugScopeKind::ThetaBody,
    });
    func.nodes[const_node].debug_scope = extra_scope;
    func.nodes[const_node].outputs[0].debug_scope = root_scope;

    let linear = linearize(&mut func);
    assert_eq!(linear.debug.root_scope, Some(root_scope));
    assert_eq!(linear.debug.scopes.len(), func.debug_scopes.len());
    assert_eq!(
        linear.debug.vreg_scopes[value_vreg.index()],
        Some(root_scope)
    );

    let const_scope = linear
        .ops
        .iter()
        .zip(linear.debug.op_scopes.iter())
        .find_map(|(op, scope)| match op {
            LinearOp::Const { dst, .. } if *dst == value_vreg => *scope,
            _ => None,
        });
    assert_eq!(const_scope, Some(extra_scope));
}

#[test]
fn linearize_gamma_two_branches() {
    // Gamma with predicate, 2 branches:
    //   branch 0: const 42 → result
    //   branch 1: const 99 → result
    let mut builder = IrBuilder::new("u32", 0);
    {
        let mut rb = builder.root_region();
        let pred = rb.const_val(0);
        let results = rb.gamma(pred, &[], 2, |branch_idx, bb| {
            let val = if branch_idx == 0 {
                bb.const_val(42)
            } else {
                bb.const_val(99)
            };
            bb.set_results(&[val]);
        });
        assert_eq!(results.len(), 1);
        let addr = rb.const_val(0);
        rb.store_to_addr(addr, results[0], Width::W4);
        rb.set_results(&[]);
    }
    let mut func = builder.finish();
    let ir = linearize(&mut func);

    // Verify structure: FuncStart, Const(pred), BranchIfZero, Branch,
    //   Label(0), Const(42), Copy, Branch(merge), Label(1), Const(99), Copy, Label(merge), ...
    let display = format!("{ir}");
    assert!(
        display.contains("br_zero"),
        "should have BranchIfZero for 2-branch gamma:\n{display}"
    );
    assert!(
        display.contains("const 42"),
        "branch 0 should produce 42:\n{display}"
    );
    assert!(
        display.contains("const 99"),
        "branch 1 should produce 99:\n{display}"
    );
}

#[test]
fn linearize_theta_loop() {
    // Theta: count down from 5 to 0.
    // loop_var = counter
    // body: counter - 1, predicate = counter > 0
    let mut builder = IrBuilder::new("u32", 0);
    {
        let mut rb = builder.root_region();
        let init_count = rb.const_val(5);
        let one = rb.const_val(1);
        let _results = rb.theta(&[init_count, one], |bb| {
            let args = bb.region_args(2);
            let counter = args[0];
            let one = args[1];
            let new_counter = bb.binop(IrOp::Sub, counter, one);
            // predicate = new_counter (0=exit)
            bb.set_results(&[new_counter, new_counter, one]);
        });
        rb.set_results(&[]);
    }
    let mut func = builder.finish();
    let ir = linearize(&mut func);

    let display = format!("{ir}");
    assert!(
        display.contains("br_if"),
        "should have BranchIf back-edge:\n{display}"
    );
    assert!(
        display.contains("Sub"),
        "should have subtraction:\n{display}"
    );
}

#[test]
fn linearize_call_intrinsic() {
    use kajit_ir::FnPtr;

    unsafe extern "C" fn dummy_intrinsic(_ctx: *mut core::ffi::c_void) {}

    let mut builder = IrBuilder::new("bool", 0);
    {
        let mut rb = builder.root_region();
        rb.call_intrinsic(FnPtr(dummy_intrinsic as *const () as usize), &[], false);
        rb.set_results(&[]);
    }
    let mut func = builder.finish();
    let ir = linearize(&mut func);

    let has_call = ir
        .ops
        .iter()
        .any(|op| matches!(op, LinearOp::CallIntrinsic { .. }));
    assert!(has_call, "should contain CallIntrinsic");
}

#[test]
fn linearize_display() {
    let mut builder = IrBuilder::new("u32", 0);
    {
        let mut rb = builder.root_region();
        let data = rb.const_val(42);
        let addr = rb.const_val(0u64);
        rb.store_to_addr(addr, data, Width::W4);
        rb.set_results(&[]);
    }
    let mut func = builder.finish();
    let ir = linearize(&mut func);

    let display = format!("{ir}");
    assert!(
        display.contains("func"),
        "display should start with func:\n{display}"
    );
    assert!(
        display.contains("store_addr [W4]"),
        "display should contain store:\n{display}"
    );
    assert!(
        display.contains("end"),
        "display should end with end:\n{display}"
    );
}

#[test]
fn optimize_linear_ops_elides_dead_copy_chain() {
    let v0 = VReg::new(0);
    let v1 = VReg::new(1);
    let v2 = VReg::new(2);
    let mut ops = vec![
        LinearOp::FuncStart {
            lambda_id: LambdaId::new(0),
            label: "u32".into(),
            output_size: 0,
            data_args: vec![],
            data_results: vec![],
        },
        LinearOp::Const { dst: v0, value: 7 },
        LinearOp::Copy { dst: v1, src: v0 },
        LinearOp::Copy { dst: v2, src: v1 },
        LinearOp::StoreToAddr {
            addr: v0,
            src: v2,
            width: Width::W4,
        },
        LinearOp::FuncEnd,
    ];

    let mut op_scopes = vec![None; ops.len()];
    let mut op_values = vec![None; ops.len()];
    optimize_linear_ops(&mut ops, &mut op_scopes, &mut op_values);

    let copy_count = ops
        .iter()
        .filter(|op| matches!(op, LinearOp::Copy { .. }))
        .count();
    assert_eq!(copy_count, 0, "dead copy chain should be eliminated");
    let write_src = ops.iter().find_map(|op| match op {
        LinearOp::StoreToAddr { src, .. } => Some(*src),
        _ => None,
    });
    assert_eq!(write_src, Some(v0), "store should use propagated source");
}

#[test]
fn optimize_linear_ops_keeps_copy_feeding_func_end_result() {
    let v0 = VReg::new(0);
    let v1 = VReg::new(1);
    let mut ops = vec![
        LinearOp::FuncStart {
            lambda_id: LambdaId::new(0),
            label: "u32".into(),
            output_size: 0,
            data_args: vec![],
            data_results: vec![v1],
        },
        LinearOp::Const { dst: v0, value: 9 },
        LinearOp::Copy { dst: v1, src: v0 },
        LinearOp::FuncEnd,
    ];

    let mut op_scopes = vec![None; ops.len()];
    let mut op_values = vec![None; ops.len()];
    optimize_linear_ops(&mut ops, &mut op_scopes, &mut op_values);

    assert!(
        ops.iter()
            .any(|op| matches!(op, LinearOp::Copy { dst, src } if *dst == v1 && *src == v0)),
        "copy into function result vreg must be preserved"
    );
}

#[test]
fn optimize_linear_ops_keeps_debug_values_aligned_with_rewritten_ops() {
    let v0 = VReg::new(0);
    let v1 = VReg::new(1);
    let v2 = VReg::new(2);
    let debug_value = DebugValueId::new(0);
    let mut ops = vec![
        LinearOp::FuncStart {
            lambda_id: LambdaId::new(0),
            label: "u32".into(),
            output_size: 0,
            data_args: vec![],
            data_results: vec![],
        },
        LinearOp::Const { dst: v0, value: 7 },
        LinearOp::Copy { dst: v1, src: v0 },
        LinearOp::Copy { dst: v2, src: v1 },
        LinearOp::StoreToAddr {
            addr: v0,
            src: v2,
            width: Width::W4,
        },
        LinearOp::FuncEnd,
    ];
    let mut op_scopes = vec![None; ops.len()];
    let mut op_values = vec![None; ops.len()];
    op_values[4] = Some(debug_value);

    optimize_linear_ops(&mut ops, &mut op_scopes, &mut op_values);

    assert_eq!(ops.len(), op_values.len(), "debug values must stay aligned");
    let write_index = ops
        .iter()
        .position(|op| matches!(op, LinearOp::StoreToAddr { .. }))
        .expect("optimized ops should still contain store");
    assert_eq!(
        op_values[write_index],
        Some(debug_value),
        "semantic debug value should stay attached to the write op",
    );
}

#[test]
fn linearize_theta_gamma_passthrough_after_slot2reg() {
    // Theta with gamma inside, one branch doesn't modify the slot.
    // After slot2reg, the slot becomes a loop-carried variable.
    let input = r#"
lambda @0 (shape: "test") {
  region {
    args: [%ms]
    n0 = Const(0x0) [] -> [v0]
    n1 = WriteToSlot(0) [v0, %ms:arg] -> [%ms]
    n14 = theta [%ms:n1] {
      region {
        args: [%ms]
        n2 = ReadFromSlot(0) [%ms:arg] -> [v1, %ms]
        n3 = Const(0x4) [] -> [v2]
        n4 = CmpNe [v1, v2] -> [v3]
        n11 = gamma [
          pred: v3
          in0: %ms:n2
        ] {
          branch 0:
            region {
              args: [%ms]
              n5 = ReadFromSlot(0) [%ms:arg] -> [v4, %ms]
              n6 = Const(0x1) [] -> [v5]
              n7 = Add [v4, v5] -> [v6]
              n8 = WriteToSlot(0) [v6, %ms:n5] -> [%ms]
              results: [%ms:n8]
            }
          branch 1:
            region {
              args: [%ms]
              results: [%ms:arg]
            }
        } -> [%ms]
        n12 = Const(0x0) [] -> [v7]
        results: [v7, %ms:n11]
      }
    } -> [%ms]
    n13 = ReadFromSlot(0) [%ms:n14] -> [v8, %ms]
    n15 = StoreToAddr(W4) [v7, v8, %ms:n13] -> [%ms]
    results: [%ms:n15]
  }
}
"#;
    let registry = kajit_ir::IntrinsicRegistry::empty();
    let mut func = kajit_ir_text::parse_ir(input, &registry).unwrap();
    kajit_ir::slot2reg::slot_to_reg(&mut func);
    let _ir = linearize(&mut func);
}

#[test]
fn linearize_theta_shared_predicate_and_loopvar() {
    // Theta where a gamma output is used both as predicate AND as a
    // loop-carried variable result — the pattern from the real array
    // decoder that triggers v_N from v_N in register allocation.
    let input = r#"
lambda @0 (shape: "test") {
  region {
    args: [%ms]
    n0 = Const(0x0) [] -> [v0]
    n1 = Const(0x1) [] -> [v1]
    n10 = theta [v0, v1, %ms:arg] {
      region {
        args: [arg0, arg1, %ms]
        n2 = Const(0x4) [] -> [v2]
        n3 = CmpNe [arg0, v2] -> [v3]
        n8 = gamma [
          pred: v3
          in0: arg0
          in1: arg1
          in2: %ms:arg
        ] {
          branch 0:
            region {
              args: [arg0, arg1, %ms]
              n4 = Const(0x1) [] -> [v4]
              n5 = Add [arg0, v4] -> [v5]
              results: [v5, arg1, %ms:arg]
            }
          branch 1:
            region {
              args: [arg0, arg1, %ms]
              results: [arg0, arg1, %ms:arg]
            }
        } -> [v6, v7, %ms]
        results: [v7, v6, v7, %ms:n8]
      }
    } -> [v8, v9, %ms]
    n9 = StoreToAddr(W4) [v8, v8, %ms:n10] -> [%ms]
    results: [%ms:n9]
  }
}
"#;
    let registry = kajit_ir::IntrinsicRegistry::empty();
    let mut func = kajit_ir_text::parse_ir(input, &registry).unwrap();
    let _ir = linearize(&mut func);
}
