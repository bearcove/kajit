use kajit_ir::{IntrinsicRegistry, slot2reg::slot_to_reg, verify};

fn run_slot2reg(input: &str) -> String {
    let registry = IntrinsicRegistry::empty();
    let mut func = crate::parse_ir(input, &registry).unwrap();
    slot_to_reg(&mut func);
    if let Err(e) = verify(&func) {
        panic!("verification failed after slot2reg: {e}");
    }
    format!("{}", func.display_with_registry(&registry))
}

// ── 1. Write then read in the same region → read replaced by const ──

#[test]
fn write_then_read_same_region() {
    let ir = run_slot2reg(
        r#"
lambda @0 (shape: "test") {
  region {
    args: [%cs, %os]
    n0 = Const(0x2a) [] -> [v0]
    n1 = WriteToSlot(0) [v0, %cs:arg] -> [%cs]
    n2 = ReadFromSlot(0) [%cs:n1] -> [v1, %cs]
    n3 = WriteToField(offset=0, W4) [v1, %os:arg] -> [%os]
    results: [%cs:n2, %os:n3]
  }
}
"#,
    );
    insta::assert_snapshot!(ir);
}

// ── 2. Slot read inside gamma → becomes passthrough arg ──

#[test]
fn slot_read_inside_gamma() {
    let ir = run_slot2reg(
        r#"
lambda @0 (shape: "test") {
  region {
    args: [%cs, %os]
    n0 = Const(0x2a) [] -> [v0]
    n1 = WriteToSlot(0) [v0, %cs:arg] -> [%cs]
    n2 = Const(0x0) [] -> [v1]
    n7 = gamma [
      pred: v1
      in0: %cs:n1
      in1: %os:arg
    ] {
      branch 0:
        region {
          args: [%cs, %os]
          n3 = ReadFromSlot(0) [%cs:arg] -> [v2, %cs]
          n4 = WriteToField(offset=0, W4) [v2, %os:arg] -> [%os]
          results: [%cs:n3, %os:n4]
        }
      branch 1:
        region {
          args: [%cs, %os]
          n5 = ReadFromSlot(0) [%cs:arg] -> [v3, %cs]
          n6 = WriteToField(offset=0, W4) [v3, %os:arg] -> [%os]
          results: [%cs:n5, %os:n6]
        }
    } -> [%cs, %os]
    results: [%cs:n7, %os:n7]
  }
}
"#,
    );
    insta::assert_snapshot!(ir);
}

// ── 3. Slot written in one gamma branch, read after ──

#[test]
fn slot_written_in_gamma_branch() {
    let ir = run_slot2reg(
        r#"
lambda @0 (shape: "test") {
  region {
    args: [%cs, %os]
    n0 = Const(0x0) [] -> [v0]
    n1 = WriteToSlot(0) [v0, %cs:arg] -> [%cs]
    n2 = Const(0x0) [] -> [v1]
    n7 = gamma [
      pred: v1
      in0: %cs:n1
      in1: %os:arg
    ] {
      branch 0:
        region {
          args: [%cs, %os]
          results: [%cs:arg, %os:arg]
        }
      branch 1:
        region {
          args: [%cs, %os]
          n3 = Const(0x63) [] -> [v2]
          n4 = WriteToSlot(0) [v2, %cs:arg] -> [%cs]
          results: [%cs:n4, %os:arg]
        }
    } -> [%cs, %os]
    n8 = ReadFromSlot(0) [%cs:n7] -> [v3, %cs]
    n9 = WriteToField(offset=0, W4) [v3, %os:n7] -> [%os]
    results: [%cs:n8, %os:n9]
  }
}
"#,
    );
    insta::assert_snapshot!(ir);
}

// ── 4. Slot threads through theta loop ──

#[test]
fn slot_read_inside_theta() {
    let ir = run_slot2reg(
        r#"
lambda @0 (shape: "test") {
  region {
    args: [%cs, %os]
    n0 = Const(0xa) [] -> [v0]
    n1 = WriteToSlot(0) [v0, %cs:arg] -> [%cs]
    n6 = theta [%cs:n1, %os:arg] {
      region {
        args: [%cs, %os]
        n2 = ReadFromSlot(0) [%cs:arg] -> [v1, %cs]
        n3 = WriteToField(offset=0, W4) [v1, %os:arg] -> [%os]
        n4 = Const(0x0) [] -> [v2]
        results: [v2, %cs:n2, %os:n3]
      }
    } -> [%cs, %os]
    results: [%cs:n6, %os:n6]
  }
}
"#,
    );
    insta::assert_snapshot!(ir);
}

// ── 5. Slot modified inside theta body (counter pattern) ──

#[test]
fn slot_modified_in_theta() {
    let ir = run_slot2reg(
        r#"
lambda @0 (shape: "test") {
  region {
    args: [%cs, %os]
    n0 = Const(0x0) [] -> [v0]
    n1 = WriteToSlot(0) [v0, %cs:arg] -> [%cs]
    n8 = theta [%cs:n1, %os:arg] {
      region {
        args: [%cs, %os]
        n2 = ReadFromSlot(0) [%cs:arg] -> [v1, %cs]
        n3 = Const(0x1) [] -> [v2]
        n4 = Add [v1, v2] -> [v3]
        n5 = WriteToSlot(0) [v3, %cs:n2] -> [%cs]
        n6 = Const(0xa) [] -> [v4]
        n7 = Sub [v4, v3] -> [v5]
        results: [v5, %cs:n5, %os:arg]
      }
    } -> [%cs, %os]
    n9 = ReadFromSlot(0) [%cs:n8] -> [v6, %cs]
    n10 = WriteToField(offset=0, W4) [v6, %os:n8] -> [%os]
    results: [%cs:n9, %os:n10]
  }
}
"#,
    );
    insta::assert_snapshot!(ir);
}

// ── 6. Two slots through gamma ──

#[test]
fn two_slots_through_gamma() {
    let ir = run_slot2reg(
        r#"
lambda @0 (shape: "test") {
  region {
    args: [%cs, %os]
    n0 = Const(0x1) [] -> [v0]
    n1 = Const(0x2) [] -> [v1]
    n2 = WriteToSlot(0) [v0, %cs:arg] -> [%cs]
    n3 = WriteToSlot(1) [v1, %cs:n2] -> [%cs]
    n4 = Const(0x0) [] -> [v2]
    n11 = gamma [
      pred: v2
      in0: %cs:n3
      in1: %os:arg
    ] {
      branch 0:
        region {
          args: [%cs, %os]
          n5 = ReadFromSlot(0) [%cs:arg] -> [v3, %cs]
          n6 = ReadFromSlot(1) [%cs:n5] -> [v4, %cs]
          n7 = Add [v3, v4] -> [v5]
          n8 = WriteToField(offset=0, W4) [v5, %os:arg] -> [%os]
          results: [%cs:n6, %os:n8]
        }
      branch 1:
        region {
          args: [%cs, %os]
          n9 = ReadFromSlot(0) [%cs:arg] -> [v6, %cs]
          n10 = ReadFromSlot(1) [%cs:n9] -> [v7, %cs]
          results: [%cs:n10, %os:arg]
        }
    } -> [%cs, %os]
    results: [%cs:n11, %os:n11]
  }
}
"#,
    );
    insta::assert_snapshot!(ir);
}

// ── 7. Slot first written inside gamma (not before) ──

#[test]
fn slot_first_written_inside_gamma() {
    let ir = run_slot2reg(
        r#"
lambda @0 (shape: "test") {
  region {
    args: [%cs, %os]
    n0 = Const(0x0) [] -> [v0]
    n7 = gamma [
      pred: v0
      in0: %cs:arg
      in1: %os:arg
    ] {
      branch 0:
        region {
          args: [%cs, %os]
          n1 = Const(0xa) [] -> [v1]
          n2 = WriteToSlot(0) [v1, %cs:arg] -> [%cs]
          results: [%cs:n2, %os:arg]
        }
      branch 1:
        region {
          args: [%cs, %os]
          n3 = Const(0x14) [] -> [v2]
          n4 = WriteToSlot(0) [v2, %cs:arg] -> [%cs]
          results: [%cs:n4, %os:arg]
        }
    } -> [%cs, %os]
    n5 = ReadFromSlot(0) [%cs:n7] -> [v3, %cs]
    n6 = WriteToField(offset=0, W4) [v3, %os:n7] -> [%os]
    results: [%cs:n5, %os:n6]
  }
}
"#,
    );
    insta::assert_snapshot!(ir);
}

// ── 8. Nested gamma: slot access in inner gamma ──

#[test]
fn slot_in_nested_gamma() {
    let ir = run_slot2reg(
        r#"
lambda @0 (shape: "test") {
  region {
    args: [%cs, %os]
    n0 = Const(0x2a) [] -> [v0]
    n1 = WriteToSlot(0) [v0, %cs:arg] -> [%cs]
    n2 = Const(0x0) [] -> [v1]
    n9 = gamma [
      pred: v1
      in0: %cs:n1
      in1: %os:arg
    ] {
      branch 0:
        region {
          args: [%cs, %os]
          n3 = Const(0x1) [] -> [v2]
          n8 = gamma [
            pred: v2
            in0: %cs:arg
            in1: %os:arg
          ] {
            branch 0:
              region {
                args: [%cs, %os]
                n4 = ReadFromSlot(0) [%cs:arg] -> [v3, %cs]
                n5 = WriteToField(offset=0, W4) [v3, %os:arg] -> [%os]
                results: [%cs:n4, %os:n5]
              }
            branch 1:
              region {
                args: [%cs, %os]
                n6 = ReadFromSlot(0) [%cs:arg] -> [v4, %cs]
                n7 = WriteToField(offset=0, W4) [v4, %os:arg] -> [%os]
                results: [%cs:n6, %os:n7]
              }
          } -> [%cs, %os]
          results: [%cs:n8, %os:n8]
        }
      branch 1:
        region {
          args: [%cs, %os]
          results: [%cs:arg, %os:arg]
        }
    } -> [%cs, %os]
    results: [%cs:n9, %os:n9]
  }
}
"#,
    );
    insta::assert_snapshot!(ir);
}

// ── 9. Gamma with error exit branch ──

#[test]
fn slot_with_error_branch() {
    let ir = run_slot2reg(
        r#"
lambda @0 (shape: "test") {
  region {
    args: [%cs, %os]
    n0 = Const(0x2a) [] -> [v0]
    n1 = WriteToSlot(0) [v0, %cs:arg] -> [%cs]
    n2 = Const(0x0) [] -> [v1]
    n7 = gamma [
      pred: v1
      in0: %cs:n1
      in1: %os:arg
    ] {
      branch 0:
        region {
          args: [%cs, %os]
          n3 = ReadFromSlot(0) [%cs:arg] -> [v2, %cs]
          n4 = WriteToField(offset=0, W4) [v2, %os:arg] -> [%os]
          results: [%cs:n3, %os:n4]
        }
      branch 1:
        region {
          args: [%cs, %os]
          n5 = ErrorExit(UnexpectedEof) [%cs:arg] -> []
          results: [%cs:arg, %os:arg]
        }
    } -> [%cs, %os]
    results: [%cs:n7, %os:n7]
  }
}
"#,
    );
    insta::assert_snapshot!(ir);
}

// ── 10. Theta with unchanged slot (read only, no write) ──

#[test]
fn theta_unchanged_slot() {
    let ir = run_slot2reg(
        r#"
lambda @0 (shape: "test") {
  region {
    args: [%cs, %os]
    n0 = Const(0x2a) [] -> [v0]
    n1 = WriteToSlot(0) [v0, %cs:arg] -> [%cs]
    n6 = theta [%cs:n1, %os:arg] {
      region {
        args: [%cs, %os]
        n2 = ReadFromSlot(0) [%cs:arg] -> [v1, %cs]
        n3 = WriteToField(offset=0, W4) [v1, %os:arg] -> [%os]
        n4 = Const(0x0) [] -> [v2]
        results: [v2, %cs:n2, %os:n3]
      }
    } -> [%cs, %os]
    results: [%cs:n6, %os:n6]
  }
}
"#,
    );
    insta::assert_snapshot!(ir);
}

// ── 11. Slot in theta with nested gamma (conditional update) ──

#[test]
fn slot_in_theta_with_nested_gamma() {
    let ir = run_slot2reg(
        r#"
lambda @0 (shape: "test") {
  region {
    args: [%cs, %os]
    n0 = Const(0x0) [] -> [v0]
    n1 = WriteToSlot(0) [v0, %cs:arg] -> [%cs]
    n10 = theta [%cs:n1, %os:arg] {
      region {
        args: [%cs, %os]
        n2 = Const(0x1) [] -> [v1]
        n9 = gamma [
          pred: v1
          in0: %cs:arg
          in1: %os:arg
        ] {
          branch 0:
            region {
              args: [%cs, %os]
              results: [%cs:arg, %os:arg]
            }
          branch 1:
            region {
              args: [%cs, %os]
              n3 = Const(0x1) [] -> [v2]
              n4 = ReadFromSlot(0) [%cs:arg] -> [v3, %cs]
              n5 = Add [v3, v2] -> [v4]
              n6 = WriteToSlot(0) [v4, %cs:n4] -> [%cs]
              results: [%cs:n6, %os:arg]
            }
        } -> [%cs, %os]
        n7 = Const(0x0) [] -> [v5]
        results: [v5, %cs:n9, %os:n9]
      }
    } -> [%cs, %os]
    n8 = ReadFromSlot(0) [%cs:n10] -> [v6, %cs]
    n11 = WriteToField(offset=0, W4) [v6, %os:n10] -> [%os]
    results: [%cs:n8, %os:n11]
  }
}
"#,
    );
    insta::assert_snapshot!(ir);
}

// ── 12. Two slots through theta ──

#[test]
fn two_slots_through_theta() {
    let ir = run_slot2reg(
        r#"
lambda @0 (shape: "test") {
  region {
    args: [%cs, %os]
    n0 = Const(0x0) [] -> [v0]
    n1 = Const(0x64) [] -> [v1]
    n2 = WriteToSlot(0) [v0, %cs:arg] -> [%cs]
    n3 = WriteToSlot(1) [v1, %cs:n2] -> [%cs]
    n12 = theta [%cs:n3, %os:arg] {
      region {
        args: [%cs, %os]
        n4 = ReadFromSlot(0) [%cs:arg] -> [v2, %cs]
        n5 = ReadFromSlot(1) [%cs:n4] -> [v3, %cs]
        n6 = Const(0x1) [] -> [v4]
        n7 = Add [v2, v4] -> [v5]
        n8 = Sub [v3, v4] -> [v6]
        n9 = WriteToSlot(0) [v5, %cs:n5] -> [%cs]
        n10 = WriteToSlot(1) [v6, %cs:n9] -> [%cs]
        results: [v6, %cs:n10, %os:arg]
      }
    } -> [%cs, %os]
    n13 = ReadFromSlot(0) [%cs:n12] -> [v7, %cs]
    n14 = ReadFromSlot(1) [%cs:n13] -> [v8, %cs]
    n15 = WriteToField(offset=0, W4) [v7, %os:n12] -> [%os]
    n16 = WriteToField(offset=4, W4) [v8, %os:n15] -> [%os]
    results: [%cs:n14, %os:n16]
  }
}
"#,
    );
    insta::assert_snapshot!(ir);
}

// ── 13. ReadFromSlot output used directly inside gamma branch (cross-scope) ──

#[test]
fn slot_read_output_used_in_gamma_branch() {
    let ir = run_slot2reg(
        r#"
lambda @0 (shape: "test") {
  region {
    args: [%cs, %os]
    n0 = Const(0x2a) [] -> [v0]
    n1 = WriteToSlot(0) [v0, %cs:arg] -> [%cs]
    n2 = ReadFromSlot(0) [%cs:n1] -> [v1, %cs]
    n3 = Const(0x0) [] -> [v2]
    n8 = gamma [
      pred: v2
      in0: %cs:n2
      in1: %os:arg
    ] {
      branch 0:
        region {
          args: [%cs, %os]
          n4 = WriteToField(offset=0, W4) [v1, %os:arg] -> [%os]
          results: [%cs:arg, %os:n4]
        }
      branch 1:
        region {
          args: [%cs, %os]
          n5 = WriteToField(offset=0, W4) [v1, %os:arg] -> [%os]
          results: [%cs:arg, %os:n5]
        }
    } -> [%cs, %os]
    results: [%cs:n8, %os:n8]
  }
}
"#,
    );
    insta::assert_snapshot!(ir);
}
