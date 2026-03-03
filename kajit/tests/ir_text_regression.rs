/// Regression tests that compile and execute from IR text (RVSDG representation).
///
/// These tests are for isolating and minimizing codegen bugs at the IR level:
/// paste a failing IR snapshot, strip it down, and re-run until you have a
/// minimal reproducer. See docs/develop/ for debugging workflows.
use facet::Facet;

fn run_ir<'a, T: Facet<'a>>(
    ir_text: &str,
    input: &'a [u8],
    with_passes: bool,
) -> Result<T, kajit::DeserError> {
    let registry = kajit::ir::IntrinsicRegistry::empty();
    let dec = kajit::compile_decoder_from_ir_text(ir_text, T::SHAPE, &registry, with_passes);
    kajit::deserialize::<T>(&dec, input)
}

/// Simple read-and-write: reads 1 byte and writes it to a u8 output.
#[test]
fn postcard_read_u8() {
    let ir = r#"
lambda @0 (shape: "u8") {
  region {
    args: [%cs, %os]
    n0 = BoundsCheck(1) [%cs:arg] -> [%cs]
    n1 = ReadBytes(1) [%cs:n0] -> [v0, %cs]
    n2 = WriteToField(offset=0, W1) [v0, %os:arg] -> [%os]
    results: [%cs:n1, %os:n2]
  }
}
"#;
    let result: u8 = run_ir(ir, &[42], false).unwrap();
    assert_eq!(result, 42);

    let result: u8 = run_ir(ir, &[0], false).unwrap();
    assert_eq!(result, 0);

    let result: u8 = run_ir(ir, &[255], false).unwrap();
    assert_eq!(result, 255);
}

/// Gamma (if/else): read a tag byte, branch on it, write different constants.
///
/// Derived from the `ir_behavior_enum_like_gamma_tag_branch` test in the
/// generated IR behavior corpus.
#[test]
fn gamma_tag_branch() {
    let ir = r#"
lambda @0 (shape: "u8") {
  region {
    args: [%cs, %os]
    n0 = BoundsCheck(1) [%cs:arg] -> [%cs]
    n1 = ReadBytes(1) [%cs:n0] -> [v0, %cs]
    n2 = Const(0x0) [] -> [v1]
    n3 = CmpNe [v0, v1] -> [v2]
    n4 = gamma [
      pred: v2
      in0: %cs:n1
      in1: %os:arg
    ] {
      branch 0:
        region {
          args: [%cs, %os]
          n5 = Const(0x2a) [] -> [v3]
          results: [v3, %cs:arg, %os:arg]
        }
      branch 1:
        region {
          args: [%cs, %os]
          n6 = Const(0x63) [] -> [v4]
          results: [v4, %cs:arg, %os:arg]
        }
    } -> [v5, %cs, %os]
    n7 = WriteToField(offset=0, W1) [v5, %os:n4] -> [%os]
    results: [%cs:n4, %os:n7]
  }
}
"#;
    // tag=0 → branch 0 → 42
    let result: u8 = run_ir(ir, &[0], false).unwrap();
    assert_eq!(result, 42);

    // tag=1 → branch 1 → 99
    let result: u8 = run_ir(ir, &[1], false).unwrap();
    assert_eq!(result, 99);

    // Also test with optimization passes
    let result: u8 = run_ir(ir, &[0], true).unwrap();
    assert_eq!(result, 42);
    let result: u8 = run_ir(ir, &[1], true).unwrap();
    assert_eq!(result, 99);
}
