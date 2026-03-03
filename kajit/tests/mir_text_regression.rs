/// Regression tests that compile and execute from RA-MIR text.
///
/// These tests are for isolating and minimizing codegen bugs: paste a failing
/// RA-MIR snapshot, strip it down, and re-run until you have a minimal
/// reproducer. See DEVELOP.md for the full workflow.
use facet::Facet;

fn run_mir<'a, T: Facet<'a>>(mir_text: &str, input: &'a [u8]) -> Result<T, kajit::DeserError> {
    let dec = kajit::compile_decoder_from_ra_mir_text(mir_text);
    kajit::deserialize::<T>(&dec, input)
}

const POSTCARD_U32_V0_X86_64_MIR: &str = r#"
ra_func @0 { ; entry: b0
  block b0: ; insts: 11
    bounds_check(1) ; sem: ensure 1 input byte(s) available
    v1:gpr = const(0x7f) ; sem: materialize constant 0x7f
    v3:gpr = const(0x80) ; sem: materialize constant 0x80
    v5:gpr = const(0x0) ; sem: materialize constant 0x0
    v37:gpr = const(0x0) ; sem: materialize constant 0x0
    v39:gpr = const(0x20) ; sem: materialize constant 0x20
    v41:gpr = const(0x0) ; sem: materialize constant 0x0
    v0:gpr = read_bytes(1) ; sem: consume 1 input byte(s)
    v2:gpr = And v0:gpr, v1:gpr ; sem: bitwise and
    v4:gpr = And v0:gpr, v3:gpr ; sem: bitwise and
    v6:gpr = CmpNe v4:gpr, v5:gpr ; sem: compare not-equal (produces 0/1)
    term: branch_if_zero v6 -> b2, fallthrough b1 ; uses: [v6] ; sem: branch when condition is zero
    succs: b2 [v2, v37, v39, v41] b1 [v2, v37, v39, v41] ; count: 2
  block b1 [params: v2, v37, v39, v41] (preds: b0): ; insts: 0
    term: branch b3 ; uses: [] ; sem: unconditional jump
    succs: b3 [v2, v37, v39, v41] ; count: 1
  block b2 [params: v2, v37, v39, v41] (preds: b0): ; insts: 4
    v7:gpr = const(0x4) ; sem: materialize constant 0x4
    v8:gpr = const(0x0) ; sem: materialize constant 0x0
    v33:gpr = copy v2:gpr ; sem: ssa copy/move
    v36:gpr = copy v8:gpr ; sem: ssa copy/move
    term: branch b6 ; uses: [] ; sem: unconditional jump
    succs: b6 [v33, v8, v37, v39, v41] ; count: 1
  block b3 [params: v2, v37, v39, v41] (preds: b1): ; insts: 12
    v9:gpr = const(0x7) ; sem: materialize constant 0x7
    v10:gpr = const(0x4) ; sem: materialize constant 0x4
    v11:gpr = const(0x1) ; sem: materialize constant 0x1
    v13:gpr = const(0x7f) ; sem: materialize constant 0x7f
    v17:gpr = const(0x7) ; sem: materialize constant 0x7
    v19:gpr = const(0x1) ; sem: materialize constant 0x1
    v21:gpr = const(0x80) ; sem: materialize constant 0x80
    v23:gpr = const(0x0) ; sem: materialize constant 0x0
    v25:gpr = const(0x0) ; sem: materialize constant 0x0
    v45:gpr = copy v2:gpr ; sem: ssa copy/move
    v46:gpr = copy v9:gpr ; sem: ssa copy/move
    v47:gpr = copy v10:gpr ; sem: ssa copy/move
    term: branch b4 ; uses: [] ; sem: unconditional jump
    succs: b4 [v13, v17, v19, v21, v23, v25, v37, v39, v41, v45, v46, v47] ; count: 1
  block b4 [params: v13, v17, v19, v21, v23, v25, v37, v39, v41, v45, v46, v47] (preds: b3, b4): ; insts: 16
    bounds_check(1) ; sem: ensure 1 input byte(s) available
    v18:gpr = Add v46:gpr, v17:gpr ; sem: integer addition
    v20:gpr = Sub v47:gpr, v19:gpr ; sem: integer subtraction
    v12:gpr = read_bytes(1) ; sem: consume 1 input byte(s)
    v26:gpr = CmpNe v20:gpr, v25:gpr ; sem: compare not-equal (produces 0/1)
    v14:gpr = And v12:gpr, v13:gpr ; sem: bitwise and
    v22:gpr = And v12:gpr, v21:gpr ; sem: bitwise and
    v15:gpr = Shl v14:gpr, v46:gpr/hw1 ; sem: logical shift left
    v24:gpr = CmpNe v22:gpr, v23:gpr ; sem: compare not-equal (produces 0/1)
    v16:gpr = Or v45:gpr, v15:gpr ; sem: bitwise or
    v27:gpr = And v24:gpr, v26:gpr ; sem: bitwise and
    v45:gpr = copy v16:gpr ; sem: ssa copy/move
    v46:gpr = copy v18:gpr ; sem: ssa copy/move
    v47:gpr = copy v20:gpr ; sem: ssa copy/move
    v48:gpr = copy v14:gpr ; sem: ssa copy/move
    v49:gpr = copy v24:gpr ; sem: ssa copy/move
    term: branch_if v27 -> b4, fallthrough b5 ; uses: [v27] ; sem: branch when condition is non-zero
    succs: b4 [v13, v17, v19, v21, v23, v25, v37, v39, v41, v45, v46, v47] b5 [v37, v39, v41, v45, v49] ; count: 2
  block b5 [params: v37, v39, v41, v45, v49] (preds: b4): ; insts: 2
    v33:gpr = copy v45:gpr ; sem: ssa copy/move
    v36:gpr = copy v49:gpr ; sem: ssa copy/move
    term: branch b6 ; uses: [] ; sem: unconditional jump
    succs: b6 [v33, v36, v37, v39, v41] ; count: 1
  block b6 [params: v33, v36, v37, v39, v41] (preds: b2, b5): ; insts: 2
    v38:gpr = CmpNe v36:gpr, v37:gpr ; sem: compare not-equal (produces 0/1)
    v40:gpr = Shr v33:gpr, v39:gpr/hw1 ; sem: logical shift right
    term: branch_if_zero v38 -> b8, fallthrough b7 ; uses: [v38] ; sem: branch when condition is zero
    succs: b8 [v33, v40, v41] b7 ; count: 2
  block b7 (preds: b6): ; insts: 0
    term: branch b9 ; uses: [] ; sem: unconditional jump
    succs: b9 ; count: 1
  block b8 [params: v33, v40, v41] (preds: b6): ; insts: 0
    term: branch b10 ; uses: [] ; sem: unconditional jump
    succs: b10 [v33, v40, v41] ; count: 1
  block b9 (preds: b7): ; insts: 0
    term: error_exit(InvalidVarint) ; uses: [] ; sem: terminate with error
    succs: (none)
  block b10 [params: v33, v40, v41] (preds: b8): ; insts: 1
    v42:gpr = CmpNe v40:gpr, v41:gpr ; sem: compare not-equal (produces 0/1)
    term: branch_if_zero v42 -> b12, fallthrough b11 ; uses: [v42] ; sem: branch when condition is zero
    succs: b12 [v33] b11 ; count: 2
  block b11 (preds: b10): ; insts: 0
    term: branch b13 ; uses: [] ; sem: unconditional jump
    succs: b13 ; count: 1
  block b12 [params: v33] (preds: b10): ; insts: 0
    term: branch b14 ; uses: [] ; sem: unconditional jump
    succs: b14 [v33] ; count: 1
  block b13 (preds: b11): ; insts: 0
    term: error_exit(NumberOutOfRange) ; uses: [] ; sem: terminate with error
    succs: (none)
  block b14 [params: v33] (preds: b12): ; insts: 1
    store([0:W4]) v33:gpr ; sem: write output field at +0 (W4)
    term: return ; uses: [] ; sem: finish function
    succs: (none)
}
"#;

/// Postcard varint decoder for u32 — single-byte case (value < 128).
///
/// Derived from `corpus__generated_ra_mir_postcard_scalar_u32__v0_x86_64.snap`.
#[test]
fn postcard_u32_single_byte_varint() {
    let mir = POSTCARD_U32_V0_X86_64_MIR;
    // 42 encodes as single byte 0x2a in postcard varint
    let result: u32 = run_mir(mir, &[0x2a]).unwrap();
    assert_eq!(result, 42);

    // 0 encodes as single byte 0x00
    let result: u32 = run_mir(mir, &[0x00]).unwrap();
    assert_eq!(result, 0);

    // 127 encodes as single byte 0x7f
    let result: u32 = run_mir(mir, &[0x7f]).unwrap();
    assert_eq!(result, 127);
}

/// Postcard varint decoder for u32 — multi-byte case (value >= 128).
///
/// 128 encodes as [0x80, 0x01] in postcard varint.
#[test]
fn postcard_u32_multi_byte_varint() {
    let mir = POSTCARD_U32_V0_X86_64_MIR;
    // 128 encodes as [0x80, 0x01] in postcard varint
    let result: u32 = run_mir(mir, &[0x80, 0x01]).unwrap();
    assert_eq!(result, 128);

    // 300 encodes as [0xAC, 0x02] in postcard varint
    let result: u32 = run_mir(mir, &[0xac, 0x02]).unwrap();
    assert_eq!(result, 300);
}
