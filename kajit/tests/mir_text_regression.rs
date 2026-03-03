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

fn run_mir_interpreter_u32(
    mir_text: &str,
    input: &[u8],
) -> Result<u32, kajit_mir::InterpreterTrap> {
    let program = kajit_mir_text::parse_ra_mir(mir_text).expect("fixture should parse");
    let outcome = kajit_mir::execute_program(&program, input).expect("interpreter should execute");
    if let Some(trap) = outcome.trap {
        return Err(trap);
    }
    assert!(
        outcome.output.len() >= 4,
        "interpreter output must contain at least 4 bytes for u32 result"
    );
    let bytes: [u8; 4] = outcome.output[..4].try_into().expect("slice must fit");
    Ok(u32::from_le_bytes(bytes))
}

fn run_mir_interpreter_trap(mir_text: &str, input: &[u8]) -> kajit_mir::InterpreterTrap {
    let program = kajit_mir_text::parse_ra_mir(mir_text).expect("fixture should parse");
    let outcome = kajit_mir::execute_program(&program, input).expect("interpreter should execute");
    outcome
        .trap
        .expect("input should trigger an interpreter trap")
}

const POSTCARD_U32_V0_X86_64_MIR: &str = r#"
ra_func @0 { ; entry: b0
  block b0: ; insts: 11
    bounds_check(1)
    v1:gpr = const(0x7f)
    v3:gpr = const(0x80)
    v5:gpr = const(0x0)
    v37:gpr = const(0x0)
    v39:gpr = const(0x20)
    v41:gpr = const(0x0)
    v0:gpr = read_bytes(1)
    v2:gpr = And v0:gpr, v1:gpr
    v4:gpr = And v0:gpr, v3:gpr
    v6:gpr = CmpNe v4:gpr, v5:gpr
    term: branch_if_zero v6 -> b2, fallthrough b1 ; uses: [v6]
    succs: b2 [v2, v37, v39, v41] b1 [v2, v37, v39, v41] ; count: 2
  block b1 [params: v2, v37, v39, v41] (preds: b0): ; insts: 0
    term: branch b3 ; uses: []
    succs: b3 [v2, v37, v39, v41] ; count: 1
  block b2 [params: v2, v37, v39, v41] (preds: b0): ; insts: 4
    v7:gpr = const(0x4)
    v8:gpr = const(0x0)
    v33:gpr = copy v2:gpr
    v36:gpr = copy v8:gpr
    term: branch b6 ; uses: []
    succs: b6 [v33, v8, v37, v39, v41] ; count: 1
  block b3 [params: v2, v37, v39, v41] (preds: b1): ; insts: 12
    v9:gpr = const(0x7)
    v10:gpr = const(0x4)
    v11:gpr = const(0x1)
    v13:gpr = const(0x7f)
    v17:gpr = const(0x7)
    v19:gpr = const(0x1)
    v21:gpr = const(0x80)
    v23:gpr = const(0x0)
    v25:gpr = const(0x0)
    v45:gpr = copy v2:gpr
    v46:gpr = copy v9:gpr
    v47:gpr = copy v10:gpr
    term: branch b4 ; uses: []
    succs: b4 [v13, v17, v19, v21, v23, v25, v37, v39, v41, v45, v46, v47] ; count: 1
  block b4 [params: v13, v17, v19, v21, v23, v25, v37, v39, v41, v45, v46, v47] (preds: b3, b4): ; insts: 16
    bounds_check(1)
    v18:gpr = Add v46:gpr, v17:gpr
    v20:gpr = Sub v47:gpr, v19:gpr
    v12:gpr = read_bytes(1)
    v26:gpr = CmpNe v20:gpr, v25:gpr
    v14:gpr = And v12:gpr, v13:gpr
    v22:gpr = And v12:gpr, v21:gpr
    v15:gpr = Shl v14:gpr, v46:gpr/hw1
    v24:gpr = CmpNe v22:gpr, v23:gpr
    v16:gpr = Or v45:gpr, v15:gpr
    v27:gpr = And v24:gpr, v26:gpr
    v45:gpr = copy v16:gpr
    v46:gpr = copy v18:gpr
    v47:gpr = copy v20:gpr
    v48:gpr = copy v14:gpr
    v49:gpr = copy v24:gpr
    term: branch_if v27 -> b4, fallthrough b5 ; uses: [v27]
    succs: b4 [v13, v17, v19, v21, v23, v25, v37, v39, v41, v45, v46, v47] b5 [v37, v39, v41, v45, v49] ; count: 2
  block b5 [params: v37, v39, v41, v45, v49] (preds: b4): ; insts: 2
    v33:gpr = copy v45:gpr
    v36:gpr = copy v49:gpr
    term: branch b6 ; uses: []
    succs: b6 [v33, v36, v37, v39, v41] ; count: 1
  block b6 [params: v33, v36, v37, v39, v41] (preds: b2, b5): ; insts: 2
    v38:gpr = CmpNe v36:gpr, v37:gpr
    v40:gpr = Shr v33:gpr, v39:gpr/hw1
    term: branch_if_zero v38 -> b8, fallthrough b7 ; uses: [v38]
    succs: b8 [v33, v40, v41] b7 ; count: 2
  block b7 (preds: b6): ; insts: 0
    term: branch b9 ; uses: []
    succs: b9 ; count: 1
  block b8 [params: v33, v40, v41] (preds: b6): ; insts: 0
    term: branch b10 ; uses: []
    succs: b10 [v33, v40, v41] ; count: 1
  block b9 (preds: b7): ; insts: 0
    term: error_exit(InvalidVarint) ; uses: []
    succs: (none)
  block b10 [params: v33, v40, v41] (preds: b8): ; insts: 1
    v42:gpr = CmpNe v40:gpr, v41:gpr
    term: branch_if_zero v42 -> b12, fallthrough b11 ; uses: [v42]
    succs: b12 [v33] b11 ; count: 2
  block b11 (preds: b10): ; insts: 0
    term: branch b13 ; uses: []
    succs: b13 ; count: 1
  block b12 [params: v33] (preds: b10): ; insts: 0
    term: branch b14 ; uses: []
    succs: b14 [v33] ; count: 1
  block b13 (preds: b11): ; insts: 0
    term: error_exit(NumberOutOfRange) ; uses: []
    succs: (none)
  block b14 [params: v33] (preds: b12): ; insts: 1
    store([0:W4]) v33:gpr
    term: return ; uses: []
    succs: (none)
}
"#;

#[test]
fn postcard_u32_human_ra_mir_snapshot() {
    let program =
        kajit_mir_text::parse_ra_mir(POSTCARD_U32_V0_X86_64_MIR).expect("fixture should parse");
    insta::assert_snapshot!(
        "mir_text_regression__postcard_u32_v0_x86_64_human",
        format!("{}", program.human())
    );
}

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

#[test]
fn postcard_u32_interpreter_single_and_multi_byte_varint() {
    let mir = POSTCARD_U32_V0_X86_64_MIR;

    let result = run_mir_interpreter_u32(mir, &[0x2a]).unwrap();
    assert_eq!(result, 42);

    let result = run_mir_interpreter_u32(mir, &[0x00]).unwrap();
    assert_eq!(result, 0);

    let result = run_mir_interpreter_u32(mir, &[0x7f]).unwrap();
    assert_eq!(result, 127);

    let result = run_mir_interpreter_u32(mir, &[0x80, 0x01]).unwrap();
    assert_eq!(result, 128);

    let result = run_mir_interpreter_u32(mir, &[0xac, 0x02]).unwrap();
    assert_eq!(result, 300);
}

#[test]
fn postcard_u32_interpreter_error_cases() {
    let mir = POSTCARD_U32_V0_X86_64_MIR;

    let malformed = run_mir_interpreter_trap(mir, &[0x80, 0x80, 0x80, 0x80, 0x80]);
    assert_eq!(malformed.code, kajit_ir::ErrorCode::InvalidVarint);

    let overflow = run_mir_interpreter_trap(mir, &[0x80, 0x80, 0x80, 0x80, 0x10]);
    assert_eq!(overflow.code, kajit_ir::ErrorCode::NumberOutOfRange);
}
