#![cfg(target_arch = "x86_64")]

use kajit_ir::ErrorCode;
use kajit_lir::LinearOp;
use kajit_mir::RaProgram;

const POSTCARD_U32_V0_X86_64_SNAPSHOT: &str =
    include_str!("snapshots/corpus__generated_ra_mir_postcard_scalar_u32__v0_x86_64.snap");
const POSTCARD_U16_V0_X86_64_SNAPSHOT: &str =
    include_str!("snapshots/corpus__generated_ra_mir_postcard_scalar_u16__v0_x86_64.snap");

fn snapshot_body(snapshot: &'static str) -> &'static str {
    let snapshot = snapshot
        .strip_prefix("---\n")
        .expect("insta snapshot should start with frontmatter");
    let (_, body) = snapshot
        .split_once("\n---\n")
        .expect("insta snapshot frontmatter should end with separator");
    body.trim()
}

fn infer_output_size(program: &RaProgram) -> usize {
    program
        .funcs
        .iter()
        .flat_map(|func| func.blocks.iter())
        .flat_map(|block| block.insts.iter())
        .filter_map(|inst| match &inst.op {
            LinearOp::WriteToField { offset, width, .. }
            | LinearOp::ReadFromField { offset, width, .. } => {
                Some(*offset as usize + width.bytes() as usize)
            }
            _ => None,
        })
        .max()
        .unwrap_or(0)
}

fn format_bytes(bytes: &[u8]) -> String {
    let mut out = String::from("[");
    for (index, byte) in bytes.iter().enumerate() {
        if index > 0 {
            out.push(' ');
        }
        out.push_str(&format!("{byte:02x}"));
    }
    out.push(']');
    out
}

fn decode_le_u64(bytes: &[u8]) -> Option<u64> {
    if bytes.len() > 8 {
        return None;
    }
    let mut padded = [0u8; 8];
    padded[..bytes.len()].copy_from_slice(bytes);
    Some(u64::from_le_bytes(padded))
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum NormalizedOutcome {
    Success {
        output: Vec<u8>,
        value_le_u64: Option<u64>,
    },
    Failure {
        code: ErrorCode,
        offset: u32,
    },
    HarnessError(String),
}

fn format_outcome(outcome: &NormalizedOutcome) -> String {
    match outcome {
        NormalizedOutcome::Success {
            output,
            value_le_u64,
        } => {
            let value = match value_le_u64 {
                Some(value) => format!("{value} (0x{value:x})"),
                None => "n/a".to_owned(),
            };
            format!(
                "success bytes={} value_le_u64={value}",
                format_bytes(output)
            )
        }
        NormalizedOutcome::Failure { code, offset } => {
            format!("failure code={code} offset={offset}")
        }
        NormalizedOutcome::HarnessError(message) => format!("harness_error {message}"),
    }
}

struct DifferentialHarness {
    mir_text: &'static str,
    program: RaProgram,
    output_size: usize,
    decoder: kajit::compiler::CompiledDecoder,
}

impl DifferentialHarness {
    fn from_snapshot(snapshot: &'static str) -> Self {
        let mir_text = snapshot_body(snapshot);
        let program = kajit_mir_text::parse_ra_mir(mir_text).expect("fixture should parse");
        let output_size = infer_output_size(&program);
        let decoder = kajit::compile_decoder_from_ra_mir_text(mir_text);
        Self {
            mir_text,
            program,
            output_size,
            decoder,
        }
    }

    fn run_interpreter(&self, input: &[u8]) -> NormalizedOutcome {
        match kajit_mir::execute_program(&self.program, input) {
            Ok(outcome) => match outcome.trap {
                Some(trap) => NormalizedOutcome::Failure {
                    code: trap.code,
                    offset: trap.offset,
                },
                None => {
                    let mut output = outcome.output;
                    if output.len() < self.output_size {
                        output.resize(self.output_size, 0);
                    } else if output.len() > self.output_size {
                        output.truncate(self.output_size);
                    }
                    let value_le_u64 = decode_le_u64(&output);
                    NormalizedOutcome::Success {
                        output,
                        value_le_u64,
                    }
                }
            },
            Err(error) => NormalizedOutcome::HarnessError(error.to_string()),
        }
    }

    fn run_jit(&self, input: &[u8]) -> NormalizedOutcome {
        match kajit::deserialize_raw(&self.decoder, input, self.output_size) {
            Ok(output) => {
                let value_le_u64 = decode_le_u64(&output);
                NormalizedOutcome::Success {
                    output,
                    value_le_u64,
                }
            }
            Err(error) => NormalizedOutcome::Failure {
                code: error.code,
                offset: error.offset,
            },
        }
    }

    fn compare(&self, input: &[u8]) -> (NormalizedOutcome, NormalizedOutcome) {
        (self.run_interpreter(input), self.run_jit(input))
    }

    fn mismatch_report(
        &self,
        input: &[u8],
        interpreter: &NormalizedOutcome,
        jit: &NormalizedOutcome,
    ) -> String {
        format!(
            "interpreter-vs-jit mismatch\ninput={}\ninterpreter={}\njit={}\nra_mir:\n{}\n",
            format_bytes(input),
            format_outcome(interpreter),
            format_outcome(jit),
            self.mir_text
        )
    }

    /// Returns a deterministic mismatch report when interpreter and JIT diverge.
    fn mismatch_report_for_input(&self, input: &[u8]) -> Option<String> {
        let (interpreter, jit) = self.compare(input);
        if interpreter == jit {
            None
        } else {
            Some(self.mismatch_report(input, &interpreter, &jit))
        }
    }

    fn assert_match(&self, input: &[u8]) -> NormalizedOutcome {
        let (interpreter, jit) = self.compare(input);
        assert_eq!(
            interpreter,
            jit,
            "{}",
            self.mismatch_report(input, &interpreter, &jit)
        );
        interpreter
    }
}

fn expect_success_u32(outcome: &NormalizedOutcome) -> u32 {
    match outcome {
        NormalizedOutcome::Success { output, .. } => {
            assert!(
                output.len() >= 4,
                "expected at least 4 output bytes, got {}",
                output.len()
            );
            let bytes: [u8; 4] = output[..4].try_into().expect("slice must fit");
            u32::from_le_bytes(bytes)
        }
        other => panic!("expected success outcome for u32, got {other:?}"),
    }
}

fn expect_success_u16(outcome: &NormalizedOutcome) -> u16 {
    match outcome {
        NormalizedOutcome::Success { output, .. } => {
            assert!(
                output.len() >= 2,
                "expected at least 2 output bytes, got {}",
                output.len()
            );
            let bytes: [u8; 2] = output[..2].try_into().expect("slice must fit");
            u16::from_le_bytes(bytes)
        }
        other => panic!("expected success outcome for u16, got {other:?}"),
    }
}

#[test]
fn postcard_u32_interpreter_vs_jit_varint_cases() {
    let harness = DifferentialHarness::from_snapshot(POSTCARD_U32_V0_X86_64_SNAPSHOT);
    let cases: [(&[u8], u32); 5] = [
        (&[0x2a], 42),
        (&[0x00], 0),
        (&[0x7f], 127),
        (&[0x80, 0x01], 128),
        (&[0xac, 0x02], 300),
    ];

    for (input, expected) in cases {
        let outcome = harness.assert_match(input);
        assert_eq!(expect_success_u32(&outcome), expected);
    }
}

#[test]
fn postcard_u32_interpreter_vs_jit_detects_invalid_varint_offset_mismatch() {
    let harness = DifferentialHarness::from_snapshot(POSTCARD_U32_V0_X86_64_SNAPSHOT);
    let report = harness
        .mismatch_report_for_input(&[0x80, 0x80, 0x80, 0x80, 0x80])
        .expect("malformed varint should currently diverge between interpreter and JIT");
    assert!(report.contains("code=invalid varint encoding"));
    assert!(report.contains("interpreter=failure code=invalid varint encoding offset=5"));
    assert!(report.contains("jit=failure code=invalid varint encoding offset=0"));
}

#[test]
fn postcard_u16_interpreter_vs_jit_varint_cases() {
    let harness = DifferentialHarness::from_snapshot(POSTCARD_U16_V0_X86_64_SNAPSHOT);
    let cases: [(&[u8], u16); 5] = [
        (&[0x00], 0),
        (&[0x7f], 127),
        (&[0x80, 0x01], 128),
        (&[0x81, 0x01], 129),
        (&[0xff, 0x01], 255),
    ];

    for (input, expected) in cases {
        let outcome = harness.assert_match(input);
        assert_eq!(expect_success_u16(&outcome), expected);
    }
}

#[test]
fn postcard_u16_interpreter_vs_jit_deterministic_mismatch_predicate() {
    let harness = DifferentialHarness::from_snapshot(POSTCARD_U16_V0_X86_64_SNAPSHOT);
    assert!(
        harness.mismatch_report_for_input(&[0x81, 0x01]).is_none(),
        "u16=129 repro should not diverge"
    );
}
