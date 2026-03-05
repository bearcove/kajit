#![cfg(target_arch = "x86_64")]

use facet::Facet;
use kajit_ir::ErrorCode;
use kajit_lir::{LinearIr, LinearOp};

const POSTCARD_U32_V0_X86_64_SNAPSHOT: &str =
    include_str!("snapshots/corpus__generated_rvsdg_postcard_scalar_u32__v0_x86_64.snap");
const POSTCARD_U16_V0_X86_64_SNAPSHOT: &str =
    include_str!("snapshots/corpus__generated_rvsdg_postcard_scalar_u16__v0_x86_64.snap");

fn snapshot_body(snapshot: &'static str) -> &'static str {
    let snapshot = snapshot
        .strip_prefix("---\n")
        .expect("insta snapshot should start with frontmatter");
    let (_, body) = snapshot
        .split_once("\n---\n")
        .expect("insta snapshot frontmatter should end with separator");
    body.trim()
}

fn infer_output_size(linear: &LinearIr) -> usize {
    linear
        .ops
        .iter()
        .filter_map(|op| match op {
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
    ir_text: &'static str,
    output_size: usize,
    allocated: kajit::regalloc_engine::AllocatedCfgProgram,
    decoder: kajit::compiler::CompiledDecoder,
}

impl DifferentialHarness {
    fn from_snapshot(snapshot: &'static str, shape: &'static facet::Shape) -> Self {
        let ir_text = snapshot_body(snapshot);
        let registry = kajit::ir::IntrinsicRegistry::new();
        let mut ir_func = kajit::ir_parse::parse_ir(ir_text, shape, &registry)
            .expect("RVSDG snapshot should parse");
        let linear = kajit::linearize::linearize(&mut ir_func);
        let output_size = infer_output_size(&linear);
        let cfg_program = kajit::regalloc_engine::cfg_mir::lower_linear_ir(&linear);
        let allocated = kajit::regalloc_engine::allocate_cfg_program(&cfg_program)
            .expect("regalloc2 should allocate cfg_mir");
        let decoder = kajit::compile_decoder_linear_ir(&linear, false);
        Self {
            ir_text,
            output_size,
            allocated,
            decoder,
        }
    }

    fn run_simulator(&self, input: &[u8]) -> NormalizedOutcome {
        match kajit::regalloc_engine::simulate_execution_cfg(&self.allocated, input) {
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
        (self.run_simulator(input), self.run_jit(input))
    }

    fn mismatch_report(
        &self,
        input: &[u8],
        interpreter: &NormalizedOutcome,
        jit: &NormalizedOutcome,
    ) -> String {
        format!(
            "simulator-vs-jit mismatch\ninput={}\nsimulator={}\njit={}\nrvsdg:\n{}\n",
            format_bytes(input),
            format_outcome(interpreter),
            format_outcome(jit),
            self.ir_text
        )
    }

    /// Returns a deterministic mismatch report when simulator and JIT diverge.
    fn mismatch_report_for_input(&self, input: &[u8]) -> Option<String> {
        let (simulator, jit) = self.compare(input);
        if simulator == jit {
            None
        } else {
            Some(self.mismatch_report(input, &simulator, &jit))
        }
    }

    fn assert_match(&self, input: &[u8]) -> NormalizedOutcome {
        let (simulator, jit) = self.compare(input);
        assert_eq!(
            simulator,
            jit,
            "{}",
            self.mismatch_report(input, &simulator, &jit)
        );
        simulator
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
    let harness =
        DifferentialHarness::from_snapshot(POSTCARD_U32_V0_X86_64_SNAPSHOT, <u32 as Facet>::SHAPE);
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
    let harness =
        DifferentialHarness::from_snapshot(POSTCARD_U32_V0_X86_64_SNAPSHOT, <u32 as Facet>::SHAPE);
    let report = harness
        .mismatch_report_for_input(&[0x80, 0x80, 0x80, 0x80, 0x80])
        .expect("malformed varint should currently diverge between simulator and JIT");
    assert!(report.contains("code=invalid varint encoding"));
    assert!(report.contains("simulator=failure code=invalid varint encoding offset=5"));
    assert!(report.contains("jit=failure code=invalid varint encoding offset=0"));
}

#[test]
fn postcard_u16_interpreter_vs_jit_varint_cases() {
    let harness =
        DifferentialHarness::from_snapshot(POSTCARD_U16_V0_X86_64_SNAPSHOT, <u16 as Facet>::SHAPE);
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
    let harness =
        DifferentialHarness::from_snapshot(POSTCARD_U16_V0_X86_64_SNAPSHOT, <u16 as Facet>::SHAPE);
    assert!(
        harness.mismatch_report_for_input(&[0x81, 0x01]).is_none(),
        "u16=129 repro should not diverge"
    );
}
