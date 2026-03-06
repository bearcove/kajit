use crate::{
    DifferentialCheckResult, InterpreterEventKind, InterpreterOutcome, allocate_cfg_program,
    cfg_mir, differential_check_cfg, execute_program, execute_program_with_event_trace,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DebugCfgMirCommand {
    Run,
    Trace,
    Diff,
    WhyVreg {
        vreg: kajit_ir::VReg,
    },
    Block {
        lambda: kajit_ir::LambdaId,
        block: cfg_mir::BlockId,
    },
}

pub fn run_debug_cfg_mir_command(
    program: &cfg_mir::Program,
    input: &[u8],
    command: &DebugCfgMirCommand,
) -> Result<String, String> {
    match command {
        DebugCfgMirCommand::Run => {
            let outcome = execute_program(program, input)
                .map_err(|err| format!("CFG-MIR interpreter failed: {err}"))?;
            Ok(render_interpreter_outcome(&outcome))
        }
        DebugCfgMirCommand::Trace => {
            let (_, trace) = execute_program_with_event_trace(program, input)
                .map_err(|err| format!("CFG-MIR event trace failed: {err}"))?;
            let mut text = trace.render_text();
            if !text.is_empty() {
                text.push('\n');
            }
            Ok(text)
        }
        DebugCfgMirCommand::Diff => {
            let allocated = allocate_cfg_program(program)
                .map_err(|err| format!("regalloc2 allocation failed: {err}"))?;
            Ok(render_differential_check_result(&differential_check_cfg(
                program, &allocated, input,
            )))
        }
        DebugCfgMirCommand::WhyVreg { vreg } => {
            let (_, trace) = execute_program_with_event_trace(program, input)
                .map_err(|err| format!("CFG-MIR event trace failed: {err}"))?;
            match trace.last_write_to_vreg(*vreg) {
                Some(event) => {
                    let mut out = String::new();
                    out.push_str(&format!("last write to v{}:\n", vreg.index()));
                    out.push_str(
                        &trace
                            .render_event_text(event.event_index)
                            .unwrap_or_else(|| "<missing event>".to_owned()),
                    );
                    out.push('\n');
                    Ok(out)
                }
                None => Ok(format!("no writes to v{}\n", vreg.index())),
            }
        }
        DebugCfgMirCommand::Block { lambda, block } => {
            let (_, trace) = execute_program_with_event_trace(program, input)
                .map_err(|err| format!("CFG-MIR event trace failed: {err}"))?;
            let lines = trace
                .events
                .iter()
                .filter(|event| {
                    event.location.lambda == *lambda
                        && (event.location.block == *block
                            || matches!(
                                event.kind,
                                InterpreterEventKind::BlockEnter { target, .. } if target == *block
                            ))
                })
                .filter_map(|event| trace.render_event_text(event.event_index))
                .collect::<Vec<_>>();
            if lines.is_empty() {
                return Ok(format!(
                    "no trace events for lambda @{} block b{}\n",
                    lambda.index(),
                    block.0
                ));
            }
            Ok(lines.join("\n") + "\n")
        }
    }
}

fn render_interpreter_outcome(outcome: &InterpreterOutcome) -> String {
    let mut out = String::new();
    out.push_str(&format!("cursor: {}\n", outcome.cursor));
    out.push_str(&format!("trap: {:?}\n", outcome.trap));
    out.push_str("returned_vregs:");
    for value in &outcome.vregs {
        out.push_str(&format!(" {value:#x}"));
    }
    out.push('\n');
    out.push_str("output_hex: ");
    out.push_str(&bytes_to_hex(&outcome.output));
    out.push('\n');
    out
}

fn render_differential_check_result(result: &DifferentialCheckResult) -> String {
    match result {
        DifferentialCheckResult::Match { steps, .. } => {
            format!("match after {steps} steps\n")
        }
        DifferentialCheckResult::Diverged(divergence) => {
            let mut out = String::new();
            out.push_str(&format!(
                "diverged at step {} field {}\n",
                divergence.step_index, divergence.field
            ));
            out.push_str(&format!(
                "ideal: position=b{}:{}:{} cursor={} trap={:?} returned={} output={}\n",
                divergence.ideal.position.block.0,
                divergence.ideal.position.next_inst_index,
                divergence.ideal.position.at_terminator,
                divergence.ideal.cursor,
                divergence.ideal.trap,
                divergence.ideal.returned,
                bytes_to_hex(&divergence.ideal.output)
            ));
            out.push_str(&format!(
                "post:  position=b{}:{}:{} cursor={} trap={:?} returned={} output={}\n",
                divergence.post_regalloc.position.block.0,
                divergence.post_regalloc.position.next_inst_index,
                divergence.post_regalloc.position.at_terminator,
                divergence.post_regalloc.cursor,
                divergence.post_regalloc.trap,
                divergence.post_regalloc.returned,
                bytes_to_hex(&divergence.post_regalloc.output)
            ));
            out
        }
        DifferentialCheckResult::Error(message) => {
            format!("differential error: {message}\n")
        }
    }
}

fn bytes_to_hex(bytes: &[u8]) -> String {
    let mut out = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        use std::fmt::Write as _;
        write!(&mut out, "{byte:02x}").expect("writing to String should not fail");
    }
    out
}
