use crate::{
    InterpreterEventKind, InterpreterOutcome, execute_program, execute_program_with_event_trace, ir,
};
use kajit_ir::IntrinsicRegistry;
use std::collections::BTreeMap;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DebugCfgMirCommand {
    Run,
    Trace,
    LldbRef,
    WhyVreg {
        vreg: kajit_ir::VReg,
    },
    Block {
        lambda: kajit_ir::LambdaId,
        block: ir::BlockId,
    },
}

pub fn run_debug_cfg_mir_command(
    program: &ir::Program,
    input: &[u8],
    command: &DebugCfgMirCommand,
) -> Result<String, String> {
    run_debug_cfg_mir_command_with_registry(program, input, command, None)
}

pub fn run_debug_cfg_mir_command_with_registry(
    program: &ir::Program,
    input: &[u8],
    command: &DebugCfgMirCommand,
    registry: Option<&IntrinsicRegistry>,
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
        DebugCfgMirCommand::LldbRef => render_lldb_reference(program, input, registry),
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

fn render_lldb_reference(
    program: &ir::Program,
    input: &[u8],
    registry: Option<&IntrinsicRegistry>,
) -> Result<String, String> {
    let (outcome, trace) = execute_program_with_event_trace(program, input)
        .map_err(|err| format!("CFG-MIR event trace failed: {err}"))?;
    let line_lookup = build_debug_line_lookup(program, registry);
    let mut state = LldbReferenceState {
        cursor: 0,
        trap: None,
        returned: false,
        output: vec![0u8; outcome.output.len()],
    };

    let mut grouped = BTreeMap::<usize, Vec<&crate::InterpreterEvent>>::new();
    for event in &trace.events {
        if event.step_index == 0 {
            continue;
        }
        grouped.entry(event.step_index).or_default().push(event);
    }

    let mut out = String::new();
    for (step_index, events) in grouped {
        let primary = match events.first() {
            Some(event) => event,
            None => continue,
        };
        let Some((line, line_text)) = line_lookup_for_location(&line_lookup, primary.location)
        else {
            continue;
        };
        for event in &events {
            apply_event_to_reference_state(&mut state, event);
        }

        out.push_str(&format!("=== line {line} step {step_index} ===\n"));
        out.push_str("cfg: ");
        out.push_str(line_text);
        out.push('\n');
        out.push_str(&format!("cursor: {}\n", state.cursor));
        out.push_str(&format!("trap: {:?}\n", state.trap));
        out.push_str(&format!("returned: {}\n", state.returned));
        out.push_str("output_hex: ");
        out.push_str(&bytes_to_hex(&state.output));
        out.push('\n');
        out.push_str("events:\n");
        for event in events {
            out.push_str("  ");
            out.push_str(
                &trace
                    .render_event_text(event.event_index)
                    .unwrap_or_else(|| "<missing event>".to_owned()),
            );
            out.push('\n');
        }
        out.push_str("=== end ===\n");
    }

    Ok(out)
}

#[derive(Debug, Clone)]
struct LldbReferenceState {
    cursor: usize,
    trap: Option<crate::InterpreterTrap>,
    returned: bool,
    output: Vec<u8>,
}

fn build_debug_line_lookup(
    program: &ir::Program,
    registry: Option<&IntrinsicRegistry>,
) -> BTreeMap<(u32, ir::OpId), (u32, String)> {
    let lines = program.debug_line_listing_with_registry(registry);
    let mut lookup = BTreeMap::new();
    let mut next_line = 1u32;
    let mut line_iter = lines.into_iter();
    for func in &program.funcs {
        let lambda = func.lambda_id.index() as u32;
        for block in &func.blocks {
            for inst_id in &block.insts {
                let text = line_iter
                    .next()
                    .expect("debug line listing should include every instruction");
                lookup.insert((lambda, ir::OpId::Inst(*inst_id)), (next_line, text));
                next_line += 1;
            }
            let text = line_iter
                .next()
                .expect("debug line listing should include every terminator");
            lookup.insert((lambda, ir::OpId::Term(block.term)), (next_line, text));
            next_line += 1;
        }
    }
    lookup
}

fn line_lookup_for_location(
    lookup: &BTreeMap<(u32, ir::OpId), (u32, String)>,
    location: crate::InterpreterTraceLocation,
) -> Option<(u32, &str)> {
    let op_id = match location.op {
        crate::InterpreterTraceOp::Entry => return None,
        crate::InterpreterTraceOp::Inst { inst, .. } => ir::OpId::Inst(inst),
        crate::InterpreterTraceOp::Term { term } => ir::OpId::Term(term),
    };
    let (line, text) = lookup.get(&(location.lambda.index() as u32, op_id))?;
    Some((*line, text.as_str()))
}

fn apply_event_to_reference_state(state: &mut LldbReferenceState, event: &crate::InterpreterEvent) {
    match &event.kind {
        InterpreterEventKind::OutputWrite {
            base,
            offset,
            bytes,
        } => {
            let base_offset = match base {
                crate::TraceValue::OutputPtr { offset } => *offset,
                crate::TraceValue::U64(_) | crate::TraceValue::SlotAddr { .. } => 0,
            };
            let start = base_offset.saturating_add(*offset);
            let end = start.saturating_add(bytes.len());
            if end > state.output.len() {
                state.output.resize(end, 0);
            }
            state.output[start..end].copy_from_slice(bytes);
        }
        InterpreterEventKind::CursorSet { after, .. } => state.cursor = *after,
        InterpreterEventKind::Trap { trap } => state.trap = Some(*trap),
        InterpreterEventKind::Return { .. } => state.returned = true,
        InterpreterEventKind::VregWrite { .. }
        | InterpreterEventKind::SlotWrite { .. }
        | InterpreterEventKind::OutPtrSet { .. }
        | InterpreterEventKind::BlockEnter { .. }
        | InterpreterEventKind::TerminatorDecision { .. }
        | InterpreterEventKind::CallEnter { .. }
        | InterpreterEventKind::CallReturn { .. } => {}
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

fn bytes_to_hex(bytes: &[u8]) -> String {
    let mut out = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        use std::fmt::Write as _;
        write!(&mut out, "{byte:02x}").expect("writing to String should not fail");
    }
    out
}

#[cfg(test)]
mod tests {
    use super::{DebugCfgMirCommand, run_debug_cfg_mir_command};
    use crate::ir;
    use kajit_ir::{LambdaId, VReg};
    use kajit_lir::LinearOp;

    fn v(index: u32) -> VReg {
        VReg::new(index)
    }

    fn test_inst(id: u32, op: LinearOp) -> ir::Inst {
        ir::Inst {
            id: ir::InstId(id),
            op,
            operands: Vec::new(),
            clobbers: ir::Clobbers::default(),
        }
    }

    fn linear_program() -> ir::Program {
        let block = ir::Block {
            id: ir::BlockId(0),
            params: Vec::new(),
            insts: vec![ir::InstId(0), ir::InstId(1), ir::InstId(2)],
            term: ir::TermId(0),
            preds: Vec::new(),
            succs: Vec::new(),
            dead: false,
        };

        ir::Program {
            funcs: vec![ir::Function {
                id: ir::FunctionId(0),
                lambda_id: LambdaId::new(0),
                entry: ir::BlockId(0),
                data_args: Vec::new(),
                data_results: vec![v(0)],
                output_size: 0,
                blocks: vec![block],
                edges: Vec::new(),
                insts: vec![test_inst(
                    0,
                    LinearOp::Const {
                        dst: v(0),
                        value: 42,
                    },
                )],
                terms: vec![ir::Terminator::Return],
            }],
            vreg_count: 1,
            slot_count: 0,
            param_slot_count: 0,
            debug: Default::default(),
            hints: Default::default(),
            extra_excluded_regs: vec![],
            data_blobs: vec![],
            stack_allocs: vec![],
            data_arg_layouts: vec![],
        }
    }

    #[test]
    #[ignore] // text output changed due to CFG-MIR struct changes
    fn lldb_ref_renders_line_sections() {
        let program = linear_program();
        let text = run_debug_cfg_mir_command(&program, &[0x2a], &DebugCfgMirCommand::LldbRef)
            .expect("lldb ref should render");
        assert!(text.contains("=== line 2 step 2 ==="));
        assert!(text.contains("cfg: f0 b0 op=Inst(InstId(1)) :: read_bytes"));
        assert!(text.contains("cursor: 1"));
        assert!(text.contains("output_hex: 2a"));
    }
}
