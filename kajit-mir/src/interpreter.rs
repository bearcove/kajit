use std::collections::HashMap;

use kajit_ir::ErrorCode;
use kajit_lir::{BinOpKind, LinearOp, UnaryOpKind};

use crate::cfg_mir;

const MAX_EXEC_STEPS: usize = 1_000_000;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct InterpreterTrap {
    pub code: ErrorCode,
    pub offset: u32,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct InterpreterOutcome {
    pub vregs: Vec<u64>,
    pub output: Vec<u8>,
    pub cursor: usize,
    pub trap: Option<InterpreterTrap>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TraceValue {
    U64(u64),
    OutputPtr { offset: usize },
    SlotAddr { slot: usize },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InterpreterTraceOp {
    Entry,
    Inst {
        inst: cfg_mir::InstId,
        inst_index: usize,
    },
    Term {
        term: cfg_mir::TermId,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct InterpreterTraceLocation {
    pub lambda: kajit_ir::LambdaId,
    pub block: cfg_mir::BlockId,
    pub op: InterpreterTraceOp,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum InterpreterEventKind {
    VregWrite {
        vreg: kajit_ir::VReg,
        value: TraceValue,
    },
    SlotWrite {
        slot: kajit_ir::SlotId,
        value: TraceValue,
    },
    OutputWrite {
        base: TraceValue,
        offset: usize,
        bytes: Vec<u8>,
    },
    CursorSet {
        before: usize,
        after: usize,
    },
    OutPtrSet {
        before: TraceValue,
        after: TraceValue,
    },
    BlockEnter {
        via_edge: Option<cfg_mir::EdgeId>,
        target: cfg_mir::BlockId,
    },
    TerminatorDecision {
        detail: String,
    },
    CallEnter {
        target: kajit_ir::LambdaId,
    },
    CallReturn {
        target: kajit_ir::LambdaId,
        results: Vec<TraceValue>,
    },
    Trap {
        trap: InterpreterTrap,
    },
    Return {
        results: Vec<TraceValue>,
    },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct InterpreterEvent {
    pub event_index: usize,
    pub step_index: usize,
    pub location: InterpreterTraceLocation,
    pub kind: InterpreterEventKind,
}

#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct InterpreterExecutionTrace {
    pub events: Vec<InterpreterEvent>,
}

impl InterpreterExecutionTrace {
    pub fn writes_to_vreg(&self, vreg: kajit_ir::VReg) -> Vec<&InterpreterEvent> {
        self.events
            .iter()
            .filter(|event| {
                matches!(
                    event.kind,
                    InterpreterEventKind::VregWrite { vreg: written, .. } if written == vreg
                )
            })
            .collect()
    }

    pub fn last_write_to_vreg(&self, vreg: kajit_ir::VReg) -> Option<&InterpreterEvent> {
        self.events.iter().rev().find(|event| {
            matches!(
                event.kind,
                InterpreterEventKind::VregWrite { vreg: written, .. } if written == vreg
            )
        })
    }

    pub fn entries_to_block(
        &self,
        lambda: kajit_ir::LambdaId,
        block: cfg_mir::BlockId,
    ) -> Vec<&InterpreterEvent> {
        self.events
            .iter()
            .filter(|event| {
                event.location.lambda == lambda
                    && matches!(
                        event.kind,
                        InterpreterEventKind::BlockEnter { target, .. } if target == block
                    )
            })
            .collect()
    }

    pub fn render_text(&self) -> String {
        self.events
            .iter()
            .map(render_interpreter_event)
            .collect::<Vec<_>>()
            .join("\n")
    }

    pub fn render_event_text(&self, event_index: usize) -> Option<String> {
        self.events.get(event_index).map(render_interpreter_event)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct InterpreterTraceEntry {
    pub step_index: usize,
    pub block: cfg_mir::BlockId,
    pub next_inst_index: usize,
    pub at_terminator: bool,
    pub cursor: usize,
    pub vregs: Vec<u64>,
    pub output: Vec<u8>,
    pub trap: Option<InterpreterTrap>,
    pub returned: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum InterpreterError {
    NoFunctions,
    UnknownLambda {
        lambda: usize,
    },
    UnknownBlock {
        block: cfg_mir::BlockId,
    },
    UnknownEdge {
        edge: cfg_mir::EdgeId,
    },
    UnknownInst {
        inst: cfg_mir::InstId,
    },
    UnknownTerm {
        term: cfg_mir::TermId,
    },
    MissingEdge {
        from: cfg_mir::BlockId,
        to: cfg_mir::BlockId,
    },
    EdgeArgArityMismatch {
        from: cfg_mir::BlockId,
        to: cfg_mir::BlockId,
        expected: usize,
        got: usize,
    },
    UnsupportedOp {
        block: cfg_mir::BlockId,
        op: String,
    },
    UnsupportedTerminator {
        block: cfg_mir::BlockId,
        term: String,
    },
    StepLimitExceeded {
        limit: usize,
    },
}

impl std::fmt::Display for InterpreterError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NoFunctions => write!(f, "CFG-MIR program has no functions"),
            Self::UnknownLambda { lambda } => write!(f, "unknown lambda @{lambda}"),
            Self::UnknownBlock { block } => write!(f, "unknown block b{}", block.0),
            Self::UnknownEdge { edge } => write!(f, "unknown edge e{}", edge.0),
            Self::UnknownInst { inst } => write!(f, "unknown inst i{}", inst.0),
            Self::UnknownTerm { term } => write!(f, "unknown term t{}", term.0),
            Self::MissingEdge { from, to } => {
                write!(f, "missing CFG edge b{} -> b{}", from.0, to.0)
            }
            Self::EdgeArgArityMismatch {
                from,
                to,
                expected,
                got,
            } => write!(
                f,
                "edge arg arity mismatch on b{} -> b{}: expected {}, got {}",
                from.0, to.0, expected, got
            ),
            Self::UnsupportedOp { block, op } => {
                write!(
                    f,
                    "unsupported CFG-MIR op in block b{} for interpreter MVP: {}",
                    block.0, op
                )
            }
            Self::UnsupportedTerminator { block, term } => write!(
                f,
                "unsupported CFG-MIR terminator in block b{} for interpreter MVP: {}",
                block.0, term
            ),
            Self::StepLimitExceeded { limit } => {
                write!(f, "CFG-MIR interpreter exceeded step limit ({limit})")
            }
        }
    }
}

impl std::error::Error for InterpreterError {}

struct InterpreterState<'a> {
    input: &'a [u8],
    input_base: *const u8,
    cursor: usize,
    output: Vec<u8>,
    out_ptr: *mut u8,
    vregs: Vec<u64>,
    trace_vregs: Vec<TraceValue>,
    slots: Vec<u64>,
    trace_slots: Vec<TraceValue>,
    slot_mem: Vec<u8>,
    ctx: RuntimeDeserContext,
    trap: Option<InterpreterTrap>,
    trace_out_ptr: TraceValue,
}

impl<'a> InterpreterState<'a> {
    fn new(input: &'a [u8], vreg_count: usize, slot_count: usize, output_size: usize) -> Self {
        let input_base = input.as_ptr();
        let mut output = vec![0u8; output_size];
        let out_ptr = output.as_mut_ptr();
        let slot_mem = vec![0u8; slot_count.saturating_mul(SLOT_ADDR_STRIDE)];
        Self {
            input,
            input_base,
            cursor: 0,
            output,
            out_ptr,
            vregs: vec![0u64; vreg_count],
            trace_vregs: vec![TraceValue::U64(0); vreg_count],
            slots: vec![0u64; slot_count],
            trace_slots: vec![TraceValue::U64(0); slot_count],
            slot_mem,
            ctx: RuntimeDeserContext::new(input),
            trap: None,
            trace_out_ptr: TraceValue::OutputPtr { offset: 0 },
        }
    }

    fn read_vreg(&self, idx: usize) -> u64 {
        self.vregs.get(idx).copied().unwrap_or(0)
    }

    fn write_vreg(&mut self, idx: usize, value: u64) {
        if idx >= self.vregs.len() {
            self.vregs.resize(idx + 1, 0);
            self.trace_vregs.resize(idx + 1, TraceValue::U64(0));
        }
        self.vregs[idx] = value;
    }

    fn read_trace_vreg(&self, idx: usize) -> TraceValue {
        self.trace_vregs
            .get(idx)
            .cloned()
            .unwrap_or(TraceValue::U64(0))
    }

    fn write_trace_vreg(&mut self, idx: usize, value: TraceValue) {
        if idx >= self.trace_vregs.len() {
            self.trace_vregs.resize(idx + 1, TraceValue::U64(0));
        }
        self.trace_vregs[idx] = value;
    }

    fn ensure_output_len(&mut self, len: usize) {
        if self.output.len() < len {
            self.output.resize(len, 0);
        }
    }

    fn out_ptr_in_output(&self) -> Option<usize> {
        let base = self.output.as_ptr() as usize;
        let end = base + self.output.len();
        let ptr = self.out_ptr as usize;
        if ptr >= base && ptr <= end {
            Some(ptr - base)
        } else {
            None
        }
    }

    fn ensure_output_range_for_out_ptr(&mut self, offset: usize, width_bytes: usize) {
        let Some(ptr_offset) = self.out_ptr_in_output() else {
            return;
        };
        let needed = ptr_offset
            .saturating_add(offset)
            .saturating_add(width_bytes);
        self.ensure_output_len(needed);
        self.out_ptr = unsafe { self.output.as_mut_ptr().add(ptr_offset) };
    }

    fn trap(&mut self, code: ErrorCode) {
        if self.trap.is_none() {
            self.trap = Some(InterpreterTrap {
                code,
                offset: self.cursor as u32,
            });
        }
    }

    fn ensure_slot(&mut self, slot: usize) {
        if slot >= self.slots.len() {
            self.slots.resize(slot + 1, 0);
            self.trace_slots.resize(slot + 1, TraceValue::U64(0));
        }
        let needed_bytes = (slot + 1) * SLOT_ADDR_STRIDE;
        if self.slot_mem.len() < needed_bytes {
            self.slot_mem.resize(needed_bytes, 0);
        }
    }

    fn slot_addr_value(&mut self, slot: usize) -> u64 {
        self.ensure_slot(slot);
        let ptr = self.slot_mem.as_mut_ptr();
        unsafe { ptr.add(slot * SLOT_ADDR_STRIDE) as u64 }
    }
}

fn format_trace_value(value: &TraceValue) -> String {
    match value {
        TraceValue::U64(value) => format!("{value:#x}"),
        TraceValue::OutputPtr { offset } => format!("output+{offset}"),
        TraceValue::SlotAddr { slot } => format!("slot[{slot}]"),
    }
}

fn render_interpreter_event(event: &InterpreterEvent) -> String {
    let op = match event.location.op {
        InterpreterTraceOp::Entry => "entry".to_string(),
        InterpreterTraceOp::Inst { inst, inst_index } => format!("inst i{}#{inst_index}", inst.0),
        InterpreterTraceOp::Term { term } => format!("term t{}", term.0),
    };
    let detail = match &event.kind {
        InterpreterEventKind::VregWrite { vreg, value } => {
            format!("write v{} = {}", vreg.index(), format_trace_value(value))
        }
        InterpreterEventKind::SlotWrite { slot, value } => {
            format!("write slot{} = {}", slot.index(), format_trace_value(value))
        }
        InterpreterEventKind::OutputWrite {
            base,
            offset,
            bytes,
        } => {
            let bytes = bytes
                .iter()
                .map(|byte| format!("{byte:02x}"))
                .collect::<String>();
            format!("write {}+{offset} [{}]", format_trace_value(base), bytes)
        }
        InterpreterEventKind::CursorSet { before, after } => {
            format!("cursor {before} -> {after}")
        }
        InterpreterEventKind::OutPtrSet { before, after } => format!(
            "out_ptr {} -> {}",
            format_trace_value(before),
            format_trace_value(after)
        ),
        InterpreterEventKind::BlockEnter { via_edge, target } => match via_edge {
            Some(edge) => format!("enter b{} via e{}", target.0, edge.0),
            None => format!("enter b{}", target.0),
        },
        InterpreterEventKind::TerminatorDecision { detail } => detail.clone(),
        InterpreterEventKind::CallEnter { target } => format!("call @{}", target.index()),
        InterpreterEventKind::CallReturn { target, results } => format!(
            "return from @{} [{}]",
            target.index(),
            results
                .iter()
                .map(format_trace_value)
                .collect::<Vec<_>>()
                .join(", ")
        ),
        InterpreterEventKind::Trap { trap } => {
            format!("trap {:?} @{}", trap.code, trap.offset)
        }
        InterpreterEventKind::Return { results } => format!(
            "return [{}]",
            results
                .iter()
                .map(format_trace_value)
                .collect::<Vec<_>>()
                .join(", ")
        ),
    };
    format!(
        "#{:04} step={:04} @{} b{} {} :: {}",
        event.event_index,
        event.step_index,
        event.location.lambda.index(),
        event.location.block.0,
        op,
        detail
    )
}

fn push_interpreter_event(
    trace: &mut InterpreterExecutionTrace,
    step_index: usize,
    location: InterpreterTraceLocation,
    kind: InterpreterEventKind,
) {
    let event_index = trace.events.len();
    trace.events.push(InterpreterEvent {
        event_index,
        step_index,
        location,
        kind,
    });
}

fn diff_output_regions(before: &[u8], after: &[u8]) -> Vec<(usize, Vec<u8>)> {
    let mut regions = Vec::new();
    let mut start = None::<usize>;
    let max_len = before.len().max(after.len());
    for idx in 0..max_len {
        let before_byte = before.get(idx).copied().unwrap_or(0);
        let after_byte = after.get(idx).copied().unwrap_or(0);
        if before_byte != after_byte {
            start.get_or_insert(idx);
        } else if let Some(region_start) = start.take() {
            regions.push((region_start, after[region_start..idx].to_vec()));
        }
    }
    if let Some(region_start) = start {
        regions.push((region_start, after[region_start..max_len].to_vec()));
    }
    regions
}

/// Execute the first function in a CFG-MIR program.
pub fn execute_program(
    program: &cfg_mir::Program,
    input: &[u8],
) -> Result<InterpreterOutcome, InterpreterError> {
    execute_program_with_trace(program, input).map(|(outcome, _)| outcome)
}

/// Execute a single CFG-MIR function.
pub fn execute_function(
    func: &cfg_mir::Function,
    vreg_count: usize,
    slot_count: usize,
    input: &[u8],
) -> Result<InterpreterOutcome, InterpreterError> {
    execute_function_with_trace(func, vreg_count, slot_count, input).map(|(outcome, _)| outcome)
}

pub fn execute_program_with_event_trace(
    program: &cfg_mir::Program,
    input: &[u8],
) -> Result<(InterpreterOutcome, InterpreterExecutionTrace), InterpreterError> {
    if program.funcs.is_empty() {
        return Err(InterpreterError::NoFunctions);
    }
    let mut lambda_indices = HashMap::with_capacity(program.funcs.len());
    for (index, func) in program.funcs.iter().enumerate() {
        lambda_indices.insert(func.lambda_id.index(), index);
    }

    let mut state = InterpreterState::new(
        input,
        program.vreg_count as usize,
        program.slot_count as usize,
        infer_program_output_size(program),
    );
    let mut trace = InterpreterExecutionTrace::default();
    let mut steps = 0usize;
    let root = &program.funcs[0];
    push_interpreter_event(
        &mut trace,
        0,
        InterpreterTraceLocation {
            lambda: root.lambda_id,
            block: root.entry,
            op: InterpreterTraceOp::Entry,
        },
        InterpreterEventKind::BlockEnter {
            via_edge: None,
            target: root.entry,
        },
    );
    let _ = execute_function_inner_with_event_trace(
        program,
        &lambda_indices,
        0,
        &mut state,
        &mut trace,
        &mut steps,
    )?;
    let outcome = InterpreterOutcome {
        vregs: state.vregs,
        output: state.output,
        cursor: state.cursor,
        trap: state.trap,
    };
    Ok((outcome, trace))
}

pub fn execute_function_with_event_trace(
    func: &cfg_mir::Function,
    vreg_count: usize,
    slot_count: usize,
    input: &[u8],
) -> Result<(InterpreterOutcome, InterpreterExecutionTrace), InterpreterError> {
    let program = cfg_mir::Program {
        funcs: vec![func.clone()],
        vreg_count: vreg_count as u32,
        slot_count: slot_count as u32,
        debug: Default::default(),
    };
    execute_program_with_event_trace(&program, input)
}

pub fn execute_program_with_trace(
    program: &cfg_mir::Program,
    input: &[u8],
) -> Result<(InterpreterOutcome, Vec<InterpreterTraceEntry>), InterpreterError> {
    if program.funcs.is_empty() {
        return Err(InterpreterError::NoFunctions);
    }
    let mut lambda_indices = HashMap::with_capacity(program.funcs.len());
    for (index, func) in program.funcs.iter().enumerate() {
        lambda_indices.insert(func.lambda_id.index(), index);
    }

    let mut state = InterpreterState::new(
        input,
        program.vreg_count as usize,
        program.slot_count as usize,
        infer_program_output_size(program),
    );
    let mut trace = Vec::new();
    let mut steps = 0usize;
    let _ = execute_function_inner(
        program,
        &lambda_indices,
        0,
        &mut state,
        &mut trace,
        &mut steps,
    )?;
    let outcome = InterpreterOutcome {
        vregs: state.vregs,
        output: state.output,
        cursor: state.cursor,
        trap: state.trap,
    };
    Ok((outcome, trace))
}

pub fn execute_function_with_trace(
    func: &cfg_mir::Function,
    vreg_count: usize,
    slot_count: usize,
    input: &[u8],
) -> Result<(InterpreterOutcome, Vec<InterpreterTraceEntry>), InterpreterError> {
    let program = cfg_mir::Program {
        funcs: vec![func.clone()],
        vreg_count: vreg_count as u32,
        slot_count: slot_count as u32,
        debug: Default::default(),
    };
    execute_program_with_trace(&program, input)
}

fn execute_function_inner(
    program: &cfg_mir::Program,
    lambda_indices: &HashMap<usize, usize>,
    func_index: usize,
    state: &mut InterpreterState<'_>,
    trace: &mut Vec<InterpreterTraceEntry>,
    steps: &mut usize,
) -> Result<Vec<u64>, InterpreterError> {
    let func = program
        .funcs
        .get(func_index)
        .ok_or(InterpreterError::NoFunctions)?;
    let block_indices = build_block_index(func);
    let mut current = func.entry;

    loop {
        if *steps >= MAX_EXEC_STEPS {
            return Err(InterpreterError::StepLimitExceeded {
                limit: MAX_EXEC_STEPS,
            });
        }
        *steps += 1;
        let step_index = *steps;

        let block_idx = *block_indices
            .get(&current)
            .ok_or(InterpreterError::UnknownBlock { block: current })?;
        let block = &func.blocks[block_idx];

        for (inst_index, inst_id) in block.insts.iter().copied().enumerate() {
            let inst = func
                .inst(inst_id)
                .ok_or(InterpreterError::UnknownInst { inst: inst_id })?;
            match &inst.op {
                LinearOp::Const { dst, value } => state.write_vreg(dst.index(), *value),
                LinearOp::Copy { dst, src } => {
                    let value = state.read_vreg(src.index());
                    state.write_vreg(dst.index(), value);
                }
                LinearOp::BinOp { op, dst, lhs, rhs } => {
                    let lhs = state.read_vreg(lhs.index());
                    let rhs = state.read_vreg(rhs.index());
                    let value = exec_binop(*op, lhs, rhs);
                    state.write_vreg(dst.index(), value);
                }
                LinearOp::UnaryOp { op, dst, src } => {
                    let src = state.read_vreg(src.index());
                    let value = exec_unaryop(*op, src);
                    state.write_vreg(dst.index(), value);
                }
                LinearOp::BoundsCheck { count } => {
                    let count = *count as usize;
                    if state.cursor + count > state.input.len() {
                        state.trap(ErrorCode::UnexpectedEof);
                        break;
                    }
                }
                LinearOp::ReadBytes { dst, count } => {
                    let count = *count as usize;
                    if state.cursor + count > state.input.len() {
                        state.trap(ErrorCode::UnexpectedEof);
                        break;
                    }
                    let mut value = 0u64;
                    for i in 0..count {
                        value |= (state.input[state.cursor + i] as u64) << (i * 8);
                    }
                    state.cursor += count;
                    state.write_vreg(dst.index(), value);
                }
                LinearOp::PeekByte { dst } => {
                    if state.cursor >= state.input.len() {
                        state.trap(ErrorCode::UnexpectedEof);
                        break;
                    }
                    state.write_vreg(dst.index(), state.input[state.cursor] as u64);
                }
                LinearOp::AdvanceCursor { count } => {
                    let count = *count as usize;
                    if state.cursor + count > state.input.len() {
                        state.trap(ErrorCode::UnexpectedEof);
                        break;
                    }
                    state.cursor += count;
                }
                LinearOp::AdvanceCursorBy { src } => {
                    let count = state.read_vreg(src.index()) as usize;
                    if state.cursor + count > state.input.len() {
                        state.trap(ErrorCode::UnexpectedEof);
                        break;
                    }
                    state.cursor += count;
                }
                LinearOp::SaveCursor { dst } => {
                    state.write_vreg(dst.index(), state.cursor as u64);
                }
                LinearOp::RestoreCursor { src } => {
                    let next = state.read_vreg(src.index()) as usize;
                    if next > state.input.len() {
                        state.trap(ErrorCode::UnexpectedEof);
                        break;
                    }
                    state.cursor = next;
                }
                LinearOp::WriteToField { src, offset, width } => {
                    let value = state.read_vreg(src.index());
                    let width_bytes = width.bytes() as usize;
                    let base = *offset as usize;
                    state.ensure_output_range_for_out_ptr(base, width_bytes);
                    unsafe {
                        let dst = state.out_ptr.add(base);
                        for i in 0..width_bytes {
                            *dst.add(i) = ((value >> (i * 8)) & 0xff) as u8;
                        }
                    }
                }
                LinearOp::ErrorExit { code } => {
                    state.trap(*code);
                    break;
                }
                LinearOp::ReadFromField { dst, offset, width } => {
                    let width_bytes = width.bytes() as usize;
                    let base = *offset as usize;
                    state.ensure_output_range_for_out_ptr(base, width_bytes);
                    let mut value = 0u64;
                    unsafe {
                        let src = state.out_ptr.add(base);
                        for i in 0..width_bytes {
                            value |= (*src.add(i) as u64) << (i * 8);
                        }
                    }
                    state.write_vreg(dst.index(), value);
                }
                LinearOp::SaveOutPtr { dst } => {
                    state.write_vreg(dst.index(), state.out_ptr as u64);
                }
                LinearOp::SetOutPtr { src } => {
                    state.out_ptr = state.read_vreg(src.index()) as *mut u8;
                }
                LinearOp::SlotAddr { dst, slot } => {
                    let addr = state.slot_addr_value(slot.index());
                    state.write_vreg(dst.index(), addr);
                }
                LinearOp::WriteToSlot { slot, src } => {
                    let slot = slot.index();
                    state.ensure_slot(slot);
                    state.slots[slot] = state.read_vreg(src.index());
                }
                LinearOp::ReadFromSlot { dst, slot } => {
                    let slot = slot.index();
                    state.ensure_slot(slot);
                    state.write_vreg(dst.index(), state.slots[slot]);
                }
                LinearOp::CallIntrinsic {
                    func,
                    args,
                    dst,
                    field_offset,
                } => {
                    let args_values: Vec<u64> =
                        args.iter().map(|v| state.read_vreg(v.index())).collect();
                    let out_ptr = if dst.is_none() {
                        let offset = *field_offset as usize;
                        state.ensure_output_range_for_out_ptr(offset, 64);
                        Some(unsafe { state.out_ptr.add(offset) })
                    } else {
                        None
                    };
                    let ret = run_call_intrinsic(state, func.0, &args_values, out_ptr);
                    if state.trap.is_some() {
                        break;
                    }
                    if let Some(dst) = dst {
                        state.write_vreg(dst.index(), ret);
                    }
                }
                LinearOp::CallPure { func, args, dst } => {
                    let args_values: Vec<u64> =
                        args.iter().map(|v| state.read_vreg(v.index())).collect();
                    let ret = run_call_pure(func.0, &args_values);
                    state.write_vreg(dst.index(), ret);
                }
                LinearOp::CallLambda {
                    target,
                    args,
                    results,
                } => {
                    let target_index = if let Some(&target_index) =
                        lambda_indices.get(&target.index())
                    {
                        target_index
                    } else {
                        let mut known = lambda_indices.keys().copied().collect::<Vec<_>>();
                        known.sort_unstable();
                        return Err(InterpreterError::UnsupportedOp {
                            block: block.id,
                            op: format!(
                                "CallLambda @{} has no target function; known lambdas: {known:?}",
                                target.index()
                            ),
                        });
                    };
                    let target_func = &program.funcs[target_index];
                    if target_func.data_args.len() != args.len() {
                        return Err(InterpreterError::UnsupportedOp {
                            block: block.id,
                            op: format!(
                                "CallLambda @{} arg arity mismatch: expected {}, got {}",
                                target.index(),
                                target_func.data_args.len(),
                                args.len()
                            ),
                        });
                    }

                    let arg_values = args
                        .iter()
                        .map(|v| state.read_vreg(v.index()))
                        .collect::<Vec<_>>();
                    for (param, value) in target_func.data_args.iter().zip(arg_values) {
                        state.write_vreg(param.index(), value);
                    }
                    let caller_out_ptr = state.out_ptr;
                    let callee_results = execute_function_inner(
                        program,
                        lambda_indices,
                        target_index,
                        state,
                        trace,
                        steps,
                    )?;
                    state.out_ptr = caller_out_ptr;
                    if state.trap.is_some() {
                        break;
                    }
                    if callee_results.len() != results.len() {
                        return Err(InterpreterError::UnsupportedOp {
                            block: block.id,
                            op: format!(
                                "CallLambda @{} result arity mismatch: expected {}, got {}",
                                target.index(),
                                results.len(),
                                callee_results.len()
                            ),
                        });
                    }
                    for (dst, value) in results.iter().zip(callee_results.iter()) {
                        state.write_vreg(dst.index(), *value);
                    }
                }
                op => {
                    return Err(InterpreterError::UnsupportedOp {
                        block: block.id,
                        op: format!("{op:?}"),
                    });
                }
            }

            trace.push(InterpreterTraceEntry {
                step_index,
                block: block.id,
                next_inst_index: inst_index + 1,
                at_terminator: inst_index + 1 >= block.insts.len(),
                cursor: state.cursor,
                vregs: state.vregs.clone(),
                output: state.output.clone(),
                trap: state.trap,
                returned: false,
            });
        }

        if state.trap.is_some() {
            return Ok(Vec::new());
        }

        let term = func
            .term(block.term)
            .ok_or(InterpreterError::UnknownTerm { term: block.term })?;
        match term {
            cfg_mir::Terminator::Return => {
                trace.push(InterpreterTraceEntry {
                    step_index,
                    block: block.id,
                    next_inst_index: block.insts.len(),
                    at_terminator: true,
                    cursor: state.cursor,
                    vregs: state.vregs.clone(),
                    output: state.output.clone(),
                    trap: None,
                    returned: true,
                });
                let data_results = func
                    .data_results
                    .iter()
                    .map(|vreg| state.read_vreg(vreg.index()))
                    .collect::<Vec<_>>();
                return Ok(data_results);
            }
            cfg_mir::Terminator::ErrorExit { code } => {
                state.trap(*code);
                trace.push(InterpreterTraceEntry {
                    step_index,
                    block: block.id,
                    next_inst_index: block.insts.len(),
                    at_terminator: true,
                    cursor: state.cursor,
                    vregs: state.vregs.clone(),
                    output: state.output.clone(),
                    trap: state.trap,
                    returned: false,
                });
                return Ok(Vec::new());
            }
            cfg_mir::Terminator::Branch { edge } => {
                let target = apply_edge(func, &block_indices, state, *edge)?;
                let target_block_idx = *block_indices
                    .get(&target)
                    .ok_or(InterpreterError::UnknownBlock { block: target })?;
                let at_terminator = 0 >= func.blocks[target_block_idx].insts.len();
                trace.push(InterpreterTraceEntry {
                    step_index,
                    block: target,
                    next_inst_index: 0,
                    at_terminator,
                    cursor: state.cursor,
                    vregs: state.vregs.clone(),
                    output: state.output.clone(),
                    trap: state.trap,
                    returned: false,
                });
                current = target;
            }
            cfg_mir::Terminator::BranchIf {
                cond,
                taken,
                fallthrough,
            } => {
                let branch = state.read_vreg(cond.index()) != 0;
                let edge = if branch { *taken } else { *fallthrough };
                let next = apply_edge(func, &block_indices, state, edge)?;
                let target_block_idx = *block_indices
                    .get(&next)
                    .ok_or(InterpreterError::UnknownBlock { block: next })?;
                let at_terminator = 0 >= func.blocks[target_block_idx].insts.len();
                trace.push(InterpreterTraceEntry {
                    step_index,
                    block: next,
                    next_inst_index: 0,
                    at_terminator,
                    cursor: state.cursor,
                    vregs: state.vregs.clone(),
                    output: state.output.clone(),
                    trap: state.trap,
                    returned: false,
                });
                current = next;
            }
            cfg_mir::Terminator::BranchIfZero {
                cond,
                taken,
                fallthrough,
            } => {
                let branch = state.read_vreg(cond.index()) == 0;
                let edge = if branch { *taken } else { *fallthrough };
                let next = apply_edge(func, &block_indices, state, edge)?;
                let target_block_idx = *block_indices
                    .get(&next)
                    .ok_or(InterpreterError::UnknownBlock { block: next })?;
                let at_terminator = 0 >= func.blocks[target_block_idx].insts.len();
                trace.push(InterpreterTraceEntry {
                    step_index,
                    block: next,
                    next_inst_index: 0,
                    at_terminator,
                    cursor: state.cursor,
                    vregs: state.vregs.clone(),
                    output: state.output.clone(),
                    trap: state.trap,
                    returned: false,
                });
                current = next;
            }
            term => {
                return Err(InterpreterError::UnsupportedTerminator {
                    block: block.id,
                    term: format!("{term:?}"),
                });
            }
        }
    }
}

fn execute_function_inner_with_event_trace(
    program: &cfg_mir::Program,
    lambda_indices: &HashMap<usize, usize>,
    func_index: usize,
    state: &mut InterpreterState<'_>,
    trace: &mut InterpreterExecutionTrace,
    steps: &mut usize,
) -> Result<Vec<u64>, InterpreterError> {
    let func = program
        .funcs
        .get(func_index)
        .ok_or(InterpreterError::NoFunctions)?;
    let block_indices = build_block_index(func);
    let mut current = func.entry;

    loop {
        if *steps >= MAX_EXEC_STEPS {
            return Err(InterpreterError::StepLimitExceeded {
                limit: MAX_EXEC_STEPS,
            });
        }

        let block_idx = *block_indices
            .get(&current)
            .ok_or(InterpreterError::UnknownBlock { block: current })?;
        let block = &func.blocks[block_idx];

        for (inst_index, inst_id) in block.insts.iter().copied().enumerate() {
            *steps += 1;
            let step_index = *steps;
            let location = InterpreterTraceLocation {
                lambda: func.lambda_id,
                block: block.id,
                op: InterpreterTraceOp::Inst {
                    inst: inst_id,
                    inst_index,
                },
            };
            let inst = func
                .inst(inst_id)
                .ok_or(InterpreterError::UnknownInst { inst: inst_id })?;
            let before_cursor = state.cursor;
            let before_output = state.output.clone();
            let before_out_ptr = state.trace_out_ptr.clone();
            let before_trap = state.trap;

            match &inst.op {
                LinearOp::Const { dst, value } => {
                    let trace_value = TraceValue::U64(*value);
                    state.write_vreg(dst.index(), *value);
                    state.write_trace_vreg(dst.index(), trace_value.clone());
                    push_interpreter_event(
                        trace,
                        step_index,
                        location,
                        InterpreterEventKind::VregWrite {
                            vreg: *dst,
                            value: trace_value,
                        },
                    );
                }
                LinearOp::Copy { dst, src } => {
                    let value = state.read_vreg(src.index());
                    let trace_value = state.read_trace_vreg(src.index());
                    state.write_vreg(dst.index(), value);
                    state.write_trace_vreg(dst.index(), trace_value.clone());
                    push_interpreter_event(
                        trace,
                        step_index,
                        location,
                        InterpreterEventKind::VregWrite {
                            vreg: *dst,
                            value: trace_value,
                        },
                    );
                }
                LinearOp::BinOp { op, dst, lhs, rhs } => {
                    let lhs = state.read_vreg(lhs.index());
                    let rhs = state.read_vreg(rhs.index());
                    let value = exec_binop(*op, lhs, rhs);
                    let trace_value = TraceValue::U64(value);
                    state.write_vreg(dst.index(), value);
                    state.write_trace_vreg(dst.index(), trace_value.clone());
                    push_interpreter_event(
                        trace,
                        step_index,
                        location,
                        InterpreterEventKind::VregWrite {
                            vreg: *dst,
                            value: trace_value,
                        },
                    );
                }
                LinearOp::UnaryOp { op, dst, src } => {
                    let src = state.read_vreg(src.index());
                    let value = exec_unaryop(*op, src);
                    let trace_value = TraceValue::U64(value);
                    state.write_vreg(dst.index(), value);
                    state.write_trace_vreg(dst.index(), trace_value.clone());
                    push_interpreter_event(
                        trace,
                        step_index,
                        location,
                        InterpreterEventKind::VregWrite {
                            vreg: *dst,
                            value: trace_value,
                        },
                    );
                }
                LinearOp::BoundsCheck { count } => {
                    let count = *count as usize;
                    if state.cursor + count > state.input.len() {
                        state.trap(ErrorCode::UnexpectedEof);
                    }
                }
                LinearOp::ReadBytes { dst, count } => {
                    let count = *count as usize;
                    if state.cursor + count > state.input.len() {
                        state.trap(ErrorCode::UnexpectedEof);
                    } else {
                        let mut value = 0u64;
                        for i in 0..count {
                            value |= (state.input[state.cursor + i] as u64) << (i * 8);
                        }
                        state.cursor += count;
                        let trace_value = TraceValue::U64(value);
                        state.write_vreg(dst.index(), value);
                        state.write_trace_vreg(dst.index(), trace_value.clone());
                        push_interpreter_event(
                            trace,
                            step_index,
                            location,
                            InterpreterEventKind::VregWrite {
                                vreg: *dst,
                                value: trace_value,
                            },
                        );
                    }
                }
                LinearOp::PeekByte { dst } => {
                    if state.cursor >= state.input.len() {
                        state.trap(ErrorCode::UnexpectedEof);
                    } else {
                        let value = state.input[state.cursor] as u64;
                        let trace_value = TraceValue::U64(value);
                        state.write_vreg(dst.index(), value);
                        state.write_trace_vreg(dst.index(), trace_value.clone());
                        push_interpreter_event(
                            trace,
                            step_index,
                            location,
                            InterpreterEventKind::VregWrite {
                                vreg: *dst,
                                value: trace_value,
                            },
                        );
                    }
                }
                LinearOp::AdvanceCursor { count } => {
                    let count = *count as usize;
                    if state.cursor + count > state.input.len() {
                        state.trap(ErrorCode::UnexpectedEof);
                    } else {
                        state.cursor += count;
                    }
                }
                LinearOp::AdvanceCursorBy { src } => {
                    let count = state.read_vreg(src.index()) as usize;
                    if state.cursor + count > state.input.len() {
                        state.trap(ErrorCode::UnexpectedEof);
                    } else {
                        state.cursor += count;
                    }
                }
                LinearOp::SaveCursor { dst } => {
                    let value = state.cursor as u64;
                    let trace_value = TraceValue::U64(value);
                    state.write_vreg(dst.index(), value);
                    state.write_trace_vreg(dst.index(), trace_value.clone());
                    push_interpreter_event(
                        trace,
                        step_index,
                        location,
                        InterpreterEventKind::VregWrite {
                            vreg: *dst,
                            value: trace_value,
                        },
                    );
                }
                LinearOp::RestoreCursor { src } => {
                    let next = state.read_vreg(src.index()) as usize;
                    if next > state.input.len() {
                        state.trap(ErrorCode::UnexpectedEof);
                    } else {
                        state.cursor = next;
                    }
                }
                LinearOp::WriteToField { src, offset, width } => {
                    let value = state.read_vreg(src.index());
                    let width_bytes = width.bytes() as usize;
                    let base = *offset as usize;
                    state.ensure_output_range_for_out_ptr(base, width_bytes);
                    unsafe {
                        let dst = state.out_ptr.add(base);
                        for i in 0..width_bytes {
                            *dst.add(i) = ((value >> (i * 8)) & 0xff) as u8;
                        }
                    }
                }
                LinearOp::ErrorExit { code } => {
                    state.trap(*code);
                }
                LinearOp::ReadFromField { dst, offset, width } => {
                    let width_bytes = width.bytes() as usize;
                    let base = *offset as usize;
                    state.ensure_output_range_for_out_ptr(base, width_bytes);
                    let mut value = 0u64;
                    unsafe {
                        let src = state.out_ptr.add(base);
                        for i in 0..width_bytes {
                            value |= (*src.add(i) as u64) << (i * 8);
                        }
                    }
                    let trace_value = TraceValue::U64(value);
                    state.write_vreg(dst.index(), value);
                    state.write_trace_vreg(dst.index(), trace_value.clone());
                    push_interpreter_event(
                        trace,
                        step_index,
                        location,
                        InterpreterEventKind::VregWrite {
                            vreg: *dst,
                            value: trace_value,
                        },
                    );
                }
                LinearOp::SaveOutPtr { dst } => {
                    let raw = state.out_ptr as u64;
                    let trace_value = state.trace_out_ptr.clone();
                    state.write_vreg(dst.index(), raw);
                    state.write_trace_vreg(dst.index(), trace_value.clone());
                    push_interpreter_event(
                        trace,
                        step_index,
                        location,
                        InterpreterEventKind::VregWrite {
                            vreg: *dst,
                            value: trace_value,
                        },
                    );
                }
                LinearOp::SetOutPtr { src } => {
                    let before = state.trace_out_ptr.clone();
                    state.out_ptr = state.read_vreg(src.index()) as *mut u8;
                    state.trace_out_ptr = state.read_trace_vreg(src.index());
                    push_interpreter_event(
                        trace,
                        step_index,
                        location,
                        InterpreterEventKind::OutPtrSet {
                            before,
                            after: state.trace_out_ptr.clone(),
                        },
                    );
                }
                LinearOp::SlotAddr { dst, slot } => {
                    let addr = state.slot_addr_value(slot.index());
                    let trace_value = TraceValue::SlotAddr { slot: slot.index() };
                    state.write_vreg(dst.index(), addr);
                    state.write_trace_vreg(dst.index(), trace_value.clone());
                    push_interpreter_event(
                        trace,
                        step_index,
                        location,
                        InterpreterEventKind::VregWrite {
                            vreg: *dst,
                            value: trace_value,
                        },
                    );
                }
                LinearOp::WriteToSlot { slot, src } => {
                    let slot_index = slot.index();
                    state.ensure_slot(slot_index);
                    let value = state.read_vreg(src.index());
                    let trace_value = state.read_trace_vreg(src.index());
                    state.slots[slot_index] = value;
                    state.trace_slots[slot_index] = trace_value.clone();
                    push_interpreter_event(
                        trace,
                        step_index,
                        location,
                        InterpreterEventKind::SlotWrite {
                            slot: *slot,
                            value: trace_value,
                        },
                    );
                }
                LinearOp::ReadFromSlot { dst, slot } => {
                    let slot_index = slot.index();
                    state.ensure_slot(slot_index);
                    let value = state.slots[slot_index];
                    let trace_value = state.trace_slots[slot_index].clone();
                    state.write_vreg(dst.index(), value);
                    state.write_trace_vreg(dst.index(), trace_value.clone());
                    push_interpreter_event(
                        trace,
                        step_index,
                        location,
                        InterpreterEventKind::VregWrite {
                            vreg: *dst,
                            value: trace_value,
                        },
                    );
                }
                LinearOp::CallIntrinsic {
                    func,
                    args,
                    dst,
                    field_offset,
                } => {
                    let args_values: Vec<u64> =
                        args.iter().map(|v| state.read_vreg(v.index())).collect();
                    let out_ptr = if dst.is_none() {
                        let offset = *field_offset as usize;
                        state.ensure_output_range_for_out_ptr(offset, 64);
                        Some(unsafe { state.out_ptr.add(offset) })
                    } else {
                        None
                    };
                    let ret = run_call_intrinsic(state, func.0, &args_values, out_ptr);
                    if let Some(dst) = dst {
                        let trace_value = TraceValue::U64(ret);
                        state.write_vreg(dst.index(), ret);
                        state.write_trace_vreg(dst.index(), trace_value.clone());
                        push_interpreter_event(
                            trace,
                            step_index,
                            location,
                            InterpreterEventKind::VregWrite {
                                vreg: *dst,
                                value: trace_value,
                            },
                        );
                    }
                }
                LinearOp::CallPure { func, args, dst } => {
                    let args_values: Vec<u64> =
                        args.iter().map(|v| state.read_vreg(v.index())).collect();
                    let ret = run_call_pure(func.0, &args_values);
                    let trace_value = TraceValue::U64(ret);
                    state.write_vreg(dst.index(), ret);
                    state.write_trace_vreg(dst.index(), trace_value.clone());
                    push_interpreter_event(
                        trace,
                        step_index,
                        location,
                        InterpreterEventKind::VregWrite {
                            vreg: *dst,
                            value: trace_value,
                        },
                    );
                }
                LinearOp::CallLambda {
                    target,
                    args,
                    results,
                } => {
                    let target_index = if let Some(&target_index) =
                        lambda_indices.get(&target.index())
                    {
                        target_index
                    } else {
                        let mut known = lambda_indices.keys().copied().collect::<Vec<_>>();
                        known.sort_unstable();
                        return Err(InterpreterError::UnsupportedOp {
                            block: block.id,
                            op: format!(
                                "CallLambda @{} has no target function; known lambdas: {known:?}",
                                target.index()
                            ),
                        });
                    };
                    let target_func = &program.funcs[target_index];
                    if target_func.data_args.len() != args.len() {
                        return Err(InterpreterError::UnsupportedOp {
                            block: block.id,
                            op: format!(
                                "CallLambda @{} arg arity mismatch: expected {}, got {}",
                                target.index(),
                                target_func.data_args.len(),
                                args.len()
                            ),
                        });
                    }

                    for (param, arg) in target_func.data_args.iter().zip(args.iter()) {
                        let value = state.read_vreg(arg.index());
                        let trace_value = state.read_trace_vreg(arg.index());
                        state.write_vreg(param.index(), value);
                        state.write_trace_vreg(param.index(), trace_value.clone());
                        push_interpreter_event(
                            trace,
                            step_index,
                            location,
                            InterpreterEventKind::VregWrite {
                                vreg: *param,
                                value: trace_value,
                            },
                        );
                    }
                    push_interpreter_event(
                        trace,
                        step_index,
                        location,
                        InterpreterEventKind::CallEnter { target: *target },
                    );
                    push_interpreter_event(
                        trace,
                        step_index,
                        InterpreterTraceLocation {
                            lambda: target_func.lambda_id,
                            block: target_func.entry,
                            op: InterpreterTraceOp::Entry,
                        },
                        InterpreterEventKind::BlockEnter {
                            via_edge: None,
                            target: target_func.entry,
                        },
                    );
                    let caller_out_ptr = state.out_ptr;
                    let caller_trace_out_ptr = state.trace_out_ptr.clone();
                    let callee_results = execute_function_inner_with_event_trace(
                        program,
                        lambda_indices,
                        target_index,
                        state,
                        trace,
                        steps,
                    )?;
                    state.out_ptr = caller_out_ptr;
                    state.trace_out_ptr = caller_trace_out_ptr;
                    let call_result_values = results
                        .iter()
                        .zip(callee_results.iter())
                        .map(|(_, value)| TraceValue::U64(*value))
                        .collect::<Vec<_>>();
                    push_interpreter_event(
                        trace,
                        step_index,
                        location,
                        InterpreterEventKind::CallReturn {
                            target: *target,
                            results: call_result_values.clone(),
                        },
                    );
                    if state.trap.is_some() {
                        break;
                    }
                    if callee_results.len() != results.len() {
                        return Err(InterpreterError::UnsupportedOp {
                            block: block.id,
                            op: format!(
                                "CallLambda @{} result arity mismatch: expected {}, got {}",
                                target.index(),
                                results.len(),
                                callee_results.len()
                            ),
                        });
                    }
                    for ((dst, value), trace_value) in results
                        .iter()
                        .zip(callee_results.iter())
                        .zip(call_result_values.into_iter())
                    {
                        state.write_vreg(dst.index(), *value);
                        state.write_trace_vreg(dst.index(), trace_value.clone());
                        push_interpreter_event(
                            trace,
                            step_index,
                            location,
                            InterpreterEventKind::VregWrite {
                                vreg: *dst,
                                value: trace_value,
                            },
                        );
                    }
                }
                op => {
                    return Err(InterpreterError::UnsupportedOp {
                        block: block.id,
                        op: format!("{op:?}"),
                    });
                }
            }

            if before_cursor != state.cursor {
                push_interpreter_event(
                    trace,
                    step_index,
                    location,
                    InterpreterEventKind::CursorSet {
                        before: before_cursor,
                        after: state.cursor,
                    },
                );
            }

            for (offset, bytes) in diff_output_regions(&before_output, &state.output) {
                push_interpreter_event(
                    trace,
                    step_index,
                    location,
                    InterpreterEventKind::OutputWrite {
                        base: before_out_ptr.clone(),
                        offset,
                        bytes,
                    },
                );
            }

            if before_trap != state.trap
                && let Some(trap) = state.trap
            {
                push_interpreter_event(
                    trace,
                    step_index,
                    location,
                    InterpreterEventKind::Trap { trap },
                );
            }

            if state.trap.is_some() {
                return Ok(Vec::new());
            }
        }

        *steps += 1;
        let step_index = *steps;
        let location = InterpreterTraceLocation {
            lambda: func.lambda_id,
            block: block.id,
            op: InterpreterTraceOp::Term { term: block.term },
        };
        let term = func
            .term(block.term)
            .ok_or(InterpreterError::UnknownTerm { term: block.term })?;
        match term {
            cfg_mir::Terminator::Return => {
                let results = func
                    .data_results
                    .iter()
                    .map(|vreg| state.read_trace_vreg(vreg.index()))
                    .collect::<Vec<_>>();
                push_interpreter_event(
                    trace,
                    step_index,
                    location,
                    InterpreterEventKind::Return {
                        results: results.clone(),
                    },
                );
                let data_results = func
                    .data_results
                    .iter()
                    .map(|vreg| state.read_vreg(vreg.index()))
                    .collect::<Vec<_>>();
                return Ok(data_results);
            }
            cfg_mir::Terminator::ErrorExit { code } => {
                state.trap(*code);
                if let Some(trap) = state.trap {
                    push_interpreter_event(
                        trace,
                        step_index,
                        location,
                        InterpreterEventKind::Trap { trap },
                    );
                }
                return Ok(Vec::new());
            }
            cfg_mir::Terminator::Branch { edge } => {
                let next = apply_edge_with_event_trace(
                    func,
                    &block_indices,
                    state,
                    *edge,
                    trace,
                    step_index,
                    location,
                )?;
                push_interpreter_event(
                    trace,
                    step_index,
                    location,
                    InterpreterEventKind::TerminatorDecision {
                        detail: format!("branch e{} -> b{}", edge.0, next.0),
                    },
                );
                push_interpreter_event(
                    trace,
                    step_index,
                    location,
                    InterpreterEventKind::BlockEnter {
                        via_edge: Some(*edge),
                        target: next,
                    },
                );
                current = next;
            }
            cfg_mir::Terminator::BranchIf {
                cond,
                taken,
                fallthrough,
            } => {
                let cond_value = state.read_vreg(cond.index());
                let edge = if cond_value != 0 {
                    *taken
                } else {
                    *fallthrough
                };
                let next = apply_edge_with_event_trace(
                    func,
                    &block_indices,
                    state,
                    edge,
                    trace,
                    step_index,
                    location,
                )?;
                push_interpreter_event(
                    trace,
                    step_index,
                    location,
                    InterpreterEventKind::TerminatorDecision {
                        detail: format!(
                            "branch_if v{}={:#x} -> e{} -> b{}",
                            cond.index(),
                            cond_value,
                            edge.0,
                            next.0
                        ),
                    },
                );
                push_interpreter_event(
                    trace,
                    step_index,
                    location,
                    InterpreterEventKind::BlockEnter {
                        via_edge: Some(edge),
                        target: next,
                    },
                );
                current = next;
            }
            cfg_mir::Terminator::BranchIfZero {
                cond,
                taken,
                fallthrough,
            } => {
                let cond_value = state.read_vreg(cond.index());
                let edge = if cond_value == 0 {
                    *taken
                } else {
                    *fallthrough
                };
                let next = apply_edge_with_event_trace(
                    func,
                    &block_indices,
                    state,
                    edge,
                    trace,
                    step_index,
                    location,
                )?;
                push_interpreter_event(
                    trace,
                    step_index,
                    location,
                    InterpreterEventKind::TerminatorDecision {
                        detail: format!(
                            "branch_if_zero v{}={:#x} -> e{} -> b{}",
                            cond.index(),
                            cond_value,
                            edge.0,
                            next.0
                        ),
                    },
                );
                push_interpreter_event(
                    trace,
                    step_index,
                    location,
                    InterpreterEventKind::BlockEnter {
                        via_edge: Some(edge),
                        target: next,
                    },
                );
                current = next;
            }
            term => {
                return Err(InterpreterError::UnsupportedTerminator {
                    block: block.id,
                    term: format!("{term:?}"),
                });
            }
        }
    }
}

const SLOT_ADDR_STRIDE: usize = 16;

#[repr(C)]
struct RuntimeErrorSlot {
    code: u32,
    offset: u32,
}

#[repr(C)]
struct RuntimeDeserContext {
    input_ptr: *const u8,
    input_end: *const u8,
    error: RuntimeErrorSlot,
    key_scratch_ptr: *mut u8,
    key_scratch_cap: usize,
    trusted_utf8: bool,
}

impl RuntimeDeserContext {
    fn new(input: &[u8]) -> Self {
        let ptr = input.as_ptr();
        Self {
            input_ptr: ptr,
            input_end: unsafe { ptr.add(input.len()) },
            error: RuntimeErrorSlot { code: 0, offset: 0 },
            key_scratch_ptr: core::ptr::null_mut(),
            key_scratch_cap: 0,
            trusted_utf8: false,
        }
    }
}

impl Drop for RuntimeDeserContext {
    fn drop(&mut self) {
        if self.key_scratch_cap > 0 {
            unsafe {
                let layout = std::alloc::Layout::from_size_align_unchecked(self.key_scratch_cap, 1);
                std::alloc::dealloc(self.key_scratch_ptr, layout);
            }
        }
    }
}

fn error_code_from_u32(code: u32) -> ErrorCode {
    match code {
        0 => ErrorCode::Ok,
        1 => ErrorCode::UnexpectedEof,
        2 => ErrorCode::InvalidVarint,
        3 => ErrorCode::InvalidUtf8,
        4 => ErrorCode::UnsupportedShape,
        5 => ErrorCode::ExpectedObjectStart,
        6 => ErrorCode::ExpectedColon,
        7 => ErrorCode::ExpectedStringKey,
        8 => ErrorCode::UnterminatedString,
        9 => ErrorCode::InvalidJsonNumber,
        10 => ErrorCode::MissingRequiredField,
        11 => ErrorCode::UnexpectedCharacter,
        12 => ErrorCode::NumberOutOfRange,
        13 => ErrorCode::InvalidBool,
        14 => ErrorCode::UnknownVariant,
        15 => ErrorCode::ExpectedTagKey,
        16 => ErrorCode::AmbiguousVariant,
        17 => ErrorCode::AllocError,
        18 => ErrorCode::InvalidEscapeSequence,
        19 => ErrorCode::UnknownField,
        _ => ErrorCode::UnexpectedCharacter,
    }
}

fn run_call_intrinsic(
    state: &mut InterpreterState<'_>,
    func: usize,
    args: &[u64],
    out_ptr: Option<*mut u8>,
) -> u64 {
    state.ctx.input_ptr = unsafe { state.input_base.add(state.cursor) };
    state.ctx.input_end = unsafe { state.input_base.add(state.input.len()) };
    state.ctx.error.code = 0;
    state.ctx.error.offset = 0;

    let ctx_ptr = &mut state.ctx as *mut RuntimeDeserContext;

    let ret = unsafe {
        match (out_ptr, args.len()) {
            (None, 0) => {
                let f: unsafe extern "C" fn(*mut RuntimeDeserContext) -> u64 =
                    core::mem::transmute(func);
                f(ctx_ptr)
            }
            (None, 1) => {
                let f: unsafe extern "C" fn(*mut RuntimeDeserContext, u64) -> u64 =
                    core::mem::transmute(func);
                f(ctx_ptr, args[0])
            }
            (None, 2) => {
                let f: unsafe extern "C" fn(*mut RuntimeDeserContext, u64, u64) -> u64 =
                    core::mem::transmute(func);
                f(ctx_ptr, args[0], args[1])
            }
            (None, 3) => {
                let f: unsafe extern "C" fn(*mut RuntimeDeserContext, u64, u64, u64) -> u64 =
                    core::mem::transmute(func);
                f(ctx_ptr, args[0], args[1], args[2])
            }
            (None, 4) => {
                let f: unsafe extern "C" fn(*mut RuntimeDeserContext, u64, u64, u64, u64) -> u64 =
                    core::mem::transmute(func);
                f(ctx_ptr, args[0], args[1], args[2], args[3])
            }
            (None, 5) => {
                let f: unsafe extern "C" fn(
                    *mut RuntimeDeserContext,
                    u64,
                    u64,
                    u64,
                    u64,
                    u64,
                ) -> u64 = core::mem::transmute(func);
                f(ctx_ptr, args[0], args[1], args[2], args[3], args[4])
            }
            (Some(out), 0) => {
                let f: unsafe extern "C" fn(*mut RuntimeDeserContext, *mut u8) =
                    core::mem::transmute(func);
                f(ctx_ptr, out);
                0
            }
            (Some(out), 1) => {
                let f: unsafe extern "C" fn(*mut RuntimeDeserContext, u64, *mut u8) =
                    core::mem::transmute(func);
                f(ctx_ptr, args[0], out);
                0
            }
            (Some(out), 2) => {
                let f: unsafe extern "C" fn(*mut RuntimeDeserContext, u64, u64, *mut u8) =
                    core::mem::transmute(func);
                f(ctx_ptr, args[0], args[1], out);
                0
            }
            (Some(out), 3) => {
                let f: unsafe extern "C" fn(*mut RuntimeDeserContext, u64, u64, u64, *mut u8) =
                    core::mem::transmute(func);
                f(ctx_ptr, args[0], args[1], args[2], out);
                0
            }
            _ => 0,
        }
    };

    state.cursor = unsafe { state.ctx.input_ptr.offset_from(state.input_base) as usize };
    if state.ctx.error.code != 0 {
        let code = error_code_from_u32(state.ctx.error.code);
        state.trap = Some(InterpreterTrap {
            code,
            offset: state.ctx.error.offset,
        });
    }
    ret
}

fn run_call_pure(func: usize, args: &[u64]) -> u64 {
    unsafe {
        match args.len() {
            0 => {
                let f: unsafe extern "C" fn() -> u64 = core::mem::transmute(func);
                f()
            }
            1 => {
                let f: unsafe extern "C" fn(u64) -> u64 = core::mem::transmute(func);
                f(args[0])
            }
            2 => {
                let f: unsafe extern "C" fn(u64, u64) -> u64 = core::mem::transmute(func);
                f(args[0], args[1])
            }
            3 => {
                let f: unsafe extern "C" fn(u64, u64, u64) -> u64 = core::mem::transmute(func);
                f(args[0], args[1], args[2])
            }
            4 => {
                let f: unsafe extern "C" fn(u64, u64, u64, u64) -> u64 = core::mem::transmute(func);
                f(args[0], args[1], args[2], args[3])
            }
            5 => {
                let f: unsafe extern "C" fn(u64, u64, u64, u64, u64) -> u64 =
                    core::mem::transmute(func);
                f(args[0], args[1], args[2], args[3], args[4])
            }
            6 => {
                let f: unsafe extern "C" fn(u64, u64, u64, u64, u64, u64) -> u64 =
                    core::mem::transmute(func);
                f(args[0], args[1], args[2], args[3], args[4], args[5])
            }
            _ => 0,
        }
    }
}

fn build_block_index(func: &cfg_mir::Function) -> HashMap<cfg_mir::BlockId, usize> {
    let mut out = HashMap::with_capacity(func.blocks.len());
    for (idx, block) in func.blocks.iter().enumerate() {
        out.insert(block.id, idx);
    }
    out
}

fn infer_program_output_size(program: &cfg_mir::Program) -> usize {
    program
        .funcs
        .iter()
        .map(infer_output_size)
        .max()
        .unwrap_or(0)
}

fn infer_output_size(func: &cfg_mir::Function) -> usize {
    func.blocks
        .iter()
        .flat_map(|block| block.insts.iter())
        .filter_map(|inst_id| {
            let inst = func.inst(*inst_id)?;
            match &inst.op {
                LinearOp::WriteToField { offset, width, .. } => {
                    Some(*offset as usize + width.bytes() as usize)
                }
                _ => None,
            }
        })
        .max()
        .unwrap_or(0)
}

fn exec_binop(op: BinOpKind, lhs: u64, rhs: u64) -> u64 {
    match op {
        BinOpKind::Add => lhs.wrapping_add(rhs),
        BinOpKind::Sub => lhs.wrapping_sub(rhs),
        BinOpKind::And => lhs & rhs,
        BinOpKind::Or => lhs | rhs,
        BinOpKind::Xor => lhs ^ rhs,
        BinOpKind::Shl => {
            if rhs >= 64 {
                0
            } else {
                lhs.wrapping_shl(rhs as u32)
            }
        }
        BinOpKind::Shr => {
            if rhs >= 64 {
                0
            } else {
                lhs.wrapping_shr(rhs as u32)
            }
        }
        BinOpKind::CmpNe => u64::from(lhs != rhs),
    }
}

fn exec_unaryop(op: UnaryOpKind, src: u64) -> u64 {
    match op {
        UnaryOpKind::ZigzagDecode { wide } => {
            if wide {
                ((src >> 1) as i64 ^ -((src & 1) as i64)) as u64
            } else {
                let s = src as u32;
                let v = ((s >> 1) as i32) ^ -((s & 1) as i32);
                v as i64 as u64
            }
        }
        UnaryOpKind::SignExtend { from_width } => match from_width {
            kajit_ir::Width::W1 => (src as u8 as i8 as i64) as u64,
            kajit_ir::Width::W2 => (src as u16 as i16 as i64) as u64,
            kajit_ir::Width::W4 => (src as u32 as i32 as i64) as u64,
            kajit_ir::Width::W8 => src,
        },
    }
}

fn apply_edge(
    func: &cfg_mir::Function,
    block_indices: &HashMap<cfg_mir::BlockId, usize>,
    state: &mut InterpreterState<'_>,
    edge_id: cfg_mir::EdgeId,
) -> Result<cfg_mir::BlockId, InterpreterError> {
    let edge = func
        .edge(edge_id)
        .ok_or(InterpreterError::UnknownEdge { edge: edge_id })?;
    let from = edge.from;
    let to = edge.to;
    let to_idx = *block_indices
        .get(&to)
        .ok_or(InterpreterError::UnknownBlock { block: to })?;
    let to_block = &func.blocks[to_idx];

    if edge.args.len() != to_block.params.len() {
        return Err(InterpreterError::EdgeArgArityMismatch {
            from,
            to,
            expected: to_block.params.len(),
            got: edge.args.len(),
        });
    }

    for arg in &edge.args {
        let value = state.read_vreg(arg.source.index());
        state.write_vreg(arg.target.index(), value);
    }

    Ok(to)
}

fn apply_edge_with_event_trace(
    func: &cfg_mir::Function,
    block_indices: &HashMap<cfg_mir::BlockId, usize>,
    state: &mut InterpreterState<'_>,
    edge_id: cfg_mir::EdgeId,
    trace: &mut InterpreterExecutionTrace,
    step_index: usize,
    location: InterpreterTraceLocation,
) -> Result<cfg_mir::BlockId, InterpreterError> {
    let edge = func
        .edge(edge_id)
        .ok_or(InterpreterError::UnknownEdge { edge: edge_id })?;
    let from = edge.from;
    let to = edge.to;
    let to_idx = *block_indices
        .get(&to)
        .ok_or(InterpreterError::UnknownBlock { block: to })?;
    let to_block = &func.blocks[to_idx];

    if edge.args.len() != to_block.params.len() {
        return Err(InterpreterError::EdgeArgArityMismatch {
            from,
            to,
            expected: to_block.params.len(),
            got: edge.args.len(),
        });
    }

    for arg in &edge.args {
        let value = state.read_vreg(arg.source.index());
        let trace_value = state.read_trace_vreg(arg.source.index());
        state.write_vreg(arg.target.index(), value);
        state.write_trace_vreg(arg.target.index(), trace_value.clone());
        push_interpreter_event(
            trace,
            step_index,
            location,
            InterpreterEventKind::VregWrite {
                vreg: arg.target,
                value: trace_value,
            },
        );
    }

    Ok(to)
}

#[cfg(test)]
mod tests {
    use super::*;
    use kajit_ir::{LambdaId, VReg, Width};

    fn v(index: u32) -> VReg {
        VReg::new(index)
    }

    fn test_inst(id: u32, op: LinearOp) -> cfg_mir::Inst {
        cfg_mir::Inst {
            id: cfg_mir::InstId(id),
            op,
            operands: Vec::new(),
            clobbers: cfg_mir::Clobbers::default(),
        }
    }

    fn linear_program() -> cfg_mir::Program {
        let b0 = cfg_mir::Block {
            id: cfg_mir::BlockId(0),
            params: Vec::new(),
            insts: vec![cfg_mir::InstId(0), cfg_mir::InstId(1), cfg_mir::InstId(2)],
            term: cfg_mir::TermId(0),
            preds: Vec::new(),
            succs: Vec::new(),
        };
        cfg_mir::Program {
            funcs: vec![cfg_mir::Function {
                id: cfg_mir::FunctionId(0),
                lambda_id: LambdaId::new(0),
                entry: cfg_mir::BlockId(0),
                data_args: Vec::new(),
                data_results: vec![v(0)],
                blocks: vec![b0],
                edges: Vec::new(),
                insts: vec![
                    test_inst(0, LinearOp::BoundsCheck { count: 1 }),
                    test_inst(
                        1,
                        LinearOp::ReadBytes {
                            dst: v(0),
                            count: 1,
                        },
                    ),
                    test_inst(
                        2,
                        LinearOp::WriteToField {
                            src: v(0),
                            offset: 0,
                            width: Width::W1,
                        },
                    ),
                ],
                terms: vec![cfg_mir::Terminator::Return],
            }],
            vreg_count: 1,
            slot_count: 0,
            debug: Default::default(),
        }
    }

    fn branching_program() -> cfg_mir::Program {
        let b0 = cfg_mir::Block {
            id: cfg_mir::BlockId(0),
            params: Vec::new(),
            insts: vec![cfg_mir::InstId(0)],
            term: cfg_mir::TermId(0),
            preds: Vec::new(),
            succs: vec![cfg_mir::EdgeId(0), cfg_mir::EdgeId(1)],
        };
        let b1 = cfg_mir::Block {
            id: cfg_mir::BlockId(1),
            params: vec![v(1)],
            insts: vec![cfg_mir::InstId(1)],
            term: cfg_mir::TermId(1),
            preds: vec![cfg_mir::EdgeId(0)],
            succs: Vec::new(),
        };
        let b2 = cfg_mir::Block {
            id: cfg_mir::BlockId(2),
            params: vec![v(1)],
            insts: vec![cfg_mir::InstId(2)],
            term: cfg_mir::TermId(2),
            preds: vec![cfg_mir::EdgeId(1)],
            succs: Vec::new(),
        };
        cfg_mir::Program {
            funcs: vec![cfg_mir::Function {
                id: cfg_mir::FunctionId(0),
                lambda_id: LambdaId::new(0),
                entry: cfg_mir::BlockId(0),
                data_args: Vec::new(),
                data_results: vec![v(1)],
                blocks: vec![b0, b1, b2],
                edges: vec![
                    cfg_mir::Edge {
                        id: cfg_mir::EdgeId(0),
                        from: cfg_mir::BlockId(0),
                        to: cfg_mir::BlockId(1),
                        args: vec![cfg_mir::EdgeArg {
                            target: v(1),
                            source: v(0),
                        }],
                    },
                    cfg_mir::Edge {
                        id: cfg_mir::EdgeId(1),
                        from: cfg_mir::BlockId(0),
                        to: cfg_mir::BlockId(2),
                        args: vec![cfg_mir::EdgeArg {
                            target: v(1),
                            source: v(0),
                        }],
                    },
                ],
                insts: vec![
                    test_inst(
                        0,
                        LinearOp::Const {
                            dst: v(0),
                            value: 0,
                        },
                    ),
                    test_inst(
                        1,
                        LinearOp::Const {
                            dst: v(1),
                            value: 0xaa,
                        },
                    ),
                    test_inst(
                        2,
                        LinearOp::Const {
                            dst: v(1),
                            value: 0xbb,
                        },
                    ),
                ],
                terms: vec![
                    cfg_mir::Terminator::BranchIfZero {
                        cond: v(0),
                        taken: cfg_mir::EdgeId(0),
                        fallthrough: cfg_mir::EdgeId(1),
                    },
                    cfg_mir::Terminator::Return,
                    cfg_mir::Terminator::Return,
                ],
            }],
            vreg_count: 2,
            slot_count: 0,
            debug: Default::default(),
        }
    }

    #[test]
    fn event_trace_captures_cursor_vreg_output_and_return() {
        let program = linear_program();
        let (outcome, trace) =
            execute_program_with_event_trace(&program, &[0x2a]).expect("trace should execute");

        assert_eq!(outcome.cursor, 1);
        assert_eq!(outcome.output[0], 0x2a);

        let text = trace.render_text();
        assert_eq!(
            text,
            "\
#0000 step=0000 @0 b0 entry :: enter b0\n\
#0001 step=0002 @0 b0 inst i1#1 :: write v0 = 0x2a\n\
#0002 step=0002 @0 b0 inst i1#1 :: cursor 0 -> 1\n\
#0003 step=0003 @0 b0 inst i2#2 :: write output+0+0 [2a]\n\
#0004 step=0004 @0 b0 term t0 :: return [0x2a]"
        );
    }

    #[test]
    fn event_trace_supports_reverse_queries() {
        let program = branching_program();
        let (_, trace) =
            execute_program_with_event_trace(&program, &[]).expect("trace should execute");

        let entries = trace.entries_to_block(LambdaId::new(0), cfg_mir::BlockId(1));
        assert_eq!(entries.len(), 1);

        let last = trace
            .last_write_to_vreg(v(1))
            .expect("v1 should be written on the taken branch");
        assert_eq!(last.location.block, cfg_mir::BlockId(1));
        assert!(matches!(
            &last.kind,
            InterpreterEventKind::VregWrite {
                value: TraceValue::U64(0xaa),
                ..
            }
        ));
    }
}
