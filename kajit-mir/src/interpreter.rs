use std::collections::HashMap;

use kajit_ir::ErrorCode;
use kajit_lir::{BinOpKind, LinearOp, UnaryOpKind};

use crate::{BlockId, RaFunction, RaProgram, RaTerminator};

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
pub struct InterpreterTraceEntry {
    pub step_index: usize,
    pub block: BlockId,
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
    UnknownBlock {
        block: BlockId,
    },
    MissingEdge {
        from: BlockId,
        to: BlockId,
    },
    EdgeArgArityMismatch {
        from: BlockId,
        to: BlockId,
        expected: usize,
        got: usize,
    },
    UnsupportedOp {
        block: BlockId,
        op: String,
    },
    UnsupportedTerminator {
        block: BlockId,
        term: String,
    },
    StepLimitExceeded {
        limit: usize,
    },
}

impl std::fmt::Display for InterpreterError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NoFunctions => write!(f, "RA-MIR program has no functions"),
            Self::UnknownBlock { block } => write!(f, "unknown block b{}", block.0),
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
                    "unsupported RA-MIR op in block b{} for interpreter MVP: {}",
                    block.0, op
                )
            }
            Self::UnsupportedTerminator { block, term } => write!(
                f,
                "unsupported RA-MIR terminator in block b{} for interpreter MVP: {}",
                block.0, term
            ),
            Self::StepLimitExceeded { limit } => {
                write!(f, "RA-MIR interpreter exceeded step limit ({limit})")
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
    vregs: Vec<u64>,
    slots: Vec<u64>,
    slot_mem: Vec<u8>,
    ctx: RuntimeDeserContext,
    trap: Option<InterpreterTrap>,
}

impl<'a> InterpreterState<'a> {
    fn new(
        input: &'a [u8],
        vreg_count: usize,
        slot_count: usize,
        output_size: usize,
    ) -> Self {
        let input_base = input.as_ptr();
        let slot_mem = vec![0u8; slot_count.saturating_mul(SLOT_ADDR_STRIDE)];
        Self {
            input,
            input_base,
            cursor: 0,
            output: vec![0u8; output_size],
            vregs: vec![0u64; vreg_count],
            slots: vec![0u64; slot_count],
            slot_mem,
            ctx: RuntimeDeserContext::new(input),
            trap: None,
        }
    }

    fn read_vreg(&self, idx: usize) -> u64 {
        self.vregs.get(idx).copied().unwrap_or(0)
    }

    fn write_vreg(&mut self, idx: usize, value: u64) {
        if idx >= self.vregs.len() {
            self.vregs.resize(idx + 1, 0);
        }
        self.vregs[idx] = value;
    }

    fn ensure_output_len(&mut self, len: usize) {
        if self.output.len() < len {
            self.output.resize(len, 0);
        }
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

/// Execute the first function in an RA-MIR program.
pub fn execute_program(
    program: &RaProgram,
    input: &[u8],
) -> Result<InterpreterOutcome, InterpreterError> {
    let func = program.funcs.first().ok_or(InterpreterError::NoFunctions)?;
    execute_function(func, program.vreg_count as usize, program.slot_count as usize, input)
}

/// Execute a single RA-MIR function.
pub fn execute_function(
    func: &RaFunction,
    vreg_count: usize,
    slot_count: usize,
    input: &[u8],
) -> Result<InterpreterOutcome, InterpreterError> {
    execute_function_with_trace(func, vreg_count, slot_count, input).map(|(outcome, _)| outcome)
}

pub fn execute_function_with_trace(
    func: &RaFunction,
    vreg_count: usize,
    slot_count: usize,
    input: &[u8],
) -> Result<(InterpreterOutcome, Vec<InterpreterTraceEntry>), InterpreterError> {
    let block_indices = build_block_index(func);
    let mut state = InterpreterState::new(input, vreg_count, slot_count, infer_output_size(func));
    let mut current = func.entry;
    let mut steps = 0usize;
    let mut trace = Vec::new();

    loop {
        if steps >= MAX_EXEC_STEPS {
            return Err(InterpreterError::StepLimitExceeded {
                limit: MAX_EXEC_STEPS,
            });
        }
        steps += 1;

        let block_idx = *block_indices
            .get(&current)
            .ok_or(InterpreterError::UnknownBlock { block: current })?;
        let block = &func.blocks[block_idx];

        for (inst_index, inst) in block.insts.iter().enumerate() {
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
                    state.ensure_output_len(base + width_bytes);
                    for i in 0..width_bytes {
                        state.output[base + i] = ((value >> (i * 8)) & 0xff) as u8;
                    }
                }
                LinearOp::ErrorExit { code } => {
                    state.trap(*code);
                    break;
                }
                LinearOp::ReadFromField { dst, offset, width } => {
                    let width_bytes = width.bytes() as usize;
                    let base = *offset as usize;
                    state.ensure_output_len(base + width_bytes);
                    let mut value = 0u64;
                    for i in 0..width_bytes {
                        value |= (state.output[base + i] as u64) << (i * 8);
                    }
                    state.write_vreg(dst.index(), value);
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
                    let args_values: Vec<u64> = args.iter().map(|v| state.read_vreg(v.index())).collect();
                    let out_ptr = if dst.is_none() {
                        let offset = *field_offset as usize;
                        state.ensure_output_len(offset + 64);
                        Some(unsafe { state.output.as_mut_ptr().add(offset) })
                    } else {
                        None
                    };
                    let ret = run_call_intrinsic(&mut state, func.0, &args_values, out_ptr);
                    if state.trap.is_some() {
                        break;
                    }
                    if let Some(dst) = dst {
                        state.write_vreg(dst.index(), ret);
                    }
                }
                LinearOp::CallPure { func, args, dst } => {
                    let args_values: Vec<u64> = args.iter().map(|v| state.read_vreg(v.index())).collect();
                    let ret = run_call_pure(func.0, &args_values);
                    state.write_vreg(dst.index(), ret);
                }
                op => {
                    return Err(InterpreterError::UnsupportedOp {
                        block: block.id,
                        op: format!("{op:?}"),
                    });
                }
            }

            trace.push(InterpreterTraceEntry {
                step_index: steps,
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
            let outcome = InterpreterOutcome {
                vregs: state.vregs.clone(),
                output: state.output.clone(),
                cursor: state.cursor,
                trap: state.trap,
            };
            return Ok((outcome, trace));
        }

        match &block.term {
            RaTerminator::Return => {
                trace.push(InterpreterTraceEntry {
                    step_index: steps,
                    block: block.id,
                    next_inst_index: block.insts.len(),
                    at_terminator: true,
                    cursor: state.cursor,
                    vregs: state.vregs.clone(),
                    output: state.output.clone(),
                    trap: None,
                    returned: true,
                });
                let outcome = InterpreterOutcome {
                    vregs: state.vregs,
                    output: state.output,
                    cursor: state.cursor,
                    trap: None,
                };
                return Ok((outcome, trace));
            }
            RaTerminator::ErrorExit { code } => {
                state.trap(*code);
                trace.push(InterpreterTraceEntry {
                    step_index: steps,
                    block: block.id,
                    next_inst_index: block.insts.len(),
                    at_terminator: true,
                    cursor: state.cursor,
                    vregs: state.vregs.clone(),
                    output: state.output.clone(),
                    trap: state.trap,
                    returned: false,
                });
                let outcome = InterpreterOutcome {
                    vregs: state.vregs,
                    output: state.output,
                    cursor: state.cursor,
                    trap: state.trap,
                };
                return Ok((outcome, trace));
            }
            RaTerminator::Branch { target } => {
                apply_edge_args(func, &block_indices, &mut state, block.id, *target)?;
                let target_block_idx = *block_indices
                    .get(target)
                    .ok_or(InterpreterError::UnknownBlock { block: *target })?;
                let at_terminator = 0 >= func.blocks[target_block_idx].insts.len();
                trace.push(InterpreterTraceEntry {
                    step_index: steps,
                    block: *target,
                    next_inst_index: 0,
                    at_terminator,
                    cursor: state.cursor,
                    vregs: state.vregs.clone(),
                    output: state.output.clone(),
                    trap: state.trap,
                    returned: false,
                });
                current = *target;
            }
            RaTerminator::BranchIf {
                cond,
                target,
                fallthrough,
            } => {
                let branch = state.read_vreg(cond.index()) != 0;
                let next = if branch { *target } else { *fallthrough };
                apply_edge_args(func, &block_indices, &mut state, block.id, next)?;
                let target_block_idx = *block_indices
                    .get(&next)
                    .ok_or(InterpreterError::UnknownBlock { block: next })?;
                let at_terminator = 0 >= func.blocks[target_block_idx].insts.len();
                trace.push(InterpreterTraceEntry {
                    step_index: steps,
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
            RaTerminator::BranchIfZero {
                cond,
                target,
                fallthrough,
            } => {
                let branch = state.read_vreg(cond.index()) == 0;
                let next = if branch { *target } else { *fallthrough };
                apply_edge_args(func, &block_indices, &mut state, block.id, next)?;
                let target_block_idx = *block_indices
                    .get(&next)
                    .ok_or(InterpreterError::UnknownBlock { block: next })?;
                let at_terminator = 0 >= func.blocks[target_block_idx].insts.len();
                trace.push(InterpreterTraceEntry {
                    step_index: steps,
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
                let f: unsafe extern "C" fn(*mut RuntimeDeserContext, u64, u64, u64, u64, u64) -> u64 =
                    core::mem::transmute(func);
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

fn build_block_index(func: &RaFunction) -> HashMap<BlockId, usize> {
    let mut out = HashMap::with_capacity(func.blocks.len());
    for (idx, block) in func.blocks.iter().enumerate() {
        out.insert(block.id, idx);
    }
    out
}

fn infer_output_size(func: &RaFunction) -> usize {
    func.blocks
        .iter()
        .flat_map(|block| block.insts.iter())
        .filter_map(|inst| match &inst.op {
            LinearOp::WriteToField { offset, width, .. } => {
                Some(*offset as usize + width.bytes() as usize)
            }
            _ => None,
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

fn apply_edge_args(
    func: &RaFunction,
    block_indices: &HashMap<BlockId, usize>,
    state: &mut InterpreterState<'_>,
    from: BlockId,
    to: BlockId,
) -> Result<(), InterpreterError> {
    let from_idx = *block_indices
        .get(&from)
        .ok_or(InterpreterError::UnknownBlock { block: from })?;
    let to_idx = *block_indices
        .get(&to)
        .ok_or(InterpreterError::UnknownBlock { block: to })?;

    let from_block = &func.blocks[from_idx];
    let to_block = &func.blocks[to_idx];

    let edge = from_block
        .succs
        .iter()
        .find(|edge| edge.to == to)
        .ok_or(InterpreterError::MissingEdge { from, to })?;

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

    Ok(())
}
