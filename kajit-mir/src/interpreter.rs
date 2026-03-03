use std::collections::HashMap;

use kajit_abi::DeserContext;
use kajit_ir::ErrorCode;
use kajit_lir::{BinOpKind, LinearOp, UnaryOpKind};

use crate::intrinsic_calls::{
    call_intrinsic_with_output, call_intrinsic_with_result, call_pure_u64,
};
use crate::{BlockId, RaFunction, RaProgram, RaTerminator};

const MAX_EXEC_STEPS: usize = 1_000_000;
const INTRINSIC_OUT_SCRATCH_BYTES: usize = 32;

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
    IntrinsicCallFailed {
        block: BlockId,
        op: String,
        detail: String,
    },
    UnknownIntrinsicErrorCode {
        raw: u32,
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
            Self::IntrinsicCallFailed { block, op, detail } => write!(
                f,
                "intrinsic call failed in block b{} ({}): {}",
                block.0, op, detail
            ),
            Self::UnknownIntrinsicErrorCode { raw } => {
                write!(f, "intrinsic wrote unknown error code: {raw}")
            }
            Self::StepLimitExceeded { limit } => {
                write!(f, "RA-MIR interpreter exceeded step limit ({limit})")
            }
        }
    }
}

impl std::error::Error for InterpreterError {}

struct InterpreterState<'a> {
    input: &'a [u8],
    cursor: usize,
    ctx: DeserContext,
    input_start: *const u8,
    output: Vec<u8>,
    slots: Vec<u64>,
    vregs: Vec<u64>,
    trap: Option<InterpreterTrap>,
}

impl<'a> InterpreterState<'a> {
    fn new(input: &'a [u8], vreg_count: usize, output_size: usize, slot_count: usize) -> Self {
        let input_start = input.as_ptr();
        Self {
            input,
            cursor: 0,
            ctx: DeserContext::from_bytes(input),
            input_start,
            output: vec![0u8; output_size],
            slots: vec![0u64; slot_count],
            vregs: vec![0u64; vreg_count],
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

    fn ensure_slot_index(&mut self, slot: usize) {
        if slot >= self.slots.len() {
            self.slots.resize(slot + 1, 0);
        }
    }

    fn read_slot(&self, slot: usize) -> u64 {
        self.slots.get(slot).copied().unwrap_or(0)
    }

    fn write_slot(&mut self, slot: usize, value: u64) {
        self.ensure_slot_index(slot);
        self.slots[slot] = value;
    }

    fn slot_addr(&mut self, slot: usize) -> u64 {
        self.ensure_slot_index(slot);
        (&mut self.slots[slot] as *mut u64 as usize) as u64
    }

    fn sync_ctx_from_cursor(&mut self) {
        self.ctx.input_ptr = unsafe { self.input_start.add(self.cursor) };
    }

    fn sync_cursor_from_ctx(&mut self) {
        self.cursor = unsafe { self.ctx.input_ptr.offset_from(self.input_start) as usize };
    }

    fn trap(&mut self, code: ErrorCode) {
        if self.trap.is_none() {
            self.trap = Some(InterpreterTrap {
                code,
                offset: self.cursor as u32,
            });
        }
    }
}

/// Execute the first function in an RA-MIR program.
pub fn execute_program(
    program: &RaProgram,
    input: &[u8],
) -> Result<InterpreterOutcome, InterpreterError> {
    let func = program.funcs.first().ok_or(InterpreterError::NoFunctions)?;
    execute_function_with_slots(
        func,
        program.vreg_count as usize,
        program.slot_count as usize,
        input,
    )
}

/// Execute a single RA-MIR function.
pub fn execute_function(
    func: &RaFunction,
    vreg_count: usize,
    input: &[u8],
) -> Result<InterpreterOutcome, InterpreterError> {
    execute_function_with_slots(func, vreg_count, infer_slot_count(func), input)
}

fn execute_function_with_slots(
    func: &RaFunction,
    vreg_count: usize,
    slot_count: usize,
    input: &[u8],
) -> Result<InterpreterOutcome, InterpreterError> {
    let block_indices = build_block_index(func);
    let mut state = InterpreterState::new(input, vreg_count, infer_output_size(func), slot_count);
    let mut current = func.entry;
    let mut steps = 0usize;

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

        for inst in &block.insts {
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
                    let value = exec_unary(*op, src);
                    state.write_vreg(dst.index(), value);
                }
                LinearOp::BoundsCheck { count } => {
                    let count = *count as usize;
                    if state.cursor + count > state.input.len() {
                        state.trap(ErrorCode::UnexpectedEof);
                        break;
                    }
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
                    state.sync_ctx_from_cursor();
                }
                LinearOp::AdvanceCursorBy { src } => {
                    let count = state.read_vreg(src.index()) as usize;
                    if state.cursor + count > state.input.len() {
                        state.trap(ErrorCode::UnexpectedEof);
                        break;
                    }
                    state.cursor += count;
                    state.sync_ctx_from_cursor();
                }
                LinearOp::SaveCursor { dst } => {
                    let ptr = unsafe { state.input_start.add(state.cursor) } as usize as u64;
                    state.write_vreg(dst.index(), ptr);
                }
                LinearOp::RestoreCursor { src } => {
                    let ptr = state.read_vreg(src.index()) as usize;
                    let start = state.input_start as usize;
                    let end = start + state.input.len();
                    if !(start..=end).contains(&ptr) {
                        state.trap(ErrorCode::UnexpectedEof);
                        break;
                    }
                    state.cursor = ptr - start;
                    state.sync_ctx_from_cursor();
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
                    state.sync_ctx_from_cursor();
                    state.write_vreg(dst.index(), value);
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
                    let ptr = state.slot_addr(slot.index());
                    state.write_vreg(dst.index(), ptr);
                }
                LinearOp::WriteToSlot { slot, src } => {
                    let value = state.read_vreg(src.index());
                    state.write_slot(slot.index(), value);
                }
                LinearOp::ReadFromSlot { dst, slot } => {
                    let value = state.read_slot(slot.index());
                    state.write_vreg(dst.index(), value);
                }
                LinearOp::CallPure { func, args, dst } => {
                    let arg_values: Vec<u64> = args
                        .iter()
                        .map(|arg| state.read_vreg(arg.index()))
                        .collect();
                    let result = unsafe { call_pure_u64(func.0, &arg_values) }.map_err(|err| {
                        InterpreterError::IntrinsicCallFailed {
                            block: block.id,
                            op: "CallPure".to_owned(),
                            detail: err.to_string(),
                        }
                    })?;
                    state.write_vreg(dst.index(), result);
                }
                LinearOp::CallIntrinsic {
                    func,
                    args,
                    dst,
                    field_offset,
                } => {
                    state.ctx.error.code = 0;
                    state.ctx.error.offset = 0;
                    let arg_values: Vec<u64> = args
                        .iter()
                        .map(|arg| state.read_vreg(arg.index()))
                        .collect();
                    match dst {
                        Some(dst) => {
                            let result = unsafe {
                                call_intrinsic_with_result(func.0, &mut state.ctx, &arg_values)
                            }
                            .map_err(|err| {
                                InterpreterError::IntrinsicCallFailed {
                                    block: block.id,
                                    op: "CallIntrinsic(result)".to_owned(),
                                    detail: err.to_string(),
                                }
                            })?;
                            state.write_vreg(dst.index(), result);
                        }
                        None => {
                            let out_base = *field_offset as usize;
                            state.ensure_output_len(out_base + INTRINSIC_OUT_SCRATCH_BYTES);
                            let out_ptr = unsafe { state.output.as_mut_ptr().add(out_base) };
                            unsafe {
                                call_intrinsic_with_output(
                                    func.0,
                                    &mut state.ctx,
                                    &arg_values,
                                    out_ptr,
                                )
                            }
                            .map_err(|err| {
                                InterpreterError::IntrinsicCallFailed {
                                    block: block.id,
                                    op: "CallIntrinsic(output)".to_owned(),
                                    detail: err.to_string(),
                                }
                            })?;
                        }
                    }

                    state.sync_cursor_from_ctx();
                    if state.ctx.error.code != 0 {
                        let code = decode_error_code(state.ctx.error.code).ok_or(
                            InterpreterError::UnknownIntrinsicErrorCode {
                                raw: state.ctx.error.code,
                            },
                        )?;
                        state.trap = Some(InterpreterTrap {
                            code,
                            offset: state.ctx.error.offset,
                        });
                        break;
                    }
                }
                op => {
                    return Err(InterpreterError::UnsupportedOp {
                        block: block.id,
                        op: format!("{op:?}"),
                    });
                }
            }
        }

        if state.trap.is_some() {
            return Ok(InterpreterOutcome {
                vregs: state.vregs,
                output: state.output,
                cursor: state.cursor,
                trap: state.trap,
            });
        }

        match &block.term {
            RaTerminator::Return => {
                return Ok(InterpreterOutcome {
                    vregs: state.vregs,
                    output: state.output,
                    cursor: state.cursor,
                    trap: None,
                });
            }
            RaTerminator::ErrorExit { code } => {
                state.trap(*code);
                return Ok(InterpreterOutcome {
                    vregs: state.vregs,
                    output: state.output,
                    cursor: state.cursor,
                    trap: state.trap,
                });
            }
            RaTerminator::Branch { target } => {
                apply_edge_args(func, &block_indices, &mut state, block.id, *target)?;
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
                current = next;
            }
            RaTerminator::JumpTable {
                predicate,
                targets,
                default,
            } => {
                let pred = state.read_vreg(predicate.index()) as usize;
                let next = targets.get(pred).copied().unwrap_or(*default);
                apply_edge_args(func, &block_indices, &mut state, block.id, next)?;
                current = next;
            }
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
            LinearOp::ReadFromField { offset, width, .. } => {
                Some(*offset as usize + width.bytes() as usize)
            }
            LinearOp::CallIntrinsic {
                dst: None,
                field_offset,
                ..
            } => Some(*field_offset as usize + INTRINSIC_OUT_SCRATCH_BYTES),
            _ => None,
        })
        .max()
        .unwrap_or(0)
}

fn infer_slot_count(func: &RaFunction) -> usize {
    func.blocks
        .iter()
        .flat_map(|block| block.insts.iter())
        .filter_map(|inst| match &inst.op {
            LinearOp::SlotAddr { slot, .. }
            | LinearOp::WriteToSlot { slot, .. }
            | LinearOp::ReadFromSlot { slot, .. } => Some(slot.index() + 1),
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

fn exec_unary(op: UnaryOpKind, src: u64) -> u64 {
    match op {
        UnaryOpKind::ZigzagDecode { wide: true } => {
            ((src >> 1) as i64 ^ -((src & 1) as i64)) as u64
        }
        UnaryOpKind::ZigzagDecode { wide: false } => {
            let src = src as u32;
            ((src >> 1) as i32 ^ -((src & 1) as i32)) as i64 as u64
        }
        UnaryOpKind::SignExtend { from_width } => match from_width.bytes() {
            1 => (src as u8 as i8 as i64) as u64,
            2 => (src as u16 as i16 as i64) as u64,
            4 => (src as u32 as i32 as i64) as u64,
            8 => src,
            _ => src,
        },
    }
}

fn decode_error_code(raw: u32) -> Option<ErrorCode> {
    Some(match raw {
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
        _ => return None,
    })
}

#[cfg(test)]
mod tests {
    use kajit_abi::DeserContext;
    use kajit_ir::{IntrinsicFn, LambdaId, SlotId, VReg, Width};
    use kajit_lir::LinearOp;

    use crate::{BlockId, RaBlock, RaClobbers, RaFunction, RaInst, RaProgram, RaTerminator};

    use super::execute_program;

    fn v(index: u32) -> VReg {
        VReg::new(index)
    }

    fn inst(op: LinearOp) -> RaInst {
        RaInst {
            linear_op_index: 0,
            op,
            operands: Vec::new(),
            clobbers: RaClobbers {
                caller_saved_gpr: false,
                caller_saved_simd: false,
            },
        }
    }

    fn program(insts: Vec<LinearOp>, vreg_count: u32, slot_count: u32) -> RaProgram {
        RaProgram {
            funcs: vec![RaFunction {
                lambda_id: LambdaId::new(0),
                entry: BlockId(0),
                data_args: Vec::new(),
                data_results: Vec::new(),
                blocks: vec![RaBlock {
                    id: BlockId(0),
                    label: None,
                    params: Vec::new(),
                    insts: insts.into_iter().map(inst).collect(),
                    term_linear_op_index: None,
                    term: RaTerminator::Return,
                    preds: Vec::new(),
                    succs: Vec::new(),
                }],
            }],
            vreg_count,
            slot_count,
        }
    }

    unsafe extern "C" fn add_u64(a: u64, b: u64) -> u64 {
        a.wrapping_add(b)
    }

    unsafe extern "C" fn read_u8_intrinsic(ctx: *mut DeserContext) -> u64 {
        let ctx = unsafe { &mut *ctx };
        if ctx.input_ptr >= ctx.input_end {
            ctx.error.code = kajit_ir::ErrorCode::UnexpectedEof as u32;
            ctx.error.offset = 0;
            return 0;
        }
        let byte = unsafe { *ctx.input_ptr };
        ctx.input_ptr = unsafe { ctx.input_ptr.add(1) };
        byte as u64
    }

    unsafe extern "C" fn store_u8_intrinsic(_ctx: *mut DeserContext, value: u64, out: *mut u8) {
        unsafe { *out = value as u8 };
    }

    #[test]
    fn executes_call_pure() {
        let program = program(
            vec![
                LinearOp::Const {
                    dst: v(0),
                    value: 40,
                },
                LinearOp::Const {
                    dst: v(1),
                    value: 2,
                },
                LinearOp::CallPure {
                    func: IntrinsicFn(add_u64 as *const () as usize),
                    args: vec![v(0), v(1)],
                    dst: v(2),
                },
                LinearOp::WriteToField {
                    src: v(2),
                    offset: 0,
                    width: Width::W1,
                },
            ],
            3,
            0,
        );
        let outcome = execute_program(&program, &[]).expect("program should execute");
        assert_eq!(outcome.trap, None);
        assert_eq!(outcome.output[0], 42);
    }

    #[test]
    fn executes_call_intrinsic_result_and_advances_cursor() {
        let program = program(
            vec![
                LinearOp::CallIntrinsic {
                    func: IntrinsicFn(read_u8_intrinsic as *const () as usize),
                    args: vec![],
                    dst: Some(v(0)),
                    field_offset: 0,
                },
                LinearOp::WriteToField {
                    src: v(0),
                    offset: 0,
                    width: Width::W1,
                },
            ],
            1,
            0,
        );

        let outcome = execute_program(&program, &[0x2a]).expect("program should execute");
        assert_eq!(outcome.trap, None);
        assert_eq!(outcome.cursor, 1);
        assert_eq!(outcome.output[0], 0x2a);
    }

    #[test]
    fn executes_call_intrinsic_output_and_slot_ops() {
        let program = program(
            vec![
                LinearOp::Const {
                    dst: v(0),
                    value: 42,
                },
                LinearOp::WriteToSlot {
                    slot: SlotId::new(0),
                    src: v(0),
                },
                LinearOp::ReadFromSlot {
                    dst: v(1),
                    slot: SlotId::new(0),
                },
                LinearOp::CallIntrinsic {
                    func: IntrinsicFn(store_u8_intrinsic as *const () as usize),
                    args: vec![v(1)],
                    dst: None,
                    field_offset: 3,
                },
            ],
            2,
            1,
        );
        let outcome = execute_program(&program, &[]).expect("program should execute");
        assert_eq!(outcome.trap, None);
        assert!(outcome.output.len() > 3);
        assert_eq!(outcome.output[3], 42);
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

    let values: Vec<u64> = edge
        .args
        .iter()
        .map(|arg| state.read_vreg(arg.index()))
        .collect();

    for (param, value) in to_block.params.iter().zip(values.into_iter()) {
        state.write_vreg(param.index(), value);
    }

    Ok(())
}
