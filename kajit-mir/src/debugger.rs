use std::collections::HashMap;

use kajit_abi::DeserContext;
use kajit_ir::ErrorCode;
use kajit_lir::{BinOpKind, LinearOp, UnaryOpKind};

use crate::intrinsic_calls::{
    call_intrinsic_with_output, call_intrinsic_with_result, call_pure_u64,
};
use crate::{BlockId, InterpreterTrap, RaFunction, RaProgram, RaTerminator};

const INTRINSIC_OUT_SCRATCH_BYTES: usize = 32;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DebuggerError {
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
}

impl std::fmt::Display for DebuggerError {
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
                write!(f, "unsupported RA-MIR op in block b{}: {}", block.0, op)
            }
            Self::UnsupportedTerminator { block, term } => {
                write!(
                    f,
                    "unsupported RA-MIR terminator in block b{}: {}",
                    block.0, term
                )
            }
            Self::IntrinsicCallFailed { block, op, detail } => {
                write!(
                    f,
                    "intrinsic call failed in block b{} ({}): {}",
                    block.0, op, detail
                )
            }
            Self::UnknownIntrinsicErrorCode { raw } => {
                write!(f, "intrinsic wrote unknown error code: {raw}")
            }
        }
    }
}

impl std::error::Error for DebuggerError {}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ProgramLocation {
    pub block: BlockId,
    pub next_inst_index: usize,
    pub at_terminator: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DebuggerState {
    pub step_count: usize,
    pub location: ProgramLocation,
    pub cursor: usize,
    pub vregs: Vec<u64>,
    pub output: Vec<u8>,
    pub trap: Option<InterpreterTrap>,
    pub returned: bool,
    pub halted: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StepKind {
    Instruction,
    Terminator,
    HaltedNoop,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StepEvent {
    pub step_index: usize,
    pub kind: StepKind,
    pub location_before: ProgramLocation,
    pub location_after: ProgramLocation,
    pub cursor_before: usize,
    pub cursor_after: usize,
    pub trap: Option<InterpreterTrap>,
    pub returned: bool,
    pub halted_after: bool,
    pub detail: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RunUntilTarget {
    Block(BlockId),
    Trap,
    Return,
}

#[derive(Debug, Clone)]
struct SessionSnapshot {
    cursor: usize,
    output: Vec<u8>,
    slots: Vec<u64>,
    vregs: Vec<u64>,
    trap: Option<InterpreterTrap>,
    returned: bool,
    current: BlockId,
    next_inst: usize,
    steps: usize,
}

pub struct DebuggerSession {
    func: RaFunction,
    block_indices: HashMap<BlockId, usize>,
    input: Vec<u8>,
    input_start: *const u8,
    ctx: DeserContext,
    cursor: usize,
    output: Vec<u8>,
    slots: Vec<u64>,
    vregs: Vec<u64>,
    trap: Option<InterpreterTrap>,
    returned: bool,
    current: BlockId,
    next_inst: usize,
    steps: usize,
    history: Vec<SessionSnapshot>,
}

impl DebuggerSession {
    pub fn new(program: &RaProgram, input: &[u8]) -> Result<Self, DebuggerError> {
        let func = program
            .funcs
            .first()
            .ok_or(DebuggerError::NoFunctions)?
            .clone();
        let block_indices = build_block_index(&func);
        let input = input.to_vec();
        let input_start = input.as_ptr();
        let ctx = DeserContext::from_bytes(input.as_slice());
        Ok(Self {
            cursor: 0,
            output: vec![0u8; infer_output_size(&func)],
            slots: vec![0u64; infer_slot_count(&func).max(program.slot_count as usize)],
            vregs: vec![0u64; program.vreg_count as usize],
            trap: None,
            returned: false,
            current: func.entry,
            next_inst: 0,
            steps: 0,
            history: Vec::new(),
            func,
            block_indices,
            input,
            input_start,
            ctx,
        })
    }

    pub fn state(&self) -> DebuggerState {
        DebuggerState {
            step_count: self.steps,
            location: self.location(),
            cursor: self.cursor,
            vregs: self.vregs.clone(),
            output: self.output.clone(),
            trap: self.trap,
            returned: self.returned,
            halted: self.is_halted(),
        }
    }

    pub fn inspect_vreg(&self, vreg_index: usize) -> u64 {
        self.read_vreg(vreg_index)
    }

    pub fn inspect_output(&self, start: usize, len: usize) -> Vec<u8> {
        if start >= self.output.len() {
            return Vec::new();
        }
        let end = start.saturating_add(len).min(self.output.len());
        self.output[start..end].to_vec()
    }

    pub fn step_forward(&mut self) -> Result<StepEvent, DebuggerError> {
        let location_before = self.location();
        let cursor_before = self.cursor;
        if self.is_halted() {
            return Ok(StepEvent {
                step_index: self.steps,
                kind: StepKind::HaltedNoop,
                location_before,
                location_after: location_before,
                cursor_before,
                cursor_after: self.cursor,
                trap: self.trap,
                returned: self.returned,
                halted_after: true,
                detail: "halted".to_owned(),
            });
        }

        let snapshot = self.snapshot();
        self.history.push(snapshot.clone());

        let step_detail;
        let step_kind;
        let block = self.current_block()?;
        if self.next_inst < block.insts.len() {
            let op = block.insts[self.next_inst].op.clone();
            self.execute_op(block.id, &op)?;
            self.next_inst += 1;
            step_detail = format!("{op:?}");
            step_kind = StepKind::Instruction;
        } else {
            let term = block.term.clone();
            self.execute_terminator(block.id, &term)?;
            step_detail = format!("{term:?}");
            step_kind = StepKind::Terminator;
        }

        self.steps += 1;
        let location_after = self.location();
        Ok(StepEvent {
            step_index: self.steps,
            kind: step_kind,
            location_before,
            location_after,
            cursor_before,
            cursor_after: self.cursor,
            trap: self.trap,
            returned: self.returned,
            halted_after: self.is_halted(),
            detail: step_detail,
        })
    }

    pub fn step_back(&mut self) -> bool {
        match self.history.pop() {
            Some(snapshot) => {
                self.restore(snapshot);
                true
            }
            None => false,
        }
    }

    pub fn run_until(
        &mut self,
        target: RunUntilTarget,
        max_steps: usize,
    ) -> Result<Vec<StepEvent>, DebuggerError> {
        let mut events = Vec::new();
        if self.target_reached(target) {
            return Ok(events);
        }

        for _ in 0..max_steps {
            let event = self.step_forward()?;
            events.push(event);
            if self.target_reached(target) || self.is_halted() {
                break;
            }
        }

        Ok(events)
    }

    fn target_reached(&self, target: RunUntilTarget) -> bool {
        match target {
            RunUntilTarget::Block(block) => self.current == block,
            RunUntilTarget::Trap => self.trap.is_some(),
            RunUntilTarget::Return => self.returned,
        }
    }

    fn is_halted(&self) -> bool {
        self.trap.is_some() || self.returned
    }

    fn location(&self) -> ProgramLocation {
        let at_terminator = self
            .current_block()
            .map(|block| self.next_inst >= block.insts.len())
            .unwrap_or(true);
        ProgramLocation {
            block: self.current,
            next_inst_index: self.next_inst,
            at_terminator,
        }
    }

    fn current_block(&self) -> Result<&crate::RaBlock, DebuggerError> {
        let idx = *self
            .block_indices
            .get(&self.current)
            .ok_or(DebuggerError::UnknownBlock {
                block: self.current,
            })?;
        Ok(&self.func.blocks[idx])
    }

    fn snapshot(&self) -> SessionSnapshot {
        SessionSnapshot {
            cursor: self.cursor,
            output: self.output.clone(),
            slots: self.slots.clone(),
            vregs: self.vregs.clone(),
            trap: self.trap,
            returned: self.returned,
            current: self.current,
            next_inst: self.next_inst,
            steps: self.steps,
        }
    }

    fn restore(&mut self, snapshot: SessionSnapshot) {
        self.cursor = snapshot.cursor;
        self.output = snapshot.output;
        self.slots = snapshot.slots;
        self.vregs = snapshot.vregs;
        self.trap = snapshot.trap;
        self.returned = snapshot.returned;
        self.current = snapshot.current;
        self.next_inst = snapshot.next_inst;
        self.steps = snapshot.steps;
        self.sync_ctx_from_cursor();
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

    fn execute_op(&mut self, block: BlockId, op: &LinearOp) -> Result<(), DebuggerError> {
        match op {
            LinearOp::Const { dst, value } => self.write_vreg(dst.index(), *value),
            LinearOp::Copy { dst, src } => {
                let value = self.read_vreg(src.index());
                self.write_vreg(dst.index(), value);
            }
            LinearOp::UnaryOp { op, dst, src } => {
                let src = self.read_vreg(src.index());
                let value = exec_unary(*op, src);
                self.write_vreg(dst.index(), value);
            }
            LinearOp::BinOp { op, dst, lhs, rhs } => {
                let lhs = self.read_vreg(lhs.index());
                let rhs = self.read_vreg(rhs.index());
                self.write_vreg(dst.index(), exec_binop(*op, lhs, rhs));
            }
            LinearOp::BoundsCheck { count } => {
                if self.cursor + (*count as usize) > self.input.len() {
                    self.trap(ErrorCode::UnexpectedEof);
                }
            }
            LinearOp::PeekByte { dst } => {
                if self.cursor >= self.input.len() {
                    self.trap(ErrorCode::UnexpectedEof);
                } else {
                    self.write_vreg(dst.index(), self.input[self.cursor] as u64);
                }
            }
            LinearOp::AdvanceCursor { count } => {
                let count = *count as usize;
                if self.cursor + count > self.input.len() {
                    self.trap(ErrorCode::UnexpectedEof);
                } else {
                    self.cursor += count;
                    self.sync_ctx_from_cursor();
                }
            }
            LinearOp::AdvanceCursorBy { src } => {
                let count = self.read_vreg(src.index()) as usize;
                if self.cursor + count > self.input.len() {
                    self.trap(ErrorCode::UnexpectedEof);
                } else {
                    self.cursor += count;
                    self.sync_ctx_from_cursor();
                }
            }
            LinearOp::SaveCursor { dst } => {
                let ptr = unsafe { self.input_start.add(self.cursor) } as usize as u64;
                self.write_vreg(dst.index(), ptr);
            }
            LinearOp::RestoreCursor { src } => {
                let ptr = self.read_vreg(src.index()) as usize;
                let start = self.input_start as usize;
                let end = start + self.input.len();
                if !(start..=end).contains(&ptr) {
                    self.trap(ErrorCode::UnexpectedEof);
                } else {
                    self.cursor = ptr - start;
                    self.sync_ctx_from_cursor();
                }
            }
            LinearOp::ReadBytes { dst, count } => {
                let count = *count as usize;
                if self.cursor + count > self.input.len() {
                    self.trap(ErrorCode::UnexpectedEof);
                } else {
                    let mut value = 0u64;
                    for i in 0..count {
                        value |= (self.input[self.cursor + i] as u64) << (i * 8);
                    }
                    self.cursor += count;
                    self.sync_ctx_from_cursor();
                    self.write_vreg(dst.index(), value);
                }
            }
            LinearOp::WriteToField { src, offset, width } => {
                let value = self.read_vreg(src.index());
                let base = *offset as usize;
                let width = width.bytes() as usize;
                self.ensure_output_len(base + width);
                for i in 0..width {
                    self.output[base + i] = ((value >> (i * 8)) & 0xff) as u8;
                }
            }
            LinearOp::ReadFromField { dst, offset, width } => {
                let base = *offset as usize;
                let width = width.bytes() as usize;
                self.ensure_output_len(base + width);
                let mut value = 0u64;
                for i in 0..width {
                    value |= (self.output[base + i] as u64) << (i * 8);
                }
                self.write_vreg(dst.index(), value);
            }
            LinearOp::SlotAddr { dst, slot } => {
                let ptr = self.slot_addr(slot.index());
                self.write_vreg(dst.index(), ptr);
            }
            LinearOp::WriteToSlot { slot, src } => {
                let value = self.read_vreg(src.index());
                self.write_slot(slot.index(), value);
            }
            LinearOp::ReadFromSlot { dst, slot } => {
                let value = self.read_slot(slot.index());
                self.write_vreg(dst.index(), value);
            }
            LinearOp::CallPure { func, args, dst } => {
                let arg_values: Vec<u64> =
                    args.iter().map(|arg| self.read_vreg(arg.index())).collect();
                let result = unsafe { call_pure_u64(func.0, &arg_values) }.map_err(|err| {
                    DebuggerError::IntrinsicCallFailed {
                        block,
                        op: "CallPure".to_owned(),
                        detail: err.to_string(),
                    }
                })?;
                self.write_vreg(dst.index(), result);
            }
            LinearOp::CallIntrinsic {
                func,
                args,
                dst,
                field_offset,
            } => {
                self.ctx.error.code = 0;
                self.ctx.error.offset = 0;
                let arg_values: Vec<u64> =
                    args.iter().map(|arg| self.read_vreg(arg.index())).collect();
                match dst {
                    Some(dst) => {
                        let result = unsafe {
                            call_intrinsic_with_result(func.0, &mut self.ctx, &arg_values)
                        }
                        .map_err(|err| {
                            DebuggerError::IntrinsicCallFailed {
                                block,
                                op: "CallIntrinsic(result)".to_owned(),
                                detail: err.to_string(),
                            }
                        })?;
                        self.write_vreg(dst.index(), result);
                    }
                    None => {
                        let out_base = *field_offset as usize;
                        self.ensure_output_len(out_base + INTRINSIC_OUT_SCRATCH_BYTES);
                        let out_ptr = unsafe { self.output.as_mut_ptr().add(out_base) };
                        unsafe {
                            call_intrinsic_with_output(func.0, &mut self.ctx, &arg_values, out_ptr)
                        }
                        .map_err(|err| {
                            DebuggerError::IntrinsicCallFailed {
                                block,
                                op: "CallIntrinsic(output)".to_owned(),
                                detail: err.to_string(),
                            }
                        })?;
                    }
                }

                self.sync_cursor_from_ctx();
                if self.ctx.error.code != 0 {
                    let code = decode_error_code(self.ctx.error.code).ok_or(
                        DebuggerError::UnknownIntrinsicErrorCode {
                            raw: self.ctx.error.code,
                        },
                    )?;
                    self.trap = Some(InterpreterTrap {
                        code,
                        offset: self.ctx.error.offset,
                    });
                }
            }
            LinearOp::ErrorExit { code } => {
                self.trap(*code);
            }
            op => {
                return Err(DebuggerError::UnsupportedOp {
                    block,
                    op: format!("{op:?}"),
                });
            }
        }

        Ok(())
    }

    fn execute_terminator(
        &mut self,
        block_id: BlockId,
        term: &RaTerminator,
    ) -> Result<(), DebuggerError> {
        match term {
            RaTerminator::Return => {
                self.returned = true;
            }
            RaTerminator::ErrorExit { code } => {
                self.trap(*code);
            }
            RaTerminator::Branch { target } => {
                self.apply_edge_args(block_id, *target)?;
                self.current = *target;
                self.next_inst = 0;
            }
            RaTerminator::BranchIf {
                cond,
                target,
                fallthrough,
            } => {
                let next = if self.read_vreg(cond.index()) != 0 {
                    *target
                } else {
                    *fallthrough
                };
                self.apply_edge_args(block_id, next)?;
                self.current = next;
                self.next_inst = 0;
            }
            RaTerminator::BranchIfZero {
                cond,
                target,
                fallthrough,
            } => {
                let next = if self.read_vreg(cond.index()) == 0 {
                    *target
                } else {
                    *fallthrough
                };
                self.apply_edge_args(block_id, next)?;
                self.current = next;
                self.next_inst = 0;
            }
            RaTerminator::JumpTable {
                predicate,
                targets,
                default,
            } => {
                let pred = self.read_vreg(predicate.index()) as usize;
                let next = targets.get(pred).copied().unwrap_or(*default);
                self.apply_edge_args(block_id, next)?;
                self.current = next;
                self.next_inst = 0;
            }
        }

        Ok(())
    }

    fn apply_edge_args(&mut self, from: BlockId, to: BlockId) -> Result<(), DebuggerError> {
        let from_idx = *self
            .block_indices
            .get(&from)
            .ok_or(DebuggerError::UnknownBlock { block: from })?;
        let to_idx = *self
            .block_indices
            .get(&to)
            .ok_or(DebuggerError::UnknownBlock { block: to })?;
        let from_block = &self.func.blocks[from_idx];
        let to_block = &self.func.blocks[to_idx];
        let edge = from_block
            .succs
            .iter()
            .find(|edge| edge.to == to)
            .ok_or(DebuggerError::MissingEdge { from, to })?;

        if edge.args.len() != to_block.params.len() {
            return Err(DebuggerError::EdgeArgArityMismatch {
                from,
                to,
                expected: to_block.params.len(),
                got: edge.args.len(),
            });
        }

        let values: Vec<u64> = edge
            .args
            .iter()
            .map(|arg| self.read_vreg(arg.index()))
            .collect();
        let params: Vec<usize> = to_block.params.iter().map(|param| param.index()).collect();
        for (param, value) in params.into_iter().zip(values.into_iter()) {
            self.write_vreg(param, value);
        }
        Ok(())
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
            LinearOp::WriteToField { offset, width, .. }
            | LinearOp::ReadFromField { offset, width, .. } => {
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
    use kajit_ir::{ErrorCode, IntrinsicFn, LambdaId, SlotId, VReg, Width};
    use kajit_lir::LinearOp;

    use crate::{
        BlockId, DebuggerSession, RaBlock, RaClobbers, RaFunction, RaInst, RaProgram, RaTerminator,
    };

    fn v(index: u32) -> VReg {
        VReg::new(index)
    }

    fn test_inst(op: LinearOp) -> RaInst {
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

    fn make_simple_program() -> RaProgram {
        let b0 = RaBlock {
            id: BlockId(0),
            label: None,
            params: Vec::new(),
            insts: vec![
                test_inst(LinearOp::Const {
                    dst: v(0),
                    value: 0x2a,
                }),
                test_inst(LinearOp::WriteToField {
                    src: v(0),
                    offset: 0,
                    width: Width::W1,
                }),
            ],
            term_linear_op_index: None,
            term: RaTerminator::Return,
            preds: Vec::new(),
            succs: Vec::new(),
        };
        RaProgram {
            funcs: vec![RaFunction {
                lambda_id: LambdaId::new(0),
                entry: BlockId(0),
                data_args: Vec::new(),
                data_results: Vec::new(),
                blocks: vec![b0],
            }],
            vreg_count: 1,
            slot_count: 0,
        }
    }

    fn make_trap_program() -> RaProgram {
        let b0 = RaBlock {
            id: BlockId(0),
            label: None,
            params: Vec::new(),
            insts: vec![test_inst(LinearOp::BoundsCheck { count: 1 })],
            term_linear_op_index: None,
            term: RaTerminator::Return,
            preds: Vec::new(),
            succs: Vec::new(),
        };
        RaProgram {
            funcs: vec![RaFunction {
                lambda_id: LambdaId::new(0),
                entry: BlockId(0),
                data_args: Vec::new(),
                data_results: Vec::new(),
                blocks: vec![b0],
            }],
            vreg_count: 0,
            slot_count: 0,
        }
    }

    unsafe extern "C" fn read_u8_intrinsic(ctx: *mut DeserContext) -> u64 {
        let ctx = unsafe { &mut *ctx };
        if ctx.input_ptr >= ctx.input_end {
            ctx.error.code = ErrorCode::UnexpectedEof as u32;
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

    fn make_intrinsic_program() -> RaProgram {
        let b0 = RaBlock {
            id: BlockId(0),
            label: None,
            params: Vec::new(),
            insts: vec![
                test_inst(LinearOp::CallIntrinsic {
                    func: IntrinsicFn(read_u8_intrinsic as *const () as usize),
                    args: vec![],
                    dst: Some(v(0)),
                    field_offset: 0,
                }),
                test_inst(LinearOp::WriteToSlot {
                    slot: SlotId::new(0),
                    src: v(0),
                }),
                test_inst(LinearOp::ReadFromSlot {
                    dst: v(1),
                    slot: SlotId::new(0),
                }),
                test_inst(LinearOp::CallIntrinsic {
                    func: IntrinsicFn(store_u8_intrinsic as *const () as usize),
                    args: vec![v(1)],
                    dst: None,
                    field_offset: 2,
                }),
            ],
            term_linear_op_index: None,
            term: RaTerminator::Return,
            preds: Vec::new(),
            succs: Vec::new(),
        };
        RaProgram {
            funcs: vec![RaFunction {
                lambda_id: LambdaId::new(0),
                entry: BlockId(0),
                data_args: Vec::new(),
                data_results: Vec::new(),
                blocks: vec![b0],
            }],
            vreg_count: 2,
            slot_count: 1,
        }
    }

    #[test]
    fn step_forward_and_back_restores_state() {
        let program = make_simple_program();
        let mut session = DebuggerSession::new(&program, &[]).expect("debugger should init");

        let first = session.step_forward().expect("step should work");
        assert_eq!(first.kind, crate::StepKind::Instruction);
        assert_eq!(session.inspect_vreg(0), 0x2a);

        let second = session.step_forward().expect("step should work");
        assert_eq!(second.kind, crate::StepKind::Instruction);
        assert_eq!(session.inspect_output(0, 1), vec![0x2a]);

        assert!(session.step_back());
        assert_eq!(session.inspect_output(0, 1), vec![0x00]);

        assert!(session.step_back());
        assert_eq!(session.inspect_vreg(0), 0);

        assert!(!session.step_back());
    }

    #[test]
    fn bounds_check_trap_has_expected_offset() {
        let program = make_trap_program();
        let mut session = DebuggerSession::new(&program, &[]).expect("debugger should init");
        let event = session.step_forward().expect("step should work");
        assert!(event.halted_after);
        let trap = event.trap.expect("trap should be recorded");
        assert_eq!(trap.code, ErrorCode::UnexpectedEof);
        assert_eq!(trap.offset, 0);
    }

    #[test]
    fn call_intrinsic_and_slot_ops_step_and_rewind() {
        let program = make_intrinsic_program();
        let mut session = DebuggerSession::new(&program, &[0x2a]).expect("debugger should init");

        session.step_forward().expect("step should work");
        assert_eq!(session.state().cursor, 1);
        assert_eq!(session.inspect_vreg(0), 0x2a);

        session.step_forward().expect("step should work");
        session.step_forward().expect("step should work");
        session.step_forward().expect("step should work");
        assert_eq!(session.inspect_output(2, 1), vec![0x2a]);

        assert!(session.step_back());
        assert_eq!(session.inspect_output(2, 1), vec![0x00]);
    }
}
