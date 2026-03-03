use std::collections::HashMap;

use kajit_ir::ErrorCode;
use kajit_lir::{BinOpKind, LinearOp};

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
    cursor: usize,
    output: Vec<u8>,
    vregs: Vec<u64>,
    trap: Option<InterpreterTrap>,
}

impl<'a> InterpreterState<'a> {
    fn new(input: &'a [u8], vreg_count: usize, output_size: usize) -> Self {
        Self {
            input,
            cursor: 0,
            output: vec![0u8; output_size],
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
    execute_function(func, program.vreg_count as usize, input)
}

/// Execute a single RA-MIR function.
pub fn execute_function(
    func: &RaFunction,
    vreg_count: usize,
    input: &[u8],
) -> Result<InterpreterOutcome, InterpreterError> {
    let block_indices = build_block_index(func);
    let mut state = InterpreterState::new(input, vreg_count, infer_output_size(func));
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
            term => {
                return Err(InterpreterError::UnsupportedTerminator {
                    block: block.id,
                    term: format!("{term:?}"),
                });
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
