//! MIR lowering entrypoints.
//!
//! This takes schema-owned MIR (`kajit_reprs::mir`) plus its `Graph` storage
//! and produces schema-owned ASM (`kajit_reprs::asm`) for the current host
//! architecture.

use std::collections::HashSet;

use kajit_reprs::{asm, mir};

#[derive(Debug, Clone)]
pub struct LoweredAsm {
    /// Canonical `.k-asm` text for `program`.
    ///
    /// We keep this around because the assembler uses `Prov` spans as source
    /// locations, and those spans are relative to the text it is given.
    pub source: String,
    pub program: asm::Program,
}

#[derive(Debug)]
pub enum LowerError {
    Unsupported(&'static str),
    Invalid(String),
}

impl std::fmt::Display for LowerError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Unsupported(msg) => write!(f, "unsupported: {msg}"),
            Self::Invalid(msg) => write!(f, "{msg}"),
        }
    }
}

impl std::error::Error for LowerError {}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum TargetArch {
    AArch64,
    X86_64,
}

fn host_arch() -> Result<TargetArch, LowerError> {
    if cfg!(target_arch = "aarch64") {
        Ok(TargetArch::AArch64)
    } else if cfg!(target_arch = "x86_64") {
        Ok(TargetArch::X86_64)
    } else {
        Err(LowerError::Unsupported("host arch is not supported"))
    }
}

fn require<T>(value: Option<T>, msg: impl Into<String>) -> Result<T, LowerError> {
    value.ok_or_else(|| LowerError::Invalid(msg.into()))
}

fn default_prov() -> asm::Prov {
    asm::Prov::default()
}

fn label_name(block_id: mir::BlockId) -> asm::LabelName {
    asm::LabelName {
        prov: default_prov(),
        text: format!("b{}", block_id.0),
    }
}

fn asm_imm(value: u64) -> asm::Imm {
    asm::Imm {
        prov: default_prov(),
        value,
    }
}

/// Minimal MIR→ASM lowering:
/// - Host-arch only
/// - Single function
/// - Linearized blocks with explicit labels and unconditional branches
/// - Supports: `const(<imm>)`, `copy`, `Add`, `Sub`
/// - Requires fixed-register operands
pub fn lower_program_to_asm(
    mir_graph: &mir::Graph,
    mir_program: &mir::Program,
) -> Result<LoweredAsm, LowerError> {
    let arch = host_arch()?;

    let function = require(
        mir_program.functions.iter().next(),
        "MIR program has no functions",
    )?;
    if mir_program.functions.len() != 1 {
        return Err(LowerError::Unsupported(
            "multiple functions are not supported yet",
        ));
    }

    let func_storage = mir::FunctionStorage::new(mir_graph, function)
        .map_err(|e| LowerError::Invalid(e.to_string()))?;
    let blocks = schedule_blocks(&func_storage)?;

    let program = match arch {
        TargetArch::AArch64 => asm::Program::AArch64 {
            dialect: Box::new(asm::AArch64DialectKeyword {
                prov: default_prov(),
            }),
            docs: None,
            items: lower_aarch64_items(&func_storage, &blocks)?,
            keyword: Box::new(asm::AsmKeyword {
                prov: default_prov(),
            }),
            prov: default_prov(),
        },
        TargetArch::X86_64 => asm::Program::X86_64 {
            dialect: Box::new(asm::X86_64DialectKeyword {
                prov: default_prov(),
            }),
            docs: None,
            items: lower_x64_items(&func_storage, &blocks)?,
            keyword: Box::new(asm::AsmKeyword {
                prov: default_prov(),
            }),
            prov: default_prov(),
        },
    };

    let source = program.to_string();
    Ok(LoweredAsm { source, program })
}

fn schedule_blocks<'a>(
    storage: &'a mir::FunctionStorage<'a>,
) -> Result<Vec<&'a mir::Block>, LowerError> {
    let entry = storage
        .entry_block()
        .map_err(|e| LowerError::Invalid(e.to_string()))?;

    let mut blocks = Vec::new();
    let mut seen = HashSet::new();
    blocks.push(entry);
    seen.insert(entry.id);

    for block in storage.blocks() {
        if seen.insert(block.id) {
            blocks.push(block);
        }
    }

    Ok(blocks)
}

fn lower_aarch64_items(
    storage: &mir::FunctionStorage<'_>,
    blocks: &[&mir::Block],
) -> Result<Vec<asm::A64Item>, LowerError> {
    let mut items = Vec::new();
    for block in blocks {
        items.push(asm::A64Item::Label {
            name: label_name(block.id),
            prov: default_prov(),
        });

        for inst_id in block.insts.iter() {
            let inst = require(
                storage.inst(*inst_id),
                format!("block references missing inst id {inst_id:?}"),
            )?;
            lower_aarch64_inst(&mut items, inst)?;
        }

        lower_aarch64_terminator(&mut items, storage, block)?;
    }

    Ok(items)
}

fn lower_x64_items(
    storage: &mir::FunctionStorage<'_>,
    blocks: &[&mir::Block],
) -> Result<Vec<asm::X64Item>, LowerError> {
    let mut items = Vec::new();
    for block in blocks {
        items.push(asm::X64Item::Label {
            name: label_name(block.id),
            prov: default_prov(),
        });

        for inst_id in block.insts.iter() {
            let inst = require(
                storage.inst(*inst_id),
                format!("block references missing inst id {inst_id:?}"),
            )?;
            lower_x64_inst(&mut items, inst)?;
        }

        lower_x64_terminator(&mut items, storage, block)?;
    }

    Ok(items)
}

fn lower_aarch64_inst(items: &mut Vec<asm::A64Item>, inst: &mir::Inst) -> Result<(), LowerError> {
    match inst.op.as_ref() {
        mir::InstOp::Const { value, .. } => {
            let dst = require(inst.defs.as_ref(), "const instruction missing defs")?;
            if dst.len() != 1 {
                return Err(LowerError::Unsupported(
                    "const instructions with multiple defs are not supported yet",
                ));
            }
            let dst = lower_aarch64_operand(&dst[0])?;
            let imm = match value.as_ref() {
                mir::ConstRef::Value { value, .. } => value.value,
                mir::ConstRef::Named { .. } => {
                    return Err(LowerError::Unsupported(
                        "named constants are not supported yet",
                    ));
                }
            };
            let imm = u16::try_from(imm).map_err(|_| {
                LowerError::Unsupported("aarch64 lowering only supports u16 immediates for now")
            })?;
            items.push(asm::A64Item::Movz {
                imm: asm_imm(imm as u64),
                op: Box::new(asm::MovzKeyword {
                    prov: default_prov(),
                }),
                prov: default_prov(),
                rd: Box::new(dst),
            });
        }
        mir::InstOp::Copy(_) => {
            lower_copy_aarch64(items, inst)?;
        }
        mir::InstOp::BinOp {
            op: mir::BinOpKind::Add,
            ..
        } => {
            lower_binop_aarch64(items, inst, true)?;
        }
        mir::InstOp::BinOp {
            op: mir::BinOpKind::Sub,
            ..
        } => {
            lower_binop_aarch64(items, inst, false)?;
        }
        _ => {
            return Err(LowerError::Unsupported(
                "only const, copy, Add, and Sub instructions are supported yet",
            ));
        }
    }

    Ok(())
}

fn lower_x64_inst(items: &mut Vec<asm::X64Item>, inst: &mir::Inst) -> Result<(), LowerError> {
    match inst.op.as_ref() {
        mir::InstOp::Const { value, .. } => {
            let dst = require(inst.defs.as_ref(), "const instruction missing defs")?;
            if dst.len() != 1 {
                return Err(LowerError::Unsupported(
                    "const instructions with multiple defs are not supported yet",
                ));
            }
            let dst = lower_x64_operand(&dst[0])?;
            let imm = match value.as_ref() {
                mir::ConstRef::Value { value, .. } => value.value,
                mir::ConstRef::Named { .. } => {
                    return Err(LowerError::Unsupported(
                        "named constants are not supported yet",
                    ));
                }
            };
            let imm = u32::try_from(imm).map_err(|_| {
                LowerError::Unsupported("x86_64 lowering only supports u32 immediates for now")
            })?;
            items.push(asm::X64Item::MovImm {
                imm: asm_imm(imm as u64),
                op: Box::new(asm::MovKeyword {
                    prov: default_prov(),
                }),
                prov: default_prov(),
                rd: Box::new(dst),
            });
        }
        mir::InstOp::Copy(_) => {
            lower_copy_x64(items, inst)?;
        }
        mir::InstOp::BinOp {
            op: mir::BinOpKind::Add,
            ..
        } => {
            lower_binop_x64(items, inst, true)?;
        }
        mir::InstOp::BinOp {
            op: mir::BinOpKind::Sub,
            ..
        } => {
            lower_binop_x64(items, inst, false)?;
        }
        _ => {
            return Err(LowerError::Unsupported(
                "only const, copy, Add, and Sub instructions are supported yet",
            ));
        }
    }

    Ok(())
}

fn lower_copy_aarch64(items: &mut Vec<asm::A64Item>, inst: &mir::Inst) -> Result<(), LowerError> {
    let defs = require(inst.defs.as_ref(), "copy instruction missing defs")?;
    let uses = require(inst.uses.as_ref(), "copy instruction missing uses")?;
    if defs.len() != 1 || uses.len() != 1 {
        return Err(LowerError::Unsupported(
            "copy instructions must have one def and one use",
        ));
    }

    let dst = lower_aarch64_operand(&defs[0])?;
    let src = lower_aarch64_operand(&uses[0])?;
    if a64_same_reg(&dst, &src) {
        return Ok(());
    }

    items.push(asm::A64Item::Mov {
        op: Box::new(asm::MovKeyword {
            prov: default_prov(),
        }),
        prov: default_prov(),
        rd: Box::new(dst),
        rm: Box::new(src),
    });
    Ok(())
}

fn lower_copy_x64(items: &mut Vec<asm::X64Item>, inst: &mir::Inst) -> Result<(), LowerError> {
    let defs = require(inst.defs.as_ref(), "copy instruction missing defs")?;
    let uses = require(inst.uses.as_ref(), "copy instruction missing uses")?;
    if defs.len() != 1 || uses.len() != 1 {
        return Err(LowerError::Unsupported(
            "copy instructions must have one def and one use",
        ));
    }

    let dst = lower_x64_operand(&defs[0])?;
    let src = lower_x64_operand(&uses[0])?;
    if x64_same_reg(&dst, &src) {
        return Ok(());
    }

    items.push(asm::X64Item::MovReg {
        op: Box::new(asm::MovKeyword {
            prov: default_prov(),
        }),
        prov: default_prov(),
        rd: Box::new(dst),
        rm: Box::new(src),
    });
    Ok(())
}

fn lower_binop_aarch64(
    items: &mut Vec<asm::A64Item>,
    inst: &mir::Inst,
    is_add: bool,
) -> Result<(), LowerError> {
    let defs = require(inst.defs.as_ref(), "binop instruction missing defs")?;
    let uses = require(inst.uses.as_ref(), "binop instruction missing uses")?;
    if defs.len() != 1 || uses.len() != 2 {
        return Err(LowerError::Unsupported(
            "binary ops must have one def and two uses",
        ));
    }

    let dst = lower_aarch64_operand(&defs[0])?;
    let lhs = lower_aarch64_operand(&uses[0])?;
    let rhs = lower_aarch64_operand(&uses[1])?;

    if a64_same_reg(&dst, &lhs) {
        emit_a64_binop(items, is_add, dst, lhs, rhs);
        return Ok(());
    }

    if !a64_same_reg(&dst, &rhs) {
        items.push(asm::A64Item::Mov {
            op: Box::new(asm::MovKeyword {
                prov: default_prov(),
            }),
            prov: default_prov(),
            rd: Box::new(dst.clone()),
            rm: Box::new(lhs),
        });
        emit_a64_binop(items, is_add, dst.clone(), dst.clone(), rhs);
        return Ok(());
    }

    let scratch = a64_temp_reg(&[&dst, &lhs, &rhs])?;
    items.push(asm::A64Item::Mov {
        op: Box::new(asm::MovKeyword {
            prov: default_prov(),
        }),
        prov: default_prov(),
        rd: Box::new(scratch.clone()),
        rm: Box::new(lhs),
    });
    emit_a64_binop(items, is_add, dst, scratch, rhs);
    Ok(())
}

fn emit_a64_binop(
    items: &mut Vec<asm::A64Item>,
    is_add: bool,
    dst: asm::A64Reg,
    lhs: asm::A64Reg,
    rhs: asm::A64Reg,
) {
    let item = if is_add {
        asm::A64Item::AddReg {
            op: Box::new(asm::AddKeyword {
                prov: default_prov(),
            }),
            prov: default_prov(),
            rd: Box::new(dst),
            rn: Box::new(lhs),
            rm: Box::new(rhs),
        }
    } else {
        asm::A64Item::SubReg {
            op: Box::new(asm::SubKeyword {
                prov: default_prov(),
            }),
            prov: default_prov(),
            rd: Box::new(dst),
            rn: Box::new(lhs),
            rm: Box::new(rhs),
        }
    };
    items.push(item);
}

fn lower_binop_x64(
    items: &mut Vec<asm::X64Item>,
    inst: &mir::Inst,
    is_add: bool,
) -> Result<(), LowerError> {
    let defs = require(inst.defs.as_ref(), "binop instruction missing defs")?;
    let uses = require(inst.uses.as_ref(), "binop instruction missing uses")?;
    if defs.len() != 1 || uses.len() != 2 {
        return Err(LowerError::Unsupported(
            "binary ops must have one def and two uses",
        ));
    }

    let dst = lower_x64_operand(&defs[0])?;
    let lhs = lower_x64_operand(&uses[0])?;
    let rhs = lower_x64_operand(&uses[1])?;

    if x64_same_reg(&dst, &lhs) {
        emit_x64_binop(items, is_add, dst, rhs);
        return Ok(());
    }

    if !x64_same_reg(&dst, &rhs) {
        items.push(asm::X64Item::MovReg {
            op: Box::new(asm::MovKeyword {
                prov: default_prov(),
            }),
            prov: default_prov(),
            rd: Box::new(dst.clone()),
            rm: Box::new(lhs),
        });
        emit_x64_binop(items, is_add, dst, rhs);
        return Ok(());
    }

    let scratch = x64_temp_reg(&[&dst, &lhs, &rhs])?;
    items.push(asm::X64Item::MovReg {
        op: Box::new(asm::MovKeyword {
            prov: default_prov(),
        }),
        prov: default_prov(),
        rd: Box::new(scratch.clone()),
        rm: Box::new(lhs),
    });
    emit_x64_binop(items, is_add, scratch.clone(), rhs);
    items.push(asm::X64Item::MovReg {
        op: Box::new(asm::MovKeyword {
            prov: default_prov(),
        }),
        prov: default_prov(),
        rd: Box::new(dst),
        rm: Box::new(scratch),
    });
    Ok(())
}

fn emit_x64_binop(items: &mut Vec<asm::X64Item>, is_add: bool, dst: asm::X64Reg, rhs: asm::X64Reg) {
    let item = if is_add {
        asm::X64Item::AddReg {
            op: Box::new(asm::AddKeyword {
                prov: default_prov(),
            }),
            prov: default_prov(),
            rd: Box::new(dst),
            rm: Box::new(rhs),
        }
    } else {
        asm::X64Item::SubReg {
            op: Box::new(asm::SubKeyword {
                prov: default_prov(),
            }),
            prov: default_prov(),
            rd: Box::new(dst),
            rm: Box::new(rhs),
        }
    };
    items.push(item);
}

fn lower_aarch64_terminator(
    items: &mut Vec<asm::A64Item>,
    storage: &mir::FunctionStorage<'_>,
    block: &mir::Block,
) -> Result<(), LowerError> {
    let term = storage
        .block_term(block)
        .map_err(|e| LowerError::Invalid(e.to_string()))?;
    match term {
        mir::Terminator::Return { .. } => {
            items.push(asm::A64Item::Ret {
                op: Box::new(asm::RetKeyword {
                    prov: default_prov(),
                }),
                prov: default_prov(),
            });
        }
        mir::Terminator::Branch { edge, .. } => {
            let edge = require(
                storage.edge(*edge),
                format!("terminator references missing edge id {edge:?}"),
            )?;
            items.push(asm::A64Item::B {
                op: Box::new(asm::BKeyword {
                    prov: default_prov(),
                }),
                prov: default_prov(),
                target: label_name(edge.to),
            });
        }
        _ => {
            return Err(LowerError::Unsupported(
                "only return and unconditional branch terminators are supported yet",
            ));
        }
    }
    Ok(())
}

fn lower_x64_terminator(
    items: &mut Vec<asm::X64Item>,
    storage: &mir::FunctionStorage<'_>,
    block: &mir::Block,
) -> Result<(), LowerError> {
    let term = storage
        .block_term(block)
        .map_err(|e| LowerError::Invalid(e.to_string()))?;
    match term {
        mir::Terminator::Return { .. } => {
            items.push(asm::X64Item::Ret {
                op: Box::new(asm::RetKeyword {
                    prov: default_prov(),
                }),
                prov: default_prov(),
            });
        }
        mir::Terminator::Branch { edge, .. } => {
            let edge = require(
                storage.edge(*edge),
                format!("terminator references missing edge id {edge:?}"),
            )?;
            items.push(asm::X64Item::Jmp {
                op: Box::new(asm::JmpKeyword {
                    prov: default_prov(),
                }),
                prov: default_prov(),
                target: label_name(edge.to),
            });
        }
        _ => {
            return Err(LowerError::Unsupported(
                "only return and unconditional branch terminators are supported yet",
            ));
        }
    }
    Ok(())
}

fn lower_aarch64_operand(operand: &mir::Operand) -> Result<asm::A64Reg, LowerError> {
    if operand.class != mir::RegClass::Gpr {
        return Err(LowerError::Unsupported(
            "only gpr operands are supported yet",
        ));
    }
    let fixed = require(
        operand.fixed.as_deref(),
        "operand missing fixed-reg constraint",
    )?;
    match fixed {
        mir::FixedReg::AbiRet { index, .. } if index.value == 0 => Ok(asm::A64Reg::X0(asm::X0 {
            prov: default_prov(),
        })),
        mir::FixedReg::HwReg { index, .. } => match index.value {
            0 => Ok(asm::A64Reg::X0(asm::X0 {
                prov: default_prov(),
            })),
            1 => Ok(asm::A64Reg::X1(asm::X1 {
                prov: default_prov(),
            })),
            2 => Ok(asm::A64Reg::X2(asm::X2 {
                prov: default_prov(),
            })),
            3 => Ok(asm::A64Reg::X3(asm::X3 {
                prov: default_prov(),
            })),
            31 => Ok(asm::A64Reg::Sp(asm::Sp {
                prov: default_prov(),
            })),
            other => Err(LowerError::Unsupported(match other {
                4 => "aarch64 lowering does not yet support x4",
                5 => "aarch64 lowering does not yet support x5",
                6 => "aarch64 lowering does not yet support x6",
                7 => "aarch64 lowering does not yet support x7",
                _ => "aarch64 lowering only supports hw regs x0-x3 and sp for now",
            })),
        },
        _ => Err(LowerError::Unsupported(
            "only ABI return register 0 and hw regs are supported yet",
        )),
    }
}

fn lower_x64_operand(operand: &mir::Operand) -> Result<asm::X64Reg, LowerError> {
    if operand.class != mir::RegClass::Gpr {
        return Err(LowerError::Unsupported(
            "only gpr operands are supported yet",
        ));
    }
    let fixed = require(
        operand.fixed.as_deref(),
        "operand missing fixed-reg constraint",
    )?;
    match fixed {
        mir::FixedReg::AbiRet { index, .. } if index.value == 0 => Ok(asm::X64Reg::Rax(asm::Rax {
            prov: default_prov(),
        })),
        mir::FixedReg::HwReg { index, .. } => match index.value {
            0 => Ok(asm::X64Reg::Rax(asm::Rax {
                prov: default_prov(),
            })),
            1 => Ok(asm::X64Reg::Rcx(asm::Rcx {
                prov: default_prov(),
            })),
            2 => Ok(asm::X64Reg::Rdx(asm::Rdx {
                prov: default_prov(),
            })),
            3 => Ok(asm::X64Reg::Rbx(asm::Rbx {
                prov: default_prov(),
            })),
            4 => Ok(asm::X64Reg::Rsp(asm::Rsp {
                prov: default_prov(),
            })),
            5 => Ok(asm::X64Reg::Rbp(asm::Rbp {
                prov: default_prov(),
            })),
            _ => Err(LowerError::Unsupported(
                "x86_64 lowering only supports hw regs rax, rcx, rdx, rbx, rsp, and rbp for now",
            )),
        },
        _ => Err(LowerError::Unsupported(
            "only ABI return register 0 and hw regs are supported yet",
        )),
    }
}

fn a64_same_reg(lhs: &asm::A64Reg, rhs: &asm::A64Reg) -> bool {
    std::mem::discriminant(lhs) == std::mem::discriminant(rhs)
}

fn x64_same_reg(lhs: &asm::X64Reg, rhs: &asm::X64Reg) -> bool {
    std::mem::discriminant(lhs) == std::mem::discriminant(rhs)
}

fn a64_temp_reg(excluded: &[&asm::A64Reg]) -> Result<asm::A64Reg, LowerError> {
    for candidate in [
        asm::A64Reg::X0(asm::X0 {
            prov: default_prov(),
        }),
        asm::A64Reg::X1(asm::X1 {
            prov: default_prov(),
        }),
        asm::A64Reg::X2(asm::X2 {
            prov: default_prov(),
        }),
        asm::A64Reg::X3(asm::X3 {
            prov: default_prov(),
        }),
    ] {
        if excluded.iter().all(|reg| !a64_same_reg(&candidate, reg)) {
            return Ok(candidate);
        }
    }
    Err(LowerError::Unsupported(
        "no free temporary register available for aarch64 lowering",
    ))
}

fn x64_temp_reg(excluded: &[&asm::X64Reg]) -> Result<asm::X64Reg, LowerError> {
    for candidate in [
        asm::X64Reg::Rax(asm::Rax {
            prov: default_prov(),
        }),
        asm::X64Reg::Rcx(asm::Rcx {
            prov: default_prov(),
        }),
        asm::X64Reg::Rdx(asm::Rdx {
            prov: default_prov(),
        }),
    ] {
        if excluded.iter().all(|reg| !x64_same_reg(&candidate, reg)) {
            return Ok(candidate);
        }
    }
    Err(LowerError::Unsupported(
        "no free temporary register available for x86_64 lowering",
    ))
}
