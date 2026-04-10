//! MIR lowering entrypoints.
//!
//! This will take schema-owned MIR (`kajit_reprs::mir`) plus its `Graph` storage
//! and produce schema-owned ASM (`kajit_reprs::asm`) or another backend-friendly
//! representation.

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

/// Minimal MIR→ASM lowering:
/// - Host-arch only
/// - Single function
/// - Single entry block with `return`
/// - Supports: `const(<imm>)` into `v*:gpr/ret0`
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
    let entry = func_storage
        .entry_block()
        .map_err(|e| LowerError::Invalid(e.to_string()))?;
    let term = func_storage
        .block_term(entry)
        .map_err(|e| LowerError::Invalid(e.to_string()))?;
    if !matches!(term, mir::Terminator::Return { .. }) {
        return Err(LowerError::Unsupported(
            "only entry blocks with `return` are supported yet",
        ));
    }

    // Find a constant assigned into the ABI return register 0.
    let mut ret0_imm: Option<u64> = None;
    for inst_id in entry.insts.iter() {
        let inst = require(
            mir_graph.inst(*inst_id),
            format!("block references missing inst id {inst_id:?}"),
        )?;
        let mir::InstOp::Const { value, .. } = inst.op.as_ref() else {
            return Err(LowerError::Unsupported(
                "only `const(...)` instructions are supported yet",
            ));
        };
        let defs = require(inst.defs.as_ref(), "const instruction missing defs")?;
        if defs.len() != 1 {
            return Err(LowerError::Unsupported(
                "const instructions with multiple defs are not supported yet",
            ));
        }
        let def = &defs[0];
        if !matches!(def.class, mir::RegClass::Gpr) {
            return Err(LowerError::Unsupported(
                "only gpr return values are supported yet",
            ));
        }
        let fixed = require(
            def.fixed.as_deref(),
            "return def missing fixed-reg constraint",
        )?;
        let mir::FixedReg::AbiRet { index, .. } = fixed else {
            return Err(LowerError::Unsupported(
                "return def must be constrained to an ABI return register",
            ));
        };
        if index.value != 0 {
            return Err(LowerError::Unsupported(
                "only ABI return register 0 is supported yet",
            ));
        }

        let imm = match value.as_ref() {
            mir::ConstRef::Value { value, .. } => value.value,
            mir::ConstRef::Named { .. } => {
                return Err(LowerError::Unsupported(
                    "named constants are not supported yet",
                ));
            }
        };
        ret0_imm = Some(imm);
    }

    let imm = require(ret0_imm, "entry block does not define ret0")?;

    let source = match arch {
        TargetArch::AArch64 => {
            let imm: u16 = imm.try_into().map_err(|_| {
                LowerError::Unsupported("aarch64 lowering only supports u16 immediates for now")
            })?;
            format!("asm aarch64 {{\n    entry:\n    movz x0, {imm}\n    ret\n}}")
        }
        TargetArch::X86_64 => {
            let imm: u32 = imm.try_into().map_err(|_| {
                LowerError::Unsupported("x86_64 lowering only supports u32 immediates for now")
            })?;
            format!("asm x86_64 {{\n    entry:\n    mov rax, {imm}\n    ret\n}}")
        }
    };

    let program = asm::parse_root_text(&source).map_err(LowerError::Invalid)?;
    Ok(LoweredAsm { source, program })
}
