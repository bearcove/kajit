//! RA-MIR textual parser for kajit's register allocator IR.
//!
//! Parses the text format produced by `RaProgram::Display` back into
//! an `RaProgram`.

use chumsky::prelude::*;

use kajit_ir::ErrorCode;
use kajit_ir::{IntrinsicFn, LambdaId, SlotId, VReg, Width};
use kajit_lir::{BinOpKind, LinearOp, UnaryOpKind};
use kajit_mir::{
    BlockId, FixedReg, OperandKind, RaBlock, RaClobbers, RaEdge, RaEdgeArg, RaFunction, RaInst,
    RaOperand, RaProgram, RaTerminator, RegClass,
};

type Extra<'src> = extra::Err<Rich<'src, char>>;

fn ws<'src>() -> impl Parser<'src, &'src str, (), Extra<'src>> + Clone {
    any()
        .filter(|c: &char| c.is_whitespace())
        .repeated()
        .ignored()
}

/// Horizontal whitespace only (spaces and tabs, not newlines).
fn hws<'src>() -> impl Parser<'src, &'src str, (), Extra<'src>> + Clone {
    any()
        .filter(|c: &char| *c == ' ' || *c == '\t')
        .repeated()
        .ignored()
}

fn uint32<'src>() -> impl Parser<'src, &'src str, u32, Extra<'src>> + Clone {
    text::int::<_, Extra<'_>>(10).map(|s: &str| s.parse::<u32>().unwrap())
}

fn uint64<'src>() -> impl Parser<'src, &'src str, u64, Extra<'src>> + Clone {
    let hex = just("0x")
        .ignore_then(text::int::<_, Extra<'_>>(16))
        .map(|s: &str| u64::from_str_radix(s, 16).unwrap());
    let dec = text::int::<_, Extra<'_>>(10).map(|s: &str| s.parse::<u64>().unwrap());
    hex.or(dec)
}

fn usize_p<'src>() -> impl Parser<'src, &'src str, usize, Extra<'src>> + Clone {
    let hex = just("0x")
        .ignore_then(text::int::<_, Extra<'_>>(16))
        .map(|s: &str| usize::from_str_radix(s, 16).unwrap());
    let dec = text::int::<_, Extra<'_>>(10).map(|s: &str| s.parse::<usize>().unwrap());
    hex.or(dec)
}

fn width<'src>() -> impl Parser<'src, &'src str, Width, Extra<'src>> + Clone {
    choice((
        just("W1").to(Width::W1),
        just("W2").to(Width::W2),
        just("W4").to(Width::W4),
        just("W8").to(Width::W8),
    ))
}

fn error_code<'src>() -> impl Parser<'src, &'src str, ErrorCode, Extra<'src>> + Clone {
    choice((
        just("UnexpectedEof").to(ErrorCode::UnexpectedEof),
        just("InvalidVarint").to(ErrorCode::InvalidVarint),
        just("InvalidUtf8").to(ErrorCode::InvalidUtf8),
        just("UnsupportedShape").to(ErrorCode::UnsupportedShape),
        just("ExpectedObjectStart").to(ErrorCode::ExpectedObjectStart),
        just("ExpectedColon").to(ErrorCode::ExpectedColon),
        just("ExpectedStringKey").to(ErrorCode::ExpectedStringKey),
        just("UnterminatedString").to(ErrorCode::UnterminatedString),
        just("InvalidJsonNumber").to(ErrorCode::InvalidJsonNumber),
        just("MissingRequiredField").to(ErrorCode::MissingRequiredField),
        just("UnexpectedCharacter").to(ErrorCode::UnexpectedCharacter),
        just("NumberOutOfRange").to(ErrorCode::NumberOutOfRange),
        just("InvalidBool").to(ErrorCode::InvalidBool),
        just("UnknownVariant").to(ErrorCode::UnknownVariant),
        just("ExpectedTagKey").to(ErrorCode::ExpectedTagKey),
        just("AmbiguousVariant").to(ErrorCode::AmbiguousVariant),
        just("AllocError").to(ErrorCode::AllocError),
        just("InvalidEscapeSequence").to(ErrorCode::InvalidEscapeSequence),
        just("UnknownField").to(ErrorCode::UnknownField),
    ))
}

fn vreg<'src>() -> impl Parser<'src, &'src str, VReg, Extra<'src>> + Clone {
    just("v").ignore_then(uint32()).map(VReg::new)
}

fn edge_arg<'src>() -> impl Parser<'src, &'src str, RaEdgeArg, Extra<'src>> + Clone {
    let mapped = vreg()
        .then_ignore(just("=>"))
        .then(vreg())
        .map(|(target, source)| RaEdgeArg { target, source });
    let identity = vreg().map(|v| RaEdgeArg {
        target: v,
        source: v,
    });
    mapped.or(identity)
}

fn block_id<'src>() -> impl Parser<'src, &'src str, BlockId, Extra<'src>> + Clone {
    just("b").ignore_then(uint32()).map(BlockId)
}

fn reg_class<'src>() -> impl Parser<'src, &'src str, RegClass, Extra<'src>> + Clone {
    choice((
        just("gpr").to(RegClass::Gpr),
        just("simd").to(RegClass::Simd),
    ))
}

fn fixed_reg<'src>() -> impl Parser<'src, &'src str, FixedReg, Extra<'src>> + Clone {
    let arg = just("/arg")
        .ignore_then(text::int::<_, Extra<'_>>(10).map(|s: &str| s.parse::<u8>().unwrap()))
        .map(FixedReg::AbiArg);
    let ret = just("/ret")
        .ignore_then(text::int::<_, Extra<'_>>(10).map(|s: &str| s.parse::<u8>().unwrap()))
        .map(FixedReg::AbiRet);
    let hw = just("/hw")
        .ignore_then(text::int::<_, Extra<'_>>(10).map(|s: &str| s.parse::<u8>().unwrap()))
        .map(FixedReg::HwReg);
    choice((arg, ret, hw))
}

/// Parse an operand: `v0:gpr`, `v1:simd/arg2`, `v2:gpr/ret0`
fn operand<'src>()
-> impl Parser<'src, &'src str, (VReg, RegClass, Option<FixedReg>), Extra<'src>> + Clone {
    vreg()
        .then_ignore(just(":"))
        .then(reg_class())
        .then(fixed_reg().or_not())
        .map(|((v, c), f)| (v, c, f))
}

/// Parse clobbers: `!gpr`, `!simd`, `!gpr,simd`
fn clobbers<'src>() -> impl Parser<'src, &'src str, RaClobbers, Extra<'src>> + Clone {
    just("!").ignore_then(choice((
        just("gpr,simd").to(RaClobbers {
            caller_saved_gpr: true,
            caller_saved_simd: true,
        }),
        just("gpr").to(RaClobbers {
            caller_saved_gpr: true,
            caller_saved_simd: false,
        }),
        just("simd").to(RaClobbers {
            caller_saved_gpr: false,
            caller_saved_simd: true,
        }),
    )))
}

/// Parse an instruction line.
/// Format: `[defs =] op_name [uses] [!clobbers]`
fn instruction<'src>() -> impl Parser<'src, &'src str, AstInst, Extra<'src>> + Clone {
    // Def operands before `=`, or none
    // Use hws() (horizontal whitespace) so instructions don't span across lines.
    let def_part = operand()
        .separated_by(just(",").padded_by(hws()))
        .at_least(1)
        .collect::<Vec<_>>()
        .then_ignore(hws().then(just("=")).then(hws()));

    let use_part = operand()
        .separated_by(just(",").padded_by(hws()))
        .at_least(1)
        .collect::<Vec<_>>();

    let clobber_part = hws().ignore_then(clobbers()).or_not();

    // With defs
    let with_defs = def_part
        .then(op_name())
        .then(hws().ignore_then(use_part.clone()).or_not())
        .then(clobber_part.clone())
        .map(|(((defs, op), uses), clob)| AstInst {
            defs,
            op,
            uses: uses.unwrap_or_default(),
            clobbers: clob.unwrap_or_default(),
        });

    // Without defs (void ops like bounds_check, store, etc.)
    let without_defs = op_name()
        .then(hws().ignore_then(use_part).or_not())
        .then(clobber_part)
        .map(|((op, uses), clob)| AstInst {
            defs: Vec::new(),
            op,
            uses: uses.unwrap_or_default(),
            clobbers: clob.unwrap_or_default(),
        });

    with_defs.or(without_defs)
}

#[derive(Debug, Clone)]
struct AstInst {
    defs: Vec<(VReg, RegClass, Option<FixedReg>)>,
    op: AstRaOp,
    uses: Vec<(VReg, RegClass, Option<FixedReg>)>,
    clobbers: RaClobbers,
}

/// Parsed operation name with params (to build LinearOp).
#[derive(Debug, Clone)]
enum AstRaOp {
    Const(u64),
    BinOp(BinOpKind),
    UnaryOp(UnaryOpKind),
    Copy,
    BoundsCheck(u32),
    ReadBytes(u32),
    PeekByte,
    Advance(u32),
    AdvanceBy,
    SaveCursor,
    RestoreCursor,
    Store(u32, Width),
    Load(u32, Width),
    SaveOutPtr,
    SetOutPtr,
    SlotAddr(u32),
    WriteSlot(u32),
    ReadSlot(u32),
    CallIntrinsic(usize, u32),
    CallPure(usize),
    CallLambda(u32),
    SimdStringScan,
    SimdWsSkip,
    ErrorExit(ErrorCode),
}

fn op_name<'src>() -> impl Parser<'src, &'src str, AstRaOp, Extra<'src>> + Clone {
    let parameterized = choice((
        just("const(")
            .ignore_then(uint64())
            .then_ignore(just(")"))
            .map(AstRaOp::Const),
        just("bounds_check(")
            .ignore_then(uint32())
            .then_ignore(just(")"))
            .map(AstRaOp::BoundsCheck),
        just("read_bytes(")
            .ignore_then(uint32())
            .then_ignore(just(")"))
            .map(AstRaOp::ReadBytes),
        just("advance(")
            .ignore_then(uint32())
            .then_ignore(just(")"))
            .map(AstRaOp::Advance),
        just("store([")
            .ignore_then(uint32())
            .then_ignore(just(":"))
            .then(width())
            .then_ignore(just("])"))
            .map(|(o, w)| AstRaOp::Store(o, w)),
        just("load([")
            .ignore_then(uint32())
            .then_ignore(just(":"))
            .then(width())
            .then_ignore(just("])"))
            .map(|(o, w)| AstRaOp::Load(o, w)),
        just("slot_addr(")
            .ignore_then(uint32())
            .then_ignore(just(")"))
            .map(AstRaOp::SlotAddr),
        just("write_slot(")
            .ignore_then(uint32())
            .then_ignore(just(")"))
            .map(AstRaOp::WriteSlot),
        just("read_slot(")
            .ignore_then(uint32())
            .then_ignore(just(")"))
            .map(AstRaOp::ReadSlot),
        just("call_intrinsic(")
            .ignore_then(usize_p())
            .then_ignore(just(",").then(ws()).then(just("fo=")))
            .then(uint32())
            .then_ignore(just(")"))
            .map(|(func, fo)| AstRaOp::CallIntrinsic(func, fo)),
    ));

    let parameterized2 = choice((
        just("call_pure(")
            .ignore_then(usize_p())
            .then_ignore(just(")"))
            .map(AstRaOp::CallPure),
        just("call_lambda(@")
            .ignore_then(uint32())
            .then_ignore(just(")"))
            .map(AstRaOp::CallLambda),
        just("error_exit(")
            .ignore_then(error_code())
            .then_ignore(just(")"))
            .map(AstRaOp::ErrorExit),
    ));

    let binops = choice((
        just("Add").to(AstRaOp::BinOp(BinOpKind::Add)),
        just("Sub").to(AstRaOp::BinOp(BinOpKind::Sub)),
        just("And").to(AstRaOp::BinOp(BinOpKind::And)),
        just("Or").to(AstRaOp::BinOp(BinOpKind::Or)),
        just("Shr").to(AstRaOp::BinOp(BinOpKind::Shr)),
        just("Shl").to(AstRaOp::BinOp(BinOpKind::Shl)),
        just("Xor").to(AstRaOp::BinOp(BinOpKind::Xor)),
        just("CmpNe").to(AstRaOp::BinOp(BinOpKind::CmpNe)),
    ));

    let unaryops = choice((
        just("ZigzagDecode { wide: true }")
            .to(AstRaOp::UnaryOp(UnaryOpKind::ZigzagDecode { wide: true })),
        just("ZigzagDecode { wide: false }")
            .to(AstRaOp::UnaryOp(UnaryOpKind::ZigzagDecode { wide: false })),
        just("SignExtend { from_width: ")
            .ignore_then(width())
            .then_ignore(just(" }"))
            .map(|w| AstRaOp::UnaryOp(UnaryOpKind::SignExtend { from_width: w })),
    ));

    let simple = choice((
        just("copy").to(AstRaOp::Copy),
        just("peek_byte").to(AstRaOp::PeekByte),
        just("advance_by").to(AstRaOp::AdvanceBy),
        just("save_cursor").to(AstRaOp::SaveCursor),
        just("restore_cursor").to(AstRaOp::RestoreCursor),
        just("save_out_ptr").to(AstRaOp::SaveOutPtr),
        just("set_out_ptr").to(AstRaOp::SetOutPtr),
        just("simd_string_scan").to(AstRaOp::SimdStringScan),
        just("simd_ws_skip").to(AstRaOp::SimdWsSkip),
    ));

    choice((parameterized, parameterized2, binops, unaryops, simple))
}

fn terminator<'src>() -> impl Parser<'src, &'src str, RaTerminator, Extra<'src>> + Clone {
    let ret = just("return").to(RaTerminator::Return);
    let error = just("error_exit(")
        .ignore_then(error_code())
        .then_ignore(just(")"))
        .map(|c| RaTerminator::ErrorExit { code: c });
    let branch = just("branch ")
        .ignore_then(block_id())
        .map(|b| RaTerminator::Branch { target: b });
    let branch_if = just("branch_if ")
        .ignore_then(vreg())
        .then_ignore(just(" -> "))
        .then(block_id())
        .then_ignore(just(", fallthrough "))
        .then(block_id())
        .map(|((cond, target), fallthrough)| RaTerminator::BranchIf {
            cond,
            target,
            fallthrough,
        });
    let branch_if_zero = just("branch_if_zero ")
        .ignore_then(vreg())
        .then_ignore(just(" -> "))
        .then(block_id())
        .then_ignore(just(", fallthrough "))
        .then(block_id())
        .map(|((cond, target), fallthrough)| RaTerminator::BranchIfZero {
            cond,
            target,
            fallthrough,
        });
    let jump_table = just("jump_table ")
        .ignore_then(vreg())
        .then_ignore(just(" ["))
        .then(block_id().separated_by(just(", ")).collect::<Vec<_>>())
        .then_ignore(just("], default "))
        .then(block_id())
        .map(|((predicate, targets), default)| RaTerminator::JumpTable {
            predicate,
            targets,
            default,
        });

    choice((branch_if_zero, branch_if, jump_table, error, branch, ret))
}

/// Parse successor edges: `succs: b1 [v0, v1] b2 [v2]` or `succs: (none)`
fn succs<'src>() -> impl Parser<'src, &'src str, Vec<RaEdge>, Extra<'src>> + Clone {
    let none = just("succs:")
        .then(ws())
        .then(just("(none)"))
        .to(Vec::new());
    let edge = block_id()
        .then(
            just(" [")
                .ignore_then(edge_arg().separated_by(just(", ")).collect::<Vec<_>>())
                .then_ignore(just("]"))
                .or_not(),
        )
        .map(|(to, args)| RaEdge {
            to,
            args: args.unwrap_or_default().into_iter().collect(),
        });
    let some = just("succs:").ignore_then(
        ws().ignore_then(edge)
            .repeated()
            .at_least(1)
            .collect::<Vec<_>>(),
    );
    none.or(some)
}

/// Parse block params: `[params: v0, v1]`
fn block_params<'src>() -> impl Parser<'src, &'src str, Vec<VReg>, Extra<'src>> + Clone {
    just("[params:")
        .ignore_then(ws())
        .ignore_then(
            vreg()
                .separated_by(just(",").then(ws()))
                .collect::<Vec<_>>(),
        )
        .then_ignore(just("]"))
}

/// Parse block preds: `(preds: b0, b1)`
fn block_preds<'src>() -> impl Parser<'src, &'src str, Vec<BlockId>, Extra<'src>> + Clone {
    just("(preds:")
        .ignore_then(ws())
        .ignore_then(
            block_id()
                .separated_by(just(",").then(ws()))
                .collect::<Vec<_>>(),
        )
        .then_ignore(just(")"))
}

fn block<'src>() -> impl Parser<'src, &'src str, AstBlock, Extra<'src>> + Clone {
    just("block")
        .padded_by(ws())
        .ignore_then(block_id())
        .then(ws().ignore_then(block_params()).or_not())
        .then(ws().ignore_then(block_preds()).or_not())
        .then_ignore(just(":").then(ws()))
        .then(instruction().padded_by(ws()).repeated().collect::<Vec<_>>())
        .then_ignore(just("term:").then(ws()))
        .then(terminator())
        .then_ignore(ws())
        .then(succs())
        .map(|(((((id, params), preds), insts), term), succs)| AstBlock {
            id,
            params: params.unwrap_or_default(),
            preds: preds.unwrap_or_default(),
            insts,
            term,
            succs,
        })
}

#[derive(Debug, Clone)]
struct AstBlock {
    id: BlockId,
    params: Vec<VReg>,
    preds: Vec<BlockId>,
    insts: Vec<AstInst>,
    term: RaTerminator,
    succs: Vec<RaEdge>,
}

fn ra_func<'src>() -> impl Parser<'src, &'src str, AstRaFunc, Extra<'src>> + Clone {
    just("ra_func")
        .padded_by(ws())
        .ignore_then(just("@"))
        .ignore_then(uint32())
        .then_ignore(ws().then(just("{")).then(ws()))
        .then(block().padded_by(ws()).repeated().collect::<Vec<_>>())
        .then_ignore(ws().then(just("}")))
        .map(|(lambda_id, blocks)| AstRaFunc { lambda_id, blocks })
}

#[derive(Debug, Clone)]
struct AstRaFunc {
    lambda_id: u32,
    blocks: Vec<AstBlock>,
}

fn ra_program<'src>() -> impl Parser<'src, &'src str, Vec<AstRaFunc>, Extra<'src>> + Clone {
    ra_func()
        .padded_by(ws())
        .repeated()
        .at_least(1)
        .collect::<Vec<_>>()
        .then_ignore(end())
}

// ─── Resolution ─────────────────────────────────────────────────────────────

#[derive(Debug)]
pub struct ParseError {
    pub message: String,
}

impl std::fmt::Display for ParseError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.message)
    }
}

impl std::error::Error for ParseError {}

/// Parse RA-MIR text format into an `RaProgram`.
pub fn parse_ra_mir(input: &str) -> Result<RaProgram, ParseError> {
    let stripped = strip_ra_mir_comments(input);
    let result = ra_program().parse(stripped.as_str());
    let funcs_ast = result.into_result().map_err(|errs| {
        let msgs: Vec<String> = errs.into_iter().map(|e| format!("{e}")).collect();
        ParseError {
            message: msgs.join("\n"),
        }
    })?;

    resolve(funcs_ast)
}

fn strip_ra_mir_comments(input: &str) -> String {
    let mut out = String::with_capacity(input.len());
    for line in input.lines() {
        if let Some((head, _)) = line.split_once(';') {
            out.push_str(head.trim_end());
        } else {
            out.push_str(line);
        }
        out.push('\n');
    }
    out
}

fn resolve(funcs_ast: Vec<AstRaFunc>) -> Result<RaProgram, ParseError> {
    let mut max_vreg: u32 = 0;
    let mut max_slot: u32 = 0;
    let mut funcs = Vec::new();

    for ast_func in funcs_ast {
        let mut blocks = Vec::new();
        // Assign contiguous linear op indices across all blocks within the
        // function so regalloc2 can place edge moves correctly.
        let mut next_linear_op_index: usize = 0;
        for ast_block in ast_func.blocks {
            let mut insts = Vec::new();
            for ast_inst in ast_block.insts.iter() {
                let (op, operands) = resolve_inst(ast_inst, &mut max_vreg, &mut max_slot);
                insts.push(RaInst {
                    linear_op_index: next_linear_op_index,
                    op,
                    operands,
                    clobbers: ast_inst.clobbers,
                });
                next_linear_op_index += 1;
            }

            // Track vregs in params.
            for p in &ast_block.params {
                max_vreg = max_vreg.max(p.index() as u32);
            }

            // The terminator gets the next index after the last instruction.
            let term_linear_op_index = Some(next_linear_op_index);
            next_linear_op_index += 1;

            blocks.push(RaBlock {
                id: ast_block.id,
                label: None,
                params: ast_block.params,
                insts,
                term_linear_op_index,
                term: ast_block.term,
                preds: ast_block.preds,
                succs: ast_block.succs,
            });
        }

        funcs.push(RaFunction {
            lambda_id: LambdaId::new(ast_func.lambda_id),
            entry: BlockId(0),
            data_args: Vec::new(),
            data_results: Vec::new(),
            blocks,
        });
    }

    Ok(RaProgram {
        funcs,
        vreg_count: max_vreg + 1,
        slot_count: max_slot + 1,
    })
}

fn resolve_inst(
    ast: &AstInst,
    max_vreg: &mut u32,
    max_slot: &mut u32,
) -> (LinearOp, Vec<RaOperand>) {
    let mut operands = Vec::new();

    // Build use operands.
    for (v, class, fixed) in &ast.uses {
        *max_vreg = (*max_vreg).max(v.index() as u32);
        operands.push(RaOperand {
            vreg: *v,
            kind: OperandKind::Use,
            class: *class,
            fixed: *fixed,
        });
    }

    // Build def operands.
    for (v, class, fixed) in &ast.defs {
        *max_vreg = (*max_vreg).max(v.index() as u32);
        operands.push(RaOperand {
            vreg: *v,
            kind: OperandKind::Def,
            class: *class,
            fixed: *fixed,
        });
    }

    // Build the LinearOp from the parsed op + operand info.
    let dst = ast.defs.first().map(|(v, _, _)| *v);
    let src = ast.uses.first().map(|(v, _, _)| *v);

    let op = match &ast.op {
        AstRaOp::Const(value) => LinearOp::Const {
            dst: dst.unwrap(),
            value: *value,
        },
        AstRaOp::BinOp(kind) => {
            let lhs = ast.uses[0].0;
            let rhs = ast.uses[1].0;
            LinearOp::BinOp {
                op: *kind,
                dst: dst.unwrap(),
                lhs,
                rhs,
            }
        }
        AstRaOp::UnaryOp(kind) => LinearOp::UnaryOp {
            op: *kind,
            dst: dst.unwrap(),
            src: src.unwrap(),
        },
        AstRaOp::Copy => LinearOp::Copy {
            dst: dst.unwrap(),
            src: src.unwrap(),
        },
        AstRaOp::BoundsCheck(count) => LinearOp::BoundsCheck { count: *count },
        AstRaOp::ReadBytes(count) => LinearOp::ReadBytes {
            dst: dst.unwrap(),
            count: *count,
        },
        AstRaOp::PeekByte => LinearOp::PeekByte { dst: dst.unwrap() },
        AstRaOp::Advance(count) => LinearOp::AdvanceCursor { count: *count },
        AstRaOp::AdvanceBy => LinearOp::AdvanceCursorBy { src: src.unwrap() },
        AstRaOp::SaveCursor => LinearOp::SaveCursor { dst: dst.unwrap() },
        AstRaOp::RestoreCursor => LinearOp::RestoreCursor { src: src.unwrap() },
        AstRaOp::Store(offset, width) => LinearOp::WriteToField {
            src: src.unwrap(),
            offset: *offset,
            width: *width,
        },
        AstRaOp::Load(offset, width) => LinearOp::ReadFromField {
            dst: dst.unwrap(),
            offset: *offset,
            width: *width,
        },
        AstRaOp::SaveOutPtr => LinearOp::SaveOutPtr { dst: dst.unwrap() },
        AstRaOp::SetOutPtr => LinearOp::SetOutPtr { src: src.unwrap() },
        AstRaOp::SlotAddr(slot) => {
            *max_slot = (*max_slot).max(*slot);
            LinearOp::SlotAddr {
                dst: dst.unwrap(),
                slot: SlotId::new(*slot),
            }
        }
        AstRaOp::WriteSlot(slot) => {
            *max_slot = (*max_slot).max(*slot);
            LinearOp::WriteToSlot {
                slot: SlotId::new(*slot),
                src: src.unwrap(),
            }
        }
        AstRaOp::ReadSlot(slot) => {
            *max_slot = (*max_slot).max(*slot);
            LinearOp::ReadFromSlot {
                dst: dst.unwrap(),
                slot: SlotId::new(*slot),
            }
        }
        AstRaOp::CallIntrinsic(func, fo) => {
            let args: Vec<VReg> = ast.uses.iter().map(|(v, _, _)| *v).collect();
            LinearOp::CallIntrinsic {
                func: IntrinsicFn(*func),
                args,
                dst,
                field_offset: *fo,
            }
        }
        AstRaOp::CallPure(func) => {
            let args: Vec<VReg> = ast.uses.iter().map(|(v, _, _)| *v).collect();
            LinearOp::CallPure {
                func: IntrinsicFn(*func),
                args,
                dst: dst.unwrap(),
            }
        }
        AstRaOp::CallLambda(target) => {
            let args: Vec<VReg> = ast.uses.iter().map(|(v, _, _)| *v).collect();
            let results: Vec<VReg> = ast.defs.iter().map(|(v, _, _)| *v).collect();
            LinearOp::CallLambda {
                target: LambdaId::new(*target),
                args,
                results,
            }
        }
        AstRaOp::SimdStringScan => {
            let pos = ast.defs[0].0;
            let kind = ast.defs[1].0;
            LinearOp::SimdStringScan { pos, kind }
        }
        AstRaOp::SimdWsSkip => LinearOp::SimdWhitespaceSkip,
        AstRaOp::ErrorExit(code) => LinearOp::ErrorExit { code: *code },
    };

    (op, operands)
}

// ─── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use kajit_ir::{IrBuilder, Width};
    use kajit_lir::linearize;
    use kajit_mir::lower_linear_ir;

    fn test_shape() -> &'static facet::Shape {
        <u8 as facet::Facet>::SHAPE
    }

    #[test]
    fn parse_simple_ra_mir() {
        let input = r#"
ra_func @0 {
  block b0:
    v0:gpr = const(0x2a)
    store([0:W4]) v0:gpr
    term: return
    succs: (none)
}
"#;

        let prog = parse_ra_mir(input).unwrap();
        assert_eq!(prog.funcs.len(), 1);
        assert_eq!(prog.funcs[0].blocks.len(), 1);
        assert_eq!(prog.funcs[0].blocks[0].insts.len(), 2);
    }

    #[test]
    fn round_trip_ra_mir() {
        // Build IR, linearize, lower to RA-MIR, display, parse, display again.
        let mut builder = IrBuilder::new(test_shape());
        {
            let mut rb = builder.root_region();
            rb.bounds_check(4);
            let val = rb.read_bytes(4);
            rb.write_to_field(val, 0, Width::W4);
            rb.set_results(&[]);
        }
        let mut func = builder.finish();
        let lin = linearize(&mut func);
        let ra = lower_linear_ir(&lin);

        let text1 = format!("{ra}");
        let ra2 = parse_ra_mir(&text1).unwrap();
        let text2 = format!("{ra2}");

        assert_eq!(
            text1, text2,
            "round trip failed:\n--- original ---\n{text1}\n--- reparsed ---\n{text2}"
        );
    }

    #[test]
    fn round_trip_ra_mir_with_branches() {
        let mut builder = IrBuilder::new(test_shape());
        {
            let mut rb = builder.root_region();
            let pred = rb.const_val(0);
            let out = rb.gamma(pred, &[], 2, |branch_idx, bb| {
                let val = if branch_idx == 0 {
                    bb.const_val(42)
                } else {
                    bb.const_val(99)
                };
                bb.set_results(&[val]);
            });
            rb.write_to_field(out[0], 0, Width::W4);
            rb.set_results(&[]);
        }
        let mut func = builder.finish();
        let lin = linearize(&mut func);
        let ra = lower_linear_ir(&lin);

        let text1 = format!("{ra}");
        let ra2 = parse_ra_mir(&text1).unwrap();
        let text2 = format!("{ra2}");

        assert_eq!(
            text1, text2,
            "round trip failed:\n--- original ---\n{text1}\n--- reparsed ---\n{text2}"
        );
    }

    #[test]
    fn round_trip_human_ra_mir_dump() {
        let mut builder = IrBuilder::new(test_shape());
        {
            let mut rb = builder.root_region();
            let pred = rb.const_val(0);
            let out = rb.gamma(pred, &[], 2, |branch_idx, bb| {
                let val = if branch_idx == 0 {
                    bb.const_val(42)
                } else {
                    bb.const_val(99)
                };
                bb.set_results(&[val]);
            });
            rb.write_to_field(out[0], 0, Width::W4);
            rb.set_results(&[]);
        }
        let mut func = builder.finish();
        let lin = linearize(&mut func);
        let ra = lower_linear_ir(&lin);

        let text1 = format!("{ra:#}");
        let ra2 = parse_ra_mir(&text1).unwrap();
        let text2 = format!("{ra2:#}");

        assert_eq!(
            text1, text2,
            "round trip failed:\n--- original ---\n{text1}\n--- reparsed ---\n{text2}"
        );
    }

    #[test]
    fn parse_ra_mir_with_semicolon_comments() {
        let input = r#"
ra_func @0 { ; entry: b0
  block b0: ; no preds
    v0:gpr = const(0x2a) ; a constant
    term: return ; done
    succs: (none) ; leaf
}
"#;

        let prog = parse_ra_mir(input).unwrap();
        assert_eq!(prog.funcs.len(), 1);
        assert_eq!(prog.funcs[0].blocks.len(), 1);
        assert_eq!(prog.funcs[0].blocks[0].insts.len(), 1);
    }

    #[test]
    fn parse_error_malformed_ra_mir() {
        let input = "not valid ra-mir";
        let result = parse_ra_mir(input);
        assert!(result.is_err());
    }
}
