//! CFG-MIR textual parser.
//!
//! Parses the canonical text format produced by `cfg_mir::Program::Display`.

use chumsky::prelude::*;

use kajit_ir::{ErrorCode, IntrinsicFn, IntrinsicRegistry, LambdaId, SlotId, VReg, Width};
use kajit_lir::{BinOpKind, LinearOp, UnaryOpKind};
use kajit_mir::cfg_mir::{
    Block, BlockId, Clobbers, Edge, EdgeArg, EdgeId, FixedReg, Function, FunctionId, Inst, InstId,
    Operand, OperandKind, Program, RegClass, TermId, Terminator,
};

type Extra<'src> = extra::Err<Rich<'src, char>>;

fn ws<'src>() -> impl Parser<'src, &'src str, (), Extra<'src>> + Clone {
    any()
        .filter(|c: &char| c.is_whitespace())
        .repeated()
        .ignored()
}

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

#[derive(Debug, Clone)]
enum ConstRef {
    Named(String),
    Value(u64),
}

fn usize_p<'src>() -> impl Parser<'src, &'src str, usize, Extra<'src>> + Clone {
    let hex = just("0x")
        .ignore_then(text::int::<_, Extra<'_>>(16))
        .map(|s: &str| usize::from_str_radix(s, 16).unwrap());
    let dec = text::int::<_, Extra<'_>>(10).map(|s: &str| s.parse::<usize>().unwrap());
    hex.or(dec)
}

fn ident<'src>() -> impl Parser<'src, &'src str, String, Extra<'src>> + Clone {
    any()
        .filter(|c: &char| c.is_ascii_alphanumeric() || matches!(c, '_' | ':' | '.'))
        .repeated()
        .at_least(1)
        .collect::<String>()
}

#[derive(Debug, Clone)]
enum IntrinsicRef {
    Named(String),
    Address(usize),
}

fn intrinsic_ref<'src>() -> impl Parser<'src, &'src str, IntrinsicRef, Extra<'src>> + Clone {
    just("@")
        .ignore_then(ident())
        .map(IntrinsicRef::Named)
        .or(usize_p().map(IntrinsicRef::Address))
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

fn operand<'src>()
-> impl Parser<'src, &'src str, (VReg, RegClass, Option<FixedReg>), Extra<'src>> + Clone {
    vreg()
        .then_ignore(just(":"))
        .then(reg_class())
        .then(fixed_reg().or_not())
        .map(|((v, c), f)| (v, c, f))
}

fn clobbers<'src>() -> impl Parser<'src, &'src str, Clobbers, Extra<'src>> + Clone {
    just("!").ignore_then(choice((
        just("gpr,simd").to(Clobbers {
            caller_saved_gpr: true,
            caller_saved_simd: true,
        }),
        just("gpr").to(Clobbers {
            caller_saved_gpr: true,
            caller_saved_simd: false,
        }),
        just("simd").to(Clobbers {
            caller_saved_gpr: false,
            caller_saved_simd: true,
        }),
    )))
}

#[derive(Debug, Clone)]
struct AstInstBody {
    defs: Vec<(VReg, RegClass, Option<FixedReg>)>,
    op: AstOp,
    uses: Vec<(VReg, RegClass, Option<FixedReg>)>,
    clobbers: Clobbers,
}

#[derive(Debug, Clone)]
enum AstOp {
    Const(ConstRef),
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
    CallIntrinsic(IntrinsicRef, u32),
    CallPure(IntrinsicRef),
    CallLambda(u32),
    SimdStringScan,
    SimdWsSkip,
    ErrorExit(ErrorCode),
}

fn op_name<'src>() -> impl Parser<'src, &'src str, AstOp, Extra<'src>> + Clone {
    let parameterized = choice((
        just("const(")
            .ignore_then(
                just("@")
                    .ignore_then(ident())
                    .map(ConstRef::Named)
                    .or(uint64().map(ConstRef::Value)),
            )
            .then_ignore(just(")"))
            .map(AstOp::Const),
        just("bounds_check(")
            .ignore_then(uint32())
            .then_ignore(just(")"))
            .map(AstOp::BoundsCheck),
        just("read_bytes(")
            .ignore_then(uint32())
            .then_ignore(just(")"))
            .map(AstOp::ReadBytes),
        just("advance(")
            .ignore_then(uint32())
            .then_ignore(just(")"))
            .map(AstOp::Advance),
        just("store([")
            .ignore_then(uint32())
            .then_ignore(just(":"))
            .then(width())
            .then_ignore(just("])"))
            .map(|(o, w)| AstOp::Store(o, w)),
        just("load([")
            .ignore_then(uint32())
            .then_ignore(just(":"))
            .then(width())
            .then_ignore(just("])"))
            .map(|(o, w)| AstOp::Load(o, w)),
        just("slot_addr(")
            .ignore_then(uint32())
            .then_ignore(just(")"))
            .map(AstOp::SlotAddr),
        just("write_slot(")
            .ignore_then(uint32())
            .then_ignore(just(")"))
            .map(AstOp::WriteSlot),
        just("read_slot(")
            .ignore_then(uint32())
            .then_ignore(just(")"))
            .map(AstOp::ReadSlot),
        just("call_intrinsic(")
            .ignore_then(intrinsic_ref())
            .then_ignore(just(",").then(ws()).then(just("fo=")))
            .then(uint32())
            .then_ignore(just(")"))
            .map(|(func, fo)| AstOp::CallIntrinsic(func, fo)),
    ));

    let parameterized2 = choice((
        just("call_pure(")
            .ignore_then(intrinsic_ref())
            .then_ignore(just(")"))
            .map(AstOp::CallPure),
        just("call_lambda(@")
            .ignore_then(uint32())
            .then_ignore(just(")"))
            .map(AstOp::CallLambda),
        just("error_exit(")
            .ignore_then(error_code())
            .then_ignore(just(")"))
            .map(AstOp::ErrorExit),
    ));

    let binops = choice((
        just("Add").to(AstOp::BinOp(BinOpKind::Add)),
        just("Sub").to(AstOp::BinOp(BinOpKind::Sub)),
        just("And").to(AstOp::BinOp(BinOpKind::And)),
        just("Or").to(AstOp::BinOp(BinOpKind::Or)),
        just("Shr").to(AstOp::BinOp(BinOpKind::Shr)),
        just("Shl").to(AstOp::BinOp(BinOpKind::Shl)),
        just("Xor").to(AstOp::BinOp(BinOpKind::Xor)),
        just("CmpNe").to(AstOp::BinOp(BinOpKind::CmpNe)),
    ));

    let unaryops = choice((
        just("ZigzagDecode { wide: true }")
            .to(AstOp::UnaryOp(UnaryOpKind::ZigzagDecode { wide: true })),
        just("ZigzagDecode { wide: false }")
            .to(AstOp::UnaryOp(UnaryOpKind::ZigzagDecode { wide: false })),
        just("SignExtend { from_width: ")
            .ignore_then(width())
            .then_ignore(just(" }"))
            .map(|w| AstOp::UnaryOp(UnaryOpKind::SignExtend { from_width: w })),
    ));

    let simple = choice((
        just("copy").to(AstOp::Copy),
        just("peek_byte").to(AstOp::PeekByte),
        just("advance_by").to(AstOp::AdvanceBy),
        just("save_cursor").to(AstOp::SaveCursor),
        just("restore_cursor").to(AstOp::RestoreCursor),
        just("save_out_ptr").to(AstOp::SaveOutPtr),
        just("set_out_ptr").to(AstOp::SetOutPtr),
        just("simd_string_scan").to(AstOp::SimdStringScan),
        just("simd_ws_skip").to(AstOp::SimdWsSkip),
    ));

    choice((parameterized, parameterized2, binops, unaryops, simple))
}

fn inst_body<'src>() -> impl Parser<'src, &'src str, AstInstBody, Extra<'src>> + Clone {
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

    let with_defs = def_part
        .then(op_name())
        .then(hws().ignore_then(use_part.clone()).or_not())
        .then(clobber_part.clone())
        .map(|(((defs, op), uses), clob)| AstInstBody {
            defs,
            op,
            uses: uses.unwrap_or_default(),
            clobbers: clob.unwrap_or_default(),
        });

    let without_defs = op_name()
        .then(hws().ignore_then(use_part).or_not())
        .then(clobber_part)
        .map(|((op, uses), clob)| AstInstBody {
            defs: Vec::new(),
            op,
            uses: uses.unwrap_or_default(),
            clobbers: clob.unwrap_or_default(),
        });

    with_defs.or(without_defs)
}

fn edge_arg<'src>() -> impl Parser<'src, &'src str, EdgeArg, Extra<'src>> + Clone {
    let mapped = vreg()
        .then_ignore(just("=>"))
        .then(vreg())
        .map(|(target, source)| EdgeArg { target, source });
    let identity = vreg().map(|v| EdgeArg {
        target: v,
        source: v,
    });
    mapped.or(identity)
}

fn vreg_list<'src>() -> impl Parser<'src, &'src str, Vec<VReg>, Extra<'src>> + Clone {
    just("[")
        .ignore_then(
            vreg()
                .separated_by(just(",").padded_by(hws()))
                .collect::<Vec<_>>(),
        )
        .then_ignore(just("]"))
}

fn inst_id<'src>() -> impl Parser<'src, &'src str, InstId, Extra<'src>> + Clone {
    just("i").ignore_then(uint32()).map(InstId::new)
}

fn inst_id_list<'src>() -> impl Parser<'src, &'src str, Vec<InstId>, Extra<'src>> + Clone {
    just("[")
        .ignore_then(
            inst_id()
                .separated_by(just(",").padded_by(hws()))
                .collect::<Vec<_>>(),
        )
        .then_ignore(just("]"))
}

fn term_id<'src>() -> impl Parser<'src, &'src str, TermId, Extra<'src>> + Clone {
    just("t").ignore_then(uint32()).map(TermId::new)
}

fn edge_id<'src>() -> impl Parser<'src, &'src str, EdgeId, Extra<'src>> + Clone {
    just("e").ignore_then(uint32()).map(EdgeId::new)
}

fn edge_id_list<'src>() -> impl Parser<'src, &'src str, Vec<EdgeId>, Extra<'src>> + Clone {
    just("[")
        .ignore_then(
            edge_id()
                .separated_by(just(",").padded_by(hws()))
                .collect::<Vec<_>>(),
        )
        .then_ignore(just("]"))
}

fn block_id<'src>() -> impl Parser<'src, &'src str, BlockId, Extra<'src>> + Clone {
    just("b").ignore_then(uint32()).map(BlockId::new)
}

fn edge_arg_list<'src>() -> impl Parser<'src, &'src str, Vec<EdgeArg>, Extra<'src>> + Clone {
    just("[")
        .ignore_then(
            edge_arg()
                .separated_by(just(",").padded_by(hws()))
                .collect::<Vec<_>>(),
        )
        .then_ignore(just("]"))
}

fn terminator<'src>() -> impl Parser<'src, &'src str, Terminator, Extra<'src>> + Clone {
    let ret = just("return").to(Terminator::Return);
    let error = just("error_exit(")
        .ignore_then(error_code())
        .then_ignore(just(")"))
        .map(|code| Terminator::ErrorExit { code });
    let branch = just("branch ")
        .ignore_then(edge_id())
        .map(|edge| Terminator::Branch { edge });
    let branch_if = just("branch_if ")
        .ignore_then(vreg())
        .then_ignore(just(" -> "))
        .then(edge_id())
        .then_ignore(just(", fallthrough "))
        .then(edge_id())
        .map(|((cond, taken), fallthrough)| Terminator::BranchIf {
            cond,
            taken,
            fallthrough,
        });
    let branch_if_zero = just("branch_if_zero ")
        .ignore_then(vreg())
        .then_ignore(just(" -> "))
        .then(edge_id())
        .then_ignore(just(", fallthrough "))
        .then(edge_id())
        .map(|((cond, taken), fallthrough)| Terminator::BranchIfZero {
            cond,
            taken,
            fallthrough,
        });
    let jump_table = just("jump_table ")
        .ignore_then(vreg())
        .then_ignore(just(" ["))
        .then(edge_id().separated_by(just(", ")).collect::<Vec<_>>())
        .then_ignore(just("], default "))
        .then(edge_id())
        .map(|((predicate, targets), default)| Terminator::JumpTable {
            predicate,
            targets,
            default,
        });
    choice((branch_if_zero, branch_if, jump_table, error, branch, ret))
}

#[derive(Debug, Clone)]
struct AstBlock {
    id: BlockId,
    params: Vec<VReg>,
    insts: Vec<InstId>,
    term: TermId,
    preds: Vec<EdgeId>,
    succs: Vec<EdgeId>,
}

#[derive(Debug, Clone)]
struct AstInst {
    id: InstId,
    body: AstInstBody,
}

#[derive(Debug, Clone)]
struct AstTerm {
    id: TermId,
    term: Terminator,
}

#[derive(Debug, Clone)]
struct AstEdge {
    id: EdgeId,
    from: BlockId,
    to: BlockId,
    args: Vec<EdgeArg>,
}

#[derive(Debug, Clone)]
enum AstFuncItem {
    DataArgs(Vec<VReg>),
    DataResults(Vec<VReg>),
    Block(AstBlock),
    Inst(AstInst),
    Term(AstTerm),
    Edge(AstEdge),
}

fn block_line<'src>() -> impl Parser<'src, &'src str, AstFuncItem, Extra<'src>> + Clone {
    just("block ")
        .ignore_then(block_id())
        .then_ignore(just(" params="))
        .then(vreg_list())
        .then_ignore(just(" insts="))
        .then(inst_id_list())
        .then_ignore(just(" term="))
        .then(term_id())
        .then_ignore(just(" preds="))
        .then(edge_id_list())
        .then_ignore(just(" succs="))
        .then(edge_id_list())
        .map(|(((((id, params), insts), term), preds), succs)| {
            AstFuncItem::Block(AstBlock {
                id,
                params,
                insts,
                term,
                preds,
                succs,
            })
        })
}

fn inst_line<'src>() -> impl Parser<'src, &'src str, AstFuncItem, Extra<'src>> + Clone {
    just("inst ")
        .ignore_then(inst_id())
        .then_ignore(just(": "))
        .then(inst_body())
        .map(|(id, body)| AstFuncItem::Inst(AstInst { id, body }))
}

fn term_line<'src>() -> impl Parser<'src, &'src str, AstFuncItem, Extra<'src>> + Clone {
    just("term ")
        .ignore_then(term_id())
        .then_ignore(just(": "))
        .then(terminator())
        .map(|(id, term)| AstFuncItem::Term(AstTerm { id, term }))
}

fn edge_line<'src>() -> impl Parser<'src, &'src str, AstFuncItem, Extra<'src>> + Clone {
    just("edge ")
        .ignore_then(edge_id())
        .then_ignore(just(": "))
        .then(block_id())
        .then_ignore(just(" -> "))
        .then(block_id())
        .then_ignore(just(" "))
        .then(edge_arg_list())
        .map(|(((id, from), to), args)| AstFuncItem::Edge(AstEdge { id, from, to, args }))
}

fn data_args_line<'src>() -> impl Parser<'src, &'src str, AstFuncItem, Extra<'src>> + Clone {
    just("data_args: ")
        .ignore_then(vreg_list())
        .map(AstFuncItem::DataArgs)
}

fn data_results_line<'src>() -> impl Parser<'src, &'src str, AstFuncItem, Extra<'src>> + Clone {
    just("data_results: ")
        .ignore_then(vreg_list())
        .map(AstFuncItem::DataResults)
}

#[derive(Debug, Clone)]
struct AstFunc {
    lambda_id: u32,
    function_id: u32,
    entry: BlockId,
    items: Vec<AstFuncItem>,
}

fn cfg_func<'src>() -> impl Parser<'src, &'src str, AstFunc, Extra<'src>> + Clone {
    just("cfg_func @")
        .ignore_then(uint32())
        .then_ignore(just(" f"))
        .then(uint32())
        .then_ignore(just(" entry="))
        .then(block_id())
        .then_ignore(ws().then(just("{")).then(ws()))
        .then(
            choice((
                data_args_line(),
                data_results_line(),
                block_line(),
                inst_line(),
                term_line(),
                edge_line(),
            ))
            .padded_by(ws())
            .repeated()
            .collect::<Vec<_>>(),
        )
        .then_ignore(ws().then(just("}")))
        .map(|(((lambda_id, function_id), entry), items)| AstFunc {
            lambda_id,
            function_id,
            entry,
            items,
        })
}

#[derive(Debug, Clone)]
struct AstProgram {
    vreg_count: u32,
    slot_count: u32,
    funcs: Vec<AstFunc>,
}

fn cfg_program<'src>() -> impl Parser<'src, &'src str, AstProgram, Extra<'src>> + Clone {
    just("cfg_program vregs=")
        .ignore_then(uint32())
        .then_ignore(just(" slots="))
        .then(uint32())
        .then_ignore(ws().then(just("{")).then(ws()))
        .then(cfg_func().padded_by(ws()).repeated().collect::<Vec<_>>())
        .then_ignore(ws().then(just("}")).then(ws()).then(end()))
        .map(|((vreg_count, slot_count), funcs)| AstProgram {
            vreg_count,
            slot_count,
            funcs,
        })
}

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

pub fn parse_cfg_mir(input: &str) -> Result<Program, ParseError> {
    parse_cfg_mir_with_registry(input, &IntrinsicRegistry::empty())
}

pub fn parse_cfg_mir_with_registry(
    input: &str,
    registry: &IntrinsicRegistry,
) -> Result<Program, ParseError> {
    let stripped = strip_comments(input);
    let parsed = cfg_program().padded_by(ws()).parse(stripped.as_str());
    let ast = parsed.into_result().map_err(|errs| {
        let msgs = errs.into_iter().map(|e| format!("{e}")).collect::<Vec<_>>();
        ParseError {
            message: msgs.join("\n"),
        }
    })?;
    resolve_program(ast, registry)
}

fn strip_comments(input: &str) -> String {
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

fn resolve_program(ast: AstProgram, registry: &IntrinsicRegistry) -> Result<Program, ParseError> {
    let funcs = ast
        .funcs
        .into_iter()
        .map(|func| resolve_function(func, registry))
        .collect::<Result<Vec<_>, _>>()?;
    let program = Program {
        funcs,
        vreg_count: ast.vreg_count,
        slot_count: ast.slot_count,
        debug: Default::default(),
    };
    program.validate().map_err(|err| ParseError {
        message: err.to_string(),
    })?;
    Ok(program)
}

fn resolve_function(ast: AstFunc, registry: &IntrinsicRegistry) -> Result<Function, ParseError> {
    let mut data_args: Option<Vec<VReg>> = None;
    let mut data_results: Option<Vec<VReg>> = None;
    let mut blocks = Vec::<AstBlock>::new();
    let mut insts = Vec::<AstInst>::new();
    let mut terms = Vec::<AstTerm>::new();
    let mut edges = Vec::<AstEdge>::new();

    for item in ast.items {
        match item {
            AstFuncItem::DataArgs(args) => data_args = Some(args),
            AstFuncItem::DataResults(results) => data_results = Some(results),
            AstFuncItem::Block(block) => blocks.push(block),
            AstFuncItem::Inst(inst) => insts.push(inst),
            AstFuncItem::Term(term) => terms.push(term),
            AstFuncItem::Edge(edge) => edges.push(edge),
        }
    }

    let data_args = data_args.ok_or_else(|| ParseError {
        message: format!("cfg_func @{} missing data_args line", ast.lambda_id),
    })?;
    let data_results = data_results.ok_or_else(|| ParseError {
        message: format!("cfg_func @{} missing data_results line", ast.lambda_id),
    })?;

    blocks.sort_by_key(|b| b.id.0);
    for (idx, block) in blocks.iter().enumerate() {
        if block.id.0 as usize != idx {
            return Err(ParseError {
                message: format!(
                    "cfg_func @{} block IDs must be dense from b0, got b{} at position {}",
                    ast.lambda_id, block.id.0, idx
                ),
            });
        }
    }

    insts.sort_by_key(|inst| inst.id.0);
    for (idx, inst) in insts.iter().enumerate() {
        if inst.id.0 as usize != idx {
            return Err(ParseError {
                message: format!(
                    "cfg_func @{} inst IDs must be dense from i0, got i{} at position {}",
                    ast.lambda_id, inst.id.0, idx
                ),
            });
        }
    }

    terms.sort_by_key(|term| term.id.0);
    for (idx, term) in terms.iter().enumerate() {
        if term.id.0 as usize != idx {
            return Err(ParseError {
                message: format!(
                    "cfg_func @{} term IDs must be dense from t0, got t{} at position {}",
                    ast.lambda_id, term.id.0, idx
                ),
            });
        }
    }

    edges.sort_by_key(|edge| edge.id.0);
    for (idx, edge) in edges.iter().enumerate() {
        if edge.id.0 as usize != idx {
            return Err(ParseError {
                message: format!(
                    "cfg_func @{} edge IDs must be dense from e0, got e{} at position {}",
                    ast.lambda_id, edge.id.0, idx
                ),
            });
        }
    }

    let blocks = blocks
        .into_iter()
        .map(|b| Block {
            id: b.id,
            params: b.params,
            insts: b.insts,
            term: b.term,
            preds: b.preds,
            succs: b.succs,
        })
        .collect::<Vec<_>>();

    let insts = insts
        .into_iter()
        .map(|inst| resolve_inst(inst, registry))
        .collect::<Result<Vec<_>, _>>()?;
    let terms = terms.into_iter().map(|t| t.term).collect::<Vec<_>>();
    let edges = edges
        .into_iter()
        .map(|e| Edge {
            id: e.id,
            from: e.from,
            to: e.to,
            args: e.args,
        })
        .collect::<Vec<_>>();

    let function = Function {
        id: FunctionId::new(ast.function_id),
        lambda_id: LambdaId::new(ast.lambda_id),
        entry: ast.entry,
        data_args,
        data_results,
        blocks,
        edges,
        insts,
        terms,
    };
    function.validate().map_err(|err| ParseError {
        message: err.to_string(),
    })?;
    Ok(function)
}

fn resolve_inst(ast: AstInst, registry: &IntrinsicRegistry) -> Result<Inst, ParseError> {
    let mut operands = Vec::<Operand>::new();
    for (vreg, class, fixed) in &ast.body.uses {
        operands.push(Operand {
            vreg: *vreg,
            kind: OperandKind::Use,
            class: *class,
            fixed: *fixed,
        });
    }
    for (vreg, class, fixed) in &ast.body.defs {
        operands.push(Operand {
            vreg: *vreg,
            kind: OperandKind::Def,
            class: *class,
            fixed: *fixed,
        });
    }

    let dst = ast.body.defs.first().map(|(v, _, _)| *v);
    let src = ast.body.uses.first().map(|(v, _, _)| *v);

    let op = match &ast.body.op {
        AstOp::Const(value) => LinearOp::Const {
            dst: dst.ok_or_else(|| ParseError {
                message: format!("inst i{} const missing dst", ast.id.0),
            })?,
            value: resolve_const(value, registry)?,
        },
        AstOp::BinOp(kind) => {
            if ast.body.uses.len() < 2 {
                return Err(ParseError {
                    message: format!("inst i{} binary op requires two uses", ast.id.0),
                });
            }
            LinearOp::BinOp {
                op: *kind,
                dst: dst.ok_or_else(|| ParseError {
                    message: format!("inst i{} binary op missing dst", ast.id.0),
                })?,
                lhs: ast.body.uses[0].0,
                rhs: ast.body.uses[1].0,
            }
        }
        AstOp::UnaryOp(kind) => LinearOp::UnaryOp {
            op: *kind,
            dst: dst.ok_or_else(|| ParseError {
                message: format!("inst i{} unary op missing dst", ast.id.0),
            })?,
            src: src.ok_or_else(|| ParseError {
                message: format!("inst i{} unary op missing src", ast.id.0),
            })?,
        },
        AstOp::Copy => LinearOp::Copy {
            dst: dst.ok_or_else(|| ParseError {
                message: format!("inst i{} copy missing dst", ast.id.0),
            })?,
            src: src.ok_or_else(|| ParseError {
                message: format!("inst i{} copy missing src", ast.id.0),
            })?,
        },
        AstOp::BoundsCheck(count) => LinearOp::BoundsCheck { count: *count },
        AstOp::ReadBytes(count) => LinearOp::ReadBytes {
            dst: dst.ok_or_else(|| ParseError {
                message: format!("inst i{} read_bytes missing dst", ast.id.0),
            })?,
            count: *count,
        },
        AstOp::PeekByte => LinearOp::PeekByte {
            dst: dst.ok_or_else(|| ParseError {
                message: format!("inst i{} peek_byte missing dst", ast.id.0),
            })?,
        },
        AstOp::Advance(count) => LinearOp::AdvanceCursor { count: *count },
        AstOp::AdvanceBy => LinearOp::AdvanceCursorBy {
            src: src.ok_or_else(|| ParseError {
                message: format!("inst i{} advance_by missing src", ast.id.0),
            })?,
        },
        AstOp::SaveCursor => LinearOp::SaveCursor {
            dst: dst.ok_or_else(|| ParseError {
                message: format!("inst i{} save_cursor missing dst", ast.id.0),
            })?,
        },
        AstOp::RestoreCursor => LinearOp::RestoreCursor {
            src: src.ok_or_else(|| ParseError {
                message: format!("inst i{} restore_cursor missing src", ast.id.0),
            })?,
        },
        AstOp::Store(offset, width) => LinearOp::WriteToField {
            src: src.ok_or_else(|| ParseError {
                message: format!("inst i{} store missing src", ast.id.0),
            })?,
            offset: *offset,
            width: *width,
        },
        AstOp::Load(offset, width) => LinearOp::ReadFromField {
            dst: dst.ok_or_else(|| ParseError {
                message: format!("inst i{} load missing dst", ast.id.0),
            })?,
            offset: *offset,
            width: *width,
        },
        AstOp::SaveOutPtr => LinearOp::SaveOutPtr {
            dst: dst.ok_or_else(|| ParseError {
                message: format!("inst i{} save_out_ptr missing dst", ast.id.0),
            })?,
        },
        AstOp::SetOutPtr => LinearOp::SetOutPtr {
            src: src.ok_or_else(|| ParseError {
                message: format!("inst i{} set_out_ptr missing src", ast.id.0),
            })?,
        },
        AstOp::SlotAddr(slot) => LinearOp::SlotAddr {
            dst: dst.ok_or_else(|| ParseError {
                message: format!("inst i{} slot_addr missing dst", ast.id.0),
            })?,
            slot: SlotId::new(*slot),
        },
        AstOp::WriteSlot(slot) => LinearOp::WriteToSlot {
            slot: SlotId::new(*slot),
            src: src.ok_or_else(|| ParseError {
                message: format!("inst i{} write_slot missing src", ast.id.0),
            })?,
        },
        AstOp::ReadSlot(slot) => LinearOp::ReadFromSlot {
            dst: dst.ok_or_else(|| ParseError {
                message: format!("inst i{} read_slot missing dst", ast.id.0),
            })?,
            slot: SlotId::new(*slot),
        },
        AstOp::CallIntrinsic(func, field_offset) => LinearOp::CallIntrinsic {
            func: resolve_intrinsic(func, registry)?,
            args: ast.body.uses.iter().map(|(v, _, _)| *v).collect(),
            dst,
            field_offset: *field_offset,
        },
        AstOp::CallPure(func) => LinearOp::CallPure {
            func: resolve_intrinsic(func, registry)?,
            args: ast.body.uses.iter().map(|(v, _, _)| *v).collect(),
            dst: dst.ok_or_else(|| ParseError {
                message: format!("inst i{} call_pure missing dst", ast.id.0),
            })?,
        },
        AstOp::CallLambda(target) => LinearOp::CallLambda {
            target: LambdaId::new(*target),
            args: ast.body.uses.iter().map(|(v, _, _)| *v).collect(),
            results: ast.body.defs.iter().map(|(v, _, _)| *v).collect(),
        },
        AstOp::SimdStringScan => {
            if ast.body.defs.len() < 2 {
                return Err(ParseError {
                    message: format!("inst i{} simd_string_scan needs two defs", ast.id.0),
                });
            }
            LinearOp::SimdStringScan {
                pos: ast.body.defs[0].0,
                kind: ast.body.defs[1].0,
            }
        }
        AstOp::SimdWsSkip => LinearOp::SimdWhitespaceSkip,
        AstOp::ErrorExit(code) => LinearOp::ErrorExit { code: *code },
    };

    Ok(Inst {
        id: ast.id,
        op,
        operands,
        clobbers: ast.body.clobbers,
    })
}

fn resolve_intrinsic(
    reference: &IntrinsicRef,
    registry: &IntrinsicRegistry,
) -> Result<IntrinsicFn, ParseError> {
    match reference {
        IntrinsicRef::Named(name) => registry.func_by_name(name).ok_or_else(|| ParseError {
            message: format!("unknown intrinsic: @{name}"),
        }),
        IntrinsicRef::Address(addr) => Ok(IntrinsicFn(*addr)),
    }
}

fn resolve_const(value: &ConstRef, registry: &IntrinsicRegistry) -> Result<u64, ParseError> {
    match value {
        ConstRef::Named(name) => registry.const_by_name(name).ok_or_else(|| ParseError {
            message: format!("unknown const: @{name}"),
        }),
        ConstRef::Value(value) => Ok(*value),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use kajit_ir::{IntrinsicFn, IntrinsicRegistry, IrBuilder, Width};
    use kajit_lir::linearize;

    fn test_shape() -> &'static facet::Shape {
        <u8 as facet::Facet>::SHAPE
    }

    #[test]
    fn parse_simple_cfg_program() {
        let input = r#"
cfg_program vregs=1 slots=0 {
  cfg_func @0 f0 entry=b0 {
    data_args: []
    data_results: []
    block b0 params=[] insts=[i0] term=t0 preds=[] succs=[]
    inst i0: v0:gpr = const(0x2a)
    term t0: return
  }
}
"#;
        let program = parse_cfg_mir(input).expect("cfg parser should succeed");
        assert_eq!(program.funcs.len(), 1);
        assert_eq!(program.funcs[0].blocks.len(), 1);
        assert_eq!(program.funcs[0].insts.len(), 1);
    }

    #[test]
    fn round_trip_cfg_mir_text() {
        let mut builder = IrBuilder::new(test_shape());
        {
            let mut rb = builder.root_region();
            rb.bounds_check(4);
            let val = rb.read_bytes(4);
            rb.write_to_field(val, 0, Width::W4);
            rb.set_results(&[]);
        }
        let mut func = builder.finish();
        let linear = linearize(&mut func);
        let cfg = kajit_mir::cfg_mir::lower_linear_ir(&linear);

        let text1 = format!("{cfg}");
        let cfg2 = parse_cfg_mir(&text1).expect("round-trip parse should succeed");
        let text2 = format!("{cfg2}");
        assert_eq!(
            text1, text2,
            "cfg round trip mismatch:\n--- original ---\n{text1}\n--- reparsed ---\n{text2}"
        );
    }

    #[test]
    fn round_trip_cfg_mir_text_with_named_intrinsics() {
        unsafe extern "C" fn test_intrinsic(_ctx: *mut core::ffi::c_void, _out: *mut u8) {}

        let mut registry = IntrinsicRegistry::empty();
        registry.register(
            "test_intrinsic",
            IntrinsicFn(test_intrinsic as *const () as usize),
        );

        let mut builder = IrBuilder::new(test_shape());
        {
            let mut rb = builder.root_region();
            let func = registry
                .func_by_name("test_intrinsic")
                .expect("registered intrinsic should resolve");
            rb.call_intrinsic(func, &[], 0, false);
            rb.set_results(&[]);
        }
        let mut func = builder.finish();
        let linear = linearize(&mut func);
        let cfg = kajit_mir::cfg_mir::lower_linear_ir(&linear);

        let text1 = format!("{}", cfg.display_with_registry(&registry));
        assert!(
            text1.contains("@test_intrinsic"),
            "display should use intrinsic names, got:\n{text1}"
        );
        let cfg2 = parse_cfg_mir_with_registry(&text1, &registry)
            .expect("round-trip parse should succeed");
        let text2 = format!("{}", cfg2.display_with_registry(&registry));
        assert_eq!(
            text1, text2,
            "cfg round trip mismatch:\n--- original ---\n{text1}\n--- reparsed ---\n{text2}"
        );
    }
}
