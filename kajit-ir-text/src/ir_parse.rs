//! RVSDG textual parser for kajit IR.
//!
//! Parses the text format produced by `IrFunc::Display` (with registry) back into
//! an `IrFunc`. Two passes: parse text → AST, then resolve references → IrFunc.

use std::collections::HashMap;

use chumsky::prelude::*;

use kajit_ir::ErrorCode;
use kajit_ir::{
    Arena, InputPort, IntrinsicFn, IntrinsicRegistry, IrFunc, IrOp, LambdaId, Node, NodeId,
    NodeKind, OutputPort, OutputRef, PortKind, PortSource, Region, RegionArg, RegionArgRef,
    RegionId, RegionResult, SlotId, VReg, Width,
};

// ─── AST types (first pass) ────────────────────────────────────────────────

/// A parsed source reference (unresolved).
#[derive(Debug, Clone)]
enum AstSource {
    /// `v0`, `v42` — data value from a node output
    VReg(u32),
    /// `%cs:n5` — cursor state from node n5's output
    CursorNode(u32),
    /// `%os:n3` — output state from node n3's output
    OutputNode(u32),
    /// `%cs:arg` — cursor state from region argument
    CursorArg,
    /// `%os:arg` — output state from region argument
    OutputArg,
    /// `arg0`, `arg1` — data value from region argument
    RegionArg(u16),
    /// `n1.2` — node output by index (fallback when no vreg)
    NodeOutput(u32, u16),
}

/// A parsed output port (unresolved).
#[derive(Debug, Clone)]
enum AstOutput {
    /// `v0` — data output with vreg
    VReg(u32),
    /// `%cs` — cursor state output
    Cursor,
    /// `%os` — output state output
    Output,
    /// `?` — data output without vreg assignment
    Unknown,
}

/// A parsed node (unresolved).
#[derive(Debug, Clone)]
enum AstNode {
    Simple {
        id: u32,
        op: AstOp,
        inputs: Vec<AstSource>,
        outputs: Vec<AstOutput>,
    },
    Gamma {
        id: u32,
        pred: AstSource,
        extra_inputs: Vec<AstSource>,
        branches: Vec<AstRegion>,
        outputs: Vec<AstOutput>,
    },
    Theta {
        id: u32,
        inputs: Vec<AstSource>,
        body: AstRegion,
        outputs: Vec<AstOutput>,
    },
    Apply {
        id: u32,
        target: u32,
        inputs: Vec<AstSource>,
        outputs: Vec<AstOutput>,
    },
}

/// A parsed region (unresolved).
#[derive(Debug, Clone)]
struct AstRegion {
    args: Vec<AstRegionArg>,
    nodes: Vec<AstNode>,
    results: Vec<AstSource>,
}

#[derive(Debug, Clone)]
enum AstRegionArg {
    Data(u16), // arg0, arg1, ...
    Cursor,    // %cs
    Output,    // %os
}

/// A parsed lambda (unresolved).
#[derive(Debug, Clone)]
struct AstLambda {
    id: u32,
    #[allow(dead_code)]
    shape: String,
    body: AstRegion,
}

// ─── Parsers ────────────────────────────────────────────────────────────────

type Extra<'src> = extra::Err<Rich<'src, char>>;

fn ws<'src>() -> impl Parser<'src, &'src str, (), Extra<'src>> + Clone {
    any()
        .filter(|c: &char| c.is_whitespace())
        .repeated()
        .ignored()
}

/// Parse a u32 decimal number.
fn uint32<'src>() -> impl Parser<'src, &'src str, u32, Extra<'src>> + Clone {
    text::int::<_, Extra<'_>>(10).map(|s: &str| s.parse::<u32>().unwrap())
}

/// Parse a u64 number (decimal or hex with 0x prefix).
fn uint64<'src>() -> impl Parser<'src, &'src str, u64, Extra<'src>> + Clone {
    let hex = just("0x")
        .ignore_then(text::int::<_, Extra<'_>>(16))
        .map(|s: &str| u64::from_str_radix(s, 16).unwrap());
    let dec = text::int::<_, Extra<'_>>(10).map(|s: &str| s.parse::<u64>().unwrap());
    hex.or(dec)
}

/// Parse a u16 decimal number.
fn uint16<'src>() -> impl Parser<'src, &'src str, u16, Extra<'src>> + Clone {
    text::int::<_, Extra<'_>>(10).map(|s: &str| s.parse::<u16>().unwrap())
}

/// Parse a width: W1, W2, W4, W8.
fn width<'src>() -> impl Parser<'src, &'src str, Width, Extra<'src>> + Clone {
    choice((
        just("W1").to(Width::W1),
        just("W2").to(Width::W2),
        just("W4").to(Width::W4),
        just("W8").to(Width::W8),
    ))
}

/// Parse an ErrorCode name.
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

/// Parse a source reference.
fn source<'src>() -> impl Parser<'src, &'src str, AstSource, Extra<'src>> + Clone {
    // Order matters: try more specific patterns first
    let cursor_node = just("%cs:n")
        .ignore_then(uint32())
        .map(AstSource::CursorNode);
    let output_node = just("%os:n")
        .ignore_then(uint32())
        .map(AstSource::OutputNode);
    let cursor_arg = just("%cs:arg").to(AstSource::CursorArg);
    let output_arg = just("%os:arg").to(AstSource::OutputArg);
    let region_arg = just("arg").ignore_then(uint16()).map(AstSource::RegionArg);
    let node_output = just("n")
        .ignore_then(uint32())
        .then_ignore(just("."))
        .then(uint16())
        .map(|(n, i)| AstSource::NodeOutput(n, i));
    let vreg = just("v").ignore_then(uint32()).map(AstSource::VReg);

    choice((
        cursor_arg,
        output_arg,
        cursor_node,
        output_node,
        region_arg,
        node_output,
        vreg,
    ))
}

/// Parse an output port.
fn output<'src>() -> impl Parser<'src, &'src str, AstOutput, Extra<'src>> + Clone {
    let cursor = just("%cs").to(AstOutput::Cursor);
    let output_state = just("%os").to(AstOutput::Output);
    let unknown = just("?").to(AstOutput::Unknown);
    let vreg = just("v").ignore_then(uint32()).map(AstOutput::VReg);

    choice((cursor, output_state, unknown, vreg))
}

/// Parse a comma-separated list inside brackets.
fn bracketed_list<'src, T: 'src>(
    inner: impl Parser<'src, &'src str, T, Extra<'src>> + Clone,
) -> impl Parser<'src, &'src str, Vec<T>, Extra<'src>> + Clone {
    inner
        .separated_by(just(",").padded_by(ws()))
        .allow_trailing()
        .collect::<Vec<_>>()
        .delimited_by(just("[").then(ws()), ws().then(just("]")))
}

/// Parse an intrinsic function reference: `@name` or hex `0xABC`.
fn intrinsic_ref<'src>() -> impl Parser<'src, &'src str, IntrinsicRef, Extra<'src>> + Clone {
    let named = just("@")
        .ignore_then(symbol_name())
        .map(IntrinsicRef::Named);
    let hex = just("0x")
        .ignore_then(text::int::<_, Extra<'_>>(16))
        .map(|s: &str| IntrinsicRef::Address(usize::from_str_radix(s, 16).unwrap()));
    named.or(hex)
}

fn symbol_name<'src>() -> impl Parser<'src, &'src str, String, Extra<'src>> + Clone {
    any()
        .filter(|c: &char| c.is_ascii_alphanumeric() || matches!(c, '_' | ':' | '.'))
        .repeated()
        .at_least(1)
        .to_slice()
        .map(str::to_owned)
}

#[derive(Debug, Clone)]
enum IntrinsicRef {
    Named(String),
    Address(usize),
}

#[derive(Debug, Clone)]
enum ConstRef {
    Named(String),
    Value(u64),
}

/// Parse an IrOp (the operation name with parameters).
fn ir_op<'src>() -> impl Parser<'src, &'src str, AstOp, Extra<'src>> + Clone {
    let cursor_ops = choice((
        just("ReadBytes(")
            .ignore_then(uint32())
            .then_ignore(just(")"))
            .map(|c| AstOp::Resolved(IrOp::ReadBytes { count: c })),
        just("PeekByte").to(AstOp::Resolved(IrOp::PeekByte)),
        just("AdvanceCursorBy").to(AstOp::Resolved(IrOp::AdvanceCursorBy)),
        just("AdvanceCursor(")
            .ignore_then(uint32())
            .then_ignore(just(")"))
            .map(|c| AstOp::Resolved(IrOp::AdvanceCursor { count: c })),
        just("BoundsCheck(")
            .ignore_then(uint32())
            .then_ignore(just(")"))
            .map(|c| AstOp::Resolved(IrOp::BoundsCheck { count: c })),
        just("SaveCursor").to(AstOp::Resolved(IrOp::SaveCursor)),
        just("RestoreCursor").to(AstOp::Resolved(IrOp::RestoreCursor)),
    ));

    let output_ops = choice((
        just("WriteToField(offset=")
            .ignore_then(uint32())
            .then_ignore(just(",").then(ws()))
            .then(width())
            .then_ignore(just(")"))
            .map(|(o, w)| {
                AstOp::Resolved(IrOp::WriteToField {
                    offset: o,
                    width: w,
                })
            }),
        just("ReadFromField(offset=")
            .ignore_then(uint32())
            .then_ignore(just(",").then(ws()))
            .then(width())
            .then_ignore(just(")"))
            .map(|(o, w)| {
                AstOp::Resolved(IrOp::ReadFromField {
                    offset: o,
                    width: w,
                })
            }),
        just("SaveOutPtr").to(AstOp::Resolved(IrOp::SaveOutPtr)),
        just("SetOutPtr").to(AstOp::Resolved(IrOp::SetOutPtr)),
    ));

    let stack_ops = choice((
        just("SlotAddr(")
            .ignore_then(uint32())
            .then_ignore(just(")"))
            .map(|s| {
                AstOp::Resolved(IrOp::SlotAddr {
                    slot: SlotId::new(s),
                })
            }),
        just("WriteToSlot(")
            .ignore_then(uint32())
            .then_ignore(just(")"))
            .map(|s| {
                AstOp::Resolved(IrOp::WriteToSlot {
                    slot: SlotId::new(s),
                })
            }),
        just("ReadFromSlot(")
            .ignore_then(uint32())
            .then_ignore(just(")"))
            .map(|s| {
                AstOp::Resolved(IrOp::ReadFromSlot {
                    slot: SlotId::new(s),
                })
            }),
    ));

    let arith_ops = choice((
        just("Const(")
            .ignore_then(
                just("@")
                    .ignore_then(symbol_name())
                    .map(ConstRef::Named)
                    .or(uint64().map(ConstRef::Value)),
            )
            .then_ignore(just(")"))
            .map(AstOp::Const),
        just("CmpNe").to(AstOp::Resolved(IrOp::CmpNe)),
        just("Add").to(AstOp::Resolved(IrOp::Add)),
        just("Sub").to(AstOp::Resolved(IrOp::Sub)),
        just("And").to(AstOp::Resolved(IrOp::And)),
        just("Or").to(AstOp::Resolved(IrOp::Or)),
        just("Shr").to(AstOp::Resolved(IrOp::Shr)),
        just("Shl").to(AstOp::Resolved(IrOp::Shl)),
        just("Xor").to(AstOp::Resolved(IrOp::Xor)),
        just("ZigzagDecode(wide=")
            .ignore_then(choice((just("true").to(true), just("false").to(false))))
            .then_ignore(just(")"))
            .map(|w| AstOp::Resolved(IrOp::ZigzagDecode { wide: w })),
        just("SignExtend(from=")
            .ignore_then(width())
            .then_ignore(just(")"))
            .map(|w| AstOp::Resolved(IrOp::SignExtend { from_width: w })),
    ));

    let call_ops = choice((
        just("CallIntrinsic(")
            .ignore_then(intrinsic_ref())
            .then_ignore(just(",").then(ws()).then(just("field_offset=")))
            .then(uint32())
            .then_ignore(just(")"))
            .map(|(func, fo)| AstOp::CallIntrinsic {
                func,
                field_offset: fo,
            }),
        just("CallPure(")
            .ignore_then(intrinsic_ref())
            .then_ignore(just(")"))
            .map(|func| AstOp::CallPure { func }),
        just("ErrorExit(")
            .ignore_then(error_code())
            .then_ignore(just(")"))
            .map(|c| AstOp::Resolved(IrOp::ErrorExit { code: c })),
        just("SimdStringScan").to(AstOp::Resolved(IrOp::SimdStringScan)),
        just("SimdWhitespaceSkip").to(AstOp::Resolved(IrOp::SimdWhitespaceSkip)),
    ));

    choice((cursor_ops, output_ops, stack_ops, arith_ops, call_ops))
}

/// AST op — some ops are fully resolved, some need intrinsic lookup.
#[derive(Debug, Clone)]
enum AstOp {
    Resolved(IrOp),
    Const(ConstRef),
    CallIntrinsic {
        func: IntrinsicRef,
        field_offset: u32,
    },
    CallPure {
        func: IntrinsicRef,
    },
}

/// Parse a region arg: `%cs`, `%os`, `arg0`, `arg1`, etc.
fn region_arg<'src>() -> impl Parser<'src, &'src str, AstRegionArg, Extra<'src>> + Clone {
    let cursor = just("%cs").to(AstRegionArg::Cursor);
    let output_state = just("%os").to(AstRegionArg::Output);
    let data = just("arg").ignore_then(uint16()).map(AstRegionArg::Data);
    choice((cursor, output_state, data))
}

fn region<'src>() -> impl Parser<'src, &'src str, AstRegion, Extra<'src>> + Clone {
    recursive(|region| {
        let args = just("args:")
            .padded_by(ws())
            .ignore_then(bracketed_list(region_arg()));

        let simple_node = just("n")
            .ignore_then(uint32())
            .then_ignore(ws().then(just("=")).then(ws()))
            .then(ir_op())
            .then_ignore(ws())
            .then(bracketed_list(source()))
            .then_ignore(ws().then(just("->")).then(ws()))
            .then(bracketed_list(output()))
            .map(|(((id, op), inputs), outputs)| AstNode::Simple {
                id,
                op,
                inputs,
                outputs,
            });

        let gamma_node = just("n")
            .ignore_then(uint32())
            .then_ignore(
                ws().then(just("="))
                    .then(ws())
                    .then(just("gamma"))
                    .then(ws()),
            )
            .then_ignore(just("[").then(ws()))
            .then(
                // pred: <source>
                just("pred:")
                    .padded_by(ws())
                    .ignore_then(source())
                    .then_ignore(ws()),
            )
            .then(
                // in0: <source>, in1: <source>, ...
                just("in")
                    .ignore_then(uint32())
                    .ignore_then(just(":").then(ws()))
                    .ignore_then(source())
                    .then_ignore(ws())
                    .repeated()
                    .collect::<Vec<_>>(),
            )
            .then_ignore(ws().then(just("]")).then(ws()).then(just("{")).then(ws()))
            .then(
                // branches
                just("branch")
                    .padded_by(ws())
                    .ignore_then(uint32())
                    .then_ignore(just(":").then(ws()))
                    .ignore_then(
                        just("region")
                            .ignore_then(ws())
                            .ignore_then(just("{"))
                            .ignore_then(ws())
                            .ignore_then(region.clone())
                            .then_ignore(ws().then(just("}")).then(ws())),
                    )
                    .repeated()
                    .at_least(1)
                    .collect::<Vec<_>>(),
            )
            .then_ignore(ws().then(just("}")).then(ws()).then(just("->")).then(ws()))
            .then(bracketed_list(output()))
            .map(
                |((((id, pred), extra_inputs), branches), outputs)| AstNode::Gamma {
                    id,
                    pred,
                    extra_inputs,
                    branches,
                    outputs,
                },
            );

        let theta_node = just("n")
            .ignore_then(uint32())
            .then_ignore(
                ws().then(just("="))
                    .then(ws())
                    .then(just("theta"))
                    .then(ws()),
            )
            .then(bracketed_list(source()))
            .then_ignore(ws().then(just("{")).then(ws()))
            .then(
                just("region")
                    .ignore_then(ws())
                    .ignore_then(just("{"))
                    .ignore_then(ws())
                    .ignore_then(region.clone())
                    .then_ignore(ws().then(just("}"))),
            )
            .then_ignore(ws().then(just("}")).then(ws()).then(just("->")).then(ws()))
            .then(bracketed_list(output()))
            .map(|(((id, inputs), body), outputs)| AstNode::Theta {
                id,
                inputs,
                body,
                outputs,
            });

        let apply_node = just("n")
            .ignore_then(uint32())
            .then_ignore(
                ws().then(just("="))
                    .then(ws())
                    .then(just("apply"))
                    .then(ws()),
            )
            .then_ignore(just("@"))
            .then(uint32())
            .then_ignore(ws())
            .then(bracketed_list(source()))
            .then_ignore(ws().then(just("->")).then(ws()))
            .then(bracketed_list(output()))
            .map(|(((id, target), inputs), outputs)| AstNode::Apply {
                id,
                target,
                inputs,
                outputs,
            });

        let node = choice((gamma_node, theta_node, apply_node, simple_node));

        let nodes = node.padded_by(ws()).repeated().collect::<Vec<_>>();

        let results = just("results:")
            .padded_by(ws())
            .ignore_then(bracketed_list(source()));

        args.then_ignore(ws())
            .then(nodes)
            .then_ignore(ws())
            .then(results)
            .map(|((args, nodes), results)| AstRegion {
                args,
                nodes,
                results,
            })
    })
}

fn lambda<'src>() -> impl Parser<'src, &'src str, AstLambda, Extra<'src>> + Clone {
    just("lambda")
        .padded_by(ws())
        .ignore_then(just("@"))
        .ignore_then(uint32())
        .then_ignore(ws())
        .then(
            just("(shape:")
                .ignore_then(ws())
                .ignore_then(
                    just("\"")
                        .ignore_then(any().filter(|c: &char| *c != '"').repeated().to_slice())
                        .then_ignore(just("\"")),
                )
                .then_ignore(just(")")),
        )
        .then_ignore(ws().then(just("{")).then(ws()))
        .then(
            just("region")
                .ignore_then(ws())
                .ignore_then(just("{"))
                .ignore_then(ws())
                .ignore_then(region())
                .then_ignore(ws().then(just("}"))),
        )
        .then_ignore(ws().then(just("}")))
        .map(|((id, shape), body)| AstLambda {
            id,
            shape: shape.to_string(),
            body,
        })
}

fn program<'src>() -> impl Parser<'src, &'src str, Vec<AstLambda>, Extra<'src>> + Clone {
    lambda()
        .padded_by(ws())
        .repeated()
        .at_least(1)
        .collect::<Vec<_>>()
        .then_ignore(end())
}

// ─── Resolution (second pass) ──────────────────────────────────────────────

/// Parse error with context.
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

/// Parse an RVSDG text format into an `IrFunc`.
///
/// The `shape` parameter provides the `&'static Shape` for the root lambda.
/// For test fixtures, use `<u8 as facet::Facet>::SHAPE` as a dummy.
///
/// The `registry` resolves `@name` intrinsic references to `IntrinsicFn` values.
pub fn parse_ir(
    input: &str,
    shape: &'static facet::Shape,
    registry: &IntrinsicRegistry,
) -> Result<IrFunc, ParseError> {
    let result = program().parse(input);

    let lambdas = result.into_result().map_err(|errs| {
        let msgs: Vec<String> = errs.into_iter().map(|e| format!("{e}")).collect();
        ParseError {
            message: msgs.join("\n"),
        }
    })?;

    resolve(lambdas, shape, registry)
}

/// Resolve AST references into a concrete `IrFunc`.
fn resolve(
    lambdas: Vec<AstLambda>,
    shape: &'static facet::Shape,
    registry: &IntrinsicRegistry,
) -> Result<IrFunc, ParseError> {
    let mut func = IrFunc {
        nodes: Arena::new(),
        regions: Arena::new(),
        root: NodeId::new(0),
        vreg_count: 0,
        slot_count: 0,
        lambdas: Vec::new(),
    };

    // Track max vreg and slot seen, to set counts at the end.
    let mut max_vreg: u32 = 0;
    let mut max_slot: u32 = 0;

    // We need to process lambdas in order. The first lambda is the root.
    // We'll do a two-pass approach:
    // 1. Pre-allocate all lambdas (so cross-references work).
    // 2. Build the body regions.

    // Mapping from AST node id → actual NodeId (within each region context).
    // Since node IDs in the text format are globally unique per lambda, we
    // build a global mapping.
    let mut node_map: HashMap<u32, NodeId> = HashMap::new();

    // Pre-allocate lambda nodes in the arena first (so they get low indices
    // matching the original IrBuilder pattern: lambda@0 → NodeId 0, etc.).
    let sentinel_region = RegionId::new(u32::MAX);
    let placeholder_body = RegionId::new(u32::MAX);
    let mut lambda_node_ids = Vec::new();
    for ast_lambda in &lambdas {
        let lambda_id = LambdaId::new(ast_lambda.id);
        let lambda_shape = shape; // non-root lambdas also get same shape for now
        let node_id = func.nodes.push(Node {
            region: sentinel_region,
            inputs: Vec::new(),
            outputs: Vec::new(),
            kind: NodeKind::Lambda {
                body: placeholder_body,
                shape: lambda_shape,
                lambda_id,
            },
        });
        lambda_node_ids.push((ast_lambda.id, node_id, lambda_id));

        while func.lambdas.len() <= lambda_id.index() {
            func.lambdas.push(NodeId::new(u32::MAX));
        }
        func.lambdas[lambda_id.index()] = node_id;
        if ast_lambda.id == 0 {
            func.root = node_id;
        }
    }

    // Now build the body regions for each lambda and patch the lambda nodes.
    for (i, ast_lambda) in lambdas.iter().enumerate() {
        let (_, node_id, lambda_id) = lambda_node_ids[i];

        let body_region_id = resolve_region(
            &ast_lambda.body,
            &mut func,
            &mut node_map,
            &mut max_vreg,
            &mut max_slot,
            registry,
        )?;

        // Patch the lambda node's body region.
        func.nodes[node_id].kind = NodeKind::Lambda {
            body: body_region_id,
            shape,
            lambda_id,
        };
    }

    func.vreg_count = if max_vreg == 0 && node_map.is_empty() {
        0
    } else {
        max_vreg + 1
    };
    func.slot_count = max_slot + 1;

    Ok(func)
}

fn resolve_region(
    ast: &AstRegion,
    func: &mut IrFunc,
    node_map: &mut HashMap<u32, NodeId>,
    max_vreg: &mut u32,
    max_slot: &mut u32,
    registry: &IntrinsicRegistry,
) -> Result<RegionId, ParseError> {
    // Build region args.
    let args: Vec<RegionArg> = ast
        .args
        .iter()
        .map(|a| match a {
            AstRegionArg::Data(idx) => {
                let _ = idx;
                RegionArg {
                    kind: PortKind::Data,
                    vreg: None,
                }
            }
            AstRegionArg::Cursor => RegionArg {
                kind: PortKind::StateCursor,
                vreg: None,
            },
            AstRegionArg::Output => RegionArg {
                kind: PortKind::StateOutput,
                vreg: None,
            },
        })
        .collect();

    let region_id = func.regions.push(Region {
        args,
        results: Vec::new(),
        nodes: Vec::new(),
    });

    // First pass: create placeholder nodes so we can resolve forward references.
    // Actually, the text format lists nodes in order, and sources always reference
    // earlier nodes or region args. So we can build them sequentially.
    for ast_node in &ast.nodes {
        let node_id = resolve_node(
            ast_node, region_id, func, node_map, max_vreg, max_slot, registry,
        )?;
        func.regions[region_id].nodes.push(node_id);
    }

    // Resolve results.
    let results: Vec<RegionResult> = ast
        .results
        .iter()
        .map(|s| {
            let (source, kind) = resolve_source(s, region_id, node_map, func);
            RegionResult { kind, source }
        })
        .collect();
    func.regions[region_id].results = results;

    Ok(region_id)
}

fn resolve_node(
    ast: &AstNode,
    region_id: RegionId,
    func: &mut IrFunc,
    node_map: &mut HashMap<u32, NodeId>,
    max_vreg: &mut u32,
    max_slot: &mut u32,
    registry: &IntrinsicRegistry,
) -> Result<NodeId, ParseError> {
    match ast {
        AstNode::Simple {
            id,
            op,
            inputs,
            outputs,
        } => {
            let resolved_op = resolve_op(op, registry)?;

            // Track slots.
            match &resolved_op {
                IrOp::SlotAddr { slot }
                | IrOp::WriteToSlot { slot }
                | IrOp::ReadFromSlot { slot } => {
                    *max_slot = (*max_slot).max(slot.index() as u32);
                }
                _ => {}
            }

            let resolved_inputs: Vec<InputPort> = inputs
                .iter()
                .map(|s| {
                    let (source, kind) = resolve_source(s, region_id, node_map, func);
                    InputPort { kind, source }
                })
                .collect();

            let resolved_outputs: Vec<OutputPort> = outputs
                .iter()
                .map(|o| resolve_output(o, max_vreg))
                .collect();

            // Patch CallIntrinsic/CallPure arg_count and has_result from actual ports.
            let resolved_op = match resolved_op {
                IrOp::CallIntrinsic {
                    func: f,
                    field_offset,
                    ..
                } => {
                    let data_inputs = resolved_inputs
                        .iter()
                        .filter(|i| i.kind == PortKind::Data)
                        .count();
                    let has_result = resolved_outputs.iter().any(|o| o.kind == PortKind::Data);
                    IrOp::CallIntrinsic {
                        func: f,
                        arg_count: data_inputs as u8,
                        has_result,
                        field_offset,
                    }
                }
                IrOp::CallPure { func: f, .. } => {
                    let arg_count = resolved_inputs.len();
                    IrOp::CallPure {
                        func: f,
                        arg_count: arg_count as u8,
                    }
                }
                other => other,
            };

            let node_id = func.nodes.push(Node {
                region: region_id,
                inputs: resolved_inputs,
                outputs: resolved_outputs,
                kind: NodeKind::Simple(resolved_op),
            });

            node_map.insert(*id, node_id);
            Ok(node_id)
        }

        AstNode::Gamma {
            id,
            pred,
            extra_inputs,
            branches,
            outputs,
        } => {
            // Build sub-regions first.
            let mut region_ids = Vec::new();
            for branch in branches {
                let rid = resolve_region(branch, func, node_map, max_vreg, max_slot, registry)?;
                region_ids.push(rid);
            }

            // Build inputs: pred + extra + cursor + output state
            let mut resolved_inputs = Vec::new();
            let (pred_source, pred_kind) = resolve_source(pred, region_id, node_map, func);
            resolved_inputs.push(InputPort {
                kind: pred_kind,
                source: pred_source,
            });
            for s in extra_inputs {
                let (source, kind) = resolve_source(s, region_id, node_map, func);
                resolved_inputs.push(InputPort { kind, source });
            }

            let resolved_outputs: Vec<OutputPort> = outputs
                .iter()
                .map(|o| resolve_output(o, max_vreg))
                .collect();

            let node_id = func.nodes.push(Node {
                region: region_id,
                inputs: resolved_inputs,
                outputs: resolved_outputs,
                kind: NodeKind::Gamma {
                    regions: region_ids,
                },
            });

            node_map.insert(*id, node_id);
            Ok(node_id)
        }

        AstNode::Theta {
            id,
            inputs,
            body,
            outputs,
        } => {
            let body_id = resolve_region(body, func, node_map, max_vreg, max_slot, registry)?;

            let resolved_inputs: Vec<InputPort> = inputs
                .iter()
                .map(|s| {
                    let (source, kind) = resolve_source(s, region_id, node_map, func);
                    InputPort { kind, source }
                })
                .collect();

            let resolved_outputs: Vec<OutputPort> = outputs
                .iter()
                .map(|o| resolve_output(o, max_vreg))
                .collect();

            // Theta invariant checks: textual IR must follow builder semantics.
            // inputs: [loop_vars..., %cs, %os]
            // body args: [loop_vars..., %cs, %os]
            // body results: [pred, loop_vars..., %cs, %os]
            // outputs: [loop_vars..., %cs, %os]
            if resolved_inputs.len() < 2 {
                return Err(ParseError {
                    message: format!("theta n{id} must have at least %cs/%os inputs"),
                });
            }
            if resolved_outputs.len() < 2 {
                return Err(ParseError {
                    message: format!("theta n{id} must have at least %cs/%os outputs"),
                });
            }

            let body_region = &func.regions[body_id];
            if body_region.args.len() != resolved_inputs.len() {
                return Err(ParseError {
                    message: format!(
                        "theta n{id} body arg count mismatch: got {}, expected {}",
                        body_region.args.len(),
                        resolved_inputs.len()
                    ),
                });
            }
            if body_region.results.len() != resolved_outputs.len() + 1 {
                return Err(ParseError {
                    message: format!(
                        "theta n{id} body result count mismatch: got {}, expected {} (predicate + outputs)",
                        body_region.results.len(),
                        resolved_outputs.len() + 1
                    ),
                });
            }

            let in_last = resolved_inputs.len() - 1;
            let out_last = resolved_outputs.len() - 1;
            let body_arg_last = body_region.args.len() - 1;
            let body_res_last = body_region.results.len() - 1;

            if resolved_inputs[in_last - 1].kind != PortKind::StateCursor
                || resolved_inputs[in_last].kind != PortKind::StateOutput
                || body_region.args[body_arg_last - 1].kind != PortKind::StateCursor
                || body_region.args[body_arg_last].kind != PortKind::StateOutput
                || resolved_outputs[out_last - 1].kind != PortKind::StateCursor
                || resolved_outputs[out_last].kind != PortKind::StateOutput
                || body_region.results[body_res_last - 1].kind != PortKind::StateCursor
                || body_region.results[body_res_last].kind != PortKind::StateOutput
            {
                return Err(ParseError {
                    message: format!(
                        "theta n{id} must thread %cs/%os as trailing inputs/args/results/outputs"
                    ),
                });
            }

            if body_region.results[0].kind != PortKind::Data {
                return Err(ParseError {
                    message: format!("theta n{id} first body result must be predicate data"),
                });
            }

            let loop_var_count = resolved_inputs.len() - 2;
            for i in 0..loop_var_count {
                if resolved_inputs[i].kind != PortKind::Data
                    || body_region.args[i].kind != PortKind::Data
                    || body_region.results[i + 1].kind != PortKind::Data
                    || resolved_outputs[i].kind != PortKind::Data
                {
                    return Err(ParseError {
                        message: format!(
                            "theta n{id} loop-carried values must be data ports at index {i}"
                        ),
                    });
                }
            }

            let node_id = func.nodes.push(Node {
                region: region_id,
                inputs: resolved_inputs,
                outputs: resolved_outputs,
                kind: NodeKind::Theta { body: body_id },
            });

            node_map.insert(*id, node_id);
            Ok(node_id)
        }

        AstNode::Apply {
            id,
            target,
            inputs,
            outputs,
        } => {
            let resolved_inputs: Vec<InputPort> = inputs
                .iter()
                .map(|s| {
                    let (source, kind) = resolve_source(s, region_id, node_map, func);
                    InputPort { kind, source }
                })
                .collect();

            let resolved_outputs: Vec<OutputPort> = outputs
                .iter()
                .map(|o| resolve_output(o, max_vreg))
                .collect();

            let node_id = func.nodes.push(Node {
                region: region_id,
                inputs: resolved_inputs,
                outputs: resolved_outputs,
                kind: NodeKind::Apply {
                    target: LambdaId::new(*target),
                },
            });

            node_map.insert(*id, node_id);
            Ok(node_id)
        }
    }
}

fn resolve_source(
    s: &AstSource,
    region_id: RegionId,
    node_map: &HashMap<u32, NodeId>,
    func: &IrFunc,
) -> (PortSource, PortKind) {
    match s {
        AstSource::VReg(v) => {
            // Find which node output has this vreg. We need to search for it.
            // Look through all known nodes.
            for &node_id in node_map.values() {
                let node = &func.nodes[node_id];
                for (idx, out) in node.outputs.iter().enumerate() {
                    if out.kind == PortKind::Data
                        && let Some(vreg) = out.vreg
                        && vreg.index() == *v as usize
                    {
                        return (
                            PortSource::Node(OutputRef {
                                node: node_id,
                                index: idx as u16,
                            }),
                            PortKind::Data,
                        );
                    }
                }
            }
            // Fallback: might be referencing a region arg with a vreg.
            // This shouldn't happen in our format, but panic clearly.
            panic!("unresolved vreg v{v}");
        }
        AstSource::CursorNode(n) => {
            let node_id = node_map[n];
            let node = &func.nodes[node_id];
            for (idx, out) in node.outputs.iter().enumerate() {
                if out.kind == PortKind::StateCursor {
                    return (
                        PortSource::Node(OutputRef {
                            node: node_id,
                            index: idx as u16,
                        }),
                        PortKind::StateCursor,
                    );
                }
            }
            panic!("no cursor state output on node n{n}");
        }
        AstSource::OutputNode(n) => {
            let node_id = node_map[n];
            let node = &func.nodes[node_id];
            for (idx, out) in node.outputs.iter().enumerate() {
                if out.kind == PortKind::StateOutput {
                    return (
                        PortSource::Node(OutputRef {
                            node: node_id,
                            index: idx as u16,
                        }),
                        PortKind::StateOutput,
                    );
                }
            }
            panic!("no output state output on node n{n}");
        }
        AstSource::CursorArg => {
            let region = &func.regions[region_id];
            for (idx, arg) in region.args.iter().enumerate() {
                if arg.kind == PortKind::StateCursor {
                    return (
                        PortSource::RegionArg(RegionArgRef {
                            region: region_id,
                            index: idx as u16,
                        }),
                        PortKind::StateCursor,
                    );
                }
            }
            panic!("no cursor state arg in region");
        }
        AstSource::OutputArg => {
            let region = &func.regions[region_id];
            for (idx, arg) in region.args.iter().enumerate() {
                if arg.kind == PortKind::StateOutput {
                    return (
                        PortSource::RegionArg(RegionArgRef {
                            region: region_id,
                            index: idx as u16,
                        }),
                        PortKind::StateOutput,
                    );
                }
            }
            panic!("no output state arg in region");
        }
        AstSource::RegionArg(idx) => (
            PortSource::RegionArg(RegionArgRef {
                region: region_id,
                index: *idx,
            }),
            PortKind::Data,
        ),
        AstSource::NodeOutput(n, idx) => {
            let node_id = node_map[n];
            let node = &func.nodes[node_id];
            let kind = node.outputs[*idx as usize].kind;
            (
                PortSource::Node(OutputRef {
                    node: node_id,
                    index: *idx,
                }),
                kind,
            )
        }
    }
}

fn resolve_output(o: &AstOutput, max_vreg: &mut u32) -> OutputPort {
    match o {
        AstOutput::VReg(v) => {
            *max_vreg = (*max_vreg).max(*v);
            OutputPort {
                kind: PortKind::Data,
                vreg: Some(VReg::new(*v)),
            }
        }
        AstOutput::Cursor => OutputPort {
            kind: PortKind::StateCursor,
            vreg: None,
        },
        AstOutput::Output => OutputPort {
            kind: PortKind::StateOutput,
            vreg: None,
        },
        AstOutput::Unknown => OutputPort {
            kind: PortKind::Data,
            vreg: None,
        },
    }
}

fn resolve_op(op: &AstOp, registry: &IntrinsicRegistry) -> Result<IrOp, ParseError> {
    match op {
        AstOp::Resolved(ir_op) => Ok(ir_op.clone()),
        AstOp::Const(value) => Ok(IrOp::Const {
            value: resolve_const(value, registry)?,
        }),
        AstOp::CallIntrinsic { func, field_offset } => {
            let intrinsic = resolve_intrinsic(func, registry)?;
            // Count args from the inputs (they'll be resolved separately).
            // For now, arg_count and has_result are inferred from the node's
            // input/output ports during resolve_node. But IrOp needs them.
            // We'll set them to 0/false here and patch them in resolve_node.
            Ok(IrOp::CallIntrinsic {
                func: intrinsic,
                arg_count: 0,      // patched later from actual inputs
                has_result: false, // patched later from actual outputs
                field_offset: *field_offset,
            })
        }
        AstOp::CallPure { func } => {
            let intrinsic = resolve_intrinsic(func, registry)?;
            Ok(IrOp::CallPure {
                func: intrinsic,
                arg_count: 0, // patched later
            })
        }
    }
}

fn resolve_intrinsic(
    r: &IntrinsicRef,
    registry: &IntrinsicRegistry,
) -> Result<IntrinsicFn, ParseError> {
    match r {
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

// ─── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use kajit_ir::{IrBuilder, Width};

    fn test_shape() -> &'static facet::Shape {
        <u8 as facet::Facet>::SHAPE
    }

    #[test]
    fn parse_simple_read_write() {
        let input = r#"
lambda @0 (shape: "u8") {
  region {
    args: [%cs, %os]
    n0 = BoundsCheck(4) [%cs:arg] -> [%cs]
    n1 = ReadBytes(4) [%cs:n0] -> [v0, %cs]
    n2 = WriteToField(offset=0, W4) [v0, %os:arg] -> [%os]
    results: [%cs:n1, %os:n2]
  }
}
"#;

        let registry = IntrinsicRegistry::empty();
        let func = parse_ir(input, test_shape(), &registry).unwrap();

        // Verify structure.
        assert_eq!(func.lambdas.len(), 1);

        let root_body = func.root_body();
        let region = &func.regions[root_body];
        assert_eq!(region.args.len(), 2); // %cs, %os
        assert_eq!(region.nodes.len(), 3); // BoundsCheck, ReadBytes, WriteToField
        assert_eq!(region.results.len(), 2);
    }

    #[test]
    fn parse_const_arithmetic() {
        let input = r#"
lambda @0 (shape: "u8") {
  region {
    args: [%cs, %os]
    n0 = Const(0x2a) [] -> [v0]
    n1 = Const(0x10) [] -> [v1]
    n2 = Add [v0, v1] -> [v2]
    n3 = WriteToField(offset=0, W8) [v2, %os:arg] -> [%os]
    results: [%cs:arg, %os:n3]
  }
}
"#;

        let registry = IntrinsicRegistry::empty();
        let func = parse_ir(input, test_shape(), &registry).unwrap();

        let root_body = func.root_body();
        let region = &func.regions[root_body];
        assert_eq!(region.nodes.len(), 4);

        // Verify Const(0x2a) parsed correctly.
        let node0 = &func.nodes[region.nodes[0]];
        match &node0.kind {
            NodeKind::Simple(IrOp::Const { value }) => assert_eq!(*value, 0x2a),
            other => panic!("expected Const, got {other:?}"),
        }
    }

    #[test]
    fn parse_gamma() {
        let input = r#"
lambda @0 (shape: "u8") {
  region {
    args: [%cs, %os]
    n0 = Const(0x0) [] -> [v0]
    n1 = gamma [
      pred: v0
      in0: %cs:arg
      in1: %os:arg
    ] {
      branch 0:
        region {
          args: [%cs, %os]
          n2 = Const(0x2a) [] -> [v1]
          results: [v1, %cs:arg, %os:arg]
        }
      branch 1:
        region {
          args: [%cs, %os]
          n3 = Const(0x63) [] -> [v2]
          results: [v2, %cs:arg, %os:arg]
        }
    } -> [v3, %cs, %os]
    n4 = WriteToField(offset=0, W4) [v3, %os:n1] -> [%os]
    results: [%cs:n1, %os:n4]
  }
}
"#;

        let registry = IntrinsicRegistry::empty();
        let func = parse_ir(input, test_shape(), &registry).unwrap();

        let root_body = func.root_body();
        let region = &func.regions[root_body];
        assert_eq!(region.nodes.len(), 3); // Const, gamma, WriteToField

        let gamma_node = &func.nodes[region.nodes[1]];
        match &gamma_node.kind {
            NodeKind::Gamma { regions } => {
                assert_eq!(regions.len(), 2);
            }
            other => panic!("expected Gamma, got {other:?}"),
        }
    }

    #[test]
    fn round_trip_simple() {
        // Build IR programmatically.
        let mut builder = IrBuilder::new(test_shape());
        {
            let mut rb = builder.root_region();
            rb.bounds_check(4);
            let val = rb.read_bytes(4);
            rb.write_to_field(val, 0, Width::W4);
            rb.set_results(&[]);
        }
        let func = builder.finish();

        // Display → parse → display, check equality.
        let registry = IntrinsicRegistry::empty();
        let text1 = format!("{}", func.display_with_registry(&registry));
        let func2 = parse_ir(&text1, test_shape(), &registry).unwrap();
        let text2 = format!("{}", func2.display_with_registry(&registry));

        assert_eq!(
            text1, text2,
            "round trip failed:\n--- original ---\n{text1}\n--- reparsed ---\n{text2}"
        );
    }

    #[test]
    fn round_trip_gamma() {
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
        let func = builder.finish();

        let registry = IntrinsicRegistry::empty();
        let text1 = format!("{}", func.display_with_registry(&registry));
        let func2 = parse_ir(&text1, test_shape(), &registry).unwrap();
        let text2 = format!("{}", func2.display_with_registry(&registry));

        assert_eq!(
            text1, text2,
            "round trip failed:\n--- original ---\n{text1}\n--- reparsed ---\n{text2}"
        );
    }

    #[test]
    fn round_trip_theta() {
        let mut builder = IrBuilder::new(test_shape());
        {
            let mut rb = builder.root_region();
            let init = rb.const_val(5);
            let one = rb.const_val(1);
            let _ = rb.theta(&[init, one], |bb| {
                let args = bb.region_args(2);
                let counter = args[0];
                let one = args[1];
                let next = bb.binop(IrOp::Sub, counter, one);
                bb.set_results(&[next, next, one]);
            });
            rb.set_results(&[]);
        }
        let func = builder.finish();

        let registry = IntrinsicRegistry::empty();
        let text1 = format!("{}", func.display_with_registry(&registry));
        let func2 = parse_ir(&text1, test_shape(), &registry).unwrap();
        let text2 = format!("{}", func2.display_with_registry(&registry));

        assert_eq!(
            text1, text2,
            "round trip failed:\n--- original ---\n{text1}\n--- reparsed ---\n{text2}"
        );
    }

    #[test]
    fn parse_error_unknown_intrinsic() {
        let input = r#"
lambda @0 (shape: "u8") {
  region {
    args: [%cs, %os]
    n0 = CallIntrinsic(@nonexistent, field_offset=0) [%cs:arg, %os:arg] -> [%cs, %os]
    results: [%cs:n0, %os:n0]
  }
}
"#;

        let registry = IntrinsicRegistry::empty();
        let result = parse_ir(input, test_shape(), &registry);
        let err = match result {
            Err(e) => e,
            Ok(_) => panic!("expected error for unknown intrinsic"),
        };
        assert!(
            err.message.contains("nonexistent"),
            "error was: {}",
            err.message
        );
    }

    #[test]
    fn parse_error_malformed() {
        let input = "this is not valid IR at all";

        let registry = IntrinsicRegistry::empty();
        let result = parse_ir(input, test_shape(), &registry);
        assert!(result.is_err());
    }

    #[test]
    fn round_trip_all_ops() {
        // Test round-tripping all simple ops.
        let mut builder = IrBuilder::new(test_shape());
        {
            let mut rb = builder.root_region();

            // Cursor ops
            rb.bounds_check(8);
            let v0 = rb.read_bytes(4);
            let _v1 = rb.peek_byte();
            rb.advance_cursor(2);
            rb.advance_cursor_by(v0);
            let saved = rb.save_cursor();
            rb.restore_cursor(saved);

            // Output ops
            let val = rb.const_val(42);
            rb.write_to_field(val, 0, Width::W4);
            let _read = rb.read_from_field(0, Width::W4);
            let ptr = rb.save_out_ptr();
            rb.set_out_ptr(ptr);

            // Arithmetic
            let a = rb.const_val(1);
            let b = rb.const_val(2);
            let _ = rb.binop(IrOp::Add, a, b);
            let _ = rb.binop(IrOp::Sub, a, b);
            let _ = rb.binop(IrOp::And, a, b);
            let _ = rb.binop(IrOp::Or, a, b);
            let _ = rb.binop(IrOp::Shr, a, b);
            let _ = rb.binop(IrOp::Shl, a, b);
            let _ = rb.binop(IrOp::Xor, a, b);
            let _ = rb.binop(IrOp::CmpNe, a, b);
            let _ = rb.unary(IrOp::ZigzagDecode { wide: false }, a);
            let _ = rb.unary(
                IrOp::SignExtend {
                    from_width: Width::W4,
                },
                a,
            );

            // Stack ops
            let slot = rb.alloc_slot();
            let _ = rb.slot_addr(slot);

            rb.set_results(&[]);
        }
        let func = builder.finish();

        let registry = IntrinsicRegistry::empty();
        let text1 = format!("{}", func.display_with_registry(&registry));
        let func2 = parse_ir(&text1, test_shape(), &registry).unwrap();
        let text2 = format!("{}", func2.display_with_registry(&registry));

        assert_eq!(
            text1, text2,
            "round trip failed:\n--- original ---\n{text1}\n--- reparsed ---\n{text2}"
        );
    }

    #[test]
    fn round_trip_with_intrinsics() {
        unsafe extern "C" fn test_intrinsic(_ctx: *mut core::ffi::c_void) {}

        let mut registry = IntrinsicRegistry::new();
        registry.register(
            "kajit_read_bool",
            IntrinsicFn(test_intrinsic as *const () as usize),
        );

        let mut builder = IrBuilder::new(test_shape());
        {
            let mut rb = builder.root_region();
            rb.bounds_check(1);
            let func_ptr = registry
                .func_by_name("kajit_read_bool")
                .expect("kajit_read_bool should be in registry");
            rb.call_intrinsic(func_ptr, &[], 0, false);
            rb.set_results(&[]);
        }
        let func = builder.finish();

        let text1 = format!("{}", func.display_with_registry(&registry));
        assert!(
            text1.contains("@kajit_read_bool"),
            "display should use @name, got:\n{text1}"
        );

        let func2 = parse_ir(&text1, test_shape(), &registry).unwrap();
        let text2 = format!("{}", func2.display_with_registry(&registry));

        assert_eq!(
            text1, text2,
            "round trip failed:\n--- original ---\n{text1}\n--- reparsed ---\n{text2}"
        );
    }
}
