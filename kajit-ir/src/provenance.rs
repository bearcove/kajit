//! PortSource provenance tracer.
//!
//! Given a `PortSource`, traces the full chain of how it resolves through
//! gamma/theta/lambda region args. Returns a structured trace that answers
//! "why can't this be folded to a constant?" at a glance.

use std::fmt;

use crate::{IrFunc, IrOp, NodeId, NodeKind, PortSource, RegionArgRef, RegionId};

/// One step in a provenance chain.
#[derive(Debug)]
pub enum ProvenanceStep {
    /// Reached a Const node — terminal, value is known.
    ConstNode { node: NodeId, value: u64 },
    /// Reached a non-constant node (Mul, Add, LoadFromAddr, etc.) — terminal.
    NonConstNode { node: NodeId, op_name: String },
    /// Traced through a gamma branch argument to the gamma's input.
    GammaArg {
        owner: NodeId,
        region: RegionId,
        arg_index: usize,
        input_index: usize,
    },
    /// Traced through a theta body argument to the theta's input.
    ThetaArg {
        owner: NodeId,
        body: RegionId,
        arg_index: usize,
        loop_invariant: bool,
    },
    /// Reached a lambda body argument — dead end (args come from apply sites).
    LambdaArg { owner: NodeId, arg_index: usize },
    /// Could not find the owner of a region.
    OrphanedRegionArg { region: RegionId },
    /// Theta arg is not loop-invariant — dead end.
    NotLoopInvariant { owner: NodeId, arg_index: usize },
}

/// Complete provenance trace for a PortSource.
#[derive(Debug)]
pub struct ProvenanceTrace {
    pub steps: Vec<ProvenanceStep>,
    /// If the trace resolved to a constant, its value.
    pub resolved_value: Option<u64>,
}

/// Trace the provenance of a PortSource through the RVSDG graph.
///
/// Follows the same logic as `const_fold::resolve_to_constant` but records
/// each step instead of just returning the final value.
pub fn trace_provenance(func: &IrFunc, source: &PortSource) -> ProvenanceTrace {
    let mut steps = Vec::new();
    let value = trace_recursive(func, source, &mut steps, 0);
    ProvenanceTrace {
        steps,
        resolved_value: value,
    }
}

const MAX_DEPTH: usize = 64;

fn trace_recursive(
    func: &IrFunc,
    source: &PortSource,
    steps: &mut Vec<ProvenanceStep>,
    depth: usize,
) -> Option<u64> {
    if depth > MAX_DEPTH {
        return None;
    }

    match source {
        PortSource::Node(out_ref) => match &func.nodes[out_ref.node].kind {
            NodeKind::Simple(IrOp::Const { value }) => {
                steps.push(ProvenanceStep::ConstNode {
                    node: out_ref.node,
                    value: *value,
                });
                Some(*value)
            }
            NodeKind::Simple(op) => {
                steps.push(ProvenanceStep::NonConstNode {
                    node: out_ref.node,
                    op_name: format!("{:?}", op),
                });
                None
            }
            NodeKind::Gamma { .. } => {
                steps.push(ProvenanceStep::NonConstNode {
                    node: out_ref.node,
                    op_name: format!("gamma output[{}]", out_ref.index),
                });
                None
            }
            NodeKind::Theta { .. } => {
                steps.push(ProvenanceStep::NonConstNode {
                    node: out_ref.node,
                    op_name: format!("theta output[{}]", out_ref.index),
                });
                None
            }
            _ => {
                steps.push(ProvenanceStep::NonConstNode {
                    node: out_ref.node,
                    op_name: "unknown".to_string(),
                });
                None
            }
        },
        PortSource::RegionArg(arg_ref) => {
            let arg_index = func.regions[arg_ref.region]
                .args
                .iter()
                .position(|a| *a == arg_ref.arg)?;

            let Some(owner) = func.find_region_owner(arg_ref.region) else {
                steps.push(ProvenanceStep::OrphanedRegionArg {
                    region: arg_ref.region,
                });
                return None;
            };

            match &func.nodes[owner].kind {
                NodeKind::Gamma { .. } => {
                    let input_index = arg_index + 1; // skip predicate
                    steps.push(ProvenanceStep::GammaArg {
                        owner,
                        region: arg_ref.region,
                        arg_index,
                        input_index,
                    });
                    let input = func.nodes[owner].inputs.get(input_index)?;
                    trace_recursive(func, &input.source, steps, depth + 1)
                }
                NodeKind::Theta { body, .. } => {
                    let input = func.nodes[owner].inputs.get(arg_index)?;
                    let body_region = *body;

                    // Check loop-invariance: result[arg_index+1] passes back arg
                    let result_index = arg_index + 1; // skip predicate
                    let is_loop_invariant = func.regions[body_region]
                        .results
                        .get(result_index)
                        .is_some_and(|&result_id| {
                            matches!(
                                func.region_results[result_id].source,
                                PortSource::RegionArg(RegionArgRef { region, arg })
                                    if region == body_region && arg == arg_ref.arg
                            )
                        });

                    steps.push(ProvenanceStep::ThetaArg {
                        owner,
                        body: body_region,
                        arg_index,
                        loop_invariant: is_loop_invariant,
                    });

                    if is_loop_invariant {
                        trace_recursive(func, &input.source, steps, depth + 1)
                    } else {
                        steps.push(ProvenanceStep::NotLoopInvariant { owner, arg_index });
                        None
                    }
                }
                NodeKind::Lambda { .. } => {
                    steps.push(ProvenanceStep::LambdaArg { owner, arg_index });
                    None
                }
                _ => None,
            }
        }
    }
}

impl fmt::Display for ProvenanceTrace {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for (i, step) in self.steps.iter().enumerate() {
            let indent = "  ".repeat(i);
            match step {
                ProvenanceStep::ConstNode { node, value } => {
                    writeln!(f, "{indent}→ n{} = Const(0x{:x})", node.index(), value)?;
                }
                ProvenanceStep::NonConstNode { node, op_name } => {
                    writeln!(f, "{indent}→ n{} = {op_name} (not constant)", node.index())?;
                }
                ProvenanceStep::GammaArg {
                    owner,
                    arg_index,
                    input_index,
                    ..
                } => {
                    writeln!(
                        f,
                        "{indent}→ gamma n{} arg[{arg_index}] → input[{input_index}]",
                        owner.index()
                    )?;
                }
                ProvenanceStep::ThetaArg {
                    owner,
                    arg_index,
                    loop_invariant,
                    ..
                } => {
                    let inv = if *loop_invariant {
                        "loop-invariant"
                    } else {
                        "NOT loop-invariant"
                    };
                    writeln!(
                        f,
                        "{indent}→ theta n{} arg[{arg_index}] ({inv})",
                        owner.index()
                    )?;
                }
                ProvenanceStep::LambdaArg { owner, arg_index } => {
                    writeln!(
                        f,
                        "{indent}→ lambda n{} arg[{arg_index}] (dead end: apply site)",
                        owner.index()
                    )?;
                }
                ProvenanceStep::OrphanedRegionArg { region } => {
                    writeln!(f, "{indent}→ orphaned region arg (region {:?})", region)?;
                }
                ProvenanceStep::NotLoopInvariant { owner, arg_index } => {
                    writeln!(
                        f,
                        "{indent}→ theta n{} arg[{arg_index}] is NOT loop-invariant (dead end)",
                        owner.index()
                    )?;
                }
            }
        }
        if let Some(val) = self.resolved_value {
            writeln!(f, "  Resolved to: 0x{val:x}")?;
        } else {
            writeln!(f, "  Could not resolve to constant")?;
        }
        Ok(())
    }
}
