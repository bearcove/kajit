//! SSA validation - check that SSA properties hold.
//!
//! Validates:
//! - Every vreg use has a definition
//! - Every use is dominated by its definition
//! - Phi parameters match incoming edge arguments (count and types)
//! - No dead code creates phantom uses

use std::collections::{HashMap, HashSet};

use kajit_ir::VReg;
use kajit_lir::LinearOp;

use crate::analysis::dominance::DominanceInfo;
use crate::cfg_mir::{BlockId, EdgeId, Function, OperandKind, Terminator};

/// Helper trait to extract destination vreg from LinearOp.
trait LinearOpDst {
    fn dst(&self) -> Option<VReg>;
}

impl LinearOpDst for LinearOp {
    fn dst(&self) -> Option<VReg> {
        match self {
            Self::Copy { dst, .. }
            | Self::Const { dst, .. }
            | Self::BinOp { dst, .. }
            | Self::UnaryOp { dst, .. }
            | Self::LoadFromAddr { dst, .. }
            | Self::SaveCursor { dst, .. }
            | Self::SaveInputEnd { dst, .. }
            | Self::PeekByte { dst, .. }
            | Self::ReadBytes { dst, .. }
            | Self::ReadFromField { dst, .. }
            | Self::SlotAddr { dst, .. }
            | Self::SaveOutPtr { dst, .. }
            | Self::ReadFromSlot { dst, .. }
            | Self::CallPure { dst, .. } => Some(*dst),
            Self::CallIntrinsic { dst, .. } => *dst,
            Self::CallLambda { .. } => None,
            Self::SimdStringScan { .. } => None,
            Self::Label(_)
            | Self::Branch { .. }
            | Self::BranchIf { .. }
            | Self::BranchIfZero { .. }
            | Self::JumpTable { .. }
            | Self::ErrorExit { .. } => None,
            Self::FuncStart { .. } | Self::FuncEnd => None,
            Self::StoreToAddr { .. }
            | Self::WriteToSlot { .. }
            | Self::WriteToField { .. }
            | Self::RestoreCursor { .. }
            | Self::AdvanceCursorBy { .. }
            | Self::SetOutPtr { .. }
            | Self::BoundsCheck { .. }
            | Self::AdvanceCursor { .. }
            | Self::SimdWhitespaceSkip => None,
        }
    }
}

/// Helper trait to extract condition vreg from Terminator.
trait TerminatorCondition {
    fn condition_vreg(&self) -> Option<VReg>;
}

impl TerminatorCondition for Terminator {
    fn condition_vreg(&self) -> Option<VReg> {
        match self {
            Self::BranchIf { cond, .. } | Self::BranchIfZero { cond, .. } => Some(*cond),
            Self::JumpTable { predicate, .. } => Some(*predicate),
            Self::Branch { .. } | Self::Return | Self::ErrorExit { .. } => None,
        }
    }
}

#[derive(Debug, Clone)]
pub enum SsaError {
    /// A vreg is used but never defined
    UseWithoutDef {
        vreg: VReg,
        inst_index: usize,
        block: BlockId,
    },

    /// A vreg use is not dominated by its definition
    UseNotDominated {
        vreg: VReg,
        def_block: BlockId,
        use_block: BlockId,
        inst_index: usize,
    },

    /// A vreg is defined multiple times (violates SSA)
    MultipleDefs {
        vreg: VReg,
        first_def_block: BlockId,
        second_def_block: BlockId,
    },

    /// Edge argument count doesn't match target block parameter count
    PhiArgCountMismatch {
        edge: EdgeId,
        from: BlockId,
        to: BlockId,
        expected: usize,
        got: usize,
    },

    /// Edge argument is missing or incorrect
    PhiArgMissing {
        edge: EdgeId,
        from: BlockId,
        to: BlockId,
        param_index: usize,
        expected_param: VReg,
    },

    /// A vreg is live-in to entry block without being a function parameter
    EntryLivein { vreg: VReg, inst_index: usize },

    /// Edge argument source vreg is not defined anywhere
    EdgeArgSourceUndefined {
        edge: EdgeId,
        from: BlockId,
        to: BlockId,
        param_index: usize,
        source_vreg: VReg,
    },

    /// Edge argument source vreg's definition does not dominate the edge's source block
    EdgeArgSourceNotDominated {
        edge: EdgeId,
        from: BlockId,
        to: BlockId,
        param_index: usize,
        source_vreg: VReg,
        def_block: BlockId,
    },
}

impl std::fmt::Display for SsaError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SsaError::UseWithoutDef {
                vreg,
                inst_index,
                block,
            } => {
                write!(
                    f,
                    "vreg {} used at inst {} in block b{} but never defined",
                    vreg.index(),
                    inst_index,
                    block.index()
                )
            }
            SsaError::UseNotDominated {
                vreg,
                def_block,
                use_block,
                inst_index,
            } => {
                write!(
                    f,
                    "vreg {} used at inst {} in block b{} but defined in non-dominating block b{}",
                    vreg.index(),
                    inst_index,
                    use_block.index(),
                    def_block.index()
                )
            }
            SsaError::PhiArgCountMismatch {
                edge,
                from,
                to,
                expected,
                got,
            } => {
                write!(
                    f,
                    "edge e{} (b{} -> b{}) has {} arguments but target expects {} parameters",
                    edge.index(),
                    from.index(),
                    to.index(),
                    got,
                    expected
                )
            }
            SsaError::PhiArgMissing {
                edge,
                from,
                to,
                param_index,
                expected_param,
            } => {
                write!(
                    f,
                    "edge e{} (b{} -> b{}) is missing argument for parameter {} (v{})",
                    edge.index(),
                    from.index(),
                    to.index(),
                    param_index,
                    expected_param.index()
                )
            }
            SsaError::MultipleDefs {
                vreg,
                first_def_block,
                second_def_block,
            } => {
                write!(
                    f,
                    "vreg {} defined multiple times: first in b{}, second in b{}",
                    vreg.index(),
                    first_def_block.index(),
                    second_def_block.index()
                )
            }
            SsaError::EntryLivein { vreg, inst_index } => {
                write!(
                    f,
                    "vreg {} is live-in to entry block at inst {} (not a function parameter)",
                    vreg.index(),
                    inst_index
                )
            }
            SsaError::EdgeArgSourceUndefined {
                edge,
                from,
                to,
                param_index,
                source_vreg,
            } => {
                write!(
                    f,
                    "edge e{} (b{} -> b{}) param {}: source vreg v{} is not defined anywhere",
                    edge.index(),
                    from.index(),
                    to.index(),
                    param_index,
                    source_vreg.index()
                )
            }
            SsaError::EdgeArgSourceNotDominated {
                edge,
                from,
                to,
                param_index,
                source_vreg,
                def_block,
            } => {
                write!(
                    f,
                    "edge e{} (b{} -> b{}) param {}: source vreg v{} defined in b{} does not dominate b{}",
                    edge.index(),
                    from.index(),
                    to.index(),
                    param_index,
                    source_vreg.index(),
                    def_block.index(),
                    from.index()
                )
            }
        }
    }
}

/// Validate SSA properties for a function.
///
/// Checks:
/// 1. Every vreg use has a definition
/// 2. Every use is dominated by its definition
/// 3. Phi parameters (block parameters) match incoming edge arguments
///
/// Returns all errors found, or Ok(()) if validation passes.
pub fn validate_ssa(func: &Function) -> Result<(), Vec<SsaError>> {
    let mut errors = Vec::new();

    // Step 1: Build def map (vreg -> defining block) and detect multiple defs
    let (def_map, mut def_errors) = build_def_map(func);
    errors.append(&mut def_errors);

    // Debug: check if we're tracking vreg 89
    if std::env::var("KAJIT_DEBUG_VREG_89").is_ok() {
        if let Some(&def_block) = def_map.get(&VReg::new(89)) {
            eprintln!("[SSA] vreg 89 is defined in block b{}", def_block.index());
        } else {
            eprintln!("[SSA] vreg 89 has NO definition!");
        }
    }

    // Step 2: Compute dominance (needed for use-def checking)
    let dom = DominanceInfo::compute(func);

    // Step 3: Check all uses
    check_uses(func, &def_map, &dom, &mut errors);

    // Step 4: Check phi consistency (edge args match block params, sources defined & dominated)
    check_phi_consistency(func, &def_map, &dom, &mut errors);

    // Step 5: Check entry block liveness (no live-ins except parameters)
    check_entry_liveness(func, &mut errors);

    if errors.is_empty() {
        Ok(())
    } else {
        Err(errors)
    }
}

/// Build a map from vreg to the block that defines it.
/// Returns (def_map, errors) where errors contains MultipleDefs violations.
fn build_def_map(func: &Function) -> (HashMap<VReg, BlockId>, Vec<SsaError>) {
    let mut def_map = HashMap::new();
    let mut errors = Vec::new();

    // Function arguments are definitions
    for &arg in &func.data_args {
        // Function args are considered to be defined at entry
        if let Some(&first_def_block) = def_map.get(&arg) {
            errors.push(SsaError::MultipleDefs {
                vreg: arg,
                first_def_block,
                second_def_block: func.entry,
            });
        } else {
            def_map.insert(arg, func.entry);
        }
    }

    for block in &func.blocks {
        // Skip dead blocks
        if block.dead {
            continue;
        }

        // Block parameters are definitions
        for &param in &block.params {
            if let Some(&first_def_block) = def_map.get(&param) {
                errors.push(SsaError::MultipleDefs {
                    vreg: param,
                    first_def_block,
                    second_def_block: block.id,
                });
            } else {
                def_map.insert(param, block.id);
            }
        }

        // Instruction defs
        for &inst_id in &block.insts {
            let inst = &func.insts[inst_id.index()];
            // Look for Def operands
            if let Some(dst) = inst.op.dst() {
                if let Some(&first_def_block) = def_map.get(&dst) {
                    errors.push(SsaError::MultipleDefs {
                        vreg: dst,
                        first_def_block,
                        second_def_block: block.id,
                    });
                } else {
                    def_map.insert(dst, block.id);
                }
            }
        }
    }

    (def_map, errors)
}

/// Check that all vreg uses are valid (defined and dominated).
fn check_uses(
    func: &Function,
    def_map: &HashMap<VReg, BlockId>,
    dom: &DominanceInfo,
    errors: &mut Vec<SsaError>,
) {
    for block in &func.blocks {
        // Skip dead blocks
        if block.dead {
            continue;
        }

        // Check instruction uses
        for (inst_idx, &inst_id) in block.insts.iter().enumerate() {
            let inst = &func.insts[inst_id.index()];
            // Check each Use operand
            for operand in &inst.operands {
                if operand.kind == OperandKind::Use {
                    check_use(operand.vreg, block.id, inst_idx, def_map, dom, errors);
                }
            }
        }

        // Check terminator uses
        let term = &func.terms[block.term.index()];
        if let Some(condition) = term.condition_vreg() {
            // For terminator, we don't have an inst_index, use block.insts.len() as sentinel
            check_use(condition, block.id, block.insts.len(), def_map, dom, errors);
        }
    }
}

/// Check a single vreg use.
fn check_use(
    vreg: VReg,
    use_block: BlockId,
    inst_index: usize,
    def_map: &HashMap<VReg, BlockId>,
    dom: &DominanceInfo,
    errors: &mut Vec<SsaError>,
) {
    // Debug tracking for vreg 89
    if std::env::var("KAJIT_DEBUG_VREG_89").is_ok() && vreg == VReg::new(89) {
        eprintln!(
            "[SSA] vreg 89 used in block b{} at inst {}",
            use_block.index(),
            inst_index
        );
    }

    // Check if vreg has a definition
    let Some(&def_block) = def_map.get(&vreg) else {
        errors.push(SsaError::UseWithoutDef {
            vreg,
            inst_index,
            block: use_block,
        });
        return;
    };

    // Check if definition dominates use
    // A block dominates itself, so same-block defs are OK
    if !dom.dominates(def_block, use_block) {
        if std::env::var("KAJIT_DEBUG_VREG_89").is_ok() && vreg == VReg::new(89) {
            eprintln!(
                "[SSA] vreg 89: def in b{} does NOT dominate use in b{}!",
                def_block.index(),
                use_block.index()
            );
        }
        errors.push(SsaError::UseNotDominated {
            vreg,
            def_block,
            use_block,
            inst_index,
        });
    } else if std::env::var("KAJIT_DEBUG_VREG_89").is_ok() && vreg == VReg::new(89) {
        eprintln!(
            "[SSA] vreg 89: def in b{} dominates use in b{} ✓",
            def_block.index(),
            use_block.index()
        );
    }
}

/// Check that phi parameters (block parameters) match incoming edge arguments.
fn check_phi_consistency(
    func: &Function,
    def_map: &HashMap<VReg, BlockId>,
    dom: &DominanceInfo,
    errors: &mut Vec<SsaError>,
) {
    for block in &func.blocks {
        // Skip dead blocks
        if block.dead {
            continue;
        }

        // Skip blocks without parameters
        if block.params.is_empty() {
            continue;
        }

        // Check each incoming edge
        for &edge_id in &block.preds {
            let edge = &func.edges[edge_id.index()];

            // Check argument count matches parameter count
            if edge.args.len() != block.params.len() {
                errors.push(SsaError::PhiArgCountMismatch {
                    edge: edge_id,
                    from: edge.from,
                    to: block.id,
                    expected: block.params.len(),
                    got: edge.args.len(),
                });
                continue;
            }

            // Check each argument
            for (param_idx, (&param, arg)) in block.params.iter().zip(edge.args.iter()).enumerate()
            {
                // The edge argument's target vreg should match the block parameter
                if arg.target != param {
                    errors.push(SsaError::PhiArgMissing {
                        edge: edge_id,
                        from: edge.from,
                        to: block.id,
                        param_index: param_idx,
                        expected_param: param,
                    });
                }

                // The edge argument's source vreg must be defined
                if let Some(&def_block) = def_map.get(&arg.source) {
                    // Source is defined — check it dominates the edge's source block
                    if !dom.dominates(def_block, edge.from) {
                        errors.push(SsaError::EdgeArgSourceNotDominated {
                            edge: edge_id,
                            from: edge.from,
                            to: block.id,
                            param_index: param_idx,
                            source_vreg: arg.source,
                            def_block,
                        });
                    }
                } else {
                    errors.push(SsaError::EdgeArgSourceUndefined {
                        edge: edge_id,
                        from: edge.from,
                        to: block.id,
                        param_index: param_idx,
                        source_vreg: arg.source,
                    });
                }
            }
        }
    }
}

/// Check that entry block has no live-ins except function parameters.
///
/// A vreg is live-in to entry if it's used in the entry block but not:
/// - A function parameter (in func.data_args)
/// - Defined in the entry block before its use
fn check_entry_liveness(func: &Function, errors: &mut Vec<SsaError>) {
    let debug = std::env::var("KAJIT_DEBUG_ENTRY_LIVEIN").is_ok();
    let entry_block = &func.blocks[func.entry.index()];

    if debug {
        eprintln!(
            "[entry_livein] Checking entry block b{}",
            func.entry.index()
        );
        eprintln!(
            "[entry_livein] Function args: {:?}",
            func.data_args.iter().map(|v| v.index()).collect::<Vec<_>>()
        );
        eprintln!(
            "[entry_livein] Entry params: {:?}",
            entry_block
                .params
                .iter()
                .map(|v| v.index())
                .collect::<Vec<_>>()
        );
    }

    // Entry block should not have parameters (no predecessors to pass them)
    if !entry_block.params.is_empty() {
        for &param in &entry_block.params {
            errors.push(SsaError::EntryLivein {
                vreg: param,
                inst_index: 0, // params are "used" at block entry
            });
        }
    }

    // Entry block should have no predecessors
    if !entry_block.preds.is_empty() {
        if debug {
            eprintln!(
                "[entry_livein] ERROR: entry block has {} predecessors!",
                entry_block.preds.len()
            );
        }
        // This is a structural error - entry block shouldn't have incoming edges
        // Any parameters would be EntryLivein errors
        for &param in &entry_block.params {
            errors.push(SsaError::EntryLivein {
                vreg: param,
                inst_index: 0,
            });
        }
    }

    // Track vregs defined so far as we scan through instructions
    let mut defined: HashSet<VReg> = func.data_args.iter().copied().collect();

    // Check each instruction in order
    for (inst_idx, &inst_id) in entry_block.insts.iter().enumerate() {
        let inst = &func.insts[inst_id.index()];

        // Check uses BEFORE adding this instruction's defs
        for operand in &inst.operands {
            if operand.kind == OperandKind::Use && !defined.contains(&operand.vreg) {
                errors.push(SsaError::EntryLivein {
                    vreg: operand.vreg,
                    inst_index: inst_idx,
                });
            }
        }

        // Add this instruction's defs
        for operand in &inst.operands {
            if operand.kind == OperandKind::Def {
                defined.insert(operand.vreg);
            }
        }
    }

    // Check terminator uses
    let term = &func.terms[entry_block.term.index()];
    if let Some(condition) = term.condition_vreg() {
        if !defined.contains(&condition) {
            errors.push(SsaError::EntryLivein {
                vreg: condition,
                inst_index: entry_block.insts.len(),
            });
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // TODO: Add tests once we have a CFG builder API
    // Test cases:
    // 1. Valid SSA program (should pass)
    // 2. Use without def (should fail)
    // 3. Use not dominated by def (should fail)
    // 4. Edge arg count mismatch (should fail)
    // 5. Edge arg doesn't match block param (should fail)
}
