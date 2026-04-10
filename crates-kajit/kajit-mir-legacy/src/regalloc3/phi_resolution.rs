//! Phi resolution: insert copy instructions for block parameters.
//!
//! ## Problem
//!
//! Block parameters (SSA phis) require values to be passed along edges:
//!
//! ```text
//! block b0:
//!   ...
//!   branch b1  // passes v0, v1, v2 to b1's params
//!
//! block b1 params=[v10, v11, v12]:
//!   ...
//! ```
//!
//! We need to insert: v10 = v0, v11 = v1, v12 = v2
//!
//! ## Where to insert
//!
//! - If edge is non-critical (pred has 1 succ OR succ has 1 pred): insert at end of pred
//! - If edge is critical: insert in the edge's copy block (after critical edge split)
//!
//! ## Parallel copy resolution
//!
//! Multiple copies must execute "simultaneously":
//!   v10 = v0
//!   v11 = v1
//!
//! If v1 == v10, naive ordering breaks! Use parallel_copy module to handle this.

use crate::regalloc3::parallel_copy::{Copy, MoveOp, ParallelCopyResolver};
use kajit_ir::VReg;
use kajit_lir::LinearOp;
use kajit_reprs::mir::{EdgeId, Function, Inst, InstId, Operand, OperandKind, RegClass};

/// Insert copy instructions for block parameter resolution.
///
/// For each edge with arguments, insert Copy instructions to pass values
/// to the successor's block parameters.
///
/// Assumes critical edges have already been split.
pub fn insert_phi_copies(func: &mut Function, temp_vreg: VReg) {
    // Process each edge that passes arguments
    for edge_id in 0..func.edges.len() {
        let edge_id = EdgeId(edge_id as u32);
        let edge = &func.edges[edge_id.index()];

        // Skip dead edges (from critical edge splitting) and edges with no args
        if edge.from.0 == u32::MAX || edge.args.is_empty() {
            continue;
        }

        // Build parallel copies from edge args: target = source
        let copies: Vec<Copy> = edge
            .args
            .iter()
            .map(|arg| Copy {
                dst: arg.target,
                src: arg.source,
            })
            .collect();

        if copies.is_empty() {
            continue;
        }

        // Resolve parallel copies into sequential moves
        let resolver = ParallelCopyResolver::new(copies);
        let moves = resolver.resolve(temp_vreg);

        // Insert copy instructions
        insert_moves_on_edge(func, edge_id, &moves);
    }
}

/// Insert move instructions on an edge.
///
/// Strategy:
/// - If predecessor has only one successor: insert copies at end of predecessor
/// - If successor has only one predecessor: insert copies at start of successor
/// - Otherwise (critical edge): edge must have been split already, panic
pub fn insert_moves_on_edge(func: &mut Function, edge_id: EdgeId, moves: &[MoveOp]) {
    let edge = &func.edges[edge_id.index()];
    let pred_block_id = edge.from;
    let succ_block_id = edge.to;

    let pred_has_one_succ = func.blocks[pred_block_id.index()].succs.len() == 1;
    let succ_has_one_pred = func.blocks[succ_block_id.index()].preds.len() == 1;

    // Determine where to insert copies
    let (target_block_id, insert_point) = if pred_has_one_succ {
        // Safe to insert at end of predecessor (only one outgoing edge)
        let insert_point = func.blocks[pred_block_id.index()].insts.len();
        (pred_block_id, insert_point)
    } else if succ_has_one_pred {
        // Predecessor has multiple successors, but successor has only one predecessor.
        // Insert at START of successor so copies only execute on this edge.
        (succ_block_id, 0)
    } else {
        panic!(
            "insert_moves_on_edge: edge e{} (b{} -> b{}) is critical — should have been split",
            edge_id.0, pred_block_id.0, succ_block_id.0
        );
    };

    // Create new Copy instructions
    let mut new_insts = Vec::new();
    for mov in moves {
        match mov {
            MoveOp::Move { dst, src } => {
                // Create Copy instruction
                let inst = Inst {
                    id: InstId(func.insts.len() as u32 + new_insts.len() as u32),
                    op: LinearOp::Copy {
                        dst: *dst,
                        src: *src,
                    },
                    operands: vec![
                        Operand {
                            vreg: *dst,
                            kind: OperandKind::Def,
                            class: RegClass::Gpr,
                            fixed: None,
                        },
                        Operand {
                            vreg: *src,
                            kind: OperandKind::Use,
                            class: RegClass::Gpr,
                            fixed: None,
                        },
                    ],
                    clobbers: Default::default(),
                };
                new_insts.push(inst);
            }
            MoveOp::Swap { .. } => {
                // Swap not supported yet - should not happen if temp_vreg provided
                panic!("Swap instruction not yet supported in phi resolution");
            }
            MoveOp::MoveToTemp { dst_temp, src } => {
                // Move to temp register (cycle breaking)
                let inst = Inst {
                    id: InstId(func.insts.len() as u32 + new_insts.len() as u32),
                    op: LinearOp::Copy {
                        dst: *dst_temp,
                        src: *src,
                    },
                    operands: vec![
                        Operand {
                            vreg: *dst_temp,
                            kind: OperandKind::Def,
                            class: RegClass::Gpr,
                            fixed: None,
                        },
                        Operand {
                            vreg: *src,
                            kind: OperandKind::Use,
                            class: RegClass::Gpr,
                            fixed: None,
                        },
                    ],
                    clobbers: Default::default(),
                };
                new_insts.push(inst);
            }
        }
    }

    // Add instructions to function's inst list
    let inst_ids: Vec<InstId> = new_insts.iter().map(|inst| inst.id).collect();
    func.insts.extend(new_insts);

    // Insert into target block's inst list
    let target_block = &mut func.blocks[target_block_id.index()];
    for (i, inst_id) in inst_ids.iter().enumerate() {
        target_block.insts.insert(insert_point + i, *inst_id);
    }
}
