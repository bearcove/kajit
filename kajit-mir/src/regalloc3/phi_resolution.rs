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

use crate::cfg_mir::{EdgeId, Function, Inst, InstId, Operand, OperandKind, RegClass};
use crate::regalloc3::parallel_copy::{Copy, MoveOp, ParallelCopyResolver};
use kajit_ir::VReg;
use kajit_lir::LinearOp;

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

        if edge.args.is_empty() {
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
/// - Find the block where these copies should go (copy block or predecessor)
/// - Insert Copy instructions before the terminator
pub fn insert_moves_on_edge(func: &mut Function, edge_id: EdgeId, moves: &[MoveOp]) {
    let edge = &func.edges[edge_id.index()];
    let pred_block_id = edge.from;

    // Find where to insert: at end of predecessor, before terminator
    let pred_block = &mut func.blocks[pred_block_id.index()];
    let insert_point = pred_block.insts.len();

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

    // Insert into block's inst list
    let pred_block = &mut func.blocks[pred_block_id.index()];
    for (i, inst_id) in inst_ids.iter().enumerate() {
        pred_block.insts.insert(insert_point + i, *inst_id);
    }
}
