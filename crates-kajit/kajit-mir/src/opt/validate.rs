//! CFG validation - check that the CFG is well-formed and consistent.

use crate::ir::{BlockId, EdgeId, Function};

#[derive(Debug)]
pub enum ValidationError {
    /// Block ID out of bounds
    BlockIdOutOfBounds {
        block_id: BlockId,
        num_blocks: usize,
    },

    /// Edge ID out of bounds
    EdgeIdOutOfBounds { edge_id: EdgeId, num_edges: usize },

    /// Edge.id doesn't match its position in func.edges
    EdgeIdMismatch {
        edge: EdgeId,
        position: usize,
        edge_id_field: EdgeId,
    },

    /// Edge.from doesn't exist
    EdgeFromInvalid { edge: EdgeId, from: BlockId },

    /// Edge.to doesn't exist
    EdgeToInvalid { edge: EdgeId, to: BlockId },

    /// Block lists edge in preds but edge.to != block
    PredEdgeMismatch {
        block: BlockId,
        edge: EdgeId,
        edge_to: BlockId,
    },

    /// Block lists edge in succs but edge.from != block
    SuccEdgeMismatch {
        block: BlockId,
        edge: EdgeId,
        edge_from: BlockId,
    },

    /// Edge.to lists edge in preds but block doesn't list it
    MissingPred { edge: EdgeId, target: BlockId },

    /// Edge.from lists edge in succs but block doesn't list it
    MissingSucc { edge: EdgeId, source: BlockId },

    /// Terminator references edge but block doesn't list it in succs
    TerminatorEdgeNotInSuccs { block: BlockId, edge: EdgeId },

    /// Block.succs contains edge not referenced by terminator
    SuccNotInTerminator { block: BlockId, edge: EdgeId },

    /// Dead block still has predecessors
    DeadBlockHasPreds { block: BlockId },

    /// Dead block still has successors
    DeadBlockHasSuccs { block: BlockId },

    /// Entry block doesn't exist
    EntryBlockInvalid { entry: BlockId },
}

impl std::fmt::Display for ValidationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ValidationError::BlockIdOutOfBounds {
                block_id,
                num_blocks,
            } => {
                write!(
                    f,
                    "block b{} out of bounds (num_blocks={})",
                    block_id.index(),
                    num_blocks
                )
            }
            ValidationError::EdgeIdOutOfBounds { edge_id, num_edges } => {
                write!(
                    f,
                    "edge e{} out of bounds (num_edges={})",
                    edge_id.index(),
                    num_edges
                )
            }
            ValidationError::EdgeIdMismatch {
                edge,
                position,
                edge_id_field,
            } => {
                write!(
                    f,
                    "edge at position {} has id e{} but should be e{}",
                    position,
                    edge_id_field.index(),
                    edge.index()
                )
            }
            ValidationError::EdgeFromInvalid { edge, from } => {
                write!(
                    f,
                    "edge e{} has invalid from block b{}",
                    edge.index(),
                    from.index()
                )
            }
            ValidationError::EdgeToInvalid { edge, to } => {
                write!(
                    f,
                    "edge e{} has invalid to block b{}",
                    edge.index(),
                    to.index()
                )
            }
            ValidationError::PredEdgeMismatch {
                block,
                edge,
                edge_to,
            } => {
                write!(
                    f,
                    "block b{} lists e{} in preds but edge.to is b{}",
                    block.index(),
                    edge.index(),
                    edge_to.index()
                )
            }
            ValidationError::SuccEdgeMismatch {
                block,
                edge,
                edge_from,
            } => {
                write!(
                    f,
                    "block b{} lists e{} in succs but edge.from is b{}",
                    block.index(),
                    edge.index(),
                    edge_from.index()
                )
            }
            ValidationError::MissingPred { edge, target } => {
                write!(
                    f,
                    "edge e{} points to b{} but b{} doesn't list e{} in preds",
                    edge.index(),
                    target.index(),
                    target.index(),
                    edge.index()
                )
            }
            ValidationError::MissingSucc { edge, source } => {
                write!(
                    f,
                    "edge e{} from b{} but b{} doesn't list e{} in succs",
                    edge.index(),
                    source.index(),
                    source.index(),
                    edge.index()
                )
            }
            ValidationError::TerminatorEdgeNotInSuccs { block, edge } => {
                write!(
                    f,
                    "block b{} terminator references e{} but it's not in succs",
                    block.index(),
                    edge.index()
                )
            }
            ValidationError::SuccNotInTerminator { block, edge } => {
                write!(
                    f,
                    "block b{} succs contains e{} but terminator doesn't reference it",
                    block.index(),
                    edge.index()
                )
            }
            ValidationError::DeadBlockHasPreds { block } => {
                write!(f, "dead block b{} still has predecessors", block.index())
            }
            ValidationError::DeadBlockHasSuccs { block } => {
                write!(f, "dead block b{} still has successors", block.index())
            }
            ValidationError::EntryBlockInvalid { entry } => {
                write!(f, "entry block b{} doesn't exist", entry.index())
            }
        }
    }
}

pub fn validate_cfg(func: &Function) -> Result<(), Vec<ValidationError>> {
    let mut errors = Vec::new();

    // Check entry block exists
    if func.entry.index() >= func.blocks.len() {
        errors.push(ValidationError::EntryBlockInvalid { entry: func.entry });
    }

    // Check all edges have correct IDs and valid from/to
    for (pos, edge) in func.edges.iter().enumerate() {
        let edge_id = EdgeId::new(pos as u32);

        // Edge ID must match position
        if edge.id != edge_id {
            errors.push(ValidationError::EdgeIdMismatch {
                edge: edge_id,
                position: pos,
                edge_id_field: edge.id,
            });
        }

        // Edge.from must be valid
        if edge.from.index() >= func.blocks.len() {
            errors.push(ValidationError::EdgeFromInvalid {
                edge: edge_id,
                from: edge.from,
            });
        }

        // Edge.to must be valid
        if edge.to.index() >= func.blocks.len() {
            errors.push(ValidationError::EdgeToInvalid {
                edge: edge_id,
                to: edge.to,
            });
        }
    }

    // Check all blocks
    for block in &func.blocks {
        // Skip validation of dead blocks (except check they have no preds/succs)
        if block.dead {
            if !block.preds.is_empty() {
                errors.push(ValidationError::DeadBlockHasPreds { block: block.id });
            }
            if !block.succs.is_empty() {
                errors.push(ValidationError::DeadBlockHasSuccs { block: block.id });
            }
            continue;
        }

        // Check block ID is in bounds
        if block.id.index() >= func.blocks.len() {
            errors.push(ValidationError::BlockIdOutOfBounds {
                block_id: block.id,
                num_blocks: func.blocks.len(),
            });
        }

        // Check all pred edges
        for &pred_edge in &block.preds {
            if pred_edge.index() >= func.edges.len() {
                errors.push(ValidationError::EdgeIdOutOfBounds {
                    edge_id: pred_edge,
                    num_edges: func.edges.len(),
                });
                continue;
            }

            let edge = &func.edges[pred_edge.index()];
            if edge.to != block.id {
                errors.push(ValidationError::PredEdgeMismatch {
                    block: block.id,
                    edge: pred_edge,
                    edge_to: edge.to,
                });
            }
        }

        // Check all succ edges
        for &succ_edge in &block.succs {
            if succ_edge.index() >= func.edges.len() {
                errors.push(ValidationError::EdgeIdOutOfBounds {
                    edge_id: succ_edge,
                    num_edges: func.edges.len(),
                });
                continue;
            }

            let edge = &func.edges[succ_edge.index()];
            if edge.from != block.id {
                errors.push(ValidationError::SuccEdgeMismatch {
                    block: block.id,
                    edge: succ_edge,
                    edge_from: edge.from,
                });
            }
        }

        // Check terminator edges match succs
        let term = &func.terms[block.term.index()];
        let term_edges = term.successor_edges();

        for &term_edge in &term_edges {
            if !block.succs.contains(&term_edge) {
                errors.push(ValidationError::TerminatorEdgeNotInSuccs {
                    block: block.id,
                    edge: term_edge,
                });
            }
        }

        for &succ_edge in &block.succs {
            if !term_edges.contains(&succ_edge) {
                errors.push(ValidationError::SuccNotInTerminator {
                    block: block.id,
                    edge: succ_edge,
                });
            }
        }
    }

    // Check all edges are referenced by their source and target blocks
    for (pos, edge) in func.edges.iter().enumerate() {
        let edge_id = EdgeId::new(pos as u32);

        // Skip if from/to are out of bounds (already reported above)
        if edge.from.index() >= func.blocks.len() || edge.to.index() >= func.blocks.len() {
            continue;
        }

        let from_block = &func.blocks[edge.from.index()];
        let to_block = &func.blocks[edge.to.index()];

        // Skip dead blocks
        if from_block.dead || to_block.dead {
            continue;
        }

        if !from_block.succs.contains(&edge_id) {
            errors.push(ValidationError::MissingSucc {
                edge: edge_id,
                source: edge.from,
            });
        }

        if !to_block.preds.contains(&edge_id) {
            errors.push(ValidationError::MissingPred {
                edge: edge_id,
                target: edge.to,
            });
        }
    }

    if errors.is_empty() {
        Ok(())
    } else {
        Err(errors)
    }
}
