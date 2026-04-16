use std::collections::BTreeMap;

use super::*;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StorageError {
    DuplicateId {
        entity: &'static str,
        id: String,
    },
    MissingRef {
        owner: &'static str,
        field: &'static str,
        target: &'static str,
        id: String,
    },
}

impl std::fmt::Display for StorageError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::DuplicateId { entity, id } => {
                write!(f, "duplicate {entity} id {id}")
            }
            Self::MissingRef {
                owner,
                field,
                target,
                id,
            } => {
                write!(f, "{owner}.{field} references missing {target} id {id}")
            }
        }
    }
}

impl std::error::Error for StorageError {}

fn duplicate_id<T: std::fmt::Debug>(entity: &'static str, id: T) -> StorageError {
    StorageError::DuplicateId {
        entity,
        id: format!("{id:?}"),
    }
}

fn missing_ref<T: std::fmt::Debug>(
    owner: &'static str,
    field: &'static str,
    target: &'static str,
    id: T,
) -> StorageError {
    StorageError::MissingRef {
        owner,
        field,
        target,
        id: format!("{id:?}"),
    }
}

fn index_pool<'a, K, V>(
    entity: &'static str,
    values: impl IntoIterator<Item = &'a V>,
    key_of: impl Fn(&V) -> K,
) -> Result<BTreeMap<K, &'a V>, StorageError>
where
    K: Ord + std::marker::Copy + std::fmt::Debug,
{
    let mut out = BTreeMap::new();
    for value in values {
        let id = key_of(value);
        if out.insert(id, value).is_some() {
            return Err(duplicate_id(entity, id));
        }
    }
    Ok(out)
}

pub struct ProgramStorage<'a> {
    graph: &'a Graph,
    program: &'a Program,
    functions_by_id: BTreeMap<FunctionId, &'a Function>,
}

impl<'a> ProgramStorage<'a> {
    pub fn new(graph: &'a Graph, program: &'a Program) -> Result<Self, StorageError> {
        let functions_by_id = index_pool("Function", program.functions.iter(), |function| {
            function.function_id
        })?;
        Ok(Self {
            graph,
            program,
            functions_by_id,
        })
    }

    pub fn program(&self) -> &'a Program {
        self.program
    }

    pub fn functions(&self) -> impl Iterator<Item = &'a Function> + '_ {
        self.program.functions.iter()
    }

    pub fn function(&self, id: FunctionId) -> Option<&'a Function> {
        self.functions_by_id.get(&id).copied()
    }

    pub fn function_storage(
        &self,
        id: FunctionId,
    ) -> Result<Option<FunctionStorage<'a>>, StorageError> {
        self.function(id)
            .map(|function| FunctionStorage::new(self.graph, function))
            .transpose()
    }
}

pub struct FunctionStorage<'a> {
    graph: &'a Graph,
    function: &'a Function,
    blocks_by_id: BTreeMap<BlockId, &'a Block>,
    edges_by_id: BTreeMap<EdgeId, &'a Edge>,
    insts_by_id: BTreeMap<InstId, &'a Inst>,
    terms_by_id: BTreeMap<TermId, &'a Terminator>,
}

impl<'a> FunctionStorage<'a> {
    pub fn new(graph: &'a Graph, function: &'a Function) -> Result<Self, StorageError> {
        Ok(Self {
            graph,
            function,
            blocks_by_id: index_pool("Block", function.blocks.iter(), |block| block.id)?,
            edges_by_id: index_pool("Edge", function.edges.iter(), |edge| edge.id)?,
            insts_by_id: index_pool("Inst", graph.all_inst(), |inst| inst.id)?,
            terms_by_id: index_pool("Terminator", function.terms.iter(), |term| match term {
                Terminator::Branch { id, .. }
                | Terminator::BranchIf { id, .. }
                | Terminator::BranchIfZero { id, .. }
                | Terminator::JumpTable { id, .. }
                | Terminator::Return { id, .. } => *id,
            })?,
        })
    }

    pub fn function(&self) -> &'a Function {
        self.function
    }

    pub fn blocks(&self) -> impl Iterator<Item = &'a Block> + '_ {
        self.function.blocks.iter()
    }

    pub fn edges(&self) -> impl Iterator<Item = &'a Edge> + '_ {
        self.function.edges.iter()
    }

    pub fn insts(&self) -> impl Iterator<Item = &'a Inst> + '_ {
        self.function
            .insts
            .iter()
            .filter_map(|id| self.graph.inst(*id))
    }

    pub fn terms(&self) -> impl Iterator<Item = &'a Terminator> + '_ {
        self.function.terms.iter()
    }

    pub fn entry_block(&self) -> Result<&'a Block, StorageError> {
        self.block(self.function.entry)
            .ok_or_else(|| missing_ref("Function", "entry", "Block", self.function.entry))
    }

    pub fn block(&self, id: BlockId) -> Option<&'a Block> {
        self.blocks_by_id.get(&id).copied()
    }

    pub fn edge(&self, id: EdgeId) -> Option<&'a Edge> {
        self.edges_by_id.get(&id).copied()
    }

    pub fn inst(&self, id: InstId) -> Option<&'a Inst> {
        self.insts_by_id.get(&id).copied()
    }

    pub fn term(&self, id: TermId) -> Option<&'a Terminator> {
        self.terms_by_id.get(&id).copied()
    }

    pub fn block_term(&self, block: &'a Block) -> Result<&'a Terminator, StorageError> {
        self.term(block.term)
            .ok_or_else(|| missing_ref("Block", "term", "Terminator", block.term))
    }

    pub fn block_insts(&self, block: &'a Block) -> Result<Vec<&'a Inst>, StorageError> {
        block
            .insts
            .iter()
            .map(|id| {
                self.inst(*id)
                    .ok_or_else(|| missing_ref("Block", "insts", "Inst", *id))
            })
            .collect()
    }

    pub fn block_preds(&self, block: &'a Block) -> Result<Vec<&'a Edge>, StorageError> {
        block
            .preds
            .iter()
            .map(|id| {
                self.edge(*id)
                    .ok_or_else(|| missing_ref("Block", "preds", "Edge", *id))
            })
            .collect()
    }

    pub fn block_succs(&self, block: &'a Block) -> Result<Vec<&'a Edge>, StorageError> {
        block
            .succs
            .iter()
            .map(|id| {
                self.edge(*id)
                    .ok_or_else(|| missing_ref("Block", "succs", "Edge", *id))
            })
            .collect()
    }

    pub fn edge_from(&self, edge: &'a Edge) -> Result<&'a Block, StorageError> {
        self.block(edge.from)
            .ok_or_else(|| missing_ref("Edge", "from", "Block", edge.from))
    }

    pub fn edge_to(&self, edge: &'a Edge) -> Result<&'a Block, StorageError> {
        self.block(edge.to)
            .ok_or_else(|| missing_ref("Edge", "to", "Block", edge.to))
    }

    pub fn terminator_edges(&self, term: &'a Terminator) -> Result<Vec<&'a Edge>, StorageError> {
        let ids: Vec<EdgeId> = match term {
            Terminator::Branch { edge, .. } => vec![*edge],
            Terminator::BranchIf {
                taken, fallthrough, ..
            }
            | Terminator::BranchIfZero {
                taken, fallthrough, ..
            } => vec![*taken, *fallthrough],
            Terminator::JumpTable {
                default, targets, ..
            } => {
                let mut ids = Vec::with_capacity(targets.len() + 1);
                ids.push(*default);
                ids.extend(targets.iter().copied());
                ids
            }
            Terminator::Return { .. } => Vec::new(),
        };

        ids.into_iter()
            .map(|id| {
                self.edge(id)
                    .ok_or_else(|| missing_ref("Terminator", "edge", "Edge", id))
            })
            .collect()
    }
}
