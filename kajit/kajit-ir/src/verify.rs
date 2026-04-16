use std::collections::HashMap;
use std::fmt;

use kajit_reprs::ir::{
    DebugScopeId, IrFunc, LambdaId, NodeId, NodeKind, OutputRef, PortKind, PortSource,
    RegionArgRef, RegionId,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum StateProducer {
    Node { node: NodeId, index: u16 },
    RegionArg { region: RegionId, index: u16 },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
struct StateUsage {
    chain_uses: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum VerifyError {
    InvalidLambdaNode {
        lambda: LambdaId,
        node: NodeId,
    },
    LambdaIdMismatch {
        lambda: LambdaId,
        node: NodeId,
        node_lambda_id: LambdaId,
    },
    InvalidRegionReference {
        region: RegionId,
    },
    InvalidNodeReference {
        node: NodeId,
    },
    InvalidDebugScope {
        scope: DebugScopeId,
    },
    RegionParentConflict {
        region: RegionId,
        first_parent: Option<RegionId>,
        second_parent: Option<RegionId>,
    },
    NodeListedInMultipleRegions {
        node: NodeId,
        first_region: RegionId,
        second_region: RegionId,
    },
    NodeRegionMismatch {
        node: NodeId,
        listed_region: RegionId,
        node_region: RegionId,
    },
    NodeInputOutputMissing {
        node: NodeId,
        input_index: u16,
        source: OutputRef,
    },
    NodeInputArgMissing {
        node: NodeId,
        input_index: u16,
        source: RegionArgRef,
    },
    NodeInputKindMismatch {
        node: NodeId,
        input_index: u16,
        expected: PortKind,
        actual: PortKind,
    },
    NodeInputOutOfScope {
        node: NodeId,
        input_index: u16,
        source: PortSource,
    },
    NodeInputTopologicalOrder {
        node: NodeId,
        input_index: u16,
        source: OutputRef,
    },
    RegionResultOutputMissing {
        region: RegionId,
        result_index: u16,
        source: OutputRef,
    },
    RegionResultArgMissing {
        region: RegionId,
        result_index: u16,
        source: RegionArgRef,
    },
    RegionResultKindMismatch {
        region: RegionId,
        result_index: u16,
        expected: PortKind,
        actual: PortKind,
    },
    RegionResultOutOfScope {
        region: RegionId,
        result_index: u16,
        source: PortSource,
    },
    StateChainViolation {
        kind: PortKind,
        producer: PortSource,
        uses: usize,
    },
}

impl fmt::Display for VerifyError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{self:?}")
    }
}

fn node_exists(func: &IrFunc, node: NodeId) -> bool {
    node.index() < func.nodes.len()
}

fn region_exists(func: &IrFunc, region: RegionId) -> bool {
    region.index() < func.regions.len()
}

fn debug_scope_exists(func: &IrFunc, scope: DebugScopeId) -> bool {
    scope.index() < func.debug_scopes.len()
}

fn is_state_kind(kind: PortKind) -> bool {
    kind.is_state()
}

fn in_scope(
    current_region: RegionId,
    source_region: RegionId,
    region_parents: &HashMap<RegionId, Option<RegionId>>,
) -> bool {
    let mut cursor = Some(current_region);
    while let Some(region) = cursor {
        if region == source_region {
            return true;
        }
        cursor = region_parents.get(&region).copied().flatten();
    }
    false
}

type ReachableInfo = (
    HashMap<RegionId, Option<RegionId>>,
    Vec<RegionId>,
    HashMap<NodeId, RegionId>,
);

fn collect_reachable(func: &IrFunc) -> Result<ReachableInfo, VerifyError> {
    let mut region_parents: HashMap<RegionId, Option<RegionId>> = HashMap::new();
    let mut region_order = Vec::new();
    let mut node_regions: HashMap<NodeId, RegionId> = HashMap::new();
    let mut stack = Vec::new();

    for (lambda_index, &lambda_node) in func.lambdas.iter().enumerate() {
        let lambda = LambdaId::new(lambda_index as u32);
        if !node_exists(func, lambda_node) {
            return Err(VerifyError::InvalidLambdaNode {
                lambda,
                node: lambda_node,
            });
        }
        let (body, node_lambda_id) = match &func.nodes[lambda_node].kind {
            NodeKind::Lambda {
                body, lambda_id, ..
            } => (*body, *lambda_id),
            _ => {
                return Err(VerifyError::InvalidLambdaNode {
                    lambda,
                    node: lambda_node,
                });
            }
        };
        if node_lambda_id != lambda {
            return Err(VerifyError::LambdaIdMismatch {
                lambda,
                node: lambda_node,
                node_lambda_id,
            });
        }
        stack.push((body, None));
    }

    while let Some((region, parent)) = stack.pop() {
        if !region_exists(func, region) {
            return Err(VerifyError::InvalidRegionReference { region });
        }

        if let Some(existing_parent) = region_parents.get(&region).copied() {
            if existing_parent != parent {
                return Err(VerifyError::RegionParentConflict {
                    region,
                    first_parent: existing_parent,
                    second_parent: parent,
                });
            }
            continue;
        }

        region_parents.insert(region, parent);
        region_order.push(region);

        for &node_id in &func.regions[region].nodes {
            if !node_exists(func, node_id) {
                return Err(VerifyError::InvalidNodeReference { node: node_id });
            }
            if let Some(existing_region) = node_regions.insert(node_id, region) {
                return Err(VerifyError::NodeListedInMultipleRegions {
                    node: node_id,
                    first_region: existing_region,
                    second_region: region,
                });
            }

            let node = &func.nodes[node_id];
            if node.region != region {
                return Err(VerifyError::NodeRegionMismatch {
                    node: node_id,
                    listed_region: region,
                    node_region: node.region,
                });
            }

            match &node.kind {
                NodeKind::Gamma { regions } => {
                    for &sub in regions {
                        stack.push((sub, Some(region)));
                    }
                }
                NodeKind::Theta { body, .. } => stack.push((*body, Some(region))),
                NodeKind::Simple(_) | NodeKind::Apply { .. } => {}
                NodeKind::Lambda { .. } => {
                    return Err(VerifyError::NodeRegionMismatch {
                        node: node_id,
                        listed_region: region,
                        node_region: node.region,
                    });
                }
            }
        }
    }

    Ok((region_parents, region_order, node_regions))
}

fn check_node_source(
    func: &IrFunc,
    source: OutputRef,
    expected_kind: PortKind,
) -> Result<PortKind, ()> {
    if !node_exists(func, source.node) {
        return Err(());
    }
    let source_node = &func.nodes[source.node];
    let Some(output) = source_node.outputs.get(source.index as usize) else {
        return Err(());
    };
    if output.kind != expected_kind {
        return Ok(output.kind);
    }
    Ok(expected_kind)
}

fn check_arg_source(
    func: &IrFunc,
    source: RegionArgRef,
    expected_kind: PortKind,
) -> Result<PortKind, ()> {
    if !region_exists(func, source.region) {
        return Err(());
    }
    let region = &func.regions[source.region];
    // Check that this arg exists in the region
    if !region.args.contains(&source.arg) {
        return Err(());
    }
    let arg = &func.region_args[source.arg];
    if arg.kind != expected_kind {
        return Ok(arg.kind);
    }
    Ok(expected_kind)
}

fn state_source(source: PortSource, func: &IrFunc) -> StateProducer {
    match source {
        PortSource::Node(out) => StateProducer::Node {
            node: out.node,
            index: out.index,
        },
        PortSource::RegionArg(aref) => {
            // Find position of this arg in the region for display
            let index = func.regions[aref.region]
                .args
                .iter()
                .position(|&id| id == aref.arg)
                .unwrap_or(0) as u16;
            StateProducer::RegionArg {
                region: aref.region,
                index,
            }
        }
    }
}

pub fn verify(func: &IrFunc) -> Result<(), VerifyError> {
    if !debug_scope_exists(func, func.root_debug_scope) {
        return Err(VerifyError::InvalidDebugScope {
            scope: func.root_debug_scope,
        });
    }
    for (_, scope) in func.debug_scopes.iter() {
        if let Some(parent) = scope.parent
            && !debug_scope_exists(func, parent)
        {
            return Err(VerifyError::InvalidDebugScope { scope: parent });
        }
    }

    let (region_parents, region_order, node_regions) = collect_reachable(func)?;

    for &region_id in &region_order {
        let region = &func.regions[region_id];
        if !debug_scope_exists(func, region.debug_scope) {
            return Err(VerifyError::InvalidDebugScope {
                scope: region.debug_scope,
            });
        }
        let mut positions: HashMap<NodeId, usize> = HashMap::with_capacity(region.nodes.len());
        for (idx, &node_id) in region.nodes.iter().enumerate() {
            positions.insert(node_id, idx);
        }

        for (node_pos, &node_id) in region.nodes.iter().enumerate() {
            let node = &func.nodes[node_id];
            if !debug_scope_exists(func, node.debug_scope) {
                return Err(VerifyError::InvalidDebugScope {
                    scope: node.debug_scope,
                });
            }
            for output in &node.outputs {
                if !debug_scope_exists(func, output.debug_scope) {
                    return Err(VerifyError::InvalidDebugScope {
                        scope: output.debug_scope,
                    });
                }
            }
            for (input_index, input) in node.inputs.iter().enumerate() {
                match input.source {
                    PortSource::Node(source) => {
                        let kind = check_node_source(func, source, input.kind).map_err(|_| {
                            VerifyError::NodeInputOutputMissing {
                                node: node_id,
                                input_index: input_index as u16,
                                source,
                            }
                        })?;
                        if kind != input.kind {
                            return Err(VerifyError::NodeInputKindMismatch {
                                node: node_id,
                                input_index: input_index as u16,
                                expected: input.kind,
                                actual: kind,
                            });
                        }

                        let source_region = func.nodes[source.node].region;
                        if !in_scope(region_id, source_region, &region_parents) {
                            return Err(VerifyError::NodeInputOutOfScope {
                                node: node_id,
                                input_index: input_index as u16,
                                source: input.source,
                            });
                        }
                        if node_regions.get(&source.node).copied() != Some(source_region) {
                            return Err(VerifyError::NodeInputOutOfScope {
                                node: node_id,
                                input_index: input_index as u16,
                                source: input.source,
                            });
                        }

                        if source_region == region_id {
                            let Some(&source_pos) = positions.get(&source.node) else {
                                return Err(VerifyError::NodeInputOutOfScope {
                                    node: node_id,
                                    input_index: input_index as u16,
                                    source: input.source,
                                });
                            };
                            if source_pos >= node_pos {
                                return Err(VerifyError::NodeInputTopologicalOrder {
                                    node: node_id,
                                    input_index: input_index as u16,
                                    source,
                                });
                            }
                        }
                    }
                    PortSource::RegionArg(source) => {
                        let kind = check_arg_source(func, source, input.kind).map_err(|_| {
                            VerifyError::NodeInputArgMissing {
                                node: node_id,
                                input_index: input_index as u16,
                                source,
                            }
                        })?;
                        if kind != input.kind {
                            return Err(VerifyError::NodeInputKindMismatch {
                                node: node_id,
                                input_index: input_index as u16,
                                expected: input.kind,
                                actual: kind,
                            });
                        }

                        if !in_scope(region_id, source.region, &region_parents) {
                            return Err(VerifyError::NodeInputOutOfScope {
                                node: node_id,
                                input_index: input_index as u16,
                                source: input.source,
                            });
                        }
                    }
                }
            }
        }

        for (result_index, &result_id) in region.results.iter().enumerate() {
            let result = &func.region_results[result_id];
            match result.source {
                PortSource::Node(source) => {
                    let kind = check_node_source(func, source, result.kind).map_err(|_| {
                        VerifyError::RegionResultOutputMissing {
                            region: region_id,
                            result_index: result_index as u16,
                            source,
                        }
                    })?;
                    if kind != result.kind {
                        return Err(VerifyError::RegionResultKindMismatch {
                            region: region_id,
                            result_index: result_index as u16,
                            expected: result.kind,
                            actual: kind,
                        });
                    }

                    let source_region = func.nodes[source.node].region;
                    if !in_scope(region_id, source_region, &region_parents) {
                        return Err(VerifyError::RegionResultOutOfScope {
                            region: region_id,
                            result_index: result_index as u16,
                            source: result.source,
                        });
                    }
                    if node_regions.get(&source.node).copied() != Some(source_region) {
                        return Err(VerifyError::RegionResultOutOfScope {
                            region: region_id,
                            result_index: result_index as u16,
                            source: result.source,
                        });
                    }
                }
                PortSource::RegionArg(source) => {
                    let kind = check_arg_source(func, source, result.kind).map_err(|_| {
                        VerifyError::RegionResultArgMissing {
                            region: region_id,
                            result_index: result_index as u16,
                            source,
                        }
                    })?;
                    if kind != result.kind {
                        return Err(VerifyError::RegionResultKindMismatch {
                            region: region_id,
                            result_index: result_index as u16,
                            expected: result.kind,
                            actual: kind,
                        });
                    }
                    if !in_scope(region_id, source.region, &region_parents) {
                        return Err(VerifyError::RegionResultOutOfScope {
                            region: region_id,
                            result_index: result_index as u16,
                            source: result.source,
                        });
                    }
                }
            }
        }
    }

    // State chain validation: track uses of each state producer (single implicit domain)
    let mut state_uses: HashMap<StateProducer, StateUsage> = HashMap::new();
    for &region_id in &region_order {
        let region = &func.regions[region_id];
        for &node_id in &region.nodes {
            let node = &func.nodes[node_id];
            for input in &node.inputs {
                if input.kind.is_state() {
                    let producer = state_source(input.source, func);
                    let usage = state_uses.entry(producer).or_default();
                    usage.chain_uses += 1;
                }
            }
        }
        for &result_id in &region.results {
            let result = &func.region_results[result_id];
            if result.kind.is_state() {
                let producer = state_source(result.source, func);
                state_uses.entry(producer).or_default().chain_uses += 1;
            }
        }
    }

    for &region_id in &region_order {
        let region = &func.regions[region_id];
        for &arg_id in &region.args {
            let arg = &func.region_args[arg_id];
            if !is_state_kind(arg.kind) {
                continue;
            }
            let producer = PortSource::RegionArg(RegionArgRef {
                region: region_id,
                arg: arg_id,
            });
            let usage = state_uses
                .get(&state_source(producer, func))
                .copied()
                .unwrap_or_default();
            if usage.chain_uses != 1 {
                return Err(VerifyError::StateChainViolation {
                    kind: arg.kind,
                    producer,
                    uses: usage.chain_uses,
                });
            }
        }

        for &node_id in &region.nodes {
            let node = &func.nodes[node_id];
            for (output_index, output) in node.outputs.iter().enumerate() {
                if !is_state_kind(output.kind) {
                    continue;
                }
                let producer = PortSource::Node(OutputRef {
                    node: node_id,
                    index: output_index as u16,
                });
                let usage = state_uses
                    .get(&state_source(producer, func))
                    .copied()
                    .unwrap_or_default();
                if usage.chain_uses != 1 {
                    return Err(VerifyError::StateChainViolation {
                        kind: output.kind,
                        producer,
                        uses: usage.chain_uses,
                    });
                }
            }
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{IrBuilder, IrOp};

    #[test]
    fn verify_accepts_builder_ir() {
        let mut builder = IrBuilder::new("u8", 0);
        {
            let mut rb = builder.root_region();
            let byte = rb.const_val(1);
            let _ = rb.gamma(byte, &[], 2, |branch_idx, branch| match branch_idx {
                0 => {
                    let c = branch.const_val(11);
                    branch.set_results(&[c]);
                }
                1 => {
                    let c = branch.const_val(22);
                    branch.set_results(&[c]);
                }
                _ => unreachable!(),
            });
            rb.set_results(&[]);
        }
        let func = builder.finish();
        assert!(verify(&func).is_ok());
    }

    #[test]
    fn verify_rejects_topological_violations() {
        let mut builder = IrBuilder::new("u8", 0);
        {
            let mut rb = builder.root_region();
            let c = rb.const_val(7);
            let _ = rb.binop(IrOp::Add, c, c);
            rb.set_results(&[]);
        }
        let mut func = builder.finish();
        let root = func.root_body();
        func.regions[root].nodes.swap(0, 1);

        let err = verify(&func).expect_err("verifier should reject non-topological region order");
        assert!(matches!(err, VerifyError::NodeInputTopologicalOrder { .. }));
    }

    #[test]
    fn verify_rejects_state_forks() {
        let mut builder = IrBuilder::new("u8", 0);
        let slot = builder.alloc_slot();
        {
            let mut rb = builder.root_region();
            let _ = rb.read_from_slot(slot);
            let _ = rb.read_from_slot(slot);
            rb.set_results(&[]);
        }
        let mut func = builder.finish();
        let root = func.root_body();
        let first = func.regions[root].nodes[0];
        let result_id = func.regions[root].results[0];
        // Point result at first read_from_slot's memory state output (index 1).
        // This creates a fork: both the second read_from_slot and this result
        // use the first read_from_slot's memory state output.
        func.region_results[result_id].source = PortSource::Node(OutputRef {
            node: first,
            index: 1,
        });

        let err = verify(&func).expect_err("verifier should reject state forks");
        assert!(
            matches!(err, VerifyError::StateChainViolation { uses: 2, .. }),
            "expected StateChainViolation with uses: 2, got: {err:?}"
        );
    }
}
