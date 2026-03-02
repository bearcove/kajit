//! Linearizer: converts the RVSDG into a flat instruction sequence.
//!
//! The RVSDG is a tree of regions and nodes. The linearizer walks this tree,
//! topologically sorts each region's nodes, and emits a flat `Vec<LinearOp>`
//! with explicit labels and branches for control flow (gamma/theta).

use std::collections::{HashMap, HashSet, VecDeque};
use std::fmt;

use kajit_ir::ErrorCode;
use kajit_ir::{
    Id, IrFunc, IrOp, LambdaId, Node, NodeId, NodeKind, PortKind, PortSource, RegionId, SlotId,
    VReg, Width,
};

// ─── Label ID ────────────────────────────────────────────────────────────────

/// Marker type for label IDs.
pub struct LabelMarker;
/// A label in the linear instruction sequence.
pub type LabelId = Id<LabelMarker>;

// ─── BinOpKind / UnaryOpKind ─────────────────────────────────────────────────

/// Binary operation kind for linear IR.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BinOpKind {
    Add,
    Sub,
    And,
    Or,
    Shr,
    Shl,
    Xor,
    CmpNe,
}

/// Unary operation kind for linear IR.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UnaryOpKind {
    ZigzagDecode { wide: bool },
    SignExtend { from_width: Width },
}

// ─── IntrinsicFn re-export ───────────────────────────────────────────────────

use kajit_ir::IntrinsicFn;

// ─── LinearOp ────────────────────────────────────────────────────────────────

/// A single instruction in the linearized IR.
///
/// Each variant corresponds to an RVSDG `IrOp`, but flattened into a linear
/// sequence with explicit labels and branches for control flow.
// r[impl ir.linearize]
#[derive(Debug, Clone)]
pub enum LinearOp {
    // ── Values ──
    Const {
        dst: VReg,
        value: u64,
    },
    BinOp {
        op: BinOpKind,
        dst: VReg,
        lhs: VReg,
        rhs: VReg,
    },
    UnaryOp {
        op: UnaryOpKind,
        dst: VReg,
        src: VReg,
    },
    /// Copy a value between virtual registers (for gamma merge / theta feedback).
    Copy {
        dst: VReg,
        src: VReg,
    },

    // ── Cursor ──
    BoundsCheck {
        count: u32,
    },
    ReadBytes {
        dst: VReg,
        count: u32,
    },
    PeekByte {
        dst: VReg,
    },
    AdvanceCursor {
        count: u32,
    },
    AdvanceCursorBy {
        src: VReg,
    },
    SaveCursor {
        dst: VReg,
    },
    RestoreCursor {
        src: VReg,
    },

    // ── Output ──
    WriteToField {
        src: VReg,
        offset: u32,
        width: Width,
    },
    ReadFromField {
        dst: VReg,
        offset: u32,
        width: Width,
    },
    SaveOutPtr {
        dst: VReg,
    },
    SetOutPtr {
        src: VReg,
    },

    // ── Stack ──
    SlotAddr {
        dst: VReg,
        slot: SlotId,
    },
    WriteToSlot {
        slot: SlotId,
        src: VReg,
    },
    ReadFromSlot {
        dst: VReg,
        slot: SlotId,
    },

    // ── Calls ──
    CallIntrinsic {
        func: IntrinsicFn,
        args: Vec<VReg>,
        dst: Option<VReg>,
        field_offset: u32,
    },
    CallPure {
        func: IntrinsicFn,
        args: Vec<VReg>,
        dst: VReg,
    },

    // ── Control flow ──
    Label(LabelId),
    Branch(LabelId),
    /// Branch if condition is nonzero.
    BranchIf {
        cond: VReg,
        target: LabelId,
    },
    /// Branch if condition is zero.
    BranchIfZero {
        cond: VReg,
        target: LabelId,
    },
    /// Jump table: jump to `labels[predicate]`, or to `default` if out of range.
    JumpTable {
        predicate: VReg,
        labels: Vec<LabelId>,
        default: LabelId,
    },

    // ── Error ──
    ErrorExit {
        code: ErrorCode,
    },

    // ── SIMD ──
    SimdStringScan {
        pos: VReg,
        kind: VReg,
    },
    SimdWhitespaceSkip,

    // ── Function structure ──
    FuncStart {
        lambda_id: LambdaId,
        shape: &'static facet::Shape,
        data_args: Vec<VReg>,
        data_results: Vec<VReg>,
    },
    FuncEnd,
    CallLambda {
        target: LambdaId,
        args: Vec<VReg>,
        results: Vec<VReg>,
    },
}

// ─── LinearIr ────────────────────────────────────────────────────────────────

/// The linearized form of an RVSDG function.
pub struct LinearIr {
    /// The flat instruction sequence.
    pub ops: Vec<LinearOp>,
    /// Total number of labels allocated.
    pub label_count: u32,
    /// Total number of virtual registers.
    pub vreg_count: u32,
    /// Total number of stack slots.
    pub slot_count: u32,
}

// ─── Linearizer state ────────────────────────────────────────────────────────

struct Linearizer<'a> {
    func: &'a IrFunc,
    ops: Vec<LinearOp>,
    label_count: u32,
}

impl<'a> Linearizer<'a> {
    fn new(func: &'a IrFunc) -> Self {
        Self {
            func,
            ops: Vec::new(),
            label_count: 0,
        }
    }

    fn fresh_label(&mut self) -> LabelId {
        let id = LabelId::new(self.label_count);
        self.label_count += 1;
        id
    }

    fn emit(&mut self, op: LinearOp) {
        self.ops.push(op);
    }

    /// Resolve a PortSource to the VReg it produces.
    fn resolve_vreg(&self, source: PortSource) -> VReg {
        match source {
            PortSource::Node(output_ref) => {
                let node = &self.func.nodes[output_ref.node];
                node.outputs[output_ref.index as usize]
                    .vreg
                    .expect("data port must have vreg assigned")
            }
            PortSource::RegionArg(arg_ref) => {
                let region = &self.func.regions[arg_ref.region];
                region.args[arg_ref.index as usize]
                    .vreg
                    .expect("data region arg must have vreg assigned")
            }
        }
    }

    // ─── Topological sort ────────────────────────────────────────────

    fn collect_subregion_parent_deps(
        &self,
        region_id: RegionId,
        node_pos: &HashMap<NodeId, usize>,
        deps: &mut HashSet<usize>,
    ) {
        let region = &self.func.regions[region_id];
        for &nid in &region.nodes {
            let node = &self.func.nodes[nid];
            for input in &node.inputs {
                if let PortSource::Node(output_ref) = input.source
                    && let Some(&dep_pos) = node_pos.get(&output_ref.node)
                {
                    deps.insert(dep_pos);
                }
            }
            match &node.kind {
                NodeKind::Gamma { regions } => {
                    for &sub in regions {
                        self.collect_subregion_parent_deps(sub, node_pos, deps);
                    }
                }
                NodeKind::Theta { body } => {
                    self.collect_subregion_parent_deps(*body, node_pos, deps);
                }
                _ => {}
            }
        }
        for result in &region.results {
            if let PortSource::Node(output_ref) = result.source
                && let Some(&dep_pos) = node_pos.get(&output_ref.node)
            {
                deps.insert(dep_pos);
            }
        }
    }

    // r[impl ir.linearize.schedule]
    /// Topologically sort a region's nodes respecting data + state edges.
    fn topo_sort(&self, region_id: RegionId) -> Vec<NodeId> {
        let region = &self.func.regions[region_id];
        if region.nodes.is_empty() {
            return Vec::new();
        }

        // Map NodeId -> position in region.nodes for O(1) lookup.
        let mut node_pos: std::collections::HashMap<NodeId, usize> =
            std::collections::HashMap::new();
        for (i, &nid) in region.nodes.iter().enumerate() {
            node_pos.insert(nid, i);
        }

        let n = region.nodes.len();
        let mut in_degree = vec![0u32; n];
        // adjacency: for each node position, list of node positions that depend on it
        let mut dependents: Vec<Vec<usize>> = vec![Vec::new(); n];

        for (i, &nid) in region.nodes.iter().enumerate() {
            let node = &self.func.nodes[nid];
            let mut deps_for_node = HashSet::new();
            for input in &node.inputs {
                if let PortSource::Node(output_ref) = input.source
                    && let Some(&dep_pos) = node_pos.get(&output_ref.node)
                {
                    deps_for_node.insert(dep_pos);
                }
            }
            match &node.kind {
                NodeKind::Gamma { regions } => {
                    for &sub in regions {
                        self.collect_subregion_parent_deps(sub, &node_pos, &mut deps_for_node);
                    }
                }
                NodeKind::Theta { body } => {
                    self.collect_subregion_parent_deps(*body, &node_pos, &mut deps_for_node);
                }
                _ => {}
            }
            for dep_pos in deps_for_node {
                if dep_pos == i {
                    continue;
                }
                in_degree[i] += 1;
                dependents[dep_pos].push(i);
            }
        }

        // Kahn's algorithm with a queue (preserves insertion order for ties).
        let mut queue = VecDeque::new();
        for (i, &deg) in in_degree.iter().enumerate() {
            if deg == 0 {
                queue.push_back(i);
            }
        }

        let mut sorted = Vec::with_capacity(n);
        while let Some(pos) = queue.pop_front() {
            sorted.push(region.nodes[pos]);
            for &dep in &dependents[pos] {
                in_degree[dep] -= 1;
                if in_degree[dep] == 0 {
                    queue.push_back(dep);
                }
            }
        }

        assert_eq!(
            sorted.len(),
            n,
            "cycle detected in region's node dependencies"
        );
        sorted
    }

    // ─── Region linearization ────────────────────────────────────────

    fn linearize_region(&mut self, region_id: RegionId) {
        let sorted = self.topo_sort(region_id);
        for node_id in sorted {
            self.linearize_node(node_id);
        }
    }

    // ─── Node linearization ─────────────────────────────────────────

    fn linearize_node(&mut self, node_id: NodeId) {
        let kind = {
            let node = &self.func.nodes[node_id];
            clone_node_kind(&node.kind)
        };

        match kind {
            NodeKindRef::Simple(op) => self.linearize_simple(node_id, op),
            NodeKindRef::Gamma { regions } => self.linearize_gamma(node_id, &regions),
            NodeKindRef::Theta { body } => self.linearize_theta(node_id, body),
            NodeKindRef::Lambda {
                body,
                shape,
                lambda_id,
            } => {
                self.linearize_lambda(body, shape, lambda_id);
            }
            NodeKindRef::Apply { target } => self.linearize_apply(node_id, target),
        }
    }

    fn linearize_simple(&mut self, node_id: NodeId, op: &IrOp) {
        let node = &self.func.nodes[node_id];

        // Helper: get the VReg of data output at index.
        let data_dst =
            |idx: usize| -> VReg { node.outputs[idx].vreg.expect("data output must have vreg") };

        // Helper: resolve data input at index.
        let data_in = |idx: usize| -> VReg {
            let input = &node.inputs[idx];
            assert_eq!(input.kind, PortKind::Data);
            self.resolve_vreg(input.source)
        };

        match op {
            // ── Constants ──
            IrOp::Const { value } => {
                self.emit(LinearOp::Const {
                    dst: data_dst(0),
                    value: *value,
                });
            }

            // ── Binary arithmetic ──
            IrOp::Add => self.emit_binop(BinOpKind::Add, node),
            IrOp::Sub => self.emit_binop(BinOpKind::Sub, node),
            IrOp::And => self.emit_binop(BinOpKind::And, node),
            IrOp::Or => self.emit_binop(BinOpKind::Or, node),
            IrOp::Shr => self.emit_binop(BinOpKind::Shr, node),
            IrOp::Shl => self.emit_binop(BinOpKind::Shl, node),
            IrOp::Xor => self.emit_binop(BinOpKind::Xor, node),
            IrOp::CmpNe => self.emit_binop(BinOpKind::CmpNe, node),

            // ── Unary ──
            IrOp::ZigzagDecode { wide } => {
                self.emit(LinearOp::UnaryOp {
                    op: UnaryOpKind::ZigzagDecode { wide: *wide },
                    dst: data_dst(0),
                    src: data_in(0),
                });
            }
            IrOp::SignExtend { from_width } => {
                self.emit(LinearOp::UnaryOp {
                    op: UnaryOpKind::SignExtend {
                        from_width: *from_width,
                    },
                    dst: data_dst(0),
                    src: data_in(0),
                });
            }

            // ── Cursor ops ──
            IrOp::BoundsCheck { count } => {
                self.emit(LinearOp::BoundsCheck { count: *count });
            }
            IrOp::ReadBytes { count } => {
                self.emit(LinearOp::ReadBytes {
                    dst: data_dst(0),
                    count: *count,
                });
            }
            IrOp::PeekByte => {
                self.emit(LinearOp::PeekByte { dst: data_dst(0) });
            }
            IrOp::AdvanceCursor { count } => {
                self.emit(LinearOp::AdvanceCursor { count: *count });
            }
            IrOp::AdvanceCursorBy => {
                self.emit(LinearOp::AdvanceCursorBy { src: data_in(0) });
            }
            IrOp::SaveCursor => {
                self.emit(LinearOp::SaveCursor { dst: data_dst(0) });
            }
            IrOp::RestoreCursor => {
                self.emit(LinearOp::RestoreCursor { src: data_in(0) });
            }

            // ── Output ops ──
            IrOp::WriteToField { offset, width } => {
                self.emit(LinearOp::WriteToField {
                    src: data_in(0),
                    offset: *offset,
                    width: *width,
                });
            }
            IrOp::ReadFromField { offset, width } => {
                self.emit(LinearOp::ReadFromField {
                    dst: data_dst(0),
                    offset: *offset,
                    width: *width,
                });
            }
            IrOp::SaveOutPtr => {
                self.emit(LinearOp::SaveOutPtr { dst: data_dst(0) });
            }
            IrOp::SetOutPtr => {
                self.emit(LinearOp::SetOutPtr { src: data_in(0) });
            }

            // ── Stack ops ──
            IrOp::SlotAddr { slot } => {
                self.emit(LinearOp::SlotAddr {
                    dst: data_dst(0),
                    slot: *slot,
                });
            }
            IrOp::WriteToSlot { slot } => {
                self.emit(LinearOp::WriteToSlot {
                    slot: *slot,
                    src: data_in(0),
                });
            }
            IrOp::ReadFromSlot { slot } => {
                self.emit(LinearOp::ReadFromSlot {
                    dst: data_dst(0),
                    slot: *slot,
                });
            }

            // ── Call ops ──
            IrOp::CallIntrinsic {
                func,
                arg_count,
                has_result,
                field_offset,
            } => {
                let args: Vec<VReg> = (0..*arg_count as usize).map(&data_in).collect();
                let dst = if *has_result { Some(data_dst(0)) } else { None };
                self.emit(LinearOp::CallIntrinsic {
                    func: *func,
                    args,
                    dst,
                    field_offset: *field_offset,
                });
            }
            IrOp::CallPure { func, arg_count } => {
                let args: Vec<VReg> = (0..*arg_count as usize).map(&data_in).collect();
                self.emit(LinearOp::CallPure {
                    func: *func,
                    args,
                    dst: data_dst(0),
                });
            }

            // ── Error ──
            IrOp::ErrorExit { code } => {
                self.emit(LinearOp::ErrorExit { code: *code });
            }

            // ── SIMD ──
            IrOp::SimdStringScan => {
                self.emit(LinearOp::SimdStringScan {
                    pos: data_dst(0),
                    kind: data_dst(1),
                });
            }
            IrOp::SimdWhitespaceSkip => {
                self.emit(LinearOp::SimdWhitespaceSkip);
            }
        }
    }

    fn emit_binop(&mut self, op: BinOpKind, node: &Node) {
        let dst = node.outputs[0].vreg.expect("binop must have vreg");
        let lhs = self.resolve_vreg(node.inputs[0].source);
        let rhs = self.resolve_vreg(node.inputs[1].source);
        self.emit(LinearOp::BinOp { op, dst, lhs, rhs });
    }

    // ─── Gamma (conditional) ─────────────────────────────────────────

    fn linearize_gamma(&mut self, node_id: NodeId, regions: &[RegionId]) {
        let node = &self.func.nodes[node_id];
        let branch_count = regions.len();

        // The predicate is the first data input.
        let predicate = self.resolve_vreg(node.inputs[0].source);

        // Allocate labels: one per branch + merge label.
        let branch_labels: Vec<LabelId> = (0..branch_count).map(|_| self.fresh_label()).collect();
        let merge_label = self.fresh_label();

        // Emit JumpTable if > 2 branches, or BranchIfZero for 2-branch case.
        if branch_count == 2 {
            // predicate==0 → branch 0, predicate!=0 → branch 1
            self.emit(LinearOp::BranchIfZero {
                cond: predicate,
                target: branch_labels[0],
            });
            self.emit(LinearOp::Branch(branch_labels[1]));
        } else {
            // General case: jump table
            self.emit(LinearOp::JumpTable {
                predicate,
                labels: branch_labels.clone(),
                default: branch_labels[branch_count - 1],
            });
        }

        // Determine the data output count from the gamma node.
        let data_output_count = node
            .outputs
            .iter()
            .filter(|o| o.kind == PortKind::Data)
            .count();

        // Emit each branch.
        for (branch_idx, &region_id) in regions.iter().enumerate() {
            self.emit(LinearOp::Label(branch_labels[branch_idx]));

            // Emit copies for passthrough inputs → region args.
            self.emit_gamma_entry_copies(node, region_id);

            // Linearize the branch body.
            self.linearize_region(region_id);

            // Emit copies for region results → gamma output vregs.
            self.emit_gamma_exit_copies(node_id, region_id, data_output_count);

            // Branch to merge (skip for last branch — it falls through).
            if branch_idx < branch_count - 1 {
                self.emit(LinearOp::Branch(merge_label));
            }
        }

        self.emit(LinearOp::Label(merge_label));
    }

    /// Emit Copy ops for passthrough data inputs entering a gamma branch region.
    fn emit_gamma_entry_copies(&mut self, node: &Node, region_id: RegionId) {
        let region = &self.func.regions[region_id];
        // Inputs layout: [predicate, passthrough..., cursor_state, output_state]
        // Region args layout: [passthrough..., cursor_state, output_state]
        // Skip predicate (input 0), skip state inputs at the end.
        let passthrough_count = node.inputs.len() - 3; // minus predicate, cursor, output

        for i in 0..passthrough_count {
            let src_input = &node.inputs[i + 1]; // +1 to skip predicate
            if src_input.kind == PortKind::Data {
                let src_vreg = self.resolve_vreg(src_input.source);
                if let Some(dst_vreg) = region.args[i].vreg
                    && src_vreg != dst_vreg
                {
                    self.emit(LinearOp::Copy {
                        dst: dst_vreg,
                        src: src_vreg,
                    });
                }
            }
        }
    }

    /// Emit Copy ops for gamma branch results → gamma node output vregs.
    fn emit_gamma_exit_copies(
        &mut self,
        node_id: NodeId,
        region_id: RegionId,
        data_output_count: usize,
    ) {
        let region = &self.func.regions[region_id];
        let node = &self.func.nodes[node_id];
        // Region results: [data..., cursor_state, output_state]
        // Gamma outputs: [data..., cursor_state, output_state]
        for i in 0..data_output_count {
            let result = &region.results[i];
            if result.kind == PortKind::Data {
                let src_vreg = self.resolve_vreg(result.source);
                let dst_vreg = node.outputs[i]
                    .vreg
                    .expect("gamma data output must have vreg");
                if src_vreg != dst_vreg {
                    self.emit(LinearOp::Copy {
                        dst: dst_vreg,
                        src: src_vreg,
                    });
                }
            }
        }
    }

    // ─── Theta (loop) ────────────────────────────────────────────────

    fn linearize_theta(&mut self, node_id: NodeId, body: RegionId) {
        let node = &self.func.nodes[node_id];
        let body_region = &self.func.regions[body];

        // Theta inputs: [loop_vars..., cursor_state, output_state]
        // Body args: [loop_vars..., cursor_state, output_state]
        // Body results: [predicate, loop_vars..., cursor_state, output_state]
        // Theta outputs: [loop_vars..., cursor_state, output_state]

        let total_inputs = node.inputs.len();
        let loop_var_count = total_inputs - 2; // minus cursor_state, output_state

        // Emit copies for initial loop var values → body region args.
        for i in 0..loop_var_count {
            let input = &node.inputs[i];
            if input.kind == PortKind::Data {
                let src_vreg = self.resolve_vreg(input.source);
                if let Some(dst_vreg) = body_region.args[i].vreg
                    && src_vreg != dst_vreg
                {
                    self.emit(LinearOp::Copy {
                        dst: dst_vreg,
                        src: src_vreg,
                    });
                }
            }
        }

        // Loop top label.
        let loop_top = self.fresh_label();
        let loop_exit = self.fresh_label();
        self.emit(LinearOp::Label(loop_top));

        // Linearize the body.
        self.linearize_region(body);

        // Body results: [predicate, loop_vars..., cursor_state, output_state]
        // predicate: 0 = exit, nonzero = continue
        let predicate_source = body_region.results[0].source;
        let predicate_vreg = self.resolve_vreg(predicate_source);

        // Emit copies for body results → body region args (feedback).
        // Results[1..1+loop_var_count] → args[0..loop_var_count]
        for i in 0..loop_var_count {
            let result = &body_region.results[i + 1]; // +1 to skip predicate
            if result.kind == PortKind::Data {
                let src_vreg = self.resolve_vreg(result.source);
                if let Some(dst_vreg) = body_region.args[i].vreg
                    && src_vreg != dst_vreg
                {
                    self.emit(LinearOp::Copy {
                        dst: dst_vreg,
                        src: src_vreg,
                    });
                }
            }
        }

        // Branch back to loop top if predicate is nonzero.
        self.emit(LinearOp::BranchIf {
            cond: predicate_vreg,
            target: loop_top,
        });

        self.emit(LinearOp::Label(loop_exit));

        // Emit copies for final loop var values → theta output vregs.
        // After the loop exits, the body's region args hold the final values
        // (or the body results, depending on convention). We use body args
        // since the feedback copies already wrote there.
        for i in 0..loop_var_count {
            if let Some(src_vreg) = body_region.args[i].vreg
                && let Some(dst_vreg) = node.outputs[i].vreg
                && src_vreg != dst_vreg
            {
                self.emit(LinearOp::Copy {
                    dst: dst_vreg,
                    src: src_vreg,
                });
            }
        }
    }

    // ─── Lambda ──────────────────────────────────────────────────────

    fn linearize_lambda(
        &mut self,
        body: RegionId,
        shape: &'static facet::Shape,
        lambda_id: LambdaId,
    ) {
        let region = &self.func.regions[body];
        let data_args: Vec<VReg> = region
            .args
            .iter()
            .filter(|a| a.kind == PortKind::Data)
            .map(|a| a.vreg.expect("lambda data arg must have vreg assigned"))
            .collect();
        let data_results: Vec<VReg> = region
            .results
            .iter()
            .filter(|r| r.kind == PortKind::Data)
            .map(|r| self.resolve_vreg(r.source))
            .collect();
        self.emit(LinearOp::FuncStart {
            lambda_id,
            shape,
            data_args,
            data_results,
        });
        self.linearize_region(body);
        self.emit(LinearOp::FuncEnd);
    }

    // ─── Apply ───────────────────────────────────────────────────────

    fn linearize_apply(&mut self, node_id: NodeId, target: LambdaId) {
        let node = &self.func.nodes[node_id];
        let args: Vec<VReg> = node
            .inputs
            .iter()
            .filter(|i| i.kind == PortKind::Data)
            .map(|i| self.resolve_vreg(i.source))
            .collect();
        let results: Vec<VReg> = node
            .outputs
            .iter()
            .filter(|o| o.kind == PortKind::Data)
            .filter_map(|o| o.vreg)
            .collect();
        self.emit(LinearOp::CallLambda {
            target,
            args,
            results,
        });
    }
}

#[derive(Clone, Debug)]
struct LinearBlock {
    start: usize,
    end: usize,
    succs: Vec<usize>,
}

fn is_block_terminator(op: &LinearOp) -> bool {
    matches!(
        op,
        LinearOp::Branch(_)
            | LinearOp::BranchIf { .. }
            | LinearOp::BranchIfZero { .. }
            | LinearOp::JumpTable { .. }
            | LinearOp::ErrorExit { .. }
            | LinearOp::FuncEnd
    )
}

fn op_uses(op: &LinearOp, func_end_uses: Option<&[VReg]>) -> Vec<VReg> {
    match op {
        LinearOp::BinOp { lhs, rhs, .. } => vec![*lhs, *rhs],
        LinearOp::UnaryOp { src, .. } => vec![*src],
        LinearOp::Copy { src, .. } => vec![*src],
        LinearOp::AdvanceCursorBy { src } => vec![*src],
        LinearOp::RestoreCursor { src } => vec![*src],
        LinearOp::WriteToField { src, .. } => vec![*src],
        LinearOp::SetOutPtr { src } => vec![*src],
        LinearOp::WriteToSlot { src, .. } => vec![*src],
        LinearOp::CallIntrinsic { args, .. } => args.clone(),
        LinearOp::CallPure { args, .. } => args.clone(),
        LinearOp::BranchIf { cond, .. } => vec![*cond],
        LinearOp::BranchIfZero { cond, .. } => vec![*cond],
        LinearOp::JumpTable { predicate, .. } => vec![*predicate],
        LinearOp::SimdStringScan { pos, kind } => vec![*pos, *kind],
        LinearOp::CallLambda { args, .. } => args.clone(),
        LinearOp::FuncEnd => func_end_uses.unwrap_or_default().to_vec(),
        LinearOp::Const { .. }
        | LinearOp::BoundsCheck { .. }
        | LinearOp::ReadBytes { .. }
        | LinearOp::PeekByte { .. }
        | LinearOp::AdvanceCursor { .. }
        | LinearOp::SaveCursor { .. }
        | LinearOp::ReadFromField { .. }
        | LinearOp::SaveOutPtr { .. }
        | LinearOp::SlotAddr { .. }
        | LinearOp::ReadFromSlot { .. }
        | LinearOp::Label(_)
        | LinearOp::Branch(_)
        | LinearOp::ErrorExit { .. }
        | LinearOp::SimdWhitespaceSkip
        | LinearOp::FuncStart { .. } => Vec::new(),
    }
}

fn op_defs(op: &LinearOp) -> Vec<VReg> {
    match op {
        LinearOp::Const { dst, .. } => vec![*dst],
        LinearOp::BinOp { dst, .. } => vec![*dst],
        LinearOp::UnaryOp { dst, .. } => vec![*dst],
        LinearOp::Copy { dst, .. } => vec![*dst],
        LinearOp::ReadBytes { dst, .. } => vec![*dst],
        LinearOp::PeekByte { dst } => vec![*dst],
        LinearOp::SaveCursor { dst } => vec![*dst],
        LinearOp::ReadFromField { dst, .. } => vec![*dst],
        LinearOp::SaveOutPtr { dst } => vec![*dst],
        LinearOp::SlotAddr { dst, .. } => vec![*dst],
        LinearOp::ReadFromSlot { dst, .. } => vec![*dst],
        LinearOp::CallIntrinsic { dst, .. } => dst.iter().copied().collect(),
        LinearOp::CallPure { dst, .. } => vec![*dst],
        LinearOp::SimdStringScan { pos, kind } => vec![*pos, *kind],
        LinearOp::FuncStart { data_args, .. } => data_args.clone(),
        LinearOp::CallLambda { results, .. } => results.clone(),
        LinearOp::BoundsCheck { .. }
        | LinearOp::AdvanceCursor { .. }
        | LinearOp::AdvanceCursorBy { .. }
        | LinearOp::RestoreCursor { .. }
        | LinearOp::WriteToField { .. }
        | LinearOp::SetOutPtr { .. }
        | LinearOp::WriteToSlot { .. }
        | LinearOp::Label(_)
        | LinearOp::Branch(_)
        | LinearOp::BranchIf { .. }
        | LinearOp::BranchIfZero { .. }
        | LinearOp::JumpTable { .. }
        | LinearOp::ErrorExit { .. }
        | LinearOp::SimdWhitespaceSkip
        | LinearOp::FuncEnd => Vec::new(),
    }
}

fn collect_func_end_uses(ops: &[LinearOp]) -> HashMap<usize, Vec<VReg>> {
    let mut out = HashMap::new();
    let mut current_results: Option<Vec<VReg>> = None;
    for (i, op) in ops.iter().enumerate() {
        match op {
            LinearOp::FuncStart { data_results, .. } => {
                current_results = Some(data_results.clone());
            }
            LinearOp::FuncEnd => {
                out.insert(i, current_results.clone().unwrap_or_default());
                current_results = None;
            }
            _ => {}
        }
    }
    out
}

fn rewrite_op_uses(op: &mut LinearOp, mut resolve: impl FnMut(VReg) -> VReg) {
    let rewrite = |v: &mut VReg, resolve: &mut dyn FnMut(VReg) -> VReg| {
        *v = resolve(*v);
    };
    match op {
        LinearOp::BinOp { lhs, rhs, .. } => {
            rewrite(lhs, &mut resolve);
            rewrite(rhs, &mut resolve);
        }
        LinearOp::UnaryOp { src, .. } => rewrite(src, &mut resolve),
        LinearOp::Copy { src, .. } => rewrite(src, &mut resolve),
        LinearOp::AdvanceCursorBy { src } => rewrite(src, &mut resolve),
        LinearOp::RestoreCursor { src } => rewrite(src, &mut resolve),
        LinearOp::WriteToField { src, .. } => rewrite(src, &mut resolve),
        LinearOp::SetOutPtr { src } => rewrite(src, &mut resolve),
        LinearOp::WriteToSlot { src, .. } => rewrite(src, &mut resolve),
        LinearOp::CallIntrinsic { args, .. }
        | LinearOp::CallPure { args, .. }
        | LinearOp::CallLambda { args, .. } => {
            for arg in args {
                rewrite(arg, &mut resolve);
            }
        }
        LinearOp::BranchIf { cond, .. } | LinearOp::BranchIfZero { cond, .. } => {
            rewrite(cond, &mut resolve);
        }
        LinearOp::JumpTable { predicate, .. } => rewrite(predicate, &mut resolve),
        LinearOp::SimdStringScan { pos, kind } => {
            rewrite(pos, &mut resolve);
            rewrite(kind, &mut resolve);
        }
        LinearOp::Const { .. }
        | LinearOp::BoundsCheck { .. }
        | LinearOp::ReadBytes { .. }
        | LinearOp::PeekByte { .. }
        | LinearOp::AdvanceCursor { .. }
        | LinearOp::SaveCursor { .. }
        | LinearOp::ReadFromField { .. }
        | LinearOp::SaveOutPtr { .. }
        | LinearOp::SlotAddr { .. }
        | LinearOp::ReadFromSlot { .. }
        | LinearOp::Label(_)
        | LinearOp::Branch(_)
        | LinearOp::ErrorExit { .. }
        | LinearOp::SimdWhitespaceSkip
        | LinearOp::FuncStart { .. }
        | LinearOp::FuncEnd => {}
    }
}

fn build_blocks(ops: &[LinearOp]) -> Vec<LinearBlock> {
    if ops.is_empty() {
        return Vec::new();
    }
    let mut starts = vec![0usize];
    for (i, op) in ops.iter().enumerate() {
        if matches!(op, LinearOp::Label(_)) {
            starts.push(i);
        }
        if is_block_terminator(op) && i + 1 < ops.len() {
            starts.push(i + 1);
        }
    }
    starts.sort_unstable();
    starts.dedup();

    let mut blocks = Vec::new();
    for idx in 0..starts.len() {
        let start = starts[idx];
        let end = starts.get(idx + 1).copied().unwrap_or(ops.len());
        if start < end {
            blocks.push(LinearBlock {
                start,
                end,
                succs: Vec::new(),
            });
        }
    }

    let mut label_to_block = HashMap::<LabelId, usize>::new();
    for (bi, block) in blocks.iter().enumerate() {
        if let LinearOp::Label(label) = ops[block.start] {
            label_to_block.insert(label, bi);
        }
    }

    for bi in 0..blocks.len() {
        let mut succs = Vec::new();
        let term = &ops[blocks[bi].end - 1];
        match term {
            LinearOp::Branch(label) => {
                succs.push(
                    *label_to_block
                        .get(label)
                        .expect("branch target label must be block entry"),
                );
            }
            LinearOp::BranchIf { target, .. } | LinearOp::BranchIfZero { target, .. } => {
                succs.push(
                    *label_to_block
                        .get(target)
                        .expect("branch target label must be block entry"),
                );
                if bi + 1 < blocks.len() {
                    succs.push(bi + 1);
                }
            }
            LinearOp::JumpTable {
                labels, default, ..
            } => {
                for label in labels {
                    succs.push(
                        *label_to_block
                            .get(label)
                            .expect("jumptable label must be block entry"),
                    );
                }
                succs.push(
                    *label_to_block
                        .get(default)
                        .expect("jumptable default label must be block entry"),
                );
            }
            LinearOp::ErrorExit { .. } | LinearOp::FuncEnd => {}
            _ => {
                if bi + 1 < blocks.len() {
                    succs.push(bi + 1);
                }
            }
        }
        succs.sort_unstable();
        succs.dedup();
        blocks[bi].succs = succs;
    }

    blocks
}

fn kill_alias(alias: &mut HashMap<VReg, VReg>, defined: VReg) {
    alias.remove(&defined);
    alias.retain(|_, src| *src != defined);
}

fn resolve_alias(alias: &HashMap<VReg, VReg>, mut v: VReg) -> VReg {
    let mut seen = HashSet::new();
    while seen.insert(v) {
        let Some(&next) = alias.get(&v) else { break };
        if next == v {
            break;
        }
        v = next;
    }
    v
}

fn optimize_linear_ops(ops: &mut Vec<LinearOp>) {
    let blocks = build_blocks(ops);
    if blocks.is_empty() {
        return;
    }

    let mut remove = vec![false; ops.len()];

    for block in &blocks {
        let mut alias = HashMap::<VReg, VReg>::new();
        for i in block.start..block.end {
            rewrite_op_uses(&mut ops[i], |v| resolve_alias(&alias, v));

            if let LinearOp::Copy { dst, src } = ops[i] {
                if dst == src {
                    remove[i] = true;
                    continue;
                }
                kill_alias(&mut alias, dst);
                alias.insert(dst, src);
                continue;
            }

            for d in op_defs(&ops[i]) {
                kill_alias(&mut alias, d);
            }
        }
    }

    let func_end_uses = collect_func_end_uses(ops);
    let mut block_uses = vec![HashSet::<VReg>::new(); blocks.len()];
    let mut block_defs = vec![HashSet::<VReg>::new(); blocks.len()];
    for (bi, block) in blocks.iter().enumerate() {
        let mut uses = HashSet::new();
        let mut defs = HashSet::new();
        #[allow(clippy::needless_range_loop)]
        for i in block.start..block.end {
            let op_uses = op_uses(&ops[i], func_end_uses.get(&i).map(Vec::as_slice));
            for u in op_uses {
                if !defs.contains(&u) {
                    uses.insert(u);
                }
            }
            for d in op_defs(&ops[i]) {
                defs.insert(d);
            }
        }
        block_uses[bi] = uses;
        block_defs[bi] = defs;
    }

    let mut live_in = vec![HashSet::<VReg>::new(); blocks.len()];
    let mut live_out = vec![HashSet::<VReg>::new(); blocks.len()];
    loop {
        let mut changed = false;
        for bi in (0..blocks.len()).rev() {
            let mut out = HashSet::new();
            for &succ in &blocks[bi].succs {
                out.extend(live_in[succ].iter().copied());
            }
            let mut in_set = block_uses[bi].clone();
            let mut out_minus_defs = out.clone();
            for d in &block_defs[bi] {
                out_minus_defs.remove(d);
            }
            in_set.extend(out_minus_defs);

            if out != live_out[bi] {
                live_out[bi] = out;
                changed = true;
            }
            if in_set != live_in[bi] {
                live_in[bi] = in_set;
                changed = true;
            }
        }
        if !changed {
            break;
        }
    }

    for (bi, block) in blocks.iter().enumerate() {
        let mut live = live_out[bi].clone();
        for i in (block.start..block.end).rev() {
            if remove[i] {
                continue;
            }
            if let LinearOp::Copy { dst, .. } = ops[i]
                && !live.contains(&dst)
            {
                remove[i] = true;
                continue;
            }
            let defs = op_defs(&ops[i]);
            let uses = op_uses(&ops[i], func_end_uses.get(&i).map(Vec::as_slice));
            for d in defs {
                live.remove(&d);
            }
            for u in uses {
                live.insert(u);
            }
        }
    }

    let old_ops = std::mem::take(ops);
    *ops = old_ops
        .into_iter()
        .enumerate()
        .filter_map(|(i, op)| (!remove[i]).then_some(op))
        .collect();
}

/// A lightweight enum mirroring NodeKind but owning the data needed
/// for linearization (avoids borrow issues with self.func).
enum NodeKindRef<'a> {
    Simple(&'a IrOp),
    Gamma {
        regions: Vec<RegionId>,
    },
    Theta {
        body: RegionId,
    },
    Lambda {
        body: RegionId,
        shape: &'static facet::Shape,
        lambda_id: LambdaId,
    },
    Apply {
        target: LambdaId,
    },
}

fn clone_node_kind(kind: &NodeKind) -> NodeKindRef<'_> {
    match kind {
        NodeKind::Simple(op) => NodeKindRef::Simple(op),
        NodeKind::Gamma { regions } => NodeKindRef::Gamma {
            regions: regions.clone(),
        },
        NodeKind::Theta { body } => NodeKindRef::Theta { body: *body },
        NodeKind::Lambda {
            body,
            shape,
            lambda_id,
        } => NodeKindRef::Lambda {
            body: *body,
            shape,
            lambda_id: *lambda_id,
        },
        NodeKind::Apply { target } => NodeKindRef::Apply { target: *target },
    }
}

// ─── VReg assignment pass ────────────────────────────────────────────────────

/// Assign VRegs to all data output ports and region args that don't have one.
fn assign_vregs(func: &mut IrFunc) {
    // Assign to all node output ports.
    let node_count = func.nodes.len();
    for i in 0..node_count {
        let node_id = NodeId::new(i as u32);
        let output_count = func.nodes[node_id].outputs.len();
        for j in 0..output_count {
            if func.nodes[node_id].outputs[j].kind == PortKind::Data
                && func.nodes[node_id].outputs[j].vreg.is_none()
            {
                let vreg = func.fresh_vreg();
                func.nodes[node_id].outputs[j].vreg = Some(vreg);
            }
        }
    }

    // Assign to all region args.
    let region_count = func.regions.len();
    for i in 0..region_count {
        let region_id = RegionId::new(i as u32);
        let arg_count = func.regions[region_id].args.len();
        for j in 0..arg_count {
            if func.regions[region_id].args[j].kind == PortKind::Data
                && func.regions[region_id].args[j].vreg.is_none()
            {
                let vreg = func.fresh_vreg();
                func.regions[region_id].args[j].vreg = Some(vreg);
            }
        }
    }
}

// ─── Public API ──────────────────────────────────────────────────────────────

/// Linearize an RVSDG function into a flat instruction sequence.
pub fn linearize(func: &mut IrFunc) -> LinearIr {
    // Pass 1: ensure all data ports have VRegs.
    assign_vregs(func);

    // Pass 2: walk the RVSDG and emit linear ops.
    let lambda_nodes = func.lambdas.clone();
    let mut lin = Linearizer::new(func);
    for (i, node) in lambda_nodes.iter().enumerate() {
        if i == 0 {
            lin.linearize_node(func.root);
        } else {
            lin.linearize_node(*node);
        }
    }

    let mut ops = lin.ops;
    optimize_linear_ops(&mut ops);

    LinearIr {
        ops,
        label_count: lin.label_count,
        vreg_count: func.vreg_count(),
        slot_count: func.slot_count(),
    }
}

// ─── Display ─────────────────────────────────────────────────────────────────

impl fmt::Display for LinearIr {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for op in &self.ops {
            // Labels get no indentation, everything else gets 2 spaces.
            match op {
                LinearOp::Label(label) => {
                    writeln!(f, "L{}:", label.index())?;
                }
                LinearOp::FuncStart {
                    lambda_id, shape, ..
                } => {
                    writeln!(
                        f,
                        "func λ{} ({}):",
                        lambda_id.index(),
                        shape.type_identifier
                    )?;
                }
                LinearOp::FuncEnd => {
                    writeln!(f, "end")?;
                }
                _ => {
                    write!(f, "  ")?;
                    fmt_op(f, op)?;
                    writeln!(f)?;
                }
            }
        }
        Ok(())
    }
}

fn fmt_vreg(f: &mut fmt::Formatter<'_>, v: VReg) -> fmt::Result {
    write!(f, "v{}", v.index())
}

fn fmt_op(f: &mut fmt::Formatter<'_>, op: &LinearOp) -> fmt::Result {
    match op {
        LinearOp::Const { dst, value } => {
            fmt_vreg(f, *dst)?;
            write!(f, " = const {value}")
        }
        LinearOp::BinOp { op, dst, lhs, rhs } => {
            fmt_vreg(f, *dst)?;
            write!(f, " = ")?;
            fmt_vreg(f, *lhs)?;
            write!(f, " {op:?} ")?;
            fmt_vreg(f, *rhs)
        }
        LinearOp::UnaryOp { op, dst, src } => {
            fmt_vreg(f, *dst)?;
            write!(f, " = {op:?} ")?;
            fmt_vreg(f, *src)
        }
        LinearOp::Copy { dst, src } => {
            fmt_vreg(f, *dst)?;
            write!(f, " = copy ")?;
            fmt_vreg(f, *src)
        }
        LinearOp::BoundsCheck { count } => write!(f, "bounds_check {count}"),
        LinearOp::ReadBytes { dst, count } => {
            fmt_vreg(f, *dst)?;
            write!(f, " = read_bytes {count}")
        }
        LinearOp::PeekByte { dst } => {
            fmt_vreg(f, *dst)?;
            write!(f, " = peek_byte")
        }
        LinearOp::AdvanceCursor { count } => write!(f, "advance {count}"),
        LinearOp::AdvanceCursorBy { src } => {
            write!(f, "advance_by ")?;
            fmt_vreg(f, *src)
        }
        LinearOp::SaveCursor { dst } => {
            fmt_vreg(f, *dst)?;
            write!(f, " = save_cursor")
        }
        LinearOp::RestoreCursor { src } => {
            write!(f, "restore_cursor ")?;
            fmt_vreg(f, *src)
        }
        LinearOp::WriteToField { src, offset, width } => {
            write!(f, "store [{offset}:{width}] ")?;
            fmt_vreg(f, *src)
        }
        LinearOp::ReadFromField { dst, offset, width } => {
            fmt_vreg(f, *dst)?;
            write!(f, " = load [{offset}:{width}]")
        }
        LinearOp::SaveOutPtr { dst } => {
            fmt_vreg(f, *dst)?;
            write!(f, " = save_out_ptr")
        }
        LinearOp::SetOutPtr { src } => {
            write!(f, "set_out_ptr ")?;
            fmt_vreg(f, *src)
        }
        LinearOp::SlotAddr { dst, slot } => {
            fmt_vreg(f, *dst)?;
            write!(f, " = slot_addr {}", slot.index())
        }
        LinearOp::WriteToSlot { slot, src } => {
            write!(f, "slot[{}] = ", slot.index())?;
            fmt_vreg(f, *src)
        }
        LinearOp::ReadFromSlot { dst, slot } => {
            fmt_vreg(f, *dst)?;
            write!(f, " = slot[{}]", slot.index())
        }
        LinearOp::CallIntrinsic {
            func,
            args,
            dst,
            field_offset,
        } => {
            if let Some(d) = dst {
                fmt_vreg(f, *d)?;
                write!(f, " = ")?;
            }
            write!(f, "call_intrinsic {func}(")?;
            for (i, a) in args.iter().enumerate() {
                if i > 0 {
                    write!(f, ", ")?;
                }
                fmt_vreg(f, *a)?;
            }
            write!(f, ") @{field_offset}")
        }
        LinearOp::CallPure { func, args, dst } => {
            fmt_vreg(f, *dst)?;
            write!(f, " = call_pure {func}(")?;
            for (i, a) in args.iter().enumerate() {
                if i > 0 {
                    write!(f, ", ")?;
                }
                fmt_vreg(f, *a)?;
            }
            write!(f, ")")
        }
        LinearOp::Branch(target) => write!(f, "br L{}", target.index()),
        LinearOp::BranchIf { cond, target } => {
            write!(f, "br_if ")?;
            fmt_vreg(f, *cond)?;
            write!(f, " L{}", target.index())
        }
        LinearOp::BranchIfZero { cond, target } => {
            write!(f, "br_zero ")?;
            fmt_vreg(f, *cond)?;
            write!(f, " L{}", target.index())
        }
        LinearOp::JumpTable {
            predicate,
            labels,
            default,
        } => {
            write!(f, "jump_table ")?;
            fmt_vreg(f, *predicate)?;
            write!(f, " [")?;
            for (i, l) in labels.iter().enumerate() {
                if i > 0 {
                    write!(f, ", ")?;
                }
                write!(f, "L{}", l.index())?;
            }
            write!(f, "] default L{}", default.index())
        }
        LinearOp::ErrorExit { code } => write!(f, "error_exit {code:?}"),
        LinearOp::SimdStringScan { pos, kind } => {
            fmt_vreg(f, *pos)?;
            write!(f, ", ")?;
            fmt_vreg(f, *kind)?;
            write!(f, " = simd_string_scan")
        }
        LinearOp::SimdWhitespaceSkip => write!(f, "simd_whitespace_skip"),
        LinearOp::CallLambda {
            target,
            args,
            results,
        } => {
            if !results.is_empty() {
                for (i, r) in results.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    fmt_vreg(f, *r)?;
                }
                write!(f, " = ")?;
            }
            write!(f, "call λ{}(", target.index())?;
            for (i, a) in args.iter().enumerate() {
                if i > 0 {
                    write!(f, ", ")?;
                }
                fmt_vreg(f, *a)?;
            }
            write!(f, ")")
        }
        // FuncStart/FuncEnd/Label handled in Display for LinearIr
        LinearOp::Label(_) | LinearOp::FuncStart { .. } | LinearOp::FuncEnd => {
            unreachable!("handled in Display for LinearIr")
        }
    }
}

impl fmt::Debug for LinearIr {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "LinearIr {{")?;
        writeln!(
            f,
            "  labels: {}, vregs: {}, slots: {}",
            self.label_count, self.vreg_count, self.slot_count
        )?;
        for op in &self.ops {
            writeln!(f, "  {op:?}")?;
        }
        writeln!(f, "}}")
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use kajit_ir::{IrBuilder, IrOp, LambdaId, VReg, Width};

    #[test]
    fn linearize_simple_chain() {
        // BoundsCheck(4) → ReadBytes(4) → WriteToField(offset=0, W4)
        let mut builder = IrBuilder::new(<u32 as facet::Facet>::SHAPE);
        {
            let mut rb = builder.root_region();
            rb.bounds_check(4);
            let data = rb.read_bytes(4);
            rb.write_to_field(data, 0, Width::W4);
            rb.set_results(&[]);
        }
        let mut func = builder.finish();
        let ir = linearize(&mut func);

        // Expected: FuncStart, BoundsCheck(4), ReadBytes(4), WriteToField, FuncEnd
        assert!(matches!(ir.ops[0], LinearOp::FuncStart { .. }));
        assert!(matches!(ir.ops[1], LinearOp::BoundsCheck { count: 4 }));
        assert!(matches!(ir.ops[2], LinearOp::ReadBytes { count: 4, .. }));
        assert!(matches!(
            ir.ops[3],
            LinearOp::WriteToField {
                offset: 0,
                width: Width::W4,
                ..
            }
        ));
        assert!(matches!(ir.ops[4], LinearOp::FuncEnd));
        assert_eq!(ir.ops.len(), 5);
    }

    #[test]
    fn linearize_gamma_two_branches() {
        // Gamma with predicate, 2 branches:
        //   branch 0: const 42 → result
        //   branch 1: const 99 → result
        let mut builder = IrBuilder::new(<u32 as facet::Facet>::SHAPE);
        {
            let mut rb = builder.root_region();
            let pred = rb.const_val(0);
            let results = rb.gamma(pred, &[], 2, |branch_idx, bb| {
                let val = if branch_idx == 0 {
                    bb.const_val(42)
                } else {
                    bb.const_val(99)
                };
                bb.set_results(&[val]);
            });
            assert_eq!(results.len(), 1);
            rb.write_to_field(results[0], 0, Width::W4);
            rb.set_results(&[]);
        }
        let mut func = builder.finish();
        let ir = linearize(&mut func);

        // Verify structure: FuncStart, Const(pred), BranchIfZero, Branch,
        //   Label(0), Const(42), Copy, Branch(merge), Label(1), Const(99), Copy, Label(merge), ...
        let display = format!("{ir}");
        assert!(
            display.contains("br_zero"),
            "should have BranchIfZero for 2-branch gamma:\n{display}"
        );
        assert!(
            display.contains("const 42"),
            "branch 0 should produce 42:\n{display}"
        );
        assert!(
            display.contains("const 99"),
            "branch 1 should produce 99:\n{display}"
        );
    }

    #[test]
    fn linearize_theta_loop() {
        // Theta: count down from 5 to 0.
        // loop_var = counter
        // body: counter - 1, predicate = counter > 0
        let mut builder = IrBuilder::new(<u32 as facet::Facet>::SHAPE);
        {
            let mut rb = builder.root_region();
            let init_count = rb.const_val(5);
            let one = rb.const_val(1);
            let _results = rb.theta(&[init_count, one], |bb| {
                let args = bb.region_args(2);
                let counter = args[0];
                let one = args[1];
                let new_counter = bb.binop(IrOp::Sub, counter, one);
                // predicate = new_counter (0=exit)
                bb.set_results(&[new_counter, new_counter, one]);
            });
            rb.set_results(&[]);
        }
        let mut func = builder.finish();
        let ir = linearize(&mut func);

        let display = format!("{ir}");
        assert!(
            display.contains("br_if"),
            "should have BranchIf back-edge:\n{display}"
        );
        assert!(
            display.contains("Sub"),
            "should have subtraction:\n{display}"
        );
    }

    #[test]
    fn linearize_call_intrinsic() {
        use crate::intrinsics;
        use kajit_ir::IntrinsicFn;

        let mut builder = IrBuilder::new(<bool as facet::Facet>::SHAPE);
        {
            let mut rb = builder.root_region();
            rb.bounds_check(1);
            rb.call_intrinsic(
                IntrinsicFn(intrinsics::kajit_read_bool as *const () as usize),
                &[],
                0,
                false,
            );
            rb.set_results(&[]);
        }
        let mut func = builder.finish();
        let ir = linearize(&mut func);

        let has_call = ir
            .ops
            .iter()
            .any(|op| matches!(op, LinearOp::CallIntrinsic { .. }));
        assert!(has_call, "should contain CallIntrinsic");
    }

    #[test]
    fn linearize_display() {
        let mut builder = IrBuilder::new(<u32 as facet::Facet>::SHAPE);
        {
            let mut rb = builder.root_region();
            rb.bounds_check(4);
            let data = rb.read_bytes(4);
            rb.write_to_field(data, 0, Width::W4);
            rb.set_results(&[]);
        }
        let mut func = builder.finish();
        let ir = linearize(&mut func);

        let display = format!("{ir}");
        assert!(
            display.contains("func"),
            "display should start with func:\n{display}"
        );
        assert!(
            display.contains("bounds_check 4"),
            "display should contain bounds_check:\n{display}"
        );
        assert!(
            display.contains("read_bytes 4"),
            "display should contain read_bytes:\n{display}"
        );
        assert!(
            display.contains("store [0:W4]"),
            "display should contain store:\n{display}"
        );
        assert!(
            display.contains("end"),
            "display should end with end:\n{display}"
        );
    }

    #[test]
    fn optimize_linear_ops_elides_dead_copy_chain() {
        let v0 = VReg::new(0);
        let v1 = VReg::new(1);
        let v2 = VReg::new(2);
        let mut ops = vec![
            LinearOp::FuncStart {
                lambda_id: LambdaId::new(0),
                shape: <u32 as facet::Facet>::SHAPE,
                data_args: vec![],
                data_results: vec![],
            },
            LinearOp::Const { dst: v0, value: 7 },
            LinearOp::Copy { dst: v1, src: v0 },
            LinearOp::Copy { dst: v2, src: v1 },
            LinearOp::WriteToField {
                src: v2,
                offset: 0,
                width: Width::W4,
            },
            LinearOp::FuncEnd,
        ];

        optimize_linear_ops(&mut ops);

        let copy_count = ops
            .iter()
            .filter(|op| matches!(op, LinearOp::Copy { .. }))
            .count();
        assert_eq!(copy_count, 0, "dead copy chain should be eliminated");
        let write_src = ops.iter().find_map(|op| match op {
            LinearOp::WriteToField { src, .. } => Some(*src),
            _ => None,
        });
        assert_eq!(write_src, Some(v0), "store should use propagated source");
    }

    #[test]
    fn optimize_linear_ops_keeps_copy_feeding_func_end_result() {
        let v0 = VReg::new(0);
        let v1 = VReg::new(1);
        let mut ops = vec![
            LinearOp::FuncStart {
                lambda_id: LambdaId::new(0),
                shape: <u32 as facet::Facet>::SHAPE,
                data_args: vec![],
                data_results: vec![v1],
            },
            LinearOp::Const { dst: v0, value: 9 },
            LinearOp::Copy { dst: v1, src: v0 },
            LinearOp::FuncEnd,
        ];

        optimize_linear_ops(&mut ops);

        assert!(
            ops.iter()
                .any(|op| matches!(op, LinearOp::Copy { dst, src } if *dst == v1 && *src == v0)),
            "copy into function result vreg must be preserved"
        );
    }
}
