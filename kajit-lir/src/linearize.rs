//! Linearizer: converts the RVSDG into a flat instruction sequence.
//!
//! The RVSDG is a tree of regions and nodes. The linearizer walks this tree,
//! topologically sorts each region's nodes, and emits a flat `Vec<LinearOp>`
//! with explicit labels and branches for control flow (gamma/theta).

use std::collections::{HashMap, HashSet, VecDeque};
use std::fmt;

use kajit_ir::ErrorCode;
use kajit_ir::{
    Arena, DebugScope, DebugScopeId, DebugValue, DebugValueId, Id, IntrinsicRegistry, IrFunc, IrOp,
    LambdaId, Node, NodeId, NodeKind, PortKind, PortSource, RegionId, SlotId, VReg, Width,
};

// ─── Label ID ────────────────────────────────────────────────────────────────

/// Marker type for label IDs.
pub struct LabelMarker;
/// A label in the linear instruction sequence.
pub type LabelId = Id<LabelMarker>;

// ─── BinOpKind / UnaryOpKind ─────────────────────────────────────────────────

/// Binary operation kind for linear IR.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BinOpKind {
    Add,
    Sub,
    Mul,
    And,
    Or,
    Shr,
    Shl,
    Xor,
    CmpEq,
    CmpNe,
    CmpLt,
    CmpLe,
    CmpGt,
    CmpGe,
}

/// Unary operation kind for linear IR.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
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
    SaveInputEnd {
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
    StoreToAddr {
        addr: VReg,
        src: VReg,
        width: Width,
    },
    LoadFromAddr {
        dst: VReg,
        addr: VReg,
        width: Width,
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
    /// Unconditional branch. `phi_args` carry explicit (source → target_param)
    /// mappings for phi data flow; vregs not listed flow through unchanged.
    Branch {
        target: LabelId,
        phi_args: Vec<(VReg, VReg)>,
    },
    /// Branch if condition is nonzero. `phi_args` / `fallthrough_phi_args`
    /// carry (source, target_param) for the taken / fallthrough edges.
    BranchIf {
        cond: VReg,
        target: LabelId,
        phi_args: Vec<(VReg, VReg)>,
        fallthrough_phi_args: Vec<(VReg, VReg)>,
    },
    /// Branch if condition is zero. Same semantics as BranchIf.
    BranchIfZero {
        cond: VReg,
        target: LabelId,
        phi_args: Vec<(VReg, VReg)>,
        fallthrough_phi_args: Vec<(VReg, VReg)>,
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
        label: String,
        /// Minimum output buffer size in bytes. Used by the interpreter/simulator
        /// to allocate the output buffer when static inference from WriteToField is insufficient.
        output_size: usize,
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
    /// Preserved debug scope provenance copied from RVSDG.
    pub debug: LinearDebugProvenance,
}

#[derive(Clone, Default)]
pub struct LinearDebugProvenance {
    pub scopes: Arena<DebugScope>,
    pub values: Arena<DebugValue>,
    pub root_scope: Option<DebugScopeId>,
    pub op_scopes: Vec<Option<DebugScopeId>>,
    pub op_values: Vec<Option<DebugValueId>>,
    pub vreg_scopes: Vec<Option<DebugScopeId>>,
    pub vreg_values: Vec<Option<DebugValueId>>,
}

// ─── Linearizer state ────────────────────────────────────────────────────────

struct Linearizer<'a> {
    func: &'a IrFunc,
    ops: Vec<LinearOp>,
    label_count: u32,
    op_scopes: Vec<Option<DebugScopeId>>,
    op_values: Vec<Option<DebugValueId>>,
    vreg_scopes: Vec<Option<DebugScopeId>>,
    vreg_values: Vec<Option<DebugValueId>>,
}

impl<'a> Linearizer<'a> {
    fn new(func: &'a IrFunc) -> Self {
        Self {
            func,
            ops: Vec::new(),
            label_count: 0,
            op_scopes: Vec::new(),
            op_values: Vec::new(),
            vreg_scopes: vec![None; func.vreg_count() as usize],
            vreg_values: vec![None; func.vreg_count() as usize],
        }
    }

    fn fresh_label(&mut self) -> LabelId {
        let id = LabelId::new(self.label_count);
        self.label_count += 1;
        id
    }

    fn emit_with_value(
        &mut self,
        scope: Option<DebugScopeId>,
        debug_value: Option<DebugValueId>,
        op: LinearOp,
    ) {
        self.ops.push(op);
        self.op_scopes.push(scope);
        self.op_values.push(debug_value);
    }

    fn emit(&mut self, scope: Option<DebugScopeId>, op: LinearOp) {
        self.emit_with_value(scope, None, op);
    }

    fn emit_node(&mut self, node: &Node, op: LinearOp) {
        self.emit_with_value(Some(node.debug_scope), node.debug_value, op);
    }

    fn record_vreg_scope(&mut self, vreg: VReg, scope: DebugScopeId) {
        let slot = self
            .vreg_scopes
            .get_mut(vreg.index())
            .expect("vreg scope index must fit");
        *slot = Some(scope);
    }

    fn record_vreg_value(&mut self, vreg: VReg, debug_value: DebugValueId) {
        let slot = self
            .vreg_values
            .get_mut(vreg.index())
            .expect("vreg value index must fit");
        *slot = Some(debug_value);
    }

    fn record_output_scopes(&mut self, node: &Node) {
        for output in &node.outputs {
            if output.kind == PortKind::Data
                && let Some(vreg) = output.vreg
            {
                self.record_vreg_scope(vreg, output.debug_scope);
                if let Some(debug_value) = node.debug_value {
                    self.record_vreg_value(vreg, debug_value);
                }
            }
        }
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
            PortSource::RegionArg(arg_ref) => self.func.region_args[arg_ref.arg]
                .vreg
                .expect("data region arg must have vreg assigned"),
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
        for &result_id in &region.results {
            let result = &self.func.region_results[result_id];
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
                label,
                output_size,
                lambda_id,
            } => {
                self.linearize_lambda(body, label, output_size, lambda_id);
            }
            NodeKindRef::Apply { target } => self.linearize_apply(node_id, target),
        }
    }

    fn linearize_simple(&mut self, node_id: NodeId, op: &IrOp) {
        let node = &self.func.nodes[node_id];
        self.record_output_scopes(node);

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
                self.emit_node(
                    node,
                    LinearOp::Const {
                        dst: data_dst(0),
                        value: *value,
                    },
                );
            }

            // ── Binary arithmetic ──
            IrOp::Add => self.emit_binop(BinOpKind::Add, node),
            IrOp::Sub => self.emit_binop(BinOpKind::Sub, node),
            IrOp::Mul => self.emit_binop(BinOpKind::Mul, node),
            IrOp::And => self.emit_binop(BinOpKind::And, node),
            IrOp::Or => self.emit_binop(BinOpKind::Or, node),
            IrOp::Shr => self.emit_binop(BinOpKind::Shr, node),
            IrOp::Shl => self.emit_binop(BinOpKind::Shl, node),
            IrOp::Xor => self.emit_binop(BinOpKind::Xor, node),
            IrOp::CmpEq => self.emit_binop(BinOpKind::CmpEq, node),
            IrOp::CmpNe => self.emit_binop(BinOpKind::CmpNe, node),
            IrOp::CmpLt => self.emit_binop(BinOpKind::CmpLt, node),
            IrOp::CmpLe => self.emit_binop(BinOpKind::CmpLe, node),
            IrOp::CmpGt => self.emit_binop(BinOpKind::CmpGt, node),
            IrOp::CmpGe => self.emit_binop(BinOpKind::CmpGe, node),

            // ── Unary ──
            IrOp::ZigzagDecode { wide } => {
                self.emit_node(
                    node,
                    LinearOp::UnaryOp {
                        op: UnaryOpKind::ZigzagDecode { wide: *wide },
                        dst: data_dst(0),
                        src: data_in(0),
                    },
                );
            }
            IrOp::SignExtend { from_width } => {
                self.emit_node(
                    node,
                    LinearOp::UnaryOp {
                        op: UnaryOpKind::SignExtend {
                            from_width: *from_width,
                        },
                        dst: data_dst(0),
                        src: data_in(0),
                    },
                );
            }

            // ── Cursor ops ──
            IrOp::BoundsCheck { count } => {
                self.emit_node(node, LinearOp::BoundsCheck { count: *count });
            }
            IrOp::ReadBytes { count } => {
                self.emit_node(
                    node,
                    LinearOp::ReadBytes {
                        dst: data_dst(0),
                        count: *count,
                    },
                );
            }
            IrOp::PeekByte => {
                self.emit_node(node, LinearOp::PeekByte { dst: data_dst(0) });
            }
            IrOp::AdvanceCursor { count } => {
                self.emit_node(node, LinearOp::AdvanceCursor { count: *count });
            }
            IrOp::AdvanceCursorBy => {
                self.emit_node(node, LinearOp::AdvanceCursorBy { src: data_in(0) });
            }
            IrOp::SaveCursor => {
                self.emit_node(node, LinearOp::SaveCursor { dst: data_dst(0) });
            }
            IrOp::SaveInputEnd => {
                self.emit_node(node, LinearOp::SaveInputEnd { dst: data_dst(0) });
            }
            IrOp::RestoreCursor => {
                self.emit_node(node, LinearOp::RestoreCursor { src: data_in(0) });
            }

            // ── Output ops ──
            IrOp::WriteToField { offset, width } => {
                self.emit_node(
                    node,
                    LinearOp::WriteToField {
                        src: data_in(0),
                        offset: *offset,
                        width: *width,
                    },
                );
            }
            IrOp::ReadFromField { offset, width } => {
                self.emit_node(
                    node,
                    LinearOp::ReadFromField {
                        dst: data_dst(0),
                        offset: *offset,
                        width: *width,
                    },
                );
            }
            IrOp::SaveOutPtr => {
                self.emit_node(node, LinearOp::SaveOutPtr { dst: data_dst(0) });
            }
            IrOp::SetOutPtr => {
                self.emit_node(node, LinearOp::SetOutPtr { src: data_in(0) });
            }

            // ── Stack ops ──
            IrOp::SlotAddr { slot } => {
                self.emit_node(
                    node,
                    LinearOp::SlotAddr {
                        dst: data_dst(0),
                        slot: *slot,
                    },
                );
            }
            IrOp::StoreToAddr { width } => {
                self.emit_node(
                    node,
                    LinearOp::StoreToAddr {
                        addr: data_in(0),
                        src: data_in(1),
                        width: *width,
                    },
                );
            }
            IrOp::LoadFromAddr { width } => {
                self.emit_node(
                    node,
                    LinearOp::LoadFromAddr {
                        dst: data_dst(0),
                        addr: data_in(0),
                        width: *width,
                    },
                );
            }
            IrOp::WriteToSlot { slot } => {
                self.emit_node(
                    node,
                    LinearOp::WriteToSlot {
                        slot: *slot,
                        src: data_in(0),
                    },
                );
            }
            IrOp::ReadFromSlot { slot } => {
                self.emit_node(
                    node,
                    LinearOp::ReadFromSlot {
                        dst: data_dst(0),
                        slot: *slot,
                    },
                );
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
                self.emit_node(
                    node,
                    LinearOp::CallIntrinsic {
                        func: *func,
                        args,
                        dst,
                        field_offset: *field_offset,
                    },
                );
            }
            IrOp::CallPure { func, arg_count } => {
                let args: Vec<VReg> = (0..*arg_count as usize).map(&data_in).collect();
                self.emit_node(
                    node,
                    LinearOp::CallPure {
                        func: *func,
                        args,
                        dst: data_dst(0),
                    },
                );
            }

            // ── Error ──
            IrOp::ErrorExit { code } => {
                self.emit_node(node, LinearOp::ErrorExit { code: *code });
            }

            // ── SIMD ──
            IrOp::SimdStringScan => {
                self.emit_node(
                    node,
                    LinearOp::SimdStringScan {
                        pos: data_dst(0),
                        kind: data_dst(1),
                    },
                );
            }
            IrOp::SimdWhitespaceSkip => {
                self.emit_node(node, LinearOp::SimdWhitespaceSkip);
            }
            IrOp::Nop => {
                // No-op; skip.
            }
            IrOp::Identity => {
                let dst = node.outputs[0].vreg.expect("identity must have vreg");
                let src = self.resolve_vreg(node.inputs[0].source);
                self.record_vreg_scope(dst, node.outputs[0].debug_scope);
                self.emit_node(node, LinearOp::Copy { dst, src });
            }
        }
    }

    fn emit_binop(&mut self, op: BinOpKind, node: &Node) {
        let dst = node.outputs[0].vreg.expect("binop must have vreg");
        let lhs = self.resolve_vreg(node.inputs[0].source);
        let rhs = self.resolve_vreg(node.inputs[1].source);
        self.record_vreg_scope(dst, node.outputs[0].debug_scope);
        self.emit_node(node, LinearOp::BinOp { op, dst, lhs, rhs });
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

        // Determine the data output count from the gamma node.
        let data_output_count = node
            .outputs
            .iter()
            .filter(|o| o.kind == PortKind::Data)
            .count();

        // Collect entry phi_args for each branch: (source=gamma_input, target=branch_region_arg).
        // Include even self-referential phis (src == dst) since slot2reg can make
        // gamma inputs and region args share the same vreg.
        let state_count = self.func.state_domains.len();
        let passthrough_count = node.inputs.len() - 1 - state_count;
        let mut branch_entry_phis: Vec<Vec<(VReg, VReg)>> = Vec::new();
        for &region_id in regions.iter() {
            let region = &self.func.regions[region_id];
            let mut phis = Vec::new();
            for i in 0..passthrough_count {
                let src_input = &node.inputs[i + 1]; // +1 to skip predicate
                if src_input.kind == PortKind::Data {
                    let src_vreg = self.resolve_vreg(src_input.source);
                    let arg = &self.func.region_args[region.args[i]];
                    if let Some(dst_vreg) = arg.vreg {
                        self.record_vreg_scope(dst_vreg, region.debug_scope);
                        phis.push((src_vreg, dst_vreg));
                    }
                }
            }
            branch_entry_phis.push(phis);
        }

        // Emit JumpTable if > 2 branches, or BranchIfZero for 2-branch case.
        if branch_count == 2 {
            // predicate==0 → branch 0, predicate!=0 → branch 1
            // BranchIfZero: taken=branch 0 (phi_args), fallthrough=next instruction (branch 1's Branch)
            self.emit(
                Some(node.debug_scope),
                LinearOp::BranchIfZero {
                    cond: predicate,
                    target: branch_labels[0],
                    phi_args: branch_entry_phis[0].clone(),
                    fallthrough_phi_args: vec![],
                },
            );
            self.emit(
                Some(node.debug_scope),
                LinearOp::Branch {
                    target: branch_labels[1],
                    phi_args: branch_entry_phis[1].clone(),
                },
            );
        } else {
            // General case: jump table — entry phis not supported yet, emit as copies.
            self.emit(
                Some(node.debug_scope),
                LinearOp::JumpTable {
                    predicate,
                    labels: branch_labels.clone(),
                    default: branch_labels[branch_count - 1],
                },
            );
        }

        // Emit each branch.
        for (branch_idx, &region_id) in regions.iter().enumerate() {
            self.emit(
                Some(self.func.regions[region_id].debug_scope),
                LinearOp::Label(branch_labels[branch_idx]),
            );

            // For jump table (>2 branches), emit entry copies since JumpTable
            // can't carry phi_args yet.
            if branch_count > 2 {
                self.emit_gamma_entry_copies(node, region_id);
            }

            // Linearize the branch body.
            self.linearize_region(region_id);

            // Skip exit and merge branch if the branch contains an error exit
            // (code after error_exit is unreachable and causes regalloc issues).
            let ends_with_error = self.region_is_error_only(region_id);

            if !ends_with_error {
                // Collect exit phi_args: (source=branch_result, target=gamma_output).
                // Note: we must include phi_args even when src == dst, because after
                // slot2reg promotion the same vreg may be used as both the branch
                // result and the gamma output. Without the phi_arg, the vreg would
                // be undefined at the merge point.
                let region = &self.func.regions[region_id];
                let mut exit_phis = Vec::new();
                for i in 0..data_output_count {
                    let result = &self.func.region_results[region.results[i]];
                    if result.kind == PortKind::Data {
                        let src_vreg = self.resolve_vreg(result.source);
                        let dst_vreg = node.outputs[i]
                            .vreg
                            .expect("gamma data output must have vreg");
                        self.record_vreg_scope(dst_vreg, node.outputs[i].debug_scope);
                        exit_phis.push((src_vreg, dst_vreg));
                    }
                }

                // Always emit Branch to merge (even for last branch) to carry phi_args.
                self.emit(
                    Some(self.func.regions[region_id].debug_scope),
                    LinearOp::Branch {
                        target: merge_label,
                        phi_args: exit_phis,
                    },
                );
            }
        }

        self.emit(Some(node.debug_scope), LinearOp::Label(merge_label));
    }

    /// Check if a region's ONLY code path is an error exit (no normal return).
    /// A region is error-only if it directly contains ErrorExit and no gamma/theta
    /// nodes (which would provide alternative code paths).
    fn region_is_error_only(&self, region_id: RegionId) -> bool {
        let region = &self.func.regions[region_id];
        let has_error = region.nodes.iter().any(|&nid| {
            matches!(
                &self.func.nodes[nid].kind,
                NodeKind::Simple(IrOp::ErrorExit { .. })
            )
        });
        let has_structural = region.nodes.iter().any(|&nid| {
            matches!(
                &self.func.nodes[nid].kind,
                NodeKind::Gamma { .. } | NodeKind::Theta { .. }
            )
        });
        // If the region has an error exit and no structural nodes (gamma/theta),
        // it's purely an error path.
        has_error && !has_structural
    }

    /// Emit Copy ops for passthrough data inputs entering a gamma branch region.
    fn emit_gamma_entry_copies(&mut self, node: &Node, region_id: RegionId) {
        let region = &self.func.regions[region_id];
        let state_count = self.func.state_domains.len();
        // Inputs layout: [predicate, passthrough..., state domains...]
        // Region args layout: [passthrough..., state domains...]
        // Skip predicate (input 0), skip state inputs at the end.
        let passthrough_count = node.inputs.len() - 1 - state_count;

        for i in 0..passthrough_count {
            let src_input = &node.inputs[i + 1]; // +1 to skip predicate
            if src_input.kind == PortKind::Data {
                let src_vreg = self.resolve_vreg(src_input.source);
                let arg = &self.func.region_args[region.args[i]];
                if let Some(dst_vreg) = arg.vreg
                    && src_vreg != dst_vreg
                {
                    self.emit(
                        Some(region.debug_scope),
                        LinearOp::Copy {
                            dst: dst_vreg,
                            src: src_vreg,
                        },
                    );
                    self.record_vreg_scope(dst_vreg, region.debug_scope);
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
            let result = &self.func.region_results[region.results[i]];
            if result.kind == PortKind::Data {
                let src_vreg = self.resolve_vreg(result.source);
                let dst_vreg = node.outputs[i]
                    .vreg
                    .expect("gamma data output must have vreg");
                if src_vreg != dst_vreg {
                    self.record_vreg_scope(dst_vreg, node.outputs[i].debug_scope);
                    self.emit(
                        Some(node.outputs[i].debug_scope),
                        LinearOp::Copy {
                            dst: dst_vreg,
                            src: src_vreg,
                        },
                    );
                }
            }
        }
    }

    // ─── Theta (loop) ────────────────────────────────────────────────

    fn linearize_theta(&mut self, node_id: NodeId, body: RegionId) {
        let node = &self.func.nodes[node_id];
        let body_region = &self.func.regions[body];
        let state_count = self.func.state_domains.len();

        // Theta inputs: [loop_vars..., state domains...]
        // Body args: [loop_vars..., state domains...]
        // Body results: [predicate, loop_vars..., state domains...]
        // Theta outputs: [loop_vars..., state domains...]

        let total_inputs = node.inputs.len();
        let loop_var_count = total_inputs - state_count;

        // Collect initial phi args: (source=init_value, target=body_arg_vreg).
        // These replace the old "initial copies" — the Branch to the loop header
        // carries the initial values explicitly.
        let mut entry_phi_args = Vec::new();
        for i in 0..loop_var_count {
            let input = &node.inputs[i];
            if input.kind == PortKind::Data {
                let src_vreg = self.resolve_vreg(input.source);
                let arg = &self.func.region_args[body_region.args[i]];
                if let Some(dst_vreg) = arg.vreg {
                    self.record_vreg_scope(dst_vreg, body_region.debug_scope);
                    entry_phi_args.push((src_vreg, dst_vreg));
                }
            }
        }

        // Loop top label — body arg vregs are block params here (one def each).
        let loop_top = self.fresh_label();
        let loop_exit = self.fresh_label();
        self.emit(
            Some(body_region.debug_scope),
            LinearOp::Branch {
                target: loop_top,
                phi_args: entry_phi_args,
            },
        );
        self.emit(Some(body_region.debug_scope), LinearOp::Label(loop_top));

        // Linearize the body.
        self.linearize_region(body);

        // Body results: [predicate, loop_vars..., cursor_state, output_state]
        // predicate: 0 = exit, nonzero = continue
        let predicate_result = &self.func.region_results[body_region.results[0]];
        let predicate_vreg = self.resolve_vreg(predicate_result.source);

        // Collect feedback phi args: (source=result_vreg, target=body_arg_vreg).
        // These replace the old "feedback copies" — the back-edge BranchIf carries
        // the feedback values explicitly. No Copy instruction redefines the body
        // arg vregs, preserving SSA (one def per vreg at the block param).
        let mut feedback_phi_args = Vec::new();
        for i in 0..loop_var_count {
            let result = &self.func.region_results[body_region.results[i + 1]]; // +1 to skip predicate
            if result.kind == PortKind::Data {
                let src_vreg = self.resolve_vreg(result.source);
                let arg = &self.func.region_args[body_region.args[i]];
                if let Some(dst_vreg) = arg.vreg
                    && src_vreg != dst_vreg
                {
                    feedback_phi_args.push((src_vreg, dst_vreg));
                }
            }
        }

        // Emit exit copies BEFORE the BranchIf — these copy from the body result
        // vregs (not body arg vregs) to the theta output vregs. They execute every
        // iteration but are only live on the exit path.
        for i in 0..loop_var_count {
            let result = &self.func.region_results[body_region.results[i + 1]];
            if result.kind == PortKind::Data {
                let src_vreg = self.resolve_vreg(result.source);
                if let Some(dst_vreg) = node.outputs[i].vreg
                    && src_vreg != dst_vreg
                {
                    self.record_vreg_scope(dst_vreg, node.outputs[i].debug_scope);
                    self.emit(
                        Some(node.outputs[i].debug_scope),
                        LinearOp::Copy {
                            dst: dst_vreg,
                            src: src_vreg,
                        },
                    );
                }
            }
        }

        // Branch back to loop top if predicate is nonzero.
        // The back-edge carries feedback values as phi args.
        self.emit(
            Some(body_region.debug_scope),
            LinearOp::BranchIf {
                cond: predicate_vreg,
                target: loop_top,
                phi_args: feedback_phi_args,
                fallthrough_phi_args: vec![],
            },
        );

        self.emit(Some(node.debug_scope), LinearOp::Label(loop_exit));
    }

    // ─── Lambda ──────────────────────────────────────────────────────

    fn linearize_lambda(
        &mut self,
        body: RegionId,
        label: &str,
        output_size: usize,
        lambda_id: LambdaId,
    ) {
        let region = &self.func.regions[body];
        let data_args: Vec<VReg> = region
            .args
            .iter()
            .map(|&arg_id| &self.func.region_args[arg_id])
            .filter(|a| a.kind == PortKind::Data)
            .map(|a| a.vreg.expect("lambda data arg must have vreg assigned"))
            .collect();
        let data_results: Vec<VReg> = region
            .results
            .iter()
            .map(|&result_id| &self.func.region_results[result_id])
            .filter(|r| r.kind == PortKind::Data)
            .map(|r| self.resolve_vreg(r.source))
            .collect();
        for arg in &data_args {
            self.record_vreg_scope(*arg, region.debug_scope);
        }
        self.emit(
            Some(region.debug_scope),
            LinearOp::FuncStart {
                lambda_id,
                label: label.to_owned(),
                output_size,
                data_args,
                data_results,
            },
        );
        self.linearize_region(body);
        self.emit(Some(region.debug_scope), LinearOp::FuncEnd);
    }

    // ─── Apply ───────────────────────────────────────────────────────

    fn linearize_apply(&mut self, node_id: NodeId, target: LambdaId) {
        let node = &self.func.nodes[node_id];
        self.record_output_scopes(node);
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
        self.emit(
            Some(node.debug_scope),
            LinearOp::CallLambda {
                target,
                args,
                results,
            },
        );
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
        LinearOp::Branch { .. }
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
        LinearOp::StoreToAddr { addr, src, .. } => vec![*addr, *src],
        LinearOp::LoadFromAddr { addr, .. } => vec![*addr],
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
        | LinearOp::SaveInputEnd { .. }
        | LinearOp::ReadFromField { .. }
        | LinearOp::SaveOutPtr { .. }
        | LinearOp::SlotAddr { .. }
        | LinearOp::ReadFromSlot { .. }
        | LinearOp::Label(_)
        | LinearOp::Branch { .. }
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
        LinearOp::SaveInputEnd { dst } => vec![*dst],
        LinearOp::ReadFromField { dst, .. } => vec![*dst],
        LinearOp::SaveOutPtr { dst } => vec![*dst],
        LinearOp::SlotAddr { dst, .. } => vec![*dst],
        LinearOp::LoadFromAddr { dst, .. } => vec![*dst],
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
        | LinearOp::StoreToAddr { .. }
        | LinearOp::WriteToSlot { .. }
        | LinearOp::Label(_)
        | LinearOp::Branch { .. }
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
        LinearOp::StoreToAddr { addr, src, .. } => {
            rewrite(addr, &mut resolve);
            rewrite(src, &mut resolve);
        }
        LinearOp::LoadFromAddr { addr, .. } => rewrite(addr, &mut resolve),
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
        | LinearOp::SaveInputEnd { .. }
        | LinearOp::ReadFromField { .. }
        | LinearOp::SaveOutPtr { .. }
        | LinearOp::SlotAddr { .. }
        | LinearOp::ReadFromSlot { .. }
        | LinearOp::Label(_)
        | LinearOp::Branch { .. }
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
            LinearOp::Branch { target: label, .. } => {
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

fn optimize_linear_ops(
    ops: &mut Vec<LinearOp>,
    op_scopes: &mut Vec<Option<DebugScopeId>>,
    op_values: &mut Vec<Option<DebugValueId>>,
) {
    assert_eq!(
        ops.len(),
        op_scopes.len(),
        "linear op scopes must stay aligned with ops",
    );
    assert_eq!(
        ops.len(),
        op_values.len(),
        "linear op debug values must stay aligned with ops",
    );
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
    let old_scopes = std::mem::take(op_scopes);
    let old_values = std::mem::take(op_values);
    *ops = old_ops
        .into_iter()
        .enumerate()
        .filter_map(|(i, op)| (!remove[i]).then_some(op))
        .collect();
    *op_scopes = old_scopes
        .into_iter()
        .enumerate()
        .filter_map(|(i, scope)| (!remove[i]).then_some(scope))
        .collect();
    *op_values = old_values
        .into_iter()
        .enumerate()
        .filter_map(|(i, debug_value)| (!remove[i]).then_some(debug_value))
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
        label: &'a str,
        output_size: usize,
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
            label,
            output_size,
            lambda_id,
        } => NodeKindRef::Lambda {
            body: *body,
            label,
            output_size: *output_size,
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
        let arg_ids: Vec<_> = func.regions[region_id].args.clone();
        for arg_id in arg_ids {
            if func.region_args[arg_id].kind == PortKind::Data
                && func.region_args[arg_id].vreg.is_none()
            {
                let vreg = func.fresh_vreg();
                func.region_args[arg_id].vreg = Some(vreg);
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
    let mut op_scopes = lin.op_scopes;
    let mut op_values = lin.op_values;
    optimize_linear_ops(&mut ops, &mut op_scopes, &mut op_values);

    LinearIr {
        ops,
        label_count: lin.label_count,
        vreg_count: func.vreg_count(),
        slot_count: func.slot_count(),
        debug: LinearDebugProvenance {
            scopes: func.debug_scopes.clone(),
            values: func.debug_values.clone(),
            root_scope: Some(func.root_debug_scope),
            op_scopes,
            op_values,
            vreg_scopes: lin.vreg_scopes,
            vreg_values: lin.vreg_values,
        },
    }
}

// ─── Display ─────────────────────────────────────────────────────────────────

impl fmt::Display for LinearIr {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let display = LinearIrDisplay {
            linear: self,
            registry: None,
        };
        fmt::Display::fmt(&display, f)
    }
}

pub struct LinearIrDisplay<'a> {
    linear: &'a LinearIr,
    registry: Option<&'a IntrinsicRegistry>,
}

impl<'a> fmt::Display for LinearIrDisplay<'a> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for op in &self.linear.ops {
            // Labels get no indentation, everything else gets 2 spaces.
            match op {
                LinearOp::Label(label) => {
                    writeln!(f, "L{}:", label.index())?;
                }
                LinearOp::FuncStart {
                    lambda_id, label, ..
                } => {
                    writeln!(f, "func λ{} ({label}):", lambda_id.index())?;
                }
                LinearOp::FuncEnd => {
                    writeln!(f, "end")?;
                }
                _ => {
                    write!(f, "  ")?;
                    fmt_op(f, op, self.registry)?;
                    writeln!(f)?;
                }
            }
        }
        Ok(())
    }
}

impl LinearIr {
    pub fn display_with_registry<'a>(
        &'a self,
        registry: &'a IntrinsicRegistry,
    ) -> LinearIrDisplay<'a> {
        LinearIrDisplay {
            linear: self,
            registry: Some(registry),
        }
    }
}

fn fmt_vreg(f: &mut fmt::Formatter<'_>, v: VReg) -> fmt::Result {
    write!(f, "v{}", v.index())
}

fn fmt_op(
    f: &mut fmt::Formatter<'_>,
    op: &LinearOp,
    registry: Option<&IntrinsicRegistry>,
) -> fmt::Result {
    match op {
        LinearOp::Const { dst, value } => {
            fmt_vreg(f, *dst)?;
            write!(f, " = const ")?;
            fmt_const(f, *value, registry)
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
        LinearOp::SaveInputEnd { dst } => {
            fmt_vreg(f, *dst)?;
            write!(f, " = save_input_end")
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
        LinearOp::StoreToAddr { addr, src, width } => {
            write!(f, "store_addr [{width}] ")?;
            fmt_vreg(f, *addr)?;
            write!(f, ", ")?;
            fmt_vreg(f, *src)
        }
        LinearOp::LoadFromAddr { dst, addr, width } => {
            fmt_vreg(f, *dst)?;
            write!(f, " = load_addr [{width}] ")?;
            fmt_vreg(f, *addr)
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
            write!(f, "call_intrinsic ")?;
            fmt_intrinsic(f, *func, registry)?;
            write!(f, "(")?;
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
            write!(f, " = call_pure ")?;
            fmt_intrinsic(f, *func, registry)?;
            write!(f, "(")?;
            for (i, a) in args.iter().enumerate() {
                if i > 0 {
                    write!(f, ", ")?;
                }
                fmt_vreg(f, *a)?;
            }
            write!(f, ")")
        }
        LinearOp::Branch { target, .. } => write!(f, "br L{}", target.index()),
        LinearOp::BranchIf { cond, target, .. } => {
            write!(f, "br_if ")?;
            fmt_vreg(f, *cond)?;
            write!(f, " L{}", target.index())
        }
        LinearOp::BranchIfZero { cond, target, .. } => {
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

fn fmt_intrinsic(
    f: &mut fmt::Formatter<'_>,
    func: IntrinsicFn,
    registry: Option<&IntrinsicRegistry>,
) -> fmt::Result {
    if let Some(registry) = registry
        && let Some(name) = registry.name_of(func)
    {
        return write!(f, "@{name}");
    }
    write!(f, "{func}")
}

fn fmt_const(
    f: &mut fmt::Formatter<'_>,
    value: u64,
    registry: Option<&IntrinsicRegistry>,
) -> fmt::Result {
    if let Some(registry) = registry
        && let Some(name) = registry.const_name_of(value)
    {
        return write!(f, "@{name}");
    }
    write!(f, "{value}")
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
    use kajit_ir::{
        DebugScope, DebugScopeKind, IrBuilder, IrOp, LambdaId, PortSource, VReg, Width,
    };

    #[test]
    fn linearize_simple_chain() {
        // BoundsCheck(4) → ReadBytes(4) → WriteToField(offset=0, W4)
        let mut builder = IrBuilder::new("u32", 0);
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
    fn linearize_preserves_debug_scope_provenance() {
        let mut builder = IrBuilder::new("u32", 0);
        let (const_node, output_index, root_scope) = {
            let mut rb = builder.root_region();
            let value = rb.const_val(42);
            rb.set_results(&[value]);
            let output_ref = match value {
                PortSource::Node(output_ref) => output_ref,
                other => panic!("expected node output, got {other:?}"),
            };
            (output_ref.node, output_ref.index as usize, rb.debug_scope())
        };

        let mut func = builder.finish();
        let value_vreg = func.nodes[const_node].outputs[output_index]
            .vreg
            .expect("expected vreg on const output");
        let extra_scope = func.debug_scopes.push(DebugScope {
            parent: Some(root_scope),
            kind: DebugScopeKind::ThetaBody,
        });
        func.nodes[const_node].debug_scope = extra_scope;
        func.nodes[const_node].outputs[0].debug_scope = root_scope;

        let linear = linearize(&mut func);
        assert_eq!(linear.debug.root_scope, Some(root_scope));
        assert_eq!(linear.debug.scopes.len(), func.debug_scopes.len());
        assert_eq!(
            linear.debug.vreg_scopes[value_vreg.index()],
            Some(root_scope)
        );

        let const_scope = linear
            .ops
            .iter()
            .zip(linear.debug.op_scopes.iter())
            .find_map(|(op, scope)| match op {
                LinearOp::Const { dst, .. } if *dst == value_vreg => *scope,
                _ => None,
            });
        assert_eq!(const_scope, Some(extra_scope));
    }

    #[test]
    fn linearize_gamma_two_branches() {
        // Gamma with predicate, 2 branches:
        //   branch 0: const 42 → result
        //   branch 1: const 99 → result
        let mut builder = IrBuilder::new("u32", 0);
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
        let mut builder = IrBuilder::new("u32", 0);
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
        use kajit_ir::IntrinsicFn;

        unsafe extern "C" fn dummy_intrinsic(_ctx: *mut core::ffi::c_void) {}

        let mut builder = IrBuilder::new("bool", 0);
        {
            let mut rb = builder.root_region();
            rb.bounds_check(1);
            rb.call_intrinsic(
                IntrinsicFn(dummy_intrinsic as *const () as usize),
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
        let mut builder = IrBuilder::new("u32", 0);
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
                label: "u32".into(),
                output_size: 0,
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

        let mut op_scopes = vec![None; ops.len()];
        let mut op_values = vec![None; ops.len()];
        optimize_linear_ops(&mut ops, &mut op_scopes, &mut op_values);

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
                label: "u32".into(),
                output_size: 0,
                data_args: vec![],
                data_results: vec![v1],
            },
            LinearOp::Const { dst: v0, value: 9 },
            LinearOp::Copy { dst: v1, src: v0 },
            LinearOp::FuncEnd,
        ];

        let mut op_scopes = vec![None; ops.len()];
        let mut op_values = vec![None; ops.len()];
        optimize_linear_ops(&mut ops, &mut op_scopes, &mut op_values);

        assert!(
            ops.iter()
                .any(|op| matches!(op, LinearOp::Copy { dst, src } if *dst == v1 && *src == v0)),
            "copy into function result vreg must be preserved"
        );
    }

    #[test]
    fn optimize_linear_ops_keeps_debug_values_aligned_with_rewritten_ops() {
        let v0 = VReg::new(0);
        let v1 = VReg::new(1);
        let v2 = VReg::new(2);
        let debug_value = DebugValueId::new(0);
        let mut ops = vec![
            LinearOp::FuncStart {
                lambda_id: LambdaId::new(0),
                label: "u32".into(),
                output_size: 0,
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
        let mut op_scopes = vec![None; ops.len()];
        let mut op_values = vec![None; ops.len()];
        op_values[4] = Some(debug_value);

        optimize_linear_ops(&mut ops, &mut op_scopes, &mut op_values);

        assert_eq!(ops.len(), op_values.len(), "debug values must stay aligned");
        let write_index = ops
            .iter()
            .position(|op| matches!(op, LinearOp::WriteToField { .. }))
            .expect("optimized ops should still contain write");
        assert_eq!(
            op_values[write_index],
            Some(debug_value),
            "semantic debug value should stay attached to the write op",
        );
    }

    #[test]
    fn linearize_theta_gamma_passthrough_after_slot2reg() {
        // Theta with gamma inside, one branch doesn't modify the slot.
        // After slot2reg, the slot becomes a loop-carried variable.
        let input = r#"
lambda @0 (shape: "test") {
  region {
    args: [%cs, %os]
    n0 = Const(0x0) [] -> [v0]
    n1 = WriteToSlot(0) [v0, %cs:arg] -> [%cs]
    n14 = theta [%cs:n1, %os:arg] {
      region {
        args: [%cs, %os]
        n2 = ReadFromSlot(0) [%cs:arg] -> [v1, %cs]
        n3 = Const(0x4) [] -> [v2]
        n4 = CmpNe [v1, v2] -> [v3]
        n11 = gamma [
          pred: v3
          in0: %cs:n2
          in1: %os:arg
        ] {
          branch 0:
            region {
              args: [%cs, %os]
              n5 = ReadFromSlot(0) [%cs:arg] -> [v4, %cs]
              n6 = Const(0x1) [] -> [v5]
              n7 = Add [v4, v5] -> [v6]
              n8 = WriteToSlot(0) [v6, %cs:n5] -> [%cs]
              results: [%cs:n8, %os:arg]
            }
          branch 1:
            region {
              args: [%cs, %os]
              results: [%cs:arg, %os:arg]
            }
        } -> [%cs, %os]
        n12 = Const(0x0) [] -> [v7]
        results: [v7, %cs:n11, %os:n11]
      }
    } -> [%cs, %os]
    n13 = ReadFromSlot(0) [%cs:n14] -> [v8, %cs]
    n15 = WriteToField(offset=0, W4) [v8, %os:n14] -> [%os]
    results: [%cs:n13, %os:n15]
  }
}
"#;
        let registry = kajit_ir::IntrinsicRegistry::empty();
        let mut func = kajit_ir_text::parse_ir(input, &registry).unwrap();
        kajit_ir::slot2reg::slot_to_reg(&mut func);
        let _ir = linearize(&mut func);
    }

    #[test]
    fn linearize_real_array_u32_4_after_slot2reg() {
        let result = std::thread::Builder::new()
            .stack_size(8 * 1024 * 1024)
            .spawn(linearize_real_array_u32_4_impl)
            .unwrap()
            .join()
            .unwrap();
        if let Err(msg) = result {
            panic!("{msg}");
        }
    }

    fn linearize_real_array_u32_4_impl() -> Result<(), String> {
        let input = include_str!("../tests/array_u32_4_after_slot2reg.vixen-ir");
        let registry = kajit_ir::IntrinsicRegistry::empty();
        let mut func =
            kajit_ir_text::parse_ir(input, &registry).map_err(|e| format!("parse failed: {e}"))?;
        let ir = linearize(&mut func);
        let mut self_copies = vec![];
        for (i, op) in ir.ops.iter().enumerate() {
            if let LinearOp::Copy { dst, src } = op {
                if dst == src {
                    self_copies.push(format!(
                        "  op[{i}]: Copy v{} -> v{}",
                        src.index(),
                        dst.index()
                    ));
                }
            }
        }
        if !self_copies.is_empty() {
            return Err(format!("self-copies found:\n{}", self_copies.join("\n")));
        }
        Ok(())
    }

    #[test]
    fn linearize_theta_shared_predicate_and_loopvar() {
        // Theta where a gamma output is used both as predicate AND as a
        // loop-carried variable result — the pattern from the real array
        // decoder that triggers v_N from v_N in regalloc2.
        let input = r#"
lambda @0 (shape: "test") {
  region {
    args: [%cs, %os]
    n0 = Const(0x0) [] -> [v0]
    n1 = Const(0x1) [] -> [v1]
    n10 = theta [v0, v1, %cs:arg, %os:arg] {
      region {
        args: [arg0, arg1, %cs, %os]
        n2 = Const(0x4) [] -> [v2]
        n3 = CmpNe [arg0, v2] -> [v3]
        n8 = gamma [
          pred: v3
          in0: arg0
          in1: arg1
          in2: %cs:arg
          in3: %os:arg
        ] {
          branch 0:
            region {
              args: [arg0, arg1, %cs, %os]
              n4 = Const(0x1) [] -> [v4]
              n5 = Add [arg0, v4] -> [v5]
              results: [v5, arg1, %cs:arg, %os:arg]
            }
          branch 1:
            region {
              args: [arg0, arg1, %cs, %os]
              results: [arg0, arg1, %cs:arg, %os:arg]
            }
        } -> [v6, v7, %cs, %os]
        results: [v7, v6, v7, %cs:n8, %os:n8]
      }
    } -> [v8, v9, %cs, %os]
    n9 = WriteToField(offset=0, W4) [v8, %os:n10] -> [%os]
    results: [%cs:n10, %os:n9]
  }
}
"#;
        let registry = kajit_ir::IntrinsicRegistry::empty();
        let mut func = kajit_ir_text::parse_ir(input, &registry).unwrap();
        let _ir = linearize(&mut func);
    }
}
