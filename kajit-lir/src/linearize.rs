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
    Sar,
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

impl LinearOp {
    /// Visit every VReg *use* (read) in this op, mutably.
    pub fn for_each_use_mut(&mut self, mut f: impl FnMut(&mut VReg)) {
        use LinearOp::*;
        match self {
            // Values
            Const { .. } => {}
            BinOp { lhs, rhs, .. } => {
                f(lhs);
                f(rhs);
            }
            UnaryOp { src, .. } | Copy { src, .. } => f(src),

            // Cursor
            BoundsCheck { .. } | AdvanceCursor { .. } => {}
            ReadBytes { .. } | PeekByte { .. } | SaveCursor { .. } | SaveInputEnd { .. } => {}
            AdvanceCursorBy { src } | RestoreCursor { src } => f(src),

            // Output
            WriteToField { src, .. } => f(src),
            ReadFromField { .. } => {}
            SaveOutPtr { .. } => {}
            SetOutPtr { src } => f(src),

            // Stack
            SlotAddr { .. } => {}
            StoreToAddr { addr, src, .. } => {
                f(addr);
                f(src);
            }
            LoadFromAddr { addr, .. } => f(addr),
            WriteToSlot { src, .. } => f(src),
            ReadFromSlot { .. } => {}

            // Calls
            CallIntrinsic { args, .. } => {
                for arg in args {
                    f(arg);
                }
            }
            CallPure { args, .. } => {
                for arg in args {
                    f(arg);
                }
            }

            // Control flow
            Label(_) | ErrorExit { .. } => {}
            Branch { phi_args, .. } => {
                for (src, _dst) in phi_args {
                    f(src);
                }
            }
            BranchIf {
                cond,
                phi_args,
                fallthrough_phi_args,
                ..
            } => {
                f(cond);
                for (src, _dst) in phi_args {
                    f(src);
                }
                for (src, _dst) in fallthrough_phi_args {
                    f(src);
                }
            }
            BranchIfZero {
                cond,
                phi_args,
                fallthrough_phi_args,
                ..
            } => {
                f(cond);
                for (src, _dst) in phi_args {
                    f(src);
                }
                for (src, _dst) in fallthrough_phi_args {
                    f(src);
                }
            }
            JumpTable { predicate, .. } => f(predicate),

            // SIMD
            SimdStringScan { pos, kind } => {
                f(pos);
                f(kind);
            }
            SimdWhitespaceSkip => {}

            // Function structure
            FuncStart { data_args, .. } => {
                for arg in data_args {
                    f(arg);
                }
            }
            FuncEnd => {}
            CallLambda { args, .. } => {
                for arg in args {
                    f(arg);
                }
            }
        }
    }

    /// Visit every VReg *use* (read) in this op, immutably.
    pub fn for_each_use(&self, mut f: impl FnMut(&VReg)) {
        // Clone and delegate to mutable version (avoids duplicating the match)
        let mut clone = self.clone();
        clone.for_each_use_mut(|v| f(v));
    }

    /// Visit every VReg *definition* (write) in this op, immutably.
    pub fn for_each_def(&self, mut f: impl FnMut(&VReg)) {
        let mut clone = self.clone();
        clone.for_each_def_mut(|v| f(v));
    }

    /// Visit every VReg *definition* (write) in this op, mutably.
    pub fn for_each_def_mut(&mut self, mut f: impl FnMut(&mut VReg)) {
        use LinearOp::*;
        match self {
            Const { dst, .. }
            | BinOp { dst, .. }
            | UnaryOp { dst, .. }
            | Copy { dst, .. }
            | ReadBytes { dst, .. }
            | PeekByte { dst }
            | SaveCursor { dst }
            | SaveInputEnd { dst }
            | ReadFromField { dst, .. }
            | SaveOutPtr { dst }
            | SlotAddr { dst, .. }
            | LoadFromAddr { dst, .. }
            | ReadFromSlot { dst, .. }
            | CallPure { dst, .. } => f(dst),

            CallIntrinsic { dst, .. } => {
                if let Some(dst) = dst {
                    f(dst);
                }
            }

            // Phi targets in branches
            Branch { phi_args, .. } => {
                for (_src, dst) in phi_args {
                    f(dst);
                }
            }
            BranchIf {
                phi_args,
                fallthrough_phi_args,
                ..
            }
            | BranchIfZero {
                phi_args,
                fallthrough_phi_args,
                ..
            } => {
                for (_src, dst) in phi_args {
                    f(dst);
                }
                for (_src, dst) in fallthrough_phi_args {
                    f(dst);
                }
            }

            // Function structure defs
            FuncStart { data_results, .. } => {
                for r in data_results {
                    f(r);
                }
            }
            CallLambda { results, .. } => {
                for r in results {
                    f(r);
                }
            }

            // No defs
            BoundsCheck { .. }
            | AdvanceCursor { .. }
            | AdvanceCursorBy { .. }
            | RestoreCursor { .. }
            | WriteToField { .. }
            | SetOutPtr { .. }
            | StoreToAddr { .. }
            | WriteToSlot { .. }
            | Label(_)
            | ErrorExit { .. }
            | JumpTable { .. }
            | SimdStringScan { .. }
            | SimdWhitespaceSkip
            | FuncEnd => {}
        }
    }

    /// Visit every VReg in this op (both uses and defs), mutably.
    pub fn for_each_vreg_mut(&mut self, mut f: impl FnMut(&mut VReg)) {
        use LinearOp::*;
        match self {
            Const { dst, .. } => f(dst),
            BinOp { dst, lhs, rhs, .. } => {
                f(dst);
                f(lhs);
                f(rhs);
            }
            UnaryOp { dst, src, .. } | Copy { dst, src } => {
                f(dst);
                f(src);
            }

            BoundsCheck { .. } | AdvanceCursor { .. } => {}
            ReadBytes { dst, .. } => f(dst),
            PeekByte { dst } => f(dst),
            AdvanceCursorBy { src } => f(src),
            SaveCursor { dst } => f(dst),
            SaveInputEnd { dst } => f(dst),
            RestoreCursor { src } => f(src),

            WriteToField { src, .. } => f(src),
            ReadFromField { dst, .. } => f(dst),
            SaveOutPtr { dst } => f(dst),
            SetOutPtr { src } => f(src),

            SlotAddr { dst, .. } => f(dst),
            StoreToAddr { addr, src, .. } => {
                f(addr);
                f(src);
            }
            LoadFromAddr { dst, addr, .. } => {
                f(dst);
                f(addr);
            }
            WriteToSlot { src, .. } => f(src),
            ReadFromSlot { dst, .. } => f(dst),

            CallIntrinsic { args, dst, .. } => {
                for arg in args {
                    f(arg);
                }
                if let Some(dst) = dst {
                    f(dst);
                }
            }
            CallPure { args, dst, .. } => {
                for arg in args {
                    f(arg);
                }
                f(dst);
            }

            Label(_) | ErrorExit { .. } => {}
            Branch { phi_args, .. } => {
                for (src, dst) in phi_args {
                    f(src);
                    f(dst);
                }
            }
            BranchIf {
                cond,
                phi_args,
                fallthrough_phi_args,
                ..
            } => {
                f(cond);
                for (src, dst) in phi_args {
                    f(src);
                    f(dst);
                }
                for (src, dst) in fallthrough_phi_args {
                    f(src);
                    f(dst);
                }
            }
            BranchIfZero {
                cond,
                phi_args,
                fallthrough_phi_args,
                ..
            } => {
                f(cond);
                for (src, dst) in phi_args {
                    f(src);
                    f(dst);
                }
                for (src, dst) in fallthrough_phi_args {
                    f(src);
                    f(dst);
                }
            }
            JumpTable { predicate, .. } => f(predicate),

            SimdStringScan { pos, kind } => {
                f(pos);
                f(kind);
            }
            SimdWhitespaceSkip => {}

            FuncStart {
                data_args,
                data_results,
                ..
            } => {
                for arg in data_args {
                    f(arg);
                }
                for r in data_results {
                    f(r);
                }
            }
            FuncEnd => {}
            CallLambda { args, results, .. } => {
                for arg in args {
                    f(arg);
                }
                for r in results {
                    f(r);
                }
            }
        }
    }
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

/// Context for the active passthrough-exit chain, passed to inner gammas so
/// they can emit direct exits to the chain's landing instead of creating
/// merge blocks for control-only outputs.
#[derive(Clone)]
struct ChainExitCtx {
    landing_label: LabelId,
    landing_vregs: Vec<VReg>,
    state_env: Vec<VReg>,
    output_to_landing: Vec<usize>,
}

struct Linearizer<'a> {
    func: &'a IrFunc,
    ops: Vec<LinearOp>,
    label_count: u32,
    op_scopes: Vec<Option<DebugScopeId>>,
    op_values: Vec<Option<DebugValueId>>,
    vreg_scopes: Vec<Option<DebugScopeId>>,
    vreg_values: Vec<Option<DebugValueId>>,
    /// Active passthrough-exit chain context. When set, inner gammas with a
    /// passthrough branch can use this to emit direct exits to the chain's
    /// landing instead of merging control-only flag outputs.
    chain_exit_ctx: Option<ChainExitCtx>,
    /// Counter for allocating fresh vregs (starts above all RVSDG vregs).
    next_vreg: u32,
}

impl<'a> Linearizer<'a> {
    fn new(func: &'a IrFunc) -> Self {
        Self {
            func,
            ops: Vec::new(),
            chain_exit_ctx: None,
            next_vreg: func.vreg_count,
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
        if vreg.index() >= self.vreg_scopes.len() {
            self.vreg_scopes.resize(vreg.index() + 1, None);
        }
        self.vreg_scopes[vreg.index()] = Some(scope);
    }

    fn record_vreg_value(&mut self, vreg: VReg, debug_value: DebugValueId) {
        if vreg.index() >= self.vreg_values.len() {
            self.vreg_values.resize(vreg.index() + 1, None);
        }
        self.vreg_values[vreg.index()] = Some(debug_value);
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
                NodeKind::Theta { body, .. } => {
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
                NodeKind::Theta { body, .. } => {
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
            NodeKindRef::Theta { body, .. } => self.linearize_theta(node_id, body),
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
            IrOp::Sar => self.emit_binop(BinOpKind::Sar, node),
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
            IrOp::SlotAddr { slot, .. } => {
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
        // Try one-sided gamma lowering: if one branch is terminal (error exit),
        // emit as conditional branch + inline continuation — no merge block.
        if self.try_linearize_one_sided_gamma(node_id, regions) {
            return;
        }

        // Try passthrough-exit lowering: if one branch is data-passthrough,
        // emit conditional exit to shared landing block + inline the other.
        if self.try_linearize_passthrough_exit(node_id, regions, None) {
            return;
        }

        // Try chain-exit lowering: if we're inside a passthrough-exit chain's
        // continue branch and this gamma has a passthrough branch, lower the
        // non-passthrough ("done") branch as a direct exit to the chain's landing.
        // This eliminates the merge block for control-only flag outputs.
        if self.try_linearize_inner_chain_exit(node_id, regions) {
            return;
        }

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

    // ─── One-sided gamma lowering ──────────────────────────────────────

    /// Try to lower a gamma as a conditional exit (no merge block).
    ///
    /// Applicable when exactly one branch is terminal (error exit / unreachable)
    /// and the other branch is the continuation. The terminal branch becomes a
    /// conditional branch to an error/exit label, and the continuation branch
    /// is emitted inline. No merge block is created.
    ///
    /// The gamma's output vregs are defined by the continuation branch's results
    /// (via Copy instructions from branch results → gamma outputs).
    fn try_linearize_one_sided_gamma(&mut self, node_id: NodeId, regions: &[RegionId]) -> bool {
        if regions.len() != 2 {
            return false;
        }

        let branch0_terminal = self.region_is_error_only(regions[0]);
        let branch1_terminal = self.region_is_error_only(regions[1]);

        // Exactly one branch must be terminal
        if branch0_terminal == branch1_terminal {
            return false; // both terminal or neither
        }

        let (terminal_branch, continue_branch, terminal_is_zero) = if branch0_terminal {
            (regions[0], regions[1], true) // pred==0 → error, pred!=0 → continue
        } else {
            (regions[1], regions[0], false) // pred!=0 → error, pred==0 → continue
        };

        let node = &self.func.nodes[node_id];
        let predicate = self.resolve_vreg(node.inputs[0].source);
        let state_count = self.func.state_domains.len();
        let passthrough_count = node.inputs.len() - 1 - state_count;

        let data_output_count = node
            .outputs
            .iter()
            .filter(|o| o.kind == PortKind::Data)
            .count();

        // Build entry phis for the terminal branch
        let terminal_region = &self.func.regions[terminal_branch];
        let mut terminal_entry_phis = Vec::new();
        for i in 0..passthrough_count {
            let src_input = &node.inputs[i + 1];
            if src_input.kind == PortKind::Data {
                let arg = &self.func.region_args[terminal_region.args[i]];
                if let Some(dst_vreg) = arg.vreg {
                    let src_vreg = self.resolve_vreg(src_input.source);
                    self.record_vreg_scope(dst_vreg, terminal_region.debug_scope);
                    terminal_entry_phis.push((src_vreg, dst_vreg));
                }
            }
        }

        // Build entry phis for the continue branch
        let continue_region = &self.func.regions[continue_branch];
        let mut continue_entry_phis = Vec::new();
        for i in 0..passthrough_count {
            let src_input = &node.inputs[i + 1];
            if src_input.kind == PortKind::Data {
                let arg = &self.func.region_args[continue_region.args[i]];
                if let Some(dst_vreg) = arg.vreg {
                    let src_vreg = self.resolve_vreg(src_input.source);
                    self.record_vreg_scope(dst_vreg, continue_region.debug_scope);
                    continue_entry_phis.push((src_vreg, dst_vreg));
                }
            }
        }

        // Emit conditional branch to the terminal path
        let terminal_label = self.fresh_label();

        if terminal_is_zero {
            // pred==0 → terminal (error), pred!=0 → continue (fallthrough)
            self.emit(
                Some(node.debug_scope),
                LinearOp::BranchIfZero {
                    cond: predicate,
                    target: terminal_label,
                    phi_args: terminal_entry_phis,
                    fallthrough_phi_args: continue_entry_phis,
                },
            );
        } else {
            // pred!=0 → terminal (error), pred==0 → continue (fallthrough)
            self.emit(
                Some(node.debug_scope),
                LinearOp::BranchIf {
                    cond: predicate,
                    target: terminal_label,
                    phi_args: terminal_entry_phis,
                    fallthrough_phi_args: continue_entry_phis,
                },
            );
        }

        // Emit the continuation branch inline (no label, just fall through)
        self.linearize_region(continue_branch);

        // Copy continuation results → gamma output vregs
        let continue_region = &self.func.regions[continue_branch];
        for i in 0..data_output_count {
            let result = &self.func.region_results[continue_region.results[i]];
            if result.kind == PortKind::Data {
                let src_vreg = self.resolve_vreg(result.source);
                let dst_vreg = node.outputs[i]
                    .vreg
                    .expect("gamma data output must have vreg");
                self.record_vreg_scope(dst_vreg, node.outputs[i].debug_scope);
                if src_vreg != dst_vreg {
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

        // Emit the terminal branch body (after the continuation, as a side path)
        let after_label = self.fresh_label();
        self.emit(
            Some(node.debug_scope),
            LinearOp::Branch {
                target: after_label,
                phi_args: vec![],
            },
        );
        self.emit(
            Some(terminal_region.debug_scope),
            LinearOp::Label(terminal_label),
        );
        self.linearize_region(terminal_branch);
        // Terminal branch is error-only, no branch to merge needed
        self.emit(Some(node.debug_scope), LinearOp::Label(after_label));

        true
    }

    // ─── Passthrough-exit gamma lowering ────────────────────────────────

    /// Check if a region is data-passthrough: each data result equals the
    /// corresponding incoming region arg. Ignores non-data computation.
    fn is_data_passthrough_region(&self, region_id: RegionId) -> bool {
        let region = &self.func.regions[region_id];
        let data_results: Vec<_> = region
            .results
            .iter()
            .enumerate()
            .filter(|&(_, &rid)| self.func.region_results[rid].kind == PortKind::Data)
            .collect();
        let data_args: Vec<_> = region
            .args
            .iter()
            .enumerate()
            .filter(|&(_, &aid)| self.func.region_args[aid].kind == PortKind::Data)
            .collect();
        if data_results.len() != data_args.len() {
            return false;
        }
        for (&(_, &rid), &(_, &aid)) in data_results.iter().zip(data_args.iter()) {
            match self.func.region_results[rid].source {
                PortSource::RegionArg(arg_ref) => {
                    if arg_ref.region != region_id || arg_ref.arg != aid {
                        return false;
                    }
                }
                _ => return false,
            }
        }
        true
    }

    /// Lower a passthrough-exit gamma as a conditional exit to a shared landing.
    ///
    /// `exit_ctx`: if Some, contains (landing_label, landing_vregs, state_env).
    /// state_env[i] = current vreg for landing param i in the environment.
    ///
    /// The key idea: exits branch to the landing block carrying the CURRENT
    /// environment's state, not the inner gamma's raw outputs. This handles
    /// arity mismatches between nested gammas naturally.
    /// `exit_ctx`: (landing_label, landing_vregs, state_env, output_to_landing)
    /// output_to_landing[j] = index into state_env/landing_vregs for this gamma's output j.
    fn try_linearize_passthrough_exit(
        &mut self,
        node_id: NodeId,
        regions: &[RegionId],
        exit_ctx: Option<(LabelId, Vec<VReg>, Vec<VReg>, Vec<usize>)>,
    ) -> bool {
        if regions.len() != 2 {
            return false;
        }
        let b0_pt = self.is_data_passthrough_region(regions[0]);
        let b1_pt = self.is_data_passthrough_region(regions[1]);
        if !b0_pt && !b1_pt {
            return false;
        }
        if b0_pt && b1_pt {
            return false;
        }
        let (cont_branch, exit_on_zero) = if b0_pt {
            (regions[1], true)
        } else {
            (regions[0], false)
        };
        if self.region_is_error_only(cont_branch) {
            return false;
        }

        // Only apply at top level if there's a chain to exploit.
        // A single passthrough-exit gamma without chaining just creates a
        // local landing (same cost as a merge block).
        if exit_ctx.is_none() && self.find_tail_passthrough_gamma(cont_branch).is_none() {
            return false;
        }

        let node = &self.func.nodes[node_id];
        let predicate = self.resolve_vreg(node.inputs[0].source);
        let state_count = self.func.state_domains.len();
        let passthrough_count = node.inputs.len() - 1 - state_count;
        let data_output_count = node
            .outputs
            .iter()
            .filter(|o| o.kind == PortKind::Data)
            .count();

        // Create or reuse landing block
        let owns_landing = exit_ctx.is_none();
        let (landing_label, landing_vregs, mut state_env, output_to_landing) = exit_ctx
            .unwrap_or_else(|| {
                let label = self.fresh_label();
                let vregs: Vec<VReg> = (0..data_output_count)
                    .map(|i| {
                        let v = node.outputs[i].vreg.expect("gamma output vreg");
                        self.record_vreg_scope(v, node.outputs[i].debug_scope);
                        v
                    })
                    .collect();
                // Initial state = gamma inputs (current environment)
                let state: Vec<VReg> = (0..data_output_count)
                    .map(|i| self.resolve_vreg(node.inputs[i + 1].source))
                    .collect();
                // Identity mapping: output j = landing param j
                let mapping: Vec<usize> = (0..data_output_count).collect();
                (label, vregs, state, mapping)
            });

        // Exit phis: project current state_env onto landing params.
        // On the passthrough exit, the state_env already has the right values
        // (passthrough = no change = current env is correct).
        let exit_phis: Vec<(VReg, VReg)> = state_env
            .iter()
            .zip(landing_vregs.iter())
            .filter(|(s, d)| s != d)
            .map(|(s, d)| (*s, *d))
            .collect();

        // Entry phis for continue branch
        let cont_region = &self.func.regions[cont_branch];
        let mut cont_entry_phis = Vec::new();
        for i in 0..passthrough_count {
            let src_input = &node.inputs[i + 1];
            if src_input.kind == PortKind::Data {
                let arg = &self.func.region_args[cont_region.args[i]];
                if let Some(dst_vreg) = arg.vreg {
                    let src_vreg = self.resolve_vreg(src_input.source);
                    self.record_vreg_scope(dst_vreg, cont_region.debug_scope);
                    cont_entry_phis.push((src_vreg, dst_vreg));
                }
            }
        }

        // Conditional exit to landing
        if exit_on_zero {
            self.emit(
                Some(node.debug_scope),
                LinearOp::BranchIfZero {
                    cond: predicate,
                    target: landing_label,
                    phi_args: exit_phis,
                    fallthrough_phi_args: cont_entry_phis,
                },
            );
        } else {
            self.emit(
                Some(node.debug_scope),
                LinearOp::BranchIf {
                    cond: predicate,
                    target: landing_label,
                    phi_args: exit_phis,
                    fallthrough_phi_args: cont_entry_phis,
                },
            );
        }

        // Check for tail passthrough gamma to chain into
        let tail_gamma = self.find_tail_passthrough_gamma(cont_branch);
        if let Some(tail_id) = tail_gamma {
            // Linearize everything except the tail gamma, with chain exit context
            // so inner gammas can emit direct exits to the landing.
            let prev_ctx = self.chain_exit_ctx.take();
            self.chain_exit_ctx = Some(ChainExitCtx {
                landing_label,
                landing_vregs: landing_vregs.clone(),
                state_env: state_env.clone(),
                output_to_landing: output_to_landing.clone(),
            });
            let node_ids: Vec<NodeId> = self.func.regions[cont_branch].nodes.clone();
            for &nid in &node_ids {
                if nid != tail_id {
                    self.linearize_node(nid);
                }
            }
            self.chain_exit_ctx = prev_ctx;

            // Update state_env for entries that DON'T come from the tail gamma.
            // Build the output_to_landing mapping for the tail gamma.
            let cont_results = self.func.regions[cont_branch].results.clone();
            let cont_data_results: Vec<usize> = cont_results
                .iter()
                .enumerate()
                .filter(|&(_, &rid)| self.func.region_results[rid].kind == PortKind::Data)
                .map(|(i, _)| i)
                .collect();

            // For each data result of the continue region: it maps to
            // this gamma's output[j], which maps to landing param output_to_landing[j].
            let mut tail_output_to_landing: Vec<usize> = Vec::new();
            for (j, &result_idx) in cont_data_results.iter().enumerate() {
                let landing_idx = if j < output_to_landing.len() {
                    let idx = output_to_landing[j];
                    if idx == usize::MAX {
                        continue;
                    }
                    idx
                } else {
                    continue;
                };
                let result = &self.func.region_results[cont_results[result_idx]];
                match result.source {
                    PortSource::Node(out_ref) if out_ref.node == tail_id => {
                        // This result comes from the tail gamma's output.
                        // Map tail gamma output[out_ref.index] → landing_idx.
                        let tail_out_idx = out_ref.index as usize;
                        while tail_output_to_landing.len() <= tail_out_idx {
                            tail_output_to_landing.push(usize::MAX);
                        }
                        tail_output_to_landing[tail_out_idx] = landing_idx;

                        // ALSO update state_env to the tail gamma's input for
                        // this output. Since the exit path is passthrough, when
                        // the tail gamma exits, it carries its input values.
                        // The input for data output j is inputs[j+1] (skip pred).
                        let tail_node = &self.func.nodes[tail_id];
                        let tail_data_inputs: Vec<usize> = tail_node
                            .inputs
                            .iter()
                            .enumerate()
                            .skip(1) // skip predicate
                            .filter(|(_, inp)| inp.kind == PortKind::Data)
                            .map(|(idx, _)| idx)
                            .collect();
                        if tail_out_idx < tail_data_inputs.len() {
                            let input_idx = tail_data_inputs[tail_out_idx];
                            state_env[landing_idx] =
                                self.resolve_vreg(tail_node.inputs[input_idx].source);
                        }
                    }
                    _ => {
                        // From earlier computation — update state now
                        state_env[landing_idx] = self.resolve_vreg(result.source);
                    }
                }
            }

            eprintln!(
                "[passthrough-exit] chain: tail gamma #{}, tail_output_to_landing={:?}, state_env={:?}",
                tail_id.index(),
                tail_output_to_landing,
                state_env
                    .iter()
                    .enumerate()
                    .map(|(i, v)| format!("{}→v{}", i, v.index()))
                    .collect::<Vec<_>>()
            );

            // Keep placeholders — indexing must match tail gamma output positions.
            // Entries with usize::MAX mean "this output doesn't map to any landing param."

            // Recurse with updated state and tail mapping
            let NodeKind::Gamma { regions: tr } = &self.func.nodes[tail_id].kind else {
                unreachable!()
            };
            let tr = tr.clone();
            self.try_linearize_passthrough_exit(
                tail_id,
                &tr,
                Some((
                    landing_label,
                    landing_vregs.clone(),
                    state_env,
                    tail_output_to_landing,
                )),
            );
        } else {
            // No chain — linearize full continue region with chain exit context
            let prev_ctx = self.chain_exit_ctx.take();
            self.chain_exit_ctx = Some(ChainExitCtx {
                landing_label,
                landing_vregs: landing_vregs.clone(),
                state_env: state_env.clone(),
                output_to_landing: output_to_landing.clone(),
            });
            self.linearize_region(cont_branch);
            self.chain_exit_ctx = prev_ctx;

            // Branch to landing with continue region's results mapped to landing params
            let cont_results = self.func.regions[cont_branch].results.clone();
            let cont_data_results: Vec<usize> = cont_results
                .iter()
                .enumerate()
                .filter(|&(_, &rid)| self.func.region_results[rid].kind == PortKind::Data)
                .map(|(i, _)| i)
                .collect();
            let mut final_phis = Vec::new();
            for (j, &result_idx) in cont_data_results.iter().enumerate() {
                if j >= output_to_landing.len() {
                    break;
                }
                let landing_idx = output_to_landing[j];
                if landing_idx == usize::MAX {
                    continue;
                }
                let result = &self.func.region_results[cont_results[result_idx]];
                if result.kind == PortKind::Data {
                    let src = self.resolve_vreg(result.source);
                    final_phis.push((src, landing_vregs[landing_idx]));
                }
            }
            // Also carry any state_env values for landing params not covered
            // by this gamma's outputs (they keep their current values)
            for (i, &lv) in landing_vregs.iter().enumerate() {
                if !final_phis.iter().any(|(_, d)| *d == lv) {
                    final_phis.push((state_env[i], lv));
                }
            }
            self.emit(
                Some(self.func.regions[cont_branch].debug_scope),
                LinearOp::Branch {
                    target: landing_label,
                    phi_args: final_phis,
                },
            );
        }

        if owns_landing {
            self.emit(Some(node.debug_scope), LinearOp::Label(landing_label));
        }
        true
    }

    /// Lower an inner gamma as a chain exit: when we're inside a passthrough-exit
    /// chain's continue branch and this gamma has a passthrough branch ("more data")
    /// and a non-passthrough branch ("done"), the "done" branch exits directly to
    /// the chain's landing. The passthrough branch falls through inline.
    ///
    /// The done branch's exit values are computed analytically: for each gamma output,
    /// resolve the done branch's result to either a passthrough (gamma input) or a
    /// constant. This avoids emitting the done branch body inline and the control
    /// flow problems that would cause.
    fn try_linearize_inner_chain_exit(&mut self, node_id: NodeId, regions: &[RegionId]) -> bool {
        let ctx = match &self.chain_exit_ctx {
            Some(ctx) => ctx.clone(),
            None => return false,
        };

        if regions.len() != 2 {
            return false;
        }

        let b0_pt = self.is_data_passthrough_region(regions[0]);
        let b1_pt = self.is_data_passthrough_region(regions[1]);
        if !b0_pt && !b1_pt {
            return false;
        }
        if b0_pt && b1_pt {
            return false;
        }

        let (continue_branch, done_branch, done_on_nonzero) = if b0_pt {
            (regions[0], regions[1], true)
        } else {
            (regions[1], regions[0], false)
        };

        eprintln!(
            "[inner-chain-exit] considering gamma #{} (done_branch has {} nodes, {} results)",
            node_id.index(),
            self.func.regions[done_branch].nodes.len(),
            self.func.regions[done_branch].results.len()
        );

        let node = &self.func.nodes[node_id];
        let state_count = self.func.state_domains.len();
        let passthrough_count = node.inputs.len() - 1 - state_count;
        let data_output_count = node
            .outputs
            .iter()
            .filter(|o| o.kind == PortKind::Data)
            .count();

        // Analytically resolve the done branch's data results.
        // Each result must be either a region arg (passthrough = gamma input)
        // or resolvable to a constant. If any result can't be resolved, bail out.
        let done_results = self.func.regions[done_branch].results.clone();
        let done_data_results: Vec<usize> = done_results
            .iter()
            .enumerate()
            .filter(|&(_, &rid)| self.func.region_results[rid].kind == PortKind::Data)
            .map(|(i, _)| i)
            .collect();

        eprintln!(
            "[inner-chain-exit]   done_data_results={:?}",
            done_data_results
        );

        // For each done data result, determine the exit value.
        // Collect results first WITHOUT emitting, then emit only if all succeed.
        enum DoneExitValue {
            Passthrough(usize), // index into gamma inputs (+1 for predicate)
            Const {
                vreg: VReg,
                value: u64,
                scope: Option<DebugScopeId>,
            },
        }
        let mut done_exit_plan: Vec<DoneExitValue> = Vec::new();
        let mut allocated_region_vregs: usize = 0;
        for &result_idx in &done_data_results {
            let result = &self.func.region_results[done_results[result_idx]];
            match result.source {
                PortSource::RegionArg(arg_ref) => {
                    let arg_pos = self.func.regions[done_branch]
                        .args
                        .iter()
                        .position(|a| *a == arg_ref.arg);
                    if let Some(pos) = arg_pos {
                        if pos < passthrough_count {
                            done_exit_plan.push(DoneExitValue::Passthrough(pos + 1));
                            continue;
                        }
                    }
                    eprintln!(
                        "[inner-chain-exit]   result[{}]: RegionArg FAILED (pos={:?}, passthrough_count={})",
                        result_idx, arg_pos, passthrough_count
                    );
                    return false;
                }
                PortSource::Node(out_ref) => {
                    let source_node = &self.func.nodes[out_ref.node];
                    if let NodeKind::Simple(IrOp::Const { value }) = &source_node.kind {
                        let vreg = self.fresh_vreg();
                        done_exit_plan.push(DoneExitValue::Const {
                            vreg,
                            value: *value,
                            scope: None,
                        });
                        continue;
                    }
                    // Try resolving through the RVSDG (handles nested gammas with known predicates)
                    if let Some(val) =
                        kajit_ir::const_fold::resolve_to_constant(self.func, &result.source)
                    {
                        let vreg = self.fresh_vreg();
                        done_exit_plan.push(DoneExitValue::Const {
                            vreg,
                            value: val,
                            scope: None,
                        });
                        continue;
                    }
                    return false;
                }
            }
        }

        // All done results resolved — now emit const instructions and build exit vregs
        let mut done_exit_vregs: Vec<VReg> = Vec::new();
        for plan in &done_exit_plan {
            match plan {
                DoneExitValue::Passthrough(input_idx) => {
                    done_exit_vregs.push(self.resolve_vreg(node.inputs[*input_idx].source));
                }
                DoneExitValue::Const { vreg, value, scope } => {
                    self.record_vreg_scope(*vreg, scope.unwrap_or(node.debug_scope));
                    self.emit(
                        Some(node.debug_scope),
                        LinearOp::Const {
                            dst: *vreg,
                            value: *value,
                        },
                    );
                    done_exit_vregs.push(*vreg);
                }
            }
        }

        eprintln!(
            "[inner-chain-exit] gamma #{}: done_exit_vregs={:?}, data_output_count={}",
            node_id.index(),
            done_exit_vregs
                .iter()
                .map(|v| v.index())
                .collect::<Vec<_>>(),
            data_output_count
        );

        // All done results resolved. Now emit the lowering.
        let predicate = self.resolve_vreg(node.inputs[0].source);

        // Build exit phis for the done path → landing
        let mut exit_phis: Vec<(VReg, VReg)> = Vec::new();
        let mut used_landing = std::collections::HashSet::new();
        for (j, &done_vreg) in done_exit_vregs.iter().enumerate() {
            if j >= ctx.output_to_landing.len() {
                break;
            }
            let landing_idx = ctx.output_to_landing[j];
            if landing_idx == usize::MAX {
                continue;
            }
            let dst = ctx.landing_vregs[landing_idx];
            if done_vreg != dst {
                exit_phis.push((done_vreg, dst));
            }
            used_landing.insert(landing_idx);
        }
        for (i, &lv) in ctx.landing_vregs.iter().enumerate() {
            if !used_landing.contains(&i) {
                let src = ctx.state_env[i];
                if src != lv {
                    exit_phis.push((src, lv));
                }
            }
        }

        // Entry phis for the continue branch (passthrough)
        let cont_region = &self.func.regions[continue_branch];
        let mut cont_entry_phis = Vec::new();
        for i in 0..passthrough_count {
            let src_input = &node.inputs[i + 1];
            if src_input.kind == PortKind::Data {
                let arg = &self.func.region_args[cont_region.args[i]];
                if let Some(dst_vreg) = arg.vreg {
                    let src_vreg = self.resolve_vreg(src_input.source);
                    self.record_vreg_scope(dst_vreg, cont_region.debug_scope);
                    cont_entry_phis.push((src_vreg, dst_vreg));
                }
            }
        }

        // Emit: done path branches to landing, continue path falls through
        if done_on_nonzero {
            self.emit(
                Some(node.debug_scope),
                LinearOp::BranchIf {
                    cond: predicate,
                    target: ctx.landing_label,
                    phi_args: exit_phis,
                    fallthrough_phi_args: cont_entry_phis,
                },
            );
        } else {
            self.emit(
                Some(node.debug_scope),
                LinearOp::BranchIfZero {
                    cond: predicate,
                    target: ctx.landing_label,
                    phi_args: exit_phis,
                    fallthrough_phi_args: cont_entry_phis,
                },
            );
        }

        // For the passthrough (continue) branch: gamma outputs = gamma inputs.
        for i in 0..data_output_count {
            let out_vreg = node.outputs[i].vreg.expect("gamma output vreg");
            let in_vreg = self.resolve_vreg(node.inputs[i + 1].source);
            self.record_vreg_scope(out_vreg, node.outputs[i].debug_scope);
            if out_vreg != in_vreg {
                self.emit(
                    Some(node.debug_scope),
                    LinearOp::Copy {
                        dst: out_vreg,
                        src: in_vreg,
                    },
                );
            }
        }

        true
    }

    /// Allocate a fresh vreg (for synthetic instructions like chain-exit consts).
    fn fresh_vreg(&mut self) -> VReg {
        let v = VReg::new(self.next_vreg);
        self.next_vreg += 1;
        v
    }

    /// Find a tail gamma that feeds all data results and has a passthrough branch.
    fn find_tail_passthrough_gamma(&self, region_id: RegionId) -> Option<NodeId> {
        let region = &self.func.regions[region_id];
        let gamma_node = region
            .nodes
            .iter()
            .rev()
            .find(|&&nid| matches!(&self.func.nodes[nid].kind, NodeKind::Gamma { .. }))?;
        for &rid in &region.results {
            if self.func.region_results[rid].kind != PortKind::Data {
                continue;
            }
            match self.func.region_results[rid].source {
                PortSource::Node(out_ref) if out_ref.node == *gamma_node => {}
                _ => return None,
            }
        }
        let NodeKind::Gamma { regions } = &self.func.nodes[*gamma_node].kind else {
            return None;
        };
        if regions.len() != 2 {
            return None;
        }
        if !self.is_data_passthrough_region(regions[0])
            && !self.is_data_passthrough_region(regions[1])
        {
            return None;
        }
        Some(*gamma_node)
    }

    /// Linearize all nodes in a region EXCEPT the specified node.
    fn linearize_region_except(&mut self, region_id: RegionId, except_node: NodeId) {
        let region = &self.func.regions[region_id];
        for &nid in &region.nodes {
            if nid != except_node {
                self.linearize_node(nid);
            }
        }
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
    #[allow(dead_code)]
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
        LinearOp::BranchIf {
            cond,
            phi_args,
            fallthrough_phi_args,
            ..
        } => {
            let mut v = vec![*cond];
            v.extend(phi_args.iter().map(|(src, _dst)| *src));
            v.extend(fallthrough_phi_args.iter().map(|(src, _dst)| *src));
            v
        }
        LinearOp::BranchIfZero {
            cond,
            phi_args,
            fallthrough_phi_args,
            ..
        } => {
            let mut v = vec![*cond];
            v.extend(phi_args.iter().map(|(src, _dst)| *src));
            v.extend(fallthrough_phi_args.iter().map(|(src, _dst)| *src));
            v
        }
        LinearOp::Branch { phi_args, .. } => phi_args.iter().map(|(src, _dst)| *src).collect(),
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
        LinearOp::BranchIf {
            cond,
            phi_args,
            fallthrough_phi_args,
            ..
        }
        | LinearOp::BranchIfZero {
            cond,
            phi_args,
            fallthrough_phi_args,
            ..
        } => {
            rewrite(cond, &mut resolve);
            for (src, _dst) in phi_args.iter_mut() {
                rewrite(src, &mut resolve);
            }
            for (src, _dst) in fallthrough_phi_args.iter_mut() {
                rewrite(src, &mut resolve);
            }
        }
        LinearOp::Branch { phi_args, .. } => {
            for (src, _dst) in phi_args.iter_mut() {
                rewrite(src, &mut resolve);
            }
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
        NodeKind::Theta { body, .. } => NodeKindRef::Theta { body: *body },
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

/// For one-sided gammas (one branch is error-only), unify the gamma's output
/// vregs with the continuation branch's result vregs. This eliminates the need
/// for Copy instructions between the continuation result and the gamma output.
fn unify_one_sided_gamma_vregs(func: &mut IrFunc) {
    let node_count = func.nodes.len();
    for i in 0..node_count {
        let node_id = NodeId::new(i as u32);
        let NodeKind::Gamma { regions } = &func.nodes[node_id].kind else {
            continue;
        };
        if regions.len() != 2 {
            continue;
        }
        let regions = regions.clone();

        // Check which branch is error-only
        let b0_error = region_is_error_only_static(func, regions[0]);
        let b1_error = region_is_error_only_static(func, regions[1]);
        if b0_error == b1_error {
            continue; // both or neither
        }
        let continue_region = if b0_error { regions[1] } else { regions[0] };

        // For each data output: unify gamma output vreg ← continuation result source vreg
        let data_outputs: Vec<usize> = (0..func.nodes[node_id].outputs.len())
            .filter(|&j| func.nodes[node_id].outputs[j].kind == PortKind::Data)
            .collect();

        let cont_results = func.regions[continue_region].results.clone();
        for (data_idx, &output_idx) in data_outputs.iter().enumerate() {
            let gamma_vreg = func.nodes[node_id].outputs[output_idx].vreg;
            let Some(gamma_vreg) = gamma_vreg else {
                continue;
            };

            // Find the continuation result's source vreg
            let result_id = cont_results[data_idx];
            let result_source = func.region_results[result_id].source;
            match result_source {
                PortSource::Node(out_ref) => {
                    // The continuation's result comes from a node output.
                    // Set that node's output vreg to the gamma's output vreg.
                    let src_vreg = func.nodes[out_ref.node].outputs[out_ref.index as usize].vreg;
                    if src_vreg != Some(gamma_vreg) {
                        func.nodes[out_ref.node].outputs[out_ref.index as usize].vreg =
                            Some(gamma_vreg);
                    }
                }
                PortSource::RegionArg(arg_ref) => {
                    // The continuation's result is a region arg (pass-through).
                    // Set the region arg's vreg to the gamma's output vreg.
                    let arg_vreg = func.region_args[arg_ref.arg].vreg;
                    if arg_vreg != Some(gamma_vreg) {
                        func.region_args[arg_ref.arg].vreg = Some(gamma_vreg);
                    }
                }
            }
        }
    }
}

/// Check if a region is error-only (static version, no Linearizer self needed).
fn region_is_error_only_static(func: &IrFunc, region_id: RegionId) -> bool {
    let region = &func.regions[region_id];
    let has_error = region.nodes.iter().any(|&nid| {
        matches!(
            &func.nodes[nid].kind,
            NodeKind::Simple(IrOp::ErrorExit { .. })
        )
    });
    let has_structured_control = region.nodes.iter().any(|&nid| {
        matches!(
            &func.nodes[nid].kind,
            NodeKind::Gamma { .. } | NodeKind::Theta { .. }
        )
    });
    has_error && !has_structured_control
}

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

/// A level in a gamma cascade: one iteration body + its exit predicate.
struct CascadeLevel {
    /// The gamma node at this level
    gamma_node: NodeId,
    /// The "exit" branch region (branch 0, passthrough)
    #[allow(dead_code)]
    exit_region: RegionId,
    /// The "continue" branch region (branch 1, contains body + maybe next gamma)
    continue_region: RegionId,
}

// ─── Public API ──────────────────────────────────────────────────────────────

/// Linearize an RVSDG function into a flat instruction sequence.
pub fn linearize(func: &mut IrFunc) -> LinearIr {
    // Pass 1: ensure all data ports have VRegs.
    assign_vregs(func);

    // Pass 1b: unify vregs for one-sided gammas (error-exit gammas).
    // This makes the continuation result produce directly into the gamma output vreg,
    // eliminating copies when the one-sided lowering is used.
    unify_one_sided_gamma_vregs(func);

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

    // Use the linearizer's next_vreg if it allocated fresh vregs
    let vreg_count = lin.next_vreg.max(func.vreg_count());

    LinearIr {
        ops,
        label_count: lin.label_count,
        vreg_count,
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
        LinearOp::Branch { target, phi_args } => {
            write!(f, "br L{}", target.index())?;
            if !phi_args.is_empty() {
                write!(f, " phi[")?;
                for (i, (src, dst)) in phi_args.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    fmt_vreg(f, *src)?;
                    write!(f, "→")?;
                    fmt_vreg(f, *dst)?;
                }
                write!(f, "]")?;
            }
            Ok(())
        }
        LinearOp::BranchIf {
            cond,
            target,
            phi_args,
            fallthrough_phi_args,
        } => {
            write!(f, "br_if ")?;
            fmt_vreg(f, *cond)?;
            write!(f, " L{}", target.index())?;
            if !phi_args.is_empty() {
                write!(f, " phi[")?;
                for (i, (src, dst)) in phi_args.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    fmt_vreg(f, *src)?;
                    write!(f, "→")?;
                    fmt_vreg(f, *dst)?;
                }
                write!(f, "]")?;
            }
            if !fallthrough_phi_args.is_empty() {
                write!(f, " fall[")?;
                for (i, (src, dst)) in fallthrough_phi_args.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    fmt_vreg(f, *src)?;
                    write!(f, "→")?;
                    fmt_vreg(f, *dst)?;
                }
                write!(f, "]")?;
            }
            Ok(())
        }
        LinearOp::BranchIfZero {
            cond,
            target,
            phi_args,
            fallthrough_phi_args,
        } => {
            write!(f, "br_zero ")?;
            fmt_vreg(f, *cond)?;
            write!(f, " L{}", target.index())?;
            if !phi_args.is_empty() {
                write!(f, " phi[")?;
                for (i, (src, dst)) in phi_args.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    fmt_vreg(f, *src)?;
                    write!(f, "→")?;
                    fmt_vreg(f, *dst)?;
                }
                write!(f, "]")?;
            }
            if !fallthrough_phi_args.is_empty() {
                write!(f, " fall[")?;
                for (i, (src, dst)) in fallthrough_phi_args.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    fmt_vreg(f, *src)?;
                    write!(f, "→")?;
                    fmt_vreg(f, *dst)?;
                }
                write!(f, "]")?;
            }
            Ok(())
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
