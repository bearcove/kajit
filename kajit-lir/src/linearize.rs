//! Linearizer: converts the RVSDG into a flat instruction sequence.
//!
//! The RVSDG is a tree of regions and nodes. The linearizer walks this tree,
//! topologically sorts each region's nodes, and emits a flat `Vec<LinearOp>`
//! with explicit labels and branches for control flow (gamma/theta).

use std::collections::{HashMap, HashSet, VecDeque};
use std::fmt;

use kajit_ir::{
    Arena, DebugScope, DebugScopeId, DebugValue, DebugValueId, Id, IntrinsicRegistry, IrFunc, IrOp,
    LambdaId, Node, NodeId, NodeKind, OutputUseKind, PortKind, PortSource, RegionId, SlotId, VReg,
    Width,
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
    /// Load the runtime address of an embedded data blob (relocation target).
    DataAddr {
        dst: VReg,
        blob_id: u32,
    },
    /// Load the address of an external symbol (vtable function pointer etc.).
    /// The runtime address is resolved from a symbol table at emit/interpret time.
    ExternAddr {
        dst: VReg,
        symbol: kajit_types::SymbolName,
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
    },
    CallPure {
        func: IntrinsicFn,
        args: Vec<VReg>,
        dst: VReg,
    },
    /// Effectful call with direct ABI (no runtime context). Same calling
    /// convention as CallPure but must not be CSE'd or DCE'd.
    CallEffect {
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

    // ── Function structure ──
    FuncStart {
        lambda_id: LambdaId,
        label: String,
        /// Minimum output buffer size in bytes. Used by the interpreter/simulator
        /// to allocate the output buffer when static inference from StoreToAddr is insufficient.
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
            Const { .. } | DataAddr { .. } | ExternAddr { .. } => {}
            BinOp { lhs, rhs, .. } => {
                f(lhs);
                f(rhs);
            }
            UnaryOp { src, .. } | Copy { src, .. } => f(src),

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
            CallPure { args, .. } | CallEffect { args, .. } => {
                for arg in args {
                    f(arg);
                }
            }

            // Control flow
            Label(_) => {}
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
            | DataAddr { dst, .. }
            | ExternAddr { dst, .. }
            | BinOp { dst, .. }
            | UnaryOp { dst, .. }
            | Copy { dst, .. }
            | SlotAddr { dst, .. }
            | LoadFromAddr { dst, .. }
            | ReadFromSlot { dst, .. }
            | CallPure { dst, .. }
            | CallEffect { dst, .. } => f(dst),

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
            StoreToAddr { .. } | WriteToSlot { .. } | Label(_) | JumpTable { .. } | FuncEnd => {}
        }
    }

    /// Visit every VReg in this op (both uses and defs), mutably.
    pub fn for_each_vreg_mut(&mut self, mut f: impl FnMut(&mut VReg)) {
        use LinearOp::*;
        match self {
            Const { dst, .. } | DataAddr { dst, .. } | ExternAddr { dst, .. } => f(dst),
            BinOp { dst, lhs, rhs, .. } => {
                f(dst);
                f(lhs);
                f(rhs);
            }
            UnaryOp { dst, src, .. } | Copy { dst, src } => {
                f(dst);
                f(src);
            }

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
            CallPure { args, dst, .. } | CallEffect { args, dst, .. } => {
                for arg in args {
                    f(arg);
                }
                f(dst);
            }

            Label(_) => {}
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
    /// Number of data args passed in calling-convention registers.
    pub param_slot_count: u32,
    /// Preserved debug scope provenance copied from RVSDG.
    pub debug: LinearDebugProvenance,
    /// Embedded constant data blobs (string literals, etc.).
    pub data_blobs: Vec<Vec<u8>>,
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

/// A value in the safe exit environment: either an already-defined vreg
/// or a constant that needs a fresh vreg emitted at exit time.
#[derive(Clone, Copy, Debug)]
enum SafeExitVal {
    Vreg(VReg),
    Const(u64),
}

/// A group of control-only outputs that carry the same logical boolean.
struct BooleanGroupInfo {
    output_indices: Vec<usize>,
    done_value: u64,
}

/// Classification of a control-only output.
#[derive(Debug, Clone, Copy, PartialEq)]
enum CtrlOutputKind {
    /// Passthrough on ALL branches — invariant, not a real boolean.
    Invariant,
    /// One branch passthrough, other branch const — real boolean control flag.
    Boolean { done_value: u64 },
    /// Couldn't determine (complex inner structure).
    Unknown,
}

/// Classification of a chain landing position.
#[derive(Debug, Clone, Copy, PartialEq)]
enum ChainOutputClass {
    /// Real state: carries computation results consumed downstream.
    RealState,
    /// Logical boolean control flag with a known done_value constant.
    ControlBoolean { done_value: u64 },
    /// Invariant: passthrough on all branches, carries unchanged data.
    Invariant,
}

/// State inherited through recursive passthrough-exit chain calls.
/// Carries the shared landing block info plus output classification.
struct ChainInheritedState {
    landing_label: LabelId,
    landing_vregs: Vec<VReg>,
    state_env: Vec<VReg>,
    output_to_landing: Vec<usize>,
    /// Classification of each landing position. Empty if not computed.
    output_classes: Vec<ChainOutputClass>,
    /// If exactly one logical boolean exists, its done_value.
    control_done_value: Option<u64>,
}

/// Context for the active passthrough-exit chain, passed to inner gammas so
/// they can emit direct exits to the chain's landing instead of creating
/// merge blocks for control-only outputs.
#[derive(Clone)]
struct ChainExitCtx {
    landing_label: LabelId,
    landing_vregs: Vec<VReg>,
    #[allow(dead_code)]
    state_env: Vec<VReg>,
    #[allow(dead_code)]
    output_to_landing: Vec<usize>,
    /// Safe exit environment for inner chain exits. Unlike state_env, this
    /// resolves gamma outputs that may not be defined yet (because the gamma
    /// is still being linearized) into available vregs or constants.
    safe_exit_env: Vec<SafeExitVal>,
    /// Classification of each landing position: real state, control boolean,
    /// or invariant. Indexed by landing position (same as landing_vregs).
    /// Empty if classification was not computed (inner chain without top-level info).
    #[allow(dead_code)]
    output_classes: Vec<ChainOutputClass>,
    /// If exactly one logical boolean control state exists, its done_value.
    /// None if unsupported or classification was not performed.
    #[allow(dead_code)]
    control_done_value: Option<u64>,
}

/// Info about a control-state fusion: an upstream gamma whose control-only
/// output feeds a downstream gamma's predicate.
#[derive(Clone)]
struct ControlFusionInfo {
    /// The upstream gamma (deferred from normal linearization).
    upstream: NodeId,
    /// Which output of the upstream gamma carries the control boolean.
    ctrl_output_idx: usize,
    /// The done_value of the control boolean (e.g. 0 for is_more=false).
    done_value: u64,
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
    /// Theta nesting depth. Control-only exits are suppressed inside thetas
    /// to avoid issues with chain exit env scoping across loop iterations.
    theta_depth: u32,
    /// Stack of gamma nodes whose branches are currently being linearized.
    /// Used to detect if a safe_exit_env vreg is from a not-yet-merged gamma.
    gamma_stack: Vec<NodeId>,
    /// Nodes deferred from normal linearization (handled by a downstream gamma).
    deferred_nodes: HashSet<NodeId>,
    /// Fusion info for downstream gammas whose predicate is a control-only output
    /// of a deferred upstream gamma. Keyed by downstream NodeId.
    control_fusions: HashMap<NodeId, ControlFusionInfo>,
    /// Whether fusion has been applied. Only allow one fusion per linearization
    /// (the root lambda's body). Prevents issues with inner lambda bodies.
    fusion_applied: bool,
}

impl<'a> Linearizer<'a> {
    fn new(func: &'a IrFunc) -> Self {
        Self {
            func,
            ops: Vec::new(),
            chain_exit_ctx: None,
            next_vreg: func.vreg_count,
            theta_depth: 0,
            gamma_stack: Vec::new(),
            deferred_nodes: HashSet::new(),
            control_fusions: HashMap::new(),
            fusion_applied: false,
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
            if self.deferred_nodes.contains(&node_id) {
                continue;
            }
            self.linearize_node(node_id);
        }
    }

    /// Pre-scan: identify upstream gammas whose control-only output feeds a
    /// downstream gamma's predicate. Defer the upstream gamma and record fusion info.
    fn pre_scan_control_fusions(&mut self, region_id: RegionId) {
        let region = &self.func.regions[region_id];
        let initial_count = self.control_fusions.len();
        let mut new_upstreams: Vec<NodeId> = Vec::new();
        for &nid in &region.nodes {
            let node = &self.func.nodes[nid];
            let NodeKind::Gamma { regions } = &node.kind else {
                continue;
            };
            if regions.len() != 2 {
                continue;
            }

            // Check: is the predicate from a control-only gamma output in this region?
            let PortSource::Node(pred_ref) = &node.inputs[0].source else {
                continue;
            };
            let upstream_id = pred_ref.node;
            let ctrl_idx = pred_ref.index as usize;

            // Upstream must be in the same region
            if self.func.nodes[upstream_id].region != region_id {
                continue;
            }
            // Upstream must be a gamma
            let NodeKind::Gamma {
                regions: up_regions,
            } = &self.func.nodes[upstream_id].kind
            else {
                continue;
            };
            // Upstream must be a passthrough-exit candidate (has a tail gamma)
            if up_regions.len() != 2 {
                continue;
            }
            let up_b0_pt = self.is_data_passthrough_region(up_regions[0]);
            let up_b1_pt = self.is_data_passthrough_region(up_regions[1]);
            if up_b0_pt == up_b1_pt {
                continue;
            }
            let up_cont = if up_b0_pt {
                up_regions[1]
            } else {
                up_regions[0]
            };
            if self.region_is_error_only(up_cont) {
                continue;
            }
            // Upstream output must be control-only
            if !self
                .func
                .is_chain_control_only_output(upstream_id, ctrl_idx, 0)
            {
                continue;
            }
            // Classify: must be a Boolean (not Invariant)
            let ctrl_kind = self.classify_ctrl_output(upstream_id, up_regions, ctrl_idx);
            let done_value = match ctrl_kind {
                CtrlOutputKind::Boolean { done_value } => done_value,
                _ => continue,
            };

            // Downstream must be a passthrough-exit candidate
            let b0_pt = self.is_data_passthrough_region(regions[0]);
            let b1_pt = self.is_data_passthrough_region(regions[1]);
            if b0_pt == b1_pt {
                continue;
            }

            // Only fuse if the downstream gamma would start a passthrough-exit chain
            let ds_cont = if b0_pt { regions[1] } else { regions[0] };
            if self.region_is_error_only(ds_cont) {
                continue;
            }
            // Must have a chain to exploit (tail gamma in continue branch)
            if self.find_tail_passthrough_gamma(ds_cont).is_none() {
                // The downstream gamma might still be a passthrough-exit if
                // it has an exit_ctx. But for top-level fusion, we require a chain.
                continue;
            }

            // ALL data outputs of the upstream must ONLY feed the downstream gamma.
            let up_data_count = self.func.nodes[upstream_id]
                .outputs
                .iter()
                .filter(|o| o.kind == PortKind::Data)
                .count();
            let mut all_outputs_feed_downstream = true;
            for out_idx in 0..up_data_count {
                let consumers = self.func.find_output_consumers(upstream_id, out_idx as u16);
                if !consumers.iter().all(|c| match &c.kind {
                    kajit_ir::ConsumerKind::NodeInput { node: consumer, .. } => *consumer == nid,
                    _ => false,
                }) {
                    all_outputs_feed_downstream = false;
                    break;
                }
            }
            if !all_outputs_feed_downstream {
                continue;
            }

            // Don't fuse in regions that contain thetas (vec/loop types).
            let has_theta = region
                .nodes
                .iter()
                .any(|&nid2| matches!(&self.func.nodes[nid2].kind, NodeKind::Theta { .. }));
            if has_theta {
                continue;
            }

            // Structural validation: the upstream's non-passthrough branch must
            // contain an inner gamma that produces the control output, and that
            // inner gamma must have exactly 2 branches with simple structure
            // (one passthrough, one with constants for the control output).
            let up_cont_region = &self.func.regions[up_cont];
            let ctrl_result = &self.func.region_results[up_cont_region.results[ctrl_idx]];
            let inner_gamma_valid = match ctrl_result.source {
                PortSource::Node(out_ref) => {
                    if let NodeKind::Gamma {
                        regions: ig_regions,
                    } = &self.func.nodes[out_ref.node].kind
                    {
                        // Inner gamma must have exactly 2 branches, one passthrough
                        ig_regions.len() == 2
                            && (self.is_data_passthrough_region(ig_regions[0])
                                || self.is_data_passthrough_region(ig_regions[1]))
                    } else {
                        false
                    }
                }
                // Direct constant or passthrough — simple case, ok
                PortSource::RegionArg(_) => true,
            };
            if !inner_gamma_valid {
                continue;
            }

            // All checks pass. Record as a candidate.
            new_upstreams.push(upstream_id);
            self.deferred_nodes.insert(upstream_id);
            self.control_fusions.insert(
                nid,
                ControlFusionInfo {
                    upstream: upstream_id,
                    ctrl_output_idx: ctrl_idx,
                    done_value,
                },
            );

            if std::env::var("KAJIT_CHAIN_ANALYSIS").is_ok() {
                eprintln!(
                    "[fusion] candidate: upstream #{} output {} (done_val={}) → downstream #{} predicate",
                    upstream_id.index(),
                    ctrl_idx,
                    done_value,
                    nid.index()
                );
            }
        }

        // Safety: only commit fusions found in THIS scan if there's exactly one
        // new candidate. Multiple new candidates indicate compound types —
        // revert only the new ones to avoid miscompilation.
        let new_count = self.control_fusions.len() - initial_count;
        if new_count > 1 {
            if std::env::var("KAJIT_CHAIN_ANALYSIS").is_ok() {
                eprintln!(
                    "[fusion] {} new candidates in region, reverting (compound type safety)",
                    new_count
                );
            }
            // Remove only the new candidates
            for &up in &new_upstreams {
                self.deferred_nodes.remove(&up);
            }
            // Remove fusions added in this scan
            self.control_fusions
                .retain(|_, v| !new_upstreams.contains(&v.upstream));
        } else if new_count == 1 && std::env::var("KAJIT_CHAIN_ANALYSIS").is_ok() {
            eprintln!("[fusion] committed: 1 candidate");
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
            IrOp::DataAddr { blob_id } => {
                self.emit_node(
                    node,
                    LinearOp::DataAddr {
                        dst: data_dst(0),
                        blob_id: *blob_id,
                    },
                );
            }
            IrOp::ExternAddr { symbol } => {
                self.emit_node(
                    node,
                    LinearOp::ExternAddr {
                        dst: data_dst(0),
                        symbol: symbol.clone(),
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
            } => {
                let args: Vec<VReg> = (0..*arg_count as usize).map(&data_in).collect();
                let dst = if *has_result { Some(data_dst(0)) } else { None };
                self.emit_node(
                    node,
                    LinearOp::CallIntrinsic {
                        func: *func,
                        args,
                        dst,
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
            IrOp::CallEffect { func, arg_count } => {
                let args: Vec<VReg> = (0..*arg_count as usize).map(&data_in).collect();
                self.emit_node(
                    node,
                    LinearOp::CallEffect {
                        func: *func,
                        args,
                        dst: data_dst(0),
                    },
                );
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

        // Control-only chain exit: if ALL data outputs are control-only booleans
        // (is_more flags), the "done" branch exits directly to the chain landing
        // and the "more" (passthrough) branch falls through.
        if self.try_linearize_control_only_chain_exit(node_id, regions) {
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
        let state_count = 1;
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

            // Track that we're inside this gamma during branch linearization.
            self.gamma_stack.push(node_id);
            self.linearize_region(region_id);
            self.gamma_stack.pop();

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
        let state_count = 1;
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
        exit_ctx: Option<ChainInheritedState>,
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
            tracing::debug!(
                node = node_id.index(),
                "both branches passthrough, skipping"
            );
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

        tracing::debug!(
            node = node_id.index(),
            has_exit_ctx = exit_ctx.is_some(),
            cont_branch = cont_branch.index(),
            b0_pt,
            b1_pt,
            "passthrough_exit enter"
        );

        // Only apply at top level if there's a chain to exploit.
        // A single passthrough-exit gamma without chaining just creates a
        // local landing (same cost as a merge block).
        if exit_ctx.is_none() && self.find_tail_passthrough_gamma(cont_branch).is_none() {
            return false;
        }

        // Chain control-state analysis (Stage 14.1 diagnostics)
        if exit_ctx.is_none() && std::env::var("KAJIT_CHAIN_ANALYSIS").is_ok() {
            self.analyze_chain_control_state(node_id);
        }

        // Stage 14.3: if this gamma's predicate comes from a deferred upstream
        // gamma (control-state fusion), linearize the upstream inline with
        // two-landing routing, eliminating the predicate materialization.
        if exit_ctx.is_none()
            && let Some(fusion) = self.control_fusions.get(&node_id).cloned()
        {
            return self.try_linearize_fused_passthrough_exit(node_id, regions, &fusion, exit_ctx);
        }

        let node = &self.func.nodes[node_id];
        let predicate = self.resolve_vreg(node.inputs[0].source);
        let state_count = 1;
        let passthrough_count = node.inputs.len() - 1 - state_count;
        let data_output_count = node
            .outputs
            .iter()
            .filter(|o| o.kind == PortKind::Data)
            .count();

        // Create or reuse landing block
        let owns_landing = exit_ctx.is_none();
        let inherited = exit_ctx.unwrap_or_else(|| {
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
            let mapping: Vec<usize> = (0..data_output_count).collect();

            // Classify each output at top level (Stage 14.2)
            let (output_classes, control_done_value) =
                self.classify_chain_outputs(node_id, regions, data_output_count);

            if std::env::var("KAJIT_CHAIN_ANALYSIS").is_ok() {
                eprintln!(
                    "[chain-ctx] #{}: classes={:?}, ctrl_done={:?}",
                    node_id.index(),
                    output_classes,
                    control_done_value
                );
            }

            ChainInheritedState {
                landing_label: label,
                landing_vregs: vregs,
                state_env: state,
                output_to_landing: mapping,
                output_classes,
                control_done_value,
            }
        });
        let landing_label = inherited.landing_label;
        let landing_vregs = inherited.landing_vregs;
        let mut state_env = inherited.state_env;
        let output_to_landing = inherited.output_to_landing;
        let output_classes = inherited.output_classes;
        let control_done_value = inherited.control_done_value;

        // Exit phis: project current state_env onto landing params.
        // On the passthrough exit, the state_env already has the right values
        // (passthrough = no change = current env is correct).
        let exit_phis: Vec<(VReg, VReg)> = state_env
            .iter()
            .zip(landing_vregs.iter())
            .filter(|(s, d)| s != d)
            .map(|(s, d)| (*s, *d))
            .collect();

        tracing::debug!(
            node = node_id.index(),
            ?landing_vregs,
            ?state_env,
            ?exit_phis,
            predicate = predicate.index(),
            "passthrough_exit phis"
        );

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
            // Pre-compute state_env and safe_exit_env BEFORE linearizing non-tail
            // nodes. resolve_vreg uses RVSDG-assigned vregs (not linear IR defs),
            // so it works before linearization. This lets inner control-only gammas
            // exit directly to the chain landing with correct post-decode state.
            let cont_results = self.func.regions[cont_branch].results.clone();
            let cont_data_results: Vec<usize> = cont_results
                .iter()
                .enumerate()
                .filter(|&(_, &rid)| self.func.region_results[rid].kind == PortKind::Data)
                .map(|(i, _)| i)
                .collect();

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
                        let tail_out_idx = out_ref.index as usize;
                        while tail_output_to_landing.len() <= tail_out_idx {
                            tail_output_to_landing.push(usize::MAX);
                        }
                        tail_output_to_landing[tail_out_idx] = landing_idx;

                        let tail_node = &self.func.nodes[tail_id];
                        let tail_data_inputs: Vec<usize> = tail_node
                            .inputs
                            .iter()
                            .enumerate()
                            .skip(1)
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
                        state_env[landing_idx] = self.resolve_vreg(result.source);
                    }
                }
            }

            // Build safe_exit_env: for each state_env entry, resolve gamma outputs
            // that may not be defined yet (because the gamma is being linearized)
            // into available vregs or constants. Only resolve through gammas that
            // contain child gammas (those are the ones that may have inner exits
            // fire while the gamma's merge block hasn't been emitted yet).
            let non_tail_gammas: HashSet<NodeId> = self.func.regions[cont_branch]
                .nodes
                .iter()
                .filter(|&&nid| {
                    if nid == tail_id {
                        return false;
                    }
                    let NodeKind::Gamma { regions: gr } = &self.func.nodes[nid].kind else {
                        return false;
                    };
                    // Only include gammas that have child gammas in their branches
                    gr.iter().any(|&rid| {
                        self.func.regions[rid].nodes.iter().any(|&child| {
                            matches!(self.func.nodes[child].kind, NodeKind::Gamma { .. })
                        })
                    })
                })
                .copied()
                .collect();

            let safe_exit_env: Vec<SafeExitVal> = if non_tail_gammas.is_empty() {
                // No gammas with child gammas → all state_env vregs are safe
                state_env.iter().map(|&v| SafeExitVal::Vreg(v)).collect()
            } else {
                state_env
                    .iter()
                    .map(|&v| self.resolve_safe_exit_val_from_vreg(v, &non_tail_gammas))
                    .collect()
            };

            // Set chain exit context and linearize non-tail nodes
            let prev_ctx = self.chain_exit_ctx.take();
            self.chain_exit_ctx = Some(ChainExitCtx {
                landing_label,
                landing_vregs: landing_vregs.clone(),
                state_env: state_env.clone(),
                output_to_landing: output_to_landing.clone(),
                safe_exit_env,
                output_classes: output_classes.clone(),
                control_done_value,
            });
            let node_ids: Vec<NodeId> = self.func.regions[cont_branch].nodes.clone();
            for &nid in &node_ids {
                if nid != tail_id {
                    self.linearize_node(nid);
                }
            }
            self.chain_exit_ctx = prev_ctx;

            // Recurse with updated state and tail mapping
            tracing::debug!(
                node = node_id.index(),
                tail = tail_id.index(),
                ?tail_output_to_landing,
                ?state_env,
                "chaining to tail gamma"
            );
            let NodeKind::Gamma { regions: tr } = &self.func.nodes[tail_id].kind else {
                unreachable!()
            };
            let tr = tr.clone();
            let chained = self.try_linearize_passthrough_exit(
                tail_id,
                &tr,
                Some(ChainInheritedState {
                    landing_label,
                    landing_vregs: landing_vregs.clone(),
                    state_env: state_env.clone(),
                    output_to_landing: tail_output_to_landing,
                    output_classes: output_classes.clone(),
                    control_done_value,
                }),
            );
            if !chained {
                // Tail gamma rejected chaining (e.g. both branches passthrough).
                // Linearize it normally and emit a branch to the landing.
                tracing::debug!(
                    node = node_id.index(),
                    tail = tail_id.index(),
                    "tail gamma rejected chaining, falling back to normal linearize + branch"
                );
                self.linearize_node(tail_id);
                // Build final phis from the tail gamma's outputs to landing vregs
                let tail_node = &self.func.nodes[tail_id];
                let tail_data_outputs: Vec<VReg> = tail_node
                    .outputs
                    .iter()
                    .filter(|o| o.kind == PortKind::Data)
                    .map(|o| o.vreg.expect("gamma output vreg"))
                    .collect();
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
        } else {
            // No chain — linearize full continue region with chain exit context.
            // Pre-compute state_env from continue region results so inner
            // control-only exits carry post-computation state, not stale parent state.
            let cont_results = self.func.regions[cont_branch].results.clone();
            let cont_data_results: Vec<usize> = cont_results
                .iter()
                .enumerate()
                .filter(|&(_, &rid)| self.func.region_results[rid].kind == PortKind::Data)
                .map(|(i, _)| i)
                .collect();

            // Update state_env with all continue region data results
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
                    state_env[landing_idx] = self.resolve_vreg(result.source);
                }
            }

            // Build safe_exit_env with resolution through non-tail gammas
            let non_tail_gammas: HashSet<NodeId> = self.func.regions[cont_branch]
                .nodes
                .iter()
                .filter(|&&nid| {
                    let NodeKind::Gamma { regions: gr } = &self.func.nodes[nid].kind else {
                        return false;
                    };
                    gr.iter().any(|&rid| {
                        self.func.regions[rid].nodes.iter().any(|&child| {
                            matches!(self.func.nodes[child].kind, NodeKind::Gamma { .. })
                        })
                    })
                })
                .copied()
                .collect();

            let safe_env: Vec<SafeExitVal> = if non_tail_gammas.is_empty() {
                state_env.iter().map(|&v| SafeExitVal::Vreg(v)).collect()
            } else {
                state_env
                    .iter()
                    .map(|&v| self.resolve_safe_exit_val_from_vreg(v, &non_tail_gammas))
                    .collect()
            };

            let prev_ctx = self.chain_exit_ctx.take();
            self.chain_exit_ctx = Some(ChainExitCtx {
                landing_label,
                landing_vregs: landing_vregs.clone(),
                state_env: state_env.clone(),
                output_to_landing: output_to_landing.clone(),
                safe_exit_env: safe_env,
                output_classes: output_classes.clone(),
                control_done_value,
            });
            self.linearize_region(cont_branch);
            self.chain_exit_ctx = prev_ctx;
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
            tracing::debug!(
                node = node_id.index(),
                ?final_phis,
                ?state_env,
                "no-tail branch to landing"
            );
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

    /// Resolve a vreg to a SafeExitVal. If the vreg is an output of a gamma
    /// in `non_tail_gammas` (which may not be fully linearized when inner exits
    /// fire), trace through the gamma to find an available value.
    fn resolve_safe_exit_val_from_vreg(
        &self,
        vreg: VReg,
        non_tail_gammas: &HashSet<NodeId>,
    ) -> SafeExitVal {
        // Find which node produces this vreg by scanning non-tail gammas
        for &gid in non_tail_gammas {
            let gnode = &self.func.nodes[gid];
            for (out_idx, out) in gnode.outputs.iter().enumerate() {
                if out.kind == PortKind::Data && out.vreg == Some(vreg) {
                    return self.resolve_gamma_exit_val(gid, out_idx, non_tail_gammas);
                }
            }
        }
        // Not from a non-tail gamma → available as-is
        SafeExitVal::Vreg(vreg)
    }

    /// Resolve a gamma's data output to a safe exit value by tracing through
    /// the gamma's non-passthrough (active) branch.
    fn resolve_gamma_exit_val(
        &self,
        gamma_id: NodeId,
        output_idx: usize,
        non_tail_gammas: &HashSet<NodeId>,
    ) -> SafeExitVal {
        let node = &self.func.nodes[gamma_id];
        let NodeKind::Gamma { regions } = &node.kind else {
            return SafeExitVal::Vreg(node.outputs[output_idx].vreg.unwrap());
        };

        // Find the non-passthrough (active) branch
        let active_branch = if self.is_data_passthrough_region(regions[0]) {
            1
        } else if self.is_data_passthrough_region(regions[1]) {
            0
        } else {
            return SafeExitVal::Vreg(node.outputs[output_idx].vreg.unwrap());
        };

        let active_region = &self.func.regions[regions[active_branch]];
        let data_results: Vec<_> = active_region
            .results
            .iter()
            .enumerate()
            .filter(|&(_, &rid)| self.func.region_results[rid].kind == PortKind::Data)
            .collect();

        if output_idx >= data_results.len() {
            return SafeExitVal::Vreg(node.outputs[output_idx].vreg.unwrap());
        }

        let (_, &result_id) = data_results[output_idx];
        let result = &self.func.region_results[result_id];

        match result.source {
            PortSource::RegionArg(arg_ref) => {
                // Passthrough in the active branch → resolve to gamma input
                self.resolve_gamma_arg_to_input(node, active_region, arg_ref.arg, non_tail_gammas)
            }
            PortSource::Node(inner_ref) => {
                let inner_node = &self.func.nodes[inner_ref.node];
                match &inner_node.kind {
                    NodeKind::Simple(IrOp::Const { value }) => SafeExitVal::Const(*value),
                    NodeKind::Gamma {
                        regions: inner_regions,
                    } => {
                        // Inner gamma (e.g. the bit-test gamma): resolve through its
                        // non-passthrough (done) branch to find const values
                        let inner_done = if self.is_data_passthrough_region(inner_regions[0]) {
                            1
                        } else if self.is_data_passthrough_region(inner_regions[1]) {
                            0
                        } else {
                            return SafeExitVal::Vreg(
                                inner_node.outputs[inner_ref.index as usize].vreg.unwrap(),
                            );
                        };

                        let done_region = &self.func.regions[inner_regions[inner_done]];
                        let done_data_results: Vec<_> = done_region
                            .results
                            .iter()
                            .enumerate()
                            .filter(|&(_, &rid)| {
                                self.func.region_results[rid].kind == PortKind::Data
                            })
                            .collect();

                        let idx = inner_ref.index as usize;
                        if idx >= done_data_results.len() {
                            return SafeExitVal::Vreg(inner_node.outputs[idx].vreg.unwrap());
                        }

                        let (_, &done_rid) = done_data_results[idx];
                        match self.func.region_results[done_rid].source {
                            PortSource::Node(const_ref) => {
                                if let NodeKind::Simple(IrOp::Const { value }) =
                                    &self.func.nodes[const_ref.node].kind
                                {
                                    SafeExitVal::Const(*value)
                                } else {
                                    SafeExitVal::Vreg(inner_node.outputs[idx].vreg.unwrap())
                                }
                            }
                            PortSource::RegionArg(arg_ref) => {
                                // Passthrough in done branch → resolve to inner gamma input
                                self.resolve_gamma_arg_to_input(
                                    inner_node,
                                    done_region,
                                    arg_ref.arg,
                                    non_tail_gammas,
                                )
                            }
                        }
                    }
                    _ => SafeExitVal::Vreg(
                        inner_node.outputs[inner_ref.index as usize].vreg.unwrap(),
                    ),
                }
            }
        }
    }

    /// Map a region arg back to the gamma's corresponding input vreg.
    fn resolve_gamma_arg_to_input(
        &self,
        gamma_node: &Node,
        branch_region: &kajit_ir::Region,
        arg_id: kajit_ir::ArgId,
        non_tail_gammas: &HashSet<NodeId>,
    ) -> SafeExitVal {
        let data_args: Vec<_> = branch_region
            .args
            .iter()
            .enumerate()
            .filter(|&(_, &aid)| self.func.region_args[aid].kind == PortKind::Data)
            .collect();
        for (data_idx, &(_, aid)) in data_args.iter().enumerate() {
            if *aid == arg_id {
                let data_inputs: Vec<_> = gamma_node
                    .inputs
                    .iter()
                    .enumerate()
                    .skip(1)
                    .filter(|&(_, inp)| inp.kind == PortKind::Data)
                    .collect();
                if data_idx < data_inputs.len() {
                    let (input_idx, _) = data_inputs[data_idx];
                    let v = self.resolve_vreg(gamma_node.inputs[input_idx].source);
                    // Check if this vreg is itself from a non-tail gamma
                    return self.resolve_safe_exit_val_from_vreg(v, non_tail_gammas);
                }
            }
        }
        SafeExitVal::Vreg(VReg::new(0)) // shouldn't reach here
    }

    /// Lower a control-only gamma as a chain exit. When a gamma inside a
    /// passthrough-exit chain has ALL data outputs that are control-only
    /// (is_more flags), its "done" branch (which produces constants) can exit
    /// directly to the chain landing, and its "more" branch (passthrough)
    /// falls through with outputs = inputs.
    fn try_linearize_control_only_chain_exit(
        &mut self,
        node_id: NodeId,
        regions: &[RegionId],
    ) -> bool {
        // Control-only exits inside theta bodies are suppressed: the safe_exit_env
        // resolution doesn't account for loop-scoped vregs correctly.
        if self.theta_depth > 0 {
            return false;
        }
        let ctx = match &self.chain_exit_ctx {
            Some(ctx) => ctx.clone(),
            None => return false,
        };

        // Stage 14.2 invariant: classification matches landing size
        debug_assert!(
            ctx.output_classes.is_empty() || ctx.output_classes.len() == ctx.landing_vregs.len(),
            "output_classes length mismatch: {} vs {} landing_vregs",
            ctx.output_classes.len(),
            ctx.landing_vregs.len()
        );

        if regions.len() != 2 {
            return false;
        }

        let node = &self.func.nodes[node_id];
        let data_output_count = node
            .outputs
            .iter()
            .filter(|o| o.kind == PortKind::Data)
            .count();

        if data_output_count == 0 {
            return false;
        }

        // All data outputs must be control-only
        for i in 0..data_output_count {
            if !self.func.is_chain_control_only_output(node_id, i, 0) {
                return false;
            }
        }

        // One branch must be passthrough (the "more" branch), the other not
        let b0_pt = self.is_data_passthrough_region(regions[0]);
        let b1_pt = self.is_data_passthrough_region(regions[1]);
        if b0_pt == b1_pt {
            return false;
        }

        // "more" = passthrough branch (continues), "done" = const branch (exits)
        let (more_branch, done_branch, done_on_nonzero) = if b0_pt {
            // Branch 0 = passthrough (more), done when pred != 0
            (regions[0], regions[1], true)
        } else {
            // Branch 1 = passthrough (more), done when pred == 0
            (regions[1], regions[0], false)
        };

        // Don't apply to error-only regions
        if self.region_is_error_only(more_branch) || self.region_is_error_only(done_branch) {
            return false;
        }

        // The "done" branch must produce ONLY constants for data outputs (no
        // inner gammas, no computation). This prevents matching wrapper gammas
        // (like bounds-check gates) whose non-passthrough branch contains the
        // actual bit-test computation.
        let done_region = &self.func.regions[done_branch];
        let done_has_structural = done_region.nodes.iter().any(|&nid| {
            matches!(
                &self.func.nodes[nid].kind,
                NodeKind::Gamma { .. } | NodeKind::Theta { .. }
            )
        });
        if done_has_structural {
            return false;
        }
        // Verify all data results are constants
        for &rid in &done_region.results {
            let result = &self.func.region_results[rid];
            if result.kind != PortKind::Data {
                continue;
            }
            match result.source {
                PortSource::Node(out_ref) => {
                    if !matches!(
                        &self.func.nodes[out_ref.node].kind,
                        NodeKind::Simple(IrOp::Const { .. })
                    ) {
                        return false;
                    }
                }
                _ => return false,
            }
        }

        let predicate = self.resolve_vreg(node.inputs[0].source);

        // Build exit phis from safe_exit_env → landing_vregs
        let mut exit_phis: Vec<(VReg, VReg)> = Vec::new();
        for (i, &lv) in ctx.landing_vregs.iter().enumerate() {
            if i >= ctx.safe_exit_env.len() {
                break;
            }
            let src = match ctx.safe_exit_env[i] {
                SafeExitVal::Vreg(v) => {
                    // Verify this vreg is NOT from a gamma we're currently
                    // inside (whose merge block hasn't been emitted yet),
                    // NOR from this gamma node itself.
                    for out in &node.outputs {
                        if out.vreg == Some(v) {
                            return false;
                        }
                    }
                    for &gid in &self.gamma_stack {
                        let gnode = &self.func.nodes[gid];
                        for out in &gnode.outputs {
                            if out.vreg == Some(v) {
                                return false;
                            }
                        }
                    }
                    v
                }
                SafeExitVal::Const(val) => {
                    let v = self.fresh_vreg();
                    self.record_vreg_scope(v, node.debug_scope);
                    self.emit(
                        Some(node.debug_scope),
                        LinearOp::Const { dst: v, value: val },
                    );
                    v
                }
            };
            if src != lv {
                exit_phis.push((src, lv));
            }
        }

        // Entry phis for the "more" (passthrough) branch
        let state_count = 1;
        let passthrough_count = node.inputs.len() - 1 - state_count;
        let more_region = &self.func.regions[more_branch];
        let mut more_entry_phis = Vec::new();
        for i in 0..passthrough_count {
            let src_input = &node.inputs[i + 1];
            if src_input.kind == PortKind::Data
                && let Some(&arg_id) = more_region.args.get(i)
            {
                let arg = &self.func.region_args[arg_id];
                if let Some(dst_vreg) = arg.vreg {
                    let src_vreg = self.resolve_vreg(src_input.source);
                    self.record_vreg_scope(dst_vreg, more_region.debug_scope);
                    more_entry_phis.push((src_vreg, dst_vreg));
                }
            }
        }

        // Emit: done → landing, more → fallthrough
        if done_on_nonzero {
            self.emit(
                Some(node.debug_scope),
                LinearOp::BranchIf {
                    cond: predicate,
                    target: ctx.landing_label,
                    phi_args: exit_phis,
                    fallthrough_phi_args: more_entry_phis,
                },
            );
        } else {
            self.emit(
                Some(node.debug_scope),
                LinearOp::BranchIfZero {
                    cond: predicate,
                    target: ctx.landing_label,
                    phi_args: exit_phis,
                    fallthrough_phi_args: more_entry_phis,
                },
            );
        }

        // For the "more" (passthrough) branch: gamma outputs = gamma inputs
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

    // ─── Chain control-state analysis (Stage 14.1) ────────────────────

    /// Analyze and print control-state information for a passthrough-exit chain.
    /// Gated by KAJIT_CHAIN_ANALYSIS env var. Prints:
    /// - real state outputs vs transitive control-only outputs
    /// - grouping of control-only outputs into logical booleans
    /// - downstream branch sites that consume each logical boolean
    fn analyze_chain_control_state(&self, chain_gamma: NodeId) {
        let node = &self.func.nodes[chain_gamma];
        let NodeKind::Gamma { regions } = &node.kind else {
            return;
        };

        let data_output_count = node
            .outputs
            .iter()
            .filter(|o| o.kind == PortKind::Data)
            .count();

        // 1. Classify each data output as real state or control-only,
        //    and find downstream consumers of each.
        let mut real_state_indices = Vec::new();
        let mut control_only_indices = Vec::new();
        let mut live_ctrl_indices = Vec::new();
        let mut live_real_indices = Vec::new();
        for i in 0..data_output_count {
            let consumers = self.func.find_output_consumers(chain_gamma, i as u16);
            let is_live = !consumers.is_empty();
            let is_ctrl = self.func.is_chain_control_only_output(chain_gamma, i, 0);
            if is_ctrl {
                control_only_indices.push(i);
                if is_live {
                    live_ctrl_indices.push((i, consumers));
                }
            } else {
                real_state_indices.push(i);
                if is_live {
                    live_real_indices.push((i, consumers));
                }
            }
        }

        eprintln!(
            "[chain-analysis] chain gamma #{} ({} data outputs)",
            chain_gamma.index(),
            data_output_count
        );
        eprintln!(
            "  real state: {:?} ({} live)",
            real_state_indices,
            live_real_indices.len()
        );
        eprintln!(
            "  control-only: {:?} ({} live)",
            control_only_indices,
            live_ctrl_indices.len()
        );

        // 2. Show live consumers
        for (idx, consumers) in &live_real_indices {
            for c in consumers {
                eprintln!("  output {} (real) → {}", idx, self.format_consumer(c));
            }
        }
        for (idx, consumers) in &live_ctrl_indices {
            for c in consumers {
                eprintln!("  output {} (ctrl) → {}", idx, self.format_consumer(c));
            }
        }

        // 3. Analyze predicate source — this is the KEY boolean
        let pred_source = &node.inputs[0].source;
        let pred_is_ctrl = match pred_source {
            PortSource::Node(out_ref) => {
                self.func
                    .is_chain_control_only_output(out_ref.node, out_ref.index as usize, 0)
            }
            _ => false,
        };
        self.analyze_predicate_source(chain_gamma, pred_source);

        // 4. Classify control-only outputs: invariant vs boolean vs unknown.
        let (invariant_indices, boolean_groups) =
            self.group_control_only_by_boolean(chain_gamma, regions, &control_only_indices);

        if !invariant_indices.is_empty() {
            eprintln!("  invariant (passthrough-all): {:?}", invariant_indices);
        }

        // Separate live vs dead boolean groups
        let mut live_boolean_groups = Vec::new();
        let mut dead_boolean_groups = Vec::new();
        for g in &boolean_groups {
            let any_live = g
                .output_indices
                .iter()
                .any(|idx| live_ctrl_indices.iter().any(|(li, _)| li == idx));
            if any_live {
                live_boolean_groups.push(g);
            } else {
                dead_boolean_groups.push(g);
            }
        }

        for g in &live_boolean_groups {
            eprintln!(
                "  LIVE boolean: outputs {:?} (done_val={})",
                g.output_indices, g.done_value
            );
        }
        if !dead_boolean_groups.is_empty() {
            let dead_out_count: usize = dead_boolean_groups
                .iter()
                .map(|g| g.output_indices.len())
                .sum();
            eprintln!(
                "  dead boolean outputs: {} across {} groups",
                dead_out_count,
                dead_boolean_groups.len()
            );
        }

        // 5. Walk the chain (follow tail passthrough gammas)
        self.walk_chain_analysis(chain_gamma, 1);

        // 6. Summary: count distinct logical booleans
        // The predicate boolean counts as one if it's control-only.
        // Live boolean groups add additional distinct booleans.
        // A live boolean group with done_val=0 that matches the predicate's
        // done_val is the SAME boolean (just carried through as data).
        let pred_done_val = if pred_is_ctrl {
            match pred_source {
                PortSource::Node(out_ref) => {
                    if let NodeKind::Gamma {
                        regions: pred_regions,
                    } = &self.func.nodes[out_ref.node].kind
                    {
                        self.classify_ctrl_output(
                            out_ref.node,
                            pred_regions,
                            out_ref.index as usize,
                        )
                    } else {
                        CtrlOutputKind::Unknown
                    }
                }
                _ => CtrlOutputKind::Unknown,
            }
        } else {
            CtrlOutputKind::Unknown
        };

        // Count distinct booleans:
        // - Predicate counts as 1 if it's a real Boolean (not Invariant)
        // - Each live group with a DIFFERENT done_val adds 1
        // - A live group matching the predicate's done_val is the same boolean
        let pred_is_real_boolean = matches!(pred_done_val, CtrlOutputKind::Boolean { .. });
        let mut distinct_count = if pred_is_real_boolean { 1usize } else { 0 };
        for g in &live_boolean_groups {
            let same_as_pred = match pred_done_val {
                CtrlOutputKind::Boolean { done_value } => done_value == g.done_value,
                _ => false,
            };
            if !same_as_pred {
                distinct_count += 1;
            }
        }

        if pred_is_ctrl && !pred_is_real_boolean {
            eprintln!(
                "  note: predicate is control-only but {:?} (not a varying boolean)",
                pred_done_val
            );
        }

        eprintln!(
            "[chain-analysis] summary: {} distinct boolean(s), {} invariant, {} real state",
            distinct_count,
            invariant_indices.len(),
            live_real_indices.len()
        );

        if distinct_count == 1 {
            eprintln!("  → SUPPORTED: exactly one logical boolean control state");
        } else if distinct_count == 0 {
            eprintln!("  → no boolean control state (nothing to optimize)");
        } else {
            eprintln!(
                "  → UNSUPPORTED: {} distinct boolean control states",
                distinct_count
            );
        }
    }

    fn format_consumer(&self, c: &kajit_ir::OutputConsumer) -> String {
        let use_str = match c.use_kind {
            OutputUseKind::GammaPredicate => "gamma_predicate",
            OutputUseKind::DataInput => "data_input",
            OutputUseKind::RegionExit => "region_exit",
        };
        match &c.kind {
            kajit_ir::ConsumerKind::NodeInput {
                node: cnode,
                input_index,
            } => {
                let ckind = match &self.func.nodes[*cnode].kind {
                    NodeKind::Simple(op) => format!("{:?}", op),
                    NodeKind::Gamma { .. } => "Gamma".to_string(),
                    NodeKind::Theta { .. } => "Theta".to_string(),
                    _ => "other".to_string(),
                };
                format!(
                    "#{}.input[{}] ({}) as {}",
                    cnode.index(),
                    input_index,
                    ckind,
                    use_str
                )
            }
            kajit_ir::ConsumerKind::RegionResult { .. } => {
                format!("region_result as {}", use_str)
            }
        }
    }

    /// Walk the chain recursively, analyzing each tail gamma.
    fn walk_chain_analysis(&self, gamma_id: NodeId, depth: usize) {
        let node = &self.func.nodes[gamma_id];
        let NodeKind::Gamma { regions } = &node.kind else {
            return;
        };

        // Find the continue (non-passthrough) branch
        let cont_branch = if self.is_data_passthrough_region(regions[0]) {
            regions[1]
        } else if self.is_data_passthrough_region(regions[1]) {
            regions[0]
        } else {
            return;
        };

        // Find the tail passthrough gamma in the continue branch
        let Some(tail_id) = self.find_tail_passthrough_gamma(cont_branch) else {
            eprintln!(
                "  [depth {}] no tail gamma in continue branch (chain ends)",
                depth
            );
            return;
        };

        let tail_node = &self.func.nodes[tail_id];
        let tail_data_count = tail_node
            .outputs
            .iter()
            .filter(|o| o.kind == PortKind::Data)
            .count();

        let mut tail_real = Vec::new();
        let mut tail_ctrl = Vec::new();
        for i in 0..tail_data_count {
            if self.func.is_chain_control_only_output(tail_id, i, 0) {
                tail_ctrl.push(i);
            } else {
                tail_real.push(i);
            }
        }

        eprintln!(
            "  [depth {}] tail gamma #{} ({} data outputs, {} real, {} ctrl-only)",
            depth,
            tail_id.index(),
            tail_data_count,
            tail_real.len(),
            tail_ctrl.len()
        );

        // Analyze the tail's predicate
        self.analyze_predicate_source(tail_id, &tail_node.inputs[0].source);

        // Recurse
        let NodeKind::Gamma { regions: tr } = &self.func.nodes[tail_id].kind else {
            return;
        };
        let tr = tr.clone();
        let cont = if self.is_data_passthrough_region(tr[0]) {
            tr[1]
        } else {
            tr[0]
        };
        if let Some(next_tail) = self.find_tail_passthrough_gamma(cont) {
            let _ = next_tail;
            self.walk_chain_analysis(tail_id, depth + 1);
        } else {
            eprintln!("  [depth {}] chain ends (no further tail gamma)", depth + 1);
        }
    }

    /// Analyze where a gamma's predicate comes from.
    fn analyze_predicate_source(&self, gamma_id: NodeId, source: &PortSource) {
        match source {
            PortSource::Node(out_ref) => {
                let src_node = &self.func.nodes[out_ref.node];
                let is_ctrl_only =
                    self.func
                        .is_chain_control_only_output(out_ref.node, out_ref.index as usize, 0);
                let kind_str = match &src_node.kind {
                    NodeKind::Simple(op) => format!("{:?}", op),
                    NodeKind::Gamma { .. } => "Gamma".to_string(),
                    NodeKind::Theta { .. } => "Theta".to_string(),
                    _ => "other".to_string(),
                };
                eprintln!(
                    "  predicate of #{}: #{}.output[{}] ({}){}",
                    gamma_id.index(),
                    out_ref.node.index(),
                    out_ref.index,
                    kind_str,
                    if is_ctrl_only { " [CONTROL-ONLY]" } else { "" }
                );
            }
            PortSource::RegionArg(arg_ref) => {
                eprintln!(
                    "  predicate of #{}: region_arg {:?}",
                    gamma_id.index(),
                    arg_ref
                );
            }
        }
    }

    /// Classify a control-only output: invariant (passthrough everywhere),
    /// boolean (one branch passthrough, other const), or unknown.
    fn classify_ctrl_output(
        &self,
        _gamma_id: NodeId,
        regions: &[RegionId],
        output_idx: usize,
    ) -> CtrlOutputKind {
        if regions.len() != 2 {
            return CtrlOutputKind::Unknown;
        }

        // Check if BOTH branches are passthrough for this output.
        // A branch is "effectively passthrough" if:
        // - the result is directly RegionArg at the matching position, OR
        // - the result comes from an inner gamma whose output is invariant
        //   (transitively passthrough on all branches)
        let mut all_passthrough = true;
        let mut done_value = None;

        for &region_id in regions.iter() {
            let region = &self.func.regions[region_id];
            let data_results: Vec<_> = region
                .results
                .iter()
                .enumerate()
                .filter(|&(_, &rid)| self.func.region_results[rid].kind == PortKind::Data)
                .collect();
            if output_idx >= data_results.len() {
                return CtrlOutputKind::Unknown;
            }
            let (_, &result_id) = data_results[output_idx];
            let result = &self.func.region_results[result_id];
            match result.source {
                PortSource::RegionArg(arg_ref) => {
                    if region.args.get(output_idx) != Some(&arg_ref.arg) {
                        all_passthrough = false;
                    }
                }
                PortSource::Node(out_ref) => {
                    // Check if this is from an inner gamma that's invariant for this output
                    if self.is_invariant_gamma_output(out_ref.node, out_ref.index as usize) {
                        // Effectively passthrough — the inner gamma always passes through
                    } else {
                        all_passthrough = false;
                        // Try to extract constant value
                        let val =
                            self.extract_const_from_source(out_ref.node, out_ref.index as usize, 0);
                        if let Some(v) = val {
                            done_value = Some(v);
                        }
                    }
                }
            }
        }

        if all_passthrough {
            CtrlOutputKind::Invariant
        } else if let Some(val) = done_value {
            CtrlOutputKind::Boolean { done_value: val }
        } else {
            CtrlOutputKind::Unknown
        }
    }

    /// Check if a gamma's output is invariant: passthrough on ALL branches,
    /// possibly through inner gammas that are themselves invariant.
    fn is_invariant_gamma_output(&self, node_id: NodeId, output_idx: usize) -> bool {
        self.is_invariant_gamma_output_inner(node_id, output_idx, 0)
    }

    fn is_invariant_gamma_output_inner(
        &self,
        node_id: NodeId,
        output_idx: usize,
        depth: usize,
    ) -> bool {
        if depth > 8 {
            return false;
        }
        let NodeKind::Gamma { regions } = &self.func.nodes[node_id].kind else {
            return false;
        };
        for &region_id in regions.iter() {
            let region = &self.func.regions[region_id];
            let data_results: Vec<_> = region
                .results
                .iter()
                .enumerate()
                .filter(|&(_, &rid)| self.func.region_results[rid].kind == PortKind::Data)
                .collect();
            if output_idx >= data_results.len() {
                return false;
            }
            let (_, &result_id) = data_results[output_idx];
            let result = &self.func.region_results[result_id];
            match result.source {
                PortSource::RegionArg(arg_ref) => {
                    if region.args.get(output_idx) != Some(&arg_ref.arg) {
                        return false;
                    }
                }
                PortSource::Node(out_ref) => {
                    // Recursively check inner gamma
                    if !self.is_invariant_gamma_output_inner(
                        out_ref.node,
                        out_ref.index as usize,
                        depth + 1,
                    ) {
                        return false;
                    }
                }
            }
        }
        true
    }

    /// Extract a constant value from a node output, recursing through inner gammas.
    fn extract_const_from_source(
        &self,
        node_id: NodeId,
        output_idx: usize,
        depth: usize,
    ) -> Option<u64> {
        if depth > 16 {
            return None;
        }
        let node = &self.func.nodes[node_id];
        match &node.kind {
            NodeKind::Simple(IrOp::Const { value }) => Some(*value),
            NodeKind::Gamma { regions } if regions.len() == 2 => {
                // Recurse: check the non-passthrough branch of the inner gamma
                let inner_active = if self.is_data_passthrough_region(regions[0]) {
                    1
                } else if self.is_data_passthrough_region(regions[1]) {
                    0
                } else {
                    return None;
                };
                let inner_region = &self.func.regions[regions[inner_active]];
                let data_results: Vec<_> = inner_region
                    .results
                    .iter()
                    .enumerate()
                    .filter(|&(_, &rid)| self.func.region_results[rid].kind == PortKind::Data)
                    .collect();
                if output_idx >= data_results.len() {
                    return None;
                }
                let (_, &result_id) = data_results[output_idx];
                let result = &self.func.region_results[result_id];
                match result.source {
                    PortSource::Node(out_ref) => self.extract_const_from_source(
                        out_ref.node,
                        out_ref.index as usize,
                        depth + 1,
                    ),
                    PortSource::RegionArg(_) => None,
                }
            }
            _ => None,
        }
    }

    /// Group control-only outputs by their logical boolean identity.
    /// Two outputs are in the same boolean group if they have the same done_value.
    /// Invariant outputs (passthrough on all branches) are reported separately.
    fn group_control_only_by_boolean(
        &self,
        gamma_id: NodeId,
        regions: &[RegionId],
        ctrl_indices: &[usize],
    ) -> (Vec<usize>, Vec<BooleanGroupInfo>) {
        let mut invariant_indices = Vec::new();
        let mut groups: Vec<BooleanGroupInfo> = Vec::new();

        for &idx in ctrl_indices {
            match self.classify_ctrl_output(gamma_id, regions, idx) {
                CtrlOutputKind::Invariant => {
                    invariant_indices.push(idx);
                }
                CtrlOutputKind::Boolean { done_value } => {
                    let mut found = false;
                    for g in &mut groups {
                        if g.done_value == done_value {
                            g.output_indices.push(idx);
                            found = true;
                            break;
                        }
                    }
                    if !found {
                        groups.push(BooleanGroupInfo {
                            output_indices: vec![idx],
                            done_value,
                        });
                    }
                }
                CtrlOutputKind::Unknown => {
                    // Unknown — treat as its own group
                    groups.push(BooleanGroupInfo {
                        output_indices: vec![idx],
                        done_value: u64::MAX,
                    });
                }
            }
        }

        (invariant_indices, groups)
    }

    // ─── Fused passthrough-exit (Stage 14.3) ────────────────────────

    /// Linearize a passthrough-exit gamma whose predicate comes from a deferred
    /// upstream gamma. Instead of materializing the control boolean and branching,
    /// this linearizes the upstream gamma inline with split exits:
    /// - exits where the control boolean = done_value → downstream's landing
    /// - exits where the control boolean ≠ done_value → downstream's continue code
    fn try_linearize_fused_passthrough_exit(
        &mut self,
        downstream_id: NodeId,
        downstream_regions: &[RegionId],
        fusion: &ControlFusionInfo,
        exit_ctx: Option<ChainInheritedState>,
    ) -> bool {
        let ds_node = &self.func.nodes[downstream_id];
        let state_count = 1;
        let ds_passthrough_count = ds_node.inputs.len() - 1 - state_count;
        let ds_data_output_count = ds_node
            .outputs
            .iter()
            .filter(|o| o.kind == PortKind::Data)
            .count();

        // Identify the downstream's passthrough vs continue branch
        let ds_b0_pt = self.is_data_passthrough_region(downstream_regions[0]);
        let (ds_cont_branch, _ds_exit_on_zero) = if ds_b0_pt {
            (downstream_regions[1], true)
        } else {
            (downstream_regions[0], false)
        };

        // Create or reuse landing (where "done" exits go)
        let owns_landing = exit_ctx.is_none();
        let (ds_landing_label, ds_landing_vregs, ds_state_env, ds_output_to_landing) =
            if let Some(inherited) = &exit_ctx {
                (
                    inherited.landing_label,
                    inherited.landing_vregs.clone(),
                    Some(inherited.state_env.clone()),
                    inherited.output_to_landing.clone(),
                )
            } else {
                let label = self.fresh_label();
                let vregs: Vec<VReg> = (0..ds_data_output_count)
                    .map(|i| {
                        let v = ds_node.outputs[i].vreg.expect("gamma output vreg");
                        self.record_vreg_scope(v, ds_node.outputs[i].debug_scope);
                        v
                    })
                    .collect();
                let mapping: Vec<usize> = (0..ds_data_output_count).collect();
                (label, vregs, None, mapping)
            };

        // Create the "continue" label (where "more" exits go)
        let continue_label = self.fresh_label();

        // Get the upstream gamma's info
        let upstream_id = fusion.upstream;
        let up_node = &self.func.nodes[upstream_id];
        let NodeKind::Gamma {
            regions: up_regions,
        } = &up_node.kind
        else {
            return false;
        };
        let up_regions = up_regions.clone();
        let up_predicate = self.resolve_vreg(up_node.inputs[0].source);
        let up_passthrough_count = up_node.inputs.len() - 1 - state_count;
        let up_data_output_count = up_node
            .outputs
            .iter()
            .filter(|o| o.kind == PortKind::Data)
            .count();

        // Helper: given upstream output vregs for a specific path, compute
        // phis for the downstream landing (done path = passthrough of ds inputs).
        // When ds_state_env is set (inner chain), use the chain's state_env for
        // landing positions not covered by the downstream gamma's outputs.
        let compute_done_phis =
            |lin: &mut Linearizer, up_out_vregs: &[VReg]| -> Vec<(VReg, VReg)> {
                let mut phis = Vec::new();
                if let Some(ref env) = ds_state_env {
                    // Inner chain: project onto chain landing via output_to_landing
                    let mut used = std::collections::HashSet::new();
                    for i in 0..ds_data_output_count {
                        if i >= ds_output_to_landing.len() {
                            break;
                        }
                        let landing_idx = ds_output_to_landing[i];
                        if landing_idx == usize::MAX {
                            continue;
                        }
                        let src = &ds_node.inputs[i + 1];
                        if src.kind != PortKind::Data {
                            continue;
                        }
                        let src_vreg = match src.source {
                            PortSource::Node(out_ref) if out_ref.node == upstream_id => {
                                let out_idx = out_ref.index as usize;
                                if out_idx < up_out_vregs.len() {
                                    up_out_vregs[out_idx]
                                } else {
                                    lin.resolve_vreg(src.source)
                                }
                            }
                            _ => lin.resolve_vreg(src.source),
                        };
                        let dst = ds_landing_vregs[landing_idx];
                        used.insert(landing_idx);
                        if src_vreg != dst {
                            phis.push((src_vreg, dst));
                        }
                    }
                    // Carry any landing positions not covered by this gamma's outputs
                    for (idx, &lv) in ds_landing_vregs.iter().enumerate() {
                        if !used.contains(&idx) {
                            let src = env[idx];
                            if src != lv {
                                phis.push((src, lv));
                            }
                        }
                    }
                } else {
                    // Top level: direct 1:1 mapping
                    #[allow(clippy::needless_range_loop)]
                    for i in 0..ds_data_output_count {
                        let src = &ds_node.inputs[i + 1];
                        if src.kind != PortKind::Data {
                            continue;
                        }
                        let src_vreg = match src.source {
                            PortSource::Node(out_ref) if out_ref.node == upstream_id => {
                                let out_idx = out_ref.index as usize;
                                if out_idx < up_out_vregs.len() {
                                    up_out_vregs[out_idx]
                                } else {
                                    lin.resolve_vreg(src.source)
                                }
                            }
                            _ => lin.resolve_vreg(src.source),
                        };
                        let dst = ds_landing_vregs[i];
                        if src_vreg != dst {
                            phis.push((src_vreg, dst));
                        }
                    }
                }
                phis
            };

        // Helper: given upstream output vregs, compute entry phis for downstream
        // continue branch (n277 branch 1 region args).
        let compute_more_phis =
            |lin: &mut Linearizer, up_out_vregs: &[VReg]| -> Vec<(VReg, VReg)> {
                let ds_cont_region = &lin.func.regions[ds_cont_branch];
                let mut phis = Vec::new();
                for i in 0..ds_passthrough_count {
                    let src = &ds_node.inputs[i + 1];
                    if src.kind != PortKind::Data {
                        continue;
                    }
                    let src_vreg = match src.source {
                        PortSource::Node(out_ref) if out_ref.node == upstream_id => {
                            let out_idx = out_ref.index as usize;
                            if out_idx < up_out_vregs.len() {
                                up_out_vregs[out_idx]
                            } else {
                                lin.resolve_vreg(src.source)
                            }
                        }
                        _ => lin.resolve_vreg(src.source),
                    };
                    let arg = &lin.func.region_args[ds_cont_region.args[i]];
                    if let Some(dst_vreg) = arg.vreg {
                        lin.record_vreg_scope(dst_vreg, ds_cont_region.debug_scope);
                        phis.push((src_vreg, dst_vreg));
                    }
                }
                phis
            };

        // Phase 1: Linearize the upstream gamma as a standard 2-branch gamma,
        // but route each branch's exit to done_landing or continue_label based
        // on the control boolean value.
        let branch_count = up_regions.len();
        debug_assert_eq!(branch_count, 2);

        let branch_labels: Vec<LabelId> = (0..branch_count).map(|_| self.fresh_label()).collect();

        // Entry phis for each branch
        let mut branch_entry_phis: Vec<Vec<(VReg, VReg)>> = Vec::new();
        for &region_id in up_regions.iter() {
            let region = &self.func.regions[region_id];
            let mut phis = Vec::new();
            for i in 0..up_passthrough_count {
                let src_input = &up_node.inputs[i + 1];
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

        // Emit 2-branch conditional
        self.emit(
            Some(up_node.debug_scope),
            LinearOp::BranchIfZero {
                cond: up_predicate,
                target: branch_labels[0],
                phi_args: branch_entry_phis[0].clone(),
                fallthrough_phi_args: vec![],
            },
        );
        self.emit(
            Some(up_node.debug_scope),
            LinearOp::Branch {
                target: branch_labels[1],
                phi_args: branch_entry_phis[1].clone(),
            },
        );

        // Emit each branch with routing to done_landing or continue_label.
        // For branches where the control output is a direct constant or passthrough,
        // route the entire branch exit. For branches where the control output comes
        // from an inner gamma, linearize all non-inner-gamma nodes then split the
        // inner gamma's branches.
        let ctrl_idx = fusion.ctrl_output_idx;
        for (branch_idx, &region_id) in up_regions.iter().enumerate() {
            self.emit(
                Some(self.func.regions[region_id].debug_scope),
                LinearOp::Label(branch_labels[branch_idx]),
            );

            let region = &self.func.regions[region_id];
            let ctrl_result = &self.func.region_results[region.results[ctrl_idx]];

            // Check if the control output is from an inner gamma (needs recursive split)
            let inner_gamma = match ctrl_result.source {
                PortSource::Node(out_ref) => {
                    if matches!(&self.func.nodes[out_ref.node].kind, NodeKind::Gamma { .. }) {
                        Some(out_ref.node)
                    } else {
                        None
                    }
                }
                _ => None,
            };

            if let Some(inner_gamma_id) = inner_gamma {
                // Branch contains an inner gamma that produces the control output.
                // Linearize all other nodes, then handle the inner gamma with split exits.
                let node_ids: Vec<NodeId> = region.nodes.clone();
                for &nid in &node_ids {
                    if nid != inner_gamma_id {
                        self.linearize_node(nid);
                    }
                }

                // Now handle the inner gamma. Its branches determine done vs more.
                let ig_node = &self.func.nodes[inner_gamma_id];
                let NodeKind::Gamma {
                    regions: ig_regions,
                } = &ig_node.kind
                else {
                    return false;
                };
                let ig_regions = ig_regions.clone();
                let ig_pred = self.resolve_vreg(ig_node.inputs[0].source);
                let ig_pt_count = ig_node.inputs.len() - 1 - state_count;

                let ig_labels: Vec<LabelId> =
                    (0..ig_regions.len()).map(|_| self.fresh_label()).collect();

                // Inner gamma entry phis
                let mut ig_entry_phis: Vec<Vec<(VReg, VReg)>> = Vec::new();
                for &ig_rid in ig_regions.iter() {
                    let ig_reg = &self.func.regions[ig_rid];
                    let mut phis = Vec::new();
                    for i in 0..ig_pt_count {
                        let src_input = &ig_node.inputs[i + 1];
                        if src_input.kind == PortKind::Data {
                            let src_vreg = self.resolve_vreg(src_input.source);
                            let arg = &self.func.region_args[ig_reg.args[i]];
                            if let Some(dst_vreg) = arg.vreg {
                                self.record_vreg_scope(dst_vreg, ig_reg.debug_scope);
                                phis.push((src_vreg, dst_vreg));
                            }
                        }
                    }
                    ig_entry_phis.push(phis);
                }

                self.emit(
                    Some(ig_node.debug_scope),
                    LinearOp::BranchIfZero {
                        cond: ig_pred,
                        target: ig_labels[0],
                        phi_args: ig_entry_phis[0].clone(),
                        fallthrough_phi_args: vec![],
                    },
                );
                self.emit(
                    Some(ig_node.debug_scope),
                    LinearOp::Branch {
                        target: ig_labels[1],
                        phi_args: ig_entry_phis[1].clone(),
                    },
                );

                // Emit each inner gamma branch with routing
                for (ig_bi, &ig_rid) in ig_regions.iter().enumerate() {
                    self.emit(
                        Some(self.func.regions[ig_rid].debug_scope),
                        LinearOp::Label(ig_labels[ig_bi]),
                    );
                    self.linearize_region(ig_rid);

                    // Build upstream output vregs for this path:
                    // outputs from the parent region (n119 branch 1) results,
                    // with inner gamma outputs resolved from this specific branch.
                    let ig_reg = &self.func.regions[ig_rid];
                    let mut branch_out_vregs: Vec<VReg> = Vec::new();
                    for i in 0..up_data_output_count {
                        let parent_result = &self.func.region_results[region.results[i]];
                        if parent_result.kind != PortKind::Data {
                            continue;
                        }
                        match parent_result.source {
                            PortSource::Node(out_ref) if out_ref.node == inner_gamma_id => {
                                // This result comes from the inner gamma — resolve from this branch
                                let ig_out_idx = out_ref.index as usize;
                                let ig_data_results: Vec<_> = ig_reg
                                    .results
                                    .iter()
                                    .enumerate()
                                    .filter(|&(_, &rid)| {
                                        self.func.region_results[rid].kind == PortKind::Data
                                    })
                                    .collect();
                                if ig_out_idx < ig_data_results.len() {
                                    let (_, &ig_rid) = ig_data_results[ig_out_idx];
                                    let ig_result = &self.func.region_results[ig_rid];
                                    branch_out_vregs.push(self.resolve_vreg(ig_result.source));
                                } else {
                                    branch_out_vregs.push(self.resolve_vreg(parent_result.source));
                                }
                            }
                            _ => {
                                branch_out_vregs.push(self.resolve_vreg(parent_result.source));
                            }
                        }
                    }

                    // Check the control value for this inner gamma branch.
                    // We need to find what the PARENT region's ctrl output resolves
                    // to on this inner gamma branch. The parent result at ctrl_idx
                    // points to inner_gamma.output[N]. So we check inner gamma branch
                    // result at output index N.
                    let parent_ctrl_result = &self.func.region_results[region.results[ctrl_idx]];
                    let ig_out_idx = match parent_ctrl_result.source {
                        PortSource::Node(out_ref) if out_ref.node == inner_gamma_id => {
                            Some(out_ref.index as usize)
                        }
                        _ => None,
                    };
                    let ig_ctrl = if let Some(ig_oi) = ig_out_idx {
                        // Find the ig_oi-th DATA result in this inner gamma branch
                        let ig_data_results: Vec<_> = ig_reg
                            .results
                            .iter()
                            .enumerate()
                            .filter(|&(_, &rid)| {
                                self.func.region_results[rid].kind == PortKind::Data
                            })
                            .collect();
                        if ig_oi < ig_data_results.len() {
                            let (_, &rid) = ig_data_results[ig_oi];
                            let r = &self.func.region_results[rid];
                            match r.source {
                                PortSource::Node(out_ref) => matches!(
                                    &self.func.nodes[out_ref.node].kind,
                                    NodeKind::Simple(IrOp::Const { value })
                                        if *value == fusion.done_value
                                ),
                                PortSource::RegionArg(_) => false,
                            }
                        } else {
                            false
                        }
                    } else {
                        false
                    };

                    if ig_ctrl {
                        let phis = compute_done_phis(self, &branch_out_vregs);
                        self.emit(
                            Some(self.func.regions[ig_rid].debug_scope),
                            LinearOp::Branch {
                                target: ds_landing_label,
                                phi_args: phis,
                            },
                        );
                    } else {
                        let phis = compute_more_phis(self, &branch_out_vregs);
                        self.emit(
                            Some(self.func.regions[ig_rid].debug_scope),
                            LinearOp::Branch {
                                target: continue_label,
                                phi_args: phis,
                            },
                        );
                    }
                }
            } else {
                // Simple case: no inner gamma. The control output is a direct
                // constant or passthrough. Linearize and route.
                self.linearize_region(region_id);

                let mut branch_out_vregs: Vec<VReg> = Vec::new();
                for i in 0..up_data_output_count {
                    let result = &self.func.region_results[region.results[i]];
                    if result.kind == PortKind::Data {
                        branch_out_vregs.push(self.resolve_vreg(result.source));
                    }
                }

                let is_done = match ctrl_result.source {
                    PortSource::Node(out_ref) => matches!(
                        &self.func.nodes[out_ref.node].kind,
                        NodeKind::Simple(IrOp::Const { value }) if *value == fusion.done_value
                    ),
                    PortSource::RegionArg(_) => false,
                };

                if is_done {
                    let phis = compute_done_phis(self, &branch_out_vregs);
                    self.emit(
                        Some(self.func.regions[region_id].debug_scope),
                        LinearOp::Branch {
                            target: ds_landing_label,
                            phi_args: phis,
                        },
                    );
                } else {
                    let phis = compute_more_phis(self, &branch_out_vregs);
                    self.emit(
                        Some(self.func.regions[region_id].debug_scope),
                        LinearOp::Branch {
                            target: continue_label,
                            phi_args: phis,
                        },
                    );
                }
            }
        }

        // Phase 2: Emit continue_label + downstream's chain body.
        self.emit(Some(ds_node.debug_scope), LinearOp::Label(continue_label));

        // Now process the downstream's continue branch as a passthrough-exit chain.
        let ds_tail = self.find_tail_passthrough_gamma(ds_cont_branch);
        if let Some(ds_tail_id) = ds_tail {
            // Compute state_env and classification for the chain
            let (output_classes, control_done_value) = self.classify_chain_outputs(
                downstream_id,
                downstream_regions,
                ds_data_output_count,
            );

            let mut ds_state_env: Vec<VReg> = ds_landing_vregs.clone();

            // Pre-compute state_env from downstream continue branch results
            let ds_cont_results = self.func.regions[ds_cont_branch].results.clone();
            let ds_cont_data_results: Vec<usize> = ds_cont_results
                .iter()
                .enumerate()
                .filter(|&(_, &rid)| self.func.region_results[rid].kind == PortKind::Data)
                .map(|(i, _)| i)
                .collect();

            let mut ds_tail_output_to_landing: Vec<usize> = Vec::new();
            let ds_output_to_landing: Vec<usize> = (0..ds_data_output_count).collect();
            for (j, &result_idx) in ds_cont_data_results.iter().enumerate() {
                let landing_idx = if j < ds_output_to_landing.len() {
                    ds_output_to_landing[j]
                } else {
                    continue;
                };
                let result = &self.func.region_results[ds_cont_results[result_idx]];
                match result.source {
                    PortSource::Node(out_ref) if out_ref.node == ds_tail_id => {
                        let tail_out_idx = out_ref.index as usize;
                        while ds_tail_output_to_landing.len() <= tail_out_idx {
                            ds_tail_output_to_landing.push(usize::MAX);
                        }
                        ds_tail_output_to_landing[tail_out_idx] = landing_idx;

                        let tail_node = &self.func.nodes[ds_tail_id];
                        let tail_data_inputs: Vec<usize> = tail_node
                            .inputs
                            .iter()
                            .enumerate()
                            .skip(1)
                            .filter(|(_, inp)| inp.kind == PortKind::Data)
                            .map(|(idx, _)| idx)
                            .collect();
                        if tail_out_idx < tail_data_inputs.len() {
                            let input_idx = tail_data_inputs[tail_out_idx];
                            ds_state_env[landing_idx] =
                                self.resolve_vreg(tail_node.inputs[input_idx].source);
                        }
                    }
                    _ => {
                        ds_state_env[landing_idx] = self.resolve_vreg(result.source);
                    }
                }
            }

            // Build safe_exit_env
            let non_tail_gammas: HashSet<NodeId> = self.func.regions[ds_cont_branch]
                .nodes
                .iter()
                .filter(|&&nid| {
                    if nid == ds_tail_id {
                        return false;
                    }
                    let NodeKind::Gamma { regions: gr } = &self.func.nodes[nid].kind else {
                        return false;
                    };
                    gr.iter().any(|&rid| {
                        self.func.regions[rid].nodes.iter().any(|&child| {
                            matches!(self.func.nodes[child].kind, NodeKind::Gamma { .. })
                        })
                    })
                })
                .copied()
                .collect();

            let safe_exit_env: Vec<SafeExitVal> = if non_tail_gammas.is_empty() {
                ds_state_env.iter().map(|&v| SafeExitVal::Vreg(v)).collect()
            } else {
                ds_state_env
                    .iter()
                    .map(|&v| self.resolve_safe_exit_val_from_vreg(v, &non_tail_gammas))
                    .collect()
            };

            // Set chain exit context
            let prev_ctx = self.chain_exit_ctx.take();
            self.chain_exit_ctx = Some(ChainExitCtx {
                landing_label: ds_landing_label,
                landing_vregs: ds_landing_vregs.clone(),
                state_env: ds_state_env.clone(),
                output_to_landing: ds_output_to_landing.clone(),
                safe_exit_env,
                output_classes: output_classes.clone(),
                control_done_value,
            });
            let node_ids: Vec<NodeId> = self.func.regions[ds_cont_branch].nodes.clone();
            for &nid in &node_ids {
                if nid != ds_tail_id {
                    self.linearize_node(nid);
                }
            }
            self.chain_exit_ctx = prev_ctx;

            // Recurse with the tail gamma
            let NodeKind::Gamma { regions: tr } = &self.func.nodes[ds_tail_id].kind else {
                unreachable!()
            };
            let tr = tr.clone();
            self.try_linearize_passthrough_exit(
                ds_tail_id,
                &tr,
                Some(ChainInheritedState {
                    landing_label: ds_landing_label,
                    landing_vregs: ds_landing_vregs.clone(),
                    state_env: ds_state_env,
                    output_to_landing: ds_tail_output_to_landing,
                    output_classes,
                    control_done_value,
                }),
            );
        } else {
            // No tail gamma — shouldn't happen for our target, bail
            return false;
        }

        // Phase 5: Emit downstream's landing label (only if we own it)
        if owns_landing {
            self.emit(Some(ds_node.debug_scope), LinearOp::Label(ds_landing_label));
        }

        true
    }

    /// Classify all data outputs of a chain gamma into ChainOutputClass.
    /// Returns (classes, control_done_value).
    /// control_done_value is Some(val) if exactly one distinct boolean exists.
    fn classify_chain_outputs(
        &self,
        node_id: NodeId,
        regions: &[RegionId],
        data_output_count: usize,
    ) -> (Vec<ChainOutputClass>, Option<u64>) {
        let mut classes = Vec::with_capacity(data_output_count);

        // First pass: classify each output using the existing machinery
        let mut ctrl_indices = Vec::new();
        for i in 0..data_output_count {
            if self.func.is_chain_control_only_output(node_id, i, 0) {
                ctrl_indices.push(i);
            }
        }

        // Group control-only outputs into invariant vs boolean
        let (invariant_set, boolean_groups) =
            self.group_control_only_by_boolean(node_id, regions, &ctrl_indices);

        // Build the per-output classification
        for i in 0..data_output_count {
            if !ctrl_indices.contains(&i) {
                // Not control-only → real state
                classes.push(ChainOutputClass::RealState);
            } else if invariant_set.contains(&i) {
                classes.push(ChainOutputClass::Invariant);
            } else {
                // Find which boolean group this belongs to
                let mut found = false;
                for g in &boolean_groups {
                    if g.output_indices.contains(&i) {
                        classes.push(ChainOutputClass::ControlBoolean {
                            done_value: g.done_value,
                        });
                        found = true;
                        break;
                    }
                }
                if !found {
                    // Shouldn't happen, but treat as real state for safety
                    classes.push(ChainOutputClass::RealState);
                }
            }
        }

        // Determine if the predicate is a control-only boolean.
        // This is the key boolean for the two-landing optimization.
        // Dead control outputs don't affect this — only the predicate matters.
        let node = &self.func.nodes[node_id];
        let control_done_value = match &node.inputs[0].source {
            PortSource::Node(out_ref) => {
                let pred_is_ctrl =
                    self.func
                        .is_chain_control_only_output(out_ref.node, out_ref.index as usize, 0);
                if pred_is_ctrl {
                    if let NodeKind::Gamma {
                        regions: pred_regions,
                    } = &self.func.nodes[out_ref.node].kind
                    {
                        match self.classify_ctrl_output(
                            out_ref.node,
                            pred_regions,
                            out_ref.index as usize,
                        ) {
                            CtrlOutputKind::Boolean { done_value } => Some(done_value),
                            // Invariant predicate (always same value) — also record it
                            // as it means the chain entry is unconditional
                            _ => None,
                        }
                    } else {
                        None
                    }
                } else {
                    None
                }
            }
            _ => None,
        };

        // Debug assertion: every output is classified
        debug_assert_eq!(
            classes.len(),
            data_output_count,
            "classify_chain_outputs: output count mismatch"
        );

        (classes, control_done_value)
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

    /// Check if a region's ONLY code path is an error exit (no normal return).
    /// ErrorExit has been removed; this always returns false.
    fn region_is_error_only(&self, _region_id: RegionId) -> bool {
        false
    }

    /// Emit Copy ops for passthrough data inputs entering a gamma branch region.
    fn emit_gamma_entry_copies(&mut self, node: &Node, region_id: RegionId) {
        let region = &self.func.regions[region_id];
        let state_count = 1;
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
        // Track theta depth and clear chain_exit_ctx — theta is a loop, so
        // inner gammas must NOT exit to an outer chain landing.
        self.theta_depth += 1;
        let prev_chain_ctx = self.chain_exit_ctx.take();

        let node = &self.func.nodes[node_id];
        let body_region = &self.func.regions[body];
        let state_count = 1;

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
        let debug_theta = std::env::var("KAJIT_DEBUG_THETA_LIN").is_ok();
        for i in 0..loop_var_count {
            let result = &self.func.region_results[body_region.results[i + 1]]; // +1 to skip predicate
            if result.kind == PortKind::Data {
                let src_vreg = self.resolve_vreg(result.source);
                let arg = &self.func.region_args[body_region.args[i]];
                if let Some(dst_vreg) = arg.vreg
                    && src_vreg != dst_vreg
                {
                    feedback_phi_args.push((src_vreg, dst_vreg));
                } else if debug_theta {
                    eprintln!(
                        "[theta-lin] DROPPED feedback phi theta={:?} i={}/{}: src=v{} dst={:?} (eq={})",
                        node_id,
                        i,
                        loop_var_count,
                        src_vreg.index(),
                        arg.vreg.map(|v| v.index()),
                        arg.vreg == Some(src_vreg)
                    );
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

        // Restore chain_exit_ctx and theta depth after theta body
        self.chain_exit_ctx = prev_chain_ctx;
        self.theta_depth -= 1;
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
        if !self.fusion_applied {
            self.pre_scan_control_fusions(body);
            self.fusion_applied = true;
        }
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
            | LinearOp::FuncEnd
    )
}

fn op_uses(op: &LinearOp, func_end_uses: Option<&[VReg]>) -> Vec<VReg> {
    match op {
        LinearOp::BinOp { lhs, rhs, .. } => vec![*lhs, *rhs],
        LinearOp::UnaryOp { src, .. } => vec![*src],
        LinearOp::Copy { src, .. } => vec![*src],
        LinearOp::StoreToAddr { addr, src, .. } => vec![*addr, *src],
        LinearOp::LoadFromAddr { addr, .. } => vec![*addr],
        LinearOp::WriteToSlot { src, .. } => vec![*src],
        LinearOp::CallIntrinsic { args, .. } => args.clone(),
        LinearOp::CallPure { args, .. } | LinearOp::CallEffect { args, .. } => args.clone(),
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
        LinearOp::CallLambda { args, .. } => args.clone(),
        LinearOp::FuncEnd => func_end_uses.unwrap_or_default().to_vec(),
        LinearOp::Const { .. }
        | LinearOp::DataAddr { .. }
        | LinearOp::ExternAddr { .. }
        | LinearOp::SlotAddr { .. }
        | LinearOp::ReadFromSlot { .. }
        | LinearOp::Label(_)
        | LinearOp::FuncStart { .. } => Vec::new(),
    }
}

fn op_defs(op: &LinearOp) -> Vec<VReg> {
    match op {
        LinearOp::Const { dst, .. }
        | LinearOp::DataAddr { dst, .. }
        | LinearOp::ExternAddr { dst, .. } => vec![*dst],
        LinearOp::BinOp { dst, .. } => vec![*dst],
        LinearOp::UnaryOp { dst, .. } => vec![*dst],
        LinearOp::Copy { dst, .. } => vec![*dst],
        LinearOp::SlotAddr { dst, .. } => vec![*dst],
        LinearOp::LoadFromAddr { dst, .. } => vec![*dst],
        LinearOp::ReadFromSlot { dst, .. } => vec![*dst],
        LinearOp::CallIntrinsic { dst, .. } => dst.iter().copied().collect(),
        LinearOp::CallPure { dst, .. } | LinearOp::CallEffect { dst, .. } => vec![*dst],
        LinearOp::FuncStart { data_args, .. } => data_args.clone(),
        LinearOp::CallLambda { results, .. } => results.clone(),
        LinearOp::StoreToAddr { .. }
        | LinearOp::WriteToSlot { .. }
        | LinearOp::Label(_)
        | LinearOp::Branch { .. }
        | LinearOp::BranchIf { .. }
        | LinearOp::BranchIfZero { .. }
        | LinearOp::JumpTable { .. }
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
        LinearOp::StoreToAddr { addr, src, .. } => {
            rewrite(addr, &mut resolve);
            rewrite(src, &mut resolve);
        }
        LinearOp::LoadFromAddr { addr, .. } => rewrite(addr, &mut resolve),
        LinearOp::WriteToSlot { src, .. } => rewrite(src, &mut resolve),
        LinearOp::CallIntrinsic { args, .. }
        | LinearOp::CallPure { args, .. }
        | LinearOp::CallEffect { args, .. }
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
        LinearOp::Const { .. }
        | LinearOp::DataAddr { .. }
        | LinearOp::ExternAddr { .. }
        | LinearOp::SlotAddr { .. }
        | LinearOp::ReadFromSlot { .. }
        | LinearOp::Label(_)
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
            LinearOp::FuncEnd => {}
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
/// ErrorExit has been removed; this always returns false.
fn region_is_error_only_static(_func: &IrFunc, _region_id: RegionId) -> bool {
    false
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
        param_slot_count: func.param_slot_count,
        data_blobs: func.data_blobs.clone(),
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
        LinearOp::DataAddr { dst, blob_id } => {
            fmt_vreg(f, *dst)?;
            write!(f, " = data_addr({blob_id})")
        }
        LinearOp::ExternAddr { dst, symbol, .. } => {
            fmt_vreg(f, *dst)?;
            write!(f, " = extern_addr(@{symbol})")
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
        LinearOp::CallIntrinsic { func, args, dst } => {
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
            write!(f, ")")
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
        LinearOp::CallEffect { func, args, dst } => {
            fmt_vreg(f, *dst)?;
            write!(f, " = call_effect ")?;
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
        // Const(42) → StoreToAddr
        let mut builder = IrBuilder::new("u32", 0);
        {
            let mut rb = builder.root_region();
            let data = rb.const_val(42);
            let addr = rb.const_val(0);
            rb.store_to_addr(addr, data, Width::W4);
            rb.set_results(&[]);
        }
        let mut func = builder.finish();
        let ir = linearize(&mut func);

        // Expected: FuncStart, Const(42), Const(0), StoreToAddr, FuncEnd
        assert!(matches!(ir.ops[0], LinearOp::FuncStart { .. }));
        assert!(matches!(ir.ops[1], LinearOp::Const { .. }));
        assert!(matches!(ir.ops[2], LinearOp::Const { .. }));
        assert!(matches!(
            ir.ops[3],
            LinearOp::StoreToAddr {
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
            let addr = rb.const_val(0);
            rb.store_to_addr(addr, results[0], Width::W4);
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
            rb.call_intrinsic(
                IntrinsicFn(dummy_intrinsic as *const () as usize),
                &[],
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
            let data = rb.const_val(42);
            let addr = rb.const_val(0u64);
            rb.store_to_addr(addr, data, Width::W4);
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
            display.contains("store_addr [W4]"),
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
            LinearOp::StoreToAddr {
                addr: v0,
                src: v2,
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
            LinearOp::StoreToAddr { src, .. } => Some(*src),
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
            LinearOp::StoreToAddr {
                addr: v0,
                src: v2,
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
            .position(|op| matches!(op, LinearOp::StoreToAddr { .. }))
            .expect("optimized ops should still contain store");
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
    args: [%ms]
    n0 = Const(0x0) [] -> [v0]
    n1 = WriteToSlot(0) [v0, %ms:arg] -> [%ms]
    n14 = theta [%ms:n1] {
      region {
        args: [%ms]
        n2 = ReadFromSlot(0) [%ms:arg] -> [v1, %ms]
        n3 = Const(0x4) [] -> [v2]
        n4 = CmpNe [v1, v2] -> [v3]
        n11 = gamma [
          pred: v3
          in0: %ms:n2
        ] {
          branch 0:
            region {
              args: [%ms]
              n5 = ReadFromSlot(0) [%ms:arg] -> [v4, %ms]
              n6 = Const(0x1) [] -> [v5]
              n7 = Add [v4, v5] -> [v6]
              n8 = WriteToSlot(0) [v6, %ms:n5] -> [%ms]
              results: [%ms:n8]
            }
          branch 1:
            region {
              args: [%ms]
              results: [%ms:arg]
            }
        } -> [%ms]
        n12 = Const(0x0) [] -> [v7]
        results: [v7, %ms:n11]
      }
    } -> [%ms]
    n13 = ReadFromSlot(0) [%ms:n14] -> [v8, %ms]
    n15 = StoreToAddr(W4) [v7, v8, %ms:n13] -> [%ms]
    results: [%ms:n15]
  }
}
"#;
        let registry = kajit_ir::IntrinsicRegistry::empty();
        let mut func = kajit_ir_text::parse_ir(input, &registry).unwrap();
        kajit_ir::slot2reg::slot_to_reg(&mut func);
        let _ir = linearize(&mut func);
    }

    #[test]
    fn linearize_theta_shared_predicate_and_loopvar() {
        // Theta where a gamma output is used both as predicate AND as a
        // loop-carried variable result — the pattern from the real array
        // decoder that triggers v_N from v_N in register allocation.
        let input = r#"
lambda @0 (shape: "test") {
  region {
    args: [%ms]
    n0 = Const(0x0) [] -> [v0]
    n1 = Const(0x1) [] -> [v1]
    n10 = theta [v0, v1, %ms:arg] {
      region {
        args: [arg0, arg1, %ms]
        n2 = Const(0x4) [] -> [v2]
        n3 = CmpNe [arg0, v2] -> [v3]
        n8 = gamma [
          pred: v3
          in0: arg0
          in1: arg1
          in2: %ms:arg
        ] {
          branch 0:
            region {
              args: [arg0, arg1, %ms]
              n4 = Const(0x1) [] -> [v4]
              n5 = Add [arg0, v4] -> [v5]
              results: [v5, arg1, %ms:arg]
            }
          branch 1:
            region {
              args: [arg0, arg1, %ms]
              results: [arg0, arg1, %ms:arg]
            }
        } -> [v6, v7, %ms]
        results: [v7, v6, v7, %ms:n8]
      }
    } -> [v8, v9, %ms]
    n9 = StoreToAddr(W4) [v8, v8, %ms:n10] -> [%ms]
    results: [%ms:n9]
  }
}
"#;
        let registry = kajit_ir::IntrinsicRegistry::empty();
        let mut func = kajit_ir_text::parse_ir(input, &registry).unwrap();
        let _ir = linearize(&mut func);
    }
}
