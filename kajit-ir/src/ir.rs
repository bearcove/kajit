//! RVSDG-based intermediate representation for kajit's JIT compiler.
//!
//! The IR captures deserialization/serialization semantics as a Regionalized
//! Value State Dependence Graph. Values flow through data edges, effects
//! (cursor movement, output writes) are ordered by state edges, and control
//! flow is represented by structured region nodes (gamma/theta/lambda).
//!
//! Formats produce IR via the builder API. The linearizer converts the RVSDG
//! to a linear instruction sequence. The backend emits machine code.

use std::fmt;
use std::marker::PhantomData;

use crate::ErrorCode;

// ─── Arena and ID types ─────────────────────────────────────────────────────

/// Typed index into a [`Arena`]. Generic over the element type for type safety.
pub struct Id<T> {
    index: u32,
    _phantom: PhantomData<T>,
}

// Manual impls to avoid requiring T: Clone/Copy/Debug/PartialEq/Eq/Hash.
// The derived versions would propagate T's bounds, but Id<T> equality
// depends only on the index, not on T.
impl<T> Clone for Id<T> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<T> Copy for Id<T> {}

impl<T> PartialEq for Id<T> {
    fn eq(&self, other: &Self) -> bool {
        self.index == other.index
    }
}

impl<T> Eq for Id<T> {}

impl<T> PartialOrd for Id<T> {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl<T> Ord for Id<T> {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.index.cmp(&other.index)
    }
}

impl<T> std::hash::Hash for Id<T> {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.index.hash(state);
    }
}

impl<T> fmt::Debug for Id<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "#{}", self.index)
    }
}

impl<T> Id<T> {
    pub fn new(index: u32) -> Self {
        Self {
            index,
            _phantom: PhantomData,
        }
    }

    /// The raw index into the arena.
    pub fn index(self) -> usize {
        self.index as usize
    }
}

/// Vec-backed arena with typed indexing via [`Id`].
pub struct Arena<T> {
    items: Vec<T>,
}

impl<T> Default for Arena<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T> Arena<T> {
    pub fn new() -> Self {
        Self { items: Vec::new() }
    }

    pub fn push(&mut self, item: T) -> Id<T> {
        let id = Id::new(self.items.len() as u32);
        self.items.push(item);
        id
    }

    pub fn len(&self) -> usize {
        self.items.len()
    }

    pub fn is_empty(&self) -> bool {
        self.items.is_empty()
    }

    pub fn iter(&self) -> impl Iterator<Item = (Id<T>, &T)> {
        self.items
            .iter()
            .enumerate()
            .map(|(i, item)| (Id::new(i as u32), item))
    }
}

impl<T> std::ops::Index<Id<T>> for Arena<T> {
    type Output = T;
    fn index(&self, id: Id<T>) -> &T {
        &self.items[id.index()]
    }
}

impl<T> std::ops::IndexMut<Id<T>> for Arena<T> {
    fn index_mut(&mut self, id: Id<T>) -> &mut T {
        &mut self.items[id.index()]
    }
}

impl<T: fmt::Debug> fmt::Debug for Arena<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_list().entries(self.items.iter()).finish()
    }
}

// ─── Type aliases ───────────────────────────────────────────────────────────

pub type NodeId = Id<Node>;
pub type RegionId = Id<Region>;

/// Marker type for virtual register IDs.
pub struct VRegMarker;
/// A virtual register — unlimited, the backend maps to physical registers.
// r[impl ir.vregs]
pub type VReg = Id<VRegMarker>;

/// Marker type for stack slot IDs.
pub struct SlotMarker;
/// An abstract stack slot — the backend assigns frame offsets.
// r[impl ir.slots]
pub type SlotId = Id<SlotMarker>;

/// Marker type for lambda IDs.
pub struct LambdaMarker;
/// A lambda identifier for cross-referencing between IrFuncs.
pub type LambdaId = Id<LambdaMarker>;

// ─── Value types ────────────────────────────────────────────────────────────

/// Width of a memory operation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Width {
    /// 1 byte
    W1,
    /// 2 bytes
    W2,
    /// 4 bytes
    W4,
    /// 8 bytes
    W8,
}

impl Width {
    /// The width in bytes.
    pub fn bytes(self) -> u32 {
        match self {
            Width::W1 => 1,
            Width::W2 => 2,
            Width::W4 => 4,
            Width::W8 => 8,
        }
    }
}

impl fmt::Display for Width {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Width::W1 => write!(f, "W1"),
            Width::W2 => write!(f, "W2"),
            Width::W4 => write!(f, "W4"),
            Width::W8 => write!(f, "W8"),
        }
    }
}

/// An intrinsic function pointer, wrapped for type safety.
#[derive(Clone, Copy, PartialEq, Eq)]
pub struct IntrinsicFn(pub usize);

impl fmt::Debug for IntrinsicFn {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "IntrinsicFn({:#x})", self.0)
    }
}

impl fmt::Display for IntrinsicFn {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:#x}", self.0)
    }
}

/// Bidirectional mapping between string names and [`IntrinsicFn`] values.
///
/// Used by the text-format display and parser to print/resolve intrinsic names
/// and named constants instead of raw hex addresses.
pub struct IntrinsicRegistry {
    entries: Vec<(String, IntrinsicFn)>,
    const_entries: Vec<(String, u64)>,
}

impl IntrinsicRegistry {
    /// Build a registry from all known intrinsics (postcard + JSON).
    pub fn new() -> Self {
        let entries = Vec::new();
        // no default intrinsic registry in kajit-ir

        Self {
            entries,
            const_entries: Vec::new(),
        }
    }

    /// Build an empty registry (for tests that don't use real intrinsics).
    pub fn empty() -> Self {
        Self {
            entries: Vec::new(),
            const_entries: Vec::new(),
        }
    }

    /// Register a custom intrinsic (e.g. test-only functions).
    pub fn register(&mut self, name: impl Into<String>, func: IntrinsicFn) {
        let name = name.into();
        if self
            .entries
            .iter()
            .any(|(existing_name, existing_func)| existing_name == &name && *existing_func == func)
        {
            return;
        }
        self.entries.push((name, func));
    }

    /// Register a named constant (e.g. pointer-valued shape-specific metadata).
    pub fn register_const(&mut self, name: impl Into<String>, value: u64) {
        let name = name.into();
        if self
            .const_entries
            .iter()
            .any(|(existing_name, existing_value)| {
                existing_name == &name && *existing_value == value
            })
        {
            return;
        }
        self.const_entries.push((name, value));
    }

    /// Look up the name for an intrinsic function pointer.
    pub fn name_of(&self, func: IntrinsicFn) -> Option<&str> {
        self.entries
            .iter()
            .find(|(_, f)| *f == func)
            .map(|(name, _)| name.as_str())
    }

    /// Look up the function pointer for a name.
    pub fn func_by_name(&self, name: &str) -> Option<IntrinsicFn> {
        self.entries
            .iter()
            .find(|(n, _)| n == name)
            .map(|(_, f)| *f)
    }

    /// Look up the name for a named constant.
    pub fn const_name_of(&self, value: u64) -> Option<&str> {
        self.const_entries
            .iter()
            .find(|(_, existing_value)| *existing_value == value)
            .map(|(name, _)| name.as_str())
    }

    /// Look up the value for a named constant.
    pub fn const_by_name(&self, name: &str) -> Option<u64> {
        self.const_entries
            .iter()
            .find(|(existing_name, _)| existing_name == name)
            .map(|(_, value)| *value)
    }

    /// Iterate over named constants in insertion order.
    pub fn const_entries(&self) -> impl Iterator<Item = (&str, u64)> + '_ {
        self.const_entries
            .iter()
            .map(|(name, value)| (name.as_str(), *value))
    }
}

impl Default for IntrinsicRegistry {
    fn default() -> Self {
        Self::new()
    }
}

// ─── Ports and edges ────────────────────────────────────────────────────────

// r[impl ir.rvsdg.ports]

/// What a port carries.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PortKind {
    /// Carries a data value (backed by a VReg at runtime).
    Data,
    // r[impl ir.edges.state.cursor]
    /// Carries a cursor state token (orders cursor operations).
    StateCursor,
    // r[impl ir.edges.state.output]
    /// Carries an output state token (orders output writes).
    StateOutput,
}

// r[impl ir.edges.data]
// r[impl ir.edges.state]

/// A reference to a node's output port.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct OutputRef {
    pub node: NodeId,
    pub index: u16,
}

/// A reference to a region argument.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RegionArgRef {
    pub region: RegionId,
    pub index: u16,
}

/// Where an input port gets its value.
///
/// Every input port has exactly one source (RVSDG invariant). Edges are
/// implicit — stored inline in the input port rather than as separate objects.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PortSource {
    /// Connected to an output of a node within the same region.
    Node(OutputRef),
    /// Connected to a region argument (value entering from outside).
    RegionArg(RegionArgRef),
}

/// An input port on a node.
#[derive(Debug, Clone)]
pub struct InputPort {
    pub kind: PortKind,
    pub source: PortSource,
}

/// An output port on a node.
#[derive(Debug, Clone)]
pub struct OutputPort {
    pub kind: PortKind,
    /// For data outputs, the VReg assigned to this output.
    pub vreg: Option<VReg>,
}

// ─── Region ─────────────────────────────────────────────────────────────────

// r[impl ir.rvsdg.regions]

/// An argument entering a region from outside.
#[derive(Debug, Clone)]
pub struct RegionArg {
    pub kind: PortKind,
    pub vreg: Option<VReg>,
}

/// A result leaving a region to outside.
#[derive(Debug, Clone)]
pub struct RegionResult {
    pub kind: PortKind,
    pub source: PortSource,
}

/// A region is an ordered set of nodes with input arguments and output results.
/// Regions nest — a node inside a region may itself contain sub-regions.
pub struct Region {
    /// Arguments entering this region (correspond to outer input ports).
    pub args: Vec<RegionArg>,
    /// Results leaving this region (correspond to outer output ports).
    pub results: Vec<RegionResult>,
    /// Nodes contained in this region, in insertion order.
    pub nodes: Vec<NodeId>,
}

impl fmt::Debug for Region {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Region")
            .field("args", &self.args)
            .field("results", &self.results)
            .field("nodes", &self.nodes)
            .finish()
    }
}

// ─── Node ───────────────────────────────────────────────────────────────────

/// A node in the RVSDG.
pub struct Node {
    /// The containing region.
    pub region: RegionId,
    /// Input ports.
    pub inputs: Vec<InputPort>,
    /// Output ports.
    pub outputs: Vec<OutputPort>,
    /// The node's type and payload.
    pub kind: NodeKind,
}

impl fmt::Debug for Node {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Node")
            .field("region", &self.region)
            .field("inputs", &self.inputs)
            .field("outputs", &self.outputs)
            .field("kind", &self.kind)
            .finish()
    }
}

// r[impl ir.rvsdg.nodes.simple]
// r[impl ir.rvsdg.nodes.gamma]
// r[impl ir.rvsdg.nodes.theta]
// r[impl ir.rvsdg.nodes.lambda]
// r[impl ir.rvsdg.nodes.apply]

/// The type-specific payload of a node.
#[derive(Debug)]
pub enum NodeKind {
    /// A simple node representing a single operation.
    Simple(IrOp),

    /// A conditional (gamma node). The first input is the predicate.
    /// Each sub-region receives the same passthrough inputs and produces
    /// the same number of outputs. At runtime, exactly one region executes.
    Gamma {
        /// Sub-regions, one per branch. Branch index matches predicate value.
        regions: Vec<RegionId>,
    },

    /// A tail-controlled loop (theta node). The body region's first result
    /// is the loop predicate (0 = exit). Remaining results feed back as
    /// inputs for the next iteration.
    Theta {
        /// The single body region.
        body: RegionId,
    },

    /// A function definition containing a single body region.
    Lambda {
        body: RegionId,
        shape: &'static facet::Shape,
        lambda_id: LambdaId,
    },

    /// A function call to a lambda.
    Apply { target: LambdaId },
}

// ─── IrOp ───────────────────────────────────────────────────────────────────

// r[impl ir.ops]

/// The operation vocabulary for RVSDG simple nodes.
///
/// Each variant documents its port signature as:
/// `Inputs: [port_kind, ...] Outputs: [port_kind, ...]`
#[derive(Debug, Clone)]
pub enum IrOp {
    // ── Cursor ops ──────────────────────────────────────────────────
    // r[impl ir.ops.cursor]
    /// Read N bytes from cursor. Advances cursor.
    /// Inputs: [StateCursor]. Outputs: [Data, StateCursor].
    ReadBytes { count: u32 },

    /// Read one byte without advancing.
    /// Inputs: [StateCursor]. Outputs: [Data, StateCursor].
    PeekByte,

    /// Skip N bytes (static count).
    /// Inputs: [StateCursor]. Outputs: [StateCursor].
    AdvanceCursor { count: u32 },

    /// Skip N bytes (dynamic count from data input).
    /// Inputs: [Data, StateCursor]. Outputs: [StateCursor].
    AdvanceCursorBy,

    /// Assert N bytes remain. Error exit on failure.
    /// Inputs: [StateCursor]. Outputs: [StateCursor].
    BoundsCheck { count: u32 },

    /// Snapshot cursor position into a data value.
    /// Inputs: [StateCursor]. Outputs: [Data, StateCursor].
    SaveCursor,

    /// Restore cursor from a saved snapshot.
    /// Inputs: [Data, StateCursor]. Outputs: [StateCursor].
    RestoreCursor,

    // ── Output ops ──────────────────────────────────────────────────
    // r[impl ir.ops.output]
    /// Write src to out+offset.
    /// Inputs: [Data, StateOutput]. Outputs: [StateOutput].
    WriteToField { offset: u32, width: Width },

    /// Read from out+offset.
    /// Inputs: [StateOutput]. Outputs: [Data, StateOutput].
    ReadFromField { offset: u32, width: Width },

    /// Save the current output pointer (`out` base) as a data value.
    /// Inputs: [StateOutput]. Outputs: [Data, StateOutput].
    SaveOutPtr,

    /// Set the current output pointer (`out` base) from a data value.
    /// Inputs: [Data, StateOutput]. Outputs: [StateOutput].
    SetOutPtr,

    // ── Stack ops ───────────────────────────────────────────────────
    // r[impl ir.ops.stack]
    /// Compute the address of a stack slot (`sp + slot_offset`).
    /// Inputs: []. Outputs: [Data].
    SlotAddr { slot: SlotId },

    /// Write to an abstract stack slot.
    /// Inputs: [Data]. Outputs: [].
    WriteToSlot { slot: SlotId },

    /// Read from an abstract stack slot.
    /// Inputs: []. Outputs: [Data].
    ReadFromSlot { slot: SlotId },

    // ── Arithmetic (pure) ───────────────────────────────────────────
    // r[impl ir.ops.arithmetic]
    /// Load an immediate constant.
    /// Inputs: []. Outputs: [Data].
    Const { value: u64 },

    /// Binary addition.
    /// Inputs: [Data, Data]. Outputs: [Data].
    Add,

    /// Binary subtraction.
    /// Inputs: [Data, Data]. Outputs: [Data].
    Sub,

    /// Bitwise AND.
    /// Inputs: [Data, Data]. Outputs: [Data].
    And,

    /// Bitwise OR.
    /// Inputs: [Data, Data]. Outputs: [Data].
    Or,

    /// Logical right shift.
    /// Inputs: [Data, Data]. Outputs: [Data].
    Shr,

    /// Logical left shift.
    /// Inputs: [Data, Data]. Outputs: [Data].
    Shl,

    /// Bitwise XOR.
    /// Inputs: [Data, Data]. Outputs: [Data].
    Xor,

    /// Compare not-equal. Returns 1 if lhs != rhs, else 0.
    /// Inputs: [Data, Data]. Outputs: [Data].
    CmpNe,

    /// Zigzag decode for postcard signed integers.
    /// Inputs: [Data]. Outputs: [Data].
    ZigzagDecode { wide: bool },

    /// Sign-extend a narrow value.
    /// Inputs: [Data]. Outputs: [Data].
    SignExtend { from_width: Width },

    // ── Call ops ─────────────────────────────────────────────────────
    // r[impl ir.ops.call]
    /// Call an `extern "C"` intrinsic. Full barrier.
    /// Inputs: [Data * arg_count, StateCursor, StateOutput].
    /// Outputs: [Data (if has_result), StateCursor, StateOutput].
    // r[impl ir.edges.state.barrier]
    CallIntrinsic {
        func: IntrinsicFn,
        arg_count: u8,
        has_result: bool,
        field_offset: u32,
    },

    /// Call a pure function (no side effects).
    /// Inputs: [Data * arg_count]. Outputs: [Data].
    CallPure { func: IntrinsicFn, arg_count: u8 },

    // ── Error ops ───────────────────────────────────────────────────
    // r[impl ir.ops.error]
    /// Set error code and abort the containing region.
    /// Inputs: [StateCursor]. Outputs: [].
    ErrorExit { code: ErrorCode },

    // ── SIMD ops ────────────────────────────────────────────────────
    // r[impl ir.ops.simd]
    /// Vectorized scan for `"` or `\` in a string.
    /// Inputs: [StateCursor].
    /// Outputs: [Data (position), Data (kind: quote vs escape), StateCursor].
    SimdStringScan,

    /// Skip whitespace bytes using SIMD.
    /// Inputs: [StateCursor]. Outputs: [StateCursor].
    SimdWhitespaceSkip,
}

// ─── Effect classification ──────────────────────────────────────────────────

// r[impl ir.effects]

/// Effect classification of an operation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Effect {
    /// No side effects. Data edges only. Can be reordered, CSE'd, DCE'd.
    Pure,
    /// Reads or modifies cursor state.
    Cursor,
    /// Writes to the output struct.
    Output,
    /// May touch any state (full barrier).
    Barrier,
}

/// Per-op metadata used by optimization passes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct IrOpMetadata {
    /// Coarse effect class for scheduling/analysis.
    pub effect: Effect,
    /// Static cursor advancement in bytes, if known.
    ///
    /// `Some(0)` means the op preserves the cursor position.
    /// `None` means the op either does not participate in the cursor chain, or
    /// the advancement is dynamic/unknown.
    pub cursor_advance: Option<u32>,
    /// Whether the op has any observable side effects and must not be removed
    /// solely based on dead outputs.
    pub has_side_effects: bool,
}

// r[impl ir.effects.independence]
impl IrOp {
    /// Returns pass metadata for this op.
    pub fn metadata(&self) -> IrOpMetadata {
        let (effect, cursor_advance) = match self {
            // Pure ops
            IrOp::Const { .. }
            | IrOp::Add
            | IrOp::Sub
            | IrOp::And
            | IrOp::Or
            | IrOp::Shr
            | IrOp::Shl
            | IrOp::Xor
            | IrOp::CmpNe
            | IrOp::ZigzagDecode { .. }
            | IrOp::SignExtend { .. }
            | IrOp::SlotAddr { .. }
            | IrOp::CallPure { .. } => (Effect::Pure, None),

            // Cursor ops
            IrOp::ReadBytes { count } => (Effect::Cursor, Some(*count)),
            IrOp::AdvanceCursor { count } => (Effect::Cursor, Some(*count)),
            IrOp::PeekByte | IrOp::SaveCursor | IrOp::BoundsCheck { .. } => {
                (Effect::Cursor, Some(0))
            }
            IrOp::AdvanceCursorBy
            | IrOp::RestoreCursor
            | IrOp::WriteToSlot { .. }
            | IrOp::ReadFromSlot { .. }
            | IrOp::SimdStringScan
            | IrOp::SimdWhitespaceSkip
            | IrOp::ErrorExit { .. } => (Effect::Cursor, None),

            // Output ops
            IrOp::WriteToField { .. }
            | IrOp::ReadFromField { .. }
            | IrOp::SaveOutPtr
            | IrOp::SetOutPtr => (Effect::Output, None),

            // Barrier ops
            IrOp::CallIntrinsic { .. } => (Effect::Barrier, None),
        };

        IrOpMetadata {
            effect,
            cursor_advance,
            has_side_effects: effect != Effect::Pure,
        }
    }

    /// Returns the effect classification of this op.
    pub fn effect(&self) -> Effect {
        self.metadata().effect
    }

    /// Returns the static cursor advancement in bytes when known.
    pub fn cursor_advance(&self) -> Option<u32> {
        self.metadata().cursor_advance
    }

    /// Returns whether this op has side effects.
    pub fn has_side_effects(&self) -> bool {
        self.metadata().has_side_effects
    }
}

// ─── IrFunc ─────────────────────────────────────────────────────────────────

// r[impl ir.rvsdg]

/// Sentinel region ID for the root lambda node (not inside any region).
const ROOT_REGION: RegionId = Id {
    index: u32::MAX,
    _phantom: PhantomData,
};

/// The top-level IR container for one compiled shape.
pub struct IrFunc {
    /// All nodes.
    pub nodes: Arena<Node>,
    /// All regions.
    pub regions: Arena<Region>,
    /// The root lambda node.
    pub root: NodeId,
    /// Next VReg to allocate.
    pub vreg_count: u32,
    /// Next stack slot to allocate.
    pub slot_count: u32,
    /// Lambda registry: maps LambdaId to the NodeId of the lambda node.
    pub lambdas: Vec<NodeId>,
}

impl IrFunc {
    /// Allocate a fresh virtual register.
    pub fn fresh_vreg(&mut self) -> VReg {
        let id = VReg::new(self.vreg_count);
        self.vreg_count += 1;
        id
    }

    /// Allocate a fresh stack slot.
    pub fn fresh_slot(&mut self) -> SlotId {
        let id = SlotId::new(self.slot_count);
        self.slot_count += 1;
        id
    }

    /// Register a lambda and return its ID.
    pub fn register_lambda(&mut self, node: NodeId) -> LambdaId {
        let id = LambdaId::new(self.lambdas.len() as u32);
        self.lambdas.push(node);
        id
    }

    /// Look up the node ID for a lambda.
    pub fn lambda_node(&self, id: LambdaId) -> NodeId {
        self.lambdas[id.index()]
    }

    /// Total number of virtual registers allocated.
    pub fn vreg_count(&self) -> u32 {
        self.vreg_count
    }

    /// Total number of stack slots allocated.
    pub fn slot_count(&self) -> u32 {
        self.slot_count
    }

    /// Get the body region of the root lambda.
    pub fn root_body(&self) -> RegionId {
        match &self.nodes[self.root].kind {
            NodeKind::Lambda { body, .. } => *body,
            _ => unreachable!("root node must be a lambda"),
        }
    }

    /// Get the body region for a lambda by ID.
    pub fn lambda_body(&self, id: LambdaId) -> RegionId {
        let node = self.lambda_node(id);
        match &self.nodes[node].kind {
            NodeKind::Lambda { body, .. } => *body,
            _ => unreachable!("lambda registry must point to lambda nodes"),
        }
    }
}

// ─── Builder API ────────────────────────────────────────────────────────────

/// Builder for constructing an [`IrFunc`].
pub struct IrBuilder {
    func: IrFunc,
}

impl IrBuilder {
    /// Create a new builder for a shape's IR function.
    pub fn new(shape: &'static facet::Shape) -> Self {
        let mut func = IrFunc {
            nodes: Arena::new(),
            regions: Arena::new(),
            root: NodeId::new(0), // placeholder, set below
            vreg_count: 0,
            slot_count: 0,
            lambdas: Vec::new(),
        };

        // Create the root lambda's body region.
        // Standard arguments: cursor state, output state.
        let body = func.regions.push(Region {
            args: vec![
                RegionArg {
                    kind: PortKind::StateCursor,
                    vreg: None,
                },
                RegionArg {
                    kind: PortKind::StateOutput,
                    vreg: None,
                },
            ],
            results: Vec::new(),
            nodes: Vec::new(),
        });

        // Pre-allocate lambda ID 0 for the root.
        let lambda_id = LambdaId::new(0);
        func.lambdas.push(NodeId::new(0)); // placeholder, patched below

        let root = func.nodes.push(Node {
            region: ROOT_REGION,
            inputs: Vec::new(),
            outputs: Vec::new(),
            kind: NodeKind::Lambda {
                body,
                shape,
                lambda_id,
            },
        });

        func.lambdas[lambda_id.index()] = root;
        func.root = root;

        IrBuilder { func }
    }

    /// Get a [`RegionBuilder`] for the root lambda's body.
    pub fn root_region(&mut self) -> RegionBuilder<'_> {
        self.lambda_region(LambdaId::new(0))
    }

    /// Create a new lambda and return its ID.
    pub fn create_lambda(&mut self, shape: &'static facet::Shape) -> LambdaId {
        self.create_lambda_with_data_args(shape, 0)
    }

    /// Create a new lambda with `data_arg_count` leading data args.
    pub fn create_lambda_with_data_args(
        &mut self,
        shape: &'static facet::Shape,
        data_arg_count: usize,
    ) -> LambdaId {
        let lambda_id = LambdaId::new(self.func.lambdas.len() as u32);
        let mut args = Vec::with_capacity(data_arg_count + 2);
        for _ in 0..data_arg_count {
            args.push(RegionArg {
                kind: PortKind::Data,
                vreg: None,
            });
        }
        args.push(RegionArg {
            kind: PortKind::StateCursor,
            vreg: None,
        });
        args.push(RegionArg {
            kind: PortKind::StateOutput,
            vreg: None,
        });
        let body = self.func.regions.push(Region {
            args,
            results: Vec::new(),
            nodes: Vec::new(),
        });
        let node = self.func.nodes.push(Node {
            region: ROOT_REGION,
            inputs: Vec::new(),
            outputs: Vec::new(),
            kind: NodeKind::Lambda {
                body,
                shape,
                lambda_id,
            },
        });
        self.func.lambdas.push(node);
        lambda_id
    }

    /// Get a [`RegionBuilder`] for a lambda's body.
    pub fn lambda_region(&mut self, id: LambdaId) -> RegionBuilder<'_> {
        let body = self.func.lambda_body(id);
        let arg_count = self.func.regions[body].args.len();
        assert!(
            arg_count >= 2,
            "lambda body must have cursor/output state args"
        );
        let cs_idx = (arg_count - 2) as u16;
        let os_idx = (arg_count - 1) as u16;
        let cursor_state = PortSource::RegionArg(RegionArgRef {
            region: body,
            index: cs_idx,
        });
        let output_state = PortSource::RegionArg(RegionArgRef {
            region: body,
            index: os_idx,
        });

        RegionBuilder {
            func: &mut self.func,
            region: body,
            cursor_state,
            output_state,
        }
    }

    /// Finalize and return the [`IrFunc`].
    pub fn finish(self) -> IrFunc {
        self.func
    }
}

/// Builder for adding nodes to a specific region.
///
/// Tracks current state tokens and auto-threads them through effectful ops.
/// Formats call methods on this builder to construct the RVSDG without
/// manually wiring state edges.
pub struct RegionBuilder<'a> {
    func: &'a mut IrFunc,
    region: RegionId,
    cursor_state: PortSource,
    output_state: PortSource,
}

impl<'a> RegionBuilder<'a> {
    /// The region this builder targets.
    pub fn region(&self) -> RegionId {
        self.region
    }

    /// Current cursor state token source.
    pub fn cursor_state(&self) -> PortSource {
        self.cursor_state
    }

    /// Current output state token source.
    pub fn output_state(&self) -> PortSource {
        self.output_state
    }

    /// Access the underlying IrFunc (for fresh_vreg, fresh_slot, etc.).
    pub fn func(&mut self) -> &mut IrFunc {
        self.func
    }

    // ── Internal helpers ────────────────────────────────────────────

    fn add_node(&mut self, node: Node) -> NodeId {
        let id = self.func.nodes.push(node);
        self.func.regions[self.region].nodes.push(id);
        id
    }

    fn data_output(&mut self) -> OutputPort {
        let vreg = self.func.fresh_vreg();
        OutputPort {
            kind: PortKind::Data,
            vreg: Some(vreg),
        }
    }

    fn cursor_output() -> OutputPort {
        OutputPort {
            kind: PortKind::StateCursor,
            vreg: None,
        }
    }

    fn output_output() -> OutputPort {
        OutputPort {
            kind: PortKind::StateOutput,
            vreg: None,
        }
    }

    // ── Pure operations ─────────────────────────────────────────────

    /// Add a constant value. Returns the data output.
    pub fn const_val(&mut self, value: u64) -> PortSource {
        let out = self.data_output();
        let node = self.add_node(Node {
            region: self.region,
            inputs: vec![],
            outputs: vec![out],
            kind: NodeKind::Simple(IrOp::Const { value }),
        });
        PortSource::Node(OutputRef { node, index: 0 })
    }

    /// Add a binary arithmetic op. Returns the data output.
    pub fn binop(&mut self, op: IrOp, lhs: PortSource, rhs: PortSource) -> PortSource {
        debug_assert_eq!(op.effect(), Effect::Pure);
        let out = self.data_output();
        let node = self.add_node(Node {
            region: self.region,
            inputs: vec![
                InputPort {
                    kind: PortKind::Data,
                    source: lhs,
                },
                InputPort {
                    kind: PortKind::Data,
                    source: rhs,
                },
            ],
            outputs: vec![out],
            kind: NodeKind::Simple(op),
        });
        PortSource::Node(OutputRef { node, index: 0 })
    }

    /// Add a unary pure op (`ZigzagDecode`, `SignExtend`). Returns the data output.
    pub fn unary(&mut self, op: IrOp, src: PortSource) -> PortSource {
        debug_assert_eq!(op.effect(), Effect::Pure);
        let out = self.data_output();
        let node = self.add_node(Node {
            region: self.region,
            inputs: vec![InputPort {
                kind: PortKind::Data,
                source: src,
            }],
            outputs: vec![out],
            kind: NodeKind::Simple(op),
        });
        PortSource::Node(OutputRef { node, index: 0 })
    }

    // ── Cursor operations (auto-threaded) ───────────────────────────

    /// Read N bytes from cursor. Returns data output.
    pub fn read_bytes(&mut self, count: u32) -> PortSource {
        let data_out = self.data_output();
        let node = self.add_node(Node {
            region: self.region,
            inputs: vec![InputPort {
                kind: PortKind::StateCursor,
                source: self.cursor_state,
            }],
            outputs: vec![data_out, Self::cursor_output()],
            kind: NodeKind::Simple(IrOp::ReadBytes { count }),
        });
        self.cursor_state = PortSource::Node(OutputRef { node, index: 1 });
        PortSource::Node(OutputRef { node, index: 0 })
    }

    /// Peek one byte without advancing. Returns data output.
    pub fn peek_byte(&mut self) -> PortSource {
        let data_out = self.data_output();
        let node = self.add_node(Node {
            region: self.region,
            inputs: vec![InputPort {
                kind: PortKind::StateCursor,
                source: self.cursor_state,
            }],
            outputs: vec![data_out, Self::cursor_output()],
            kind: NodeKind::Simple(IrOp::PeekByte),
        });
        self.cursor_state = PortSource::Node(OutputRef { node, index: 1 });
        PortSource::Node(OutputRef { node, index: 0 })
    }

    /// Advance cursor by N bytes (static count).
    pub fn advance_cursor(&mut self, count: u32) {
        let node = self.add_node(Node {
            region: self.region,
            inputs: vec![InputPort {
                kind: PortKind::StateCursor,
                source: self.cursor_state,
            }],
            outputs: vec![Self::cursor_output()],
            kind: NodeKind::Simple(IrOp::AdvanceCursor { count }),
        });
        self.cursor_state = PortSource::Node(OutputRef { node, index: 0 });
    }

    /// Advance cursor by a dynamic amount.
    pub fn advance_cursor_by(&mut self, count: PortSource) {
        let node = self.add_node(Node {
            region: self.region,
            inputs: vec![
                InputPort {
                    kind: PortKind::Data,
                    source: count,
                },
                InputPort {
                    kind: PortKind::StateCursor,
                    source: self.cursor_state,
                },
            ],
            outputs: vec![Self::cursor_output()],
            kind: NodeKind::Simple(IrOp::AdvanceCursorBy),
        });
        self.cursor_state = PortSource::Node(OutputRef { node, index: 0 });
    }

    /// Assert N bytes remain. Error exits on failure.
    pub fn bounds_check(&mut self, count: u32) {
        let node = self.add_node(Node {
            region: self.region,
            inputs: vec![InputPort {
                kind: PortKind::StateCursor,
                source: self.cursor_state,
            }],
            outputs: vec![Self::cursor_output()],
            kind: NodeKind::Simple(IrOp::BoundsCheck { count }),
        });
        self.cursor_state = PortSource::Node(OutputRef { node, index: 0 });
    }

    /// Save cursor position. Returns data output (the saved position).
    // r[impl ir.cursor.snapshot]
    pub fn save_cursor(&mut self) -> PortSource {
        let data_out = self.data_output();
        let node = self.add_node(Node {
            region: self.region,
            inputs: vec![InputPort {
                kind: PortKind::StateCursor,
                source: self.cursor_state,
            }],
            outputs: vec![data_out, Self::cursor_output()],
            kind: NodeKind::Simple(IrOp::SaveCursor),
        });
        self.cursor_state = PortSource::Node(OutputRef { node, index: 1 });
        PortSource::Node(OutputRef { node, index: 0 })
    }

    /// Restore cursor from a saved position.
    pub fn restore_cursor(&mut self, saved: PortSource) {
        let node = self.add_node(Node {
            region: self.region,
            inputs: vec![
                InputPort {
                    kind: PortKind::Data,
                    source: saved,
                },
                InputPort {
                    kind: PortKind::StateCursor,
                    source: self.cursor_state,
                },
            ],
            outputs: vec![Self::cursor_output()],
            kind: NodeKind::Simple(IrOp::RestoreCursor),
        });
        self.cursor_state = PortSource::Node(OutputRef { node, index: 0 });
    }

    // ── Output operations (auto-threaded) ───────────────────────────

    /// Write a value to out+offset.
    pub fn write_to_field(&mut self, src: PortSource, offset: u32, width: Width) {
        let node = self.add_node(Node {
            region: self.region,
            inputs: vec![
                InputPort {
                    kind: PortKind::Data,
                    source: src,
                },
                InputPort {
                    kind: PortKind::StateOutput,
                    source: self.output_state,
                },
            ],
            outputs: vec![Self::output_output()],
            kind: NodeKind::Simple(IrOp::WriteToField { offset, width }),
        });
        self.output_state = PortSource::Node(OutputRef { node, index: 0 });
    }

    /// Read from out+offset. Returns data output.
    pub fn read_from_field(&mut self, offset: u32, width: Width) -> PortSource {
        let data_out = self.data_output();
        let node = self.add_node(Node {
            region: self.region,
            inputs: vec![InputPort {
                kind: PortKind::StateOutput,
                source: self.output_state,
            }],
            outputs: vec![data_out, Self::output_output()],
            kind: NodeKind::Simple(IrOp::ReadFromField { offset, width }),
        });
        self.output_state = PortSource::Node(OutputRef { node, index: 1 });
        PortSource::Node(OutputRef { node, index: 0 })
    }

    /// Save the current output pointer (`out` base). Returns data output.
    pub fn save_out_ptr(&mut self) -> PortSource {
        let data_out = self.data_output();
        let node = self.add_node(Node {
            region: self.region,
            inputs: vec![InputPort {
                kind: PortKind::StateOutput,
                source: self.output_state,
            }],
            outputs: vec![data_out, Self::output_output()],
            kind: NodeKind::Simple(IrOp::SaveOutPtr),
        });
        self.output_state = PortSource::Node(OutputRef { node, index: 1 });
        PortSource::Node(OutputRef { node, index: 0 })
    }

    /// Set the current output pointer (`out` base).
    pub fn set_out_ptr(&mut self, ptr: PortSource) {
        let node = self.add_node(Node {
            region: self.region,
            inputs: vec![
                InputPort {
                    kind: PortKind::Data,
                    source: ptr,
                },
                InputPort {
                    kind: PortKind::StateOutput,
                    source: self.output_state,
                },
            ],
            outputs: vec![Self::output_output()],
            kind: NodeKind::Simple(IrOp::SetOutPtr),
        });
        self.output_state = PortSource::Node(OutputRef { node, index: 0 });
    }

    /// Compute the address of a stack slot (`sp + slot_offset`).
    pub fn slot_addr(&mut self, slot: SlotId) -> PortSource {
        let data_out = self.data_output();
        let node = self.add_node(Node {
            region: self.region,
            inputs: vec![],
            outputs: vec![data_out],
            kind: NodeKind::Simple(IrOp::SlotAddr { slot }),
        });
        PortSource::Node(OutputRef { node, index: 0 })
    }

    /// Write a value to an abstract stack slot.
    pub fn write_to_slot(&mut self, slot: SlotId, src: PortSource) {
        let node = self.add_node(Node {
            region: self.region,
            inputs: vec![
                InputPort {
                    kind: PortKind::Data,
                    source: src,
                },
                InputPort {
                    kind: PortKind::StateCursor,
                    source: self.cursor_state,
                },
            ],
            outputs: vec![Self::cursor_output()],
            kind: NodeKind::Simple(IrOp::WriteToSlot { slot }),
        });
        self.cursor_state = PortSource::Node(OutputRef { node, index: 0 });
    }

    /// Read a value from an abstract stack slot.
    pub fn read_from_slot(&mut self, slot: SlotId) -> PortSource {
        let data_out = self.data_output();
        let node = self.add_node(Node {
            region: self.region,
            inputs: vec![InputPort {
                kind: PortKind::StateCursor,
                source: self.cursor_state,
            }],
            outputs: vec![data_out, Self::cursor_output()],
            kind: NodeKind::Simple(IrOp::ReadFromSlot { slot }),
        });
        self.cursor_state = PortSource::Node(OutputRef { node, index: 1 });
        PortSource::Node(OutputRef { node, index: 0 })
    }

    // ── Barrier operations (auto-threaded, all states) ──────────────

    /// Call an intrinsic. Full barrier: consumes and produces all state tokens.
    /// Returns the data result if `has_result` is true.
    pub fn call_intrinsic(
        &mut self,
        func: IntrinsicFn,
        args: &[PortSource],
        field_offset: u32,
        has_result: bool,
    ) -> Option<PortSource> {
        let mut inputs: Vec<InputPort> = args
            .iter()
            .map(|&src| InputPort {
                kind: PortKind::Data,
                source: src,
            })
            .collect();
        inputs.push(InputPort {
            kind: PortKind::StateCursor,
            source: self.cursor_state,
        });
        inputs.push(InputPort {
            kind: PortKind::StateOutput,
            source: self.output_state,
        });

        let mut outputs = Vec::new();
        if has_result {
            outputs.push(self.data_output());
        }
        let cursor_out_idx = outputs.len() as u16;
        outputs.push(Self::cursor_output());
        let output_out_idx = outputs.len() as u16;
        outputs.push(Self::output_output());

        let node = self.add_node(Node {
            region: self.region,
            inputs,
            outputs,
            kind: NodeKind::Simple(IrOp::CallIntrinsic {
                func,
                arg_count: args.len() as u8,
                has_result,
                field_offset,
            }),
        });

        self.cursor_state = PortSource::Node(OutputRef {
            node,
            index: cursor_out_idx,
        });
        self.output_state = PortSource::Node(OutputRef {
            node,
            index: output_out_idx,
        });

        if has_result {
            Some(PortSource::Node(OutputRef { node, index: 0 }))
        } else {
            None
        }
    }

    /// Call a pure function. No state tokens are consumed or produced.
    /// Returns the data result.
    pub fn call_pure(&mut self, func: IntrinsicFn, args: &[PortSource]) -> PortSource {
        let inputs: Vec<InputPort> = args
            .iter()
            .map(|&src| InputPort {
                kind: PortKind::Data,
                source: src,
            })
            .collect();
        let out = self.data_output();
        let node = self.add_node(Node {
            region: self.region,
            inputs,
            outputs: vec![out],
            kind: NodeKind::Simple(IrOp::CallPure {
                func,
                arg_count: args.len() as u8,
            }),
        });
        PortSource::Node(OutputRef { node, index: 0 })
    }

    // ── Error ───────────────────────────────────────────────────────

    /// Emit an error exit. Consumes cursor state (for byte offset recording).
    pub fn error_exit(&mut self, code: ErrorCode) {
        self.add_node(Node {
            region: self.region,
            inputs: vec![InputPort {
                kind: PortKind::StateCursor,
                source: self.cursor_state,
            }],
            outputs: vec![],
            kind: NodeKind::Simple(IrOp::ErrorExit { code }),
        });
    }

    // ── Structured control flow ─────────────────────────────────────

    /// Add a gamma node (conditional).
    ///
    /// `predicate`: the branch selector.
    /// `passthrough`: data values available to all branches.
    /// `branch_count`: number of branches.
    /// `build_branch`: called for each branch with (branch_index, RegionBuilder).
    ///   Must call `set_results` on each branch's builder before returning.
    ///
    /// Returns data outputs merged from all branches.
    pub fn gamma(
        &mut self,
        predicate: PortSource,
        passthrough: &[PortSource],
        branch_count: usize,
        mut build_branch: impl FnMut(usize, &mut RegionBuilder<'_>),
    ) -> Vec<PortSource> {
        // Create sub-regions, one per branch.
        let mut region_ids = Vec::with_capacity(branch_count);
        for _ in 0..branch_count {
            // Each region gets: passthrough data args + cursor state + output state.
            let mut args = Vec::with_capacity(passthrough.len() + 2);
            for &pt in passthrough {
                let _ = pt; // just for the count
                args.push(RegionArg {
                    kind: PortKind::Data,
                    vreg: None,
                });
            }
            args.push(RegionArg {
                kind: PortKind::StateCursor,
                vreg: None,
            });
            args.push(RegionArg {
                kind: PortKind::StateOutput,
                vreg: None,
            });

            let region = self.func.regions.push(Region {
                args,
                results: Vec::new(),
                nodes: Vec::new(),
            });
            region_ids.push(region);
        }

        // Build each branch.
        for (i, &region) in region_ids.iter().enumerate() {
            let cs_idx = passthrough.len() as u16;
            let os_idx = cs_idx + 1;
            let cursor_state = PortSource::RegionArg(RegionArgRef {
                region,
                index: cs_idx,
            });
            let output_state = PortSource::RegionArg(RegionArgRef {
                region,
                index: os_idx,
            });

            let mut branch_builder = RegionBuilder {
                func: self.func,
                region,
                cursor_state,
                output_state,
            };
            build_branch(i, &mut branch_builder);
        }

        // Determine the number of data results from the first branch.
        // All branches must produce the same number of results (RVSDG invariant).
        // Results layout: [data_results..., cursor_state, output_state].
        let total_results = self.func.regions[region_ids[0]].results.len();
        assert!(
            total_results >= 2,
            "gamma branch must have at least cursor + output state results"
        );
        let data_result_count = total_results - 2;

        // Build the gamma node's inputs: predicate + passthrough + state tokens.
        let mut inputs = Vec::with_capacity(1 + passthrough.len() + 2);
        inputs.push(InputPort {
            kind: PortKind::Data,
            source: predicate,
        });
        for &pt in passthrough {
            inputs.push(InputPort {
                kind: PortKind::Data,
                source: pt,
            });
        }
        inputs.push(InputPort {
            kind: PortKind::StateCursor,
            source: self.cursor_state,
        });
        inputs.push(InputPort {
            kind: PortKind::StateOutput,
            source: self.output_state,
        });

        // Build outputs: data results + state tokens.
        let mut outputs = Vec::with_capacity(data_result_count + 2);
        let mut data_outputs = Vec::with_capacity(data_result_count);
        for _ in 0..data_result_count {
            outputs.push(self.data_output());
        }
        let cursor_out_idx = outputs.len() as u16;
        outputs.push(Self::cursor_output());
        let output_out_idx = outputs.len() as u16;
        outputs.push(Self::output_output());

        let node = self.add_node(Node {
            region: self.region,
            inputs,
            outputs,
            kind: NodeKind::Gamma {
                regions: region_ids,
            },
        });

        // Update state tokens.
        self.cursor_state = PortSource::Node(OutputRef {
            node,
            index: cursor_out_idx,
        });
        self.output_state = PortSource::Node(OutputRef {
            node,
            index: output_out_idx,
        });

        // Return data output sources.
        for i in 0..data_result_count {
            data_outputs.push(PortSource::Node(OutputRef {
                node,
                index: i as u16,
            }));
        }
        data_outputs
    }

    /// Add a theta node (loop).
    ///
    /// `loop_vars`: initial values for loop-carried variables.
    /// `build_body`: called with a [`RegionBuilder`] for the body.
    ///   The body region receives `[loop_vars..., cursor_state, output_state]`
    ///   as arguments.
    ///   Must call `set_results` with `[predicate, loop_vars..., ...]`.
    ///   Predicate 0 = exit, nonzero = continue.
    ///
    /// Returns the final values of the loop variables after the loop exits.
    pub fn theta(
        &mut self,
        loop_vars: &[PortSource],
        build_body: impl FnOnce(&mut RegionBuilder<'_>),
    ) -> Vec<PortSource> {
        // Create the body region with args: [loop_vars..., cursor_state, output_state].
        let mut args = Vec::with_capacity(loop_vars.len() + 2);
        for _ in loop_vars {
            args.push(RegionArg {
                kind: PortKind::Data,
                vreg: None,
            });
        }
        args.push(RegionArg {
            kind: PortKind::StateCursor,
            vreg: None,
        });
        args.push(RegionArg {
            kind: PortKind::StateOutput,
            vreg: None,
        });

        let body = self.func.regions.push(Region {
            args,
            results: Vec::new(),
            nodes: Vec::new(),
        });

        // Build the body.
        let cs_idx = loop_vars.len() as u16;
        let os_idx = cs_idx + 1;
        let cursor_state = PortSource::RegionArg(RegionArgRef {
            region: body,
            index: cs_idx,
        });
        let output_state = PortSource::RegionArg(RegionArgRef {
            region: body,
            index: os_idx,
        });

        let mut body_builder = RegionBuilder {
            func: self.func,
            region: body,
            cursor_state,
            output_state,
        };
        build_body(&mut body_builder);

        // Body results layout: [predicate, loop_vars..., cursor_state, output_state].
        let total_results = self.func.regions[body].results.len();
        assert!(
            total_results >= 3,
            "theta body must have at least predicate + cursor + output state results"
        );
        let data_result_count = total_results - 2; // predicate + loop_vars
        let loop_var_count = data_result_count - 1; // minus predicate

        // Build theta node inputs: [loop_vars..., cursor_state, output_state].
        let mut inputs = Vec::with_capacity(loop_vars.len() + 2);
        for &lv in loop_vars {
            inputs.push(InputPort {
                kind: PortKind::Data,
                source: lv,
            });
        }
        inputs.push(InputPort {
            kind: PortKind::StateCursor,
            source: self.cursor_state,
        });
        inputs.push(InputPort {
            kind: PortKind::StateOutput,
            source: self.output_state,
        });

        // Outputs: [loop_vars..., cursor_state, output_state].
        let mut outputs = Vec::with_capacity(loop_var_count + 2);
        let mut data_outputs = Vec::new();
        for _ in 0..loop_var_count {
            outputs.push(self.data_output());
        }
        let cursor_out_idx = outputs.len() as u16;
        outputs.push(Self::cursor_output());
        let output_out_idx = outputs.len() as u16;
        outputs.push(Self::output_output());

        let node = self.add_node(Node {
            region: self.region,
            inputs,
            outputs,
            kind: NodeKind::Theta { body },
        });

        self.cursor_state = PortSource::Node(OutputRef {
            node,
            index: cursor_out_idx,
        });
        self.output_state = PortSource::Node(OutputRef {
            node,
            index: output_out_idx,
        });

        for i in 0..loop_var_count {
            data_outputs.push(PortSource::Node(OutputRef {
                node,
                index: i as u16,
            }));
        }
        data_outputs
    }

    /// Add an apply node (lambda call).
    ///
    /// `args`: data arguments to pass to the callee.
    /// `result_count`: number of data results returned by the callee.
    /// Cursor/output state are threaded automatically.
    pub fn apply(
        &mut self,
        target: LambdaId,
        args: &[PortSource],
        result_count: usize,
    ) -> Vec<PortSource> {
        let mut inputs = Vec::with_capacity(args.len() + 2);
        for &arg in args {
            inputs.push(InputPort {
                kind: PortKind::Data,
                source: arg,
            });
        }
        inputs.push(InputPort {
            kind: PortKind::StateCursor,
            source: self.cursor_state,
        });
        inputs.push(InputPort {
            kind: PortKind::StateOutput,
            source: self.output_state,
        });

        let mut outputs = Vec::with_capacity(result_count + 2);
        for _ in 0..result_count {
            outputs.push(self.data_output());
        }
        let cursor_out_idx = outputs.len() as u16;
        outputs.push(Self::cursor_output());
        let output_out_idx = outputs.len() as u16;
        outputs.push(Self::output_output());

        let node = self.add_node(Node {
            region: self.region,
            inputs,
            outputs,
            kind: NodeKind::Apply { target },
        });

        self.cursor_state = PortSource::Node(OutputRef {
            node,
            index: cursor_out_idx,
        });
        self.output_state = PortSource::Node(OutputRef {
            node,
            index: output_out_idx,
        });

        (0..result_count)
            .map(|i| {
                PortSource::Node(OutputRef {
                    node,
                    index: i as u16,
                })
            })
            .collect()
    }

    /// Set the region's results. Call at the end of building a region.
    ///
    /// `data_results`: any data values to pass out of the region.
    /// State tokens (cursor, output) are appended automatically.
    pub fn set_results(&mut self, data_results: &[PortSource]) {
        let region = &mut self.func.regions[self.region];
        region.results.clear();
        for &src in data_results {
            region.results.push(RegionResult {
                kind: PortKind::Data,
                source: src,
            });
        }
        region.results.push(RegionResult {
            kind: PortKind::StateCursor,
            source: self.cursor_state,
        });
        region.results.push(RegionResult {
            kind: PortKind::StateOutput,
            source: self.output_state,
        });
    }

    /// Allocate a fresh stack slot.
    pub fn alloc_slot(&mut self) -> SlotId {
        self.func.fresh_slot()
    }

    /// Provide region argument sources for passthrough data values
    /// inside a gamma/theta body. Returns sources for indices 0..count.
    pub fn region_args(&self, count: usize) -> Vec<PortSource> {
        (0..count)
            .map(|i| {
                PortSource::RegionArg(RegionArgRef {
                    region: self.region,
                    index: i as u16,
                })
            })
            .collect()
    }
}

// ─── Display ────────────────────────────────────────────────────────────────

impl fmt::Display for IrFunc {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let display = IrFuncDisplay {
            func: self,
            registry: None,
        };
        fmt::Display::fmt(&display, f)
    }
}

/// Wrapper for displaying an [`IrFunc`] with an optional [`IntrinsicRegistry`]
/// for resolving intrinsic names.
pub struct IrFuncDisplay<'a> {
    pub func: &'a IrFunc,
    pub registry: Option<&'a IntrinsicRegistry>,
}

impl<'a> fmt::Display for IrFuncDisplay<'a> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.func.fmt_node(f, self.func.root, 0, self.registry)
    }
}

impl IrFunc {
    /// Display this IrFunc using an intrinsic registry for name resolution.
    pub fn display_with_registry<'a>(
        &'a self,
        registry: &'a IntrinsicRegistry,
    ) -> IrFuncDisplay<'a> {
        IrFuncDisplay {
            func: self,
            registry: Some(registry),
        }
    }

    fn fmt_node(
        &self,
        f: &mut fmt::Formatter<'_>,
        node_id: NodeId,
        indent: usize,
        registry: Option<&IntrinsicRegistry>,
    ) -> fmt::Result {
        let node = &self.nodes[node_id];
        let pad = "  ".repeat(indent);

        match &node.kind {
            NodeKind::Lambda {
                body,
                shape,
                lambda_id,
            } => {
                writeln!(
                    f,
                    "{pad}lambda @{} (shape: {:?}) {{",
                    lambda_id.index(),
                    shape.type_identifier,
                )?;
                self.fmt_region(f, *body, indent + 1, registry)?;
                writeln!(f, "{pad}}}")?;
            }
            NodeKind::Simple(op) => {
                write!(f, "{pad}n{} = ", node_id.index())?;
                self.fmt_op(f, op, registry)?;
                write!(f, " [")?;
                for (i, inp) in node.inputs.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    self.fmt_source(f, &inp.source)?;
                }
                write!(f, "] -> [")?;
                for (i, out) in node.outputs.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    self.fmt_output(f, out)?;
                }
                writeln!(f, "]")?;
            }
            NodeKind::Gamma { regions } => {
                writeln!(f, "{pad}n{} = gamma [", node_id.index())?;
                // Show inputs.
                let inputs_pad = "  ".repeat(indent + 1);
                write!(f, "{inputs_pad}pred: ")?;
                if let Some(inp) = node.inputs.first() {
                    self.fmt_source(f, &inp.source)?;
                }
                writeln!(f)?;
                for (i, inp) in node.inputs.iter().skip(1).enumerate() {
                    write!(f, "{inputs_pad}in{i}: ")?;
                    self.fmt_source(f, &inp.source)?;
                    writeln!(f)?;
                }
                writeln!(f, "{pad}] {{")?;
                for (i, &region) in regions.iter().enumerate() {
                    writeln!(f, "{inputs_pad}branch {i}:")?;
                    self.fmt_region(f, region, indent + 2, registry)?;
                }
                write!(f, "{pad}}} -> [")?;
                for (i, out) in node.outputs.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    self.fmt_output(f, out)?;
                }
                writeln!(f, "]")?;
            }
            NodeKind::Theta { body } => {
                write!(f, "{pad}n{} = theta [", node_id.index())?;
                for (i, inp) in node.inputs.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    self.fmt_source(f, &inp.source)?;
                }
                writeln!(f, "] {{")?;
                self.fmt_region(f, *body, indent + 1, registry)?;
                write!(f, "{pad}}} -> [")?;
                for (i, out) in node.outputs.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    self.fmt_output(f, out)?;
                }
                writeln!(f, "]")?;
            }
            NodeKind::Apply { target } => {
                write!(f, "{pad}n{} = apply @{} [", node_id.index(), target.index())?;
                for (i, inp) in node.inputs.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    self.fmt_source(f, &inp.source)?;
                }
                write!(f, "] -> [")?;
                for (i, out) in node.outputs.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    self.fmt_output(f, out)?;
                }
                writeln!(f, "]")?;
            }
        }
        Ok(())
    }

    fn fmt_region(
        &self,
        f: &mut fmt::Formatter<'_>,
        region_id: RegionId,
        indent: usize,
        registry: Option<&IntrinsicRegistry>,
    ) -> fmt::Result {
        let region = &self.regions[region_id];
        let pad = "  ".repeat(indent);

        writeln!(f, "{pad}region {{")?;
        let inner_pad = "  ".repeat(indent + 1);

        // Args.
        write!(f, "{inner_pad}args: [")?;
        for (i, arg) in region.args.iter().enumerate() {
            if i > 0 {
                write!(f, ", ")?;
            }
            self.fmt_region_arg(f, region_id, i as u16, arg)?;
        }
        writeln!(f, "]")?;

        // Nodes.
        for &node_id in &region.nodes {
            self.fmt_node(f, node_id, indent + 1, registry)?;
        }

        // Results.
        write!(f, "{inner_pad}results: [")?;
        for (i, result) in region.results.iter().enumerate() {
            if i > 0 {
                write!(f, ", ")?;
            }
            self.fmt_source(f, &result.source)?;
        }
        writeln!(f, "]")?;

        writeln!(f, "{pad}}}")?;
        Ok(())
    }

    fn fmt_op(
        &self,
        f: &mut fmt::Formatter<'_>,
        op: &IrOp,
        registry: Option<&IntrinsicRegistry>,
    ) -> fmt::Result {
        match op {
            IrOp::ReadBytes { count } => write!(f, "ReadBytes({count})"),
            IrOp::PeekByte => write!(f, "PeekByte"),
            IrOp::AdvanceCursor { count } => write!(f, "AdvanceCursor({count})"),
            IrOp::AdvanceCursorBy => write!(f, "AdvanceCursorBy"),
            IrOp::BoundsCheck { count } => write!(f, "BoundsCheck({count})"),
            IrOp::SaveCursor => write!(f, "SaveCursor"),
            IrOp::RestoreCursor => write!(f, "RestoreCursor"),
            IrOp::WriteToField { offset, width } => {
                write!(f, "WriteToField(offset={offset}, {width})")
            }
            IrOp::ReadFromField { offset, width } => {
                write!(f, "ReadFromField(offset={offset}, {width})")
            }
            IrOp::SaveOutPtr => write!(f, "SaveOutPtr"),
            IrOp::SetOutPtr => write!(f, "SetOutPtr"),
            IrOp::SlotAddr { slot } => write!(f, "SlotAddr({})", slot.index()),
            IrOp::WriteToSlot { slot } => write!(f, "WriteToSlot({})", slot.index()),
            IrOp::ReadFromSlot { slot } => write!(f, "ReadFromSlot({})", slot.index()),
            IrOp::Const { value } => {
                write!(f, "Const(")?;
                Self::fmt_const(f, *value, registry)?;
                write!(f, ")")
            }
            IrOp::Add => write!(f, "Add"),
            IrOp::Sub => write!(f, "Sub"),
            IrOp::And => write!(f, "And"),
            IrOp::Or => write!(f, "Or"),
            IrOp::Shr => write!(f, "Shr"),
            IrOp::Shl => write!(f, "Shl"),
            IrOp::Xor => write!(f, "Xor"),
            IrOp::CmpNe => write!(f, "CmpNe"),
            IrOp::ZigzagDecode { wide } => write!(f, "ZigzagDecode(wide={wide})"),
            IrOp::SignExtend { from_width } => write!(f, "SignExtend(from={from_width})"),
            IrOp::CallIntrinsic {
                func, field_offset, ..
            } => {
                write!(f, "CallIntrinsic(")?;
                Self::fmt_intrinsic(f, *func, registry)?;
                write!(f, ", field_offset={field_offset})")
            }
            IrOp::CallPure { func, .. } => {
                write!(f, "CallPure(")?;
                Self::fmt_intrinsic(f, *func, registry)?;
                write!(f, ")")
            }
            IrOp::ErrorExit { code } => write!(f, "ErrorExit({code:?})"),
            IrOp::SimdStringScan => write!(f, "SimdStringScan"),
            IrOp::SimdWhitespaceSkip => write!(f, "SimdWhitespaceSkip"),
        }
    }

    fn fmt_intrinsic(
        f: &mut fmt::Formatter<'_>,
        func: IntrinsicFn,
        registry: Option<&IntrinsicRegistry>,
    ) -> fmt::Result {
        if let Some(reg) = registry
            && let Some(name) = reg.name_of(func)
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
        if let Some(reg) = registry
            && let Some(name) = reg.const_name_of(value)
        {
            return write!(f, "@{name}");
        }
        write!(f, "{value:#x}")
    }

    fn fmt_source(&self, f: &mut fmt::Formatter<'_>, source: &PortSource) -> fmt::Result {
        match source {
            PortSource::Node(oref) => {
                let out = &self.nodes[oref.node].outputs[oref.index as usize];
                match out.kind {
                    PortKind::Data => {
                        if let Some(vreg) = out.vreg {
                            write!(f, "v{}", vreg.index())
                        } else {
                            write!(f, "n{}.{}", oref.node.index(), oref.index)
                        }
                    }
                    PortKind::StateCursor => write!(f, "%cs:n{}", oref.node.index()),
                    PortKind::StateOutput => write!(f, "%os:n{}", oref.node.index()),
                }
            }
            PortSource::RegionArg(aref) => {
                let arg = &self.regions[aref.region].args[aref.index as usize];
                match arg.kind {
                    PortKind::Data => write!(f, "arg{}", aref.index),
                    PortKind::StateCursor => write!(f, "%cs:arg"),
                    PortKind::StateOutput => write!(f, "%os:arg"),
                }
            }
        }
    }

    fn fmt_output(&self, f: &mut fmt::Formatter<'_>, out: &OutputPort) -> fmt::Result {
        match out.kind {
            PortKind::Data => {
                if let Some(vreg) = out.vreg {
                    write!(f, "v{}", vreg.index())
                } else {
                    write!(f, "?")
                }
            }
            PortKind::StateCursor => write!(f, "%cs"),
            PortKind::StateOutput => write!(f, "%os"),
        }
    }

    fn fmt_region_arg(
        &self,
        f: &mut fmt::Formatter<'_>,
        _region: RegionId,
        index: u16,
        arg: &RegionArg,
    ) -> fmt::Result {
        match arg.kind {
            PortKind::Data => write!(f, "arg{index}"),
            PortKind::StateCursor => write!(f, "%cs"),
            PortKind::StateOutput => write!(f, "%os"),
        }
    }
}

// ─── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // A dummy shape for testing — we just need any valid &'static Shape.
    fn test_shape() -> &'static facet::Shape {
        <u8 as facet::Facet>::SHAPE
    }

    #[test]
    fn arena_basics() {
        let mut arena: Arena<u32> = Arena::new();
        assert!(arena.is_empty());

        let a = arena.push(10);
        let b = arena.push(20);
        let c = arena.push(30);

        assert_eq!(arena.len(), 3);
        assert_eq!(arena[a], 10);
        assert_eq!(arena[b], 20);
        assert_eq!(arena[c], 30);

        arena[b] = 42;
        assert_eq!(arena[b], 42);
    }

    #[test]
    fn id_type_safety() {
        // Id<Node> and Id<Region> are different types — this is compile-time checked.
        // We just verify the index round-trips.
        let id: NodeId = Id::new(7);
        assert_eq!(id.index(), 7);

        let vreg: VReg = Id::new(3);
        assert_eq!(vreg.index(), 3);
    }

    #[test]
    fn linear_chain_state_threading() {
        // Build: bounds_check(4) -> read_bytes(4) -> write_to_field(0, W4)
        // Verify that each cursor op's input is the previous one's output.
        let mut builder = IrBuilder::new(test_shape());

        let body = builder.func.root_body();
        let initial_cs = PortSource::RegionArg(RegionArgRef {
            region: body,
            index: 0,
        });
        let initial_os = PortSource::RegionArg(RegionArgRef {
            region: body,
            index: 1,
        });

        let (read_node, after_check_cs) = {
            let mut rb = builder.root_region();

            // Check initial state.
            assert_eq!(rb.cursor_state(), initial_cs);
            assert_eq!(rb.output_state(), initial_os);

            // bounds_check(4)
            rb.bounds_check(4);
            let after_check_cs = rb.cursor_state();
            assert_ne!(after_check_cs, initial_cs);
            assert_eq!(rb.output_state(), initial_os); // output unchanged

            // read_bytes(4)
            let data = rb.read_bytes(4);
            let after_read_cs = rb.cursor_state();
            assert_ne!(after_read_cs, after_check_cs);

            // Save the read node ID for inspection after the builder is dropped.
            let read_node = match data {
                PortSource::Node(OutputRef { node, .. }) => node,
                _ => panic!("expected Node source"),
            };

            // write_to_field
            rb.write_to_field(data, 0, Width::W4);
            assert_eq!(rb.cursor_state(), after_read_cs); // cursor unchanged by output op
            assert_ne!(rb.output_state(), initial_os); // output updated

            rb.set_results(&[]);
            (read_node, after_check_cs)
        };

        let func = builder.finish();

        // Verify the read_bytes node's cursor input is the bounds_check output.
        let read_input = &func.nodes[read_node].inputs[0];
        assert_eq!(read_input.kind, PortKind::StateCursor);
        assert_eq!(read_input.source, after_check_cs);

        // 3 nodes in the root region.
        assert_eq!(func.regions[func.root_body()].nodes.len(), 3);
    }

    #[test]
    fn effect_classification() {
        assert_eq!(IrOp::Const { value: 0 }.effect(), Effect::Pure);
        assert_eq!(IrOp::Add.effect(), Effect::Pure);
        assert_eq!(IrOp::ZigzagDecode { wide: false }.effect(), Effect::Pure);
        assert_eq!(
            IrOp::CallPure {
                func: IntrinsicFn(0),
                arg_count: 0
            }
            .effect(),
            Effect::Pure
        );

        assert_eq!(IrOp::ReadBytes { count: 1 }.effect(), Effect::Cursor);
        assert_eq!(IrOp::BoundsCheck { count: 1 }.effect(), Effect::Cursor);
        assert_eq!(IrOp::SaveCursor.effect(), Effect::Cursor);
        assert_eq!(IrOp::PeekByte.effect(), Effect::Cursor);

        assert_eq!(
            IrOp::WriteToField {
                offset: 0,
                width: Width::W4
            }
            .effect(),
            Effect::Output
        );
        assert_eq!(
            IrOp::ReadFromField {
                offset: 0,
                width: Width::W4
            }
            .effect(),
            Effect::Output
        );

        assert_eq!(
            IrOp::CallIntrinsic {
                func: IntrinsicFn(0),
                arg_count: 0,
                has_result: false,
                field_offset: 0,
            }
            .effect(),
            Effect::Barrier
        );
    }

    #[test]
    fn op_metadata_cursor_advance_and_side_effects() {
        assert_eq!(IrOp::ReadBytes { count: 4 }.cursor_advance(), Some(4));
        assert_eq!(IrOp::AdvanceCursor { count: 7 }.cursor_advance(), Some(7));
        assert_eq!(IrOp::PeekByte.cursor_advance(), Some(0));
        assert_eq!(IrOp::SaveCursor.cursor_advance(), Some(0));
        assert_eq!(IrOp::BoundsCheck { count: 32 }.cursor_advance(), Some(0));
        assert_eq!(IrOp::AdvanceCursorBy.cursor_advance(), None);
        assert_eq!(IrOp::SimdWhitespaceSkip.cursor_advance(), None);
        assert_eq!(IrOp::Const { value: 1 }.cursor_advance(), None);

        assert!(!IrOp::Const { value: 1 }.has_side_effects());
        assert!(
            !IrOp::CallPure {
                func: IntrinsicFn(0),
                arg_count: 0
            }
            .has_side_effects()
        );
        assert!(IrOp::ReadBytes { count: 1 }.has_side_effects());
        assert!(
            IrOp::WriteToField {
                offset: 0,
                width: Width::W4
            }
            .has_side_effects()
        );
        assert!(
            IrOp::CallIntrinsic {
                func: IntrinsicFn(0),
                arg_count: 0,
                has_result: false,
                field_offset: 0,
            }
            .has_side_effects()
        );
    }

    #[test]
    fn gamma_two_branches() {
        let mut builder = IrBuilder::new(test_shape());

        {
            let mut rb = builder.root_region();

            // Read a byte, use it as predicate.
            rb.bounds_check(1);
            let tag = rb.read_bytes(1);

            // Gamma: branch on tag.
            // Branch 0: read 4 bytes, write to field 0.
            // Branch 1: read 8 bytes, write to field 0.
            let _results = rb.gamma(tag, &[], 2, |branch_idx, branch| match branch_idx {
                0 => {
                    branch.bounds_check(4);
                    let val = branch.read_bytes(4);
                    branch.write_to_field(val, 0, Width::W4);
                    branch.set_results(&[]);
                }
                1 => {
                    branch.bounds_check(8);
                    let val = branch.read_bytes(8);
                    branch.write_to_field(val, 0, Width::W8);
                    branch.set_results(&[]);
                }
                _ => unreachable!(),
            });

            rb.set_results(&[]);
        }

        let func = builder.finish();

        // Root region: bounds_check + read_bytes + gamma = 3 nodes.
        assert_eq!(func.regions[func.root_body()].nodes.len(), 3);

        // Find the gamma node.
        let gamma_id = func.regions[func.root_body()].nodes[2];
        let gamma = &func.nodes[gamma_id];
        match &gamma.kind {
            NodeKind::Gamma { regions } => {
                assert_eq!(regions.len(), 2);
                // Each branch has 3 nodes: bounds_check, read_bytes, write_to_field.
                assert_eq!(func.regions[regions[0]].nodes.len(), 3);
                assert_eq!(func.regions[regions[1]].nodes.len(), 3);
                // Each branch has 2 results (cursor state + output state).
                assert_eq!(func.regions[regions[0]].results.len(), 2);
                assert_eq!(func.regions[regions[1]].results.len(), 2);
            }
            _ => panic!("expected gamma node"),
        }
    }

    #[test]
    fn theta_counted_loop() {
        let mut builder = IrBuilder::new(test_shape());

        {
            let mut rb = builder.root_region();

            // Read a count.
            rb.bounds_check(1);
            let count = rb.read_bytes(1);

            // Loop `count` times, reading 4 bytes each iteration.
            let _results = rb.theta(&[count], |body| {
                let args = body.region_args(1);
                let counter = args[0];

                // Read 4 bytes and write to field.
                body.bounds_check(4);
                let val = body.read_bytes(4);
                body.write_to_field(val, 0, Width::W4);

                // Decrement counter.
                let one = body.const_val(1);
                let new_counter = body.binop(IrOp::Sub, counter, one);

                // Predicate: continue if counter > 0 (nonzero).
                body.set_results(&[new_counter, new_counter]);
            });

            rb.set_results(&[]);
        }

        let func = builder.finish();

        // Root region: bounds_check + read_bytes + theta = 3 nodes.
        assert_eq!(func.regions[func.root_body()].nodes.len(), 3);

        // Find the theta node.
        let theta_id = func.regions[func.root_body()].nodes[2];
        let theta = &func.nodes[theta_id];
        match &theta.kind {
            NodeKind::Theta { body } => {
                // Body: bounds_check, read_bytes, write_to_field, const, sub = 5 nodes.
                assert_eq!(func.regions[*body].nodes.len(), 5);
                // Results: predicate + loop_var + cursor_state + output_state = 4.
                assert_eq!(func.regions[*body].results.len(), 4);
            }
            _ => panic!("expected theta node"),
        }
    }

    #[test]
    fn display_linear_chain() {
        let mut builder = IrBuilder::new(test_shape());

        {
            let mut rb = builder.root_region();
            rb.bounds_check(4);
            let val = rb.read_bytes(4);
            rb.write_to_field(val, 0, Width::W4);
            rb.set_results(&[]);
        }

        let func = builder.finish();
        let output = format!("{func}");

        // Verify the output contains expected fragments.
        assert!(output.contains("lambda @0"), "missing lambda header");
        assert!(output.contains("BoundsCheck(4)"), "missing BoundsCheck");
        assert!(output.contains("ReadBytes(4)"), "missing ReadBytes");
        assert!(
            output.contains("WriteToField(offset=0, W4)"),
            "missing WriteToField"
        );
        assert!(output.contains("results:"), "missing results");
    }
}
