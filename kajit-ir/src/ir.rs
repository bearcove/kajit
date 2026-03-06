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
    pub const fn new(index: u32) -> Self {
        Self {
            index,
            _phantom: PhantomData,
        }
    }

    /// The raw index into the arena.
    pub const fn index(self) -> usize {
        self.index as usize
    }
}

/// Vec-backed arena with typed indexing via [`Id`].
pub struct Arena<T> {
    items: Vec<T>,
}

impl<T: Clone> Clone for Arena<T> {
    fn clone(&self) -> Self {
        Self {
            items: self.items.clone(),
        }
    }
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

/// A named RVSDG state domain.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StateDomain {
    pub name: String,
}
/// A state-domain identifier used for generic state tokens.
pub type StateDomainId = Id<StateDomain>;

/// The built-in cursor state domain.
pub const CURSOR_STATE_DOMAIN: StateDomainId = StateDomainId::new(0);
/// The built-in output state domain.
pub const OUTPUT_STATE_DOMAIN: StateDomainId = StateDomainId::new(1);
/// The built-in cursor state domain name.
pub const CURSOR_STATE_DOMAIN_NAME: &str = "cursor";
/// The built-in output state domain name.
pub const OUTPUT_STATE_DOMAIN_NAME: &str = "output";

/// A debug scope identifier carried through the RVSDG pipeline.
pub type DebugScopeId = Id<DebugScope>;
pub type DebugValueId = Id<DebugValue>;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DebugScopeKind {
    LambdaBody { lambda_id: LambdaId },
    GammaBranch { branch_index: u16 },
    ThetaBody,
    Synthetic,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DebugScope {
    pub parent: Option<DebugScopeId>,
    pub kind: DebugScopeKind,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DebugValueKind {
    Field { offset: u32 },
    Named,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DebugValue {
    pub name: String,
    pub kind: DebugValueKind,
}

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
    /// Carries a named state token (orders operations in that state domain).
    State(StateDomainId),
}

impl PortKind {
    pub const fn state(domain: StateDomainId) -> Self {
        Self::State(domain)
    }

    pub const fn state_domain(self) -> Option<StateDomainId> {
        match self {
            Self::Data => None,
            Self::State(domain) => Some(domain),
        }
    }

    pub const fn is_state(self) -> bool {
        matches!(self, Self::State(_))
    }
}

/// The built-in cursor state token kind.
pub const CURSOR_STATE_PORT: PortKind = PortKind::State(CURSOR_STATE_DOMAIN);
/// The built-in output state token kind.
pub const OUTPUT_STATE_PORT: PortKind = PortKind::State(OUTPUT_STATE_DOMAIN);

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
    /// Debug-scope provenance for this produced value/token.
    pub debug_scope: DebugScopeId,
}

// ─── Region ─────────────────────────────────────────────────────────────────

// r[impl ir.rvsdg.regions]

/// An argument entering a region from outside.
#[derive(Debug, Clone)]
pub struct RegionArg {
    pub kind: PortKind,
    pub vreg: Option<VReg>,
    pub debug_value: Option<DebugValueId>,
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
    /// Debug-scope provenance for this structured region body.
    pub debug_scope: DebugScopeId,
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
            .field("debug_scope", &self.debug_scope)
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
    /// Debug-scope provenance for this node.
    pub debug_scope: DebugScopeId,
    /// Semantic debug value provenance for this node's operation, if any.
    pub debug_value: Option<DebugValueId>,
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
            .field("debug_scope", &self.debug_scope)
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
    /// Reads or mutates a named state domain.
    Domain(StateDomainId),
    /// May touch any state (full barrier).
    Barrier,
}

/// The built-in cursor effect kind.
pub const CURSOR_EFFECT: Effect = Effect::Domain(CURSOR_STATE_DOMAIN);
/// The built-in output effect kind.
pub const OUTPUT_EFFECT: Effect = Effect::Domain(OUTPUT_STATE_DOMAIN);

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
            IrOp::ReadBytes { count } => (Effect::Domain(CURSOR_STATE_DOMAIN), Some(*count)),
            IrOp::AdvanceCursor { count } => (Effect::Domain(CURSOR_STATE_DOMAIN), Some(*count)),
            IrOp::PeekByte | IrOp::SaveCursor | IrOp::BoundsCheck { .. } => {
                (Effect::Domain(CURSOR_STATE_DOMAIN), Some(0))
            }
            IrOp::AdvanceCursorBy
            | IrOp::RestoreCursor
            | IrOp::WriteToSlot { .. }
            | IrOp::ReadFromSlot { .. }
            | IrOp::SimdStringScan
            | IrOp::SimdWhitespaceSkip
            | IrOp::ErrorExit { .. } => (Effect::Domain(CURSOR_STATE_DOMAIN), None),

            // Output ops
            IrOp::WriteToField { .. }
            | IrOp::ReadFromField { .. }
            | IrOp::SaveOutPtr
            | IrOp::SetOutPtr => (Effect::Domain(OUTPUT_STATE_DOMAIN), None),

            // Barrier ops
            IrOp::CallIntrinsic { .. } => (Effect::Barrier, None),
        };

        IrOpMetadata {
            effect,
            cursor_advance,
            has_side_effects: !matches!(effect, Effect::Pure),
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
    /// Named state domains used by state ports and effects.
    pub state_domains: Arena<StateDomain>,
    /// The root lambda node.
    pub root: NodeId,
    /// Next VReg to allocate.
    pub vreg_count: u32,
    /// Next stack slot to allocate.
    pub slot_count: u32,
    /// Lambda registry: maps LambdaId to the NodeId of the lambda node.
    pub lambdas: Vec<NodeId>,
    /// All debug scopes.
    pub debug_scopes: Arena<DebugScope>,
    /// Semantic debug values carried through lowering.
    pub debug_values: Arena<DebugValue>,
    /// Root debug scope for the root lambda body.
    pub root_debug_scope: DebugScopeId,
}

impl IrFunc {
    pub fn builtin_state_domains() -> Arena<StateDomain> {
        let mut domains = Arena::new();
        let cursor = domains.push(StateDomain {
            name: CURSOR_STATE_DOMAIN_NAME.to_owned(),
        });
        let output = domains.push(StateDomain {
            name: OUTPUT_STATE_DOMAIN_NAME.to_owned(),
        });
        debug_assert_eq!(cursor, CURSOR_STATE_DOMAIN);
        debug_assert_eq!(output, OUTPUT_STATE_DOMAIN);
        domains
    }

    pub fn add_state_domain(&mut self, name: impl Into<String>) -> StateDomainId {
        self.state_domains.push(StateDomain { name: name.into() })
    }

    pub fn has_state_domain(&self, id: StateDomainId) -> bool {
        id.index() < self.state_domains.len()
    }

    pub fn state_domain_name(&self, id: StateDomainId) -> Option<&str> {
        self.state_domains
            .iter()
            .find_map(|(domain_id, domain)| (domain_id == id).then_some(domain.name.as_str()))
    }

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

    pub fn region_debug_scope(&self, region: RegionId) -> DebugScopeId {
        self.regions[region].debug_scope
    }

    pub fn node_debug_scope(&self, node: NodeId) -> DebugScopeId {
        self.nodes[node].debug_scope
    }

    pub fn output_debug_scope(&self, output: OutputRef) -> DebugScopeId {
        self.nodes[output.node].outputs[output.index as usize].debug_scope
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
            state_domains: IrFunc::builtin_state_domains(),
            root: NodeId::new(0), // placeholder, set below
            vreg_count: 0,
            slot_count: 0,
            lambdas: Vec::new(),
            debug_scopes: Arena::new(),
            debug_values: Arena::new(),
            root_debug_scope: DebugScopeId::new(0), // placeholder, set below
        };

        let lambda_id = LambdaId::new(0);
        let root_debug_scope = func.debug_scopes.push(DebugScope {
            parent: None,
            kind: DebugScopeKind::LambdaBody { lambda_id },
        });
        func.root_debug_scope = root_debug_scope;

        // Create the root lambda's body region.
        // Standard arguments: cursor state, output state.
        let body = func.regions.push(Region {
            debug_scope: root_debug_scope,
            args: vec![
                RegionArg {
                    kind: PortKind::state(CURSOR_STATE_DOMAIN),
                    vreg: None,
                    debug_value: None,
                },
                RegionArg {
                    kind: PortKind::state(OUTPUT_STATE_DOMAIN),
                    vreg: None,
                    debug_value: None,
                },
            ],
            results: Vec::new(),
            nodes: Vec::new(),
        });

        // Pre-allocate lambda ID 0 for the root.
        func.lambdas.push(NodeId::new(0)); // placeholder, patched below

        let root = func.nodes.push(Node {
            region: ROOT_REGION,
            debug_scope: root_debug_scope,
            debug_value: None,
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
        let debug_scope = self.func.debug_scopes.push(DebugScope {
            parent: Some(self.func.root_debug_scope),
            kind: DebugScopeKind::LambdaBody { lambda_id },
        });
        let mut args = Vec::with_capacity(data_arg_count + 2);
        for _ in 0..data_arg_count {
            args.push(RegionArg {
                kind: PortKind::Data,
                vreg: None,
                debug_value: None,
            });
        }
        args.push(RegionArg {
            kind: PortKind::state(CURSOR_STATE_DOMAIN),
            vreg: None,
            debug_value: None,
        });
        args.push(RegionArg {
            kind: PortKind::state(OUTPUT_STATE_DOMAIN),
            vreg: None,
            debug_value: None,
        });
        let body = self.func.regions.push(Region {
            debug_scope,
            args,
            results: Vec::new(),
            nodes: Vec::new(),
        });
        let node = self.func.nodes.push(Node {
            region: ROOT_REGION,
            debug_scope,
            debug_value: None,
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
        let debug_scope = self.func.regions[body].debug_scope;
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
            debug_scope,
            debug_value: None,
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
    debug_scope: DebugScopeId,
    debug_value: Option<DebugValueId>,
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

    pub fn debug_scope(&self) -> DebugScopeId {
        self.debug_scope
    }

    pub fn debug_value(&self) -> Option<DebugValueId> {
        self.debug_value
    }

    // ── Internal helpers ────────────────────────────────────────────

    fn add_node(&mut self, node: Node) -> NodeId {
        let id = self.func.nodes.push(node);
        self.func.regions[self.region].nodes.push(id);
        id
    }

    fn debug_value_of_source(&self, source: PortSource) -> Option<DebugValueId> {
        match source {
            PortSource::Node(output_ref) => self.func.nodes[output_ref.node].debug_value,
            PortSource::RegionArg(arg_ref) => {
                self.func.regions[arg_ref.region].args[arg_ref.index as usize].debug_value
            }
        }
    }

    pub fn define_debug_field(&mut self, name: impl Into<String>, offset: u32) -> DebugValueId {
        self.func.debug_values.push(DebugValue {
            name: name.into(),
            kind: DebugValueKind::Field { offset },
        })
    }

    pub fn define_debug_value(&mut self, name: impl Into<String>) -> DebugValueId {
        self.func.debug_values.push(DebugValue {
            name: name.into(),
            kind: DebugValueKind::Named,
        })
    }

    pub fn with_debug_value<R>(
        &mut self,
        debug_value: Option<DebugValueId>,
        f: impl FnOnce(&mut RegionBuilder<'_>) -> R,
    ) -> R {
        let previous = self.debug_value;
        self.debug_value = debug_value;
        let result = f(self);
        self.debug_value = previous;
        result
    }

    fn data_output(&mut self) -> OutputPort {
        let vreg = self.func.fresh_vreg();
        OutputPort {
            kind: PortKind::Data,
            vreg: Some(vreg),
            debug_scope: self.debug_scope,
        }
    }

    fn cursor_output(debug_scope: DebugScopeId) -> OutputPort {
        OutputPort {
            kind: PortKind::state(CURSOR_STATE_DOMAIN),
            vreg: None,
            debug_scope,
        }
    }

    fn output_output(debug_scope: DebugScopeId) -> OutputPort {
        OutputPort {
            kind: PortKind::state(OUTPUT_STATE_DOMAIN),
            vreg: None,
            debug_scope,
        }
    }

    // ── Pure operations ─────────────────────────────────────────────

    /// Add a constant value. Returns the data output.
    pub fn const_val(&mut self, value: u64) -> PortSource {
        let out = self.data_output();
        let node = self.add_node(Node {
            region: self.region,
            debug_scope: self.debug_scope,
            debug_value: self.debug_value,
            inputs: vec![],
            outputs: vec![out],
            kind: NodeKind::Simple(IrOp::Const { value }),
        });
        PortSource::Node(OutputRef { node, index: 0 })
    }

    /// Add a binary arithmetic op. Returns the data output.
    pub fn binop(&mut self, op: IrOp, lhs: PortSource, rhs: PortSource) -> PortSource {
        debug_assert!(matches!(op.effect(), Effect::Pure));
        let out = self.data_output();
        let node = self.add_node(Node {
            region: self.region,
            debug_scope: self.debug_scope,
            debug_value: self.debug_value,
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
        debug_assert!(matches!(op.effect(), Effect::Pure));
        let out = self.data_output();
        let node = self.add_node(Node {
            region: self.region,
            debug_scope: self.debug_scope,
            debug_value: self.debug_value,
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
            debug_scope: self.debug_scope,
            debug_value: self.debug_value,
            inputs: vec![InputPort {
                kind: CURSOR_STATE_PORT,
                source: self.cursor_state,
            }],
            outputs: vec![data_out, Self::cursor_output(self.debug_scope)],
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
            debug_scope: self.debug_scope,
            debug_value: self.debug_value,
            inputs: vec![InputPort {
                kind: CURSOR_STATE_PORT,
                source: self.cursor_state,
            }],
            outputs: vec![data_out, Self::cursor_output(self.debug_scope)],
            kind: NodeKind::Simple(IrOp::PeekByte),
        });
        self.cursor_state = PortSource::Node(OutputRef { node, index: 1 });
        PortSource::Node(OutputRef { node, index: 0 })
    }

    /// Advance cursor by N bytes (static count).
    pub fn advance_cursor(&mut self, count: u32) {
        let node = self.add_node(Node {
            region: self.region,
            debug_scope: self.debug_scope,
            debug_value: self.debug_value,
            inputs: vec![InputPort {
                kind: CURSOR_STATE_PORT,
                source: self.cursor_state,
            }],
            outputs: vec![Self::cursor_output(self.debug_scope)],
            kind: NodeKind::Simple(IrOp::AdvanceCursor { count }),
        });
        self.cursor_state = PortSource::Node(OutputRef { node, index: 0 });
    }

    /// Advance cursor by a dynamic amount.
    pub fn advance_cursor_by(&mut self, count: PortSource) {
        let node = self.add_node(Node {
            region: self.region,
            debug_scope: self.debug_scope,
            debug_value: self.debug_value,
            inputs: vec![
                InputPort {
                    kind: PortKind::Data,
                    source: count,
                },
                InputPort {
                    kind: CURSOR_STATE_PORT,
                    source: self.cursor_state,
                },
            ],
            outputs: vec![Self::cursor_output(self.debug_scope)],
            kind: NodeKind::Simple(IrOp::AdvanceCursorBy),
        });
        self.cursor_state = PortSource::Node(OutputRef { node, index: 0 });
    }

    /// Assert N bytes remain. Error exits on failure.
    pub fn bounds_check(&mut self, count: u32) {
        let node = self.add_node(Node {
            region: self.region,
            debug_scope: self.debug_scope,
            debug_value: self.debug_value,
            inputs: vec![InputPort {
                kind: CURSOR_STATE_PORT,
                source: self.cursor_state,
            }],
            outputs: vec![Self::cursor_output(self.debug_scope)],
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
            debug_scope: self.debug_scope,
            debug_value: self.debug_value,
            inputs: vec![InputPort {
                kind: CURSOR_STATE_PORT,
                source: self.cursor_state,
            }],
            outputs: vec![data_out, Self::cursor_output(self.debug_scope)],
            kind: NodeKind::Simple(IrOp::SaveCursor),
        });
        self.cursor_state = PortSource::Node(OutputRef { node, index: 1 });
        PortSource::Node(OutputRef { node, index: 0 })
    }

    /// Restore cursor from a saved position.
    pub fn restore_cursor(&mut self, saved: PortSource) {
        let node = self.add_node(Node {
            region: self.region,
            debug_scope: self.debug_scope,
            debug_value: self.debug_value,
            inputs: vec![
                InputPort {
                    kind: PortKind::Data,
                    source: saved,
                },
                InputPort {
                    kind: CURSOR_STATE_PORT,
                    source: self.cursor_state,
                },
            ],
            outputs: vec![Self::cursor_output(self.debug_scope)],
            kind: NodeKind::Simple(IrOp::RestoreCursor),
        });
        self.cursor_state = PortSource::Node(OutputRef { node, index: 0 });
    }

    // ── Output operations (auto-threaded) ───────────────────────────

    /// Write a value to out+offset.
    pub fn write_to_field(&mut self, src: PortSource, offset: u32, width: Width) {
        let node = self.add_node(Node {
            region: self.region,
            debug_scope: self.debug_scope,
            debug_value: self.debug_value,
            inputs: vec![
                InputPort {
                    kind: PortKind::Data,
                    source: src,
                },
                InputPort {
                    kind: OUTPUT_STATE_PORT,
                    source: self.output_state,
                },
            ],
            outputs: vec![Self::output_output(self.debug_scope)],
            kind: NodeKind::Simple(IrOp::WriteToField { offset, width }),
        });
        self.output_state = PortSource::Node(OutputRef { node, index: 0 });
    }

    /// Read from out+offset. Returns data output.
    pub fn read_from_field(&mut self, offset: u32, width: Width) -> PortSource {
        let data_out = self.data_output();
        let node = self.add_node(Node {
            region: self.region,
            debug_scope: self.debug_scope,
            debug_value: self.debug_value,
            inputs: vec![InputPort {
                kind: OUTPUT_STATE_PORT,
                source: self.output_state,
            }],
            outputs: vec![data_out, Self::output_output(self.debug_scope)],
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
            debug_scope: self.debug_scope,
            debug_value: self.debug_value,
            inputs: vec![InputPort {
                kind: OUTPUT_STATE_PORT,
                source: self.output_state,
            }],
            outputs: vec![data_out, Self::output_output(self.debug_scope)],
            kind: NodeKind::Simple(IrOp::SaveOutPtr),
        });
        self.output_state = PortSource::Node(OutputRef { node, index: 1 });
        PortSource::Node(OutputRef { node, index: 0 })
    }

    /// Set the current output pointer (`out` base).
    pub fn set_out_ptr(&mut self, ptr: PortSource) {
        let node = self.add_node(Node {
            region: self.region,
            debug_scope: self.debug_scope,
            debug_value: self.debug_value,
            inputs: vec![
                InputPort {
                    kind: PortKind::Data,
                    source: ptr,
                },
                InputPort {
                    kind: OUTPUT_STATE_PORT,
                    source: self.output_state,
                },
            ],
            outputs: vec![Self::output_output(self.debug_scope)],
            kind: NodeKind::Simple(IrOp::SetOutPtr),
        });
        self.output_state = PortSource::Node(OutputRef { node, index: 0 });
    }

    /// Compute the address of a stack slot (`sp + slot_offset`).
    pub fn slot_addr(&mut self, slot: SlotId) -> PortSource {
        let data_out = self.data_output();
        let node = self.add_node(Node {
            region: self.region,
            debug_scope: self.debug_scope,
            debug_value: self.debug_value,
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
            debug_scope: self.debug_scope,
            debug_value: self.debug_value,
            inputs: vec![
                InputPort {
                    kind: PortKind::Data,
                    source: src,
                },
                InputPort {
                    kind: CURSOR_STATE_PORT,
                    source: self.cursor_state,
                },
            ],
            outputs: vec![Self::cursor_output(self.debug_scope)],
            kind: NodeKind::Simple(IrOp::WriteToSlot { slot }),
        });
        self.cursor_state = PortSource::Node(OutputRef { node, index: 0 });
    }

    /// Read a value from an abstract stack slot.
    pub fn read_from_slot(&mut self, slot: SlotId) -> PortSource {
        let data_out = self.data_output();
        let node = self.add_node(Node {
            region: self.region,
            debug_scope: self.debug_scope,
            debug_value: self.debug_value,
            inputs: vec![InputPort {
                kind: CURSOR_STATE_PORT,
                source: self.cursor_state,
            }],
            outputs: vec![data_out, Self::cursor_output(self.debug_scope)],
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
            kind: CURSOR_STATE_PORT,
            source: self.cursor_state,
        });
        inputs.push(InputPort {
            kind: OUTPUT_STATE_PORT,
            source: self.output_state,
        });

        let mut outputs = Vec::new();
        if has_result {
            outputs.push(self.data_output());
        }
        let cursor_out_idx = outputs.len() as u16;
        outputs.push(Self::cursor_output(self.debug_scope));
        let output_out_idx = outputs.len() as u16;
        outputs.push(Self::output_output(self.debug_scope));

        let node = self.add_node(Node {
            region: self.region,
            debug_scope: self.debug_scope,
            debug_value: self.debug_value,
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
            debug_scope: self.debug_scope,
            debug_value: self.debug_value,
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
            debug_scope: self.debug_scope,
            debug_value: self.debug_value,
            inputs: vec![InputPort {
                kind: CURSOR_STATE_PORT,
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
        for branch_index in 0..branch_count {
            // Each region gets: passthrough data args + cursor state + output state.
            let mut args = Vec::with_capacity(passthrough.len() + 2);
            for &pt in passthrough {
                args.push(RegionArg {
                    kind: PortKind::Data,
                    vreg: None,
                    debug_value: self.debug_value_of_source(pt),
                });
            }
            args.push(RegionArg {
                kind: CURSOR_STATE_PORT,
                vreg: None,
                debug_value: None,
            });
            args.push(RegionArg {
                kind: OUTPUT_STATE_PORT,
                vreg: None,
                debug_value: None,
            });

            let debug_scope = self.func.debug_scopes.push(DebugScope {
                parent: Some(self.debug_scope),
                kind: DebugScopeKind::GammaBranch {
                    branch_index: branch_index as u16,
                },
            });
            let region = self.func.regions.push(Region {
                debug_scope,
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
            let debug_scope = self.func.regions[region].debug_scope;
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
                debug_scope,
                debug_value: self.debug_value,
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
            kind: CURSOR_STATE_PORT,
            source: self.cursor_state,
        });
        inputs.push(InputPort {
            kind: OUTPUT_STATE_PORT,
            source: self.output_state,
        });

        // Build outputs: data results + state tokens.
        let mut outputs = Vec::with_capacity(data_result_count + 2);
        let mut data_outputs = Vec::with_capacity(data_result_count);
        for _ in 0..data_result_count {
            outputs.push(self.data_output());
        }
        let cursor_out_idx = outputs.len() as u16;
        outputs.push(Self::cursor_output(self.debug_scope));
        let output_out_idx = outputs.len() as u16;
        outputs.push(Self::output_output(self.debug_scope));

        let node = self.add_node(Node {
            region: self.region,
            debug_scope: self.debug_scope,
            debug_value: self.debug_value,
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
        for &loop_var in loop_vars {
            args.push(RegionArg {
                kind: PortKind::Data,
                vreg: None,
                debug_value: self.debug_value_of_source(loop_var),
            });
        }
        args.push(RegionArg {
            kind: CURSOR_STATE_PORT,
            vreg: None,
            debug_value: None,
        });
        args.push(RegionArg {
            kind: OUTPUT_STATE_PORT,
            vreg: None,
            debug_value: None,
        });

        let body_debug_scope = self.func.debug_scopes.push(DebugScope {
            parent: Some(self.debug_scope),
            kind: DebugScopeKind::ThetaBody,
        });
        let body = self.func.regions.push(Region {
            debug_scope: body_debug_scope,
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
            debug_scope: body_debug_scope,
            debug_value: self.debug_value,
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
            kind: CURSOR_STATE_PORT,
            source: self.cursor_state,
        });
        inputs.push(InputPort {
            kind: OUTPUT_STATE_PORT,
            source: self.output_state,
        });

        // Outputs: [loop_vars..., cursor_state, output_state].
        let mut outputs = Vec::with_capacity(loop_var_count + 2);
        let mut data_outputs = Vec::new();
        for _ in 0..loop_var_count {
            outputs.push(self.data_output());
        }
        let cursor_out_idx = outputs.len() as u16;
        outputs.push(Self::cursor_output(self.debug_scope));
        let output_out_idx = outputs.len() as u16;
        outputs.push(Self::output_output(self.debug_scope));

        let node = self.add_node(Node {
            region: self.region,
            debug_scope: self.debug_scope,
            debug_value: self.debug_value,
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
            kind: CURSOR_STATE_PORT,
            source: self.cursor_state,
        });
        inputs.push(InputPort {
            kind: OUTPUT_STATE_PORT,
            source: self.output_state,
        });

        let mut outputs = Vec::with_capacity(result_count + 2);
        for _ in 0..result_count {
            outputs.push(self.data_output());
        }
        let cursor_out_idx = outputs.len() as u16;
        outputs.push(Self::cursor_output(self.debug_scope));
        let output_out_idx = outputs.len() as u16;
        outputs.push(Self::output_output(self.debug_scope));

        let node = self.add_node(Node {
            region: self.region,
            debug_scope: self.debug_scope,
            debug_value: self.debug_value,
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
            kind: CURSOR_STATE_PORT,
            source: self.cursor_state,
        });
        region.results.push(RegionResult {
            kind: OUTPUT_STATE_PORT,
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
        self.func.fmt_state_domains(f)?;
        self.func.fmt_debug_scopes(f)?;
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
                write!(f, "{pad}lambda @{} ", lambda_id.index())?;
                self.fmt_scope_ref(f, node.debug_scope)?;
                writeln!(f, " (shape: {:?}) {{", shape.type_identifier,)?;
                self.fmt_region(f, *body, indent + 1, registry)?;
                writeln!(f, "{pad}}}")?;
            }
            NodeKind::Simple(op) => {
                write!(f, "{pad}n{}", node_id.index())?;
                if node.debug_scope != self.regions[node.region].debug_scope {
                    write!(f, " ")?;
                    self.fmt_scope_ref(f, node.debug_scope)?;
                }
                write!(f, " = ")?;
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
                    self.fmt_output(f, out, node.debug_scope)?;
                }
                writeln!(f, "]")?;
            }
            NodeKind::Gamma { regions } => {
                write!(f, "{pad}n{}", node_id.index())?;
                if node.debug_scope != self.regions[node.region].debug_scope {
                    write!(f, " ")?;
                    self.fmt_scope_ref(f, node.debug_scope)?;
                }
                write!(f, " = ")?;
                writeln!(f, "gamma [")?;
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
                    self.fmt_output(f, out, node.debug_scope)?;
                }
                writeln!(f, "]")?;
            }
            NodeKind::Theta { body } => {
                write!(f, "{pad}n{}", node_id.index())?;
                if node.debug_scope != self.regions[node.region].debug_scope {
                    write!(f, " ")?;
                    self.fmt_scope_ref(f, node.debug_scope)?;
                }
                write!(f, " = theta [")?;
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
                    self.fmt_output(f, out, node.debug_scope)?;
                }
                writeln!(f, "]")?;
            }
            NodeKind::Apply { target } => {
                write!(f, "{pad}n{}", node_id.index())?;
                if node.debug_scope != self.regions[node.region].debug_scope {
                    write!(f, " ")?;
                    self.fmt_scope_ref(f, node.debug_scope)?;
                }
                write!(f, " = ")?;
                write!(f, "apply @{} [", target.index())?;
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
                    self.fmt_output(f, out, node.debug_scope)?;
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

        write!(f, "{pad}region ")?;
        self.fmt_scope_ref(f, region.debug_scope)?;
        writeln!(f, " {{")?;
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
                    PortKind::State(domain) if domain == CURSOR_STATE_DOMAIN => {
                        write!(f, "%cs:n{}", oref.node.index())
                    }
                    PortKind::State(domain) if domain == OUTPUT_STATE_DOMAIN => {
                        write!(f, "%os:n{}", oref.node.index())
                    }
                    PortKind::State(domain) => {
                        write!(f, "%s{}:n{}", domain.index(), oref.node.index())
                    }
                }
            }
            PortSource::RegionArg(aref) => {
                let arg = &self.regions[aref.region].args[aref.index as usize];
                match arg.kind {
                    PortKind::Data => write!(f, "arg{}", aref.index),
                    PortKind::State(domain) if domain == CURSOR_STATE_DOMAIN => {
                        write!(f, "%cs:arg")
                    }
                    PortKind::State(domain) if domain == OUTPUT_STATE_DOMAIN => {
                        write!(f, "%os:arg")
                    }
                    PortKind::State(domain) => write!(f, "%s{}:arg", domain.index()),
                }
            }
        }
    }

    fn fmt_output(
        &self,
        f: &mut fmt::Formatter<'_>,
        out: &OutputPort,
        default_scope: DebugScopeId,
    ) -> fmt::Result {
        match out.kind {
            PortKind::Data => {
                if let Some(vreg) = out.vreg {
                    write!(f, "v{}", vreg.index())?;
                } else {
                    write!(f, "?")?;
                }
            }
            PortKind::State(domain) if domain == CURSOR_STATE_DOMAIN => write!(f, "%cs")?,
            PortKind::State(domain) if domain == OUTPUT_STATE_DOMAIN => write!(f, "%os")?,
            PortKind::State(domain) => write!(f, "%s{}", domain.index())?,
        }

        if out.debug_scope != default_scope {
            self.fmt_scope_ref(f, out.debug_scope)?;
        }
        Ok(())
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
            PortKind::State(domain) if domain == CURSOR_STATE_DOMAIN => write!(f, "%cs"),
            PortKind::State(domain) if domain == OUTPUT_STATE_DOMAIN => write!(f, "%os"),
            PortKind::State(domain) => write!(f, "%s{}", domain.index()),
        }
    }

    fn fmt_state_domains(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "state_domains {{")?;
        for (domain_id, domain) in self.state_domains.iter() {
            writeln!(f, "  d{} = {}", domain_id.index(), domain.name)?;
        }
        writeln!(f, "}}")?;
        Ok(())
    }

    fn fmt_debug_scopes(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "scopes {{")?;
        for (scope_id, scope) in self.debug_scopes.iter() {
            write!(f, "  s{} = ", scope_id.index())?;
            self.fmt_debug_scope_kind(f, &scope.kind)?;
            if let Some(parent) = scope.parent {
                write!(f, " parent s{}", parent.index())?;
            }
            writeln!(f)?;
        }
        writeln!(f, "}}")?;
        Ok(())
    }

    fn fmt_debug_scope_kind(
        &self,
        f: &mut fmt::Formatter<'_>,
        kind: &DebugScopeKind,
    ) -> fmt::Result {
        match kind {
            DebugScopeKind::LambdaBody { lambda_id } => {
                write!(f, "lambda_body(@{})", lambda_id.index())
            }
            DebugScopeKind::GammaBranch { branch_index } => {
                write!(f, "gamma_branch({branch_index})")
            }
            DebugScopeKind::ThetaBody => write!(f, "theta_body"),
            DebugScopeKind::Synthetic => write!(f, "synthetic"),
        }
    }

    fn fmt_scope_ref(&self, f: &mut fmt::Formatter<'_>, scope: DebugScopeId) -> fmt::Result {
        write!(f, "@s{}", scope.index())
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
        assert_eq!(read_input.kind, CURSOR_STATE_PORT);
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

        assert_eq!(IrOp::ReadBytes { count: 1 }.effect(), CURSOR_EFFECT);
        assert_eq!(IrOp::BoundsCheck { count: 1 }.effect(), CURSOR_EFFECT);
        assert_eq!(IrOp::SaveCursor.effect(), CURSOR_EFFECT);
        assert_eq!(IrOp::PeekByte.effect(), CURSOR_EFFECT);

        assert_eq!(
            IrOp::WriteToField {
                offset: 0,
                width: Width::W4
            }
            .effect(),
            OUTPUT_EFFECT
        );
        assert_eq!(
            IrOp::ReadFromField {
                offset: 0,
                width: Width::W4
            }
            .effect(),
            OUTPUT_EFFECT
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
    fn debug_scopes_track_structured_region_nesting() {
        let mut builder = IrBuilder::new(test_shape());

        {
            let mut rb = builder.root_region();
            let predicate = rb.const_val(1);
            let parent_scope = rb.debug_scope();
            let _ = rb.gamma(predicate, &[], 2, |branch_idx, branch| {
                match branch.func.debug_scopes[branch.debug_scope()].kind {
                    DebugScopeKind::GammaBranch { branch_index } => {
                        assert_eq!(branch_index as usize, branch_idx);
                    }
                    other => panic!("expected gamma branch scope, got {other:?}"),
                }
                assert_eq!(
                    branch.func.debug_scopes[branch.debug_scope()].parent,
                    Some(parent_scope)
                );
                branch.set_results(&[]);
            });
            let _ = rb.theta(&[], |body| {
                assert_eq!(
                    body.func.debug_scopes[body.debug_scope()].kind,
                    DebugScopeKind::ThetaBody
                );
                assert_eq!(
                    body.func.debug_scopes[body.debug_scope()].parent,
                    Some(parent_scope)
                );
                let predicate = body.const_val(0);
                body.set_results(&[predicate]);
            });
            rb.set_results(&[]);
        }

        let func = builder.finish();
        assert_eq!(
            func.debug_scopes[func.root_debug_scope].kind,
            DebugScopeKind::LambdaBody {
                lambda_id: LambdaId::new(0)
            }
        );
    }

    #[test]
    fn debug_scope_provenance_is_stored_on_nodes_and_outputs() {
        let mut builder = IrBuilder::new(test_shape());

        let (const_node, output_ref, region_scope) = {
            let mut rb = builder.root_region();
            let src = rb.const_val(7);
            rb.set_results(&[src]);
            let output_ref = match src {
                PortSource::Node(output_ref) => output_ref,
                other => panic!("expected node source, got {other:?}"),
            };
            (output_ref.node, output_ref, rb.debug_scope())
        };

        let func = builder.finish();
        assert_eq!(func.node_debug_scope(const_node), region_scope);
        assert_eq!(func.output_debug_scope(output_ref), region_scope);
        assert_eq!(func.region_debug_scope(func.root_body()), region_scope);
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
