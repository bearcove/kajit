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
pub type ArgId = Id<RegionArg>;
pub type ResultId = Id<RegionResult>;

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

/// The built-in output state domain.
pub const OUTPUT_STATE_DOMAIN: StateDomainId = StateDomainId::new(0);
/// The built-in computed-address memory state domain.
pub const MEMORY_STATE_DOMAIN: StateDomainId = StateDomainId::new(1);
/// The built-in output state domain name.
pub const OUTPUT_STATE_DOMAIN_NAME: &str = "output";
/// The built-in computed-address memory state domain name.
pub const MEMORY_STATE_DOMAIN_NAME: &str = "memory";
/// Byte stride between adjacent abstract slot addresses.
pub const SLOT_ADDR_STRIDE_BYTES: usize = 8;

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
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
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
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
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

    /// Look up the 64-bit address for a function name.
    pub fn address_of_name(&self, name: &str) -> Option<u64> {
        self.func_by_name(name).map(|f| f.0 as u64)
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

/// The built-in output state token kind.
pub const OUTPUT_STATE_PORT: PortKind = PortKind::State(OUTPUT_STATE_DOMAIN);
/// The built-in computed-address memory state token kind.
pub const MEMORY_STATE_PORT: PortKind = PortKind::State(MEMORY_STATE_DOMAIN);

// r[impl ir.edges.data]
// r[impl ir.edges.state]

/// A reference to a node's output port.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct OutputRef {
    pub node: NodeId,
    pub index: u16,
}

/// A reference to a region argument.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct RegionArgRef {
    pub region: RegionId,
    pub arg: ArgId,
}

/// Where an input port gets its value.
///
/// Every input port has exactly one source (RVSDG invariant). Edges are
/// implicit — stored inline in the input port rather than as separate objects.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
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
#[derive(Clone)]
pub struct Region {
    /// Debug-scope provenance for this structured region body.
    pub debug_scope: DebugScopeId,
    /// Arguments entering this region (correspond to outer input ports).
    /// Stored as stable IDs; the data lives in `IrFunc.region_args`.
    pub args: Vec<ArgId>,
    /// Results leaving this region (correspond to outer output ports).
    /// Stored as stable IDs; the data lives in `IrFunc.region_results`.
    pub results: Vec<ResultId>,
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
#[derive(Clone)]
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

/// What kind of use a consumer makes of a node output.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OutputUseKind {
    /// Used as a gamma node's predicate (input 0).
    GammaPredicate,
    /// Used as a normal data input to a node.
    DataInput,
    /// Used as a region result (value leaving the region).
    RegionExit,
}

/// Where a node output is consumed.
#[derive(Debug, Clone)]
pub enum ConsumerKind {
    /// Consumed by another node's input port.
    NodeInput { node: NodeId, input_index: u16 },
    /// Consumed by a region result (exits the region).
    RegionResult { result: ResultId },
}

/// A consumer of a node output.
#[derive(Debug, Clone)]
pub struct OutputConsumer {
    pub kind: ConsumerKind,
    pub use_kind: OutputUseKind,
}

/// The type-specific payload of a node.
#[derive(Debug, Clone)]
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
        /// Optional upper bound on iteration count. If set, the loop
        /// executes at most this many times. Used by the unrolling pass
        /// to convert bounded thetas into gamma cascades.
        max_iterations: Option<u32>,
    },

    /// A function definition containing a single body region.
    Lambda {
        body: RegionId,
        /// Debug label for the type this lambda decodes (e.g. "MyStruct").
        label: String,
        /// Minimum output buffer size in bytes for this function.
        output_size: usize,
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
    /// `num_slots` is the number of consecutive 8-byte slots starting at `slot`
    /// that the pointer may access. Used by slot_to_reg to prevent promotion
    /// of address-taken struct fields.
    /// Inputs: []. Outputs: [Data].
    SlotAddr { slot: SlotId, num_slots: u32 },

    /// Store a scalar value to a computed address.
    /// Inputs: [Data (addr), Data (value), State(memory)]. Outputs: [State(memory)].
    StoreToAddr { width: Width },

    /// Load a scalar value from a computed address.
    /// Inputs: [Data (addr), State(memory)]. Outputs: [Data, State(memory)].
    LoadFromAddr { width: Width },

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

    /// Load the runtime address of an embedded data blob.
    /// The blob is stored in IrFunc::data_blobs[blob_id].
    /// Inputs: []. Outputs: [Data].
    DataAddr { blob_id: u32 },

    /// Binary addition.
    /// Inputs: [Data, Data]. Outputs: [Data].
    Add,

    /// Binary subtraction.
    /// Inputs: [Data, Data]. Outputs: [Data].
    Sub,

    /// Binary multiplication.
    /// Inputs: [Data, Data]. Outputs: [Data].
    Mul,

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

    /// Arithmetic right shift (sign-extending).
    /// Inputs: [Data, Data]. Outputs: [Data].
    Sar,

    /// Bitwise XOR.
    /// Inputs: [Data, Data]. Outputs: [Data].
    Xor,

    /// Compare equal. Returns 1 if lhs == rhs, else 0.
    /// Inputs: [Data, Data]. Outputs: [Data].
    CmpEq,

    /// Compare not-equal. Returns 1 if lhs != rhs, else 0.
    /// Inputs: [Data, Data]. Outputs: [Data].
    CmpNe,

    /// Compare less-than. Returns 1 if lhs < rhs, else 0.
    /// Inputs: [Data, Data]. Outputs: [Data].
    CmpLt,

    /// Compare less-than-or-equal. Returns 1 if lhs <= rhs, else 0.
    /// Inputs: [Data, Data]. Outputs: [Data].
    CmpLe,

    /// Compare greater-than. Returns 1 if lhs > rhs, else 0.
    /// Inputs: [Data, Data]. Outputs: [Data].
    CmpGt,

    /// Compare greater-than-or-equal. Returns 1 if lhs >= rhs, else 0.
    /// Inputs: [Data, Data]. Outputs: [Data].
    CmpGe,

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

    /// Call an extern "C" function with side effects but no runtime context.
    /// Same ABI as CallPure (direct args in registers, no DeserContext).
    /// Threaded on MEMORY state domain to prevent reordering/DCE/CSE.
    /// Inputs: [Data * arg_count, StateMemory]. Outputs: [Data, StateMemory].
    CallEffect { func: IntrinsicFn, arg_count: u8 },

    // ── Error ops ───────────────────────────────────────────────────
    // r[impl ir.ops.error]
    /// Set error code and abort the containing region.
    /// Inputs: [StateCursor]. Outputs: [].
    ErrorExit { code: ErrorCode },

    /// No-op placeholder for removed nodes (used during optimization passes).
    /// Inputs: []. Outputs: [].
    Nop,

    /// Identity copy — passes input through unchanged.
    /// Used by slot2reg to break self-referential theta loop vars.
    /// Inputs: [Data]. Outputs: [Data].
    Identity,
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

/// The built-in output effect kind.
pub const OUTPUT_EFFECT: Effect = Effect::Domain(OUTPUT_STATE_DOMAIN);
/// The built-in computed-address memory effect kind.
pub const MEMORY_EFFECT: Effect = Effect::Domain(MEMORY_STATE_DOMAIN);

/// Per-op metadata used by optimization passes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct IrOpMetadata {
    /// Coarse effect class for scheduling/analysis.
    pub effect: Effect,
    /// Whether the op has any observable side effects and must not be removed
    /// solely based on dead outputs.
    pub has_side_effects: bool,
}

// r[impl ir.effects.independence]
impl IrOp {
    /// Returns pass metadata for this op.
    pub fn metadata(&self) -> IrOpMetadata {
        let effect = match self {
            // Pure ops
            IrOp::Const { .. }
            | IrOp::DataAddr { .. }
            | IrOp::Add
            | IrOp::Sub
            | IrOp::Mul
            | IrOp::And
            | IrOp::Or
            | IrOp::Shr
            | IrOp::Shl
            | IrOp::Sar
            | IrOp::Xor
            | IrOp::CmpEq
            | IrOp::CmpNe
            | IrOp::CmpLt
            | IrOp::CmpLe
            | IrOp::CmpGt
            | IrOp::CmpGe
            | IrOp::ZigzagDecode { .. }
            | IrOp::SignExtend { .. }
            | IrOp::SlotAddr { .. }
            | IrOp::CallPure { .. }
            | IrOp::Nop
            | IrOp::Identity => Effect::Pure,

            // Memory ops (slots, computed addresses, calls with effects)
            IrOp::CallEffect { .. }
            | IrOp::WriteToSlot { .. }
            | IrOp::ReadFromSlot { .. }
            | IrOp::ErrorExit { .. }
            | IrOp::StoreToAddr { .. }
            | IrOp::LoadFromAddr { .. } => Effect::Domain(MEMORY_STATE_DOMAIN),

            // Output ops
            IrOp::WriteToField { .. }
            | IrOp::ReadFromField { .. }
            | IrOp::SaveOutPtr
            | IrOp::SetOutPtr => Effect::Domain(OUTPUT_STATE_DOMAIN),

            // Barrier ops
            IrOp::CallIntrinsic { .. } => Effect::Barrier,
        };

        IrOpMetadata {
            effect,
            has_side_effects: !matches!(effect, Effect::Pure),
        }
    }

    /// Returns the effect classification of this op.
    pub fn effect(&self) -> Effect {
        self.metadata().effect
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
#[derive(Clone)]
pub struct IrFunc {
    /// All nodes.
    pub nodes: Arena<Node>,
    /// All regions.
    pub regions: Arena<Region>,
    /// All region arguments (referenced by ArgId).
    pub region_args: Arena<RegionArg>,
    /// All region results (referenced by ResultId).
    pub region_results: Arena<RegionResult>,
    /// Named state domains used by state ports and effects.
    pub state_domains: Arena<StateDomain>,
    /// The root lambda node.
    pub root: NodeId,
    /// Next VReg to allocate.
    pub vreg_count: u32,
    /// Next stack slot to allocate.
    pub slot_count: u32,
    /// Number of data args passed in calling-convention registers.
    pub param_slot_count: u32,
    /// True for scalar functions (plain calling convention, no decoder ABI).
    pub is_scalar: bool,
    /// Slots that are part of a multi-slot group (struct sub-fields).
    pub multi_slot_group: std::collections::BTreeSet<SlotId>,
    /// Slots that are scalar temporaries from HIR locals (not struct sub-fields,
    /// not control-flow slots, not cursor shadow slots). Safe for dead-port
    /// elimination when other conditions are met.
    pub scalar_temp_slots: std::collections::BTreeSet<SlotId>,
    /// For each theta: maps theta port index to the slot it was promoted from.
    /// Populated by slot_to_reg, consumed by dead_theta_ports.
    pub theta_port_slots: std::collections::HashMap<NodeId, Vec<SlotId>>,
    /// For each theta: set of slots that are pure zero-init temps (written
    /// with Const(0) first, never written with non-zero, not read at gamma
    /// level). Populated by slot_to_reg, consumed by dead_theta_ports.
    pub theta_reinit_slots: std::collections::HashMap<NodeId, std::collections::BTreeSet<SlotId>>,
    /// Lambda registry: maps LambdaId to the NodeId of the lambda node.
    pub lambdas: Vec<NodeId>,
    /// Embedded constant data blobs (string literals, etc.).
    /// Indexed by blob_id in IrOp::DataAddr.
    pub data_blobs: Vec<Vec<u8>>,
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
        let output = domains.push(StateDomain {
            name: OUTPUT_STATE_DOMAIN_NAME.to_owned(),
        });
        debug_assert_eq!(output, OUTPUT_STATE_DOMAIN);
        let memory = domains.push(StateDomain {
            name: MEMORY_STATE_DOMAIN_NAME.to_owned(),
        });
        debug_assert_eq!(memory, MEMORY_STATE_DOMAIN);
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

    /// Find all consumers of a specific node output within the node's parent region.
    ///
    /// Returns a list of `OutputConsumer` entries describing each use.
    pub fn find_output_consumers(&self, node_id: NodeId, output_index: u16) -> Vec<OutputConsumer> {
        let target = PortSource::Node(OutputRef {
            node: node_id,
            index: output_index,
        });
        let parent_region = self.nodes[node_id].region;
        let region = &self.regions[parent_region];
        let mut consumers = Vec::new();

        // Check node inputs in the same region
        for &nid in &region.nodes {
            let consumer_node = &self.nodes[nid];
            for (input_idx, input) in consumer_node.inputs.iter().enumerate() {
                if input.source == target {
                    let use_kind = match &consumer_node.kind {
                        // Input 0 of a gamma is always the predicate
                        NodeKind::Gamma { .. } if input_idx == 0 => OutputUseKind::GammaPredicate,
                        _ => OutputUseKind::DataInput,
                    };
                    consumers.push(OutputConsumer {
                        kind: ConsumerKind::NodeInput {
                            node: nid,
                            input_index: input_idx as u16,
                        },
                        use_kind,
                    });
                }
            }
        }

        // Check region results (value leaving the region)
        for &rid in &region.results {
            let result = &self.region_results[rid];
            if result.source == target {
                consumers.push(OutputConsumer {
                    kind: ConsumerKind::RegionResult { result: rid },
                    use_kind: OutputUseKind::RegionExit,
                });
            }
        }

        consumers
    }

    /// Check if a gamma node's data output is "chain-control-only":
    /// a boolean flag that is {passthrough, const} or {passthrough, another_gamma_output}
    /// where the other gamma output is itself chain-control-only.
    ///
    /// These outputs encode control decisions (is_more, still_running) as data values.
    /// They can be excluded from passthrough-exit landing tuples.
    ///
    /// `depth` limits recursion (transitive check through tail gamma chains).
    pub fn is_chain_control_only_output(
        &self,
        node_id: NodeId,
        output_index: usize,
        depth: usize,
    ) -> bool {
        if depth > 16 {
            return false;
        }
        let node = &self.nodes[node_id];
        let NodeKind::Gamma { regions } = &node.kind else {
            return false;
        };
        if regions.len() != 2 {
            return false;
        }

        // For each branch, check what the result source is
        for &region_id in regions.iter() {
            let region = &self.regions[region_id];
            let Some(&result_id) = region.results.get(output_index) else {
                return false;
            };
            let result = &self.region_results[result_id];
            if result.kind != PortKind::Data {
                return false;
            }
            match result.source {
                // Passthrough of the corresponding region arg — OK
                PortSource::RegionArg(arg_ref) => {
                    if region.args.get(output_index) != Some(&arg_ref.arg) {
                        return false;
                    }
                }
                // Constant — OK (the control flag value)
                PortSource::Node(out_ref) => {
                    let src_node = &self.nodes[out_ref.node];
                    match &src_node.kind {
                        NodeKind::Simple(IrOp::Const { .. }) => {
                            // Direct const — this branch produces a control constant
                        }
                        NodeKind::Gamma { .. } => {
                            // Another gamma's output — transitively check it
                            if !self.is_chain_control_only_output(
                                out_ref.node,
                                out_ref.index as usize,
                                depth + 1,
                            ) {
                                return false;
                            }
                        }
                        _ => return false,
                    }
                }
            }
        }

        // Both branches are either passthrough or const/transitive-control
        // At least one must be passthrough (otherwise it's just two constants)
        let r0 = &self.regions[regions[0]];
        let r1 = &self.regions[regions[1]];
        let r0_pt = r0.results.get(output_index).is_some_and(|&rid| {
            matches!(self.region_results[rid].source,
                PortSource::RegionArg(a) if r0.args.get(output_index) == Some(&a.arg))
        });
        let r1_pt = r1.results.get(output_index).is_some_and(|&rid| {
            matches!(self.region_results[rid].source,
                PortSource::RegionArg(a) if r1.args.get(output_index) == Some(&a.arg))
        });
        r0_pt || r1_pt
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

    /// Find the node that owns a given region (gamma branch, theta body, or lambda body).
    /// Returns None if the region is the root lambda body or is orphaned.
    pub fn find_region_owner(&self, region: RegionId) -> Option<NodeId> {
        self.nodes
            .iter()
            .find(|(_, node)| match &node.kind {
                NodeKind::Gamma { regions } => regions.contains(&region),
                NodeKind::Theta { body, .. } => *body == region,
                NodeKind::Lambda { body, .. } => *body == region,
                _ => false,
            })
            .map(|(nid, _)| nid)
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
    /// Create a new builder for an IR function with the given debug label.
    pub fn new(label: impl Into<String>, output_size: usize) -> Self {
        let mut func = IrFunc {
            nodes: Arena::new(),
            regions: Arena::new(),
            region_args: Arena::new(),
            region_results: Arena::new(),
            state_domains: IrFunc::builtin_state_domains(),
            root: NodeId::new(0), // placeholder, set below
            vreg_count: 0,
            slot_count: 0,
            param_slot_count: 0,
            is_scalar: false,
            lambdas: Vec::new(),
            data_blobs: Vec::new(),
            multi_slot_group: std::collections::BTreeSet::new(),
            scalar_temp_slots: std::collections::BTreeSet::new(),
            theta_port_slots: std::collections::HashMap::new(),
            theta_reinit_slots: std::collections::HashMap::new(),
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
        // Standard arguments: all declared state domains.
        let mut root_args = Vec::with_capacity(func.state_domains.len());
        for (domain_id, _) in func.state_domains.iter() {
            let arg_id = func.region_args.push(RegionArg {
                kind: PortKind::state(domain_id),
                vreg: None,
                debug_value: None,
            });
            root_args.push(arg_id);
        }
        let body = func.regions.push(Region {
            debug_scope: root_debug_scope,
            args: root_args,
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
                label: label.into(),
                output_size,
                lambda_id,
            },
        });

        func.lambdas[lambda_id.index()] = root;
        func.root = root;

        IrBuilder { func }
    }

    /// Create a new builder with data args on the root lambda.
    /// Returns the builder plus the `PortSource` for each data arg.
    pub fn new_with_data_args(
        label: impl Into<String>,
        output_size: usize,
        data_arg_count: usize,
    ) -> (Self, Vec<PortSource>) {
        let mut builder = Self::new(label, output_size);
        let body = builder.func.lambda_body(LambdaId::new(0));
        // Insert data args before the existing state domain args.
        let existing_args = builder.func.regions[body].args.clone();
        let mut new_args = Vec::with_capacity(data_arg_count + existing_args.len());
        let mut data_sources = Vec::with_capacity(data_arg_count);
        for _ in 0..data_arg_count {
            let arg_id = builder.func.region_args.push(RegionArg {
                kind: PortKind::Data,
                vreg: None,
                debug_value: None,
            });
            data_sources.push(PortSource::RegionArg(RegionArgRef {
                region: body,
                arg: arg_id,
            }));
            new_args.push(arg_id);
        }
        new_args.extend(existing_args);
        builder.func.regions[body].args = new_args;
        (builder, data_sources)
    }

    /// Get a [`RegionBuilder`] for the root lambda's body.
    pub fn root_region(&mut self) -> RegionBuilder<'_> {
        self.lambda_region(LambdaId::new(0))
    }

    /// Allocate a fresh abstract stack slot.
    pub fn alloc_slot(&mut self) -> SlotId {
        self.func.fresh_slot()
    }

    /// Add a new named state domain before any lambda bodies are populated.
    pub fn add_state_domain(&mut self, name: impl Into<String>) -> StateDomainId {
        let domain_id = self.func.add_state_domain(name);
        let lambda_ids: Vec<_> = (0..self.func.lambdas.len())
            .map(|idx| LambdaId::new(idx as u32))
            .collect();
        for lambda_id in lambda_ids {
            let body = self.func.lambda_body(lambda_id);
            let region = &mut self.func.regions[body];
            assert!(
                region.nodes.is_empty() && region.results.is_empty(),
                "state domains must be declared before populating lambda bodies"
            );
            let arg_id = self.func.region_args.push(RegionArg {
                kind: PortKind::state(domain_id),
                vreg: None,
                debug_value: None,
            });
            region.args.push(arg_id);
        }
        domain_id
    }

    /// Create a new lambda and return its ID.
    pub fn create_lambda(&mut self, label: impl Into<String>, output_size: usize) -> LambdaId {
        self.create_lambda_with_data_args(label, output_size, 0)
    }

    /// Create a new lambda with `data_arg_count` leading data args.
    pub fn create_lambda_with_data_args(
        &mut self,
        label: impl Into<String>,
        output_size: usize,
        data_arg_count: usize,
    ) -> LambdaId {
        let lambda_id = LambdaId::new(self.func.lambdas.len() as u32);
        let debug_scope = self.func.debug_scopes.push(DebugScope {
            parent: Some(self.func.root_debug_scope),
            kind: DebugScopeKind::LambdaBody { lambda_id },
        });
        let mut args = Vec::with_capacity(data_arg_count + self.func.state_domains.len());
        for _ in 0..data_arg_count {
            let arg_id = self.func.region_args.push(RegionArg {
                kind: PortKind::Data,
                vreg: None,
                debug_value: None,
            });
            args.push(arg_id);
        }
        for (domain_id, _) in self.func.state_domains.iter() {
            let arg_id = self.func.region_args.push(RegionArg {
                kind: PortKind::state(domain_id),
                vreg: None,
                debug_value: None,
            });
            args.push(arg_id);
        }
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
                label: label.into(),
                output_size,
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
        let state_count = self.func.state_domains.len();
        assert!(
            arg_count >= state_count,
            "lambda body must have declared state args"
        );
        let data_arg_count = arg_count - state_count;
        let state_sources = self
            .func
            .state_domains
            .iter()
            .enumerate()
            .map(|(i, _)| {
                let arg_id = self.func.regions[body].args[data_arg_count + i];
                PortSource::RegionArg(RegionArgRef {
                    region: body,
                    arg: arg_id,
                })
            })
            .collect();

        RegionBuilder {
            func: &mut self.func,
            region: body,
            debug_scope,
            debug_value: None,
            state_sources,
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
    state_sources: Vec<PortSource>,
}

impl<'a> RegionBuilder<'a> {
    /// The region this builder targets.
    pub fn region(&self) -> RegionId {
        self.region
    }

    /// Current output state token source.
    pub fn output_state(&self) -> PortSource {
        self.state_source(OUTPUT_STATE_DOMAIN)
    }

    /// Current memory state token source.
    pub fn memory_state(&self) -> PortSource {
        self.state_source(MEMORY_STATE_DOMAIN)
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
            PortSource::RegionArg(arg_ref) => self.func.region_args[arg_ref.arg].debug_value,
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

    fn state_source(&self, domain: StateDomainId) -> PortSource {
        self.state_sources[domain.index()]
    }

    fn set_state_source(&mut self, domain: StateDomainId, source: PortSource) {
        self.state_sources[domain.index()] = source;
    }

    fn state_output(domain: StateDomainId, debug_scope: DebugScopeId) -> OutputPort {
        OutputPort {
            kind: PortKind::state(domain),
            vreg: None,
            debug_scope,
        }
    }

    fn state_input(&self, domain: StateDomainId) -> InputPort {
        InputPort {
            kind: PortKind::state(domain),
            source: self.state_source(domain),
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

    /// Register a constant data blob and return its blob_id.
    pub fn add_data_blob(&mut self, bytes: Vec<u8>) -> u32 {
        let id = self.func.data_blobs.len() as u32;
        self.func.data_blobs.push(bytes);
        id
    }

    /// Add a DataAddr node that produces the runtime address of a data blob.
    pub fn data_addr(&mut self, blob_id: u32) -> PortSource {
        let out = self.data_output();
        let node = self.add_node(Node {
            region: self.region,
            debug_scope: self.debug_scope,
            debug_value: self.debug_value,
            inputs: vec![],
            outputs: vec![out],
            kind: NodeKind::Simple(IrOp::DataAddr { blob_id }),
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
                    source: self.state_source(OUTPUT_STATE_DOMAIN),
                },
            ],
            outputs: vec![Self::state_output(OUTPUT_STATE_DOMAIN, self.debug_scope)],
            kind: NodeKind::Simple(IrOp::WriteToField { offset, width }),
        });
        self.set_state_source(
            OUTPUT_STATE_DOMAIN,
            PortSource::Node(OutputRef { node, index: 0 }),
        );
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
                source: self.state_source(OUTPUT_STATE_DOMAIN),
            }],
            outputs: vec![
                data_out,
                Self::state_output(OUTPUT_STATE_DOMAIN, self.debug_scope),
            ],
            kind: NodeKind::Simple(IrOp::ReadFromField { offset, width }),
        });
        self.set_state_source(
            OUTPUT_STATE_DOMAIN,
            PortSource::Node(OutputRef { node, index: 1 }),
        );
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
                source: self.state_source(OUTPUT_STATE_DOMAIN),
            }],
            outputs: vec![
                data_out,
                Self::state_output(OUTPUT_STATE_DOMAIN, self.debug_scope),
            ],
            kind: NodeKind::Simple(IrOp::SaveOutPtr),
        });
        self.set_state_source(
            OUTPUT_STATE_DOMAIN,
            PortSource::Node(OutputRef { node, index: 1 }),
        );
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
                    source: self.state_source(OUTPUT_STATE_DOMAIN),
                },
            ],
            outputs: vec![Self::state_output(OUTPUT_STATE_DOMAIN, self.debug_scope)],
            kind: NodeKind::Simple(IrOp::SetOutPtr),
        });
        self.set_state_source(
            OUTPUT_STATE_DOMAIN,
            PortSource::Node(OutputRef { node, index: 0 }),
        );
    }

    /// Compute the address of a stack slot (`sp + slot_offset`).
    /// `num_slots` is the number of consecutive 8-byte slots that may be accessed
    /// through the returned pointer (used to prevent slot_to_reg promotion).
    pub fn slot_addr(&mut self, slot: SlotId, num_slots: u32) -> PortSource {
        let data_out = self.data_output();
        let node = self.add_node(Node {
            region: self.region,
            debug_scope: self.debug_scope,
            debug_value: self.debug_value,
            inputs: vec![],
            outputs: vec![data_out],
            kind: NodeKind::Simple(IrOp::SlotAddr { slot, num_slots }),
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
                    kind: MEMORY_STATE_PORT,
                    source: self.state_source(MEMORY_STATE_DOMAIN),
                },
            ],
            outputs: vec![Self::state_output(MEMORY_STATE_DOMAIN, self.debug_scope)],
            kind: NodeKind::Simple(IrOp::WriteToSlot { slot }),
        });
        self.set_state_source(
            MEMORY_STATE_DOMAIN,
            PortSource::Node(OutputRef { node, index: 0 }),
        );
    }

    /// Read a value from an abstract stack slot.
    pub fn read_from_slot(&mut self, slot: SlotId) -> PortSource {
        let data_out = self.data_output();
        let node = self.add_node(Node {
            region: self.region,
            debug_scope: self.debug_scope,
            debug_value: self.debug_value,
            inputs: vec![InputPort {
                kind: MEMORY_STATE_PORT,
                source: self.state_source(MEMORY_STATE_DOMAIN),
            }],
            outputs: vec![
                data_out,
                Self::state_output(MEMORY_STATE_DOMAIN, self.debug_scope),
            ],
            kind: NodeKind::Simple(IrOp::ReadFromSlot { slot }),
        });
        self.set_state_source(
            MEMORY_STATE_DOMAIN,
            PortSource::Node(OutputRef { node, index: 1 }),
        );
        PortSource::Node(OutputRef { node, index: 0 })
    }

    /// Store a scalar value to a computed address.
    pub fn store_to_addr(&mut self, addr: PortSource, src: PortSource, width: Width) {
        let node = self.add_node(Node {
            region: self.region,
            debug_scope: self.debug_scope,
            debug_value: self.debug_value,
            inputs: vec![
                InputPort {
                    kind: PortKind::Data,
                    source: addr,
                },
                InputPort {
                    kind: PortKind::Data,
                    source: src,
                },
                InputPort {
                    kind: MEMORY_STATE_PORT,
                    source: self.state_source(MEMORY_STATE_DOMAIN),
                },
            ],
            outputs: vec![Self::state_output(MEMORY_STATE_DOMAIN, self.debug_scope)],
            kind: NodeKind::Simple(IrOp::StoreToAddr { width }),
        });
        self.set_state_source(
            MEMORY_STATE_DOMAIN,
            PortSource::Node(OutputRef { node, index: 0 }),
        );
    }

    /// Load a scalar value from a computed address.
    pub fn load_from_addr(&mut self, addr: PortSource, width: Width) -> PortSource {
        let data_out = self.data_output();
        let node = self.add_node(Node {
            region: self.region,
            debug_scope: self.debug_scope,
            debug_value: self.debug_value,
            inputs: vec![
                InputPort {
                    kind: PortKind::Data,
                    source: addr,
                },
                InputPort {
                    kind: MEMORY_STATE_PORT,
                    source: self.state_source(MEMORY_STATE_DOMAIN),
                },
            ],
            outputs: vec![
                data_out,
                Self::state_output(MEMORY_STATE_DOMAIN, self.debug_scope),
            ],
            kind: NodeKind::Simple(IrOp::LoadFromAddr { width }),
        });
        self.set_state_source(
            MEMORY_STATE_DOMAIN,
            PortSource::Node(OutputRef { node, index: 1 }),
        );
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
        for (domain_id, _) in self.func.state_domains.iter() {
            inputs.push(self.state_input(domain_id));
        }

        let mut outputs = Vec::new();
        if has_result {
            outputs.push(self.data_output());
        }
        let state_output_base = outputs.len() as u16;
        for (domain_id, _) in self.func.state_domains.iter() {
            outputs.push(Self::state_output(domain_id, self.debug_scope));
        }

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

        let state_domain_ids: Vec<_> = self
            .func
            .state_domains
            .iter()
            .map(|(domain_id, _)| domain_id)
            .collect();
        for (offset, domain_id) in state_domain_ids.into_iter().enumerate() {
            self.set_state_source(
                domain_id,
                PortSource::Node(OutputRef {
                    node,
                    index: state_output_base + offset as u16,
                }),
            );
        }

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

    /// Call an effectful function with no runtime context. Same ABI as
    /// `call_pure` (direct args in registers), but threaded on the MEMORY
    /// state domain so calls cannot be reordered, DCE'd, or CSE'd.
    /// Returns the data result.
    pub fn call_effect(&mut self, func: IntrinsicFn, args: &[PortSource]) -> PortSource {
        let mut inputs: Vec<InputPort> = args
            .iter()
            .map(|&src| InputPort {
                kind: PortKind::Data,
                source: src,
            })
            .collect();
        inputs.push(InputPort {
            kind: MEMORY_STATE_PORT,
            source: self.state_source(MEMORY_STATE_DOMAIN),
        });
        let data_out = self.data_output();
        let state_out = Self::state_output(MEMORY_STATE_DOMAIN, self.debug_scope);
        let node = self.add_node(Node {
            region: self.region,
            debug_scope: self.debug_scope,
            debug_value: self.debug_value,
            inputs,
            outputs: vec![data_out, state_out],
            kind: NodeKind::Simple(IrOp::CallEffect {
                func,
                arg_count: args.len() as u8,
            }),
        });
        self.set_state_source(
            MEMORY_STATE_DOMAIN,
            PortSource::Node(OutputRef { node, index: 1 }),
        );
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
                kind: MEMORY_STATE_PORT,
                source: self.state_source(MEMORY_STATE_DOMAIN),
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
            // Each region gets: passthrough data args + all state domain args.
            let mut args = Vec::with_capacity(passthrough.len() + self.func.state_domains.len());
            for &pt in passthrough {
                let arg_id = self.func.region_args.push(RegionArg {
                    kind: PortKind::Data,
                    vreg: None,
                    debug_value: self.debug_value_of_source(pt),
                });
                args.push(arg_id);
            }
            for (domain_id, _) in self.func.state_domains.iter() {
                let arg_id = self.func.region_args.push(RegionArg {
                    kind: PortKind::state(domain_id),
                    vreg: None,
                    debug_value: None,
                });
                args.push(arg_id);
            }

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
            let debug_scope = self.func.regions[region].debug_scope;
            let state_sources = self
                .func
                .state_domains
                .iter()
                .enumerate()
                .map(|(j, _)| {
                    let arg_id = self.func.regions[region].args[passthrough.len() + j];
                    PortSource::RegionArg(RegionArgRef {
                        region,
                        arg: arg_id,
                    })
                })
                .collect();

            let mut branch_builder = RegionBuilder {
                func: self.func,
                region,
                debug_scope,
                debug_value: self.debug_value,
                state_sources,
            };
            build_branch(i, &mut branch_builder);
        }

        // Determine the number of data results from the first branch.
        // All branches must produce the same number of results (RVSDG invariant).
        let state_count = self.func.state_domains.len();
        // Results layout: [data_results..., state domains...].
        let total_results = self.func.regions[region_ids[0]].results.len();
        assert!(
            total_results >= state_count,
            "gamma branch must have state results for all declared domains"
        );
        let data_result_count = total_results - state_count;

        // Build the gamma node's inputs: predicate + passthrough + state tokens.
        let mut inputs = Vec::with_capacity(1 + passthrough.len() + state_count);
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
        for (domain_id, _) in self.func.state_domains.iter() {
            inputs.push(self.state_input(domain_id));
        }

        // Build outputs: data results + state tokens.
        let mut outputs = Vec::with_capacity(data_result_count + state_count);
        let mut data_outputs = Vec::with_capacity(data_result_count);
        for _ in 0..data_result_count {
            outputs.push(self.data_output());
        }
        let state_output_base = outputs.len() as u16;
        for (domain_id, _) in self.func.state_domains.iter() {
            outputs.push(Self::state_output(domain_id, self.debug_scope));
        }

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

        let state_domain_ids: Vec<_> = self
            .func
            .state_domains
            .iter()
            .map(|(domain_id, _)| domain_id)
            .collect();
        for (offset, domain_id) in state_domain_ids.into_iter().enumerate() {
            self.set_state_source(
                domain_id,
                PortSource::Node(OutputRef {
                    node,
                    index: state_output_base + offset as u16,
                }),
            );
        }

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
    ///   The body region receives `[loop_vars..., output_state, memory_state]`
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
        let state_count = self.func.state_domains.len();
        // Create the body region with args: [loop_vars..., state domains...].
        let mut args = Vec::with_capacity(loop_vars.len() + state_count);
        for &loop_var in loop_vars {
            let arg_id = self.func.region_args.push(RegionArg {
                kind: PortKind::Data,
                vreg: None,
                debug_value: self.debug_value_of_source(loop_var),
            });
            args.push(arg_id);
        }
        for (domain_id, _) in self.func.state_domains.iter() {
            let arg_id = self.func.region_args.push(RegionArg {
                kind: PortKind::state(domain_id),
                vreg: None,
                debug_value: None,
            });
            args.push(arg_id);
        }

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
        let state_sources = self
            .func
            .state_domains
            .iter()
            .enumerate()
            .map(|(j, _)| {
                let arg_id = self.func.regions[body].args[loop_vars.len() + j];
                PortSource::RegionArg(RegionArgRef {
                    region: body,
                    arg: arg_id,
                })
            })
            .collect();

        let mut body_builder = RegionBuilder {
            func: self.func,
            region: body,
            debug_scope: body_debug_scope,
            debug_value: self.debug_value,
            state_sources,
        };
        build_body(&mut body_builder);

        // Body results layout: [predicate, loop_vars..., state domains...].
        let total_results = self.func.regions[body].results.len();
        assert!(
            total_results > state_count,
            "theta body must have predicate plus state results for all declared domains"
        );
        let data_result_count = total_results - state_count; // predicate + loop_vars
        let loop_var_count = data_result_count - 1; // minus predicate

        // Build theta node inputs: [loop_vars..., state domains...].
        let mut inputs = Vec::with_capacity(loop_vars.len() + state_count);
        for &lv in loop_vars {
            inputs.push(InputPort {
                kind: PortKind::Data,
                source: lv,
            });
        }
        for (domain_id, _) in self.func.state_domains.iter() {
            inputs.push(self.state_input(domain_id));
        }

        // Outputs: [loop_vars..., state domains...].
        let mut outputs = Vec::with_capacity(loop_var_count + state_count);
        let mut data_outputs = Vec::new();
        for _ in 0..loop_var_count {
            outputs.push(self.data_output());
        }
        let state_output_base = outputs.len() as u16;
        for (domain_id, _) in self.func.state_domains.iter() {
            outputs.push(Self::state_output(domain_id, self.debug_scope));
        }

        let node = self.add_node(Node {
            region: self.region,
            debug_scope: self.debug_scope,
            debug_value: self.debug_value,
            inputs,
            outputs,
            kind: NodeKind::Theta {
                body,
                max_iterations: None,
            },
        });

        let state_domain_ids: Vec<_> = self
            .func
            .state_domains
            .iter()
            .map(|(domain_id, _)| domain_id)
            .collect();
        for (offset, domain_id) in state_domain_ids.into_iter().enumerate() {
            self.set_state_source(
                domain_id,
                PortSource::Node(OutputRef {
                    node,
                    index: state_output_base + offset as u16,
                }),
            );
        }

        for i in 0..loop_var_count {
            data_outputs.push(PortSource::Node(OutputRef {
                node,
                index: i as u16,
            }));
        }
        data_outputs
    }

    /// Create a bounded theta (loop with known max iteration count).
    ///
    /// Same as `theta()` but annotates the node with a maximum iteration count,
    /// enabling the unrolling pass to convert it to straight-line code.
    pub fn theta_bounded(
        &mut self,
        loop_vars: &[PortSource],
        max_iterations: u32,
        build_body: impl FnOnce(&mut RegionBuilder<'_>),
    ) -> Vec<PortSource> {
        let results = self.theta(loop_vars, build_body);
        // Find the theta node we just created (last node in the region with Theta kind)
        let theta_node_id = self.func.regions[self.region]
            .nodes
            .iter()
            .rev()
            .find(|&&nid| matches!(self.func.nodes[nid].kind, NodeKind::Theta { .. }))
            .copied()
            .expect("theta_bounded: theta node not found after theta()");
        if let NodeKind::Theta {
            max_iterations: ref mut mi,
            ..
        } = self.func.nodes[theta_node_id].kind
        {
            *mi = Some(max_iterations);
        }
        results
    }

    /// Add an apply node (lambda call).
    ///
    /// `args`: data arguments to pass to the callee.
    /// `result_count`: number of data results returned by the callee.
    /// All declared state domains are threaded automatically.
    pub fn apply(
        &mut self,
        target: LambdaId,
        args: &[PortSource],
        result_count: usize,
    ) -> Vec<PortSource> {
        let mut inputs = Vec::with_capacity(args.len() + self.func.state_domains.len());
        for &arg in args {
            inputs.push(InputPort {
                kind: PortKind::Data,
                source: arg,
            });
        }
        for (domain_id, _) in self.func.state_domains.iter() {
            inputs.push(self.state_input(domain_id));
        }

        let mut outputs = Vec::with_capacity(result_count + self.func.state_domains.len());
        for _ in 0..result_count {
            outputs.push(self.data_output());
        }
        let state_output_base = outputs.len() as u16;
        for (domain_id, _) in self.func.state_domains.iter() {
            outputs.push(Self::state_output(domain_id, self.debug_scope));
        }

        let node = self.add_node(Node {
            region: self.region,
            debug_scope: self.debug_scope,
            debug_value: self.debug_value,
            inputs,
            outputs,
            kind: NodeKind::Apply { target },
        });

        let state_domain_ids: Vec<_> = self
            .func
            .state_domains
            .iter()
            .map(|(domain_id, _)| domain_id)
            .collect();
        for (offset, domain_id) in state_domain_ids.into_iter().enumerate() {
            self.set_state_source(
                domain_id,
                PortSource::Node(OutputRef {
                    node,
                    index: state_output_base + offset as u16,
                }),
            );
        }

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
    /// State tokens for all declared domains are appended automatically.
    pub fn set_results(&mut self, data_results: &[PortSource]) {
        // Clear existing results
        self.func.regions[self.region].results.clear();

        // Add data results
        for &src in data_results {
            let result_id = self.func.region_results.push(RegionResult {
                kind: PortKind::Data,
                source: src,
            });
            self.func.regions[self.region].results.push(result_id);
        }

        // Add state results for all declared domains
        for (domain_id, _) in self.func.state_domains.iter() {
            let result_id = self.func.region_results.push(RegionResult {
                kind: PortKind::state(domain_id),
                source: self.state_source(domain_id),
            });
            self.func.regions[self.region].results.push(result_id);
        }
    }

    /// Allocate a fresh stack slot.
    pub fn alloc_slot(&mut self) -> SlotId {
        self.func.fresh_slot()
    }

    /// Provide region argument sources for passthrough data values
    /// inside a gamma/theta body. Returns sources for indices 0..count.
    pub fn region_args(&self, count: usize) -> Vec<PortSource> {
        let region = &self.func.regions[self.region];
        (0..count)
            .map(|i| {
                let arg_id = region.args[i];
                PortSource::RegionArg(RegionArgRef {
                    region: self.region,
                    arg: arg_id,
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
                label,
                output_size,
                lambda_id,
            } => {
                write!(f, "{pad}lambda @{} ", lambda_id.index())?;
                self.fmt_scope_ref(f, node.debug_scope)?;
                if *output_size > 0 {
                    writeln!(f, " (shape: {label:?}, output_size: {output_size}) {{")?;
                } else {
                    writeln!(f, " (shape: {label:?}) {{")?;
                }
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
            NodeKind::Theta {
                body,
                max_iterations,
            } => {
                write!(f, "{pad}n{}", node_id.index())?;
                if node.debug_scope != self.regions[node.region].debug_scope {
                    write!(f, " ")?;
                    self.fmt_scope_ref(f, node.debug_scope)?;
                }
                if let Some(max) = max_iterations {
                    write!(f, " = theta(max={max}) [")?;
                } else {
                    write!(f, " = theta [")?;
                }
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
        for (i, &arg_id) in region.args.iter().enumerate() {
            if i > 0 {
                write!(f, ", ")?;
            }
            let arg = &self.region_args[arg_id];
            self.fmt_region_arg(f, region_id, i as u16, arg)?;
        }
        writeln!(f, "]")?;

        // Nodes.
        for &node_id in &region.nodes {
            self.fmt_node(f, node_id, indent + 1, registry)?;
        }

        // Results.
        write!(f, "{inner_pad}results: [")?;
        for (i, &result_id) in region.results.iter().enumerate() {
            if i > 0 {
                write!(f, ", ")?;
            }
            let result = &self.region_results[result_id];
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
            IrOp::WriteToField { offset, width } => {
                write!(f, "WriteToField(offset={offset}, {width})")
            }
            IrOp::ReadFromField { offset, width } => {
                write!(f, "ReadFromField(offset={offset}, {width})")
            }
            IrOp::SaveOutPtr => write!(f, "SaveOutPtr"),
            IrOp::SetOutPtr => write!(f, "SetOutPtr"),
            IrOp::SlotAddr { slot, num_slots } => {
                write!(f, "SlotAddr({}, n={})", slot.index(), num_slots)
            }
            IrOp::StoreToAddr { width } => write!(f, "StoreToAddr({width})"),
            IrOp::LoadFromAddr { width } => write!(f, "LoadFromAddr({width})"),
            IrOp::WriteToSlot { slot } => write!(f, "WriteToSlot({})", slot.index()),
            IrOp::ReadFromSlot { slot } => write!(f, "ReadFromSlot({})", slot.index()),
            IrOp::DataAddr { blob_id } => {
                write!(f, "DataAddr(blob_id={blob_id})")?;
                Ok(())
            }
            IrOp::Const { value } => {
                write!(f, "Const(")?;
                Self::fmt_const(f, *value, registry)?;
                write!(f, ")")
            }
            IrOp::Add => write!(f, "Add"),
            IrOp::Sub => write!(f, "Sub"),
            IrOp::Mul => write!(f, "Mul"),
            IrOp::And => write!(f, "And"),
            IrOp::Or => write!(f, "Or"),
            IrOp::Shr => write!(f, "Shr"),
            IrOp::Shl => write!(f, "Shl"),
            IrOp::Sar => write!(f, "Sar"),
            IrOp::Xor => write!(f, "Xor"),
            IrOp::CmpEq => write!(f, "CmpEq"),
            IrOp::CmpNe => write!(f, "CmpNe"),
            IrOp::CmpLt => write!(f, "CmpLt"),
            IrOp::CmpLe => write!(f, "CmpLe"),
            IrOp::CmpGt => write!(f, "CmpGt"),
            IrOp::CmpGe => write!(f, "CmpGe"),
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
            IrOp::CallEffect { func, .. } => {
                write!(f, "CallEffect(")?;
                Self::fmt_intrinsic(f, *func, registry)?;
                write!(f, ")")
            }
            IrOp::ErrorExit { code } => write!(f, "ErrorExit({code:?})"),
            IrOp::Nop => write!(f, "Nop"),
            IrOp::Identity => write!(f, "Identity"),
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
                    PortKind::State(domain) if domain == OUTPUT_STATE_DOMAIN => {
                        write!(f, "%os:n{}", oref.node.index())
                    }
                    PortKind::State(domain) if domain == MEMORY_STATE_DOMAIN => {
                        write!(f, "%ms:n{}", oref.node.index())
                    }
                    PortKind::State(domain) => {
                        write!(f, "%s{}:n{}", domain.index(), oref.node.index())
                    }
                }
            }
            PortSource::RegionArg(aref) => {
                let arg = &self.region_args[aref.arg];
                // Find the index of this arg in the region's arg list for display
                let display_index = self.regions[aref.region]
                    .args
                    .iter()
                    .position(|&id| id == aref.arg)
                    .unwrap_or(0);
                match arg.kind {
                    PortKind::Data => write!(f, "arg{}", display_index),
                    PortKind::State(domain) if domain == OUTPUT_STATE_DOMAIN => {
                        write!(f, "%os:arg")
                    }
                    PortKind::State(domain) if domain == MEMORY_STATE_DOMAIN => {
                        write!(f, "%ms:arg")
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
            PortKind::State(domain) if domain == OUTPUT_STATE_DOMAIN => write!(f, "%os")?,
            PortKind::State(domain) if domain == MEMORY_STATE_DOMAIN => write!(f, "%ms")?,
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
            PortKind::State(domain) if domain == OUTPUT_STATE_DOMAIN => write!(f, "%os"),
            PortKind::State(domain) if domain == MEMORY_STATE_DOMAIN => write!(f, "%ms"),
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
    fn test_label() -> &'static str {
        "u8"
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
        // Build: read_from_slot -> write_to_field
        // Verify that each stateful op's input is the previous one's output.
        let mut builder = IrBuilder::new(test_label(), 0);
        let slot = builder.alloc_slot();

        let body = builder.func.root_body();
        let initial_os = PortSource::RegionArg(RegionArgRef {
            region: body,
            arg: builder.func.regions[body].args[0],
        });
        let initial_ms = PortSource::RegionArg(RegionArgRef {
            region: body,
            arg: builder.func.regions[body].args[1],
        });

        {
            let mut rb = builder.root_region();

            // Check initial state.
            assert_eq!(rb.output_state(), initial_os);
            assert_eq!(rb.memory_state(), initial_ms);

            // read_from_slot (memory-domain op)
            let data = rb.read_from_slot(slot);
            let after_read_ms = rb.memory_state();
            assert_ne!(after_read_ms, initial_ms);
            assert_eq!(rb.output_state(), initial_os); // output unchanged

            // write_to_field (output-domain op)
            rb.write_to_field(data, 0, Width::W4);
            assert_eq!(rb.memory_state(), after_read_ms); // memory unchanged by output op
            assert_ne!(rb.output_state(), initial_os); // output updated

            rb.set_results(&[]);
        };

        let func = builder.finish();

        // 2 nodes in the root region.
        assert_eq!(func.regions[func.root_body()].nodes.len(), 2);
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

        assert_eq!(
            IrOp::CallEffect {
                func: IntrinsicFn(0),
                arg_count: 0,
            }
            .effect(),
            MEMORY_EFFECT
        );
        assert!(
            IrOp::CallEffect {
                func: IntrinsicFn(0),
                arg_count: 0,
            }
            .has_side_effects()
        );
    }

    #[test]
    fn op_metadata_side_effects() {
        assert!(!IrOp::Const { value: 1 }.has_side_effects());
        assert!(
            !IrOp::CallPure {
                func: IntrinsicFn(0),
                arg_count: 0
            }
            .has_side_effects()
        );
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
        let mut builder = IrBuilder::new(test_label(), 0);

        {
            let mut rb = builder.root_region();

            // Use a const as predicate.
            let tag = rb.const_val(1);

            // Gamma: branch on tag.
            // Branch 0: write const to field 0.
            // Branch 1: write const to field 0.
            let _results = rb.gamma(tag, &[], 2, |branch_idx, branch| match branch_idx {
                0 => {
                    let val = branch.const_val(4);
                    branch.write_to_field(val, 0, Width::W4);
                    branch.set_results(&[]);
                }
                1 => {
                    let val = branch.const_val(8);
                    branch.write_to_field(val, 0, Width::W8);
                    branch.set_results(&[]);
                }
                _ => unreachable!(),
            });

            rb.set_results(&[]);
        }

        let func = builder.finish();

        // Root region: const + gamma = 2 nodes.
        assert_eq!(func.regions[func.root_body()].nodes.len(), 2);

        // Find the gamma node.
        let gamma_id = func.regions[func.root_body()].nodes[1];
        let gamma = &func.nodes[gamma_id];
        match &gamma.kind {
            NodeKind::Gamma { regions } => {
                assert_eq!(regions.len(), 2);
                // Each branch has 2 nodes: const, write_to_field.
                assert_eq!(func.regions[regions[0]].nodes.len(), 2);
                assert_eq!(func.regions[regions[1]].nodes.len(), 2);
                // Each branch has 2 results (cursor state + output state).
                assert_eq!(func.regions[regions[0]].results.len(), 2);
                assert_eq!(func.regions[regions[1]].results.len(), 2);
            }
            _ => panic!("expected gamma node"),
        }
    }

    #[test]
    fn theta_counted_loop() {
        let mut builder = IrBuilder::new(test_label(), 0);

        {
            let mut rb = builder.root_region();

            // Use a const as count.
            let count = rb.const_val(3);

            // Loop `count` times, writing a const each iteration.
            let _results = rb.theta(&[count], |body| {
                let args = body.region_args(1);
                let counter = args[0];

                // Write a const to field.
                let val = body.const_val(42);
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

        // Root region: const + theta = 2 nodes.
        assert_eq!(func.regions[func.root_body()].nodes.len(), 2);

        // Find the theta node.
        let theta_id = func.regions[func.root_body()].nodes[1];
        let theta = &func.nodes[theta_id];
        match &theta.kind {
            NodeKind::Theta { body, .. } => {
                // Body: const, write_to_field, const, sub = 4 nodes.
                assert_eq!(func.regions[*body].nodes.len(), 4);
                // Results: predicate + loop_var + cursor_state + output_state = 4.
                assert_eq!(func.regions[*body].results.len(), 4);
            }
            _ => panic!("expected theta node"),
        }
    }

    #[test]
    fn debug_scopes_track_structured_region_nesting() {
        let mut builder = IrBuilder::new(test_label(), 0);

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
        let mut builder = IrBuilder::new(test_label(), 0);

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
        let mut builder = IrBuilder::new(test_label(), 0);

        {
            let mut rb = builder.root_region();
            let val = rb.const_val(42);
            rb.write_to_field(val, 0, Width::W4);
            rb.set_results(&[]);
        }

        let func = builder.finish();
        let output = format!("{func}");

        // Verify the output contains expected fragments.
        assert!(output.contains("lambda @0"), "missing lambda header");
        assert!(output.contains("Const("), "missing Const");
        assert!(
            output.contains("WriteToField(offset=0, W4)"),
            "missing WriteToField"
        );
        assert!(output.contains("results:"), "missing results");
    }
}
