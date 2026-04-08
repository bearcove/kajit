pub mod display;
pub mod parse;

pub(crate) use crate::ir as kajit_ir;
use kajit_ir::{Arena, DebugScope, DebugScopeId, DebugValueId, Id, VReg, Width};

// ─── LinearIr ────────────────────────────────────────────────────────────────

/// The linearized form of an RVSDG function.
///
/// `LinearIr` is the bridge between structured RVSDG and lower control-flow
/// based IRs. Structured regions have been flattened into a single instruction
/// stream with explicit labels and branches, but values still live in virtual
/// registers and stack allocations remain abstract.
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
    /// Stack allocations (variable-size frame regions).
    pub stack_allocs: Vec<kajit_ir::StackAllocInfo>,
}

/// Debug provenance copied from RVSDG onto linear operations and vregs.
///
/// The linearizer preserves source/debug scope information so later stages can
/// recover where a linear op or virtual register originated.
#[derive(Clone, Default)]
pub struct LinearDebugProvenance {
    /// Scope arena copied from the source RVSDG.
    pub scopes: Arena<DebugScope>,
    /// Semantic value labels copied from the source RVSDG.
    pub values: Arena<DebugValue>,
    /// Root scope of the function, if known.
    pub root_scope: Option<DebugScopeId>,
    /// Per-op scope provenance, indexed by `LinearIr::ops`.
    pub op_scopes: Vec<Option<DebugScopeId>>,
    /// Per-op semantic value provenance, indexed by `LinearIr::ops`.
    pub op_values: Vec<Option<DebugValueId>>,
    /// Scope provenance for each VReg index.
    pub vreg_scopes: Vec<Option<DebugScopeId>>,
    /// Semantic value provenance for each VReg index.
    pub vreg_values: Vec<Option<DebugValueId>>,
}

// ─── Linearizer state ────────────────────────────────────────────────────────

/// A value in the safe exit environment: either an already-defined vreg
/// or a constant that needs a fresh vreg emitted at exit time.
#[derive(Clone, Copy, Debug)]
enum SafeExitVal {
    Vreg(VReg),
    Const(u64),
    /// Value is computed inside a gamma branch and not available at the exit point.
    Unavailable,
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

// ─── Label ID ────────────────────────────────────────────────────────────────

/// Marker type for label IDs.
pub struct LabelMarker;
/// A label in the linear instruction sequence.
pub type LabelId = Id<LabelMarker>;

// ─── BinOpKind / UnaryOpKind ─────────────────────────────────────────────────

/// Binary operation kind for linear IR.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BinOpKind {
    /// Integer addition.
    Add,
    /// Integer subtraction.
    Sub,
    /// Integer multiplication.
    Mul,
    /// Bitwise AND.
    And,
    /// Bitwise OR.
    Or,
    /// Logical right shift.
    Shr,
    /// Logical left shift.
    Shl,
    /// Arithmetic right shift.
    Sar,
    /// Bitwise XOR.
    Xor,
    /// Equality comparison, producing 0/1.
    CmpEq,
    /// Inequality comparison, producing 0/1.
    CmpNe,
    /// Signed/unsigned less-than according to upstream semantics.
    CmpLt,
    /// Signed/unsigned less-than-or-equal according to upstream semantics.
    CmpLe,
    /// Signed/unsigned greater-than according to upstream semantics.
    CmpGt,
    /// Signed/unsigned greater-than-or-equal according to upstream semantics.
    CmpGe,
}

/// Unary operation kind for linear IR.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum UnaryOpKind {
    /// Extend a narrow integer to 64 bits using the source width's sign bit.
    SignExtend { from_width: Width },
}

// ─── LinearOp ────────────────────────────────────────────────────────────────

/// A single instruction in the linearized IR.
///
/// Each variant corresponds to an RVSDG `IrOp`, but flattened into a linear
/// sequence with explicit labels and branches for control flow.
// r[impl ir.linearize]
#[derive(Debug, Clone)]
pub enum LinearOp {
    // ── Values ──
    /// Materialize an immediate constant.
    Const { dst: VReg, value: u64 },
    /// Load the runtime address of an embedded data blob (relocation target).
    DataAddr { dst: VReg, blob_id: u32 },
    /// Load the address of an external symbol (vtable function pointer etc.).
    /// The runtime address is resolved from a symbol table at emit/interpret time.
    ExternAddr {
        dst: VReg,
        symbol: kajit_types::SymbolName,
    },
    /// Execute a binary arithmetic or comparison op.
    BinOp {
        op: BinOpKind,
        dst: VReg,
        lhs: VReg,
        rhs: VReg,
    },
    /// Execute a unary op.
    UnaryOp {
        op: UnaryOpKind,
        dst: VReg,
        src: VReg,
    },
    /// Copy a value between virtual registers (for gamma merge / theta feedback).
    Copy { dst: VReg, src: VReg },

    // ── Stack ──
    /// Produce the address of an abstract stack allocation.
    StackAlloc {
        dst: VReg,
        id: kajit_ir::StackAllocId,
    },
    /// Store a scalar value to memory.
    StoreToAddr { addr: VReg, src: VReg, width: Width },
    /// Load a scalar value from memory.
    LoadFromAddr { dst: VReg, addr: VReg, width: Width },

    // ── Calls ──
    /// Call an external/raw function pointer.
    Call {
        func: FnPtr,
        args: Vec<VReg>,
        dst: VReg,
    },

    // ── Control flow ──
    /// Label marking a control-flow target.
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
    /// Delimit the start of a linearized lambda body.
    FuncStart {
        lambda_id: LambdaId,
        label: String,
        /// Minimum output buffer size in bytes. Used by the interpreter/simulator
        /// to allocate the output buffer when static inference from StoreToAddr is insufficient.
        output_size: usize,
        data_args: Vec<VReg>,
        data_results: Vec<VReg>,
    },
    /// Delimit the end of a linearized lambda body.
    FuncEnd,
    /// Call another linearized IR lambda by ID.
    CallLambda {
        target: LambdaId,
        args: Vec<VReg>,
        results: Vec<VReg>,
    },
}

impl LinearOp {
    /// Visit every VReg use (read) in this op, mutably.
    ///
    /// Phi edges are represented explicitly as `(source, destination)` pairs on
    /// branch instructions; this visitor sees only the source side as a use.
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
            StoreToAddr { addr, src, .. } => {
                f(addr);
                f(src);
            }
            LoadFromAddr { addr, .. } => f(addr),

            // Calls
            Call { args, .. } => {
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

    /// Visit every VReg use (read) in this op, immutably.
    pub fn for_each_use(&self, mut f: impl FnMut(&VReg)) {
        // Clone and delegate to mutable version (avoids duplicating the match)
        let mut clone = self.clone();
        clone.for_each_use_mut(|v| f(v));
    }

    /// Visit every VReg definition (write) in this op, immutably.
    pub fn for_each_def(&self, mut f: impl FnMut(&VReg)) {
        let mut clone = self.clone();
        clone.for_each_def_mut(|v| f(v));
    }

    /// Visit every VReg definition (write) in this op, mutably.
    ///
    /// For branches, the destination side of each phi pair counts as a def in
    /// the successor environment.
    pub fn for_each_def_mut(&mut self, mut f: impl FnMut(&mut VReg)) {
        use LinearOp::*;
        match self {
            Const { dst, .. }
            | DataAddr { dst, .. }
            | ExternAddr { dst, .. }
            | BinOp { dst, .. }
            | UnaryOp { dst, .. }
            | Copy { dst, .. }
            | StackAlloc { dst, .. }
            | LoadFromAddr { dst, .. }
            | Call { dst, .. } => f(dst),

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
            StoreToAddr { .. } | Label(_) | JumpTable { .. } | FuncEnd => {}
        }
    }

    /// Visit every VReg mentioned by this op, including both uses and defs.
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

            StoreToAddr { addr, src, .. } => {
                f(addr);
                f(src);
            }
            LoadFromAddr { dst, addr, .. } => {
                f(dst);
                f(addr);
            }
            Call { args, dst, .. } => {
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
