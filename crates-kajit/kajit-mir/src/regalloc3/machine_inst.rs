//! Machine instruction operand model (explicit contract).
//!
//! This module defines the canonical representation of machine instructions
//! for register allocation. The key principle: ONE canonical operand record
//! per operand, with all helper views derived from it.

use kajit_ir::VReg;

use kajit_reprs::mir::{BlockId, InstId};

/// Physical register (target-specific)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct PReg(pub u8);

impl PReg {
    pub fn new(n: u8) -> Self {
        Self(n)
    }

    pub fn index(self) -> u8 {
        self.0
    }
}

/// Machine operand (ONE canonical representation)
///
/// This is the single source of truth for instruction operands.
/// All helper views (uses, defs, fixed constraints) are derived from this.
#[derive(Debug, Clone)]
pub struct MachineOperand {
    /// Virtual register
    pub vreg: VReg,

    /// Use or def
    pub kind: OperandKind,

    /// Early or late (timing)
    pub pos: OperandPos,

    /// Constraint (any register, fixed physical register, etc.)
    pub constraint: OperandConstraint,

    /// Two-address constraint: output tied to input
    /// If Some(N), this def must be in same register as operand N
    pub tied_to: Option<usize>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OperandKind {
    Use,
    Def,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OperandPos {
    /// Read/written before main operation
    Early,
    /// Read/written after main operation
    Late,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OperandConstraint {
    /// Any register in class
    Any,
    /// Must be in specific physical register
    Fixed(PReg),
    /// Must be in same register as operand N (for two-address instructions)
    SameAs(usize),
}

/// Machine instruction interface (exposes operands + metadata)
pub trait MachineInst {
    /// Canonical operand list (SINGLE SOURCE OF TRUTH)
    fn operands(&self) -> &[MachineOperand];

    /// Implicit clobbers (calls, special ops)
    /// These are NOT in operands list (they're implicit)
    fn clobbers(&self) -> &[PReg];

    /// Is this a call instruction? (affects callee-saved usage)
    fn is_call(&self) -> bool;

    /// Is this a move? (for debugging, coalescing in later phases)
    /// Returns (dst, src) if this is a move
    fn is_move(&self) -> Option<(VReg, VReg)>;
}

/// Helper views derived from operands (NOT separate sources of truth)
pub trait MachineInstExt: MachineInst {
    /// Iterate over use operands
    fn uses(&self) -> impl Iterator<Item = &MachineOperand> {
        self.operands()
            .iter()
            .filter(|op| matches!(op.kind, OperandKind::Use))
    }

    /// Iterate over def operands
    fn defs(&self) -> impl Iterator<Item = &MachineOperand> {
        self.operands()
            .iter()
            .filter(|op| matches!(op.kind, OperandKind::Def))
    }

    /// Get fixed-register constraints
    fn fixed_constraints(&self) -> impl Iterator<Item = (VReg, PReg, OperandKind)> + '_ {
        self.operands().iter().filter_map(|op| {
            if let OperandConstraint::Fixed(preg) = op.constraint {
                Some((op.vreg, preg, op.kind))
            } else {
                None
            }
        })
    }
}

// Blanket implementation
impl<T: MachineInst + ?Sized> MachineInstExt for T {}

/// Block parameter handling (separate from instructions)
pub trait MachineBlock {
    /// Block ID
    fn id(&self) -> BlockId;

    /// Block parameters (phi values)
    /// These are DEFS at block entry
    fn params(&self) -> &[VReg];

    /// Incoming edge arguments for this block from given predecessor
    /// These are USES on predecessor edges
    fn edge_args(&self, pred: BlockId) -> &[VReg];

    /// All predecessor blocks
    fn preds(&self) -> &[BlockId];

    /// All successor blocks
    fn succs(&self) -> &[BlockId];

    /// Instructions in this block
    fn insts(&self) -> &[InstId];
}

/// ABI information (calling convention)
///
/// The allocator CANNOT be ignorant of the ABI.
/// Calls implicitly use/define fixed registers, and callee-saved
/// registers require prologue/epilogue save/restore.
#[derive(Debug, Clone)]
pub struct AbiInfo {
    /// Caller-saved GPRs (clobbered by calls)
    pub caller_saved_gpr: &'static [PReg],

    /// Callee-saved GPRs (preserved across calls)
    /// If allocator uses these, prologue/epilogue must save/restore
    pub callee_saved_gpr: &'static [PReg],

    /// Argument registers (implicit uses in calls)
    pub arg_gprs: &'static [PReg],

    /// Return registers (implicit defs in calls)
    pub ret_gprs: &'static [PReg],

    /// Stack red zone size (if any)
    /// Red zone = space below stack pointer that won't be clobbered by signals
    pub red_zone_size: usize,
}

// AArch64 ABI
#[cfg(target_arch = "aarch64")]
pub const TARGET_ABI: AbiInfo = {
    // GPRs: x0-x30 (x31 is sp/zr depending on context)
    const fn gpr(n: u8) -> PReg {
        PReg(n)
    }

    AbiInfo {
        // x0-x18 are caller-saved (clobbered by calls)
        caller_saved_gpr: &[
            gpr(0),
            gpr(1),
            gpr(2),
            gpr(3),
            gpr(4),
            gpr(5),
            gpr(6),
            gpr(7),
            gpr(8),
            gpr(9),
            gpr(10),
            gpr(11),
            gpr(12),
            gpr(13),
            gpr(14),
            gpr(15),
            gpr(16), // IP0 (also scratch)
            gpr(17), // IP1 (also scratch)
            gpr(18), // platform register
        ],
        // x19-x28 are callee-saved (preserved across calls)
        callee_saved_gpr: &[
            gpr(19),
            gpr(20),
            gpr(21),
            gpr(22),
            gpr(23),
            gpr(24),
            gpr(25),
            gpr(26),
            gpr(27),
            gpr(28),
        ],
        // x0-x7 are argument registers
        arg_gprs: &[
            gpr(0),
            gpr(1),
            gpr(2),
            gpr(3),
            gpr(4),
            gpr(5),
            gpr(6),
            gpr(7),
        ],
        // x0-x1 are return registers
        ret_gprs: &[gpr(0), gpr(1)],
        // AArch64 has no red zone
        red_zone_size: 0,
    }
};

// x86_64 ABI (System V)
#[cfg(target_arch = "x86_64")]
pub const TARGET_ABI: AbiInfo = {
    // x86_64 registers: rax=0, rcx=1, rdx=2, rbx=3, rsp=4, rbp=5,
    // rsi=6, rdi=7, r8=8, ..., r15=15
    const fn gpr(n: u8) -> PReg {
        PReg(n)
    }

    AbiInfo {
        // rax, rcx, rdx, rsi, rdi, r8-r11 are caller-saved
        caller_saved_gpr: &[
            gpr(0),  // rax
            gpr(1),  // rcx
            gpr(2),  // rdx
            gpr(6),  // rsi
            gpr(7),  // rdi
            gpr(8),  // r8
            gpr(9),  // r9
            gpr(10), // r10
            gpr(11), // r11
        ],
        // rbx, rbp, r12-r15 are callee-saved
        callee_saved_gpr: &[
            gpr(3),  // rbx
            gpr(5),  // rbp
            gpr(12), // r12
            gpr(13), // r13
            gpr(14), // r14
            gpr(15), // r15
        ],
        // rdi, rsi, rdx, rcx, r8, r9 are argument registers
        arg_gprs: &[gpr(7), gpr(6), gpr(2), gpr(1), gpr(8), gpr(9)],
        // rax, rdx are return registers
        ret_gprs: &[gpr(0), gpr(2)],
        // x86_64 has 128-byte red zone
        red_zone_size: 128,
    }
};

/// Scratch register policy (TARGET/BACKEND CONTRACT)
///
/// These registers are NEVER allocated for user values.
/// Codegen, helpers, and calls may rely on their availability.
#[derive(Debug, Clone)]
pub struct ScratchPolicy {
    /// Reserved scratch registers (never allocated, always available)
    pub reserved: &'static [PReg],

    /// Maximum simultaneous spills per instruction
    /// If an instruction needs more scratch regs than reserved,
    /// we must split the instruction or fail gracefully
    pub max_simultaneous_spills: usize,
}

#[cfg(target_arch = "aarch64")]
pub const SCRATCH_POLICY: ScratchPolicy = ScratchPolicy {
    // x16, x17 are IP0, IP1 (intra-procedure-call scratch registers)
    reserved: &[PReg(16), PReg(17)],
    max_simultaneous_spills: 2,
};

#[cfg(target_arch = "x86_64")]
pub const SCRATCH_POLICY: ScratchPolicy = ScratchPolicy {
    // r10, r11 are available as scratch (caller-saved, not used for args)
    reserved: &[PReg(10), PReg(11)],
    max_simultaneous_spills: 2,
};
