# Register Allocation Roadmap (regalloc3)

**Status:** Planning
**Author:** Architecture discussion 2026-03-24
**Problem:** regalloc2 adapter fragility causing correctness issues

## Executive Summary

We will build our own register allocator ("regalloc3") to eliminate the fragile adapter layer between our CFG-MIR and regalloc2. The allocator will be **simple and correct**, not sophisticated. RVSDG will provide allocation hints upstream, but physical register assignment happens at CFG-MIR level after instruction selection.

**Key insight:** The problem is not "register allocation is hard" but "continuously translating our IR into regalloc2's mental model is where correctness goes to die."

## The Problem

### Impedance Mismatch

regalloc2 wants:
- Dense linear indices (0, 1, 2, ...)
- Array-based data structures
- Snapshot of entire function upfront
- Explicit operand constraints in specific format

Our CFG-MIR uses:
- Stable IDs (`InstId`, `BlockId`, `VReg`)
- Indirection (blocks contain inst IDs, not insts)
- Non-dense allocation (gaps in vreg numbering)
- Incremental construction and transformation

### Adapter Fragility

Every CFG transformation must maintain adapter invariants:
- `const_phi_elim` eliminates block parameters → adapter must reflect this
- `block_merge` combines blocks → adapter indices must update
- `copyprop` rewrites vregs → adapter must track changes

**The adapter is where correctness goes to die.**

Recent bugs:
- 2026-03-24: `const_phi_elim` broke regalloc2 (UnknownValueInAllocation errors)
- 2026-03-23: `loop_phi_elim` + `remat` interaction caused SSA violations
- Our SSA validator passes but regalloc2's checker fails

**Root cause:** We don't own the allocator, so we can't debug it properly.

## The Solution

### Architecture: Three Levels

```
┌─────────────────────────────────────────────────┐
│ RVSDG / IR Level                                │
│ ─────────────────                               │
│ Pressure-aware optimization                     │
│ Generate allocation hints:                      │
│  • Loop-carried values → high spill cost        │
│  • Pure nodes → rematerializable                │
│  • Cross-call values → prefer callee-saved      │
│  • Gamma merges → coalescing affinity           │
│                                                  │
│ This is the INTERESTING compiler work!          │
└─────────────────────────────────────────────────┘
                      ↓ (thread hints)
┌─────────────────────────────────────────────────┐
│ CFG-MIR Level                                   │
│ ─────────────                                   │
│ Actual register allocation:                     │
│  • Compute liveness (still needed!)             │
│  • Linear scan allocation                       │
│  • Use hints as spill weights                   │
│  • Insert spill/reload code                     │
│  • Work directly with stable IDs                │
│                                                  │
│ NO ADAPTER LAYER                                │
└─────────────────────────────────────────────────┘
                      ↓
┌─────────────────────────────────────────────────┐
│ Backend                                         │
│ ───────                                         │
│ Emit machine code with physical registers       │
└─────────────────────────────────────────────────┘
```

### Key Principles

1. **Simple over sophisticated** - Linear scan, not graph coloring
2. **Correct over optimal** - First goal is removing adapter bugs
3. **Hints from structure** - RVSDG provides spill costs, not register assignments
4. **Late binding** - Allocate after instruction selection, when constraints are known

## Non-Goals

**We are explicitly NOT:**
- Trying to beat regalloc2 on code quality (at least not initially)
- Doing allocation at RVSDG level (too early, constraints unknown)
- Implementing graph coloring (unnecessary complexity)
- Inventing a new allocation paradigm (use proven algorithms)
- **Rejecting dense orderings internally** (we need them for liveness!)

**We ARE:**
- Eliminating the fragile external adapter (own the allocator)
- Working with stable IDs in our API (CFG-MIR stays unchanged)
- **Deriving internal program points** (per-function, lossless ordering)
- Using RVSDG structure for hints (novel part, but Phase 2+)
- Building something simple and debuggable

**Key distinction:**
> We are not rejecting dense orderings internally. We are rejecting a fragile external adapter. The allocator may derive its own per-function ProgPoint numbering while still operating on our native MIR and stable IDs.

## Implementation Phases

### Phase 0: Prerequisites (Complete)

- [x] CFG-MIR with stable IDs
- [x] SSA validator
- [x] RVSDG optimization passes
- [x] Basic backend emission

### Phase 1: Minimal Native Allocator (1-2 weeks, not days)

**Goal:** Build a minimal native allocator over our CFG-MIR that operates on an internally derived ordered program-point space, supports one register class (GPR only), handles calls/clobbers and block-parameter edge moves correctly, inserts spills/reloads safely, and is backed by a symbolic allocation verifier.

**This is the actual milestone. Do not add hints, coalescing, rematerialization, or SIMD until this works.**

#### Step 1.1: Define the Machine MIR Contract

Before allocating anything, explicitly define what every instruction exposes.

**Rule 1: One canonical operand record (not decomposed trait methods)**

```rust
/// Machine operand (ONE canonical representation)
#[derive(Debug, Clone)]
pub struct MachineOperand {
    pub vreg: VReg,
    pub kind: OperandKind,
    pub pos: OperandPos,
    pub constraint: OperandConstraint,
    pub tied_to: Option<usize>,  // two-address: output tied to input
}

#[derive(Debug, Clone, Copy)]
pub enum OperandKind {
    Use,
    Def,
}

#[derive(Debug, Clone, Copy)]
pub enum OperandPos {
    Early,  // read/written before main operation
    Late,   // read/written after main operation
}

#[derive(Debug, Clone, Copy)]
pub enum OperandConstraint {
    Any,                    // any register in class
    Fixed(PReg),            // must be in specific physical register
    SameAs(usize),          // must be in same register as operand N
}

/// Machine instruction (exposes operands + metadata)
pub trait MachineInst {
    /// Canonical operand list (SINGLE SOURCE OF TRUTH)
    fn operands(&self) -> &[MachineOperand];

    /// Implicit clobbers (calls, special ops)
    /// These are NOT in operands list (implicit)
    fn clobbers(&self) -> &[PReg];

    /// Is this a call? (affects callee-saved usage)
    fn is_call(&self) -> bool;

    /// Is this a move? (for debugging, coalescing later)
    fn is_move(&self) -> Option<(VReg, VReg)>;
}

// Helper views (DERIVED from operands, not separate methods)
impl dyn MachineInst {
    fn uses(&self) -> impl Iterator<Item = &MachineOperand> {
        self.operands().iter().filter(|op| matches!(op.kind, OperandKind::Use))
    }

    fn defs(&self) -> impl Iterator<Item = &MachineOperand> {
        self.operands().iter().filter(|op| matches!(op.kind, OperandKind::Def))
    }
}

/// Block parameter handling (separate from instructions)
pub trait MachineBlock {
    /// Block parameters (phi values)
    fn params(&self) -> &[VReg];

    /// Incoming edge arguments for this block
    /// Maps predecessor block → list of vregs passed as args
    fn edge_args(&self, pred: BlockId) -> &[VReg];
}
```

**Why one canonical record:**
- Prevents inconsistencies (uses() and operands() can't disagree)
- Matches regalloc2's model (timing is part of semantics)
- Matches LLVM's model (implicit uses/defs are first-class)

**Deliverable:** Implement this for our CFG-MIR `Inst` and `Block` types.

**Rule 5: ABI ownership (allocator knows about calls)**

The allocator CANNOT be ignorant of calling conventions:
- Calls implicitly use/define fixed registers (ABI args/returns)
- Caller-saved registers are clobbered by calls
- If allocator uses callee-saved registers, prologue/epilogue must save/restore

**ABI contract:**
```rust
pub struct AbiInfo {
    /// Caller-saved GPRs (clobbered by calls)
    pub caller_saved_gpr: &'static [PReg],

    /// Callee-saved GPRs (preserved across calls)
    pub callee_saved_gpr: &'static [PReg],

    /// Argument registers (implicit uses)
    pub arg_gprs: &'static [PReg],

    /// Return registers (implicit defs)
    pub ret_gprs: &'static [PReg],

    /// Stack red zone (if any)
    pub red_zone_size: usize,
}

// AArch64 example
pub const AARCH64_ABI: AbiInfo = AbiInfo {
    caller_saved_gpr: &[x0, x1, ..., x18],
    callee_saved_gpr: &[x19, x20, ..., x28],
    arg_gprs: &[x0, x1, x2, x3, x4, x5, x6, x7],
    ret_gprs: &[x0, x1],
    red_zone_size: 0,  // AArch64 has no red zone
};
```

**Usage:**
- Call instructions implicitly clobber `caller_saved_gpr`
- Allocator tracks which `callee_saved_gpr` it uses
- Backend generates prologue/epilogue to save/restore used callee-saved regs

**Why this matters:** Fuzzy ABI handling breaks calling conventions. Get it explicit.

**Function entry/exit semantics:**

Function arguments and returns have special semantics:

```rust
/// Function entry (entry block)
/// Arguments are NOT block params - they're precolored defs
fn entry_block_semantics() {
    // Entry block has implicit instruction:
    // def x0:fixed, def x1:fixed, ..., def x7:fixed
    // (for ABI arg registers that are actually used)
    //
    // These are MachineOperands with:
    //   kind: Def
    //   pos: Early
    //   constraint: Fixed(x0), Fixed(x1), etc.
    //
    // Backend materializes these (they're already there from caller)
}

/// Function exit (return instruction)
/// Return values are fixed uses of ABI return registers
fn return_semantics() {
    // Return instruction has implicit operands:
    // use x0:fixed, use x1:fixed (for return values)
    //
    // These are MachineOperands with:
    //   kind: Use
    //   pos: Late
    //   constraint: Fixed(x0), Fixed(x1)
    //
    // Allocator must ensure return values are in these regs
}
```

**Why this matters:** Forgetting entry/exit semantics causes "value has no reaching def" bugs at boundaries.

#### Step 1.2: Program Point Model (NOT InstId!)

**Problem:** `InstId` is a stable identity, not a program order. Live intervals need ordering and can have holes.

**Solution:** Derive dense program points per function:

```rust
/// Program point (derived fresh per function, NOT InstId)
#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
struct ProgPoint(u32);

/// Live interval (can have holes!)
struct LiveInterval {
    vreg: VReg,
    segments: Vec<(ProgPoint, ProgPoint)>,  // multiple ranges
    uses: Vec<ProgPoint>,                    // all use points
}

/// Map from program order to stable IDs (for reconstruction)
struct ProgPointMap {
    point_to_inst: Vec<InstId>,     // ProgPoint -> InstId
    inst_to_point: HashMap<InstId, ProgPoint>,  // InstId -> ProgPoint
    block_entry: HashMap<BlockId, ProgPoint>,   // Block entry points
    block_exit: HashMap<BlockId, ProgPoint>,    // Block exit points
}
```

**Algorithm to build program points:**
```rust
fn build_program_points(func: &Function) -> ProgPointMap {
    let mut point = 0;
    let mut map = ProgPointMap::new();

    for block in RPO order {
        map.block_entry[block.id] = ProgPoint(point);
        point += 2;  // leave gaps for spill insertion

        for inst_id in &block.insts {
            map.point_to_inst.push(inst_id);
            map.inst_to_point.insert(inst_id, ProgPoint(point));
            point += 2;  // gaps between instructions
        }

        map.block_exit[block.id] = ProgPoint(point);
        point += 2;
    }

    map
}
```

**Why gaps:** Leave room to insert spill/reload instructions without renumbering.

**Deliverable:** `regalloc3/progpoint.rs` - program point derivation and mapping

#### Step 1.3: Edge Copies and Block Parameters (THE HARD PART)

**This is where correctness goes to die if you hand-wave it.**

**Rule 3: Split critical edges BEFORE allocation; edge-copy blocks own phi resolution**

Critical edge = predecessor has multiple successors AND successor has multiple predecessors.

**Hard rule for v1:** All critical edges are split before allocation. All block-param resolution lives in explicit edge-copy blocks.

**Why:** This is the least clever and most debuggable approach. regalloc2 explicitly requires this for some cases. SSA linear-scan literature treats edge resolution and phi resolution as the same problem.

```rust
/// Critical edge splitting (before allocation)
fn split_critical_edges(func: &mut Function) {
    for edge_id in func.edges.iter() {
        let edge = &func.edges[edge_id];
        let pred = &func.blocks[edge.from.index()];
        let succ = &func.blocks[edge.to.index()];

        if pred.succs.len() > 1 && succ.preds.len() > 1 {
            // Critical edge! Insert copy block
            let copy_block = insert_empty_block_on_edge(func, edge_id);
            // Now: pred -> copy_block -> succ
            // Parallel copies go in copy_block
        }
    }
}
```

Block parameters require parallel copies at block boundaries:
```rust
// Example:
// b1 ends with: jump b2(v1, v2, v3)
// b2 starts with: params(v10, v11, v12)
//
// Semantically: v10 := v1; v11 := v2; v12 := v3 (in parallel!)
```

**Problem:** Copies must happen in parallel (cycles possible):
```rust
// Edge: b1 -> b2(v1, v2)
// b2 params: (v20, v21)
//
// What if allocated as:
//   v1 = x0, v2 = x1
//   v20 = x1, v21 = x0
//
// This is a SWAP! Can't do serially:
//   x1 := x0  // clobbers x1 before v2 is read!
//   x0 := x1  // reads wrong value
```

**Solution:** Dedicated parallel copy resolver:

```rust
/// Resolve parallel copy at edge/block boundary
struct ParallelCopyResolver {
    copies: Vec<(PReg, PReg)>,  // dst, src pairs
}

impl ParallelCopyResolver {
    fn resolve(&self) -> Vec<MoveOp> {
        // Detect cycles
        // Break cycles with swaps or temp register
        // Order copies to avoid clobbering
        // Return sequence of moves/swaps
    }
}

enum MoveOp {
    Move { dst: PReg, src: PReg },
    Swap { a: PReg, b: PReg },          // if target supports swap
    MoveToTemp { dst: PReg, temp: PReg },  // for cycle breaking
}
```

**Critical edge handling:**
```rust
// Critical edge: predecessor has multiple successors,
//                successor has multiple predecessors
//
// Must insert empty block to hold copies:
//
//   b1 -> b2    becomes    b1 -> b1_2_copy -> b2
//                          (copies go in b1_2_copy)
```

**Deliverable:**
- `regalloc3/parallel_copy.rs` - parallel copy resolution
- `regalloc3/critical_edge.rs` - critical edge splitting
- Tests with cycles, swaps, critical edges

**This is a subsystem, not a "backend handles it" footnote.**

#### Step 1.4: Liveness Analysis (With Holes)

Compute live intervals using program points.

**CRITICAL: Block params and edge args are part of liveness!**

```rust
fn compute_liveness(
    func: &Function,
    progpoints: &ProgPointMap,
) -> HashMap<VReg, LiveInterval> {
    let mut intervals = HashMap::new();

    // 1. Build def-use chains
    let mut defs: HashMap<VReg, ProgPoint> = HashMap::new();
    let mut uses: HashMap<VReg, Vec<ProgPoint>> = HashMap::new();

    // 1a. Block params are DEFS at block entry
    for block in &func.blocks {
        let entry_point = progpoints.block_entry[&block.id];
        for &param_vreg in block.params() {
            defs.insert(param_vreg, entry_point);
        }
    }

    // 1b. Edge args are USES on predecessor edges
    for block in &func.blocks {
        for &pred_id in &block.preds {
            let pred_exit = progpoints.block_exit[&pred_id];
            for &arg_vreg in block.edge_args(pred_id) {
                uses.entry(arg_vreg).or_default().push(pred_exit);
            }
        }
    }

    // 1c. Instruction defs and uses
    for (point, inst_id) in progpoints.iter() {
        let inst = &func.insts[inst_id];

        for op in inst.operands() {
            match op.kind {
                OperandKind::Def => {
                    defs.insert(op.vreg, point);
                }
                OperandKind::Use => {
                    uses.entry(op.vreg).or_default().push(point);
                }
            }
        }
    }

    // 2. Propagate liveness through CFG
    // (Standard dataflow analysis)
    let live_in = compute_live_in_per_block(func, &defs, &uses);

    // 3. Build intervals with holes
    for vreg in all_vregs {
        let def_point = defs[&vreg];
        let use_points = uses.get(&vreg).unwrap_or(&vec![]);

        // Compute segments (can have holes from dead regions)
        let segments = compute_segments(def_point, use_points, &live_in);

        intervals.insert(vreg, LiveInterval {
            vreg,
            segments,
            uses: use_points.clone(),
        });
    }

    intervals
}
```

**Why this matters:** Forgetting that block params are defs and edge args are uses causes "value has no reaching def" or "value used but not live" bugs.

**Deliverable:** `regalloc3/liveness.rs` - liveness with holes, program points, block params

#### Step 1.5: Linear Scan Allocation (One Register Class Only)

**Rule 2: Whole-interval allocation only (no live-range splitting in v1)**

Each vreg gets ONE home for its entire lifetime:
- `vreg -> Allocation::Reg(preg)` means vreg lives in preg for its ENTIRE interval
- `vreg -> Allocation::Stack(slot)` means vreg lives on stack for ENTIRE interval
- Spilled values are materialized through scratch temporaries during rewrite (not split ranges)

**This is deliberate simplification for v1.** Later phases can add splitting.

**Rule 4: Explicit scratch register policy**

When spilling under pressure, we need temporary registers. Policy for v1:

```rust
/// Scratch register policy (TARGET/BACKEND CONTRACT)
/// These registers are NEVER allocated for user values
/// Codegen, helpers, and calls may rely on their availability
pub struct ScratchPolicy {
    /// Reserved scratch registers (never allocated, always available)
    /// AArch64: reserve x16, x17 (IP0, IP1 - platform temp registers)
    pub reserved: &'static [PReg],

    /// Maximum simultaneous spills per instruction
    /// If an instruction needs more scratch regs than reserved,
    /// we must split the instruction or fail gracefully
    pub max_simultaneous_spills: usize,
}
```

**CRITICAL:** Reserved scratch registers are a target/backend contract, not just an allocator convenience. The backend MAY assume these registers are never allocated and use them freely in lowering, helpers, or inline assembly.

**Why this matters:** Cycle breaking and spill rewriting need temps. Accidental reuse breaks the contract.

```rust
struct LinearScanAllocator {
    intervals: Vec<LiveInterval>,  // sorted by start point
    active: Vec<LiveInterval>,     // currently live

    // Available registers (excludes scratch regs!)
    free_regs: Vec<PReg>,

    // ONE home per vreg (whole-interval allocation)
    allocations: HashMap<VReg, Allocation>,

    // Scratch registers (reserved, always available)
    scratch_policy: ScratchPolicy,

    // Spill state
    spill_slots: SlotAllocator,
    next_spill_slot: usize,

    // Track which callee-saved regs we use (for prologue/epilogue)
    used_callee_saved: HashSet<PReg>,
}

impl LinearScanAllocator {
    fn allocate(&mut self) {
        // Sort intervals by start point
        self.intervals.sort_by_key(|i| i.segments[0].0);

        for interval in &self.intervals {
            // Expire old intervals
            self.expire_old_intervals(interval.start());

            // Handle fixed constraints first
            if let Some(fixed_preg) = self.get_fixed_constraint(interval.vreg) {
                self.allocate_fixed(interval, fixed_preg);
                continue;
            }

            // Try to allocate free register
            if let Some(preg) = self.free_regs.pop() {
                self.allocations.insert(interval.vreg, Allocation::Reg(preg));
                self.active.push(interval.clone());
            } else {
                // Must spill
                self.spill(interval);
            }
        }
    }

    fn spill(&mut self, interval: &LiveInterval) {
        // Spill heuristic: pick interval with furthest next use
        let victim = self.choose_spill_victim(interval);

        if victim.vreg == interval.vreg {
            // Spill this interval
            let slot = self.allocate_spill_slot();
            self.allocations.insert(interval.vreg, Allocation::Stack(slot));
        } else {
            // Spill victim, allocate this interval to freed register
            let preg = self.allocations.remove(&victim.vreg).unwrap().as_reg();
            let slot = self.allocate_spill_slot();
            self.allocations.insert(victim.vreg, Allocation::Stack(slot));
            self.allocations.insert(interval.vreg, Allocation::Reg(preg));
        }
    }
}
```

**Deliverable:** `regalloc3/linear_scan.rs` - allocation algorithm

#### Step 1.6: Spill/Reload Insertion (NOT TRIVIAL!)

**Do not hand-wave this.** Choose explicit strategy:

**Strategy A: Rewrite-once with scratch temporaries**
```rust
// For each spilled vreg use:
//   - Allocate scratch temporary (pick unused register)
//   - Insert reload: temp := load [spill_slot]
//   - Rewrite use: use temp instead of vreg
//
// For each spilled vreg def:
//   - Allocate scratch temporary
//   - Rewrite def: def temp instead of vreg
//   - Insert store: store temp -> [spill_slot]
```

**Strategy B: Allocate, rewrite, re-run**
```rust
// 1. Run allocation (marks some vregs as spilled)
// 2. Insert spill/reload vregs (new vregs!)
// 3. Re-run local allocation for new vregs (simpler problem)
```

**Pick one and implement it completely.** Strategy A is simpler for v1.

**Deliverable:** `regalloc3/spill_rewrite.rs` - spill insertion

#### Step 1.7: Symbolic Verification (CRITICAL)

After allocation, symbolically execute location flow:

```rust
struct AllocationVerifier {
    allocations: HashMap<VReg, Allocation>,
    progpoints: ProgPointMap,
}

impl AllocationVerifier {
    fn verify(&self, func: &Function) -> Result<(), Vec<VerifyError>> {
        let mut errors = vec![];

        for block in RPO order {
            let mut state = LocationState::new();

            // Initialize from block parameters
            for (i, param_vreg) in block.params.iter().enumerate() {
                // Check incoming edge args all provide same location
                for pred in &block.preds {
                    let edge_arg = func.edges[pred].args[i];
                    let arg_loc = self.allocations[&edge_arg];
                    // Verify location consistency
                }
            }

            // Walk instructions
            for inst_id in &block.insts {
                let inst = &func.insts[inst_id];

                // Check operands (SINGLE SOURCE OF TRUTH)
                for op in inst.operands() {
                    let loc = self.allocations.get(&op.vreg);

                    match op.kind {
                        OperandKind::Use => {
                            // Check use has reaching definition
                            if !state.has_value_at(op.vreg, loc) {
                                errors.push(VerifyError::UseBeforeDef {
                                    vreg: op.vreg,
                                    inst_id
                                });
                            }

                            // Check fixed constraint
                            if let OperandConstraint::Fixed(fixed_preg) = op.constraint {
                                if loc.as_reg() != Some(fixed_preg) {
                                    errors.push(VerifyError::FixedConstraintViolated {
                                        vreg: op.vreg,
                                        expected: fixed_preg,
                                        actual: loc,
                                    });
                                }
                            }
                        }

                        OperandKind::Def => {
                            // Check fixed constraint
                            if let OperandConstraint::Fixed(fixed_preg) = op.constraint {
                                if loc.as_reg() != Some(fixed_preg) {
                                    errors.push(VerifyError::FixedConstraintViolated {
                                        vreg: op.vreg,
                                        expected: fixed_preg,
                                        actual: loc,
                                    });
                                }
                            }

                            // Apply def
                            state.define(op.vreg, loc);
                        }
                    }
                }

                // Check implicit clobbers
                for &clobbered in inst.clobbers() {
                    if state.preg_holds_live_value(clobbered) {
                        errors.push(VerifyError::ClobberLiveValue {
                            preg: clobbered,
                            inst_id
                        });
                    }
                }

                // Check for register conflicts
                for (preg, vregs) in state.preg_contents() {
                    if vregs.len() > 1 && any_are_live(vregs) {
                        errors.push(VerifyError::RegisterConflict {
                            preg,
                            vregs: vregs.clone(),
                            inst_id,
                        });
                    }
                }
            }
        }

        if errors.is_empty() { Ok(()) } else { Err(errors) }
    }
}
```

**Deliverable:** `regalloc3/verify.rs` - symbolic verifier

**This should be a headline deliverable, not an afterthought.**

#### Phase 1 Summary

**The 5 Hard Rules (locked in before coding):**

1. **One canonical operand record** - `MachineOperand` struct, not decomposed trait methods
2. **Whole-interval allocation only** - One home per vreg for entire lifetime (no splitting)
3. **Critical edges split before RA** - All phi resolution in explicit edge-copy blocks
4. **Explicit scratch register policy** - Reserved temps for spills and cycle breaking
5. **ABI ownership** - Allocator knows about calls, clobbers, callee-saved registers

**These are not negotiable for v1. Deviating from them is scope creep.**

**Deliverables:**
1. Machine MIR operand contract (`machine_inst.rs`)
2. Program point derivation (`progpoint.rs`)
3. Parallel copy resolution (`parallel_copy.rs`)
4. Critical edge splitting (`critical_edge.rs`)
5. Liveness analysis with holes (`liveness.rs`)
6. Linear scan allocator - GPR only (`linear_scan.rs`)
7. Spill/reload insertion (`spill_rewrite.rs`)
8. Symbolic verifier (`verify.rs`)
9. Integration tests (differential behavior, not allocation)

**Success criteria (revised):**
- ✅ **Correct** - symbolic verifier passes on all corpus tests
- ✅ **Same behavior** - programs produce identical output to regalloc2 version
- ❌ **NOT same allocation** - two correct allocators can differ in decisions
- ✅ **Understandable** - each subsystem is clearly defined
- ✅ **Inspectable** - can dump allocation state at any point
- ✅ **Debuggable** - when it fails, we understand why
- ⚠️ **Performance** - if 30% worse for a week, that's FINE if it's correct

**Correctness arbiter:** Symbolic verifier, not "same edits as regalloc2"

**Diagnostics (not correctness):** Spill counts, move counts, register pressure

**Timeline:** 1-2 weeks, not "2-3 days"
- Don't plan emotionally around a weekend hack
- This is a real allocator with real correctness obligations

**Descoped for Phase 1:**
- SIMD registers (just GPR)
- Coalescing (copies are fine)
- Rematerialization (spill everything)
- RVSDG hints (allocate dumbly)
- Profile guidance (no profiling)
- Fancy heuristics (simple is fine)

**Explicit limitations (deferred by design):**

Phase 1 does NOT support:
- **Live-range splitting** - one home per vreg for entire lifetime
- **Spill-slot coalescing** - each spilled vreg gets its own slot
- **Multiple homes per vreg** - no "in x3 here, stack there, x7 later"

Any need for these is explicitly deferred to later phases.

### Phase 2: RVSDG Hints (Only After Phase 1 Works!)

**Prerequisites:** Phase 1 allocator must be correct and shipping before starting this.

**Goal:** Add simple allocation hints from RVSDG structure (spill costs only).

**Hint types (SIMPLE - just spill costs for now):**
```rust
/// Allocation hint (just spill cost for Phase 2)
struct AllocationHint {
    spill_cost: SpillCost,  // that's it!
}

enum SpillCost {
    High,     // loop-carried in nested loop
    Medium,   // loop-carried in single loop
    Low,      // not loop-carried
}
```

**Descoped for Phase 2:**
- ❌ Rematerialization (add later if needed)
- ❌ Coalescing affinity (copies are fine)
- ❌ Call-crossing analysis (simple heuristic is fine)
- ❌ Profile guidance (no profiling yet)

**RVSDG analysis (SIMPLE):**
```rust
// During RVSDG optimization, annotate theta entry ports only
fn analyze_theta_node(theta: &Theta) -> HashMap<PortId, AllocationHint> {
    let mut hints = HashMap::new();
    let nesting = compute_nesting_depth(theta);

    // Entry ports are loop-carried → high spill cost
    for port in &theta.entry_ports {
        hints.insert(port.id, AllocationHint {
            spill_cost: if nesting > 1 {
                SpillCost::High     // nested loop
            } else {
                SpillCost::Medium   // single loop
            },
        });
    }

    hints
}

// Everything else gets default (Low)
```

**Threading through pipeline:**
```rust
// When lowering RVSDG → LIR → CFG-MIR, preserve hints
struct VRegMetadata {
    spill_cost: SpillCost,           // from RVSDG
    debug_name: Option<String>,      // for debugging
}

// In allocator, use as spill weight (simple!)
fn choose_spill_victim(&self, candidates: &[VReg]) -> VReg {
    candidates.iter()
        .min_by_key(|&vreg| {
            let meta = self.vreg_metadata.get(vreg);
            match meta.map(|m| m.spill_cost).unwrap_or(SpillCost::Low) {
                SpillCost::High => 100,
                SpillCost::Medium => 10,
                SpillCost::Low => 1,
            }
        })
        .copied()
        .unwrap()
}
```

**Deliverable:**
- Simple RVSDG analysis (theta entry ports only)
- Thread spill costs through lowering
- Use in allocator victim selection
- Benchmarks: before/after with hints

**Success criteria:**
- Loop-carried values spill measurably less
- Benchmarks show improvement (any improvement is good!)
- Still correct (verifier still passes)

### Phase 3: Pressure-Aware RVSDG Transforms (Future)

**Goal:** Use pressure information to guide RVSDG optimization.

**This is the NOVEL part - the research contribution!**

Examples:
- **Hoist-or-duplicate decision:** If hoisting extends live range too much, duplicate instead
- **Sink decision:** Move computation into branches to reduce pressure on merge
- **Loop-state reduction:** Aggressively eliminate theta entry ports
- **Rematerialization:** Duplicate pure nodes rather than carry values

```rust
// Example: pressure-aware hoisting
fn should_hoist(node: &Node, target: Region, pressure: &PressureMap) -> bool {
    let current_pressure = pressure.at(node.region);
    let target_pressure = pressure.at(target);

    if target_pressure.gpr > 24 {  // approaching limit (28 on AArch64)
        return false;  // too much pressure already
    }

    let extended_lifetime = estimate_lifetime_extension(node, target);
    if extended_lifetime > 10 {  // arbitrary threshold
        // Long extension - prefer duplicate over hoist
        return false;
    }

    true  // safe to hoist
}
```

**This is where we're BETTER than Cranelift:**
- Cranelift flattens to CLIF before optimization
- We keep RVSDG structure longer
- Can make pressure-aware decisions that preserve structure
- Can use theta/gamma semantics directly

**Deliverable:**
- Pressure estimation for RVSDG regions
- Hoist/sink decisions based on pressure
- Loop-state reduction pass (minimize theta entry ports)
- Benchmarks showing benefit

**Success criteria:**
- Fewer spills in hot loops
- Better code than regalloc2 on deserializer patterns
- Novel optimization strategy (worth publishing?)

### Phase 4: Advanced Features (Optional)

Once simple allocator is proven, consider:

**Coalescing:**
- Eliminate copies where possible
- Use affinity hints from RVSDG
- Still simpler than regalloc2's approach

**Rematerialization:**
- Actually use remat hints from Phase 2
- Recompute cheap values instead of spilling
- Common in deserializers (constants, simple arithmetic)

**Move optimization:**
- Parallel copy resolution (block parameters)
- Swap chains, cycle breaking
- Currently handled by backend, could improve

**Profile-guided allocation:**
- Run interpreter to collect profiles
- Allocate hot vregs to registers
- Deserializers have predictable profiles

## Migration Strategy

### Parallel Development

1. **Keep regalloc2 working** - default path for now
2. **Add regalloc3 behind flag** - `KAJIT_USE_REGALLOC3=1`
3. **Run both in CI** - differential testing
4. **Switch default when confident**
5. **Remove regalloc2** - delete adapter code

### Rollback Plan

If regalloc3 doesn't work out:
- Keep using regalloc2 (it works, just fragile)
- Fix adapter bugs as they arise
- Lessons learned: understand allocator requirements better

### Success Metrics

**Correctness:** (non-negotiable)
- All corpus tests pass
- Differential testing vs regalloc2 (same results)
- Symbolic verification passes

**Performance:** (acceptable if close)
- Within 20% of regalloc2 (Phase 1)
- Within 10% of regalloc2 (Phase 2)
- Equal or better (Phase 3)

**Debuggability:** (the real win)
- Allocation failures are debuggable
- No mysterious adapter bugs
- Can inspect allocation state at any point

## Prior Art & References

### Academic

**VSDG/RVSDG:**
- Reissmann et al., "RVSDG: An Intermediate Representation for Optimizing Compilers" (2020)
- Lawrence, "VSDG: Value State Dependence Graphs" PhD thesis (2007)
  - Section on register allocation and scheduling
  - Johnson's RACM algorithm adapted to VSDG
  - Explicit discussion of phase-order problems

**Register Allocation:**
- Poletto & Sarkar, "Linear Scan Register Allocation" (1999)
  - Simple algorithm, good results for JITs
- Wimmer & Franz, "Linear Scan Register Allocation on SSA Form" (2010)
  - Handles SSA phi functions directly
- Braun & Hack, "Register Spilling and Live-Range Splitting for SSA-Form Programs" (2009)
  - Spill decisions on SSA form

### Industrial

**regalloc2:**
- Ion allocator from SpiderMonkey (backtracking)
- Used in Cranelift/Wasmtime (production quality)
- We're not criticizing it - just addressing impedance mismatch

**LLVM:**
- Multiple allocators: Fast, Basic, Greedy, PBQP
- Post-instruction-selection (matches our plan)
- Extensive machine-specific constraints

**V8:**
- TurboFan uses linear scan for tier 1
- Good enough for JIT use case

## Open Questions (Unresolved)

1. **Should we support multiple register classes in v1?**
   - Decision: Start with just GPR, add SIMD in later phase

2. **How much should we invest in coalescing?**
   - Decision: Measure copy overhead first, add coalescing only if needed

3. **How to handle long functions?**
   - Linear scan is O(n log n) in vreg count
   - Decision: Cross that bridge when we hit it (not a JIT problem yet)

## Resolved Questions

**Should allocator understand ABI directly?** → YES (Rule 5)
- Allocator MUST know about calls, clobbers, and callee-saved registers
- Not a future decision, it's a hard requirement for v1

## Summary

**The pitch:**
> We should not do final physical register allocation at RVSDG level. But given the fragility of the regalloc2 adapter, it is reasonable to build a small native allocator over our own machine-ish MIR with stable IDs. RVSDG should contribute hints and pressure-aware rewrites upstream; MIR should do the real allocation downstream.

**The slogan:**
> **"Do register-allocation-aware optimization in RVSDG; do actual physical-register assignment after lowering."**

**The scope:**
> First goal is not better code than regalloc2. First goal is a correct, understandable allocator that matches our IR and removes adapter bugs.

That's a much saner project than "invent a new allocation paradigm from RVSDG."

---

## Revision History

**2026-03-24 (v4):** Final consistency pass before coding
1. Fixed internal contradictions (Open Question 3 → Resolved, ABI is Rule 5)
2. Made block params/edge args explicit in liveness (defs at entry, uses on edges)
3. Added function entry/exit semantics (precolored defs, fixed uses)
4. Updated verifier to use canonical operand model (no decomposed helpers)
5. Clarified scratch register policy (target/backend contract)
6. Added explicit limitations section (no splitting, no spill coalescing, no multiple homes)

**2026-03-24 (v3):** Locked in the 5 hard rules before coding
1. One canonical operand record (not decomposed trait)
2. Whole-interval allocation only (no splitting in v1)
3. Critical edges split before RA (edge-copy blocks own phi resolution)
4. Explicit scratch register policy (reserved temps)
5. ABI ownership (allocator knows about calls/clobbers/callee-saved)

**2026-03-24 (v2):** Tightened Phase 1 based on feedback
- Added explicit program point model (not InstId!)
- Made operand contract explicit (MachineInst trait)
- Elevated edge copies to first-class subsystem
- Clarified spill insertion needs real strategy
- Made verifier a headline deliverable
- Descoped Phase 1 aggressively (GPR only, no hints)
- Revised timeline to 1-2 weeks (not "2-3 days")
- Updated success criteria (correct > fast)

**Key insights from feedback:**
> "The main trap is not 'can we outsmart regalloc2 with RVSDG hints?' The main trap is: sloppy operand modeling, sloppy program-point modeling, sloppy edge-copy handling, sloppy spill rewriting. If you nail those, this becomes a very sane project."

> "Before coding, lock in these five rules: one canonical operand record type, v1 = whole-interval allocation only, critical edges split before RA, explicit scratch/temp policy, allocator/backend contract for calls/clobbers/callee-saved usage. If you bake those in, Phase 1 stops looking like 'maybe we'll accidentally write a register allocator' and starts looking like an actual plan."

---

**Next steps:**
1. ✅ Roadmap revised with 5 hard rules (you are here)
2. **Ready to start coding** - Phase 1 is now a real plan, not "maybe we'll accidentally write an allocator"
3. Start with Step 1.1 (`MachineOperand` struct + `MachineInst` trait)
4. Build incrementally, verify each step with tests
5. Run symbolic verifier on every commit
6. Measure vs regalloc2 when Phase 1 complete (behavior, not allocation decisions)
