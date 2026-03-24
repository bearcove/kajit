# Optimization Roadmap

## Current State (2026-03-24)

### Performance Gap (regalloc3 native backend)
- **Kajit**: 5.1 ns/op for scalar u32 varint decode (regalloc3 native backend)
- **Kajit (regalloc2, historical)**: 3.5 ns/op — regalloc2 path is currently broken due to adapter fragility, not actively maintained
- **Serde**: 0.7 ns/op (7x faster than current kajit)
- **Primary bottleneck**: RVSDG-level structure — 32 blocks for a varint decoder, constant-valued branches that should be folded upstream before reaching the backend

### Structural Issues

**scalar_u32 CFG-MIR metrics:**
- 125 vregs total
- 39 basic blocks (35 are empty forwarding blocks - 90%)
- Loop header has **11 parameters**
- 30+ mov instructions in emitted assembly

**Loop header parameter breakdown:**
```
edge e0: b0 -> b1 [
  v86=>v0,   # cursor
  v87=>v2,   # length
  v88=>v3,   # const 0 (redundant!)
  v89=>v3,   # const 0 (redundant!)
  v90=>v3,   # const 0 (redundant!)
  v91=>v3,   # const 0 (redundant!)
  v92=>v3,   # const 0 (redundant!)
  v93=>v3,   # const 0 (redundant!)
  v94=>v83,  # const 1
  v95=>v83,  # const 1 (duplicate!)
  v96=>v85   # const 1
]
```

6 parameters are constant zero, 2 are duplicate const 1. Only 3 parameters are actually loop-variant!

### Serde's Advantage

Serde (via LLVM) uses completely different optimizations:
- Loop unrolling (4 iterations inline)
- Bit field insert (`bfi`) instead of shift+mask+OR
- Test-and-branch (`tbz`) instead of AND+compare+branch
- No actual loop in fast path

## Root Causes

### 1. Excessive Phi Parameters
Loop headers carry 11 values per iteration, but most are loop-invariant constants. This creates:
- Unnecessary live ranges through the loop
- Register pressure (forces spills)
- Mov instructions to satisfy phi edges

### 2. Empty Forwarding Blocks
RVSDG gamma nodes create empty blocks that just pass values through:
```
block bN params=[v1, v2] insts=[] term=branch -> bM
```
These exist solely to merge control flow from RVSDG lowering.

### 3. Lack of Analysis Infrastructure
We have no:
- Dominance tree (can't safely eliminate phis)
- Loop analysis (can't identify loop-invariant values)
- Def-use chains (can't trace value flow)
- Liveness analysis (can't eliminate dead code safely)

### 4. Conservative Optimization Passes
Current passes are too conservative:
- Gamma simplification: only eliminated 1/33 gammas
- Phi simplification: disabled due to SSA violations
- Block merging: skeleton only, not implemented
- Copy propagation: doesn't handle phi parameters

## The Plan

### Phase 1: Analysis Infrastructure

Build the data structures that enable safe, aggressive optimization.

**1.1 Dominance Analysis** (`kajit-mir/src/analysis/dominance.rs`)
```rust
pub struct DominanceInfo {
    /// Immediate dominator for each block
    idom: HashMap<BlockId, BlockId>,
    /// Dominance tree (children of each block)
    dom_tree: HashMap<BlockId, Vec<BlockId>>,
    /// Dominance frontiers for SSA construction
    dom_frontier: HashMap<BlockId, HashSet<BlockId>>,
}

impl DominanceInfo {
    pub fn compute(func: &Function) -> Self;
    pub fn dominates(&self, a: BlockId, b: BlockId) -> bool;
    pub fn strictly_dominates(&self, a: BlockId, b: BlockId) -> bool;
}
```

**1.2 Loop Analysis** (`kajit-mir/src/analysis/loops.rs`)
```rust
pub struct LoopInfo {
    /// Map from loop header to loop body blocks
    loops: HashMap<BlockId, LoopData>,
}

pub struct LoopData {
    pub header: BlockId,
    pub body: HashSet<BlockId>,
    pub backedges: Vec<EdgeId>,
    pub exits: Vec<EdgeId>,
    pub depth: usize, // for nested loops
}

impl LoopInfo {
    pub fn compute(func: &Function, dom: &DominanceInfo) -> Self;
    pub fn is_loop_header(&self, block: BlockId) -> bool;
    pub fn loop_for_block(&self, block: BlockId) -> Option<&LoopData>;
}
```

**1.3 Def-Use Chains** (`kajit-mir/src/analysis/defuse.rs`)
```rust
pub struct DefUseInfo {
    /// For each vreg, where it's defined
    defs: HashMap<VReg, DefSite>,
    /// For each vreg, all use sites
    uses: HashMap<VReg, Vec<UseSite>>,
}

pub enum DefSite {
    Inst(BlockId, InstId),
    BlockParam(BlockId, usize),
    FuncArg(usize),
}

pub struct UseSite {
    pub block: BlockId,
    pub inst: Option<InstId>, // None for terminator/edge
    pub operand_idx: usize,
}

impl DefUseInfo {
    pub fn compute(func: &Function) -> Self;
    pub fn uses_of(&self, vreg: VReg) -> &[UseSite];
    pub fn single_use(&self, vreg: VReg) -> Option<&UseSite>;
}
```

### Phase 2: Loop-Invariant Phi Elimination

Eliminate phi parameters that are loop-invariant.

**Pass**: `eliminate_loop_invariant_phis` (`kajit-mir/src/opt/loop_phi_elim.rs`)

**Algorithm**:
```
for each loop header H with params [p1, p2, ..., pN]:
  for each parameter pi:
    collect all values assigned to pi on edges into H
    if all values are identical (or equal to pi itself):
      # This parameter is loop-invariant
      let invariant_value = that value

      # Replace all uses of pi with invariant_value
      for each use of pi:
        replace pi with invariant_value

      # Remove parameter from header
      remove pi from H.params

      # Update all edges into H
      for each edge E into H:
        remove argument at position i
```

**Expected impact**: 11 → 3 loop header parameters

**Safety**: Requires dominance analysis to verify invariant_value dominates all uses.

### Phase 3: Empty Block Merging

Merge empty forwarding blocks into their predecessors.

**Pass**: `merge_empty_blocks` (`kajit-mir/src/opt/block_merge.rs`)

**Algorithm**:
```
repeat until no changes:
  for each block B:
    if B has single predecessor P
    and B has no instructions
    and P has single successor B:
      # Merge B into P

      # Replace P's terminator with B's terminator
      P.term = B.term

      # Update B's successors to point to P
      for each successor S of B:
        replace references to B with P in S.preds

        # Update phi arguments
        if S has params:
          for edge E: B -> S:
            let values = E.args
            find edge E': P -> B
            let incoming = E'.args

            # Create new edge P -> S
            create edge with args = resolve(values, incoming)
```

**Expected impact**: 39 → ~10 blocks (eliminate 29 empty blocks)

**Complexity**: Phi parameter resolution is tricky when blocks have parameters.

**Alternative simpler version**: Only merge blocks with zero parameters.

### Phase 4: Copy Propagation + Dead Code Elimination

Re-run with better infrastructure.

**4.1 Enhanced Copy Propagation**

Current pass handles instruction operands but not phi parameters well.

**Enhancement**:
- Use def-use chains to find all uses
- Handle phi parameters specially
- Iterate to fixed point

**4.2 Aggressive Dead Code Elimination**

**Pass**: `eliminate_dead_code` (`kajit-mir/src/opt/dce.rs`)

**Algorithm**:
```
# Mark phase
let mut live = HashSet::new()
let mut worklist = Vec::new()

# Initially mark side-effecting operations
for inst in all_instructions:
  if inst.has_side_effects():
    live.insert(inst)
    worklist.push(inst)

# Propagate liveness backwards
while let Some(inst) = worklist.pop():
  for operand in inst.operands:
    let def = defuse.def_of(operand)
    if !live.contains(def):
      live.insert(def)
      worklist.push(def)

# Sweep phase
for inst in all_instructions:
  if !live.contains(inst):
    remove inst

# Remove unused phi parameters
for block in all_blocks:
  for param_idx in (0..block.params.len()).rev():
    let vreg = block.params[param_idx]
    if defuse.uses_of(vreg).is_empty():
      # Remove parameter
      remove block.params[param_idx]
      for edge into block:
        remove edge.args[param_idx]
```

**Expected impact**: Eliminate unused copy instructions, reduce phi parameters further.

### Phase 5: Backend Optimizations

After CFG is clean, improve instruction selection.

**5.1 Bit Test Peephole** (`kajit/src/backends/aarch64/peephole.rs`)

Pattern: `(x & (1 << k)) == 0` → `tbz x, k`
Pattern: `(x & (1 << k)) != 0` → `tbnz x, k`

**5.2 Bit Field Insert**

Pattern: `(acc & ~mask) | ((val & mask2) << shift)` → `bfi acc, val, shift, width`

**5.3 Redundant Mov Elimination**

Post-regalloc pass:
- Track register contents
- Eliminate `mov x1, x1`
- Eliminate `mov x2, x1` followed by use of x1 where x2 would work

### Phase 6: Loop Unrolling (Optional)

For small, predictable loops (like varint decode):
- Unroll 4 iterations
- Enable better instruction scheduling
- Reduce branch mispredicts

**This is lower priority** - the structural cleanups should get us most of the way there.

## Implementation Order

### Sprint 1: Analysis Foundation
1. Dominance analysis
2. Loop detection
3. Def-use chains

**Success metric**: Can query "does X dominate Y?" and "is B a loop header?"

### Sprint 2: Phi Elimination
1. Implement loop-invariant phi elimination
2. Test on scalar_u32

**Success metric**: Loop header has 3 parameters instead of 11

### Sprint 3: Block Merging
1. Implement empty block merging (simple version: zero-parameter blocks only)
2. Test on scalar_u32

**Success metric**: 39 blocks → ~15 blocks (at least 50% reduction)

### Sprint 4: Cleanup
1. Re-run copy propagation
2. Aggressive dead code elimination

**Success metric**: No unused copy instructions in CFG

### Sprint 5: Backend
1. tbz/tbnz peephole
2. Post-regalloc mov elimination

**Success metric**: Fewer than 10 mov instructions in emitted assembly

## Success Metrics

### Target Performance
- **Stretch goal**: 1.0 ns/op (close to serde's 0.7 ns)
- **Realistic goal**: 1.5 ns/op (2.3x improvement)
- **Minimum goal**: 2.5 ns/op (1.4x improvement)

### Structural Targets
- Loop header: 11 → 3 parameters (73% reduction)
- Blocks: 39 → 15 (62% reduction)
- Mov instructions: 30+ → 10 (67% reduction)
- Code size: 424 bytes → 200 bytes (53% reduction)

### Pass Validation
Each optimization pass must:
1. Preserve semantics (differential harness passes)
2. Maintain SSA invariants (regalloc2 accepts output)
3. Actually improve metrics (measure before/after)

## Non-Goals

### What We're NOT Doing

**Domain-specific pattern matching**: No special "varint decoder" or "json parser" patterns. Build general-purpose optimizations that help all code.

**LLVM-level sophistication**: We're not trying to match LLVM's 30 years of optimization passes. Focus on the highest-impact optimizations for JIT code.

**Perfect codegen**: We're building a JIT, not a static compiler. Some overhead is acceptable. The goal is "good enough" not "perfect".

## References

- **SSA Book**: "SSA-based Compiler Design" (Rastello & Bouchez Tichadou)
  - Chapter 9: Loop optimizations
  - Chapter 3: Dominance and SSA construction

- **LLVM**: Study LLVM's loop passes
  - LoopSimplify.cpp (canonicalizes loops)
  - LICM.cpp (loop-invariant code motion)
  - SimplifyCFG.cpp (merges blocks)

- **regalloc2**: Understand SSA invariants
  - Dominance requirements
  - Critical edge splitting

## Appendix: Why This Approach Will Work

### Previous Failures
1. **Gamma simplification at RVSDG level**: Only caught trivial cases (1/33). The redundancy emerges during CFG construction.

2. **Phi simplification without dominance**: Broke SSA by replacing phis with values that don't dominate uses. Need proper analysis.

3. **Block merging skeleton**: Never implemented the hard part (phi resolution). This roadmap includes the full algorithm.

### This Approach Is Different

1. **Infrastructure first**: Build the tools (dominance, loops, def-use) before attempting optimizations.

2. **Incremental validation**: Test each pass independently with differential harness.

3. **Realistic scope**: Focus on 3-4 high-impact passes, not 30 tiny ones.

4. **Clear metrics**: Every change must show measurable improvement.

5. **One invariant at a time**: Don't try to fix SSA and merge blocks simultaneously. Do analysis passes, then transformation passes.

This is a real compiler project. It's not a weekend hack. But it's tractable - maybe 2-3 weeks of focused work across the 5 sprints.
