# SSA Structural Guarantees Roadmap

**Goal**: Make SSA violations structurally impossible through IR design, type system, and validation infrastructure.

**Performance impact**: All checks happen at JIT compilation time, zero impact on generated code execution.

## Current State (2026-03-24)

### What We Have
- ✅ CFG structural validation (edge consistency, block connectivity)
- ✅ Block tombstoning (prevents index invalidation)
- ✅ Edge argument copying in merge_blocks (prevents one class of phi mismatches)

### What's Missing
- ❌ SSA property validation (dominance, def-before-use)
- ❌ Phi parameter consistency validation
- ❌ Type-level guarantees against SSA violations
- ❌ Incremental validation during IR construction

### Active Bugs
1. **loop_phi_elim + remat SSA violation** (regalloc2 error: vreg 89 at inst 14)
   - Minimal failing case: `-all,+loop_phi_elim,+remat`
   - Any of {cse, gvn, copyprop, fuse_cmpz, elim_imm, dce, merge_blocks} masks it
   - Root cause: loop_phi_elim eliminates phi params, remat creates SSA violation
   - **Status**: Root cause not yet identified

## Phase 1: Validation Infrastructure (Week 1)

**Goal**: Catch SSA violations immediately after they're created, with clear error messages.

### 1.1 SSA Validation Pass
**Cost**: O(n) in instructions, runs after each optimization in debug mode

Validates:
- Every vreg use has a definition
- Every use is dominated by its definition
- Phi parameters match incoming edge arguments (count and types)
- No dead code creates phantom uses

**Implementation**:
```rust
pub fn validate_ssa(func: &Function) -> Result<(), Vec<SsaError>>
```

Error types:
- `UseWithoutDef { vreg, inst, block }`
- `UseNotDominated { vreg, def_block, use_block }`
- `PhiArgCountMismatch { block, expected, got, edge }`
- `PhiArgMissing { block, param_index, edge }`

### 1.2 Integration Points
- Add `validate_ssa()` call after each opt in `lower_and_optimize()` (debug mode)
- Add `KAJIT_VALIDATE_SSA=1` env var for always-on validation
- Run in all tests and fuzzing

### 1.3 Debug Bisection
When SSA validation fails:
1. Re-run with each opt individually to isolate culprit
2. Dump CFG before/after failing pass
3. Show which vreg/block violates SSA and why

**Acceptance**: loop_phi_elim + remat bug caught with clear error message

## Phase 2: Edge/Phi Consistency (Week 1-2)

**Goal**: Make it impossible to create edges without matching phi parameters.

### 2.1 Runtime Validation During Construction
Add checks when creating/modifying edges:

```rust
impl Function {
    pub fn add_edge(&mut self, from: BlockId, to: BlockId, args: Vec<(VReg, VReg)>) -> EdgeId {
        let target = &self.blocks[to.index()];
        debug_assert_eq!(
            args.len(),
            target.params.len(),
            "edge args {} != block params {} for b{}",
            args.len(), target.params.len(), to.index()
        );
        // ... create edge
    }

    pub fn retarget_edge(&mut self, edge_id: EdgeId, new_target: BlockId) {
        let edge = &self.edges[edge_id.index()];
        let target = &self.blocks[new_target.index()];
        debug_assert_eq!(
            edge.args.len(),
            target.params.len(),
            "retargeting edge e{} to b{}: args {} != params {}",
            edge_id.index(), new_target.index(),
            edge.args.len(), target.params.len()
        );
        // ... update edge
    }
}
```

### 2.2 Builder API for Blocks with Parameters
```rust
pub struct BlockBuilder<'a> {
    func: &'a mut Function,
    block_id: BlockId,
}

impl<'a> BlockBuilder<'a> {
    pub fn with_params(mut self, params: Vec<VReg>) -> Self {
        self.func.blocks[self.block_id.index()].params = params;
        self
    }

    pub fn add_pred(&mut self, from: BlockId, args: Vec<(VReg, VReg)>) -> EdgeId {
        // Validates args match params
        self.func.add_edge(from, self.block_id, args)
    }
}
```

**Acceptance**: Can't create mismatched edges even with bad optimization pass

## Phase 3: Type-Level Guarantees (Week 2-3)

**Goal**: Use Rust's type system to prevent SSA violations at compile time.

### 3.1 Def vs Use Types
```rust
/// A VReg that has been defined (can be used)
pub struct Def<V>(V);

/// A VReg that is being used (must have a Def)
pub struct Use<V>(V);

impl Def<VReg> {
    /// Convert a definition site into a use site
    pub fn as_use(&self) -> Use<VReg> {
        Use(self.0)
    }
}
```

Instruction APIs:
```rust
// Before: any VReg can appear anywhere
pub enum LinearOp {
    Add { dst: VReg, lhs: VReg, rhs: VReg },
}

// After: dst is a Def, operands are Uses
pub enum LinearOp {
    Add { dst: Def<VReg>, lhs: Use<VReg>, rhs: Use<VReg> },
}
```

### 3.2 Phantom Types for Edge Arity
```rust
pub struct Edge<Args> {
    id: EdgeId,
    from: BlockId,
    to: BlockId,
    args: Args,  // type-level arity checking
}

pub struct Block<Params> {
    id: BlockId,
    params: Params,
    // ...
}

// Type system ensures edges and blocks match:
fn add_edge<Args>(from: BlockId, to: Block<Args>, args: Args) -> Edge<Args>
```

**Challenge**: This is a big refactor. May need gradual migration.

**Acceptance**: Common SSA violation patterns caught at compile time

## Phase 4: Dominance Tracking (Week 3-4)

**Goal**: Make dominance relationships explicit, not inferred.

### 4.1 Cache Dominance Info
```rust
pub struct FunctionWithDominance {
    pub func: Function,
    pub dom: DominanceInfo,
}

impl FunctionWithDominance {
    /// Invalidate dominance after CFG changes
    pub fn invalidate_dominance(&mut self) {
        self.dom = DominanceInfo::empty();
    }

    /// Recompute dominance if needed
    pub fn ensure_dominance(&mut self) {
        if self.dom.is_empty() {
            self.dom = DominanceInfo::compute(&self.func);
        }
    }
}
```

### 4.2 Dominance-Aware APIs
```rust
impl FunctionWithDominance {
    /// Add a use of a vreg, checking dominance
    pub fn add_use(&mut self, vreg: VReg, in_block: BlockId) -> Result<(), DominanceError> {
        let def_block = self.func.find_def(vreg)?;
        self.ensure_dominance();

        if !self.dom.dominates(def_block, in_block) {
            return Err(DominanceError { vreg, def_block, use_block: in_block });
        }

        // ... add use
    }
}
```

**Cost**: Dominance computation is O(n log n), but only runs when CFG structure changes.

**Acceptance**: Can't add uses that violate dominance

## Phase 5: Builder Pattern Refactor (Week 4-5)

**Goal**: All IR construction goes through builders that maintain invariants.

### 5.1 CFG Builder
```rust
pub struct CfgBuilder {
    blocks: Vec<BlockBuilder>,
    edges: Vec<Edge>,
}

impl CfgBuilder {
    pub fn new_block(&mut self) -> BlockBuilder { /* ... */ }

    pub fn finalize(self) -> Result<Function, BuildError> {
        // Final SSA validation
        validate_ssa(&self)?;
        Ok(Function { blocks: self.blocks, edges: self.edges })
    }
}
```

### 5.2 Optimization Pass Trait
```rust
pub trait CfgOptimization {
    fn apply(&self, func: &mut Function) -> Result<bool, OptError>;
}

pub fn run_optimization<O: CfgOptimization>(
    func: &mut Function,
    opt: O,
) -> Result<bool, OptError> {
    let changed = opt.apply(func)?;

    #[cfg(debug_assertions)]
    validate_ssa(func).map_err(|errs| OptError::SsaViolation(errs))?;

    Ok(changed)
}
```

**Acceptance**: All opts go through validated APIs, can't bypass checks

## Success Metrics

1. **Current bug fixed**: loop_phi_elim + remat passes all tests
2. **No more silent SSA violations**: All violations caught with clear errors
3. **Zero execution overhead**: Validation is JIT-time only
4. **Type-safe by default**: Common mistakes prevented at compile time
5. **Fuzz-resistant**: Fuzzing finds no SSA violations even with random opt combinations

## Non-Goals (For Now)

- RVSDG-style regions in CFG-MIR (too big a refactor, would need new regalloc interface)
- Arena-based stable IDs (nice-to-have, but tombstoning works)
- Effect system for tracking side effects (future work)

## Migration Strategy

1. **Phase 1-2** can be done independently, low risk
2. **Phase 3** (type-level) needs gradual migration:
   - Add `Def`/`Use` wrappers alongside existing `VReg`
   - Migrate one pass at a time
   - Use newtype pattern so conversion is zero-cost
3. **Phase 4-5** build on earlier phases

Each phase delivers value independently. Can stop at any phase if cost/benefit doesn't justify continuing.
