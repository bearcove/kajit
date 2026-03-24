# Investigation: Block Merge SSA Violations (2026-03-24)

## Summary

Initially appeared to be a bug in `merge_blocks` optimization, but investigation revealed:
1. **Real culprit**: Interaction between `loop_phi_elim` and `remat` optimizations
2. **Side discovery**: Found and fixed a bug in `merge_blocks` (edge argument copying)
3. **Root cause**: CFG-MIR optimizations can break SSA without any structural safeguards

## Timeline of Discovery

### Initial Symptom
```
cargo test postcard::scalar_u32_v0
→ FAIL: InvalidVarint error
```

### Bisection Process

**Step 1: Identify optimization dependency**
```bash
KAJIT_CFG_OPTS=-all          # PASS
KAJIT_CFG_OPTS=-merge_blocks # PASS
KAJIT_CFG_OPTS=(default)     # FAIL
```
Initial hypothesis: merge_blocks is the problem.

**Step 2: Find compensating optimizations**
Discovered that excluding ANY of these causes failure:
- cse (common subexpression elimination)
- gvn (global value numbering)
- copyprop (copy propagation)
- fuse_cmpz (compare-zero fusion)
- elim_imm (immediate elimination)

Initially thought: "These opts mask merge_blocks' bug"

**Step 3: Test order sensitivity**
```bash
KAJIT_CFG_OPTS=-all,+merge_blocks              # PASS
KAJIT_CFG_OPTS=-all,+cse,+gvn,+merge_blocks    # PASS
KAJIT_CFG_OPTS=(all opts including merge)      # FAIL
```
Discovery: Prior optimizations create a CFG structure that merge_blocks can't handle!

**Step 4: Binary search for minimal failing set**
```bash
# First half
KAJIT_CFG_OPTS=-all,+loop_phi_elim,+remat,+cse,+gvn,+merge_blocks  # FAIL

# Narrow down
KAJIT_CFG_OPTS=-all,+loop_phi_elim,+remat,+merge_blocks            # FAIL

# Test individual
KAJIT_CFG_OPTS=-all,+loop_phi_elim,+merge_blocks                   # PASS
KAJIT_CFG_OPTS=-all,+remat,+merge_blocks                           # PASS
```

**Step 5: The plot twist**
```bash
KAJIT_CFG_OPTS=-all,+loop_phi_elim,+remat      # FAIL (no merge_blocks!)
```

**Conclusion**: merge_blocks is innocent! The bug is in `loop_phi_elim + remat`.

## Bug in merge_blocks (Fixed)

While investigating, found that `merge_blocks` wasn't copying edge arguments when retargeting:

```rust
// Before:
func.edges[edge_into_empty.index()].to = final_target;

// After:
let edge_out_args = func.edges[edge_out_of_empty.index()].args.clone();
func.edges[edge_into_empty.index()].to = final_target;
func.edges[edge_into_empty.index()].args = edge_out_args;
```

**Example**:
```
b1 --e2[]-> b2(empty) --e3[v107=>v88]-> b4

After merging b2 into b1:
b1 --e2[v107=>v88]-> b4  (e2 now has e3's arguments)
```

Without this fix, b4 would receive no arguments even though it expects parameters.

**Note**: This wasn't causing the current test failures, but would cause problems in other cases. The code already had a check to skip merging blocks where edges have arguments, but this fix makes the code more robust for when that check is relaxed in the future.

## Actual Bug: loop_phi_elim + remat

### Minimal Failing Case
```bash
KAJIT_CFG_OPTS=-all,+loop_phi_elim,+remat
→ FAIL: regalloc2 allocation failed: SSA(VReg(vreg = 89, class = Int), Inst(14))
```

### Error Analysis
- `vreg 89` is defined in block b0 as `v89:gpr = const(0x1)`
- Some use at instruction 14 is not dominated by this definition
- loop_phi_elim eliminates some phi parameters
- remat does something that creates the SSA violation

### Compensating Optimizations
Adding ANY of these fixes the SSA violation:
- cse, gvn, copyprop, fuse_cmpz, elim_imm, dce, merge_blocks

This suggests they're accidentally fixing up the broken SSA by:
- Eliminating the problematic use
- Rematerializing the const locally
- Propagating the value differently

### Root Cause (Not Yet Identified)
- loop_phi_elim alone: works
- remat alone: works
- Together: breaks SSA

Need to investigate:
1. What phi parameters does loop_phi_elim remove?
2. What does remat do that assumes those parameters exist?
3. Why doesn't remat validate its assumptions?

## Validation Gaps

### What We Check
- ✅ CFG structure (edges, preds, succs)
- ✅ Terminator consistency
- ✅ Block connectivity

### What We Don't Check
- ❌ SSA properties (dominance, def-before-use)
- ❌ Phi parameter consistency with edges
- ❌ Vreg liveness through blocks
- ❌ Incremental validation during construction

## Lessons Learned

1. **Optimization interactions are complex**: Two passes that work individually can break when combined.

2. **Validation is reactive, not preventive**: We only catch SSA violations when regalloc2 fails. By then, we don't know which optimization broke it.

3. **Debug bisection is essential**: Environment variable controls (`KAJIT_CFG_OPTS`) made it possible to isolate the bug. Without them, this would have been much harder.

4. **Structural guarantees > runtime checks**: The fact that we *can* create SSA violations means our IR design has a fundamental flaw. We need stronger invariants.

5. **False leads are part of investigation**: Spent significant time on merge_blocks before discovering the real culprit. But we found a real bug along the way, so not wasted effort.

## Next Steps

See `ssa-structural-guarantees-roadmap.md` for the plan to make SSA violations structurally impossible.

Immediate priorities:
1. Add SSA validation pass (Phase 1.1)
2. Fix loop_phi_elim + remat bug
3. Run validation after each optimization in debug mode
4. Add SSA validation to fuzzing harness
