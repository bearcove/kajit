# Fix gamma_output_partition compaction

Limits codegen quality now.

## Current state

**File:** `kajit-ir/src/gamma_output_partition.rs` (146 lines)

### The limitation (lines 10-14):
```
//! Limitation: does not compact gamma output tuples. Compaction changes vreg
//! assignments that flow into regalloc hints, triggering a regalloc bug that
//! produces wrong code for vec types. The output rewriting alone creates a
//! cleaner IR for potential future optimization.
```

`rewrite_same_on_both_branches()` (lines 57-146) rewrites output references so consumers use the gamma input directly instead of the gamma output, but leaves the dead output slot in the tuple. The output tuple is never compacted because doing so breaks regalloc hints for vec types.

## What to do

1. Write a text test where compaction is needed (gamma with multiple same-on-both-branches outputs)
2. Root-cause the regalloc hint bug for vec types — the hint system shouldn't depend on output tuple positions
3. Fix the regalloc assumption
4. Enable output tuple compaction
