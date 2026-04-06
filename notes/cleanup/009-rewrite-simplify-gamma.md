# Rewrite simplify_gamma pass

Nice to have. Only implement empty branch coalescing if real programs hit it, otherwise just delete the stub.

## Current state

**File:** `kajit-ir/src/simplify_gamma.rs` (408 lines)

Three transforms exist:

### 1. Constant Predicate Folding (lines 146-211) — WORKING
- If gamma predicate resolves to known constant, inline the selected branch
- Uses `resolve_constant_source()` which handles direct Const nodes and theta loop-invariant args
- Moves selected branch nodes into parent, rewrites RegionArg → gamma inputs, replaces outputs

### 2. All-Passthrough Elimination (lines 213-277) — WORKING
- If every branch produces identical passthrough results (same arg position per output)
- Verifies no side effects, replaces gamma outputs with corresponding inputs, removes node

### 3. Empty Branch Coalescing (lines 334-383) — STUB
- Only debug logging, no actual transforms
- Line 380: `// TODO: Actually implement optimizations for empty branches`

## Pipeline invocation

Runs twice in the default pass pipeline (`ir_passes.rs`):
1. `simplify_trivial_gammas` — early, before unrolling
2. `post_unroll_simplify` — after unroll + const_fold, catches newly-constant predicates

Fixed-point loop: restarts on each change to avoid iterator invalidation.

## Related: gamma_output_partition
- **File:** `kajit-ir/src/gamma_output_partition.rs` (146 lines)
- Eliminates same-on-both-branches outputs, but does NOT compact output tuples
- See 008-fix-gamma-output-partition-compaction.md

## What to do

The cleanup note originally said "delete the stub, implement two clean transforms" — but transforms 1 and 2 already work. Remaining work:

1. **Implement empty branch coalescing** (transform 3) or delete the stub if it's not needed
2. **Add golden text tests** for all three transforms (see 001)
3. **Conservative predicate resolution** (lines 58-66) is documented as intentional — don't change
