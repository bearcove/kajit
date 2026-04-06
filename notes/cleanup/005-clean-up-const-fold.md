# Clean up const_fold

Small, well-scoped cleanup.

## Current state

**File:** `kajit-ir/src/const_fold.rs`

The pass is A- quality. Two folding strategies:
1. **Full folding** (`try_fold_node`, lines 117-168): Evaluate pure ops with all-constant inputs
2. **Algebraic simplification** (`try_simplify_node`, lines 178-260): Identity/absorbing rules (x+0→x, x*0→0)

Folds: Add, Sub, Mul, And, Or, Xor, Shl, Shr, Sar, all comparisons, ZigzagDecode, SignExtend.

## The duplicate constant resolver problem

Three resolvers exist:

### 1. `resolve_to_constant` (const_fold.rs, line 288) — PRIMARY
- Full graph tracing: Const nodes, ReadFromSlot→WriteToSlot chains, gamma branches (all agree), loop-invariant theta args
- Depth-limited (32) to prevent infinite recursion
- Used by: const_fold internals, unroll_theta (lines 349, 610)

### 2. `resolve_to_const` (dead_theta_ports.rs, line 362) — MINIMAL
- Only handles direct `Const` node. No gamma/theta tracing.
- Used by: dead_theta_ports (line 143)
- Acceptable as-is for this module's narrow use case

### 3. `resolve_to_const_value` (slot2reg.rs, line 919) — DUPLICATE
- Identical to #2 (copy-paste)
- Used by: slot2reg (lines 810, 848, 871)
- **Should be consolidated** with #2

### 4. `resolve_to_constant_skip_errors` (const_fold.rs, line 441) — DEAD CODE
- Same as #1 but skips error-exit gamma branches
- **Zero call sites** — delete it (see 002)

## What to do

1. Delete `resolve_to_constant_skip_errors` and its helper `region_has_error_exit` (lines 441-531) — never called
2. Factor out the simple `resolve_to_const` into a shared utility (or have slot2reg import from dead_theta_ports)
3. Keep `resolve_to_constant` as the primary full resolver
4. Add golden text tests (see 001)
