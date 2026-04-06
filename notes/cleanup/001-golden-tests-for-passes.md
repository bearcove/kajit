# Write golden tests for each pass

**Phase:** 0 (do first, after harness)

## Strategy

Write 5-10 golden test cases per pass for the *desired behavior*, not the current implementation. These become the spec. Also write negative tests for what passes should NOT do.

## IR-level passes (`.vixen-ir` → `.expected`)

### const_fold (`kajit-ir/src/const_fold.rs`)
Current implementation is A- quality (408 lines). Folds:
- Pure arithmetic: Add, Sub, Mul, And, Or, Xor, Shl, Shr, Sar
- All comparisons
- ZigzagDecode, SignExtend

**Positive tests:**
- `add_zero.vixen-ir` — `Add(x, 0)` → `x`
- `mul_one.vixen-ir` — `Mul(x, 1)` → `x`
- `mul_zero.vixen-ir` — `Mul(x, 0)` → `Const(0)`
- `zigzag_const.vixen-ir` — `ZigzagDecode(Const(4))` → `Const(2)`
- `chained_const.vixen-ir` — `Add(Const(1), Const(2))` → `Const(3)`

**Negative tests:**
- `no_fold_through_state.vixen-ir` — should not fold ops with state edges
- `no_fold_impure.vixen-ir` — should not fold non-pure ops

### simplify_gamma (`kajit-ir/src/simplify_gamma.rs`)
Three transforms exist:
1. Constant predicate folding (lines 146-211) — WORKING
2. All-passthrough elimination (lines 213-277) — WORKING
3. Empty branch coalescing (lines 334-383) — STUB (only logging)

**Positive tests:**
- `const_predicate.vixen-ir` — gamma with Const(0) pred → inline branch 0
- `all_passthrough.vixen-ir` — all branches return same arg → eliminate gamma
- `passthrough_subset.vixen-ir` — some outputs are passthrough, some aren't

**Negative tests:**
- `no_fold_side_effects.vixen-ir` — don't inline branches with side effects
- `no_fold_through_gamma_args.vixen-ir` — conservative: branch-local const may not be safe as control pred

### dead_theta_ports (`kajit-ir/src/dead_theta_ports.rs`)
**Positive tests:**
- `single_dead_port.vixen-ir` — one constant loop-carried port removed
- `three_dead_ports.vixen-ir` — verify current 3-port cap works
- `five_dead_ports.vixen-ir` — will fail until 006 is done (bug in port shifting)

### dce (currently at `kajit-ir/src/ir_passes.rs` for RVSDG level)
**Positive tests:**
- `unused_node.vixen-ir` — node with no consumers removed
- `used_by_output.vixen-ir` — node used by region output survives
- `chain_of_dead.vixen-ir` — A→B→C all dead, all removed

## MIR-level passes (`.cfg-mir` → `.expected`)

### eliminate_dead_block_params (`kajit-mir/src/opt/dce.rs`)
- `dead_block_param.cfg-mir` — unused block param removed along with edge args
- `chain_dead_params.cfg-mir` — chained dead dependencies eliminated in one pass

### const_branch_fold (`kajit-mir/src/opt/const_branch_fold.rs`)
- `const_true_branch.cfg-mir` — branch on Const(1) → unconditional jump
- `const_false_branch.cfg-mir` — branch on Const(0) → unconditional jump

### simplify_cfg (`kajit-mir/src/opt/simplify_cfg.rs`)
- `empty_block.cfg-mir` — empty block with single successor merged
- `redundant_jump.cfg-mir` — jump-to-jump collapsed
