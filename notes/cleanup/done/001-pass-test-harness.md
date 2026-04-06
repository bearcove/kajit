# Create a pass test harness

IR harness already exists. The gap is MIR-level pass tests.

## What already exists

A `datatest-stable` harness already exists at `kajit-ir-text/tests/ir_transform.rs` that does exactly this pattern for IR-level passes:
- Input `.vixen-ir` files parsed → pass applied → output compared against `.expected`
- Post-pass `verify()` call validates IR integrity
- Current pass registry (lines 5-15):
  - `slot2reg` → `kajit_ir::slot2reg::slot_to_reg(func)`
  - `unroll_const_fold` → unroll + const_fold + simplify_gamma composed

Existing test cases:
- `kajit-ir-text/tests/slot2reg/` — 5+ tests (write_then_read, slot_read_inside_gamma, etc.)
- `kajit-ir-text/tests/unroll_const_fold/` — 3+ tests (counter_mul, runtime_pred_const_shift, varint_mul_reduced)

Round-trip tests also exist:
- `kajit-ir-text/src/ir_parse.rs` (lines 1739+) — IR text round-trip
- `kajit-mir-text/src/cfg_mir_parse.rs` (lines 1088+) — MIR text round-trip

## What's missing

1. **No MIR-level text pass tests.** All MIR tests are round-trip or integration. Need a `datatest-stable` harness for CFG-MIR passes (const_branch_fold, dce, simplify_cfg, etc.) using `.cfg-mir` input files and `.expected` output.

2. **No test directories for most IR passes.** Only `slot2reg` and `unroll_const_fold` have test cases. Missing: `const_fold`, `dce`, `simplify_gamma`, `dead_theta_ports`, `gamma_output_partition`.

3. **No pipeline composition testing.** No structured way to test sequences of passes with intermediate snapshots.

## Work to do

1. Add new pass names to the registry in `ir_transform.rs` (trivial — just add match arms for `const_fold`, `simplify_gamma`, `dce`, `dead_theta_ports`)
2. Create a parallel `cfg_mir_transform.rs` harness in `kajit-mir-text/tests/` for MIR-level passes
3. Create test directories per pass under `kajit-ir-text/tests/` and `kajit-mir-text/tests/`

## Key files

- `kajit-ir-text/tests/ir_transform.rs` — existing IR pass test harness
- `kajit-ir-text/Cargo.toml` — `datatest-stable` dependency
- `kajit-mir-text/src/cfg_mir_parse.rs` — MIR text parser (for new harness)
- `kajit-mir/src/cfg_mir.rs:lower_and_optimize()` — MIR pass orchestration
