# Create a pass test harness

**Phase:** 0 (do first)

Create a test harness that takes a `.vixen-ir` (or `.cfg-mir`) input, runs one named pass, and compares against `.expected` output.

The text parsers already exist (`kajit-ir-text`, `kajit-mir-text`). This is ~100 lines of test infrastructure.

## Shape

```
kajit-ir/tests/passes/
  const_fold/
    add_zero.vixen-ir          → add_zero.expected
    mul_one.vixen-ir           → mul_one.expected
    no_fold_through_state.vixen-ir → no_fold_through_state.expected
  dce/
    unused_node.vixen-ir       → unused_node.expected
    used_by_output.vixen-ir    → used_by_output.expected
  simplify_gamma/
    all_passthrough.vixen-ir   → all_passthrough.expected
    const_predicate.vixen-ir   → const_predicate.expected
```

Each test is 10-30 lines, runs in milliseconds, documents intent, catches regressions at the pass level.
