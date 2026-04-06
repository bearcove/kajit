# Write golden tests for each pass

**Phase:** 0 (do first, after harness)

Write 5-10 golden test cases per pass for the *desired behavior*, not the current implementation. These become the spec.

Passes to cover:
- `const_fold` — `Add(x, 0)`, `Mul(x, 1)`, zigzag constants, etc.
- `dce` — unused node, node used by output, etc.
- `simplify_gamma` — all-passthrough, const predicate
- `simplify_cfg` — (already A-, but pin its behavior)

Also write **negative tests**: what should a pass *not* do?
- `const_fold` should not fold through state edges
- `simplify_gamma` should not inline a branch with side effects
