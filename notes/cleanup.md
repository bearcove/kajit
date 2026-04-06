You're right, and this is actually the correct instinct for this codebase specifically. Here's why:

## What's actually valuable vs. what isn't

**Keep (the contracts and interfaces):**
- The IR data structures themselves (`IrOp`, `Node`, `Region`, `Block`, `Edge`, `Operand`)
- The text formats (`kajit-hir-text`, `kajit-ir-text`, `kajit-mir-text`) — these are already round-trippable
- The testing infrastructure (differential harness, corpus generation, benchmark harness)
- The backends (B/B+ quality, actually work)
- The pipeline *architecture* (HIR → IR → LIR → CFG-MIR → backends)

**Don't try to save (the pass implementations):**
- A third of them are broken, disabled, or capped by unfixed bugs
- Three dominator implementations, three constant resolvers
- 1,700 lines of dead code
- An inverted spill heuristic
- A "GVN" that's local value numbering with incomplete scaffolding

Refactoring these is harder than rewriting them because you're fighting ghost assumptions from the AI that wrote them. You don't know *why* a particular guard exists — is it papering over a bug in this pass, or in a downstream pass that depends on the broken behavior?

## The "text format first" approach is exactly right

Here's why it's especially good for a compiler:

**1. Text formats are your specification.**

If you can write a `.vixen-ir` file by hand that says "this is the input" and a `.expected` file that says "this is what the pass should produce," you have a test before you have an implementation. You already have this infrastructure — `kajit-ir-text` and `kajit-mir-text` both parse and print. You just need to use them more.

**2. Text tests pin the contract, not the implementation.**

Right now your tests are mostly end-to-end (corpus tests: type in → bytes out). Those are great for catching regressions but terrible for developing passes. If `const_fold` produces wrong output but `dce` happens to clean it up, the corpus test passes. A text-level pass test catches it immediately.

**3. You can develop passes in isolation.**

```
# test: const_fold_add_zero.vixen-ir
input: |
  %1 = Const(0)
  %2 = Add(%x, %1)
expected: |
  # %2 replaced with %x
```

No need to run the full pipeline. No need to generate machine code. Fast iteration.

## Concrete strategy

### Phase 0: Define the testing contract (do this first)

1. **Create a pass test harness** that takes a `.vixen-ir` (or `.cfg-mir`) input, runs one named pass, and compares against `.expected` output. You already have the text parsers. This is maybe 100 lines of test infrastructure.

2. **Write golden tests for each pass you care about.** Not for the current implementation — for the *desired behavior*. What should `const_fold` do with `Add(x, 0)`? What should DCE do with a node that has no consumers? Write 5-10 cases per pass. These become your spec.

3. **Write negative tests.** What should a pass *not* do? `const_fold` should not fold through state edges. `simplify_gamma` should not inline a branch with side effects. These catch the over-eager bugs that AI implementations love to introduce.

### Phase 1: Rewrite passes against the spec

Now you have failing tests. Implement passes one at a time to make them green. Each pass is a pure function: `IR in → IR out`. No global state, no spooky interaction with other passes.

Priority order based on what actually matters for code quality:

1. **`slot2reg`** — keep the existing one, it works despite being ugly. It's too entangled with the IR construction to rewrite independently.
2. **`const_fold`** — existing one is A-, keep it, clean up the duplicate resolver.
3. **`dce`** — rewrite. The current one is O(n²) for no reason. This is a textbook algorithm.
4. **`simplify_gamma`** — rewrite. Delete the stub. Implement two clean transforms.
5. **`simplify_cfg`** — keep. It's the A- pass. It's good.
6. **Everything else** — rewrite as needed against text tests.

**Delete outright:**
- `loop_phi_elim.rs` (482 lines, disabled)
- `linear_scan.rs` (721 lines, dead code)
- `gvn` inline implementation (misleading name, duplicate infrastructure, incomplete)
- `remat` Phase 2 (behind `if false`, violates SSA)
- The duplicate `DomTree` and `compute_dominators` — use `DominanceInfo` everywhere

### Phase 2: Fix the real bugs instead of capping them

Once you have text-level tests, you can actually root-cause the bugs that the caps are hiding:

- **`dead_theta_ports` 3-port cap**: The bug is in port index shifting during multi-port removal. Write a text test with 5 dead ports, fix the indexing, remove the cap.
- **`post_unroll_canonicalize` 800-node cap**: Write a text test with 900 nodes, understand what breaks, fix it.
- **`gamma_output_partition` not compacting**: Write a text test where compaction is needed, fix the regalloc assumption.

### Phase 3: Consolidate shared infrastructure

After the passes work individually:

1. **One `UseLists` / use-def structure**, maintained incrementally, not rebuilt from scratch per pass.
2. **One dominator implementation** (`DominanceInfo`), shared by all passes.
3. **One constant resolver** with configurable conservatism (control vs data role).
4. **One `replace_uses` that's scoped** to a region, not a global scan.

## What this looks like in practice

You'd end up with a directory like:

```
kajit-ir/tests/passes/
  const_fold/
    add_zero.vixen-ir          → add_zero.expected
    mul_one.vixen-ir           → mul_one.expected  
    zigzag_const.vixen-ir      → zigzag_const.expected
    no_fold_through_state.vixen-ir → no_fold_through_state.expected
  dce/
    unused_node.vixen-ir       → unused_node.expected
    used_by_output.vixen-ir    → used_by_output.expected
  simplify_gamma/
    all_passthrough.vixen-ir   → all_passthrough.expected
    const_predicate.vixen-ir   → const_predicate.expected
```

Each test is 10-30 lines. Runs in milliseconds. Documents intent. Catches regressions at the pass level, not the "did the final binary produce the right answer" level.

## The meta-point

You're right that AI is better at greenfield than refactoring. But the key insight is: **you don't need to throw away the architecture to throw away the implementation**. The IR designs, the text formats, the pipeline stages, the test harnesses — those are the architecture. The pass bodies are just implementations of well-defined transforms. Rewrite those against text-level specs and you get a codebase you can actually trust, without losing the structural work that's already done.
