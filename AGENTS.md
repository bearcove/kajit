# kajit

Kajit is a JIT deserializer for Rust that generates native machine code at startup from facet type reflection. It walks a type's `Shape` through a multi-stage pipeline: IR (RVSDG with explicit data/side-effect tokens), optimization passes, linearization to `LinearIr`, register allocation via regalloc2, and codegen to aarch64 or x86_64.

## Architecture

Expand backend capabilities to support required IR semantics. Do not work around backend limitations in lowering or intrinsic selection. Prefer explicit backend/ABI support over adapters that hide contract mismatches. If you catch yourself thinking "workaround", stop and implement the proper fix in the backend/compiler/runtime instead.

## Tests

Tests are run with `cargo nextest run`.

On Apple Silicon, x86_64 tests run via Rosetta 2 — no Docker needed:

```
cargo nextest run --target x86_64-apple-darwin
```

The overwhelming majority of tests and benchmarks are generated from `xtask/src/cases.rs`. Don't add test cases by hand — add them there.

Regenerate with `cargo xtask gen`.

Always run tests after making code changes. Run at least the tests directly covering changed code before reporting completion.

## Debugging

Full reference: `docs/pipeline-debugging.md`

### Bisecting with `KAJIT_OPTS`

Disable parts of the pipeline at runtime to isolate bugs. Syntax: comma-separated `+name` / `-name` tokens.

**Top-level switches:**
- `all_opts` — all RVSDG optimization passes (pre-linearization)
- `regalloc` — regalloc edge-edit application during emission

**Per-pass switches** (4 passes, run in this order):
1. `pass.bounds_check_coalescing` — coalesce redundant BoundsCheck chains
2. `pass.theta_loop_invariant_hoist` — hoist loop-invariant setup out of theta bodies
3. `pass.inline_apply` — inline apply/lambda calls
4. `pass.dead_code_elimination` — remove dead nodes and unreachable regions

**Bisect workflow** — when a test fails, narrow the cause:
```bash
# Does it pass with ALL opts disabled? → bug is in an optimization pass
KAJIT_OPTS='-all_opts' cargo nextest run -p kajit --test corpus -E 'test(=the::test)'

# Disable one pass at a time to find the culprit
KAJIT_OPTS='-pass.theta_loop_invariant_hoist' cargo nextest run ...
KAJIT_OPTS='-pass.inline_apply' cargo nextest run ...

# Does it pass with regalloc edits disabled? → bug is in regalloc/spill/reload
KAJIT_OPTS='-regalloc' cargo nextest run ...
```

Print all available options: `KAJIT_OPTS=help cargo nextest run -p kajit --test corpus -E 'test(=any::test)'`

### Stage dumps

Dump pipeline artifacts with environment variables:
- `KAJIT_DUMP_STAGES` — comma-separated: `ir`, `linear`, `ra`, `edits`, `opts`, `all`
- `KAJIT_DUMP_FILTER` — substring match on `<format>::<case>` (e.g. `postcard::scalar_u64_v3`)
- `KAJIT_DUMP_DIR` — output directory (default: `target/kajit-stage-dumps`)

Dump files are named `<format>__<case>__<arch>__<stage>.txt`.

Use `opts` stage to see RVSDG snapshots between each optimization pass.
