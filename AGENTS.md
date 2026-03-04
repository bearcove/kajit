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

For a curated regression subset (faster) or full suite:

```
cargo xtask test-x86_64           # regression subset only
cargo xtask test-x86_64 --full    # full suite
```

The overwhelming majority of tests and benchmarks are generated from `xtask/src/cases.rs`. Don't add test cases by hand — add them there.

Regenerate with `cargo xtask gen`.

Always run tests after making code changes. Run at least the tests directly covering changed code before reporting completion.

## Debugging

For pipeline bisecting (`KAJIT_OPTS`) and on-demand stage dumps (`KAJIT_DUMP_STAGES`), see `docs/pipeline-debugging.md`.
