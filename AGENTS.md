# kajit

Kajit is a JIT deserializer for Rust that generates native machine code at startup from facet type reflection.

## Installing

After making code changes, install the CLI/MCP binary with:

```bash
cargo xtask install
```

This builds `kajit-cli`, installs it to `~/.cargo/bin/kajit`, and validates the binary. Required before using the MCP debugger tools.

## Pipeline

```
Schema (facet Shape)
  → HIR (kajit-hir)        — statements, locals, loops, if/match
  → IR (kajit-ir)          — RVSDG: theta/gamma nodes, ports, SSA
  → LIR (kajit-lir)        — linearized IR
  → CFG-MIR (kajit-mir)    — control flow graph, VRegs
  → Register Allocation    — regalloc3 (native SSA coloring)
  → Backend                — aarch64 or x86_64 machine code
```

**File extensions:**
- `.k-hir` — HIR text
- `.k-ir` — RVSDG IR text
- `.k-mir` — CFG-MIR text
- `.k-asm` — assembly text

**Why HIR exists:** Multiple frontends need a shared semantic layer. RVSDG is too low-level to be human-readable. HIR preserves names, scopes, comments, spans. HIR is the debugger source view. See `docs/hir-design.md`.

**Key files:**
- `kajit/src/compiler.rs` — HIR generation (`build_postcard_decoder_hir`) and HIR→IR lowering (`build_structural_hir_ir`)
- `kajit-hir/src/lib.rs` — HIR data structures
- `kajit-hir/src/text.rs` — `impl Display for Module` (print HIR with `module.to_string()`)
- `kajit-ir/src/ir.rs` — IR/RVSDG data structures
- `kajit-lir/src/linearize.rs` — IR→LIR
- `kajit-mir/src/cfg_mir.rs` — LIR→CFG-MIR, optimization passes

**Design docs:**
- `docs/hir-design.md` — why HIR exists, what it should/shouldn't do
- `docs/hir-generated-decoder-boundary.md` — what stays above HIR vs inside HIR
- `docs/spec.md` — full specification
- `docs/pipeline-debugging.md` — debugging reference

## Architecture

Expand backend capabilities to support required IR semantics. Do not work around backend limitations in lowering or intrinsic selection. Prefer explicit backend/ABI support over adapters that hide contract mismatches. If you catch yourself thinking "workaround", stop and implement the proper fix in the backend/compiler/runtime instead.

## Generated Code

Never ever patch generated code. Patch the generator or the schema, then regenerate.

Never ever patch generated code. Patch the generator or the schema, then regenerate.

Never ever patch generated code. Patch the generator or the schema, then regenerate.

Never ever patch generated code. Patch the generator or the schema, then regenerate.

Never ever patch generated code. Patch the generator or the schema, then regenerate.

This applies even in the narrow "quick experiment" case: do not patch the generated file. Patch the generator. If the generator is unclear, understand it and patch the generator anyway.

## Tests

Tests are run with `cargo nextest run`.

On Apple Silicon, x86_64 tests run via Rosetta 2 — no Docker needed:

```
cargo nextest run --target x86_64-apple-darwin
```

The overwhelming majority of tests and benchmarks are generated from `xtask/src/cases.rs`. Don't add test cases by hand — add them there.

Regenerate with `cargo xtask gen`.

Always run tests after making code changes. Run at least the tests directly covering changed code before reporting completion.

## CLI (`kajit`)

The `kajit` binary (in `kajit-cli`) is the primary tool for inspecting and debugging the compilation pipeline.

```bash
# Compile a .k-hir file through the full pipeline
cargo run -p kajit-cli -- compile /tmp/test.k-hir
cargo run -p kajit-cli -- compile /tmp/test.k-hir -s cfg
cargo run -p kajit-cli -- compile /tmp/test.k-hir -s hir,ir,cfg,asm

# Dump all pipeline stages for u32 (format decoder)
cargo run -p kajit-cli -- compile-format postcard u32

# Dump specific stages
cargo run -p kajit-cli -- compile-format postcard u32 -s cfg
cargo run -p kajit-cli -- compile-format postcard u32 -s hir,ir,cfg,asm

# Run JIT vs interpreter comparison
cargo run -p kajit-cli -- compile-format postcard u32 -s exec -i 2a

# Dump assembly only
cargo run -p kajit-cli -- compile-format postcard u32 -s asm

# Build standalone debuggable harness
cargo run -p kajit-cli -- compile-format postcard u32 -s harness -i 2a

# Run CFG-MIR reducer (find minimal diverging program)
cargo run -p kajit-cli -- compile-format postcard u32 --reduce differential

# Evaluate a CFG-MIR file with the interpreter
cargo run -p kajit-cli -- eval /tmp/test.cfg-mir 2a

# Lockstep differential debugger (interpreter + LLDB in parallel)
cargo run -p kajit-cli -- debug-diff postcard u32 2a
```

**Stages:** `hir`, `ir`, `opts` (RVSDG pass timeline), `linear`, `cfg`, `asm`/`emit`, `exec` (JIT vs interpreter), `harness`, `all`.

**Supported types:** `u8`, `u16`, `u32`, `u64`, `i8`, `i16`, `i32`, `i64`, `bool`, `String`.

**The `exec` stage** is the go-to for debugging: it runs both the JIT and the CFG-MIR interpreter on the same input and shows a byte-by-byte diff if they diverge. No need for corpus tests or env vars.

## Legacy Notes

Historical and transitional guidance that is still occasionally useful lives in [`/Users/amos/bearcove/kajit/AGENTS-legacy.md`](/Users/amos/bearcove/kajit/AGENTS-legacy.md).

That file is where old deep-debugging workflows, benchmark notes, manual assembly-editing tricks, MCP-specific recipes, and other non-primary guidance should go instead of accreting in this file.
