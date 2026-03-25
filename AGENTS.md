# kajit

Kajit is a JIT deserializer for Rust that generates native machine code at startup from facet type reflection.

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

**Why HIR exists:** Multiple frontends (deserializers + Vixen rule language) need a shared semantic layer. RVSDG is too low-level to be human-readable. HIR preserves names, scopes, comments, spans. HIR is the debugger source view. See `docs/hir-design.md`.

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

## Tests

Tests are run with `cargo nextest run`.

On Apple Silicon, x86_64 tests run via Rosetta 2 — no Docker needed:

```
cargo nextest run --target x86_64-apple-darwin
```

The overwhelming majority of tests and benchmarks are generated from `xtask/src/cases.rs`. Don't add test cases by hand — add them there.

Regenerate with `cargo xtask gen`.

Always run tests after making code changes. Run at least the tests directly covering changed code before reporting completion.

## Benchmarks

Benchmarks use a custom harness (not criterion/divan). Output is NDJSON to stdout, human-readable to stderr.

```bash
# List all available benchmarks
cargo bench -p kajit --bench synthetic -- --list

# Run all synthetic benchmarks
cargo bench -p kajit --bench synthetic

# Filter by substring match on bench name
cargo bench -p kajit --bench synthetic -- scalar_u32/postcard

# Real-world JSON datasets
cargo bench -p kajit --bench canada
cargo bench -p kajit --bench twitter
cargo bench -p kajit --bench citm_catalog
```

**Naming convention:** `<case>/<format>/<impl>_<op>`

For example, `scalar_u32/postcard` runs:
- `scalar_u32/postcard/serde_deser` — postcard via serde
- `scalar_u32/postcard/kajit_deser` — postcard via kajit JIT
- `scalar_u32/postcard/serde_ser` — serialization baseline

**Key files:**
- `kajit/benches/harness.rs` — custom benchmark harness (warmup, calibration, percentiles)
- `kajit/benches/synthetic.rs` — generated cases (from `xtask/src/cases.rs`)

**Common cases:**
- `scalar_u32`, `scalar_u64`, `scalar_i32`, etc. — single varint values
- `flat_struct`, `nested_struct`, `deep_struct` — struct nesting
- `vec_scalar_small`, `vec_scalar_large` — vectors of primitives
- `option_u32`, `option_string`, `option_struct` — optional fields

## CLI (`kajit`)

The `kajit` binary (in `kajit-cli`) is the primary tool for inspecting and debugging the compilation pipeline. Use it instead of env-var-driven test dumps.

```bash
# Dump all pipeline stages for u32
cargo run -p kajit-cli -- compile postcard u32

# Dump specific stages
cargo run -p kajit-cli -- compile postcard u32 -s cfg
cargo run -p kajit-cli -- compile postcard u32 -s hir,ir,cfg,asm

# Run JIT vs interpreter comparison
cargo run -p kajit-cli -- compile postcard u32 -s exec -i 2a

# Dump assembly only
cargo run -p kajit-cli -- compile postcard u32 -s asm

# Build standalone debuggable harness
cargo run -p kajit-cli -- compile postcard u32 -s harness -i 2a

# Run CFG-MIR reducer (find minimal diverging program)
cargo run -p kajit-cli -- compile postcard u32 --reduce differential

# Evaluate a CFG-MIR file with the interpreter
cargo run -p kajit-cli -- eval /tmp/test.cfg-mir 2a

# Lockstep differential debugger (interpreter + LLDB in parallel)
cargo run -p kajit-cli -- debug-diff postcard u32 2a
```

**Stages:** `hir`, `ir`, `opts` (RVSDG pass timeline), `linear`, `cfg`, `asm`/`emit`, `exec` (JIT vs interpreter), `harness`, `all`.

**Supported types:** `u8`, `u16`, `u32`, `u64`, `i8`, `i16`, `i32`, `i64`, `bool`, `String`.

**The `exec` stage** is the go-to for debugging: it runs both the JIT and the CFG-MIR interpreter on the same input and shows a byte-by-byte diff if they diverge. No need for corpus tests or env vars.

## Debugging

Full reference: `docs/pipeline-debugging.md`

### Quick Start: Common Workflows

**Compare assembly for different types (e.g., u32 vs i32):**
```bash
# Dump emit stage for both types
KAJIT_DUMP_STAGES=emit,cfg KAJIT_DUMP_DIR=/tmp/kajit-dump cargo nextest run -p kajit --test corpus -E 'test(=postcard::scalar_u32_v3) or test(=postcard::scalar_i32_v3)'

# Compare the assembly
diff /tmp/kajit-dump/postcard__scalar_u32_v3__aarch64__emit.txt /tmp/kajit-dump/postcard__scalar_i32_v3__aarch64__emit.txt

# Compare the CFG-MIR
diff /tmp/kajit-dump/postcard__scalar_u32_v3__aarch64__cfg.txt /tmp/kajit-dump/postcard__scalar_i32_v3__aarch64__cfg.txt
```

**Investigate performance regression:**
```bash
# Dump all stages for a specific case
KAJIT_DUMP_STAGES=all KAJIT_DUMP_FILTER=postcard::scalar_u32 KAJIT_DUMP_DIR=/tmp/kajit-dump cargo nextest run -p kajit --test corpus -E 'test(=postcard::scalar_u32_v3)'

# Files created: postcard__scalar_u32_v3__aarch64__{hir,ir,linear,cfg,emit}.txt
```

**Try a manual optimization (edit assembly by hand):**
```bash
# 1. Dump the emit stage
KAJIT_DUMP_STAGES=emit KAJIT_DUMP_DIR=/tmp/kajit-dump cargo nextest run -p kajit --test corpus -E 'test(=postcard::scalar_u32_v3)'

# 2. Copy to .alt.vixen-asm and edit by hand
cp /tmp/kajit-dump/postcard__scalar_u32_v3__aarch64__emit.txt postcard__scalar_u32_v3__aarch64__emit.alt.vixen-asm
# Edit the file: remove redundant moves, reorder instructions, etc.

# 3. Run test again - it will use your edited assembly
cargo nextest run -p kajit --test corpus -E 'test(=postcard::scalar_u32_v3)'
```

**Compare serde vs kajit optimized assembly:**
```bash
# Run in release mode with LTO (shows fully optimized serde code)
cargo bench -p kajit --bench synthetic -- --dump-asm scalar_u32/postcard
```

### Differential Harness (first step)

For regalloc/backend failures, run differential harnesses before dumps/LLDB.

- Ideal interpreter vs post-regalloc CFG simulation:
  - `kajit_mir::regalloc_engine::differential_check_cfg`
- CFG simulation vs JIT machine code:
  - `kajit::differential_check_linear_ir_vs_jit`

Quick checks:

```bash
cargo nextest run -p kajit-mir -E 'test(regalloc_engine::tests::differential_)'
cargo nextest run -p kajit -E 'test(differential_harness_)'
```

Use the first divergent `step_index`/field to narrow to one specific IR/RA op,
then proceed with `KAJIT_OPTS` bisect and stage dumps.

### Bisecting with `KAJIT_OPTS`

Disable parts of the pipeline at runtime to isolate bugs. Syntax: comma-separated `+name` / `-name` tokens.

**Top-level switches:**
- `all_opts` — all RVSDG optimization passes (pre-linearization)
- `regalloc` — regalloc allocation + edit application during emission
  - disabled (`-regalloc`): skip regalloc and use canonical `vreg -> stackslot` lowering

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

# Does it pass with regalloc disabled? → bug is in regalloc/spill/reload
KAJIT_OPTS='-regalloc' cargo nextest run ...
```

Print all available options: `KAJIT_OPTS=help cargo nextest run -p kajit --test corpus -E 'test(=any::test)'`

### Stage dumps

Dump pipeline artifacts with environment variables:
- `KAJIT_DUMP_STAGES` — comma-separated: `hir`, `ir`, `linear`, `cfg`, `edits`, `opts`, `emit`, `all`
- `KAJIT_DUMP_FILTER` — substring match on `<format>::<case>` (e.g. `postcard::scalar_u64_v3`)
- `KAJIT_DUMP_DIR` — output directory (default: `target/kajit-stage-dumps`)

`KAJIT_DUMP_DIR` accepts absolute or relative paths. Relative paths are resolved
from the test process working directory (often the crate directory), so use an
absolute path if you want dumps in a specific workspace location.

Dump files are named `<format>__<case>__<arch>__<stage>.txt`.

**Gotcha:** Not all types produce dumps. Some types (e.g. u16, i8) may use
a code path that doesn't hit the dump logic. If no dump file appears, try
u32 or u64 instead — those reliably dump. You can also dump the CFG-MIR
before a specific optimization pass by adding a temporary `eprintln!("{cfg}")`
before the pass call in `kajit-mir/src/cfg_mir.rs` (search for the pass name).

**CFG-MIR pass debugging env vars:**
- `KAJIT_CFG_OPTS=-all,-const_phi_elim` — disable ALL CFG-MIR opts plus force-disable const_phi_elim
- `KAJIT_CFG_OPTS=-remat,-cse` — disable specific passes
- `KAJIT_DUMP_BEFORE_PHI_ELIM=1` — dump CFG to stderr before const_phi_elim runs
- `KAJIT_UNROLL=1` — enable bounded-theta unrolling (currently opt-in during development)
- `KAJIT_RA_DEBUG=1` — print SSA coloring allocator debug output (vreg→preg assignments, conflicts)

### Comparing serde vs kajit disassembly

Compare optimized disassembly of serde vs kajit for any benchmark case:

```bash
cargo bench -p kajit --bench synthetic -- --dump-asm scalar_u32
```

This runs in release mode with LTO, so serde code is fully inlined. Output shows both:
- `scalar_u32/postcard/serde_deser` — serde's optimized decode
- `scalar_u32/postcard/kajit_deser` — kajit JIT code

For corpus tests (debug mode, less useful):
```bash
KAJIT_SHOW_ASM=1 cargo nextest run -p kajit --test corpus -E 'test(=postcard::scalar_u32_v0)'
```

Use `opts` stage to see RVSDG snapshots between each optimization pass.

### Manual assembly editing (.alt.vixen-asm)

Test assembly optimizations by hand-editing dumps and having them reassembled:

**Workflow:**
1. **Dump the emit stage** to get the current assembly:
   ```bash
   KAJIT_DUMP_STAGES=emit KAJIT_DUMP_DIR=/tmp/kajit-dump cargo nextest run -p kajit --test corpus -E 'test(=postcard::scalar_u32_v3)'
   # Creates: /tmp/kajit-dump/postcard__scalar_u32_v3__aarch64__emit.txt
   ```

2. **Copy to `.alt.vixen-asm` in the working directory**:
   ```bash
   cp /tmp/kajit-dump/postcard__scalar_u32_v3__aarch64__emit.txt postcard__scalar_u32_v3__aarch64__emit.alt.vixen-asm
   ```

3. **Edit the assembly** by hand:
   - Remove redundant `mov` instructions
   - Reorder instructions to improve ILP
   - Try different register allocations
   - Test peephole optimizations

   The format is the same as the emit dump: each line has address, source location, hex bytes, and disassembly.

4. **Run the test again** — it will automatically use your edited assembly:
   ```bash
   cargo nextest run -p kajit --test corpus -E 'test(=postcard::scalar_u32_v3)'
   ```

   The test harness detects `.alt.vixen-asm` files, reassembles them, and uses the modified code instead of JIT-generated code.

**Use cases:**
- **Prototyping optimizations**: Test if removing certain instructions actually improves performance before implementing the optimization in the compiler
- **Debugging codegen**: Simplify generated code to isolate which instructions cause failures
- **Benchmarking**: Compare hand-optimized assembly against compiler output to measure optimization headroom

**Important:**
- The `.alt.vixen-asm` file must be in the **test process working directory** (usually the crate root, not /tmp)
- File name must exactly match the dump file pattern: `<format>__<case>__<arch>__emit.alt.vixen-asm`
- Preflight validation ensures the assembly parses correctly before execution

### CFG-MIR text format

Canonical CFG-MIR now has a round-trippable text format.

- Render canonical text: `format!("{}", cfg_program)` where `cfg_program: kajit_mir::cfg_mir::Program`
- Parse canonical text: `kajit_mir_text::parse_cfg_mir(&text)`
- Round-trip reference: `kajit-mir-text/src/cfg_mir_parse.rs` tests (`round_trip_cfg_mir_text`)

### LLDB debugging of JIT code

Debug JIT-compiled code with source-level stepping through CFG-MIR listings:

```bash
scripts/lldb-test.sh json::bool_true_false
```

Set `KAJIT_DEBUG=1` to enable DWARF emission (the helper script does this automatically). This generates:
- CFG-MIR listing files at `/tmp/kajit-debug/*.cfg-mir`
- DWARF `.debug_line` + `.debug_info` + `.debug_abbrev` in the JIT ELF
- GDB JIT interface registration so LLDB/GDB can discover the code

Full reference: `docs/pipeline-debugging.md` § "LLDB debugging of JIT code"

**Key architecture detail:** Both backends (`aarch64/mod.rs`, `x86_64/mod.rs`) call `set_source_location()` in their instruction emission loops, mapping each emitted CFG-MIR op (`OpId`) to a DWARF line number in the generated `.cfg-mir` listing. The DWARF sections are built in `jit_dwarf.rs` and attached to the in-memory ELF in `jit_debug.rs`. LLDB requires all three DWARF sections (`.debug_info` with a CU referencing `.debug_line` via `DW_AT_stmt_list`, plus `.debug_abbrev`) — `.debug_line` alone is silently ignored.

### LLDB via MCP (for Claude Code agents)

To debug corpus tests with the LLDB MCP tool:

```
# 1. Build the test binary
cargo test -p kajit --test corpus --no-run 2>&1 | grep Executable
# Output: Executable tests/corpus.rs (target/debug/deps/corpus-HASH)

# 2. Start LLDB session via MCP
lldb_start

# 3. Load the binary (MUST use absolute path)
lldb_command: file /Users/amos/bearcove/kajit/target/debug/deps/corpus-HASH

# 4. Set environment variables
lldb_command: env KAJIT_CFG_OPTS=-all,-const_phi_elim
lldb_command: env KAJIT_DEBUG=1
lldb_command: env KAJIT_DUMP_STAGES=emit
lldb_command: env KAJIT_DUMP_DIR=/tmp/kajit-dump

# 5. Set test filter
lldb_command: settings set target.run-args -- "postcard::scalar_u32_v0" --exact --nocapture

# 6. Run
lldb_command: run
```

The binary hash changes on recompilation. Always re-discover it with `cargo test --no-run`.

## Multi-agent workflow (bud)

Large tasks are delegated to a buddy agent via `bud assign`. The captain (lead agent) stays in conversation with the user, reviews work, commits, and steers.

### Principles

- **Small scope per assignment.** One issue at a time. Don't assign 3 issues in one task.
- **Plan before code.** Tell the buddy to read the code and send a plan (`bud update`) before implementing. Review the plan before giving the go-ahead.
- **Frequent check-ins.** Ask for `bud update` after each milestone. Review diffs before committing.
- **Captain commits.** The buddy writes code; the captain reviews, runs tests, commits, and pushes. Never let the buddy push directly.
- **No fallbacks.** If the buddy adds a "fallback" or "workaround" path, stop them. Find the real bug.
- **Watch the diff.** Before committing, always check `git diff --stat`. If the file count or line count is unexpectedly large, investigate before committing. A differential harness shouldn't touch 21 files.
- **Separate concerns.** If `cargo fmt` touches unrelated files, commit it separately from functional changes.

### Commands

```bash
cat <<'EOF' | bud assign --title "short-title" --issue 42
Task description here.
EOF

cat <<'EOF' | bud assign --keep    # follow-up task, keeps buddy context
EOF

bud list                           # check in-flight tasks
bud spy <id>                       # peek at buddy's pane
cat <<'EOF' | bud steer <id>       # mid-task course correction
EOF
```

### Staleness alerts

If a buddy pane is unchanged for 2 minutes, bud sends a staleness alert to the captain. When this happens, spy on them and steer — they may have "finished" without running `bud respond`.
