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

### Differential Harness (first step)

For regalloc/backend failures, run differential harnesses before dumps/LLDB.

- RA interpreter vs post-regalloc simulation:
  - `kajit_mir::regalloc_engine::differential_check_program`
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
- `KAJIT_DUMP_STAGES` — comma-separated: `ir`, `linear`, `ra`, `edits`, `opts`, `all`
- `KAJIT_DUMP_FILTER` — substring match on `<format>::<case>` (e.g. `postcard::scalar_u64_v3`)
- `KAJIT_DUMP_DIR` — output directory (default: `target/kajit-stage-dumps`)

Dump files are named `<format>__<case>__<arch>__<stage>.txt`.

Use `opts` stage to see RVSDG snapshots between each optimization pass.

### LLDB debugging of JIT code

Debug JIT-compiled code with source-level stepping through RA-MIR listings:

```bash
scripts/lldb-test.sh json::bool_true_false
```

Set `KAJIT_DEBUG=1` to enable DWARF emission (the helper script does this automatically). This generates:
- RA-MIR listing files at `/tmp/kajit-debug/*.ra-mir`
- DWARF `.debug_line` + `.debug_info` + `.debug_abbrev` in the JIT ELF
- GDB JIT interface registration so LLDB/GDB can discover the code

Full reference: `docs/pipeline-debugging.md` § "LLDB debugging of JIT code"

**Key architecture detail:** Both backends (`aarch64/mod.rs`, `x86_64/mod.rs`) call `set_source_location()` in their instruction emission loops, mapping each linear op index to a DWARF line number. The DWARF sections are built in `jit_dwarf.rs` and attached to the in-memory ELF in `jit_debug.rs`. LLDB requires all three DWARF sections (`.debug_info` with a CU referencing `.debug_line` via `DW_AT_stmt_list`, plus `.debug_abbrev`) — `.debug_line` alone is silently ignored.

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
