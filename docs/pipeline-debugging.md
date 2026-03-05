# Pipeline debugging

## Choosing the artifact

Use the latest representation that still reproduces the bug.

- RVSDG IR text: use when the bug disappears after re-lowering or you need to
  change high-level control/dataflow before linearization.
- CFG-MIR text: use for regalloc, backend, JIT, and codegen bugs. This is the
  preferred artifact for differential checking, minimization, and LLDB.
- Stage dumps: use when you still need to discover where the bad transform is
  introduced.
- LLDB: use after differential checking has narrowed the problem to one exact
  transition and you need machine-code inspection.

As a rule: do not start from LLDB if a differential oracle can already tell you
which field diverged first.

## Debugging decision tree

1. Reproduce the failure with one exact test and one exact input.
2. Run with `KAJIT_OPTS='-regalloc'`.
   - If the failure disappears, the bug is in CFG-MIR allocation/edit handling
     or backend use of allocated locations.
   - If it remains, the bug is earlier in the pipeline or in canonical
     stackslot lowering.
3. Run the differential harness.
   - Allocator/simulation mismatch: investigate CFG-MIR, allocation, edits, and
     post-allocation execution.
   - CFG/JIT mismatch: investigate backend emission or runtime/JIT plumbing.
4. Dump stages only after you know which phase is suspect.
5. Freeze the reproducer as CFG-MIR text.
6. Minimize that CFG-MIR program against the same concrete input.
7. Use LLDB or disassembly on the minimized reproducer if needed.

## Bisecting with `KAJIT_OPTS`

Selectively disable parts of the compile pipeline at runtime without rebuilding.

Syntax: comma-separated tokens. `+name` enables, `-name` disables, bare `name` is treated as `+name`. Unknown names fail fast.

Options:
- `all_opts` — default RVSDG optimization passes before linearization
- `regalloc` — regalloc allocation + edit application during backend emission
  - disabled (`-regalloc`): skip regalloc entirely and use canonical `vreg -> stackslot` lowering

Per-pass options:
- `pass.bounds_check_coalescing`
- `pass.theta_loop_invariant_hoist`
- `pass.inline_apply`
- `pass.dead_code_elimination`

Examples:

```bash
# disable all optimization passes
KAJIT_OPTS='-all_opts' cargo nextest run -p kajit <test_filter>

# disable regalloc entirely (canonical vreg->stack lowering)
KAJIT_OPTS='-regalloc' cargo nextest run -p kajit <test_filter>

# disable a single pass
KAJIT_OPTS='-pass.inline_apply' cargo nextest run -p kajit <test_filter>
```

Show built-in help:

```bash
KAJIT_OPTS=help cargo nextest run -p kajit <test_filter>
```

## Differential harness (use first for regalloc/backend bugs)

Before stage dumps or LLDB, run the differential harness to find the first
semantic divergence.

- Ideal interpreter vs post-regalloc CFG simulation (allocator/edit correctness):
  - `kajit_mir::regalloc_engine::differential_check_cfg`
- CFG simulation vs JIT machine code (backend/codegen correctness):
  - `kajit::differential_check_linear_ir_vs_jit`

Fast sanity commands:

```bash
# allocator/simulation differential harness tests
cargo nextest run -p kajit-mir -E 'test(regalloc_engine::tests::differential_)'

# JIT differential harness tests
cargo nextest run -p kajit -E 'test(differential_harness_)'
```

For a failing corpus case, build Linear IR for the same shape/input and run
both harnesses in a focused test:

1. `let linear = kajit::debug_linear_ir(shape, decoder)`
2. `kajit::differential_check_linear_ir_vs_jit(&linear, input)`
3. (optional CFG simulation check)
   - `let cfg = kajit_mir::cfg_mir::lower_linear_ir(&linear);`
   - `let alloc = kajit_mir::regalloc_engine::allocate_cfg_program(&cfg)?;`
   - `kajit_mir::regalloc_engine::simulate_execution_cfg(&alloc, input)?`

These report the first divergent `step_index` and field (`position`, `cursor`,
`trap`, `returned`, or `output`) so you can target one exact transition.

### What the differential result means

The harness is not only answering "did this fail?". It gives a stable
interestingness predicate for minimization and debugging.

A useful divergence signature is:
- divergent field
- ideal trap vs post-allocation/post-JIT trap
- whether either side returned
- first divergent step index

When minimizing, preserve the same divergence class. Do not accept a smaller
program that merely fails in some unrelated way.

## CFG-MIR text format

Canonical CFG-MIR now supports a round-trippable text format.

- Print canonical text from `kajit_mir::cfg_mir::Program`:
  - `let text = format!("{cfg_program}");`
- Parse it back:
  - `let cfg_program = kajit_mir_text::parse_cfg_mir(&text)?;`

The parser is strict and validates function/block/edge/inst/term IDs and CFG invariants.

### How to get seed CFG-MIR text

Use one of these sources:

- Programmatic dump from a shape/decoder pair:
  - `kajit::debug_cfg_mir_text(shape, decoder)`
- Paired RVSDG + CFG-MIR dump for a test fixture:
  - `kajit::debug_ir_and_cfg_mir_text(shape, decoder)`
- On-demand corpus dump via `KAJIT_DUMP_STAGES='cfg'`

Prefer canonical CFG-MIR text over ad-hoc handwritten snippets once you are
investigating a real bug. It is easier to round-trip, minimize, and replay.

## Minimizing a codegen bug from text

Use text entrypoints to reproduce and shrink failures without rebuilding from a
Rust type each time.

### Workflow A: RVSDG IR text -> compile -> run

```rust
let registry = kajit::ir::IntrinsicRegistry::new();
let decoder = kajit::compile_decoder_from_ir_text(
    ir_text,
    shape,
    &registry,
    /* with_passes */ false,
);
let value: T = kajit::deserialize(&decoder, input)?;
```

One-shot helper:

```rust
let value: T = kajit::deserialize_from_ir_text(
    ir_text,
    shape,
    &registry,
    false,
    input,
)?;
```

### Workflow B: CFG-MIR text -> compile -> run

```rust
let decoder = kajit::compile_decoder_from_cfg_mir_text(
    cfg_mir_text,
    /* trusted_utf8_input */ false,
);
let value: T = kajit::deserialize(&decoder, input)?;
```

One-shot helper:

```rust
let value: T = kajit::deserialize_from_cfg_mir_text(cfg_mir_text, input)?;
```

### When to choose RVSDG vs CFG-MIR text

- Choose RVSDG text when you are debugging an optimization or lowering bug and
  need the pipeline to re-run.
- Choose CFG-MIR text when you already have a post-linearization reproducer and
  want the fastest loop for regalloc/backend debugging.

If the bug only exists after a specific lowering decision, freezing the case as
CFG-MIR avoids reintroducing unrelated churn from earlier stages.

### Differential minimizer CLI

For differential minimization from a saved CFG-MIR file:

```bash
cargo run --manifest-path xtask/Cargo.toml -- \
  minimize-cfg-mir \
  path/to/failing.cfg-mir \
  8080808080
```

Accepted input formats:

```text
8080808080
0x8080808080
[0x80, 0x80, 0x80, 0x80, 0x80]
```

The command:
- parses the CFG-MIR text
- runs predicate-preserving minimization with the regalloc differential oracle
- prints a reduction summary and preserved divergence signature to `stderr`
- writes the reduced CFG-MIR program to `stdout`

Typical shell usage:

```bash
cargo run --manifest-path xtask/Cargo.toml -- \
  minimize-cfg-mir failing.cfg-mir 8080808080 \
  > minimized.cfg-mir
```

### What minimization preserves

This is not semantics-preserving optimization. It is
predicate-preserving reduction.

The minimizer is allowed to change the program's behavior as long as the chosen
oracle still says the reduced program is interesting. For the built-in CLI, the
oracle is:

- allocate CFG-MIR
- run `differential_check_cfg`
- require the same divergence signature to remain

That means the reduced program may decode a different value, execute fewer
blocks, or trap earlier, as long as it still preserves the same differential
mismatch class being investigated.

### What minimization does today

The current reducer set operates on valid CFG-MIR and keeps only accepted
reductions. It includes:

- unreachable block deletion
- unused block-parameter and edge-argument removal
- narrow trampoline collapse

This is expected to grow over time. The important property is the oracle loop,
not any one reducer.

### End-to-end reduction workflow

1. Reproduce the failing test with one fixed input.
2. Dump or print canonical CFG-MIR text.
3. Confirm the CFG-MIR text still reproduces with
   `compile_decoder_from_cfg_mir_text` or `deserialize_from_cfg_mir_text`.
4. Run `xtask minimize-cfg-mir` on that exact input.
5. Save the reduced text.
6. Re-run the differential harness, dumps, or LLDB on the reduced program.
7. Only after the reduced repro is stable, start patching codegen/regalloc.

### Common minimizer outcomes

- `seed program is not differentially interesting`: the chosen input does not
  reproduce a regalloc differential mismatch for that CFG-MIR program.
- `failed to parse CFG-MIR`: the seed text is not canonical or violates CFG-MIR
  invariants.
- Reduced output is still large: the current reducers could not delete more
  structure without breaking the predicate. Add more reducers instead of
  hand-editing blindly.

## On-demand pipeline dumps

Generated corpus tests do not enforce IR/CFG-MIR/edit snapshots by default. Dump pipeline artifacts on demand with:

- `KAJIT_DUMP_STAGES` — comma-separated stage list: `ir`, `linear`, `cfg`, `edits`, `opts`, or `all`
- `KAJIT_DUMP_FILTER` — optional comma-separated substring filters matched against `"<format>::<case>"` (e.g. `json::all_scalars`)
- `KAJIT_DUMP_DIR` — output directory (default: `target/kajit-stage-dumps`)
- `KAJIT_ASSERT_CODEGEN_SNAPSHOTS=1` — opt back into legacy snapshot assertions

`KAJIT_DUMP_DIR` accepts absolute or relative paths. Relative paths are resolved
from the test process working directory (often the crate directory), so use an
absolute path if you want dumps in a specific workspace location.

Example:

```bash
KAJIT_OPTS='+all_opts,+regalloc,-pass.theta_loop_invariant_hoist' \
KAJIT_DUMP_STAGES='ir,linear,cfg,edits' \
KAJIT_DUMP_FILTER='json::all_scalars' \
cargo nextest run -p kajit --test corpus -E 'test(=json::all_scalars)'
```

To dump RVSDG between every optimization pass:

```bash
KAJIT_DUMP_STAGES='opts' \
KAJIT_DUMP_FILTER='json::all_scalars' \
cargo nextest run -p kajit --test corpus -E 'test(=rvsdg_json::all_scalars)'
```

On Apple Silicon, run x86_64 tests via Rosetta:

```bash
KAJIT_OPTS='-regalloc' \
KAJIT_DUMP_STAGES='ir,linear,cfg,edits' \
KAJIT_DUMP_FILTER='postcard::scalar_u16_v1' \
cargo nextest run -p kajit --target x86_64-apple-darwin \
  --test corpus -E 'test(=postcard::scalar_u16_v1)'
```

## Reading dump files

Each dump file is named `<format>__<case>__<arch>__<stage>.txt`, e.g. `postcard__scalar_u64__v3__x86_64__cfg.txt`.

**`ir.txt`** — RVSDG IR after optimization passes. Nodes are `nN = Op [inputs] -> [outputs]`. Region arguments are `argN`.

**`linear.txt`** — LinearIr after linearization and register allocation lowering. Instructions are numbered; `lin=N` indices in other dumps refer to these.

**`cfg.txt`** — CFG-MIR (regalloc input). Vregs are `vN`; hardware-pinned operands appear as `vN/hwM` where `hwM` is the physical register index (arch-specific: on x86_64, hw0=rax, hw1=rcx, hw2=rdx, ...).

**`edits.txt`** — Regalloc output edits. Each line is a move inserted on a block edge: `pN -> pM` (reg-to-reg), `pN -> stackM` (spill), or `stackM -> pN` (reload). Physical register indices match `cfg.txt`. The file may also contain just a count if there are many edits.

**`opts.txt`** — RVSDG snapshots between each optimization pass, labeled by pass name.

## LLDB debugging of JIT code

### Quick start

Start LLDB for a specific test with the standalone helper:

```bash
scripts/lldb-test.sh json::bool_true_false
```

The script resolves the concrete test binary via `cargo nextest list`, then launches LLDB with:
- Sets `KAJIT_DEBUG=1` (enables DWARF `.debug_line` + `.debug_info` + `.debug_abbrev` emission)
- Enables the GDB JIT loader (`settings set plugin.jit-loader.gdb.enable on`)
- Sets a breakpoint on `__jit_debug_register_code`
- Passes test args `--exact <test_name> --nocapture`
- Leaves LLDB interactive (does not auto-run; type `run` yourself)

### How it works

When `KAJIT_DEBUG=1`:
1. The compiler generates a CFG-MIR listing file at `/tmp/kajit-debug/<type>.cfg-mir` (one line per CFG-MIR op, including `f/b/op` IDs and derived `idx`)
2. Both backends call `set_source_location()` before each instruction, recording canonical CFG-MIR op order (`OpId`-based listing lines) as DWARF line numbers
3. A minimal DWARF v4 compilation unit is built (`.debug_info` referencing `.debug_line` via `DW_AT_stmt_list`, plus `.debug_abbrev`)
4. The JIT ELF registered via the GDB JIT interface contains `.text`, `.symtab`, `.debug_line`, `.debug_abbrev`, and `.debug_info`
5. LLDB (with JIT loader enabled) parses the ELF, loads the line table, and maps code offsets to `.cfg-mir` listing lines

### LLDB commands for JIT code

When LLDB stops at `__jit_debug_register_code`, the JIT code has just been registered:

```
# Confirm JIT images are loaded — look for JIT(...) entries
(lldb) image list

# Find registered decode/encode symbols
(lldb) image lookup -rn 'kajit::decode::'

# Inspect a JIT address (verbose — shows LineEntry if DWARF is working)
(lldb) image lookup -va <address>

# Check which sections the JIT ELF contains
(lldb) image dump sections JIT(0x...)

# Dump the line table (use the .cfg-mir filename, not the full path)
(lldb) image dump line-table kajit__decode__Bools.cfg-mir

# Set a breakpoint on JIT code by regex name (-r, not -n)
(lldb) breakpoint set -r 'kajit::decode::Bools'

# Or by address (get address from image lookup -rn)
(lldb) breakpoint set -a <address>

# Once stopped in JIT code, source-level stepping uses the .cfg-mir listing
(lldb) source info
(lldb) step
```

### Key files

| File | Purpose |
|------|---------|
| `kajit/src/jit_dwarf.rs` | Builds DWARF v4 `.debug_line`, `.debug_abbrev`, `.debug_info` sections |
| `kajit/src/jit_debug.rs` | GDB JIT interface: builds in-memory ELF, registers with debugger, writes perf map |
| `kajit/src/compiler.rs` | Glue: `build_dwarf_from_source_map()` converts backend source maps to DWARF |
| `scripts/lldb-test.sh` | Standalone LLDB launcher for one exact test |
| `/tmp/kajit-debug/*.cfg-mir` | Generated listing files (one per JIT-compiled type) |
| `/tmp/perf-<pid>.map` | perf sampling map (always written, even without `KAJIT_DEBUG`) |

### Known limitations

- **Breakpoints by name** require `-r` (regex), not `-n` (exact name). LLDB's `-n` doesn't resolve JIT symbols. Use `breakpoint set -r 'kajit::decode::Bools'` or `breakpoint set -a <address>`.
- The helper script is intended for local interactive LLDB sessions; it is not a nextest run wrapper.
- The GDB JIT loader must be explicitly enabled on LLDB: `settings set plugin.jit-loader.gdb.enable on` (the wrapper script does this automatically).

## Getting disassembly

There is no `disasm` dump stage. To disassemble JIT-compiled output, use the `disasm_bytes` helper inside a backend test:

```rust
// in kajit/src/backends/x86_64/mod.rs tests
let lin = linearize(&mut func);
let deser = compiler::compile_linear_ir_decoder(&lin, false);
println!("{}", disasm_bytes(deser.code(), Some(deser.entry_offset())));
```

For corpus tests, add a temporary `println!` to the corpus test harness in `kajit/tests/corpus.rs`, wrapping the compile step and calling the same `disasm_bytes` helper (you'll need to expose it or inline the capstone call).

To check what comparison type (`jb` vs `jl`, etc.) a branch emits, look at `kajit/src/backends/x86_64/emit.rs` for the relevant IR op (e.g. `CmpLt`, `CmpLtu`) and note which conditional jump instruction it produces.
