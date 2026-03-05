# Pipeline debugging

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

## CFG-MIR text format

Canonical CFG-MIR now supports a round-trippable text format.

- Print canonical text from `kajit_mir::cfg_mir::Program`:
  - `let text = format!("{cfg_program}");`
- Parse it back:
  - `let cfg_program = kajit_mir_text::parse_cfg_mir(&text)?;`

The parser is strict and validates function/block/edge/inst/term IDs and CFG invariants.

## On-demand pipeline dumps

Generated corpus tests do not enforce IR/CFG-MIR/edit snapshots by default. Dump pipeline artifacts on demand with:

- `KAJIT_DUMP_STAGES` — comma-separated stage list: `ir`, `linear`, `cfg`, `edits`, `opts`, or `all`
- `KAJIT_DUMP_FILTER` — optional comma-separated substring filters matched against `"<format>::<case>"` (e.g. `json::all_scalars`)
- `KAJIT_DUMP_DIR` — output directory (default: `target/kajit-stage-dumps`)
- `KAJIT_ASSERT_CODEGEN_SNAPSHOTS=1` — opt back into legacy snapshot assertions

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
1. The compiler generates a CFG-MIR listing file at `/tmp/kajit-debug/<type>.cfg-mir` (one line per CFG-MIR instruction)
2. Both backends call `set_source_location()` before each instruction, recording the linear op index as the DWARF line number
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
