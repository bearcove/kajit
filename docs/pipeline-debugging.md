# Pipeline debugging

## Bisecting with `KAJIT_OPTS`

Selectively disable parts of the compile pipeline at runtime without rebuilding.

Syntax: comma-separated tokens. `+name` enables, `-name` disables, bare `name` is treated as `+name`. Unknown names fail fast.

Options:
- `all_opts` — default RVSDG optimization passes before linearization
- `regalloc` — regalloc edit application during backend emission

Per-pass options:
- `pass.bounds_check_coalescing`
- `pass.theta_loop_invariant_hoist`
- `pass.inline_apply`
- `pass.dead_code_elimination`

Examples:

```bash
# disable all optimization passes
KAJIT_OPTS='-all_opts' cargo nextest run -p kajit <test_filter>

# disable regalloc edit application
KAJIT_OPTS='-regalloc' cargo nextest run -p kajit <test_filter>

# disable a single pass
KAJIT_OPTS='-pass.inline_apply' cargo nextest run -p kajit <test_filter>
```

Show built-in help:

```bash
KAJIT_OPTS=help cargo nextest run -p kajit <test_filter>
```

## On-demand pipeline dumps

Generated corpus tests do not enforce IR/RA-MIR/edit snapshots by default. Dump pipeline artifacts on demand with:

- `KAJIT_DUMP_STAGES` — comma-separated stage list: `ir`, `linear`, `ra`, `edits`, `opts`, or `all`
- `KAJIT_DUMP_FILTER` — optional comma-separated substring filters matched against `"<format>::<case>"` (e.g. `json::all_scalars`)
- `KAJIT_DUMP_DIR` — output directory (default: `target/kajit-stage-dumps`)
- `KAJIT_ASSERT_CODEGEN_SNAPSHOTS=1` — opt back into legacy snapshot assertions

Example:

```bash
KAJIT_OPTS='+all_opts,+regalloc,-pass.theta_loop_invariant_hoist' \
KAJIT_DUMP_STAGES='ir,linear,ra,edits' \
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
KAJIT_DUMP_STAGES='ir,linear,ra,edits' \
KAJIT_DUMP_FILTER='postcard::scalar_u16_v1' \
cargo nextest run -p kajit --target x86_64-apple-darwin \
  --test corpus -E 'test(=postcard::scalar_u16_v1)'
```

## Reading dump files

Each dump file is named `<format>__<case>__<arch>__<stage>.txt`, e.g. `postcard__scalar_u64__v3__x86_64__ra.txt`.

**`ir.txt`** — RVSDG IR after optimization passes. Nodes are `nN = Op [inputs] -> [outputs]`. Region arguments are `argN`.

**`linear.txt`** — LinearIr after linearization and register allocation lowering. Instructions are numbered; `lin=N` indices in other dumps refer to these.

**`ra.txt`** — RA-MIR (regalloc input). Vregs are `vN`; hardware-pinned operands appear as `vN/hwM` where `hwM` is the physical register index (arch-specific: on x86_64, hw0=rax, hw1=rcx, hw2=rdx, ...).

**`edits.txt`** — Regalloc output edits. Each line is a move inserted on a block edge: `pN -> pM` (reg-to-reg), `pN -> stackM` (spill), or `stackM -> pN` (reload). Physical register indices match `ra.txt`. The file may also contain just a count if there are many edits.

**`opts.txt`** — RVSDG snapshots between each optimization pass, labeled by pass name.

## LLDB debugging of JIT code

Run any test under LLDB with the `lldb` nextest profile. This enables the GDB JIT loader, sets `KAJIT_DEBUG=1` (so DWARF `.debug_line` is emitted), and breaks on JIT code registration:

```bash
cargo nextest run --profile lldb -E 'test(=json::bool_true_false)'
```

When LLDB stops at `__jit_debug_register_code`, the JIT code has been registered. You can then:

```
# Confirm JIT images are loaded
(lldb) image list
# Look for JIT(...) entries

# Look up a JIT address
(lldb) image lookup -a <pc>

# Find registered decode/encode symbols
(lldb) image lookup -rn 'fad::decode::'
(lldb) image lookup -rn 'fad::encode::'

# Set a breakpoint on a JIT function and continue
(lldb) breakpoint set -rn 'fad::decode::.*'
(lldb) continue

# Once stopped in JIT code, source-level stepping uses the .ra-mir listing
(lldb) source info
(lldb) step
```

The `.ra-mir` listing files are written to `/tmp/kajit-debug/`. Each listing line corresponds to an RA-MIR instruction — DWARF `.debug_line` maps native code offsets to these lines, so `source info` and `step` navigate the RA-MIR.

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
