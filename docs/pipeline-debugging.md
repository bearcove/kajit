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

On Apple Silicon, combine with x86_64 Rosetta validation:

```bash
KAJIT_OPTS='-regalloc' cargo xtask test-x86_64 --full -- \
  --test corpus -E 'test(=postcard::scalar_u16_v1)'
```
