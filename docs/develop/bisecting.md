# Bisecting pipeline failures

kajit compiles through several stages: RVSDG construction → optimization passes
→ linearization → register allocation → native codegen. When a test produces
wrong output, you want to find the *first* stage that breaks things. The tools
below let you toggle individual passes and dump intermediate representations at
each stage — without rebuilding.

## `KAJIT_OPTS` — runtime pipeline toggles

`KAJIT_OPTS` is an environment variable that selectively enables or disables
parts of the compile pipeline.

### Syntax

- Comma-separated tokens
- `+name` enables an option
- `-name` disables an option
- Bare `name` is treated like `+name`
- Unknown names fail fast

### Supported options

| Option | What it controls |
|--------|-----------------|
| `all_opts` | Default RVSDG optimization passes before linearization |
| `regalloc` | Register allocation edit application during backend emission |

### Per-pass options

| Option | Pass |
|--------|------|
| `pass.bounds_check_coalescing` | Bounds check coalescing |
| `pass.theta_loop_invariant_hoist` | Theta loop invariant hoisting |
| `pass.inline_apply` | Apply node inlining |
| `pass.dead_code_elimination` | Dead code elimination |

### Examples

```bash
# disable optimization passes
KAJIT_OPTS='-all_opts' cargo nextest run -p kajit <test_filter>

# disable regalloc edit application
KAJIT_OPTS='-regalloc' cargo nextest run -p kajit <test_filter>

# disable both
KAJIT_OPTS='-all_opts,-regalloc' cargo nextest run -p kajit <test_filter>

# disable one specific default pass
KAJIT_OPTS='-pass.inline_apply' cargo nextest run -p kajit <test_filter>
```

Show built-in help:
```bash
KAJIT_OPTS=help cargo nextest run -p kajit <test_filter>
```

### Interpretation notes

- Disabling `all_opts` can change IR/RA-MIR snapshots while preserving runtime
  behavior — the unoptimized code is correct, just bigger.
- Disabling `regalloc` can intentionally surface semantic regressions and is
  useful for isolating bugs in the regalloc/edit-application path.

## `cargo xtask opts-matrix` — automated pass-combination sweep

When you suspect a specific pass (or combination of passes) is at fault, the
opts-matrix command runs every combination for you:

```bash
# default target: json::all_scalars in kajit/tests/corpus.rs
cargo xtask opts-matrix

# focused subset (example)
cargo xtask opts-matrix --pass theta_loop_invariant_hoist --pass inline_apply

# pass extra nextest args through to each run
cargo xtask opts-matrix -- --no-fail-fast
```

### Behavior

- Runs `cargo nextest run -p kajit --test corpus -E '<expr>'` for every pass
  bitmask.
- Uses `KAJIT_OPTS='+all_opts,<regalloc>,...explicit +/-pass toggles...'`.
- When you provide a `--pass` subset, non-selected default passes are forced
  off.
- Writes failing-run logs to `target/opts-matrix-logs/<timestamp>/`.

## On-demand pipeline dumps

Generated corpus tests don't enforce large IR/RA-MIR/edit snapshots by default.
Instead, dump pipeline artifacts only when you need them:

| Variable | What it does |
|----------|-------------|
| `KAJIT_DUMP_STAGES` | Comma-separated stage list: `ir`, `linear`, `ra`, `postreg`, `edits`, `opts`, or `all` |
| `KAJIT_DUMP_FILTER` | Optional comma-separated substring filters matched against `"<format>::<case>"` (e.g. `json::all_scalars`) |
| `KAJIT_DUMP_DIR` | Output directory (default: workspace `target/kajit-stage-dumps`) |
| `KAJIT_ASSERT_CODEGEN_SNAPSHOTS` | Set to `1` to opt back into legacy snapshot assertions |

### Examples

Focus on a specific test with optimization toggles:
```bash
KAJIT_OPTS='+all_opts,+regalloc,-pass.theta_loop_invariant_hoist' \
KAJIT_DUMP_STAGES='ir,linear,ra,postreg,edits' \
KAJIT_DUMP_FILTER='json::all_scalars' \
cargo nextest run -p kajit --test corpus -E 'test(=json::all_scalars)'
```

Dump RVSDG between every optimization pass (plus `initial`):
```bash
KAJIT_DUMP_STAGES='opts' \
KAJIT_DUMP_FILTER='json::all_scalars' \
cargo nextest run -p kajit --test corpus -E 'test(=rvsdg_json::all_scalars)'
```

Combine with x86_64 Rosetta validation:
```bash
KAJIT_OPTS='-regalloc' cargo xtask test-x86_64 --full -- \
  --test corpus -E 'test(=postcard::scalar_u16_v1)'
```
