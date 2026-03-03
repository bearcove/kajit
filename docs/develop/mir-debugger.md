# MIR interpreter debugger

The MIR interpreter should be your **first** debugging tool. It executes RA-MIR
programs deterministically and supports stepping both forward and backward — so
you can trace exactly what happens at each operation, find where output diverges,
and step back to re-examine decisions. No native code is involved, which means
no platform-specific tooling, no Rosetta, no debugger setup.

Reach for [LLDB/GDB](native-debugging.md) only when a bug doesn't reproduce in
the interpreter — that tells you it's a native codegen or ABI issue rather than
a logic error in the compiled program.

The debugger is exposed as an MCP server (`kajit-mir-mcp`), so it integrates
directly into Claude Code and other MCP-aware editors.

## Install and register

From repo root:

```bash
cargo run --manifest-path xtask/Cargo.toml -- install
```

Register in Claude Code:
```bash
claude mcp add --transport stdio kajit-mir -- /Users/amos/.cargo/bin/kajit-mir-mcp
```

Then restart the client.

## Session model

Each MCP session is in-memory and process-local.

1. `session_new` parses text and creates one debugger session:
   - `input_kind = "ra_mir"`: parse RA-MIR directly
   - `input_kind = "ir"`: parse IR, linearize, lower to RA-MIR automatically
2. Step or run with `session_step`, `session_back`, `session_run_until`.
3. Inspect with `session_state`, `session_inspect_vreg`, `session_inspect_output`.
4. Always `session_close` when done.

## Tool surface

| Tool | Arguments | Notes |
|------|-----------|-------|
| `session_new` | `input_kind` (required: `ir` or `ra_mir`), `program_text` (required), `input_hex` (optional) | |
| `session_close` | `session_id` | |
| `session_step` | `session_id`, `count` (optional, default `1`) | |
| `session_back` | `session_id`, `count` (optional, default `1`) | |
| `session_run_until` | `session_id`, exactly one of: `block_id`, `trap = true`, `until_return = true`; optional `max_steps` (default `10000`) | |
| `session_state` | `session_id` | Full state snapshot |
| `session_inspect_vreg` | `session_id`, `vreg` | Read one virtual register |
| `session_inspect_output` | `session_id`, `start` (optional), `len` (optional) | Prefer over `session_state` for large buffers |

## Typical workflow

1. Create session with target IR/RA-MIR text + repro input bytes.
2. Capture baseline `session_state`.
3. Use `session_step` for local reasoning around suspicious ops.
4. Use `session_back` to re-check branches without re-creating session.
5. Use `session_run_until` to jump to trap/return/block checkpoints.
6. Use `session_inspect_output` for small deterministic slices.
7. Close session and report:
   - input bytes
   - step index/location
   - trap code + offset (if any)
   - relevant vreg/output values

## Gotchas

- Use `until_return`, not `return`, in `session_run_until`.
- `session_new` requires explicit `input_kind` so callers cannot accidentally
  treat IR as RA-MIR or vice versa.
- `input_hex` accepts compact hex and forgiving forms like `[0x81, 0x01]`.
- For IR input from debug dumps, intrinsic refs like `0x<ptr>` are normalized
  to `0x0` for parser compatibility.
- `session_state` includes full `output_hex`; prefer `session_inspect_output` for
  large buffers.
- `run_until` is bounded by `max_steps` to avoid runaway sessions.
- State is deterministic for the same RA-MIR + input + step sequence.
