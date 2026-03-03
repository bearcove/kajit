# KAJIT MIR MCP

This document is for humans and agents using the `kajit-mir` MCP server as the
first interface to RA-MIR interpreter debugging.

## Why this exists

`kajit-mir-mcp` exposes a deterministic, reversible debugger session over MCP.
It is meant to be the default interface for:

- stepping RA-MIR programs forward and backward
- inspecting interpreter state (vregs, cursor, output bytes)
- reproducing and triaging decode failures with exact offsets/error codes
- creating repeatable debugging transcripts an agent can share

## Install and register

From repo root:

```bash
cargo run --manifest-path xtask/Cargo.toml -- install
```

Register in Codex:

```bash
codex mcp add kajit-mir -- /Users/amos/.cargo/bin/kajit-mir-mcp
```

Register in Claude Code:

```bash
claude mcp add --transport stdio kajit-mir -- /Users/amos/.cargo/bin/kajit-mir-mcp
```

Then restart the client.

## Session model

Each MCP session is in-memory and process-local.

1. `session_new` parses RA-MIR text and creates one debugger session.
2. Step or run with `session_step`, `session_back`, `session_run_until`.
3. Inspect with `session_state`, `session_inspect_vreg`, `session_inspect_output`.
4. Always `session_close` when done.

## Tool surface

- `session_new`
  - args: `ra_mir_text` (required), `input_hex` (optional)
- `session_close`
  - args: `session_id`
- `session_step`
  - args: `session_id`, `count` (optional, default `1`)
- `session_back`
  - args: `session_id`, `count` (optional, default `1`)
- `session_run_until`
  - args: `session_id`, exactly one of:
    - `block_id`
    - `trap = true`
    - `until_return = true`
  - optional: `max_steps` (default `10000`)
- `session_state`
  - args: `session_id`
- `session_inspect_vreg`
  - args: `session_id`, `vreg`
- `session_inspect_output`
  - args: `session_id`, `start` (optional), `len` (optional)

## Typical agent workflow

1. Create session with target RA-MIR text + repro input bytes.
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

## Notes and gotchas

- Use `until_return`, not `return`, in `session_run_until`.
- `input_hex` accepts compact hex and forgiving forms like `[0x81, 0x01]`.
- `session_state` includes full `output_hex`; prefer `session_inspect_output` for
  large buffers.
- `run_until` is bounded by `max_steps` to avoid runaway sessions.
- State is deterministic for the same RA-MIR + input + step sequence.

## Suggested next layering

- Keep this MCP as the stable debugger API.
- Build higher-level traces/minimization as separate tooling on top of these
  deterministic primitives.
