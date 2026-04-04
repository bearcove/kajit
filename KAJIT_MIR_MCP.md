# KAJIT MCP

This document is for humans and agents using the `kajit` MCP server as the
first interface to Kajit debugging.

## Why this exists

`kajit mcp` exposes:

- deterministic, reversible CFG-MIR interpreter sessions
- persistent LLDB-backed lockstep sessions for compiled decoders

It is meant to be the default interface for:

- stepping RA-MIR programs forward and backward
- inspecting interpreter state (vregs, cursor, output bytes)
- launching compiled standalone harnesses under LLDB
- stepping lockstep sessions without rerunning a one-shot batch command
- reproducing and triaging decode failures with exact offsets/error codes
- creating repeatable debugging transcripts an agent can share

## Install and register

From repo root:

```bash
cargo run --manifest-path xtask/Cargo.toml -- install
```

Register in Codex:

```bash
codex mcp add kajit -- /Users/amos/.cargo/bin/kajit mcp
```

Register in Claude Code:

```bash
claude mcp add --transport stdio kajit -- /Users/amos/.cargo/bin/kajit mcp
```

Then restart the client.

## Session model

Each MCP session is in-memory and process-local.

Interpreter sessions:
1. `session_new` parses CFG-MIR text from a file and creates one debugger session.
2. Step or run with `session_step`, `session_back`, `session_run_until`.
3. Inspect with `session_state`, `session_inspect_vreg`, `session_inspect_output`.
4. Always `session_close` when done.

Lockstep sessions:
1. `debug_session_new` compiles a decoder, generates a standalone harness, and launches LLDB.
2. Step with `debug_session_step`.
3. Inspect with `debug_session_state`, `debug_session_disassemble`, `debug_session_lldb`.
4. Always `debug_session_close` when done.

## Tool surface

- `session_new`
  - args: `cfg_mir_path` (required), `input_hex` (optional)
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
- `debug_session_new`
  - args: `format`, `ty`, `input_hex`
- `debug_session_close`
  - args: `session_id`
- `debug_session_step`
  - args: `session_id`, `count` (optional, default `1`)
- `debug_session_state`
  - args: `session_id`
- `debug_session_disassemble`
  - args: `session_id`, `context` (optional)
- `debug_session_registers`
  - args: `session_id`, `names` (optional comma-separated register names)
- `debug_session_memory`
  - args: `session_id`, `address`, `len` (optional)
- `debug_session_backtrace`
  - args: `session_id`
- `debug_session_source_info`
  - args: `session_id`
- `debug_session_lldb`
  - args: `session_id`, `command`

## Typical agent workflow

1. Create an interpreter session with target CFG-MIR text + repro input bytes, or a lockstep session with `format`/`type`/input bytes.
2. Capture baseline `session_state`.
3. Use `session_step` for local reasoning around suspicious ops.
4. Use `session_back` to re-check branches without re-creating session.
5. Use `session_run_until` to jump to trap/return/block checkpoints.
6. Use `session_inspect_output` for small deterministic slices.
7. For compiled/JIT debugging, use `debug_session_step` for lockstep stepping and `debug_session_registers` / `debug_session_memory` / `debug_session_backtrace` / `debug_session_source_info` for structured machine inspection.
8. Use `debug_session_lldb` only as an escape hatch when Kajit MCP does not yet expose a structured primitive you need.
9. Close session and report:
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
- `debug_session_*` requires Kajit to be built with the `lldb` feature and a working LLDB runtime.

## Suggested next layering

- Keep `kajit mcp` as the stable debugger API.
- Build richer provenance, multi-layer source views, and minimization workflows on top of these session primitives.
