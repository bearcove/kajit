# AGENTS.md

## Development/debugging guide
- For debugger workflows (LLDB/GDB), JIT symbolization, and step-by-step JIT debugging, see `/Users/amos/bearcove/kajit/DEVELOP.md`.

## Compiler architecture principles
- Expand backend capabilities to support required IR semantics; do not work around backend limitations in lowering or intrinsic selection.
- Prefer explicit backend/ABI support (for example, dedicated pure-call ABI support) over adapters that hide contract mismatches.
- If you catch yourself thinking "workaround", stop and implement the proper fix in backend/compiler/runtime instead.

## Test policy
- Always run tests after making code changes.
- Use `cargo nextest run` for Rust tests.
- Run at least the tests directly covering the changed code before reporting completion.
- If a user asks for broader validation, run the full requested test set.
- Report the exact test command(s) and pass/fail result in your handoff.
