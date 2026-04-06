# Remove cursor knowledge from backends

After cursor ops are gone from IR (006), the backends no longer need special cursor handling.

## aarch64 (`kajit/src/backends/aarch64/regalloc3_backend.rs`)

Remove:
- `sync_ctx_cursor_around_calls` flag
- `cursor_writeback_reg` field
- `cursor_reg` / `end_reg` in EmitContext
- Special register assignment for x19/x20 as cursor/end
- `PrologueConfig::load_cursor_x19_x20` and `writeback_cursor_to_ctx`
- Cursor flush before intrinsic calls and reload after
- `SaveCursor`, `SaveInputEnd`, `RestoreCursor` emission (these ops no longer exist)

Also in `kajit/src/arch/aarch64.rs`:
- Cursor-specific prologue/epilogue logic
- Comments about x19=cursor, x20=input_end register convention

## x86_64 (`kajit/src/backends/x86_64/regalloc3_backend/`)

Remove:
- `sync_ctx_cursor_around_calls` flag in `context.rs`
- `cursor_writeback_enc` field
- Cursor register handling in `inst.rs` (SaveCursor/SaveInputEnd/RestoreCursor emission)
- CTX_INPUT_PTR / CTX_INPUT_END offset references

## What remains

The backends still need:
- Standard calling convention support (pass struct pointers as arguments)
- Callee-saved register management (driven by regalloc, not by "is this a decoder")
- Intrinsic call emission (generic: marshal args into ABI registers, call, handle return)
