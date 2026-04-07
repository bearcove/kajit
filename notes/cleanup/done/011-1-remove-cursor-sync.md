# Remove cursor sync from backends

`sync_ctx_cursor_around_calls` writes/reads `[ctx + CTX_INPUT_PTR]` around intrinsic calls. Already dead: `uses_cursor_ops = false`.

## What to delete

- `sync_ctx_cursor_around_calls` field from EmitContext (x86_64 + aarch64)
- `cursor_writeback_enc` field from EmitContext
- Cursor sync code in `emit_call_intrinsic` (both backends)
- `uses_cursor_ops` variable
- `ctx_cursor_abi` variable
- Cursor load in decoder prologue (already gated by `uses_cursor_ops`)
- Cursor writeback in decoder epilogue

## Depends on

010 (state domain concept removed).

## Enables

011-2 (output_ptr as vreg — fewer moving parts in the backend).
