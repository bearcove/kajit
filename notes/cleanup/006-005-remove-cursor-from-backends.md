# Remove cursor knowledge from backends

After cursor ops are gone from LIR/CFG-MIR, backends no longer need special cursor handling.

## aarch64 (kajit/src/backends/aarch64/regalloc3_backend.rs)

Remove:
- `cursor_writeback_reg` field (line 81)
- `sync_ctx_cursor_around_calls` flag (line 71)
- `cursor_reg` / `end_reg` in EmitContext (arch/aarch64.rs lines 24, 27)
- x19/x20 reserved as cursor/end registers
- `PrologueConfig::load_cursor_x19_x20`, `writeback_cursor_to_ctx`
- Cursor flush before intrinsic calls, reload after (lines 1291-1297, 1383-1393)
- SaveCursor/SaveInputEnd/RestoreCursor emission (lines 594-664)
- Fused address offset tracking for RestoreCursor (lines 1822-1927)

## x86_64 (kajit/src/backends/x86_64/regalloc3_backend/)

Remove:
- `sync_ctx_cursor_around_calls` in context.rs (line 44)
- `cursor_writeback_enc` (line 46)
- r12/r13 reserved as cursor/end registers
- SaveCursor/SaveInputEnd/RestoreCursor emission (inst.rs lines 82-172)
- Cursor sync in calls.rs (lines 27-140)
- Fused cursor offsets in fusion.rs (lines 118-180)

## What remains

Backends still need:
- Generic calling convention (pass struct pointers as args)
- Callee-saved register management (driven by regalloc, not decoder knowledge)
- Intrinsic call emission (generic: marshal args, call, handle return)

## Depends on

006-004 (cursor ops gone from CFG-MIR, so backends never see them)
