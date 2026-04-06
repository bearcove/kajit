# Replace cursor IR ops with ordinary memory operations — PARTIAL

7 test-only cursor ops removed (BoundsCheck, ReadBytes, PeekByte, AdvanceCursor, AdvanceCursorBy, SimdStringScan, SimdWhitespaceSkip). Committed.

## Remaining: 3 production ops

| Cursor Op | Replacement |
|-----------|-------------|
| `SaveCursor` | `LoadFromAddr(ctx_ptr + CTX_INPUT_PTR, W8)` |
| `SaveInputEnd` | `LoadFromAddr(ctx_ptr + CTX_INPUT_END, W8)` |
| `RestoreCursor` | `StoreToAddr(ctx_ptr + CTX_INPUT_PTR, saved_value, W8)` |

## What needs to happen

1. Make the context pointer an explicit data argument in the IR (it's currently implicit in the cursor state domain). The HIR already has it as a parameter — the lowerer just needs to thread it through as a PortSource.

2. In `hir_to_ir.rs`, emit `Add(ctx_ptr, offset) + LoadFromAddr/StoreToAddr` instead of `save_cursor()` / `save_input_end()` / `restore_cursor()`.

3. These ops use MEMORY_STATE_DOMAIN (like all other LoadFromAddr/StoreToAddr), not CURSOR_STATE_DOMAIN.

4. Delete the 3 ops + builders from `kajit-ir/src/ir.rs`, and remove from the entire pipeline: linearization, CFG-MIR, interpreter, backends (both aarch64 and x86_64), parsers, analysis passes.

5. Delete backend cursor-caching logic: x19/x20 reservation (aarch64), r14/r15 reservation (x86_64), cursor writeback in epilogue, cursor flush/reload around calls.

## Key files

- `kajit/src/compiler/hir_to_ir.rs` — change lowering from cursor ops to memory ops
- `kajit-ir/src/ir.rs` — delete SaveCursor/SaveInputEnd/RestoreCursor + builders
- `kajit-lir/src/linearize.rs` — delete LinearOp cursor variants
- `kajit-mir/src/cfg_mir.rs`, `interpreter.rs`, `debugger.rs` — delete cursor op handling
- `kajit/src/backends/aarch64/regalloc3_backend.rs` — delete cursor emission + caching
- `kajit/src/backends/x86_64/regalloc3_backend/{inst,mod,fusion,context}.rs` — same
- `kajit/src/arch/aarch64.rs` — delete cursor writeback in prologue/epilogue

## Depends on

Nothing (007 test-only removal is done).

## Enables

008 (delete CURSOR_STATE_DOMAIN — no ops use it after this).
