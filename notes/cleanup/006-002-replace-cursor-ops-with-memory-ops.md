# Replace cursor IR ops with ordinary memory operations

The 8 cursor-specific IrOp variants become ordinary loads, stores, and arithmetic on a `&mut Cursor` pointer.

## Ops to replace

| Cursor Op | Replacement |
|-----------|-------------|
| `ReadBytes { count }` | Load from cursor.pos ptr, then cursor.pos += count |
| `PeekByte` | Load from cursor.pos ptr (no advance) |
| `AdvanceCursor { count }` | cursor.pos += count |
| `AdvanceCursorBy` | cursor.pos += dynamic_value |
| `BoundsCheck { count }` | Compare cursor.pos + count <= cursor.len, branch to error |
| `SaveCursor` | Load cursor.pos (just a value read) |
| `SaveInputEnd` | Load cursor.len or compute end pointer |
| `RestoreCursor` | Store cursor.pos = saved_value |

## Also replace in LinearOp

The same 8 variants exist in `kajit-lir/src/linearize.rs` LinearOp enum. They become load/store/add LinearOps.

## Key decision

The cursor is passed as a data argument (a pointer) to the function, not threaded as state. Cursor state domain disappears — ordering is enforced by data dependencies on the cursor pointer, or by a generic "memory" effect.

## Depends on

006-001 (state domains are frontend-declared, so removing cursor domain doesn't break the API)
