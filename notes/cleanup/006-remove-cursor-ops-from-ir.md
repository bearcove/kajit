# Remove cursor ops from IR

Replace the 8 cursor-specific IR ops with ordinary loads/stores/arithmetic through the borrowed cursor struct.

## Ops to remove

In `kajit-ir/src/ir.rs` (IrOp enum) and `kajit-lir/src/linearize.rs` (LinearOp enum):

| Op | Replacement |
|----|-------------|
| `ReadBytes { count }` | Load from cursor.bytes.ptr + cursor.pos, then store cursor.pos += count |
| `PeekByte` | Load from cursor.bytes.ptr + cursor.pos (no advance) |
| `AdvanceCursor { count }` | Store cursor.pos += count |
| `AdvanceCursorBy` | Store cursor.pos += dynamic_value |
| `BoundsCheck { count }` | Compare cursor.pos + count <= cursor.bytes.len, branch to error |
| `SaveCursor` | Load cursor.pos (ordinary value) |
| `SaveInputEnd` | Load cursor.bytes.len or compute cursor.bytes.ptr + cursor.bytes.len |
| `RestoreCursor` | Store cursor.pos = saved_value |

## Also remove

- `CURSOR_STATE_DOMAIN` hardcoded constant (ID 0) — make domains frontend-declared
- `cursor_advance` field in `IrOpMetadata`
- Format-specific ops that should be intrinsics: `ZigzagDecode`, `SimdStringScan`, `SimdWhitespaceSkip`

## Ripple effects

- `kajit-mir/src/cfg_mir.rs` — wraps LinearOp, inherits removal
- All RVSDG passes that pattern-match on cursor ops
- `dead_theta_ports.rs` — currently tracks cursor state threading; simplifies dramatically
- `unroll_theta.rs` — cursor-carrying loop ports change shape
- Interpreter (`kajit-mir/src/interpreter.rs`) — cursor op simulation removed
