# 014: Delete ZigzagDecode op

## Goal

Remove `ZigzagDecode { wide: bool }` from the IR. The frontend should emit
`Shr` + `Xor` (+ `SignExtend` if wide) instead.

## What changes

### Frontend (`hir_to_ir.rs`)
- Find where ZigzagDecode is emitted, replace with shift+xor sequence

### IR/LIR/CFG-MIR
- Delete `ZigzagDecode` variant from `IrOp`, `LinearOp`

### Backends
- Delete `emit_zigzag_decode` / ZigzagDecode match arms

### Interpreter
- Delete ZigzagDecode match arm

Small, self-contained change.
