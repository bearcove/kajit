# Update linearization and interpreter

After cursor IR ops are replaced (006-002), the LIR and interpreter layers need matching updates.

## Linearization (kajit-lir/src/linearize.rs)

- Remove cursor-specific LinearOp variants (ReadBytes, PeekByte, AdvanceCursor, etc.)
- IR cursor memory ops lower to generic load/store/add LinearOps
- VReg dependency tracking updates automatically (cursor pointer is just a VReg)

## Interpreter (kajit-mir/src/interpreter.rs)

- Remove cursor state machine simulation (lines 752-811 fast path, lines 1230-1325 trace path)
- Cursor ops become ordinary load/store/add — the interpreter already handles those
- Remove `input_ptr`/`input_end` from interpreter state — cursor is just a pointer value

## CFG-MIR (kajit-mir/src/cfg_mir.rs)

- Remove cursor op Display formatting (lines 905-912)
- Remove cursor op operand lowering (lines 1300-1411 subset)
- Remove SSA rewriting for cursor-specific ops

## Depends on

006-002 (cursor ops are replaced at IR level)
