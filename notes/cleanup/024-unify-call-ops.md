# 015: Unify three Call ops into one

## Goal

Replace `CallIntrinsic`, `CallPure`, `CallEffect` with a single `Call` op.

## What changes

Now that `CallIntrinsic` no longer has implicit ctx/field_offset, the only
differences between the three are:

- `CallIntrinsic`: has error check after call (will move to explicit IR in 016)
- `CallPure`: no side effects, eligible for CSE/DCE
- `CallEffect`: has side effects, threaded on memory state

A single `Call { func, pure: bool }` can replace all three:
- `pure: true` = old CallPure (no state threading, CSE-able)
- `pure: false` = old CallEffect/CallIntrinsic (state-threaded)

The error check is a separate concern (016).

## Affects

- `kajit-ir/src/ir.rs` — op definition
- `kajit-lir/src/linearize.rs` — linearization
- `kajit-mir/src/cfg_mir.rs` — operand handling
- Both backends — call emission
- Interpreter — call execution
- Text parsers
