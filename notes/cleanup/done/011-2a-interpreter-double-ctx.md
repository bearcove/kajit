# 011-2a: Interpreter double-ctx bug

## Problem

`run_call_intrinsic` in `kajit-mir/src/interpreter.rs` prepends its own
`ctx_ptr` before calling the intrinsic. But now that `ctx_ptr` is an explicit
arg in the IR, `args[0]` already IS `ctx_ptr`. So the intrinsic receives
`(interpreter_ctx, ir_ctx, arg1, arg2, ...)` — double ctx.

## Fix

`run_call_intrinsic` should stop prepending ctx. The args already contain
ctx_ptr as the first element. Just pass args through to the C function directly,
like `run_call_pure` does.

This also means removing the cursor sync (`state.ctx.input_ptr = ...`) and
error check from `run_call_intrinsic` — those will become explicit IR ops
(011-3 for error check, cursor sync is already dead).

## Affects

- `kajit-mir/src/interpreter.rs` — `run_call_intrinsic` and both call sites
- Option/vec test timeouts are likely caused by this bug
