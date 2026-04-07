# 011-3: Explicit error check after intrinsic calls

## Goal

The backend emits no implicit loads from the context struct. Error checking after
`CallIntrinsic` is explicit in the IR/CFG-MIR, visible to optimizations and the interpreter.

## What changes

### HIR→IR lowering (`kajit/src/compiler/hir_to_ir.rs`)
- After each `call_intrinsic()`, emit:
  1. `LoadFromAddr(ctx_ptr + CTX_ERROR_CODE, W4)` → error_code vreg
  2. `CmpNe(error_code, 0)` → has_error vreg
  3. Conditional branch / ErrorExit on has_error

### Backends (`kajit/src/backends/{aarch64,x86_64}/regalloc3_backend/calls.rs`)
- `emit_call_intrinsic` stops emitting the post-call error check
  (delete the `load [ctx + ERROR_CODE]; test; jnz error_exit` block)
- The error check is now just regular instructions emitted by the frontend

### Interpreter (`kajit-mir/src/interpreter.rs`)
- `CallIntrinsic` execution no longer implicitly checks `ctx.error_code`
- The explicit LoadFromAddr + CmpNe + ErrorExit ops handle it

### Context (`kajit/src/context.rs`)
- `CTX_ERROR_CODE` constant stays (used by the frontend to generate the right offset)
- But the backend no longer imports or uses it directly

## Depends on

011-2 (unified ABI — ctx_ptr is a vreg, so we can compute ctx_ptr + offset in IR).

## Enables

011-4 (naming cleanup).
