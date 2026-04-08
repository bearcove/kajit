# 016: Explicit error check after intrinsic calls

## Goal

The backend emits no implicit loads from the context struct. Error checking after
calls is explicit in the IR/CFG-MIR, visible to optimizations and the interpreter.

## What changes

### Frontend (`hir_to_ir.rs`)
- After each call that can fail, emit:
  1. `Load(Add(ctx_ptr, Const(CTX_ERROR_CODE)), W4)` → error_code vreg
  2. `CmpNe(error_code, Const(0))` → has_error vreg
  3. Gamma branch: if has_error → return (the error path)

### Backends
- Delete the post-call error check from `emit_call_intrinsic`
- Delete `ctx_enc` from EmitContext (both arches)
- Delete `ErrorExit` op and `emit_error_trampolines` / `emit_error_with_ctx`

### Interpreter
- Nothing to do — Load/CmpNe/Branch already work

## Depends on

015 (unified Call op — so error check isn't tied to a specific call variant).

## Enables

017+ (further simplification).
