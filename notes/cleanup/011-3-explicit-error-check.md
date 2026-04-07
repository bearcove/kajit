# 011-3: Explicit error check after intrinsic calls

## Goal

The backend emits no implicit loads from the context struct. Error checking after
calls is explicit in the IR/CFG-MIR, visible to optimizations and the interpreter.

## What changes

This is mostly a consequence of 011-2 (general-purpose IR). Once ctx_ptr is a
real vreg and CallIntrinsic is replaced by a unified Call op, the backend no
longer has any special-case code for error checking.

### Frontend (`hir_to_ir.rs`)
- After each call that can fail, emit:
  1. `Load(Add(ctx_ptr, Const(CTX_ERROR_CODE)), W4)` → error_code vreg
  2. `CmpNe(error_code, Const(0))` → has_error vreg
  3. Gamma branch: if has_error → return (the error path)

### Backends
- Nothing to do — these are regular Load/CmpNe/Branch instructions
- Delete the post-call error check code from `emit_call_intrinsic` (if not already gone from 011-2)

### Interpreter
- Nothing to do — Load/CmpNe/Branch already work

### ErrorExit op
- Delete entirely. Frontend emits `Store(ctx + ERROR_OFFSET, code)` + return.
- The backend's `emit_error_trampolines` / `emit_error_with_ctx` go away.

### DWARF note

Today `deser_dwarf_variables` exposes `error_code` / `error_offset` using `ctx` as a fixed
register base. Once 011-2 makes `ctx_ptr` a normal vreg, that expression will be wrong unless
we also teach DWARF emission how to express `*(ctx + k)` based on vreg allocation, or we drop
those variables during cleanup.

## Depends on

011-2 step 2 (ctx_ptr as real vreg, unified Call op).

## Enables

011-4 (naming cleanup).
