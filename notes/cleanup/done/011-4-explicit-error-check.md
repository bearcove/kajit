# Make error check after intrinsic calls explicit

Currently the backend hardcodes: load `[ctx + CTX_ERROR_CODE]`, test, branch to error_exit after every `CallIntrinsic`.

## What changes

- Frontend generates explicit ops after each CallIntrinsic: LoadFromAddr(ctx + CTX_ERROR_CODE, W4), CmpNe(error_code, 0), ErrorExit
- Backend `emit_call_intrinsic` stops emitting the error check
- Error check becomes visible in IR/CFG-MIR, not hidden in backend

## Depends on

011-3 (ctx_ptr as vreg — needed to compute error field address).

## Enables

011-5 (kill is_scalar).
