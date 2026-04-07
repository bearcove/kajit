# Make ctx_ptr an explicit vreg

Currently `CallIntrinsic` implicitly passes `ctx_enc` as first argument to intrinsics. ctx_enc is a pinned register.

## What changes

- `CallIntrinsic` gets ctx as an explicit operand (first in args list or dedicated field)
- Backend: reads ctx from vreg operand, not `self.ctx_enc`
- Backend: delete `ctx_enc` from EmitContext
- Frontend (hir_to_ir): ctx_ptr becomes a data_arg on root lambda

## Depends on

011-2 (output_ptr as vreg).

## Enables

011-4 (explicit error check).
