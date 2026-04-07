# Make output_ptr an explicit vreg

Currently `WriteToField { src, offset, width }` stores to `[output_enc + offset]`. The output pointer is a pinned register, not a vreg.

## What changes

- `IrOp::WriteToField { offset, width }` takes `[out_ptr, value]` as inputs (2 data inputs instead of 1)
- `IrOp::SetOutPtr` takes `[new_ptr]`, returns nothing — out_ptr is just a vreg, so SetOutPtr becomes Add(out_ptr, offset) or similar
- `IrOp::GetOutPtr` — unnecessary if out_ptr is a vreg, delete it
- Backend: `WriteToField` reads out_ptr from vreg operand, not `self.output_enc`
- Backend: delete `output_enc` from EmitContext
- Frontend (hir_to_ir): output_ptr becomes first data_arg on root lambda
- `output_size` on Lambda stays (metadata)

## Depends on

011-1 (cursor sync removed — fewer complications in backend).

## Enables

011-3 (ctx_ptr as vreg).
