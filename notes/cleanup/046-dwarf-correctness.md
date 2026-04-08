# 019: Keep DWARF correct

## Goal

DWARF debug info doesn't emit wrong information after all the cleanup.

## What changed

1. Field-write detection used to key off `WriteToField` / `CallIntrinsic(field_offset)`.
   Now keys off `StoreToAddr` with `DebugValueKind::Field` — already done by
   another agent (semantic field DWARF derives out_ptr from data_args[0]).

2. `out_ptr` / `ctx_ptr` used to be pinned physical registers. Now they're
   normal vregs. DWARF field location expressions must use the RA-assigned
   location of the out_ptr vreg.

## Status

Partially done. May need verification that `dwarf_expr_for_out_field` (currently
dead code) is either updated or removed.
