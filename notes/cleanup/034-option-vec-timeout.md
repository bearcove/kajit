# 021a: Option/vec corpus tests timeout (infinite loop in JIT)

## Problem

After the 011-2 / 021 changes, these corpus tests hang:

- `postcard::option_u32_v0`, `postcard::option_u32_v1`
- `postcard::option_string_v0`
- `postcard::vec_scalar_small`, `postcard::vec_scalar_large`
- `postcard::vec_u32_v0`, `postcard::vec_u32_v1`
- Corresponding `prop::*` variants

The interpreter path appears to work (no timeouts in `kajit-mir` interpreter
tests). The hang is in JIT-compiled code.

## Likely cause

These types use `CallIntrinsic` (option_init_none_ctx, option_init_some_ctx).
The backend's `emit_call_intrinsic` was rewritten to use RA-placed ABI args
instead of manual arg shuffling. Something in the post-call error check or
arg placement may be wrong, causing an infinite retry loop or corrupted
control flow.

## Likely root cause (updated)

The `out` parameter is typed as `u64` in HIR but is actually a pointer.
The frontend generates `addr_of(Place::Local(out_local))` to pass to vtable
init functions, but RVSDG value locals don't have addresses. The lowerer
treats `addr_of(Local)` as identity (returns the port source), which happens
to pass the pointer value — but the rest of the codegen may be confused by
the type mismatch.

See 031-type-out-as-pointer.md and 032-explicit-stores-through-out.md.

## Diagnosis approach

- Fix 031+032 first — the option/vec failures may resolve automatically
- If not: inspect generated assembly around CallIntrinsic sites
- Check if the post-call error check reads ctx from the right location
