# 033: Fix addr_of semantics for pointer-valued locals

## Problem

`Expr::AddrOf(Place)` means "compute the address of this storage location."
RVSDG value locals don't have storage locations — they're pure register values.
So `addr_of(Local(l3))` is only meaningful if `l3` is addressable.

Currently the lowerer handles `AddrOf` by calling `lower_place_addr`, which
for `Place::Local` just returns the local's port source value — effectively
treating it as an identity operation. This is accidentally correct when `l3`
holds a pointer, but semantically wrong.

## After 031+032

With `out` typed as `&mut T` and writes going through `deref(out)`:
- `addr_of(deref(l3))` → evaluates to `l3` (the pointer value). Correct.
- `addr_of(field(deref(l3), "x"))` → `l3 + field_offset`. Correct.
- `addr_of(Local(l3))` → should be illegal for pointer-valued params.

## What to change

After 032 lands, audit all `addr_of(Place::Local(...))` uses:
- Option vtable calls should pass `l3` directly (not `addr_of(l3)`)
- For payload locals like `l7` that are value locals, `addr_of(l7)` is used
  to pass their address to vtable init functions. These locals need to either:
  - Be stack-allocated (lowerer emits `StackAlloc` + store, then passes address)
  - Or be passed by value if the callee supports it

This is a correctness issue for the Option/enum path specifically.

## Depends on

032 (explicit stores through out).

## Status

May be partially solved by 032 if the frontend stops using `addr_of` on
`out_local`. The remaining `addr_of` usage is for payload locals passed to
vtable init functions — that's a separate problem (stack allocation for
value-to-address conversion).
