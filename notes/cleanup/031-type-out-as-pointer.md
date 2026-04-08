# 031: Type `out` as a pointer in HIR

## Problem

The postcard frontend declares `out` as `u64`:

```
l3 param "out": u64
```

But `out` is actually a pointer to caller-allocated output storage. The caller
passes `output.as_mut_ptr()` — a `*mut T`. Treating it as `u64` causes:

1. **Scalar writes fail silently**: `init l3 = l5` overwrites the RVSDG port
   source (the pointer value) instead of writing through it. Result: 0.

2. **Tuple/struct field writes panic**: `init field(l3, "0") = l5` hits
   `resolve_local_field` which expects a Named struct type, but l3 is `u64`.
   Panics at hir_to_ir.rs:458.

3. **`addr_of(l3)` is semantically wrong**: For Option, the frontend emits
   `addr_of(Place::Local(l3))`. But RVSDG locals don't have addresses. What
   we actually want is to pass `l3` directly (it's already a pointer).

## The fix

Change the postcard frontend to declare `out` with a proper pointer type:

```
l3 param "out": &mut <root_output_type>
```

For `u32`: `&mut u32`. For `(u32, u32)`: `&mut t2` (where t2 is the tuple
struct type). For `Option<u32>`: `&mut t2` (where t2 is the Option enum type).

This means:
- `out` stays a pure RVSDG value (it's a pointer value in a register)
- The lowerer knows what `*out` points to from the type
- Field projections like `(*out).0` resolve correctly

## Where to change

- `kajit-postcard/src/lib.rs:3163` — change `hir::Type::u(64)` to
  `hir::Type::Ref { mutable: true, pointee: Box::new(root_output_type) }`
- The root output type needs to be computed from the shape being decoded

## Depends on

Nothing — this is a type annotation change.

## Enables

032 (explicit stores through out).
