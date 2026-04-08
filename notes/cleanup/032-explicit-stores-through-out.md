# 032: Emit explicit stores through `out` pointer

## Problem

With `out` typed as `&mut T` (031), the frontend must stop generating
`init l3 = value` and instead emit writes through the pointee.

## Current HIR (broken)

**Scalar u32:**
```
stmt19: init l3 = l5          ← overwrites the pointer register
```

**Tuple (u32, u32):**
```
stmt19: init field(l3, "0") = l5   ← field on a non-struct type
stmt39: init field(l3, "1") = l8
```

**Option\<u32\>:**
```
stmt29: expr call c1(l4, @vtable, addr_of(l3), addr_of(l7))
                                   ^^^^^^^^^^^ addr_of a value local
```

## Target HIR (correct)

**Scalar u32:**
```
stmt19: assign deref(l3) = l5      ← store through the pointer
```
or equivalently:
```
stmt19: store w4(l3, l5)
```

**Tuple (u32, u32):**
```
stmt19: assign field(deref(l3), "0") = l5   ← field of pointee
stmt39: assign field(deref(l3), "1") = l8
```

**Option\<u32\>:**
```
stmt29: expr call c1(l4, @vtable, l3, addr_of(l7))
                                  ^^ l3 IS the pointer already
```

## What changes in the postcard frontend

The key insight: `lower_shape_into_place` is called with `Place::Local(out_local)`
as the destination. Instead, call it with:

```rust
Place::Deref { base: Box::new(Expr::Local(out_local)) }
```

This single change propagates correctly:
- Scalar: `push_init(stmts, Place::Deref{..}, value)` → `assign deref(l3) = value`
- Struct fields: recursive `Place::Field { base: Place::Deref{..}, field }` →
  `assign field(deref(l3), "name") = value`
- Arrays: `Place::Index { base: Place::Deref{..}, index }` →
  `assign index(deref(l3), i) = value`

For Option vtable calls, `addr_of(place)` where place is `Place::Local(out_local)`
must become just `Expr::Local(out_local)` — the pointer itself.

## Where to change

- `kajit-postcard/src/lib.rs:3136` — change `Place::Local(out_local)` to
  `Place::Deref { base: Box::new(Expr::Local(out_local)) }`
- `kajit-postcard/src/lib.rs:~2828` — option vtable calls: pass `l3` directly
  instead of `addr_of(Place::Local(l3))`

## Lowerer support (already exists)

The scalar lowerer already handles:
- `Place::Deref` writes → `store_value_to_addr` (hir_to_ir.rs:1004-1010)
- `Place::Field` with non-local base → `lower_place_addr` + `store_value_to_addr`
  (hir_to_ir.rs:998-1001)
- `Place::Index` writes → element address computation + store (hir_to_ir.rs:1012-1023)
- `StmtKind::Store` → direct store emission (hir_to_ir.rs:753-757)

## Depends on

031 (out must be typed as `&mut T` so the lowerer knows the pointee type).

## Enables

Fixes all corpus test failures:
- Scalar v3+ variants (wrong output: 0 instead of expected)
- Tuple types (panic at resolve_local_field)
- Option types (panic at addr_of)
- Array types (panic at index place)
