# Cherry-pick cursor branch Part 1 (HIR refactoring) — DONE

Already on main (cherry-picked previously).

14 commits on `codex/borrowed-cursor-abi-wip` that make HIR lowering generic. These are the architectural core of the generalization.

## Commits (oldest first)

1. `18a4e45` Make structural lowering HIR-driven
2. `9ed3eba` Classify HIR runtime intrinsics explicitly
3. `727c008` Make cursor binding explicit in HIR
4. `9488093` Make cursor entry state explicit in HIR
5. `c2a0ffc` Lower malum vec construction through HIR fields
6. `59f9a67` Make option init explicit in HIR
7. `75d024f` Remove option lowering special case
8. `3099d47` Remove runtime cursor parameter binding
9. `170cdea` Collapse decoder IR wrappers into HIR lowering
10. `8b9e399` Refresh HIR lowering terminology
11. `96067d5` Factor runtime dialect dispatch out of HIR lowering
12. `aa80736` Lower owned strings through explicit HIR steps
13. `18f3cd8` Make cursor ABI helpers explicit
14. `d52c579` Add HIR refs and indirect place lowering

## What they change

- Cursor: implicit shadow state → ordinary `&mut Cursor` parameter
- Options: Shape-coupled enum materialization → runtime vtable init
- Added to HIR: `Type::Ref`, `Expr::Deref`, `Place::Deref`, `Expr::AddrOf`
- `RuntimeIntrinsic` enum replaces string-based callable dispatch
- Parameter init: generic `data_arg_sources` distribution
- Removed: cursor shadow slots, sync logic, Shape-based option_defs caching

## Files touched

- `kajit-hir/src/lib.rs` — new types + intrinsic enum
- `kajit-hir/src/text.rs` — display for new types
- `kajit-hir-text/src/hir_parse.rs` — parse new types
- `kajit-postcard/src/lib.rs` — cursor as `&mut` param, vec/option/string lowering changes
- `kajit/src/compiler/hir_to_ir.rs` — structural place resolution, indirect access, deref lowering
- `kajit/src/compiler/mod.rs` — `RootDecoderDataAbi` enum
- `kajit/src/compiler/tests/` — snapshot updates

## Conflict risk

Low. These files had minimal changes on main (regalloc2 excision was in backends/regalloc, not HIR/postcard/hir_to_ir). Snapshot files may need regeneration.
