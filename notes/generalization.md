# Kajit as a general-purpose compiler backend

Kajit is a JIT compiler. Deserialization is one use of it, not the defining feature. The pipeline (HIR → IR → LIR → CFG-MIR → machine code) is general. But deserializer assumptions are currently baked into every layer.

This note maps out what needs to change and how it relates to the pass-level cleanup in `notes/cleanup/`.

## The cursor branch (`codex/borrowed-cursor-abi-wip`)

17 commits of work toward this goal exist on a diverged branch. It hasn't been merged. Main diverged with the regalloc2 excision (19 commits). The branch has two parts:

**Part 1: HIR refactoring (14 commits, mostly clean).** This is the architectural core:
- Cursor became an ordinary `&mut Cursor` parameter instead of implicit shadow state
- Options use runtime vtable init instead of Shape-coupled enum materialization
- Added `Type::Ref`, `Expr::Deref`, `Place::Deref`, `Expr::AddrOf` to HIR
- `RuntimeIntrinsic` enum replaced string-based callable dispatch
- Parameter init became generic `data_arg_sources` distribution
- Removed: cursor shadow slots, sync logic, Shape-based option_defs caching
- `RootDecoderDataAbi::CursorRef` inference from HIR structure

**Part 2: WIP backend ABI plumbing (3 commits, partially broken).** Contains:
- Runtime marshalling for `RuntimeCursorArg` as third ABI argument
- Real regalloc3 bug fixes (ssa_coloring last-use, parallel-copy moves, intrinsic arg setup)
- Still has 2-3 native failures (SIGSEGV on borrowed_header, UnexpectedEof on multi_options)

The ssa_coloring fix already landed on main independently. The other fixes are valuable.

**Salvage plan:** Cherry-pick Part 1 onto main (low conflict — touches HIR/postcard/hir_to_ir, not the deleted regalloc2 backends). Then rebase Part 2 on top (higher conflict in aarch64 backend, but regalloc2 excision was mostly deletion, not modification of the regalloc3 backend).

## What's deser-specific today

### HIR (`kajit-hir/src/lib.rs`)
- `RuntimeIntrinsic::LoadInputPtr`, `LoadInputEnd`, `StoreInputPtr` — cursor ABI seams, temporary bridge during transition. Should be eliminated once cursor is fully ordinary data.

### IR (`kajit-ir/src/ir.rs`)
- **Hardcoded state domains:** `CURSOR_STATE_DOMAIN` (ID 0), `OUTPUT_STATE_DOMAIN` (ID 1), `MEMORY_STATE_DOMAIN` (ID 2). These should be user-defined, not baked in.
- **8 cursor ops:** `ReadBytes`, `PeekByte`, `AdvanceCursor`, `AdvanceCursorBy`, `BoundsCheck`, `SaveCursor`, `SaveInputEnd`, `RestoreCursor`. These should become ordinary loads/stores through a mutable reference.
- **Format-specific ops:** `ZigzagDecode` (postcard), `SimdStringScan`, `SimdWhitespaceSkip` (JSON). These should be intrinsics, not IR primitives.
- `cursor_advance` field in `IrOpMetadata`.

### Linear IR / CFG-MIR
Mirror of the 8 cursor ops above. `LinearOp` carries them; `cfg_mir::Inst` wraps `LinearOp`.

### Backends (`aarch64/regalloc3_backend.rs`, `x86_64/regalloc3_backend/`)
- `sync_ctx_cursor_around_calls` flag
- `cursor_writeback_reg` / `cursor_writeback_enc`
- Special prologue/epilogue for cursor registers (x19/x20 on aarch64)
- Hardcoded `CTX_INPUT_PTR`, `CTX_INPUT_END` offsets from `DeserContext`
- `SaveCursor`, `SaveInputEnd`, `RestoreCursor` emission with special register logic

### Runtime (`context.rs`, `intrinsics.rs`, `lib.rs`)
- `DeserContext` struct with hardcoded field layout
- All intrinsics take `&mut DeserContext` — varint reading, zigzag, bool, UTF-8 validation
- `invoke_decoder()` knows about `RuntimeCursorArg` specifically
- `compile_decoder()` as the public API name

### Postcard frontend (`kajit-postcard/src/lib.rs`)
- `input_region`, `cursor_type` fields
- `shape_has_input_borrow()` 
- Region parameterization for borrowed data

## What general-purpose looks like

Once the cursor is ordinary borrowed data:

1. **No cursor ops in IR.** `ReadBytes` becomes a load from a pointer field + pointer arithmetic. `BoundsCheck` becomes a comparison. `SaveCursor`/`RestoreCursor` become ordinary moves. The IR only has loads, stores, arithmetic, branches, and calls.

2. **No hardcoded state domains.** Domains are declared by the frontend (postcard declares a cursor domain, or doesn't — if the cursor is just a `&mut` param, it flows through the normal memory domain).

3. **No cursor knowledge in backends.** No special registers for cursor state. No sync-around-calls. The register allocator handles cursor-carrying registers like any other value. Prologue/epilogue are driven by what the regalloc needs saved, not by "is this a decoder."

4. **`DeserContext` becomes a user-defined struct.** The compiler doesn't know its layout. The frontend declares parameters with types; the backend passes them per the calling convention. Error reporting becomes an intrinsic call, not a magic struct field write.

5. **Intrinsics are pluggable.** Postcard registers varint-reading intrinsics. A different frontend registers different ones. The compiler sees `CallIntrinsic(fn_ptr, args)` and doesn't care what the intrinsic does.

6. **The public API is `compile_program()`, not `compile_decoder()`.** It takes an HIR module and returns compiled code. Whether that code deserializes postcard or evaluates Vixen rules or transforms data is the frontend's business.

## How the cleanups interact

### Do before or during generalization
- **000 (delete dead code)** — clears noise, do immediately
- **001-002 (test harness + golden tests)** — safety net. Pin pass behavior before changing what flows through them. Text-level tests survive the cursor→ordinary-data transition because they test IR transforms, not the cursor model.
- **005 (const_fold cleanup)** — independent, small

### Fall out naturally from generalization
- **003 (dead_theta_ports 3-port cap)** — the port structure changes when cursor state threading is removed from thetas. The bug may vanish or change shape. Don't fix it against the old model.
- **004 (gamma_output_partition compaction)** — the regalloc hint bug it triggers may be the same class of regalloc bug the cursor branch already found and fixed. Land those fixes first.
- **006 (post_unroll_canonicalize 800-node cap)** — unrolling cursor-carrying thetas changes once cursor is data. Revisit after.

### Still needed after generalization
- **007 (rewrite DCE)** — O(n²) is O(n²) regardless of cursor model
- **008 (consolidate domtree + replace_uses)** — infrastructure cleanup, independent
- **009 (simplify_gamma stub)** — if needed at all, still needed

## Execution order

See `notes/cleanup/` for detailed work items:

- `000` delete dead code
- `001` pass test harness
- `002` golden tests
- `003` cherry-pick cursor branch Part 1 (HIR refactoring)
- `004` rebase cursor branch Part 2 (backend ABI + regalloc fixes)
- `005` fix cursor ABI failures
- `006` remove cursor ops from IR
- `007` remove cursor knowledge from backends
- `008` generalize DeserContext
- `009` clean up const_fold
- `010` fix dead_theta_ports cap (after 006)
- `011` fix gamma_output_partition compaction (after 005)
- `012` fix post_unroll_canonicalize cap (after 006)
- `013` rewrite DCE
- `014` consolidate shared infrastructure
- `015` rewrite simplify_gamma
