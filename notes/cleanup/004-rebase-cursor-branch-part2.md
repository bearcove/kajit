# Rebase cursor branch Part 2 (backend ABI + regalloc fixes)

3 commits from `codex/borrowed-cursor-abi-wip` with backend ABI plumbing and real bug fixes.

## Commits

1. `82dc17f` WIP borrowed cursor decoder ABI
2. `3b2bd8a` WIP borrowed cursor ABI and backend debugging
3. `7e99a3b` Add Linux handoff for borrowed cursor debugging

## Valuable bug fixes to preserve

- **regalloc_engine.rs:** `apply_moves()` fixed from sequential to parallel-copy semantics (snapshot all sources, then write destinations)
- **aarch64/regalloc3_backend.rs:** `emit_call_intrinsic()` arg setup fixed for parallel-copy safety (naive sequential loop was clobbering source registers)
- **arch/aarch64.rs:** frame setup/teardown fixes for reduced frame size when callee-saved saves are skipped

Note: the ssa_coloring.rs last-use fix already landed on main independently.

## Backend ABI plumbing

- `RootDecoderDataAbi::CursorRef` runtime marshalling via `RuntimeCursorArg`
- Third ABI argument (x2/rdx) for cursor struct
- `invoke_decoder()` dispatch for CursorRef path
- Regalloc register exclusion for data argument registers

## Conflict risk

Higher. The aarch64 backend had ~1700 lines of diff from regalloc2 excision on main. But that was mostly deleting the old backend, not modifying `regalloc3_backend.rs`. Conflicts should be resolvable by taking the cursor branch's additions.

## Known failures after rebase

- `postcard_hir_lowering_decodes_borrowed_header` — SIGSEGV
- `postcard_hir_lowering_decodes_multi_options` — UnexpectedEof
- `postcard_hir_lowering_multi_options_matches_jit_differential_harness` — byte mismatch (may be pointer noise)

These are the subject of 005.
