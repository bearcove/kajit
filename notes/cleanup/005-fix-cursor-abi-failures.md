# Fix remaining cursor ABI failures

Debug and fix the 2-3 native bugs in the borrowed-cursor path.

## Failing tests

1. `compiler::tests::postcard_hir_lowering_decodes_borrowed_header` — SIGSEGV on macOS/aarch64
   - Cleanest repro: borrowed string with borrowed cursor root arg
   - Assembly entry looks sane (x0=out, x1=ctx, x2=cursor_arg all arrive correctly)
   - Failure is deeper than "third arg never arrived"

2. `compiler::tests::postcard_hir_lowering_decodes_multi_options` — `DeserError { code: UnexpectedEof, offset: 0 }`
   - Previous failure was garbage UTF-8 length (110313459 bytes) — fixed by intrinsic arg setup fix
   - Current failure is narrower

3. `compiler::tests::postcard_hir_lowering_multi_options_matches_jit_differential_harness` — byte mismatch at index 8
   - Likely pointer noise in owned-string output, not semantic error
   - Needs semantic comparison instead of byte-for-byte

## Approach

- Use differential harnesses (HIR/IR/MIR/ASM interpreter vs JIT) — they already found 3 real bugs
- Linux + valgrind/rr/gdb may be easier than macOS for the SIGSEGV
- Start with `borrowed_header` (cleanest crash), then `multi_options`
- `KAJIT_TRACE_UTF8=1` for UTF-8 length issues
- Do NOT revert to shadow cursor model to make tests pass

## Key files

- `kajit/src/backends/aarch64/regalloc3_backend.rs` — primary suspect for native bug
- `kajit/src/intrinsics.rs` — verify host helper ABIs
- `kajit/src/lib.rs` — `RuntimeCursorArg` marshalling
- `kajit/src/arch/aarch64.rs` — frame setup if save-restore issue
