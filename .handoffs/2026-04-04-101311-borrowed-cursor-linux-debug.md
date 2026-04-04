# Handoff: Borrowed Cursor ABI Linux Debug

## Completed
- Added root decoder `CursorRef` ABI plumbing through compiler/backend/runtime on branch `codex/borrowed-cursor-abi-wip`, current pushed commit `3b2bd8a`.
- Made HIR support `&mut` references and `deref(...)` places earlier in this branch so postcard can model cursor as ordinary borrowed struct state.
- Fixed one real regalloc3 allocator bug in `kajit-mir/src/regalloc3/ssa_coloring.rs` around last-use accounting for dying values.
- Fixed one real regalloc3 AArch64 backend bug in `kajit/src/backends/aarch64/regalloc3_backend.rs` where `CallIntrinsic` arg setup was not parallel-move safe.
- Fixed one real post-regalloc simulator bug in `kajit-mir/src/regalloc_engine.rs` where regalloc edits were applied sequentially instead of as parallel copies.
- Fixed AArch64 frame setup/teardown bugs in `kajit/src/arch/aarch64.rs` where reduced frame size was incorrectly used when some fixed callee-saved saves were skipped.

## Active Work

### Origin
User’s original request started as:
- "Hi! I think there's two lowering paths right now; one for deserialization. and a generic one. I think we wanted to move deserialization to the generic one. can you help?"
- "speaking of JSON, let's remove /all of it/"

That work progressed into removing decoder-specific lowering structure and making postcard go through generic HIR lowering. The current open task came from the user pushing further on cursor handling:
- "sure, but passing the cursor as a mutable ref to a struct is very much a goal."
- "no more special/shadow cursor knowledge."
- "go for it"
- "next cut please (after commit)"

Later, when the borrowed-cursor ABI work started to shake out backend bugs, the user explicitly said:
- "Also, `-regalloc` is useless, it shouldn't even exist"
- "would this be easier to debug on linux, with valgrind etc.? we can switch oses if you want."
- "are you using differential harnesses (e.g HIR/IR/MIR/ASM interpreter vs JIT) to help find  the bug?"
- "push everything and give me a handoff for an agent to work on linux then. a COMPREHENSIVE handoff"

### The Problem
The architectural goal is correct and mostly implemented: root decoders can now take a borrowed cursor-like root data argument instead of relying entirely on privileged runtime cursor state. But the current branch is not correct yet.

There are still native/runtime bugs in the borrowed-cursor path after the ABI shift:

1. `postcard_hir_lowering_decodes_borrowed_header` crashes with SIGSEGV on macOS/aarch64.
2. `postcard_hir_lowering_decodes_multi_options` returns `UnexpectedEof`.
3. `postcard_hir_lowering_multi_options_matches_jit_differential_harness` now reports a raw output-byte mismatch, but after the latest fixes this looks likely to be mostly heap-pointer noise rather than the earlier garbage-length UTF-8 failure.

Important distinction:
- The old failure mode for `multi_options` was definitely real: JIT called `kajit_validate_utf8_range` with a correct data pointer and a garbage length like `110313459`, which was traced with `KAJIT_TRACE_UTF8=1`.
- That specific bug improved after fixing `CallIntrinsic` arg setup in the regalloc3 AArch64 backend.
- What remains is narrower: at least one real borrowed-cursor native bug (`borrowed_header` SIGSEGV), and probably one remaining runtime/backend correctness bug for `multi_options` in the direct decode path.

### Current State
- Branch: `codex/borrowed-cursor-abi-wip`
- Remote branch: `origin/codex/borrowed-cursor-abi-wip`
- Latest pushed commit: `3b2bd8a` (`WIP borrowed cursor ABI and backend debugging`)
- No PR opened yet.

What is already in this branch:
- Root decoder ABI metadata and runtime marshalling for `CursorRef`.
- Regalloc3 backend support for root `data_args` in decoder functions.
- Cursor modeling in HIR as ordinary borrowed data (`&mut` + `deref(...)`).
- Various backend/regalloc fixes discovered while chasing the borrowed-cursor failures.

What is still broken:
- The branch is intentionally WIP and does not pass all relevant tests.
- I pushed with `--no-verify` specifically because the branch still fails targeted tests.

### Technical Context

#### 1. The current borrowed-cursor ABI model
Compiler/runtime now recognizes a root decoder ABI with one extra data arg:
- `kajit/src/compiler/mod.rs`
- `kajit/src/lib.rs`
- `kajit/src/ir_backend.rs`
- `kajit/src/backends/aarch64/regalloc3_backend.rs`

Key shape:
- `RootDecoderDataAbi::CursorRef`
- runtime side in `kajit/src/lib.rs` builds:
  - `RuntimeSliceU8 { ptr, len }`
  - `RuntimeCursorArg { bytes, pos }`
- root decoder invocation becomes:
  - `fn(*mut u8, *mut DeserContext, *mut RuntimeCursorArg)`

Important runtime snippet in `kajit/src/lib.rs`:
- `invoke_decoder` constructs a stack `RuntimeCursorArg`
- passes `&mut cursor` as third arg
- after return writes back:
  - `ctx.input_ptr = cursor.bytes.ptr.wrapping_add(cursor.pos as usize);`

This part looked sane when inspected.

#### 2. HIR/lowering side is not the main blocker anymore
The current failures are below HIR and generic HIR lowering.

Facts already established:
- postcard HIR is using an ordinary borrowed cursor param
- generic HIR supports:
  - `Type::Ref { mutable, pointee }`
  - `Place::Deref`
  - `Expr::Deref`
- destination-writing lowering initializes non-destination params correctly
- postcard still relies on a temporary ABI seam for `load_input_ptr/load_input_end/store_input_ptr`, but the crashing path is now in backend/runtime behavior, not “HIR can’t express it”

The user should not be pulled back into redesign here yet. The next session should debug the existing native/runtime bug first.

#### 3. Differential harnesses already used and what they proved
Yes, differential harnesses were used repeatedly.

They exposed three distinct classes of issues:

1. Real regalloc allocator bug
- File: `kajit-mir/src/regalloc3/ssa_coloring.rs`
- Symptom before fix: bogus allocator panic like “no register available for v300…”
- Root cause: last-use bookkeeping had been converted to 1-based instruction numbering, but dying-source checks still compared against 0-based `inst_idx`
- Result: dying values were not freed on time

2. Real post-regalloc simulator bug
- File: `kajit-mir/src/regalloc_engine.rs`
- `apply_moves(...)` applied regalloc edits sequentially
- but regalloc edits are parallel-copy semantics
- This produced false post-regalloc differential divergences
- Fixed by snapshotting all source values first, then writing destinations

3. Real regalloc3 AArch64 intrinsic-call arg bug
- File: `kajit/src/backends/aarch64/regalloc3_backend.rs`
- `emit_call_intrinsic(...)` used a naive sequential loop to move explicit args into `x1+`
- That can clobber a later source register before it is read
- After fixing this to use parallel-copy semantics for register-resident args, the earlier bogus UTF-8 length failure changed into a much narrower mismatch

This means the harnesses were useful and should continue to be used on Linux.

#### 4. Important test cases
These are the main repros:

- `cargo nextest run -p kajit -E 'test(=compiler::tests::postcard_hir_lowering_decodes_borrowed_header)'`
  - current behavior on macOS/aarch64: SIGSEGV
  - this is the cleanest “borrowed string with borrowed cursor root arg” repro

- `cargo nextest run -p kajit -E 'test(=compiler::tests::postcard_hir_lowering_decodes_multi_options)'`
  - current behavior: `DeserError { code: UnexpectedEof, offset: 0 }`

- `cargo nextest run -p kajit -E 'test(=compiler::tests::postcard_hir_lowering_multi_options_matches_jit_differential_harness)'`
  - previously: JIT `InvalidUtf8`
  - after latest backend fix: `FirstDivergentByte { index: 8, ... }`
  - likely not a pure semantic mismatch anymore; byte 8 is in owned-string host data and may just be a pointer byte

- `cargo nextest run -p kajit -E 'test(=compiler::tests::postcard_hir_lowering_multi_options_matches_post_regalloc_simulation)'`
  - current result should be treated carefully
  - some remaining mismatches may still be raw pointer bytes in output snapshots rather than semantic errors

- `cargo nextest run -p kajit-mir -E 'test(regalloc_engine::tests::differential_)'`
  - this passed after the regalloc fixes and should remain green

#### 5. `KAJIT_TRACE_UTF8` output that mattered
This was the strongest clue before the latest backend patch:

Running:
```bash
KAJIT_TRACE_UTF8=1 cargo nextest run -p kajit -E 'test(=compiler::tests::postcard_hir_lowering_multi_options_matches_jit_differential_harness)'
```

Produced:
- `kajit_validate_utf8_range` called with a huge bogus `len=110313459`
- data bytes started with `68 65 6c 6c 6f ...`, so the pointer was plausibly near the correct `"hello"` payload
- this strongly implicated bad explicit-arg setup to a ctx-first intrinsic call, not a totally random pointer

After the backend arg-shuffle fix, that specific symptom disappeared. Keep this in mind so the Linux session does not waste time rediscovering it.

#### 6. Borrowed-header assembly/artifact facts already established
Artifacts I dumped on macOS:
- `/tmp/borrowed-header.hir.txt`
- `/tmp/borrowed-header.cfg.txt`
- `/tmp/borrowed-header.asm.txt`
- `/tmp/borrowed-header.root-abi.txt`

Important facts from those:
- root ABI inference was correct: `CursorRef`
- borrowed-header HIR looked correct:
  - param `cursor: &mut Cursor`
  - struct fields `bytes` and `pos`
  - explicit borrowed-field projections via `deref`
- assembly entry looked sane:
  - `mov x21, x0`
  - `mov x22, x1`
  - `mov x23, x2`
  so root cursor arg was at least entering the function in the expected place

This did not solve the SIGSEGV, but it means the failure is deeper than “third arg never arrived.”

#### 7. Files changed in this WIP commit
These are the main touched files and why they matter:

- `kajit-mir/src/interpreter.rs`
  - root cursor arg support in interpreter-side runtime model

- `kajit-mir/src/regalloc3/ssa_coloring.rs`
  - allocator last-use bug fix

- `kajit-mir/src/regalloc_engine.rs`
  - root data arg simulation support
  - post-regalloc `apply_moves` parallel-copy fix

- `kajit/src/arch/aarch64.rs`
  - frame-size/prologue/epilogue fixes

- `kajit/src/backends/aarch64/calls.rs`
  - still contains the older backend’s call conventions; useful as the correct conceptual reference for `CallIntrinsic` vs `CallPure`

- `kajit/src/backends/aarch64/mod.rs`
  - touched as part of backend integration / formatting

- `kajit/src/backends/aarch64/regalloc3_backend.rs`
  - main active seam
  - root data arg materialization
  - cursor/ctx ABI glue
  - call lowering bugs
  - intrinsic arg setup fix

- `kajit/src/compiler/mod.rs`
  - root decoder ABI inference / wiring

- `kajit/src/compiler/tests/mod.rs`
  - contains the focused failing tests and some temporary dump-writing helpers

- `kajit/src/intrinsics.rs`
  - runtime helper signatures, especially `kajit_validate_utf8_range`

- `kajit/src/ir_backend.rs`
  - root ABI plumbed through backend compile entry

### Success Criteria
1. `cargo nextest run -p kajit -E 'test(=compiler::tests::postcard_hir_lowering_decodes_borrowed_header)'` passes on Linux.
2. `cargo nextest run -p kajit -E 'test(=compiler::tests::postcard_hir_lowering_decodes_multi_options)'` passes on Linux.
3. `cargo nextest run -p kajit -E 'test(=compiler::tests::postcard_hir_lowering_multi_options_matches_jit_differential_harness)'` either:
   - compares semantically and passes, or
   - is updated so pointer bytes in owned-string outputs are not treated as semantic mismatches.
4. No reliance on `-regalloc` as a solution path; fix the real native/runtime/backend issue.
5. Root borrowed-cursor ABI remains in place; do not revert to special shadow-cursor lowering to make tests pass.

### Files to Touch
- `/Users/amos/bearcove/kajit/kajit/src/backends/aarch64/regalloc3_backend.rs`
  - primary likely file for remaining native bug
  - inspect `emit_call_intrinsic`, root `data_args` prologue setup, and any path that still assumes ctx/cursor specialness

- `/Users/amos/bearcove/kajit/kajit/src/intrinsics.rs`
  - verify exact host helper ABIs, especially ctx-first intrinsics

- `/Users/amos/bearcove/kajit/kajit/src/compiler/tests/mod.rs`
  - contains repros
  - likely place to harden/adjust differential expectations if raw pointer bytes are the only mismatch

- `/Users/amos/bearcove/kajit/kajit-mir/src/regalloc_engine.rs`
  - if Linux differential work finds more simulation issues
  - especially around root data args and output comparison semantics

- `/Users/amos/bearcove/kajit/kajit/src/lib.rs`
  - only if the runtime marshalling of `RuntimeCursorArg` turns out to be wrong in practice

- `/Users/amos/bearcove/kajit/kajit/src/arch/aarch64.rs`
  - only if Linux debugging shows another frame/save-restore bug

### Decisions Made
- Do not back out the borrowed-cursor ABI shift. The user explicitly wants cursor as an ordinary borrowed struct and no more shadow/special cursor knowledge.
- Do not use `-regalloc` as a “green path.” The user explicitly rejected that.
- Use differential harnesses first to classify bugs by layer before dropping straight into native debugging.
- Treat raw owned-string pointer bytes with suspicion in differential output. After the intrinsic arg fix, some remaining byte-level mismatches may not be semantic.
- Prefer real backend/runtime fixes over adapters or fallback paths. This matches repo instructions and the user’s direction.

### What NOT to Do
- Do not revert to the old special/shadow cursor model.
- Do not add backend workarounds in HIR/lowering just because the native path is buggy.
- Do not use `KAJIT_OPTS='-regalloc'` as the answer.
- Do not “fix” the branch by disabling or deleting the failing tests.
- Do not assume a byte-for-byte output mismatch on owned `String` means semantic failure without checking whether the differing bytes are just heap addresses.

### Blockers/Gotchas
- The pushed branch is intentionally WIP and does not pass the targeted tests above.
- On macOS/aarch64, JIT crash debugging is slower/more awkward than it would be on Linux.
- `borrowed_header` is still the cleanest real native crash; use it first on Linux with valgrind/rr/gdb.
- Some `multi_options` differential mismatches compare raw serialized output memory containing heap pointers for owned strings; that can be misleading.
- The old `KAJIT_TRACE_UTF8` failure is already partly fixed; if you still see giant bogus lengths on Linux, re-check whether the current local backend changes actually built and are being exercised.

## Bootstrap
```bash
git switch codex/borrowed-cursor-abi-wip
git pull --ff-only

# Clean baseline: reproduce the main failures
cargo nextest run -p kajit -E 'test(=compiler::tests::postcard_hir_lowering_decodes_borrowed_header)'
cargo nextest run -p kajit -E 'test(=compiler::tests::postcard_hir_lowering_decodes_multi_options)'
cargo nextest run -p kajit -E 'test(=compiler::tests::postcard_hir_lowering_multi_options_matches_jit_differential_harness)'

# Keep the MIR differential harness green while iterating
cargo nextest run -p kajit-mir -E 'test(regalloc_engine::tests::differential_)'

# If Linux tooling is available, start with the borrowed-header crash
# and only then revisit whether the multi_options byte mismatch is semantic or pointer noise.
```
