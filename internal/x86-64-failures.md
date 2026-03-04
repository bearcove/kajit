# x86_64 Test Failures — 2026-03-04

20 failures total across 4 distinct groups.

---

## Group 1: u64/i64 runtime decode broken (15 failures)

The overwhelming majority of failures share a single error: the x86_64 backend produces code that fails to decode u64/i64 values correctly at runtime.

### Postcard — `InvalidVarint` at offset 0

```
DeserError { code: InvalidVarint, offset: 0 }
```

Failing tests:
- `postcard::scalar_u64_v3` through `postcard::scalar_u64_v8` (v0/v1/v2 pass)
- `postcard::deny_unknown_fields_v2`, `postcard::deny_unknown_fields_v3` (v0/v1 pass)
- `postcard::vec_scalar_large` (`vec_scalar_small` passes)
- `linear_backend_vec_u32_matches_serde` (same error, different test site: `kajit/src/backends/x86_64/mod.rs:967`)

The pattern of which versions fail is diagnostic: small u64 values (v0/v1/v2, small vecs) pass; larger values requiring multi-byte varint encoding fail. Postcard uses LEB128-style varints where values ≥ 128 need 2+ bytes. The error fires at offset 0, meaning the very first byte read produces a bad result — likely the x86_64 codegen for the varint loop is misreading or miswriting.

### Proptest — `InvalidVarint` at offset 0 for all input sizes

```
called `Result::unwrap()` on an `Err` value: DeserError { code: InvalidVarint, offset: 0 }
```

Failing tests:
- `prop::scalar_u64`
- `prop::flat_struct`
- `prop::nested_struct`
- `prop::transparent_composite`
- `prop::shared_inner_type`
- `prop::deny_unknown_fields`

Proptest generates random u64 values — most will be ≥ 128, triggering the same broken path. The corresponding `postcard::flat_struct` (fixed corpus, small values) passes.

### JSON — i64 decoded as wrong value

```
assertion `left == right` failed
  left:  AllScalars { a_i64: 0, ... }
  right: AllScalars { a_i64: -1000000000000, ... }
```

Failing test: `json::all_scalars`

The i64 field reads 0 instead of -1000000000000. Every other scalar is correct. This is almost certainly the same root cause as the postcard failures — the x86_64 codegen for 64-bit integer load/store/decode is broken. JSON encodes i64 in text; the JIT-generated decoder is presumably converting the string to an integer, and the 64-bit conversion path produces 0.

**Diagnosis (Prompt A)**: Not a 32-bit truncation bug, not a calling convention bug. Root cause is the `rcx/cl` fixed-register constraint for x86 shift instructions in the varint loop.

- `scalar_u64_v3` = `128u64` — first value requiring a multi-byte postcard varint
- IR and LinearIr are correct and identical across arches
- x86 RA-MIR forces `Shl` shift-count operand to `hw1` (rcx/cl): `kajit-mir/src/regalloc_mir.rs:1214-1217`
- x86 emits `shl r10, cl` and depends on that constraint: `kajit/src/backends/x86_64/emit.rs:143-144`
- This generates **24 register edits** on x86 vs **1** on aarch64 for this loop — massive spill/shuffle pressure
- `W8` loads/stores and shift ops are correctly 64-bit; `r10d` uses elsewhere are intentional (error codes, booleans)
- `kajit_read_u64` is an out-pointer intrinsic, not a return-value intrinsic; return-value path captures full `rax`

**Next question**: Are those 24 edge edits around the `rcx/cl` constraint correct? Is the shift count vreg being saved/restored correctly across loop iterations, or is a clobber of `rcx` corrupting the accumulated u64 value?

See **Prompt A2** below.

---

## ~~Group 2: Codegen snapshot diffs — control flow and intrinsic emission~~ FIXED (commit e4d888d / fix in x86_64/mod.rs)

**Root cause**: In `emit_terminator` (`kajit/src/backends/x86_64/mod.rs:394`), the fall-through suppression check was gated entirely on `term_linear_op_index` being `Some`. Synthetic `TempTerm::Fallthrough` blocks lowered to `RaTerminator::Branch` with `term_linear_op_index = None` always emitted a jump. Fix: extend the `None` branch to still check whether the branch target equals the next sequential block, same as the `Some` branch does.

Both backends had the same logic; aarch64 snapshots showed the analogous no-op `b $+0x4`. These snapshot tests were removed as part of the same cleanup batch.

---

## ~~Group 3: IR optimizer — theta loop variant incorrectly hoisted~~ MOOT (test deleted in first commit)

```
expected to keep/preserve: n3 = Add [arg0, arg1] -> [v2]
```

Failing test: `ir_opt_asserts_theta_loop_variant_not_hoisted` (was in `generated_ir_opt_corpus.rs`, now deleted)

**Actual root cause**: Not a LICM hoisting bug. The `Add [arg0, arg1]` node's result `v2` is unused, so DCE removes it before LICM runs. LICM invariance logic was correct — `arg0/arg1` are `RegionArg` with `region == body_region`, so LICM correctly classifies them as variant. The test (added `0b7ca157`, 2026-03-02) predated LICM (added `e1fba076`, 2026-03-03) and the test IR needed the node's result to be live so DCE wouldn't kill it before the assertion. Test deleted with the corpus cleanup; the LICM invariance logic itself is fine.

---

## Investigation Prompts

### ~~Prompt A~~ — DIAGNOSED (see above)

### Prompt A2 — Verify rcx/cl edge edits around the varint shift loop

> You are investigating a runtime failure in kajit's x86_64 JIT: postcard `scalar_u64_v3` (value = 128u64) fails with `DeserError { code: InvalidVarint, offset: 0 }`. The IR and LinearIr are correct. The root cause has been narrowed to the x86 shift constraint.
>
> Known facts:
> - x86 RA-MIR forces `Shl` shift-count to `hw1` (rcx/cl): `kajit-mir/src/regalloc_mir.rs:1214-1217`
> - x86 emits `shl r10, cl`: `kajit/src/backends/x86_64/emit.rs:143-144`
> - This generates 24 register edits for x86 vs 1 for aarch64 on this loop
> - The error fires at offset 0, meaning the accumulated value is wrong from the very first multi-byte read
>
> Use `KAJIT_DUMP_STAGES=1` (see `docs/pipeline-debugging.md`) to dump the x86_64 pipeline for `postcard::scalar_u64_v3` and examine the edits file.
>
> Investigate:
> 1. Read `docs/pipeline-debugging.md` to understand how to produce and read the stage dumps.
> 2. Look at the 24 edge edits in the x86_64 edits dump. What vregs are being moved around the shift instruction?
> 3. Is `rcx` (hw1) live with the shift-count value when `shl` executes, or has it been overwritten by another edit?
> 4. Does the varint loop use `rcx` for anything else (e.g., as a scratch reg for a move), potentially clobbering the shift count?
> 5. Is there a missing `rcx` clobber declaration for the `Shl` instruction that would cause regalloc2 to leave a live value in `rcx` across the shift?
>
> Do not fix anything yet — produce a diagnosis with specific file:line evidence.

---

### Prompt B — Extra jump at fall-through in x86_64 gamma codegen

> You are investigating why the x86_64 backend emits a spurious `jmp $+0x0` at a fall-through point in `linear_ir_micro_gamma_u32`. The snapshot diff shows:
>
> ```diff
> -  jmp $+0x17
> +  jmp $+0x1c
>    mov r10, 0x14
>    ...
>    mov dword [r14], r10d
> +  jmp $+0x0     ← spurious
>    mov qword [r15], r12
> ```
>
> The jump offset in the first line grew by 5 (= size of one `jmp rel32`), meaning the backend inserted a branch where none should be. The aarch64 backend presumably does not have this issue.
>
> Investigate:
> 1. Find where the x86_64 backend emits unconditional jumps for LinearIr blocks — look in `kajit/src/backends/x86_64/`.
> 2. Is there a "emit jump to successor" step that doesn't check whether the successor is already the fall-through block?
> 3. Compare to the aarch64 backend's branch emission to see if there's a missing fall-through check.
>
> Do not fix anything yet — produce a diagnosis with specific file:line evidence.

---

### Prompt C — Theta loop variant hoisted by LICM optimizer

> You are investigating why `ir_opt_asserts_theta_loop_variant_not_hoisted` fails. The error is:
>
> ```
> expected to keep/preserve: n3 = Add [arg0, arg1] -> [v2]
> ```
>
> This test asserts that an `Add` node whose inputs are theta loop arguments is NOT hoisted out of the loop. The test is failing, meaning the optimizer's LICM pass is incorrectly treating it as invariant.
>
> Investigate:
> 1. Find the test in `kajit/tests/generated_ir_opt_corpus.rs` around line 94 to understand the exact IR being tested.
> 2. Find the LICM pass in `kajit/` (likely under `kajit/src/` in an `optimize` or `passes` module).
> 3. How does the pass determine whether a node's inputs are loop-variant? Does it correctly identify theta region arguments as varying per iteration?
> 4. Was this test recently added to catch a known bug, or did a recent optimizer change break the invariance check?
>
> Do not fix anything yet — produce a diagnosis with specific file:line evidence.
