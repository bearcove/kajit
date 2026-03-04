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

**Prompt A2 diagnosis**: rcx is not clobbered before `shl`. The 24 edits are spill/reload pressure, none write p1 at the shift site. rcx is live with the correct count when `shl r10, cl` executes.

**BUT**: the investigator reported concrete allocations `lhs=p7, rhs=p1, dst=p1`. On x86, `shl r10, cl` writes the result to r10 (lhs/p7). Yet regalloc allocated `dst=p1` (rcx). Any subsequent read of v15 (the shift result) reads from rcx, which still holds the old shift count. **This is the bug.** The result register is wrong.

Root cause: the Shl RA-MIR instruction's `dst` is being allocated to the same physical register as `rhs` (hw1/rcx) instead of being tied to `lhs`. See **Prompt A3** below.

**Prompt A3 diagnosis**: Also wrong. The emitter does NOT rely on x86 two-address semantics. It loads lhs → r10, operates (shl r10, cl), then explicitly stores r10 → dst (`emit_store_def_r10`). So dst=rcx is fine — the correct shifted value is written to rcx. Same pattern on aarch64: load lhs → x9, lsl x9, store x9 → dst. No dst-lhs tie is needed. Both hypotheses (rcx clobber, dst mismatch) are ruled out.

**New hypothesis**: With 24 spill edits on x86 vs 1 on aarch64, the spill/reload machinery is under heavy stress. Possible culprits: wrong stack slot assigned, wrong load width (32-bit load of a 64-bit spilled value), or frame setup that misaligns the stack and causes a reload to read garbage. See **Prompt A4** below.

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

### ~~Prompt A~~ — DIAGNOSED

### ~~Prompt A2~~ — DIAGNOSED (rcx not clobbered)

### ~~Prompt A3~~ — DIAGNOSED (dst=rhs is not a bug; emitter uses r10 temp + explicit store to dst)

### Prompt A3 — Confirm the dst=rhs misallocation for Shl and trace how it happens

> You are investigating a miscompilation in kajit's x86_64 JIT for `postcard::scalar_u64_v3` (128u64).
>
> **The bug**: For the `Shl` instruction in the varint theta loop, regalloc2 allocates `dst=p1` (rcx), but x86 `shl lhs, cl` writes the result to `lhs` (p7), not to rcx. Any use of the Shl result (v15) reads from rcx (still holding the shift count), not the shifted value.
>
> Concrete allocation observed: `lhs=p7, rhs=p1, dst=p1` at lin=39 for `v15 = Shl v14, v48/hw1`.
>
> Known facts:
> - `Shl` RHS is pinned to `hw1` (rcx) in RA lowering: `kajit-mir/src/regalloc_mir.rs:1214-1217`
> - `emit.rs:143-144` emits `shl r10, cl` (writes result to lhs register)
> - `dst=p1` means regalloc believes the output is in rcx — contradicting x86 semantics
>
> Investigate:
> 1. In `kajit-mir/src/regalloc_mir.rs`, find the `Shl` lowering. How is the `dst` operand passed to regalloc2 — is it expressed as a fixed constraint, a tied-to-lhs constraint, or a free allocation? Cite the exact lines.
> 2. In `regalloc_engine.rs`, how does the engine handle a binop whose result should be tied to its lhs? Is there a `tied_input` / `reuse_input` constraint mechanism?
> 3. Is the `dst` of `Shl` being marked as a fresh output (free allocation) instead of tied to lhs? If so, regalloc2 can freely assign it to any register including rcx — which it does, since rcx is already live there.
> 4. Check `emit.rs` at the Shl case: does it write the result to `dst` register or to `lhs` register? If it writes to lhs but regalloc assigned dst≠lhs, the result lands in the wrong place.
> 5. How does the aarch64 backend handle this? Does aarch64 Shl tie dst to lhs, and if so, where?
>
> Do not fix anything — produce a diagnosis with specific file:line evidence.

### Prompt A4 — Trace the actual spill/reload machinery for the varint loop on x86_64

> You are investigating a miscompilation in kajit's x86_64 JIT for `postcard::scalar_u64_v3` (128u64, encoded as two postcard varint bytes [0x80, 0x01]).
>
> **What has been ruled out:**
> - 32-bit register truncation — loads/stores are correctly 64-bit
> - rcx/cl clobber before `shl r10, cl` — rcx is correctly live at the shift site
> - dst-lhs mismatch — emitter uses r10 as temp, stores r10 → dst explicitly; dst=rcx is fine
>
> **Current hypothesis:** With 24 spill/reload edits on x86 vs 1 on aarch64, the failure may be in the spill machinery itself. Candidates: wrong stack slot index used for a reload, 32-bit load width when the spilled value is 64-bit, or a stack frame layout bug that misaligns slots and causes a reload to read garbage.
>
> Investigate:
> 1. Use `KAJIT_DUMP_STAGES='ir,linear,ra,edits'` + `KAJIT_DUMP_FILTER='postcard::scalar_u64__v3'` to produce x86_64 stage dumps (run a test that exercises the postcard scalar_u64_v3 corpus path, e.g. `cargo nextest run -p kajit --target x86_64-apple-darwin --test corpus -E 'test(=postcard::scalar_u64_v3)'`). Read `docs/pipeline-debugging.md` for the full dump workflow.
> 2. In the edits dump, list all 24 edits. For each stack-slot load (reload), note the slot index and the width of the load instruction emitted. Are all 64-bit vregs reloaded with 64-bit loads (QWORD), or are any using 32-bit (DWORD)?
> 3. In `kajit/src/backends/x86_64/edits.rs`, find the function that emits a stack-slot reload. What width does it use? Does it vary by vreg type, or is it always the same?
> 4. In `kajit/src/backends/x86_64/` (frame setup / stack allocation), how are stack slot sizes determined? Is there any place a 64-bit value might get a 32-bit slot?
> 5. As a direct empirical check: look at the actual emitted disassembly for the varint loop (from the x86_64 stage dump or by adding a disasm call). Do any of the reload instructions use `mov r_, dword [rsp+...]` where `QWORD` would be expected?
>
> Do not fix anything — produce a diagnosis with specific file:line evidence.

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
