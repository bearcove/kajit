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

**Working hypothesis**: There is a single bug in the x86_64 backend's handling of 64-bit integer operations — either in instruction selection (wrong register size: using 32-bit ops on 64-bit values), in the varint loop codegen, or in how u64/i64 intrinsics are called/returned. All 15 failures likely share this root cause.

---

## Group 2: Codegen snapshot diffs — control flow and intrinsic emission (2 failures)

### `linear_ir_micro_gamma_u32` — extra unconditional jump emitted

```diff
-  jmp $+0x17
+  jmp $+0x1c
   mov r10, 0x14
   mov rbx, r10
   mov r10, rbx
   mov dword [r14], r10d
+  jmp $+0x0
   mov qword [r15], r12
```

The jump offset grew by 5 bytes (the size of the newly inserted `jmp`), and an extra `jmp $+0x0` appears at what should be a fall-through point. The x86_64 backend is emitting an unconditional branch at a point where the next block is already the fall-through target.

### `linear_ir_micro_intrinsic_u64` — real address leaking into snapshot

```diff
-  mov rax, 0x<imm>
+  mov rax, 0x100cb8300
```

The snapshot normalization that replaces function pointer immediates with `0x<imm>` isn't firing for this test (or this test is new and has no stored snapshot). Secondary issue relative to Group 1.

---

## Group 3: IR optimizer — theta loop variant incorrectly hoisted (1 failure)

```
expected to keep/preserve: n3 = Add [arg0, arg1] -> [v2]
```

Failing test: `ir_opt_asserts_theta_loop_variant_not_hoisted`

The optimizer's loop-invariant code motion (LICM) pass is hoisting an `Add` node whose operands (`arg0`, `arg1`) are theta loop variables — i.e., they change each iteration. The LICM analysis incorrectly classifies this node as invariant. The test exists specifically to guard against this; it's now failing, meaning a recent change to the optimizer broke the invariance check for theta nodes.

---

## Investigation Prompts

### Prompt A — Root cause of u64/i64 x86_64 failures

> You are investigating why u64 and i64 decoding is broken on x86_64 in the kajit JIT compiler. All tests are run with `cargo nextest run --target x86_64-apple-darwin --no-fail-fast`.
>
> Start with the simplest failing case: `postcard::scalar_u64_v3`. This test encodes a u64 value using postcard's varint format and then decodes it with kajit's JIT-generated x86_64 code. It fails with `DeserError { code: InvalidVarint, offset: 0 }` — a decode error at the very first byte.
>
> Key observations:
> - `scalar_u64_v0`, `v1`, `v2` pass; `v3`–`v8` fail. The value in v3 is likely ≥ 128, requiring a multi-byte varint.
> - `json::all_scalars` fails because `a_i64` decodes as 0 instead of -1000000000000.
> - The `linear_backend_vec_u32_matches_serde` test gets the same `InvalidVarint` from `kajit/src/backends/x86_64/mod.rs:967`.
>
> Investigate:
> 1. What corpus value is in `scalar_u64_v3` (check `kajit/tests/` or `xtask/src/cases.rs`)?
> 2. What IR and LinearIr does kajit generate for a u64 field decode?
> 3. What x86_64 instructions does the backend emit for that LinearIr? Compare to what aarch64 emits.
> 4. Is the backend using 32-bit register variants (e.g., `eax` instead of `rax`) anywhere it should use 64-bit? Check `kajit/src/backends/x86_64/` thoroughly.
> 5. Is there a u64 intrinsic (e.g., `kajit_read_u64`) and does the x86_64 calling convention handle its 64-bit return value correctly?
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
