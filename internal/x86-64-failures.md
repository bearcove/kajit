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

**ROOT CAUSE FOUND (Prompt A12)**: RA block params are built from `live_in` bitset in ascending vreg order (`regalloc_mir.rs:1001-1012`). When LICM hoists 6 constants out of the theta body, they get lower vreg indices and sort BEFORE existing loop-carried values, shifting them by +6. Edge edits then read/write the wrong params. Without LICM: `b4 [v37, v39, v47, v48, v49]`. With LICM: `b4 [v13, v17, v19, v21, v23, v25, v37, v39, v47, v48, v49]`. This is a **positional indexing** bug — see issue #163.

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

**Prompt A4 diagnosis**: Also ruled out. All 24 edits are 64-bit. Slot stride is consistently 8 bytes. Emitted disassembly shows `qword [rsp+...]` throughout. No 32-bit reload or misalignment.

**New hypothesis**: The fast-path branch in the varint decoder uses a **signed comparison** instead of unsigned. For `0x80` (128 unsigned, -128 signed), a signed `< 128` check is TRUE, so the fast path is taken when it should not be. The fast path returns `byte & 0x7F = 0` or hits a validity check, producing `InvalidVarint`. This explains all values ≥ 128 failing — they all have the high bit set, making them negative in a signed comparison. See **Prompt A5** below.

**Prompt A5 diagnosis**: Also ruled out. Fast-path split is `(byte & 0x80) != 0` booleanized via `cmp/setnz/movzx`, then `test/jz`. Not a signed/unsigned `<` comparison at all. Slow path IS correctly taken for 0x80. Bug is inside the theta loop body or its exit condition. See **Prompt A6** below.

**Prompt A6 diagnosis**:
- With `-regalloc` (edits disabled): `UnexpectedEof` — loop runs past the buffer end; loop condition uses stale data (no edge edits to set it up correctly).
- With edits: `InvalidVarint` — loop terminates after exactly 10 iterations. `offset: 0` in the error is the input cursor position — it never advanced past byte 0. So the same byte (0x80, with continuation bit) is being read every iteration until the 10-byte safety limit fires.
- No intrinsic call in the loop body — `read_bytes(1)` is inline.
- Theta loop condition `v27` → `test r8, r8` / `jnz` is correct in structure.

**New hypothesis**: The cursor is not advancing inside the theta loop on x86_64. `read_bytes(1)` emits an inline cursor advance. One of the 24 edge edits on the theta back-edge is incorrectly restoring the cursor to its pre-loop value (spilled before the loop, reloaded on each iteration). Without those edits, the cursor advances correctly in a register but the loop condition is stale → `UnexpectedEof`. See **Prompt A7** below.

**Prompt A7 diagnosis**: Also wrong. Cursor is hardwired in r12 (non-allocatable). No back-edge edit touches r12. Cursor advances correctly.

**Pivot**: Seven hypotheses, seven dead ends. Stop theorizing top-down. Find WHERE `InvalidVarint` is actually emitted in the generated x86_64 code and trace backward from there. See **Prompt A8** below.

**Prompt A8 diagnosis**:
- `InvalidVarint` has two sites: `n35` (`last_cont != 0`) and `n43` (`extra != 0`), both in post-loop gamma nodes.
- Manual IR trace with `[0x80, 0x01]`: loop exits after 1 iteration (byte=0x01), `last_cont=0`, `last_low=1`, `extra=0`. **Neither site should fire.**
- IR is structurally identical between x86_64 and aarch64 dumps. aarch64 passes.
- Bug is purely in x86_64 codegen, downstream of IR.

**New hypothesis**: The post-loop gamma checks are using the WRONG values. `last_cont` (v24, theta output) and `last_low` (v14, theta output) are placed into registers by exit-edge edits. If the exit-edge edit puts `v26` (rem_bool=8≠0=true) where `v24` (cont_bool=0=false) is expected, the `last_cont != 0` check fires → `InvalidVarint`. The `-regalloc` case corroborates: without exit-edge edits, the wrong register is used and the loop runs too long → `UnexpectedEof` instead. See **Prompt A9** below.

**Prompt A9 diagnosis**: Exit-edge value crossing hypothesis NOT supported — there are no theta exit-edge edits. All 4 edits are back-edge edits on `lin=50, succ=0`. aarch64 has 0 edits total for this case.

**BREAKTHROUGH — per-pass bisect**:
- `-all_opts` → **PASS** (confirmed: bug is in an optimization pass)
- `-pass.bounds_check_coalescing` → FAIL
- `-pass.theta_loop_invariant_hoist` → **PASS** ← the culprit
- `-pass.inline_apply` → FAIL
- `-pass.dead_code_elimination` → FAIL

**Root cause is in `theta_loop_invariant_hoist`** (the LICM pass). It hoists something out of the theta loop that should stay inside, corrupting the loop's semantics. This also explains why the IR "looked correct" in earlier prompts — the IR was dumped post-optimization, showing the already-corrupted LICM output. See **Prompt A10** below.

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

### ~~Prompt A4~~ — DIAGNOSED (spill machinery is correct; all 64-bit)

### Prompt A5 — Read the actual disassembly; find the fast-path branch type

> You are investigating a miscompilation in kajit's x86_64 JIT for `postcard::scalar_u64_v3` (128u64, encoded as `[0x80, 0x01]`).
>
> **What has been ruled out:**
> - 32-bit register truncation
> - rcx/cl clobber before `shl r10, cl`
> - dst-lhs mismatch (emitter uses explicit r10 temp + store to dst)
> - Spill/reload width or layout (all 64-bit, slot stride 8 bytes)
>
> **Current hypothesis:** The fast-path branch in the varint decoder uses a signed comparison instead of unsigned. `0x80` = 128 unsigned = -128 signed. A signed `< 128` check is TRUE for 0x80, so the fast path fires when it should not, returning `0x80 & 0x7F = 0` or triggering a validity error.
>
> The key question is: what comparison instruction and jump type does the emitted x86_64 code use for the "is this byte a single-byte varint?" branch?
>
> Investigate:
> 1. Get the full disassembly of the JIT-compiled function for postcard scalar_u64. Use the existing disasm infrastructure in the test suite (e.g. `disasm_bytes` helper in `kajit/src/backends/x86_64/mod.rs`) or add a temporary `println!` call in a test. Alternatively, use `KAJIT_DUMP_STAGES` with stage `disasm` if it exists — check `docs/pipeline-debugging.md`.
> 2. In the disassembly, find the branch that splits the fast path (single-byte varint) from the slow path (multi-byte). What is the comparison instruction (`cmp`/`test`) and what is the conditional jump (`jl`/`jb`/`jge`/`jae`)? `jb`/`jae` are unsigned; `jl`/`jge` are signed.
> 3. In `kajit/src/backends/x86_64/emit.rs`, find how `CmpLt` (or whatever IR comparison op is used for the fast-path check) is emitted for u8 operands. Does it emit a signed or unsigned jump?
> 4. In the IR / LinearIr for this function (`KAJIT_DUMP_STAGES='ir,linear'`), find the fast-path comparison node. What type is the comparison — `u8`, `i8`, signed, unsigned?
> 5. On aarch64, what branch instruction is emitted for the same comparison? (`b.lo` = unsigned below; `b.lt` = signed less-than)
>
> Do not fix anything — produce a diagnosis with specific file:line evidence.

### ~~Prompt A5~~ — DIAGNOSED (not signed comparison; slow path correctly taken; bug is inside theta loop)

### Prompt A6 — Bisect with KAJIT_OPTS, then trace the theta loop exit condition

> You are investigating a miscompilation in kajit's x86_64 JIT for `postcard::scalar_u64_v3` (128u64, encoded as `[0x80, 0x01]`). All static analysis hypotheses have been ruled out. The slow path (theta loop) IS entered correctly. The bug is somewhere inside it.
>
> **What has been ruled out:**
> - 32-bit register truncation
> - rcx/cl clobber before `shl r10, cl`
> - dst-lhs mismatch (explicit r10 temp + store to dst)
> - Spill/reload width or layout (all 64-bit, slot stride 8 bytes)
> - Signed vs unsigned fast-path branch (it's a bitmask `!= 0` check, `jz/jnz`)
>
> **Step 1 — empirical bisect (do this first):**
>
> Run the failing test with `KAJIT_OPTS='-regalloc'` to disable application of regalloc edge edits:
> ```bash
> KAJIT_OPTS='-regalloc' cargo nextest run -p kajit --target x86_64-apple-darwin \
>   --test corpus -E 'test(=postcard::scalar_u64_v3)'
> ```
> Does the test pass or fail? Report the exact outcome (pass / fail with same error / fail with different error).
>
> - If it **passes**: one of the 24 edge edits is the bug. Identify which edit fires on which block edge inside the theta loop (entry edge, back-edge, or exit edge) and whether it corrupts a loop-carried value (accumulator or shift count).
> - If it **fails**: the bug is in base register allocation or emission. Proceed to Step 2.
>
> **Step 2 — theta loop exit condition:**
>
> In the RA-MIR dump (`postcard__scalar_u64__v3__x86_64__ra.txt`), find the `BranchIf` instruction that controls whether the theta loop continues. What vreg holds the condition? What physical register does regalloc assign it? What does the emitted x86_64 code look like for that branch (instruction + register)?
>
> Also: what intrinsic is called inside the theta loop to read the next byte? On x86_64, calls clobber rax, rcx, rdx, rsi, rdi, r8, r9, r10, r11. After the call, are loop-carried values (accumulator vreg, shift-count vreg) in caller-saved registers that the call may have clobbered — and if so, are they saved/restored by edits before/after the call?
>
> Do not fix anything — report the bisect outcome and produce a diagnosis with specific file:line evidence.

### ~~Prompt A6~~ — DIAGNOSED (cursor not advancing; edits restore cursor to pre-loop value on back-edge)

### Prompt A7 — Find the back-edge edit that clobbers the cursor; trace read_bytes emit

> You are investigating a miscompilation in kajit's x86_64 JIT for `postcard::scalar_u64_v3` (128u64).
>
> **Known**: With regalloc edits applied, the varint theta loop reads byte 0 (0x80) on every iteration — the input cursor never advances. After 10 iterations the 10-byte safety limit fires: `InvalidVarint at offset 0`. Without edits (`KAJIT_OPTS='-regalloc'`), the cursor DOES advance correctly (different failure mode: `UnexpectedEof`). Therefore one of the 24 edge edits on the theta loop back-edge is restoring the cursor register to its pre-loop (pre-advance) value, undoing the advance on each iteration.
>
> Investigate:
>
> 1. In `kajit/src/backends/x86_64/emit.rs:45`, read the full implementation of `emit_read_bytes`. What registers does it use? Which register holds the cursor (input position) before and after the emit? Specifically: does it read the cursor from a register, or from a memory field in the DeserContext struct? Does it write the updated cursor back to memory or keep it in a register?
>
> 2. In the RA-MIR dump (`postcard__scalar_u64__v3__x86_64__ra.txt`), find the `read_bytes(1)` instruction in loop block `b4`. What vregs are its inputs and outputs? Which vreg represents the cursor / input-position state, and what physical register (`pN`) does it get allocated to?
>
> 3. From the 24 edits, identify which ones are on the **theta back-edge** (the edge from the loop body back to the loop header). List them. Does any of them write to the physical register that holds the cursor vreg's output (after the read)?
>
> 4. If a back-edge edit writes the cursor register: what is the source of that write? Is it reloading from a stack slot that was spilled before the loop (i.e. the pre-advance cursor value)?
>
> 5. Compare to aarch64: in the aarch64 RA-MIR dump for the same case, what edits (if any) appear on the theta back-edge? Does aarch64 have the same edit pattern or not?
>
> Do not fix anything — produce a diagnosis with specific file:line evidence.

### ~~Prompt A7~~ — DIAGNOSED (cursor is hardwired in r12, non-allocatable, no edit touches it)

### Prompt A8 — Bottom-up: find where InvalidVarint is set, trace backward

> You are investigating a miscompilation in kajit's x86_64 JIT for `postcard::scalar_u64_v3` (128u64, `[0x80, 0x01]`). Seven static analysis hypotheses have been exhausted. The approach changes: find the ground truth first, then reason backward.
>
> **Step 1 — Find every place InvalidVarint can be set.**
>
> Search the kajit source for `InvalidVarint` — every place the error code is assigned, returned, or emitted inline (in Rust source, IR lowering, or inline emit code). List each site with file:line. Then look at the IR dump (`postcard__scalar_u64__v3__x86_64__ir.txt`) — are there `ErrorExit` nodes, and which ones use `InvalidVarint`?
>
> **Step 2 — Show the complete theta loop IR.**
>
> From the IR dump, show ALL nodes in the theta loop's body region. List them in order with their inputs and outputs. What is the full computation being performed per iteration?
>
> **Step 3 — Trace with input [0x80, 0x01] through the IR.**
>
> Walk through the theta loop body IR manually with:
> - Iteration 1: the byte being read is `0x01` (byte 1, since byte 0 was consumed by the fast/slow path split)
> - What should each node compute? What is the final accumulator value after the iteration?
> - What is the loop continuation condition value? Does the loop exit?
>
> **Step 4 — Does aarch64 produce the correct result?**
>
> Run the equivalent aarch64 test to confirm it passes, then note any structural differences between the aarch64 and x86_64 IR dumps for this case (they should be identical — if they differ, that difference is the bug).
>
> Do not fix anything — produce a diagnosis with specific file:line evidence.

### ~~Prompt A8~~ — DIAGNOSED (IR correct; both ErrorExit sites should NOT fire; bug is in x86_64 post-loop output values)

### ~~Prompt A9~~ — DIAGNOSED (no exit-edge edits; led to per-pass bisect breakthrough)

### ~~Prompt A10~~ — DIAGNOSED (LICM hoists 6 constants; code inspection says logic is correct; but dumps confirm structural change)

**Prompt A10 result**: LICM moves 6 `Const` nodes out of the theta body. The invariance classification logic IS correct (constants are genuinely invariant). But hoisting them changes the theta's interface: 6 new inputs/outputs, loop block arity changes, edit count jumps 18→24. The IR is semantically valid but the structural change causes x86_64 regalloc to miscompile.

### Prompt A11 — Diff theta wiring before/after LICM (see `.handoffs/a11-licm-wiring.md`)

### Prompt A9 — Trace the theta EXIT-edge edits; find which post-loop value is wrong

> You are investigating a miscompilation in kajit's x86_64 JIT for `postcard::scalar_u64_v3` (128u64, `[0x80, 0x01]`).
>
> **Known**:
> - IR is correct and identical to aarch64. Manual trace: loop exits after 1 iteration, `last_cont=0`, `last_low=1`, neither `InvalidVarint` site should fire.
> - With edits: `InvalidVarint` fires anyway — post-loop checks see wrong values.
> - Without edits (`-regalloc`): `UnexpectedEof` — loop runs too long (stale condition register).
> - Theta outputs `v14` (last_low = low 7 bits of last byte) and `v24` (last_cont = continuation bit of last byte) are placed into registers by exit-edge edits on the theta loop's exit.
>
> **Hypothesis**: An exit-edge edit is putting `v26` (rem_bool = true) where `v24` (cont_bool = false) is expected, or some equivalent value crossing. The post-loop `last_cont != 0` check then fires.
>
> **Step 1 — bisect with `-all_opts`:**
> ```bash
> KAJIT_OPTS='-all_opts' cargo nextest run -p kajit --target x86_64-apple-darwin \
>   --test corpus -E 'test(=postcard::scalar_u64_v3)'
> ```
> Does it pass or fail? If it passes, the bug is in an optimization pass. If it fails, the bug is in base codegen regardless of opts.
>
> **Step 2 — theta exit-edge edits:**
> From the 24 edits, identify which ones are on the **theta exit edge** (from the loop body/header to the post-loop block, taken when the loop condition is false). List them. What does each one move, and where does it put it?
>
> **Step 3 — post-loop register assignments:**
> In the RA dump, after the theta loop exits into the post-loop block, what physical registers hold `v14` and `v24`? Are those the same registers that the post-loop `n35`/`n43` gamma nodes read from?
>
> **Step 4 — compare exit-edge on aarch64:**
> In the aarch64 RA dump, what edits (if any) are on the theta exit edge? Does aarch64 have any crossed values there?
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
