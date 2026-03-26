# Unroll-Specialize-Clean Pipeline

## The Goal

Close the 3.6x gap between kajit (3.2 ns) and serde/LLVM (0.9 ns) on
scalar u32 varint decode. Serde produces 15 straight-line instructions
with `bfi`/`tbz`. Kajit produces 83 loop-based instructions with
forwarding blocks, dynamic shift computation, and redundant moves.

The path: **unroll bounded thetas at RVSDG level, specialize each copy
with its iteration index, then clean up the resulting structure** so it
lowers to straight-line code.

## Current State (2026-03-26)

### What works

- **Regalloc3**: 99/99 corpus tests, deterministic (BTreeSet), unified
  location-based parallel copy resolver, coalesce phase enabled.
- **Bounded-theta unroller** (`kajit-ir/src/unroll_theta.rs`): exists,
  works correctly (99/99 with `KAJIT_UNROLL=1`), gated behind env var.
- **Analysis infrastructure**: dominance, loops, def-use all exist.
- **CFG-MIR passes**: const_phi_elim, remat, cse, gvn, copyprop,
  fuse_cmpz, elim_imm, dce, loop_phi_elim, const_branch_fold,
  merge_blocks, simplify_phis.

### What's broken

Unrolling **regresses** performance: 3.2 ns -> 3.8 ns. The unrolled
code is 313 instructions in 66 blocks (vs 83 instructions in 16 blocks).
Each of the 5 iterations is a full copy of the loop body with dynamic
computation -- the iteration index isn't specialized.

### The assembly evidence

**Serde (0.9 ns, 15 instructions)**:
```asm
cbz x1, error                ; empty input
ldrb w9, [x0]               ; byte 0
and w0, w9, #0x7f           ; bits[0:7]
tbz w9, #0x7, done           ; 1-byte fast path
ldrsb w9, [x8, #0x1]        ; byte 1
bfi w0, w9, #0x7, #0x7      ; bits[7:14]
tbz w9, #0x1f, done          ; 2-byte fast path
ldrsb w9, [x8, #0x2]        ; byte 2
bfi w0, w9, #0xe, #0x7      ; bits[14:21]
tbz w9, #0x1f, done          ; 3-byte fast path
ldrsb w9, [x8, #0x3]        ; byte 3
bfi w0, w9, #0x15, #0x7     ; bits[21:28]
tbnz w9, #0x1f, overflow     ; 4-byte overflow check
ret
```

Key properties: unrolled, `bfi` (bitfield insert), `tbz`/`tbnz`
(test-and-branch), no loop, no stack frame, 3 registers.

**Kajit without unroll (3.2 ns, 83 instructions)**:
```asm
; prologue: 8 insns (stack frame, callee-saves)
; loop setup: 11 insns (5 are phi-init movs)
.L4:                          ; loop header: bounds check
  add x13, x1, #1
  cmp x13, x4
  cset x13, hi
  cbz x13, .L6
.L6:                          ; loop body: decode one byte
  add x8, x0, x1             ; addr = base + offset
  ldrb w9, [x8]              ; load byte
  mul x11, x6, x14           ; shift = iter * 7  <-- DYNAMIC
  lsl x11, x12, x14          ; shifted = (byte & 0x7f) << shift
  orr x11, x5, x12           ; accum |= shifted
  ; ... 18 insns total per iteration ...
.L7:                          ; check high bit
  tbnz x7, #7, .L19          ; more bytes -> continue
; ... forwarding blocks ...
.L20: b .L15                  ; forwarding
.L21: b .L14                  ; forwarding
.L22: b .L14                  ; forwarding
.L23: b .L4                   ; back-edge forwarding
```

Key problems: loop, `mul iter, 7` (dynamic shift), forwarding blocks,
redundant movs, stack frame for callee-saves.

**Kajit with unroll (3.8 ns, 313 instructions, SLOWER)**:
```asm
; iteration 0:
  ldrb w9, [x5]
  mul x11, x1, x8            ; shift = 0 * 7 = 0  <-- SHOULD BE CONST
  lsl x11, x7, x8
  orr x11, x1, x7
  ; ... 18 insns + 12 forwarding insns = 30 per iteration ...
; iteration 1:
  ldrb w9, [x8]
  mul x11, x6, x14           ; shift = 1 * 7 = 7  <-- SHOULD BE CONST
  ; ... same 30 insns ...
; iterations 2, 3, 4: same pattern
```

5 copies x 30 = 150 hot instructions. The gamma cascade from
unrolling creates MORE merge blocks, and the iteration index stays
dynamic. Unroll without specialize makes things worse.

## The Pipeline

### Phase 1: RVSDG Iteration Specialization

**Where**: `kajit-ir/src/unroll_theta.rs` or a new post-unroll pass.

**What**: After unrolling a theta into a gamma cascade, substitute the
iteration index as a constant in each clone body. For clone k of an
N-iteration unroll:

- The loop counter arg = constant k
- `mul k, 7` folds to `7*k`
- `add k, 1` folds to `k+1`
- Bounds check predicates on k may become statically known

**Two approaches**:

1. *During unroll*: When cloning the body for iteration k, replace the
   iteration-counter region arg with a Const node producing k. Existing
   RVSDG simplification passes (const folding, dead node elimination)
   then propagate this.

2. *Post-unroll canonicalization*: Run a fixpoint of
   const-fold + simplify + DCE after unrolling. This is the MLIR
   "transform then canonicalize" pattern.

Approach 1 is simpler and more targeted. The unroller already knows
which arg is the iteration counter.

**Expected result**: `mul iter, 7` in each clone becomes a constant
(0, 7, 14, 21, 28). `add iter, 1` becomes a constant. Branch
predicates on the counter may fold.

### Phase 2: RVSDG Gamma Simplification

**Where**: `kajit-ir/src/ir_passes.rs` (new or enhanced pass).

**What**: After iteration specialization, many gamma branches have
known predicates. Simplify:

- **Constant-predicate gamma**: If the gamma selector is a constant,
  replace the gamma with its selected branch body. This collapses
  the cascade for iterations that are statically known to continue.

- **Trivial gamma elimination**: If both branches of a gamma produce
  identical results, replace with the result directly.

- **Passthrough elimination**: If a gamma branch just forwards its
  inputs to outputs (no computation), the gamma is a no-op and can be
  removed.

- **Dead region elimination**: After folding, some gamma branches are
  unreachable. Remove them.

Run these to a fixpoint after unrolling + specialization.

**Expected result**: The 5-deep gamma cascade collapses. For a 1-byte
varint (the common case), the fast path becomes a straight-line sequence
with no branches -- exactly serde's shape.

### Phase 3: CFG SimplifyCFG-lite

**Where**: `kajit-mir/src/opt/` (new pass or enhanced merge_blocks).

**What**: After RVSDG simplification, the remaining structure lowers
to CFG-MIR. Clean up the CFG artifacts:

- **Unconditional branch chain collapse**: If block A branches
  unconditionally to B, and B branches unconditionally to C, rewrite A
  to branch directly to C. Compose phi args through the chain.

- **Constant branch folding** (enhanced): `const_branch_fold` already
  exists. Ensure it handles the patterns from gamma-cascade lowering
  where branch predicates are constants or single-use comparisons
  against constants.

- **Merge block forwarding**: If a merge block (multiple preds) has no
  instructions and only branches unconditionally, and all predecessors
  supply the same value for a param, eliminate that param and forward
  the value.

- **Identical-arg phi collapse**: If all incoming edges supply the same
  value for a block param, replace the param with that value. This is
  the dual of loop-invariant phi elimination, but for non-loop merge
  points.

**Expected result**: 66 blocks -> ~20 blocks. Forwarding jumps
eliminated. Phi params reduced to only genuinely divergent values.

### Phase 4: Jump Threading

**Where**: `kajit-mir/src/opt/` (new pass).

**What**: For the remaining merge-and-branch diamonds where one
predecessor determines the successor:

```
b_pred: branch_if cond -> e_true, e_false
b_merge(cond_result): branch_if cond_result -> e_next, e_other
```

Thread b_pred's true edge directly to e_next, skipping b_merge.

This is a targeted version of LLVM's JumpThreading: when a predecessor
constrains which successor a block takes, forward the edge directly.

**Expected result**: Eliminates the remaining forwarding hops on hot
paths. The varint "done, no more bytes" path goes directly to the
range check and store, without bouncing through merge blocks.

### Phase 5: Backend Combines

**Where**: `kajit/src/backends/aarch64/` (peephole or isel patterns).

**What**: After the CFG is clean and shift amounts are constants,
pattern-match target-specific instructions:

- **`bfi` (bitfield insert)**: `(acc & ~mask) | ((val & 0x7f) << const_shift)`
  becomes `bfi acc, val, shift, 7`. This is `BFM` on aarch64.

- **`tbz`/`tbnz` (test and branch)**: `and x, reg, #(1<<k)` + `cbz/cbnz`
  becomes `tbz/tbnz reg, k`. Already partially handled by `fuse_cmpz`
  but needs extension for the AND+branch pattern.

- **Redundant mov elimination**: Post-regalloc, track register contents
  and eliminate `mov xN, xN` and cases where the source register could
  be used directly.

- **Leaf function detection**: If the unrolled code has no calls (which
  it shouldn't -- varint decode is pure), skip the stack frame entirely.
  Use only caller-saved registers. This saves 8 prologue/epilogue
  instructions.

**Expected result**: `mul + lsl + orr` per byte becomes `bfi`. Branch
sequences become `tbz`/`tbnz`. Prologue/epilogue disappear for leaf
functions.

## Target Assembly

After the full pipeline, scalar u32 varint decode should look approximately:

```asm
; no prologue (leaf, caller-saved only)
ldrb w9, [x0]               ; byte 0
and w1, w9, #0x7f           ; bits[0:7]
tbz w9, #0x7, .done          ; 1-byte -> store
ldrb w9, [x0, #1]           ; byte 1
bfi w1, w9, #7, #7          ; bits[7:14]
tbz w9, #0x7, .done2         ; 2-byte -> store
ldrb w9, [x0, #2]           ; byte 2
bfi w1, w9, #14, #7         ; bits[14:21]
tbz w9, #0x7, .done3         ; 3-byte -> store
ldrb w9, [x0, #3]           ; byte 3
bfi w1, w9, #21, #7         ; bits[21:28]
tbnz w9, #0x7, .overflow     ; 4-byte overflow
.done4:
  add x0, x0, #4            ; advance cursor by 4
  str w1, [x21]              ; store result
  b .epilogue
.done3:
  add x0, x0, #3
  str w1, [x21]
  b .epilogue
; ...
```

~20 instructions, no loop, no stack frame. Competitive with serde.

## Implementation Priority

```
Phase 1 (iteration specialization)     ████████████ highest leverage
Phase 2 (gamma simplification)         ████████████ enables everything
Phase 3 (CFG SimplifyCFG-lite)         ████████     mops up lowering artifacts
Phase 5 (backend combines)             ██████       bfi/tbz polish
Phase 4 (jump threading)               ████         diminishing returns
```

Phase 1 + 2 are tightly coupled and should ship together. Without
specialization, the gammas don't simplify. Without gamma simplification,
specialization doesn't produce better code.

Phase 3 is needed because even after RVSDG simplification, the
linearizer creates some CFG artifacts (empty merge blocks for remaining
gammas). This is a targeted cleanup, not a general-purpose pass.

Phase 5 (backend combines) is independent and can be done in parallel
with any of the above.

Phase 4 (jump threading) is only needed for patterns that survive
phases 1-3. It may not be needed at all if the RVSDG simplification
is aggressive enough.

## Validation

Every phase must:
1. Pass 99/99 corpus tests (with unrolling enabled)
2. Pass differential harness (interpreter vs JIT)
3. Show measurable improvement on `cargo bench -p kajit --bench synthetic`
4. Not regress other benchmarks (vec, string, nested struct cases)

The lockstep debugger (with the new LocationMap) can verify any
divergence at the instruction level.

## Success Criteria

| Metric | Current | After unroll+specialize | Target |
|--------|---------|------------------------|--------|
| scalar_u32 ns/op | 3.2 | ? | < 1.5 |
| Instructions | 83 | ? | < 30 |
| Blocks | 16 | ? | < 10 |
| Forwarding blocks | 4 | ? | 0 |
| Stack frame | yes | ? | no (leaf) |
| `mul` in hot path | yes | no | no |
| `bfi` used | no | no | yes |

## References

- LLVM LoopUnroll + InstSimplify pipeline: unroll then simplify to fixpoint
- MLIR canonicalize: "transform, then canonicalize" pattern for region-based IR
- LLVM SimplifyCFG: peephole CFG cleanup after structural transforms
- LLVM JumpThreading: forward edges through blocks when predecessor constrains successor
- Cranelift e-graph RFC: target-independent rewrites before machine lowering
- Briggs/Cooper/Simpson: "Practical Improvements to the Construction and
  Destruction of Static Single Assignment Form" (parallel copy resolution)
