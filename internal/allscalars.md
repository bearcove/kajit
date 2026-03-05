# json::all_scalars (aarch64) Failure Notes

## Symptom

Native `aarch64-apple-darwin`:

- `kajit::corpus json::all_scalars` fails with `DeserError { code: MissingRequiredField, offset: 0 }`
- Failure persists under `KAJIT_OPTS='-all_opts'` (so not an RVSDG optimization pass issue)
- Failure is intermittent under some dump/ENV combinations, but reliably reproducible in the failing configuration.

## Minimal Repro / Validation Commands

- Repro:
  - `cargo nextest run -p kajit --test corpus -E 'test(=json::all_scalars)'`
- Key bisect:
  - `KAJIT_OPTS='-regalloc' cargo nextest run -p kajit --test corpus -E 'test(=json::all_scalars)'` passes
    - Indicates the regression is in regalloc-on backend lowering/emission (edit application / ABI / clobber), not optimization passes.

## What We Know From Stage Dumps

From `kajit/target/kajit-stage-dumps/json__all_scalars__aarch64__linear.txt`:

- Required-field mask:
  - `v211 = const 131071` (`0x1ffff`)
- Presence check at function end:
  - `v212 = v210 And v211`
  - `v213 = v212 CmpNe v211`
  - `error_exit MissingRequiredField`
- `v210` is the presence-bitset accumulator updated via `Or` with constants such as `32768 (0x8000)` and `65536 (0x10000)` earlier in the decode.

This points at “presence bit(s) not being set” on aarch64 under regalloc-on, but the exact first wrong instruction/op is still unknown.

## Captured Edit Duplication Pattern (Context)

Observed in dumps (both platforms/cases) before the shared mover refactor:

- Same move tuples appear both as progpoint edits and as edge edits near key control-flow joins.
- Example noted (aarch64 `json::all_scalars`):
  - progpoint `373-pre`: `{stack70->p12, p23->stack64, p26->stack68, p25->stack69, p24->stack71}`
  - edge edits repeat same tuple set at `edge lin=405 succ=0`
- Similar duplication exists for x86_64 varint loop cases.

This duplication is a symptom to investigate (mapping/ownership of edits), but a naive “dedup” (skipping progpoint moves if seen in edge set) was tested and rejected.

## Changes / Experiments So Far

### Shared Parallel-Move Execution

- Added shared parallel-move execution:
  - `kajit/src/backends/parallel_moves.rs`
  - `MoveEmitter` + `emit_parallel_moves(...)`
- Wired both backends to use it for:
  - progpoint edits
  - fallthrough edge edits
  - edge trampolines
- Added unit tests covering:
  - self-copy
  - chain
  - reg<->reg 2-cycle
  - reg<->stack cycle
  - stack->stack via temp

This refactor did not by itself fix `json::all_scalars` (aarch64) and did not fix the x86_64 InvalidVarint cluster, but it reduced backend divergence in move execution semantics.

### ORR-Immediate Hypothesis

- Hypothesis: aarch64 ORR-immediate encoding limitations could cause incorrect `Or` with constants like `0x8000`/`0x10000`.
- Mate reports experiments did not pan out (no concrete evidence yet that OR/ORR immediate encoding is the issue).

### Call-Arg Clobber Experiment

- Tried a fix in `kajit/src/backends/aarch64/calls.rs` to avoid ABI arg clobber during sequential register setup.
- `json::all_scalars` still fails after this change.

### Reverted x86_64 “dedup” Guards

- Removed x86_64 edit-map filtering that skipped progpoint moves when the same `(from,to)` existed in an edge move set:
  - `edge_move_set.contains(...)` guards in `kajit/src/backends/x86_64/mod.rs`
- Reason: it was an unproven semantic change and caused failure-mode shifts without actually fixing the failures.

## Current Working Theory (Unproven)

Because `KAJIT_OPTS='-regalloc'` makes the test pass, likely culprits are:

- Incorrect mapping of regalloc `Edit::Move` / `edge_edits` to emission sites (progpoint vs edge ownership), or
- A clobber/ABI modeling hole where a live value (presence bitset) is overwritten across `call_intrinsic` or during edit application, or
- A bug in how allocations are applied to linear ops at emission time (allocation-to-linear-op mapping drift).

## Next Investigation Steps

1. Pinpoint the first divergence:
   - Use stage dumps + targeted disassembly/trace to identify the first time the presence bitset (`v210`/its allocated location) should be updated but is not.
2. Verify edit application around the key-read/key-compare loop:
   - Compare emitted code/edits at the relevant progpoints and edges.
3. Verify clobber modeling for `call_intrinsic`:
   - Ensure values live across calls are either in preserved locations or correctly saved/restored.
4. Validate on both:
   - `cargo nextest run --no-fail-fast` (native)
   - `cargo nextest run --target x86_64-apple-darwin --no-fail-fast`

## Tooling Notes

- Mate reported `mate update b905d857` failed with “no request found for 3/b905d857” due to a mate server/session mismatch.

