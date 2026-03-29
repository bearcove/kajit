# Loop-carried state canonicalization

## Problem

`slot2reg` promotes stack slots to theta ports unconditionally. When a slot is
a loop-local temporary (e.g. varint accumulator, shift counter, byte buffer),
it becomes a loop-carried theta port even though its previous-iteration value
is never observable on the continue path — it is reinited to a constant at the
start of every iteration.

This creates unnecessary loop-carried state that:
- inflates the theta's port count (vec_nested_struct: 28 ports, ~7 semantically needed)
- forces regalloc to preserve values across the loop backedge
- generates spill/reload storms and backedge shuffle sequences
- increases frame size

## Why post-hoc removal failed

The `dead_theta_ports` pass attempted to remove these ports after promotion.
It used slot-level metadata (`theta_reinit_slots`) from the pre-promotion scan
to identify candidates. This metadata says "the slot has WriteToSlot(Const(0))
before any ReadFromSlot."

The problem: after slot2reg promotion, the body arg carries the previous
iteration's value and is used by gamma pass-throughs *before* the reinit
constant is produced. The slot-level analysis doesn't capture this port-level
data flow. The body result (backedge) carries the inner decode's final value
(non-zero), not the reinit constant.

Replacing the body arg with Const(0) is only safe if the backedge value is
provably always equal to the input constant. For varint temps, it isn't —
the inner decode produces non-zero byte values that flow back via the
backedge. These values are unobservable (the output has no consumers and the
next iteration reinits), but they *do* flow through gamma pass-throughs that
affect the body result.

## Proposed approach: promote differently

Instead of promoting all slots to theta ports and then trying to remove
the unnecessary ones, `promote_theta` in `slot2reg.rs` should detect
reinited loop-local temps at promotion time and handle them differently:

**For a slot that is reinited to Const(C) at the start of every iteration:**
1. Do NOT create a theta port (no input, no body arg, no body result, no output)
2. Emit Const(C) directly inside the theta body as the initial value
3. The inner computation uses this constant as its starting point
4. The body result for the inner computation feeds the correct gamma
   structure for break/continue paths

**Key invariant:** A value whose previous-iteration result is unobservable on
the continue path should not become a real loop-carried port.

**Break/exit path correctness:** On the break path, gamma pass-throughs
currently carry the body arg (previous iteration's value). If we don't create
a body arg, the break path needs the reinit constant instead. Since the slot
is reinited before being read, the break path value is always the reinit
constant or a computed value — both of which are available inside the body
without a theta port.

## Feasibility questions

1. Can `promote_theta` reliably identify reinited slots before promotion?
   The `find_reinit_slots` scan already exists. But it needs to be strengthened
   to verify that the reinit covers ALL paths (not just the dominant path).

2. Can the gamma structure for break/continue be built correctly without a
   body arg? The break-path pass-through currently references the body arg.
   Without it, the pass-through must reference the reinit constant.

3. Does this interact with nested thetas? Inner thetas have their own
   promotion. The outer theta only needs to handle the slots that belong to
   its own level.

## Spike result (2026-03-29)

**Attempted:** Skip theta port creation for reinited slots in `promote_theta`
Phase 1. Emit Const(0) inside the body instead. Skip body result and output.

**Result:** Same pair of slots (24+25, varint_shift + varint_byte) that failed
in dead_theta_ports ALSO fails here. Each slot works individually. The pair
fails together. The mechanism is identical:

1. Both slots feed the same gamma (n260) as inputs 12 and 13
2. When BOTH are Const(0) (no body arg), the gamma receives two constants
3. After `simplify_trivial_gammas` folds constant-predicate gammas, the code
   structure changes
4. The changed structure exposes incorrect values flowing through the gamma
   pass-through path

**Root cause:** `find_reinit_slots` guarantees "first access is
WriteToSlot(Const(0))" but doesn't account for gamma pass-through data flow
that occurs at the same topological level through a different branch path.

The gamma's branch 0 (break path) passes through the current slot value.
If no WriteToSlot has executed yet at the gamma level (the write is INSIDE
the gamma's branch 1), the pass-through carries the initial value. With a
body arg, this is the previous iteration's computed value. With Const(0),
this is always 0. These differ when the inner decode produces non-zero values.

**Implication:** The correct fix requires understanding gamma-level data flow,
not just slot-level access ordering. The reinit analysis must verify that the
slot's value is dead on ALL gamma branches at the point where the gamma
occurs — not just that the first sequential access is a write.

## What would make this work

The eligibility analysis needs to be: for each candidate slot S and each gamma
G in the theta body that uses S as a pass-through input:
- The pass-through must be on the break/exit path only
- On the continue path (branch where inner computation runs), S must be
  written before being read
- The exit-path pass-through value must not flow to any live consumer
  (the theta output for this port must have no consumers)

This is essentially the same analysis as dead_theta_ports tried to do, but
at a different point in the pipeline. Moving it to slot2reg doesn't avoid the
analysis complexity — it just moves where the analysis runs.

## Next direction

The fundamental issue is that the RVSDG gamma/theta structure makes
"is this value observable?" analysis non-trivial. The value flows through
structural pass-throughs that exist for correctness (break-path value
preservation) even when the value is semantically dead.

Options:
1. **Strengthen the eligibility analysis** — make it gamma-aware, proving that
   pass-through values are dead on all exit paths. This is complex but correct.
2. **Redundant phi elimination in CFG-MIR** — after linearization, identify
   phi args that carry the same value as their initial definition and remove
   them. This operates on the CFG where data flow is explicit.
3. **Live range splitting in regalloc** — let regalloc handle the pressure by
   splitting live ranges of loop-carried values that are only live across the
   backedge shuffle, not across the loop body.

Option 2 is probably the most practical: it operates on explicit control flow
(not structural RVSDG regions), the analysis is standard (phi elimination),
and it applies to any program automatically.
