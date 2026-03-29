# Next project: CFG-MIR loop-carried redundancy elimination

## Status: Not started

## Context

The RVSDG/slot2reg seam for dead theta port elimination is closed.
Both post-hoc removal (dead_theta_ports) and promotion-time avoidance
(slot2reg Phase 1 reinit) hit the same fundamental problem: gamma
pass-through observability analysis is non-trivial in the RVSDG.

The next approach operates at the CFG-MIR level, where control flow
is explicit and standard dataflow analysis applies.

## Target

vec_nested_struct outer loop only, as the initial case.

## Goal

Identify loop-header block params (phi values) and their corresponding
backedge values that are loop-carried but redundant: the backedge value
is overwritten before its first meaningful use on the looping path.

Eliminate those carried values using explicit CFG dataflow — not RVSDG
gamma/theta reasoning.

## Approach

1. Get a CFG-MIR roundtrip dump of vec_nested_struct
2. Identify the outer loop header block and its params
3. For each param: trace the backedge value and determine if it is
   always overwritten (re-defined) before any use on the looping path
4. For params where this holds: replace uses with the re-definition,
   remove the param from the loop header, remove the backedge phi arg
5. Validate with interpreter/JIT agreement (differential harness or
   exec stage)

## Legality condition (in CFG terms)

A loop-header param P with backedge value V is redundant if:
- On every path from the loop header to any use of P (within the loop),
  P is redefined before use
- Equivalently: P is dead at the loop header on the looping path —
  its live range starts at the redefinition, not at the header

This is standard liveness analysis on the CFG.

## Deliverables

1. The exact CFG-MIR pattern (loop header param + backedge + redefinition)
2. The legality condition stated in CFG terms
3. A narrow implementation on vec_nested_struct outer loop
4. Before/after metrics (port count, asm size, frame size, benchmark)
5. Whether the rule clearly generalizes beyond this benchmark

## Constraints

- Frame in terms of generic CFG patterns, not specific slot/port numbers
- Use CFG roundtrip dumps and interpreter/JIT agreement to validate
- Keep the implementation narrow until the pattern is proven
