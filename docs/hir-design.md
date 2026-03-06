# kajit HIR design

Status: planned, not yet implemented.

This document defines the role of a high-level intermediate representation
(HIR) above RVSDG. It is the design reference for issue #208 and the
follow-up implementation issues in the HIR epic.

## Why HIR exists

Kajit now has two plausible frontend domains:

- shape-driven serialization/deserialization
- a rule language for a build system

RVSDG remains a good shared optimization boundary, but it is too low-level to
be the primary human-facing program model once multiple frontends exist.

HIR is the shared semantic boundary above RVSDG.

## Goals

- Provide one frontend-neutral semantic program model.
- Preserve human-meaningful names, scopes, comments, and spans.
- Be readable enough to serve as the default debugger/source view.
- Lower cleanly into RVSDG without backend-specific workarounds.
- Root semantic debug provenance before RVSDG lowering.

## Non-goals

- HIR is not the optimization IR.
- HIR is not a graph IR.
- HIR is not a machine-oriented SSA form.
- HIR is not required to model backend calling conventions or register classes.
- HIR should not encode RVSDG gamma/theta/apply structure directly.

## Pipeline role

Current pipeline:

```text
Shape tree / frontend lowering -> RVSDG -> LIR -> CFG-MIR -> machine code
```

Target pipeline:

```text
frontend language / generated frontend
  -> HIR
  -> RVSDG
  -> LIR
  -> CFG-MIR
  -> machine code
```

The intent is:

- HIR is the semantic/source-facing layer.
- RVSDG is the optimization layer.
- CFG-MIR is the backend/debug-codegen layer.

## Core design decisions

### HIR should be typed

HIR should be typed enough to lower deterministically and support readable
debugging, but not machine-layout typed.

That means HIR types should describe source-semantic values such as:

- booleans
- integers
- strings / byte slices
- structs / tuples
- maps / sequences
- rule-language values

HIR should not require backend-facing details such as:

- register classes
- concrete stack layout
- ABI-specific calling convention details
- CFG block parameters

### HIR should keep structured control flow

HIR should preserve source-shaped control flow:

- `if`
- `loop`
- `break`
- `continue`
- `match` or structured multi-branch dispatch

This is the right place to represent human-meaningful control structure. RVSDG
lowering is where those structures become gamma/theta/apply.

### HIR should use explicit operations, not state tokens

HIR still needs an effect model, but it should be semantic and statement-based,
not RVSDG-style token threading.

Examples of HIR-visible effectful operations:

- read input
- advance cursor
- write field
- call intrinsic/runtime helper
- emit build action
- report error

Ordering should come from statement sequence and structured regions, not from
explicit dataflow state edges. Lowering to RVSDG is responsible for making
effect ordering explicit.

### Debug provenance belongs in HIR first

Semantic debug identity should be rooted in HIR, not invented late from CFG-MIR.

HIR owns:

- logical variable names
- lexical scope hierarchy
- source spans
- comments/doc strings intended for humans
- stable statement IDs and local IDs

RVSDG, LIR, CFG-MIR, regalloc, and codegen should preserve or refine that
provenance, not create the first user-facing meaning.

## Ownership boundaries

### HIR owns

- named locals
- lexical scopes
- source spans
- comments/doc strings
- semantic statement boundaries
- source-shaped control flow
- semantic calls/intrinsics
- debugger-facing variable identity

### RVSDG owns

- graph/dataflow normalization
- explicit effect ordering via state edges
- gamma/theta/apply structure
- optimization-friendly value dependencies
- transformations that are awkward in structured AST/HIR form

### CFG-MIR owns

- linearized post-RVSDG control flow
- backend-facing operand structure
- block/edge/terminator layout
- regalloc and emission friendliness
- exact low-level stepping for backend/debug-codegen work

## Debug/source model

Long-term, the default human-facing source view for JIT debugging should be
HIR, not CFG-MIR.

Rationale:

- HIR is the best place to show semantic variable names and structured flow.
- CFG-MIR is still valuable, but primarily for codegen/regalloc debugging.
- RVSDG may also be a useful optional source view, but it should not be the
  default for most users.

### Source views

The debugger/source pipeline should support multiple source views over the same
machine code:

- `hir`: default human-facing source
- `rvsdg`: optional optimization/debugging source
- `cfg-mir`: expert/backend source

Only one primary source location will usually be active for a given PC, so the
compiler must choose one primary view per debug build. Auxiliary views can be
emitted as side listings or alternate artifacts.

### Default choice

When HIR exists and is usable, the default should be:

```text
primary source view = HIR
expert fallback = CFG-MIR
optimization fallback = RVSDG
```

Before then, CFG-MIR remains the current primary source view.

## Debug identity model

HIR should assign stable IDs to:

- statements
- locals
- scopes

Expressions do not need their own stable IDs by default if they are fully
contained inside a statement or local binding. If a frontend has expression
spans that matter independently, those can lower to statement-local subranges,
but the base design should stay simple:

- statement IDs drive source mapping
- local IDs drive debugger variables
- scope IDs drive lexical visibility

This avoids over-modeling every expression as a debugger entity.

## Comments and human-readable intent

HIR is the right place for comments or doc strings that explain intent.

Examples:

- deserializer-generated comments such as "match key against field `a`"
- rule-frontend comments such as "emit compile action for each src"

These comments may later be projected into:

- HIR listings
- debugger-facing source files
- CFG-MIR comments when low-level stepping is still desired

CFG-MIR may still carry mirrored comments, but HIR is the source of truth for
that intent.

## Frontend relationship

HIR should be frontend-neutral.

That means:

- the deserializer frontend lowers into HIR
- the rule-language frontend lowers into the same HIR

The frontends do not need to share a single concrete surface language. They do
need to share the same semantic lowering target.

This keeps:

- parser/front-end concerns separate
- optimizer/backend reuse high
- debugger/source tooling unified below the frontend boundary

## Sketch of the intended HIR shape

HIR should print like a boring, readable program, not like SSA or a graph.

Example shape:

```text
fn decode_bools(ctx, input, out) {
  let mut seen_mask = 0

  loop {
    skip_ws(ctx, input)
    if peek_byte(input) == '}' {
      break
    }

    let key_ptr, key_len = read_key(ctx, input)
    expect_colon(ctx, input)

    let mut handled_field = false

    let is_field_a = key_equals(key_ptr, key_len, "a")
    if is_field_a && !handled_field {
      let a = read_bool(ctx, input)
      write_field(out, .a, a)
      seen_mask = seen_mask | 0x1
      handled_field = true
    }
  }
}
```

This is the level of readability HIR is meant to preserve.

## Open questions

These are intentionally left open for follow-up implementation work:

- whether HIR needs a dedicated type system crate or can reuse existing shape
  metadata directly for the first prototype
- whether `match` should be first-class in HIR or lowered into nested `if`
  before RVSDG
- how much expression nesting should be preserved before introducing explicit
  temporaries during lowering
- whether source-view selection should be a compile-time option, runtime debug
  option, or both

## Summary

HIR is the planned semantic layer above RVSDG.

The design decisions in this document are:

- HIR is typed, but not machine-layout typed.
- HIR keeps structured control flow.
- HIR uses statement sequencing for effects instead of state tokens.
- HIR owns semantic names, scopes, spans, comments, and debug identities.
- RVSDG remains the optimization boundary.
- CFG-MIR remains the backend/debug-codegen boundary.
- HIR should become the default debugger source view once implemented.
