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

## Execution model

HIR should not bake deserializer-specific ambient concepts such as "input
operations" or a privileged cursor abstraction into the language itself.

Programs should operate on explicit state passed through normal parameters and
locals.

Examples:

- a deserializer function may take a context struct containing `input_ptr`,
  `input_end`, output pointers, and error state
- a rule-evaluation function may take an evaluation context, environment, and
  host handles

The language stays generic. Different frontends choose different state
structures.

That means the shared semantic layer should look more like:

- field access
- local bindings
- calls
- structured control flow
- loads/stores and pointer-aware operations where needed

and less like:

- `peek_byte`
- `advance_cursor`
- `read_key`

Those may still exist as frontend library operations or sugar, but they should
lower to the same generic core semantics instead of being built into the shared
language model.

## Core design decisions

### HIR should be typed

HIR should be typed enough to lower deterministically and support readable
debugging, but not machine-layout typed.

That means HIR types should describe source-semantic values such as:

- booleans
- integers
- strings / byte slices
- structs / tuples
- Rust-shaped enums with payload-bearing variants
- maps / sequences
- rule-language values

HIR should not require backend-facing details such as:

- register classes
- concrete stack layout
- ABI-specific calling convention details
- CFG block parameters

### HIR should include first-class structs and enums

HIR should treat structs and Rust-style enums as ordinary source-semantic
types, not as something that needs to be reconstructed later from lower-level
primitives.

That means HIR should support:

- named-field structs
- tuple-like product types where useful
- tagged enums with named variants
- payload-bearing variants
- pattern matching / variant tests at the structured-control-flow level

This is especially important because Rust-shaped enums are too useful to force
through a less expressive encoding. They are directly valuable for:

- parser state machines
- `Option` / `Result`-like flows
- rule-language values
- domain-specific planner data

### Struct and enum layout should be semantic by default

HIR should define source-semantic shape, not machine layout.

For structs, the default rules should be:

- field names and declaration order are preserved semantically
- field access uses names, not byte offsets
- layout is opaque unless a boundary explicitly requires a layout contract

For enums, the default rules should be:

- variants are named and tagged semantically
- payloads are variant-local fields
- discriminant size and payload packing are not fixed at HIR level

This keeps HIR readable and frontend-neutral. It also avoids prematurely
locking the shared language to one backend ABI or one storage convention.

If exact layout matters, that should be explicit and exceptional.

Examples where explicit layout may matter:

- Rust interop boundaries
- serialized in-memory planner data that must match a host layout
- low-level runtime structs shared with handwritten code

Those cases should use explicit representation/layout annotations, or lower to
a later IR layer where layout is a first-class concern. They should not define
the default meaning of structs and enums in HIR.

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

- load/store through explicit state or memory
- update fields on explicit context structs
- write output fields
- call intrinsic/runtime helper
- emit build action
- report error

Ordering should come from statement sequence and structured regions, not from
explicit dataflow state edges. Lowering to RVSDG is responsible for making
effect ordering explicit.

### HIR should be memory-safe by default

HIR should default to safe semantics that are stronger than C and less
annotation-heavy than Rust.

The current design direction is:

- no garbage collector in the core model
- no raw pointers as an ordinary source-language concept
- no explicit lifetime annotations in normal source code
- ownership and borrowing rules may exist semantically, but should be inferred
  or otherwise kept compiler-internal where possible

This means the user-facing language should primarily expose:

- plain value types for scalars, structs, and enums
- owned containers for growable or heap-backed data
- borrowed views such as slices and strings
- explicit state structs passed through normal locals and parameters
- typed handles or arena-backed references when shared long-lived objects are
  needed

### Raw pointers should be a lowering concern, not a default source feature

The shared language should not force frontends to reason in terms of
`ptr + len`, nullability, arbitrary pointer arithmetic, or integer-to-pointer
casts.

For example:

- a deserializer frontend can model input as a `Cursor { bytes, pos }` or other
  explicit state struct
- a planner frontend can model shared objects through typed handles or owned
  values

If lower layers want to optimize those shapes into raw pointers and reserved
registers, that is a job for lowering, regalloc, and backend/runtime contracts.

### Practical safety tools

Without a GC and without explicit lifetime syntax, the most promising toolbox
looks like:

- ownership for heap-backed values
- compiler-inferred borrowing for short-lived views
- typed slices/strings instead of naked address ranges
- typed handles for graph-like or planner-owned shared objects
- arena or region allocation for coarse-grained shared lifetimes where that is
  the right tradeoff

The important design point is that these are generic language/runtime tools,
not deserializer-specific features.

### Unsafe escape hatches should be explicit

There will likely still be a need for explicit low-level escape hatches for:

- runtime interop
- layout-sensitive host boundaries
- especially performance-sensitive kernels

But those should be opt-in and visibly unsafe. They should not define the
baseline semantics of ordinary HIR programs.

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

### HIR should not define register residency as domain semantics

The user-visible language should not special-case parser state just to get good
register allocation.

If we want values such as `input_ptr`, `input_end`, or hot loop locals to stay
resident in registers, that is a storage/placement concern, not a reason to add
deserializer-specific primitives to the language.

The design implication is:

- HIR owns semantic state and locals
- a later lowering layer may attach residency hints or hard placement
  constraints
- those hints must remain generic enough to be useful for non-deserializer
  programs too

Examples of generic placement concepts that may exist later:

- local may be freely placed
- local is address-taken and must have stable storage
- local is hot and should prefer register residency
- function parameter is bound to a reserved runtime register set

The exact mechanism does not need to be in HIR yet, but the key decision is
that "keep this live in a register" is not the same thing as "this language has
input operations."

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

If we choose to lower closer to a generic systems language core, the same logic
may also be expressible in a more explicit style like:

```text
let key_ptr = ctx.key_ptr
let key_len = ctx.key_len
let is_field_a = key_equals(key_ptr, key_len, "a")
if is_field_a && !handled_field {
  let a = parse_bool(ctx)
  out.a = a
  seen_mask = seen_mask | 0x1
  handled_field = true
}
```

The important property is still the same: the program stays source-readable and
state is explicit, rather than encoded as a set of privileged parser-only
primitives.

## Open questions

These are intentionally left open for follow-up implementation work:

- whether HIR needs a dedicated type system crate or can reuse existing shape
  metadata directly for the first prototype
- what the exact safe memory model should be for inferred borrowing, handles,
  and arena-backed values
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
- HIR is memory-safe by default and does not expose raw pointers as a normal
  source feature.
- HIR owns semantic names, scopes, spans, comments, and debug identities.
- RVSDG remains the optimization boundary.
- CFG-MIR remains the backend/debug-codegen boundary.
- HIR should become the default debugger source view once implemented.
