# kajit HIR design

Status: planned, not yet implemented.

This document defines the role of a high-level intermediate representation
(HIR) above RVSDG. It is the design reference for issue #208 and the
follow-up implementation issues in the HIR epic.

The corrective boundary note for generated deserializers lives in
[hir-generated-decoder-boundary.md](./hir-generated-decoder-boundary.md).
That note freezes what planner concepts stay above HIR and what low-level
responsibilities belong inside HIR programs.

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

## RVSDG changes needed

The current RVSDG op vocabulary is deserialization-specific. It is built around
cursor movement, field writes, slots, and format/runtime intrinsics.

That is acceptable for the current frontend, but it is not a sufficient shared
optimization boundary for a second frontend such as a rule/build language.

The design direction is:

- the structural part of RVSDG remains shared: regions, state edges, gamma,
  theta, apply, and the existing structural optimization passes
- the primary generalization is in state domains, not in replacing every op
- domain-specific ops may still exist, but only within a broader shared model

The current gap is narrower than "rewrite RVSDG from scratch." Arithmetic,
constants, calls, slots, gamma/theta structure, and most structural passes are
already generic enough to survive into a shared middle layer.

The main thing that is too specialized today is the hardcoded state-domain
model:

- cursor state is baked in as a distinguished domain
- output state is baked in as a distinguished domain
- cursor-specific ops and SIMD helpers are treated as universal rather than as
  one domain's operations

The intended direction for #209 is:

- replace hardcoded cursor/output state with named state domains
- make `cursor` and `output` simply the first two state-domain instances
- keep a shared base vocabulary for calls, arithmetic, slots, control-relevant
  ops, and layout-resolved field access
- let truly domain-specific operations survive as dialect/domain ops where that
  is useful

For #209, the recommended structural representation is:

- `PortKind::State(StateDomainId)` instead of separate `StateCursor` and
  `StateOutput` variants
- `StateDomainId` is an index or interned ID, with `cursor` and `output` as the
  first concrete instances
- the coarse RVSDG effect model generalizes in the same direction, for example
  from `Effect::Cursor` / `Effect::Output` to `Effect::Domain(StateDomainId)`

That keeps the structural optimizer shared while avoiding a false choice between
"one tiny common denominator IR" and "two unrelated IRs."

Shared optimization passes should continue to operate primarily on structural
properties and coarse effect information. Domain-specific ops that are opaque to
an optimization may still pass through as barriers.

This means "HIR lowers to RVSDG" is not just a new frontend layer. It also
implies that RVSDG itself must stop being purely deserializer-shaped.

## Execution model

The generated-decoder boundary matters here:

- planner concepts such as `flatten`, `untagged`, candidate-set narrowing,
  evidence accumulation, and ambiguity resolution stay above HIR
- generated HIR executes the chosen algorithm using ordinary low-level control
  flow, state, and data structures

HIR is therefore the language used to implement generated solver plans, not the
place where those planning concepts become first-class language features.

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

HIR types also need a real identity of their own. Once borrowed values exist,
type identity and layout identity diverge:

- `Foo<r_input>` and `Foo<r_tmp>` may have the same concrete layout
- but they do not have the same borrow or escape semantics

So a bare host `Shape` reference is not sufficient as the whole HIR type
identity. HIR needs a semantic type layer that can represent region/provenance
parameters directly. Layout metadata may still be attached separately for
HIR->RVSDG lowering.

That does not imply every host-side runtime value must become a first-class
HIR value. Some host values are better treated as destination-materialization
problems:

- HIR computes raw ingredients such as addresses, lengths, capacities, and
  validated byte ranges
- host layout/schema identifies the destination subtree
- lowering/runtime calls materialize the final host value directly into that
  destination

Rust `String` is the clearest example. Generated decoder HIR does not need a
semantic `String` value just because the host result contains one.

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

Layout resolution — mapping field names to concrete offsets and widths — happens
at the HIR→RVSDG boundary. HIR stays semantic; RVSDG continues to use concrete
layout where optimization and effect independence need it.

For the first implementation, HIR should assume a real semantic type layer plus
a separate layout source. That layout source may still reuse facet `Shape`
metadata where appropriate, but the contract is:

- HIR names fields and variants semantically
- HIR type identity is not just host `Shape` identity
- HIR→RVSDG lowering resolves concrete layout when required
- RVSDG/LIR/CFG-MIR remain free to reason in terms of offsets and widths

### HIR should keep structured control flow

HIR should preserve source-shaped control flow:

- `if`
- `loop`
- `break`
- `continue`
- `match` as a first-class structured multi-branch dispatch

This is the right place to represent human-meaningful control structure. RVSDG
lowering is where those structures become gamma/theta/apply.

`match` should be first-class in HIR, not desugared to nested `if` before
RVSDG lowering. A first-class `match` lowers naturally to a flat gamma rather
than a nest of smaller gammas, and it preserves variant exclusivity
information that is useful for optimization and debugging.

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

### HIR operations should carry a coarse effect classification

HIR does not need RVSDG's full state-token model, but it does need enough
effect information for useful lowering.

The minimum useful classification is:

- `pure`: no side effects
- `reads`: reads external state but does not mutate it
- `mutates`: updates a known state domain
- `barrier`: may touch arbitrary state

This classification may come from the operation itself, from callee metadata,
from argument types, or from a combination of those sources. The important
constraint is that HIR→RVSDG lowering must not treat every statement as a full
barrier by default.

RVSDG remains the place where effects become explicit state edges and where
independence can be refined more aggressively.

Today, RVSDG's effect model is still specialized:

- `pure`
- `cursor`
- `output`
- `barrier`

HIR's effect classes are intentionally more general than that current enum.
The initial lowering contract is:

- HIR `pure` -> RVSDG `Pure`
- HIR `mutates` -> a concrete RVSDG state domain such as `cursor`, `output`, or
  a later generalized named domain
- HIR `barrier` -> RVSDG `Barrier`
- HIR `reads` is a refinement the current RVSDG does not model separately yet

That means initial HIR->RVSDG lowering may conservatively map `reads` to the
same state-domain chain as `mutates` in that domain. This loses optimization
opportunity, but not correctness. A future generalized RVSDG may distinguish
read-only and mutating access more precisely.

### Effect summaries must be domain-aware

The four effect classes are the user-facing lattice. They are not enough on
their own to drive lowering.

HIR operations and calls also need a domain-aware effect summary that answers:

- which state domains are read
- which state domains are mutated
- whether the operation is a barrier across all domains

That summary may live on builtin/intrinsic signatures, typed call metadata, or
operation definitions. The important contract is that HIR->RVSDG lowering must
not guess.

Examples:

- `peek_byte(cursor)`:
  - effect class: `reads`
  - domains: reads `cursor`
- `advance_cursor(cursor, n)`:
  - effect class: `mutates`
  - domains: mutates `cursor`
- `init out.name = value`:
  - effect class: `mutates`
  - domains: mutates `output`
- `emit.node(...)`:
  - effect class: `mutates`
  - domains: mutates `ruleplan`
- opaque runtime helper:
  - effect class: `barrier`
  - domains: all / unknown

### HIR should be memory-safe by default

HIR should default to safe semantics that are stronger than C.

The goal is not to invent a post-Rust universal trick. The design should pick a
clear, ownership-oriented point on the tradeoff frontier and say so honestly.

The current design direction is:

- no garbage collector in the core model
- no raw pointers as an ordinary source-language concept
- explicit address values are allowed when the language needs to talk about
  allocated memory directly, but those addresses should remain typed by
  allocation domain rather than degenerating into untyped pointers
- no explicit lifetime annotations in normal source code
- ownership and borrowing are real semantic constraints, not optional compiler
  magic
- surface languages may elide lifetime syntax where inference is sufficient, but
  HIR itself must still model provenance/region relationships explicitly

This means the user-facing language should primarily expose:

- plain value types for scalars, structs, and enums
- owned containers for growable or heap-backed data
- borrowed views such as slices and strings
- borrowed structs/enums whose fields may carry input provenance
- explicit state structs passed through normal locals and parameters
- typed handles or arena-backed references when shared long-lived objects are
  needed

For generated deserializers, there is also a lower-level allocation concern
that should be explicit in HIR:

- some allocated memory is transient scratch/chunk storage that dies with the
  decode
- some allocated memory becomes part of the returned Rust value and must live
  on the persistent Rust heap

HIR should make that distinction explicit with typed address values or another
domain-aware memory abstraction. It should not collapse both cases into a
single undifferentiated "heap pointer".

### Borrowing must be first-class in HIR

Kajit's deserialization model requires more than owned values.

In particular:

- the input is borrowed from the caller
- some decoded outputs may borrow from that input
- zero-copy decoding must be representable without lowering the source language
  to raw pointers

So HIR must be able to express types and functions that are semantically like:

```text
fn decode<r_input>(input: Slice<r_input, Byte>) -> Foo<r_input>
```

where `Foo<r_input>` may contain fields such as:

- `Str<r_input>`
- `Slice<r_input, T>`
- nested structs/enums carrying the same borrowed provenance

This means the HIR memory model is not "owned values only." It is at least:

- owned values
- borrowed views tied to an explicit provenance/region
- unique mutable destinations or places where mutation is required
- store/arena handles for shared long-lived data

The surface language may hide most lifetime syntax in common cases. The HIR
cannot. By the time a frontend has lowered into HIR, borrow provenance must be
represented explicitly enough to drive type checking, lowering, and debugging.

### Borrow/provenance core

The minimum borrow/provenance contract for HIR should be:

- explicit region parameters on function signatures and ADT types
- explicit borrow-carrying types such as `Slice<r, T>` and `Str<r>`
- borrowed structs/enums parameterized by those regions
- explicit places/destinations for mutation and construction
- a small set of provenance-preserving operations such as:
  - field or variant projection
  - subslice/view creation
  - validated string view creation
- an escape rule: returned values may mention only regions named in the
  signature, never unnamed local or temporary regions

The HIR should store regions and places directly. More detailed borrow facts
may be derived by later checking/lowering passes; they do not need to become a
separate shared HIR database in v1.

For deserialization specifically, there is one important provenance rule:

- advancing a cursor mutates cursor state only
- borrowed results derive from the cursor's immutable
  `bytes: Slice<r_input, Byte>` root, not from the mutable cursor place itself

That keeps immutable input provenance separate from mutable decode state.

### Places and initialization are part of the core model

HIR already needs unique mutable destinations for deserialization and lowering.
That should be explicit.

A `Place` is a writable or readable storage location such as:

- a local
- a parameter/destination parameter
- a field projection
- an index projection where the type system allows it

The core rules should be:

- `init place = value` is distinct from `assign place = value`
- reads from an uninitialized or partially initialized place are invalid
- deserializer-shaped HIR may take explicit destination parameters
- lowering may map place writes to RVSDG output-state operations

The initial v1 destination contract should be:

- a deserializer-style destination parameter is uninitialized on entry unless
  the signature says otherwise
- on normal return, that destination must be fully initialized
- on failure or abnormal exit, a partially initialized destination must not be
  exposed to the caller as an initialized value
- control-flow joins must respect definite initialization before a place can be
  read or returned
- `assign` to an already-initialized owning place means replace the old value
  with defined drop/release behavior; it is not equivalent to `init`

That gives the first implementation a real contract to target:

- `init` is for first construction into uninitialized storage
- `assign` is for overwriting an existing initialized value
- success establishes full initialization
- failure must preserve safety even if construction was only partial

This is not just lowering detail. Kajit already has destination-oriented
semantics, partial initialization concerns, and caller-provided output storage.
HIR should name that directly.

One pressure point remains: `Place<T>` may be too neat as a universal model.
It works naturally for ordinary HIR storage typed by HIR semantic types. It is
less obviously correct for generated host-schema destinations. The boundary note
above freezes that as an active design pressure instead of silently assuming
that one generic `Place<T>` covers both cases.

One practical consequence is that generated decoder lowering should not force
every host-owned aggregate through a "compute typed value, then assign" path.
Direct destination materialization is part of the intended model.

### Raw pointers should be a lowering concern, not a default source feature

The shared language should not force frontends to reason in terms of
`ptr + len`, nullability, arbitrary pointer arithmetic, or integer-to-pointer
casts.

For example:

- a deserializer frontend can model input as `Cursor<r_input> { bytes:
  Slice<r_input, Byte>, pos: usize }` or another explicit borrowed state shape
- a planner frontend can model shared objects through typed handles or owned
  values

If lower layers want to optimize those shapes into raw pointers and reserved
registers, that is a job for lowering, regalloc, and backend/runtime contracts.

That still leaves room for HIR to model direct memory construction when it is
the clearest honest representation of the generated algorithm. In those cases,
the model should be:

- typed address values, not naked pointers
- allocation domains that distinguish transient scratch/chunk storage from
  persistent returned allocation
- explicit runtime finalization calls when a low-level memory assembly process
  becomes a Rust value such as `Vec<T>` or `String`

So HIR may need to talk about addresses. It should not talk about *unsafe*
addresses.

### Practical safety tools

Without a GC and without explicit lifetime syntax, the most promising toolbox
looks like:

- ownership for heap-backed values
- explicit borrow provenance in HIR, with inferred or elided lifetime syntax in
  surface code where possible
- typed slices/strings instead of naked address ranges
- borrowed structs/enums for zero-copy decode results
- typed handles for graph-like or planner-owned shared objects
- arena or region allocation for coarse-grained shared lifetimes where that is
  the right tradeoff

The important design point is that these are generic language/runtime tools,
not deserializer-specific features.

The split is intentional:

- borrowing/provenance handles caller-owned or input-tied data, which Kajit
  deserializers need
- handles/arenas handle shared long-lived data, which planner/build frontends
  are more likely to need

Those are different problems and should not be forced into one mechanism.

For the initial safe core, handles should also be region-closed:

- `Handle<store, T>` must not silently capture external input-borrow regions
- crossing that boundary requires an explicit future extension or an unsafe /
  interop mechanism

That keeps store-owned shared data from becoming a hidden escape hatch for
caller-tied borrows.

For tree-shaped and linear data, this model is relatively straightforward. For
graph-shaped data, it becomes much less trivial. The current design should be
read honestly:

- arena/handle style storage is the primary escape hatch for shared graph-like
  data
- that deliberately gives shared references arena granularity rather than
  per-reference precision
- this avoids requiring either a garbage collector or explicit lifetime syntax
- the cost is coarser-grained deallocation and a stricter ownership model for
  shared objects

The exact safety mechanism for handles is still open. Plausible options are:

- compile-time arena lifetime tracking
- runtime generation checks
- a combination of compile-time ownership with runtime validation at selected
  boundaries

For the first Vixen rule-language frontend specifically, the shared-data side of
this problem may be less immediate than it is in the general case. Vixen's planned value model is
primarily:

- owned typed values
- scoped variable bindings
- host callbacks that accumulate `RulePlan` output externally

That means the first rule-language frontend may not need first-class shared
arena references in user code right away. Even so, the general HIR design must
leave room for typed handles or arena-backed references when a future frontend
or host integration genuinely needs them.

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

### What changes below HIR

Introducing HIR should not require a fundamental rewrite of LIR, CFG-MIR, or
the backends. The expected changes are:

- RVSDG op vocabulary generalizes or gains a dialect mechanism
- debug provenance definitions move up so that HIR is the source of truth
- RVSDG, LIR, and CFG-MIR carry references to that provenance rather than
  inventing the first user-facing meaning themselves
- new frontends such as Vixen target HIR from the start
- existing direct-to-RVSDG frontends may continue lowering that way during the
  transition, then retarget through HIR once the shared layer is stable enough
  to justify the migration

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

The initial selection mechanism should be simple:

- a compile/debug option selects the primary source view for emitted DWARF
- non-primary views are still dumped as auxiliary files for inspection

That keeps the first implementation compatible with today's one-primary-view
JIT debug path.

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

## Debug provenance transition

Today, debug provenance definitions live effectively at the RVSDG layer and are
threaded downward through LIR and CFG-MIR.

With HIR in place, the intended ownership becomes:

- HIR owns the definition of semantic scopes, local identities, comments, and
  statement/source IDs
- RVSDG/LIR/CFG-MIR carry references or derived projections of that HIR-owned
  provenance
- DWARF generation ultimately reports locations for HIR-rooted semantic values,
  even when those values lower through many intermediate vregs or ops

The exact migration path is still open, but the direction is not: provenance
stops originating in RVSDG and starts originating in HIR.

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

- the long-term shared semantic target is HIR
- new frontends such as the Vixen rule language should lower into HIR from the
  start
- existing frontends may temporarily keep direct RVSDG lowering during the
  migration period

The frontends do not need to share a single concrete surface language. They do
need to share the same semantic lowering target.

That does not mean every frontend exposes the full HIR surface directly.

For example:

- the Vixen rule language is intentionally a constrained build-rule language
  with limited looping and other deliberate surface restrictions
- the deserializer frontend and the underlying Kajit HIR are broader and should
  not inherit those restrictions

HIR is the shared semantic target, not the user-visible promise that every
frontend supports every HIR construct.

Frontend-specific checked IRs may still exist above shared HIR.

For example, Vixen may keep a frontend-specific checked layer such as
`TypedExpr` for parsing, name resolution, and type checking. That checked
frontend IR is not the shared HIR. It lowers into shared HIR, which is the
contract eventually shared by the interpreter and Kajit lowering.

This keeps:

- parser/front-end concerns separate
- optimizer/backend reuse high
- debugger/source tooling unified below the frontend boundary

### Host callbacks and runtime interop need a typed contract

HIR needs an early typed contract for calls that cross into host/runtime code.

That includes at least:

- parameter and return types
- control transfer / fallibility metadata
- effect class
- domain-aware effect summary
- capability or policy metadata where applicable
- whether the call is part of the safe core, an opaque barrier, or an explicit
  unsafe / interop escape

The fallibility/control-transfer axis should stay orthogonal to the effect
lattice. The minimum useful categories are:

- returns normally
- may signal failure
- does not return

That metadata is needed both for host/runtime calls and for any helper calls
that participate in destination initialization or cleanup behavior.

This matters immediately for the Vixen side (`emit.*`, `env.*`) and for any
deserializer helpers that remain library/runtime calls rather than becoming
primitive HIR operations.

## Sketch of the intended HIR shape

HIR should print like a boring, readable program, not like SSA or a graph.

Example shape with borrowed output typing:

```text
fn decode_name<r_input>(
  cursor: Cursor<r_input>,
  destination out: Header<r_input>,
) {
  let mut handled = false

  loop {
    skip_ws(cursor)
    if peek_byte(cursor) == '}' {
      break
    }

    let key = read_key(cursor)
    expect_colon(cursor)

    match key {
      "name" => {
        let name_bytes = read_string_slice(cursor)
        let name = utf8_view(name_bytes)
        init out.name = name
        handled = true
      }
      _ => {}
    }

    if !handled {
      skip_unknown(cursor)
    }
  }
}
```

This is the level of readability HIR is meant to preserve.

If we choose to lower closer to a generic systems language core, the same logic
may also be expressible in a more explicit style like:

```text
let key = read_key(cursor)
if key == "name" && !handled {
  let name_bytes = read_string_slice(cursor)
  let name = utf8_view(name_bytes)
  init out.name = name
  handled = true
}
```

The important property is still the same: the program stays source-readable and
state is explicit, rather than encoded as a set of privileged parser-only
primitives.

## Open questions

These are intentionally left open for follow-up implementation work:

- how named state domains should be declared, interned, and surfaced to
  lowering once `PortKind::State(StateDomainId)` exists
- whether HIR's semantic type layer and layout metadata should live in one crate
  or in layered crates
- what the HIR story should be for dynamically typed rule-language values during
  the transition from today's Vixen runtime to a typed frontend
- how much lifetime/region syntax a frontend or pretty-printer should elide once
  HIR already stores provenance explicitly
- how destination modes should be represented in signatures once the v1
  entry/success/failure contract is fixed
- what handle safety strategy to use first: compile-time restrictions, runtime
  generation checks, or a combination
- how much expression nesting should be preserved before introducing explicit
  temporaries during lowering
- what concrete signature table to use first for host callbacks and runtime
  interop in the first HIR consumer
- whether source-view selection should be a compile-time option, runtime debug
  option, or both

## Summary

HIR is the planned semantic layer above RVSDG.

The design decisions in this document are:

- HIR is typed, but not machine-layout typed.
- HIR includes first-class structs and Rust-shaped enums.
- Struct and enum layout is semantic by default; concrete layout resolves at the
  HIR→RVSDG boundary.
- HIR keeps structured control flow.
- `match` is first-class in HIR.
- HIR uses statement sequencing for effects instead of state tokens.
- HIR operations carry a coarse effect classification for lowering.
- HIR operations and calls also carry domain-aware effect summaries for
  deterministic lowering into named RVSDG state domains.
- HIR calls also carry fallibility/control-transfer metadata orthogonal to the
  effect lattice.
- HIR is memory-safe by default.
- Borrowing and provenance are first-class in HIR, including borrowed input and
  borrowed output types for zero-copy deserialization.
- Places and definite initialization are part of the HIR contract, including a
  v1 destination contract for success and failure.
- Generated decoder planning concepts such as `flatten` and `untagged` stay
  above HIR; HIR executes generated low-level algorithms rather than modeling
  those concepts directly.
- `Place<T>` should be treated as provisional where generated host destinations
  are concerned; do not assume it is already the final universal abstraction.
- HIR does not expose raw pointers as a normal source feature.
- HIR owns semantic names, scopes, spans, comments, and debug identities.
- RVSDG remains the optimization boundary.
- RVSDG must generalize beyond the current deserializer-specific state-domain
  model, with cursor/output becoming named domains rather than hardcoded ones.
- CFG-MIR remains the backend/debug-codegen boundary.
- HIR should become the default debugger source view once implemented.
