# HIR Boundary For Generated Decoders

Status: corrective note for the first HIR implementation wave.

This note freezes one important boundary that the main HIR design doc only
partially captured: generated deserializers are implemented *in* HIR, but the
compile-time planning concepts that produce those programs live *above* HIR.

## What HIR is

HIR is a real low-level systems language.

It should be able to express:

- locals and mutable state
- structured control flow
- fixed arrays and indexing
- structs and enums
- slices and borrowed views
- typed address values when generated code needs to talk about allocated memory
- explicit initialization and overwrite
- calls with typed signatures and effect summaries

Generated decode programs are ordinary HIR programs that happen to be emitted
from `Shape`-driven planning logic.

## What HIR is not

HIR is not where we represent format-planning concepts directly.

In particular, HIR should not know about:

- `flatten` as a semantic feature
- `untagged` as a semantic feature
- candidate-set narrowing
- evidence feeding
- solver plans
- ambiguity-resolution policies

Those are compile-time concepts above HIR.

The shape planner is responsible for:

- analyzing Rust types and facet attributes
- building candidate spaces and ambiguity rules
- computing evidence tables / dispatch plans
- choosing concrete generated algorithms

HIR is only the target language used to implement the chosen algorithm.

## Consequence For Flatten And Untagged

`flatten` + `untagged` should compile to ordinary low-level HIR code.

That generated HIR code may contain:

- mutable scratch state
- bitmasks or fixed tables
- partially initialized aggregate state
- loops over fields
- branch-and-merge control flow
- indexed writes into structured storage

But the HIR program should not carry first-class notions like "candidate set"
or "feed evidence". Those are planner-level concepts that explain *why* the
generated code exists, not HIR-level semantics.

## HIR Types Versus Host Types

Rust host types do not map 1:1 to HIR types.

Examples:

- a Rust `Vec<T>` does not imply that HIR should have a semantic `Vec<T>` type
- a Rust `#[repr(u8)] enum` does not imply HIR must use the same physical
  layout
- a Rust flattened/untagged subtree does not imply HIR needs flatten/untagged
  syntax

HIR has its own semantic type system. Host layout/schema information may still
be referenced during lowering, but that is a separate concern.

This matters especially for host-owned values such as Rust `String`.

Generated HIR does not need to model every such host value as a first-class HIR
value. A generated decoder may instead:

- compute low-level ingredients such as byte ranges, lengths, capacities, and
  addresses
- use host layout/schema information to identify the destination subtree
- materialize the host value directly into that destination

So the important boundary is:

- HIR values are HIR semantic values
- host values may be constructed by destination-directed materialization without
  ever existing as ordinary HIR locals or return values

That same boundary applies to format logic. Format-specific parsing should be
ordinary HIR code where possible:

- byte loads
- scalar arithmetic and comparisons
- explicit cursor updates
- structured control flow

Runtime calls remain appropriate for true runtime boundaries such as allocation
and host-value materialization. They are not the place to hide format semantics
that should live in generated HIR programs.

## Place Pressure Point

The current `Place<T>` spelling is under pressure.

`T` is well-defined when the place refers to ordinary HIR storage whose type is
an HIR semantic type. It is *not* obviously well-defined for generated
deserializer output storage, where the destination may be better described as a
host-schema subtree than as "an HIR value of type `T`".

So the important boundary is:

- HIR values have HIR types
- generated output destinations may need schema/layout identity in addition to,
  or instead of, pure HIR type identity

This note does not settle the replacement design. It does freeze one thing:
we should stop assuming that `Place<T>` is automatically the correct universal
abstraction for both ordinary HIR values and generated host destinations.

One practical consequence is that destination-directed materialization is a
first-class lowering pattern for generated decoders. If a host value is best
described as "a destination subtree with known layout plus some raw
ingredients", HIR lowering should use that model instead of forcing the host
value through a "compute value, then assign" path.

## Safety Direction

Raw pointers stay out of HIR.

The intended mutation model is still:

- structured places/projections
- definite initialization
- overwrite with defined replace/drop semantics
- borrowed views with explicit provenance
- safe builders or scratch-state constructs when needed

If generated code needs to assemble heap-backed results honestly, HIR may still
need direct memory construction primitives. The intended boundary there is:

- HIR may talk about typed addresses
- address values should be explicit about allocation domain
- at minimum, generated decoders need to distinguish transient scratch/chunk
  allocation from persistent allocation that becomes part of the returned Rust
  value
- raw pointers, integer casts, and unconstrained aliasing still belong below
  HIR

If generated code wants *untyped* pointer-shaped lowering, that belongs below
HIR.

## Immediate Implementation Rule

During the current implementation phase:

- only add HIR features that are justified by small executable HIR kernels
- do not add planner concepts as HIR syntax or HIR runtime objects
- treat any feature that bakes host-type identity directly into HIR semantics as
  suspect until the boundary is explicit
