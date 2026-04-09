# Next Steps

This note captures the immediate post-PoC task stack for the unified AST work.

It is ordered on purpose.

## Current State

What is real today:

- schema parsing via `facet-styx`
- normalized schema model in `kajit-foundation`
- generated HIR PoC AST defs
- generated provenance/common placeholder types
- generated visitors/walkers
- generated `chumsky` parser for a tiny HIR subset
- PoC smoke test in `kajit-reprs`

What is not real yet:

- formatter generation
- full HIR schema coverage
- real shared HIR support types generated from schema
- full provenance model
- invariant/validation framework
- MIR pilot on the same footing

## Ordered Tasks

### 1. Generate The Canonical Formatter

Generate formatter/display code from `canonical_print`.

Why first:

- parse without print is not a canonical text boundary
- round-tripping is still incomplete until both sides are generated
- this is the next boring structural win after parser generation

Deliverables:

- generated formatter entry points
- generated canonical-print helpers
- generated round-trip tests for the HIR pilot

### 2. Replace Placeholder Shared Types

Move the shared HIR support types into the schema instead of faking them in codegen.

Likely first targets:

- `Type`
- `Literal`
- `BinaryOp`
- `TypeDef`
- `FieldDef`
- `GenericParam`

Why second:

- the AST cannot become serious while these remain placeholders
- the formatter/parser story depends on these types being real schema-owned data

Deliverables:

- schema-owned defs for the shared HIR support surface
- generated Rust types for those defs
- generator no longer hardcodes placeholder stand-ins for them

### 3. Expand The HIR Schema Beyond The Pilot Slice

Broaden the HIR schema from the tiny PoC subset to the real HIR surface.

Why third:

- once support types are real, the node set can expand without multiplying fake edges
- this is the point where the schema starts describing actual HIR instead of a demo

Deliverables:

- broader HIR node coverage
- parser/formatter coverage expanded accordingly
- fewer hand-maintained HIR repr pieces outside the schema

### 4. Settle Identity Policy

Write down and encode what is:

- symbolic
- a stable typed ID
- purely structural

Why fourth:

- identity choices leak everywhere once the schema gets larger
- this needs to be explicit before generated AST shape hardens too much

Deliverables:

- HIR contract update for identities
- schema conventions for symbolic names vs typed IDs
- generator support for those conventions

### 5. Upgrade Provenance

Move beyond "every node has `Prov`" into a proper origin model.

Questions to answer:

- direct source span vs synthesized nodes
- cross-stage origin tracking
- merged provenance
- what diagnostics need at HIR stage

Why fifth:

- a shell exists already
- the model should mature after the real node/type surface is clearer

Deliverables:

- stronger provenance schema model
- generated provenance plumbing that reflects actual origin cases

### 6. Add Invariants And Validation Hooks

Only after the real HIR surface is better modeled.

Why last in this sequence:

- premature validation tends to formalize the wrong thing
- the node/type/identity/provenance story should settle first

Deliverables:

- schema-level invariant declarations or hooks
- generated validation entry points
- clear split between generated structural checks and handwritten semantic checks

## Explicitly Not Next

Not yet:

- rewrite DSL
- lint DSL
- tree-sitter generation
- LSP generation
- MIR expansion before HIR is less fake

Those still make sense long-term, but they are not the next implementation tasks.

## Practical Recommendation

The next concrete implementation step should be:

1. formatter generation from `canonical_print`
2. then schema-owned shared HIR support types

That keeps the current PoC honest and extends it in the most useful direction.
