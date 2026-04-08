# Why Kajit Needs A Language For Languages

Kajit currently has at least five serious representation layers to care about:

- HIR
- IR
- LIR
- CFG-MIR
- ASM

Possibly six, depending on whether we treat frontend-specific ASTs as first-class reprs too.

All of them need the same broad class of infrastructure:

- Rust node types
- parse / display / formatting
- round-tripping guarantees
- provenance
- diagnostics
- validation
- visitors / walkers / folders
- syntax highlighting
- linting
- eventually editor tooling, tree-sitter grammars, maybe LSP support

Right now, we mostly hand-roll all of this. That was tolerable when the project was smaller and the representations were in flux. It is no longer a reasonable strategy.

This note captures the long-term case for building custom representation infrastructure: a schema language, code generators, provenance support, validation support, and eventually rewrite tooling, instead of continuing to hand-author six overlapping little compiler frontends.

This is a vision note, not an implementation plan.

## The Current Situation

The existing text formats and ASTs all have deep structural problems.

### 1. They leak implementation accidents

Too many current node types and text formats encode compiler history instead of stage intent.

Examples:

- arena IDs and synthetic handles where symbols or explicit structure should appear
- builder-era scaffolding leaking into canonical text
- lowering-time details surviving into supposedly stable reprs
- internal conveniences becoming permanent surface area

The result is that many formats are neither clean human-facing syntax nor clean machine-facing interchange. They are just snapshots of whatever the implementation happened to need at the time.

### 2. They do not mechanically guarantee round-tripping

Today, round-tripping mostly works because we hope it does and because some tests happen to cover some cases.

That is not enough.

If text is supposed to be canonical, then parse/display round-tripping needs to be a property of the infrastructure, not a social aspiration backed by scattered handwritten tests.

Current failure mode:

- a repr evolves
- display changes a little
- parser lags behind, or vice versa
- nothing catches it until some test happens to exercise it

This is a bad contract for a compiler whose text formats are supposed to keep the implementation honest.

### 3. Provenance is not first-class

Without a strong provenance story, diagnostics will always be worse than they should be.

Every repr needs an answer to questions like:

- where did this node come from
- what source span or prior-stage artifact does it map back to
- if validation fails here, what do we point at
- if a node is rewritten, how does provenance propagate

Right now, provenance is weak, inconsistent, or absent. That means diagnostic quality is capped by the representation design.

### 4. Structural boilerplate is eating the codebase

There are hundreds of lines of traversal and visitor code that exist only because the ASTs are hand-authored snowflakes.

That is time spent on:

- visiting fields
- re-wrapping nodes
- keeping exhaustive matches in sync
- writing bespoke parse/display glue

This is exactly the kind of work that should be generated.

### 5. Reprs do not declare their own contracts clearly

A good repr should make it obvious:

- what is valid before parsing
- what is valid after parsing
- what invariants are guaranteed after normalization
- what information belongs at this stage
- what must not survive into this stage

Today, those preconditions and postconditions are mostly implicit. That makes every pass and every parser more fragile than it needs to be.

## The Underlying Pattern

The problem is not that HIR is bad, or IR is bad, or MIR is bad in isolation.

The problem is that we are solving the same representation problem repeatedly:

1. define node types
2. define syntax
3. define traversal
4. define provenance
5. define validation
6. define diagnostics
7. define tests

We are repeating this process for each repr by hand.

That is a scaling failure.

## The Proposal

Build custom infrastructure for representations.

Not just "a parser generator". Not just "a derive macro for visitors". The real need is a representation toolchain.

At minimum:

1. A schema language for reprs
2. Code generation from that schema
3. A provenance model baked into the schema
4. Validation hooks with explicit preconditions / postconditions
5. Derived parse/display/round-trip infrastructure
6. Derived visitors / walkers / folders
7. Eventually, editor/tooling artifacts downstream of the same schema

## Vision Versus First Implementation

The long-term description really is "a language for languages".

That is the right destination because the point is to separate representation design from representation implementation.

But it is the wrong near-term planning label.

The first implementation should be much smaller:

- a representation schema system for stable Kajit layers
- initially piloted on HIR and MIR
- generating only boring structural infrastructure

Specifically, the first implementation should focus on:

- node definitions
- visitor / walker / folder scaffolding
- parse / display glue
- round-trip harnesses
- provenance shells

And should explicitly defer:

- rewrite DSLs
- lint DSLs
- tree-sitter generation
- LSP generation
- ambitious validation DSLs

Those may still be the right direction later. They are not version 1.

## Stage Contracts Come First

Before building the generator, Kajit needs sharper per-stage contracts.

For each repr, we need explicit answers to:

- what information belongs here
- what information must not survive from earlier stages
- what identities are canonical here
- what round-trip guarantee the text format actually promises
- what provenance is required here
- what invariants are guaranteed after parsing or normalization

Without those contracts, a schema system will just generate cleaner confusion.

## The Schema Should Describe

For each repr:

- node definitions
- enums and structs
- fields and field cardinality
- identity model
  - symbol
  - index
  - block id
  - port id
  - etc.
- textual form
  - canonical syntax
  - optional sugar
  - formatting hints
- provenance attachment
- normalization phases
- invariants
  - parse-time invariants
  - canonical-form invariants
  - stage-specific validation rules

This would become the real source of truth.

## The Generated Artifacts

From that schema, we should eventually generate a ton of code.

That is not a side effect. That is the point.

Expected outputs:

- Rust structs/enums for nodes
- builders for well-formed nodes
- field accessors
- visitors / walkers / folders
- parse skeletons or full parsers
- display / pretty-printing support
- round-trip tests
- provenance plumbing
- validation skeletons
- formatting helpers
- later: tree-sitter grammars
- later: semantic tokens / LSP scaffolding

The line should be:

- generated code handles structure
- handwritten code handles semantics

Meaning:

- generated: AST shape, traversal, serialization syntax glue, boilerplate validation surfaces
- handwritten: lowering, analysis, cost models, nontrivial optimizations, backend logic

## Why This Is Not Overengineering

This sounds indulgent until you count the alternatives.

Without a shared representation framework, we are committing to:

- hand-maintaining five or six AST families
- hand-maintaining five or six text formats
- hand-maintaining round-trip correctness
- hand-maintaining provenance conventions
- hand-maintaining visitors and folds
- hand-maintaining syntax tooling forever

That is far more expensive than building infrastructure once.

The project has crossed the threshold where custom infrastructure is cheaper than repeated bespoke infrastructure.

## This Also Enables Mechanical Rewrite DSLs

Once the node model is schema-driven, we can generate typed code that manipulates those nodes.

That means mechanical optimization passes can themselves move into a DSL.

Potentially:

- pattern-matching rewrites
- canonicalization rules
- peephole simplifications
- structural lowering steps
- validation-aware rewrites
- provenance-preserving rewrites

The key point is that rewrite tooling gets much easier once the AST structure is generated and uniform.

Instead of hand-writing endless pattern matches over bespoke enums, we can generate:

- typed matchers
- builders
- rewrite helpers
- exhaustiveness checks
- provenance propagation code

This does not replace all hand-written optimization work. It does remove a huge class of mechanical pass boilerplate.

This should be treated as a later payoff, not the first milestone.

## Desired End State

For every repr in Kajit:

- canonical text format exists and round-trips by construction
- nodes carry provenance in a standard way
- invariants are explicit
- traversal is derived, not hand-authored
- syntax tooling is downstream of the same schema
- rewrites can be expressed declaratively where appropriate

And the repr crates become what they should have been all along:

- stable semantic definitions
- syntax ownership
- validation ownership

Not ad hoc bags of types and whatever helper logic happened to accrete nearby.

## Immediate Next Questions

1. What are the per-stage contracts for HIR, IR, MIR, ASM, and does LIR survive?
2. What is the smallest viable schema for one repr, probably HIR?
3. What provenance model should every node share?
4. Which invariants belong in the schema and which remain handwritten?
4. How much parse/display should be fully generated vs generated skeleton + handwritten details?
5. What is the first useful codegen target:
   - visitors
   - ASTs
   - parse/display
   - round-trip harness
6. What is the smallest rewrite DSL worth building after the schema exists?

## Recommendation

Start with one pilot repr, probably HIR, and force ourselves to answer:

- what the canonical syntax is
- what the canonical AST is
- what provenance means
- what post-parse invariants are

Then make the generator prove that it can eliminate:

- handwritten visitor boilerplate
- handwritten parse/display drift
- ad hoc provenance holes

If that works for HIR, the rest of the stack stops looking like six separate problems and starts looking like one infrastructure problem solved repeatedly.
