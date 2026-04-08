# First Implementation Scope

This note narrows the work from the long-term "language for languages" vision to a first implementation that can actually ship.

## Goal

Build a representation schema + generation framework for Kajit's stable layers.

The first pilot should target:

- HIR
- MIR

Not all layers equally.

## Why HIR And MIR

HIR:

- clearly real
- human-facing
- diagnostics-sensitive
- full of boilerplate and syntax concerns

MIR:

- clearly real
- executable CFG contract
- good target for canonical parse/display and visitors
- different enough from HIR to prove the framework is not too narrow

## Explicit Non-Goals For Version 1

- no rewrite DSL
- no lint DSL
- no tree-sitter generation
- no LSP generation
- no grand unification of every layer at once
- no heroic validation framework
- no special investment in LIR until it justifies itself

## Generation Targets

For version 1, generation should only handle boring structural wins:

- Rust node types
- visitor / walker / folder traits and impls
- parser scaffolding
- formatter / display scaffolding
- round-trip test harnesses
- provenance attachment shells

Handwritten code still owns:

- semantic lowering
- serious analyses
- nontrivial validation
- optimization passes
- backend logic

## Preconditions

Before the pilot really takes off, each target repr needs a contract note that answers:

- purpose
- canonical identities
- required provenance
- round-trip promise
- forbidden leaks from earlier stages
- post-parse / post-normalization invariants

Without that, the generator will formalize the wrong thing.
