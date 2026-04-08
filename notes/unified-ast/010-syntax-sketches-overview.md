# Syntax Sketches Overview

The next useful deliverable is not code. It is a set of syntax sketches for the pipeline layers.

The point is to answer:

- what is each text format for
- who is it for
- what are the primary visual units
- what should be symbolic
- what should be explicit
- what should be impossible to express

These sketches are deliberately not constrained by the current parsers or ASTs.

## Principles

### 1. Each stage should have syntax shaped around its job

Not every repr should look alike.

- HIR should read like structured typed source IR
- IR should read like a graph language
- LIR should read like a linear transport form, if it survives
- MIR should read like a CFG
- ASM should stay close to symbolic assembly

### 2. Canonical print form, optional parse sugar

The compiler should print one canonical form.

Parsers may accept a little sugar if useful, but the display format should be stable and boring.

### 3. No implementation accidents in syntax

Do not print:

- arena handles just because they exist
- lowering hacks
- builder-era scaffolding
- host addresses
- internal bookkeeping that is not part of the stage contract

### 4. Text is a contract

Each format should make clear:

- what is valid at this stage
- what identities exist at this stage
- what kind of provenance is available
- what invariants are expected after parsing

## Files

- `011-hir-syntax-sketch.md`
- `012-ir-syntax-sketch.md`
- `013-lir-syntax-sketch.md`
- `014-mir-syntax-sketch.md`
- `015-asm-syntax-sketch.md`

These are sketches, not commitments.
