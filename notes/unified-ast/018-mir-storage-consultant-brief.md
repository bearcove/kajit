# MIR Storage Consultant Brief

This document is a review packet for the current schema-first MIR storage work in Kajit.

It is intended to let an external reviewer answer:

- Is the current `pool` / `order` split the right first storage ontology?
- Is the current `keyed pool` / `directional ref` extension the right second step?
- What should the next storage step be?
- How should this evolve toward real MIR storage without entangling the existing handwritten CFG-MIR implementation?

## Goal

We are **not** rewriting the existing CFG-MIR implementation yet.

We are building a **parallel, schema-owned representation** for MIR so that storage semantics can become explicit in the schema and generated code before any migration of live passes.

The current step is:

- express storage intent in schema
- preserve that intent through normalization/codegen
- generate distinct Rust surface types for:
  - entity pools
  - ordered slot lists
  - keyed pools
  - directional id references to pool entities

The current step is **not**:

- arena-backed storage
- tombstones
- compaction/remap tables
- replacing handwritten CFG-MIR

## Pipeline Context

Kajit’s pipeline is:

```text
Schema (facet Shape)
  → HIR
  → IR (RVSDG)
  → LIR
  → CFG-MIR
  → Register Allocation
  → Backend
```

MIR/CFG-MIR is the stage we care about here:

- explicit basic blocks
- explicit CFG edges
- explicit instruction/terminator ids
- close to backend/regalloc needs
- likely rewrite-heavy enough that storage shape matters

The existing handwritten CFG-MIR implementation is still the real one today. This report is about the **new schema-owned MIR pilot**.

## Relevant Files

### Schema / Draft

- MIR pilot schema:
  - [/Users/amos/bearcove/kajit/notes/unified-ast/pilot/mir.repr.styx](/Users/amos/bearcove/kajit/notes/unified-ast/pilot/mir.repr.styx)

### Foundation source

- schema AST for `.styx` repr docs:
  - [/Users/amos/bearcove/kajit/crates-kajit/kajit-foundation/src/schema.rs](/Users/amos/bearcove/kajit/crates-kajit/kajit-foundation/src/schema.rs)
- normalization:
  - [/Users/amos/bearcove/kajit/crates-kajit/kajit-foundation/src/normalize.rs](/Users/amos/bearcove/kajit/crates-kajit/kajit-foundation/src/normalize.rs)
- Rust type rendering helpers:
  - [/Users/amos/bearcove/kajit/crates-kajit/kajit-foundation/src/render_helpers.rs](/Users/amos/bearcove/kajit/crates-kajit/kajit-foundation/src/render_helpers.rs)
- module/root codegen:
  - [/Users/amos/bearcove/kajit/crates-kajit/kajit-foundation/src/render_module.rs](/Users/amos/bearcove/kajit/crates-kajit/kajit-foundation/src/render_module.rs)
- parser codegen:
  - [/Users/amos/bearcove/kajit/crates-kajit/kajit-foundation/src/parser_codegen.rs](/Users/amos/bearcove/kajit/crates-kajit/kajit-foundation/src/parser_codegen.rs)
- tests:
  - [/Users/amos/bearcove/kajit/crates-kajit/kajit-foundation/src/tests.rs](/Users/amos/bearcove/kajit/crates-kajit/kajit-foundation/src/tests.rs)

### Generated output

- generated shared wrappers:
  - [/Users/amos/bearcove/kajit/crates-kajit/kajit-reprs/src/schema_poc/mod.rs](/Users/amos/bearcove/kajit/crates-kajit/kajit-reprs/src/schema_poc/mod.rs)
- generated MIR AST:
  - [/Users/amos/bearcove/kajit/crates-kajit/kajit-reprs/src/schema_poc/mir/ast.rs](/Users/amos/bearcove/kajit/crates-kajit/kajit-reprs/src/schema_poc/mir/ast.rs)
- generated MIR metadata:
  - [/Users/amos/bearcove/kajit/crates-kajit/kajit-reprs/src/schema_poc/mir/meta.rs](/Users/amos/bearcove/kajit/crates-kajit/kajit-reprs/src/schema_poc/mir/meta.rs)

### Existing handwritten CFG-MIR

For comparison only, not being changed in this slice:

- [/Users/amos/bearcove/kajit/crates-kajit/kajit-mir/src/cfg_mir.rs](/Users/amos/bearcove/kajit/crates-kajit/kajit-mir/src/cfg_mir.rs)

## Current Design Direction

The schema can now express four distinct ideas:

1. `@entity`
2. `@slot`
3. type-shape wrappers:
   - `@pool(T)`
   - `@order(T)`
4. relationship annotations:
   - `@pool(T @key(@IdType))`
   - `@ref_to(@IdType @TargetEntity)`

Interpretation:

- `@entity`
  - identity-bearing node kind
- `@slot`
  - ordered parent-owned node kind
- `@pool(T)`
  - storage of entity-ish things
  - current generated Rust type: `Pool<T>`
- `@order(T)`
  - meaningful ordered slots/references
  - current generated Rust type: `Order<T>`
- `@pool(T @key(@IdType))`
  - says which id type keys the entity pool
- `@ref_to(@IdType @TargetEntity)`
  - says a field stores ids resolving to a specific entity pool

## Schema Excerpts

From the MIR pilot schema:

```styx
nodes {
    Program @node{
        prov @Prov
        docs @optional(@DocBlock)
        vreg_count @Nat
        slot_count @Nat
        functions @pool(@Function @key(@FunctionId))
    }

    Function @entity{
        prov @Prov
        docs @optional(@DocBlock)
        lambda_id @LambdaId
        function_id @FunctionId
        entry @ref_to(@BlockId @Block)
        data_args @DataArgsLine
        data_results @DataResultsLine
        blocks @pool(@Block @key(@BlockId))
        insts @pool(@Inst @key(@InstId))
        terms @pool(@Terminator @key(@TermId))
        edges @pool(@Edge @key(@EdgeId))
    }

    Block @entity{
        prov @Prov
        docs @optional(@DocBlock)
        id @BlockId
        params @order(@VReg)
        insts @order(@ref_to(@InstId @Inst))
        term @ref_to(@TermId @Terminator)
        preds @order(@ref_to(@EdgeId @Edge))
        succs @order(@ref_to(@EdgeId @Edge))
    }

    Edge @entity{
        prov @Prov
        docs @optional(@DocBlock)
        id @EdgeId
        from @ref_to(@BlockId @Block)
        to @ref_to(@BlockId @Block)
        args @order(@EdgeArg)
    }
}
```

The intended meaning is:

- a function **owns pools** of blocks/instructions/terminators/edges
- blocks and edges then **arrange references** through ordered lists

This is intended to reflect the distinction:

- entity existence/ownership
- versus semantic ordering/membership

## Generated Rust Surface

The generator now emits real wrappers for storage-shape distinctions.

From the shared generated module:

```rust
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct Pool<T>(pub Vec<T>);

#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct Order<T>(pub Vec<T>);
```

Current behavior:

- wrappers are still just `Vec`-backed
- they implement `From<Vec<T>>`, `Deref`, `DerefMut`, and iteration
- this is a **semantic surface distinction**, not yet a backend storage distinction

From generated MIR AST:

```rust
pub struct Function {
    pub blocks: super::super::Pool<Block>,
    pub edges: super::super::Pool<Edge>,
    pub insts: super::super::Pool<Inst>,
    pub terms: super::super::Pool<Terminator>,
    pub entry: BlockId,
    // ...
}

pub struct Block {
    pub insts: super::super::Order<InstId>,
    pub params: super::super::Order<VReg>,
    pub preds: super::super::Order<EdgeId>,
    pub succs: super::super::Order<EdgeId>,
    pub term: TermId,
    // ...
}

pub struct Edge {
    pub args: super::super::Order<EdgeArg>,
    // ...
}
```

This is the first point where generated MIR no longer collapses both concepts into raw `Vec<_>`.

## Generated Metadata

Generated metadata preserves the storage distinction in machine-readable form.

Examples from generated MIR metadata:

```rust
FieldSpec { owner: "Function", field: "blocks", kind: "pool<Block key=BlockId>" }
FieldSpec { owner: "Function", field: "insts", kind: "pool<Inst key=InstId>" }
FieldSpec { owner: "Function", field: "entry", kind: "ref<BlockId -> Block>" }
FieldSpec { owner: "Block", field: "insts", kind: "order<ref<InstId -> Inst>>" }
FieldSpec { owner: "Block", field: "preds", kind: "order<ref<EdgeId -> Edge>>" }
FieldSpec { owner: "Edge", field: "args", kind: "order<EdgeArg>" }
```

The current slice also generates explicit descriptors:

```rust
PoolSpec { owner: "Function", field: "blocks", item: "Block", key: "BlockId" }
PoolSpec { owner: "Function", field: "insts", item: "Inst", key: "InstId" }
PoolSpec { owner: "Function", field: "terms", item: "Terminator", key: "TermId" }

RefSpec { owner: "Function", field: "entry", id: "BlockId", target: "Block" }
RefSpec { owner: "Block", field: "insts", id: "InstId", target: "Inst" }
RefSpec { owner: "Block", field: "preds", id: "EdgeId", target: "Edge" }
RefSpec { owner: "Block", field: "term", id: "TermId", target: "Terminator" }
RefSpec { owner: "Edge", field: "from", id: "BlockId", target: "Block" }
```

This matters because a later generator can consume metadata rather than re-infer intent from field names.

## Foundation Implementation Notes

### Schema layer

`TypeUse` now supports:

```rust
pub(crate) enum TypeUse {
    Optional(Vec<TypeUse>),
    Seq(Vec<TypeUse>),
    Pool(Vec<TypeUse>),
    Order(Vec<TypeUse>),
    Key(Vec<TypeUse>),
    RefTo(Vec<TypeUse>),
    Ref { name: Option<String> },
}
```

### Normalization layer

`SyntaxTypeUse` now preserves:

```rust
pub(crate) enum SyntaxTypeUse {
    Optional(Box<SyntaxTypeUse>),
    Seq(Box<SyntaxTypeUse>),
    Pool { item: Box<SyntaxTypeUse>, key: Option<String> },
    Order(Box<SyntaxTypeUse>),
    RefTo { id: Box<SyntaxTypeUse>, target: String },
    Ref { name: String },
}
```

### Rendering layer

Current rendering policy:

- `Seq(T)` => `Vec<T>`
- `Pool(T)` => `Pool<T>`
- `Order(T)` => `Order<T>`
- `RefTo(Id, Target)` => renders as `Id` in Rust, but preserves `Target` in metadata

Current parser behavior:

- repeat parsers still collect `Vec<T>`
- then wrap them with:
  - `Pool::from(vec)`
  - `Order::from(vec)`

So this is still a parser for value/tree structures, not arena-backed storage construction.

## What This Is Good For Already

This work already pays rent in several ways:

1. MIR schema can state storage intent explicitly.
2. Generated Rust can distinguish:
   - entity storage
   - ordered slots
3. Generated metadata can drive future storage-aware codegen.
4. We can review MIR storage shape without touching handwritten CFG-MIR.

## What This Does Not Solve Yet

This is still **not** a real storage backend.

Missing pieces:

- no arena-backed pools
- no typed pool handles beyond whatever support ids the schema already defines
- no tombstones/liveness policy
- no compaction/remap support
- no generated “entity pool” APIs
- no generated validation that ordered membership references only live entities
- no generated remap/compaction scaffolding from `PoolSpec` / `RefSpec`

In other words:

- we now have a **storage vocabulary**
- we do not yet have a **storage runtime/model**

## The Main Architectural Question

We want advice on the next honest step.

Right now the generated MIR world looks like:

- syntax-aware
- tooling-aware
- schema-aware
- still tree/value-backed

The question is what should come next.

## Specific Questions For Review

1. Is `pool` vs `order` the right fundamental split?

Put differently:

- `pool` = existence/ownership of entities
- `order` = parent-owned meaningful slot order

Is this the correct base ontology for CFG-MIR storage?

2. Should `Pool<T>` remain a wrapper over `Vec<T>` for a while, or should the next step be an arena API immediately?

Possible directions:

- keep `Pool<T>` as semantic wrapper only
- evolve `Pool<T>` into arena-like storage
- generate a parallel storage layer from metadata instead of changing `Pool<T>` directly

3. Are keyed pools and `ref_to(...)` the right way to express storage relationships?

The schema now explicitly says things like:

- `blocks @pool(@Block @key(@BlockId))`
- `edges @pool(@Edge @key(@EdgeId))`
- `preds @order(@ref_to(@EdgeId @Edge))`
- `term @ref_to(@TermId @Terminator)`

Is this the right schema shape, or should pool keys / refs live in a separate storage-specific section?

4. Should `order` always store ids/references, or can it also store inline slot nodes?

We currently use both:

- `Order<InstId>` for block instruction order, but now annotated as `Order<RefTo<InstId, Inst>>` at schema level
- `Order<EdgeArg>` for inline edge-argument slots

Is that still the right abstraction, or should “ordered inline slots” and “ordered entity references” be distinct concepts?

5. What is the right next generated artifact?

Candidates:

- richer metadata only
- storage descriptors
- generated validation from `PoolSpec` / `RefSpec`
- generated parallel arena-backed storage structs
- generated remap/compaction scaffolding

Our bias is:

- do not mutate current handwritten CFG-MIR yet
- generate validation and/or a parallel storage-oriented view first

6. How should tombstones/liveness enter the model?

We have already discussed internally that CFG-MIR likely wants:

- stable ids
- explicit order lists
- tombstones before compaction

But none of that is in the generated system yet.

Should liveness be:

- a schema-level property of entities/pools
- a generated storage-layer concern
- or kept out of the schema initially?

## Suggested Review Focus

If time is limited, the highest-value review would be:

1. sanity-check the `pool` / `order` distinction
2. say whether `Pool<T>` / `Order<T>` wrappers are a good intermediate step
3. recommend the next storage artifact to generate
4. say whether the schema should explicitly encode:
   - entity id type
   - pool key type
   - liveness/removal policy
   - directional refs to pool entities

## Verification Status

The current state is checked and installed.

Commands run successfully:

```bash
cargo run -p kajit-foundation-cli -- repr-poc
cargo nextest run -p kajit-foundation
cargo check -p kajit-reprs
cargo nextest run -p kajit-cli
cargo xtask install
```

Recent commits for this work:

- `763d31f` `Add pool and order storage shapes`
- `2a2b79a` `Emit pool and order wrapper types`
- current uncommitted slice adds:
  - `@key(...)`
  - `@ref_to(...)`
  - generated `PoolSpec`
  - generated `RefSpec`

## Bottom Line

Current status:

- MIR storage semantics are now explicit enough to review seriously.
- The schema can distinguish entity pools from ordered slots.
- The schema can state which ids key which pools.
- The schema can state which fields are directional refs into which pools.
- Generated Rust and generated metadata preserve those distinctions.
- We are still one level away from real generated storage.

The next decision is no longer “what do we mean by storage?”  
It is “what is the next generated storage artifact that should exist on top of `PoolSpec` / `RefSpec`?”
