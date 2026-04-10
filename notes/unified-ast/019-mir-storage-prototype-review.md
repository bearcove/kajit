# MIR Storage Prototype Review

This document is a review packet for the current MIR storage prototype in Kajit.

The point of this packet is not "should we keep hand-writing this?" We should not.

The point is:

- we now have a small handwritten storage API over schema-owned MIR
- it is explicitly a prototype target for future generation
- we want review on whether this API shape is the right one to generate

## Short Version

Current state:

- MIR schema can express:
  - `@entity`
  - `@slot`
  - `@pool(T @key(@Id))`
  - `@order(T)`
  - `@ref_to(@Id @Target)`
- generated code already preserves:
  - pool descriptors
  - ref descriptors
  - validation from those descriptors
- we added one **temporary handwritten prototype**:
  - typed storage/query access over generated MIR

Question for review:

- is this storage/query API the right shape to move into generated code?

## What We Are Not Doing

We are **not**:

- lowering generated MIR into the legacy handwritten CFG-MIR implementation
- rewriting live `kajit-mir` passes yet
- committing to arena runtime/tombstones/compaction yet

We are trying to answer a narrower question:

- once MIR is schema-owned, what should the first generated storage/query layer look like?

## Relevant Files

### Schema / Generation Inputs

- MIR pilot schema:
  - [/Users/amos/bearcove/kajit/notes/unified-ast/pilot/mir.repr.styx](/Users/amos/bearcove/kajit/notes/unified-ast/pilot/mir.repr.styx)
- foundation module generation:
  - [/Users/amos/bearcove/kajit/crates-kajit/kajit-foundation/src/render_module.rs](/Users/amos/bearcove/kajit/crates-kajit/kajit-foundation/src/render_module.rs)

### Generated MIR

- generated MIR AST:
  - [/Users/amos/bearcove/kajit/crates-kajit/kajit-reprs/src/mir/ast.rs](/Users/amos/bearcove/kajit/crates-kajit/kajit-reprs/src/mir/ast.rs)
- generated MIR metadata:
  - [/Users/amos/bearcove/kajit/crates-kajit/kajit-reprs/src/mir/meta.rs](/Users/amos/bearcove/kajit/crates-kajit/kajit-reprs/src/mir/meta.rs)
- generated MIR validation:
  - [/Users/amos/bearcove/kajit/crates-kajit/kajit-reprs/src/mir/validate.rs](/Users/amos/bearcove/kajit/crates-kajit/kajit-reprs/src/mir/validate.rs)

### Prototype Storage API

- temporary handwritten storage layer:
  - [/Users/amos/bearcove/kajit/crates-kajit/kajit-reprs/src/mir/storage.rs](/Users/amos/bearcove/kajit/crates-kajit/kajit-reprs/src/mir/storage.rs)
- tests for that layer:
  - [/Users/amos/bearcove/kajit/crates-kajit/kajit-reprs/tests/mir_storage.rs](/Users/amos/bearcove/kajit/crates-kajit/kajit-reprs/tests/mir_storage.rs)

### Legacy Runtime For Comparison Only

- current handwritten CFG-MIR:
  - [/Users/amos/bearcove/kajit/crates-kajit/kajit-mir/src/cfg_mir.rs](/Users/amos/bearcove/kajit/crates-kajit/kajit-mir/src/cfg_mir.rs)

## Schema Storage Facts We Already Have

The schema now knows:

- which fields are keyed pools:
  - `blocks @pool(@Block @key(@BlockId))`
  - `insts @pool(@Inst @key(@InstId))`
  - `terms @pool(@Terminator @key(@TermId))`
  - `edges @pool(@Edge @key(@EdgeId))`
- which fields are refs into those pools:
  - `entry @ref_to(@BlockId @Block)`
  - `term @ref_to(@TermId @Terminator)`
  - `preds @order(@ref_to(@EdgeId @Edge))`
  - `succs @order(@ref_to(@EdgeId @Edge))`
  - `insts @order(@ref_to(@InstId @Inst))`
- which ordered fields are inline payloads instead of refs:
  - `params @order(@VReg)`
  - `args @order(@EdgeArg)`

Generated metadata exposes that explicitly through:

```rust
pub static POOLS: &[PoolSpec] = &[
    PoolSpec { owner: "Function", field: "blocks", item: "Block", key: "BlockId" },
    PoolSpec { owner: "Function", field: "edges", item: "Edge", key: "EdgeId" },
    PoolSpec { owner: "Function", field: "insts", item: "Inst", key: "InstId" },
    PoolSpec { owner: "Function", field: "terms", item: "Terminator", key: "TermId" },
    PoolSpec { owner: "Program", field: "functions", item: "Function", key: "FunctionId" },
];

pub static REFS: &[RefSpec] = &[
    RefSpec { owner: "Block", field: "insts", id: "InstId", target: "Inst" },
    RefSpec { owner: "Block", field: "preds", id: "EdgeId", target: "Edge" },
    RefSpec { owner: "Block", field: "succs", id: "EdgeId", target: "Edge" },
    RefSpec { owner: "Block", field: "term", id: "TermId", target: "Terminator" },
    RefSpec { owner: "Edge", field: "from", id: "BlockId", target: "Block" },
    RefSpec { owner: "Edge", field: "to", id: "BlockId", target: "Block" },
    RefSpec { owner: "Function", field: "entry", id: "BlockId", target: "Block" },
    // plus terminator edge refs...
];
```

Generated validation already uses those facts to reject bad refs.

## Temporary Prototype API

The current temporary handwritten layer exposes:

```rust
pub struct ProgramStorage<'a> {
    // lookup functions by FunctionId
}

pub struct FunctionStorage<'a> {
    // lookup blocks/edges/insts/terms by typed id
}
```

And the operations it supports are:

- `ProgramStorage::new(&Program)`
- `ProgramStorage::function(FunctionId)`
- `ProgramStorage::function_storage(FunctionId)`
- `FunctionStorage::block(BlockId)`
- `FunctionStorage::edge(EdgeId)`
- `FunctionStorage::inst(InstId)`
- `FunctionStorage::term(TermId)`
- `FunctionStorage::entry_block()`
- `FunctionStorage::block_insts(&Block)`
- `FunctionStorage::block_preds(&Block)`
- `FunctionStorage::block_succs(&Block)`
- `FunctionStorage::block_term(&Block)`
- `FunctionStorage::edge_from(&Edge)`
- `FunctionStorage::edge_to(&Edge)`
- `FunctionStorage::terminator_edges(&Terminator)`

It also rejects duplicate ids in pools:

- duplicate `BlockId`
- duplicate `EdgeId`
- duplicate `InstId`
- duplicate `TermId`
- duplicate `FunctionId`

with a typed `StorageError`.

## Why This Prototype Exists

We wanted to stop reasoning abstractly about storage and force one concrete question:

- when code wants to consume schema-owned MIR, what should the first useful API be?

This prototype is meant to answer that question before we commit to generating it.

## Review Questions

1. Is this the right first generated API shape?

More concretely:

- should generated MIR expose `ProgramStorage` / `FunctionStorage`-style query types?
- or should the first generated API be flatter, for example free functions or descriptor-driven helpers?

2. Is duplicate-id rejection the right responsibility for this layer?

- should storage construction reject duplicate pool keys?
- or should that be generated validation only?

3. Is the block/edge/term traversal surface the right one?

Specifically:

- `entry_block`
- `block_insts`
- `block_term`
- `block_preds`
- `block_succs`
- `edge_from`
- `edge_to`
- `terminator_edges`

Are these the right "compiler-facing" primitives to generate first?

4. Should this layer return borrowed references, ids, or richer handles?

Right now it returns borrowed generated AST values.

Should the generated storage/query layer instead produce:

- borrowed AST nodes
- typed handles
- lightweight views
- or some hybrid

5. Is this still the right step before arena-backed runtime storage?

Current intended progression:

1. schema expresses pools / refs / orders
2. generated metadata and validation prove the relationships
3. generated storage/query layer exposes typed lookup/traversal
4. only then consider generated arena/runtime storage

Is that sequence sound?

## Current Bias

Our current bias is:

- this prototype should **not** survive as handwritten code
- but it might still be the right generated target

So the real question is not:

- "is handwritten `storage.rs` good?"

It is:

- "if we generate a first MIR storage/query layer, should it look roughly like this?"

## Verification State

Current checks passing for this slice:

- `cargo run -p kajit-foundation-cli -- repr-poc`
- `cargo check -p kajit-reprs`
- `cargo nextest run -p kajit-reprs --test mir_storage --test schema_poc_validation`
- `cargo nextest run -p kajit-foundation`
- `cargo nextest run -p kajit-cli`
- `cargo xtask install`

## Intended Next Step

Unless review says the API shape is wrong, the next step is:

- delete the handwritten storage layer
- generate the equivalent storage/query layer from schema facts and descriptors

That would keep MIR on the "schema-owned all the way down" path instead of accreting a second handwritten runtime.
