# MIR Compiler Storage Consultant Brief

This document captures the current storage/API discussion for Kajit MIR after a hard pivot.

The earlier direction was:

- put `Pool`/`Order`-style storage concepts into the generated MIR AST
- then add query/index layers on top

The current conclusion is:

- that was the wrong abstraction boundary
- AST and compiler storage got conflated
- compiler-owned MIR storage should live outside the AST
- optimization passes should not see `Vec`, `IndexMap`, arenas, or container mechanics

This packet is intended for a consultant review of that pivot.

## Short Version

We now think the right architecture is:

- AST:
  - structural only
  - parseable / printable / schema-owned
- compiler storage:
  - owns MIR arenas / collections
  - owns handles / identity
  - enforces graph mutation rules
- passes:
  - operate on graph APIs
  - not on raw containers

The key realization was:

- if we control the representation, then building an AST and then a separate handwritten indexing layer on top is wasteful
- but pushing storage layout into the AST is also the wrong move

So the new target is:

- schema-owned MIR syntax and structural model
- compiler-owned MIR graph storage with typed handles and explicit mutation operations

## Why We Pivoted

We started by introducing schema concepts like:

- `@pool`
- `@order`
- keyed pools
- `@ref_to(...)`

That led to a generated MIR AST shaped roughly like:

```rust
pub struct Function {
    pub blocks: Pool<Block>,
    pub edges: Pool<Edge>,
    pub insts: Pool<Inst>,
    pub terms: Pool<Terminator>,
    pub entry: BlockId,
}
```

Then we wrote a handwritten prototype layer like:

```rust
pub struct FunctionStorage<'a> {
    function: &'a Function,
    blocks_by_id: BTreeMap<BlockId, &'a Block>,
    edges_by_id: BTreeMap<EdgeId, &'a Edge>,
    insts_by_id: BTreeMap<InstId, &'a Inst>,
    terms_by_id: BTreeMap<TermId, &'a Terminator>,
}
```

That immediately exposed two problems:

1. It was wasteful.

We already had all the data in `Function`, then we walked it again to build sidecar indexes.

2. It was the wrong level entirely.

If storage/queryability matters, either:

- it belongs in the real representation

or

- the real storage should not be the AST in the first place

The second answer is the one we now believe.

## The New Position

The AST should not be the compiler’s working storage.

The AST should remain:

- structural
- schema-owned
- round-trippable
- good for text, docs, formatting, semantic tokens, hover, validation, debugging

The compiler should own a separate MIR graph/storage layer.

That storage layer is where we should put:

- identity
- handles
- fast lookup
- mutation
- rewrite operations
- consistency enforcement

That means:

- no AST-owned arenas
- no AST-owned keyed maps as the primary compiler truth
- no optimization passes poking `Vec`s or `IndexMap`s directly

## Legacy MIR: What It Got Right and Wrong

The old handwritten MIR is in:

- [/Users/amos/bearcove/kajit/crates-kajit/kajit-reprs-legacy/src/mir/mod.rs](/Users/amos/bearcove/kajit/crates-kajit/kajit-reprs-legacy/src/mir/mod.rs)

Its core shape is:

```rust
pub struct Function {
    pub blocks: Vec<Block>,
    pub edges: Vec<Edge>,
    pub insts: Vec<Inst>,
    pub terms: Vec<Terminator>,
    pub entry: BlockId,
    // ...
}
```

with lookups like:

```rust
pub fn block(&self, id: BlockId) -> Option<&Block> {
    self.blocks.get(id.index())
}
```

and validation that insists ids match positions.

What legacy MIR got right:

- compiler-owned storage
- fast direct access
- no AST/storage split

What it got wrong:

- ids are positional
- deletion / mutation caused bugs
- tombstones were introduced to keep indices stable
- renumbering and sentinel values (`u32::MAX`) appeared
- storage details leak everywhere into passes

So the lesson is not “legacy MIR was bad because it was compiler-owned.”

The lesson is:

- compiler-owned storage was the right level
- positional identity was the wrong implementation strategy

## Proposed Storage Direction

We want compiler-owned MIR storage with:

- opaque typed handles
- append-only allocation
- no public `u32` ids that can be forged
- no positional identity assumptions in passes

The user’s current preferred model is:

- global arenas owned by the compiler
- but handles can still be tagged with the function they belong to

That means a handle is conceptually not just:

```rust
BlockHandle(123)
```

but something more like:

```rust
BlockHandle {
    func: FunctionHandle,
    raw: BlockSlot,
}
```

or an opaque equivalent.

The point is:

- storage can be global
- but function-local identity/ownership is still encoded in the handle

This is important because the real mutation bug we care about is not “forged handle.”
Under opaque append-only handles, that goes away.

The real bug is:

- cross-function contamination during rewrites

Example:

- someone copies a block from function A into function B
- but leaves it referring to instructions / edges / terminators that belong to A’s logical graph region

Tagged handles and restricted mutation APIs are meant to prevent exactly that.

## What Passes Should and Should Not See

Optimization passes should not know whether storage is:

- `Vec`
- `IndexMap`
- arena
- some custom slab

They should operate against graph operations and typed handles.

So instead of:

- mutating `func.blocks`
- mutating `block.succs`
- inserting into vectors directly

they should use APIs like:

- create block
- duplicate block
- append instruction
- replace terminator
- create edge
- retarget edge
- split block
- inline callee

The underlying storage representation can change later if those operations remain stable.

## Draft API Direction

We are currently thinking in terms like:

```rust
pub struct MirGraph {
    // compiler-owned storage
}

pub struct FunctionHandle(/* opaque */);
pub struct BlockHandle(/* opaque, function-tagged */);
pub struct InstHandle(/* opaque, function-tagged */);
pub struct EdgeHandle(/* opaque, function-tagged */);
pub struct TermHandle(/* opaque, function-tagged */);
```

Read-only access:

```rust
impl MirGraph {
    pub fn function(&self, f: FunctionHandle) -> FunctionView<'_>;
}

pub struct FunctionView<'a> { /* ... */ }

impl<'a> FunctionView<'a> {
    pub fn entry(&self) -> BlockHandle;
    pub fn blocks(&self) -> impl Iterator<Item = BlockHandle> + 'a;
    pub fn block(&self, b: BlockHandle) -> BlockView<'a>;
    pub fn edge(&self, e: EdgeHandle) -> EdgeView<'a>;
    pub fn inst(&self, i: InstHandle) -> InstView<'a>;
    pub fn term(&self, t: TermHandle) -> TermView<'a>;
}
```

Mutation/editing:

```rust
impl MirGraph {
    pub fn function_mut(&mut self, f: FunctionHandle) -> FunctionEditor<'_>;
}

pub struct FunctionEditor<'a> { /* ... */ }

impl<'a> FunctionEditor<'a> {
    pub fn create_block(&mut self) -> BlockHandle;
    pub fn append_inst(&mut self, block: BlockHandle, inst: InstData) -> InstHandle;
    pub fn set_terminator(&mut self, block: BlockHandle, term: TermData) -> TermHandle;
    pub fn create_edge(
        &mut self,
        from: BlockHandle,
        to: BlockHandle,
        args: Vec<EdgeArgData>,
    ) -> EdgeHandle;
    pub fn set_entry(&mut self, block: BlockHandle);
    pub fn duplicate_block(&mut self, block: BlockHandle) -> BlockHandle;
    pub fn split_block(&mut self, block: BlockHandle, at: InstHandle) -> BlockHandle;
    pub fn retarget_edge(&mut self, edge: EdgeHandle, to: BlockHandle);
}
```

Cross-function operations must be explicit:

```rust
pub fn inline_call(
    &mut self,
    call_block: BlockHandle,
    call_inst: InstHandle,
    callee: FunctionHandle,
) -> InlineResult
```

The point is:

- ordinary APIs should reject accidental cross-function mixing
- explicit cross-function transforms are responsible for cloning/remapping

## What Invariants We Actually Care About

We explicitly do **not** want to list things as invariants if the representation already guarantees them.

For example:

- if a block structurally has exactly one terminator, that is not a validation rule
- if handles are opaque and append-only, “forged/stale handle” is not the main problem

The real graph invariants we care about are things passes can actually violate:

- edge arg count matches destination block param count
- edge args align positionally with destination block params
- entry block is set
- cross-function contamination does not happen accidentally
- SSA invariants
- pass-specific semantic preconditions

We also identified that:

- successors are cheap to derive from a block’s terminator
- predecessors are the expensive reverse relation

Given expected function sizes, the current bias is:

- do not cache predecessors until it is a real performance issue
- derive what we can
- avoid duplicated graph structure unless proven necessary

## Relation To Storage Implementation

The user explicitly pointed out that:

- a wrapper over `IndexMap` plus `Vec` can already be a major step up from the old positional-id design

We agree with that.

But the important conclusion is:

- those storage details should not leak to passes

So even if the internal implementation is:

- append-only arenas
- or `IndexMap`-like keyed storage
- or some hybrid

the pass API should remain at the graph-operation level.

## Why We Are Asking For Review

We want an outside opinion on whether this pivot is the right one:

- away from AST-owned storage concepts
- toward compiler-owned graph storage and opaque handles

Specific questions:

1. Is the AST/compiler-storage split described here the right one for MIR?

2. Is global compiler-owned storage with function-tagged handles a good direction?

3. Is the draft pass-facing API the right level of abstraction?

4. Which mutation operations should definitely exist first?

5. Which invariants should be enforced by construction, and which ones should remain validation checks?

6. Is there a cleaner way to prevent cross-function contamination during rewrites than the tagged-handle plus explicit-transform approach described here?

## Current Bottom Line

Our current bottom line is:

- putting pools/arenas into the AST was a mistake
- handwritten index/query facades over the AST were also a mistake
- compiler-owned storage is the right level
- opaque handles and centralized graph mutation APIs are the real next design target
- the old MIR was right about the level, wrong about positional identity

That is the pivot we want reviewed.
