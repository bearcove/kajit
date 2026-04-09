# Identity And Storage Shapes For IR And CFG-MIR

## Why this note exists

We stopped to look at stable identifiers, storage shape, and deletion semantics because the current schema-poc ASTs are still plain tree values, while RVSDG IR and CFG-MIR are already pushing on a different set of needs:

- stable references
- rewrite-heavy mutation
- diffing and debugging
- validation of dangling references
- eventual schema-owned storage generation

This note captures the current audit, the consultant-style problem statement, and the working design direction.

## The three foundation gaps we surfaced

When reviewing the current pilot work and the `ir.repr.styx` draft, three foundation gaps emerged:

1. richer leaf/support modeling
2. generic token support
3. identity-aware modeling

The second item is the token cleanup work already underway in `kajit-foundation`.

This note is about the third item.

## What these two representations are for

### RVSDG IR

RVSDG is the structured semantic graph layer.

It is used for:

- representing structured control/data flow before linearization
- modeling gamma/theta/apply/regions/results explicitly
- optimization and transformation at a graph/region level
- preserving semantics better than a flat CFG during earlier and middle compiler stages

Operationally, RVSDG is:

- graph/region oriented
- identity-sensitive
- only partly order-sensitive
- rewrite-heavy

### CFG-MIR

CFG-MIR is the lowered executable control-flow graph layer.

It is used for:

- explicit basic blocks and edges
- lowered operations close to backend needs
- register allocation and later backend preparation
- CFG optimizations and cleanup
- debugging and execution-oriented inspection

Operationally, CFG-MIR is:

- block/edge/op oriented
- more sequential than RVSDG
- still rewrite-heavy
- still identity-sensitive in a way that plain vector position does not model well

## Current state audit

### RVSDG today

RVSDG already has a coherent identity model.

The current shape is roughly:

- typed ids over `u32`
- append-only arenas
- ordered membership lists separate from identity

Examples of entity identity include:

- `NodeId`
- `RegionId`
- `ArgId`
- `ResultId`

Important current behavior:

- region membership order lives in vectors on the owning region
- entity identity does not come from membership position
- deletion usually means removing ids from membership/signature lists
- arena entries can remain allocated and become orphaned
- compaction is optional and separate

This model already feels philosophically correct for RVSDG.

### CFG-MIR today

CFG-MIR has typed ids, but it does not yet have a coherent identity model.

The current shape is more like:

- typed ids over `u32`
- backing storage in plain `Vec`s
- validation and cleanup logic that still assume positional identity in practice

Important current behavior:

- blocks, edges, and instructions are often treated as if id and vector position are effectively the same thing
- some passes renumber ids after cleanup
- removal/death semantics are inconsistent:
  - `dead: bool`
  - sentinel invalid ids like `u32::MAX`
  - tolerated orphans

This is a hybrid model:

- typed ids in the API
- positional identity in the implementation
- multiple incompatible deletion conventions

That is the real tension, more than "should we use generations?"

## The key distinction

The most important architectural distinction to extract from this discussion is:

- identity-bearing entities
- ordered slots

Those are not the same thing.

### Identity-bearing entities

These are things that should have stable ids and may be referenced from elsewhere.

Examples:

- RVSDG regions
- RVSDG nodes
- CFG-MIR blocks
- CFG-MIR edges
- CFG-MIR instructions

### Ordered slots

These are things whose meaning is fundamentally positional inside a parent-owned list or signature.

Examples:

- block parameter order
- edge argument order
- operand order
- result order
- region result order
- successor order when semantically ordered

The schema system should eventually model this distinction directly.

## Stable ids: initial position

The current strong default is:

- explicit typed ids
- small integer newtypes
- not hashes
- not vector positions

Examples:

- `NodeId(u32)`
- `RegionId(u32)`
- `BlockId(u32)`
- `EdgeId(u32)`
- `InstId(u32)`
- `VRegId(u32)`

Why not hashes:

- too expensive/noisy
- couple identity to content
- bad fit for rewrites where content changes but identity should survive

Why not vector position:

- ties semantics to storage order
- deletion and reordering become fragile
- stale references silently become wrong references

Why `u32`:

- large enough for practical compiler IR entity counts
- compact
- easy to wrap in typed newtypes

The more important question is not the scalar width of the id, but:

- which things are identities at all
- what scope those ids are unique within
- what the deletion/removal story is

## Generational arenas

Generational arenas came up as a possible tool, but not as the first fix.

What generations buy:

- stale-handle detection under slot reuse
- use-after-delete becomes detectable instead of silently wrong

What they do not buy:

- a coherent identity model on their own
- semantic stability across full rebuilds
- a replacement for separating identity from position

The working conclusion is:

- if ids are monotonic and slots are not reused, generations do not buy much
- if we later want slot reuse plus stale-handle detection, generations become attractive
- for now, the first problem is not "should we use generations?"
- the first problem is "what is the coherent identity/removal/storage model?"

## Consultant-style brief

Below is the distilled consultant-style problem statement that came out of the discussion.

### Short verdict

- RVSDG: the current model is basically the right one.
- CFG-MIR: move it to the same philosophical model: stable typed ids + arena storage + explicit ordered membership/signatures + one coherent death/removal story.
- Generations: probably not the first thing we need.
- The main fix is not "better ids," it is separating identity from position and unifying removal semantics.

### RVSDG recommendation

Keep the model close to what we already have:

- append-only typed-id arenas
- ordered membership lists on regions
- entity identity independent from membership position
- removal by:
  - taking entities out of membership/signature lists
  - optionally marking them dead/orphaned
  - optionally sweeping/compacting later

This is a good fit for a rewrite-heavy semantic graph.

### CFG-MIR recommendation

Redesign CFG-MIR toward:

- stable ids for blocks, edges, instructions
- arena-backed storage for all three
- block membership lists for ordered instruction order
- explicit ordered lists for:
  - block params
  - edge args
  - terminator successors
- one coherent removal model

That implies:

- no "id must equal current vec position" assumptions
- no mixed `dead: bool` vs `u32::MAX` sentinel vs orphan soup
- no silent renumbering as part of ordinary cleanup

### Stable vs positional

Stable identity should apply to:

- RVSDG regions
- RVSDG nodes
- possibly args/results if they are first-class reference targets
- CFG-MIR blocks
- CFG-MIR edges
- CFG-MIR instructions

Positional-only meaning should remain for:

- block param order
- edge arg order
- operand order
- result order
- region result order
- successor order when the meaning is slot-based

### Removal model

Best default:

- stable arena entries
- explicit liveness state
- membership/signature lists updated by rewrites
- occasional explicit rebuild/compaction pass when desired

For CFG-MIR specifically, prefer tombstones first over "orphaned but maybe still meaningful" entries.

Why:

- easier to reason about in passes
- easier to validate
- easier to debug
- clearer than mixed liveness conventions

### Compaction

Compaction should be explicit and opt-in.

Prefer a named operation like:

- `rebuild_cfg_dense() -> RemapTables`

This clarifies:

- what identity preservation means
- what gets remapped
- what clients are allowed to cache

### Validation implications

Once CFG-MIR moves to this model, validation should explicitly check:

- all referenced ids are alive
- membership lists only contain alive ids
- sentinel invalid ids do not exist
- edge endpoints are alive blocks
- instruction order comes from explicit membership, not implicit position

## Working engineering conclusion

This is the current working conclusion.

### RVSDG

- keep arena + membership
- orphan tolerance is acceptable
- compaction remains optional and explicit

### CFG-MIR

- stable opaque ids
- arena-backed entities
- explicit order/signature lists
- tombstones as the first coherent removal model
- no sentinel ids
- no positional identity assumptions
- explicit dense rebuild/remap operation later

### Schema consequence

The schema system should eventually gain a first-class distinction between:

- identity-bearing entity
- ordered parent-owned slot

That distinction is more important than generations.

## Suggested migration order

Migration order matters.

The safest order is:

1. Keep current storage for the moment.
2. Stop assuming `id.index() == vec position` means anything semantically.
3. Make pass APIs treat ids as opaque.
4. Only then swap backing storage and cleanup behavior.
5. Add one explicit rebuild/remap operation early.
6. Revisit generations only if slot reuse becomes desirable.

This reduces the blast radius and gets the cultural change first:

- ids stop meaning position

Then the storage model can follow.

## Implications for schema-owned generation

If the unified schema system eventually owns storage generation as well as AST shape, it should be able to express at least:

- identity-bearing entity kinds
- ordered child/signature collections
- lifecycle policy:
  - stable
  - removable
  - compactable
- optional debug liveness checks

That would let the same overall system describe both:

- plain tree/value layers
- graph/CFG layers with stable identity

without pretending they are the same kind of structure.

## Relation to the current IR draft

The current `notes/unified-ast/pilot/ir.repr.styx` draft is useful partly because it exposed this gap clearly.

It is blocked on foundation work, not just more draft text.

The relevant blocker here is:

- graph layers want explicit identity/storage semantics that the current schema model does not express

This note does not solve that schema design, but it does describe the direction.

## Immediate next steps

1. Finish the generic token-support cleanup in `kajit-foundation`.
2. Review the replacement CFG-MIR pilot draft when it lands.
3. Start a concrete CFG-MIR migration plan around:
   - opaque ids
   - removal semantics
   - validation
4. Only then decide how much of the storage model should become schema-expressible in the next pass.

