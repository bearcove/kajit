# Strict IR/Data Model (Ideal End State)

## Current Toolchain (Today, strict branch)

Pipeline stages and representation names:

`RVSDG => LIR => CFG-MIR => Typed AP => Emission => Runtime`

1. RVSDG
- Type: `IrFunc`
- Built by: `build_decoder_ir(...)`
- Optimized by: default RVSDG passes (`all_opts` / `pass.*`)

2. LIR
- Type: `LinearIr`
- Built by: `linearize(...)`

3. CFG-MIR
- Type: `cfg_mir::Program`
- Built by: `cfg_mir::lower_linear_ir(&LinearIr)`
- Shape: explicit CFG with typed `BlockId` / `EdgeId` / `OpId`

4. AP (Allocated Program)
- Type: `AllocatedCfgProgram`
- Built by:
  - `allocate_cfg_program(&cfg_mir::Program)` when `+regalloc`
  - synthetic no-regalloc allocation when `-regalloc`
- Carries operand allocations + progpoint edits + edge edits keyed by typed IDs

5. Emission
- Type: `LinearBackendResult` (contains finalized aarch64 code buffer + entry + source map)
- Built by:
  - `backends::aarch64::compile(&cfg_mir::Program, &AllocatedCfgProgram, apply_regalloc_edits)`
  - `backends::x86_64::compile(&cfg_mir::Program, &AllocatedCfgProgram)`

6. Runtime
- Type: `CompiledDecoder`
- Built by: compiler wrapper that converts emission into callable function pointer and optional JIT debug registration

Notes:
- `compile_linear_ir_decoder` now goes through CFG-MIR + typed regalloc artifacts.
- RA-program compile entrypoints are intentionally disabled on this branch (strict cutover).

## Where Canonical CFG MIR Fits

It sits immediately after `LinearIr` and replaces the current RA-MIR-as-identity
layer for all post-linearization stages.

### Before (legacy)

```text
[RVSDG: IrFunc]
        |
        v
[LIR: LinearIr]
        |
        v
[RA-MIR: RaProgram]
  (CFG + linear-op-index identity baggage)
        |
        v
[AP: AllocatedProgram]
  (edits/edge-edits keyed by linear indices)
        |
        v
[aarch64 emission]
        |
        v
[Runtime: CompiledDecoder]
```

### After (ideal)

```text
[RVSDG: IrFunc]
        |
        v
[LIR: LinearIr]
        |
        v
[Canonical CFG MIR: CfgMir]
  (BlockId/EdgeId/OpId are semantic identity)
        | \
        |  \---> [Derived Schedule (ephemeral)]
        |         (OpId <-> dense index for regalloc internals only)
        v
[Regalloc Result]
  (allocs/edits keyed by ProgPoint + OpId + EdgeId)
        |
        v
[aarch64 emission]
  (consumes CfgMir + typed regalloc result)
        |
        v
[Runtime: CompiledDecoder]
```

### What This Replaces

1. Replace `RaProgram` as the canonical post-linearization identity model with `CfgMir`.
2. Replace linear-index-keyed regalloc artifacts in `AllocatedProgram` with typed keys (`OpId`, `EdgeId`, `ProgPoint`).
3. Keep dense instruction numbering only as a derived temporary schedule, never as persisted semantic identity.

## Direct-Cutover Milestones (No Compat Shims)

This is the "go straight to goal" plan. Each milestone is expected to break the build
until the full sequence lands.

Progress (strict-main-cleanup branch):
- Completed: 1 through 10 (canonical CFG-MIR introduced, regalloc + simulator + differential + debugger ported, legacy RA-MIR plumbing removed, debug/dump + DWARF mapping updated to CFG-MIR op identity, and end-to-end stabilization validated on both aarch64 and x86_64 test matrices).

1. Introduce canonical post-linearization IR
- Add `CfgMir` types (`Function/Block/Edge/Inst/Term`) with typed IDs (`OpId`, `EdgeId`, ...).
- Terminator is required and explicit in every block.
- Done when: old/new code both compile at type level with `CfgMir` available.

2. Make linearization target `CfgMir` directly
- Replace `LinearIr -> RaProgram` lowering with `LinearIr -> CfgMir` construction.
- Encode edge args and successor identity as `EdgeId`, not positional `(from_linear, succ_index)`.
- Done when: compiler path can produce `CfgMir` for real corpus cases.

3. Replace regalloc adapter input
- Rewrite regalloc adapter to consume `CfgMir` and generate regalloc2 function view from typed IDs.
- Remove dependency on `term_linear_op_index` semantics.
- Done when: allocation succeeds on existing regalloc unit tests.

4. Replace regalloc result model
- Replace `AllocatedProgram` linear-index keyed maps with typed keys:
  - point edits: `Before/After(OpId)`
  - edge edits: `Edge(EdgeId)`
- Keep any dense index mapping internal/private to adapter only.
- Done when: verifier and differential checker run only on typed keys.

5. Rewrite post-regalloc simulator + checker on typed program points
- Port simulator and differential harness to `CfgMir + typed alloc result`.
- Remove linear-op-index lookup maps from simulation correctness logic.
- Done when: differential harness tests pass using typed-key path only.

6. Rewrite aarch64 backend consume path
- Backend emission consumes `CfgMir` + typed regalloc result directly.
- Branch/edge handling uses `EdgeId` identity only.
- Done when: aarch64 backend builds without any linear-index keyed edit lookup.

7. Rewrite x86_64 backend consume path
- Same contract as aarch64 (`CfgMir` + typed result).
- Keep backend parity and shared invariants.
- Done when: both backends compile and pass backend micro tests.

8. Delete legacy RA-MIR + linear-index plumbing
- Remove `RaProgram`, `term_linear_op_index`, and all edge-edit keying by `(linear, succ_index)`.
- Remove legacy dump/report paths that depend on linear-op semantic identity.
- Done when: no production code references those legacy fields/types.

9. Rebuild debugging/dumps on new identity model
- Stage dumps print block/edge/op IDs; optional schedule index shown as derived metadata only.
- LLDB mapping uses `OpId` primary identity with derived line ordering.
- Done when: docs and tooling reflect new model and produce usable dumps.

10. End-to-end stabilization pass
- Fix all failing corpus/micro tests on new path only.
- Re-enable/refresh differential checks and verifier assertions as hard gates.
- Done when: target test matrix is green on both aarch64 and x86_64.

## Context

This note describes the "infinite refactor budget" design: make control-flow and program
points first-class, and make linear indices a derived artifact instead of persisted semantics.

Current pain points this design removes:

- Optional/sentinel terminator indices (`Option<usize>`, fake indices).
- Ambiguity between "real linear op index" and "adapter-generated synthetic index".
- Edge-edit keying by unstable positional indices.

## Design Goals

- Single canonical CFG-shaped MIR for post-linearization stages.
- Explicit required terminator per block.
- Stable typed IDs for instructions, terminators, blocks, edges.
- Regalloc + backend APIs keyed by typed program points, not raw `usize`.
- Any dense linear order is derived and disposable.

## Canonical Data Model

```rust
type FunctionId = u32;
type BlockId = u32;
type EdgeId = u32;
type InstId = u32;
type TermId = u32;
type OpId = u32; // union id over Inst + Term

struct Program {
    functions: Vec<FunctionId>,
    func_data: IndexVec<FunctionId, Function>,
}

struct Function {
    entry: BlockId,
    data_args: Vec<VReg>,
    data_results: Vec<VReg>,
    blocks: IndexVec<BlockId, Block>,
    edges: IndexVec<EdgeId, Edge>,
    insts: IndexVec<InstId, Inst>,
    terms: IndexVec<TermId, Terminator>,
}

struct Block {
    params: Vec<VReg>,
    insts: Vec<InstId>, // ordered non-terminators
    term: TermId,       // always present, exactly one
    preds: Vec<EdgeId>,
    succs: Vec<EdgeId>,
}

struct Edge {
    from: BlockId,
    to: BlockId,
    args: Vec<EdgeArg>, // block-param transfer pairs
}

struct Inst {
    op: LinearOpLike,
    operands: Vec<RaOperandLike>,
    clobbers: ClobberSet,
}

enum Terminator {
    Return,
    ErrorExit { code: ErrorCode },
    Branch { edge: EdgeId },
    BranchIf { cond: VReg, taken: EdgeId, fallthrough: EdgeId },
    BranchIfZero { cond: VReg, taken: EdgeId, fallthrough: EdgeId },
    JumpTable { predicate: VReg, targets: Vec<EdgeId>, default: EdgeId },
}
```

Notes:

- Edge identity is explicit and stable (`EdgeId`), not implied by `(from_linear, succ_index)`.
- Terminators are never optional and never synthesized by index tricks.

## Program Point Model

Regalloc, edits, verification, and backend hooks use typed program points:

```rust
enum ProgPoint {
    Before(OpId),
    After(OpId),
    Edge(EdgeId), // edge transfer point
}
```

This removes all ambiguity about "where" a move happens.

## Derived Schedule (Not Canonical)

When a dense ordering is needed (regalloc internals, debug dumps, emission bookkeeping),
compute a temporary schedule:

```rust
struct Schedule {
    op_order: Vec<OpId>,                     // dense execution order
    op_to_index: SecondaryMap<OpId, u32>,    // reverse map
    block_ranges: IndexVec<BlockId, Range<u32>>,
}
```

Rules:

- The schedule is derived from `Function`, never source-of-truth.
- Rebuild after transforms; do not persist `usize` indices in canonical IR.
- If regalloc wants dense instruction numbering, translate at the boundary.

## Hard Invariants

1. Every block has exactly one terminator (`block.term` is required).
2. Terminator has no following normal instruction in block order.
3. `preds`/`succs` match edge table exactly (`edges[e].from/to`).
4. Terminator successor references are edge IDs that belong to that block's `succs`.
5. Edge args cardinality/types match destination block params.
6. Entry block has no incoming edges.
7. No critical-edge "identity loss": edge split creates a new `BlockId` + `EdgeId`s, preserving references.

## Regalloc Contract in This Model

- Input: canonical `Function` (+ optional derived `Schedule` adapter).
- Output keyed by `ProgPoint` + `OpId` + `EdgeId`, for example:
  - allocs for operand slots on each `OpId`,
  - point edits at `Before/After(OpId)`,
  - edge edits at `Edge(EdgeId)`.

No API should require `term_linear_op_index` or raw linear indices.

## Backend Contract in This Model

Backends consume canonical CFG:

- Iterate blocks, emit all `insts`, then emit `term`.
- Apply point edits via `ProgPoint::Before/After(OpId)`.
- Apply edge edits via `ProgPoint::Edge(EdgeId)` when lowering branch edges.

Debug/source mapping uses `OpId` as the primary key, with optional schedule index as
derived metadata only.

## Why This Is Better

- No optional/sentinel terminator index state.
- No fake "synthetic linear op" values.
- Strongly typed identities prevent accidental aliasing/collision bugs.
- Easier verifier logic: compare structural identities, not guessed indices.
- Easier backend parity: x86_64 and aarch64 consume the same canonical shape.

## Cost

- Broad churn across linearization, regalloc adapter layer, verifiers, dumps, and tests.
- Need conversion shims while migrating old `linear_op_index` call sites.
- Snapshot format changes (explicit block/edge identities in text dumps).

## Compatibility Strategy (If We Ever Build It)

1. Introduce canonical `CfgMir` types in parallel.
2. Add adapters:
   - old -> new for pipeline entry,
   - new -> old only where unavoidable short-term.
3. Flip regalloc verifier/checker to `OpId`/`EdgeId` keys first.
4. Flip backends to canonical CFG API.
5. Remove legacy linear-index keyed plumbing.

End state: linear indices exist only in ephemeral schedule views, never as semantic identity.
