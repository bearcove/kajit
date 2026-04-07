# HIR→IR Lowering Redesign: Consultant Brief

## What kajit is

Kajit is a JIT deserializer for Rust. Given a type's shape (via facet reflection), it generates native machine code at startup that deserializes binary formats (currently postcard) into Rust values. The pipeline:

```
Schema (facet Shape)
  → HIR (kajit-hir)        — statements, locals, loops, if/match
  → IR (kajit-ir)          — RVSDG: theta/gamma nodes, ports, SSA
  → LIR (kajit-lir)        — linearized IR
  → CFG-MIR (kajit-mir)    — control flow graph, VRegs
  → Register Allocation    — regalloc3 (native SSA coloring)
  → Backend                — aarch64 or x86_64 machine code
```

**HIR** is imperative: mutable locals, assignments, loops, memory writes through pointers. It's human-readable and debuggable.

**IR** is RVSDG (Regionalized Value State Dependence Graph): a functional, region-based IR. No mutation — values flow through ports. Side effects are ordered via a state token.

## The RVSDG model (as implemented in kajit-ir)

### Regions and scoping

Every value (port source) belongs to a region. A port source from region A **cannot** be used in region B — the IR verifier rejects `NodeInputOutOfScope`. Values must be explicitly threaded across region boundaries.

### Theta (loop)

```
let outputs = rb.theta(loop_vars, |body_rb| {
    let args = body_rb.region_args(N);  // inner port sources for loop vars
    // ... body ...
    body_rb.set_results(&[predicate, updated_var_0, updated_var_1, ...]);
});
// outputs[0], outputs[1], ... are final values after loop exits
```

- `loop_vars`: port sources from the outer region that enter the loop
- Inside: they appear as new port sources (region args) scoped to the body
- Body results: `[continue_predicate, updated_vars..., state]`
- After: caller gets final values as new outer-region port sources

### Gamma (branch: if/match)

```
let outputs = rb.gamma(predicate, invariants, branch_count, |branch_idx, branch_rb| {
    let args = branch_rb.region_args(N);  // inner port sources
    // ... branch body ...
    branch_rb.set_results(&[result_0, result_1, ...]);
});
```

Each branch is a separate region. Values must be threaded in explicitly.

### State token

Memory operations (`store_to_addr`, `load_from_addr`, `call_effect`, `call_intrinsic`) consume and produce a state token. This token threads through all regions automatically — the `RegionBuilder` manages it. **Memory operations work correctly across region boundaries without any special handling.**

### Slots (the escape hatch we're removing)

Slots are mutable cells: `rb.alloc_slot()`, `rb.write_to_slot(slot, val)`, `rb.read_from_slot(slot)`. They're globally accessible across all regions — they bypass RVSDG scoping. They're NOT standard RVSDG. We still use them for loop control flow (active_slot, continue_slot for break/continue), but we want to stop relying on them for data flow.

## What we had: two lowering paths

### Structural path (deleted, was ~1900 lines)

Used for postcard-generated HIR. The postcard frontend generated a `destination` parameter — a pointer to the output buffer. The structural lowerer:

- Allocated **slots** for every local (params + temporaries)
- Tracked a `dest_local` (the output pointer) specially
- Resolved places through a `ResolvedStructuralPlace` enum:
  - `Destination { ty, byte_offset }` — write to output buffer via out_ptr + offset
  - `Local { ty, storage, slot_offset }` — read/write slots
  - `Indirect { ty, addr }` — write through a dereferenced pointer
- All local state lived in slots, so region boundary crossings were invisible

### Scalar path (~800 lines, now the only path)

Used for handwritten HIR (Vixen language, test functions). No destination pointer — functions return values in registers.

- Tracks locals as **port source vectors** in a `HashMap<LocalId, Vec<PortSource>>`
  - e.g., a `u32` local is `vec![port_source]` (1 word)
  - a struct with 3 fields is `vec![ps0, ps1, ps2]` (3 words)
- Field writes update specific indices in the vector
- Returns values via `rb.set_results(&[...])`
- **Port sources are region-scoped** — this is the fundamental constraint

## What we're doing now

We deleted the structural path and changed the postcard frontend to use the scalar path. The postcard frontend now emits:

- `out` as a regular `Param` of type `u64` (a pointer, but just a number)
- `Store { addr, width, value }` statements for memory writes
- `Place::Deref { base }` for writing through pointers
- `Place::Index { base, index }` for array element writes

We extended the scalar lowerer to handle Store, Deref, and Index — these lower to `store_to_addr` / `load_from_addr` which thread through the state token and work correctly.

## The problem

The postcard-generated HIR has loops that reference outer locals:

```
function "decode_u32" {
  params {
    param l0: Cursor<"input">   // cursor into input buffer
    param l1: u64               // output pointer (was Destination)
    param l2: &mut DeserContext  // error context
  }
  locals {
    let l3: u64   // accumulator
    let l4: u64   // shift amount
    let l5: u64   // byte value
  }
  body {
    init l3 = 0
    init l4 = 0
    loop max_iterations=5 {
      // Read a byte from cursor
      assign l5 = load(l0.pos, w1)           // ← references l0 (param, outer region)
      assign l0.pos = l0.pos + 1             // ← modifies l0 field (outer region)

      // Accumulate
      assign l3 = l3 | ((l5 & 0x7f) << l4)  // ← references l3 (outer local)
      assign l4 = l4 + 7                     // ← references l4 (outer local)

      // Check continuation bit
      if l5 & 0x80 == 0 {
        break
      }
    }
    // Write result to output
    store(l1, w4, l3)                        // ← references l1, l3
    // Update cursor position in ctx
    assign (*l2).pos = l0.pos                // ← deref write through pointer
  }
}
```

Inside the loop body (a theta region), the code references `l0`, `l1`, `l2`, `l3`, `l4`, `l5`. These port sources were created in the root region. The theta body is a different region. **The IR verifier rejects this.**

### What works

- `store_to_addr` / `load_from_addr` — memory operations thread through the state token, which crosses region boundaries automatically. These are fine.
- `active_slot` / `continue_slot` for break/continue — these use slots, which are global. These work.

### What doesn't work

- `self.local_values[l0]` inside a theta body returns a port source from the outer region. Using it triggers `NodeInputOutOfScope`.

## The design space

### Option A: Thread locals through theta/gamma as loop variables

Before entering a theta, collect all live locals' port sources, pass them as loop variables, remap `local_values` inside the body, thread updated values back out.

**Pros:** Correct RVSDG. Clean resulting IR. No slots.
**Cons:** The *lowerer implementation* gets messy — saving/restoring a HashMap around every theta/gamma. The resulting IR is clean but the lowering code has bookkeeping overhead.

Sketch:
```rust
// Before theta:
let (local_ids, flat_sources) = self.collect_live_locals();
let outputs = rb.theta(&flat_sources, |body_rb| {
    let args = body_rb.region_args(flat_sources.len());
    self.remap_locals(&local_ids, &args);  // point local_values at inner port sources
    // ... lower body ...
    let updated = self.collect_local_sources(&local_ids);
    body_rb.set_results(&[predicate, ...updated]);
});
self.remap_locals(&local_ids, &outputs);  // point local_values at theta outputs
```

### Option B: Change HIR to make data flow explicit

Make HIR closer to SSA / functional style so the lowering is trivial. Loops would explicitly declare their carried variables. Places would resolve without HashMap lookups.

**Pros:** Clean lowering, clean IR.
**Cons:** HIR becomes less human-readable. The postcard frontend would need significant rework. May defeat the purpose of HIR as a debuggable source representation.

### Option C: Keep using slots for locals that cross region boundaries

Detect which locals are used inside theta/gamma bodies and allocate slots for them. Read/write via slots instead of port source vectors.

**Pros:** Simple, works immediately.
**Cons:** We said we're done with slots. Slots prevent some RVSDG optimizations (they're opaque mutable state). Slots are a kajit-specific escape hatch, not standard RVSDG.

### Option D: Split locals into register-tracked and memory-backed

Locals that never cross region boundaries stay as port source vectors. Locals that do (params, loop-carried temps) get lowered differently — perhaps as explicit theta loop variables at allocation time rather than via a HashMap.

The lowerer would classify each local upfront:
- **Register-local**: only used within the region where it's defined. Stays as port sources in the HashMap.
- **Loop-carried**: used inside theta bodies. Becomes a theta loop variable — the lowerer manages the threading explicitly.
- **Param**: immutable across the function. Could be threaded as theta invariants (if theta had an invariant mechanism) or as loop variables that happen to not change.

**Pros:** Targeted, no blanket approach needed.
**Cons:** Classification logic adds complexity. Edge cases if a local is sometimes loop-carried and sometimes not.

### Option E: Two-level lowering

Instead of going directly from HIR to RVSDG, introduce an intermediate step that makes region boundary crossings explicit. This intermediate form would annotate each theta/gamma with the set of live-in and live-out variables, making the lowering mechanical.

**Pros:** Clean separation of concerns.
**Cons:** Another intermediate representation to maintain.

## Real-world HIR examples

### Handwritten HIR (works today, no region crossings in loops)

```
hir_module {
  types {}
  callables {}
  functions {
    function "add" {
      params {
        param l0: u64
        param l1: u64
      }
      body {
        return l0 + l1
      }
    }
  }
}
```

### Postcard scalar (u32) — has a loop referencing outer locals

```
hir_module {
  regions { region r0 "input" }
  types {
    type t0 "Cursor" struct { pos: u64 }
    type t1 "DeserContext" struct { error_code: u64, error_offset: u64 }
  }
  callables {}
  functions {
    function "postcard_u32" {
      params {
        param l0: Cursor<r0>      // cursor
        param l1: u64             // out pointer
        param l2: &mut t1         // ctx
      }
      locals {
        let l3: u64               // result accumulator
        let l4: u64               // shift
      }
      body {
        init l3 = 0
        init l4 = 0
        loop max_iterations=5 {
          // Every expression here references l0, l3, l4 from outer scope
          let l5: u64 = load((*l0).pos, w1)
          assign (*l0).pos = (*l0).pos + 1
          assign l3 = l3 | ((l5 & 0x7f) << l4)
          assign l4 = l4 + 7
          if (l5 & 0x80) == 0 { break }
        }
        store(l1, w4, l3)
      }
    }
  }
}
```

### Postcard struct — has nested loops, dynamic indexing, option vtable calls

```
// Simplified — real HIR is much larger
function "postcard_MyStruct" {
  params {
    param l0: Cursor<r0>
    param l1: u64             // out pointer
    param l2: &mut DeserContext
  }
  body {
    // Decode field 0 (u32) — writes to out + 0
    // ... varint loop referencing l0, l2 ...
    store(l1 + 0, w4, result)

    // Decode field 1 (Vec<u8>) — writes to out + 4
    // ... length varint loop ...
    // ... element loop with dynamic indexing ...

    // Decode field 2 (Option<String>)
    // ... discriminant read ...
    // ... option_init_some/option_init_none vtable calls ...
    //     these take addr_of(place) arguments
  }
}
```

## Current state

- Structural lowerer: **deleted** (~1900 lines removed)
- Scalar lowerer: **extended** with Store, Deref, Index, AddrOf, Unary, plus memory-backed field access
- Postcard frontend: changed `Destination` → `Param` with type `u64`
- Compiles cleanly
- Fails at runtime: `NodeInputOutOfScope` when postcard HIR loops reference outer locals
- The `output_size` field (used by differential testing infrastructure to allocate output buffers) is currently 0 — will need a solution later

## Questions for the consultant

1. Is Option A (threading all live locals through theta/gamma) the idiomatic RVSDG approach? Is the bookkeeping overhead acceptable in the lowerer, or does it indicate a design smell?

2. Should HIR change to make loop-carried variables explicit? What would that look like while keeping HIR human-readable?

3. Is there a standard RVSDG technique for handling imperative-style locals that we're missing? (We've looked at the RVSDG literature but kajit's IR is a practical subset, not a textbook implementation.)

4. The slot mechanism (global mutable cells) works but feels like an anti-pattern. Are there RVSDG implementations that use something similar, or is this unique to kajit?

5. Given that memory operations (Store/Load) already work correctly through the state token, is the problem actually smaller than it seems? Could we restructure the lowerer to avoid the HashMap-of-port-sources pattern entirely?
