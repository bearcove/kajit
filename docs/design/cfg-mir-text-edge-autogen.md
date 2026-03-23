# CFG-MIR Text Format: Auto-Generate Edges from Terminators

**Status:** Proposed
**Date:** 2026-03-23
**Author:** Claude (via user request)

## Problem

The current CFG-MIR text format requires manual edge management, which is verbose and error-prone:

```
block b3 succs=[e5, e6]
term t3: branch_if v14 -> e5, fallthrough e6
edge e5: b3 -> b15 []
edge e6: b3 -> b4 []
```

This has several issues:

1. **Manual edge numbering** - Users must track EdgeIds and ensure they're sequential and unique
2. **Triple specification** - Each control flow transition is specified in three places:
   - Block's `succs` field
   - Terminator's edge references
   - Explicit `edge` definitions
3. **Validation burden** - Easy to create inconsistencies (wrong edge.from, mismatched succs order, etc.)
4. **Cognitive overhead** - The CFG graph structure is obscured by bookkeeping

The validation error that motivated this design:
```
func @0 block b3 lists succ e2 but edge.from is b1
```

This error occurs when trying to reuse an edge from a different source block, or when edge topology doesn't match block successor lists.

## Proposed Solution

Allow users to specify control flow **directly** in terminators, with edges auto-generated:

```
block b3
term t3: branch_if v14 -> b15[], fallthrough b4[]
```

The parser would:
1. Generate unique EdgeIds automatically
2. Infer block `succs`/`preds` from terminators
3. Create Edge objects with correct `from`/`to` fields
4. Parse phi args inline: `b12[v53 => v10]`

### Syntax

**Unconditional branch:**
```
term t0: branch b1[]
```

**Conditional branches:**
```
term t1: branch_if v6 -> b15[], fallthrough b2[]
term t2: branch_if_zero v12 -> b12[v53 => v10], fallthrough b3[]
```

**Jump table:**
```
term t3: jump_table v0 [b1[], b2[], b3[]], default b4[]
```

**Error/return:**
```
term t4: error_exit(UnexpectedEof)
term t5: return
```

**Edge args** are specified inline within `[]`:
- `b12[]` - no phi args (empty edge)
- `b12[v53]` - identity phi (v53 → v53)
- `b12[v53 => v10]` - mapped phi (v10 from source becomes v53 in target)
- `b12[v1 => v0, v2]` - multiple args (comma-separated)

### Block Header Simplification

Block headers become simpler:

**Before:**
```
block b12 params=[v53] insts=[...] term=t12 preds=[e4, e8, e12] succs=[e21, e22]
```

**After:**
```
block b12 params=[v53] insts=[...] term=t12
```

Preds/succs are inferred from control flow edges discovered during parsing.

### Complete Example

**Current format (verbose):**
```
cfg_program vregs=10 slots=0 {
  cfg_func @0 f0 entry=b0 {
    data_args: []
    data_results: []

    block b0 params=[] insts=[i0, i1] term=t0 preds=[] succs=[e0]
    block b1 params=[v2] insts=[i2] term=t1 preds=[e0] succs=[e1, e2]
    block b2 params=[] insts=[] term=t2 preds=[e1] succs=[]
    block b3 params=[] insts=[i3] term=t3 preds=[e2] succs=[]

    inst i0: v0:gpr = const(0x1)
    inst i1: v1:gpr = const(0x2)
    inst i2: v3:gpr = CmpEq v2:gpr, v0:gpr
    inst i3: store([0:W4]) v2:gpr

    term t0: branch e0
    term t1: branch_if v3 -> e1, fallthrough e2
    term t2: error_exit(InvalidVarint)
    term t3: return

    edge e0: b0 -> b1 [v2 => v1]
    edge e1: b1 -> b2 []
    edge e2: b1 -> b3 []
  }
}
```

**Proposed format (concise):**
```
cfg_program vregs=10 slots=0 {
  cfg_func @0 f0 entry=b0 {
    data_args: []
    data_results: []

    block b0 params=[] insts=[i0, i1] term=t0
    block b1 params=[v2] insts=[i2] term=t1
    block b2 params=[] insts=[] term=t2
    block b3 params=[] insts=[i3] term=t3

    inst i0: v0:gpr = const(0x1)
    inst i1: v1:gpr = const(0x2)
    inst i2: v3:gpr = CmpEq v2:gpr, v0:gpr
    inst i3: store([0:W4]) v2:gpr

    term t0: branch b1[v2 => v1]
    term t1: branch_if v3 -> b2[], fallthrough b3[]
    term t2: error_exit(InvalidVarint)
    term t3: return
  }
}
```

**Savings:** 8 lines eliminated (edges + block preds/succs), 20% shorter, no edge ID bookkeeping.

## Implementation Strategy

### Phase 1: Parser Changes

1. **New AST representation** - `AstTerminator` variants hold `BlockId` + `Vec<EdgeArg>` instead of `EdgeId`
   ```rust
   enum AstTerminator {
       Branch { target: BlockId, args: Vec<EdgeArg> },
       BranchIf { cond: VReg, taken: (BlockId, Vec<EdgeArg>), fallthrough: (BlockId, Vec<EdgeArg>) },
       // ...
   }
   ```

2. **Inline edge arg parser**
   ```rust
   fn block_with_args<'src>() -> impl Parser<'src, &'src str, (BlockId, Vec<EdgeArg>), Extra<'src>> {
       block_id()
           .then(edge_arg_list())  // reuse existing parser
           .map(|(block, args)| (block, args))
   }
   ```

3. **Update terminator parsers** to accept `block_with_args()` instead of `edge_id()`

### Phase 2: Resolution Pass

After parsing, before validation:

```rust
fn resolve_edges(ast: AstProgram) -> Program {
    let mut edge_counter = 0u32;
    let mut edges = Vec::new();

    for block in &ast.blocks {
        let term = &ast.terms[block.term.index()];

        // For each control flow target in the terminator:
        for (target_block, args) in term.targets() {
            let edge_id = EdgeId::new(edge_counter);
            edge_counter += 1;

            edges.push(Edge {
                id: edge_id,
                from: block.id,
                to: target_block,
                args,
            });

            // Update block succs
            block.succs.push(edge_id);

            // Update target block preds
            target_block.preds.push(edge_id);
        }

        // Convert AST terminator to CFG terminator (BlockId → EdgeId)
        let cfg_term = convert_terminator(term, &edge_mapping);
    }

    // ... rest of resolution
}
```

### Phase 3: Backward Compatibility

**Option A:** Support both syntaxes (parser accepts either format)
- Parse attempt 1: Try new syntax (block IDs in terminators)
- Parse attempt 2: Fall back to old syntax (edge IDs everywhere)
- Detection: If terminator contains `->` followed by `b\d+`, use new syntax

**Option B:** One-way migration (only new syntax)
- Simpler implementation
- Requires regenerating any saved CFG-MIR text
- Display always emits new format

**Recommendation:** Option A for gradual adoption, Option B for long-term simplicity.

### Phase 4: Display Format

Update `fmt_terminator()` to emit block IDs with inline args:

```rust
fn fmt_terminator(f: &mut Formatter, term: &Terminator, edges: &[Edge]) -> fmt::Result {
    match term {
        Terminator::Branch { edge } => {
            let e = &edges[edge.index()];
            write!(f, "branch {}", fmt_block_with_args(e.to, &e.args))
        }
        Terminator::BranchIf { cond, taken, fallthrough } => {
            let t = &edges[taken.index()];
            let ft = &edges[fallthrough.index()];
            write!(f, "branch_if v{} -> {}, fallthrough {}",
                cond.index(),
                fmt_block_with_args(t.to, &t.args),
                fmt_block_with_args(ft.to, &ft.args))
        }
        // ...
    }
}

fn fmt_block_with_args(block: BlockId, args: &[EdgeArg]) -> String {
    format!("b{}{}", block.0, fmt_edge_arg_list_bracketed(args))
}
```

## Benefits

1. **Reduced cognitive load** - Control flow is explicit and local to terminators
2. **Fewer errors** - Can't create invalid edge topology (wrong from/to, mismatched succs)
3. **Easier hand-editing** - No EdgeId bookkeeping, no triple specification
4. **Better diffing** - Changes to control flow are localized
5. **Faster iteration** - Draft optimized IR without edge wrangling

## Non-Goals

- **Not changing the CFG-MIR data structure** - `Edge` still exists, still has `from`/`to` fields
- **Not affecting compilation** - This is purely a text format improvement
- **Not adding new CFG features** - Same expressiveness, better UX

## Alternatives Considered

### A1: Keep current format, improve validation errors
❌ Doesn't address root cause (manual edge management is inherently error-prone)

### A2: Generate edges from block succs instead of terminators
❌ Still requires manual edge numbering and triple specification

### A3: Use a completely different format (e.g., LLVM-style)
❌ Too disruptive, abandons existing tooling and familiarity

## Migration Path

1. **Phase 1:** Implement parser support for new syntax (alongside old syntax)
2. **Phase 2:** Update Display to emit new format
3. **Phase 3:** Deprecate old syntax (warning on parse)
4. **Phase 4:** Remove old syntax support (1-2 releases later)

Timeline: ~2-4 hours implementation, 1-2 releases for deprecation cycle.

## Open Questions

1. **Should block headers keep preds/succs for redundancy checking?**
   - Pro: Catches inconsistencies between user intent and parsed result
   - Con: More verbose, undermines simplification goal
   - **Recommendation:** Make preds/succs optional (for manual verification if desired)

2. **How to handle forward references (blocks defined after terminators)?**
   - Current: Two-pass parsing (collect blocks, then resolve)
   - Proposed: Same approach (AST stage → resolution stage)
   - **No change needed**

3. **Should we support named edges for documentation/debugging?**
   - Example: `term t0: branch b1[] // edge: "happy_path"`
   - Pro: Self-documenting CFGs
   - Con: More complexity, questionable value
   - **Recommendation:** Defer until proven need

## References

- Similar approaches in other IR text formats:
  - **LLVM IR:** `br i1 %cond, label %iftrue, label %iffalse` (block names, no edge IDs)
  - **MLIR:** `cf.br ^bb1(%arg0 : i32)` (block names, inline args)
  - **Cranelift:** `brif v0, block1(v1, v2), block2` (block names, inline args)

All three use direct block references with inline args, not explicit edge objects in the text format.

## Summary

Auto-generating edges from terminators eliminates manual bookkeeping, reduces errors, and makes hand-written CFG-MIR practical for iterative optimization design. The implementation is straightforward (2-4 hours), backward-compatible, and aligns with industry-standard IR text formats.

**Action item:** Implement parser and resolution changes, then use for the hand-optimized u32 varint decoder to validate the design.
