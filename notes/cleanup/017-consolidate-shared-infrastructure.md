# Consolidate shared infrastructure

Only the easy parts: domtree consolidation + replace_uses scoping. Skip incremental use-lists unless profiling justifies it.

## 1. UseLists / use-def structure

**File:** `kajit-ir/src/ir_passes.rs` (lines 19-110)

**Problem:** Rebuilt from scratch ~6 times per compilation. Three passes rebuild it:
- `bounds_check_coalescing_pass` — rebuilds in a loop (line 302)
- `dead_code_elimination_pass` — single rebuild (line 861)
- `inline_apply_pass` — single rebuild (line 910)

Each `UseLists::build(func)` scans all regions, nodes, and results. Not maintained incrementally.

**Fix:** Maintain incrementally. Build once, update on mutations. Or at minimum, build once and pass through the pipeline.

## 2. Dominator implementations — consolidate to one

**Good implementation:** `kajit-mir/src/analysis/dominance.rs` (258 lines)
- `DominanceInfo` struct with Cooper-Harvey-Kennedy algorithm
- Used by: `constant_phi_elim.rs`, `simplify_cfg.rs`, `validate_ssa.rs`, `ssa_coloring.rs`

**Duplicates to delete:**

### compute_dominators + dominates (cfg_mir.rs lines 2980-3067)
- 86 lines, returns `HashMap<BlockId, Option<BlockId>>`
- Used 1 time at line 2765 (in `global_copy_propagation`)
- **Replace** with `DominanceInfo::compute()`

### DomTree (cfg_mir.rs lines 3784-3929)
- 146 lines, separate implementation
- Used 3 times at lines 4976, 5047, 5142 — all in test code within cfg_mir.rs
- **Replace** test uses with `DominanceInfo`, delete struct

## 3. Constant resolver — see 005-clean-up-const-fold.md

Already covered there. Three resolvers → keep primary `resolve_to_constant`, factor out the simple one, delete the unused one.

## 4. replace_uses — two implementations

### Global: `replace_output_use` (ir_passes.rs line 1092)
- Part of `UseLists` system
- Scans and replaces across ALL regions and results

### Region-scoped: `replace_uses_in_region` (slot2reg.rs line 1113)
- Starts at given region, recurses into Gamma/Theta children
- Correctly scoped

**Fix:** One `replace_uses` with configurable scope (region or global). The region-scoped version is the right default — global scan should be opt-in.

## Also noted

`DefUseInfo` is defined in `kajit-mir/src/analysis/defuse.rs` but not actively used by any passes — potential foundation for incremental use-def tracking.
