# Rewrite DCE pass

**Phase:** 1

## Current state

Two DCE implementations at different IR levels:

### CFG-MIR DCE: `eliminate_dead_block_params`
- **File:** `kajit-mir/src/opt/dce.rs` (156 lines)
- **Algorithm:** Iterative fixed-point elimination of dead block parameters + edge arguments
- **O(n²) evidence (lines 60-154):**
  - Outer `while changed` loop (unbounded iterations)
  - For each dead param found, **rescans the entire function** to recompute the `used` set (lines 116-150)
  - Chain of dead dependencies (A→B→C) requires multiple iterations
- **Invoked 3 times** in `lower_and_optimize()`:
  - Line 2077: Primary DCE
  - Line 2105: After `control_thread`
  - Line 2116: After `simplify_cfg`

### RVSDG DCE: `eliminate_dead_theta_ports`
- **File:** `kajit-ir/src/dead_theta_ports.rs` (510 lines)
- **Algorithm:** Remove loop-carried theta ports that always feed constants
- **Capped at 3 ports** (line 274) — see 006-fix-dead-theta-ports-cap.md
- **Skips theta bodies > 50 nodes** (line 63) to avoid O(n²) scanning

## How to fix

The CFG-MIR DCE is the one to rewrite. Textbook worklist algorithm:
1. Build initial use set in one pass
2. Mark all unused block params
3. Remove dead params + edge args, updating use set incrementally
4. No need for `while changed` — a single reverse-postorder pass with a worklist suffices

Each pass should be a pure function: `&mut Function` in, modifications in-place, no global state.

## Key files
- `kajit-mir/src/opt/dce.rs` — rewrite target
- `kajit-mir/src/opt/mod.rs` — module listing
- `kajit-mir/src/cfg_mir.rs` — 3 invocation sites in `lower_and_optimize()`
