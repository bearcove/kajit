# Delete dead code outright

**Phase:** 1

## Items to delete

### 1. loop_phi_elim.rs — 481 lines, DISABLED
- **File:** `kajit-mir/src/opt/loop_phi_elim.rs`
- **Status:** Module declared in `opt/mod.rs` line 15, called in `cfg_mir.rs` lines 2044-2051 via `opts.enabled("loop_phi_elim")`
- **Disabled by default:** Line 2205 sets `"loop_phi_elim" => false`
- **Comment:** Line 2043: `// DEPRECATED: old loop-specific phi elimination (disabled by default)`
- **Action:** Delete file, remove mod declaration, remove call site

### 2. Remat Phase 2 — 85 lines behind `if false`
- **File:** `kajit-mir/src/cfg_mir.rs` lines 2615-2699
- **Status:** Explicitly `if false { ... }` guarded
- **Comment:** "DISABLED - this phase created multiple defs which violate SSA"
- **Action:** Delete the `if false` block

### 3. resolve_to_constant_skip_errors — NEVER CALLED
- **File:** `kajit-ir/src/const_fold.rs` lines 441-531
- **Status:** Defined but zero call sites anywhere in the codebase
- **Helper:** `region_has_error_exit()` (lines 507-531) also only used by this
- **Action:** Delete both functions

### 4. resolve_to_const_value duplicate in slot2reg
- **File:** `kajit-ir/src/slot2reg.rs` lines 919-927
- **Status:** Copy-paste of `resolve_to_const` from `dead_theta_ports.rs` lines 362-370
- **Action:** Factor out to shared location or have slot2reg call the one from dead_theta_ports (or better: use `resolve_to_constant` from const_fold.rs with appropriate fallback)

## Items that are NOT dead (cleanup.md was wrong)

### linear_scan.rs — 721 lines, ACTIVELY USED
- **File:** `kajit-mir/src/regalloc3/linear_scan.rs`
- **Status:** Despite `#[allow(dead_code)]` on the struct, it's actively used:
  - `regalloc_engine.rs` lines 117, 124, 132, 139
  - `ssa_coloring.rs`, `spill_rewrite.rs`, `verify.rs`
  - `integration_tests.rs`
- **DO NOT DELETE**

### GVN inline implementation — ACTIVELY USED
- **File:** `kajit-mir/src/cfg_mir.rs` lines 3779-4050
- **Status:** Per-block value numbering (local CSE), not true GVN. Enabled via pass options.
- **Comment:** "For now, we do per-block value numbering."
- **DO NOT DELETE** (but consider renaming from "gvn" to "local_cse" for accuracy)
