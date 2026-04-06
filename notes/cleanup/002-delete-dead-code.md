# Delete dead code outright

**Phase:** 1

Delete these files/modules — they are dead, disabled, or fundamentally broken:

- `loop_phi_elim.rs` (482 lines, disabled)
- `linear_scan.rs` (721 lines, dead code)
- `gvn` inline implementation (misleading name, duplicate infrastructure, incomplete)
- `remat` Phase 2 (behind `if false`, violates SSA)
- The duplicate `DomTree` and `compute_dominators` — use `DominanceInfo` everywhere
