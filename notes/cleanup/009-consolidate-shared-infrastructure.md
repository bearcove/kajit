# Consolidate shared infrastructure

**Phase:** 3 (after passes work individually)

1. **One `UseLists` / use-def structure** — maintained incrementally, not rebuilt from scratch per pass
2. **One dominator implementation** — `DominanceInfo`, shared by all passes (delete duplicate `DomTree` and `compute_dominators`)
3. **One constant resolver** — with configurable conservatism (control vs data role), replacing the current three
4. **One `replace_uses` that's scoped** — to a region, not a global scan
