# Remove state domain concept from IR

With only one domain remaining after 009, the domain concept is vestigial. Remove it.

## What changes

- `PortKind::State(StateDomainId)` → `PortKind::State` (no ID parameter)
- `Effect::Domain(StateDomainId)` → `Effect::SideEffect` (single effectful category)
- Delete `StateDomain` struct, `StateDomainId` type, `state_domains: Arena<StateDomain>` from `IrFunc`
- Delete `builtin_state_domains()`, `add_state_domain()`, `has_state_domain()`, `state_domain_name()`
- Simplify all RVSDG node builders: no more iterating `state_domains.iter()` to create state ports per domain. Each region threads exactly one state token (or zero for pure functions).
- `RegionBuilder::state_source()` / `set_state_source()` lose the domain parameter
- Update `slot2reg.rs`: `state_count` is always 0 or 1, no more N-domain port arithmetic
- Update `linearize.rs`: state port layout simplifies
- Update `ir_parse.rs`: remove `state_domains { }` block from text format
- Update all golden test snapshots (~40 files)

## Scale

~411 occurrences across ~62 files. But it's mechanical simplification — removing a dimension, not changing semantics.

## Depends on

009 (merged to single domain, so removing the concept is trivially correct).

## Enables

011 (generalize DeserContext — compiler no longer has domain-specific knowledge).
