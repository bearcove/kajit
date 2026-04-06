# Merge OUTPUT_STATE_DOMAIN into MEMORY_STATE_DOMAIN

After 008 deletes the cursor domain, two domains remain: output and memory. Both protect memory operations (output writes are just stores to a pointer). Merge them.

## What changes

- All ops currently tagged `Effect::Domain(OUTPUT_STATE_DOMAIN)` become `Effect::Domain(MEMORY_STATE_DOMAIN)`
- Affected ops: `WriteToField`, `WriteToFieldRange`, `SetOutPtr`, `WriteToOutput`
- Delete `OUTPUT_STATE_DOMAIN`, `OUTPUT_STATE_DOMAIN_NAME`, `OUTPUT_EFFECT` constants
- `builtin_state_domains()` creates only one domain: "memory"
- Update all IR text tests that reference `%os` (output state) args

## Semantic consequence

Output writes and memory loads/stores are now fully serialized (single state chain). This is conservative but correct. Future alias analysis (from HIR value semantics) can reintroduce fine-grained ordering.

## Depends on

008 (cursor domain gone, only output + memory remain).

## Enables

010 (remove domain concept entirely — only one domain left, so the concept is trivial).
