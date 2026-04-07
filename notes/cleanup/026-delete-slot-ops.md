# 017: Replace slot ops with StackAlloc + Store/Load

## Goal

Remove `SlotAddr`, `WriteToSlot`, `ReadFromSlot` from the IR. Replace with
`StackAlloc { size, align }` + `StoreToAddr` / `LoadFromAddr`.

## What changes

### Frontend (`hir_to_ir.rs`)
- `alloc_local_storage` uses `StackAlloc` instead of slot IDs
- `write_to_slot` → `store_to_addr` with stack address
- `read_from_slot` → `load_from_addr` from stack address
- `slot_addr` → just use the StackAlloc result directly

### IR/LIR/CFG-MIR
- Delete `SlotAddr`, `WriteToSlot`, `ReadFromSlot` variants
- Keep `StackAlloc` (or add it if not present)
- Delete `SlotId` type if no longer needed

### Backends
- Delete slot op emission
- `StackAlloc` returns a frame pointer + offset

### Interpreter
- Delete slot op handling

## Notes

Slots are currently all 8 bytes wide. StackAlloc allows arbitrary size/align,
which is more general.
