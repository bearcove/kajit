# 011-2: General-purpose IR — remove deserializer ops

## Goal

The IR is a general-purpose compiler IR. No ops that assume "output pointer",
"cursor", or "context". The frontend lowers format-specific concepts to
primitives before they reach the IR.

## Current deserializer ops to remove

### Output ops → Store/Load with explicit base vreg
- `WriteToField { offset, width }` → `Store { width }` with `Add(out_ptr, Const(offset))` as address
- `ReadFromField { offset, width }` → `Load { width }` with `Add(out_ptr, Const(offset))` as address
- `SaveOutPtr` → just use the out_ptr vreg directly (it's a data value)
- `SetOutPtr` → just assign to the out_ptr vreg (or `Add` for offset)

### Slot ops → StackAlloc + Store/Load
- `SlotAddr { slot, num_slots }` → `StackAlloc { size, align }` returns address vreg
- `WriteToSlot { slot }` → `Store` to stack-allocated address
- `ReadFromSlot { slot }` → `Load` from stack-allocated address

### Call ops → one unified Call
- `CallIntrinsic` (implicit ctx, implicit error check) → `Call` with ctx as explicit arg
- `CallPure` → `Call { pure: true }`
- `CallEffect` → `Call { pure: false }`

### Other
- `ZigzagDecode { wide }` → frontend emits `Shr` + `Xor` (+ `SignExtend` if wide)
- `ErrorExit { code }` → frontend emits `Store(ctx + ERROR_OFFSET, code)` + return

## State domains

Currently: `StateOutput`, `StateCursor`, `StateMemory` — three domains.
After: just `StateMemory`. Output pointer and cursor are regular data values,
not state-threaded magic.

## Backend impact

- Delete `output_enc`, `ctx_enc`, `cursor_enc`, `end_enc` from EmitCtx (both arches)
- `WriteToField` emission → becomes regular `Store` emission (already exists)
- `CallIntrinsic` emission → becomes regular `Call` emission, no implicit ctx arg or error check
- `ErrorExit` emission → gone (it's just Store + Return now)
- Delete `emit_decoder_prologue/epilogue`, `begin_func_with_config/end_func_with_config`
- All functions use the scalar prologue/epilogue (already partially done)

## Regalloc impact

- Delete `extra_excluded_regs` — no more pinned registers
- Delete `is_scalar` everywhere (already partially done in wip commit)
- data_args pre-coloring stays (standard ABI arg mapping)

## Frontend impact (`hir_to_ir.rs`)

- `out_ptr` becomes `data_args[0]` — a real vreg passed by caller
- `ctx_ptr` becomes `data_args[1]` — a real vreg passed by caller
- Cursor state pointer stays as `data_args[2]` where applicable
- Frontend emits `Store`/`Load` instead of `WriteToField`/`ReadFromField`
- Frontend emits `Call` with explicit ctx arg instead of `CallIntrinsic`
- Frontend emits explicit error check after calls (→ 011-3)

## Interpreter impact

- No hardcoded `out_ptr`/`ctx_ptr` — they're just vreg values from data_args
- `Store`/`Load` already implemented
- `Call` replaces three call ops

## Approach

This is mostly deletion. The primitives (`Store`, `Load`, `Call`) already exist.
The work is: make the frontend use them instead of the shortcut ops, then delete
the shortcut ops and their backend/interpreter handling.

### Order
1. Make out_ptr a real vreg: frontend emits Store/Load, delete WriteToField/ReadFromField/SaveOutPtr/SetOutPtr
2. Make ctx_ptr a real vreg: CallIntrinsic gets explicit ctx arg (prep for 011-3)
3. Delete ZigzagDecode — frontend emits shift+xor
4. Unify three Call ops into one
5. Replace SlotAddr/WriteToSlot/ReadFromSlot with StackAlloc+Store/Load
6. Kill is_scalar, unify prologues (the easy part, once the above is done)

Each step should leave tests passing.

## Depends on

011-1c (split backends done) ✓

## Enables

011-3 (explicit error check — natural consequence of ctx_ptr as vreg).
