# Bug: string literal multi-value return produces swapped (ptr, len)

## Status

Open. The data blob relocation infrastructure works end-to-end, but
functions returning a string literal as `(ptr, len)` get the two values
swapped in the return registers.

## Reproduction

```rust
// fn greeting(_dummy: u64) -> Str { return "hello"; }
let (ptr, len) = unsafe { greeting(0) };
// Expected: ptr = <address of "hello">, len = 5
// Actual:   ptr = 5, len = 8  (or some small number that isn't the address)
```

Tests: `vixen_typed_function_string_literal_return` and
`vixen_typed_function_string_literal_conditional` in
`kajit/src/compiler/tests/hir_to_ir.rs`.

## What works

- Single-value scalar returns (u64): `add(a, b) -> u64` ✓
- Multi-value struct returns without DataAddr: `str_slice(s, start, end) -> Str` ✓
- Multi-value conditional struct returns: `pick(s, flag) -> Str` ✓
- DataAddr relocation patching (the address IS being patched correctly)
- The parallel move epilogue for struct returns without DataAddr

## What fails

When the return value is a string literal lowered as
`[DataAddr(blob_id), Const(len)]`, the epilogue puts the values in the
wrong registers. Specifically `x0` gets `len` and `x1` gets `ptr`
(swapped), or `x1` gets a value like `8` that looks like the aligned
data section size rather than the string length.

## Observations

1. The value `8` appearing as `len` when the string is `"hello"` (5 bytes)
   or `"one"` (3 bytes) is suspicious. `8` is the 8-byte-aligned size of
   these short strings in the data section. This might mean a data section
   offset or padding value is leaking into a return register.

2. The bug does NOT occur for multi-value returns that use only `Const`
   and register-to-register data flow (e.g., the `str_slice` and
   `str_conditional_return` tests pass). It only occurs when one of the
   return values comes from a `DataAddr` op.

3. The `DataAddr` emission uses `emit_load_u64_fixed` (always 4
   instructions = 16 bytes) instead of the variable-length
   `emit_load_u64` used by `Const`. This difference in instruction count
   might confuse one of the backend's fusion/skip analyses.

## Likely causes (ranked)

### 1. `elim_imm` or `fused_skip` incorrectly skipping DataAddr

The backend has several optimization passes that skip instructions whose
results are folded into later operations:

- `compute_fusable_addr_offsets`: fuses `Add(base, const_offset)` into a
  single `[base, #offset]` addressing mode, skipping the Add and its
  Const operand.
- `compute_fusable_cmps`: fuses `CmpEq` with a following `BranchIf`.
- The `fused_skip` set tracks vregs whose defining instructions should
  not be emitted.

If `DataAddr` is incorrectly classified as a `Const` by one of these
analyses (because it appears in `const_values` or similar), its 16-byte
movz/movk sequence might be skipped. The return register would then
contain whatever was previously in that register.

**Where to look:**
- `regalloc3_backend.rs`: `compute_fusable_addr_offsets`,
  `compute_fusable_cmps`, `compute_fusable_bit_tests`
- The `const_values` HashMap built from `LinearOp::Const` instructions —
  check that `DataAddr` is NOT being added to it.
- The `emit_inst` match: the `fused_skip` check at the top applies to
  `Const { dst, .. } | DataAddr { dst, .. }` — if fused_skip contains the
  DataAddr's dst vreg, the entire 16-byte sequence is skipped.

### 2. Rematerialization of DataAddr

The regalloc marks `Const` as `Rematerializable` (spill cost = free,
re-emit movz at use site). If `DataAddr` accidentally gets the same
treatment, the regalloc might re-emit a `movz #0` (the placeholder value)
at the use site instead of reading from the register that holds the
patched address.

**Where to look:**
- `regalloc_engine.rs` around line 1399: the loop that marks `Const` as
  rematerializable. Verify `DataAddr` is excluded.
- The `rematerializable` HashMap in `AllocatedCfgFunctionRa3`.

### 3. Parallel move epilogue ordering

The epilogue parallel move code (in `compile_regalloc3`, around the
`// Resolve each result vreg` section) handles register swaps using x9 as
scratch. If both return values happen to be assigned to x0 and x1 in
reversed order, the swap logic might have a bug in the cycle detection.

**Where to look:**
- `regalloc3_backend.rs`: the `result_regs` / parallel move section in
  the scalar function epilogue.
- Test by setting `KAJIT_RA_DEBUG=1` to print vreg→preg assignments and
  check if data_results[0] and data_results[1] are in x1 and x0.

### 4. DataAddr 16-byte sequence overlapping with branch fixup

The `emit_load_u64_fixed` emits 4 instructions (16 bytes) with placeholder
value 0. The branch fixup pass in `Emitter::finalize()` resolves label
offsets. If a branch fixup happens to target an offset inside the DataAddr
sequence (unlikely but possible if the code layout is unusual), it could
corrupt one of the movz/movk instructions.

**Where to look:**
- Check that no branch labels land inside a DataAddr instruction sequence.
- The data section is appended after all code and after the `ret`
  instruction, so branches should not target it.

## Debugging approach

1. Run with `KAJIT_RA_DEBUG=1` to see vreg→preg assignments.
2. Dump the CFG-MIR (`KAJIT_DUMP_STAGES=cfg`) to see the instruction
   sequence and data_results ordering.
3. Disassemble the JIT output to verify the movz/movk sequences are
   correct and the patching produced the right address.
4. Check if `const_values` contains the DataAddr vreg.
5. Check if `fused_skip` contains the DataAddr vreg.

## Files involved

- `kajit/src/backends/aarch64/regalloc3_backend.rs` — emission, parallel
  move epilogue, fusion analyses
- `kajit-mir/src/regalloc_engine.rs` — rematerialization marking
- `kajit/src/compiler/hir_to_ir.rs` — string literal lowering
  (`lower_expr_multi` for `Literal::String`)
- `kajit/src/compiler/tests/hir_to_ir.rs` — failing test cases
