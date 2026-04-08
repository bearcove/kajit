# 048a: Remove legacy address fields from DebuggerSession

## Problem

`DebuggerSession` has four deserialization-specific fields:

```rust
pub input_base_addr: Option<u64>,
pub input_end_addr: Option<u64>,
pub output_base_addr: Option<u64>,
pub root_cursor_arg_addr: Option<u64>,
```

These exist because the interpreter has a dual-mode design: abstract offsets
for its own load/store simulation, real pointers for intrinsic FFI calls.
The legacy fields bridge between the two modes.

With the general-purpose IR model, the interpreter should work entirely with
real pointers. `seed_data_args()` already writes real pointer values into
vregs. The interpreter's owned buffers (`self.input`, `self.output`) already
exist at real memory addresses.

## What to change

1. **Always use real pointers.** `LoadFromAddr` and `StoreToAddr` should
   read/write real memory (via the pointer values in vregs), not abstract
   offsets into `self.input`/`self.output`.

2. **Delete the interception layers:**
   - `load_root_cursor_arg()` — intercepts cursor struct field loads
   - `store_root_cursor_arg()` — intercepts cursor struct field stores
   - `translate_debug_address_to_host_ptr()` — converts between modes

3. **Delete the legacy fields:** `input_base_addr`, `input_end_addr`,
   `output_base_addr`, `root_cursor_arg_addr`.

4. **Delete `set_real_addresses()` and `set_root_cursor_arg_addr()`.**

5. **Update `DebuggerSession::new()` to accept `&Arguments`** and seed
   data_args from that.

## Impact

- 27 references in debugger.rs
- Callers: lockstep.rs, mcp.rs, reduce.rs, debugger tests
- The `self.cursor` field tracking may also need rethinking — currently
  updated by `store_root_cursor_arg`, but with real pointers the cursor
  position lives in memory at the cursor struct address.

## Depends on

031+032 (out is a proper pointer in HIR).

## Enables

048 (generalize debugger — once the legacy fields are gone, the debugger
has no deserialization-specific knowledge).
