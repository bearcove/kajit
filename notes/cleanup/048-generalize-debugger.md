# 030: Generalize debugger away from deserialization assumptions

## Problem

`DebuggerSession` is hardcoded to deserialization:

- `input`, `cursor`, `input_base_addr`, `input_end_addr` fields
- `RuntimeDeserContext` with `input_ptr`, `input_end`, `error` baked in
- `execute_call_intrinsic` syncs cursor/error into ctx before/after every call
- `set_root_cursor_arg_addr` assumes data_args layout is deserialization-specific
- `seed_root_data_args` hardcodes `data_args[0]=out_ptr`, `data_args[1]=ctx_ptr`

If we want to debug a general-purpose program (e.g. one that reads files,
makes network requests, processes data), none of this applies.

## Goal

- Initial vreg values as `Vec<(usize, u64)>` — caller provides, debugger
  doesn't interpret
- `execute_call_intrinsic` just calls the function with vreg values as args —
  no cursor sync, no ctx fixup
- Deserialization-specific behavior (cursor tracking, ctx sync) becomes an
  optional layer on top, not baked into the core debugger
