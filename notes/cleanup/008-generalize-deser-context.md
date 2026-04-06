# Generalize DeserContext

Make the context a user-defined struct instead of a compiler-known layout.

## Current state

`kajit/src/context.rs` defines `DeserContext` with hardcoded fields:
- `input_ptr`, `input_end` — cursor state (removed by 006/007)
- `error: ErrorSlot` — error reporting
- `key_scratch_ptr`, `key_scratch_cap` — JSON-specific scratch buffer
- `trusted_utf8` — format-specific hint

Field offsets (`CTX_INPUT_PTR`, `CTX_INPUT_END`, `CTX_ERROR_CODE`, `CTX_ERROR_OFFSET`) are used by backends for direct memory access.

## Target state

- The compiler doesn't know the context struct layout
- The frontend declares context parameters with types; the backend passes them per calling convention
- Error reporting becomes an intrinsic call (e.g., `set_error(code, offset)`) not a magic struct field write
- Format-specific fields (scratch, trusted_utf8) live in the frontend's context type

## Also generalize

- `kajit/src/intrinsics.rs` — all intrinsics take `&mut DeserContext`. Make intrinsics take generic context types or receive context as an opaque pointer + field offsets.
- `kajit/src/lib.rs` — `compile_decoder()` → `compile_program()` or similar. `invoke_decoder()` → generic entry point.
- `kajit-postcard/src/lib.rs` — postcard defines its own context type and registers its intrinsics
