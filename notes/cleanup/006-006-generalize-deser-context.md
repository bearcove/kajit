# Generalize DeserContext

Make the context a user-defined struct instead of a compiler-known layout.

## Current state

`kajit/src/context.rs` defines `DeserContext` with hardcoded fields and offset constants:
- `CTX_INPUT_PTR: u32 = 0`
- `CTX_INPUT_END: u32 = 8`
- `CTX_ERROR_CODE: u32 = 16`
- `CTX_ERROR_OFFSET: u32 = 20`

Backends use these offsets for direct memory access. All intrinsics take `&mut DeserContext`.

## Target state

- The compiler doesn't know the context struct layout
- The frontend declares context fields with types and offsets
- Error reporting becomes an intrinsic call (`set_error(code, offset)`) not a struct field write
- Format-specific fields (key_scratch, trusted_utf8) live in the frontend's context type
- `compile_decoder()` → `compile_program()` or similar
- `invoke_decoder()` → generic entry point

## Also generalize

- `kajit/src/intrinsics.rs` — intrinsics take generic context types or receive context as opaque ptr
- `kajit/src/lib.rs` — decoder-specific entry points become generic
- `kajit-postcard/src/lib.rs` — postcard defines its own context type

## Depends on

006-005 (backends don't use CTX_INPUT_PTR/CTX_INPUT_END anymore)
