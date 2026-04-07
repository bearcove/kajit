# Clean up decoder-specific naming

Cosmetic pass. Low priority.

## Candidates

- `compile_decoder()` → `compile_program()`
- `CompiledDecoder` → `CompiledProgram`
- `invoke_decoder()` → generic name
- Move `DeserContext` definition to `kajit-postcard`

## Depends on

011-5 (is_scalar killed — unified ABI).
