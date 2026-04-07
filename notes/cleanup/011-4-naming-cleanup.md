# 011-4: Naming cleanup

Cosmetic pass. Low priority.

## Candidates

- `compile_decoder()` → `compile_program()`
- `CompiledDecoder` → `CompiledProgram`
- `invoke_decoder()` → generic name
- Move `DeserContext` / `RuntimeDeserContext` to `kajit-postcard`
- `hir_to_ir.rs` decoder-specific naming

## Depends on

011-3 (explicit error check — no more backend knowledge of context fields).
