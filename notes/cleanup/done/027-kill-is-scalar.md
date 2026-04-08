# 018: Kill is_scalar, unify prologues

## Goal

No "decoder ABI" vs "scalar ABI" distinction. All functions use the same
prologue/epilogue.

## What changes

- Delete `is_scalar` from `IrFunc`, `LinearIr`, `Program`, all propagation
- Delete `emit_decoder_prologue`, `emit_decoder_epilogue`
- Delete `RootDecoderDataAbi`, `infer_root_decoder_data_abi`
- All functions use `emit_scalar_prologue` / `emit_scalar_epilogue`
- Update `CompiledDecoder::invoke` — output_ptr and ctx_ptr are regular args

## Depends on

All previous items (no more pinned registers, no implicit args, no decoder ops).
