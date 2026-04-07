# Kill is_scalar and decoder ABI

With output_ptr and ctx_ptr as regular data_args, and no implicit cursor/error management, there's no difference between "decoder" and "scalar" functions.

## What to delete

- `is_scalar` from IrFunc, LinearIr, Program, all propagation (17 files)
- `emit_decoder_prologue`, `emit_decoder_epilogue`
- `RootDecoderDataAbi`, `decoder_data_arg_enc`
- `infer_root_decoder_data_abi`
- All functions use `emit_scalar_prologue`/`emit_scalar_epilogue`

## Depends on

011-4 (explicit error check — no more backend-managed error checking).

## Enables

011-6 (naming cleanup).
