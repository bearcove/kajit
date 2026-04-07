# 011-2: Unify ABI — output_ptr/ctx_ptr as regular data_args, kill is_scalar

## Goal

The compiler compiles general-purpose functions. No "decoder ABI" vs "scalar ABI" distinction.
Output pointer and context pointer are just regular function arguments passed by the caller.

## What changes

### HIR→IR lowering (`kajit/src/compiler/hir_to_ir.rs`)
- Decoder root lambda gets `output_ptr` as `data_args[0]` and `ctx_ptr` as `data_args[1]`
- Existing data_args (cursor ref etc.) shift to `data_args[2..]`

### Backends (`kajit/src/backends/{aarch64,x86_64}/regalloc3_backend/`)
- `output_reg`/`ctx_reg` on EmitContext are set from the RA-assigned location of `data_args[0]`/`data_args[1]`
- Delete decoder prologue/epilogue — all functions use scalar prologue/epilogue
- Delete `emit_decoder_prologue`, `emit_decoder_epilogue`, `begin_func_with_config`, `end_func_with_config`
- Delete `output_enc`/`ctx_enc` pinned register fields

### Register allocation (`kajit-mir/src/regalloc_engine.rs`)
- Delete `abi_arg_offset` — always 0
- Delete callee-saved preference logic for decoder args

### Compiler (`kajit/src/compiler/mod.rs`)
- Delete `RootDecoderDataAbi`, `infer_root_decoder_data_abi`, `decoder_data_arg_enc`
- Delete register exclusion for decoder ABI
- `CompiledDecoder::invoke` passes output_ptr and ctx_ptr as first two args

### IR/LIR/CFG-MIR
- Delete `is_scalar` from `IrFunc`, `LinearIr`, `Program`
- `WriteToField`/`ReadFromField`/`SaveOutPtr`/`SetOutPtr` ops stay as-is in the IR —
  the backend resolves "the output pointer" by reading `func.data_args[0]`'s assigned register

### Interpreter (`kajit-mir/src/interpreter.rs`)
- `out_ptr` initialized from `data_args[0]` vreg value (passed by caller)
- `ctx` pointer from `data_args[1]` vreg value (or interpreter builds its own ctx and writes the pointer to the vreg)

## Depends on

011-1c (split backends done).

## Enables

011-3 (explicit error check — backend no longer knows about ctx struct fields).
