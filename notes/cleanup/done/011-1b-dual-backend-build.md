# Build both backends on all platforms

Remove `#[cfg(target_arch)]` gates so both x86_64 and aarch64 backends compile everywhere. Select backend at runtime.

## Why now

Every subsequent 011-x change touches both backends. Building both on the host platform catches compilation errors immediately instead of discovering them in CI.

## What changes

- `CompiledDecoder` / `CompiledFunction` hold an enum instead of cfg-gated fields
- `compile_decoder` / backend selection becomes runtime (based on target arch or caller choice)
- `kajit_emit` crates for both arches must compile on all platforms (they should already — they're pure codegen)
- Tests run the native backend; cross-backend only needs to compile

## Depends on

011-1 (cursor sync removed).
