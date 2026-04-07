# 021a: Option/vec corpus tests timeout (infinite loop in JIT)

## Problem

After the 011-2 / 021 changes, these corpus tests hang:

- `postcard::option_u32_v0`, `postcard::option_u32_v1`
- `postcard::option_string_v0`
- `postcard::vec_scalar_small`, `postcard::vec_scalar_large`
- `postcard::vec_u32_v0`, `postcard::vec_u32_v1`
- Corresponding `prop::*` variants

The interpreter path appears to work (no timeouts in `kajit-mir` interpreter
tests). The hang is in JIT-compiled code.

## Likely cause

These types use `CallIntrinsic` (option_init_none_ctx, option_init_some_ctx).
The backend's `emit_call_intrinsic` was rewritten to use RA-placed ABI args
instead of manual arg shuffling. Something in the post-call error check or
arg placement may be wrong, causing an infinite retry loop or corrupted
control flow.

## Diagnosis approach

- Run `cargo run -p kajit-cli -- compile postcard 'Option<u32>' -s asm` and
  inspect the generated assembly around CallIntrinsic sites
- Use the differential harness to compare interpreter vs JIT
- Check if the post-call error check reads ctx from the right location
  (ctx vreg may be spilled and the reload after the call may read stale data)
