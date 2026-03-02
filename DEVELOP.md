# Development notes

## JIT debugging guide

kajit registers generated code with the GDB JIT interface. You can debug JIT code,
but you should expect partial symbolization unless/until full unwind metadata is emitted.

## LLDB (macOS)

1. Build tests in debug mode:
```bash
cargo nextest run <test_name> --no-run
```
2. Find the exact test binary path:
```bash
cargo nextest list <test_name> --message-format json-pretty
```
Then copy `rust-suites.kajit.binary-path` (for unit tests this is usually
`target/debug/deps/kajit-<hash>`).

Practical binary-path extraction examples:
```bash
# Integration test binary (corpus suite)
cargo nextest list -p kajit --test corpus --message-format json-pretty \
  | jq -r '.["rust-suites"]["kajit::corpus"]["binary-path"]'

# Quick fallback without jq
cargo nextest list -p kajit --test corpus --message-format json-pretty \
  | rg '"binary-path"'
```

For `nextest` names like `kajit::corpus prop::all_scalars`, run that test from
the binary as:
```text
run --exact prop::all_scalars --nocapture
```

3. Start LLDB on the test binary:
```bash
lldb target/debug/deps/kajit-<hash>
```
4. Enable the GDB JIT loader plugin (required on macOS):
```text
settings set plugin.jit-loader.gdb.enable on
```
5. Break on JIT registration:
```text
breakpoint set -n __jit_debug_register_code
```
6. Run the test:
```text
run --exact tests::<test_name> --nocapture
```
7. Confirm a JIT image exists and resolve symbols explicitly:
```text
image list
image lookup -a $pc
image lookup -rn 'kajit::decode::'
image lookup -rn 'kajit::encode::'
```

Notes:
- Seeing `warning: ... no plugin for the language "rust"` in LLDB is expected;
  JIT symbol lookup still works.
- `image list` should show an entry like `JIT(0x...)`.

## GDB (Linux)

1. Build tests in debug mode:
```bash
cargo nextest run <test_name> --no-run
```
2. Start GDB on the test binary:
```bash
gdb target/debug/deps/kajit-<hash>
```
3. Break on JIT registration:
```gdb
break __jit_debug_register_code
run --exact <test_name> --nocapture
```
4. Resolve JIT symbols / PC:
```gdb
info functions kajit::decode::
info functions kajit::encode::
info symbol $pc
disassemble $pc-64, $pc+64
```

## Known limitations

- `thread backtrace` may still show raw PCs for top JIT frames.
- Explicit symbol lookup (`image lookup` in LLDB, `info symbol` in GDB) is more reliable.
- Current JIT symbolization is enough for function-level resolution, not full source-level stepping.

## Practical JIT codegen debugging tips

- Breakpoint both the JIT registration and the suspect intrinsic:
```text
breakpoint set -n __jit_debug_register_code
breakpoint set -n <intrinsic_name>
```
- For arm64, inspect call arguments with:
```text
register read x0 x1 x2 x3 x4
thread backtrace
```
- In current IR backend ABI:
  - `CallIntrinsic` with `has_result=true` expects intrinsic signature
    `fn(ctx, args...) -> value`.
  - `CallIntrinsic` with `has_result=false` expects intrinsic signature
    `fn(ctx, args..., out_plus_field_offset)`.
- If an intrinsic with `has_result=true` is declared without leading `ctx`, LLDB
  will show nonsensical argument values (for example huge lengths/pointers). This
  is a strong sign of ABI mismatch in lowering/codegen rather than parser logic.

## Existing detailed doc

For the focused macOS LLDB workflow, see:
`/Users/amos/bearcove/kajit/docs/jit-debugging.md`
