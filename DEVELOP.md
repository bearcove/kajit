# Development notes

## JIT debugging guide

kajit registers generated code with the GDB JIT interface. You can debug JIT code,
but you should expect partial symbolization unless/until full unwind metadata is emitted.

## x86_64 testing on Apple Silicon (Rosetta 2)

When validating x86_64 codegen on an Apple Silicon machine, run tests as an
x86_64 process under Rosetta.

### One-time setup

1. Install Rosetta 2:
```bash
softwareupdate --install-rosetta --agree-to-license
```
2. Install the Rust target:
```bash
rustup target add x86_64-apple-darwin
```

### Fast x86_64 smoke loop

Run the focused x86_64 smoke set for quick local iteration:
```bash
cargo test-x86_64
# equivalent:
cargo xtask test-x86_64
```

Current smoke set:
- `prop::deny_unknown_fields`
- `prop::flat_struct`
- `prop::scalar_i64`
- `prop::nested_struct`
- `prop::transparent_composite`
- `prop::shared_inner_type`

### Full x86_64 test suite

```bash
cargo xtask test-x86_64 --full
# or directly:
cargo nextest run -p kajit --target x86_64-apple-darwin
```

You can pass extra nextest arguments to the helper:
```bash
cargo xtask test-x86_64 -- --no-fail-fast
```

### Pipeline bisecting with `KAJIT_OPTS`

Use `KAJIT_OPTS` to selectively disable parts of the compile pipeline at
runtime, without rebuilding.

Syntax:
- comma-separated tokens
- `+name` enables an option
- `-name` disables an option
- bare `name` is treated like `+name`
- unknown names fail fast

Supported options:
- `all_opts`: default RVSDG optimization passes before linearization
- `regalloc`: regalloc edit application during backend emission

Per-pass options:
- `pass.bounds_check_coalescing`
- `pass.theta_loop_invariant_hoist`
- `pass.inline_apply`
- `pass.dead_code_elimination`

Examples:
```bash
# disable optimization passes
KAJIT_OPTS='-all_opts' cargo nextest run -p kajit <test_filter>

# disable regalloc edit application
KAJIT_OPTS='-regalloc' cargo nextest run -p kajit <test_filter>

# disable both
KAJIT_OPTS='-all_opts,-regalloc' cargo nextest run -p kajit <test_filter>

# disable one specific default pass
KAJIT_OPTS='-pass.inline_apply' cargo nextest run -p kajit <test_filter>
```

Show built-in help:
```bash
KAJIT_OPTS=help cargo nextest run -p kajit <test_filter>
```

On Apple Silicon, combine with x86_64 Rosetta validation:
```bash
KAJIT_OPTS='-regalloc' cargo xtask test-x86_64 --full -- \
  --test corpus -E 'test(=postcard::scalar_u16_v1)'
```

Interpretation notes:
- Disabling `all_opts` can change IR/RA-MIR snapshots while preserving runtime behavior.
- Disabling `regalloc` can intentionally surface semantic regressions and is useful
  for isolating bugs in the regalloc/edit-application path.

### Debugging x86_64 tests with LLDB under Rosetta

Build and run the x86_64 test binary through Rosetta:
```bash
cargo nextest run -p kajit --target x86_64-apple-darwin <test_name> --no-run
arch -x86_64 lldb target/x86_64-apple-darwin/debug/deps/<test_binary>
```

Within LLDB, source breakpoints and backtraces work as usual for translated
x86_64 binaries.

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

## GDB (Linux / OrbStack)

### OrbStack on Apple Silicon

OrbStack runs x86_64 Linux binaries via Rosetta. GDB and LLDB **cannot** attach
directly to Rosetta-emulated processes — you'll get `Cannot PTRACE_GETREGS:
Input/output error`. This is not a ptrace_scope issue; it's fundamental to the
Rosetta emulation layer.

The workaround is `qemu-user` as a GDB remote stub:

```bash
sudo apt install qemu-user
```

1. Build tests:
```bash
cargo nextest run <test_name> --no-run
# find the binary:
cargo nextest list -p kajit --test corpus --message-format json-pretty | rg '"binary-path"'
```

2. Start the binary under qemu with a GDB server:
```bash
qemu-x86_64 -g 1234 ./target/debug/deps/corpus-<hash> --exact postcard::deny_unknown_fields_v1 --nocapture
```

3. Connect GDB (in another terminal, or via the MDB-MCP GDB server):
```gdb
file target/debug/deps/corpus-<hash>
target remote :1234
break kajit::deserialize_with_ctx<corpus::Strict>
continue
```

4. Once stopped at the JIT call site, step into JIT code:
```gdb
stepi   # repeat until you land in fad::decode::*
x/80i $pc   # disassemble JIT code
```

### MDB-MCP (GDB via MCP)

A GDB MCP server is configured in `.mcp.json` using
[MDB-MCP](https://github.com/smadi0x86/MDB-MCP), cloned to
`~/.local/share/MDB-MCP`:

```json
{
  "mcpServers": {
    "gdb": {
      "command": "uv",
      "args": ["run", "--directory", "/home/amos/.local/share/MDB-MCP", "server.py"]
    }
  }
}
```

This exposes `gdb_start`, `gdb_command`, `gdb_terminate` etc. as MCP tools.
Use `gdb_command` with `target remote :1234` to connect to a qemu-user stub.

### Native GDB (non-Rosetta Linux)

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
