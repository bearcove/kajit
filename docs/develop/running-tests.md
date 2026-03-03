# Running tests

kajit uses [cargo-nextest](https://nexte.st/) for test execution. The main test
suites live in `kajit/tests/` — `corpus.rs` contains generated round-trip tests
for every supported format/type combination, and `mir_text_regression.rs` /
`ir_text_regression.rs` are for pasting minimal reproducers from pipeline dumps.

Most development happens on aarch64 (Apple Silicon). When you need to verify
x86_64 codegen, Rosetta lets you run x86_64 test binaries without leaving your
Mac.

## x86_64 testing on Apple Silicon (Rosetta 2)

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
