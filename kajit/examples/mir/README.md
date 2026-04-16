# MIR examples

Run these directly with the CLI:

```bash
cargo run -p kajit-cli -- compile examples/mir/const-ret.k-mir
cargo run -p kajit-cli -- compile examples/mir/copy-add-sub-branch.k-mir
```

The files are intentionally tiny and use fixed registers so they exercise the
current host-arch-only lowering path without any regalloc machinery.
