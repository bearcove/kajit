# lldb-sys

[![](https://img.shields.io/crates/v/lldb-sys.svg)](https://crates.io/crates/lldb-sys)
[![docs.rs](https://img.shields.io/docsrs/lldb-sys)](https://docs.rs/lldb-sys)

Dual licensed under the MIT and Apache 2 licenses. The C bindings for LLDB are
licensed under the same license as LLVM.

## Installation

This crate works with Cargo and is on
[crates.io](https://crates.io/crates/lldb-sys).
Add it to your `Cargo.toml` like so:

```toml
[dependencies]
lldb-sys = "0.0.31"
```

### Linux

Install the `lldb` and `liblldb-dev` packages for your platform so that
you have both LLDB itself installed as well as the headers and
other support files required.

### macOS:

Install `llvm` via homebrew:

```shell
brew install llvm
```

Then, tell this crate how to find `llvm-config`:

```shell
export LLVM_CONFIG="`brew --prefix llvm`/bin/llvm-config"
```

## Contribution

Unless you explicitly state otherwise, any contribution
intentionally submitted for inclusion in the work by you,
as defined in the Apache-2.0 license, shall be dual licensed
as above, without any additional terms or conditions.

## Binding Re-generation

If something happens where the Rust bindings need to be re-generated,
they were generated from this set of files with `bindgen`. We have a
script to automate some of the processing that is needed:

```shell
bin/generate_bindings.sh
```

After running that, you will want to selectively merge the changes
with what was previously there as the output of this process is
hand-edited to add usage of `libc` as well as doc comments in critical
places.
