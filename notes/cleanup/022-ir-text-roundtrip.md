# 013: IR text round-trip broken for data_args

## Problem

The `exec` stage in kajit-cli panics with "index out of bounds: the len is 1
but the index is 1" in `kajit-ir-text/src/ir_parse.rs:1320` when compiling
`Option<u32>`.

Root cause: the IR text format doesn't correctly round-trip functions with
multiple `data_args`. The parser creates regions with fewer args than expected.

## Fix

Update `kajit-ir-text/src/ir_parse.rs` to handle multiple data_args in the
text representation. May also need updates to the IR printer.

## Affects

- `kajit-ir-text/src/ir_parse.rs`
- The `exec` CLI stage (which round-trips through IR text)
- Not a blocker for JIT compilation — only affects text-based tooling
