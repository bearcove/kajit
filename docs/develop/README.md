# Development guide

This directory contains guides for debugging and testing kajit. Each page is
self-contained — pick the one that matches the problem you're dealing with.

For project architecture and the compilation pipeline, see
[`docs/architecture.md`](../architecture.md). For the full format specification,
see [`docs/spec.md`](../spec.md).

## Where to start

**A test produces wrong output.** Start with [bisecting.md](bisecting.md) to
narrow down which pipeline pass introduces the bug. Once you know the pass, use
the [MIR debugger](mir-debugger.md) to trace execution and find the exact
operation that goes wrong.

**A test crashes or segfaults.** Try to reproduce it in the
[MIR interpreter](mir-debugger.md) first — it's deterministic and reversible,
so you can step backward from the failure. If the crash doesn't reproduce in the
interpreter, it's a native codegen or ABI issue; go to
[native-debugging.md](native-debugging.md) for LLDB/GDB workflows.

**You need to validate x86_64 codegen on Apple Silicon.**
[running-tests.md](running-tests.md) covers one-time Rosetta setup, the smoke
loop, and the full x86_64 test suite.

## Pages

| Page | What it covers |
|------|---------------|
| [running-tests.md](running-tests.md) | Running tests, x86_64 cross-testing under Rosetta |
| [bisecting.md](bisecting.md) | Isolating failures with `KAJIT_OPTS`, `opts-matrix`, pipeline dumps |
| [mir-debugger.md](mir-debugger.md) | MIR interpreter debugger — the first tool to reach for |
| [native-debugging.md](native-debugging.md) | LLDB/GDB for native codegen bugs that don't reproduce in MIR |
