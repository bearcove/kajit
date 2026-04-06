# Rewrite DCE pass

**Phase:** 1

The current DCE is O(n²) for no reason. This is a textbook algorithm — rewrite it clean against the golden text tests from 001.

Each pass should be a pure function: IR in → IR out. No global state, no spooky interaction with other passes.
