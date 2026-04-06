# Fix post_unroll_canonicalize 800-node cap

Limits codegen for large types. **Wait until after 006 (remove cursor ops from IR)** — unrolling cursor-carrying thetas changes once cursor is ordinary data.

## Current state

**File:** `kajit-ir/src/post_unroll_canonicalize.rs`

### The cap (lines 488-492):
```rust
pub fn post_unroll_canonicalize(func: &mut IrFunc) -> bool {
    let total_nodes: usize = func.regions.iter().map(|(_, r)| r.nodes.len()).sum();
    if total_nodes > 800 {
        return false;
    }
```

Skips the entire pass for functions with >800 total nodes across all regions, citing "compile-time regression."

## What to do

1. Write a text test with ~900 nodes that should be canonicalized
2. Profile: is the regression in canonicalize itself, or in a downstream pass that gets a larger input?
3. Fix the performance issue (likely algorithmic, not inherent)
4. Remove the cap
