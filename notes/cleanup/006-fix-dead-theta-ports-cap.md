# Fix dead_theta_ports 3-port cap

**Phase:** 2

## Current state

**File:** `kajit-ir/src/dead_theta_ports.rs`

### The cap (lines 271-276):
```rust
dead_ports.push((p, const_val));
// Safety cap: adjacent port removal (e.g., 20+21) triggers a bug
// in Phase 2+3 interaction that hasn't been root-caused yet. Cap at 3
// as fallback — matches the previous validated set (17, 18, 20).
if dead_ports.len() >= 3 {
    break;
}
```

### The port removal code (lines 466-484):
```rust
fn remove_theta_data_port(func, theta_id, body, p) {
    func.nodes[theta_id].inputs.remove(p);
    func.regions[body].args.remove(p);
    func.regions[body].results.remove(1 + p);  // +1 for predicate
    func.nodes[theta_id].outputs.remove(p);
    shift_output_refs_down(func, theta_id, from: (p+1) as u16);
}
```

### Additional limit (lines 63-66):
Skips theta bodies with > 50 nodes to avoid O(n²) scanning.

## Root cause hypothesis

The bug is in port index shifting during multi-port removal. When removing ports in order (e.g., ports 20 and 21), the first removal shifts indices so port 21 becomes port 20. If `shift_output_refs_down` doesn't account for the cascading shift of previously-removed ports, references get misaligned.

## What to do

1. Write a text test with 5+ dead ports (including adjacent indices)
2. Root-cause the Phase 2+3 interaction bug — likely in `shift_output_refs_down` or the removal loop order
3. Fix: either remove in reverse order, or adjust port indices for already-removed ports
4. Remove the cap
5. Consider also removing or raising the 50-node body limit
