# Update RVSDG passes for cursor-as-data

After cursor ops become ordinary memory ops, several RVSDG passes need updating.

## Passes that break

### slot2reg (CRITICAL)
- Lines 358, 429, 441, 456, 480: `state_count = func.state_domains.len()` assumes contiguous trailing state block
- If cursor domain is gone, state_count changes, port offsets shift
- Needs explicit state port layout instead of assuming trailing block

### bounds_check_coalescing (COMPLETE REWRITE)
- `state_cursor_input_source()` and `state_cursor_output_ref()` match on `CURSOR_STATE_PORT`
- `cursor_chain_step()` traces BoundsCheck → ReadBytes → AdvanceCursor chains via state ports
- Must be rewritten to track cursor values as data, not state

### const_fold
- `resolve_slot_value()` walks backwards through cursor state chain (lines 388-402)
- Pattern matching on state input would fail if cursors become data values
- Needs rewrite of chain-walking logic

## Passes that work unchanged

- `dead_theta_ports` — correctly separates state from data generically
- `unroll_theta` — body cloning preserves all threading
- `simplify_gamma` — operates on predicates, cursor-agnostic

## Depends on

006-002 (cursor ops are replaced)
