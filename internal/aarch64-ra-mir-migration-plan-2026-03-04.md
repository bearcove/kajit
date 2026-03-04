# aarch64 RA-MIR Backend Migration Plan (2026-03-04)

## Current Snapshot

- Repo branch: `main`
- Working tree status at handoff time:
  - modified: `kajit/src/ir_backend_aarch64.rs`
- Existing internal note: `internal/theta.md`

Primary active issue:
- aarch64 miscompiles JSON when `theta_loop_invariant_hoist` is enabled.
- Repro examples:
  - `json::integer_boundaries` fails with theta-hoist ON, passes OFF.
  - `json::all_integers` fails ON, passes OFF.
  - `json::all_scalars` fails ON, passes OFF.

Minimal known semantic repro:
- Dedicated test file added: `kajit/tests/theta_hoist_min_repro.rs`
- `2-field` case does not repro.
- `3-field` case repros.
- `4-field` variant does not repro.
- `6-field` boundaries repros.

## What We Learned

### 1) Native JIT divergence is real and localized
LLDB investigation established:
- `kajit_json_key_equals("i32_min")` returns `1` in both ON/OFF runs.
- Divergence happens after match in JIT control flow in `fad::decode::Boundaries`.
- ON path reaches `kajit_json_skip_value`; OFF path reaches `kajit_json_read_i32`.
- First practical split observed around a `cbnz x12` gate where ON has stale/wrong predicate value.

### 2) This is not parser logic
- Same input/key matching behavior.
- Same handler blocks present in emitted code.
- Difference is block reachability due to predicate value transport/control flow.

### 3) Disabling aarch64 branch peepholes did not fix it
A worker disabled:
- `pending_and_branch`
- branch compaction lookahead
- short-circuit and-branch lookahead

Result:
- `json::integer_boundaries` ON still fails.
- So those peepholes are not sufficient explanation / not the root fix.

## Core Architectural Conclusion

The bug class persists because aarch64 backend is still on legacy linear backend-adapter path.

Key fact:
- x86_64 backend consumes `RaProgram + AllocatedProgram` directly.
- aarch64 backend still consumes flat `LinearIr` and emulates CFG behavior.

Evidence in code:
- `kajit/src/ir_backend.rs`
  - x64: comment + call path says reads from `ra_mir` directly.
  - aarch64: comment + call path says uses flat `LinearIr`.
  - `compile_ra_program` on aarch64 currently panics.

This mismatch is the systemic reason this class of CFG/value-transport bugs can survive.

## Direction We Agreed On

Do not keep investing in legacy aarch64 linear adapter behavior.

Primary direction:
- Port aarch64 backend to consume `RaProgram + AllocatedProgram` (same contract as x64).
- Eliminate semantic/CFG rewrites in linear-op streaming context.
- Use RA blocks/terminators/edges as source of truth.

## Pipeline Placement (where this fits)

Current compile path:
1. RVSDG IR build
2. pre-linearization passes (incl. theta-hoist)
3. Linearization -> `LinearIr`
4. RA lowering -> `RaProgram`
5. regalloc2 -> `AllocatedProgram`
6. backend emission

Required migration:
- Keep steps 1-5 as-is.
- Replace aarch64 step 6 implementation to use step 4+5 artifacts directly.

## Execution Plan

### Phase A: Stopgap (optional)
- Keep peephole disables only as temporary safety/diagnostic scaffolding.
- Do not rely on this as final fix.

### Phase B: Main migration (highest priority)
1. Change aarch64 backend entry:
   - from `compile(&LinearIr, max_spillslots, &AllocatedProgram)`
   - to `compile(&RaProgram, &AllocatedProgram)`
2. Refactor `Lowerer` in `ir_backend_aarch64.rs` to iterate:
   - `RaFunction`
   - `RaBlock`
   - `RaInst`
   - `RaTerminator`
3. Branch emission from `RaTerminator` only.
4. Use `inst.linear_op_index` / `block.term_linear_op_index` only for edit lookups/debug metadata.
5. Remove `op_index + N`/label-lookahead branch synthesis from the aarch64 backend path.

### Phase C: Wire backend dispatch
1. In `ir_backend.rs` for aarch64:
   - `compile_linear_ir_with_alloc(...)` should call aarch64 backend with `ra_mir`.
2. Implement aarch64 `compile_ra_program(...)` (remove panic).

### Phase D: Edge identity cleanup (systemic hardening)
Current edge edits keying:
- keyed by `(from_linear_op_index, succ_index)`.

Hardening target:
- key by `(from_block_id, succ_index)` primarily.
- keep linear-op index as debug metadata only.

Reason:
- linear-op indexing is transitional and easier to break with adapter rewrites.
- block identity aligns with CFG semantics.

### Phase E: Remove legacy dead paths
- Delete/retire aarch64-only linear backend branches that no longer apply.
- Ensure no code path still emits aarch64 from flat `LinearIr` semantics directly.

## Verification Plan

### Must-pass targeted tests
- `cargo nextest run -p kajit --test theta_hoist_min_repro -E 'test(=theta_hoist_smallest_known_struct_repro_is_three_fields)'`
- `KAJIT_OPTS='+all_opts,+regalloc,+pass.theta_loop_invariant_hoist' cargo nextest run -p kajit --test corpus -E 'test(=json::integer_boundaries)'`
- `KAJIT_OPTS='+all_opts,+regalloc,-pass.theta_loop_invariant_hoist' cargo nextest run -p kajit --test corpus -E 'test(=json::integer_boundaries)'`
- `KAJIT_OPTS='+all_opts,+regalloc,+pass.theta_loop_invariant_hoist' cargo nextest run -p kajit --test corpus -E 'test(=json::all_integers) | test(=json::all_scalars)'`

### Regression checks around existing failures
- `cargo nextest run -p kajit --test mir_text_regression -E 'test(postcard_u32_single_byte_varint) | test(postcard_u32_multi_byte_varint)'`

### If available
- RA-MIR text compile path on aarch64 (once `compile_ra_program` is enabled) to ensure parity and unblock debugger/MCP workflows.

## Risks / Gotchas

1. Edit mapping assumptions
- aarch64 currently builds edit maps heavily around linear-op indices.
- Migration should preserve correctness first, then clean keying.

2. Hidden linear-path couplings
- Watch for helpers that assume linear scan state (`current_lambda_linear_op_index`, label-forwarding shortcuts, etc.).

3. False confidence from interpreter parity
- Interpreter parity does not validate native predicate transport after regalloc.

## Expected End State

- aarch64 and x64 consume the same backend contract (`RaProgram + AllocatedProgram`).
- No semantic branch rewrites in flat linear stream space.
- `compile_ra_program` works on aarch64.
- Theta-hoist ON/OFF behavior differences only reflect actual IR semantics, not backend adapter artifacts.

## Suggested Work Ordering for Next Session

1. Land migration skeleton (compile signature + dispatch wiring + compile_ra_program support).
2. Get aarch64 backend emitting from RA blocks/terminators with parity on core tests.
3. Only then remove temporary peephole toggles/legacy branches.
4. Perform edge keying cleanup and add guard tests.

## Notes from discussion sentiment (important context)

- Team frustration was specifically about spending time debugging symptoms while root architecture mismatch (aarch64 not RA-MIR-native) remained.
- Strategic decision: prioritize architectural convergence over local patching.
