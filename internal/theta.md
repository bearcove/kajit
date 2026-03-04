# Theta Hoist aarch64 Native Stepping Investigation (`json::all_scalars`)

Date: 2026-03-04  
Repo: `/Users/amos/bearcove/kajit`  
Platform: macOS arm64 (aarch64)  
Method: Native LLDB stepping via MCP, JIT symbols enabled, no code changes, no instrumentation/logging additions.

## Goal
Find where `json::all_scalars` diverges on aarch64 when `theta_loop_invariant_hoist` is enabled.

## Exact commands used

### Repro commands
```bash
cargo nextest list -p kajit --test corpus -E 'test(=json::all_scalars)'
KAJIT_OPTS='+all_opts,+regalloc,+pass.theta_loop_invariant_hoist' cargo nextest run -p kajit --test corpus -E 'test(=json::all_scalars)'
KAJIT_OPTS='+all_opts,+regalloc,-pass.theta_loop_invariant_hoist' cargo nextest run -p kajit --test corpus -E 'test(=json::all_scalars)'
```

### Test binary resolution
```bash
cargo nextest list -p kajit --test corpus --message-format json-pretty \
  | jq -r '.["rust-suites"]["kajit::corpus"]["binary-path"]'
```
Resolved binary:
`/Users/amos/bearcove/kajit/target/debug/deps/corpus-27b388e0b25c2bef`

### LLDB command pattern (MCP)
- `target create /Users/amos/bearcove/kajit/target/debug/deps/corpus-27b388e0b25c2bef`
- `settings set plugin.jit-loader.gdb.enable on`
- `settings set target.env-vars KAJIT_OPTS=...`
- `breakpoint set -n __jit_debug_register_code`
- `breakpoint set -r kajit_json_key_equals`
- `breakpoint set -r kajit_json_read_isize`
- `breakpoint set -r kajit_json_read_f64`
- `breakpoint set -r kajit_json_skip_value`
- `run --exact json::all_scalars --nocapture`
- `register read ...`
- `thread step-out`
- `disassemble --start-address ... --count ...`
- `image lookup -a ...`

## Repro result (required ON/OFF)

- Theta ON (`+pass.theta_loop_invariant_hoist`): **FAIL**
- Theta OFF (`-pass.theta_loop_invariant_hoist`): **PASS**

Observed failure signature (ON): wrong `a_isize` and wrong `a_f64` in decoded struct.

## Intrinsic hit sequences and counts

## Theta ON
Observed ordered behavior:
1. `kajit_json_key_equals` repeatedly in key dispatch (focused run: hit count 12)
2. no hits for `kajit_json_read_isize` (hit count 0)
3. no hits for `kajit_json_read_f64` (hit count 0)
4. enters `kajit_json_skip_value` from same JIT callsite (hit count 2 while enabled)

Breakpoint-count snapshot from ON session:
- `kajit_json_key_equals`: 12
- `kajit_json_read_isize`: 0
- `kajit_json_read_f64`: 0
- `kajit_json_skip_value`: 2

## Theta OFF
Observed ordered behavior:
1. `kajit_json_key_equals` (dispatch path active)
2. `kajit_json_read_isize` (hit count 1)
3. `kajit_json_read_f64` (hit count 1)
4. no `kajit_json_skip_value` hits (hit count 0)

Breakpoint-count snapshot from OFF session:
- `kajit_json_key_equals`: 1 (was disabled early to reduce noise)
- `kajit_json_read_isize`: 1
- `kajit_json_read_f64`: 1
- `kajit_json_skip_value`: 0

## `kajit_json_key_equals` capture details

For stepped hits:
- arm64 calling convention matched intrinsic signature:
  - `x0=key_ptr`, `x1=key_len`, `x2=expected_ptr`, `x3=expected_len`
- return captured via `thread step-out` and `x0`:
  - `x0=1` for match
  - `x0=0` for non-match
- post-return flow returned into JIT `fad::decode::AllScalars` and then took either handler path (OFF) or skip path (ON), depending on subsequent branch outcome.

## First divergent control-flow point after match

### OFF (working)
JIT `fad::decode::AllScalars` dispatch region reaches field handlers:
- call at `0x101538cbc` (`+3260`) -> `kajit_json_read_isize`
- call at `0x101538d8c` (`+3468`) -> `kajit_json_read_f64`

### ON (failing)
In corresponding region, code for handler calls exists, but runtime falls into unknown-field path and reaches:
- call at `0x101538ee0` (`+3808`) -> `kajit_json_skip_value`
  - confirmed by breakpoint on `kajit_json_skip_value`
  - `lr=0x101538ee4` (`fad::decode::AllScalars + 3812`)

### Divergence statement
First practical divergence is at the dispatch branch chain that should route matched keys into `read_isize/read_f64` handler blocks (OFF) but instead falls through to skip-value block (ON), culminating at `fad::decode::AllScalars + 3808` (`blr x16` to skip path).

## Best hypothesis (evidence-based)

Most likely cause: **aarch64 codegen/regalloc interaction triggered by `theta_loop_invariant_hoist`**, affecting the dispatch predicate/availability of comparison result used by branch lowering.

Evidence:
1. Toggle-only behavior: identical test and target, only theta-hoist ON/OFF changes result.
2. ON path does not invoke `read_isize/read_f64`, but repeatedly invokes `skip_value`.
3. ON disassembly still contains `read_isize/read_f64` call blocks, so handlers are present in codegen; failure is control-flow selection at runtime, not missing intrinsic emission.
4. Key-equals intrinsic itself receives sane args and returns sensible match/non-match values when stepped.

## Notes
- Investigation used native JIT stepping only (LLDB MCP), per request.
- No source modifications were made.
- No additional probes/logging were added.
