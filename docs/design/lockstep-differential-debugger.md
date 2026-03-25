# Lockstep Differential Debugger

## Vision

A debugger that drives the CFG-MIR ideal interpreter and LLDB (attached to JIT
machine code) **simultaneously**, stepping both in lockstep and stopping the
instant they diverge. The ultimate differential harness — not just "output
differs" but "this specific instruction on this specific vreg produced the wrong
value."

## Architecture

```
┌─────────────────────┐     ┌─────────────────────┐
│  CFG-MIR Interpreter│     │  LLDB (JIT process)  │
│  (MCP or in-proc)   │     │  (MCP or direct API) │
│                     │     │                     │
│  step() → state     │     │  step → registers   │
│  vreg values        │     │  DWARF line → OpId   │
│  cursor, output     │     │  memory state        │
└────────┬────────────┘     └────────┬────────────┘
         │                           │
         └─────────┬─────────────────┘
                   │
         ┌─────────▼─────────────┐
         │   Lockstep Controller  │
         │                       │
         │  1. Step both          │
         │  2. Map DWARF line →   │
         │     CFG-MIR OpId       │
         │  3. Compare state      │
         │  4. Stop on divergence │
         └───────────────────────┘
```

## Key Insight: DWARF Line ↔ OpId Mapping

We already emit DWARF `.debug_line` info that maps each emitted machine
instruction back to a CFG-MIR `OpId` (via `set_source_location()` in the
backend emission loop). This is the bridge:

- **Interpreter side**: step one CFG-MIR op, get the new vreg state.
- **LLDB side**: step until the DWARF line number advances to the next OpId,
  then read registers.
- **Compare**: map physical registers back to vregs via the regalloc allocation
  map, then compare vreg values.

## Components

### 1. Vreg-to-Physical-Register Map

The regalloc3 allocator produces a mapping from vreg → physical register (or
stack slot) at each program point. We need to expose this as a queryable
structure:

```
given (OpId, VReg) → PhysicalLocation { Register(preg) | Stack(offset) }
```

This already exists implicitly in the allocation result — it just needs to be
serialized into a form the debugger can query.

### 2. State Comparison

At each OpId boundary, compare:

- **Vreg values**: For each vreg live at this point, read the physical register
  (or stack slot) from LLDB and the vreg value from the interpreter. Flag
  mismatches.
- **Cursor position**: The interpreter tracks `cursor` explicitly. The JIT
  stores it in a register. Compare.
- **Output buffer**: Compare bytes written so far.
- **Control flow**: Both should be at the same block/op. If the interpreter
  takes a branch but the JIT doesn't, that's a divergence.

### 3. LLDB Integration

Two options:

**Option A: LLDB MCP tools (already available)**
- Use `lldb_start`, `lldb_command` to drive LLDB
- Step with `thread step-inst` until DWARF line advances
- Read registers with `register read`
- Parse output text

**Option B: Direct LLDB Python API**
- Write a Python script that LLDB sources
- Programmatic access to registers, memory, stepping
- Faster, no text parsing
- Could be a standalone tool or integrated into `kajit` CLI

Recommendation: Start with Option A (MCP tools) for prototyping, move to
Option B for production use.

### 4. Controller Modes

**`kajit debug-diff <format> <type> <input-hex>`**

- Compile with DWARF (`KAJIT_DEBUG=1`)
- Start interpreter session
- Attach LLDB to JIT function
- Step both in lockstep
- Print divergence report:

```
DIVERGENCE at OpId Inst(42) in block b7
  CFG-MIR: v23 = BinOp Shl v19, v20
  Interpreter: v23 = 896
  JIT (x3):   v23 = 0
  ─────
  v19: interpreter=7, JIT(x1)=7  ✓
  v20: interpreter=7, JIT(x2)=7  ✓
  v23: interpreter=896, JIT(x3)=0  ✗ ← FIRST DIVERGENCE
```

**`kajit debug-diff --reduce <format> <type> <input-hex>`**

- Run the lockstep debugger
- Capture the divergence witness (OpId, vreg, expected vs actual)
- Feed into the CFG-MIR reducer with predicate: "does this specific vreg
  still diverge at this op?"
- Produce minimal reproducer

### 5. Trace Mode

**`kajit compile <format> <type> -s trace --input <hex>`**

Non-interactive mode: run both, dump full trace to file:

```
OpId        | Block | Op                        | v23 (interp) | v23 (JIT/x3) | match
Inst(40)    | b7    | v19 = peek_byte           | 0x80          | 0x80          | ✓
Inst(41)    | b7    | v20 = const(7)            | 7             | 7             | ✓
Inst(42)    | b7    | v23 = Shl v19, v20        | 896           | 0             | ✗ ← STOP
```

## Integration with Existing Infrastructure

- **MCP debugger** (`kajit mcp`): Already has `session_step`, `session_state`,
  `session_inspect_vreg`. The lockstep controller can use these.
- **LLDB MCP tools**: Already available as `lldb_start`, `lldb_command`, etc.
- **DWARF emission** (`jit_dwarf.rs`, `jit_debug.rs`): Already maps OpId →
  DWARF line numbers.
- **Regalloc allocation map** (`regalloc3_result.rs`): Has the vreg → preg
  mapping, needs a query API.
- **Differential harness** (`regalloc_engine.rs`): Existing interpreter vs
  simulator check. The lockstep debugger supersedes this with instruction-level
  granularity.

## Consultant Recommendation (2026-03-25)

**Primary path:** Standalone child process + LLDB SB API (Rust bindings).
**Secondary path:** Child-process true JIT + LLDB JITLoaderGDB.
**Fallback:** Unicorn for deterministic instruction tracing.
**Skip:** QEMU (unnecessary on Apple Silicon), custom ptrace/Mach (too much work).

Key insight: Use `step_over_until(frame, file, line)` or `RunToAddress` to
efficiently advance to the next OpId boundary, rather than single-stepping
every instruction and re-checking the DWARF line.

## Technology Stack

### Rust LLDB Bindings (`lldb` crate)

Use the `lldb` crate (https://lib.rs/crates/lldb, v0.0.12, Dec 2024) which
wraps the LLDB SB API. May need to vendor + upgrade `lldb` + `lldb-sys` for
latest LLDB compatibility.

Key types and their roles in the lockstep debugger:

```rust
// Setup
let debugger = SBDebugger::create(false);         // no source manager
let target = debugger.create_target(&harness_path); // standalone executable
let bp = target.breakpoint_create_by_name("jit_entry");
let process = target.launch(&launch_info);         // launch child

// Per-OpId lockstep loop
let thread = process.selected_thread();
let frame = thread.selected_frame();

// Step to next OpId boundary (DWARF line = OpId index)
thread.step_over_until(&frame, &file_spec, next_dwarf_line)?;

// Read physical registers
let x3 = frame.find_register("x3").unwrap();
let x3_val: u64 = x3.value_as_unsigned(0);

// Read stack slot (for spilled vregs)
let mut buf = [0u8; 8];
process.read_memory(sp + spill_offset, &mut buf)?;

// Get current DWARF line → OpId
let line_entry = frame.line_entry().unwrap();
let dwarf_line = line_entry.line();  // maps to OpId
```

### Standalone Test Harness

Phase 1 generates a standalone Mach-O executable per test case:

```
┌────────────────────────────────────────┐
│  Generated test harness binary         │
│                                        │
│  .text:                                │
│    _main:                              │
│      set up input buffer from argv     │
│      call jit_entry                    │
│      write output to stdout            │
│      exit                              │
│                                        │
│  .text (JIT):                          │
│    jit_entry:                          │
│      <JIT-compiled decoder code>       │
│                                        │
│  .debug_line:  OpId → DWARF lines      │
│  .debug_info:  CU + function info      │
│  .debug_abbrev                         │
└────────────────────────────────────────┘
```

This avoids the complexity of in-process JIT debugging. LLDB gets a normal
binary with standard DWARF, no JITLoaderGDB needed.

### DWARF Line → OpId Index

We already emit DWARF line info mapping each machine instruction to an OpId.
The lockstep controller precomputes an index:

```rust
struct OpIdAddressMap {
    /// Sorted by address. Each entry = first machine address for an OpId.
    entries: Vec<(u64, OpId)>,
}

impl OpIdAddressMap {
    /// Given a DWARF line number, return the OpId.
    fn op_id_for_line(&self, line: u32) -> Option<OpId>;

    /// Given an OpId, return the start address of the next OpId.
    /// Used for RunToAddress / step_over_until.
    fn next_op_address(&self, current: OpId) -> Option<u64>;
}
```

### Vreg → Physical Location Map

The regalloc3 allocator produces vreg assignments. We need to expose:

```rust
enum PhysicalLocation {
    Register(u8),       // aarch64 GPR index (0=x0, 1=x1, ...)
    StackSlot(i32),     // offset from SP
}

struct AllocationMap {
    /// For each (OpId, VReg), where is the value after this op executes?
    assignments: HashMap<(OpId, VReg), PhysicalLocation>,
}
```

This already exists implicitly in the regalloc3 result — it needs to be
extracted and serialized into the harness binary (or a sidecar file).

## Implementation Plan

### Phase 1: `exec` stage (DONE)
- [x] Add `-s exec` to `kajit compile` that runs JIT on test input
- [x] Compare with interpreter output, byte-level diff
- [x] Confirmed unrolled u32 varint bug: JIT=0, interpreter=128

### Phase 2: Standalone harness generation
- Generate a Mach-O executable with embedded JIT code + DWARF
- Harness reads input from argv, runs the JIT function, writes output to stdout
- Can be run standalone: `./harness 8001` → prints output hex
- Can be debugged normally: `lldb ./harness -- 8001`

### Phase 3: Allocation map export
- Extract vreg → preg/stack mapping from regalloc3 result
- Serialize as a sidecar JSON file alongside the harness
- Format: `{ "Inst(42)": { "v23": "x3", "v19": "x1" }, ... }`

### Phase 4: Lockstep controller (`kajit debug-diff`)
- Use `lldb` crate to launch harness as child process
- Set breakpoint at `jit_entry`
- Step interpreter one OpId, step LLDB to matching DWARF line
- Read registers, map to vregs via allocation map
- Compare, stop on first divergence
- Print report:
  ```
  DIVERGENCE at Inst(42) in b7: v23 = Shl v19, v20
    interpreter: v23 = 896
    JIT (x3):   v23 = 0
    context: v19=7 (x1, ✓), v20=7 (x2, ✓)
  ```

### Phase 5: Reducer integration
- Use divergence as the predicate for CFG-MIR reduction
- Predicate: "does this CFG still diverge at any op?"
- Produces minimal CFG that triggers the specific bug

### Phase 6: Unicorn fallback (optional)
- In-process aarch64 emulation for instruction-accurate tracing
- Hook every instruction, full register dump
- Useful when LLDB stepping is too coarse or for CI without LLDB

## Why This Matters

The current debugging flow for JIT bugs is:
1. Dump assembly
2. Stare at 200+ instructions
3. Manually trace register values
4. Guess where it went wrong

With the lockstep debugger:
1. Run `kajit debug-diff postcard u32 8001`
2. Get: "v23 diverges at Inst(42), Shl should produce 896 but JIT gives 0"
3. Look at one instruction
4. Fix the bug

For the current unrolled u32 varint bug (JIT returns 0 instead of 128), this
would instantly pinpoint which shift/accumulate operation across the unrolled
iterations is being lost.
