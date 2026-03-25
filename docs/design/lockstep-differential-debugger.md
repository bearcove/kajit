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

## Implementation Plan

### Phase 1: `exec` stage (immediate)
- Add `-s exec` to `kajit compile` that runs JIT on test input and prints result
- Compare with interpreter output
- No LLDB, just "do they agree?"

### Phase 2: Allocation map query API
- Expose `(OpId, VReg) → PhysicalLocation` from regalloc3 result
- Serialize alongside DWARF info

### Phase 3: Lockstep prototype (MCP-based)
- Controller that alternates between interpreter MCP and LLDB MCP
- Steps both, reads state, compares
- Text-based output of divergence

### Phase 4: Integrated `debug-diff` command
- Single CLI command that sets up everything
- Automatic DWARF compilation, LLDB attachment, interpreter setup
- Reducer integration for automated minimization

### Phase 5: Trace mode + Python LLDB integration
- Full trace dump for offline analysis
- Python LLDB script for faster stepping (no MCP text overhead)
- Integration with the CFG-MIR reducer's predicate system

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

For the current unrolled u32 varint bug (JIT returns 1 instead of 128), this
would instantly pinpoint which shift/accumulate operation across the unrolled
iterations is being lost.
