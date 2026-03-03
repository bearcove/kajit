# kajit specification

kajit yearns for fast deserialization in Rust, based on JIT compilation of
serialization/deserialization code based on [facet] reflection.

[facet]: https://github.com/facet-rs/facet

## Context

kajit uses facet's reflection system (proc macros exposing type information: layout,
fields, vtables) as the basis for all serialization and deserialization.

kajit emits machine code at runtime via dynasmrt, caching compiled code per
(type, format, direction) combination.

kajit targets at minimum JSON and postcard as format crates.

## API surface

r[api.compile]
kajit exposes a `compile_deser` function that takes a `&'static Shape` and a format
implementation, and returns a compiled deserializer.

r[api.input]
All inputs are `&[u8]` representing a complete document (no streaming).

r[api.output]
The compiled deserializer writes into a caller-provided `MaybeUninit<T>`.

r[api.cache]
Compiled deserializers and serializers are cached per `(Shape, Format,
Direction)` triple. Subsequent calls with the same triple return the cached
compiled function.

r[api.cache.thread-safety]
The compilation cache is thread-safe: concurrent compilations for the same
key produce a single compiled function; concurrent compilations for different
keys proceed in parallel.

```rust
use facet::Facet;
// this implements a `kajit` trait
use kajit_json::KajitJson;

// A random type that happens to derive Facet
#[derive(Facet)]
struct MyDocument {
    users: Vec<User>,
    birthdays: HashMap<u32, String>,
    // etc.
}

let deser = kajit::compile_deser(<MyDocument as Facet>::SHAPE, KajitJson)?;
let doc = MaybeUninit::<MyDocument>::uninit();

unsafe { deser.call(&mut doc, r#"{ "users": [], "birthdays": {} }"#.as_bytes()); };
let doc = unsafe { doc.assume_init() };
// doc.users is len 0, etc.
```

## Shape intelligence vs format intelligence

The compiler combines shape intelligence (field names, types, layouts, offsets) with
format intelligence (wire encoding rules) to emit specialized machine code.

For example, let's say our input shape is:

```rust
struct Friend {
    age: u32,
    name: String,
}
```

We know the name, type, layout, etc. of both fields. We know how to allocate
such a value, how to partially construct it, how to check that it's been fully
constructed, how to drop it etc. etc. That's the shape intelligence.

When it comes to JSON, we know that...

```json
{
  "name": "Didier",
  "age": 432
}
```

...objects are delimited by `{` and `}`, that keys are always strings
(double-quoted), keys are separated from values by colons, etc. etc. This is
format intelligence.

On the other hand, when it comes to
[postcard](https://postcard.jamesmunns.com/), we know that keys come in order,
have no 'name' (they're not maps, they're known lists of fields, conceptually):
postcard is non-self-describing.

kajit generates machine code at runtime via [dynasmrt](https://crates.io/crates/dynasmrt).

## Intermediate representation

r[ir]
kajit compiles shapes through an intermediate representation before emitting
machine code. The pipeline is:

```
Shape tree → Format::lower() → RVSDG (per shape) → [passes] → Linearizer → Backend → machine code
```

Formats produce IR. The compiler decides inlining. The backend decides
instructions. These concerns are fully separated.

### RVSDG

r[ir.rvsdg]
The IR is a Regionalized Value State Dependence Graph (RVSDG). Values flow
through data edges. Effects (cursor movement, output writes, intrinsic calls)
are ordered by state edges. Control flow is represented by structured region
nodes, not a CFG.

r[ir.rvsdg.regions]
A region is an ordered set of nodes with explicit input and output ports.
Every region has a set of argument ports (values entering the region) and
result ports (values leaving the region). Regions nest — a node inside a
region may itself contain sub-regions.

r[ir.rvsdg.ports]
Nodes communicate through ports. Each port carries either a data value
(a VReg) or a state token. An output port of one node connects to an input
port of another node via an edge. Every input port has exactly one source
edge. Output ports may have zero or more consumers.

### Node types

r[ir.rvsdg.nodes.simple]
A simple node represents a single operation (an `IrOp`). It has typed input
ports and output ports. Pure ops have only data ports. Effectful ops consume
and produce state tokens to enforce ordering.

r[ir.rvsdg.nodes.gamma]
A gamma node represents a conditional. It has a predicate input, N
sub-regions (one per branch), and output ports that merge the results.
Each sub-region receives the same set of passthrough inputs and produces
the same number of outputs. At runtime, exactly one region executes based
on the predicate.

Untagged enum dispatch, externally tagged enum dispatch, and option
deserialization all lower to gamma nodes.

r[ir.rvsdg.nodes.theta]
A theta node represents a tail-controlled loop. It has a single body region
with a loop predicate output. The body region's non-predicate outputs feed
back as inputs for the next iteration. When the predicate is false, the loop
exits and the outputs flow to the theta node's output ports.

Vec/array element loops, JSON field-dispatch loops, and varint byte loops
all lower to theta nodes.

r[ir.rvsdg.nodes.lambda]
A lambda node represents a function. It contains a single body region.
Each shape that requires its own emitted function (recursive types,
non-inlined nested structs) gets a lambda node. The compiler produces
one lambda per unique shape, same as today.

r[ir.rvsdg.nodes.apply]
An apply node calls a lambda. It passes arguments (out pointer, state
tokens) and receives results. This is how inter-shape calls are represented
before inlining.

### Edges and state tokens

r[ir.edges.data]
Data edges carry values between ports. A data edge from node A's output
to node B's input means B uses the value A produced. Data edges impose no
ordering — if two nodes have no transitive data or state dependency, they
are independent and may be scheduled in either order.

r[ir.edges.state]
State edges carry state tokens between ports. A state edge from node A to
node B means A's effect must complete before B's effect begins. State tokens
are not values — they carry no runtime data. They exist solely to order
effects in the graph.

r[ir.edges.state.cursor]
Cursor state tokens order operations on the input cursor: reads, advances,
bounds checks, saves, and restores. Any two ops that touch cursor state are
connected by a cursor state edge (directly or transitively).

r[ir.edges.state.output]
Output state tokens order writes to the output struct. Writes to
non-overlapping field offsets are independent and need not be ordered.
Writes to the same offset (e.g., enum variant setup followed by field
writes) must be ordered.

r[ir.edges.state.barrier]
Intrinsic calls are full barriers: they consume all live state tokens
(cursor + output) and produce fresh ones. This is because intrinsics may
read/write cursor state and output memory through the context pointer.

### Op vocabulary

r[ir.ops]
The op vocabulary captures deserialization and serialization semantics.
Ops are format-agnostic where possible (e.g., `WriteToField` is the same
for postcard and JSON). Format-specific behavior lives in how formats
compose ops, not in the ops themselves.

r[ir.ops.cursor]
Cursor ops read from or advance the input cursor:

- `ReadBytes { dst, count }` — read N bytes from cursor into dst, advance.
  Consumes and produces cursor state.
- `PeekByte { dst }` — read one byte without advancing. Consumes and
  produces cursor state.
- `AdvanceCursor { count }` — skip N bytes. Consumes and produces cursor
  state.
- `BoundsCheck { count }` — assert N bytes remain. Consumes and produces
  cursor state. On failure, triggers an error exit.
- `SaveCursor { dst }` — snapshot cursor position into a data output.
  Consumes cursor state, produces cursor state + data output.
- `RestoreCursor { src }` — restore cursor from a saved snapshot.
  Consumes cursor state + data input, produces cursor state.

r[ir.ops.output]
Output ops write to the output struct:

- `WriteToField { src, offset, width }` — write src to out+offset.
  Consumes and produces output state.
- `ReadFromField { dst, offset, width }` — read from out+offset into dst.
  Consumes and produces output state (ordered relative to writes).

r[ir.ops.stack]
Stack ops use abstract stack slots for scratch space:

- `WriteToSlot { src, slot }` — write to a stack slot.
- `ReadFromSlot { dst, slot }` — read from a stack slot.

Stack slots are abstract — the backend assigns frame offsets.

r[ir.ops.arithmetic]
Pure arithmetic ops have no state edges:

- `Const { dst, value }` — load an immediate.
- `Add`, `Sub`, `And`, `Or`, `Shr`, `Shl`, `Xor`, `CmpNe` — binary ops on VRegs.
- `ZigzagDecode { dst, src, wide }` — zigzag decode (postcard signed ints).
- `SignExtend { dst, src, from_width }` — sign-extend narrow values.

These ops can float freely within their containing region. The linearizer
schedules them at the latest point before their first consumer.

r[ir.ops.call]
Call ops invoke functions:

- `CallIntrinsic { func, args, dst }` — call an `extern "C"` intrinsic.
  Full barrier: consumes all state tokens, produces fresh ones.
- `CallPure { func, args, dst }` — call a pure function (no side effects).
  No state edges.
- `Apply { lambda, args, dst }` — call another compiled shape. Consumes
  and produces cursor + output state.

r[ir.ops.error]
Error ops signal failure:

- `ErrorExit { code }` — set the error code and abort the current
  function. In the RVSDG, this terminates the containing region abnormally.
  The linearizer emits it as a write to the context error fields followed
  by a branch to the function's error cleanup path.

r[ir.ops.simd]
SIMD ops are opaque blocks that each backend implements natively:

- `SimdStringScan { found_quote, found_escape, unterminated }` — scan
  16 bytes at a time for `"` or `\`. Uses NEON `cmeq`/`umaxv` on aarch64,
  SSE2 `pcmpeqb`/`pmovmskb` on x86_64.
- `SimdWhitespaceSkip` — skip whitespace bytes using SIMD.

These are not decomposed into scalar IR ops. The vectorized
implementations are tightly coupled sequences of 30–40 platform-specific
instructions. Representing them as scalar ops would require vector types,
lane operations, and mask operations — complexity that pays off only when
there are 3+ backends.

### Effect classification

r[ir.effects]
Every op is classified by its effects. This classification determines
which state edges are required:

- **Pure**: no side effects. Data edges only. Can be reordered, CSE'd,
  DCE'd freely. Examples: `Const`, `Add`, `CmpNe`, `ZigzagDecode`.
- **Cursor**: reads or modifies input cursor state. Ordered relative to
  other cursor ops via cursor state edges. Examples: `ReadBytes`,
  `BoundsCheck`, `AdvanceCursor`.
- **Output**: writes to the output struct. Ordered relative to other
  output ops via output state edges. Examples: `WriteToField`.
- **Barrier**: may touch any state. Consumes and produces all state
  tokens. Examples: `CallIntrinsic`, `Apply`.

r[ir.effects.independence]
Ops with disjoint effect sets are independent and need not be ordered
relative to each other. A pure op and a cursor op have no state edge
between them (unless the pure op consumes the cursor op's data output).
An output write to offset 0 and an output write to offset 8 are
independent (different memory, no aliasing).

### Virtual registers and stack slots

r[ir.vregs]
Values are named by virtual registers (VRegs). VRegs are unlimited — each
op that produces a value gets a fresh VReg. The backend maps VRegs to
physical registers or spill slots. There is no register allocator in the
IR layer.

r[ir.slots]
Stack scratch space is named by abstract stack slots. The backend assigns
each slot a frame offset. Formats request slots for scratch space
(e.g., JSON needs slots for key pointer, key length, bitset, saved cursor).

### Cursor model

r[ir.cursor]
The input cursor (current read position and end-of-input) is implicit
mutable state, not an explicit value threaded through the graph. Cursor
ops consume and produce cursor state tokens to enforce ordering, but the
cursor "value" is not a VReg — it lives in a dedicated register at
runtime.

r[ir.cursor.register]
The backend pins the cursor to callee-saved registers (x19/r12 for
position, x20/r13 for end). Cursor state tokens in the RVSDG correspond
to "the cursor is in a valid state at this point." The linearizer ensures
cursor ops emit in state-edge order, which the backend translates to
sequential register operations.

r[ir.cursor.flush]
Before barrier ops (intrinsic calls), the backend flushes the cursor
register to the context struct. After the call returns, it reloads the
cursor register. The IR does not represent flush/reload — the backend
inserts them around every barrier op.

r[ir.cursor.snapshot]
Backtracking (for untagged enums, speculative parsing) uses
`SaveCursor`/`RestoreCursor` ops. `SaveCursor` captures the cursor
position as a data value (a VReg). `RestoreCursor` sets the cursor
back to a previously saved position. These are explicit in the RVSDG
and visible to optimization passes.

### Error model

r[ir.error]
Errors use a branch-to-exit model. Fallible ops (bounds checks, intrinsic
calls) may trigger an error exit: the error code and byte offset are
written to the context struct, and control transfers to the function's
error cleanup path.

r[ir.error.in-rvsdg]
In the RVSDG, error exits are modeled as abnormal region termination.
A fallible op's cursor state output is only valid on success. If the op
fails at runtime, the function exits without executing any downstream
nodes. The RVSDG does not represent the error path explicitly — the
linearizer generates it.

r[ir.error.cleanup]
The error cleanup path restores callee-saved registers and returns to the
caller. For ops that have allocated resources (Vec buffers, String
allocations), the cleanup path frees them before returning. The linearizer
emits cleanup code at the end of each lambda.

### Format trait

r[ir.format-trait]
Format crates implement a trait that the compiler calls to lower
format-specific operations into RVSDG nodes. Formats produce IR — they
do not emit machine code.

r[ir.format-trait.lower]
The Format trait provides lowering methods that take a mutable reference
to a region builder and produce nodes within that region:

- `lower_struct_fields(builder, fields)` — produce nodes for the field
  iteration logic. Positional formats (postcard) emit sequential reads.
  Keyed formats (JSON) emit a theta node containing key dispatch.
- `lower_read_scalar(builder, scalar_type)` — produce nodes to read a
  scalar value from the input. Returns a data output port.
- `lower_read_string(builder)` — produce nodes to read a string from the
  input, allocate it, and return a data output port.

r[ir.format-trait.stateless]
Format trait implementations are stateless at JIT-compile time: they
produce IR nodes but hold no mutable state between calls. Runtime state
lives in the `format_state` pointer inside `DeserContext`.

### Inlining

r[ir.inline]
Inlining is an IR-level decision, not a format-level decision. The
compiler can replace an `Apply` node with the callee lambda's body
region, remapping input/output ports. This decouples "what to compute"
from "whether to inline."

r[ir.inline.decision]
The inlining pass considers: node count in the callee, number of call
sites, whether the type is recursive (back-edges are never inlined),
and format hints (positional formats benefit more from inlining because
their sequential reads fuse with the caller's reads).

r[ir.inline.remap]
Inlining remaps: VRegs are offset by the caller's vreg count, stack slots
are offset similarly, and `WriteToField` offsets are adjusted by the
field offset in the parent struct. State edges are spliced: the caller's
cursor state before the `Apply` feeds into the inlined body's first
cursor op, and the inlined body's final cursor state feeds the caller's
next cursor op.

### Linearization

r[ir.linearize]
The linearizer converts the RVSDG into a linear instruction sequence
suitable for the backend. It walks the graph recursively:

- **Simple nodes**: topological sort within each region, respecting state
  edges. Pure nodes are scheduled at the latest point before their first
  consumer (to minimize register pressure).
- **Gamma nodes**: emit as a conditional branch — evaluate predicate,
  branch to each region's code, merge outputs.
- **Theta nodes**: emit as a loop — body code, evaluate predicate,
  conditional back-edge.
- **Lambda nodes**: emit as a function with prologue/epilogue.

r[ir.linearize.error-paths]
Error exits are placed at the end of each function, after all
success-path code. This keeps the hot path straight-line and
branch-free where possible.

r[ir.linearize.schedule]
Within a region, independent nodes (no transitive state or data
dependency) may be scheduled in any order. The linearizer uses a
reverse-post-order traversal of the data/state dependency graph.
This produces good instruction locality without a full scheduling
algorithm.

### Backends

r[ir.backends]
kajit supports two backends — aarch64 and x86_64 — selected at compile
time via `#[cfg(target_arch)]`. Both backends consume the same linearized
IR and produce native machine code via dynasmrt.

r[ir.backends.native-only]
The emitted code is always native machine code. There is no interpreter
fallback. dynasmrt handles label management, relocation, memory
protection (RW→RX), and cache flushing on aarch64.

r[ir.backends.cursor-registers]
The backend pins the cursor to fixed callee-saved registers and inserts
flush/reload sequences around barrier ops. The mapping is:

- **aarch64**: x19=cursor, x20=end, x21=out, x22=ctx
- **x86_64**: r12=cursor, r13=end, r14=out, r15=ctx

r[ir.backends.simd]
Each backend implements SIMD ops natively. The linearized IR contains
opaque SIMD op references; the backend expands them into platform-specific
instruction sequences.

r[ir.backends.post-regalloc.branch-test]
Post-regalloc backend lowering should prefer direct compare/test+branch forms
from allocated operands and avoid unnecessary boolean materialization when
control-flow semantics are equivalent.

r[ir.backends.post-regalloc.shuffle]
Post-regalloc backend lowering should avoid unnecessary fixed scratch-register
shuttling when source and destination allocations can be used directly.

### Register allocation

r[ir.regalloc]
Backends must use CFG-aware whole-function register allocation for hot-path
code generation. Register location decisions must remain valid across control
flow edges and joins without relying on blanket boundary flushes.

r[ir.regalloc.ra-mir]
Before machine emission, linearized IR is lowered to an allocator-oriented
machine IR (RA-MIR) with explicit basic blocks and instruction order.

r[ir.regalloc.ra-mir.operands]
Each RA-MIR instruction declares virtual-register uses/defs, register class
requirements (at minimum GPR vs SIMD), and any fixed-register constraints.

r[ir.regalloc.ra-mir.block-params]
Control-flow merges are represented with block parameters and branch
arguments, so loop-carried and join-carried values are explicit allocator
inputs rather than implicit stack state.

r[ir.regalloc.ra-mir.calls]
Call instructions declare ABI constraints: fixed argument/return registers
and clobbered registers. Values live across calls must be placed in locations
that satisfy these clobber rules.

r[ir.regalloc.engine]
The register allocation engine consumes RA-MIR and returns final locations
plus edit operations (moves, spills, reloads) required to satisfy constraints.

r[ir.regalloc.edits]
Machine emission applies allocator-provided edits at the exact program points
required by the allocation result; spill placement is allocator-directed, not
implemented as unconditional boundary spilling.

r[ir.regalloc.edits.minimize]
Machine emission should drop no-op/self edit moves and avoid unnecessary
boundary/edit trampoline traffic when equivalent in-place edge edit application
is available.

r[ir.regalloc.no-boundary-flush]
Steady-state hot-path code generation must not depend on unconditional
flush-all behavior at every branch, label, or call boundary. Any required
materialization must be justified by liveness and clobber constraints.

r[ir.regalloc.checker]
The allocation pipeline includes checker-backed validation in test/debug
configuration to ensure allocated code preserves virtual-register semantics.

r[ir.regalloc.regressions]
Regression coverage includes loop-heavy postcard deserialization paths
(notably medium/large scalar `Vec` decoding) and checks IR output parity
against legacy backend and serde reference decoding.

### Intrinsics

r[ir.intrinsics]
Operations too complex to represent as IR nodes (allocation, string
growth, UTF-8 validation, complex number parsing) are implemented as
Rust functions with `extern "C"` ABI. The IR represents these as
`CallIntrinsic` nodes. The backend emits calls using the platform's
C calling convention.

r[ir.intrinsics.representable-inline]
Operations representable in the core IR vocabulary must not be modeled as
format-specific intrinsic calls. They must be lowered to generic IR nodes so
that optimization and backend logic remain format-agnostic.

r[ir.varint.lowering]
Postcard varint decoding is represented in RVSDG using generic nodes:
cursor ops (`ReadBytes`, `BoundsCheck`), arithmetic ops (`And`, `Or`, `Shl`,
comparisons), and structured control (`Theta`/`Gamma`) for the byte loop and
termination checks. There is no postcard-specific varint opcode.

r[ir.varint.errors]
The varint lowering handles all error cases inside IR control flow:
unexpected EOF, malformed/overlong varint, and numeric narrowing overflow.
Errors feed the standard IR error model and exit path.

### Optimization passes

r[ir.passes]
The compiler starts with zero optimization passes — RVSDG is lowered
to a linear sequence and emitted directly. Passes are added one at a
time as the need arises.

r[ir.passes.planned]
Planned passes, in rough priority order:

1. **Bounds check coalescing**: merge adjacent `BoundsCheck` nodes within
   a region into a single check for the combined byte count. Enabled by
   state edge analysis — two bounds checks with only cursor reads between
   them can be merged.
2. **Inlining**: replace `Apply` nodes with callee lambda bodies.
3. **Dead node elimination**: remove nodes with no consumers. Trivial in
   RVSDG — a node with no outgoing data or state edges is dead.
4. **Common subexpression elimination**: merge nodes with identical ops
   and identical input edges. Pure nodes are always candidates. Cursor
   ops are candidates only if they have the same state predecessor.
5. **Cold path sinking**: move error-construction nodes into error-only
   regions so they don't pollute the hot path's register pressure.
6. **Global register allocation**: lower linearized IR to a CFG-oriented
   allocator input form, run a whole-function allocator, and apply
   allocator-provided move/spill/reload edits during machine emission.

r[ir.passes.pre-regalloc.coalescing]
Pre-regalloc shaping may coalesce copies across gamma/theta boundaries when
value equivalence is proven and block-parameter/control-flow semantics are
preserved.

r[ir.passes.pre-regalloc.loop-invariants]
Pre-regalloc shaping may hoist loop-invariant setup out of theta bodies when
the motion is side-effect safe and preserves data/state dependencies.

r[ir.passes.lower-to-core]
Normalization passes may rewrite format-lowering helper forms into the core
IR vocabulary before linearization. This includes replacing placeholder calls
with explicit loop/control/data subgraphs when the behavior is representable
as core IR.

### Pipeline toggles (`KAJIT_OPTS`)

r[compiler.opts]
The compiler pipeline supports runtime toggles via the `KAJIT_OPTS`
environment variable for fast pass-level bisecting without rebuilding.

r[compiler.opts.syntax]
`KAJIT_OPTS` is a comma-separated list of option tokens. Each token is
`+name` (enable) or `-name` (disable). Unknown or malformed tokens are errors.

r[compiler.opts.defaults]
If `KAJIT_OPTS` is unset or empty, pipeline behavior remains unchanged from
the entrypoint defaults (options act as explicit overrides only).

r[compiler.opts.all-opts]
Option `all_opts` controls whether default RVSDG optimization passes run
before linearization in compile paths that use the default pipeline.

r[compiler.opts.regalloc]
Option `regalloc` controls whether regalloc edit application is enabled
during machine emission. When disabled, backend emission must skip applying
regalloc instruction/edge edits while keeping deterministic behavior.

r[compiler.opts.composition]
Multiple option tokens compose left-to-right; for repeated option names, the
last token wins.

r[compiler.opts.pass-registry]
Default IR passes are individually addressable by stable option names and
descriptions. Each pass can be toggled independently through `KAJIT_OPTS`.

r[compiler.opts.help]
When `KAJIT_OPTS=help`, compilation fails fast with a deterministic help
message describing syntax, top-level options, and per-pass toggle names.

r[compiler.opts.api]
Pipeline option enable/disable controls are available through a public API so
callers can configure compile behavior without relying on process environment
variables.

r[compiler.opts.invalid]
On invalid `KAJIT_OPTS` content, compilation fails with a clear error that
identifies the offending token and lists supported option names.

## Scalar types

r[scalar.types]
kajit supports deserialization and serialization of all Rust integer types
(u8, u16, u32, u64, u128, i8, i16, i32, i64, i128), floating-point types
(f32, f64), bool, and char.

r[scalar.overflow]
When a wire value does not fit in the target type (e.g., 256 into u8, or a
negative value into an unsigned type), the deserializer reports an overflow
error with the byte offset.

### JSON scalars

r[deser.json.scalar.integer]
JSON integers are parsed from decimal text representation and range-checked
against the target integer type at runtime. Out-of-range values are an error.

r[deser.json.scalar.float]
JSON numbers are parsed into f64 (or f32 via f64 with a narrowing cast).
The f64 parser is emitted as inline machine code — no call to a Rust intrinsic
or standard library function. The emitted code covers the entire pipeline:
whitespace skip, sign detection, digit extraction, decimal-to-float conversion,
IEEE 754 packing, and store.

#### Digit extraction

r[deser.json.scalar.float.ws]
Leading whitespace (space, tab, newline, carriage return) is skipped before
the number.

r[deser.json.scalar.float.sign]
An optional leading `-` sets the sign bit. Bare `+` is not accepted (per JSON
spec).

r[deser.json.scalar.float.digits]
The emitted code extracts decimal digits byte-at-a-time from the input,
accumulating a mantissa `d` (u64), a digit count `nd`, and a fractional
digit count `frac_digits`. Leading zeros are skipped without incrementing `nd`.

r[deser.json.scalar.float.overflow-digits]
When the digit count exceeds 19 (the maximum number of decimal digits that
fit in a u64), additional digits are counted but not accumulated into the
mantissa. The count of dropped digits is tracked and added to the exponent.

r[deser.json.scalar.float.dot]
A decimal point `.` separates the integer and fractional parts. Digits after
the dot increment `frac_digits`. At least one digit must appear somewhere
in the number (before or after the dot).

r[deser.json.scalar.float.exponent]
An optional `e` or `E` introduces an exponent with optional sign (`+` or `-`)
and one or more decimal digits. The exponent is clamped to ±9999 to prevent
overflow in intermediate calculations.

r[deser.json.scalar.float.validation]
If no digits were consumed (e.g., bare `.`, bare `e`, bare `-`), the
deserializer reports `InvalidJsonNumber`.

#### Conversion

r[deser.json.scalar.float.zero]
If the mantissa `d` is zero, the result is ±0.0 (preserving the sign bit).

r[deser.json.scalar.float.exact-int]
If the number has no fractional part, no exponent, and the mantissa fits in
2^53, the result is produced by direct integer-to-float conversion (UCVTF on
aarch64, CVTSI2SD on x86_64). This avoids the full uscale pipeline for simple
integer literals that target f64 fields.

r[deser.json.scalar.float.uscale]
Non-trivial numbers are converted using the uscale algorithm (unrounded
scaling). The combined power-of-ten exponent `p` is computed as
`explicit_exponent - frac_digits + dropped_digits`. The algorithm performs:
1. Left-justify the mantissa: `x = d << (64 - clz(d))`.
2. Compute `log2(10^p)` via the integer approximation `(p * 108853) >> 15`.
3. Look up precomputed 128-bit power-of-five values `(pm_hi, pm_lo)` from a
   static table indexed by `p + 348`.
4. Compute the unrounded significand via one or two 64×64→128 widening
   multiplies, a shift, and sticky-bit collection.

r[deser.json.scalar.float.uscale.table]
The power-of-five table covers exponents from -348 to +347 (696 entries, each
16 bytes). The table base address is baked into the emitted code as an
immediate at JIT-compile time.

r[deser.json.scalar.float.uscale.mul128]
The widening multiply uses native instructions: `MUL`+`UMULH` on aarch64,
`MUL` (implicit `RDX:RAX`) on x86_64. A second multiply against `pm_lo` is
performed only when the sticky bit from the first multiply is ambiguous
(high bits of the result are all zeros under the shift mask).

r[deser.json.scalar.float.uscale.clz]
Leading-zero count uses the native `CLZ` instruction on aarch64 and `LZCNT`
on x86_64.

#### Packing

r[deser.json.scalar.float.pack]
The unrounded significand is converted to IEEE 754 binary64 via round-to-
nearest-even: `rounded = (u + 1 + ((u >> 2) & 1)) >> 2`. The biased exponent
is computed and combined with the significand to form the final 64-bit
representation. The sign bit is OR'd in last.

r[deser.json.scalar.float.pack.subnormal]
Subnormal results (biased exponent would be ≤ 0) are handled by keeping the
unrounded significand without an implicit leading bit.

r[deser.json.scalar.float.pack.overflow]
If `p > 347`, the result is ±infinity. If `p < -348`, the result is ±0.0.
If the biased exponent after rounding reaches 2047, the result is ±infinity.
These are not errors — they are valid IEEE 754 values.

r[deser.json.scalar.float.correctness]
The JIT f64 parser produces bit-identical results to Rust's
`str::parse::<f64>()` for all valid JSON number inputs. This is verified by
round-trip testing against a corpus of real-world coordinates (canada.json,
~111k floats) and edge cases (subnormals, max/min finite, large exponents,
19+ digit mantissas).

r[deser.json.scalar.bool]
JSON `true` and `false` literals are parsed into Rust `bool`. Any other
value is an error.

r[deser.json.scalar.char]
JSON single-character strings are parsed into Rust `char`. Multi-character
strings or non-string JSON values are an error.

r[ser.json.scalar.integer]
JSON integer serialization emits the decimal text representation.

r[ser.json.scalar.float]
JSON float serialization emits a decimal representation with enough precision
to round-trip.

r[ser.json.scalar.bool]
JSON bool serialization emits `true` or `false`.

### Postcard scalars

r[deser.postcard.scalar.varint]
Postcard encodes unsigned integers as LEB128 varints and signed integers as
ZigZag-encoded varints. The deserializer decodes the varint and stores the
result into the target type.

r[deser.postcard.scalar.varint.no-rust-call]
Postcard varint decode on the deserialization hot path is emitted as native
JIT code from IR, not as a call into a Rust varint helper.

r[deser.postcard.scalar.float]
Postcard encodes f32 as 4 little-endian bytes and f64 as 8 little-endian
bytes.

r[deser.postcard.scalar.bool]
Postcard encodes bool as a single byte: 0x00 = false, 0x01 = true. Any
other value is an error.

r[ser.postcard.scalar]
Postcard scalar serialization emits the corresponding varint or
fixed-width encoding.

## Strings

r[deser.json.string]
JSON string deserialization reads a double-quoted string, processes escape
sequences, and allocates the result as a Rust `String`.

r[deser.json.string.escape]
JSON string deserialization handles all JSON escape sequences: `\\`, `\/`,
`\"`, `\b`, `\f`, `\n`, `\r`, `\t`, and `\uXXXX` (including surrogate
pairs for non-BMP characters).

r[deser.json.string.utf8]
Invalid UTF-8 in JSON strings is reported as an error.

r[deser.postcard.string]
Postcard string deserialization reads a varint length followed by that many
UTF-8 bytes, and allocates a Rust `String`.

r[deser.string.allocation]
String deserialization allocates heap memory for the result. The allocation
is owned by the output struct.

r[ser.json.string]
JSON string serialization emits a double-quoted string with special
characters escaped according to the JSON specification.

r[ser.postcard.string]
Postcard string serialization emits a varint length followed by the UTF-8
bytes.

## The compiler

r[compiler.walk]
The compiler walks the shape tree at JIT-compile time and emits machine code
for each node. The recursion is in the host (Rust) code — the emitted code
is flat.

```rust
fn compile_deser(shape: &Shape, fmt: &dyn Format, code: &mut Code) {
    match shape.kind() {
        ShapeKind::Struct(fields) => {
            fmt.emit_expect_begin_object(code);
            // `build_field_dispatch` runs at JIT-compile time: all field names
            // are known statically, so we bake in a trie/phf over them.
            let dispatch = build_field_dispatch(fields);
            fmt.emit_field_loop(code, dispatch, |code, field| {
                // `field` is known at JIT-compile time; recurse into its shape.
                compile_deser(field.shape, fmt, code);
                emit_store(code, field.offset); // write result into struct slot
            });
            emit_assert_all_required_set(code, fields); // checks stack-allocated bitset
            fmt.emit_expect_end_object(code);
        }
        ShapeKind::Primitive(p) => fmt.emit_read_primitive(code, p),
        ShapeKind::Sequence(elem) => {
            fmt.emit_begin_array(code);
            fmt.emit_array_loop(code, |code| compile_deser(elem.shape, fmt, code));
            fmt.emit_end_array(code);
        }
        // ...
    }
}
```

Key points:

r[compiler.field-dispatch]
Field dispatch tables (trie, perfect hash, or sorted array) are built at
JIT-compile time from the known field names. The emitted code dispatches
runtime keys against these baked-in tables.

r[compiler.required-fields]
Required-field tracking uses a stack-allocated bitset sized
`ceil(field_count / 64)` bits, allocated at emitted-function entry.
The emitted code checks the bitset before the function returns.

## Recursive types

r[compiler.recursive]
When the compiler encounters a type that directly or indirectly contains itself,
it emits a call via a forward reference instead of recursing infinitely.
Forward references are patched to the correct address once the target function
is compiled.

Consider:

```rust
struct Node {
    value: i32,
    children: Vec<Node>,
}
```

`Node` contains `Vec<Node>`, which contains `Node`. If `compile_deser` naively
recurses into each field's shape, it loops forever at JIT-compile time — it
never bottoms out.

The fix is standard: track which shapes are currently being compiled, and
treat a back-edge as a call instead of an inline.

```rust
struct Compiler {
    fmt: Box<dyn Format>,
    code: Code,
    finished: HashMap<&'static Shape, FuncRef>,
    in_progress: HashSet<&'static Shape>,
    forward_refs: HashMap<&'static Shape, Vec<PatchSite>>,
}

impl Compiler {
    fn compile_deser(&mut self, shape: &'static Shape) {
        if let Some(func) = self.finished.get(shape) {
            // Already compiled: emit a direct call to the finished function.
            self.code.emit_call(*func);
            return;
        }

        if self.in_progress.contains(shape) {
            // Back-edge: emit a call via a forward reference; patch it later.
            let site = self.code.emit_call_fwd();
            self.forward_refs.entry(shape).or_default().push(site);
            return;
        }

        self.in_progress.insert(shape);
        let func_start = self.code.current_offset();

        match shape.kind() {
            ShapeKind::Struct(fields) => {
                self.fmt.emit_expect_begin_object(&mut self.code);
                let dispatch = build_field_dispatch(fields);
                // NOTE: fmt methods take &mut self.code separately to avoid
                // a simultaneous borrow of self through the closure.
                self.fmt.emit_field_loop(&mut self.code, dispatch, |code, field| {
                    self.compile_deser(field.shape); // may emit a call, not an inline
                    code.emit_store(field.offset);
                });
                self.code.emit_assert_all_required_set(fields);
                self.fmt.emit_expect_end_object(&mut self.code);
            }
            // ...
        }

        self.in_progress.remove(shape);
        let func = self.code.finish_func(func_start);
        // Fix up any forward-reference call sites that pointed here.
        if let Some(sites) = self.forward_refs.remove(shape) {
            for site in sites {
                self.code.patch(site, func);
            }
        }
        self.finished.insert(shape, func);
    }
}
```

r[compiler.recursive.one-func-per-shape]
Each unique shape produces at most one emitted function. If a shape has already
been compiled, the compiler emits a direct call to the finished function.
Mutual recursion is handled the same way as self-recursion.

The result for `Node` is two emitted functions:

- `deser_Node(out: *mut Node, input: &[u8])` — reads the struct, calls
  `deser_Vec_Node` for the `children` field.
- `deser_Vec_Node(out: *mut Vec<Node>, input: &[u8])` — allocates the vec,
  loops, calls `deser_Node` for each element.

They call each other exactly like hand-written Rust would.

## Nested structs

r[deser.nested-struct]
When a struct field's shape is itself a struct, the compiler emits code to
deserialize the inner struct in-place at its offset within the outer struct.
The strategy depends on the format's `supports_inline_nested()`.

r[deser.nested-struct.offset]
The emitted code passes `out + field_offset` as the `out` pointer for the
inner struct, so it is written directly into its slot in the outer struct's
memory layout.

r[deser.nested-struct.inline]
When `supports_inline_nested()` returns true, nested struct fields are
flattened into the parent function at JIT-compile time. The compiler collects
the inner struct's fields, adds the parent field's byte offset to each, and
emits them directly in the parent function body. No separate function or
inter-function call is emitted.

For example, `Person { name: String, age: u32, address: Address { city: String, zip: u32 } }`
emits a single function:

```
read_string(offset=0)    ; name
read_varint(offset=24)   ; age
read_string(offset=32)   ; address.city  (address_offset + 0)
read_varint(offset=56)   ; address.zip   (address_offset + 24)
```

r[deser.nested-struct.inline.recursive]
Inline flattening recurses: if an inlined struct itself contains nested
structs, those are also inlined, producing a single flat function for
arbitrarily deep nesting.

r[deser.nested-struct.call]
When `supports_inline_nested()` returns false, each unique struct shape gets
its own emitted function with a full prologue/epilogue. The parent calls the
nested function via
`r[callconv.inter-function]`. Shared inner types (e.g., `Address` used in
both `home` and `work` fields) are compiled once and called from multiple
sites, per `r[compiler.recursive.one-func-per-shape]`.

r[ser.nested-struct]
When serializing a struct field whose shape is itself a struct, the compiler
recursively emits a serializer that reads from `inp + field_offset`.

## Newtype wrappers and tuple structs

r[deser.newtype]
Newtype wrappers (single-field tuple structs like `struct Age(u32)`) are
deserialized transparently: the emitted code deserializes the inner type
directly into the newtype's memory, which has the same layout.

r[deser.json.tuple-struct]
JSON tuple structs with multiple fields are deserialized as JSON arrays.

r[deser.postcard.tuple-struct]
Postcard tuple structs are deserialized as their fields in declaration order,
same as named structs but without field names.

r[ser.newtype]
Newtype wrappers are serialized transparently: the emitted code serializes
the inner type directly.

r[ser.json.tuple-struct]
JSON tuple structs with multiple fields are serialized as JSON arrays.

## Transparent wrappers

r[deser.transparent]
`#[facet(transparent)]` on a struct causes it to be deserialized as its
single inner field's type. The compiler detects the attribute via
`shape.is_transparent()`, extracts the inner type from `shape.inner`, and
emits deserialization of the inner type at the wrapper field's offset.

r[deser.transparent.forwarding]
Transparent deserialization is purely structural forwarding — no struct
framing (no `{`/`}` for JSON, no field count for postcard). The wrapper
is invisible on the wire. This works identically for JSON and postcard.

r[deser.transparent.composite]
If the inner type is composite (struct, enum, vec, map), the compiler
pre-compiles it as a separate function and emits a call. If the inner type
is a scalar or string, the compiler emits the appropriate read inline.

r[ser.transparent]
`#[facet(transparent)]` structs are serialized as their inner type directly,
with no wrapper framing.

## emit_field_loop for postcard

r[deser.postcard.struct]
For postcard, struct deserialization emits a straight sequence of field
deserializers in declaration order — no keys, no framing, no dispatch, no
loop. The dispatch table is ignored.

```rust
impl Format for KajitPostcard {
    fn emit_field_loop(
        &self,
        code: &mut Code,
        dispatch: FieldDispatch, // ignored for postcard
        mut per_field: impl FnMut(&mut Code, &FieldInfo),
    ) {
        // No begin-object marker, no key reading, no branching.
        // Just emit each field deserializer in declaration order.
        for field in dispatch.fields_in_order() {
            per_field(code, field);
        }
    }
}
```

For `Friend { age: u32, name: String }` this produces (in pseudocode):

```
deser_Friend:
    deser_u32(out + offset_of(age))     ; read varint, store
    deser_String(out + offset_of(name)) ; read length-prefixed bytes, store
    ret
```

r[deser.postcard.struct.no-bitset]
Postcard structs require no required-field bitset. All fields are always
present — if any field is missing, the input runs out and the primitive
deserializer reports an error.

## emit_field_loop for JSON

r[deser.json.struct]
For JSON, struct deserialization emits a loop that reads `"key": value` pairs
in arbitrary order. The emitted code must:
1. Read a key from the input at runtime.
2. Dispatch to the right field deserializer based on that key.
3. Track which required fields have been seen (the bitset).
4. Loop until `}`.

r[deser.json.struct.trie]
JSON field dispatch uses a compile-time trie built from known field names.
The trie is baked into the emitted code as a branch tree — no heap allocation
at runtime.

```rust
impl Format for KajitJson {
    fn emit_field_loop(
        &self,
        code: &mut Code,
        dispatch: FieldDispatch,
        mut per_field: impl FnMut(&mut Code, &FieldInfo),
    ) {
        // Allocate a bitset on the emitted function's stack frame:
        // one bit per required field, cleared at function entry.
        let bitset = code.alloc_stack_bitset(dispatch.required_field_count());

        // Build a compile-time trie over the known field names.
        // The trie is baked into the emitted code as a branch tree —
        // no heap allocation at runtime.
        let trie = build_trie(dispatch.fields());

        let loop_label = code.begin_loop();

        // Skip whitespace, peek at next byte.
        // If `}`, break.
        self.emit_skip_whitespace(code);
        self.emit_break_if_end_object(code, loop_label);

        // Read the quoted key into a temporary &[u8] on the stack.
        let key_slot = self.emit_read_key(code);

        self.emit_expect_colon(code);

        // Emit the trie as a branch tree. Each leaf calls per_field for
        // that field and sets the corresponding bit in the bitset.
        // The default branch (unknown key) emits a skip-value call.
        emit_trie_dispatch(code, &trie, key_slot, |code, field| {
            per_field(code, field);
            code.emit_set_bit(bitset, field.required_index);
        });

        // Skip optional trailing comma.
        self.emit_skip_comma(code);

        code.end_loop(loop_label);

        // After the loop: assert all required fields were seen.
        code.emit_assert_bitset_full(bitset, dispatch.required_field_count());
    }
}
```

For `Friend { age: u32, name: String }` this produces (in pseudocode):

```
deser_Friend:
    bitset = 0b00          ; two required fields, neither seen yet
loop:
    skip_whitespace
    if peek() == '}': break
    key = read_quoted_key()
    expect_colon()
    if key == "age":        ; \
        deser_u32(&out.age) ;  trie branch for "age"
        bitset |= 0b01      ; /
    elif key == "name":        ; \
        deser_String(&out.name) ;  trie branch for "name"
        bitset |= 0b10          ; /
    else:
        skip_value()        ; unknown field
    skip_comma()
    goto loop
    assert bitset == 0b11  ; both fields present
    ret
```

r[deser.json.struct.unknown-keys]
Unknown keys in JSON objects are skipped via a `skip_value` call. They do not
cause an error (unless `deny_unknown_fields` is set).

## Field renaming

r[deser.rename]
`#[facet(rename = "wireName")]` on a field causes the deserializer to match
against the renamed string instead of the Rust field name. The rename is
applied at JIT-compile time: `field.effective_name()` returns the renamed
name, which is baked into the key-dispatch chain.

r[deser.rename.all]
`#[facet(rename_all = "camelCase")]` on a container applies a naming
convention to all fields. The derive macro computes each field's renamed
value at Rust compile time and stores it in the field metadata. The
JIT compiler sees the final renamed names via `effective_name()` — no
case conversion logic is needed in kajit itself.

r[deser.rename.json]
For JSON, renamed field names are used in the key-matching dispatch chain.
The emitted code compares runtime keys against the renamed strings.

r[deser.rename.postcard-irrelevant]
For postcard, field names are irrelevant (positional format). Rename
attributes have no effect on postcard deserialization.

r[ser.rename]
Renamed field names are used as JSON keys during serialization.

## Deny unknown fields

r[deser.deny-unknown-fields]
`#[facet(deny_unknown_fields)]` on a struct causes the deserializer to
report an error when an unrecognized key is encountered, instead of
silently skipping it.

r[deser.deny-unknown-fields.json]
For JSON, the unknown-key branch in the key-dispatch loop emits an error
(`ErrorCode::UnknownField`) instead of calling `skip_value`. The check is
a JIT-compile-time decision: the flag is read from `shape` metadata and
controls which code path is emitted.

r[deser.deny-unknown-fields.postcard-irrelevant]
For postcard, deny_unknown_fields has no effect (positional format, no
field names, no concept of unknown fields).

## Skip fields

r[deser.skip]
`#[facet(skip)]` and `#[facet(skip_deserializing)]` on a field cause the
deserializer to exclude that field from the wire format. The field is not
included in the key-dispatch table (JSON) or the positional field sequence
(postcard).

r[deser.skip.default-required]
Skipped fields must have a default value (`#[facet(default)]` or the
container has `#[facet(default)]`). If a field is skipped but has no
default source, the compiler panics at JIT-compile time.

r[deser.skip.init]
The emitted code initializes skipped fields to their default value at
function entry, before the field-dispatch loop. This ensures the field
has a valid value regardless of what appears in the input.

r[deser.skip.json]
For JSON, skipped fields are omitted from the key-dispatch chain. If the
input contains a key matching a skipped field, it is treated as an unknown
key (skipped or errored depending on `deny_unknown_fields`).

r[deser.skip.postcard]
For postcard, skipped fields are omitted from the positional field sequence.
The emitted code does not read any bytes for skipped fields — it only
writes their default value.

r[ser.skip]
`#[facet(skip)]` and `#[facet(skip_serializing)]` cause the serializer to
omit the field from the output.

### JSON parsing behaviors

r[deser.json.whitespace]
The JSON deserializer skips whitespace (space, tab, newline, carriage return)
between all tokens.

r[deser.json.null]
The JSON literal `null` is recognized and used for Option fields (init_none)
and as an acceptable value for unit enum variants.

r[deser.json.depth-limit]
The JSON deserializer enforces a maximum nesting depth to prevent stack
overflow from deeply nested inputs.

r[deser.json.duplicate-keys]
If a JSON object contains duplicate keys, the last value wins.

## Flatten

r[deser.flatten]
`#[facet(flatten)]` merges a nested struct's fields into the parent's key
namespace. The flattened fields appear at the same level in the wire format,
not nested under a key.

```rust
#[derive(Facet)]
struct Metadata {
    version: u32,
    author: String,
}

#[derive(Facet)]
struct Document {
    title: String,
    #[facet(flatten)]
    meta: Metadata,
}
```

Wire JSON:

```json
{ "title": "Hello", "version": 1, "author": "Amos" }
```

Not:

```json
{ "title": "Hello", "meta": { "version": 1, "author": "Amos" } }
```

### How build_field_dispatch handles it

r[deser.flatten.offset-accumulation]
`build_field_dispatch` walks the struct's fields at JIT-compile time. When it
encounters a flattened field, it recurses into the flattened shape and adds
its fields with offsets adjusted by the flattened field's offset within the
parent. The trie is built over the flat field list.

r[deser.flatten.inline]
No separate deserializer function is emitted for the flattened struct — its
fields are inlined into the parent's field loop.

```rust
fn build_field_dispatch(fields: &[FieldInfo]) -> FieldDispatch {
    let mut flat_fields = Vec::new();
    collect_fields(fields, 0, &mut flat_fields);
    FieldDispatch::new(flat_fields)
}

fn collect_fields(fields: &[FieldInfo], base_offset: usize, out: &mut Vec<FlatField>) {
    for field in fields {
        if field.is_flatten() {
            // Recurse into the flattened shape, accumulating the offset.
            collect_fields(field.shape.fields(), base_offset + field.offset, out);
        } else {
            out.push(FlatField {
                name: field.name,
                shape: field.shape,
                offset: base_offset + field.offset, // absolute offset from outer `out`
                required: !field.is_option(),
            });
        }
    }
}
```

The trie is then built over `flat_fields` exactly as before — `"title"`,
`"version"`, and `"author"` all appear as siblings. The store offsets point
directly into the right place within the outer struct's allocation:

```
deser_Document:
    bitset = 0b000         ; title, version, author — all required
loop:
    ...
    if key == "title":
        deser_String(out + offset_of(Document::title))
        bitset |= 0b001
    elif key == "version":
        deser_u32(out + offset_of(Document::meta) + offset_of(Metadata::version))
        bitset |= 0b010
    elif key == "author":
        deser_String(out + offset_of(Document::meta) + offset_of(Metadata::author))
        bitset |= 0b100
    else:
        skip_value()
    ...
    assert bitset == 0b111
    ret
```

For postcard, `collect_fields` similarly expands flattened fields in-order into
the sequence, with adjusted offsets.

r[deser.flatten.conflict]
If a flattened struct's field names collide with the parent struct's field
names, the compiler reports an error at JIT-compile time.

r[deser.flatten.multiple]
Multiple `#[facet(flatten)]` fields in the same struct are supported; all
their fields are merged into the parent's key namespace.

r[ser.flatten]
For JSON, struct serialization with `#[facet(flatten)]` inlines the flattened
struct's fields into the parent object — no nested object is emitted.

## Enums

```rust
#[derive(Facet)]
enum Animal {
    Cat,
    Dog { name: String, good_boy: bool },
    Parrot(String),
}
```

r[deser.enum.variant-kinds]
Enums have three kinds of variants: unit (`Cat`), struct (`Dog`), and tuple
(`Parrot`). The compiler knows all variants and their fields at JIT-compile
time.

### Postcard enums

r[deser.postcard.enum]
Postcard encodes enums as a varint discriminant followed by the variant's
payload (if any), in field-declaration order. The discriminant is the variant
index (0, 1, 2, ...).

Wire bytes for `Animal::Dog { name: "Rex", good_boy: true }`:

```
01              ; variant index 1 (Dog)
03 52 65 78    ; name: length-prefixed "Rex"
01              ; good_boy: true
```

Wire bytes for `Animal::Cat`:

```
00              ; variant index 0 (Cat), no payload
```

Wire bytes for `Animal::Parrot("Polly")`:

```
02                    ; variant index 2 (Parrot)
05 50 6f 6c 6c 79    ; "Polly"
```

r[deser.postcard.enum.dispatch]
The emitted code reads the varint discriminant, branches to the right variant
via a switch, then deserializes its fields in order. Unknown discriminant
values are an error.

r[deser.postcard.enum.unit]
Unit enum variants are encoded as just the varint discriminant with no
payload bytes.

```
deser_Animal:
    tag = read_varint()
    switch tag:
        case 0:                                ; Cat
            set_enum_variant(out, 0)
            ret                                ; no fields
        case 1:                                ; Dog
            set_enum_variant(out, 1)
            deser_String(out + offset_of(Dog::name))
            deser_bool(out + offset_of(Dog::good_boy))
            ret
        case 2:                                ; Parrot
            set_enum_variant(out, 2)
            deser_String(out + offset_of(Parrot::0))
            ret
        default:
            error("unknown variant")
```

r[deser.enum.set-variant]
`set_enum_variant` writes the Rust discriminant value into the enum's tag
slot (known offset and size from the shape). The payload offsets are relative
to the enum's base address, known at JIT-compile time from each variant's
layout.

### JSON enums — externally tagged (default)

r[deser.json.enum.external]
The default JSON representation for enums is externally tagged: the variant
name is the key of a single-key object.

```json
"Cat"
```

```json
{ "Dog": { "name": "Rex", "good_boy": true } }
```

```json
{ "Parrot": "Polly" }
```

r[deser.json.enum.external.unit-as-string]
Unit variants serialize as bare strings.

r[deser.json.enum.external.struct-variant]
Struct variants become `{ "VariantName": { fields... } }`.

r[deser.json.enum.external.tuple-variant]
Tuple variants with a single field become `{ "VariantName": value }`.

r[deser.json.enum.external.trie]
Variant name dispatch uses a trie built at JIT-compile time, same as struct
field dispatch.

```
deser_Animal:
    skip_whitespace()
    if peek() == '"':
        ; Might be a unit variant as a bare string.
        key = read_quoted_string()
        switch key:             ; trie over variant names
            case "Cat":
                set_enum_variant(out, 0)
                ret
            default:
                error("unknown variant")
    expect('{')
    skip_whitespace()
    key = read_quoted_key()
    expect_colon()
    switch key:                 ; trie over variant names
        case "Cat":
            set_enum_variant(out, 0)
            ; Cat might also appear as { "Cat": null } — accept both
            skip_value()
        case "Dog":
            set_enum_variant(out, 1)
            ; Dog is a struct variant — deserialize as object
            deser_Dog_fields(out)
        case "Parrot":
            set_enum_variant(out, 2)
            ; Parrot is a single-field tuple — deserialize the inner value
            deser_String(out + offset_of(Parrot::0))
        default:
            error("unknown variant")
    skip_whitespace()
    expect('}')
    ret
```

`deser_Dog_fields` is the same struct-body code as for a standalone
struct (field loop, bitset, trie dispatch over `"name"` and `"good_boy"`).

### JSON enums — adjacently tagged

r[deser.json.enum.adjacent]
`#[facet(tag = "type", content = "data")]` uses adjacently tagged encoding:
the tag and content are sibling keys in an object.

```json
{ "type": "Dog", "data": { "name": "Rex", "good_boy": true } }
```

r[deser.json.enum.adjacent.key-order]
Since JSON keys can arrive in any order, the emitted code handles both
type-first and data-first orderings. Data-before-type requires buffering the
raw value.

```
deser_Animal:
    expect('{')
    ; Read first key
    first_key = read_quoted_key()
    expect_colon()
    if first_key == "type":
        variant = read_quoted_string()    ; e.g. "Dog"
        skip_comma()
        expect_key("data")
        expect_colon()
        dispatch_variant(variant, out)    ; trie branch → deser payload
    elif first_key == "data":
        ; data arrived before type — must buffer or defer.
        ; Option A: buffer the raw JSON value, read type, then parse.
        raw = capture_raw_value()
        skip_comma()
        expect_key("type")
        expect_colon()
        variant = read_quoted_string()
        dispatch_variant_from_raw(variant, out, raw)
    else:
        error("expected \"type\" or \"data\"")
    skip_whitespace()
    expect('}')
    ret
```

The data-before-type case requires buffering the raw value into the format
arena. An alternative is to require type-first ordering and error otherwise —
simpler emitted code, less compatible.

r[deser.json.enum.adjacent.unit-variant]
For adjacently tagged enums, unit variants may omit the `content` key or
have it set to `null`.

r[deser.json.enum.adjacent.tuple-variant]
For adjacently tagged enums, tuple variants with a single field have the
field as the `content` value directly.

### JSON enums — internally tagged

r[deser.json.enum.internal]
`#[facet(tag = "type")]` (no `content`) uses internally tagged encoding: the
tag field lives alongside the variant's own fields.

```json
{ "type": "Dog", "name": "Rex", "good_boy": true }
```

r[deser.json.enum.internal.struct-only]
Internally tagged encoding only works for struct variants and unit variants.
Tuple variants are not supported with internal tagging.

r[deser.json.enum.internal.unit-variant]
For internally tagged enums, unit variants are objects with only the tag key:
`{ "type": "Cat" }`.

r[deser.json.enum.internal.buffering]
Because the tag key may not appear first, the emitted code buffers key-value
pairs until the tag is seen, then replays them against the correct variant's
field dispatch trie.

```
deser_Animal:
    expect('{')
    bitset = 0b000         ; type, + variant fields
    variant = UNSET
loop:
    skip_whitespace()
    if peek() == '}': break
    key = read_quoted_key()
    expect_colon()
    if key == "type":
        variant_name = read_quoted_string()
        bitset |= 0b001
    else:
        ; Can't dispatch to a field until we know the variant.
        ; Two strategies:
        ;   1. Require "type" to appear first (simple, fast).
        ;   2. Buffer unknown keys, replay after "type" is known (flexible).
        buffer_key_value(key)
    skip_comma()
    goto loop
    ; After the loop, variant must be known.
    assert variant != UNSET
    switch variant_name:
        case "Cat":
            set_enum_variant(out, 0)
        case "Dog":
            set_enum_variant(out, 1)
            replay_buffered_fields(out, Dog_field_trie)
        ...
    expect('}')
    ret
```

Internally tagged enums are inherently awkward for a JIT compiler: you can't
know which fields to expect until you've seen the tag, but the tag might
not come first. Buffering is the general solution; requiring tag-first is the
fast-path optimization.

### JSON enums — untagged

r[deser.json.enum.untagged]
`#[facet(untagged)]` enums have no discriminant in the wire format. The
deserializer determines the variant from the value itself.

r[deser.json.enum.untagged.no-trial]
kajit does NOT use trial deserialization. Instead, it analyzes variant shapes at
JIT-compile time to build a dispatch strategy, then resolves at runtime by
peeking at the value type.

#### Step 1: classify variants by expected value type

r[deser.json.enum.untagged.bucket]
At JIT-compile time, the compiler examines each variant's shape and buckets
it by the JSON value type it expects (bool, integer, float, string, array,
object, null).

```rust
enum ValueTypeBucket {
    Bool,       // newtype wrapping bool
    Integer,    // newtype wrapping u32, i64, etc.
    Float,      // newtype wrapping f32, f64
    String,     // newtype wrapping String, &str, char, unit variant
    Array,      // newtype wrapping Vec<T>, tuple variant
    Object,     // struct variant, newtype wrapping a struct or map
    Null,       // unit variant (if represented as null)
}
```

For our `Animal` example:

| Variant    | Classification |
|------------|---------------|
| `Cat`      | String (or Null) |
| `Dog { .. }` | Object |
| `Parrot(String)` | String |

#### Step 2: peek at the JSON value type

r[deser.json.enum.untagged.peek]
The emitted code peeks at the first non-whitespace byte to determine the JSON
value type (`{` = object, `"` = string, `[` = array, `t`/`f` = bool, `n` =
null, digit/`-` = number). This eliminates entire categories of variants
without touching the input.

```
deser_Animal:
    skip_whitespace()
    b = peek()
    switch b:
        case '{':  goto object_variants
        case '"':  goto string_variants
        case 'n':  goto null_variants
        default:   error("no variant matches this value type")
```

#### Step 3: disambiguate within a bucket

r[deser.json.enum.untagged.object-solver]
If multiple struct variants fall in the object bucket, the emitted code scans
top-level keys without parsing values, using an inverted index (field name →
bitmask of candidate variants) to narrow down to exactly one variant. The
inverted index is built at JIT-compile time.

r[deser.json.enum.untagged.string-trie]
If multiple variants fall in the string bucket, unit variants are checked first
via a trie over their known names. If no unit variant matches, the value is
passed to the newtype string variant.

r[deser.json.enum.untagged.scalar-unique]
If multiple variants wrap the same scalar type (e.g., two variants both
wrapping `u32`), the compiler reports an error at JIT-compile time.

**Object bucket** — if there's only one struct variant (like `Dog`), emit its
deserializer directly. If there are multiple struct variants, use the
constraint-solver approach described above.

```
object_variants:
    ; Only Dog expects an object — emit directly.
    set_enum_variant(out, 1)
    deser_Dog_fields(out)
    ret
```

If there were multiple struct variants, the emitted code would scan keys first:

```
object_variants:
    save_pos = input_position()
    candidates = 0b11          ; both StructA and StructB are candidates
    ; Scan top-level keys at depth 1 only
    expect('{')
scan_loop:
    skip_whitespace()
    if peek() == '}': goto resolve
    key = read_quoted_key()
    expect_colon()
    skip_value()               ; skip the value entirely — we only need keys
    candidates &= key_to_candidates[key]   ; inverted index lookup
    if popcount(candidates) == 1: goto resolve
    skip_comma()
    goto scan_loop
resolve:
    input_position = save_pos  ; rewind — now parse for real
    switch candidates:
        case 0b01: deser_StructA(out) ...
        case 0b10: deser_StructB(out) ...
        default: error("ambiguous or no variant matched")
```

The inverted index `key_to_candidates` is built at JIT-compile time from the
known field names of all struct variants in the bucket. Each key maps to a
bitmask of variants that contain that field. ANDing narrows the candidate set.
Typically resolves after 1-2 keys.

r[deser.json.enum.untagged.value-type]
When key presence alone cannot disambiguate (multiple candidates share identical
key sets), the solver peeks at the first byte of each value to determine its
JSON type (object, string, number, bool, null). If candidates have different
field types at the same key (e.g. one has `value: u32`, another `value: String`),
the peek byte narrows further. Per-key, per-type masks are built at JIT-compile
time and AND'd into the candidate bitmask.

r[deser.json.enum.untagged.nested-key]
When multiple candidates have the same key mapping to different struct types,
the solver runs a sub-scan of the nested object's keys. Instead of calling
`skip_value`, it consumes the nested `{...}`, scanning inner keys and ANDing
per-inner-key masks (which map back to outer candidates) into the candidate
bitmask. This handles cases like `data: SuccessPayload{items}` vs
`data: ErrorPayload{message}` where the top-level key `"data"` is shared but
the nested structs have distinguishing fields.

r[deser.json.enum.untagged.ambiguity-error]
If key presence, value-type evidence, and nested-key evidence together cannot
resolve the ambiguity, the compiler reports an error at JIT-compile time.
Two variants with identical key sets, identical field types at every key, and
identical nested structure are genuinely indistinguishable from the wire format.

**String bucket** — if multiple variants expect strings (like `Cat` as a unit
variant string and `Parrot(String)`), the compiler checks whether they can be
distinguished. Unit variants have a fixed set of known string values; newtype
string variants accept any string. So: read the string, check against the
known unit variant names via trie, and fall through to the newtype variant if
no name matches.

```
string_variants:
    s = read_quoted_string()
    switch s:                   ; trie over unit variant names
        case "Cat":
            set_enum_variant(out, 0)
            ret
    ; No unit variant matched — must be Parrot
    set_enum_variant(out, 2)
    store_string(out + offset_of(Parrot::0), s)
    ret
```

**Scalar buckets** (bool, integer, float) — if only one variant wraps that
scalar type, emit directly. If multiple variants wrap the same scalar type
(e.g., two variants both wrapping `u32`), that's genuinely ambiguous and the
compiler should error at JIT-compile time rather than guess.

#### Full pseudocode for Animal

```
deser_Animal:
    skip_whitespace()
    b = peek()
    if b == '{':
        ; Only Dog expects an object
        set_enum_variant(out, 1)
        deser_Dog_fields(out)
        ret
    if b == '"':
        s = read_quoted_string()
        if s == "Cat":
            set_enum_variant(out, 0)
            ret
        ; Fall through to Parrot
        set_enum_variant(out, 2)
        store_string(out + offset_of(Parrot::0), s)
        ret
    error("no variant matches this value type")
```

No buffering, no trial deserialization, no O(variants) retries. The dispatch
is a peek + trie, same cost structure as externally tagged enums.

## Calling convention

r[callconv.signature]
Every emitted deserializer function has the same machine-level signature:
`out` (pointer to the output slot) and `ctx` (pointer to a `DeserContext`).

r[callconv.out]
`out` is a pointer to the output slot (e.g., `*mut Friend`). The callee
writes deserialized fields at known offsets from this pointer.

r[callconv.ctx]
`ctx` is a pointer to a `DeserContext` struct that lives on the caller's
(Rust) stack. It carries the input cursor, error state, and format-specific
scratch space. Passed by pointer so all emitted functions share the same
context without copying.

r[callconv.void-return]
Emitted functions return void. Errors are signaled through the context's
error slot, not through return values.

### DeserContext

r[callconv.deser-context]
`DeserContext` is `#[repr(C)]` so emitted code can access fields at known
offsets.

```rust
#[repr(C)]
struct DeserContext {
    // Input cursor — all emitted code reads/advances these.
    input_ptr: *const u8,     // current position
    input_end: *const u8,     // one past the last byte

    // Error reporting — set by emitted code or intrinsics on failure.
    error: ErrorSlot,

    // Format-specific scratch space — opaque to the compiler,
    // used by format intrinsics (e.g., JSON key buffering).
    format_state: *mut u8,
}

#[repr(C)]
struct ErrorSlot {
    code: u32,                // 0 = no error, nonzero = error kind
    offset: u32,              // byte offset in input where error occurred
    // Optional: pointer to a heap-allocated error message,
    // written by intrinsics that can afford the allocation.
    detail: *const u8,
    detail_len: usize,
}
```

### Register assignment

r[callconv.registers.c-abi]
`out` and `ctx` are passed in the platform's C calling convention argument
registers. Emitted functions can be called directly from Rust `extern "C"`
code with no thunk.

r[callconv.registers.aarch64]
On aarch64, the emitted function's prologue moves arguments into callee-saved
registers and loads the input cursor from `ctx`:

| Register | Role |
|----------|------|
| `x0`     | `out` — C ABI argument (moved to `x21` in prologue) |
| `x1`     | `ctx` — C ABI argument (moved to `x22` in prologue) |
| `x19`    | cached `input_ptr` (callee-saved, loaded from `ctx`) |
| `x20`    | cached `input_end` (callee-saved, loaded from `ctx`) |
| `x21`    | cached `out` pointer (callee-saved) |
| `x22`    | cached `ctx` pointer (callee-saved) |

r[callconv.registers.x86-64]
On x86_64, the emitted function's prologue moves arguments into callee-saved
registers and loads the input cursor from `ctx`:

| Register | Role |
|----------|------|
| `rdi`    | `out` — C ABI argument (moved to `r14` in prologue) |
| `rsi`    | `ctx` — C ABI argument (moved to `r15` in prologue) |
| `r12`    | cached `input_ptr` (callee-saved, loaded from `ctx`) |
| `r13`    | cached `input_end` (callee-saved, loaded from `ctx`) |
| `r14`    | cached `out` pointer (callee-saved) |
| `r15`    | cached `ctx` pointer (callee-saved) |

r[callconv.cursor-caching]
The input cursor (`input_ptr`, `input_end`) and the `out`/`ctx` pointers are
all cached in callee-saved registers for the lifetime of the function. On
function entry, the emitted code loads them from arguments and `ctx`. Before
calling an intrinsic, the emitted code stores `input_ptr` back to `ctx`.
After the intrinsic returns, it reloads `input_ptr` (the intrinsic may have
advanced it).

### Stack frame

r[callconv.stack-frame]
Each emitted function allocates a 16-byte-aligned stack frame. The base
layout reserves 48 bytes:

```
[sp+0..16)   frame pointer + return address (aarch64: stp x29,x30)
[sp+16..32)  callee-saved pair 1 (x19,x20 / r12,r13)
[sp+32..48)  callee-saved pair 2 (x21,x22 / r14,r15)
[sp+48..)    format-specific extra space
```

Total frame size = `(48 + extra_stack + 15) & !15`.

r[callconv.stack-frame.extra]
Format-specific extra stack space begins at offset 48 from the stack pointer.
The size is determined by `Format::extra_stack_space()` at JIT-compile time.
Postcard needs zero extra bytes; JSON needs space for key pointer/length
and a required-field bitset.

### Calling intrinsics

r[callconv.intrinsics]
Intrinsics are regular Rust functions with `extern "C"` ABI. They receive
`ctx` as their first argument (so they can read/advance the cursor, report
errors) plus whatever other arguments they need.

```rust
// Example: allocate a chunk for sequence building (see "Sequence construction").
extern "C" fn intrinsic_chunk_alloc(ctx: *mut DeserContext, elem_size: usize, elem_align: usize, capacity: usize) -> *mut u8;

// Example: finalize a chunk chain into a Vec<T>.
extern "C" fn intrinsic_chunk_finalize_vec(ctx: *mut DeserContext, chain: *mut ChunkChain, vec_out: *mut u8, elem_size: usize);

// Example: read a JSON quoted string into a scratch buffer.
extern "C" fn intrinsic_json_read_string(ctx: *mut DeserContext) -> StringRef;
```

r[callconv.intrinsics.address]
The emitted code calls intrinsics with a normal `call` instruction. The
function pointer is baked into the emitted code at JIT-compile time as an
immediate or a pc-relative load from a constant pool.

### Calling between emitted functions

r[callconv.inter-function]
Calls between emitted functions use the same `(out, ctx)` convention. The
caller sets `out` to the address of the field being deserialized (e.g.,
`out + offset_of(Node::children)`), keeps `ctx` as-is, and emits a `call`
(or `bl` on aarch64). On return, the caller reloads the cached input cursor
from `ctx` in case the callee advanced it.

## Error handling

r[error.types]
The error type provides at minimum: the byte offset in the input where the
error occurred and a human-readable description of what was expected vs. what
was found.

r[error.api]
`DeserError` is the public error type returned by `CompiledDeser::call`. It
is constructed from the `ErrorSlot` after the emitted function returns.

r[error.slot]
Errors are signaled through the context's error slot, not through return
values. When an emitted function or intrinsic encounters an error, it writes
the error code and input offset to `ctx.error` and branches to the function's
error exit label.

r[error.exit-label]
Each emitted function has an error exit label at its end. The error exit
restores callee-saved registers, stores the cached `input_ptr` back to `ctx`,
and returns.

r[error.propagation]
After each intrinsic call or inter-function call, the emitted code checks
`ctx.error.code != 0` via `cbnz` and branches to its own error exit if set.

```
deser_Friend:
    ; prologue: save callee-saved regs, load cached cursor
    ldr x19, [x1, #CTX_INPUT_PTR]
    ldr x20, [x1, #CTX_INPUT_END]
    ...
    ; call intrinsic (e.g., expect '{')
    str x19, [x1, #CTX_INPUT_PTR]       ; flush cursor
    bl intrinsic_json_expect_lbrace
    ldr x19, [x1, #CTX_INPUT_PTR]       ; reload cursor
    ldr w8, [x1, #CTX_ERROR_CODE]       ; check error
    cbnz w8, .Lerror_exit                ; propagate if set
    ...
.Lerror_exit:
    ; store cursor back, restore callee-saved regs, ret
    str x19, [x1, #CTX_INPUT_PTR]
    ; restore x19, x20 from stack
    ret
```

### Why not setjmp/longjmp?

kajit does not use setjmp/longjmp. longjmp skips destructors, leaving
partially-constructed values in an undroppable state. setjmp has its own
cost, and it makes error messages (offset, context) harder to produce.

The error-slot approach pays ~1 `cbnz` per intrinsic call on the happy path.
That's one cycle, almost always correctly predicted as not-taken.

### Why not two-register return?

kajit does not use two-register return for errors. Emitted functions write into
`out` by pointer — they don't "return" the deserialized value. Error
propagation would still require a branch after every call, same cost as the
error slot, but with error state split between return registers and the
context.

## Format context and scratch space

r[format-state.pointer]
Format-specific runtime state lives in the `format_state` pointer inside
`DeserContext`. Each format crate defines its own `#[repr(C)]` state struct.

r[format-state.json]
JSON's format state includes a bump-allocated arena for temporary allocations
(buffered values, strings) and a scratch buffer for key scanning (untagged
struct disambiguation).

r[format-state.postcard]
Postcard requires no format state. `format_state` can be null.

```rust
// For JSON:
#[repr(C)]
struct JsonState {
    // Arena for temporary allocations (buffered values, strings).
    // Bump-allocated, reset after each top-level deserialization.
    arena_base: *mut u8,
    arena_ptr: *mut u8,
    arena_end: *mut u8,

    // Scratch buffer for key scanning (untagged struct disambiguation).
    key_buf: *mut u8,
    key_buf_len: usize,
    key_buf_cap: usize,
}

// For postcard: no state needed — postcard is stateless.
// format_state can be null.
```

### Arena allocation

r[format-state.arena]
The arena is bump-allocated: intrinsics call `arena_alloc(ctx, size, align)`
which advances `arena_ptr` and returns the old pointer. If the arena is
exhausted, the intrinsic grows it (realloc the backing allocation).

r[format-state.arena.reset]
The arena is reset (`ptr = base`) after each top-level `deser.call()` — all
temporary allocations are freed in bulk.

Buffered data is raw bytes plus a small index of key offsets — no parsing, no
intermediate `Value` type.

### Who allocates the context?

r[callconv.context-allocation]
The Rust entry point (safe wrapper around `deser.call()`) allocates
`DeserContext` on the stack, initializes the input cursor from the `&[u8]`
argument, zero-initializes the error slot, and sets `format_state` to the
format-specific state struct. Then it calls the emitted function with `out`
and `ctx`.

```rust
impl CompiledDeser {
    pub fn call(&self, out: &mut MaybeUninit<T>, input: &[u8]) -> Result<(), DeserError> {
        let mut json_state = JsonState::new();    // stack + small heap alloc for arena
        let mut ctx = DeserContext {
            input_ptr: input.as_ptr(),
            input_end: input.as_ptr().add(input.len()),
            error: ErrorSlot::default(),
            format_state: &mut json_state as *mut _ as *mut u8,
        };
        unsafe {
            (self.fn_ptr)(out as *mut _ as *mut u8, &mut ctx);
        }
        if ctx.error.code != 0 {
            Err(DeserError::from_slot(&ctx.error, input))
        } else {
            Ok(())
        }
    }
}
```

r[format-state.zero-cost]
If a format needs no scratch space (e.g., postcard), `format_state` is null
and no allocation cost is paid.

## Assumed layout mode (`malum`)

r[malum]
kajit has a feature called `malum` (Latin for "the fruit of the tree of the
knowledge of good and evil") that enables direct manipulation of standard
library types whose memory layouts are not guaranteed by the language but
are de facto stable. This includes `Vec<T>`, `String`, `Box<T>`, and
potentially `HashMap<K,V>`.

r[malum.default-on]
`malum` is enabled by default (`default = ["malum"]`). It can be disabled
via `default-features = false` for maximum safety at the cost of
performance.

r[malum.what-it-enables]
When `malum` is enabled, kajit writes directly into the memory representation
of standard library types instead of going through vtable function pointers
or intermediate staging areas. For example, `Vec<T>` is known to be
`(ptr, len, cap)` — 24 bytes on 64-bit — and kajit writes those three words
directly.

r[malum.jit-time-validation]
At JIT-compile time, `malum` validates its assumptions by constructing
real values and checking their layouts match expectations. For `Vec<T>`,
the validator:

1. Creates a `Vec<T>` with known contents via `Vec::from_raw_parts`
2. Reads `ptr`, `len`, `cap` from known offsets (0, 8, 16)
3. Asserts they match the values that were written
4. Repeats with different element types (`u8`, `String`, a struct)

If any assertion fails, compilation panics with a message identifying
exactly which layout assumption broke. This catches rustc layout changes
at JIT-compile time, not at runtime.

For LLDB workflows when debugging generated JIT code and resolving JIT symbol
names on macOS, see `docs/jit-debugging.md`.

```rust
fn validate_vec_layout() {
    // Build a Vec we control, read its guts at known offsets
    let mut v: Vec<u32> = Vec::with_capacity(10);
    v.extend_from_slice(&[1, 2, 3]);
    let raw = &v as *const Vec<u32> as *const u8;
    unsafe {
        let ptr = *(raw.add(0) as *const *const u32);
        let len = *(raw.add(8) as *const usize);
        let cap = *(raw.add(16) as *const usize);
        assert_eq!(ptr, v.as_ptr());
        assert_eq!(len, 3);
        assert_eq!(cap, 10);
    }
    std::mem::forget(v);
}
```

r[malum.compile-time-guards]
In addition to JIT-time validation, `malum` includes compile-time
`const` assertions:

```rust
const _: () = assert!(size_of::<Vec<u8>>() == 24);
const _: () = assert!(align_of::<Vec<u8>>() == 8);
```

These catch changes before any code runs.

r[malum.fallback]
When `malum` is disabled, kajit falls back to the chunk-chain strategy
(see below) for all collection types. This is slower but makes zero
assumptions about standard library internals.

## Sequences and arrays

r[deser.json.vec]
For JSON, `Vec<T>` is deserialized from a JSON array.

r[deser.postcard.vec]
For postcard, `Vec<T>` is deserialized as a varint length followed by that
many elements.

r[deser.array]
Fixed-size arrays `[T; N]` are deserialized by emitting N element
deserializers in sequence. The emitted code verifies the input contains
exactly N elements.

r[deser.json.array]
For JSON, `[T; N]` is deserialized from a JSON array. If the array has
fewer or more than N elements, an error is reported.

r[deser.postcard.array]
For postcard, `[T; N]` is deserialized as N elements in sequence with no
length prefix (the count is known from the type).

r[ser.array]
Fixed-size arrays `[T; N]` are serialized by emitting N element serializers
in sequence.

## Drop safety

r[deser.drop-safety]
If deserialization fails after partially constructing a value, all
fully-initialized fields and sub-values must be dropped. The compiler emits
drop glue calls for fields that were successfully written before the error.

r[deser.drop-safety.bitset]
The required-field bitset doubles as a "fields initialized" tracker for
drop-safety purposes: on error, only fields whose bits are set need to be
dropped.

## Vec deserialization strategy

r[seq.no-vec-push]
kajit does NOT use `Vec::push` or `Vec::with_capacity` + push during
deserialization. `Vec::push` requires a staging area per element and a
vtable call per push — both unacceptable for a JIT deserializer.

### Fast path: direct layout (`malum`)

r[seq.malum]
When the `malum` feature is enabled, kajit writes `Vec<T>` directly as
`(ptr, len, cap)` — 24 bytes at the output location. The emitted code
manages its own backing buffer, deserializes elements in-place, and writes
the three words when done.

r[seq.malum.postcard]
For postcard, the element count is known upfront (varint length prefix).
The emitted code allocates a single buffer of exactly `count * size_of(T)`
bytes (using the global allocator, same layout as `Vec` would use),
deserializes all elements directly into it, and writes
`(buf_ptr, count, count)` to the output — zero copy, zero finalization.

```
deser_Vec_Friend:                      ; postcard, malum
    count = read_varint()
    buf = alloc(count * size_of(Friend), align_of(Friend))
    i = 0
loop:
    if i >= count: break
    slot = buf + i * size_of(Friend)
    call deser_Friend(slot, ctx)
    check_error
    i += 1
    goto loop
    ; Write Vec<Friend> directly: ptr, len, cap
    store(out + 0, buf)
    store(out + 8, count)
    store(out + 16, count)
    ret
```

r[seq.malum.json]
For JSON, the element count is unknown. The emitted code allocates an
initial buffer, deserializes elements into it, and grows it when full.
Growth allocates a new buffer (double capacity), `memcpy`s existing
elements, frees the old buffer. When done, writes `(ptr, len, cap)` to
the output.

```
deser_Vec_Friend:                      ; JSON, malum
    expect('[')
    cap = 16
    buf = alloc(cap * size_of(Friend), align_of(Friend))
    len = 0
loop:
    skip_whitespace()
    if peek() == ']': break
    if len == cap:
        new_cap = cap * 2
        new_buf = alloc(new_cap * size_of(Friend), align_of(Friend))
        memcpy(new_buf, buf, len * size_of(Friend))
        free(buf, cap * size_of(Friend), align_of(Friend))
        buf = new_buf
        cap = new_cap
    slot = buf + len * size_of(Friend)
    call deser_Friend(slot, ctx)
    check_error
    len += 1
    skip_comma()
    goto loop
    expect(']')
    store(out + 0, buf)
    store(out + 8, len)
    store(out + 16, cap)
    ret
```

r[seq.malum.alloc-compat]
Buffers allocated by the emitted code MUST use the same allocator and
layout that `Vec<T>` would use (`std::alloc::alloc` with
`Layout::array::<T>(cap)`). This ensures the resulting `Vec` can be
dropped, resized, and deallocated normally by Rust code that receives it.

r[seq.malum.empty]
An empty array (`[]` in JSON, count=0 in postcard) writes
`(dangling_ptr, 0, 0)` to the output — same as `Vec::new()`. No
allocation is performed.

### Slow path: chunk chain (no `malum`)

r[seq.chunk-chain]
When `malum` is disabled, deserializing variable-length collections
(`Vec<T>`, `HashSet<T>`, etc.) uses a chunk chain: a linked list of
fixed-size buffers. Elements are constructed in-place in the current chunk.
When a chunk fills up, a new one is allocated and linked. No existing chunk
ever moves.

```rust
#[repr(C)]
struct ChunkChain {
    // Current chunk — elements are written here.
    current: *mut u8,
    current_len: usize,     // elements written so far in this chunk
    current_cap: usize,     // element capacity of this chunk

    // Linked list of full chunks (newest first).
    full_chunks: *mut FullChunk,
    total_len: usize,       // total elements across all chunks
}

#[repr(C)]
struct FullChunk {
    next: *mut FullChunk,
    data: *mut u8,
    len: usize,             // always == capacity (chunk was full)
    cap: usize,
}
```

r[seq.chain-lifecycle]
The emitted code allocates a chain, loops to get slots and deserialize
elements in-place, commits each element, then finalizes the chain into the
target collection.

```
deser_Vec_Friend:                      ; chunk chain fallback
    call intrinsic_chain_new(ctx, size_of(Friend), align_of(Friend), size_hint)
loop:
    ; Format-specific: check for end of array.
    slot = call intrinsic_chain_next_slot(ctx, chain_ptr, size_of(Friend), align_of(Friend))
    call deser_Friend(slot, ctx)
    check_error
    call intrinsic_chain_commit(ctx, chain_ptr)
    goto loop
    call intrinsic_chain_to_vec(ctx, chain_ptr, out, size_of(Friend), align_of(Friend))
    ret
```

### Finalization (chunk chain)

r[seq.finalize.one-chunk]
If the chain has a single chunk, its buffer is transferred directly to the
`Vec` via vtable — the chain was allocated with a compatible layout. No
copy if the vtable can accept ownership of the buffer.

r[seq.finalize.multi-chunk]
If the chain has multiple chunks, a single buffer of `total_len` capacity is
allocated, each chunk's data is `memcpy`'d into it in order, and the `Vec`
is built from that — one copy total.

### Size hints

r[seq.size-hint.postcard]
Postcard provides an exact element count (length-prefixed). With `malum`,
this means a single allocation of exact size. With chunk chain, the chain
is initialized with a single chunk of exactly the right capacity, so
finalization is always the zero-copy one-chunk path.

r[seq.size-hint.json]
JSON doesn't know the count upfront. With `malum`, the buffer starts at a
reasonable size (e.g., 16 elements) and doubles on growth. With chunk
chain, the chain starts at a reasonable chunk size and adds chunks as
needed.

### Drop safety (sequences)

r[seq.drop-safety]
If deserialization fails mid-array, the error path must:
1. Drop the N-1 fully committed elements (by calling their drop glue).
2. Drop any partially constructed fields of element N.
3. Free the backing buffer(s).

r[seq.drop-safety.committed-count]
Both the `malum` path and the chunk chain track committed element count
separately from the write cursor, so they know exactly how many elements
to drop.

## Option, Result, and opaque types

r[opaque.vtable]
`Option<T>` and `Result<T, E>` are opaque types — kajit cannot treat them as
regular enums because niche optimization makes their memory layout
unpredictable. Construction and inspection go through facet's dedicated
vtables (`OptionVTable`, `ResultVTable`).

```rust
// OptionVTable — all operations go through these function pointers.
struct OptionVTable {
    is_some:      fn(option: PtrConst) -> bool,
    get_value:    fn(option: PtrConst) -> Option<PtrConst>,
    init_some:    fn(option: PtrUninit, value: PtrMut) -> PtrMut,
    init_none:    fn(option: PtrUninit) -> PtrMut,
    replace_with: fn(option: PtrMut, value: Option<PtrMut>),
}

// ResultVTable — similar pattern.
struct ResultVTable {
    is_ok:    fn(result: PtrConst) -> bool,
    get_ok:   fn(result: PtrConst) -> Option<PtrConst>,
    get_err:  fn(result: PtrConst) -> Option<PtrConst>,
    init_ok:  fn(result: PtrUninit, value: PtrMut) -> PtrMut,
    init_err: fn(result: PtrUninit, value: PtrMut) -> PtrMut,
}
```

These vtable functions are monomorphized per `T` at `const`-eval time. They
know the exact layout (including niche optimization) because they're compiled
by `rustc` for the concrete type. kajit calls them as intrinsics — it cannot
inline their logic into emitted code because the logic depends on `T`.

### How kajit deserializes Option<T>

r[deser.json.option]
For JSON, `Option<T>` fields have three behaviors:
- **Field absent**: `vtable.init_none(out + field_offset)` is called after
  the field loop for any unset optional fields.
- **Field present with value**: the inner `T` is deserialized into a
  temporary slot, then `vtable.init_some(out + field_offset, &temp)`.
- **Field present with `null`**: `vtable.init_none(out + field_offset)`.

r[deser.postcard.option]
For postcard, `Option<T>` is encoded as `0x00` (None) or `0x01` + value
(Some). The emitted code reads the tag byte and calls the vtable.

```
; In the JSON field loop, after dispatching to the "age" field
; (which is Option<u32>):
    skip_whitespace()
    if peek() == 'n':
        expect_null()
        call vtable_option_init_none(out + offset_of(age))
    else:
        deser_u32(&temp)
        call vtable_option_init_some(out + offset_of(age), &temp)
    bitset |= 0b01  ; mark as seen (so we don't init_none again)
```

After the field loop:

```
    ; For each optional field whose bit is NOT set:
    if !(bitset & 0b01):
        call vtable_option_init_none(out + offset_of(age))
```

For postcard, `Option<T>` is encoded as a `0x00` (None) or `0x01` followed
by the value (Some). The emitted code reads the tag byte, branches, and
calls the vtable:

```
deser_Option_u32:
    tag = read_byte()
    if tag == 0:
        call vtable_option_init_none(out)
        ret
    if tag == 1:
        deser_u32(&temp)
        call vtable_option_init_some(out, &temp)
        ret
    error("invalid option tag")
```

### How kajit deserializes Result<T, E>

r[deser.json.result]
For JSON, `Result<T, E>` uses externally tagged encoding (`{ "Ok": value }`
or `{ "Err": value }`). The emitted code reads the variant key, deserializes
the inner value into a temporary, and calls the vtable.

r[deser.postcard.result]
For postcard, `Result` is a varint tag (0 = Ok, 1 = Err) followed by the
payload. Construction goes through the vtable.

```
deser_Result_u32_String:
    expect('{')
    key = read_quoted_key()
    expect_colon()
    if key == "Ok":
        deser_u32(&temp)
        call vtable_result_init_ok(out, &temp)
    elif key == "Err":
        deser_String(&temp)
        call vtable_result_init_err(out, &temp)
    else:
        error("expected Ok or Err")
    expect('}')
    ret
```

### Why not just use the enum path?

kajit does not attempt to inline niche optimization logic into emitted code.
The vtable functions already handle every niche pattern correctly. The cost
is one indirect call per Option/Result construction, negligible compared to
the actual deserialization work. Special-casing known niche patterns is a
future optimization, not the default path.

### Smart pointers (Box, Arc, Rc)

r[deser.pointer]
Smart pointers (`Box<T>`, `Arc<T>`, `Rc<T>`) are wire-transparent: the wire
format contains just `T`. kajit detects them via `Def::Pointer` with a
`KnownPointer` of `Box`, `Arc`, or `Rc`, and a non-None `new_into_fn` in
the vtable.

r[deser.pointer.scratch]
Deserialization uses the same scratch area as `Option<T>`: the inner `T` is
deserialized into a temporary stack slot, then the vtable's `new_into_fn`
is called to move the value into the heap-allocated pointer.

r[deser.pointer.new-into]
The `new_into_fn` trampoline bridges thin raw pointers from JIT code to
facet's wide pointer types (`PtrUninit`, `PtrMut`). It has the same
three-argument ABI as `kajit_option_init_some`: `(fn_ptr, out, value_ptr)`.

r[deser.pointer.format-transparent]
No format-level changes are needed — both JSON and postcard deserialize the
inner `T` identically to a bare field. The pointer wrapping is purely a
compiler concern.

r[deser.pointer.nesting]
Pointers compose with other wrappers:
- `Option<Box<T>>`: Option dispatches on null/presence, then the Some path
  deserializes `Box<T>` (which deserializes `T` + wraps).
- `Box<Option<T>>`: the pointer wraps an Option, so `T` = `Option<U>` and
  the inner deserialization handles the Option wire protocol.
- `Vec<Box<T>>`: each Vec element is a pointer, using the scratch area per
  element.

### Default values

r[deser.default]
`#[facet(default)]` on a field means: if absent from the input, use
`T::default()` instead of erroring. The field is treated as optional in the
required-field bitset. After the field loop, if the bit is unset, the
emitted code calls an intrinsic that invokes `T::default()`.

r[deser.default.fn-ptr]
The default function pointer is baked into the emitted code at JIT-compile
time from the shape's metadata.

```
    ; After the JSON field loop, for a field with #[facet(default)]:
    if !(bitset & 0b10):
        call intrinsic_write_default(out + offset_of(name), default_fn_ptr)
```

r[deser.default.postcard-irrelevant]
For postcard, defaults don't apply: all fields are always present in the wire
format (postcard is non-self-describing, there's no concept of "absent
field").

## Maps and sets

r[map.chunk-chain]
Maps and sets are opaque collection types. kajit deserializes them using the
same chunk-chain strategy as sequences: build a chain of flat entries, then
finalize into the collection via vtable.

### Maps

r[map.entry-layout]
Map entries are laid out as `[K, padding, V]` in the chunk chain, with
alignment `max(align_of(K), align_of(V))`. The entry stride is computed at
JIT-compile time.

r[deser.json.map]
For JSON, maps are objects. Keys are always strings in the wire format, but
`K` might be `String`, `u32`, `Cow<str>`, etc. The format crate reads the
quoted key and the compiler emits code to deserialize it into the key slot.

```
deser_HashMap_String_u32:
    expect('{')
    call intrinsic_chain_new(ctx, entry_size, entry_align, 0)
loop:
    skip_whitespace()
    if peek() == '}': break
    slot = call intrinsic_chain_next_slot(ctx, chain_ptr, entry_size, entry_align)
    ; Deserialize the key (always a JSON string, parsed into K).
    deser_String(slot + key_offset, ctx)
    check_error
    expect_colon()
    ; Deserialize the value.
    deser_u32(slot + value_offset, ctx)
    check_error
    call intrinsic_chain_commit(ctx, chain_ptr)
    skip_comma()
    goto loop
    expect('}')
    ; Finalize: build the HashMap from the flat (K, V) entries.
    call intrinsic_chain_to_map(ctx, chain_ptr, out, map_vtable)
    ret
```

r[map.finalize]
`intrinsic_chain_to_map` iterates the committed entries and calls the map's
vtable insert function for each `(K, V)` pair. The vtable handles hashing
(HashMap) or ordering (BTreeMap).

r[deser.postcard.map]
For postcard, maps are length-prefixed sequences of `(K, V)` pairs. The
chain gets an exact capacity from the length prefix.

### Sets

r[set.entries]
Sets use the same chunk-chain strategy as maps but with entries of just `K`
(no value). Finalization calls a set-specific vtable insert.

```
deser_HashSet_String:
    ; JSON: expect an array of values.
    expect('[')
    call intrinsic_chain_new(ctx, elem_size, elem_align, 0)
loop:
    skip_whitespace()
    if peek() == ']': break
    slot = call intrinsic_chain_next_slot(ctx, chain_ptr, elem_size, elem_align)
    deser_String(slot, ctx)
    check_error
    call intrinsic_chain_commit(ctx, chain_ptr)
    skip_comma()
    goto loop
    expect(']')
    call intrinsic_chain_to_set(ctx, chain_ptr, out, set_vtable)
    ret
```

r[deser.json.set]
JSON sets are deserialized as arrays (there's no set literal in JSON).

r[deser.postcard.set]
Postcard sets are length-prefixed sequences of `K`.

## Serialization

r[ser.overview]
Serialization walks a fully-constructed Rust value and emits bytes. It
requires no field dispatch, no bitsets, no chunk chains, and no drop safety
— the input value is borrowed, not mutated.

### Calling convention

r[ser.callconv]
Serializer functions take `inp` (pointer to the value being serialized) and
`ctx` (pointer to a `SerContext`).

```rust
#[repr(C)]
struct SerContext {
    // Output buffer — a growable byte vec.
    out_ptr: *mut u8,
    out_len: usize,
    out_cap: usize,

    // Error reporting — same pattern as deserialization.
    error: ErrorSlot,

    // Format-specific state (e.g., JSON indentation depth).
    format_state: *mut u8,
}
```

r[ser.output-buffer]
The output buffer is a `Vec<u8>` in disguise (ptr, len, cap). Emitted code
writes bytes by storing to `out_ptr + out_len` and incrementing `out_len`.
When `out_len` would exceed `out_cap`, the emitted code calls an intrinsic
to grow the buffer.

r[ser.registers]
Register assignment mirrors deserialization: `inp` in the first argument
register, `ctx` in the second. The output pointer + length can be cached in
callee-saved registers for the hot path.

### Struct serialization

r[ser.json.struct]
For JSON, struct serialization emits each field in declaration order — no
dispatch, just a straight sequence of key + value pairs.

```
ser_Friend:
    emit_byte('{')
    emit_quoted_key("age")
    emit_byte(':')
    ser_u32(inp + offset_of(age))
    emit_byte(',')
    emit_quoted_key("name")
    emit_byte(':')
    ser_String(inp + offset_of(name))
    emit_byte('}')
    ret
```

r[ser.json.struct.inline-keys]
Key strings are baked into the emitted code as immediate byte sequences. For
small keys, they're written inline. For longer keys, `memcpy` from a
constant pool.

r[ser.postcard.struct]
For postcard, struct serialization emits each
field in order, no keys, no delimiters:

```
ser_Friend:
    ser_u32(inp + offset_of(age))
    ser_String(inp + offset_of(name))
    ret
```

### Enum serialization

r[ser.enum.discriminant]
The emitted code reads the enum's discriminant (known offset and size from
the shape), then branches to the right variant's serializer.

r[ser.json.enum]
For JSON externally tagged:

```
ser_Animal:
    variant = read_enum_discriminant(inp)
    switch variant:
        case 0:  ; Cat
            emit_quoted_string("Cat")
            ret
        case 1:  ; Dog
            emit_byte('{')
            emit_quoted_key("Dog")
            emit_byte(':')
            ser_Dog_fields(inp)
            emit_byte('}')
            ret
        case 2:  ; Parrot
            emit_byte('{')
            emit_quoted_key("Parrot")
            emit_byte(':')
            ser_String(inp + offset_of(Parrot::0))
            emit_byte('}')
            ret
```

r[ser.postcard.enum]
For postcard, enums are serialized as a varint discriminant followed by the
variant's fields in order.

```
ser_Animal:
    variant = read_enum_discriminant(inp)
    emit_varint(variant)
    switch variant:
        case 0: ret                              ; Cat — no payload
        case 1:                                  ; Dog
            ser_String(inp + offset_of(Dog::name))
            ser_bool(inp + offset_of(Dog::good_boy))
            ret
        case 2:                                  ; Parrot
            ser_String(inp + offset_of(Parrot::0))
            ret
```

r[ser.json.enum.adjacent]
JSON adjacently tagged enum serialization emits
`{ "tag_key": "VariantName", "content_key": payload }`.

r[ser.json.enum.internal]
JSON internally tagged enum serialization emits the tag field alongside the
variant's own fields in a single object.

r[ser.json.enum.untagged]
JSON untagged enum serialization emits the variant's payload directly with
no discriminant in the output.

### Option serialization

r[ser.json.option]
For JSON, `Option<T>` fields that are `None` are omitted from the output.
The emitted code checks `vtable.is_some(inp + field_offset)` and
conditionally emits the key + value. The comma logic tracks whether a comma
is needed before the next field.

r[ser.postcard.option]
For postcard, `Option<T>` emits `0x00` for None or `0x01` + value for Some.

```
ser_struct_with_optional_age:
    emit_byte('{')
    need_comma = false
    ; age: Option<u32>
    if vtable_option_is_some(inp + offset_of(age)):
        if need_comma: emit_byte(',')
        emit_quoted_key("age")
        emit_byte(':')
        value_ptr = vtable_option_get_value(inp + offset_of(age))
        ser_u32(value_ptr)
        need_comma = true
    ; name: String (always present)
    if need_comma: emit_byte(',')
    emit_quoted_key("name")
    emit_byte(':')
    ser_String(inp + offset_of(name))
    emit_byte('}')
    ret
```

For postcard, `Option<T>` emits `0x00` for None or `0x01` + value for Some:

```
    if vtable_option_is_some(inp + offset_of(age)):
        emit_byte(0x01)
        value_ptr = vtable_option_get_value(inp + offset_of(age))
        ser_u32(value_ptr)
    else:
        emit_byte(0x00)
```

### Result serialization

r[ser.json.result]
JSON `Result<T, E>` serialization uses externally tagged encoding:
`{ "Ok": value }` or `{ "Err": value }`.

r[ser.postcard.result]
Postcard `Result<T, E>` serialization emits a varint tag (0 = Ok, 1 = Err)
followed by the payload.

### Sequence serialization

r[ser.json.seq]
For JSON, sequence serialization emits `[`, then iterates the Vec's elements
(ptr, len from known offsets — Vec layout verified at JIT-compile time),
serializes each element with commas between, then `]`.

r[ser.postcard.seq]
For postcard, sequence serialization emits the length as a varint, then each
element with no delimiters.

```
ser_Vec_Friend:
    emit_byte('[')
    ptr = load(inp + VEC_PTR_OFFSET)
    len = load(inp + VEC_LEN_OFFSET)
    i = 0
loop:
    if i >= len: break
    if i > 0: emit_byte(',')
    ser_Friend(ptr + i * size_of(Friend), ctx)
    check_error
    i += 1
    goto loop
    emit_byte(']')
    ret
```

### Map serialization

r[ser.json.map]
For JSON, maps are serialized as objects. The emitted code iterates via
vtable (maps are opaque — no known memory layout). The vtable provides an
iterator yielding `(K_ptr, V_ptr)` pairs.

r[ser.postcard.map]
For postcard, maps emit the length as a varint, then each `(K, V)` pair.

```
ser_HashMap_String_u32:
    emit_byte('{')
    iter = call vtable_map_iter(inp)
    first = true
loop:
    entry = call vtable_map_iter_next(iter)
    if entry == null: break
    if !first: emit_byte(',')
    first = false
    ser_String_as_key(entry.key_ptr)    ; emit as quoted JSON key
    emit_byte(':')
    ser_u32(entry.value_ptr)
    check_error
    goto loop
    emit_byte('}')
    ret
```

### Set serialization

r[ser.json.set]
JSON sets are serialized as arrays.

r[ser.postcard.set]
Postcard sets are serialized as length-prefixed sequences of elements.

### Output buffer growth

r[ser.output-growth]
When the emitted code needs to write N bytes and `out_len + N > out_cap`, it
calls `intrinsic_grow_output` which reallocates the buffer (doubling or
adding `additional`, whichever is larger), updates `out_ptr`/`out_cap` in the
context, and returns. The emitted code reloads the cached output pointer
after the call.

r[ser.output-growth.inline-check]
For small writes (single bytes, short keys), the emitted code checks
capacity inline and only calls the intrinsic on the slow path. For large
writes, the intrinsic handles the capacity check internally.

### Compile-time optimizations

r[ser.merged-constants]
Since field names and delimiters are known at JIT-compile time, the
serializer merges adjacent constant byte sequences into the fewest possible
store instructions. For example, `,"name":` (8 bytes) can be written as a
single 8-byte store.

## Future: two-pass deserialization for non-contiguous formats

r[twopass]
Some formats scatter values for a single logical field across the document.
TOML is the primary example: `[[array_of_tables]]` spreads array elements
across the file, so a single-pass deserializer would need to re-enter
subtrees or buffer raw bytes for deferred parsing.

r[twopass.format-level]
Two-pass deserialization is a format-level decision, not a per-field one.
A format declares whether it requires two passes. JSON and postcard are
single-pass — all values are contiguous. TOML would be two-pass.

r[twopass.passes]
- **Pass 1 (index)**: Scan the document, identify keys and record byte
  ranges for each value. No value parsing, no allocation beyond the index
  itself. The result is a format-specific `DocIndex` — a compact mapping
  from paths to `(start, end)` byte offsets in the input.
- **Pass 2 (materialize)**: Walk the shape tree. For each field, look up
  its byte range in the index, seek to that range, and deserialize the
  value. This gives random access to the document in type-tree order.

r[twopass.index-storage]
The `DocIndex` must be compact. For TOML, it stores `(key_path, start, end)`
triples. Key paths can be interned or referenced as byte ranges into the
original input (zero-copy). The index is allocated on the format arena and
freed in bulk after deserialization.

r[twopass.not-needed-yet]
JSON and postcard do not use two-pass deserialization. All their values are
contiguous — arrays are delimited by `[...]`, objects by `{...}`, postcard
sequences are length-prefixed. Even with `#[facet(flatten)]`, the value
for each key is consumed in one shot when the key dispatches to it.

Two-pass is reserved for formats where this property does not hold.
