# kajit architecture

kajit is a JIT-compiled deserializer for Rust. It uses [facet] reflection to
inspect type metadata at runtime and [dynasmrt] to emit native machine code
for aarch64 and x86_64.

[facet]: https://github.com/facet-rs/facet
[dynasmrt]: https://crates.io/crates/dynasmrt

## Compilation flow

```
Shape (facet reflection)
  │
  ▼
compiler.rs: compile_deser(shape, format)
  │
  ├── Compiler::compile_shape(shape)     ← recursive, one function per Shape
  │     │
  │     ├── collect_fields(shape)        ← offset, name, inner Shape per field
  │     ├── depth-first compile nested struct shapes (if !inline)
  │     ├── EmitCtx::begin_func()        ← prologue (save registers, cache cursor)
  │     ├── Format::emit_struct_fields() ← format controls field iteration order
  │     │     │
  │     │     └── per field:
  │     │           ├── nested struct + inline  → emit_inline_struct (flatten)
  │     │           ├── nested struct + !inline → emit_call_emitted_func (call)
  │     │           ├── String                  → Format::emit_read_string
  │     │           └── scalar                  → Format::emit_read_scalar
  │     │
  │     └── EmitCtx::end_func()          ← epilogue (restore registers, error exit)
  │
  ├── EmitCtx::finalize()               ← commit + finalize → ExecutableBuffer
  └── CompiledDeser { buf, entry, func }
```

## Source files

| File | Role |
|------|------|
| `src/compiler.rs` | Recursive compiler, one function per Shape |
| `src/format.rs` | `Format` trait — format-specific code emission |
| `src/postcard.rs` | Postcard format (positional, varint integers) |
| `src/json.rs` | JSON format (keyed, key-matching state machine) |
| `src/arch/aarch64.rs` | aarch64 code emission (EmitCtx) |
| `src/arch/x64.rs` | x86_64 code emission (EmitCtx) |
| `src/context.rs` | `DeserContext` — runtime state passed to emitted code |
| `src/intrinsics.rs` | Rust functions callable from emitted code |
| `src/json_intrinsics.rs` | JSON-specific intrinsics (key parsing, skip, etc.) |
| `src/lib.rs` | Public API: `compile_deser`, `deserialize` |

## Calling conventions

### Entry point

Every emitted function follows the platform C ABI:

| Platform | arg0 (out ptr) | arg1 (ctx ptr) |
|----------|----------------|----------------|
| aarch64  | x0             | x1             |
| x86_64   | rdi            | rsi            |

### Register assignments (callee-saved, cached across intrinsic calls)

| Role | aarch64 | x86_64 |
|------|---------|--------|
| Cached input_ptr | x19 | r12 |
| Cached input_end | x20 | r13 |
| Out pointer | x21 | r14 |
| Ctx pointer | x22 | r15 |

These are loaded from arguments/context in the prologue and kept live across
the entire function. The cursor (input_ptr) is flushed to `ctx.input_ptr`
before each intrinsic call and reloaded after, since intrinsics advance
the cursor through `DeserContext`.

### Stack frame

```
[sp+0..16)   frame pointer + return address (aarch64: stp x29,x30)
[sp+16..32)  callee-saved pair 1 (x19,x20 / r12,r13)
[sp+32..48)  callee-saved pair 2 (x21,x22 / r14,r15)
[sp+48..)    format-specific extra space (JSON: 32 bytes)
```

Total frame = `(48 + extra_stack + 15) & !15` (16-byte aligned).

## DeserContext

```rust
#[repr(C)]
struct DeserContext {
    input_ptr: *const u8,   // offset 0
    input_end: *const u8,   // offset 8
    error: ErrorSlot,       // offset 16
}

#[repr(C)]
struct ErrorSlot {
    code: u32,              // offset 16 (0 = ok, nonzero = error)
    offset: u32,            // offset 20 (byte position in input)
}
```

After every intrinsic call, the emitted code checks `ctx.error.code != 0`
and branches to the error exit if set. The error exit restores registers
and returns — the caller sees the error in the context.

## Intrinsics

Intrinsics are `unsafe extern "C"` Rust functions called from emitted code.
They read from `ctx.input_ptr`, advance it, and write the result to the
output pointer. On error they set `ctx.error.code`.

Signature: `fn(ctx: *mut DeserContext, out: *mut T)`

The emitted call sequence (aarch64):

```asm
str x19, [x22]           ; flush cached cursor to ctx.input_ptr
mov x0, x22              ; arg0 = ctx
add x1, x21, #offset     ; arg1 = out + field_offset
movz x8, #lo16(fn)       ; materialize 64-bit function pointer
movk x8, #hi16(fn), lsl 16
movk x8, #lo16(fn>>32), lsl 32
movk x8, #hi16(fn>>48), lsl 48
blr x8                   ; indirect call
ldr x19, [x22]           ; reload cursor (intrinsic may have advanced it)
ldr w9, [x22, #0x10]     ; load error code
cbnz w9, =>error_exit    ; branch if error
```

This is 11 instructions per intrinsic call. Future work (milestone 5.6)
will inline scalar decode logic directly, eliminating the indirect call
and cursor flush/reload.

## Format trait

```rust
trait Format {
    fn extra_stack_space(&self, fields: &[FieldEmitInfo]) -> u32;
    fn supports_inline_nested(&self) -> bool;
    fn emit_struct_fields(&self, ectx, fields, emit_field_callback);
    fn emit_read_scalar(&self, ectx, offset, scalar_type);
    fn emit_read_string(&self, ectx, offset);
}
```

The format controls **field iteration order**. This is the key abstraction:

- **Postcard** (`emit_struct_fields`): simple `for` loop in declaration order.
  No extra stack. `supports_inline_nested() = true`.
- **JSON** (`emit_struct_fields`): emits a key-matching state machine —
  read key, compare against each field name, dispatch to the matching field's
  emitter, track seen fields in a bitset, check all required fields at the end.
  Needs 32 bytes of extra stack. `supports_inline_nested() = false`.

## Nested structs

Two strategies depending on the format:

### Inlined (positional formats like postcard)

When `supports_inline_nested() = true`, nested struct fields are flattened
into the parent function. The compiler collects the inner struct's fields,
adds the parent field's byte offset to each, and emits them directly:

```
Person { name: String, age: u32, address: Address { city: String, zip: u32 } }

Emitted as a single function:
  read_string(offset=0)    ; name
  read_varint(offset=24)   ; age
  read_string(offset=32)   ; address.city  (32 = address offset + 0)
  read_varint(offset=56)   ; address.zip   (32 = address offset + 24)
```

No function call overhead between nesting levels. Recurses for deeper nesting.

### Function calls (keyed formats like JSON)

When `supports_inline_nested() = false`, each unique Shape gets its own
emitted function with a full prologue/epilogue. The compiler uses depth-first
compilation with memoization (one function per `*const Shape` identity).

Inter-function calls flush/reload the cursor and check for errors:

```asm
; aarch64
str x19, [x22]           ; flush cursor
add x0, x21, #offset     ; out + field_offset
mov x1, x22              ; ctx
bl =>nested_label         ; PC-relative call (patched by dynasmrt)
ldr x19, [x22]           ; reload cursor
ldr w9, [x22, #0x10]     ; check error
cbnz w9, =>error_exit
```

Shared inner types (e.g., `Address` used in both `home` and `work` fields)
are compiled once and called from multiple sites.

## JSON key matching

JSON's `emit_struct_fields` emits a state machine:

```
1. Zero bitset at [sp+48]
2. Call kajit_json_expect_object_start (consumes '{')
3. Peek for empty object
4. Loop:
   a. Read key → (key_ptr, key_len) at [sp+56], [sp+64]
   b. Consume ':'
   c. Linear compare: for each field, call kajit_json_key_equals
      - Match → jump to field handler, set bit in bitset
   d. No match → kajit_json_skip_value
   e. Read comma-or-end → '}' breaks loop, ',' continues
5. Check bitset has all required field bits set
```

This is a linear scan (O(fields) per key). Adequate for small structs;
future work may add perfect hashing for larger ones.

## Disassembly

Tests can inspect emitted code using yaxpeax-arm (aarch64) and yaxpeax-x86
(x86_64) as dev-dependencies. Two helpers:

- `disasm_jit(deser)` — disassemble a `CompiledDeser`'s code buffer
- `disasm_native(fn_ptr, max_bytes)` — disassemble any function pointer

Use `cargo nextest run --release disasm_ --no-capture` for optimized
output. This enables side-by-side comparison of kajit's JIT output vs
LLVM-optimized serde codegen.

For CFG-MIR text workflows, differential checking, minimization, and LLDB/JIT debugging, see `docs/pipeline-debugging.md`.

## Milestones

| # | Status | Description |
|---|--------|-------------|
| 1 | Done | Flat struct from postcard |
| 2 | Done | Flat struct from JSON |
| 3 | Done | x86_64 backend |
| 4 | Done | All primitive scalar types |
| 5 | Done | Nested structs (recursive compiler) |
| 5.5 | Done | Inline nested structs for positional formats |
| 5.6 | Done | Inline scalar intrinsics (#8) |
| 6 | Open | Enums — externally tagged (#3) |
| 7 | Open | Flatten + internally/adjacently tagged enums (#4) |
| 8 | Open | Untagged enums (#5) |
