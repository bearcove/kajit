# kajit

JIT deserializer for Rust. Generates native machine code at startup from
[facet](https://github.com/facet-rs/facet) type reflection. No proc macros,
no schema files — `#[derive(Facet)]` on your types is all it needs.

## How it works

kajit walks a type's `Shape` (facet's reflection metadata) and compiles a
deserializer through a multi-stage pipeline:

1. **IR** — the type shape is lowered into an RVSDG (Regionalized Value State
   Dependence Graph). Structured control flow (loops, branches) maps directly
   onto theta and gamma nodes. Data flow is explicit; side effects (cursor
   movement, output writes, error state) are tracked as typed tokens through
   the graph.

2. **Passes** — the IR is optimized before codegen: bounds checks are
   coalesced, loop-invariant setup is hoisted out of theta bodies, and small
   lambdas are inlined.

3. **Linearization** — the RVSDG is flattened to a linear instruction sequence
   (`LinearIr`) with explicit labels and branches.

4. **Register allocation** — the linear IR is lowered to a CFG-based machine IR
   (`RaProgram`) and fed into [regalloc2](https://crates.io/crates/regalloc2).

5. **Codegen** — the allocated program is emitted as aarch64 or x86_64 machine
   code via [dynasmrt](https://crates.io/crates/dynasmrt) and mapped executable.

One function per type, composed recursively. The goal is to generate the same
code a human would write by hand for each specific type — no runtime dispatch,
no intermediate allocations, no format-generic slow paths.

## Design direction

Format-specific runtime helpers (varint loops, JSON string scanning, decimal
parsing) are being progressively replaced by IR — each operation expressed
directly in the graph and compiled inline. The only permanent runtime callouts
are heap allocation and `Default::default()` for missing optional fields.

The IR pipeline is structured as a set of focused crates (`kajit-ir`,
`kajit-lir`, `kajit-mir`) with stable boundaries, so the infrastructure can be
reused for future formats and encoders without rebuilding from scratch.

## What it supports

**Formats:**

- Postcard — varint integers, zigzag signed integers, length-prefixed strings,
  raw bytes for u8/i8, IEEE 754 LE floats
- JSON — objects, arrays, all escape sequences including `\uXXXX` and surrogate
  pairs, key-order-independent field matching

**Types:**

- All primitive scalars: `bool`, `u8`–`u64`, `i8`–`i64`, `f32`, `f64`
- `String` (UTF-8 validated, heap-allocated)
- Structs (flat, nested, `#[facet(flatten)]`)
- Enums with `#[repr(u8)]`: unit, struct, and tuple variants
- Enum tagging: external, adjacent, internal, untagged
- `Option<T>`
- `Vec<T>` with exact-capacity allocation

**Architectures:**

- aarch64 (native on Apple Silicon, CI on depot.dev ARM runners)
- x86_64 (CI on depot.dev, local testing via Docker on aarch64 hosts)

## Performance

Varint decoding has a single-byte fast path inlined into the hot loop.
Vec loops over scalar elements write directly to the output buffer with no
intermediate copies. String deserialization discovers `String`'s field layout
at startup and writes `(ptr, len, cap)` directly.

Benchmarks are in `benches/deser_postcard.rs` and `benches/deser_json.rs`,
runnable with `cargo bench`.

## License

Licensed under either of:

- MIT license ([LICENSE-MIT](LICENSE-MIT) or http://opensource.org/licenses/MIT)
- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE) or http://www.apache.org/licenses/LICENSE-2.0)

at your option.
