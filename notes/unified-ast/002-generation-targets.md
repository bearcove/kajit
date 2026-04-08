# Generation Targets For Version 1

This note describes what the first schema-driven generator should emit.

It is intentionally narrow.

## Inputs

For version 1, the input is a repr schema written in Styx.

Example pilot files:

- `notes/unified-ast/pilot/hir.repr.styx`
- `notes/unified-ast/pilot/mir.repr.styx`

Those files are the source of truth for the pilot.

## Outputs

### 1. Rust AST Types

Generated:

- structs for product nodes
- enums for sum nodes
- field types
- lightweight constructors
- obvious derives where appropriate

Not generated yet:

- clever builders
- arena/storage optimizations
- hand-tuned ergonomics

### 2. Visitor / Walker / Folder Scaffolding

Generated:

- immutable visitor trait
- mutable visitor trait
- fold/transform trait
- default recursive traversal

This is one of the highest-value early wins because it removes repetitive boilerplate immediately.

### 3. Parse / Display Scaffolding

Generated:

- parser skeletons tied to the declared syntax model
- formatter / display skeletons
- canonical-print entry points

Version 1 does not need fully magical parser generation. Even a structured scaffold that eliminates repetitive wiring is useful.

### 4. Round-Trip Harnesses

Generated:

- `parse(print(x)) == x` style harnesses
- per-repr round-trip smoke tests
- snapshot helpers if useful

The point is to make round-tripping a standard property, not a pile of custom tests.

### 5. Provenance Shells

Generated:

- provenance field placement
- constructor parameters or defaults
- traversal support so provenance is not silently dropped

Version 1 does not need perfect provenance semantics. It does need a consistent place for provenance to live.

## Explicitly Deferred

Version 1 should not try to generate:

- rewrite DSLs
- lint DSLs
- tree-sitter grammars
- LSP support
- advanced validation engines
- optimized storage backends

Those are later layers, not bootstrap requirements.

## Success Criteria

The pilot is successful if:

1. HIR and MIR schemas feel descriptive rather than decorative.
2. Generated ASTs are usable enough that hand-authored equivalents look wasteful.
3. Generated traversal removes a lot of repetitive code.
4. Parse / display drift gets harder to introduce.
5. Provenance becomes structurally unavoidable instead of optional by omission.

## Failure Modes To Avoid

1. The schema is too weak, so important structure is still handwritten elsewhere.
2. The schema is too grand, so version 1 turns into a meta-compiler project.
3. The generated code is so rigid or ugly that nobody wants to use it.
4. We generate syntax machinery before stage contracts are settled.

## Recommendation

Treat the pilot as successful if it makes HIR and MIR cheaper to evolve and harder to accidentally break.

That is enough.
