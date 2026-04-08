# HIR Syntax Sketch

## Purpose

HIR is the human-semantic layer.

It should optimize for:

- readability
- preserving names
- explicit scopes and control flow
- good diagnostics
- symbolic references

It should not look like a dump format.

## Primary Visual Units

- module
- type definitions
- function definitions
- locals
- statements
- expressions

## Desired Properties

- function bodies are easy to read top to bottom
- locals have names, not just synthetic IDs
- types are explicit where useful, not everywhere out of fear
- callsites are symbolic
- docs/comments/provenance can be attached cleanly
- canonical formatting is stable

## Sketch

```text
module {
  type Cursor<'input> = struct {
    bytes: Slice<'input, u8>
    pos: u64
  }

  type MaybeBorrowedName<'input> = struct {
    name: Option<Str<'input>>
  }

  fn decode_MaybeBorrowedName<'input>(
    cursor: Cursor<'input>,
    out: MaybeBorrowedName<'input>,
  ) -> unit {
    let option_is_some: bool
    let option_value: Str<'input>

    option_is_some = call @postcard.read_option_tag(cursor)

    if option_is_some {
      option_value = call @postcard.read_str(cursor)
      out.name = Some { value = option_value }
    } else {
      out.name = None {}
    }

    return
  }
}
```

## Notes

- No `callables [...]` table.
- Calls name symbolic callees directly.
- Local IDs may still exist internally, but should not be the primary printed identity.
- We may still want a canonical way to print scope IDs or provenance anchors when debugging.

## Open Questions

1. Should locals print only names, or `name#id` for stability?
2. Should pattern matching look source-like or more explicit?
3. Should HIR text include provenance annotations in canonical form, or only optionally?
4. How much sugar should exist around variants and structs?
