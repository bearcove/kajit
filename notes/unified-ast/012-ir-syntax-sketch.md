# IR Syntax Sketch

## Purpose

IR is the canonical RVSDG layer.

It should optimize for:

- semantic precision
- explicit value flow
- explicit effect flow
- explicit region structure
- mechanical comparability

It should not pretend to be source code.

## Primary Visual Units

- function
- region
- node
- result
- region arguments
- gamma / theta structure

## Desired Properties

- data dependencies are explicit
- effect/state dependencies are explicit
- node identity is visible and stable
- nesting of regions is obvious
- symbolic addresses/callees remain symbolic

## Sketch

```text
ir fn decode_MaybeBorrowedName {
  region root(%arg0: Cursor<'input>, %arg1: MaybeBorrowedName<'input>) -> () {
    %is_some = call @postcard.read_option_tag(%arg0)

    gamma %is_some
      then region (%cursor_then = %arg0, %out_then = %arg1) -> () {
        %value = call @postcard.read_str(%cursor_then)
        %updated = set_field %out_then "name", variant Option::Some { value = %value }
        yield ()
      }
      else region (%cursor_else = %arg0, %out_else = %arg1) -> () {
        %updated = set_field %out_else "name", variant Option::None {}
        yield ()
      }

    return ()
  }
}
```

## Notes

- This is only a sketch. Real RVSDG syntax may want stronger result and port notation.
- The important thing is that the graph structure is honest.
- If effect domains survive explicitly, they should be named explicitly here.

## Open Questions

1. Are results named SSA-style, node-style, or both?
2. How much port structure should be printed?
3. Are types printed on every result, or inferred except at boundaries?
4. How do we print provenance without drowning the graph?
