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
- port / result
- region arguments
- gamma / theta structure

## Desired Properties

- data dependencies are explicit
- effect/state dependencies are explicit
- node identity is visible and stable
- region interfaces are explicit
- gamma/theta contracts are explicit
- symbolic addresses/callees remain symbolic

## Sketch

```text
ir fn decode_MaybeBorrowedName {
  region root(
    data %arg0: Cursor<'input>,
    data %arg1: MaybeBorrowedName<'input>,
    state %mem0: memory,
  ) -> (
    state %mem_out: memory
  ) {
    node %n0 = call @postcard.read_option_tag(
      data %arg0
      state %mem0
    ) -> (
      data %is_some: bool
      state %mem1: memory
    )

    node %n1 = gamma(
      pred %is_some
      inputs {
        data %arg0
        data %arg1
        state %mem1
      }
      then region(
        data %cursor_then: Cursor<'input>,
        data %out_then: MaybeBorrowedName<'input>,
        state %mem_then_in: memory,
      ) -> (
        state %mem_then_out: memory
      ) {
        node %n2 = call @postcard.read_str(
          data %cursor_then
          state %mem_then_in
        ) -> (
          data %value: Str<'input>
          state %mem2: memory
        )

        node %n3 = set_field(
          data %out_then
          field "name"
          data variant Option::Some { value = %value }
        ) -> (
          data %out_updated: MaybeBorrowedName<'input>
        )

        yield state %mem2
      }
      else region(
        data %cursor_else: Cursor<'input>,
        data %out_else: MaybeBorrowedName<'input>,
        state %mem_else_in: memory,
      ) -> (
        state %mem_else_out: memory
      ) {
        node %n4 = set_field(
          data %out_else
          field "name"
          data variant Option::None {}
        ) -> (
          data %out_updated: MaybeBorrowedName<'input>
        )

        yield state %mem_else_in
      }
    ) -> (
      state %mem_out: memory
    )

    return state %mem_out
  }
}
```

## Notes

- This sketch is intentionally less source-like and more graph-honest.
- The important thing is not this exact syntax, but that ports, region interfaces, and state threading are explicit.
- If effect domains survive explicitly, they should be named explicitly here.

## Open Questions

1. Are results named SSA-style, node-style, or both?
2. How much port structure should be printed in canonical form?
3. Are types printed on every result, or inferred except at boundaries?
4. How do we print provenance without drowning the graph?
5. Should state domains always be printed, even when there is only one?
