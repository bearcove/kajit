# MIR Syntax Sketch

## Purpose

MIR is the executable CFG layer.

It should optimize for:

- block structure
- explicit branches
- explicit vregs
- explicit edge arguments / phi structure
- machine-adjacent clarity

## Primary Visual Units

- function
- blocks
- instructions
- terminators
- block parameters or phi-equivalents

## Desired Properties

- block boundaries are visually strong
- defs and uses are obvious
- branch arguments are explicit
- symbolic relocations remain symbolic
- easy to diff before and after optimization passes

## Sketch

```text
mir fn decode_MaybeBorrowedName(%arg0: vreg, %arg1: vreg) -> ()

bb0:
  %v0 = call @postcard.read_option_tag(%arg0)
  br %v0, bb1(%arg0, %arg1), bb2(%arg0, %arg1)

bb1(%cursor: vreg, %out: vreg):
  %v1 = call @postcard.read_str(%cursor)
  %v2 = variant Option::Some { value = %v1 }
  %v3 = set_field %out, "name", %v2
  jump bb3()

bb2(%cursor: vreg, %out: vreg):
  %v4 = variant Option::None {}
  %v5 = set_field %out, "name", %v4
  jump bb3()

bb3:
  ret
```

## Open Questions

1. Should block parameters be the canonical phi representation?
2. How much type information should be printed?
3. Should clobbers, constraints, or regalloc hints live in canonical MIR text?
4. What provenance should an instruction carry visibly, if any?
