# LIR Syntax Sketch

## Purpose

LIR is only worth keeping if it has a real role.

Two possibilities:

1. It is a true pipeline boundary with a clear contract.
2. It is an implementation detail and should eventually disappear.

If it stays, its syntax should optimize for:

- linear order
- explicit temporaries
- explicit labels
- low ceremony

## Primary Visual Units

- block-like labels or sections
- linear instructions
- operands
- symbolic references

## Sketch

```text
lir fn decode_MaybeBorrowedName
entry:
  %0 = call @postcard.read_option_tag %arg0
  br_if %0, then, else

then:
  %1 = call @postcard.read_str %arg0
  %2 = make_variant Option::Some { value = %1 }
  %3 = set_field %arg1 "name", %2
  jump exit

else:
  %4 = make_variant Option::None {}
  %5 = set_field %arg1 "name", %4
  jump exit

exit:
  ret
```

## Open Questions

1. Does LIR deserve a first-class text format at all?
2. If yes, is it closer to a debug dump or a real interchange format?
3. Should it share syntax conventions with MIR or intentionally differ?
