# ASM Syntax Sketch

## Purpose

ASM is symbolic low-level code before final relocation.

It should optimize for:

- familiarity
- symbolic relocations
- explicit labels
- architecture-specific readability

It should not contain resolved host pointers in canonical form.

## Primary Visual Units

- directives
- labels
- instructions
- relocations / extern references

## Sketch

```text
fn decode_MaybeBorrowedName:
entry:
  stp x29, x30, [sp, #-16]!
  mov x29, sp

  bl @postcard.read_option_tag
  cbz x0, else

then:
  bl @postcard.read_str
  b exit

else:
  b exit

exit:
  ldp x29, x30, [sp], #16
  ret
```

## Notes

- Actual per-arch syntax may remain close to native assembler.
- The important part is that relocation-bearing operands remain symbolic.
- If there is extra debug provenance, it should be attached in a disciplined way, not as random comments.

## Open Questions

1. How close to real assembler do we want the canonical text?
2. Should comments carry provenance, scheduling info, or pass annotations?
3. What is the boundary between canonical ASM and debug-enriched ASM?
