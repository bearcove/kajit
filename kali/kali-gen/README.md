# kali-gen

The idea: define a language spec once, and have:

  * A parser (chumsky or other)
  * A formatter (those should match)
  * An AST (with, if needed, arenas etc.)
  * LSP support (via symbol resolution etc.)
  
## Defining a language

Language definitions are .styx files

```styx
/// Human-facing representation name.
name ASM

/// Canonical text file extension for ASM documents.
file_ext .k-asm

/// High-level purpose of this representation.
description "Symbolic architecture-specific assembly with canonical print round-trip and required provenance."

rules {
    // rules go here
}
```

We have to define _everything at once_ so it can get a bit busy.

### Defining rules

Everything we show off in this section is implicitly under `rules {}`

Every rule ends up being a struct, with a `Prov` (which contains `FileId`, and
`Span`)

Every rule defines how to parse+format it (syntax), how to highlight it
(if highlight is present, it's used to publish semantic tokens via LSP)

Syntax can be a literal:

```styx
rules { 
    RetKw {
        syntax ret
        highlight keyword
    }
}
```

In which case it'll match literally `ret`.

It can be a regexp (regular styx literal that starts and end in `/`):

```styx
rules {
    Register {
        syntax /[A-Za-z_][A-Za-z0-9_]*/
        highlight register
    }
}
```

You can refer to another rule from within a `syntax` expression by writing `@RuleName`.
That means “parse using the rule named `RuleName` here”, instead of matching raw text directly.
This is how larger rules are built out of smaller reusable ones.

```styx
rules {
    Ret {
        syntax @RetKw
    }
}
```

> Note: in styx, `@RetKw` is equivalent to `@RetKw@`, it's a unit value (`@`)
> tagged with `RetKw`.

`Ret` will then match exactly the same input as `RetKw`, but as a distinct rule in the grammar.
References can also be combined with other syntax forms, such as sequences.

```styx
rules {
    RetInstr {
        syntax (@RetKw @Register)
    }
}
```

## Appendix: list of semantic tokens

| Token | Description |
| --- | --- |
| namespace | For identifiers that declare or reference a namespace, module, or package. |
| class | For identifiers that declare or reference a class type. |
| enum | For identifiers that declare or reference an enumeration type. |
| interface | For identifiers that declare or reference an interface type. |
| struct | For identifiers that declare or reference a struct type. |
| typeParameter | For identifiers that declare or reference a type parameter. |
| type | For identifiers that declare or reference a type that is not covered above. |
| parameter | For identifiers that declare or reference a function or method parameters. |
| variable | For identifiers that declare or reference a local or global variable. |
| property | For identifiers that declare or reference a member property, member field, or member variable. |
| enumMember | For identifiers that declare or reference an enumeration property, constant, or member. |
| decorator | For identifiers that declare or reference decorators and annotations. |
| event | For identifiers that declare an event property. |
| function | For identifiers that declare a function. |
| method | For identifiers that declare a member function or method. |
| macro | For identifiers that declare a macro. |
| label | For identifiers that declare a label. |
| comment | For tokens that represent a comment. |
| string | For tokens that represent a string literal. |
| keyword | For tokens that represent a language keyword. |
| number | For tokens that represent a number literal. |
| regexp | For tokens that represent a regular expression literal. |
| operator | For tokens that represent an operator. |
