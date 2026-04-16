# kali-gen

`kali-gen` is a language definition generator.

The idea is to define a language spec once, and have:

- a parser
- a formatter
- an AST
- editor/language-tooling metadata
- enough structure to grow into validation, symbol resolution, and LSP features

This is not just a parser generator. The goal is to define the **surface syntax** and the **AST shape** together, in one schema.

## Design principles

Kali is still in development, but the current direction is guided by a few principles:

- define syntax and AST together
- keep the core schema small
- make common cases concise
- prefer reusable templates over many built-in helpers
- grow progressively from tiny grammars to real representation languages like ASM, HIR, IR, and MIR

## Defining a language

Language definitions are `.styx` files.

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

A language definition describes:

- the top-level metadata (`name`, `file_ext`, `description`)
- a set of rules
- optionally, a set of reusable templates

The grammar and the AST are described together, so a schema can get busy quickly. That is intentional: the goal is one source of truth.

## Rule kinds

There are two core rule kinds: `@struct` and `@enum`.

### `@struct`

A `@struct` defines one AST node shape.

Its fields are inferred from named captures in `syntax`.

```styx
rules {
    Label @struct {
        syntax ({name @Ident} ":")
    }
}
```

That rule says:

- parse a `Label`
- match an `Ident`
- bind it to the field `name`
- then match the literal `:`

The generated AST node would conceptually look like:

- `Label { name, prov }`

where `prov` is the provenance/span metadata automatically attached to nodes.

### `@enum`

An `@enum` defines a sum type.

Its variants are typically written as nested rules like `EnumName.VariantName`.

```styx
rules {
    Operand @enum

    Operand.Reg @struct {
        syntax {reg @Register}
    }

    Operand.Imm @struct {
        syntax {value @Int}
    }
}
```

This says that `Operand` has multiple AST shapes:

- a register operand
- an immediate operand

A useful mental model is:

- `@struct` = one AST shape
- `@enum` = alternatives in AST shape

## Field inference

Named captures inside a `@struct` become fields on that struct.

### Simple capture

```styx
rules {
    RetInstr @struct {
        syntax (@RetKw {target @Register})
    }
}
```

This infers the field:

- `target: Register`

### Optional capture

```styx
rules {
    MaybeLabelRef @struct {
        syntax {label @maybe(@LabelName)}
    }
}
```

This infers:

- `label: Option<LabelName>`

### Repeated capture

```styx
rules {
    Block @struct {
        syntax {items @repeat0(@Item)}
    }
}
```

This infers:

- `items: Vec<Item>`

### Separated repetition

```styx
rules {
    ArgList @struct {
        syntax {args @sep1(@Expr @Comma)}
    }
}
```

This infers:

- `args: Vec<Expr>`

### Consistency rule for alternatives inside a struct

A `@struct` may use alternatives in its `syntax`, but all alternatives must infer the **same field set** with compatible types.

For example, this is fine:

```styx
rules {
    BoolLit @struct {
        syntax @alt(
            ("true" {value @TrueTag})
            ("false" {value @FalseTag})
        )
    }
}
```

because every branch binds the same field, `value`.

If different branches want different fields, that likely wants an `@enum`, not a single `@struct`.

## Syntax expressions

The syntax language is intentionally small.

A syntax expression can be:

- a literal, like `"ret"`
- a regex, like `/[A-Za-z_][A-Za-z0-9_]*/`
- a rule reference, like `@Ident`
- a sequence, like `(@MovKw {dst @Register} "," {src @Operand})`
- a named capture, like `{dst @Register}`
- a combinator form, such as `@alt(...)` or `@repeat0(...)`

### Literals

```styx
rules {
    RetKw {
        syntax "ret"
        highlight keyword
    }
}
```

This matches the literal text `ret`.

### Regexes

```styx
rules {
    Ident {
        syntax /[A-Za-z_][A-Za-z0-9_]*/
        highlight variable
    }
}
```

This matches identifier-like text.

### Rule references

A syntax expression can refer to another rule using `@RuleName`.

```styx
rules {
    Ret @struct {
        syntax @RetKw
    }
}
```

That means “parse using the rule named `RetKw` here”.

### Sequences

A sequence is written using parentheses:

```styx
rules {
    RetInstr @struct {
        syntax (@RetKw @soft_space {reg @Register})
    }
}
```

The sequence matches each element in order.

## Core combinators

Because Styx is still a document format, the current direction is to prefer **prefix combinator forms** rather than symbolic operators like `|`, `?`, `*`, or `+`.

The core combinators are:

- `@alt(...)`
- `@maybe(...)`
- `@repeat0(...)`
- `@repeat1(...)`
- `@sep0(item sep)`
- `@sep1(item sep)`

### Alternatives

```styx
rules {
    ZeroOrOne @struct {
        syntax @alt(
            "zero"
            "one"
        )
    }
}
```

`@alt(...)` means “try these alternatives”.

### Optional syntax

```styx
rules {
    SignedInt @struct {
        syntax (
            {sign @maybe(@Minus)}
            {value @Int}
        )
    }
}
```

### Repetition

```styx
rules {
    Lines @struct {
        syntax {items @repeat1(@Line)}
    }
}
```

### Separated repetition

```styx
rules {
    List @struct {
        syntax {items @sep0(@Item @Comma)}
    }
}
```

The exact punctuation of these forms may still evolve a bit, but the intended semantics are stable:

- `@maybe(x)` = zero or one `x`
- `@repeat0(x)` = zero or more `x`
- `@repeat1(x)` = one or more `x`
- `@sep0(x sep)` = zero or more `x`, separated by `sep`
- `@sep1(x sep)` = one or more `x`, separated by `sep`

## Templates

Templates are a first-class part of the design.

A template is a reusable syntax macro. It exists to factor common syntax forms without creating AST structure of its own.

That distinction is important:

- rules define AST nodes
- templates factor surface syntax

A template should not secretly invent fields. Fields still come from captures at the call site.

### Why templates matter

Templates are what make schemas ergonomic.

They are especially useful for:

- instruction families
- punctuated lists
- block wrappers
- keyword-led forms
- common delimiter/layout patterns

### Example template

```styx
templates {
    Instr2 {
        params ({kw @expr} {lhs @expr} {rhs @expr})
        body (
            $kw
            @soft_space
            $lhs
            ","
            @soft_space
            $rhs
        )
    }
}
```

Then it can be used like this:

```styx
rules {
    Mov @struct {
        syntax @Instr2("mov", {dst @Register}, {src @Operand})
    }
}
```

The important point is that the fields are still `dst` and `src`, because those bindings were introduced by the caller.

The template only arranges syntax.

## Highlighting and semantic tokens

Rules may optionally attach a `highlight`.

If present, it can be used to drive semantic token publication through LSP/editor integration.

```styx
rules {
    Register {
        syntax /[A-Za-z_][A-Za-z0-9_]*/
        highlight variable
    }
}
```

The exact tooling generated from this metadata may evolve, but the intended role is stable: syntax definitions can also annotate semantic meaning.

## A minimal worked example

This example is intentionally small, but it shows the main pieces together:

- token-like rules
- an enum
- struct variants
- field inference
- a template
- repetition

```styx
name MiniAsm
file_ext .mini-asm
description "Tiny assembly-like language used to demonstrate kali schemas."

rules {
    AsmKw {
        syntax "asm"
        highlight keyword
    }

    MovKw {
        syntax "mov"
        highlight keyword
    }

    RetKw {
        syntax "ret"
        highlight keyword
    }

    Comma {
        syntax ","
        highlight operator
    }

    Ident {
        syntax /[A-Za-z_][A-Za-z0-9_]*/
        highlight variable
    }

    Int {
        syntax /[0-9]+/
        highlight number
    }

    Register @struct {
        syntax {name @Ident}
    }

    Operand @enum

    Operand.Reg @struct {
        syntax {reg @Register}
    }

    Operand.Imm @struct {
        syntax {value @Int}
    }

    Item @enum

    Item.Label @struct {
        syntax ({name @Ident} ":")
    }

    Item.Mov @struct {
        syntax @Instr2(@MovKw, {dst @Register}, {src @Operand})
    }

    Item.Ret @struct {
        syntax @RetKw
    }

    Program @struct {
        syntax (
            @AsmKw
            @soft_space
            "{"
            {items @repeat0(@Item)}
            "}"
        )
    }
}

templates {
    Instr2 {
        params ({kw @expr} {lhs @expr} {rhs @expr})
        body (
            $kw
            @soft_space
            $lhs
            @Comma
            @soft_space
            $rhs
        )
    }
}
```

This is enough to express:

- labels
- instructions
- operand alternatives
- a list of items inside a program

It is not “full asm”, but it demonstrates the shape Kali is aiming for.

## What belongs in the core vs templates

A good rule of thumb is:

### Core schema primitives

These should be built into the language definition model:

- `@struct`
- `@enum`
- rule references
- sequence
- named capture
- `@alt`
- `@maybe`
- `@repeat0`
- `@repeat1`
- `@sep0`
- `@sep1`

### Templates

These should usually be expressed by the schema author:

- `Instr0`
- `Instr1`
- `Instr2`
- `Instr3`
- brace/block helpers
- list/layout helpers
- language-specific conveniences

This keeps Kali itself small, while still making schemas pleasant to write.

## What is still open

The broad direction is clear, but many details are still intentionally unsettled.

Open questions include:

- exact template parameter kinds
- exact syntax for some combinator payloads
- how much formatter control should be implicit vs explicit
- how comments/trivia should be represented
- how much validation can be generated mechanically
- how symbol resolution and semantic analysis hooks should plug in
- how schema modularity/imports should work

The README should be read as a description of the **current model and direction**, not a frozen final spec.

## Near-term roadmap

Near-term goals for `kali-gen` are:

1. finalize the schema model for `@struct`, `@enum`, captures, and core combinators
2. finalize template syntax and expansion semantics
3. express a small real grammar in Kali, likely ASM first
4. generate AST / parser / formatter from that schema
5. iterate toward HIR / IR / MIR use cases

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