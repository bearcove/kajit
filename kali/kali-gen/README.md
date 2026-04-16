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

## Tiny Styx primer

Kali schemas are written in Styx, so it helps to understand the substrate first.

The most important thing to keep in mind is this:

- Styx is a **document format**
- Kali schemas are **Styx documents**
- forms like `@struct`, `@enum`, `@maybe`, and `@repeat0` are **tagged values**, not declaration keywords

That means a line like:

```styx
Label @struct{
    syntax ({name @Ident} ":")
}
```

should be read as:

- key: `Label`
- value: a value tagged `struct`
- payload: an object containing fields like `syntax`

not as “declare a struct named `Label`”.

### Objects

At the top level, a Styx document is an object: a set of keys with values.

```styx
name ASM
file_ext .k-asm
description "Symbolic architecture-specific assembly."
```

Here, `name`, `file_ext`, and `description` are keys.

### Tagged values

A value can be tagged. For Kali, tags are how the schema describes structure.

```styx
Operand @enum
```

Here, the value of `Operand` is tagged with `enum`.

A tagged value can also carry a payload:

```styx
Label @struct{
    syntax ({name @Ident} ":")
}
```

Here, the value of `Label` is tagged with `struct`, and its payload is an object.

### Nested objects and payloads

Tags can apply to different shapes of payloads:

- objects: `@struct{ ... }`
- sequences/grouped forms: `@repeat0(...)`
- unit-like tagged values with no payload: `@enum`

The exact shapes available depend on the schema being parsed, but the important point is that these are still just Styx values.

### How Kali uses Styx

Kali uses Styx objects and tagged values to describe:

- rule kinds like `@struct` and `@enum`
- syntax combinators like `@maybe(...)` and `@repeat0(...)`
- metadata like doc comments
- reusable templates

So when reading the rest of this README, try to think in terms of:

- keys and values
- tags and payloads
- objects and sequences

rather than a custom declaration language.

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

Some rules are concrete language rules.
Some rules are reusable templates defined with `@template{...}`.

The grammar and the AST are described together, so a schema can get busy quickly. That is intentional: the goal is one source of truth.

## Rule kinds

There are three important rule-value forms: `@struct`, `@enum`, and `@template`.

### `@struct`

A `@struct` defines one AST node shape.

Its fields are inferred from named captures in `syntax`.

```styx
rules {
    Label @struct{
        syntax ({name @Ident} ":")
    }
}
```

That rule says:

- parse a `Label`
- match an `Ident`
- bind it to the field `name`
- then match the literal `:`

For example, it parses text like:

- `entry:`
- `loop_start:`

and canonical formatting would emit the same shape:

- `entry:`
- `loop_start:`

The generated Rust type definition would look roughly like:

```rust
pub struct Label {
    pub name: Ident,
    pub prov: Prov,
}
```

where `prov` is the provenance/span metadata automatically attached to nodes.

### `@enum`

An `@enum` defines a sum type.

Its variants are typically written as nested rules like `EnumName.VariantName`.

```styx
rules {
    Operand @enum

    Operand.Reg @struct{
        syntax {reg @Register}
    }

    Operand.Imm @struct{
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

The generated Rust type definitions would look roughly like:

```rust
pub enum Operand {
    Reg(OperandReg),
    Imm(OperandImm),
}

pub struct OperandReg {
    pub reg: Register,
    pub prov: Prov,
}

pub struct OperandImm {
    pub value: Int,
    pub prov: Prov,
}
```

### `@template`

A `@template` is a reusable rule definition.

Like everything else in Styx, it is just a tagged value. It lives in `rules {}` alongside ordinary rules.

A template has:

- `params`
- `body`

and its body is just another rule-shaped value.

```styx
rules {
    Keyword @template{
        params ({text @expr})
        body {
            syntax $text
            highlight keyword
        }
    }

    RetKw @Keyword("ret")
}
```

This should be read as:

- `Keyword` is a reusable template rule
- `RetKw` is a rule defined by applying that template

Templates are for factoring repeated rule shapes. They are not a separate subsystem and they do not need their own top-level section.

## Field inference

Named captures inside a `@struct` become fields on that struct.

### Simple capture

```styx
rules {
    RetInstr @struct{
        syntax (@RetKw {target @Register})
    }
}
```

This infers the field:

- `target: Register`

For example, with `RetKw = "ret"` and a register rule, this would parse:

- `ret x0`

and format back to the canonical spelling:

- `ret x0`

The generated Rust type definition would look roughly like:

```rust
pub struct RetInstr {
    pub target: Register,
    pub prov: Prov,
}
```

### Optional capture

```styx
rules {
    MaybeLabelRef @struct{
        syntax {label @maybe(@LabelName)}
    }
}
```

This infers:

- `label: Option<LabelName>`

For example, if this rule is used inside a larger sequence, it can accept either:

- `target_label`
- nothing at all

and formatting would either emit the label or omit that position entirely, depending on whether the field is `Some(...)` or `None`.

The generated Rust type definition would look roughly like:

```rust
pub struct MaybeLabelRef {
    pub label: Option<LabelName>,
    pub prov: Prov,
}
```

### Repeated capture

```styx
rules {
    Program @struct{
        syntax (
            "{"
            {items @repeat0(@Item)}
            "}"
        )
    }
}
```

This infers:

- `items: Vec<Item>`

For example, it parses braced sequences like:

- `{}`
- `{ entry: ret }`

and canonical formatting would emit braces even when the list is empty:

- `{}`
- `{ entry: ret }`

The generated Rust type definition would look roughly like:

```rust
pub struct Program {
    pub items: Vec<Item>,
    pub prov: Prov,
}
```

### Separated repetition

```styx
rules {
    ArgList @struct{
        syntax {args @sep1(@Expr @Comma)}
    }
}
```

This infers:

- `args: Vec<Expr>`

For example, it parses comma-separated text like:

- `x`
- `x, y, z`

and formats it with canonical comma spacing:

- `x`
- `x, y, z`

The generated Rust type definition would look roughly like:

```rust
pub struct ArgList {
    pub args: Vec<Expr>,
    pub prov: Prov,
}
```

### Variation belongs in regexes or enums

Kali does not currently have a general-purpose inline alternation form.

Instead:

- use regexes for lexical variation
- use `@enum` for structural variation

For example, if a language wants to accept multiple token spellings, that can live in a token-like rule. If it wants multiple AST shapes, that should be modeled explicitly as an enum with variants.

## Syntax expressions

The syntax language is intentionally small.

A syntax expression can be:

- a literal, like `"ret"`
- a regex, like `/[A-Za-z_][A-Za-z0-9_]*/`
- a rule reference, like `@Ident`
- a sequence, like `(@MovKw {dst @Register} "," {src @Operand})`
- a named capture, like `{dst @Register}`
- a tagged combinator form, such as `@repeat0(...)`

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

So it parses:

- `ret`

and formats as:

- `ret`

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

For example, it parses:

- `x0`
- `entry`
- `tmp_42`

and formats them back as written, subject to whatever canonicalization the enclosing rule imposes.

### Rule references

A syntax expression can refer to another rule using `@RuleName`.

```styx
rules {
    Ret @struct{
        syntax @RetKw
    }
}
```

That means “parse using the rule named `RetKw` here”.

So if `RetKw` matches `ret`, then this rule also parses:

- `ret`

and formats as:

- `ret`

### Sequences

A sequence is written using parentheses:

```styx
rules {
    RetInstr @struct{
        syntax (@RetKw @soft_space {reg @Register})
    }
}
```

The sequence matches each element in order.

For example, this parses:

- `ret x0`

and formats canonically as:

- `ret x0`

because the sequence explicitly includes both the keyword and the separating layout.

The generated Rust type definition would look roughly like:

```rust
pub struct RetInstr {
    pub reg: Register,
    pub prov: Prov,
}
```

## Layout controls

Formatting is part of the schema, so layout-related syntax forms matter just as much as tokens and rule references.

Kali needs to be able to express not only what parses, but how canonical formatting should be laid out. That includes soft spacing, line breaks, and indentation-sensitive structure.

Important layout forms include:

- `@soft_space` — a formatting-aware space boundary
- `@newline` — a line break
- indentation-related helpers, such as `@indent(...)`, for nested block content

These are especially important for line-oriented representations like ASM, HIR, IR, and MIR.

### Soft space

```styx
rules {
    RetInstr @struct{
        syntax (@RetKw @soft_space {reg @Register})
    }
}
```

A `@soft_space` marks a spacing boundary that should round-trip canonically in formatted output without having to model raw whitespace as ordinary tokens.

For example, a rule using `(@RetKw @soft_space {reg @Register})` might parse variants like:

- `ret x0`
- `ret   x0`

but format them canonically as:

- `ret x0`

### Newlines

```styx
rules {
    Program @struct{
        syntax (
            "{"
            @newline
            {items @repeat0((@Item @newline))}
            "}"
        )
    }
}
```

This makes line structure explicit in the syntax instead of treating it as an afterthought.

For example, it parses block-shaped text like:

- `{
ret
}`
- `{
ret
mov x0, 42
}`

and a formatter can normalize that into a canonical line-based layout.

### Indentation

```styx
rules {
    Program @struct{
        syntax (
            "{"
            @newline
            @indent({items @repeat0((@Item @newline))})
            "}"
        )
    }
}
```

The exact shape of indentation helpers may still evolve, but the schema needs a way to express nested layout intentionally, because formatter generation depends on it.

For example, a formatter for the rule above would typically emit:

- `{
    ret
}`
- `{
    ret
    mov x0, 42
}`

rather than leaving indentation implicit or ad hoc.

## Core combinators

Because Styx is still a document format, the current direction is to prefer **prefix combinator forms** rather than symbolic operators like `?`, `*`, or `+`.

The core combinators are:

- `@maybe(...)`
- `@repeat0(...)`
- `@repeat1(...)`
- `@sep0(item sep)`
- `@sep1(item sep)`

### Optional syntax

```styx
rules {
    SignedInt @struct{
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
    Program @struct {
        syntax (
            "{"
            @newline
            @indent({items @repeat1((@Line @newline))})
            "}"
        )
    }
}
```

This parses one-or-more line-oriented items inside braces, such as:

- `{
    ret
}`
- `{
    ret
    mov x0, 42
}`

and formats them with one item per line in the indented block.

### Separated repetition

```styx
rules {
    List @struct{
        syntax {items @sep0(@Item @Comma)}
    }
}
```

This parses list text like:

- ``
- `x`
- `x, y, z`

and formats it with canonical comma placement and spacing:

- ``
- `x`
- `x, y, z`

The exact punctuation of these forms may still evolve a bit, but the intended semantics are stable:

- `@maybe(x)` = zero or one `x`
- `@repeat0(x)` = zero or more `x`
- `@repeat1(x)` = one or more `x`
- `@sep0(x sep)` = zero or more `x`, separated by `sep`
- `@sep1(x sep)` = one or more `x`, separated by `sep`

## Templates

Templates are a first-class part of the design.

A template is a reusable rule definition written as `@template{...}` inside `rules {}`.

That means:

- templates and ordinary rules live in the same namespace
- a template has `params` and a `body`
- applying a template produces a rule body

Templates are useful because they let you factor repeated rule shapes without inventing a separate template language.

### Why templates matter

Templates are what make schemas ergonomic.

They are especially useful for:

- keyword-like token rules
- operator-like token rules
- instruction families
- punctuated lists
- common delimiter/layout patterns

### Example template

```styx
rules {
    Instr2 @template{
        params ({kw @expr} {lhs @expr} {rhs @expr})
        body {
            syntax (
                $kw
                @soft_space
                $lhs
                ","
                @soft_space
                $rhs
            )
        }
    }

    Mov @struct{
        syntax @Instr2("mov", {dst @Register}, {src @Operand})
    }
}
```

The important point is that the fields are still `dst` and `src`, because those bindings were introduced by the caller.

So this rule parses text like:

- `mov x0, 42`
- `mov rax, rbx`

and formats it canonically as:

- `mov x0, 42`
- `mov rax, rbx`

The generated Rust type definition would look roughly like:

```rust
pub struct Mov {
    pub dst: Register,
    pub src: Operand,
    pub prov: Prov,
}
```

The template just factors the repeated rule body shape.

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
- a template rule
- repetition

```styx
name MiniAsm
file_ext .mini-asm
description "Tiny assembly-like language used to demonstrate kali schemas."

rules {
    Keyword @template{
        params ({text @expr})
        body {
            syntax $text
            highlight keyword
        }
    }

    Instr2 @template{
        params ({kw @expr} {lhs @expr} {rhs @expr})
        body {
            syntax (
                $kw
                @soft_space
                $lhs
                @Comma
                @soft_space
                $rhs
            )
        }
    }

    AsmKw @Keyword("asm")
    MovKw @Keyword("mov")
    RetKw @Keyword("ret")

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

    Register @struct{
        syntax {name @Ident}
    }

    Operand @enum

    Operand.Reg @struct{
        syntax {reg @Register}
    }

    Operand.Imm @struct{
        syntax {value @Int}
    }

    Item @enum

    Item.Label @struct{
        syntax ({name @Ident} ":")
    }

    Item.Mov @struct{
        syntax @Instr2(@MovKw, {dst @Register}, {src @Operand})
    }

    Item.Ret @struct{
        syntax @RetKw
    }

    Program @struct{
        syntax (
            @AsmKw
            @soft_space
            "{"
            @newline
            @indent({items @repeat0((@Item @newline))})
            "}"
        )
    }
}
```

This is enough to express:

- labels
- instructions
- operand alternatives
- a list of items inside a program

For example, it can parse text like:

- `asm {
    entry:
    mov x0, 42
    ret
}`

and format it back into the same canonical layout:

- `asm {
    entry:
    mov x0, 42
    ret
}`

The generated Rust types would look roughly like:

```rust
pub struct Register {
    pub name: Ident,
    pub prov: Prov,
}

pub enum Operand {
    Reg(OperandReg),
    Imm(OperandImm),
}

pub struct OperandReg {
    pub reg: Register,
    pub prov: Prov,
}

pub struct OperandImm {
    pub value: Int,
    pub prov: Prov,
}

pub enum Item {
    Label(ItemLabel),
    Mov(ItemMov),
    Ret(ItemRet),
}

pub struct ItemLabel {
    pub name: Ident,
    pub prov: Prov,
}

pub struct ItemMov {
    pub dst: Register,
    pub src: Operand,
    pub prov: Prov,
}

pub struct ItemRet {
    pub prov: Prov,
}

pub struct Program {
    pub items: Vec<Item>,
    pub prov: Prov,
}
```

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
- `@maybe`
- `@repeat0`
- `@repeat1`
- `@sep0`
- `@sep1`

### Template rules

These should usually be expressed by the schema author as rules defined with `@template{...}`:

- `Keyword`
- `Operator`
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

- exact template parameter syntax
- exact syntax for some combinator payloads
- how much formatter control should be implicit vs explicit
- how comments/trivia should be represented
- how much validation can be generated mechanically
- how symbol resolution and semantic analysis hooks should plug in
- how schema modularity/imports should work

The README should be read as a description of the **current model and direction**, not a frozen final spec.

## Near-term roadmap

Near-term goals for `kali-gen` are:

1. finalize the schema model for `@struct`, `@enum`, `@template`, captures, and core combinators
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
