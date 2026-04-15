# kajit-foundation

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

A simple keyword:

```
rules { 
  
}
```
