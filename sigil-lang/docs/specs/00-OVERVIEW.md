# Sigil Language Specification

> *"Each symbol binds intent to execution"*

## Overview

**Sigil** is a systems programming language with polysynthetic syntax design, Rust-level performance, and memory safety guarantees. It draws inspiration from agglutinative and polysynthetic natural languages where complex meanings are expressed through morpheme composition rather than isolated words.

### Design Philosophy

1. **Density of Expression** — Single expressions encode what other languages require multiple statements to express
2. **Morphemic Composition** — Operators, type modifiers, and transformations compose through affixation
3. **Evidentiality** — The type system tracks the provenance and certainty of values
4. **Zero-Cost Abstraction** — Polysynthetic ergonomics compile to optimal machine code
5. **Memory Safety** — Ownership semantics prevent data races and memory corruption at compile time
6. **Polycultural Awareness** — Color, sound, number systems, and symbolism respect diverse cultural traditions

### Influences

| Aspect | Inspiration |
|--------|-------------|
| Performance & Safety | Rust |
| Expression Density | APL, J, K |
| Morpheme Composition | Inuktitut, Mohawk, Yupik |
| Evidentiality | Quechua, Turkish, Tibetan |
| Symbolic Operators | Haskell, Scala |
| Polycultural Mathematics | Mayan vigesimal, Babylonian sexagesimal |
| World Music | 22-Shruti, Arabic maqam, Gamelan |
| Sacred Geometry | I Ching, Kabbalah, Ayurveda |
| Color Systems | Wu Xing, Chakras, Yoruba Orisha |
| Aesthetic | Daemoniorum mythological tradition |

---

## Language Name

**Sigil** /ˈsɪdʒɪl/ — From Latin *sigillum* (seal, sign)

In occult traditions, a sigil is a symbol created to encode and manifest specific intent. This mirrors the language's core principle: each expression is a dense encoding of computational intent.

Alternative names considered:
- **Glyph** — Reserved for the visual/graphical subsystem
- **Rune** — Reserved for the macro/metaprogramming system
- **Incant** — Reserved for the build/invocation toolchain

---

## Core Characteristics

### Polysynthetic Syntax

Traditional (analytic) approach:
```rust
// Rust
let result: Vec<i32> = data
    .iter()
    .filter(|x| *x > 0)
    .map(|x| x * 2)
    .collect();
```

Sigil (polysynthetic) approach:
```sigil
// Morphemes compose into single expressions
result: [i32] = data|φ>0|τ*2|vec
```

### Evidentiality Markers

Values carry provenance information in their types:

```sigil
known!    : i32 = compute()      // Direct: computed here
cached?   : i32 = store.get(k)   // Uncertain: might exist
remote~   : i32 = api.fetch()    // Reported: external source
assumed‽  : i32 = unsafe_cast()  // Paradox: trust boundary
```

### Incorporation

Noun-verb fusion creates compound operations:

```sigil
// Traditional
file.open().read().parse().validate()

// Incorporated
file·open·read·parse·validate(path)
```

---

## Document Structure

| Spec | Contents |
|------|----------|
| [01-LEXICAL](./01-LEXICAL.md) | Tokens, morphemes, sigils, and lexical structure |
| [02-SYNTAX](./02-SYNTAX.md) | Grammar, expressions, and composition rules |
| [03-TYPES](./03-TYPES.md) | Type system, evidentiality, and inference |
| [04-MEMORY](./04-MEMORY.md) | Ownership, borrowing, and lifetime semantics |
| [05-MODULES](./05-MODULES.md) | Module system, visibility, and imports |
| [06-CONCURRENCY](./06-CONCURRENCY.md) | Async, parallelism, and synchronization |
| [07-METAPROGRAMMING](./07-METAPROGRAMMING.md) | Runes (macros) and compile-time evaluation |
| [08-FFI](./08-FFI.md) | Foreign function interface and C/Rust interop |
| [09-STDLIB](./09-STDLIB.md) | Standard library overview |
| [10-TOOLING](./10-TOOLING.md) | Incant (build), Glyph (formatter), Augur (linter) |

---

## Quick Example

```sigil
// A complete program demonstrating polysynthetic style

use std·io·{Read, Write}
use std·net·TcpStream

// Evidentiality in function signatures
fn fetch_user(id: u64) -> User~ {
    // ~ marks this as externally-sourced data
    let response~ = http·get!("https://api/users/{id}")
    response~|json·parse|User·from
}

// Morpheme chain with aspect markers
fn process_users(ids: [u64]) -> [User!] {
    ids
    |par·map(fetch_user)      // Parallel map
    |φ·valid                   // Filter valid
    |σ·name                    // Sort by name
    |τ{.normalize()}          // Transform each
    |!                         // Assert to direct knowledge
}

// Entry point with incorporated file operations
fn main() {
    let users! = process_users([1, 2, 3, 4, 5])

    // Compound verb: file·create·write·close
    "users.json"|file·create·write·close(users!|json·encode)

    print!("Processed {users!.len} users")
}
```

---

## Version

This specification describes **Sigil 0.1.0** (Codename: *First Seal*)

Status: **Draft** — Under active development by Daemoniorum, Inc.
