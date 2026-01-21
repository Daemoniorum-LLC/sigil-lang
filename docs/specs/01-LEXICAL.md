# Sigil Lexical Specification

## 1. Character Set

Sigil source files are UTF-8 encoded. The language uses both ASCII operators and Unicode sigils for maximum expressiveness.

### 1.1 Reserved Sigils (Primary Morphemes)

These single-character sigils form the core morpheme inventory:

| Sigil | Name | Category | Meaning |
|-------|------|----------|---------|
| `τ` (tau) | Transform | Operation | Map/transform elements |
| `φ` (phi) | Filter | Operation | Filter/select elements |
| `σ` (sigma) | Sort | Operation | Sort/order elements |
| `ρ` (rho) | Reduce | Operation | Fold/reduce to single value |
| `λ` (lambda) | Lambda | Operation | Anonymous function |
| `π` (pi) | Project | Operation | Select fields/project |
| `δ` (delta) | Difference | Operation | Set difference/change |
| `ε` (epsilon) | Empty | Value | Empty/null/none |
| `ω` (omega) | End | Control | Terminal/final |
| `α` (alpha) | First | Access | First element |
| `Ω` (Omega) | Last | Access | Last element |
| `∀` | ForAll | Quantifier | Universal quantification |
| `∃` | Exists | Quantifier | Existential quantification |
| `∈` | In | Membership | Element of |
| `∉` | NotIn | Membership | Not element of |
| `→` | Arrow | Flow | Unidirectional flow |
| `⇒` | FatArrow | Flow | Pure/guaranteed flow |
| `↔` | BiArrow | Flow | Bidirectional/reversible |
| `∷` | Cons | Structure | Type annotation/cons |
| `⊕` | XPlus | Combine | Exclusive addition/XOR |
| `⊗` | XTimes | Combine | Tensor product |
| `⊥` | Bottom | Type | Never type/unreachable |
| `⊤` | Top | Type | Any type/unknown |

### 1.2 ASCII Equivalents

For environments without Unicode support, ASCII digraphs are available:

| Sigil | ASCII | Example |
|-------|-------|---------|
| `τ` | `\t` or `map` | `data\|map{x * 2}` |
| `φ` | `\f` or `filter` | `data\|filter{x > 0}` |
| `σ` | `\s` or `sort` | `data\|sort.name` |
| `ρ` | `\r` or `reduce` | `data\|reduce(+)` |
| `λ` | `\l` or `fn` | `\l x -> x + 1` |
| `→` | `->` | `fn(x) -> y` |
| `⇒` | `=>` | `pure(x) => y` |
| `↔` | `<->` | `a <-> b` |
| `∷` | `::` | `x :: i32` |

---

## 2. Morpheme Categories

### 2.1 Operator Morphemes (Transformational)

Operator morphemes modify data flow and transform values.

```sigil
// Basic operators
data|τ{expr}      // Transform: apply expr to each
data|φ{pred}      // Filter: keep where pred is true
data|σ.field      // Sort: order by field
data|ρ(op, init)  // Reduce: fold with operator

// Chained morphemes compose left-to-right
users|φ.active|σ.created|τ{.name}|α
// Filter active → sort by created → extract names → first
```

### 2.2 Evidentiality Morphemes (Epistemological)

These suffixes encode the provenance and certainty of values:

| Morpheme | Name | Meaning | Propagation |
|----------|------|---------|-------------|
| `!` | Known | Direct computation | Downgrades on external |
| `?` | Uncertain | Possibly absent | Requires handling |
| `~` | Reported | External source | Untrusted by default |
| `‽` | Paradox | Trust boundary | Explicit assertion |

```sigil
// Evidentiality in practice
let computed! = 2 + 2           // Known: we computed it
let cached? = cache.get(key)    // Uncertain: might not exist
let fetched~ = api.get(url)     // Reported: from outside
let trusted‽ = unsafe { ptr }   // Paradox: trust assertion

// Evidentiality flows through operations
let result! = computed! + 1     // Known + literal = Known
let result? = computed! + cached?  // Known + Uncertain = Uncertain
let result~ = fetched~|parse    // Reported stays Reported
```

### 2.3 Aspect Morphemes (Temporal)

Encode the temporal state of operations:

| Morpheme | Name | Meaning | Example |
|----------|------|---------|---------|
| `·ing` | Progressive | Ongoing/streaming | `process·ing(stream)` |
| `·ed` | Perfective | Completed | `process·ed(data)` |
| `·able` | Potential | Capability | `process·able(input)` |
| `·ive` | Resultative | Produces result | `destruct·ive(x)` |

```sigil
// Aspect-aware APIs
fn stream·ing(source: Reader) -> Stream<Chunk>  // Returns ongoing stream
fn load·ed(path: Path) -> Data!                 // Returns completed data
fn parse·able(input: &str) -> bool              // Returns capability check

// Aspect affects control flow
match process·ing(data) {
    Complete(result!) => handle(result!),
    Ongoing(stream) => stream|τ{handle}|await,
    Failed(err~) => log(err~),
}
```

### 2.4 Valency Morphemes (Arity/Direction)

| Morpheme | Name | Arity | Meaning |
|----------|------|-------|---------|
| `·un` | Nullary | 0 | No arguments |
| `·mono` | Unary | 1 | Single argument |
| `·bi` | Binary | 2 | Two arguments |
| `·poly` | Variadic | N | Multiple arguments |
| `·in` | Inward | → | Consuming |
| `·out` | Outward | ← | Producing |
| `·mut` | Mutable | ↔ | In-place modification |

```sigil
// Valency in function types
type Consumer·in<T> = fn(T) -> ()
type Producer·out<T> = fn() -> T
type Transformer·mut<T> = fn(&mut T) -> ()

// Clear intent from signature
fn sort·mut(data: &mut [T])     // Mutates in place
fn sort·out(data: &[T]) -> [T]  // Returns new sorted
```

---

## 3. Compound Formation

### 3.1 Incorporation (Noun-Verb Fusion)

The `·` (middle dot) fuses nouns and verbs into compound operations:

```sigil
// Without incorporation (verbose)
let file = File::open(path)?;
let content = file.read_to_string()?;
let data = json::parse(&content)?;
file.close();

// With incorporation (polysynthetic)
let data! = path|file·open·read·parse·close

// The compiler desugars this to efficient code
// with proper error handling and resource cleanup
```

### 3.2 Incorporation Grammar

```
compound := noun (· verb)+ [· noun]*
          | verb (· noun)+ [· verb]*

noun := identifier | type_name
verb := identifier | operator_morpheme
```

**Valid compounds:**
```sigil
file·read·lines        // noun·verb·noun
tcp·connect·send·recv  // noun·verb·verb·verb
json·parse·validate    // noun·verb·verb
data|τ·double·round    // pipe into verb·verb·verb
```

### 3.3 Morpheme Stacking

Multiple morphemes can stack on a single stem:

```sigil
// Stacked morphemes
data|φ>0|τ*2|σ|ρ+
//   │   │   │ └─ reduce with +
//   │   │   └─── sort (default)
//   │   └─────── transform: multiply by 2
//   └─────────── filter: greater than 0

// With evidentiality
result!? = compute()?   // Known if present, uncertain existence
value~! = remote()!     // Reported but asserted as known
```

---

## 4. Lexical Structure

### 4.1 Identifiers

```
identifier := (letter | _) (letter | digit | _ | ·)*
letter := [a-zA-Z] | unicode_letter
digit := [0-9]
```

Identifiers may contain middle dots for namespacing:
```sigil
std·io·File
http·client·Config
my·module·MyType
```

### 4.2 Numeric Literals

```sigil
42              // Decimal
0x2A            // Hexadecimal
0o52            // Octal
0b101010        // Binary
3.14159         // Float
6.022e23        // Scientific
1_000_000       // Separator allowed
```

### 4.3 String Literals

```sigil
"hello"                     // Basic string
"hello {name}"              // Interpolated
r"raw\nstring"              // Raw (no escapes)
b"byte string"              // Byte array
"""
  Multi-line
  string
"""                         // Multi-line

// Sigil strings (executable templates)
σ"SELECT * FROM {table}"    // SQL sigil string
ρ"/api/v1/{resource}/{id}"  // Route sigil string
```

### 4.4 Comments

```sigil
// Line comment

/* Block comment */

/// Doc comment (outer)
//! Doc comment (inner)

/*!
 * Module-level documentation
 */

// Rune comments (preserved for metaprogramming)
//@ rune: derive(Debug, Clone)
//@ rune: inline(always)
```

---

## 5. Operator Precedence

From highest to lowest precedence:

| Level | Operators | Associativity |
|-------|-----------|---------------|
| 1 | `·` (incorporation) | Left |
| 2 | `()` `[]` `{}` (grouping) | — |
| 3 | `!` `?` `~` `‽` (evidentiality) | Postfix |
| 4 | `τ` `φ` `σ` `ρ` (morphemes) | Left |
| 5 | `*` `/` `%` | Left |
| 6 | `+` `-` | Left |
| 7 | `<<` `>>` | Left |
| 8 | `&` | Left |
| 9 | `^` | Left |
| 10 | `\|` (pipe) | Left |
| 11 | `<` `>` `<=` `>=` | Left |
| 12 | `==` `!=` | Left |
| 13 | `&&` | Left |
| 14 | `\|\|` | Left |
| 15 | `→` `⇒` `↔` | Right |
| 16 | `=` `:=` | Right |

---

## 6. Keywords

### 6.1 Declaration Keywords

```
fn       struct    enum      trait     impl
type     const     static    let       mut
use      mod       pub       priv      crate
```

### 6.2 Control Flow Keywords

```
if       else      match     loop      while
for      in        break     continue  return
yield    await     async     try       throw
```

### 6.3 Type Keywords

```
Self     self      super     where     as
dyn      impl      ref       own       borrow
```

### 6.4 Evidentiality Keywords

```
known    uncertain reported  paradox   trust
assert   assume    verify    validate
```

### 6.5 Reserved for Future Use

```
macro    rune      glyph     seal      bind
invoke   conjure   summon    banish    ward
```

---

## 7. Token Examples

Complete tokenization of a sample expression:

```sigil
users|φ.active|σ·desc.created|τ{.name·upper}|α?
```

Tokens:
1. `users` — Identifier
2. `|` — Pipe operator
3. `φ` — Filter morpheme
4. `.` — Field access
5. `active` — Identifier
6. `|` — Pipe operator
7. `σ` — Sort morpheme
8. `·` — Incorporation
9. `desc` — Identifier (descending modifier)
10. `.` — Field access
11. `created` — Identifier
12. `|` — Pipe operator
13. `τ` — Transform morpheme
14. `{` — Block open
15. `.` — Field access
16. `name` — Identifier
17. `·` — Incorporation
18. `upper` — Identifier
19. `}` — Block close
20. `|` — Pipe operator
21. `α` — First morpheme
22. `?` — Uncertain evidentiality

---

## 8. Keyboard Input Methods

### 8.1 IDE Integration

Modern IDEs will provide:
- **Sigil palette** — Click-to-insert symbol panel
- **Autocomplete** — Type `\t` → suggests `τ`
- **Ligatures** — Font rendering `->` as `→`

### 8.2 Input Sequences

| Type | Result | Method |
|------|--------|--------|
| `\tau` | τ | Backslash completion |
| `\phi` | φ | Backslash completion |
| `Ctrl+Shift+T` | τ | Hotkey |
| `Ctrl+Shift+F` | φ | Hotkey |
| `:tau:` | τ | Emoji-style |

### 8.3 ASCII-Only Mode

For terminals or restricted environments:

```bash
sigil build --ascii-mode src/main.sg
```

This accepts ASCII equivalents and produces standard output.
