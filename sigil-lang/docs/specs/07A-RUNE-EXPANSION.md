# Sigil Rune Expansion Specification

> *"A rune's power lies not in its inscription, but in the ritual of its unfolding."*

## 1. Overview

This specification defines the **algorithmic semantics** of rune (macro) expansion in Sigil. It complements 07-METAPROGRAMMING.md which documents rune syntax and patterns. This document specifies:

1. **When** expansion occurs in the compilation pipeline
2. **How** patterns are matched against input
3. **How** matched fragments are substituted into output
4. **How** hygiene prevents identifier collisions
5. **How** errors are detected and reported
6. **How** recursion is controlled

---

## 2. Compilation Pipeline Position

Rune expansion occurs **after parsing, before type checking**:

```
Source Code
    │
    ▼
┌──────────────────┐
│      LEXER       │  → Token Stream
└──────────────────┘
    │
    ▼
┌──────────────────┐
│      PARSER      │  → AST (with unexpanded rune invocations)
└──────────────────┘
    │
    ▼
┌──────────────────────────────────────────┐
│         RUNE EXPANSION (this spec)        │
│                                           │
│  1. Collect rune definitions              │
│  2. Find rune invocations in AST          │
│  3. For each invocation:                  │
│     a. Match against rune patterns        │
│     b. Capture fragments                  │
│     c. Substitute into expansion body     │
│     d. Apply hygiene transformations      │
│     e. Parse expansion result             │
│     f. Replace invocation with result     │
│  4. Repeat until no invocations remain    │
│                                           │
└──────────────────────────────────────────┘
    │
    ▼
┌──────────────────┐
│   TYPE CHECKER   │  → Typed AST
└──────────────────┘
    │
    ▼
    ... (lowering, codegen)
```

### 2.1 Key Properties

- **Eager expansion**: Runes expand immediately when encountered
- **Outside-in**: Outer rune invocations expand before inner ones
- **Fixed-point**: Expansion continues until no invocations remain
- **Deterministic**: Same input always produces same output

---

## 3. Rune Definition Structure

A rune definition consists of one or more **arms**, each with a **pattern** and an **expansion body**:

```
rune name! {
    (pattern₁) => { body₁ }
    (pattern₂) => { body₂ }
    ...
}
```

**Internal Representation:**

```sigil
struct RuneDefinition {
    name: Ident,              // e.g., "vec"
    arms: [RuneArm],
    span: Span,
}

struct RuneArm {
    pattern: RunePattern,
    body: TokenStream,        // Unexpanded tokens
    span: Span,
}
```

---

## 4. Pattern Matching Algorithm

### 4.1 Pattern Grammar

```ebnf
pattern       = token_tree* ;
token_tree    = token | group | repetition | fragment ;
group         = "(" pattern ")" | "[" pattern "]" | "{" pattern "}" ;
repetition    = "$(" pattern ")" sep? rep_op ;
sep           = "," | ";" | any_token ;
rep_op        = "*" | "+" | "?" ;
fragment      = "$" ident ":" frag_spec ;
frag_spec     = "expr" | "ty" | "ident" | "path" | "pat" | "stmt"
              | "block" | "item" | "meta" | "tt" | "literal" | "vis" ;
```

### 4.2 Fragment Specifiers

Each specifier defines what tokens constitute a valid match:

| Specifier | Matches | Parsing Rule |
|-----------|---------|--------------|
| `expr` | Expression | Parse as expression, stop at `,` `;` `=>` |
| `ty` | Type | Parse as type, stop at `,` `;` `>` `=` |
| `ident` | Identifier | Single identifier token |
| `path` | Path | `ident (:: ident)*` |
| `pat` | Pattern | Parse as pattern, stop at `=>` `=` `if` `,` |
| `stmt` | Statement | Parse as statement |
| `block` | Block | `{ ... }` delimited |
| `item` | Item | fn, struct, enum, impl, etc. |
| `meta` | Attribute content | Inside `#[...]` |
| `tt` | Token tree | Any single token or balanced group |
| `literal` | Literal | String, number, char, bool literal |
| `vis` | Visibility | `pub`, `pub(crate)`, etc., or empty |

### 4.3 Matching Algorithm

```
Algorithm: match_pattern(pattern, tokens) → Option<Bindings>

Input:  pattern - the rune pattern
        tokens  - input token stream
Output: Bindings map from fragment names to captured tokens, or None

1. Initialize bindings = {}
2. Initialize token cursor at start of tokens
3. For each element in pattern:

   a. If element is LITERAL TOKEN:
      - If cursor token matches, advance cursor
      - Else return None

   b. If element is FRAGMENT ($name:spec):
      - Call parse_fragment(spec, cursor) → (tokens, new_cursor)
      - If parse fails, return None
      - bindings[name] = tokens
      - cursor = new_cursor

   c. If element is GROUP (delimited):
      - If cursor not at matching delimiter, return None
      - Recursively match_pattern(group_contents, inner_tokens)
      - Advance cursor past closing delimiter

   d. If element is REPETITION ($(...) sep rep_op):
      - Call match_repetition(inner_pattern, sep, rep_op, cursor)
      - Accumulate bindings as lists
      - Advance cursor to end of matched repetitions

4. If cursor not at end of tokens:
   - Return None (unconsumed tokens)

5. Return bindings
```

### 4.4 Repetition Matching

```
Algorithm: match_repetition(pattern, sep, op, cursor) → (Bindings[], new_cursor)

1. Initialize matches = []
2. Loop:
   a. Try match_pattern(pattern, cursor)
   b. If match succeeds:
      - Append bindings to matches
      - If sep specified and cursor at sep, consume sep
      - Else if sep specified, break loop
   c. If match fails:
      - Break loop

3. Validate against rep_op:
   - If op is '+' and matches.len() == 0, fail
   - If op is '?' and matches.len() > 1, fail

4. Return (matches, cursor)
```

### 4.5 First-Match Semantics

When multiple arms could match, **the first matching arm wins**:

```sigil
rune example! {
    ($x:expr) => { "single" }      // Arm 1
    ($x:expr, $y:expr) => { "pair" }  // Arm 2
}

example!(1)      // Matches arm 1
example!(1, 2)   // Matches arm 2 (arm 1 fails: unconsumed tokens)
```

---

## 5. Expansion Algorithm

### 5.1 Substitution

```
Algorithm: expand(arm, bindings) → TokenStream

Input:  arm      - the matched rune arm
        bindings - captured fragments
Output: Expanded token stream

1. Initialize output = []
2. For each element in arm.body:

   a. If element is LITERAL TOKEN:
      - Append token to output

   b. If element is FRAGMENT REFERENCE ($name):
      - Look up bindings[name]
      - If not found, ERROR: undefined fragment
      - Append captured tokens to output

   c. If element is GROUP:
      - Recursively expand group contents
      - Wrap in matching delimiters
      - Append to output

   d. If element is REPETITION ($(...) sep rep_op):
      - For each index i in 0..repetition_count:
        - Expand inner pattern with bindings[*][i]
        - If not last iteration and sep specified, append sep
      - Append all expansions to output

3. Return output
```

### 5.2 Repetition Expansion

Repetition in the expansion body iterates over captured repetition bindings:

```sigil
rune make_tuple! {
    ($($x:expr),*) => { ($($x),*) }
}

make_tuple!(1, 2, 3)
// Pattern captures: x = [1, 2, 3]
// Expansion iterates: (1, 2, 3)
```

**Nested repetitions** must align:

```sigil
rune nested! {
    ($($x:expr => $y:expr),*) => {
        $( process($x, $y); )*
    }
}

nested!(a => 1, b => 2)
// x = [a, b], y = [1, 2]
// Expands to: process(a, 1); process(b, 2);
```

### 5.3 Repetition Depth Validation

All fragment references within a repetition must have matching depths:

```sigil
// VALID: $x and $y both captured in same repetition
rune valid! {
    ($($x:expr, $y:expr),*) => { $($x + $y),* }
}

// ERROR: $single not in repetition context
rune invalid! {
    ($single:expr, $($multi:expr),*) => {
        $($single + $multi),*  // ERROR: $single has wrong repetition depth
    }
}
```

---

## 6. Hygiene

Hygiene prevents identifiers introduced by runes from colliding with user code.

### 6.1 Syntax Contexts

Each identifier carries a **syntax context** (SynCtx) that tracks its origin:

```sigil
struct SynCtx {
    expansion_id: u32,    // Which rune expansion
    transparency: Transparency,
}

enum Transparency {
    Transparent,  // Shares caller's scope
    Opaque,       // Has its own scope
    SemiOpaque,   // Mixed (default)
}
```

### 6.2 Hygiene Rules

1. **Rune-introduced identifiers** get a fresh SynCtx from the rune definition site
2. **User-provided fragments** retain their original SynCtx
3. **Identifiers with different SynCtx cannot shadow each other**

```sigil
rune safe_swap! {
    ($a:expr, $b:expr) => {
        {
            let temp = $a;    // 'temp' has rune's SynCtx
            $a = $b;
            $b = temp;
        }
    }
}

let temp = 100;           // User's 'temp'
let mut x = 1;
let mut y = 2;
safe_swap!(x, y);         // Rune's 'temp' doesn't shadow user's 'temp'
println(temp);            // Still 100
```

### 6.3 Hygiene Escape

The `#` prefix escapes hygiene, exposing the identifier to the caller's scope:

```sigil
rune define_var! {
    ($name:ident = $val:expr) => {
        let #$name = $val;    // # exposes to caller
    }
}

define_var!(x = 42);
println(x);               // Works: x is visible
```

### 6.4 Implementation

```
Algorithm: apply_hygiene(tokens, syn_ctx) → tokens'

1. For each token in tokens:
   a. If token is IDENTIFIER and not fragment-derived:
      - Mark with syn_ctx
   b. If token is FRAGMENT REFERENCE:
      - Keep original syn_ctx from capture site
   c. If token is #IDENTIFIER (escaped):
      - Mark as transparent (caller's syn_ctx)

2. Return modified tokens
```

---

## 7. Pipe-Invoked Rune Expansion

When a rune is invoked in pipe position, special handling applies.

### 7.1 Desugaring

```sigil
// Pipe-invoked form
value|rune!{ args }

// Desugars to
rune!(__pipe: value, { args })
```

### 7.2 `__pipe` Binding

Within the rune body, `__pipe` is implicitly bound to the piped value:

```sigil
rune validate! {
    ({ $($field:ident : $check:expr),* }) => {
        {
            let __input = __pipe;  // __pipe available automatically
            $(
                if !($check)(&__input.$field) {
                    return Err(ValidationError::field(stringify!($field)))
                }
            )*
            Ok(__input)
        }
    }
}

// Usage
request|validate!{ name: non_empty, age: positive }

// Expands to (approximately)
{
    let __input = request;
    if !(non_empty)(&__input.name) {
        return Err(ValidationError::field("name"))
    }
    if !(positive)(&__input.age) {
        return Err(ValidationError::field("age"))
    }
    Ok(__input)
}
```

### 7.3 Pipe Hygiene

The `__pipe` identifier:
- Has **semi-opaque** transparency
- Is accessible within the rune body
- Does not leak to caller's scope

---

## 8. Recursion Control

### 8.1 Recursion Depth Limit

Rune expansion has a configurable depth limit (default: 128):

```sigil
// Each recursive call increments depth
rune count! {
    () => { 0 }
    ($head:tt $($tail:tt)*) => { 1 + count!($($tail)*) }
}

count!(a b c d ... 200 items ...)
// ERROR: recursion limit exceeded (if > 128 deep)
```

### 8.2 Infinite Recursion Detection

The expander detects trivially infinite patterns:

```sigil
// ERROR: unconditional self-reference
rune infinite! {
    () => { infinite!() }
}
```

### 8.3 Configuration

```sigil
// Per-rune limit
//@ rune: recursion_limit(256)
rune deep_rune! { ... }

// Crate-wide default
//@ rune: crate_recursion_limit(64)
```

---

## 9. Error Handling

### 9.1 Error Types

| Error | Description | Example |
|-------|-------------|---------|
| `E7001` | No matching arm | `vec!("not" "a" "list")` |
| `E7002` | Ambiguous match | (shouldn't occur with first-match) |
| `E7003` | Undefined fragment | `$x` where `x` not captured |
| `E7004` | Repetition depth mismatch | See §5.3 |
| `E7005` | Recursion limit exceeded | Deeply nested rune calls |
| `E7006` | Parse error in expansion | Expanded tokens don't parse |
| `E7007` | Invalid fragment for specifier | `$x:expr` but can't parse expr |

### 9.2 Error Location

Errors point to the **invocation site**, with notes showing the rune definition:

```
error[E7001]: no matching arm for rune invocation
  --> src/main.sg:10:5
   |
10 |     vec!("a" "b" "c")
   |     ^^^^^^^^^^^^^^^^^
   |
note: rune `vec!` defined here
  --> std/runes.sg:15:1
   |
15 | rune vec! {
   | ^^^^^^^^^^
   |
   = help: expected comma-separated expressions: vec![a, b, c]
```

### 9.3 Expansion Tracing

Debug mode shows expansion steps:

```
//@ rune: trace_expansion
let v = vec![1, 2, 3];

// Compiler output:
// [trace] vec![1, 2, 3]
// [trace] → matched arm: ($($elem:expr),+ $(,)?)
// [trace] → bindings: elem = [1, 2, 3]
// [trace] → expanded to:
// [trace]   {
// [trace]       let mut v = Vec::with_capacity(3);
// [trace]       v.push(1);
// [trace]       v.push(2);
// [trace]       v.push(3);
// [trace]       v
// [trace]   }
```

---

## 10. Procedural Runes (Invocations)

Procedural runes (invocations) use arbitrary code to perform transformations.

### 10.1 Invocation API

```sigil
use sigil::invoke::{TokenStream, TokenTree, Span}

//@ rune: invoke
pub fn my_macro(input: TokenStream) -> TokenStream {
    // Parse input
    let parsed = parse_input(input)?;

    // Transform
    let output = transform(parsed);

    // Generate output tokens
    quote! { #output }
}
```

### 10.2 Derive Invocations

```sigil
//@ rune: derive
pub fn MyDerive(input: TokenStream) -> TokenStream {
    let ast = parse_derive_input(input)?;

    let name = ast.ident;
    let fields = ast.fields;

    quote! {
        impl MyTrait for #name {
            fn method(&self) {
                #(self.#fields.do_thing();)*
            }
        }
    }
}
```

### 10.3 Attribute Invocations (Inscriptions)

```sigil
//@ rune: inscription
pub fn my_attribute(attr: TokenStream, item: TokenStream) -> TokenStream {
    let config = parse_attr(attr)?;
    let item = parse_item(item)?;

    // Modify item based on attribute
    transform_item(item, config)
}
```

---

## 11. Optimization

### 11.1 Common Pattern Optimization

The expander recognizes common patterns and optimizes:

```sigil
// vec![] → Vec::new()  (not full expansion)
// vec![x; n] → vec_repeat(x, n)  (specialized)
```

### 11.2 Memoization

Identical invocations in the same syntax context can be memoized:

```sigil
// Only expands once, result reused
let a = vec![1, 2, 3];
let b = vec![1, 2, 3];  // Memoized
```

### 11.3 Incremental Expansion

In IDE/language server mode, only re-expand runes affected by edits.

---

## 12. Interaction with Other Features

### 12.1 With Type Inference

Rune expansion happens **before** type inference. Expanded code participates in normal type inference.

### 12.2 With Const Evaluation

Runes can produce `const` items. Const evaluation happens **after** expansion.

### 12.3 With Borrow Checking

Expanded code undergoes normal borrow checking. Rune authors must ensure generated code is borrow-safe.

### 12.4 With Modules

Runes are resolved using normal module visibility. A rune must be in scope (via `use`) to be invoked.

---

## 13. Formal Semantics

### 13.1 Expansion Relation

The expansion relation `⊢ e ⟹ e'` denotes that expression `e` expands to `e'`:

```
LITERAL
────────────────
⊢ lit ⟹ lit

VAR
────────────────
⊢ x ⟹ x

RUNE-INVOKE (where rune R has arm (p => b) matching input I with bindings σ)
────────────────────────────────────────────────────────────────────────────
⊢ R!(I) ⟹ expand(b, σ)

RECURSIVE (expansion may produce new invocations)
─────────────────────────────────────────────────
⊢ e ⟹ e'    ⊢ e' ⟹ e''
────────────────────────
⊢ e ⟹* e''
```

### 13.2 Hygiene Preservation

For any rune expansion:

```
If ⊢ R!(I) ⟹ e' and x is introduced by R
Then for all user-bound y: rename(x) ≠ rename(y)
```

This guarantees no identifier capture.

---

## 14. Implementation Checklist

A compliant implementation must:

- [ ] Implement pattern matching with all fragment specifiers
- [ ] Implement substitution with proper repetition handling
- [ ] Implement hygiene with syntax contexts
- [ ] Implement pipe-invoked rune desugaring
- [ ] Enforce recursion depth limits
- [ ] Produce useful error messages with source locations
- [ ] Support procedural runes via the invocation API
- [ ] Handle edge cases (empty repetitions, nested groups, etc.)

---

## 15. Version History

| Version | Date | Changes |
|---------|------|---------|
| 0.1.0 | 2026-01 | Initial specification |

---

*This specification defines the algorithmic foundation for Sigil's metaprogramming system.*
