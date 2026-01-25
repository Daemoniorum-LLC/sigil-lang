# Sigil Language Cheatsheet

Quick reference for Sigil syntax, derived from working examples in the codebase.

## Core Syntax

### Variable Declaration

```sigil
// Assignment operator is ≔ (not let/const)
≔ name! = "value"           // Known (!)
≔ data~ = api.fetch(url)    // Reported (~)
≔ maybe? = find_item(id)    // Uncertain (?)
≔ trust‽ = unsafe_op()      // Paradox (‽)

// Type annotations
≔ count: !i32 = 42          // Type with evidentiality prefix
≔ items: Vec<Todo>! = vec![]
```

### Functions

```sigil
// Basic function (rite, not fn)
rite greet(name: str!) → str! {
    "Hello, {name}!"
}

// Exported function (☉ prefix)
☉ rite public_api() → i32! {
    42
}

// Async function
async rite fetch_data(url: str!) → Data~ {
    Http·get(url).await
}

// Method with self (use 'this')
☉ rite get_value(this) → i32! {
    this.x
}

// Mutable self (&vary)
☉ rite set_value(&vary this, val: i32!) {
    this.x = val
}
```

### Structs (sigil keyword)

```sigil
// Struct definition (☉ sigil, not struct)
☉ sigil Point {
    x: !i32,
    y: !i32
}

// With evidentiality in fields
☉ sigil ApiResponse {
    data: ~String,      // Reported - from external source
    cached: !bool       // Known - computed locally
}
```

### Impl Blocks (⊢ symbol)

```sigil
// Implementation block (⊢, not impl)
⊢ Point {
    // Constructor
    ☉ rite new(x: !i32, y: !i32) → This! {
        Point { x, y }
    }

    // Method
    ☉ rite distance(this, other: &Point) → f64! {
        // ...
    }
}
```

### Method Calls (middot ·)

```sigil
// Use · for method calls in many contexts
≔ result! = Http·get(url)
≔ parsed! = json·parse(data)
≔ html! = renderer·to_html()

// Regular . also works for field access and some methods
≔ len! = items.len()
≔ name! = user.name
```

### Modules

```sigil
// Module declaration
☉ mod components;
☉ mod pages;

// Public module
☉ mod prelude {
    ☉ use crate::core::App;
    // ...
}
```

### Imports

```sigil
// Use statements
use qliphoth::prelude::*
use crate::components::*

// Re-exports
☉ use crate::hooks::use_state;
```

## Evidentiality Markers

| Marker | Name | Meaning | Example |
|--------|------|---------|---------|
| `!` | Known | Verified, computed locally | `count!`, `!i32` |
| `?` | Uncertain | Might be absent | `user?`, `Option<T>?` |
| `~` | Reported | From external source | `data~`, `api_response~` |
| `◊` | Predicted | AI/ML output | `sentiment◊` |
| `‽` | Paradox | Explicit trust boundary | `ptr‽` |

### Evidentiality Propagation

```sigil
≔ local! = 100              // Known
≔ remote~ = api.get()       // Reported
≔ result~ = local * remote  // Known + Reported = Reported
```

## Morpheme Operators

```sigil
≔ result = users~
    |φ{.active}              // φ filter
    |τ{.normalize()}         // τ transform
    |σ·by{.created_at}       // σ sort
    |ρ{0, |acc, x| acc + x}  // ρ reduce
    |α                       // α first
    |ω                       // ω last

// Greek letters
// φ (phi)   - filter
// τ (tau)   - transform/map
// σ (sigma) - sort
// ρ (rho)   - reduce/fold
// Σ (Sigma) - sum
// Π (Pi)    - product
// α (alpha) - first
// ω (omega) - last
```

## Control Flow

```sigil
// If expression
≔ result! = if condition { a } else { b }

// Match
match value {
    Some(x) => process(x),
    None => default()
}

// For loop
for item in items {
    process(item)
}

// While
while condition {
    // ...
}
```

## Entry Points

```sigil
// Main entry point (☉ prefix)
☉ rite main() {
    // Program entry
}

// WASM exports
extern "wasm" {
    rite vdom_create_vnode(tag: str!) → i32!;
}
```

## String Operations

```sigil
// String concatenation with ++
≔ full! = first ++ " " ++ last

// String interpolation
≔ msg! = "Hello, {name}!"

// Multiline strings
≔ code! = r#"
    function example() {
        return 42;
    }
"#
```

## Common Patterns

### Optional Chaining

```sigil
≔ name? = user?.profile?.name
```

### Validation Pipeline

```sigil
≔ verified! = external_data~
    |validate!{ check_signature() }
```

### Error Handling

```sigil
≔ result? = operation()
    .map_err(|e| log_error(e))
    .ok()
```

## Symbols Reference

| Symbol | Name | Usage |
|--------|------|-------|
| `≔` | Assignment | Variable binding |
| `→` | Arrow | Return type |
| `☉` | Sun | Export/public marker |
| `⊢` | Turnstile | Impl block |
| `·` | Middot | Method call separator |
| `!` | Bang | Known evidentiality |
| `?` | Question | Uncertain evidentiality |
| `~` | Tilde | Reported evidentiality |
| `‽` | Interrobang | Paradox evidentiality |
| `◊` | Diamond | Predicted evidentiality |

## Traits (aspect keyword)

```sigil
// Trait definition (aspect, not trait)
☉ aspect Component: Sized {
    rite render(&this) → VNode;

    // Default implementation
    rite type_id() → u64! {
        std::any::type_name::<This>()·hash()
    }
}

// Generic trait with bounds
☉ aspect FunctionalComponent<P>: Fn(P) → VNode {
    rite type_id() → u64!;
}
```

## Mutable References

```sigil
// Mutable borrow (&vary, not &mut)
☉ rite set_value(&vary this, val: i32!) {
    this.x = val
}

// Moving mutable self
☉ rite attr(vary this, name: &str, value: AttrValue) → This! {
    this.element = this.element·attr(name, value)
    this
}

// Mutable variable in assignment
≔ vary count! = 0
count += 1
```

## Conditionals

```sigil
// If statement (⎇ symbol, or standard if)
⎇ idx >= this.states·len() {
    // ...
}

// Standard if/else also works
if condition {
    // ...
} else {
    // ...
}
```

## Boolean Literals

```sigil
// Boolean values
≔ active! = yea    // true
≔ hidden! = nay    // false
```

## Anti-Patterns (Common Mistakes)

```sigil
// WRONG: Rust-style syntax
fn greet() -> String { }     // Use: rite greet() → String!
let x = 5;                   // Use: ≔ x! = 5
impl Point { }               // Use: ⊢ Point { }
struct Data { }              // Use: ☉ sigil Data { }
pub fn api() { }             // Use: ☉ rite api() { }
trait Foo { }                // Use: ☉ aspect Foo { }
&mut self                    // Use: &vary this
true/false                   // Use: yea/nay

// WRONG: Missing evidentiality
≔ x = 5                      // Use: ≔ x! = 5 (mark as known)
```

---

*Last updated: 2026-01-21*
*Source: Derived from jormungandr tests, qliphoth/src, main.sigil*
