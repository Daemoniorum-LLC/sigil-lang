# Rust Codegen Backend Specification

**Version:** 0.1.0
**Status:** Draft
**Date:** 2026-02-04
**Author:** Sigil Compiler Team
**Spec Reference:** SDD Methodology

---

## 1. Purpose

This specification defines the translation rules for the Sigil → Rust codegen backend.
The backend transpiles Sigil source code to idiomatic, compilable Rust source code.

### 1.1 Goals

1. Generate valid Rust code that compiles with `rustc --edition 2021`
2. Produce readable, maintainable output (not obfuscated transpiler output)
3. Preserve semantic equivalence between Sigil and generated Rust
4. Support the full Sigil language except explicitly excluded features

### 1.2 Non-Goals

1. Runtime performance optimization (focus on correctness first)
2. Incremental compilation (full transpilation each time)
3. Source map generation (future enhancement)
4. Bi-directional translation (Rust → Sigil)

---

## 2. CLI Interface

```bash
sigil rust <input> [options]
```

### 2.1 Arguments

| Argument | Description |
|----------|-------------|
| `<input>` | Sigil source file (`.sg`) or directory |

### 2.2 Options

| Option | Default | Description |
|--------|---------|-------------|
| `-o, --output <path>` | stdout | Output file or directory |
| `--edition <year>` | 2021 | Rust edition (2018, 2021, 2024) |
| `--preserve-evidence` | false | Keep evidentiality wrapper types |
| `--emit-comments` | true | Include translation comments |
| `--emit-cargo` | false | Generate Cargo.toml from Sigil.toml |
| `--single-file` | true | Combine all modules into one file |
| `--format` | true | Run rustfmt on output |

### 2.3 Examples

```bash
# Transpile single file
sigil rust program.sg -o program.rs

# Transpile directory preserving module structure
sigil rust src/ -o rust-src/ --single-file=false

# Generate with Cargo.toml
sigil rust project/ -o rust-project/ --emit-cargo
```

---

## 3. Keyword Translation

### 3.1 Declaration Keywords

| Sigil | Rust | Notes |
|-------|------|-------|
| `rite` | `fn` | Function declaration |
| `sigil` | `struct` | Struct declaration |
| `aspect` | `trait` | Trait declaration |
| `scroll` | `mod` | Module declaration |
| `invoke` | `use` | Import statement |
| `type` | `type` | Type alias (unchanged) |
| `enum` | `enum` | Enum declaration (unchanged) |
| `const` / `◆` | `const` | Constant declaration |
| `static` | `static` | Static variable |

### 3.2 Control Flow Keywords

| Sigil | Rust | Notes |
|-------|------|-------|
| `⎇` / `if` | `if` | Conditional |
| `∅` / `else` | `else` | Else branch |
| `↦` / `match` | `match` | Pattern match |
| `∀` / `for` | `for` | For loop |
| `⊗` / `break` | `break` | Loop break |
| `↻` / `continue` | `continue` | Loop continue |
| `⏎` / `return` | `return` | Return statement |
| `∞` / `loop` / `forever` | `loop` | Infinite loop |
| `while` | `while` | While loop (unchanged) |

### 3.3 Modifier Keywords

| Sigil | Rust | Notes |
|-------|------|-------|
| `☉` / `pub` | `pub` | Public visibility |
| `vary` / `Δ` / `mut` | `mut` | Mutability |
| `async` / `⌛` (prefix) | `async` | Async function |
| `await` / `⌛` (postfix) | `.await` | Await expression |
| `unsafe` | `unsafe` | Unsafe block (unchanged) |
| `move` | `move` | Move closure (unchanged) |
| `dyn` | `dyn` | Trait object (unchanged) |

### 3.4 Self/This Keywords

| Sigil | Rust | Context |
|-------|------|---------|
| `this` | `self` | Instance reference (lowercase) |
| `This` | `Self` | Type reference (capitalized) |
| `&this` | `&self` | Shared reference |
| `&Δ this` | `&mut self` | Mutable reference |

### 3.5 Boolean Literals

| Sigil | Rust |
|-------|------|
| `yea` / `yay` / `true` | `true` |
| `nay` / `false` | `false` |

---

## 4. Operator Translation

### 4.1 Binding Operators

| Sigil | Rust | Example |
|-------|------|---------|
| `≔` | `let` | `≔ x = 5` → `let x = 5` |
| `≔ Δ` | `let mut` | `≔ Δ x = 5` → `let mut x = 5` |

### 4.2 Arrow Operators

| Sigil | Rust | Context |
|-------|------|---------|
| `→` | `->` | Return type |
| `=>` | `=>` | Match arm, closure body |
| `←` | N/A | Reserved (not translated) |

### 4.3 Path Separator

| Sigil | Rust | Context |
|-------|------|---------|
| `·` (middledot) | `::` | Static path: `Type·method` → `Type::method` |
| `·` (middledot) | `.` | Instance method: `obj·method()` → `obj.method()` |

**Disambiguation Rule:**
- If left-hand side is a type name (capitalized), emit `::`
- If left-hand side is a value/variable, emit `.`
- In turbofish context (`Type·<T>·method`), emit `::`

### 4.4 Reference Operators

| Sigil | Rust |
|-------|------|
| `&` | `&` |
| `&Δ` | `&mut` |
| `*` (deref) | `*` |

### 4.5 Arithmetic/Logic Operators

All arithmetic and logic operators pass through unchanged:
`+`, `-`, `*`, `/`, `%`, `==`, `!=`, `<`, `>`, `<=`, `>=`, `&&`, `||`, `!`, `&`, `|`, `^`, `<<`, `>>`

### 4.6 Assignment Operators

All assignment operators pass through unchanged:
`=`, `+=`, `-=`, `*=`, `/=`, `%=`, `&=`, `|=`, `^=`, `<<=`, `>>=`

### 4.7 Range Operators

| Sigil | Rust |
|-------|------|
| `..` | `..` |
| `..=` | `..=` |

### 4.8 Error Handling Operators

| Sigil | Rust | Notes |
|-------|------|-------|
| `?` (postfix) | `?` | Try operator (unchanged) |
| `!` (postfix) | `.unwrap()` | Unwrap for Option/Result |

### 4.9 Membership Operator

| Sigil | Rust | Context |
|-------|------|---------|
| `∈` | `in` | For loop: `∀ x ∈ items` → `for x in items` |

---

## 5. Type Translation

### 5.1 Primitive Types

All primitive types pass through unchanged:
`i8`, `i16`, `i32`, `i64`, `i128`, `isize`
`u8`, `u16`, `u32`, `u64`, `u128`, `usize`
`f32`, `f64`
`bool`, `char`
`()` (unit)

### 5.2 String Types

| Sigil | Rust |
|-------|------|
| `String` | `String` |
| `&str` | `&str` |

### 5.3 Collection Types

| Sigil | Rust |
|-------|------|
| `[T; N]` | `[T; N]` |
| `[T]` | `[T]` |
| `Vec<T>` | `Vec<T>` |
| `HashMap<K, V>` | `std::collections::HashMap<K, V>` |
| `HashSet<T>` | `std::collections::HashSet<T>` |

### 5.4 Smart Pointer Types

| Sigil | Rust |
|-------|------|
| `Box<T>` | `Box<T>` |
| `Rc<T>` | `std::rc::Rc<T>` |
| `Arc<T>` | `std::sync::Arc<T>` |
| `Cell<T>` | `std::cell::Cell<T>` |
| `RefCell<T>` | `std::cell::RefCell<T>` |

### 5.5 Option/Result

| Sigil | Rust |
|-------|------|
| `Option<T>` | `Option<T>` |
| `Result<T, E>` | `Result<T, E>` |
| `Some(v)` | `Some(v)` |
| `None` / `∅` | `None` |
| `Ok(v)` | `Ok(v)` |
| `Err(e)` | `Err(e)` |

### 5.6 Generic Parameters

| Sigil | Rust |
|-------|------|
| `<T>` | `<T>` |
| `<T: Trait>` | `<T: Trait>` |
| `<T: Trait + Other>` | `<T: Trait + Other>` |
| `<◆ N: usize>` | `<const N: usize>` |
| `<'a>` | `<'a>` |

### 5.7 Function Types

| Sigil | Rust |
|-------|------|
| `rite(A) → B` | `fn(A) -> B` |
| `rite(A, B) → C` | `fn(A, B) -> C` |
| `Fn(A) → B` | `Fn(A) -> B` |
| `FnMut(A) → B` | `FnMut(A) -> B` |
| `FnOnce(A) → B` | `FnOnce(A) -> B` |

### 5.8 Pointer Types

| Sigil | Rust |
|-------|------|
| `*◆ T` | `*const T` |
| `*vary T` | `*mut T` |

---

## 6. Evidentiality Handling

Sigil's epistemic type markers indicate data certainty levels.

### 6.1 Evidence Markers

| Sigil | Meaning |
|-------|---------|
| `T!` | Known (verified/computed) |
| `T?` | Uncertain (validated) |
| `T~` | Reported (external data) |
| `T‽` | Paradox (self-referential) |

### 6.2 Default Strategy: Strip

By default, evidence markers are stripped:

| Sigil | Rust |
|-------|------|
| `T!` | `T` |
| `T?` | `Option<T>` |
| `T~` | `T` |
| `T‽` | `T` |

### 6.3 Preserve Strategy (`--preserve-evidence`)

With the flag, generate wrapper types in the prelude:

```rust
// sigil_evidence.rs (generated)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Known<T>(pub T);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Uncertain<T>(pub Option<T>);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Reported<T>(pub T);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Paradox<T>(pub T);

impl<T> Known<T> {
    pub fn value(self) -> T { self.0 }
}

impl<T> Uncertain<T> {
    pub fn value(self) -> Option<T> { self.0 }
}

impl<T> Reported<T> {
    pub fn value(self) -> T { self.0 }
}
```

Then translate:

| Sigil | Rust |
|-------|------|
| `T!` | `Known<T>` |
| `T?` | `Uncertain<T>` |
| `T~` | `Reported<T>` |
| `T‽` | `Paradox<T>` |

---

## 7. Morpheme/Pipe Translation

Sigil's pipe operators translate to Rust iterator chains.

### 7.1 Basic Morphemes

| Sigil | Rust | Notes |
|-------|------|-------|
| `\|τ{f}` | `.map(\|x\| f(x))` | Transform/map |
| `\|φ{p}` | `.filter(\|x\| p(x))` | Filter |
| `\|ρ{f, init}` | `.fold(init, \|acc, x\| f(acc, x))` | Reduce/fold |
| `\|ρ+` / `\|Σ` | `.sum()` | Sum reduction |
| `\|ρ*` / `\|Π` | `.product()` | Product reduction |
| `\|σ` | `.sorted()` | Sort (requires itertools) |
| `\|α` | `.next()` | First element |
| `\|ω` | `.last()` | Last element |
| `\|∀{p}` | `.all(\|x\| p(x))` | All match predicate |
| `\|∃{p}` | `.any(\|x\| p(x))` | Any match predicate |
| `\|→[]` | `.collect::<Vec<_>>()` | Collect to Vec |
| `\|→{}` | `.collect::<HashSet<_>>()` | Collect to HashSet |
| `\|#` / `\|len` | `.count()` | Count elements |

### 7.2 Parallel Morphemes

| Sigil | Rust | Notes |
|-------|------|-------|
| `∥τ{f}` | `.par_iter().map(\|x\| f(x))` | Parallel map (rayon) |
| `∥φ{p}` | `.par_iter().filter(\|x\| p(x))` | Parallel filter |

Parallel morphemes require `use rayon::prelude::*` in generated code.

### 7.3 Iterator Initialization

When a pipe chain starts from a collection:
- `Vec<T>` → `.iter()` (or `.into_iter()` if consuming)
- `&[T]` → `.iter()`
- Ranges → used directly

### 7.4 Example Translation

```sigil
numbers |τ{x => x * 2} |φ{x => x > 10} |σ |→[]
```

Becomes:

```rust
numbers.iter()
    .map(|x| x * 2)
    .filter(|x| *x > 10)
    .sorted()
    .collect::<Vec<_>>()
```

---

## 8. Expression Translation

### 8.1 Closures

| Sigil | Rust |
|-------|------|
| `{x => x + 1}` | `\|x\| x + 1` |
| `{(a, b) => a + b}` | `\|(a, b)\| a + b` |
| `move {x => x + 1}` | `move \|x\| x + 1` |

### 8.2 Match Expressions

```sigil
↦ value {
    Pattern1 => expr1,
    Pattern2(x) => expr2,
    _ => default,
}
```

Becomes:

```rust
match value {
    Pattern1 => expr1,
    Pattern2(x) => expr2,
    _ => default,
}
```

### 8.3 If-Let

```sigil
⎇ Some(x) = maybe {
    use(x)
} ∅ {
    fallback
}
```

Becomes:

```rust
if let Some(x) = maybe {
    use(x)
} else {
    fallback
}
```

### 8.4 While-Let

```sigil
while Some(x) = iter·next() {
    process(x)
}
```

Becomes:

```rust
while let Some(x) = iter.next() {
    process(x)
}
```

### 8.5 For Loops

```sigil
∀ item ∈ collection {
    process(item)
}
```

Becomes:

```rust
for item in collection {
    process(item)
}
```

### 8.6 Struct Literals

```sigil
Point { x: 10, y: 20 }
Point { x, y }  // shorthand
Point { x, ..default }  // spread
```

Becomes:

```rust
Point { x: 10, y: 20 }
Point { x, y }
Point { x, ..default }
```

### 8.7 Tuple Expressions

Tuple syntax passes through unchanged.

### 8.8 Array/Vec Literals

```sigil
[1, 2, 3]
[0; 100]
```

Becomes:

```rust
[1, 2, 3]
[0; 100]
```

### 8.9 Async/Await

```sigil
async {
    result⌛
}
```

Becomes:

```rust
async {
    result.await
}
```

---

## 9. Item Translation

### 9.1 Functions

```sigil
☉ rite foo<T: Clone>(x: T, y: i32) → T {
    body
}
```

Becomes:

```rust
pub fn foo<T: Clone>(x: T, y: i32) -> T {
    body
}
```

### 9.2 Structs

```sigil
#[derive(Debug, Clone)]
☉ sigil Foo<T> {
    field1: T,
    field2: String,
}
```

Becomes:

```rust
#[derive(Debug, Clone)]
pub struct Foo<T> {
    field1: T,
    field2: String,
}
```

### 9.3 Enums

```sigil
☉ enum State {
    Active,
    Inactive,
    Error(String),
}
```

Becomes:

```rust
pub enum State {
    Active,
    Inactive,
    Error(String),
}
```

### 9.4 Traits

```sigil
☉ aspect Drawable {
    rite draw(&this);
    rite bounds(&this) → Rect;
}
```

Becomes:

```rust
pub trait Drawable {
    fn draw(&self);
    fn bounds(&self) -> Rect;
}
```

### 9.5 Impl Blocks

```sigil
⊢ Foo<T> {
    rite new(value: T) → Self {
        Self { field1: value, field2: String·new() }
    }
}
```

Becomes:

```rust
impl<T> Foo<T> {
    fn new(value: T) -> Self {
        Self { field1: value, field2: String::new() }
    }
}
```

### 9.6 Trait Implementations

```sigil
⊢ Drawable ∀ Circle {
    rite draw(&this) { ... }
}
```

Becomes:

```rust
impl Drawable for Circle {
    fn draw(&self) { ... }
}
```

### 9.7 Type Aliases

```sigil
type Callback = rite(i32) → i32;
```

Becomes:

```rust
type Callback = fn(i32) -> i32;
```

### 9.8 Constants

```sigil
◆ MAX_SIZE: usize = 1024;
```

Becomes:

```rust
const MAX_SIZE: usize = 1024;
```

---

## 10. Extern Blocks

Extern blocks pass through with minimal transformation.

```sigil
extern "C" {
    rite malloc(size: usize) → *vary u8;
    rite free(ptr: *vary u8);
    static errno: i32;
}
```

Becomes:

```rust
extern "C" {
    fn malloc(size: usize) -> *mut u8;
    fn free(ptr: *mut u8);
    static errno: i32;
}
```

---

## 11. Attributes

### 11.1 Standard Attributes

Most attributes pass through unchanged:
`#[derive(...)]`, `#[cfg(...)]`, `#[allow(...)]`, `#[warn(...)]`, `#[inline]`, etc.

### 11.2 Export Attribute

```sigil
#[export]
rite foo() { }
```

Becomes:

```rust
#[no_mangle]
pub extern "C" fn foo() { }
```

---

## 12. Module System

### 12.1 Single-File Mode (Default)

All modules are inlined into a single output file using `mod name { }` blocks.

### 12.2 Multi-File Mode (`--single-file=false`)

Preserves Sigil module structure as separate Rust files:
- `src/lib.sg` → `src/lib.rs`
- `src/utils.sg` → `src/utils.rs`
- `src/models/foo.sg` → `src/models/foo.rs`

### 12.3 Import Translation

```sigil
invoke std·collections·HashMap;
invoke crate·utils·helper;
invoke super·parent_fn;
```

Becomes:

```rust
use std::collections::HashMap;
use crate::utils::helper;
use super::parent_fn;
```

---

## 13. Error Handling

### 13.1 Untranslatable Constructs

When a construct cannot be translated, the backend:
1. Emits a `todo!("untranslatable: ...")` placeholder
2. Includes a comment explaining the issue
3. Reports a warning to stderr

### 13.2 Known Untranslatable Features

| Feature | Reason | Fallback |
|---------|--------|----------|
| Actor system (`spawn`, `send`) | No direct Rust equivalent | `todo!()` |
| Protocol expressions | Requires runtime | `todo!()` |
| Legion plurality | Holographic-specific | `todo!()` |

---

## 14. Generated Prelude

When needed, the backend generates a prelude module:

```rust
// sigil_prelude.rs (generated)

// Evidence types (if --preserve-evidence)
pub struct Known<T>(pub T);
pub struct Uncertain<T>(pub Option<T>);
pub struct Reported<T>(pub T);

// Common re-exports
pub use std::collections::{HashMap, HashSet};
pub use std::rc::Rc;
pub use std::cell::{Cell, RefCell};
```

---

## 15. Cargo.toml Generation

With `--emit-cargo`, generate from `Sigil.toml`:

**Input (Sigil.toml):**
```toml
[package]
name = "my-project"
version = "0.1.0"

[dependencies]
nihil-core = { path = "../nihil-core" }
```

**Output (Cargo.toml):**
```toml
[package]
name = "my-project"
version = "0.1.0"
edition = "2021"

[dependencies]
nihil-core = { path = "../nihil-core" }

# Generated dependencies for Sigil features
itertools = "0.12"  # if morphemes use |σ
rayon = "1"         # if parallel morphemes used
```

---

## 16. Acceptance Criteria

The Rust codegen backend is complete when:

1. **Compilation**: All generated Rust compiles with `rustc --edition 2021`
2. **Semantics**: Generated code has equivalent behavior to interpreter
3. **Coverage**: All test files in `jormungandr/tests/rust_codegen/` pass
4. **Nihil**: Nihil crates successfully transpile to buildable Rust
5. **Performance**: Transpilation completes in < 1s for typical files
6. **Readability**: Generated code is formatted with rustfmt

---

## 17. Revision History

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 0.1.0 | 2026-02-04 | Sigil Compiler Team | Initial specification |
