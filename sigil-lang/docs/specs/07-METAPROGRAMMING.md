# Sigil Metaprogramming Specification

> *"To name a thing is to have power over it. To write a Rune is to reshape reality."*

## 1. Philosophy: Code as Incantation

Metaprogramming in Sigil embraces the language's occult aesthetic. Code that writes code is not mere "macros" — it is **magic**: symbolic manipulation that transforms intent into execution.

| Concept | Traditional Term | Sigil Term | Purpose |
|---------|------------------|------------|---------|
| Declarative macros | `macro_rules!` | **Rune** | Pattern-based code generation |
| Procedural macros | `proc_macro` | **Invocation** | Arbitrary code transformation |
| Type-level programming | Generics/traits | **Seal** | Compile-time type computation |
| Compile-time checks | Static analysis | **Ward** | Safety and constraint enforcement |
| Const evaluation | `const fn` | **Glyph** | Compile-time value computation |
| Attribute macros | `#[derive]` | **Inscription** | Declarative code decoration |

---

## 2. Runes (Declarative Macros)

Runes are pattern-matching transformations — symbolic recipes that expand into code.

### 2.1 Basic Rune Definition

```sigil
// Define a rune with pattern matching
rune vec! {
    // Empty case
    () => { Vec·new() }

    // Single element
    ($elem:expr) => {
        {
            let mut v = Vec·new()
            v|push($elem)
            v
        }
    }

    // Multiple elements
    ($($elem:expr),+ $(,)?) => {
        {
            let mut v = Vec·with_capacity(count!($($elem),+))
            $(v|push($elem);)+
            v
        }
    }
}

// Usage
let numbers = vec![1, 2, 3, 4, 5]
```

### 2.2 Rune Fragment Types

Fragment types (also called "fragment specifiers") define what kind of syntax a rune parameter can capture. Each fragment type has specific parsing rules and valid contexts.

| Fragment | Syntax | Matches | Implemented |
|----------|--------|---------|-------------|
| `expr` | `$x:expr` | Any expression | ✅ Yes |
| `ty` | `$T:ty` | Any type | ✅ Yes |
| `ident` | `$name:ident` | Identifier | ✅ Yes |
| `path` | `$p:path` | Module path | ❌ No |
| `pat` | `$p:pat` | Pattern | ✅ Yes |
| `stmt` | `$s:stmt` | Statement | ✅ Yes |
| `block` | `$b:block` | Block `{ ... }` | ✅ Yes |
| `item` | `$i:item` | Top-level item | ❌ No |
| `meta` | `$m:meta` | Attribute content | ❌ No |
| `tt` | `$t:tt` | Token tree | ✅ Yes |
| `literal` | `$l:literal` | Literal value | ✅ Yes |
| `lifetime` | `$a:lifetime` | Lifetime `'a` | ❌ No |
| `vis` | `$v:vis` | Visibility | ❌ No |

#### 2.2.1 Fragment Type Definitions

**`expr` — Expression**
Matches any valid Sigil expression. This is the most permissive fragment type.

```sigil
// Valid: literals, identifiers, function calls, operators, blocks
rune double! { ($x:expr) => { $x + $x } }
double!(42)              // 42 + 42
double!(foo)             // foo + foo
double!(bar(1, 2))       // bar(1, 2) + bar(1, 2)
double!({ let y = 1; y })  // block expression
```

**Parsing:** Consumes a complete expression according to Sigil's expression grammar. Respects operator precedence and grouping.

**`ty` — Type**
Matches any type annotation.

```sigil
rune new_vec! { ($T:ty) => { Vec<$T>·new() } }
new_vec!(i32)            // Vec<i32>·new()
new_vec!(String)         // Vec<String>·new()
new_vec!(Option<i64>)    // Vec<Option<i64>>·new()
```

**Parsing:** Consumes a type expression including generic parameters, arrays, references, and function types.

**`ident` — Identifier**
Matches a single identifier token (variable name, type name, etc.).

```sigil
rune getter! { ($field:ident) => { fn $field(&self) -> i32 { self.$field } } }
getter!(x)               // fn x(&self) -> i32 { self.x }
getter!(count)           // fn count(&self) -> i32 { self.count }
```

**Parsing:** Consumes exactly one identifier token. Does not match keywords, literals, or punctuation.

**`path` — Module Path** *(Not yet implemented)*
Matches a potentially-qualified path like `std·collections·HashMap`.

```sigil
rune use_type! { ($p:path) => { let x: $p = $p·new() } }
use_type!(Vec)                       // simple path
use_type!(std·io·File)               // qualified path
use_type!(crate·models·User)         // crate-relative path
```

**Parsing:** Consumes a sequence of identifiers separated by `·` (middledot).

**`pat` — Pattern**
Matches a pattern as used in `match` arms, `let` bindings, and function parameters.

```sigil
rune match_arm! { ($p:pat => $body:expr) => { $p => { $body } } }
match_arm!(Some(x) => x + 1)         // Some(x) => { x + 1 }
match_arm!((a, b) => a * b)          // (a, b) => { a * b }
```

**Parsing:** Consumes a complete pattern including destructuring, bindings, and guards.

**`stmt` — Statement**
Matches a single statement (with or without trailing semicolon).

```sigil
rune debug_stmt! { ($s:stmt) => { println("executing..."); $s } }
debug_stmt!(let x = 42)              // println("executing..."); let x = 42
debug_stmt!(foo())                   // println("executing..."); foo()
```

**Parsing:** Consumes a complete statement. If the statement naturally has a semicolon, it is included.

**`block` — Block**
Matches a curly-brace-delimited block `{ ... }`.

```sigil
rune timed! { ($b:block) => { let t = now(); $b; elapsed(t) } }
timed!({ expensive_computation(); cleanup() })
```

**Parsing:** Consumes an opening `{`, all content, and closing `}`. The braces are part of the captured fragment.

**`item` — Item** *(Not yet implemented)*
Matches a top-level item: function, struct, enum, trait, impl, const, static, or module.

```sigil
rune trace_item! { ($i:item) => { println("defined item"); $i } }
trace_item!(fn foo() { })            // function
trace_item!(struct Bar { x: i32 })   // struct
```

**Parsing:** Consumes a complete item definition.

**`meta` — Attribute Content** *(Not yet implemented)*
Matches the content inside an attribute annotation `//@ rune: ...`.

```sigil
rune conditional! { ($m:meta) => { //@ rune: cfg($m) } }
conditional!(target_os = "linux")
```

**Parsing:** Consumes attribute-style key-value or identifier content.

**`tt` — Token Tree**
Matches a single token or balanced group of tokens. This is the most flexible but least semantic fragment type.

```sigil
rune pass_through! { ($($t:tt)*) => { $($t)* } }
pass_through!(foo bar { baz })       // foo bar { baz }
```

**Parsing:** A `tt` is either:
1. A single non-delimiter token (identifier, literal, operator)
2. A balanced group: `(...)`, `[...]`, or `{...}` including all contents

**`literal` — Literal Value**
Matches a literal: integer, float, string, character, or boolean.

```sigil
rune const_array! { ($($l:literal),*) => { [$($l),*] } }
const_array!(1, 2, 3)                // [1, 2, 3]
const_array!("a", "b")               // ["a", "b"]
```

**Parsing:** Consumes exactly one literal token.

**`lifetime` — Lifetime Annotation** *(Not yet implemented)*
Matches a lifetime like `'a` or `'static`.

```sigil
rune ref_with_lt! { ($a:lifetime, $T:ty) => { &$a $T } }
ref_with_lt!('a, str)                // &'a str
ref_with_lt!('static, i32)           // &'static i32
```

**Parsing:** Consumes a `'` followed by an identifier or `static`.

**`vis` — Visibility** *(Not yet implemented)*
Matches a visibility qualifier: empty, `pub`, `pub(crate)`, `pub(super)`, or `pub(in path)`.

```sigil
rune make_field! { ($v:vis $name:ident: $T:ty) => { $v $name: $T } }
make_field!(pub x: i32)              // pub x: i32
make_field!(pub(crate) y: String)    // pub(crate) y: String
make_field!(z: bool)                 // z: bool (private)
```

**Parsing:** Optionally consumes `pub` and any restriction parentheses. May match nothing (empty = private).

#### 2.2.2 Fragment Follow Restrictions

Certain fragment types have restrictions on what tokens may follow them, to ensure unambiguous parsing:

| Fragment | May be followed by |
|----------|-------------------|
| `expr`, `stmt` | `=>`, `,`, `;`, `)`, `]`, `}` |
| `pat` | `=>`, `,`, `=`, `|`, `if`, `in` |
| `ty`, `path` | `=>`, `,`, `;`, `=`, `>`, `{`, `[`, `:`, `as`, `where` |
| `ident`, `literal`, `lifetime` | Any token |
| `tt`, `block`, `item`, `meta`, `vis` | Any token |

**Example:** This is invalid because `:` cannot follow `$x:expr`:
```sigil
// INVALID: colon cannot follow expr
rune bad! { ($x:expr : $y:ty) => { } }
```

### 2.3 Repetition Patterns

```sigil
rune tuple! {
    // Zero or more: $(...)*
    ($($elem:expr),*) => { ($($elem),*) }
}

rune sum! {
    // One or more: $(...)+
    ($first:expr $(, $rest:expr)+) => {
        $first $(+ $rest)+
    }
}

rune maybe! {
    // Optional: $(...)?
    ($val:expr $(, default: $default:expr)?) => {
        match $val {
            Some(v) => v,
            None => { $($default)? },
        }
    }
}
```

### 2.4 Recursive Runes

```sigil
// Count elements at compile time
rune count! {
    () => { 0 }
    ($head:tt $($tail:tt)*) => { 1 + count!($($tail)*) }
}

// Reverse a token sequence
rune reverse! {
    () => {}
    ($head:tt $($tail:tt)*) => {
        reverse!($($tail)*)
        $head
    }
}

// Generate nested structure
rune nested! {
    ($val:expr; 0) => { $val }
    ($val:expr; $n:literal) => {
        Some(nested!($val; $n - 1))
    }
}
```

### 2.5 Hygiene and Scoping

```sigil
// Runes are hygienic by default — introduced identifiers don't clash
rune safe_swap! {
    ($a:expr, $b:expr) => {
        {
            let temp = $a    // 'temp' won't clash with caller's 'temp'
            $a = $b
            $b = temp
        }
    }
}

// Escape hygiene when needed
rune define_var! {
    ($name:ident = $val:expr) => {
        let #$name = $val   // # escapes hygiene, exposes to caller
    }
}

define_var!(x = 42)
print!(x)  // Works: x is visible
```

### 2.6 Pipe-Invoked Runes

Runes can be invoked using pipe syntax, enabling fluent method-chain-like patterns. When a rune is invoked via pipe (`|`), the left-hand value is bound to the magic variable `__pipe` within the rune body.

#### 2.6.1 Syntax

```sigil
// Standard invocation (prefix form)
rune_name!{args}

// Pipe invocation (infix form)
value|rune_name!{args}
```

The pipe form `value|rune!{args}` is semantically equivalent to `rune!{args}` with `__pipe` bound to `value` in the expansion environment.

#### 2.6.2 The `__pipe` Magic Variable

When a rune is pipe-invoked, the runtime automatically binds the left-hand value to `__pipe`:

```sigil
rune validate! {
    () => {
        ⎇ __pipe == "" {
            ⤺ Err(ValidationError·Empty("field"))
        }
        ⤺ Ok(__pipe)
    }
}

// Usage
≔ input = "hello"
≔ result = input|validate!{}  // __pipe = "hello" inside rune body
// result = Ok("hello")

≔ empty = ""
≔ result2 = empty|validate!{}  // __pipe = "" inside rune body
// result2 = Err(ValidationError·Empty("field"))
```

#### 2.6.3 Combining Pipe and Arguments

Pipe invocations can include additional arguments alongside `__pipe`:

```sigil
rune with_default! {
    ($default:expr) => {
        ⎇ __pipe == null {
            ⤺ $default
        }
        ⤺ __pipe
    }
}

// Usage
≔ value = null
≔ result = value|with_default!{42}  // Returns 42
```

#### 2.6.4 Chaining Pipe-Invoked Runes

Pipe invocations can be chained, with each rune receiving the output of the previous:

```sigil
rune trim! {
    () => { __pipe|String·trim() }
}

rune uppercase! {
    () => { __pipe|String·to_upper() }
}

rune validate_length! {
    ($min:expr) => {
        ⎇ __pipe|len() < $min {
            ⤺ Err("too short")
        }
        ⤺ Ok(__pipe)
    }
}

// Chained usage
≔ result = "  hello world  "|trim!{}|uppercase!{}|validate_length!{5}
// result = Ok("HELLO WORLD")
```

#### 2.6.5 Return Semantics in Pipe-Invoked Runes

Return statements (`⤺` or `return`) within a pipe-invoked rune body return from the **rune expansion**, not from the calling function. The rune body evaluates as a single expression:

```sigil
rune safe_divide! {
    ($divisor:expr) => {
        ⎇ $divisor == 0 {
            ⤺ Err("division by zero")  // Returns Err from rune, not caller
        }
        ⤺ Ok(__pipe / $divisor)
    }
}

fn calculate(x: i64) -> Result<i64, String> {
    ≔ step1 = x|safe_divide!{2}    // If divisor is 0, returns Err
    println("After macro")          // This line WILL execute
    step1                           // Return the result
}
```

#### 2.6.6 Type of `__pipe`

The `__pipe` variable has the same type as the left-hand expression. Within the rune body, `__pipe` can be used anywhere an expression of that type is expected.

**Important:** The rune author must ensure the rune body is compatible with the types passed via pipe. Type mismatches will be caught at compile time when the rune is expanded.

---

## 3. Invocations (Procedural Macros)

Invocations are arbitrary code transformations — Turing-complete metaprogramming.

### 3.1 Function-like Invocations

```sigil
// Defined in a separate crate (sigil_invoke)
//@ rune: invoke
pub fn sql!(input: TokenStream) -> TokenStream {
    let query = parse_sql(input)
    let validated = validate_against_schema(query)
    generate_query_code(validated)
}

// Usage
let users = sql!(SELECT * FROM users WHERE active = true)
```

### 3.2 Derive Invocations

```sigil
// Automatically implement traits
//@ rune: derive
pub fn Debug(input: TokenStream) -> TokenStream {
    let ast = parse_struct(input)
    let impl_block = generate_debug_impl(ast)
    impl_block
}

// Usage
//@ rune: derive(Debug, Clone, Eq)
struct Point {
    x: f64,
    y: f64,
}
```

### 3.3 Inscription Invocations (Attribute Macros)

```sigil
// Transform annotated items
//@ rune: inscription
pub fn route(attr: TokenStream, item: TokenStream) -> TokenStream {
    let method_path = parse_route(attr)
    let handler = parse_fn(item)
    generate_route_registration(method_path, handler)
}

// Usage
//@ inscription: route(GET, "/users/{id}")
fn get_user(id: u64) -> User~ {
    db|find(id)
}
```

### 3.4 Token Manipulation

```sigil
use sigil·invoke·{TokenStream, TokenTree, Span}

//@ rune: invoke
pub fn stringify_tokens!(input: TokenStream) -> TokenStream {
    let s = input|to_string
    quote! { #s }
}

// Quote quasi-quotation
//@ rune: invoke
pub fn make_fn!(input: TokenStream) -> TokenStream {
    let name: Ident = parse(input)

    quote! {
        fn #name() {
            print!("Called {}", stringify!(#name))
        }
    }
}
```

---

## 4. Seals (Type-Level Programming)

Seals are compile-time type computations — the language's type-level calculus.

### 4.1 Associated Types as Computation

```sigil
// Type-level natural numbers
seal Zero
seal Succ<N>

// Type aliases
type One = Succ<Zero>
type Two = Succ<One>
type Three = Succ<Two>

// Type-level addition
trait Add<Rhs> {
    type Result
}

impl<N> Add<Zero> for N {
    type Result = N
}

impl<N, M> Add<Succ<M>> for N
where N: Add<M>
{
    type Result = Succ<N·Add<M>·Result>
}

// Usage: type-level 2 + 3 = 5
type Five = Two·Add<Three>·Result
```

### 4.2 Type-Level Conditionals

```sigil
// Type-level boolean
seal True
seal False

// Type-level if
trait If<Then, Else> {
    type Result
}

impl<T, E> If<T, E> for True {
    type Result = T
}

impl<T, E> If<T, E> for False {
    type Result = E
}

// Type-level comparison
trait Equal<Rhs> {
    type Result  // True or False
}

impl Equal<Zero> for Zero {
    type Result = True
}

impl<N> Equal<Zero> for Succ<N> {
    type Result = False
}

impl<N> Equal<Succ<N>> for Zero {
    type Result = False
}

impl<N, M> Equal<Succ<M>> for Succ<N>
where N: Equal<M>
{
    type Result = N·Equal<M>·Result
}
```

### 4.3 Type-Level Lists

```sigil
// Heterogeneous list
seal HNil
seal HCons<Head, Tail>

// Type-level list operations
trait Append<Rhs> {
    type Result
}

impl<Rhs> Append<Rhs> for HNil {
    type Result = Rhs
}

impl<H, T, Rhs> Append<Rhs> for HCons<H, T>
where T: Append<Rhs>
{
    type Result = HCons<H, T·Append<Rhs>·Result>
}

// Length
trait Length {
    type Result  // Type-level number
}

impl Length for HNil {
    type Result = Zero
}

impl<H, T> Length for HCons<H, T>
where T: Length
{
    type Result = Succ<T·Length·Result>
}
```

### 4.4 Sealed Trait Pattern

```sigil
// Prevent external implementations
mod private {
    pub trait Sealed {}
}

pub trait MyTrait: private·Sealed {
    fn method(&self)
}

// Only types in this module can implement
impl private·Sealed for MyType {}
impl MyTrait for MyType {
    fn method(&self) { }
}
```

---

## 5. Wards (Compile-Time Safety)

Wards are protective patterns — compile-time checks that prevent misuse.

### 5.1 Const Assertions

```sigil
// Compile-time assertions
ward! {
    size_of::<Packet>() <= 1500,
    "Packet exceeds MTU"
}

ward! {
    align_of::<CacheLine>() == 64,
    "CacheLine must be 64-byte aligned"
}

// Ward as expression (fails compilation if false)
const _: () = ward!(N > 0, "N must be positive");
```

### 5.2 Type Constraints

```sigil
// Ward trait bounds
trait SafeTransmute<T>: Ward {
    ward size_of::<Self>() == size_of::<T>();
    ward align_of::<Self>() >= align_of::<T>();

    fn transmute(self) -> T {
        unsafe { std·mem·transmute(self) }
    }
}

// Conditional ward
trait ThreadSafe: Ward {
    ward Self: Send + Sync;
}
```

### 5.3 Contract Wards

```sigil
// Pre/post conditions checked at compile time where possible
fn divide(a: i32, b: i32) -> i32
ward pre { b != 0 }
ward post |result| { result * b == a }
{
    a / b
}

// For const inputs, checked at compile time
const HALF: i32 = divide(10, 2)  // OK
const BAD: i32 = divide(10, 0)   // Compile error!
```

### 5.4 State Machine Wards

```sigil
// Type-state pattern with wards
seal Closed
seal Open
seal Reading
seal Writing

struct File<State> {
    handle: RawHandle,
    _state: PhantomData<State>,
}

impl File<Closed> {
    fn open(path: &str) -> File<Open> { }
}

impl File<Open> {
    fn read(self) -> File<Reading> { }
    fn write(self) -> File<Writing> { }
    fn close(self) -> File<Closed> { }
}

impl File<Reading> {
    fn read_bytes(&mut self) -> [u8] { }
    fn finish(self) -> File<Open> { }
}

// Ward: can't read from closed file (compile error)
// let f = File·open("x")
// f.close().read()  // ERROR: File<Closed> has no method read
```

---

## 6. Glyphs (Compile-Time Evaluation)

Glyphs are values computed at compile time.

### 6.1 Const Functions

```sigil
// Evaluated at compile time when possible
const fn factorial(n: u64) -> u64 {
    match n {
        0 => 1,
        n => n * factorial(n - 1),
    }
}

const FACT_10: u64 = factorial(10)  // Computed at compile time

// Const blocks
const LOOKUP_TABLE: [u8; 256] = {
    let mut table = [0u8; 256]
    let mut i = 0
    while i < 256 {
        table[i] = (i as u8)|reverse_bits
        i += 1
    }
    table
}
```

### 6.2 Const Generics

```sigil
// Type parameterized by values
struct Array<T, const N: usize> {
    data: [T; N],
}

impl<T, const N: usize> Array<T, N> {
    const fn len(&self) -> usize { N }

    fn concat<const M: usize>(self, other: Array<T, M>) -> Array<T, {N + M}> {
        // ...
    }
}

// Usage
let a: Array<i32, 3> = Array { data: [1, 2, 3] }
let b: Array<i32, 2> = Array { data: [4, 5] }
let c: Array<i32, 5> = a|concat(b)
```

### 6.3 Const Traits

```sigil
// Trait methods usable in const context
const trait Default {
    const fn default() -> Self
}

const impl Default for i32 {
    const fn default() -> i32 { 0 }
}

const DEFAULT_INT: i32 = i32·default()
```

### 6.4 Compile-Time Strings

```sigil
// String manipulation at compile time
const fn make_greeting(name: &str) -> String {
    format!("Hello, {}!", name)
}

const GREETING: &str = make_greeting("World")

// Const format
const fn type_name<T>() -> &'static str {
    std·any·type_name::<T>()
}
```

---

## 7. Inscriptions (Attribute System)

Inscriptions are metadata that modify compilation.

### 7.1 Built-in Inscriptions

```sigil
//@ rune: derive(Debug, Clone, Eq, Hash)
struct Point { x: f64, y: f64 }

//@ rune: inline(always)
fn hot_path() { }

//@ rune: repr(C, packed)
struct FFIStruct { }

//@ rune: cfg(target_os = "linux")
fn linux_only() { }

//@ rune: deprecated("Use new_api instead")
fn old_api() { }

//@ rune: must_use("Handle the error")
fn fallible() -> Result<T, E> { }

//@ rune: test
fn test_addition() {
    assert!(1 + 1 == 2)
}
```

### 7.2 Custom Inscriptions

```sigil
// Define custom attribute
//@ rune: inscription
pub fn cached(attr: TokenStream, item: TokenStream) -> TokenStream {
    let ttl = parse_duration(attr)
    let func = parse_fn(item)

    quote! {
        fn #func.name(#func.params) -> #func.return_type {
            static CACHE: Lazy<Cache> = Lazy·new(|| Cache·new(#ttl));

            if let Some(cached) = CACHE|get(&#func.params) {
                return cached·clone()
            }

            let result = { #func.body };
            CACHE|insert(#func.params, result·clone());
            result
        }
    }
}

// Usage
//@ inscription: cached(ttl = 5·minutes)
fn expensive_computation(x: i32) -> i32 {
    // ...
}
```

### 7.3 Conditional Compilation

```sigil
//@ rune: cfg(feature = "serde")
impl Serialize for MyType { }

//@ rune: cfg(all(target_os = "linux", target_arch = "x86_64"))
fn optimized_path() { }

//@ rune: cfg(not(debug_assertions))
fn release_only() { }

// Cfg in expressions
let value = cfg_if! {
    if #[cfg(windows)] {
        windows_impl()
    } else if #[cfg(unix)] {
        unix_impl()
    } else {
        generic_impl()
    }
}
```

---

## 8. Grimoire Integration

The Grimoire is Sigil's central registry of Runes, Invocations, and Inscriptions.

### 8.1 Importing from Grimoire

```sigil
// Standard library runes
use std·runes·{vec!, format!, print!, debug!}

// Third-party runes
use serde·runes·{Serialize, Deserialize}
use tokio·runes·{async_trait, select!}

// Project-local grimoire
use crate·grimoire·{sql!, cached!, traced!}
```

### 8.2 Exporting to Grimoire

```sigil
// In lib.sg
//@ rune: export_runes
pub mod grimoire {
    pub use crate·runes·*;
    pub use crate·invocations·*;
    pub use crate·inscriptions·*;
}
```

### 8.3 Rune Documentation

```sigil
/// Creates a vector from a list of elements.
///
/// # Examples
///
/// ```sigil
/// let v = vec![1, 2, 3];
/// assert!(v.len() == 3);
/// ```
///
/// # Incantation Pattern
///
/// ```text
/// vec![]           → empty vector
/// vec![x]          → single element
/// vec![x, y, z]    → multiple elements
/// vec![x; n]       → n copies of x
/// ```
rune vec! {
    // ...
}
```

---

## 9. Domain-Specific Runes

### 9.1 SQL Rune

```sigil
// Compile-time SQL validation
let users = sql! {
    SELECT id, name, email
    FROM users
    WHERE active = true
    ORDER BY created_at DESC
    LIMIT 10
}

// With interpolation (parameterized)
let user = sql! {
    SELECT * FROM users WHERE id = ${user_id}
}

// Type-safe: columns map to struct fields
//@ rune: derive(FromRow)
struct User {
    id: u64,
    name: String,
    email: String?,
}
```

### 9.2 Regex Rune

```sigil
// Compile-time regex validation
let email_pattern = regex!(r"^[\w.+-]+@[\w.-]+\.\w+$")

// Named captures become struct fields
let parsed = regex_captures!(
    r"(?P<year>\d{4})-(?P<month>\d{2})-(?P<day>\d{2})",
    date_string
)
print!(parsed.year, parsed.month, parsed.day)
```

### 9.3 Format Rune

```sigil
// Type-checked format strings
let msg = format!("User {} has {} points", name, score)

// Compile error if types don't match
// format!("{}", some_struct)  // ERROR: SomeStruct doesn't impl Display

// Positional and named
format!("{0} {1} {0}", a, b)
format!("{name} is {age} years old", name = user.name, age = user.age)
```

### 9.4 Test Rune

```sigil
//@ rune: test
fn test_addition() {
    assert_eq!(2 + 2, 4)
}

//@ rune: test, should_panic(expected = "divide by zero")
fn test_division_by_zero() {
    divide(1, 0)
}

//@ rune: test, async
async fn test_async_fetch() {
    let result = fetch("http://example.com")|await
    assert!(result.status == 200)
}

// Property-based testing
//@ rune: quickcheck
fn prop_reverse_reverse(xs: Vec<i32>) -> bool {
    xs|reverse|reverse == xs
}
```

---

## 10. Metaprogramming Patterns

### 10.1 Builder Pattern Generation

```sigil
//@ rune: derive(Builder)
struct Config {
    host: String,
    port: u16 = 8080,
    timeout: Duration = 30·sec,
    retries: u32 = 3,
}

// Generated:
let config = Config·builder()
    |host("localhost")
    |port(3000)
    |build()
```

### 10.2 Enum Dispatch

```sigil
//@ rune: derive(EnumDispatch)
trait Animal {
    fn speak(&self) -> str
}

//@ rune: derive(EnumDispatch(Animal))
enum Pet {
    Dog(Dog),
    Cat(Cat),
    Bird(Bird),
}

// Generated: Pet::speak() dispatches to inner type
```

### 10.3 Newtype Derivation

```sigil
//@ rune: derive(Newtype)
struct UserId(u64)

// Generated: Deref, From, Into, Display, etc.
let id: UserId = 42·into()
let raw: u64 = *id
```

### 10.4 Visitor Pattern

```sigil
//@ rune: derive(Visitor)
enum Expr {
    Literal(i64),
    Add(Box<Expr>, Box<Expr>),
    Mul(Box<Expr>, Box<Expr>),
}

// Generated visitor trait and walk functions
impl ExprVisitor for Evaluator {
    fn visit_literal(&mut self, n: i64) -> i64 { n }
    fn visit_add(&mut self, a: i64, b: i64) -> i64 { a + b }
    fn visit_mul(&mut self, a: i64, b: i64) -> i64 { a * b }
}
```

---

## 11. Debugging Metaprogramming

### 11.1 Expansion Visibility

```sigil
// Show macro expansion
//@ rune: trace_expansion
let v = vec![1, 2, 3]

// Compiler output shows expanded code:
// let v = {
//     let mut v = Vec·with_capacity(3);
//     v|push(1);
//     v|push(2);
//     v|push(3);
//     v
// }
```

### 11.2 Compile-Time Printing

```sigil
// Print during compilation
rune debug_type! {
    ($t:ty) => {
        const _: () = {
            compile_print!("Type: {}", stringify!($t));
        };
    }
}

debug_type!(Vec<i32>)
// Compiler prints: "Type: Vec<i32>"
```

### 11.3 Type-Level Debugging

```sigil
// Print type-level computation
trait ShowType {
    const NAME: &'static str;
}

impl ShowType for Zero {
    const NAME: &'static str = "Zero";
}

impl<N: ShowType> ShowType for Succ<N> {
    const NAME: &'static str = concat!("Succ<", N·NAME, ">");
}

const _: () = compile_print!(Three·NAME);
// Prints: "Succ<Succ<Succ<Zero>>>"
```

---

## 12. Summary: Metaprogramming Taxonomy

| Mechanism | Evaluated | Input | Output | Use Case |
|-----------|-----------|-------|--------|----------|
| **Rune** | Compile | Tokens | Tokens | Pattern-based code gen |
| **Invocation** | Compile | Tokens | Tokens | Arbitrary transformation |
| **Seal** | Compile | Types | Types | Type-level computation |
| **Ward** | Compile | Expr | Assert | Safety constraints |
| **Glyph** | Compile | Values | Values | Const computation |
| **Inscription** | Compile | Item | Item | Declarative decoration |

The Sigil programmer wields these tools like a grimoire — each incantation reshaping code at the moment of compilation, before a single instruction executes.

---

## 13. Implementation Notes and Known Limitations

This section documents the current implementation state, known limitations, and edge cases for the interpreter-based rune system.

### 13.1 Implementation Status

| Feature | Status | Notes |
|---------|--------|-------|
| Basic rune definition | ✅ Complete | `rune name! { pattern => body }` |
| Positional parameters | ✅ Complete | `($x:expr)`, `($a:expr, $b:expr)` |
| Named parameters | ✅ Complete | `(field: $var:expr)` |
| Pipe invocation | ✅ Complete | `value\|rune!{args}` with `__pipe` |
| Return scoping | ✅ Complete | Return in macro returns from macro, not caller |
| Fragment types (core) | ✅ Complete | `expr`, `ident`, `ty`, `literal`, `block`, `stmt`, `pat`, `tt` |
| Fragment types (extended) | ❌ Pending | `path`, `item`, `meta`, `lifetime`, `vis` |
| Repetition patterns | ❌ Pending | `$(...)*`, `$(...)+`, `$(...)?` |
| Recursive runes | ❌ Pending | Runes that invoke themselves |
| Hygiene | ❌ Pending | Variable scoping in expansions |

### 13.2 Known Limitations

#### 13.2.1 Token-Based Pattern Storage

**Limitation:** Rune patterns and bodies are stored as tokenized strings rather than proper AST nodes.

**Impact:**
- Pattern matching relies on string manipulation of token sequences
- Some edge cases with complex nesting may fail to match
- Error messages may reference token positions rather than source positions

**Example:**
```sigil
// The pattern ($n:expr) is stored internally as:
// "LParen Ident(\"n\") Colon Ident(\"expr\") RParen FatArrow ..."
```

**Workaround:** Keep rune patterns simple. Avoid deeply nested token structures in patterns.

#### 13.2.2 Argument Splitting Heuristics

**Limitation:** Comma-splitting for multiple arguments uses depth tracking that may fail with complex expressions.

**Affected Cases:**
- String literals containing commas: `"a, b"` may be incorrectly split
- Nested generic types with multiple parameters
- Complex expressions with multiple comma-separated sub-expressions

**Example (may fail):**
```sigil
rune process! { ($a:expr, $b:expr) => { ... } }
process!("hello, world", x)  // May incorrectly split at the comma in the string
```

**Workaround:** Wrap complex arguments in parentheses: `process!(("hello, world"), x)`

#### 13.2.3 Named Argument Detection

**Limitation:** Named arguments are detected by checking for colon (`:`) in argument strings.

**Impact:** Arguments containing colons (like ternary expressions or type annotations) may be misinterpreted as named arguments.

**Example (may fail):**
```sigil
rune cond! { ($x:expr) => { ... } }
cond!(a ? b : c)  // The `:` may cause incorrect parsing
```

**Workaround:** Wrap such expressions: `cond!((a ? b : c))`

#### 13.2.4 Statement-Starting Body Detection

**Limitation:** The parser must detect when a macro body starts with a statement (requiring block wrapping) vs. an expression.

**Detected Patterns:**
- `≔` (let binding)
- `⎇` (if statement)
- `⤺` (return statement)
- `let `, `if `, `return ` (ASCII equivalents)

**Not Detected:**
- `while`, `for`, `loop` as statement starters
- Match expressions that don't start with known patterns
- Custom statement-like constructs

**Workaround:** Explicitly wrap complex macro bodies in blocks: `{ ... }`

#### 13.2.5 String Literal Handling in Detokenization

**Limitation:** String literals in tokenized patterns may lose escape sequences during round-trip processing.

**Impact:** Patterns or bodies containing escaped characters (`\n`, `\t`, `\"`) may not expand correctly.

**Workaround:** Avoid complex escape sequences in rune definitions. Use raw strings where possible.

### 13.3 Edge Cases

#### 13.3.1 Empty Rune Arguments

Both forms are valid and equivalent:
```sigil
rune empty! { () => { 42 } }
empty!()    // OK
empty!{}    // OK - braces treated as empty args
```

#### 13.3.2 Pipe with No Arguments

```sigil
rune identity! { () => { __pipe } }
42|identity!{}   // Returns 42
42|identity!()   // Also valid
```

#### 13.3.3 Nested Pipe Invocations

```sigil
rune add_one! { () => { __pipe + 1 } }
1|add_one!{}|add_one!{}|add_one!{}   // Returns 4
```

Each `|` creates a new scope with its own `__pipe` binding.

#### 13.3.4 Return Value Propagation

Return statements in macro bodies are captured and converted to the macro's value:

```sigil
rune early_return! {
    ($cond:expr) => {
        ⎇ $cond { ⤺ "early" }
        "normal"
    }
}

≔ r1 = true|early_return!{}   // "early" (return captured)
≔ r2 = false|early_return!{}  // "normal"
```

### 13.4 Future Work

The following improvements are planned:

1. **AST-based pattern storage** — Replace tokenized string storage with proper AST nodes
2. **Proper lexer integration** — Use the lexer for argument splitting instead of string heuristics
3. **Repetition patterns** — Implement `$(...)*`, `$(...)+`, `$(...)?` repetition
4. **Extended fragment types** — Implement `path`, `item`, `meta`, `lifetime`, `vis`
5. **Hygiene system** — Implement proper macro hygiene with gensym for introduced identifiers
6. **Better error messages** — Source-mapped errors pointing to original rune definitions
