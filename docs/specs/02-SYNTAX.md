# Sigil Syntax Specification

## 1. Program Structure

A Sigil program consists of a series of items organized into modules.

```sigil
//! Module documentation

use std·io·{Read, Write}
use std·collections·HashMap

// Module-level items
const MAX_SIZE: usize = 1024

struct Config {
    name: str,
    value: i32,
}

fn main() {
    // Entry point
}
```

---

## 2. Items

### 2.1 Functions

```sigil
// Basic function
fn add(a: i32, b: i32) -> i32 {
    a + b
}

// With evidentiality
fn fetch(url: str) -> Response~ {
    http·get(url)~
}

// Generic function
fn identity<T>(x: T) -> T {
    x
}

// With trait bounds
fn print_all<T: Display>(items: [T]) {
    items|τ{print}
}

// Async function
async fn fetch_data(id: u64) -> Data~ {
    api·get!("/data/{id}")|await~
}

// Const function (compile-time evaluable)
const fn square(x: i32) -> i32 {
    x * x
}
```

### 2.2 Polysynthetic Function Syntax

Functions can use morpheme chains in their signatures:

```sigil
// Traditional
fn filter_map_collect<T, U>(
    data: [T],
    predicate: fn(T) -> bool,
    mapper: fn(T) -> U
) -> [U] {
    data|φ{predicate}|τ{mapper}|vec
}

// Polysynthetic: morphemes in signature
fn process|φ·τ<T, U>(
    data: [T],
    predicate: fn(T) -> bool,
    mapper: fn(T) -> U
) -> [U] {
    data|φ{predicate}|τ{mapper}|vec
}

// Usage
let result = numbers|process|φ·τ(is_even, double)
```

### 2.3 Structures

```sigil
// Basic struct
struct Point {
    x: f64,
    y: f64,
}

// With default values
struct Config {
    host: str = "localhost",
    port: u16 = 8080,
    timeout: Duration = 30·sec,
}

// Tuple struct
struct Color(u8, u8, u8)

// Unit struct
struct Marker

// Generic struct with bounds
struct Container<T: Clone + Debug> {
    items: [T],
    capacity: usize,
}

// With evidentiality fields
struct User {
    id: u64!,           // Always known
    name: str!,         // Always known
    email: str?,        // Maybe absent
    avatar_url: str~,   // From external source
}
```

### 2.4 Enums

```sigil
// Basic enum
enum Direction {
    North,
    South,
    East,
    West,
}

// With associated data
enum Message {
    Quit,
    Move { x: i32, y: i32 },
    Write(str),
    Color(u8, u8, u8),
}

// With evidentiality
enum Result<T!, E~> {
    Ok(T!),      // Success is known
    Err(E~),     // Errors are reported
}

enum Option<T?> {
    Some(T),
    None,
}

// Algebraic with methods
enum Tree<T> {
    Leaf(T),
    Node {
        left: Box<Tree<T>>,
        right: Box<Tree<T>>,
    },

    fn depth(self) -> usize {
        match self {
            Leaf(_) => 1,
            Node { left, right } => 1 + max(left.depth(), right.depth()),
        }
    }
}
```

### 2.5 Traits

```sigil
// Basic trait
trait Display {
    fn fmt(self, f: &mut Formatter) -> Result
}

// With associated types
trait Iterator {
    type Item

    fn next(&mut self) -> Option<Self·Item>?
}

// With default implementations
trait Greet {
    fn name(self) -> str

    fn greet(self) -> str {
        "Hello, {self.name()}!"
    }
}

// With morpheme requirements
trait Transformable {
    fn τ<U>(self, f: fn(Self) -> U) -> U
    fn φ(self, pred: fn(&Self) -> bool) -> Option<Self>
}

// Trait with evidentiality
trait Fetch {
    type Response~    // Responses are always external

    fn fetch(self, url: str) -> Self·Response~
}
```

### 2.6 Implementations

```sigil
// Basic impl
impl Point {
    fn new(x: f64, y: f64) -> Point {
        Point { x, y }
    }

    fn distance(self, other: Point) -> f64 {
        let dx = self.x - other.x
        let dy = self.y - other.y
        (dx*dx + dy*dy)|sqrt
    }
}

// Trait impl
impl Display for Point {
    fn fmt(self, f: &mut Formatter) -> Result {
        write!(f, "({}, {})", self.x, self.y)
    }
}

// Generic impl
impl<T: Display> Debug for Container<T> {
    fn fmt(self, f: &mut Formatter) -> Result {
        self.items|τ{Display·fmt}|join(", ")|write!(f)
    }
}
```

---

## 3. Expressions

### 3.1 Pipe Expressions

The pipe `|` is the primary composition operator:

```sigil
// Left-to-right data flow
data|operation1|operation2|operation3

// Equivalent to (but more readable than)
operation3(operation2(operation1(data)))
```

**Pipe semantics:**

```sigil
// Value pipe: passes value as first argument
x|f           // f(x)
x|f(a, b)     // f(x, a, b)

// Method pipe: calls method on value
x|.method     // x.method()
x|.method(a)  // x.method(a)

// Block pipe: passes value to block
x|{ _ + 1 }   // (|_| _ + 1)(x)

// Morpheme pipe: applies transformation
x|τ{f}        // x.map(f)
x|φ{p}        // x.filter(p)
```

### 3.2 Morpheme Expressions

```sigil
// Transform (τ): map over elements
[1, 2, 3]|τ{_ * 2}           // [2, 4, 6]
[1, 2, 3]|τ*2                // [2, 4, 6] (shorthand)
[1, 2, 3]|τ{x => x * x}      // [1, 4, 9]

// Filter (φ): select elements
[1, 2, 3, 4]|φ{_ > 2}        // [3, 4]
[1, 2, 3, 4]|φ>2             // [3, 4] (shorthand)
users|φ.active               // filter where .active is true

// Sort (σ): order elements
[3, 1, 2]|σ                  // [1, 2, 3]
users|σ.name                 // sort by name ascending
users|σ·desc.created         // sort by created descending

// Reduce (ρ): fold to single value
[1, 2, 3]|ρ+                 // 6 (sum)
[1, 2, 3]|ρ(*, 1)            // 6 (product)
[1, 2, 3]|ρ{acc, x => acc + x, 0}  // 6 (explicit)

// First/Last (α/Ω)
[1, 2, 3]|α                  // 1
[1, 2, 3]|Ω                  // 3
[1, 2, 3]|α?                 // Some(1) or None

// Project (π): select fields
users|π{.name, .email}       // [{name, email}, ...]
record|π.name                // record.name
```

### 3.3 Compound Expressions (Incorporation)

```sigil
// Basic incorporation
path|file·read·lines         // open file, read, split to lines

// With arguments
url|http·get·json·parse(Config)

// Nested incorporation
data|json·parse·validate·transform·{
    .name|str·trim·upper,
    .age|int·clamp(0, 150),
}

// Incorporation with error handling
path|file·read·parse? |> {
    Ok(data) => process(data),
    Err(e) => log(e),
}
```

### 3.4 Block Expressions

```sigil
// Expression blocks (last expression is value)
let result = {
    let x = compute()
    let y = transform(x)
    x + y    // returned
}

// Lambda blocks
let double = |x| x * 2
let add = |a, b| a + b
let complex = |x| {
    let intermediate = x * 2
    intermediate + 1
}

// Placeholder blocks
data|τ{_ * 2}           // |x| x * 2
data|τ{_1 + _2}         // |a, b| a + b
data|φ{_.active}        // |x| x.active
```

### 3.5 Control Flow Expressions

```sigil
// If expression
let max = if a > b { a } else { b }

// Match expression (exhaustive)
let name = match user {
    Admin { name, .. } => "Admin: {name}",
    User { name, .. } => name,
    Guest => "Anonymous",
}

// Match with guards
match value {
    n if n < 0 => "negative",
    0 => "zero",
    n if n < 10 => "small",
    _ => "large",
}

// Match with evidentiality
match result? {
    Some(v!) => use(v!),     // Known value
    Some(v~) => validate(v~), // External value
    None => default(),
}

// Loop expressions
let sum = loop {
    if condition { break accumulated }
    accumulated += next()
}

// For expressions
let squares = for x in 1..10 {
    yield x * x
}

// While expressions
while let Some(item) = iter.next() {
    process(item)
}
```

### 3.6 Try Expressions

```sigil
// Propagate errors (? operator)
fn load_config() -> Config? {
    let content = file·read(path)?
    let parsed = json·parse(content)?
    Config·from(parsed)?
}

// Try-catch style
let result = try {
    risky_operation()
} catch {
    IOError(e) => default_value,
    ParseError(e) => handle_parse(e),
    _ => panic!("Unexpected error"),
}

// Assert evidentiality
let known! = uncertain?!   // Panic if None
let trusted! = external~!  // Assert trust
```

---

## 4. Patterns

### 4.1 Basic Patterns

```sigil
// Literal patterns
match x {
    0 => "zero",
    1 => "one",
    _ => "many",
}

// Variable binding
match point {
    Point { x, y } => compute(x, y),
}

// Destructuring
let (a, b, c) = tuple
let [first, second, ..rest] = array
let Point { x, y } = point

// Reference patterns
match &value {
    &Some(ref inner) => use(inner),
    &None => skip(),
}
```

### 4.2 Evidentiality Patterns

```sigil
// Match on evidentiality
match value {
    known! => "definitely have it",
    uncertain? => "might have it",
    reported~ => "someone said so",
    paradox‽ => "trust boundary",
}

// Combine with structural patterns
match response {
    Ok(data!) => process(data!),
    Ok(data~) => validate_then_process(data~),
    Err(e?) => maybe_retry(e?),
    Err(e~) => log_external_error(e~),
}
```

### 4.3 Morpheme Patterns

```sigil
// Pattern in transformation
data|τ{
    Point { x, y } => Point { x: x * 2, y: y * 2 }
}

// Pattern in filter
data|φ{
    User { active: true, .. } => true,
    _ => false,
}

// Shorthand
data|φ·User{ active: true }
```

---

## 5. Statements

### 5.1 Let Bindings

```sigil
// Immutable binding
let x = 42

// With type annotation
let x: i32 = 42

// With evidentiality
let known!: i32 = compute()
let maybe?: i32 = lookup(key)
let external~: i32 = fetch(url)

// Mutable binding
let mut counter = 0

// Destructuring
let (x, y) = point.coords()
let Point { x, y } = point
```

### 5.2 Assignment

```sigil
// Simple assignment
x = 42

// Compound assignment
x += 1
x *= 2
x |= mask

// Evidentiality assignment
known! = compute()!      // Must produce known value
maybe? = lookup()?       // May be uncertain
```

---

## 6. Modules and Imports

### 6.1 Module Declaration

```sigil
// In lib.sg
mod parser        // loads parser.sg or parser/mod.sg
mod lexer
pub mod ast       // public module

// Inline module
mod helpers {
    pub fn utility() { }
}
```

### 6.2 Use Declarations

```sigil
// Simple import
use std·io

// Multiple imports
use std·io·{Read, Write, BufReader}

// Renamed import
use std·collections·HashMap as Map

// Glob import (discouraged)
use std·prelude·*

// Nested paths
use std·{
    io·{Read, Write},
    collections·{HashMap, HashSet},
    sync·{Arc, Mutex},
}

// Re-export
pub use internal·Type
```

---

## 7. Attributes (Runes)

Attributes in Sigil are called **Runes** — metadata annotations that modify compilation:

```sigil
//@ rune: derive(Debug, Clone, Eq)
struct Point {
    x: i32,
    y: i32,
}

//@ rune: inline(always)
fn hot_path() { }

//@ rune: cfg(target_os = "linux")
fn linux_only() { }

//@ rune: test
fn test_addition() {
    assert!(1 + 1 == 2)
}

//@ rune: deprecated("Use new_function instead")
fn old_function() { }

//@ rune: must_use("This Result must be handled")
fn fallible() -> Result<T, E> { }
```

### 7.1 Built-in Runes

| Rune | Purpose |
|------|---------|
| `derive(...)` | Auto-implement traits |
| `inline(...)` | Inlining hints |
| `cfg(...)` | Conditional compilation |
| `test` | Mark test function |
| `bench` | Mark benchmark |
| `deprecated` | Deprecation warning |
| `must_use` | Require result handling |
| `repr(...)` | Memory layout control |
| `allow(...)` | Suppress warnings |
| `deny(...)` | Treat warnings as errors |
| `unsafe` | Mark unsafe block/fn |

---

## 8. Grammar Summary (EBNF)

```ebnf
program       = item* ;

item          = function | struct | enum | trait | impl | mod | use | const ;

function      = "fn" IDENT generics? params "->" type block ;
params        = "(" (param ("," param)*)? ")" ;
param         = IDENT ":" type evidentiality? ;

struct        = "struct" IDENT generics? "{" (field ("," field)*)? "}" ;
field         = IDENT ":" type evidentiality? ("=" expr)? ;

enum          = "enum" IDENT generics? "{" (variant ("," variant)*)? "}" ;
variant       = IDENT ("(" types ")" | "{" fields "}")? ;

trait         = "trait" IDENT generics? "{" trait_item* "}" ;
impl          = "impl" generics? type ("for" type)? "{" impl_item* "}" ;

type          = path | "[" type "]" | "(" types ")" | type evidentiality ;
evidentiality = "!" | "?" | "~" | "‽" ;

expr          = pipe_expr | block | if_expr | match_expr | loop_expr ;
pipe_expr     = unary_expr ("|" pipe_op)* ;
pipe_op       = morpheme | "." IDENT | block | call ;
morpheme      = ("τ" | "φ" | "σ" | "ρ" | "α" | "Ω") morpheme_arg? ;

block         = "{" statement* expr? "}" ;
statement     = let_stmt | expr_stmt | item ;
let_stmt      = "let" "mut"? pattern (":" type)? "=" expr ;
```

---

## 9. Example Program

```sigil
//! User management service

use std·io·{Read, Write}
use std·net·TcpStream
use serde·{Serialize, Deserialize}

//@ rune: derive(Debug, Clone, Serialize, Deserialize)
struct User {
    id: u64!,
    name: str!,
    email: str?,
    role: Role!,
}

enum Role {
    Admin,
    Moderator,
    Member,
    Guest,
}

trait Repository<T> {
    fn find(id: u64) -> T?
    fn find_all() -> [T]!
    fn save(item: T) -> T!
}

impl Repository<User> for Database {
    fn find(id: u64) -> User? {
        sql!"SELECT * FROM users WHERE id = {id}"
            |query·first?
            |τ{User·from}
    }

    fn find_all() -> [User]! {
        sql!"SELECT * FROM users"
            |query·all!
            |τ{User·from}
            |vec
    }

    fn save(user: User) -> User! {
        sql!"INSERT INTO users VALUES ({user})"
            |execute!
        user
    }
}

fn main() {
    let db = Database·connect("postgres://localhost/app")!

    // Fetch and process users
    let admins! = db
        |find_all
        |φ{.role == Role·Admin}
        |σ.name
        |τ{.email?|unwrap_or("no-email")}
        |vec!

    // Export results
    admins!
        |json·encode!
        |file·write("admins.json")!

    print!("Exported {admins!.len} admin emails")
}
```
