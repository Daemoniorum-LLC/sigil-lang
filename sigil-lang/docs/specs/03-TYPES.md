# Sigil Type System Specification

## 1. Overview

Sigil's type system combines:
- **Strong static typing** — All types known at compile time
- **Type inference** — Types deduced when unambiguous
- **Evidentiality** — Provenance and certainty tracked in types
- **Algebraic data types** — Sum and product types
- **Traits** — Interface-based polymorphism
- **Generics** — Parametric polymorphism with bounds

---

## 2. Primitive Types

### 2.1 Numeric Types

| Type | Size | Range |
|------|------|-------|
| `i8` | 8-bit | -128 to 127 |
| `i16` | 16-bit | -32,768 to 32,767 |
| `i32` | 32-bit | -2³¹ to 2³¹-1 |
| `i64` | 64-bit | -2⁶³ to 2⁶³-1 |
| `i128` | 128-bit | -2¹²⁷ to 2¹²⁷-1 |
| `isize` | ptr | Platform-dependent |
| `u8` | 8-bit | 0 to 255 |
| `u16` | 16-bit | 0 to 65,535 |
| `u32` | 32-bit | 0 to 2³²-1 |
| `u64` | 64-bit | 0 to 2⁶⁴-1 |
| `u128` | 128-bit | 0 to 2¹²⁸-1 |
| `usize` | ptr | Platform-dependent |
| `f32` | 32-bit | IEEE 754 single |
| `f64` | 64-bit | IEEE 754 double |

### 2.2 Text Types

| Type | Description |
|------|-------------|
| `char` | Unicode scalar value (4 bytes) |
| `str` | UTF-8 string slice |
| `String` | Owned UTF-8 string |
| `byte` | Alias for `u8` |
| `bytes` | Byte slice `[u8]` |

### 2.3 Boolean and Unit

| Type | Values |
|------|--------|
| `bool` | `true`, `false` |
| `()` | Unit (single value `()`) |
| `!` / `⊥` | Never type (no values) |

---

## 3. Evidentiality Types

Evidentiality is a **type-level property** that tracks the provenance of values.

### 3.1 Evidentiality Markers

| Marker | Name | Meaning | Trust Level |
|--------|------|---------|-------------|
| `T!` | Known | Computed locally | Full |
| `T?` | Uncertain | May be absent | Requires handling |
| `T~` | Reported | External source | Untrusted |
| `T‽` | Paradox | Trust boundary | Explicit assertion |

### 3.2 Type Semantics

```sigil
// Known (!) - Direct knowledge
// Value definitely exists and was computed locally
let sum!: i32 = 1 + 2 + 3          // We computed it
let parsed!: Config = toml·parse(content)!  // Asserted

// Uncertain (?) - Possible absence
// Value might not exist; must handle both cases
let user?: User = db.find(id)      // Might not exist
let first?: i32 = list|α?          // List might be empty

// Reported (~) - External source
// Value came from outside the trust boundary
let response~: Response = http·get(url)  // From network
let input~: str = stdin·read()           // From user
let config~: Config = env·load()         // From environment

// Paradox (‽) - Trust assertion
// Explicitly crossing trust boundaries
let trusted‽: i32 = unsafe { raw_ptr }   // Unsafe trust
let assumed‽: User = unvalidated~!       // Forced trust
```

### 3.3 Evidentiality Lattice

Evidentiality forms a lattice with ordering:

```
        ⊤ (any evidence)
       /|\
      / | \
     /  |  \
    !   ?   ~
     \  |  /
      \ | /
       \|/
        ‽
        |
        ⊥ (no evidence)
```

**Combining rules:**

| Op | `!` | `?` | `~` | `‽` |
|----|-----|-----|-----|-----|
| `!` | `!` | `?` | `~` | `‽` |
| `?` | `?` | `?` | `?` | `‽` |
| `~` | `~` | `?` | `~` | `‽` |
| `‽` | `‽` | `‽` | `‽` | `‽` |

```sigil
// Evidence combines pessimistically
let a!: i32 = 10
let b?: i32 = lookup(key)
let c~: i32 = fetch(url)

let sum? = a! + b?      // Known + Uncertain = Uncertain
let sum~ = a! + c~      // Known + Reported = Reported
let sum? = b? + c~      // Uncertain + Reported = Uncertain (most restrictive)
```

### 3.4 Evidence Transitions

```sigil
// Upgrade: Uncertain → Known (by handling absence)
let value! = uncertain? else { return default }
let value! = uncertain?!  // Panic if absent

// Upgrade: Reported → Known (by validation)
let value! = reported~|validate!
let value! = reported~ match {
    Valid(v) => v!,
    Invalid(_) => panic!("validation failed"),
}

// Downgrade: Known → Uncertain (optional wrapper)
let optional?: i32 = Some(known!)

// Trust assertion: Any → Paradox → Known
let trusted‽ = untrusted~|trust  // Explicit boundary
let known! = trusted‽!           // Assert from paradox
```

### 3.5 Evidence in Function Signatures

```sigil
// Return type evidence
fn compute() -> i32!          // Always returns known value
fn lookup(k: K) -> V?         // Might not find value
fn fetch(url: str) -> Data~   // Returns external data
fn dangerous() -> Value‽      // Returns trust boundary

// Parameter evidence requirements
fn process(data: Data!)       // Requires known data
fn maybe_process(data: Data?) // Accepts uncertain
fn validate(input: Input~) -> Output!  // Transforms trust level

// Evidence polymorphism
fn transform<E>(value: T^E) -> U^E    // Preserves evidence
fn trust_boundary<E>(value: T^E) -> T! // Elevates to known
```

---

## 4. Composite Types

### 4.1 Arrays and Slices

```sigil
// Fixed-size array
let arr: [i32; 5] = [1, 2, 3, 4, 5]

// Dynamic array (vector)
let vec: [i32] = [1, 2, 3, 4, 5]
let vec: Vec<i32> = vec![1, 2, 3]

// Slice (borrowed view)
let slice: &[i32] = &arr[1..4]

// With evidentiality
let known_arr!: [i32; 3] = [1, 2, 3]
let maybe_arr?: [User] = db.query()?
```

### 4.2 Tuples

```sigil
// Basic tuple
let pair: (i32, str) = (42, "hello")

// Access by index
let x = pair.0
let y = pair.1

// Destructuring
let (x, y) = pair

// Named tuple (struct)
struct Point(x: f64, y: f64)
```

### 4.3 Structs

```sigil
// Product type
struct User {
    id: u64!,        // Known: database-generated
    name: str!,      // Known: required field
    email: str?,     // Uncertain: optional field
    bio: str~,       // Reported: user-provided
}

// Struct instantiation
let user = User {
    id: 1!,
    name: "Alice"!,
    email: Some("alice@example.com"),
    bio: form_input~,
}

// Field access preserves evidentiality
let name! = user.name    // str!
let email? = user.email  // str?
let bio~ = user.bio      // str~
```

### 4.4 Enums (Sum Types)

```sigil
// Simple enum
enum Status {
    Active,
    Inactive,
    Pending,
}

// Enum with data
enum Result<T!, E~> {
    Ok(T!),
    Err(E~),
}

enum Option<T?> {
    Some(T),
    None,
}

// Rich enum
enum Message {
    Text { content: str~, sender: User! },
    Image { url: str~, dimensions: (u32, u32)! },
    System { code: u32!, message: str! },
}
```

---

## 5. Reference Types

### 5.1 Shared References

```sigil
// Immutable borrow
let r: &T = &value

// With lifetime
let r: &'a T = &value

// Reference preserves evidentiality
let known_ref: &i32! = &known_value!
let uncertain_ref: &User? = &maybe_user?
```

### 5.2 Mutable References

```sigil
// Mutable borrow (exclusive)
let r: &mut T = &mut value

// With lifetime
let r: &'a mut T = &mut value
```

### 5.3 Smart Pointers

| Type | Ownership | Thread-safe |
|------|-----------|-------------|
| `Box<T>` | Unique | No |
| `Rc<T>` | Shared | No |
| `Arc<T>` | Shared | Yes |
| `Weak<T>` | Non-owning | Depends |
| `Cell<T>` | Interior mut | No |
| `RefCell<T>` | Interior mut | No |
| `Mutex<T>` | Interior mut | Yes |
| `RwLock<T>` | Interior mut | Yes |

```sigil
// Heap allocation
let boxed: Box<[i32]> = Box·new([1, 2, 3])

// Reference counting
let shared: Rc<Config> = Rc·new(config)
let clone = shared|Rc·clone

// Atomic reference counting
let atomic: Arc<State> = Arc·new(state)
```

---

## 6. Function Types

### 6.1 Function Signatures

```sigil
// Basic function type
type Predicate = fn(i32) -> bool
type BinOp = fn(i32, i32) -> i32

// With evidentiality
type Fetcher = fn(str) -> Data~
type Parser = fn(str~) -> Result<T!, ParseError~>

// Generic function type
type Mapper<T, U> = fn(T) -> U
type Transformer<T!> = fn(T!) -> T!
```

### 6.2 Closures

```sigil
// Closure traits (like Rust)
trait Fn<Args>: FnMut<Args> {
    fn call(&self, args: Args) -> Self·Output
}

trait FnMut<Args>: FnOnce<Args> {
    fn call_mut(&mut self, args: Args) -> Self·Output
}

trait FnOnce<Args> {
    type Output
    fn call_once(self, args: Args) -> Self·Output
}

// Closure syntax
let add = |a, b| a + b                    // Fn
let mut counter = || { count += 1 }       // FnMut
let consume = || drop(owned_value)        // FnOnce
```

---

## 7. Generics

### 7.1 Type Parameters

```sigil
// Generic struct
struct Container<T> {
    items: [T],
}

// Generic function
fn identity<T>(x: T) -> T { x }

// Multiple parameters
fn zip<A, B>(a: [A], b: [B]) -> [(A, B)] {
    // ...
}
```

### 7.2 Trait Bounds

```sigil
// Single bound
fn print<T: Display>(value: T) {
    print!("{value}")
}

// Multiple bounds
fn process<T: Clone + Debug + Send>(value: T) {
    // ...
}

// Where clauses
fn complex<T, U>(t: T, u: U) -> impl Display
where
    T: Iterator<Item = U>,
    U: Display + Clone,
{
    // ...
}
```

### 7.3 Evidence Bounds

```sigil
// Require specific evidentiality
fn process_known<T!>(value: T!) -> T! {
    // value is guaranteed to be known
}

fn handle_uncertain<T?>(value: T?) -> T! {
    value? else { panic!("required") }
}

// Evidence-polymorphic
fn transform<T, E: Evidence>(value: T^E) -> T^E {
    // preserves whatever evidence level was passed
}
```

### 7.4 Associated Types

```sigil
trait Iterator {
    type Item
    type Evidence: Evidence = Known  // default

    fn next(&mut self) -> Option<Self·Item>^Self·Evidence
}

impl Iterator for DatabaseCursor {
    type Item = Row
    type Evidence = Reported  // database rows are external

    fn next(&mut self) -> Option<Row>~ {
        self.fetch_next()~
    }
}
```

---

## 8. Type Inference

### 8.1 Local Inference

```sigil
// Type inferred from initializer
let x = 42          // i32
let y = 3.14        // f64
let z = "hello"     // &str
let v = [1, 2, 3]   // [i32; 3]

// Evidentiality inferred from source
let a = compute()   // T! if compute returns T!
let b = lookup(k)   // T? if lookup returns T?
let c = fetch(url)  // T~ if fetch returns T~
```

### 8.2 Bidirectional Inference

```sigil
// Expected type guides inference
let numbers: [i64] = [1, 2, 3]  // literals become i64

// Return type inference
fn make_pair<T>(x: T) -> (T, T) {
    (x, x)  // inferred from signature
}

// Closure parameter inference
[1, 2, 3]|τ{x => x * 2}  // x inferred as i32
```

### 8.3 Evidence Inference

```sigil
// Evidence flows through expressions
let a! = 10
let b! = 20
let c! = a! + b!  // Known + Known = Known (inferred)

let x? = lookup(key)
let y! = 5
let z? = x? + y!  // Uncertain + Known = Uncertain (inferred)

// Function return evidence inferred
fn process(data: [i32]!) -> i32! {
    data|ρ+  // reduction of known array is known
}
```

---

## 9. Subtyping and Variance

### 9.1 Lifetime Subtyping

```sigil
// 'a: 'b means 'a outlives 'b
// &'a T is a subtype of &'b T when 'a: 'b

fn longest<'a>(x: &'a str, y: &'a str) -> &'a str {
    if x.len() > y.len() { x } else { y }
}
```

### 9.2 Evidence Subtyping

Evidentiality has subtyping relationships:

```sigil
// T! <: T?  (Known is subtype of Uncertain)
// T! <: T~  (Known is subtype of Reported)
// T? <: T‽  (Uncertain is subtype of Paradox)

fn accept_uncertain(value: i32?) { }

let known!: i32 = 42
accept_uncertain(known!)  // OK: Known <: Uncertain
```

### 9.3 Variance

| Type | Variance in T |
|------|---------------|
| `T` | Covariant |
| `&T` | Covariant |
| `&mut T` | Invariant |
| `fn(T)` | Contravariant |
| `fn() -> T` | Covariant |
| `[T]` | Covariant |
| `T!` | Covariant |
| `T?` | Covariant |
| `T~` | Covariant |

---

## 10. Special Types

### 10.1 Never Type

```sigil
// ! or ⊥ represents computations that never complete
fn diverge() -> ! {
    loop { }
}

fn panic(msg: str) -> ! {
    // ...
}

// Coerces to any type
let x: i32 = if condition { 42 } else { panic!("oops") }
```

### 10.2 Any Type

```sigil
// ⊤ represents any type (for type-erased contexts)
fn accept_any(value: &dyn ⊤) { }

// Used in trait objects
let objects: [&dyn Display] = [&1, &"hello", &true]
```

### 10.3 Self Type

```sigil
trait Clone {
    fn clone(&self) -> Self
}

impl Clone for Point {
    fn clone(&self) -> Point {  // Self = Point
        Point { x: self.x, y: self.y }
    }
}
```

---

## 11. Type Aliases

```sigil
// Simple alias
type UserId = u64
type UserMap = HashMap<UserId, User>

// Generic alias
type Result<T> = std·result·Result<T!, Error~>
type Handler<T> = fn(Request~) -> Response<T>!

// Evidence alias
type Trusted<T> = T!
type External<T> = T~
type Maybe<T> = T?
```

---

## 12. Complete Type Grammar

```ebnf
type          = primitive
              | composite
              | reference
              | function
              | generic
              | path
              | type evidence ;

evidence      = "!" | "?" | "~" | "‽" ;

primitive     = int_type | float_type | "bool" | "char" | "str" | "()" | "!" ;
int_type      = "i8" | "i16" | "i32" | "i64" | "i128" | "isize"
              | "u8" | "u16" | "u32" | "u64" | "u128" | "usize" ;
float_type    = "f32" | "f64" ;

composite     = array | tuple | struct | enum ;
array         = "[" type (";" expr)? "]" ;
tuple         = "(" (type ("," type)*)? ")" ;

reference     = "&" lifetime? "mut"? type ;
lifetime      = "'" IDENT ;

function      = "fn" "(" types? ")" "->" type ;
types         = type ("," type)* ;

generic       = path "<" type_args ">" ;
type_args     = type_arg ("," type_arg)* ;
type_arg      = type | lifetime | evidence ;

path          = IDENT ("·" IDENT)* ;
```
