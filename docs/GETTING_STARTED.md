# Getting Started with Sigil

A practical guide to writing your first Sigil programs.

## Installation

### From Source

```bash
# Clone the repository
git clone https://github.com/daemoniorum/sigil.git
cd sigil

# Build all tools
cargo build --release

# Add to PATH (adjust for your shell)
export PATH="$PATH:$(pwd)/target/release"
```

### VS Code Extension

1. Open VS Code
2. Go to Extensions (Ctrl+Shift+X)
3. Search for "Sigil Language"
4. Install and reload

## Your First Program

Create a file `hello.sigil`:

```sigil
fn main() {
    println("Hello, Sigil!")
}
```

Run it:

```bash
sigil run hello.sigil
```

## Basic Syntax

### Variables and Types

```sigil
// Immutable by default
let x = 42
let name = "Sigil"

// Mutable with 'mut'
let mut counter = 0
counter = counter + 1

// Type annotations
let age: i32 = 25
let pi: f64 = 3.14159
```

### Functions

```sigil
fn add(a: i32, b: i32) -> i32 {
    a + b  // Implicit return
}

fn greet(name: str) {
    println("Hello, " ++ name ++ "!")
}
```

### Control Flow

```sigil
// If expressions
let max = if a > b { a } else { b }

// Match expressions
match value {
    0 => "zero",
    1 => "one",
    n if n < 0 => "negative",
    _ => "other",
}

// Loops
for i in 0..10 {
    println(i)
}

while condition {
    // ...
}
```

## Data Transformations with Morphemes

Sigil's power comes from its morphemes - transformation operators that compose into pipelines.

### Map with tau

```sigil
let doubled = numbers|tau{_ * 2}
// [1, 2, 3] -> [2, 4, 6]
```

### Filter with phi

```sigil
let positives = numbers|phi{_ > 0}
// [1, -2, 3, -4] -> [1, 3]
```

### Sort with sigma

```sigil
let sorted = items|sigma
// [3, 1, 2] -> [1, 2, 3]

// Sort by field
let by_name = users|sigma.name
```

### Reduce with rho

```sigil
let sum = numbers|rho+       // Sum: 1 + 2 + 3 = 6
let product = numbers|rho*   // Product: 1 * 2 * 3 = 6
```

### Chaining Transformations

```sigil
let result = data
    |phi{_.age > 18}      // Filter adults
    |tau{_.name}          // Extract names
    |sigma                // Sort alphabetically
    |Sigma                // Count total
```

## ASCII Alternatives

Prefer ASCII? All morphemes have equivalent keywords:

| Greek | ASCII   | Purpose      |
|-------|---------|--------------|
| tau     | map     | Transform    |
| phi     | filter  | Select       |
| sigma     | sort    | Order        |
| rho     | reduce  | Fold         |
| lambda     | lambda  | Anonymous fn |

```sigil
// These are equivalent:
let a = data|tau{_ * 2}
let b = data|map{_ * 2}
```

## Evidentiality Types

Sigil tracks where data comes from and how certain we are about it.

### The Four Evidentiality Markers

```sigil
// ! - Known: Directly computed, verified truth
let computed! = 1 + 1  // We know this is 2

// ? - Uncertain: Might be absent
let found? = map.get(key)  // Might not exist

// ~ - Reported: From external source (untrusted)
let data~ = api.fetch(url)  // External, needs validation

// interrobang - Paradox: Trust boundary crossing
let trusted_ptr = unsafe { raw_ptr }
```

### Why This Matters

```sigil
// This won't compile - reported data is untrusted
fn process_user(data~: UserData) {
    database.insert(data)  // ERROR: Can't use ~ data directly
}

// Fixed: Validate first
fn process_user(data~: UserData) {
    let validated! = data|validate!{
        name: non_empty,
        email: email_format,
        age: range(0, 150),
    }
    database.insert(validated!)  // OK
}
```

### Handling Uncertainty

```sigil
let result? = lookup(key)

match result {
    Some(value!) => process(value),
    None => handle_missing(),
}

// Or use the coalesce operator
let value = result? ?? default_value
```

## Structs and Enums

```sigil
struct Point {
    x: f64,
    y: f64,
}

impl Point {
    fn distance(self, other: Point) -> f64 {
        let dx = self.x - other.x
        let dy = self.y - other.y
        sqrt(dx*dx + dy*dy)
    }
}

enum Shape {
    Circle(f64),            // radius
    Rectangle(f64, f64),    // width, height
    Triangle(Point, Point, Point),
}
```

## Actors (Concurrency)

```sigil
actor Counter {
    state: i64 = 0

    on Increment(n: i64) {
        self.state += n
    }

    on GetValue() -> i64 {
        self.state
    }
}

// Usage
let counter = spawn Counter
counter <- Increment(5)
let value = counter <- GetValue()  // 5
```

## Async/Await

```sigil
async fn fetch_user(id: i64) -> User~ {
    let response~ = http.get("/users/" ++ id)|await
    response~|parse_json!
}

async fn main() {
    let user~ = fetch_user(42)|await
    let validated! = user|validate!{...}
    println(validated!)
}
```

## Project Structure

```
my_project/
├── sigil.toml          # Project manifest
├── src/
│   ├── main.sigil      # Entry point
│   └── lib.sigil       # Library code
├── tests/
│   └── test_main.sigil
└── examples/
    └── demo.sigil
```

### sigil.toml

```toml
[package]
name = "my_project"
version = "0.1.0"

[dependencies]
http = "0.3"
json = "1.0"

[dev-dependencies]
test_utils = "0.1"
```

## Tools

### Glyph (Formatter)

```bash
# Format a file
sigil-glyph src/main.sigil

# Check without modifying
sigil-glyph --check .

# Format entire project
sigil-glyph .
```

### Oracle (Language Server)

The Oracle provides IDE features:
- Real-time error checking
- Hover documentation
- Go-to-definition
- Semantic highlighting
- Completions

Already configured in the VS Code extension.

## Common Patterns

### Error Handling

```sigil
fn divide(a: f64, b: f64) -> Result<f64, str> {
    if b == 0.0 {
        Err("division by zero")
    } else {
        Ok(a / b)
    }
}

// Using the result
match divide(10.0, 2.0) {
    Ok(result!) => println(result),
    Err(msg) => println("Error: " ++ msg),
}

// Or with the ? operator
fn calculate() -> Result<f64, str> {
    let x = divide(10.0, 2.0)?
    let y = divide(x, 3.0)?
    Ok(x + y)
}
```

### Builder Pattern

```sigil
struct Request {
    url: str,
    method: str,
    headers: Map<str, str>,
    body: Option<str>,
}

impl Request {
    fn new(url: str) -> Request {
        Request { url, method: "GET", headers: Map.new(), body: None }
    }

    fn method(mut self, m: str) -> Request {
        self.method = m
        self
    }

    fn header(mut self, key: str, value: str) -> Request {
        self.headers.insert(key, value)
        self
    }

    fn body(mut self, b: str) -> Request {
        self.body = Some(b)
        self
    }
}

// Usage
let req = Request.new("https://api.example.com")
    .method("POST")
    .header("Content-Type", "application/json")
    .body("{\"key\": \"value\"}")
```

### Pipeline Processing

```sigil
// Process a CSV file
let users! = file.read("users.csv")~
    |validate!
    |lines
    |phi{!_.starts_with("#")}  // Skip comments
    |tau{_.split(",")}         // Parse CSV
    |tau{User.from_fields(_)}  // Create objects
    |phi{_.active}             // Filter active
    |sigma.last_login          // Sort by login
```

## Next Steps

- Read the [Language Specification](./SPEC.md)
- Explore [Example Programs](../examples/)
- Join the community on Discord
- Check the [API Reference](./API.md)

## Quick Reference Card

```
MORPHEMES:
  tau / map     Transform elements      data|tau{expr}
  phi / filter  Select matching         data|phi{pred}
  sigma / sort    Order elements          data|sigma or data|sigma.field
  rho / reduce  Fold to single value    data|rho+ or data|rho{acc,x => expr}
  Sigma / sum     Sum all                 data|Sigma
  Pi / product  Product all             data|Pi
  lambda        Anonymous function      lambda x -> expr

EVIDENTIALITY:
  !   Known      Computed/verified truth
  ?   Uncertain  May be absent
  ~   Reported   External/untrusted
  interrobang   Paradox    Trust boundary

OPERATORS:
  |   Pipe       data|transform
  ·   Incorporate  noun·verb·verb
  ++  Concat     str1 ++ str2
  ??  Coalesce   maybe? ?? default
  ->  Arrow      fn(x) -> result
  =>  Fat arrow  pattern => result
```
