# Sigil Agent Guide

Quick reference for AI agents working with Sigil. This document is optimized for LLM context efficiency.

---

## Commands

```bash
# Execution
sigil run file.sg                    # Interpret
sigil compile file.sg -o out         # Compile to native binary
sigil compile file.sg -o out --lto   # With link-time optimization
sigil jit file.sg                    # JIT compile and run

# Analysis
sigil check file.sg                  # Type check only
sigil fmt file.sg                    # Format code (in-place)
sigil fmt file.sg --check            # Check formatting (no modify)
sigil lint file.sg                   # Run linter

# Interactive
sigil repl                           # Start REPL
```

---

## File Extension

`.sg` or `.sigil`

---

## Syntax Quick Reference

### Symbols (Preferred)

| Symbol | ASCII | Meaning |
|--------|-------|---------|
| `λ` | `fn` | Function |
| `Σ` | `struct` | Struct |
| `≔` | `let` | Binding |
| `·` | `::` | Path separator |
| `☉` | `pub` | Public visibility |
| `⊢` | `impl` | Implementation |
| `→` | `->` | Return type arrow |
| `&Δ` | `&mut` | Mutable reference |
| `∀` | `for` | For loop |
| `∈` | `in` | In (iteration) |
| `∞` | `loop` | Infinite loop |

### Evidence Markers

| Marker | Meaning | When to Use |
|--------|---------|-------------|
| `!` | Known | Computed/verified values |
| `?` | Uncertain | User input, needs validation |
| `~` | Reported | External API data |
| `‽` | Paradox | Self-referential |

### Mutability

```sigil
≔ x = 5;           // Immutable
≔ y! = 5;          // Mutable (! suffix on binding)
y = 10;            // OK - y is mutable
```

---

## Common Patterns

### Function Definition

```sigil
λ function_name(param: Type, param2: Type) → ReturnType {
    // body
    result  // last expression returned
}

// No return value
λ print_value(x: i32) {
    println(str(x));
}
```

### Struct with Methods

```sigil
Σ MyStruct {
    field1: Type1,
    field2: Type2,
}

⊢ MyStruct {
    // Constructor (by convention)
    λ new(field1: Type1, field2: Type2) → This {
        MyStruct { field1, field2 }
    }

    // Instance method
    λ method(&this) → ReturnType {
        this.field1
    }

    // Mutable method
    λ mutate(&Δ this, value: Type1) {
        this.field1 = value;
    }
}
```

### Trait Definition and Implementation

```sigil
trait Display {
    λ display(&this) → String;
}

⊢ Display for MyStruct {
    λ display(&this) → String {
        "MyStruct { ... }"
    }
}
```

### Error Handling

```sigil
// Return Result
λ parse(s: &str) → Result<i32, String> {
    // ...
    Ok(42)
    // or
    Err("parse error")
}

// Pattern match on Result
match parse("123") {
    Ok(n) => println("Got: " + str(n)),
    Err(e) => println("Error: " + e),
}

// Return Option
λ find(id: i32) → Option<Item> {
    if found {
        Some(item)
    } else {
        None
    }
}
```

### Collections

```sigil
// Array
≔ arr = [1, 2, 3];
≔ first = arr[0];

// Vec (dynamic)
≔ vec = Vec·new();
vec.push(1);
vec.push(2);

// HashMap
≔ map = HashMap·new();
map.insert("key", "value");
```

### Control Flow

```sigil
// If expression
≔ result = if condition { a } else { b };

// Match
match value {
    Pattern1 => expression1,
    Pattern2(x) => expression2,
    _ => default,
}

// Loops
∀ item ∈ collection { }
∀ i ∈ 0..10 { }
while condition { }
∞ { if done { break; } }
```

---

## HTTP Client

```sigil
use std·http·{Client, Request};

λ main() {
    ≔ client = Client·new();

    // GET
    ≔ response = client.get("https://api.example.com/data")~;
    println(response.body);

    // POST with JSON
    ≔ response = client
        .post("https://api.example.com/data")
        .header("Content-Type", "application/json")
        .body(r#"{"key": "value"}"#)
        .send()~;
}
```

---

## File I/O

```sigil
use std·fs;

// Read file
≔ content = fs·read_to_string("path/to/file.txt")?;

// Write file
fs·write("path/to/file.txt", content)?;

// Check existence
if fs·exists("path") {
    // ...
}
```

---

## Compilation Targets

```bash
# Interpreter (fastest startup)
sigil run program.sg

# JIT (Cranelift - balanced)
sigil jit program.sg

# Native (LLVM - best performance)
sigil compile program.sg -o program

# With CUDA support
sigil compile program.sg -o program --cuda

# With SIMD (AVX-512)
# Automatically enabled when available
sigil compile program.sg -o program --release
```

---

## Project Structure Convention

```
my-project/
├── src/
│   ├── main.sg          # Entry point
│   ├── lib.sg           # Library root
│   └── modules/
│       └── module.sg
├── tests/
│   └── test_module.sg
├── Tome.toml            # Package manifest (if using tome)
└── README.md
```

---

## Common Errors and Fixes

| Error | Cause | Fix |
|-------|-------|-----|
| `type mismatch` | Wrong type passed | Check function signature |
| `unknown identifier` | Variable not in scope | Check spelling, imports |
| `borrow checker error` | Multiple mutable refs | Use single `&Δ` at a time |
| `expected !, found ?` | Evidence mismatch | Validate uncertain data first |

---

## When to Use What

| Task | Recommended Approach |
|------|---------------------|
| Quick script | `sigil run` interpreter |
| Development | `sigil run` + `sigil check` |
| Testing | `sigil run tests/` |
| Production | `sigil compile --release --lto` |
| GPU compute | `sigil compile --cuda` |

---

## Methodology Integration

When working on Sigil projects, follow these methodologies (see `methodologies/`):

1. **Spec-Driven Development**: If you discover a gap in understanding, STOP and document it before proceeding.

2. **Agent-TDD**: Write tests as executable specifications, not coverage theater.

3. **Evidence tracking**: Use appropriate markers (`!`, `?`, `~`) to track data certainty.

---

## Quick Examples

### Parse JSON and Extract Field

```sigil
use std·json;

λ get_name(json_str: &str) → Result<String, String> {
    ≔ parsed = json·parse(json_str)?;
    ≔ name = parsed["name"].as_str()?;
    Ok(name.to_string())
}
```

### Process List with Map/Filter

```sigil
λ process(items: [i32]) → [i32] {
    items
        .iter()
        .filter(|x| x > 0)
        .map(|x| x * 2)
        .collect()
}
```

### Concurrent HTTP Requests

```sigil
use std·http·Client;
use std·async·join_all;

λ async fetch_all(urls: [String]) → [Response] {
    ≔ client = Client·new();
    ≔ futures = urls.iter().map(|url| client.get(url));
    join_all(futures).await
}
```
