# Getting Started with Sigil

This guide will have you writing and running Sigil programs in 5 minutes.

---

## Prerequisites

- Sigil compiler installed (run `./install.sh` from kit root)
- Terminal access
- Text editor with UTF-8 support

---

## Your First Program

Create a file called `hello.sg`:

```sigil
λ main() {
    println("Hello, Sigil!");
}
```

Run it:

```bash
sigil run hello.sg
```

Output:
```
Hello, Sigil!
```

---

## Variables and Types

Sigil uses `≔` for variable binding (you can also use `let`):

```sigil
λ main() {
    // Immutable binding
    ≔ name = "Alice";
    ≔ age = 30;
    ≔ pi = 3.14159;
    ≔ active = true;

    println("Name: " + name);
    println("Age: " + str(age));
}
```

### Type Annotations

Types are usually inferred, but you can be explicit:

```sigil
≔ count: i32 = 42;
≔ message: String = "Hello";
≔ values: [i32] = [1, 2, 3];
```

### Basic Types

| Type | Description | Example |
|------|-------------|---------|
| `i32`, `i64` | Integers | `42`, `-17` |
| `f32`, `f64` | Floats | `3.14`, `-0.5` |
| `bool` | Boolean | `true`, `false` |
| `String` | Text | `"hello"` |
| `char` | Character | `'a'` |
| `[T]` | Array | `[1, 2, 3]` |

---

## Functions

Functions are declared with `λ` (or `fn`):

```sigil
λ add(a: i32, b: i32) → i32 {
    a + b  // Last expression is returned
}

λ greet(name: &str) {
    println("Hello, " + name + "!");
}

λ main() {
    ≔ sum = add(3, 4);
    println("Sum: " + str(sum));
    greet("World");
}
```

### Return Types

Use `→` (or `->`) for return types:

```sigil
λ square(x: i32) → i32 {
    x * x
}

λ is_even(n: i32) → bool {
    n % 2 == 0
}
```

---

## Structs

Define data structures with `Σ` (or `struct`):

```sigil
Σ Point {
    x: f64,
    y: f64,
}

⊢ Point {
    λ new(x: f64, y: f64) → This {
        Point { x, y }
    }

    λ distance(&this, other: &Point) → f64 {
        ≔ dx = this.x - other.x;
        ≔ dy = this.y - other.y;
        (dx * dx + dy * dy).sqrt()
    }
}

λ main() {
    ≔ p1 = Point·new(0.0, 0.0);
    ≔ p2 = Point·new(3.0, 4.0);
    println("Distance: " + str(p1.distance(&p2)));  // 5.0
}
```

---

## Control Flow

### If Expressions

```sigil
λ abs(x: i32) → i32 {
    if x < 0 {
        -x
    } else {
        x
    }
}
```

### Loops

```sigil
// While loop
≔ i! = 0;
while i < 5 {
    println(str(i));
    i = i + 1;
}

// For loop with range
∀ i ∈ 0..5 {
    println(str(i));
}

// For loop over array
≔ items = ["a", "b", "c"];
∀ item ∈ items {
    println(item);
}
```

---

## Enums and Pattern Matching

```sigil
Σ enum Status {
    Active,
    Inactive,
    Pending(String),
}

λ describe(s: Status) → String {
    match s {
        Status·Active => "Currently active",
        Status·Inactive => "Not active",
        Status·Pending(reason) => "Pending: " + reason,
    }
}

λ main() {
    ≔ s = Status·Pending("Awaiting approval");
    println(describe(s));
}
```

---

## Option and Result

Sigil uses `Option` and `Result` for safe error handling:

```sigil
λ find_user(id: i32) → Option<User> {
    if id == 1 {
        Some(User { name: "Alice" })
    } else {
        None
    }
}

λ parse_number(s: &str) → Result<i32, String> {
    // Returns Ok(n) or Err(message)
}

λ main() {
    match find_user(1) {
        Some(user) => println("Found: " + user.name),
        None => println("User not found"),
    }
}
```

---

## Evidence Markers

Sigil tracks data certainty with evidence markers:

```sigil
λ main() {
    // ! = Known (verified/computed)
    ≔ computed = 2 + 2!;

    // ? = Uncertain (needs validation)
    ≔ user_input = read_line()?;

    // ~ = Reported (external data)
    ≔ api_response = http_get("https://api.example.com")~;
}
```

The compiler tracks these through transformations, helping you reason about data provenance.

---

## Compiling to Native

For production, compile to a native binary:

```bash
# Basic compilation
sigil compile hello.sg -o hello

# With optimizations
sigil compile hello.sg -o hello --release

# With LTO (link-time optimization)
sigil compile hello.sg -o hello --lto

# Run the binary
./hello
```

---

## Next Steps

1. **Explore examples:** Work through `examples/` in order
2. **Learn the language:** Read `LANGUAGE-GUIDE.md` for comprehensive reference
3. **Understand symbols:** See `NATIVE-SYNTAX.md` for the full symbol table
4. **Use tooling:** Check `TOOLING.md` for LSP, formatter, and linter setup

---

## Common Commands

```bash
sigil run file.sg              # Run with interpreter
sigil compile file.sg -o out   # Compile to binary
sigil check file.sg            # Type check only
sigil fmt file.sg              # Format code
sigil lint file.sg             # Run linter
sigil repl                     # Interactive REPL
sigil --help                   # All commands
```

---

## Troubleshooting

**"command not found: sigil"**
- Run `./install.sh` from the kit directory
- Or add `kit/bin` to your PATH manually

**Unicode symbols not displaying**
- Ensure your terminal supports UTF-8
- Use a font with good Unicode coverage (Fira Code, JetBrains Mono)

**Parse errors with symbols**
- Both `λ` and `fn` are valid - use what your editor supports
- See `NATIVE-SYNTAX.md` for ASCII equivalents
