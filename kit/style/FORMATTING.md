# Sigil Code Formatting Guide

This document defines the canonical formatting style for Sigil code.

---

## Quick Reference

```sigil
// Prefer native symbols
λ function_name(param: Type) → ReturnType {
    ≔ local_var = value;
    // 4-space indentation
    // Spaces around operators
}
```

---

## Symbols

**Prefer native symbols** in new code:

| Preferred | Acceptable | Use |
|-----------|------------|-----|
| `λ` | `fn` | Functions |
| `Σ` | `struct` | Structs |
| `≔` | `let` | Bindings |
| `·` | `::` | Paths |
| `☉` | `pub` | Visibility |
| `⊢` | `impl` | Implementations |
| `→` | `->` | Return types |
| `&Δ` | `&mut` | Mutable refs |
| `∀` | `for` | Loops |
| `∈` | `in` | Iteration |

---

## Indentation

- **4 spaces**, never tabs
- Configure your editor to convert tabs to spaces

```sigil
λ example() {
    if condition {
        ≔ x = 1;
        if nested {
            ≔ y = 2;
        }
    }
}
```

---

## Line Length

- **100 characters** soft limit
- **120 characters** hard limit
- Break long lines at logical points

```sigil
// Good: broken at method chain
≔ result = collection
    .iter()
    .filter(|x| x.is_valid())
    .map(|x| x.transform())
    .collect();

// Good: broken at parameters
λ long_function_name(
    first_parameter: FirstType,
    second_parameter: SecondType,
    third_parameter: ThirdType,
) → ReturnType {
    // body
}
```

---

## Spacing

### Around Operators

```sigil
// Yes
≔ sum = a + b;
≔ product = x * y;
≔ check = a == b && c != d;

// No
≔ sum=a+b;
≔ product=x*y;
```

### After Commas

```sigil
// Yes
λ f(a: i32, b: i32, c: i32)
≔ arr = [1, 2, 3, 4];

// No
λ f(a: i32,b: i32,c: i32)
```

### No Space Before Colons

```sigil
// Yes
≔ x: i32 = 5;
Σ Point { x: f64, y: f64 }

// No
≔ x : i32 = 5;
```

### Around Braces

```sigil
// Yes
if condition {
    body
}

// No
if condition{
    body
}
```

---

## Braces

Always use **K&R style** (opening brace on same line):

```sigil
// Yes
λ function() {
    if condition {
        // body
    } else {
        // else body
    }
}

// No
λ function()
{
    if condition
    {
        // body
    }
}
```

---

## Naming Conventions

| Item | Convention | Example |
|------|------------|---------|
| Functions | snake_case | `calculate_total` |
| Variables | snake_case | `user_count` |
| Structs | PascalCase | `HttpClient` |
| Traits | PascalCase | `Drawable` |
| Enums | PascalCase | `Status` |
| Enum variants | PascalCase | `Status·Active` |
| Constants | SCREAMING_SNAKE | `MAX_RETRIES` |
| Type parameters | Single uppercase | `T`, `E`, `K`, `V` |

---

## Imports

Group imports in order:
1. Standard library
2. External crates
3. Local modules

```sigil
use std·collections·HashMap;
use std·io·{Read, Write};

use external·library·Thing;

use crate·module·LocalItem;
```

---

## Comments

### Line Comments

```sigil
// This is a comment
≔ x = 5;  // Inline comment (2 spaces before //)
```

### Doc Comments

Use `///` for documentation:

```sigil
/// Calculates the factorial of n.
///
/// # Arguments
/// * `n` - The number to calculate factorial for
///
/// # Returns
/// The factorial of n, or None if n is negative
///
/// # Examples
/// ```
/// assert_eq!(factorial(5), Some(120));
/// ```
λ factorial(n: i32) → Option<i64> {
    // implementation
}
```

### Module Documentation

Use `//!` at the top of files:

```sigil
//! # Math Utilities
//!
//! This module provides mathematical helper functions.
//!
//! ## Features
//! - Factorial calculation
//! - Prime number checking
```

---

## Struct Formatting

### Small Structs (3 or fewer fields)

```sigil
Σ Point { x: f64, y: f64 }
```

### Larger Structs

```sigil
Σ User {
    id: i64,
    name: String,
    email: String,
    created_at: DateTime,
    active: bool,
}
```

---

## Match Expressions

```sigil
// Single-line arms for simple expressions
match value {
    Pattern1 => result1,
    Pattern2 => result2,
    _ => default,
}

// Multi-line arms for complex expressions
match value {
    Pattern1 => {
        ≔ x = calculate();
        transform(x)
    },
    Pattern2 => {
        log("Pattern2 matched");
        other_result
    },
}
```

---

## Evidence Markers

Place evidence markers immediately after values:

```sigil
// Yes
≔ computed = calculate()!;
≔ user_input = read()?;
≔ api_data = fetch()~;

// No
≔ computed = calculate() !;
≔ user_input = read() ?;
```

---

## Method Chains

Break long chains with each method on its own line:

```sigil
≔ result = items
    .iter()
    .filter(|x| x.is_valid())
    .map(|x| x.value)
    .sum();
```

---

## Closures

```sigil
// Short closures on one line
≔ double = |x| x * 2;

// Longer closures with braces
≔ process = |item| {
    ≔ validated = validate(item);
    transform(validated)
};
```

---

## Automatic Formatting

Use `sigil fmt` to automatically format code:

```bash
# Format single file
sigil fmt file.sg

# Format and overwrite
sigil fmt file.sg --write

# Check formatting (CI mode)
sigil fmt file.sg --check

# Format directory
sigil fmt src/
```

---

## Editor Configuration

### .editorconfig

```ini
root = true

[*.sg]
charset = utf-8
end_of_line = lf
insert_final_newline = true
indent_style = space
indent_size = 4
max_line_length = 100
trim_trailing_whitespace = true
```

### VS Code

```json
{
  "[sigil]": {
    "editor.tabSize": 4,
    "editor.insertSpaces": true,
    "editor.formatOnSave": true,
    "editor.rulers": [100, 120]
  }
}
```

---

## Summary

1. **Native symbols** preferred (`λ`, `Σ`, `≔`, etc.)
2. **4-space indentation**
3. **100 char soft limit**, 120 hard limit
4. **K&R braces** (opening on same line)
5. **snake_case** functions/variables, **PascalCase** types
6. **Spaces around operators**, after commas
7. **Evidence markers** attached to values
8. Run `sigil fmt` before committing
