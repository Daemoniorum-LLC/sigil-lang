# Jormungandr Self-Hosting Progress

## Overview
The Jormungandr initiative aims to make Sigil self-hosting - compiling the Sigil compiler using itself. This document tracks progress on running the self-hosted compiler (in `self-hosted/src/`) through the Rust interpreter to compile Sigil programs to C code.

## Current Status: span.sg Parsing Success!

**Major Milestone**: The self-hosted parser can now successfully parse span.sg!

The compilation pipeline progresses through parsing → type checking, but fails during type checking due to missing associated function resolution (e.g., `Span::new()`).

### What's Working

| Feature | Status | Example |
|---------|--------|---------|
| Integer literals | ✅ | `42` → 42 |
| Let bindings | ✅ | `let x = 5; x * 2` → 10 |
| Function calls | ✅ | `add(1, 2)` → 3 |
| While loops | ✅ | `while x<5 { x=x+1 }; x` → 5 |
| For loops | ✅ | Lowering implemented |
| Assignments | ✅ | `x = x + 1` |
| Factorial | ✅ | `factorial(5)` → 120 |
| If-let patterns | ✅ | `if let Some(x) = opt { ... }` |
| Struct lowering | ✅ | `Point { x: 10, y: 32 }` lowers to IR |
| Struct codegen | ✅ | Compiles to working C code |
| Struct field access | ✅ | `p.x + p.y` → 42 |
| #[derive(...)] attrs | ✅ | `#[derive(Debug, Clone)]` |
| Prefix evidentiality | ✅ | `!usize`, `?T` |
| Closure parsing | ✅ | `{x => x * 2}` |
| Generic types | ✅ | `Spanned<T>` |
| Lifetime markers | ✅ | `Token::Quote` added |

### Recent Fixes (This Session)

#### Parser Fixes for span.sg

1. **#[derive(...)] attribute support** (`parser.sg`)
   - Added `Token::Derive`, `Token::Test`, `Token::Default` as valid identifiers in `parse_ident()`
   - These keywords can now be used as attribute names

2. **Prefix evidentiality parsing** (`parser.sg`)
   - Fixed `parse_type_primary()` to handle `!T` as prefix evidentiality
   - Previously `!` alone was parsed as Never type
   - Now `!usize` correctly parses as `Known<usize>`

3. **Expression path vs generics** (`parser.sg`)
   - Added `parse_expr_path()` that doesn't interpret `<` as generics
   - Fixed `parse_primary_expr()` to use `parse_expr_path()` for identifiers
   - Now `offset < self.end` correctly parses as comparison, not generics

4. **Type suffix evidentiality** (`parser.sg`)
   - Fixed `parse_type()` to only wrap in `TypeExpr::Evidential` when marker present
   - Previously `if let ev = parse_evidentiality_opt()` always matched

5. **Token::Quote for lifetimes** (`token.sg`, `lexer.sg`)
   - Added `Token::Quote` variant for `'` character
   - Added `lex_quote_or_char()` to distinguish `'a'` (char) from `'a` (lifetime)

6. **Closure parsing** (`parser.sg`, `lexer.sg`)
   - Implemented `is_closure_start()` properly (was always returning false)
   - Added `Lexer::peek_is_closure_indicator()` helper
   - Now `{x => x * 2}` correctly parses as a closure

7. **&str method support** (`interpreter.rs`)
   - Added `to_string`, `len`, `is_empty`, `as_str` methods for `Value::Ref` containing String

#### Previous Fixes

1. **Tuple field access in format! macro** (`parser.rs`)
   - Fixed string literal escaping in `parse_macro_tokens`
   - Internal quotes in strings are now properly escaped with `\"`
   - Before: `format!("\"{}\"", x)` → `""{}""` (broken)
   - After: `format!("\"{}\"", x)` → `"\"{}\""` (correct)

2. **Duplicate Field codegen handler removed** (`codegen.sg`)
   - Removed incorrect `IrOperation::Field` handler that generated `p.x` syntax
   - Kept correct handler that generates `sigil_struct_field(p, "x")`

3. **Forward declarations for struct operations** (`codegen.sg`)
   - Added `sigil_struct()` declaration to `emit_builtin_decls()`
   - Added `sigil_struct_field()` declaration to `emit_builtin_decls()`

4. **If-let pattern matching semantics** (`interpreter.rs`)
   - `Expr::Let` now checks if pattern matches before binding variables
   - Returns `Bool(true/false)` for if-let condition evaluation
   - Fixed "Undefined variable: f" errors with `if let Item::Function(f) = item`

5. **Evidence level null handling** (`typeck.sg`)
   - Added wildcard arm to `EvidenceLevel::from_ast` to handle null values
   - Prevents "No matching pattern for null" errors

6. **Struct expression support** (`lower.sg`)
   - Added `Expr::Struct` → `IrOperation::StructInit` lowering
   - Extracts struct name from path, lowers fields and rest expression

7. **Codegen for new operations** (`codegen.sg`)
   - `IrOperation::StructInit` - struct initialization with runtime helpers
   - `IrOperation::Field` - struct field access via `sigil_struct_field()`
   - `IrOperation::Index` - array/tuple indexing
   - `IrOperation::Tuple` - tuple creation
   - `IrOperation::Array` - array creation

8. **Runtime struct helpers** (`codegen.sg`)
   - `SigilStruct` typedef for runtime struct representation
   - `sigil_struct()` - creates struct values with field names/values
   - `sigil_struct_field()` - accesses struct fields by name

### Root Cause of Tuple Field Access Bug

The bug was in the parser's `parse_macro_tokens` function. When serializing `Token::StringLit(s)` to the token string, it used:
```rust
Token::StringLit(s) => format!("\"{}\"", s),
```

This didn't escape internal quotes. For Sigil code like:
```sigil
format!("\"{}\"", field_name)
```

The string literal content is `"{}"` (quote, braces, quote). When serialized without escaping, this became `""{}""` in the token string. The `eval_format_macro` function then parsed `""` as the format string (empty!) and `{}` as an argument.

**Fix**: Escape internal quotes and backslashes when serializing string literals:
```rust
Token::StringLit(s) => format!("\"{}\"", s.replace('\\', "\\\\").replace('"', "\\\""))
```

## Test Commands

```bash
# Run simple test
cargo run --release -- run-dir ../self-hosted/src -- compile /tmp/test.sg

# Test struct (WORKING!)
echo 'struct Point { x: i64, y: i64 } fn main() -> i64 { let p = Point { x: 10, y: 32 }; p.x + p.y }' > /tmp/test_struct.sg
cargo run --release -- run-dir ../self-hosted/src -- compile /tmp/test_struct.sg 2>/dev/null > /tmp/test_struct.c
gcc -o /tmp/test_struct /tmp/test_struct.c -lm && /tmp/test_struct
# Returns: 42

# Test factorial
echo 'fn factorial(n: i64) -> i64 { if n <= 1 { 1 } else { n * factorial(n - 1) } } fn main() -> i64 { factorial(5) }' > /tmp/test.sg
```

## Next Steps

1. **Fix type checker for associated functions** - `Span::new()`, `Spanned::new()` resolution
2. **Add match codegen** - `IrOperation::Match` code generation
3. **Complete span.sg compilation** - Through type checking, lowering, and codegen
4. **Self-compile token.sg** - Next module after span.sg
5. **Self-compile full compiler** - Ultimate goal

## File Changes Summary

| File | Changes |
|------|---------|
| `parser/src/interpreter.rs` | Fixed if-let semantics; added &str method support |
| `parser/src/parser.rs` | Fixed string literal escaping in `parse_macro_tokens` |
| `self-hosted/src/typeck.sg` | Added from_ast wildcard for null |
| `self-hosted/src/lower.sg` | Added Expr::Struct lowering |
| `self-hosted/src/codegen.sg` | Added StructInit, Field, Index, Tuple, Array codegen + runtime helpers; removed duplicate Field handler; added struct forward declarations |
| `self-hosted/src/parser.sg` | Fixed prefix evidentiality, expression paths, closure detection, keyword-as-ident |
| `self-hosted/src/lexer.sg` | Added lex_quote_or_char, peek_is_closure_indicator |
| `self-hosted/src/token.sg` | Added Token::Quote variant |

## Architecture Notes

The self-hosted compiler pipeline:
1. **Lexer** (`lexer.sg`) - Tokenizes source
2. **Parser** (`parser.sg`) - Builds AST
3. **Type Checker** (`typeck.sg`) - Type inference and checking
4. **Lowering** (`lower.sg`) - AST → IR transformation
5. **Codegen** (`codegen.sg`) - IR → C code generation
6. **Runtime** (`runtime.sg`) - Runtime support structures

All executed through the Rust interpreter (`parser/src/interpreter.rs`).
