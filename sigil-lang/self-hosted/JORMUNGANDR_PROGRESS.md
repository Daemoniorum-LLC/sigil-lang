# Jormungandr Self-Hosting Progress

## Overview
The Jormungandr initiative aims to make Sigil self-hosting - compiling the Sigil compiler using itself. This document tracks progress on running the self-hosted compiler (in `self-hosted/src/`) through the Rust interpreter to compile Sigil programs to C code.

## Current Status: Struct Support (In Progress)

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
| Struct codegen | 🔄 | Field name extraction issue |

### Recent Fixes (This Session)

1. **If-let pattern matching semantics** (`interpreter.rs`)
   - `Expr::Let` now checks if pattern matches before binding variables
   - Returns `Bool(true/false)` for if-let condition evaluation
   - Fixed "Undefined variable: f" errors with `if let Item::Function(f) = item`

2. **Evidence level null handling** (`typeck.sg`)
   - Added wildcard arm to `EvidenceLevel::from_ast` to handle null values
   - Prevents "No matching pattern for null" errors

3. **Struct expression support** (`lower.sg`)
   - Added `Expr::Struct` → `IrOperation::StructInit` lowering
   - Extracts struct name from path, lowers fields and rest expression

4. **Codegen for new operations** (`codegen.sg`)
   - `IrOperation::StructInit` - struct initialization with runtime helpers
   - `IrOperation::Field` - struct field access via `sigil_struct_field()`
   - `IrOperation::Index` - array/tuple indexing
   - `IrOperation::Tuple` - tuple creation
   - `IrOperation::Array` - array creation

5. **Runtime struct helpers** (`codegen.sg`)
   - `SigilStruct` typedef for runtime struct representation
   - `sigil_struct()` - creates struct values with field names/values
   - `sigil_struct_field()` - accesses struct fields by name

### Current Blocker

**Tuple field access in codegen**: When iterating over IR `fields` array in `StructInit` codegen, accessing tuple elements via `fields[i].0` produces empty strings.

Generated C code shows the bug:
```c
static const char* _t0__names[2] = { ,  };  // Empty!
SigilValue _t0__values[2];
_t0__values[0] = sigil_int(10LL);
_t0__values[1] = sigil_int(32LL);
SigilValue p = sigil_struct(;  // Incomplete
```

The IR correctly shows `fields: [(x, IrOperation::Literal...), (y, ...)]`, but the Sigil code `fields[i].0` isn't extracting the field name.

## Test Commands

```bash
# Run simple test
cargo run --release -- run-dir ../self-hosted/src -- compile /tmp/test.sg

# Test struct
echo 'struct Point { x: i64, y: i64 } fn main() -> i64 { let p = Point { x: 10, y: 32 }; p.x + p.y }' > /tmp/test_struct.sg
cargo run --release -- run-dir ../self-hosted/src -- compile /tmp/test_struct.sg 2>/dev/null

# Test factorial
echo 'fn factorial(n: i64) -> i64 { if n <= 1 { 1 } else { n * factorial(n - 1) } } fn main() -> i64 { factorial(5) }' > /tmp/test.sg
```

## Next Steps

1. **Fix tuple field access** - Debug why `fields[i].0` produces empty strings in the self-hosted codegen context
2. **Complete struct test** - Get `Point { x: 10, y: 32 }` compiling to working C code
3. **Add match codegen** - `IrOperation::Match` code generation
4. **Test field access** - Verify `p.x + p.y` works with the struct runtime
5. **Compile with gcc** - Test generated C code compilation and execution
6. **Self-compile span.sg** - First real module self-compilation test
7. **Self-compile full compiler** - Ultimate goal

## File Changes Summary

| File | Changes |
|------|---------|
| `parser/src/interpreter.rs` | Fixed if-let semantics, added debug output |
| `self-hosted/src/typeck.sg` | Added from_ast wildcard for null |
| `self-hosted/src/lower.sg` | Added Expr::Struct lowering |
| `self-hosted/src/codegen.sg` | Added StructInit, Field, Index, Tuple, Array codegen + runtime helpers |

## Architecture Notes

The self-hosted compiler pipeline:
1. **Lexer** (`lexer.sg`) - Tokenizes source
2. **Parser** (`parser.sg`) - Builds AST
3. **Type Checker** (`typeck.sg`) - Type inference and checking
4. **Lowering** (`lower.sg`) - AST → IR transformation
5. **Codegen** (`codegen.sg`) - IR → C code generation
6. **Runtime** (`runtime.sg`) - Runtime support structures

All executed through the Rust interpreter (`parser/src/interpreter.rs`).
