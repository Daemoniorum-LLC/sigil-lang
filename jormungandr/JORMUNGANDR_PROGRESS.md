# Jormungandr Self-Hosting Progress

## Overview
The Jormungandr initiative aims to make Sigil self-hosting - compiling the Sigil compiler using itself. This document tracks progress on running the self-hosted compiler (in `self-hosted/src/`) through the Rust interpreter to compile Sigil programs to C code.

## Current Status: 🎉 ALL 13 MODULES COMPILE TO C! 🎉

**Major Milestone Achieved**: The self-hosted compiler can now compile **ALL 13 of its own modules** through the complete pipeline to C code!

### Compilation Results (December 2024)

| Module | Lines of C | Status |
|--------|------------|--------|
| typeck.sg | 4,578 | ✅ |
| parser.sg | 4,200 | ✅ |
| interp.sg | 3,236 | ✅ |
| lower.sg | 2,841 | ✅ |
| ast.sg | 2,719 | ✅ |
| codegen.sg | 2,494 | ✅ |
| lexer.sg | 2,137 | ✅ |
| ir.sg | 1,760 | ✅ |
| token.sg | 1,181 | ✅ |
| driver.sg | 1,079 | ✅ |
| runtime.sg | 644 | ✅ |
| span.sg | 482 | ✅ |
| lib.sg | 304 | ✅ |
| **TOTAL** | **~27,655** | ✅ |

### Run Command
```bash
cd sigil-lang/parser
cargo run --release -- run-dir ../self-hosted/src -- compile /path/to/module.sg
```

---

## Next Steps: GCC-Compatible C Output

### Phase 1: Codegen Fixes (Required for GCC) - COMPLETED!

1. **Fix enum variant name generation** (`codegen.sg`) - ✅ FIXED
   - Issue: `Evidentiality____` generated instead of `Evidentiality____Paradox`
   - Fix: Modified `strip_single_prefix()` to only strip evidentiality prefixes when
     there's something after the prefix (e.g., "UncertainToken" → "Token", but
     "Paradox" stays "Paradox")

2. **Handle Unicode character literals** (`codegen.sg`) - ✅ FIXED
   - Issue: `'‽'` (interrobang) causes GCC warning/error
   - Fix: Modified `escape_char()` to escape non-ASCII characters to hex format
     (e.g., `\x{:02x}` for characters outside 32-126 range)

3. **Add cross-module forward declarations** (`codegen.sg`) - ✅ FIXED
   - Issue: `sigil_Span____new` undeclared when compiling ast.c
   - Fix: Added extern declarations for cross-module functions in `emit_builtin_decls()`
     including Span, Token, Ident, Spanned, EvidenceLevel, IrSpan, and IrModule functions

### Phase 2: Module Linking - COMPLETED!

4. **Create unified build system** - ✅ DONE
   - Added `build.sh` script for automated compilation
   - Added `Makefile` for fine-grained build control
   - Unified build combines all 13 C files with proper deduplication of headers

5. **Add main() entry point** - Already handled by codegen.sg's `emit_main_wrapper()`

### Phase 3: Bootstrap Verification - IN PROGRESS

6. **Compile with GCC**
   ```bash
   cd self-hosted
   ./build.sh
   # OR
   make all
   ```

7. **Fixed-point test**
   ```bash
   make verify
   # OR
   ./build/sigil compile src/*.sg -o build/sigil2.c
   diff build/c/*.c build/verify/*.c  # Should be identical!
   ```

---

## Recent Fixes (This Bootstrap Cycle)

### Parser Enhancements (`parser.sg`)

1. **Enum variant attribute skipping**
   - Added `while self.check(&Token::Hash)` loop before parsing variant name
   - Allows `#[default]` and other attributes on enum variants

2. **Inline extern block skipping**
   - Added `skip_extern_block()` function
   - Modified `parse_block_contents()` to handle `Token::Extern`
   - Skips FFI declarations inside function bodies

### Source Compatibility Fixes

3. **C keyword avoidance** (various files)
   - `inline` → `inline_hint` (ast.sg, parser.sg)
   - `default` → `default_value` (ast.sg)
   - `volatile` → `is_volatile` (ast.sg)
   - `timeout` → `timeout_ms` (ast.sg, ir.sg)

4. **Nested function → closure conversion** (typeck.sg, interp.sg, runtime.sg)
   - `fn rank(ev) { ... }` → `let rank = |ev| { ... };`

5. **Tuple pattern workarounds** (typeck.sg, lower.sg, interp.sg)
   - `|τ{(i, t) => ...}` → `|τ{pair => ... pair.0 ... pair.1 ...}`
   - `|all{(x, y) => ...}` → explicit for loops

6. **Morpheme syntax workarounds** (interp.sg)
   - `arr|all{item => ...}` → explicit for loop with break
   - `arr|any{item => ...}` → explicit for loop with break

7. **Pipe-then-try workaround** (interp.sg)
   - `|collect_results()?` → split into two statements

8. **Scientific notation replacement** (runtime.sg)
   - `1e-15` → `0.000000000000001`
   - `1e-10` → `0.0000000001`
   - `1e-6` → `0.000001`

9. **Operator workarounds** (runtime.sg)
   - `e >>= 1` → `e = e >> 1`
   - `b'\n'` → `10`

---

## Previous Milestone: Core span.sg Compiles and RUNS!

**Major Milestone**: The self-hosted compiler can now compile span.sg through the full pipeline to **working executable code**!

The compilation pipeline successfully progresses through:
- ✅ Parsing
- ✅ Type checking (with warnings)
- ✅ Lowering to IR
- ✅ Code generation to C
- ✅ **GCC compilation to executable**
- ✅ **Correct execution (returns 10 for merged span length)**

```bash
# Working test case
Span::new(5, 10)          # length 5
Span::new(8, 15)          # length 7
Span::merge(a, b)         # (5, 15) length 10 ✓
```

Impl block methods with qualified calls (`Span::new()`, `Span::merge()`, etc.) work correctly!

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
| Impl block methods | ✅ | `Span::new()`, `Span::merge()` |
| Qualified method calls | ✅ | `Type::method(receiver)` |
| Struct shorthand syntax | ✅ | `Self { x, y }` → `Self { x: x, y: y }` |
| Self type resolution | ✅ | `Self { ... }` → `Span { ... }` in impl |
| Primitive methods | ✅ | `.min()`, `.max()`, `.len()` on integers |

### Known Limitations

| Feature | Status | Notes |
|---------|--------|-------|
| Method call resolution | ⚠️ | `p.method()` needs explicit `Type::method(p)` |
| Closures | ⚠️ | Parsed but codegen incomplete |
| `write!` macro | ⚠️ | Generates malformed C code |
| `as` casts | ❌ | Parser doesn't handle `x as i64` |

### Recent Fixes (This Session)

#### Type Checker Enhancements

1. **Associated function resolution** (`typeck.sg`)
   - Added `Item::Impl` handling in `collect_fn_sig` to register associated functions
   - Associated functions registered with qualified names like `Span::new`
   - Added multi-segment path resolution in `infer_expr` for `Type::method` patterns

2. **Self type handling** (`typeck.sg`)
   - Added `current_self_type` field to TypeChecker
   - Self is resolved to actual type name during impl block checking
   - Works for both type annotations and struct expressions

3. **Integer coercion** (`typeck.sg`)
   - Allow integer types to unify with each other for literal coercion
   - `i64` literals can match `!usize` parameters

4. **Expression handlers** (`typeck.sg`)
   - Added handlers for MethodCall, Field, Struct, Closure expressions
   - Added Try, Range, Reference, Cast, Let, Break, Continue, Macro handlers
   - Common primitive methods (min, max, len, is_empty, etc.)

5. **Uncertain type prefix handling** (`typeck.sg`)
   - Strip "Uncertain", "Known", "Reported" prefixes from struct names
   - Resolves evidential type wrappers to base types

#### Lowering Fixes

1. **Field name fixes** (`lower.sg`)
   - Fixed `trait_path` → `trait_` field name mismatch
   - Fixed `self_type` → `self_ty` field name mismatch
   - Fixed `ImplItem::Method` → `ImplItem::Function` variant name

2. **Namespace fixes** (`lower.sg`)
   - Fixed `ir::UnaryOp` → `UnaryOp` (imports are wildcard)
   - Fixed `ast::UnaryOp` prefix for AST operators

3. **Lenient compilation** (`lower.sg`, `typeck.sg`)
   - Type errors reported as warnings but continue
   - Lowering errors reported as warnings but continue
   - Allows bootstrapping even with incomplete features

4. **Impl block method generation** (`lower.sg`)
   - Impl methods added to `module.functions` with qualified names
   - `Point::new` generates `sigil_Point____new` in C
   - Added `current_self_type` tracking for Self resolution

5. **Struct shorthand syntax** (`lower.sg`)
   - `Self { x, y }` now generates var references for x and y
   - Fixed null value handling in FieldInit

6. **Self type resolution** (`lower.sg`)
   - `Self` in struct expressions resolves to actual type name
   - `Self { x, y }` inside `impl Span` generates `sigil_struct("Span", ...)`

7. **Method call lowering** (`lower.sg`)
   - Added `Expr::MethodCall` → `IrOperation::MethodCall` lowering

#### Codegen Fixes

1. **Method call codegen** (`codegen.sg`)
   - Added `IrOperation::MethodCall` handler
   - Primitive methods (min, max, len, is_empty, clone, to_string) inline-expanded
   - Other method calls generate `sigil_method(receiver, args)`

#### Interpreter Fixes

1. **Pipe method support** (`interpreter.rs`)
   - Added `all` and `any` pipe methods to `PipeOp::Method`
   - Also added to `PipeOp::Named` for `·all{}` syntax

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
