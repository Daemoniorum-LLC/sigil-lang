# Jormungandr Bootstrap Compiler - Codegen Fixes

This document describes manual fixes applied to `build/sigil_bootstrap.c` to work around
codegen bugs in the Rust interpreter generating C code.

## Summary of Fixes

### 1. TAG_TUPLE / TOKEN_TYPE Collision (Critical)

**Problem**: `TAG_TUPLE = 7` collides with `TOKEN_TYPE = 7`. When the parser's `self.current`
contains a Token::Type value, checks like `_t1.tag == TAG_TUPLE` incorrectly match, causing
the code to try to extract tuple fields from a non-tuple value.

**Fix**: Added `_t1.v.tup.fields != NULL && _t1.v.tup.count == 2` checks to distinguish
real tuples from tokens that happen to have tag=7.

**Files/Functions Fixed**:
- `sigil_Parser____current_token` (line ~10519)
- `sigil_Parser____current_span` (line ~10537)
- `sigil_Parser____expect` (line ~10664)
- `sigil_Parser____expect_gt` (lines ~10704, 10708, 10713)
- `sigil_Parser____peek_next_token` (line ~10586)

### 2. Pattern Matching Generating `(1)` Instead of Conditions

**Problem**: The codegen incorrectly generates `if (1)` instead of proper pattern match
conditions for certain struct patterns like `Expr::If { .. }`.

**Functions Fixed**:

#### `sigil_Parser____is_non_callable_expr` (line ~14052)
- Changed `if ((1))` branches to proper tag checks:
  - `EXPR_IF`, `EXPR_WHILE`, `EXPR_MATCH`, `EXPR_LOOP`, `EXPR_FOR`, `EXPR_BLOCK`
- Also added TAG_REF dereference at start

#### `sigil_Lexer____lex_quote_or_char` (line ~8360)
- Changed `if (1)` to proper character range check:
  - `(_t0.tag == TAG_CHAR && ((_t0.v.c >= 'a' && _t0.v.c <= 'z') || (_t0.v.c >= 'A' && _t0.v.c <= 'Z') || _t0.v.c == '_'))`

### 3. `lex_hex_escape` For Loop Bounds

**Problem**: For loop used `sigil_null().v.arr.len` instead of `digits` parameter.

**Fix**: Changed to `(size_t)digits.v.i` to iterate correct number of times.

### 4. `sigil_len_utf8` Character Handling (Critical)

**Problem**: `sigil_len_utf8` returned 0 for character inputs, causing `Lexer::advance`
to never increment position, resulting in infinite loops.

**Fix**: Added handling for `TAG_CHAR` to return proper UTF-8 byte length (1-4).

### 5. Incompatible Type Errors (141 fixes)

**Problem**: Generated code had mismatches between `SigilValue` and `SigilValue*`:
- Functions expecting `SigilValue*` were passed `SigilValue` (need `&var`)
- Functions expecting `SigilValue` were passed `SigilValue*` (need `(*ptr)`)

**Categories Fixed**:
- Parser methods: `advance`, `peek_next_token`, `parse`, etc.
- Lexer methods: `next_token`, `peek`, `peek_is_macro_delimiter`, etc.
- TypeChecker methods: `push_scope`, `pop_scope`, `bind_pattern`, `unify`, `infer_literal`
- TypeEnv methods: `define`
- Interpreter methods: `eval_with_env`, `bind_pattern`, `call_function`, `check_evidence`
- CodeGen methods: `with_evidence`, `emit_pattern_condition`, `emit_binary_*`, `line_close`
- LoweringContext methods: `fresh_id`, `error`, `get_var_id`
- Environment methods: `define`
- Driver methods: `check`
- Helper functions: `mangle_name`, `escape_char`, `escape_string`

### 6. String Parse Method Resolution

**Problem**: Calls to `s.parse()` on strings incorrectly resolved to `Parser::parse`.

**Fix**: Added `sigil_String____parse` helper function and replaced incorrect calls.

### 7. Vec::push and String::push Result Capture

**Problem**: Methods that return new/modified values weren't having their returns captured.

**Files/Functions Fixed**:
- `sigil_Lexer____lex_ident_or_keyword`
- `sigil_Parser____parse_file`

### 8. sigil_eq Reference Comparison

**Problem**: `sigil_eq` didn't dereference TAG_REF values before comparison.

**Fix**: Added dereference logic at start of `sigil_eq` function.

### 9. Parser::advance Lexer State

**Problem**: `Parser::advance` wasn't updating the lexer field after `next_token` call.

**Fix**: Added `sigil_struct_set_field(&self, "lexer", _t0)` after `next_token`.

## Automated Fix Script

A comprehensive fix script is available at `apply_all_fixes.py`. Run from the `build/` directory:

```bash
cd sigil/sigil-lang/self-hosted/build
python3 ../apply_all_fixes.py
gcc -g -O0 -o sigil_bootstrap sigil_bootstrap.c -lm
```

The script fixes **all 141 pointer/value type mismatches**, including:

### Runtime Support
- Missing includes (ctype.h, time.h)
- `sigil_len_utf8` TAG_CHAR handling for Lexer::advance
- `sigil_String____parse` helper for string-to-number parsing

### Pointer vs Value Semantics
The main issue is the codegen generates incorrect `self` references. Functions either:
- Take `SigilValue*` (pointer) and need `self` or `&var`
- Take `SigilValue` (value) and need `(*self)` when caller has pointer

Fixed categories:
- **Lexer/Parser methods**: `_t*` temp vars need `&_t*`
- **TypeChecker methods**: Mixed - `unify/bind_pattern` expect pointer, `infer_literal` expects value
- **Interpreter methods**: Most expect pointer, but `check_evidence` expects value
- **TypeEnv/Environment**: 4-arg defines expect pointer, 3-arg macro version expects value
- **LoweringContext**: Expects pointer, `(*self)` → `self`
- **CodeGen**: Mixed - `emit_pattern_condition` takes value but other methods vary
- **Helper functions**: `mangle_name`, `escape_char`, `escape_string` expect value

### Special Cases
- Inside `emit_pattern_condition`: `self` is already `SigilValue`, so `(*self)` and `&self` are invalid
- 3-arg vs 4-arg `UncertainTypeEnv____define`: Different macro expansions with different requirements

## Known Remaining Issues

### 1. Other Pattern Match `(1)` Bugs

There may be other places where pattern matching generates `(1)`. Search for:
```
grep "if ((1))" build/sigil_bootstrap.c
```

### 2. sigil_display Returns `<value>` for Strings

The `sigil_display` function returns `<value>` instead of actual string content in error
messages, making debugging harder.

### 3. Macro Redefinition Warnings

The unified build has harmless warnings about redefined macros (e.g., MORPHEMEKIND_*,
EXPR_LET, TRAITITEM_CONST). These are due to duplicate definitions in different modules.

## Testing

After making fixes, rebuild and test:
```bash
cd sigil/sigil-lang/self-hosted/build
gcc -g -O0 -o sigil_bootstrap sigil_bootstrap.c -lm
./sigil_bootstrap dump-ast /tmp/test.sg
```

Test file for basic parsing:
```sigil
fn main() {
    let x = 42;
    println(x);
}

fn add(a: i32, b: i32) -> i32 {
    a + b
}

type MyInt = !u64;
```

## Build Command

```bash
gcc -g -O0 -o sigil_bootstrap sigil_bootstrap.c -lm
```
The `-lm` flag is required for math functions (sqrt, sin, cos, pow).

### 10. Expr::Repeat Missing Pattern Check

**Problem**: `Expr::Repeat { value, count }` pattern in lower.sg was not generating proper
variant check in the C code - it was missing from the enum constant definitions.

**Fix**:
- Added `EXPR_REPEAT 48` and `EXPR_WHILELET 49` constant definitions
- Added proper `sigil_is_struct_variant(_t0, "Expr::Repeat")` check in `sigil_lower_expr`
- Added handler that lowers `Expr::Repeat` to `IrOperation::Repeat`

**Location**: `sigil_lower_expr` function, between Expr::Array and Expr::Tuple handlers.

### 11. TAG_UNIT/TAG_NULL Expression Handling

**Problem**: Multi-file compilation was failing with "missing match arm" errors because
null/unit values were being passed to `sigil_lower_expr` instead of actual expressions.
This occurred when optional expression fields (like else branches) were None.

**Fix**: Added early return handler at the start of `sigil_lower_expr`:
```c
if (_t0.tag == TAG_UNIT || _t0.tag == TAG_NULL) {
    return sigil_struct("IrOperation::Literal", ...);  // Unit literal
}
```

**Location**: `sigil_lower_expr` function, after TAG_REF dereference.

### 12. IrOperation::Array Emission (Critical)

**Problem**: The bootstrap had broken IrOperation::Array emission - it had `if (false) { ... }` which
never executed, causing empty arrays in struct literals to generate `sigil_unit()` instead of proper
array initialization.

**Fix**: Replaced the stub with proper array emission code that:
- Gets the elements from the operation
- Creates an array with `sigil_array(len)`
- Loops to fill in each element

**Location**: `sigil_CodeGen____emit_operation`, IrOperation::Array case.

### 13. sigil_struct_field Tuple Access

**Problem**: `sigil_struct_field` only handled TAG_STRUCT, not TAG_TUPLE. When the codegen tried to
access tuple fields (like struct literal field tuples `(name, value)`) via `sigil_struct_field(field, "0")`,
it returned null.

**Fix**: Added TAG_TUPLE handling to `sigil_struct_field`:
```c
if (v.tag == TAG_TUPLE && v.v.tup.fields != NULL) {
    if (field[0] >= '0' && field[0] <= '9' && field[1] == '\0') {
        size_t idx = field[0] - '0';
        if (idx < v.v.tup.count) {
            return v.v.tup.fields[idx];
        }
    }
    return sigil_null();
}
```

**Location**: `sigil_struct_field` function.

### 14. tick() in Emitted Loops

**Problem**: Generated code had for loops and while loops without `tick()` calls, making infinite
loop detection impossible.

**Fix**:
- Added `sigil_CodeGen____line(self, sigil_string("tick(__func__);"))` after `indent_push` in:
  - LoopVariant::Infinite while loop body
  - LoopVariant::While while loop body
  - LoopVariant::For range loop body
  - LoopVariant::For array iteration loop body
  - Morpheme transform/filter loop bodies

**Location**: `sigil_CodeGen____emit_operation` Loop variants, `sigil_CodeGen____emit_morpheme`.

### 15. Parser Source Fixes

**Problem**: parser.sg had references to non-existent types and variants:
- `VariantFields::Unit` - should be `StructFields::Unit`
- `Expr::Ident` - Expr enum doesn't have an Ident variant

**Fix**:
- Changed `VariantFields::Unit` to `StructFields::Unit` (EnumVariant.fields is of type StructFields)
- Changed `Expr::Ident(...)` to `Expr::Path(TypePath { segments: [...] })` for identifier expressions

**Location**: src/parser.sg

### 16. UnaryOp Enum ID Mismatch (Critical)

**Problem**: The `sigil_lower_unaryop` function was failing to match UnaryOp variants, causing
`!expr` to be lowered to `sigil_unit()` instead of the proper negation expression. This caused
infinite loops in code like:
```sigil
while !Lexer::is_alnum_or_underscore(self.current()) { ... }
```
The generated C code had `sigil_unit()` as the condition, which is always falsy.

**Root Cause**: The parser creates `UnaryOp::Not` values with `UNARYOP_ENUM_ID` (2592675859), but
`sigil_lower_unaryop` was checking for `AST_ENUM_ID` (0xDEAD0008U = 3735928840). These are
different because the parser shares the IR `UnaryOp` enum type, not a separate AST `UnaryOp`.

**Fix**: Modified `sigil_lower_unaryop` to first check if the input is already an IR UnaryOp:
```c
/* Check if this is already an IR UnaryOp (from parser using shared enum) */
if (_t0.tag == TAG_ENUM && _t0.v.e.enum_id == UNARYOP_ENUM_ID) {
    /* Already IR UnaryOp - return as-is */
    return _t0;
}
/* Otherwise try AST enum matching (legacy path) */
```

**Location**: `sigil_lower_unaryop` function.

## Status

**Parser: WORKING** - Successfully parses basic Sigil source files.
**Lowering: WORKING** - Successfully lowers multi-file compilation to IR.
**Multi-file Compilation**: ✅ Fixed - 15 "missing match arm" errors resolved.
**Config::default**: ✅ Fixed - Properly generates field names and values.
**sigil2 Compilation**: ✅ Compiles successfully.
**sigil2 Execution**: ✅ Fixed - UnaryOp infinite loop issue resolved.
