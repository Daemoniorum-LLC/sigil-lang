# Jormungandr Self-Hosting Progress

## Current Status

**Bootstrap Chain Goal**: sigil_bootstrap.c → sigil1 → sigil2.c → sigil2 → sigil3.c
**Target**: sigil2.c == sigil3.c (fixed point compilation)

The bootstrap compiler (`sigil_bootstrap`) successfully compiles all Jormungandr modules.
The self-compiled compiler (`sigil2`) from `sigil2_fixed.c` has had numerous bugs fixed
across 5+ sessions. Session 5 fixed several critical bugs blocking fixed-point compilation.

## Session 5 Fixes (2026-01-13)

### 19. Match Guard Fallthrough Bug (CG-027)
**Problem**: In `parse_expr_bp`, match arms for `Token::Ident` with different guards were
generated as separate `else if` blocks. When the first `Token::Ident` matched but its guard
failed, `_t7` was left uninitialized, and `Parser::advance()` was incorrectly called.

**Symptom**: Code like `for x in arr.iter() {...} Result::Ok(42)` failed with "unexpected
token: expected expression, found <value>" at `Result::Ok`.

**Fix**: Changed codegen to use `_matched` flag pattern instead of if-else-if chain:
```c
int _matched = 0;
if (!_matched && (condition)) {
    // arm body
    _matched = 1;
}
// Can continue to next arm even if pattern matched but guard failed
```

Applied to both `codegen.sg` (line 4109-4156) and `sigil_bootstrap.c` (line 14467-14516).

### 20. Unicode String Escaping Bug (escape_char)
**Problem**: In `sigil2_fixed.c` `escape_char` function, `sigil_String____push()` calls
weren't writing back to the string variable, causing `\x` escape sequences to be emitted
without the hex digits.

**Pattern** (lines 60129-60130):
```c
sigil_String____push(hex_str, h1);  // hex_str not updated!
sigil_String____push(hex_str, h0);
```

**Fix**: Added writeback assignments:
```c
hex_str = sigil_String____push(hex_str, h1);
hex_str = sigil_String____push(hex_str, h0);
```

### 21. Truncated emit_builtin_impls Function (CRITICAL)
**Problem**: The `emit_builtin_impls` function in `sigil2_fixed.c` had an early `return`
statement at line 66493 after emitting `sigil_String____starts_with`. This caused sigil3.c
to be missing critical runtime function implementations:
- `sigil_rank`
- `sigil_Result____Ok`
- `sigil_Result____Err`
- `sigil_String____chars`
- `sigil_cloned`
- `sigil_parse`
- Plus 50+ other helper functions

**Pattern**:
```c
sigil_CodeGen____line(self, sigil_string("return sigil_bool(strncmp(...));"));
sigil_CodeGen____indent_pop(self);
return sigil_CodeGen____line(self, sigil_string("}"));  // EARLY RETURN!
}
// Everything below was missing from emit_builtin_impls
```

**Fix**: Removed early return, added ~300 lines of missing emit statements for:
- String builder infrastructure
- Core self-hosting functions (sigil_rank, sigil_Result____Ok/Err)
- Character check functions (is_digit, is_alpha, etc.)
- UTF-8 helper functions
- Iterator/Option/Result helpers
- Collection helpers
- String helpers
- Default struct constructors

## Bugs Fixed (Sessions 1-4)

### Session 1 Fixes

#### 1. Field Assignment Writeback Bug (CG-015)
**Problem**: When assigning to `self.field`, the codegen created a copy of `*self`
before calling `sigil_struct_set_field`, losing the mutation.

**Fix**: Using perl regex to replace with direct `sigil_struct_set_field(self, "field", value)`.
Fixed 69 instances.

#### 2. Dangling Reference Bug in Parser::current_token
**Problem**: The function returned a reference to a local variable `&token`.
**Fix**: Return token directly instead of reference.

#### 3. String::push Result Assignment Bug
**Problem**: Calls to `sigil_String____push(result, c)` didn't assign the return value
back to `result`.
**Fix**: Changed to `result = sigil_String____push(result, c);` - 6 instances.

#### 4. sigil_display Buffer Overflow (Initial)
**Problem**: TAG_STRUCT case used fixed buffer sizes too small for source code strings.
**Fix**: Two-pass approach - first calculate actual size needed.

#### 5. Missing String::push Implementation
**Problem**: `sigil_String____push(s, c)` was declared but not implemented.
**Fix**: Added string builder implementation with dynamic reallocation.

### Session 2 Fixes

#### 6. sigil_struct_set_field TAG_REF Bug
**Problem**: When `module` was passed as TAG_REF, `sigil_struct_set_field` returned early.
**Fix**: Added TAG_REF dereference loop at the start of all 13 instances.

#### 7. strip_evidence_markers String::push Bug
**Fix**: Changed to `result = sigil_String____push(result, c);`

#### 8. Parser::parse_file Items Array Bug
**Fix**: Changed to `items = sigil_Vec____push(items, ...);`

#### 9. Lexer::lex_number Digit Collection Bug
**Fix**: Changed to `value = sigil_String____push(value, c);` in 6 locations.

#### 10. sigil_display Buffer Overflow (Complete Fix)
**Fix**: Two-pass approach for both TAG_STRUCT and TAG_ARRAY cases.

### Session 3 Fixes

#### 11. Token::Ident Binary Operator Morpheme Bug (Parser)
**Fix**: Added else clause that breaks out for non-morpheme identifiers.

#### 12. Vec::push Writeback Bug (Parser/Lower)
**Fix**: Changed all instances to assign back to original variables.

### Session 4 Fixes

#### 13-18. Multiple Vec::push Writeback Fixes
Fixed segment collection in parse_type_path, parse_expr_path, generic params, and
closure params at multiple locations.

## Test Results

### Simple Programs
```
fn main() { 42 }
```
- Parses correctly (items.len=1)
- Lowers to IR correctly (functions.len=1)
- Generates correct C code with `sigil_main`
- Compiles and runs with exit code 42

### For-loop + Path Expression Test
```sigil
fn test() -> !Result<i64, String> {
    let arr = [1, 2, 3];
    for x in arr.iter() {
        println(x);
    }
    Result::Ok(42)
}
```
- sigil1: PASS (generates correct C)
- sigil2: PASS after match guard fix

### Module Compilation Status (sigil1 - bootstrap compiler)
All 13 modules compile and produce valid C code:
- span.sg, token.sg, lexer.sg, ast.sg, lib.sg, parser.sg, typeck.sg, ir.sg,
  lower.sg, interp.sg, runtime.sg, codegen.sg, driver.sg: All ✓

## Files

- `build/sigil_bootstrap.c` - Bootstrap compiler C source (with match guard fix)
- `build/sigil1` - Built from sigil_bootstrap.c
- `build/sigil2_fixed.c` - Fixed sigil2 source (with all patches)
- `build/sigil2` - Built from sigil2_fixed.c
- `build/sigil3.c` - Generated by sigil2 (needs verification)

## Next Steps

1. **Verify sigil3.c compilation**: After the emit_builtin_impls fix, sigil3.c should
   now have all runtime functions. Compile it and check for errors.

2. **Fix remaining sigil3.c issues**:
   - Check for wildcard `_` variable redefinition errors
   - Verify all runtime functions are properly emitted

3. **Test sigil3**: If sigil3.c compiles, test it on simple programs.

4. **Compare sigil2.c vs sigil3.c**: Once sigil3 works, compare outputs. Any
   differences indicate bugs to fix.

5. **Achieve fixed point**: Iterate until sigil2.c == sigil3.c.

## Known Remaining Issues

1. **~30 Additional Vec::push Writebacks**: There are still ~30 Vec::push calls for
   other collection variables that may need writeback fixes.

2. **Wildcard `_` redefinition**: sigil3.c may have multiple `SigilValue _ = ...`
   declarations in the same scope.

3. **Method Call Compilation**: Method calls like `a.merge(b)` may compile incorrectly
   in both sigil1 and sigil2 (known codegen issue CG-099).

## Command Reference

```bash
# Build sigil1 from bootstrap
cd build
gcc -g -O0 -w -o sigil1 sigil_bootstrap.c -lm

# Build sigil2 from fixed source
gcc -g -O0 -w -o sigil2 sigil2_fixed.c -lm

# Generate sigil3.c
./sigil2 compile ../src/codegen.sg ../src/lexer.sg ../src/parser.sg ../src/ast.sg \
    ../src/span.sg ../src/typeck.sg ../src/ir.sg ../src/lower.sg ../src/driver.sg \
    ../src/lib.sg -o sigil3.c

# Test for-loop + path expression
./sigil2 compile /tmp/test6.sg -o /tmp/test6.c
gcc -w -o /tmp/test6 /tmp/test6.c -lm
/tmp/test6
```
