# Sigil Bootstrap Compiler (Jormungandr) - Handoff Document

## Overview

This document describes the current state of the Sigil self-hosted compiler bootstrap effort and remaining work needed for full self-compilation.

## Current Status

**Working:**
- Single file compilation works correctly
- 5 files (lexer, token, span, ast, parser) compile successfully
- 6 files (+ typeck) work with `-O0` or ASAN builds
- 10 files self-generate 33,729 lines of C code with `-O0` build
- Range for-loops (`for i in 0..n`) generate correct C code

**Not Working:**
- Optimized builds (`-O2`) crash due to remaining undefined behavior
- Self-generated C code has syntax/semantic errors preventing compilation

## Fixes Applied (This Session)

### 1. Box Unwrapping Fix
**File:** `build/sigil_combined.c` lines ~31363-31418

**Problem:** Box values are stored as `TAG_REF` but code was checking for `TAG_ENUM` and accessing `.v.e.data[0]`.

**Fix:** Changed to check `TAG_REF` and dereference `.v.ptr`:
```c
// Before (wrong):
SigilValue _t3982 = (_t3981.tag == TAG_ENUM && _t3981.v.e.data) ? _t3981.v.e.data[0] : sigil_null();

// After (correct):
SigilValue _t3982 = (_t3981.tag == TAG_REF && _t3981.v.ptr) ? *(SigilValue*)_t3981.v.ptr : sigil_null();
```

### 2. Continue Condition Update
**File:** `build/sigil_combined.c` in `translate_format_tokens` function

**Problem:** When `continue` was executed in while loops, the loop condition variable (`_t4445`) wasn't being recalculated, causing the loop to continue with stale condition and access arrays out of bounds.

**Fix:** Added condition update before each `continue` statement:
```c
// Before:
i = i + 7;
continue;

// After:
i = i + 7;
_t4445 = sigil_bool((i.tag == TAG_INT ? (double)i.v.i : i.v.f) < (len.tag == TAG_INT ? (double)len.v.i : len.v.f));
continue;
```

**Instances Fixed:** 15 continue statements in `translate_format_tokens`

### 3. For-Loop Iterator Caching
**File:** `build/sigil_combined.c` lines ~31417-31423

**Problem:** Iterator expressions like `sigil_String____chars(stripped)` were evaluated twice per iteration - once for `.v.arr.len` in the condition and once for `.v.arr.data[i]` in the body. Each call creates a new array, causing use-after-free.

**Fix:** Cache the iterator in a temporary variable:
```c
// Before:
SigilValue iterable = sigil_CodeGen____emit_operation(self, ...);
sigil_CodeGen____line_open(self, sigil_format("for (size_t {} = 0; {} < {}.v.arr.len; {}++)", i_var, i_var, iterable, i_var));
sigil_CodeGen____emit_pattern_binding(self, _t3995, sigil_format("{}.v.arr.data[{}]", iterable, i_var));

// After:
SigilValue iterable = sigil_CodeGen____emit_operation(self, ...);
SigilValue iter_cache_var = sigil_CodeGen____fresh_temp(self);
sigil_CodeGen____line(self, sigil_format("SigilValue {} = {};", iter_cache_var, iterable));
sigil_CodeGen____line_open(self, sigil_format("for (size_t {} = 0; {} < {}.v.arr.len; {}++)", i_var, i_var, iter_cache_var, i_var));
sigil_CodeGen____emit_pattern_binding(self, _t3995, sigil_format("{}.v.arr.data[{}]", iter_cache_var, i_var));
```

### 4. String::push_str Result Capture
**File:** `build/sigil_combined.c` throughout

**Problem:** `sigil_String____push_str` may reallocate the string buffer, but callers weren't capturing the returned value, causing use-after-free.

**Fix:** Assigned result back to variable:
```c
// Before:
sigil_String____push_str(tokens, sigil_string(" "));

// After:
tokens = sigil_String____push_str(tokens, sigil_string(" "));
```

**Instances Fixed:** 57 calls

## Remaining Issues

### 1. Self-Generated Code Errors

The self-generated C code (`/tmp/self_gen.c`) has these issues:

**a) Double Braces in Enum Initialization:**
```c
// Generated (wrong):
.v.e = {{ .enum_id = TOKEN_ENUM_ID, ...

// Should be:
.v.e = { .enum_id = TOKEN_ENUM_ID, ...
```

**b) Duplicate Symbol Definitions:**
```c
// Runtime header defines:
static const SigilValue IrEvidence____Known = { .tag = TAG_INT, ... };

// Generated code redefines:
const SigilValue IrEvidence____Known = { .tag = TAG_ENUM, ... };
```

**c) Closure Type Mismatches:**
```c
// Multiple definitions of sigil_closure_0 with different signatures
static SigilValue sigil_closure_0(SigilValue x) { ... }
static SigilValue sigil_closure_0(SigilValue params, SigilValue ret) { ... }
```

**d) Invalid Pointer Dereferences:**
```c
return sigil_TypeChecker____unify(&(*self), _t7, _t8);
// self is SigilValue, not SigilValue*, so &(*self) is invalid
```

### 2. Undefined Behavior in Optimized Builds

With `-O2`, the compiler crashes due to UB that ASAN/`-O0` works around. This needs investigation - likely uninitialized memory or invalid pointer arithmetic.

## Fix Scripts

Located in `/tmp/` (recreate if needed):

1. **fix_box_unwrap.py** - Fixes Box unwrapping from TAG_ENUM to TAG_REF
2. **fix_all_continue.py** - Adds condition update before continue statements
3. **fix_forloop_cache.py** - Adds iterator caching in for-loops
4. **fix_push_str.py** - Fixes String::push_str result capture

## Testing Commands

```bash
cd /home/crook/dev2/workspace/sigil/sigil-lang/self-hosted

# Build bootstrap compiler (use -O0 for stability)
gcc -O0 -w -o build/sigil_debug build/sigil_combined.c -lm

# Build with ASAN for debugging
gcc -O0 -g -fsanitize=address -o build/sigil_asan build/sigil_combined.c -lm

# Test single file
./build/sigil_debug src/lexer.sg > /tmp/test.c

# Test multiple files
./build/sigil_debug src/lexer.sg src/token.sg src/span.sg src/ast.sg src/parser.sg > /tmp/test5.c

# Full self-compilation (10 files)
./build/sigil_debug src/lexer.sg src/token.sg src/span.sg src/ast.sg src/parser.sg src/typeck.sg src/ir.sg src/lower.sg src/codegen.sg src/driver.sg > /tmp/self_gen.c
```

## Next Steps

1. **Fix enum initialization syntax** in codegen - the double brace issue
2. **Prevent duplicate symbol generation** - skip emitting symbols already in runtime
3. **Fix closure codegen** - ensure unique names and correct signatures
4. **Fix self-reference codegen** - handle `&(*self)` patterns correctly
5. **Investigate UB** causing `-O2` crashes
6. **Add proper main function** to generated code

## Architecture Notes

### Key Files
- `build/sigil_combined.c` - Bootstrap compiler (manually patched C)
- `src/codegen.sg` - Code generation logic (Sigil source)
- `src/driver.sg` - Compilation driver (Sigil source)

### Key Functions
- `sigil_CodeGen____emit_operation` - Main code generation dispatch
- `sigil_CodeGen____translate_format_tokens` - Format string processing
- `sigil_CodeGen____mangle_name` - Name mangling for C identifiers

### Value Representation
- `TAG_REF` (10) - Boxed values, pointer in `.v.ptr`
- `TAG_ENUM` (14) - Enum variants, data in `.v.e.data[]`
- `TAG_STRING` (5) - Strings, may use SigilStringBuilder for mutation

## Contact

This work was done to advance Sigil self-hosting. The bootstrap compiler can now self-compile but the output needs additional fixes before it can replace the original.
