# String Methods Investigation - 2026-01-14

## Summary

Investigation into why String method implementations (`is_empty`, `contains`, `push`, `clone`) were missing from generated C code.

## Key Findings

### 1. Implementations ARE in Source Code ✅

All String method implementations exist in `src/codegen.sg`:
- **Line 7152**: `sigil_String____clone`
- **Line 7154**: `sigil_String____is_empty`  
- **Lines 7207-7213**: `sigil_String____contains`
- **Lines 7610-7639**: `sigil_String____push`

Located within the `emit_builtin_impls()` function (lines 6359-7926), which is unconditionally called from `emit_main_wrapper()` at line 6333.

### 2. Bootstrap Mystery

When recompiling from source with sigil2, the implementations don't appear in the generated C output, even though:
- The emit code exists and is unconditional
- Other implementations in the same function DO get generated
- The compiler generates code that CALLS `sigil_CodeGen____line()` to emit them
- But the `line()` calls don't produce output for these specific strings

### 3. Working Compiler Available ✅

**`sigil2` binary has everything:**
- ✅ All String method implementations present
- ✅ Compiles cleanly (0 errors)
- ✅ Successfully generates C from Sigil source
- ✅ All required builtins included

### 4. Common Bootstrap Issues

When compiling from source, several recurring issues appear:
1. **Missing String methods** - is_empty, contains, push, clone
2. **Missing helper builtins** - Vec____len, Box____into_raw, skip, with_note, any
3. **Duplicate `sigil_add`** - One-liner appears alongside full implementation
4. **Stray `#endif`** - SIGIL_EXTRA_STDLIB_DEFINED without matching #ifndef
5. **Underscore redefinitions** - Multiple `SigilValue _ =` declarations
6. **Missing function stubs** - sigil_lower_file, sigil_Interpreter____new

## Investigation Process

### Test 1: Examined Source Code
- Confirmed all implementations exist at correct line numbers
- Verified `emit_builtin_impls()` is called unconditionally
- Found no conditional logic that would skip these specific methods

### Test 2: Compiled with sigil2
```bash
./sigil2 compile ../src/*.sg -o sigil3_fresh.c  # ~9 seconds
```
**Result**: 2.6MB C file generated, but missing String implementations

### Test 3: Manual Patching
Added missing implementations to generated C files:
- String methods (is_empty, contains, push, clone)
- Helper builtins (Vec____len, Box____into_raw, skip, with_note, any)
- Removed duplicates and stray directives

### Test 4: Simple Compilation Test
```bash
./sigil2 compile /tmp/test_simple_fn.sg -o /tmp/test_out.c
```
**Result**: Successfully generated 73KB C file with correct structure

## Affected Files

### Source Files
- `src/codegen.sg` - Contains correct implementation code
- Lines 7130-7700 - `emit_builtin_impls()` function

### Generated Files with Issues
- `sigil.c` - Missing String implementations
- `sigil_v2.c` - Missing String implementations
- `sigil3_fresh.c` - Missing String implementations

### Working File
- `sigil2.c` - Has ALL implementations ✅

## Workarounds Applied

### Manual Patches to Generated C Files

**1. Add String method implementations** (after `push_str`):
```c
SigilValue sigil_String____is_empty(SigilValue s) { 
    return sigil_bool(s.tag != TAG_STRING || !s.v.s || s.v.s[0] == 0); 
}

SigilValue sigil_String____contains(SigilValue s, SigilValue sub) {
    if (s.tag != TAG_STRING || sub.tag != TAG_STRING) return sigil_bool(false);
    if (!s.v.s || !sub.v.s) return sigil_bool(false);
    return sigil_bool(strstr(s.v.s, sub.v.s) != NULL);
}

SigilValue sigil_String____clone(SigilValue s) { 
    return s.tag == TAG_STRING && s.v.s ? sigil_string(s.v.s) : s; 
}

// ... push implementation with StringBuilder support ...
```

**2. Add helper builtins**:
```c
SigilValue sigil_Vec____len(SigilValue v) { return sigil_len(v); }
SigilValue sigil_Box____into_raw(SigilValue b) { return b; }
SigilValue sigil_skip(SigilValue arr, SigilValue n) { /* implementation */ }
SigilValue sigil_with_note(SigilValue v, SigilValue note) { (void)note; return v; }
SigilValue sigil_any(SigilValue iter, SigilValue pred) { /* implementation */ }
```

**3. Fix duplicate sigil_add**:
```bash
sed -i '/^SigilValue sigil_add.*{ return sigil_int.*}$/d' file.c
```

**4. Remove stray #endif**:
```bash
sed -i '/^#endif \/\* SIGIL_EXTRA_STDLIB_DEFINED \*\/$/d' file.c
```

**5. Fix underscore redefinitions**:
```bash
awk '/^[[:space:]]*SigilValue _ =/ { gsub(/SigilValue _/, "SigilValue _" NR "_"); } { print }' file.c > fixed.c
```

## Hypothesis: Why Implementations Don't Appear

**Possible explanations:**

1. **String escaping issue**: The `line()` method might be dropping these specific strings due to special characters or length
2. **StringBuilder corruption**: The output buffer might be getting corrupted for multi-line implementations
3. **Code generation bug**: A subtle bug in how `self.line()` accumulates output
4. **Bootstrap divergence**: At some point, a compiler version was used that had a bug, and subsequent generations inherited missing code

**Evidence supporting StringBuilder theory:**
- Simple one-liners (as_str, from_raw) work fine
- Multi-line implementations (contains, push) don't appear
- But some other multi-line implementations (push_str) DO work

## Recommendations

### Short-term
1. **Use sigil2 directly** - It has all implementations and works correctly
2. **Apply manual patches** - When bootstrapping, patch generated C before compiling
3. **Automate workarounds** - Script the common fixes (duplicates, underscores, #endif)

### Long-term
1. **Debug `line()` method** - Add logging to see why specific strings aren't output
2. **Test StringBuilder** - Verify the output accumulation isn't dropping data
3. **Fixed-point verification** - Compare consecutive compiler generations byte-by-byte
4. **Add assertions** - Verify String methods exist in generated C as a build check

## Current Status

### Working
- ✅ sigil2 binary compiles and runs
- ✅ sigil2 can compile simple Sigil programs to C
- ✅ All String method implementations exist in source
- ✅ Manual workarounds for common issues documented

### Not Working
- ❌ Bootstrap chain (sigil2 → sigil3) produces incomplete C
- ❌ String methods don't appear in generated code
- ❌ Several helper builtins also missing
- ❌ Styx compilation (uses advanced features not yet supported)

## Files Referenced

- `src/codegen.sg:6359-7926` - emit_builtin_impls() function
- `src/codegen.sg:7152` - clone implementation
- `src/codegen.sg:7154` - is_empty implementation  
- `src/codegen.sg:7207-7213` - contains implementation
- `src/codegen.sg:7610-7639` - push implementation
- `build/sigil2.c` - Working compiler with all implementations
- `build/sigil.c` - Generated, missing implementations
- `build/sigil3_fresh.c` - Generated, missing implementations

## Next Steps

1. Add debug logging to `CodeGen::line()` method
2. Compare sigil2.c with sigil3_fresh.c to find divergence point
3. Test StringBuilder capacity and check for silent buffer overflows
4. Create automated patch script for common bootstrap issues
5. Add CI check to verify String methods exist in generated code

## Session Artifacts

- `/tmp/string_contains_push.txt` - String method implementations
- `/tmp/other_missing_builtins.txt` - Helper builtin implementations
- `build/sigil2` - Working compiler binary (4.3MB)
- `build/sigil3_fresh.c` - Generated C with known issues (2.6MB)
