# Bootstrap Fix Plan: Breaking the Cycle

## Current Situation

**Status:** Fix implemented in source but cannot be bootstrapped due to chicken-and-egg problem.

- ✅ Root cause identified and documented
- ✅ Fix implemented in `src/codegen.sg`
- ✅ Manual patches identified (8 locations)
- ❌ Bootstrap cycle prevents fix from working
- ❌ Functions still not appearing in compiled output

## The Problem

```
sigil1 (has bug) → compiles fixed source → generates buggy C → sigil2 (still has bug)
```

Even with manual patching, functions are not appearing, suggesting either:
1. Additional locations need patching beyond the 8 identified
2. The patching is incomplete or incorrect
3. There are additional codegen issues beyond struct_set_field

## Solution Paths

### Path A: Direct sigil1 Bootstrap Fix (RECOMMENDED)

**Approach:** Fix the handwritten C bootstrap compiler directly.

**Steps:**
1. Locate sigil1's C source code (likely in `bootstrap/` or `build/sigil1.c`)
2. Find all `sigil_struct_set_field` calls in the generated codegen functions
3. Apply the same fix pattern:
   - Find all mutable reference parameters in function signatures
   - Change `sigil_struct_set_field(&param, ...)` to `sigil_struct_set_field((SigilValue*)param.v.ptr, ...)`
4. Recompile sigil1 from the fixed C
5. Use fixed sigil1 to compile source → get working sigil2
6. Test sigil2 → sigil3 → sigil3b (fixed-point verification)

**Pros:**
- Most direct solution
- Breaks the bootstrap cycle permanently
- Creates a clean working compiler

**Cons:**
- Requires understanding sigil1's C structure
- May be extensive C code to review

**Estimated Effort:** 2-4 hours

### Path B: Comprehensive Manual Patching

**Approach:** Systematically find and patch ALL affected locations in generated C.

**Steps:**
1. Generate C from fixed source: `sigil1 compile-to-c src/codegen.sg > /tmp/codegen_full.c`
2. Search for ALL `sigil_struct_set_field` calls: `grep -n 'sigil_struct_set_field' /tmp/codegen_full.c`
3. For each call, check if first argument is a mutable reference parameter
4. Apply patches to all locations (not just the 8 in lower.sg)
5. Compile and test

**Pros:**
- Works with existing tools
- No need to understand sigil1 internals

**Cons:**
- May miss some locations
- Tedious and error-prone
- Still relying on buggy sigil1

**Estimated Effort:** 3-6 hours

### Path C: Alternative Bootstrap Compiler

**Approach:** Use a different compiler to bootstrap.

**Options:**
1. Write a minimal Sigil interpreter in Python/Rust to compile codegen.sg
2. Use Styx's Sigil implementation if it exists
3. Cross-compile using another language's compiler

**Pros:**
- Clean room approach
- Avoids sigil1 bugs entirely

**Cons:**
- Significant engineering effort
- May not be feasible if no alternative exists

**Estimated Effort:** 1-2 weeks

### Path D: Incremental Debugging (NOT RECOMMENDED)

**Approach:** Add extensive debug instrumentation to understand why functions are lost.

**Why Not Recommended:**
- The root cause is already known
- The fix is already implemented
- Problem is operational (bootstrap cycle), not technical (wrong fix)
- Would waste time debugging symptoms rather than fixing cause

## Recommended Action Plan

### Phase 1: Attempt Path A (Direct sigil1 Fix)

1. **Locate sigil1 source:**
   ```bash
   find . -name "sigil1.c" -o -name "bootstrap*.c"
   ls -la build/sigil1*
   ```

2. **Identify structure:**
   - Find the emit_function equivalent
   - Find struct_set_field call sites
   - Identify mutable reference parameter patterns

3. **Apply patches:**
   - Use the same logic as the fix in src/codegen.sg
   - Change `&param` to `(SigilValue*)param.v.ptr` for mutable refs

4. **Recompile and test:**
   ```bash
   gcc -o build/sigil1_fixed bootstrap/sigil1.c -lm
   build/sigil1_fixed compile src/lower.sg -o /tmp/test_fixed.c
   gcc -o /tmp/test_fixed /tmp/test_fixed.c -lm
   /tmp/test_fixed compile /tmp/test_minimal_2.sg
   ```

### Phase 2: If Path A Fails, Try Path B

1. **Generate complete C:**
   ```bash
   sigil1 compile-to-c src/*.sg > /tmp/full_compiler.c
   ```

2. **Find ALL struct_set_field calls:**
   ```bash
   grep -n 'sigil_struct_set_field(&[a-z_]*,' /tmp/full_compiler.c > /tmp/all_set_field.txt
   ```

3. **Manually patch each location:**
   - Review each line in context
   - Determine if first argument is a mutable ref parameter
   - Apply fix pattern

4. **Compile and test**

### Phase 3: Verification

Once a working compiler is obtained:

1. **Test basic compilation:**
   ```bash
   sigil2 compile /tmp/test_minimal_2.sg -o /tmp/test_out.c
   grep -c 'sigil_main' /tmp/test_out.c  # Should be > 0
   ```

2. **Test self-compilation:**
   ```bash
   sigil2 compile src/*.sg -o /tmp/sigil3.c
   gcc -o /tmp/sigil3 /tmp/sigil3.c -lm
   ```

3. **Verify fixed-point:**
   ```bash
   sigil3 compile src/*.sg -o /tmp/sigil3b.c
   diff /tmp/sigil3.c /tmp/sigil3b.c  # Should be identical or semantically equivalent
   ```

4. **Test on full codebase:**
   ```bash
   sigil3 compile src/lower.sg -o /tmp/test_lower.c
   grep -c 'sigil_struct_set_field((SigilValue*)module.v.ptr' /tmp/test_lower.c  # Should be 8
   ```

## Success Criteria

- ✅ Compiler produces non-empty output for minimal test programs
- ✅ Functions appear in generated C code
- ✅ Self-compilation succeeds (sigil2 → sigil3)
- ✅ Fixed-point achieved (sigil3 ≈ sigil3b)
- ✅ Generated C contains correct `.v.ptr` extraction for mutable refs

## Rollback Plan

If fixes break compilation:

1. Keep original `build/sigil1` as `build/sigil1.bak`
2. Keep all intermediate binaries: sigil2_patched, sigil2_fixed, etc.
3. Document all changes made to C files
4. Can restore from `/tmp/` backups

## Files to Track

- `build/sigil1` - Original bootstrap compiler
- `build/sigil1_fixed` - Fixed bootstrap compiler (if Path A)
- `/tmp/sigil2_patched` - Current manually patched attempt
- `/tmp/sigil3.c` - Self-compiled output
- `/tmp/sigil3b.c` - Fixed-point verification
- `src/codegen.sg` - Contains the correct fix

## Next Immediate Action

**Start with Path A, Step 1:** Locate and examine sigil1's source code to determine feasibility of direct patching.

```bash
# Find sigil1 source
find /home/crook/dev2/workspace/sigil/sigil-lang/jormungandr -name "sigil1.c" -o -name "bootstrap*.c" 2>/dev/null

# Check if it's a standalone binary or has source
file build/sigil1
strings build/sigil1 | grep -i "version\|copyright" | head -5
```

If sigil1 source is available → proceed with Path A
If sigil1 is binary only → proceed with Path B
