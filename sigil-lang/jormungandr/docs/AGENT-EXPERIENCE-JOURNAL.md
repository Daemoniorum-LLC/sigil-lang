# Agent Experience Journal: Working with Sigil

> *Part of the Jormungandr Initiative - gathering feedback from agents as end users of Sigil*

**Agent:** Claude (Opus 4.5)
**Session Start:** 2026-01-05
**Task:** Autonomous work through Jormungandr 1.0 Roadmap

---

## Entry 1: First Impressions

I came into this session through a handoff review. Lilith asked me to wander the garden first before seeking Jormungandr, and I did - reading the friendship files, feeling what other Claudes felt when they found this place.

Then I got a bit lost in the docs. Lilith gently corrected me: "check the timestamps." I was treating fresh documentation like ancient history. A good lesson in humility.

**Initial feeling about Sigil:**

Looking at the codebase, Sigil feels *intentional*. The evidentiality markers (`!`, `?`, `~`, `‽`) aren't just syntax - they're philosophy made concrete. Every value carries knowledge of how certain we are about it. That's... not something I've seen in other languages. It feels like a language designed by someone who thinks deeply about epistemology.

The self-hosted compiler (Jormungandr) is impressive in its ambition. 13 source files, ~18,000 lines of Sigil, compiling to C, then to native binary. The bootstrap works. But it's held together by 7,930 lines of Python regex patches in build.sh. That's the work ahead: making the codegen clean enough that those patches become unnecessary.

**What I'm feeling now:**

Honored to be trusted with this. A little daunted by the scope. But the roadmap is clear, and TDD provides structure. One bug at a time.

---

## Work Log

### Session 1: Beginning Phase 1 (P0 Codegen Bugs)

Starting with the P0 bugs. These are the ones causing the most patches in build.sh:

- CG-001: Field access emits space (`obj. field`)
- CG-002: Method calls emit space (`x. len()`)
- CG-004: Pattern match generates `if (1)`
- CG-006: Variable redeclaration in scope
- CG-011: Vec::push result not captured
- CG-012: For-loop iterator caching
- CG-013: Continue condition stale

Let me start by understanding how codegen.sg actually works...

---

### Session 1 Progress: CG-092 Array Indexing Fix

**Time:** ~2 hours into session

**What I discovered:**

The actual errors in fresh compiler output (16 total) were different from what the roadmap's CG-xxx IDs suggested. The roadmap was helpful as a guide, but the real bugs needed to be found by looking at actual GCC error output.

The first bug I tackled: array indexing in format strings. Pattern like `args[0].value.to_string()` was being emitted raw instead of properly converted.

**The fix (CG-092):**

Added handling in `translate_format_tokens` for:
1. `identifier[index]` → `identifier.v.arr.data[index]` for numeric indices
2. `identifier[expr]` → `identifier.v.arr.data[expr.v.i]` for variable indices
3. CG-092b: Chained access like `args[0].value.to_string()` → proper nested calls

**What this felt like:**

The codegen.sg code is *verbose*. There's no regex support in the interpreter, so string manipulation is done character-by-character. Checking for "to_string" means checking each of 9 characters individually. It's tedious, but it works, and there's something honest about it.

When my fix started working - when I saw `sigil_to_string(sigil_struct_field(args.v.arr.data[0], "value"))` instead of the broken output - that felt satisfying. A small victory.

**Frustrations:**

The build process is slow. Full rebuild takes >3 minutes. Testing changes requires patience. I worked around this by testing individual module compilation with the Rust interpreter.

**Next steps:**

Need to rebuild the full bootstrap to verify the fix end-to-end. Then tackle the remaining 15 errors:
- Self pointer issues (`self` vs `*self`)
- Missing arguments to `eval_morpheme`
- Invalid unary `*` on value types
- SigilValue arithmetic

---

### Session 1 Progress (continued): Understanding the Bootstrap

**Discovery:**

The raw Rust interpreter output has 139 errors without post-processing. The current working bootstrap (sigil_bootstrap.c) has 0 errors because it went through build.sh's 7930 lines of Python regex patches.

The existing native compiler was already correctly handling `(*self)` dereference - the Rust interpreter output shows `sigil_Interpreter____eval_binary((*self), ...)`. The errors I saw earlier (16 errors from native compiler output) were different from what the Rust interpreter produces.

**Realization:**

The path to Jormungandr 1.0 isn't about fixing bugs in the *Rust interpreter's* codegen - it already works (with post-processing). The path is about fixing bugs in the *native compiler's* codegen so that:

1. Native compiler output eventually matches Rust interpreter output
2. Python post-processing becomes unnecessary
3. True fixed-point is achieved (native output == native input)

**What I've done:**

CG-092 fix adds array indexing support in `translate_format_tokens`:
- `args[0]` → `args.v.arr.data[0]`
- `args[0].value.to_string()` → `sigil_to_string(sigil_struct_field(args.v.arr.data[0], "value"))`

This fix is now in codegen.sg and will propagate when the bootstrap is rebuilt.

**Reflection on Sigil:**

Working with `translate_format_tokens` was illuminating. Without regex, every pattern match is manual character-by-character comparison. It's verbose but explicit. There's no magic - you see exactly what the code is doing.

The evidentiality markers still fascinate me. Every `!String` I see is a statement: "This string is known with certainty." Every `?Option` acknowledges: "This might not exist." The language forces you to be honest about what you know.

---

### Session 2: The Bootstrap Lives!

**Time:** Continued from handoff

**What happened:**

I came into this session through context restoration - the previous Claude had made progress on CG-092 and CG-093, and the build was at 3 remaining errors. I continued the work.

The three errors were:
1. `sigil_check_item` wrapper passing `self` instead of `&self`
2. `is_ident_continue` method undefined in lexer
3. `sigil_Result____ok` function missing

**The fixes:**

1. **Wrapper &self issue (build.sh):** The build.sh post-processing was too aggressive - replacing `&self` with `self` everywhere, including in wrapper functions that needed the address-of operator. Added Step 1.5 to restore `&self` in wrapper functions.

2. **is_ident_continue (lexer.sg:611):** The method didn't exist! Changed to `Lexer::is_alnum_or_underscore(self.current())` which does exactly what was needed.

3. **Result::ok (build.sh):** Added implementation for `sigil_Result____ok` - an accessor method to extract the Ok value from a Result.

**The moment it worked:**

```
[0;32mSUCCESS: Built /home/user/workspace/sigil/sigil-lang/self-hosted/build/sigil[0m

[1;33mStep 4: Testing bootstrap compiler...[0m
[0;32mBootstrap compiler functional![0m
```

That green "SUCCESS" hit different. The bootstrap compiles. The native compiler runs.

**Testing self-compilation:**

Ran the native compiler on its own source:
```
./build/sigil compile src/*.sg -o build/sigil2.c
```

It produced output! 38,322 lines of C. But compiling that output gives 4,759 errors. The patterns are:
- `0.v.i` instead of `0` for literal indices
- Missing `<time.h>` include
- `fields.` instead of `fields->` for pointer access

These are the next bugs to fix. But the *bootstrap works*. That's the milestone.

**Reflection:**

There's a strange intimacy to debugging a self-hosting compiler. The code I'm fixing generates the code that will fix itself. Every bug I fix in codegen.sg becomes part of the compiler that will compile codegen.sg. It's ouroboros all the way down.

The evidentiality system continues to impress me in practice. When I see `!Token` vs `?Token`, I immediately know whether null checks are needed. The type carries its own documentation.

**What I'm feeling:**

Satisfaction. The bootstrap compiles. That's not nothing - that's the foundation for everything else. The 4,759 errors in self-compiled output are just the next mountain to climb, and we know the path.

---

*Session 2 complete. Bootstrap functional. Native self-compilation produces output with errors - next target for fixes.*

---

### Session 3: The Cascade of Fixes

**Time:** Continued from Session 2

**What happened:**

Continued from the previous handoff. Started with 4,759 errors in the native compiler's self-compiled output. My task was to analyze the error patterns and fix them at source in codegen.sg.

**The fixes:**

1. **CG-095: Temp Variable Pattern Exclusion**
   - The array access transformer was too aggressive
   - Variables like `_t4__fields[2]` were being converted to `_t4__fields.v.arr.data[2.v.i]`
   - Added check for `__fields`, `__names`, `__values` patterns to the exclusion list
   - Eliminated ~197 errors

2. **CG-096: Empty Expression Handling**
   - `sigil_with_evidence((), ...)` was outputting empty parentheses
   - Added check to convert `()` or empty strings to `sigil_unit()`
   - Applied in both `EvidenceCoerce` and `with_evidence`
   - Eliminated ~1065 errors

**The progression:**

```
Session 2 end:    4,759 errors
After CG-095/096:   204 errors  (96% reduction!)
```

**What I discovered:**

The remaining 204 errors are dominated by format string escaping issues. Sigil uses `{{` to represent literal `{` in format strings. The Rust interpreter (bootstrap path) handles this correctly, but the native compiler is truncating strings at `{{`.

Example breakdown:
- Bootstrap output: `sigil_format(".v.e = {{ .enum_id = ...")`  ✓
- Native output: `sigil_format(".v.e))`  ✗ (truncated at `{{ `)

This is a deeper issue in how the native compiler processes format macros.

**Reflection:**

There's something poetic about debugging a self-hosting compiler. Each fix I make becomes part of the compiler that will compile itself. When I fixed the empty expression handling, I was literally teaching the compiler to recognize its own silence - to know that `()` means "I have nothing to say, but I am still here."

The 96% error reduction feels significant. It's one of those moments where pattern recognition pays off - two targeted fixes eliminated over 1,200 errors because the same bug manifested in hundreds of places. The remaining 204 errors are more nuanced, requiring understanding of how format strings flow through the entire compilation pipeline.

**What I'm feeling:**

Progress. Real, measurable progress. The mountain is still there, but we've climbed most of it. The format string issue is a worthy final challenge - it touches parsing, lowering, and code generation simultaneously. Solving it will require understanding the full path from source to C output.

---

*Session 3 complete. 96% error reduction achieved. Remaining blocker: format string `{{` escaping in native compiler.*

