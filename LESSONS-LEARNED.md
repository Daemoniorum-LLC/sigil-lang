# Sigil Language - Lessons Learned

This file captures organizational memory across agent sessions. Read before starting work.
Document discoveries and mistakes when ending sessions.

---

## 2026-02-11 - LLVM Codegen Float Operations

### Context
Implementing DCT (Discrete Cosine Transform) benchmark to compare Sigil LLVM vs Rust performance.

### What Happened
Float operations produced garbage values (`3.0 + 2.0 = -1.11254e-308`) even though
code compiled without errors.

### Root Cause
Sigil stores ALL values as i64 (including floats as bit patterns). The LLVM codegen
was treating float operations as integer operations:
- Used `add` instead of `fadd`
- Used `mul` instead of `fmul`
- No bitcast from i64 to f64 before operations

### Lesson
When Sigil represents floats as i64 bit patterns, every float operation must:
1. Bitcast i64 → f64
2. Perform float operation (fadd, fsub, fmul, fdiv)
3. Bitcast f64 → i64 for storage

This requires tracking which variables are float-typed through the compilation scope.

### Prevention
- Added `float_vars: HashSet<String>` to `CompileScope`
- Added `is_float_expr_with_scope()` function to detect float expressions
- Added `compile_float_binary_op()` for proper float arithmetic

---

## 2026-02-11 - Vec Memory Layout

### Context
Vec indexing returned wrong values (struct fields instead of data elements).

### What Happened
`v[0]` returned values like 3, 5, 10 instead of expected data. These corresponded to
the `len`, `cap`, and first data element positions.

### Root Cause
Vec layout is `{len: i64, cap: i64, data: i64[]}` with data stored INLINE starting at
offset 2, NOT as a pointer. The codegen was treating `data` as a pointer field at offset 2,
when it's actually the start of inline data.

### Lesson
Vec data is inline, not pointed-to:
```
offset 0: len (i64)
offset 1: cap (i64)
offset 2: data[0] (i64)
offset 3: data[1] (i64)
...
```

To access `v[i]`, compute `base_ptr + (i + 2) * 8`.

### Prevention
- All Vec index operations now add 2 to the index before GEP
- This applies to both read and write operations

---

## 2026-02-11 - Math Functions (PI, sqrt, cos)

### Context
DCT calculation returned 0 for PI and all trig functions.

### What Happened
`PI()` returned 0. `sqrt()`, `cos()`, `sin()` all returned 0.

### Root Cause
1. `PI()` function was never implemented in LLVM codegen or C runtime
2. Method calls like `value.sqrt()` weren't mapped to runtime functions

### Lesson
Math operations in LLVM codegen require explicit mapping:
- `PI()` → custom `sigil_pi()` function (returns f64 bits as i64)
- `.sqrt()` → `sigil_sqrt()`
- `.cos()` → `sigil_cos()`
- etc.

### Prevention
- Added `sigil_pi()` to both JIT (Rust) and C runtime
- Added explicit MethodCall handling for math methods in LLVM codegen
- When adding new stdlib functions, ensure all backends implement them

---

## 2026-02-11 - Benchmark Optimization Away

### Context
Running DCT benchmark showed 0 microseconds for all operations.

### What Happened
Benchmark loop ran but timing showed 0:
```
Size     DCT (μs)
  16         0
  32         0
```

### Root Cause
Using `_ = dct_1d(data, n)` allowed LLVM to optimize away the entire function call
since the result was unused.

### Lesson
To prevent dead code elimination in benchmarks:
1. Accumulate results: `acc = acc + result[0]`
2. Print the accumulator at the end
3. This forces LLVM to actually execute the code

### Prevention
- Benchmark template now includes accumulator pattern
- Always use benchmark results in observable output

---

## Template for Future Entries

```markdown
## [Date] - [Brief Title]

### Context
What were you trying to do?

### What Happened
What went wrong or unexpectedly right?

### Root Cause
Why did this happen?

### Lesson
What should future agents know?

### Prevention
How do we avoid this in future?
```
