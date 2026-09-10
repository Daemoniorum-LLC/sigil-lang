# WASM Range Expression Specification

**Version:** 0.1.0
**Status:** Draft
**Date:** 2026-02-14
**Parent Spec:** 02-SYNTAX.md (Range expression syntax)

---

## Abstract

This specification defines the implementation of range expressions (`0..n`, `[1..]`,
`..=5`) for the Sigil WASM backend. Ranges are used for iteration and slicing operations
in qliphoth and other web-targeting Sigil code.

---

## 1. Conceptual Foundation

### 1.1 The Problem

qliphoth uses range expressions in two contexts:

```sigil
// Slicing: extract substring from index 1 to end
let path! = if hash·starts_with("#") { &hash[1..] } else { "/" };

// Iteration: loop from 0 to length
(0..self·length())
```

Currently, the WASM backend returns "unsupported: range expressions".

### 1.2 Range Variants

| Syntax | Name | Start | End | Inclusive |
|--------|------|-------|-----|-----------|
| `a..b` | Range | a | b | false |
| `a..=b` | RangeInclusive | a | b | true |
| `a..` | RangeFrom | a | ∞ | false |
| `..b` | RangeTo | 0 | b | false |
| `..=b` | RangeToInclusive | 0 | b | true |
| `..` | RangeFull | 0 | ∞ | false |

### 1.3 Design Goals

- **Slicing support:** Enable `array[start..end]` and `string[start..]`
- **Iteration support:** Enable `for i in 0..n` patterns
- **Minimal overhead:** Ranges should compile to simple start/end values

---

## 2. Type Architecture

### 2.1 Range Representation

```
Range representation in WASM:

Option A: Two i64 values on stack
    [start: i64] [end: i64]
    Note: inclusive flag is compile-time only

Option B: Heap-allocated struct (16 bytes)
    ┌─────────────────────────────────────┐
    │ start (i64)                         │  offset 0
    ├─────────────────────────────────────┤
    │ end (i64)                           │  offset 8
    └─────────────────────────────────────┘

For unbounded ranges (RangeFrom, RangeTo, RangeFull):
    - Missing start defaults to 0
    - Missing end defaults to i64::MAX or contextual length
```

**Decision:** Use Option A (stack values) for simple ranges used in slicing.
For iteration, ranges can be desugared to loop constructs at compile time.

### 2.2 Slicing Context

When a Range appears as an Index argument, it becomes a slice operation:

```
string[start..end] → string_slice(string, start, end)
string[start..]    → string_slice(string, start, string_length(string))
string[..end]      → string_slice(string, 0, end)
```

---

## 3. Behavioral Contracts

### 3.1 Range Expression Compilation

```
compile_range(start, end, inclusive):
    // Evaluate start (or use default)
    if start is Some:
        compile_expr(start)
    else:
        push i64.const 0

    // Evaluate end (or use sentinel)
    if end is Some:
        compile_expr(end)
        if inclusive:
            i64.const 1
            i64.add        // end + 1 for inclusive
    else:
        push i64.const -1  // Sentinel for "to end"
```

### 3.2 Index with Range (Slicing)

```
compile_index(array, Range { start, end, inclusive }):
    compile_expr(array)           // Push array/string pointer

    // Compile start
    if start is Some:
        compile_expr(start)
    else:
        i64.const 0

    // Compile end
    if end is Some:
        compile_expr(end)
        if inclusive:
            i64.const 1
            i64.add
    else:
        // Use array/string length
        local.get array_ptr
        call array_length

    // Call appropriate slice function
    call slice_function
```

### 3.3 Invariants

```
I1: start ≤ end (runtime: bounds check or trap)
I2: start ≥ 0 (compile-time: i64 is always valid)
I3: end ≤ length (runtime: bounds check)
I4: Inclusive ranges: end is adjusted by +1 at compile time
```

---

## 4. Implementation Strategy

### 4.1 Phase 1: Range as Slicing Index

Handle the most common case: ranges used for slicing strings/arrays.

```rust
fn compile_range(&mut self, start: Option<&Expr>, end: Option<&Expr>, inclusive: bool) -> WasmResult<()> {
    // Push start value
    if let Some(s) = start {
        self.compile_expr(s)?;
    } else {
        self.push_i64(0);
    }

    // Push end value (or sentinel for "to end")
    if let Some(e) = end {
        self.compile_expr(e)?;
        if inclusive {
            self.push_i64(1);
            self.emit_add();
        }
    } else {
        self.push_i64(-1); // Sentinel
    }

    Ok(())
}
```

### 4.2 Phase 2: Index Expression Update

Modify `compile_index` to detect Range indices:

```rust
fn compile_index(&mut self, array: &Expr, index: &Expr) -> WasmResult<()> {
    match index {
        Expr::Range { start, end, inclusive } => {
            self.compile_slice(array, start.as_deref(), end.as_deref(), *inclusive)
        }
        _ => {
            // Existing single-index logic
            self.compile_simple_index(array, index)
        }
    }
}
```

### 4.3 Runtime Imports

Required string/array slice imports:

```wasm
(import "string" "slice" (func $string_slice (param i32 i32 i32) (result i32)))
(import "array" "slice" (func $array_slice (param i32 i32 i32) (result i32)))
```

---

## 5. Edge Cases

### 5.1 Unbounded Ranges

| Expression | Compiled As |
|------------|-------------|
| `arr[..]` | `slice(arr, 0, len(arr))` |
| `arr[n..]` | `slice(arr, n, len(arr))` |
| `arr[..n]` | `slice(arr, 0, n)` |

### 5.2 Empty Ranges

```sigil
arr[5..5]   // Empty slice (valid)
arr[5..3]   // Invalid: start > end (trap or empty)
```

### 5.3 Inclusive Ranges

```sigil
arr[0..=2]  // Elements 0, 1, 2 (end = 3 internally)
'a'..='z'   // In patterns only, not index expressions
```

---

## 6. Error Conditions

| Condition | Error |
|-----------|-------|
| Range outside slice context | "range expressions require slice context" |
| Negative start | Runtime trap (i64 overflow) |
| start > end | Runtime trap or empty slice |
| end > length | Runtime trap (bounds check) |

---

## 7. Integration Points

### 7.1 With String Slicing

```sigil
let suffix = &path[1..];  // String slice from index 1

// Compiled:
local.get $path
i64.const 1
local.get $path
call $string_length
call $string_slice
```

### 7.2 With Array Slicing

```sigil
let sub = items[2..5];  // Array slice

// Compiled:
local.get $items
i64.const 2
i64.const 5
call $array_slice
```

### 7.3 With For Loops (Future)

```sigil
for i in 0..n {
    // Loop body
}

// Desugars to:
let mut i = 0;
while i < n {
    // Loop body
    i += 1;
}
```

---

## 8. Open Questions

1. **Bounds checking:** Trap on out-of-bounds, or return empty slice?
   - Current: Delegate to runtime slice functions

2. **Negative indices:** Support Python-style `arr[-1]`?
   - Current: Not supported

3. **Step ranges:** Support `0..10 step 2`?
   - Current: Not supported, future feature

---

## 9. Implementation Status

| Feature | Status | Notes |
|---------|--------|-------|
| `compile_range()` | :white_check_mark: | Stack-based start/end values |
| `compile_slice()` | :white_check_mark: | String slicing via runtime imports |
| `compile_index()` detection | :white_check_mark: | Detects Range indices |
| `string_slice` import | :white_check_mark: | Runtime function call |
| `string_length` import | :white_check_mark: | For unbounded ranges |
| Bounded ranges `a..b` | :white_check_mark: | |
| Unbounded from `a..` | :white_check_mark: | Uses string_length |
| Unbounded to `..b` | :white_check_mark: | Defaults start to 0 |
| Inclusive ranges `..=b` | :white_check_mark: | Adjusts end by +1 |
| For loop iteration | :x: | Future: desugar to while |

---

## 10. Revision History

| Version | Date | Changes |
|---------|------|---------|
| 0.1.0 | 2026-02-14 | Initial draft. Gap discovered during qliphoth compilation. |
| 0.2.0 | 2026-02-14 | Implemented range compilation and slice operations. |
