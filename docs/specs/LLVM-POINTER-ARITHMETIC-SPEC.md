# LLVM Pointer Arithmetic Specification

**Version:** 0.1.0
**Status:** GAP IDENTIFIED
**Date:** 2026-02-08
**Priority:** P0 (Blocking benchmarks, systems programming)

---

## 1. Gap Description

### 1.1 Observed Behavior

When compiling pointer arithmetic expressions like `*(arr + i)` where `arr` is a typed pointer (e.g., `*i64`), the LLVM codegen produces **incorrect byte-offset arithmetic** instead of **element-offset arithmetic**.

```sigil
≔ arr = alloc(80) as *i64;  // 10 i64 elements
*(arr + 5) = 100;            // SHOULD write to byte offset 40 (5 * 8)
                             // ACTUALLY writes to byte offset 5
```

### 1.2 Root Cause

The LLVM codegen erases all type information to `i64`:

1. `CompileScope` only stores `HashMap<String, PointerValue>` - no type metadata
2. `compile_binary_op(BinOp::Add, lhs, rhs)` receives raw `IntValue`s with no type context
3. Binary addition compiles to `build_int_add(lhs, rhs)` - raw integer addition
4. When `lhs` is a pointer address and `rhs` is an index, no scaling occurs

### 1.3 Expected Behavior (C semantics)

In C, pointer arithmetic automatically scales by element size:

```c
int64_t* arr = malloc(80);
arr[5] = 100;        // Writes to byte offset 40
*(arr + 5) = 100;    // Equivalent, writes to byte offset 40
```

Sigil should match C pointer arithmetic semantics.

---

## 2. Specification

### 2.1 Pointer Arithmetic Semantics

For pointer type `*T` and integer offset `n`:

| Expression | Byte Offset |
|------------|-------------|
| `ptr + n` | `ptr_addr + n * sizeof(T)` |
| `ptr - n` | `ptr_addr - n * sizeof(T)` |
| `*(ptr + n)` | Load from `ptr_addr + n * sizeof(T)` |

### 2.2 Type Sizes

| Type | Size (bytes) |
|------|--------------|
| `i8`, `u8` | 1 |
| `i16`, `u16` | 2 |
| `i32`, `u32`, `f32` | 4 |
| `i64`, `u64`, `f64` | 8 |
| `i128`, `u128` | 16 |
| `*T` (pointer) | 8 (on 64-bit) |

### 2.3 Type Tracking Requirements

The LLVM codegen must track type information for:

1. **Variable declarations**: `≔ arr: *i64 = ...` stores type `*i64` in scope
2. **Cast expressions**: `alloc(n) as *i64` produces typed pointer
3. **Binary operations**: When one operand is `*T` and other is integer, scale by `sizeof(T)`

---

## 3. Implementation Plan

### Phase 1: Extend CompileScope with Type Tracking

```rust
struct CompileScope<'ctx> {
    vars: HashMap<String, PointerValue<'ctx>>,
    var_types: HashMap<String, SigilType>,  // NEW: track original types
}

enum SigilType {
    I64,
    F64,
    Ptr(Box<SigilType>),  // *T
    // ... other types
}
```

### Phase 2: Track Types Through Expressions

When compiling expressions, return `(IntValue, Option<SigilType>)` instead of just `IntValue`:

```rust
fn compile_expr(...) -> Result<(IntValue<'ctx>, Option<SigilType>), String>
```

### Phase 3: Pointer-Aware Binary Operations

```rust
fn compile_binary_op(
    op: BinOp,
    lhs: IntValue<'ctx>,
    lhs_type: Option<SigilType>,
    rhs: IntValue<'ctx>,
    rhs_type: Option<SigilType>,
) -> Result<(IntValue<'ctx>, Option<SigilType>), String> {
    match (op, &lhs_type, &rhs_type) {
        (BinOp::Add, Some(SigilType::Ptr(elem)), _) => {
            // Pointer + integer: scale by element size
            let scale = elem.size_bytes();
            let scaled_rhs = build_int_mul(rhs, const_i64(scale));
            let result = build_int_add(lhs, scaled_rhs);
            Ok((result, lhs_type))
        }
        (BinOp::Add, _, Some(SigilType::Ptr(elem))) => {
            // Integer + pointer: scale lhs
            let scale = elem.size_bytes();
            let scaled_lhs = build_int_mul(lhs, const_i64(scale));
            let result = build_int_add(scaled_lhs, rhs);
            Ok((result, rhs_type))
        }
        _ => {
            // Regular integer arithmetic
            Ok((build_int_add(lhs, rhs), None))
        }
    }
}
```

### Phase 4: Alternative - Use GEP for Dereference

Detect `*(ptr + offset)` pattern and use LLVM GEP instruction:

```rust
// When compiling Expr::Deref(inner):
match inner {
    Expr::Binary { op: BinOp::Add, left, right } => {
        // Check if left is pointer type
        if let Some(SigilType::Ptr(elem_type)) = get_type(left) {
            let base_ptr = compile_expr(left)?;
            let index = compile_expr(right)?;
            // Use GEP which handles scaling
            let elem_ptr = build_gep(elem_type, base_ptr, &[index]);
            return build_load(elem_type, elem_ptr);
        }
    }
    _ => { /* regular dereference */ }
}
```

---

## 4. Test Suite

### 4.1 Specification Tests

```sigil
// test_ptr_arithmetic_i64.sg
#[test]
rite test_ptr_add_i64() {
    ≔ arr = alloc(80) as *i64;
    *(arr + 0) = 10;
    *(arr + 1) = 20;
    *(arr + 2) = 30;

    assert_eq(*(arr + 0), 10);
    assert_eq(*(arr + 1), 20);
    assert_eq(*(arr + 2), 30);

    free(arr as *u8);
}

#[test]
rite test_ptr_arithmetic_loop() {
    ≔ n: i64 = 10;
    ≔ arr = alloc(n * 8) as *i64;

    // Write 0..9
    ≔ vary i: i64 = 0;
    ⟳ i < n {
        *(arr + i) = i;
        i = i + 1;
    }

    // Verify
    i = 0;
    ⟳ i < n {
        assert_eq(*(arr + i), i);
        i = i + 1;
    }

    free(arr as *u8);
}

#[test]
rite test_ptr_arithmetic_i32() {
    ≔ arr = alloc(40) as *i32;  // 10 i32 elements
    *(arr + 5) = 42;
    assert_eq(*(arr + 5), 42);
    free(arr as *u8);
}
```

### 4.2 Boundary Tests

```sigil
#[test]
rite test_ptr_first_element() {
    ≔ arr = alloc(8) as *i64;
    *(arr + 0) = 999;
    assert_eq(*arr, 999);  // *arr == *(arr + 0)
    free(arr as *u8);
}

#[test]
rite test_ptr_last_element() {
    ≔ arr = alloc(80) as *i64;  // 10 elements
    *(arr + 9) = 123;
    assert_eq(*(arr + 9), 123);
    free(arr as *u8);
}
```

---

## 5. Success Criteria

1. All specification tests pass
2. Memory benchmark runs without SIGSEGV
3. Matrix multiplication benchmark produces correct results
4. No performance regression on existing benchmarks

---

## 6. Revision History

| Version | Date | Changes |
|---------|------|---------|
| 0.1.0 | 2026-02-08 | Initial gap documentation during benchmark investigation |
