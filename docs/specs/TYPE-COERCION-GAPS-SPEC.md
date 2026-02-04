# Type Coercion & Inference Gaps Spec

**Version:** 0.1.0
**Status:** ! Draft
**Date:** 2026-02-03
**Discovery:** Running uncounted top-level test files revealed type system gaps where reasonable implicit conversions are rejected by the type checker.

---

## 1. Gap Discovery

### 1.1 Context

Several test files fail with type errors that represent missing coercions or inference gaps in `typeck.rs`. These are distinct from the assert-arity issue (see ASSERT-API-SPEC.md) — even fixing assert overloads would leave these files failing.

### 1.2 Observed Errors

| Error | File(s) | Frequency |
|-------|---------|-----------|
| `expected str!, found String` | `test_cg001_field_access.sg` | 1 file |
| `expected Slice(T), found Array { element: T, size: N }` | `test_lexer.sg`, `test_try_operator.sg` | 2 files |
| `comparison operands must have same type: left=Result<T,E>, right=Int` | `test_try_operator.sg` | 1 file |
| `comparison operands must have same type: left=Int(I32), right=Unit` | `test_evidentiality.sg` | 1 file |
| `Array { element: Int(I64), size: 5 }, expected Slice(Int(I32))` | `test_try_operator.sg` | 1 file |

---

## 2. Gap Analysis

### 2.1 `String` to `str!` Coercion

**Severity:** P0
**Impact:** Any function taking `str!` cannot receive a `String` value

**Current behavior:**
```sigil
rite greet(name: !str) { ... }

≔ s = String·from("hello");
greet(s);  // ERROR: expected str!, found String
```

**Expected behavior:** `String` should implicitly coerce to `str!` (known string reference). This is analogous to Rust's `String` → `&str` deref coercion.

**Implementation:** Add a coercion rule in `typeck.rs`:
- `String` → `str!` (evidential string): always valid
- `String` → `str?` (uncertain string): always valid
- `String` → `str~` (estimated string): always valid

This is a deref-like coercion: `String` owns string data, `str` is a reference to string data. The evidentiality marker on `str` indicates the certainty level of the *reference*, not the data.

### 2.2 Array to Slice Coercion

**Severity:** P1
**Impact:** Fixed-size arrays cannot be passed to functions expecting slices

**Current behavior:**
```sigil
≔ tokens: ![Token] = [];  // Slice type annotation
// ERROR: expected Slice(Token), found Array { element: TypeVar, size: 0 }
```

**Expected behavior:** `[T; N]` (fixed-size array) should coerce to `[T]` (slice) in:
- Let bindings with explicit slice type annotations
- Function arguments expecting slices
- Return positions with slice return types

**Implementation:** Add coercion rules in `typeck.rs`:
- `Array { element: T, size: N }` → `Slice(T)`: always valid (array is a slice with known length)
- Ensure element type `T` also unifies (including integer width coercion)

### 2.3 Integer Literal Width Inference

**Severity:** P1
**Impact:** Integer literals default to `i64`, causing mismatches with `i32` contexts

**Current behavior:**
```sigil
≔ good_list: ![i32] = [1, 2, 3, 4, 5];
// ERROR: expected Slice(Int(I32)), found Array { element: Int(I64), size: 5 }
```

**Expected behavior:** Integer literals should infer their width from context:
- `≔ x: !i32 = 42` → literal `42` infers as `i32`
- `≔ arr: ![i32] = [1, 2, 3]` → literals infer as `i32`
- `foo(42)` where `foo(x: !i32)` → literal infers as `i32`
- Without context, default to `i64` (current behavior, fine)

**Implementation:** During type unification, when a literal integer type (`I64` by default) meets a concrete integer context (`I32`, `I16`, `I8`, `U32`, etc.), and the literal value fits in the target type, coerce the literal's type to match.

### 2.4 Result Comparison

**Severity:** P2
**Impact:** Cannot compare `Result<T, E>` directly with expected values

**Current behavior:**
```sigil
≔ result: !Result<i32, str> = Ok(42);
assert(result == 42);  // ERROR: can't compare Result<i32,str> with Int(I64)
```

**Expected behavior:** This is arguably correct to reject — comparing a `Result` with a bare integer is semantically ambiguous. The test should use:
```sigil
assert(result.unwrap() == 42);
// or
assert(result == Ok(42));
```

**Recommendation:** This is a test file issue, not a type system gap. The test files should be updated to use `result.unwrap()` or `result == Ok(value)` patterns. No compiler change needed.

### 2.5 `i32` vs `Unit` Comparison

**Severity:** P1
**Impact:** Expressions expected to return `i32` are inferred as `Unit`

**Current behavior:**
```sigil
// In test_evidentiality.sg:
// comparison operands must have same type: left=Int(I32), right=Unit
```

**Analysis:** This indicates an expression that should return `i32` is being inferred as `Unit` (void). Common causes:
- Block expression without trailing expression (last statement is `;`-terminated → Unit)
- Match arm returning different types
- Evidentiality extraction (`!expr`) not propagating the inner type

**Recommendation:** Investigate specific occurrences. Likely related to evidentiality extraction where `!expr` on an `Option<i32>` or similar should produce `i32` but produces `Unit`.

---

## 3. Priority

| Gap | Priority | Files Unblocked | Difficulty |
|-----|----------|-----------------|------------|
| `String` → `str!` coercion | **P0** | 1+ | Medium (typeck coercion rules) |
| Array → Slice coercion | **P1** | 2+ | Medium (typeck unification) |
| Integer literal width inference | **P1** | 2+ | Medium (typeck context propagation) |
| `i32` vs `Unit` investigation | **P1** | 1+ | Low (investigation) |
| Result comparison | **P2** | 0 (test fix) | N/A |

---

## 4. Implementation Strategy

### Phase 1: String → str coercion (P0)

| Component | Change |
|-----------|--------|
| `typeck.rs` | In type unification, when expected type is `Evidential { inner: Str, evidence: _ }` and found type is `Named { name: "String" }`, allow coercion |
| Tests | `test_cg001_field_access.sg` should pass |

### Phase 2: Array → Slice coercion + integer literal inference (P1)

| Component | Change |
|-----------|--------|
| `typeck.rs` | In type unification, when expected type is `Slice(T)` and found type is `Array { element: U, size: N }`, unify `T` with `U` and allow |
| `typeck.rs` | When unifying integer types, if one side is a literal (inferred `I64`), coerce to the other side's concrete integer type if the value fits |
| Tests | `test_lexer.sg` (partially), `test_try_operator.sg` (partially) |

### Phase 3: Investigate Unit inference gaps (P1)

| Component | Change |
|-----------|--------|
| Investigation | Read specific lines in `test_evidentiality.sg` where `i32` vs `Unit` occurs |
| Spec update | Document findings and update this spec |

---

## 5. Relationship to Other Specs

- **03-TYPES.md**: Core type system rules
- **03A-TYPE-INFERENCE.md**: Inference algorithm
- **03B-TYPECK-GENERIC-INFERENCE.md**: Generic inference enhancements (recent work)
- **EVIDENTIALITY-RULES.md**: Evidentiality marker propagation
- **ASSERT-API-SPEC.md**: Many affected files also have assert-arity issues; both specs must be resolved for full pass

---

## 6. Interaction with Assert API Spec

Most test files affected by coercion gaps **also** have assert-arity errors. The dependency is:

```
test_lexer.sg:
  - Array → Slice coercion (this spec)
  - assert(bool, str) overload (ASSERT-API-SPEC.md)
  → Both must be fixed for this file to pass

test_try_operator.sg:
  - Integer literal inference (this spec)
  - Array → Slice coercion (this spec)
  - assert(bool, str) overload (ASSERT-API-SPEC.md)
  → All three must be fixed

test_cg001_field_access.sg:
  - String → str! coercion (this spec)
  → Only this spec needed

test_evidentiality.sg:
  - i32 vs Unit investigation (this spec)
  - assert(bool, str) overload (ASSERT-API-SPEC.md)
  → Both specs needed
```

---

## Revision History

| Version | Date | Changes |
|---------|------|---------|
| 0.1.0 | 2026-02-03 | Initial gap discovery. 5 distinct type coercion/inference gaps identified from uncounted test files. |
| 0.1.1 | 2026-02-04 | Note: ASSERT-API-SPEC Phase 1 (variadic arity) now complete. These type coercion gaps are the next frontier for uncounted test improvements. |
