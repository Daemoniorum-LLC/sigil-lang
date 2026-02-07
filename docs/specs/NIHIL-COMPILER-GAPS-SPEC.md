# Nihil Compiler Gaps Spec

**Version:** 1.0.0
**Status:** ✅ ALL GAPS IMPLEMENTED
**Date:** 2026-02-06
**Discovery:** Running 41 Nihil ML framework test files through the Sigil compiler with injected `main()` wrappers revealed 3 genuine spec gaps and 3 implementation bugs. **All gaps have now been implemented and pass regression tests.**

---

## 1. Gap Discovery

### 1.1 Context

The Nihil ML framework (`/home/crook/dev/nihil`) is a 37K LOC tensor computation library written in Sigil. It contains 41 test files (`.sg`) with `#[test]` annotated functions and 61 source files (`.sigil`).

Previous documentation claimed 416/416 tests passing. This was based on `sigil run` exit codes on files without `main()`, which only validates parse+typecheck — no assertions ever execute. When test functions are actually invoked, 32/41 files fail at runtime.

**Update 2026-02-06:** All identified gaps have been implemented. Regression tests pass.

### 1.2 Methodology

- **SDD**: Gaps documented before implementation begins
- **Agent-TDD**: Failing tests written to crystallize expected behavior

### 1.3 Gap Categories

| # | Gap | Type | Affected Files | Implementation Status |
|---|-----|------|---------------|----------------------|
| 1 | Index assignment (`arr[i] = val`) | Implementation bug | 7 | ✅ IMPLEMENTED |
| 2 | Float math methods (`.exp()`, `.sqrt()`) | **Spec gap** | 6 | ✅ IMPLEMENTED |
| 3 | Ref pattern destructuring (`&var`) | Implementation bug | 3 | ✅ IMPLEMENTED |
| 4 | Closure/lambda invocation | Implementation bug | 2 | ✅ IMPLEMENTED |
| 5 | Trait associated constants | **Spec gap** | 2 | ✅ IMPLEMENTED |
| 6 | Numeric literal type suffixes (`0.0f32`) | **Spec gap** | ~70 occurrences | ✅ IMPLEMENTED |

---

## 2. Spec Gap A: Numeric Primitive Methods

### 2.1 Discovery

Nihil uses method syntax on float values throughout:

```sigil
≔ x = 2.0;
≔ y = x.exp();     // ERROR: no method 'exp' on type 'Discriminant(3)'
≔ z = x.sqrt();    // ERROR: no method 'sqrt' on type 'Discriminant(3)'
≔ w = x.abs();     // ERROR: no method 'abs' on type 'Discriminant(3)'
≔ p = x.powf(2.0); // ERROR: no method 'powf' on type 'Discriminant(3)'
```

The Sigil spec (09-STDLIB.md, lines 582-596) defines these as module-level free functions in `std·math`:

```sigil
rite sqrt(x: f64) -> f64;
rite exp(x: f64) -> f64;
rite pow(x: f64, n: f64) -> f64;
```

But no spec defines them as instance methods on `f32`/`f64` primitives. Nihil (and Rust convention) expects method syntax.

### 2.2 Specification

Numeric primitive types (`f32`, `f64`) shall support the following instance methods:

#### 2.2.1 f32 Methods

```sigil
⊢ f32 {
    // Exponential / logarithmic
    ☉ rite exp(&self) -> f32;
    ☉ rite exp2(&self) -> f32;
    ☉ rite ln(&self) -> f32;
    ☉ rite log2(&self) -> f32;
    ☉ rite log10(&self) -> f32;

    // Power / root
    ☉ rite sqrt(&self) -> f32;
    ☉ rite cbrt(&self) -> f32;
    ☉ rite powf(&self, n: f32) -> f32;
    ☉ rite powi(&self, n: i32) -> f32;

    // Trigonometric
    ☉ rite sin(&self) -> f32;
    ☉ rite cos(&self) -> f32;
    ☉ rite tan(&self) -> f32;
    ☉ rite asin(&self) -> f32;
    ☉ rite acos(&self) -> f32;
    ☉ rite atan(&self) -> f32;
    ☉ rite atan2(&self, other: f32) -> f32;

    // Rounding
    ☉ rite floor(&self) -> f32;
    ☉ rite ceil(&self) -> f32;
    ☉ rite round(&self) -> f32;
    ☉ rite trunc(&self) -> f32;
    ☉ rite fract(&self) -> f32;

    // Absolute value / sign
    ☉ rite abs(&self) -> f32;
    ☉ rite signum(&self) -> f32;
    ☉ rite copysign(&self, sign: f32) -> f32;

    // Fused multiply-add
    ☉ rite mul_add(&self, a: f32, b: f32) -> f32;

    // Min / max
    ☉ rite max(&self, other: f32) -> f32;
    ☉ rite min(&self, other: f32) -> f32;
    ☉ rite clamp(&self, min: f32, max: f32) -> f32;

    // Classification
    ☉ rite is_nan(&self) -> bool;
    ☉ rite is_infinite(&self) -> bool;
    ☉ rite is_finite(&self) -> bool;
    ☉ rite is_normal(&self) -> bool;

    // Bit conversion
    ☉ rite to_bits(&self) -> u32;
    ☉ rite from_bits(bits: u32) -> f32;
}
```

#### 2.2.2 f64 Methods

Same set as f32 with appropriate type substitutions (`f32` → `f64`, `u32` → `u64`, etc.).

#### 2.2.3 Integer Methods (Minimum Viable)

```sigil
⊢ i32 {
    ☉ rite abs(&self) -> i32;
    ☉ rite pow(&self, exp: u32) -> i32;
    ☉ rite min(&self, other: i32) -> i32;
    ☉ rite max(&self, other: i32) -> i32;
    ☉ rite clamp(&self, min: i32, max: i32) -> i32;
}
// Similarly for i64, u32, u64, usize
```

### 2.3 Implementation Strategy

The interpreter's method dispatch (`eval_method_call`) must recognize float/int primitive types and route to stdlib math functions. The "Discriminant" error indicates floats are wrapped in enum discriminants during method dispatch — this wrapper must be unwrapped before method lookup.

**Compiler components affected:**
- `interpreter.rs`: `eval_method_call` — add float/int primitive method dispatch
- `typeck.rs`: method resolution for primitive types
- `stdlib.rs`: register primitive methods (may already exist as free functions)

### 2.4 Spec Updates Required ✓

- `03-TYPES.md` Section 2.1 — Added cross-reference to 09-STDLIB.md § 9.1
- `09-STDLIB.md` Section 9.1 — Added full numeric primitive method specification

---

## 3. Spec Gap B: Trait Associated Constants

### 3.1 Discovery

Nihil defines trait constants for compile-time type metadata:

```sigil
☉ Θ DType {
    const SIZE: usize;
    const NAME: &'static str;
}

⊢ DType ∀ F16 {
    const SIZE: usize = 2;
    const NAME: &'static str = "f16";
}

// Expected: F16·SIZE evaluates to 2
// Actual: F16·SIZE evaluates to the string "F16::SIZE"
```

The Sigil spec (03-TYPES.md, lines 427-445) documents associated **types** in traits:

```sigil
trait Iterator {
    type Item;
}
```

But associated **constants** (`const NAME: Type = value;`) are not documented anywhere in the spec.

### 3.2 Specification

#### 3.2.1 Syntax

```sigil
// In trait definition
☉ Θ MyTrait {
    const NAME: Type;                    // Required constant (no default)
    const OTHER: Type = default_value;   // Constant with default
}

// In trait implementation
⊢ MyTrait ∀ MyType {
    const NAME: Type = value;            // Provide required constant
    // OTHER inherits default if not overridden
}
```

#### 3.2.2 Access Syntax

```sigil
// Via implementing type (preferred)
MyType·NAME          // Resolves to the constant value from the impl

// Via trait (when generic)
rite use_trait<T: MyTrait>() {
    T·NAME           // Resolves via trait bound
}
```

#### 3.2.3 Semantics

- Constants must be compile-time evaluable expressions
- Constants are monomorphized per implementing type
- `Type·CONST` resolves by searching: (1) inherent impl, (2) trait impls
- Default values can be overridden in implementations

### 3.3 Implementation Strategy

The interpreter resolves `F16·SIZE` as a path expression. Currently, path resolution for `Type·Name` doesn't search trait impl constant declarations — it falls through to stringification.

**Compiler components affected:**
- `interpreter.rs`: Path resolution for `Type·CONST` — must search trait impl blocks for `const` declarations
- `typeck.rs`: Type-level constant resolution in trait bounds
- `parser.rs`: Already parses `const NAME: Type = value;` in trait/impl blocks (verify)

### 3.4 Spec Updates Required ✓

- `03-TYPES.md` Section 7.5 — Added Associated Constants specification

---

## 4. Spec Gap C: Numeric Literal Type Suffixes

### 4.1 Discovery

Nihil uses typed float literals extensively (~70 occurrences):

```sigil
≔ Δ sum = 0.0f32;           // ERROR: Invalid float: 0.0f32
≔ data = [1.0f32, 2.0, 3.0]; // First element fails
2.0f32.powi(-14)              // Method disambiguation requires suffix
```

The Sigil spec (01-LEXICAL.md, line 231) only shows unsuffixed literals:

```sigil
3.14159         // Float
6.022e23        // Scientific
```

No type suffixes are documented.

### 4.2 Specification

#### 4.2.1 Float Suffixes

```
float_literal = digits "." digits? float_suffix?
              | digits float_suffix
              ;

float_suffix  = "f32" | "f64" ;
```

Examples:
```sigil
0.0f32          // f32 zero
1.0f64          // f64 one
3.14f32         // f32 pi approximation
2.0f32.powi(n)  // method call on typed float
```

Without suffix, float literals default to `f64` (current behavior).

#### 4.2.2 Integer Suffixes

```
int_literal   = digits int_suffix?
              | "0x" hex_digits int_suffix?
              | "0b" bin_digits int_suffix?
              | "0o" oct_digits int_suffix?
              ;

int_suffix    = "i8" | "i16" | "i32" | "i64" | "i128"
              | "u8" | "u16" | "u32" | "u64" | "u128"
              | "isize" | "usize"
              ;
```

Examples:
```sigil
42i32           // i32
255u8           // u8
0xFF_u16        // u16 hex
```

Without suffix, integer literals default to `i64` (current behavior).

#### 4.2.3 Disambiguation Rule

When a float suffix is followed by `.`, the suffix binds tighter than the method call:

```sigil
2.0f32.powi(n)  // Parsed as: (2.0f32).powi(n)
```

This is critical for Nihil's numeric code where method calls on typed floats are common.

### 4.3 Implementation Strategy

**Compiler components affected:**
- `lexer.rs`: Extend float/int token parsing to consume optional type suffix
- `parser.rs`: Propagate suffix type information to AST literal nodes
- `typeck.rs`: Use suffix for literal type inference instead of defaulting
- `interpreter.rs`: Respect suffix type during evaluation

### 4.4 Spec Updates Required ✓

- `01-LEXICAL.md` Section 4.2.1 — Added Typed Numeric Literals specification

---

## 5. Implementation Bugs (Specced but Not Implemented)

These are not spec gaps — the behavior is already specified. They need implementation fixes and regression tests.

### 5.1 Index Assignment

**Spec reference:** 09-STDLIB.md — `IndexMut` trait in prelude
**Error:** `Invalid index assignment target`
**Fix location:** `interpreter.rs` — `eval_assign` must handle `Expr::Index` on LHS

### 5.2 Ref Pattern Destructuring

**Spec reference:** 02-SYNTAX.md line 454, 02A-PATTERN-MATCHING.md line 32
**Error:** `Unsupported pattern: Ref`
**Fix location:** `interpreter.rs` — pattern matching must handle `Pattern::Ref`

### 5.3 Closure/Lambda Invocation

**Spec reference:** 02-SYNTAX.md line 360, 03-TYPES.md line 342
**Error:** `Cannot call non-function`
**Fix location:** `interpreter.rs` — function call dispatch must check for closure values

---

## 6. Test Plan (Agent-TDD)

Regression tests at `jormungandr/tests/spec/25_nihil_gaps/`:

| Test File | Gap | Tests | Status |
|-----------|-----|-------|--------|
| `test_float_methods.sg` | A | 6 | ✅ PASS |
| `test_trait_constants.sg` | B | 4 | ✅ PASS |
| `test_float_suffixes.sg` | C | 4 | ✅ PASS |
| `test_index_assign.sg` | 5.1 | 3 | ✅ PASS |
| `test_ref_patterns.sg` | 5.2 | 3 | ✅ PASS |
| `test_closure_ref_pattern.sg` | 5.5 | 2 | ✅ PASS |
| `test_generic_type_params.sg` | D | 3 | ✅ PASS |
| `test_const_generic_basic.sg` | - | 1 | ✅ PASS |
| `test_const_generic_shape.sg` | - | 1 | ✅ PASS |
| `test_const_generic_arithmetic.sg` | - | 1 | ✅ PASS |
| `test_const_generic_assoc_const.sg` | - | 1 | ✅ PASS |
| `test_const_generic_ctor.sg` | - | 1 | ✅ PASS |
| `test_float_ieee754.sg` | - | 1 | ✅ PASS |
| `test_vec_equality.sg` | - | 1 | ✅ PASS |
| `test_vec_reverse.sg` | - | 1 | ✅ PASS |
| `test_struct_to_string.sg` | - | 1 | ✅ PASS |
| `test_sequential_ctor.sg` | - | 1 | ✅ PASS |

**Total: 17 test files, 35+ test cases, ALL PASSING**

Run tests:
```bash
cd jormungandr/tests/spec/25_nihil_gaps
for f in *.sg; do ../../parser/target/release/sigil run "$f"; done
```

---

## 7. Implementation Status Summary

All gaps have been implemented and pass regression tests:

| Priority | Gap | Impact | Status |
|----------|-----|--------|--------|
| P1 | Index assignment | 7 test files | ✅ IMPLEMENTED |
| P1 | Float math methods | 6 test files | ✅ IMPLEMENTED |
| P1 | Ref pattern destructuring | 3 test files | ✅ IMPLEMENTED |
| P2 | Closure invocation | 2 test files | ✅ IMPLEMENTED |
| P2 | Trait associated constants | 2 test files | ✅ IMPLEMENTED |
| P2 | Float literal suffixes | ~70 occurrences | ✅ IMPLEMENTED |
| P2 | Generic type parameters | - | ✅ IMPLEMENTED |
| P2 | Closure ref patterns | - | ✅ IMPLEMENTED |

---

## 8. Spec Gap D: Generic Type Parameter Resolution

### 8.1 Discovery

Generic functions that reference trait associated constants via type parameters fail at runtime:

```sigil
☉ Θ DType {
    const SIZE: usize;
    const NAME: &'static str;
    rite size() -> usize { Self·SIZE }
    rite name() -> &'static str { Self·NAME }
}

⊢ DType ∀ F16 {
    const SIZE: usize = 2;
    const NAME: &'static str = "f16";
}

rite dtype_info<T: DType>() -> String {
    format("{} ({} bytes)", T·NAME, T·SIZE)
}

// Call:
≔ info = dtype_info·<F16>();
// Expected: "f16 (2 bytes)"
// Actual:   "T::NAME (T::SIZE bytes)"
```

### 8.2 Root Cause

The interpreter has no mechanism for generic type parameter scope:

1. **Function values lose generic info**: `create_function()` does not preserve the generic parameter names from the AST `Function.generics` field
2. **Turbofish type args are available but unused**: The parser correctly puts generics on `PathSegment.generics`, but `eval_call` does not extract or bind them
3. **Path resolution is literal**: `T·NAME` resolves as the string `"T·NAME"` rather than substituting `T` with the bound concrete type `"F16"` to look up `"F16·NAME"`

### 8.3 Expected Behavior

When calling a generic function with turbofish syntax `func·<ConcreteType>(args)`:

1. The type arguments are extracted from the call expression's path segments
2. They are matched positionally to the function's generic parameter names
3. During function body execution, multi-segment paths like `T·CONST` resolve `T` to the concrete type
4. After function returns, generic bindings are restored

### 8.4 Regression Test

See `jormungandr/tests/spec/25_nihil_gaps/test_generic_type_params.sg`

---

## 9. Implementation Bug 5.5: Closure Ref Pattern Destructuring

### 9.1 Discovery

Closures with ref pattern parameters (`|&x|`) fail because the parameter name extraction only handles `Pattern::Ident`, discarding all other patterns as `"_"`:

```sigil
≔ items = vec![1, 2, 3];
≔ pos = items.iter().position(|&x| x == 2);
// ERROR: undefined variable: `x`
// Because |&x| → param name "_", x is never bound
```

### 9.2 Root Cause

In `eval_closure` (interpreter.rs:17155), closure parameter name extraction:
```rust
.map(|p| match &p.pattern {
    Pattern::Ident { name, .. } => name.name.clone(),
    _ => "_".to_string(),  // ← All non-Ident patterns become "_"
})
```

`Pattern::Ref { pattern: Box<Pattern::Ident { name: "x" }> }` falls through to `"_"`.

### 9.3 Expected Behavior

Closure ref patterns should recursively extract the inner identifier name. For `|&x|`, the parameter name is `"x"` and the argument is auto-dereferenced when bound.

### 9.4 Regression Test

See `jormungandr/tests/spec/25_nihil_gaps/test_closure_ref_pattern.sg`

---

## 10. Relationship to Other Specs

- **01-LEXICAL.md**: Numeric literal suffixes (Gap C)
- **03-TYPES.md**: Trait associated constants (Gap B), numeric primitive types
- **03B-TYPECK-GENERIC-INFERENCE.md**: Generic type parameter resolution (Gap D)
- **09-STDLIB.md**: Numeric methods (Gap A), IndexMut trait
- **02-SYNTAX.md**: Patterns (Bug 5.2), closures (Bug 5.3), nested variable capture (Bug 5.5)
- **02A-PATTERN-MATCHING.md**: Ref patterns (Bug 5.2)
- **COMPILER-ISSUES.md** (Nihil): Cross-references this spec

---

## Revision History

| Version | Date | Changes |
|---------|------|---------|
| 0.1.0 | 2026-02-04 | Initial gap discovery. 3 spec gaps and 3 implementation bugs identified from Nihil runtime validation. |
| 0.2.0 | 2026-02-04 | Added Gap D (generic type parameter resolution) and Bug 5.5 (nested variable capture). Nihil at 28/41 (68%). |
| 1.0.0 | 2026-02-06 | **ALL GAPS IMPLEMENTED.** Updated status to reflect all 17 regression tests passing. Spec complete. |
