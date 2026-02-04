# Assert API & Variadic Builtin Functions Spec

**Version:** 0.1.0
**Status:** ! Draft
**Date:** 2026-02-03
**Discovery:** Running 36 uncounted top-level test files revealed 18 failures from `assert(expr, "message")` and `format!(..., arg1, arg2)` calls that the type checker rejects as wrong arity.

---

## 1. Gap Discovery

### 1.1 Context

The Sigil test runner (`run_tests_rust.sh`) scans `features/`, `stdlib/`, `integration/`, and `spec/*/` directories. 36 test files in the top-level `jormungandr/tests/` directory and 4 in `daemon/tests/` were never included. Running these revealed:

- **18 files** fail with `expected 1 arguments, found 2` or `expected 1 arguments, found 3`
- Root cause: these tests use `assert(expr, "message")` (2-arg) and `format!("template", arg1, arg2)` (variadic)
- The type checker treats `assert` as `assert(bool) -> ()` (1-arg only)
- The type checker does not expand `format!` before argument counting

### 1.2 Affected Files

| File | Error Pattern | Cause |
|------|---------------|-------|
| `test_simple.sg` | `assert(x, "msg")` | 2-arg assert |
| `test_codegen.sg` | `panic(format!(..., a, b))` | format! not expanded before typeck |
| `test_compile.sg` | `assert(x, "msg")` | 2-arg assert |
| `test_evidentiality.sg` | `assert(x, "msg")` | 2-arg assert |
| `test_lowering.sg` | `assert(x, "msg")` | 2-arg assert |
| `test_parser_expressions.sg` | `assert_eq(a, b)` + type mismatch | assert_eq arg types |
| `test_parser_generics.sg` | 3-arg custom assert helpers | format! expansion |
| `test_parser_items.sg` | 3-arg custom assert helpers | format! expansion |
| `test_parser_patterns.sg` | 3-arg custom assert helpers | format! expansion |
| `test_protocols.sg` | 3-arg custom assert helpers | format! expansion |
| `test_simd.sg` | 3-arg custom assert helpers | format! expansion |
| `test_typeck_inference.sg` | 3-arg custom assert helpers | format! expansion |
| `test_typeck_traits.sg` | 3-arg custom assert helpers | format! expansion |
| `test_atomics.sg` | 3-arg custom assert helpers | format! expansion |
| `test_asm.sg` | 3-arg custom assert helpers | format! expansion |
| `test_cg009_uncertain_pattern.sg` | `assert(x, "msg")` | 2-arg assert |
| `test_cg025_evidential_extraction.sg` | `assert(x, "msg")` | 2-arg assert |
| `test_closure_capture.sg` | (passes - included for reference) | N/A |

---

## 2. Specification

### 2.1 Assert Overloads

The `assert` builtin must support multiple signatures:

```sigil
// Current (working):
assert(condition: !bool)                    // panics if false

// Required:
assert(condition: !bool, message: !str)     // panics with message if false
```

### 2.2 Assert Equality Functions

These should be stdlib builtins (already partially implemented for spec tests):

```sigil
assert_eq<T: Eq>(left: T, right: T)                    // panics if left != right, shows diff
assert_eq<T: Eq>(left: T, right: T, message: !str)     // panics with message + diff
assert_ne<T: Eq>(left: T, right: T)                    // panics if left == right
assert_ne<T: Eq>(left: T, right: T, message: !str)     // panics with message
```

**Note:** `assert_eq` already works in `stdlib/` test files. The gap is specifically about `assert(bool, str)` and `format!` macro expansion in the type checker.

### 2.3 Format Macro Type Checking

The `format!` macro must be expanded before argument arity checking:

```sigil
// This currently fails type checking because typeck sees 3 args to panic():
panic(format!("FAIL {}: expected '{}'", name, pattern));

// After fix, format! expands to a single String, so panic() sees 1 arg:
panic(/* String from format expansion */);
```

**Root cause in compiler:** The type checker (`typeck.rs`) counts arguments to function calls but encounters `format!` as an unexpanded macro expression. The macro expander runs after or alongside typeck, creating a mismatch.

**Required behavior:** Either:
1. Expand `format!` before type checking (preferred), or
2. Teach the type checker that `format!(..., args)` produces a single `String` regardless of arg count

---

## 3. Implementation Strategy

### Phase 1: `assert(bool, str)` overload

| Component | Change |
|-----------|--------|
| `interpreter.rs` | Handle 2-arg `assert` calls, use second arg as panic message |
| `typeck.rs` | Accept `assert` with 1 or 2 arguments |
| `stdlib.rs` | Update `assert` intrinsic registration |

**Unblocks:** ~8 test files that use `assert(expr, "message")` pattern

### Phase 2: `format!` macro expansion before typeck

| Component | Change |
|-----------|--------|
| `typeck.rs` | Expand macros (especially `format!`) before argument arity checking |
| `interpreter.rs` | Already handles `format!` at runtime; no change needed |

**Unblocks:** ~10 test files that use `panic(format!(...))` and similar patterns

### Phase 3: Verify and integrate

- Move all 36 top-level test files into `spec/` or a new `regression/` category in the test runner
- Update `run_tests_rust.sh` to also scan top-level `*.sg` files
- Update INTERPRETER-SPEC-ROADMAP.md with new test count

---

## 4. Success Criteria

| Metric | Current | Target |
|--------|---------|--------|
| Top-level test pass rate | 10/36 (28%) | 30+/36 (83%+) |
| Total test count (runner) | 749 | 785+ (749 + 36 top-level) |
| `assert(x, "msg")` | Type error | Works |
| `panic(format!(...))` | Type error | Works |

---

## 5. Relationship to Other Specs

- **09-STDLIB.md**: Assert functions are stdlib builtins
- **03-TYPES.md**: Overload resolution for `assert` signatures
- **07-METAPROGRAMMING.md**: `format!` macro expansion timing
- **INTERPRETER-SPEC-ROADMAP.md**: Test count and wave tracking

---

## Revision History

| Version | Date | Changes |
|---------|------|---------|
| 0.1.0 | 2026-02-03 | Initial gap discovery. 18/36 uncounted tests fail on assert arity and format! expansion. |
