# TDD Wave 1 Execution Plan

**Target:** P0 Type Validation + P0 Trait System
**Goal:** Increase test pass rate from 84% to 90%
**Timeline:** 3 Sprints

---

## SDD Lifecycle Checklist

Each feature follows this checklist:

```
[ ] 1. SPEC      - Formal specification written
[ ] 2. TEST      - Failing tests created (RED)
[ ] 3. IMPLEMENT - Code written to pass tests (GREEN)
[ ] 4. REFACTOR  - Code cleaned up
[ ] 5. DOCUMENT  - Docs updated
[ ] 6. INTEGRATE - No regressions verified
```

---

## Sprint 1: Type Mismatch Validation

### 1.1 Generic Type Mismatch (P0_018)

#### SPEC

**File:** `tests/spec/19_negative/P0_018_generic_type_mismatch.sg`

```sigil
// Negative Test: Generic type parameter mismatch
// Should fail with type mismatch error
// Priority: P0 - Production Critical

fn main() {
    let v: Vec<i32> = Vec::new();
    v.push("not an integer");  // ERROR: expected i32, found String
}
```

**Expected Error (in .error_expected):**
```
type mismatch
```

**Behavior Specification:**
1. When `push(T)` is called on `Vec<U>` where T ≠ U, emit error
2. Error must mention both expected type (U) and found type (T)
3. Error location must point to the offending argument

#### TEST (RED)

```bash
# Verify test currently fails incorrectly (no error)
cd jormungandr/tests
./run_tests_rust.sh --spec 19_negative 2>&1 | grep P0_018
# Should show: ❌ FAIL: Should have errored but succeeded
```

#### IMPLEMENT (GREEN)

**Location:** `parser/src/interpreter.rs`

**Changes Required:**

1. Track generic type parameters during Vec instantiation:
```rust
// In eval_call for Vec::new()
// Store the concrete type parameter
```

2. Validate type on push():
```rust
// In eval_method_call for "push"
// Compare argument type against stored type parameter
// If mismatch, return Err(RuntimeError::new(...))
```

**Implementation Steps:**

```rust
// Step 1: Add type parameter tracking to Vec values
// In Value enum or Value::Array metadata

// Step 2: In eval_method_call for Vec::push
(Value::Array(arr), "push") => {
    if let Some(expected_type) = /* get stored type parameter */ {
        let arg_type = self.get_value_type(&arg_values[0]);
        if expected_type != arg_type {
            return Err(RuntimeError::new(format!(
                "type mismatch: expected {}, found {}",
                expected_type, arg_type
            )));
        }
    }
    // ... existing push logic
}
```

#### VERIFY (GREEN)

```bash
./run_tests_rust.sh --spec 19_negative 2>&1 | grep P0_018
# Should show: ✅ PASS (expected error)
```

#### INTEGRATE

```bash
# Full regression check
./run_tests_rust.sh

# Jormungandr smoke test
cd ../.. && ./parser/target/release/sigil run-dir jormungandr/src -- jormungandr/src/main.sg
```

---

### 1.2 Option Type Mismatch (P0_019)

#### SPEC

**File:** `tests/spec/19_negative/P0_019_option_type_mismatch.sg`

```sigil
// Negative Test: Option type parameter mismatch
// Should fail with type mismatch error
// Priority: P0 - Production Critical

fn main() {
    let opt: Option<i32> = Some("not an integer");  // ERROR
}
```

#### IMPLEMENT

**Location:** `parser/src/interpreter.rs` - Option::Some constructor

```rust
// In eval_call for Option::Some or enum variant construction
// Validate inner value type against declared Option<T>
```

---

### 1.3 Result Type Mismatch (P0_020)

#### SPEC

**File:** `tests/spec/19_negative/P0_020_result_type_mismatch.sg`

```sigil
// Negative Test: Result type parameter mismatch
// Should fail with type mismatch error
// Priority: P0 - Production Critical

fn main() {
    let r: Result<i32, String> = Ok("not an integer");  // ERROR
}
```

#### IMPLEMENT

**Location:** Same as Option - enum variant type checking

---

### 1.4 Match Arm Type Consistency (P0_027)

#### SPEC

**File:** `tests/spec/19_negative/P0_027_invalid_match_arm_type.sg`

```sigil
// Negative Test: Match arms return different types
// Should fail with type mismatch error
// Priority: P0 - Production Critical

fn main() {
    let x = match true {
        true => 42,
        false => "string",  // ERROR: expected i32
    };
}
```

#### IMPLEMENT

**Location:** `parser/src/interpreter.rs` - `eval_match`

```rust
fn eval_match(&mut self, expr: &Expr, arms: &[MatchArm]) -> Result<Value, RuntimeError> {
    let mut first_type: Option<String> = None;

    for arm in arms {
        let result = self.evaluate(&arm.body)?;
        let result_type = self.get_value_type(&result);

        match &first_type {
            None => first_type = Some(result_type),
            Some(expected) if *expected != result_type => {
                return Err(RuntimeError::new(format!(
                    "type mismatch in match arms: expected {}, found {}",
                    expected, result_type
                )));
            }
            _ => {}
        }
    }
    // ... rest of match evaluation
}
```

---

### 1.5 Negative Array Size (P0_028)

#### SPEC

**File:** `tests/spec/19_negative/P0_028_negative_array_size.sg`

```sigil
// Negative Test: Negative array index
// Should fail with index error
// Priority: P0 - Production Critical

fn main() {
    let arr = [1, 2, 3];
    let x = arr[-1];  // ERROR: negative index
}
```

#### IMPLEMENT

**Location:** `parser/src/interpreter.rs` - array indexing

```rust
// In eval_index or array access
if index < 0 {
    return Err(RuntimeError::new(format!(
        "array index cannot be negative: {}", index
    )));
}
```

---

## Sprint 2: Trait Bounds

### 2.1 Trait Bound Validation (P0_052)

#### SPEC

**Behavior:** When calling a generic function with trait bounds, verify the type argument implements the required trait.

```sigil
trait Display {
    fn display(&self) -> String;
}

fn print_it<T: Display>(item: T) {
    println(item.display());
}

fn main() {
    print_it(42);  // ERROR if i32 doesn't impl Display
}
```

#### IMPLEMENT

1. Store trait implementations in interpreter
2. On generic call, check type implements required traits
3. Error with clear message if not

---

### 2.2 Where Clause Support (P0_061)

#### SPEC

```sigil
fn process<T, U>(a: T, b: U)
where
    T: Clone,
    U: Display,
{
    // ...
}
```

#### IMPLEMENT

1. Parse where clauses in function definitions
2. Store constraints with function
3. Validate at call site

---

## Sprint 3: Method Chaining Edge Cases

### 3.1 Method Chain Type Preservation (P0_058)

#### SPEC

```sigil
fn main() {
    let result = vec![1, 2, 3]
        .iter()
        .map(|x| x * 2)
        .filter(|x| x > 2)
        .collect::<Vec<_>>();

    assert_eq!(result, vec![4, 6]);
}
```

#### IMPLEMENT

Ensure each method in chain correctly propagates type information.

---

## Verification Matrix

| Test ID | Spec | Test | Impl | Refactor | Doc | Integrate |
|---------|------|------|------|----------|-----|-----------|
| P0_018 | [x] | [x] | [x] | [x] | [x] | [x] |
| P0_019 | [x] | [x] | [x] | [x] | [x] | [x] |
| P0_020 | [x] | [x] | [x] | [x] | [x] | [x] |
| P0_027 | [x] | [x] | [x] | [x] | [x] | [x] |
| P0_028 | [x] | [x] | [x] | [x] | [x] | [x] |
| P0_052 | [x] | [x] | [x] | [x] | [x] | [x] |
| P0_061 | [x] | [x] | [x] | [x] | [x] | [x] |
| P0_058 | [x] | [x] | [x] | [x] | [x] | [x] |

**All Wave 1 tests complete!** Fixed 2026-01-21 by restructuring `typeck.rs` to check user-defined methods before hardcoded patterns.

---

## Daily Workflow

### Morning
```bash
# Check current state
cd jormungandr/tests
./run_tests_rust.sh --spec 19_negative
./run_tests_rust.sh --spec 03_types
```

### Development Cycle
```bash
# 1. Write/verify failing test (RED)
../../parser/target/release/sigil run tests/spec/19_negative/P0_018.sg
# Should error with unexpected output

# 2. Implement fix
vim ../../parser/src/interpreter.rs

# 3. Rebuild
cd ../../parser && CARGO_INCREMENTAL=0 cargo build --release

# 4. Verify test passes (GREEN)
cd ../jormungandr/tests
../../parser/target/release/sigil run tests/spec/19_negative/P0_018.sg
# Should error with expected error
```

### Evening
```bash
# Full regression check
./run_tests_rust.sh

# Commit if green
git add -A && git commit -m "feat(interpreter): implement P0_018 generic type mismatch validation"
```

---

## Success Criteria

### Wave 1 Complete When:

1. **All P0 type tests pass (5/5)**
   - P0_018, P0_019, P0_020, P0_027, P0_028

2. **All P0 trait tests pass (2/2)**
   - P0_052, P0_061

3. **Method chaining fixed (1/1)**
   - P0_058

4. **No regressions**
   - All 451 previously passing tests still pass
   - Jormungandr still functional

5. **Overall pass rate ≥ 90%**
   - 478/531 or better

---

## Risk Mitigation

| Risk | Mitigation |
|------|------------|
| Type checking breaks valid code | Run full suite after each change |
| Performance regression | Profile hot paths before/after |
| Jormungandr breaks | Smoke test after each sprint |
| Scope creep | Strict adherence to P0 only |

---

## Next Wave Preview

**Wave 2 (P1):** Memory Features ✅ COMPLETE (2026-01-21)
- Reborrow semantics (`&mut T` → `&T` coercion)
- Box<T> deref (`&Box<T>` → `&T` coercion)
- Slice borrowing (`&Vec<T>` → `&[T]` coercion)
- Lifetime elision (stdlib function shadowing)

**Wave 3 (P1):** Stdlib Completion
- Math functions (exp, log)
- Vec::clear()
- Static variables
