# Const Generic Inference Specification

**Status**: Implemented
**Version**: 0.1.0
**Date**: 2026-02-06
**Scope**: `parser/src/interpreter.rs`, `parser/src/typeck.rs`
**Extends**: 03B-TYPECK-GENERIC-INFERENCE.md (which lists const generics as non-goal)

## 1. Overview

This spec defines bidirectional inference for const generic parameters. When a constructor
is called without explicit const generics, the runtime should infer them from the expected
type context (field annotations, variable declarations, return type annotations).

### 1.1 Problem Statement

```sigil
sigil Container<◆ N: usize> {
    value: i64,
}

⊢<◆ N: usize> Container<N> {
    ☉ rite new() → This {
        Container { value: N as i64 }
    }
}

sigil Wrapper {
    c: Container<42>,  // Expected type: Container<42>
}

⊢ Wrapper {
    ☉ rite new() → This {
        This {
            // Currently fails: "undefined variable: N"
            // Expected: infer N=42 from field type annotation
            c: Container·new(),
        }
    }
}
```

### 1.2 Current Behavior

The interpreter has three mechanisms for const generic resolution (interpreter.rs):

1. **Turbofish syntax** (lines 6076-6108): `Container·<42>·new()` extracts value from path
2. **type_context.struct_generics** (lines 17565-17577): Injection from struct literal path
3. **Environment lookup** (lines 17579-17593): Checks if param name is a variable in scope

None of these handle field-context inference.

### 1.3 Desired Behavior

When evaluating a struct field initialization:
1. Look up the field's declared type from the struct definition
2. If the declared type has const generic arguments, extract them
3. Propagate these values to the initializer expression's evaluation context

---

## 2. Inference Rules

### 2.1 Field Context Propagation

```
                    Γ ⊢ S.f : T<N₁, N₂, ...>    (field f has type T with const generics)
                    Γ, N₁=n₁, N₂=n₂, ... ⊢ e : T<N₁, N₂, ...>
─────────────────────────────────────────────────────────────────────────────────────────
                    Γ ⊢ S { f: e } : S
```

When initializing field `f` with expression `e`, if `f`'s declared type is `T<n₁, n₂, ...>`:
- Bind `N₁ = n₁`, `N₂ = n₂`, etc. in the evaluation environment
- Evaluate `e` in this extended environment

### 2.2 Variable Declaration Context

```
                    ≔ x: T<N₁, ...> = e
                    Γ, N₁=n₁, ... ⊢ e : T<N₁, ...>
─────────────────────────────────────────────────────────────────────────────────────────
                    Γ, x: T<n₁, ...> ⊢ ...
```

When a variable declaration has an explicit type annotation with const generics,
propagate those values to the initializer.

### 2.3 Return Type Context

```
                    ☉ rite f() → T<N₁, ...> { e }
                    Γ, N₁=n₁, ... ⊢ e : T<N₁, ...>
─────────────────────────────────────────────────────────────────────────────────────────
                    Γ ⊢ f : () → T<n₁, ...>
```

When a function's return type has const generics, propagate them to the body.

---

## 3. Implementation Design

### 3.1 Interpreter Changes

Location: `parser/src/interpreter.rs`

#### 3.1.1 Struct Field Initialization

In `eval_struct_literal` (around line 17400), when processing each field:

```rust
// Before evaluating field value:
if let Some(field_type) = struct_def.get_field_type(&field_name) {
    if let Some(const_generics) = extract_const_generics(&field_type) {
        // Bind const generics in evaluation environment
        for (param_name, value) in const_generics {
            self.environment.borrow_mut().define(param_name, Value::Int(value));
        }
    }
}

// Evaluate field value (now has const generics in scope)
let field_value = self.eval_expr(&field_expr)?;

// Clean up bindings after evaluation
```

#### 3.1.2 Type Context Stack

Add a new mechanism to track expected types:

```rust
/// Stack of expected types for bidirectional inference
expected_type_context: Vec<Option<TypeExpr>>,

fn push_expected_type(&mut self, ty: Option<TypeExpr>) {
    self.expected_type_context.push(ty);
}

fn pop_expected_type(&mut self) -> Option<TypeExpr> {
    self.expected_type_context.pop().flatten()
}

fn current_expected_type(&self) -> Option<&TypeExpr> {
    self.expected_type_context.last().and_then(|t| t.as_ref())
}
```

#### 3.1.3 Constructor Call Enhancement

In `eval_call` (around line 6000), when calling a constructor without turbofish:

```rust
// If no turbofish const bindings, check expected type context
if turbofish_const_bindings.is_empty() {
    if let Some(expected) = self.current_expected_type() {
        if let Some(const_generics) = extract_const_generics_from_type(expected) {
            // Use expected type's const generics
            turbofish_const_bindings = const_generics;
        }
    }
}
```

### 3.2 Type Checker Changes

Location: `parser/src/typeck.rs`

Update `convert_type` at line 3814 to properly handle const expressions:

```rust
TypeExpr::ConstExpr(expr) => {
    // Try to evaluate const expression
    if let Expr::Literal(Literal::Int { value, .. }) = expr.as_ref() {
        Type::ConstValue(value.parse().unwrap_or(0))
    } else if let Expr::Path(path) = expr.as_ref() {
        // Const generic parameter reference
        let name = path.segments.first()
            .map(|s| s.ident.name.clone())
            .unwrap_or_default();
        Type::ConstParam(name)
    } else {
        Type::Var(TypeVar(0))  // Fallback
    }
}
```

Add new Type variants:
```rust
enum Type {
    // ... existing variants ...
    ConstValue(i64),      // Concrete const generic value
    ConstParam(String),   // Const generic parameter reference
}
```

---

## 4. Affected Patterns

### 4.1 Field Initialization (Primary)

```sigil
sigil Wrapper {
    c: Container<42>,
}

⊢ Wrapper {
    ☉ rite new() → This {
        This { c: Container·new() }  // Should infer N=42
    }
}
```

### 4.2 Variable Declaration

```sigil
☉ rite example() {
    ≔ c: Container<10> = Container·new();  // Should infer N=10
}
```

### 4.3 Return Type

```sigil
☉ rite make_container() → Container<5> {
    Container·new()  // Should infer N=5
}
```

### 4.4 Nested Const Generics

```sigil
sigil Outer<◆ M: usize> {
    inner: Container<M>,
}

⊢<◆ M: usize> Outer<M> {
    ☉ rite new() → This {
        This { inner: Container·new() }  // Should infer N=M
    }
}
```

---

## 5. Agent-TDD Test Suite

Location: `jormungandr/tests/interpreter_gaps/`

### 5.1 test_const_generic_field_inference.sg

```sigil
// Test: Const generic inferred from field type annotation

sigil Container<◆ N: usize> {
    value: i64,
}

⊢<◆ N: usize> Container<N> {
    ☉ rite new() → This {
        Container { value: N as i64 }
    }
}

sigil Wrapper {
    c: Container<42>,
}

⊢ Wrapper {
    ☉ rite new() → This {
        This { c: Container·new() }  // Infer N=42
    }
}

#[test]
rite test_field_inference() {
    ≔ w = Wrapper·new();
    assert_eq(w.c.value, 42);
}

☉ rite main() → !i32 {
    test_field_inference();
    println("PASS: const generic field inference");
    ⤺ 0;
}
```

### 5.2 test_const_generic_variable_inference.sg

```sigil
// Test: Const generic inferred from variable type annotation

sigil Container<◆ N: usize> {
    value: i64,
}

⊢<◆ N: usize> Container<N> {
    ☉ rite new() → This {
        Container { value: N as i64 }
    }
}

#[test]
rite test_variable_inference() {
    ≔ c: Container<100> = Container·new();  // Infer N=100
    assert_eq(c.value, 100);
}

☉ rite main() → !i32 {
    test_variable_inference();
    println("PASS: const generic variable inference");
    ⤺ 0;
}
```

### 5.3 test_const_generic_return_inference.sg

```sigil
// Test: Const generic inferred from return type annotation

sigil Container<◆ N: usize> {
    value: i64,
}

⊢<◆ N: usize> Container<N> {
    ☉ rite new() → This {
        Container { value: N as i64 }
    }
}

☉ rite make_container() → Container<7> {
    Container·new()  // Infer N=7
}

#[test]
rite test_return_inference() {
    ≔ c = make_container();
    assert_eq(c.value, 7);
}

☉ rite main() → !i32 {
    test_return_inference();
    println("PASS: const generic return inference");
    ⤺ 0;
}
```

---

## 6. Implementation Status

**Fully Implemented (2026-02-07):**

| Scenario | Status | Test File |
|----------|--------|-----------|
| Field initialization | ✅ PASS | `test_const_generic_field_inference.sg` |
| Variable declaration | ✅ PASS | `test_const_generic_variable_inference.sg` |
| Return type | ✅ PASS | `test_const_generic_return_inference.sg` |

**Changes Made:**

1. `interpreter.rs:229-238` - Added `return_type: Option<TypeExpr>` field to `Function` struct
   to preserve return type annotations for const generic inference.

2. `interpreter.rs:17487-17534` - In `eval_struct_literal`, extract const generics from field type
   annotations and set `type_context.struct_generics` before evaluating field values.

3. `interpreter.rs:6149-6165` - In `eval_call`, when no turbofish const bindings are present,
   check `type_context.struct_generics` for inferred const generics and apply them to the
   function's closure environment.

4. `interpreter.rs:6321-6357` - In `call_function`, extract const generics from function's
   return type annotation and set `type_context.struct_generics` before evaluating body.

5. `interpreter.rs:1189-1193` - Added `Environment::undefine()` method for cleanup.

---

## 7. Acceptance Criteria

1. All test files in `jormungandr/tests/interpreter_gaps/test_const_generic_*.sg` pass
2. Existing 762+ tests continue to pass
3. `sigil run /tmp/test_const_explicit.sg` continues to work (explicit turbofish)
4. nihil-serve source files that use const generics execute without "undefined variable" errors

---

## 7. Implementation Order

| Step | Description | Estimated Scope |
|------|-------------|-----------------|
| 1 | Add `expected_type_context` stack to interpreter | ~20 lines |
| 2 | Extract const generics from field type in struct literal eval | ~30 lines |
| 3 | Push expected type before evaluating field initializers | ~10 lines |
| 4 | Enhance constructor call to check expected type | ~15 lines |
| 5 | Add Type::ConstValue and Type::ConstParam to typeck | ~20 lines |
| 6 | Write and run Agent-TDD tests | test files |

---

## 8. Open Questions

1. **Nested inference depth**: How deep should const generic inference propagate?
   - Proposal: One level (direct field type → constructor call)

2. **Conflicting annotations**: What if turbofish and expected type disagree?
   - Proposal: Turbofish wins (explicit > inferred)

3. **Partial inference**: What if only some const generics can be inferred?
   - Proposal: Infer what's possible, error on remaining unbound params

---

## 9. References

- `parser/src/interpreter.rs` — Interpreter implementation
- `parser/src/typeck.rs` — Type checker
- `docs/specs/03A-TYPE-INFERENCE.md` — Bidirectional inference spec
- `docs/specs/03B-TYPECK-GENERIC-INFERENCE.md` — Generic inference (lists const generics as non-goal)
- `docs/specs/INTERPRETER-GAPS-SPEC.md` — Gap 4 documentation
