# Option Type Representation Specification

> **Status:** Draft (Investigation)
> **Last Updated:** 2026-02-08
> **Blocking Issue:** mnist_training.sigil fails with type mismatch in bias tau transform

## 1. Overview

This spec defines how `Option<T>` values should be represented, constructed, stored, and transformed throughout the Sigil interpreter pipeline.

**Motivation:** The current interpreter has inconsistent Option representation, causing `this.bias|τ{b => ...}` patterns to fail when `bias: Option<Tensor>`.

---

## 2. Option Type Definition

From 03-TYPES.md § 4.4:

```sigil
enum Option<T?> {
    Some(T),
    None,
}
```

Option is a **sum type** with two variants:
- `Some(T)` - contains a value of type T
- `None` - represents absence

---

## 3. Value Representation

### 3.1 Canonical Representation

Option values MUST be represented as `Value::Variant`:

```rust
// Some(value)
Value::Variant {
    enum_name: "Option".to_string(),
    variant_name: "Some".to_string(),
    fields: Some(vec![inner_value]),
}

// None
Value::Variant {
    enum_name: "Option".to_string(),
    variant_name: "None".to_string(),
    fields: None,
}
```

### 3.2 Alternative Representations (DEPRECATED)

The interpreter currently also uses `Value::Struct` for Some/None:

```rust
// DEPRECATED: Some as struct
Value::Struct {
    name: "Some".to_string(),
    fields: HashMap::from([("0", inner_value)]),
}

// DEPRECATED: None as struct
Value::Struct {
    name: "None".to_string(),
    fields: HashMap::new(),
}
```

**Problem:** This dual representation causes pattern matching failures because code checking for `Value::Variant` misses `Value::Struct` representations and vice versa.

### 3.3 Resolution

**Decision Required:** Should we:
1. Normalize all Option values to `Value::Variant` at construction time?
2. Support both representations everywhere? (current workaround, fragile)
3. Normalize during field access/storage?

**Recommendation:** Option (1) - normalize at construction. This is the cleanest solution and follows the principle that enums should use `Value::Variant`.

---

## 4. Construction

### 4.1 Literal Construction

```sigil
≔ x = Some(42);      // Creates Value::Variant { enum_name: "Option", variant_name: "Some", fields: Some([42]) }
≔ y = None;          // Creates Value::Variant { enum_name: "Option", variant_name: "None", fields: None }
```

### 4.2 Where Construction Occurs

Option values are constructed in:

1. **Expression evaluation** - `Some(expr)` literal
2. **Pattern matching** - destructuring creates new bindings
3. **Function returns** - functions returning `Option<T>`
4. **Struct field initialization** - `MyStruct { field: Some(value) }`
5. **Method calls** - `.first()`, `.find()`, etc. returning Option

**Investigation needed:** Which of these paths creates `Value::Struct` vs `Value::Variant`?

---

## 5. Storage and Retrieval

### 5.1 Struct Field Storage

When a struct has an Option field:

```sigil
☉ sigil Linear {
    bias: Option<Tensor>,
}

⊢ Linear {
    ☉ rite new(bias: Tensor) → This {
        This { bias: Some(bias) }  // Should store Value::Variant
    }
}
```

**Expected behavior:**
1. `Some(bias)` creates `Value::Variant { enum_name: "Option", variant_name: "Some", ... }`
2. Struct stores this value in its `fields` HashMap
3. Field access `this.bias` returns the exact same `Value::Variant`

**Observed behavior (bug):**
- `this.bias` returns `Value::Struct { name: "Tensor", ... }` (the inner value!)
- The `Some` wrapper is being stripped during storage or retrieval

### 5.2 Field Access

Field access (`expr.field`) should return the stored value unchanged:

```rust
// Correct
this.bias -> Value::Variant { enum_name: "Option", variant_name: "Some", fields: Some([tensor]) }

// Incorrect (current behavior)
this.bias -> Value::Struct { name: "Tensor", ... }
```

**Investigation needed:** Where is the unwrapping happening?
- During struct construction?
- During field storage?
- During field access?

---

## 6. Tau Transform Semantics

### 6.1 Option Tau Transform

The tau transform `|τ{pattern => body}` on Option should behave like Rust's `.map()`:

```sigil
// Input: Option<T>
// Output: Option<U>

Some(x)|τ{v => f(v)}  // → Some(f(x))
None|τ{v => f(v)}     // → None
```

### 6.2 Implementation

```rust
// When value is Option::Some
Value::Variant { enum_name: "Option", variant_name: "Some", fields: Some(inner) } => {
    // 1. Bind inner[0] to pattern
    // 2. Evaluate body
    // 3. Wrap result in Some
    Value::Variant {
        enum_name: "Option".to_string(),
        variant_name: "Some".to_string(),
        fields: Some(vec![result]),
    }
}

// When value is Option::None
Value::Variant { enum_name: "Option", variant_name: "None", fields: None } => {
    // Return None unchanged
    value.clone()
}
```

### 6.3 Pattern Matching Order

**Critical:** Option tau handling MUST be checked BEFORE Tensor tau handling.

Current problem:
```rust
// BAD: Tensor matches first because Option<Tensor> contains a Tensor
Value::Struct { name: "Tensor", .. } => { /* iterates over tensor elements */ }

// This never matches because the above catches it first
Value::Variant { enum_name: "Option", .. } => { /* Option semantics */ }
```

**Solution:** Either:
1. Check for Option variants before any Struct matching
2. Normalize all Options to Variant so they don't match Struct patterns
3. Use more specific pattern matching (check enum_name first)

---

## 7. Current Interpreter State

### 7.1 Root Cause (FOUND)

**The built-in `Linear::new` doesn't check for user-defined implementations first.**

Location: `interpreter.rs:6435-6562`

```rust
// Line 6435: Built-in Linear::new matches BEFORE user-defined nihil Linear·new
["Linear", "new"] => {
    // ...
    // Line 6549-6555: bias stored as RAW Tensor, not Some(Tensor)
    linear_fields.insert(
        "bias".to_string(),
        Value::Struct {
            name: "Tensor".to_string(),  // NOT wrapped in Option::Some!
            fields: Rc::new(RefCell::new(bias_fields)),
        },
    );
}
```

Compare with `Sequential::new` which correctly checks for user-defined first:

```rust
// Line 6576-6584: Sequential::new checks for user-defined first
["Sequential", "new"] => {
    let user_ctor = self.globals.borrow().get("Sequential·new").map(|v| v.clone());
    if let Some(Value::Function(func)) = user_ctor {
        // Use user-defined constructor
        return self.call_function(&func, evaluated_args);
    }
    // Fall back to built-in only if no user-defined
}
```

### 7.2 Known Issues

| Issue | Location | Impact |
|-------|----------|--------|
| Built-in Linear::new doesn't check user-defined first | `interpreter.rs:6435` | nihil's Linear·new ignored |
| Built-in Linear::new stores raw Tensor, not Some(Tensor) | `interpreter.rs:6549` | Option wrapper lost |
| Tensor tau matches before Option tau | `eval_tau_transform` | Option semantics broken |

### 7.3 Affected Code Paths

Files to fix:
- `parser/src/interpreter.rs`:
  - **Line 6435**: Add user-defined check like Sequential::new does
  - **Line 6549**: If keeping built-in, wrap bias in `Value::Variant { Option::Some }`
  - `eval_tau_transform`: Ensure Option variant matching before Tensor struct matching

---

## 8. Test Cases

### 8.1 Unit Tests Needed

```sigil
// Test 1: Option construction preserves wrapper
≔ x = Some(42);
assert(type_of(x) == "Option<i32>");

// Test 2: Struct field stores Option correctly
☉ sigil Test { value: Option<i32> }
≔ t = Test { value: Some(42) };
assert(type_of(t.value) == "Option<i32>");  // NOT "i32"

// Test 3: Option tau transform
≔ x = Some(10);
≔ y = x|τ{v => v * 2};
assert(y == Some(20));

// Test 4: None tau transform
≔ x: Option<i32> = None;
≔ y = x|τ{v => v * 2};
assert(y == None);

// Test 5: Option<Tensor> tau transform (the failing case)
☉ sigil Layer { bias: Option<Tensor> }
≔ layer = Layer { bias: Some(Tensor·zeros([10])) };
≔ result = layer.bias|τ{b => b.broadcast()};
// Should call broadcast on tensor, not fail with type mismatch
```

### 8.2 Integration Test

```sigil
// Minimal reproduction of mnist_training.sigil failure
☉ sigil Linear {
    weight: Tensor,
    bias: Option<Tensor>,
}

⊢ Linear {
    ☉ rite forward(this, input: Tensor) → Tensor {
        ≔ output = input @ this.weight.T;
        this.bias|τ{b => output + b.broadcast()}|unwrap_or(output)
    }
}
```

---

## 9. Resolution Plan

### Phase 1: Investigation ✅ COMPLETE
1. ~~Add debug output to trace Option value representation at each stage~~
2. ~~Identify exact location where Some wrapper is lost~~ → **Line 6435: built-in Linear::new**
3. ~~Document findings in this spec~~ → **See § 7.1**

### Phase 2: Fix
**Primary fix (recommended):** Add user-defined check to built-in Linear::new

```rust
// At line 6435, BEFORE the built-in implementation:
["Linear", "new"] => {
    // Check for user-defined Linear·new first (like Sequential does)
    let user_ctor = self.globals.borrow().get("Linear·new").map(|v| v.clone());
    if let Some(Value::Function(func)) = user_ctor {
        let mut evaluated_args = Vec::new();
        for arg in args {
            evaluated_args.push(self.evaluate(arg)?);
        }
        return self.call_function(&func, evaluated_args);
    }
    // Only use built-in if no user-defined constructor exists
    // ... existing built-in code ...
}
```

**Alternative fix:** If keeping built-in, wrap bias in Option::Some:

```rust
// Replace line 6549-6555 with:
linear_fields.insert(
    "bias".to_string(),
    Value::Variant {
        enum_name: "Option".to_string(),
        variant_name: "Some".to_string(),
        fields: Some(Rc::new(vec![Value::Struct {
            name: "Tensor".to_string(),
            fields: Rc::new(RefCell::new(bias_fields)),
        }])),
    },
);
```

### Phase 3: Verification
1. Add unit tests from § 8.1
2. Verify mnist_training.sigil runs without errors
3. Run full test suite to check for regressions

---

## 10. Related Specs

- 03-TYPES.md § 4.4 - Enum type definition
- 02A-PATTERN-MATCHING.md - Pattern matching semantics
- (TBD) - Pipe operator semantics
