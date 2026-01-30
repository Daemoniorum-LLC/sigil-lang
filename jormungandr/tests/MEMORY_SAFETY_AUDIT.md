# Sigil Memory Safety Audit

**Date:** January 16, 2026
**Auditor:** Claude Code
**Scope:** Rust-based Sigil interpreter (`parser/src/interpreter.rs`)

## Executive Summary

The Sigil interpreter implements memory safety through runtime mechanisms rather than compile-time static analysis. All P0 memory tests pass (25/25), demonstrating correct behavior for the implemented features.

## Findings

### 1. Ownership Transfer Semantics

**Status:** ⚠️ Partial Implementation

**Implementation:**
- Values are cloned when passed to functions (interpreter semantics)
- No compile-time move tracking (`Value::Moved` state does not exist)
- The `BorrowError` (E0382) error code is defined but never raised

**Implications:**
- Use-after-move is not detected at compile time
- Variables can be used multiple times even after being "moved"
- Semantic correctness relies on programmer discipline

**Location:** `parser/src/typeck.rs:221-222` (error code defined but unused)

---

### 2. Mutable Borrow Implementation

**Status:** ✅ Working via Sync-Back Mechanism

**Implementation:** (`interpreter.rs:3948-4014`)
```rust
// Track &mut path arguments for sync-back after function call
let mut mut_ref_sync: Vec<(String, Rc<RefCell<Value>>)> = Vec::new();
// ... after function call ...
// Sync mutable references back to original variables
for (var_name, ref_val) in mut_ref_sync {
    let current_value = ref_val.borrow().clone();
    let _ = self.environment.borrow_mut().set(&var_name, current_value);
}
```

**Behavior:**
- `&mut` references track the original variable name
- After function returns, modified values are written back to original variables
- Works correctly for simple path expressions (`&mut x`)
- Complex expressions (struct fields, array elements) may not sync back

**Test:** `P0_022_mutable_ref_syncback.sg` verifies this works correctly

---

### 3. Drop Trait Implementation

**Status:** ✅ Working

**Implementation:** (`interpreter.rs:4350-4384`)
```rust
// RAII: Call Drop::drop() on values going out of scope
let values_to_drop: Vec<(String, Value)> = self
    .environment.borrow().values.iter()
    .filter_map(|(name, value)| {
        if let Value::Struct { name: struct_name, .. } = value {
            if self.drop_types.contains(struct_name) {
                return Some((name.clone(), value.clone()));
            }
        }
        None
    }).collect();
for (_var_name, value) in values_to_drop {
    // Call Type·drop function
}
```

**Behavior:**
- Drop::drop() called when values leave scope
- Destructor naming convention: `Type·drop`
- Works for nested scopes
- **Note:** Drop order is HashMap iteration order (non-deterministic), not LIFO

**Tests:**
- `P0_013_drop_trait.sg` - Basic drop
- `P0_021_drop_scope_exit.sg` - Scope exit drop
- `P0_024_multiple_drops.sg` - Multiple drops (count verified)
- `P0_025_nested_scope_drops.sg` - Nested scope drops

---

### 4. Rc<T> Reference Counting

**Status:** ⚠️ Simplified Implementation (Copy Semantics)

**Implementation:** (`interpreter.rs:1904-1918`, `6875-6891`)
```rust
// Rc::new - wraps value in struct
let rc_new = Value::BuiltIn(Rc::new(BuiltInFn {
    func: |_, args| {
        let mut fields = HashMap::new();
        fields.insert("_value".to_string(), args[0].clone());
        Ok(Value::Struct { name: "Rc".to_string(), fields: ... })
    },
}));

// Rc::clone - creates independent copy
"clone" => {
    let mut new_fields = HashMap::new();
    new_fields.insert("_value".to_string(), value.clone());
    return Ok(Value::Struct { name: "Rc".to_string(), ... });
}
```

**Key Finding:**
- **Sigil's Rc is NOT true reference counting**
- `Rc::clone()` creates an independent copy of the inner value
- Modifications to one Rc do NOT affect cloned versions
- This is value/copy semantics, not shared reference semantics

**Implications:**
- Safe: No shared mutable state issues
- Different from Rust's Rc<T> semantics
- May be surprising to Rust developers

**Test:** `P0_020_rc_clone_copy_semantics.sg` documents this behavior

---

### 5. Cell<T> Interior Mutability

**Status:** ✅ Working

**Implementation:** (`interpreter.rs:6893-6912`)
- `Cell::get()` returns copy of inner value
- `Cell::set()` replaces inner value
- Uses RefCell for actual mutability

**Test:** `P0_023_cell_interior_mutation.sg`

---

## Missing Safety Features

| Feature | Rust | Sigil |
|---------|------|-------|
| Use-after-move detection | Compile-time | ❌ Not enforced |
| Borrow checker | Compile-time | ❌ Not implemented |
| Lifetime analysis | Compile-time | ❌ Not implemented |
| Double-borrow prevention | Runtime panic | ❌ Not checked |
| Data race prevention | Compile-time | ❌ Not applicable (single-threaded) |

## Recommendations

### Short-term (P0/P1)
1. Document that Rc<T> uses copy semantics in user-facing documentation
2. Consider adding runtime use-after-move tracking for debug builds
3. Document the sync-back mechanism limitations for complex expressions

### Medium-term (P2)
1. Implement basic borrow checking in typeck.rs
2. Add runtime double-borrow detection with clear error messages
3. Implement true reference counting for Rc<T> if shared semantics are desired

### Long-term (P3)
1. Full lifetime analysis and borrow checking
2. MIRI-style runtime verification for unsafe code
3. Thread safety analysis for Arc<T>

## Test Coverage

New regression tests added:
- `P0_020_rc_clone_copy_semantics.sg` - Documents Rc copy semantics
- `P0_021_drop_scope_exit.sg` - Drop on scope exit
- `P0_022_mutable_ref_syncback.sg` - Mutable reference sync-back
- `P0_023_cell_interior_mutation.sg` - Cell get/set
- `P0_024_multiple_drops.sg` - Multiple drops in scope
- `P0_025_nested_scope_drops.sg` - Nested scope drops

All P0 memory tests: **25/25 passing** ✅
