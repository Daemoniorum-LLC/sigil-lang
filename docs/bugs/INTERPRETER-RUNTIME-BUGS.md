# Interpreter Runtime Bugs Report

**Date:** 2026-01-21 (Updated)
**Reporter:** Claude Code / TDD Investigation
**Component:** `parser/src/interpreter.rs`
**Status:** MOSTLY RESOLVED - Jormungandr now functional
**Related:** Jormungandr bootstrap effort, [INTERPRETER-SPEC-ROADMAP.md](../specs/INTERPRETER-SPEC-ROADMAP.md)

---

## Resolution Summary (2026-01-21)

Multiple critical bugs have been fixed in this session:

| Bug | Status | Fix |
|-----|--------|-----|
| Option pattern matching regression | FIXED | Reverted unwrap_all, added tuple pattern handling |
| Option::None == null equality | FIXED | Added equality handling in eval_binary |
| .cloned() on enum variants | FIXED | Added generic clone support |
| Invalid field access returns null | FIXED | Now returns error |
| Invalid method call returns null | FIXED | Now returns error for primitives |
| Missing struct field validation | FIXED | Now validates required fields |
| Immutable mutation allowed | FIXED | Environment tracks mutability |
| Invalid enum variant created | FIXED | Now validates variant exists |
| Generic type mismatch (Vec::push) | FIXED | Added type validation in push() |
| Option type mismatch | FIXED | Added validate_type_annotation() |
| Result type mismatch | FIXED | Added validate_type_annotation() |
| Match arm type consistency | FIXED | Added check_match_arm_types() |
| Negative array index | FIXED | Now returns error for negative indices |
| Trait bound on generics | FIXED | Added generic setup in collect_fn_sig |
| Where clause support | FIXED | Same fix as trait bounds |
| Method chaining type inference | FIXED | Restructured typeck.rs to check user methods first |

**Current Test Status:** 463/531 (87%) passing

**Negative Tests:** 30/30 (100%) passing

**Type System Tests:** 80/80 (100%) passing - Wave 1 complete

**Memory Tests:** 35/35 (100%) passing - Wave 2 complete

---

---

## Executive Summary

During runtime testing of the Jormungandr self-hosted Sigil compiler, four interpreter bugs were discovered that prevent successful execution. The most critical bug involves mutable struct field access for imported modules, which breaks argument parsing and makes the compiler non-functional.

All 14 Jormungandr source files now type-check successfully with the canonical Rust compiler, but runtime execution fails due to these interpreter issues.

---

## Bug 1: Vec::push on Imported Struct Field Does Not Persist

### Severity: CRITICAL

### Description
When a struct is defined in module A and imported into module B, calling `push()` on a Vec field of that struct appears to succeed but does not actually modify the Vec. The mutation is lost.

### Minimal Reproduction

**File: `/home/crook/dev2/workspace/sigil/sigil-lang/jormungandr/src/driver.sg`**
```sigil
pub struct Config {
    pub input_files: ![String],
    // ... other fields
}

impl Config {
    pub fn default() -> !Config {
        Config { input_files: [], ... }
    }
}
```

**Test file:**
```sigil
invoke tome·driver·Config;

fn main() {
    let mut config = Config·default();
    println(config.input_files.len().to_string());  // Prints: 0

    config.input_files.push("test.sg".to_string());
    println(config.input_files.len().to_string());  // Prints: 0  <-- BUG: Should be 1
}
```

### Expected Behavior
After `push()`, `config.input_files.len()` should return `1`.

### Actual Behavior
After `push()`, `config.input_files.len()` still returns `0`.

### Control Test (Works Correctly)
When the struct is defined in the same file, `push()` works correctly:

```sigil
struct MyStruct {
    items: ![String],
}

fn main() {
    let mut s = MyStruct { items: [] };
    s.items.push("hello".to_string());
    println(s.items.len().to_string());  // Prints: 1  <-- Correct
}
```

### Root Cause Hypothesis
The interpreter likely creates a copy of the struct field when accessing it across module boundaries, rather than providing a mutable reference. The `push()` modifies the copy, which is then discarded.

### Impact
- **Jormungandr `Config::from_args()`** fails to collect input files
- **All argument parsing** is broken
- **The entire compiler is non-functional**

### Suggested Fix Location
`parser/src/interpreter.rs` - Look for:
- Field access handling (`Expr::Field`, `Expr::FieldAccess`)
- Module boundary crossing logic
- Mutable reference handling for struct fields

---

## Bug 2: Match on Result::Ok(struct) Fails for Imported Structs

### Severity: CRITICAL

### Description
When matching on `Result::Ok(value)` where `value` is a struct type imported from another module, the match fails with "No matching pattern for StructName { }".

### Minimal Reproduction

```sigil
invoke tome·driver·Config;

fn main() {
    let args = vec!["compile".to_string(), "test.sg".to_string()];
    let result = Config·from_args(args);

    match result {
        Result·Ok(config) => {
            // Never reached
            println("Got config");
        },
        Result·Err(e) => {
            println(e);
        },
    }
}
```

### Error Message
```
Runtime error: No matching pattern for Config {  }
```

### Expected Behavior
The match should successfully bind `config` to the `Config` struct inside `Result::Ok`.

### Actual Behavior
Runtime error claiming no pattern matches, even though `Result·Ok(config)` should match any `Result::Ok` variant.

### Control Test (Works Correctly)
Matching on `Result::Ok` with primitive types or locally-defined structs works:

```sigil
fn make_result() -> !Result<i32, String> {
    Result·Ok(42)
}

fn main() {
    match make_result() {
        Result·Ok(v) => println(v.to_string()),  // Works: prints 42
        Result·Err(e) => println(e),
    }
}
```

### Root Cause Hypothesis
The pattern matching logic may be comparing struct type identities incorrectly across module boundaries, possibly due to:
- Fully-qualified name mismatch (`tome::driver::Config` vs `Config`)
- Struct metadata not being properly resolved for imported types

### Workaround
Use `is_ok()` / `is_err()` with `unwrap()` instead of pattern matching:

```sigil
if result.is_err() {
    // handle error
} else {
    let config = result.unwrap();
    // use config
}
```

### Impact
- Cannot use idiomatic Rust-style error handling
- Workaround exists but is less ergonomic

---

## Bug 3: Result::unwrap_err() Returns Null

### Severity: MEDIUM

### Description
Calling `unwrap_err()` on a `Result::Err` value returns `null` instead of the contained error value.

### Minimal Reproduction

```sigil
fn make_err() -> !Result<i32, String> {
    Result·Err("This is the error message".to_string())
}

fn main() {
    let result = make_err();
    let err = result.unwrap_err();
    println(err);  // Prints: null  <-- BUG: Should print error message
}
```

### Expected Behavior
`unwrap_err()` should return `"This is the error message"`.

### Actual Behavior
`unwrap_err()` returns `null`.

### Control Test (Match Works)
Using match to extract the error value works correctly:

```sigil
match make_err() {
    Result·Ok(_) => {},
    Result·Err(e) => println(e),  // Prints: "This is the error message"
}
```

### Root Cause Hypothesis
The `unwrap_err()` implementation in stdlib may be incorrectly implemented, possibly:
- Returning the wrong field of the Result enum
- Not handling the Err variant extraction correctly

### Workaround
Use pattern matching instead of `unwrap_err()`.

### Impact
- Cannot use `unwrap_err()` method
- Workaround exists (use match)

---

## Bug 4: Result::is_ok() Print Behavior

### Severity: LOW

### Description
When printing the result of `is_ok()` via `to_string()`, it produces unexpected output (empty or missing).

### Minimal Reproduction

```sigil
fn main() {
    let result: Result<i32, String> = Result·Ok(42);
    println(result.is_ok().to_string());  // Prints nothing or unexpected value
}
```

### Expected Behavior
Should print `true` or `false`.

### Actual Behavior
Prints empty or unexpected output.

### Impact
- Minor inconvenience for debugging
- The actual boolean logic works correctly

---

## Testing Strategy

### Unit Tests to Add

```rust
// In parser/src/interpreter.rs tests

#[test]
fn test_imported_struct_field_mutation() {
    // Create a module with a struct containing Vec field
    // Import it in another module
    // Mutate the Vec field
    // Verify mutation persists
}

#[test]
fn test_match_result_ok_imported_struct() {
    // Create Result::Ok containing imported struct
    // Match on it with binding
    // Verify binding succeeds
}

#[test]
fn test_result_unwrap_err() {
    // Create Result::Err with string
    // Call unwrap_err()
    // Verify correct string is returned
}
```

### Integration Tests

Create test files in `jormungandr/tests/` that verify:
1. Cross-module struct field mutation
2. Result pattern matching with imported types
3. All Result methods work correctly

---

## Files to Investigate

1. **`parser/src/interpreter.rs`**
   - `eval_expr()` - Field access handling
   - `eval_method_call()` - push, unwrap_err implementation
   - `eval_match()` - Pattern matching logic
   - Module/import resolution code

2. **`parser/src/stdlib.rs`**
   - `Result` type implementation
   - `unwrap_err()` method definition

3. **`parser/src/typeck.rs`**
   - Cross-module type resolution (may affect runtime)

---

## Priority Order for Fixes

1. **Bug 1 (Vec::push imported struct)** - CRITICAL, blocks all functionality
2. **Bug 2 (Match imported struct)** - CRITICAL, blocks idiomatic code
3. **Bug 3 (unwrap_err)** - MEDIUM, has workaround
4. **Bug 4 (is_ok print)** - LOW, cosmetic

---

## Appendix: Full Reproduction Script

```bash
#!/bin/bash
# Save as: test_interpreter_bugs.sh

cd /home/crook/dev2/workspace/sigil/sigil-lang/parser

echo "=== Bug 1: Vec::push on imported struct ==="
cat > /tmp/bug1.sg << 'EOF'
invoke tome·driver·Config;
fn main() {
    let mut config = Config·default();
    println(config.input_files.len().to_string());
    config.input_files.push("test.sg".to_string());
    println(config.input_files.len().to_string());
}
EOF
./target/release/sigil run /tmp/bug1.sg

echo ""
echo "=== Bug 2: Match Result::Ok imported struct ==="
cat > /tmp/bug2.sg << 'EOF'
invoke tome·driver·*;
fn main() {
    let args = vec!["compile".to_string(), "test.sg".to_string()];
    let result = Config·from_args(args);
    match result {
        Result·Ok(config) => println("Got config"),
        Result·Err(e) => println(e),
    }
}
EOF
./target/release/sigil run /tmp/bug2.sg 2>&1

echo ""
echo "=== Bug 3: unwrap_err returns null ==="
cat > /tmp/bug3.sg << 'EOF'
fn make_err() -> !Result<i32, String> {
    Result·Err("error message".to_string())
}
fn main() {
    let err = make_err().unwrap_err();
    println(err);
}
EOF
./target/release/sigil run /tmp/bug3.sg
```
