# WASM Result Combinator Specification

**Version:** 0.1.0
**Status:** Draft
**Date:** 2026-02-14
**Parent Spec:** 03-TYPES.md (Result type definition)

---

## Abstract

This specification defines the implementation of Result type combinators (`map_err`, `map`, `and_then`, `unwrap_or`, etc.) for the Sigil WASM backend. These combinators enable ergonomic error handling in qliphoth-sys and other web-targeting Sigil code.

---

## 1. Conceptual Foundation

### 1.1 The Problem

qliphoth-sys uses Result combinators extensively:

```sigil
// 24 occurrences of map_err in qliphoth-sys
history.push_state(state, title, url)
    ·map_err(|e| e·to_string())?;
```

Currently, the WASM backend fails with "undefined function: map_err" because:
1. Result methods are not recognized as method calls
2. Closures passed to combinators are not compiled
3. No type tracking exists to know when a value is a Result

### 1.2 Design Goals

- **Zero-cost abstraction:** Result combinators should compile to efficient inline code
- **Type erasure aware:** All values are i64 at runtime; type tracking is compile-time only
- **Closure support:** Combinator closures must compile to callable WASM
- **Interop compatible:** Results crossing JS/WASM boundary need defined representation

### 1.3 Non-Goals

- Full generic type inference (use explicit type annotations where needed)
- Result methods not used in qliphoth-sys (defer until needed)

---

## 2. Type Architecture

### 2.1 Result Representation

```
Result<T, E> representation in WASM memory:

┌─────────────────────────────────────┐
│ Discriminant Tag (i64)              │  offset 0
│   0 = Ok                            │
│   1 = Err                           │
├─────────────────────────────────────┤
│ Payload (i64)                       │  offset 8
│   If Ok:  T value (or pointer)      │
│   If Err: E value (or pointer)      │
└─────────────────────────────────────┘

Total size: 16 bytes
```

### 2.2 Stack Representation (Optimization)

For simple cases, Results can be represented on the WASM stack:

```
Stack Result (2 × i64):
  [discriminant: i64] [payload: i64]

Operations can work directly on stack values without heap allocation.
```

### 2.3 Type Tracking

The compiler must track Result types during expression compilation:

```
ExprType:
    Primitive(ValType)          // i32, i64, f32, f64
    Result { ok: ExprType, err: ExprType }
    Option { inner: ExprType }
    Unknown                     // Fallback for untracked types
```

---

## 3. Behavioral Contracts

### 3.1 map_err

```
Result<T, E>.map_err(f: E → F) → Result<T, F>

Pseudocode:
    discriminant ← stack.pop()
    payload ← stack.pop()

    if discriminant = 0:        // Ok
        stack.push(payload)
        stack.push(0)           // Ok tag
    else:                       // Err
        new_err ← call_closure(f, payload)
        stack.push(new_err)
        stack.push(1)           // Err tag
```

**Invariants:**
- P1: `Ok(v).map_err(f) = Ok(v)` (Ok values pass through unchanged)
- P2: `Err(e).map_err(f) = Err(f(e))` (Err values are transformed)
- P3: Closure `f` is called exactly once for Err, never for Ok

### 3.2 map

```
Result<T, E>.map(f: T → U) → Result<U, E>

Pseudocode:
    discriminant ← stack.pop()
    payload ← stack.pop()

    if discriminant = 0:        // Ok
        new_ok ← call_closure(f, payload)
        stack.push(new_ok)
        stack.push(0)           // Ok tag
    else:                       // Err
        stack.push(payload)
        stack.push(1)           // Err tag
```

### 3.3 and_then

```
Result<T, E>.and_then(f: T → Result<U, E>) → Result<U, E>

Pseudocode:
    discriminant ← stack.pop()
    payload ← stack.pop()

    if discriminant = 0:        // Ok
        // f returns Result, which is already 2 × i64
        call_closure(f, payload)
        // Result is now on stack
    else:                       // Err
        stack.push(payload)
        stack.push(1)           // Err tag
```

### 3.4 unwrap_or

```
Result<T, E>.unwrap_or(default: T) → T

Pseudocode:
    discriminant ← stack.pop()
    payload ← stack.pop()
    default ← stack.pop()       // Already evaluated

    if discriminant = 0:        // Ok
        stack.push(payload)
    else:                       // Err
        stack.push(default)
```

### 3.5 is_ok / is_err

```
Result<T, E>.is_ok() → bool
    discriminant ← stack.pop()
    payload ← stack.pop()       // Discard
    stack.push(discriminant = 0)

Result<T, E>.is_err() → bool
    discriminant ← stack.pop()
    payload ← stack.pop()       // Discard
    stack.push(discriminant = 1)
```

### 3.6 The ? Operator

```
expr? desugars to:

    result ← evaluate(expr)
    discriminant ← result.discriminant
    payload ← result.payload

    if discriminant = 1:        // Err
        return Err(payload)
    else:
        stack.push(payload)     // Continue with Ok value
```

---

## 4. Closure Compilation

### 4.1 Inline Closure Strategy

For simple closures passed directly to combinators:

```sigil
// Source
result·map_err(|e| e·to_string())

// Compiled: closure is inlined
if discriminant = 1:
    // Inline closure body
    err_string ← call to_string(payload)
    stack.push(err_string)
    stack.push(1)
```

### 4.2 Named Closure Strategy

For complex closures or reused closures:

```sigil
// Source
≔ transform = |e| format!("Error: {}", e);
result·map_err(transform)

// Compiled: closure becomes a WASM function
func $closure_0 (param $e i64) (result i64)
    ;; Format the error
    ...
end

// Call site
if discriminant = 1:
    call $closure_0(payload)
    stack.push(result)
    stack.push(1)
```

### 4.3 Capture Handling

Closures that capture environment variables:

```sigil
≔ prefix = "Error: ";
result·map_err(|e| prefix + e·to_string())

// Compiled with capture
// Environment: [prefix_ptr: i64]
func $closure_1 (param $env i64) (param $e i64) (result i64)
    local.get $env
    i64.load         ;; Load prefix_ptr
    local.get $e
    call $to_string
    call $string_concat
end
```

---

## 5. Implementation Strategy

### 5.1 Phase 1: Method Resolution (Required)

Recognize Result combinator methods during expression compilation:

```
compile_method_call(receiver, method, args):
    receiver_type ← infer_type(receiver)

    if receiver_type is Result:
        match method:
            "map_err" → compile_map_err(receiver, args[0])
            "map" → compile_map(receiver, args[0])
            "and_then" → compile_and_then(receiver, args[0])
            "unwrap_or" → compile_unwrap_or(receiver, args[0])
            "is_ok" → compile_is_ok(receiver)
            "is_err" → compile_is_err(receiver)
            _ → error("unknown Result method")
    else:
        // Existing method call logic
```

### 5.2 Phase 2: Type Inference (Required)

Track Result types through expression compilation:

```
infer_type(expr):
    match expr:
        Call(path, args) if path ends with "Ok" → Result { ok: infer_type(args[0]), ... }
        Call(path, args) if path ends with "Err" → Result { err: infer_type(args[0]), ... }
        MethodCall(recv, "map_err", _) → Result { ok: same, err: closure_return }
        Binary(Try, inner) → unwrap Result type from inner
        _ → Unknown
```

### 5.3 Phase 3: Inline Closure Compilation (Required)

Compile closure expressions inline within combinator calls:

```
compile_inline_closure(closure_expr):
    // Save current function state
    saved_locals ← current_locals

    // Add closure parameter as temporary local
    param_local ← alloc_temp_local()

    // Compile closure body
    compile_expr(closure_expr.body)

    // Result is on stack
```

---

## 6. Constraints & Invariants

### 6.1 Stack Discipline

```
I1: After any combinator call, stack contains exactly one Result (2 × i64)
I2: Closure evaluation must leave exactly one value on stack
I3: ? operator pops Result, pushes unwrapped value OR returns early
```

### 6.2 Memory Safety

```
I4: Result payloads are copied, not moved (no use-after-move in WASM)
I5: Closure environments are heap-allocated if they escape
I6: Temporary closure locals are reclaimed after combinator completes
```

### 6.3 Correctness

```
I7: map_err on Ok preserves exact Ok value (bitwise identical)
I8: map_err on Err applies closure exactly once
I9: ? operator propagates exact Err value (no transformation)
```

---

## 7. Error Conditions

| Condition | Error |
|-----------|-------|
| Unknown receiver type for combinator | "cannot call {method} on value of unknown type" |
| Closure arity mismatch | "map_err closure must take exactly 1 argument" |
| Closure capture of mutable local | "cannot capture mutable local in closure" |
| Nested Results (for now) | "Result<Result<...>> not yet supported" |

---

## 8. Integration Points

### 8.1 With Extern Blocks

Extern functions returning `Result[T, JsValue]?` need Result construction:

```sigil
// Source
extern "js" {
    ☉ rite get_item(this: &Storage!) -> Result[String, JsValue]?;
}

// Compiled: extern returns 2 × i64 (discriminant, payload)
```

### 8.2 With Try Operator

The `?` operator depends on Result tracking:

```sigil
≔ value = storage.get_item(key)?;

// Compiled:
call $Storage_get_item
local.tee $result_disc
i64.const 1
i64.eq
if
    local.get $result_payload
    local.get $result_disc
    return    ;; Early return with Err
end
local.get $result_payload
;; Continue with Ok value
```

---

## 9. Open Questions

1. **Heap vs Stack Results:** Should large Result payloads always go to heap?
   - Currently: All fit in i64 (pointer or small value)
   - Future: May need boxing for large payloads

2. **Error type erasure:** How much type info to preserve for debugging?
   - Currently: All errors are i64 (JsValue handles)
   - Future: Could embed type tags for better error messages

3. **Chained combinators:** Optimize `result.map_err(f).map_err(g)`?
   - Currently: Each combinator is independent
   - Future: Could fuse chains

---

## 10. Implementation Status

| Feature | Status | Notes |
|---------|--------|-------|
| `map_err` | :white_check_mark: | Inline closures supported |
| `is_ok` | :white_check_mark: | Implemented |
| `is_err` | :white_check_mark: | Implemented |
| `ok_or_else` | :white_check_mark: | Option to Result conversion |
| `ok_or` | :white_check_mark: | Option to Result conversion |
| `map` | :warning: | Stub only |
| `and_then` | :warning: | Stub only |
| `unwrap_or` | :warning: | Stub only |
| `?` operator | :x: | Requires Result type tracking |

---

## 11. Gap Discovered: Range Expressions

**Status:** ⚠️ **GAP IDENTIFIED**

During qliphoth compilation, range expressions (`0..10`, `[1..]`, `..=5`) are used
but not supported by the WASM backend. This blocks compilation of:
- `qliphoth-router` (string slicing: `&hash[1..]`)
- `qliphoth-sys/storage.sigil` (iteration: `0..self·length()`)

This is a separate spec requirement (WASM-RANGE-EXPRESSIONS-SPEC.md).

---

## 12. Revision History

| Version | Date | Changes |
|---------|------|---------|
| 0.1.0 | 2026-02-14 | Initial draft. Identified during qliphoth-sys WASM compilation. |
| 0.2.0 | 2026-02-14 | Implemented map_err, is_ok, is_err. Discovered range expression gap. |
| 0.3.0 | 2026-02-14 | Implemented ok_or_else, ok_or for Option→Result conversion. |
