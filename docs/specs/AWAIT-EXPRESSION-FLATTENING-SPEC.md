# Await Expression Flattening Spec

**Version:** 0.2.0
**Status:** Draft
**Authors:** Claude (Opus 4.5) + Human
**Date:** 2026-02-18
**Methodology:** SDD + Agent-TDD
**Depends On:** ASYNC-STATE-MACHINE-SPEC.md

---

## Abstract

This specification defines a transformation pass that "flattens" complex expressions containing await points into sequences of simple let-bindings. This pass runs BEFORE the state machine transformation, ensuring the state machine only needs to handle simple `≔ x = expr⌛` patterns.

**Sigil Await Syntax:** Sigil uses the hourglass symbol `⌛` for await operations, reflecting its polysynthetic design. The pipe syntax `|await` is also supported for ASCII compatibility.

---

## 1. Problem Statement

### 1.1 Current State

The state machine transformation (ASYNC-STATE-MACHINE-SPEC.md) handles:
- `≔ x = expr⌛` - direct await in let binding ✓
- `expr⌛` - await in statement position ✓
- Await in if/while bodies ✓

It does NOT handle:
- `≔ x = foo() + bar()⌛` - await nested in binary expression
- `f(a⌛, b⌛)` - multiple awaits in argument list
- `match fetch()⌛ { ... }` - await in match scrutinee
- `arr[index()⌛]` - await in index expression

### 1.2 Goal

Transform any expression containing await points into a sequence where:
1. Each await appears in simple `≔ __temp_N = expr⌛` form
2. Original expression is rewritten to reference the temporaries
3. Evaluation order is preserved (left-to-right)
4. The state machine transformation receives only simple patterns

---

## 2. Transformation Rules

### 2.1 Core Principle

**Await Hoisting:** Any subexpression `E⌛` that is NOT already in direct let-binding position must be hoisted to a preceding let-binding.

### 2.2 Transformation Examples

#### Binary Expression with Await

```sigil
// Input
≔ x = foo() + bar()⌛

// Output
≔ __pre_0 = foo()        // Evaluated before await, must be saved
≔ __await_0 = bar()⌛    // Suspension point
≔ x = __pre_0 + __await_0   // Uses saved values
```

Note: `foo()` is evaluated BEFORE `bar()⌛` due to left-to-right evaluation, so its result must be saved across the suspension point.

#### Multiple Awaits in Arguments

```sigil
// Input
f(a⌛, b⌛, c)

// Output
≔ __await_0 = a⌛
≔ __await_1 = b⌛
f(__await_0, __await_1, c)
```

#### Await in Match Scrutinee

```sigil
// Input
match fetch()⌛ {
    Ok(x) => x,
    Err(e) => default,
}

// Output
≔ __await_0 = fetch()⌛
match __await_0 {
    Ok(x) => x,
    Err(e) => default,
}
```

#### Await in Index Expression

```sigil
// Input
arr[compute_index()⌛]

// Output
≔ __await_0 = compute_index()⌛
arr[__await_0]
```

#### Chained Await in Method Call

```sigil
// Input
fetch()⌛.process()⌛.finish()

// Output
≔ __await_0 = fetch()⌛
≔ __await_1 = __await_0.process()⌛
__await_1.finish()
```

### 2.5 Short-Circuit Operators

Short-circuit operators (`||`, `&&`) have special semantics: the right-hand side may not evaluate at all. Naive hoisting would break this.

#### Logical Or (`||`)

```sigil
// Input
check_cache() || fetch_remote()⌛

// WRONG - always awaits even if cache hits
≔ __await_0 = fetch_remote()⌛
check_cache() || __await_0

// CORRECT - transform to if/else
if check_cache() { true } else { fetch_remote()⌛ }
```

#### Logical And (`&&`)

```sigil
// Input
validate() && submit()⌛

// CORRECT - transform to if/else
if validate() { submit()⌛ } else { false }
```

#### Both Sides Have Await

```sigil
// Input
check()⌛ || fallback()⌛

// Output - LHS await hoisted, RHS becomes if/else branch
≔ __await_0 = check()⌛
if __await_0 { true } else { fallback()⌛ }
```

**Rule:** Short-circuit operators with await in RHS are transformed to if/else expressions, preserving the short-circuit semantics.

### 2.6 Closures with Await

Closures are deferred computations - they execute later, not at definition time. Flattening inside a closure would change when the await happens.

```sigil
// User writes
≔ callback = || fetch()⌛

// If we flattened (WRONG - changes semantics)
≔ __await_0 = fetch()⌛    // Executes NOW at definition
≔ callback = || __await_0  // Just returns cached value

// Correct behavior: ERROR with guidance
// Error: await inside closure requires `async ||` syntax
// Hint: Use `async || { fetch()⌛ }` for an async closure,
//       or extract the await: `≔ x = fetch()⌛; || x`
```

**Rule:** Await inside a non-async closure is an error. The error message guides users to either:
1. Use `async ||` syntax for async closures
2. Extract the await outside the closure if capturing the result is intended

### 2.7 What Gets Hoisted

| Expression Type | Contains Await | Action |
|-----------------|----------------|--------|
| `expr⌛` in let position | Yes | Keep as-is (already simple) |
| `expr⌛` in statement | Yes | Keep as-is (already simple) |
| `expr⌛` nested in larger expr | Yes | Hoist to preceding let |
| Subexpr evaluated BEFORE an await | No | Hoist if sibling/parent contains await |
| Subexpr evaluated AFTER an await | No | Keep in place |

### 2.4 Evaluation Order Preservation

Sigil uses left-to-right evaluation. For `a + b`:
1. Evaluate `a`
2. Evaluate `b`
3. Apply `+`

If `b` contains await:
- `a` must be evaluated and SAVED before the await
- After resume, use saved `a` value

For `a + b + c` with left-associativity `(a + b) + c`:
- Evaluate `a`, evaluate `b`, add them → result1
- Evaluate `c`, add to result1

If `b` is `b()⌛`:
- Evaluate `a()` → must save across await
- Evaluate `b()⌛` → suspends
- Resume with `b_result`
- Compute `a_saved + b_result` → result1
- Evaluate `c()` → computed after await, no save needed
- Compute `result1 + c_result`

```sigil
// Input
a() + b()⌛ + c()

// Output
≔ __pre_0 = a()
≔ __await_0 = b()⌛
__pre_0 + __await_0 + c()
```

**Key insight:** Only values computed BEFORE an await AND used AFTER need hoisting. Values computed after resume don't need saving.

---

## 3. Algorithm

### 3.1 High-Level Approach

```
flatten_function(func: Function) -> Function:
    for each statement in func.body:
        flatten_stmt(stmt)

flatten_stmt(stmt: Stmt) -> [Stmt]:
    match stmt:
        Let { pattern, init }:
            (hoisted, new_init) = flatten_expr(init)
            return hoisted + [Let { pattern, new_init }]

        Semi(expr):
            (hoisted, new_expr) = flatten_expr(expr)
            return hoisted + [Semi(new_expr)]

        ...

flatten_expr(expr: Expr) -> (hoisted: [Stmt], result: Expr):
    // Returns statements to prepend, and the transformed expression

    if !contains_await(expr):
        return ([], expr)  // No transformation needed

    match expr:
        Await { inner }:
            // Direct await - check if inner needs flattening
            (inner_hoisted, inner_flat) = flatten_expr(inner)
            return (inner_hoisted, Await { inner_flat })

        Binary { left, op, right }:
            return flatten_binary(left, op, right)

        Call { func, args }:
            return flatten_call(func, args)

        // ... other cases
```

### 3.2 Binary Expression Flattening

```
flatten_binary(left: Expr, op: BinOp, right: Expr) -> (hoisted, result):
    // Special case: short-circuit operators
    if op == Or || op == And:
        return flatten_short_circuit(left, op, right)

    left_has_await = contains_await(left)
    right_has_await = contains_await(right)

    hoisted = []

    // Flatten left side
    (left_hoisted, left_flat) = flatten_expr(left)
    hoisted.extend(left_hoisted)

    // If right has await, left's value must be saved
    if right_has_await and !is_simple(left_flat):
        temp = fresh_pre_temp()
        hoisted.push(Let { temp, left_flat })
        left_flat = Ident(temp)

    // Flatten right side
    (right_hoisted, right_flat) = flatten_expr(right)
    hoisted.extend(right_hoisted)

    return (hoisted, Binary { left_flat, op, right_flat })
```

### 3.3 Short-Circuit Operator Flattening

Short-circuit operators require transformation to if/else to preserve semantics:

```
flatten_short_circuit(left: Expr, op: BinOp, right: Expr) -> (hoisted, result):
    left_has_await = contains_await(left)
    right_has_await = contains_await(right)

    // If neither side has await, no transformation needed
    if !left_has_await and !right_has_await:
        return ([], Binary { left, op, right })

    hoisted = []

    // Flatten left side (may contain await)
    (left_hoisted, left_flat) = flatten_expr(left)
    hoisted.extend(left_hoisted)

    // If right has no await, simple transformation
    if !right_has_await:
        if op == Or:
            // a || b  →  if a { true } else { b }
            return (hoisted, If { cond: left_flat, then: true, else: right })
        else:  // And
            // a && b  →  if a { b } else { false }
            return (hoisted, If { cond: left_flat, then: right, else: false })

    // Right has await - transform to if/else
    // The else branch will be further processed by state machine
    if op == Or:
        // a || b⌛  →  if a { true } else { b⌛ }
        result = If {
            cond: left_flat,
            then: Literal(true),
            else: right,  // Contains await, handled by state machine
        }
    else:  // And
        // a && b⌛  →  if a { b⌛ } else { false }
        result = If {
            cond: left_flat,
            then: right,  // Contains await, handled by state machine
            else: Literal(false),
        }

    return (hoisted, result)
```

### 3.4 Call Expression Flattening

```
flatten_call(func: Expr, args: [Expr]) -> (hoisted, result):
    hoisted = []
    new_args = []

    // Check if any argument has await
    any_await = args.any(contains_await)

    // Flatten function expression
    (func_hoisted, func_flat) = flatten_expr(func)
    hoisted.extend(func_hoisted)

    // If any arg has await, save func if complex
    if any_await and !is_simple(func_flat):
        temp = fresh_temp()
        hoisted.push(Let { temp, func_flat })
        func_flat = Ident(temp)

    // Process arguments left-to-right
    for (i, arg) in args.enumerate():
        (arg_hoisted, arg_flat) = flatten_expr(arg)
        hoisted.extend(arg_hoisted)

        // If later args have await, save this one
        if args[i+1..].any(contains_await) and !is_simple(arg_flat):
            temp = fresh_temp()
            hoisted.push(Let { temp, arg_flat })
            arg_flat = Ident(temp)

        new_args.push(arg_flat)

    return (hoisted, Call { func_flat, new_args })
```

### 3.5 Simple Expression Check

An expression is "simple" (doesn't need saving) if it's:
- An identifier (already a variable)
- A literal (can be re-evaluated)
- A constant path

```
is_simple(expr: Expr) -> bool:
    match expr:
        Ident(_) | Literal(_) | Path(_) => true
        _ => false
```

### 3.6 Closure Validation

Before flattening, check for await inside non-async closures:

```
validate_no_await_in_closure(expr: Expr) -> Result<(), Error>:
    match expr:
        Closure { is_async: false, body }:
            if contains_await(body):
                return Error {
                    message: "await inside closure requires `async ||` syntax",
                    hint: "Use `async || { ... }` for an async closure, " +
                          "or extract the await outside: `≔ x = expr⌛; || x`",
                    span: closure_span,
                }
            // Recursively check nested expressions (but not nested closures)
            for expr in body.expressions():
                if !is_closure(expr):
                    validate_no_await_in_closure(expr)?
            Ok(())

        Closure { is_async: true, .. }:
            // Async closures are allowed to contain await
            // (handled separately by async closure transformation)
            Ok(())

        _:
            // Recursively check all subexpressions
            for child in expr.children():
                validate_no_await_in_closure(child)?
            Ok(())
```

This validation runs before flattening begins, providing clear errors early.

---

## 4. Temporary Naming

### 4.1 Design Philosophy: Visibility Aids Reasoning

Compiler-generated temporaries are **intentionally visible**, not hidden. This serves agents and developers who need to:
- Debug async code by seeing suspension structure
- Trace data flow through await points
- Understand error messages that reference intermediate values

The flattened form shows the computation's structure explicitly:

```sigil
// Complex expression hides two suspension points
process(fetch_a()⌛, fetch_b()⌛)

// Flattened form reveals structure
≔ __await_0 = fetch_a()⌛   // Suspend 1
≔ __await_1 = fetch_b()⌛   // Suspend 2
process(__await_0, __await_1) // No suspension here
```

### 4.2 Reserved `__` Prefix

The double-underscore prefix `__` is **reserved for compiler-generated names**. User code cannot declare variables starting with `__`.

```sigil
// User code - ERROR
≔ __my_var = 42
// Error: identifier `__my_var` uses reserved prefix `__`
// Hint: The `__` prefix is reserved for compiler-generated temporaries.
//       Use a different name like `_my_var` or `my_var`.

// Compiler-generated - OK
≔ __await_0 = fetch()⌛
```

**Rationale:**
- Simple rule, easy to remember
- Prevents accidental collision with compiler temporaries
- Follows convention established by Rust, Python, C++
- Avoids complexity of true hygienic macro systems

### 4.3 Naming Scheme

| Prefix | Meaning | Example |
|--------|---------|---------|
| `__await_N` | Result of await expression N | `__await_0`, `__await_1` |
| `__pre_N` | Value saved before await N | `__pre_0`, `__pre_1` |

**Note:** The `__sc_N` prefix is reserved for potential future use with short-circuit temporaries.
Currently, short-circuit operators (`||`/`&&`) are transformed directly to if/else without
intermediate bindings, so this prefix is not generated by the flattening pass.

### 4.4 Fresh Name Generation

Maintain counters per function to generate unique names:

```rust
struct FlattenContext {
    await_counter: u32,
    pre_counter: u32,
}

impl FlattenContext {
    fn fresh_await_temp(&mut self) -> String {
        let n = self.await_counter;
        self.await_counter += 1;
        format!("__await_{}", n)
    }

    fn fresh_pre_temp(&mut self) -> String {
        let n = self.pre_counter;
        self.pre_counter += 1;
        format!("__pre_{}", n)
    }
}
```

### 4.5 Error Messages

When errors involve temporaries, messages should help trace back to the original code:

```
Error: type mismatch in `__await_1`
  --> src/main.sigil:42:15
   |
42 |     process(fetch_a()⌛, fetch_b()⌛)
   |                         ^^^^^^^^^^ expected `Data`, found `Error`
   |
   = note: `__await_1` is the result of the second await in this expression
```

---

## 5. Scope and Limitations

### 5.1 What This Pass Handles

- Binary expressions with await: `a + b⌛`
- Call expressions with await args: `f(a⌛, b)`
- Method calls with await: `x.method(a⌛)`
- Index expressions with await: `arr[i⌛]`
- Field access on await: `fetch()⌛.field`
- Match scrutinee with await: `match x⌛ { ... }`
- Unary expressions with await: `!check()⌛`

### 5.2 What This Pass Does NOT Handle

- Await inside match arms (handled by state machine)
- Await inside if/while bodies (handled by state machine)
- Await inside closures (separate concern)

### 5.3 Interaction with Other Passes

```
Parse → Flatten Await → State Machine Transform → Backend
              ↑                    ↑
          This spec         ASYNC-STATE-MACHINE-SPEC
```

---

## 6. Testing Strategy

### 6.1 Unit Tests

```rust
mod flatten_spec {
    // Basic cases
    fn spec_no_await_unchanged()
    fn spec_direct_await_unchanged()

    // Binary expressions
    fn spec_binary_right_await_hoists()
    fn spec_binary_left_await_hoists()
    fn spec_binary_both_await_hoists_in_order()
    fn spec_binary_chain_multiple_awaits()

    // Call expressions
    fn spec_call_single_await_arg()
    fn spec_call_multiple_await_args()
    fn spec_call_func_is_await()

    // Complex cases
    fn spec_nested_binary_with_await()
    fn spec_method_chain_with_await()
    fn spec_match_scrutinee_await()

    // Evaluation order
    fn spec_preserves_left_to_right_order()
    fn spec_only_saves_values_before_await()
}
```

### 6.2 Property Tests

```rust
// Flattening preserves await count
fn property_await_count_preserved(expr: Expr) {
    let flattened = flatten_expr(expr);
    assert_eq!(count_awaits(expr), count_awaits(flattened));
}

// Flattening produces only simple await patterns
fn property_all_awaits_simple(func: Function) {
    let flattened = flatten_function(func);
    for await in find_awaits(flattened) {
        assert!(is_direct_let_binding(await));
    }
}
```

---

## 7. Implementation Phases

### Phase 1: Core Infrastructure
- [ ] `FlattenContext` struct
- [ ] `contains_await` helper
- [ ] `is_simple` helper
- [ ] Fresh name generation

### Phase 2: Expression Flattening
- [ ] Binary expression flattening
- [ ] Call expression flattening
- [ ] Unary expression flattening

### Phase 3: Statement Flattening
- [ ] Let statement flattening
- [ ] Expression statement flattening
- [ ] Block flattening

### Phase 4: Complex Cases
- [ ] Method call flattening
- [ ] Index expression flattening
- [ ] Match scrutinee flattening
- [ ] Field access on await

### Phase 5: Integration
- [ ] Connect to async transformation pipeline
- [ ] End-to-end tests

---

## 8. Gap Log

| Date | Gap | Impact | Resolution |
|------|-----|--------|------------|
| | | | |

---

## 9. Resolved Design Questions

The following questions were resolved during design review:

### 9.1 Closures with Await ✓ RESOLVED

**Question:** If `|| x⌛` appears, should we flatten inside the closure?

**Decision:** **Error with helpful guidance.** Await inside a non-async closure is an error because flattening would change when the await executes (definition time vs call time).

**Rationale:** As an agent writing code, I want the compiler to be explicit rather than guess my intent. The error message guides to two valid solutions:
- `async || { ... }` for async closures
- Extract await outside if capturing the result

See Section 2.6 and Section 3.6 for details.

### 9.2 Short-Circuit Operators ✓ RESOLVED

**Question:** For `a() || b()⌛`, if `a()` returns true, `b()` never evaluates. Handle specially?

**Decision:** **Transform to if/else.** This is the only semantically correct interpretation:
- `a || b⌛` → `if a { true } else { b⌛ }`
- `a && b⌛` → `if a { b⌛ } else { false }`

**Rationale:** Short-circuit semantics are well-defined. The compiler should just do the right thing here rather than error, since the fix is mechanical.

See Section 2.5 and Section 3.3 for details.

### 9.3 Temporary Visibility ✓ RESOLVED

**Question:** Are temporaries visible to user code? Should they be hygienic?

**Decision:** **Reserved `__` prefix, intentionally visible.**
- User code cannot declare variables starting with `__`
- Compiler temporaries use `__await_N`, `__pre_N`, etc.
- Temporaries appear in debug output and error messages

**Rationale:** Visibility aids reasoning. As an agent debugging async code, I want to see the structure of suspension points. The flattened form is actually *easier* to reason about than the nested original.

See Section 4 for details.

---

## Revision History

| Version | Date | Changes |
|---------|------|---------|
| 0.1.0 | 2026-02-18 | Initial draft. Core transformation rules and algorithm. |
| 0.2.0 | 2026-02-18 | Resolved all open questions. Added closure validation, short-circuit handling, visibility philosophy. |
