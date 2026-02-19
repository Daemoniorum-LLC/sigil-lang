# Async State Machine Transformation Spec

**Version:** 0.1.0
**Status:** Draft
**Authors:** Claude (Opus 4.5) + Human
**Date:** 2026-02-18
**Methodology:** SDD + Agent-TDD

---

## Abstract

This specification defines the transformation of async functions with multiple await points into explicit state machines. The transformation occurs as an AST-to-IR pass, producing a `StateMachineIR` that can be compiled by any backend (WASM, LLVM, interpreter).

---

## 1. Problem Statement

### 1.1 Current State

Sigil supports async functions with the `async rite` syntax:

```sigil
async rite fetch_both(url1: str, url2: str) -> (Data, Data) {
    ≔ a = fetch(url1)|await;
    ≔ b = fetch(url2)|await;
    (a, b)
}
```

**Current implementation:** Uses Asyncify/JSPI for stack switching. Works for sequential awaits when runtime supports it, but:
- Requires specific runtime support (Asyncify or JSPI)
- Not portable to all WASM runtimes
- Couples async semantics to backend implementation

### 1.2 Goal

Transform async functions into explicit state machines that:
- Work on any runtime (no Asyncify dependency)
- Are backend-independent (same IR for WASM, LLVM, interpreter)
- Handle all control flow (straight-line, conditionals, loops)
- Preserve program semantics exactly

---

## 2. State Machine IR

### 2.1 Core Types

```sigil
// The state machine intermediate representation
struct StateMachineIR {
    name: str,                    // Original function name
    params: [(str, Type)],        // Original parameters
    result_type: Type,            // Return type
    states: [State],              // All states in the machine
    locals: [LocalDecl],          // All locals across all states
    frame_layout: FrameLayout,    // Memory layout for suspension
}

struct State {
    index: u32,                   // State number (0 = entry)
    is_entry: bool,               // Reachable from initial call
    is_resume: bool,              // Reachable from resume
    resume_binding: Option<str>,  // Variable to bind resume value to
    body: [Stmt],                 // Statements to execute in this state
    exit: StateExit,              // How this state exits
}

enum StateExit {
    // Suspend at await, resume in next_state
    Await {
        promise: Expr,
        next_state: u32,
        saved_locals: [str],      // Locals to save before suspend
    },

    // Return final value, function complete
    Return {
        value: Expr,
    },

    // Unconditional transition (e.g., end of loop body)
    Goto {
        target: u32,
    },

    // Conditional transition
    Branch {
        condition: Expr,
        then_state: u32,
        else_state: u32,
    },

    // Loop construct (condition checked at head)
    LoopHead {
        condition: Option<Expr>,  // None for infinite loop
        body_state: u32,
        exit_state: u32,
    },
}

struct LocalDecl {
    name: str,
    ty: Option<Type>,             // May be None if type not yet inferred
    defined_in_state: u32,        // First state where variable is defined
    live_until_state: u32,        // Last state where variable is used
}

struct FrameLayout {
    state_offset: u32,            // Offset of state field (always 0)
    locals_offset: u32,           // Offset where locals begin
    local_offsets: [(str, u32)],  // Name -> byte offset
    total_size: u32,              // Total frame size in bytes
}
```

### 2.2 Invariants

**INV-1: State Indices**
- State 0 is always the entry state
- All state indices are contiguous: 0, 1, 2, ..., N-1
- Every state is reachable from state 0

**INV-2: Entry/Resume Flags**
- State 0 has `is_entry = true`, `is_resume = false`
- All other states have `is_entry = false`, `is_resume = true`
- (Future: early return may create non-resume states)

**INV-3: Resume Binding**
- If `is_resume = true` and previous state had `StateExit::Await`, then `resume_binding` must be `Some(_)` if the await result is used
- Entry state has `resume_binding = None`

**INV-4: Control Flow Completeness**
- Every state has exactly one `StateExit`
- Every `StateExit` target state exists
- No cycles without at least one `Await` or `Return` (prevents infinite loops without suspension)

**INV-5: Local Liveness**
- A local is only read in states where `defined_in_state <= current <= live_until_state`
- Locals are saved to frame at `Await` if live across the await

---

## 3. Transformation Algorithm

### 3.1 Phase 1: Straight-Line Code

For functions with only sequential awaits (no control flow):

```
transform_straight_line(func: Function) -> StateMachineIR:
    ir = StateMachineIR::new(func.name, func.params, func.result_type)

    current_state = ir.new_state(is_entry: true)

    for stmt in func.body:
        match stmt:
            Let { name, value: Await { inner } }:
                // End current state with await
                saved = compute_live_locals(current_state)
                current_state.exit = Await {
                    promise: inner,
                    next_state: ir.next_state_idx(),
                    saved_locals: saved
                }

                // Start new state that binds resume value
                current_state = ir.new_state(is_resume: true)
                current_state.resume_binding = Some(name)
                ir.declare_local(name, infer_type(inner))

            Let { name, value }:
                current_state.body.push(stmt)
                ir.declare_local(name, infer_type(value))

            Expr(e) if contains_await(e):
                // Await in expression position (not let binding)
                // Must still create new state for continuation
                ...

            _:
                current_state.body.push(stmt)

    // Final state returns result
    current_state.exit = Return { value: last_expression }

    return ir
```

### 3.2 Phase 2: Conditionals

For `if`/`else` with awaits in branches:

```sigil
if condition {
    ≔ x = fetch(a)|await;
    use(x)
} else {
    ≔ y = fetch(b)|await;
    use(y)
}
after()
```

Transforms to:

```
State 0 (entry):
    body: []
    exit: Branch { condition, then_state: 1, else_state: 3 }

State 1 (then, pre-await):
    body: []
    exit: Await { promise: fetch(a), next_state: 2 }

State 2 (then, post-await):
    resume_binding: "x"
    body: [use(x)]
    exit: Goto { target: 5 }

State 3 (else, pre-await):
    body: []
    exit: Await { promise: fetch(b), next_state: 4 }

State 4 (else, post-await):
    resume_binding: "y"
    body: [use(y)]
    exit: Goto { target: 5 }

State 5 (join):
    body: [after()]
    exit: Return { ... }
```

### 3.3 Phase 3: Loops

For `while` with await in body:

```sigil
while has_more() {
    ≔ item = fetch_next()|await;
    process(item);
}
done()
```

Transforms to:

```
State 0 (entry):
    body: []
    exit: Goto { target: 1 }

State 1 (loop head):
    body: []
    exit: LoopHead { condition: has_more(), body_state: 2, exit_state: 4 }

State 2 (loop body, pre-await):
    body: []
    exit: Await { promise: fetch_next(), next_state: 3 }

State 3 (loop body, post-await):
    resume_binding: "item"
    body: [process(item)]
    exit: Goto { target: 1 }  // Back to loop head

State 4 (after loop):
    body: [done()]
    exit: Return { ... }
```

**Key insight:** Loop head is re-executed on each iteration. The `Goto` back to state 1 handles this.

### 3.4 Phase 4: Complex Control Flow

- **Break/Continue:** Target specific states
- **Match expressions:** Similar to conditionals, with multiple branch targets
- **Labeled loops:** Track loop head states by label name
- **Early return:** Direct transition to a return state

---

## 4. Runtime Contract

### 4.1 Function Signature

Transformed functions have this effective signature:

```
(frame_ptr: i32, resume_value: i64) -> i64
```

- **Initial call:** Runtime passes `frame_ptr = 0`, `resume_value = 0`
- **Resume call:** Runtime passes allocated frame pointer, resolved value

### 4.2 Return Values

Return value encoding (i64):

| Bits 63-32 | Bits 31-0 | Meaning |
|------------|-----------|---------|
| 0          | value     | Complete, final result is `value` |
| 1          | cont_ptr  | Suspended, continuation at `cont_ptr` |

The continuation structure contains the frame pointer and the promise to await.

### 4.3 Frame Layout

```
Offset 0:    state (i32)     - Current state index
Offset 4:    padding (i32)   - Alignment
Offset 8:    local_0 (i64)   - First local
Offset 16:   local_1 (i64)   - Second local
...
```

All locals are stored as i64 for simplicity. Type information guides interpretation.

### 4.4 Runtime Imports

The compiled WASM module expects these imports from the runtime:

```
// Allocate memory for the suspension frame
alloc(size: i32) -> i32

// Register a continuation with the runtime
// - frame_ptr: Pointer to the suspension frame
// - state: Next state to resume at
// - promise: The promise/future value being awaited (passed to runtime scheduler)
// Returns: Continuation pointer to encode in return value
async_create_continuation(frame_ptr: i32, state: i32, promise: i64) -> i32
```

The runtime is responsible for:
1. Tracking the continuation and associated promise
2. Polling/waiting on the promise
3. Calling the function again with (frame_ptr, resolved_value) when the promise resolves

---

## 5. Backend Compilation

### 5.1 WASM Backend

Each backend implements:

```rust
fn compile_state_machine(ir: &StateMachineIR) -> Result<CompiledFunction>
```

The WASM implementation:
1. Emit prologue (initial vs resume detection)
2. Emit br_table dispatcher
3. For each state:
   - Emit body statements
   - Emit exit handling
4. Emit epilogue

### 5.2 LLVM Backend

Similar structure, using LLVM basic blocks for states.

### 5.3 Interpreter

Can directly interpret the IR, maintaining state across suspension points.

---

## 6. Testing Strategy (Agent-TDD)

### 6.1 Specification Tests

Tests that define correct behavior:

```sigil
mod async_transform_spec {
    // Phase 1: Straight-line
    fn spec_single_await_creates_two_states()
    fn spec_two_awaits_creates_three_states()
    fn spec_await_result_bound_in_next_state()
    fn spec_locals_live_across_await_are_saved()

    // Phase 2: Conditionals
    fn spec_if_with_await_creates_branch_states()
    fn spec_both_branches_join_at_common_state()
    fn spec_nested_if_await_handled()

    // Phase 3: Loops
    fn spec_while_await_creates_loop_head()
    fn spec_loop_body_returns_to_head()
    fn spec_break_targets_exit_state()
    fn spec_continue_targets_head_state()

    // Invariants
    fn spec_all_states_reachable()
    fn spec_no_dangling_state_references()
    fn spec_entry_state_is_zero()
}
```

### 6.2 Property Tests

```sigil
// Property: Transformation preserves evaluation semantics
fn property_roundtrip_semantics<F: AsyncFn>(f: F, inputs: [Any]) {
    let original_result = run_with_asyncify(f, inputs);
    let transformed = transform(f);
    let sm_result = run_state_machine(transformed, inputs);
    assert_eq(original_result, sm_result);
}

// Property: Frame size is sufficient for all states
fn property_frame_fits_all_locals(ir: StateMachineIR) {
    for state in ir.states {
        for local in state.live_locals() {
            assert(ir.frame_layout.has_offset(local));
        }
    }
}
```

### 6.3 Boundary Tests

```sigil
// Test at API boundaries
fn boundary_malformed_ast_returns_error()
fn boundary_unsupported_expr_documented()
fn boundary_max_states_handled()
```

---

## 7. Implementation Phases

### Phase 0: IR Types ✓
- [x] Design IR structures
- [x] Implement IR types in Rust
- [x] Unit tests for IR construction

### Phase 1: Straight-Line Code ✓
- [x] Transformation for sequential awaits
- [x] Specification tests (26 passing)
- [x] Validation: INV-1, INV-2, INV-3, INV-4, Unreachable check
- [ ] WASM backend compilation (blocked on backend integration)
- [ ] End-to-end test: two sequential awaits

### Phase 2: Conditionals ✓
- [x] Branch handling in transformation
- [x] Join point detection
- [x] Nested if support
- [x] Test: if/else with awaits in both branches (9 tests passing)

### Phase 3: Loops ✓
- [x] While/loop handling with LoopHead exit
- [x] Break/continue targeting loop head/exit states
- [x] Loop stack for nested loop support
- [x] Explicit errors for unsupported cases (break with value, for loops)
- [x] Test: 11 specification tests passing

### Phase 4: Complex Cases
- [ ] Match expressions
- [ ] Labeled break/continue (basic support implemented in Phase 3)
- [ ] Nested control flow
- [ ] Early return
- [ ] Break with value (loop returning value)

---

## 8. Gap Log

*This section documents gaps discovered during implementation (per SDD methodology).*

| Date | Gap | Impact | Resolution |
|------|-----|--------|------------|
| 2026-02-18 | AST structure differs from assumed pseudocode | Spec Section 3 uses `Let { name, value }` but actual AST is `Stmt::Let { pattern, ty, init }` | Updated implementation to use actual AST. Spec pseudocode remains idealized for clarity. |
| 2026-02-18 | `LocalDecl.ty` should be Optional | Spec shows `ty: Type` but type may not be known at transformation time | Implementation uses `Option<TypeExpr>`. Spec Section 2.1 updated to note this. |
| 2026-02-18 | Trailing await needs explicit binding | Magic `__resume_value` was referenced without declaration | Implemented proper binding with `RESUME_VALUE_BINDING` constant and explicit `declare_local` call. |
| 2026-02-18 | Join state exit not set without trailing expr | If statement ends with `if { await } else { await };` and no trailing expression, join state had Unreachable exit | Added fallback in `transform_block` to set Return { unit } when no trailing expression and current state has Unreachable exit. |
| 2026-02-18 | Non-await branch value lost in LetBinding context | For `let x = if { 1 } else { await }`, the value `1` was discarded when patching Return to Goto | For non-await branches, now generate synthetic `let` statement in body to assign value before Goto. |
| 2026-02-18 | Local declaration timing inconsistent | Await branches declared in join state, but should declare where value is first assigned | Locals now declared in resume state (for await) or branch state (for non-await). Added `declare_local_if_new` helper to prevent duplicates. |
| 2026-02-18 | No validation for orphaned states | States unreachable from entry could exist due to transformation bugs | Added reachability check to validation using BFS from entry state. |
| 2026-02-18 | Loop in trailing position not handled | `loop { await; break }` as function's final expression wasn't transformed | Added loop handling in `transform_trailing_expr` alongside existing if handling. |
| 2026-02-18 | Loop context needed for break/continue | Break/continue must know which loop they target | Added `LoopContext` struct and `loop_stack` field to track enclosing loops with head/exit state indices and optional labels. |
| 2026-02-18 | Break value silently ignored | `break x` would lose the value `x` | Return explicit error: "break with value not yet supported". Deferred to Phase 4. |
| 2026-02-18 | For loops with await not handled | `for` loops would hit generic unsupported error | Added explicit check and clear error: "For loops with await not yet supported". |
| 2026-02-18 | LoopContext defined after use | Type defined after struct that references it | Moved `LoopContext` definition before `AsyncTransformer` struct. |

---

## 9. Open Questions

1. **~~For expressions with await:~~** ✓ RESOLVED - See AWAIT-EXPRESSION-FLATTENING-SPEC.md
   - `≔ x = foo() + bar()⌛` is handled by expression flattening pre-pass
   - Values computed before await are hoisted to temporaries

2. **~~Multiple awaits in expression:~~** ✓ RESOLVED - See AWAIT-EXPRESSION-FLATTENING-SPEC.md
   - `f(a⌛, b⌛)` flattens to sequential `≔ __await_N = ...` bindings
   - Left-to-right evaluation order preserved

3. **~~Await in match scrutinee:~~** ✓ RESOLVED - See AWAIT-EXPRESSION-FLATTENING-SPEC.md
   - `match fetch()⌛ { ... }` flattens to `≔ __await_0 = fetch()⌛; match __await_0 { ... }`
   - Await happens before match, state machine handles match normally

4. **Error handling:** If an awaited promise rejects, how does the state machine propagate the error?
   - DECISION: Errors are values (Result types). State machine doesn't special-case errors.
   - Runtime contract may pass tagged union (success/error) as resume_value.
   - Error propagation via `?` operator is a separate desugaring concern.

---

## Revision History

| Version | Date | Changes |
|---------|------|---------|
| 0.1.0 | 2026-02-18 | Initial draft. Designed IR structure and transformation algorithm outline. |
| 0.2.0 | 2026-02-18 | Phase 1 complete: straight-line code, validation, 26 tests. |
| 0.3.0 | 2026-02-18 | Phase 2 complete: conditionals with if/else, join states, 9 tests. |
| 0.4.0 | 2026-02-18 | Phase 3 complete: while/loop, break/continue, nested loops, 8 tests. Total: 43 tests passing. |
| 0.4.1 | 2026-02-18 | Phase 3 review fixes: explicit errors for break-with-value and for-loops, continue test, code cleanup. Total: 46 tests passing. |
| 0.5.0 | 2026-02-18 | Resolved open questions 1-3 via AWAIT-EXPRESSION-FLATTENING-SPEC.md. Documented error handling decision. |
