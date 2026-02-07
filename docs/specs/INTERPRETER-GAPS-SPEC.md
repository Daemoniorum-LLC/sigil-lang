# Interpreter Gaps Specification

**Status**: Draft
**Version**: 0.1.0
**Date**: 2026-02-06
**Scope**: `/home/crook/dev/sigil-lang/parser/src/interpreter.rs`

## Overview

This spec documents gaps in the Sigil interpreter that prevent nihil-serve and other codebases from executing at runtime. The interpreter is feature-complete for 99% of test cases (745/749 passing), but specific patterns used in production code fail.

## Current State

| Metric | Value | Evidence |
|--------|-------|----------|
| Test pass rate | 99% (745/749) | `run_tests_rust.sh` |
| nihil tests | 416/416 | COMPILER_ISSUES.md |
| Source files failing | 3 | server.sigil, idioms_showcase.sigil, mnist_training.sigil |

## Gap 1: Pipe Method Dispatch Fallback

**Status**: ✅ FIXED (2026-02-06)
**Severity**: Critical (blocks nihil-serve)
**Error**: `Unknown pipe method: get`

**Fix**: Added fallback at `interpreter.rs:15306-15325` that delegates unknown pipe methods to `TypeName·method_name` lookup in globals.

### Mechanism

The interpreter handles `PipeOp::Method` at line 13839 of `interpreter.rs` with an explicit whitelist of supported methods:

```rust
PipeOp::Method { name, type_args: _, args } => {
    match name.name.as_str() {
        "collect" => Ok(value),
        "sum" | "Σ" => self.sum_values(value),
        "product" | "Π" => self.product_values(value),
        "len" => { ... }
        "reverse" => { ... }
        "iter" | "into_iter" => Ok(value),
        "enumerate" => { ... }
        "first" => { ... }
        "last" => { ... }
        "take" => { ... }
        "skip" => { ... }
        "join" => { ... }
        "all" => { ... }
        "any" => { ... }
        "map" => { ... }
        "filter" => { ... }
        "find" => { ... }
        // ... (80+ methods)
        _ => Err(RuntimeError::new(format!(
            "Unknown pipe method: {}", name.name  // Line 15307
        )))
    }
}
```

When a method isn't in the whitelist, the interpreter fails instead of delegating to regular method dispatch on the value.

### Affected Patterns

From nihil-serve/src/server.sigil:
```sigil
// Line 551-556: axum router methods
Router·new()
    |route("/v1/completions", post(Self·handle_completions))
    |route("/health", get(Self·handle_health))
    |with_state(this.clone())

// Line 381: iterator method
self.waiting_queue|drain_while{_ => condition}

// Line 407: reduction with fallback
this.running_batch|τ{r => r.kv_cache_blocks.len()}|max|unwrap_or(0)

// Line 761-763: time operations
SystemTime·now()|duration_since(UNIX_EPOCH)|unwrap|as_secs
```

### Methods Missing from Whitelist

| Method | Usage Context | Source |
|--------|---------------|--------|
| `route` | axum routing | server.sigil:551-555 |
| `with_state` | axum state | server.sigil:556 |
| `drain_while` | iterator | server.sigil:381 |
| `unwrap_or` | Option fallback | server.sigil:407 |
| `unwrap` | Option/Result | server.sigil:762 |
| `duration_since` | time | server.sigil:761 |
| `as_secs` | Duration | server.sigil:763 |
| `index` | tensor | server.sigil:362 |
| `index_put` | tensor | server.sigil:415 |
| `slice` | tensor | server.sigil:430 |
| `softmax` | tensor | server.sigil:440 |
| `cumsum` | tensor | server.sigil:442 |
| `gt` | tensor | server.sigil:443 |
| `masked_fill` | tensor | server.sigil:444 |
| `multinomial` | tensor | server.sigil:446 |
| `stack` | tensor | server.sigil:433 |
| `to_tensor` | conversion | server.sigil:368 |
| `replace` | string | adversarial.sigil:349 |
| `count` | iterator | adversarial.sigil:268 |

### Proposed Fix

Add a fallback at the end of the `PipeOp::Method` match that delegates to `eval_method_call`:

```rust
// At end of PipeOp::Method match, before the error:
_ => {
    // Fallback: treat |method(args) as value.method(args)
    self.eval_method_call(value, name.clone(), type_args.clone(), args.clone())
}
```

This preserves the optimization for known methods while enabling extensibility.

### Acceptance Criteria

1. `sigil run nihil-serve/src/server.sigil` executes without "Unknown pipe method" errors
2. All 745+ existing tests still pass
3. New TDD tests verify fallback behavior

---

## Gap 2: Type-Specialized Pipe Methods

**Status**: Open (secondary to Gap 1)
**Severity**: High
**Error**: `no method 'numel' on Array`

### Mechanism

Some pipe methods need type-specific dispatch. When `|numel` is called on a tensor, it should call the tensor's `numel()` method. But the whitelist-based approach can't handle domain-specific methods without becoming unmanageable.

The fallback solution from Gap 1 naturally addresses this by delegating unknown methods to the value's method table.

### Affected Patterns

```sigil
// examples/idioms_showcase.sigil
tensor|numel  // Needs tensor.numel()
tensor|shape  // Needs tensor.shape()
```

---

## Gap 3: Float Literal Suffixes

**Status**: ✅ ALREADY FIXED
**Severity**: Medium
**Error**: `Invalid float: 0.0f32`

**Note**: The lexer already supports float suffixes via regex at `lexer.rs:1079`:
```rust
#[regex(r"..._?(f16|f32|f64|f128)?", |lex| lex.slice().to_string())]
FloatLit(String),
```

All float suffix tests pass (`1.5f32`, `2.0f64`, `1.0e-5f32`).

### Mechanism

The lexer at `parser/src/lexer.rs` doesn't parse float literals with type suffixes like `0.0f32`, `1.0f64`.

### Affected Patterns

```sigil
≔ Δ sum = 0.0f32;   // Fails
2.0f32.powi(-14)    // Fails
```

### Workaround

Use explicit casts:
```sigil
≔ Δ sum = 0.0 as f32;
(2.0 as f32).powi(-14)
```

### Proposed Fix

Update lexer to recognize `f32`, `f64` suffixes on float literals:

```rust
// In lexer.rs, after parsing float:
if self.peek_char() == Some('f') {
    self.advance();
    if self.matches("32") {
        return Token::Float32(value);
    } else if self.matches("64") {
        return Token::Float64(value);
    }
}
```

---

## Gap 4: Const Generic Type Inference

**Status**: ✅ FIXED (2026-02-07)
**Severity**: Medium
**Error**: `undefined variable: N` (where N is a const generic parameter)

**Fix**: Added bidirectional inference from field type annotations, variable declarations,
and return type annotations. See `docs/specs/03C-CONST-GENERIC-INFERENCE.md` for full spec.

| Scenario | Status |
|----------|--------|
| Field initialization (`c: Container<42>` → `Container·new()`) | ✅ FIXED |
| Variable declaration (`≔ c: Container<10> = Container·new()`) | ✅ FIXED |
| Return type inference (`→ Container<5> { Container·new() }`) | ✅ FIXED |

### Root Cause

When calling a constructor without explicit const generics (`Container·new()` instead of `Container·<42>·new()`), the interpreter cannot infer the const generic values from context. The const generic parameters (`N`, `IN`, `OUT`, etc.) remain unbound, causing "undefined variable" errors.

### Affected Patterns

```sigil
// This fails - no const generic specified
sigil Wrapper {
    c: Container<42>,  // Field type specifies const generic
}
⊢ Wrapper {
    ☉ rite new() → This {
        This { c: Container·new() }  // ERROR: undefined variable: N
    }
}

// This works - explicit const generic
⊢ Wrapper {
    ☉ rite new() → This {
        This { c: Container·<42>·new() }  // OK
    }
}
```

### Workaround

Always specify const generics explicitly in constructor calls:

```sigil
// Instead of:
bn1: BatchNorm1d·new(0.1, 1e-5),

// Use:
bn1: BatchNorm1d·<512>·new(0.1, 1e-5),
```

### Proper Fix (Future)

Would require bidirectional type inference:
1. Track expected type from context (field assignment, variable declaration)
2. Extract const generics from expected type annotation
3. Propagate const generic values into the function call environment

This is a significant change to both the type checker (`typeck.rs`) and interpreter.

---

## Implementation Order

| Priority | Gap | Status | Notes |
|----------|-----|--------|-------|
| P0 | Pipe method fallback | ✅ FIXED | 20 lines added to interpreter.rs |
| P1 | Float literal suffixes | ✅ ALREADY FIXED | Lexer already supports f32/f64 suffixes |
| P2 | Const generic inference | Documented | Workaround: use explicit const generics |

---

## Agent-TDD Test Suite

**Location**: `/home/crook/dev/sigil-lang/jormungandr/tests/interpreter_gaps/`

**Test Status**: All gap tests now **PASS**.

| Test File | Gap | Status |
|-----------|-----|--------|
| `test_pipe_fallback.sg` | 1 | ✅ PASS |
| `test_axum_pattern.sg` | 1 | ✅ PASS |
| `test_iterator_methods.sg` | 1 | ✅ PASS |
| `test_float_suffixes.sg` | 3 | ✅ PASS (already fixed) |

**Verification Commands**:
```bash
# Run individual tests (expect failures until gaps are fixed)
./parser/target/release/sigil run jormungandr/tests/interpreter_gaps/test_pipe_fallback.sg
./parser/target/release/sigil run jormungandr/tests/interpreter_gaps/test_axum_pattern.sg
./parser/target/release/sigil run jormungandr/tests/interpreter_gaps/test_iterator_methods.sg

# These tests pass once the gap is fixed
# Success criteria: All tests print "All X tests passed!"
```

### Test: Pipe Method Fallback

```sigil
// test_pipe_fallback.sg

sigil Counter {
    value: i64,
}

⊢ Counter {
    ☉ rite new() → Self {
        Counter { value: 0 }
    }

    ☉ rite increment(&Δ self) → &Δ Self {
        self.value += 1;
        self
    }

    ☉ rite get(&self) → i64 {
        self.value
    }
}

#[test]
rite test_pipe_method_delegates_to_value_method() {
    ≔ Δ counter = Counter·new();

    // This should work: pipe delegates to Counter.increment()
    ≔ result = counter|increment|increment|get;

    assert_eq!(result, 2);
}

#[test]
rite test_pipe_unknown_method_gives_helpful_error() {
    ≔ counter = Counter·new();

    // This should fail with "no method 'nonexistent' on struct 'Counter'"
    // NOT "Unknown pipe method: nonexistent"
    ≔ result = counter|nonexistent();

    // This line shouldn't be reached
    assert!(nay);
}
```

### Test: Float Suffix Parsing

```sigil
// test_float_suffixes.sg

#[test]
rite test_f32_suffix() {
    ≔ x = 1.5f32;
    assert_eq!(x, 1.5);

    // Method calls should work
    ≔ y = 2.0f32.sqrt();
    assert!(y > 1.4 && y < 1.5);
}

#[test]
rite test_f64_suffix() {
    ≔ x = 1.5f64;
    assert_eq!(x, 1.5);
}

#[test]
rite test_suffix_in_expression() {
    ≔ result = 2.0f32.powi(-14);
    assert!(result > 0.0);
    assert!(result < 0.001);
}
```

---

## Open Questions

1. **Whitelist vs. fallback tradeoffs**: Should known methods remain optimized, or should we remove the whitelist entirely and always delegate?

2. **Error message clarity**: When the fallback fails, should the error say "no method on type X" or preserve both error paths for debugging?

3. **Tensor operations**: Should tensor-specific pipe methods (`|softmax`, `|cumsum`) be hardcoded for performance, or rely on method dispatch?

---

## References

- `/home/crook/dev/sigil-lang/parser/src/interpreter.rs` - Main interpreter (18,567 lines)
- `/home/crook/dev/nihil/COMPILER_ISSUES.md` - Nihil-specific gaps
- `/home/crook/dev/nihil/docs/specs/NIHIL-SERVE-INTEGRATION-SPEC.md` - Integration roadmap
- `/home/crook/dev/sigil-lang/parser/src/ast.rs:1439` - PipeOp enum definition
