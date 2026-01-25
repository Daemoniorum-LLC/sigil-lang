# CG-015 Investigation Report: Mutable Self Pass-by-Value Runtime Blocker

> **Status**: Investigation Complete
> **Priority**: Critical (Bootstrap Blocker)
> **Author**: Claude Code
> **Date**: 2025-12-14

---

## Executive Summary

The bootstrap compiler compiles all 13 modules to C successfully but **hangs at runtime** due to `&mut self` methods receiving `self` by value. Mutations to struct fields only affect local copies, causing infinite loops in the lexer.

**Root Cause**: C codegen passes `SigilValue self` by value instead of `SigilValue* self` by pointer for mutable methods.

---

## 1. Problem Analysis

### 1.1 The Bug

```c
// CURRENT (WRONG): self passed by value
SigilValue sigil_Lexer____advance(SigilValue self) {
    // Get field
    SigilValue pos = sigil_struct_field(self, "pos");
    // Increment
    SigilValue new_pos = sigil_int(pos.v.i + 1);
    // Set field - MODIFIES LOCAL COPY ONLY!
    sigil_struct_set_field(&self, "pos", new_pos);
    return result;
}  // self is discarded, mutations lost

// Caller code:
SigilValue lexer = ...;
SigilValue c = sigil_Lexer____advance(lexer);  // lexer.pos unchanged!
```

### 1.2 Impact Scope

**198 occurrences** of `mut self` across 8 modules:

| Module | Count | Critical Methods |
|--------|-------|------------------|
| `lexer.sg` | 43 | `advance()`, `next_token()`, `peek()`, all `lex_*()` |
| `parser.sg` | 70 | `advance()`, `expect()`, all `parse_*()` |
| `codegen.sg` | 25 | `line()`, `emit()`, `fresh_temp()`, `indent_*()` |
| `typeck.sg` | 24 | `push_scope()`, `pop_scope()`, `error()`, `define()` |
| `interp.sg` | 11 | `eval_*()`, `call_*()` |
| `lower.sg` | 4 | `lower_*()` |
| `driver.sg` | 14 | `run()`, `compile()` |
| `runtime.sg` | 7 | Various helpers |

### 1.3 Mutation Categories

1. **Position Tracking** (Lexer/Parser): `self.pos += 1`
2. **Buffer Accumulation** (CodeGen): `self.output.push_str(...)`
3. **Counter Increments** (CodeGen): `self.temp_counter += 1`
4. **Scope Management** (TypeChecker): `self.env = new_scope`
5. **Cache Updates** (Lexer): `self.peeked = Some(token)`
6. **Error Accumulation** (TypeChecker): `self.errors.push(err)`
7. **State Flags** (Parser): `self.in_condition = true`

---

## 2. Semantic Model Analysis

### 2.1 How the Rust Interpreter Handles This

The Rust interpreter uses **interior mutability** via `RefCell`:

```rust
// Value::Struct uses RefCell for fields
Value::Struct {
    name: String,
    fields: Rc<RefCell<HashMap<String, Value>>>,
}

// Mutations go through borrow_mut()
if let Value::Struct { fields, .. } = &receiver {
    fields.borrow_mut().insert(field_name, new_value);
}
```

**Key insight**: The interpreter **doesn't distinguish** `&self` from `&mut self` at runtime. It relies on Rust's type system for safety, but allows all mutations through `RefCell`.

### 2.2 How Other Languages Solve This

| Language | Strategy | C Output |
|----------|----------|----------|
| **Rust** | Explicit `&mut self` → pointer | `void vec_push(Vec* v, T x)` |
| **Go** | `func (v *Type)` → pointer receiver | `void Type_method(Type* v)` |
| **C++** | Implicit `this` pointer | `void Class::method()` |
| **Swift** | `mutating` keyword → inout | Optimized in-place |

**Consensus**: Mutable methods receive a **pointer** to the receiver, not a copy.

### 2.3 Recommended Semantic Model for Sigil

**Option A: Pass by Pointer (Recommended)**

```c
// Mutable method: pass pointer
SigilValue* sigil_Lexer____advance(SigilValue* self) {
    SigilValue pos = sigil_struct_field(*self, "pos");
    SigilValue new_pos = sigil_int(pos.v.i + 1);
    sigil_struct_set_field(self, "pos", new_pos);  // Now modifies original!
    return &result;
}

// Call site:
sigil_Lexer____advance(&lexer);  // lexer.pos is updated
```

**Option B: Return Modified Self (Functional)**

```c
// Mutable method: return modified value
SigilValue sigil_Lexer____advance(SigilValue self) {
    // ... mutations ...
    return self;  // Return modified copy
}

// Call site:
lexer = sigil_Lexer____advance(lexer);  // Explicit reassignment
```

**Recommendation**: Use **Option A (Pass by Pointer)** because:
1. Matches Rust/Go/C++ conventions
2. More efficient (no copying large structs)
3. Works naturally with nested field mutations
4. Doesn't require return type changes

---

## 3. Required Changes

### 3.1 IR Changes (`ir.sg`)

Add receiver mutability tracking:

```sigil
pub enum ReceiverKind {
    Owned,           // self (move)
    Borrowed,        // &self (immutable borrow)
    MutableBorrowed, // &mut self (mutable borrow)
}

pub struct IrFunction {
    // ... existing fields ...
    pub receiver_kind: ?ReceiverKind,  // NEW: Track receiver mutability
}

pub struct IrMethodCall {
    // ... existing fields ...
    pub receiver_mutable: !bool,  // NEW: Is receiver mutated?
}
```

### 3.2 Parser Changes (`parser.sg`)

Track `mut self` in parameter parsing:

```sigil
fn parse_self_param(mut self) -> !Param {
    let mutable = self.consume_if(&Token::Mut);
    self.expect(&Token::SelfLower)?;
    // ... return Param with mutable flag ...
}
```

### 3.3 Lowering Changes (`lower.sg`)

Propagate mutability to IR:

```sigil
fn lower_impl_method(ctx: &mut LowerCtx, method: &Function) -> IrFunction {
    let receiver_kind = if method.params[0].is_mut_self() {
        ReceiverKind::MutableBorrowed
    } else if method.params[0].is_ref_self() {
        ReceiverKind::Borrowed
    } else {
        ReceiverKind::Owned
    };
    // ... include receiver_kind in IrFunction ...
}
```

### 3.4 Codegen Changes (`codegen.sg`)

Generate correct C signatures:

```sigil
fn emit_function_signature(mut self, func: &IrFunction) {
    match func.receiver_kind {
        ?ReceiverKind::MutableBorrowed => {
            // Mutable method: pointer parameter
            self.emit("SigilValue ");
            self.emit(&func.name);
            self.emit("(SigilValue* self");
            // ... other params ...
        }
        _ => {
            // Non-mutable: value parameter
            self.emit("SigilValue ");
            self.emit(&func.name);
            self.emit("(SigilValue self");
            // ... other params ...
        }
    }
}

fn emit_method_call(mut self, call: &IrMethodCall) {
    if call.receiver_mutable {
        // Pass address for mutable receiver
        self.emit("&");
    }
    self.emit_expr(call.receiver);
    // ...
}
```

---

## 4. TDD Test Cases

### 4.1 Test File: `test_cg015_mut_self.sg`

```sigil
//! Test CG-015: Mutable self methods must persist mutations
//!
//! Bug: Methods with `mut self` receive self by value in C codegen.
//! Mutations to struct fields only affect local copies.
//! Fix: Pass self by pointer for `mut self` methods.

use crate::ir::*;
use crate::codegen::CodeGen;

// ============================================================================
// Test Helpers
// ============================================================================

fn generate(module: !IrModule) -> !String {
    let mut gen = CodeGen::new();
    gen.generate(module)
}

fn assert_contains(code: !&str, pattern: !&str, name: !&str) {
    if !code.contains(pattern) {
        panic(format!("FAIL {}: expected '{}' not found\nOutput:\n{}", name, pattern, code));
    }
}

fn assert_not_contains(code: !&str, pattern: !&str, name: !&str) {
    if code.contains(pattern) {
        panic(format!("FAIL {}: forbidden '{}' found\nOutput:\n{}", name, pattern, code));
    }
}

// ============================================================================
// TEST 1: Mutable method signature uses pointer
// ============================================================================

fn test_cg015_mut_self_signature() -> !bool {
    print("  [CG-015.1] Mutable self generates pointer signature... ");

    let mut module = IrModule::new("test.sg".to_string());
    module.functions.push(IrFunction {
        name: "Counter::increment".to_string(),
        id: "Counter::increment".to_string(),
        visibility: IrVisibility::Public,
        generics: [],
        params: [
            IrParam {
                name: "self".to_string(),
                ty: IrType::Named { name: "Counter".to_string(), generics: [] },
                evidence: IrEvidence::Known,
                mutable: true,  // <-- KEY: mutable self
            },
        ],
        return_type: IrType::Unit,
        return_evidence: IrEvidence::Known,
        receiver_kind: ?ReceiverKind::MutableBorrowed,  // <-- NEW FIELD
        body: ?IrOperation::Block {
            statements: [
                // self.count = self.count + 1
                IrOperation::Assign {
                    target: Box::new(IrOperation::Field {
                        expr: Box::new(IrOperation::Var {
                            name: "self".to_string(),
                            id: "self".to_string(),
                            ty: IrType::Named { name: "Counter".to_string(), generics: [] },
                            evidence: IrEvidence::Known,
                        }),
                        field: "count".to_string(),
                        ty: IrType::Int { width: 64, signed: true },
                        evidence: IrEvidence::Known,
                    }),
                    value: Box::new(IrOperation::Binary {
                        operator: BinaryOp::Add,
                        left: Box::new(IrOperation::Field {
                            expr: Box::new(IrOperation::Var {
                                name: "self".to_string(),
                                id: "self".to_string(),
                                ty: IrType::Named { name: "Counter".to_string(), generics: [] },
                                evidence: IrEvidence::Known,
                            }),
                            field: "count".to_string(),
                            ty: IrType::Int { width: 64, signed: true },
                            evidence: IrEvidence::Known,
                        }),
                        right: Box::new(IrOperation::Literal {
                            variant: LiteralVariant::Int,
                            value: LiteralValue::Int(1),
                            ty: IrType::Int { width: 64, signed: true },
                            evidence: IrEvidence::Known,
                        }),
                        ty: IrType::Int { width: 64, signed: true },
                        evidence: IrEvidence::Known,
                    }),
                    evidence: IrEvidence::Known,
                },
            ],
            ty: IrType::Unit,
            evidence: IrEvidence::Known,
        },
        is_async: false,
        is_unsafe: false,
        span: null,
    });

    let code = generate(module);

    // MUST have pointer parameter for mutable self
    assert_contains(&code, "SigilValue* self", "CG-015.1a: mutable self should be pointer");

    // MUST NOT have value parameter
    assert_not_contains(&code, "SigilValue self)", "CG-015.1b: should not pass self by value");

    println("PASS");
    true
}

// ============================================================================
// TEST 2: Mutable method call passes address
// ============================================================================

fn test_cg015_mut_self_call_site() -> !bool {
    print("  [CG-015.2] Mutable method call passes address... ");

    let mut module = IrModule::new("test.sg".to_string());
    module.functions.push(IrFunction {
        name: "test_call".to_string(),
        id: "test_call".to_string(),
        visibility: IrVisibility::Public,
        generics: [],
        params: [],
        return_type: IrType::Unit,
        return_evidence: IrEvidence::Known,
        receiver_kind: null,
        body: ?IrOperation::Block {
            statements: [
                // let counter = Counter { count: 0 };
                IrOperation::Let {
                    pattern: IrPattern::Ident { name: "counter".to_string(), mutable: true, evidence: null },
                    type_annotation: null,
                    init: Box::new(IrOperation::StructInit {
                        name: "Counter".to_string(),
                        fields: [("count".to_string(), IrOperation::Literal {
                            variant: LiteralVariant::Int,
                            value: LiteralValue::Int(0),
                            ty: IrType::Int { width: 64, signed: true },
                            evidence: IrEvidence::Known,
                        })],
                        ty: IrType::Named { name: "Counter".to_string(), generics: [] },
                        evidence: IrEvidence::Known,
                    }),
                    evidence: IrEvidence::Known,
                },
                // counter.increment()  -- should pass &counter
                IrOperation::MethodCall {
                    receiver: Box::new(IrOperation::Var {
                        name: "counter".to_string(),
                        id: "counter".to_string(),
                        ty: IrType::Named { name: "Counter".to_string(), generics: [] },
                        evidence: IrEvidence::Known,
                    }),
                    method: "increment".to_string(),
                    args: [],
                    type_args: [],
                    ty: IrType::Unit,
                    evidence: IrEvidence::Known,
                    receiver_mutable: true,  // <-- NEW FIELD
                },
            ],
            ty: IrType::Unit,
            evidence: IrEvidence::Known,
        },
        is_async: false,
        is_unsafe: false,
        span: null,
    });

    let code = generate(module);

    // Call site MUST pass address
    assert_contains(&code, "&counter", "CG-015.2a: should pass &counter for mutable method");

    // Call site MUST NOT pass by value
    assert_not_contains(&code, "increment(counter)", "CG-015.2b: should not pass counter by value");

    println("PASS");
    true
}

// ============================================================================
// TEST 3: Immutable method still passes by value
// ============================================================================

fn test_cg015_immut_self_by_value() -> !bool {
    print("  [CG-015.3] Immutable self passes by value... ");

    let mut module = IrModule::new("test.sg".to_string());
    module.functions.push(IrFunction {
        name: "Counter::get".to_string(),
        id: "Counter::get".to_string(),
        visibility: IrVisibility::Public,
        generics: [],
        params: [
            IrParam {
                name: "self".to_string(),
                ty: IrType::Named { name: "Counter".to_string(), generics: [] },
                evidence: IrEvidence::Known,
                mutable: false,  // <-- KEY: immutable self
            },
        ],
        return_type: IrType::Int { width: 64, signed: true },
        return_evidence: IrEvidence::Known,
        receiver_kind: ?ReceiverKind::Borrowed,  // <-- Borrowed, not MutableBorrowed
        body: ?IrOperation::Return {
            value: ?Box::new(IrOperation::Field {
                expr: Box::new(IrOperation::Var {
                    name: "self".to_string(),
                    id: "self".to_string(),
                    ty: IrType::Named { name: "Counter".to_string(), generics: [] },
                    evidence: IrEvidence::Known,
                }),
                field: "count".to_string(),
                ty: IrType::Int { width: 64, signed: true },
                evidence: IrEvidence::Known,
            }),
            evidence: IrEvidence::Known,
        },
        is_async: false,
        is_unsafe: false,
        span: null,
    });

    let code = generate(module);

    // Immutable self SHOULD be passed by value (optimization OK)
    assert_contains(&code, "SigilValue self)", "CG-015.3: immutable self can be by value");

    println("PASS");
    true
}

// ============================================================================
// TEST 4: Field mutation through pointer dereference
// ============================================================================

fn test_cg015_field_mutation_via_pointer() -> !bool {
    print("  [CG-015.4] Field mutation dereferences pointer... ");

    let mut module = IrModule::new("test.sg".to_string());
    module.functions.push(IrFunction {
        name: "Lexer::advance".to_string(),
        id: "Lexer::advance".to_string(),
        visibility: IrVisibility::Public,
        generics: [],
        params: [
            IrParam {
                name: "self".to_string(),
                ty: IrType::Named { name: "Lexer".to_string(), generics: [] },
                evidence: IrEvidence::Known,
                mutable: true,
            },
        ],
        return_type: IrType::Unit,
        return_evidence: IrEvidence::Known,
        receiver_kind: ?ReceiverKind::MutableBorrowed,
        body: ?IrOperation::Assign {
            target: Box::new(IrOperation::Field {
                expr: Box::new(IrOperation::Var {
                    name: "self".to_string(),
                    id: "self".to_string(),
                    ty: IrType::Named { name: "Lexer".to_string(), generics: [] },
                    evidence: IrEvidence::Known,
                }),
                field: "pos".to_string(),
                ty: IrType::Int { width: 64, signed: false },
                evidence: IrEvidence::Known,
            }),
            value: Box::new(IrOperation::Literal {
                variant: LiteralVariant::Int,
                value: LiteralValue::Int(42),
                ty: IrType::Int { width: 64, signed: false },
                evidence: IrEvidence::Known,
            }),
            evidence: IrEvidence::Known,
        },
        is_async: false,
        is_unsafe: false,
        span: null,
    });

    let code = generate(module);

    // Field set MUST pass pointer to sigil_struct_set_field
    assert_contains(&code, "sigil_struct_set_field(self,", "CG-015.4a: should pass self pointer to set_field");

    // Field GET must dereference: *self
    assert_contains(&code, "*self", "CG-015.4b: should dereference self for field access");

    println("PASS");
    true
}

// ============================================================================
// TEST 5: Lexer advance integration test
// ============================================================================

fn test_cg015_lexer_advance_integration() -> !bool {
    print("  [CG-015.5] Lexer::advance integration... ");

    // This test simulates the actual lexer advance pattern that causes infinite loops
    let mut module = IrModule::new("test.sg".to_string());

    // Add Lexer struct type
    module.types.push(IrTypeDef::Struct {
        name: "Lexer".to_string(),
        generics: [],
        fields: [
            IrField { name: "source".to_string(), ty: IrType::Str, evidence: IrEvidence::Known, visibility: IrVisibility::Public },
            IrField { name: "pos".to_string(), ty: IrType::Int { width: 64, signed: false }, evidence: IrEvidence::Known, visibility: IrVisibility::Public },
        ],
        visibility: IrVisibility::Public,
    });

    // Add advance method
    module.functions.push(IrFunction {
        name: "Lexer::advance".to_string(),
        id: "Lexer::advance".to_string(),
        visibility: IrVisibility::Public,
        generics: [],
        params: [
            IrParam {
                name: "self".to_string(),
                ty: IrType::Named { name: "Lexer".to_string(), generics: [] },
                evidence: IrEvidence::Known,
                mutable: true,
            },
        ],
        return_type: IrType::Unit,
        return_evidence: IrEvidence::Known,
        receiver_kind: ?ReceiverKind::MutableBorrowed,
        body: ?IrOperation::Assign {
            target: Box::new(IrOperation::Field {
                expr: Box::new(IrOperation::Var {
                    name: "self".to_string(), id: "self".to_string(),
                    ty: IrType::Named { name: "Lexer".to_string(), generics: [] },
                    evidence: IrEvidence::Known,
                }),
                field: "pos".to_string(),
                ty: IrType::Int { width: 64, signed: false },
                evidence: IrEvidence::Known,
            }),
            value: Box::new(IrOperation::Binary {
                operator: BinaryOp::Add,
                left: Box::new(IrOperation::Field {
                    expr: Box::new(IrOperation::Var {
                        name: "self".to_string(), id: "self".to_string(),
                        ty: IrType::Named { name: "Lexer".to_string(), generics: [] },
                        evidence: IrEvidence::Known,
                    }),
                    field: "pos".to_string(),
                    ty: IrType::Int { width: 64, signed: false },
                    evidence: IrEvidence::Known,
                }),
                right: Box::new(IrOperation::Literal {
                    variant: LiteralVariant::Int, value: LiteralValue::Int(1),
                    ty: IrType::Int { width: 64, signed: false }, evidence: IrEvidence::Known,
                }),
                ty: IrType::Int { width: 64, signed: false },
                evidence: IrEvidence::Known,
            }),
            evidence: IrEvidence::Known,
        },
        is_async: false,
        is_unsafe: false,
        span: null,
    });

    // Add test function that calls advance
    module.functions.push(IrFunction {
        name: "test_advance".to_string(),
        id: "test_advance".to_string(),
        visibility: IrVisibility::Public,
        generics: [],
        params: [],
        return_type: IrType::Int { width: 64, signed: false },
        return_evidence: IrEvidence::Known,
        receiver_kind: null,
        body: ?IrOperation::Block {
            statements: [
                // let mut lexer = Lexer { source: "abc", pos: 0 };
                IrOperation::Let {
                    pattern: IrPattern::Ident { name: "lexer".to_string(), mutable: true, evidence: null },
                    type_annotation: null,
                    init: Box::new(IrOperation::StructInit {
                        name: "Lexer".to_string(),
                        fields: [
                            ("source".to_string(), IrOperation::Literal {
                                variant: LiteralVariant::String, value: LiteralValue::String("abc".to_string()),
                                ty: IrType::Str, evidence: IrEvidence::Known,
                            }),
                            ("pos".to_string(), IrOperation::Literal {
                                variant: LiteralVariant::Int, value: LiteralValue::Int(0),
                                ty: IrType::Int { width: 64, signed: false }, evidence: IrEvidence::Known,
                            }),
                        ],
                        ty: IrType::Named { name: "Lexer".to_string(), generics: [] },
                        evidence: IrEvidence::Known,
                    }),
                    evidence: IrEvidence::Known,
                },
                // lexer.advance()
                IrOperation::MethodCall {
                    receiver: Box::new(IrOperation::Var {
                        name: "lexer".to_string(), id: "lexer".to_string(),
                        ty: IrType::Named { name: "Lexer".to_string(), generics: [] },
                        evidence: IrEvidence::Known,
                    }),
                    method: "advance".to_string(),
                    args: [],
                    type_args: [],
                    ty: IrType::Unit,
                    evidence: IrEvidence::Known,
                    receiver_mutable: true,
                },
                // return lexer.pos  -- should be 1, not 0!
                IrOperation::Return {
                    value: ?Box::new(IrOperation::Field {
                        expr: Box::new(IrOperation::Var {
                            name: "lexer".to_string(), id: "lexer".to_string(),
                            ty: IrType::Named { name: "Lexer".to_string(), generics: [] },
                            evidence: IrEvidence::Known,
                        }),
                        field: "pos".to_string(),
                        ty: IrType::Int { width: 64, signed: false },
                        evidence: IrEvidence::Known,
                    }),
                    evidence: IrEvidence::Known,
                },
            ],
            ty: IrType::Int { width: 64, signed: false },
            evidence: IrEvidence::Known,
        },
        is_async: false,
        is_unsafe: false,
        span: null,
    });

    let code = generate(module);

    // Signature check
    assert_contains(&code, "sigil_Lexer____advance(SigilValue* self)",
        "CG-015.5a: advance should take pointer");

    // Call site check
    assert_contains(&code, "sigil_Lexer____advance(&lexer)",
        "CG-015.5b: should pass &lexer");

    // No value passing
    assert_not_contains(&code, "sigil_Lexer____advance(lexer)",
        "CG-015.5c: should NOT pass lexer by value");

    println("PASS");
    true
}

// ============================================================================
// Main Test Runner
// ============================================================================

pub fn main() -> !i32 {
    println("Testing CG-015: Mutable Self Pass-by-Value...");
    println("");

    let mut passed: !i32 = 0;
    let mut failed: !i32 = 0;

    if test_cg015_mut_self_signature() { passed += 1; } else { failed += 1; }
    if test_cg015_mut_self_call_site() { passed += 1; } else { failed += 1; }
    if test_cg015_immut_self_by_value() { passed += 1; } else { failed += 1; }
    if test_cg015_field_mutation_via_pointer() { passed += 1; } else { failed += 1; }
    if test_cg015_lexer_advance_integration() { passed += 1; } else { failed += 1; }

    println("");
    println("Results: {}/{} tests passed", passed, passed + failed);

    if failed > 0 {
        println("FAILED: {} tests failed", failed);
        return 1;
    }

    println("");
    println("All CG-015 tests passed!");
    return 0;
}
```

---

## 5. Implementation Plan

### Phase 1: IR Extension (Required Changes)

**File: `ir.sg`**

1. Add `ReceiverKind` enum
2. Add `receiver_kind: ?ReceiverKind` to `IrFunction`
3. Add `receiver_mutable: !bool` to `IrMethodCall` (or infer from receiver_kind lookup)

### Phase 2: Parser Enhancement

**File: `parser.sg`**

1. Track `mut` keyword in `parse_self_param()`
2. Propagate to `Param.mutable` field

### Phase 3: Lowering Updates

**File: `lower.sg`**

1. Detect `mut self` in method parameters
2. Set `receiver_kind` on `IrFunction`
3. When lowering method calls, look up target method's `receiver_kind`

### Phase 4: Codegen Fix

**File: `codegen.sg`**

1. **emit_function_signature()**: Generate `SigilValue* self` for mutable methods
2. **emit_method_call()**: Prepend `&` for mutable receiver
3. **emit_field_access()**: Use `*self` when self is pointer
4. **emit_field_set()**: Pass `self` directly (already a pointer)

### Phase 5: Remove Python Hacks

**File: `build.sh`**

1. Remove `fix_codegen.py` invocation
2. Remove any sed/awk workarounds for method calls
3. Verify clean build

---

## 6. Verification Checklist

### Unit Tests
- [ ] `test_cg015_mut_self_signature` passes
- [ ] `test_cg015_mut_self_call_site` passes
- [ ] `test_cg015_immut_self_by_value` passes
- [ ] `test_cg015_field_mutation_via_pointer` passes
- [ ] `test_cg015_lexer_advance_integration` passes

### Integration Tests
- [ ] `span.sg` compiles and runs correctly
- [ ] `lexer.sg` compiles and doesn't hang
- [ ] `parser.sg` compiles and parses correctly
- [ ] Full bootstrap (`./build.sh`) succeeds
- [ ] Native binary doesn't hang on simple input

### Regression Tests
- [ ] All existing `test_codegen.sg` tests pass
- [ ] All CG-001 through CG-014 tests pass
- [ ] No new Python/sed/awk hacks needed

### Fixed-Point Verification
- [ ] `./build/sigil compile src/*.sg -o verify.c` produces valid C
- [ ] `diff build/c/*.c verify/*.c` shows expected differences only

---

## 7. Risk Analysis

### Low Risk
- IR changes are additive (new optional fields)
- Parser changes are localized to `parse_self_param`
- Codegen changes are well-isolated

### Medium Risk
- 198 method call sites need correct `receiver_mutable` detection
- Field access through pointer requires careful `*self` handling

### High Risk
- **Nested mutations**: `self.lexer.advance()` needs correct pointer chaining
- **Return value**: Methods returning `&mut Self` need special handling

### Mitigation
- Comprehensive TDD test suite (5+ tests)
- Incremental implementation (lexer first, then parser, then codegen)
- Keep existing hacks until tests pass

---

## 8. Appendix: Mutation Inventory

### Lexer (43 methods)
Critical: `advance`, `next_token`, `peek`, `match_char`

### Parser (70 methods)
Critical: `advance`, `expect`, `consume_if`, all `parse_*`

### CodeGen (25 methods)
Critical: `line`, `emit`, `fresh_temp`, `indent_push/pop`

### TypeChecker (24 methods)
Critical: `push_scope`, `pop_scope`, `error`, `define`

---

## 9. Implementation Progress (2025-12-14)

### Completed

1. **IR Extension** (`ir.sg`):
   - Added `mutable: !bool` field to `IrParam` struct

2. **Lowering Updates** (`lower.sg`):
   - Updated `lower_param()` to extract mutable flag from pattern
   - Pattern matching for `Pattern::Ident { mutable: m, .. }` to propagate mutability

3. **Codegen Enhancements** (`codegen.sg`):
   - Added `current_fn_has_mut_self: !bool` tracking to CodeGen struct
   - Added `has_mutable_self()` helper function
   - Added `is_mutable_method()` with hybrid pattern-based detection:
     - Non-mutable prefixes: `has_`, `get_`, `can_`, `from_`, `to_`, `as_`, `into_`
     - Non-mutable exact matches: `new`, `clone`, `len`, `iter`, etc.
     - Mutable prefixes: `lex_`, `parse_`, `emit_`, `skip_`, `match_`, `infer_`, `collect_`, etc.
     - Mutable exact matches: `advance`, `next_token`, `peek`, `unify`, `error`, etc.
   - Added `is_user_type()` to distinguish user types from library types:
     - User types: Lexer, Parser, TypeChecker, CodeGen, Interpreter, etc.
     - Handles `Uncertain*` prefix stripping
   - Updated function signature generation: `SigilValue* self` for mutable methods
   - Updated call site handling:
     - Pass `self` pointer when receiver is `self` in mutable context
     - Pass `&self` when receiver is `self` not in mutable context
     - Pass `&var` for local variables when calling mutable methods on user types
     - Handle expression receivers by emitting to temp and passing `&temp`
   - Updated `IrOperation::Var` handler: dereference `self` as `(*self)` in mutable context

4. **Build Script** (`build.sh`):
   - Updated forward declarations to use `SigilValue* self` for mutable methods

### Remaining Issues (141 Errors)

The heuristic approach has fundamental limitations:

1. **No type information at call sites**: Without knowing the receiver type, we can't determine method signature
2. **Same method name, different mutability**: e.g., `Parser::check(self)` vs `Driver::check(mut self)`
3. **Closure captures**: Closures capturing `self` don't propagate pointer semantics
4. **Expression receivers**: Complex expressions like `self.lexer.advance()` need proper handling

### Recommended Next Steps

1. **Type propagation**: Extend IR to track receiver types at call sites
2. **Signature registry**: Build a map of function signatures during lowering
3. **Call-site lookup**: Look up target method signature when emitting calls
4. **Closure fix**: Handle `self` capture in closures with proper pointer types

---

## References

- Commit `81bd3b592` - CG-015 identification
- `/home/user/workspace/sigil/sigil-lang/self-hosted/TDD_GUIDE.md` - TDD workflow
- `/home/user/workspace/sigil/sigil-lang/self-hosted/JORMUNGANDR_PROGRESS.md` - Bootstrap status
