# Sigil Interpreter Specification & TDD Roadmap

**Version:** 1.13.0
**Date:** 2026-01-27
**Status:** Waves 1-8 Complete + Deferred Items Resolved, 100% Pass Rate (728 tests)
**Component:** `parser/src/interpreter.rs`

---

## Executive Summary

This document defines the specification and TDD roadmap for the Sigil interpreter, based on comprehensive test analysis. The interpreter currently passes **728/728 tests (100%)**, achieving full test coverage.

### Current State (Updated 2026-01-27)

| Metric | Value | Notes |
|--------|-------|-------|
| Total Tests | 728 | Across 24 spec categories |
| Passing | 728 | 100% pass rate |
| Failing | 0 | All tests passing |
| Jormungandr | Working | Self-hosted compiler functional |
| Deferred Items | 0 | All P1 items complete |

### Wave Completion Status

| Wave | Focus | Status | Tests Fixed |
|------|-------|--------|-------------|
| **Wave 1** | P0 Type Validation & Traits | ✅ Complete | All P0 passing |
| **Wave 2** | P1 Memory Features | ✅ Complete | +4 tests |
| **Wave 3** | P1 Stdlib Completion | ✅ Complete | +3 tests |
| **Wave 4** | P1-BOOTSTRAP Native Runtime | ✅ Complete | 12 modules |
| **Wave 5** | Module Resolution | ✅ Complete | `invoke tome·` + 22 intrinsics |
| **Wave 6** | Native Syntax Migration | ✅ Complete | 157 files migrated |
| **Wave 7** | SGDOC & Tooling | ✅ Complete | Doc extraction, LSP |

---

## Priority Classification

### P0: Critical (Blocks Production Use) ✅ COMPLETE

Features required for real-world applications and compiler bootstrap.

| ID | Feature | Tests | Status | Impact |
|----|---------|-------|--------|--------|
| P0-TYPE-001 | Generic type mismatch validation | 1 | ✅ Done | Silent type errors |
| P0-TYPE-002 | Option type mismatch validation | 1 | ✅ Done | Silent type errors |
| P0-TYPE-003 | Result type mismatch validation | 1 | ✅ Done | Silent type errors |
| P0-TYPE-004 | Match arm type validation | 1 | ✅ Done | Silent type errors |
| P0-TYPE-005 | Negative array size validation | 1 | ✅ Done | Runtime crashes |
| P0-TRAIT-001 | Trait bounds (`T: Trait`) | 1 | ✅ Done | Compile-time checking |
| P0-TRAIT-002 | Where clauses | 1 | ✅ Done | Generic constraints |
| P0-METHOD-001 | Method chaining edge cases | 1 | ✅ Done | Ergonomics |

**Total P0 Gaps: 0 tests** (All P0 tests passing)

### P1: High (Quality of Life) ✅ MOSTLY COMPLETE

Features that improve developer experience and code safety.

| ID | Feature | Tests | Status | Impact |
|----|---------|-------|--------|--------|
| P1-MEM-001 | Reborrow semantics | 1 | ✅ Done | Advanced borrowing |
| P1-MEM-002 | Box<T> deref | 1 | ✅ Done | Heap allocation |
| P1-MEM-003 | Slice borrowing | 1 | ✅ Done | View into arrays |
| P1-MEM-004 | Lifetime elision | 1 | ✅ Done | Ergonomic lifetimes |
| P1-COERCE-001 | Type coercion | 1 | ✅ Done | Implicit conversions |
| P1-PTR-001 | Nullable pointer | 1 | ✅ Done | FFI compatibility, is_null() |
| P1-MATH-001 | exp/log functions | 1 | ✅ Done | Math stdlib |
| P1-VEC-001 | Vec::clear() | 1 | ✅ Done | Collection ops |
| P1-STATIC-001 | Static variables | 1 | ✅ Done | Global state |

**Total P1 Gaps: 0 tests** (9/9 complete)

### P1-BOOTSTRAP: Native Runtime (Self-Hosting) ✅ COMPLETE

Replace the C runtime with pure Sigil to achieve full self-hosting.

**Full Specification:** [NATIVE-RUNTIME-SPEC.md](./NATIVE-RUNTIME-SPEC.md)

**Current C Runtime:** `parser/runtime/sigil_runtime.c` (741 lines, 76 functions)

| Phase | Focus | Modules | Status |
|-------|-------|---------|--------|
| **A** | Platform Syscalls | `rt/sys/` (4 files, 111+ syscalls) | ✅ Complete |
| **B** | Memory Allocator | `rt/alloc/` (arena, global) | ✅ Complete |
| **C** | Core Types | `rt/types/` (Vec, String) | ✅ Complete |
| **D** | I/O | `rt/io/` (print, File) | ✅ Complete |
| **E** | Math | `rt/math/` (36 LLVM intrinsics) | ✅ Complete |
| **F** | Integration | `P1_030_rt_integration.sg` | ✅ Complete |

**Native Runtime Modules (12 files, all parsing):**
```
parser/src/rt/
├── mod.sg              # Root module
├── sys/
│   ├── mod.sg          # Platform dispatch
│   ├── linux_x64.sg    # Linux syscalls (111 items)
│   ├── darwin_x64.sg   # macOS Intel
│   └── darwin_arm64.sg # macOS ARM
├── alloc/
│   ├── mod.sg          # Allocator exports
│   └── arena.sg        # Arena allocator
├── types/
│   ├── mod.sg          # Type exports
│   ├── vec.sg          # Vec<T>
│   └── string.sg       # String (UTF-8)
├── io/
│   └── mod.sg          # I/O operations
├── math/
│   └── mod.sg          # Math functions
└── time/
    └── mod.sg          # Time operations
```

**Next Step:** Wave 5 - Module Resolution to enable `invoke tome·` linking

### Wave 5: Module Resolution ✅ COMPLETE

Enable the compiler to resolve `invoke tome·` statements and link modules together.

| Task | Description | Status |
|------|-------------|--------|
| Parse `invoke`/`tome` | Lexer/parser support | ✅ Done |
| Module path resolution | Find `.sg` files from paths | ✅ Done |
| Circular dependency detection | Prevent infinite loops | ✅ Done |
| Symbol export/import | Track pub symbols across modules | ✅ Done |
| LLVM intrinsics in interpreter | 22 math intrinsics added | ✅ Done |
| Integration test | Native math working via tome | ✅ Done |

**Implementation Location:** `parser/src/interpreter.rs` - `load_tome_module()` function

**Module Resolution Algorithm:**
```
invoke tome·rt·sys·{write, Errno}
           ↓
    module_path = ["rt", "sys"]
           ↓
    Try: src/rt/sys/mod.sg
         src/rt/sys/mod.sigil
         src/rt/sys.sg
         src/rt/sys.sigil
           ↓
    Parse and execute module items
           ↓
    Register symbols with qualified names
```

**Architecture:**
```
┌─────────────────────────────────────────┐
│           Sigil User Program            │
├─────────────────────────────────────────┤
│     Native Runtime (Pure Sigil)         │
│  Vec │ String │ Option │ Result │ File  │
├─────────────────────────────────────────┤
│           Memory Allocator              │
├─────────────────────────────────────────┤
│    Platform Syscalls (arch-specific)    │
├─────────────────────────────────────────┤
│     LLVM Intrinsics (math, atomics)     │
└─────────────────────────────────────────┘
```

**Goal:** Zero C dependencies for AOT-compiled binaries.

---

### P2: Low (Future/Experimental)

Features planned for future releases or experimental modules.

| ID | Feature | Tests | Status | Notes |
|----|---------|-------|--------|-------|
| P2-QUANTUM | Quantum computing | 10 | Experimental | Qubit, gates |
| P2-QH | Quantum-holographic | 10 | Experimental | QH primitives |
| P2-NEURAL | Neural/tensor ops | 10 | Experimental | ML primitives |
| P2-PROB | Probabilistic types | 10 | Experimental | Superposition |
| P2-PROTO | Protocol clients | 10 | Experimental | HTTP, gRPC, Kafka |
| P2-REFLECT | Reflection/IR | 10 | Experimental | Metaprogramming |
| P2-LOOP | Infinite loop (forever) | 1 | ✅ Done | `forever` keyword |
| P2-FOR | For-step syntax | 1 | ✅ Done | `each i of 1..10` |

**Total P2 Gaps: 60 tests** (experimental features only)

---

## SDD (Spec-Driven Development) Lifecycle

Each feature follows the complete SDD lifecycle:

```
┌─────────────────────────────────────────────────────────────────┐
│                    SDD Lifecycle Phases                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  1. SPEC          Write formal specification                     │
│      │            - Define behavior                              │
│      │            - Define error conditions                      │
│      │            - Define edge cases                            │
│      ▼                                                           │
│  2. TEST          Write failing tests (RED)                      │
│      │            - Positive tests (expected behavior)           │
│      │            - Negative tests (expected errors)             │
│      │            - Edge case tests                              │
│      ▼                                                           │
│  3. IMPLEMENT     Implement minimum to pass (GREEN)              │
│      │            - Focus on correctness                         │
│      │            - No premature optimization                    │
│      ▼                                                           │
│  4. REFACTOR      Clean up while green                           │
│      │            - Improve code quality                         │
│      │            - Maintain test coverage                       │
│      ▼                                                           │
│  5. DOCUMENT      Update documentation                           │
│      │            - Update CLAUDE.md                             │
│      │            - Update spec docs                             │
│      ▼                                                           │
│  6. INTEGRATE     Verify no regressions                          │
│                   - Run full test suite                          │
│                   - Verify Jormungandr still works               │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Test File Naming Convention

```
jormungandr/tests/spec/{category}/P{priority}_{number}_{feature}.sg
jormungandr/tests/spec/{category}/P{priority}_{number}_{feature}.expected
jormungandr/tests/spec/{category}/P{priority}_{number}_{feature}.error_expected
```

---

## Phase 1: Type Validation (P0-TYPE)

**Goal:** Proper type checking for generics, Option, Result, and match arms.

### Spec: P0-TYPE-001 Generic Type Mismatch

**Behavior:** When a generic type parameter is instantiated with incompatible types, emit an error.

```sigil
// MUST ERROR: Cannot assign String to Vec<i32>
fn main() {
    let v: Vec<i32> = Vec::new();
    v.push("string");  // ERROR: expected i32, found String
}
```

**Error Message:**
```
error: type mismatch in generic instantiation
  --> file.sg:4:5
   |
4  |     v.push("string");
   |            ^^^^^^^^ expected i32, found String
   |
   = note: Vec<i32> requires i32 elements
```

### Spec: P0-TYPE-002 Option Type Mismatch

**Behavior:** Option<T> must only contain values of type T.

```sigil
// MUST ERROR: Cannot put String in Option<i32>
fn main() {
    let opt: Option<i32> = Some("hello");  // ERROR
}
```

### Spec: P0-TYPE-003 Result Type Mismatch

**Behavior:** Result<T, E> must match both type parameters.

```sigil
// MUST ERROR: Wrong Ok type
fn main() {
    let r: Result<i32, String> = Ok("wrong");  // ERROR: expected i32
}
```

### Spec: P0-TYPE-004 Match Arm Type Consistency

**Behavior:** All match arms must return the same type.

```sigil
// MUST ERROR: Inconsistent arm types
fn main() {
    let x = match true {
        true => 42,       // i32
        false => "no",    // ERROR: expected i32, found String
    };
}
```

### Spec: P0-TYPE-005 Negative Array Size

**Behavior:** Array sizes must be non-negative.

```sigil
// MUST ERROR: Negative array size
fn main() {
    let arr: [i32; -5] = [];  // ERROR: array size cannot be negative
}
```

### Implementation Plan

1. **Location:** `parser/src/interpreter.rs` - `eval_call`, `eval_struct_literal`
2. **Add:** Type parameter tracking in generic instantiation
3. **Add:** Type validation in Option/Result constructors
4. **Add:** Match arm type unification

### Tests to Create

```bash
# Create test files
tests/spec/19_negative/P0_018_generic_type_mismatch.sg
tests/spec/19_negative/P0_019_option_type_mismatch.sg
tests/spec/19_negative/P0_020_result_type_mismatch.sg
tests/spec/19_negative/P0_027_invalid_match_arm_type.sg
tests/spec/19_negative/P0_028_negative_array_size.sg
```

---

## Phase 2: Trait System (P0-TRAIT)

**Goal:** Support trait bounds and where clauses in generic functions.

### Spec: P0-TRAIT-001 Trait Bounds

**Behavior:** Generic functions can constrain type parameters with trait bounds.

```sigil
trait Display {
    fn display(&self) -> String;
}

// T must implement Display
fn print_it<T: Display>(item: T) {
    println(item.display());
}
```

### Spec: P0-TRAIT-002 Where Clauses

**Behavior:** Complex trait bounds can use where clauses.

```sigil
fn process<T, U>(a: T, b: U) -> String
where
    T: Display + Clone,
    U: Debug,
{
    // ...
}
```

### Implementation Plan

1. **Location:** `parser/src/interpreter.rs` - `eval_call`, function resolution
2. **Add:** Trait bound checking before generic instantiation
3. **Add:** Where clause parsing and validation

---

## Phase 3: Memory Features (P1-MEM)

**Goal:** Advanced borrowing and lifetime features.

### Spec: P1-MEM-001 Reborrow

**Behavior:** Mutable references can be reborrowed.

```sigil
fn main() {
    let mut x = 42;
    let r1 = &mut x;
    let r2 = &mut *r1;  // Reborrow
    *r2 = 100;
    assert_eq!(x, 100);
}
```

### Spec: P1-MEM-002 Box Deref

**Behavior:** Box<T> can be dereferenced to access inner value.

```sigil
fn main() {
    let b = Box::new(42);
    assert_eq!(*b, 42);
}
```

### Spec: P1-MEM-003 Slice Borrowing

**Behavior:** Arrays can be borrowed as slices.

```sigil
fn main() {
    let arr = [1, 2, 3, 4, 5];
    let slice: &[i32] = &arr[1..4];
    assert_eq!(slice.len(), 3);
}
```

---

## Phase 4: Stdlib Completion (P1)

**Goal:** Complete standard library methods.

| Method | Type | Status |
|--------|------|--------|
| `exp()` | f64 | Missing |
| `log()` | f64 | Missing |
| `clear()` | Vec<T> | Missing |
| `static` vars | Global | Missing |

---

## TDD Roadmap: Wave 1

### Sprint 1: Type Validation (Week 1)

**Tests First (RED):**
```bash
# Day 1: Create failing tests
./run_tests_rust.sh --spec 19_negative  # Verify 5 tests fail

# Day 2-3: Implement type checking
# Edit: parser/src/interpreter.rs

# Day 4: Verify all pass (GREEN)
./run_tests_rust.sh --spec 19_negative  # All 30 should pass
```

**Deliverables:**
- [ ] P0_018_generic_type_mismatch.sg passes
- [ ] P0_019_option_type_mismatch.sg passes
- [ ] P0_020_result_type_mismatch.sg passes
- [ ] P0_027_invalid_match_arm_type.sg passes
- [ ] P0_028_negative_array_size.sg passes

### Sprint 2: Trait Bounds (Week 2)

**Tests First (RED):**
```bash
# Create trait bound tests
tests/spec/03_types/P0_052_trait_bound.sg
tests/spec/03_types/P0_061_where_clause.sg
```

**Deliverables:**
- [ ] Trait bound syntax parsed
- [ ] Trait bound validation at call sites
- [ ] Where clause support

### Sprint 3: Memory Features (Week 3)

**Tests First (RED):**
```bash
tests/spec/04_memory/P1_001_reborrow.sg
tests/spec/04_memory/P1_007_box_deref.sg
tests/spec/04_memory/P1_010_borrow_slice.sg
```

**Deliverables:**
- [ ] Reborrow semantics
- [ ] Box<T> deref
- [ ] Slice borrowing

---

## Success Metrics

### Final Status (Waves 1-7 Complete)

| Metric | Starting | Final | Delta |
|--------|----------|-------|-------|
| Test Pass Rate | 84% (451/531) | 100% (577/577) | +126 ✅ |
| P0 Gaps | 8 | 0 | -8 ✅ |
| P1 Gaps | 9 | 0 | -9 ✅ |
| P2 Tests | 65 | 0 | All passing ✅ |
| Native Syntax | 0% | 100% | 157 files migrated ✅ |

### v0.4.0 Complete

All tests passing. No remaining gaps. Ready for release.

### Verification Checklist

After each sprint:
- [ ] Run full test suite: `./run_tests_rust.sh`
- [ ] Verify Jormungandr still works: `./sigil run-dir jormungandr/src`
- [ ] No regressions in feature_interaction tests
- [ ] Update CLAUDE.md test counts

---

## File Reference

### Primary Implementation Files

| File | Purpose | Size |
|------|---------|------|
| `parser/src/interpreter.rs` | Runtime interpreter | 452KB |
| `parser/src/typeck.rs` | Type checker | 122KB |
| `parser/src/stdlib.rs` | Standard library | 1.2MB |

### Test Directories

```
jormungandr/tests/spec/
├── 01_lexical/         # Tokenization
├── 02_syntax/          # Parsing
├── 03_types/           # Type system (P0-TRAIT here)
├── 04_memory/          # Memory/lifetime (P1-MEM here)
├── 19_negative/        # Error validation (P0-TYPE here)
├── 20_edge_cases/      # Corner cases
└── 21_feature_interaction/  # Integration tests
```

---

## Appendix: Test Commands

```bash
# Run all tests
cd jormungandr/tests && ./run_tests_rust.sh

# Run specific category
./run_tests_rust.sh --spec 19_negative
./run_tests_rust.sh --spec 03_types
./run_tests_rust.sh --spec 04_memory

# Run by priority
./run_tests_rust.sh --priority P0
./run_tests_rust.sh --priority P1

# Run single test
../../parser/target/release/sigil run tests/spec/19_negative/P0_018_generic_type_mismatch.sg
```

---

---

## Wave 8: Metaprogramming (Runes)

**Status:** ✅ Complete (Core Features)
**Spec Reference:** [07-METAPROGRAMMING.md](./07-METAPROGRAMMING.md)

### P0-RUNE: Pipe-Invoked Runes (§2.6)

| ID | Feature | Test File | Status | Notes |
|----|---------|-----------|--------|-------|
| P0-RUNE-001 | Basic pipe invocation | `P0_001_pipe_invoked_basic.sg` | ✅ Done | `value\|rune!{}` with `__pipe` |
| P0-RUNE-002 | Pipe with validation | `P0_002_pipe_invoked_validate.sg` | ✅ Done | Return semantics, Result types |
| P0-RUNE-003 | Chained pipe invocations | `P0_003_pipe_chained.sg` | ✅ Done | Multiple `\|` in sequence |
| P0-RUNE-004 | Pipe with arguments | `P0_004_pipe_with_args.sg` | ✅ Done | `value\|rune!{arg}` combined |
| P0-RUNE-005 | Simple expressions | `P0_005_simple_expr.sg` | ✅ Done | Expression macros |

### P0-FRAG: Fragment Types (§2.2)

| ID | Feature | Test File | Status | Notes |
|----|---------|-----------|--------|-------|
| P0-FRAG-001 | expr fragment | `P0_010_frag_expr.sg` | ✅ Done | Expression matching |
| P0-FRAG-002 | ident fragment | `P0_011_frag_ident.sg` | ✅ Done | Identifier matching |
| P0-FRAG-003 | ty fragment | `P0_012_frag_ty.sg` | ✅ Done | Type matching (string capture) |
| P0-FRAG-004 | literal fragment | `P0_013_frag_literal.sg` | ✅ Done | Literal matching |
| P0-FRAG-005 | block fragment | `P0_014_frag_block.sg` | ✅ Done | Block matching (executed) |
| P0-FRAG-006 | stmt fragment | `P0_015_frag_stmt.sg` | ✅ Done | Statement matching (string capture) |
| P0-FRAG-007 | pat fragment | `P0_016_frag_pat.sg` | ✅ Done | Pattern matching (string capture) |
| P0-FRAG-008 | tt fragment | `P0_017_frag_tt.sg` | ✅ Done | Token tree matching (string capture) |

### P1-RUNE: Edge Cases (§13.3)

| ID | Feature | Test File | Status | Notes |
|----|---------|-----------|--------|-------|
| P1-RUNE-001 | Empty arguments | `P1_020_edge_empty_args.sg` | ✅ Done | `rune!()` vs `rune!{}` |
| P1-RUNE-002 | Nested pipes | `P0_003_pipe_chained.sg` | ✅ Done | Deep chaining tested in P0_003 |
| P1-RUNE-003 | Named parameters | `P0_002_pipe_invoked_validate.sg` | ✅ Done | Tested in P0_002 |
| P1-RUNE-004 | Multiple parameters | `P0_004_pipe_with_args.sg` | ✅ Done | Tested in P0_004 |

### P0-FRAG: Extended Fragment Types (§2.2.1)

| ID | Feature | Test File | Status | Notes |
|----|---------|-----------|--------|-------|
| P0-FRAG-009 | path fragment | `P0_023_frag_path.sg` | ✅ Done | Module path as string |
| P0-FRAG-010 | lifetime fragment | `P0_024_frag_lifetime.sg` | ✅ Done | Lifetime as string (`'static`) |
| P0-FRAG-011 | vis fragment | `P0_025_frag_vis.sg` | ✅ Done | Visibility as string (`pub`) |

### P0-FRAG: Final Fragment Types

| ID | Feature | Test File | Status | Notes |
|----|---------|-----------|--------|-------|
| P0-FRAG-012 | item fragment | `P0_018_frag_item.sg` | ✅ Done | Item matching (string capture) |
| P0-FRAG-013 | meta fragment | `P0_019_frag_meta.sg` | ✅ Done | Attribute content (string capture) |

### P0-REP: Repetition Patterns (§2.3)

| ID | Feature | Test File | Status | Notes |
|----|---------|-----------|--------|-------|
| P0-REP-001 | Zero-or-more | `P0_020_repetition_star.sg` | ✅ Done | `$($var:type),*` repetition |
| P0-REP-002 | One-or-more | `P0_021_repetition_plus.sg` | ✅ Done | `$($var:type),+` repetition |
| P0-REP-003 | Optional | `P0_022_repetition_optional.sg` | ✅ Done | `$($var:type)?` repetition |
| P0-REP-004 | Complex body sum | `P0_026_complex_body_sum.sg` | ✅ Done | `0 $(+ $x)+` expansion |
| P0-REP-005 | Complex body product | `P0_027_complex_body_product.sg` | ✅ Done | `1 $(* $x)+` expansion |

### Test Directory Structure

```
jormungandr/tests/spec/07_metaprogramming/
├── P0_001_pipe_invoked_basic.sg        ✅ Existing
├── P0_001_pipe_invoked_basic.expected  ✅ Existing
├── P0_002_pipe_invoked_validate.sg     ✅ Existing
├── P0_002_pipe_invoked_validate.expected ✅ Existing
├── P0_003_pipe_chained.sg              ❌ Create
├── P0_004_pipe_with_args.sg            ❌ Create
├── P0_005_return_scoping.sg            ❌ Create
├── P0_010_frag_expr.sg                 ❌ Create
├── P0_011_frag_ident.sg                ❌ Create
├── ...
```

---

## Revision History

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0.0 | 2026-01-21 | Claude Code | Initial spec with priority classification |
| 1.1.0 | 2026-01-21 | Claude Code | Wave 1 complete: P0 type validation, traits, method chaining |
| 1.2.0 | 2026-01-21 | Claude Code | Wave 2 complete: Memory features (reborrow, Box deref, slice coercion) |
| 1.3.0 | 2026-01-21 | Claude Code | Wave 3 complete: Stdlib (math functions, Vec::clear, static vars) |
| 1.4.0 | 2026-01-21 | Claude Code | Added P1-BOOTSTRAP: Native Runtime roadmap (shed C runtime) |
| 1.5.0 | 2026-01-21 | Claude Code | Wave 4 complete: Native Runtime (12 modules, all phases A-F) |
| 1.6.0 | 2026-01-21 | Claude Code | Wave 5 complete: Module resolution + 22 LLVM intrinsics |
| 1.7.0 | 2026-01-25 | Claude Code | Wave 6-7 complete: Native syntax migration, SGDOC tooling, 100% tests |
| 1.8.0 | 2026-01-26 | Claude Code | Wave 8: Added Metaprogramming (Runes) test requirements per §2.6, §2.2, §13 |
| 1.9.0 | 2026-01-26 | Claude Code | Wave 8 core complete: Implemented repetition patterns ($(...)*,+,?) per §2.3 |
| 1.10.0 | 2026-01-26 | Claude Code | Extended fragments (path, lifetime, vis) + complex body expansion (sum/product) |
| 1.10.1 | 2026-01-26 | Claude Code | Enabled network tests (HTTP, WebSocket) - 721 tests, websocket default feature |
| 1.11.0 | 2026-01-26 | Claude Code | All P0 fragment types complete: ty, block, stmt, pat, tt (726 tests) |
| 1.12.0 | 2026-01-26 | Claude Code | Final fragment types: item, meta - ALL 13 fragment types complete (728 tests) |
