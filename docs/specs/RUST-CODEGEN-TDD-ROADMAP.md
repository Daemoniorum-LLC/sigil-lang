# Rust Codegen TDD Roadmap

**Version:** 1.2.0
**Status:** ✅ COMPLETE
**Completed:** 2026-02-06
**Spec:** [RUST-CODEGEN-PARSER-GAPS-SPEC.md](./RUST-CODEGEN-PARSER-GAPS-SPEC.md)
**Methodology:** [Agent-TDD](../../../dev2/workspace/docs/methodologies/AGENT-TDD.md)

---

## Overview

This roadmap derives test cases from the S++ specs in RUST-CODEGEN-PARSER-GAPS-SPEC.md. Each gap's edge cases become specification tests. Implementation follows the tests.

**Philosophy:** Tests are crystallized understanding. The spec tells us WHAT; the tests verify we got it right; the implementation is creative freedom within those constraints.

---

## Implementation Priority

| Phase | Gap | Priority | Scope | Status |
|-------|-----|----------|-------|--------|
| 1 | **L: Return statement** | P0 | Codegen only | ✅ Complete |
| 2 | **B: Generic params** | P0 | Parser + Codegen | ✅ Complete |
| 3 | **H: Path resolution** | P0 | Codegen | ✅ Complete |
| 4 | **J: Impl where clauses** | P1 | Parser + Codegen | ✅ Complete |
| 5 | **K: impl Trait** | P1 | Parser + Codegen | ✅ Complete |
| 6 | **I: Raw pointers** | P1 | Codegen | ✅ Option A Complete |
| 7 | **M: Const generic defaults** | P2 | Parser + Codegen | ✅ Complete |

**All phases complete!** Gap I Option B (typechecker-informed) deferred for future architectural work.

---

## Phase 1: Gap L - Return Statement ✅ COMPLETE

**Discovery:** Already implemented! The `⤺` symbol correctly emits `return`.

### 1.1 Test File

`jormungandr/tests/rust_codegen/test_return.sg` - Created and verified.

### 1.2 Specification Tests (from EC₁-EC₁₅)

```
test_return_with_value
  Input:  rite foo() → i32 { ⤺ 42 }
  Expect: fn foo() -> i32 { return 42 }

test_early_return_in_conditional
  Input:  rite foo(x: i32) → i32 { ⎇ x < 0 { ⤺ 0 } x }
  Expect: fn foo(x: i32) -> i32 { if x < 0 { return 0 } x }

test_return_none
  Input:  rite foo() → Option<i32> { ⎇ cond { ⤺ None } Some(42) }
  Expect: fn foo() -> Option<i32> { if cond { return None } Some(42) }

test_return_err
  Input:  rite foo() → Result<i32, Error> { ⎇ cond { ⤺ Err(e) } Ok(42) }
  Expect: fn foo() -> Result<i32, Error> { if cond { return Err(e) } Ok(42) }

test_bare_return_unit
  Input:  rite foo() { ⎇ cond { ⤺ } work() }
  Expect: fn foo() { if cond { return } work() }

test_return_complex_expression
  Input:  rite foo() → Vec<i32> { ⤺ vec.iter().map(f).collect() }
  Expect: fn foo() -> Vec<i32> { return vec.iter().map(f).collect() }

test_return_struct_literal
  Input:  rite foo() → Point { ⤺ Point { x: 1, y: 2 } }
  Expect: fn foo() -> Point { return Point { x: 1, y: 2 } }

test_return_tuple
  Input:  rite foo() → (i32, i32) { ⤺ (a, b) }
  Expect: fn foo() -> (i32, i32) { return (a, b) }

test_return_in_loop
  Input:  rite find(items: Vec<T>) → T { ∀ item ∈ items { ⎇ item.ok { ⤺ item } } panic!() }
  Expect: for item in items { if item.ok { return item } }

test_return_in_match_arm
  Input:  ⌥ x { Some(v) => ⤺ v, None => ⤺ 0 }
  Expect: match x { Some(v) => return v, None => return 0 }

test_return_in_closure
  Input:  ≔ f = |x| { ⤺ x * 2 };
  Expect: let f = |x| { return x * 2 };

test_nested_function_return
  Input:  rite outer() → i32 { rite inner() → i32 { ⤺ 1 } ⤺ inner() }
  Expect: fn outer() -> i32 { fn inner() -> i32 { return 1 } return inner() }

test_implicit_vs_explicit_return
  Input:  rite explicit() → i32 { ⤺ 42 }
          rite implicit() → i32 { 42 }
  Expect: fn explicit() -> i32 { return 42 }
          fn implicit() -> i32 { 42 }

test_multiple_early_returns
  Input:  rite classify(x: i32) → &str { ⎇ x < 0 { ⤺ "neg" } ⎇ x == 0 { ⤺ "zero" } "pos" }
  Expect: if x < 0 { return "neg" } if x == 0 { return "zero" } "pos"

test_return_with_try_operator
  Input:  rite foo() → Result<T, E> { ⤺ bar()? }
  Expect: fn foo() -> Result<T, E> { return bar()? }
```

### 1.3 Implementation Checklist

- [x] Verify `Stmt::Return` or `Expr::Return` exists in AST ✅ `Expr::Return`
- [x] Add `return` keyword emission in `emit_stmt` ✅ Already present
- [x] Add `return` keyword emission in `emit_expr` (if needed) ✅ Handled
- [x] Handle bare return (no expression) ✅ Works
- [x] Semicolons handled by Rust grammar (optional) ✅ Valid Rust
- [x] All 15 edge cases pass ✅ Verified 2026-02-06

### 1.4 Success Criteria

All 15 specification tests pass. Generated Rust compiles successfully.

---

## Phase 2: Gap B - Generic Default Params vs Associated Types ✅ COMPLETE

**Discovery:** Already implemented! Parser correctly distinguishes contexts.

### 2.1 Test File

`jormungandr/tests/rust_codegen/test_generics.sg` - Created and verified.

### 2.2 Specification Tests (from EC₁-EC₁₂)

```
test_simple_default_type
  Input:  sigil Foo<T = i32> { }
  Expect: struct Foo<T = i32> { }

test_multiple_defaults
  Input:  sigil HashMap<K, V, S = RandomState> { }
  Expect: struct HashMap<K, V, S = RandomState> { }

test_associated_type_binding
  Input:  rite foo<I: Iterator<Item = i32>>() { }
  Expect: fn foo<I: Iterator<Item = i32>>() { }

test_mixed_defaults_and_associated
  Input:  sigil Foo<T, I: Iterator<Item = T> = DefaultIter<T>> { }
  Expect: struct Foo<T, I: Iterator<Item = T> = DefaultIter<T>> { }

test_impl_with_associated_type
  Input:  ⊢ Iterator for Foo { type Item = i32; }
  Expect: impl Iterator for Foo { type Item = i32; }

test_turbofish_with_associated
  Input:  collect::<Vec<_>>()
  Expect: collect::<Vec<_>>()

test_nested_associated_types
  Input:  <T as Iterator>::Item
  Expect: <T as Iterator>::Item

test_multiple_associated_types
  Input:  rite foo<I: Iterator<Item = u8, IntoIter = std·vec·IntoIter<u8>>>()
  Expect: fn foo<I: Iterator<Item = u8, IntoIter = std::vec::IntoIter<u8>>>()

test_default_after_constraint
  Input:  sigil Foo<T: Clone = String> { }
  Expect: struct Foo<T: Clone = String> { }

test_const_generic_default
  Input:  sigil Array<T, const N: usize = 10> { }
  Expect: struct Array<T, const N: usize = 10> { }

test_lifetime_with_defaults
  Input:  sigil Ref<'a, T = &'a str> { }
  Expect: struct Ref<'a, T = &'a str> { }

test_complex_real_world
  Input:  sigil Cache<K, V, S: BuildHasher = RandomState, A: Allocator = Global>
  Expect: struct Cache<K, V, S: BuildHasher = RandomState, A: Allocator = Global>
```

### 2.3 Implementation Checklist

- [x] Parser: Distinguish default type param vs associated type binding ✅ Already works
- [x] Parser: Context flag for generic param list vs type argument list ✅ Position-based
- [x] AST: Ensure `GenericParam.default` is separate from `AssocTypeBinding` ✅ Correct
- [x] Codegen: Emit `=` for defaults in param list ✅ Works
- [x] Codegen: Emit `=` for associated types in arg list ✅ Works
- [x] All 12 edge cases pass ✅ Verified 2026-02-06

### 2.4 Success Criteria

Parser correctly distinguishes contexts. No conflation of defaults with associated types. ✅ VERIFIED

---

## Phase 3: Gap H - Path Resolution ✅ COMPLETE

**Completed:** 2026-02-06

### 3.1 Test File

`jormungandr/tests/rust_codegen/test_paths.sg`

### 3.2 Specification Tests (from EC₁-EC₉)

```
test_std_path
  Input:  std·collections·HashMap
  Context: Type
  Expect: std::collections::HashMap

test_self_field_access
  Input:  self·field
  Context: Expr (self is value)
  Expect: self.field

test_self_module_path
  Input:  self·module·Type
  Context: Type
  Expect: self::module::Type

test_local_binding_field_chain
  Input:  foo·bar·baz (where foo is local binding)
  Context: Expr
  Expect: foo.bar.baz

test_type_param_associated
  Input:  T·default()
  Context: Expr (T is type param)
  Expect: T::default()

test_struct_field_chain
  Input:  x·y·z (where x: SomeStruct)
  Context: Expr
  Expect: x.y.z

test_enum_constructor
  Input:  Option·Some(v)
  Context: Expr
  Expect: Option::Some(v)

test_associated_function
  Input:  Vec·new()
  Context: Expr
  Expect: Vec::new()

test_method_call
  Input:  vec·push(x) (where vec: Vec<T>)
  Context: Expr
  Expect: vec.push(x)

test_crate_from_sigil_toml
  Input:  nihil_core·Tensor (when nihil_core in Sigil.toml)
  Context: Type
  Expect: nihil_core::Tensor

test_super_path
  Input:  super·parent_mod·Item
  Context: Type
  Expect: super::parent_mod::Item

test_crate_path
  Input:  crate·internal·Helper
  Context: Type
  Expect: crate::internal::Helper
```

### 3.3 Implementation Checklist

- [x] Track local bindings for path resolution
- [x] Add `local_bindings: HashSet<String>` to RustCompiler
- [x] Add `collect_pattern_bindings()` helper
- [x] Track function parameters as bindings
- [x] Track let/let-else bindings when emitted
- [x] Check local_bindings first in `emit_expr_path`
- [x] All tests pass

### 3.4 Success Criteria

Local bindings correctly use `.` separator for method/field access.
Type paths and module paths correctly use `::` separator.

**Note:** Hardcoded crate lists remain for external crates (std, core, nihil_*). Full Sigil.toml
parsing is deferred - the local binding tracking covers the primary use case (method calls on
local variables).

---

## Phase 4: Gap J - Impl Block Where Clauses ✅ COMPLETE

**Completed:** 2026-02-06

### 4.1 Test File

`jormungandr/tests/rust_codegen/test_impl_where.sg`

### 4.2 Specification Tests (from EC₁-EC₁₄)

```
test_single_predicate
  Input:  ⊢<T> Foo<T> ∋ T: Clone { }
  Expect: impl<T> Foo<T> where T: Clone { }

test_multiple_predicates
  Input:  ⊢<T, U> Pair<T, U> ∋ T: Clone, U: Debug { }
  Expect: impl<T, U> Pair<T, U> where T: Clone, U: Debug { }

test_multiple_bounds
  Input:  ⊢<T> Foo<T> ∋ T: Clone + Debug + Send { }
  Expect: impl<T> Foo<T> where T: Clone + Debug + Send { }

test_lifetime_bound
  Input:  ⊢<'a, T> Foo<'a, T> ∋ T: 'a { }
  Expect: impl<'a, T> Foo<'a, T> where T: 'a { }

test_lifetime_outlives
  Input:  ⊢<'a, 'b> Ref<'a, 'b> ∋ 'a: 'b { }
  Expect: impl<'a, 'b> Ref<'a, 'b> where 'a: 'b { }

test_hrtb_in_impl
  Input:  ⊢<F> Handler<F> ∋ F: for<'a> Fn(&'a str) → &'a str { }
  Expect: impl<F> Handler<F> where F: for<'a> Fn(&'a str) -> &'a str { }

test_trait_impl_where
  Input:  ⊢<T> Iterator for Counter<T> ∋ T: Numeric { }
  Expect: impl<T> Iterator for Counter<T> where T: Numeric { }

test_associated_type_bound
  Input:  ⊢<I> Sum for I ∋ I: Iterator, I·Item: Add<Output = I·Item> { }
  Expect: impl<I> Sum for I where I: Iterator, I::Item: Add<Output = I::Item> { }

test_self_reference
  Input:  ⊢<T> Foo<T> ∋ T: From<Self> { }
  Expect: impl<T> Foo<T> where T: From<Self> { }

test_unsafe_impl_where
  Input:  unsafe ⊢<T> Send for Wrapper<T> ∋ T: Sync { }
  Expect: unsafe impl<T> Send for Wrapper<T> where T: Sync { }

test_fn_trait_bound
  Input:  ⊢<F, R> Callback<F, R> ∋ F: FnOnce() → R { }
  Expect: impl<F, R> Callback<F, R> where F: FnOnce() -> R { }

test_combined_impl_and_method_where
  Input:  ⊢<T> Foo<T> ∋ T: Clone { rite bar<U>(&self, u: U) ∋ U: Debug { } }
  Expect: impl<T> Foo<T> where T: Clone { fn bar<U>(&self, u: U) where U: Debug { } }
```

### 4.3 Implementation Checklist

- [x] AST: Add `where_clause: Option<WhereClause>` to `ImplBlock`
- [x] Parser: Store parsed where clause (was already being parsed but discarded)
- [x] Codegen: Call `emit_where_clause` in `emit_impl`
- [x] All tests pass

### 4.4 Success Criteria

Impl blocks support where clauses. Method-level constraints work alongside impl-level.

**Verified:** Generated code compiles and runs correctly.

---

## Phase 5: Gap K - impl Trait ✅ COMPLETE

**Completed:** 2026-02-06

**Discovery:** Feature was already implemented! The `⊢` symbol in type position parses as `TypeExpr::ImplTrait` and emits correctly as `impl Trait`.

### 5.1 Test File

`jormungandr/tests/rust_codegen/test_impl_trait.sg`

### 5.2 Specification Tests (from EC₁-EC₁₇)

```
test_return_impl_clone
  Input:  rite foo() → impl Clone { ... }
  Expect: fn foo() -> impl Clone { ... }

test_return_impl_fn
  Input:  rite make_adder(n: i32) → impl Fn(i32) → i32 { move |x| x + n }
  Expect: fn make_adder(n: i32) -> impl Fn(i32) -> i32 { move |x| x + n }

test_return_impl_fnmut
  Input:  rite counter() → impl FnMut() → i32 { ... }
  Expect: fn counter() -> impl FnMut() -> i32 { ... }

test_return_impl_fnonce
  Input:  rite consume(v: Vec<i32>) → impl FnOnce() → i32 { ... }
  Expect: fn consume(v: Vec<i32>) -> impl FnOnce() -> i32 { ... }

test_return_impl_iterator
  Input:  rite range(n: i32) → impl Iterator<Item = i32> { ... }
  Expect: fn range(n: i32) -> impl Iterator<Item = i32> { ... }

test_multiple_bounds
  Input:  rite foo() → impl Clone + Debug + Send { ... }
  Expect: fn foo() -> impl Clone + Debug + Send { ... }

test_lifetime_bound
  Input:  rite foo<'a>(s: &'a str) → impl Display + 'a { ... }
  Expect: fn foo<'a>(s: &'a str) -> impl Display + 'a { ... }

test_static_bound
  Input:  rite foo() → impl Fn() + 'static { ... }
  Expect: fn foo() -> impl Fn() + 'static { ... }

test_hrtb_fn
  Input:  rite foo() → impl for<'a> Fn(&'a str) → &'a str { ... }
  Expect: fn foo() -> impl for<'a> Fn(&'a str) -> &'a str { ... }

test_public_impl_trait
  Input:  ☉ rite foo() → impl Clone { ... }
  Expect: pub fn foo() -> impl Clone { ... }

test_generic_returning_impl
  Input:  rite wrap<T>(x: T) → impl AsRef<T> ∋ T: Clone { ... }
  Expect: fn wrap<T>(x: T) -> impl AsRef<T> where T: Clone { ... }

test_send_sync_bounds
  Input:  rite spawn() → impl Future<Output = ()> + Send + Sync { ... }
  Expect: fn spawn() -> impl Future<Output = ()> + Send + Sync { ... }

test_sigil_syntax_impl_trait
  Input:  rite foo() → ⊢ Iterator<Item = i32> { ... }
  Expect: fn foo() -> impl Iterator<Item = i32> { ... }
```

### 5.3 Implementation Checklist

- [x] AST: `TypeExpr::ImplTrait(Vec<TypeExpr>)` already exists
- [x] Parser: `⊢` (Token::Impl) in type position → `TypeExpr::ImplTrait`
- [x] Parser: Parses trait bounds via `parse_type_bounds()`
- [x] Codegen: Emits `impl ` + bounds with proper Fn trait handling
- [x] All tests pass

### 5.4 Success Criteria

Return-position impl Trait works using `⊢` syntax.

**Note:** Rust `impl` keyword is deprecated in Sigil. Use `⊢` for impl Trait:
```sigil
rite foo() → ⊢ Clone { ... }        // fn foo() -> impl Clone { ... }
rite bar() → ⊢ Fn(i32) → i32 { ... } // fn bar() -> impl Fn(i32) -> i32 { ... }
```

---

## Phase 6: Gap I - Raw Pointer Annotations ✅ OPTION A COMPLETE

**Completed:** 2026-02-06

**Status:** Option A (explicit source annotations) works. Option B (typechecker-informed) deferred.

### 6.1 Architectural Change (Option B - DEFERRED)

Option B requires threading type information from typechecker to codegen.

### 6.2 Prerequisites

- Typechecker produces `Map<ExprId, Type>` or typed AST
- Codegen receives type map as input
- Codegen can query inferred types for bindings

### 6.3 Specification Tests

```
test_unannotated_const_pointer_with_add
  Input:  ≔ ptr = slice.as_ptr()
          unsafe { *ptr.add(i) }
  Expect: let ptr: *const T = slice.as_ptr()  // T inferred
          unsafe { *ptr.add(i) }

test_unannotated_mut_pointer
  Input:  ≔ ptr = slice.as_mut_ptr()
          unsafe { *ptr.add(i) = v }
  Expect: let ptr: *mut T = slice.as_mut_ptr()

test_pointer_no_annotation_needed
  Input:  ≔ ptr = data.as_ptr()
          ⎇ ptr.is_null() { ... }
  Expect: let ptr = data.as_ptr()  // No annotation, is_null doesn't need it
```

### 6.4 Implementation Checklist

**Option A (Complete):**
- [x] Parser: Handle `*◆ T` as `*const T`
- [x] Parser: Handle `*vary T` / `*Δ T` as `*mut T`
- [x] Codegen: Preserve explicit type annotations
- [x] Test file compiles with rustc

**Option B (Deferred):**
- [ ] Typechecker: Produce type map alongside AST
- [ ] Codegen: Accept type map parameter
- [ ] Codegen: Detect bindings where Rust needs annotation
- [ ] Codegen: Query type map and emit annotation

### 6.5 Success Criteria

**Option A:** Source with explicit pointer type annotations compiles correctly.

**Option B (future):** Unannotated pointer bindings auto-emit correct type when used with `.add()`, `.offset()`, etc.

---

## Test Infrastructure

### Running Tests

```bash
cd jormungandr/tests
./run_tests_rust.sh --spec rust_codegen
```

### Test File Structure

```
jormungandr/tests/rust_codegen/
├── test_return.sg           # Gap L
├── test_return.rs.expected  # Expected Rust output
├── test_generics.sg         # Gap B
├── test_generics.rs.expected
├── test_paths.sg            # Gap H
├── test_paths.rs.expected
├── test_impl_where.sg       # Gap J
├── test_impl_where.rs.expected
├── test_impl_trait.sg       # Gap K
├── test_impl_trait.rs.expected
└── ...
```

### Validation Process

1. Run Sigil compiler with `rust` backend
2. Diff output against `.rs.expected`
3. Compile generated Rust with `rustc --edition 2021`
4. All three steps must pass

---

## Revision History

| Version | Date | Changes |
|---------|------|---------|
| 1.0.0 | 2026-02-06 | Initial roadmap derived from S++ specs |
| 1.1.0 | 2026-02-06 | Phases 1-2 verified, Phase 7 (Gap M) implemented |
| 1.2.0 | 2026-02-06 | All phases complete: H, J, K, I (Option A) implemented |
