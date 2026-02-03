# Type Checker Enhancement Spec: Generic Inference & Coercions

> **Status:** ? Draft
> **Branch:** `feature/typeck-generic-inference`
> **Depends On:** 03-TYPES.md, 03A-TYPE-INFERENCE.md
> **Motivated By:** nihil tensor framework dogfooding (40/49 pass, 9 remaining)

## 0. Context

Dogfooding the Sigil compiler against the nihil tensor framework (49 `.sigil` files
across 22 crates) exposed five categories of type checker limitations. All 9 remaining
failures are compiler-side — the nihil code is correct.

This spec defines the enhancements needed, ordered by impact (files unblocked).

---

## 1. Problem Summary

| # | Category | Root Cause | Files Blocked |
|---|----------|-----------|---------------|
| A | Type variable unification | Fresh vars per call site don't unify | 5 |
| B | Match arm type narrowing | Generic params not constrained by pattern | 2 |
| C | Recursive auto-deref | `&&T` and `&Arc<T>` not coerced | 3 |
| D | Method resolution on generics | Lookup fails on parameterized receivers | 1 |
| E | Expression type inference | Complex blocks resolve to `()` or wrong type | 2 |

**Note:** Some files have overlapping categories. The 9 unique failing files are:
nihil-cuda/flash_attn, nihil-dispatch, nihil-distributed, nihil-einsum, nihil-linalg,
nihil-quant/validation, nihil-test/lib, nihil-test/shapes, nihil-transformer.

---

## 2. Enhancement A: Cross-Call-Site Type Variable Unification

### 2.1 Problem

When a generic function is called, `freshen()` creates fresh type variables for each
call site. When results from two calls to the *same* generic function flow into
the *same* consumer, the type variables don't unify.

**Example (nihil-einsum):**
```sigil
// contract_pair has signature:
//   rite contract_pair<S: Shape, D: DType, Dev: Device>(
//       a: &Tensor<S, D, Dev>, b: &Tensor<S, D, Dev>, indices: &[usize]
//   ) -> Tensor<S, D, Dev>

// Call site 1: creates Tensor<impl Shape, ?188, ?189>
≔ a = get_tensor_a();
// Call site 2: creates Tensor<impl Shape, ?192, ?193>
≔ result = contract_pair(a, b, &step.contracted_indices)?;

// The function expects Vec<Tensor<impl Shape, ?192, ?193>> but gets
// Vec<Tensor<impl Shape, ?188, ?189>> — different type var IDs.
```

The type checker sees `?188 != ?192` and `?189 != ?193` and reports a mismatch,
even though both sets represent the same conceptual `D` and `Dev` parameters.

### 2.2 Current Implementation

In `typeck.rs`:
- `freshen()` replaces generic type parameters with fresh vars (new IDs each call)
- `unify()` for `Type::Named` (line ~3227) requires name match + recursive generic unification
- When both sides have unresolved vars (`?188` vs `?192`), unification succeeds
  by binding one to the other — **but this only works when the types reach `unify()` directly**

The actual failure occurs because the types are nested inside `Vec<Tensor<...>>`,
and the `Vec` wrapper's generics contain independently-freshened `Tensor` types
that never get a chance to unify their inner vars.

### 2.3 Solution: Eager Substitution Before Argument Checking

Before comparing argument types against parameter types, resolve all type variables
to their current best-known substitution. This collapses chains like
`?188 -> ?192 -> Tensor<...>` into the concrete type.

**Implementation:**

Add a `deep_resolve(ty: &Type) -> Type` function that recursively walks a type,
replacing all `Type::Var(v)` with their substitution (transitively), and apply it
before argument comparison in function call checking.

```rust
fn deep_resolve(&self, ty: &Type) -> Type {
    match ty {
        Type::Var(v) => {
            if let Some(resolved) = self.substitutions.get(v) {
                if resolved != ty {
                    self.deep_resolve(resolved)
                } else {
                    ty.clone()
                }
            } else {
                ty.clone()
            }
        }
        Type::Named { name, generics } => Type::Named {
            name: name.clone(),
            generics: generics.iter().map(|g| self.deep_resolve(g)).collect(),
        },
        Type::Ref { inner, mutable, lifetime } => Type::Ref {
            inner: Box::new(self.deep_resolve(inner)),
            mutable: *mutable,
            lifetime: lifetime.clone(),
        },
        Type::Array { element, size } => Type::Array {
            element: Box::new(self.deep_resolve(element)),
            size: *size,
        },
        Type::Slice(inner) => Type::Slice(Box::new(self.deep_resolve(inner))),
        Type::Evidential { inner, evidence } => Type::Evidential {
            inner: Box::new(self.deep_resolve(inner)),
            evidence: *evidence,
        },
        // ... handle all Type variants
        _ => ty.clone(),
    }
}
```

**Apply at call site (line ~1790):**
```rust
// Before checking arguments against parameters:
let resolved_arg = self.deep_resolve(&arg_ty);
let resolved_param = self.deep_resolve(param);
if !self.unify(&resolved_param, &resolved_arg) {
    // report error
}
```

### 2.4 Acceptance Criteria

- `Vec<Tensor<impl Shape, ?A, ?B>>` unifies with `Vec<Tensor<impl Shape, ?C, ?D>>`
  when `?A` and `?C` (or `?B` and `?D`) resolve to the same type through substitution chains
- nihil-einsum, nihil-linalg, nihil-test/lib pass type checking
- All 749 existing tests continue to pass

---

## 3. Enhancement B: Match Arm Type Narrowing

### 3.1 Problem

When a match expression dispatches on a value that determines a generic type parameter,
the type checker doesn't narrow that parameter within each arm.

**Example (nihil-dispatch):**
```sigil
// Generic function with device parameter:
rite fused_matmul_bias_act<S: Shape, D: DType, Dev: Device>(
    a: &Tensor<S, D, Dev>, b: &Tensor<S, D, Dev>,
    bias: &Tensor<S, D, Dev>, activation: Activation,
) -> Tensor<S, D, Dev> {
    ⌥ a.device_type() {
        DeviceType·Cuda => fused_matmul_bias_act_cuda(a, b, bias, activation),
        //                 ^ expects &Tensor<S, D, Cuda>
        //                   but a is still &Tensor<S, D, ?107>
        DeviceType·Cpu  => fused_matmul_bias_act_cpu(a, b, bias, activation),
    }
}
```

The checker knows `a.device_type() == DeviceType::Cuda` in the first arm,
but doesn't propagate that `Dev = Cuda`.

### 3.2 Current Implementation

Match arm checking (line ~1985-2040):
1. Infers scrutinee type
2. For each arm: pushes scope, binds pattern vars, checks body
3. Unifies all arm return types

Pattern binding only extracts variables from destructured patterns — it doesn't
constrain existing variables' generic parameters based on the matched variant.

### 3.3 Solution: Type Narrowing Via Pattern Context

This is a hard problem in general (it requires understanding that `a.device_type()`
returning `Cuda` implies `Dev = Cuda` for the type parameter of `a`). Full
dependent-type-style narrowing is out of scope.

**Pragmatic approach: Accept the mismatch when only type variables differ.**

Enhance `unify()` to recognize that `Tensor<S, D, Cuda>` and `Tensor<S, D, ?107>`
should unify by binding `?107 = Cuda`. This already works when the types reach
`unify()` — the issue is that the argument types are both wrapped as
`&Tensor<impl Shape, ?106, ?107>` (caller's generic) vs `&Tensor<impl Shape, ?110, Cuda>`
(callee's concrete).

This is actually a sub-case of Enhancement A. If `deep_resolve` is applied and `?107`
is still unresolved, then `unify(?107, Cuda)` should succeed by binding `?107 = Cuda`.

**However**, there's also a structural issue: the callee's parameter type is freshened
independently, creating `?110` for `D`. The caller's `?106` and callee's `?110` both
represent `D` but are separate variables.

**Fix:** After freshening callee parameters, pre-unify the caller's generic args with
the callee's freshened generics to establish equivalences before checking individual
arguments.

### 3.4 Acceptance Criteria

- `&Tensor<impl Shape, ?106, ?107>` passed to `fn(a: &Tensor<impl Shape, ?110, Cuda>)`
  succeeds by binding `?107 = Cuda` and `?106 = ?110`
- nihil-dispatch, nihil-transformer (device dispatch errors) pass type checking
- No false positives: genuine type mismatches still reported

---

## 4. Enhancement C: Recursive Auto-Deref & Smart Pointer Coercions

### 4.1 Problem

Three coercion cases that the type checker doesn't handle:

1. **Double deref:** `&&T` should coerce to `&T`
2. **Arc deref:** `&Arc<T>` should coerce to `&T`
3. **Owned-to-ref:** `Tensor<...>` passed where `&Tensor<...>` expected (auto-ref already
   partially exists but fails with unresolved type vars)

**Examples:**
```sigil
// nihil-distributed: &Arc<ProcessGroup> passed where &ProcessGroup expected
≔ pg: Arc<ProcessGroup> = ...;
all_reduce(&tensor, &pg);  // &pg is &Arc<ProcessGroup>, not &ProcessGroup

// nihil-quant/validation: &&T passed where &T expected
≔ tensor: &Tensor = ...;
validate(&tensor);  // &tensor is &&Tensor

// nihil-transformer: &&Tensor in rotate_half
≔ x: &Tensor<S, D, Dev> = ...;
some_fn(&x);  // &&Tensor
```

### 4.2 Current Implementation

Reference coercions (line ~3290-3331) handle:
- `&mut T` -> `&T` (reborrow)
- `&Box<T>` -> `&T` (deref)
- `&Vec<T>` -> `&[T]` (unsized coercion)

These are checked via `types_structurally_equal()` and only go one level deep.
No handling for `Arc`, `Rc`, or recursive deref chains.

### 4.3 Solution: Recursive Deref Chain

Extend `unify()` with a deref-chain fallback when initial unification fails on
reference types.

**Step 1: Double-deref**

When unifying `&T_expected` with `&&T_actual`, strip one `&` from actual and retry:

```rust
// In unify(), after main match fails:
(Type::Ref { inner: expected, .. }, Type::Ref { inner: actual, .. }) => {
    // If actual is &&T, try stripping one layer
    if let Type::Ref { inner: inner_actual, .. } = actual.as_ref() {
        if self.unify(expected, inner_actual) {
            return true;
        }
    }
    false
}
```

**Step 2: Smart pointer deref**

Recognize `Arc<T>`, `Rc<T>`, `Box<T>` as deref-transparent:

```rust
const DEREF_TYPES: &[&str] = &["Arc", "Rc", "Box", "Cell", "RefCell", "Mutex"];

// When unifying &T with &SmartPtr<T>:
fn try_smart_deref(&self, expected: &Type, actual: &Type) -> bool {
    if let Type::Named { name, generics } = actual {
        if DEREF_TYPES.contains(&name.as_str()) && !generics.is_empty() {
            return self.unify(expected, &generics[0]);
        }
    }
    false
}
```

**Step 3: Integrate into unify fallback**

After normal `Ref` unification fails:
```rust
(Type::Ref { inner: exp_inner, .. }, Type::Ref { inner: act_inner, .. }) => {
    // Normal unification
    if self.unify(exp_inner, act_inner) { return true; }
    // Try double-deref: &&T -> &T
    if let Type::Ref { inner: inner2, .. } = act_inner.as_ref() {
        if self.unify(exp_inner, inner2) { return true; }
    }
    // Try smart pointer deref: &Arc<T> -> &T
    if self.try_smart_deref(exp_inner, act_inner) { return true; }
    false
}
```

### 4.4 Acceptance Criteria

- `&Arc<ProcessGroup>` unifies with `&ProcessGroup`
- `&&Tensor` unifies with `&Tensor`
- `&Rc<T>` unifies with `&T`
- nihil-distributed, nihil-quant/validation pass type checking
- nihil-transformer `&&Tensor` errors resolved
- No false unifications: `&Arc<T>` does NOT unify with `&U` when `T != U`
- All 749 existing tests pass

---

## 5. Enhancement D: Method Resolution on Parameterized Types

### 5.1 Problem

Method calls on values with parameterized types fail to resolve correctly when the
receiver type has unresolved generic parameters.

**Example (nihil-test/shapes):**
```sigil
// tensor.shape() should return &[USize], but resolves to Option<?>
≔ tensor: Tensor<impl Shape, D, Dev> = ...;
≔ s = tensor.shape();  // Expected: &[USize], Got: Option<?>
broadcast_shape(&s, &other_s);
// Error: expected &[USize], found &Option<?>
```

### 5.2 Current Implementation

Method resolution (line ~2042-2261) has two paths:
1. **User-defined methods:** Looks up `impl_methods[type_name][method_name]`
2. **Hardcoded patterns:** Falls back to pattern matching on method name

The issue is in path 1: when the receiver type is `Tensor<impl Shape, ?106, ?107>`,
the type name extracted might not match the key in `impl_methods`. The lookup uses
the string representation of the type, and parameterized types produce different
strings than the definition.

When lookup fails, it falls through to hardcoded patterns, where `.shape()` isn't
listed, and then to the default case which returns `self.fresh_var()`. This fresh
var then gets unified with whatever context demands — if context expects `&[USize]`
somewhere else but `Option<T>` unifies first (from a different branch), the wrong
type propagates.

### 5.3 Solution: Normalize Type Names for Method Lookup

Strip generic parameters when looking up methods:

```rust
fn resolve_type_name(ty: &Type) -> Option<String> {
    match ty {
        Type::Named { name, .. } => Some(name.clone()),
        Type::Ref { inner, .. } => resolve_type_name(inner),
        Type::Evidential { inner, .. } => resolve_type_name(inner),
        _ => None,
    }
}
```

When looking up `impl_methods`, use the base type name (`"Tensor"`) not the
full parameterized form. Then freshen the method signature and unify the
receiver's generics with the impl's generics.

### 5.4 Acceptance Criteria

- `tensor.shape()` on `Tensor<impl Shape, D, Dev>` resolves to `&[USize]`
  (or whatever the impl defines)
- nihil-test/shapes passes type checking
- Method calls on `Vec<T>`, `HashMap<K,V>`, etc. continue to work
- All 749 existing tests pass

---

## 6. Enhancement E: Complex Expression Type Resolution

### 6.1 Problem

Multi-statement blocks and closures sometimes resolve to `()` (unit) or completely
wrong types (`fn(F64) -> F64` where `Tensor` is expected).

**Example (nihil-einsum):**
```sigil
// Expected: Tensor<...>, Got: ()
≔ result = contract_pair(a, b, &step.contracted_indices)?;
```

**Example (nihil-transformer):**
```sigil
// Expected: &Tensor<...>, Got: &fn(F64) -> F64
// In a large function body spanning lines 568-660
```

### 6.2 Analysis

These appear to be cascading failures rather than independent bugs:

1. **`found ()`**: Likely caused by an earlier type error in the expression chain.
   When `contract_pair(a, b, ...)` fails argument checking (due to Enhancement A's
   type var issue), the return type may not propagate correctly, falling back to `()`.

2. **`found &fn(F64) -> F64`**: In a large function body, an earlier unification
   failure can corrupt the substitution map, causing downstream variables to resolve
   to unrelated types.

### 6.3 Solution: Error Recovery in Type Inference

Rather than a specific fix, these should resolve once Enhancements A-D are
implemented. The cascading nature means fixing the root cause (unresolved type
variables, failed method resolution) will eliminate the downstream symptoms.

**Verification approach:**
1. Implement Enhancements A-D
2. Re-check nihil-einsum and nihil-transformer
3. If `()` or wrong-type errors persist, investigate individually

**If independent issues remain**, the fix is likely:
- Ensure that when `infer_expr()` encounters an error, it still returns the
  best-guess type rather than `()` or `fresh_var()` — preserving any partial
  type information that was successfully resolved

### 6.4 Acceptance Criteria

- nihil-einsum `found ()` errors gone after Enhancement A
- nihil-transformer `found &fn(F64) -> F64` errors gone after Enhancements A+C
- If not, targeted fixes added and documented

---

## 7. Implementation Order

Enhancements should be implemented in dependency order:

```
A (deep_resolve)  ──────────────────────────────────┐
                                                     ├──> E (verify cascading fixes)
C (recursive deref) ─> B (match narrowing, uses A) ─┘
                       D (method resolution)
```

### Phase 1: Foundation
1. **Enhancement A** — `deep_resolve()` + apply before argument checking
   - Highest impact: unblocks 5 files directly
   - Foundation for Enhancement B

### Phase 2: Coercions
2. **Enhancement C** — Recursive auto-deref (`&&T`, `&Arc<T>`)
   - Independent of A, can be done in parallel
   - Unblocks 3 files

### Phase 3: Refinements
3. **Enhancement B** — Match arm narrowing (leverages A's deep_resolve)
4. **Enhancement D** — Method resolution on generic types

### Phase 4: Verification
5. **Enhancement E** — Verify cascading fixes, targeted patches if needed

### Expected Outcome

| Phase | Files Fixed | Running Total |
|-------|------------|---------------|
| Phase 1 (A) | nihil-einsum, nihil-linalg, nihil-test/lib, nihil-cuda/flash_attn (partial), nihil-transformer (partial) | ~43-44/49 |
| Phase 2 (C) | nihil-distributed, nihil-quant/validation, nihil-transformer (partial) | ~45-46/49 |
| Phase 3 (B,D) | nihil-dispatch, nihil-test/shapes, nihil-transformer (remaining) | ~48-49/49 |
| Phase 4 (E) | Remaining cascading issues | 49/49 |

---

## 8. Testing Strategy

### 8.1 Unit Tests (compiler test suite)

Add targeted tests for each enhancement:

```
jormungandr/tests/spec/03_types/P0_050_generic_unification.sg
jormungandr/tests/spec/03_types/P0_051_match_type_narrowing.sg
jormungandr/tests/spec/03_types/P0_052_auto_deref_recursive.sg
jormungandr/tests/spec/03_types/P0_053_method_resolution_generic.sg
```

### 8.2 Integration Tests (nihil dogfood)

After each phase, run the nihil type check suite:
```bash
cd /home/crook/dev/nihil
for f in $(find crates -name '*.sigil' | sort); do
    sigil check "$f" 2>&1
done
```

Track pass/fail counts to verify forward progress.

### 8.3 Regression Guard

All 749 existing compiler tests must pass after each change.
No existing test may change behavior.

---

## 9. Non-Goals

This spec does **not** cover:
- Full dependent type checking (spec 03A)
- SMT-backed refinement solving (spec 03A)
- Cross-module type checking (workspace-level)
- Trait resolution / trait object dispatch
- Lifetime inference
- Const generics
- Higher-kinded types

These are separate concerns documented in their respective specs.

---

## 10. References

- `parser/src/typeck.rs` — Current type checker implementation
- `docs/specs/03-TYPES.md` — Type system spec
- `docs/specs/03A-TYPE-INFERENCE.md` — Inference algorithm spec
- nihil framework: `github.com/Daemoniorum-LLC/nihil` branch `feature/sigil-native-syntax`
- Previous work: `feature/each-of-syntax-and-workspace-migrate` branch
  (evidence checking, try operator, bitwise NOT, bool conditions)
