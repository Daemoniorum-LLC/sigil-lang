# Sigil LLVM Backend: Generic Method Dispatch Specification

**Version:** 0.1.0
**Status:** Draft — Gap Identified
**Date:** 2026-02-22
**Parent Spec:** 18-COMPILER-ARCHITECTURE.md, 03C-CONST-GENERIC-INFERENCE.md

---

## 1. Conceptual Foundation

### 1.1 What This Spec Covers

The Sigil LLVM backend performs **monomorphisation**: when a generic function or method
is compiled, one concrete LLVM function is emitted per distinct set of type arguments.
The name of each concrete function encodes the type arguments (mangling).

This spec defines the rule by which the LLVM backend selects the correct monomorphised
variant when compiling a method call expression — specifically when multiple `⊢` impl
blocks exist for the same method name, differentiated only by their type arguments.

### 1.2 The Core Problem

Given:

```
sigil Buffer<Dev> { tag: i64 }

⊢ Buffer<Cpu> { ☉ rite new() → Buffer<Cpu> { Buffer { tag: 0 } } }
⊢ Buffer<Gpu> { ☉ rite new() → Buffer<Gpu> { Buffer { tag: 1 } } }
```

When a call site says:

```
≔ b: Buffer<Gpu> = Buffer·new();
```

The LLVM backend must call `Buffer_new_Gpu` (the `Gpu` impl), not `Buffer_new_Cpu`.

This requires the compiler to propagate the **declared return type annotation** from
the binding site into the method call resolution path.

### 1.3 Why This Is Non-Trivial

Method call resolution (G63-MCALL in `llvm_codegen.rs`) looks up the callee by
constructing a mangled name from the receiver type and method name. Without the return
type annotation, the compiler cannot distinguish `Buffer·new()` targeting `Buffer<Gpu>`
from `Buffer·new()` targeting `Buffer<Cpu>`, because both look identical at the call
site.

The type checker correctly records the declared annotation. The gap is that the LLVM
codegen phase does not consult it.

---

## 2. Behavioral Contracts

### 2.1 Return-Type-Directed Dispatch (R1 — MUST)

**Status:** ✅ Implemented 2026-02-22

When a method call expression is the right-hand side of a `≔` binding with an explicit
type annotation, the LLVM backend MUST use that annotation to disambiguate among impl
blocks.

```
≔ x: Foo<Bar> = Foo·method(args)
```

The compiler MUST call the `Foo<Bar>` impl of `method`, not any other concrete
impl of `Foo`.

### 2.2 Type-Argument Propagation (R2 — MUST)

**Status:** ✅ Implemented 2026-02-22

Each type parameter in the declared annotation contributes to the mangled callee name:

```
declared type: Foo<Bar, Baz>
mangled callee: Foo_method_Bar_Baz   (exact mangling per §3)
```

Both the struct's own type parameters AND any const-generic shape parameters (e.g.,
`[512, 256]`) must be reflected in the mangle.

### 2.3 Fallback Behaviour (R3 — SHOULD)

When no return type annotation is present and the method is unambiguous (only one impl
block exists for that method name on that receiver), the compiler SHOULD proceed without
annotation-based disambiguation.

When the method is ambiguous and no annotation is present, the compiler MUST emit a
meaningful diagnostic rather than silently selecting a wrong impl.

### 2.4 Argument-Type Inference (R4 — MAY)

As a future enhancement, type arguments inferred from the *argument* types (not the
return type) may also contribute to disambiguation. This is not required by this spec
version.

---

## 3. Name Mangling for Generic Impls

The following mangling scheme maps type arguments to name fragments:

| Type argument | Mangled fragment |
|---------------|-----------------|
| Named type `Cpu` | `Cpu` |
| Named type `Cuda` | `Cuda` |
| Const array shape `[512, 256]` | `shape_512x256` |
| Dynamic shape (fallback) | `DynShape` |
| Primitive `f32` | `f32` |
| Primitive `f64` | `f64` |

Full example:

```
Tensor<[512, 256], f32, Cuda>·from_storage(args)
  → Tensor_from_storage_shape_512x256_f32_Cuda
```

---

## 4. Implementation Gap Analysis

### 4.1 Root Cause

In `llvm_codegen.rs`, the G63-MCALL path (method call compilation, approximately line
7676) constructs the mangled callee name from the **receiver's inferred type** at the
call site. It does not consult the **declared return type annotation** of the enclosing
`Stmt::Let` binding.

When the receiver is a bare type name (`Tensor·from_storage`), the compiler has no
runtime value to infer generic parameters from, so it falls back to the first registered
impl for that method — typically the one registered earliest, which is `Cpu`/`DynShape`.

### 4.2 Where the Fix Belongs

The fix requires threading the expected return type through the call compilation:

```
compile_let_binding(name, annotation_ty, rhs_expr):
    if rhs_expr is Expr::MethodCall:
        compile_method_call(rhs_expr, hint=annotation_ty)
    ...

compile_method_call(expr, hint=None):
    mangle = build_mangle(receiver_type, method_name, type_args_from_hint(hint))
    callee = module.get_function(mangle) or error
    ...
```

The `annotation_ty` from the `≔ x: T = ...` binding must be passed as a `hint` into
the method call compilation, and `type_args_from_hint` must decompose `T` into its
type parameters for mangling.

### 4.3 Affected Code Paths

| Location | Description |
|----------|-------------|
| `llvm_codegen.rs` `compile_stmt` → `Stmt::Let` | Extract annotation type, pass to rhs compilation |
| `llvm_codegen.rs` `compile_call` / `compile_method_call` | Accept type hint parameter |
| `llvm_codegen.rs` `type_expr_to_name` | Already has `TypeExpr::ConstExpr` handler (partial fix applied 2026-02-22) |

---

## 5. Constraints and Invariants

```
I1: ∀ call site with declared annotation T:
    called_impl(call) = impl registered for T
    // No silent fallback to wrong impl

I2: ∀ call site without annotation:
    single_impl_exists(method, receiver) ⟹ call succeeds
    multiple_impl_exist(method, receiver) ⟹ diagnostic emitted

I3: mangling is injective:
    ∀ T1 ≠ T2: mangle(T1) ≠ mangle(T2)
    // Distinct types never produce the same mangled name
```

---

## 6. Integration Points

- **Type checker** (`typeck.rs`): Already resolves and records the declared type for
  `Stmt::Let` bindings. The LLVM codegen must consume this information.
- **`type_expr_to_name`** (`llvm_codegen.rs`): Converts a `TypeExpr` to a mangled
  name fragment. Partially updated (2026-02-22) to handle `TypeExpr::ConstExpr`.
- **Nihil tensor framework**: The primary consumer. `Tensor<[M,N], f32, Cuda>` method
  calls are the motivating case.
- **`declare_runtime_functions`** (`llvm_codegen.rs`): Unrelated to dispatch, but impl
  registration must follow the same mangling convention.

---

## 7. Open Questions

1. **Nested calls**: What if the return type annotation is on an outer expression, not
   directly on the binding? e.g., `≔ x = vec_of(Foo·new())`. Should hint propagate
   through function arguments?

2. **Turbofish syntax**: Should `Foo·new·<Gpu>()` be the canonical way to specify type
   args at the call site, rather than relying on return type annotation?
   - Pro: Explicit, doesn't require two-pass compilation
   - Con: More verbose, Rust-style turbofish is unfamiliar to new Sigil users

3. **Bidirectional inference**: Should type args inferred from argument types AND return
   type both contribute? How to handle conflicts?

---

## 8. Revision History

| Version | Date | Changes |
|---------|------|---------|
| 0.2.0 | 2026-02-22 | R1 and R2 implemented. Three-layer fix: (1) `interpreter.rs` — R1 dispatch runs before environment lookup in `eval_path`; hint set from annotation in `Stmt::Let` handler; full-type keys registered in `Item::Impl` handler. (2) `llvm_codegen.rs` — `declare_impl_methods`/`compile_impl_methods` include type-args suffix in mangled name; G128-EARLY uses `current_let_return_type` hint. (3) `typeck.rs` — registers under full key and threads `dispatch_hint`. Test `test_generic_return_type_dispatch.sg` GREEN in both `sigil run` and `sigil compile`. Suite: 783/788 passing. |
| 0.1.0 | 2026-02-22 | Initial gap documentation. Discovered during Nihil CUDA dogfooding: `Tensor<[M,N], f32, Cuda>·from_storage()` dispatches to `Tensor<DynShape, f32, Cpu>` impl instead of the declared type. |
