# Sigil Compiler Feature Audit

## Executive Summary

This document provides a comprehensive audit of features implemented in the Rust bootstrap compiler vs. the Jormungandr self-hosted compiler, compared against the language specification.

**Audit Date:** 2026-01-21

---

## 1. Compilers Overview

### 1.1 Rust Bootstrap Compiler (Canonical)

- **Location:** `parser/`
- **Status:** 466/531 tests passing (87%)
- **Components:**
  - `lexer.rs` - 1,399+ lines, full token support
  - `parser.rs` - 8,900+ lines, complete AST parsing
  - `interpreter.rs` - 2,400+ lines, runtime execution
  - `llvm_codegen.rs` - 2,200+ lines, native compilation
  - `stdlib.rs` - 1.2MB, comprehensive standard library

### 1.2 Jormungandr Self-Hosted Compiler

- **Location:** `jormungandr/src/`
- **Status:** Experimental
- **Components:**
  - `token.sg` - 611 lines, token definitions
  - `lexer.sg` - Hand-written lexer
  - `parser.sg` - Recursive descent parser
  - `interp.sg` - Interpreter
  - `codegen.sg` - C code generation

---

## 2. Core Syntax Features

### 2.1 Variable Bindings

| Feature | Bootstrap | Jormungandr | Spec |
|---------|-----------|-------------|------|
| `let x = expr` | ✅ | ✅ | ✅ |
| `let mut x = expr` | ✅ | ✅ | ✅ |
| `let x: Type = expr` | ✅ | ✅ | ✅ |
| Pattern matching in let | ✅ | ✅ | ✅ |
| `let-else` | ✅ | ❓ | ✅ |

**Note:** Neither compiler implements `≔` or `vary` keywords - these are NOT part of the Sigil spec.

### 2.2 Functions

| Feature | Bootstrap | Jormungandr | Spec |
|---------|-----------|-------------|------|
| `fn name() { }` | ✅ | ✅ | ✅ |
| `fn name() -> Type` | ✅ | ✅ | ✅ |
| `async fn` | ✅ | ✅ | ✅ |
| `const fn` | ✅ | ❓ | ✅ |
| `naked fn` | ✅ | ❌ | ❓ |
| Generic functions | ✅ | ✅ | ✅ |
| Trait bounds | ✅ | ✅ | ✅ |
| Lifetime annotations | ✅ | ✅ | ✅ |

**Note:** `rite` is NOT a Sigil keyword - use `fn`.

### 2.3 Control Flow

| Feature | Bootstrap | Jormungandr | Spec |
|---------|-----------|-------------|------|
| `if`/`else` | ✅ | ✅ | ✅ |
| `if let` | ✅ | ✅ | ✅ |
| `match` | ✅ | ✅ | ✅ |
| `loop` | ✅ | ✅ | ✅ |
| `while` | ✅ | ✅ | ✅ |
| `for`...`in` | ✅ | ✅ | ✅ |
| Labeled loops | ✅ | ❓ | ✅ |
| `break`/`continue` | ✅ | ✅ | ✅ |
| `return` | ✅ | ✅ | ✅ |
| `yield` (generators) | ❓ | ❌ | ✅ |

**Note:** `⎇`/`⎉` are NOT Sigil keywords - use standard `if`/`else`.

### 2.4 Types & Declarations

| Feature | Bootstrap | Jormungandr | Spec |
|---------|-----------|-------------|------|
| `struct` / `sigil` | ✅ | ✅ | ✅ |
| `enum` | ✅ | ✅ | ✅ |
| `trait` | ✅ | ✅ | ✅ |
| `impl` | ✅ | ✅ | ✅ |
| `type` aliases | ✅ | ✅ | ✅ |
| `const` items | ✅ | ✅ | ✅ |
| `static` items | ✅ | ✅ | ✅ |
| Generics | ✅ | ✅ | ✅ |
| Where clauses | ✅ | ✅ | ✅ |

### 2.5 Module System

| Feature | Bootstrap | Jormungandr | Spec |
|---------|-----------|-------------|------|
| `mod` / `scroll` | ✅ | ✅ | ✅ |
| `use` / `invoke` | ✅ | ✅ | ✅ |
| `pub` visibility | ✅ | ✅ | ✅ |
| `·` (middledot) access | ✅ | ✅ | ✅ |
| `crate` / `tome` | ✅ | ✅ | ✅ |
| `self` | ✅ | ✅ | ✅ |
| `super` | ✅ | ✅ | ✅ |

---

## 3. Sigil-Specific Features

### 3.1 Morphemes (Greek Letters)

| Morpheme | Unicode | Bootstrap | Jormungandr | Spec |
|----------|---------|-----------|-------------|------|
| Transform | τ/Τ | ✅ | ✅ | ✅ |
| Filter | φ/Φ | ✅ | ✅ | ✅ |
| Sort | σ/Σ | ✅ | ✅ | ✅ |
| Reduce | ρ/Ρ | ✅ | ✅ | ✅ |
| Lambda | λ/Λ | ✅ | ✅ | ✅ |
| Product | Π | ✅ | ✅ | ✅ |
| Delta | δ/Δ | ✅ | ✅ | ✅ |
| Epsilon | ε | ✅ | ✅ | ✅ |
| Omega | ω/Ω | ✅ | ✅ | ✅ |
| Alpha | α | ✅ | ✅ | ✅ |
| Zeta (combine) | ζ | ✅ | ✅ | ✅ |

### 3.2 Evidentiality Markers

| Marker | Symbol | Bootstrap | Jormungandr | Spec |
|--------|--------|-----------|-------------|------|
| Known | `!` | ✅ | ✅ | ✅ |
| Uncertain | `?` | ✅ | ✅ | ✅ |
| Reported | `~` | ✅ | ✅ | ✅ |
| Paradox | `‽` | ✅ | ✅ | ✅ |
| Predicted | `◊` | ✅ | ✅ | ✅ |

### 3.3 Set Operations

| Operation | Symbol | Bootstrap | Jormungandr | Spec |
|-----------|--------|-----------|-------------|------|
| Union | ∪ | ✅ | ✅ | ✅ |
| Intersection | ∩ | ✅ | ✅ | ✅ |
| Set Minus | ∖ | ✅ | ✅ | ✅ |
| Subset | ⊂ | ✅ | ✅ | ✅ |
| Superset | ⊃ | ✅ | ✅ | ✅ |
| For All | ∀ | ✅ | ✅ | ✅ |
| Exists | ∃ | ✅ | ✅ | ✅ |
| Element Of | ∈ | ✅ | ✅ | ✅ |

### 3.4 Logic Operators

| Operation | Symbol | Bootstrap | Jormungandr | Spec |
|-----------|--------|-----------|-------------|------|
| Logic And | ∧ | ✅ | ✅ | ✅ |
| Logic Or | ∨ | ✅ | ✅ | ✅ |
| Logic Not | ¬ | ✅ | ✅ | ✅ |
| Logic Xor | ⊻ | ✅ | ✅ | ✅ |
| Top | ⊤ | ✅ | ✅ | ✅ |
| Bottom | ⊥ | ✅ | ✅ | ✅ |

### 3.5 Category Theory

| Operation | Symbol | Bootstrap | Jormungandr | Spec |
|-----------|--------|-----------|-------------|------|
| Compose | ∘ | ✅ | ✅ | ✅ |
| Tensor | ⊗ | ✅ | ✅ | ✅ |
| Direct Sum | ⊕ | ✅ | ✅ | ✅ |

---

## 4. Low-Level Features

### 4.1 Unsafe & FFI

| Feature | Bootstrap | Jormungandr | Spec |
|---------|-----------|-------------|------|
| `unsafe` blocks | ✅ | ✅ | ✅ |
| `unsafe fn` | ✅ | ✅ | ✅ |
| `extern "C"` | ✅ | ✅ | ✅ |
| Raw pointers `*const`/`*mut` | ✅ | ✅ | ✅ |
| Pointer dereference | ✅ | ✅ | ✅ |
| `addr_of!` / `addr_of_mut!` | ❓ | ❌ | ❓ |

### 4.2 Inline Assembly

| Feature | Bootstrap | Jormungandr | Spec |
|---------|-----------|-------------|------|
| `asm!()` expression | ✅ | ❌ | ❓ |
| Input operands `in()` | ✅ | ❌ | ❓ |
| Output operands `out()` | ✅ | ❌ | ❓ |
| In-out operands `inout()` | ✅ | ❌ | ❓ |
| Clobbers | ✅ | ❌ | ❓ |
| Options (volatile, nostack, etc.) | ✅ | ❌ | ❓ |
| AT&T/Intel dialect | ✅ | ❌ | ❓ |

**Note:** `asm!()` is fully implemented in the bootstrap compiler with LLVM backend support.

### 4.3 SIMD & Atomic

| Feature | Bootstrap | Jormungandr | Spec |
|---------|-----------|-------------|------|
| `simd<T, N>` types | ✅ | ❌ | ✅ |
| `atomic<T>` types | ✅ | ❌ | ✅ |
| `F32x16` (AVX-512) | ✅ | ❌ | ✅ |
| Volatile read/write | ✅ | ❌ | ✅ |

---

## 5. Standard Library

### 5.1 Core Types

| Type | Bootstrap | Jormungandr | Notes |
|------|-----------|-------------|-------|
| `Vec<T>` | ✅ | ✅ | Dynamic array |
| `String` | ✅ | ✅ | UTF-8 string |
| `HashMap<K,V>` | ✅ | ✅ | Hash map |
| `Option<T>` | ✅ | ✅ | Maybe type |
| `Result<T,E>` | ✅ | ✅ | Error handling |
| `Box<T>` | ✅ | ✅ | Heap allocation |
| `Rc<T>` | ✅ | ✅ | Reference counting |
| `Cell<T>` | ✅ | ✅ | Interior mutability |
| `Arc<T>` | ❓ | ❌ | Atomic Rc |

### 5.2 Traits

| Trait | Bootstrap | Jormungandr | Notes |
|-------|-----------|-------------|-------|
| `Drop` | ✅ | ✅ | Destructors |
| `Clone` | ✅ | ✅ | Deep copy |
| `Copy` | ✅ | ✅ | Bitwise copy |
| `Default` | ✅ | ✅ | Default values |
| `Debug` | ✅ | ✅ | Debug printing |
| `Display` | ✅ | ✅ | User display |
| `Iterator` | ✅ | ✅ | Iteration |
| `PartialEq`/`Eq` | ✅ | ✅ | Equality |
| `PartialOrd`/`Ord` | ✅ | ✅ | Ordering |

---

## 6. Critical Missing Features for Native Runtime

For implementing a pure Sigil native runtime (to replace `sigil_runtime.c`), the following features are **required**:

### 6.1 Already Implemented (Use These)

| Feature | Status | Notes |
|---------|--------|-------|
| `asm!()` inline assembly | ✅ Bootstrap | For syscalls |
| `unsafe` blocks | ✅ Both | For raw memory |
| Raw pointers | ✅ Both | For allocator |
| `extern "C"` | ✅ Both | For FFI |
| `*mut u8` / `*const u8` | ✅ Both | Memory ops |

### 6.2 Missing from Jormungandr (Needed)

| Feature | Priority | Workaround |
|---------|----------|------------|
| `asm!()` | HIGH | Use bootstrap compiler |
| `volatile` read/write | HIGH | Inline asm |
| SIMD types | MEDIUM | Scalar fallback |
| Atomic operations | MEDIUM | Single-threaded first |

### 6.3 Syntax Clarifications

**IMPORTANT:** The following are NOT valid Sigil syntax:

| Invalid | Correct | Notes |
|---------|---------|-------|
| `≔` | `let` | Use standard binding |
| `vary` | `let mut` | Use standard mutable |
| `⎇`/`⎉` | `if`/`else` | Use standard conditionals |
| `rite` | `fn` | Use standard function |
| `→` | `->` | Use ASCII arrow |

---

## 7. Action Items

### 7.1 Immediate (Native Runtime)

1. **Rewrite `parser/src/rt/*.sg` files** using correct syntax:
   - Replace `≔` with `let`
   - Replace `vary` with `let mut`
   - Replace `⎇`/`⎉` with `if`/`else`
   - Keep `asm!()` (it IS valid)

2. **Use bootstrap compiler** for native runtime development (has `asm!()` support)

### 7.2 Short Term

1. Add `asm!()` support to Jormungandr
2. Add volatile read/write to Jormungandr
3. Test syscall wrappers on Linux x86_64

### 7.3 Long Term

1. Port SIMD support to Jormungandr
2. Port atomic operations to Jormungandr
3. Full self-hosting with native runtime

---

## 8. Conclusion

Both compilers implement the core Sigil language very well. The bootstrap compiler has additional low-level features (`asm!()`, SIMD, atomics) that are essential for the native runtime.

**Recommendation:** Use the bootstrap compiler for native runtime development, then backport features to Jormungandr.
