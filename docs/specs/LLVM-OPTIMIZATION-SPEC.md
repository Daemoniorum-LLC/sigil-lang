# LLVM Backend Optimization Specification

**Version:** 0.1.0
**Status:** Draft
**Date:** 2026-02-11
**Parent Spec:** [18-COMPILER-ARCHITECTURE.md](./18-COMPILER-ARCHITECTURE.md)

---

## Abstract

This specification defines optimizations to bring Sigil's LLVM backend to performance
parity with native Rust. Benchmark analysis shows 14% overhead on numerical workloads,
attributed to float boxing, Vec layout overhead, and C runtime function calls.

---

## 1. Conceptual Foundation

### 1.1 Current State

The Sigil LLVM backend (llvm_codegen.rs) compiles Sigil IR to LLVM IR, then uses
LLVM's optimization passes and native code generation. Current benchmark results:

| Size | Rust (μs) | Sigil (μs) | Overhead |
|------|-----------|------------|----------|
| 16   | 2.20      | 2.44       | 1.11x    |
| 32   | 8.34      | 9.73       | 1.17x    |
| 64   | 34.37     | 38.89      | 1.13x    |
| 128  | 142.01    | 175.81     | 1.24x    |
| 256  | 627.76    | 681.04     | 1.08x    |
| 512  | 2385.65   | 2624.25    | 1.10x    |

**Average overhead: 14%** on naive O(n²) DCT algorithm.

### 1.2 Overhead Sources

| Source | Estimated Cost | Root Cause |
|--------|----------------|------------|
| Float boxing | ~5-8% | Values stored as i64 bit patterns, bitcast on every op |
| Vec offset | ~2-3% | Data inline at offset 2, extra arithmetic per access |
| Runtime calls | ~3-4% | Math via C runtime, not LLVM intrinsics |
| Missing SIMD | 0%* | No auto-vectorization due to above issues |

*SIMD would provide additional speedup if above fixed.

### 1.3 Target

**Goal:** Achieve <5% overhead vs Rust on numerical workloads.

**Non-goal:** Full SIMD vectorization (separate spec for F32x16).

---

## 2. Type Architecture

### 2.1 Current Float Representation

```
// Current: All values stored as i64
value: i64  // Float bits stored as integer

// On every float operation:
f64_val ← bitcast(i64_val, f64)
result_f64 ← fadd(f64_val, other_f64)
result_i64 ← bitcast(result_f64, i64)
store(result_i64)
```

**Problem:** 2 bitcasts per operation × 2 operands = 4 bitcasts minimum per binary op.

### 2.2 Proposed Float Representation

```
// Proposed: Type-aware storage
CompileScope {
    vars: HashMap<String, (PointerValue, LLVMType)>
    // LLVMType is i64 | f64 | ptr | struct
}

// Float variables use native f64:
%float_var = alloca double
store double %value, ptr %float_var
%loaded = load double, ptr %float_var
%result = fadd double %loaded, %other
```

**Invariant P1:** Variables declared with type `f64` use LLVM `double` storage.
**Invariant P2:** Integer-float conversion uses `sitofp`/`fptosi` at boundaries only.

### 2.3 Current Vec Layout

```
// Current: Inline data with offset calculation
Vec<T> = { len: i64, cap: i64, data: [T; cap] }

// Access v[i]:
ptr ← gep(vec_ptr, 0, i + 2)  // +2 to skip len, cap
```

**Problem:** Every access requires +2 arithmetic.

### 2.4 Proposed Vec Layout

```
// Option A: Pointer to data (matches Rust)
Vec<T> = { len: i64, cap: i64, data: *T }

// Access v[i]:
data_ptr ← load(gep(vec_ptr, 0, 2))
ptr ← gep(data_ptr, 0, i)  // No offset arithmetic

// Option B: Keep inline, cache data pointer
// Less invasive, single load at function entry
data_base ← gep(vec_ptr, 0, 2)
// Then: gep(data_base, 0, i) for each access
```

**Trade-off:**
- Option A: Matches Rust semantics, requires runtime changes
- Option B: Minimal change, still one extra GEP initially

### 2.5 Math Intrinsics

```
// Current: C runtime calls
call @sigil_cos(i64 %bits)  // Converts internally

// Proposed: LLVM intrinsics
%f = bitcast i64 %bits to double  // Once at boundary
%result = call double @llvm.cos.f64(double %f)
%out = bitcast double %result to i64  // Once at boundary
```

**Better with native floats:**
%result = call double @llvm.cos.f64(double %float_var)
// No bitcasts needed
```

---

## 3. Behavioral Contracts

### 3.1 Float Type Tracking

**Contract F1:** The compiler tracks type information through the IR.

```
compile_let_statement(name, type_annotation, initializer):
    if type_annotation = f64 or infer_type(initializer) = f64:
        alloca ← create_alloca(f64_type)
        scope.register(name, alloca, Type::Float)
    else:
        alloca ← create_alloca(i64_type)
        scope.register(name, alloca, Type::Integer)
```

**Contract F2:** Binary operations dispatch on operand types.

```
compile_binary_op(op, lhs, rhs):
    if type_of(lhs) = Float or type_of(rhs) = Float:
        compile_float_binary_op(op, lhs, rhs)
    else:
        compile_int_binary_op(op, lhs, rhs)
```

**Contract F3:** Boundary conversions are explicit.

```
// Integer to float (explicit cast in source)
x as f64  →  sitofp i64 %x to double

// Float to integer (explicit cast in source)
y as i64  →  fptosi double %y to i64
```

### 3.2 Vec Access

**Contract V1:** Vec data access uses cached base pointer.

```
compile_function_with_vec_param(vec_param):
    data_base ← gep(vec_ptr, 0, 2)
    scope.register_vec_base(vec_param, data_base)
    // All subsequent accesses use cached base
```

**Contract V2:** Vec indexing computes from cached base.

```
compile_index(vec_name, index):
    base ← scope.get_vec_base(vec_name)
    element_ptr ← gep(base, 0, index)  // No +2
    return load(element_ptr)
```

### 3.3 Math Intrinsics

**Contract M1:** Known math functions use LLVM intrinsics.

| Source | LLVM Intrinsic |
|--------|----------------|
| `x.sqrt()` | `@llvm.sqrt.f64` |
| `x.sin()` | `@llvm.sin.f64` |
| `x.cos()` | `@llvm.cos.f64` |
| `x.exp()` | `@llvm.exp.f64` |
| `x.log()` | `@llvm.log.f64` |
| `x.pow(y)` | `@llvm.pow.f64` |
| `x.floor()` | `@llvm.floor.f64` |
| `x.ceil()` | `@llvm.ceil.f64` |
| `x.abs()` | `@llvm.fabs.f64` |

**Contract M2:** Intrinsics operate on native types.

```
compile_method_call(receiver, "sqrt"):
    if type_of(receiver) = Float:
        result ← call @llvm.sqrt.f64(receiver)
    else:
        // Convert, call, convert back (fallback)
        f ← sitofp(receiver)
        result_f ← call @llvm.sqrt.f64(f)
        result ← fptosi(result_f)
```

---

## 4. Constraints & Invariants

### 4.1 Type System Invariants

```
P1: ∀ var with type f64:
    llvm_type(var) = double

P2: ∀ binary_op on f64:
    uses fadd | fsub | fmul | fdiv | fcmp
    // Never uses add | sub | mul | sdiv | icmp

P3: ∀ float_literal:
    stored as double, not i64 bit pattern
```

### 4.2 Performance Invariants

```
P4: ∀ float binary op:
    bitcast_count ≤ 0  // After optimization
    // Compare to current: bitcast_count = 4

P5: ∀ Vec access v[i]:
    gep_offset_arithmetic = 0  // Uses cached base
    // Compare to current: +2 per access

P6: ∀ intrinsic-supported math:
    call_target = @llvm.*  // Not @sigil_*
```

### 4.3 Compatibility Invariants

```
P7: ∀ existing test:
    behavior_unchanged

P8: ∀ FFI boundary:
    i64 representation preserved  // For C interop
```

---

## 5. Implementation Phases

### Phase 1: Type-Aware Compilation (HIGH PRIORITY)

**Status:** ❌ Not implemented

**Changes:**
1. Add `LLVMType` enum to `CompileScope` variable tracking
2. Modify `compile_let_stmt` to select type based on annotation/inference
3. Modify `compile_binary_op` to dispatch on type
4. Modify `compile_literal` to emit native f64 for float literals

**Expected improvement:** 5-8%

**Test criteria:**
- Float operations use no bitcasts in generated IR
- Existing tests pass unchanged
- DCT benchmark shows <10% overhead

### Phase 2: Vec Base Caching (MEDIUM PRIORITY)

**Status:** ❌ Not implemented

**Changes:**
1. Add `vec_bases: HashMap<String, PointerValue>` to `CompileScope`
2. At function entry, compute base pointers for Vec parameters
3. Modify `compile_index` to use cached base

**Expected improvement:** 2-3%

**Test criteria:**
- Vec access GEP instructions have no +2 offset
- Vec mutation tests pass
- Memory correctness verified

### Phase 3: LLVM Intrinsics (MEDIUM PRIORITY)

**Status:** ❌ Not implemented

**Changes:**
1. Add intrinsic declarations to module setup
2. Modify math method compilation to use intrinsics
3. Handle type conversion at boundaries

**Expected improvement:** 3-4%

**Test criteria:**
- Math operations call `@llvm.*` not `@sigil_*`
- PI() still works (constant, not intrinsic)
- Numerical accuracy unchanged

### Phase 4: Constant Folding (LOW PRIORITY)

**Status:** ❌ Not implemented

**Changes:**
1. Fold constant expressions at compile time
2. Propagate known values through control flow
3. Let LLVM handle complex cases

**Expected improvement:** Variable

---

## 6. Open Questions

### Q1: Type Inference Scope

How deep should type inference go for floats?

```sigil
≔ x = 1.0;       // Obviously f64
≔ y = x + 2.0;   // Infer f64 from x
≔ z = foo(x);    // Infer from return type?
```

**Options:**
- A: Local inference only (explicit types for function returns)
- B: Full inference (requires type checker integration)

### Q2: Mixed Operations

What happens with `i64 + f64`?

```sigil
≔ x = 5;       // i64
≔ y = 3.14;    // f64
≔ z = x + y;   // What type?
```

**Options:**
- A: Compilation error (require explicit cast)
- B: Implicit promotion to f64 (like C)

### Q3: Vec Layout Migration

Should we change Vec layout for pointer-based data?

**Options:**
- A: Yes - matches Rust, enables better optimization
- B: No - keep inline, just cache base pointers
- C: Make it configurable

### Q4: FFI Boundaries

How do native f64 values cross FFI boundaries?

**Options:**
- A: Convert to i64 bits at boundary (compatibility)
- B: Pass as f64 directly (requires runtime changes)
- C: Generate both calling conventions

---

## 7. Verification

### 7.1 Benchmark Suite

```bash
# Run DCT benchmark comparison
cd parser
./target/release/sigil compile /tmp/bench_dct.sg -o /tmp/bench_dct
/tmp/bench_dct

# Compare against Rust baseline
/home/crook/dev/haagenti/sigil/benchmarks/bench_dct_rust
```

### 7.2 IR Inspection

After each phase, verify generated IR:

```bash
# Emit LLVM IR (when implemented)
./target/release/sigil compile --emit-llvm program.sg

# Check for:
# - No unnecessary bitcasts in float code
# - No +2 offsets in Vec access
# - @llvm.* intrinsics for math
```

### 7.3 Test Suite

```bash
cd jormungandr/tests
./run_tests_rust.sh

# All 745 tests must pass
# No regressions allowed
```

---

## 8. Revision History

| Version | Date | Changes |
|---------|------|---------|
| 0.1.0 | 2026-02-11 | Initial draft based on benchmark analysis |

---

## References

- [LLVM Language Reference - Float Types](https://llvm.org/docs/LangRef.html#floating-point-types)
- [LLVM Intrinsic Functions](https://llvm.org/docs/LangRef.html#intrinsic-functions)
- Sigil LESSONS-LEARNED.md - Float operations, Vec layout discoveries
