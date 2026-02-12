# LLVM Optimization TDD Roadmap

**Version:** 0.1.0
**Status:** Draft
**Date:** 2026-02-11
**Parent Spec:** [LLVM-OPTIMIZATION-SPEC.md](./LLVM-OPTIMIZATION-SPEC.md)

---

## Overview

This roadmap defines test-first implementation of LLVM optimizations. Each phase
has specification tests that define required behavior, followed by implementation.

**Methodology:** Agent-TDD integrated with SDD

---

## Phase 1: Type-Aware Float Compilation

### 1.1 Specification Tests

```sigil
// tests/llvm/float_type_tracking.sg

scroll tests {
    //@ spec: Float literals use native double storage
    rite test_float_literal_type() {
        ≔ x = 3.14;
        // IR should show: %x = alloca double
        // NOT: %x = alloca i64
        assert_eq(x, 3.14);
    }

    //@ spec: Float binary ops use fadd/fmul
    rite test_float_addition() {
        ≔ a = 1.5;
        ≔ b = 2.5;
        ≔ c = a + b;
        // IR should show: %c = fadd double %a, %b
        // NOT: bitcast + add + bitcast
        assert_eq(c, 4.0);
    }

    //@ spec: Float comparison uses fcmp
    rite test_float_comparison() {
        ≔ x = 3.14;
        ≔ y = 2.71;
        assert(x > y);
        // IR should show: fcmp ogt double %x, %y
    }

    //@ spec: Mixed int-float requires explicit cast
    rite test_int_to_float_cast() {
        ≔ i = 42;
        ≔ f = i as f64;
        // IR should show: sitofp i64 %i to double
        assert_eq(f, 42.0);
    }

    //@ spec: Float to int truncates
    rite test_float_to_int_cast() {
        ≔ f = 3.7;
        ≔ i = f as i64;
        // IR should show: fptosi double %f to i64
        assert_eq(i, 3);
    }
}
```

### 1.2 Implementation Checklist

- [ ] Add `ValueType` enum: `Integer | Float | Pointer | Struct`
- [ ] Extend `CompileScope.vars` to track `(PointerValue, ValueType)`
- [ ] Modify `compile_literal` for f64 literals → native double
- [ ] Modify `compile_let_stmt` to infer and track type
- [ ] Modify `compile_binary_op` to dispatch on type
- [ ] Modify `compile_comparison` for float comparisons
- [ ] Add cast compilation (`as f64`, `as i64`)

### 1.3 Acceptance Criteria

- [ ] All Phase 1 spec tests pass
- [ ] Generated IR shows `double` type for float variables
- [ ] No `bitcast` between float operations
- [ ] DCT benchmark shows improvement

---

## Phase 2: Vec Base Caching

### 2.1 Specification Tests

```sigil
// tests/llvm/vec_base_caching.sg

scroll tests {
    //@ spec: Vec access uses cached base
    rite test_vec_single_access() {
        ≔ Δ v = Vec·with_capacity(10);
        v.push(42);
        ≔ x = v[0];
        // IR should compute base once, not per-access
        assert_eq(x, 42);
    }

    //@ spec: Multiple accesses reuse base
    rite test_vec_multiple_access() {
        ≔ Δ v = Vec·with_capacity(3);
        v.push(1);
        v.push(2);
        v.push(3);
        ≔ sum = v[0] + v[1] + v[2];
        // IR should have single base computation
        assert_eq(sum, 6);
    }

    //@ spec: Vec write uses cached base
    rite test_vec_write() {
        ≔ Δ v = Vec·with_capacity(3);
        v.push(0);
        v.push(0);
        v.push(0);
        v[1] = 99;
        assert_eq(v[1], 99);
    }

    //@ spec: Nested loops with Vec access
    rite test_vec_loop_access() {
        ≔ Δ v = Vec·with_capacity(100);
        ≔ Δ i = 0;
        ⟳ i < 100 {
            v.push(i);
            i = i + 1;
        }
        
        ≔ Δ sum = 0;
        i = 0;
        ⟳ i < 100 {
            sum = sum + v[i];
            i = i + 1;
        }
        // Base should be computed once before loop
        assert_eq(sum, 4950);
    }
}
```

### 2.2 Implementation Checklist

- [ ] Add `vec_bases: HashMap<String, PointerValue>` to `CompileScope`
- [ ] At function entry, scan for Vec parameters
- [ ] Compute and cache base pointer (`gep(vec, 0, 2)`)
- [ ] Modify `compile_index` to use `scope.get_vec_base()`
- [ ] Modify `compile_index_assign` similarly
- [ ] Handle Vec created within function

### 2.3 Acceptance Criteria

- [ ] All Phase 2 spec tests pass
- [ ] IR shows single base computation per Vec
- [ ] No `+2` offset in index GEP instructions
- [ ] Vec mutation still works correctly

---

## Phase 3: LLVM Math Intrinsics

### 3.1 Specification Tests

```sigil
// tests/llvm/math_intrinsics.sg

scroll tests {
    //@ spec: sqrt uses llvm.sqrt.f64
    rite test_sqrt_intrinsic() {
        ≔ x = 16.0;
        ≔ y = x.sqrt();
        // IR should show: call double @llvm.sqrt.f64(double %x)
        assert_eq(y, 4.0);
    }

    //@ spec: sin uses llvm.sin.f64
    rite test_sin_intrinsic() {
        ≔ x = 0.0;
        ≔ y = sin(x);
        assert_eq(y, 0.0);
    }

    //@ spec: cos uses llvm.cos.f64
    rite test_cos_intrinsic() {
        ≔ x = 0.0;
        ≔ y = cos(x);
        assert_eq(y, 1.0);
    }

    //@ spec: exp uses llvm.exp.f64
    rite test_exp_intrinsic() {
        ≔ x = 0.0;
        ≔ y = x.exp();
        assert_eq(y, 1.0);
    }

    //@ spec: floor uses llvm.floor.f64
    rite test_floor_intrinsic() {
        ≔ x = 3.7;
        ≔ y = x.floor();
        assert_eq(y, 3.0);
    }

    //@ spec: abs uses llvm.fabs.f64
    rite test_fabs_intrinsic() {
        ≔ x = -5.0;
        ≔ y = x.abs();
        assert_eq(y, 5.0);
    }

    //@ spec: PI is compile-time constant
    rite test_pi_constant() {
        ≔ pi = PI();
        // Should inline as constant, not runtime call
        assert(pi > 3.14);
        assert(pi < 3.15);
    }
}
```

### 3.2 Implementation Checklist

- [ ] Add intrinsic declarations to module initialization
- [ ] Create `declare_math_intrinsics()` function
- [ ] Map method names to intrinsic names
- [ ] Modify `compile_method_call` for math methods
- [ ] Handle `sin(x)` function syntax
- [ ] Special case `PI()` as constant

### 3.3 Acceptance Criteria

- [ ] All Phase 3 spec tests pass
- [ ] IR shows `@llvm.*` calls, not `@sigil_*`
- [ ] Numerical accuracy matches C runtime
- [ ] DCT benchmark matches Rust within 5%

---

## Phase 4: Integration Testing

### 4.1 DCT Benchmark Validation

```sigil
// benchmarks/dct_comprehensive.sg

rite dct_1d(input: Vec<f64>, n: i64) → Vec<f64> {
    ≔ pi = PI();
    ≔ scale = (2.0 / (n as f64)).sqrt();
    ≔ sqrt2 = 2.0_f64.sqrt();
    
    ≔ Δ output = Vec·with_capacity(n);
    ≔ Δ i = 0;
    ⟳ i < n {
        output.push(0.0);
        i = i + 1;
    }
    
    ≔ Δ k = 0;
    ⟳ k < n {
        ≔ Δ sum = 0.0_f64;
        ≔ Δ j = 0;
        ⟳ j < n {
            ≔ angle = pi * (k as f64) * ((j as f64) + 0.5) / (n as f64);
            sum = sum + input[j] * cos(angle);
            j = j + 1;
        }
        output[k] = sum * scale;
        k = k + 1;
    }
    
    output[0] = output[0] / sqrt2;
    output
}

//@ benchmark: DCT performance within 5% of Rust
rite benchmark_dct() {
    ≔ sizes = [16, 32, 64, 128, 256, 512];
    ≔ rust_baseline = [2.20, 8.34, 34.37, 142.01, 627.76, 2385.65];
    
    // Each size should be within 5% of Rust
    // Implementation: compare timings
}
```

### 4.2 Regression Test Suite

- [ ] Run full test suite: `./run_tests_rust.sh`
- [ ] Verify 745/749 tests pass (same as before)
- [ ] No numerical accuracy regressions
- [ ] Memory safety preserved

---

## Execution Order

1. **Phase 1** (Type-Aware Floats) - Highest impact
2. **Phase 3** (Math Intrinsics) - Pairs well with Phase 1
3. **Phase 2** (Vec Caching) - Independent optimization
4. **Phase 4** (Integration) - Validates all phases

---

## Success Metrics

| Metric | Baseline | Target | Achieved |
|--------|----------|--------|----------|
| DCT-16 overhead | 1.11x | <1.05x | ❌ |
| DCT-64 overhead | 1.13x | <1.05x | ❌ |
| DCT-256 overhead | 1.08x | <1.05x | ❌ |
| Test pass rate | 745/749 | 745/749 | ❌ |
| Bitcasts per float op | 4 | 0 | ❌ |
| Vec GEP offset adds | 1/access | 0 | ❌ |
| Runtime math calls | 100% | 0% | ❌ |

---

## Revision History

| Version | Date | Changes |
|---------|------|---------|
| 0.1.0 | 2026-02-11 | Initial roadmap |
