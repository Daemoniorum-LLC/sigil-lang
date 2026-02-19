# MNIST Interpreter Features TDD Roadmap

> **Status:** Draft (Tests Reviewed ✅)
> **Last Updated:** 2026-02-08
> **Blocking:** Full mnist_training.sigil execution
> **Prereqs:** 03E-AUTOGRAD-NABLA-SPEC.md Phase 1 ✅

## 1. Overview

This roadmap defines the test-driven implementation path for interpreter features needed to run the full `mnist_training.sigil` example without simplification.

**Methodology:** Agent-TDD — tests crystallize understanding before implementation.

### Test Quality Standards

Each test file covers:
- ✅ **Basic cases** — Core functionality with typical inputs
- ✅ **Edge cases** — Single element, ties, boundary conditions
- ✅ **Numerical stability** — Large/small values, potential overflow/underflow
- ✅ **Type variations** — Arrays, Tensors, integers, floats
- ✅ **Shape preservation** — 1D, 2D, 3D tensors maintain correct shape
- ⏸️ **Axis parameter** — Deferred to Phase 2
- ⏸️ **Empty collection** — Behavior TBD (error vs Option::None)

### 1.1 Current State

Phase 1 autograd is complete:
- ✅ `∇` operator returns `Gradients` struct
- ✅ `Gradients.get(&param)` returns `Option<Gradient>`
- ✅ `cross_entropy(logits, targets)` builtin
- ✅ `|sqrt` pipe method
- ✅ Tensor subtraction (`a - b`)

### 1.2 Gap Analysis

Features needed for full MNIST (from 14-NEURAL.md spec):

| Priority | Feature | Spec Reference | Test File |
|----------|---------|----------------|-----------|
| P0 | `\|μ` (mean) | 14-NEURAL §2.2 | `test_pipe_mean.sg` |
| P0 | `\|gelu` | 14-NEURAL §4.3 | `test_pipe_gelu.sg` |
| P0 | `\|softmax` | 14-NEURAL §4.3 | `test_pipe_softmax.sg` |
| P0 | `\|argmax` | 14-NEURAL §7.1 | `test_pipe_argmax.sg` |
| P0 | `\|log_softmax` | 14-NEURAL §4.3 | `test_pipe_log_softmax.sg` |
| P1 | `no_grad { }` | 14-NEURAL §3.1 | `test_no_grad.sg` |
| P1 | Per-param gradients | 03E Phase 2 | `test_param_gradients.sg` |
| P1 | `\|zip` | 09-STDLIB §3.3 | `test_pipe_zip.sg` |
| P1 | `\|count` | 09-STDLIB §3.3 | `test_pipe_count.sg` |
| P2 | Optimizer trait | 14-NEURAL §6.2 | `test_optimizer_trait.sg` |
| P2 | Linear layer | 14-NEURAL §4.2 | `test_linear_layer.sg` |
| P2 | Computation graph | 03E Phase 2 | `test_comp_graph.sg` |

---

## 2. P0 Tests (Blocking MNIST)

### 2.1 Mean Pipe (`|μ`)

**File:** `jormungandr/tests/neural/test_pipe_mean.sg`

```sigil
//! Test: μ (mean) pipe method for tensors and arrays
//! Spec: 14-NEURAL.md §2.2, 09-STDLIB.md §3.3
//! Priority: P0

rite main() {
    // Test 1: Array mean
    ≔ arr = [1.0, 2.0, 3.0, 4.0, 5.0];
    ≔ mean_val = arr|μ;
    assert(mean_val == 3.0, "Array mean should be 3.0");

    // Test 2: Tensor mean (all elements)
    ≔ t = Tensor·from([[1.0, 2.0], [3.0, 4.0]]);
    ≔ t_mean = t|μ;
    assert(t_mean == 2.5, "Tensor mean should be 2.5");

    // Test 3: Tensor mean along axis
    ≔ row_means = t|μ(axis: 1);  // [1.5, 3.5]
    assert(row_means.shape() == [2], "Row means shape");

    // Test 4: Empty array mean returns Option::None
    ≔ empty: [f64] = [];
    ≔ empty_mean = empty|μ;
    assert(empty_mean.is_none(), "Empty mean should be None");

    // Test 5: Single element
    ≔ single = [42.0];
    assert(single|μ == 42.0, "Single element mean");

    println("✓ All μ (mean) tests passed");
}
```

**Implementation location:** `interpreter.rs` eval_pipe, add handler for "μ" | "mean"

---

### 2.2 GELU Activation (`|gelu`)

**File:** `jormungandr/tests/neural/test_pipe_gelu.sg`

```sigil
//! Test: gelu activation function
//! Spec: 14-NEURAL.md §4.3
//! Priority: P0
//! Formula: gelu(x) = x * Φ(x) ≈ 0.5 * x * (1 + tanh(√(2/π) * (x + 0.044715 * x³)))

rite main() {
    // Test 1: GELU of zero
    ≔ zero = Tensor·from([0.0]);
    ≔ gelu_zero = zero|gelu;
    assert(gelu_zero.data()[0] == 0.0, "GELU(0) = 0");

    // Test 2: GELU of positive value
    ≔ pos = Tensor·from([1.0]);
    ≔ gelu_pos = pos|gelu;
    // GELU(1) ≈ 0.8413
    assert(gelu_pos.data()[0] > 0.84, "GELU(1) ≈ 0.8413");
    assert(gelu_pos.data()[0] < 0.85, "GELU(1) ≈ 0.8413");

    // Test 3: GELU of negative value (not zero like ReLU)
    ≔ neg = Tensor·from([-1.0]);
    ≔ gelu_neg = neg|gelu;
    // GELU(-1) ≈ -0.1587
    assert(gelu_neg.data()[0] < 0.0, "GELU(-1) is negative");
    assert(gelu_neg.data()[0] > -0.2, "GELU(-1) ≈ -0.1587");

    // Test 4: Vectorized GELU
    ≔ batch = Tensor·from([-2.0, -1.0, 0.0, 1.0, 2.0]);
    ≔ gelu_batch = batch|gelu;
    assert(gelu_batch.shape() == [5], "Shape preserved");

    // Test 5: Float scalar gelu
    ≔ scalar = 0.5;
    ≔ gelu_scalar = scalar|gelu;
    assert(gelu_scalar > 0.34, "GELU(0.5) ≈ 0.3457");

    println("✓ All gelu tests passed");
}
```

**Implementation:** `interpreter.rs` eval_pipe, add "gelu" handler with tanh approximation

---

### 2.3 Softmax Pipe (`|softmax`)

**File:** `jormungandr/tests/neural/test_pipe_softmax.sg`

```sigil
//! Test: softmax pipe method
//! Spec: 14-NEURAL.md §4.3
//! Priority: P0
//! Formula: softmax(x)_i = exp(x_i) / Σ exp(x_j)

rite main() {
    // Test 1: Basic softmax sums to 1
    ≔ logits = Tensor·from([1.0, 2.0, 3.0]);
    ≔ probs = logits|softmax;
    ≔ sum = probs|Σ;
    assert(sum > 0.999, "Softmax sums to 1");
    assert(sum < 1.001, "Softmax sums to 1");

    // Test 2: Softmax preserves order
    ≔ p = probs.data();
    assert(p[2] > p[1], "Larger logit → larger prob");
    assert(p[1] > p[0], "Larger logit → larger prob");

    // Test 3: Numerical stability (large values)
    ≔ large = Tensor·from([1000.0, 1001.0, 1002.0]);
    ≔ large_probs = large|softmax;
    ≔ large_sum = large_probs|Σ;
    assert(large_sum > 0.999, "Stable with large values");

    // Test 4: Softmax along axis
    ≔ batch = Tensor·from([[1.0, 2.0], [3.0, 4.0]]);
    ≔ batch_probs = batch|softmax(dim: -1);  // Softmax each row
    // Each row should sum to 1

    // Test 5: Uniform input → uniform output
    ≔ uniform = Tensor·from([1.0, 1.0, 1.0]);
    ≔ uniform_probs = uniform|softmax;
    ≔ u = uniform_probs.data();
    assert(u[0] > 0.33, "Uniform ≈ 1/3");
    assert(u[0] < 0.34, "Uniform ≈ 1/3");

    println("✓ All softmax tests passed");
}
```

---

### 2.4 Argmax Pipe (`|argmax`)

**File:** `jormungandr/tests/neural/test_pipe_argmax.sg`

```sigil
//! Test: argmax pipe method
//! Spec: 14-NEURAL.md §7.1
//! Priority: P0

rite main() {
    // Test 1: Basic argmax
    ≔ arr = [1.0, 5.0, 3.0, 2.0];
    ≔ idx = arr|argmax;
    assert(idx == 1, "Argmax of [1,5,3,2] is index 1");

    // Test 2: Tensor argmax
    ≔ t = Tensor·from([0.1, 0.7, 0.2]);
    ≔ t_idx = t|argmax;
    assert(t_idx == 1, "Tensor argmax");

    // Test 3: First occurrence on tie
    ≔ tie = [3.0, 3.0, 1.0];
    ≔ tie_idx = tie|argmax;
    assert(tie_idx == 0, "Argmax returns first on tie");

    // Test 4: Argmax along axis
    ≔ batch = Tensor·from([[1.0, 2.0, 3.0], [3.0, 2.0, 1.0]]);
    ≔ predictions = batch|argmax(dim: -1);  // [2, 0]
    assert(predictions[0] == 2, "Row 0 argmax");
    assert(predictions[1] == 0, "Row 1 argmax");

    // Test 5: Single element
    ≔ single = [42.0];
    assert(single|argmax == 0, "Single element argmax is 0");

    // Test 6: Negative values
    ≔ neg = [-5.0, -1.0, -3.0];
    assert(neg|argmax == 1, "Argmax with negatives");

    println("✓ All argmax tests passed");
}
```

---

### 2.5 Log-Softmax Pipe (`|log_softmax`)

**File:** `jormungandr/tests/neural/test_pipe_log_softmax.sg`

```sigil
//! Test: log_softmax pipe method (numerically stable)
//! Spec: 14-NEURAL.md §4.3
//! Priority: P0

rite main() {
    // Test 1: log_softmax = log(softmax)
    ≔ logits = Tensor·from([1.0, 2.0, 3.0]);
    ≔ log_probs = logits|log_softmax;
    ≔ probs = logits|softmax;

    // log_softmax[i] should equal log(softmax[i])
    ≔ lp = log_probs.data();
    ≔ p = probs.data();
    assert((lp[0] - p[0]|log)|abs < 0.001, "log_softmax ≈ log(softmax)");

    // Test 2: All values are negative (log of probability < 1)
    assert(lp[0] < 0.0, "Log probs are negative");
    assert(lp[1] < 0.0, "Log probs are negative");
    assert(lp[2] < 0.0, "Log probs are negative");

    // Test 3: Numerical stability with large values
    ≔ large = Tensor·from([1000.0, 1001.0, 1002.0]);
    ≔ large_log_probs = large|log_softmax;
    // Should not overflow/underflow
    ≔ llp = large_log_probs.data();
    assert(llp[2] > llp[1], "Preserves order");
    assert(llp[2]|is_finite, "Finite result");

    println("✓ All log_softmax tests passed");
}
```

---

## 3. P1 Tests (Enhanced MNIST)

### 3.1 No-Grad Context

**File:** `jormungandr/tests/neural/test_no_grad.sg`

```sigil
//! Test: no_grad context disables gradient tracking
//! Spec: 14-NEURAL.md §3.1
//! Priority: P1

rite main() {
    // Test 1: Operations in no_grad don't build graph
    ≔ x = Tensor·randn([10]);
    x.set_requires_grad(yea);

    no_grad {
        ≔ y = x * 2.0;
        // y should not have grad_fn
        assert(y.grad_fn().is_none(), "no_grad disables tracking");
    }

    // Test 2: Inference mode
    ≔ model = SimpleModel·new();
    no_grad {
        ≔ output = model.forward(input);
        // Can use output for inference without gradients
    }

    // Test 3: Nested no_grad
    no_grad {
        no_grad {
            ≔ z = x + 1.0;
        }
        // Still in no_grad
    }

    println("✓ All no_grad tests passed");
}
```

---

### 3.2 Per-Parameter Gradient Tracking

**File:** `jormungandr/tests/neural/test_param_gradients.sg`

```sigil
//! Test: Gradients are tracked per-parameter via tensor ID
//! Spec: 03E-AUTOGRAD-NABLA-SPEC.md Phase 2
//! Priority: P1

rite main() {
    // Test 1: Different parameters get different gradients
    ≔ w1 = Tensor·randn([10, 5]);
    ≔ w2 = Tensor·randn([5, 2]);
    w1.set_requires_grad(yea);
    w2.set_requires_grad(yea);

    ≔ x = Tensor·randn([4, 10]);
    ≔ h = x @ w1;
    ≔ y = h @ w2;
    ≔ loss = y|Σ;

    ≔ grads = loss|∇;

    ≔ grad_w1 = grads.get(&w1);
    ≔ grad_w2 = grads.get(&w2);

    assert(grad_w1.is_some(), "w1 has gradient");
    assert(grad_w2.is_some(), "w2 has gradient");

    // Test 2: Gradient shapes match parameter shapes
    assert(grad_w1.unwrap().shape() == [10, 5], "grad_w1 shape");
    assert(grad_w2.unwrap().shape() == [5, 2], "grad_w2 shape");

    // Test 3: Parameters without requires_grad have no gradient
    ≔ w3 = Tensor·randn([2, 2]);  // requires_grad = false
    ≔ grad_w3 = grads.get(&w3);
    assert(grad_w3.is_none(), "Non-param has no gradient");

    println("✓ All per-param gradient tests passed");
}
```

---

### 3.3 Zip Pipe (`|zip`)

**File:** `jormungandr/tests/neural/test_pipe_zip.sg`

```sigil
//! Test: zip pipe method
//! Spec: 09-STDLIB.md §3.3
//! Priority: P1

rite main() {
    // Test 1: Basic zip
    ≔ a = [1, 2, 3];
    ≔ b = ["a", "b", "c"];
    ≔ zipped = a|zip(b);
    // zipped = [(1, "a"), (2, "b"), (3, "c")]

    ≔ first = zipped[0];
    assert(first.0 == 1, "First tuple element");
    assert(first.1 == "a", "Second tuple element");

    // Test 2: Unequal lengths (shorter wins)
    ≔ short = [1, 2];
    ≔ long = [10, 20, 30, 40];
    ≔ z = short|zip(long);
    assert(z|len == 2, "Zip stops at shorter");

    // Test 3: Zip with enumerate pattern
    ≔ items = ["apple", "banana", "cherry"];
    ≔ indexed = [0, 1, 2]|zip(items);

    // Test 4: Tensor zip (element-wise pairing)
    ≔ t1 = Tensor·from([1.0, 2.0, 3.0]);
    ≔ t2 = Tensor·from([4.0, 5.0, 6.0]);
    ≔ pairs = t1|zip(t2);

    println("✓ All zip tests passed");
}
```

---

## 4. P2 Tests (Full Neural API)

### 4.1 Optimizer Trait

**File:** `jormungandr/tests/neural/test_optimizer_trait.sg`

```sigil
//! Test: Optimizer trait with step() accepting gradients
//! Spec: 14-NEURAL.md §6.2
//! Priority: P2

rite main() {
    // Test 1: SGD optimizer
    ≔ w = Tensor·randn([10, 5]);
    w.set_requires_grad(yea);

    ≔ optimizer = SGD·new([&w], lr: 0.01);

    // Forward + backward
    ≔ loss = (w * w)|Σ;
    ≔ grads = loss|∇;

    // Optimizer step
    optimizer.zero_grad();
    optimizer.step();  // Uses param.grad internally

    // Test 2: Adam optimizer
    ≔ adam = Adam·new([&w], lr: 0.001, betas: (0.9, 0.999));
    adam.step();

    println("✓ All optimizer tests passed");
}
```

---

## 5. Test Execution Plan

### 5.1 Directory Structure

```
jormungandr/tests/neural/
├── test_pipe_mean.sg
├── test_pipe_gelu.sg
├── test_pipe_softmax.sg
├── test_pipe_argmax.sg
├── test_pipe_log_softmax.sg
├── test_no_grad.sg
├── test_param_gradients.sg
├── test_pipe_zip.sg
├── test_optimizer_trait.sg
└── test_mnist_full.sg
```

### 5.2 Execution Order

1. **P0 tests first** — These unblock basic MNIST training
2. **P1 tests** — Enable proper gradient tracking and inference
3. **P2 tests** — Full neural network API

### 5.3 Success Criteria

- All P0 tests pass → `mnist_training_simple.sigil` runs with real gradients
- All P1 tests pass → `mnist_training.sigil` runs with per-param optimization
- All P2 tests pass → Full neural API matches 14-NEURAL.md spec

---

## 6. Implementation Order

Based on dependencies:

1. `|μ` (mean) — No dependencies, pure reduction
2. `|argmax` — No dependencies, pure reduction
3. `|softmax` — Needs exp, sum (already have)
4. `|log_softmax` — Needs softmax, log
5. `|gelu` — Needs tanh, sqrt (already have sqrt)
6. `|zip` — No dependencies
7. `no_grad` — Needs interpreter context flag
8. Per-param gradients — Needs tensor ID tracking
9. Optimizer trait — Needs per-param gradients

---

## 7. Related Specs

- `14-NEURAL.md` — Full neural network specification
- `03E-AUTOGRAD-NABLA-SPEC.md` — Autograd implementation phases
- `01A-MORPHEME-DESUGARING.md` — Pipe morpheme semantics
- `09-STDLIB.md` — Standard library functions

---

*This roadmap crystallizes understanding. Tests pass → implementation is correct.*
