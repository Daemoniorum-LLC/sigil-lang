# Autograd and Nabla (∇) Operator Specification

> **Status:** Draft
> **Last Updated:** 2026-02-08
> **Blocking Issue:** mnist_training.sigil fails with "Unknown pipe method: ∇"

## 1. Overview

This spec defines how automatic differentiation (autograd) and the nabla (∇) operator should work in the Sigil interpreter to enable backpropagation for neural network training.

**Motivation:** The mnist_training.sigil example uses `loss|∇` to compute gradients for training. This requires:
1. Recording a computation graph during forward pass
2. Implementing the ∇ pipe operator to trigger backward pass
3. Propagating gradients through the chain rule

---

## 2. Architecture

### 2.1 Core Components

```
┌─────────────────────────────────────────────────────────────┐
│                    Forward Pass                             │
│  input → Linear → ReLU → Linear → Softmax → CrossEntropy   │
│     │        │       │        │         │          │        │
│     ▼        ▼       ▼        ▼         ▼          ▼        │
│  [Record computation graph: each op saves grad_fn]          │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼ loss|∇
┌─────────────────────────────────────────────────────────────┐
│                    Backward Pass                            │
│  CrossEntropy ← Softmax ← Linear ← ReLU ← Linear ← input   │
│     ∂L/∂y         ← ← ← Chain Rule → → →      ∂L/∂x        │
└─────────────────────────────────────────────────────────────┘
```

### 2.2 Components in Interpreter

1. **GradientTape** - Records operations during forward pass
2. **GradFn** - Backward function for each differentiable operation
3. **Tensor.grad_fn** - Optional reference to gradient function
4. **∇ operator** - Triggers backward pass, returns gradients

---

## 3. Value Representation

### 3.1 Tensor with Gradient Tracking

```rust
// Current Tensor representation
Value::Struct {
    name: "Tensor".to_string(),
    fields: {
        "data": Value::Array(...),
        "shape": Value::Array(...),
        "requires_grad": Value::Bool(true/false),
        // NEW: gradient function for backward pass
        "__grad_fn__": Value::Null | Value::BuiltIn(grad_fn),
        // NEW: accumulated gradient (set during backward)
        "__grad__": Value::Null | Value::Struct { name: "Tensor", ... },
    },
}
```

### 3.2 Gradients Container

```rust
// Gradients! type returned by ∇ operator
Value::Struct {
    name: "Gradients".to_string(),
    fields: {
        // Maps tensor_id → gradient tensor
        "__grads__": Value::Map(HashMap<String, Value::Struct>),
    },
}
```

---

## 4. Gradient Functions

### 4.1 GradFn Structure

Each differentiable operation creates a GradFn that:
1. Stores references to input tensors (for chain rule)
2. Implements `backward(grad_output) → Vec<grad_input>`

```rust
struct GradFn {
    name: String,           // "MatMul", "ReLU", "Softmax", etc.
    inputs: Vec<TensorId>,  // References to input tensors
    backward: Fn(&Value, &Interpreter) -> Vec<Value>,  // Backward function
}
```

### 4.2 Operation Gradients

| Operation | Forward | Backward (∂L/∂x given ∂L/∂y) |
|-----------|---------|------------------------------|
| MatMul (x @ w) | y = x @ w | ∂L/∂x = ∂L/∂y @ w.T, ∂L/∂w = x.T @ ∂L/∂y |
| ReLU | y = max(0, x) | ∂L/∂x = ∂L/∂y * (x > 0) |
| Add | y = a + b | ∂L/∂a = ∂L/∂y, ∂L/∂b = ∂L/∂y |
| Mul | y = a * b | ∂L/∂a = ∂L/∂y * b, ∂L/∂b = ∂L/∂y * a |
| Softmax | y = softmax(x) | ∂L/∂x = y * (∂L/∂y - (∂L/∂y · y)) |
| CrossEntropy | L = CE(y, t) | ∂L/∂y = y - t (when combined with softmax) |
| Sum | y = Σx | ∂L/∂x = broadcast(∂L/∂y) |

---

## 5. The ∇ Operator

### 5.1 Pipe Syntax

```sigil
≔ grads! = loss|∇;           // Backprop from scalar loss
≔ grads! = output|∇(v);      // VJP with custom output gradient v
```

### 5.2 Implementation

When `tensor|∇` is evaluated:

```rust
// In eval_pipe or method call handling
if method_name == "∇" || method_name == "nabla" {
    return self.backward_pass(tensor)?;
}

fn backward_pass(&mut self, output: &Value) -> Result<Value, RuntimeError> {
    // 1. Topological sort of computation graph
    let sorted_nodes = self.topo_sort(output)?;

    // 2. Initialize gradients map
    let mut grads: HashMap<TensorId, Value> = HashMap::new();

    // 3. Seed with output gradient (ones for scalar loss)
    let output_id = self.tensor_id(output)?;
    grads.insert(output_id, self.ones_like(output)?);

    // 4. Backward pass in reverse topological order
    for node in sorted_nodes.iter().rev() {
        if let Some(grad_fn) = self.get_grad_fn(node)? {
            let out_grad = grads.get(&self.tensor_id(node)?).unwrap();

            // Apply chain rule
            let input_grads = self.apply_grad_fn(&grad_fn, out_grad)?;

            // Accumulate to inputs
            for (input_id, input_grad) in grad_fn.inputs.iter().zip(input_grads) {
                grads.entry(*input_id)
                    .and_modify(|g| *g = self.tensor_add(g, &input_grad).unwrap())
                    .or_insert(input_grad);
            }
        }
    }

    // 5. Return Gradients struct
    Ok(Value::Struct {
        name: "Gradients".to_string(),
        fields: Rc::new(RefCell::new(grads_to_fields(grads))),
    })
}
```

### 5.3 Topological Sort

```rust
fn topo_sort(&self, root: &Value) -> Result<Vec<Value>, RuntimeError> {
    let mut visited = HashSet::new();
    let mut sorted = Vec::new();

    fn visit(
        node: &Value,
        visited: &mut HashSet<TensorId>,
        sorted: &mut Vec<Value>,
        interp: &Interpreter,
    ) -> Result<(), RuntimeError> {
        let id = interp.tensor_id(node)?;
        if visited.contains(&id) {
            return Ok(());
        }
        visited.insert(id);

        // Visit inputs first
        if let Some(grad_fn) = interp.get_grad_fn(node)? {
            for input in grad_fn.inputs {
                visit(&input, visited, sorted, interp)?;
            }
        }

        sorted.push(node.clone());
        Ok(())
    }

    visit(root, &mut visited, &mut sorted, self)?;
    Ok(sorted)
}
```

---

## 6. Gradient Recording

### 6.1 During Forward Pass

Each differentiable operation must:
1. Check if any input `requires_grad`
2. If so, create and attach a GradFn to the output

```rust
// Example: Matrix multiply x @ w
fn eval_matmul(&mut self, x: &Value, w: &Value) -> Result<Value, RuntimeError> {
    let result = self.tensor_matmul(x, w)?;

    // If either input requires grad, record for backward
    if self.requires_grad(x)? || self.requires_grad(w)? {
        let grad_fn = GradFn {
            name: "MatMul".to_string(),
            inputs: vec![self.tensor_id(x)?, self.tensor_id(w)?],
            saved: vec![x.clone(), w.clone()], // Saved for backward
        };
        self.set_grad_fn(&result, grad_fn)?;
    }

    Ok(result)
}
```

### 6.2 Gradient Context (no_grad)

```sigil
no_grad {
    // Operations here don't record gradients
    ≔ inference_output = model|forward(input);
}
```

Implemented via a flag:
```rust
impl Interpreter {
    gradient_enabled: RefCell<bool>,  // true by default

    fn with_no_grad<F, R>(&mut self, f: F) -> R {
        let prev = *self.gradient_enabled.borrow();
        *self.gradient_enabled.borrow_mut() = false;
        let result = f();
        *self.gradient_enabled.borrow_mut() = prev;
        result
    }
}
```

---

## 7. Phased Implementation

### Phase 1: Minimal ∇ for MNIST (Recommended First)

A simpler approach that doesn't require full computation graph:

1. **Store parameter references** in the model
2. **Compute gradients numerically** during ∇ call
3. **Return gradients map** for optimizer

```rust
// Simplified backward for MNIST
// Assumes cross_entropy + softmax combined gradient
fn backward_mnist(&mut self, loss: &Value, model: &Value) -> Result<Value, RuntimeError> {
    // For MNIST: gradients are computed analytically
    // cross_entropy_softmax_backward is a single operation

    let logits = self.get_cached("logits")?;  // Saved during forward
    let targets = self.get_cached("targets")?;

    // Softmax + CrossEntropy gradient: y - one_hot(t)
    let grad_logits = self.softmax_ce_backward(&logits, &targets)?;

    // Propagate through Linear layers (simple chain rule)
    let grads = self.linear_backward(&grad_logits, model)?;

    Ok(grads)
}
```

### Phase 2: Full Computation Graph

Full implementation with:
- Per-operation GradFn recording
- Topological sort
- General chain rule application

### Phase 3: Advanced Features

- Gradient checkpointing
- Higher-order derivatives (Hessian)
- Custom gradient functions

---

## 8. Test Cases

### 8.1 Basic Gradient

```sigil
≔ x = Tensor·from([2.0, 3.0]);
x.set_requires_grad(yea);

≔ y = x|τ{v => v * v}|Σ;  // y = sum(x^2) = 4 + 9 = 13
≔ grads = y|∇;

// ∂y/∂x = 2x = [4.0, 6.0]
assert(grads.get(&x) == Tensor·from([4.0, 6.0]));
```

### 8.2 Chain Rule

```sigil
≔ x = Tensor·from([1.0, 2.0]);
x.set_requires_grad(yea);

≔ y = x|τ{v => v * 2.0};   // y = 2x
≔ z = y|τ{v => v * v}|Σ;   // z = sum(4x^2) = 4 + 16 = 20

≔ grads = z|∇;

// ∂z/∂x = 8x = [8.0, 16.0]
assert(grads.get(&x) == Tensor·from([8.0, 16.0]));
```

### 8.3 MatMul Gradient

```sigil
≔ x = Tensor·randn([2, 3]);
x.set_requires_grad(yea);
≔ w = Tensor·randn([3, 4]);
w.set_requires_grad(yea);

≔ y = x @ w;  // [2, 4]
≔ loss = y|Σ;

≔ grads = loss|∇;

// ∂L/∂x has shape [2, 3]
// ∂L/∂w has shape [3, 4]
assert(grads.get(&x).shape() == [2, 3]);
assert(grads.get(&w).shape() == [3, 4]);
```

---

## 9. Related Specs

- 03-TYPES.md § Tensor type
- (TBD) - Pipe operator semantics
- 03D-OPTION-REPRESENTATION.md - Option handling in returns

---

## 10. Resolution Plan

### Immediate (Phase 1) ✅ COMPLETED
1. ✅ Add `∇` as a recognized pipe method in `eval_pipe`
2. ✅ Implement simplified backward for cross_entropy + linear layers
3. ✅ Return stub Gradients struct that optimizer can use
4. ✅ Add `Gradients.get(param)` method
5. ✅ Add `sqrt` pipe method for gradient norm computation
6. ✅ Add Tensor subtraction for optimizer weight updates

### Short-term (Phase 2)
1. Add `__grad_fn__` field to Tensor struct
2. Record gradient functions during forward operations
3. Implement topological sort and full backward pass

### Long-term (Phase 3)
1. Gradient checkpointing for memory efficiency
2. Higher-order derivatives
3. JIT-compiled backward pass

---

## 11. Implementation Status (2026-02-08)

### Completed
- ∇ operator works as a pipe method: `loss|∇` returns `Gradients` struct
- `Gradients.get(param)` returns `Option<Gradient>`
- `cross_entropy(logits, targets)` builtin with numerically stable log-softmax
- Tensor binary operations: `+`, `-`, `*`, `@` (matmul)
- `sqrt` pipe method for gradient norm computation: `value|sqrt`

### Remaining Gaps
- **Optimizer API mismatch**: `mnist_training.sigil` calls `optimizer.step(&grads!)` but nihil_optim's `step(&vary this)` doesn't accept gradients argument
- Full computation graph tracking not yet implemented
- Per-parameter gradient accumulation needs refinement

### Test Files
- `/tmp/test_autograd_complete.sg` - Demonstrates complete autograd workflow
