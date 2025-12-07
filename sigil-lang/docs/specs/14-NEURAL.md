# Neural Networks and Differentiable Programming

> *"A neural network is a function approximator that learns its own features."*

> *"The weights of a trained network holographically encode the training data."*

## 1. Overview

Sigil provides first-class support for neural networks through:

1. **Differentiable tensors** — Automatic gradient tracking
2. **Polysynthetic layer composition** — Networks as morpheme chains
3. **Evidentiality integration** — Model outputs are inherently `◊` (uncertain)
4. **Holographic connection** — Networks as information compression

### 1.1 Core Types

```sigil
/// A tensor with automatic differentiation
type Tensor<Shape, Dtype = f32> = {
    data: Storage<Dtype>,
    shape: Shape,
    grad: Option<Tensor<Shape, Dtype>>,
    requires_grad: bool,
}

/// Shape as type-level list
type Shape = [usize; N]

/// Common tensor aliases
type Scalar = Tensor<[]>
type Vector<const N: usize> = Tensor<[N]>
type Matrix<const M: usize, const N: usize> = Tensor<[M, N]>
type Tensor3D<const A: usize, const B: usize, const C: usize> = Tensor<[A, B, C]>

/// A neural network module
trait Module {
    type Input;
    type Output;

    fn forward(&self, input: Self::Input) -> Self::Output;
    fn parameters(&self) -> impl Iterator<&Tensor>;
}
```

### 1.2 Neural Operators

| Symbol | Name | Meaning |
|--------|------|---------|
| `∇` | Nabla | Gradient / backpropagation |
| `⊙` | Hadamard | Element-wise multiplication |
| `⊗` | Tensor | Outer product / tensor product |
| `@` | MatMul | Matrix multiplication |
| `◊` | Possibility | Uncertain output (predictions) |

---

## 2. Tensor Operations

### 2.1 Creating Tensors

```sigil
use neural·tensor·{Tensor, zeros, ones, randn, arange}

fn create_tensors() {
    // Explicit shape
    let a: Tensor<[3, 4]> = zeros()
    let b: Tensor<[3, 4]> = ones()
    let c: Tensor<[3, 4]> = randn()  // Normal distribution

    // From data
    let d = Tensor::from([[1.0, 2.0], [3.0, 4.0]])

    // Requires gradient for training
    let w = randn::<[784, 256]>()|requires_grad(true)

    // Range
    let indices = arange(0, 100)  // [0, 1, 2, ..., 99]
}
```

### 2.2 Tensor Arithmetic

```sigil
fn tensor_ops() {
    let a: Tensor<[3, 4]> = randn()
    let b: Tensor<[3, 4]> = randn()
    let c: Tensor<[4, 5]> = randn()

    // Element-wise operations
    let sum = a + b
    let diff = a - b
    let hadamard = a ⊙ b      // Element-wise multiply
    let scaled = a * 2.0
    let powered = a ** 2

    // Matrix multiplication
    let product = a @ c        // [3, 4] @ [4, 5] = [3, 5]

    // Reductions
    let total = a|Σ            // Sum all elements
    let row_sums = a|Σ(axis: 1)  // Sum along axis
    let mean = a|μ             // Mean
    let max = a|max
    let min = a|min

    // Broadcasting
    let broadcasted = a + ones::<[4]>()  // Broadcasts [4] to [3, 4]
}
```

### 2.3 Shape Operations

```sigil
fn shape_ops() {
    let a: Tensor<[2, 3, 4]> = randn()

    // Reshape
    let b: Tensor<[6, 4]> = a|reshape([6, 4])
    let c: Tensor<[24]> = a|flatten

    // Transpose
    let d: Tensor<[4, 3, 2]> = a|transpose([2, 1, 0])
    let e: Tensor<[3, 2, 4]> = a|T  // Swap last two dims

    // Expand/squeeze
    let f: Tensor<[1, 2, 3, 4]> = a|unsqueeze(0)
    let g: Tensor<[2, 3, 4]> = f|squeeze(0)

    // Concatenate
    let h = [a, a, a]|cat(axis: 0)  // [6, 3, 4]

    // Stack
    let i = [a, a]|stack(axis: 0)  // [2, 2, 3, 4]

    // Split
    let chunks = a|chunk(2, axis: 0)  // [[1, 3, 4], [1, 3, 4]]
}
```

---

## 3. Automatic Differentiation

### 3.1 The Gradient Operator (∇)

```sigil
use neural·autograd·{grad, backward, no_grad}

fn autodiff_example() {
    let x = Tensor::from([2.0])|requires_grad(true)
    let y = Tensor::from([3.0])|requires_grad(true)

    // Forward pass builds computation graph
    let z = x ** 2 + y ** 3  // z = x² + y³

    // Backward pass computes gradients
    z|∇  // or z|backward

    // Access gradients
    let dx = x.grad  // ∂z/∂x = 2x = 4.0
    let dy = y.grad  // ∂z/∂y = 3y² = 27.0

    print!("∂z/∂x = {dx}, ∂z/∂y = {dy}")
}

/// Compute gradient of function
fn gradient_function() {
    // Define a function
    let f = {x: Tensor => x ** 2 + x|sin}

    // Get gradient function
    let df = f|∇  // df(x) = 2x + cos(x)

    // Evaluate gradient at point
    let x = Tensor::from([1.0])
    let gradient = df(x)  // 2.0 + cos(1.0) ≈ 2.54
}
```

### 3.2 Computation Graphs

```sigil
/// Computation graph for backpropagation
struct ComputationGraph {
    nodes: Vec<Node>,
    edges: Vec<Edge>,
}

struct Node {
    tensor: TensorId,
    operation: Operation,
    inputs: Vec<TensorId>,
    grad_fn: Option<fn(Tensor) -> Vec<Tensor>>,
}

/// Gradient flows backward through graph
fn backward_pass(output: Tensor) {
    // Initialize output gradient to 1
    output.grad = Some(ones_like(output))

    // Topological sort, reversed
    let nodes = output.graph|toposort|reverse

    // Propagate gradients
    for node in nodes {
        if let Some(grad_fn) = node.grad_fn {
            let input_grads = grad_fn(node.tensor.grad)

            for (input, grad) in node.inputs|zip(input_grads) {
                input.grad += grad  // Accumulate gradients
            }
        }
    }
}
```

### 3.3 Higher-Order Derivatives

```sigil
/// Hessian (second derivatives)
fn hessian(f: fn(Tensor) -> Tensor, x: Tensor) -> Tensor {
    let grad_f = f|∇
    let hess_f = grad_f|∇  // Gradient of gradient
    hess_f(x)
}

/// Jacobian matrix
fn jacobian(f: fn(Tensor<[N]>) -> Tensor<[M]>, x: Tensor<[N]>) -> Tensor<[M, N]> {
    // Compute each row of Jacobian
    (0..M)|τ{i =>
        let ei = one_hot(i, M)  // Unit vector
        let vjp = f|vjp(x, ei)  // Vector-Jacobian product
        vjp
    }|stack
}

/// Jacobian-vector product (forward mode)
fn jvp(f: fn(Tensor) -> Tensor, x: Tensor, v: Tensor) -> Tensor {
    // Efficient for tall Jacobians (N < M)
    f|forward_diff(x, v)
}

/// Vector-Jacobian product (reverse mode)
fn vjp(f: fn(Tensor) -> Tensor, x: Tensor, v: Tensor) -> Tensor {
    // Efficient for wide Jacobians (N > M)
    let y = f(x)
    y.grad = v
    y|backward
    x.grad
}
```

---

## 4. Neural Network Layers

### 4.1 Layer Trait

```sigil
/// A neural network layer
trait Layer: Module {
    /// Number of trainable parameters
    fn num_parameters(&self) -> usize;

    /// Initialize parameters
    fn reset_parameters(&mut self);
}

/// Composable layers
impl<A: Layer, B: Layer> Layer for (A, B)
where A::Output == B::Input
{
    type Input = A::Input;
    type Output = B::Output;

    fn forward(&self, x: Self::Input) -> Self::Output {
        x|self.0.forward|self.1.forward
    }
}
```

### 4.2 Core Layers

```sigil
use neural·layer·{Linear, Conv2d, BatchNorm, LayerNorm, Dropout}

/// Linear (fully connected) layer
struct Linear<const IN: usize, const OUT: usize> {
    weight: Tensor<[OUT, IN]>,
    bias: Option<Tensor<[OUT]>>,
}

impl Linear<IN, OUT> {
    fn new() -> Self {
        Self {
            weight: kaiming_uniform([OUT, IN]),
            bias: Some(zeros([OUT])),
        }
    }

    fn forward(&self, x: Tensor<[B, IN]>) -> Tensor<[B, OUT]> {
        let y = x @ self.weight.T
        match &self.bias {
            Some(b) => y + b,
            None => y,
        }
    }
}

/// 2D Convolution
struct Conv2d<const IN: usize, const OUT: usize, const K: usize> {
    weight: Tensor<[OUT, IN, K, K]>,
    bias: Option<Tensor<[OUT]>>,
    stride: usize,
    padding: usize,
}

impl Conv2d<IN, OUT, K> {
    fn forward(&self, x: Tensor<[B, IN, H, W]>) -> Tensor<[B, OUT, H', W']> {
        let y = x|conv2d(self.weight, self.stride, self.padding)
        match &self.bias {
            Some(b) => y + b|unsqueeze(-1)|unsqueeze(-1),
            None => y,
        }
    }
}

/// Batch Normalization
struct BatchNorm<const C: usize> {
    gamma: Tensor<[C]>,      // Learnable scale
    beta: Tensor<[C]>,       // Learnable shift
    running_mean: Tensor<[C]>,
    running_var: Tensor<[C]>,
    momentum: f32,
    eps: f32,
}

/// Layer Normalization
struct LayerNorm<const D: usize> {
    gamma: Tensor<[D]>,
    beta: Tensor<[D]>,
    eps: f32,
}

/// Dropout (training only)
struct Dropout {
    p: f32,  // Drop probability
}

impl Dropout {
    fn forward(&self, x: Tensor, training: bool) -> Tensor {
        if training {
            let mask = (randn_like(x) > self.p)|to_float
            x ⊙ mask / (1.0 - self.p)  // Scale to maintain expectation
        } else {
            x
        }
    }
}
```

### 4.3 Activation Functions

```sigil
use neural·activation·{relu, sigmoid, tanh, gelu, softmax, log_softmax}

/// Activation functions as morphemes
fn activations(x: Tensor) {
    // ReLU family
    let a = x|relu           // max(0, x)
    let b = x|leaky_relu(0.01)  // max(0.01x, x)
    let c = x|elu(1.0)       // x if x > 0, α(eˣ - 1) otherwise
    let d = x|selu           // Self-normalizing

    // Smooth activations
    let e = x|sigmoid        // 1 / (1 + e⁻ˣ)
    let f = x|tanh           // (eˣ - e⁻ˣ) / (eˣ + e⁻ˣ)
    let g = x|gelu           // x · Φ(x), Gaussian CDF
    let h = x|swish          // x · sigmoid(x)
    let i = x|mish           // x · tanh(softplus(x))

    // Softmax (for classification)
    let j = x|softmax(dim: -1)      // eˣⁱ / Σeˣʲ
    let k = x|log_softmax(dim: -1)  // Numerically stable log(softmax)
}

/// Custom activation via morpheme
fn custom_activation() {
    let snake = {x: Tensor => x + (x|sin ** 2) / 2.0}

    let output = input|snake
}
```

---

## 5. Polysynthetic Network Definition

### 5.1 Sequential Composition

```sigil
use neural·compose·{Sequential, seq}

/// Build network with pipe chains
fn mlp() -> impl Module {
    seq![
        Linear::<784, 256>,
        relu,
        Dropout(0.5),
        Linear::<256, 128>,
        relu,
        Dropout(0.5),
        Linear::<128, 10>,
        log_softmax,
    ]
}

/// More concise with morphemes
fn mlp_morpheme() -> impl Module {
    Linear::<784, 256>
        |then(relu)
        |then(Dropout(0.5))
        |then(Linear::<256, 128>)
        |then(relu)
        |then(Linear::<128, 10>)
        |then(log_softmax)
}

/// Using incorporation syntax
fn mlp_incorporation() -> impl Module {
    input·Linear::<784, 256>·relu·drop(0.5)
         ·Linear::<256, 128>·relu·drop(0.5)
         ·Linear::<128, 10>·log_softmax
}
```

### 5.2 Residual Connections

```sigil
use neural·compose·{Residual, residual}

/// Residual block
struct ResidualBlock<const C: usize> {
    conv1: Conv2d<C, C, 3>,
    bn1: BatchNorm<C>,
    conv2: Conv2d<C, C, 3>,
    bn2: BatchNorm<C>,
}

impl Module for ResidualBlock<C> {
    fn forward(&self, x: Tensor) -> Tensor {
        let residual = x.clone()

        let y = x
            |self.conv1
            |self.bn1
            |relu
            |self.conv2
            |self.bn2

        (y + residual)|relu  // Skip connection
    }
}

/// Residual as combinator
fn residual_combinator(block: impl Module) -> impl Module {
    Residual(block)  // Automatically adds skip connection
}

/// With morpheme
fn resnet_layer() -> impl Module {
    Conv2d::<64, 64, 3>
        |then(BatchNorm::<64>)
        |then(relu)
        |then(Conv2d::<64, 64, 3>)
        |then(BatchNorm::<64>)
        |residual  // Wrap in residual connection
        |then(relu)
}
```

### 5.3 Attention Mechanisms

```sigil
use neural·attention·{MultiHeadAttention, SelfAttention}

/// Multi-head self-attention
struct MultiHeadAttention<const D: usize, const H: usize> {
    q_proj: Linear<D, D>,
    k_proj: Linear<D, D>,
    v_proj: Linear<D, D>,
    out_proj: Linear<D, D>,
}

impl MultiHeadAttention<D, H> {
    fn forward(&self, x: Tensor<[B, S, D]>, mask: Option<Tensor>) -> Tensor<[B, S, D]> {
        let head_dim = D / H

        // Project to Q, K, V
        let q = x|self.q_proj|reshape([B, S, H, head_dim])|transpose([0, 2, 1, 3])
        let k = x|self.k_proj|reshape([B, S, H, head_dim])|transpose([0, 2, 1, 3])
        let v = x|self.v_proj|reshape([B, S, H, head_dim])|transpose([0, 2, 1, 3])

        // Scaled dot-product attention
        let scores = (q @ k.T) / (head_dim as f32).sqrt()

        // Apply mask if provided
        let scores = match mask {
            Some(m) => scores|masked_fill(m, f32::NEG_INFINITY),
            None => scores,
        }

        let attn = scores|softmax(dim: -1)
        let out = attn @ v

        // Concatenate heads and project
        out
            |transpose([0, 2, 1, 3])
            |reshape([B, S, D])
            |self.out_proj
    }
}

/// Transformer block
struct TransformerBlock<const D: usize, const H: usize, const FF: usize> {
    attn: MultiHeadAttention<D, H>,
    ff: Sequential<Linear<D, FF>, GELU, Linear<FF, D>>,
    norm1: LayerNorm<D>,
    norm2: LayerNorm<D>,
    dropout: Dropout,
}

impl Module for TransformerBlock<D, H, FF> {
    fn forward(&self, x: Tensor) -> Tensor {
        // Pre-norm architecture
        let attn_out = x|self.norm1|self.attn|self.dropout
        let x = x + attn_out

        let ff_out = x|self.norm2|self.ff|self.dropout
        x + ff_out
    }
}
```

---

## 6. Training

### 6.1 Loss Functions

```sigil
use neural·loss·{mse_loss, cross_entropy, nll_loss, binary_cross_entropy}

/// Common losses
fn losses(pred: Tensor, target: Tensor) {
    // Regression
    let mse = (pred - target) ** 2|μ      // Mean squared error
    let mae = (pred - target)|abs|μ       // Mean absolute error
    let huber = smooth_l1_loss(pred, target, beta: 1.0)

    // Classification
    let ce = cross_entropy(pred, target)           // Softmax + NLL
    let nll = nll_loss(pred|log_softmax, target)   // Negative log likelihood
    let bce = binary_cross_entropy(pred|sigmoid, target)

    // Custom
    let focal = focal_loss(pred, target, gamma: 2.0)  // For imbalanced
}

/// Loss with reduction
fn loss_reduction(pred: Tensor, target: Tensor) {
    let per_sample = mse_loss(pred, target, reduction: None)  // [B]
    let mean_loss = mse_loss(pred, target, reduction: Mean)   // scalar
    let sum_loss = mse_loss(pred, target, reduction: Sum)     // scalar
}
```

### 6.2 Optimizers

```sigil
use neural·optim·{SGD, Adam, AdamW, LAMB, Optimizer}

/// Optimizer trait
trait Optimizer {
    fn step(&mut self, grads: impl Iterator<(&Tensor, &Tensor)>);
    fn zero_grad(&mut self);
}

/// SGD with momentum
struct SGD {
    lr: f32,
    momentum: f32,
    weight_decay: f32,
    velocity: HashMap<TensorId, Tensor>,
}

impl Optimizer for SGD {
    fn step(&mut self, params: impl Iterator<&mut Tensor>) {
        for param in params {
            if let Some(grad) = &param.grad {
                // Weight decay
                let grad = grad + self.weight_decay * param

                // Momentum
                let v = self.velocity
                    |entry(param.id)
                    |or_insert(zeros_like(param))
                *v = self.momentum * *v + grad

                // Update
                *param -= self.lr * *v
            }
        }
    }
}

/// Adam optimizer
struct Adam {
    lr: f32,
    betas: (f32, f32),
    eps: f32,
    weight_decay: f32,
    m: HashMap<TensorId, Tensor>,  // First moment
    v: HashMap<TensorId, Tensor>,  // Second moment
    t: usize,                      // Timestep
}

impl Optimizer for Adam {
    fn step(&mut self, params: impl Iterator<&mut Tensor>) {
        self.t += 1

        for param in params {
            if let Some(grad) = &param.grad {
                let m = self.m|entry(param.id)|or_insert(zeros_like(param))
                let v = self.v|entry(param.id)|or_insert(zeros_like(param))

                // Update biased moments
                *m = self.betas.0 * *m + (1.0 - self.betas.0) * grad
                *v = self.betas.1 * *v + (1.0 - self.betas.1) * grad ** 2

                // Bias correction
                let m_hat = *m / (1.0 - self.betas.0.pow(self.t))
                let v_hat = *v / (1.0 - self.betas.1.pow(self.t))

                // Update with AdamW weight decay
                *param -= self.lr * (m_hat / (v_hat.sqrt() + self.eps)
                    + self.weight_decay * param)
            }
        }
    }
}
```

### 6.3 Training Loop

```sigil
use neural·train·{Trainer, fit}

/// Training loop with morphemes
fn train_model(
    model: &mut impl Module,
    train_loader: DataLoader,
    optimizer: &mut impl Optimizer,
    epochs: usize,
) {
    for epoch in 0..epochs {
        let epoch_loss = train_loader
            |τ{(batch, labels) =>
                // Forward pass
                let output = model|forward(batch)
                let loss = cross_entropy(output, labels)

                // Backward pass
                optimizer|zero_grad
                loss|∇

                // Update
                optimizer|step(model|parameters)

                loss
            }
            |μ  // Mean loss over batches

        print!("Epoch {epoch}: loss = {epoch_loss}")
    }
}

/// More concise with trainer
fn train_concise() {
    let trainer = Trainer::new(model, optimizer)
        |loss_fn(cross_entropy)
        |metrics([accuracy, f1_score])
        |callbacks([
            EarlyStopping(patience: 5),
            ModelCheckpoint("best.pt"),
            LRScheduler(CosineAnnealing(T_max: 100)),
        ])

    trainer|fit(train_loader, val_loader, epochs: 100)
}
```

---

## 7. Evidentiality for Neural Outputs

### 7.1 Predictions are Uncertain (◊)

```sigil
/// Neural network outputs are INHERENTLY uncertain
/// They should carry ◊ evidence

struct Model {
    network: impl Module,
}

impl Model {
    /// Predict returns uncertain result
    fn predict(&self, input: Tensor!) -> Tensor◊ {
        // Input is known, but output is approximation
        self.network|forward(input)◊  // Mark as uncertain
    }

    /// Get confidence along with prediction
    fn predict_with_confidence(&self, input: Tensor!) -> (Tensor◊, Confidence◊) {
        let logits = self.network|forward(input)
        let probs = logits|softmax

        let prediction◊ = probs|argmax
        let confidence◊ = probs|max  // How confident?

        (prediction◊, confidence◊)
    }
}

/// Using predictions requires acknowledging uncertainty
fn use_prediction(model: &Model, input: Tensor!) {
    let pred◊ = model|predict(input)

    // ❌ Can't use directly as known
    let result! = pred◊  // ERROR: ◊ is not !

    // ✅ Must handle uncertainty
    let result? = pred◊|threshold(0.9)  // Only accept high confidence

    // ✅ Or explicitly acknowledge
    let result◊ = pred◊  // Keep it uncertain
}
```

### 7.2 Calibrated Uncertainty

```sigil
use neural·uncertainty·{MCDropout, Ensemble, BayesianLayer}

/// Monte Carlo Dropout for uncertainty estimation
fn mc_dropout_predict(model: &Model, input: Tensor, samples: usize) -> (Tensor◊, Tensor◊) {
    let predictions = (0..samples)
        |τ{_ =>
            model|forward_with_dropout(input)  // Keep dropout on
        }
        |stack

    let mean◊ = predictions|μ(axis: 0)
    let std◊ = predictions|σ(axis: 0)  // Epistemic uncertainty

    (mean◊, std◊)
}

/// Ensemble for uncertainty
fn ensemble_predict(models: [Model], input: Tensor) -> (Tensor◊, Tensor◊) {
    let predictions = models
        |τ{m => m|predict(input)}
        |stack

    let mean◊ = predictions|μ(axis: 0)
    let std◊ = predictions|σ(axis: 0)

    (mean◊, std◊)
}

/// Bayesian neural network
struct BayesianLinear<const IN: usize, const OUT: usize> {
    weight_mean: Tensor<[OUT, IN]>,
    weight_logvar: Tensor<[OUT, IN]>,
    bias_mean: Tensor<[OUT]>,
    bias_logvar: Tensor<[OUT]>,
}

impl BayesianLinear<IN, OUT> {
    fn forward(&self, x: Tensor, sample: bool) -> Tensor {
        if sample {
            // Reparameterization trick
            let weight_std = (self.weight_logvar / 2.0)|exp
            let weight = self.weight_mean + weight_std ⊙ randn_like(weight_std)

            let bias_std = (self.bias_logvar / 2.0)|exp
            let bias = self.bias_mean + bias_std ⊙ randn_like(bias_std)

            x @ weight.T + bias
        } else {
            x @ self.weight_mean.T + self.bias_mean
        }
    }

    fn kl_divergence(&self) -> Tensor {
        // KL(q(w) || p(w)) for ELBO
        kl_gaussian(
            self.weight_mean, self.weight_logvar,
            zeros_like(self.weight_mean), ones_like(self.weight_logvar)
        )
    }
}
```

---

## 8. Neural-Holographic Connection

### 8.1 Networks as Information Compression

```sigil
/// A trained network holographically encodes its training data
/// Weights are the "boundary"; training data is the "bulk"

/// Information bottleneck view
struct InformationBottleneck {
    encoder: impl Module,  // X → Z (compress)
    decoder: impl Module,  // Z → Y (decompress)
}

impl InformationBottleneck {
    /// The bottleneck contains "essential" information
    fn forward(&self, x: Tensor) -> Tensor {
        let z = self.encoder|forward(x)  // Compress to bottleneck
        self.decoder|forward(z)          // Expand to output
    }

    /// Measure information content
    fn mutual_information(&self, x: Tensor, z: Tensor) -> f32 {
        // I(X; Z) should be minimized while I(Z; Y) maximized
        estimate_mutual_info(x, z)
    }
}

/// Connection to holography:
/// weights ≈ boundary CFT
/// training data ≈ bulk spacetime
/// generalization ≈ bulk reconstruction from boundary
fn holographic_analogy() {
    // Training: encode bulk (data) into boundary (weights)
    let weights = train(data)  // data|⊗ code → weights

    // Inference: reconstruct "bulk" (predictions) from boundary
    let predictions = infer(weights, new_input)  // weights|∀ → output

    // The ∀ operator applies!
    // Network "reconstructs" the data distribution from weights
}
```

### 8.2 Tensor Network Neural Architectures

```sigil
use neural·tensor_network·{TensorTrain, MERA, TensorRing}

/// Tensor train (matrix product state) layer
struct TensorTrainLinear<const IN: usize, const OUT: usize, const R: usize> {
    cores: Vec<Tensor<[R, D, R]>>,  // TT cores with rank R
}

impl TensorTrainLinear<IN, OUT, R> {
    /// Exponentially fewer parameters than dense
    fn num_parameters(&self) -> usize {
        // O(N D R²) instead of O(D^N)
        self.cores.len * R * R * D
    }

    fn forward(&self, x: Tensor) -> Tensor {
        // Contract tensor train
        self.cores
            |fold(x, {acc, core => acc|contract(core)})
    }
}

/// MERA-inspired architecture
struct MERALayer {
    disentanglers: Vec<Tensor<[4, 4]>>,
    isometries: Vec<Tensor<[4, 2]>>,
}

impl Module for MERALayer {
    fn forward(&self, x: Tensor) -> Tensor {
        // Coarse-graining like in holographic MERA
        let x = x|apply_local_unitaries(self.disentanglers)
        x|coarse_grain(self.isometries)
    }
}

/// This is literal quantum-inspired ML!
fn quantum_classical_connection() {
    // MERA compresses like RG flow
    // Neural network compresses like information bottleneck
    // Both are holographic!
}
```

### 8.3 Attention is All You Need (Holographically)

```sigil
/// Attention computes "which parts of input matter for each output"
/// This is analogous to entanglement wedge reconstruction!

fn attention_holography() {
    let query: Tensor<[B, S, D]> = ...   // "What am I looking for?"
    let key: Tensor<[B, S, D]> = ...     // "What do I contain?"
    let value: Tensor<[B, S, D]> = ...   // "What information do I have?"

    // Attention weights = "entanglement structure"
    let attn = (query @ key.T)|softmax

    // Output = "reconstruction from attended region"
    let output = attn @ value  // Like ∀ over attended positions!
}

/// Transformer as holographic code
struct HolographicTransformer {
    /// Each layer is a "scale" in the holographic RG
    layers: Vec<TransformerBlock>,
}

impl HolographicTransformer {
    fn forward(&self, x: Tensor) -> Tensor {
        // Progressive coarse-graining
        self.layers|fold(x, {state, layer => layer|forward(state)})
    }

    /// Information flows like in MERA
    fn information_flow(&self, x: Tensor) -> Vec<Tensor> {
        self.layers|scan(x, {state, layer =>
            let new_state = layer|forward(state)
            // Mutual information decreases (compression)
            (new_state, new_state)
        })
    }
}
```

---

## 9. Quantum Neural Networks

### 9.1 Variational Quantum Circuits as Neural Nets

```sigil
use quantum·circuit·{Circuit, qubit}
use neural·quantum·{QuantumLayer, hybrid_forward}

/// Quantum layer in classical network
struct QuantumLayer<const N: usize> {
    params: Tensor<[N * 3]>,  // Rotation angles
    circuit: fn([f32; N * 3]) -> Circuit,
}

impl Module for QuantumLayer<N> {
    fn forward(&self, x: Tensor<[B, N]>) -> Tensor<[B, M]> {
        // Encode classical data into quantum circuit
        x|encode_amplitude
         |apply_variational_circuit(self.params)
         |measure_expectations
    }
}

/// Hybrid quantum-classical network
struct HybridNetwork {
    classical_encoder: impl Module,
    quantum_layer: QuantumLayer<4>,
    classical_decoder: impl Module,
}

impl Module for HybridNetwork {
    fn forward(&self, x: Tensor) -> Tensor◊ {
        x
            |self.classical_encoder
            |self.quantum_layer    // Quantum processing
            |self.classical_decoder
            ◊  // Quantum introduces inherent uncertainty
    }
}
```

### 9.2 Quantum Gradients

```sigil
/// Parameter shift rule for quantum gradients
fn parameter_shift_gradient(
    circuit: Circuit,
    param_idx: usize,
    shift: f32 = π / 2,
) -> f32 {
    let plus = circuit|with_param(param_idx, +shift)|execute|expectation
    let minus = circuit|with_param(param_idx, -shift)|execute|expectation

    (plus - minus) / (2.0 * shift.sin())
}

/// Train quantum circuit like neural network
fn train_qnn(
    circuit: &mut ParameterizedCircuit,
    data: DataLoader,
    optimizer: &mut Adam,
) {
    for (x, y) in data {
        // Forward
        let output◊ = circuit|execute_batch(x)|expectations

        // Loss
        let loss = mse_loss(output◊, y)

        // Quantum gradients via parameter shift
        let grads = circuit.params
            |enumerate
            |τ{(i, _) => parameter_shift_gradient(circuit, i)}
            |collect

        // Classical optimization step
        optimizer|step_with_grads(circuit.params, grads)
    }
}
```

---

## 10. Standard Library: `neural`

```sigil
//! Neural network and differentiable programming primitives

pub mod tensor {
    pub type Tensor<Shape, Dtype>;
    pub fn zeros, ones, randn, arange, linspace;
    pub fn cat, stack, chunk, split;
    pub fn reshape, transpose, flatten, squeeze, unsqueeze;
}

pub mod autograd {
    pub fn backward, grad, no_grad;
    pub fn jacobian, hessian, jvp, vjp;
}

pub mod layer {
    pub struct Linear, Conv1d, Conv2d, Conv3d;
    pub struct BatchNorm, LayerNorm, GroupNorm;
    pub struct Dropout, AlphaDropout;
    pub struct Embedding, EmbeddingBag;
    pub struct LSTM, GRU, RNN;
    pub struct MultiHeadAttention, TransformerEncoder, TransformerDecoder;
}

pub mod activation {
    pub fn relu, leaky_relu, elu, selu, gelu, swish, mish;
    pub fn sigmoid, tanh, softmax, log_softmax;
}

pub mod loss {
    pub fn mse_loss, l1_loss, smooth_l1_loss;
    pub fn cross_entropy, nll_loss, binary_cross_entropy;
    pub fn focal_loss, dice_loss, contrastive_loss;
}

pub mod optim {
    pub trait Optimizer;
    pub struct SGD, Adam, AdamW, LAMB, RMSprop;
    pub mod lr_scheduler {
        pub struct StepLR, CosineAnnealing, OneCycle, WarmupLR;
    }
}

pub mod data {
    pub struct Dataset, DataLoader;
    pub fn random_split, k_fold;
}

pub mod uncertainty {
    pub struct MCDropout, Ensemble, BayesianModule;
    pub fn calibrate, expected_calibration_error;
}

pub mod quantum {
    pub struct QuantumLayer, HybridNetwork;
    pub fn parameter_shift, quantum_natural_gradient;
}

pub mod holographic {
    pub struct TensorTrainLayer, MERALayer;
    pub fn information_bottleneck, mutual_information;
}
```

---

## 11. Summary: The Neural-Holographic-Quantum Triangle

```
                        ╔═══════════════╗
                        ║    Neural     ║
                        ║   Networks    ║
                        ╚═══════╤═══════╝
                                │
            ┌───────────────────┼───────────────────┐
            │                   │                   │
            ▼                   ▼                   ▼
    ┌───────────────┐   ┌───────────────┐   ┌───────────────┐
    │  Information  │   │    Tensor     │   │   Uncertain   │
    │  Compression  │   │   Networks    │   │    Outputs    │
    │    (⊗ → ∀)    │   │   (MERA,TT)   │   │     (◊)       │
    └───────┬───────┘   └───────┬───────┘   └───────┬───────┘
            │                   │                   │
            ▼                   ▼                   ▼
    ╔═══════════════╗   ╔═══════════════╗   ╔═══════════════╗
    ║  Holographic  ║   ║   Quantum     ║   ║  Evidential   ║
    ║   Principle   ║◄─►║   Mechanics   ║◄─►║     Types     ║
    ╚═══════════════╝   ╚═══════════════╝   ╚═══════════════╝

The unified operators:
- ∇ : Gradient (backprop, quantum parameter shift)
- ⊗ : Encoding (training, entanglement)
- ∀ : Reconstruction (inference, measurement)
- ◊ : Uncertainty (predictions, superposition)
```

Neural networks are:
1. **Holographic** — Weights compress training data like boundary encodes bulk
2. **Quantum-adjacent** — Tensor networks, variational circuits, uncertainty
3. **Evidentially uncertain** — Outputs are inherently `◊`, not `!`

The same operators unify all three domains.
