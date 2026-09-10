# NIHIL-CUDA-BACKWARD-SPEC

**Version:** 0.1.0
**Status:** ⚠️ Draft — under active development
**Date:** 2026-02-23
**Methodology:** SDD v1.1.0 + Agent-TDD v1.0.0
**Parent Spec:** SIGIL-NATIVE-RUNTIME-SPEC.md
**TDD Roadmap:** NIHIL-CUDA-BACKWARD-TDD-ROADMAP.md

---

## 1. Conceptual Foundation

### 1.1 Purpose

This spec covers the CUDA kernel interface required to add a **backward pass** to
the Nihil GPU training framework. The forward pass is fully operational
(`sigil_cuda_attn_fwd_f32` and friends in `sigil_runtime_cuda.c`). Training
requires gradient computation through every forward operation.

The backward kernels follow the same design contract as the existing forward
kernels: **device-pointer interface** — all tensor arguments are `int64_t` raw
CUDA device pointers; no Sigil fat-pointer ABI is involved.

### 1.2 Scope

**In scope:**
- Attention backward (combined fwd-store + bwd kernels) — `attn_fwd_store`, `attn_bwd`
- Cross-entropy softmax backward — `ce_backward`
- RMSNorm backward — `rmsnorm_backward`
- SwiGLU backward — `swiglu_backward`
- Embedding scatter-add — `embed_scatter`
- GPU AdamW step — `adamw_step`

**Out of scope:**
- SGEMM backward (already handled: `sgemm_nt_raw` / `sgemm_nn_raw` cover
  both weight gradients and input gradients)
- Residual add backward (identity: gradient passes unchanged, no kernel needed)
- Host-side gradient accumulation (handled in Sigil caller)
- Autograd tape / automatic differentiation (manual wiring in Sigil)

### 1.3 Tensor Memory Layout

All tensors use **row-major (C-contiguous)** layout, consistent with the
existing forward kernels.

```
Q, K, V, dQ, dK, dV — shape [B*S, H*HD], stride [H*HD, 1]
  Access: ptr + (b*S + s) * (H*HD) + h*HD + d
  where B=batch, S=seq_len, H=heads, HD=head_dim, D_MODEL=H*HD

logits, d_logits   — shape [B*S, V]   stride [V, 1]
hidden, d_hidden   — shape [B*S, D]   stride [D, 1]
wte, d_wte         — shape [V, D]     stride [D, 1]

probs (attention)  — shape [B*H, S, S] stride [S*S, S, 1]
  Access: ptr + (b*H + h)*S*S + s*S + t
```

---

## 2. Type Architecture

### 2.1 Function Signatures (C ABI)

All functions are declared `extern "C"` and registered in
`declare_runtime_functions` in `llvm_codegen.rs`. All pointer arguments are
`int64_t` (raw CUDA device pointers cast to i64). All shape/count arguments
are `int64_t`.

```
// ── Attention ──────────────────────────────────────────────────────────────

// Forward pass that also saves attention probabilities for backward.
// probs_ptr: caller-allocated [B*H, S, S] f32 buffer (≈ 1MB for B=4,H=4,S=128)
void sigil_cuda_attn_fwd_store_f32(
    int64_t out_ptr, int64_t probs_ptr,
    int64_t q_ptr, int64_t k_ptr, int64_t v_ptr,
    int64_t batch, int64_t seq, int64_t heads, int64_t hd);

// Backward pass. Reads stored probabilities; writes dQ, dK, dV.
// dK and dV must be zeroed by caller before this call (atomicAdd accumulates).
// dQ is written (not accumulated); no pre-zeroing required.
void sigil_cuda_attn_bwd_f32(
    int64_t dq_ptr, int64_t dk_ptr, int64_t dv_ptr,
    int64_t probs_ptr,
    int64_t q_ptr, int64_t k_ptr, int64_t v_ptr,
    int64_t dout_ptr,
    int64_t batch, int64_t seq, int64_t heads, int64_t hd);

// ── Cross-Entropy ──────────────────────────────────────────────────────────

// Backward through CE loss. targets are int64_t token IDs.
// d_logits[t,v] ← (softmax(logits[t])[v] − δ(v, targets[t])) / batch_seq
// d_logits must be pre-zeroed or overwritten (kernel writes, not adds).
void sigil_cuda_ce_backward_f32(
    int64_t d_logits_ptr,
    int64_t logits_ptr, int64_t targets_ptr,
    int64_t batch_seq, int64_t vocab);

// ── RMSNorm ───────────────────────────────────────────────────────────────

// Backward through RMSNorm. Recomputes rms_recip from x (no saved activations).
// dx and dw must be pre-zeroed (dw accumulates atomically across rows).
void sigil_cuda_rmsnorm_backward_f32(
    int64_t dx_ptr, int64_t dw_ptr,
    int64_t dout_ptr, int64_t x_ptr, int64_t w_ptr,
    int64_t rows, int64_t d_model, int64_t eps_bits);

// ── SwiGLU ────────────────────────────────────────────────────────────────

// Backward through SwiGLU: out = silu(gate) * up
// d_gate[i] = d_out[i] * up[i] * σ(gate[i]) * (1 + gate[i] * (1 − σ(gate[i])))
// d_up[i]   = d_out[i] * silu(gate[i])
// Kernel writes d_gate and d_up directly (no race; one thread per element).
void sigil_cuda_swiglu_backward_f32(
    int64_t d_gate_ptr, int64_t d_up_ptr,
    int64_t d_out_ptr, int64_t gate_ptr, int64_t up_ptr,
    int64_t n);

// ── Embedding ─────────────────────────────────────────────────────────────

// Scatter-add: d_wte[ids[t], :] += d_hidden[t, :] for t in [0, batch_seq).
// d_wte must be pre-zeroed. Uses atomicAdd (safe for concurrent same-token updates).
void sigil_cuda_embed_scatter_f32(
    int64_t d_wte_ptr,
    int64_t ids_ptr, int64_t d_hidden_ptr,
    int64_t batch_seq, int64_t d_model);

// ── Optimizer ────────────────────────────────────────────────────────────

// In-place AdamW step on device pointers.
// w[i]  ← w[i] * (1 − lr * wd) − lr * m_hat[i] / (sqrt(v_hat[i]) + eps)
// m[i]  ← β1 * m[i] + (1 − β1) * g[i]
// v[i]  ← β2 * v[i] + (1 − β2) * g[i]²
// Bias correction: m_hat = m / (1 − β1^step),  v_hat = v / (1 − β2^step)
// All scalar hyperparams encoded as f32 bits in int64_t (same ABI as fill_const).
void sigil_cuda_adamw_step_f32(
    int64_t w_ptr, int64_t g_ptr, int64_t m_ptr, int64_t v_ptr,
    int64_t n,
    int64_t step,       // current step number (1-based for bias correction)
    int64_t lr_bits,    // f32 bits: learning rate
    int64_t wd_bits,    // f32 bits: weight decay
    int64_t b1_bits,    // f32 bits: beta1 (default 0.9)
    int64_t b2_bits,    // f32 bits: beta2 (default 0.999)
    int64_t eps_bits);  // f32 bits: epsilon (default 1e-8)
```

### 2.2 Probs Buffer Sizing

The `probs_ptr` buffer for attention must be pre-allocated by the caller:

```
probs_bytes ← B × H × S × S × sizeof(f32)
            = batch × heads × seq × seq × 4
```

For the default config (B=4, H=4, S=128): `4 × 4 × 128 × 128 × 4 = 1,048,576 bytes` (1 MB).

---

## 3. Behavioral Contracts

### 3.1 Attention Forward-Store (`attn_fwd_store`)

**Preconditions:**
- `q_ptr`, `k_ptr`, `v_ptr` point to valid device buffers of shape `[B*S, H*HD]`
- `out_ptr` points to writable device buffer of shape `[B*S, H*HD]`
- `probs_ptr` points to writable device buffer of size `B*H*S*S` f32 elements

**Postconditions:**
- `out_ptr` ← causal self-attention output (identical to `sigil_cuda_attn_fwd_f32`)
- `probs_ptr[b*H+h, s, t]` ← `P[b,h,s,t]` for `t ≤ s`, zero for `t > s`

**Kernel semantics (per thread (bh, s) where bh = b*H + h):**
```
scale ← rsqrt(hd)
for t in 0..=s:
    score[t] ← dot(Q[b,s,h,:], K[b,t,h,:]) * scale
for t in s+1..S:
    score[t] ← -inf
P[0..S]  ← softmax(score[0..S])           // causal: P[t>s] = 0
out[b,s,h,:] ← Σ_t P[t] * V[b,t,h,:]
probs[bh, s, 0..S] ← P[0..S]             // ← NEW vs. attn_fwd_f32
```

**Invariants:**
- `! Σ_t P[b,h,s,t] = 1.0` for all (b, h, s)
- `! P[b,h,s,t] = 0` for `t > s` (causal mask)
- `! P[b,h,s,t] ≥ 0` for all entries

### 3.2 Attention Backward (`attn_bwd`)

**Preconditions:**
- `probs_ptr` contains the saved output of `attn_fwd_store` for the same Q/K/V
- `dq_ptr`, `dk_ptr`, `dv_ptr` point to writable device buffers of shape `[B*S, H*HD]`
- `dk_ptr` and `dv_ptr` are **zeroed** by caller before this call
- `dq_ptr` is **not** required to be pre-zeroed (kernel writes directly)

**Postconditions:**
- `dq_ptr` ← `∂L/∂Q` accumulated from backward through attention
- `dk_ptr` ← `∂L/∂K` (accumulated with atomicAdd)
- `dv_ptr` ← `∂L/∂V` (accumulated with atomicAdd)

**Kernel semantics (per thread (bh, s)):**
```
P[0..S] ← probs[bh, s, 0..S]                // load saved probabilities

// 1. Gradient wrt V (scatter)
for t in 0..=s:
    for d in 0..HD:
        atomicAdd(dV[b,t,h,d], P[t] * dOut[b,s,h,d])

// 2. Gradient wrt P (dot product with dOut and V)
for t in 0..=s:
    dP[t] ← dot(dOut[b,s,h,:], V[b,t,h,:])

// 3. Softmax backward
dp_sum ← Σ_{t=0}^{s} P[t] * dP[t]
for t in 0..=s:
    dScore[t] ← P[t] * (dP[t] − dp_sum) * scale  // scale = 1/sqrt(HD)

// 4. Gradient wrt Q (direct write, no race)
dQ[b,s,h,:] ← Σ_{t=0}^{s} dScore[t] * K[b,t,h,:]

// 5. Gradient wrt K (scatter)
for t in 0..=s:
    for d in 0..HD:
        atomicAdd(dK[b,t,h,d], dScore[t] * Q[b,s,h,d])
```

**Correctness note:** `dScore` includes the `1/√HD` scaling factor so that the
chain rule through `score = Q·K^T / √HD` is fully accounted for.

**Invariants:**
- `! Σ_d dQ[b,s,h,d]` is finite for all (b,s,h)
- Finite difference check: `◊ ‖analytical_grad − numerical_grad‖ / max(‖analytical‖, ε) < 1e-3`

### 3.3 CE Backward (`ce_backward`)

**Kernel semantics (per thread block, one block per token t):**
```
// Block of 'vocab' threads computes softmax then subtracts one-hot
max_val ← max over v of logits[t, v]           // reduce in shared mem
Σ_exp   ← Σ_v exp(logits[t,v] − max_val)
for v assigned to this thread:
    sm_v ← exp(logits[t,v] − max_val) / Σ_exp
    d_logits[t,v] ← (sm_v − (v == targets[t] ? 1.0 : 0.0)) / batch_seq
```

**Invariants:**
- `! Σ_v d_logits[t,v] = 0` for every token t (sum of CE gradient is zero)
- `! d_logits[t, targets[t]] < 0` when model assigns low probability to correct token

### 3.4 RMSNorm Backward (`rmsnorm_backward`)

**Math:**
```
rms     ← sqrt(mean(x²) + eps)         // recomputed from x
r       ← 1 / rms                      // rms_recip
// Forward was: out = (x * r) * w

// dx: gradient through normalization + scale
xn[d]   ← x[d] * r                    // normalized x
s_dot   ← Σ_d dout[d] * w[d] * xn[d] // scalar: 'dy · w · xn'
for d in 0..D:
    dx[d] ← r * (dout[d] * w[d] - xn[d] * s_dot / D)

// dw: gradient through per-dimension scale (accumulate across rows)
for d in 0..D:
    atomicAdd(dw[d], dout[d] * xn[d])
```

**Preconditions:** `dx_ptr` and `dw_ptr` must be pre-zeroed by caller.

**Invariants:**
- `! Σ_d dx[d] ≈ 0` (gradient is orthogonal to normalization direction)

### 3.5 SwiGLU Backward (`swiglu_backward`)

**Math (per element i):**
```
σ(g)    ← 1 / (1 + exp(-gate[i]))
silu(g) ← gate[i] * σ(g)

d_gate[i] ← d_out[i] * up[i] * σ(g) * (1 + gate[i] * (1 - σ(g)))
d_up[i]   ← d_out[i] * silu(g)
```

**Thread structure:** one thread per element, `n = B*S*D_FF`.
No race conditions; kernel writes `d_gate` and `d_up` directly.

**Invariants:**
- `! |d_gate[i]|` is bounded when `|gate[i]|` is bounded (sigmoid derivative ≤ 0.25)

### 3.6 Embedding Scatter (`embed_scatter`)

**Math:**
```
for t in 0..batch_seq:
    for d in 0..d_model:
        atomicAdd(d_wte[ids[t], d], d_hidden[t, d])
```

**Thread structure:** Grid = `(batch_seq * d_model / BLOCK, )`, Block = `BLOCK`.
Each thread handles one (t, d) pair.

**Preconditions:** `d_wte_ptr` must be pre-zeroed by caller.

**Invariants:**
- `! Σ_{t: ids[t]=v} d_hidden[t,:]` = `d_wte[v,:]` (correct scatter-add)

### 3.7 AdamW Step (`adamw_step`)

**Math (per element i, standard Adam with decoupled weight decay):**
```
m[i] ← β1 * m[i] + (1 − β1) * g[i]
v[i] ← β2 * v[i] + (1 − β2) * g[i]²

m_hat ← m[i] / (1 − β1^step)
v_hat ← v[i] / (1 − β2^step)

w[i] ← w[i] * (1 − lr * wd) − lr * m_hat / (sqrt(v_hat) + eps)
```

**Thread structure:** one thread per element. All scalar hyperparams passed
as `int64_t` bit patterns encoding f32 (same ABI as `sigil_cuda_fill_const_f32`).

**Invariants:**
- `! |w[i]|` does not increase unboundedly (weight decay ensures contraction)
- `! v[i] ≥ 0` for all i (squares are non-negative; momentum preserves sign)

---

## 4. Constraints & Invariants

### 4.1 Global Constraints

```
C1: All device pointers are valid CUDA device memory (allocated via cuMemAlloc)
C2: Kernel launches are synchronous — cuCtxSynchronize() called after each launch
C3: Thread count ≤ 1024 per block (CUDA hardware limit; seq ≤ 1024 required)
C4: scores[] local array ≤ 512 elements (existing constraint from attn_fwd)
    → seq_len ≤ 512 for attention kernels
C5: AtomicAdd for f32 requires SM 2.0+ (all target GPUs satisfy this)
C6: Probs buffer must be allocated for B*H*S*S f32 elements
C7: dK and dV must be zeroed before attn_bwd_f32 call
C8: dw must be zeroed before rmsnorm_backward call
C9: d_wte must be zeroed before embed_scatter call
```

### 4.2 Numerical Constraints

```
N1: eps_bits for RMSNorm: use 0x358637BD (1e-6f) — same as forward pass
N2: eps_bits for AdamW: use 0x322BCC77 (1e-8f)
N3: Softmax computed with max-subtraction for numerical stability (already in fwd)
N4: CE backward recomputes softmax from logits (avoids storing activations)
N5: ◊ Finite difference gradient check tolerance: relative error < 1e-3
    (f32 precision; not the 1e-6 achievable with f64)
```

---

## 5. Error Conditions

| Condition | Behavior |
|-----------|----------|
| seq > 512 | Silent UB — scores array overflows local register file |
| Invalid device ptr | cuLaunchKernel returns error; printed to stderr |
| Kernel compile failure | fprintf to stderr; function is no-op |
| NaN in logits during CE bwd | NaN propagates through softmax → NaN gradients |
| step = 0 in adamw_step | Division by zero in bias correction; caller must pass step ≥ 1 |

---

## 6. Integration Points

### 6.1 Caller Responsibilities in Sigil

The backward pass in `stage7_nihil_gpu.sigil` must:
1. Allocate gradient buffers (same shape as weights) in `TransformerLM::new`
2. Allocate `probs_ptr` buffer of size `B*H*S*S` f32 in scratch buffers
3. Allocate AdamW moment buffers `m_*`, `v_*` (same shapes as weights, zero-initialized)
4. Call `sigil_cuda_zero_f32` on `dk_ptrs`, `dv_ptrs`, `dln*_ptrs`, `d_wte_ptr`,
   `d_final_norm_ptr` before each backward pass
5. Wire backward in reverse layer order vs. forward

### 6.2 Sigil extern "C" declarations

**No changes to `llvm_codegen.rs` are needed.** The LLVM backend processes
`extern "C"` blocks in `.sigil` source files via `declare_extern_function`,
which creates external LLVM declarations resolved at link time.

Each `.sigil` file that uses these kernels declares them in an `extern "C"` block:

```sigil
extern "C" {
    // Attention fwd+bwd
    rite sigil_cuda_attn_fwd_store_f32(out: i64, probs: i64,
                                        q: i64, k: i64, v: i64,
                                        batch: i64, seq: i64, heads: i64, hd: i64);
    rite sigil_cuda_attn_bwd_f32(dq: i64, dk: i64, dv: i64,
                                  probs: i64, q: i64, k: i64, v: i64, dout: i64,
                                  batch: i64, seq: i64, heads: i64, hd: i64);

    // Elementwise backward kernels
    rite sigil_cuda_ce_backward_f32(d_logits: i64, logits: i64, targets: i64,
                                     batch_seq: i64, vocab: i64);
    rite sigil_cuda_rmsnorm_backward_f32(dx: i64, dw: i64, dout: i64, x: i64, w: i64,
                                          rows: i64, d_model: i64, eps_bits: i64);
    rite sigil_cuda_swiglu_backward_f32(d_gate: i64, d_up: i64,
                                         d_out: i64, gate: i64, up: i64, n: i64);
    rite sigil_cuda_embed_scatter_f32(d_wte: i64, ids: i64, d_hidden: i64,
                                       batch_seq: i64, d_model: i64);

    // Optimizer
    rite sigil_cuda_adamw_step_f32(w: i64, g: i64, m: i64, v: i64, n: i64,
                                    step: i64, lr_bits: i64, wd_bits: i64,
                                    b1_bits: i64, b2_bits: i64, eps_bits: i64);
}
```

The link-time resolution uses `libsigil_runtime_cuda.a`.

### 6.3 Runtime Build

After any change to `sigil_runtime_cuda.c`:
```bash
cd /home/lilith/development/projects/sigil-lang/parser/runtime
make cuda
cp libsigil_runtime_cuda.a /home/lilith/development/projects/lucifer/experiments/spectral_validation/exp002_compression_aware/runtime/
```

---

## 7. Open Questions

```
❓ Q1: Should scores[] local array size be bumped from 512 to match SEQ_LEN at
       compile time, or keep 512 as a fixed limit?
       → Current experiments use seq_len ≤ 256; 512 is sufficient.
       → If seq_len=512+ is ever needed, kernel must use global scratch memory.

❓ Q2: For rmsnorm_backward, should dw accumulation use __shared__ memory
       reduction instead of global atomicAdd?
       → AtomicAdd is correct; shared memory would be faster but complex.
       → Deferred: evaluate after correctness is validated.

❓ Q3: Should attn_bwd be split into two separate kernels (dV/dP kernel,
       then dQ/dK kernel) to avoid atomicAdd?
       → Single kernel with atomicAdd is simpler; correctness is the priority.
       → For S=128, atomic contention is acceptable.

❓ Q4: Should we store log-sum-exp (the softmax normalizer) in attn_fwd_store
       instead of the full P matrix to save memory?
       → 1MB is acceptable for current config. Deferred.
```

---

## 8. Revision History

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 0.1.0 | 2026-02-23 | Claude Sonnet 4.6 | Initial spec — all backward kernels for Nihil training loop |
