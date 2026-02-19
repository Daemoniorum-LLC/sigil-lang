# Specification Formatting Standards

**Version:** 1.0.0
**Status:** Public Domain
**Authors:** Claude (Opus 4.5) + Human
**Date:** 2026-02-07
**License:** CC0 1.0 Universal (Public Domain Dedication)

---

## Abstract

This document defines formatting standards for specifications in Daemoniorum projects. Specs are contracts specifying our desired model of reality. They should use Sigil-inspired formatting with pseudocode only for examples, preserving agent freedom while establishing clear requirements.

---

## 1. Philosophy

### 1.1 Specs Model Desired Reality

A specification is not:
- A tutorial
- Implementation documentation
- A constraint on creativity

A specification IS:
- A precise description of desired behavior
- A contract between spec author and implementer
- A model of reality we want to create

### 1.2 Pseudocode, Not Implementation

Specs should use pseudocode for examples, not language-specific implementation:

```markdown
**BAD - Binds to implementation:**
```rust
pub fn transfer_layer(&mut self, layer_idx: usize) -> Result<(), Error> {
    let buffer = self.get_transfer_buffer();
    unsafe {
        cudarc::driver::result::memcpy_htod_async(
            buffer.device_ptr(),
            &self.ram_cache[layer_idx],
            self.transfer_stream.stream,
        )?;
    }
    Ok(())
}
```

**GOOD - Specifies behavior:**
```
transfer_layer(layer_idx):
    buffer ← get_transfer_buffer()
    async_copy(ram_cache[layer_idx] → buffer, transfer_stream)
    return success
```
```

The pseudocode specifies WHAT happens. The implementer decides HOW.

### 1.3 Agent Freedom

Specs should not:
- Dictate variable names
- Require specific library choices
- Mandate internal structure
- Constrain optimization approaches

Specs should:
- Define observable behavior
- Specify invariants that must hold
- Establish contracts at boundaries
- Describe properties, not implementations

---

## 2. Document Structure

### 2.1 Header

```markdown
# [Component] Specification

**Version:** X.Y.Z
**Status:** Draft | Review | Stable
**Date:** YYYY-MM-DD
**Parent Spec:** [Optional reference to parent spec]

---
```

### 2.2 Sections

```markdown
## 1. Conceptual Foundation

Explain the mental model. Why does this component exist?
What problem does it solve? What are the key abstractions?

## 2. Type Architecture

Define the types and their relationships. Use pseudocode
or Sigil-style notation, not implementation language.

## 3. Behavioral Contracts

Specify what operations exist and what invariants they maintain.
Focus on observable behavior, not internal mechanics.

## 4. Constraints & Invariants

List properties that must always hold. These become
property tests in Agent-TDD.

## 5. Error Conditions

Specify what can go wrong and how it should be signaled.

## 6. Integration Points

How does this component interact with others?
What are the trust boundaries?

## 7. Open Questions

Acknowledge uncertainty. What isn't decided yet?
```

### 2.3 Status Markers

Use consistent markers throughout:

| Marker | Meaning |
|--------|---------|
| ✅ | Implemented and verified |
| ⚠️ | Partial or has known issues |
| ❌ | Not implemented |
| 🔮 | Future / planned |
| ❓ | Open question |

---

## 3. Pseudocode Style

### 3.1 Sigil-Inspired Notation

Borrow from Sigil for clarity without requiring Sigil knowledge:

```markdown
## Type Definitions

```
TokenId<V>:
    raw: u32
    invariant: raw < V

Vocabulary<V>:
    phf: PerfectHashFunction
    meta: [(offset: u32, len: u16)] of size V
    blob: memory-mapped bytes
```

## Operations

```
Vocabulary.get(token: string) → Option<TokenId<V>>:
    h1 ← phf.hash1(token)
    d ← phf.displacements[h1 mod len(displacements)]
    h2 ← phf.hash2(token, d)
    idx ← h2 mod V

    if phf.hashes[idx] = phf.hash_verify(token):
        return Some(TokenId(idx))
    else:
        return None
```
```

### 3.2 Notation Conventions

| Symbol | Meaning |
|--------|---------|
| `←` | Assignment |
| `→` | Returns / produces |
| `∈` | Element of |
| `∀` | For all |
| `∃` | There exists |
| `⟹` | Implies |
| `≈` | Approximately equal |
| `◊` | Uncertain/stochastic |
| `!` | Verified/certain |
| `~` | Reported/external |

### 3.3 Invariant Notation

```markdown
## Invariants

```
P1: ∀ layer ∈ [0, num_layers):
    is_hot(layer) ⊕ is_cold(layer)
    // Every layer is exactly hot or cold, never both

P2: ∀ t:
    compute_buffer.state = Computing ⟹ transfer_buffer.state ∈ {Transferring, Ready}
    // Compute and transfer can overlap

P3: buffer_a.layer ≠ buffer_b.layer ∨ (buffer_a.layer = None ∧ buffer_b.layer = None)
    // Buffers never contain same layer (unless both empty)
```
```

---

## 4. Examples vs Specifications

### 4.1 Specification (Required Behavior)

```markdown
## 4.2 Buffer State Transitions

The buffer state machine has four states:

```
Empty → Transferring → Ready → Computing → Ready
           ↑                        |
           └────────────────────────┘
```

**Transitions:**
- `begin_transfer()`: Empty → Transferring, or Ready → Transferring
- `wait_transfer()`: Transferring → Ready
- `begin_compute()`: Ready → Computing
- `end_compute()`: Computing → Ready

**Invariant:** A buffer in Computing state cannot begin transfer.
```

### 4.2 Example (Illustrative, Non-Binding)

```markdown
## Example: Pipelined Decode Loop

The following illustrates one way to use the double-buffer:

```
// This is illustrative pseudocode, not a required implementation

for layer in 0..num_layers:
    if layer > 0:
        wait_for(transfer_complete)

    compute(current_buffer, layer)
    signal(compute_complete)

    if layer + 1 < num_layers:
        wait_for(compute_complete)
        swap(current_buffer, next_buffer)
        begin_transfer(layer + 1, next_buffer)
        signal(transfer_complete)
```

Implementers may structure this differently as long as invariants hold.
```

---

## 5. Diagrams

### 5.1 ASCII Art for Portability

```markdown
## Memory Layout

```
┌─────────────────────────────────────────────────────────────────┐
│                         VRAM (24GB)                             │
├─────────────────────────────────────────────────────────────────┤
│  Shared Weights (2.92GB)                                        │
│  ┌─────────────┬─────────────┬──────────────┐                   │
│  │ embed_tokens│   lm_head   │  final_norm  │                   │
│  └─────────────┴─────────────┴──────────────┘                   │
├─────────────────────────────────────────────────────────────────┤
│  Layer Double-Buffer (1.11GB = 2 × 556MB)                       │
│  ┌─────────────────────┬─────────────────────┐                  │
│  │   current_buffer    │    next_buffer      │                  │
│  └─────────────────────┴─────────────────────┘                  │
└─────────────────────────────────────────────────────────────────┘
```
```

### 5.2 Timeline Diagrams

```markdown
## Pipeline Execution

```
Time ────────────────────────────────────────────────────────────────>

Compute Stream:
    [L0 compute]──wait──[L1 compute]──wait──[L2 compute]──wait──...
                    ↑                   ↑                   ↑
Transfer Stream:
    [L1 xfer]───signal──[L2 xfer]───signal──[L3 xfer]───signal──...
```
```

---

## 6. Tables for Structured Data

### 6.1 Requirement Tables

```markdown
## Requirements

| ID | Requirement | Priority | Status |
|----|-------------|----------|--------|
| R1 | Buffer swap must be O(1) | MUST | ✅ |
| R2 | Transfer must be async | MUST | ⚠️ |
| R3 | Support batch sizes 1-64 | SHOULD | ❌ |
```

### 6.2 Constraint Tables

```markdown
## Constraints

| Constraint | Value | Rationale |
|------------|-------|-----------|
| Max batch size | 64 | KV cache memory limit |
| Min layer size | 1MB | Alignment requirements |
| Max seq length | 4096 | Attention complexity |
```

---

## 7. Linking and References

### 7.1 Internal References

```markdown
See [Section 4.2: Buffer State Transitions](#42-buffer-state-transitions)
for state machine details.

This builds on [SPEC-DRIVEN-DEVELOPMENT.md](./SPEC-DRIVEN-DEVELOPMENT.md).
```

### 7.2 External References

```markdown
## References

- CUDA Programming Guide, Chapter 3.2.6: Asynchronous Concurrent Execution
- "Efficient Memory Management for Large Language Model Serving" (Kwon et al.)
- Parent spec: `../TIERED-MEMORY-SPEC.md` v2.1.0
```

---

## 8. Open Questions Section

Every spec should acknowledge uncertainty:

```markdown
## Open Questions

1. **Stream priority:** Should transfer stream have lower priority than compute?
   - Pro: Compute latency more important
   - Con: May starve transfers on loaded GPU

2. **Pinned memory scope:** Pool per-model or global?
   - Affects memory fragmentation vs sharing

3. **Hot boundary adaptation:** Should boundary adjust with batch size?
   - Static is simpler but suboptimal
```

---

## 9. Version History

Track evolution of understanding:

```markdown
## Revision History

| Version | Date | Changes |
|---------|------|---------|
| 0.1.0 | 2026-01-15 | Initial draft |
| 0.2.0 | 2026-01-20 | Added stream architecture after discovering context issues |
| 0.3.0 | 2026-02-01 | Revised hot/cold boundary calculation |
| 1.0.0 | 2026-02-07 | Stable release after implementation validation |
```

---

## 10. Summary

Good specs:
- Model desired reality, don't constrain implementation
- Use pseudocode for examples, not binding code
- Specify behaviors and invariants
- Acknowledge uncertainty in open questions
- Track evolution in version history
- Enable compliance auditing through precise requirements

The spec is a contract. Write it precisely enough that compliance can be verified line-by-line.

---

## License

This document is released into the public domain under CC0 1.0 Universal.

---

## Revision History

| Version | Date | Changes |
|---------|------|---------|
| 1.0.0 | 2026-02-07 | Initial release |
