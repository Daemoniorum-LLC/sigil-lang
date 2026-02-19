# Sigil Evidentiality System - Definitive Reference

> **TODO**: Sync this documentation to sigil-lang.com wherever evidentiality markers are explained. Ensure all 6 markers (`!`, `?`, `◊`, `⁂`, `~`, `‽`) are documented with their semantic meanings.

## Overview

Evidentiality is Sigil's type-level tracking of data provenance and trustworthiness. It prevents whole categories of security vulnerabilities by ensuring untrusted data cannot be used where trusted data is required.

## Evidence Levels

| Level | Symbol | Name | Trust | Use Case |
|-------|--------|------|-------|----------|
| Known | `!` | Direct | Full | Computed locally, validated |
| Uncertain | `?` | Possible | Partial | May be absent (like Option) |
| Predicted | `◊` | Speculative | Partial | AI/ML model outputs, forecasts |
| Chaos | `⁂` | Entropic | Partial | Intentional randomness, RNG, sampling |
| Reported | `~` | External | None | From network, user input, files |
| Paradox | `‽` | Boundary | Explicit | Trust assertions, FFI |

**Note:** `Predicted (◊)`, `Uncertain (?)`, and `Chaos (⁂)` share the same trust level in the type checker but have distinct semantic meanings:
- `?` — existence uncertainty (value may be absent)
- `◊` — inference uncertainty (model made a pattern-based guess)
- `⁂` — entropic uncertainty (intentionally random, patternless by design)

## Syntax Forms

**Both PREFIX and SUFFIX forms are valid and semantically equivalent:**

```sigil
// SUFFIX form (spec canonical)
let x: i32! = 42;
let y: String~ = fetch(url);
fn compute() -> i32! { ... }

// PREFIX form (also valid)
let x: !i32 = 42;
let y: ~String = fetch(url);
fn compute() -> !i32 { ... }
```

**Recommendation:** Use SUFFIX form (`T!`) as it matches the spec examples.

## Evidence Placement

### 1. Type Annotations

```sigil
// On variable types
let known_value: i32! = compute();
let external_data: String~ = fetch(url);
let maybe_value: User? = db.find(id);

// On struct fields
struct User {
    id: u64!,        // Known: system-generated
    name: String!,   // Known: required, validated
    email: String?,  // Uncertain: optional
    bio: String~,    // Reported: user-provided
}

// On function parameters
fn process(data: Data!) { ... }      // Requires known data
fn handle(input: String~) { ... }    // Accepts external data

// On return types
fn compute() -> i32! { ... }         // Returns known value
fn lookup(k: K) -> V? { ... }        // May not find value
fn fetch(url: str) -> Data~ { ... }  // Returns external data
```

### 2. Binding Patterns

```sigil
// Explicit evidence on binding (overrides inference)
let x! = possibly_uncertain;  // Assert as Known
let y~ = compute();           // Mark as Reported

// Evidence is inferred from initializer
let a = compute();            // Inferred from compute()'s return type
let b = fetch(url);           // Inferred as Reported if fetch returns T~
```

### 3. Function Names

```sigil
// Evidence on function name indicates evidence transformation
fn validate!(input: Data~) -> Data! {
    // Validates and promotes Reported → Known
}

fn assume!(data: Data~, reason: str) -> Data! {
    // Trust assertion (for auditing)
}
```

## Evidence Lattice

Evidence forms a lattice with subtyping:

```
                Known (!)                    ← Most certain
                    ↓
    Uncertain (?) = Predicted (◊) = Chaos (⁂)
                    ↓
               Reported (~)
                    ↓
               Paradox (‽)                   ← Least certain
```

**Note:** Three markers share the "Uncertain" trust level but have distinct semantic meanings:
- `?` — optional/nullable values (existence uncertainty)
- `◊` — AI/ML outputs (inference uncertainty)
- `⁂` — random/stochastic values (entropic uncertainty)

**Subtyping:** More certain evidence satisfies less certain requirements.

```sigil
fn accept_uncertain(x: i32?) { }
fn accept_reported(x: i32~) { }

let known: i32! = 42;
accept_uncertain(known);  // OK: i32! <: i32?
accept_reported(known);   // OK: i32! <: i32~
```

## Evidence Propagation

**Binary operations combine evidence pessimistically (join):**

```sigil
let a: i32! = 10;
let b: i32? = lookup(key);
let c: i32~ = fetch(url);

let sum1 = a + b;  // i32? (Known + Uncertain = Uncertain)
let sum2 = a + c;  // i32~ (Known + Reported = Reported)
let sum3 = b + c;  // i32? (Uncertain + Reported = Uncertain)
```

**Propagation table:**

| Op | `!` | `?`/`◊`/`⁂` | `~` | `‽` |
|----|-----|-------------|-----|-----|
| `!` | `!` | `?` | `~` | `‽` |
| `?`/`◊`/`⁂` | `?` | `?` | `?` | `‽` |
| `~` | `~` | `?` | `~` | `‽` |
| `‽` | `‽` | `‽` | `‽` | `‽` |

**Note:** `◊` (Predicted), `?` (Uncertain), and `⁂` (Chaos) behave identically in propagation — they share the same trust level.

## Evidence Transitions

### Promotion (making more certain)

```sigil
// Reported → Known via validation
let trusted: Data! = external~|validate!{x => x.is_valid()};

// Uncertain → Known via unwrap
let value: i32! = maybe?!;  // Panics if absent
let value: i32! = maybe? else { return };  // Early return if absent

// Match promotes through exhaustive handling
let value: i32! = maybe? match {
    Some(v) => v,
    None => 0,
};
```

### Demotion (marking less certain)

```sigil
// Known → Uncertain (wrapping in Option)
let optional: i32? = Some(known!);

// Known → Reported (marking as external)
let external: Data~ = Data { ...known }~;
```

## Type Checker Enforcement

The type checker:

1. **Tracks evidence for each binding:** `(Type, EvidenceLevel)`
2. **Checks evidence compatibility at function calls:**
   ```sigil
   fn requires_known(x: i32!) { }
   requires_known(uncertain?);  // ERROR: evidence mismatch
   ```
3. **Propagates evidence through expressions:**
   - Binary ops join evidence
   - Field access preserves evidence
   - Match arms join evidence
4. **Validates evidence transitions:**
   - `|validate!{...}` can only promote
   - Cannot silently demote evidence

## Common Patterns

### External Data Handling

```sigil
// Fetch returns reported data
fn fetch_user(id: u64) -> User~ {
    http·get(format!("/users/{}", id))~
}

// Validate before use
fn process_user(id: u64) -> Result<Report!, Error~>! {
    let user~ = fetch_user(id);

    // Validate and promote
    let trusted! = user~|validate!{u =>
        u.id > 0 && u.name.len() > 0
    };

    // Now safe to use in trusted contexts
    generate_report(trusted!)
}
```

### Optional Values

```sigil
fn find_user(id: u64) -> User? {
    db.query("SELECT * FROM users WHERE id = ?", id)?
}

fn get_user_name(id: u64) -> String! {
    let user? = find_user(id);

    // Handle absence explicitly
    user? match {
        Some(u) => u.name,
        None => "Unknown".to_string(),
    }
}
```

### AI/ML Predictions

```sigil
// Model outputs are predicted - inherently uncertain
fn classify(image: Image!) -> Label◊ {
    model.predict(image)◊
}

// Predictions need validation before trusted use
fn process_classification(image: Image!) -> Result<Action!, Error~>! {
    let prediction◊ = classify(image);

    // Check confidence before promoting
    if prediction◊.confidence > 0.95 {
        let label! = prediction◊.label|validate!{l => l.is_valid()};
        Ok(take_action(label!))
    } else {
        // Low confidence - require human review
        Err(Error::NeedsReview(prediction◊))
    }
}

// Ensemble predictions combine multiple model outputs
fn ensemble_predict(data: Data!) -> Prediction◊ {
    let p1◊ = model_a.predict(data)◊;
    let p2◊ = model_b.predict(data)◊;
    let p3◊ = model_c.predict(data)◊;

    // Combined prediction is still predicted
    vote(p1◊, p2◊, p3◊)◊
}
```

### Stochastic Values

```sigil
// Random number generation returns chaotic values
fn rand_f32⁂() -> f32⁂ {
    rng.next_f32()⁂
}

// Fill tensor with random values - output is entropic
fn fill_randn⁂(tensor: &mut Tensor!) {
    for i in 0..tensor.len() {
        tensor[i] = randn()⁂;
    }
}

// Monte Carlo integration - result inherits chaos
fn monte_carlo_pi⁂(samples: usize!) -> f64⁂ {
    let mut inside⁂ = 0;
    for _ in 0..samples {
        let x⁂ = rand_f64()⁂;
        let y⁂ = rand_f64()⁂;
        if x⁂ * x⁂ + y⁂ * y⁂ <= 1.0 {
            inside⁂ += 1;
        }
    }
    4.0 * (inside⁂ as f64) / (samples as f64)
}

// Stochastic gradient descent - minibatch selection is chaotic
fn sgd_step⁂(model: &mut Model!, data: &Dataset!, batch_size: usize!) {
    let batch⁂ = data.random_sample(batch_size)⁂;
    let gradients⁂ = compute_gradients(model, batch⁂);
    model.update(gradients⁂);
}

// Dropout during training - mask is entropic
fn dropout⁂(x: Tensor!, p: f32!) -> Tensor⁂ {
    let mask⁂ = random_mask(x.shape(), p)⁂;
    x * mask⁂  // Result is chaotic due to random mask
}

// Cryptographic key generation - must be entropic
fn generate_key⁂(bits: usize!) -> Key⁂ {
    let entropy⁂ = secure_random_bytes(bits / 8)⁂;
    Key::from_bytes(entropy⁂)⁂
}
```

**Note:** Chaos (`⁂`) differs from Predicted (`◊`) in that:
- `◊` implies pattern-based inference (a model learned something)
- `⁂` implies intentional patternlessness (entropy by design)

Both are uncertain, but for opposite reasons: predictions try to find patterns; chaos ensures there are none.

### Trust Boundaries

```sigil
// FFI data is paradoxical - we must trust it
unsafe fn read_native(ptr: *void) -> Data‽ {
    // Reading raw memory is a trust boundary
    *(ptr as *Data)‽
}

// Promote with explicit assumption
fn use_native(ptr: *void) -> Data! {
    let raw‽ = read_native(ptr);
    raw‽|assume!("FFI contract guarantees validity")
}
```

## Security Guarantees

Evidentiality prevents:

1. **Injection attacks:** External strings can't be used as SQL/commands
2. **TOCTOU bugs:** Uncertain values must be re-checked
3. **Trust confusion:** External data visibly different from trusted
4. **Silent failures:** Absence must be explicitly handled

```sigil
// PREVENTED: SQL injection
fn bad_query(user_input~: String~) {
    // ERROR: String~ cannot be used where String! required
    db.execute(format!("SELECT * FROM users WHERE name = '{}'", user_input~));
}

// SAFE: Validated input
fn safe_query(user_input~: String~) {
    let sanitized! = user_input~|validate!{s => is_safe_sql_string(s)};
    db.execute(format!("SELECT * FROM users WHERE name = '{}'", sanitized!));
}
```

## Native Runtime Guidelines

For the native runtime (syscalls, allocators), evidence rules are:

1. **Syscall returns:** Always `~` or `?` (external/may fail)
2. **Pointer arithmetic:** `!` (computed locally)
3. **Memory contents:** `~` (came from kernel/external)
4. **Validated results:** `!` after explicit checks

```sigil
// Syscall returns reported data
pub fn read(fd: i32!, buf: *mut u8!, len: u64!) -> Result<u64!, Errno~>~ {
    // Returns reported because kernel provides the data
}

// After validation, can assert known
pub fn read_exact(fd: i32!, buf: *mut u8!, len: u64!) -> Result<(), Errno~>! {
    let n~ = read(fd, buf, len)?;
    if n~ as u64 != len {
        return Err(Errno·EIO);
    }
    // We validated the read succeeded, result is known
    Ok(())!
}
```
