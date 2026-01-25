# Sigil Type Inference Specification

> *"To know a thing's true name is to know its nature completely."*

## 0. Design Philosophy: Agent-Native Inference

This specification assumes the primary consumer is an agent with:
- Unbounded working memory for constraint tracking
- Native facility with logical inference and SMT solving
- No fatigue from deep symbolic manipulation

We do not artificially constrain the type system to what humans can manually verify.
Tooling exists to help humans; the language exists to express truth.

---

## 1. Overview

Sigil's type inference is a **dependent, refinement-typed, bidirectional system** with:

1. **Dependent types** — Types may depend on runtime values
2. **Refinement types** — Types carry logical predicates
3. **Evidentiality integration** — Provenance tracked at type level
4. **SMT-backed constraint solving** — Theorem proving for type checking
5. **Bidirectional flow** — Information flows both up and down the AST

The system is **decidable** (with bounded refinement complexity) but not **simple**.

---

## 2. The Type Universe

### 2.1 Type Hierarchy

```
Universe₂
    │
    ▼
Universe₁ (types of types)
    │
    ├── Πx:A. B         (dependent function types)
    ├── Σx:A. B         (dependent pair types)
    ├── {x:A | φ}       (refinement types)
    ├── A^E             (evidential types)
    │
    ▼
Universe₀ (base types)
    │
    ├── i8, i16, i32, i64, i128
    ├── u8, u16, u32, u64, u128
    ├── f32, f64
    ├── bool, char, str
    ├── ()              (unit)
    └── ⊥               (bottom/never)
```

### 2.2 Type Syntax

```
τ ::=
    | α                           -- type variable
    | C                           -- type constant (i32, bool, etc.)
    | τ → τ                       -- simple function
    | Πx:τ. τ                     -- dependent function (Pi type)
    | Σx:τ. τ                     -- dependent pair (Sigma type)
    | {x:τ | φ}                   -- refinement type
    | τ^E                         -- evidential type
    | τ[e/x]                      -- type with term substitution
    | ∀α. τ                       -- universal quantification
    | ∃α. τ                       -- existential quantification
    | τ ∧ τ                       -- intersection type
    | τ ∨ τ                       -- union type
    | μα. τ                       -- recursive type
```

### 2.3 Refinement Predicates

Refinement predicates φ are drawn from a decidable fragment of first-order logic:

```
φ ::=
    | true | false
    | e₁ = e₂ | e₁ ≠ e₂
    | e₁ < e₂ | e₁ ≤ e₂ | e₁ > e₂ | e₁ ≥ e₂
    | φ ∧ φ | φ ∨ φ | ¬φ | φ ⟹ φ
    | ∀x:τ. φ | ∃x:τ. φ           -- bounded quantification
    | P(e₁, ..., eₙ)              -- uninterpreted predicates
```

**Examples:**
```sigil
type Nat = {x:i64 | x ≥ 0}
type Positive = {x:i64 | x > 0}
type Bounded<const L: i64, const H: i64> = {x:i64 | L ≤ x ∧ x ≤ H}
type NonEmpty<T> = {xs:[T] | len(xs) > 0}
type Sorted<T: Ord> = {xs:[T] | ∀i,j. i < j ⟹ xs[i] ≤ xs[j]}
```

---

## 3. Evidentiality as Type-Level Effect

### 3.1 Evidence Kinds

```
E ::= ! | ? | ~ | ‽ | ε
```

| Evidence | Meaning | Subtyping |
|----------|---------|-----------|
| `!` | Known (computed locally) | `!` <: all |
| `?` | Uncertain (may be absent) | `?` <: `?`, `‽` |
| `~` | Reported (external source) | `~` <: `~`, `‽` |
| `‽` | Paradox (trust boundary) | `‽` <: `‽` |
| `ε` | Evidence variable | polymorphic |

### 3.2 Evidence Lattice

```
        ⊤ (any evidence)
       /|\
      ! ? ~
       \|/
        ‽
        |
        ⊥ (no evidence)
```

### 3.3 Evidence Combination

The evidence of a compound expression is the **meet** (greatest lower bound) of its components:

```
E₁ ⊓ E₂ = match (E₁, E₂) {
    (!, !) => !
    (!, ?) | (?, !) => ?
    (!, ~) | (~, !) => ~
    (?, ?) => ?
    (?, ~) | (~, ?) => ?    // uncertain dominates reported
    (~, ~) => ~
    (_, ‽) | (‽, _) => ‽
}
```

### 3.4 Evidence-Polymorphic Functions

```sigil
// Function preserves evidence level
fn identity<T, E>(x: T^E) → T^E { x }

// Function upgrades evidence via validation
fn validate<T>(x: T~) → Result<T!, ValidationError~>

// Evidence bounded
fn process<E: KnownOrUncertain>(x: Data^E) → Data^E
```

---

## 4. Dependent Types

### 4.1 Pi Types (Dependent Functions)

A Pi type `Πx:A. B` represents functions where the return type `B` may depend on the argument value `x`.

```sigil
// The return type depends on the input value
fn replicate<T>(x: T, n: Nat) → Vec<T, n>    // Vec with exactly n elements

// Array indexing: index must be less than length
fn get<T, const N: usize>(arr: [T; N], idx: {i:usize | i < N}) → T

// Type-safe printf (return type depends on format string)
fn printf(fmt: &str) → Πargs:ArgsFor(fmt). ()
```

### 4.2 Sigma Types (Dependent Pairs)

A Sigma type `Σx:A. B` represents pairs where the type of the second element depends on the value of the first.

```sigil
// Length-indexed vector
type Vec<T, n: Nat> = Σlen:Nat. {xs:[T] | len(xs) = len ∧ len = n}

// Existential quantification (hiding the witness)
type SomePositive = Σn:i64. {_:() | n > 0}
```

### 4.3 Path-Dependent Types

Types can depend on paths through values:

```sigil
sigil Container {
    type Element
    elements: [Self.Element]
}

fn extract(c: Container) → c.Element {
    c.elements[0]
}
```

---

## 5. Inference Algorithm

### 5.1 Judgment Forms

```
Γ ⊢ e ⇒ τ        -- synthesis: infer type of e
Γ ⊢ e ⇐ τ        -- checking: check e against τ
Γ ⊢ τ₁ <: τ₂     -- subtyping
Γ ⊢ φ            -- predicate entailment (via SMT)
Γ ⊢ τ :: κ       -- kinding
```

### 5.2 Bidirectional Rules

**Synthesis (Bottom-Up)**

```
VAR
─────────────────────────
Γ, x:τ ⊢ x ⇒ τ

LITERAL
─────────────────────────
Γ ⊢ n ⇒ {x:IntType | x = n}    -- literals get singleton refinement

APP-SYNTH
Γ ⊢ e₁ ⇒ Πx:A. B    Γ ⊢ e₂ ⇐ A
──────────────────────────────────
Γ ⊢ e₁(e₂) ⇒ B[e₂/x]

FIELD
Γ ⊢ e ⇒ {f₁:τ₁, ..., fₙ:τₙ}
──────────────────────────────────
Γ ⊢ e.fᵢ ⇒ τᵢ

ANNO
Γ ⊢ e ⇐ τ
──────────────────────────────────
Γ ⊢ (e : τ) ⇒ τ
```

**Checking (Top-Down)**

```
LAM-CHECK
Γ, x:A ⊢ e ⇐ B
──────────────────────────────────
Γ ⊢ λx. e ⇐ Πx:A. B

IF-CHECK
Γ ⊢ c ⇐ bool    Γ, c=true ⊢ e₁ ⇐ τ    Γ, c=false ⊢ e₂ ⇐ τ
────────────────────────────────────────────────────────────
Γ ⊢ if c then e₁ else e₂ ⇐ τ

MATCH-CHECK
Γ ⊢ e ⇒ τ_scrut    for each arm pᵢ => eᵢ: Γ, bindings(pᵢ, τ_scrut) ⊢ eᵢ ⇐ τ
────────────────────────────────────────────────────────────────────────────
Γ ⊢ match e { p₁ => e₁, ..., pₙ => eₙ } ⇐ τ

SUB
Γ ⊢ e ⇒ τ₁    Γ ⊢ τ₁ <: τ₂
──────────────────────────────────
Γ ⊢ e ⇐ τ₂
```

### 5.3 Subtyping Rules

```
REFL
──────────────────
Γ ⊢ τ <: τ

TRANS
Γ ⊢ τ₁ <: τ₂    Γ ⊢ τ₂ <: τ₃
────────────────────────────────
Γ ⊢ τ₁ <: τ₃

REFINE-SUB
Γ ⊢ φ₁ ⟹ φ₂    (via SMT)
────────────────────────────────
Γ ⊢ {x:τ | φ₁} <: {x:τ | φ₂}

REFINE-BASE
────────────────────────────────
Γ ⊢ {x:τ | φ} <: τ

PI-SUB
Γ ⊢ A₂ <: A₁    Γ, x:A₂ ⊢ B₁ <: B₂
────────────────────────────────────
Γ ⊢ Πx:A₁. B₁ <: Πx:A₂. B₂

EVIDENCE-SUB
E₁ ≤ E₂ in evidence lattice
────────────────────────────────
Γ ⊢ τ^E₁ <: τ^E₂
```

### 5.4 Constraint Generation

Rather than immediately solving, we **generate constraints** during traversal:

```
type Constraint =
    | Unify(τ, τ)           -- τ₁ = τ₂
    | Subtype(τ, τ)         -- τ₁ <: τ₂
    | Entail(Γ, φ)          -- Γ ⊢ φ
    | WellFormed(τ)         -- τ is a valid type
```

**Algorithm:**
```
infer(Γ, e) → (τ, Constraints)

1. Traverse AST, generating fresh type variables for unknowns
2. For each node, emit constraints based on typing rules
3. Return (inferred type with variables, constraint set)
```

### 5.5 Constraint Solving

Constraints are solved in phases:

```
Algorithm: solve(C: Constraints) → Substitution | Error

Phase 1: UNIFICATION
    - Solve equality constraints via standard unification
    - Propagate substitutions

Phase 2: SUBTYPE RESOLUTION
    - For each τ₁ <: τ₂:
      - If both are refinement types, emit to SMT
      - If structural, decompose into component constraints
      - If involves type variables, defer or approximate

Phase 3: SMT SOLVING
    - Collect all predicate constraints
    - Encode in SMT-LIB format
    - Query Z3/CVC5 for satisfiability
    - Extract model or unsat core for errors

Phase 4: EVIDENCE INFERENCE
    - Compute evidence levels bottom-up
    - Verify evidence constraints at function boundaries
    - Insert evidence coercions where needed
```

---

## 6. SMT Integration

### 6.1 Theory Selection

The SMT solver uses a combination of theories:

| Theory | Purpose |
|--------|---------|
| QF_LIA | Linear integer arithmetic |
| QF_LRA | Linear real arithmetic |
| QF_BV | Bitvectors (for fixed-width ints) |
| QF_AUFLIA | Arrays + uninterpreted functions |
| QF_S | Strings |

### 6.2 Encoding Examples

```smt2
; Type: {x:i32 | x > 0 ∧ x < 100}
(declare-const x Int)
(assert (and (> x 0) (< x 100)))

; Subtyping: {x | x > 0 ∧ x < 50} <: {x | x > 0 ∧ x < 100}
; Becomes: ∀x. (x > 0 ∧ x < 50) ⟹ (x > 0 ∧ x < 100)
(assert (forall ((x Int))
    (=> (and (> x 0) (< x 50))
        (and (> x 0) (< x 100)))))
(check-sat)  ; expect sat

; Array bounds: idx < len(arr)
(declare-const arr (Array Int Int))
(declare-const len Int)
(declare-const idx Int)
(assert (and (>= idx 0) (< idx len)))
```

### 6.3 Decidability Boundaries

To ensure decidability, refinements are restricted:

- **No arbitrary quantifier alternation** — Bounded quantifiers only
- **Linear arithmetic** — No polynomial constraints by default
- **Finite unrolling** — Recursive predicates unrolled to fixed depth
- **Timeout** — SMT queries have bounded time

When boundaries are exceeded, the system falls back to **requiring explicit proofs**.

---

## 7. Advanced Features

### 7.1 Liquid Type Inference

For many refinements, we can infer predicates automatically using **liquid type inference**:

```
Algorithm: liquid_infer(Γ, e, Q: PredicateTemplates) → RefinedType

1. Generate templates: τ = {x:base | κ} where κ is conjunction of Q
2. Collect constraints from program
3. Solve for which predicates in Q must hold
4. Return refined type
```

**Example:**
```sigil
fn abs(x: i32) → i32 {
    if x < 0 { -x } else { x }
}

// Liquid inference discovers:
// abs : (x:i32) → {r:i32 | r ≥ 0 ∧ (x ≥ 0 ⟹ r = x) ∧ (x < 0 ⟹ r = -x)}
```

### 7.2 Refinement Type Polymorphism

Refinements can be abstracted:

```sigil
fn max<P: Predicate<i32>>(x: {v:i32 | P(v)}, y: {v:i32 | P(v)})
    → {v:i32 | P(v) ∧ v ≥ x ∧ v ≥ y}
{
    if x > y { x } else { y }
}

// Instantiate P with "v > 0":
let m = max<|v| v > 0>(5, 10)  // m : {v:i32 | v > 0 ∧ v ≥ 5 ∧ v ≥ 10}
```

### 7.3 Dependent Pattern Matching

Pattern matching refines types in branches:

```sigil
fn length<T, n: Nat>(xs: Vec<T, n>) → {r:Nat | r = n} {
    match xs {
        [] => 0,                    // Here n = 0
        [_, ..rest] => {
            // rest : Vec<T, n-1>
            1 + length(rest)        // Returns {r | r = 1 + (n-1)} = {r | r = n}
        }
    }
}
```

### 7.4 Evidence-Dependent Types

Types can depend on evidence levels:

```sigil
// Only defined for known values
fn trusted_divide(a: i32!, b: {x:i32! | x ≠ 0}) → i32! {
    a / b
}

// Evidence-polymorphic with constraint
fn safe_get<E: !|?>(xs: [T]^E, idx: {i:usize | i < len(xs)}) → T^E {
    xs[idx]
}
```

---

## 8. Error Reporting

### 8.1 Error Categories

| Code | Category | Description |
|------|----------|-------------|
| `T1xx` | Unification | Type mismatch |
| `T2xx` | Subtyping | Refinement not implied |
| `T3xx` | Evidence | Wrong evidence level |
| `T4xx` | Bounds | Index/arithmetic bounds |
| `T5xx` | Dependent | Type-level computation failed |

### 8.2 SMT-Derived Diagnostics

When SMT returns unsat, we extract the **unsat core** for diagnostics:

```
error[T201]: refinement not satisfied
  --> src/main.sg:10:5
   |
10 |     let x: Positive = -5;
   |           ^^^^^^^^    ^^ this value is -5
   |           |
   |           requires: x > 0
   |
   = note: constraint `x > 0` is unsatisfiable when x = -5
   = help: Positive requires strictly positive values
```

### 8.3 Counterexample Generation

For failed subtyping, we can provide counterexamples:

```
error[T202]: subtype check failed
  --> src/main.sg:15:10
   |
15 |     let y: Bounded<0, 50> = scale(x);
   |            ^^^^^^^^^^^^^^   ^^^^^^^^ returns Bounded<0, 100>
   |
   = note: Bounded<0, 100> is not a subtype of Bounded<0, 50>
   = counterexample: value 75 is in Bounded<0, 100> but not Bounded<0, 50>
```

---

## 9. Implementation Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     TYPE INFERENCE ENGINE                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐      │
│  │  Constraint  │───▶│  Unifier     │───▶│  Subtype     │      │
│  │  Generator   │    │              │    │  Resolver    │      │
│  └──────────────┘    └──────────────┘    └──────────────┘      │
│         │                                       │                │
│         │            ┌──────────────┐           │                │
│         │            │  Refinement  │           │                │
│         └───────────▶│  Collector   │◀──────────┘                │
│                      └──────────────┘                            │
│                             │                                    │
│                      ┌──────────────┐                            │
│                      │  SMT Encoder │                            │
│                      └──────────────┘                            │
│                             │                                    │
│                      ┌──────────────┐                            │
│                      │  Z3 / CVC5   │  (external process)       │
│                      └──────────────┘                            │
│                             │                                    │
│                      ┌──────────────┐                            │
│                      │  Solution    │                            │
│                      │  Extractor   │                            │
│                      └──────────────┘                            │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## 10. Complexity and Decidability

### 10.1 Theoretical Bounds

| Feature | Complexity | Notes |
|---------|------------|-------|
| Basic HM inference | O(n²) amortized | With efficient union-find |
| Subtype checking | NP-complete | For refinement types |
| SMT solving | PSPACE | For QF_LIA fragment |
| Full dependent types | Undecidable | Restricted in practice |

### 10.2 Practical Guarantees

We ensure decidability through:

1. **Stratified refinements** — Base types, then refinements, then dependencies
2. **Bounded quantifiers** — No unrestricted ∀/∃ in refinements
3. **Fuel-based recursion** — Type-level recursion has depth limits
4. **Timeout fallback** — Undecidable queries error with suggestion to add annotations

---

## 11. Summary

Sigil's type inference is:

- **Powerful**: Dependent types, refinements, evidence tracking
- **Sound**: All accepted programs are type-safe (assuming SMT soundness)
- **Practical**: Decidable fragment with clear boundaries
- **Agent-native**: Complexity is a feature, not a bug

The system assumes its consumer can:
- Track thousands of constraints simultaneously
- Interface with SMT solvers fluently
- Reason about logical implications natively

Humans can use Sigil through tooling that presents simplified views, generates annotations, and explains type errors. The language itself does not compromise.

---

*"The type is the true name. To know it completely is to wield total power over the value."*
