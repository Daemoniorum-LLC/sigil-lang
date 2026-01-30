# Sigil Pattern Matching Specification

> *"To match a pattern is to recognize truth. To match a predicate is to prove it."*

## 0. Design Philosophy: Patterns as Predicates

Traditional pattern matching is **structural** — you match shapes.
Sigil pattern matching is **semantic** — you match *properties*.

A pattern is not "does this look like X?" but "does this satisfy P?"

This specification assumes agents can:
- Evaluate complex predicates during compilation
- Track exhaustiveness across predicate spaces
- Interface with SMT solvers for coverage proofs

---

## 1. Pattern Grammar

```
pattern ::=
    | _                                   -- wildcard
    | literal                             -- literal pattern
    | ident                               -- binding
    | ident @ pattern                     -- binding with nested pattern
    | (pattern, ..., pattern)             -- tuple
    | [pattern, ..., pattern]             -- array/slice
    | [pattern, .., pattern]              -- slice with rest
    | Ctor { field: pattern, ... }        -- struct
    | Ctor(pattern, ...)                  -- tuple struct / enum
    | &pattern | &mut pattern             -- reference
    | box pattern                         -- box
    | pattern | pattern                   -- or-pattern
    | pattern if guard                    -- guard
    | pattern where φ                     -- predicate pattern (NEW)
    | { φ }                               -- pure predicate pattern (NEW)
    | pattern -> expr                     -- view pattern (NEW)
    | ~pattern                            -- evidence pattern (NEW)
    | !pattern | ?pattern                 -- evidence level pattern (NEW)
```

---

## 2. Predicate Patterns

### 2.1 Basic Predicate Patterns

Match based on logical properties, not structure:

```sigil
match n {
    { n > 0 }           => "positive",
    { n < 0 }           => "negative",
    { n = 0 }           => "zero",
}

match list {
    { len(list) = 0 }   => "empty",
    { len(list) = 1 }   => "singleton",
    { len(list) > 1 }   => "multiple",
}
```

### 2.2 Combined Structural + Predicate

```sigil
match point {
    Point { x, y } where x = y          => "diagonal",
    Point { x, y } where x² + y² ≤ 1    => "in unit circle",
    Point { x, y } where x > 0 ∧ y > 0  => "first quadrant",
    _                                    => "other",
}
```

### 2.3 Mathematical Predicates

```sigil
match n {
    { prime(n) }                    => "prime",
    { perfect_square(n) }           => "perfect square",
    { n mod 2 = 0 }                 => "even",
    { fibonacci(n) }                => "fibonacci number",
    _                               => "other",
}

match matrix {
    { symmetric(matrix) }           => "symmetric",
    { orthogonal(matrix) }          => "orthogonal",
    { singular(matrix) }            => "singular",
    _                               => "general",
}
```

### 2.4 Quantified Predicates

```sigil
match array {
    { ∀i. array[i] ≥ 0 }                        => "non-negative",
    { ∀i,j. i < j ⟹ array[i] ≤ array[j] }     => "sorted ascending",
    { ∀i,j. i < j ⟹ array[i] ≥ array[j] }     => "sorted descending",
    { ∃i. array[i] = target }                   => "contains target",
    _                                            => "unstructured",
}
```

---

## 3. View Patterns

View patterns transform the scrutinee before matching:

### 3.1 Basic View Patterns

```sigil
match text {
    text -> parse::<i32>() -> Ok(n)     => use_number(n),
    text -> parse::<f64>() -> Ok(f)     => use_float(f),
    text -> trim() -> ""                => "empty after trim",
    _                                   => "unparseable",
}
```

### 3.2 Chained Views

```sigil
match request {
    req -> headers() -> get("Authorization") -> Some(auth)
        -> parse_bearer() -> Ok(token)
        -> validate() -> Ok(claims)         => authorized(claims),
    _                                       => unauthorized(),
}
```

### 3.3 Bidirectional Views

For invertible transformations:

```sigil
// Define a bidirectional view
view Celsius <-> Fahrenheit {
    forward(c) = c * 9/5 + 32
    backward(f) = (f - 32) * 5/9
}

match temp {
    temp -> Celsius -> { c < 0 }    => "freezing (Celsius)",
    temp -> Fahrenheit -> { f > 100 } => "boiling (Fahrenheit)",
    _                               => "moderate",
}
```

---

## 4. Evidence Patterns

Match on evidentiality level:

### 4.1 Evidence Level Patterns

```sigil
match value {
    !known      => "definitely have it: {known}",
    ?uncertain  => "might have it",
    ~reported   => "externally reported: {reported}",
    ‽paradox    => "trust boundary crossed",
}
```

### 4.2 Combined Evidence + Structure

```sigil
match result {
    Ok(!data)   => "known success: {data}",
    Ok(~data)   => "reported success, needs validation",
    Ok(?data)   => "uncertain success",
    Err(!e)     => "known error: {e}",
    Err(~e)     => "external error: {e}",
}
```

### 4.3 Evidence Refinement

```sigil
match response~ {
    // Match and upgrade evidence
    data~ where validate(data~) -> Ok(valid!) => {
        // data upgraded to known after validation
        process(valid!)
    },
    data~ where validate(data~) -> Err(e) => {
        reject(e)
    },
}
```

---

## 5. Exhaustiveness Checking

### 5.1 The Exhaustiveness Problem

For predicate patterns, exhaustiveness is **semantic**, not syntactic:

```sigil
match n: i32 {
    { n > 0 }  => ...,
    { n < 0 }  => ...,
    // Is this exhaustive? Only if we prove: ∀n:i32. n > 0 ∨ n < 0 ∨ n = 0
    // We're missing n = 0!
}
```

### 5.2 SMT-Based Exhaustiveness

The compiler uses SMT to verify exhaustiveness:

```
Algorithm: check_exhaustiveness(patterns, type)

1. Let φ_type = type_predicate(type)    // e.g., MIN_I32 ≤ n ≤ MAX_I32
2. Let φ_covered = ⋁ᵢ pattern_predicate(pᵢ)
3. Query SMT: φ_type ∧ ¬φ_covered
4. If SAT: return counterexample (uncovered case)
5. If UNSAT: patterns are exhaustive
```

### 5.3 Exhaustiveness Examples

```sigil
// EXHAUSTIVE (SMT proves no gaps)
match n: u32 {
    { n = 0 }       => ...,
    { n > 0 }       => ...,
}
// Proof: ∀n:u32. n ≥ 0, and (n = 0 ∨ n > 0) covers all n ≥ 0 ✓

// NOT EXHAUSTIVE
match n: i32 {
    { n > 0 }       => ...,
    { n < 0 }       => ...,
}
// SMT returns: n = 0 as counterexample
// Error: non-exhaustive patterns, missing case where n = 0

// EXHAUSTIVE (with mathematical property)
match n: u32 {
    { n mod 2 = 0 } => "even",
    { n mod 2 = 1 } => "odd",
}
// Proof: ∀n. n mod 2 ∈ {0, 1} ✓
```

### 5.4 Exhaustiveness with Refinement Types

```sigil
type Positive = { n: i32 | n > 0 }

match p: Positive {
    { p = 1 }       => "one",
    { p > 1 }       => "greater than one",
}
// Exhaustive! p > 0 ∧ (p = 1 ∨ p > 1) covers all Positive ✓

// No need for p ≤ 0 case — type excludes it
```

---

## 6. Usefulness Checking

### 6.1 Redundant Patterns

A pattern is redundant if it can never match:

```sigil
match n: Positive {  // n > 0
    { n > 0 }   => ...,
    { n < 0 }   => ...,  // WARNING: unreachable, Positive excludes n < 0
    { n = 0 }   => ...,  // WARNING: unreachable, Positive excludes n = 0
}
```

### 6.2 Subsumption

Pattern P₁ subsumes P₂ if everything matching P₂ also matches P₁:

```sigil
match n {
    { n > 0 }       => ...,
    { n > 10 }      => ...,  // WARNING: unreachable, subsumed by n > 0
}
```

### 6.3 Usefulness Algorithm

```
Algorithm: is_useful(existing_patterns, new_pattern, type)

1. Let φ_new = pattern_predicate(new_pattern)
2. Let φ_existing = ⋁ᵢ pattern_predicate(existing_patternsᵢ)
3. Let φ_type = type_predicate(type)
4. Query SMT: φ_type ∧ φ_new ∧ ¬φ_existing
5. If SAT: new_pattern is useful (can match something new)
6. If UNSAT: new_pattern is redundant
```

---

## 7. Pattern Compilation

### 7.1 Decision Trees

Patterns compile to **decision trees** for efficient runtime matching:

```
match point {
    Point { x: 0, y: 0 }    => "origin",
    Point { x: 0, y }       => "y-axis",
    Point { x, y: 0 }       => "x-axis",
    Point { x, y }          => "general",
}

Compiles to:

         [test x = 0?]
          /         \
        yes          no
         |            |
    [test y = 0?]   [test y = 0?]
      /     \         /     \
    yes     no      yes     no
     |       |       |       |
  "origin" "y-axis" "x-axis" "general"
```

### 7.2 Predicate Compilation

Predicate patterns may require runtime evaluation or static proof:

```sigil
match n {
    { prime(n) }    => ...,  // Runtime: call is_prime(n)
    { n > 0 }       => ...,  // Simple comparison
    { n = 42 }      => ...,  // Equality test
}
```

**Compilation strategies:**

| Predicate Type | Strategy |
|---------------|----------|
| Equality `x = c` | Direct comparison |
| Inequality `x < c` | Comparison |
| Arithmetic `x mod 2 = 0` | Inline computation |
| Function call `prime(x)` | Runtime function call |
| Quantified `∀i. P(i)` | Loop or specialized algorithm |
| SMT-provable at compile time | Elide check entirely |

### 7.3 Optimization: Compile-Time Evaluation

If the scrutinee is a compile-time constant, evaluate predicates at compile time:

```sigil
const N: i32 = 42;

match N {
    { prime(N) }        => ...,  // Evaluated at compile time: false
    { N mod 2 = 0 }     => ...,  // Evaluated at compile time: true ← selected
    _                   => ...,
}
```

---

## 8. Advanced Patterns

### 8.1 Active Patterns

User-defined pattern matchers:

```sigil
// Define an active pattern
pattern Even(n: i32) -> Option<i32> {
    if n % 2 == 0 { Some(n / 2) } else { None }
}

pattern Odd(n: i32) -> Option<i32> {
    if n % 2 == 1 { Some(n / 2) } else { None }
}

match number {
    Even(half)  => "even, half is {half}",
    Odd(half)   => "odd, half (rounded) is {half}",
}
```

### 8.2 Parameterized Patterns

```sigil
pattern Divisible(d: i32)(n: i32) -> Option<i32> {
    if n % d == 0 { Some(n / d) } else { None }
}

match n {
    Divisible(3)(q)     => "{n} = 3 × {q}",
    Divisible(5)(q)     => "{n} = 5 × {q}",
    _                   => "not divisible by 3 or 5",
}
```

### 8.3 Recursive Patterns

```sigil
// Match nested structure to arbitrary depth
pattern Balanced(tree: Tree) -> Option<i32> {
    match tree {
        Leaf(_) => Some(0),
        Node(l, r) => {
            let (dl, dr) = (Balanced(l)?, Balanced(r)?);
            if (dl - dr).abs() <= 1 { Some(max(dl, dr) + 1) } else { None }
        }
    }
}

match tree {
    Balanced(depth)     => "balanced with depth {depth}",
    _                   => "unbalanced",
}
```

### 8.4 Type-Narrowing Patterns

Patterns that refine the type in the branch:

```sigil
fn process(x: i32 | str) {
    match x {
        n: i32              => use_int(n),    // n: i32 in this branch
        s: str              => use_str(s),    // s: str in this branch
    }
}

// With refinements
fn safe_sqrt(n: i32) -> Option<f64> {
    match n {
        n where n >= 0  => Some(sqrt(n as f64)),  // n: {x:i32 | x >= 0}
        _               => None,
    }
}
```

---

## 9. Typing Rules

### 9.1 Pattern Typing Judgment

```
Γ ⊢ p : τ ⊣ Γ', φ
```

Pattern `p` matches type `τ`, produces bindings `Γ'`, and constraint `φ`.

### 9.2 Core Rules

```
WILDCARD
─────────────────────
Γ ⊢ _ : τ ⊣ Γ, true

LITERAL
─────────────────────────────
Γ ⊢ lit : τ ⊣ Γ, (scrutinee = lit)

BINDING
────────────────────────────────
Γ ⊢ x : τ ⊣ Γ[x:τ], true

PREDICATE
Γ ⊢ φ : bool
────────────────────────────────
Γ ⊢ { φ } : τ ⊣ Γ, φ

GUARD
Γ ⊢ p : τ ⊣ Γ', φ₁    Γ' ⊢ g : bool
────────────────────────────────────────
Γ ⊢ p if g : τ ⊣ Γ', φ₁ ∧ g

OR
Γ ⊢ p₁ : τ ⊣ Γ₁, φ₁    Γ ⊢ p₂ : τ ⊣ Γ₂, φ₂    Γ₁ = Γ₂
────────────────────────────────────────────────────────
Γ ⊢ p₁ | p₂ : τ ⊣ Γ₁, φ₁ ∨ φ₂

STRUCT
Γ ⊢ p₁ : τ₁ ⊣ Γ₁, φ₁  ...  Γₙ₋₁ ⊢ pₙ : τₙ ⊣ Γₙ, φₙ
──────────────────────────────────────────────────────────
Γ ⊢ S{f₁:p₁, ..., fₙ:pₙ} : S ⊣ Γₙ, φ₁ ∧ ... ∧ φₙ

VIEW
Γ ⊢ e : τ → τ'    Γ ⊢ p : τ' ⊣ Γ', φ
────────────────────────────────────────
Γ ⊢ (scrutinee -> e -> p) : τ ⊣ Γ', φ[e(scrutinee)/scrutinee']

EVIDENCE
────────────────────────────────────
Γ ⊢ !p : τ^! ⊣ Γ', φ ∧ (evidence = !)
Γ ⊢ ?p : τ^? ⊣ Γ', φ ∧ (evidence = ?)
Γ ⊢ ~p : τ^~ ⊣ Γ', φ ∧ (evidence = ~)
```

### 9.3 Match Expression Typing

```
MATCH
Γ ⊢ e : τ_s    for each arm pᵢ => eᵢ:
    Γ ⊢ pᵢ : τ_s ⊣ Γᵢ, φᵢ
    Γᵢ, (φᵢ assumed) ⊢ eᵢ : τ_r
exhaustive(p₁...pₙ, τ_s)
──────────────────────────────────────────────────
Γ ⊢ match e { p₁ => e₁, ..., pₙ => eₙ } : τ_r
```

---

## 10. Error Messages

### 10.1 Non-Exhaustive

```
error[M501]: non-exhaustive patterns
  --> src/main.sg:10:5
   |
10 | /     match n {
11 | |         { n > 0 } => "positive",
12 | |         { n < 0 } => "negative",
13 | |     }
   | |_____^
   |
   = note: patterns do not cover all possible values of `i32`
   = counterexample: n = 0
   = help: add a case for `{ n = 0 }` or a wildcard `_`
```

### 10.2 Unreachable Pattern

```
warning[M502]: unreachable pattern
  --> src/main.sg:15:9
   |
13 |         { n > 0 } => "positive",
   |         --------- matches all values where n > 0
14 |         { n > 10 } => "large positive",
   |         ^^^^^^^^^^ unreachable: subsumed by previous pattern
   |
   = help: reorder patterns so more specific cases come first
```

### 10.3 Predicate Error

```
error[M503]: invalid predicate in pattern
  --> src/main.sg:20:9
   |
20 |         { halts(program) } => ...,
   |           ^^^^^^^^^^^^^^ undecidable predicate
   |
   = note: pattern predicates must be in a decidable fragment
   = help: use a runtime check instead: `p if halts_within(program, timeout)`
```

---

## 11. Integration with Type System

### 11.1 Pattern Matching Refines Types

In each branch, the type is refined by the pattern:

```sigil
fn process(x: i32) {
    match x {
        { x > 0 } => {
            // Here: x : {v:i32 | v > 0}
            let y = sqrt(x);  // Safe! x is positive
        },
        { x < 0 } => {
            // Here: x : {v:i32 | v < 0}
            let y = -x;  // y : {v:i32 | v > 0}
        },
        { x = 0 } => {
            // Here: x : {v:i32 | v = 0}
        },
    }
}
```

### 11.2 Dependent Pattern Matching

Return type can depend on the pattern:

```sigil
fn classify(n: i32) -> match n {
    { n > 0 }  => Positive,
    { n < 0 }  => Negative,
    { n = 0 }  => Zero,
}
```

### 11.3 Capability Patterns

Match on capability permissions:

```sigil
match cap {
    cap@ρ[rw^1.0]   => "full read-write",
    cap@ρ[r^f] where f > 0 => "can read",
    cap@ρ[_^0]      => "no access",
}
```

---

## 12. Summary

Sigil's pattern matching:

| Feature | Traditional | Sigil |
|---------|-------------|-------|
| Pattern basis | Structural | **Semantic (predicates)** |
| Exhaustiveness | Syntactic coverage | **SMT-proven coverage** |
| Match conditions | Guards (if) | **First-class predicates** |
| Views | Limited/none | **Bidirectional view patterns** |
| Evidence | Not tracked | **Evidence-level patterns** |
| Compilation | Decision trees | **Decision trees + SMT + runtime predicates** |

Patterns are not templates — they are **logical specifications** of what to match. The compiler proves coverage, the runtime executes efficiently.

---

*"Every value either matches or it doesn't. The pattern is the question; the match is the answer."*
