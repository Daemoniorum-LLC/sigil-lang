# Sigil Mathematical Foundations

## 1. Philosophy: Poly-Cultural Mathematics

Mathematics is not a single tradition but a convergence of human insight across civilizations. Sigil's mathematical foundations draw from multiple traditions:

| Tradition | Contributions | Sigil Integration |
|-----------|---------------|-------------------|
| **Indian** | Zero (śūnya), infinity (ananta), combinatorics, iteration | First-class void, infinite streams, recursive patterns |
| **Chinese** | Rod calculus, matrices, remainder theorem | Matrix primitives, modular arithmetic |
| **Mayan** | Vigesimal (base-20), positional zero | Multi-base literals, calendar arithmetic |
| **Islamic** | Algebra (al-jabr), algorithms | Symbolic manipulation, named algorithms |
| **African** | Fractals, binary divination, Egyptian fractions | Fractal generators, unit fractions |
| **Babylonian** | Sexagesimal, quadratics | Base-60 time/angle, polynomial roots |
| **Japanese (Wasan)** | Sangaku geometry, elliptic integrals | Geometric constraints, temple problems |
| **Indigenous** | Kinship algebra, cyclical time, spatial logic | Relational types, cyclic arithmetic |

---

## 2. Numeric Plurality

### 2.1 Multi-Base Literals

Numbers exist in whatever base serves the domain:

```sigil
// Base prefixes
let decimal     = 42              // Base 10 (default)
let binary      = 0b101010        // Base 2
let octal       = 0o52            // Base 8
let hex         = 0x2A            // Base 16
let vigesimal   = 0v22            // Base 20 (Mayan)
let sexagesimal = 0s42            // Base 60 (Babylonian)
let duodecimal  = 0d36            // Base 12

// Explicit base annotation
let base7 = 60:₇                  // 42 in base 7
let base36 = 16:₃₆                // 42 in base 36

// Mixed radix (like time: hours-minutes-seconds)
let time = [1, 23, 45]:₆₀         // 1:23:45 in sexagesimal cascade
let mayan_date = [9, 12, 11, 5, 18]:₂₀  // Long count
```

### 2.2 Number Systems as Types

```sigil
// Different number systems are distinct types
type Decimal = Num·base(10)
type Binary = Num·base(2)
type Sexagesimal = Num·base(60)
type MayanLong = Num·mixed([20, 20, 20, 18, 20])  // Mayan long count

// Conversion is explicit
let x: Decimal = 42
let y: Sexagesimal = x·to_base(60)  // 0s42
let z: Binary = x·to_base(2)        // 0b101010

// Domain-specific defaults
//@ rune: default_base(60)
mod astronomy {
    let angle = 45·30·0  // degrees, minutes, seconds
}
```

### 2.3 Śūnya (Zero) and Ananta (Infinity)

Indian mathematics formalized zero and infinity. Sigil treats them as first-class:

```sigil
// Zero is not just 0, but a concept
let śūnya = ∅           // Void/emptiness
let zero = 0            // Numeric zero
let origin = ◯          // Geometric zero/origin

// Infinity has character
let ananta = ∞          // Unsigned infinity
let pos_inf = +∞        // Positive infinity
let neg_inf = -∞        // Negative infinity
let aleph = ℵ₀          // Countable infinity
let continuum = 𝔠       // Cardinality of reals

// Operations with infinity (Brahmagupta's rules)
∞ + 1 == ∞             // true
∞ - ∞ == ⊥             // undefined (bottom type)
1 / ∅ == ∞             // approaches infinity
∅ / ∅ == ⊥             // undefined

// Infinite sequences (lazy)
let naturals = 0..∞
let primes = sieve(2..∞)
```

---

## 3. Algebraic Traditions

### 3.1 Al-Jabr (Restoration/Algebra)

From al-Khwarizmi's *al-Kitāb al-mukhtaṣar*. Symbolic manipulation as core operation:

```sigil
// Symbolic expressions
let expr = sym!"x² + 2x + 1"
let factored = expr|factor        // (x + 1)²
let expanded = factored|expand    // x² + 2x + 1

// Al-jabr operations
let equation = sym!"x² + 5 = 2x + 6"
let restored = equation|jabr      // x² = 2x + 1 (restore: move terms)
let balanced = equation|muqabala  // x² - 2x = 1 (balance: combine like terms)

// Solve
let roots = equation|solve(x)     // x = 1 ± √2

// Symbolic in types
fn quadratic<T: Ring>(a: T, b: T, c: T, x: Sym) -> Expr<T> {
    a * x² + b * x + c
}
```

### 3.2 Chinese Remainder Theorem

From *Sunzi Suanjing* (3rd century). Modular arithmetic as primitive:

```sigil
// Modular types
type Mod<N: const usize> = i64 mod N

let a: Mod<7> = 3
let b: Mod<7> = 5
let c = a + b              // 1 (mod 7)

// Chinese remainder theorem
let system = [
    x ≡ 2 (mod 3),
    x ≡ 3 (mod 5),
    x ≡ 2 (mod 7),
]
let solution = system|crt  // x ≡ 23 (mod 105)

// Modular morphemes
value|mod(n)               // Reduce modulo n
value|mod·inv(n)           // Modular inverse
values|crt                 // Solve system via CRT
```

### 3.3 Indian Combinatorics (Pingala)

From *Chandaḥśāstra* (2nd century BCE). Binary expansion, Fibonacci, Pascal's triangle:

```sigil
// Pingala's meru (Pascal's triangle)
let meru = pascal·triangle()
meru|take(5)  // [[1], [1,1], [1,2,1], [1,3,3,1], [1,4,6,4,1]]

// Mātrāmeru (Fibonacci via prosody)
let matrameru = fib·sequence()
matrameru|take(10)  // [1, 1, 2, 3, 5, 8, 13, 21, 34, 55]

// Pratyaya (binary representation)
let pratyaya = 42|to_laghu_guru  // "GLLGLG" (heavy/light syllables)

// Combinatorial morphemes
n|choose(k)            // Binomial coefficient (nCk)
n|permute(k)           // Permutations (nPk)
set|powerset           // All subsets
seq|partitions         // Integer partitions (also Indian origin)
```

---

## 4. Geometric Traditions

### 4.1 Egyptian/African Fractals

Recursive self-similarity in African architecture and art:

```sigil
// Fractal definition
fractal Koch(depth: u32) -> Path {
    if depth == 0 {
        line(1)
    } else {
        let sub = Koch(depth - 1)|scale(1/3)
        sub ++ sub|rotate(60°) ++ sub|rotate(-60°) ++ sub
    }
}

// Iterative fractal (L-system style)
let sierpinski = lsystem {
    axiom: "A"
    rules: {
        A => "B-A-B"
        B => "A+B+A"
    }
    angle: 60°
}

// African fractal patterns
let benin_wall = fractal·recursive {
    base: rectangle(1, 1)
    transform: |shape| [
        shape|scale(0.5)|translate(0, 0),
        shape|scale(0.5)|translate(0.5, 0),
        shape|scale(0.5)|translate(0.25, 0.5),
    ]
}
```

### 4.2 Egyptian Fractions (Unit Fractions)

Representing rationals as sums of distinct unit fractions:

```sigil
// Egyptian fraction type
type EgyptianFrac = [u64]  // Denominators of 1/n terms

// Convert to Egyptian fractions (greedy algorithm)
let frac = 4/5
let egyptian = frac|to_egyptian  // [2, 4, 20] meaning 1/2 + 1/4 + 1/20

// Operations preserve Egyptian form
let sum = egyptian·add([3, 6])   // Add 1/3 + 1/6

// Rhind papyrus 2/n table
let rhind_table = (3..101)
    |φ{_ % 2 == 1}
    |τ{n => (2, n)|to_egyptian}
```

### 4.3 Wasan (Japanese Temple Geometry)

Sangaku problems — geometric constraints and relationships:

```sigil
// Geometric constraint system
let sangaku = constraints {
    // Three circles tangent to each other and a line
    circle A: center(xa, ya), radius(ra)
    circle B: center(xb, yb), radius(rb)
    circle C: center(xc, yc), radius(rc)
    line L: y = 0

    // Tangency constraints
    A·tangent(L)
    B·tangent(L)
    C·tangent(L)
    A·tangent(B)
    B·tangent(C)

    // Descartes circle theorem emerges
    solve for: ra, rb, rc given ra = 1, rb = 2
}

// Elliptic integrals (Wasan contribution)
let arc_length = ellipse(a, b)|arc·length(0, π/4)
```

### 4.4 Indigenous Spatial Logic

Non-Cartesian spatial reasoning:

```sigil
// Directional types (not just x,y)
enum Direction {
    Toward(Landmark),
    Away(Landmark),
    Sunrise,
    Sunset,
    Upriver,
    Downriver,
    Inland,
    Seaward,
}

// Relational positioning
struct Position {
    relative_to: Landmark,
    direction: Direction,
    distance: Distance,
}

// Cyclic/seasonal time
type Season = Cycle<4>  // Four seasons, wrapping
type MoonPhase = Cycle<28>
type Dreamtime = Eternal  // Non-linear time

// Movement as transformation
fn journey(from: Place, to: Place) -> Path {
    // Path described by landmarks and directions
    // Not coordinate deltas
}
```

---

## 5. Relational Mathematics

### 5.1 Kinship Algebra

Many indigenous cultures developed sophisticated relational calculi:

```sigil
// Kinship as type relations
trait Kin {
    type Parent: Kin
    type Child: Kin
    type Sibling: Kin
    type Spouse: Kin
}

// Relational composition
type Aunt = Parent·Sibling·Female
type Cousin = Parent·Sibling·Child
type NephewNiece = Sibling·Child

// Moiety systems (binary division)
enum Moiety { Sun, Moon }
impl Moiety {
    fn marriage_rule(self) -> Moiety {
        match self {
            Sun => Moon,   // Must marry opposite moiety
            Moon => Sun,
        }
    }
}

// Section systems (4 or 8 divisions)
type Section = Mod<8>
fn marriage_section(s: Section) -> Section {
    (s + 4) mod 8  // Marry into opposite section
}
```

### 5.2 Category-Theoretic Foundations

Abstract relational structure:

```sigil
// Categories as first-class
trait Category {
    type Object
    type Morphism<A: Object, B: Object>

    fn id<A: Object>() -> Morphism<A, A>
    fn compose<A, B, C>(
        f: Morphism<A, B>,
        g: Morphism<B, C>
    ) -> Morphism<A, C>
}

// Functors preserve structure
trait Functor<C: Category, D: Category> {
    fn map_obj(obj: C·Object) -> D·Object
    fn map_mor<A, B>(f: C·Morphism<A, B>) -> D·Morphism<map_obj(A), map_obj(B)>
}

// Natural transformations
trait NatTrans<F: Functor, G: Functor> {
    fn component<A>(fa: F·Apply<A>) -> G·Apply<A>
}
```

---

## 6. Algorithmic Traditions

### 6.1 Named Algorithms (Honoring Origins)

```sigil
// Algorithms named for their cultural origins
fn euclid·gcd(a: u64, b: u64) -> u64          // Greek
fn eratosthenes·sieve(n: u64) -> [u64]        // Greek
fn pingala·binary(n: u64) -> [bool]           // Indian
fn alkhwarizmi·sqrt(n: f64) -> f64            // Islamic
fn sunzi·crt(residues: [(i64, u64)]) -> i64   // Chinese
fn rhind·divide(a: Frac, b: Frac) -> Egyptian // Egyptian
fn aryabhata·sine(θ: Angle) -> f64            // Indian
fn zhu·elimination(m: Matrix) -> Matrix       // Chinese (Gaussian elimination predated)

// Modern algorithms similarly attributed
fn dijkstra·shortest(g: Graph) -> Path
fn hoare·quicksort<T: Ord>(arr: &mut [T])
fn lamport·clock() -> LogicalTime
```

### 6.2 Iteration vs Recursion vs Tabulation

Different traditions emphasize different computational patterns:

```sigil
// Indian style: recursive with accumulator
fn factorial·recursive(n: u64) -> u64 {
    match n {
        0 => 1,
        n => n * factorial·recursive(n - 1),
    }
}

// Chinese style: tabulation (building tables)
fn factorial·tabulated(n: u64) -> u64 {
    let table = [1u64; n + 1]
    for i in 1..=n {
        table[i] = i * table[i - 1]
    }
    table[n]
}

// Babylonian style: iterative refinement
fn sqrt·babylonian(n: f64) -> f64 {
    let mut guess = n / 2
    loop {
        let next = (guess + n / guess) / 2
        if (next - guess)|abs < ε { break next }
        guess = next
    }
}
```

---

## 7. Geometric Algebra

Unifying geometric traditions through Clifford algebra:

```sigil
// Geometric primitives
type Scalar = G0           // Grade 0
type Vector = G1           // Grade 1 (direction)
type Bivector = G2         // Grade 2 (oriented plane)
type Trivector = G3        // Grade 3 (oriented volume)
type Multivector = G       // Mixed grade

// Geometric product
let a: Vector = vec![1, 0, 0]
let b: Vector = vec![0, 1, 0]
let ab: Bivector = a * b   // Oriented plane

// Inner product (symmetric)
let dot = a · b            // Scalar

// Outer product (antisymmetric)
let wedge = a ∧ b          // Bivector

// Rotations via rotors (no matrices needed)
let rotor = exp(θ/2 * e12)  // Rotation in e1-e2 plane
let rotated = rotor * v * rotor·reverse

// Reflections
let reflected = -n * v * n  // Reflect v in plane normal to n

// Projections
let projected = (v · n) * n / (n · n)
```

---

## 8. Mathematical Morphemes

### 8.1 Arithmetic Morphemes

```sigil
// Basic operations as morphemes
x|+y          // Add
x|-y          // Subtract
x|*y          // Multiply
x|/y          // Divide
x|%y          // Modulo
x|**y         // Power
x|√            // Square root
x|∛            // Cube root
x|ⁿ√y         // nth root
x|!           // Factorial
x|abs         // Absolute value
x|neg         // Negate
x|recip       // Reciprocal (1/x)
```

### 8.2 Sequence Morphemes

```sigil
seq|Σ              // Sum (sigma)
seq|Π              // Product (pi)
seq|Δ              // Differences
seq|∫              // Cumulative sum (discrete integral)
seq|∂              // Discrete derivative

// With bounds
seq|Σ[0..n]        // Sum from 0 to n
seq|Π[1..=n]       // Product from 1 to n inclusive

// Named reductions
seq|sum            // Same as Σ
seq|product        // Same as Π
seq|mean           // Arithmetic mean
seq|gmean          // Geometric mean (Babylonian)
seq|hmean          // Harmonic mean (Greek)
```

### 8.3 Set/Collection Morphemes

```sigil
a|∪b               // Union
a|∩b               // Intersection
a|∖b               // Set difference
a|△b               // Symmetric difference
a|×b               // Cartesian product

a|⊂b               // Subset test
a|⊃b               // Superset test
x|∈a               // Membership test

set|℘              // Power set
set|#              // Cardinality
```

### 8.4 Logical Morphemes

```sigil
a|∧b               // And (conjunction)
a|∨b               // Or (disjunction)
a|⊻b               // Xor (exclusive or)
a|⊼b               // Nand
a|⊽b               // Nor
¬a                 // Not (negation)
a|→b               // Implies
a|↔b               // Iff (biconditional)

// Quantifiers
∀x ∈ S: P(x)       // For all
∃x ∈ S: P(x)       // There exists
∃!x ∈ S: P(x)      // There exists unique
```

---

## 9. Example: Multi-Cultural Computation

```sigil
//! Computing π using various cultural methods

use math·{π, Σ, Π}

// Madhava-Leibniz series (Indian, 14th century)
fn π·madhava(terms: u64) -> f64 {
    4 * (0..terms)|τ{n => (-1)^n / (2*n + 1)}|Σ
}

// Wallis product (English, 1656 — but using infinite product idea)
fn π·wallis(terms: u64) -> f64 {
    2 * (1..=terms)|τ{n =>
        (2*n) * (2*n) / ((2*n - 1) * (2*n + 1))
    }|Π
}

// Ramanujan series (Indian, 1914)
fn π·ramanujan(terms: u64) -> f64 {
    let coeff = 2 * √2 / 9801
    let series = (0..terms)|τ{k =>
        k|!·quad * (1103 + 26390*k) / (k|! ** 4 * 396 ** (4*k))
    }|Σ
    1 / (coeff * series)
}
where {
    fn quad·factorial(n: u64) -> u64 {
        (1..=4*n)|φ{i => i % 2 == 1}|Π
    }
}

// Zu Chongzhi approximation (Chinese, 5th century)
const π·zu: Rational = 355 / 113  // Accurate to 6 decimal places

// Archimedes bounds (Greek, 3rd century BCE)
fn π·archimedes(n: u64) -> (f64, f64) {
    // n-gon inscribed and circumscribed
    let inscribed = n * sin(π / n)
    let circumscribed = n * tan(π / n)
    (inscribed, circumscribed)
}

fn main() {
    print!("Madhava (100 terms): {π·madhava(100)}")
    print!("Wallis (1000 terms): {π·wallis(1000)}")
    print!("Ramanujan (2 terms): {π·ramanujan(2)}")
    print!("Zu Chongzhi: {π·zu·to_f64()}")
    print!("Archimedes (96-gon): {π·archimedes(96)}")
}
```

---

## 10. Design Principles

1. **No privileged notation** — Western symbols are available but not default
2. **Multiple representations** — Same value, different cultural expressions
3. **Named origins** — Algorithms honor their discoverers/cultures
4. **Structural universals** — Category theory as neutral meeting ground
5. **Domain-appropriate bases** — Use base-60 for time, base-20 for calendrics
6. **Relational thinking** — Kinship and relationship as mathematical primitives
7. **Geometric plurality** — Euclidean is one option among many
