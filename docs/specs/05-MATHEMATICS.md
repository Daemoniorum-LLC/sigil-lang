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

## 10. Temporal Mathematics

### 10.1 Cyclical Time

Many cultures understand time as cyclical rather than linear:

```sigil
// Cycle type: wrapping arithmetic
type Cycle<N: const usize> = i64 mod N

// Cultural calendar systems
type MayanTzolkin = Cycle<260>      // Sacred calendar
type MayanHaab = Cycle<365>         // Civil calendar
type ChineseStem = Cycle<10>        // Heavenly stems
type ChineseBranch = Cycle<12>      // Earthly branches
type IslamicMonth = Cycle<12>       // Hijri months
type HebrewMonth = Cycle<13>        // Including Adar II

// Calendar round (Mayan): two interlocking cycles
struct CalendarRound {
    tzolkin: MayanTzolkin,
    haab: MayanHaab,
}
// Repeats every lcm(260, 365) = 18,980 days ≈ 52 years

// Chinese sexagenary cycle
struct Sexagenary {
    stem: ChineseStem,
    branch: ChineseBranch,
}
// 60-year cycle: 10 stems × 12 branches, but only 60 valid pairs

// Cycle operations
let today: MayanTzolkin = 1
let future = today + 300  // Wraps automatically
let diff = future - today // Accounts for wrapping
```

### 10.2 Non-Linear Time

```sigil
// Aboriginal Dreamtime: events exist outside linear time
type Dreamtime = Eternal

trait Atemporal {
    // Events that exist in mythic time
    fn in_dreamtime(self) -> Dreamtime
    fn manifests_at(self) -> [LinearTime]  // When it appears in linear time
}

// Recurring time (events that repeat)
struct RecurringEvent<C: Cycle> {
    phase: C,
    meaning: str,
}

// Astrological aspects (angles between celestial bodies)
type Aspect = Cycle<360>
const CONJUNCTION: Aspect = 0
const SEXTILE: Aspect = 60
const SQUARE: Aspect = 90
const TRINE: Aspect = 120
const OPPOSITION: Aspect = 180
```

### 10.3 Polychronic Computation

```sigil
// Multiple time streams
struct PolyTime {
    linear: Duration,           // Western linear
    cyclic: [CyclicTime],       // Various cycles
    relational: EventGraph,     // "After X, before Y"
    seasonal: Season,           // Agricultural/natural
}

// Time can be queried in any system
let event = Event·new("harvest")
event|time·linear     // "2024-09-21T14:00:00Z"
event|time·mayan      // "13.0.11.14.5, 8 Chikchan, 13 Ch'en"
event|time·chinese    // "甲辰年八月十九" (Year of Dragon, 8th month, 19th day)
event|time·seasonal   // Autumn.mid
event|time·lunar      // WaxingGibbous(0.82)

// Temporal morphemes
event|when·before(other)   // Relative ordering
event|when·during(period)  // Containment
event|when·cycle(tzolkin)  // Position in cycle
```

---

## 11. Harmonic Mathematics (Music & Sound)

### 11.1 Tuning Systems

Every culture developed distinct approaches to dividing the octave:

```sigil
// Pythagorean tuning (Greek/Chinese/many others)
// Based on perfect fifths (3:2 ratio)
mod tuning·pythagorean {
    const FIFTH: Ratio = 3/2
    const FOURTH: Ratio = 4/3
    const TONE: Ratio = 9/8
    const SEMITONE: Ratio = 256/243  // Limma

    fn spiral_of_fifths(n: i32) -> Ratio {
        FIFTH ** n / (2 ** (n * 7 / 12))  // Reduced to octave
    }
}

// Just intonation (pure ratios)
mod tuning·just {
    const RATIOS: [Ratio; 12] = [
        1/1,    // Unison
        16/15,  // Minor second
        9/8,    // Major second
        6/5,    // Minor third
        5/4,    // Major third
        4/3,    // Perfect fourth
        45/32,  // Tritone
        3/2,    // Perfect fifth
        8/5,    // Minor sixth
        5/3,    // Major sixth
        9/5,    // Minor seventh
        15/8,   // Major seventh
    ]
}

// Equal temperament (Western standard)
mod tuning·equal {
    fn semitone(n: i32) -> f64 {
        2.0 ** (n / 12.0)
    }
}

// Gamelan pelog (Indonesian) - 7 unequal steps
mod tuning·pelog {
    // Approximate cents from tonic (varies by gamelan)
    const SCALE: [Cents; 7] = [0, 120, 270, 400, 550, 670, 800]
}

// Maqam (Arabic) - quarter tones
mod tuning·maqam {
    type QuarterTone = Cycle<24>  // 24 quarter-tones per octave

    const RAST: [QuarterTone; 8] = [0, 4, 7, 10, 14, 18, 21, 24]
    const BAYATI: [QuarterTone; 8] = [0, 3, 6, 10, 14, 17, 20, 24]
    const HIJAZ: [QuarterTone; 8] = [0, 2, 8, 10, 14, 16, 22, 24]
}

// Shruti (Indian) - 22 divisions
mod tuning·shruti {
    type Shruti = Cycle<22>

    const SA: Shruti = 0
    const RE_KOMAL: Shruti = 2
    const RE: Shruti = 4
    const GA_KOMAL: Shruti = 5
    // ... etc
}
```

### 11.2 Rhythmic Mathematics

```sigil
// Euclidean rhythms (Godfried Toussaint, drawing from African traditions)
fn euclidean·rhythm(pulses: u32, steps: u32) -> [bool] {
    // Distributes pulses as evenly as possible over steps
    // E(3,8) = [x . . x . . x .] = Cuban tresillo
    // E(5,8) = [x . x x . x x .] = Cuban cinquillo
    bresenham·line(pulses, steps)|τ{_ > 0}
}

// Tala (Indian rhythmic cycles)
struct Tala {
    name: str,
    beats: u32,
    subdivisions: [u32],  // Vibhag structure
}

const ADI_TALA: Tala = Tala {
    name: "Adi",
    beats: 8,
    subdivisions: [4, 2, 2],  // Chatusra-jati
}

const RUPAK_TALA: Tala = Tala {
    name: "Rupak",
    beats: 7,
    subdivisions: [3, 2, 2],
}

// Aksak (Balkan irregular meters)
// Expressed as combinations of 2s and 3s
type Aksak = [u32]  // e.g., [2,2,2,3] for 9/8

const DAICHOVO: Aksak = [2, 2, 2, 3]      // 9/8 Bulgarian
const KOPANITSA: Aksak = [2, 2, 3, 2, 2]  // 11/8 Bulgarian
const LESNOTO: Aksak = [3, 2, 2]          // 7/8 Macedonian

// Polyrhythm
struct Polyrhythm {
    layers: [(u32, u32)],  // (beats, per cycle)
}

let three_against_two = Polyrhythm {
    layers: [(3, 1), (2, 1)],
}

// Cross-rhythm analysis
fn cross_rhythm(a: u32, b: u32) -> [Onset] {
    let cycle = lcm(a, b)
    (0..cycle)|φ{t => t % a == 0 || t % b == 0}
}
```

### 11.3 Pitch and Frequency

```sigil
// Frequency as type
type Frequency = f64  // Hz
type Pitch = (NoteName, Octave)
type Cents = f64      // 1200 cents = 1 octave

// Conversions
fn pitch·to_freq(p: Pitch, tuning: Tuning) -> Frequency {
    tuning.reference * tuning.ratio(p)
}

fn freq·to_midi(f: Frequency) -> f64 {
    69 + 12 * log2(f / 440)
}

fn cents·between(a: Frequency, b: Frequency) -> Cents {
    1200 * log2(b / a)
}

// Harmonic series
fn harmonics(fundamental: Frequency, n: u32) -> [Frequency] {
    (1..=n)|τ{k => fundamental * k}
}

// Difference tones (Tartini tones)
fn difference·tone(a: Frequency, b: Frequency) -> Frequency {
    (a - b)|abs
}

// Beating frequency
fn beat·frequency(a: Frequency, b: Frequency) -> Frequency {
    (a - b)|abs
}
```

### 11.4 Scale Construction

```sigil
// Scale as pitch class set
type PitchClass = Cycle<12>  // In 12-TET
type Scale = Set<PitchClass>

const MAJOR: Scale = {0, 2, 4, 5, 7, 9, 11}
const MINOR_NATURAL: Scale = {0, 2, 3, 5, 7, 8, 10}
const PENTATONIC_MAJOR: Scale = {0, 2, 4, 7, 9}
const WHOLE_TONE: Scale = {0, 2, 4, 6, 8, 10}
const OCTATONIC: Scale = {0, 1, 3, 4, 6, 7, 9, 10}

// Raga (Indian) - ascending/descending may differ
struct Raga {
    name: str,
    arohana: [Shruti],   // Ascending
    avarohana: [Shruti], // Descending
    vadi: Shruti,        // Most important note
    samvadi: Shruti,     // Second most important
    time: TimeOfDay?,    // Associated time
    rasa: Emotion?,      // Associated emotion
}

// Mode generation (rotate scale)
fn mode(scale: Scale, degree: u32) -> Scale {
    scale|τ{pc => (pc - scale[degree]) mod 12}
}

// Scale operations
scale|transpose(n)      // Move all pitches by n
scale|invert            // Mirror around axis
scale|complement        // Notes NOT in scale
a|∩b                    // Common tones
a|∪b                    // Combined pitch content
```

### 11.5 Spectral Mathematics

```sigil
// Fourier analysis
fn fft(signal: [f64]) -> [Complex] {
    cooley·tukey(signal)
}

fn spectrum(signal: [f64], sample_rate: Frequency) -> [(Frequency, Amplitude)] {
    let coeffs = fft(signal)
    coeffs|enumerate|τ{(i, c) =>
        (i * sample_rate / signal.len(), c|magnitude)
    }
}

// Spectral centroid (brightness measure)
fn spectral·centroid(spectrum: [(Frequency, Amplitude)]) -> Frequency {
    let weighted = spectrum|τ{(f, a) => f * a}|Σ
    let total = spectrum|τ{(_, a) => a}|Σ
    weighted / total
}

// Consonance/dissonance (multiple models)
mod consonance {
    // Simple ratio model (Pythagoras)
    fn ratio_simplicity(a: Frequency, b: Frequency) -> f64 {
        let ratio = (a / b)|to_ratio|simplify
        1.0 / (ratio.numer + ratio.denom)
    }

    // Critical bandwidth model (Plomp-Levelt)
    fn plomp_levelt(a: Frequency, b: Frequency) -> f64 {
        let cb = critical_bandwidth((a + b) / 2)
        let x = (a - b)|abs / cb
        // Dissonance curve
        // ...
    }
}
```

---

## 12. Symbolic and Esoteric Mathematics

### 12.1 Gematria and Numerology

```sigil
// Letter-number correspondences
trait Gematria {
    fn value(self) -> u64
}

// Hebrew gematria
impl Gematria for Hebrew {
    fn value(self) -> u64 {
        match self {
            'א' => 1, 'ב' => 2, 'ג' => 3, /* ... */
            'י' => 10, 'כ' => 20, /* ... */
            'ק' => 100, 'ר' => 200, /* ... */
        }
    }
}

// Greek isopsephy
impl Gematria for Greek {
    fn value(self) -> u64 {
        match self {
            'α' => 1, 'β' => 2, 'γ' => 3, /* ... */
        }
    }
}

// Arabic abjad
impl Gematria for Arabic {
    fn value(self) -> u64 {
        match self {
            'ا' => 1, 'ب' => 2, 'ج' => 3, /* ... */
        }
    }
}

// Word value
fn word·value<G: Gematria>(word: [G]) -> u64 {
    word|τ{Gematria·value}|Σ
}

// Find words with same value
fn isopsephic(corpus: [Word], target: u64) -> [Word] {
    corpus|φ{w => w|word·value == target}
}
```

### 12.2 Sacred Geometry Ratios

```sigil
// Golden ratio
const φ: f64 = (1 + √5) / 2  // 1.618...
const Φ: f64 = 1 / φ         // 0.618... (reciprocal)

// Silver ratio
const δ_S: f64 = 1 + √2      // 2.414...

// Plastic ratio
const ρ: f64 = // Real root of x³ = x + 1

// Sacred proportions
const VESICA_PISCIS: f64 = √3
const SQRT_2: f64 = √2       // Ad quadratum
const SQRT_3: f64 = √3       // Ad triangulum
const SQRT_5: f64 = √5       // Pentagon diagonal

// Fibonacci-like sequences
fn fibonacci() -> impl Iterator<u64> {
    iterate((1, 1), |(a, b)| (b, a + b))|τ{|(a, _)| a}
}

fn lucas() -> impl Iterator<u64> {
    iterate((2, 1), |(a, b)| (b, a + b))|τ{|(a, _)| a}
}

fn pell() -> impl Iterator<u64> {
    iterate((0, 1), |(a, b)| (b, 2*b + a))|τ{|(a, _)| a}
}
```

### 12.3 I Ching Mathematics

```sigil
// Hexagram structure
type Line = enum { Yin, Yang, OldYin, OldYang }
type Trigram = [Line; 3]
type Hexagram = [Line; 6]

// 64 hexagrams as 6-bit patterns
type HexagramNumber = Cycle<64>

// King Wen sequence (traditional ordering)
const KING_WEN: [HexagramNumber; 64] = [
    1, 2, 3, 4, /* ... traditional sequence ... */
]

// Binary (Shao Yong) sequence
fn shao_yong(h: Hexagram) -> u8 {
    h|τ{line => if line.is_yang() { 1 } else { 0 }}
     |enumerate
     |τ{(i, b) => b << i}
     |Σ
}

// Transformations
fn complement(h: Hexagram) -> Hexagram {
    h|τ{Line·flip}
}

fn invert(h: Hexagram) -> Hexagram {
    h|reverse
}

fn nuclear(h: Hexagram) -> Hexagram {
    [h[1], h[2], h[3], h[2], h[3], h[4]]
}
```

---

## 13. Design Principles

1. **No privileged notation** — Western symbols are available but not default
2. **Multiple representations** — Same value, different cultural expressions
3. **Named origins** — Algorithms honor their discoverers/cultures
4. **Structural universals** — Category theory as neutral meeting ground
5. **Domain-appropriate bases** — Use base-60 for time, base-20 for calendrics
6. **Relational thinking** — Kinship and relationship as mathematical primitives
7. **Geometric plurality** — Euclidean is one option among many
8. **Temporal plurality** — Linear time is one model among many
9. **Harmonic universals** — Music theory from all traditions as first-class
10. **Symbolic bridges** — Gematria, sacred geometry as valid mathematical domains
