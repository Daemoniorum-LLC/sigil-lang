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

## 2. Numeric Plurality and Base Systems

Sigil treats numeral systems as first-class — not merely syntax sugar for decimal, but distinct mathematical objects with their own properties, operations, and cultural histories.

### 2.1 Historical Numeral Systems

#### Positional Systems

```sigil
// Standard positional bases with prefixes
let decimal     = 42              // Base 10 (Hindu-Arabic)
let binary      = 0b101010        // Base 2 (Leibniz, but also Pingala, Ifá)
let octal       = 0o52            // Base 8
let hex         = 0x2A            // Base 16
let vigesimal   = 0v22            // Base 20 (Mayan, Aztec, Basque, Celtic)
let sexagesimal = 0s42            // Base 60 (Babylonian, still used in time/angles)
let duodecimal  = 0z36            // Base 12 (widespread: 12 hours, 12 inches, dozens)

// Explicit base annotation for any radix
let base7 = 60:₇                  // 42 in base 7
let base36 = 16:₃₆                // 42 in base 36 (alphanumeric)
let base62 = g:₆₂                 // 42 in base 62 (0-9, A-Z, a-z)
```

#### Babylonian Cuneiform (Base 60)

```sigil
// Babylonian used two symbols: 𒐕 (1) and 𒌋 (10)
// Combined additively within each position, then positional for 60s
mod numeral·babylonian {
    type Digit = u8  // 0-59 per position
    type Cuneiform = [Digit]  // Most significant first

    // 42 = 42 (single position)
    // 3661 = 1·60² + 1·60 + 1 = [1, 1, 1]

    fn to_cuneiform(n: u64) -> Cuneiform {
        if n == 0 { return [] }
        let mut digits = []
        let mut remaining = n
        while remaining > 0 {
            digits|push_front(remaining % 60)
            remaining /= 60
        }
        digits
    }

    fn render(c: Cuneiform) -> str {
        c|τ{d => render_digit(d)}|join(" ")
    }

    fn render_digit(d: Digit) -> str {
        let tens = d / 10
        let ones = d % 10
        "𒌋"|repeat(tens) ++ "𒐕"|repeat(ones)
    }
}

let babylon_42 = 42|numeral·babylonian·render  // "𒌋𒌋𒌋𒌋𒐕𒐕"
```

#### Mayan Numerals (Base 20)

```sigil
mod numeral·mayan {
    // Mayan used dots (•) for 1-4, bars (—) for 5
    // Shell glyph (𝋠) for zero
    // Vertical stacking for positions (20⁰, 20¹, 20², ...)

    type MayanDigit = u8  // 0-19

    fn render_digit(d: MayanDigit) -> str {
        if d == 0 { return "𝋠" }  // Shell (zero)
        let bars = d / 5
        let dots = d % 5
        let bar_line = "—"|repeat(bars)|join("")
        let dot_line = "•"|repeat(dots)|join("")
        "{dot_line}\n{bar_line}"
    }

    // Long Count calendar uses modified base: 20, 18, 20, 20, 20...
    // (18 in position 1 because 18×20 = 360 ≈ solar year)
    type LongCount = [u8; 5]  // [baktun, katun, tun, winal, kin]

    const LONG_COUNT_RADIX: [u64; 5] = [
        144000,  // baktun = 20×18×20×20
        7200,    // katun = 18×20×20
        360,     // tun = 18×20
        20,      // winal = 20
        1,       // kin
    ]

    fn from_days(days: u64) -> LongCount {
        let mut remaining = days
        let mut result = [0u8; 5]
        for (i, radix) in LONG_COUNT_RADIX|enumerate {
            result[i] = (remaining / radix) as u8
            remaining %= radix
        }
        result
    }
}

let today = 1_872_000|numeral·mayan·from_days  // [13, 0, 11, ...]
```

#### Chinese Rod Numerals

```sigil
mod numeral·rod {
    // Chinese counting rods: vertical (|) and horizontal (—)
    // Alternating orientation by position to avoid ambiguity

    enum Orientation { Vertical, Horizontal }

    fn digit_rods(d: u8, pos: u32) -> str {
        let orient = if pos % 2 == 0 { Vertical } else { Horizontal }
        match (d, orient) {
            (0, _) => " ",
            (1, Vertical) => "𝍠",
            (2, Vertical) => "𝍡",
            (3, Vertical) => "𝍢",
            // ... up to 9
            (1, Horizontal) => "𝍩",
            // ...
        }
    }

    // Rod calculus supports negative numbers via color
    // Red rods = positive, Black rods = negative
    enum Sign { Red, Black }

    struct RodNumber {
        sign: Sign,
        digits: [u8],
    }
}
```

#### Egyptian Hieratic/Demotic

```sigil
mod numeral·egyptian {
    // Additive system: symbols for 1, 10, 100, 1000, etc.
    // 𓏺 = 1 (stroke)
    // 𓎆 = 10 (arch)
    // 𓍢 = 100 (coil of rope)
    // 𓆼 = 1000 (lotus)
    // 𓂭 = 10000 (finger)
    // 𓆐 = 100000 (tadpole)
    // 𓁨 = 1000000 (god with raised arms)

    fn to_hieroglyphic(n: u64) -> str {
        let mut result = ""
        let mut remaining = n

        result ++= "𓁨"|repeat((remaining / 1_000_000) as usize)
        remaining %= 1_000_000
        result ++= "𓆐"|repeat((remaining / 100_000) as usize)
        remaining %= 100_000
        // ... and so on

        result
    }
}
```

### 2.2 Balanced Number Systems

Balanced bases use signed digits centered around zero — powerful for certain computations.

#### Balanced Ternary (Setun Computer)

```sigil
mod numeral·balanced_ternary {
    // Digits: T (-1), 0, 1  (or: -, 0, +)
    // Used in the Soviet Setun computer (1958)
    // Advantages: negation is just flipping signs, no separate sign bit

    type Trit = enum { N = -1, Z = 0, P = 1 }  // Negative, Zero, Positive
    type BalancedTernary = [Trit]

    fn from_int(mut n: i64) -> BalancedTernary {
        let mut trits = []
        while n != 0 {
            let rem = n % 3
            let (trit, carry) = match rem {
                0 => (Z, 0),
                1 => (P, 0),
                2 => (N, 1),   // 2 = 3 - 1, so -1 with carry
                -1 => (N, 0),
                -2 => (P, -1), // -2 = -3 + 1, so +1 with borrow
                _ => unreachable!(),
            }
            trits|push_front(trit)
            n = n / 3 + carry
        }
        trits
    }

    fn negate(bt: BalancedTernary) -> BalancedTernary {
        bt|τ{t => match t { N => P, Z => Z, P => N }}
    }

    fn to_string(bt: BalancedTernary) -> str {
        bt|τ{t => match t { N => "T", Z => "0", P => "1" }}|join("")
    }
}

let forty_two = 42|balanced_ternary·from_int  // 1TT00 (1×81 - 1×27 - 1×9 + 0×3 + 0×1 = 81-27-9 = 45... wait)
// Actually: 42 = 27 + 27 - 9 - 3 = 1×27 + 1×27... let me recalculate
// 42 = 1×27 + 1×9 + 1×3 + 1×3 ... = 1T1T0
```

#### Balanced Quinary, Septenary, etc.

```sigil
// Generalized balanced base
mod numeral·balanced<const B: u32> where B % 2 == 1 {
    // For odd base B, digits range from -(B-1)/2 to +(B-1)/2

    const HALF: i32 = (B as i32 - 1) / 2

    type Digit = i32  // Range: -HALF..=HALF

    fn from_int(mut n: i64) -> [Digit] {
        let mut digits = []
        while n != 0 {
            let mut rem = (n % B as i64) as i32
            if rem > HALF {
                rem -= B as i32
                n += B as i64
            } else if rem < -HALF {
                rem += B as i32
                n -= B as i64
            }
            digits|push_front(rem)
            n /= B as i64
        }
        digits
    }
}

type BalancedQuinary = numeral·balanced<5>   // Digits: -2, -1, 0, 1, 2
type BalancedSeptenary = numeral·balanced<7> // Digits: -3, -2, -1, 0, 1, 2, 3
```

### 2.3 Non-Integer Bases

Bases need not be integers — irrational and transcendental bases have remarkable properties.

#### Golden Ratio Base (Phinary, Base φ)

```sigil
mod numeral·phinary {
    // Base φ = (1 + √5) / 2 ≈ 1.618
    // Uses digits 0 and 1 only
    // Key property: φ² = φ + 1, so 100_φ = 11_φ
    // Unique representation if no consecutive 1s (Zeckendorf)

    const φ: f64 = (1.0 + 5.0|sqrt) / 2.0

    type Phinary = [bool]  // true = 1, false = 0

    // Every positive integer has a finite representation!
    fn from_int(n: u64) -> Phinary {
        // Use Zeckendorf representation (Fibonacci coding)
        let fibs = fibonacci()|take_while{f => f <= n}|collect
        let mut remaining = n
        let mut bits = []

        for f in fibs|reverse {
            if remaining >= f {
                bits|push(true)
                remaining -= f
            } else {
                bits|push(false)
            }
        }
        bits|reverse
    }

    fn to_float(p: Phinary) -> f64 {
        p|enumerate|τ{(i, bit) =>
            if bit { φ ** (p.len() - 1 - i) as f64 } else { 0.0 }
        }|Σ
    }

    // Normalize: replace 011 with 100 (since φ + 1 = φ²)
    fn normalize(p: Phinary) -> Phinary {
        // Remove consecutive 1s
        let mut result = p·clone
        loop {
            let mut changed = false
            for i in 0..result.len()-1 {
                if result[i] && result[i+1] {
                    result[i] = false
                    result[i+1] = false
                    if i > 0 {
                        // Carry to next position
                        result[i-1] = !result[i-1]  // Toggle (with possible cascade)
                    } else {
                        result|push_front(true)
                    }
                    changed = true
                }
            }
            if !changed { break }
        }
        result
    }
}

// 42 in phinary (Zeckendorf: 34 + 8 = F₉ + F₆)
let forty_two_phi = 42|phinary·from_int  // 10100100 (representing fib positions)
```

#### Base e (Natural Logarithm Base)

```sigil
mod numeral·base_e {
    // Base e ≈ 2.71828...
    // Digits: 0, 1, 2 (since e > 2)
    // Representations are not unique, but greedy algorithm works

    const E: f64 = 2.718281828459045

    fn from_float(x: f64, precision: u32) -> [u8] {
        let mut digits = []
        let mut remaining = x

        // Integer part
        let mut power = 0
        while E ** (power + 1) as f64 <= remaining {
            power += 1
        }

        for p in ((-precision as i32)..=power)|reverse {
            let place_value = E ** p as f64
            let digit = (remaining / place_value)|floor as u8
            digits|push(digit|min(2))  // Cap at 2
            remaining -= digit as f64 * place_value
        }
        digits
    }
}
```

#### Base π, Base √2

```sigil
mod numeral·base_pi {
    const π: f64 = 3.14159265358979

    // Base π uses digits 0, 1, 2, 3
    fn from_float(x: f64, precision: u32) -> [u8] {
        // Similar greedy algorithm
        // ...
    }
}

mod numeral·base_sqrt2 {
    const √2: f64 = 1.41421356237

    // Base √2 is interesting: √2² = 2, so 100_√2 = 2₁₀
    // Uses digits 0, 1
    // Related to paper sizes (A0, A1, A2... each is √2 ratio)
}
```

### 2.4 Negative Bases

Negative bases can represent all integers without a sign!

#### Negabinary (Base -2)

```sigil
mod numeral·negabinary {
    // Base -2: no sign needed for negative numbers!
    // Digits: 0, 1

    fn from_int(mut n: i64) -> [bool] {
        if n == 0 { return [false] }

        let mut bits = []
        while n != 0 {
            let mut rem = n % -2
            n /= -2
            if rem < 0 {
                rem += 2
                n += 1
            }
            bits|push_front(rem == 1)
        }
        bits
    }

    fn to_int(bits: [bool]) -> i64 {
        bits|enumerate|τ{(i, bit) =>
            if bit { (-2i64) ** (bits.len() - 1 - i) } else { 0 }
        }|Σ
    }
}

let neg_42 = (-42)|negabinary·from_int  // 110110 (no minus sign!)
let pos_42 = 42|negabinary·from_int     // 1010110
```

#### Negadecimal (Base -10)

```sigil
mod numeral·negadecimal {
    // Base -10: digits 0-9, alternating sign by position

    fn from_int(mut n: i64) -> [u8] {
        let mut digits = []
        while n != 0 {
            let mut rem = n % -10
            n /= -10
            if rem < 0 {
                rem += 10
                n += 1
            }
            digits|push_front(rem as u8)
        }
        digits
    }
}
```

### 2.5 Complex Bases

Bases in the complex plane can represent Gaussian integers!

#### Quater-Imaginary (Base 2i)

```sigil
mod numeral·quater_imaginary {
    // Base 2i (where i = √-1)
    // Digits: 0, 1, 2, 3
    // Can represent ALL Gaussian integers (a + bi where a,b ∈ ℤ)
    // No sign needed!

    use complex·{Complex, I}

    fn from_gaussian(z: Complex<i64>) -> [u8] {
        let mut digits = []
        let mut remaining = z

        while remaining != 0 {
            // z mod 2i: find digit 0,1,2,3 such that (z - d) is divisible by 2i
            let d = ((remaining.re % 4) + 4) % 4
            digits|push_front(d as u8)

            // Divide by 2i: multiply by -i/2 = i/2 * -1
            remaining = (remaining - d) / (2 * I)
        }
        digits
    }

    fn to_gaussian(digits: [u8]) -> Complex<i64> {
        digits|enumerate|τ{(i, d) =>
            d as Complex * (2 * I) ** (digits.len() - 1 - i)
        }|Σ
    }
}

// Complex number 5 + 3i in quater-imaginary
let z = Complex(5, 3)|quater_imaginary·from_gaussian  // Some sequence of 0,1,2,3
```

#### Base (-1 + i)

```sigil
mod numeral·dragon_base {
    // Base (-1 + i), also called "dragon base"
    // Digits: 0, 1
    // Related to the dragon curve fractal!

    use complex·{Complex, I}

    const BASE: Complex<i64> = Complex(-1, 1)

    fn from_gaussian(z: Complex<i64>) -> [bool] {
        let mut bits = []
        let mut remaining = z

        while remaining != Complex(0, 0) {
            let d = remaining.re & 1  // Parity of real part
            bits|push_front(d == 1)
            remaining = (remaining - d) / BASE
        }
        bits
    }

    // Iteration of digit map traces dragon curve!
}
```

### 2.6 Mixed Radix Systems

Different positions use different bases — natural for human-scale quantities.

```sigil
mod numeral·mixed_radix {
    // A mixed radix system has different base per position
    type MixedRadix = [u64]  // Radix for each position

    // Time: seconds-minutes-hours
    const TIME_HMS: MixedRadix = [60, 60, 24]

    // Old British currency: pence-shillings-pounds (pre-1971)
    const OLD_BRITISH: MixedRadix = [12, 20]  // 12 pence/shilling, 20 shillings/pound

    // Degrees-minutes-seconds (like Babylonian)
    const DMS: MixedRadix = [60, 60]

    // Mayan Long Count (with irregular 18 in second position)
    const MAYAN_LONG: MixedRadix = [20, 18, 20, 20, 20]

    fn to_mixed(n: u64, radices: &MixedRadix) -> [u64] {
        let mut digits = []
        let mut remaining = n
        for radix in radices {
            digits|push_front(remaining % radix)
            remaining /= radix
        }
        if remaining > 0 {
            digits|push_front(remaining)
        }
        digits
    }

    fn from_mixed(digits: &[u64], radices: &MixedRadix) -> u64 {
        let mut result = 0u64
        let mut multiplier = 1u64
        for (digit, radix) in digits|reverse|zip(radices) {
            result += digit * multiplier
            multiplier *= radix
        }
        result
    }
}

// 90061 seconds = 1 day, 1 hour, 1 minute, 1 second
let time = 90061|mixed_radix·to_mixed(TIME_HMS)  // [1, 1, 1, 1]
```

### 2.7 Bijective Numeration

No zero! Every positive integer has exactly one representation.

```sigil
mod numeral·bijective {
    // Bijective base-k: digits are 1, 2, 3, ..., k (no zero!)
    // Used in spreadsheet columns: A, B, C, ... Z, AA, AB, ...

    fn to_bijective<const K: u32>(mut n: u64) -> [u8] {
        let mut digits = []
        while n > 0 {
            let mut d = n % K as u64
            if d == 0 {
                d = K as u64
                n -= 1
            }
            digits|push_front(d as u8)
            n /= K as u64
        }
        digits
    }

    // Spreadsheet column names (bijective base-26)
    fn to_column(n: u64) -> str {
        to_bijective::<26>(n)|τ{d => ('A' as u8 + d - 1) as char}|collect
    }

    fn from_column(s: &str) -> u64 {
        s.chars()|τ{c => (c as u8 - 'A' as u8 + 1) as u64}
         |enumerate|τ{(i, d) => d * 26u64 ** (s.len() - 1 - i)}
         |Σ
    }
}

let col = 42|bijective·to_column  // "AP" (1×26 + 16 = 42)
```

### 2.8 Residue Number Systems

Represent numbers by their residues modulo coprime bases — enables parallel arithmetic!

```sigil
mod numeral·residue {
    // RNS: represent n by (n mod m₁, n mod m₂, ..., n mod mₖ)
    // where m₁, m₂, ... are pairwise coprime
    // Range: 0 to M-1 where M = m₁ × m₂ × ... × mₖ

    struct RNS<const N: usize> {
        moduli: [u64; N],
        residues: [u64; N],
    }

    fn encode<const N: usize>(n: u64, moduli: &[u64; N]) -> RNS<N> {
        let residues = moduli|τ{m => n % m}|collect_array
        RNS { moduli: *moduli, residues }
    }

    fn decode<const N: usize>(rns: &RNS<N>) -> u64 {
        // Chinese Remainder Theorem
        sunzi·crt(rns.residues|zip(rns.moduli)|collect())
    }

    // Addition is parallel and carry-free!
    fn add<const N: usize>(a: &RNS<N>, b: &RNS<N>) -> RNS<N> {
        let residues = a.moduli|zip(a.residues)|zip(b.residues)
            |τ{((m, ra), rb) => (ra + rb) % m}
            |collect_array
        RNS { moduli: a.moduli, residues }
    }

    // Multiplication is also parallel!
    fn mul<const N: usize>(a: &RNS<N>, b: &RNS<N>) -> RNS<N> {
        let residues = a.moduli|zip(a.residues)|zip(b.residues)
            |τ{((m, ra), rb) => (ra * rb) % m}
            |collect_array
        RNS { moduli: a.moduli, residues }
    }
}

// Common RNS for 32-bit range
const RNS_MODULI: [u64; 4] = [257, 263, 269, 271]  // Product > 2³²

let a = 1000|residue·encode(RNS_MODULI)  // [229, 211, 193, 187]
let b = 2000|residue·encode(RNS_MODULI)  // [201, 159, 117, 103]
let c = residue·mul(&a, &b)               // Parallel multiplication!
let result = c|residue·decode             // 2000000
```

### 2.9 p-Adic Numbers

Infinite expansions to the left, used in number theory and cryptography.

```sigil
mod numeral·p_adic<const P: u64> where P·is_prime {
    // p-adic integers: ...d₂d₁d₀ (infinite to the left!)
    // Convergence is measured differently: numbers "close" if they
    // agree on many rightmost digits

    // Finite approximation to n digits
    struct PAdicApprox {
        digits: [u8],  // Rightmost first
        precision: u32,
    }

    // p-adic norm: |n|_p = p^(-v) where p^v || n
    fn norm(n: i64) -> f64 {
        if n == 0 { return 0.0 }
        let mut v = 0
        let mut m = n|abs
        while m % P == 0 {
            v += 1
            m /= P
        }
        (P as f64) ** (-v as f64)
    }

    // In p-adics, negative numbers have infinite representations!
    // -1 in p-adic = ....(p-1)(p-1)(p-1) (all digits = p-1)
    fn neg_one(precision: u32) -> PAdicApprox {
        PAdicApprox {
            digits: [(P - 1) as u8]|repeat(precision)|collect,
            precision,
        }
    }

    // Addition with carry propagating leftward
    fn add(a: &PAdicApprox, b: &PAdicApprox) -> PAdicApprox {
        let mut digits = []
        let mut carry = 0u8
        for (da, db) in a.digits|zip(b.digits) {
            let sum = da + db + carry
            digits|push(sum % P as u8)
            carry = sum / P as u8
        }
        PAdicApprox { digits, precision: a.precision }
    }
}

// 2-adic numbers
type Dyadic = numeral·p_adic<2>

// In 2-adics: ...1111 + 1 = 0 (so ...1111 represents -1!)
let minus_one = Dyadic·neg_one(8)  // [1,1,1,1,1,1,1,1]
let one = PAdicApprox { digits: [1], precision: 8 }
let zero = Dyadic·add(&minus_one, &one)  // [0,0,0,0,0,0,0,0]
```

### 2.10 Number Systems as Types

```sigil
// All numeral systems implement a common trait
trait NumeralSystem {
    type Digit
    type Representation

    fn encode(n: i64) -> Self·Representation
    fn decode(repr: &Self·Representation) -> i64
    fn radix(self) -> Option<i64>  // None for non-positional
}

// Type-safe conversions
let x: Decimal = 42
let y: Sexagesimal = x·to_base(60)
let z: Binary = x·to_base(2)
let w: BalancedTernary = x·to_balanced(3)
let φ: Phinary = x·to_phinary()

// Domain-specific defaults
//@ rune: default_base(60)
mod astronomy {
    let angle = 45·30·0  // degrees, minutes, seconds
}

//@ rune: default_base(-2)
mod two_complement_free {
    let x = 42   // Stored as negabinary, no sign bit needed!
    let y = -42  // Also no sign bit!
}
```

### 2.11 Śūnya (Zero) and Ananta (Infinity)

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
