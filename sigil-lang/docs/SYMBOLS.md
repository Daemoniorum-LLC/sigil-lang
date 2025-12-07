# Sigil Unicode Symbol Reference

This document provides a comprehensive reference for all Unicode symbols supported in the Sigil programming language.

## Table of Contents
- [Evidentiality Markers](#evidentiality-markers)
- [Greek Letter Morphemes](#greek-letter-morphemes)
- [Access Morphemes](#access-morphemes)
- [Data Operation Symbols](#data-operation-symbols)
- [Set Theory Operators](#set-theory-operators)
- [Logic Operators](#logic-operators)
- [Bitwise Operators](#bitwise-operators)
- [Category Theory](#category-theory)
- [Analysis/Calculus](#analysiscalculus)
- [Special Literals](#special-literals)
- [Aspect Morphemes](#aspect-morphemes)
- [Quantifiers](#quantifiers)

---

## Evidentiality Markers

Sigil uses evidentiality markers to track the provenance and trustworthiness of data:

| Symbol | Name | ASCII | Meaning | Example |
|--------|------|-------|---------|---------|
| `!` | Bang | `!` | Known/direct - verified, trusted data | `let x! = compute();` |
| `?` | Question | `?` | Uncertain - possibly absent | `let x? = maybe_get();` |
| `~` | Tilde | `~` | Reported - from external source | `let x~ = fetch_api();` |
| `‽` | Interrobang | N/A | Paradox - trust boundary | `let x‽ = contradictory();` |

---

## Greek Letter Morphemes

### Transform Operations (Pipe Syntax)

| Symbol | Name | Function | Usage |
|--------|------|----------|-------|
| `τ` / `Τ` | Tau | Transform/map | `data \|τ{_ * 2}` |
| `φ` / `Φ` | Phi | Filter | `data \|φ{_ > 0}` |
| `σ` / `Σ` | Sigma | Sort / Sum | `data \|σ` (sort), `data \|Σ` (sum) |
| `ρ` / `Ρ` | Rho | Reduce/fold | `data \|ρ{acc + _}` |
| `λ` / `Λ` | Lambda | Lambda function | `λ x -> x + 1` |
| `Π` | Pi | Product | `data \|Π` |

### Other Greek Letters

| Symbol | Name | Purpose |
|--------|------|---------|
| `δ` / `Δ` | Delta | Difference/change |
| `ε` | Epsilon | Empty/null |
| `ζ` | Zeta | Zip/combine |

---

## Access Morphemes

These morphemes extract elements from collections:

| Symbol | Name | Function | Pipe Usage | Function Usage |
|--------|------|----------|------------|----------------|
| `α` | Alpha | First element | `data \|α` | `first(data)` |
| `ω` / `Ω` | Omega | Last element | `data \|ω` | `last(data)` |
| `μ` / `Μ` | Mu | Middle/median | `data \|μ` | `middle(data)` |
| `χ` / `Χ` | Chi | Random choice | `data \|χ` | `choice(data)` |
| `ν` / `Ν` | Nu | Nth element | `data \|ν{2}` | `nth(data, 2)` |
| `ξ` / `Ξ` | Xi | Next (iterator) | `data \|ξ` | `next(data)` |

### Examples

```sigil
let numbers = [10, 20, 30, 40, 50];

// Using pipe syntax
let first = numbers |α;      // 10
let last = numbers |ω;       // 50
let mid = numbers |μ;        // 30
let random = numbers |χ;     // random element
let third = numbers |ν{2};   // 30 (0-indexed)

// Using functions
let first = first(numbers);
let mid = middle(numbers);
let rand = choice(numbers);
```

---

## Data Operation Symbols

| Symbol | Name | Function | Usage |
|--------|------|----------|-------|
| `⋈` | Bowtie | Zip with operation | `zip_with(a, b, "add")` |
| `⋳` | Element-bar | Flatten | `flatten(nested)` |
| `⊔` | Square Cup | Lattice join (max) | `supremum(a, b)` |
| `⊓` | Square Cap | Lattice meet (min) | `infimum(a, b)` |

### Examples

```sigil
let a = [1, 2, 3];
let b = [10, 20, 30];

// Element-wise operations
let sums = zip_with(a, b, "add");     // [11, 22, 33]
let products = zip_with(a, b, "mul"); // [10, 40, 90]

// Lattice operations
let max_val = supremum(5, 10);        // 10
let min_val = infimum(5, 10);         // 5

// Element-wise max/min
let maxes = supremum([1, 5, 3], [2, 4, 6]);  // [2, 5, 6]
let mins = infimum([1, 5, 3], [2, 4, 6]);    // [1, 4, 3]
```

---

## Set Theory Operators

| Symbol | Name | Meaning |
|--------|------|---------|
| `∪` | Union | Set union |
| `∩` | Intersection | Set intersection |
| `∖` | Set Minus | Set difference |
| `⊂` | Subset | Proper subset |
| `⊆` | SubsetEq | Subset or equal |
| `⊃` | Superset | Proper superset |
| `⊇` | SupersetEq | Superset or equal |
| `∈` | ElementOf | Membership test |
| `∉` | NotElementOf | Non-membership |

---

## Logic Operators

| Symbol | Name | ASCII | Meaning |
|--------|------|-------|---------|
| `∧` | LogicAnd | `&&` | Logical AND |
| `∨` | LogicOr | `\|\|` | Logical OR |
| `¬` | LogicNot | `!` | Logical NOT |
| `⊻` | LogicXor | `^` | Exclusive OR |
| `⊤` | Top | N/A | True/any type |
| `⊥` | Bottom | N/A | False/never type |

---

## Bitwise Operators

| Symbol | Name | ASCII | Meaning |
|--------|------|-------|---------|
| `⋏` | BitwiseAnd | `&` | Bitwise AND |
| `⋎` | BitwiseOr | N/A | Bitwise OR |
| `^` | Caret | `^` | Bitwise XOR |
| `<<` | Shl | `<<` | Shift left |
| `>>` | Shr | `>>` | Shift right |

### Examples

```sigil
let a = 0b1100;
let b = 0b1010;

let and_result = a ⋏ b;   // 0b1000 = 8
let or_result = a ⋎ b;    // 0b1110 = 14
let xor_result = a ^ b;   // 0b0110 = 6
```

---

## Category Theory

| Symbol | Name | Meaning |
|--------|------|---------|
| `∘` | Compose | Function composition |
| `⊗` | Tensor | Tensor product |
| `⊕` | DirectSum | Direct sum / XOR |

---

## Analysis/Calculus

| Symbol | Name | Meaning |
|--------|------|---------|
| `∫` | Integral | Cumulative sum |
| `∂` | Partial | Discrete derivative |
| `√` | Sqrt | Square root |
| `∛` | Cbrt | Cube root |

---

## Special Literals

| Symbol | Name | Value |
|--------|------|-------|
| `∅` | Empty | Void/emptiness |
| `∞` | Infinity | Unbounded value |
| `◯` | Circle | Geometric zero |
| `null` | Null | Null value |

---

## Aspect Morphemes

Verb aspects modify function semantics:

| Token | Aspect | Meaning | Example |
|-------|--------|---------|---------|
| `·ing` | Progressive | Ongoing/streaming | `fn stream·ing()` |
| `·ed` | Perfective | Completed | `fn process·ed()` |
| `·able` | Potential | Capability check | `fn parse·able()` |
| `·ive` | Resultative | Produces result | `fn destruct·ive()` |

### Examples

```sigil
// Progressive aspect - ongoing operation
fn read·ing(source: Reader) -> Stream {
    // Returns a stream for ongoing reading
}

// Perfective aspect - completed operation
fn process·ed(data: Data) -> Result {
    // Returns completed result
}

// Potential aspect - capability check
fn parse·able(input: String) -> Bool {
    // Returns whether parsing is possible
}

// Resultative aspect - produces result
fn destruct·ive(object: Object) -> Parts {
    // Produces result by destruction
}
```

---

## Quantifiers

| Symbol | Name | Meaning |
|--------|------|---------|
| `∀` | ForAll | Universal quantification |
| `∃` | Exists | Existential quantification |

---

## Keyboard Input

### macOS
- Use the Character Viewer (Edit > Emoji & Symbols or Ctrl+Cmd+Space)
- Create custom keyboard shortcuts in System Preferences

### Linux
- Use Compose key sequences
- Use `Ctrl+Shift+U` followed by Unicode codepoint

### Windows
- Use Win+. for emoji panel
- Use Alt codes or Character Map

### IDE Support
Most IDEs support:
- Unicode input with backslash notation: `\tau` -> `τ`
- Autocomplete for symbol names
- Custom snippets for frequently used symbols

---

---

## I Ching Trigrams

The 8 fundamental trigrams used in spirituality functions:

| Symbol | Name | Element | Meaning |
|--------|------|---------|---------|
| `☰` | Qian | Heaven | Creative, strong, initiating |
| `☱` | Dui | Lake | Joyous, pleasure, openness |
| `☲` | Li | Fire | Clinging, clarity, awareness |
| `☳` | Zhen | Thunder | Arousing, movement, action |
| `☴` | Xun | Wind | Gentle, penetrating, flexible |
| `☵` | Kan | Water | Abysmal, danger, depth |
| `☶` | Gen | Mountain | Keeping Still, meditation |
| `☷` | Kun | Earth | Receptive, yielding, devoted |

---

## Zodiac Signs

Used by `zodiac()` function:

| Symbol | Sign | Element | Dates |
|--------|------|---------|-------|
| `♈` | Aries | Fire | Mar 21 - Apr 19 |
| `♉` | Taurus | Earth | Apr 20 - May 20 |
| `♊` | Gemini | Air | May 21 - Jun 20 |
| `♋` | Cancer | Water | Jun 21 - Jul 22 |
| `♌` | Leo | Fire | Jul 23 - Aug 22 |
| `♍` | Virgo | Earth | Aug 23 - Sep 22 |
| `♎` | Libra | Air | Sep 23 - Oct 22 |
| `♏` | Scorpio | Water | Oct 23 - Nov 21 |
| `♐` | Sagittarius | Fire | Nov 22 - Dec 21 |
| `♑` | Capricorn | Earth | Dec 22 - Jan 19 |
| `♒` | Aquarius | Air | Jan 20 - Feb 18 |
| `♓` | Pisces | Water | Feb 19 - Mar 20 |

---

## Waveform Morphemes

Audio synthesis symbols:

| Symbol | Name | Function |
|--------|------|----------|
| `∿` | Sine | `sine_wave(freq, t)` |
| `⊓` | Square | `square_wave(freq, t)` |
| `⋀` | Sawtooth | `sawtooth_wave(freq, t)` |
| `△` | Triangle | `triangle_wave(freq, t)` |

---

## Cultural Color Systems

### Wu Xing (五行) Elements

| Element | Chinese | Color | Direction |
|---------|---------|-------|-----------|
| Wood | 木 | Green | East |
| Fire | 火 | Red | South |
| Earth | 土 | Yellow | Center |
| Metal | 金 | White/Gold | West |
| Water | 水 | Black/Blue | North |

### Chakra Colors

| Chakra | Sanskrit | Color | Frequency |
|--------|----------|-------|-----------|
| Root | मूलाधार | Red | 396 Hz |
| Sacral | स्वाधिष्ठान | Orange | 417 Hz |
| Solar | मणिपूर | Yellow | 528 Hz |
| Heart | अनाहत | Green | 639 Hz |
| Throat | विशुद्ध | Blue | 741 Hz |
| Third Eye | आज्ञा | Indigo | 852 Hz |
| Crown | सहस्रार | Violet | 963 Hz |

---

## Quick Reference Card

```
EVIDENTIALITY:  !  ?  ~  ‽
TRANSFORM:      τ  φ  σ  ρ  λ  Π  Σ
ACCESS:         α  ω  μ  χ  ν  ξ
DATA OPS:       ⋈  ⋳  ⊔  ⊓
SETS:           ∪  ∩  ∖  ⊂  ⊆  ⊃  ⊇  ∈  ∉
LOGIC:          ∧  ∨  ¬  ⊻  ⊤  ⊥
BITWISE:        ⋏  ⋎
MATH:           ∘  ⊗  ⊕  ∫  ∂  √  ∛
LITERALS:       ∅  ∞  ◯
QUANTIFIERS:    ∀  ∃
ASPECTS:        ·ing  ·ed  ·able  ·ive
I CHING:        ☰  ☱  ☲  ☳  ☴  ☵  ☶  ☷
ZODIAC:         ♈  ♉  ♊  ♋  ♌  ♍  ♎  ♏  ♐  ♑  ♒  ♓
WAVEFORMS:      ∿  ⊓  ⋀  △
```
