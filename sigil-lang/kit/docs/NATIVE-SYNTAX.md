# Sigil Native Syntax Reference

Sigil supports two syntax forms: **native symbols** and **ASCII keywords**. Both are semantically identical. Native symbols are preferred in new code for density and visual clarity.

---

## Symbol Table

### Keywords

| Native | ASCII | Description | Example |
|--------|-------|-------------|---------|
| `λ` | `fn` | Function definition | `λ add(a: i32, b: i32) → i32 { a + b }` |
| `Σ` | `struct` | Struct definition | `Σ Point { x: f64, y: f64 }` |
| `≔` | `let` | Variable binding | `≔ x = 42;` |
| `☉` | `pub` | Public visibility | `☉ λ public_fn() { }` |
| `⊢` | `impl` | Implementation block | `⊢ Point { λ new() → This { } }` |
| `∀` | `for` | For loop | `∀ i ∈ 0..10 { }` |
| `∈` | `in` | Loop iteration | `∀ item ∈ list { }` |
| `∞` | `loop` | Infinite loop | `∞ { if done { break; } }` |

### Operators and Punctuation

| Native | ASCII | Description | Example |
|--------|-------|-------------|---------|
| `→` | `->` | Return type arrow | `λ f() → i32` |
| `·` | `::` | Path separator | `Vec·new()`, `std·io` |
| `&Δ` | `&mut` | Mutable reference | `λ modify(&Δ this)` |
| `Δ` | `mut` | Mutable modifier | `≔ x: Δ i32 = 0;` |

### Evidence Markers

| Marker | Name | Meaning | Usage |
|--------|------|---------|-------|
| `!` | Known | Verified, computed, or certain | `≔ result = calculate()!` |
| `?` | Uncertain | Needs validation, could fail | `≔ input = read()?` |
| `~` | Reported | External data, unverified | `≔ data = fetch_api()~` |
| `‽` | Paradox | Self-referential, special | `≔ meta = self_describe()‽` |

---

## Typing Native Symbols

### macOS
- `λ` (lambda): Option + L (or configure keyboard)
- `Σ` (sigma): Option + W
- `≔` (assignment): Option + ; (may need custom mapping)
- `→` (arrow): Option + Shift + .
- `·` (middot): Option + Shift + 9

### Linux (X11/Wayland)
Configure compose keys or use:
- `Ctrl+Shift+U` then Unicode codepoint:
  - `λ` = 03BB
  - `Σ` = 03A3
  - `≔` = 2254
  - `→` = 2192
  - `·` = 00B7

### Windows
- Use WinCompose utility
- Or Alt codes with numeric keypad
- Or configure custom keyboard layout

### Editor Snippets

Most editors support snippet expansion:
- Type `fn` → expand to `λ`
- Type `struct` → expand to `Σ`
- Type `let` → expand to `≔`
- Type `::` → expand to `·`

See `TOOLING.md` for LSP setup which provides automatic completion.

---

## Migration from ASCII

Use the built-in migrate command:

```bash
# Preview changes (dry run)
sigil migrate file.sg --dry-run

# Migrate file in place
sigil migrate file.sg

# Migrate directory recursively
sigil migrate src/ --recursive
```

### Before Migration

```sigil
pub struct Point {
    x: f64,
    y: f64,
}

impl Point {
    pub fn new(x: f64, y: f64) -> Point {
        Point { x, y }
    }

    pub fn distance(&self, other: &Point) -> f64 {
        let dx = self.x - other.x;
        let dy = self.y - other.y;
        (dx * dx + dy * dy).sqrt()
    }
}

fn main() {
    let p1 = Point::new(0.0, 0.0);
    let p2 = Point::new(3.0, 4.0);
    println(p1.distance(&p2));
}
```

### After Migration

```sigil
☉ Σ Point {
    x: f64,
    y: f64,
}

⊢ Point {
    ☉ λ new(x: f64, y: f64) → Point {
        Point { x, y }
    }

    ☉ λ distance(&this, other: &Point) → f64 {
        ≔ dx = this.x - other.x;
        ≔ dy = this.y - other.y;
        (dx * dx + dy * dy).sqrt()
    }
}

λ main() {
    ≔ p1 = Point·new(0.0, 0.0);
    ≔ p2 = Point·new(3.0, 4.0);
    println(p1.distance(&p2));
}
```

---

## Why Native Symbols?

### Information Density

Native symbols reduce visual noise and increase semantic density:

```sigil
// ASCII (67 characters)
pub fn process(items: &mut Vec<Item>) -> Result<i32, Error>

// Native (52 characters)
☉ λ process(items: &Δ Vec<Item>) → Result<i32, Error>
```

### Visual Distinction

Symbols provide instant visual categorization:
- `λ` immediately signals "function"
- `Σ` immediately signals "data structure"
- `≔` immediately signals "binding"

### Mathematical Heritage

Programming shares roots with mathematics. Native symbols honor that heritage:
- `λ` from lambda calculus
- `Σ` from set theory (sum types)
- `∀` from predicate logic (universal quantifier)
- `→` from type theory (function arrow)

### Polyglot Future

As AI-assisted programming grows, code must be readable by both humans and machines. Symbolic notation is more universal than English keywords.

---

## Compatibility

| Context | Native | ASCII |
|---------|--------|-------|
| Source files | ✅ | ✅ |
| REPL | ✅ | ✅ |
| LSP | ✅ | ✅ |
| Formatter | ✅ | ✅ |
| Git diffs | ✅ | ✅ |
| Terminal output | ✅ (UTF-8) | ✅ |

Both forms can be mixed in the same file, though consistency is preferred.

---

## Quick Reference Card

```
┌──────────────────────────────────────────────────────────┐
│                   SIGIL SYMBOLS                          │
├──────────────────────────────────────────────────────────┤
│  λ = fn        Function definition                       │
│  Σ = struct    Data structure                            │
│  ≔ = let       Variable binding                          │
│  ☉ = pub       Public visibility                         │
│  ⊢ = impl      Implementation                            │
│  → = ->        Return type                               │
│  · = ::        Path separator                            │
│  &Δ = &mut     Mutable reference                         │
│  ∀ = for       Loop                                      │
│  ∈ = in        Iteration                                 │
│  ∞ = loop      Infinite loop                             │
├──────────────────────────────────────────────────────────┤
│                 EVIDENCE MARKERS                         │
├──────────────────────────────────────────────────────────┤
│  !  Known      Computed/verified                         │
│  ?  Uncertain  Needs validation                          │
│  ~  Reported   External data                             │
│  ‽  Paradox    Self-referential                          │
└──────────────────────────────────────────────────────────┘
```
