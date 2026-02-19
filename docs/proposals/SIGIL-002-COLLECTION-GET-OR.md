# SIGIL-002: Collection `get_or()` Method

**Status**: Draft
**Author**: Orpheus TDD Session
**Date**: 2026-02-12
**Affects**: Standard Library (Collections)

---

## Abstract

This proposal adds a `get_or(index, default)` method to collection types, providing ergonomic bounds-checked access with a fallback value. This replaces the verbose Rust pattern of `.get().copied().unwrap_or()`.

---

## Motivation

When implementing the Orpheus UI mixer tests, a common pattern emerged for bounds-checked access:

```sigil
// Rust-style pattern (doesn't work - no copied() on Option)
self.levels.get(index).copied().unwrap_or((-60.0, -60.0))

// Current workaround in Sigil
⎇ index < self.levels.len() {
    self.levels[index]
} ⎉ {
    (-60.0, -60.0)
}
```

The workaround is correct but verbose for a common operation. The Rust pattern requires three method calls, and Sigil currently lacks `copied()` on `Option`.

---

## Proposal

Add a single method that combines bounds checking with default value:

```sigil
// New method
self.levels.get_or(index, (-60.0, -60.0))
```

### Semantics

```sigil
⊢ Vec<T> {
    /// Returns element at index, or default if out of bounds.
    ///
    /// Equivalent to:
    ///   if index < self.len() { self[index] } else { default }
    rite get_or(&self, index: usize, default: T) -> T! {
        ⎇ index < self.len() {
            self[index]
        } ⎉ {
            default
        }
    }
}
```

### Type Constraints

- `T` must be `Copy` (returns by value, not reference)
- For non-Copy types, use the existing `get()` → `Option<&T>` pattern

---

## Usage Examples

```sigil
// Audio levels with silent default
≔ level = state.levels.get_or(channel_idx, (-60.0, -60.0));

// Configuration with fallback
≔ timeout = config.timeouts.get_or(service_idx, 30_000);

// Chained with other operations
≔ normalized = samples.get_or(i, 0.0) / max_amplitude;

// In loops with bounds uncertainty
∀ i ∈ 0..requested_count {
    process(data.get_or(i, default_value));
}
```

---

## Alternatives Considered

### 1. Add `copied()` to Option

```sigil
self.levels.get(index).copied().unwrap_or(default)
```

**Rejected because:**
- Requires three method calls for a simple operation
- Copies Rust's API verbatim rather than improving ergonomics
- Still requires understanding `Option` method chaining

### 2. Make `get()` return `T` for Copy types

```sigil
self.levels.get(index).unwrap_or(default)
```

**Rejected because:**
- Changes semantics of existing `get()` method
- Inconsistent behavior based on `T` being Copy or not
- Breaking change for existing code

### 3. Index operator with default syntax

```sigil
self.levels[index ?? default]
```

**Rejected because:**
- Novel syntax requires grammar changes
- `??` is null-coalescing in other languages (semantic mismatch)
- Less discoverable than a named method

---

## Implementation

### Standard Library Addition

Add to `stdlib.rs` for `Vec<T>`, `Array<T, N>`, and `Slice<T>`:

```rust
// In stdlib Vec implementation
fn get_or(&self, index: usize, default: T) -> T
where T: Copy
{
    if index < self.len() {
        self.data[index]
    } else {
        default
    }
}
```

### Parser/Type Checker

No changes required—this is a pure stdlib addition.

---

## Compatibility

- **Backward compatible**: New method, no existing code affected
- **No grammar changes**: Standard method syntax
- **Incremental adoption**: Existing `get()` + conditionals continue to work

---

## Related Work

| Language | Pattern |
|----------|---------|
| Rust | `.get().copied().unwrap_or()` |
| Python | `list[i] if i < len(list) else default` |
| Kotlin | `list.getOrElse(i) { default }` |
| Swift | `array[safe: i] ?? default` (extension) |

Kotlin's `getOrElse` is closest to this proposal, though Sigil's version uses a value rather than a closure for the default.

---

## Future Extensions

### Lazy Default (`get_or_else`)

For expensive default computations:

```sigil
// Lazy evaluation - closure only called if needed
≔ value = cache.get_or_else(key, || compute_expensive_default());
```

This could be added later without affecting `get_or()`.

---

## Open Questions

1. **Naming**: `get_or` vs `get_default` vs `at_or`?
2. **Non-Copy types**: Should there be a `get_or_clone()` variant?
3. **HashMap/BTreeMap**: Should associative containers get similar treatment?

---

## References

- [Orpheus UI TDD Session](../../orpheus/orpheus-desktop/crates/orpheus-ui/TDD-ROADMAP.md) - Discovery context
- [Kotlin getOrElse](https://kotlinlang.org/api/latest/jvm/stdlib/kotlin.collections/get-or-else.html)
- [SIGIL-001: Ergonomic Refinements](./SIGIL-001-ERGONOMIC-REFINEMENTS.md) - Related ergonomic improvements
