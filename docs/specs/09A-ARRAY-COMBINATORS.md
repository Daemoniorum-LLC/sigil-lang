# Array Collection Combinators Specification

**Version:** 1.0.0
**Status:** Draft
**Date:** 2026-02-21
**Parent Spec:** [09-STDLIB.md](./09-STDLIB.md), [01A-MORPHEME-DESUGARING.md](./01A-MORPHEME-DESUGARING.md)

---

## 1. Conceptual Foundation

Sigil provides three complementary interfaces for transforming arrays and collections:

1. **Morpheme pipeline** — The Sigil-native form. Pipe morphemes (`|τ`, `|φ`, `|σ`, `|ρ`, `|π`)
   express dense, composable transformations with evidence-preserving semantics.
   See [01A-MORPHEME-DESUGARING.md](./01A-MORPHEME-DESUGARING.md) for full morpheme semantics.

2. **Method dispatch** — Object-oriented form via middledot. `arr·map(fn)`, `arr·filter(fn)`,
   `arr·fold(init, fn)`. Desugars to morpheme operations.

3. **Stdlib functions** — Free-function form for use as pipe arguments or standalone calls.
   `map(arr, fn)`, `filter(arr, fn)`, `fold(arr, init, fn)`.

These three interfaces are semantically equivalent. The morpheme form is preferred in
idiomatic Sigil; the function form is used when composing with pipes (`|>`).

### 1.1 Scope

This spec covers:
- **Section 2**: What currently exists (audit-verified)
- **Section 3**: Gaps to be filled (this implementation)
- **Section 4**: Behavioral contracts for all combinators
- **Section 5**: Invariants and properties
- **Section 6**: Integration with morpheme evidence system

---

## 2. Type Architecture

### 2.1 Core Types

```
Array   — dynamic sequence of heterogeneous values
Fn      — callable: closure or named function
Any     — value of unknown type (used for nullability)
Bool    — true or false
Int     — integer value
```

### 2.2 Combinator Signatures

```
// Transformation
map(arr: Array, fn: Fn(Any) → Any) → Array
flat_map(arr: Array, fn: Fn(Any) → Array) → Array     [GAP — §3.1]

// Selection
filter(arr: Array, pred: Fn(Any) → Bool) → Array
find(arr: Array, pred: Fn(Any) → Bool) → Any | null
find_index(arr: Array, pred: Fn(Any) → Bool) → Int | null   [GAP — §3.2]

// Quantifiers
any_where(arr: Array, pred: Fn(Any) → Bool) → Bool    [GAP — §3.3]
all_where(arr: Array, pred: Fn(Any) → Bool) → Bool    [GAP — §3.3]
none_where(arr: Array, pred: Fn(Any) → Bool) → Bool   [GAP — §3.3]

// Counting
count_where(arr: Array, pred: Fn(Any) → Bool) → Int   [GAP — §3.4]

// Reduction
fold(arr: Array, init: Any, fn: Fn(Any, Any) → Any) → Any
reduce(arr: Array, fn: Fn(Any, Any) → Any) → Any | null

// Access
first(arr: Array) → Any | null
last(arr: Array) → Any | null

// Ordering
sort(arr: Array) → Array
sort_by(arr: Array, fn: Fn(Any, Any) → Int) → Array

// Structural
flatten(arr: Array) → Array
zip(arr1: Array, arr2: Array) → Array
enumerate(arr: Array) → Array    // [(0, val0), (1, val1), ...]
take(arr: Array, n: Int) → Array
skip(arr: Array, n: Int) → Array
take_while(arr: Array, pred: Fn(Any) → Bool) → Array
drop_while(arr: Array, pred: Fn(Any) → Bool) → Array
unique(arr: Array) → Array
reverse(arr: Array) → Array
```

### 2.3 Existing vs. Gap Summary

| Combinator | Status | Interface |
|------------|--------|-----------|
| `map` | ✅ exists | stdlib fn + method + `\|τ` morpheme |
| `filter` | ✅ exists | stdlib fn + method + `\|φ` morpheme |
| `find` | ✅ exists | method |
| `fold` | ✅ exists | stdlib fn + method + `\|ρ` morpheme |
| `reduce` | ✅ exists | method + `\|ρ` morpheme |
| `flatten` | ✅ exists | stdlib fn + `\|π·flat` morpheme |
| `zip` | ✅ exists | stdlib fn |
| `enumerate` | ✅ exists | stdlib fn |
| `take_while` | ✅ exists | stdlib fn |
| `drop_while` | ✅ exists | stdlib fn |
| `sort` | ✅ exists | stdlib fn + `\|σ` morpheme |
| `sort_by` | ✅ exists | stdlib fn + `\|σ(fn)` morpheme |
| `unique` | ✅ exists | stdlib fn |
| `reverse` | ✅ exists | stdlib fn |
| `first` / `last` | ✅ exists | stdlib fn |
| `take` / `skip` | ✅ exists | stdlib fn |
| `any(arr)` | ✅ exists | truthy check only — no predicate |
| `all(arr)` | ✅ exists | truthy check only — no predicate |
| `none(arr)` | ✅ exists | truthy check only — no predicate |
| `count(arr)` | ✅ exists | no predicate version only |
| `flat_map` | ❌ gap | see §3.1 |
| `find_index` | ❌ gap | see §3.2 |
| `any_where` | ❌ gap | see §3.3 |
| `all_where` | ❌ gap | see §3.3 |
| `none_where` | ❌ gap | see §3.3 |
| `count_where` | ❌ gap | see §3.4 |

---

## 3. Gap Specifications

### 3.1 flat_map

`flat_map(arr, fn)` applies `fn` to each element — where `fn` returns an Array — then
flattens the results into a single Array. Equivalent to `map` followed by `flatten`.

```
flat_map(arr, fn):
    result ← []
    ∀ item ∈ arr:
        inner ← fn(item)
        ∀ x ∈ inner:
            push(result, x)
    return result
```

**Morpheme equivalent:** `arr|τ{fn(it)}|π·flat`

**Examples:**
```
flat_map([[1,2],[3,4]], |x| x)         → [1, 2, 3, 4]
flat_map(["hi","bye"], |s| chars(s))   → ["h","i","b","y","e"]
flat_map([], |x| [x, x])              → []
```

**Edge cases:**
- If `fn` returns a non-array, the result is treated as a single-element array
- Empty arrays in output are silently omitted (flatten behavior)

### 3.2 find_index

`find_index(arr, pred)` returns the integer index of the first element where `pred`
returns truthy, or `null` if no element matches.

```
find_index(arr, pred):
    ∀ i, item ∈ enumerate(arr):
        if pred(item): return i
    return null
```

**Examples:**
```
find_index([10, 20, 30], |x| x > 15)   → 1
find_index([1, 2, 3], |x| x > 10)      → null
find_index([], |x| true)               → null
```

### 3.3 Predicate Quantifiers: any_where, all_where, none_where

The existing `any(arr)`, `all(arr)`, `none(arr)` check for truthy values without a
predicate. The `*_where` variants accept a predicate closure.

```
any_where(arr, pred):
    ∀ item ∈ arr:
        if pred(item): return true
    return false

all_where(arr, pred):
    ∀ item ∈ arr:
        if !pred(item): return false
    return true

none_where(arr, pred):
    return !any_where(arr, pred)
```

**Examples:**
```
any_where([1, 2, 3], |x| x > 2)        → true
any_where([1, 2, 3], |x| x > 10)       → false
all_where([2, 4, 6], |x| x % 2 == 0)   → true
all_where([2, 3, 6], |x| x % 2 == 0)   → false
none_where([1, 3, 5], |x| x % 2 == 0)  → true
none_where([1, 2, 5], |x| x % 2 == 0)  → false
```

**Empty array behavior:**
- `any_where([], pred)` → `false` (vacuously — no element satisfies)
- `all_where([], pred)` → `true` (vacuously — no element violates)
- `none_where([], pred)` → `true` (vacuously — no element satisfies)

### 3.4 count_where

`count_where(arr, pred)` returns the number of elements where `pred` returns truthy.

```
count_where(arr, pred):
    return len(filter(arr, pred))
```

**Examples:**
```
count_where([1, 2, 3, 4], |x| x % 2 == 0)   → 2
count_where(["a", "bb", "ccc"], |s| len(s) > 1) → 2
count_where([], |x| true)                    → 0
```

---

## 4. Behavioral Contracts

### 4.1 Transformation Contracts

**map:**
- Length preserved: `len(map(arr, fn)) == len(arr)`
- Order preserved: element at index i in result = fn(arr[i])
- Empty input → empty output

**flat_map:**
- Length: `len(flat_map(arr, fn)) >= 0` (may be less than input if fn returns empty)
- Order: inner elements maintain order; outer order maintained
- `flat_map(arr, |x| [x]) == arr` (identity when fn wraps in singleton)

**filter:**
- Subset: every element in result is in arr
- Order preserved: relative order of matching elements unchanged
- `len(filter(arr, fn)) <= len(arr)`

### 4.2 Predicate Quantifier Contracts

- `any_where(arr, pred) == (count_where(arr, pred) > 0)`
- `all_where(arr, pred) == (count_where(arr, pred) == len(arr))`
- `none_where(arr, pred) == !any_where(arr, pred)`
- `all_where(arr, pred) == none_where(arr, |x| !pred(x))`

### 4.3 Composition Contracts

- `map(map(arr, f), g) == map(arr, |x| g(f(x)))` (functor composition)
- `filter(filter(arr, p), q) == filter(arr, |x| p(x) ∧ q(x))` (filter fusion)
- `flat_map(arr, fn) == flatten(map(arr, fn))`
- `find_index(arr, pred) == first(filter(enumerate(arr), |[i,x]| pred(x)))[0]`
- `count_where(arr, pred) == len(filter(arr, pred))`

---

## 5. Constraints and Invariants

**P1:** All combinators return a new array without modifying the input.

**P2:** Predicates are called exactly once per element, in order.

**P3:** Combinators on empty arrays return empty arrays (or null/false/0 for
quantifiers/finders/counters as specified in §3).

**P4:** `flat_map`, `filter`, `map` never produce null — always return Array.

**P5:** `find` and `find_index` return null (not error) when no element matches.

**P6:** Predicate closures may produce side effects — execution order is guaranteed
left-to-right, matching array index order.

---

## 6. Error Conditions

| Condition | Behavior |
|-----------|----------|
| `fn` argument is not callable | Runtime error: "expected function" |
| `fn` called on each element; element causes error | Error propagates up |
| Array argument is null | Runtime error: "expected array, got null" |
| Non-array argument | Runtime error: "expected array, got TYPE" |

No combinator silently swallows errors from user-provided functions.

---

## 7. Integration Points

### 7.1 With Morpheme Pipeline

All gap functions are expressible in terms of existing morphemes:

```
flat_map(arr, fn)          ≡  arr|τ{fn(it)}|π·flat
find_index(arr, pred)      ≡  first(arr|φ{pred}|enumerate|τ{it[0]})  *approx
any_where(arr, pred)       ≡  len(arr|φ{pred}) > 0
all_where(arr, pred)       ≡  len(arr|φ{!pred(it)}) == 0
none_where(arr, pred)      ≡  len(arr|φ{pred}) == 0
count_where(arr, pred)     ≡  len(arr|φ{pred})
```

The stdlib functions exist for ergonomics and to eliminate temp allocations in
common patterns (`any_where` does not need to build the filtered array).

### 7.2 With Ecosystem Libraries

These combinators are the Sigil-idiomatic replacements for Rust patterns found
in the daemon/commune/engram libs:

| Rust pattern | Sigil equivalent |
|---|---|
| `.iter().any(\|x\| pred)` | `any_where(arr, \|x\| pred)` |
| `.iter().all(\|x\| pred)` | `all_where(arr, \|x\| pred)` |
| `.iter().filter(p).count()` | `count_where(arr, p)` |
| `.iter().flat_map(f).collect()` | `flat_map(arr, f)` |
| `.iter().position(p)` | `find_index(arr, p)` |

---

## 8. Open Questions

1. **Naming:** `any_where`/`all_where` vs `any_with`/`all_with` vs overloading `any`/`all`?
   Current choice: `any_where` etc. to avoid breaking existing truthy-check uses of `any`.
   Revisit when predicate overloading is considered.

2. **flat_map with non-array fn return:** If fn returns a scalar, wrap in singleton or error?
   Current choice: wrap in singleton (lenient). May reconsider for strictness.

3. **find_index exact semantics with enumerate:** The morpheme approximation is not exact
   when the array contains arrays as elements — enumerate returns `[index, value]` pairs
   which conflict with nested arrays. The stdlib implementation uses direct index tracking.

---

## Revision History

| Version | Date | Changes |
|---------|------|---------|
| 1.0.0 | 2026-02-21 | Initial draft. Grounded in audit of stdlib.rs and morpheme spec. Identifies 6 gaps: flat_map, find_index, any_where, all_where, none_where, count_where. |
