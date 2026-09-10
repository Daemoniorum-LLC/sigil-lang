# Sigil arrays are reference-like, and one write pretended otherwise

**Recorded:** 2026-09-10, closing #115.

`arr[i] = v` inside a function was **silently discarded** when the function
returned. The callee saw its own write; the caller did not.

## The table

| mutation inside a function | propagated before #115 |
|---|---|
| `arr[0] = 42` | **no** |
| `arr[0].field = 42` | yes |
| `arr[0][0] = 42` | yes |
| `push(arr, v)` | yes |
| `s.items[0] = 42` (array through a struct field) | yes |

One row of five. That is what made it dangerous: three neighbouring mutations
work, so nobody suspects the fourth, and there is no error to suspect it *by*.

## What it cost

It shaped a codebase before anyone identified it. morgoth carried the comment

```sigil
// Swap focused pane with previous; inline — array slot swap only
// works in same scope as the panes declaration (Phase 26)
```

Pane swapping stayed inline in a ~1000-line `main()` because extracting it into
a `rite` silently stopped working. A compiler bug produced a monofile, and the
diagnosis stopped at the symptom.

## The cause

`eval_assign`'s `Expr::Index` arm had a special case for a **single-segment
path** base — `arr[i] = v`, and nothing else:

```rust
if let Value::Array(arr) = current {
    let borrowed = arr.borrow();
    let mut new_arr = borrowed.clone();          // clone the whole vector
    drop(borrowed);
    if idx < new_arr.len() {
        new_arr[idx] = val.clone();
        self.environment
            .borrow_mut()
            .set(name, Value::Array(Rc::new(RefCell::new(new_arr))))?;  // rebind to a NEW Rc
        return Ok(val);
    }
}
```

It cloned the vector, wrote the element, and repointed the *local binding* at a
new `Rc`. The callee then read its own write through the repointed name; the
caller still held the original array.

Every other row falls through to the generic path below it, which does
`arr.borrow_mut(); borrowed[idx] = val;` — a write through the shared `Rc`.
`s.items[0]` and `arr[0][0]` never took the special case because their base is
a field or an index, not a single-segment path. That asymmetry is what located
the bug: both spellings end at "set element 0 of an array", so the difference
had to be in how the base was reached.

The fix is to write through the `Rc` the binding already holds.

## Why this is not a semantics change

Sigil arrays were **already** reference-like, and provably so before the fix:

```sigil
≔ mut a = [1];  ≔ mut b = a;
push(b, 99);      // len(a) == 2   -- already aliased
b[0].field = 42;  // a[0].field    -- already aliased
b[0][0] = 42;     // a[0][0]       -- already aliased
b[0] = 42;        // a[0] == 1     -- the odd one out
```

The broken path was not value semantics either. A copy would have given the
caller an independent array; this rebound one binding and left every *other*
alias pointing at the original. It was neither aliasing nor copying — it was a
lost write.

**One visible consequence:** the last line above now leaves `a[0] == 42`. That
is the outlier joining the other three, not aliasing being introduced. It is
pinned by `an_array_binding_aliases_rather_than_copies` so a reversal fails
loudly.

## If you want value semantics instead

That is a language decision, not this fix. It would mean changing `push`,
`arr[i].f = v`, `arr[i][j] = v` and array-through-a-struct-field as well, and
deciding what `≔ b = a` means. Silently dropping one write is the only outcome
that cannot be right, which is what #115 says.

## Scope

Interpreter only. The Cranelift JIT rejects a program with an untyped array
parameter outright (`Undefined variable: arr`), before and after, so it never
reached this shape.
