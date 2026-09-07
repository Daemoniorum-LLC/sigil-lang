# S16 — if-let does not narrow a *computed* Option

An `⎇ ≔ ?x = <expr>` binding narrows `Option<T>` to `T` when the scrutinee is a **declared**
field, but not when it is the **result of a call**.

## Passes

```sigil
☉ Σ I { m: !Option<u32> }
⊢ I {
    ☉ rite f(self) -> !u32 {
        ⎇ ≔ ?id = self.m { ⤺ id; }   // id : u32 ✅
        ↩ 0;
    }
}
```

## Fails

```sigil
rite mk() -> !Option<u32> { ↩ null; }
rite f() -> !u32 {
    ⎇ ≔ ?x = mk() { ↩ x; }           // x : Option<u32> ❌
    ↩ 0;
}
```
```
[E0003] type mismatch in return: expected U32!, found Option<U32>
```

Same for a method call: `m.get("k")` where `m: !Map<String, u32>` now correctly yields
`Option<U32>` (see S14) and `?x` still binds the whole `Option<U32>` rather than `U32`.

## Reading

The payload type is right; the **binding** is not narrowed. It looks like the pattern is
typed before the scrutinee's type is resolved, so a declared type works and an inferred one
does not.

## Impact

This is the last thing blocking `jormungandr/src/runtime.sg`, and through it the
regeneration of the bootstrap. `StringInterner::intern` is exactly this shape:

```sigil
⎇ ≔ ?id = self.strings.get(s) { ⤺ id; }
```

## Not attempted

Fixing it means changing when if-let patterns are unified against their scrutinee — a
change to inference ordering, with a much wider blast radius than the localised fixes made
so far. It wants a Sigil owner rather than a drive-by patch.
