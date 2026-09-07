# S16 — if-let does not narrow an Option binding ✅ FIXED

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

## Root cause — narrower than the title suggested

`?x` in `⎇ ≔ ?x = e` does not parse as a Some-pattern at all. It is
`Pattern::Ident { evidentiality: Some(Uncertain) }` — an **evidence-annotated identifier**.
`bind_pattern` then binds the name to the whole scrutinee type, so `x : Option<T>`.

The narrowing was therefore never happening — not for calls, and not for fields either.
The "passing" field case that made this look call-specific was passing by accident: the
field's type had not been resolved, so the binding was a fresh var that unified with
anything. Fixing S14 resolved those types, which is what made the real bug visible.

## Fix

`iflet_binding_type()` in `typeck.rs`: when the pattern is an `Uncertain`-marked identifier
and the resolved scrutinee is `Option<T>` or `Result<T, E>`, the binding takes `T`. Any
other pattern, or a non-Option/Result scrutinee, is passed through untouched — so this
cannot affect bindings it does not understand.

Applied only on the if-let path, not in `bind_pattern` itself, so plain `let` and match-arm
binding are unchanged.

**Verified:** `runtime.sg` now checks clean; jormungandr/src reaches 27/28. Full suite
716 pass / 11 fail, failure set byte-identical to baseline.
