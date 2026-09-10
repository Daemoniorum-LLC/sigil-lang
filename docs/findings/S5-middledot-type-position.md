# S5 — parser rejects `·`-qualified paths in type positions

**Found:** 2026-09-07, probing Qliphoth for the Lares client.
**Impact:** 16 of 39 Qliphoth source files fail `sigil check`, including all of
`qliphoth-router` and `qliphoth-sys`'s `storage`, `timers`, `websocket`, `closure`,
`history`.

## Reproduction

Compiler built with `cargo build --release --no-default-features --features jit,native`.

**Works** — middledot path in *expression* position:

```sigil
rite main() {
    ≔ m = std·collections·HashMap·new();
}
```
```
✓ no errors
```

**Fails** — the same path shape in a *parameter type*:

```sigil
rite f(x: &std·collections·HashMap<String, String>) -> i32 { 0 }
```
```
[E0002] Unexpected token: expected RParen, found MiddleDot at 14..16
```

**Fails** — in a *type argument*:

```sigil
☉ Σ S { ☉ state: Option[serde_json·Value]? }
```
```
[E0002] Unexpected token: expected RBracket, found MiddleDot at 39..41
```

**Fails** — in a *generic bound*:

```sigil
☉ rite g[T: serde·de·DeserializeOwned](k: &str!) -> i32 { 0 }
```
```
[E0002] Unexpected token: expected RBracket, found MiddleDot at 19..21
```

## Reading

The expression parser handles `·` paths; the **type** parser does not. Every failure is a
type position — parameter type, type argument, generic bound — and the error always names
the closing delimiter the parser wanted instead.

## Real instances in Qliphoth

| File | Line | Source |
|---|---|---|
| `packages/qliphoth-sys/src/storage.sigil` | 32 | `☉ rite get_json[T: serde·de·DeserializeOwned](...)` |
| `packages/qliphoth-router/src/router.sigil` | 40 | `☉ state: Option[serde_json·Value]?` |
| `src/core/mod.sigil` | 136 | `rite render_attrs(attrs: &std·collections·HashMap<String, String>)` |

## Why it matters beyond Qliphoth

Sigil's own docs present `·` as the native path separator, so this affects any code using
qualified types in signatures — which is most non-trivial code. That 23 files pass is
because they happen to keep qualified paths out of type positions.

It may also explain why `apps/wraith`'s current source sits beside a `src.old/`.

## Suggested fix

Teach the type parser the same qualified-path production the expression parser already
has. Likely small and localized.
