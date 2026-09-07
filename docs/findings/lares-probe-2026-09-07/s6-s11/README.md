# S6–S11 — defects behind S5

Found after the S5 fix let 8 more Qliphoth files reach (or get past) parsing.
Compiler: `cargo build --release --no-default-features --features jit,native`.

Run each with `sigil check <file>`.

| File | Defect | Expected today |
|---|---|---|
| `S6_pub_tome.sigil` | `☉ tome <name>;` public module decl | `expected item, found Crate` |
| `S7_super_path.sigil` | `super·` in an `invoke` path | `expected ';' or new item, found AspectPerfective` |
| `S8_multihash_rawstring.sigil` | `r##"…"##` raw strings | `expected LBracket, found Hash` |
| `S9_structlit_call_arg.sigil` | qualified struct literal as a call argument | `expected RParen, found LBrace` |
| `S10_ne_in_condition.sigil` | `≠` inside `⎇` | `expected LBrace, found IntLit` |

Each should print `no errors` once fixed.

## Contrast cases — these already pass, and must keep passing

```sigil
≔ b = a ≠ 2;                              // S10: ≠ is fine in an assignment
⎇ a != 2 { }                              // S10: ASCII != is fine in a condition
≔ o = crate·router·Opts { replace: true }; // S9: fine in an assignment
r#"hello"#                                 // S8: single-hash raw strings are fine
invoke qliphoth·prelude·*;                 // glob imports are fine
```

The contrast is the point in each case: the feature works in one position and not
another, so a fix should be scoped to the failing context rather than the construct.

## S11 — not reproduced standalone

`packages/qliphoth-sys/src/storage.sigil`:

```sigil
☉ rite len() -> usize! {
    local_storage()
        ·map(|s| s·len())
        ·unwrap_or(0)
}
```

→ `[E0003] evidence mismatch in return type of 'len': expected known (!), found uncertain (?)`

`unwrap_or` takes an uncertain value and a default and yields a definite one, so `usize!`
looks right and the checker looks like it is not narrowing `?` → `!` through it. But that
is a reading of intent, not a demonstration — **this one needs a Sigil owner's judgment**
on whether the checker or the annotation is wrong. `route.sigil` and `guards.sigil` fail
in the same family.
