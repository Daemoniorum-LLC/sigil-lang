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

## S12 — `migrate` renames keywords the parser already accepts

Found while merging `develop` into this branch, not while running the probe, so it
sits in this directory despite its name.

    $S parse S12_keyword_field_names.sigil            # no errors

`sigil migrate`'s keyword-identifier pass (19a0f8c) renames every Sigil keyword it
finds in a binding position, on the stated premise that "a keyword can never
legitimately be an identifier". The premise does not hold. In field position the
parser accepts most keywords:

| Field name | `Σ P { <name>: Int }` |
|---|---|
| `anima`, `state`, `body`, `layer`, `scope` | parses |
| `aspect` | `expected identifier, found Trait` |
| `alter` | `expected identifier, found Alter` |

So the pass is right for `aspect` and `alter` and wrong for the other five, and the
`_` suffix it appends to those is churn on source that was already valid.

The visible damage is `CONCLAVE.sigil`, whose header reads "APPEND-ONLY: Never
modify existing entries." Running migrate over the repository rewrote `anima` to
`anima_` in nine existing session entries — the only difference between this
branch's copy and develop's. The merge restores the original nine, since the rename
buys no parseability. (Neither copy parses as a whole: the file's top-level
`acolyte : Reflecting { … }` form is not an item the grammar knows, which is a
separate question about what CONCLAVE.sigil is meant to be.)

Expected once fixed: the pass asks whether the parser actually rejects the keyword
in the position it was found, rather than renaming every keyword unconditionally.
