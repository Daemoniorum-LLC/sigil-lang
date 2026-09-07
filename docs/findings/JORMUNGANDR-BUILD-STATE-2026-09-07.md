# Jormungandr build state — 2026-09-07

**Question asked:** the Rust compiler at `parser/` had five parser/lexer defects fixed
(S5, S7, S9, S10). Does the self-hosted compiler share them, and can it be fixed too?

**Answer:** it shares two of them, is *more* correct than the Rust parser on a third — and
it **cannot currently be built**, so no fix to it can be verified.

---

## 1. Which defects jormungandr shares

| Defect | Rust parser | jormungandr (`src/*.sg`) |
|---|---|---|
| **S5** — `·` paths rejected in type position | had it | **Does not have it.** `parser.sg:1789,1801,1816` consume `MiddleDot` unconditionally, with no first-segment-case heuristic. jormungandr was *more correct here than the Rust parser was.* |
| **S7** — aspect markers eat path segments | had it | **Has it, with a wider surface.** `lexer.sg:1349 lex_middle_dot` matches `ing`, `ed`, `able`, `ive` with no boundary check, so `·editor`, `·ingest`, `·ability`, `·ivory` all mis-lex. The Rust lexer only had two such markers. |
| **S10** — `≠ ≤ ≥` absent | had it | **Has it.** `lex_bang` accepts `!=` only; no `≠`, `≤`, `≥` anywhere in `lexer.sg` / `lexer_operator.sg`. |
| **S12** — unknown characters | silently dropped | **Different failure.** `lexer.sg:1341` turns an unknown character into `Token·Ident(c)`. Still wrong, but it becomes a visible identifier rather than vanishing. |

So a fix is genuinely needed for S7 and S10; S5 needs nothing.

## 2. Why it cannot be built

### The documented build does not exist

`CLAUDE.md` says:

```sh
cd jormungandr/build
gcc -g -O0 -o sigil2 sigil2.c -lm
```

There is no `sigil2.c` anywhere in the repository. `jormungandr/build/` holds **eight**
generated C files whose names read as successive repair attempts:
`bootstrap_fixed.c`, `_fixed2`, `_fixed3`, `_fixed4`, `fixed_bootstrap.c`,
`fixed_bootstrap_v2.c`, `new_bootstrap.c`, `from_sigil2.c`.

### Seven of the eight do not compile

`gcc -fsyntax-only` error counts:

| File | Errors |
|---|---|
| `bootstrap_fixed4.c` | **0** |
| `bootstrap_fixed3.c` | 1 |
| `bootstrap_fixed.c` | 5 |
| `fixed_bootstrap_v2.c` | 6 |
| `fixed_bootstrap.c` | 7 |
| `bootstrap_fixed2.c`, `from_sigil2.c`, `new_bootstrap.c` | 8 each |

Typical errors are type faults in the **generated** code:

```
error: invalid initializer
    SigilValue name = sigil_qualify_name(ctx, sigil_String____as_str(_t1));
error: incompatible types when assigning to type 'SigilValue' from type 'int'
```

### The one that compiles cannot link

`bootstrap_fixed4.c` compiles cleanly and then fails at link with **46 distinct undefined
symbols** — Sigil stdlib methods the code generator emitted calls to but never emitted
bodies for:

```
undefined reference to `sigil_Result____ok'
undefined reference to `sigil_String____new'
undefined reference to `sigil_String____as_str'
undefined reference to `sigil_String____chars'
```

This is not a missing build flag. `parser/runtime/` builds cleanly via its Makefile
(`make` → `libsigil_runtime.a`), but that archive exports only C-level primitives —
`sigil_alloc`, `sigil_abs`, `print`, `println` — and **none** of the 46. Neither does the
prebuilt `libsigil_runtime.a` shipped in `qliphoth/apps/wraith/runtime/`.

So the blocker is a **codegen gap**: jormungandr emits calls to stdlib methods that
nothing defines.

### The Rust compiler cannot read jormungandr's sources either

`sigil check` over `jormungandr/src/*.sg`: **13 of 28 pass, 15 fail** — unchanged by the
S5/S7/S9/S10 fixes, so those are yet another distinct set of defects
(`expected identifier, found Trait`, among others).

jormungandr is therefore blocked from both directions: its own bootstrap will not link,
and the canonical compiler cannot process its source.

## 3. Incidental finding

`parser/runtime/runtime` is a symlink to `/home/crook/dev/sigil-lang/parser/runtime` — an
absolute path to a specific developer's machine, committed to the repository. It dangles
in every other checkout.

## 4. What this means for fixing S7 and S10 in jormungandr

The source changes are small and well-understood:

- `lexer.sg` `lex_middle_dot` — require a non-identifier boundary after `ing`/`ed`/`able`/`ive`,
  the same fix applied to the Rust lexer.
- `lexer.sg` / `lexer_operator.sg` — add `≠`, `≤`, `≥` beside `!=`, `<=`, `>=`.

They can be written today. They **cannot be compiled or tested**, because building
jormungandr requires a working jormungandr (or a Rust compiler that can read its source,
which currently cannot). Any such patch ships unverified.

Recommended order:

1. **Make jormungandr buildable.** Close the 46-symbol codegen gap, or regenerate the
   bootstrap C from source. This is a project, not a patch, and everything else depends
   on it.
2. Fix `CLAUDE.md`'s build instructions, which name a file that does not exist.
3. Only then port S7 and S10, where they can be verified.

The unbuildability is a larger finding than any individual parser defect: a self-hosted
compiler that cannot be built is not currently self-hosting.
