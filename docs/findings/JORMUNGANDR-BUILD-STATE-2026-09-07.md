# Jormungandr build state — 2026-09-07

**Question asked:** the Rust compiler at `parser/` had five parser/lexer defects fixed
(S5, S7, S9, S10). Does the self-hosted compiler share them, and can it be fixed too?

**Answer:** it shares two of them, is *more* correct than the Rust parser on a third, and
it could not be built at all — **now fixed. It builds and runs.** See §5.

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


---

## 5. RESOLVED — it builds

`jormungandr/build/build.sh` produces a working 3.8 MB compiler binary.

### What the blocker actually was

Not a missing library, and not the 46 symbols being unimplemented. `bootstrap_fixed4.c`
contains a builtin-runtime block at its tail, but that block came from an **older codegen
and was ~245 lines short** — it omitted 45 of the 46 stdlib definitions that the rest of
the file declares and calls. So the bootstrap could compile and could never link.

The definitions were recoverable from the file itself: the compiled body of
`CodeGen::emit_builtin_impls` is 1061 straight-line `line()` calls with **zero control
flow**, so extracting its string literals in order reproduces exactly what a running
jormungandr would emit.

`bootstrap_completion.c` (14 KB) is that recovery. It supplies:

| | |
|---|---|
| 45 stdlib definitions | extracted verbatim from `emit_builtin_impls` |
| `sigil_any` | **written by hand** — codegen emits a declaration and call sites for `.any(pred)` but never a definition, so nothing ever provided it. Written as the exact dual of `sigil_all`, matching its closure convention. |
| `TAG_STRINGBUILDER 17`, `TAG_MAP 18` | values verbatim from `codegen.sg`; the older prelude predates them |
| `StringBuilder`, `SigilStringBuilder` and helpers | two distinct builder types the newer builtins need |

### A second corruption instance

`sb_new`, `sb_ensure_cap`, `sb_to_string` had to be written by hand rather than extracted,
because `codegen.sg` now emits them as:

```
typedef Σ StringBuilder {     ⤺ sb;     ⎇ (sb->len + needed >= sb->cap) {
```

`Σ`, `⤺`, `⎇` are Sigil's `struct`, `return`, `if` — emitted into **C**. This is the same
keyword-migration corruption of string literals recorded in §3, and it means the current
`codegen.sg` cannot generate a working compiler even once it can run. Counted in
`emit_builtin_impls` alone: **249 `⤺`, 165 `⎇`, 13 `⎉`, 10 `⟳`**, plus `Σ` in the prelude
emitter. That is the next thing to fix.

### Verified

```
$ ./build.sh
==> built ./jormungandr
$ ./jormungandr
no input files specified
$ ./jormungandr compile hello.sg -o hello.c     # emits 91 KB of C
```

### Follow-up: the C templates are fixed too

The keyword corruption described in §3 has been repaired: 552 lines across
`codegen.sg` and `codegen_operation.sg`, with mappings taken from
`parser/src/lexer.rs` rather than guessed (`⤺`→`return`, `∀`→`for`, `Σ`→`struct`,
`λ`→`fn`, `⊗`/`⊲`→`break`, `↻`/`⊳`→`continue`, and so on). `·` was deliberately left
alone — it is legitimate in identifiers like `Lexer·advance`. Verified that code
outside string literals is byte-identical on every touched line.

`typeck.sg`'s diagnostics were corrupted the same way and are now readable English
again (`"⎇ condition must be bool"` → `"if condition must be bool"`).

So a regenerated compiler would now emit valid C. Whether it can *be* regenerated is
the next section.

### Still broken, and worth knowing

The compiler **runs but does not yet compile user code correctly**. Given

```sigil
rite main() { println("hello"); }
```

it exits 0, emits 91 KB of C — and that C contains only the runtime builtins. No
`sigil_main`, no translation of the input, no diagnostic. `fn main()` is correctly
*rejected* with a real `CompileError`, so the parser is not simply ignoring everything;
`rite` and `λ` forms parse and are then silently dropped before codegen.

So this closes the *bootstrap* gap — the thing that made jormungandr unbuildable — and
exposes the next one. That failure has the same shape as `S1`, `S10` and `S12`: exit 0,
plausible output, nothing actually done.

#### Diagnosis of the next gap — two distinct bugs

`dump-tokens` and `dump-ast` narrow it precisely.

**The lexer is fine.** `dump-tokens` on `rite zzdistinctive() { … }` produces correct
spans (`0..4`, `5..18`, …).

**The parser stops after the first item.** `dump-ast` item counts:

| input | items reported |
|---|---|
| empty file | **0** |
| one struct | 1 |
| two structs | **1** |
| three functions | **1** |

Zero for empty and one for everything else, regardless of content. So parsing halts
after the first item rather than looping.

**Lowering separately drops functions.** Even a single-function file emits no
`sigil_main` and no function body, and `codegen.sg`'s generate loop iterates
`module.functions` while `dump-ast` iterates `ast.items` — different collections. So the
one item that does parse is not reaching `module.functions` either.

**Correction — neither is a source bug.** Both live in the stale bootstrap binary, not in
`parser.sg`. The source's item loop is correct (`while !is_eof { items.push(parse_item()) }`)
and its `parse_item` handles `Async`, `At`, `Fn`, `Hash`, `Ident`. The **compiled**
`parse_item` in `bootstrap_fixed4.c` handles only `At` and `Hash`. The binary is materially
older than the source.

So these cannot be fixed by editing source — the bootstrap must be **regenerated**, which
requires the canonical compiler to read jormungandr's source. That is the real critical
path, and §6 is progress on it.

**A second red herring, recorded because it cost time.** The "parser stops after one item"
measurement was taken with test files using `λ name() { }`. The canonical compiler *rejects*
that — `λ` is a closure expression, not a declaration — so those inputs were invalid Sigil
and jormungandr was accepting garbage. The item counts proved nothing about the loop.

**A red herring worth recording.** `dump-ast` labels every item `enum`. That is not a
misparse: it calls `item.node.kind_name()`, which reports the *runtime value tag*, and AST
items are enum values. The label is uninformative by construction, and it briefly looked
like the smoking gun.

**Also note the lexer's keyword set is stale.** It knows `fn` and `λ` but not `rite`, which
the Rust compiler accepts and which the ecosystem's `.sg` files use throughout. `fn` is
rejected too, with a diagnostic that prints `expected <value>, found <value>` — the token
display is broken as well.

Also noted: `CLAUDE.md`'s build line references `../src/main.sg`, which does not exist
either, and `jormungandr/src/` carries `.bak`, `.new` and `.broke` copies of `ast.sg` and
`codegen.sg`.


---

## 6. Progress toward regenerating the bootstrap

Regenerating requires the canonical Rust compiler to read `jormungandr/src`. It read 13 of
28 files.

**11 of the 15 failures had one cause.** jormungandr declares every function and method as
`λ name(...)`. That is not canonical Sigil — the Rust lexer defines `λ` as `LambdaExpr`, a
closure *expression*, and rejects it in item position:

```sigil
⊢ T { ☉ λ get(self) -> i64! { … } }      // found LambdaExpr
⊢ T { ☉ rite get(self) -> i64! { … } }   // no errors
```

659 declarations were migrated across 12 files, matching only `λ` followed by an identifier
and `(` or `<` at line start. The 5 genuine uses are untouched — `λ(A, B) -> C` function
types in doc comments and the generic bound `F: λ(T) -> U`. jormungandr's own lexer knew
`fn` but not `rite`, so `rite` was added there too; otherwise it could not lex its own
migrated source.

**Result: 13/28 → 18/28 checking clean.**

### The remaining 10 are not one more cheap fix

| Kind | Files |
|---|---|
| Evidence / type mismatches (need semantic judgment) | `interp_eval`, `lexer`, `lexer_string`, `runtime` |
| `Trait` used as an identifier / pattern | `ast`, `parser` |
| `use` in item position | `wasm_bridge` |
| `Eq` where `LBrace` expected | `lower`, `typeck` |
| Generics followed by `LParen` | `span` |

Each needs its own investigation. Some may be further dialect drift like the `λ` case;
others (the E0003s) may be genuine type errors in jormungandr.

### Order of work from here

1. Close the remaining 10, so the canonical compiler can read all of `jormungandr/src`.
2. Regenerate `bootstrap_fixed4.c` from current source. That alone fixes the parser and
   lowering failures in §5, since the source is already correct.
3. Rebuild via `build.sh` and re-test — at which point `bootstrap_completion.c` may become
   unnecessary, because a current codegen emits the full builtin block.
