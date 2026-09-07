# S15 — E0003 argument errors carry no usable span ✅ FIXED

`sigil check jormungandr/src/parser.sg` reports:

```
[E0003] Error: evidence mismatch in argument 2: expected known (!), found uncertain (?)
   … x8, with no location
```

`--format=json` gives a span, but it is the **enclosing impl block**, not the call site:

```json
"span": { "start": 2673, "end": 187625 },
"line": 97, "end_line": 4754
```

Lines 97–4754 of a 5000-line file. The eight errors are therefore effectively unlocatable:
the offending values have *inferred* uncertainty rather than a written `?`, so they cannot
be found by search either.

Other E0003s in the same run (`return type of 'lex_escape_sequence'`, `let binding 'idx'`)
do carry precise spans and were straightforward to fix. It is specifically the
**argument** case that reports the enclosing item.

**Impact:** this is the only thing now blocking `parser.sg`, and through it the last of
`jormungandr/src`. Fixing the span would likely make the eight errors trivial.


---

## Fixed 2026-09-07

`check_evidence` created its `TypeError` with no span, and a fallback then substituted
`current_item_span` — the enclosing item.

The fix adds `expr_span()` to `typeck.rs`: a best-effort span that walks an expression to
its nearest identifier (`Path`, `Field`, `MethodCall`, `Struct`, recursing through `Call`,
`Unary`, `Index`, `Binary`, and the wrappers `Evidential`, `Try`, `Await`, `Morpheme`,
`Pipe`, `Macro`). `check_evidence` gained an `Option<Span>`; the argument loop now zips the
argument *expressions* alongside their types and passes each one's span.

Returning `None` is safe — the item-span fallback still applies — and the return-type
caller passes `None` deliberately, so its behaviour is unchanged.

**`Evidential` was the variant that mattered.** An argument written `x?` arrives wrapped in
it, which is exactly the shape that triggers this diagnostic. A first version of
`expr_span` without it fixed a synthetic test and still returned `None` for every real
case — worth remembering: the minimal repro passed while the actual bug did not move.

### Before / after

```
before:  line 97 (end 4754)   x8      # the whole impl block
after:   lines 370, 497, 932, 1032, 3117, 4079, 4364, 4530
```

### What it revealed

All eight were one mistake: `ParseError·unexpected(expected, found: !Token, span)` called
with `t?`. `t` is a match binding holding a Token, so `?` is the evidence marker rather
than the try operator, and it contradicts the declared `!Token`. Identical to the
`c => c?` arm found earlier in `lex_escape_sequence`.

Fixed at all eight sites; `parser.sg` now checks clean.

**Verified:** full suite 716 pass / 11 fail, failure set byte-identical to baseline.
