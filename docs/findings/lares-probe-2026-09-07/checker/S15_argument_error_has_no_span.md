# S15 — E0003 argument errors carry no usable span

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
