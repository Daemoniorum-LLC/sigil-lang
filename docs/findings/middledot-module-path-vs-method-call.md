# `·` — module path or method call? Where the decision lives

**Recorded:** 2026-09-10, closing #95.
**History:** #61 → #86 → #87 → #95. The same ambiguity has been rediscovered
four times; this note exists so it is not rediscovered a fifth.

## The ambiguity

In expression position `·` is overloaded:

```sigil
tome·lerp(a, b, t)     // path:        module → free function
tag·to_string()        // method call: receiver → method
```

Both are `lowercase · lowercase (`. **No amount of lookahead separates them**,
because the difference is not in the token stream at all — it is what the first
segment is *bound* to. That is a name-resolution question, not a syntactic one.

Every attempt to answer it with a wider parser guard has broken the other half:

* **#61** made `·` a path separator more eagerly, so `tag·to_string()` became a
  call to a path `tag·to_string` dispatched to a built-in of the wrong arity.
* **#86** was the fallout: narrowing it back broke `geom·P·new(…)`, which turned
  into a field access on a variable that was never bound —
  `undefined variable: geom`.

## What the parser decides

`parse_type_path` (parser/src/parser.rs) accepts `·` as a path separator in
expression position only when it can be sure syntactically:

1. **In a type context** — `·` is unambiguous, types have no method calls (S5).
2. **A path-keyword root** — `tome` and `super` are lexer keywords and can never
   be values, so `tome·lerp(…)` is a path with no ambiguity to resolve. This is
   why the `tome·` spelling was never affected by #95.
3. **An uppercase root** — `HashMap·new()`.
4. **#87: the `·ident` chain ahead reaches an uppercase segment** — Sigil types
   are uppercase and methods are not, so `geom·P·new` reaches `P` and is a path,
   while `tag·to_string` reaches nothing and stays a method call.

Rule 4 is a *cover*, not a general solution: it only catches module paths that
happen to name a type on the way.

## What the interpreter decides (#95)

`geom·lerp(…)` reaches nothing uppercase, so the parser leaves it as a method
call on `geom`, and the parser is right to. The rest is settled in
`Interpreter::module_qualified_path` (parser/src/interpreter.rs), on the one
piece of information the parser does not have — the binding:

* The receiver is **bound** → it is a value, and `value·method()` is a method
  call. This is what keeps `tag·to_string()` working, and it is why a local
  variable shadowing a scroll name still resolves to the variable.
* The receiver is **unbound** *and* names a scroll in scope *and* that scroll
  owns the member → retry it as the path it always was.

The retry can only ever replace an error. An unbound receiver had exactly one
possible outcome before — `undefined variable: geom` — so nothing that worked
can be masked by it. The member check matters for the same reason: without it an
unresolvable `geom·nosuchfn(…)` falls into the generic path-call machinery and
comes back as an empty struct named `geom`, which is worse than the error.

## If you are about to widen the parser guard

Don't. `tome·lerp(…)` and `tag·to_string()` are lexically identical; whatever
you add to tell them apart will get one of them wrong. Add the case to
`module_qualified_path` instead, where the answer is knowable.
