# Why `run-dir` alternated between the right answer and "Invalid struct operation"

**Recorded:** 2026-09-10, closing #104.

Same directory, same binary, ~40% of runs failing with
`[R0000] Invalid struct operation`. Reported against `aether`'s
`engine/math/src/{lib,vector}.sg`. It nearly produced a false A/B result: a fix
looked like a regression when the two runs simply landed differently.

Two defects compose. Neither is sufficient alone, which is why a minimal
two-file case did not reproduce it and the real one did.

## 1. The trigger: `Self` is not bound through a module-qualified call

`eval_call` decides what `Self` means for the callee from the call path
(`interpreter.rs`, `type_name_for_self`). It tested only the **first** segment:

```rust
let first = &path.segments[0].ident.name;
if self.types.contains_key(first) { Some(first.clone()) } else { None }
```

That is right for `P·new(…)` and wrong for `geom·P·new(…)`, where segment 0 is
the scroll. `Self` stayed unbound, and a `Self { .. }` literal in the callee
falls back to its own raw name — so the constructor returned a struct named,
literally, **`"Self"`**.

Nothing is filed under `"Self"`. `lookup_operator_impl` keys operator impls by
the value's struct name, so the next `+` on that value found no `impl Add` and
reported `Invalid struct operation`.

This half is deterministic: with two modules it fails every time.

## 2. The nondeterminism: recovering a `Self`-named value walked a `HashMap`

A `Self`-named value is not a dead end — a method call on one is recovered by
matching its field names against every registered struct and using the first
type that fits. That scan (two copies, both in `eval_method_call`) iterated
`self.types`, a `HashMap`, and returned on the first hit:

```rust
for (type_name, type_def) in &self.types {
    …
    let matches = field_names.iter().all(|f| def_fields.contains(f));
    if matches { /* call type_name·method, bind Self to type_name */ }
}
```

`HashMap` iteration order is seeded per process. The match is also a **subset**
test, so `{x}` matches every struct that has an `x`. With more than one
candidate the recovered type — and therefore the name of whatever the method
returns — changed from run to run.

That is the coin flip. In the reduced case below the scan lands on `P`,
`shapes·P`, `Q` or `shapes·Q`; only bare `P` carries the `Add` impl, so the
program worked in 8 runs out of 20 and reported `Invalid struct operation` in
the other 12.

```sigil
// shapes.sg — two structs of the same shape, one with an operator impl
☉ Σ P { ☉ x: f32 }
⊢ P {
    ☉ rite new(x: f32) -> Self { Self { x } }
    ☉ rite twice(self) -> Self { Self·new(self.x * 2.0) }
}
⊢ Add ∀ P {
    type Output = Self;
    rite add(self, o: Self) -> Self { Self·new(self.x + o.x) }
}
☉ Σ Q { ☉ x: f32 }
⊢ Q {
    ☉ rite new(x: f32) -> Self { Self { x } }
    ☉ rite twice(self) -> Self { Self·new(self.x * 2.0) }
}

// main.sg
☉ rite main() -> !i32 {
    ≔ a = shapes·P·new(1.0);   // (1) returns a struct named "Self"
    ≔ b = a·twice();           // (2) recovered as P, shapes·P, Q or shapes·Q
    ≔ c = b + b;               // succeeds only when (2) landed on P
    println(c·x);
    ⤺ 0;
}
```

## What was ruled out

Recorded so it is not re-checked. Neither of these is the cause:

* **File iteration order.** `run_directory` (`main.rs`) sorts lib-first,
  main-last, and the comparator is consistent for those cases.
* **Impl-registry order.** `ImplRegistry`'s `generic_impls` and
  `concrete_impls` are `Vec`s, and `operator_impls`' candidate lists are `Vec`s
  too. All stable.

## The fix

* `type_name_for_self` takes the type from the path *prefix* — everything
  before the final segment — preferring its bare last segment, which is the
  name an unqualified call binds and the name impl methods and operator impls
  are filed under, and falling back to the fully qualified prefix.
* Both recovery scans go through `struct_types_matching_fields`, which returns
  candidates in a stable order: exact field-set matches first, then supersets,
  each group sorted by name.
* The enum-variant suffix scan in `eval_path` had the same shape — first hit
  while walking `self.types` — and is now collected and sorted the same way.

Fixing only the trigger would leave the coin flip in place for any other way a
value acquires the name `Self`; fixing only the scan would make the failure
deterministic rather than absent. Both are needed.

## The general rule

**A `HashMap` may not decide name resolution.** Anywhere the interpreter walks
`self.types` (or any other hash container) and returns on the first match, the
answer is per-process random and the symptom is an intermittent failure with no
input that explains it. Collect, order, then choose.
