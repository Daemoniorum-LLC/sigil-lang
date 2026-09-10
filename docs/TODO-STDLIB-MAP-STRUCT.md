# TODO: map_get/map_keys/map_set Should Work on Struct Values

**Filed:** 2026-02-19
**Origin:** Morgoth Phase 9 (Dynamic Startup + Profiles)
**Component:** `parser/src/stdlib.rs` (map_get, map_keys, map_set)
**Severity:** Ergonomics / DX — runtime crash with no compile-time warning

## Problem

`map_get()`, `map_keys()`, and `map_set()` only accept `Value::Map` but not
`Value::Struct`. Object literals (`{key: val}`) produce `Value::Struct`, so
calling any map function on them crashes at runtime:

```sigil
≔ cfg = { shell: "/bin/bash", max_panes: 12 };
map_get(cfg, "shell")   // Runtime error: map_get() requires map
map_keys(cfg)            // Runtime error: map_keys() requires map
```

This is surprising because `Value::Struct` stores its fields in the same
`Rc<RefCell<HashMap<String, Value>>>` as `Value::Map` uses internally.

## Current Code (stdlib.rs)

```rust
// line 1849
define(interp, "map_get", Some(2), |_, args| {
    let key = match &args[1] {
        Value::String(s) => s.to_string(),
        _ => return Err(RuntimeError::new("map_get() key must be string")),
    };
    match &args[0] {
        Value::Map(map) => Ok(map.borrow().get(&key).cloned().unwrap_or(Value::Null)),
        _ => Err(RuntimeError::new("map_get() requires map")),
    }
});
```

## Proposed Fix

Add `Value::Struct { fields, .. }` arms to `map_get`, `map_keys`, and
`map_set`:

```rust
match &args[0] {
    Value::Map(map) => Ok(map.borrow().get(&key).cloned().unwrap_or(Value::Null)),
    Value::Struct { fields, .. } => Ok(fields.borrow().get(&key).cloned().unwrap_or(Value::Null)),
    _ => Err(RuntimeError::new("map_get() requires map or struct")),
}
```

Same pattern for `map_keys` and `map_set`. The `fields` HashMap has the
identical type signature, so no conversion needed.

## Impact

- **Morgoth workaround:** Tests use `json_parse` round-trips or `.field` access
  instead of `map_get` on config objects. Not blocking, but adds friction.
- **General DX:** Any user who creates an object literal and passes it to
  `map_get` will hit a confusing runtime error with no guidance.
- **Type checker gap:** No compile-time error — the crash is purely runtime.

## Test Plan

```sigil
// Should pass after fix
rite main() {
    ≔ obj = { name: "test", count: 42 };
    ⤺ map_get(obj, "name") == "test";
    ⤺ map_get(obj, "missing") == null;
    ⤺ len(map_keys(obj)) == 2;
    map_set(obj, "extra", true);
    ⤺ map_get(obj, "extra") == true;
}
```

## References

- Morgoth `LESSONS-LEARNED.md` LL-014
- `parser/src/interpreter.rs:39` — `Value::Struct` definition
- `parser/src/stdlib.rs:1849` — `map_get` implementation
