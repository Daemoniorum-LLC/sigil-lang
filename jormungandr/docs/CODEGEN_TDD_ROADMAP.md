# Sigil Codegen TDD Roadmap

## Goal
Achieve fixed-point compilation: native Sigil compiler produces identical output to bootstrap.

## Current State
- Native compiler processes all 13 source modules
- Generated C code has 811 compilation errors
- Main categories identified below

---

## Phase 1: Evidence Markers (541 errors)

**Problem**: Empty `()` generated instead of evidence level like `SIGIL_KNOWN`

**Example**:
```c
// BAD: sigil_with_evidence(val, ())
// GOOD: sigil_with_evidence(val, SIGIL_KNOWN)
```

### Test 1.1: Known evidence marker
```sigil
// test_evidence_known.sg
fn test() -> !i32 { 42 }
```
**Expected C**: `sigil_with_evidence(..., SIGIL_KNOWN)`

### Test 1.2: Uncertain evidence marker
```sigil
fn test() -> ?i32 { ?42 }
```
**Expected C**: `sigil_with_evidence(..., SIGIL_UNCERTAIN)`

### Test 1.3: Evidence in struct field
```sigil
struct Foo { x: !i32 }
fn test() -> !Foo { Foo { x: 1 } }
```

**Fix location**: `src/codegen.sg` - evidence emission logic

---

## Phase 2: Method Resolution (36+ errors)

**Problem**: Methods like `.to_string()`, `.join()` not resolved to runtime functions

### Test 2.1: to_string method
```sigil
fn test() -> !String { 42.to_string() }
```
**Expected C**: `sigil_to_string(sigil_int(42))`

### Test 2.2: join method on array
```sigil
fn test() -> !String { ["a", "b"].join(",") }
```
**Expected C**: `sigil_join(arr, sigil_string(","))`

### Test 2.3: len method
```sigil
fn test() -> !i64 { "hello".len() }
```
**Expected C**: `sigil_len(sigil_string("hello"))`

**Fix location**: `src/codegen.sg` - method call emission

---

## Phase 3: Field Access (34+ errors)

**Problem**: `.name`, `.symbol` etc. not using `sigil_struct_field()`

### Test 3.1: Simple field access
```sigil
struct Point { x: !i32, y: !i32 }
fn test(p: !Point) -> !i32 { p.x }
```
**Expected C**: `sigil_struct_field(p, "x")`

### Test 3.2: Nested field access
```sigil
fn test(p: !Point) -> !i32 { p.x.abs() }
```

### Test 3.3: Field access on self
```sigil
impl Point {
    fn get_x(self) -> !i32 { self.x }
}
```

**Fix location**: `src/codegen.sg` - field access emission

---

## Phase 4: Type Coercion (17+ errors)

### Test 4.1: Option unwrap type
```sigil
fn test(opt: ?i32) -> !i32 { opt.unwrap() }
```

### Test 4.2: Result unwrap
```sigil
fn test(r: !Result<i32, String>) -> !i32 { r.unwrap() }
```

**Fix location**: `src/codegen.sg` - type handling in call emission

---

## Phase 5: Wildcard Patterns

**Problem**: `_` used as variable name instead of ignored

### Test 5.1: Wildcard in match
```sigil
fn test(x: !i32) -> !i32 {
    match x {
        1 => 10,
        _ => 0,
    }
}
```
**Expected C**: Default case, not `_` variable

---

## Test Harness

Each test follows this pattern:

```bash
# 1. Create minimal test file
echo 'fn test() -> !i32 { 42 }' > /tmp/test.sg

# 2. Compile with native compiler
./build/sigil compile /tmp/test.sg > /tmp/test.c

# 3. Verify C compiles
gcc -c /tmp/test.c -o /dev/null

# 4. Check expected patterns
grep -q "SIGIL_KNOWN" /tmp/test.c && echo "PASS" || echo "FAIL"
```

---

## Success Criteria

1. Each phase reduces error count significantly
2. All tests compile to valid C
3. Final: `diff build/sigil_bootstrap.c build/sigil2.c` shows minimal differences
4. Ultimate: Native compiler builds itself, output is identical (fixed-point)

---

## Priority Order

1. **Phase 1** (evidence) - Fixes 67% of errors
2. **Phase 3** (field access) - Common pattern
3. **Phase 2** (methods) - Common pattern
4. **Phase 4** (types) - Edge cases
5. **Phase 5** (wildcards) - Edge cases
