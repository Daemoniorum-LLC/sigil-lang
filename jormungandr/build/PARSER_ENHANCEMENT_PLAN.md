# Parser Enhancement Plan for Styx Compilation

**Date:** 2026-01-14
**Goal:** Enable full Styx compilation by enhancing parser to handle real-world Rust-style code

---

## Identified Parser Gaps

### 1. ✅ WORKING - Basic Files
These files parse successfully with current parser:
- `result.sigil` - Type aliases with generics
- `time.sigil` - Struct with methods
- `config.sigil` - Configuration structs
- `error.sigil` - Enum with methods

### 2. ❌ FAILING - Typed Closures
**File:** `secrets.sigil:2954`

**Pattern:**
```sigil
let end = rest.find(|c: char| !c.is_ascii_digit()).unwrap_or(rest.len());
```

**Issue:** Parser doesn't handle closure parameters with type annotations

**Syntax:**
```sigil
|param: Type| expression
|param1: T1, param2: T2| expression
```

**Needed:** Parse typed closure parameters in lambda expressions

---

### 3. ❌ FAILING - Trait Bounds with Multiple Constraints
**File:** `id.sigil:12`

**Pattern:**
```sigil
pub trait Identifier: Clone + Eq + Hash + Display {
    ...
}
```

**Issue:** Multiple trait bounds on trait declaration

**Needed:** Parse `Trait: Bound1 + Bound2 + Bound3` syntax

---

### 4. ❌ FAILING - Complex Generic Types
**File:** `audit.sigil` (from earlier testing)

**Pattern:**
```sigil
pub data: HashMap<String, serde_json::Value>!
```

**Issue:** Nested types with paths (`serde_json::Value`)

**Needed:** Parse `Module::Type` inside generic parameters

---

### 5. ❌ SEGFAULT - Unknown Pattern in id.sigil/hash.sigil
**Files:** `id.sigil`, `hash.sigil`, `pii.sigil`

**Issue:** Silent segfault - no error message

**Suspects:**
- Array type annotations: `[u8; 32]`
- Reference syntax in return types: `&[u8]`
- Trait bounds on methods
- Complex morpheme operations: `bytes|τ{b => format!("{:02x}", b)}|collect::<String>()`

**Investigation needed:** Run under debugger to find crash location

---

## Priority Order

### Phase 1: Critical (Blocks most files)
1. **Typed closure parameters** - `|c: char| expr`
2. **Multiple trait bounds** - `Trait: A + B + C`
3. **Nested path types** - `HashMap<String, Module::Type>`

### Phase 2: Important (Blocks advanced features)
4. **Fixed-size array types** - `[u8; 32]`
5. **Reference types in signatures** - `&[u8]`, `&str`
6. **Associated types in traits** - `type Item; fn next() -> Option<Self::Item>`

### Phase 3: Advanced (Optional features)
7. **Turbofish in method calls** - `collect::<Vec<T>>()`
8. **Impl Trait syntax** - `impl Iterator<Item = T>`
9. **Complex morpheme expressions** - Advanced morpheme chaining

---

## Implementation Strategy

### Step 1: Add Typed Closure Support
**Location:** `src/parser.sg` - `parse_lambda` or `parse_closure`

**Current:**
```sigil
|param| expr        // ✅ Works
```

**Add:**
```sigil
|param: Type| expr  // ❌ Needs implementation
```

**Changes:**
1. In closure parameter parsing, after identifier
2. Check for `:` token
3. Parse type annotation
4. Store type in AST node

### Step 2: Multi-Trait Bounds
**Location:** `src/parser.sg` - `parse_trait_decl`

**Current:**
```sigil
trait Foo: Bar { }   // ✅ Works (single bound)
```

**Add:**
```sigil
trait Foo: Bar + Baz + Qux { }  // ❌ Needs implementation
```

**Changes:**
1. After first bound, loop checking for `+` token
2. Parse additional trait names
3. Store as Vec of bounds in AST

### Step 3: Path Types in Generics
**Location:** `src/parser.sg` - `parse_type`

**Current:**
```sigil
HashMap<String, Value>  // ✅ Works
```

**Add:**
```sigil
HashMap<String, serde_json::Value>  // ❌ Path not parsed
```

**Changes:**
1. In generic argument parsing
2. Parse full path including `::`
3. Handle `Module::Submodule::Type` recursively

### Step 4: Debug Segfaults
**Tool:** Run sigil2 under gdb

```bash
gdb ./sigil2
run check /path/to/id.sigil
bt  # Get backtrace
```

**Likely causes:**
- Infinite recursion in type parsing
- Array index out of bounds
- Null pointer dereference in AST construction

---

## Testing Strategy

### Test 1: Typed Closures
```sigil
fn test() {
    let nums = [1, 2, 3, 4, 5];
    let evens = nums|φ{|x: i32| x % 2 == 0};
    let doubled = nums|τ{|x: i32| x * 2};
}
```

### Test 2: Multi-Trait Bounds
```sigil
pub trait Identifier: Clone + Eq + Hash + Display {
    fn as_bytes(&self) -> &[u8];
}
```

### Test 3: Nested Path Types
```sigil
use std::collections::HashMap;
use serde_json::Value;

struct Config {
    data: HashMap<String, Value>!
}
```

### Test 4: Array Types
```sigil
struct Sha256 {
    bytes: [u8; 32]!
}

impl Sha256 {
    fn as_bytes(&self) -> &[u8; 32] {
        &self.bytes
    }
}
```

---

## Expected Outcomes

### After Phase 1
- ✅ `secrets.sigil` compiles (typed closures)
- ✅ `id.sigil` compiles (multi-trait bounds)
- ✅ `audit.sigil` compiles (path types)
- ⏸️ Some files may still fail (array types, references)

### After Phase 2
- ✅ `hash.sigil` compiles (array types, references)
- ✅ All basic styx-core files compile
- ⏸️ Advanced features still limited

### After Phase 3
- ✅ Complete styx-core compilation
- ✅ Can compile full Styx platform (26 crates)
- ✅ Parser feature-complete for real-world Sigil code

---

## Risk Assessment

### Low Risk
- Typed closures - well-defined syntax
- Multi-trait bounds - straightforward extension

### Medium Risk
- Path types in generics - may affect existing code
- Array types - interaction with slice syntax

### High Risk
- Segfault fixes - need careful debugging
- May uncover deeper issues in type checker or codegen

---

## Rollback Plan

If enhancements break existing functionality:
1. Use git to revert changes
2. Test with existing working files
3. Apply fixes incrementally
4. Maintain test suite

---

## Success Criteria

**Minimum Success:**
- [ ] `secrets.sigil` compiles (typed closures)
- [ ] `id.sigil` compiles (multi-trait bounds)
- [ ] `audit.sigil` compiles (path types)

**Full Success:**
- [ ] All 15 styx-core files compile
- [ ] Can generate working binaries from Styx code
- [ ] Method resolution still works correctly

**Stretch Goal:**
- [ ] Compile entire styx-db crate
- [ ] Compile styx-git crate
- [ ] Full Styx platform builds

---

## Next Steps

1. **Investigate segfaults first** - Understand what's breaking
2. **Implement typed closures** - Highest impact, lowest risk
3. **Add multi-trait bounds** - Critical for Styx code
4. **Test incrementally** - Verify each enhancement works
5. **Document changes** - Update COMPILER_GAPS.md

---

**Let's make the parser handle real-world Sigil code!** 🚀
