# WASM Method Dispatch Roadmap

## Overview

This roadmap covers implementing method call dispatch for the WASM compiler, enabling sigil-web and other libraries to compile to WebAssembly.

## Current State

### What Works
- Multi-file module linking (`scroll foo;` file-based modules)
- Impl method registration with qualified names (`Type::method`)
- Type aliases with inline enum/struct definitions
- Primitive type generics (`Cell[bool]`, `Option[i64]`)
- Basic function calls and morpheme pipelines
- 22 array/morpheme imports, DOM, events, signals, async

### What's Missing
Method calls like `receiver·method(args)` need type-aware dispatch. Currently fails with "undefined function" errors.

## Phase 1: Primitive Method Dispatch

**Goal**: Handle method calls on primitive types by mapping to existing imports.

### 1.1 String Methods (Priority: HIGH)
```
receiver·to_string()  -> string::from_int (if i64) / string::from_float (if f64)
receiver·len()        -> string::length
receiver·clone()      -> no-op (strings are immutable in WASM)
```

**Implementation**:
- Add `compile_primitive_method()` in closures.rs
- Check receiver type context (literal type, variable declaration type)
- Map method names to import functions

### 1.2 Numeric Methods (Priority: MEDIUM)
```
i64·abs()   -> math::abs (cast to f64, back to i64)
f64·sqrt()  -> math::sqrt
f64·floor() -> math::floor
f64·ceil()  -> math::ceil
f64·round() -> math::round
```

### 1.3 Option/Result Methods (Priority: HIGH)
These are critical for sigil-web (36 `unwrap()` calls, 9 `expect()` calls).

```
Option·unwrap()    -> extract value or trap
Option·is_some()   -> check tag
Option·is_none()   -> check tag
Option·expect(msg) -> extract value or trap with message
Option·ok()        -> identity (Option -> Option)
Result·unwrap()    -> extract Ok value or trap
Result·ok()        -> convert Result to Option
```

**Implementation**:
- Option/Result are enums with tag + payload
- `unwrap()` checks tag, returns payload or traps
- Need to add WASM `unreachable` instruction for panics

## Phase 2: Collection Methods

**Goal**: Implement Vec, HashMap, and iterator methods.

### 2.1 Vec Methods (Priority: HIGH)
sigil-web uses: `push` (15), `get` (31), `len` (4), `iter` (6)

```
Vec·new()      -> morpheme::array_new
Vec·push(val)  -> morpheme::array_push
Vec·get(idx)   -> morpheme::array_get (returns Option)
Vec·len()      -> morpheme::array_len
Vec·is_empty() -> morpheme::array_len == 0
Vec·iter()     -> return array handle for morpheme pipeline
Vec·first()    -> morpheme::array_first
Vec·last()     -> morpheme::array_last
```

### 2.2 HashMap Methods (Priority: MEDIUM)
sigil-web uses: `new` (in VElement), `insert` (5), `get`, `contains_key` (3)

**Approach**: Implement HashMap as two parallel arrays (keys, values) + hash function.

```
HashMap·new()           -> create key/value arrays
HashMap·insert(k, v)    -> hash, store in arrays
HashMap·get(k)          -> hash, linear probe, return Option
HashMap·contains_key(k) -> get().is_some()
HashMap·keys()          -> return keys array
HashMap·iter()          -> zip keys/values
```

**New imports needed**:
```
hashmap::new()           -> i64 (handle)
hashmap::insert(h,k,v)   -> void
hashmap::get(h,k)        -> i64 (Option-encoded)
hashmap::contains(h,k)   -> i32 (bool)
hashmap::keys(h)         -> i64 (array handle)
```

### 2.3 Iterator Methods (Priority: LOW)
```
iter·map(f)     -> morpheme::array_map
iter·filter(f)  -> morpheme::array_filter
iter·collect()  -> identity (already an array)
iter·enumerate()-> zip with index array
```

## Phase 3: Cell/RefCell Methods

**Goal**: Implement interior mutability types.

### 3.1 Cell Methods (Priority: HIGH)
sigil-web uses: `get` (via Cell), `set` (22), `new` (55 total)

```
Cell·new(val)  -> allocate, store value
Cell·get()     -> load value (Copy types only)
Cell·set(val)  -> store value
Cell·replace() -> swap and return old
```

**Implementation**:
- Cell is just a heap pointer to a value
- `get()` loads from memory
- `set()` stores to memory

### 3.2 RefCell Methods (Priority: HIGH)
sigil-web uses: `borrow` (16), `borrow_mut` (20), `new`

```
RefCell·new(val)    -> allocate value + borrow flag
RefCell·borrow()    -> check flag, return Ref
RefCell·borrow_mut()-> check flag, return RefMut
Ref·deref()         -> load value
RefMut·deref_mut()  -> load/store value
```

**Implementation**:
- RefCell: [borrow_count: i32, value: T]
- borrow_count: 0 = unborrowed, >0 = shared borrows, -1 = exclusive borrow
- `borrow()` increments count (trap if -1)
- `borrow_mut()` sets to -1 (trap if != 0)
- Drop decrements/resets count

**New imports needed**:
```
refcell::new(val)      -> i64 (handle)
refcell::borrow(h)     -> i64 (Ref handle, traps on conflict)
refcell::borrow_mut(h) -> i64 (RefMut handle, traps on conflict)
refcell::drop_ref(h)   -> void
refcell::drop_refmut(h)-> void
refcell::get(ref)      -> i64 (value)
refcell::set(refmut,v) -> void
```

## Phase 4: Trait Method Dispatch

**Goal**: Support trait methods like `Into::into()`, `Clone::clone()`.

### 4.1 Into/From Traits (Priority: HIGH)
sigil-web uses: `into()` (18)

```
val·into()  -> look up From/Into impl, call conversion
```

**Implementation**:
- Build impl registry during compilation
- At call site, look up `Into<TargetType>` impl for receiver type
- If found, call the impl function

### 4.2 Clone Trait (Priority: HIGH)
sigil-web uses: `clone()` (49)

```
val·clone() -> deep copy value
```

**Implementation**:
- Primitives: no-op (Copy)
- Strings: no-op (immutable/refcounted in JS)
- Structs: recursive clone of fields
- Enums: clone tag + payload

### 4.3 Default Trait (Priority: LOW)
```
Type::default() -> call Default::default impl
```

## Implementation Order

### Sprint 1: Core Primitives (This PR)
1. [ ] `to_string()` on i64/f64/bool
2. [ ] `unwrap()` / `expect()` on Option
3. [ ] `clone()` on primitives (no-op)
4. [ ] Basic type inference for method receiver

### Sprint 2: Collections
1. [ ] Vec methods via morpheme imports
2. [ ] HashMap basic operations (new imports)
3. [ ] `iter()` returning array handle

### Sprint 3: Interior Mutability
1. [ ] Cell (simple heap allocation)
2. [ ] RefCell with borrow checking

### Sprint 4: Traits
1. [ ] Into/From dispatch
2. [ ] Clone for compound types
3. [ ] Impl registry for trait lookup

## Technical Approach

### Type Inference for Method Dispatch

Currently `compile_method_call` receives only the method name, not the receiver type. We need:

1. **Option A: Runtime dispatch** (simpler)
   - Encode type tag in values
   - Check tag at runtime, branch to appropriate handler
   - Slower but works without full type inference

2. **Option B: Compile-time inference** (better)
   - Track variable types during compilation
   - Look up method based on inferred type
   - Requires type environment threading through compilation

**Recommendation**: Start with Option A for quick wins, migrate to Option B for performance.

### Adding New Imports

For each new runtime function:
1. Add to `imports.rs` with signature
2. Implement JS side in runtime harness
3. Add compile-time mapping in method dispatch

## Testing Strategy

1. **Unit tests**: Add WASM compilation tests for each method
2. **Integration**: Compile progressively more of sigil-web
3. **Runtime**: Test with Node.js harness

## Success Criteria

- [ ] sigil-web/src/vdom.sigil compiles without errors
- [ ] sigil-web/src/signals.sigil compiles without errors
- [ ] sigil-web/src/lib.sigil compiles to working WASM
- [ ] Basic counter example runs in browser
