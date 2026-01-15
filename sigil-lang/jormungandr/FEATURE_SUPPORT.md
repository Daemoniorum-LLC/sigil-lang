# Jormungandr Feature Support Matrix

Status: 2026-01-14

## ✅ Working Features

### Basic Syntax
- [x] Function definitions with explicit types
- [x] Type annotations (required on parameters)
- [x] Evidentiality markers (!~?‽)
- [x] Basic arithmetic (2 + 3 * 4)
- [x] String literals
- [x] Comments (// and /* */)
- [x] Return statements

### Types
- [x] Primitives: bool, i32, f64, String, char
- [x] Arrays: [T]
- [x] Tuples
- [x] Structs
- [x] Enums
- [x] References: &T, &mut T
- [x] Option<T>, Result<T, E>

### Control Flow
- [x] if/else conditionals
- [x] while loops
- [x] for loops
- [x] match expressions
- [x] break, continue

### Functions & Methods
- [x] Free functions
- [x] Methods on types (impl blocks)
- [x] Method calls with dot notation
- [x] Function calls
- [x] Closures with explicit types

### Modules
- [x] Module declarations (mod)
- [x] Use statements
- [x] Visibility (pub)

## ❌ Missing / Broken Features

### Type Inference
- [ ] **Parameter type inference** - Parameters MUST be explicitly typed
  ```sigil
  // BROKEN:
  fn greet(name) { ... }

  // REQUIRED:
  fn greet(name: !String) { ... }
  ```

- [ ] **Let binding type inference** - May work but not tested
  ```sigil
  let x = 42;  // Should infer i32
  ```

### Operators
- [ ] **Pipe operator (τ)** - Not tested
  ```sigil
  nums|τ{_ * 2}
  ```

- [ ] **Placeholder underscore in closures** - Not tested
  ```sigil
  {_ * 2}  // Single-argument closure shorthand
  ```

### String Operations
- [x] **String methods** - Fixed in source, but sigil2 has runtime bug
  - clone() - Fixed via multiline emission
  - is_empty() - Fixed via multiline emission
  - contains() - Implementation exists
  - push() - Implementation exists

### Standard Library
- [ ] **print() function** - Works but may be builtin only
- [ ] **Vec methods** - Partially implemented
- [ ] **Iterator methods** (map, filter, etc.) - Not tested
- [ ] **Collection methods** - Not tested

### Advanced Features
- [ ] **Generics** - Declared but not tested
- [✅] **Traits & User-Defined Type Methods** - FIXED!
  - Trait declarations work ✅
  - impl Trait for Type compiles ✅
  - impl blocks for custom types work ✅
  - Method calls now generate correct `sigil_TypeName____method()` ✅
  - **Fix:** Added IR type extraction for user-defined types (commit 3045b7efc)
  - **Note:** Bootstrap requires fixing cascading issues from old broken behavior
- [ ] **Lifetimes** - May not be implemented
- [ ] **Async/await** - Not implemented
- [ ] **Macros** - Not implemented

### Code Generation Issues
- [x] **Duplicate sigil_add** - Generates one-liner + full implementation
- [x] **Stray #endif** - Missing opening #ifndef
- [ ] **Method dispatch** - Uses sigil_contains() not sigil_String____contains()

## Testing Status

### Tested Programs
1. ✅ Simple typed function with print
2. ❌ hello.sigil (untyped parameters)
3. ⏳ Pipe operators
4. ⏳ Closure with _
5. ⏳ String methods
6. ⏳ Full Jormungandr self-compilation

### Critical for Styx
For Styx to compile, we minimally need:
1. ✅ Typed function parameters (Styx has these)
2. ✅ impl blocks for methods (should work)
3. ✅ Structs and enums (should work)
4. ❓ Module system with paths (needs testing)
5. ❓ Pattern matching (needs testing)
6. ❓ Generic functions (needs testing)
7. ❓ Trait bounds (needs testing)

## Next Steps
1. Test pipe operators and closures
2. Test module system thoroughly
3. Test generics and traits
4. Fix code generation bugs (duplicate sigil_add, stray endif)
5. Test Styx compilation
