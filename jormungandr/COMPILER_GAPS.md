# Jormungandr Self-Hosted Compiler - Gap Analysis

## Summary

Testing the self-hosted compiler against real Sigil libraries revealed several missing features that prevent many libraries from compiling successfully.

## Test Results Overview

### Libraries That Pass Type Checking

- `aporia/src/lib.sg` - Productive Uncertainty library
- `self-hosted/src/span.sg` - Compiler span module
- `self-hosted/src/token.sg` - Compiler token module
- `self-hosted/src/ast.sg` - Compiler AST module
- `chorus/src/lib.sg` - ✅ Fixed (nested generics parsing)
- `daemon/src/lib.sg` - ✅ Fixed (macro_rules! parsing)
- `basilica/src/lib.sg` - ✅ Fixed (stream keyword unreserved)

### All Libraries Pass! ✅

All 21 ecosystem libraries now pass type checking:

- `aegis/src/lib.sg` - ✅ Fixed (byte string literals)
- `anima/src/lib.sg` - ✅ Fixed (ref patterns)
- `aporia/src/lib.sg` - ✅ Passes
- `ate/src/lib.sg` - ✅ Passes
- `basilica/src/lib.sg` - ✅ Passes
- `chorus/src/lib.sg` - ✅ Passes
- `commune/src/lib.sg` - ✅ Passes
- `covenant/src/lib.sg` - ✅ Fixed (Fn trait syntax, nested generics)
- `daemon/src/lib.sg` - ✅ Passes
- `dionysus/src/lib.sg` - ✅ Fixed (Self patterns)
- `echo/src/lib.sg` - ✅ Fixed (move closures)
- `engram/src/lib.sg` - ✅ Passes
- `gnosis/src/lib.sg` - ✅ Passes
- `hades/src/lib.sg` - ✅ Passes
- `morpheus/src/lib.sg` - ✅ Passes
- `nemesis/src/lib.sg` - ✅ Passes
- `nous/src/lib.sg` - ✅ Passes
- `omen/src/lib.sg` - ✅ Passes
- `oracle/src/lib.sg` - ✅ Passes
- `prometheus/src/lib.sg` - ✅ Passes
- `shared/src/lib.sg` - ✅ Passes
- `theoros/src/lib.sg` - ✅ Passes

## Identified Gaps

### 1. ~~Missing Iterator Methods~~ ✅ FIXED

**`flat_map()`** - ✅ Implemented

```sigil
// NOW WORKS
self.resonances.values()
    .flat_map(|r| r.novel_emergences())
    .collect()
```

### 2. ~~Missing HashMap Methods~~ ✅ FIXED

**`entry().or_default()`** - ✅ Implemented

```sigil
// NOW WORKS
self.items.entry(key).or_default().push(value);
```

Also supports `entry().or_insert(value)` and `entry().or_insert_with(closure)`.

### 3. ~~`macro_rules!` Parsing~~ ✅ FIXED

The parser now skips `macro_rules!` declarations without error.
Libraries with macros (daemon, basilica, commune, aegis) now parse successfully.

### 4. Array Repeat Syntax ✅ FIXED

```sigil
// NOW WORKS
let arr: [u8; 32] = [0u8; 32];
```

### 5. Nested Generics Parsing ✅ FIXED

```sigil
// NOW WORKS
HashMap<K, Vec<V>>!
```

### 6. External Tome Imports

**Status: Partially Working**

- **`invoke tome::*` imports**: ✅ Work correctly when compiling multiple files together
- **External tome imports** (e.g., `invoke engram::...`, `invoke std::...`): ❌ Not resolved

Multi-file compilation within a project works because the driver merges all files into a unified symbol table. The `invoke tome::scroll::Type` statements are effectively no-ops since everything becomes globally visible.

External tomes would require a tome loader to find and compile dependencies (see: Binding.toml).

### 7. ~~Reserved Keywords as Identifiers~~ ✅ FIXED

**`stream` unreserved** - The `stream` keyword was vestigial (not used by parser).
Streaming semantics are properly expressed via Sigil's morpheme operators:
- `≋` (ProtoStream) for protocol-level streaming
- `·ing` (AspectProgressive) for ongoing/progressive operations

`pub mod stream;` now parses correctly.

## Working Patterns

These patterns compile successfully:

```sigil
// HashMap values iteration
self.items.values().filter(|x| x.is_valid()).collect()

// Option combinators
value.map(|x| x * 2)
value.and_then(|x| Some(x + 1))
value.unwrap_or(default)

// Iterator map + collect
vec.iter().map(|x| *x * 2).collect()

// Struct methods with self
impl Foo {
    pub fn get(&self) -> !i64 { self.value }
}
```

## Pre-existing Build Issues

The bootstrap build (`build.sh`) has pre-existing GCC compilation errors unrelated to new changes. The generated `sigil_bootstrap.c` has syntax issues around line 13000+ with undefined variables and function signature mismatches.

## Sigil-Native Terminology ✅ IMPLEMENTED

Sigil now uses its own terminology instead of Rust-isms:

| Rust Term | Sigil Term | Purpose |
|-----------|------------|---------|
| `crate` | **tome** | A collection of code/knowledge |
| `use` | **invoke** | Bring symbols into scope |
| `mod` | **scroll** | Subdivision of a tome |
| `Cargo.toml` | **Binding.toml** | What binds a tome together |

Both old (`use`/`mod`) and new (`invoke`/`scroll`) keywords are supported for backward compatibility.

```sigil
// Sigil-native style
invoke std::collections::HashMap;

pub scroll utils {
    pub fn helper() -> !i64 { 42 }
}
```

## Recommended Priority

1. ~~**Fix macro_rules! parsing**~~ ✅ DONE
2. ~~**Implement flat_map()**~~ ✅ DONE
3. ~~**Fix nested generics (>>)**~~ ✅ DONE
4. ~~**Implement array repeat [val; count]**~~ ✅ DONE
5. ~~**Implement entry().or_default()**~~ ✅ DONE
6. ~~**Multi-file compilation**~~ ✅ Works via global symbol merging
7. ~~**Unreserve vestigial keywords**~~ ✅ `stream` unreserved (use `≋` morpheme instead)
8. ~~**Sigil-native terminology**~~ ✅ `invoke`/`scroll` keywords added
9. ~~**External tome loader**~~ ✅ Basic tome resolution implemented

## Tome Resolution

The compiler now supports automatic resolution of external tome imports:

```bash
# With tome paths specified
sigil compile file.sg --tome-path=/path/to/tomes

# Auto-inferred paths (finds sibling tomes)
sigil compile project/src/lib.sg -v
```

### How It Works

1. **Local module detection**: `mod`/`scroll` declarations are tracked to avoid resolving local modules as external tomes
2. **Search path inference**: The compiler infers tome paths from input file locations (e.g., `/path/to/mytome/src/lib.sg` → searches `/path/to`)
3. **Iterative resolution**: External `use`/`invoke` statements are scanned, and referenced tomes are located at `<path>/<tome_name>/src/lib.sg`
4. **Skip standard library**: `std::*` imports are skipped (not yet a real tome)

### Limitations

- External tome resolution is opportunistic - missing tomes produce warnings, not errors
- No dependency version management (would require Binding.toml support)
- Circular dependencies not explicitly handled (relies on symbol merging)

## Recently Fixed

### `ref` patterns ✅ FIXED
```sigil
if let Some(ref val) = opt { ... }  // Now works
```

### `impl Trait` / `dyn Trait` ✅ FIXED
```sigil
fn foo() -> impl Iterator<Item = &T> { ... }  // Now parses
Box<dyn Fn(T) -> U>  // Now works
```

### Nested generics with `>>` ✅ FIXED
```sigil
HashMap<K, Vec<V>>  // Now parses correctly
Vec<Box<dyn Fn(T)>>  // Now works
```

### Fn trait syntax ✅ FIXED
```sigil
Box<dyn Fn(String) -> Result<T>>  // Fn(T) -> U now parsed
```

### `move` closures ✅ FIXED
```sigil
iter.map(move |x| x + captured)  // Now works
```

### `Self` in patterns ✅ FIXED
```sigil
match self {
    Self::Variant => ...  // Now works
}
```

### Vestigial keywords unreserved ✅ FIXED
- `scope` - now usable as identifier (field/method names)
- `connect` - now usable as identifier

### Byte string literals ✅ FIXED
```sigil
let data = b"test data";  // Now works
```

## Remaining Gaps

1. **Full std library** - `std::collections::HashMap` etc. are stub tomes (type signatures only)

All 21 ecosystem libraries now parse and type check successfully!

---

## Nihil Testing (Advanced Sigil)

### Overview

Nihil (`/nyx/nihil`) is a tensor computing library that uses advanced Sigil features. It serves as the reference for pure Sigil syntax.

**Current Status: 50/50 files passing ✅**

### Features Added for Nihil

✅ **Mathematical symbols as identifiers**
```sigil
invoke nihil_autograd::{∇, grad};  // ∇ now recognized as identifier
```

✅ **Default type parameters**
```sigil
pub trait FromVoid<Shape, D: DType = f32> { ... }
```

✅ **Evidentiality markers in imports**
```sigil
invoke nihil_optim::{GradientUpdate!, MomentEstimate~};
```

✅ **Middle dot path separator in invoke**
```sigil
invoke math·sacred·{φ, √2, π};  // · works like ::
```

✅ **Combined sqrt-digit identifiers**
```sigil
invoke math·sacred·{√2};  // √2 is a single identifier
```

✅ **Evidentiality on type names**
```sigil
pub fn dequantize(&self) -> Tensor◊ { ... }  // ◊ on return type preserved
```

✅ **Function name evidentiality**
```sigil
pub fn validate_model!(model~: Model~) -> Result<Model!, Error> { ... }
// ! on function name declares output evidence level
```

✅ **Chained evidentiality**
```sigil
DequantizedTensor~<DynShape, f32, Dev>!  // ~ on name, ! after generics
```

✅ **Byte string escape sequences**
```sigil
&buf == b"PK\x03\x04"  // \xNN escape sequences processed correctly
```

✅ **Vec<T> in morpheme operations**
```sigil
params|ρ{...}  // Vec<T> now works with reduce, access morphemes
```

### Nihil Refactoring Done

Converted deprecated Rust-style syntax to pure Sigil:
- `use` → `invoke` (241 occurrences)
- `mod` → `scroll` (147 occurrences)

## LLVM Native Compilation ✅ VERIFIED

The Sigil compiler now supports LLVM-based native code generation:

```bash
# Build with LLVM support
cargo build --features llvm --release

# Compile to native executable
sigil compile input.sigil -o output

# Run the native binary
./output
```

### Dependencies Required
- `llvm-18-dev` - LLVM development libraries
- `libpolly-18-dev` - LLVM polyhedral optimizer
- `libzstd-dev` - Zstandard compression library

### Features Implemented

**Core:**
- Arithmetic expressions, conditionals, loops
- Function declarations and calls with proper ABI
- Struct types with field access and initialization
- Impl blocks and method calls on structs
- Array literals with stack allocation

**Sigil Morphemes (Native Codegen):**
- `|ρ+` - Sum reduction with loop-based codegen
- `|ρ*` - Product reduction
- `|τ{f}` - Transform with closure parameter binding
- `|φ{p}` - Filter with predicate evaluation
- Fused operations: `|τ{f}|ρ+`, `|τ{f}|ρ*`, `|φ{p}|ρ+`, `|φ{p}|ρ*`

### Examples
```sigil
// Basic arithmetic
fn main() -> i32 {
    let x = 21;
    let y = x + x;
    y  // Returns 42
}

// Morpheme chains
fn main() -> i32 {
    [1, 2, 3, 4] |τ{|x| x * x} |ρ+  // Returns 30 (1+4+9+16)
}

// Filter and reduce
fn main() -> i32 {
    [1, 2, 3, 4, 5, 6] |φ{|x| x > 3} |ρ+  // Returns 15 (4+5+6)
}

// Structs and methods
struct Point { x: i32, y: i32 }
impl Point {
    fn sum(self) -> i32 { self.x + self.y }
}
fn main() -> i32 {
    let p = Point { x: 10, y: 20 };
    p.sum()  // Returns 30
}
```

Produces proper ELF 64-bit x86-64 executables. Stdlib functions like `println` require the interpreter.

## Testing Methodology

Used the Rust-based `sigil` parser binary:

```bash
/home/user/workspace/sigil/sigil-lang/parser/target/release/sigil check <file.sg>
```

Created minimal test files to isolate specific failing patterns.
