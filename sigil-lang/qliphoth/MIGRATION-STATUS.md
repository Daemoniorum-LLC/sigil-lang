# Qliphoth v0.3.0 Migration Status

## Completed

### Syntax Migration (100%)
All 45 `.sigil` files converted from Rust-style to native Sigil syntax.

### Verified Examples (100%)
All examples compile and run with the Sigil compiler:

| Example | Status | Features Demonstrated |
|---------|--------|----------------------|
| `counter.sigil` | ✓ | Struct state, methods, increment/decrement |
| `counter_simple.sigil` | ✓ | Basic struct and impl |
| `todo.sigil` | ✓ | Vec, CRUD operations, filtering |
| `hello_vdom.sigil` | ✓ | Recursive VNode tree, HTML rendering |
| `simple_vdom.sigil` | ✓ | Basic Element struct |
| `qliphoth_demo.sigil` | ✓ | Page routing, theme, counter |

### P0 Sigil Syntax Reference

**Correct syntax patterns discovered during testing:**

```sigil
// Struct definition (no ☉ prefix)
sigil Counter {
    value: i64,
    name: String,
}

// Impl block
⊢ Counter {
    // Static method
    rite new() → Counter {
        Counter { value: 0, name: "".to_string() }
    }

    // Immutable method
    rite get(&this) → i64 {
        this.value
    }

    // Mutable method
    rite increment(&vary this) {
        this.value = this.value + 1;
    }
}

// Function parameters: use `str` for string literals
rite greet(name: str) → String {
    f"Hello, {name}!"
}

// While loop uses ⟳
rite count() {
    ≔ vary i = 0;
    ⟳ i < 10 {
        println(i);
        i = i + 1;
    }
}

// If/else uses ⎇/⎉
rite check(x: i64) {
    ⎇ x > 0 {
        println("positive");
    } ⎉ {
        println("non-positive");
    }
}

// Main function
rite main() {
    ≔ vary c = Counter·new();
    c.increment();
    println(c.get());
}
```

**Key syntax rules:**
- `sigil` not `☉ sigil` for struct definitions
- `rite` not `☉ rite` for methods/functions
- `&this` for immutable self, `&vary this` for mutable self
- `⟳` for while loops (not `⍟`)
- `⎇`/`⎉` for if/else
- `↩` for return
- `yea`/`nay` for true/false
- `str` for string literal parameters, `String` for owned strings
- `.to_string()` to convert `str` to `String`
- `Vec·new()` for new vectors
- `f"..."` for string interpolation

### Evidentiality Markers (Framework Code Only)
Note: Evidentiality markers (`!`, `?`, `~`) are used in the Qliphoth framework source code, not in application examples. The P0 compiler handles these at the framework level.

## Commands

```bash
# Run from sigil root
cd /home/crook/dev2/workspace/sigil

# Compile and run examples
./parser/target/release/sigil run sigil-lang/qliphoth/examples/counter_simple.sigil
./parser/target/release/sigil run sigil-lang/qliphoth/examples/counter.sigil
./parser/target/release/sigil run sigil-lang/qliphoth/examples/todo.sigil
./parser/target/release/sigil run sigil-lang/qliphoth/examples/hello_vdom.sigil
./parser/target/release/sigil run sigil-lang/qliphoth/examples/simple_vdom.sigil
./parser/target/release/sigil run sigil-lang/qliphoth/examples/qliphoth_demo.sigil

# Run test suite
cd jormungandr/tests && ./run_tests_rust.sh
```

## WASM Compilation Status

### Tested (January 2026)

**Build with WASM support:**
```bash
cd parser && CARGO_INCREMENTAL=0 cargo build --release --features wasm
```

**WASM Compilation Command:**
```bash
./parser/target/release/sigil wasm <file.sigil> -o output.wasm [--target browser|wasi]
```

### What Works

| Feature | Status | Notes |
|---------|--------|-------|
| Functions | ✓ | Regular functions compile to WASM |
| Control flow | ✓ | `⎇`/`⎉`, `⟳`, recursion all work |
| Arrays | ✓ | Fixed-size arrays with indexing |
| Morpheme operators | ✓ | `\|ρ+`, `\|ρ*` compile correctly |
| Basic arithmetic | ✓ | All numeric operations |
| **Structs with impl** | ✓ | Static methods (`Type·new()`) and instance methods work |
| **Multiple structs** | ✓ | Nested struct fields compile correctly |
| **Method calls** | ✓ | Both `&this` and `&vary this` patterns work |
| **println / print** | ✓ | Maps to `console.println_i64` import (JS runtime required) |

**Example with structs (works):**
```sigil
sigil Counter {
    count: i64,
}

⊢ Counter {
    rite new() → Counter {
        Counter { count: 0 }
    }

    rite increment(&vary this) {
        this.count = this.count + 1;
    }

    rite get(&this) → i64 {
        this.count
    }
}

rite main() → i64 {
    ≔ vary counter = Counter·new();
    counter.increment();
    counter.increment();
    counter.get()  // Returns 2
}
```

### Recently Added (January 2026)

| Feature | Status | Notes |
|---------|--------|-------|
| **Structs with impl** | ✓ | Static methods (`Type·new()`) and instance methods |
| **String interpolation** | ✓ | `f"Hello {name}"` compiles to concat chain |
| **to_string() method** | ✓ | Maps to `string.from_int` import |
| **println / print** | ✓ | Maps to `console.println_i64` import |
| **Vec<T>** | ✓ | `Vec·new()`, `.push()`, `.len()`, indexing all work |
| **clone() method** | ✓ | No-op for value types (stack copy) |

### Qliphoth Examples WASM Status

**All 6 examples compile to WASM!**

| Example | Status | Size | Features Used |
|---------|--------|------|---------------|
| `counter_simple.sigil` | ✓ | 2.8 KB | Structs, impl, println |
| `counter.sigil` | ✓ | 3.6 KB | Interpolated strings |
| `simple_vdom.sigil` | ✓ | 3.4 KB | to_string() |
| `hello_vdom.sigil` | ✓ | 4.2 KB | Vec<T>, push, len |
| `todo.sigil` | ✓ | 5.3 KB | Vec<T>, clone, filtering |
| `qliphoth_demo.sigil` | ✓ | 6.0 KB | Interpolated strings, structs |

### Browser Integration

The WASM backend generates valid WebAssembly bytecode. A JavaScript runtime is required to:
- Provide import functions (90+ registered in `imports.rs`)
- Handle string operations via JS interop
- Manage memory allocation

**Runtime:** `qliphoth/runtime/sigil_runtime.js` (✓ Updated January 2026)
- Added `println_i64`, `println_f64`, `println_str` console imports
- Full support for all 90+ WASM imports

The `playground/` directory contains a browser-based runner that uses a backend API. Direct WASM execution is now possible using `sigil_runtime.js`.

**Browser Usage:**
```html
<script type="module">
  import { loadSigilModule } from './runtime/sigil_runtime.js';
  const { runtime, instance } = await loadSigilModule('/path/to/module.wasm');
</script>
```

## Next Steps

1. ~~Apply evidentiality markers to core modules~~ ✓
2. ~~Complete accessibility implementation~~ ✓
3. ~~Rewrite examples that use `static vary`~~ ✓
4. ~~Verify all examples compile~~ ✓
5. ~~Test WASM compilation target~~ ✓
6. ~~Fix WASM impl block support~~ ✓ (January 2026)
7. ~~Add println WASM support~~ ✓ (January 2026)
8. ~~Add interpolated string support~~ ✓ (January 2026)
9. ~~Add to_string() method support~~ ✓ (January 2026)
10. ~~Add Vec<T> support to WASM~~ ✓ (January 2026)
11. ~~All 6 Qliphoth examples compile to WASM~~ ✓ (January 2026)
12. ~~JS runtime updated~~ ✓ (January 2026)
13. Browser integration testing with real examples
