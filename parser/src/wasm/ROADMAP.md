# Sigil WASM Codegen Roadmap

## Overview

Modular WASM backend for Sigil, compiled with `--features wasm`.
Target: Complete feature parity with interpreter for web deployment.

## Module Structure

```
parser/src/wasm/
├── mod.rs              # Public API, WasmCompiler struct
├── types.rs            # Type definitions (LocalVar, CompiledFunction, etc.)
├── constants.rs        # Evidence tags, memory layout, type tags
├── imports.rs          # JS runtime import registration
├── literals.rs         # Literal value compilation
├── operators.rs        # Binary/unary operator emission
├── expressions.rs      # Expression compilation dispatcher
├── statements.rs       # Statement compilation
├── control_flow.rs     # If/while/for/match
├── morphemes.rs        # Pipe operators (τ, φ, σ, ρ, α, Ω, etc.)
├── closures.rs         # Closure compilation, capture analysis
├── patterns.rs         # Pattern matching compilation
├── structs.rs          # Struct/enum layout and access
├── async_sm.rs         # Async state machine generation
├── codegen.rs          # WASM module generation (bytecode emission)
├── error.rs            # Error types
└── tests/              # Integration tests
    ├── mod.rs
    ├── literals_test.rs
    ├── operators_test.rs
    ├── control_flow_test.rs
    ├── morphemes_test.rs
    ├── closures_test.rs
    ├── patterns_test.rs
    └── async_test.rs
```

## Phases

### Phase 1: Core Infrastructure ✅
- [x] Type definitions (LocalVar, CompiledFunction, ImportFn)
- [x] Constants (evidence tags, memory layout)
- [x] Import registration (90+ JS functions incl. string, async)
- [x] Error types with spans
- [x] Tests: Compiler instantiation, type registration

### Phase 2: Literals & Operators ✅
- [x] Integer, float, bool, null literals
- [x] String literals with data segment
- [x] Binary operators (+, -, *, /, %, etc.)
- [x] Comparison operators (==, !=, <, >, etc.)
- [x] Unary operators (-, !)
- [x] String concatenation (++) - calls string_concat runtime
- [x] Compound assignment (+=, -=, *=, /=) - desugared by parser
- [x] Tests: All literal types, operator combinations

### Phase 3: Control Flow ✅
- [x] If/else expressions
- [x] While loops
- [x] For loops (array iteration)
- [x] Break/continue
- [x] Return statements
- [x] Loop labels (break 'label, continue 'label)
- [x] Tests: Nested control flow, early returns, labeled loops

### Phase 4: Morpheme Operators ✅
- [x] τ (Transform/Map)
- [x] φ (Filter)
- [x] σ (Sort)
- [x] ρ (Reduce)
- [x] α (First), Ω (Last)
- [x] μ (Middle), ν (Nth)
- [x] χ (Random choice)
- [x] Σ (Sum), Π (Product) - via ρ+ and ρ*
- [x] Parallel morphemes (par·τ, par·φ) - compile closures to table, call runtime
- [x] Tests: Pipeline compositions, edge cases, parallel morphemes

### Phase 5: Closures ✅
- [x] Basic closure compilation
- [x] Capture analysis (free variables)
- [x] Environment allocation
- [x] Environment pointer extraction for captured closures
- [x] Mutable captures - cell indirection for captured variables
- [x] Nested closures - recursive capture analysis through closure scopes
- [x] Tests: Capture semantics, closure as value, mutable captures, nested closures

### Phase 6: Pattern Matching ✅
- [x] Wildcard patterns
- [x] Identifier binding
- [x] Literal patterns
- [x] Tuple patterns
- [x] Struct patterns
- [x] TupleStruct patterns (enum variants with data)
- [x] Path patterns (unit enum variants)
- [x] Slice patterns
- [x] Or patterns (a | b)
- [x] Range patterns
- [x] Rest patterns (..)
- [x] Guard expressions
- [x] Tests: Pattern matching compilation

### Phase 7: Async/Await ✅
- [x] Async function detection (is_async flag)
- [x] Async function body wrapping (Promise creation/resolution)
- [x] Await expression compilation (calls await_promise)
- [x] Evidentiality tagging for await results
- [x] Promise integration (promise_new, promise_resolve, await_promise imports)
- [x] Async state machine module (async_sm.rs) with await point analysis
- [x] Tests: Sequential awaits, evidentiality tagging, state machine analysis

### Phase 8: Module Generation
- [x] Type section
- [x] Import section
- [x] Function section
- [x] Memory section
- [x] Export section
- [x] Table section (indirect calls)
- [x] Data section (strings)
- [ ] Custom sections (debug info)
- [ ] Source maps
- [x] Tests: Valid WASM output, wasmparser validation (21 tests)

## TDD Workflow

For each feature:
1. Write failing test in appropriate `tests/*.rs`
2. Implement minimum code to pass
3. Refactor for clarity
4. Run full test suite
5. Update this roadmap

## Test Commands

```bash
# Run all WASM tests
cargo test --features wasm wasm::

# Run specific module tests
cargo test --features wasm wasm::tests::literals
cargo test --features wasm wasm::tests::morphemes

# Run with output
cargo test --features wasm wasm:: -- --nocapture
```

## Dependencies

Add to `Cargo.toml`:
```toml
[features]
wasm = ["wasm-encoder"]

[dependencies]
wasm-encoder = { version = "0.219", optional = true }
```

## Test Coverage (189 tests)

| Module           | Tests | Status |
|------------------|-------|--------|
| validation_tests | 27    | ✅     |
| operators        | 24    | ✅     |
| literals         | 22    | ✅     |
| statements       | 21    | ✅     |
| error            | 18    | ✅     |
| constants        | 13    | ✅     |
| expressions      | 11    | ✅     |
| morphemes        | 9     | ✅     |
| control_flow     | 9     | ✅     |
| closures         | 8     | ✅     |
| types            | 8     | ✅     |
| mod              | 8     | ✅     |
| imports          | 7     | ✅     |

## Success Metrics

- [ ] All 244 existing interpreter tests pass equivalent WASM tests
- [ ] demo.sigil compiles and runs in browser
- [ ] Bundle size < 10KB for hello world
- [ ] Compilation time < 100ms for typical programs
