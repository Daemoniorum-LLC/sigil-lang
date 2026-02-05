# Rust Codegen TDD Test Suite

This directory contains the TDD test suite for the Sigil → Rust codegen backend.

## Methodology: Agent-TDD

Tests are written **BEFORE** implementation. They crystallize our understanding
of the expected behavior and serve as acceptance criteria.

## Test Structure

Each test consists of:
1. `test_*.sg` - Sigil input file
2. `test_*.rs.expected` - Expected Rust output

## Test Categories

| Test File | Coverage | Priority |
|-----------|----------|----------|
| `test_primitives.sg` | Functions, primitives, let bindings | P0 |
| `test_structs.sg` | Structs, generics, impl blocks | P0 |
| `test_traits.sg` | Traits, trait impls | P0 |
| `test_morphemes.sg` | Pipe operators, iterator chains | P0 |
| `test_const_generics.sg` | Const generic parameters | P0 |
| `test_evidence.sg` | Evidentiality markers | P1 |
| `test_async.sg` | Async functions, await | P1 |
| `test_extern.sg` | Extern blocks, FFI | P1 |

## Running Tests

```bash
# Run all rust codegen tests (once backend is implemented)
cd jormungandr/tests
./run_tests_rust.sh --section rust_codegen

# Or manually:
../../parser/target/release/sigil rust test_primitives.sg > output.rs
diff output.rs test_primitives.rs.expected
rustc --edition 2021 output.rs -o /dev/null
```

## Test Validation

A test passes when:
1. Sigil source parses successfully
2. Generated Rust matches expected output (modulo whitespace/formatting)
3. Generated Rust compiles with `rustc --edition 2021`

## Current Status

| Test | Parses | Generates | Compiles |
|------|:------:|:---------:|:--------:|
| test_primitives | - | - | - |
| test_structs | - | - | - |
| test_traits | - | - | - |
| test_morphemes | - | - | - |
| test_const_generics | - | - | - |
| test_evidence | - | - | - |
| test_async | - | - | - |
| test_extern | - | - | - |

**Legend:** ✓ = passing, ✗ = failing, - = not yet implemented

## Spec Reference

See: `docs/specs/RUST-CODEGEN-SPEC.md`
