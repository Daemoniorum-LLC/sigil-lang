# Rust Codegen TDD Test Suite

**Methodology:** Agent-TDD + Spec-Driven Development (SDD)
**Spec Reference:** `docs/specs/RUST-CODEGEN-PARSER-GAPS-SPEC.md`

This directory contains the TDD test suite for the Sigil → Rust codegen backend.

## Methodology

Tests are written following **Agent-TDD** principles:

1. **Tests crystallize understanding** — Each test specifies expected behavior
2. **Tests before fixes** — Write failing test, then implement fix
3. **Spec integration** — When tests reveal gaps, update spec (SDD)
4. **Semantic density** — Each test should teach something specific

## Test Structure

Each test consists of:
1. `test_*.sg` — Sigil input file
2. `test_*.rs.expected` — Expected Rust output

## Test Categories

### Core Tests (P0)

| Test File | Coverage | Gap | Status |
|-----------|----------|-----|--------|
| `test_primitives.sg` | Functions, primitives, let bindings | — | ✅ |
| `test_structs.sg` | Structs, generics, impl blocks | — | ✅ |
| `test_traits.sg` | Traits, trait impls | — | ✅ |
| `test_morphemes.sg` | Pipe operators, iterator chains | — | ✅ |
| `test_const_generics.sg` | Const generic parameters | — | ✅ |

### Extended Tests (P1)

| Test File | Coverage | Gap | Status |
|-----------|----------|-----|--------|
| `test_evidence.sg` | Evidentiality markers | — | ✅ |
| `test_async.sg` | Async functions, await | — | ✅ |
| `test_extern.sg` | Extern blocks, FFI | — | ✅ |

### Gap-Specific Tests (v0.4.0)

| Test File | Coverage | Gap Ref | Status |
|-----------|----------|---------|--------|
| `test_public_fields.sg` | Public struct field visibility | Gap E | ✅ |
| `test_where_clauses.sg` | Where clause emission | Gap F | ✅ |
| `test_fn_traits.sg` | Fn/FnMut/FnOnce trait syntax | Gap G | ✅ |
| `test_raw_pointers.sg` | Raw pointer type annotations | Gap I | ✅ |

## Running Tests

### Manual Verification

```bash
# Generate Rust from a single test
../../parser/target/release/sigil rust test_primitives.sg > output.rs

# Compare with expected output
diff output.rs test_primitives.rs.expected

# Verify generated Rust compiles
rustc --edition 2021 output.rs -o /dev/null --emit=metadata 2>&1
```

### Automated Test Runner

```bash
# Run all rust codegen tests
cd jormungandr/tests
./run_tests_rust.sh --section rust_codegen

# Run a specific test
./run_tests_rust.sh rust_codegen/test_where_clauses.sg
```

### Batch Verification Script

```bash
#!/bin/bash
# verify_rust_codegen.sh

SIGIL="../../parser/target/release/sigil"
PASS=0
FAIL=0

for sg in test_*.sg; do
    base="${sg%.sg}"
    expected="${base}.rs.expected"

    if [ ! -f "$expected" ]; then
        echo "SKIP: $sg (no expected file)"
        continue
    fi

    # Generate Rust
    $SIGIL rust "$sg" > /tmp/generated.rs 2>&1

    # Check if it compiles
    if rustc --edition 2021 /tmp/generated.rs -o /dev/null --emit=metadata 2>/dev/null; then
        echo "PASS: $sg"
        ((PASS++))
    else
        echo "FAIL: $sg"
        ((FAIL++))
    fi
done

echo ""
echo "Results: $PASS passed, $FAIL failed"
```

## Test Validation Criteria

A test passes when:

1. ✅ Sigil source parses successfully
2. ✅ Generated Rust compiles with `rustc --edition 2021`
3. ✅ Generated Rust matches expected output (semantic equivalence)

**Note:** Whitespace/formatting differences are acceptable. Focus is on semantic correctness.

## Writing New Tests

When adding a new test:

1. **Identify the gap** — What behavior needs testing?
2. **Check the spec** — Is this documented in RUST-CODEGEN-PARSER-GAPS-SPEC.md?
3. **Write the test** — Create `test_<name>.sg` with comprehensive examples
4. **Write expected output** — Create `test_<name>.rs.expected`
5. **Verify it fails first** — Ensure the test catches the issue
6. **Implement the fix** — Update rust_codegen.rs
7. **Verify it passes** — Run the test
8. **Update the spec** — Document the fix in the gaps spec

### Test File Template

```sigil
//! Test: <Feature Name> (Gap <X>)
//!
//! Spec Reference: RUST-CODEGEN-PARSER-GAPS-SPEC.md Section <N>
//!
//! This test verifies <what the test covers>.
//!
//! Key behaviors:
//! 1. <behavior 1>
//! 2. <behavior 2>
//! ...

// Test 1: <description>
<sigil code>

// Test 2: <description>
<sigil code>

// ... more tests covering edge cases
```

## Real-World Validation

In addition to these unit tests, the codegen is validated against the Nihil ML framework:

| Crate | Status | Notes |
|-------|--------|-------|
| nihil-core | ✅ | 8 warnings (scaffolding) |
| nihil-ops | ✅ | 9 warnings (unused) |
| nihil-nn | 🔲 | Pending |
| nihil-optim | 🔲 | Pending |

## Revision History

| Version | Date | Changes |
|---------|------|---------|
| 1.0 | 2026-02-04 | Initial test suite (8 tests) |
| 2.0 | 2026-02-05 | Added Gap E-I tests following SDD methodology |
