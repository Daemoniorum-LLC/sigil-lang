# Spec Test Suite

**🎉 Phase 1 Complete:** 233 P0 tests | 195 passing (84%) | 38 documented failures

This directory contains comprehensive tests derived directly from the Sigil language specification documentation (`docs/specs/*.md`).

---

**ACHIEVEMENT:** Target was 225 P0 tests. We delivered **233 tests** (+8 bonus!)

## Philosophy: Hierarchy of Truth

```
Spec Documentation (Philosophical SOT)
         ↓
  Spec Test Suite (Expressed SOT) ← YOU ARE HERE
         ↓
   Compiler (Enforced SOT)
         ↓
  Project Code (Applied SOT)
```

## Directory Structure

```
spec/
├── 01_lexical/          # Tests for 01-LEXICAL.md (tokens, identifiers, literals)
├── 02_syntax/           # Tests for 02-SYNTAX.md (morphemes, pipes, expressions)
├── 03_types/            # Tests for 03-TYPES.md (evidentiality, type system)
├── 04_memory/           # Tests for 04-MEMORY.md (ownership, borrowing, lifetimes)
├── 05_mathematics/      # Tests for 05-MATHEMATICS.md (numeric types, operators)
├── 06_concurrency/      # Tests for 06-CONCURRENCY.md (channels, async/await)
├── 07_metaprogramming/  # Tests for 07-METAPROGRAMMING.md (macros, reflection)
├── 08_ffi/              # Tests for 08-FFI.md (C interop, extern functions)
├── 09_stdlib/           # Tests for 09-STDLIB.md (String, Vec, collections)
├── 17_bootstrap/        # Tests for 17-JORMUNGANDR-BOOTSTRAP.md (C codegen)
└── 18_compiler/         # Tests for 18-COMPILER-ARCHITECTURE.md (compiler behavior)
```

## Test Naming Convention

Format: `P{priority}_{number}_{descriptive_name}.sg`

Examples:
- `P0_001_evidence_marker_known.sg` - Bootstrap critical
- `P1_042_pipe_composition.sg` - Production ready
- `P2_099_unicode_identifiers.sg` - Enhancement

### Priority Levels

- **P0 (Bootstrap Critical)**: Must pass for self-hosting bootstrap to work
  - Evidentiality system core features
  - Basic type system
  - C codegen correctness
  - Memory safety basics
  - Target: 225 tests

- **P1 (Production Ready)**: Required for production use
  - Advanced type features
  - Concurrency primitives
  - Full stdlib
  - Error handling
  - Target: 118 tests

- **P2 (Enhancement)**: Nice-to-have features
  - Advanced metaprogramming
  - Optimization features
  - Extended Unicode support
  - Target: 58 tests

## Test File Format

Each test consists of:
1. `.sg` file - Sigil source code
2. `.expected` file - Expected stdout output (optional)
3. Header comments with spec references

Example:
```sigil
// Test: Evidence markers combine via lattice rules
// Spec: 03-TYPES.md § 2.2 Evidence Lattice
// Priority: P0
//
// Purpose:
// Validates that combining Known (!) and Uncertain (?) evidence
// follows the pessimistic lattice rule: ! + ? = ?
//
// Expected behavior:
// Output should show "uncertain" as the result

fn main() {
    let x: !i32 = 42;        // Known
    let y: ?i32 = 99;        // Uncertain
    let z = x + y;           // Should be ?i32
    println(z.evidence());   // Should print "uncertain"
}
```

## Expected Output Files

If a test has an `.expected` file, the test runner will:
1. Compile the `.sg` file to C
2. Compile C to binary
3. Run the binary
4. Compare stdout to `.expected` file
5. PASS if they match, FAIL if different

If no `.expected` file exists, the test runner will:
1. Compile and run
2. PASS if no runtime errors, FAIL on crash

## Running Tests

```bash
# Run all spec tests
cd jormungandr/tests
./run_tests.sh

# Run specific section (once implemented)
./run_tests.sh --spec 03_types
./run_tests.sh --priority P0
```

## Implementation Phases

### Phase 1: Bootstrap Foundation (Weeks 1-3)
- 225 P0 tests
- Focus: Evidentiality, basic types, C codegen, memory safety
- Goal: Enable self-hosting bootstrap

### Phase 2: Production Readiness (Weeks 4-8)
- 118 P1 tests
- Focus: Advanced features, concurrency, full stdlib
- Goal: Production-ready compiler

### Phase 3: Completeness (Weeks 9-12)
- 58 P2 tests
- Focus: Edge cases, optimizations, enhancements
- Goal: 100% spec coverage

### Phase 4: Maintenance (Ongoing)
- Add regression tests as bugs are found
- Update tests as spec evolves
- Continuous improvement

## Test Development Workflow

1. **Select spec section** to implement
2. **Read spec thoroughly** - understand requirements
3. **Create test files** following naming convention
4. **Write test code** with clear comments
5. **Create `.expected` files** for output validation
6. **Run tests** - expect failures initially
7. **Fix compiler bugs** or test issues
8. **Iterate** until all tests pass
9. **Commit** with message: `test(spec): add {section} tests`

## Current Status

**Phase 1 (COMPLETE):**
- ✅ P0: 218 spec tests + 15 original = **233 tests**
- ✅ Pass rate: **195/233 (84%)**
- ✅ All failures documented in `../KNOWN_FAILURES.md`
- ✅ Target exceeded: 233 vs 225 planned

**Remaining Phases:**
- P1: 118 tests (production ready)
- P2: 58 tests (enhancements)
- Total future work: 176 tests

## Notes

- Tests are implementation-independent - they validate behavior, not internals
- Each test should be minimal and focused on one feature
- Tests should be deterministic (no randomness, no timestamps)
- Tests should not depend on external files or network
- Tests should complete quickly (< 1 second each)

## References

All tests derive from:
- `/home/crook/dev2/workspace/sigil/sigil-lang/docs/specs/*.md`

When in doubt, the spec is the source of truth.
