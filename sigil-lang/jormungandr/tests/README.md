# Sigil Self-Hosted Compiler Tests

This directory contains tests for the Sigil self-hosted compiler, including the critical bootstrap test that verifies self-hosting.

## Test Files

| File | Description |
|------|-------------|
| `test_simple.sg` | Basic language features: functions, structs, enums, control flow |
| `test_evidentiality.sg` | Evidentiality system: !, ?, ~, ‽ markers |
| `test_morphemes.sg` | Morpheme operators: τ, φ, σ, ρ, Σ, Π, α, ω, λ, δ, γ |
| `bootstrap_test.sg` | Self-compilation fixed-point verification |

## Running Tests

### Individual Tests

```bash
# Compile and run a test
sigil run tests/test_simple.sg
sigil run tests/test_evidentiality.sg
sigil run tests/test_morphemes.sg
```

### Bootstrap Test

The bootstrap test verifies that the self-hosted compiler can compile itself:

```bash
# Run the full bootstrap test
sigil run tests/bootstrap_test.sg

# Run with verbose output
sigil run tests/bootstrap_test.sg -- -v

# Run smoke tests only
sigil run tests/bootstrap_test.sg -- --smoke
```

## Fixed Point Verification

The bootstrap test performs three phases:

1. **Phase 1**: Compile all compiler sources with the Rust-based Sigil compiler
2. **Phase 2**: Compile all compiler sources with the self-hosted Sigil compiler
3. **Phase 3**: Compare the generated C code from both phases

When the outputs are identical, we have achieved the **fixed point** - the compiler can compile itself to produce identical output.

```
┌─────────────────────────────────────────────────────────────┐
│                    Fixed Point Diagram                       │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│   compiler.sg ──→ Rust Compiler ──→ compiler1.c             │
│                                          │                   │
│                                          ▼                   │
│   compiler.sg ──→ Sigil Compiler ──→ compiler2.c            │
│                   (compiled by Rust)     │                   │
│                                          ▼                   │
│                                   compiler1.c == compiler2.c │
│                                          │                   │
│                                          ▼                   │
│                                   FIXED POINT! ✓             │
└─────────────────────────────────────────────────────────────┘
```

## Evidentiality in Tests

The test files themselves use Sigil's evidentiality system:

- `!Type` - Known/verified values (test assertions)
- `?Type` - Optional values (testing null handling)
- `~Type` - Reported values (simulated external input)

This dogfooding ensures the evidentiality system works correctly.

## Adding New Tests

When adding new tests:

1. Create a new `.sg` file in this directory
2. Include a `pub fn main() -> !i32` entry point
3. Use `assert()` for test assertions
4. Return 0 for success, non-zero for failure
5. Add the file to the test list if it should be part of bootstrap verification
