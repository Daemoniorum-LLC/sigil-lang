# CLAUDE.md - Sigil

Sigil programming language ecosystem.

## Overview

Sigil is a custom programming language designed for AI-native development, featuring evidentiality markers and compile-time safety guarantees.

**Tech Stack:** Rust (canonical compiler), Sigil (self-hosted experimental compiler)

**Status:** Production-ready Rust compiler with 100% test pass rate ✅🔥

## Structure

- `sigil-lang/parser/` - **Canonical Rust compiler** (3.1MB, production-ready)
- `sigil-lang/jormungandr/` - Legacy self-hosted compiler (experimental)
- `sigil-web-interface/` - Web playground (submodule)
- `docs/` - Language documentation

## Quick Start

```bash
cd sigil-lang/parser
cargo build --release

# Run Sigil programs (interpreter)
./target/release/sigil run program.sg

# Compile to native binary (LLVM)
./target/release/sigil compile program.sg -o output

# JIT execution (Cranelift)
./target/release/sigil jit program.sg

# Run test suite (233/233 passing = 100%)
cd ../jormungandr/tests
./run_tests_rust.sh
```

## Compiler Features

The Rust compiler includes:
- ✅ **Interpreter** - Direct execution of Sigil programs
- ✅ **JIT Compiler** - Cranelift-based just-in-time compilation
- ✅ **AOT Compiler** - LLVM-based ahead-of-time compilation to native binaries
- ✅ **Type Checker** - Static type analysis
- ✅ **Standard Library** - Comprehensive stdlib (1.2MB, optimized)

## Test Results

**Current Status**: 233/233 P0 tests passing (100%) 🏆

**No Known Limitations**: All tests pass, including mutable borrow semantics and Drop trait!

See `sigil-lang/jormungandr/tests/TEST-RESULTS-2026-01-15.md` for detailed results.

## Skills

- `sigil-build` - Build Sigil projects
- `sigil-test` - Run Sigil tests
- `sigil-struct` - Generate structs
- `sigil-impl` - Generate implementations
- `sigil-evidentiality` - Evidentiality reference

## Related Projects

- `styx/` - AI-native git platform built with Sigil
- `nyx/` - Uses Sigil components

## Development

For detailed development information, see:
- `sigil-lang/CLAUDE.md` - Compiler architecture and commands
- `sigil-lang/jormungandr/tests/TEST-RESULTS-2026-01-15.md` - Test results

## Recent Achievements

- ✅ Rust compiler restored from git history (Jan 15, 2026)
- ✅ 99% test pass rate achieved
- ✅ All fixable bugs resolved
- ✅ Declared canonical/production-ready

**The Sigil Rust compiler is ready for serious development work!** 🔥
