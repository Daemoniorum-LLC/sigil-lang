# Sigil Programming Language

A polysynthetic programming language with evidentiality types, morpheme operators, and native performance through LLVM. Includes comprehensive **Agent Infrastructure** for building intelligent, self-aware systems.

## v0.4.0 — The Native Tongue Release

This release completes the transition to native Sigil syntax:

- **Native symbol syntax** — Full symbolic vocabulary (`λ` fn, `≔` let, `Σ` struct, `⊢` impl, `☉` pub, `·` paths)
- **WASM playground** — Browser-based Sigil execution environment
- **Evidentiality linting** — Correctness checking validates proper marker usage for data sources
- **LLVM FFI support** — Direct foreign function interface from Sigil to native code
- **Runtime enhancements** — Networking syscalls, threading primitives, async I/O with epoll
- **SIMD Backend** — AVX-512 F32x16 operations
- **CUDA Backend** — GPU compute via `--cuda` flag
- LSP server, formatter, linter, package manager
- HTTP and WebSocket clients
- Agent infrastructure libraries:
  - **Aegis** - Security & safety
  - **Anima** - State modeling
  - **Commune** - Multi-agent communication
  - **Covenant** - Collaborative protocols
  - **Daemon** - Background processes
  - **Gnosis** - Knowledge & learning
  - **Omen** - Planning & prediction
  - **Oracle** - Explainable decisions
  - **Engram** - Memory (episodic, semantic, procedural)

## Execution Backends

- **Interpreter** - Fast startup for development and scripting
- **Cranelift JIT** - Quick compilation for interactive use
- **LLVM JIT** - Optimized just-in-time execution
- **LLVM AOT** - Ahead-of-time compilation to native binaries

## Installation

### Via Cargo (Recommended)

```bash
cargo install sigil-parser
```

### Via Homebrew (macOS/Linux)

```bash
brew tap daemoniorum/sigil https://github.com/Daemoniorum-LLC/sigil-lang
brew install sigil
```

### Via npm (MCP Server for AI)

```bash
npm install -g @daemoniorum/sigil-mcp
```

### From Source

```bash
git clone https://github.com/Daemoniorum-LLC/sigil-lang.git
cd sigil-lang/parser
cargo build --release
```

## Quick Start

```bash
# Run a program (interpreter)
sigil run hello.sg

# Type check
sigil check hello.sg

# Interactive REPL
sigil repl

# JIT compile (Cranelift)
sigil jit program.sg

# Native compile (LLVM)
sigil compile program.sg -o program
./program

# With CUDA support
sigil compile program.sg -o program --cuda
```

## Building with LLVM Backend

```bash
# Install LLVM 18 development headers
apt install llvm-18-dev libpolly-18-dev libzstd-dev clang-18

# Build with LLVM support
CC=clang-18 cargo build --release --features llvm

# Compile to native binary
./target/release/sigil compile program.sg -o program
./program

# Or with Link-Time Optimization
./target/release/sigil compile program.sg -o program --lto
```

## Hello World

```sigil
λ main() {
    println("Hello, Sigil!")
}
```

## Core Features

### Morpheme Operators

Pipeline syntax for data transformation:

```sigil
≔ result = data
    |tau{_ * 2}       // Map: double each element
    |phi{_ > 10}      // Filter: keep if > 10
    |sigma            // Sort ascending
    |rho+             // Reduce: sum all
```

### Evidentiality Types

Track data provenance at the type level:

```sigil
≔ computed! = 1 + 1          // Known: verified truth
≔ found? = map·get(key)      // Uncertain: may be absent
≔ data~ = api·fetch(url)     // Reported: external, untrusted
```

The type system forces explicit handling of trust boundaries.

### SGDOC Documentation

Evidential documentation with epistemic markers:

```sigil
//! Verified: This function is thoroughly tested
//~ Experimental: API may change
//? Uncertain: Needs review
//◊ Predicted: Based on patterns
//‽ Contested: Multiple valid interpretations

λ calculate(x: i32) → i32 {
    x * 2
}
```

### Graphics & Physics Primitives

```sigil
≔ pos = vec3(1.0, 2.0, 3.0)
≔ rot = quat_from_axis_angle(vec3(0, 1, 0), 0.5)
≔ transformed = quat_rotate(rot, pos)

≔ force = spring_force(p1, p2, rest_length, stiffness)
≔ next_pos = verlet_integrate(pos, prev_pos, accel, dt)
```

### Automatic Differentiation

```sigil
λ f(x) { x * x }
≔ derivative = grad(f, 3.0)        // 6.0
≔ j = jacobian(multi_fn, [x, y])   // Jacobian matrix
≔ h = hessian(f, [x, y])           // Hessian matrix
```

### Entity Component System

```sigil
≔ world = ecs_world()
≔ entity = ecs_spawn(world)
ecs_attach(entity, "Position", pos)
ecs_attach(entity, "Velocity", vel)
≔ movables = ecs_query(world, "Position", "Velocity")
```

## Compilation Modes

| Command | Description | Use Case |
|---------|-------------|----------|
| `sigil run file.sg` | Interpreted | Development, debugging |
| `sigil jit file.sg` | Cranelift JIT | Fast iteration |
| `sigil llvm file.sg` | LLVM JIT | Optimized execution |
| `sigil compile file.sg -o out` | LLVM AOT | Production deployment |
| `sigil compile file.sg -o out --lto` | LLVM AOT+LTO | Maximum optimization |

## Project Structure

```
sigil-lang/
├── parser/              # Core compiler and runtime
│   ├── src/
│   │   ├── main.rs      # CLI entry point
│   │   ├── codegen.rs   # Cranelift JIT backend
│   │   ├── llvm_codegen.rs  # LLVM backend
│   │   ├── interpreter.rs   # Tree-walking interpreter
│   │   └── stdlib.rs    # Standard library
│   ├── runtime/         # Native runtime (zero C dependency)
│   └── tests/           # Test suite
├── jormungandr/         # Self-hosted compiler (Sigil-in-Sigil)
├── docs/                # Language specification
├── tools/
│   ├── oracle/          # LSP server
│   └── glyph/           # Code formatter
├── editor/vscode/       # VS Code extension
└── examples/            # Example programs
```

## Testing

```bash
cd jormungandr/tests
./run_tests_rust.sh      # Run all 577 tests (100% pass rate)
./run_tests_rust.sh --priority P0  # Run P0 tests only
```

## Documentation

- [Getting Started](sigil-lang/docs/GETTING_STARTED.md) - Tutorial and examples
- [Language Specification](sigil-lang/docs/specs/) - Complete language spec
- [Benchmark Report](sigil-lang/BENCHMARK_REPORT.md) - Performance analysis
- [Symbol Reference](sigil-lang/docs/SYMBOLS.md) - Unicode operators
- [SGDOC Spec](sigil-lang/docs/specs/24-SGDOC.md) - Documentation system

## Requirements

### Basic Build (Cranelift JIT)
- Rust 1.85+

### LLVM Backend (Production Performance)
- LLVM 18
- Clang 18
- libzstd-dev
- libpolly-18-dev

## License

Dual-licensed under MIT and Apache 2.0.

Copyright (c) 2025-2026 Daemoniorum, LLC

---

*"The void is not empty - it is full of potential."*
