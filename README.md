# Sigil Programming Language

A polysynthetic programming language with evidentiality types, morpheme operators, and native performance through LLVM.

## Performance

Sigil compiles to native code that **outperforms hand-written Rust**:

| Backend | Time | vs Interpreter | vs Rust |
|---------|------|----------------|---------|
| Interpreter | 39.4s | 1x | 985x slower |
| Cranelift JIT | 0.59s | 67x faster | 13x slower |
| LLVM JIT | 0.62s | 64x faster | 14x slower |
| **LLVM AOT** | **0.011s** | **3,582x faster** | **3.6x FASTER** |

## Quick Start

```bash
# Clone and build
git clone https://github.com/daemoniorum/sigil.git
cd sigil/parser
cargo build --release

# Run interpreted (for development)
./target/release/sigil run hello.sigil

# Run with Cranelift JIT (fast)
./target/release/sigil jit program.sigil

# Compile to native binary (fastest)
./target/release/sigil compile program.sigil -o program
./program
```

## Building for Best Performance

### Option 1: Cranelift JIT (Default)

```bash
cargo build --release
./target/release/sigil jit program.sigil
```

- 67x faster than interpreter
- No external dependencies
- Good for development and testing

### Option 2: LLVM Backend (Production)

```bash
# Install LLVM 18 development headers
apt install llvm-18-dev libpolly-18-dev libzstd-dev clang-18

# Build with LLVM support
CC=clang-18 cargo build --release --features llvm

# Run with LLVM JIT
./target/release/sigil llvm program.sigil

# Or compile to native binary (fastest)
./target/release/sigil compile program.sigil -o program
./program
```

- 3,582x faster than interpreter
- Produces standalone native binaries
- **3.6x faster than equivalent Rust code**

### Option 3: LLVM with LTO (Maximum Performance)

```bash
./target/release/sigil compile program.sigil -o program --lto
```

Link-Time Optimization enables cross-module optimization between your Sigil code and the runtime.

## Hello World

```sigil
fn main() {
    println("Hello, Sigil!")
}
```

## Core Features

### Morpheme Operators

Transform data with elegant pipeline syntax:

```sigil
let result = data
    |tau{_ * 2}       // Map: double each element
    |phi{_ > 10}      // Filter: keep if > 10
    |sigma            // Sort ascending
    |rho+             // Reduce: sum all
```

### Evidentiality Types

Track data provenance at the type level:

```sigil
let computed! = 1 + 1          // Known: verified truth
let found? = map.get(key)       // Uncertain: may be absent
let data~ = api.fetch(url)      // Reported: external, untrusted
```

### Graphics & Physics Primitives

Native support for game/graphics development:

```sigil
let pos = vec3(1.0, 2.0, 3.0)
let rot = quat_from_axis_angle(vec3(0, 1, 0), 0.5)
let transformed = quat_rotate(rot, pos)

let force = spring_force(p1, p2, rest_length, stiffness)
let next_pos = verlet_integrate(pos, prev_pos, accel, dt)
```

### Geometric Algebra

Full Cl(3,0,0) multivector support:

```sigil
let mv = mv_new(1.0, 2.0, 3.0, 4.0, 0.5, 0.6, 0.7, 0.1)
let rotor = rotor_from_axis_angle(vec3(0, 1, 0), 0.5)
let rotated = rotor_apply(rotor, vec3(1, 0, 0))
```

### Automatic Differentiation

```sigil
fn f(x) { return x * x }
let derivative = grad(f, 3.0)        // 6.0
let j = jacobian(multi_fn, [x, y])   // Jacobian matrix
let h = hessian(f, [x, y])           // Hessian matrix
```

### Entity Component System

```sigil
let world = ecs_world()
let entity = ecs_spawn(world)
ecs_attach(entity, "Position", pos)
ecs_attach(entity, "Velocity", vel)
let movables = ecs_query(world, "Position", "Velocity")
```

## Compilation Modes

| Command | Description | Performance |
|---------|-------------|-------------|
| `sigil run file.sigil` | Interpreted | Development, debugging |
| `sigil jit file.sigil` | Cranelift JIT | Fast iteration |
| `sigil llvm file.sigil` | LLVM JIT | Near-native |
| `sigil compile file.sigil -o out` | LLVM AOT | **Production** |
| `sigil compile file.sigil -o out --lto` | LLVM AOT+LTO | **Maximum** |

## Project Structure

```
sigil/
├── parser/              # Core compiler and runtime
│   ├── src/
│   │   ├── main.rs      # CLI entry point
│   │   ├── codegen.rs   # Cranelift JIT backend
│   │   ├── llvm_codegen.rs  # LLVM backend
│   │   ├── interpreter.rs   # Tree-walking interpreter
│   │   └── stdlib.rs    # Standard library
│   ├── runtime/         # C runtime for AOT binaries
│   └── tests/           # Test suite (244 tests)
├── rust_comparison/     # Benchmarks vs Rust
├── docs/                # Language specification
└── tools/               # LSP server, formatter
```

## Testing

```bash
cd parser
cargo test              # Run all 244 tests
cargo test --release    # Optimized test run
```

## Documentation

- [Getting Started](docs/GETTING_STARTED.md) - Tutorial and examples
- [Language Specification](docs/specs/) - Complete language spec
- [Benchmark Report](BENCHMARK_REPORT.md) - Detailed performance analysis
- [Symbol Reference](docs/SYMBOLS.md) - Unicode operators

## Requirements

- Rust 1.70+
- For LLVM backend:
  - LLVM 18
  - Clang 18
  - libzstd-dev
  - libpolly-18-dev

## License

MIT License - Daemoniorum, Inc.
