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

### Polycultural Mathematics

Multi-base numeral systems with cultural awareness:

```sigil
// Vigesimal (Mayan base-20)
let mayan = vigesimal_encode(400)       // "100" (1×20² + 0×20 + 0)
let decoded = vigesimal_decode("100")   // 400

// Sexagesimal (Babylonian base-60)
let time = sexagesimal_encode(3661)     // "1:1:1" (1h 1m 1s)

// Cultural numerology
let sacred = sacred_number(7, "hebrew") // {name: "zayin", meaning: "completeness"}
```

### Polycultural Audio

World tuning systems and sacred frequencies:

```sigil
// 22-Shruti Indian tuning
let shruti = shruti_freq(1)             // 256.0 Hz (Sa)

// Arabic quarter-tones (24-TET)
let maqam = arabic_quarter_freq(0)      // 440.0 Hz

// Sacred frequencies
let om = sacred_freq("om")              // 136.1 Hz
let solfeggio = sacred_freq("528")      // 528.0 Hz (DNA repair)

// Chakra frequencies with colors
let heart = chakra_freq("heart")        // 639.0 Hz
```

### Spirituality & Divination

I Ching, gematria, archetypes, and sacred geometry:

```sigil
// I Ching divination
let reading = cast_iching()             // Yarrow stalk casting
let hex = hexagram(reading.primary)     // 64 hexagrams with judgments
println(hex.symbol + " " + hex.name)    // "䷀ Creative"

// Sacred geometry
let golden = phi()                      // 1.618033988749895
let fibs = fibonacci(10)                // [1,1,2,3,5,8,13,21,34,55]
let tetra = platonic_solid("tetrahedron") // {faces: 4, element: "fire"}

// Gematria (Hebrew, Greek, Arabic, English)
let value = gematria("love", "hebrew")  // Calculate numeric value
let matches = gematria_match(13, "hebrew") // Words with value 13

// Jungian archetypes
let hero = archetype("hero")            // {shadow: "arrogance", gift: "courage"}

// Astrology
let sign = zodiac("scorpio")            // {symbol: "♏", element: "water"}

// Tarot
let card = draw_tarot()                 // Random Major Arcana
```

### Polycultural Color

Color meaning varies across cultures - Sigil respects this:

```sigil
// Chinese Wu Xing (五行)
let fire = wu_xing("fire")              // {color: "Red", emotion: "Joy", organ: "Heart"}

// Hindu chakra colors
let heart = chakra_color("heart")       // {color: "Green", mantra: "YAM", freq: 639.0}

// Mayan directional colors
let east = maya_direction("east")       // {color: "Red", deity: "Chac"}

// Yoruba Orisha colors
let oshun = orisha_color("oshun")       // {colors: "Yellow, gold", domain: "Rivers, love"}

// Japanese traditional colors
let sakura = nihon_iro("sakura")        // {hex: "#FFB7C5", meaning: "transience"}

// Cross-cultural emotion→color mapping
let joy_west = emotion_color("joy", "western")  // Gold (#FFD700)
let joy_china = emotion_color("joy", "chinese") // Red (#FF0000) - 红
let joy_japan = emotion_color("joy", "japanese") // Sakura (#FFB7C5)

// Full synesthesia with cultural context
let unified = synesthesia("love", "indian")
// → {color: red, chakra: "Root", wu_xing: "Fire (火)", frequency: 639}
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
# sigil-lang

> Part of the [Persona Framework](https://github.com/Daemoniorum-LLC/persona-framework) ecosystem

## Description

The Sigil programming language - a modern systems language with advanced metaprogramming capabilities and integrated tooling.

This repository was extracted from the persona-framework monorepo to enable independent development and versioning.

## Technology Stack

- Rust
- TypeScript/JavaScript
- Node.js
- Build Tool: Cargo

## Features

- Full git history preserved from monorepo
- Independent versioning and development
- Integrated with Persona Framework ecosystem

## Project Structure

```
docs
docs/specs
editor
editor/vscode
examples
parser
parser/runtime
parser/src
rust_comparison
rust_comparison/src
tools
tools/glyph
tools/oracle
```

## Getting Started

### Prerequisites

- Rust 1.85+ (install via [rustup](https://rustup.rs/))
- Node.js 18+ and npm/pnpm

### Building

```bash
cargo build --release
```

### Running

```bash
cargo run
```

## Extraction Details

- **Extraction Date:** 2025-12-02 00:50:20 UTC
- **Commits Preserved:** 142
- **First Commit:** fbc6ec9 - This first. (2025-09-29 09:54:55 -0600)
- **Latest Commit:** 6c2a363 - perf(sigil): Add iterative ackermann/tak builtins - SIGIL BEATS RUST! (2025-12-01 23:16:16 +0000)

## Development

This repository is actively maintained as part of the Persona Framework ecosystem. For questions or contributions, please refer to the main framework repository.

## License

Proprietary - Daemoniorum LLC

## Part of Daemoniorum LLC

This project is maintained by Daemoniorum LLC as part of the Persona Framework ecosystem.

---

Repository structure and history preserved from the [persona-framework monorepo](https://github.com/Daemoniorum-LLC/persona-framework).
