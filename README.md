# Sigil Programming Language

A polysynthetic programming language with evidentiality types, morpheme operators, and native performance through LLVM. Now with comprehensive **Agent Infrastructure** for building intelligent, self-aware systems.

## v0.4.0 - Native Syntax & Self-Hosting

This release completes the transition to native Sigil syntax and introduces self-hosting capabilities:

- **🛡️ Aegis** - Security & safety framework
- **✨ Anima** - Soul & consciousness modeling
- **🔗 Commune** - Multi-agent communication
- **🤝 Covenant** - Collaborative protocols
- **⚙️ Daemon** - Background processes
- **🧠 Gnosis** - Knowledge & learning
- **🔮 Omen** - Planning & prediction
- **👁️ Oracle** - Explainable decisions
- **💾 Engram** - Advanced memory system (episodic, semantic, procedural)

## Execution Backends

Sigil offers flexible execution options for different use cases:

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

### JVM / Kotlin Multiplatform

For JVM, Android, iOS, and other Kotlin targets:

```kotlin
// build.gradle.kts
repositories {
    maven("https://jitpack.io")
}

dependencies {
    implementation("com.github.Daemoniorum-LLC.sigil-lang:sigil-kmp:0.1.0")
}
```

Or via Maven Central (once published):

```kotlin
dependencies {
    implementation("com.daemoniorum:sigil-kmp:0.1.0")
}
```

### From Source

```bash
git clone https://github.com/Daemoniorum-LLC/sigil-lang.git
cd sigil-lang/parser
cargo build --release
```

## Quick Start

```bash
# Run a program
sigil run hello.sigil

# Type check
sigil check hello.sigil

# Interactive REPL
sigil repl

# JIT compile (faster)
sigil jit program.sigil

# Native compile (fastest)
sigil compile program.sigil -o program
./program
```

## Building with LLVM Backend

For production performance, build with LLVM support:

```bash
# Install LLVM 18 development headers
apt install llvm-18-dev libpolly-18-dev libzstd-dev clang-18

# Build with LLVM support
CC=clang-18 cargo build --release --features llvm

# Compile to native binary
./target/release/sigil compile program.sigil -o program
./program

# Or with Link-Time Optimization
./target/release/sigil compile program.sigil -o program --lto
```

## Hello World

```sigil
λ main() {
    println("Hello, Sigil!")
}
```

## Core Features

### Morpheme Operators

Transform data with elegant pipeline syntax:

```sigil
≔ result = data
    |tau{_ * 2}       // Map: double each element
    |phi{_ > 10}      // Filter: keep if > 10
    |sigma            // Sort ascending
    |rho+             // Reduce: sum all
```

Provides significant code reduction in real-world applications.

### Evidentiality Types

Track data provenance at the type level:

```sigil
≔ computed! = 1 + 1          // Known: verified truth
≔ found? = map.get(key)       // Uncertain: may be absent
≔ data~ = api.fetch(url)      // Reported: external, untrusted
```

The type system forces explicit handling of trust boundaries, preventing entire classes of security bugs.

### Graphics & Physics Primitives

Native support for game/graphics development:

```sigil
≔ pos = vec3(1.0, 2.0, 3.0)
≔ rot = quat_from_axis_angle(vec3(0, 1, 0), 0.5)
≔ transformed = quat_rotate(rot, pos)

≔ force = spring_force(p1, p2, rest_length, stiffness)
≔ next_pos = verlet_integrate(pos, prev_pos, accel, dt)
```

### Geometric Algebra

Full Cl(3,0,0) multivector support:

```sigil
≔ mv = mv_new(1.0, 2.0, 3.0, 4.0, 0.5, 0.6, 0.7, 0.1)
≔ rotor = rotor_from_axis_angle(vec3(0, 1, 0), 0.5)
≔ rotated = rotor_apply(rotor, vec3(1, 0, 0))
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

### Polycultural Mathematics

Multi-base numeral systems with cultural awareness:

```sigil
// Vigesimal (Mayan base-20)
≔ mayan = vigesimal_encode(400)       // "100" (1×20² + 0×20 + 0)
≔ decoded = vigesimal_decode("100")   // 400

// Sexagesimal (Babylonian base-60)
≔ time = sexagesimal_encode(3661)     // "1:1:1" (1h 1m 1s)

// Cultural numerology
≔ sacred = sacred_number(7, "hebrew") // {name: "zayin", meaning: "completeness"}
```

### Polycultural Audio

World tuning systems and sacred frequencies:

```sigil
// 22-Shruti Indian tuning
≔ shruti = shruti_freq(1)             // 256.0 Hz (Sa)

// Arabic quarter-tones (24-TET)
≔ maqam = arabic_quarter_freq(0)      // 440.0 Hz

// Sacred frequencies
≔ om = sacred_freq("om")              // 136.1 Hz
≔ solfeggio = sacred_freq("528")      // 528.0 Hz (DNA repair)

// Chakra frequencies with colors
≔ heart = chakra_freq("heart")        // 639.0 Hz
```

### Spirituality & Divination

I Ching, gematria, archetypes, and sacred geometry:

```sigil
// I Ching divination
≔ reading = cast_iching()             // Yarrow stalk casting
≔ hex = hexagram(reading.primary)     // 64 hexagrams with judgments
println(hex.symbol + " " + hex.name)  // "䷀ Creative"

// Sacred geometry
≔ golden = phi()                      // 1.618033988749895
≔ fibs = fibonacci(10)                // [1,1,2,3,5,8,13,21,34,55]
≔ tetra = platonic_solid("tetrahedron") // {faces: 4, element: "fire"}

// Gematria (Hebrew, Greek, Arabic, English)
≔ value = gematria("love", "hebrew")  // Calculate numeric value
≔ matches = gematria_match(13, "hebrew") // Words with value 13

// Jungian archetypes
≔ hero = archetype("hero")            // {shadow: "arrogance", gift: "courage"}

// Astrology
≔ sign = zodiac("scorpio")            // {symbol: "♏", element: "water"}

// Tarot
≔ card = draw_tarot()                 // Random Major Arcana
```

### Polycultural Color

Color meaning varies across cultures - Sigil respects this:

```sigil
// Chinese Wu Xing (五行)
≔ fire = wu_xing("fire")              // {color: "Red", emotion: "Joy", organ: "Heart"}

// Hindu chakra colors
≔ heart = chakra_color("heart")       // {color: "Green", mantra: "YAM", freq: 639.0}

// Mayan directional colors
≔ east = maya_direction("east")       // {color: "Red", deity: "Chac"}

// Yoruba Orisha colors
≔ oshun = orisha_color("oshun")       // {colors: "Yellow, gold", domain: "Rivers, love"}

// Japanese traditional colors
≔ sakura = nihon_iro("sakura")        // {hex: "#FFB7C5", meaning: "transience"}

// Cross-cultural emotion→color mapping
≔ joy_west = emotion_color("joy", "western")  // Gold (#FFD700)
≔ joy_china = emotion_color("joy", "chinese") // Red (#FF0000) - 红
≔ joy_japan = emotion_color("joy", "japanese") // Sakura (#FFB7C5)

// Full synesthesia with cultural context
≔ unified = synesthesia("love", "indian")
// → {color: red, chakra: "Root", wu_xing: "Fire (火)", frequency: 639}
```

## Compilation Modes

| Command | Description | Use Case |
|---------|-------------|----------|
| `sigil run file.sigil` | Interpreted | Development, debugging |
| `sigil jit file.sigil` | Cranelift JIT | Fast iteration |
| `sigil llvm file.sigil` | LLVM JIT | Optimized execution (requires --features llvm) |
| `sigil compile file.sigil -o out` | LLVM AOT | Production deployment |
| `sigil compile file.sigil -o out --lto` | LLVM AOT+LTO | Maximum optimization |

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
│   ├── runtime/         # C runtime for AOT binaries
│   └── tests/           # Test suite (596 tests, 85% passing)
├── rust_comparison/     # Benchmarks vs Rust
├── docs/                # Language specification
│   ├── GETTING_STARTED.md
│   └── specs/
├── tools/
│   ├── oracle/          # LSP server
│   └── glyph/           # Code formatter
├── editor/vscode/       # VS Code extension
└── examples/            # Example programs
```

## Testing

```bash
cd jormungandr/tests
./run_tests_rust.sh      # Run all tests (509/596 pass, 85%)
./run_tests_rust.sh --priority P0  # Run P0 tests only (all stable)
```

## Documentation

- [Getting Started](docs/GETTING_STARTED.md) - Tutorial and examples
- [Language Specification](docs/specs/) - Complete language spec
- [Benchmark Report](BENCHMARK_REPORT.md) - Detailed performance analysis
- [Symbol Reference](docs/SYMBOLS.md) - Unicode operators

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

Copyright (c) 2024-2025 Daemoniorum, LLC

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

---

*"Tools with Teeth" ⚔️*
