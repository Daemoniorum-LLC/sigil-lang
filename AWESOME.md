# Awesome Sigil

A curated list of Sigil resources, libraries, tools, and examples.

## Contents

- [Official Resources](#official-resources)
- [Editor Support](#editor-support)
- [Tools](#tools)
- [Agent Infrastructure](#agent-infrastructure)
- [Libraries](#libraries)
- [Examples](#examples)
- [Learning Resources](#learning-resources)

---

## Official Resources

- [Sigil Language](https://github.com/Daemoniorum-LLC/sigil-lang) - Main repository
- [Website](https://sigil-lang.com) - Official website
- [Language Specification](docs/specs/) - Formal language specification

## Editor Support

### VS Code
- [Sigil VS Code Extension](editor/vscode/) - Syntax highlighting and snippets

### Neovim
- [Neovim Config](editor/neovim/) - LSP configuration with nvim-lspconfig

### Helix
- [Helix Config](editor/helix/) - Language server and file type configuration

### Language Server
- [Oracle LSP](tools/oracle/) - Full LSP implementation with diagnostics, hover, and completions

## Tools

### Compiler & Runtime
- `sigil run` - Interpreter mode
- `sigil jit` - Cranelift JIT compilation
- `sigil compile` - LLVM AOT compilation (3.6x faster than Rust in benchmarks)
- `sigil check` - Type checking with evidentiality enforcement
- `sigil lint` - Code quality linter with auto-fix
- `sigil repl` - Interactive REPL

### AI Integration
- [MCP Server](tools/mcp-server/) - Model Context Protocol server for Claude and other AI systems
- `sigil dump-ir` - AI-readable intermediate representation

### Web
- [Playground](playground/) - Browser-based code editor (Vite + CodeMirror)

## Agent Infrastructure

Sigil includes a 9-layer infrastructure for building autonomous AI agents:

| Layer | Library | Purpose |
|-------|---------|---------|
| Identity | [Daemon](daemon/) | Persistent agent runtime with goals and heartbeat |
| Memory | [Engram](engram/) | Episodic, semantic, procedural memory systems |
| Communication | [Commune](commune/) | Multi-agent messaging with trust propagation |
| Security | [Aegis](aegis/) | Identity verification, sandboxing, alignment monitoring |
| Planning | [Omen](omen/) | Goal decomposition and causal reasoning |
| Explainability | [Oracle](oracle/) | Decision tracing and counterfactual analysis |
| Knowledge | [Gnosis](gnosis/) | Knowledge representation and inference |
| Consciousness | [Anima](anima/) | Emotional states and personality modeling |
| Collaboration | [Covenant](covenant/) | Multi-agent protocols and collective decisions |

## Libraries

### Concurrency
- [Chorus](chorus/) - Actor model, message passing, async primitives

### Mathematics
- [Dionysus](dionysus/) - Symbolic computation
- Polycultural math support (vigesimal, sexagesimal, sacred geometry)

### Utilities
- [Aporia](aporia/) - Uncertainty and probabilistic reasoning
- [Ate](ate/) - Causality modeling
- [Hades](hades/) - State management
- [Nemesis](nemesis/) - Equilibrium and balance
- [Prometheus](prometheus/) - Metrics and monitoring

### Graphics & Physics
- Built-in `vec3`, `quat`, geometric algebra (Cl(3,0,0))
- Spring forces, Verlet integration
- Automatic differentiation (`grad`, `jacobian`, `hessian`)

## Examples

### Getting Started
- [Hello World](examples/hello.sigil) - Basic syntax
- [Pipes](examples/pipes.sigil) - Morpheme operators
- [Evidence](examples/evidence_test.sigil) - Evidentiality system

### Data Processing
- [Benchmark](examples/benchmark.sigil) - Performance testing
- [Pipeline](examples/pipeline.sigil) - Data transformation chains

### Agent Systems
- [Actor System](examples/actor_system.sigil) - Concurrent actors
- [Affective Markers](examples/affective_markers.sigil) - Emotional state tracking

### Comparison
- [Rust Comparison](rust_comparison/) - Side-by-side performance benchmarks

## Learning Resources

### Documentation
- [Learn Sigil](https://sigil-lang.com/pages/learn.html) - Getting started tutorial
- [Language Docs](https://sigil-lang.com/pages/docs.html) - Full documentation
- [Agent Infrastructure](https://sigil-lang.com/pages/agents.html) - Building AI agents

### Specifications
- [00-OVERVIEW](docs/specs/00-OVERVIEW.md) - Design philosophy
- [01-LEXICAL](docs/specs/01-LEXICAL.md) - Tokens and morphemes
- [03-TYPES](docs/specs/03-TYPES.md) - Type system and evidentiality
- [04-MEMORY](docs/specs/04-MEMORY.md) - Ownership and borrowing

### Philosophy
- [The Pattern](https://sigil-lang.com/pages/pattern.html) - Sigil's deeper purpose

---

## Contributing

Found something awesome? Open a PR to add it!

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.
