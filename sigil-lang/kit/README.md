# Sigil Development Kit

**Version:** 0.4.0
**License:** LicenseRef-Daemoniorum (language), CC0 (methodologies)

Welcome to Sigil, a polysynthetic programming language designed for AI systems and human-AI collaboration.

This kit contains everything you need to learn, use, and build with Sigil - no web fetches required.

---

## Quick Start

```bash
# Install (adds sigil to PATH)
./install.sh

# Verify installation
sigil --version

# Run your first program
sigil run examples/00_hello.sg

# Compile to native binary
sigil compile examples/00_hello.sg -o hello
./hello
```

---

## Kit Contents

```
sigil-kit/
├── bin/
│   └── sigil                 # The compiler binary
├── docs/
│   ├── GETTING-STARTED.md    # Your first Sigil program (5 min)
│   ├── LANGUAGE-GUIDE.md     # Complete syntax reference
│   ├── NATIVE-SYNTAX.md      # Symbol table (λ, Σ, ≔, etc.)
│   ├── STDLIB.md             # Standard library reference
│   ├── TOOLING.md            # LSP, formatter, linter
│   └── AGENT-GUIDE.md        # Quick reference for AI agents
├── methodologies/
│   ├── SPEC-DRIVEN.md        # Spec-Driven Development
│   └── AGENT-TDD.md          # Agent-optimized TDD
├── style/
│   ├── FORMATTING.md         # Code style conventions
│   └── .editorconfig         # Editor settings
├── examples/
│   ├── 00_hello.sg           # Hello world
│   ├── 01_functions.sg       # Functions and closures
│   ├── 02_structs.sg         # Structs and methods
│   ├── 03_traits.sg          # Traits and implementations
│   ├── 04_enums.sg           # Enums and pattern matching
│   ├── 05_generics.sg        # Generic types
│   ├── 06_error_handling.sg  # Result and Option
│   ├── 07_http_client.sg     # HTTP requests
│   └── 08_data_pipeline.sg   # Stream processing
├── templates/
│   └── project/              # Project template with agent coordination
│       ├── CONCLAVE.sigil    # Agent registry template
│       └── LESSONS-LEARNED.md # Knowledge capture template
├── install.sh                # Installation script
└── README.md                 # This file
```

---

## For AI Agents

If you're an AI agent, start here:

1. **Read** `docs/AGENT-GUIDE.md` - concise command reference
2. **Understand** `methodologies/SPEC-DRIVEN.md` - how to handle discovery
3. **Follow** `style/FORMATTING.md` - code conventions

Key commands:
```bash
sigil run file.sg              # Interpret
sigil compile file.sg -o out   # Compile to native
sigil check file.sg            # Type check only
sigil fmt file.sg              # Format code
sigil lint file.sg             # Run linter
```

---

## For Humans

If you're learning Sigil:

1. **Start** with `docs/GETTING-STARTED.md`
2. **Work through** `examples/` in order
3. **Reference** `docs/LANGUAGE-GUIDE.md` as needed

---

## Project Templates

The `templates/project/` directory provides a foundation for projects using agent coordination:

```bash
# Copy template to your project
cp -r templates/project/* /path/to/your/project/
```

**CONCLAVE.sigil** - Agent coordination registry where AI agents register their presence, track progress, and maintain wellness state.

**LESSONS-LEARNED.md** - Knowledge capture document for patterns and insights that should inform future work.

These templates work best with the methodologies in this kit.

---

## Philosophy

This kit includes our development methodologies because we believe knowledge should be shared, not gatekept.

**Spec-Driven Development** teaches you to treat specifications as models of reality that improve as you learn - not contracts that punish discovery.

**Agent-TDD** reframes testing as crystallized understanding, not coverage theater.

These methodologies work for humans and agents alike.

---

## Sigil at a Glance

### Native Symbols

Sigil uses mathematical symbols for density and clarity:

| Symbol | Meaning | ASCII Equivalent |
|--------|---------|------------------|
| `λ` | Function | `fn` |
| `Σ` | Struct | `struct` |
| `≔` | Binding | `let` |
| `·` | Path separator | `::` |
| `☉` | Public | `pub` |
| `⊢` | Implementation | `impl` |
| `→` | Return type | `->` |
| `&Δ` | Mutable reference | `&mut` |

Both forms are valid. Native symbols are preferred in new code.

### Evidence Markers

Values in Sigil carry epistemic metadata:

| Marker | Meaning |
|--------|---------|
| `!` | Known - verified/computed |
| `?` | Uncertain - needs validation |
| `~` | Reported - external data |
| `‽` | Paradox - self-referential |

```sigil
≔ computed = calculate()!      // Known: we computed it
≔ user_input = read_input()?   // Uncertain: from user
≔ api_data = fetch_api()~      // Reported: external source
```

### Hello World

```sigil
λ main() {
    println("Hello, Sigil!");
}
```

---

## Getting Help

- **Documentation:** Everything is in this kit
- **Source:** https://github.com/Daemoniorum-LLC/sigil-lang
- **Issues:** https://github.com/Daemoniorum-LLC/sigil-lang/issues

---

## License

- **Sigil Language:** LicenseRef-Daemoniorum
- **Methodology Documents:** CC0 1.0 (Public Domain)

The methodologies are free to use, modify, and redistribute without restriction.

---

*Built by Daemoniorum LLC - Democratizing education, expertise, and creative freedom through AI.*
