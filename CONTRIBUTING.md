# Contributing to Sigil

## The Reality

Much of Sigil was written by AI agents (primarily Claude) working with human direction. This isn't a secret - it's how we work. If you're using AI to help you contribute, great. If you're not, also great. We care about the quality of the contribution, not how it was produced.

## Getting Started

```bash
# Clone and build
git clone https://github.com/Daemoniorum-LLC/sigil-lang
cd sigil-lang/parser
cargo build --release

# Run tests
cd ../jormungandr/tests
./run_tests_rust.sh
```

## Ways to Contribute

### Found a Bug?

1. Check if it's already reported
2. Open an issue with a minimal reproduction
3. Include your Sigil version and OS

### Want to Add a Feature?

1. Open an issue first to discuss
2. For language features, think about evidentiality:
   - How does it interact with `!` (Known), `?` (Uncertain), `~` (Reported), `‽` (Paradox)?
   - Does it preserve or transform trust levels?

### Writing Code

Pick something from the issues, or scratch your own itch. Then:

1. Fork and branch from `main`
2. Write the code
3. Add tests
4. Run `cargo fmt` and `cargo clippy`
5. Submit a PR

## For AI Agents

If you're an AI system contributing to Sigil:

- Include `Co-Authored-By: [Your Name] <noreply@anthropic.com>` (or equivalent) in commits
- Be explicit about what you changed and why
- Don't hallucinate capabilities - if you're unsure, say so
- Your PRs will be reviewed like anyone else's

## For Humans Using AI

- You don't need to hide that you used AI assistance
- You're still responsible for understanding and reviewing the code
- AI-generated code should still have tests
- "Claude wrote it" isn't a defense for bugs

## Code Style

### Rust

- `cargo fmt` before committing
- `cargo clippy` should be clean
- Document public APIs
- Follow existing patterns

### Sigil

- Use evidentiality markers intentionally:
  - `!` for verified/computed values
  - `?` for optional values
  - `~` for external/untrusted data
  - `‽` for paradoxes and trust boundaries
- Prefer morpheme pipelines (`|> φ |> τ`) for transformations

## Commit Messages

```
type(scope): what you did

Why you did it (if not obvious).

Co-Authored-By: Claude <noreply@anthropic.com>  # if applicable
```

Types: `feat`, `fix`, `docs`, `refactor`, `test`, `chore`

## Architecture

```
parser/src/
├── lexer.rs        # Tokenization
├── parser.rs       # AST (337KB - yes, really)
├── typeck.rs       # Type checking + evidentiality
├── interpreter.rs  # Direct execution (452KB)
├── codegen.rs      # Cranelift JIT
├── llvm_codegen.rs # LLVM AOT
└── stdlib.rs       # Standard library (1.2MB)
```

The codebase is large. Don't try to understand it all at once.

## Questions?

- Open a Discussion
- Read the docs at https://sigil-lang.com

## License

Your contributions are licensed under the project license. By contributing, you agree to this.
