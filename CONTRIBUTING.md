# Contributing to Sigil

Thank you for your interest in contributing to Sigil! This document provides guidelines for contributing to the project.

## Getting Started

1. **Fork the repository** and clone it locally
2. **Build the compiler:**
   ```bash
   cd parser
   cargo build --release
   ```
3. **Run the test suite:**
   ```bash
   cd ../jormungandr/tests
   ./run_tests_rust.sh
   ```

## Ways to Contribute

### Reporting Bugs

- Search existing issues to avoid duplicates
- Use the bug report template
- Include: Sigil version, OS, minimal reproduction case, expected vs actual behavior

### Suggesting Features

- Open a discussion or issue describing the feature
- Explain the use case and why it benefits Sigil users
- For language features, consider how they interact with evidentiality markers

### Code Contributions

1. **Pick an issue** labeled `good first issue` or `help wanted`
2. **Comment on the issue** to let others know you're working on it
3. **Create a branch** from `main`
4. **Make your changes** following the code style below
5. **Add tests** for new functionality
6. **Submit a pull request**

## Code Style

### Rust (Compiler)

- Run `cargo fmt` before committing
- Run `cargo clippy` and address warnings
- Follow existing patterns in the codebase
- Document public APIs with doc comments

### Sigil (Standard Library / Examples)

- Use evidentiality markers appropriately:
  - `!` for values computed locally or verified
  - `?` for optional/nullable values
  - `~` for data from external sources
  - `‽` for explicit trust boundaries
- Prefer morpheme pipelines for data transformations
- Keep functions focused and composable

## Pull Request Process

1. **Update documentation** if your change affects user-facing behavior
2. **Add tests** covering your changes
3. **Ensure CI passes** - all tests must pass
4. **Request review** from a maintainer
5. **Address feedback** promptly

### Commit Messages

Use conventional commit format:

```
type(scope): description

[optional body]

[optional footer]
```

Types: `feat`, `fix`, `docs`, `style`, `refactor`, `perf`, `test`, `build`, `ci`, `chore`

Examples:
- `feat(parser): add support for async/await syntax`
- `fix(interpreter): correct string interpolation escaping`
- `docs: update installation instructions`

## Development Setup

### Prerequisites

- Rust 1.75+ (stable)
- LLVM 18 (for AOT compilation, optional)

### Building

```bash
# Debug build
cd parser && cargo build

# Release build
cd parser && cargo build --release

# With LLVM backend
cd parser && cargo build --release --features llvm
```

### Testing

```bash
# Run all tests
cd jormungandr/tests && ./run_tests_rust.sh

# Run specific test section
./run_tests_rust.sh --spec 03_types

# Run compiler unit tests
cd parser && cargo test
```

## Architecture Overview

```
parser/
├── src/
│   ├── lexer.rs       # Tokenization
│   ├── parser.rs      # AST construction
│   ├── typeck.rs      # Type checking
│   ├── interpreter.rs # Direct execution
│   ├── codegen.rs     # Cranelift JIT backend
│   ├── llvm_codegen.rs# LLVM AOT backend
│   └── stdlib.rs      # Standard library
```

## Questions?

- Open a [Discussion](https://github.com/Daemoniorum-LLC/sigil-lang/discussions)
- Check the [documentation](https://sigil-lang.com/pages/docs.html)

## License

By contributing, you agree that your contributions will be licensed under the same license as the project.
