# Changelog

All notable changes to Sigil will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed

- **`sigil migrate` no longer defaults to overwriting its input** (#92). The
  command now requires one of three destinations, and refuses with exit code 2
  if given none:
  - `-o <dir>` / `--output <dir>` — write the migrated tree to `<dir>`,
    mirroring the input structure and converting `.rs` to `.sg`. The input is
    left byte-for-byte alone. This is the recommended mode.
  - `--dry-run` (alias `--diff`) — report what would change and write nothing
    anywhere. `.rs` files are in scope, so this doubles as a read-only pass for
    the reserved-name diagnostic over a Rust crate.
  - `--in-place` — the old behaviour, kept but now opt-in.

  `--in-place` cannot be combined with `-o` or `--dry-run`. `--backup` applies
  to `--in-place`. `--workspace` accepts `--dry-run` or `--in-place`.

  Previously `sigil migrate <path>` rewrote `<path>`, so the destructive mode
  was the one you got by typing the least, with no backup and no preview.

## [0.4.0] - 2026-01-25

### Added

#### Native Syntax Migration
- **Native symbol vocabulary**: Full transition to Sigil-native syntax
  - `λ` for function declarations (replaces `fn`)
  - `Σ` for struct declarations (replaces `struct`)
  - `≔` for variable binding (replaces `let`)
  - `·` (middledot) for module paths (replaces `::`)
  - `&Δ` for mutable references (replaces `&mut`)
  - `∀` for universal quantification in for loops
  - `∞` for infinite loops
  - `⊗` for break, `↻` for continue
- **Syntax migration tool**: `sigil migrate` command to upgrade legacy code
- **Helpful error messages**: Detects Rust syntax and suggests native equivalents

#### Native Runtime (Self-Hosting)
- **Zero C dependency runtime**: Pure Sigil/assembly runtime replacing C runtime
  - Platform syscalls for Linux x64, ARM64, macOS Intel/ARM, Windows
  - Arena allocator with O(1) allocation
  - SIMD vector math (SSE/AVX/AVX-512)
  - Native file I/O, networking, and async support
- **Threading and async I/O** primitives
- **22 LLVM math intrinsics** exposed to interpreter

#### Tooling
- **LSP server**: Full Language Server Protocol support with diagnostics
- **Code formatter**: `sigil fmt` with configurable style
- **Linter**: `sigil lint` with evidentiality correctness checking
- **Package manager**: `tome` for dependency management
- **SGDOC**: Documentation extraction with evidential doc comments
  - `sigil doc-extract` CLI command
  - JSON, Markdown, and HTML output formats
  - 5 evidentiality markers: `//!` (verified), `//~` (reported), `//?` (uncertain), `//◊` (predicted), `//‽` (paradox)

#### Compute Backends
- **SIMD backend**: AVX-512 F32x16 vector operations
- **CUDA backend**: GPU compute via CUDA Driver API (`--cuda` flag)
- **Link-time optimization**: `--lto` flag for maximum optimization

#### Module System
- **Tome module resolution**: `invoke tome·` statements for cross-module linking
- **Circular dependency detection**
- **Symbol export/import tracking**

#### Protocol Support (v0.4.0 Stable)
- **HTTP client**: Production-ready with connection pooling, TLS
- **WebSocket client**: Full duplex communication support

### Changed
- **Ecosystem libraries**: 157 files migrated to native Sigil syntax
- **Test suite expanded**: 509/596 tests passing (85% pass rate)
- **Self-hosted compiler**: Jormungandr now functional for bootstrap experiments

### Fixed
- Type system: Proper generic type validation, Option/Result mismatch detection
- Parser: Middledot path syntax, attribute paths, bracket generics
- Memory: Reborrow semantics, Box<T> deref, slice borrowing
- Runtime: Stack warning suppression, proper assembly integration

### Deprecated
- Rust-style syntax (`fn`, `let`, `struct`, `::`) - use native symbols

## [0.3.0] - 2025-12-02

### Added
- LLVM AOT compilation backend
- Cranelift JIT compilation
- Complete standard library with Rc<T>, Cell<T>, Drop trait
- Mutable reference sync-back mechanism
- WASM compilation support

### Changed
- Extracted from persona-framework monorepo
- 142 commits preserved from original history

## [0.2.1] - 2025-11-15

### Added
- Initial public release
- Tree-walking interpreter
- Basic type system with evidentiality markers

---

[0.4.0]: https://github.com/Daemoniorum-LLC/sigil-lang/compare/v0.3.0...v0.4.0
[0.3.0]: https://github.com/Daemoniorum-LLC/sigil-lang/compare/v0.2.1...v0.3.0
[0.2.1]: https://github.com/Daemoniorum-LLC/sigil-lang/releases/tag/v0.2.1
