# Sigil Tooling Specification

> *"A practitioner is only as capable as their implements."*

## 1. The Sigil Toolchain

| Tool | Purpose | Analogy |
|------|---------|---------|
| **Incant** | Build system & package manager | Cargo |
| **Glyph** | Code formatter | rustfmt |
| **Augur** | Linter & static analyzer | clippy |
| **Oracle** | Language server (LSP) | rust-analyzer |
| **Scribe** | Documentation generator | rustdoc |
| **Crucible** | Test runner | cargo test |
| **Alembic** | Benchmark runner | criterion |
| **Vessel** | Binary packager/distributor | cargo-dist |

---

## 2. Incant (Build System & Package Manager)

### 2.1 Project Structure

```
my_project/
├── Sigil.toml          # Manifest
├── Sigil.lock          # Lockfile
├── src/
│   ├── lib.sg          # Library root
│   ├── main.sg         # Binary entry point
│   └── bin/
│       └── other.sg    # Additional binaries
├── tests/
│   └── integration.sg
├── benches/
│   └── performance.sg
├── examples/
│   └── usage.sg
└── target/
    ├── debug/
    └── release/
```

### 2.2 Sigil.toml Manifest

```toml
[project]
name = "my_project"
version = "0.1.0"
authors = ["Name <email@example.com>"]
edition = "2025"
description = "A brief description"
license = "MIT"
repository = "https://github.com/user/my_project"
documentation = "https://docs.rs/my_project"
keywords = ["sigil", "example"]
categories = ["development-tools"]

[lib]
name = "my_lib"
path = "src/lib.sg"

[[bin]]
name = "my_binary"
path = "src/main.sg"

[dependencies]
serde = "1.0"
tokio = { version = "1.0", features = ["full"] }
local_crate = { path = "../local" }
git_crate = { git = "https://github.com/user/repo", branch = "main" }

[dependencies.rust]
# Import Rust crates
arcanum = { crate = "arcanum", version = "0.1" }
infernum = { crate = "infernum", version = "0.1" }

[dev-dependencies]
proptest = "1.0"

[build-dependencies]
sigil-bindgen = "0.1"

[features]
default = ["std"]
std = []
async = ["tokio"]
full = ["std", "async", "serde"]

[profile.dev]
opt-level = 0
debug = true
overflow-checks = true

[profile.release]
opt-level = 3
debug = false
lto = "fat"
panic = "abort"
strip = true

[profile.bench]
inherits = "release"
debug = true  # For profiling symbols
```

### 2.3 Incant Commands

```bash
# Project management
incant new my_project           # Create new project
incant new --lib my_library     # Create library project
incant init                     # Initialize in current directory

# Building
incant build                    # Debug build
incant build --release          # Release build
incant build --target x86_64-unknown-linux-gnu
incant build --features "async,serde"
incant build --all-features
incant build --no-default-features

# Running
incant run                      # Build and run
incant run --release
incant run --bin other_binary
incant run --example usage
incant run -- arg1 arg2         # Pass args to binary

# Testing
incant test                     # Run all tests
incant test test_name           # Run specific test
incant test --lib               # Library tests only
incant test --doc               # Documentation tests
incant test --release           # Test with release profile
incant test -- --nocapture      # Show stdout

# Benchmarking
incant bench                    # Run benchmarks
incant bench bench_name         # Specific benchmark

# Documentation
incant doc                      # Generate docs
incant doc --open               # Generate and open in browser
incant doc --no-deps            # Skip dependencies

# Dependencies
incant add serde                # Add dependency
incant add tokio --features full
incant add --dev proptest       # Dev dependency
incant remove serde             # Remove dependency
incant update                   # Update dependencies
incant update serde             # Update specific dep
incant tree                     # Show dependency tree
incant audit                    # Security audit

# Publishing
incant login                    # Login to registry
incant publish                  # Publish to registry
incant publish --dry-run        # Verify without publishing
incant yank --version 0.1.0     # Yank a version

# Utilities
incant check                    # Type check without building
incant clean                    # Remove target directory
incant fmt                      # Run glyph formatter
incant lint                     # Run augur linter
incant fix                      # Auto-fix lint issues
```

### 2.4 Workspaces

```toml
# Root Sigil.toml
[workspace]
members = [
    "crates/core",
    "crates/utils",
    "crates/cli",
]
resolver = "2"

[workspace.dependencies]
# Shared dependencies
serde = "1.0"
tokio = "1.0"

[workspace.package]
# Shared metadata
edition = "2025"
license = "MIT"
repository = "https://github.com/org/workspace"
```

```bash
# Workspace commands
incant build --workspace        # Build all members
incant test --workspace         # Test all members
incant build -p core            # Build specific member
```

---

## 3. Glyph (Code Formatter)

### 3.1 Usage

```bash
# Format files
glyph                           # Format current project
glyph src/main.sg               # Format specific file
glyph --check                   # Check without modifying
glyph --diff                    # Show diff

# Via incant
incant fmt                      # Same as glyph
incant fmt --check              # CI check mode
```

### 3.2 Configuration (Glyph.toml)

```toml
# Glyph.toml
edition = "2025"

# Line width
max_width = 100

# Indentation
tab_spaces = 4
hard_tabs = false

# Imports
imports_granularity = "module"  # crate, module, item
group_imports = "preserve"      # preserve, stdext, one
reorder_imports = true

# Pipes and chains
chain_width = 60
single_line_pipe_threshold = 50
pipe_indent = "block"           # block, visual

# Braces
brace_style = "same_line"       # same_line, always_next, prefer_same

# Spaces
spaces_around_ranges = false
space_before_colon = false
space_after_colon = true

# Morphemes
sigil_spacing = "tight"         # tight, spaced

# Comments
wrap_comments = true
comment_width = 80
normalize_comments = true

# Other
newline_style = "auto"          # auto, unix, windows
trailing_comma = "vertical"     # always, never, vertical
trailing_semicolon = false
```

### 3.3 Formatting Rules

```sigil
// BEFORE glyph
let x=1+2*3;users|φ{.active}|τ{.name}|σ.created|collect

// AFTER glyph
let x = 1 + 2 * 3
users
    |φ{.active}
    |τ{.name}
    |σ.created
    |collect

// Imports grouped and sorted
use std·io·{Read, Write}
use std·collections·HashMap

use crate·utils·helper
use crate·core·Engine

use external·library·Thing
```

---

## 4. Augur (Linter)

### 4.1 Usage

```bash
# Run linter
augur                           # Lint current project
augur src/main.sg               # Lint specific file
augur --fix                     # Auto-fix issues
augur --fix-suggested           # Fix suggestions too

# Via incant
incant lint                     # Same as augur
incant fix                      # Same as augur --fix
```

### 4.2 Lint Categories

| Category | Description |
|----------|-------------|
| `correctness` | Likely bugs |
| `style` | Code style issues |
| `complexity` | Overly complex code |
| `performance` | Potential inefficiencies |
| `security` | Security concerns |
| `pedantic` | Strict best practices |
| `nursery` | Experimental lints |

### 4.3 Configuration (Augur.toml)

```toml
# Augur.toml

[lint]
# Lint levels: allow, warn, deny, forbid
default_level = "warn"

# Enable/disable categories
correctness = "deny"
style = "warn"
complexity = "warn"
performance = "warn"
security = "deny"
pedantic = "allow"

# Specific lint configuration
[lint.rules]
unused_variables = "warn"
unused_imports = "warn"
dead_code = "warn"
unreachable_code = "deny"
missing_docs = "allow"
unsafe_code = "warn"
unwrap_used = "warn"
expect_used = "allow"
panic_in_result = "deny"
todo = "warn"
dbg_macro = "warn"
print_stdout = "warn"
large_stack_frames = { level = "warn", threshold = 512000 }
cognitive_complexity = { level = "warn", threshold = 25 }
too_many_arguments = { level = "warn", threshold = 7 }
too_many_lines = { level = "warn", threshold = 100 }

# Ignore patterns
[lint.ignore]
paths = ["target/", "generated/"]
```

### 4.4 Inline Configuration

```sigil
// Disable lint for block
//@ augur: allow(unused_variables)
fn experimental() {
    let x = 42  // No warning
}

// Disable for single line
let _ = risky_operation()  //@ augur: allow(unwrap_used)

// Module-level
//! @ augur: deny(unsafe_code)
//! @ augur: allow(missing_docs)
```

### 4.5 Example Lints

```sigil
// correctness::infinite_loop
loop { }  // Warning: infinite loop without break

// style::redundant_clone
let s2 = s1·clone|clone  // Warning: redundant clone

// complexity::cognitive_complexity
fn complex() {  // Warning: function has high cognitive complexity
    if a {
        if b {
            match c {
                // ...deeply nested...
            }
        }
    }
}

// performance::unnecessary_collect
data|collect|iter|map(f)  // Warning: use |τ{f} directly

// security::sql_injection
sql!("SELECT * FROM users WHERE id = " + user_input)  // Error!

// style::inconsistent_morpheme
data.map(f).filter(g)  // Warning: prefer morpheme style |τ{f}|φ{g}
```

---

## 5. Oracle (Language Server)

### 5.1 Features

| Feature | Description |
|---------|-------------|
| Completion | Context-aware code completion |
| Hover | Type info and documentation |
| Go to Definition | Navigate to declarations |
| Find References | Find all usages |
| Rename | Rename symbols across project |
| Code Actions | Quick fixes and refactors |
| Diagnostics | Real-time error reporting |
| Inlay Hints | Type and parameter hints |
| Semantic Highlighting | Rich syntax highlighting |

### 5.2 Configuration

```json
// settings.json (VS Code example)
{
    "oracle-sigil.checkOnSave": true,
    "oracle-sigil.cargo.buildScripts.enable": true,
    "oracle-sigil.inlayHints.typeHints": true,
    "oracle-sigil.inlayHints.parameterHints": true,
    "oracle-sigil.inlayHints.chainingHints": true,
    "oracle-sigil.inlayHints.evidentialityHints": true,
    "oracle-sigil.lens.enable": true,
    "oracle-sigil.lens.run": true,
    "oracle-sigil.lens.debug": true,
    "oracle-sigil.completion.autoimport.enable": true,
    "oracle-sigil.completion.morphemes.enable": true
}
```

### 5.3 Code Actions

```sigil
// Quick fixes
let x = foo()  // Oracle suggests: add type annotation
let x: i32 = foo()

// Refactors
users|filter(|u| u.active)|map(|u| u.name)
// Oracle action: "Convert to morpheme syntax"
users|φ{.active}|τ{.name}

// Auto-import
HashMap·new()  // Oracle action: "Import std·collections·HashMap"
```

---

## 6. Scribe (Documentation Generator)

### 6.1 Documentation Comments

```sigil
/// A user in the system.
///
/// # Examples
///
/// ```sigil
/// let user = User·new("Alice", 30)
/// assert!(user.name == "Alice")
/// ```
///
/// # Panics
///
/// Panics if `age` is greater than 150.
///
/// # Errors
///
/// Returns `UserError::InvalidName` if name is empty.
struct User {
    /// The user's full name.
    name: String,
    /// Age in years.
    age: u32,
}

impl User {
    /// Creates a new user with the given name and age.
    ///
    /// # Arguments
    ///
    /// * `name` - The user's name (must not be empty)
    /// * `age` - The user's age (must be <= 150)
    ///
    /// # Returns
    ///
    /// A new `User` instance.
    pub fn new(name: impl Into<String>, age: u32) -> Result<User!, UserError~> {
        // ...
    }
}
```

### 6.2 Module Documentation

```sigil
//! # My Library
//!
//! `my_library` provides utilities for doing amazing things.
//!
//! ## Quick Start
//!
//! ```sigil
//! use my_library·prelude·*
//!
//! let result = do_thing()
//! ```
//!
//! ## Features
//!
//! - Feature 1
//! - Feature 2
```

### 6.3 Scribe Commands

```bash
# Generate documentation
scribe                          # Generate docs
scribe --open                   # Generate and open browser
scribe --no-deps                # Skip dependencies
scribe --document-private       # Include private items
scribe --all-features           # Enable all features

# Via incant
incant doc
incant doc --open
```

### 6.4 Documentation Tests

```sigil
/// Returns the sum of two numbers.
///
/// ```sigil
/// let result = add(2, 3)
/// assert_eq!(result, 5)
/// ```
///
/// ```sigil,should_panic
/// add(i32·MAX, 1)  // Overflow!
/// ```
///
/// ```sigil,no_run
/// // This example compiles but doesn't run
/// let server = start_server()
/// ```
///
/// ```sigil,ignore
/// // This example is ignored entirely
/// platform_specific_code()
/// ```
fn add(a: i32, b: i32) -> i32 {
    a + b
}
```

---

## 7. Crucible (Test Runner)

### 7.1 Test Organization

```sigil
// Unit tests (same file)
fn add(a: i32, b: i32) -> i32 { a + b }

//@ rune: cfg(test)
mod tests {
    use super·*

    //@ rune: test
    fn test_add() {
        assert_eq!(add(2, 2), 4)
    }

    //@ rune: test
    //@ rune: should_panic(expected = "overflow")
    fn test_overflow() {
        add(i32·MAX, 1)
    }

    //@ rune: test
    //@ rune: ignore
    fn test_slow() {
        // Ignored unless explicitly run
    }
}
```

```sigil
// tests/integration.sg (integration tests)
use my_crate·*

//@ rune: test
fn test_full_workflow() {
    let result = my_crate·process(input)
    assert!(result.is_ok())
}
```

### 7.2 Test Attributes

| Attribute | Purpose |
|-----------|---------|
| `test` | Mark as test |
| `ignore` | Skip unless `--ignored` |
| `should_panic` | Expect panic |
| `timeout(duration)` | Fail if exceeds duration |
| `async` | Async test |
| `parallel` | Allow parallel execution |
| `serial` | Force serial execution |

### 7.3 Crucible Commands

```bash
# Run tests
crucible                        # All tests
crucible test_name              # Specific test
crucible --lib                  # Library tests only
crucible --doc                  # Doc tests only
crucible --test integration     # Integration tests
crucible --ignored              # Run ignored tests
crucible --include-ignored      # Run all including ignored

# Filtering
crucible pattern                # Tests matching pattern
crucible --exact test_name      # Exact match only

# Output
crucible -- --nocapture         # Show stdout/stderr
crucible -- --test-threads=1    # Single-threaded
crucible --format json          # JSON output

# Via incant
incant test
incant test --release
```

---

## 8. Alembic (Benchmark Runner)

### 8.1 Benchmark Definition

```sigil
use std·test·Bencher

//@ rune: bench
fn bench_sort(b: &mut Bencher) {
    let data = (0..1000)|collect::<Vec<_>>
    b|iter(|| {
        data·clone|sort
    })
}

//@ rune: bench
fn bench_hashmap_insert(b: &mut Bencher) {
    b|iter(|| {
        let mut map = HashMap·new()
        for i in 0..1000 {
            map|insert(i, i * 2)
        }
        map
    })
}

// Parameterized benchmarks
//@ rune: bench
fn bench_sizes(b: &mut Bencher) {
    for size in [100, 1000, 10000] {
        b|with_name("size_{size}")
         |iter(|| process(size))
    }
}
```

### 8.2 Alembic Commands

```bash
# Run benchmarks
alembic                         # All benchmarks
alembic bench_name              # Specific benchmark
alembic --baseline main         # Compare to baseline
alembic --save-baseline new     # Save as baseline

# Output
alembic --format json           # JSON output
alembic --output results/       # Save results

# Via incant
incant bench
```

### 8.3 Benchmark Output

```
Running benches/sort.sg

bench_sort              ... bench:       1,234 ns/iter (+/- 45)
bench_hashmap_insert    ... bench:      12,345 ns/iter (+/- 234)

Comparison with baseline 'main':
  bench_sort            -5.2% (improvement)
  bench_hashmap_insert  +2.1% (regression)
```

---

## 9. Vessel (Distribution)

### 9.1 Binary Packaging

```toml
# Sigil.toml
[package.metadata.vessel]
# Target platforms
targets = [
    "x86_64-unknown-linux-gnu",
    "x86_64-apple-darwin",
    "aarch64-apple-darwin",
    "x86_64-pc-windows-msvc",
]

# Installers to generate
installers = ["shell", "powershell", "homebrew", "msi"]

# Archive formats
archives = ["tar.gz", "tar.xz", "zip"]

# Include additional files
include = [
    "README.md",
    "LICENSE",
    "CHANGELOG.md",
]
```

### 9.2 Vessel Commands

```bash
# Build distribution artifacts
vessel build                    # Build for all targets
vessel build --target x86_64-unknown-linux-gnu

# Generate installers
vessel generate                 # Generate all installers

# Create release
vessel release                  # Full release workflow
vessel release --dry-run        # Preview without releasing
```

---

## 10. Sigil REPL (Interactive Shell)

### 10.1 Usage

```bash
sigil                           # Start REPL
sigil -e "1 + 1"                # Evaluate expression
sigil script.sg                 # Run script file
```

### 10.2 REPL Features

```
sigil> let x = 42
sigil> x * 2
84

sigil> fn greet(name: str) { print!("Hello, {name}!") }
sigil> greet("World")
Hello, World!

sigil> :type x
i32!

sigil> :help
Commands:
  :help       Show this help
  :type <e>   Show type of expression
  :quit       Exit REPL
  :clear      Clear screen
  :load <f>   Load file
  :save <f>   Save session

sigil> :quit
```

---

## 11. Editor Integration

### 11.1 VS Code Extension

```json
// Features
{
    "sigil.oracle.path": "oracle",
    "sigil.incant.path": "incant",
    "sigil.formatOnSave": true,
    "sigil.lintOnSave": true,
    "sigil.inlayHints.enable": true,
    "sigil.codeLens.enable": true
}
```

### 11.2 Vim/Neovim

```lua
-- init.lua
require('lspconfig').oracle.setup{}

vim.api.nvim_create_autocmd("BufWritePre", {
    pattern = "*.sg",
    callback = function()
        vim.lsp.buf.format()
    end,
})
```

### 11.3 Emacs

```elisp
;; init.el
(use-package sigil-mode
  :mode "\\.sg\\'"
  :hook (sigil-mode . lsp-deferred)
  :config
  (setq sigil-format-on-save t))
```

---

## 12. Toolchain Management

### 12.1 Conjure (Toolchain Manager)

```bash
# Install toolchains
conjure install stable
conjure install nightly
conjure install 0.1.0

# Set default
conjure default stable

# Project-specific
conjure override set nightly

# List installed
conjure show
conjure show installed

# Update
conjure update
conjure update stable

# Components
conjure component add oracle
conjure component add glyph
conjure component add augur
conjure component list
```

### 12.2 sigil-toolchain.toml

```toml
# Project toolchain override
[toolchain]
channel = "nightly-2025-01-15"
components = ["oracle", "glyph", "augur"]
targets = ["wasm32-unknown-unknown"]
```

---

## 13. Summary

| Tool | Command | Purpose |
|------|---------|---------|
| **Incant** | `incant build/run/test` | Build & manage |
| **Glyph** | `incant fmt` | Format code |
| **Augur** | `incant lint` | Lint code |
| **Oracle** | (LSP) | IDE support |
| **Scribe** | `incant doc` | Generate docs |
| **Crucible** | `incant test` | Run tests |
| **Alembic** | `incant bench` | Benchmarks |
| **Vessel** | `vessel build` | Distribution |
| **Conjure** | `conjure install` | Toolchain mgmt |
