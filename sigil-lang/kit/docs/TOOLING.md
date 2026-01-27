# Sigil Tooling Guide

This guide covers the development tools included with Sigil: LSP server, formatter, linter, and documentation generator.

---

## Overview

| Tool | Command | Purpose |
|------|---------|---------|
| Formatter | `sigil fmt` | Auto-format code |
| Linter | `sigil lint` | Check code quality |
| LSP Server | `sigil lsp` | IDE integration |
| Doc Generator | `sigil doc` | Generate documentation |
| REPL | `sigil repl` | Interactive shell |

---

## Formatter (sigil fmt)

The formatter enforces consistent code style.

### Usage

```bash
# Check formatting (no changes)
sigil fmt file.sg --check

# Format file in place
sigil fmt file.sg

# Format directory
sigil fmt src/

# Format with verbose output
sigil fmt file.sg --verbose
```

### What It Does

- Consistent indentation (4 spaces)
- Proper spacing around operators
- Line length enforcement (100 soft, 120 hard)
- Import organization
- Trailing comma normalization
- Native symbol preference (configurable)

### Configuration

Create `.sigilfmt.toml` in project root:

```toml
# .sigilfmt.toml

# Maximum line length before wrapping
max_line_length = 100

# Use native symbols (λ, Σ, ≔) vs ASCII (fn, struct, let)
prefer_native_symbols = true

# Trailing commas in multi-line constructs
trailing_comma = "always"  # "always", "never", "multiline"

# Import grouping
group_imports = true
```

---

## Linter (sigil lint)

The linter catches potential issues and enforces best practices.

### Usage

```bash
# Lint single file
sigil lint file.sg

# Lint directory
sigil lint src/

# Show all warnings (including style)
sigil lint file.sg --all

# Only show errors
sigil lint file.sg --errors-only

# JSON output (for CI)
sigil lint file.sg --format json
```

### Rules

| Rule | Severity | Description |
|------|----------|-------------|
| `unused-variable` | Warning | Unused variable declared |
| `unused-import` | Warning | Import not used |
| `dead-code` | Warning | Unreachable code detected |
| `evidence-mismatch` | Error | Evidence marker inconsistency |
| `trust-boundary` | Warning | Unchecked data crossing boundary |
| `shadowing` | Info | Variable shadows outer scope |
| `complexity` | Warning | Function too complex |
| `missing-doc` | Info | Public item lacks documentation |

### Evidentiality Rules

The linter enforces Sigil's evidence tracking:

```sigil
// Warning: external data used without validation
≔ data = fetch_api()~;
process(data);  // ⚠️ Reported (~) data used directly

// OK: validate before use
≔ data = fetch_api()~;
≔ validated = validate(data)?;
process(validated!);  // ✓ Now Known (!)
```

### Configuration

Create `.sigillint.toml`:

```toml
# .sigillint.toml

# Enable/disable specific rules
[rules]
unused-variable = "warn"
unused-import = "warn"
dead-code = "warn"
evidence-mismatch = "error"
trust-boundary = "warn"
shadowing = "allow"
complexity = "warn"
missing-doc = "allow"

# Complexity thresholds
[thresholds]
max_function_lines = 50
max_parameters = 7
max_nesting_depth = 4

# Paths to ignore
[ignore]
paths = ["tests/", "examples/", "vendor/"]
```

---

## LSP Server (sigil lsp)

The LSP server provides IDE features.

### Starting the Server

```bash
# Start LSP server (typically called by editor)
sigil lsp

# With logging
sigil lsp --log-level debug
```

### Features

- **Diagnostics** - Real-time error and warning reporting
- **Completion** - Intelligent code completion
- **Hover** - Type information and documentation
- **Go to Definition** - Jump to symbol definition
- **Find References** - Find all usages
- **Rename** - Rename symbols across project
- **Formatting** - On-save formatting
- **Code Actions** - Quick fixes and refactorings

### VS Code Setup

Install the Sigil extension, or configure manually:

```json
// .vscode/settings.json
{
  "sigil.server.path": "sigil",
  "sigil.server.args": ["lsp"],
  "sigil.format.enable": true,
  "sigil.lint.enable": true,
  "[sigil]": {
    "editor.formatOnSave": true,
    "editor.defaultFormatter": "daemoniorum.sigil"
  }
}
```

### Neovim Setup (nvim-lspconfig)

```lua
-- init.lua
local lspconfig = require('lspconfig')

lspconfig.sigil_ls.setup {
  cmd = { "sigil", "lsp" },
  filetypes = { "sigil", "sg" },
  root_dir = lspconfig.util.root_pattern("Tome.toml", ".git"),
}
```

### Helix Setup

```toml
# ~/.config/helix/languages.toml
[[language]]
name = "sigil"
scope = "source.sigil"
file-types = ["sg", "sigil"]
roots = ["Tome.toml"]
language-server = { command = "sigil", args = ["lsp"] }
indent = { tab-width = 4, unit = "    " }
```

---

## Documentation Generator (sigil doc)

Generate documentation from doc comments.

### Usage

```bash
# Generate docs for current project
sigil doc

# Output to specific directory
sigil doc --output docs/api/

# Generate for single file
sigil doc src/lib.sg

# Open in browser after generation
sigil doc --open
```

### Doc Comment Syntax

```sigil
/// Brief description of the function.
///
/// Longer description can span multiple paragraphs.
///
/// # Arguments
///
/// * `param1` - Description of first parameter
/// * `param2` - Description of second parameter
///
/// # Returns
///
/// Description of return value.
///
/// # Examples
///
/// ```
/// ≔ result = my_function(1, 2);
/// assert_eq!(result, 3);
/// ```
///
/// # Panics
///
/// Describe conditions that cause panic.
///
/// # Errors
///
/// Describe error conditions for Result returns.
☉ λ my_function(param1: i32, param2: i32) → i32 {
    param1 + param2
}
```

### Module Documentation

```sigil
//! # Module Name
//!
//! Brief description of the module.
//!
//! ## Overview
//!
//! Detailed overview...
//!
//! ## Examples
//!
//! ```
//! use my_module·function;
//! ```
```

---

## REPL (sigil repl)

Interactive Sigil shell for experimentation.

### Starting

```bash
sigil repl
```

### Commands

| Command | Description |
|---------|-------------|
| `:help` | Show help |
| `:quit` | Exit REPL |
| `:clear` | Clear screen |
| `:type <expr>` | Show type of expression |
| `:load <file>` | Load file into session |
| `:reset` | Reset session state |

### Example Session

```
$ sigil repl
Sigil 0.4.0 REPL
Type :help for help, :quit to exit

>>> ≔ x = 42
x: i32 = 42

>>> x * 2
84

>>> λ double(n: i32) → i32 { n * 2 }
double: λ(i32) → i32

>>> double(x)
84

>>> :type double
λ(i32) → i32

>>> :quit
Goodbye!
```

---

## CI Integration

### GitHub Actions

```yaml
# .github/workflows/sigil.yml
name: Sigil CI

on: [push, pull_request]

jobs:
  check:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Install Sigil
        run: |
          curl -sSL https://sigil-lang.com/install.sh | sh
          echo "$HOME/.sigil/bin" >> $GITHUB_PATH

      - name: Check formatting
        run: sigil fmt --check src/

      - name: Lint
        run: sigil lint src/ --format json > lint-results.json

      - name: Type check
        run: sigil check src/main.sg

      - name: Test
        run: sigil test
```

---

## Project Structure

Recommended project layout:

```
my-project/
├── Tome.toml           # Package manifest
├── src/
│   ├── main.sg         # Entry point
│   └── lib.sg          # Library root
├── tests/
│   └── test_lib.sg     # Tests
├── docs/               # Generated documentation
├── .sigilfmt.toml      # Formatter config
├── .sigillint.toml     # Linter config
└── .editorconfig       # Editor config
```
