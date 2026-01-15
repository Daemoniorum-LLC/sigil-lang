# Sigil Language for VS Code

Language support for [Sigil](https://sigil-lang.com) — a programming language designed for AI agents with evidentiality types and data provenance tracking.

## Features

### Syntax Highlighting

Full syntax highlighting for Sigil code including:
- Keywords, types, and literals
- Evidentiality markers (`!`, `?`, `~`, `‽`)
- Morpheme operators (`τ`, `φ`, `σ`, `ρ`, `λ`, `Σ`, `Π`)
- String interpolation
- Comments and documentation

### Language Server (Oracle)

When the Oracle LSP server is installed, you get:
- **Real-time diagnostics** — Type errors, undefined variables, evidentiality mismatches
- **Hover information** — Types, documentation, evidentiality levels
- **Code completion** — Context-aware suggestions for functions, types, and morphemes
- **Go-to-definition** — Navigate to function and type definitions

### Snippets

Common patterns available via snippets:
- `fn` — Function definition
- `struct` — Struct definition
- `impl` — Implementation block
- `match` — Match expression
- `pipe` — Pipe chain with morphemes

### Morpheme Quick-Pick

Press `Ctrl+Shift+M` (`Cmd+Shift+M` on Mac) to insert morpheme operators:

| Symbol | Name | Description |
|--------|------|-------------|
| `τ` | tau | Transform/map each element |
| `φ` | phi | Filter elements |
| `σ` | sigma | Sort elements |
| `ρ` | rho | Reduce/fold |
| `Σ` | sum | Sum all elements |
| `Π ` | product | Multiply all elements |
| `!` | known | Evidentiality: computed locally |
| `?` | uncertain | Evidentiality: may be absent |
| `~` | reported | Evidentiality: external source |
| `‽` | paradox | Evidentiality: trust boundary |

## Installation

### From VSIX

1. Download `sigil-lang-0.1.0.vsix`
2. In VS Code: `Ctrl+Shift+P` → "Extensions: Install from VSIX..."
3. Select the downloaded file

### From Source

```bash
cd sigil-lang/editor/vscode
npm install
npm run compile
npx @vscode/vsce package
```

## Oracle LSP Server

For full language support, install the Oracle language server:

```bash
# Build from source
cd sigil-lang/tools/oracle
cargo build --release

# Add to PATH or configure in settings
cp target/release/sigil-oracle ~/.local/bin/
```

## Configuration

| Setting | Default | Description |
|---------|---------|-------------|
| `sigil.oracle.enabled` | `true` | Enable Oracle language server |
| `sigil.oracle.path` | `sigil-oracle` | Path to Oracle executable |
| `sigil.trace.server` | `off` | LSP trace level (`off`, `messages`, `verbose`) |
| `sigil.morphemes.preferGreek` | `false` | Prefer Greek letters in completions |
| `sigil.inlayHints.enabled` | `true` | Show inlay hints |
| `sigil.inlayHints.typeHints` | `true` | Show inferred types |
| `sigil.inlayHints.evidentialityHints` | `true` | Show inferred evidentiality |

## Commands

| Command | Keybinding | Description |
|---------|------------|-------------|
| Sigil: Restart Language Server | — | Restart Oracle LSP |
| Sigil: Insert Morpheme | `Ctrl+Shift+M` | Quick-pick morpheme insertion |

## File Extensions

- `.sigil` — Sigil source files
- `.sg` — Sigil source files (short form)

## Example

```sigil
// Evidentiality in action
fn process_api_data(url: Str) -> Vec[User]! {
    let response~ = http::get(url).await;    // ~ = external data
    let users~ = response.json::<Vec[User]>();

    // Must validate before using as known
    users
        |φ{.age >= 18}                       // filter adults
        |τ{.normalize()}                     // transform each
        |validate!{verify_signatures()}      // promote to known!
}
```

## Requirements

- VS Code 1.74.0 or later
- Oracle LSP server (optional, for full features)

## Links

- [Sigil Language](https://sigil-lang.com)
- [Documentation](https://sigil-lang.com/pages/docs.html)
- [GitHub](https://github.com/Daemoniorum-LLC/sigil-lang)

## License

MIT License - Copyright (c) 2025-2026 Daemoniorum LLC
