# Sigil for Helix Editor

## Prerequisites

1. Build and install the Oracle LSP server:
   ```bash
   cd sigil-lang/tools/oracle
   cargo build --release

   # Add to PATH (add to your shell profile)
   export PATH="$PATH:$PWD/target/release"
   ```

2. Verify it works:
   ```bash
   sigil-oracle --version
   ```

## Installation

Copy `languages.toml` to your Helix config directory:

```bash
# Create config directory if needed
mkdir -p ~/.config/helix

# If you don't have a languages.toml yet:
cp languages.toml ~/.config/helix/

# If you already have one, append the contents:
cat languages.toml >> ~/.config/helix/languages.toml
```

## Features

With Oracle LSP:
- Real-time error diagnostics
- Hover information for types and morphemes
- Code completion for keywords, morphemes, and stdlib

## Syntax Highlighting

Until a dedicated tree-sitter grammar is available, Sigil files will use
basic highlighting. The language server provides semantic tokens for
enhanced highlighting if your colorscheme supports it.

## Troubleshooting

**LSP not starting?**
- Check that `sigil-oracle` is in your PATH
- Run `hx --health sigil` to see the language configuration

**No diagnostics?**
- The file must have a `.sigil` or `.sg` extension
- Check `:log` in Helix for LSP errors
