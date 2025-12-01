# Oracle - Sigil Language Server

Oracle is the Language Server Protocol (LSP) implementation for the Sigil programming language. It provides IDE features for both human developers and AI agents.

## Features

- **Real-time Diagnostics**: Parse errors and type errors as you type
- **Hover Information**: Documentation for morphemes, keywords, and operators
- **Code Completion**: Keywords, morphemes, types, and stdlib functions
- **Evidentiality Awareness**: Full support for Sigil's trust-tracking type system

## Installation

### From Source

```bash
cd sigil/tools/oracle
cargo build --release
```

The binary will be at `target/release/sigil-oracle`.

### Add to PATH

```bash
# Add to your shell profile
export PATH="$PATH:/path/to/sigil/tools/oracle/target/release"
```

## Usage

Oracle communicates via stdio using the Language Server Protocol.

### With VS Code

1. Install the Sigil extension
2. Ensure `sigil-oracle` is in your PATH
3. Open a `.sigil` or `.sg` file

### With Neovim (nvim-lspconfig)

```lua
require('lspconfig').sigil_oracle.setup{
  cmd = { "sigil-oracle" },
  filetypes = { "sigil" },
  root_dir = function(fname)
    return require('lspconfig').util.find_git_ancestor(fname)
  end,
}
```

### With Helix

Add to `languages.toml`:

```toml
[[language]]
name = "sigil"
scope = "source.sigil"
file-types = ["sigil", "sg"]
language-server = { command = "sigil-oracle" }
```

## For AI Agents

Oracle is designed to work well with AI agents that generate Sigil code:

1. **Immediate Feedback**: Diagnostics are published on every document change
2. **Rich Completions**: Includes both Greek morphemes and ASCII equivalents
3. **Type Information**: Hover provides type and evidentiality details
4. **Structured Output**: All responses follow LSP JSON-RPC format

### Example: Using with Claude

```python
# AI can check its generated code by:
# 1. Writing to a .sigil file
# 2. Opening it in an editor with Oracle
# 3. Reading diagnostics from the LSP

# Or programmatically via the LSP protocol
import subprocess
import json

# Start Oracle
proc = subprocess.Popen(
    ['sigil-oracle'],
    stdin=subprocess.PIPE,
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE
)

# Send LSP initialize request
# ... (standard LSP protocol)
```

## Capabilities

| Feature | Status |
|---------|--------|
| textDocument/publishDiagnostics | ✅ |
| textDocument/hover | ✅ |
| textDocument/completion | ✅ |
| textDocument/definition | 🚧 Planned |
| textDocument/references | 🚧 Planned |
| textDocument/formatting | 🚧 Planned |
| textDocument/semanticTokens | 🚧 Planned |

## Configuration

Oracle respects the following environment variables:

- `RUST_LOG`: Set to `sigil_oracle=debug` for verbose logging
- `SIGIL_ORACLE_CACHE`: Directory for caching parsed files (default: none)

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                     Oracle LSP                           │
├─────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────┐  │
│  │   Lexer     │→ │   Parser    │→ │  Type Checker   │  │
│  │ (sigil_parser)│ │(sigil_parser)│ │ (sigil_parser)  │  │
│  └─────────────┘  └─────────────┘  └─────────────────┘  │
│         ↓                ↓                  ↓           │
│  ┌──────────────────────────────────────────────────┐   │
│  │              Diagnostics & Analysis               │   │
│  └──────────────────────────────────────────────────┘   │
│         ↓                ↓                  ↓           │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────┐  │
│  │   Hover     │  │ Completions │  │   Diagnostics   │  │
│  └─────────────┘  └─────────────┘  └─────────────────┘  │
└─────────────────────────────────────────────────────────┘
              ↓ (LSP JSON-RPC over stdio)
┌─────────────────────────────────────────────────────────┐
│                    Editor / AI Agent                     │
└─────────────────────────────────────────────────────────┘
```

## License

MIT License - Daemoniorum, Inc.
