# Sigil for Neovim

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

3. Have [nvim-lspconfig](https://github.com/neovim/nvim-lspconfig) installed

## Installation

### Option 1: Copy the config file

```bash
mkdir -p ~/.config/nvim/after/plugin
cp sigil.lua ~/.config/nvim/after/plugin/
```

### Option 2: Add to your init.lua

Copy the contents of `sigil.lua` into your Neovim configuration.

### Option 3: Use with lazy.nvim or packer

```lua
-- lazy.nvim example
{
  "neovim/nvim-lspconfig",
  config = function()
    -- Your other LSP configs...

    -- Add Sigil
    local configs = require("lspconfig.configs")
    if not configs.sigil_oracle then
      configs.sigil_oracle = {
        default_config = {
          cmd = { "sigil-oracle" },
          filetypes = { "sigil" },
          root_dir = require("lspconfig").util.find_git_ancestor,
        },
      }
    end
    require("lspconfig").sigil_oracle.setup({})
  end,
}
```

## Features

With Oracle LSP:
- Real-time error diagnostics (`:LspInfo` to verify)
- Hover information (`K` by default)
- Code completion (with nvim-cmp or similar)
- Evidentiality marker support

## Syntax Highlighting

The config uses Rust syntax as a fallback. For better highlighting,
a tree-sitter grammar is planned.

## Troubleshooting

**LSP not attaching?**
- Run `:LspInfo` while in a `.sigil` file
- Check `:checkhealth lsp`
- Ensure `sigil-oracle` is in your PATH

**No completions?**
- Make sure you have a completion plugin (nvim-cmp, etc.)
- Check that the LSP is attached with `:LspInfo`
