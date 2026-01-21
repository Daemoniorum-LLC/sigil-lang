# Tree-sitter Grammar for Sigil

A [tree-sitter](https://tree-sitter.github.io/) grammar for the [Sigil programming language](https://github.com/Daemoniorum-LLC/sigil-lang).

## Features

- Full syntax support for Sigil's unique features:
  - **Evidentiality markers**: `!` (known), `?` (uncertain), `~` (reported), `‽` (paradox)
  - **Morpheme operators**: τ (transform), φ (filter), σ (sort/sum), ρ (reduce), and more
  - **Pipeline syntax**: `data |τ{_ * 2} |φ{_ > 10}`
- Rust-like constructs: structs, enums, traits, impl blocks
- Query files for syntax highlighting, indentation, and locals

## Installation

### Neovim (with nvim-treesitter)

Add to your nvim-treesitter config:

```lua
local parser_config = require("nvim-treesitter.parsers").get_parser_configs()

parser_config.sigil = {
  install_info = {
    url = "https://github.com/Daemoniorum-LLC/sigil-lang",
    files = { "tree-sitter-sigil/src/parser.c" },
    branch = "main",
    location = "tree-sitter-sigil",
  },
  filetype = "sigil",
}
```

Then run `:TSInstall sigil`

### Helix

The grammar will be available in Helix once it's added to the [helix-editor/helix](https://github.com/helix-editor/helix) grammars.

For now, you can build manually:

```bash
cd tree-sitter-sigil
npm install
npm run generate
```

Then configure in `~/.config/helix/languages.toml`:

```toml
[[language]]
name = "sigil"
scope = "source.sigil"
file-types = ["sigil", "sg"]
roots = ["Sigil.toml"]

[[grammar]]
name = "sigil"
source = { path = "/path/to/sigil-lang/tree-sitter-sigil" }
```

## Development

### Prerequisites

- Node.js >= 18
- tree-sitter CLI: `npm install -g tree-sitter-cli`

### Building

```bash
npm install
npm run generate
```

### Testing

```bash
npm test
```

### Parsing a file

```bash
tree-sitter parse /path/to/file.sigil
```

## Query Files

- `queries/highlights.scm` - Syntax highlighting
- `queries/indents.scm` - Automatic indentation
- `queries/locals.scm` - Local variable scoping

## Sigil-Specific Highlighting

The grammar provides special highlighting for Sigil's unique features:

### Evidentiality Markers

```sigil
let known! = compute();      // ! = known (verified)
let uncertain? = maybe();    // ? = uncertain (validated)
let reported~ = external();  // ~ = reported (external)
```

### Morpheme Operators

```sigil
// Greek letters get @function.builtin highlighting
let doubled = nums |τ{_ * 2};  // τ = transform
let filtered = nums |φ{_ > 0}; // φ = filter
let total = nums |Σ;           // Σ = sum
```

## License

MIT License - Daemoniorum, LLC
