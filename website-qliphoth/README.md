# Sigil Language Website (Qliphoth)

The official website for [sigil-lang.com](https://sigil-lang.com), written entirely in Sigil and compiled to WebAssembly.

## Built with Qliphoth

This website demonstrates the Sigil web framework:

1. **Written entirely in Sigil** using the Qliphoth UI framework
2. **Compiles to WebAssembly** for zero-JavaScript execution
3. **Proof of concept**: Sigil → WASM web applications

## Current Status

| Component | Status |
|-----------|--------|
| Sigil source | Complete - `src/main.sigil` (270 lines) |
| WASM compilation | ✅ Working - 6.3 KB output |
| JavaScript runtime | Complete - `sigil_runtime.js` provides host bindings |
| CSS styling | Complete - All WASM-generated components styled |
| Browser testing | In progress - Testing with Sitra browser |

**This is a working proof of concept.** The WASM module runs in browsers and generates the full website UI via virtual DOM operations.

## Architecture

```
website-qliphoth/
├── README.md           # This file
├── src/
│   └── main.sigil      # Complete website source (270 lines)
└── dist/
    ├── index.html      # HTML shell with CSS and loader
    ├── site.wasm       # Compiled WASM module (6.3 KB)
    └── sigil_runtime.js # JavaScript runtime (host bindings)
```

## Components

The website is built with these Sigil functions:

- **header()** - Navigation with brand, links, and version badge
- **hero()** - Landing section with title, tagline, and evidentiality markers
- **announcement()** - Fixed point celebration banner
- **features_section()** - Four feature cards in a grid
- **code_example()** - Sigil code sample with syntax examples
- **footer()** - Site footer with attribution

## Build

### Prerequisites

- Sigil compiler (Rust parser): `cargo build --release --features wasm` in `sigil-lang/`
- Python 3 (for dev server)

### Commands

```bash
# Compile to WASM
cd /path/to/sigil-lang
cargo run --release --features wasm -- compile \
  website-qliphoth/src/main.sigil \
  -o website-qliphoth/dist/site.wasm \
  --target wasm

# Serve locally
cd website-qliphoth/dist
python3 -m http.server 5181
```

### Build Pipeline

```
Sigil → WebAssembly (direct)

1. sigil-lang parser with --features wasm reads main.sigil
2. Generates WASM binary with 116 host imports
3. sigil_runtime.js provides JavaScript bindings for imports
4. Browser loads index.html → site.wasm → renders UI
```

## Technology

- **Language**: Sigil 0.3.0 (Development)
- **Compilation**: Direct WASM generation via Rust parser
- **Runtime**: sigil_runtime.js (14 import modules, 116 functions)
- **Output**: 6.3 KB WASM binary

## Design System

The site uses the Sigil color palette:

| Color | Hex | Use |
|-------|-----|-----|
| Background | `#050507` | Primary background |
| Accent | `#14A088` | Phthalo green, links, highlights |
| Known (!) | `#14A088` | Verified data marker |
| Uncertain (?) | `#D4A017` | Uncertain data marker |
| Reported (~) | `#6B9BD1` | External data marker |
| Paradox (‽) | `#D47171` | Trust boundary marker |

Typography uses IBM Plex Sans (body) and IBM Plex Mono (code).

## WASM Imports

The runtime provides 14 import modules with 116 total functions:

| Module | Functions | Purpose |
|--------|-----------|---------|
| console | 4 | Logging (i64, f64, str, print) |
| string | 6 | String operations (concat, slice, parse) |
| dom | 12 | Direct DOM manipulation |
| events | 7 | Event handling |
| timing | 6 | setTimeout, setInterval, RAF |
| fetch | 5 | HTTP requests |
| storage | 3 | localStorage |
| router | 3 | History API |
| memory | 4 | Memory allocation |
| morpheme | 20 | Array operations |
| math | 13 | Math functions |
| vdom | 9 | Virtual DOM |
| signal | 10 | Reactive signals |
| async | 12 | Promises, continuations |

## String Encoding

Strings in WASM use a 4-byte length prefix (little-endian i32) followed by UTF-8 bytes:

```
[len: i32][bytes: u8...]
```

The runtime reads from linear memory starting at HEAP_START (0x4000 = 16384).

## Fallback

If WASM fails to load, users see an error screen with a link to the static HTML version at `../website/index.html`.

## License

MIT License - Daemoniorum LLC
