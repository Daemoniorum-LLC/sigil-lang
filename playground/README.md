# Sigil Playground

A browser-based environment for writing, running, and sharing Sigil code.

## Features

- **Live code editor** with Sigil syntax highlighting
- **Run code** with instant output
- **Type checking** with evidentiality enforcement
- **AI IR viewer** for structured code analysis
- **Share** code via URL
- **Examples** to get started quickly

## Quick Start

```bash
# Install dependencies
npm install

# Start development server
npm run dev

# Build for production
npm run build
```

## Execution Modes

### Mock Mode (Default)
The playground includes a mock interpreter for demonstration purposes. This allows the playground to work as a static site without a backend.

### Backend Mode
For full execution support, set environment variables:

```bash
VITE_USE_BACKEND=true
VITE_API_BASE=http://localhost:8080
```

This connects to a Sigil execution backend (see `tools/mcp-server` for reference).

### WASM Mode (Future)
Full client-side execution via WebAssembly is planned. This will require:

```bash
# Install wasm-pack
cargo install wasm-pack

# Build WASM module
npm run build:wasm
```

## Keyboard Shortcuts

| Shortcut | Action |
|----------|--------|
| `Ctrl/Cmd + Enter` | Run code |
| `Ctrl/Cmd + Shift + Enter` | Check types |

## URL Sharing

Code can be shared via URL:

```
https://sigil-lang.org/playground#code=BASE64_ENCODED_CODE
```

The playground automatically loads code from the URL hash when opened.

## Development

### Project Structure

```
playground/
├── index.html      # Main HTML page
├── src/
│   └── main.js     # Playground logic & editor setup
├── package.json    # Dependencies
└── vite.config.js  # Build configuration
```

### Adding Examples

Edit the `EXAMPLES` object in `src/main.js`:

```javascript
const EXAMPLES = {
  myExample: `fn main() {
    // Your example code
}`,
};
```

Then add it to the dropdown in `index.html`.

## Deployment

### Static Hosting (GitHub Pages, Vercel, Netlify)

```bash
npm run build
# Deploy contents of dist/ folder
```

### With Backend

1. Deploy the Sigil execution backend
2. Set `VITE_API_BASE` to the backend URL
3. Build and deploy the frontend

## Evidence Markers

The playground highlights Sigil's evidentiality markers:

| Marker | Color | Meaning |
|--------|-------|---------|
| `!` | Green | Known - verified/computed |
| `?` | Orange | Uncertain - validated |
| `~` | Blue | Reported - external data |
| `‽` | Red | Paradox - self-referential |

## License

MIT - See LICENSE in the repository root.
