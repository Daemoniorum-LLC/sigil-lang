# Sigil MCP Server

An MCP (Model Context Protocol) server that enables AI systems to write, run, type-check, and analyze Sigil code.

## Why?

Sigil is a programming language built for AI, by AI. This MCP server puts Sigil directly in the hands of AI assistants, allowing them to:

- **Write and execute Sigil code** with proper evidentiality tracking
- **Type-check code** with enforcement of evidence levels
- **Analyze code as structured data** via the AI IR format
- **Generate explanations** of what Sigil code does

The evidentiality system (`!` known, `?` uncertain, `~` reported) lets AI systems be honest about the certainty of their data - and the type checker enforces it.

## Installation

```bash
cd tools/mcp-server
npm install
npm run build
```

## Configuration

### Claude Desktop

Add to your `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "sigil": {
      "command": "node",
      "args": ["/path/to/sigil-lang/tools/mcp-server/dist/index.js"],
      "env": {
        "SIGIL_BIN": "/path/to/sigil-lang/parser/target/release/sigil"
      }
    }
  }
}
```

### Environment Variables

- `SIGIL_BIN`: Path to the Sigil binary (default: `sigil` in PATH)

## Tools

### `sigil_run`

Execute Sigil code and return the output.

```typescript
{
  code: string,      // Sigil source code
  backend?: string   // "interpreter" (default) or "jit"
}
```

**Example:**
```sigil
fn main() {
    let nums = [1, 2, 3, 4, 5];
    let doubled = nums |τ{_ * 2};
    print(doubled);
}
```

### `sigil_check`

Type-check Sigil code with evidentiality enforcement.

```typescript
{
  code: string  // Sigil source code
}
```

Returns type errors including evidence mismatches. This is crucial for AI safety - it tracks data provenance at the type level.

### `sigil_ir`

Get the AI-readable intermediate representation.

```typescript
{
  code: string,     // Sigil source code
  format?: string   // "json" (default) or "pretty"
}
```

Returns a JSON structure optimized for AI analysis, allowing reasoning about code without parsing source text.

### `sigil_explain`

Generate a natural language explanation of Sigil code.

```typescript
{
  code: string  // Sigil source code
}
```

Analyzes code structure and describes:
- What the code computes
- How data flows through pipelines
- What evidence levels are used and why

## Example Session

**AI writes and runs Sigil code:**

```
User: Calculate the sum of squares from 1 to 10

AI: I'll write this in Sigil using morpheme operators:

[calls sigil_run with:]
fn main() {
    let numbers = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
    let sum_of_squares = numbers
        |τ{_ * _}
        |ρ{0, acc, x => acc + x};
    print(sum_of_squares);
}

Output: 385
```

**AI handles external data with evidence tracking:**

```
User: Process this user input safely

AI: I'll mark the input as reported (~) and validate it:

[calls sigil_check with:]
fn process_input(data~: str) -> int? {
    // data is reported - we can't trust it as known
    let validated = data |validate?{parse_int(_)};
    validated  // returns uncertain, not known
}
```

## Evidentiality Quick Reference

| Marker | Name | Meaning |
|--------|------|---------|
| `!` | Known | Computed/verified data - highest certainty |
| `?` | Uncertain | May vary, needs handling |
| `~` | Reported | External data - requires validation |
| `‽` | Paradox | Self-referential or contradictory |

Evidence flows: `known → uncertain → reported`

Known can satisfy any requirement. Reported cannot satisfy known requirements without explicit validation.

## Development

```bash
npm run dev    # Watch mode
npm run build  # Build for production
npm start      # Run the server
```

## License

MIT - See LICENSE in the repository root.
