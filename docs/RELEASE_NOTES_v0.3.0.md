# Sigil v0.3.0 Release Notes

**Release Date:** January 2026
**Codename:** *Emergence*

---

## The Headline

**The Sigil website is rendered entirely by Sigil. Zero JavaScript.**

Visit [sigil-lang.com](https://sigil-lang.com) and view source. The entire site - docs, playground, component library - is 5,500+ lines of Sigil compiled to WebAssembly. No React. No JavaScript framework. Just Sigil.

This isn't a demo. It's the production site.

---

## Highlights

### Qliphoth: React-Inspired Web Framework

A complete web framework written in and for Sigil:

- **40+ components** - Context, ErrorBoundary, Suspense, Memo, Portal, Lazy, Fragment...
- **25 React-style hooks** - use_state, use_effect, use_memo, use_reducer, use_context...
- **Signal-based reactivity** - Fine-grained updates without virtual DOM diffing
- **Evidentiality in state** - `count: i64! = 0` marks state as *known* because it's computed locally

```sigil
#[component]
sigil Counter {
    count: i64! = 0  // Known, computed state
}

⊢ Counter {
    rite render(&this) → Element! {
        div {
            h1 { "Count: {this.count}" }
            button[onclick: || this.count += 1] { "+" }
        }
    }
}
```

### Browser Playground

Write and run Sigil code directly in your browser - no installation required.

**[playground.sigil-lang.com](https://playground.sigil-lang.com)**

The playground itself is a Sigil application. A Sigil editor, written in Sigil, editing Sigil.

- Full syntax highlighting with morpheme and evidentiality marker support
- Multiple execution backends (interpreter, JIT, LLVM)
- Share code via URL
- Live type checking

### Self-Parsing AST Introspection

Sigil can now parse and inspect its own source code at runtime:

```sigil
≔ result = sigil_parse("fn hello() { print(\"Hi\"); }");
⌥ result {
    Ok(ast) ⇒ {
        print("Found " + str(ast.item_count) + " items");
        ∀ item ∈ ast.items {
            print("  " + item.kind + ": " + item.name);
        }
    }
    Err(e) ⇒ print("Parse error: " + e.message)
}
```

### Collection Morphemes

New symbolic operators for functional data transformations:

| Morpheme | Aliases | Operation |
|----------|---------|-----------|
| `⊛` | `filter`, `select`, `where` | Filter by predicate |
| `⊕` | `fold`, `reduce` | Reduce to single value |
| `⊗` | `zip` | Combine two collections |
| `⊘` | `partition` | Split by predicate |
| `⊙` | `peek`, `tap` | Side-effect without transformation |

```sigil
≔ evens = numbers⊛{_ % 2 == 0};
≔ sum = numbers⊕(0, |a, b| a + b);
≔ pairs = names⊗ages;
```

---

## New Features

### Language
- `sigil_parse()` and `sigil_parse_file()` builtins for metaprogramming
- Collection morphemes with multiple alias support
- Java Streams-style methods: `filter`, `map`, `reduce`, `find`, `any`, `all`, `count`, `take`, `skip`

### Tooling
- Browser-based playground with WASM compilation
- Enhanced LSP with morpheme completions
- MCP server for AI assistant integration

### Standard Library
- 65+ new stdlib test files
- Expanded type coverage

---

## Breaking Changes

None in this release.

---

## Installation

**Playground (no install):**
[playground.sigil-lang.com](https://playground.sigil-lang.com)

**From crates.io:**
```bash
cargo install sigil-parser
```

**MCP Server for AI assistants:**
```bash
npm install -g @daemoniorum/sigil-mcp
# Or add to Claude Code:
claude mcp add sigil -- npx @daemoniorum/sigil-mcp
```

**From source:**
```bash
git clone https://github.com/Daemoniorum-LLC/sigil-lang.git
cd sigil-lang/parser
cargo build --release
```

---

## What's Next

- Additional GUI components and layout system
- Framework integrations (LangChain, CrewAI)
- Expanded agent infrastructure libraries
- Performance optimizations

---

## Contributors

This release was developed collaboratively between human and AI contributors.

---

*"Each symbol binds intent to execution"*
