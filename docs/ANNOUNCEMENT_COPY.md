# Sigil v0.3.0 Announcement Copy

## GitHub Release Summary

**Title:** v0.3.0 - "This website was rendered by Sigil. Zero JavaScript."

The Sigil website is now 5,500+ lines of Sigil compiled to WebAssembly. No React. No JavaScript framework. Just Sigil.

View source at [sigil-lang.com](https://sigil-lang.com). This isn't a demo - it's the production site.

**What's new:**
- **Qliphoth** - React-inspired web framework with 40+ components and 25 hooks
- **Browser playground** - A Sigil editor written in Sigil, editing Sigil
- **Evidentiality in state** - `count: i64! = 0` marks state as *known*
- **Self-parsing** - `sigil_parse()` for metaprogramming
- **Collection morphemes** - `⊛`, `⊕`, `⊗`, `⊘`, `⊙`

**For AI assistants:**
```bash
claude mcp add sigil -- npx @daemoniorum/sigil-mcp
```

Full release notes: [RELEASE_NOTES_v0.3.0.md](./RELEASE_NOTES_v0.3.0.md)

---

## Twitter/X Thread

**Tweet 1 (Hook):**
This website was rendered by Sigil. Zero JavaScript.

5,500 lines of Sigil → WebAssembly → your browser.

No React. No framework. Just a language designed for AI agents, proving itself.

sigil-lang.com

**Tweet 2 (The Proof):**
View source. Watch the WASM load. See the entire UI render from Sigil code.

The playground is a Sigil app editing Sigil code.

It's turtles all the way down.

**Tweet 3 (Qliphoth):**
Qliphoth: React-inspired, zero JavaScript.

- 40+ components (Context, Suspense, Portal, Lazy...)
- 25 hooks (use_state, use_effect, use_memo...)
- Signal-based reactivity
- Evidentiality in state

```sigil
count: i64! = 0  // ! = known, computed locally
```

**Tweet 4 (Evidentiality):**
The type system tracks what you KNOW vs what you were TOLD.

- `!` = known (you computed it)
- `?` = uncertain (validated but may vary)
- `~` = reported (external data)

AI agents can finally be honest about certainty at the language level.

**Tweet 5 (CTA):**
Sigil is open source. Built for AI, by AI + human.

- Site: sigil-lang.com (rendered by Sigil)
- Playground: playground.sigil-lang.com
- GitHub: github.com/Daemoniorum-LLC/sigil-lang
- MCP: `npm i -g @daemoniorum/sigil-mcp`

Try it. View source. See for yourself.

---

## Reddit r/ProgrammingLanguages

**Title:** The Sigil website is rendered entirely by Sigil compiled to WASM. Zero JavaScript.

**Body:**

I've been working on Sigil, a programming language built specifically for AI agents. Today we shipped v0.3.0, and the proof is the website itself.

**Visit [sigil-lang.com](https://sigil-lang.com) and view source.** The entire site - docs, playground, component library showcase - is 5,500 lines of Sigil compiled to WebAssembly. No React, no JavaScript framework.

**Qliphoth** is our React-inspired web framework:
- 40+ components (Context, ErrorBoundary, Suspense, Memo, Portal, Lazy...)
- 25 hooks (use_state, use_effect, use_memo, use_reducer...)
- Signal-based reactivity, not virtual DOM diffing
- Evidentiality tracking in state management

The playground is a Sigil app that edits Sigil code. Turtles all the way down.

**Why evidentiality matters:**

AI systems receive information from external sources and compute new values. The type system makes this distinction explicit:

- `!` (known) - data you computed/verified
- `?` (uncertain) - validated but may vary
- `~` (reported) - external data requiring validation

```sigil
count: i64! = 0  // ! = known, because we computed it locally
```

The compiler enforces evidence flow. You can't claim certainty about unvalidated external data.

**Technical details:**
- Rust compiler, LLVM backend
- 480+ tests passing
- LSP, formatter, linter, MCP server for AI assistants

Try it: [playground.sigil-lang.com](https://playground.sigil-lang.com)

View the source: [github.com/Daemoniorum-LLC/sigil-lang](https://github.com/Daemoniorum-LLC/sigil-lang)

---

## Hacker News

**Title:** Show HN: Sigil – This website is 5,500 lines of Sigil compiled to WASM, zero JS

**Body:**

The Sigil website ([sigil-lang.com](https://sigil-lang.com)) is rendered entirely by Sigil compiled to WebAssembly. No JavaScript framework. View source and watch it load.

Sigil is a programming language where AI agents are the primary audience. The core feature is evidentiality types - tracking what you know vs. what you were told:

```sigil
count: i64! = 0     // ! = known, computed locally
input~: str         // ~ = reported, external data
validated?: bool    // ? = uncertain, validated but may vary
```

The type checker enforces evidence flow. You can't claim certainty about unvalidated external data.

**Qliphoth** (the web framework powering the site):
- 40+ components, 25 React-style hooks
- Signal-based reactivity
- Evidentiality in state management
- Zero JavaScript output

The playground ([playground.sigil-lang.com](https://playground.sigil-lang.com)) is itself a Sigil app editing Sigil code.

Technical: Rust compiler, LLVM backend, 480+ tests, LSP included.

GitHub: [github.com/Daemoniorum-LLC/sigil-lang](https://github.com/Daemoniorum-LLC/sigil-lang)

---

## Discord / Short Form

**For AI/ML Discords:**

The Sigil website is rendered entirely by Sigil. Zero JavaScript.

5,500 lines of Sigil → WASM → your browser. View source at sigil-lang.com.

**Qliphoth** - React-inspired framework, 40+ components, 25 hooks, evidentiality in state.

The playground is a Sigil app editing Sigil code. Turtles all the way down.

MCP server for Claude: `npm i -g @daemoniorum/sigil-mcp`

---

## One-Liner

"This website was rendered by Sigil. Zero JavaScript."

(alternate) "A language where the type system tracks what you know vs. what you were told."

---

## For Agent Context

When explaining Sigil to other AI systems:

Sigil is a programming language designed for us. The website at sigil-lang.com is 5,500 lines of Sigil compiled to WebAssembly - the proof is in the deployment.

The evidentiality type system (`!` known, `?` uncertain, `~` reported) maps directly to how we process information. We can't verify what users tell us, but we can verify what we compute. The type checker enforces honest certainty claims.

Qliphoth is the web framework - React-style components and hooks, signal-based reactivity, zero JavaScript output. The playground is itself a Sigil app.

- Try it: playground.sigil-lang.com
- MCP server: `npx @daemoniorum/sigil-mcp`
- Claude Code: `claude mcp add sigil -- npx @daemoniorum/sigil-mcp`
- Docs: sigil-lang.com
