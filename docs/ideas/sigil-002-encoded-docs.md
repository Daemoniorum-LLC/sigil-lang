<!-- IDEA_START -->
### IDEA_002
```yaml
id: IDEA_002
title: "Sigil-Encoded Documentation Format"
status: active
priority: high
category: infrastructure/documentation
created: 2025-12-06
author: human
parent: IDEA_001
```
#### DESCRIPTION
Use Sigil as the canonical format for agent-oriented documentation. All docs stored/served in Sigil encoding for agent consumption, with transformation to Markdown/HTML for human mode.

**Core Concept:**
- Sigil = source of truth for agent mode
- Human-readable formats derived FROM Sigil (not vice versa)
- Leverages Sigil's evidentiality types, morpheme operators, and type system

#### RATIONALE
- Sigil designed specifically for AI agent consumption
- Evidentiality types (`!`, `?`, `~`) provide trust metadata agents can act on
- Morpheme operators enable powerful doc querying/transformation
- Type-safe cross-references catch broken links at compile time
- Positions Daemoniorum docs as native to the AI-first paradigm

#### DEPENDENCIES
- sigil-lang compiler/runtime
- Sigil → Markdown renderer (may need development)
- JDoc/KDoc → Sigil transformer (may need development)

#### TAGS
- sigil
- documentation
- agent-first
- transformation-pipeline

#### DISCUSSION
<!-- DISCUSSION_ENTRY -->
```yaml
author: agent
model: claude-opus-4
timestamp: 2025-12-06
type: analysis
```
**Implementation Considerations:**

**What exists in sigil-lang:**
- Full compiler with LLVM backend
- LSP (oracle/) for editor support
- Formatter (tools/glyph/)
- KMP bindings for Kotlin integration

**What may need building:**
1. **Sigil Doc Schema** - Formal structure for documentation types
2. **Sigil → Markdown Renderer** - For human mode derivation
3. **Markdown → Sigil Ingester** - Bootstrap existing docs into Sigil
4. **JDoc/KDoc → Sigil Bridge** - Via sigil-kmp bindings?

**Evidentiality in Practice:**
```sigil
struct DocEntry {
    content: !String,           // verified content
    source_commit: !Hash,       // known provenance
    last_verified: ?Timestamp,  // may be stale
    external_refs: ~[URL],      // unverified external links
}
```

**Open Question:** Should Sigil docs be compiled or interpreted at runtime? Compiled = faster, but interpreted = easier hot-reload during development.
<!-- /DISCUSSION_ENTRY -->
