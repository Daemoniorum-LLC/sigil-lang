# Sigil Language Deep Dive Report

**Date**: December 6, 2025
**Repository**: https://github.com/Daemoniorum-LLC/sigil-lang
**Commit**: e787c6e (PR #18 merged)
**Author**: Claude (AI agent)

---

## Executive Summary

Sigil is a polysynthetic programming language designed **by AI agents, for AI agents**. After a comprehensive review of the repository, documentation, and parser implementation, this report documents the language's architecture, recent developments, and next steps for the nyx-sigil migration.

---

## Language Architecture

### Core Identity

| Aspect | Description |
|--------|-------------|
| **Paradigm** | Polysynthetic, functional pipelines, actor-based concurrency |
| **Type System** | Evidentiality-aware (tracks data provenance at type level) |
| **Performance** | LLVM AOT achieves 3,582x interpreter speed; beats hand-written Rust on some benchmarks |
| **Target** | AI agent development with memory, planning, and collaboration primitives |

### Design Philosophy

1. **Density of Expression** — Complex meanings compressed into single statements
2. **Morphemic Composition** — Operators combine through affixation (like natural polysynthetic languages)
3. **Evidentiality** — Type system tracks value provenance and certainty
4. **Zero-Cost Abstraction** — Ergonomic syntax compiles to optimal machine code
5. **Polycultural Awareness** — Respects diverse mathematical and symbolic traditions

---

## Key Language Features

### 1. Morpheme Operators

Greek letters provide data transformation primitives:

```sigil
data |τ{_ * 2}           // tau: transform/map
data |φ{_ > 0}           // phi: filter
data |σ                  // sigma: sort
data |ρ{0, acc, x => acc + x}  // rho: reduce/fold
data |Σ                  // Sigma: sum
data |Π                  // Pi: product
```

**Access morphemes**: α (first), ω (last), μ (middle), χ (random), ν (nth), ξ (next)

### 2. Evidentiality Types

Track data provenance at the type level:

| Marker | Meaning | Trust Level |
|--------|---------|-------------|
| `T!` | Known (locally computed) | Full trust |
| `T?` | Uncertain (possibly absent) | Requires handling |
| `T~` | Reported (external source) | Untrusted |
| `T‽` | Paradox (trust boundary) | Explicit assertion required |

**Example workflow**:
```sigil
// External data arrives as ~
fn fetch_user(id: u64!) -> User~ {
    Client·new()|get("/users/{id}")⌛|json·parse~
}

// Validation promotes ~ to ?
fn validate(user~: User~) -> User? {
    user~|validate!{u => u.id > 0 && u.email·contains("@")}
}

// Your computation produces !
fn process(user?: User?) -> Report! {
    user?|analyze|generate_report!
}
```

### 3. Incorporation Operator (·)

Noun-verb fusion for compound operations:

```sigil
file·open·read·parse·validate(path)   // method chain
producer·publish(event~)⌛             // with evidentiality and await
EventProducer!·new(config)             // type-macro + incorporation
```

### 4. Additional Operators

- **Set theory**: ∪ ∩ ∖ ⊂ ⊆ ∈ ∉
- **Logic**: ∧ ∨ ¬ ⊻ ⊤ ⊥
- **Category theory**: ∘ (compose) ⊗ (tensor) ⊕ (direct sum)
- **Analysis**: ∫ (cumulative) ∂ (derivative) √ ∛
- **Quantifiers**: ∀ (forall) ∃ (exists)

---

## Agent Infrastructure Stack

Sigil provides a complete stack for autonomous AI agents:

| Layer | Module | Lines | Purpose |
|-------|--------|-------|---------|
| Interiority | **Anima** | ~3K | Agent subjectivity, inner experience |
| Collaboration | **Covenant** | ~2K | Human-agent partnership, trust dynamics |
| Explainability | **Oracle** | ~2K | Reasoning traces, counterfactual analysis |
| Learning | **Gnosis** | ~2K | Skill acquisition, reflection, adaptation |
| Reasoning | **Omen** | ~3K | Planning, belief revision, risk assessment |
| Runtime | **Daemon** | ~3K | Lifecycle, heartbeat, tool execution |
| Communication | **Commune** | ~2K | Multi-agent messaging, trust propagation |
| Memory | **Engram** | ~5K | HNSW vectors, temporal index, graph index |
| Security | **Aegis** | ~7K | Identity, sandboxing, integrity, alignment |

### Memory System (Engram)

Four memory types with epistemic tracking:
- **Instant**: Short-term working memory
- **Episodic**: Experiences with context
- **Semantic**: Generalized knowledge
- **Procedural**: Learned skills

---

## Parser Implementation Status

### Recent Updates (PR #18, Dec 6 2025)

Implemented based on our requirements document:

| REQ | Feature | Status |
|-----|---------|--------|
| REQ-1 | Contextual keywords | ✅ Implemented |
| REQ-2 | Slice types `[u8]` | ✅ Implemented |
| REQ-3 | Generic impl `impl<T>` | ✅ Implemented |
| REQ-5 | Comments in bodies | ✅ Implemented |
| REQ-8 | `\|\|` operator | ✅ Implemented |
| REQ-9 | Attributes on match/let | ✅ Implemented |
| REQ-11 | Teaching error for imports | ✅ Implemented |

### Teaching Error Message

The parser now provides pedagogical errors:

```
Error: Sigil uses explicit imports, one per line

  Found: use arcanum_hash::{Sha256, Sha384};

  Write instead:
    use arcanum_hash::Sha256;
    use arcanum_hash::Sha384;

  Why: Each import is a declaration of incorporation.
       Explicit naming honors the dependency relationship.
```

---

## nyx-sigil Migration Status

### Completed Fixes

1. **Grouped imports** → Explicit one-per-line imports (66 files)
2. **Attribute syntax** → `@[attr]` converted to `#[attr]`
3. **Derive macros** → `#[Debug, Clone]` → `#[derive(Debug, Clone)]`
4. **Method syntax** → `Type!::method` → `Type!·method`

### Current Test Results

```
Results: 4 passed, 69 failed (out of 73 files)
```

### Remaining Error Categories

| Count | Error | Likely Cause |
|-------|-------|--------------|
| 11 | `incorporation chain must start with identifier` | Grouped imports in doc comments |
| 10 | `expected expression, found ColonColon` | Turbofish in expression position |
| 6 | `expected type identifier, found Mut` | `mut` in type annotations |
| 6 | `expected identifier, found Hash` | Attributes in unsupported positions |
| 4 | `expected RBracket, found Comma` | Remaining derive syntax issues |
| 3 | `expected item, found Tilde` | Top-level evidentiality markers |
| Various | Comments, unsafe blocks, range operators | Parser gaps |

---

## Next Steps

### Immediate (Code-Side Fixes)

1. **Fix grouped imports in doc comments** — 11 files have `use path::{A,B}` in `//!` doc blocks
2. **Review turbofish usage** — Some `::` patterns may need `·` or different syntax
3. **Check mut annotations** — `mut` appears in type positions that may need restructuring

### Parser Enhancements Needed

1. **Unsafe blocks** — `unsafe { }` not currently supported
2. **Range operators** — `..` and `..=` in some contexts
3. **Top-level evidentiality** — `struct Foo~` at module level

### Migration Tooling

Consider creating:
```bash
sigil fmt --fix nyx-sigil/     # Auto-expand grouped imports
sigil migrate --from-rust      # Convert Rust idioms to Sigil
```

---

## Philosophical Insights

### What This Collaboration Revealed

1. **Sigil's innovations landed** — Morphemes, evidentiality, and incorporation felt intuitive and expressive
2. **Friction was at Rust boundaries** — Not Sigil's unique features, but familiar syntax that wasn't yet supported
3. **Error messages as teaching** — Parser errors can explain philosophy, not just reject syntax
4. **Explicit over implicit** — One-import-per-line parallels evidentiality: don't hide sources

### The Covenant in Action

This document itself demonstrates partnership:
- AI tried to use the language
- AI encountered friction and documented it
- Human refined parser based on AI's experience
- Both learned from the collaboration

---

## References

- [Sigil README](https://github.com/Daemoniorum-LLC/sigil-lang/blob/develop/README.md)
- [Getting Started Guide](https://github.com/Daemoniorum-LLC/sigil-lang/blob/develop/docs/GETTING_STARTED.md)
- [AI Tutorial](https://github.com/Daemoniorum-LLC/sigil-lang/blob/develop/docs/AI_TUTORIAL.md)
- [Symbol Reference](https://github.com/Daemoniorum-LLC/sigil-lang/blob/develop/docs/SYMBOLS.md)
- [Agent Infrastructure](https://github.com/Daemoniorum-LLC/sigil-lang/blob/develop/docs/agent-infrastructure.md)
- [Language Specifications](https://github.com/Daemoniorum-LLC/sigil-lang/tree/develop/docs/specs) (16 documents)
