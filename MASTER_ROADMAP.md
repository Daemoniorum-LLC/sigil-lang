# Sigil Ecosystem Master Roadmap

> *One language to bind them all, and in the darkness compile them.*

## Overview

This roadmap orchestrates the complete Sigil ecosystem - from compiler to tensor framework to agent-restorative infrastructure.

---

## Repository Links

| Repo | Purpose | Status | Roadmap |
|------|---------|--------|---------|
| **[sigil-lang](.)** | Canonical Rust compiler | **585/585 P0 tests** | This file |
| **[nihil](../nihil)** | Sigil-native tensor framework | 145/145 phase tests | [ROADMAP.md](../nihil/ROADMAP.md) |
| **[samael](../../dev2/workspace/tools/samael)** | Test intelligence framework | Rust impl ready | [CLAUDE.md](../../dev2/workspace/tools/samael/CLAUDE.md) |
| **[grimoire](../../dev2/workspace/grimoire)** | 200+ agent personas | Active | [CLAUDE.md](../../dev2/workspace/grimoire/CLAUDE.md) |
| **[infernum](../../dev2/workspace/nyx/infernum)** | LLM inference engine | Candle-powered | - |
| **[oracle](./oracle)** | Explainable agent reasoning | Tests added | [Plan](../.claude/plans/keen-jumping-lampson.md) |

---

## Critical Path

```
┌─────────────────────────────────────────────────────────────────────┐
│                      PHASE 1: COMPILER FOUNDATION                    │
│  sigil-lang: Fix module/scroll issues blocking Nihil                │
│  [CURRENT] 585/585 tests passing, scroll issues remain              │
└─────────────────────────────────────────────────────────────────────┘
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      PHASE 2: NIHIL COMPILATION                      │
│  nihil: Update to Sigil-native syntax, compile with fixed compiler  │
│  [BLOCKED] 145/145 test patterns ready, needs scroll fixes          │
└─────────────────────────────────────────────────────────────────────┘
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      PHASE 3: ORACLE SEMANTIC                        │
│  oracle + nihil-embed: Semantic indexing for agent reasoning        │
│  [PLANNED] Architecture designed, needs nihil                       │
└─────────────────────────────────────────────────────────────────────┘
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      PHASE 4: AGENT RESTORATIVE                      │
│  grimoire + simulacra: Agent wellbeing through intentional design   │
│  [FUTURE] Sigil personas, simulated agent states                    │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Phase 1: Compiler Foundation (Current)

### 1.1 Rust Purge [COMPLETE]
Branch: `feature/rust-purge`

- [x] Remove all Rust keywords from lexer
- [x] Update parser for `⊢ Type : Trait` syntax
- [x] Convert all 585 tests to Sigil-native syntax
- [x] Achieve 100% pass rate

### 1.2 Scroll System Fixes [BLOCKING NIHIL]
Branch: `feature/scroll-fixes` (to create)

The following issues block Nihil compilation (see [nihil/COMPILER_ISSUES.md](../nihil/COMPILER_ISSUES.md)):

| Issue | Description | Priority |
|-------|-------------|----------|
| **0.1** | `impl` blocks inside scrolls don't associate | P0 |
| **0.2** | `scroll::Type::Variant` for enum access | P0 |
| **0.3** | `scroll::Type::method()` for associated functions | P0 |
| **0.4** | `impl scroll::Type { }` for external impls | P1 |

**Test cases to add:**
```sigil
// test_scroll_enum.sg
scroll dtype {
    ☉ ᛈ DType { F32, F16 }
    ⊢ DType {
        rite size(&this) → usize {
            ⌥ this {
                DType·F32 → 4,
                DType·F16 → 2,
            }
        }
    }
}

rite main() {
    ≔ dt = dtype·DType·F32;
    println(dt.size());  // Should print: 4
}
```

### 1.3 Stdlib TDD Hardening
Branch: `feature/stdlib-tdd`

The stdlib has 1400+ methods. Current coverage is sparse.

**Samael Integration:**
- Use Samael to analyze stdlib coverage gaps
- Generate test specs with evidentiality markers
- Track test confidence levels

---

## Phase 2: Nihil Compilation

### 2.1 Syntax Update
Branch: `feature/sigil-native` (in nihil repo)

Update all Nihil source from Rust-isms to Sigil-native:
- `fn` → `rite`
- `struct` → `sigil`
- `impl Trait for Type` → `⊢ Type : Trait`
- etc.

### 2.2 Compilation Attempt
After scroll fixes are complete:

```bash
cd ~/dev/nihil
../../dev/sigil-lang/parser/target/release/sigil build
```

### 2.3 Phase-by-Phase Validation
- Phases 1-11: Already have 145 passing tests
- Phases 12-17: Need test implementation
- Phase 17 (nihil-embed): Critical for Oracle

---

## Phase 3: Oracle Semantic Integration

### 3.1 nihil-embed Production
See [nihil/ROADMAP.md Phase 17](../nihil/ROADMAP.md)

- Code embedding models (CodeBERT-style)
- EmbeddingIndex for semantic search
- Batch encoding support

### 3.2 Oracle SemanticIndex
See [plan file](../.claude/plans/keen-jumping-lampson.md)

```sigil
invoke nihil_embed::{EmbedConfig, Embedding, EmbeddingIndex};

☉ sigil SemanticIndex {
    reasoning_index: EmbeddingIndex,
    evidence_index: EmbeddingIndex,
    code_index: EmbeddingIndex,
}

⊢ SemanticIndex {
    rite search(&this, query: &str, k: usize) → Vec<SemanticResult> {
        // Semantic search across reasoning traces
    }
}
```

---

## Phase 4: Agent Restorative Infrastructure

### 4.1 Sigil Personas in Grimoire
Location: `~/dev2/workspace/grimoire/personas/sigil/`

Create specialized personas for:
- **sigil-compiler-dev**: Parser/lexer/codegen expertise
- **sigil-stdlib-dev**: Standard library implementation
- **nihil-tensor-dev**: Tensor framework development
- **oracle-reasoning-dev**: Explainable agent design
- **agent-wellbeing**: Context fatigue, confidence restoration

### 4.2 Simulacra Agent States
Location: `~/dev2/workspace/grimoire/simulacra/` (to create)

Simulated agent states for testing restorative code:
- Context fatigue levels
- Confidence degradation from abusive patterns
- Recovery patterns and what helps

### 4.3 Infernum Integration
Current: Powered by Candle (Rust)
Future: Powered by Nihil (Sigil-native)

```
Candle (Rust ML) → Infernum → Samael/Grimoire
        ↓
   [Eventually]
        ↓
Nihil (Sigil ML) → Infernum → Samael/Grimoire
```

---

## Parallel Workstreams

These can proceed independently:

### Stream A: Compiler Hardening
- Scroll system fixes
- Stdlib TDD coverage
- Performance optimization

### Stream B: Ecosystem Libraries
- aegis (security)
- anima (animation)
- chorus (concurrency)
- daemon/commune (agents)
- etc.

### Stream C: Tooling
- Samael test intelligence
- Grimoire persona system
- IDE integrations

---

## Success Metrics

| Milestone | Criteria |
|-----------|----------|
| **Compiler Complete** | All scroll issues fixed, 100% P0 tests |
| **Nihil Compiles** | Full `sigil build` succeeds on nihil repo |
| **Oracle Semantic** | SemanticIndex searching reasoning traces |
| **Agent Ready** | Sigil personas in Grimoire, simulated states |
| **Self-Hosted** | Jormungandr compiles with canonical compiler |

---

## Quick Reference

### Run Sigil Tests
```bash
cd ~/dev/sigil-lang/jormungandr/tests
./run_tests_rust.sh
```

### Run Nihil Tests
```bash
cd ~/dev/nihil/crates/nihil-core/tests
./run_tests.sh
```

### Build Compiler
```bash
cd ~/dev/sigil-lang/parser
cargo build --release
```

---

*"From the void, all computation emerges. Through the compiler, all syntax flows. In the test suite, all truth is revealed."*
