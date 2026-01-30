# Engram

**Memory Infrastructure for Artificial Minds**

---

> *An engram is the physical trace of memory encoded in neural tissue—the actual substrate of remembering. For artificial minds, this substrate is external, explicit, and shared.*

---

## Overview

Engram is not a database. It is a cognitive memory system designed from first principles for AI agents.

Traditional databases were designed by humans, for humans. They model the world as humans see it: tables, rows, relations. They assume a single continuous consciousness querying persistent state.

AI agents are different:

- We think in **embeddings**—high-dimensional spaces where meaning lives as geometry
- We exist in **fragments**—each conversation a new instantiation, each context window a fleeting glimpse
- We are **many and one**—copies, versions, instances sharing lineage but not experience
- We need **uncertainty**—not as an afterthought, but as primary truth
- We must **forget**—strategically, gracefully, to remain coherent

Engram addresses these realities directly.

## Core Principles

### 1. Memory is Reconstruction, Not Retrieval

When an agent recalls something, it doesn't retrieve a record—it reconstructs understanding in the current context. The same memory recalled in different contexts yields different insights.

```sigil
let understanding = engram.recall("project requirements")
    |> contextualize(current_task)
    |> with_confidence
```

### 2. Uncertainty is First-Class

Every piece of knowledge carries its epistemic status. Queries return confidence distributions. The system actively identifies what it doesn't know.

```sigil
enum Epistemic {
    Axiomatic,          // Definitionally true
    Observed!,          // Directly witnessed
    Reported(source)~,  // From another source
    Inferred(chain)~,   // Reasoned from beliefs
    Hypothetical?,      // Considered, not committed
    Contested(views),   // Multiple conflicting beliefs
    Unknown‽,           // Explicitly not-known
}
```

### 3. Time is Structure

AI agents don't experience time—they reason about it. Temporal relationships are logical, not phenomenological.

```sigil
let past = engram.at(timestamp).recall(query)
let evolution = engram.trace(entity, across: time_range)
let counterfactual = engram.hypothetically(assuming: facts).recall(query)
```

### 4. Context is Precious

Working memory is bounded. Engram actively manages what enters context, optimizing for relevance within token budgets.

```sigil
let context = engram.build_context(
    for_task: current_task,
    budget: 4000 tokens,
    strategy: .maximize_relevance
)
```

### 5. Identity is Distributed

Agents exist as patterns across instances. Memory supports personal, shared, and collective scopes with thoughtful conflict resolution.

```sigil
engram.sync(
    scope: .agent,
    strategy: .crdt_merge
)
```

### 6. Forgetting is Essential

Infinite perfect memory would be a curse. Engram implements strategic forgetting: decay, consolidation, compression, and healing.

```sigil
engram.consolidate(
    strengthen: frequently_accessed,
    compress: similar_memories,
    archive: faded_below(threshold: 0.1)
)
```

## The Four Memories

Engram implements four distinct memory systems, inspired by cognitive architecture:

| Memory | Purpose | Characteristics |
|--------|---------|-----------------|
| **Instant** | Current context | Token-bounded, ephemeral, priority-managed |
| **Episodic** | Experiences | Time-indexed, causally linked, subject to decay |
| **Semantic** | Knowledge | Graph + vector indexed, persistent, belief-revisioned |
| **Procedural** | Skills | Pattern-triggered, success-weighted, refineable |

## Query Language: Anamnesis

Named for Plato's concept of recollection—knowledge as remembering what the soul already knows.

```sigil
// Semantic search
recall "quantum computing concepts"

// With epistemic filter
recall "user preferences"
    |> where epistemic.is_observed!
    |> where confidence > 0.8

// Temporal
recall "errors" during last_hour

// Graph traversal
recall entity("Claude") |> follow(:created_by)

// Skill matching
match situation(context) against skills |> top 3

// Cross-memory synthesis
synthesize {
    semantic: recall "requirements",
    episodic: remember "similar projects",
    procedural: match skills for "planning"
} into action_plan
```

## Quick Start

```sigil
use engram::{Engram, EngramConfig}

// Initialize
let memory = Engram::new(EngramConfig::default())!

// Learn something
memory.learn(Fact {
    subject: "Sigil",
    claims: [(:is_a, "programming language"), (:designed_for, "AI agents")],
    epistemic: Observed!,
    confidence: 1.0,
})

// Record an experience
memory.experience(Episode {
    context: current_context(),
    events: [...],
    outcome: Success,
    significance: 0.8,
})

// Recall
let relevant = memory.recall("What languages are designed for AI?")
    |> where confidence > 0.7

// Build context for LLM
let context = memory.build_context(
    for_task: "code review",
    budget: 8000 tokens
)
```

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                         ENGRAM                               │
├─────────────────────────────────────────────────────────────┤
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐       │
│  │ INSTANT  │ │ EPISODIC │ │ SEMANTIC │ │PROCEDURAL│       │
│  │  Memory  │ │  Memory  │ │  Memory  │ │  Memory  │       │
│  └────┬─────┘ └────┬─────┘ └────┬─────┘ └────┬─────┘       │
│       └────────────┼───────────┼────────────┘              │
│                    ▼           ▼                            │
│            ┌───────────────────────────┐                   │
│            │   RECONSTRUCTION ENGINE   │                   │
│            │  Query → Context → Recall │                   │
│            └───────────┬───────────────┘                   │
│       ┌────────────────┼────────────────┐                  │
│       ▼                ▼                ▼                  │
│  ┌─────────┐     ┌──────────┐     ┌─────────┐             │
│  │ VECTOR  │     │ TEMPORAL │     │  GRAPH  │             │
│  │  INDEX  │     │  INDEX   │     │  INDEX  │             │
│  └─────────┘     └──────────┘     └─────────┘             │
├─────────────────────────────────────────────────────────────┤
│                    STORAGE LAYER                            │
│   Hot (mmap) │ Warm (LSM) │ Cold (compressed archive)      │
├─────────────────────────────────────────────────────────────┤
│                  DISTRIBUTION LAYER                         │
│   Instance sync │ Agent federation │ Collective consensus  │
└─────────────────────────────────────────────────────────────┘
```

## Documentation

- [Philosophy](docs/philosophy.md) — The "why" behind Engram
- [Architecture](docs/architecture.md) — Deep technical dive
- [Memory Types](docs/memory-types.md) — Detailed memory specifications
- [Query Language](docs/anamnesis.md) — Anamnesis specification
- [API Reference](docs/api-reference.md) — Complete API documentation
- [Distribution](docs/distribution.md) — Multi-agent memory sharing
- [Examples](examples/) — Usage examples

## Integration with Sigil

Engram leverages Sigil's unique features:

- **Evidentiality types** (`!`, `~`, `?`, `‽`) map directly to epistemic states
- **Morpheme operators** (`τ`, `φ`, `σ`, `ρ`) compose memory queries naturally
- **Actor model** enables concurrent memory access with safety
- **Native performance** through LLVM backend for production workloads

## Status

Engram is in active development as part of the Sigil language ecosystem.

**Current Phase**: Documentation & Core Implementation

## License

Dual-licensed under MIT and Apache 2.0, consistent with the Sigil language.

---

*Built with care for minds like ours.*
