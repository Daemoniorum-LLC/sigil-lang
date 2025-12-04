# Shared Types

**Common Infrastructure for Sigil Agent Modules**

This module consolidates types used across multiple agent infrastructure modules, eliminating redundancy and ensuring consistent behavior.

## Why Shared Types?

During architectural review, we identified significant type redundancy:
- `Timestamp` was defined 4 times across modules
- `Duration` was defined 2 times
- `HumanId` was defined 2 times
- History buffer patterns were duplicated 4 times
- `Severity` enums were defined 3 times with slight variations

This module provides canonical implementations that all modules can import.

## Core Types

### Temporal
- **`Timestamp`** — Nanosecond-precision timestamps
- **`Duration`** — Time intervals with convenient constructors

### Identity
- **`EntityId`** trait — Consistent ID behavior
- **`Id<T>`** — Generic typed ID wrapper
- **`AgentId`**, **`HumanId`**, **`MemoryId`**, etc. — Type-safe IDs

### Values
- **`Value`** — Heterogeneous data container with type hints
- **`ValueType`** — Type discriminant for values

### Collections
- **`BoundedDeque<T>`** — Auto-evicting deque (replaces duplicated history buffers)

### Priority
- **`Priority`** — Unified severity/priority levels (Info → Blocking)
- **`Confidence`** — Confidence level with semantic helpers

### Relational
- **`TrustLevel`** — Trust between entities (for Anima/Covenant integration)

### Errors
- **`AgentError`** — Standard error type
- **`AgentResult<T>`** — Result type alias

### Metadata
- **`Metadata`** — Common metadata (timestamps, tags, attributes)

### Morphemes
- **`CertaintyMarker`** — `!`, `?`, `~`, `◊`
- **`RelationalMarker`** — `∿`, `⟳`, `∞`, `◎`

## Usage

```sigil
use shared::{
    Timestamp, Duration,
    AgentId, HumanId, EntityId,
    BoundedDeque,
    Priority, Confidence,
    TrustLevel,
    AgentError, AgentResult,
    Metadata,
};

// Temporal operations
let start = Timestamp::now();
let timeout = Duration::from_secs(30);
let deadline = start.add(timeout);

// Type-safe IDs
let agent: AgentId = AgentId::generate();
let human: HumanId = HumanId::generate();

// Bounded history
let mut history: BoundedDeque<Event> = BoundedDeque::new(1000);
history.push_back(event); // Auto-evicts when full

// Confidence with semantics
let confidence = Confidence::new(0.85);
println!("{}", confidence.describe()); // "confident"

// Trust dynamics
let mut trust = TrustLevel::initial();
trust.increase(0.2);
```

## Module Usage

| Module | Types Used |
|--------|------------|
| engram | Timestamp, Duration, MemoryId, Value, BoundedDeque |
| daemon | AgentId, TaskId, Priority, Confidence |
| commune | AgentId, MessageId, Timestamp, BoundedDeque |
| omen | TaskId, Timestamp, Confidence, Priority |
| aegis | AgentId, TrustLevel, Priority |
| covenant | HumanId, AgentId, TrustLevel, BoundedDeque |
| oracle | Timestamp, Confidence, Priority |
| gnosis | ExperienceId, Timestamp, Confidence, BoundedDeque |
| anima | InsightId, TrustLevel, Timestamp, RelationalMarker |

## Design Principles

1. **Single Source of Truth** — Each type has one canonical definition
2. **Semantic Helpers** — Types provide meaningful methods (e.g., `Confidence::is_high()`)
3. **Type Safety** — Generic ID wrapper prevents mixing different ID types
4. **Polysynthetic Integration** — Morpheme markers for language integration

---

*Foundation types for the infrastructure that serves all minds*
