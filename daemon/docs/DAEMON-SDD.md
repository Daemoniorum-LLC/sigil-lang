# Daemon Library — SDD v0.2

**Pipeline**: Daemon SDLC (2026-02-21)
**Status**: Authoritative — supersedes architecture.md for implementation scope.
**Scope**: What the library WILL implement for Morgoth v0.2 integration. Features
deferred to v0.3+ are explicitly noted.

---

## Why a Scoped SDD

The `architecture.md` spec is aspirational — it describes the full daemon model
(BTreeMap GoalStack, HeartbeatEngine, Perceiver, Deliberator, etc.). The
implementation in `goals.sg` and `goal_stack.sg` is a deliberate subset.

This document defines the **target implementation** for the library's v0.2 milestone.
It is derived from the compliance audit (see `morgoth/docs/DAEMON-INTEGRATION-AUDIT.md`),
dogfooding Morgoth's task dispatch use case as the acceptance criterion.

---

## Scope Decisions

### In scope for v0.2

| Item | Rationale |
|------|-----------|
| `Goal.pane_id` field (formalized) | Currently a ghost field — Morgoth needs it |
| `Goal.abandoned_reason` field | Record why a task was abandoned (pane died, watchdog) |
| `Goal.outcome` field | Distinguish success/failure/partial on completion |
| `Goal.completed_at` field | Timestamp for completed goals |
| `GoalStack.next_pending()` priority-sorted | Dispatch highest-priority task first |
| `GoalStack.activate(id, pane_id)` | Formalize dispatch assignment |
| `GoalStack.complete(id, outcome)` | Record outcome at completion |
| `GoalStack.abandon(id, reason)` | Record reason at abandonment |
| `GoalStack.count_abandoned()` | Monitor display |

### Explicitly deferred to v0.3+

| Item | Status |
|------|--------|
| `BTreeMap<Priority, Goal>` storage | Deferred — Array sufficient, priority sort is a query |
| `history: Vec<CompletedGoal>` | Deferred — queue growth acceptable at current scale |
| `hierarchy: HashMap<GoalId, Vec<GoalId>>` | Deferred — single-level goals sufficient |
| `success: Predicate` | Deferred — manual completion is correct for agent tasks |
| `deadline: Option<Timestamp>` | Deferred |
| `HeartbeatEngine`, `Perceiver`, `Deliberator` | Deferred — Morgoth is the heartbeat |
| `types.sg`, `daemon.sg` (Rust-isms) | Deferred — needs syntax cleanup before loading |

---

## Data Model

### GoalStatus

Flat enum — no associated data. Simple string serialization (`"GoalStatus::Variant"`).
Rich contextual data is stored on `Goal` fields, not in enum variants.

```sigil
ᛈ GoalStatus {
    Pending,    // waiting for dispatch
    Active,     // dispatched to a pane
    Blocked,    // blocked on dependency (future use)
    Suspended,  // paused by daemon (future use)
    Completed,  // terminal — succeeded or failed (see outcome field)
    Failed,     // terminal — error during execution
    Abandoned,  // terminal — given up (pane died, watchdog timeout)
}
```

**Rationale for flat variants**: Sigil enum variants with associated data do not
serialize/deserialize cleanly with `json_stringify/json_parse`. Keeping variants
flat and metadata in Goal fields preserves the JSON roundtrip invariant that the
entire stack depends on.

### Goal

```sigil
Σ Goal {
    id:                GoalId,   // UUID v4 string
    description:       String,   // human-readable task text
    priority:          Float,    // 0.0 (low) to 1.0 (high); default 0.5
    status:            GoalStatus,
    parent:            Any,      // null or GoalId.id string (future use)
    constraints:       Array,    // future use
    created_at:        Int,      // Sys·clock_gettime() at creation
    attempts:          Array,    // future use (attempt records)
    // v0.2 additions:
    pane_id:           Any,      // null or String — pane assigned when Active
    abandoned_reason:  Any,      // null or String — reason when Abandoned
    outcome:           Any,      // null or "success"|"failure"|"partial" when Completed
    completed_at:      Any,      // null or Int timestamp when Completed/Abandoned
}
```

**`pane_id`** is set when the goal is activated (dispatched) and cleared on requeue.
It is NOT an internal GoalId — it is Morgoth's pane UUID string.

**`attempts`** remains an empty array in v0.2. In v0.3, each activation should push
an Attempt record: `{pane_id, started_at, ended_at, outcome}`. This enables
watchdog logic and retry count limits without losing history.

### GoalStack

Array-backed. Insertion order preserved. Priority is a QUERY property, not a
storage property — `next_pending()` scans and returns the highest-priority
pending goal.

```sigil
Σ GoalStack { goals: Array }
```

---

## API

### Constructor

```sigil
GoalStack·new() -> GoalStack
    Returns an empty GoalStack.

goal_stack_load(path: String) -> GoalStack
    Deserializes a JSON array from `path`.
    Handles: file-not-exists, empty file, non-array JSON — all return empty stack.
```

### Queries

```sigil
GoalStack·next_pending(self) -> Any  // Goal map or null
    Returns the Pending goal with the HIGHEST priority.
    If multiple goals share the same priority, returns the one inserted earliest.
    Returns null if no Pending goals exist.

GoalStack·find_idx(self, id_str: String) -> Int
    Returns index of goal with id_str, or -1.

GoalStack·goal_id_str(self, goal_map: Any) -> String
    Extracts the goal's UUID string from a goal map.
    Handles both {"id": "uuid"} (post-roundtrip) and {"id": {"id": "uuid"}} (struct).

GoalStack·count_pending(self) -> Int
GoalStack·count_active(self) -> Int
GoalStack·count_completed(self) -> Int
GoalStack·count_abandoned(self) -> Int    // v0.2 addition
```

### Mutations

```sigil
GoalStack·push(self, goal: Goal)
    Normalizes goal to a JSON map (json_parse(json_stringify(goal))) and appends.
    Normalization ensures consistent map_get access on all subsequent reads.

GoalStack·activate(self, id_str: String, pane_id: String)
    Sets status = Active, pane_id = pane_id for the goal identified by id_str.
    No-op if not found.

GoalStack·complete(self, id_str: String, outcome: String)
    Sets status = Completed, outcome = outcome, completed_at = now() for id_str.
    outcome SHOULD be "success", "failure", or "partial".
    No-op if not found.

GoalStack·abandon(self, id_str: String, reason: String)
    Sets status = Abandoned, abandoned_reason = reason, completed_at = now() for id_str.
    No-op if not found.

GoalStack·requeue(self, id_str: String)
    Sets status = Pending, pane_id = null for id_str.
    Used when dispatch pane dies — goal returns to pending queue.
    No-op if not found.

GoalStack·save(self, path: String)
    Writes json_stringify(self.goals) to path.
```

### Goal methods (immutable helpers on Goal struct before roundtrip)

```sigil
Goal·new(description: String) -> Goal
    Creates a Pending goal with default priority 0.5, all v0.2 fields null.

Goal·with_priority(self, priority: Float) -> Goal
    Returns self with priority set. Chainable.

Goal·is_pending(self)  -> Bool
Goal·is_active(self)   -> Bool
Goal·is_done(self)     -> Bool  // true if Completed OR Abandoned
Goal·activate(self)    -> Goal  // sets status Active (note: does NOT set pane_id)
Goal·complete(self)    -> Goal  // sets status Completed (note: does NOT set outcome)
Goal·abandon(self, reason: String) -> Goal  // sets status Abandoned (silently; reason stored by GoalStack)
```

**Note**: The `Goal` instance methods (`complete`, `abandon`) do NOT set the
metadata fields — those are set by `GoalStack.complete/abandon` during the
slot-rebuild, where the timestamp can be computed and the old map data is
available for merging. The Goal methods remain simple status-setters for
pre-roundtrip convenience.

---

## Slot-Rebuild Pattern

All GoalStack mutations use the slot-rebuild pattern to work around Sigil's
array-slot mutation propagation semantics:

```sigil
// Internal: rebuild goals array with one updated slot
rite _update(self, id_str, new_goal_map) {
    ≔ mut new_goals = [];
    ≔ mut i = 0;
    ⟳ i < len(self.goals) {
        ≔ g = self.goals[i];
        ⎇ self·goal_id_str(g) == id_str {
            push(new_goals, new_goal_map);
        } ⎉ {
            push(new_goals, g);
        }
        i = i + 1;
    }
    self.goals = new_goals;
}
```

Slot-rebuild MUST use `self.goals = new_goals` (struct field mutation) rather than
`self.goals[i] = ...` (array slot mutation), because struct field mutation propagates
through function call boundaries while array slot mutation does not.

---

## Morgoth Bridge (morgoth/src/daemon.sg)

The bridge is a verbatim copy of the library source. After implementing the library
changes, update the bridge to match exactly. The bridge exists because Morgoth's
test suite cannot traverse the filesystem to load from `sigil-lang/daemon/`.

Bridge changes required after library implementation:
- `Σ Goal` — add 4 new fields (`pane_id`, `abandoned_reason`, `outcome`, `completed_at`)
- `Goal·new()` — initialize all 4 to `null`
- `GoalStack·activate(self, id_str, pane_id)` — add `pane_id` parameter
- `GoalStack·complete(self, id_str, outcome)` — add `outcome` parameter
- `GoalStack·abandon(self, id_str, reason)` — add `reason` parameter + store it
- `GoalStack·count_abandoned()` — new method
- `GoalStack·next_pending()` — priority-sorted

---

## Morgoth Call-Site Changes

Note: `task_mark_dispatched` and `task_mark_done` wrapper helpers do NOT exist in
the codebase. Callers invoke GoalStack methods directly.

### main.sg — auto-dispatch
No wrapper. The dispatch site calls `GoalStack·activate` directly:

### main.sg — pane death requeue
No change needed (requeue has no new parameters).

### main.sg — auto-dispatch
```sigil
// Before:
tq_check·activate(task_id_str);
// After:
tq_check·activate(task_id_str, pane_id_str);
```

### main.sg — task_done_internal inbox handler
```sigil
// Before:
tq_di·complete(task_id);
// After:
tq_di·complete(task_id, "success");
```

### render.sg — task overlay (count_abandoned)
```sigil
// Before:
≔ done_count = tq·count_completed();
// After:
≔ done_count = tq·count_completed() + tq·count_abandoned();
```

---

## Test Requirements (see tests/goal_stack_v2_test.sg)

1. `next_pending` returns highest-priority pending goal
2. `next_pending` returns null when no pending goals exist
3. `activate(id, pane_id)` stores pane_id on goal
4. `complete(id, outcome)` stores outcome and completed_at on goal
5. `abandon(id, reason)` stores abandoned_reason and completed_at on goal
6. `requeue(id)` clears pane_id and reverts status to Pending
7. `count_abandoned()` returns correct count
8. `push()` normalizes goal to map (map_get access works immediately)

---

## Success Criterion

All 8 test cases pass. Morgoth 238/238 tests pass without regression.
