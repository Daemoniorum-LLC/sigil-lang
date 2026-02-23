# Commune Library — Software Design Document (v0.2)

**Scope:** Minimum viable in-process message router for Morgoth mq.sg
integration (Phase C4 of the Morgoth module split plan).

**Supersedes:** `architecture.md` for implementation purposes. Full
architecture remains aspirational specification; this SDD governs what is
actually built and tested.

---

## Context and Motivation

Commune is designed as a full multi-agent communication infrastructure
(Intent, Trust, Channels, Swarms, Consensus, CRDT). All source files in
`src/` use an aspirational Sigil dialect with native symbols (`☉ ᛈ ⌥ Δ`)
incompatible with Morgoth's test suite.

Morgoth's current messaging layer in `mq.sg` uses:
- `MqMessage { sender, recipient, kind, payload, ts }` — ad-hoc struct
- `pane.inbox: Array` — per-pane message buffer embedded in the Pane struct
- `mq_send(panes, from, to, kind, payload)` — iterates panes to push to inbox
- `mq_persist(msg)` — appends JSON to `~/.morgoth/messages.jsonl`

The goal of Phase C4 is to replace that with:
1. A commune-typed `Message` struct (replacing `MqMessage`)
2. A `Commune` in-process routing table (replacing `pane.inbox` and the
   mq_send iteration loop)

This eliminates scattered inbox management, gives messages a commune-typed
home, and establishes the integration pattern for future Intent/Trust adoption.

---

## Scope Decision Table

| Subsystem | Status | Rationale |
|-----------|--------|-----------|
| `Message` struct + constructor | **IN SCOPE** | Direct replacement for `MqMessage` |
| `Commune` in-process routing table | **IN SCOPE** | Replaces `pane.inbox` + `mq_send` loop |
| `register / unregister / is_registered` | **IN SCOPE** | Agent lifecycle management |
| `send(msg)` — route to recipient queue | **IN SCOPE** | Replaces inbox push loop |
| `recv(id)` — drain agent queue | **IN SCOPE** | Replaces `pane.inbox` drain in main loop |
| `AgentId` wrapper struct | **DEFERRED** | String UUIDs sufficient; wrap in C3 alongside DaemonId |
| `Intent` enum | **DEFERRED** | `kind` + `payload` strings sufficient at this scope |
| FIFO transport | **OUT OF SCOPE** | OS primitives; stays hand-rolled in mq.sg |
| Unix socket transport | **OUT OF SCOPE** | OS primitives; stays hand-rolled in mq.sg |
| `mq_persist()` | **OUT OF SCOPE** | File I/O concern; stays in mq.sg (uses engram bridge) |
| `write_registry / write_manifest` | **OUT OF SCOPE** | Pane serialization utilities; not messaging |
| Trust, channels, swarms, consensus | **DEFERRED** | No current consumer |
| Epistemic types | **DEFERRED** | Not relevant at this scope |

---

## Message Design

### Struct

```sigil
Σ Message { sender: String, recipient: String, kind: String, payload: String, ts: String }
```

`ts` is stored as a string (`to_string(Sys·clock_gettime())`) to match the
existing `messages.jsonl` serialization format. This avoids breaking
`mq_load_pending` replay.

### Constructor

```sigil
rite Message·new(sender, recipient, kind, payload) -> Message
```

Sets `ts` to `to_string(Sys·clock_gettime())`.

### Replaces

`MqMessage` struct and `mq_message()` constructor are removed from `mq.sg`.
`invoke tome·commune·{Message}` replaces the struct definition.

---

## Commune Design

### Struct

```sigil
Σ Commune { agents: Array, messages: Array }
```

`agents` is a flat array of registered agent id strings. `messages` is a
flat array of all pending `Message` objects across all agents. Flat storage
avoids nested-array slot mutation limitations in Sigil; `recv(id)` partitions
the array and writes back via struct field assignment.

### API

```sigil
// Constructor — empty routing table
rite Commune·new() -> Commune

// Add an agent to the routing table (creates an empty queue)
// No-op if agent already registered
rite register(self, id_str)

// Remove agent and discard its pending messages
// No-op if agent not registered
rite unregister(self, id_str)

// Returns true if id_str is currently registered
rite is_registered(self, id_str) -> Bool

// Route msg to the recipient's queue
// Silently drops the message if recipient is not registered
rite send(self, msg)

// Drain and return all pending messages for id_str
// Returns an empty array if agent is not registered or has no messages
// Clears the queue after returning
rite recv(self, id_str) -> Array
```

### Behaviour Invariants

- `Commune·new()` returns an empty routing table with zero agents.
- After `register(id)`, `is_registered(id)` returns `true`.
- After `unregister(id)`, `is_registered(id)` returns `false` and any
  pending messages for `id` are discarded.
- `send(msg)` where `msg.recipient` is registered appends `msg` to that
  agent's queue. No error is raised.
- `send(msg)` where `msg.recipient` is **not** registered is silently
  dropped (no error, no persistence side-effect).
- `recv(id)` returns the messages in FIFO order (oldest first).
- `recv(id)` clears the queue: a second consecutive `recv(id)` returns `[]`.
- Messages sent to different agents do not appear in each other's queues.
- Multiple messages sent to the same agent accumulate in queue order.

---

## Morgoth Call-Site Changes

### mq.sg

```sigil
// Remove from mq.sg:
//   Σ MqMessage { ... }
//   rite mq_message(...) { ... }

// Add import:
invoke tome·commune·{Message, Commune};

// mq_send signature changes: panes array no longer needed for routing
// Before:
rite mq_send(panes, from_id, to_id, kind, payload) {
    ≔ msg = mq_message(from_id, to_id, kind, payload);
    ≔ mut mi = 0;
    ⟳ mi < len(panes) {
        ⎇ panes[mi].id == to_id { push(panes[mi].inbox, msg); }
        mi = mi + 1;
    }
    mq_persist(msg);
}

// After:
rite mq_send(commune, from_id, to_id, kind, payload) {
    ≔ msg = Message·new(from_id, to_id, kind, payload);
    commune·send(msg);
    mq_persist(msg);
}

// mq_persist is unchanged (file I/O, not commune)
// mq_process_external: replace push(panes[N].inbox, ...) with commune·send(...)
// mq_load_pending: replace push(panes[fi].inbox, ...) with commune·send(...)
// copy_yank_to_claude: replace mq_send(panes, ...) with mq_send(commune, ...)
```

### pane.sg

```sigil
// Remove from Pane struct:
//   inbox: Array

// inbox is now managed by Commune; pane struct no longer owns it
```

### main.sg

```sigil
// Initialization — after panes are created:
≔ mut commune = Commune·new();
≔ mut ci = 0;
⟳ ci < len(panes) {
    commune·register(panes[ci].id);
    ci = ci + 1;
}

// Pane creation (create_pane called): commune·register(new_pane.id)
// Pane death (alive = false):         commune·unregister(dead_pane.id)

// Inbox drain loop — before:
⟳ ii < len(panes) {
    ⎇ len(panes[ii].inbox) > 0 {
        ⟳ imi < len(panes[ii].inbox) {
            ≔ imsg = panes[ii].inbox[imi];
            // ... handle imsg.kind ...
        }
        panes[ii].inbox = [];
    }
}

// After:
⟳ ii < len(panes) {
    ≔ msgs = commune·recv(panes[ii].id);
    ⎇ len(msgs) > 0 {
        ≔ mut imi = 0;
        ⟳ imi < len(msgs) {
            ≔ imsg = msgs[imi];
            // ... handle imsg.kind (unchanged) ...
            imi = imi + 1;
        }
    }
    ii = ii + 1;
}

// Pass commune to: mq_send, mq_process_external, mq_load_pending, copy_yank_to_claude
```

---

## Implementation Location

Following the daemon/engram bridge pattern:

- **Library**: `sigil-lang/commune/src/commune_core.sg` — plain executable Sigil
- **Bridge**: `morgoth/src/commune.sg` — verbatim copy of commune_core.sg
- **Tests**: `sigil-lang/commune/tests/commune_core_test.sg` + `.expected`

Existing aspirational `src/commune.sg`, `src/types.sg`, `src/lib.sg` are
untouched.

---

## Test Requirements

10 Agent-TDD tests, all written before production code is touched:

| # | Test | Requirement |
|---|------|-------------|
| 1 | `test_new_empty` | `Commune·new()` has no registered agents |
| 2 | `test_register_is_registered` | `register(id)` → `is_registered(id)` true |
| 3 | `test_unregister_clears` | `unregister(id)` → `is_registered(id)` false; pending msgs discarded |
| 4 | `test_send_recv_basic` | `send(msg)` then `recv(id)` returns that message |
| 5 | `test_recv_clears_queue` | second `recv(id)` returns empty |
| 6 | `test_recv_unknown_agent` | `recv(unknown_id)` returns `[]` without error |
| 7 | `test_send_to_unregistered_dropped` | `send` to unregistered recipient is silently dropped |
| 8 | `test_multiple_agents_isolated` | messages to agent A do not appear in agent B's recv |
| 9 | `test_message_fields` | `Message·new(s, r, k, p)` fields are set correctly |
| 10 | `test_queue_ordering` | multiple sends arrive in FIFO order |

---

## Deferred

- `AgentId` wrapper type (Phase C3, alongside `DaemonId`)
- `Intent` enum replacing `kind`+`payload` strings
- FIFO channels (`create_pane_fifo` stays hand-rolled)
- Unix socket endpoint (`setup_mq_socket` stays hand-rolled)
- Trust, swarm, consensus, epistemic types
- Full `Commune` from aspirational `src/commune.sg`
