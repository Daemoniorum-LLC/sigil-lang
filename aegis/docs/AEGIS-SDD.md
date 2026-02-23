# Aegis Library — Software Design Document (v0.1)

**Scope:** Minimum viable security policy engine for Morgoth Phase D integration.

**Supersedes:** `architecture.md` and `philosophy.md` for implementation purposes.
Full architecture remains aspirational specification; this SDD governs what is
actually built and tested.

---

## Context and Motivation

Morgoth exposes a Unix domain socket at `~/.morgoth/morgoth.sock` for external
clients to send messages (task_add, task_done, notify, etc.). Any process running
as the same user can connect and send arbitrary message kinds.

Phase D requires a security layer that:
1. Allows known-safe message kinds (notify, query, paste)
2. Denies dangerous message kinds by policy (e.g., shell_exec, eval — future)
3. Provides a clear extension point for future constraints (peer UID, sandboxing)

**Note on socket-level auth:** `Sys·chmod` is not available in the Sigil stdlib
at this scope; socket file permissions cannot be set programmatically. Unix domain
sockets are local-only by transport; peer UID verification via `SO_PEERCRED`
requires raw getsockopt pointer manipulation (deferred to stdlib extension).

The v0.1 security model is therefore: **kind-based message filtering at accept time.**

---

## Scope Decision Table

| Feature | Status | Rationale |
|---------|--------|-----------|
| `AegisConfig` struct | **IN SCOPE** | Config holder for policy |
| `AegisConfig·default()` | **IN SCOPE** | Safe defaults |
| `aegis_check_kind(cfg, kind)` | **IN SCOPE** | Kind-based filtering |
| Socket chmod (0600) | **DEFERRED** | Requires `Sys·chmod` stdlib addition |
| Peer UID verification (SO_PEERCRED) | **DEFERRED** | Requires raw getsockopt pointer |
| `Constitution` / `Directive` types | **DEFERRED** | Phase D-extended or Phase E |
| Audit logging | **DEFERRED** | No current consumer |
| Memory integrity proofs | **DEFERRED** | No current consumer |
| Sandboxing | **DEFERRED** | No current consumer |

---

## AegisConfig Design

### Struct

```sigil
Σ AegisConfig { deny_kinds: Array }
```

`deny_kinds` is a flat Array of String message kind names that are rejected when
received from external socket clients. Empty by default — all kinds permitted.

### Constructor

```sigil
rite AegisConfig·default() -> AegisConfig
```

Returns an `AegisConfig` with `deny_kinds: []`. All message kinds are permitted.
Callers can push kind names into `deny_kinds` before passing to functions.

---

## aegis_check_kind

```sigil
rite aegis_check_kind(config, kind) -> Bool
```

Returns `true` if `kind` is permitted by the policy, `false` if it is in
`config.deny_kinds`. Linear scan (O(n)); `deny_kinds` list is expected to be
short (< 20 entries).

**Behaviour invariants:**
- Returns `true` for any kind when `deny_kinds` is empty
- Returns `false` for any kind present in `deny_kinds` (string equality)
- Returns `true` for any kind not in `deny_kinds`, regardless of list length
- Does not mutate `config`
- `null` kind returns `true` (nil-safety; kind is validated separately)

---

## Morgoth Call-Site Changes

### mq.sg — mq_process_external

```sigil
// Add import:
invoke tome·aegis·{AegisConfig, aegis_check_kind};

// mq_process_external gains aegis_cfg parameter:
rite mq_process_external(commune, panes, sock_fd, aegis_cfg) {
    // ...
    // Before dispatching task_add or any kind:
    ⎇ aegis_check_kind(aegis_cfg, kind) == false {
        Sys·close(client);
        ↩ null;
    }
    // ... rest of dispatch
}
```

### main.sg

```sigil
// Create aegis config at startup:
≔ aegis_cfg = AegisConfig·default();
// future: push(aegis_cfg.deny_kinds, "shell_exec");

// Pass to mq_process_external:
mq_process_external(commune, panes, mq_sock_fd, aegis_cfg);
```

---

## Implementation Location

Following the bridge pattern:

- **Library**: `sigil-lang/aegis/src/aegis_core.sg` — plain executable Sigil
- **Bridge**: `morgoth/src/aegis.sg` — verbatim copy of aegis_core.sg
- **Tests**: `sigil-lang/aegis/tests/aegis_core_test.sg` + `.expected`

Existing aspirational `src/aegis.sg`, `src/types.sg`, `src/lib.sg` are untouched.

---

## Test Requirements

6 Agent-TDD tests:

| # | Test | Requirement |
|---|------|-------------|
| 1 | `test_default_empty` | `AegisConfig·default()` has empty deny_kinds |
| 2 | `test_allow_all_empty` | All kinds pass when deny_kinds is empty |
| 3 | `test_deny_listed_kind` | A listed kind is rejected |
| 4 | `test_allow_unlisted_kind` | An unlisted kind passes when others are denied |
| 5 | `test_deny_multiple_kinds` | Multiple denied kinds all rejected |
| 6 | `test_null_kind_allowed` | null kind returns true (nil-safety) |

---

## Deferred

- Socket chmod (0600) — add to `setup_mq_socket` when `Sys·chmod` is available
- Peer UID verification via `SO_PEERCRED`
- `Constitution` / `Directive` declarative policy types
- `AegisContext` — holds config + audit log + identity
- Full aegis from aspirational `src/aegis.sg`
