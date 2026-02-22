# Engram Library — Software Design Document (v0.2)

**Scope:** Minimum viable FileBackend for Morgoth session.sg + config.sg
integration (Phase C1 of the Morgoth module split plan).

**Supersedes:** `architecture.md` and `api-reference.md` for the purposes of
this implementation. Full architecture remains aspirational specification; this
SDD governs what is actually built and tested.

---

## Context and Motivation

The engram library is designed as a full cognitive memory system for AI agents
(Instant/Episodic/Semantic/Procedural memory, HNSW indexing, CRDT sync,
embeddings). The library files use a typed Sigil dialect with Rust-like generics,
traits, and pattern matching that cannot be executed by Morgoth's test suite.

Morgoth's current persistence layer in `session.sg` and `config.sg` uses
ad-hoc `fs_write`/`fs_read`/`fs_exists`/`fs_list` calls with manually constructed
paths. The goal of Phase C1 is to replace that with a clean, testable
`FileBackend` abstraction that:

1. Centralizes path construction and directory management
2. Provides a uniform key→file mapping (`"session"` → `session.json`)
3. Establishes the engram integration pattern for future EpisodicMemory adoption
4. Can be tested in isolation without Morgoth's full runtime

---

## Scope Decision Table

| Subsystem | Status | Rationale |
|-----------|--------|-----------|
| `FileBackend` struct + CRUD methods | **IN SCOPE** | Direct Morgoth need |
| Directory auto-creation on `new()` | **IN SCOPE** | Eliminates scattered `fs_mkdir` guards |
| `list(prefix)` for profile enumeration | **IN SCOPE** | Replaces `fs_list` + manual `.json` stripping in `list_profiles()` |
| `delete(key)` | **IN SCOPE** | Clean API completeness |
| `EpisodicMemory` | **DEFERRED** | No current consumer beyond a wrapper pattern |
| `InstantMemory` | **DEFERRED** | No current consumer |
| `SemanticMemory` | **DEFERRED** | No current consumer |
| `ProceduralMemory` | **DEFERRED** | No current consumer |
| `Engram` (unified interface) | **DEFERRED** | Overkill for file I/O |
| `EngramConfig` | **DEFERRED** | No configuration variation needed |
| HNSW indexing | **DEFERRED** | No embedding/search use case |
| CRDT sync | **DEFERRED** | No distributed use case |
| Epistemic types | **DEFERRED** | Not relevant to file storage |
| `MemoryBackend` (in-memory) | **DEFERRED** | Tests use real tmp files |
| `AppendLogBackend` | **DEFERRED** | Overkill for small JSON files |

---

## FileBackend Design

### Struct

```sigil
Σ FileBackend { base_dir: String }
```

### Key-to-Path Mapping

A key is a simple string identifier (`"session"`, `"default"`, `"myconfig"`).
Keys map to files in `base_dir` by appending `.json`:

```
key "session"  → {base_dir}/session.json
key "default"  → {base_dir}/default.json
key "myconfig" → {base_dir}/myconfig.json
```

Keys should not contain path separators. Each FileBackend instance owns one
directory — callers use separate instances for different directories:

```sigil
// Session data
≔ fb_session  = FileBackend·new(home + "/.morgoth");

// Profile data
≔ fb_profiles = FileBackend·new(home + "/.morgoth/profiles");
```

### API

```sigil
// Constructor — creates base_dir if it does not exist
rite new(dir) -> FileBackend

// Persist json_str under key
rite save(self, key, json_str)

// Retrieve content for key; returns null if key does not exist
rite load(self, key) -> String | null

// Returns true if key has been saved
rite exists(self, key) -> Bool

// Remove the file for key (no-op if key does not exist)
rite delete(self, key)

// Return all keys in base_dir whose filename starts with prefix.
// Pass empty string "" to list all keys.
// Keys are returned without the .json suffix.
rite list(self, prefix) -> Array
```

### Behaviour Invariants

- `new(dir)` MUST create `dir` (one level) if it does not exist.
  After `new()` returns, `fs_exists(dir)` is true. Parent directories must
  already exist; `fs_mkdir` does not recurse.
- `save(key, str)` then `load(key)` returns exactly `str` for any non-empty
  `str`. Saving an empty string is treated as absent: a subsequent `load`
  returns `null`. This is intentional for a JSON store where empty files
  indicate corruption.
- `load(key)` returns `null` when the key has never been saved or was deleted.
- `exists(key)` returns `false` before first save and after delete; `true` otherwise.
- `list("")` returns all keys. `list("pre")` returns only keys whose names start with `"pre"`.
- `save(key, str)` on an existing key overwrites silently.
- `delete(key)` on a nonexistent key is a no-op (no error).
- Multiple keys in the same FileBackend do not interfere with one another.

---

## Morgoth Call-Site Changes

### session.sg — `save_session` / `load_session`

```sigil
// Before: manual path construction + fs_write/fs_read
rite save_session(panes) {
    ≔ home = env("HOME");
    ≔ dir_path = home + "/.morgoth";
    ⎇ fs_exists(dir_path) == false { fs_mkdir(dir_path); }
    ...
    fs_write(dir_path + "/session.json", json);
}

// After: FileBackend handles dir creation and path construction
rite save_session(panes) {
    ≔ fb = FileBackend·new(env("HOME") + "/.morgoth");
    ...
    fb·save("session", json);
}

// Before:
rite load_session() {
    ≔ session_path = home + "/.morgoth/session.json";
    ⎇ fs_exists(session_path) == false { ↩ null }
    ≔ raw = fs_read(session_path);
    ...
}

// After:
rite load_session() {
    ≔ fb = FileBackend·new(env("HOME") + "/.morgoth");
    ≔ raw = fb·load("session");
    ⎇ raw == null { ↩ null }
    ...
}
```

### config.sg — profile functions

```sigil
// Before: manual path + mkdir guard on each function
rite save_profile(panes) {
    ≔ dir_path = home + "/.morgoth/profiles";
    ⎇ fs_exists(dir_path) == false { fs_mkdir(dir_path); }
    fs_write(dir_path + "/default.json", json);
}

// After: FileBackend handles everything
rite save_profile(panes) {
    ≔ fb = FileBackend·new(env("HOME") + "/.morgoth/profiles");
    fb·save("default", json);
}

// list_profiles() before: fs_list + manual .json strip
rite list_profiles() {
    ≔ files = fs_list(home + "/.morgoth/profiles");
    // iterate, filter .json, strip extension...
}

// After: FileBackend·list replaces the whole loop
rite list_profiles() {
    ≔ fb = FileBackend·new(env("HOME") + "/.morgoth/profiles");
    ≔ names = fb·list("");
    ⎇ len(names) == 0 { push(names, "default"); }
    ↩ names
}
```

---

## Implementation Location

Following the daemon bridge pattern:

- **Library**: `sigil-lang/engram/src/file_backend.sg` — plain executable Sigil
  (`rite`, `≔`, `⎇`, `⟳`; no `☉`/`⌥`/generics)
- **Bridge**: `morgoth/src/engram.sg` — verbatim copy of file_backend.sg loaded
  via `invoke tome·engram·{FileBackend}`
- **Tests**: `sigil-lang/engram/tests/file_backend_test.sg` +
  `file_backend_test.expected`

The existing aspirational library files (`engram.sg`, `types.sg`, `storage/mod.sg`,
etc.) are untouched — they remain as the full spec target.

---

## Test Requirements

8 Agent-TDD tests, all written before production code is touched:

| # | Test | Requirement |
|---|------|-------------|
| 1 | `test_new_creates_dir` | `new(nonexistent_dir)` creates the directory |
| 2 | `test_save_load_roundtrip` | `save(k, s)` then `load(k)` returns `s` exactly |
| 3 | `test_load_missing_returns_null` | `load(k)` returns null when k was never saved |
| 4 | `test_exists_lifecycle` | `exists` false before save, true after, false after delete |
| 5 | `test_delete_removes` | `delete(k)` removes file; subsequent `load` returns null |
| 6 | `test_list_prefix` | `list("")` returns all keys; `list("pre")` filters by prefix |
| 7 | `test_multiple_keys_isolated` | saving key A does not affect key B |
| 8 | `test_overwrite` | second `save(k, s2)` makes `load(k)` return `s2`, not `s1` |

---

## Deferred

- `EpisodicMemory` wrapper over FileBackend (session events as episodes)
- `Query·recent()·with_tag()` recall API
- `MemoryBackend` (in-memory storage for tests without filesystem)
- HNSW, embeddings, CRDT sync, epistemic types
- `Engram` unified interface
