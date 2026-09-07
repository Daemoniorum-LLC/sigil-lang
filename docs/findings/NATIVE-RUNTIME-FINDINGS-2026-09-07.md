# Native Runtime Findings — 2026-09-07

**Source:** capability probe for a Lares rebuild on Sigil/Qliphoth.
Full context: `lares` repo, `docs/specs/LARES-SIGIL-REBUILD-SPEC.md` §8.
**Reproductions:** `./lares-probe-2026-09-07/`

The probe asked one question: can Sigil host a real server application — HTTP routes, a
WebSocket terminal, spawned processes, a database? Three defects block it. All were found
by running code; none were visible from reading `stdlib.rs`.

Build used (the default build needs `llvm-c` dev headers, which were unavailable):

```
cargo build --release --no-default-features --features jit,native
```

---

## S1 — `TcpListener·bind` opens no socket, but reports success

**Severity: high.** This is the one to fix first.

```sigil
rite main() {
    ≔ listener = TcpListener·bind("127.0.0.1:18100");
    println("listening");
    ≔ conn = TcpListener·accept(listener);
}
```

```
[Sigil] TcpListener bound to 127.0.0.1:18100 (id=1)
listening
Runtime error: [R0000] accept requires TcpListener
```

Verified from outside the process:

```
$ ss -ltn | grep 18100                 → nothing
$ exec 3<>/dev/tcp/127.0.0.1/18100     → Connection refused
```

`bind` prints a success line, increments an id counter, opens no socket, and returns
something that is not a `TcpListener` — which is why `accept` rejects its own argument.

`Sys·listen`/`Sys·accept` fail too: `P1_042_socket_server` reports
`bind: OK / listen: FAILED / accept: FAILED`.

**Why this ranks above a missing feature.** A missing feature fails honestly and is found
in minutes. This one reports success, so it is found only when a client is refused —
typically after a routing layer has been built on top of it. It also makes every other
networking symbol suspect: `HttpClient` is currently assumed to work on the strength of
its name alone.

**Blocks:** HTTP server, WebSocket server, MCP endpoint, any SQL wire protocol — i.e. the
entire server side.

---

## S2 — PTY I/O hangs indefinitely

`Pty·open()` is **fine**: returns distinct master/slave fds (3 and 4).

The read/write round-trip hangs. `P1_065_pty` — whose own header calls PTY *"the most
critical primitive for Morgoth"* — never returns. It held the suite for eight minutes
without producing a total, so the 745/749 figure in `CLAUDE.md` could not be reproduced.

**Blocks:** any terminal or interactive-process feature, and any CI that runs
`22_native_runtime`. A hang is worse than a failure here — it takes the runner with it.

**Suggested first step:** a timeout on the test, so the suite fails fast rather than
hanging, independent of the underlying fix.

---

## S3 — stdlib and FFI live in different backends

| | `sigil run` (interpreter) | `sigil jit` (Cranelift) |
|---|---|---|
| `extern "C"` → libc | ✗ `undefined variable` (exit 1) | ✅ real — `getpid`/`getppid` return true PIDs |
| `Pty·open` | ✅ fds 3, 4 | ✗ `Unknown function: open` |
| `println` | ✅ | ✗ `Unknown function: println` |
| `Sys·*` | ✅ | ✗ largely absent |

The interpreter has the stdlib but no FFI. The JIT has FFI but not the stdlib.

**Consequence:** FFI is not a workaround for S1/S2. You cannot sit in the JIT backend,
bind libc `bind`/`listen`, and still use the rest of the language. S1 and S2 have to be
fixed in the runtime.

`sigil compile` (AOT) might combine both, but it requires the LLVM feature, which did not
build in this environment — untested.

---

## S4 — WebSocket echo round-trip returns `true`, not the payload

`spec/15_protocols/P0_003_websocket_real.sg`, marked **P0 — Bootstrap Critical**:

```sigil
≔ conn = "wss://ws.postman-echo.com/raw"|connect;
≔ response! = conn|send{"Hello from Sigil!"};
println(response!);
```

Expected `Hello from Sigil!`. Actual: `true`.

**Checked first: this is not the sandbox.** `curl https://ws.postman-echo.com/raw` returns
HTTP 404 from the real host — reachable, just wanting a WS upgrade rather than a GET. So
outbound networking works and the failure is ours.

Same shape as S1: a networking call returning a success value instead of doing the work.
Whether `connect`/`send` are stubbed or the test tracks an older API is unresolved, but
**two instances make it a pattern worth sweeping for** — see priority 3 below.

This also puts `HttpClient` in question. Nothing here has run it; it is assumed to work on
the strength of its name.

---

## S5 — parser rejects `·`-qualified paths in type positions ✅ FIXED

**Fixed 2026-09-07** in `parser/src/parser.rs`. Details at the end of this section.

`sigil check` across all 39 Qliphoth sources: **23 pass, 16 fail** — every failure the
same parse error.

```sigil
≔ m = std·collections·HashMap·new();                      // ✅ expression position
rite f(x: &std·collections·HashMap<String, String>) {}    // ❌ E0002 expected RParen
☉ Σ S { ☉ state: Option[serde_json·Value]? }              // ❌ E0002 expected RBracket
☉ rite g[T: serde·de·DeserializeOwned]() {}               // ❌ E0002 expected RBracket
```

The **expression** parser handles `·` paths. The **type** parser does not. Every failure
is a type position — parameter type, type argument, or generic bound.

**What it takes out:** the entire `qliphoth-router` package, plus `qliphoth-sys`'s
`storage`, `timers`, `websocket`, `closure` and `history`. Real instances:

| File | Line | Source |
|---|---|---|
| `qliphoth-sys/src/storage.sigil` | 32 | `☉ rite get_json[T: serde·de·DeserializeOwned](...)` |
| `qliphoth-router/src/router.sigil` | 40 | `☉ state: Option[serde_json·Value]?` |
| `qliphoth/src/core/mod.sigil` | 136 | `rite render_attrs(attrs: &std·collections·HashMap<String, String>)` |

**Beyond Qliphoth:** `·` is presented as *the* native path separator, so this hits any
code using qualified types in signatures — most non-trivial code. The 23 files that pass
do so by keeping qualified paths out of type positions.

It may also explain why `apps/wraith` has a `src.old/` sitting beside its current `src/`.

### The fix

`parse_type_path` was shared between two grammars with only a heuristic to separate them:

```rust
let first_segment_is_type = segments.first()
    .map(|s| s.ident.name.chars().next().map_or(false, |c| c.is_uppercase()))
```

That heuristic is **correct for expressions** — `HashMap·new()` is a path, `tag·to_string()`
is a method call for `parse_postfix_expr`, and case is the only distinguishing signal. It is
**wrong for types**, where `·` is unambiguous because types have no method calls. So
`std·collections·HashMap` stopped parsing at `std`.

Fixed by passing context rather than guessing:

```rust
let is_path_sep =
    (in_type_context || first_segment_is_type) && self.consume_if(&Token::MiddleDot);
```

Seven call sites updated: the two in `parse_type_base` pass `true`; the five in
`parse_primary_expr` / `parse_const_expr_primary` / `parse_macro_invocation` pass `false`,
preserving expression behaviour exactly. 30 insertions, 9 deletions, one file.

### Verification

| Check | Result |
|---|---|
| The three S5 reproductions | all now parse ✅ |
| `tag·to_string()` still a method call | ✅ parses and executes |
| `Vec·new()` / `HashMap·new()` paths | ✅ unchanged |
| Qualified return types | ✅ |
| Qliphoth sweep | **23 → 31 passing**, 16 → 8 failing |
| Full test suite | **713 pass / 11 fail — identical to baseline, no regressions** |

Regression test added: `jormungandr/tests/spec/02_syntax/P1_021_qualified_path_type_position.sg`,
pinning both halves — qualified paths in type positions *and* lowercase method calls.

### What remains — S6 to S11

The S5 fix let 8 more files reach parsing, revealing **six further defects**. Each is
reduced to a minimal reproduction in `./lares-probe-2026-09-07/s6-s11/`.

Three of my first hypotheses about these were **wrong** — glob imports, basic raw strings
and qualified struct literals all turned out to work fine, and the real causes were
narrower. Worth stating because the error messages consistently pointed one level away
from the actual cause.

| ID | Defect | Reproduction | Error |
|---|---|---|---|
| `S6` | ~~`☉ tome <name>;`~~ | — | **NOT A COMPILER BUG.** `tome` is *crate*; *mod* is `scroll`. `☉ scroll tokenizer;` checks clean. Athame's to fix. |
| `S7` | ✅ **FIXED** — `·ed`/`·ing` swallowed path-segment heads | `invoke super·editor·{A, B};` | Was far broader than `super·`: any segment starting "ed"/"ing" — `·editor`, `·edit`, `·ingest`, `·index`. |
| `S8` | Multi-hash raw strings `r##"…"##` | see repro | Open. `expected LBracket, found Hash` |
| `S9` | ✅ **FIXED** — keyword path roots parsed as variables | `crate·router·Opts { … }` | Was misdiagnosed: it failed in *every* position, not just as a call argument. See below. |
| `S10` | ✅ **FIXED** — `≠` `≤` `≥` absent from the lexer | `⎇ a ≠ 2 { }` | Was far worse than a parse error. See below. |
| `S11` | Evidentiality not narrowed through `unwrap_or` | `len() -> usize!` ending `·unwrap_or(0)` | Open. Needs an owner's judgment. |
| `S12` | Unknown characters are silently discarded | a file of `≤ § ⌘` | Open. `sigil check` reports **no errors**. Root cause of S10's silence. |
| `S13` | Lowercase module roots are treated as variables | `std·collections·HashMap·new()` | Open. Runtime: *undefined variable: `std`*. Same root as S9, but unfixable by enumerating keywords. |

### S10 was silent, not loud

`≠` produced **no token at all**. `Lexer::read_next` skips anything logos rejects:

```rust
Some(Err(_)) => { /* Skip invalid tokens and try next */ self.read_next() }
```

So `a ≠ 2` lexed as `a 2` and evaluated to `a` — `1 ≠ 2` and `1 ≠ 1` both printed `1`.
The condition-position parse error was a downstream symptom, which is why the first
diagnosis ("fails in conditions, works in assignments") was wrong in the more dangerous
direction: the assignment case was silently computing the wrong answer.

`S12` is that skip, generalised: **any** unrecognised character vanishes without a
diagnostic, so a typo in a symbol silently changes program meaning. Left open deliberately
— making it fatal is the right call but is a behavioural change that belongs to whoever
owns the language.

### S9 was misdiagnosed, and S13 is what remains

`crate·router·Opts { … }` was reported as "works in an assignment, fails as a call
argument". It never worked: `sigil check` passed and the runtime then said *undefined
variable: `crate`*. `parse_type_path` infers path-vs-method from the first segment's
**case**, and `crate` is lowercase.

Fixed for `tome`/`crate`/`super`, which are keywords and can never be values. `self` is
excluded on purpose — `self·field` is a real field access.

But the general case remains: `std·collections·HashMap·new()` still fails, because any
module name may be lowercase and the case heuristic cannot distinguish
`module·path` from `variable·method`. That is `S13`, and enumerating keywords will not
close it — it needs resolution against known modules, or a syntactic distinction.

### Verification of all three fixes

| Check | Result |
|---|---|
| Full suite | **714 pass / 11 fail** vs 713/11 baseline; failure set identical. The +1 is the S5 test. |
| Qliphoth sweep | 31/39 → **32/39** |
| `params.sigil`, `native.sigil` | now fail *later* — evidentiality, and deprecated `&mut` — not at S9/S10 |
| `self·field`, `Vec·new()`, lowercase method calls | unchanged |

Tests added: `P1_022_native_comparison_symbols` (pins ≠ ≤ ≥ *semantics*, not just
parsing) and `P1_023_aspect_marker_path_boundary` (pins `·edit`/`·ingest`/`·index`).

**Each of S9 and S10 works in a neighbouring position and fails in one specific context:**

```sigil
≔ b = a ≠ 2;                               // ✅   ⎇ a ≠ 2 { }                    // ❌ S10
⎇ a != 2 { }                               // ✅   (ASCII form is fine)
≔ o = crate·router·Opts { replace: true }; // ✅   go("/x", crate·router·Opts{…}) // ❌ S9
```

So both fixes should be scoped to the failing context, not the construct.

**S6, S7 and S10 are all native vocabulary** — `tome`, `super·`, `≠` — which the language
presents as first-class. `S10` is the sharpest: `≠` parses in an assignment but not in an
`⎇` condition, which is exactly where a writer reaches for it.

**S8 note:** Qliphoth's `playground.sigil` embeds sample code containing
`"#preview-target"`. The `"#` correctly closes a `r#"…"#` literal early; the proper fix is
`r##"…"##`, which the lexer does not support. The Qliphoth code is *unwritable*, not
merely wrong.

**S11 needs an owner's judgment,** not a bug report from me. `unwrap_or` turns an uncertain
value plus a default into a definite one, so `usize!` reads as correct and the checker
looks like it is not narrowing `?` → `!` through it. But that is intent-reading, not
demonstration. `route.sigil` and `guards.sigil` fail in the same family.

Full reproduction of the original defect: `./S5-middledot-type-position.md`.

---

## Test suite status

Run 2026-09-07, minimal build: **713 pass, 11 fail**, then a hard hang on `P1_065_pty`
until a 900s timeout killed it.

That is broadly consistent with the 745/749 in `CLAUDE.md` — the harness is real and most
of the language works. Of the 11: 4 are the Kafka/AMQP broker tests already documented as
infrastructure-dependent, 5 are `test_stdlib_*` runtime errors possibly tied to the
minimal feature set, and 2 are genuine (`P1_042_socket_server` = S1,
`P0_003_websocket_real` = S4).

**The issue is not the pass rate — it is that no total can be produced at all** while one
case hangs. That blocks CI as surely as a failure would. A per-test timeout in
`run_tests_rust.sh` is worth adding regardless of when S2 is fixed.

---

## What is already good

The probe expected to find a thin POSIX layer. It found a broad one, and most of it
passes:

`Sys·spawn` · `spawn_bg` · `spawn_pty` · `clone` · `waitpid` · `kill` · `getpid` ·
`gettid` · `open` · `read` · `read_string` · `write` · `close` · `pipe` · `dup2` ·
`epoll_create1` · `epoll_ctl` · `epoll_wait` · `poll_fd` · `poll_fds` ·
`signal_register` · `signal_pending` · `signal_send` · `mmap` · `munmap` · `futex` ·
`getenv` · `socket` · `setsockopt` · terminal mode-setting · `Pty·open` · JSON

`P1_064_fork_exec`, `P1_051_epoll`, `P1_052_mutex`, `P1_063_pipe_dup2`,
`P1_060/061/062_term_*` all pass.

**Fix S1 and S2 and a Sigil-hosted server application becomes realistic**, because
everything else it needs is already here. That is a much better position than the symbol
list suggests in either direction — the gaps are narrower than feared, and the ones that
exist are invisible to inspection.

---

## Suggested priority

1. **S1** — unblocks the whole server story. Still the top item.
2. **S12** — silent character-dropping. Cheap to fix, and it is the reason S10 went
   unnoticed. Every future symbol gap will be silent until this is closed.
3. **S13** — lowercase module roots. Needs a design decision, not a patch.
4. **S2** — unblocks terminals, and stops the suite hanging.
5. **S8**, **S11** — smaller; S11 needs an owner's judgment first.
6. ~~S5, S7, S9, S10~~ ✅ fixed. Qliphoth 23/39 → 32/39.
4. **Sweep for other stubs.** S1 and S4 both show
   a symbol that exists, exports, reports success and does not do the work. Two is a
   pattern. A behavioural smoke test per stdlib module would be cheap and would likely
   find more. `HttpClient` first: nothing has run it, and it is widely assumed to work.

5. **Add a per-test timeout to `run_tests_rust.sh`.** Independent of S2, and it restores
   the suite's ability to report a total.
6. **S3** — may be acceptable as documented behaviour rather than a fix, but it should be
   written down either way; today it is discovered by surprise.
