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

1. **S1** — unblocks the whole server story.
2. **S2** — unblocks terminals, and stops the suite hanging.
3. **Sweep for other stubs.** S1 proves a symbol can exist, export, print success and do
   nothing. A behavioural smoke test per stdlib module would be cheap and would probably
   find more. `HttpClient` first — it is widely assumed to work.
4. **S3** — may be acceptable as documented behaviour rather than a fix, but it should be
   written down either way; today it is discovered by surprise.
