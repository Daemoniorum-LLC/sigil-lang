# Sigil capability probes

Reproductions for the defects in `../LARES-SIGIL-REBUILD-SPEC.md` §8.1.2.
Run against a compiler built with:

    cd sigil-lang/parser
    cargo build --release --no-default-features --features jit,native

(The default build needs `llvm-c` dev headers; see spec §8.3.)

    export S=.../parser/target/release/sigil

## S1 — TcpListener·bind opens no socket, but reports success

    $S run S1_tcplistener_stub.sigil &
    ss -ltn | grep 18100                              # expect: a listener. actual: nothing
    exec 3<>/dev/tcp/127.0.0.1/18100                  # actual: Connection refused

Observed: prints `[Sigil] TcpListener bound to 127.0.0.1:18100 (id=1)`, then
`accept` fails with `accept requires TcpListener` — `bind` does not return a listener.

Blocks: HTTP server, WebSocket server, MCP endpoint, any Postgres wire protocol.

## S2 — PTY I/O hangs

    $S run S2_pty_open_ok.sigil                       # PASSES: fds 3 and 4
    timeout 20 $S run <sigil-lang>/jormungandr/tests/spec/22_native_runtime/P1_065_pty.sg

`Pty·open` alone is fine. The read/write round-trip hangs indefinitely and takes the
whole test suite with it.

Blocks: the live terminal, and any CI that runs the native-runtime suite.

## S3 — stdlib and FFI live in different backends

    $S jit S3_ffi_backend_split.sigil                 # works: two real PIDs via libc
    $S run S3_ffi_backend_split.sigil                 # fails: undefined variable getpid

Conversely `Pty·open` and `println` work under `run` and are unknown under `jit`.
So FFI cannot be used to route around S1/S2 while still using the rest of Sigil.
