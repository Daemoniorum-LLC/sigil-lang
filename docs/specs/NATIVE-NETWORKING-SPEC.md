# Sigil Native Networking Stack Specification

**Version:** 0.3.0
**Status:** Approved (with prerequisites)
**Authors:** Claude + Human
**Date:** 2026-01-20

---

## 1. Executive Summary

This specification defines a **zero-C-dependency networking stack** for the Sigil programming language. The stack provides native HTTP and WebSocket clients implemented entirely in Sigil, with only minimal inline assembly for OS syscalls.

### 1.1 Goals

1. **Zero C Runtime Dependency** - Compiled Sigil binaries require no libc
2. **Native Protocol Implementation** - HTTP/1.1 and WebSocket in pure Sigil
3. **Multi-Platform Support** - Linux x86_64 (primary), macOS x86_64, Linux ARM64
4. **Production Quality** - Suitable for real-world use, not just demos
5. **Educational Value** - Clear, readable implementation that teaches networking

### 1.2 Non-Goals

1. HTTP/2, HTTP/3, QUIC (future work)
2. Full TLS implementation in Sigil (will use OS-provided TLS or FFI initially)
3. Windows support in v1.0 (different syscall model)
4. Async I/O (blocking I/O first, async later)

---

## 2. Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    Application Layer                         │
│                  (User Sigil Code)                          │
├─────────────────────────────────────────────────────────────┤
│                    Protocol Layer                            │
│              ┌─────────────┬─────────────┐                  │
│              │  HTTP/1.1   │  WebSocket  │                  │
│              │  (http.sg)  │   (ws.sg)   │                  │
│              └─────────────┴─────────────┘                  │
├─────────────────────────────────────────────────────────────┤
│                     TLS Layer                                │
│                    (tls.sg)                                  │
│         ┌────────────────┬────────────────┐                 │
│         │ OpenSSL FFI    │ Security.fwk   │                 │
│         │ (Linux)        │ (macOS)        │                 │
│         └────────────────┴────────────────┘                 │
├─────────────────────────────────────────────────────────────┤
│                     DNS Layer                                │
│                    (dns.sg)                                  │
├─────────────────────────────────────────────────────────────┤
│                    Socket Layer                              │
│                   (socket.sg)                                │
├─────────────────────────────────────────────────────────────┤
│                    Syscall Layer                             │
│    ┌─────────────┬─────────────┬─────────────┐              │
│    │ Linux x86_64│ macOS x86_64│ Linux ARM64 │              │
│    │(sys/linux/) │(sys/darwin/)│(sys/linux/) │              │
│    └─────────────┴─────────────┴─────────────┘              │
├─────────────────────────────────────────────────────────────┤
│                    LLVM Backend                              │
│              (Inline Assembly Support)                       │
└─────────────────────────────────────────────────────────────┘

                    ┌─────────────────────┐
                    │   Haagenti 2.0      │
                    │   (Optional)        │
                    │   ───────────────   │
                    │   • gzip/deflate    │
                    │   • lz4, zstd       │
                    │   • brotli          │
                    └─────────────────────┘
```

### 2.1 Layer Responsibilities

| Layer | Language | Responsibility |
|-------|----------|----------------|
| Syscall | Sigil + inline asm | Raw OS interface via `syscall` instruction |
| Socket | Pure Sigil | BSD socket abstraction (connect, send, recv) |
| TLS | Sigil + FFI | TLS 1.2/1.3 via OpenSSL/Security.framework |
| DNS | Pure Sigil | DNS protocol over UDP, hostname resolution |
| HTTP | Pure Sigil | HTTP/1.1 request/response, headers, chunking, keep-alive, proxy |
| WebSocket | Pure Sigil | WS handshake, framing, ping/pong |
| Haagenti | Pure Sigil (optional) | Compression: gzip, deflate, lz4, zstd, brotli |

---

## 3. Compiler Prerequisites

### 3.1 Inline Assembly Support

**Status:** ⚠️ **GAP IDENTIFIED** - Parser supports `asm!`, backends do not implement it.

The syscall layer requires inline assembly support in the compiler. This is a **blocking prerequisite** before the networking stack can be implemented in pure Sigil.

#### 3.1.1 Current State

| Component | `asm!` Support | Status |
|-----------|----------------|--------|
| Parser (`parser.rs`) | ✅ Full syntax | `parse_inline_asm()` at line 5091 |
| AST (`ast.rs`) | ✅ `InlineAsm` type | `Expr::InlineAsm` with operands, clobbers, options |
| Interpreter | ❌ Not implemented | Falls through to error |
| Cranelift JIT | ❌ Not implemented | No `InlineAsm` handling |
| LLVM AOT | ❌ Not implemented | Falls to catch-all error at line 1844 |
| WASM | ❌ Explicitly unsupported | Returns `WasmError::unsupported` |

#### 3.1.2 Required Implementation

The **LLVM backend** (`llvm_codegen.rs`) must implement `Expr::InlineAsm` handling:

```rust
// In compile_expr match block, add before the catch-all:
Expr::InlineAsm(asm) => {
    self.compile_inline_asm(fn_value, scope, asm)
}

fn compile_inline_asm(
    &mut self,
    fn_value: FunctionValue<'ctx>,
    scope: &mut HashMap<String, PointerValue<'ctx>>,
    asm: &InlineAsm,
) -> Result<IntValue<'ctx>, String> {
    // Use inkwell's inline assembly support:
    // - Build constraint string from asm.outputs, asm.inputs, asm.clobbers
    // - Call context.create_inline_asm() with template and constraints
    // - Handle volatile/sideeffects flags from asm.options
}
```

LLVM (via inkwell) supports inline assembly through:
- `context.create_inline_asm(fn_type, template, constraints, has_side_effects, is_align_stack)`
- Constraint format: `"=r,r,~{memory}"` (outputs, inputs, clobbers)

#### 3.1.3 Test Coverage Required

Before proceeding with syscall implementation, verify with:

```sigil
// tests/asm/basic_asm_test.sg
rite test_inline_asm_nop() {
    unsafe {
        asm!("nop", options(nostack));
    }
    assert_eq!(1, 1);  // Reached = asm executed
}

rite test_inline_asm_output() {
    ≔ result: i64 = 0;
    unsafe {
        asm!("mov {0}, 42",
            out("rax") result,
            options(nostack));
    }
    assert_eq!(result, 42);
}

rite test_syscall_getpid() {
    ≔ pid: i64 = 0;
    unsafe {
        asm!("syscall",
            inout("rax") 39_i64 => pid,  // getpid = 39
            out("rcx") _,
            out("r11") _,
            options(nostack));
    }
    assert!(pid > 0);
}
```

#### 3.1.4 Implementation Order

1. **Phase 0**: Implement `Expr::InlineAsm` in LLVM backend
2. **Phase 1**: Pass basic asm tests (nop, output registers)
3. **Phase 2**: Pass syscall test (getpid)
4. **Phase 3**: Proceed with syscall layer implementation

---

## 4. Syscall Layer Specification

### 4.1 Design Principles

1. **Minimal Surface** - Only syscalls needed for networking
2. **Raw Return Values** - Return kernel values directly, no translation
3. **Architecture-Specific Modules** - Separate file per OS/arch
4. **Compile-Time Selection** - Target selected at compile time

**Prerequisite:** Section 3.1 (Inline Assembly Support) must be complete before this layer.

### 4.2 Linux x86_64 Syscall ABI

```
Syscall Number: RAX
Arguments:      RDI, RSI, RDX, R10, R8, R9
Return Value:   RAX (negative = -errno)
Clobbered:      RCX, R11
```

### 4.3 Required Syscalls

| Syscall | Number | Signature | Purpose |
|---------|--------|-----------|---------|
| `read` | 0 | `(fd, buf, count) → ssize_t` | Read from fd |
| `write` | 1 | `(fd, buf, count) → ssize_t` | Write to fd |
| `close` | 3 | `(fd) → int` | Close fd |
| `socket` | 41 | `(domain, type, protocol) → int` | Create socket |
| `connect` | 42 | `(fd, addr, addrlen) → int` | Connect socket |
| `accept` | 43 | `(fd, addr, addrlen) → int` | Accept connection |
| `sendto` | 44 | `(fd, buf, len, flags, addr, addrlen) → ssize_t` | Send data |
| `recvfrom` | 45 | `(fd, buf, len, flags, addr, addrlen) → ssize_t` | Receive data |
| `bind` | 49 | `(fd, addr, addrlen) → int` | Bind socket |
| `listen` | 50 | `(fd, backlog) → int` | Listen for connections |
| `setsockopt` | 54 | `(fd, level, optname, optval, optlen) → int` | Set socket option |
| `getsockopt` | 55 | `(fd, level, optname, optval, optlen) → int` | Get socket option |
| `fcntl` | 72 | `(fd, cmd, arg) → int` | File control (for non-blocking) |
| `poll` | 7 | `(fds, nfds, timeout) → int` | Wait for events |

### 4.4 Sigil Syscall Syntax

```sigil
// Proposed syntax for raw syscall
☉ rite sys_read(fd: i32, buf: *u8, count: usize) → isize {
    asm!(
        "syscall",
        in("rax") 0,        // syscall number
        in("rdi") fd,
        in("rsi") buf,
        in("rdx") count,
        out("rax") result,
        clobber("rcx", "r11")
    )
}
```

### 4.5 Error Handling

Syscalls return negative values on error (Linux returns -errno).

```sigil
☉ enum SyscallError {
    EPERM           = 1,    // Operation not permitted
    ENOENT          = 2,    // No such file or directory
    EINTR           = 4,    // Interrupted system call
    EIO             = 5,    // I/O error
    EBADF           = 9,    // Bad file descriptor
    EAGAIN          = 11,   // Try again (also EWOULDBLOCK)
    ENOMEM          = 12,   // Out of memory
    EACCES          = 13,   // Permission denied
    EFAULT          = 14,   // Bad address
    EINVAL          = 22,   // Invalid argument
    EPIPE           = 32,   // Broken pipe
    ECONNREFUSED    = 111,  // Connection refused
    ECONNRESET      = 104,  // Connection reset by peer
    ETIMEDOUT       = 110,  // Connection timed out
    EHOSTUNREACH    = 113,  // No route to host
    ENETUNREACH     = 101,  // Network is unreachable
}

☉ rite syscall_result(raw: isize) → Result<usize, SyscallError>! {
    ⎇ raw >= 0 {
        Ok(raw as usize)
    } ⎉ {
        Err(SyscallError::from_errno(-raw as i32))
    }
}
```

### 4.6 Platform Detection

```sigil
// Compile-time platform detection
#[cfg(target_os = "linux", target_arch = "x86_64")]
use sys::linux::x86_64::*;

#[cfg(target_os = "macos", target_arch = "x86_64")]
use sys::darwin::x86_64::*;

#[cfg(target_os = "linux", target_arch = "aarch64")]
use sys::linux::aarch64::*;
```

---

## 5. Socket Layer Specification

### 5.1 Socket Types

```sigil
☉ enum AddressFamily {
    AF_INET  = 2,   // IPv4
    AF_INET6 = 10,  // IPv6
}

☉ enum SocketType {
    SOCK_STREAM = 1,  // TCP
    SOCK_DGRAM  = 2,  // UDP
}

☉ sigil SocketAddr {
    family: AddressFamily!,
    port: u16!,
    addr: [u8; 16]!,  // Holds IPv4 (4 bytes) or IPv6 (16 bytes)
}

☉ sigil TcpStream {
    fd: i32!,
    peer_addr: SocketAddr!,
    connected: bool!,
}

☉ sigil UdpSocket {
    fd: i32!,
    bound_addr: SocketAddr?,
}
```

### 5.2 TCP API

```sigil
☉ impl TcpStream {
    // Connect to remote host:port
    ☉ rite connect(host: &str, port: u16) → Result<TcpStream, SocketError>!;

    // Read up to `buf.len()` bytes, returns bytes read
    ☉ rite read(&vary self, buf: &vary [u8]) → Result<usize, SocketError>!;

    // Read exactly `buf.len()` bytes or error
    ☉ rite read_exact(&vary self, buf: &vary [u8]) → Result<(), SocketError>!;

    // Write all bytes from buf
    ☉ rite write_all(&vary self, buf: &[u8]) → Result<(), SocketError>!;

    // Close the connection
    ☉ rite close(&vary self) → Result<(), SocketError>!;

    // Set read timeout (0 = no timeout)
    ☉ rite set_read_timeout(&vary self, ms: u32) → Result<(), SocketError>!;

    // Set write timeout (0 = no timeout)
    ☉ rite set_write_timeout(&vary self, ms: u32) → Result<(), SocketError>!;
}
```

### 5.3 UDP API

```sigil
☉ impl UdpSocket {
    // Bind to local address
    ☉ rite bind(addr: &str, port: u16) → Result<UdpSocket, SocketError>!;

    // Send datagram to address
    ☉ rite send_to(&self, buf: &[u8], addr: &SocketAddr) → Result<usize, SocketError>!;

    // Receive datagram, returns (bytes_read, sender_addr)
    ☉ rite recv_from(&self, buf: &vary [u8]) → Result<(usize, SocketAddr), SocketError>!;

    // Close socket
    ☉ rite close(&vary self) → Result<(), SocketError>!;
}
```

### 5.4 Socket Errors

```sigil
☉ enum SocketError {
    ConnectionRefused!,
    ConnectionReset!,
    ConnectionTimedOut!,
    HostUnreachable!,
    NetworkUnreachable!,
    AddressInUse!,
    AddressNotAvailable!,
    InvalidAddress!,
    DnsResolutionFailed { hostname: String~ }!,
    Syscall { errno: i32~, message: String~ }!,
}
```

---

## 6. DNS Layer Specification

### 6.1 Overview

DNS resolution over UDP port 53. Implements enough of RFC 1035 for A and AAAA record lookups.

### 6.2 DNS API

```sigil
☉ sigil DnsResolver {
    servers: [SocketAddr]!,     // DNS server addresses
    timeout_ms: u32!,           // Query timeout
    max_retries: u8!,           // Retry count
}

☉ impl DnsResolver {
    // Create resolver with system DNS servers
    ☉ rite system() → Result<DnsResolver, DnsError>!;

    // Create resolver with specific servers
    ☉ rite new(servers: &[&str]) → Result<DnsResolver, DnsError>!;

    // Resolve hostname to IPv4 addresses
    ☉ rite resolve_a(&self, hostname: &str) → Result<[IpAddr], DnsError>!;

    // Resolve hostname to IPv6 addresses
    ☉ rite resolve_aaaa(&self, hostname: &str) → Result<[IpAddr], DnsError>!;

    // Resolve hostname to any IP (prefers IPv4)
    ☉ rite resolve(&self, hostname: &str) → Result<IpAddr, DnsError>!;
}
```

### 6.3 DNS Packet Format (RFC 1035)

```
+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+
|                      ID                       |  16 bits
+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+
|QR|   Opcode  |AA|TC|RD|RA|   Z    |   RCODE   |  16 bits
+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+
|                    QDCOUNT                    |  16 bits
+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+
|                    ANCOUNT                    |  16 bits
+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+
|                    NSCOUNT                    |  16 bits
+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+
|                    ARCOUNT                    |  16 bits
+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+
```

### 6.4 DNS Error Types

```sigil
☉ enum DnsError {
    NoServersConfigured!,
    QueryTimedOut!,
    ServerError { rcode: u8~ }!,
    NameNotFound!,
    MalformedResponse!,
    NetworkError { inner: SocketError~ }!,
}
```

---

## 7. HTTP/1.1 Layer Specification

### 7.1 Overview

HTTP/1.1 client implementation per RFC 7230-7235.

### 7.2 HTTP API

```sigil
☉ sigil HttpClient {
    timeout_ms: u32!,
    max_redirects: u8!,
    user_agent: String!,
}

☉ sigil HttpRequest {
    method: HttpMethod!,
    url: Url!,
    headers: [(String, String)]!,
    body: [u8]?,
}

☉ sigil HttpResponse {
    status: u16!,
    status_text: String!,
    headers: [(String, String)]!,
    body: [u8]!,
}

☉ enum HttpMethod {
    GET!,
    POST!,
    PUT!,
    DELETE!,
    PATCH!,
    HEAD!,
    OPTIONS!,
}

☉ impl HttpClient {
    ☉ rite new() → HttpClient!;

    // Simple GET request
    ☉ rite get(&self, url: &str) → Result<HttpResponse, HttpError>!;

    // Simple POST request
    ☉ rite post(&self, url: &str, body: &[u8]) → Result<HttpResponse, HttpError>!;

    // POST with JSON content type
    ☉ rite post_json(&self, url: &str, json: &str) → Result<HttpResponse, HttpError>!;

    // Full request with all options
    ☉ rite request(&self, req: &HttpRequest) → Result<HttpResponse, HttpError>!;
}
```

### 7.3 HTTP Request Format

```
GET /path HTTP/1.1\r\n
Host: example.com\r\n
User-Agent: Sigil/0.4.0\r\n
Accept: */*\r\n
Connection: close\r\n
\r\n
```

### 7.4 HTTP Response Parsing

Must handle:
- Status line parsing
- Header parsing (case-insensitive keys)
- Content-Length body
- Chunked Transfer-Encoding
- Connection: close

### 7.5 HTTP Errors

```sigil
☉ enum HttpError {
    InvalidUrl { url: String~ }!,
    DnsResolutionFailed { hostname: String~ }!,
    ConnectionFailed { inner: SocketError~ }!,
    RequestTimedOut!,
    TooManyRedirects!,
    InvalidResponse { message: String~ }!,
    ChunkedEncodingError!,
}
```

---

## 8. WebSocket Layer Specification

### 8.1 Overview

WebSocket client implementation per RFC 6455.

### 8.2 WebSocket API

```sigil
☉ sigil WebSocket {
    stream: TcpStream!,
    connected: bool!,
    mask_key: [u8; 4]!,
}

☉ enum WsMessage {
    Text(String)!,
    Binary([u8])!,
    Ping([u8])!,
    Pong([u8])!,
    Close { code: u16?, reason: String? }!,
}

☉ impl WebSocket {
    // Connect to WebSocket URL (ws:// or wss://)
    ☉ rite connect(url: &str) → Result<WebSocket, WsError>!;

    // Connect with custom headers (for auth, etc.)
    ☉ rite connect_with_headers(url: &str, headers: &[(String, String)]) → Result<WebSocket, WsError>!;

    // Send text message
    ☉ rite send_text(&vary self, text: &str) → Result<(), WsError>!;

    // Send binary message
    ☉ rite send_binary(&vary self, data: &[u8]) → Result<(), WsError>!;

    // Receive next message (blocks)
    ☉ rite recv(&vary self) → Result<WsMessage, WsError>!;

    // Send ping
    ☉ rite ping(&vary self, data: &[u8]) → Result<(), WsError>!;

    // Close connection gracefully
    ☉ rite close(&vary self, code: u16, reason: &str) → Result<(), WsError>!;
}
```

### 8.3 WebSocket Handshake

```
GET /path HTTP/1.1\r\n
Host: example.com\r\n
Upgrade: websocket\r\n
Connection: Upgrade\r\n
Sec-WebSocket-Key: dGhlIHNhbXBsZSBub25jZQ==\r\n
Sec-WebSocket-Version: 13\r\n
\r\n
```

Response:
```
HTTP/1.1 101 Switching Protocols\r\n
Upgrade: websocket\r\n
Connection: Upgrade\r\n
Sec-WebSocket-Accept: s3pPLMBiTxaQ9kYGzzhZRbK+xOo=\r\n
\r\n
```

### 8.4 WebSocket Frame Format

```
 0                   1                   2                   3
 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1
+-+-+-+-+-------+-+-------------+-------------------------------+
|F|R|R|R| opcode|M| Payload len |    Extended payload length    |
|I|S|S|S|  (4)  |A|     (7)     |             (16/64)           |
|N|V|V|V|       |S|             |   (if payload len==126/127)   |
| |1|2|3|       |K|             |                               |
+-+-+-+-+-------+-+-------------+ - - - - - - - - - - - - - - - +
|     Extended payload length continued, if payload len == 127  |
+ - - - - - - - - - - - - - - - +-------------------------------+
|                               |Masking-key, if MASK set to 1  |
+-------------------------------+-------------------------------+
| Masking-key (continued)       |          Payload Data         |
+-------------------------------- - - - - - - - - - - - - - - - +
:                     Payload Data continued ...                :
+ - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - +
|                     Payload Data continued ...                |
+---------------------------------------------------------------+
```

### 8.5 WebSocket Errors

```sigil
☉ enum WsError {
    InvalidUrl { url: String~ }!,
    ConnectionFailed { inner: SocketError~ }!,
    HandshakeFailed { status: u16~, message: String~ }!,
    InvalidFrame { message: String~ }!,
    ConnectionClosed { code: u16?, reason: String? }!,
    MessageTooLarge { size: usize~, max: usize~ }!,
}
```

---

## 9. TLS Layer Specification (v1.0 - OS FFI)

TLS support is included in v1.0 via FFI to OS-provided TLS libraries.

### 9.1 Platform TLS Backends

| Platform | Library | Notes |
|----------|---------|-------|
| Linux | OpenSSL / LibreSSL | Most common, well-documented |
| macOS | Security.framework | Native, no external deps |
| (Future) Windows | SChannel | Native Windows TLS |

### 9.2 TLS API

```sigil
☉ sigil TlsStream {
    inner: TcpStream!,
    ssl_ctx: *void!,      // Opaque pointer to SSL context
    ssl: *void!,          // Opaque pointer to SSL connection
}

☉ impl TlsStream {
    // Wrap a TCP connection with TLS (client mode)
    ☉ rite connect(stream: TcpStream, hostname: &str) → Result<TlsStream, TlsError>!;

    // Read decrypted data
    ☉ rite read(&vary self, buf: &vary [u8]) → Result<usize, TlsError>!;

    // Write data (will be encrypted)
    ☉ rite write_all(&vary self, buf: &[u8]) → Result<(), TlsError>!;

    // Graceful TLS shutdown
    ☉ rite close(&vary self) → Result<(), TlsError>!;
}
```

### 9.3 TLS Errors

```sigil
☉ enum TlsError {
    HandshakeFailed { message: String~ }!,
    CertificateError { message: String~ }!,
    ProtocolError { message: String~ }!,
    IoError { inner: SocketError~ }!,
}
```

### 9.4 OpenSSL FFI (Linux)

Required FFI functions:

```sigil
// OpenSSL initialization
extern "C" rite SSL_library_init() → i32;
extern "C" rite SSL_load_error_strings();
extern "C" rite TLS_client_method() → *void;

// Context management
extern "C" rite SSL_CTX_new(method: *void) → *void;
extern "C" rite SSL_CTX_free(ctx: *void);

// Connection management
extern "C" rite SSL_new(ctx: *void) → *void;
extern "C" rite SSL_set_fd(ssl: *void, fd: i32) → i32;
extern "C" rite SSL_set_tlsext_host_name(ssl: *void, name: *u8) → i32;
extern "C" rite SSL_connect(ssl: *void) → i32;
extern "C" rite SSL_read(ssl: *void, buf: *u8, num: i32) → i32;
extern "C" rite SSL_write(ssl: *void, buf: *u8, num: i32) → i32;
extern "C" rite SSL_shutdown(ssl: *void) → i32;
extern "C" rite SSL_free(ssl: *void);

// Error handling
extern "C" rite SSL_get_error(ssl: *void, ret: i32) → i32;
extern "C" rite ERR_get_error() → u64;
extern "C" rite ERR_error_string(e: u64, buf: *u8) → *u8;
```

### 9.5 Future: Pure Sigil TLS (v2.0+)

Aspirational goal for complete zero-C purity using **Arcanum 2.0** (Sigil crypto library).

**Arcanum 2.0** will be a Sigil rewrite of the existing Rust Arcanum library, providing:
- **Symmetric:** AES-256-GCM, ChaCha20-Poly1305, XChaCha20
- **Asymmetric:** X25519, X448, ECDH
- **Signatures:** Ed25519, ECDSA
- **Hashing:** SHA-256, SHA-384, SHA-512, Blake3
- **KDFs:** HKDF, Argon2id
- **Post-Quantum:** ML-KEM (Kyber), ML-DSA (Dilithium)

**TLS 1.3 Requirements from Arcanum:**
```
TLS_AES_128_GCM_SHA256        → AES-GCM + SHA-256 + HKDF
TLS_AES_256_GCM_SHA384        → AES-GCM + SHA-384 + HKDF
TLS_CHACHA20_POLY1305_SHA256  → ChaCha20-Poly1305 + SHA-256 + HKDF
Key Exchange                  → X25519 or ECDH P-256
Certificate Verification      → Ed25519 or ECDSA + X.509 parsing
```

**Dependency Chain:**
```
stdlib/net/tls.sg (v2.0) → arcanum 2.0 → stdlib/net/socket.sg → syscalls
```

See `/home/crook/dev/arcanum` for the existing Rust implementation.

---

## 10. Testing Strategy

### 10.1 Unit Tests

| Layer | Test Focus |
|-------|------------|
| Syscall | Mock kernel responses, error paths |
| Socket | Connection lifecycle, timeout handling |
| DNS | Packet encoding/decoding, response parsing |
| HTTP | Request formatting, response parsing, chunked encoding |
| WebSocket | Handshake, frame encoding/decoding, masking |

### 10.2 Integration Tests

- Connect to real DNS servers
- HTTP requests to httpbin.org
- WebSocket echo server tests

### 10.3 Fuzz Testing

- DNS packet parsing
- HTTP response parsing
- WebSocket frame parsing

### 10.4 Performance Benchmarks

- Syscall overhead vs libc
- HTTP request latency
- WebSocket throughput

---

## 11. File Structure

```
sigil-lang/
├── stdlib/
│   └── net/
│       ├── sys/
│       │   ├── mod.sg              # Platform detection
│       │   ├── linux_x86_64.sg     # Linux x86_64 syscalls
│       │   ├── darwin_x86_64.sg    # macOS x86_64 syscalls
│       │   └── linux_aarch64.sg    # Linux ARM64 syscalls
│       ├── tls/
│       │   ├── mod.sg              # TLS abstraction
│       │   ├── openssl.sg          # OpenSSL FFI (Linux)
│       │   └── security.sg         # Security.framework FFI (macOS)
│       ├── socket.sg               # Socket abstraction (TCP/UDP, IPv4/IPv6)
│       ├── dns.sg                  # DNS client (A, AAAA records)
│       ├── http.sg                 # HTTP/1.1 client (keep-alive, proxy)
│       └── ws.sg                   # WebSocket client
└── parser/
    └── src/
        └── llvm_codegen.rs         # Inline asm support

# Separate repository (optional dependency)
haagenti/
├── src/
│   ├── deflate.sg                  # Deflate/Gzip/Zlib
│   ├── lz4.sg                      # LZ4
│   ├── zstd.sg                     # Zstandard
│   └── brotli.sg                   # Brotli
└── Sigil.toml                      # Tome package manifest
```

---

## 12. Success Criteria

### 12.1 Functional Requirements

- [ ] Can make HTTP GET request to any URL
- [ ] Can make HTTP POST request with body
- [ ] Can parse chunked transfer encoding
- [ ] Can follow redirects (up to limit)
- [ ] Can establish WebSocket connection
- [ ] Can send/receive WebSocket messages
- [ ] Can handle WebSocket ping/pong
- [ ] DNS resolution works for any hostname
- [ ] Timeouts work correctly
- [ ] Errors are properly propagated

### 12.2 Non-Functional Requirements

- [ ] No libc dependency in compiled binary
- [ ] HTTP request latency within 10% of curl
- [ ] Memory usage reasonable (no leaks)
- [ ] Works on Linux x86_64
- [ ] Works on macOS x86_64 (stretch goal)
- [ ] Works on Linux ARM64 (stretch goal)

---

## 13. Design Decisions (Resolved)

| Question | Decision | Rationale |
|----------|----------|-----------|
| **TLS** | v1.0 via OS FFI | HTTPS is table stakes for production use |
| **IPv6** | Full support | Modern networks require it |
| **Keep-Alive** | Yes | Significant performance gain for real-world use |
| **Compression** | Deferred to Haagenti 2.0 | Separate concern; HTTP works without it |
| **Proxy** | Yes | Corporate environments require it |

### 13.1 Compression Strategy

HTTP compression (gzip, deflate, brotli) is **not** included in stdlib.

**Rationale:**
- Compression is a separate concern from networking
- Haagenti (existing Rust library) will be rewritten in Sigil as Haagenti 2.0
- HTTP client will work without compression
- Optional integration: if Haagenti is present, HTTP client can use it

**Integration Pattern:**
```sigil
// Without Haagenti - no compression
≔ client = HttpClient·new();
≔ response = client·get("https://example.com")?;

// With Haagenti - automatic decompression
use haagenti::deflate;
≔ client = HttpClient·new()
    ·with_decompressor(deflate::decompress);
≔ response = client·get("https://example.com")?;  // Auto-decompresses
```

### 13.2 Keep-Alive Implementation

Connection pooling with configurable limits:

```sigil
☉ sigil ConnectionPool {
    max_idle_per_host: usize!,      // Default: 4
    idle_timeout_ms: u32!,          // Default: 90_000 (90 seconds)
    max_total_connections: usize!,  // Default: 100
}
```

### 13.3 Proxy Support

HTTP CONNECT proxy for HTTPS, direct proxy for HTTP:

```sigil
☉ sigil ProxyConfig {
    http_proxy: Url?,   // For HTTP requests
    https_proxy: Url?,  // For HTTPS requests (uses CONNECT)
    no_proxy: [String]!, // Hostnames to bypass proxy
}

☉ impl HttpClient {
    ☉ rite with_proxy(&vary self, config: ProxyConfig) → &vary Self!;
}

---

## Appendix A: Linux x86_64 Syscall Numbers

| Name | Number | Notes |
|------|--------|-------|
| read | 0 | |
| write | 1 | |
| close | 3 | |
| poll | 7 | |
| socket | 41 | |
| connect | 42 | |
| accept | 43 | |
| sendto | 44 | |
| recvfrom | 45 | |
| bind | 49 | |
| listen | 50 | |
| setsockopt | 54 | |
| getsockopt | 55 | |
| fcntl | 72 | |
| getpeername | 52 | Optional |
| getsockname | 51 | Optional |

---

## Appendix B: macOS x86_64 Syscall Numbers

macOS syscalls are numbered differently and use a different calling convention (syscall class in high bits).

| Name | Number | Notes |
|------|--------|-------|
| read | 0x2000003 | |
| write | 0x2000004 | |
| close | 0x2000006 | |
| socket | 0x2000061 | |
| connect | 0x2000062 | |
| ... | ... | TBD |

---

## Revision History

| Version | Date | Changes |
|---------|------|---------|
| 0.1.0-draft | 2026-01-20 | Initial draft |
| 0.2.0 | 2026-01-20 | Approved: Added TLS FFI layer (OpenSSL/Security.framework), IPv6 full support, Keep-Alive connection pooling, Proxy support (HTTP CONNECT), Haagenti 2.0 integration pattern for compression, Arcanum 2.0 roadmap for future pure Sigil TLS |
| 0.3.0 | 2026-01-20 | **CRITICAL**: Added Section 3 (Compiler Prerequisites) documenting inline assembly gap. Parser supports `asm!` syntax but no backend implements it. LLVM backend implementation is a blocking prerequisite before syscall layer can be built. Renumbered all subsequent sections. |
