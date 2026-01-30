# Sigil Native Networking Stack - TDD Roadmap

**Version:** 1.0.0
**Status:** Active
**Parent Spec:** [NATIVE-NETWORKING-SPEC.md](./NATIVE-NETWORKING-SPEC.md)
**Date:** 2026-01-20

---

## Overview

This document defines the Test-Driven Development roadmap for the Sigil Native Networking Stack. Tests are organized by layer, with dependencies explicitly stated.

**TDD Cycle:** Red (failing tests) → Green (minimal implementation) → Refactor

---

## Phase 1: Syscall Layer

**Dependencies:** None (foundation layer)
**Location:** `sigil-lang/stdlib/net/sys/`

### 1.1 Linux x86_64 Syscalls

#### Test File: `tests/net/sys_linux_x86_64_test.sg`

| Test ID | Test Name | Description | Expected Behavior |
|---------|-----------|-------------|-------------------|
| SYS-001 | `test_sys_socket_creates_fd` | Create TCP socket via syscall | Returns fd > 0 |
| SYS-002 | `test_sys_socket_invalid_domain` | Invalid domain parameter | Returns -EINVAL |
| SYS-003 | `test_sys_close_valid_fd` | Close valid socket fd | Returns 0 |
| SYS-004 | `test_sys_close_invalid_fd` | Close invalid fd (-1) | Returns -EBADF |
| SYS-005 | `test_sys_connect_localhost` | Connect to 127.0.0.1:port | Returns 0 on success |
| SYS-006 | `test_sys_connect_refused` | Connect to closed port | Returns -ECONNREFUSED |
| SYS-007 | `test_sys_bind_port` | Bind to local port | Returns 0 |
| SYS-008 | `test_sys_bind_address_in_use` | Bind to already bound port | Returns -EADDRINUSE |
| SYS-009 | `test_sys_listen_backlog` | Set socket to listen mode | Returns 0 |
| SYS-010 | `test_sys_accept_connection` | Accept incoming connection | Returns new fd > 0 |
| SYS-011 | `test_sys_read_from_socket` | Read data from connected socket | Returns bytes read |
| SYS-012 | `test_sys_write_to_socket` | Write data to connected socket | Returns bytes written |
| SYS-013 | `test_sys_setsockopt_reuseaddr` | Set SO_REUSEADDR | Returns 0 |
| SYS-014 | `test_sys_poll_readable` | Poll socket for read readiness | Returns 1, POLLIN set |
| SYS-015 | `test_sys_poll_timeout` | Poll with timeout, no events | Returns 0 |

#### Sigil Syntax Tests

| Test ID | Test Name | Description |
|---------|-----------|-------------|
| SYS-016 | `test_asm_syscall_syntax` | `asm!()` macro compiles correctly |
| SYS-017 | `test_syscall_result_ok` | Positive return → Ok(usize) |
| SYS-018 | `test_syscall_result_err` | Negative return → Err(SyscallError) |
| SYS-019 | `test_syscall_error_display` | Error messages are human-readable |

### 1.2 macOS x86_64 Syscalls

#### Test File: `tests/net/sys_darwin_x86_64_test.sg`

| Test ID | Test Name | Description |
|---------|-----------|-------------|
| SYS-020 | `test_darwin_socket_creates_fd` | macOS socket syscall works |
| SYS-021 | `test_darwin_syscall_class` | High bits encode syscall class (0x2000000) |
| SYS-022 | `test_darwin_connect_localhost` | Connect on macOS |

### 1.3 Linux ARM64 Syscalls

#### Test File: `tests/net/sys_linux_aarch64_test.sg`

| Test ID | Test Name | Description |
|---------|-----------|-------------|
| SYS-023 | `test_arm64_socket_creates_fd` | ARM64 socket syscall |
| SYS-024 | `test_arm64_svc_instruction` | `svc #0` instruction used |

---

## Phase 2: Socket Layer

**Dependencies:** Phase 1 (Syscall Layer)
**Location:** `sigil-lang/stdlib/net/socket.sg`

### 2.1 Socket Address Tests

#### Test File: `tests/net/socket_addr_test.sg`

| Test ID | Test Name | Description | Expected Behavior |
|---------|-----------|-------------|-------------------|
| SOCK-001 | `test_socket_addr_ipv4_parse` | Parse "192.168.1.1:8080" | SocketAddr with correct bytes |
| SOCK-002 | `test_socket_addr_ipv4_localhost` | Parse "127.0.0.1:80" | Localhost address |
| SOCK-003 | `test_socket_addr_ipv6_parse` | Parse "[::1]:8080" | IPv6 localhost |
| SOCK-004 | `test_socket_addr_ipv6_full` | Parse full IPv6 address | Correct 16-byte representation |
| SOCK-005 | `test_socket_addr_invalid` | Parse "invalid" | Error |
| SOCK-006 | `test_socket_addr_to_sockaddr` | Convert to raw sockaddr struct | Correct binary layout |
| SOCK-007 | `test_socket_addr_from_sockaddr` | Parse from raw sockaddr | Correct SocketAddr |

### 2.2 TCP Stream Tests

#### Test File: `tests/net/tcp_stream_test.sg`

| Test ID | Test Name | Description | Expected Behavior |
|---------|-----------|-------------|-------------------|
| TCP-001 | `test_tcp_connect_localhost` | Connect to local TCP server | Connected TcpStream |
| TCP-002 | `test_tcp_connect_by_hostname` | Connect to "localhost:8080" | Resolves and connects |
| TCP-003 | `test_tcp_connect_refused` | Connect to closed port | ConnectionRefused error |
| TCP-004 | `test_tcp_connect_timeout` | Connect to unreachable host | ConnectionTimedOut error |
| TCP-005 | `test_tcp_read_available` | Read available data | Returns bytes read |
| TCP-006 | `test_tcp_read_exact` | Read exact byte count | Fills buffer completely |
| TCP-007 | `test_tcp_read_exact_eof` | read_exact on closed conn | UnexpectedEof error |
| TCP-008 | `test_tcp_write_all` | Write complete buffer | All bytes sent |
| TCP-009 | `test_tcp_write_broken_pipe` | Write to closed peer | BrokenPipe error |
| TCP-010 | `test_tcp_close` | Close connection | Clean shutdown |
| TCP-011 | `test_tcp_set_read_timeout` | Set read timeout | Timeout takes effect |
| TCP-012 | `test_tcp_set_write_timeout` | Set write timeout | Timeout takes effect |
| TCP-013 | `test_tcp_peer_addr` | Get remote address | Correct SocketAddr |

### 2.3 UDP Socket Tests

#### Test File: `tests/net/udp_socket_test.sg`

| Test ID | Test Name | Description | Expected Behavior |
|---------|-----------|-------------|-------------------|
| UDP-001 | `test_udp_bind` | Bind to local port | Bound socket |
| UDP-002 | `test_udp_bind_any_port` | Bind to port 0 | OS assigns port |
| UDP-003 | `test_udp_send_to` | Send datagram | Bytes sent |
| UDP-004 | `test_udp_recv_from` | Receive datagram | Data and sender addr |
| UDP-005 | `test_udp_roundtrip` | Send and receive back | Data matches |

---

## Phase 3: DNS Layer

**Dependencies:** Phase 2 (Socket Layer - UDP)
**Location:** `sigil-lang/stdlib/net/dns.sg`

### 3.1 DNS Packet Tests

#### Test File: `tests/net/dns_packet_test.sg`

| Test ID | Test Name | Description | Expected Behavior |
|---------|-----------|-------------|-------------------|
| DNS-001 | `test_dns_header_encode` | Encode DNS header | Correct 12-byte header |
| DNS-002 | `test_dns_header_decode` | Decode DNS header | Correct fields |
| DNS-003 | `test_dns_question_encode` | Encode A query for "example.com" | Correct wire format |
| DNS-004 | `test_dns_name_encode` | Encode domain name | Label encoding (length-prefixed) |
| DNS-005 | `test_dns_name_decode` | Decode domain name | String reconstruction |
| DNS-006 | `test_dns_name_compression` | Handle name compression pointers | Resolve pointers correctly |
| DNS-007 | `test_dns_answer_decode_a` | Decode A record response | IPv4 address extracted |
| DNS-008 | `test_dns_answer_decode_aaaa` | Decode AAAA record response | IPv6 address extracted |
| DNS-009 | `test_dns_packet_id_random` | Query IDs are random | Different each call |
| DNS-010 | `test_dns_response_truncated` | Handle truncated response | TC flag detected |

### 3.2 DNS Resolver Tests

#### Test File: `tests/net/dns_resolver_test.sg`

| Test ID | Test Name | Description | Expected Behavior |
|---------|-----------|-------------|-------------------|
| DNS-011 | `test_resolver_system` | Load system DNS servers | At least one server |
| DNS-012 | `test_resolver_custom` | Create with custom servers | Uses provided servers |
| DNS-013 | `test_resolve_a_google` | Resolve "google.com" A | IPv4 addresses returned |
| DNS-014 | `test_resolve_aaaa_google` | Resolve "google.com" AAAA | IPv6 addresses returned |
| DNS-015 | `test_resolve_nonexistent` | Resolve "nonexistent.invalid" | NameNotFound error |
| DNS-016 | `test_resolve_timeout` | DNS server unreachable | QueryTimedOut error |
| DNS-017 | `test_resolve_caches_result` | Same query twice | Second is faster (cached) |

---

## Phase 4: TLS Layer (FFI)

**Dependencies:** Phase 2 (Socket Layer - TCP)
**Location:** `sigil-lang/stdlib/net/tls/`

### 4.1 OpenSSL FFI Tests (Linux)

#### Test File: `tests/net/tls_openssl_test.sg`

| Test ID | Test Name | Description | Expected Behavior |
|---------|-----------|-------------|-------------------|
| TLS-001 | `test_ssl_library_init` | Initialize OpenSSL | No error |
| TLS-002 | `test_ssl_ctx_new` | Create SSL context | Non-null pointer |
| TLS-003 | `test_ssl_ctx_free` | Free SSL context | No crash |
| TLS-004 | `test_ssl_new` | Create SSL connection | Non-null pointer |
| TLS-005 | `test_ssl_set_fd` | Attach socket fd | Returns 1 |
| TLS-006 | `test_ssl_set_hostname` | Set SNI hostname | Returns 1 |
| TLS-007 | `test_ssl_connect_google` | Handshake with google.com:443 | Successful connection |
| TLS-008 | `test_ssl_read` | Read from TLS stream | Decrypted data |
| TLS-009 | `test_ssl_write` | Write to TLS stream | Encrypted and sent |
| TLS-010 | `test_ssl_shutdown` | Graceful TLS shutdown | Clean close |
| TLS-011 | `test_ssl_cert_verify_pass` | Valid certificate | Verification passes |
| TLS-012 | `test_ssl_cert_verify_fail` | Self-signed cert (no trust) | Verification fails |
| TLS-013 | `test_ssl_error_string` | Get error description | Human-readable string |

### 4.2 Security.framework FFI Tests (macOS)

#### Test File: `tests/net/tls_security_test.sg`

| Test ID | Test Name | Description | Expected Behavior |
|---------|-----------|-------------|-------------------|
| TLS-014 | `test_sectrust_create` | Create SecTrust object | Non-null pointer |
| TLS-015 | `test_sslcontext_create` | Create SSLContext | Non-null pointer |
| TLS-016 | `test_darwin_tls_connect` | TLS handshake on macOS | Successful connection |

### 4.3 TLS Stream Tests

#### Test File: `tests/net/tls_stream_test.sg`

| Test ID | Test Name | Description | Expected Behavior |
|---------|-----------|-------------|-------------------|
| TLS-017 | `test_tls_stream_connect` | TLS over TCP connection | Connected TlsStream |
| TLS-018 | `test_tls_stream_read` | Read decrypted data | Plaintext returned |
| TLS-019 | `test_tls_stream_write_all` | Write data (encrypted) | All bytes sent |
| TLS-020 | `test_tls_stream_close` | Close TLS connection | Graceful shutdown |
| TLS-021 | `test_tls_handshake_failed` | Connect to non-TLS port | HandshakeFailed error |
| TLS-022 | `test_tls_hostname_mismatch` | Wrong SNI hostname | CertificateError |

---

## Phase 5: HTTP/1.1 Layer

**Dependencies:** Phase 3 (DNS), Phase 4 (TLS)
**Location:** `sigil-lang/stdlib/net/http.sg`

### 5.1 URL Parsing Tests

#### Test File: `tests/net/http_url_test.sg`

| Test ID | Test Name | Description | Expected Behavior |
|---------|-----------|-------------|-------------------|
| HTTP-001 | `test_url_parse_http` | Parse "http://example.com/path" | Scheme=http, host, path |
| HTTP-002 | `test_url_parse_https` | Parse "https://example.com/path" | Scheme=https |
| HTTP-003 | `test_url_parse_port` | Parse "http://example.com:8080" | Port=8080 |
| HTTP-004 | `test_url_parse_default_port` | Parse without port | Port=80/443 by scheme |
| HTTP-005 | `test_url_parse_query` | Parse "?key=value" | Query string extracted |
| HTTP-006 | `test_url_parse_fragment` | Parse "#section" | Fragment extracted |
| HTTP-007 | `test_url_parse_auth` | Parse "user:pass@host" | Auth extracted |
| HTTP-008 | `test_url_parse_invalid` | Parse "not a url" | InvalidUrl error |

### 5.2 HTTP Request Tests

#### Test File: `tests/net/http_request_test.sg`

| Test ID | Test Name | Description | Expected Behavior |
|---------|-----------|-------------|-------------------|
| HTTP-009 | `test_request_format_get` | Format GET request | Valid request line |
| HTTP-010 | `test_request_format_post` | Format POST with body | Content-Length header |
| HTTP-011 | `test_request_host_header` | Host header added | Present and correct |
| HTTP-012 | `test_request_user_agent` | User-Agent header | Present |
| HTTP-013 | `test_request_custom_headers` | Add custom headers | Headers included |
| HTTP-014 | `test_request_content_type` | POST with Content-Type | Header set correctly |

### 5.3 HTTP Response Parsing Tests

#### Test File: `tests/net/http_response_test.sg`

| Test ID | Test Name | Description | Expected Behavior |
|---------|-----------|-------------|-------------------|
| HTTP-015 | `test_response_parse_status` | Parse "HTTP/1.1 200 OK" | status=200 |
| HTTP-016 | `test_response_parse_headers` | Parse header lines | Headers map populated |
| HTTP-017 | `test_response_header_case` | Case-insensitive lookup | "Content-Type" == "content-type" |
| HTTP-018 | `test_response_content_length` | Body by Content-Length | Correct body length |
| HTTP-019 | `test_response_chunked` | Chunked Transfer-Encoding | Body reassembled |
| HTTP-020 | `test_response_chunked_trailer` | Chunked with trailers | Trailers parsed |
| HTTP-021 | `test_response_connection_close` | No Content-Length, conn close | Read until EOF |
| HTTP-022 | `test_response_malformed` | Invalid status line | InvalidResponse error |

### 5.4 HTTP Client Integration Tests

#### Test File: `tests/net/http_client_test.sg`

| Test ID | Test Name | Description | Expected Behavior |
|---------|-----------|-------------|-------------------|
| HTTP-023 | `test_http_get_httpbin` | GET httpbin.org/get | 200, JSON body |
| HTTP-024 | `test_http_post_httpbin` | POST httpbin.org/post | Body echoed back |
| HTTP-025 | `test_https_get_google` | GET https://www.google.com | 200, HTML body |
| HTTP-026 | `test_http_redirect_follow` | GET redirect endpoint | Follows to final URL |
| HTTP-027 | `test_http_redirect_limit` | Too many redirects | TooManyRedirects error |
| HTTP-028 | `test_http_timeout` | Slow server | RequestTimedOut error |
| HTTP-029 | `test_http_404` | GET nonexistent path | status=404 |
| HTTP-030 | `test_http_connection_refused` | Server not running | ConnectionFailed error |

### 5.5 Keep-Alive Tests

#### Test File: `tests/net/http_keepalive_test.sg`

| Test ID | Test Name | Description | Expected Behavior |
|---------|-----------|-------------|-------------------|
| HTTP-031 | `test_keepalive_reuses_connection` | Two requests same host | Same underlying socket |
| HTTP-032 | `test_keepalive_max_per_host` | Exceed per-host limit | New connection created |
| HTTP-033 | `test_keepalive_idle_timeout` | Wait past idle timeout | Connection closed, new one opened |
| HTTP-034 | `test_keepalive_different_hosts` | Requests to different hosts | Separate connections |
| HTTP-035 | `test_keepalive_connection_close` | Server sends Connection: close | Not reused |

### 5.6 Proxy Tests

#### Test File: `tests/net/http_proxy_test.sg`

| Test ID | Test Name | Description | Expected Behavior |
|---------|-----------|-------------|-------------------|
| HTTP-036 | `test_http_proxy_direct` | HTTP through HTTP proxy | Request via proxy |
| HTTP-037 | `test_https_proxy_connect` | HTTPS through proxy | CONNECT tunnel |
| HTTP-038 | `test_proxy_auth_basic` | Proxy authentication | Proxy-Authorization header |
| HTTP-039 | `test_no_proxy_bypass` | Host in no_proxy list | Direct connection |

---

## Phase 6: WebSocket Layer

**Dependencies:** Phase 5 (HTTP - for handshake)
**Location:** `sigil-lang/stdlib/net/ws.sg`

### 6.1 WebSocket Handshake Tests

#### Test File: `tests/net/ws_handshake_test.sg`

| Test ID | Test Name | Description | Expected Behavior |
|---------|-----------|-------------|-------------------|
| WS-001 | `test_ws_key_generation` | Generate Sec-WebSocket-Key | 16-byte base64 string |
| WS-002 | `test_ws_accept_calculation` | Calculate Sec-WebSocket-Accept | Correct SHA-1 hash |
| WS-003 | `test_ws_handshake_request` | Format upgrade request | Valid HTTP upgrade |
| WS-004 | `test_ws_handshake_response` | Parse 101 response | Upgrade successful |
| WS-005 | `test_ws_handshake_reject` | Parse non-101 response | HandshakeFailed error |
| WS-006 | `test_ws_handshake_wrong_accept` | Invalid Accept header | HandshakeFailed error |

### 6.2 WebSocket Frame Tests

#### Test File: `tests/net/ws_frame_test.sg`

| Test ID | Test Name | Description | Expected Behavior |
|---------|-----------|-------------|-------------------|
| WS-007 | `test_frame_encode_text` | Encode text frame | Correct opcode, masked |
| WS-008 | `test_frame_encode_binary` | Encode binary frame | Correct opcode |
| WS-009 | `test_frame_encode_ping` | Encode ping frame | Control frame, correct opcode |
| WS-010 | `test_frame_encode_pong` | Encode pong frame | Control frame |
| WS-011 | `test_frame_encode_close` | Encode close frame | Code and reason encoded |
| WS-012 | `test_frame_decode_text` | Decode text frame | Payload extracted |
| WS-013 | `test_frame_decode_binary` | Decode binary frame | Payload extracted |
| WS-014 | `test_frame_small_payload` | Payload < 126 bytes | 1-byte length |
| WS-015 | `test_frame_medium_payload` | Payload 126-65535 bytes | 2-byte extended length |
| WS-016 | `test_frame_large_payload` | Payload > 65535 bytes | 8-byte extended length |
| WS-017 | `test_frame_masking` | Client frames are masked | XOR mask applied |
| WS-018 | `test_frame_fragmented` | Fragmented message | FIN=0 then FIN=1 |

### 6.3 WebSocket Client Tests

#### Test File: `tests/net/ws_client_test.sg`

| Test ID | Test Name | Description | Expected Behavior |
|---------|-----------|-------------|-------------------|
| WS-019 | `test_ws_connect_echo` | Connect to echo.websocket.org | Connected WebSocket |
| WS-020 | `test_ws_send_text` | Send text message | Message sent |
| WS-021 | `test_ws_recv_text` | Receive text message | Message received |
| WS-022 | `test_ws_send_binary` | Send binary message | Binary sent |
| WS-023 | `test_ws_recv_binary` | Receive binary message | Binary received |
| WS-024 | `test_ws_ping_pong` | Send ping | Pong received |
| WS-025 | `test_ws_auto_pong` | Receive ping | Auto-pong sent |
| WS-026 | `test_ws_close_graceful` | Send close frame | Close handshake |
| WS-027 | `test_ws_close_with_code` | Close with status code | Code transmitted |
| WS-028 | `test_wss_connect` | Connect to wss:// | TLS WebSocket |
| WS-029 | `test_ws_custom_headers` | Connect with auth header | Headers in handshake |
| WS-030 | `test_ws_message_too_large` | Receive huge message | MessageTooLarge error |

---

## Phase 7: Integration & Performance

**Dependencies:** All previous phases
**Location:** `tests/net/integration/`

### 7.1 Full Stack Integration Tests

#### Test File: `tests/net/integration/full_stack_test.sg`

| Test ID | Test Name | Description |
|---------|-----------|-------------|
| INT-001 | `test_https_json_api` | GET JSON from HTTPS API, parse body |
| INT-002 | `test_websocket_chat` | Send/receive multiple WS messages |
| INT-003 | `test_http_download_large` | Download 10MB file |
| INT-004 | `test_dns_then_http` | Resolve hostname, then HTTP request |
| INT-005 | `test_redirect_chain` | Follow 5+ redirects |
| INT-006 | `test_connection_pool_stress` | 100 requests, verify pooling |

### 7.2 Performance Benchmarks

#### Test File: `tests/net/integration/benchmarks.sg`

| Test ID | Benchmark | Target | Comparison |
|---------|-----------|--------|------------|
| PERF-001 | `bench_syscall_overhead` | < 100ns | vs libc |
| PERF-002 | `bench_dns_resolution` | < 50ms | vs system resolver |
| PERF-003 | `bench_http_get_latency` | < 10% overhead | vs curl |
| PERF-004 | `bench_http_throughput` | > 100 MB/s | vs curl |
| PERF-005 | `bench_ws_frame_encode` | < 1us per frame | raw speed |
| PERF-006 | `bench_ws_throughput` | > 50 MB/s | message rate |

---

## Test Infrastructure

### Test Helper Functions

```sigil
// tests/net/test_helpers.sg

// Start a local TCP echo server for testing
☉ rite start_echo_server(port: u16) → Result<ServerHandle, TestError>!;

// Wait for server to be ready
☉ rite wait_for_port(port: u16, timeout_ms: u32) → Result<(), TestError>!;

// Get a free port for testing
☉ rite get_free_port() → u16!;

// Create a mock DNS server
☉ rite start_mock_dns(responses: &[(String, IpAddr)]) → Result<ServerHandle, TestError>!;

// Create a mock HTTP server
☉ rite start_mock_http(routes: &[MockRoute]) → Result<ServerHandle, TestError>!;
```

### Test Configuration

```sigil
// tests/net/test_config.sg

☉ sigil TestConfig {
    // Skip tests that require network access
    skip_network_tests: bool = false,

    // Timeout for network operations
    network_timeout_ms: u32 = 5000,

    // External test endpoints
    httpbin_url: String = "https://httpbin.org",
    ws_echo_url: String = "wss://echo.websocket.org",
}
```

---

## Execution Order

### Phase Dependencies

```
Phase 1: Syscall Layer
    ↓
Phase 2: Socket Layer
    ├─────────────────────┐
    ↓                     ↓
Phase 3: DNS Layer    Phase 4: TLS Layer
    ├─────────────────────┘
    ↓
Phase 5: HTTP Layer
    ↓
Phase 6: WebSocket Layer
    ↓
Phase 7: Integration & Performance
```

### Recommended Order

1. **Week 1: Foundation**
   - SYS-001 through SYS-019 (Syscall tests)
   - SOCK-001 through SOCK-007 (Socket address tests)

2. **Week 2: Sockets**
   - TCP-001 through TCP-013 (TCP stream tests)
   - UDP-001 through UDP-005 (UDP socket tests)

3. **Week 3: DNS**
   - DNS-001 through DNS-017 (DNS tests)

4. **Week 4: TLS**
   - TLS-001 through TLS-022 (TLS tests)

5. **Week 5: HTTP**
   - HTTP-001 through HTTP-035 (HTTP tests)

6. **Week 6: WebSocket**
   - WS-001 through WS-030 (WebSocket tests)

7. **Week 7: Integration**
   - INT-001 through INT-006 (Integration tests)
   - PERF-001 through PERF-006 (Benchmarks)

---

## Success Criteria

### Phase Completion Criteria

| Phase | Passing Tests | Coverage Target |
|-------|--------------|-----------------|
| Syscall | 24/24 | 100% of syscall wrappers |
| Socket | 18/18 | 100% of socket API |
| DNS | 17/17 | 100% of DNS API |
| TLS | 22/22 | 100% of TLS API |
| HTTP | 39/39 | 100% of HTTP API |
| WebSocket | 30/30 | 100% of WebSocket API |
| Integration | 6/6 | End-to-end scenarios |

### Final Acceptance

- [ ] All 156 tests passing
- [ ] No memory leaks (valgrind clean)
- [ ] Performance within targets
- [ ] Works on Linux x86_64
- [ ] Works on macOS x86_64 (stretch)
- [ ] Works on Linux ARM64 (stretch)

---

## Revision History

| Version | Date | Changes |
|---------|------|---------|
| 1.0.0 | 2026-01-20 | Initial TDD roadmap |
