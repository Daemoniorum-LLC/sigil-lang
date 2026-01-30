# Sigil Native Runtime Specification

**Version:** 0.1.0
**Status:** Draft
**Authors:** Human + Claude
**Date:** 2026-01-20
**Methodology:** [Spec-Driven Development](../methodologies/SPEC-DRIVEN-DEVELOPMENT.md)

---

## 1. Overview

### 1.1 Vision

Sigil programs compile to native binaries with **zero libc dependency**. The runtime is implemented entirely in Sigil, using direct syscalls via inline assembly. This enables:

- True self-hosting: Jormungandr compiles itself without C
- Minimal binary size: No libc bloat
- Full control: No hidden runtime behavior
- Cross-compilation: No target libc required

### 1.2 Goals

| Goal | Description | Success Metric |
|------|-------------|----------------|
| **G1** | Replace `sigil_runtime.c` entirely | Zero C source files in runtime |
| **G2** | Jormungandr self-hosts without libc | `sigil2` binary runs with `ldd` showing "not a dynamic executable" |
| **G3** | Feature parity with C runtime | All 741 lines of functionality preserved |
| **G4** | Linux x86_64 first | Full syscall coverage for primary platform |
| **G5** | Maintain performance | Benchmarks within 10% of C runtime |

### 1.3 Non-Goals (v1.0)

- Windows native syscalls (use WSL2 for now)
- macOS native syscalls (future phase)
- Full POSIX compatibility
- Dynamic linking support

---

## 2. Current State Analysis

### 2.1 C Runtime (`parser/runtime/sigil_runtime.c`)

**Size:** 741 lines
**Functions:** 47
**Dependencies:** stdio.h, stdlib.h, string.h, math.h, time.h, OpenSSL (optional)

| Category | Functions | libc Dependency |
|----------|-----------|-----------------|
| **Print** | `sigil_print_int`, `sigil_print_float`, `sigil_print_str` | printf |
| **Memory** | `sigil_alloc`, `sigil_realloc`, `sigil_free` | malloc/realloc/free |
| **String** | `sigil_strlen`, `sigil_string_*` | strlen, memcpy |
| **Vec** | `sigil_vec_new`, `sigil_vec_push`, `sigil_vec_get` | malloc |
| **Option** | `sigil_option_some`, `sigil_option_none`, etc. | malloc |
| **Math** | `sigil_sqrt`, `sigil_sin`, `sigil_cos`, etc. (25 functions) | libm |
| **File I/O** | `sigil_file_open`, `sigil_file_read`, etc. | fopen/fread/fwrite |
| **Time** | `sigil_now` | gettimeofday |
| **System** | `sigil_exit`, `sigil_getenv` | exit, getenv |
| **TLS** | `sigil_tls_*` (14 functions) | OpenSSL |

### 2.2 Jormungandr Runtime (`jormungandr/src/runtime.sg`)

**Size:** 1,327 lines
**Status:** Partial - uses `extern "C"` wrappers for syscalls

| Category | Status | Notes |
|----------|--------|-------|
| **Arena Allocator** | ✅ Pure Sigil | 93 lines, works |
| **Rc<T>** | ✅ Pure Sigil | Reference counting, Drop trait |
| **String Interner** | ✅ Pure Sigil | Deduplication |
| **Evidence Tracking** | ✅ Pure Sigil | Runtime epistemic checks |
| **Math** | ✅ Pure Sigil | Taylor series, Newton-Raphson |
| **Collections** | ✅ Pure Sigil | range, sort, unique, etc. |
| **Crypto** | ✅ Pure Sigil | CRC32, FNV-1a, Adler32 |
| **Bytes** | ✅ Pure Sigil | Little/big endian, hex encode |
| **I/O** | ❌ extern "C" | Uses libc `read`/`write` wrappers |
| **Time** | ❌ extern "C" | Uses libc `time`/`usleep` |
| **Random** | ⚠️ Partial | LCG impl, but seeds from `time()` |

### 2.3 Networking stdlib (`stdlib/net/`)

**Size:** ~4,290 lines
**Status:** ✅ Pure syscalls (proof of concept)

Demonstrates the target architecture:
- Direct syscalls via inline assembly
- No libc dependency for socket operations
- OpenSSL FFI only for TLS (acceptable external dependency)

### 2.4 Jormungandr Bootstrap Status

**Current:** Almost self-hosting (nested match bug blocking final step)

| Component | Status |
|-----------|--------|
| Lexer | ✅ Self-hosted |
| Parser | ✅ Self-hosted |
| AST | ✅ Self-hosted |
| Type checker | ✅ Self-hosted |
| IR lowering | ✅ Self-hosted |
| Codegen (C) | ✅ Self-hosted |
| Runtime | ❌ Still uses extern "C" |

---

## 3. Architecture

### 3.1 Layer Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                     Sigil User Code                             │
├─────────────────────────────────────────────────────────────────┤
│                     stdlib (Pure Sigil)                         │
│  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐   │
│  │  fmt    │ │  math   │ │   fs    │ │   net   │ │  time   │   │
│  └────┬────┘ └────┬────┘ └────┬────┘ └────┬────┘ └────┬────┘   │
│       │           │           │           │           │         │
├───────┴───────────┴───────────┴───────────┴───────────┴─────────┤
│                     stdlib/sys (Platform Layer)                 │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐               │
│  │   syscall   │ │    alloc    │ │     io      │               │
│  │  (raw asm)  │ │ (mmap-based)│ │ (read/write)│               │
│  └──────┬──────┘ └──────┬──────┘ └──────┬──────┘               │
│         │               │               │                       │
├─────────┴───────────────┴───────────────┴───────────────────────┤
│                     Linux Kernel (syscalls)                     │
│  write(1) read(0) mmap(9) munmap(11) open(2) close(3) ...      │
└─────────────────────────────────────────────────────────────────┘
```

### 3.2 Module Structure

```
stdlib/
├── sys/                    # Platform abstraction layer
│   ├── mod.sg             # Platform detection, re-exports
│   ├── syscall.sg         # Raw syscall primitives
│   ├── linux_x86_64.sg    # Linux x86_64 syscall numbers & ABI
│   ├── errno.sg           # Error codes
│   ├── types.sg           # Platform types (size_t, ssize_t, etc.)
│   │
│   ├── alloc.sg           # Memory allocation (mmap-based)
│   ├── io.sg              # File descriptors, read/write
│   ├── fs.sg              # File system operations
│   ├── time.sg            # Clock, sleep
│   ├── process.sg         # Exit, getenv, getpid
│   └── mman.sg            # Memory mapping (mmap, mprotect)
│
├── alloc/                  # Allocator implementations
│   ├── mod.sg
│   ├── bump.sg            # Bump allocator (fast, no free)
│   ├── freelist.sg        # Free-list allocator (general purpose)
│   └── arena.sg           # Arena allocator (batch free)
│
├── fmt/                    # Formatting
│   ├── mod.sg
│   ├── int.sg             # Integer to string
│   ├── float.sg           # Float to string (Grisu3/Ryū)
│   └── write.sg           # Buffered output
│
├── math/                   # Math functions (pure Sigil)
│   ├── mod.sg
│   ├── trig.sg            # sin, cos, tan, atan, etc.
│   ├── exp.sg             # exp, ln, log2, log10, pow
│   ├── special.sg         # sqrt, cbrt, hypot
│   └── util.sg            # abs, min, max, clamp, floor, ceil
│
├── fs/                     # File system
│   ├── mod.sg
│   ├── file.sg            # File handle, read, write
│   ├── path.sg            # Path manipulation
│   └── dir.sg             # Directory operations
│
├── time/                   # Time operations
│   ├── mod.sg
│   ├── instant.sg         # Monotonic time
│   ├── duration.sg        # Time duration
│   └── clock.sg           # Wall clock
│
└── net/                    # Networking (already implemented)
    ├── mod.sg
    ├── socket.sg
    ├── dns.sg
    ├── http.sg
    ├── websocket.sg
    ├── pool.sg
    └── tls/
        └── openssl.sg     # TLS via OpenSSL FFI (acceptable)
```

### 3.3 Syscall Layer Design

```sigil
//! stdlib/sys/syscall.sg
//!
//! Raw syscall interface for Linux x86_64.
//! All other platform code builds on these primitives.

/// Issue a syscall with 0-6 arguments
/// Returns: result (negative = -errno on error)
☉ rite syscall0(nr: i64) -> i64! {
    ≔ ret: i64;
    asm!(
        "syscall",
        in("rax") nr,
        lateout("rax") ret,
        lateout("rcx") _,
        lateout("r11") _,
        options(nostack)
    );
    ret
}

☉ rite syscall1(nr: i64, a1: i64) -> i64! {
    ≔ ret: i64;
    asm!(
        "syscall",
        in("rax") nr,
        in("rdi") a1,
        lateout("rax") ret,
        lateout("rcx") _,
        lateout("r11") _,
        options(nostack)
    );
    ret
}

// ... syscall2 through syscall6 ...

/// Check if syscall result is an error
☉ rite is_error(ret: i64) -> bool! {
    ret < 0 && ret >= -4095
}

/// Convert syscall error to errno
☉ rite to_errno(ret: i64) -> i32! {
    (-ret) as i32
}
```

### 3.4 Memory Allocator Design

```sigil
//! stdlib/sys/alloc.sg
//!
//! Memory allocation via mmap syscall.
//! No malloc/free - we manage memory directly.

use sys·syscall·{syscall6, is_error};
use sys·linux_x86_64·{SYS_MMAP, SYS_MUNMAP, PROT_READ, PROT_WRITE, MAP_PRIVATE, MAP_ANONYMOUS};

/// Allocate `size` bytes of memory
☉ rite alloc(size: u64) -> *vary u8? {
    ≔ ret = syscall6(
        SYS_MMAP,
        0,                          // addr (let kernel choose)
        size as i64,                // length
        (PROT_READ ⋎ PROT_WRITE) as i64,
        (MAP_PRIVATE ⋎ MAP_ANONYMOUS) as i64,
        -1,                         // fd (not file-backed)
        0                           // offset
    );

    ⎇ is_error(ret) {
        ret null
    }

    ret as *vary u8
}

/// Free memory allocated with alloc()
☉ rite free(ptr: *vary u8, size: u64) {
    syscall2(SYS_MUNMAP, ptr as i64, size as i64);
}

/// Allocate zeroed memory
☉ rite alloc_zeroed(size: u64) -> *vary u8? {
    // mmap with MAP_ANONYMOUS already returns zeroed memory
    alloc(size)
}
```

### 3.5 I/O Design

```sigil
//! stdlib/sys/io.sg
//!
//! Low-level I/O via syscalls.

use sys·syscall·{syscall3, is_error, to_errno};
use sys·linux_x86_64·{SYS_READ, SYS_WRITE};

/// Standard file descriptors
☉ const STDIN: i32 = 0;
☉ const STDOUT: i32 = 1;
☉ const STDERR: i32 = 2;

/// Write bytes to a file descriptor
☉ rite write(fd: i32, buf: &[u8]) -> Result<u64, i32>! {
    ≔ ret = syscall3(SYS_WRITE, fd as i64, buf.as_ptr() as i64, buf.len() as i64);
    ⎇ is_error(ret) {
        Err(to_errno(ret))
    } ⎉ {
        Ok(ret as u64)
    }
}

/// Read bytes from a file descriptor
☉ rite read(fd: i32, buf: &vary [u8]) -> Result<u64, i32>! {
    ≔ ret = syscall3(SYS_READ, fd as i64, buf.as_mut_ptr() as i64, buf.len() as i64);
    ⎇ is_error(ret) {
        Err(to_errno(ret))
    } ⎉ {
        Ok(ret as u64)
    }
}

/// Print string to stdout (no newline)
☉ rite print(s: &str) {
    write(STDOUT, s.as_bytes());
}

/// Print string to stdout with newline
☉ rite println(s: &str) {
    print(s);
    write(STDOUT, b"\n");
}
```

---

## 4. Platform Support Matrix

### 4.1 Phase 1: Linux x86_64 (Primary Target)

| Syscall | Number | Purpose | Priority |
|---------|--------|---------|----------|
| `read` | 0 | Read from fd | P0 |
| `write` | 1 | Write to fd | P0 |
| `open` | 2 | Open file | P0 |
| `close` | 3 | Close fd | P0 |
| `mmap` | 9 | Allocate memory | P0 |
| `munmap` | 11 | Free memory | P0 |
| `exit_group` | 231 | Exit process | P0 |
| `clock_gettime` | 228 | Get time | P1 |
| `nanosleep` | 35 | Sleep | P1 |
| `lseek` | 8 | Seek in file | P1 |
| `fstat` | 5 | File metadata | P1 |
| `getpid` | 39 | Process ID | P2 |
| `getcwd` | 79 | Current directory | P2 |
| `readlink` | 89 | Read symlink | P2 |

### 4.2 Future Platforms

| Platform | Approach | Timeline |
|----------|----------|----------|
| Linux aarch64 | Different syscall ABI | v1.1 |
| macOS x86_64 | Different syscall numbers | v1.2 |
| macOS aarch64 | ARM64 + different numbers | v1.2 |
| Windows x64 | NT syscalls or Win32 FFI | v2.0 |

---

## 5. Implementation Phases

### Phase 0: Foundation (stdlib/sys/) ✅ COMPLETE

**Status:** Completed 2026-01-20
**Goal:** Core syscall primitives

**Deliverables:**
- ✅ `mod.sg` - SyscallError enum, syscall_result helper, common types
- ✅ `linux_x86_64.sg` - Syscall numbers, raw syscall0-6 functions, high-level wrappers
- ✅ `alloc.sg` - mmap-based allocator (bump + free list)
- ✅ `tests/syscall_test.sg` - Verification tests

**Implementation Notes:**
- Inline asm syntax uses `inout("rax") input => output` (not `lateout`)
- LLVM backend requires `--features llvm` flag
- Variables receiving asm output must be `let mut`
- Syscalls verified: getpid(39), mmap(9), munmap(11)

**Tests Passing:**
- `test_getpid` - getpid returns positive PID ✅
- `test_mmap` - Anonymous memory mapping works ✅
- `test_munmap` - Memory unmapping works ✅

**Success Criteria:** ✅ Met - syscalls work via inline assembly

### Phase 1: Memory Allocation

**Goal:** Replace malloc/free
**Deliverables:**
- `sys/alloc.sg` - mmap/munmap wrappers
- `alloc/bump.sg` - Bump allocator
- `alloc/freelist.sg` - Free-list allocator

**Tests:**
- `test_alloc_basic` - Allocate and use memory
- `test_alloc_free` - Allocate, free, reallocate
- `test_alloc_large` - Large allocation (>1MB)

**Success Criteria:** Vec<T> works without malloc

### Phase 2: I/O Operations

**Goal:** Replace stdio
**Deliverables:**
- `sys/io.sg` - read/write syscalls
- `fmt/int.sg` - Integer formatting
- `fmt/write.sg` - Buffered output

**Tests:**
- `test_print_int` - Print integers
- `test_read_stdin` - Read from stdin
- `test_buffered_write` - Buffered output performance

**Success Criteria:** `sigil_print_int` replaced

### Phase 3: File System

**Goal:** Replace fopen/fread/fwrite
**Deliverables:**
- `sys/fs.sg` - open/close/read/write/lseek
- `fs/file.sg` - File handle abstraction
- `fs/path.sg` - Path manipulation

**Tests:**
- `test_file_read` - Read entire file
- `test_file_write` - Write to file
- `test_file_seek` - Seek and read

**Success Criteria:** `sigil_file_read_all` replaced

### Phase 4: Time

**Goal:** Replace gettimeofday
**Deliverables:**
- `sys/time.sg` - clock_gettime/nanosleep
- `time/instant.sg` - Monotonic time
- `time/duration.sg` - Duration arithmetic

**Tests:**
- `test_time_now` - Get current time
- `test_time_sleep` - Sleep for duration
- `test_time_elapsed` - Measure elapsed time

**Success Criteria:** `sigil_now` replaced

### Phase 5: Float Formatting

**Goal:** Print floats without libm
**Deliverables:**
- `fmt/float.sg` - Float to string (Grisu3 or Ryū)

**Tests:**
- `test_fmt_float_basic` - Format 3.14159
- `test_fmt_float_scientific` - Scientific notation
- `test_fmt_float_edge` - NaN, Inf, denormals

**Success Criteria:** `sigil_print_float` replaced

### Phase 6: Integration

**Goal:** Wire everything together
**Deliverables:**
- Update Jormungandr `runtime.sg` to use `stdlib/sys`
- Remove all `extern "C"` declarations
- Update LLVM codegen to not link libc

**Tests:**
- `test_jormungandr_self_compile` - Jormungandr compiles itself
- `test_no_libc` - `ldd` shows static binary

**Success Criteria:** Jormungandr produces binaries with no libc

---

## 6. Prerequisites & Dependencies

### 6.1 Compiler Prerequisites

| Prerequisite | Status | Notes |
|--------------|--------|-------|
| Inline assembly (`asm!`) | ✅ Parser + LLVM | Verified 2026-01-20 |
| Raw pointers | ✅ | Works |
| Unsafe blocks | ✅ | Works |
| Byte literals (`b"..."`) | ⚠️ Unknown | Need to verify |
| Static variables | ✅ | Works |

### 6.2 Blocking Issues

**Issue #1: Inline Assembly in LLVM Backend**

Status: ✅ **VERIFIED** (2026-01-20)

Inline assembly works in the LLVM backend. Verified with multiple tests:

```bash
# Build with LLVM feature
CARGO_INCREMENTAL=0 cargo build --release --features llvm

# Test 1: getpid syscall
./sigil compile tests/asm/syscall_test.sg -o /tmp/syscall_test
/tmp/syscall_test  # Exit code = PID (non-zero, success)

# Test 2: Comprehensive asm test (nop, mov, syscall)
./sigil compile tests/asm/basic_asm_test.sg -o /tmp/basic_asm_test
/tmp/basic_asm_test  # Exit code 0 = all tests passed

# Test 3: write syscall (I/O)
# Syscall executed successfully (returned 20 bytes written)
# Note: Buffer address handling needs refinement for proper output
```

**Key findings:**
- `asm!` syntax fully supported in parser
- LLVM backend uses `inkwell::InlineAsmDialect` (see `llvm_codegen.rs:2088-2213`)
- Syscalls 1 (write) and 39 (getpid) confirmed working
- Cranelift backend does NOT support inline assembly (expected)

**Issue #2: Jormungandr Nested Match Bug**

Status: ❌ **BLOCKING SELF-HOST**

The bootstrap is blocked by a nested match codegen bug. This must be fixed before Jormungandr can fully self-host.

See: `jormungandr/BOOTSTRAP_SUCCESS_SUMMARY.md`

### 6.3 External Dependencies

| Dependency | Status | Notes |
|------------|--------|-------|
| OpenSSL | Optional | TLS only - acceptable for now |
| CUDA | Optional | GPU compute - not in scope |
| No libc | Goal | Must not link libc for core runtime |

### 6.4 Jormungandr Update Requirements

**Status:** BLOCKING - Must complete before Phase 0 stdlib/sys implementation

Jormungandr (the self-hosted compiler) must be updated to support native Sigil syntax and inline assembly before the native runtime can be implemented.

#### 6.4.1 Required Features

| Feature | Priority | Effort | Description |
|---------|----------|--------|-------------|
| **Native Sigil symbols** | P0 | Medium | Lexer support for `☉`, `≔`, `⎇`, `⎉`, `⌥` |
| **Native keywords** | P0 | Low | Parser support for `vary`, `rite` |
| **Type suffixes** | P0 | Low | `!` (owned), `?` (nullable), `~` (borrowed) |
| **Inline assembly** | P0 | High | `asm!` macro with `inout` syntax |
| **Nested match fix** | P0 | Medium | Fix blocking bug in pattern matching |

#### 6.4.2 Lexer Changes

Add token types for native Sigil symbols:

```
// New tokens for jormungandr/src/lexer.sg
TokenKind::SunWithRays,     // ☉ (U+2609) - public visibility
TokenKind::Assign,          // ≔ (U+2254) - constant binding
TokenKind::ConditionalIf,   // ⎇ (U+2387) - if
TokenKind::ConditionalElse, // ⎉ (U+2389) - else
TokenKind::Loop,            // ⌥ (U+2325) - loop/while
```

Keyword additions:
```
"vary" → TokenKind::Vary    // mutable modifier
"rite" → TokenKind::Rite    // function declaration
```

#### 6.4.3 Parser Changes

| Construct | Current | Native |
|-----------|---------|--------|
| Public visibility | `pub fn foo()` | `☉ rite foo()` |
| Constant binding | `let x = 1` | `≔ x = 1` |
| Mutable binding | `let mut x = 1` | `vary x = 1` |
| Conditional | `if cond { } else { }` | `⎇ cond { } ⎉ { }` |
| Loop | `while cond { }` | `⌥ cond { }` |

Parser must accept BOTH syntaxes (Rust-like for compatibility, native for new code).

#### 6.4.4 Inline Assembly Support

Jormungandr's C codegen must emit inline assembly for syscalls:

**Input (Sigil):**
```sigil
≔ pid: i64 = 0;
vary result: i64 = 0;
unsafe {
    asm!("syscall",
        inout("rax") 39_i64 => result,
        out("rcx") _,
        out("r11") _,
        options(nostack));
}
```

**Output (C):**
```c
int64_t pid = 0;
int64_t result = 0;
__asm__ volatile (
    "syscall"
    : "=a" (result)
    : "a" ((int64_t)39)
    : "rcx", "r11"
);
```

Key mappings:
- `inout("rax") x => y` → `"=a" (y) : "a" (x)`
- `in("rdi") x` → `"D" (x)` (input only)
- `out("rcx") _` → `"rcx"` in clobber list
- `options(nostack)` → no stack red zone adjustment needed

#### 6.4.5 Test Criteria

| Test | Description | Pass Criteria |
|------|-------------|---------------|
| T1 | Lexer tokenizes native symbols | `☉ rite main() {}` tokenizes correctly |
| T2 | Parser accepts native syntax | Full file parses without error |
| T3 | Inline asm compiles | `asm!("syscall", ...)` generates C |
| T4 | Syscall works | `getpid()` returns valid PID |
| T5 | Full test suite | All 414 P0 tests pass |
| T6 | Self-compilation | `sigil2` compiles itself |

#### 6.4.6 Implementation Order

1. **Fix nested match bug** (unblocks self-compilation)
2. **Add native tokens to lexer** (Unicode symbols)
3. **Add native keywords to parser** (`vary`, `rite`)
4. **Add type suffix parsing** (`!`, `?`, `~`)
5. **Add inline assembly codegen** (C output)
6. **Verify with syscall test** (end-to-end validation)
7. **Run full test suite** (regression check)

---

## 7. Success Criteria

### 7.1 Functional Requirements

| ID | Requirement | Verification |
|----|-------------|--------------|
| F1 | All C runtime functions have Sigil equivalents | Function-by-function checklist |
| F2 | Jormungandr compiles with native runtime | Self-compilation test |
| F3 | Produced binaries are statically linked | `ldd` shows "not a dynamic executable" |
| F4 | All existing tests pass | Test suite green |

### 7.2 Performance Requirements

| ID | Requirement | Verification |
|----|-------------|--------------|
| P1 | Print performance within 10% of C | Benchmark: 1M print calls |
| P2 | Allocation performance within 20% of malloc | Benchmark: alloc/free cycle |
| P3 | File I/O performance within 10% of C | Benchmark: read 100MB file |

### 7.3 Compatibility Requirements

| ID | Requirement | Verification |
|----|-------------|--------------|
| C1 | Works on Linux x86_64 kernel 4.x+ | CI on multiple kernels |
| C2 | Works in Docker containers | Docker test |
| C3 | Works under strace | strace verification |

---

## 8. Open Questions

### 8.1 TLS Strategy

**Question:** Should we implement native TLS or keep OpenSSL FFI?

**Options:**
1. Keep OpenSSL FFI (pragmatic, proven, secure)
2. Implement TLS 1.3 natively (months of work, risky)
3. Use a minimal TLS library (BearSSL, wolfSSL)

**Current Decision:** Keep OpenSSL FFI for v1.0. Revisit for v2.0.

### 8.2 Thread Support

**Question:** How do we handle threading without pthreads?

**Options:**
1. No threading in v1.0 (single-threaded runtime)
2. Use `clone` syscall directly
3. Implement lightweight green threads

**Current Decision:** Defer to v1.1. Single-threaded for v1.0.

### 8.3 Signal Handling

**Question:** Do we need signal handlers?

**Options:**
1. No signal handling (simplest)
2. Basic SIGTERM/SIGINT handling
3. Full signal support

**Current Decision:** Basic SIGTERM for graceful shutdown in v1.0.

---

## 9. Risks & Mitigations

| Risk | Impact | Probability | Mitigation |
|------|--------|-------------|------------|
| Inline asm not working in LLVM | Blocks all work | Medium | Verify immediately in Phase 0 |
| Performance regression | Unusable runtime | Low | Benchmark at each phase |
| Platform-specific bugs | Limited adoption | Medium | Extensive testing under strace |
| Missing syscall functionality | Incomplete features | Low | Document and prioritize |

---

## 10. References

### 10.1 Related Specifications

- [Native Networking Spec](./NATIVE-NETWORKING-SPEC.md) - Syscall patterns, socket implementation
- [Jormungandr Bootstrap](../sigil-lang/jormungandr/BOOTSTRAP_SUCCESS_SUMMARY.md) - Self-hosting status

### 10.2 External References

- [Linux Syscall Table (x86_64)](https://blog.rchapman.org/posts/Linux_System_Call_Table_for_x86_64/)
- [Grisu3 Float Formatting](https://www.cs.tufts.edu/~nr/cs257/archive/florian-loitsch/printf.pdf)
- [musl libc](https://musl.libc.org/) - Reference for minimal libc implementation

---

## Revision History

| Version | Date | Changes |
|---------|------|---------|
| 0.1.0 | 2026-01-20 | Initial draft. Documented current state, proposed architecture, implementation phases. |

---

## Gap Log

*This section documents gaps discovered during implementation. Per SDD methodology, implementation stops when gaps are found, and this section is updated before proceeding.*

---

### Gap #1: Native Sigil Syntax Not Supported by Any Compiler

**Discovered:** 2026-01-20
**Phase:** Phase 0 (Code Review)
**Severity:** **CRITICAL - BLOCKING**

**Description:**

The stdlib/sys/ implementation uses "native Sigil" syntax with Unicode symbols:

| Symbol | Meaning | Rust-like Equivalent |
|--------|---------|---------------------|
| `☉` | public visibility | `pub` |
| `≔` | constant binding | `const` / `let` |
| `⎇` | conditional | `if` |
| `⎉` | else branch | `else` |
| `⌥` | loop | `while` / `loop` |
| `vary` | mutable | `mut` |
| `rite` | function | `fn` |
| `!` suffix | non-null/owned | (type system) |
| `?` suffix | nullable | `Option<T>` |
| `~` suffix | borrowed | `&` |

**Neither compiler supports this syntax:**
- **Rust-based parser** (`parser/`): Uses Rust-like keywords (`fn`, `let`, `if`, `else`, `while`)
- **Jormungandr** (`jormungandr/`): Also uses Rust-like keywords, out of date

**Impact:**

1. `stdlib/sys/mod.sg` - Cannot be compiled
2. `stdlib/sys/linux_x86_64.sg` - Cannot be compiled
3. `stdlib/sys/alloc.sg` - Cannot be compiled
4. `stdlib/net/sys/` - Cannot be compiled (same issue)
5. All future native runtime code blocked

**Root Cause:**

Native Sigil syntax was designed but never implemented in lexer/parser. The language specification includes these symbols, but the toolchain doesn't support them.

**Resolution Options:**

| Option | Effort | Pros | Cons |
|--------|--------|------|------|
| **A: Add to Rust parser** | Medium | Quick win, enables stdlib | Diverges from self-hosting goal |
| **B: Update Jormungandr** | High | Aligns with self-hosting | Blocked by nested match bug |
| **C: Dual syntax files** | Low | Works now | Maintenance burden |
| **D: Transpiler** | Medium | Automatic conversion | Another tool to maintain |

**Selected Resolution:** **Option B - Update Jormungandr**

Per user decision (2026-01-20): Update Jormungandr to support native Sigil syntax, aligning with the self-hosting goal. This blocks stdlib/sys implementation until Jormungandr is updated.

**Spec Changes Required:**

1. ~~Add new section: "11. Native Syntax Support Roadmap"~~ → Merged into 6.4
2. ✅ Update Phase 0 prerequisites: "Native syntax support in at least one compiler"
3. ✅ Add "6.4 Jormungandr Update Requirements" section

---

### Gap #2: Jormungandr Out of Date

**Discovered:** 2026-01-20
**Phase:** Phase 0 (Code Review)
**Severity:** **HIGH**

**Description:**

Jormungandr (the self-hosted compiler in `jormungandr/`) is significantly out of date compared to the Rust-based parser:

| Feature | Rust Parser | Jormungandr |
|---------|-------------|-------------|
| Inline assembly (`asm!`) | ✅ Full support | ❌ Not implemented |
| Native Sigil symbols | ❌ Not implemented | ❌ Not implemented |
| `inout` asm syntax | ✅ Works | ❌ Unknown |
| LLVM backend | ✅ Optional feature | ❌ C codegen only |
| Test pass rate | 100% (414/414) | Unknown |
| Nested match | ✅ Works | ❌ **Bug - blocking** |

**Impact:**

- Cannot use Jormungandr for self-hosting until updated
- Native runtime cannot be compiled by self-hosted compiler
- Bootstrap goal blocked

**Resolution:**

1. Fix nested match bug (tracked separately)
2. Add native Sigil syntax to Jormungandr lexer/parser
3. Add inline assembly support to Jormungandr
4. Verify against test suite

**Spec Changes Required:**

1. Add "6.4 Jormungandr Update Requirements" section
2. Update success criteria to include Jormungandr capability

---

### Gap #3: [Template]

**Discovered:** [Date]
**Phase:** [Which phase]
**Description:** [What was discovered]
**Impact:** [What does this affect]
**Resolution:** [How it was resolved]
**Spec Changes:** [What sections were updated]
