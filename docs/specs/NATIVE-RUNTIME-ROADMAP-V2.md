# Native Runtime Roadmap v2

**Version:** 2.0.0
**Date:** 2026-01-24
**Status:** Draft
**Authors:** Claude (Opus 4.5) + Human
**Methodology:** SDD + Agent-TDD

---

## Executive Summary

This specification extends the native runtime implementation (Phase 3, completed) with additional capabilities. The pure assembly runtime now exists for Linux x86_64, macOS (x86_64 + ARM64), and Windows x64. This roadmap covers:

1. **LLVM Backend Integration** - Wire native runtime into compiled binaries
2. **Linux ARM64 Port** - Extend to Raspberry Pi, AWS Graviton
3. **Test Coverage** - Push from 87% to 100%
4. **Performance Benchmarks** - Measure native vs C runtime
5. **Threading/Async Runtime** - Concurrency primitives
6. **Networking Runtime** - Socket syscalls
7. **FFI Improvements** - Better C interop
8. **Cleanup** - Linker warnings, polish

### Current State (Phase 3 Complete)

| Platform | Assembly | Status |
|----------|----------|--------|
| Linux x86_64 | `sigil_runtime_linux_x86_64.s` | ✅ Complete |
| macOS x86_64 | `sigil_runtime_macos_x86_64.s` | ✅ Complete |
| macOS ARM64 | `sigil_runtime_macos_arm64.s` | ✅ Complete |
| Windows x64 | `sigil_runtime_windows_x64.s` | ✅ Complete |
| Linux ARM64 | (not yet) | ❌ Planned |

### Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     Sigil Source Code                           │
├─────────────────────────────────────────────────────────────────┤
│                     LLVM Codegen (llvm_codegen.rs)              │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │  Emit calls to native runtime symbols (sigil_*)          │   │
│  └──────────────────────────────────────────────────────────┘   │
├─────────────────────────────────────────────────────────────────┤
│                     Linker                                      │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │  Link object file + libsigil_native.a                    │   │
│  │  Flags: -nostdlib -static (no libc!)                     │   │
│  └──────────────────────────────────────────────────────────┘   │
├─────────────────────────────────────────────────────────────────┤
│                     Native Runtime (Assembly)                   │
│  ┌──────────┬──────────┬──────────┬──────────┬──────────┐      │
│  │  Print   │  Memory  │  String  │   Vec    │   Math   │      │
│  ├──────────┴──────────┴──────────┴──────────┴──────────┤      │
│  │              Arena Allocator (1MB chunks)             │      │
│  ├───────────────────────────────────────────────────────┤      │
│  │         Direct Syscalls (no libc)                     │      │
│  └───────────────────────────────────────────────────────┘      │
├─────────────────────────────────────────────────────────────────┤
│                     Operating System Kernel                     │
└─────────────────────────────────────────────────────────────────┘
```

---

## Phase 4: LLVM Backend Integration

### 4.1 Objective

Make `sigil compile program.sg -o binary` produce a standalone executable that uses the native assembly runtime instead of the C runtime.

### 4.2 Current State

The LLVM backend (`llvm_codegen.rs`) currently:
- Declares external symbols for runtime functions
- Links against `libsigil_runtime.a` (C runtime)

### 4.3 Required Changes

#### 4.3.1 Symbol Name Mapping

Ensure LLVM codegen emits calls to symbols that match assembly exports:

| Sigil Operation | Current Symbol | Native Symbol |
|-----------------|----------------|---------------|
| Print string | `sigil_print_str` | `sigil_println` |
| Print int | `sigil_print_i64` | `sigil_print_int` |
| Allocate | `sigil_alloc` | `sigil_alloc` |
| Vec new | `sigil_vec_new` | `sigil_vec_new` |
| Vec push | `sigil_vec_push` | `sigil_vec_push` |
| String concat | `sigil_string_concat` | `sigil_string_concat` |

#### 4.3.2 Entry Point

The native runtime provides `_start` which calls `sigil_main`. LLVM must:
1. Emit `sigil_main` as the entry function (not `main`)
2. Not emit its own `_start` or `main`

#### 4.3.3 Linker Integration

Update compile command to use:
```bash
# Current (C runtime)
clang -o output program.o -L runtime -lsigil_runtime

# Native (no libc)
ld -o output program.o runtime/libsigil_native.a -nostdlib -static
```

#### 4.3.4 Build Flag

Add `--native-runtime` flag to sigil compile:
```bash
sigil compile program.sg -o binary --native-runtime
```

### 4.4 Test Cases

```sigil
// P4_001_native_compile_hello.sg
// Test: Basic native compilation produces working binary
λ main() {
    println("Hello from native runtime!");
}
// Expected: Executable runs without libc
// Verify: ldd output shows "not a dynamic executable"
```

```sigil
// P4_002_native_compile_math.sg
// Test: Math operations work in native binary
λ main() {
    ≔ a = 3.14159;
    ≔ b = sqrt(a);
    println(b);
}
```

```sigil
// P4_003_native_compile_alloc.sg
// Test: Memory allocation works in native binary
λ main() {
    ≔ v = Vec·new();
    ∀ i ∈ 0..1000 {
        v.push(i);
    }
    println(v.len());
}
```

### 4.5 Implementation Steps

1. Add `--native-runtime` CLI flag to `main.rs`
2. Modify `llvm_codegen.rs` to emit `sigil_main` entry point
3. Update linker invocation to use native library
4. Add tests in `jormungandr/tests/spec/23_native_compile/`

### 4.6 Acceptance Criteria

- [ ] `sigil compile --native-runtime` produces working binary
- [ ] `ldd binary` shows "not a dynamic executable"
- [ ] All existing tests pass with native runtime
- [ ] Binary size < 100KB for hello world

---

## Phase 5: Linux ARM64 Port

### 5.1 Objective

Create `sigil_runtime_linux_arm64.s` for Raspberry Pi, AWS Graviton, and other ARM64 Linux systems.

### 5.2 ARM64 Linux Syscall ABI

| Register | Purpose |
|----------|---------|
| x8 | Syscall number |
| x0-x5 | Arguments 1-6 |
| x0 | Return value |
| Instruction | `svc #0` |

### 5.3 Syscall Numbers (Linux ARM64)

| Syscall | Number | Purpose |
|---------|--------|---------|
| read | 63 | Read from fd |
| write | 64 | Write to fd |
| openat | 56 | Open file (relative) |
| close | 57 | Close fd |
| mmap | 222 | Map memory |
| munmap | 215 | Unmap memory |
| exit | 93 | Exit process |
| clock_gettime | 113 | Get time |

Note: ARM64 Linux uses `openat` (56) instead of `open` (not available).

### 5.4 Implementation Structure

```asm
// sigil_runtime_linux_arm64.s
.section .text

.set SYS_read,    63
.set SYS_write,   64
.set SYS_openat,  56
.set SYS_close,   57
.set SYS_mmap,    222
.set SYS_munmap,  215
.set SYS_exit,    93

.global _start
_start:
    bl sigil_main
    mov x8, SYS_exit
    svc #0

.global sigil_println
sigil_println:
    // x0 = string ptr, x1 = length
    mov x8, SYS_write
    mov x2, x1         // count
    mov x1, x0         // buf
    mov x0, #1         // stdout
    svc #0
    ret
```

### 5.5 SIMD (NEON)

ARM64 has NEON SIMD (already implemented in macOS ARM64):

```asm
// simd_f32x4_add
ld1 {v0.4s}, [x1]
ld1 {v1.4s}, [x2]
fadd v0.4s, v0.4s, v1.4s
st1 {v0.4s}, [x0]
ret
```

### 5.6 Test Cases

Same as Linux x86_64, run on ARM64 hardware or QEMU.

### 5.7 Acceptance Criteria

- [ ] Assembly file builds with `aarch64-linux-gnu-as`
- [ ] Test binary runs on ARM64 Linux (or QEMU)
- [ ] All runtime functions implemented
- [ ] NEON SIMD operations work

---

## Phase 6: Test Coverage Improvement

### 6.1 Objective

Increase test pass rate from 87% (466/531) to 100%.

### 6.2 Current Gaps

Review `INTERPRETER-SPEC-ROADMAP.md` for failing tests:

```bash
cd jormungandr/tests
./run_tests_rust.sh 2>&1 | grep "FAIL"
```

### 6.3 Categories to Address

1. **P2 Tests** - Advanced features (protocols, holographic, etc.)
2. **Edge Cases** - Corner cases in existing features
3. **Integration** - Cross-feature interactions

### 6.4 Test-First Approach (Agent-TDD)

For each failing test:
1. Read the test and expected output
2. Understand what behavior is being specified
3. Identify gap in implementation
4. Implement the fix
5. Verify test passes
6. Run full suite for regression

### 6.5 Acceptance Criteria

- [ ] 531/531 tests passing (100%)
- [ ] No regressions in existing tests
- [ ] Document any spec updates needed

---

## Phase 7: Performance Benchmarks

### 7.1 Objective

Quantify performance difference between C runtime and native assembly runtime.

### 7.2 Benchmark Suite

Create `benchmarks/` directory with:

```sigil
// bench_alloc.sg - Memory allocation
λ main() {
    ≔ start = Sys·clock_gettime(CLOCK_MONOTONIC());
    ∀ i ∈ 0..1000000 {
        ≔ ptr = Arena·alloc(64);
    }
    ≔ end = Sys·clock_gettime(CLOCK_MONOTONIC());
    println(end - start);
}
```

```sigil
// bench_string.sg - String operations
λ main() {
    ≔ start = Sys·clock_gettime(CLOCK_MONOTONIC());
    ≔ s = "";
    ∀ i ∈ 0..10000 {
        s = s + "x";
    }
    ≔ end = Sys·clock_gettime(CLOCK_MONOTONIC());
    println(end - start);
}
```

```sigil
// bench_simd.sg - SIMD operations
λ main() {
    ≔ a = [1.0, 2.0, 3.0, 4.0];
    ≔ b = [5.0, 6.0, 7.0, 8.0];
    ≔ result = [0.0, 0.0, 0.0, 0.0];

    ≔ start = Sys·clock_gettime(CLOCK_MONOTONIC());
    ∀ i ∈ 0..10000000 {
        simd_f32x4_add(&result, &a, &b);
    }
    ≔ end = Sys·clock_gettime(CLOCK_MONOTONIC());
    println(end - start);
}
```

### 7.3 Metrics

| Metric | C Runtime | Native Runtime | Delta |
|--------|-----------|----------------|-------|
| Binary size (hello) | ? KB | ? KB | ? |
| Startup time | ? μs | ? μs | ? |
| 1M allocations | ? ms | ? ms | ? |
| 10K string concats | ? ms | ? ms | ? |
| 10M SIMD ops | ? ms | ? ms | ? |

### 7.4 Acceptance Criteria

- [ ] Benchmark suite created
- [ ] Results documented
- [ ] Native runtime equal or faster than C runtime
- [ ] Binary size comparable or smaller

---

## Phase 8: Threading/Async Runtime

### 8.1 Objective

Add concurrency primitives to native runtime.

### 8.2 Required Syscalls

| Syscall | Number (x86_64) | Purpose |
|---------|-----------------|---------|
| clone | 56 | Create thread |
| futex | 202 | Fast userspace mutex |
| set_tid_address | 218 | Thread ID |
| exit_group | 231 | Exit all threads |

### 8.3 Primitives

```sigil
// Thread creation
λ Thread·spawn(f: λ()) → ThreadHandle

// Mutex
λ Mutex·new() → Mutex
λ Mutex·lock(&self)
λ Mutex·unlock(&self)

// Channel
λ Channel·new() → (Sender, Receiver)
λ Sender·send(&self, value: T)
λ Receiver·recv(&self) → T
```

### 8.4 Implementation Complexity

**HIGH** - Threading requires:
- Thread-local storage (TLS)
- Stack allocation per thread
- Atomic operations
- Careful synchronization

### 8.5 Status

**DEFERRED** - Focus on single-threaded runtime first. Mark as future work.

---

## Phase 9: Networking Runtime

### 9.1 Objective

Add socket syscalls for native networking without libc.

### 9.2 Required Syscalls

| Syscall | Number (x86_64) | Purpose |
|---------|-----------------|---------|
| socket | 41 | Create socket |
| connect | 42 | Connect to address |
| accept | 43 | Accept connection |
| bind | 49 | Bind to address |
| listen | 50 | Listen for connections |
| sendto | 44 | Send data |
| recvfrom | 45 | Receive data |
| setsockopt | 54 | Set socket options |
| getsockopt | 55 | Get socket options |

### 9.3 API

```sigil
// TCP Client
λ TcpStream·connect(addr: &str, port: u16) → Result<TcpStream, Error>
λ TcpStream·read(&self, buf: &mut [u8]) → i64
λ TcpStream·write(&self, buf: &[u8]) → i64
λ TcpStream·close(&self)

// TCP Server
λ TcpListener·bind(addr: &str, port: u16) → Result<TcpListener, Error>
λ TcpListener·accept(&self) → Result<TcpStream, Error>
```

### 9.4 Status

**PLANNED** - Implement after Phase 4-6 complete.

---

## Phase 10: FFI Improvements

### 10.1 Objective

Better interoperability when linking against C libraries (optional).

### 10.2 Features

1. **Dynamic linking support** - Load .so/.dylib at runtime
2. **C calling convention** - Ensure ABI compatibility
3. **Struct layout** - Match C struct padding

### 10.3 Status

**LOW PRIORITY** - Native runtime goal is zero dependencies.

---

## Phase 11: Cleanup

### 11.1 Linker Warning Fix

Add `.note.GNU-stack` section to assembly files:

```asm
// At end of each .s file
.section .note.GNU-stack,"",@progbits
```

This tells the linker the stack is not executable.

### 11.2 Code Quality

- [ ] Add comments to all assembly functions
- [ ] Consistent formatting across platforms
- [ ] Remove dead code

### 11.3 Documentation

- [ ] Update CLAUDE.md with native runtime docs
- [ ] Add usage examples
- [ ] Document platform differences

---

## Implementation Order

| Phase | Priority | Complexity | Dependencies |
|-------|----------|------------|--------------|
| 4: LLVM Integration | P0 | Medium | None |
| 11: Cleanup | P0 | Low | None |
| 5: Linux ARM64 | P1 | Medium | None |
| 6: Test Coverage | P1 | Medium | None |
| 7: Benchmarks | P2 | Low | Phase 4 |
| 9: Networking | P2 | High | Phase 4 |
| 8: Threading | P3 | Very High | Phase 4, 9 |
| 10: FFI | P3 | Medium | None |

**Recommended execution order:**
1. Phase 11 (Cleanup) - Quick win
2. Phase 4 (LLVM Integration) - Core functionality
3. Phase 5 (Linux ARM64) - Platform coverage
4. Phase 6 (Test Coverage) - Quality
5. Phase 7 (Benchmarks) - Validation
6. Phase 9 (Networking) - Feature expansion
7. Phase 8, 10 - Future work

---

## Revision History

| Version | Date | Changes |
|---------|------|---------|
| 2.0.0 | 2026-01-24 | Initial draft. Roadmap for post-Phase-3 work. |
