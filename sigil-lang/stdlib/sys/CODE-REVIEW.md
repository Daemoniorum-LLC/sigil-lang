# Phase 0 Code Review - stdlib/sys/

**Date:** 2026-01-20
**Reviewer:** Claude (automated)
**Spec:** `docs/specs/SIGIL-NATIVE-RUNTIME-SPEC.md`

---

## Summary

| File | Lines | Quality | Completeness | Spec Compliance | Issues |
|------|-------|---------|--------------|-----------------|--------|
| `mod.sg` | 303 | ⚠️ | ✅ | ✅ | 2 |
| `linux_x86_64.sg` | 573 | ⚠️ | ✅ | ⚠️ | 4 |
| `alloc.sg` | 465 | ❌ | ⚠️ | ⚠️ | 6 |

**Overall Rating:** 6.5/10 - Functional but needs refinement

---

## File: `mod.sg`

### Quality Issues

| ID | Severity | Line | Issue | Remediation |
|----|----------|------|-------|-------------|
| M1 | LOW | 47-88 | EAGAIN and EWOULDBLOCK both have value 11, but enum allows duplicate values | Consider using alias pattern or single variant |
| M2 | MEDIUM | 147-198 | `errno()` method duplicates match arms from `from_errno()` | Consider storing errno in struct field |

### Completeness ✅

- [x] SyscallError enum covers common errors
- [x] Display impl for error messages
- [x] syscall_result helper for Result conversion
- [x] syscall_result_zero for void syscalls
- [x] File descriptor constants

### Spec Compliance ✅

- [x] Matches spec Section 3.3 error handling design
- [x] Linux errno values correct per POSIX/Linux docs

### Missing Items (Non-blocking)

- [ ] `Eq` impl for SyscallError (mentioned in net/sys/mod.sg but not here)
- [ ] `Debug` impl for SyscallError

---

## File: `linux_x86_64.sg`

### Quality Issues

| ID | Severity | Line | Issue | Remediation |
|----|----------|------|-------|-------------|
| L1 | **HIGH** | 91-221 | Uses `lateout("rax")` syntax but tests showed `inout("rax") x => y` is required | **CRITICAL**: Update to match verified working syntax |
| L2 | MEDIUM | 332, 372 | mmap error detection uses `raw >= 0 || (raw as u64) < 0xFFFF...` - fragile | Use explicit range check for MAP_FAILED |
| L3 | LOW | 21-84 | Syscall numbers not exported (missing `☉`) | Add visibility markers |
| L4 | LOW | 524-527 | `print()` uses `s.as_ptr()` and `s.len()` which may not exist for `&str` | Verify string methods exist in stdlib |

### Completeness ✅

- [x] syscall0 through syscall6 implemented
- [x] I/O syscalls (read, write, open, close, lseek)
- [x] Memory syscalls (mmap, munmap, mprotect, mremap)
- [x] Process syscalls (exit, exit_group, getpid, getppid)
- [x] Time syscalls (clock_gettime, nanosleep)
- [x] Random syscall (getrandom)
- [x] All flag constants defined

### Spec Compliance ⚠️

- [x] Syscall numbers match Linux x86_64 (verified against kernel headers)
- [x] ABI documented in header comment
- [x] Result<T, SyscallError> return types
- [ ] **ISSUE**: Inline asm syntax doesn't match verified working pattern
- [ ] Socket syscalls defined but not wrapped (delegated to net/sys/)

### Critical Fix Required

The raw syscall functions use `lateout` which was NOT verified to work. The test that passed used:
```sigil
inout("rax") 39_i64 => pid,  // WORKING
```

But the implementation uses:
```sigil
in("rax") nr,
lateout("rax") result,  // NOT VERIFIED
```

---

## File: `alloc.sg`

### Quality Issues

| ID | Severity | Line | Issue | Remediation |
|----|----------|------|-------|-------------|
| A1 | **CRITICAL** | 405-409 | `size_of<T>()` returns hardcoded 8 - BROKEN | Must be compiler intrinsic or per-type |
| A2 | **HIGH** | 156-161 | Global mutable static without synchronization | Add `// SAFETY:` comments, document single-threaded requirement |
| A3 | **HIGH** | 135 | `destroy()` calculates wrong base address | Chunk base is `mem`, not `self - size_of::<Chunk>()` |
| A4 | MEDIUM | 109 | `try_alloc` return type `?*vary u8` non-standard | Use `Option<*vary u8>` |
| A5 | MEDIUM | 412-427 | memset/memcpy are O(n) byte-by-byte | Acceptable for now, mark as optimization target |
| A6 | LOW | 36-40 | AllocHeader padding may cause alignment issues | Add `#[repr(C)]` or explicit padding |

### Completeness ⚠️

- [x] alloc() implemented
- [x] free() implemented
- [x] realloc() implemented
- [x] alloc_zeroed() implemented
- [x] Statistics functions
- [ ] **Missing**: `alloc_aligned()` for specific alignment
- [ ] **Missing**: Chunk coalescing/defragmentation
- [ ] **Missing**: Thread-safe variant

### Spec Compliance ⚠️

- [x] Uses mmap/munmap as specified
- [x] Free list architecture matches spec
- [x] Large allocation threshold correct (4KB)
- [ ] **ISSUE**: size_of<T>() not a real intrinsic
- [ ] **ISSUE**: Code uses native Sigil syntax, won't compile with current parser

---

## Automated Review Checklist

### Pre-merge Checklist

```
[ ] All files parse without error
[ ] No use of unverified inline asm syntax
[ ] All public functions have doc comments
[ ] All unsafe blocks have SAFETY comments
[ ] No hardcoded placeholder implementations (size_of)
[ ] Thread safety documented
[ ] Error handling complete (no panics)
[ ] Spec reference in file header
```

### Spec Compliance Checklist

```
[ ] Syscall numbers match kernel headers
[ ] ABI documented (registers, clobbers)
[ ] Result types use SyscallError
[ ] Constants exported with visibility
[ ] Memory safety invariants documented
[ ] Platform conditionals correct
```

### Test Coverage Checklist

```
[ ] getpid syscall test
[ ] mmap/munmap syscall test
[ ] write syscall test (I/O)
[ ] clock_gettime syscall test
[ ] Allocator basic test
[ ] Allocator free list test
[ ] Allocator large allocation test
```

---

## Remediation Priority

### P0 - Must Fix Before Phase 1

1. **L1**: Update `linux_x86_64.sg` syscall functions to use verified `inout(...) x => y` syntax
2. **A1**: Remove fake `size_of<T>()` - use explicit sizes or compiler intrinsic
3. **A3**: Fix `Chunk::destroy()` base address calculation

### P1 - Should Fix Soon

4. **A2**: Add safety documentation for global mutable static
5. **L2**: Improve mmap error detection robustness
6. **M2**: Refactor SyscallError to avoid code duplication

### P2 - Can Defer

7. **L3**: Add visibility markers to syscall numbers
8. **A4**: Standardize Option type syntax
9. **A5**: Optimize memset/memcpy (SIMD later)
10. **A6**: Add explicit struct layout

---

## Test Results

```
stdlib/sys/tests/syscall_test.sg:
  - test_getpid: ✅ PASS
  - test_mmap:   ✅ PASS
  - test_munmap: ✅ PASS

stdlib/sys/alloc.sg:
  - NOT TESTABLE (uses uncompilable syntax)
```

---

## Conclusion

Phase 0 delivers the core syscall infrastructure but has critical issues in:

1. **Inline asm syntax mismatch** - Implementation doesn't match verified pattern
2. **Broken size_of<T>()** - Allocator cannot work correctly
3. **Syntax compatibility** - Files use native Sigil syntax not yet supported by parser

**Recommendation:** Fix P0 issues before declaring Phase 0 complete. The syscall test proves the mechanism works; implementation details need alignment with verified patterns.
