# Sigil Native Runtime Specification

**Version:** 1.0.0
**Date:** 2026-01-21
**Status:** Draft
**Goal:** Pure Sigil runtime with zero C dependencies

---

## Executive Summary

This specification defines the Native Runtime for Sigil, replacing the current C-based runtime (`parser/runtime/sigil_runtime.c`) with a pure Sigil implementation. This achieves full self-hosting: Jormungandr (compiler) + Native Runtime (stdlib) = zero external dependencies.

### Current C Runtime Analysis

| Category | Functions | Lines | Dependencies |
|----------|-----------|-------|--------------|
| Time | 1 | 25 | `<sys/time.h>` / `<windows.h>` |
| Print | 4 | 20 | `<stdio.h>` |
| Memory | 3 | 15 | `<stdlib.h>` |
| Vec | 4 | 50 | Memory |
| Option | 7 | 50 | Memory |
| String | 8 | 100 | Memory, `<string.h>` |
| Math (f64) | 23 | 130 | `<math.h>` |
| Math (i64) | 5 | 25 | None |
| File I/O | 7 | 80 | `<stdio.h>` |
| TLS | 12 | 120 | `<openssl/ssl.h>` (optional) |
| System | 2 | 15 | `<stdlib.h>` |
| **Total** | **76** | **~630** | 6 headers |

### Target Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     Sigil User Program                          │
├─────────────────────────────────────────────────────────────────┤
│                    Native Runtime (Sigil)                       │
│  ┌──────────┬──────────┬──────────┬──────────┬──────────┐       │
│  │   Vec    │  String  │  Option  │  Result  │  HashMap │       │
│  ├──────────┴──────────┴──────────┴──────────┴──────────┤       │
│  │                    Memory Allocator                   │       │
│  ├───────────────────────────────────────────────────────┤       │
│  │         Platform Syscall Layer (arch-specific)        │       │
│  └───────────────────────────────────────────────────────┘       │
├─────────────────────────────────────────────────────────────────┤
│              LLVM Intrinsics (math, atomics)                    │
├─────────────────────────────────────────────────────────────────┤
│                   Operating System Kernel                        │
└─────────────────────────────────────────────────────────────────┘
```

---

## Module 1: Platform Syscalls (`rt::sys`)

The foundation layer providing direct kernel access.

### 1.1 Specification

```sigil
/// Platform-specific syscall module
mod rt::sys {
    /// Target triples
    #[cfg(target = "x86_64-linux")]
    mod linux_x64;

    #[cfg(target = "x86_64-darwin")]
    mod darwin_x64;

    #[cfg(target = "x86_64-windows")]
    mod windows_x64;

    #[cfg(target = "aarch64-linux")]
    mod linux_arm64;

    #[cfg(target = "aarch64-darwin")]
    mod darwin_arm64;
}
```

### 1.2 Linux x86_64 Syscalls

| Syscall | Number | Signature | Purpose |
|---------|--------|-----------|---------|
| `read` | 0 | `(fd: i32, buf: *u8, count: u64) → i64` | Read from fd |
| `write` | 1 | `(fd: i32, buf: *u8, count: u64) → i64` | Write to fd |
| `open` | 2 | `(path: *u8, flags: i32, mode: i32) → i32` | Open file |
| `close` | 3 | `(fd: i32) → i32` | Close fd |
| `mmap` | 9 | `(addr: *u8, len: u64, prot: i32, flags: i32, fd: i32, off: i64) → *u8` | Map memory |
| `munmap` | 11 | `(addr: *u8, len: u64) → i32` | Unmap memory |
| `brk` | 12 | `(addr: *u8) → *u8` | Heap boundary |
| `exit` | 60 | `(code: i32) → !` | Exit process |
| `clock_gettime` | 228 | `(clk: i32, ts: *Timespec) → i32` | Get time |

### 1.3 Syscall Implementation (Linux x86_64)

```sigil
/// Raw syscall with 3 arguments (Linux x86_64)
#[inline(always)]
unsafe fn syscall3(num: u64, a1: u64, a2: u64, a3: u64) → i64 {
    ≔ ret: i64;
    asm!(
        "syscall",
        in("rax") num,
        in("rdi") a1,
        in("rsi") a2,
        in("rdx") a3,
        lateout("rax") ret,
        clobber_abi("system")
    );
    ret
}

/// Write to file descriptor
pub fn write(fd: i32, buf: &[u8]) → Result<u64, Errno>! {
    unsafe {
        ≔ ret = syscall3(1, fd as u64, buf.as_ptr() as u64, buf.len() as u64);
        ⎇ ret < 0 {
            Err(Errno::from(-ret as i32))
        } ⎉ {
            Ok(ret as u64)
        }
    }
}

/// Memory map (anonymous)
pub fn mmap_anon(size: u64) → Result<*mut u8, Errno>! {
    ≔ PROT_READ_WRITE = 0x3;  // PROT_READ | PROT_WRITE
    ≔ MAP_PRIVATE_ANON = 0x22;  // MAP_PRIVATE | MAP_ANONYMOUS
    unsafe {
        ≔ ret = syscall6(9, 0, size, PROT_READ_WRITE, MAP_PRIVATE_ANON, -1i64 as u64, 0);
        ⎇ ret as i64 == -1 {
            Err(Errno::ENOMEM)
        } ⎉ {
            Ok(ret as *mut u8)
        }
    }
}
```

### 1.4 Error Type

```sigil
/// POSIX error numbers
pub enum Errno {
    EPERM = 1,      // Operation not permitted
    ENOENT = 2,     // No such file or directory
    ESRCH = 3,      // No such process
    EINTR = 4,      // Interrupted system call
    EIO = 5,        // I/O error
    ENOMEM = 12,    // Out of memory
    EACCES = 13,    // Permission denied
    EFAULT = 14,    // Bad address
    EEXIST = 17,    // File exists
    ENOTDIR = 20,   // Not a directory
    EISDIR = 21,    // Is a directory
    EINVAL = 22,    // Invalid argument
    EMFILE = 24,    // Too many open files
    ENOSPC = 28,    // No space left on device
    EPIPE = 32,     // Broken pipe
    // ... complete set
}

impl Errno {
    pub fn from(code: i32) → Self! { /* ... */ }
    pub fn message(&self) → &str! { /* ... */ }
}
```

### 1.5 Tests

```sigil
#[test]
fn test_write_stdout() {
    ≔ msg = "Hello\n";
    ≔ result = sys::write(1, msg.as_bytes());
    assert!(result.is_ok());
    assert_eq!(result.unwrap(), 6);
}

#[test]
fn test_mmap_anon() {
    ≔ result = sys::mmap_anon(4096);
    assert!(result.is_ok());
    ≔ ptr = result.unwrap();
    assert!(ptr as u64 != 0);

    // Write and read back
    unsafe {
        *ptr = 42;
        assert_eq!(*ptr, 42);
    }

    // Cleanup
    sys::munmap(ptr, 4096);
}

#[test]
fn test_clock_gettime() {
    ≔ result = sys::clock_gettime_realtime();
    assert!(result.is_ok());
    ≔ ts = result.unwrap();
    assert!(ts.tv_sec > 1700000000);  // After 2023
}
```

---

## Module 2: Memory Allocator (`rt::alloc`)

A simple bump allocator with arena support, suitable for bootstrapping.

### 2.1 Design Options

| Allocator | Complexity | Performance | Fragmentation | Use Case |
|-----------|------------|-------------|---------------|----------|
| Bump | Simple | O(1) alloc | High | Short-lived |
| Arena | Simple | O(1) alloc | None (bulk free) | Compiler passes |
| Free List | Medium | O(n) alloc | Medium | General purpose |
| Buddy | Complex | O(log n) | Low | OS kernel |
| **mimalloc-style** | Complex | O(1) amortized | Low | Production |

**Recommended:** Start with Arena allocator, evolve to mimalloc-style.

### 2.2 Arena Allocator Specification

```sigil
/// Memory arena for bulk allocation
pub struct Arena {
    /// Current allocation pointer
    ptr: *mut u8,
    /// End of current block
    end: *mut u8,
    /// Block size for new allocations
    block_size: u64,
    /// List of allocated blocks (for cleanup)
    blocks: Vec<(*mut u8, u64)>,
}

impl Arena {
    /// Create arena with default 64KB blocks
    pub fn new() → Self! {
        Self::with_block_size(64 * 1024)
    }

    /// Create arena with custom block size
    pub fn with_block_size(size: u64) → Self! {
        Arena {
            ptr: null_mut(),
            end: null_mut(),
            block_size: size,
            blocks: Vec::new(),
        }
    }

    /// Allocate bytes (8-byte aligned)
    pub fn alloc(&mut self, size: u64) → *mut u8! {
        ≔ aligned_size = (size + 7) & !7;  // Round up to 8

        ⎇ self.ptr.offset(aligned_size as isize) > self.end {
            self.grow(aligned_size)?;
        }

        ≔ result = self.ptr;
        self.ptr = self.ptr.offset(aligned_size as isize);
        result
    }

    /// Allocate and zero-initialize
    pub fn alloc_zeroed(&mut self, size: u64) → *mut u8! {
        ≔ ptr = self.alloc(size)?;
        unsafe { ptr.write_bytes(0, size as usize); }
        ptr
    }

    /// Allocate typed value
    pub fn alloc_val<T>(&mut self, val: T) → &mut T! {
        ≔ ptr = self.alloc(size_of::<T>())? as *mut T;
        unsafe {
            ptr.write(val);
            &mut *ptr
        }
    }

    /// Reset arena (invalidates all allocations, keeps memory)
    pub fn reset(&mut self) {
        ⎇ !self.blocks.is_empty() {
            ≔ (first_block, _) = self.blocks[0];
            self.ptr = first_block;
            self.end = first_block.offset(self.block_size as isize);
        }
    }

    /// Free all memory
    pub fn free_all(&mut self) {
        for (ptr, size) in &self.blocks {
            sys::munmap(*ptr, *size);
        }
        self.blocks.clear();
        self.ptr = null_mut();
        self.end = null_mut();
    }

    // Private: grow arena
    fn grow(&mut self, min_size: u64) → Result<(), Errno>! {
        ≔ size = max(self.block_size, min_size);
        ≔ ptr = sys::mmap_anon(size)?;
        self.blocks.push((ptr, size));
        self.ptr = ptr;
        self.end = ptr.offset(size as isize);
        Ok(())
    }
}

impl Drop for Arena {
    fn drop(&mut self) {
        self.free_all();
    }
}
```

### 2.3 Global Allocator

```sigil
/// Global allocator for general-purpose allocation
static mut GLOBAL_ARENA: Arena = Arena::uninitialized();

/// Initialize global allocator (call once at startup)
pub fn init_allocator() {
    unsafe {
        GLOBAL_ARENA = Arena::new();
    }
}

/// Allocate memory
#[no_mangle]
pub extern "C" fn sigil_alloc(size: i64) → *mut u8 {
    unsafe {
        GLOBAL_ARENA.alloc(size as u64).unwrap_or(null_mut())
    }
}

/// Free memory (no-op for arena, tracked for future allocator)
#[no_mangle]
pub extern "C" fn sigil_free(ptr: *mut u8) {
    // Arena doesn't support individual frees
    // Future: add to free list for reuse
    let _ = ptr;
}

/// Reallocate memory
#[no_mangle]
pub extern "C" fn sigil_realloc(ptr: *mut u8, old_size: i64, new_size: i64) → *mut u8 {
    unsafe {
        ≔ new_ptr = GLOBAL_ARENA.alloc(new_size as u64).unwrap_or(null_mut());
        ⎇ !new_ptr.is_null() && !ptr.is_null() {
            ptr.copy_to(new_ptr, min(old_size, new_size) as usize);
        }
        new_ptr
    }
}
```

### 2.4 Tests

```sigil
#[test]
fn test_arena_alloc() {
    vary arena = Arena::new();

    ≔ p1 = arena.alloc(100);
    assert!(!p1.is_null());

    ≔ p2 = arena.alloc(200);
    assert!(!p2.is_null());
    assert!(p2 > p1);  // Sequential allocation
}

#[test]
fn test_arena_alignment() {
    vary arena = Arena::new();

    ≔ p1 = arena.alloc(1);  // 1 byte
    ≔ p2 = arena.alloc(1);  // 1 byte

    // Both should be 8-byte aligned
    assert_eq!((p1 as u64) % 8, 0);
    assert_eq!((p2 as u64) % 8, 0);
    assert_eq!(p2 as u64 - p1 as u64, 8);  // Padded to 8
}

#[test]
fn test_arena_large_alloc() {
    vary arena = Arena::with_block_size(1024);

    // Allocate more than block size
    ≔ p = arena.alloc(2048);
    assert!(!p.is_null());
}

#[test]
fn test_arena_reset() {
    vary arena = Arena::new();

    ≔ p1 = arena.alloc(100);
    arena.reset();
    ≔ p2 = arena.alloc(100);

    // After reset, should reuse same address
    assert_eq!(p1, p2);
}

#[test]
fn test_global_allocator() {
    init_allocator();

    ≔ p = sigil_alloc(256);
    assert!(!p.is_null());

    // Write pattern
    unsafe {
        for i in 0..256 {
            *p.offset(i) = (i & 0xFF) as u8;
        }
    }

    sigil_free(p);
}
```

---

## Module 3: Core Types (`rt::types`)

### 3.1 Vec<T>

```sigil
/// Growable array type
pub struct Vec<T> {
    ptr: *mut T,
    len: u64,
    cap: u64,
}

impl<T> Vec<T> {
    /// Create empty Vec
    pub fn new() → Self! {
        Vec { ptr: null_mut(), len: 0, cap: 0 }
    }

    /// Create Vec with capacity
    pub fn with_capacity(cap: u64) → Self! {
        ≔ ptr = sigil_alloc((cap * size_of::<T>()) as i64) as *mut T;
        Vec { ptr, len: 0, cap }
    }

    /// Push element
    pub fn push(&mut self, val: T) {
        ⎇ self.len == self.cap {
            self.grow();
        }
        unsafe {
            self.ptr.offset(self.len as isize).write(val);
        }
        self.len += 1;
    }

    /// Pop element
    pub fn pop(&mut self) → Option<T>? {
        ⎇ self.len == 0 {
            None
        } ⎉ {
            self.len -= 1;
            unsafe {
                Some(self.ptr.offset(self.len as isize).read())
            }
        }
    }

    /// Get element by index
    pub fn get(&self, idx: u64) → Option<&T>? {
        ⎇ idx >= self.len {
            None
        } ⎉ {
            unsafe { Some(&*self.ptr.offset(idx as isize)) }
        }
    }

    /// Get mutable element
    pub fn get_mut(&mut self, idx: u64) → Option<&mut T>? {
        ⎇ idx >= self.len {
            None
        } ⎉ {
            unsafe { Some(&mut *self.ptr.offset(idx as isize)) }
        }
    }

    /// Length
    pub fn len(&self) → u64! { self.len }

    /// Capacity
    pub fn capacity(&self) → u64! { self.cap }

    /// Is empty
    pub fn is_empty(&self) → bool! { self.len == 0 }

    /// Clear (keeps capacity)
    pub fn clear(&mut self) {
        // Drop elements if T has Drop
        ⎇ needs_drop::<T>() {
            for i in 0..self.len {
                unsafe { drop_in_place(self.ptr.offset(i as isize)); }
            }
        }
        self.len = 0;
    }

    /// As slice
    pub fn as_slice(&self) → &[T]! {
        unsafe { slice::from_raw_parts(self.ptr, self.len as usize) }
    }

    /// As mutable slice
    pub fn as_mut_slice(&mut self) → &mut [T]! {
        unsafe { slice::from_raw_parts_mut(self.ptr, self.len as usize) }
    }

    // Private: grow capacity
    fn grow(&mut self) {
        ≔ new_cap = ⎇ self.cap == 0 { 4 } ⎉ { self.cap * 2 };
        ≔ new_size = new_cap * size_of::<T>();
        ≔ new_ptr = sigil_realloc(
            self.ptr as *mut u8,
            (self.cap * size_of::<T>()) as i64,
            new_size as i64
        ) as *mut T;
        self.ptr = new_ptr;
        self.cap = new_cap;
    }
}

impl<T> Drop for Vec<T> {
    fn drop(&mut self) {
        self.clear();
        ⎇ !self.ptr.is_null() {
            sigil_free(self.ptr as *mut u8);
        }
    }
}

impl<T> Index<u64> for Vec<T> {
    type Output = T;
    fn index(&self, idx: u64) → &T! {
        self.get(idx).expect("index out of bounds")
    }
}
```

### 3.2 String

```sigil
/// UTF-8 string type
pub struct String {
    vec: Vec<u8>,
}

impl String {
    /// Create empty String
    pub fn new() → Self! {
        String { vec: Vec::new() }
    }

    /// Create String with capacity
    pub fn with_capacity(cap: u64) → Self! {
        String { vec: Vec::with_capacity(cap) }
    }

    /// Create from &str
    pub fn from(s: &str) → Self! {
        vary string = String::with_capacity(s.len() as u64);
        string.push_str(s);
        string
    }

    /// Push string slice
    pub fn push_str(&mut self, s: &str) {
        for b in s.as_bytes() {
            self.vec.push(*b);
        }
    }

    /// Push single char
    pub fn push(&mut self, ch: char) {
        ≔ buf: [u8; 4] = [0; 4];
        ≔ encoded = ch.encode_utf8(&mut buf);
        self.push_str(encoded);
    }

    /// Length in bytes
    pub fn len(&self) → u64! { self.vec.len() }

    /// Is empty
    pub fn is_empty(&self) → bool! { self.vec.is_empty() }

    /// As &str
    pub fn as_str(&self) → &str! {
        unsafe { str::from_utf8_unchecked(self.vec.as_slice()) }
    }

    /// As bytes
    pub fn as_bytes(&self) → &[u8]! {
        self.vec.as_slice()
    }

    /// Clear
    pub fn clear(&mut self) {
        self.vec.clear();
    }

    /// Concatenate
    pub fn concat(&self, other: &String) → String! {
        vary result = String::with_capacity(self.len() + other.len());
        result.push_str(self.as_str());
        result.push_str(other.as_str());
        result
    }
}

impl Add<&String> for String {
    type Output = String;
    fn add(self, other: &String) → String! {
        self.concat(other)
    }
}
```

### 3.3 Option<T> / Result<T, E>

These are already language-level types with compiler support. The runtime provides additional methods:

```sigil
impl<T> Option<T> {
    /// Unwrap or panic
    pub fn unwrap(self) → T! {
        match self {
            Some(v) => v,
            None => panic!("unwrap called on None"),
        }
    }

    /// Unwrap with default
    pub fn unwrap_or(self, default: T) → T! {
        match self {
            Some(v) => v,
            None => default,
        }
    }

    /// Unwrap with closure
    pub fn unwrap_or_else<F: FnOnce() → T>(self, f: F) → T! {
        match self {
            Some(v) => v,
            None => f(),
        }
    }

    /// Map inner value
    pub fn map<U, F: FnOnce(T) → U>(self, f: F) → Option<U>? {
        match self {
            Some(v) => Some(f(v)),
            None => None,
        }
    }

    /// Is Some
    pub fn is_some(&self) → bool! {
        matches!(self, Some(_))
    }

    /// Is None
    pub fn is_none(&self) → bool! {
        matches!(self, None)
    }
}

impl<T, E> Result<T, E> {
    /// Unwrap or panic
    pub fn unwrap(self) → T! {
        match self {
            Ok(v) => v,
            Err(e) => panic!("unwrap called on Err"),
        }
    }

    /// Unwrap error or panic
    pub fn unwrap_err(self) → E! {
        match self {
            Ok(_) => panic!("unwrap_err called on Ok"),
            Err(e) => e,
        }
    }

    /// Is Ok
    pub fn is_ok(&self) → bool! {
        matches!(self, Ok(_))
    }

    /// Is Err
    pub fn is_err(&self) → bool! {
        matches!(self, Err(_))
    }

    /// Map Ok value
    pub fn map<U, F: FnOnce(T) → U>(self, f: F) → Result<U, E>! {
        match self {
            Ok(v) => Ok(f(v)),
            Err(e) => Err(e),
        }
    }

    /// Map Err value
    pub fn map_err<F2, G: FnOnce(E) → F2>(self, g: G) → Result<T, F2>! {
        match self {
            Ok(v) => Ok(v),
            Err(e) => Err(g(e)),
        }
    }
}
```

### 3.4 Tests

```sigil
#[test]
fn test_vec_push_pop() {
    vary v: Vec<i64> = Vec::new();
    v.push(1);
    v.push(2);
    v.push(3);

    assert_eq!(v.len(), 3);
    assert_eq!(v.pop(), Some(3));
    assert_eq!(v.pop(), Some(2));
    assert_eq!(v.len(), 1);
}

#[test]
fn test_vec_grow() {
    vary v: Vec<i64> = Vec::new();
    for i in 0..1000 {
        v.push(i);
    }
    assert_eq!(v.len(), 1000);
    assert!(v.capacity() >= 1000);
    assert_eq!(v[500], 500);
}

#[test]
fn test_string_basic() {
    vary s = String::from("Hello");
    s.push_str(", World!");
    assert_eq!(s.as_str(), "Hello, World!");
    assert_eq!(s.len(), 13);
}

#[test]
fn test_string_concat() {
    ≔ s1 = String::from("Hello");
    ≔ s2 = String::from(" World");
    ≔ s3 = s1.concat(&s2);
    assert_eq!(s3.as_str(), "Hello World");
}

#[test]
fn test_option_unwrap_or() {
    ≔ some: Option<i64> = Some(42);
    ≔ none: Option<i64> = None;

    assert_eq!(some.unwrap_or(0), 42);
    assert_eq!(none.unwrap_or(0), 0);
}

#[test]
fn test_result_map() {
    ≔ ok: Result<i64, &str> = Ok(42);
    ≔ err: Result<i64, &str> = Err("error");

    assert_eq!(ok.map(|x| x * 2), Ok(84));
    assert_eq!(err.map(|x| x * 2), Err("error"));
}
```

---

## Module 4: I/O (`rt::io`)

### 4.1 Print Functions

```sigil
/// Standard output writer
pub struct Stdout;

impl Stdout {
    /// Write string to stdout
    pub fn write(&self, s: &str) → Result<u64, Errno>! {
        sys::write(1, s.as_bytes())
    }

    /// Write line to stdout
    pub fn writeln(&self, s: &str) → Result<u64, Errno>! {
        self.write(s)?;
        self.write("\n")
    }
}

/// Global stdout
pub fn stdout() → Stdout! {
    Stdout
}

/// Print without newline
pub fn print(s: &str) {
    let _ = stdout().write(s);
}

/// Print with newline
pub fn println(s: &str) {
    let _ = stdout().writeln(s);
}

/// Print integer
#[no_mangle]
pub extern "C" fn sigil_print_int(val: i64) {
    // Convert i64 to string
    vary buf: [u8; 21] = [0; 21];  // Max i64 is 20 digits + sign
    ≔ s = i64_to_str(val, &mut buf);
    println(s);
}

/// Print float
#[no_mangle]
pub extern "C" fn sigil_print_float(bits: i64) {
    ≔ val: f64 = transmute(bits);
    vary buf: [u8; 32] = [0; 32];
    ≔ s = f64_to_str(val, &mut buf);
    println(s);
}

/// Print string pointer
#[no_mangle]
pub extern "C" fn sigil_print_str(ptr: *const u8) {
    ⎇ ptr.is_null() {
        println("");
    } ⎉ {
        ≔ s = unsafe { cstr_to_str(ptr) };
        println(s);
    }
}

// Helper: i64 to string
fn i64_to_str(val: i64, buf: &mut [u8]) → &str! {
    ⎇ val == 0 {
        buf[0] = b'0';
        return unsafe { str::from_utf8_unchecked(&buf[0..1]) };
    }

    vary n = val;
    vary i = buf.len() - 1;
    ≔ negative = n < 0;
    ⎇ negative { n = -n; }

    while n > 0 {
        buf[i] = b'0' + (n % 10) as u8;
        n /= 10;
        i -= 1;
    }

    ⎇ negative {
        buf[i] = b'-';
        i -= 1;
    }

    unsafe { str::from_utf8_unchecked(&buf[i + 1..]) }
}
```

### 4.2 File I/O

```sigil
/// File handle
pub struct File {
    fd: i32,
}

impl File {
    /// Open file
    pub fn open(path: &str, mode: FileMode) → Result<File, Errno>! {
        ≔ flags = mode.to_flags();
        ≔ fd = sys::open(path.as_ptr(), flags, 0o644)?;
        Ok(File { fd })
    }

    /// Create file (truncate if exists)
    pub fn create(path: &str) → Result<File, Errno>! {
        Self::open(path, FileMode::WriteCreate)
    }

    /// Read into buffer
    pub fn read(&self, buf: &mut [u8]) → Result<u64, Errno>! {
        sys::read(self.fd, buf)
    }

    /// Read all into Vec
    pub fn read_all(&self) → Result<Vec<u8>, Errno>! {
        vary result = Vec::with_capacity(4096);
        vary buf: [u8; 4096] = [0; 4096];
        loop {
            ≔ n = self.read(&mut buf)?;
            ⎇ n == 0 { break; }
            for i in 0..n {
                result.push(buf[i as usize]);
            }
        }
        Ok(result)
    }

    /// Read all as String
    pub fn read_to_string(&self) → Result<String, Errno>! {
        ≔ bytes = self.read_all()?;
        Ok(String { vec: bytes })
    }

    /// Write buffer
    pub fn write(&self, buf: &[u8]) → Result<u64, Errno>! {
        sys::write(self.fd, buf)
    }

    /// Write all
    pub fn write_all(&self, buf: &[u8]) → Result<(), Errno>! {
        vary written = 0u64;
        while written < buf.len() as u64 {
            ≔ n = self.write(&buf[written as usize..])?;
            written += n;
        }
        Ok(())
    }

    /// Close file
    pub fn close(self) → Result<(), Errno>! {
        sys::close(self.fd)?;
        Ok(())
    }
}

impl Drop for File {
    fn drop(&mut self) {
        let _ = sys::close(self.fd);
    }
}

/// File open mode
pub enum FileMode {
    Read,
    Write,
    WriteCreate,
    ReadWrite,
    Append,
}

impl FileMode {
    fn to_flags(&self) → i32! {
        match self {
            FileMode::Read => 0,  // O_RDONLY
            FileMode::Write => 1,  // O_WRONLY
            FileMode::WriteCreate => 1 | 64 | 512,  // O_WRONLY | O_CREAT | O_TRUNC
            FileMode::ReadWrite => 2,  // O_RDWR
            FileMode::Append => 1 | 1024,  // O_WRONLY | O_APPEND
        }
    }
}

/// Convenience functions
pub fn read_file(path: &str) → Result<String, Errno>! {
    ≔ f = File::open(path, FileMode::Read)?;
    f.read_to_string()
}

pub fn write_file(path: &str, content: &str) → Result<(), Errno>! {
    ≔ f = File::create(path)?;
    f.write_all(content.as_bytes())
}

pub fn file_exists(path: &str) → bool! {
    File::open(path, FileMode::Read).is_ok()
}
```

### 4.3 Tests

```sigil
#[test]
fn test_print_int() {
    sigil_print_int(42);
    sigil_print_int(-123);
    sigil_print_int(0);
    sigil_print_int(9223372036854775807);  // i64::MAX
}

#[test]
fn test_i64_to_str() {
    vary buf: [u8; 21] = [0; 21];
    assert_eq!(i64_to_str(0, &mut buf), "0");
    assert_eq!(i64_to_str(42, &mut buf), "42");
    assert_eq!(i64_to_str(-123, &mut buf), "-123");
}

#[test]
fn test_file_write_read() {
    ≔ path = "/tmp/sigil_test.txt";
    ≔ content = "Hello, Native Runtime!";

    // Write
    write_file(path, content).unwrap();

    // Read
    ≔ read = read_file(path).unwrap();
    assert_eq!(read.as_str(), content);

    // Cleanup
    sys::unlink(path);
}

#[test]
fn test_file_exists() {
    assert!(file_exists("/etc/passwd"));  // Should exist on Linux
    assert!(!file_exists("/nonexistent/file/path"));
}
```

---

## Module 5: Math (`rt::math`)

### 5.1 LLVM Intrinsics Strategy

Most math functions can use LLVM intrinsics directly, avoiding any C dependency:

| Function | LLVM Intrinsic | Fallback |
|----------|----------------|----------|
| `sqrt` | `llvm.sqrt.f64` | - |
| `sin` | `llvm.sin.f64` | - |
| `cos` | `llvm.cos.f64` | - |
| `exp` | `llvm.exp.f64` | - |
| `log` | `llvm.log.f64` | - |
| `log2` | `llvm.log2.f64` | - |
| `log10` | `llvm.log10.f64` | - |
| `pow` | `llvm.pow.f64` | - |
| `floor` | `llvm.floor.f64` | - |
| `ceil` | `llvm.ceil.f64` | - |
| `round` | `llvm.round.f64` | - |
| `trunc` | `llvm.trunc.f64` | - |
| `fabs` | `llvm.fabs.f64` | - |
| `copysign` | `llvm.copysign.f64` | - |
| `fma` | `llvm.fma.f64` | - |
| `tan` | - | `sin/cos` |
| `asin` | - | Taylor/Cordic |
| `acos` | - | Taylor/Cordic |
| `atan` | - | Taylor/Cordic |
| `atan2` | - | `atan` + quadrant |
| `sinh` | - | `(exp(x) - exp(-x)) / 2` |
| `cosh` | - | `(exp(x) + exp(-x)) / 2` |
| `tanh` | - | `sinh/cosh` |
| `hypot` | - | `sqrt(x*x + y*y)` with overflow handling |

### 5.2 Implementation

```sigil
/// Declare LLVM intrinsics
extern "llvm" {
    fn llvm_sqrt_f64(x: f64) → f64 = "llvm.sqrt.f64";
    fn llvm_sin_f64(x: f64) → f64 = "llvm.sin.f64";
    fn llvm_cos_f64(x: f64) → f64 = "llvm.cos.f64";
    fn llvm_exp_f64(x: f64) → f64 = "llvm.exp.f64";
    fn llvm_log_f64(x: f64) → f64 = "llvm.log.f64";
    fn llvm_log2_f64(x: f64) → f64 = "llvm.log2.f64";
    fn llvm_log10_f64(x: f64) → f64 = "llvm.log10.f64";
    fn llvm_pow_f64(x: f64, y: f64) → f64 = "llvm.pow.f64";
    fn llvm_floor_f64(x: f64) → f64 = "llvm.floor.f64";
    fn llvm_ceil_f64(x: f64) → f64 = "llvm.ceil.f64";
    fn llvm_round_f64(x: f64) → f64 = "llvm.round.f64";
    fn llvm_trunc_f64(x: f64) → f64 = "llvm.trunc.f64";
    fn llvm_fabs_f64(x: f64) → f64 = "llvm.fabs.f64";
    fn llvm_copysign_f64(x: f64, y: f64) → f64 = "llvm.copysign.f64";
    fn llvm_fma_f64(a: f64, b: f64, c: f64) → f64 = "llvm.fma.f64";
}

/// Mathematical constants
pub const PI: f64 = 3.14159265358979323846;
pub const E: f64 = 2.71828182845904523536;
pub const TAU: f64 = 6.28318530717958647692;

/// Square root
#[inline]
pub fn sqrt(x: f64) → f64! {
    unsafe { llvm_sqrt_f64(x) }
}

/// Sine
#[inline]
pub fn sin(x: f64) → f64! {
    unsafe { llvm_sin_f64(x) }
}

/// Cosine
#[inline]
pub fn cos(x: f64) → f64! {
    unsafe { llvm_cos_f64(x) }
}

/// Tangent (derived)
#[inline]
pub fn tan(x: f64) → f64! {
    sin(x) / cos(x)
}

/// Exponential
#[inline]
pub fn exp(x: f64) → f64! {
    unsafe { llvm_exp_f64(x) }
}

/// Natural logarithm
#[inline]
pub fn ln(x: f64) → f64! {
    unsafe { llvm_log_f64(x) }
}

/// Base-2 logarithm
#[inline]
pub fn log2(x: f64) → f64! {
    unsafe { llvm_log2_f64(x) }
}

/// Base-10 logarithm
#[inline]
pub fn log10(x: f64) → f64! {
    unsafe { llvm_log10_f64(x) }
}

/// Power
#[inline]
pub fn pow(x: f64, y: f64) → f64! {
    unsafe { llvm_pow_f64(x, y) }
}

/// Floor
#[inline]
pub fn floor(x: f64) → f64! {
    unsafe { llvm_floor_f64(x) }
}

/// Ceiling
#[inline]
pub fn ceil(x: f64) → f64! {
    unsafe { llvm_ceil_f64(x) }
}

/// Round to nearest
#[inline]
pub fn round(x: f64) → f64! {
    unsafe { llvm_round_f64(x) }
}

/// Truncate toward zero
#[inline]
pub fn trunc(x: f64) → f64! {
    unsafe { llvm_trunc_f64(x) }
}

/// Absolute value
#[inline]
pub fn fabs(x: f64) → f64! {
    unsafe { llvm_fabs_f64(x) }
}

/// Floating-point modulo
#[inline]
pub fn fmod(x: f64, y: f64) → f64! {
    x - trunc(x / y) * y
}

/// Hyperbolic sine
#[inline]
pub fn sinh(x: f64) → f64! {
    ≔ ex = exp(x);
    (ex - 1.0 / ex) * 0.5
}

/// Hyperbolic cosine
#[inline]
pub fn cosh(x: f64) → f64! {
    ≔ ex = exp(x);
    (ex + 1.0 / ex) * 0.5
}

/// Hyperbolic tangent
#[inline]
pub fn tanh(x: f64) → f64! {
    ≔ ex = exp(2.0 * x);
    (ex - 1.0) / (ex + 1.0)
}

/// Hypotenuse (overflow-safe)
#[inline]
pub fn hypot(x: f64, y: f64) → f64! {
    ≔ ax = fabs(x);
    ≔ ay = fabs(y);
    ⎇ ax > ay {
        ≔ r = ay / ax;
        ax * sqrt(1.0 + r * r)
    } ⎉ ⎇ ay > 0.0 {
        ≔ r = ax / ay;
        ay * sqrt(1.0 + r * r)
    } ⎉ {
        0.0
    }
}

/// Arc tangent (using polynomial approximation)
pub fn atan(x: f64) → f64! {
    // Reduce to |x| <= 1 using atan(x) = pi/2 - atan(1/x) for |x| > 1
    ⎇ fabs(x) > 1.0 {
        ≔ sign = ⎇ x > 0.0 { 1.0 } ⎉ { -1.0 };
        sign * (PI / 2.0 - atan_small(1.0 / fabs(x)))
    } ⎉ {
        atan_small(x)
    }
}

// Polynomial approximation for |x| <= 1
fn atan_small(x: f64) → f64! {
    // 7th order polynomial approximation
    ≔ x2 = x * x;
    x * (1.0 - x2 * (1.0/3.0 - x2 * (1.0/5.0 - x2 * (1.0/7.0))))
}

/// Arc tangent of y/x (full quadrant)
pub fn atan2(y: f64, x: f64) → f64! {
    ⎇ x > 0.0 {
        atan(y / x)
    } ⎉ ⎇ x < 0.0 && y >= 0.0 {
        atan(y / x) + PI
    } ⎉ ⎇ x < 0.0 && y < 0.0 {
        atan(y / x) - PI
    } ⎉ ⎇ x == 0.0 && y > 0.0 {
        PI / 2.0
    } ⎉ ⎇ x == 0.0 && y < 0.0 {
        -PI / 2.0
    } ⎉ {
        0.0  // x == 0 && y == 0
    }
}

/// Arc sine
pub fn asin(x: f64) → f64! {
    // asin(x) = atan(x / sqrt(1 - x^2))
    ⎇ fabs(x) >= 1.0 {
        ⎇ x > 0.0 { PI / 2.0 } ⎉ { -PI / 2.0 }
    } ⎉ {
        atan(x / sqrt(1.0 - x * x))
    }
}

/// Arc cosine
pub fn acos(x: f64) → f64! {
    PI / 2.0 - asin(x)
}
```

### 5.3 Integer Math

```sigil
/// Absolute value (integer)
#[inline]
pub fn abs(x: i64) → i64! {
    ⎇ x < 0 { -x } ⎉ { x }
}

/// Minimum
#[inline]
pub fn min(a: i64, b: i64) → i64! {
    ⎇ a < b { a } ⎉ { b }
}

/// Maximum
#[inline]
pub fn max(a: i64, b: i64) → i64! {
    ⎇ a > b { a } ⎉ { b }
}

/// Clamp to range
#[inline]
pub fn clamp(x: i64, lo: i64, hi: i64) → i64! {
    min(max(x, lo), hi)
}

/// Sign of integer
#[inline]
pub fn sign(x: i64) → i64! {
    ⎇ x < 0 { -1 } ⎉ ⎇ x > 0 { 1 } ⎉ { 0 }
}

/// Greatest common divisor
pub fn gcd(a: i64, b: i64) → i64! {
    vary x = abs(a);
    vary y = abs(b);
    while y != 0 {
        ≔ t = y;
        y = x % y;
        x = t;
    }
    x
}

/// Least common multiple
pub fn lcm(a: i64, b: i64) → i64! {
    ⎇ a == 0 || b == 0 { 0 } ⎉ { abs(a / gcd(a, b) * b) }
}
```

### 5.4 C FFI Wrappers

For compatibility with existing compiled code:

```sigil
/// Wrapper: i64 bits to f64, call function, f64 to i64 bits
#[no_mangle]
pub extern "C" fn sigil_sqrt(x: i64) → i64 {
    ≔ val: f64 = transmute(x);
    ≔ result = sqrt(val);
    transmute(result)
}

#[no_mangle]
pub extern "C" fn sigil_sin(x: i64) → i64 {
    ≔ val: f64 = transmute(x);
    ≔ result = sin(val);
    transmute(result)
}

// ... similar for all other math functions
```

### 5.5 Tests

```sigil
#[test]
fn test_sqrt() {
    assert_eq!(sqrt(4.0), 2.0);
    assert_eq!(sqrt(9.0), 3.0);
    assert!((sqrt(2.0) - 1.41421356).abs() < 0.0001);
}

#[test]
fn test_trig() {
    assert!((sin(0.0) - 0.0).abs() < 0.0001);
    assert!((sin(PI / 2.0) - 1.0).abs() < 0.0001);
    assert!((cos(0.0) - 1.0).abs() < 0.0001);
    assert!((cos(PI) - (-1.0)).abs() < 0.0001);
}

#[test]
fn test_exp_log() {
    assert!((exp(0.0) - 1.0).abs() < 0.0001);
    assert!((exp(1.0) - E).abs() < 0.0001);
    assert!((ln(E) - 1.0).abs() < 0.0001);
    assert!((log10(100.0) - 2.0).abs() < 0.0001);
    assert!((log2(8.0) - 3.0).abs() < 0.0001);
}

#[test]
fn test_pow() {
    assert_eq!(pow(2.0, 3.0), 8.0);
    assert_eq!(pow(10.0, 2.0), 100.0);
}

#[test]
fn test_rounding() {
    assert_eq!(floor(3.7), 3.0);
    assert_eq!(ceil(3.2), 4.0);
    assert_eq!(round(3.5), 4.0);
    assert_eq!(trunc(-3.7), -3.0);
}

#[test]
fn test_atan2_quadrants() {
    // Quadrant I
    assert!((atan2(1.0, 1.0) - PI / 4.0).abs() < 0.0001);
    // Quadrant II
    assert!((atan2(1.0, -1.0) - 3.0 * PI / 4.0).abs() < 0.0001);
    // Quadrant III
    assert!((atan2(-1.0, -1.0) - (-3.0 * PI / 4.0)).abs() < 0.0001);
    // Quadrant IV
    assert!((atan2(-1.0, 1.0) - (-PI / 4.0)).abs() < 0.0001);
}

#[test]
fn test_hypot_overflow() {
    // Should not overflow even with large values
    ≔ large = 1e200;
    ≔ result = hypot(large, large);
    assert!(result.is_finite());
    assert!((result / large - sqrt(2.0)).abs() < 0.0001);
}

#[test]
fn test_integer_math() {
    assert_eq!(abs(-42), 42);
    assert_eq!(min(3, 7), 3);
    assert_eq!(max(3, 7), 7);
    assert_eq!(clamp(5, 0, 10), 5);
    assert_eq!(clamp(-5, 0, 10), 0);
    assert_eq!(sign(-42), -1);
    assert_eq!(gcd(12, 18), 6);
    assert_eq!(lcm(4, 6), 12);
}
```

---

## Module 6: Time (`rt::time`)

### 6.1 Specification

```sigil
/// Timespec structure (POSIX)
#[repr(C)]
pub struct Timespec {
    pub tv_sec: i64,
    pub tv_nsec: i64,
}

/// Duration type
pub struct Duration {
    secs: u64,
    nanos: u32,
}

impl Duration {
    pub fn from_secs(secs: u64) → Self! {
        Duration { secs, nanos: 0 }
    }

    pub fn from_millis(millis: u64) → Self! {
        Duration {
            secs: millis / 1000,
            nanos: ((millis % 1000) * 1_000_000) as u32,
        }
    }

    pub fn from_nanos(nanos: u64) → Self! {
        Duration {
            secs: nanos / 1_000_000_000,
            nanos: (nanos % 1_000_000_000) as u32,
        }
    }

    pub fn as_secs(&self) → u64! { self.secs }
    pub fn as_millis(&self) → u64! { self.secs * 1000 + self.nanos as u64 / 1_000_000 }
    pub fn as_nanos(&self) → u64! { self.secs * 1_000_000_000 + self.nanos as u64 }
}

/// Instant for measuring elapsed time
pub struct Instant {
    ts: Timespec,
}

impl Instant {
    /// Get current instant
    pub fn now() → Self! {
        ≔ ts = sys::clock_gettime_monotonic().unwrap_or(Timespec { tv_sec: 0, tv_nsec: 0 });
        Instant { ts }
    }

    /// Elapsed time since this instant
    pub fn elapsed(&self) → Duration! {
        ≔ now = Self::now();
        ≔ secs = (now.ts.tv_sec - self.ts.tv_sec) as u64;
        ≔ nanos = now.ts.tv_nsec - self.ts.tv_nsec;
        ⎇ nanos < 0 {
            Duration { secs: secs - 1, nanos: (nanos + 1_000_000_000) as u32 }
        } ⎉ {
            Duration { secs, nanos: nanos as u32 }
        }
    }
}

/// Get current time in milliseconds since Unix epoch
#[no_mangle]
pub extern "C" fn sigil_now() → i64 {
    ≔ ts = sys::clock_gettime_realtime().unwrap_or(Timespec { tv_sec: 0, tv_nsec: 0 });
    ts.tv_sec * 1000 + ts.tv_nsec / 1_000_000
}
```

### 6.2 Tests

```sigil
#[test]
fn test_now_reasonable() {
    ≔ now = sigil_now();
    // Should be after 2024-01-01 (1704067200000 ms)
    assert!(now > 1704067200000);
}

#[test]
fn test_instant_elapsed() {
    ≔ start = Instant::now();

    // Busy wait for ~10ms
    vary count = 0u64;
    while count < 1_000_000 {
        count += 1;
    }

    ≔ elapsed = start.elapsed();
    assert!(elapsed.as_millis() >= 1);  // At least some time passed
}

#[test]
fn test_duration_conversions() {
    ≔ d = Duration::from_millis(1500);
    assert_eq!(d.as_secs(), 1);
    assert_eq!(d.as_millis(), 1500);
    assert_eq!(d.as_nanos(), 1_500_000_000);
}
```

---

## Module 7: System (`rt::sys`)

### 7.1 Process Control

```sigil
/// Exit process
#[no_mangle]
pub extern "C" fn sigil_exit(code: i64) → ! {
    sys::exit(code as i32)
}

/// Get environment variable
#[no_mangle]
pub extern "C" fn sigil_getenv(name: *const u8) → *mut u8 {
    ⎇ name.is_null() {
        return null_mut();
    }

    // Read environment block and search
    // Implementation depends on platform
    // Returns heap-allocated String if found

    // Placeholder: would need to parse environ
    null_mut()
}
```

### 7.2 Panic Handler

```sigil
/// Panic handler (called on unwrap None, etc.)
#[panic_handler]
fn panic(msg: &str) → ! {
    // Write to stderr
    ≔ _ = sys::write(2, b"panic: ");
    ≔ _ = sys::write(2, msg.as_bytes());
    ≔ _ = sys::write(2, b"\n");

    sys::exit(1)
}
```

---

## TDD Roadmap

### Phase A: Platform Syscalls (Week 1-2)

**Goal:** Direct kernel access without libc.

| Day | Task | Tests |
|-----|------|-------|
| 1-2 | Linux x86_64 syscall wrapper | `test_syscall_write`, `test_syscall_mmap` |
| 3-4 | Darwin x86_64 syscall wrapper | Same tests on macOS |
| 5-6 | Windows x64 syscall wrapper | Same tests on Windows |
| 7-8 | Error type (Errno) | `test_errno_from`, `test_errno_message` |

**Deliverables:**
- [ ] `sys::write()` works on all platforms
- [ ] `sys::mmap_anon()` works on all platforms
- [ ] `sys::clock_gettime()` works on all platforms

### Phase B: Memory Allocator (Week 3-4)

**Goal:** Replace malloc/free with native allocator.

| Day | Task | Tests |
|-----|------|-------|
| 1-2 | Arena allocator | `test_arena_alloc`, `test_arena_alignment` |
| 3-4 | Global allocator | `test_global_alloc`, `test_realloc` |
| 5-6 | Large allocations | `test_arena_large`, `test_grow` |
| 7-8 | Reset/cleanup | `test_arena_reset`, `test_arena_drop` |

**Deliverables:**
- [ ] `sigil_alloc()` works without libc
- [ ] `sigil_free()` works (or no-op for arena)
- [ ] `sigil_realloc()` works

### Phase C: Core Types (Week 5-6)

**Goal:** Vec, String, Option, Result in pure Sigil.

| Day | Task | Tests |
|-----|------|-------|
| 1-2 | Vec<T> implementation | `test_vec_push_pop`, `test_vec_grow` |
| 3-4 | String implementation | `test_string_basic`, `test_string_concat` |
| 5-6 | Option<T> methods | `test_option_unwrap`, `test_option_map` |
| 7-8 | Result<T, E> methods | `test_result_unwrap`, `test_result_map` |

**Deliverables:**
- [ ] Vec<T> fully functional
- [ ] String fully functional
- [ ] Option/Result methods complete

### Phase D: I/O (Week 7-8)

**Goal:** Print and file I/O without stdio.h.

| Day | Task | Tests |
|-----|------|-------|
| 1-2 | Integer-to-string conversion | `test_i64_to_str` |
| 3-4 | Float-to-string conversion | `test_f64_to_str` |
| 5-6 | Print functions | `test_print_int`, `test_print_str` |
| 7-8 | File I/O | `test_file_write_read`, `test_file_exists` |

**Deliverables:**
- [ ] `sigil_print_int()` works without printf
- [ ] `sigil_print_float()` works without printf
- [ ] File read/write works without fopen

### Phase E: Math (Week 9-10)

**Goal:** Math functions via LLVM intrinsics.

| Day | Task | Tests |
|-----|------|-------|
| 1-2 | LLVM intrinsic declarations | `test_sqrt`, `test_sin`, `test_cos` |
| 3-4 | Derived functions (tan, etc.) | `test_tan`, `test_sinh`, `test_cosh` |
| 5-6 | Inverse trig (atan, etc.) | `test_atan`, `test_atan2`, `test_asin` |
| 7-8 | Integer math | `test_abs`, `test_gcd`, `test_lcm` |

**Deliverables:**
- [ ] All 23 math functions work without libm
- [ ] Integer math functions complete

### Phase F: Integration (Week 11-12)

**Goal:** Replace C runtime completely.

| Day | Task | Tests |
|-----|------|-------|
| 1-2 | Link runtime with LLVM codegen | `test_hello_world_native` |
| 3-4 | Run existing test suite | All 466+ tests |
| 5-6 | Jormungandr self-compile | Bootstrap test |
| 7-8 | Performance benchmarks | Comparison vs C runtime |

**Deliverables:**
- [ ] All existing tests pass with native runtime
- [ ] Jormungandr compiles itself with native runtime
- [ ] Binary size comparison documented

---

## Success Metrics

| Metric | C Runtime | Target | Notes |
|--------|-----------|--------|-------|
| Binary size (hello world) | ~20KB | <15KB | No libc overhead |
| Startup time | ~1ms | <0.5ms | No dynamic linking |
| External dependencies | 6 headers | 0 | Fully self-contained |
| Platform support | 3 | 3 | Linux, macOS, Windows |
| Test pass rate | 100% | 100% | No regressions |

---

## File Structure

```
parser/
├── src/
│   └── rt/                    # NEW: Native runtime
│       ├── mod.sg             # Module root
│       ├── sys/               # Platform syscalls
│       │   ├── mod.sg
│       │   ├── linux_x64.sg
│       │   ├── darwin_x64.sg
│       │   ├── windows_x64.sg
│       │   └── errno.sg
│       ├── alloc/             # Memory allocator
│       │   ├── mod.sg
│       │   ├── arena.sg
│       │   └── global.sg
│       ├── types/             # Core types
│       │   ├── mod.sg
│       │   ├── vec.sg
│       │   ├── string.sg
│       │   ├── option.sg
│       │   └── result.sg
│       ├── io/                # I/O functions
│       │   ├── mod.sg
│       │   ├── print.sg
│       │   └── file.sg
│       ├── math/              # Math functions
│       │   ├── mod.sg
│       │   ├── intrinsics.sg
│       │   ├── trig.sg
│       │   └── integer.sg
│       └── time/              # Time functions
│           └── mod.sg
└── runtime/
    └── sigil_runtime.c        # DEPRECATED after Phase F
```

---

## Revision History

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0.0 | 2026-01-21 | Claude Code | Initial specification |
