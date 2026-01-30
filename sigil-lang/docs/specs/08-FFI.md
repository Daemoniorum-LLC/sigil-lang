# Sigil Foreign Function Interface (FFI) Specification

> *"To speak with spirits from other realms, one must learn their tongue."*

## 1. Philosophy: Bridges Between Worlds

Sigil does not exist in isolation. It must communicate with:
- **C** — The lingua franca of systems programming
- **Rust** — Daemoniorum's existing infrastructure (Nyx, Infernum, Arcanum, Aether)
- **System APIs** — OS kernels, hardware interfaces
- **Legacy code** — Existing libraries and ecosystems

The FFI is designed around **safety boundaries** — foreign code is inherently `reported~` (external evidence) until validated.

---

## 2. C Interoperability

### 2.1 Calling C Functions

```sigil
// Declare external C function
extern "C" {
    fn strlen(s: *const c_char) -> usize
    fn malloc(size: usize) -> *mut c_void
    fn free(ptr: *mut c_void)
    fn printf(format: *const c_char, ...) -> c_int
}

// Usage (requires unsafe block)
fn string_length(s: &str) -> usize {
    let c_str = CString·new(s)!
    unsafe {
        strlen(c_str.as_ptr())~  // Result is reported (external)
    }
}
```

### 2.2 C Types

```sigil
// Primitive C type aliases
type c_char = i8       // or u8 on some platforms
type c_short = i16
type c_int = i32
type c_long = i64      // platform-dependent
type c_longlong = i64
type c_float = f32
type c_double = f64
type c_void = ()

// Unsigned variants
type c_uchar = u8
type c_ushort = u16
type c_uint = u32
type c_ulong = u64
type c_ulonglong = u64

// Size types
type size_t = usize
type ssize_t = isize
type ptrdiff_t = isize

// Pointer types
type c_str = *const c_char
type c_str_mut = *mut c_char
```

### 2.3 C-Compatible Structs

```sigil
// Ensure C-compatible layout
//@ rune: repr(C)
struct Point {
    x: c_double,
    y: c_double,
}

//@ rune: repr(C, packed)
struct PackedHeader {
    magic: [u8; 4],
    version: u16,
    flags: u16,
}

//@ rune: repr(C, align(16))
struct AlignedBuffer {
    data: [u8; 64],
}
```

### 2.4 C Enums

```sigil
//@ rune: repr(C)
enum Status {
    Ok = 0,
    Error = -1,
    Pending = 1,
}

//@ rune: repr(i32)
enum ErrorCode {
    Success = 0,
    NotFound = 404,
    Internal = 500,
}
```

### 2.5 Callbacks and Function Pointers

```sigil
// C function pointer type
type Callback = extern "C" fn(data: *mut c_void, len: usize) -> c_int

// Passing Sigil functions to C
extern "C" fn my_callback(data: *mut c_void, len: usize) -> c_int {
    // Implementation
    0
}

extern "C" {
    fn register_callback(cb: Callback, user_data: *mut c_void)
}

// Usage
unsafe {
    register_callback(my_callback, std·ptr·null_mut())
}
```

### 2.6 Strings

```sigil
use ffi·c·{CString, CStr}

// Sigil string → C string
let sigil_str = "Hello, World!"
let c_string = CString·new(sigil_str)?  // Adds null terminator
let ptr: *const c_char = c_string.as_ptr()

// C string → Sigil string
unsafe {
    let c_str = CStr·from_ptr(external_ptr)~
    let sigil_str: &str = c_str.to_str()?
}

// C string with explicit lifetime
unsafe {
    let borrowed: &CStr = CStr·from_ptr(ptr)~
    // borrowed only valid while ptr is valid
}
```

---

## 3. Rust Interoperability

Sigil has first-class Rust interop — essential for Daemoniorum's ecosystem.

### 3.1 Importing Rust Crates

```sigil
// In Sigil.toml
[dependencies.rust]
arcanum = { crate = "arcanum", version = "0.1" }
infernum = { crate = "infernum", version = "0.1" }
nyx-kernel = { crate = "nyx-kernel", version = "0.1" }

// In Sigil code
use rust·arcanum·{encrypt, decrypt, AesGcm}
use rust·infernum·{LlmEngine, ChatMessage}
use rust·nyx_kernel·{Syscall, Process}
```

### 3.2 Calling Rust Functions

```sigil
// Rust functions are called directly
let cipher = rust·arcanum·AesGcm·new(key)
let encrypted = cipher.encrypt(plaintext, nonce)~

// Async Rust functions
let response = rust·infernum·complete(prompt)|await~

// Rust traits work seamlessly
impl rust·serde·Serialize for MyType {
    fn serialize<S>(&self, serializer: S) -> Result<S·Ok, S·Error>
    where S: rust·serde·Serializer
    {
        // ...
    }
}
```

### 3.3 Type Mapping

| Sigil Type | Rust Type |
|------------|-----------|
| `i8..i128` | `i8..i128` |
| `u8..u128` | `u8..u128` |
| `f32, f64` | `f32, f64` |
| `bool` | `bool` |
| `char` | `char` |
| `str` | `str` |
| `String` | `String` |
| `[T]` | `Vec<T>` |
| `[T; N]` | `[T; N]` |
| `&T` | `&T` |
| `&mut T` | `&mut T` |
| `Box<T>` | `Box<T>` |
| `Option<T>` | `Option<T>` |
| `Result<T, E>` | `Result<T, E>` |

### 3.4 Exposing Sigil to Rust

```sigil
// Make Sigil functions callable from Rust
//@ rune: export_rust
pub fn process_data(input: &[u8]) -> Vec<u8> {
    input|τ{transform}|collect
}

// Generates Rust-compatible extern
// extern "Rust" fn process_data(input: &[u8]) -> Vec<u8>
```

### 3.5 Shared Memory Model

```sigil
// Sigil and Rust share the same memory model
// Ownership transfers seamlessly

// Take ownership from Rust
let data: Vec<u8> = rust·function_returning_vec()~

// Give ownership to Rust
rust·function_taking_vec(sigil_vec)

// Borrow to/from Rust
let slice: &[u8] = rust·get_slice()~
rust·process_slice(&sigil_data)
```

---

## 4. Platform APIs

### 4.1 POSIX

```sigil
use ffi·posix·{open, read, write, close, O_RDONLY, O_WRONLY}

fn read_file(path: &str) -> Result<Vec<u8>, Error~> {
    let fd = unsafe {
        open(path.as_c_str().as_ptr(), O_RDONLY)~
    }
    ward!(fd >= 0, "Failed to open file")

    let mut buffer = [0u8; 4096]
    let mut result = Vec·new()

    loop {
        let n = unsafe {
            read(fd, buffer.as_mut_ptr() as *mut c_void, buffer.len())~
        }
        if n <= 0 { break }
        result|extend_from_slice(&buffer[..n as usize])
    }

    unsafe { close(fd) }
    Ok(result)
}
```

### 4.2 Windows API

```sigil
use ffi·windows·{
    CreateFileW, ReadFile, WriteFile, CloseHandle,
    HANDLE, DWORD, GENERIC_READ, OPEN_EXISTING,
}

//@ rune: cfg(target_os = "windows")
fn read_file_windows(path: &str) -> Result<Vec<u8>, Error~> {
    let wide_path: Vec<u16> = path.encode_utf16().chain([0]).collect()

    let handle = unsafe {
        CreateFileW(
            wide_path.as_ptr(),
            GENERIC_READ,
            0,
            std·ptr·null_mut(),
            OPEN_EXISTING,
            0,
            std·ptr·null_mut(),
        )~
    }
    // ...
}
```

### 4.3 System Calls (Nyx OS)

```sigil
// Direct syscall interface for Nyx OS
use ffi·nyx·syscall·{syscall, SYS_read, SYS_write, SYS_open}

fn nyx_read(fd: i32, buf: &mut [u8]) -> isize {
    unsafe {
        syscall(SYS_read, fd, buf.as_mut_ptr(), buf.len())~
    }
}
```

---

## 5. Safety and Evidence

### 5.1 FFI Evidence Rules

All FFI calls return `reported~` evidence — external data is untrusted:

```sigil
// C function returns external evidence
extern "C" {
    fn get_data() -> *mut Data
}

let ptr~ = unsafe { get_data()~ }  // Explicitly reported

// Must validate before trusting
let validated! = ptr~|validate!  // Becomes known after validation

// Or acknowledge the trust boundary
let trusted‽ = unsafe { (*ptr~)|trust }
```

### 5.2 Safe Wrappers

```sigil
// Wrap unsafe FFI in safe interface
mod safe_bindings {
    use super::ffi

    /// Safe wrapper around C library
    pub fn get_string() -> Result<String!, FfiError~> {
        let ptr~ = unsafe { ffi·get_string_ptr()~ }

        if ptr~.is_null() {
            return Err(FfiError·NullPointer)~
        }

        let c_str~ = unsafe { CStr·from_ptr(ptr~)~ }
        let string! = c_str~.to_str()?.to_owned()!

        // Free the C string
        unsafe { ffi·free_string(ptr~) }

        Ok(string!)
    }
}
```

### 5.3 Panic Safety

```sigil
// Catch panics at FFI boundary
extern "C" fn callback_from_c(data: *mut c_void) -> c_int {
    // Panics must not cross FFI boundary!
    std·panic·catch_unwind(|| {
        let data = unsafe { &mut *(data as *mut MyData) }
        process(data)
    })
    |map(|_| 0)
    |unwrap_or(-1)
}
```

---

## 6. Memory Management Across Boundaries

### 6.1 Ownership Transfer

```sigil
// Give ownership to C (C must free)
//@ rune: repr(C)
struct Buffer {
    data: *mut u8,
    len: usize,
    capacity: usize,
}

fn create_buffer_for_c() -> Buffer {
    let mut vec = vec![0u8; 1024]
    let buf = Buffer {
        data: vec.as_mut_ptr(),
        len: vec.len(),
        capacity: vec.capacity(),
    }
    std·mem·forget(vec)  // Don't drop, C owns it now
    buf
}

// Take ownership from C
fn take_buffer_from_c(buf: Buffer) -> Vec<u8> {
    unsafe {
        Vec·from_raw_parts(buf.data, buf.len, buf.capacity)
    }
}
```

### 6.2 Borrowed Data

```sigil
// Borrow to C (Sigil retains ownership)
fn process_with_c(data: &[u8]) {
    extern "C" {
        fn c_process(ptr: *const u8, len: usize)
    }

    unsafe {
        c_process(data.as_ptr(), data.len())
    }
    // data still valid here, Sigil owns it
}
```

### 6.3 Custom Allocators

```sigil
// Use C allocator
use ffi·c·{malloc, free, realloc}

struct CAllocator

impl Allocator for CAllocator {
    fn allocate(&self, layout: Layout) -> Result<NonNull<[u8]>, AllocError> {
        unsafe {
            let ptr = malloc(layout.size())
            if ptr.is_null() {
                Err(AllocError)
            } else {
                Ok(NonNull·slice_from_raw_parts(
                    NonNull·new_unchecked(ptr as *mut u8),
                    layout.size()
                ))
            }
        }
    }

    fn deallocate(&self, ptr: NonNull<u8>, _layout: Layout) {
        unsafe { free(ptr.as_ptr() as *mut c_void) }
    }
}
```

---

## 7. Bindgen Integration

### 7.1 Automatic Binding Generation

```sigil
// In build.sg
use sigil·bindgen·Builder

fn main() {
    Builder·default()
        |header("wrapper.h")
        |clang_arg("-I/usr/include")
        |allowlist_function("my_.*")
        |allowlist_type("My.*")
        |generate()
        |write_to_file("src/bindings.sg")
}

// Generated bindings
mod bindings {
    //@ rune: auto_generated
    extern "C" {
        pub fn my_init() -> c_int
        pub fn my_process(data: *mut MyData) -> c_int
        pub fn my_cleanup()
    }

    //@ rune: repr(C)
    pub struct MyData {
        pub field1: c_int,
        pub field2: *mut c_char,
    }
}
```

### 7.2 Rust Crate Bindings

```sigil
// Automatic Rust type import
use sigil·rustgen·import_crate

// In build.sg
import_crate("serde", features: ["derive"])
import_crate("tokio", features: ["full"])

// Types and functions automatically available
use rust·serde·{Serialize, Deserialize}
use rust·tokio·{spawn, select}
```

---

## 8. Platform-Specific FFI

### 8.1 Conditional Compilation

```sigil
//@ rune: cfg(target_os = "linux")
mod linux {
    extern "C" {
        fn epoll_create1(flags: c_int) -> c_int
        fn epoll_ctl(epfd: c_int, op: c_int, fd: c_int, event: *mut epoll_event) -> c_int
    }
}

//@ rune: cfg(target_os = "macos")
mod macos {
    extern "C" {
        fn kqueue() -> c_int
        fn kevent(kq: c_int, ...) -> c_int
    }
}

//@ rune: cfg(target_os = "windows")
mod windows {
    extern "C" {
        fn CreateIoCompletionPort(...) -> HANDLE
    }
}
```

### 8.2 Architecture-Specific

```sigil
//@ rune: cfg(target_arch = "x86_64")
mod x86_64 {
    // SIMD intrinsics
    extern "C" {
        fn _mm256_add_ps(a: __m256, b: __m256) -> __m256
    }
}

//@ rune: cfg(target_arch = "aarch64")
mod aarch64 {
    extern "C" {
        fn vaddq_f32(a: float32x4_t, b: float32x4_t) -> float32x4_t
    }
}
```

---

## 9. Inline Assembly

### 9.1 Basic Inline Assembly

```sigil
// Global assembly
//@ rune: global_asm
asm!(
    ".section .text",
    ".global my_asm_func",
    "my_asm_func:",
    "    mov rax, rdi",
    "    ret",
)

// Inline assembly
fn cpuid(leaf: u32) -> (u32, u32, u32, u32) {
    let (eax, ebx, ecx, edx): (u32, u32, u32, u32)
    unsafe {
        asm!(
            "cpuid",
            inout("eax") leaf => eax,
            out("ebx") ebx,
            out("ecx") ecx,
            out("edx") edx,
        )
    }
    (eax, ebx, ecx, edx)
}
```

### 9.2 Constraints and Clobbers

```sigil
fn syscall3(n: usize, a1: usize, a2: usize, a3: usize) -> usize {
    let ret: usize
    unsafe {
        asm!(
            "syscall",
            inout("rax") n => ret,
            in("rdi") a1,
            in("rsi") a2,
            in("rdx") a3,
            out("rcx") _,
            out("r11") _,
            options(nostack),
        )
    }
    ret
}
```

---

## 10. Complete Example: Wrapping a C Library

```sigil
//! Safe wrapper around libpng

mod ffi {
    use super::*

    //@ rune: link(name = "png")
    extern "C" {
        fn png_create_read_struct(
            version: *const c_char,
            error_ptr: *mut c_void,
            error_fn: Option<extern "C" fn(*mut c_void, *const c_char)>,
            warn_fn: Option<extern "C" fn(*mut c_void, *const c_char)>,
        ) -> *mut PngStruct

        fn png_create_info_struct(png_ptr: *mut PngStruct) -> *mut PngInfo
        fn png_destroy_read_struct(
            png_ptr: *mut *mut PngStruct,
            info_ptr: *mut *mut PngInfo,
            end_info_ptr: *mut *mut PngInfo,
        )
        fn png_read_png(
            png_ptr: *mut PngStruct,
            info_ptr: *mut PngInfo,
            transforms: c_int,
            params: *mut c_void,
        )
        fn png_get_rows(png_ptr: *mut PngStruct, info_ptr: *mut PngInfo) -> *mut *mut u8
        fn png_get_image_width(png_ptr: *mut PngStruct, info_ptr: *mut PngInfo) -> u32
        fn png_get_image_height(png_ptr: *mut PngStruct, info_ptr: *mut PngInfo) -> u32
    }

    pub enum PngStruct {}
    pub enum PngInfo {}
}

/// Safe PNG reader
pub struct PngReader {
    png_ptr: *mut ffi·PngStruct,
    info_ptr: *mut ffi·PngInfo,
}

impl PngReader {
    pub fn new() -> Result<Self!, PngError~> {
        let png_ptr~ = unsafe {
            ffi·png_create_read_struct(
                PNG_LIBPNG_VER_STRING.as_ptr(),
                std·ptr·null_mut(),
                None,
                None,
            )~
        }

        if png_ptr~.is_null() {
            return Err(PngError·CreateFailed)~
        }

        let info_ptr~ = unsafe { ffi·png_create_info_struct(png_ptr~)~ }

        if info_ptr~.is_null() {
            unsafe {
                ffi·png_destroy_read_struct(&mut png_ptr~, std·ptr·null_mut(), std·ptr·null_mut())
            }
            return Err(PngError·CreateFailed)~
        }

        Ok(Self! {
            png_ptr: png_ptr~|trust‽,
            info_ptr: info_ptr~|trust‽,
        })
    }

    pub fn dimensions(&self) -> (u32!, u32!) {
        unsafe {
            let width = ffi·png_get_image_width(self.png_ptr, self.info_ptr)!
            let height = ffi·png_get_image_height(self.png_ptr, self.info_ptr)!
            (width, height)
        }
    }
}

impl Drop for PngReader {
    fn drop(&mut self) {
        unsafe {
            ffi·png_destroy_read_struct(
                &mut self.png_ptr,
                &mut self.info_ptr,
                std·ptr·null_mut(),
            )
        }
    }
}
```

---

## 11. FFI Morphemes

| Morpheme | Purpose | Example |
|----------|---------|---------|
| `·as_ptr` | Get raw pointer | `slice·as_ptr` |
| `·as_mut_ptr` | Get mutable raw pointer | `slice·as_mut_ptr` |
| `·from_raw` | Construct from raw parts | `Vec·from_raw(ptr, len, cap)` |
| `·into_raw` | Consume and return raw | `string·into_raw` |
| `·as_c_str` | Convert to C string | `string·as_c_str` |
| `·from_c_str` | Parse from C string | `String·from_c_str(ptr)~` |
| `·trust` | Assert FFI value as trusted | `ptr~·trust` |
| `·validate` | Validate external data | `data~·validate!` |
