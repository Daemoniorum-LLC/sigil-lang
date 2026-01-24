// Sigil Native Runtime - macOS ARM64 (Apple Silicon)
// Pure assembly, no libc dependency
// Uses BSD syscall interface via svc #0x80

.section __TEXT,__text,regular,pure_instructions

// ============================================================================
// macOS ARM64 Syscall Numbers (BSD)
// ============================================================================
.set SYS_exit,       1
.set SYS_read,       3
.set SYS_write,      4
.set SYS_open,       5
.set SYS_close,      6
.set SYS_mmap,       197
.set SYS_munmap,     73
.set SYS_access,     33
.set SYS_lseek,      199
.set SYS_fstat,      339

// File flags
.set O_RDONLY,   0x0000
.set O_WRONLY,   0x0001
.set O_RDWR,     0x0002
.set O_CREAT,    0x0200
.set O_TRUNC,    0x0400

// mmap flags
.set PROT_READ,  0x01
.set PROT_WRITE, 0x02
.set MAP_PRIVATE, 0x0002
.set MAP_ANONYMOUS, 0x1000

.set ARENA_SIZE, 0x100000    // 1MB arenas

// ============================================================================
// Entry Point
// ============================================================================
.global _start
.align 4
_start:
    // Call main function
    bl _sigil_main

    // Exit with return value
    mov x16, SYS_exit
    svc #0x80

// ============================================================================
// Print Functions
// ============================================================================

// sigil_println(str: *const u8, len: i64)
.global _sigil_println
.align 4
_sigil_println:
    stp x29, x30, [sp, #-32]!
    stp x19, x20, [sp, #16]
    mov x29, sp
    mov x19, x0              // Save string pointer
    mov x20, x1              // Save length

    // Write the string
    mov x16, SYS_write
    mov x0, #1               // stdout
    mov x1, x19
    mov x2, x20
    svc #0x80

    // Write newline
    adrp x1, newline@PAGE
    add x1, x1, newline@PAGEOFF
    mov x16, SYS_write
    mov x0, #1
    mov x2, #1
    svc #0x80

    ldp x19, x20, [sp, #16]
    ldp x29, x30, [sp], #32
    ret

.section __DATA,__const
.align 2
newline:
    .byte 10

.section __TEXT,__text

// sigil_print_int(n: i64)
.global _sigil_print_int
.align 4
_sigil_print_int:
    stp x29, x30, [sp, #-64]!
    mov x29, sp

    mov x9, x0               // Number to print
    add x10, sp, #56         // Buffer end

    // Handle negative
    mov x11, #0              // Sign flag
    cmp x9, #0
    b.ge .Lpositive
    neg x9, x9
    mov x11, #1

.Lpositive:
    mov x12, #10

.Ldigit_loop:
    udiv x13, x9, x12
    msub x14, x13, x12, x9   // remainder
    add x14, x14, #'0'
    sub x10, x10, #1
    strb w14, [x10]
    mov x9, x13
    cbnz x9, .Ldigit_loop

    // Add minus if negative
    cbz x11, .Lprint_num
    sub x10, x10, #1
    mov w14, #'-'
    strb w14, [x10]

.Lprint_num:
    // Calculate length
    add x2, sp, #56
    sub x2, x2, x10

    // Write
    mov x16, SYS_write
    mov x0, #1               // stdout
    mov x1, x10
    svc #0x80

    // Write newline
    adrp x1, newline@PAGE
    add x1, x1, newline@PAGEOFF
    mov x16, SYS_write
    mov x0, #1
    mov x2, #1
    svc #0x80

    ldp x29, x30, [sp], #64
    ret

// ============================================================================
// Memory Management - Arena Allocator
// ============================================================================

.section __DATA,__bss
.align 3
_arena_current:   .quad 0
_arena_bump:      .quad 0
_arena_end:       .quad 0

.section __TEXT,__text

// sigil_alloc(size: i64) -> *mut u8
.global _sigil_alloc
.align 4
_sigil_alloc:
    stp x29, x30, [sp, #-32]!
    stp x19, x20, [sp, #16]
    mov x29, sp
    mov x19, x0              // Size to allocate

    // Check if we have space in current arena
    adrp x9, _arena_bump@PAGE
    ldr x10, [x9, _arena_bump@PAGEOFF]
    cbz x10, .Lneed_arena

    add x10, x10, x19
    adrp x11, _arena_end@PAGE
    ldr x12, [x11, _arena_end@PAGEOFF]
    cmp x10, x12
    b.gt .Lneed_arena

    // Bump allocate
    adrp x9, _arena_bump@PAGE
    ldr x0, [x9, _arena_bump@PAGEOFF]
    add x10, x0, x19
    str x10, [x9, _arena_bump@PAGEOFF]

    ldp x19, x20, [sp, #16]
    ldp x29, x30, [sp], #32
    ret

.Lneed_arena:
    // Allocate new arena via mmap
    mov x16, SYS_mmap
    mov x0, #0               // addr = NULL
    mov x1, ARENA_SIZE       // len
    mov x2, #(PROT_READ | PROT_WRITE)
    mov x3, #(MAP_PRIVATE | MAP_ANONYMOUS)
    mov x4, #-1              // fd = -1
    mov x5, #0               // offset = 0
    svc #0x80

    cmn x0, #1
    b.eq .Lalloc_fail

    // Set up arena
    mov x20, x0              // Save result
    adrp x9, _arena_current@PAGE
    str x0, [x9, _arena_current@PAGEOFF]

    add x10, x0, x19
    adrp x9, _arena_bump@PAGE
    str x10, [x9, _arena_bump@PAGEOFF]

    add x10, x0, ARENA_SIZE
    adrp x9, _arena_end@PAGE
    str x10, [x9, _arena_end@PAGEOFF]

    mov x0, x20

    ldp x19, x20, [sp, #16]
    ldp x29, x30, [sp], #32
    ret

.Lalloc_fail:
    mov x0, #0
    ldp x19, x20, [sp, #16]
    ldp x29, x30, [sp], #32
    ret

// sigil_free(ptr: *mut u8)
.global _sigil_free
.align 4
_sigil_free:
    // No-op for arena allocator
    ret

// ============================================================================
// String Functions
// ============================================================================

// sigil_string_from(data: *const u8, len: i64) -> *mut String
.global _sigil_string_from
.align 4
_sigil_string_from:
    stp x29, x30, [sp, #-48]!
    stp x19, x20, [sp, #16]
    stp x21, x22, [sp, #32]
    mov x29, sp
    mov x19, x0              // data
    mov x20, x1              // len

    // Allocate: 16 bytes header + len + 1
    add x0, x1, #17
    bl _sigil_alloc

    cbz x0, .Lstring_from_fail
    mov x21, x0              // Save result

    // Store length and capacity
    str x20, [x0]
    str x20, [x0, #8]

    // Copy data
    add x0, x21, #16
    mov x1, x19
    mov x2, x20

.Lcopy_loop:
    cbz x2, .Lcopy_done
    ldrb w3, [x1], #1
    strb w3, [x0], #1
    sub x2, x2, #1
    b .Lcopy_loop

.Lcopy_done:
    // Null terminate
    strb wzr, [x0]

    mov x0, x21
    ldp x21, x22, [sp, #32]
    ldp x19, x20, [sp, #16]
    ldp x29, x30, [sp], #48
    ret

.Lstring_from_fail:
    mov x0, #0
    ldp x21, x22, [sp, #32]
    ldp x19, x20, [sp, #16]
    ldp x29, x30, [sp], #48
    ret

// sigil_string_len(s: *const String) -> i64
.global _sigil_string_len
.align 4
_sigil_string_len:
    ldr x0, [x0]
    ret

// sigil_string_as_ptr(s: *const String) -> *const u8
.global _sigil_string_as_ptr
.align 4
_sigil_string_as_ptr:
    add x0, x0, #16
    ret

// sigil_string_print(s: *const String)
.global _sigil_string_print
.align 4
_sigil_string_print:
    ldr x2, [x0]             // length
    add x1, x0, #16          // data ptr
    mov x0, #1               // stdout
    mov x16, SYS_write
    svc #0x80
    ret

// sigil_string_eq(a: *const String, b: *const String) -> i64
.global _sigil_string_eq
.align 4
_sigil_string_eq:
    ldr x2, [x0]             // len(a)
    ldr x3, [x1]             // len(b)
    cmp x2, x3
    b.ne .Lstrings_not_equal

    add x0, x0, #16
    add x1, x1, #16

.Lcmp_loop:
    cbz x2, .Lstrings_equal
    ldrb w4, [x0], #1
    ldrb w5, [x1], #1
    cmp w4, w5
    b.ne .Lstrings_not_equal
    sub x2, x2, #1
    b .Lcmp_loop

.Lstrings_equal:
    mov x0, #1
    ret

.Lstrings_not_equal:
    mov x0, #0
    ret

// ============================================================================
// Vec Functions
// ============================================================================

// sigil_vec_new() -> *mut Vec
.global _sigil_vec_new
.align 4
_sigil_vec_new:
    stp x29, x30, [sp, #-16]!
    mov x29, sp

    mov x0, #16
    bl _sigil_alloc

    cbz x0, .Lvec_new_fail
    str xzr, [x0]            // len = 0
    str xzr, [x0, #8]        // capacity = 0

.Lvec_new_fail:
    ldp x29, x30, [sp], #16
    ret

// sigil_vec_push(v: *mut Vec, val: i64)
.global _sigil_vec_push
.align 4
_sigil_vec_push:
    stp x29, x30, [sp, #-48]!
    stp x19, x20, [sp, #16]
    stp x21, x22, [sp, #32]
    mov x29, sp
    mov x19, x0              // vec
    mov x20, x1              // value

    ldr x21, [x19]           // len
    ldr x22, [x19, #8]       // capacity
    cmp x21, x22
    b.lt .Lhas_capacity

    // Need to grow
    cbz x22, .Lstart_cap
    lsl x0, x22, #1          // double capacity
    b .Lalloc_new

.Lstart_cap:
    mov x0, #8

.Lalloc_new:
    mov x22, x0              // new capacity
    lsl x0, x0, #3           // * 8 bytes
    add x0, x0, #16          // + header
    bl _sigil_alloc

    cbz x0, .Lpush_fail

    // Copy old data
    ldr x2, [x19]            // old len
    cbz x2, .Lskip_copy

    add x3, x0, #16          // new data
    add x4, x19, #16         // old data

.Lvec_copy:
    cbz x2, .Lskip_copy
    ldr x5, [x4], #8
    str x5, [x3], #8
    sub x2, x2, #1
    b .Lvec_copy

.Lskip_copy:
    ldr x2, [x19]            // old len
    str x2, [x0]             // new len
    str x22, [x0, #8]        // new capacity
    mov x19, x0

.Lhas_capacity:
    ldr x21, [x19]           // len
    add x0, x19, #16
    str x20, [x0, x21, lsl #3]
    add x21, x21, #1
    str x21, [x19]

.Lpush_fail:
    ldp x21, x22, [sp, #32]
    ldp x19, x20, [sp, #16]
    ldp x29, x30, [sp], #48
    ret

// sigil_vec_get(v: *const Vec, idx: i64) -> i64
.global _sigil_vec_get
.align 4
_sigil_vec_get:
    ldr x2, [x0]
    cmp x1, x2
    b.ge .Lvec_bounds_error
    add x0, x0, #16
    ldr x0, [x0, x1, lsl #3]
    ret

.Lvec_bounds_error:
    mov x0, #0
    ret

// sigil_vec_len(v: *const Vec) -> i64
.global _sigil_vec_len
.align 4
_sigil_vec_len:
    ldr x0, [x0]
    ret

// ============================================================================
// File I/O
// ============================================================================

// sigil_file_open(path: *const u8, flags: i64) -> i64
.global _sigil_file_open
.align 4
_sigil_file_open:
    mov x2, #0644            // mode
    mov x16, SYS_open
    svc #0x80
    ret

// sigil_file_close(fd: i64) -> i64
.global _sigil_file_close
.align 4
_sigil_file_close:
    mov x16, SYS_close
    svc #0x80
    ret

// sigil_file_read(fd: i64, buf: *mut u8, len: i64) -> i64
.global _sigil_file_read
.align 4
_sigil_file_read:
    mov x16, SYS_read
    svc #0x80
    ret

// sigil_file_write(fd: i64, buf: *const u8, len: i64) -> i64
.global _sigil_file_write
.align 4
_sigil_file_write:
    mov x16, SYS_write
    svc #0x80
    ret

// sigil_file_exists(path: *const u8) -> i64
.global _sigil_file_exists
.align 4
_sigil_file_exists:
    mov x1, #0               // F_OK
    mov x16, SYS_access
    svc #0x80
    cmp x0, #0
    cset x0, eq
    ret

// sigil_file_seek(fd: i64, offset: i64, whence: i64) -> i64
.global _sigil_file_seek
.align 4
_sigil_file_seek:
    mov x16, SYS_lseek
    svc #0x80
    ret

// ============================================================================
// Math Functions
// ============================================================================

// sigil_sqrt(x: f64) -> f64
.global _sigil_sqrt
.align 4
_sigil_sqrt:
    fsqrt d0, d0
    ret

// sigil_abs(x: f64) -> f64
.global _sigil_abs
.align 4
_sigil_abs:
    fabs d0, d0
    ret

// sigil_min(a: i64, b: i64) -> i64
.global _sigil_min
.align 4
_sigil_min:
    cmp x0, x1
    csel x0, x0, x1, lt
    ret

// sigil_max(a: i64, b: i64) -> i64
.global _sigil_max
.align 4
_sigil_max:
    cmp x0, x1
    csel x0, x0, x1, gt
    ret

// ============================================================================
// SIMD Functions (NEON)
// ============================================================================

// simd_f32x4_add(dst: *mut f32, a: *const f32, b: *const f32)
.global _simd_f32x4_add
.align 4
_simd_f32x4_add:
    ld1 {v0.4s}, [x1]
    ld1 {v1.4s}, [x2]
    fadd v0.4s, v0.4s, v1.4s
    st1 {v0.4s}, [x0]
    ret

// simd_f32x4_sub(dst: *mut f32, a: *const f32, b: *const f32)
.global _simd_f32x4_sub
.align 4
_simd_f32x4_sub:
    ld1 {v0.4s}, [x1]
    ld1 {v1.4s}, [x2]
    fsub v0.4s, v0.4s, v1.4s
    st1 {v0.4s}, [x0]
    ret

// simd_f32x4_mul(dst: *mut f32, a: *const f32, b: *const f32)
.global _simd_f32x4_mul
.align 4
_simd_f32x4_mul:
    ld1 {v0.4s}, [x1]
    ld1 {v1.4s}, [x2]
    fmul v0.4s, v0.4s, v1.4s
    st1 {v0.4s}, [x0]
    ret

// simd_f32x4_div(dst: *mut f32, a: *const f32, b: *const f32)
.global _simd_f32x4_div
.align 4
_simd_f32x4_div:
    ld1 {v0.4s}, [x1]
    ld1 {v1.4s}, [x2]
    fdiv v0.4s, v0.4s, v1.4s
    st1 {v0.4s}, [x0]
    ret

// simd_f32x4_min(dst: *mut f32, a: *const f32, b: *const f32)
.global _simd_f32x4_min
.align 4
_simd_f32x4_min:
    ld1 {v0.4s}, [x1]
    ld1 {v1.4s}, [x2]
    fmin v0.4s, v0.4s, v1.4s
    st1 {v0.4s}, [x0]
    ret

// simd_f32x4_max(dst: *mut f32, a: *const f32, b: *const f32)
.global _simd_f32x4_max
.align 4
_simd_f32x4_max:
    ld1 {v0.4s}, [x1]
    ld1 {v1.4s}, [x2]
    fmax v0.4s, v0.4s, v1.4s
    st1 {v0.4s}, [x0]
    ret

// simd_f32x4_sqrt(dst: *mut f32, a: *const f32)
.global _simd_f32x4_sqrt
.align 4
_simd_f32x4_sqrt:
    ld1 {v0.4s}, [x1]
    fsqrt v0.4s, v0.4s
    st1 {v0.4s}, [x0]
    ret

// simd_f32x4_dot(a: *const f32, b: *const f32) -> f32
.global _simd_f32x4_dot
.align 4
_simd_f32x4_dot:
    ld1 {v0.4s}, [x0]
    ld1 {v1.4s}, [x1]
    fmul v0.4s, v0.4s, v1.4s
    faddp v0.4s, v0.4s, v0.4s
    faddp s0, v0.2s
    ret

// ============================================================================
// Syscall Wrappers
// ============================================================================

// Sys_write(fd: i64, buf: *const u8, len: i64) -> i64
.global _Sys_write
.align 4
_Sys_write:
    mov x16, SYS_write
    svc #0x80
    ret

// Sys_read(fd: i64, buf: *mut u8, len: i64) -> i64
.global _Sys_read
.align 4
_Sys_read:
    mov x16, SYS_read
    svc #0x80
    ret

// Sys_open(path: *const u8, flags: i64, mode: i64) -> i64
.global _Sys_open
.align 4
_Sys_open:
    mov x16, SYS_open
    svc #0x80
    ret

// Sys_close(fd: i64) -> i64
.global _Sys_close
.align 4
_Sys_close:
    mov x16, SYS_close
    svc #0x80
    ret

// Sys_mmap(addr: *mut u8, len: i64, prot: i64, flags: i64, fd: i64, off: i64) -> *mut u8
.global _Sys_mmap
.align 4
_Sys_mmap:
    mov x16, SYS_mmap
    svc #0x80
    ret

// Sys_munmap(addr: *mut u8, len: i64) -> i64
.global _Sys_munmap
.align 4
_Sys_munmap:
    mov x16, SYS_munmap
    svc #0x80
    ret

// Sys_exit(code: i64)
.global _Sys_exit
.align 4
_Sys_exit:
    mov x16, SYS_exit
    svc #0x80
    // No return
