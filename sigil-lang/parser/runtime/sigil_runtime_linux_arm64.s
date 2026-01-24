// Sigil Native Runtime - Linux ARM64 (AArch64)
//
// Pure syscall implementation with no libc dependency.
// Provides the same interface as the x86_64 runtime.
//
// Syscall ABI (Linux ARM64):
//   - syscall number in X8
//   - arguments in X0, X1, X2, X3, X4, X5
//   - return value in X0 (negative = -errno)
//
// Syscall numbers (ARM64 Linux):
//   63 = read     64 = write   56 = openat   57 = close
//  222 = mmap    215 = munmap  93 = exit    113 = clock_gettime
//   62 = lseek    48 = faccessat
//
// Calling convention (AAPCS64):
//   - Arguments in x0-x7
//   - Return value in x0 (and x1 for 128-bit)
//   - Callee-saved: x19-x28, sp
//   - Frame pointer: x29 (fp)
//   - Link register: x30 (lr)

.global _start
.global main_sigil

// ============================================================================
// Constants
// ============================================================================

.equ SYS_read, 63
.equ SYS_write, 64
.equ SYS_openat, 56
.equ SYS_close, 57
.equ SYS_lseek, 62
.equ SYS_mmap, 222
.equ SYS_munmap, 215
.equ SYS_exit, 93
.equ SYS_clock_gettime, 113
.equ SYS_faccessat, 48

.equ AT_FDCWD, -100
.equ F_OK, 0
.equ CLOCK_REALTIME, 0

.equ ARENA_SIZE, 1048576
.equ LARGE_ALLOC_THRESHOLD, 4096
.equ ARENA_HEADER_SIZE, 24

.equ PROT_READ_WRITE, 3
.equ MAP_PRIVATE_ANONYMOUS, 34

.equ O_RDONLY, 0
.equ O_WRONLY, 1
.equ O_CREAT, 64
.equ O_TRUNC, 512
.equ O_WRONLY_CREAT_TRUNC, 577

.equ SEEK_SET, 0
.equ SEEK_END, 2

// ============================================================================
// Entry Point
// ============================================================================

.section .text

_start:
    // Zero frame pointer for clean stack traces
    mov x29, #0
    mov x30, #0

    // Call the Sigil main function
    bl main_sigil

    // Exit with return value from main
    mov x8, #SYS_exit
    svc #0

// ============================================================================
// Print Functions
// ============================================================================

// sigil_println(str: *const String)
// Prints a Sigil String (with [len][capacity][data] header) followed by newline
// String layout: [len: i64][capacity: i64][data: u8[]]
.global sigil_println
sigil_println:
    stp x29, x30, [sp, #-32]!
    mov x29, sp
    stp x19, x20, [sp, #16]

    // Handle NULL
    cbz x0, .println_newline_only

    // Get length from header (first 8 bytes)
    ldr x19, [x0]            // len = str[0]

    // Get data pointer (skip 16-byte header)
    add x20, x0, #16         // data = str + 16

    // Write string data
    mov x8, #SYS_write
    mov x0, #1               // stdout
    mov x1, x20              // buffer = data ptr
    mov x2, x19              // length from header
    svc #0

.println_newline_only:
    // Write newline
    adr x1, .newline_char
    mov x8, #SYS_write
    mov x0, #1               // stdout
    mov x2, #1               // 1 byte
    svc #0

    ldp x19, x20, [sp, #16]
    ldp x29, x30, [sp], #32
    ret

.newline_char:
    .byte 10
    .align 2

// sigil_print_int(value: i64)
// Prints an integer followed by newline
.global sigil_print_int
sigil_print_int:
    stp x29, x30, [sp, #-64]!
    mov x29, sp
    stp x19, x20, [sp, #16]
    stp x21, x22, [sp, #32]

    mov x19, x0              // Save value
    add x20, sp, #63         // End of buffer (within frame)
    strb wzr, [x20]          // Null terminator
    mov x21, #0              // Negative flag

    // Handle negative numbers
    cmp x19, #0
    b.ge .print_positive_arm
    neg x19, x19
    mov x21, #1              // Negative flag
    b .print_convert_arm

.print_positive_arm:
    // Already positive

.print_convert_arm:
    // Convert to decimal digits (reverse order)
    mov x22, #10             // Divisor

.digit_loop_arm:
    sub x20, x20, #1
    udiv x1, x19, x22        // x1 = quotient
    msub x2, x1, x22, x19    // x2 = remainder (x19 - x1 * 10)
    add w2, w2, #'0'
    strb w2, [x20]
    mov x19, x1              // quotient for next iteration
    cbnz x19, .digit_loop_arm

    // Add minus sign if negative
    cbz x21, .print_number_arm
    sub x20, x20, #1
    mov w2, #'-'
    strb w2, [x20]

.print_number_arm:
    // Calculate length
    add x1, sp, #63
    sub x2, x1, x20          // Length = end - start

    // Write number
    mov x8, #SYS_write
    mov x0, #1               // stdout
    mov x1, x20              // buffer start
    svc #0

    // Write newline
    adr x1, .newline_char
    mov x8, #SYS_write
    mov x0, #1
    mov x2, #1
    svc #0

    ldp x21, x22, [sp, #32]
    ldp x19, x20, [sp, #16]
    ldp x29, x30, [sp], #64
    ret

// sigil_print_float(value: f64)
// Prints a floating point number (placeholder)
.global sigil_print_float
sigil_print_float:
    stp x29, x30, [sp, #-16]!
    mov x29, sp

    adr x1, .float_placeholder
    mov x8, #SYS_write
    mov x0, #1
    mov x2, #8
    svc #0

    ldp x29, x30, [sp], #16
    ret

.float_placeholder:
    .asciz "<float>\n"
    .align 2

// ============================================================================
// Memory Functions - Arena/Bump Allocator
// ============================================================================

.section .bss
    .align 3
arena_current:    .quad 0
arena_bump:       .quad 0
arena_end:        .quad 0

.section .text

// sigil_arena_init() -> i64
.global sigil_arena_init
sigil_arena_init:
    stp x29, x30, [sp, #-32]!
    mov x29, sp
    str x19, [sp, #16]

    // Check if already initialized
    adrp x0, arena_current
    add x0, x0, :lo12:arena_current
    ldr x1, [x0]
    cbnz x1, .arena_already_init_arm

    // Allocate first arena via mmap
    mov x8, #SYS_mmap
    mov x0, #0               // addr = NULL
    mov x1, #ARENA_SIZE
    mov x2, #PROT_READ_WRITE
    mov x3, #MAP_PRIVATE_ANONYMOUS
    mov x4, #-1              // fd = -1
    mov x5, #0               // offset = 0
    svc #0

    // Check for error (negative value with high bits set)
    cmn x0, #4096
    b.hi .arena_init_failed_arm

    mov x19, x0              // Save arena pointer

    // Initialize arena header
    adrp x1, arena_current
    add x1, x1, :lo12:arena_current
    str x0, [x1]             // arena_current = arena

    str xzr, [x0]            // next = NULL

    add x2, x0, #ARENA_SIZE
    str x2, [x0, #8]         // end

    add x3, x0, #ARENA_HEADER_SIZE
    str x3, [x0, #16]        // bump (after header)

    adrp x1, arena_bump
    add x1, x1, :lo12:arena_bump
    str x3, [x1]

    adrp x1, arena_end
    add x1, x1, :lo12:arena_end
    str x2, [x1]

    mov x0, #1               // Success
    ldr x19, [sp, #16]
    ldp x29, x30, [sp], #32
    ret

.arena_already_init_arm:
    mov x0, #1
    ldr x19, [sp, #16]
    ldp x29, x30, [sp], #32
    ret

.arena_init_failed_arm:
    mov x0, #0               // Failure
    ldr x19, [sp, #16]
    ldp x29, x30, [sp], #32
    ret

// sigil_alloc(size: i64) -> *mut u8
.global sigil_alloc
sigil_alloc:
    stp x29, x30, [sp, #-48]!
    mov x29, sp
    stp x19, x20, [sp, #16]
    str x21, [sp, #32]

    mov x19, x0              // Save requested size

    // Ensure arena is initialized
    adrp x0, arena_current
    add x0, x0, :lo12:arena_current
    ldr x0, [x0]
    cbnz x0, .arena_ready_arm

    bl sigil_arena_init
    cbz x0, .alloc_failed_arm

.arena_ready_arm:
    // Align size to 16 bytes
    add x19, x19, #15
    and x19, x19, #-16

    // Check if large allocation (bypass arena)
    cmp x19, #LARGE_ALLOC_THRESHOLD
    b.ge .large_alloc_arm

    // Try bump allocation
    adrp x0, arena_bump
    add x0, x0, :lo12:arena_bump
    ldr x20, [x0]            // current bump

    add x21, x20, x19        // new_bump = bump + size

    // Check if fits in current arena
    adrp x1, arena_end
    add x1, x1, :lo12:arena_end
    ldr x1, [x1]
    cmp x21, x1
    b.hi .need_new_arena_arm

    // Bump allocation succeeded
    str x21, [x0]            // Update arena_bump
    mov x0, x20              // Return old bump pointer

    ldr x21, [sp, #32]
    ldp x19, x20, [sp, #16]
    ldp x29, x30, [sp], #48
    ret

.need_new_arena_arm:
    // Allocate new arena
    mov x8, #SYS_mmap
    mov x0, #0
    mov x1, #ARENA_SIZE
    mov x2, #PROT_READ_WRITE
    mov x3, #MAP_PRIVATE_ANONYMOUS
    mov x4, #-1
    mov x5, #0
    svc #0

    cmn x0, #4096
    b.hi .alloc_failed_arm

    mov x20, x0              // New arena

    // Link new arena to current
    adrp x1, arena_current
    add x1, x1, :lo12:arena_current
    ldr x2, [x1]
    str x2, [x0]             // new->next = current
    str x0, [x1]             // arena_current = new

    // Set end
    add x2, x0, #ARENA_SIZE
    str x2, [x0, #8]
    adrp x1, arena_end
    add x1, x1, :lo12:arena_end
    str x2, [x1]

    // Set bump and allocate
    add x1, x0, #ARENA_HEADER_SIZE
    add x2, x1, x19
    str x2, [x0, #16]

    adrp x3, arena_bump
    add x3, x3, :lo12:arena_bump
    str x2, [x3]

    mov x0, x1               // Return start of allocation

    ldr x21, [sp, #32]
    ldp x19, x20, [sp, #16]
    ldp x29, x30, [sp], #48
    ret

.large_alloc_arm:
    // Direct mmap for large allocations
    add x19, x19, #8         // Add header for size

    mov x8, #SYS_mmap
    mov x0, #0
    mov x1, x19
    mov x2, #PROT_READ_WRITE
    mov x3, #MAP_PRIVATE_ANONYMOUS
    mov x4, #-1
    mov x5, #0
    svc #0

    cmn x0, #4096
    b.hi .alloc_failed_arm

    // Store size in header
    sub x19, x19, #8
    str x19, [x0]
    add x0, x0, #8           // Return pointer after header

    ldr x21, [sp, #32]
    ldp x19, x20, [sp, #16]
    ldp x29, x30, [sp], #48
    ret

.alloc_failed_arm:
    mov x0, #0
    ldr x21, [sp, #32]
    ldp x19, x20, [sp, #16]
    ldp x29, x30, [sp], #48
    ret

// sigil_free(ptr: *mut u8)
.global sigil_free
sigil_free:
    // No-op for arena allocations
    ret

// sigil_realloc(ptr: *mut u8, new_size: i64) -> *mut u8
.global sigil_realloc
sigil_realloc:
    stp x29, x30, [sp, #-48]!
    mov x29, sp
    stp x19, x20, [sp, #16]
    str x21, [sp, #32]

    mov x19, x0              // old ptr
    mov x20, x1              // new size

    // Handle NULL ptr
    cbz x19, .realloc_just_alloc_arm

    // Allocate new block
    mov x0, x20
    bl sigil_alloc
    cbz x0, .realloc_failed_arm

    mov x21, x0              // new ptr

    // Copy data
    mov x0, x21              // dest
    mov x1, x19              // src
    mov x2, x20              // count
    bl .memcpy_arm

    mov x0, x21

    ldr x21, [sp, #32]
    ldp x19, x20, [sp, #16]
    ldp x29, x30, [sp], #48
    ret

.realloc_just_alloc_arm:
    mov x0, x20
    bl sigil_alloc
    ldr x21, [sp, #32]
    ldp x19, x20, [sp, #16]
    ldp x29, x30, [sp], #48
    ret

.realloc_failed_arm:
    mov x0, #0
    ldr x21, [sp, #32]
    ldp x19, x20, [sp, #16]
    ldp x29, x30, [sp], #48
    ret

// Simple memcpy implementation
.memcpy_arm:
    cbz x2, .memcpy_done_arm
.memcpy_loop_arm:
    ldrb w3, [x1], #1
    strb w3, [x0], #1
    subs x2, x2, #1
    b.ne .memcpy_loop_arm
.memcpy_done_arm:
    ret

// sigil_arena_reset()
.global sigil_arena_reset
sigil_arena_reset:
    adrp x0, arena_current
    add x0, x0, :lo12:arena_current
    ldr x0, [x0]
    cbz x0, .reset_done_arm

    add x1, x0, #ARENA_HEADER_SIZE
    adrp x2, arena_bump
    add x2, x2, :lo12:arena_bump
    str x1, [x2]
    str x1, [x0, #16]

.reset_done_arm:
    ret

// sigil_arena_stats() -> (total_arenas: i64, total_bytes: i64)
.global sigil_arena_stats
sigil_arena_stats:
    mov x0, #0               // arena count
    mov x1, #0               // total bytes

    adrp x2, arena_current
    add x2, x2, :lo12:arena_current
    ldr x2, [x2]

.stats_loop_arm:
    cbz x2, .stats_done_arm
    add x0, x0, #1
    add x1, x1, #ARENA_SIZE
    ldr x2, [x2]             // next arena
    b .stats_loop_arm

.stats_done_arm:
    ret

// ============================================================================
// Time Functions
// ============================================================================

// sigil_now() -> i64
.global sigil_now
sigil_now:
    stp x29, x30, [sp, #-32]!
    mov x29, sp

    mov x8, #SYS_clock_gettime
    mov x0, #CLOCK_REALTIME
    add x1, sp, #16          // timespec on stack
    svc #0

    // Convert to milliseconds
    ldr x0, [sp, #16]        // tv_sec
    mov x2, #1000
    mul x0, x0, x2           // * 1000

    ldr x1, [sp, #24]        // tv_nsec
    mov x2, #1000000
    udiv x1, x1, x2          // / 1000000
    add x0, x0, x1

    ldp x29, x30, [sp], #32
    ret

// ============================================================================
// String Functions
// ============================================================================

// sigil_strlen(str: *const u8) -> i64
.global sigil_strlen
sigil_strlen:
    mov x1, #0
    cbz x0, .strlen_ret_arm
.strlen_loop_arm:
    ldrb w2, [x0, x1]
    cbz w2, .strlen_ret_arm
    add x1, x1, #1
    b .strlen_loop_arm
.strlen_ret_arm:
    mov x0, x1
    ret

// sigil_string_from(cstr: *const u8) -> *mut String
.global sigil_string_from
sigil_string_from:
    stp x29, x30, [sp, #-48]!
    mov x29, sp
    stp x19, x20, [sp, #16]
    str x21, [sp, #32]

    mov x19, x0              // Save source

    bl sigil_strlen
    mov x20, x0              // Save length

    // Allocate: 16 bytes header + len + 1
    add x0, x20, #17
    bl sigil_alloc
    cbz x0, .string_from_failed_arm

    mov x21, x0              // Save string pointer

    // Set header
    str x20, [x21]           // len
    add x1, x20, #1
    str x1, [x21, #8]        // capacity

    // Copy data
    add x0, x21, #16         // dest
    mov x1, x19              // src
    mov x2, x20              // count
    bl .memcpy_arm

    // Null terminate
    add x1, x21, #16
    add x1, x1, x20
    strb wzr, [x1]

    mov x0, x21

    ldr x21, [sp, #32]
    ldp x19, x20, [sp, #16]
    ldp x29, x30, [sp], #48
    ret

.string_from_failed_arm:
    mov x0, #0
    ldr x21, [sp, #32]
    ldp x19, x20, [sp, #16]
    ldp x29, x30, [sp], #48
    ret

// sigil_string_len(str: *mut String) -> i64
.global sigil_string_len
sigil_string_len:
    cbz x0, .string_len_zero_arm
    ldr x0, [x0]
    ret
.string_len_zero_arm:
    mov x0, #0
    ret

// sigil_string_as_ptr(str: *mut String) -> *const u8
.global sigil_string_as_ptr
sigil_string_as_ptr:
    cbz x0, .string_ptr_null_arm
    add x0, x0, #16
    ret
.string_ptr_null_arm:
    mov x0, #0
    ret

// sigil_string_print(str: *mut String)
.global sigil_string_print
sigil_string_print:
    stp x29, x30, [sp, #-32]!
    mov x29, sp
    str x19, [sp, #16]

    cbz x0, .string_print_done_arm

    mov x19, x0

    mov x8, #SYS_write
    mov x0, #1               // stdout
    add x1, x19, #16         // data
    ldr x2, [x19]            // len
    svc #0

.string_print_done_arm:
    ldr x19, [sp, #16]
    ldp x29, x30, [sp], #32
    ret

// sigil_string_concat(a: *mut String, b: *mut String) -> *mut String
.global sigil_string_concat
sigil_string_concat:
    stp x29, x30, [sp, #-64]!
    mov x29, sp
    stp x19, x20, [sp, #16]
    stp x21, x22, [sp, #32]

    mov x19, x0              // String a
    mov x20, x1              // String b
    mov x21, #0              // Total length

    cbz x19, .concat_no_a_arm
    ldr x1, [x19]
    add x21, x21, x1
.concat_no_a_arm:
    cbz x20, .concat_no_b_arm
    ldr x1, [x20]
    add x21, x21, x1
.concat_no_b_arm:

    // Allocate new string
    add x0, x21, #17
    bl sigil_alloc
    cbz x0, .concat_failed_arm

    mov x22, x0              // New string

    // Set header
    str x21, [x22]
    add x1, x21, #1
    str x1, [x22, #8]

    // Copy first string
    add x0, x22, #16
    cbz x19, .concat_copy_b_arm
    add x1, x19, #16
    ldr x2, [x19]
    bl .memcpy_arm

    // Update destination pointer
    ldr x3, [x19]
    add x0, x22, #16
    add x0, x0, x3

.concat_copy_b_arm:
    cbz x20, .concat_done_arm
    add x1, x20, #16
    ldr x2, [x20]
    bl .memcpy_arm

.concat_done_arm:
    // Null terminate
    add x1, x22, #16
    add x1, x1, x21
    strb wzr, [x1]

    mov x0, x22

    ldp x21, x22, [sp, #32]
    ldp x19, x20, [sp, #16]
    ldp x29, x30, [sp], #64
    ret

.concat_failed_arm:
    mov x0, #0
    ldp x21, x22, [sp, #32]
    ldp x19, x20, [sp, #16]
    ldp x29, x30, [sp], #64
    ret

// sigil_string_eq(a: *mut String, b: *mut String) -> i64
.global sigil_string_eq
sigil_string_eq:
    // Handle NULL cases
    cbz x0, .eq_check_b_null_arm
    cbz x1, .eq_not_equal_arm

    // Compare lengths
    ldr x2, [x0]
    ldr x3, [x1]
    cmp x2, x3
    b.ne .eq_not_equal_arm

    cbz x2, .eq_equal_arm    // Both empty

    // Compare data
    add x0, x0, #16
    add x1, x1, #16
.eq_compare_loop_arm:
    ldrb w4, [x0], #1
    ldrb w5, [x1], #1
    cmp w4, w5
    b.ne .eq_not_equal_arm
    subs x2, x2, #1
    b.ne .eq_compare_loop_arm

.eq_equal_arm:
    mov x0, #1
    ret

.eq_check_b_null_arm:
    cbz x1, .eq_equal_arm
.eq_not_equal_arm:
    mov x0, #0
    ret

// sigil_string_clone(str: *mut String) -> *mut String
.global sigil_string_clone
sigil_string_clone:
    stp x29, x30, [sp, #-48]!
    mov x29, sp
    stp x19, x20, [sp, #16]
    str x21, [sp, #32]

    cbz x0, .clone_null_arm

    mov x19, x0
    ldr x20, [x0]            // len

    add x0, x20, #17
    bl sigil_alloc
    cbz x0, .clone_failed_arm

    mov x21, x0

    // Set header
    str x20, [x21]
    add x1, x20, #1
    str x1, [x21, #8]

    // Copy data
    add x0, x21, #16
    add x1, x19, #16
    mov x2, x20
    bl .memcpy_arm

    // Null terminate
    add x1, x21, #16
    add x1, x1, x20
    strb wzr, [x1]

    mov x0, x21

    ldr x21, [sp, #32]
    ldp x19, x20, [sp, #16]
    ldp x29, x30, [sp], #48
    ret

.clone_null_arm:
.clone_failed_arm:
    mov x0, #0
    ldr x21, [sp, #32]
    ldp x19, x20, [sp, #16]
    ldp x29, x30, [sp], #48
    ret

// sigil_string_is_empty(str: *mut String) -> i64
.global sigil_string_is_empty
sigil_string_is_empty:
    cbz x0, .is_empty_true_arm
    ldr x1, [x0]
    cbz x1, .is_empty_true_arm
    mov x0, #0
    ret
.is_empty_true_arm:
    mov x0, #1
    ret

// sigil_string_char_at(str: *mut String, idx: i64) -> i64
.global sigil_string_char_at
sigil_string_char_at:
    cbz x0, .char_at_invalid_arm
    cmp x1, #0
    b.lt .char_at_invalid_arm
    ldr x2, [x0]
    cmp x1, x2
    b.ge .char_at_invalid_arm

    add x0, x0, #16
    ldrb w0, [x0, x1]
    ret

.char_at_invalid_arm:
    mov x0, #-1
    ret

// ============================================================================
// Vec Operations
// ============================================================================

// sigil_vec_new(capacity: i64) -> *mut Vec
.global sigil_vec_new
sigil_vec_new:
    stp x29, x30, [sp, #-32]!
    mov x29, sp
    str x19, [sp, #16]

    cmp x0, #4
    b.ge .vec_cap_ok_arm
    mov x0, #4
.vec_cap_ok_arm:
    mov x19, x0

    lsl x0, x0, #3           // capacity * 8
    add x0, x0, #16          // + header
    bl sigil_alloc

    cbz x0, .vec_alloc_failed_arm

    str xzr, [x0]            // len = 0
    str x19, [x0, #8]        // capacity

    ldr x19, [sp, #16]
    ldp x29, x30, [sp], #32
    ret

.vec_alloc_failed_arm:
    mov x0, #0
    ldr x19, [sp, #16]
    ldp x29, x30, [sp], #32
    ret

// sigil_vec_push(vec: *mut Vec, value: i64)
.global sigil_vec_push
sigil_vec_push:
    cbz x0, .vec_push_ret_arm

    ldr x2, [x0]             // len
    ldr x3, [x0, #8]         // capacity
    cmp x2, x3
    b.ge .vec_push_ret_arm

    add x4, x0, #16
    str x1, [x4, x2, lsl #3] // data[len] = value
    add x2, x2, #1
    str x2, [x0]             // len++

.vec_push_ret_arm:
    ret

// sigil_vec_get(vec: *mut Vec, index: i64) -> i64
.global sigil_vec_get
sigil_vec_get:
    cbz x0, .vec_get_zero_arm
    ldr x2, [x0]             // len
    cmp x1, x2
    b.ge .vec_get_zero_arm
    cmp x1, #0
    b.lt .vec_get_zero_arm

    add x0, x0, #16
    ldr x0, [x0, x1, lsl #3]
    ret

.vec_get_zero_arm:
    mov x0, #0
    ret

// sigil_vec_len(vec: *mut Vec) -> i64
.global sigil_vec_len
sigil_vec_len:
    cbz x0, .vec_len_zero_arm
    ldr x0, [x0]
    ret
.vec_len_zero_arm:
    mov x0, #0
    ret

// ============================================================================
// File I/O Functions
// ============================================================================

// sigil_file_open(path: *const u8, flags: i64, mode: i64) -> i64
.global sigil_file_open
sigil_file_open:
    mov x8, #SYS_openat
    mov x3, x2               // mode -> x3
    mov x2, x1               // flags -> x2
    mov x1, x0               // path -> x1
    mov x0, #AT_FDCWD        // dirfd = AT_FDCWD
    svc #0
    ret

// sigil_file_close(fd: i64) -> i64
.global sigil_file_close
sigil_file_close:
    mov x8, #SYS_close
    svc #0
    ret

// sigil_file_read(fd: i64, buf: *mut u8, count: i64) -> i64
.global sigil_file_read
sigil_file_read:
    mov x8, #SYS_read
    svc #0
    ret

// sigil_file_write(fd: i64, buf: *const u8, count: i64) -> i64
.global sigil_file_write
sigil_file_write:
    mov x8, #SYS_write
    svc #0
    ret

// sigil_file_exists(path: *const u8) -> i64
.global sigil_file_exists
sigil_file_exists:
    mov x8, #SYS_faccessat
    mov x3, #0               // flags
    mov x2, #F_OK            // mode
    mov x1, x0               // path
    mov x0, #AT_FDCWD        // dirfd
    svc #0
    cmp x0, #0
    cset x0, eq              // 1 if exists, 0 otherwise
    ret

// sigil_file_seek(fd: i64, offset: i64, whence: i64) -> i64
.global sigil_file_seek
sigil_file_seek:
    mov x8, #SYS_lseek
    svc #0
    ret

// sigil_file_read_all(path: *const u8) -> *mut String
.global sigil_file_read_all
sigil_file_read_all:
    stp x29, x30, [sp, #-64]!
    mov x29, sp
    stp x19, x20, [sp, #16]
    stp x21, x22, [sp, #32]

    mov x19, x0              // path

    // Open file
    mov x8, #SYS_openat
    mov x1, x19
    mov x0, #AT_FDCWD
    mov x2, #O_RDONLY
    mov x3, #0
    svc #0

    cmp x0, #0
    b.lt .read_all_failed_arm
    mov x20, x0              // fd

    // Seek to end
    mov x8, #SYS_lseek
    mov x0, x20
    mov x1, #0
    mov x2, #SEEK_END
    svc #0

    cmp x0, #0
    b.lt .read_all_close_fail_arm
    mov x21, x0              // file size

    // Seek to start
    mov x8, #SYS_lseek
    mov x0, x20
    mov x1, #0
    mov x2, #SEEK_SET
    svc #0

    // Allocate string
    add x0, x21, #17
    bl sigil_alloc
    cbz x0, .read_all_close_fail_arm
    mov x22, x0

    // Set header
    str x21, [x22]
    add x1, x21, #1
    str x1, [x22, #8]

    // Read file
    mov x8, #SYS_read
    mov x0, x20
    add x1, x22, #16
    mov x2, x21
    svc #0

    // Null terminate
    add x1, x22, #16
    add x1, x1, x21
    strb wzr, [x1]

    // Close file
    mov x8, #SYS_close
    mov x0, x20
    svc #0

    mov x0, x22

    ldp x21, x22, [sp, #32]
    ldp x19, x20, [sp, #16]
    ldp x29, x30, [sp], #64
    ret

.read_all_close_fail_arm:
    mov x8, #SYS_close
    mov x0, x20
    svc #0
.read_all_failed_arm:
    mov x0, #0
    ldp x21, x22, [sp, #32]
    ldp x19, x20, [sp, #16]
    ldp x29, x30, [sp], #64
    ret

// sigil_file_write_all(path: *const u8, content: *mut String) -> i64
.global sigil_file_write_all
sigil_file_write_all:
    stp x29, x30, [sp, #-48]!
    mov x29, sp
    stp x19, x20, [sp, #16]
    str x21, [sp, #32]

    mov x19, x0              // path
    mov x20, x1              // content

    cbz x20, .write_all_empty_arm

    // Open file
    mov x8, #SYS_openat
    mov x0, #AT_FDCWD
    mov x1, x19
    mov x2, #O_WRONLY_CREAT_TRUNC
    mov x3, #0644
    svc #0

    cmp x0, #0
    b.lt .write_all_failed_arm
    mov x21, x0              // fd

    // Write
    mov x8, #SYS_write
    mov x0, x21
    add x1, x20, #16
    ldr x2, [x20]
    svc #0
    mov x19, x0              // bytes written

    // Close
    mov x8, #SYS_close
    mov x0, x21
    svc #0

    mov x0, x19

    ldr x21, [sp, #32]
    ldp x19, x20, [sp, #16]
    ldp x29, x30, [sp], #48
    ret

.write_all_empty_arm:
    mov x0, #0
    ldr x21, [sp, #32]
    ldp x19, x20, [sp, #16]
    ldp x29, x30, [sp], #48
    ret

.write_all_failed_arm:
    ldr x21, [sp, #32]
    ldp x19, x20, [sp, #16]
    ldp x29, x30, [sp], #48
    ret

// sigil_file_size(path: *const u8) -> i64
.global sigil_file_size
sigil_file_size:
    stp x29, x30, [sp, #-32]!
    mov x29, sp
    stp x19, x20, [sp, #16]

    // Open file
    mov x8, #SYS_openat
    mov x1, x0
    mov x0, #AT_FDCWD
    mov x2, #O_RDONLY
    mov x3, #0
    svc #0

    cmp x0, #0
    b.lt .size_failed_arm
    mov x19, x0              // fd

    // Seek to end
    mov x8, #SYS_lseek
    mov x0, x19
    mov x1, #0
    mov x2, #SEEK_END
    svc #0
    mov x20, x0              // size

    // Close
    mov x8, #SYS_close
    mov x0, x19
    svc #0

    mov x0, x20

    ldp x19, x20, [sp, #16]
    ldp x29, x30, [sp], #32
    ret

.size_failed_arm:
    ldp x19, x20, [sp, #16]
    ldp x29, x30, [sp], #32
    ret

// ============================================================================
// Math Functions
// ============================================================================

// For ARM64, we use NEON/FP instructions

// sigil_sqrt(x: i64) -> i64
.global sigil_sqrt
sigil_sqrt:
    fmov d0, x0
    fsqrt d0, d0
    fmov x0, d0
    ret

// sigil_sin(x: i64) -> i64
// Note: ARM64 doesn't have native sin instruction, so we use a series approximation
// For production, link against libm or use a lookup table
.global sigil_sin
sigil_sin:
    // Placeholder - returns input for now
    // A real implementation would use Taylor series or CORDIC
    ret

// sigil_cos(x: i64) -> i64
.global sigil_cos
sigil_cos:
    // Placeholder
    ret

// sigil_abs(x: i64) -> i64
.global sigil_abs
sigil_abs:
    fmov d0, x0
    fabs d0, d0
    fmov x0, d0
    ret

// sigil_floor(x: i64) -> i64
.global sigil_floor
sigil_floor:
    fmov d0, x0
    frintm d0, d0            // Round toward -infinity
    fcvtzs x0, d0
    ret

// sigil_ceil(x: i64) -> i64
.global sigil_ceil
sigil_ceil:
    fmov d0, x0
    frintp d0, d0            // Round toward +infinity
    fcvtzs x0, d0
    ret

// sigil_pow(x: i64, y: i64) -> i64
// Placeholder - real implementation needs exp/log
.global sigil_pow
sigil_pow:
    // Placeholder
    mov x0, #0
    ret

// sigil_min(a: i64, b: i64) -> i64
.global sigil_min
sigil_min:
    cmp x0, x1
    csel x0, x0, x1, lt
    ret

// sigil_max(a: i64, b: i64) -> i64
.global sigil_max
sigil_max:
    cmp x0, x1
    csel x0, x0, x1, gt
    ret

// ============================================================================
// SIMD Functions (NEON)
// ============================================================================
// ARM64 NEON provides 128-bit vectors (v0-v31)
// F32x4 operations use 4S arrangement

// simd_f32x4_add(dst: *mut f32, a: *const f32, b: *const f32)
.global simd_f32x4_add
simd_f32x4_add:
    ldr q0, [x1]
    ldr q1, [x2]
    fadd v0.4s, v0.4s, v1.4s
    str q0, [x0]
    ret

// simd_f32x4_sub(dst: *mut f32, a: *const f32, b: *const f32)
.global simd_f32x4_sub
simd_f32x4_sub:
    ldr q0, [x1]
    ldr q1, [x2]
    fsub v0.4s, v0.4s, v1.4s
    str q0, [x0]
    ret

// simd_f32x4_mul(dst: *mut f32, a: *const f32, b: *const f32)
.global simd_f32x4_mul
simd_f32x4_mul:
    ldr q0, [x1]
    ldr q1, [x2]
    fmul v0.4s, v0.4s, v1.4s
    str q0, [x0]
    ret

// simd_f32x4_div(dst: *mut f32, a: *const f32, b: *const f32)
.global simd_f32x4_div
simd_f32x4_div:
    ldr q0, [x1]
    ldr q1, [x2]
    fdiv v0.4s, v0.4s, v1.4s
    str q0, [x0]
    ret

// simd_f32x4_sqrt(dst: *mut f32, a: *const f32)
.global simd_f32x4_sqrt
simd_f32x4_sqrt:
    ldr q0, [x1]
    fsqrt v0.4s, v0.4s
    str q0, [x0]
    ret

// simd_f32x4_min(dst: *mut f32, a: *const f32, b: *const f32)
.global simd_f32x4_min
simd_f32x4_min:
    ldr q0, [x1]
    ldr q1, [x2]
    fmin v0.4s, v0.4s, v1.4s
    str q0, [x0]
    ret

// simd_f32x4_max(dst: *mut f32, a: *const f32, b: *const f32)
.global simd_f32x4_max
simd_f32x4_max:
    ldr q0, [x1]
    ldr q1, [x2]
    fmax v0.4s, v0.4s, v1.4s
    str q0, [x0]
    ret

// simd_f64x2_add(dst: *mut f64, a: *const f64, b: *const f64)
.global simd_f64x2_add
simd_f64x2_add:
    ldr q0, [x1]
    ldr q1, [x2]
    fadd v0.2d, v0.2d, v1.2d
    str q0, [x0]
    ret

// simd_f64x2_sub(dst: *mut f64, a: *const f64, b: *const f64)
.global simd_f64x2_sub
simd_f64x2_sub:
    ldr q0, [x1]
    ldr q1, [x2]
    fsub v0.2d, v0.2d, v1.2d
    str q0, [x0]
    ret

// simd_f64x2_mul(dst: *mut f64, a: *const f64, b: *const f64)
.global simd_f64x2_mul
simd_f64x2_mul:
    ldr q0, [x1]
    ldr q1, [x2]
    fmul v0.2d, v0.2d, v1.2d
    str q0, [x0]
    ret

// simd_f64x2_div(dst: *mut f64, a: *const f64, b: *const f64)
.global simd_f64x2_div
simd_f64x2_div:
    ldr q0, [x1]
    ldr q1, [x2]
    fdiv v0.2d, v0.2d, v1.2d
    str q0, [x0]
    ret

// simd_f64x2_sqrt(dst: *mut f64, a: *const f64)
.global simd_f64x2_sqrt
simd_f64x2_sqrt:
    ldr q0, [x1]
    fsqrt v0.2d, v0.2d
    str q0, [x0]
    ret

// ============================================================================
// Low-level Syscall Wrappers
// ============================================================================

// Sys_write(fd: i64, buf: *const u8, len: i64) -> i64
.global Sys_write
Sys_write:
    mov x8, #SYS_write
    svc #0
    ret

// Sys_read(fd: i64, buf: *mut u8, len: i64) -> i64
.global Sys_read
Sys_read:
    mov x8, #SYS_read
    svc #0
    ret

// Sys_open(path: *const u8, flags: i64, mode: i64) -> i64
.global Sys_open
Sys_open:
    mov x8, #SYS_openat
    mov x3, x2
    mov x2, x1
    mov x1, x0
    mov x0, #AT_FDCWD
    svc #0
    ret

// Sys_close(fd: i64) -> i64
.global Sys_close
Sys_close:
    mov x8, #SYS_close
    svc #0
    ret

// Sys_mmap(addr: i64, len: i64, prot: i64, flags: i64, fd: i64, off: i64) -> i64
.global Sys_mmap
Sys_mmap:
    mov x8, #SYS_mmap
    svc #0
    ret

// Sys_munmap(addr: i64, len: i64) -> i64
.global Sys_munmap
Sys_munmap:
    mov x8, #SYS_munmap
    svc #0
    ret

// Sys_exit(code: i64) -> !
.global Sys_exit
Sys_exit:
    mov x8, #SYS_exit
    svc #0

// Sys_clock_gettime(clock_id: i64, ts: *mut timespec) -> i64
.global Sys_clock_gettime
Sys_clock_gettime:
    mov x8, #SYS_clock_gettime
    svc #0
    ret

// ============================================================================
// Stub functions for AVX (x86-only)
// ============================================================================
// These return 0/fail gracefully since AVX isn't available on ARM

.global simd_f32x8_add
.global simd_f32x8_sub
.global simd_f32x8_mul
.global simd_f32x8_div
.global simd_f32x8_min
.global simd_f32x8_max
.global simd_f32x8_sqrt
.global simd_f32x8_splat
.global simd_f32x8_reduce_add
.global simd_f32x8_dot
.global simd_f32x8_fmadd
.global simd_f64x4_add
.global simd_f64x4_sub
.global simd_f64x4_mul
.global simd_f64x4_div
.global simd_f64x4_sqrt
.global simd_f64x4_reduce_add
.global simd_f64x4_dot
.global simd_check_avx
.global simd_check_avx2
.global simd_check_fma
.global simd_f32x4_splat
.global simd_f32x4_reduce_add
.global simd_f32x4_dot
.global simd_f32x4_fmadd
.global simd_f64x2_reduce_add
.global simd_f64x2_dot
.global simd_alloc_aligned

simd_f32x8_add:
simd_f32x8_sub:
simd_f32x8_mul:
simd_f32x8_div:
simd_f32x8_min:
simd_f32x8_max:
simd_f32x8_sqrt:
simd_f32x8_splat:
simd_f32x8_reduce_add:
simd_f32x8_dot:
simd_f32x8_fmadd:
simd_f64x4_add:
simd_f64x4_sub:
simd_f64x4_mul:
simd_f64x4_div:
simd_f64x4_sqrt:
simd_f64x4_reduce_add:
simd_f64x4_dot:
simd_check_avx:
simd_check_avx2:
simd_check_fma:
simd_f32x4_splat:
simd_f32x4_reduce_add:
simd_f32x4_dot:
simd_f32x4_fmadd:
simd_f64x2_reduce_add:
simd_f64x2_dot:
    mov x0, #0
    ret

simd_alloc_aligned:
    // Fall back to regular allocation
    b sigil_alloc

// ============================================================================
// Note section for stack protection
// ============================================================================
.section .note.GNU-stack,"",@progbits
