# Sigil Native Runtime - macOS x86_64
# Pure assembly, no libc dependency
# Uses BSD syscall interface (0x2000000 + syscall_number)

.section __TEXT,__text,regular,pure_instructions

# ============================================================================
# macOS x86_64 Syscall Numbers (BSD)
# Note: macOS uses 0x2000000 | syscall_number for BSD syscalls
# ============================================================================
.set SYS_exit,       0x2000001
.set SYS_read,       0x2000003
.set SYS_write,      0x2000004
.set SYS_open,       0x2000005
.set SYS_close,      0x2000006
.set SYS_mmap,       0x20000C5   # 197
.set SYS_munmap,     0x2000049   # 73
.set SYS_gettimeofday, 0x2000074 # 116
.set SYS_access,     0x2000021   # 33
.set SYS_lseek,      0x20000C7   # 199
.set SYS_fstat,      0x2000153   # 339 (fstat64)

# File flags
.set O_RDONLY,   0x0000
.set O_WRONLY,   0x0001
.set O_RDWR,     0x0002
.set O_CREAT,    0x0200
.set O_TRUNC,    0x0400
.set O_APPEND,   0x0008

# mmap flags
.set PROT_READ,  0x01
.set PROT_WRITE, 0x02
.set MAP_PRIVATE, 0x0002
.set MAP_ANONYMOUS, 0x1000

# ============================================================================
# Entry Point
# ============================================================================
.global _start
_start:
    # Call main function
    call _main_sigil

    # Exit with return value
    mov rdi, rax
    mov rax, SYS_exit
    syscall

# ============================================================================
# Print Functions
# ============================================================================

# sigil_println(str: *const u8, len: i64)
.global _sigil_println
_sigil_println:
    push rbx
    push r12
    mov r12, rdi             # Save string pointer
    mov rbx, rsi             # Save length

    # Write the string
    mov rax, SYS_write
    mov rdi, 1               # stdout
    mov rsi, r12
    mov rdx, rbx
    syscall

    # Write newline
    lea rsi, [rip + .newline]
    mov rax, SYS_write
    mov rdi, 1
    mov rdx, 1
    syscall

    pop r12
    pop rbx
    ret

.newline:
    .byte 10

# sigil_print_int(n: i64)
.global _sigil_print_int
_sigil_print_int:
    push rbx
    push r12
    sub rsp, 32              # Buffer for digits

    mov rax, rdi
    mov r12, rsp             # Buffer end
    add r12, 31
    mov byte ptr [r12], 10   # Newline
    dec r12

    # Handle negative
    xor rbx, rbx             # Sign flag
    test rax, rax
    jns .positive
    neg rax
    mov rbx, 1

.positive:
    mov rcx, 10

.digit_loop:
    xor rdx, rdx
    div rcx
    add dl, '0'
    mov [r12], dl
    dec r12
    test rax, rax
    jnz .digit_loop

    # Add minus if negative
    test rbx, rbx
    jz .print_num
    mov byte ptr [r12], '-'
    dec r12

.print_num:
    inc r12
    lea rdx, [rsp + 32]
    sub rdx, r12             # Length

    mov rax, SYS_write
    mov rdi, 1
    mov rsi, r12
    syscall

    add rsp, 32
    pop r12
    pop rbx
    ret

# sigil_print_float(x: f64)
.global _sigil_print_float
_sigil_print_float:
    # Simple float print - integer part only for now
    push rbx
    sub rsp, 16

    movsd [rsp], xmm0
    cvttsd2si rdi, xmm0
    call _sigil_print_int

    add rsp, 16
    pop rbx
    ret

# ============================================================================
# Memory Management - Arena Allocator
# ============================================================================

.section __DATA,__bss
    .align 8
_arena_current:   .quad 0
_arena_bump:      .quad 0
_arena_end:       .quad 0

.section __TEXT,__text

.set ARENA_SIZE, 0x100000    # 1MB arenas

# sigil_alloc(size: i64) -> *mut u8
.global _sigil_alloc
_sigil_alloc:
    push rbx
    push r12
    mov rbx, rdi             # Size to allocate

    # Check if we have space in current arena
    mov rax, [rip + _arena_bump]
    test rax, rax
    jz .need_arena

    add rax, rbx
    cmp rax, [rip + _arena_end]
    jg .need_arena

    # Bump allocate
    mov rax, [rip + _arena_bump]
    add [rip + _arena_bump], rbx
    pop r12
    pop rbx
    ret

.need_arena:
    # Allocate new arena via mmap
    mov rax, SYS_mmap
    xor rdi, rdi             # addr = NULL
    mov rsi, ARENA_SIZE      # len
    mov rdx, PROT_READ
    or rdx, PROT_WRITE       # prot = RW
    mov r10, MAP_PRIVATE
    or r10, MAP_ANONYMOUS    # flags
    mov r8, -1               # fd = -1
    xor r9, r9               # offset = 0
    syscall

    cmp rax, -1
    je .alloc_fail

    # Set up arena
    mov [rip + _arena_current], rax
    lea rcx, [rax + rbx]
    mov [rip + _arena_bump], rcx
    lea rcx, [rax + ARENA_SIZE]
    mov [rip + _arena_end], rcx

    pop r12
    pop rbx
    ret

.alloc_fail:
    xor rax, rax
    pop r12
    pop rbx
    ret

# sigil_free(ptr: *mut u8)
.global _sigil_free
_sigil_free:
    # No-op for arena allocator
    ret

# ============================================================================
# String Functions
# ============================================================================

# sigil_string_from(data: *const u8, len: i64) -> *mut String
.global _sigil_string_from
_sigil_string_from:
    push rbx
    push r12
    push r13
    mov r12, rdi             # data
    mov r13, rsi             # len

    # Allocate: 16 bytes header + len + 1
    lea rdi, [rsi + 17]
    call _sigil_alloc

    test rax, rax
    jz .string_from_fail

    # Store length and capacity
    mov [rax], r13
    mov [rax + 8], r13

    # Copy data
    lea rdi, [rax + 16]
    mov rsi, r12
    mov rcx, r13
    rep movsb

    # Null terminate
    mov byte ptr [rdi], 0

    pop r13
    pop r12
    pop rbx
    ret

.string_from_fail:
    xor rax, rax
    pop r13
    pop r12
    pop rbx
    ret

# sigil_string_len(s: *const String) -> i64
.global _sigil_string_len
_sigil_string_len:
    mov rax, [rdi]
    ret

# sigil_string_as_ptr(s: *const String) -> *const u8
.global _sigil_string_as_ptr
_sigil_string_as_ptr:
    lea rax, [rdi + 16]
    ret

# sigil_string_print(s: *const String)
.global _sigil_string_print
_sigil_string_print:
    mov rax, SYS_write
    mov rsi, rdi
    add rsi, 16              # data ptr
    mov rdx, [rdi]           # length
    mov rdi, 1               # stdout
    syscall
    ret

# sigil_string_concat(a: *const String, b: *const String) -> *mut String
.global _sigil_string_concat
_sigil_string_concat:
    push rbx
    push r12
    push r13
    push r14
    mov r12, rdi             # a
    mov r13, rsi             # b

    mov r14, [r12]           # len(a)
    add r14, [r13]           # + len(b)

    # Allocate new string
    lea rdi, [r14 + 17]
    call _sigil_alloc

    test rax, rax
    jz .concat_fail

    mov rbx, rax             # Save result

    # Store header
    mov [rax], r14
    mov [rax + 8], r14

    # Copy first string
    lea rdi, [rax + 16]
    lea rsi, [r12 + 16]
    mov rcx, [r12]
    rep movsb

    # Copy second string
    lea rsi, [r13 + 16]
    mov rcx, [r13]
    rep movsb

    mov byte ptr [rdi], 0    # Null terminate

    mov rax, rbx
    pop r14
    pop r13
    pop r12
    pop rbx
    ret

.concat_fail:
    xor rax, rax
    pop r14
    pop r13
    pop r12
    pop rbx
    ret

# sigil_string_eq(a: *const String, b: *const String) -> i64
.global _sigil_string_eq
_sigil_string_eq:
    mov rax, [rdi]
    cmp rax, [rsi]
    jne .strings_not_equal

    # Compare bytes
    lea rdi, [rdi + 16]
    lea rsi, [rsi + 16]
    mov rcx, rax
    repe cmpsb
    jne .strings_not_equal

    mov rax, 1
    ret

.strings_not_equal:
    xor rax, rax
    ret

# ============================================================================
# Vec Functions
# ============================================================================

# sigil_vec_new() -> *mut Vec
.global _sigil_vec_new
_sigil_vec_new:
    mov rdi, 16              # Header only
    call _sigil_alloc
    test rax, rax
    jz .vec_new_fail

    mov qword ptr [rax], 0       # len = 0
    mov qword ptr [rax + 8], 0   # capacity = 0
    ret

.vec_new_fail:
    xor rax, rax
    ret

# sigil_vec_push(v: *mut Vec, val: i64)
.global _sigil_vec_push
_sigil_vec_push:
    push rbx
    push r12
    push r13
    mov rbx, rdi             # vec
    mov r12, rsi             # value

    mov rax, [rbx]           # len
    cmp rax, [rbx + 8]       # capacity
    jl .has_capacity

    # Need to grow - double capacity or start with 8
    mov rdi, [rbx + 8]
    test rdi, rdi
    jnz .double_cap
    mov rdi, 8
    jmp .alloc_data

.double_cap:
    shl rdi, 1

.alloc_data:
    mov r13, rdi             # new capacity
    shl rdi, 3               # * 8 bytes
    add rdi, 16              # + header
    call _sigil_alloc

    test rax, rax
    jz .push_fail

    # Copy old data if any
    mov rcx, [rbx]           # old len
    test rcx, rcx
    jz .skip_copy

    push rax
    lea rdi, [rax + 16]
    lea rsi, [rbx + 16]
    shl rcx, 3
    rep movsb
    pop rax

.skip_copy:
    mov rcx, [rbx]           # len
    mov [rax], rcx
    mov [rax + 8], r13       # new capacity
    mov rbx, rax

.has_capacity:
    mov rax, [rbx]           # len
    lea rdi, [rbx + 16]
    mov [rdi + rax*8], r12   # store value
    inc qword ptr [rbx]      # len++

    pop r13
    pop r12
    pop rbx
    ret

.push_fail:
    pop r13
    pop r12
    pop rbx
    ret

# sigil_vec_get(v: *const Vec, idx: i64) -> i64
.global _sigil_vec_get
_sigil_vec_get:
    cmp rsi, [rdi]
    jge .vec_bounds_error
    lea rax, [rdi + 16]
    mov rax, [rax + rsi*8]
    ret

.vec_bounds_error:
    xor rax, rax
    ret

# sigil_vec_len(v: *const Vec) -> i64
.global _sigil_vec_len
_sigil_vec_len:
    mov rax, [rdi]
    ret

# ============================================================================
# File I/O
# ============================================================================

# sigil_file_open(path: *const u8, flags: i64) -> i64
.global _sigil_file_open
_sigil_file_open:
    mov rax, SYS_open
    mov rdx, 0644            # mode (octal)
    syscall
    ret

# sigil_file_close(fd: i64) -> i64
.global _sigil_file_close
_sigil_file_close:
    mov rax, SYS_close
    syscall
    ret

# sigil_file_read(fd: i64, buf: *mut u8, len: i64) -> i64
.global _sigil_file_read
_sigil_file_read:
    mov rax, SYS_read
    syscall
    ret

# sigil_file_write(fd: i64, buf: *const u8, len: i64) -> i64
.global _sigil_file_write
_sigil_file_write:
    mov rax, SYS_write
    syscall
    ret

# sigil_file_exists(path: *const u8) -> i64
.global _sigil_file_exists
_sigil_file_exists:
    mov rax, SYS_access
    xor rsi, rsi             # F_OK = 0
    syscall
    cmp rax, 0
    je .exists
    xor rax, rax
    ret
.exists:
    mov rax, 1
    ret

# sigil_file_seek(fd: i64, offset: i64, whence: i64) -> i64
.global _sigil_file_seek
_sigil_file_seek:
    mov rax, SYS_lseek
    syscall
    ret

# ============================================================================
# Math Functions (x87 FPU)
# ============================================================================

# sigil_sqrt(x: i64) -> i64
.global _sigil_sqrt
_sigil_sqrt:
    push rdi
    fld qword ptr [rsp]
    fsqrt
    fstp qword ptr [rsp]
    pop rax
    ret

# sigil_sin(x: i64) -> i64
.global _sigil_sin
_sigil_sin:
    push rdi
    fld qword ptr [rsp]
    fsin
    fstp qword ptr [rsp]
    pop rax
    ret

# sigil_cos(x: i64) -> i64
.global _sigil_cos
_sigil_cos:
    push rdi
    fld qword ptr [rsp]
    fcos
    fstp qword ptr [rsp]
    pop rax
    ret

# sigil_abs(x: i64) -> i64
.global _sigil_abs
_sigil_abs:
    push rdi
    fld qword ptr [rsp]
    fabs
    fstp qword ptr [rsp]
    pop rax
    ret

# sigil_floor(x: i64) -> i64
.global _sigil_floor
_sigil_floor:
    push rdi
    sub rsp, 8

    fnstcw [rsp]
    mov ax, [rsp]
    and ax, 0xF3FF
    or ax, 0x0400
    push ax
    fldcw [rsp]
    add rsp, 2

    fld qword ptr [rsp + 8]
    frndint
    fistp qword ptr [rsp + 8]

    fldcw [rsp]
    add rsp, 8
    pop rax
    ret

# sigil_ceil(x: i64) -> i64
.global _sigil_ceil
_sigil_ceil:
    push rdi
    sub rsp, 8

    fnstcw [rsp]
    mov ax, [rsp]
    and ax, 0xF3FF
    or ax, 0x0800
    push ax
    fldcw [rsp]
    add rsp, 2

    fld qword ptr [rsp + 8]
    frndint
    fistp qword ptr [rsp + 8]

    fldcw [rsp]
    add rsp, 8
    pop rax
    ret

# sigil_min(a: i64, b: i64) -> i64
.global _sigil_min
_sigil_min:
    mov rax, rdi
    cmp rdi, rsi
    cmovg rax, rsi
    ret

# sigil_max(a: i64, b: i64) -> i64
.global _sigil_max
_sigil_max:
    mov rax, rdi
    cmp rdi, rsi
    cmovl rax, rsi
    ret

# ============================================================================
# SIMD Functions (SSE/AVX)
# ============================================================================

# simd_f32x4_add(dst: *mut f32, a: *const f32, b: *const f32)
.global _simd_f32x4_add
_simd_f32x4_add:
    movaps xmm0, [rsi]
    movaps xmm1, [rdx]
    addps xmm0, xmm1
    movaps [rdi], xmm0
    ret

# simd_f32x4_mul(dst: *mut f32, a: *const f32, b: *const f32)
.global _simd_f32x4_mul
_simd_f32x4_mul:
    movaps xmm0, [rsi]
    movaps xmm1, [rdx]
    mulps xmm0, xmm1
    movaps [rdi], xmm0
    ret

# simd_f32x4_dot(a: *const f32, b: *const f32) -> f32
.global _simd_f32x4_dot
_simd_f32x4_dot:
    movaps xmm0, [rdi]
    movaps xmm1, [rsi]
    mulps xmm0, xmm1
    movhlps xmm1, xmm0
    addps xmm0, xmm1
    movaps xmm1, xmm0
    shufps xmm1, xmm1, 1
    addss xmm0, xmm1
    ret

# simd_check_avx() -> i64
.global _simd_check_avx
_simd_check_avx:
    push rbx
    mov eax, 1
    cpuid
    xor rax, rax
    test ecx, 0x10000000
    setnz al
    pop rbx
    ret

# ============================================================================
# Syscall Wrappers
# ============================================================================

# Sys_write(fd: i64, buf: *const u8, len: i64) -> i64
.global _Sys_write
_Sys_write:
    mov rax, SYS_write
    syscall
    ret

# Sys_read(fd: i64, buf: *mut u8, len: i64) -> i64
.global _Sys_read
_Sys_read:
    mov rax, SYS_read
    syscall
    ret

# Sys_open(path: *const u8, flags: i64, mode: i64) -> i64
.global _Sys_open
_Sys_open:
    mov rax, SYS_open
    syscall
    ret

# Sys_close(fd: i64) -> i64
.global _Sys_close
_Sys_close:
    mov rax, SYS_close
    syscall
    ret

# Sys_mmap(addr: *mut u8, len: i64, prot: i64, flags: i64, fd: i64, off: i64) -> *mut u8
.global _Sys_mmap
_Sys_mmap:
    mov rax, SYS_mmap
    mov r10, rcx             # flags in r10 for syscall
    syscall
    ret

# Sys_munmap(addr: *mut u8, len: i64) -> i64
.global _Sys_munmap
_Sys_munmap:
    mov rax, SYS_munmap
    syscall
    ret

# Sys_exit(code: i64)
.global _Sys_exit
_Sys_exit:
    mov rax, SYS_exit
    syscall
    # No return
