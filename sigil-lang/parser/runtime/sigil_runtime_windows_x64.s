# Sigil Native Runtime - Windows x64
# Pure assembly, minimal runtime dependency
# Uses Windows API via import library (kernel32)

.intel_syntax noprefix

.section .data
    # Standard handles (populated at startup)
    stdin_handle:  .quad 0
    stdout_handle: .quad 0
    stderr_handle: .quad 0

    # Arena allocator state
    arena_current: .quad 0
    arena_bump:    .quad 0
    arena_end:     .quad 0

.section .text

# Windows Constants
.set STD_INPUT_HANDLE,  -10
.set STD_OUTPUT_HANDLE, -11
.set STD_ERROR_HANDLE,  -12

.set MEM_COMMIT,   0x1000
.set MEM_RESERVE,  0x2000
.set MEM_RELEASE,  0x8000
.set PAGE_READWRITE, 0x04

.set GENERIC_READ,  0x80000000
.set GENERIC_WRITE, 0x40000000
.set FILE_SHARE_READ, 0x01
.set CREATE_ALWAYS, 2
.set OPEN_EXISTING, 3

.set ARENA_SIZE, 0x100000    # 1MB arenas

# ============================================================================
# External Windows API imports
# ============================================================================
.extern GetStdHandle
.extern WriteFile
.extern ReadFile
.extern VirtualAlloc
.extern VirtualFree
.extern CreateFileA
.extern CloseHandle
.extern GetFileSizeEx
.extern SetFilePointerEx
.extern GetLastError
.extern ExitProcess

# ============================================================================
# Entry Point
# ============================================================================
.global main
.global _main
main:
_main:
    push rbp
    mov rbp, rsp
    sub rsp, 32              # Shadow space for Windows x64 ABI

    # Get standard handles
    mov ecx, STD_INPUT_HANDLE
    call GetStdHandle
    mov [rip + stdin_handle], rax

    mov ecx, STD_OUTPUT_HANDLE
    call GetStdHandle
    mov [rip + stdout_handle], rax

    mov ecx, STD_ERROR_HANDLE
    call GetStdHandle
    mov [rip + stderr_handle], rax

    # Call Sigil main function
    call sigil_main

    # Exit with return value
    mov ecx, eax
    call ExitProcess

    # Never returns
    add rsp, 32
    pop rbp
    ret

# ============================================================================
# Print Functions
# ============================================================================

# sigil_println(str: *const u8, len: i64)
.global sigil_println
sigil_println:
    push rbp
    mov rbp, rsp
    push rbx
    push r12
    sub rsp, 48              # Local vars + shadow space

    mov r12, rcx             # Save string pointer
    mov rbx, rdx             # Save length

    # WriteFile(stdout, str, len, &written, NULL)
    mov rcx, [rip + stdout_handle]
    mov rdx, r12
    mov r8d, ebx
    lea r9, [rsp + 32]       # &written
    mov qword ptr [rsp + 32], 0
    call WriteFile

    # Write newline
    mov byte ptr [rsp + 40], 10
    mov rcx, [rip + stdout_handle]
    lea rdx, [rsp + 40]
    mov r8d, 1
    lea r9, [rsp + 32]
    call WriteFile

    add rsp, 48
    pop r12
    pop rbx
    pop rbp
    ret

# sigil_print_int(n: i64)
.global sigil_print_int
sigil_print_int:
    push rbp
    mov rbp, rsp
    push rbx
    sub rsp, 64              # Buffer + shadow space

    mov rax, rcx             # Number to print
    lea r11, [rsp + 56]      # Buffer end

    # Handle negative
    xor rbx, rbx
    test rax, rax
    jns .pi_positive
    neg rax
    mov rbx, 1

.pi_positive:
    mov r10, 10

.pi_digit_loop:
    xor rdx, rdx
    div r10
    add dl, '0'
    dec r11
    mov [r11], dl
    test rax, rax
    jnz .pi_digit_loop

    # Add minus if negative
    test rbx, rbx
    jz .pi_print
    dec r11
    mov byte ptr [r11], '-'

.pi_print:
    # Calculate length
    lea rax, [rsp + 56]
    sub rax, r11

    # WriteFile
    mov rcx, [rip + stdout_handle]
    mov rdx, r11
    mov r8, rax
    lea r9, [rsp + 32]
    call WriteFile

    # Write newline
    mov byte ptr [rsp + 40], 10
    mov rcx, [rip + stdout_handle]
    lea rdx, [rsp + 40]
    mov r8d, 1
    lea r9, [rsp + 32]
    call WriteFile

    add rsp, 64
    pop rbx
    pop rbp
    ret

# ============================================================================
# Memory Management - Arena Allocator
# ============================================================================

# sigil_alloc(size: i64) -> *mut u8
.global sigil_alloc
sigil_alloc:
    push rbp
    mov rbp, rsp
    push rbx
    push r12
    sub rsp, 32

    mov rbx, rcx             # Size to allocate

    # Check if we have space in current arena
    mov rax, [rip + arena_bump]
    test rax, rax
    jz .need_arena

    add rax, rbx
    cmp rax, [rip + arena_end]
    jg .need_arena

    # Bump allocate
    mov rax, [rip + arena_bump]
    add [rip + arena_bump], rbx

    add rsp, 32
    pop r12
    pop rbx
    pop rbp
    ret

.need_arena:
    # VirtualAlloc(NULL, ARENA_SIZE, MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE)
    xor rcx, rcx
    mov rdx, ARENA_SIZE
    mov r8d, MEM_COMMIT | MEM_RESERVE
    mov r9d, PAGE_READWRITE
    call VirtualAlloc

    test rax, rax
    jz .alloc_fail

    # Set up arena
    mov [rip + arena_current], rax
    lea rcx, [rax + rbx]
    mov [rip + arena_bump], rcx
    lea rcx, [rax + ARENA_SIZE]
    mov [rip + arena_end], rcx

    add rsp, 32
    pop r12
    pop rbx
    pop rbp
    ret

.alloc_fail:
    xor rax, rax
    add rsp, 32
    pop r12
    pop rbx
    pop rbp
    ret

# sigil_free(ptr: *mut u8)
.global sigil_free
sigil_free:
    # No-op for arena allocator
    ret

# ============================================================================
# String Functions
# ============================================================================

# sigil_string_from(data: *const u8, len: i64) -> *mut String
.global sigil_string_from
sigil_string_from:
    push rbp
    mov rbp, rsp
    push rbx
    push r12
    push r13
    sub rsp, 40

    mov r12, rcx             # data
    mov r13, rdx             # len

    # Allocate: 16 bytes header + len + 1
    lea rcx, [rdx + 17]
    call sigil_alloc

    test rax, rax
    jz .sf_fail

    mov rbx, rax             # Save result

    # Store length and capacity
    mov [rax], r13
    mov [rax + 8], r13

    # Copy data
    lea rdi, [rax + 16]
    mov rsi, r12
    mov rcx, r13

.sf_copy:
    test rcx, rcx
    jz .sf_done
    mov al, [rsi]
    mov [rdi], al
    inc rsi
    inc rdi
    dec rcx
    jmp .sf_copy

.sf_done:
    mov byte ptr [rdi], 0    # Null terminate
    mov rax, rbx

    add rsp, 40
    pop r13
    pop r12
    pop rbx
    pop rbp
    ret

.sf_fail:
    xor rax, rax
    add rsp, 40
    pop r13
    pop r12
    pop rbx
    pop rbp
    ret

# sigil_string_len(s: *const String) -> i64
.global sigil_string_len
sigil_string_len:
    mov rax, [rcx]
    ret

# sigil_string_as_ptr(s: *const String) -> *const u8
.global sigil_string_as_ptr
sigil_string_as_ptr:
    lea rax, [rcx + 16]
    ret

# sigil_string_print(s: *const String)
.global sigil_string_print
sigil_string_print:
    push rbp
    mov rbp, rsp
    sub rsp, 48

    mov r10, rcx             # string ptr
    mov r8, [rcx]            # length
    lea rdx, [rcx + 16]      # data ptr
    mov rcx, [rip + stdout_handle]
    lea r9, [rsp + 32]
    call WriteFile

    add rsp, 48
    pop rbp
    ret

# sigil_string_eq(a: *const String, b: *const String) -> i64
.global sigil_string_eq
sigil_string_eq:
    mov rax, [rcx]
    cmp rax, [rdx]
    jne .seq_ne

    lea r8, [rcx + 16]
    lea r9, [rdx + 16]
    mov rcx, rax

.seq_cmp:
    test rcx, rcx
    jz .seq_eq
    mov al, [r8]
    cmp al, [r9]
    jne .seq_ne
    inc r8
    inc r9
    dec rcx
    jmp .seq_cmp

.seq_eq:
    mov rax, 1
    ret

.seq_ne:
    xor rax, rax
    ret

# ============================================================================
# Vec Functions
# ============================================================================

# sigil_vec_new() -> *mut Vec
.global sigil_vec_new
sigil_vec_new:
    push rbp
    mov rbp, rsp
    sub rsp, 32

    mov rcx, 16
    call sigil_alloc

    test rax, rax
    jz .vn_fail

    mov qword ptr [rax], 0       # len = 0
    mov qword ptr [rax + 8], 0   # capacity = 0

.vn_fail:
    add rsp, 32
    pop rbp
    ret

# sigil_vec_push(v: *mut Vec, val: i64)
.global sigil_vec_push
sigil_vec_push:
    push rbp
    mov rbp, rsp
    push rbx
    push r12
    push r13
    sub rsp, 40

    mov rbx, rcx             # vec
    mov r12, rdx             # value

    mov rax, [rbx]           # len
    cmp rax, [rbx + 8]       # capacity
    jl .vp_has_cap

    # Need to grow
    mov rcx, [rbx + 8]
    test rcx, rcx
    jnz .vp_double
    mov rcx, 8
    jmp .vp_alloc

.vp_double:
    shl rcx, 1

.vp_alloc:
    mov r13, rcx             # new capacity
    shl rcx, 3
    add rcx, 16
    call sigil_alloc

    test rax, rax
    jz .vp_fail

    # Copy old data
    mov rcx, [rbx]           # old len
    test rcx, rcx
    jz .vp_skip_copy

    lea rdi, [rax + 16]
    lea rsi, [rbx + 16]

.vp_copy:
    test rcx, rcx
    jz .vp_skip_copy
    mov r8, [rsi]
    mov [rdi], r8
    add rsi, 8
    add rdi, 8
    dec rcx
    jmp .vp_copy

.vp_skip_copy:
    mov rcx, [rbx]           # old len
    mov [rax], rcx
    mov [rax + 8], r13       # new capacity
    mov rbx, rax

.vp_has_cap:
    mov rax, [rbx]           # len
    lea rcx, [rbx + 16]
    mov [rcx + rax*8], r12   # store value
    inc qword ptr [rbx]      # len++

.vp_fail:
    add rsp, 40
    pop r13
    pop r12
    pop rbx
    pop rbp
    ret

# sigil_vec_get(v: *const Vec, idx: i64) -> i64
.global sigil_vec_get
sigil_vec_get:
    cmp rdx, [rcx]
    jge .vg_bounds
    lea rax, [rcx + 16]
    mov rax, [rax + rdx*8]
    ret

.vg_bounds:
    xor rax, rax
    ret

# sigil_vec_len(v: *const Vec) -> i64
.global sigil_vec_len
sigil_vec_len:
    mov rax, [rcx]
    ret

# ============================================================================
# File I/O
# ============================================================================

# sigil_file_open(path: *const u8, flags: i64) -> i64
# flags: 0 = read, 1 = write, 2 = read/write
.global sigil_file_open
sigil_file_open:
    push rbp
    mov rbp, rsp
    sub rsp, 64

    # Determine access mode
    mov r10, rdx
    xor r8d, r8d             # dwShareMode = 0

    test r10d, 1
    jz .fo_not_write
    mov edx, GENERIC_WRITE
    mov r9d, CREATE_ALWAYS
    jmp .fo_call

.fo_not_write:
    mov edx, GENERIC_READ
    mov r9d, OPEN_EXISTING

.fo_call:
    # CreateFileA(path, access, share, NULL, creation, 0, NULL)
    mov [rsp + 32], r9       # dwCreationDisposition
    mov qword ptr [rsp + 40], 0  # dwFlagsAndAttributes
    mov qword ptr [rsp + 48], 0  # hTemplateFile
    xor r9, r9               # lpSecurityAttributes = NULL
    or r8d, FILE_SHARE_READ
    call CreateFileA

    # Return handle (or -1 on error)
    add rsp, 64
    pop rbp
    ret

# sigil_file_close(fd: i64) -> i64
.global sigil_file_close
sigil_file_close:
    push rbp
    mov rbp, rsp
    sub rsp, 32

    call CloseHandle

    add rsp, 32
    pop rbp
    ret

# sigil_file_read(fd: i64, buf: *mut u8, len: i64) -> i64
.global sigil_file_read
sigil_file_read:
    push rbp
    mov rbp, rsp
    sub rsp, 48

    # ReadFile(handle, buf, len, &bytesRead, NULL)
    lea r9, [rsp + 32]       # &bytesRead
    mov qword ptr [rsp + 40], 0  # lpOverlapped = NULL
    call ReadFile

    test eax, eax
    jz .fr_error
    mov eax, [rsp + 32]      # Return bytes read
    jmp .fr_done

.fr_error:
    mov rax, -1

.fr_done:
    add rsp, 48
    pop rbp
    ret

# sigil_file_write(fd: i64, buf: *const u8, len: i64) -> i64
.global sigil_file_write
sigil_file_write:
    push rbp
    mov rbp, rsp
    sub rsp, 48

    # WriteFile(handle, buf, len, &bytesWritten, NULL)
    lea r9, [rsp + 32]
    mov qword ptr [rsp + 40], 0
    call WriteFile

    test eax, eax
    jz .fw_error
    mov eax, [rsp + 32]
    jmp .fw_done

.fw_error:
    mov rax, -1

.fw_done:
    add rsp, 48
    pop rbp
    ret

# sigil_file_seek(fd: i64, offset: i64, whence: i64) -> i64
.global sigil_file_seek
sigil_file_seek:
    push rbp
    mov rbp, rsp
    sub rsp, 48

    # SetFilePointerEx(handle, distance, &newPos, moveMethod)
    mov [rsp + 32], r8       # moveMethod
    lea r9, [rsp + 40]       # &newPos
    mov r8d, r8d
    call SetFilePointerEx

    test eax, eax
    jz .fs_error
    mov rax, [rsp + 40]
    jmp .fs_done

.fs_error:
    mov rax, -1

.fs_done:
    add rsp, 48
    pop rbp
    ret

# ============================================================================
# Math Functions (x87 FPU / SSE)
# ============================================================================

# sigil_sqrt(x: i64) -> i64 (bits of f64)
.global sigil_sqrt
sigil_sqrt:
    movq xmm0, rcx
    sqrtsd xmm0, xmm0
    movq rax, xmm0
    ret

# sigil_abs(x: i64) -> i64 (bits of f64)
.global sigil_abs
sigil_abs:
    movq xmm0, rcx
    mov rax, 0x7FFFFFFFFFFFFFFF
    movq xmm1, rax
    andpd xmm0, xmm1
    movq rax, xmm0
    ret

# sigil_min(a: i64, b: i64) -> i64
.global sigil_min
sigil_min:
    mov rax, rcx
    cmp rcx, rdx
    cmovg rax, rdx
    ret

# sigil_max(a: i64, b: i64) -> i64
.global sigil_max
sigil_max:
    mov rax, rcx
    cmp rcx, rdx
    cmovl rax, rdx
    ret

# ============================================================================
# SIMD Functions (SSE/AVX)
# ============================================================================

# simd_f32x4_add(dst: *mut f32, a: *const f32, b: *const f32)
.global simd_f32x4_add
simd_f32x4_add:
    movaps xmm0, [rdx]
    movaps xmm1, [r8]
    addps xmm0, xmm1
    movaps [rcx], xmm0
    ret

# simd_f32x4_mul(dst: *mut f32, a: *const f32, b: *const f32)
.global simd_f32x4_mul
simd_f32x4_mul:
    movaps xmm0, [rdx]
    movaps xmm1, [r8]
    mulps xmm0, xmm1
    movaps [rcx], xmm0
    ret

# simd_f32x4_dot(a: *const f32, b: *const f32) -> f32
.global simd_f32x4_dot
simd_f32x4_dot:
    movaps xmm0, [rcx]
    movaps xmm1, [rdx]
    mulps xmm0, xmm1
    movhlps xmm1, xmm0
    addps xmm0, xmm1
    movaps xmm1, xmm0
    shufps xmm1, xmm1, 1
    addss xmm0, xmm1
    ret

# simd_f32x8_add(dst: *mut f32, a: *const f32, b: *const f32)
.global simd_f32x8_add
simd_f32x8_add:
    vmovaps ymm0, [rdx]
    vmovaps ymm1, [r8]
    vaddps ymm0, ymm0, ymm1
    vmovaps [rcx], ymm0
    vzeroupper
    ret

# simd_f32x8_mul(dst: *mut f32, a: *const f32, b: *const f32)
.global simd_f32x8_mul
simd_f32x8_mul:
    vmovaps ymm0, [rdx]
    vmovaps ymm1, [r8]
    vmulps ymm0, ymm0, ymm1
    vmovaps [rcx], ymm0
    vzeroupper
    ret

# simd_check_avx() -> i64
.global simd_check_avx
simd_check_avx:
    push rbx
    mov eax, 1
    cpuid
    xor rax, rax
    test ecx, 0x10000000
    setnz al
    pop rbx
    ret

# ============================================================================
# Syscall-style wrappers (for API compatibility)
# ============================================================================

# Sys_write(fd: i64, buf: *const u8, len: i64) -> i64
.global Sys_write
Sys_write:
    push rbp
    mov rbp, rsp
    sub rsp, 48

    # Map fd to handle
    cmp rcx, 1
    jne .sw_stderr
    mov rcx, [rip + stdout_handle]
    jmp .sw_write

.sw_stderr:
    cmp rcx, 2
    jne .sw_fail
    mov rcx, [rip + stderr_handle]

.sw_write:
    lea r9, [rsp + 32]
    mov qword ptr [rsp + 40], 0
    call WriteFile

    test eax, eax
    jz .sw_fail
    mov eax, [rsp + 32]
    jmp .sw_done

.sw_fail:
    mov rax, -1

.sw_done:
    add rsp, 48
    pop rbp
    ret

# Sys_read(fd: i64, buf: *mut u8, len: i64) -> i64
.global Sys_read
Sys_read:
    push rbp
    mov rbp, rsp
    sub rsp, 48

    cmp rcx, 0
    jne .sr_fail
    mov rcx, [rip + stdin_handle]

    lea r9, [rsp + 32]
    mov qword ptr [rsp + 40], 0
    call ReadFile

    test eax, eax
    jz .sr_fail
    mov eax, [rsp + 32]
    jmp .sr_done

.sr_fail:
    mov rax, -1

.sr_done:
    add rsp, 48
    pop rbp
    ret

# Sys_exit(code: i64)
.global Sys_exit
Sys_exit:
    sub rsp, 40
    call ExitProcess
    # No return
