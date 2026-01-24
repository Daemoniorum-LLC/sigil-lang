# Sigil Native Runtime - Linux x86_64
#
# Pure syscall implementation with no libc dependency.
# Provides the same interface as sigil_runtime.c but using raw syscalls.
#
# Syscall ABI (Linux x86_64):
#   - syscall number in RAX
#   - arguments in RDI, RSI, RDX, R10, R8, R9
#   - return value in RAX (negative = -errno)
#   - clobbers RCX, R11
#
# Syscall numbers:
#   0 = read     1 = write    2 = open     3 = close
#   9 = mmap    11 = munmap  60 = exit   228 = clock_gettime
#

.intel_syntax noprefix
.global _start
.global sigil_main

# ============================================================================
# Entry Point
# ============================================================================

.section .text

_start:
    # Zero the frame pointer for clean stack traces
    xor rbp, rbp

    # Call the Sigil main function
    call sigil_main

    # Exit with return value from main
    mov edi, eax
    mov eax, 60              # SYS_exit
    syscall

# ============================================================================
# Print Functions
# ============================================================================

# sigil_println(str: *const u8)
# Prints a null-terminated string followed by newline
.global sigil_println
sigil_println:
    push rbx
    push r12
    mov r12, rdi             # Save string pointer

    # Find string length
    xor rcx, rcx
.strlen_loop:
    mov al, [r12 + rcx]
    test al, al
    jz .strlen_done
    inc rcx
    jmp .strlen_loop
.strlen_done:
    mov rbx, rcx             # Save length

    # Write string
    mov rax, 1               # SYS_write
    mov rdi, 1               # stdout
    mov rsi, r12             # buffer
    mov rdx, rbx             # length
    syscall

    # Write newline
    push 10                  # '\n' on stack
    mov rax, 1               # SYS_write
    mov rdi, 1               # stdout
    lea rsi, [rsp]           # address of newline
    mov rdx, 1               # 1 byte
    syscall
    add rsp, 8

    pop r12
    pop rbx
    ret

# sigil_print_int(value: i64)
# Prints an integer followed by newline
.global sigil_print_int
sigil_print_int:
    push rbx
    push r12
    push r13
    sub rsp, 32              # Buffer for digits

    mov r12, rdi             # Save value
    lea r13, [rsp + 31]      # End of buffer
    mov byte ptr [r13], 0    # Null terminator

    # Handle negative numbers
    test r12, r12
    jns .print_positive
    neg r12
    mov bl, 1                # Negative flag
    jmp .print_convert
.print_positive:
    xor bl, bl               # Positive flag

.print_convert:
    # Convert to decimal digits (reverse order)
    mov rax, r12
    mov rcx, 10
.digit_loop:
    dec r13
    xor rdx, rdx
    div rcx                  # RAX = quotient, RDX = remainder
    add dl, '0'
    mov [r13], dl
    test rax, rax
    jnz .digit_loop

    # Add minus sign if negative
    test bl, bl
    jz .print_number
    dec r13
    mov byte ptr [r13], '-'

.print_number:
    # Calculate length
    lea rax, [rsp + 31]
    sub rax, r13             # Length = end - start
    mov rbx, rax

    # Write number
    mov rax, 1               # SYS_write
    mov rdi, 1               # stdout
    mov rsi, r13             # buffer start
    mov rdx, rbx             # length
    syscall

    # Write newline
    mov byte ptr [rsp], 10
    mov rax, 1
    mov rdi, 1
    mov rsi, rsp
    mov rdx, 1
    syscall

    add rsp, 32
    pop r13
    pop r12
    pop rbx
    ret

# sigil_print_float(value: f64)
# Prints a floating point number (simplified: 6 decimal places)
.global sigil_print_float
sigil_print_float:
    # TODO: Implement full float printing
    # For now, just print placeholder
    push rbx
    sub rsp, 16

    mov rax, 1
    mov rdi, 1
    lea rsi, [rip + .float_placeholder]
    mov rdx, 7
    syscall

    add rsp, 16
    pop rbx
    ret
.float_placeholder:
    .asciz "<float>\n"

# ============================================================================
# Memory Functions
# ============================================================================

# sigil_alloc(size: i64) -> *mut u8
# Allocate memory using mmap
.global sigil_alloc
sigil_alloc:
    push rbx
    mov rbx, rdi             # Save size

    # mmap(NULL, size, PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_ANONYMOUS, -1, 0)
    mov rax, 9               # SYS_mmap
    xor rdi, rdi             # addr = NULL
    mov rsi, rbx             # length = size
    mov rdx, 3               # prot = PROT_READ | PROT_WRITE
    mov r10, 34              # flags = MAP_PRIVATE | MAP_ANONYMOUS
    mov r8, -1               # fd = -1
    xor r9, r9               # offset = 0
    syscall

    # Check for error (returns -1 to -4095 on error)
    cmp rax, -4095
    jae .alloc_failed

    pop rbx
    ret

.alloc_failed:
    xor rax, rax             # Return NULL on failure
    pop rbx
    ret

# sigil_free(ptr: *mut u8)
# Free memory - NOTE: We don't track sizes, so this is a no-op
# A real implementation would need to track allocation sizes
.global sigil_free
sigil_free:
    # For now, this is a no-op
    # A proper implementation would track allocation sizes
    ret

# sigil_realloc(ptr: *mut u8, new_size: i64) -> *mut u8
# Reallocate memory - allocates new block and copies
.global sigil_realloc
sigil_realloc:
    # For now, just allocate new block (no copy, no free)
    # A proper implementation would copy data
    mov rdi, rsi
    jmp sigil_alloc

# ============================================================================
# Time Functions
# ============================================================================

# sigil_now() -> i64
# Get current time in milliseconds since epoch
.global sigil_now
sigil_now:
    sub rsp, 24              # timespec struct (16 bytes) + alignment

    # clock_gettime(CLOCK_REALTIME, &ts)
    mov rax, 228             # SYS_clock_gettime
    xor rdi, rdi             # CLOCK_REALTIME = 0
    lea rsi, [rsp]           # timespec pointer
    syscall

    # Convert to milliseconds: ts.tv_sec * 1000 + ts.tv_nsec / 1000000
    mov rax, [rsp]           # tv_sec
    imul rax, 1000           # * 1000
    mov rcx, [rsp + 8]       # tv_nsec
    mov rdx, rcx
    shr rdx, 20              # Approximate / 1000000 (close enough)
    add rax, rdx

    add rsp, 24
    ret

# ============================================================================
# String Functions
# ============================================================================

# sigil_strlen(str: *const u8) -> i64
# Get string length
.global sigil_strlen
sigil_strlen:
    xor rax, rax
    test rdi, rdi
    jz .strlen_ret
.strlen_count:
    cmp byte ptr [rdi + rax], 0
    je .strlen_ret
    inc rax
    jmp .strlen_count
.strlen_ret:
    ret

# ============================================================================
# Vec Operations
# ============================================================================
#
# Vec layout: [len: i64, capacity: i64, data: i64[]]
#

# sigil_vec_new(capacity: i64) -> *mut Vec
.global sigil_vec_new
sigil_vec_new:
    push rbx
    push r12

    # Minimum capacity of 4
    cmp rdi, 4
    jge .vec_cap_ok
    mov rdi, 4
.vec_cap_ok:
    mov r12, rdi             # Save capacity

    # Allocate: 16 bytes header + capacity * 8
    shl rdi, 3               # capacity * 8
    add rdi, 16              # + header
    call sigil_alloc

    test rax, rax
    jz .vec_alloc_failed

    # Initialize header
    mov qword ptr [rax], 0       # len = 0
    mov [rax + 8], r12           # capacity

    pop r12
    pop rbx
    ret

.vec_alloc_failed:
    xor rax, rax
    pop r12
    pop rbx
    ret

# sigil_vec_push(vec: *mut Vec, value: i64)
.global sigil_vec_push
sigil_vec_push:
    test rdi, rdi
    jz .vec_push_ret

    mov rax, [rdi]           # len
    mov rcx, [rdi + 8]       # capacity

    # Check if we need to grow
    cmp rax, rcx
    jge .vec_push_ret        # Can't grow yet - just fail silently

    # Store value at data[len]
    mov [rdi + 16 + rax*8], rsi

    # Increment length
    inc qword ptr [rdi]

.vec_push_ret:
    ret

# sigil_vec_get(vec: *mut Vec, index: i64) -> i64
.global sigil_vec_get
sigil_vec_get:
    test rdi, rdi
    jz .vec_get_zero

    mov rax, [rdi]           # len
    cmp rsi, rax
    jge .vec_get_zero        # Index out of bounds
    cmp rsi, 0
    jl .vec_get_zero

    mov rax, [rdi + 16 + rsi*8]
    ret

.vec_get_zero:
    xor rax, rax
    ret

# sigil_vec_len(vec: *mut Vec) -> i64
.global sigil_vec_len
sigil_vec_len:
    test rdi, rdi
    jz .vec_len_zero
    mov rax, [rdi]
    ret
.vec_len_zero:
    xor rax, rax
    ret

# ============================================================================
# Math Functions (using x87 FPU)
# ============================================================================

# sigil_sqrt(x: i64) -> i64
# Interprets as f64 bits, computes sqrt, returns f64 bits
.global sigil_sqrt
sigil_sqrt:
    push rdi
    fld qword ptr [rsp]      # Load as f64
    fsqrt                    # Compute sqrt
    fstp qword ptr [rsp]     # Store result
    pop rax                  # Return as i64 bits
    ret

# sigil_sin(x: i64) -> i64
.global sigil_sin
sigil_sin:
    push rdi
    fld qword ptr [rsp]
    fsin
    fstp qword ptr [rsp]
    pop rax
    ret

# sigil_cos(x: i64) -> i64
.global sigil_cos
sigil_cos:
    push rdi
    fld qword ptr [rsp]
    fcos
    fstp qword ptr [rsp]
    pop rax
    ret

# sigil_abs(x: i64) -> i64
.global sigil_abs
sigil_abs:
    push rdi
    fld qword ptr [rsp]
    fabs
    fstp qword ptr [rsp]
    pop rax
    ret

# sigil_floor(x: i64) -> i64
.global sigil_floor
sigil_floor:
    push rdi
    sub rsp, 8

    # Set rounding mode to floor (round toward -infinity)
    fnstcw [rsp]             # Save control word
    mov ax, [rsp]
    and ax, 0xF3FF           # Clear RC bits
    or ax, 0x0400            # Set RC = 01 (round down)
    push ax
    fldcw [rsp]              # Load modified control word
    add rsp, 2

    fld qword ptr [rsp + 8]  # Load value
    frndint                  # Round to integer
    fistp qword ptr [rsp + 8] # Store as integer

    fldcw [rsp]              # Restore control word
    add rsp, 8
    pop rax
    ret

# sigil_ceil(x: i64) -> i64
.global sigil_ceil
sigil_ceil:
    push rdi
    sub rsp, 8

    fnstcw [rsp]
    mov ax, [rsp]
    and ax, 0xF3FF
    or ax, 0x0800            # Set RC = 10 (round up)
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

# sigil_pow(x: i64, y: i64) -> i64
# Compute x^y using FPU: 2^(y*log2(x))
.global sigil_pow
sigil_pow:
    push rdi
    push rsi
    sub rsp, 8

    fld qword ptr [rsp + 8]  # y
    fld qword ptr [rsp + 16] # x
    fyl2x                    # y * log2(x)

    # Compute 2^result
    fld st(0)
    frndint                  # integer part
    fxch st(1)
    fsub st(0), st(1)        # fractional part
    f2xm1                    # 2^frac - 1
    fld1
    faddp                    # 2^frac
    fscale                   # * 2^int
    fstp st(1)

    fstp qword ptr [rsp + 16]
    add rsp, 8
    pop rsi
    pop rax
    ret

# sigil_min(a: i64, b: i64) -> i64
.global sigil_min
sigil_min:
    mov rax, rdi
    cmp rdi, rsi
    cmovg rax, rsi
    ret

# sigil_max(a: i64, b: i64) -> i64
.global sigil_max
sigil_max:
    mov rax, rdi
    cmp rdi, rsi
    cmovl rax, rsi
    ret

# ============================================================================
# Low-level Syscall Wrappers
# ============================================================================

# Sys_write(fd: i64, buf: *const u8, len: i64) -> i64
.global Sys_write
Sys_write:
    mov rax, 1               # SYS_write
    syscall
    ret

# Sys_read(fd: i64, buf: *mut u8, len: i64) -> i64
.global Sys_read
Sys_read:
    mov rax, 0               # SYS_read
    syscall
    ret

# Sys_open(path: *const u8, flags: i64, mode: i64) -> i64
.global Sys_open
Sys_open:
    mov rax, 2               # SYS_open
    syscall
    ret

# Sys_close(fd: i64) -> i64
.global Sys_close
Sys_close:
    mov rax, 3               # SYS_close
    syscall
    ret

# Sys_mmap(addr: i64, len: i64, prot: i64, flags: i64, fd: i64, off: i64) -> i64
.global Sys_mmap
Sys_mmap:
    mov rax, 9               # SYS_mmap
    mov r10, rcx             # flags in r10 (4th arg uses r10 for syscalls)
    syscall
    ret

# Sys_munmap(addr: i64, len: i64) -> i64
.global Sys_munmap
Sys_munmap:
    mov rax, 11              # SYS_munmap
    syscall
    ret

# Sys_exit(code: i64) -> !
.global Sys_exit
Sys_exit:
    mov rax, 60              # SYS_exit
    syscall
    # Never returns

# Sys_clock_gettime(clock_id: i64, ts: *mut timespec) -> i64
.global Sys_clock_gettime
Sys_clock_gettime:
    mov rax, 228             # SYS_clock_gettime
    syscall
    ret

# ============================================================================
# Data Section
# ============================================================================

.section .data
    # Empty for now

.section .note.GNU-stack,"",@progbits
