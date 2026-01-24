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
# Memory Functions - Arena/Bump Allocator
# ============================================================================
#
# Fast O(1) bump allocator with arena management:
# - Small allocations (< 4KB): bump pointer in current arena
# - Large allocations (>= 4KB): direct mmap
# - Arena size: 1MB (expandable)
# - 16-byte alignment for all allocations
#
# Arena layout:
#   [next_arena: ptr][end: ptr][bump: ptr][...data...]
#   ^-- arena_current points here
#

.section .bss
    .align 8
arena_current:    .quad 0      # Current arena pointer
arena_bump:       .quad 0      # Current bump pointer
arena_end:        .quad 0      # End of current arena

.section .text

# Constants
.equ ARENA_SIZE, 1048576       # 1MB arena
.equ LARGE_ALLOC_THRESHOLD, 4096
.equ ARENA_HEADER_SIZE, 24     # next + end + bump pointers

# sigil_arena_init() -> i64
# Initialize the arena allocator. Called automatically on first alloc.
.global sigil_arena_init
sigil_arena_init:
    push rbx

    # Check if already initialized
    mov rax, [rip + arena_current]
    test rax, rax
    jnz .arena_already_init

    # Allocate first arena via mmap
    mov rax, 9               # SYS_mmap
    xor rdi, rdi             # addr = NULL
    mov rsi, ARENA_SIZE      # length
    mov rdx, 3               # PROT_READ | PROT_WRITE
    mov r10, 34              # MAP_PRIVATE | MAP_ANONYMOUS
    mov r8, -1               # fd = -1
    xor r9, r9               # offset = 0
    syscall

    cmp rax, -4095
    jae .arena_init_failed

    # Initialize arena header
    mov [rip + arena_current], rax
    mov qword ptr [rax], 0           # next = NULL
    lea rbx, [rax + ARENA_SIZE]
    mov [rax + 8], rbx               # end
    lea rbx, [rax + ARENA_HEADER_SIZE]
    mov [rax + 16], rbx              # bump (after header)
    mov [rip + arena_bump], rbx
    mov rax, [rip + arena_current]
    add rax, ARENA_SIZE
    mov [rip + arena_end], rax

    mov rax, 1               # Success
    pop rbx
    ret

.arena_already_init:
    mov rax, 1
    pop rbx
    ret

.arena_init_failed:
    xor rax, rax             # Failure
    pop rbx
    ret

# sigil_alloc(size: i64) -> *mut u8
# Fast bump allocation with large alloc fallback
.global sigil_alloc
sigil_alloc:
    push rbx
    push r12

    # Ensure arena is initialized
    mov rax, [rip + arena_current]
    test rax, rax
    jnz .arena_ready
    call sigil_arena_init
    test rax, rax
    jz .alloc_failed_arena

.arena_ready:
    mov r12, rdi             # Save requested size

    # Align size to 16 bytes
    add r12, 15
    and r12, -16

    # Check if large allocation (bypass arena)
    cmp r12, LARGE_ALLOC_THRESHOLD
    jge .large_alloc

    # Try bump allocation
    mov rax, [rip + arena_bump]
    lea rbx, [rax + r12]     # new_bump = bump + size

    # Check if fits in current arena
    cmp rbx, [rip + arena_end]
    ja .need_new_arena

    # Bump allocation succeeded
    mov [rip + arena_bump], rbx
    pop r12
    pop rbx
    ret

.need_new_arena:
    # Allocate new arena
    push r12                 # Save size
    mov rax, 9               # SYS_mmap
    xor rdi, rdi
    mov rsi, ARENA_SIZE
    mov rdx, 3
    mov r10, 34
    mov r8, -1
    xor r9, r9
    syscall
    pop r12

    cmp rax, -4095
    jae .alloc_failed_arena

    # Link new arena to current
    mov rbx, [rip + arena_current]
    mov [rax], rbx           # new->next = current

    # Update current arena
    mov [rip + arena_current], rax
    lea rbx, [rax + ARENA_SIZE]
    mov [rax + 8], rbx       # end
    mov [rip + arena_end], rbx
    lea rbx, [rax + ARENA_HEADER_SIZE]
    mov [rax + 16], rbx      # bump

    # Now allocate from new arena
    lea rax, [rbx + r12]     # new_bump
    mov [rip + arena_bump], rax
    mov rax, rbx             # Return start of allocation

    pop r12
    pop rbx
    ret

.large_alloc:
    # Direct mmap for large allocations
    # Add 8 bytes header to store size for potential realloc/free
    add r12, 8

    mov rax, 9               # SYS_mmap
    xor rdi, rdi
    mov rsi, r12
    mov rdx, 3
    mov r10, 34
    mov r8, -1
    xor r9, r9
    syscall

    cmp rax, -4095
    jae .alloc_failed_arena

    # Store size in header
    sub r12, 8
    mov [rax], r12
    add rax, 8               # Return pointer after header

    pop r12
    pop rbx
    ret

.alloc_failed_arena:
    xor rax, rax
    pop r12
    pop rbx
    ret

# sigil_free(ptr: *mut u8)
# Free memory - only works for large allocations (mmap'd directly)
# Arena allocations are freed when arena is reset/dropped
.global sigil_free
sigil_free:
    # Check for NULL
    test rdi, rdi
    jz .free_done

    # Check if this is a large allocation (outside arena range)
    # For now, we don't track this properly - just no-op
    # A full implementation would check if ptr is in arena range

.free_done:
    ret

# sigil_realloc(ptr: *mut u8, new_size: i64) -> *mut u8
# Reallocate memory - for arena allocs, just allocate new and copy
.global sigil_realloc
sigil_realloc:
    push rbx
    push r12
    push r13

    mov r12, rdi             # old ptr
    mov r13, rsi             # new size

    # Handle NULL ptr - just allocate
    test r12, r12
    jz .realloc_just_alloc

    # Allocate new block
    mov rdi, r13
    call sigil_alloc
    test rax, rax
    jz .realloc_failed

    mov rbx, rax             # new ptr

    # Copy data (assume old size >= new size for safety)
    # In a real impl, we'd track the old size
    mov rdi, rbx             # dest
    mov rsi, r12             # src
    mov rcx, r13             # count (use new size as upper bound)
    rep movsb

    mov rax, rbx
    pop r13
    pop r12
    pop rbx
    ret

.realloc_just_alloc:
    mov rdi, r13
    call sigil_alloc
    pop r13
    pop r12
    pop rbx
    ret

.realloc_failed:
    xor rax, rax
    pop r13
    pop r12
    pop rbx
    ret

# sigil_arena_reset()
# Reset all arenas (free all allocations but keep arena memory)
.global sigil_arena_reset
sigil_arena_reset:
    mov rax, [rip + arena_current]
    test rax, rax
    jz .reset_done

    # Reset bump pointer to after header
    lea rbx, [rax + ARENA_HEADER_SIZE]
    mov [rip + arena_bump], rbx
    mov [rax + 16], rbx

.reset_done:
    ret

# sigil_arena_stats() -> (total_arenas: i64, total_bytes: i64)
# Returns stats in rax (arenas) and rdx (bytes)
.global sigil_arena_stats
sigil_arena_stats:
    xor rax, rax             # arena count
    xor rdx, rdx             # total bytes

    mov rcx, [rip + arena_current]
.stats_loop:
    test rcx, rcx
    jz .stats_done

    inc rax
    add rdx, ARENA_SIZE
    mov rcx, [rcx]           # next arena
    jmp .stats_loop

.stats_done:
    ret

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
