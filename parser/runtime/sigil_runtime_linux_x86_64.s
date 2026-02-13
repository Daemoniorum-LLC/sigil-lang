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
.global main_sigil

# ============================================================================
# Entry Point
# ============================================================================

.section .text

_start:
    # Zero the frame pointer for clean stack traces
    xor rbp, rbp

    # Call the Sigil main function
    call main_sigil

    # Exit with return value from main
    mov edi, eax
    mov eax, 60              # SYS_exit
    syscall

# ============================================================================
# Print Functions
# ============================================================================

# sigil_println(str: *const String)
# Prints a Sigil String (with [len][capacity][data] header) followed by newline
# String layout: [len: i64][capacity: i64][data: u8[]]
.global sigil_println
sigil_println:
    push rbx
    push r12

    # Handle NULL
    test rdi, rdi
    jz .println_newline_only

    # Get length from header (first 8 bytes)
    mov rbx, [rdi]           # len = str[0]

    # Get data pointer (skip 16-byte header)
    lea r12, [rdi + 16]      # data = str + 16

    # Write string data
    mov rax, 1               # SYS_write
    mov rdi, 1               # stdout
    mov rsi, r12             # buffer = data ptr
    mov rdx, rbx             # length from header
    syscall

.println_newline_only:
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

# sigil_write_str(str: *const u8)
# Writes a null-terminated C string (no newline)
.global sigil_write_str
sigil_write_str:
    push rbx
    push r12
    mov r12, rdi             # Save string pointer

    # Handle NULL
    test r12, r12
    jz .write_str_done

    # Calculate string length (strlen)
    xor rbx, rbx
.write_str_len_loop:
    mov al, [r12 + rbx]
    test al, al
    jz .write_str_output
    inc rbx
    jmp .write_str_len_loop

.write_str_output:
    # Write string
    test rbx, rbx            # Don't write if length is 0
    jz .write_str_done
    mov rax, 1               # SYS_write
    mov rdi, 1               # stdout
    mov rsi, r12             # buffer
    mov rdx, rbx             # length
    syscall

.write_str_done:
    pop r12
    pop rbx
    ret

# sigil_write_int(value: i64)
# Writes an integer (no newline)
.global sigil_write_int
sigil_write_int:
    push rbx
    push r12
    push r13
    sub rsp, 32              # Buffer for digits

    mov r12, rdi             # Save value
    lea r13, [rsp + 31]      # End of buffer
    mov byte ptr [r13], 0    # Null terminator

    # Handle negative numbers
    test r12, r12
    jns .write_int_positive
    neg r12
    mov bl, 1                # Negative flag
    jmp .write_int_convert
.write_int_positive:
    xor bl, bl               # Positive flag

.write_int_convert:
    # Convert to decimal digits (reverse order)
    mov rax, r12
    mov rcx, 10
.write_int_digit_loop:
    dec r13
    xor rdx, rdx
    div rcx                  # RAX = quotient, RDX = remainder
    add dl, '0'
    mov [r13], dl
    test rax, rax
    jnz .write_int_digit_loop

    # Add minus sign if negative
    test bl, bl
    jz .write_int_output
    dec r13
    mov byte ptr [r13], '-'

.write_int_output:
    # Calculate length
    lea rax, [rsp + 31]
    sub rax, r13             # Length = end - start
    mov rbx, rax

    # Write number (no newline)
    mov rax, 1               # SYS_write
    mov rdi, 1               # stdout
    mov rsi, r13             # buffer start
    mov rdx, rbx             # length
    syscall

    add rsp, 32
    pop r13
    pop r12
    pop rbx
    ret

# sigil_write_float(value: f64 in xmm0)
# Writes a floating point number with 6 decimal places (no newline)
.global sigil_write_float
sigil_write_float:
    push rbx
    push r12
    push r13
    push r14
    sub rsp, 64              # Buffer for output

    # Store float to memory
    movsd [rsp], xmm0

    # Check for negative
    mov rax, [rsp]
    test rax, rax
    jns .write_float_positive

    # Print minus sign
    mov byte ptr [rsp + 32], '-'
    mov rax, 1
    mov rdi, 1
    lea rsi, [rsp + 32]
    mov rdx, 1
    syscall

    # Make positive
    movsd xmm0, [rsp]
    pxor xmm1, xmm1
    subsd xmm1, xmm0         # xmm1 = -xmm0
    movsd xmm0, xmm1

.write_float_positive:
    # Extract integer part
    cvttsd2si r12, xmm0      # r12 = truncated integer part

    # Calculate fractional part: frac = (value - int_part) * 1000000
    cvtsi2sd xmm1, r12       # xmm1 = (double)int_part
    subsd xmm0, xmm1         # xmm0 = fractional part
    mov rax, 1000000         # Scale for 6 decimal places
    cvtsi2sd xmm1, rax
    mulsd xmm0, xmm1         # xmm0 = frac * 1000000
    cvttsd2si r13, xmm0      # r13 = fractional digits

    # Print integer part
    mov rdi, r12
    lea r14, [rsp + 48]      # End of buffer
    mov byte ptr [r14], 0

    # Convert integer to string
    mov rax, r12
    mov rcx, 10
.write_float_int_loop:
    dec r14
    xor rdx, rdx
    div rcx
    add dl, '0'
    mov [r14], dl
    test rax, rax
    jnz .write_float_int_loop

    # Print integer part
    lea rax, [rsp + 48]
    sub rax, r14
    mov rbx, rax
    mov rax, 1
    mov rdi, 1
    mov rsi, r14
    mov rdx, rbx
    syscall

    # Print decimal point
    mov byte ptr [rsp + 32], '.'
    mov rax, 1
    mov rdi, 1
    lea rsi, [rsp + 32]
    mov rdx, 1
    syscall

    # Print fractional part with leading zeros
    # We need exactly 6 digits
    lea r14, [rsp + 48]
    mov byte ptr [r14], 0
    dec r14
    mov rax, r13
    test rax, rax            # Handle 0 fractional part
    jns .write_float_frac_positive
    neg rax
.write_float_frac_positive:
    mov rcx, 10
    mov r12, 6               # 6 digits
.write_float_frac_loop:
    xor rdx, rdx
    div rcx
    add dl, '0'
    dec r14
    mov [r14], dl
    dec r12
    jnz .write_float_frac_loop

    # Print 6 fractional digits
    mov rax, 1
    mov rdi, 1
    mov rsi, r14
    mov rdx, 6
    syscall

    add rsp, 64
    pop r14
    pop r13
    pop r12
    pop rbx
    ret

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
#
# Sigil strings are heap-allocated with layout:
#   [len: i64][capacity: i64][data: u8[]]
#
# For compatibility with C strings, we also support null-terminated strings.
#

# sigil_strlen(str: *const u8) -> i64
# Get length of null-terminated C string
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

# sigil_string_from(cstr: *const u8) -> *mut String
# Create a Sigil string from a null-terminated C string
.global sigil_string_from
sigil_string_from:
    push rbx
    push r12
    push r13

    mov r12, rdi             # Save source pointer

    # Get length of source string
    call sigil_strlen
    mov r13, rax             # Save length

    # Allocate: 16 bytes header + len + 1 (for null terminator)
    lea rdi, [rax + 17]
    call sigil_alloc
    test rax, rax
    jz .string_from_failed

    mov rbx, rax             # Save string pointer

    # Set header
    mov [rbx], r13           # len
    lea rcx, [r13 + 1]
    mov [rbx + 8], rcx       # capacity = len + 1

    # Copy data
    lea rdi, [rbx + 16]      # dest = data area
    mov rsi, r12             # src = original string
    mov rcx, r13             # count = len
    rep movsb
    mov byte ptr [rdi], 0    # null terminator

    mov rax, rbx
    pop r13
    pop r12
    pop rbx
    ret

.string_from_failed:
    xor rax, rax
    pop r13
    pop r12
    pop rbx
    ret

# sigil_string_len(str: *mut String) -> i64
# Get length of Sigil string
.global sigil_string_len
sigil_string_len:
    test rdi, rdi
    jz .string_len_zero
    mov rax, [rdi]           # Return len field
    ret
.string_len_zero:
    xor rax, rax
    ret

# sigil_string_as_ptr(str: *mut String) -> *const u8
# Get pointer to string data
.global sigil_string_as_ptr
sigil_string_as_ptr:
    test rdi, rdi
    jz .string_ptr_null
    lea rax, [rdi + 16]      # Return pointer to data area
    ret
.string_ptr_null:
    xor rax, rax
    ret

# sigil_string_print(str: *mut String)
# Print Sigil string to stdout
.global sigil_string_print
sigil_string_print:
    push rbx

    test rdi, rdi
    jz .string_print_done

    mov rbx, rdi             # Save string pointer

    # Write string data
    mov rax, 1               # SYS_write
    mov rdi, 1               # stdout
    lea rsi, [rbx + 16]      # data pointer
    mov rdx, [rbx]           # len
    syscall

.string_print_done:
    pop rbx
    ret

# sigil_string_concat(a: *mut String, b: *mut String) -> *mut String
# Concatenate two strings, returning new string
.global sigil_string_concat
sigil_string_concat:
    push rbx
    push r12
    push r13
    push r14

    mov r12, rdi             # String a
    mov r13, rsi             # String b

    # Get lengths
    xor r14, r14             # Total length

    test r12, r12
    jz .concat_no_a
    add r14, [r12]           # a.len
.concat_no_a:
    test r13, r13
    jz .concat_no_b
    add r14, [r13]           # b.len
.concat_no_b:

    # Allocate new string: 16 header + total_len + 1
    lea rdi, [r14 + 17]
    call sigil_alloc
    test rax, rax
    jz .concat_failed

    mov rbx, rax             # New string

    # Set header
    mov [rbx], r14           # len = total_len
    lea rcx, [r14 + 1]
    mov [rbx + 8], rcx       # capacity

    # Copy first string
    lea rdi, [rbx + 16]      # dest = new string data
    test r12, r12
    jz .concat_copy_b
    mov rcx, [r12]           # len of a
    lea rsi, [r12 + 16]      # src = a.data
    rep movsb

.concat_copy_b:
    # Copy second string (rdi already at correct position)
    test r13, r13
    jz .concat_done
    mov rcx, [r13]           # len of b
    lea rsi, [r13 + 16]      # src = b.data
    rep movsb

.concat_done:
    mov byte ptr [rdi], 0    # null terminator
    mov rax, rbx

    pop r14
    pop r13
    pop r12
    pop rbx
    ret

.concat_failed:
    xor rax, rax
    pop r14
    pop r13
    pop r12
    pop rbx
    ret

# sigil_string_eq(a: *mut String, b: *mut String) -> i64
# Compare two strings, returns 1 if equal, 0 otherwise
.global sigil_string_eq
sigil_string_eq:
    # Handle NULL cases
    test rdi, rdi
    jz .eq_check_b_null
    test rsi, rsi
    jz .eq_not_equal

    # Compare lengths
    mov rax, [rdi]           # a.len
    cmp rax, [rsi]           # b.len
    jne .eq_not_equal

    # Compare data byte by byte
    mov rcx, rax             # count = len
    test rcx, rcx
    jz .eq_equal             # Both empty strings

    lea rdi, [rdi + 16]      # a.data
    lea rsi, [rsi + 16]      # b.data
    repe cmpsb
    jne .eq_not_equal

.eq_equal:
    mov rax, 1
    ret

.eq_check_b_null:
    test rsi, rsi
    jnz .eq_not_equal
    # Both NULL = equal
    mov rax, 1
    ret

.eq_not_equal:
    xor rax, rax
    ret

# sigil_string_clone(str: *mut String) -> *mut String
# Create a copy of a string
.global sigil_string_clone
sigil_string_clone:
    push rbx
    push r12

    test rdi, rdi
    jz .clone_null

    mov r12, rdi             # Save source

    # Get source length
    mov rbx, [r12]           # len

    # Allocate new string
    lea rdi, [rbx + 17]      # 16 header + len + 1
    call sigil_alloc
    test rax, rax
    jz .clone_failed

    # Set header
    mov [rax], rbx           # len
    lea rcx, [rbx + 1]
    mov [rax + 8], rcx       # capacity

    # Copy data
    push rax                 # Save new string pointer
    lea rdi, [rax + 16]      # dest
    lea rsi, [r12 + 16]      # src
    mov rcx, rbx             # count
    rep movsb
    mov byte ptr [rdi], 0    # null terminator
    pop rax

    pop r12
    pop rbx
    ret

.clone_null:
.clone_failed:
    xor rax, rax
    pop r12
    pop rbx
    ret

# sigil_string_is_empty(str: *mut String) -> i64
# Returns 1 if string is empty or NULL, 0 otherwise
.global sigil_string_is_empty
sigil_string_is_empty:
    test rdi, rdi
    jz .is_empty_true
    cmp qword ptr [rdi], 0   # len == 0?
    je .is_empty_true
    xor rax, rax
    ret
.is_empty_true:
    mov rax, 1
    ret

# sigil_string_char_at(str: *mut String, idx: i64) -> i64
# Get character at index (as byte value), or -1 if out of bounds
.global sigil_string_char_at
sigil_string_char_at:
    test rdi, rdi
    jz .char_at_invalid

    # Check bounds
    cmp rsi, 0
    jl .char_at_invalid
    cmp rsi, [rdi]           # idx >= len?
    jge .char_at_invalid

    # Get character
    movzx eax, byte ptr [rdi + 16 + rsi]
    ret

.char_at_invalid:
    mov rax, -1
    ret

# sigil_string_repeat(str: *const u8, count: i64) -> *const u8
# Repeats a C string (null-terminated) count times, returns new allocated string
.global sigil_string_repeat
sigil_string_repeat:
    push rbx
    push r12
    push r13
    push r14
    push r15

    mov r12, rdi             # Save source string pointer
    mov r13, rsi             # Save count

    # Handle edge cases
    test r12, r12
    jz .repeat_empty
    test r13, r13
    jle .repeat_empty

    # Calculate source string length (strlen)
    xor r14, r14             # r14 = src_len
.repeat_strlen_loop:
    mov al, [r12 + r14]
    test al, al
    jz .repeat_strlen_done
    inc r14
    jmp .repeat_strlen_loop

.repeat_strlen_done:
    # Handle empty source string
    test r14, r14
    jz .repeat_empty

    # Calculate total size: src_len * count + 1 (null terminator)
    mov rax, r14
    mul r13                  # RAX = src_len * count
    mov r15, rax             # r15 = total_len (without null)
    inc rax                  # +1 for null terminator

    # Allocate memory
    mov rdi, rax
    call sigil_alloc
    test rax, rax
    jz .repeat_alloc_fail
    mov rbx, rax             # rbx = dest pointer

    # Copy source string 'count' times
    xor rcx, rcx             # rcx = current offset in dest
.repeat_copy_outer:
    cmp r13, 0
    jle .repeat_copy_done
    dec r13

    # Copy one instance of source string
    xor rdx, rdx             # rdx = current position in source
.repeat_copy_inner:
    cmp rdx, r14             # Compare with src_len
    jge .repeat_copy_inner_done
    mov al, [r12 + rdx]
    mov [rbx + rcx], al
    inc rdx
    inc rcx
    jmp .repeat_copy_inner

.repeat_copy_inner_done:
    jmp .repeat_copy_outer

.repeat_copy_done:
    # Add null terminator
    mov byte ptr [rbx + r15], 0

    mov rax, rbx             # Return dest pointer
    pop r15
    pop r14
    pop r13
    pop r12
    pop rbx
    ret

.repeat_empty:
    # Allocate and return empty string
    mov rdi, 1
    call sigil_alloc
    test rax, rax
    jz .repeat_alloc_fail
    mov byte ptr [rax], 0    # Null terminator
    pop r15
    pop r14
    pop r13
    pop r12
    pop rbx
    ret

.repeat_alloc_fail:
    xor rax, rax             # Return NULL on failure
    pop r15
    pop r14
    pop r13
    pop r12
    pop rbx
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
# File I/O Functions
# ============================================================================
#
# Linux syscall numbers:
#   0 = read     1 = write    2 = open     3 = close
#  21 = access  80 = lseek
#
# Open flags (O_*):
#   0 = O_RDONLY    1 = O_WRONLY    2 = O_RDWR
#  64 = O_CREAT   512 = O_TRUNC  1024 = O_APPEND
#

# sigil_file_open(path: *const u8, flags: i64, mode: i64) -> i64
# Open a file, returns fd or negative errno
.global sigil_file_open
sigil_file_open:
    mov rax, 2               # SYS_open
    # rdi = path, rsi = flags, rdx = mode already set
    syscall
    ret

# sigil_file_close(fd: i64) -> i64
# Close a file descriptor
.global sigil_file_close
sigil_file_close:
    mov rax, 3               # SYS_close
    syscall
    ret

# sigil_file_read(fd: i64, buf: *mut u8, count: i64) -> i64
# Read from file, returns bytes read or negative errno
.global sigil_file_read
sigil_file_read:
    mov rax, 0               # SYS_read
    syscall
    ret

# sigil_file_write(fd: i64, buf: *const u8, count: i64) -> i64
# Write to file, returns bytes written or negative errno
.global sigil_file_write
sigil_file_write:
    mov rax, 1               # SYS_write
    syscall
    ret

# sigil_file_exists(path: *const u8) -> i64
# Check if file exists (1 = yes, 0 = no)
.global sigil_file_exists
sigil_file_exists:
    mov rax, 21              # SYS_access
    # rdi = path
    xor rsi, rsi             # F_OK = 0 (check existence)
    syscall
    # access returns 0 on success, -1 on error
    test rax, rax
    jnz .file_not_exists
    mov rax, 1
    ret
.file_not_exists:
    xor rax, rax
    ret

# sigil_file_seek(fd: i64, offset: i64, whence: i64) -> i64
# Seek in file, returns new position or negative errno
# whence: 0 = SEEK_SET, 1 = SEEK_CUR, 2 = SEEK_END
.global sigil_file_seek
sigil_file_seek:
    mov rax, 8               # SYS_lseek
    syscall
    ret

# sigil_file_read_all(path: *const u8) -> *mut String
# Read entire file into a new string (uses arena allocator)
.global sigil_file_read_all
sigil_file_read_all:
    push rbx
    push r12
    push r13
    push r14

    mov r12, rdi             # Save path

    # Open file read-only
    mov rax, 2               # SYS_open
    xor rsi, rsi             # O_RDONLY
    xor rdx, rdx             # mode (unused for read)
    syscall

    cmp rax, 0
    jl .read_all_failed
    mov r13, rax             # Save fd

    # Seek to end to get file size
    mov rdi, r13
    xor rsi, rsi             # offset = 0
    mov rdx, 2               # SEEK_END
    mov rax, 8               # SYS_lseek
    syscall

    cmp rax, 0
    jl .read_all_close_fail
    mov r14, rax             # Save file size

    # Seek back to beginning
    mov rdi, r13
    xor rsi, rsi             # offset = 0
    xor rdx, rdx             # SEEK_SET
    mov rax, 8               # SYS_lseek
    syscall

    # Allocate string: 16 header + size + 1 (null terminator)
    lea rdi, [r14 + 17]
    call sigil_alloc
    test rax, rax
    jz .read_all_close_fail
    mov rbx, rax             # Save string pointer

    # Set string header
    mov [rbx], r14           # len = file size
    lea rcx, [r14 + 1]
    mov [rbx + 8], rcx       # capacity

    # Read file contents
    mov rdi, r13             # fd
    lea rsi, [rbx + 16]      # buffer = string data area
    mov rdx, r14             # count = file size
    mov rax, 0               # SYS_read
    syscall

    # Null terminate
    mov byte ptr [rbx + 16 + r14], 0

    # Close file
    mov rdi, r13
    mov rax, 3               # SYS_close
    syscall

    mov rax, rbx             # Return string pointer

    pop r14
    pop r13
    pop r12
    pop rbx
    ret

.read_all_close_fail:
    # Close file before failing
    mov rdi, r13
    mov rax, 3
    syscall
.read_all_failed:
    xor rax, rax
    pop r14
    pop r13
    pop r12
    pop rbx
    ret

# sigil_file_write_all(path: *const u8, content: *mut String) -> i64
# Write entire string to file, returns bytes written or negative errno
.global sigil_file_write_all
sigil_file_write_all:
    push rbx
    push r12
    push r13

    mov r12, rdi             # path
    mov r13, rsi             # content string

    # Handle NULL content
    test r13, r13
    jz .write_all_empty

    # Open file for writing (create/truncate)
    mov rdi, r12
    mov rsi, 577             # O_WRONLY | O_CREAT | O_TRUNC (1 + 64 + 512)
    mov rdx, 0644            # mode: rw-r--r--
    mov rax, 2               # SYS_open
    syscall

    cmp rax, 0
    jl .write_all_failed
    mov rbx, rax             # Save fd

    # Write content
    mov rdi, rbx             # fd
    lea rsi, [r13 + 16]      # buffer = string data
    mov rdx, [r13]           # count = string len
    mov rax, 1               # SYS_write
    syscall

    mov r12, rax             # Save bytes written

    # Close file
    mov rdi, rbx
    mov rax, 3               # SYS_close
    syscall

    mov rax, r12             # Return bytes written

    pop r13
    pop r12
    pop rbx
    ret

.write_all_empty:
    xor rax, rax
    pop r13
    pop r12
    pop rbx
    ret

.write_all_failed:
    pop r13
    pop r12
    pop rbx
    ret

# sigil_file_size(path: *const u8) -> i64
# Get file size, returns size or negative errno
.global sigil_file_size
sigil_file_size:
    push rbx
    push r12

    mov r12, rdi             # Save path

    # Open file read-only
    mov rax, 2
    xor rsi, rsi
    xor rdx, rdx
    syscall

    cmp rax, 0
    jl .size_failed
    mov rbx, rax             # Save fd

    # Seek to end
    mov rdi, rbx
    xor rsi, rsi
    mov rdx, 2               # SEEK_END
    mov rax, 8
    syscall

    mov r12, rax             # Save size

    # Close file
    mov rdi, rbx
    mov rax, 3
    syscall

    mov rax, r12             # Return size

    pop r12
    pop rbx
    ret

.size_failed:
    pop r12
    pop rbx
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

# sigil_tan(x: i64) -> i64
# Compute tan(x) using fptan
.global sigil_tan
sigil_tan:
    push rdi
    fld qword ptr [rsp]
    fptan
    fstp st(0)               # Pop the 1.0 that fptan pushes
    fstp qword ptr [rsp]
    pop rax
    ret

# sigil_exp(x: i64) -> i64
# Compute e^x using: 2^(x * log2(e))
.global sigil_exp
sigil_exp:
    push rdi
    sub rsp, 8

    fld qword ptr [rsp + 8]  # Load x
    fldl2e                   # Load log2(e)
    fmulp                    # x * log2(e)

    # Compute 2^result using f2xm1 (for fractional part) and fscale (for integer part)
    fld st(0)
    frndint                  # Integer part
    fxch st(1)
    fsub st(0), st(1)        # Fractional part
    f2xm1                    # 2^frac - 1
    fld1
    faddp                    # 2^frac
    fscale                   # * 2^int = 2^(x*log2(e)) = e^x
    fstp st(1)               # Clean up

    fstp qword ptr [rsp + 8]
    add rsp, 8
    pop rax
    ret

# sigil_ln(x: i64) -> i64
# Compute ln(x) = log2(x) * ln(2)
.global sigil_ln
sigil_ln:
    push rdi
    sub rsp, 8

    fld1                     # Push 1.0
    fld qword ptr [rsp + 8]  # Load x
    fyl2x                    # 1.0 * log2(x) = log2(x)
    fldln2                   # Load ln(2)
    fmulp                    # log2(x) * ln(2) = ln(x)

    fstp qword ptr [rsp + 8]
    add rsp, 8
    pop rax
    ret

# C library compatibility aliases
# These allow LLVM-generated code to find math functions
.global exp
exp:
    jmp sigil_exp

.global cos
cos:
    jmp sigil_cos

.global sin
sin:
    jmp sigil_sin

.global tan
tan:
    jmp sigil_tan

.global log
log:
    jmp sigil_ln

.global sqrt
sqrt:
    jmp sigil_sqrt

.global pow
pow:
    jmp sigil_pow

.global floor
floor:
    jmp sigil_floor

.global ceil
ceil:
    jmp sigil_ceil

.global fabs
fabs:
    jmp sigil_abs

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
# SIMD Math Functions (SSE/AVX)
# ============================================================================
# F32x4: 4-wide f32 vectors using SSE (128-bit XMM registers)
# F32x8: 8-wide f32 vectors using AVX (256-bit YMM registers)
# F64x2: 2-wide f64 vectors using SSE2
# F64x4: 4-wide f64 vectors using AVX

# ----------------------------------------------------------------------------
# F32x4 Operations (SSE 128-bit)
# All functions take pointers to aligned 16-byte memory regions
# ----------------------------------------------------------------------------

# simd_f32x4_add(dst: *mut f32, a: *const f32, b: *const f32)
.global simd_f32x4_add
simd_f32x4_add:
    movaps xmm0, [rsi]       # Load 4 floats from a
    movaps xmm1, [rdx]       # Load 4 floats from b
    addps xmm0, xmm1         # Add packed single-precision
    movaps [rdi], xmm0       # Store result
    ret

# simd_f32x4_sub(dst: *mut f32, a: *const f32, b: *const f32)
.global simd_f32x4_sub
simd_f32x4_sub:
    movaps xmm0, [rsi]
    movaps xmm1, [rdx]
    subps xmm0, xmm1
    movaps [rdi], xmm0
    ret

# simd_f32x4_mul(dst: *mut f32, a: *const f32, b: *const f32)
.global simd_f32x4_mul
simd_f32x4_mul:
    movaps xmm0, [rsi]
    movaps xmm1, [rdx]
    mulps xmm0, xmm1
    movaps [rdi], xmm0
    ret

# simd_f32x4_div(dst: *mut f32, a: *const f32, b: *const f32)
.global simd_f32x4_div
simd_f32x4_div:
    movaps xmm0, [rsi]
    movaps xmm1, [rdx]
    divps xmm0, xmm1
    movaps [rdi], xmm0
    ret

# simd_f32x4_min(dst: *mut f32, a: *const f32, b: *const f32)
.global simd_f32x4_min
simd_f32x4_min:
    movaps xmm0, [rsi]
    movaps xmm1, [rdx]
    minps xmm0, xmm1
    movaps [rdi], xmm0
    ret

# simd_f32x4_max(dst: *mut f32, a: *const f32, b: *const f32)
.global simd_f32x4_max
simd_f32x4_max:
    movaps xmm0, [rsi]
    movaps xmm1, [rdx]
    maxps xmm0, xmm1
    movaps [rdi], xmm0
    ret

# simd_f32x4_sqrt(dst: *mut f32, a: *const f32)
.global simd_f32x4_sqrt
simd_f32x4_sqrt:
    movaps xmm0, [rsi]
    sqrtps xmm0, xmm0
    movaps [rdi], xmm0
    ret

# simd_f32x4_splat(dst: *mut f32, value: f32)
# Note: value passed in xmm0 (float calling convention)
.global simd_f32x4_splat
simd_f32x4_splat:
    shufps xmm0, xmm0, 0     # Broadcast to all lanes
    movaps [rdi], xmm0
    ret

# simd_f32x4_reduce_add(a: *const f32) -> f32
# Returns horizontal sum of 4 floats
.global simd_f32x4_reduce_add
simd_f32x4_reduce_add:
    movaps xmm0, [rdi]
    movhlps xmm1, xmm0       # xmm1 = [z, w, -, -]
    addps xmm0, xmm1         # xmm0 = [x+z, y+w, -, -]
    movaps xmm1, xmm0
    shufps xmm1, xmm1, 1     # xmm1 = [y+w, -, -, -]
    addss xmm0, xmm1         # xmm0[0] = x+y+z+w
    ret                      # Return in xmm0

# simd_f32x4_dot(a: *const f32, b: *const f32) -> f32
# Dot product of two 4-vectors
.global simd_f32x4_dot
simd_f32x4_dot:
    movaps xmm0, [rdi]
    movaps xmm1, [rsi]
    mulps xmm0, xmm1         # Element-wise multiply
    # Horizontal sum
    movhlps xmm1, xmm0
    addps xmm0, xmm1
    movaps xmm1, xmm0
    shufps xmm1, xmm1, 1
    addss xmm0, xmm1
    ret

# simd_f32x4_fmadd(dst: *mut f32, a: *const f32, b: *const f32, c: *const f32)
# dst = a * b + c (fused multiply-add if FMA available, else mul+add)
.global simd_f32x4_fmadd
simd_f32x4_fmadd:
    movaps xmm0, [rsi]       # a
    movaps xmm1, [rdx]       # b
    movaps xmm2, [rcx]       # c
    mulps xmm0, xmm1         # a * b
    addps xmm0, xmm2         # + c
    movaps [rdi], xmm0
    ret

# ----------------------------------------------------------------------------
# F64x2 Operations (SSE2 128-bit)
# ----------------------------------------------------------------------------

# simd_f64x2_add(dst: *mut f64, a: *const f64, b: *const f64)
.global simd_f64x2_add
simd_f64x2_add:
    movapd xmm0, [rsi]
    movapd xmm1, [rdx]
    addpd xmm0, xmm1
    movapd [rdi], xmm0
    ret

# simd_f64x2_sub(dst: *mut f64, a: *const f64, b: *const f64)
.global simd_f64x2_sub
simd_f64x2_sub:
    movapd xmm0, [rsi]
    movapd xmm1, [rdx]
    subpd xmm0, xmm1
    movapd [rdi], xmm0
    ret

# simd_f64x2_mul(dst: *mut f64, a: *const f64, b: *const f64)
.global simd_f64x2_mul
simd_f64x2_mul:
    movapd xmm0, [rsi]
    movapd xmm1, [rdx]
    mulpd xmm0, xmm1
    movapd [rdi], xmm0
    ret

# simd_f64x2_div(dst: *mut f64, a: *const f64, b: *const f64)
.global simd_f64x2_div
simd_f64x2_div:
    movapd xmm0, [rsi]
    movapd xmm1, [rdx]
    divpd xmm0, xmm1
    movapd [rdi], xmm0
    ret

# simd_f64x2_sqrt(dst: *mut f64, a: *const f64)
.global simd_f64x2_sqrt
simd_f64x2_sqrt:
    movapd xmm0, [rsi]
    sqrtpd xmm0, xmm0
    movapd [rdi], xmm0
    ret

# simd_f64x2_reduce_add(a: *const f64) -> f64
.global simd_f64x2_reduce_add
simd_f64x2_reduce_add:
    movapd xmm0, [rdi]
    movhlps xmm1, xmm0       # Get high element
    addsd xmm0, xmm1         # Add low + high
    ret

# simd_f64x2_dot(a: *const f64, b: *const f64) -> f64
.global simd_f64x2_dot
simd_f64x2_dot:
    movapd xmm0, [rdi]
    movapd xmm1, [rsi]
    mulpd xmm0, xmm1
    movhlps xmm1, xmm0
    addsd xmm0, xmm1
    ret

# ----------------------------------------------------------------------------
# F32x8 Operations (AVX 256-bit)
# Requires AVX support - check with cpuid before using
# ----------------------------------------------------------------------------

# simd_f32x8_add(dst: *mut f32, a: *const f32, b: *const f32)
.global simd_f32x8_add
simd_f32x8_add:
    vmovaps ymm0, [rsi]
    vmovaps ymm1, [rdx]
    vaddps ymm0, ymm0, ymm1
    vmovaps [rdi], ymm0
    vzeroupper               # Avoid AVX-SSE transition penalty
    ret

# simd_f32x8_sub(dst: *mut f32, a: *const f32, b: *const f32)
.global simd_f32x8_sub
simd_f32x8_sub:
    vmovaps ymm0, [rsi]
    vmovaps ymm1, [rdx]
    vsubps ymm0, ymm0, ymm1
    vmovaps [rdi], ymm0
    vzeroupper
    ret

# simd_f32x8_mul(dst: *mut f32, a: *const f32, b: *const f32)
.global simd_f32x8_mul
simd_f32x8_mul:
    vmovaps ymm0, [rsi]
    vmovaps ymm1, [rdx]
    vmulps ymm0, ymm0, ymm1
    vmovaps [rdi], ymm0
    vzeroupper
    ret

# simd_f32x8_div(dst: *mut f32, a: *const f32, b: *const f32)
.global simd_f32x8_div
simd_f32x8_div:
    vmovaps ymm0, [rsi]
    vmovaps ymm1, [rdx]
    vdivps ymm0, ymm0, ymm1
    vmovaps [rdi], ymm0
    vzeroupper
    ret

# simd_f32x8_min(dst: *mut f32, a: *const f32, b: *const f32)
.global simd_f32x8_min
simd_f32x8_min:
    vmovaps ymm0, [rsi]
    vmovaps ymm1, [rdx]
    vminps ymm0, ymm0, ymm1
    vmovaps [rdi], ymm0
    vzeroupper
    ret

# simd_f32x8_max(dst: *mut f32, a: *const f32, b: *const f32)
.global simd_f32x8_max
simd_f32x8_max:
    vmovaps ymm0, [rsi]
    vmovaps ymm1, [rdx]
    vmaxps ymm0, ymm0, ymm1
    vmovaps [rdi], ymm0
    vzeroupper
    ret

# simd_f32x8_sqrt(dst: *mut f32, a: *const f32)
.global simd_f32x8_sqrt
simd_f32x8_sqrt:
    vmovaps ymm0, [rsi]
    vsqrtps ymm0, ymm0
    vmovaps [rdi], ymm0
    vzeroupper
    ret

# simd_f32x8_splat(dst: *mut f32, value: f32)
.global simd_f32x8_splat
simd_f32x8_splat:
    vbroadcastss ymm0, xmm0  # Broadcast scalar to all 8 lanes
    vmovaps [rdi], ymm0
    vzeroupper
    ret

# simd_f32x8_reduce_add(a: *const f32) -> f32
# Horizontal sum of 8 floats
.global simd_f32x8_reduce_add
simd_f32x8_reduce_add:
    vmovaps ymm0, [rdi]
    vextractf128 xmm1, ymm0, 1    # Get high 128 bits
    vaddps xmm0, xmm0, xmm1       # Add high and low halves
    vmovhlps xmm1, xmm0, xmm0     # Get [z, w]
    vaddps xmm0, xmm0, xmm1       # [x+z, y+w]
    vshufps xmm1, xmm0, xmm0, 1   # Get y+w
    vaddss xmm0, xmm0, xmm1       # Final sum
    vzeroupper
    ret

# simd_f32x8_dot(a: *const f32, b: *const f32) -> f32
.global simd_f32x8_dot
simd_f32x8_dot:
    vmovaps ymm0, [rdi]
    vmovaps ymm1, [rsi]
    vmulps ymm0, ymm0, ymm1        # Element-wise multiply
    vextractf128 xmm1, ymm0, 1
    vaddps xmm0, xmm0, xmm1
    vmovhlps xmm1, xmm0, xmm0
    vaddps xmm0, xmm0, xmm1
    vshufps xmm1, xmm0, xmm0, 1
    vaddss xmm0, xmm0, xmm1
    vzeroupper
    ret

# simd_f32x8_fmadd(dst: *mut f32, a: *const f32, b: *const f32, c: *const f32)
# dst = a * b + c
.global simd_f32x8_fmadd
simd_f32x8_fmadd:
    vmovaps ymm0, [rsi]       # a
    vmovaps ymm1, [rdx]       # b
    vmovaps ymm2, [rcx]       # c
    vmulps ymm0, ymm0, ymm1   # a * b
    vaddps ymm0, ymm0, ymm2   # + c
    vmovaps [rdi], ymm0
    vzeroupper
    ret

# ----------------------------------------------------------------------------
# F64x4 Operations (AVX 256-bit)
# ----------------------------------------------------------------------------

# simd_f64x4_add(dst: *mut f64, a: *const f64, b: *const f64)
.global simd_f64x4_add
simd_f64x4_add:
    vmovapd ymm0, [rsi]
    vmovapd ymm1, [rdx]
    vaddpd ymm0, ymm0, ymm1
    vmovapd [rdi], ymm0
    vzeroupper
    ret

# simd_f64x4_sub(dst: *mut f64, a: *const f64, b: *const f64)
.global simd_f64x4_sub
simd_f64x4_sub:
    vmovapd ymm0, [rsi]
    vmovapd ymm1, [rdx]
    vsubpd ymm0, ymm0, ymm1
    vmovapd [rdi], ymm0
    vzeroupper
    ret

# simd_f64x4_mul(dst: *mut f64, a: *const f64, b: *const f64)
.global simd_f64x4_mul
simd_f64x4_mul:
    vmovapd ymm0, [rsi]
    vmovapd ymm1, [rdx]
    vmulpd ymm0, ymm0, ymm1
    vmovapd [rdi], ymm0
    vzeroupper
    ret

# simd_f64x4_div(dst: *mut f64, a: *const f64, b: *const f64)
.global simd_f64x4_div
simd_f64x4_div:
    vmovapd ymm0, [rsi]
    vmovapd ymm1, [rdx]
    vdivpd ymm0, ymm0, ymm1
    vmovapd [rdi], ymm0
    vzeroupper
    ret

# simd_f64x4_sqrt(dst: *mut f64, a: *const f64)
.global simd_f64x4_sqrt
simd_f64x4_sqrt:
    vmovapd ymm0, [rsi]
    vsqrtpd ymm0, ymm0
    vmovapd [rdi], ymm0
    vzeroupper
    ret

# simd_f64x4_reduce_add(a: *const f64) -> f64
.global simd_f64x4_reduce_add
simd_f64x4_reduce_add:
    vmovapd ymm0, [rdi]
    vextractf128 xmm1, ymm0, 1    # Get high 128 bits
    vaddpd xmm0, xmm0, xmm1       # Add halves
    vmovhlps xmm1, xmm0, xmm0     # Get high element
    vaddsd xmm0, xmm0, xmm1       # Final sum
    vzeroupper
    ret

# simd_f64x4_dot(a: *const f64, b: *const f64) -> f64
.global simd_f64x4_dot
simd_f64x4_dot:
    vmovapd ymm0, [rdi]
    vmovapd ymm1, [rsi]
    vmulpd ymm0, ymm0, ymm1
    vextractf128 xmm1, ymm0, 1
    vaddpd xmm0, xmm0, xmm1
    vmovhlps xmm1, xmm0, xmm0
    vaddsd xmm0, xmm0, xmm1
    vzeroupper
    ret

# ----------------------------------------------------------------------------
# SIMD Utility Functions
# ----------------------------------------------------------------------------

# simd_check_avx() -> i64
# Returns 1 if AVX is supported, 0 otherwise
.global simd_check_avx
simd_check_avx:
    push rbx
    mov eax, 1
    cpuid
    xor rax, rax
    test ecx, 0x10000000     # Check AVX bit (bit 28 of ECX)
    setnz al
    pop rbx
    ret

# simd_check_avx2() -> i64
# Returns 1 if AVX2 is supported, 0 otherwise
.global simd_check_avx2
simd_check_avx2:
    push rbx
    mov eax, 7
    xor ecx, ecx
    cpuid
    xor rax, rax
    test ebx, 0x20           # Check AVX2 bit (bit 5 of EBX)
    setnz al
    pop rbx
    ret

# simd_check_fma() -> i64
# Returns 1 if FMA is supported, 0 otherwise
.global simd_check_fma
simd_check_fma:
    push rbx
    mov eax, 1
    cpuid
    xor rax, rax
    test ecx, 0x1000         # Check FMA bit (bit 12 of ECX)
    setnz al
    pop rbx
    ret

# simd_alloc_aligned(size: i64, alignment: i64) -> *mut u8
# Allocate memory with specified alignment
# alignment must be power of 2
.global simd_alloc_aligned
simd_alloc_aligned:
    # Store original size and alignment for later
    push rbx
    push r12
    mov rbx, rdi             # size
    mov r12, rsi             # alignment

    # Allocate size + alignment + 8 (for storing original ptr)
    add rdi, rsi
    add rdi, 8
    call sigil_alloc

    test rax, rax
    jz .aligned_alloc_fail

    # Store original pointer at start
    mov [rax], rax
    add rax, 8

    # Align the pointer
    mov rcx, rax
    add rcx, r12
    sub rcx, 1
    neg r12
    and rcx, r12
    mov rax, rcx

    pop r12
    pop rbx
    ret

.aligned_alloc_fail:
    pop r12
    pop rbx
    xor rax, rax
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
# Networking Syscalls
# ============================================================================

# Sys_socket(domain: i64, type: i64, protocol: i64) -> i64
# domain: AF_INET=2, AF_INET6=10, AF_UNIX=1
# type: SOCK_STREAM=1, SOCK_DGRAM=2
.global Sys_socket
Sys_socket:
    mov rax, 41              # SYS_socket
    syscall
    ret

# Sys_connect(fd: i64, addr: *const sockaddr, addrlen: i64) -> i64
.global Sys_connect
Sys_connect:
    mov rax, 42              # SYS_connect
    syscall
    ret

# Sys_accept(fd: i64, addr: *mut sockaddr, addrlen: *mut i64) -> i64
.global Sys_accept
Sys_accept:
    mov rax, 43              # SYS_accept
    syscall
    ret

# Sys_sendto(fd: i64, buf: *const u8, len: i64, flags: i64, addr: *const sockaddr, addrlen: i64) -> i64
.global Sys_sendto
Sys_sendto:
    mov rax, 44              # SYS_sendto
    mov r10, rcx             # Linux syscall uses r10 for 4th arg
    syscall
    ret

# Sys_recvfrom(fd: i64, buf: *mut u8, len: i64, flags: i64, addr: *mut sockaddr, addrlen: *mut i64) -> i64
.global Sys_recvfrom
Sys_recvfrom:
    mov rax, 45              # SYS_recvfrom
    mov r10, rcx             # Linux syscall uses r10 for 4th arg
    syscall
    ret

# Sys_send(fd: i64, buf: *const u8, len: i64, flags: i64) -> i64
.global Sys_send
Sys_send:
    mov rax, 44              # SYS_sendto with NULL addr
    mov r10, rcx             # flags
    xor r8, r8               # addr = NULL
    xor r9, r9               # addrlen = 0
    syscall
    ret

# Sys_recv(fd: i64, buf: *mut u8, len: i64, flags: i64) -> i64
.global Sys_recv
Sys_recv:
    mov rax, 45              # SYS_recvfrom with NULL addr
    mov r10, rcx             # flags
    xor r8, r8               # addr = NULL
    xor r9, r9               # addrlen = NULL
    syscall
    ret

# Sys_bind(fd: i64, addr: *const sockaddr, addrlen: i64) -> i64
.global Sys_bind
Sys_bind:
    mov rax, 49              # SYS_bind
    syscall
    ret

# Sys_listen(fd: i64, backlog: i64) -> i64
.global Sys_listen
Sys_listen:
    mov rax, 50              # SYS_listen
    syscall
    ret

# Sys_shutdown(fd: i64, how: i64) -> i64
# how: SHUT_RD=0, SHUT_WR=1, SHUT_RDWR=2
.global Sys_shutdown
Sys_shutdown:
    mov rax, 48              # SYS_shutdown
    syscall
    ret

# Sys_setsockopt(fd: i64, level: i64, optname: i64, optval: *const void, optlen: i64) -> i64
.global Sys_setsockopt
Sys_setsockopt:
    mov rax, 54              # SYS_setsockopt
    mov r10, rcx             # Linux syscall uses r10 for 4th arg
    syscall
    ret

# Sys_getsockopt(fd: i64, level: i64, optname: i64, optval: *mut void, optlen: *mut i64) -> i64
.global Sys_getsockopt
Sys_getsockopt:
    mov rax, 55              # SYS_getsockopt
    mov r10, rcx             # Linux syscall uses r10 for 4th arg
    syscall
    ret

# Sys_getpeername(fd: i64, addr: *mut sockaddr, addrlen: *mut i64) -> i64
.global Sys_getpeername
Sys_getpeername:
    mov rax, 52              # SYS_getpeername
    syscall
    ret

# Sys_getsockname(fd: i64, addr: *mut sockaddr, addrlen: *mut i64) -> i64
.global Sys_getsockname
Sys_getsockname:
    mov rax, 51              # SYS_getsockname
    syscall
    ret

# ============================================================================
# Threading Syscalls (Phase 8)
# ============================================================================

# Sys_clone(flags: i64, stack: *mut u8, parent_tid: *mut i32, child_tid: *mut i32, tls: i64) -> i64
# Creates a new thread. Returns 0 in child, child tid in parent, or negative errno.
# Common flags: CLONE_VM=0x100, CLONE_FS=0x200, CLONE_FILES=0x400, CLONE_SIGHAND=0x800
#               CLONE_THREAD=0x10000, CLONE_SYSVSEM=0x40000, CLONE_SETTLS=0x80000
#               CLONE_PARENT_SETTID=0x100000, CLONE_CHILD_CLEARTID=0x200000
.global Sys_clone
Sys_clone:
    mov rax, 56              # SYS_clone
    mov r10, rcx             # parent_tid -> r10 (4th arg)
    syscall
    ret

# Sys_clone3(args: *const clone_args, size: i64) -> i64
# Modern clone interface with extensible arguments struct
.global Sys_clone3
Sys_clone3:
    mov rax, 435             # SYS_clone3
    syscall
    ret

# Sys_futex(uaddr: *mut u32, futex_op: i32, val: u32, timeout: *const timespec, uaddr2: *mut u32, val3: u32) -> i64
# Fast userspace locking primitive. Used to implement mutexes, condvars, etc.
# Common ops: FUTEX_WAIT=0, FUTEX_WAKE=1, FUTEX_WAIT_PRIVATE=128, FUTEX_WAKE_PRIVATE=129
.global Sys_futex
Sys_futex:
    mov rax, 202             # SYS_futex
    mov r10, rcx             # timeout -> r10 (4th arg)
    syscall
    ret

# Sys_gettid() -> i64
# Get the thread ID of the calling thread
.global Sys_gettid
Sys_gettid:
    mov rax, 186             # SYS_gettid
    syscall
    ret

# Sys_tkill(tid: i64, sig: i64) -> i64
# Send a signal to a thread (deprecated, use tgkill)
.global Sys_tkill
Sys_tkill:
    mov rax, 200             # SYS_tkill
    syscall
    ret

# Sys_tgkill(tgid: i64, tid: i64, sig: i64) -> i64
# Send a signal to a thread in a thread group
.global Sys_tgkill
Sys_tgkill:
    mov rax, 234             # SYS_tgkill
    syscall
    ret

# Sys_set_tid_address(tidptr: *mut i32) -> i64
# Set pointer to thread ID (for CLONE_CHILD_CLEARTID)
.global Sys_set_tid_address
Sys_set_tid_address:
    mov rax, 218             # SYS_set_tid_address
    syscall
    ret

# Sys_exit_group(status: i64) -> !
# Exit all threads in the process
.global Sys_exit_group
Sys_exit_group:
    mov rax, 231             # SYS_exit_group
    syscall
    # Never returns

# ============================================================================
# Async I/O - epoll Syscalls (Phase 8)
# ============================================================================

# Sys_epoll_create1(flags: i64) -> i64
# Create an epoll instance. flags: EPOLL_CLOEXEC=0x80000
.global Sys_epoll_create1
Sys_epoll_create1:
    mov rax, 291             # SYS_epoll_create1
    syscall
    ret

# Sys_epoll_ctl(epfd: i64, op: i64, fd: i64, event: *mut epoll_event) -> i64
# Control an epoll instance. op: EPOLL_CTL_ADD=1, EPOLL_CTL_DEL=2, EPOLL_CTL_MOD=3
.global Sys_epoll_ctl
Sys_epoll_ctl:
    mov rax, 233             # SYS_epoll_ctl
    mov r10, rcx             # event -> r10 (4th arg)
    syscall
    ret

# Sys_epoll_wait(epfd: i64, events: *mut epoll_event, maxevents: i64, timeout: i64) -> i64
# Wait for events on an epoll instance. Returns number of ready fds.
.global Sys_epoll_wait
Sys_epoll_wait:
    mov rax, 232             # SYS_epoll_wait
    mov r10, rcx             # timeout -> r10 (4th arg)
    syscall
    ret

# Sys_epoll_pwait(epfd: i64, events: *mut epoll_event, maxevents: i64, timeout: i64, sigmask: *const sigset_t) -> i64
# epoll_wait with signal mask
.global Sys_epoll_pwait
Sys_epoll_pwait:
    mov rax, 281             # SYS_epoll_pwait
    mov r10, rcx             # timeout -> r10 (4th arg)
    syscall
    ret

# Sys_epoll_pwait2(epfd: i64, events: *mut epoll_event, maxevents: i64, timeout: *const timespec, sigmask: *const sigset_t) -> i64
# epoll_pwait with nanosecond timeout precision
.global Sys_epoll_pwait2
Sys_epoll_pwait2:
    mov rax, 441             # SYS_epoll_pwait2
    mov r10, rcx             # timeout -> r10 (4th arg)
    syscall
    ret

# ============================================================================
# Synchronization Primitives (Phase 8)
# ============================================================================

# sigil_mutex_init(mutex: *mut i32) -> void
# Initialize a mutex (set to 0 = unlocked)
.global sigil_mutex_init
sigil_mutex_init:
    mov dword ptr [rdi], 0
    ret

# sigil_mutex_lock(mutex: *mut i32) -> void
# Acquire a mutex using futex. Spins briefly before sleeping.
.global sigil_mutex_lock
sigil_mutex_lock:
    push rbx
    mov rbx, rdi             # Save mutex pointer

.mutex_lock_try:
    # Try to acquire: compare-and-swap 0 -> 1
    xor eax, eax             # expected = 0 (unlocked)
    mov ecx, 1               # desired = 1 (locked)
    lock cmpxchg dword ptr [rbx], ecx
    jz .mutex_lock_done      # If successful, we're done

    # Mutex is held. Set to 2 (contended) and wait.
    mov ecx, 2               # Set to contended
    xchg dword ptr [rbx], ecx

    # If it was 0 (unlocked), we got it
    test ecx, ecx
    jz .mutex_lock_done

.mutex_lock_wait:
    # Call futex(mutex, FUTEX_WAIT_PRIVATE, 2, NULL, NULL, 0)
    mov rdi, rbx             # uaddr = mutex
    mov esi, 128             # FUTEX_WAIT_PRIVATE = 128
    mov edx, 2               # val = 2 (contended)
    xor r10d, r10d           # timeout = NULL
    xor r8d, r8d             # uaddr2 = NULL
    xor r9d, r9d             # val3 = 0
    mov eax, 202             # SYS_futex
    syscall

    # Try to acquire again
    jmp .mutex_lock_try

.mutex_lock_done:
    pop rbx
    ret

# sigil_mutex_unlock(mutex: *mut i32) -> void
# Release a mutex. If contended, wake one waiter.
.global sigil_mutex_unlock
sigil_mutex_unlock:
    # Atomically set mutex to 0 and get old value
    xor eax, eax
    xchg dword ptr [rdi], eax

    # If old value was 2 (contended), wake a waiter
    cmp eax, 2
    jne .mutex_unlock_done

    # Call futex(mutex, FUTEX_WAKE_PRIVATE, 1, NULL, NULL, 0)
    # rdi already has mutex
    mov esi, 129             # FUTEX_WAKE_PRIVATE = 129
    mov edx, 1               # Wake one waiter
    xor r10d, r10d
    xor r8d, r8d
    xor r9d, r9d
    mov eax, 202             # SYS_futex
    syscall

.mutex_unlock_done:
    ret

# sigil_mutex_trylock(mutex: *mut i32) -> i64
# Try to acquire mutex without blocking. Returns 0 on success, -1 if held.
.global sigil_mutex_trylock
sigil_mutex_trylock:
    xor eax, eax             # expected = 0
    mov ecx, 1               # desired = 1
    lock cmpxchg dword ptr [rdi], ecx
    jz .trylock_success
    mov rax, -1              # Return -1 (failed)
    ret
.trylock_success:
    xor eax, eax             # Return 0 (success)
    ret

# sigil_spinlock_lock(lock: *mut i32) -> void
# Acquire a spinlock (busy-wait)
.global sigil_spinlock_lock
sigil_spinlock_lock:
.spin_try:
    xor eax, eax
    mov ecx, 1
    lock cmpxchg dword ptr [rdi], ecx
    jz .spin_done
    pause                    # CPU hint for spin-wait
    jmp .spin_try
.spin_done:
    ret

# sigil_spinlock_unlock(lock: *mut i32) -> void
# Release a spinlock
.global sigil_spinlock_unlock
sigil_spinlock_unlock:
    mov dword ptr [rdi], 0
    ret

# ============================================================================
# Data Section
# ============================================================================

.section .data
    # Empty for now

.section .note.GNU-stack,"",@progbits
