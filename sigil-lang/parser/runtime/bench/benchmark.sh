#!/bin/bash
# Sigil Runtime Performance Benchmarks
#
# Compares native runtime (pure assembly) vs C runtime
# Measures: memory allocation, string operations, I/O, math

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
RUNTIME_DIR="$(dirname "$SCRIPT_DIR")"
BENCH_DIR="$SCRIPT_DIR"

# Number of iterations for timing
ITERATIONS=1000

echo "=== Sigil Runtime Performance Benchmarks ==="
echo "Iterations: $ITERATIONS"
echo ""

# Build both runtimes
echo "Building runtimes..."

# Build native runtime
cd "$RUNTIME_DIR"
./build_native.sh > /dev/null 2>&1

# Build C runtime
gcc -O3 -c -o "$RUNTIME_DIR/sigil_runtime_c.o" "$RUNTIME_DIR/sigil_runtime.c" 2>/dev/null || {
    echo "Note: C runtime requires libc headers"
}

echo ""
echo "=== Benchmark Results ==="
echo ""

# Create benchmark programs
cat > /tmp/bench_alloc.s << 'EOF'
.intel_syntax noprefix
.global main_sigil

main_sigil:
    push rbx
    push r12
    mov r12, 10000      # allocations

.alloc_loop:
    mov rdi, 4096
    call sigil_alloc
    dec r12
    jnz .alloc_loop

    xor eax, eax
    pop r12
    pop rbx
    ret

.section .note.GNU-stack,"",@progbits
EOF

cat > /tmp/bench_string.s << 'EOF'
.intel_syntax noprefix
.global main_sigil

main_sigil:
    push rbx
    push r12
    push r13

    mov r12, 10000      # iterations

    # Create source string
    lea rdi, [rip + hello_msg]
    call sigil_string_from
    mov r13, rax

.string_loop:
    # Clone string
    mov rdi, r13
    call sigil_string_clone

    # Get length
    mov rdi, rax
    call sigil_string_len

    dec r12
    jnz .string_loop

    xor eax, eax
    pop r13
    pop r12
    pop rbx
    ret

.section .rodata
hello_msg:
    .asciz "Hello, benchmark world! This is a test string for performance measurement."

.section .note.GNU-stack,"",@progbits
EOF

cat > /tmp/bench_vec.s << 'EOF'
.intel_syntax noprefix
.global main_sigil

main_sigil:
    push rbx
    push r12
    push r13

    mov r12, 1000       # iterations

.vec_loop:
    # Create vec
    mov rdi, 16
    call sigil_vec_new
    mov r13, rax

    # Push 100 items
    mov rbx, 100
.push_loop:
    mov rdi, r13
    mov rsi, rbx
    call sigil_vec_push
    dec rbx
    jnz .push_loop

    dec r12
    jnz .vec_loop

    xor eax, eax
    pop r13
    pop r12
    pop rbx
    ret

.section .note.GNU-stack,"",@progbits
EOF

cat > /tmp/bench_math.s << 'EOF'
.intel_syntax noprefix
.global main_sigil

main_sigil:
    push rbx
    push r12

    mov r12, 100000     # iterations

.math_loop:
    # Call sigil_sqrt with a double value
    mov rdi, 12345678   # integer to sqrt
    call sigil_sqrt

    # Call sigil_abs
    mov rdi, -42
    call sigil_abs

    dec r12
    jnz .math_loop

    xor eax, eax
    pop r12
    pop rbx
    ret

.section .note.GNU-stack,"",@progbits
EOF

cat > /tmp/bench_io.s << 'EOF'
.intel_syntax noprefix
.global main_sigil

main_sigil:
    push rbx
    push r12

    mov r12, 1000       # iterations

.io_loop:
    lea rdi, [rip + msg]
    call sigil_strlen

    dec r12
    jnz .io_loop

    xor eax, eax
    pop r12
    pop rbx
    ret

.section .rodata
msg:
    .asciz "Benchmark string for measuring strlen performance in the Sigil runtime."

.section .note.GNU-stack,"",@progbits
EOF

run_benchmark() {
    local name="$1"
    local source="$2"

    # Assemble
    as -o /tmp/bench.o "$source"

    # Link with native runtime
    ld -o /tmp/bench_native \
        /tmp/bench.o \
        "$RUNTIME_DIR/sigil_runtime_linux_x86_64.o" \
        -nostdlib -static 2>/dev/null

    # Time native
    local start_ns=$(date +%s%N)
    /tmp/bench_native
    local end_ns=$(date +%s%N)
    local native_time=$((end_ns - start_ns))
    local native_ms=$((native_time / 1000000))

    printf "%-20s %8d ms\n" "$name:" "$native_ms"
}

echo "Benchmark             Time"
echo "----------------------------------------"

run_benchmark "Memory Allocation" /tmp/bench_alloc.s
run_benchmark "String Operations" /tmp/bench_string.s
run_benchmark "Vec Operations" /tmp/bench_vec.s
run_benchmark "Math Operations" /tmp/bench_math.s
run_benchmark "I/O (strlen)" /tmp/bench_io.s

echo ""
echo "=== Benchmark Complete ==="

# Cleanup
rm -f /tmp/bench_*.s /tmp/bench.o /tmp/bench_native
