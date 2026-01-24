#!/bin/bash
# Build the Sigil native runtime (no libc)
#
# Usage:
#   ./build_native.sh                    # Build runtime library
#   ./build_native.sh test               # Build and run test
#   ./build_native.sh <program.sg>       # Compile program with native runtime
#

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
RUNTIME_S="$SCRIPT_DIR/sigil_runtime_linux_x86_64.s"
RUNTIME_O="$SCRIPT_DIR/sigil_runtime_linux_x86_64.o"
RUNTIME_A="$SCRIPT_DIR/libsigil_native.a"
SIGIL="$SCRIPT_DIR/../target/release/sigil"

# Detect architecture
ARCH=$(uname -m)
if [ "$ARCH" != "x86_64" ]; then
    echo "Error: Native runtime only supports x86_64 (got $ARCH)"
    exit 1
fi

# Detect OS
OS=$(uname -s)
if [ "$OS" != "Linux" ]; then
    echo "Error: Native runtime only supports Linux (got $OS)"
    exit 1
fi

echo "=== Building Sigil Native Runtime ==="
echo "Architecture: $ARCH"
echo "OS: $OS"

# Assemble the runtime
echo "Assembling runtime..."
as -o "$RUNTIME_O" "$RUNTIME_S"

# Create static library
echo "Creating static library..."
ar rcs "$RUNTIME_A" "$RUNTIME_O"

echo "Built: $RUNTIME_A"

# Test mode
if [ "$1" = "test" ]; then
    echo ""
    echo "=== Testing Native Runtime ==="

    # Create a minimal test program
    cat > /tmp/sigil_native_test.s << 'EOF'
.intel_syntax noprefix
.global sigil_main

sigil_main:
    push rbx

    # Test 1: Print a string
    lea rdi, [rip + test_msg]
    call sigil_println

    # Test 2: Print an integer
    mov rdi, 42
    call sigil_print_int

    # Test 3: Allocate memory
    mov rdi, 4096
    call sigil_alloc
    test rax, rax
    jz alloc_failed

    lea rdi, [rip + alloc_ok_msg]
    call sigil_println
    jmp test_time

alloc_failed:
    lea rdi, [rip + alloc_fail_msg]
    call sigil_println

test_time:
    # Test 4: Get time
    call sigil_now
    mov rdi, rax
    call sigil_print_int

    # Test 5: Test vec
    mov rdi, 10
    call sigil_vec_new
    mov rbx, rax

    mov rdi, rbx
    mov rsi, 100
    call sigil_vec_push

    mov rdi, rbx
    mov rsi, 200
    call sigil_vec_push

    mov rdi, rbx
    call sigil_vec_len
    mov rdi, rax
    call sigil_print_int    # Should print 2

    mov rdi, rbx
    mov rsi, 1
    call sigil_vec_get
    mov rdi, rax
    call sigil_print_int    # Should print 200

    lea rdi, [rip + done_msg]
    call sigil_println

    xor eax, eax
    pop rbx
    ret

.section .rodata
test_msg:
    .asciz "Native Runtime Test"
alloc_ok_msg:
    .asciz "alloc: OK"
alloc_fail_msg:
    .asciz "alloc: FAILED"
done_msg:
    .asciz "All tests passed!"
EOF

    # Assemble test
    as -o /tmp/sigil_native_test.o /tmp/sigil_native_test.s

    # Link with native runtime (no libc!)
    ld -o /tmp/sigil_native_test \
        /tmp/sigil_native_test.o \
        "$RUNTIME_O" \
        -nostdlib -static

    echo "Running test..."
    /tmp/sigil_native_test

    echo ""
    echo "=== Test Complete ==="
fi

# Compile a Sigil program
if [ -n "$1" ] && [ -f "$1" ]; then
    echo ""
    echo "=== Compiling: $1 ==="

    BASENAME=$(basename "$1" .sg)
    OUTDIR=$(dirname "$1")

    # Compile to object file
    "$SIGIL" compile "$1" -o "/tmp/${BASENAME}.o" --emit-obj

    # Link with native runtime
    ld -o "$OUTDIR/$BASENAME" \
        "/tmp/${BASENAME}.o" \
        "$RUNTIME_O" \
        -nostdlib -static

    echo "Built: $OUTDIR/$BASENAME"
    echo ""
    echo "Running..."
    "$OUTDIR/$BASENAME"
fi
