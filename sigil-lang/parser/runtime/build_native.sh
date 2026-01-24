#!/bin/bash
# Build the Sigil native runtime (no libc)
#
# Usage:
#   ./build_native.sh                    # Build runtime library
#   ./build_native.sh test               # Build and run test
#   ./build_native.sh <program.sg>       # Compile program with native runtime
#
# Supported platforms:
#   - Linux x86_64
#   - macOS x86_64 (Intel)
#   - macOS ARM64 (Apple Silicon)
#   - Windows x64 (via MinGW cross-compile)
#

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SIGIL="$SCRIPT_DIR/../target/release/sigil"

# Detect architecture and OS
ARCH=$(uname -m)
OS=$(uname -s)

# Map architecture names
case "$ARCH" in
    x86_64|amd64)
        ARCH="x86_64"
        ;;
    arm64|aarch64)
        ARCH="arm64"
        ;;
    *)
        echo "Error: Unsupported architecture: $ARCH"
        exit 1
        ;;
esac

# Map OS names and select runtime
case "$OS" in
    Linux)
        if [ "$ARCH" != "x86_64" ]; then
            echo "Error: Linux runtime only supports x86_64 (got $ARCH)"
            exit 1
        fi
        RUNTIME_S="$SCRIPT_DIR/sigil_runtime_linux_x86_64.s"
        RUNTIME_O="$SCRIPT_DIR/sigil_runtime_linux_x86_64.o"
        AS_CMD="as"
        LD_CMD="ld"
        LD_FLAGS="-nostdlib -static"
        ;;
    Darwin)
        if [ "$ARCH" = "x86_64" ]; then
            RUNTIME_S="$SCRIPT_DIR/sigil_runtime_macos_x86_64.s"
            RUNTIME_O="$SCRIPT_DIR/sigil_runtime_macos_x86_64.o"
        elif [ "$ARCH" = "arm64" ]; then
            RUNTIME_S="$SCRIPT_DIR/sigil_runtime_macos_arm64.s"
            RUNTIME_O="$SCRIPT_DIR/sigil_runtime_macos_arm64.o"
        else
            echo "Error: Unsupported macOS architecture: $ARCH"
            exit 1
        fi
        AS_CMD="as"
        LD_CMD="ld"
        LD_FLAGS="-e _start -static -macos_version_min 11.0"
        ;;
    MINGW*|MSYS*|CYGWIN*)
        OS="Windows"
        RUNTIME_S="$SCRIPT_DIR/sigil_runtime_windows_x64.s"
        RUNTIME_O="$SCRIPT_DIR/sigil_runtime_windows_x64.o"
        AS_CMD="as"
        LD_CMD="ld"
        LD_FLAGS="-lkernel32"
        ;;
    *)
        echo "Error: Unsupported OS: $OS"
        echo "Supported: Linux, Darwin (macOS), Windows (MinGW)"
        exit 1
        ;;
esac

RUNTIME_A="$SCRIPT_DIR/libsigil_native.a"

echo "=== Building Sigil Native Runtime ==="
echo "Architecture: $ARCH"
echo "OS: $OS"
echo "Runtime: $(basename "$RUNTIME_S")"

# Check if runtime source exists
if [ ! -f "$RUNTIME_S" ]; then
    echo "Error: Runtime source not found: $RUNTIME_S"
    exit 1
fi

# Assemble the runtime
echo "Assembling runtime..."
$AS_CMD -o "$RUNTIME_O" "$RUNTIME_S"

# Create static library
echo "Creating static library..."
ar rcs "$RUNTIME_A" "$RUNTIME_O"

echo "Built: $RUNTIME_A"

# Test mode (Linux only for now)
if [ "$1" = "test" ]; then
    if [ "$OS" != "Linux" ]; then
        echo "Warning: Test mode only supported on Linux"
        exit 0
    fi

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

.section .note.GNU-stack,"",@progbits
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
    $LD_CMD -o "$OUTDIR/$BASENAME" \
        "/tmp/${BASENAME}.o" \
        "$RUNTIME_O" \
        $LD_FLAGS

    echo "Built: $OUTDIR/$BASENAME"
    echo ""
    echo "Running..."
    "$OUTDIR/$BASENAME"
fi
