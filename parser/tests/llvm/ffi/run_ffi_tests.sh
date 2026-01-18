#!/bin/bash
# FFI Test Runner for LLVM Backend
# Runs all FFI tests and reports results

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PARSER_DIR="$(cd "$SCRIPT_DIR/../../.." && pwd)"
SIGIL="$PARSER_DIR/target/release/sigil"
TMP_DIR="/tmp/sigil_ffi_tests"

mkdir -p "$TMP_DIR"

PASSED=0
FAILED=0
TOTAL=0

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo "========================================"
echo "FFI Test Suite - LLVM Backend"
echo "========================================"
echo ""

# Test function
run_test() {
    local test_file="$1"
    local expected="$2"
    local should_fail="$3"  # "yes" if test should fail to compile
    local test_name=$(basename "$test_file" .sg)

    TOTAL=$((TOTAL + 1))

    echo -n "[$TOTAL] Testing $test_name... "

    local output_bin="$TMP_DIR/$test_name"

    if [ "$should_fail" = "yes" ]; then
        # Test should fail to compile
        if $SIGIL compile "$test_file" -o "$output_bin" --backend llvm 2>/dev/null; then
            echo -e "${RED}FAILED${NC} (expected compilation error but succeeded)"
            FAILED=$((FAILED + 1))
            return
        else
            echo -e "${GREEN}PASSED${NC} (correctly rejected)"
            PASSED=$((PASSED + 1))
            return
        fi
    fi

    # Compile
    if ! $SIGIL compile "$test_file" -o "$output_bin" --backend llvm 2>/dev/null; then
        echo -e "${RED}FAILED${NC} (compilation error)"
        FAILED=$((FAILED + 1))
        return
    fi

    # Run and check exit code
    set +e
    "$output_bin"
    local actual=$?
    set -e

    if [ "$actual" -eq "$expected" ]; then
        echo -e "${GREEN}PASSED${NC} (exit code: $actual)"
        PASSED=$((PASSED + 1))
    else
        echo -e "${RED}FAILED${NC} (expected: $expected, got: $actual)"
        FAILED=$((FAILED + 1))
    fi
}

# Run tests
echo "Running FFI tests..."
echo ""

run_test "$SCRIPT_DIR/01_basic_abs.sg" 42 "no"
run_test "$SCRIPT_DIR/02_labs_long.sg" 77 "no"
run_test "$SCRIPT_DIR/03_multiple_params.sg" 30 "no"
run_test "$SCRIPT_DIR/04_multiple_extern_funcs.sg" 15 "no"
run_test "$SCRIPT_DIR/05_sigil_types.sg" 99 "no"
run_test "$SCRIPT_DIR/06_float_types.sg" 33 "no"
run_test "$SCRIPT_DIR/07_pointer_types.sg" 0 "no"
run_test "$SCRIPT_DIR/08_void_return.sg" 7 "no"
run_test "$SCRIPT_DIR/09_no_params.sg" 1 "no"
run_test "$SCRIPT_DIR/10_lowercase_c_abi.sg" 25 "no"
run_test "$SCRIPT_DIR/11_multiple_blocks.sg" 10 "no"
run_test "$SCRIPT_DIR/12_size_t.sg" 42 "no"
run_test "$SCRIPT_DIR/13_unsupported_abi.sg" 0 "yes"

echo ""
echo "========================================"
echo "Results: $PASSED/$TOTAL passed"
if [ $FAILED -gt 0 ]; then
    echo -e "${RED}$FAILED tests failed${NC}"
    exit 1
else
    echo -e "${GREEN}All tests passed!${NC}"
fi
echo "========================================"

# Cleanup
rm -rf "$TMP_DIR"
