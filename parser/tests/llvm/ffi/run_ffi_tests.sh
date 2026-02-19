#!/bin/bash
# FFI Test Runner for LLVM Backend
# Runs all FFI tests and reports results

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PARSER_DIR="$(cd "$SCRIPT_DIR/../../.." && pwd)"
SIGIL="$PARSER_DIR/target/release/sigil"
TMP_DIR="/tmp/sigil_ffi_tests"

mkdir -p "$TMP_DIR"

# Change to parser directory so runtime can be found
cd "$PARSER_DIR"

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
run_test "$SCRIPT_DIR/14_extern_static_read.sg" 1 "no"
run_test "$SCRIPT_DIR/15_extern_static_mutable.sg" 55 "no"
run_test "$SCRIPT_DIR/16_extern_static_immutable_error.sg" 0 "yes"

# Sprint 1: Function Pointer Types
run_test "$SCRIPT_DIR/20_fn_ptr_type_decl.sg" 0 "no"
run_test "$SCRIPT_DIR/21_fn_ptr_param.sg" 0 "no"
run_test "$SCRIPT_DIR/22_fn_ptr_return.sg" 0 "no"
run_test "$SCRIPT_DIR/23_fn_ptr_typedef.sg" 0 "no"

# Sprint 2: Function Pointer Values
run_test "$SCRIPT_DIR/24_fn_addr.sg" 1 "no"
run_test "$SCRIPT_DIR/25_fn_ptr_call.sg" 42 "no"
run_test "$SCRIPT_DIR/26_fn_ptr_assign.sg" 100 "no"

# Sprint 3: C Callbacks
run_test "$SCRIPT_DIR/27_c_callback_simple.sg" 50 "no"
run_test "$SCRIPT_DIR/28_callback_with_args.sg" 99 "no"
run_test "$SCRIPT_DIR/29_callback_chain.sg" 24 "no"

# Sprint 4: Platform Infrastructure
run_test "$SCRIPT_DIR/30_cfg_linux.sg" 1 "no"
run_test "$SCRIPT_DIR/31_cfg_feature.sg" 42 "no"
run_test "$SCRIPT_DIR/32_cfg_any.sg" 1 "no"
run_test "$SCRIPT_DIR/33_link_attr.sg" 0 "no"
run_test "$SCRIPT_DIR/34_link_multiple.sg" 1 "no"

# Sprint 5: GTK Bindings (Linux only, requires GTK4)
if pkg-config --exists gtk4 2>/dev/null; then
    run_test "$SCRIPT_DIR/40_gtk_init.sg" 0 "no"
    run_test "$SCRIPT_DIR/41_gtk_window.sg" 0 "no"
    run_test "$SCRIPT_DIR/42_gtk_button.sg" 0 "no"
    run_test "$SCRIPT_DIR/43_gtk_signal.sg" 0 "no"
    run_test "$SCRIPT_DIR/44_gtk_button_clicked.sg" 0 "no"
else
    echo -e "${YELLOW}Skipping GTK tests (GTK4 not installed)${NC}"
fi

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
