#!/bin/bash
# LLVM Backend Test Runner
# Runs all LLVM backend tests and reports results

# Don't exit on errors - we want to run all tests
set +e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PARSER_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
SIGIL="$PARSER_DIR/target/release/sigil"
TMP_DIR="/tmp/sigil_llvm_tests"

# Change to parser directory so runtime can be found
cd "$PARSER_DIR"

mkdir -p "$TMP_DIR"

TOTAL_PASSED=0
TOTAL_FAILED=0
TOTAL_SKIPPED=0

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}╔════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║        LLVM Backend Test Suite                 ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════════════╝${NC}"
echo ""

# Check compiler exists
if [ ! -f "$SIGIL" ]; then
    echo -e "${RED}Error: Compiler not found at $SIGIL${NC}"
    echo "Run: cargo build --release"
    exit 1
fi

# Check runtime exists
if [ ! -f "$PARSER_DIR/runtime/libsigil_runtime.a" ]; then
    echo -e "${RED}Error: Runtime not found at $PARSER_DIR/runtime/libsigil_runtime.a${NC}"
    echo "Run: cd runtime && make"
    exit 1
fi

# Test function for compilation + execution
run_test() {
    local test_file="$1"
    local expected="$2"
    local should_fail="${3:-no}"
    local test_name=$(basename "$test_file" .sg)
    local test_name="${test_name%.sigil}"  # Also strip .sigil

    local output_bin="$TMP_DIR/$test_name"

    if [ "$should_fail" = "yes" ]; then
        # Test should fail to compile
        if $SIGIL compile "$test_file" -o "$output_bin" --backend llvm 2>/dev/null; then
            echo -e "  ${RED}FAIL${NC}: $test_name (expected compilation error)"
            TOTAL_FAILED=$((TOTAL_FAILED + 1))
            return 1
        else
            echo -e "  ${GREEN}PASS${NC}: $test_name (correctly rejected)"
            TOTAL_PASSED=$((TOTAL_PASSED + 1))
            return 0
        fi
    fi

    # Compile
    local compile_output
    if ! compile_output=$($SIGIL compile "$test_file" -o "$output_bin" --backend llvm 2>&1); then
        echo -e "  ${RED}FAIL${NC}: $test_name (compilation error)"
        TOTAL_FAILED=$((TOTAL_FAILED + 1))
        return 1
    fi

    # Run and check exit code
    set +e
    "$output_bin" >/dev/null 2>&1
    local actual=$?
    set -e

    if [ "$actual" -eq "$expected" ]; then
        echo -e "  ${GREEN}PASS${NC}: $test_name"
        TOTAL_PASSED=$((TOTAL_PASSED + 1))
        return 0
    else
        echo -e "  ${RED}FAIL${NC}: $test_name (expected: $expected, got: $actual)"
        TOTAL_FAILED=$((TOTAL_FAILED + 1))
        return 1
    fi
}

# Run tests from a category
run_category() {
    local category="$1"
    local category_dir="$SCRIPT_DIR/$category"

    if [ ! -d "$category_dir" ]; then
        return
    fi

    echo -e "${BLUE}Testing: $category${NC}"

    # Look for test definitions file or run all .sg files
    local count=0
    for test_file in "$category_dir"/*.sg "$category_dir"/*.sigil; do
        [ -f "$test_file" ] || continue
        count=$((count + 1))
        # Default expected exit code is 0, can be overridden with .expected file
        local expected=0
        local expected_file
        if [[ "$test_file" == *.sg ]]; then
            expected_file="${test_file%.sg}.expected"
        else
            expected_file="${test_file%.sigil}.expected"
        fi
        if [ -f "$expected_file" ]; then
            expected=$(cat "$expected_file")
        fi
        run_test "$test_file" "$expected" || true
    done

    if [ $count -eq 0 ]; then
        echo "  (no tests found)"
    fi
    echo ""
}

# Run FFI tests with specific expectations
run_ffi_tests() {
    echo -e "${BLUE}Testing: ffi${NC}"
    local ffi_dir="$SCRIPT_DIR/ffi"

    run_test "$ffi_dir/01_basic_abs.sg" 42
    run_test "$ffi_dir/02_labs_long.sg" 77
    run_test "$ffi_dir/03_multiple_params.sg" 30
    run_test "$ffi_dir/04_multiple_extern_funcs.sg" 15
    run_test "$ffi_dir/05_sigil_types.sg" 99
    run_test "$ffi_dir/06_float_types.sg" 33
    run_test "$ffi_dir/07_pointer_types.sg" 0
    run_test "$ffi_dir/08_void_return.sg" 7
    run_test "$ffi_dir/09_no_params.sg" 1
    run_test "$ffi_dir/10_lowercase_c_abi.sg" 25
    run_test "$ffi_dir/11_multiple_blocks.sg" 10
    run_test "$ffi_dir/12_size_t.sg" 42
    run_test "$ffi_dir/13_unsupported_abi.sg" 0 "yes"
    run_test "$ffi_dir/14_extern_static_read.sg" 1
    run_test "$ffi_dir/15_extern_static_mutable.sg" 55
    run_test "$ffi_dir/16_extern_static_immutable_error.sg" 0 "yes"

    # Function pointer tests
    # Note: 20 and 23 require parser support for `type` in extern blocks (KNOWN ISSUE)
    run_test "$ffi_dir/21_fn_ptr_param.sg" 0
    run_test "$ffi_dir/22_fn_ptr_return.sg" 0
    run_test "$ffi_dir/24_fn_addr.sg" 1
    run_test "$ffi_dir/25_fn_ptr_call.sg" 42
    run_test "$ffi_dir/26_fn_ptr_assign.sg" 100

    # C callback tests
    run_test "$ffi_dir/27_c_callback_simple.sg" 50
    run_test "$ffi_dir/28_callback_with_args.sg" 99
    run_test "$ffi_dir/29_callback_chain.sg" 24

    # Platform infrastructure
    run_test "$ffi_dir/30_cfg_linux.sg" 1
    run_test "$ffi_dir/31_cfg_feature.sg" 42
    run_test "$ffi_dir/32_cfg_any.sg" 1
    run_test "$ffi_dir/33_link_attr.sg" 0
    # Note: 34_link_multiple has math function interception bug (KNOWN ISSUE)

    # GTK tests - currently require parser support for opaque types in extern blocks
    # Skip until parser supports: `struct GtkApplication;` syntax in extern blocks
    echo -e "  ${YELLOW}SKIP${NC}: GTK tests (requires opaque type declarations in extern blocks)"
    TOTAL_SKIPPED=$((TOTAL_SKIPPED + 5))
    echo ""
}

# Run all test categories
run_ffi_tests
run_category "structs"
run_category "impl"
run_category "enums"
run_category "option"
run_category "vec"
run_category "string"
run_category "io"
run_category "modules"
run_category "native"

# Summary
echo -e "${BLUE}═══════════════════════════════════════════════${NC}"
echo -e "${BLUE}Test Summary${NC}"
echo -e "${BLUE}═══════════════════════════════════════════════${NC}"
echo -e "${GREEN}Passed: $TOTAL_PASSED${NC}"
echo -e "${RED}Failed: $TOTAL_FAILED${NC}"
if [ $TOTAL_SKIPPED -gt 0 ]; then
    echo -e "${YELLOW}Skipped: $TOTAL_SKIPPED${NC}"
fi
TOTAL=$((TOTAL_PASSED + TOTAL_FAILED))
if [ $TOTAL -gt 0 ]; then
    PERCENT=$((TOTAL_PASSED * 100 / TOTAL))
    echo -e "${BLUE}Pass rate: ${PERCENT}%${NC}"
fi
echo -e "${BLUE}═══════════════════════════════════════════════${NC}"

# Cleanup
rm -rf "$TMP_DIR"

# Exit with failure if any tests failed
if [ $TOTAL_FAILED -gt 0 ]; then
    exit 1
fi
