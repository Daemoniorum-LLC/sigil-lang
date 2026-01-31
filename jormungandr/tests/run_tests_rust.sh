#!/bin/bash
# Jormungandr Test Runner (Rust Bootstrap)
# Runs all test cases using the Rust compiler as interpreter

set +e  # Don't exit on errors - we want to run all tests

SIGIL_COMPILER="../../parser/target/release/sigil"
TEST_DIR="."
PASS=0
FAIL=0
SKIP=0
TEMP_DIR="/tmp/sigil_tests_$$"

# Command-line options
FILTER_SPEC=""
FILTER_PRIORITY=""

# Parse command-line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --spec)
            FILTER_SPEC="$2"
            shift 2
            ;;
        --priority)
            FILTER_PRIORITY="$2"
            shift 2
            ;;
        --help)
            echo "Usage: $0 [options]"
            echo ""
            echo "Options:"
            echo "  --spec SECTION       Run only tests from specified spec section (e.g., 03_types)"
            echo "  --priority LEVEL     Run only tests with specified priority (P0, P1, P2)"
            echo "  --help               Show this help message"
            echo ""
            echo "Examples:"
            echo "  $0                           # Run all tests"
            echo "  $0 --spec 03_types           # Run only type system tests"
            echo "  $0 --priority P0             # Run only P0 (bootstrap critical) tests"
            echo "  $0 --spec 03_types --priority P0  # Run P0 type tests"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Create temp directory
mkdir -p "$TEMP_DIR"

# Cleanup on exit
cleanup() {
    rm -rf "$TEMP_DIR"
}
trap cleanup EXIT

echo -e "${BLUE}╔════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║     Jormungandr Test Suite (Rust Bootstrap)   ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════════════╝${NC}"
echo ""

# Check compiler exists
if [ ! -f "$SIGIL_COMPILER" ]; then
    echo -e "${RED}❌ Error: Rust compiler not found at $SIGIL_COMPILER${NC}"
    echo "Run: cd ../../parser && cargo build --release"
    exit 1
fi

# Function to run a single test
run_test() {
    local test_file="$1"
    local test_name=$(basename "$test_file" .sg)
    local test_dir=$(dirname "$test_file")
    local expected="${test_file%.sg}.expected"
    local error_expected="${test_file%.sg}.error_expected"
    local test_out="$TEMP_DIR/${test_name}.out"
    local test_err="$TEMP_DIR/${test_name}.err"

    # Check if test should be skipped
    if grep -q "^// SKIP" "$test_file" 2>/dev/null; then
        echo -e "  ${YELLOW}⏭  SKIP${NC}: $test_name (marked as SKIP)"
        ((SKIP++))
        return 0
    fi

    # Check priority filter (only for spec tests)
    if [ -n "$FILTER_PRIORITY" ]; then
        # Extract priority from test name (format: P0_001_name.sg)
        if [[ "$test_name" =~ ^(P[0-2])_ ]]; then
            test_priority="${BASH_REMATCH[1]}"
            if [ "$test_priority" != "$FILTER_PRIORITY" ]; then
                # Skip silently - filtered out
                ((SKIP++))
                return 0
            fi
        fi
    fi

    echo -n "  Testing: $test_name ... "

    # NEGATIVE TEST: Check if this is a test that should FAIL to compile/run
    if [ -f "$error_expected" ]; then
        # Run test and expect failure
        if "$SIGIL_COMPILER" run "$test_file" > "$test_out" 2>"$test_err"; then
            # Test ran successfully when it should have failed!
            echo -e "${RED}❌ FAIL${NC}: Should have errored but succeeded"
            echo "    Expected error containing:"
            sed 's/^/      /' "$error_expected" | head -5
            ((FAIL++))
            return 1
        fi

        # Check if error output contains expected error message
        local expected_error=$(cat "$error_expected")
        if grep -qF "$expected_error" "$test_err" 2>/dev/null || grep -qF "$expected_error" "$test_out" 2>/dev/null; then
            echo -e "${GREEN}✅ PASS${NC} (expected error)"
            ((PASS++))
            return 0
        else
            echo -e "${RED}❌ FAIL${NC}: Wrong error message"
            echo "    Expected error containing: $expected_error"
            echo "    Got:"
            sed 's/^/      /' "$test_err" | head -5
            sed 's/^/      /' "$test_out" | head -5
            ((FAIL++))
            return 1
        fi
    fi

    # POSITIVE TEST: Run test with Rust compiler (interpreter mode)
    if [ -f "$expected" ]; then
        # Test has expected output - check it
        if ! "$SIGIL_COMPILER" run "$test_file" > "$test_out" 2>"$test_err"; then
            echo -e "${RED}❌ FAIL${NC}: Runtime error"
            if [ -s "$test_err" ]; then
                sed 's/^/    /' "$test_err" | head -10
            fi
            ((FAIL++))
            return 1
        fi

        if diff -q "$expected" "$test_out" > /dev/null 2>&1; then
            echo -e "${GREEN}✅ PASS${NC}"
            ((PASS++))
            return 0
        else
            echo -e "${RED}❌ FAIL${NC}: Output mismatch"
            echo "    Expected:"
            sed 's/^/      /' "$expected"
            echo "    Got:"
            sed 's/^/      /' "$test_out"
            ((FAIL++))
            return 1
        fi
    else
        # No expected output - just check if it runs
        if "$SIGIL_COMPILER" run "$test_file" > "$test_out" 2>&1; then
            echo -e "${GREEN}✅ PASS${NC} (no output check)"
            ((PASS++))
            return 0
        else
            echo -e "${RED}❌ FAIL${NC}: Runtime error"
            if [ -s "$test_out" ]; then
                sed 's/^/    /' "$test_out" | head -10
            fi
            ((FAIL++))
            return 1
        fi
    fi
}

# Run tests from each category
if [ -n "$FILTER_SPEC" ]; then
    # Run only specified spec section
    categories="spec/$FILTER_SPEC"
else
    # Run all categories
    categories="features stdlib integration spec/*"
fi

for category in $categories; do
    if [ ! -d "$category" ]; then
        continue
    fi

    echo -e "${BLUE}Testing category: $category${NC}"

    # Find all .sg files in category
    test_files=$(find "$category" -name "*.sg" -type f | sort)

    if [ -z "$test_files" ]; then
        echo -e "  ${YELLOW}No tests found${NC}"
        continue
    fi

    # Run each test
    for test_file in $test_files; do
        run_test "$test_file"
    done

    echo ""
done

# Print summary
echo -e "${BLUE}═══════════════════════════════════════════════${NC}"
echo -e "${BLUE}Test Summary${NC}"
echo -e "${BLUE}═══════════════════════════════════════════════${NC}"
echo -e "${GREEN}✅ Passed: $PASS${NC}"
echo -e "${RED}❌ Failed: $FAIL${NC}"
echo -e "${YELLOW}⏭  Skipped: $SKIP${NC}"
echo -e "${BLUE}───────────────────────────────────────────────${NC}"

TOTAL=$((PASS + FAIL))
if [ $TOTAL -gt 0 ]; then
    PERCENTAGE=$((PASS * 100 / TOTAL))
    echo -e "${BLUE}Pass rate: ${PERCENTAGE}%${NC}"
fi

# Exit with error if any tests failed
if [ $FAIL -gt 0 ]; then
    exit 1
fi

exit 0
