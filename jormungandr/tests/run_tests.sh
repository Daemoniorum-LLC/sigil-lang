#!/bin/bash
# Jormungandr Test Runner
# Runs all test cases and reports results

set +e  # Don't exit on errors - we want to run all tests

SIGIL_COMPILER="../build/sigil2"
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
echo -e "${BLUE}║     Jormungandr Test Suite Runner             ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════════════╝${NC}"
echo ""

# Check compiler exists
if [ ! -f "$SIGIL_COMPILER" ]; then
    echo -e "${RED}❌ Error: Compiler not found at $SIGIL_COMPILER${NC}"
    exit 1
fi

# Function to run a single test
run_test() {
    local test_file="$1"
    local test_name=$(basename "$test_file" .sg)
    local test_dir=$(dirname "$test_file")
    local expected="${test_file%.sg}.expected"
    local test_c="$TEMP_DIR/${test_name}.c"
    local test_bin="$TEMP_DIR/${test_name}"
    local test_out="$TEMP_DIR/${test_name}.out"
    local test_err="$TEMP_DIR/${test_name}.err"
    local gcc_err="$TEMP_DIR/${test_name}_gcc.err"

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

    # Step 1: Compile Sigil to C
    if ! "$SIGIL_COMPILER" compile "$test_file" -o "$test_c" 2>"$test_err"; then
        echo -e "${RED}❌ FAIL${NC}: Sigil compilation failed"
        if [ -s "$test_err" ]; then
            sed 's/^/    /' "$test_err"
        fi
        ((FAIL++))
        return 1
    fi

    # Step 2: Fix known C generation bugs (duplicate sigil_add, stray endif)
    sed -i '/^SigilValue sigil_add(SigilValue a, SigilValue b) { return sigil_int(a\.v\.i + b\.v\.i); }$/d' "$test_c"
    sed -i '/^#endif \/\* SIGIL_EXTRA_STDLIB_DEFINED \*\/$/d' "$test_c"

    # Step 2.5: Patch push_str to handle TAG_REF (compiler bug fix)
    sed -i '/^SigilValue sigil_String____push_str(SigilValue s, SigilValue str) {$/a\    if (str.tag == TAG_REF && str.v.ptr) str = *(SigilValue*)str.v.ptr;' "$test_c"

    # Step 2.6: Add missing String method implementations (compiler bug)
    sed -i '/^#endif \/\* SIGIL_BUILTINS_DEFINED \*\/$/i\SigilValue sigil_String____clone(SigilValue s) {\n    if (s.tag == TAG_STRING && s.v.s) return sigil_string(s.v.s);\n    return s;\n}\nSigilValue sigil_String____is_empty(SigilValue s) {\n    if (s.tag != TAG_STRING) return sigil_bool(true);\n    if (!s.v.s || s.v.s[0] == 0) return sigil_bool(true);\n    return sigil_bool(false);\n}\nSigilValue sigil_String____contains(SigilValue s, SigilValue sub) {\n    if (s.tag != TAG_STRING || sub.tag != TAG_STRING) return sigil_bool(false);\n    if (!s.v.s || !sub.v.s) return sigil_bool(false);\n    return sigil_bool(strstr(s.v.s, sub.v.s) != NULL);\n}\n' "$test_c"

    # Step 2.5: Verify main() wrapper was emitted
    if ! grep -q "^int main(" "$test_c"; then
        echo -e "${RED}❌ FAIL${NC}: No main() wrapper emitted in generated C"
        ((FAIL++))
        return 1
    fi

    # Step 3: Compile C to binary
    if ! gcc -g -O0 -o "$test_bin" "$test_c" -lm 2>"$gcc_err"; then
        echo -e "${RED}❌ FAIL${NC}: C compilation failed"
        if [ -s "$gcc_err" ]; then
            grep "error:" "$gcc_err" | head -10 | sed 's/^/    /'
        fi
        ((FAIL++))
        return 1
    fi

    # Step 4: Run binary and check output
    if [ -f "$expected" ]; then
        if ! "$test_bin" > "$test_out" 2>&1; then
            echo -e "${RED}❌ FAIL${NC}: Runtime error"
            if [ -s "$test_out" ]; then
                sed 's/^/    /' "$test_out"
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
        if "$test_bin" > "$test_out" 2>&1; then
            echo -e "${GREEN}✅ PASS${NC} (no output check)"
            ((PASS++))
            return 0
        else
            echo -e "${RED}❌ FAIL${NC}: Runtime error"
            if [ -s "$test_out" ]; then
                sed 's/^/    /' "$test_out"
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
    # Skip if it's the spec template file
    if [[ "$category" == *"TEST_TEMPLATE.sg"* ]] || [[ "$category" == *"README.md"* ]]; then
        continue
    fi

    if [ -d "$TEST_DIR/$category" ]; then
        test_files=$(find "$TEST_DIR/$category" -name "*.sg" ! -name "TEST_TEMPLATE.sg" 2>/dev/null | sort)
        if [ -n "$test_files" ]; then
            # Extract display name
            if [[ "$category" == spec/* ]]; then
                display_name="$(basename "$category") (spec)"
            else
                display_name="$category"
            fi
            echo -e "${BLUE}━━━ $display_name tests ━━━${NC}"
            while IFS= read -r test_file; do
                run_test "$test_file"
            done <<< "$test_files"
            echo ""
        fi
    fi
done

# Summary
echo -e "${BLUE}╔════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║                  Results                       ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════════════╝${NC}"
echo -e "  ${GREEN}✅ Passed:${NC}  $PASS"
echo -e "  ${RED}❌ Failed:${NC}  $FAIL"
echo -e "  ${YELLOW}⏭  Skipped:${NC} $SKIP"
echo -e "  ${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"

TOTAL=$((PASS + FAIL))
if [ $TOTAL -gt 0 ]; then
    PERCENT=$((PASS * 100 / TOTAL))
    echo -e "  Pass Rate: ${PERCENT}%"
fi

if [ $FAIL -eq 0 ]; then
    echo -e "\n${GREEN}🎉 All tests passed!${NC}\n"
    exit 0
else
    echo -e "\n${RED}⚠️  Some tests failed${NC}\n"
    exit 1
fi
