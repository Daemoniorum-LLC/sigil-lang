#!/bin/bash
# ============================================================================
# Jormungandr WASM Test Runner
# TDD Phase 0: Test Infrastructure
# ============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
JORMUNGANDR_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
WASM_FILE="$JORMUNGANDR_DIR/build/jormungandr.wasm"
HARNESS="$SCRIPT_DIR/test_harness.js"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

PASS_COUNT=0
FAIL_COUNT=0
SKIP_COUNT=0

echo "============================================"
echo "  Jormungandr WASM Test Suite"
echo "============================================"
echo ""

# Check if WASM file exists
if [ ! -f "$WASM_FILE" ]; then
    echo -e "${YELLOW}WARNING: jormungandr.wasm not found at:${NC}"
    echo "  $WASM_FILE"
    echo ""
    echo "This is expected in Phase 0 (TDD - tests written before implementation)."
    echo "Build the WASM module with: make wasm"
    echo ""
    echo -e "${YELLOW}Skipping all tests.${NC}"
    exit 0
fi

# Check if Node.js is available
if ! command -v node &> /dev/null; then
    echo -e "${RED}ERROR: Node.js is required but not installed.${NC}"
    exit 1
fi

# Check if test harness exists
if [ ! -f "$HARNESS" ]; then
    echo -e "${RED}ERROR: Test harness not found at: $HARNESS${NC}"
    exit 1
fi

echo "WASM File: $WASM_FILE"
echo "WASM Size: $(wc -c < "$WASM_FILE") bytes"
echo ""

# Run each fixture
for fixture in "$SCRIPT_DIR/fixtures/"*.sg; do
    name=$(basename "$fixture" .sg)

    # Check if this is an error test
    if grep -q "EXPECTED_ERROR:" "$fixture"; then
        expected_error=$(grep "EXPECTED_ERROR:" "$fixture" | sed 's/.*EXPECTED_ERROR: //')
        is_error_test=true
    else
        expected_output=$(grep "EXPECTED_OUTPUT:" "$fixture" | sed 's/.*EXPECTED_OUTPUT: //')
        is_error_test=false
    fi

    expected_exit=$(grep "EXPECTED_EXIT:" "$fixture" | sed 's/.*EXPECTED_EXIT: //')

    printf "Testing: %-20s " "$name"

    # Run via Node.js harness
    set +e
    result=$(node "$HARNESS" "$WASM_FILE" "$fixture" 2>&1)
    harness_exit=$?
    set -e

    if [ $harness_exit -ne 0 ]; then
        echo -e "${RED}FAIL${NC} (harness error)"
        echo "  Harness output: $result"
        FAIL_COUNT=$((FAIL_COUNT + 1))
        continue
    fi

    # Parse result (JSON)
    actual_ok=$(echo "$result" | node -e "const r=JSON.parse(require('fs').readFileSync(0,'utf8')); console.log(r.ok)")
    actual_output=$(echo "$result" | node -e "const r=JSON.parse(require('fs').readFileSync(0,'utf8')); console.log(r.output || '')" | tr -d '\n')
    actual_error=$(echo "$result" | node -e "const r=JSON.parse(require('fs').readFileSync(0,'utf8')); console.log(r.error || '')")
    actual_exit=$(echo "$result" | node -e "const r=JSON.parse(require('fs').readFileSync(0,'utf8')); console.log(r.exitCode || 1)")

    if [ "$is_error_test" = true ]; then
        # Error test: expect failure with specific error message
        if [ "$actual_ok" = "false" ] && echo "$actual_error" | grep -qi "$expected_error"; then
            echo -e "${GREEN}PASS${NC}"
            PASS_COUNT=$((PASS_COUNT + 1))
        else
            echo -e "${RED}FAIL${NC}"
            echo "  Expected error containing: $expected_error"
            echo "  Actual ok: $actual_ok"
            echo "  Actual error: $actual_error"
            FAIL_COUNT=$((FAIL_COUNT + 1))
        fi
    else
        # Success test: expect specific output
        if [ "$actual_ok" = "true" ] && [ "$actual_output" = "$expected_output" ]; then
            echo -e "${GREEN}PASS${NC}"
            PASS_COUNT=$((PASS_COUNT + 1))
        else
            echo -e "${RED}FAIL${NC}"
            echo "  Expected output: '$expected_output'"
            echo "  Actual output:   '$actual_output'"
            echo "  Actual ok: $actual_ok"
            if [ -n "$actual_error" ]; then
                echo "  Error: $actual_error"
            fi
            FAIL_COUNT=$((FAIL_COUNT + 1))
        fi
    fi
done

echo ""
echo "============================================"
echo "  Results"
echo "============================================"
echo -e "  ${GREEN}Passed:${NC}  $PASS_COUNT"
echo -e "  ${RED}Failed:${NC}  $FAIL_COUNT"
echo -e "  ${YELLOW}Skipped:${NC} $SKIP_COUNT"
echo ""

if [ $FAIL_COUNT -gt 0 ]; then
    echo -e "${RED}TESTS FAILED${NC}"
    exit 1
else
    echo -e "${GREEN}ALL TESTS PASSED${NC}"
    exit 0
fi
