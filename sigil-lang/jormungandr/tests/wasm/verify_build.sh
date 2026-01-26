#!/bin/bash
# ============================================================================
# Jormungandr WASM Build Verification
# TDD Phase 2: Build System Verification
# ============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
JORMUNGANDR_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo "============================================"
echo "  Jormungandr WASM Build Verification"
echo "============================================"
echo ""

PASS_COUNT=0
FAIL_COUNT=0

# Helper function for tests
check() {
    local name="$1"
    local condition="$2"

    printf "  %-40s " "$name"

    if eval "$condition"; then
        echo -e "${GREEN}PASS${NC}"
        PASS_COUNT=$((PASS_COUNT + 1))
        return 0
    else
        echo -e "${RED}FAIL${NC}"
        FAIL_COUNT=$((FAIL_COUNT + 1))
        return 1
    fi
}

# ============================================================================
# Source File Checks
# ============================================================================

echo "Source Files:"
check "wasm_bridge.sg exists" "[ -f '$JORMUNGANDR_DIR/src/wasm_bridge.sg' ]"
check "lib.sg exports wasm_bridge" "grep -q 'wasm_bridge' '$JORMUNGANDR_DIR/src/lib.sg'"

# ============================================================================
# Makefile Checks
# ============================================================================

echo ""
echo "Makefile:"
check "Makefile exists" "[ -f '$JORMUNGANDR_DIR/Makefile' ]"
check "Makefile has 'wasm' target" "grep -q '^wasm:' '$JORMUNGANDR_DIR/Makefile'"
check "Makefile has 'wasm-test' target" "grep -q '^wasm-test:' '$JORMUNGANDR_DIR/Makefile'"
check "Makefile has 'wasm-verify' target" "grep -q '^wasm-verify:' '$JORMUNGANDR_DIR/Makefile'"

# ============================================================================
# Test Infrastructure Checks
# ============================================================================

echo ""
echo "Test Infrastructure:"
check "run_wasm_tests.sh exists" "[ -f '$SCRIPT_DIR/run_wasm_tests.sh' ]"
check "run_wasm_tests.sh is executable" "[ -x '$SCRIPT_DIR/run_wasm_tests.sh' ]"
check "test_harness.js exists" "[ -f '$SCRIPT_DIR/test_harness.js' ]"
check "verify_exports.js exists" "[ -f '$SCRIPT_DIR/verify_exports.js' ]"
check "fixtures directory exists" "[ -d '$SCRIPT_DIR/fixtures' ]"
FIXTURE_COUNT=$(ls -1 "$SCRIPT_DIR/fixtures/"*.sg 2>/dev/null | wc -l)
check "At least 5 test fixtures exist ($FIXTURE_COUNT found)" "[ $FIXTURE_COUNT -ge 5 ]"

# ============================================================================
# Quality Gate Checks
# ============================================================================

echo ""
echo "Quality Gate:"
# Check for stub patterns - grep returns 0 if found, 1 if not found
# We want NOT found, so we invert with !
check "No TODOs in wasm_bridge.sg" "! grep -q -w 'TODO' '$JORMUNGANDR_DIR/src/wasm_bridge.sg'"
check "No FIXMEs in wasm_bridge.sg" "! grep -q -w 'FIXME' '$JORMUNGANDR_DIR/src/wasm_bridge.sg'"
check "No STUBs in wasm_bridge.sg" "! grep -q -w 'STUB' '$JORMUNGANDR_DIR/src/wasm_bridge.sg'"

# ============================================================================
# Build Directory Checks
# ============================================================================

echo ""
echo "Build Directory:"
check "build directory exists" "[ -d '$JORMUNGANDR_DIR/build' ]"

# Check for WASM output (optional - may not exist in TDD before implementation)
if [ -f "$JORMUNGANDR_DIR/build/jormungandr.wasm" ]; then
    WASM_SIZE=$(wc -c < "$JORMUNGANDR_DIR/build/jormungandr.wasm")
    check "jormungandr.wasm exists" "true"
    check "jormungandr.wasm > 0 bytes" "[ $WASM_SIZE -gt 0 ]"
else
    echo -e "  ${YELLOW}jormungandr.wasm not built yet (expected in TDD)${NC}"
fi

# ============================================================================
# Summary
# ============================================================================

echo ""
echo "============================================"
echo "  Results"
echo "============================================"
echo -e "  ${GREEN}Passed:${NC}  $PASS_COUNT"
echo -e "  ${RED}Failed:${NC}  $FAIL_COUNT"
echo ""

if [ $FAIL_COUNT -gt 0 ]; then
    echo -e "${RED}BUILD VERIFICATION FAILED${NC}"
    exit 1
else
    echo -e "${GREEN}BUILD VERIFICATION PASSED${NC}"
    exit 0
fi
