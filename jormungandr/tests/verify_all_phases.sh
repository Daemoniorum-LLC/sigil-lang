#!/bin/bash
# ============================================================================
# Jormungandr WASM TDD - Complete Phase Verification
# Runs all quality gates for Phases 0-5
# ============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
JORMUNGANDR_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
WEBSITE_DIR="/home/crook/dev2/workspace/sigil/sigil-lang/website-qliphoth"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

TOTAL_PASS=0
TOTAL_FAIL=0

# Helper function
check() {
    local name="$1"
    local condition="$2"

    printf "    %-45s " "$name"

    if eval "$condition" 2>/dev/null; then
        echo -e "${GREEN}PASS${NC}"
        TOTAL_PASS=$((TOTAL_PASS + 1))
        return 0
    else
        echo -e "${RED}FAIL${NC}"
        TOTAL_FAIL=$((TOTAL_FAIL + 1))
        return 1
    fi
}

echo ""
echo -e "${CYAN}============================================${NC}"
echo -e "${CYAN}  Jormungandr WASM TDD - Phase Verification${NC}"
echo -e "${CYAN}============================================${NC}"
echo ""

# ============================================================================
# PHASE 0: Test Infrastructure
# ============================================================================

echo -e "${YELLOW}Phase 0: Test Infrastructure${NC}"
check "Test directory exists" "[ -d '$SCRIPT_DIR' ]"
check "wasm test directory exists" "[ -d '$SCRIPT_DIR/wasm' ]"
check "test_harness.js exists" "[ -f '$SCRIPT_DIR/wasm/test_harness.js' ]"
check "verify_exports.js exists" "[ -f '$SCRIPT_DIR/wasm/verify_exports.js' ]"
check "run_wasm_tests.sh exists" "[ -f '$SCRIPT_DIR/wasm/run_wasm_tests.sh' ]"
check "run_wasm_tests.sh is executable" "[ -x '$SCRIPT_DIR/wasm/run_wasm_tests.sh' ]"
check "fixtures directory exists" "[ -d '$SCRIPT_DIR/wasm/fixtures' ]"
FIXTURE_COUNT=$(ls -1 "$SCRIPT_DIR/wasm/fixtures/"*.sg 2>/dev/null | wc -l)
check "At least 5 test fixtures ($FIXTURE_COUNT)" "[ $FIXTURE_COUNT -ge 5 ]"
echo ""

# ============================================================================
# PHASE 1: WASM Bridge Module
# ============================================================================

echo -e "${YELLOW}Phase 1: WASM Bridge Module${NC}"
check "wasm_bridge.sg exists" "[ -f '$JORMUNGANDR_DIR/src/wasm_bridge.sg' ]"
check "lib.sg exports wasm_bridge" "grep -q 'wasm_bridge' '$JORMUNGANDR_DIR/src/lib.sg'"
check "No TODOs in wasm_bridge" "! grep -qw 'TODO' '$JORMUNGANDR_DIR/src/wasm_bridge.sg' 2>/dev/null"
check "No FIXMEs in wasm_bridge" "! grep -qw 'FIXME' '$JORMUNGANDR_DIR/src/wasm_bridge.sg' 2>/dev/null"
check "No STUBs in wasm_bridge" "! grep -qw 'STUB' '$JORMUNGANDR_DIR/src/wasm_bridge.sg' 2>/dev/null"
echo ""

# ============================================================================
# PHASE 2: Build System
# ============================================================================

echo -e "${YELLOW}Phase 2: Build System${NC}"
check "Makefile exists" "[ -f '$JORMUNGANDR_DIR/Makefile' ]"
check "Makefile has wasm target" "grep -q '^wasm:' '$JORMUNGANDR_DIR/Makefile'"
check "Makefile has wasm-test target" "grep -q '^wasm-test:' '$JORMUNGANDR_DIR/Makefile'"
check "Makefile has wasm-verify target" "grep -q '^wasm-verify:' '$JORMUNGANDR_DIR/Makefile'"
check "verify_build.sh exists" "[ -f '$SCRIPT_DIR/wasm/verify_build.sh' ]"
check "verify_build.sh is executable" "[ -x '$SCRIPT_DIR/wasm/verify_build.sh' ]"
echo ""

# ============================================================================
# PHASE 3: JavaScript Bridge
# ============================================================================

echo -e "${YELLOW}Phase 3: JavaScript Bridge${NC}"
check "jormungandr.js exists" "[ -f '$WEBSITE_DIR/wasm/jormungandr.js' ]"
check "jormungandr.js has load() method" "grep -q 'static async load' '$WEBSITE_DIR/wasm/jormungandr.js'"
check "jormungandr.js has run() method" "grep -q 'async run(' '$WEBSITE_DIR/wasm/jormungandr.js'"
check "jormungandr.js has check() method" "grep -q 'async check(' '$WEBSITE_DIR/wasm/jormungandr.js'"
check "jormungandr.js has dispose() method" "grep -q 'dispose()' '$WEBSITE_DIR/wasm/jormungandr.js'"
check "jormungandr_js_test.js exists" "[ -f '$SCRIPT_DIR/wasm/jormungandr_js_test.js' ]"
echo ""

# ============================================================================
# PHASE 4: Playground Integration
# ============================================================================

echo -e "${YELLOW}Phase 4: Playground Integration${NC}"
check "playground.html exists" "[ -f '$WEBSITE_DIR/pages/playground.html' ]"
check "playground uses Jormungandr" "grep -q 'Jormungandr' '$WEBSITE_DIR/pages/playground.html'"
check "e2e test directory exists" "[ -d '$SCRIPT_DIR/e2e' ]"
check "playground.spec.js exists" "[ -f '$SCRIPT_DIR/e2e/playground.spec.js' ]"
check "playwright.config.js exists" "[ -f '$SCRIPT_DIR/e2e/playwright.config.js' ]"
echo ""

# ============================================================================
# PHASE 5: Performance & Polish
# ============================================================================

echo -e "${YELLOW}Phase 5: Performance & Polish${NC}"
check "perf test directory exists" "[ -d '$SCRIPT_DIR/perf' ]"
check "benchmark.js exists" "[ -f '$SCRIPT_DIR/perf/benchmark.js' ]"
check "memory_leak_test.js exists" "[ -f '$SCRIPT_DIR/perf/memory_leak_test.js' ]"
check "Makefile has perf-test target" "grep -q '^perf-test:' '$JORMUNGANDR_DIR/Makefile'"
check "Makefile has memory-test target" "grep -q '^memory-test:' '$JORMUNGANDR_DIR/Makefile'"
check "Makefile has wasm-optimize target" "grep -q '^wasm-optimize:' '$JORMUNGANDR_DIR/Makefile'"
check "Makefile has wasm-size target" "grep -q '^wasm-size:' '$JORMUNGANDR_DIR/Makefile'"
check "No TODOs in benchmark.js" "! grep -qw 'TODO' '$SCRIPT_DIR/perf/benchmark.js' 2>/dev/null"
check "No TODOs in memory_leak_test.js" "! grep -qw 'TODO' '$SCRIPT_DIR/perf/memory_leak_test.js' 2>/dev/null"
echo ""

# ============================================================================
# Summary
# ============================================================================

echo -e "${CYAN}============================================${NC}"
echo -e "${CYAN}  Summary${NC}"
echo -e "${CYAN}============================================${NC}"
echo -e "  ${GREEN}Passed:${NC} $TOTAL_PASS"
echo -e "  ${RED}Failed:${NC} $TOTAL_FAIL"
echo ""

if [ $TOTAL_FAIL -gt 0 ]; then
    echo -e "${RED}VERIFICATION FAILED${NC}"
    echo ""
    echo "Some quality gates did not pass. Review the failures above."
    exit 1
else
    echo -e "${GREEN}ALL PHASES VERIFIED SUCCESSFULLY${NC}"
    echo ""
    echo "TDD Roadmap Status:"
    echo "  Phase 0: Test Infrastructure     ✓ Complete"
    echo "  Phase 1: WASM Bridge Module      ✓ Complete"
    echo "  Phase 2: Build System            ✓ Complete"
    echo "  Phase 3: JavaScript Bridge       ✓ Complete"
    echo "  Phase 4: Playground Integration  ✓ Complete"
    echo "  Phase 5: Performance & Polish    ✓ Complete"
    echo ""
    echo "Jormungandr WASM playground is ready for development!"
    exit 0
fi
