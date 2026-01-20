#!/bin/bash
# Native Platform (GTK4) Test Runner
# Tests Sigil's native desktop platform support

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PARSER_DIR="$(cd "$SCRIPT_DIR/../../.." && pwd)"
SIGIL="$PARSER_DIR/target/release/sigil"
TMP_DIR="/tmp/sigil_native_tests"

mkdir -p "$TMP_DIR"

PASSED=0
FAILED=0
SKIPPED=0
TOTAL=0

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo "========================================"
echo "Native Platform Test Suite (GTK4)"
echo "========================================"
echo ""

# Check for GTK4
if ! pkg-config --exists gtk4 2>/dev/null; then
    echo -e "${YELLOW}GTK4 not found - skipping native tests${NC}"
    echo "Install with: sudo apt install libgtk-4-dev"
    exit 0
fi

echo "GTK4 version: $(pkg-config --modversion gtk4)"
echo ""

# Test function
run_test() {
    local test_file="$1"
    local expected="$2"
    local test_name=$(basename "$test_file" .sg)

    TOTAL=$((TOTAL + 1))

    echo -n "[$TOTAL] Testing $test_name... "

    local output_bin="$TMP_DIR/$test_name"
    local compile_output

    # Compile the test
    compile_output=$($SIGIL compile "$test_file" -o "$output_bin" 2>&1)
    local compile_status=$?

    if [ $compile_status -ne 0 ]; then
        echo -e "${RED}FAILED${NC} (compilation error)"
        echo "    $compile_output" | head -3
        FAILED=$((FAILED + 1))
        return
    fi

    # Run the test (with display handling for GTK)
    local actual
    if [ -n "$DISPLAY" ] || [ -n "$WAYLAND_DISPLAY" ]; then
        # Has display - run normally
        actual=$("$output_bin" 2>/dev/null; echo $?)
    else
        # No display - try with virtual framebuffer
        if command -v xvfb-run &> /dev/null; then
            actual=$(xvfb-run --auto-servernum "$output_bin" 2>/dev/null; echo $?)
        else
            echo -e "${YELLOW}SKIPPED${NC} (no display, install xvfb)"
            SKIPPED=$((SKIPPED + 1))
            return
        fi
    fi

    # Get just the exit code (last line)
    actual=$(echo "$actual" | tail -1)

    if [ "$actual" = "$expected" ]; then
        echo -e "${GREEN}PASSED${NC} (exit code: $actual)"
        PASSED=$((PASSED + 1))
    else
        echo -e "${RED}FAILED${NC} (expected: $expected, got: $actual)"
        FAILED=$((FAILED + 1))
    fi
}

# Run tests
echo "Running native platform tests..."
echo ""

run_test "$SCRIPT_DIR/01_gtk_init.sg" 0
run_test "$SCRIPT_DIR/02_window_create.sg" 0
run_test "$SCRIPT_DIR/03_label_create.sg" 0
run_test "$SCRIPT_DIR/04_button_create.sg" 0
run_test "$SCRIPT_DIR/05_box_container.sg" 0
run_test "$SCRIPT_DIR/06_signal_connect.sg" 0
run_test "$SCRIPT_DIR/07_full_hierarchy.sg" 0
run_test "$SCRIPT_DIR/08_entry_input.sg" 0
run_test "$SCRIPT_DIR/09_callback_with_data.sg" 0
run_test "$SCRIPT_DIR/10_glib_timeout.sg" 0

echo ""
echo "========================================"
echo "Results: $PASSED passed, $FAILED failed, $SKIPPED skipped (of $TOTAL)"
if [ $FAILED -gt 0 ]; then
    echo -e "${RED}Some tests failed${NC}"
    exit 1
elif [ $SKIPPED -gt 0 ]; then
    echo -e "${YELLOW}Some tests skipped (no display)${NC}"
else
    echo -e "${GREEN}All tests passed!${NC}"
fi
echo "========================================"

# Cleanup
rm -rf "$TMP_DIR"
