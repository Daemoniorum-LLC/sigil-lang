#!/bin/bash
# Qliphoth WASM Integration Tests
# Validates WASM compilation and basic execution

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PARSER_DIR="/home/crook/dev2/workspace/sigil/parser"
EXAMPLES_DIR="${SCRIPT_DIR}/../../examples"
SIGIL="$PARSER_DIR/target/release/sigil"

PASS=0
FAIL=0

echo "=== Qliphoth WASM Integration Tests ==="
echo ""

# Ensure compiler is built with WASM feature
if [ ! -f "$SIGIL" ]; then
    echo "Building Sigil compiler..."
    cd "$PARSER_DIR"
    cargo build --release --features wasm
fi

cd "$EXAMPLES_DIR"

echo "--- WASM Compilation Tests ---"

# Test 1: counter_simple
echo -n "Testing counter_simple... "
if $SIGIL wasm counter_simple.sigil -o counter_simple_test.wasm --target browser >/dev/null 2>&1; then
    if [ -f counter_simple_test.wasm ]; then
        echo "PASS"
        PASS=$((PASS + 1))
    else
        echo "FAIL (no output)"
        FAIL=$((FAIL + 1))
    fi
else
    echo "FAIL (compile error)"
    FAIL=$((FAIL + 1))
fi

# Test 2: todo
echo -n "Testing todo... "
if $SIGIL wasm todo.sigil -o todo_test.wasm --target browser >/dev/null 2>&1; then
    if [ -f todo_test.wasm ]; then
        echo "PASS"
        PASS=$((PASS + 1))
    else
        echo "FAIL (no output)"
        FAIL=$((FAIL + 1))
    fi
else
    echo "FAIL (compile error)"
    FAIL=$((FAIL + 1))
fi

# Test 3: qliphoth_demo
echo -n "Testing qliphoth_demo... "
if $SIGIL wasm qliphoth_demo.sigil -o qliphoth_demo_test.wasm --target browser >/dev/null 2>&1; then
    if [ -f qliphoth_demo_test.wasm ]; then
        echo "PASS"
        PASS=$((PASS + 1))
    else
        echo "FAIL (no output)"
        FAIL=$((FAIL + 1))
    fi
else
    echo "FAIL (compile error)"
    FAIL=$((FAIL + 1))
fi

# Test 4: hello_vdom
echo -n "Testing hello_vdom... "
if $SIGIL wasm hello_vdom.sigil -o hello_vdom_test.wasm --target browser >/dev/null 2>&1; then
    if [ -f hello_vdom_test.wasm ]; then
        echo "PASS"
        PASS=$((PASS + 1))
    else
        echo "FAIL (no output)"
        FAIL=$((FAIL + 1))
    fi
else
    echo "FAIL (compile error)"
    FAIL=$((FAIL + 1))
fi

# Test 5: simple_vdom
echo -n "Testing simple_vdom... "
if $SIGIL wasm simple_vdom.sigil -o simple_vdom_test.wasm --target browser >/dev/null 2>&1; then
    if [ -f simple_vdom_test.wasm ]; then
        echo "PASS"
        PASS=$((PASS + 1))
    else
        echo "FAIL (no output)"
        FAIL=$((FAIL + 1))
    fi
else
    echo "FAIL (compile error)"
    FAIL=$((FAIL + 1))
fi

echo ""
echo "--- Interpreter Validation Tests ---"

# Test 6: counter_simple interpreter
echo -n "Testing counter_simple (interpreter)... "
if $SIGIL run counter_simple.sigil 2>&1 | grep -q "=== Sigil Counter Demo ==="; then
    echo "PASS"
    PASS=$((PASS + 1))
else
    echo "FAIL"
    FAIL=$((FAIL + 1))
fi

# Test 7: todo interpreter
echo -n "Testing todo (interpreter)... "
if $SIGIL run todo.sigil 2>&1 | grep -q "=== Sigil Todo Demo ==="; then
    echo "PASS"
    PASS=$((PASS + 1))
else
    echo "FAIL"
    FAIL=$((FAIL + 1))
fi

# Test 8: qliphoth_demo interpreter
echo -n "Testing qliphoth_demo (interpreter)... "
if $SIGIL run qliphoth_demo.sigil 2>&1 | grep -q "Qliphoth Demo"; then
    echo "PASS"
    PASS=$((PASS + 1))
else
    echo "FAIL"
    FAIL=$((FAIL + 1))
fi

# Cleanup test files
rm -f *_test.wasm

echo ""
echo "=== Summary ==="
TOTAL=$((PASS + FAIL))
echo "Passed: ${PASS}/${TOTAL}"
echo "Failed: ${FAIL}/${TOTAL}"

if [ "$FAIL" -gt 0 ]; then
    exit 1
fi

echo ""
echo "All WASM integration tests passed!"
