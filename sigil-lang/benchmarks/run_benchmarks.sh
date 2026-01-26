#!/bin/bash

# Sigil Benchmark Suite
# Compares: Interpreter, JIT (Cranelift), LLVM

set -e

SIGIL="../parser/target/release/sigil"
BENCHMARKS_DIR="$(dirname "$0")"
RESULTS_FILE="$BENCHMARKS_DIR/RESULTS.md"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${BLUE}╔═══════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║           SIGIL BENCHMARK SUITE                           ║${NC}"
echo -e "${BLUE}╚═══════════════════════════════════════════════════════════╝${NC}"
echo ""

# Check if sigil binary exists
if [ ! -f "$SIGIL" ]; then
    echo -e "${RED}Error: Sigil binary not found at $SIGIL${NC}"
    echo "Run: cd ../parser && cargo build --release"
    exit 1
fi

# Initialize results file
cat > "$RESULTS_FILE" << 'EOF'
# Sigil Benchmark Results

**Date:** $(date +%Y-%m-%d)
**Platform:** $(uname -s) $(uname -m)
**CPU:** $(grep "model name" /proc/cpuinfo 2>/dev/null | head -1 | cut -d: -f2 | xargs || sysctl -n machdep.cpu.brand_string 2>/dev/null || echo "Unknown")

## Results

| Benchmark | Interpreter | JIT (Cranelift) | LLVM | Speedup (LLVM vs Interp) |
|-----------|-------------|-----------------|------|--------------------------|
EOF

# Function to run a single benchmark
run_benchmark() {
    local name="$1"
    local file="$2"
    local timeout="${3:-120}"

    echo -e "${YELLOW}━━━ $name ━━━${NC}"

    # Interpreter
    echo -n "  Interpreter: "
    INTERP_TIME=$( { time -p timeout "$timeout" "$SIGIL" run "$file" > /dev/null 2>&1; } 2>&1 | grep real | awk '{print $2}' )
    if [ -z "$INTERP_TIME" ]; then
        INTERP_TIME="timeout"
        echo -e "${RED}TIMEOUT${NC}"
    else
        echo -e "${GREEN}${INTERP_TIME}s${NC}"
    fi

    # JIT (Cranelift)
    echo -n "  JIT:         "
    JIT_TIME=$( { time -p timeout "$timeout" "$SIGIL" jit "$file" > /dev/null 2>&1; } 2>&1 | grep real | awk '{print $2}' )
    if [ -z "$JIT_TIME" ]; then
        JIT_TIME="timeout"
        echo -e "${RED}TIMEOUT${NC}"
    else
        echo -e "${GREEN}${JIT_TIME}s${NC}"
    fi

    # LLVM (compile and run)
    echo -n "  LLVM:        "
    local exe_name="/tmp/sigil_bench_$(basename "$file" .sg)"
    if timeout 30 "$SIGIL" compile "$file" -o "$exe_name" > /dev/null 2>&1; then
        LLVM_TIME=$( { time -p timeout "$timeout" "$exe_name" > /dev/null 2>&1; } 2>&1 | grep real | awk '{print $2}' )
        rm -f "$exe_name"
        if [ -z "$LLVM_TIME" ]; then
            LLVM_TIME="timeout"
            echo -e "${RED}TIMEOUT${NC}"
        else
            echo -e "${GREEN}${LLVM_TIME}s${NC}"
        fi
    else
        LLVM_TIME="compile_err"
        echo -e "${RED}COMPILE ERROR${NC}"
    fi

    # Calculate speedup
    if [[ "$INTERP_TIME" != "timeout" && "$LLVM_TIME" != "timeout" && "$LLVM_TIME" != "compile_err" ]]; then
        SPEEDUP=$(echo "scale=1; $INTERP_TIME / $LLVM_TIME" | bc 2>/dev/null || echo "N/A")
        echo -e "  ${BLUE}Speedup: ${SPEEDUP}x${NC}"
    else
        SPEEDUP="N/A"
    fi

    # Append to results file
    echo "| $name | ${INTERP_TIME}s | ${JIT_TIME}s | ${LLVM_TIME}s | ${SPEEDUP}x |" >> "$RESULTS_FILE"
    echo ""
}

echo ""
echo -e "${BLUE}Running benchmarks...${NC}"
echo ""

# Run all benchmarks
run_benchmark "Fibonacci (recursive, n=35)" "$BENCHMARKS_DIR/fib_recursive.sg" 300
run_benchmark "Fibonacci (iterative, n=50)" "$BENCHMARKS_DIR/fib_iterative.sg" 60
run_benchmark "Prime Sieve (n=100000)" "$BENCHMARKS_DIR/primes_sieve.sg" 120
run_benchmark "Matrix Multiply (150x150)" "$BENCHMARKS_DIR/matrix_mult.sg" 120
run_benchmark "String Operations" "$BENCHMARKS_DIR/string_ops.sg" 60
run_benchmark "Collection Operations" "$BENCHMARKS_DIR/collection_ops.sg" 60
run_benchmark "N-Body (500k iterations)" "$BENCHMARKS_DIR/nbody.sg" 300

# Finalize results
cat >> "$RESULTS_FILE" << 'EOF'

## System Info

```
EOF

echo "Sigil Version: $($SIGIL --version 2>/dev/null || echo 'unknown')" >> "$RESULTS_FILE"
echo "Date: $(date)" >> "$RESULTS_FILE"
echo "Kernel: $(uname -r)" >> "$RESULTS_FILE"
echo '```' >> "$RESULTS_FILE"

echo -e "${GREEN}═══════════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}Benchmarks complete! Results saved to: $RESULTS_FILE${NC}"
echo -e "${GREEN}═══════════════════════════════════════════════════════════${NC}"
