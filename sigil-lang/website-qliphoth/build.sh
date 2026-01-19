#!/bin/bash
#
# Sigil Website Build Script
# Compiles Sigil source files to WebAssembly
#
# Usage:
#   ./build.sh              # Build all WASM files
#   ./build.sh --wasm-only  # Only compile WASM (skip checks)
#   ./build.sh --clean      # Clean and rebuild
#
# Requirements:
#   - Sigil compiler with WASM support
#   - Source files must use native Sigil syntax
#

set -e

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC_DIR="$SCRIPT_DIR/src"
OUT_DIR="$SCRIPT_DIR/public/wasm"
SIGIL_COMPILER="${SIGIL_COMPILER:-/home/crook/dev2/workspace/sigil/parser/target/release/sigil}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Files to compile (in dependency order)
SIGIL_FILES=(
    "minimal"
    "index"
    "docs"
    "learn"
    "agents"
    "pattern"
    "qliphoth"
)

# Parse arguments
WASM_ONLY=false
CLEAN=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --wasm-only)
            WASM_ONLY=true
            shift
            ;;
        --clean)
            CLEAN=true
            shift
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${BLUE}  Sigil Website Build${NC}"
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""

# Clean if requested
if $CLEAN; then
    echo -e "${YELLOW}Cleaning old WASM files...${NC}"
    rm -f "$OUT_DIR"/*.wasm
fi

# Check for Sigil compiler
if [[ ! -x "$SIGIL_COMPILER" ]]; then
    echo -e "${YELLOW}Warning: Sigil compiler not found at $SIGIL_COMPILER${NC}"
    echo -e "${YELLOW}Attempting to build compiler...${NC}"

    PARSER_DIR="/home/crook/dev2/workspace/sigil/parser"
    if [[ -f "$PARSER_DIR/Cargo.toml" ]]; then
        echo "Building Sigil compiler with WASM support..."
        cd "$PARSER_DIR"
        cargo build --release --features wasm 2>&1
        cd "$SCRIPT_DIR"
    fi

    if [[ ! -x "$SIGIL_COMPILER" ]]; then
        echo -e "${RED}Error: Sigil compiler not available${NC}"
        echo "Please build the compiler first:"
        echo "  cd /home/crook/dev2/workspace/sigil/parser"
        echo "  cargo build --release --features wasm"
        exit 1
    fi
fi

echo -e "${GREEN}Using compiler: $SIGIL_COMPILER${NC}"
echo ""

# Compile each file
COMPILED=0
FAILED=0
SKIPPED=0

for name in "${SIGIL_FILES[@]}"; do
    src_file="$SRC_DIR/$name.sigil"
    out_file="$OUT_DIR/$name.wasm"

    if [[ ! -f "$src_file" ]]; then
        echo -e "${YELLOW}⊘ Skipping $name.sigil (source not found)${NC}"
        ((SKIPPED++))
        continue
    fi

    # Check if source is newer than output
    if [[ -f "$out_file" ]] && [[ "$src_file" -ot "$out_file" ]]; then
        echo -e "${BLUE}↷ Skipping $name.sigil (up to date)${NC}"
        ((SKIPPED++))
        continue
    fi

    echo -e "${BLUE}⊙ Compiling $name.sigil...${NC}"

    if "$SIGIL_COMPILER" wasm "$src_file" -o "$out_file" 2>&1; then
        size=$(ls -lh "$out_file" | awk '{print $5}')
        echo -e "${GREEN}  ✓ $name.wasm ($size)${NC}"
        ((COMPILED++))
    else
        echo -e "${RED}  ✗ Failed to compile $name.sigil${NC}"
        ((FAILED++))

        # Note: Source files may need native Sigil syntax conversion
        echo -e "${YELLOW}    Note: Source may need native Sigil syntax (⎇/⎉, yea/nay, etc.)${NC}"
    fi
done

echo ""
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${GREEN}Compiled: $COMPILED${NC}  ${YELLOW}Skipped: $SKIPPED${NC}  ${RED}Failed: $FAILED${NC}"
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"

# List generated files
echo ""
echo "Generated WASM files:"
ls -lh "$OUT_DIR"/*.wasm 2>/dev/null || echo "  (none)"

if [[ $FAILED -gt 0 ]]; then
    echo ""
    echo -e "${YELLOW}Some files failed to compile.${NC}"
    echo -e "${YELLOW}The source files may need conversion to native Sigil syntax.${NC}"
    echo -e "${YELLOW}Existing .wasm files will continue to work.${NC}"
    exit 1
fi

echo ""
echo -e "${GREEN}Build complete!${NC}"
