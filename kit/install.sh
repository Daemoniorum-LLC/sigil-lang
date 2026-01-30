#!/bin/bash
#
# Sigil Installation Script
#
# This script installs the Sigil compiler to ~/.sigil/bin
# and optionally adds it to your PATH.
#

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Installation directory
INSTALL_DIR="${SIGIL_HOME:-$HOME/.sigil}"
BIN_DIR="$INSTALL_DIR/bin"

# Script directory (where kit was extracted)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo ""
echo "╔═══════════════════════════════════════╗"
echo "║     Sigil Development Kit Installer    ║"
echo "╚═══════════════════════════════════════╝"
echo ""

# Check if sigil binary exists in kit
if [ ! -f "$SCRIPT_DIR/bin/sigil" ]; then
    echo -e "${RED}Error: sigil binary not found in kit/bin/${NC}"
    echo "Please ensure you extracted the complete kit."
    exit 1
fi

# Create installation directory
echo "Installing to: $INSTALL_DIR"
mkdir -p "$BIN_DIR"

# Copy binary
echo "Copying sigil binary..."
cp "$SCRIPT_DIR/bin/sigil" "$BIN_DIR/sigil"
chmod +x "$BIN_DIR/sigil"

# Copy documentation (optional but recommended)
if [ -d "$SCRIPT_DIR/docs" ]; then
    echo "Copying documentation..."
    mkdir -p "$INSTALL_DIR/docs"
    cp -r "$SCRIPT_DIR/docs/"* "$INSTALL_DIR/docs/"
fi

if [ -d "$SCRIPT_DIR/methodologies" ]; then
    echo "Copying methodologies..."
    mkdir -p "$INSTALL_DIR/methodologies"
    cp -r "$SCRIPT_DIR/methodologies/"* "$INSTALL_DIR/methodologies/"
fi

if [ -d "$SCRIPT_DIR/examples" ]; then
    echo "Copying examples..."
    mkdir -p "$INSTALL_DIR/examples"
    cp -r "$SCRIPT_DIR/examples/"* "$INSTALL_DIR/examples/"
fi

if [ -d "$SCRIPT_DIR/style" ]; then
    echo "Copying style guide..."
    mkdir -p "$INSTALL_DIR/style"
    cp -r "$SCRIPT_DIR/style/"* "$INSTALL_DIR/style/"
fi

# Verify installation
if "$BIN_DIR/sigil" --version > /dev/null 2>&1; then
    VERSION=$("$BIN_DIR/sigil" --version 2>&1 | head -n1)
    echo -e "${GREEN}✓ Sigil installed successfully: $VERSION${NC}"
else
    echo -e "${YELLOW}Warning: sigil binary installed but version check failed${NC}"
fi

# Check if already in PATH
if command -v sigil &> /dev/null; then
    EXISTING=$(command -v sigil)
    if [ "$EXISTING" != "$BIN_DIR/sigil" ]; then
        echo -e "${YELLOW}Note: Another sigil found at $EXISTING${NC}"
    else
        echo -e "${GREEN}✓ sigil is already in your PATH${NC}"
        echo ""
        echo "Installation complete! Try: sigil --help"
        exit 0
    fi
fi

# PATH setup
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Add sigil to your PATH by adding this line to your shell config:"
echo ""

# Detect shell
SHELL_NAME=$(basename "$SHELL")
case "$SHELL_NAME" in
    bash)
        CONFIG_FILE="$HOME/.bashrc"
        ;;
    zsh)
        CONFIG_FILE="$HOME/.zshrc"
        ;;
    fish)
        CONFIG_FILE="$HOME/.config/fish/config.fish"
        ;;
    *)
        CONFIG_FILE="your shell config"
        ;;
esac

if [ "$SHELL_NAME" = "fish" ]; then
    echo -e "${GREEN}set -gx PATH $BIN_DIR \$PATH${NC}"
else
    echo -e "${GREEN}export PATH=\"$BIN_DIR:\$PATH\"${NC}"
fi

echo ""
echo "Config file: $CONFIG_FILE"
echo ""

# Offer to add automatically
read -p "Add to $CONFIG_FILE automatically? [y/N] " -n 1 -r
echo ""

if [[ $REPLY =~ ^[Yy]$ ]]; then
    if [ "$SHELL_NAME" = "fish" ]; then
        echo "set -gx PATH $BIN_DIR \$PATH" >> "$CONFIG_FILE"
    else
        echo "" >> "$CONFIG_FILE"
        echo "# Sigil" >> "$CONFIG_FILE"
        echo "export PATH=\"$BIN_DIR:\$PATH\"" >> "$CONFIG_FILE"
    fi
    echo -e "${GREEN}✓ Added to $CONFIG_FILE${NC}"
    echo ""
    echo "Run 'source $CONFIG_FILE' or open a new terminal."
else
    echo "Skipped. Add manually when ready."
fi

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo -e "${GREEN}Installation complete!${NC}"
echo ""
echo "Quick start:"
echo "  sigil run $INSTALL_DIR/examples/00_hello.sg"
echo ""
echo "Documentation:"
echo "  $INSTALL_DIR/docs/GETTING-STARTED.md"
echo "  $INSTALL_DIR/docs/AGENT-GUIDE.md"
echo ""
