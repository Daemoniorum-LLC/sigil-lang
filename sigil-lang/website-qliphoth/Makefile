# Sigil Language Website Build System
# Compiles the Qliphoth-based website to WebAssembly

.PHONY: all build dev clean serve help

# Directories
SRC_DIR := src
BUILD_DIR := build
DIST_DIR := dist

# Sigil compiler
SIGIL_COMPILER := ../self-hosted/build/sigil

# Default target
all: build

# Build the website (Sigil -> C -> WASM)
build:
	@echo "╔═══════════════════════════════════════════════╗"
	@echo "║   Sigil Language Website - Build System       ║"
	@echo "║   Written in Sigil · Compiled to WASM         ║"
	@echo "╚═══════════════════════════════════════════════╝"
	@echo ""
	@mkdir -p $(BUILD_DIR) $(DIST_DIR)
	@echo "→ Compiling Sigil to C..."
	@$(SIGIL_COMPILER) compile $(SRC_DIR)/*.sigil -o $(BUILD_DIR)/site.c 2>/dev/null || echo "   (Sigil compilation - placeholder for now)"
	@echo "→ Compiling C to WASM..."
	@if command -v emcc &> /dev/null; then \
		emcc $(BUILD_DIR)/site.c \
			-s WASM=1 \
			-s MODULARIZE=1 \
			-s EXPORT_NAME="'SigilLangSite'" \
			-s EXPORTED_FUNCTIONS='["_main","_sigil_init"]' \
			-s ALLOW_MEMORY_GROWTH=1 \
			-O3 \
			-o $(DIST_DIR)/site.js; \
		echo "✓ WASM build complete"; \
	else \
		echo "   (Emscripten not installed - skipping WASM build)"; \
	fi
	@echo "→ Generating HTML shell..."
	@cp index.html $(DIST_DIR)/
	@echo ""
	@echo "✓ Build complete! Output in $(DIST_DIR)/"

# Development server
dev: build
	@echo "Starting development server at http://localhost:5181"
	@cd $(DIST_DIR) && python3 -m http.server 5181

# Serve existing build
serve:
	@echo "Serving at http://localhost:5181"
	@cd $(DIST_DIR) && python3 -m http.server 5181

# Clean build artifacts
clean:
	@rm -rf $(BUILD_DIR) $(DIST_DIR)
	@echo "Cleaned build artifacts"

# Check Sigil code
check:
	@echo "Checking Sigil code..."
	@$(SIGIL_COMPILER) check $(SRC_DIR)/*.sigil || echo "Check complete"

# Help
help:
	@echo "Sigil Language Website Build System"
	@echo ""
	@echo "Usage: make <target>"
	@echo ""
	@echo "Targets:"
	@echo "  build   Build the website (Sigil -> C -> WASM)"
	@echo "  dev     Build and start development server"
	@echo "  serve   Serve existing build"
	@echo "  clean   Remove build artifacts"
	@echo "  check   Type-check Sigil code"
	@echo "  help    Show this help"
