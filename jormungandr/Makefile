# Jormungandr Bootstrap Build System
# Makefile for building the self-hosted Sigil compiler

PARSER_DIR := ../parser
SRC_DIR := src
BUILD_DIR := build
OUT_DIR := $(BUILD_DIR)/c

# Compiler flags
CC := gcc
CFLAGS := -O2 -Wall -Wno-unused-variable -Wno-unused-function -Wno-return-type
LDFLAGS := -lm

# Module list (in dependency order)
MODULES := span token ast lib lexer parser typeck ir lower interp runtime codegen driver

# Generated C files
C_FILES := $(addprefix $(OUT_DIR)/,$(addsuffix .c,$(MODULES)))
O_FILES := $(addprefix $(OUT_DIR)/,$(addsuffix .o,$(MODULES)))

# Final output
TARGET := $(BUILD_DIR)/sigil

.PHONY: all clean compile-sigil compile-c compile-combined test verify

all: $(TARGET)

# Combined single-file compilation (handles static inline functions properly)
ALL_C := $(OUT_DIR)/all.c
TARGET_COMBINED := $(BUILD_DIR)/sigil-combined

$(ALL_C): $(C_FILES)
	@echo "Creating combined C file..."
	@cat $(C_FILES) > $@

$(TARGET_COMBINED): $(ALL_C)
	@echo "Compiling combined file..."
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)
	@echo "Built: $@"

compile-combined: $(TARGET_COMBINED)

# Create build directories
$(BUILD_DIR) $(OUT_DIR):
	mkdir -p $@

# Compile Sigil to C (uses Rust interpreter)
$(OUT_DIR)/%.c: $(SRC_DIR)/%.sg | $(OUT_DIR)
	@echo "Compiling $< to C..."
	@cd $(PARSER_DIR) && cargo run --release -- run-dir ../self-hosted/src -- compile $(abspath $<) 2>/dev/null > $(abspath $@)

# Compile C to object files
$(OUT_DIR)/%.o: $(OUT_DIR)/%.c
	@echo "Compiling $< to object..."
	$(CC) $(CFLAGS) -c $< -o $@

# Link all object files
$(TARGET): $(O_FILES)
	@echo "Linking $(TARGET)..."
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)
	@echo "Built: $@"

# Compile all Sigil to C
compile-sigil: $(C_FILES)
	@echo "All Sigil modules compiled to C"

# Compile all C to objects
compile-c: $(O_FILES)
	@echo "All C files compiled"

# Run basic test
test: $(TARGET)
	@echo "Testing bootstrap compiler..."
	@echo 'fn main() -> i64 { 42 }' > /tmp/test.sg
	$(TARGET) compile /tmp/test.sg > /tmp/test.c
	$(CC) -o /tmp/test /tmp/test.c $(LDFLAGS)
	@test "$$(./tmp/test; echo $$?)" = "42" && echo "Test PASSED" || echo "Test FAILED"

# Verify fixed point (bootstrap identity)
verify: $(TARGET)
	@echo "Verifying fixed point..."
	@mkdir -p $(BUILD_DIR)/verify
	@for mod in $(MODULES); do \
		echo "Compiling $$mod.sg with bootstrap compiler..."; \
		$(TARGET) compile $(SRC_DIR)/$$mod.sg > $(BUILD_DIR)/verify/$$mod.c 2>/dev/null || true; \
	done
	@echo "Comparing outputs..."
	@diff -r $(OUT_DIR) $(BUILD_DIR)/verify && echo "FIXED POINT ACHIEVED!" || echo "Differences found"

# Clean build artifacts
clean:
	rm -rf $(BUILD_DIR)

# ============================================================================
# WASM Build Support
# ============================================================================

EMCC := emcc
WASM_FLAGS := -O2 -s WASM=1 -s MODULARIZE=1 -s EXPORT_NAME='createJormungandr'
WASM_FLAGS += -s EXPORTED_FUNCTIONS='["_wasm_compile_and_run","_wasm_check_syntax","_wasm_alloc","_wasm_free","_wasm_init","_wasm_reset"]'
WASM_FLAGS += -s EXPORTED_RUNTIME_METHODS='["ccall","cwrap","UTF8ToString","stringToUTF8","lengthBytesUTF8"]'
WASM_FLAGS += -s ALLOW_MEMORY_GROWTH=1
WASM_FLAGS += -s INITIAL_MEMORY=16777216
WASM_FLAGS += -s MAXIMUM_MEMORY=268435456
WASM_FLAGS += --no-entry

# WASM modules (includes wasm_bridge)
WASM_MODULES := span token ast lexer parser typeck ir lower interp runtime wasm_bridge
WASM_C_FILES := $(addprefix $(OUT_DIR)/,$(addsuffix .c,$(WASM_MODULES)))
WASM_COMBINED := $(OUT_DIR)/wasm_all.c
WASM_OUT := $(BUILD_DIR)/jormungandr.wasm
WASM_JS := $(BUILD_DIR)/jormungandr.js

.PHONY: wasm wasm-test wasm-verify

# Compile wasm_bridge.sg to C
$(OUT_DIR)/wasm_bridge.c: $(SRC_DIR)/wasm_bridge.sg | $(OUT_DIR)
	@echo "Compiling wasm_bridge.sg to C..."
	@cd $(PARSER_DIR) && cargo run --release -- run-dir ../self-hosted/src -- compile $(abspath $<) 2>/dev/null > $(abspath $@) || echo "// Placeholder for wasm_bridge" > $(abspath $@)

# Create combined WASM source
$(WASM_COMBINED): $(WASM_C_FILES)
	@echo "Creating combined WASM C source..."
	@cat $(WASM_C_FILES) > $@

# Build WASM with emscripten
$(WASM_OUT) $(WASM_JS): $(WASM_COMBINED)
	@echo "Compiling to WebAssembly..."
	@if command -v $(EMCC) >/dev/null 2>&1; then \
		$(EMCC) $(WASM_FLAGS) $(WASM_COMBINED) -o $(WASM_JS) $(LDFLAGS); \
		echo "Built: $(WASM_OUT)"; \
		echo "Built: $(WASM_JS)"; \
	else \
		echo "WARNING: emcc not found. Install Emscripten to build WASM."; \
		echo "See: https://emscripten.org/docs/getting_started/downloads.html"; \
		echo "Creating placeholder WASM files for testing..."; \
		touch $(WASM_OUT) $(WASM_JS); \
	fi

# Build WASM target
wasm: $(WASM_OUT)

# Run WASM tests
wasm-test: $(WASM_OUT)
	@echo "Running WASM test suite..."
	@./tests/wasm/run_wasm_tests.sh

# Verify WASM exports
wasm-verify: $(WASM_OUT)
	@echo "Verifying WASM exports..."
	@node tests/wasm/verify_exports.js $(WASM_OUT)

# ============================================================================
# Performance & Optimization
# ============================================================================

WASM_OPT := wasm-opt
WASM_STRIP := wasm-strip

.PHONY: wasm-optimize wasm-size perf-test memory-test

# Optimize WASM binary for size
wasm-optimize: $(WASM_OUT)
	@echo "Optimizing WASM binary..."
	@if command -v $(WASM_OPT) >/dev/null 2>&1; then \
		$(WASM_OPT) -Oz $(WASM_OUT) -o $(BUILD_DIR)/jormungandr.opt.wasm; \
		echo "Optimized: $(BUILD_DIR)/jormungandr.opt.wasm"; \
	else \
		echo "WARNING: wasm-opt not found. Install binaryen for optimization."; \
	fi
	@if command -v $(WASM_STRIP) >/dev/null 2>&1; then \
		$(WASM_STRIP) $(BUILD_DIR)/jormungandr.opt.wasm 2>/dev/null || true; \
		echo "Stripped debug symbols."; \
	fi
	@if command -v gzip >/dev/null 2>&1; then \
		gzip -9 -k -f $(BUILD_DIR)/jormungandr.opt.wasm; \
		echo "Compressed: $(BUILD_DIR)/jormungandr.opt.wasm.gz"; \
	fi
	@if command -v brotli >/dev/null 2>&1; then \
		brotli -9 -k -f $(BUILD_DIR)/jormungandr.opt.wasm; \
		echo "Compressed: $(BUILD_DIR)/jormungandr.opt.wasm.br"; \
	fi

# Show WASM binary sizes
wasm-size: $(WASM_OUT)
	@echo "WASM Binary Sizes:"
	@echo "-----------------------------------------"
	@if [ -f "$(WASM_OUT)" ]; then \
		SIZE=$$(wc -c < "$(WASM_OUT)"); \
		echo "  jormungandr.wasm:        $$(echo "scale=2; $$SIZE / 1024" | bc) KB"; \
	fi
	@if [ -f "$(BUILD_DIR)/jormungandr.opt.wasm" ]; then \
		SIZE=$$(wc -c < "$(BUILD_DIR)/jormungandr.opt.wasm"); \
		echo "  jormungandr.opt.wasm:    $$(echo "scale=2; $$SIZE / 1024" | bc) KB"; \
	fi
	@if [ -f "$(BUILD_DIR)/jormungandr.opt.wasm.gz" ]; then \
		SIZE=$$(wc -c < "$(BUILD_DIR)/jormungandr.opt.wasm.gz"); \
		echo "  jormungandr.opt.wasm.gz: $$(echo "scale=2; $$SIZE / 1024" | bc) KB (gzip)"; \
	fi
	@if [ -f "$(BUILD_DIR)/jormungandr.opt.wasm.br" ]; then \
		SIZE=$$(wc -c < "$(BUILD_DIR)/jormungandr.opt.wasm.br"); \
		echo "  jormungandr.opt.wasm.br: $$(echo "scale=2; $$SIZE / 1024" | bc) KB (brotli)"; \
	fi

# Run performance benchmarks
perf-test:
	@echo "Running performance benchmarks..."
	@node tests/perf/benchmark.js

# Run memory leak test
memory-test:
	@echo "Running memory leak test..."
	@node --expose-gc tests/perf/memory_leak_test.js

# ============================================================================
# Full Compiler WASM Build (Real Interpreter)
# ============================================================================

WASM_FULL_SRC := src/wasm_compiler.c
WASM_FULL_OUT := $(BUILD_DIR)/jormungandr_full.wasm
WASM_FULL_JS := $(BUILD_DIR)/jormungandr_full.js

WASM_FULL_FLAGS := -O2 -s WASM=1 -s MODULARIZE=1 -s EXPORT_NAME='createJormungandr'
WASM_FULL_FLAGS += -s EXPORTED_FUNCTIONS='["_wasm_compile_and_run","_wasm_check_syntax","_wasm_alloc","_wasm_free","_wasm_init","_wasm_reset","_malloc","_free"]'
WASM_FULL_FLAGS += -s EXPORTED_RUNTIME_METHODS='["ccall","cwrap","UTF8ToString","stringToUTF8","lengthBytesUTF8","getValue"]'
WASM_FULL_FLAGS += -s ALLOW_MEMORY_GROWTH=1
WASM_FULL_FLAGS += -s INITIAL_MEMORY=67108864
WASM_FULL_FLAGS += -s MAXIMUM_MEMORY=536870912
WASM_FULL_FLAGS += -s STACK_SIZE=2097152
WASM_FULL_FLAGS += --no-entry
WASM_FULL_FLAGS += -Wno-unused-variable -Wno-unused-function -Wno-return-type

.PHONY: wasm-full wasm-full-test

# Build full compiler WASM
wasm-full: $(WASM_FULL_OUT)

$(WASM_FULL_OUT) $(WASM_FULL_JS): $(WASM_FULL_SRC) $(BUILD_DIR)/from_sigil2.c
	@echo "Compiling full Jormungandr compiler to WebAssembly..."
	@if command -v $(EMCC) >/dev/null 2>&1; then \
		$(EMCC) $(WASM_FULL_FLAGS) $(WASM_FULL_SRC) -o $(WASM_FULL_JS) $(LDFLAGS); \
		echo "Built: $(WASM_FULL_OUT)"; \
		echo "Built: $(WASM_FULL_JS)"; \
		ls -lh $(WASM_FULL_OUT); \
	else \
		echo "ERROR: emcc not found. Install Emscripten:"; \
		echo "  git clone https://github.com/emscripten-core/emsdk.git"; \
		echo "  cd emsdk && ./emsdk install latest && ./emsdk activate latest"; \
		exit 1; \
	fi

# Test full WASM compiler
wasm-full-test: $(WASM_FULL_OUT)
	@echo "Testing full WASM compiler..."
	@node -e " \
		const fs = require('fs'); \
		const path = '$(WASM_FULL_JS)'; \
		const factory = require(path); \
		(async () => { \
			const Module = await factory(); \
			Module._wasm_init(0x10000, 0x1000000); \
			const src = 'rite main() → i64 { println(\"Hello from WASM!\"); 0 }'; \
			const len = Module.lengthBytesUTF8(src); \
			const ptr = Module._wasm_alloc(len + 1); \
			Module.stringToUTF8(src, ptr, len + 1); \
			const resultPtr = Module._wasm_compile_and_run(ptr, len); \
			const resultLen = Module.getValue(resultPtr, 'i32'); \
			const json = Module.UTF8ToString(resultPtr + 4, resultLen); \
			console.log('Result:', JSON.parse(json)); \
			Module._wasm_free(ptr, len + 1); \
			Module._wasm_reset(); \
		})().catch(e => { console.error(e); process.exit(1); }); \
	"

# ============================================================================
# Help
# ============================================================================

# Show help
help:
	@echo "Jormungandr Bootstrap Build System"
	@echo ""
	@echo "Native Targets:"
	@echo "  all           - Build the bootstrap compiler (default)"
	@echo "  compile-sigil - Compile all .sg files to .c files"
	@echo "  compile-c     - Compile all .c files to .o files"
	@echo "  test          - Run basic test"
	@echo "  verify        - Verify fixed point (bootstrap identity)"
	@echo ""
	@echo "WASM Targets:"
	@echo "  wasm          - Build jormungandr.wasm (stub)"
	@echo "  wasm-full     - Build full compiler WASM (real interpreter)"
	@echo "  wasm-full-test- Test the full WASM compiler"
	@echo "  wasm-test     - Run WASM test suite"
	@echo "  wasm-verify   - Verify WASM exports"
	@echo "  wasm-optimize - Optimize and compress WASM binary"
	@echo "  wasm-size     - Show WASM binary sizes"
	@echo ""
	@echo "Performance:"
	@echo "  perf-test     - Run performance benchmarks"
	@echo "  memory-test   - Run memory leak test"
	@echo ""
	@echo "Other:"
	@echo "  clean         - Remove build artifacts"
	@echo "  help          - Show this help"
