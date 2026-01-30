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

# Show help
help:
	@echo "Jormungandr Bootstrap Build System"
	@echo ""
	@echo "Targets:"
	@echo "  all          - Build the bootstrap compiler (default)"
	@echo "  compile-sigil - Compile all .sg files to .c files"
	@echo "  compile-c    - Compile all .c files to .o files"
	@echo "  test         - Run basic test"
	@echo "  verify       - Verify fixed point (bootstrap identity)"
	@echo "  clean        - Remove build artifacts"
	@echo "  help         - Show this help"
