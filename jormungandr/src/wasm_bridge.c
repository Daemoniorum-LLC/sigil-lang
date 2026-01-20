/**
 * WASM Bridge for Jormungandr
 *
 * This C module provides the entry points for calling the Sigil compiler
 * from WebAssembly. It bridges JavaScript to the Jormungandr interpreter.
 *
 * Compile with Emscripten:
 *   emcc wasm_bridge.c from_sigil2.c -o jormungandr.js \
 *        -s WASM=1 -s EXPORTED_FUNCTIONS='["_wasm_compile_and_run", ...]'
 */

#include <stdint.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// ============================================================================
// Memory Management
// ============================================================================

static uint8_t* heap_base = NULL;
static uint8_t* heap_ptr = NULL;
static uint8_t* heap_end = NULL;
static bool initialized = false;

/**
 * Initialize the WASM heap
 * Called once at startup to set up the memory region.
 */
void wasm_init(uint32_t heap_start, uint32_t heap_size) {
    heap_base = (uint8_t*)(uintptr_t)heap_start;
    heap_ptr = heap_base;
    heap_end = heap_base + heap_size;
    initialized = true;
}

/**
 * Allocate memory from the heap
 * Returns a pointer, or 0 on failure. Memory is 8-byte aligned.
 */
uint32_t wasm_alloc(uint32_t size) {
    // Ensure 8-byte alignment
    uintptr_t aligned = ((uintptr_t)heap_ptr + 7) & ~7;
    uint8_t* new_ptr = (uint8_t*)aligned + size;

    if (new_ptr > heap_end) {
        return 0; // Out of memory
    }

    heap_ptr = new_ptr;
    return (uint32_t)aligned;
}

/**
 * Free memory (no-op for bump allocator)
 */
void wasm_free(uint32_t ptr, uint32_t size) {
    // Bump allocator doesn't free individual allocations
}

/**
 * Reset the heap - frees all allocations
 */
void wasm_reset(void) {
    heap_ptr = heap_base;
}

// ============================================================================
// Output Buffer
// ============================================================================

#define OUTPUT_BUFFER_SIZE (1024 * 1024)  // 1MB output buffer
static char output_buffer[OUTPUT_BUFFER_SIZE];
static size_t output_len = 0;

static void clear_output(void) {
    output_buffer[0] = '\0';
    output_len = 0;
}

static void append_output(const char* str) {
    size_t len = strlen(str);
    if (output_len + len < OUTPUT_BUFFER_SIZE - 1) {
        strcpy(output_buffer + output_len, str);
        output_len += len;
    }
}

// Hook for println - called by the Sigil runtime
void sigil_println(const char* str) {
    append_output(str);
    append_output("\n");
}

// ============================================================================
// Result Encoding
// ============================================================================

#define RESULT_BUFFER_SIZE (2 * 1024 * 1024)  // 2MB result buffer
static char result_buffer[RESULT_BUFFER_SIZE];

/**
 * Escape a string for JSON (handles newlines, quotes, backslashes)
 */
static void escape_json_string(const char* src, char* dest, size_t dest_size) {
    size_t di = 0;
    for (size_t si = 0; src[si] && di < dest_size - 2; si++) {
        char c = src[si];
        if (c == '\n') {
            dest[di++] = '\\';
            dest[di++] = 'n';
        } else if (c == '\r') {
            dest[di++] = '\\';
            dest[di++] = 'r';
        } else if (c == '\t') {
            dest[di++] = '\\';
            dest[di++] = 't';
        } else if (c == '"') {
            dest[di++] = '\\';
            dest[di++] = '"';
        } else if (c == '\\') {
            dest[di++] = '\\';
            dest[di++] = '\\';
        } else {
            dest[di++] = c;
        }
    }
    dest[di] = '\0';
}

/**
 * Build a JSON success result
 */
static uint32_t build_success_result(int64_t exit_code, double duration_ms) {
    // Escape output for JSON
    static char escaped_output[OUTPUT_BUFFER_SIZE * 2];
    escape_json_string(output_buffer, escaped_output, sizeof(escaped_output));

    // Format: length (4 bytes) + JSON string
    int json_len = snprintf(
        result_buffer + 4,
        RESULT_BUFFER_SIZE - 4,
        "{\"ok\":true,\"output\":\"%s\",\"exit_code\":%lld,\"duration_ms\":%.2f}",
        escaped_output,
        (long long)exit_code,
        duration_ms
    );

    // Write length prefix (little-endian)
    result_buffer[0] = (json_len >> 0) & 0xFF;
    result_buffer[1] = (json_len >> 8) & 0xFF;
    result_buffer[2] = (json_len >> 16) & 0xFF;
    result_buffer[3] = (json_len >> 24) & 0xFF;

    return (uint32_t)(uintptr_t)result_buffer;
}

/**
 * Build a JSON error result
 */
static uint32_t build_error_result(const char* error, const char* phase) {
    int json_len = snprintf(
        result_buffer + 4,
        RESULT_BUFFER_SIZE - 4,
        "{\"ok\":false,\"error\":\"%s\",\"phase\":\"%s\"}",
        error,
        phase
    );

    result_buffer[0] = (json_len >> 0) & 0xFF;
    result_buffer[1] = (json_len >> 8) & 0xFF;
    result_buffer[2] = (json_len >> 16) & 0xFF;
    result_buffer[3] = (json_len >> 24) & 0xFF;

    return (uint32_t)(uintptr_t)result_buffer;
}

// ============================================================================
// Forward Declarations for Sigil Compiler Functions
// ============================================================================

// These functions should be provided by the compiled Sigil code (from_sigil2.c)
// For now, we'll provide stub implementations

typedef struct {
    int success;
    char error[1024];
    int64_t exit_code;
} CompileResult;

// Stub: Parse source code
static int sigil_parse(const char* source, size_t len, char* error_out) {
    // In real implementation, this calls into the Sigil parser
    return 1; // Success
}

// Stub: Type check
static int sigil_typecheck(char* error_out) {
    return 1; // Success
}

// Stub: Interpret/execute
static int64_t sigil_execute(char* error_out) {
    // Stub: just print hello
    sigil_println("Hello from Jormungandr!");
    return 0;
}

// ============================================================================
// WASM Entry Points
// ============================================================================

/**
 * Compile and run Sigil source code
 *
 * @param src_ptr Pointer to source code in WASM memory
 * @param src_len Length of source code in bytes
 * @return Pointer to length-prefixed JSON result
 */
uint32_t wasm_compile_and_run(uint32_t src_ptr, uint32_t src_len) {
    clear_output();

    // Copy source from WASM memory
    char* source = (char*)(uintptr_t)src_ptr;

    // Null-terminate the source
    char* src_copy = malloc(src_len + 1);
    if (!src_copy) {
        return build_error_result("Out of memory", "init");
    }
    memcpy(src_copy, source, src_len);
    src_copy[src_len] = '\0';

    char error[1024] = {0};

    // Parse
    if (!sigil_parse(src_copy, src_len, error)) {
        free(src_copy);
        return build_error_result(error, "parse");
    }

    // Type check
    if (!sigil_typecheck(error)) {
        free(src_copy);
        return build_error_result(error, "typecheck");
    }

    // Execute
    int64_t exit_code = sigil_execute(error);
    if (error[0] != '\0') {
        free(src_copy);
        return build_error_result(error, "runtime");
    }

    free(src_copy);
    return build_success_result(exit_code, 0.0);
}

/**
 * Check syntax without executing
 *
 * @param src_ptr Pointer to source code
 * @param src_len Length of source code
 * @return Pointer to length-prefixed JSON diagnostics
 */
uint32_t wasm_check_syntax(uint32_t src_ptr, uint32_t src_len) {
    char* source = (char*)(uintptr_t)src_ptr;

    char* src_copy = malloc(src_len + 1);
    if (!src_copy) {
        return build_error_result("Out of memory", "init");
    }
    memcpy(src_copy, source, src_len);
    src_copy[src_len] = '\0';

    char error[1024] = {0};

    // Parse only
    if (!sigil_parse(src_copy, src_len, error)) {
        free(src_copy);
        return build_error_result(error, "parse");
    }

    // Type check only
    if (!sigil_typecheck(error)) {
        free(src_copy);
        return build_error_result(error, "typecheck");
    }

    free(src_copy);

    // Return success (no diagnostics)
    return build_success_result(0, 0.0);
}
