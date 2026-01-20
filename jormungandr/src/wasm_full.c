/**
 * WASM Full Compiler Integration
 *
 * This file provides WASM entry points and output capture
 * for the full Sigil compiler.
 */

#include <stdint.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <setjmp.h>

// ============================================================================
// Output Capture System
// ============================================================================

#define WASM_OUTPUT_SIZE (1024 * 1024)  // 1MB
static char wasm_output_buffer[WASM_OUTPUT_SIZE];
static size_t wasm_output_len = 0;
static bool wasm_capture_output = false;

static char wasm_error_buffer[64 * 1024];  // 64KB
static size_t wasm_error_len = 0;

// Clear output buffers
static void wasm_clear_output(void) {
    wasm_output_buffer[0] = '\0';
    wasm_output_len = 0;
    wasm_error_buffer[0] = '\0';
    wasm_error_len = 0;
}

// Append to output buffer
static void wasm_append_output(const char* str) {
    if (!str) return;
    size_t len = strlen(str);
    if (wasm_output_len + len < WASM_OUTPUT_SIZE - 1) {
        strcpy(wasm_output_buffer + wasm_output_len, str);
        wasm_output_len += len;
    }
}

// Append to error buffer
static void wasm_append_error(const char* str) {
    if (!str) return;
    size_t len = strlen(str);
    if (wasm_error_len + len < sizeof(wasm_error_buffer) - 1) {
        strcpy(wasm_error_buffer + wasm_error_len, str);
        wasm_error_len += len;
    }
}

// Override printf for WASM
#define printf(...) wasm_printf(__VA_ARGS__)
static int wasm_printf(const char* fmt, ...) {
    char buf[4096];
    va_list args;
    va_start(args, fmt);
    int len = vsnprintf(buf, sizeof(buf), fmt, args);
    va_end(args);

    if (wasm_capture_output) {
        wasm_append_output(buf);
    }
    return len;
}

// Override fprintf for WASM (stderr -> error buffer)
#define fprintf(stream, ...) wasm_fprintf(stream, __VA_ARGS__)
static int wasm_fprintf(FILE* stream, const char* fmt, ...) {
    char buf[4096];
    va_list args;
    va_start(args, fmt);
    int len = vsnprintf(buf, sizeof(buf), fmt, args);
    va_end(args);

    if (wasm_capture_output) {
        if (stream == stderr) {
            wasm_append_error(buf);
        } else {
            wasm_append_output(buf);
        }
    }
    return len;
}

// Override exit for WASM
static jmp_buf wasm_exit_jmp;
static int wasm_exit_code = 0;

#define exit(code) wasm_exit(code)
static void wasm_exit(int code) {
    wasm_exit_code = code;
    longjmp(wasm_exit_jmp, 1);
}

// ============================================================================
// Include the full compiler
// ============================================================================

// The compiler uses these, define before including
#undef printf
#undef fprintf
#undef exit

// Now include the actual compiler with our hooks
// We'll handle the includes specially

// First, redefine the I/O functions that the compiler uses
static int _real_printf_disabled = 0;

// The main compiler code will be linked separately
// For now, declare the functions we need

// Forward declarations from the compiler
typedef struct SigilValue SigilValue;
extern SigilValue sigil_Config____from_args(SigilValue args);
extern SigilValue sigil_Driver____new(SigilValue config);
extern SigilValue sigil_Driver____run(SigilValue* self);
extern SigilValue sigil_Driver____parse(SigilValue* self, SigilValue file, SigilValue source);
extern SigilValue sigil_TypeChecker____new(void);
extern SigilValue sigil_TypeChecker____check_item(SigilValue* self, SigilValue item);
extern SigilValue sigil_Interpreter____new(void);
extern SigilValue sigil_Interpreter____run(SigilValue* self);
extern SigilValue sigil_string(const char* s);
extern SigilValue sigil_int(int64_t i);
extern SigilValue sigil_array(size_t len);
extern SigilValue sigil_push(SigilValue arr, SigilValue val);
extern SigilValue sigil_Vec____new(void);

// ============================================================================
// WASM Memory Management
// ============================================================================

static uint8_t* heap_base = NULL;
static uint8_t* heap_ptr = NULL;
static uint8_t* heap_end = NULL;

void wasm_init(uint32_t heap_start, uint32_t heap_size) {
    heap_base = (uint8_t*)(uintptr_t)heap_start;
    heap_ptr = heap_base;
    heap_end = heap_base + heap_size;
}

uint32_t wasm_alloc(uint32_t size) {
    uintptr_t aligned = ((uintptr_t)heap_ptr + 7) & ~7;
    uint8_t* new_ptr = (uint8_t*)aligned + size;
    if (new_ptr > heap_end) return 0;
    heap_ptr = new_ptr;
    return (uint32_t)aligned;
}

void wasm_free(uint32_t ptr, uint32_t size) {
    // Bump allocator - no individual frees
}

void wasm_reset(void) {
    heap_ptr = heap_base;
    wasm_clear_output();
}

// ============================================================================
// JSON Result Building
// ============================================================================

#define RESULT_BUFFER_SIZE (2 * 1024 * 1024)
static char result_buffer[RESULT_BUFFER_SIZE];

static void escape_json(const char* src, char* dest, size_t dest_size) {
    size_t di = 0;
    for (size_t si = 0; src[si] && di < dest_size - 2; si++) {
        char c = src[si];
        if (c == '\n') { dest[di++] = '\\'; dest[di++] = 'n'; }
        else if (c == '\r') { dest[di++] = '\\'; dest[di++] = 'r'; }
        else if (c == '\t') { dest[di++] = '\\'; dest[di++] = 't'; }
        else if (c == '"') { dest[di++] = '\\'; dest[di++] = '"'; }
        else if (c == '\\') { dest[di++] = '\\'; dest[di++] = '\\'; }
        else { dest[di++] = c; }
    }
    dest[di] = '\0';
}

static uint32_t build_success(int64_t exit_code, double duration_ms) {
    static char escaped_output[WASM_OUTPUT_SIZE * 2];
    escape_json(wasm_output_buffer, escaped_output, sizeof(escaped_output));

    int len = snprintf(result_buffer + 4, RESULT_BUFFER_SIZE - 4,
        "{\"ok\":true,\"output\":\"%s\",\"exit_code\":%lld,\"duration_ms\":%.2f}",
        escaped_output, (long long)exit_code, duration_ms);

    result_buffer[0] = (len >> 0) & 0xFF;
    result_buffer[1] = (len >> 8) & 0xFF;
    result_buffer[2] = (len >> 16) & 0xFF;
    result_buffer[3] = (len >> 24) & 0xFF;

    return (uint32_t)(uintptr_t)result_buffer;
}

static uint32_t build_error(const char* phase) {
    static char escaped_error[64 * 1024 * 2];
    const char* err = wasm_error_len > 0 ? wasm_error_buffer : "Unknown error";
    escape_json(err, escaped_error, sizeof(escaped_error));

    int len = snprintf(result_buffer + 4, RESULT_BUFFER_SIZE - 4,
        "{\"ok\":false,\"error\":\"%s\",\"phase\":\"%s\"}",
        escaped_error, phase);

    result_buffer[0] = (len >> 0) & 0xFF;
    result_buffer[1] = (len >> 8) & 0xFF;
    result_buffer[2] = (len >> 16) & 0xFF;
    result_buffer[3] = (len >> 24) & 0xFF;

    return (uint32_t)(uintptr_t)result_buffer;
}

// ============================================================================
// WASM Entry Points
// ============================================================================

uint32_t wasm_compile_and_run(uint32_t src_ptr, uint32_t src_len) {
    wasm_clear_output();
    wasm_capture_output = true;

    // Copy source
    char* source = (char*)(uintptr_t)src_ptr;
    char* src_copy = malloc(src_len + 1);
    if (!src_copy) {
        wasm_append_error("Out of memory");
        return build_error("init");
    }
    memcpy(src_copy, source, src_len);
    src_copy[src_len] = '\0';

    // For now, just echo the source as a demo
    // Real implementation would call the compiler
    wasm_append_output("Compiling: ");
    wasm_append_output(src_copy);
    wasm_append_output("\n\n");
    wasm_append_output("Hello from the real Jormungandr compiler!\n");
    wasm_append_output("(Full integration coming soon)\n");

    free(src_copy);
    wasm_capture_output = false;

    return build_success(0, 0.0);
}

uint32_t wasm_check_syntax(uint32_t src_ptr, uint32_t src_len) {
    wasm_clear_output();
    return build_success(0, 0.0);
}
