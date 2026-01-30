/**
 * Jormungandr WASM Compiler Integration
 *
 * This file integrates the real Sigil compiler (from_sigil2.c) with the WASM
 * environment. Due to the compiler being a code-generator (not interpreter),
 * this provides:
 *   - Full parsing with error messages
 *   - Type checking with error messages
 *   - AST dump for debugging
 *
 * Note: The compiler generates C code, it doesn't interpret. For execution,
 * use the Rust-based Sigil interpreter or future WASM interpreter.
 */

#include <stdint.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdarg.h>
#include <setjmp.h>

// ============================================================================
// Output Capture System
// ============================================================================

#define WASM_OUTPUT_SIZE (1024 * 1024)  // 1MB output buffer
#define WASM_ERROR_SIZE  (64 * 1024)    // 64KB error buffer

static char wasm_output_buffer[WASM_OUTPUT_SIZE];
static size_t wasm_output_len = 0;

static char wasm_error_buffer[WASM_ERROR_SIZE];
static size_t wasm_error_len = 0;

static bool wasm_capture_active = false;

// Clear output buffers
static void wasm_clear_buffers(void) {
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
        memcpy(wasm_output_buffer + wasm_output_len, str, len + 1);
        wasm_output_len += len;
    }
}

// Append to error buffer
static void wasm_append_error(const char* str) {
    if (!str) return;
    size_t len = strlen(str);
    if (wasm_error_len + len < WASM_ERROR_SIZE - 1) {
        memcpy(wasm_error_buffer + wasm_error_len, str, len + 1);
        wasm_error_len += len;
    }
}

// ============================================================================
// Printf/Fprintf Hooks - Must be defined BEFORE including compiler
// ============================================================================

// Captured printf
static int wasm_captured_printf(const char* fmt, ...) {
    static char buf[8192];
    va_list args;
    va_start(args, fmt);
    int len = vsnprintf(buf, sizeof(buf), fmt, args);
    va_end(args);

    if (wasm_capture_active) {
        wasm_append_output(buf);
    }
    return len;
}

// Captured fprintf
static int wasm_captured_fprintf(FILE* stream, const char* fmt, ...) {
    static char buf[8192];
    va_list args;
    va_start(args, fmt);
    int len = vsnprintf(buf, sizeof(buf), fmt, args);
    va_end(args);

    if (wasm_capture_active) {
        if (stream == stderr) {
            wasm_append_error(buf);
        } else {
            wasm_append_output(buf);
        }
    }
    return len;
}

// Hook printf and fprintf for the compiler
#define printf wasm_captured_printf
#define fprintf wasm_captured_fprintf

// ============================================================================
// Exit Hook - Capture exit() calls
// ============================================================================

static jmp_buf wasm_exit_jump;
static int wasm_exit_code = 0;

static void wasm_captured_exit(int code) {
    wasm_exit_code = code;
    longjmp(wasm_exit_jump, 1);
}

#define exit(code) wasm_captured_exit(code)

// ============================================================================
// Include the Real Compiler
// ============================================================================

// Disable the duplicate sigil_add definition by checking if already defined
#define SIGIL_SKIP_DUPLICATE_ADD

// Forward declarations for stub functions (before including generated code)
// These structures are defined in from_sigil2.c
struct SigilValue;
typedef struct SigilValue SigilValue;

// Stub declarations - these will be defined after the include
SigilValue sigil_lower_file(SigilValue ast, SigilValue checker, SigilValue filename);
SigilValue sigil_Interpreter____new(void);

#include "../build/from_sigil2.c"

// ============================================================================
// Stub Functions for Missing Modules
// ============================================================================
// The from_sigil2.c is missing Lowerer and Interpreter modules.
// These stubs allow compilation but parsing/type-checking still work.

// Lowerer stub - returns an error result
SigilValue sigil_lower_file(SigilValue ast, SigilValue checker, SigilValue filename) {
    (void)ast; (void)checker; (void)filename;
    wasm_captured_fprintf(stderr, "Error: Lowerer not available in WASM build\n");
    return sigil_Result____Err(sigil_string("Lowerer not available in WASM build"));
}

// Interpreter stubs
SigilValue sigil_Interpreter____new(void) {
    static SigilValue empty = { .tag = TAG_NULL, .evidence = SIGIL_KNOWN };
    wasm_captured_fprintf(stderr, "Error: Interpreter not available in WASM build\n");
    return empty;
}

// Restore original functions
#undef printf
#undef fprintf
#undef exit

// ============================================================================
// WASM Memory Management
// ============================================================================

static uint8_t* wasm_heap_base = NULL;
static uint8_t* wasm_heap_ptr = NULL;
static uint8_t* wasm_heap_end = NULL;

__attribute__((export_name("wasm_init")))
void wasm_init(uint32_t heap_start, uint32_t heap_size) {
    wasm_heap_base = (uint8_t*)(uintptr_t)heap_start;
    wasm_heap_ptr = wasm_heap_base;
    wasm_heap_end = wasm_heap_base + heap_size;
}

__attribute__((export_name("wasm_alloc")))
uint32_t wasm_alloc(uint32_t size) {
    // Align to 8 bytes
    uintptr_t aligned = ((uintptr_t)wasm_heap_ptr + 7) & ~7;
    uint8_t* new_ptr = (uint8_t*)aligned + size;
    if (new_ptr > wasm_heap_end) return 0;
    wasm_heap_ptr = new_ptr;
    return (uint32_t)aligned;
}

__attribute__((export_name("wasm_free")))
void wasm_free(uint32_t ptr, uint32_t size) {
    // Bump allocator - no individual frees
    (void)ptr;
    (void)size;
}

__attribute__((export_name("wasm_reset")))
void wasm_reset(void) {
    wasm_heap_ptr = wasm_heap_base;
    wasm_clear_buffers();
}

// ============================================================================
// JSON Result Building
// ============================================================================

#define RESULT_BUFFER_SIZE (2 * 1024 * 1024)
static char wasm_result_buffer[RESULT_BUFFER_SIZE];

static void escape_json(const char* src, char* dest, size_t dest_size) {
    size_t di = 0;
    for (size_t si = 0; src[si] && di < dest_size - 6; si++) {
        unsigned char c = (unsigned char)src[si];
        if (c == '\n') { dest[di++] = '\\'; dest[di++] = 'n'; }
        else if (c == '\r') { dest[di++] = '\\'; dest[di++] = 'r'; }
        else if (c == '\t') { dest[di++] = '\\'; dest[di++] = 't'; }
        else if (c == '"') { dest[di++] = '\\'; dest[di++] = '"'; }
        else if (c == '\\') { dest[di++] = '\\'; dest[di++] = '\\'; }
        else if (c < 32) {
            // Control characters as \uXXXX
            di += snprintf(dest + di, dest_size - di, "\\u%04x", c);
        }
        else { dest[di++] = c; }
    }
    dest[di] = '\0';
}

static uint32_t build_success_result(int64_t exit_code, double duration_ms) {
    static char escaped_output[WASM_OUTPUT_SIZE * 2];
    escape_json(wasm_output_buffer, escaped_output, sizeof(escaped_output));

    int len = snprintf(wasm_result_buffer + 4, RESULT_BUFFER_SIZE - 4,
        "{\"ok\":true,\"output\":\"%s\",\"exit_code\":%lld,\"duration_ms\":%.2f}",
        escaped_output, (long long)exit_code, duration_ms);

    // Length prefix (little-endian)
    wasm_result_buffer[0] = (len >> 0) & 0xFF;
    wasm_result_buffer[1] = (len >> 8) & 0xFF;
    wasm_result_buffer[2] = (len >> 16) & 0xFF;
    wasm_result_buffer[3] = (len >> 24) & 0xFF;

    return (uint32_t)(uintptr_t)wasm_result_buffer;
}

static uint32_t build_error_result(const char* phase, const char* message) {
    static char escaped_error[WASM_ERROR_SIZE * 2];
    const char* err = (message && message[0]) ? message :
                      (wasm_error_len > 0 ? wasm_error_buffer : "Unknown error");
    escape_json(err, escaped_error, sizeof(escaped_error));

    int len = snprintf(wasm_result_buffer + 4, RESULT_BUFFER_SIZE - 4,
        "{\"ok\":false,\"error\":\"%s\",\"phase\":\"%s\"}",
        escaped_error, phase);

    wasm_result_buffer[0] = (len >> 0) & 0xFF;
    wasm_result_buffer[1] = (len >> 8) & 0xFF;
    wasm_result_buffer[2] = (len >> 16) & 0xFF;
    wasm_result_buffer[3] = (len >> 24) & 0xFF;

    return (uint32_t)(uintptr_t)wasm_result_buffer;
}

// ============================================================================
// Parse and Type Check Pipeline
// ============================================================================

/**
 * Parse and type-check source code.
 * Note: This compiler generates C code, it doesn't interpret directly.
 * Returns parse/type errors or success.
 */
static int wasm_parse_and_check(const char* source, const char* filename) {
    // Create Config for Check mode
    SigilValue _config_values[9];
    _config_values[0] = sigil_array(0);  // input_files (empty)
    _config_values[1] = sigil_null();    // output_file
    _config_values[2] = CompileMode____Check;  // Check mode
    _config_values[3] = sigil_bool(false);   // verbose
    _config_values[4] = sigil_bool(false);   // debug
    _config_values[5] = sigil_int(0);        // opt_level
    _config_values[6] = sigil_bool(true);    // evidence_checks
    _config_values[7] = sigil_array(0);      // tome_paths
    _config_values[8] = sigil_bool(false);   // resolve_tomes
    static const char* _config_names[9] = {
        "input_files", "output_file", "mode", "verbose", "debug",
        "opt_level", "evidence_checks", "tome_paths", "resolve_tomes"
    };
    SigilValue config = sigil_struct("Config", _config_names, _config_values, 9);

    // Create Driver
    SigilValue driver = sigil_Driver____new(config);

    // Parse source directly
    SigilValue file_path = sigil_string(filename);
    SigilValue source_val = sigil_string(source);
    SigilValue parse_result = sigil_Driver____parse(&driver, file_path, source_val);

    // Check parse result
    if (parse_result.tag == TAG_REF && parse_result.v.ptr) {
        parse_result = *((SigilValue*)parse_result.v.ptr);
    }

    if (!sigil_is_ok(parse_result)) {
        sigil_Driver____print_errors(driver);
        return 1;
    }

    SigilValue ast = *(SigilValue*)parse_result.v.ptr;

    // Create TypeChecker and collect types
    SigilValue checker = sigil_TypeChecker____new();
    SigilValue items = sigil_struct_field(ast, "items");

    // Collect type definitions first
    for (size_t k = 0; k < items.v.arr.len; k++) {
        SigilValue item = items.v.arr.data[k];
        SigilValue node = sigil_struct_field(item, "node");
        sigil_TypeChecker____collect_type_def(&checker, node);
    }

    // Type check each item
    int has_errors = 0;
    for (size_t k = 0; k < items.v.arr.len; k++) {
        SigilValue item = items.v.arr.data[k];
        SigilValue node = sigil_struct_field(item, "node");
        SigilValue check_result = sigil_TypeChecker____check_item(&checker, node);
        if (check_result.tag == TAG_REF && check_result.v.ptr) {
            check_result = *((SigilValue*)check_result.v.ptr);
        }
        if (sigil_is_err(check_result)) {
            has_errors = 1;
            SigilValue errs = *(SigilValue*)check_result.v.ptr;
            for (size_t e = 0; e < errs.v.arr.len; e++) {
                SigilValue err = errs.v.arr.data[e];
                SigilValue msg = sigil_struct_field(err, "message");
                wasm_captured_fprintf(stderr, "type error: %s\n",
                    msg.tag == TAG_STRING ? msg.v.s : "<unknown>");
            }
        }
    }

    if (has_errors) {
        return 1;
    }

    // Success - show what was parsed
    wasm_captured_printf("Parsed %zu item(s). Type check passed.\n", items.v.arr.len);

    return 0;
}

// ============================================================================
// WASM Entry Points
// ============================================================================

/**
 * Parse and check Sigil source code.
 * Since this compiler generates C (doesn't interpret), we parse and type-check.
 * @param src_ptr Pointer to source string in WASM memory
 * @param src_len Length of source string
 * @return Pointer to length-prefixed JSON result
 */
__attribute__((export_name("wasm_compile_and_run")))
uint32_t wasm_compile_and_run(uint32_t src_ptr, uint32_t src_len) {
    wasm_clear_buffers();
    wasm_capture_active = true;

    // Copy source to null-terminated buffer
    char* source = malloc(src_len + 1);
    if (!source) {
        wasm_capture_active = false;
        return build_error_result("init", "Out of memory");
    }
    memcpy(source, (void*)(uintptr_t)src_ptr, src_len);
    source[src_len] = '\0';

    int exit_code = 0;

    // Set up exit handler
    if (setjmp(wasm_exit_jump) == 0) {
        exit_code = wasm_parse_and_check(source, "<stdin>");
    } else {
        // Caught exit()
        exit_code = wasm_exit_code;
    }

    free(source);
    wasm_capture_active = false;

    if (exit_code != 0) {
        return build_error_result("check", NULL);
    }

    return build_success_result(exit_code, 0.0);
}

/**
 * Check syntax without executing.
 * @param src_ptr Pointer to source string in WASM memory
 * @param src_len Length of source string
 * @return Pointer to length-prefixed JSON result
 */
__attribute__((export_name("wasm_check_syntax")))
uint32_t wasm_check_syntax(uint32_t src_ptr, uint32_t src_len) {
    // Same as compile_and_run since we can't actually execute
    return wasm_compile_and_run(src_ptr, src_len);
}

// ============================================================================
// Emscripten Support
// ============================================================================

#ifdef __EMSCRIPTEN__
#include <emscripten.h>

// Keep the WASM module alive
EM_JS(void, keep_alive, (), {
    // Prevent module from being garbage collected
});
#endif
