/*
 * Sigil Runtime Library
 *
 * Provides runtime functions for AOT-compiled Sigil programs.
 * This gets linked with the compiled object file to create
 * a standalone native executable.
 *
 * Platform Support:
 * - Linux (POSIX)
 * - macOS (POSIX)
 * - Windows (Win32 API)
 *
 * Build Options:
 * - SIGIL_RUNTIME_LIB_ONLY: Exclude main() for library/testing use
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>

/* Platform-specific includes */
#ifdef _WIN32
#include <windows.h>
#else
#include <sys/time.h>
#endif

/* ============================================================================
 * Time Functions
 * ============================================================================ */

/* Get current time in milliseconds since Unix epoch */
int64_t sigil_now(void) {
#ifdef _WIN32
    /* Windows: Use GetSystemTimeAsFileTime
     * FILETIME is 100-nanosecond intervals since Jan 1, 1601.
     * Convert to milliseconds since Unix epoch (Jan 1, 1970).
     */
    FILETIME ft;
    GetSystemTimeAsFileTime(&ft);
    ULARGE_INTEGER ull;
    ull.LowPart = ft.dwLowDateTime;
    ull.HighPart = ft.dwHighDateTime;
    /* Subtract Windows epoch offset (1601 to 1970) and convert to ms */
    /* 116444736000000000 = 100-ns intervals from 1601 to 1970 */
    return (int64_t)((ull.QuadPart - 116444736000000000ULL) / 10000);
#else
    /* POSIX: Use gettimeofday */
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (int64_t)(tv.tv_sec * 1000 + tv.tv_usec / 1000);
#endif
}

/* ============================================================================
 * Print Functions
 * ============================================================================ */

/* Print an integer value */
void sigil_print_int(int64_t value) {
    printf("%lld\n", (long long)value);
}

/* Print a float value */
void sigil_print_float(double value) {
    printf("%g\n", value);
}

/* Print a string */
void sigil_print_str(const char* str) {
    printf("%s\n", str);
}

/* Get string length */
int64_t sigil_strlen(const char* str) {
    if (str == NULL) return 0;
    return (int64_t)strlen(str);
}

/* ============================================================================
 * Memory Functions
 * ============================================================================ */

/* Allocate memory */
void* sigil_alloc(int64_t size) {
    return malloc((size_t)size);
}

/* Reallocate memory */
void* sigil_realloc(void* ptr, int64_t new_size) {
    return realloc(ptr, (size_t)new_size);
}

/* Free memory */
void sigil_free(void* ptr) {
    free(ptr);
}

/* ============================================================================
 * Vec Operations (simple fixed-size array on heap)
 * Vec is represented as: ptr to {len: i64, capacity: i64, data: i64[]}
 * ============================================================================ */

/* Create a new Vec with given capacity */
void* sigil_vec_new(int64_t capacity) {
    if (capacity < 4) capacity = 4;
    // Allocate: 2 i64s for len/cap + data
    size_t size = 2 * sizeof(int64_t) + (size_t)capacity * sizeof(int64_t);
    int64_t* vec = (int64_t*)malloc(size);
    if (vec) {
        vec[0] = 0;         // len
        vec[1] = capacity;  // capacity
    }
    return vec;
}

/* Push a value to the Vec */
void sigil_vec_push(void* vec_ptr, int64_t value) {
    if (!vec_ptr) return;
    int64_t* vec = (int64_t*)vec_ptr;
    int64_t len = vec[0];
    int64_t cap = vec[1];

    if (len >= cap) {
        // Need to grow - this would require returning new pointer
        // For simplicity, we panic (don't support growing yet)
        return;
    }

    vec[2 + len] = value;  // data starts at index 2
    vec[0] = len + 1;
}

/* Get a value from the Vec */
int64_t sigil_vec_get(void* vec_ptr, int64_t index) {
    if (!vec_ptr) return 0;
    int64_t* vec = (int64_t*)vec_ptr;
    int64_t len = vec[0];

    if (index < 0 || index >= len) return 0;

    return vec[2 + index];
}

/* Get Vec length */
int64_t sigil_vec_len(void* vec_ptr) {
    if (!vec_ptr) return 0;
    return ((int64_t*)vec_ptr)[0];
}


/* ============================================================================
 * Option Operations (nullable value wrapper)
 * Option is represented as: NULL for None, or ptr to i64 value for Some
 * ============================================================================ */

/* Create Some(value) - allocates and stores value */
void* sigil_option_some(int64_t value) {
    int64_t* ptr = (int64_t*)malloc(sizeof(int64_t));
    if (ptr) {
        *ptr = value;
    }
    return ptr;
}

/* Create None - returns NULL */
void* sigil_option_none(void) {
    return NULL;
}

/* Check if Option is Some */
int64_t sigil_option_is_some(void* opt_ptr) {
    return opt_ptr != NULL ? 1 : 0;
}

/* Check if Option is None */
int64_t sigil_option_is_none(void* opt_ptr) {
    return opt_ptr == NULL ? 1 : 0;
}

/* Unwrap Option (returns value, undefined behavior if None) */
int64_t sigil_option_unwrap(void* opt_ptr) {
    if (!opt_ptr) {
        fprintf(stderr, "Error: unwrap called on None\n");
        return 0;
    }
    return *((int64_t*)opt_ptr);
}

/* Unwrap Option with default value */
int64_t sigil_option_unwrap_or(void* opt_ptr, int64_t default_val) {
    if (!opt_ptr) {
        return default_val;
    }
    return *((int64_t*)opt_ptr);
}

/* Free Option (if Some) */
void sigil_option_free(void* opt_ptr) {
    if (opt_ptr) {
        free(opt_ptr);
    }
}


/* ============================================================================
 * String Operations (heap-allocated growable string)
 * String is represented as: ptr to {len: i64, capacity: i64, data: char[]}
 * ============================================================================ */

/* Create a new empty String with given capacity */
void* sigil_string_new(int64_t capacity) {
    if (capacity < 16) capacity = 16;
    size_t size = 2 * sizeof(int64_t) + (size_t)capacity + 1;
    int64_t* str = (int64_t*)malloc(size);
    if (str) {
        str[0] = 0;
        str[1] = capacity;
        ((char*)(str + 2))[0] = '\0';
    }
    return str;
}

/* Create a String from a C string literal */
void* sigil_string_from(const char* src) {
    if (!src) return sigil_string_new(16);
    int64_t len = (int64_t)strlen(src);
    int64_t capacity = len + 16;
    size_t size = 2 * sizeof(int64_t) + (size_t)capacity + 1;
    int64_t* str = (int64_t*)malloc(size);
    if (str) {
        str[0] = len;
        str[1] = capacity;
        memcpy((char*)(str + 2), src, len + 1);
    }
    return str;
}

/* Get String length */
int64_t sigil_string_len(void* str_ptr) {
    if (!str_ptr) return 0;
    return ((int64_t*)str_ptr)[0];
}

/* Get pointer to String data (null-terminated) */
const char* sigil_string_as_ptr(void* str_ptr) {
    if (!str_ptr) return "";
    return (const char*)((int64_t*)str_ptr + 2);
}

/* Push a character to the String */
void sigil_string_push_char(void* str_ptr, int64_t ch) {
    if (!str_ptr) return;
    int64_t* str = (int64_t*)str_ptr;
    int64_t len = str[0];
    int64_t cap = str[1];
    if (len >= cap) return;
    char* data = (char*)(str + 2);
    data[len] = (char)ch;
    data[len + 1] = '\0';
    str[0] = len + 1;
}

/* Concatenate two Strings, returns new String */
void* sigil_string_concat(void* str1_ptr, void* str2_ptr) {
    int64_t len1 = str1_ptr ? ((int64_t*)str1_ptr)[0] : 0;
    int64_t len2 = str2_ptr ? ((int64_t*)str2_ptr)[0] : 0;
    int64_t total = len1 + len2;
    int64_t capacity = total + 16;
    size_t size = 2 * sizeof(int64_t) + (size_t)capacity + 1;
    int64_t* result = (int64_t*)malloc(size);
    if (!result) return NULL;
    result[0] = total;
    result[1] = capacity;
    char* data = (char*)(result + 2);
    if (str1_ptr && len1 > 0) {
        memcpy(data, (char*)((int64_t*)str1_ptr + 2), len1);
    }
    if (str2_ptr && len2 > 0) {
        memcpy(data + len1, (char*)((int64_t*)str2_ptr + 2), len2);
    }
    data[total] = '\0';
    return result;
}

/* Print a String */
void sigil_string_print(void* str_ptr) {
    if (!str_ptr) { printf("\n"); return; }
    const char* data = (const char*)((int64_t*)str_ptr + 2);
    printf("%s\n", data);
}

/* Free a String */
void sigil_string_free(void* str_ptr) {
    if (str_ptr) free(str_ptr);
}

/* ============================================================================
 * Math Functions (operate on i64 bits representing f64)
 *
 * These functions take f64 values encoded as i64 bit patterns and return
 * the result encoded the same way. This allows LLVM IR to pass floats
 * through i64 registers uniformly.
 * ============================================================================ */

/* Helper: convert i64 bits to double */
static inline double bits_to_double(int64_t bits) {
    union { int64_t i; double d; } u;
    u.i = bits;
    return u.d;
}

/* Helper: convert double to i64 bits */
static inline int64_t double_to_bits(double d) {
    union { int64_t i; double d; } u;
    u.d = d;
    return u.i;
}

/* Square root */
int64_t sigil_sqrt(int64_t x) {
    return double_to_bits(sqrt(bits_to_double(x)));
}

/* Sine */
int64_t sigil_sin(int64_t x) {
    return double_to_bits(sin(bits_to_double(x)));
}

/* Cosine */
int64_t sigil_cos(int64_t x) {
    return double_to_bits(cos(bits_to_double(x)));
}

/* Tangent */
int64_t sigil_tan(int64_t x) {
    return double_to_bits(tan(bits_to_double(x)));
}

/* Arc sine */
int64_t sigil_asin(int64_t x) {
    return double_to_bits(asin(bits_to_double(x)));
}

/* Arc cosine */
int64_t sigil_acos(int64_t x) {
    return double_to_bits(acos(bits_to_double(x)));
}

/* Arc tangent */
int64_t sigil_atan(int64_t x) {
    return double_to_bits(atan(bits_to_double(x)));
}

/* Arc tangent of y/x */
int64_t sigil_atan2(int64_t y, int64_t x) {
    return double_to_bits(atan2(bits_to_double(y), bits_to_double(x)));
}

/* Hyperbolic sine */
int64_t sigil_sinh(int64_t x) {
    return double_to_bits(sinh(bits_to_double(x)));
}

/* Hyperbolic cosine */
int64_t sigil_cosh(int64_t x) {
    return double_to_bits(cosh(bits_to_double(x)));
}

/* Hyperbolic tangent */
int64_t sigil_tanh(int64_t x) {
    return double_to_bits(tanh(bits_to_double(x)));
}

/* Exponential (e^x) */
int64_t sigil_exp(int64_t x) {
    return double_to_bits(exp(bits_to_double(x)));
}

/* Exponential minus 1 (e^x - 1, more accurate for small x) */
int64_t sigil_expm1(int64_t x) {
    return double_to_bits(expm1(bits_to_double(x)));
}

/* Natural logarithm */
int64_t sigil_ln(int64_t x) {
    return double_to_bits(log(bits_to_double(x)));
}

/* Natural logarithm of (1 + x), more accurate for small x */
int64_t sigil_ln1p(int64_t x) {
    return double_to_bits(log1p(bits_to_double(x)));
}

/* Base-2 logarithm */
int64_t sigil_log2(int64_t x) {
    return double_to_bits(log2(bits_to_double(x)));
}

/* Base-10 logarithm */
int64_t sigil_log10(int64_t x) {
    return double_to_bits(log10(bits_to_double(x)));
}

/* Power (x^y) */
int64_t sigil_pow(int64_t x, int64_t y) {
    return double_to_bits(pow(bits_to_double(x), bits_to_double(y)));
}

/* Floor */
int64_t sigil_floor(int64_t x) {
    return double_to_bits(floor(bits_to_double(x)));
}

/* Ceiling */
int64_t sigil_ceil(int64_t x) {
    return double_to_bits(ceil(bits_to_double(x)));
}

/* Round to nearest integer */
int64_t sigil_round(int64_t x) {
    return double_to_bits(round(bits_to_double(x)));
}

/* Truncate toward zero */
int64_t sigil_trunc(int64_t x) {
    return double_to_bits(trunc(bits_to_double(x)));
}

/* Absolute value (float) */
int64_t sigil_fabs(int64_t x) {
    return double_to_bits(fabs(bits_to_double(x)));
}

/* Floating-point modulo */
int64_t sigil_fmod(int64_t x, int64_t y) {
    return double_to_bits(fmod(bits_to_double(x), bits_to_double(y)));
}

/* Copy sign of y to x */
int64_t sigil_copysign(int64_t x, int64_t y) {
    return double_to_bits(copysign(bits_to_double(x), bits_to_double(y)));
}

/* Hypotenuse (sqrt(x^2 + y^2)) */
int64_t sigil_hypot(int64_t x, int64_t y) {
    return double_to_bits(hypot(bits_to_double(x), bits_to_double(y)));
}

/* ============================================================================
 * Integer Math Functions
 * ============================================================================ */

/* Absolute value (integer) */
int64_t sigil_abs(int64_t x) {
    return x < 0 ? -x : x;
}

/* Minimum of two integers */
int64_t sigil_min(int64_t a, int64_t b) {
    return a < b ? a : b;
}

/* Maximum of two integers */
int64_t sigil_max(int64_t a, int64_t b) {
    return a > b ? a : b;
}

/* Clamp value to range [lo, hi] */
int64_t sigil_clamp(int64_t x, int64_t lo, int64_t hi) {
    if (x < lo) return lo;
    if (x > hi) return hi;
    return x;
}

/* Sign of integer (-1, 0, or 1) */
int64_t sigil_sign(int64_t x) {
    if (x < 0) return -1;
    if (x > 0) return 1;
    return 0;
}

/* ============================================================================
 * Entry Point (excluded when building as library)
 * ============================================================================ */


/* ============================================================================
 * File I/O Functions
 * ============================================================================ */

/* Open a file, returns file handle as i64 (0 on failure) */
int64_t sigil_file_open(const char* path, const char* mode) {
    if (!path || !mode) return 0;
    FILE* f = fopen(path, mode);
    return (int64_t)(uintptr_t)f;
}

/* Read from file into buffer, returns bytes read */
int64_t sigil_file_read(int64_t handle, void* buffer, int64_t size) {
    if (!handle || !buffer || size <= 0) return 0;
    FILE* f = (FILE*)(uintptr_t)handle;
    return (int64_t)fread(buffer, 1, (size_t)size, f);
}

/* Write to file from buffer, returns bytes written */
int64_t sigil_file_write(int64_t handle, const void* buffer, int64_t size) {
    if (!handle || !buffer || size <= 0) return 0;
    FILE* f = (FILE*)(uintptr_t)handle;
    return (int64_t)fwrite(buffer, 1, (size_t)size, f);
}

/* Close a file */
void sigil_file_close(int64_t handle) {
    if (handle) {
        fclose((FILE*)(uintptr_t)handle);
    }
}

/* Read entire file as String, returns String ptr (0 on failure) */
void* sigil_file_read_all(const char* path) {
    if (!path) return NULL;
    FILE* f = fopen(path, "rb");
    if (!f) return NULL;
    
    fseek(f, 0, SEEK_END);
    long size = ftell(f);
    fseek(f, 0, SEEK_SET);
    
    if (size <= 0) {
        fclose(f);
        return sigil_string_new(16);
    }
    
    // Allocate String with exact size
    void* str = sigil_string_new(size + 1);
    if (!str) {
        fclose(f);
        return NULL;
    }
    
    int64_t* str_header = (int64_t*)str;
    char* data = (char*)(str_header + 2);
    size_t read = fread(data, 1, (size_t)size, f);
    data[read] = '\0';
    str_header[0] = (int64_t)read;
    
    fclose(f);
    return str;
}

/* Write String to file, returns 1 on success */
int64_t sigil_file_write_all(const char* path, void* str_ptr) {
    if (!path || !str_ptr) return 0;
    FILE* f = fopen(path, "wb");
    if (!f) return 0;
    
    int64_t* str_header = (int64_t*)str_ptr;
    int64_t len = str_header[0];
    const char* data = (const char*)(str_header + 2);
    
    size_t written = fwrite(data, 1, (size_t)len, f);
    fclose(f);
    
    return (written == (size_t)len) ? 1 : 0;
}

/* Check if file exists */
int64_t sigil_file_exists(const char* path) {
    if (!path) return 0;
    FILE* f = fopen(path, "r");
    if (f) {
        fclose(f);
        return 1;
    }
    return 0;
}

/* ============================================================================
 * System Functions
 * ============================================================================ */

/* Exit with status code */
void sigil_exit(int64_t code) {
    exit((int)code);
}

/* Get environment variable as String (0 if not set) */
void* sigil_getenv(const char* name) {
    if (!name) return NULL;
    const char* value = getenv(name);
    if (!value) return NULL;
    return sigil_string_from(value);
}

#ifndef SIGIL_RUNTIME_LIB_ONLY

/* Entry point - calls the Sigil main function */
extern int64_t main_sigil(void);

int main(int argc, char** argv) {
    (void)argc;
    (void)argv;
    int64_t result = main_sigil();
    return (int)result;
}

#endif /* SIGIL_RUNTIME_LIB_ONLY */
