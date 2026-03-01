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

/* Get current time in microseconds since Unix epoch */
int64_t sigil_now_micros(void) {
#ifdef _WIN32
    /* Windows: Use GetSystemTimeAsFileTime (100-ns intervals) */
    FILETIME ft;
    GetSystemTimeAsFileTime(&ft);
    ULARGE_INTEGER ull;
    ull.LowPart = ft.dwLowDateTime;
    ull.HighPart = ft.dwHighDateTime;
    /* Convert to microseconds since Unix epoch */
    return (int64_t)((ull.QuadPart - 116444736000000000ULL) / 10);
#else
    /* POSIX: Use gettimeofday */
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (int64_t)(tv.tv_sec * 1000000 + tv.tv_usec);
#endif
}

/* ============================================================================
 * Print Functions
 * ============================================================================ */

/* Print an integer value (with newline) */
void sigil_print_int(int64_t value) {
    printf("%lld\n", (long long)value);
}

/* Write an integer value (no newline) - for format strings */
void sigil_write_int(int64_t value) {
    printf("%lld", (long long)value);
}

/* Print a float value (with newline) */
void sigil_print_float(double value) {
    printf("%g\n", value);
}

/* Write a float value (no newline) - for format strings */
void sigil_write_float(double value) {
    printf("%g", value);
}

/* Print a string (with newline) */
void sigil_print_str(const char* str) {
    printf("%s\n", str);
}

/* Write a string (no newline) - for format strings */
void sigil_write_str(const char* str) {
    printf("%s", str);
    fflush(stdout);
}

/* Jormungandr-compatible print functions */
void print(const char* str) {
    printf("%s", str);
    fflush(stdout);
}

void println(const char* str) {
    printf("%s\n", str);
}

void eprint(const char* str) {
    fprintf(stderr, "%s", str);
    fflush(stderr);
}

void eprintln(const char* str) {
    fprintf(stderr, "%s\n", str);
}

/* Print just a newline */
void sigil_print_newline(void) {
    printf("\n");
    fflush(stdout);
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
    // G139: Use calloc to zero-initialize; prevents garbage len fields in
    // HashMap/enum/struct allocations from causing spurious sigil_vec_get crashes.
    return calloc(1, (size_t)size);
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
 * Vec Operations (growable array on heap)
 * Vec is represented as: ptr to {len: i64, capacity: i64, data_ptr: i64*}
 * The data is stored in a separate heap allocation pointed to by data_ptr.
 * This allows growing without invalidating the Vec pointer itself.
 *
 * TODO: Consider moving Vec to pure LLVM IR generation to eliminate runtime
 * dependency. Would require implementing growth logic directly in codegen.
 * ============================================================================ */

typedef struct {
    int64_t len;
    int64_t capacity;
    int64_t* data;  // separate heap allocation for data
} SigilVec;

/* Create a new Vec with given capacity */
void* sigil_vec_new(int64_t capacity) {
    if (capacity < 8) capacity = 8;

    SigilVec* vec = (SigilVec*)malloc(sizeof(SigilVec));
    if (!vec) return NULL;

    vec->len = 0;
    vec->capacity = capacity;
    vec->data = (int64_t*)malloc((size_t)capacity * sizeof(int64_t));
    if (!vec->data) {
        free(vec);
        return NULL;
    }

    return vec;
}

/* Push a value to the Vec (grows automatically if needed) */
void sigil_vec_push(void* vec_ptr, int64_t value) {
    if (!vec_ptr) return;
    SigilVec* vec = (SigilVec*)vec_ptr;

    // Grow if needed
    if (vec->len >= vec->capacity) {
        int64_t new_cap = vec->capacity * 2;
        int64_t* new_data = (int64_t*)realloc(vec->data, (size_t)new_cap * sizeof(int64_t));
        if (!new_data) return;  // allocation failed, silently fail
        vec->data = new_data;
        vec->capacity = new_cap;
    }

    vec->data[vec->len] = value;
    vec->len++;
}

/* Get a value from the Vec */
int64_t sigil_vec_get(void* vec_ptr, int64_t index) {
    if (!vec_ptr) return 0;
    SigilVec* vec = (SigilVec*)vec_ptr;

    if (index < 0 || index >= vec->len) return 0;

    return vec->data[index];
}

/* Set a value in the Vec */
void sigil_vec_set(void* vec_ptr, int64_t index, int64_t value) {
    if (!vec_ptr) return;
    SigilVec* vec = (SigilVec*)vec_ptr;

    if (index < 0 || index >= vec->len) return;

    vec->data[index] = value;
}

/* Set Vec<f32> element from f64 bits in i64.
 * Interprets f64_bits as a double, truncates to f32, stores f32 bits zero-extended to i64.
 * Used when Sigil scalar float arithmetic (f64 bits in i64) needs to be stored into Vec<f32>.
 * G-F32-ASSIGN: Fixes NaN in backward pass caused by f64 bits stored in f32 Vec slots. */
void sigil_vec_set_f32_from_i64(void* vec_ptr, int64_t index, int64_t f64_bits) {
    if (!vec_ptr) return;
    SigilVec* vec = (SigilVec*)vec_ptr;
    if (index < 0 || index >= vec->len) return;
    double f64_val;
    memcpy(&f64_val, &f64_bits, sizeof(double));
    float f32_val = (float)f64_val;
    uint32_t bits;
    memcpy(&bits, &f32_val, sizeof(float));
    vec->data[index] = (int64_t)bits;  /* zero-extend f32 bits to i64 */
}

/* Get Vec length */
int64_t sigil_vec_len(void* vec_ptr) {
    if (!vec_ptr) return 0;
    return ((SigilVec*)vec_ptr)->len;
}

/* Get Vec capacity */
int64_t sigil_vec_capacity(void* vec_ptr) {
    if (!vec_ptr) return 0;
    return ((SigilVec*)vec_ptr)->capacity;
}

/* Free a Vec and its data */
void sigil_vec_free(void* vec_ptr) {
    if (!vec_ptr) return;
    SigilVec* vec = (SigilVec*)vec_ptr;
    free(vec->data);
    free(vec);
}

/* G75: Get raw pointer to Vec data for slice conversion */
void* sigil_vec_u8_as_ptr(void* vec_ptr) {
    if (!vec_ptr) return NULL;
    return (void*)((SigilVec*)vec_ptr)->data;
}

/* Clone a Vec (deep copy) */
void* sigil_vec_clone(void* vec_ptr) {
    if (!vec_ptr) return NULL;
    SigilVec* src = (SigilVec*)vec_ptr;

    // Create new Vec with same capacity as source length
    SigilVec* dest = (SigilVec*)malloc(sizeof(SigilVec));
    if (!dest) return NULL;

    int64_t new_cap = src->len < 8 ? 8 : src->len;
    dest->data = (int64_t*)malloc((size_t)new_cap * sizeof(int64_t));
    if (!dest->data) {
        free(dest);
        return NULL;
    }

    // Copy elements
    dest->len = src->len;
    dest->capacity = new_cap;
    for (int64_t i = 0; i < src->len; i++) {
        dest->data[i] = src->data[i];
    }

    return dest;
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

/* Clone a String (inline layout: [len, capacity, data...])
 * The string layout stores char data inline at offset 16, NOT as a pointer.
 * sigil_vec_clone cannot be used on Strings. */
void* sigil_string_clone(void* str_ptr) {
    if (!str_ptr) return NULL;
    int64_t* src = (int64_t*)str_ptr;
    int64_t len = src[0];
    int64_t capacity = src[1];
    size_t size = 2 * sizeof(int64_t) + (size_t)capacity + 1;
    void* dest = malloc(size);
    if (dest) memcpy(dest, src, size);
    return dest;
}

/* Free a String */
void sigil_string_free(void* str_ptr) {
    if (str_ptr) free(str_ptr);
}

/* ============================================================================
 * File I/O Functions
 * ============================================================================ */

/* Read entire file into a Sigil String (inline layout: [len, capacity, data...])
 * Returns pointer to new String, or NULL on error
 */
void* sigil_fs_read(const char* path) {
    FILE* f = fopen(path, "rb");
    if (!f) {
        fprintf(stderr, "Error: Could not open file: %s\n", path);
        return NULL;
    }

    /* Get file size */
    fseek(f, 0, SEEK_END);
    long size = ftell(f);
    fseek(f, 0, SEEK_SET);

    /* Allocate String: [len, capacity, data...] */
    int64_t capacity = size + 16;  /* Extra space */
    size_t alloc_size = 2 * sizeof(int64_t) + (size_t)capacity + 1;
    int64_t* str = (int64_t*)malloc(alloc_size);
    if (!str) {
        fclose(f);
        return NULL;
    }

    /* Read file */
    char* data = (char*)(str + 2);
    size_t read = fread(data, 1, size, f);
    fclose(f);

    data[read] = '\0';
    str[0] = (int64_t)read;      /* len */
    str[1] = capacity;            /* capacity */

    return str;
}

/* Get bytes pointer from Sigil String (compatible with String layout)
 * Returns pointer to the underlying byte data
 */
const char* sigil_rust_string_as_bytes(void* str_ptr) {
    if (!str_ptr) return NULL;
    /* String layout is [len, capacity, data...] */
    return (const char*)((int64_t*)str_ptr + 2);
}

/* Create a substring from a Sigil String
 * Returns a new String containing the slice [start, end)
 */
void* sigil_rust_string_slice(void* str_ptr, int64_t start, int64_t end) {
    if (!str_ptr) return sigil_string_new(16);

    int64_t* src = (int64_t*)str_ptr;
    int64_t src_len = src[0];
    const char* src_data = (const char*)(src + 2);

    /* Clamp indices */
    if (start < 0) start = 0;
    if (end > src_len) end = src_len;
    if (start > end) start = end;

    int64_t slice_len = end - start;
    int64_t capacity = slice_len + 16;
    size_t alloc_size = 2 * sizeof(int64_t) + (size_t)capacity + 1;
    int64_t* dest = (int64_t*)malloc(alloc_size);
    if (!dest) return NULL;

    char* dest_data = (char*)(dest + 2);
    memcpy(dest_data, src_data + start, slice_len);
    dest_data[slice_len] = '\0';
    dest[0] = slice_len;      /* len */
    dest[1] = capacity;       /* capacity */

    return dest;
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

/* PI constant */
int64_t sigil_pi(void) {
    return double_to_bits(3.14159265358979323846);
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
 * Raw memory helpers — bypass Sigil LLVM codegen nested struct access bugs
 *
 * Background: Sigil LLVM codegen for `t.storage.data_ptr` generates:
 *   load(t_ptr + 0)             → reads Arc inner ptr (field 0 of StoragePtr)
 *   load(Arc_inner_ptr + 8)     → reads StorageInner.len (wrong)
 * Instead of the correct:
 *   load(t_ptr + 8)             → reads StoragePtr.data_ptr (field 1)
 *
 * These helpers let Sigil pass the struct address as i64 and read the
 * correct field directly via raw pointer arithmetic.
 * ============================================================================ */

/* Read i64 at ptr + byte_offset. Used by tensor_raw_ptr to get StoragePtr.data_ptr */
int64_t sigil_read_i64_at_offset(int64_t ptr, int64_t byte_offset) {
    if (!ptr) return 0;
    return *((int64_t*)(ptr + byte_offset));
}

/* ============================================================================
 * In-Place SGD Optimizer
 *
 * Sigil Vec<f32> ABI: Vec<T> is a POINTER (i64) to a heap-allocated struct:
 *   typedef struct { int64_t len; int64_t capacity; int64_t* data; } SigilVec;
 * f32 bits are stored in the low 4 bytes of each int64_t element.
 *
 * lr_bits / clip_bits: f32 passed as int64_t (Sigil's universal representation).
 *
 * This avoids allocating new Vec<f32> for each weight/gradient per step —
 * fixing the OOM leak from the previous allocate-and-abandon approach.
 * ============================================================================ */

/* Clamped f32 exp with Sigil ABI: argument and return are f64 bits packed in int64_t.
 * Sigil stores f32 arithmetic results as f64 bits in i64 (fp_extend f32→f64, bitcast f64→i64).
 * Returns 0.0 for x < -20 (handles -1e9 causal mask safely), expf(x) otherwise.
 * Prevents 10-term Taylor series overflow → NaN without touching manual_exp_f32 inlining. */
int64_t sigil_expf32_clamped(int64_t x_bits) {
    /* Sigil passes f32 values as i64 with f32 bits in the LOW 32 bits (zero-extended).
     * Extract via 32-bit memcpy into float, not 64-bit memcpy into double. */
    int32_t lo32 = (int32_t)(x_bits & 0xFFFFFFFFLL);
    float x;
    memcpy(&x, &lo32, 4);
    float result;
    if (x < -20.0f) {
        result = 0.0f;
    } else {
        result = expf(x);
    }
    /* Return f32 result as i64 with f32 bits in low 32 (zero-extended). */
    int32_t result32;
    memcpy(&result32, &result, 4);
    return (int64_t)(uint32_t)result32;
}

/* Softmax in-place on a Vec<f32> passed as SigilVec pointer.
 * Processes the first n elements: find max, subtract, exp, normalize.
 * Uses hardware expf (fast) vs Sigil's 10-term Taylor loop (slow).
 * n <= 0 is a no-op. Clamps x-max < -20 to 0 for causal mask safety.
 *
 * ABI note: Vec<f32> stores each f32 in the LOW 4 bytes of an int64_t slot
 * (8 bytes/element). Must read/write via memcpy(&f, &data[i], 4), NOT float*
 * indexing (which strides by 4 bytes, reading zeros from the high half). */
void sigil_softmax_inplace(int64_t v_ptr, int64_t n) {
    if (n <= 0 || !v_ptr) return;
    typedef struct { int64_t len; int64_t cap; int64_t* data; } SigilVecSM;
    SigilVecSM* v = (SigilVecSM*)(uintptr_t)v_ptr;
    int64_t* d = v->data;
    /* Find max for numerical stability */
    float max_v;
    memcpy(&max_v, &d[0], sizeof(float));
    for (int64_t i = 1; i < n; i++) {
        float xi;
        memcpy(&xi, &d[i], sizeof(float));
        if (xi > max_v) max_v = xi;
    }
    /* Compute exp(x - max) in-place, accumulate sum */
    float sum = 0.0f;
    for (int64_t i = 0; i < n; i++) {
        float xi;
        memcpy(&xi, &d[i], sizeof(float));
        float e = xi - max_v;
        float ei = (e < -20.0f) ? 0.0f : expf(e);
        sum += ei;
        int64_t slot = 0;
        memcpy(&slot, &ei, sizeof(float));
        d[i] = slot;
    }
    /* Normalize */
    if (sum > 0.0f) {
        float inv_sum = 1.0f / sum;
        for (int64_t i = 0; i < n; i++) {
            float xi;
            memcpy(&xi, &d[i], sizeof(float));
            xi *= inv_sum;
            int64_t slot = 0;
            memcpy(&slot, &xi, sizeof(float));
            d[i] = slot;
        }
    }
}

/* Vectorized SwiGLU helpers — avoid per-element Sigil→C call overhead.
 * All Vec<f32> pointers follow the same int64_t-slot ABI as sigil_softmax_inplace.
 *
 * sigil_mul_silu_into(a, b, out, n):
 *   out[i] = a[i] * silu(b[i])   where silu(x) = x / (1 + expf(-x))
 *   Used for: swiglu forward (gate * silu(up)) and d_gate backward (d_mid * silu(up))
 *
 * sigil_d_up_into(d_mid, gate, up, out, n):
 *   sg = sigmoid(up[i]) = 1/(1+expf(-up[i]))
 *   dsilu = sg * (1 + up[i]*(1-sg))
 *   out[i] = d_mid[i] * gate[i] * dsilu
 *   Used for: swiglu d_up backward
 */
static inline void _sv_read(int64_t* data, int64_t i, float* f) {
    memcpy(f, &data[i], sizeof(float));
}
static inline void _sv_write(int64_t* data, int64_t i, float f) {
    int64_t slot = 0;
    memcpy(&slot, &f, sizeof(float));
    data[i] = slot;
}

void sigil_mul_silu_into(int64_t a_ptr, int64_t b_ptr, int64_t out_ptr, int64_t n) {
    typedef struct { int64_t len; int64_t cap; int64_t* data; } SV;
    SV* a = (SV*)(uintptr_t)a_ptr;
    SV* b = (SV*)(uintptr_t)b_ptr;
    SV* o = (SV*)(uintptr_t)out_ptr;
    for (int64_t i = 0; i < n; i++) {
        float ai, bi;
        _sv_read(a->data, i, &ai);
        _sv_read(b->data, i, &bi);
        float silu_bi = bi / (1.0f + expf(-bi));
        _sv_write(o->data, i, ai * silu_bi);
    }
}

void sigil_d_up_into(int64_t dm_ptr, int64_t g_ptr, int64_t up_ptr, int64_t out_ptr, int64_t n) {
    typedef struct { int64_t len; int64_t cap; int64_t* data; } SV;
    SV* dm = (SV*)(uintptr_t)dm_ptr;
    SV* gv = (SV*)(uintptr_t)g_ptr;
    SV* up = (SV*)(uintptr_t)up_ptr;
    SV* o  = (SV*)(uintptr_t)out_ptr;
    for (int64_t i = 0; i < n; i++) {
        float dmi, gi, ui;
        _sv_read(dm->data, i, &dmi);
        _sv_read(gv->data, i, &gi);
        _sv_read(up->data, i, &ui);
        float sg = 1.0f / (1.0f + expf(-ui));
        float dsilu = sg * (1.0f + ui * (1.0f - sg));
        _sv_write(o->data, i, dmi * gi * dsilu);
    }
}

/* ============================================================================
 * GQA Attention C Kernels
 *
 * All Vec<f32> args follow the int64_t-slot ABI:
 *   SV_READ(data, i, f)  — read f32 from slot i (low 4 bytes of int64_t)
 *   SV_WRITE(data, i, f) — write f32 to slot i (zero-extend into int64_t)
 *   SV_ACC(data, i, f)   — read-add-write (accumulate)
 *
 * inv_scale is computed internally from head_dim to avoid f32 bits-in-i64 ABI.
 * ============================================================================ */

#define SV_READ(data, i, f)  memcpy(&(f), &(data)[i], sizeof(float))
#define SV_WRITE(data, i, f) do { int64_t _s=0; memcpy(&_s,&(f),sizeof(float)); (data)[i]=_s; } while(0)
#define SV_ACC(data, i, f)   do { float _t; SV_READ(data,i,_t); _t+=(f); SV_WRITE(data,i,_t); } while(0)
typedef struct { int64_t len; int64_t cap; int64_t* data; } SigilGQAVec;
#define GQA_VEC(ptr) ((SigilGQAVec*)(uintptr_t)(int64_t)(ptr))

/* Softmax in-place on float* row of length seq. */
static void _gqa_softmax(float* row, int64_t seq) {
    float max_s = row[0];
    for (int64_t i = 1; i < seq; i++) if (row[i] > max_s) max_s = row[i];
    float sum = 0.0f;
    for (int64_t i = 0; i < seq; i++) {
        float e = row[i] - max_s;
        row[i] = (e < -20.0f) ? 0.0f : expf(e);
        sum += row[i];
    }
    if (sum > 0.0f) { float inv = 1.0f / sum; for (int64_t i = 0; i < seq; i++) row[i] *= inv; }
}

/* Forward: Q×K^T + causal softmax + weighted V sum → out[seq, hidden_dim].
 * out must be pre-allocated and zeroed. */
void sigil_gqa_attn_fwd_c(int64_t q_ptr, int64_t k_ptr, int64_t v_ptr, int64_t out_ptr,
                            int64_t seq, int64_t hidden_dim, int64_t num_heads, int64_t num_kv_heads) {
    SigilGQAVec *q = GQA_VEC(q_ptr), *k = GQA_VEC(k_ptr), *v = GQA_VEC(v_ptr), *o = GQA_VEC(out_ptr);
    int64_t head_dim = hidden_dim / num_heads;
    int64_t kv_dim = num_kv_heads * head_dim;
    int64_t heads_per_kv = num_heads / num_kv_heads;
    float inv_scale = 1.0f / sqrtf((float)head_dim);
    float* scores = (float*)malloc((size_t)seq * sizeof(float));
    if (!scores) return;
    for (int64_t h = 0; h < num_heads; h++) {
        int64_t kv_h = h / heads_per_kv;
        int64_t q_off = h * head_dim, kv_off = kv_h * head_dim;
        for (int64_t tq = 0; tq < seq; tq++) {
            for (int64_t tk = 0; tk < seq; tk++) {
                if (tk > tq) { scores[tk] = -1e9f; continue; }
                float dot = 0.0f;
                for (int64_t d = 0; d < head_dim; d++) {
                    float qi, ki;
                    SV_READ(q->data, tq * hidden_dim + q_off + d, qi);
                    SV_READ(k->data, tk * kv_dim  + kv_off + d, ki);
                    dot += qi * ki;
                }
                scores[tk] = dot * inv_scale;
            }
            _gqa_softmax(scores, seq);
            for (int64_t d = 0; d < head_dim; d++) {
                float acc = 0.0f;
                for (int64_t tk = 0; tk < seq; tk++) {
                    float vi; SV_READ(v->data, tk * kv_dim + kv_off + d, vi);
                    acc += scores[tk] * vi;
                }
                SV_WRITE(o->data, tq * hidden_dim + q_off + d, acc);
            }
        }
    }
    free(scores);
}

/* Compute attention weights aw[num_heads × seq × seq] with causal softmax.
 * aw must be pre-allocated. */
void sigil_gqa_weights_c(int64_t q_ptr, int64_t k_ptr, int64_t aw_ptr,
                          int64_t seq, int64_t hidden_dim, int64_t num_heads, int64_t num_kv_heads) {
    SigilGQAVec *q = GQA_VEC(q_ptr), *k = GQA_VEC(k_ptr), *aw = GQA_VEC(aw_ptr);
    int64_t head_dim = hidden_dim / num_heads;
    int64_t kv_dim = num_kv_heads * head_dim;
    int64_t heads_per_kv = num_heads / num_kv_heads;
    float inv_scale = 1.0f / sqrtf((float)head_dim);
    float* scores = (float*)malloc((size_t)seq * sizeof(float));
    if (!scores) return;
    for (int64_t h = 0; h < num_heads; h++) {
        int64_t kv_h = h / heads_per_kv;
        int64_t q_off = h * head_dim, kv_off = kv_h * head_dim;
        for (int64_t tq = 0; tq < seq; tq++) {
            for (int64_t tk = 0; tk < seq; tk++) {
                if (tk > tq) { scores[tk] = -1e9f; continue; }
                float dot = 0.0f;
                for (int64_t d = 0; d < head_dim; d++) {
                    float qi, ki;
                    SV_READ(q->data, tq * hidden_dim + q_off + d, qi);
                    SV_READ(k->data, tk * kv_dim  + kv_off + d, ki);
                    dot += qi * ki;
                }
                scores[tk] = dot * inv_scale;
            }
            _gqa_softmax(scores, seq);
            for (int64_t tk = 0; tk < seq; tk++) {
                SV_WRITE(aw->data, h * seq * seq + tq * seq + tk, scores[tk]);
            }
        }
    }
    free(scores);
}

/* Weighted V sum: ctx[seq, hidden] = aw × V. ctx pre-allocated+zeroed. */
void sigil_gqa_ctx_c(int64_t aw_ptr, int64_t v_ptr, int64_t ctx_ptr,
                      int64_t seq, int64_t hidden_dim, int64_t num_heads, int64_t num_kv_heads) {
    SigilGQAVec *aw = GQA_VEC(aw_ptr), *v = GQA_VEC(v_ptr), *ctx = GQA_VEC(ctx_ptr);
    int64_t head_dim = hidden_dim / num_heads;
    int64_t kv_dim = num_kv_heads * head_dim;
    int64_t heads_per_kv = num_heads / num_kv_heads;
    for (int64_t h = 0; h < num_heads; h++) {
        int64_t kv_h = h / heads_per_kv;
        int64_t q_off = h * head_dim, kv_off = kv_h * head_dim;
        for (int64_t tq = 0; tq < seq; tq++) {
            for (int64_t d = 0; d < head_dim; d++) {
                float acc = 0.0f;
                for (int64_t tk = 0; tk < seq; tk++) {
                    float awi, vi;
                    SV_READ(aw->data, h * seq * seq + tq * seq + tk, awi);
                    SV_READ(v->data,  tk * kv_dim  + kv_off + d,     vi);
                    acc += awi * vi;
                }
                SV_WRITE(ctx->data, tq * hidden_dim + q_off + d, acc);
            }
        }
    }
}

/* Backward d_aw[h,tq,tk] = dot(d_ctx[tq, h*hd:], V[tk, kv_h*hd:]) */
void sigil_gqa_d_aw_c(int64_t dctx_ptr, int64_t v_ptr, int64_t daw_ptr,
                       int64_t seq, int64_t hidden_dim, int64_t num_heads, int64_t num_kv_heads) {
    SigilGQAVec *dc = GQA_VEC(dctx_ptr), *v = GQA_VEC(v_ptr), *daw = GQA_VEC(daw_ptr);
    int64_t head_dim = hidden_dim / num_heads;
    int64_t kv_dim = num_kv_heads * head_dim;
    int64_t heads_per_kv = num_heads / num_kv_heads;
    for (int64_t h = 0; h < num_heads; h++) {
        int64_t kv_h = h / heads_per_kv;
        int64_t q_off = h * head_dim, kv_off = kv_h * head_dim;
        for (int64_t tq = 0; tq < seq; tq++) {
            for (int64_t tk = 0; tk < seq; tk++) {
                float dot = 0.0f;
                for (int64_t d = 0; d < head_dim; d++) {
                    float dci, vi;
                    SV_READ(dc->data, tq * hidden_dim + q_off + d, dci);
                    SV_READ(v->data,  tk * kv_dim  + kv_off + d,   vi);
                    dot += dci * vi;
                }
                SV_WRITE(daw->data, h * seq * seq + tq * seq + tk, dot);
            }
        }
    }
}

/* Backward d_v: accumulate aw[h,tq,tk] * d_ctx[tq, h*hd:] → d_v[tk, kv_h*hd:] */
void sigil_gqa_d_v_c(int64_t dctx_ptr, int64_t aw_ptr, int64_t dv_ptr,
                      int64_t seq, int64_t hidden_dim, int64_t num_heads, int64_t num_kv_heads) {
    SigilGQAVec *dc = GQA_VEC(dctx_ptr), *aw = GQA_VEC(aw_ptr), *dv = GQA_VEC(dv_ptr);
    int64_t head_dim = hidden_dim / num_heads;
    int64_t kv_dim = num_kv_heads * head_dim;
    int64_t heads_per_kv = num_heads / num_kv_heads;
    for (int64_t h = 0; h < num_heads; h++) {
        int64_t kv_h = h / heads_per_kv;
        int64_t q_off = h * head_dim, kv_off = kv_h * head_dim;
        for (int64_t tk = 0; tk < seq; tk++) {
            for (int64_t d = 0; d < head_dim; d++) {
                float acc = 0.0f;
                for (int64_t tq = 0; tq < seq; tq++) {
                    float awi, dci;
                    SV_READ(aw->data, h * seq * seq + tq * seq + tk, awi);
                    SV_READ(dc->data, tq * hidden_dim + q_off + d,   dci);
                    acc += awi * dci;
                }
                SV_ACC(dv->data, tk * kv_dim + kv_off + d, acc);
            }
        }
    }
}

/* Backward softmax: d_scores[h,tq,tk] = aw[h,tq,tk]*(d_aw[h,tq,tk] - dot(d_aw[tq,:],aw[tq,:])) */
void sigil_gqa_d_scores_c(int64_t daw_ptr, int64_t aw_ptr, int64_t ds_ptr,
                            int64_t seq, int64_t num_heads) {
    SigilGQAVec *daw = GQA_VEC(daw_ptr), *aw = GQA_VEC(aw_ptr), *ds = GQA_VEC(ds_ptr);
    for (int64_t h = 0; h < num_heads; h++) {
        for (int64_t tq = 0; tq < seq; tq++) {
            float dot = 0.0f;
            for (int64_t tk = 0; tk < seq; tk++) {
                float dawi, awi;
                int64_t idx = h * seq * seq + tq * seq + tk;
                SV_READ(daw->data, idx, dawi); SV_READ(aw->data, idx, awi);
                dot += dawi * awi;
            }
            for (int64_t tk = 0; tk < seq; tk++) {
                int64_t idx = h * seq * seq + tq * seq + tk;
                float dawi, awi;
                SV_READ(daw->data, idx, dawi); SV_READ(aw->data, idx, awi);
                float val = awi * (dawi - dot);
                SV_WRITE(ds->data, idx, val);
            }
        }
    }
}

/* Backward d_q[tq,h*hd+d] += inv_scale * sum_tk d_scores[h,tq,tk] * K[tk,kv_h*hd+d] */
void sigil_gqa_d_q_c(int64_t ds_ptr, int64_t k_ptr, int64_t dq_ptr,
                      int64_t seq, int64_t hidden_dim, int64_t num_heads, int64_t num_kv_heads) {
    SigilGQAVec *ds = GQA_VEC(ds_ptr), *k = GQA_VEC(k_ptr), *dq = GQA_VEC(dq_ptr);
    int64_t head_dim = hidden_dim / num_heads;
    int64_t kv_dim = num_kv_heads * head_dim;
    int64_t heads_per_kv = num_heads / num_kv_heads;
    float inv_scale = 1.0f / sqrtf((float)head_dim);
    for (int64_t h = 0; h < num_heads; h++) {
        int64_t kv_h = h / heads_per_kv;
        int64_t q_off = h * head_dim, kv_off = kv_h * head_dim;
        for (int64_t tq = 0; tq < seq; tq++) {
            for (int64_t d = 0; d < head_dim; d++) {
                float acc = 0.0f;
                for (int64_t tk = 0; tk < seq; tk++) {
                    float dsi, ki;
                    SV_READ(ds->data, h * seq * seq + tq * seq + tk, dsi);
                    SV_READ(k->data,  tk * kv_dim  + kv_off + d,     ki);
                    acc += dsi * ki;
                }
                SV_ACC(dq->data, tq * hidden_dim + q_off + d, acc * inv_scale);
            }
        }
    }
}

/* Backward d_k[tk,kv_h*hd+d] += inv_scale * sum_{h,tq} d_scores[h,tq,tk] * Q[tq,h*hd+d] */
void sigil_gqa_d_k_c(int64_t ds_ptr, int64_t q_ptr, int64_t dk_ptr,
                      int64_t seq, int64_t hidden_dim, int64_t num_heads, int64_t num_kv_heads) {
    SigilGQAVec *ds = GQA_VEC(ds_ptr), *q = GQA_VEC(q_ptr), *dk = GQA_VEC(dk_ptr);
    int64_t head_dim = hidden_dim / num_heads;
    int64_t kv_dim = num_kv_heads * head_dim;
    int64_t heads_per_kv = num_heads / num_kv_heads;
    float inv_scale = 1.0f / sqrtf((float)head_dim);
    for (int64_t h = 0; h < num_heads; h++) {
        int64_t kv_h = h / heads_per_kv;
        int64_t q_off = h * head_dim, kv_off = kv_h * head_dim;
        for (int64_t tk = 0; tk < seq; tk++) {
            for (int64_t d = 0; d < head_dim; d++) {
                float acc = 0.0f;
                for (int64_t tq = 0; tq < seq; tq++) {
                    float dsi, qi;
                    SV_READ(ds->data, h * seq * seq + tq * seq + tk, dsi);
                    SV_READ(q->data,  tq * hidden_dim + q_off + d,   qi);
                    acc += dsi * qi;
                }
                SV_ACC(dk->data, tk * kv_dim + kv_off + d, acc * inv_scale);
            }
        }
    }
}

/* In-place SGD: w[i] -= lr * clip(g[i], ±clip_val); then frees the gradient Vec.
 * All pointers are i64; lr and clip are f32 bits packed as i64. */
void sigil_sgd_inplace_free_grad(int64_t w_ptr, int64_t g_ptr, int64_t n,
                                  int64_t lr_bits, int64_t clip_bits) {
    typedef struct { int64_t len; int64_t cap; int64_t* data; } SigilVecSGD;
    SigilVecSGD* wv = (SigilVecSGD*)(uintptr_t)w_ptr;
    SigilVecSGD* gv = (SigilVecSGD*)(uintptr_t)g_ptr;
    float lr, clip_val;
    memcpy(&lr,       &lr_bits,   sizeof(float));
    memcpy(&clip_val, &clip_bits, sizeof(float));

    /* G-DIAG: Print first SGD call each step to verify gradients are non-zero and weights update */
    static int64_t sgd_call_count = 0;
    static int64_t sgd_step_count = 0;
    /* 57 weights per step: print on call 0 of each step (W_out gradient, 131072 elements) */
    if (sgd_call_count % 57 == 0) {
        float w0 = 0.0f, g0 = 0.0f, g1 = 0.0f, g_max = 0.0f;
        int64_t nonzero_g = 0;
        if (wv && wv->data && n > 0) memcpy(&w0, &wv->data[0], sizeof(float));
        if (gv && gv->data) {
            if (n > 0) memcpy(&g0, &gv->data[0], sizeof(float));
            if (n > 1) memcpy(&g1, &gv->data[1], sizeof(float));
            for (int64_t i = 0; i < n && i < 1024; i++) {
                float gi; memcpy(&gi, &gv->data[i], sizeof(float));
                if (gi != 0.0f) nonzero_g++;
                if (gi > g_max) g_max = gi;
            }
        }
        fprintf(stderr, "[SGD-DIAG step=%lld] w0=%.6f g0=%.6f g1=%.6f g_max=%.6f nonzero=%lld/1024 lr=%.6f\n",
                (long long)sgd_step_count, w0, g0, g1, g_max, (long long)nonzero_g, lr);
        fflush(stderr);
        sgd_step_count++;
    }
    sgd_call_count++;

    if (wv && gv && wv->data && gv->data) {
        for (int64_t i = 0; i < n; i++) {
            float wi, gi;
            memcpy(&wi, &wv->data[i], sizeof(float));
            memcpy(&gi, &gv->data[i], sizeof(float));
            if (gi > clip_val) gi = clip_val;
            else if (gi < -clip_val) gi = -clip_val;
            wi -= lr * gi;
            int64_t wi64 = 0;
            memcpy(&wi64, &wi, sizeof(float));
            wv->data[i] = wi64;
        }
    }
    /* Free the gradient Vec to prevent per-step memory leak */
    if (gv) {
        if (gv->data) free(gv->data);
        free(gv);
    }
}

/* ============================================================================
 * Checkpoint I/O Helpers
 *
 * Sigil Vec<f32> ABI: Vec<T> is a POINTER (i64) to a heap-allocated struct:
 *   typedef struct { int64_t len; int64_t capacity; int64_t* data; } SigilVec;
 * Elements in Vec<f32> are stored as int64_t (8 bytes), f32 bits in low 4.
 * ============================================================================ */

typedef struct { int64_t len; int64_t capacity; int64_t* data; } SigilVecCkpt;

/* Write n elements of a Vec<f32> (passed as i64 pointer) to file.
 * Elements are i64 (8 bytes each — Sigil Vec<f32> stores f32 bits as i64).
 * Returns number of elements written, or -1 on error. */
int64_t sigil_ckpt_vec_write(int64_t handle, int64_t vec_ptr, int64_t n) {
    FILE* f = (FILE*)(uintptr_t)handle;
    SigilVecCkpt* vec = (SigilVecCkpt*)(uintptr_t)vec_ptr;
    if (!f || !vec || !vec->data || n <= 0) return -1;
    return (int64_t)fwrite(vec->data, sizeof(int64_t), (size_t)n, f);
}

/* Allocate a new SigilVec, read n i64 elements from file into it.
 * Returns the pointer as i64 (new Vec<f32>), or 0 on error. */
int64_t sigil_ckpt_vec_load(int64_t handle, int64_t n) {
    FILE* f = (FILE*)(uintptr_t)handle;
    if (!f || n <= 0) return 0;
    SigilVecCkpt* vec = (SigilVecCkpt*)malloc(sizeof(SigilVecCkpt));
    if (!vec) return 0;
    int64_t* data = (int64_t*)malloc((size_t)n * sizeof(int64_t));
    if (!data) { free(vec); return 0; }
    fread(data, sizeof(int64_t), (size_t)n, f);
    vec->len = n;
    vec->capacity = n;
    vec->data = data;
    return (int64_t)(uintptr_t)vec;
}

/* Write a single i64 to file (used for step header) */
void sigil_ckpt_write_i64(int64_t handle, int64_t val) {
    FILE* f = (FILE*)(uintptr_t)handle;
    if (f) fwrite(&val, sizeof(int64_t), 1, f);
}

/* Read a single i64 from file */
int64_t sigil_ckpt_read_i64(int64_t handle) {
    int64_t val = -1;
    FILE* f = (FILE*)(uintptr_t)handle;
    if (f) fread(&val, sizeof(int64_t), 1, f);
    return val;
}

/* ============================================================================
 * SGEMM CPU stubs — resolve linker symbols for GPU SGEMM variants.
 * These return 0 so Sigil code falls through to its CPU triple-loop fallback.
 * The real GPU implementations live in sigil_runtime_cuda.c.
 * Excluded from CUDA builds (SIGIL_CUDA_EXTERNAL) to avoid multiple definitions.
 * ============================================================================ */

#ifndef SIGIL_CUDA_EXTERNAL
int64_t sigil_sgemm_nt_sv(int64_t a_ptr, int64_t b_ptr, int64_t out_ptr,
                           int64_t M, int64_t N, int64_t K) {
    (void)a_ptr; (void)b_ptr; (void)out_ptr; (void)M; (void)N; (void)K;
    return 0; /* no GPU — caller uses CPU fallback */
}

int64_t sigil_sgemm_nn_sv(int64_t a_ptr, int64_t b_ptr, int64_t out_ptr,
                           int64_t M, int64_t N, int64_t K) {
    (void)a_ptr; (void)b_ptr; (void)out_ptr; (void)M; (void)N; (void)K;
    return 0; /* no GPU — caller uses CPU fallback */
}

int64_t sigil_sgemm_tn_sv(int64_t a_ptr, int64_t b_ptr, int64_t out_ptr,
                           int64_t M, int64_t N, int64_t K) {
    (void)a_ptr; (void)b_ptr; (void)out_ptr; (void)M; (void)N; (void)K;
    return 0; /* no GPU — caller uses CPU fallback */
}
#endif /* SIGIL_CUDA_EXTERNAL */

/* ============================================================================
 * Gradient norm accumulation + global norm clipping
 * sigil_gnorm_reset()               — reset accumulator
 * sigil_gnorm_add_sv(vec_ptr: i64)  — add L2^2 of one SigilVec<f32>
 * sigil_gnorm_finish_print(step)    — sqrt, compute clip scale, print, reset
 *
 * Global norm clipping: after accumulating all grads, finish_print computes
 *   clip_scale = min(1.0, GNORM_CLIP / norm)
 * and stores it in g_grad_clip_scale. sigil_adamw_step then multiplies each
 * gradient element by this scale before updating m/v/w. This preserves gradient
 * direction (all elements scaled equally) unlike element-wise clipping.
 *
 * GNORM_CLIP = 2500.0: halved from 5000 for seq_len=256 (Exp I).
 * Gradient variance scales with seq_len; sqrt(2)x seq_len → sqrt(2)x gnorm in expectation.
 * 5000 / sqrt(2) ≈ 3535; using 2500 for extra conservatism to stabilise past step 250.
 *
 * SigilVec layout: { int64_t len, int64_t capacity, int64_t* data }
 * Each data element stores float bits zero-extended into int64_t.
 * ============================================================================ */
static double g_gnorm_ss = 0.0;
static float  g_grad_clip_scale = 1.0f;
#define GNORM_CLIP 2500.0

void sigil_gnorm_reset() {
    g_gnorm_ss = 0.0;
}

void sigil_gnorm_add_sv(int64_t vec_ptr) {
    if (!vec_ptr) return;
    int64_t* v = (int64_t*)(uintptr_t)vec_ptr;
    int64_t n = v[0];
    if (n <= 0) return;
    int64_t* raw = (int64_t*)(uintptr_t)v[2];
    if (!raw) return;
    for (int64_t i = 0; i < n; i++) {
        uint32_t bits = (uint32_t)(uint64_t)raw[i];
        float val;
        memcpy(&val, &bits, 4);
        g_gnorm_ss += (double)val * (double)val;
    }
}

void sigil_gnorm_finish_print(int64_t step) {
    double norm = sqrt(g_gnorm_ss);
    /* Compute global norm clip scale — applied by sigil_adamw_step */
    double scale = (norm > GNORM_CLIP && norm > 0.0) ? (GNORM_CLIP / norm) : 1.0;
    g_grad_clip_scale = (float)scale;
    printf("[GNORM] step=%lld global=%.6f scale=%.4f\n",
           (long long)step, norm, g_grad_clip_scale);
    fflush(stdout);
    g_gnorm_ss = 0.0;
}

/* ============================================================================
 * AdamW optimizer (Vec<f32>-based, G-FIX compatible)
 *
 * sigil_adamw_step(w, g, m, v, n, lr_bits, wd_bits, t)
 *   w, g, m, v : i64 pointers to SigilVec structs (same ABI as SGD above)
 *   n          : element count
 *   lr_bits    : learning rate as f32 bits packed in i64
 *   wd_bits    : weight decay coefficient as f32 bits packed in i64
 *   t          : step count, 1-indexed (for bias correction)
 *
 * Updates m and v in-place; applies decoupled weight decay to w; frees g.
 * β1=0.9, β2=0.999, ε=1e-8 are hardcoded.
 * ============================================================================ */

void sigil_adamw_step(int64_t w_ptr, int64_t g_ptr, int64_t m_ptr, int64_t v_ptr,
                      int64_t n, int64_t lr_bits, int64_t wd_bits, int64_t t) {
    typedef struct { int64_t len; int64_t cap; int64_t* data; } SigilVecAW;
    SigilVecAW* wv = (SigilVecAW*)(uintptr_t)w_ptr;
    SigilVecAW* gv = (SigilVecAW*)(uintptr_t)g_ptr;
    SigilVecAW* mv = (SigilVecAW*)(uintptr_t)m_ptr;
    SigilVecAW* vv = (SigilVecAW*)(uintptr_t)v_ptr;

    float lr, wd;
    memcpy(&lr, &lr_bits, sizeof(float));
    memcpy(&wd, &wd_bits, sizeof(float));

    const float beta1 = 0.9f;
    const float beta2 = 0.999f;
    const float eps   = 1e-8f;

    if (t < 1) t = 1;
    /* Cosine LR decay: lr(t) = lr * 0.5 * (1 + cos(pi * t / T_max))
     * T_max = 500. Brakes LR at the natural convergence point (~step 200-300 for
     * seq_len=256). Both Exp K (T_max=500) and Exp L (T_max=2000) show best loss
     * at step ~220; T_max=2000 diverged at step ~250 (LR still 2.41e-6, too high).
     * Floor raised from 1% → 25%: after step 500, effective LR = 2.5e-6*0.25 = 6.25e-7.
     * High enough for continued learning; low enough to stay stable in the basin. */
    #ifndef M_PI
    #define M_PI 3.14159265358979323846f
    #endif
    const float T_max = 500.0f;
    float t_sched = (float)t;
    if (t_sched > T_max) t_sched = T_max;  /* clamp: one-shot decay, no restart */
    float cos_decay = 0.5f * (1.0f + cosf((float)M_PI * t_sched / T_max));
    if (cos_decay < 0.25f) cos_decay = 0.25f;  /* floor at 25%: LR=6.25e-7 for continued learning */
    float lr_t = lr * cos_decay;
    /* Bias correction: alpha = lr_t * sqrt(1 - beta2^t) / (1 - beta1^t) */
    float bc1   = 1.0f - powf(beta1, (float)t);
    float bc2   = 1.0f - powf(beta2, (float)t);
    float alpha = lr_t * sqrtf(bc2) / bc1;

    if (!wv || !gv || !mv || !vv) return;
    if (!wv->data || !gv->data || !mv->data || !vv->data) return;

    for (int64_t i = 0; i < n; i++) {
        float wi, gi, mi, vi;
        memcpy(&wi, &wv->data[i], sizeof(float));
        memcpy(&gi, &gv->data[i], sizeof(float));
        memcpy(&mi, &mv->data[i], sizeof(float));
        memcpy(&vi, &vv->data[i], sizeof(float));

        /* Global norm clipping: scale computed once per step by sigil_gnorm_finish_print.
         * Preserves gradient direction (all elements scaled equally).
         * No-op (scale=1.0) when gnorm <= GNORM_CLIP (5000). */
        gi *= g_grad_clip_scale;

        /* Update biased first and second moment estimates */
        mi = beta1 * mi + (1.0f - beta1) * gi;
        vi = beta2 * vi + (1.0f - beta2) * gi * gi;

        /* Decoupled weight decay + Adam update */
        wi = wi * (1.0f - lr * wd) - alpha * mi / (sqrtf(vi) + eps);

        int64_t wi64 = 0, mi64 = 0, vi64 = 0;
        memcpy(&wi64, &wi, sizeof(float));
        memcpy(&mi64, &mi, sizeof(float));
        memcpy(&vi64, &vi, sizeof(float));
        wv->data[i] = wi64;
        mv->data[i] = mi64;
        vv->data[i] = vi64;
    }

    /* Free gradient Vec (prevents per-step memory leak) */
    if (gv) {
        if (gv->data) free(gv->data);
        free(gv);
    }
}

/* ============================================================================
 * Gradient accumulation helper
 *
 * sigil_vec_acc_scaled_free(dst, src, n, accum_steps)
 *   dst         : i64 — accumulated gradient SigilVec (modified in-place, NOT freed)
 *   src         : i64 — per-micro-batch gradient SigilVec (FREED after accumulation)
 *   n           : i64 — number of f32 elements
 *   accum_steps : i64 — number of micro-steps (scale = 1.0 / accum_steps)
 *
 * Adds (1/accum_steps) * src[i] into dst[i] for all i, then frees src.
 * Scale is computed in C from the integer accum_steps to avoid Sigil f32 ABI issues.
 * Called once per weight per micro-step during gradient accumulation.
 * After accum_steps calls, dst holds the averaged gradient ready for AdamW.
 * ============================================================================ */

void sigil_vec_acc_scaled_free(int64_t dst_ptr, int64_t src_ptr, int64_t n, int64_t accum_steps) {
    typedef struct { int64_t len; int64_t cap; int64_t* data; } SigilVecAcc;
    SigilVecAcc* dst = (SigilVecAcc*)(uintptr_t)dst_ptr;
    SigilVecAcc* src = (SigilVecAcc*)(uintptr_t)src_ptr;
    float scale = (accum_steps > 0) ? (1.0f / (float)accum_steps) : 1.0f;
    if (dst && src && dst->data && src->data) {
        float* d = (float*)dst->data;
        float* s = (float*)src->data;
        for (int64_t i = 0; i < n; i++) {
            d[i] += scale * s[i];
        }
    }
    if (src) {
        if (src->data) free(src->data);
        free(src);
    }
}

/* ============================================================================
 * Vec free helper — explicit destructor for leaked forward/backward intermediates
 *
 * sigil_vec_free_raw(ptr): frees the SigilVec struct AND its data array.
 * Used by the Sigil `vec_drop(v)` wrapper to manually release intermediates
 * that Sigil's LLVM backend cannot automatically Drop (no RAII in codegen).
 * ============================================================================ */

void sigil_vec_free_raw(int64_t ptr) {
    typedef struct { int64_t len; int64_t cap; int64_t* data; } SigilVec;
    SigilVec* v = (SigilVec*)(uintptr_t)ptr;
    if (v) {
        if (v->data) free(v->data);
        free(v);
    }
}

/* ============================================================================
 * Raw tensor fill helpers — bypass &StoragePtr fat-pointer ABI bug
 *
 * Sigil LLVM ABI: `&T` reference params become fat pointers {ptr, metadata}.
 * Methods called on `&StoragePtr` read fat-pointer fields, not StoragePtr data.
 * Fix: extract data_ptr (i64) and data_len (usize) while value is unborowed,
 * then call these C functions with the raw pointer directly.
 *
 * sigil_fill_uniform_f32_raw(ptr, len, low, high)
 *   ptr  : i64 — StoragePtr.data_ptr (raw float* cast to i64)
 *   len  : i64 — number of f32 elements
 *   low  : f64 — lower bound (inclusive)
 *   high : f64 — upper bound (exclusive)
 *
 * sigil_fill_randn_f32_raw(ptr, len)
 *   ptr  : i64 — raw float*
 *   len  : i64 — number of f32 elements
 *   Uses Box-Muller transform for standard normal samples.
 * ============================================================================ */

/* ---- Kaiming / Xavier / randn / rand raw inits ----
 * All params are i64 — no f64 ABI issues with Sigil LLVM calling convention.
 * Compute bounds entirely in C; Sigil passes only (ptr, len, fan_in/fan_out) as i64. */

/* Kaiming (He) uniform init: fills [-bound, bound] where bound = sqrt(6/fan_in) */
void sigil_kaiming_init_raw(int64_t ptr, int64_t len, int64_t fan_in) {
    float* data = (float*)(uintptr_t)ptr;
    if (!data || len <= 0 || fan_in <= 0) return;
    double bound = sqrt(6.0 / (double)fan_in);
    double two_bound = 2.0 * bound;
    for (int64_t i = 0; i < len; i++) {
        double u = ((double)rand()) / ((double)RAND_MAX + 1.0);
        data[i] = (float)(-bound + u * two_bound);
    }
}

/* Xavier (Glorot) uniform init: fills [-bound, bound] where bound = sqrt(6/(fan_in+fan_out)) */
void sigil_xavier_init_raw(int64_t ptr, int64_t len, int64_t fan_in, int64_t fan_out) {
    float* data = (float*)(uintptr_t)ptr;
    if (!data || len <= 0 || (fan_in + fan_out) <= 0) return;
    double bound = sqrt(6.0 / (double)(fan_in + fan_out));
    double two_bound = 2.0 * bound;
    for (int64_t i = 0; i < len; i++) {
        double u = ((double)rand()) / ((double)RAND_MAX + 1.0);
        data[i] = (float)(-bound + u * two_bound);
    }
}

/* Normal (N(0,1)) init using Box-Muller transform */
void sigil_fill_randn_f32_raw(int64_t ptr, int64_t len) {
    float* data = (float*)(uintptr_t)ptr;
    if (!data || len <= 0) return;
    for (int64_t i = 0; i < len; i += 2) {
        double u1, u2;
        do { u1 = ((double)rand() + 0.5) / ((double)RAND_MAX + 1.0); } while (u1 <= 0.0);
        u2 = ((double)rand() + 0.5) / ((double)RAND_MAX + 1.0);
        double r = sqrt(-2.0 * log(u1));
        double theta = 2.0 * 3.14159265358979323846 * u2;
        data[i] = (float)(r * cos(theta));
        if (i + 1 < len) data[i + 1] = (float)(r * sin(theta));
    }
}

/* Uniform [0,1) init */
void sigil_fill_uniform_01_raw(int64_t ptr, int64_t len) {
    float* data = (float*)(uintptr_t)ptr;
    if (!data || len <= 0) return;
    for (int64_t i = 0; i < len; i++) {
        data[i] = (float)(((double)rand()) / ((double)RAND_MAX + 1.0));
    }
}

/* Generic uniform [low, high) — low_bits/high_bits are f64 IEEE 754 bits packed in i64.
 * Matches Sigil LLVM ABI: all numerics as i64; memcpy recovers doubles.
 * Same pattern as sigil_adamw_step lr_bits/wd_bits. */
void sigil_fill_uniform_f32_raw(int64_t ptr, int64_t len,
                                int64_t low_bits, int64_t high_bits) {
    float* data = (float*)(uintptr_t)ptr;
    if (!data || len <= 0) return;
    double low, high;
    memcpy(&low, &low_bits, sizeof(double));
    memcpy(&high, &high_bits, sizeof(double));
    double range = high - low;
    for (int64_t i = 0; i < len; i++) {
        double u = ((double)rand()) / ((double)RAND_MAX + 1.0);
        data[i] = (float)(low + u * range);
    }
}

/* ============================================================================
 * SIMD Functions (AVX-512 F32x16)
 * ============================================================================ */

#ifdef __AVX512F__
#include <immintrin.h>

/* Allocate aligned memory for SIMD vectors (64-byte aligned for AVX-512) */
void* sigil_simd_alloc(int64_t num_floats) {
    size_t size = (size_t)num_floats * sizeof(float);
    return aligned_alloc(64, size);
}

/* Free aligned SIMD memory */
void sigil_simd_free(void* ptr) {
    free(ptr);
}

/* Splat scalar to all 16 lanes of F32x16 */
void sigil_simd_splat_f32x16(float* dest, float value) {
    __m512 v = _mm512_set1_ps(value);
    _mm512_store_ps(dest, v);
}

/* Load aligned F32x16 - just memcpy wrapper for consistency */
void sigil_simd_load_f32x16(float* dest, const float* src) {
    __m512 v = _mm512_load_ps(src);
    _mm512_store_ps(dest, v);
}

/* Store aligned F32x16 */
void sigil_simd_store_f32x16(float* dest, const float* src) {
    __m512 v = _mm512_load_ps(src);
    _mm512_store_ps(dest, v);
}

/* F32x16 add: dest = a + b */
void sigil_simd_add_f32x16(float* dest, const float* a, const float* b) {
    __m512 va = _mm512_load_ps(a);
    __m512 vb = _mm512_load_ps(b);
    __m512 vr = _mm512_add_ps(va, vb);
    _mm512_store_ps(dest, vr);
}

/* F32x16 subtract: dest = a - b */
void sigil_simd_sub_f32x16(float* dest, const float* a, const float* b) {
    __m512 va = _mm512_load_ps(a);
    __m512 vb = _mm512_load_ps(b);
    __m512 vr = _mm512_sub_ps(va, vb);
    _mm512_store_ps(dest, vr);
}

/* F32x16 multiply: dest = a * b */
void sigil_simd_mul_f32x16(float* dest, const float* a, const float* b) {
    __m512 va = _mm512_load_ps(a);
    __m512 vb = _mm512_load_ps(b);
    __m512 vr = _mm512_mul_ps(va, vb);
    _mm512_store_ps(dest, vr);
}

/* F32x16 divide: dest = a / b */
void sigil_simd_div_f32x16(float* dest, const float* a, const float* b) {
    __m512 va = _mm512_load_ps(a);
    __m512 vb = _mm512_load_ps(b);
    __m512 vr = _mm512_div_ps(va, vb);
    _mm512_store_ps(dest, vr);
}

/* F32x16 fused multiply-add: dest = a * b + c */
void sigil_simd_fmadd_f32x16(float* dest, const float* a, const float* b, const float* c) {
    __m512 va = _mm512_load_ps(a);
    __m512 vb = _mm512_load_ps(b);
    __m512 vc = _mm512_load_ps(c);
    __m512 vr = _mm512_fmadd_ps(va, vb, vc);
    _mm512_store_ps(dest, vr);
}

/* F32x16 horizontal sum (reduce add) */
float sigil_simd_reduce_add_f32x16(const float* src) {
    __m512 v = _mm512_load_ps(src);
    return _mm512_reduce_add_ps(v);
}

/* F32x16 extract single element */
float sigil_simd_extract_f32x16(const float* src, int64_t index) {
    return src[index & 15];
}

/* F32x16 dot product of two vectors */
float sigil_simd_dot_f32x16(const float* a, const float* b) {
    __m512 va = _mm512_load_ps(a);
    __m512 vb = _mm512_load_ps(b);
    __m512 vr = _mm512_mul_ps(va, vb);
    return _mm512_reduce_add_ps(vr);
}

#else
/* Fallback scalar implementations when AVX-512 is not available */

void* sigil_simd_alloc(int64_t num_floats) {
    return malloc((size_t)num_floats * sizeof(float));
}

void sigil_simd_free(void* ptr) {
    free(ptr);
}

void sigil_simd_splat_f32x16(float* dest, float value) {
    for (int i = 0; i < 16; i++) dest[i] = value;
}

void sigil_simd_load_f32x16(float* dest, const float* src) {
    for (int i = 0; i < 16; i++) dest[i] = src[i];
}

void sigil_simd_store_f32x16(float* dest, const float* src) {
    for (int i = 0; i < 16; i++) dest[i] = src[i];
}

void sigil_simd_add_f32x16(float* dest, const float* a, const float* b) {
    for (int i = 0; i < 16; i++) dest[i] = a[i] + b[i];
}

void sigil_simd_sub_f32x16(float* dest, const float* a, const float* b) {
    for (int i = 0; i < 16; i++) dest[i] = a[i] - b[i];
}

void sigil_simd_mul_f32x16(float* dest, const float* a, const float* b) {
    for (int i = 0; i < 16; i++) dest[i] = a[i] * b[i];
}

void sigil_simd_div_f32x16(float* dest, const float* a, const float* b) {
    for (int i = 0; i < 16; i++) dest[i] = a[i] / b[i];
}

void sigil_simd_fmadd_f32x16(float* dest, const float* a, const float* b, const float* c) {
    for (int i = 0; i < 16; i++) dest[i] = a[i] * b[i] + c[i];
}

float sigil_simd_reduce_add_f32x16(const float* src) {
    float sum = 0.0f;
    for (int i = 0; i < 16; i++) sum += src[i];
    return sum;
}

float sigil_simd_extract_f32x16(const float* src, int64_t index) {
    return src[index & 15];
}

float sigil_simd_dot_f32x16(const float* a, const float* b) {
    float sum = 0.0f;
    for (int i = 0; i < 16; i++) sum += a[i] * b[i];
    return sum;
}

#endif /* __AVX512F__ */

/* ============================================================================
 * CUDA Functions (using CUDA Driver API)
 * ============================================================================ */

/* When linking with sigil_runtime_cuda.c, define SIGIL_CUDA_EXTERNAL to avoid
 * duplicate definitions. sigil_runtime_cuda.c provides the full implementations. */
#ifndef SIGIL_CUDA_EXTERNAL

#ifdef SIGIL_CUDA_SUPPORT
#include <cuda.h>
#include <nvrtc.h>

static CUcontext g_cuda_context = NULL;
static CUdevice g_cuda_device = 0;
static int g_cuda_initialized = 0;

/* Initialize CUDA - must be called before any other CUDA operations */
int64_t sigil_cuda_init(void) {
    if (g_cuda_initialized) return 1;

    CUresult err = cuInit(0);
    if (err != CUDA_SUCCESS) {
        fprintf(stderr, "CUDA init failed: %d\n", err);
        return 0;
    }

    err = cuDeviceGet(&g_cuda_device, 0);
    if (err != CUDA_SUCCESS) {
        fprintf(stderr, "CUDA device get failed: %d\n", err);
        return 0;
    }

    err = cuCtxCreate(&g_cuda_context, 0, g_cuda_device);
    if (err != CUDA_SUCCESS) {
        fprintf(stderr, "CUDA context create failed: %d\n", err);
        return 0;
    }

    g_cuda_initialized = 1;
    return 1;
}

/* Cleanup CUDA resources */
void sigil_cuda_cleanup(void) {
    if (g_cuda_context) {
        cuCtxDestroy(g_cuda_context);
        g_cuda_context = NULL;
    }
    g_cuda_initialized = 0;
}

/* Get CUDA device properties */
int64_t sigil_cuda_get_device_count(void) {
    int count = 0;
    if (cuDeviceGetCount(&count) != CUDA_SUCCESS) return 0;
    return (int64_t)count;
}

/* Allocate device memory - returns device pointer as i64 */
int64_t sigil_cuda_malloc(int64_t size) {
    if (!g_cuda_initialized) {
        if (!sigil_cuda_init()) return 0;
    }

    CUdeviceptr ptr = 0;
    CUresult err = cuMemAlloc(&ptr, (size_t)size);
    if (err != CUDA_SUCCESS) {
        fprintf(stderr, "CUDA malloc failed: %d\n", err);
        return 0;
    }
    return (int64_t)ptr;
}

/* Free device memory */
void sigil_cuda_free(int64_t device_ptr) {
    if (device_ptr) {
        cuMemFree((CUdeviceptr)device_ptr);
    }
}

/* Copy host to device */
int64_t sigil_cuda_memcpy_h2d(int64_t dst_device, void* src_host, int64_t size) {
    CUresult err = cuMemcpyHtoD((CUdeviceptr)dst_device, src_host, (size_t)size);
    return (err == CUDA_SUCCESS) ? 1 : 0;
}

/* Copy device to host */
int64_t sigil_cuda_memcpy_d2h(void* dst_host, int64_t src_device, int64_t size) {
    CUresult err = cuMemcpyDtoH(dst_host, (CUdeviceptr)src_device, (size_t)size);
    return (err == CUDA_SUCCESS) ? 1 : 0;
}

/* Copy device to device */
int64_t sigil_cuda_memcpy_d2d(int64_t dst_device, int64_t src_device, int64_t size) {
    CUresult err = cuMemcpyDtoD((CUdeviceptr)dst_device, (CUdeviceptr)src_device, (size_t)size);
    return (err == CUDA_SUCCESS) ? 1 : 0;
}

/* Synchronize device */
void sigil_cuda_sync(void) {
    cuCtxSynchronize();
}

/* Fill device buffer with N(0,1) random values via host staging buffer.
 * Avoids passing &StoragePtr (fat-pointer ABI bug) by taking raw device_ptr + n.
 * Called as: sigil_cuda_fill_randn_f32(device_ptr, n)
 * where device_ptr is i64 CUDA device address, n is element count. */
void sigil_cuda_fill_randn_f32(int64_t device_ptr, int64_t n) {
    if (n <= 0) return;
    float* host = (float*)malloc((size_t)n * sizeof(float));
    if (!host) return;
    /* Box-Muller transform */
    for (int64_t i = 0; i + 1 < n; i += 2) {
        double u1, u2;
        do { u1 = (double)rand() / ((double)RAND_MAX + 1.0); } while (u1 < 1e-10);
        u2 = (double)rand() / ((double)RAND_MAX + 1.0);
        double r = sqrt(-2.0 * log(u1));
        double t = 6.28318530718 * u2;
        host[i]     = (float)(r * cos(t));
        host[i + 1] = (float)(r * sin(t));
    }
    if (n & 1) { host[n-1] = host[0]; } /* if odd, duplicate first */
    cuMemcpyHtoD((CUdeviceptr)device_ptr, host, (size_t)n * sizeof(float));
    free(host);
}

/* Fill device buffer with zeros using cuMemsetD8. */
void sigil_cuda_zero_f32(int64_t device_ptr, int64_t n) {
    if (n <= 0) return;
    cuMemsetD8((CUdeviceptr)device_ptr, 0, (size_t)n * sizeof(float));
}

/* Zero exactly `bytes` bytes of device memory (byte-granularity, for non-float dtypes). */
void sigil_cuda_memset_zero(int64_t device_ptr, int64_t bytes) {
    if (bytes <= 0) return;
    cuMemsetD8((CUdeviceptr)device_ptr, 0, (size_t)bytes);
}

/* Kernel module storage */
#define MAX_CUDA_MODULES 64
static CUmodule g_cuda_modules[MAX_CUDA_MODULES];
static CUfunction g_cuda_functions[MAX_CUDA_MODULES];
static int g_num_cuda_modules = 0;

/* Compile PTX string and load as module, returns function handle */
int64_t sigil_cuda_load_ptx(const char* ptx_source, const char* kernel_name) {
    if (!g_cuda_initialized) {
        if (!sigil_cuda_init()) return -1;
    }

    if (g_num_cuda_modules >= MAX_CUDA_MODULES) {
        fprintf(stderr, "Too many CUDA modules\n");
        return -1;
    }

    CUmodule module;
    CUresult err = cuModuleLoadData(&module, ptx_source);
    if (err != CUDA_SUCCESS) {
        fprintf(stderr, "CUDA module load failed: %d\n", err);
        return -1;
    }

    CUfunction func;
    err = cuModuleGetFunction(&func, module, kernel_name);
    if (err != CUDA_SUCCESS) {
        fprintf(stderr, "CUDA get function '%s' failed: %d\n", kernel_name, err);
        cuModuleUnload(module);
        return -1;
    }

    int handle = g_num_cuda_modules;
    g_cuda_modules[handle] = module;
    g_cuda_functions[handle] = func;
    g_num_cuda_modules++;

    return (int64_t)handle;
}

/* Launch kernel with 1D grid/block configuration
 * handle: kernel handle from sigil_cuda_load_ptx
 * grid_x: number of blocks
 * block_x: threads per block
 * args: array of argument pointers (device pointers as i64)
 * num_args: number of arguments
 */
int64_t sigil_cuda_launch_kernel_1d(int64_t handle, int64_t grid_x, int64_t block_x,
                                     void** args, int64_t num_args) {
    if (handle < 0 || handle >= g_num_cuda_modules) {
        fprintf(stderr, "Invalid kernel handle: %lld\n", (long long)handle);
        return 0;
    }

    CUfunction func = g_cuda_functions[handle];

    CUresult err = cuLaunchKernel(
        func,
        (unsigned)grid_x, 1, 1,   // grid dim
        (unsigned)block_x, 1, 1,  // block dim
        0,                         // shared mem
        NULL,                      // stream
        args,                      // args
        NULL                       // extra
    );

    if (err != CUDA_SUCCESS) {
        fprintf(stderr, "CUDA kernel launch failed: %d\n", err);
        return 0;
    }

    return 1;
}

/* Launch kernel with 2D grid/block configuration */
int64_t sigil_cuda_launch_kernel_2d(int64_t handle,
                                     int64_t grid_x, int64_t grid_y,
                                     int64_t block_x, int64_t block_y,
                                     void** args, int64_t num_args) {
    if (handle < 0 || handle >= g_num_cuda_modules) {
        fprintf(stderr, "Invalid kernel handle: %lld\n", (long long)handle);
        return 0;
    }

    CUfunction func = g_cuda_functions[handle];

    CUresult err = cuLaunchKernel(
        func,
        (unsigned)grid_x, (unsigned)grid_y, 1,   // grid dim
        (unsigned)block_x, (unsigned)block_y, 1, // block dim
        0,                                        // shared mem
        NULL,                                     // stream
        args,                                     // args
        NULL                                      // extra
    );

    if (err != CUDA_SUCCESS) {
        fprintf(stderr, "CUDA kernel launch failed: %d\n", err);
        return 0;
    }

    return 1;
}

/* Compile CUDA source to PTX using NVRTC */
char* sigil_cuda_compile_to_ptx(const char* cuda_source, const char* kernel_name) {
    nvrtcProgram prog;
    nvrtcResult res = nvrtcCreateProgram(&prog, cuda_source, kernel_name, 0, NULL, NULL);
    if (res != NVRTC_SUCCESS) {
        fprintf(stderr, "NVRTC create program failed: %d\n", res);
        return NULL;
    }

    // Compile with compute capability for Ada (sm_89)
    const char* opts[] = {"--gpu-architecture=compute_89"};
    res = nvrtcCompileProgram(prog, 1, opts);
    if (res != NVRTC_SUCCESS) {
        size_t log_size;
        nvrtcGetProgramLogSize(prog, &log_size);
        char* log = (char*)malloc(log_size);
        nvrtcGetProgramLog(prog, log);
        fprintf(stderr, "NVRTC compile failed:\n%s\n", log);
        free(log);
        nvrtcDestroyProgram(&prog);
        return NULL;
    }

    size_t ptx_size;
    nvrtcGetPTXSize(prog, &ptx_size);
    char* ptx = (char*)malloc(ptx_size);
    nvrtcGetPTX(prog, ptx);
    nvrtcDestroyProgram(&prog);

    return ptx;
}

/* High-level: compile CUDA source and get kernel handle */
int64_t sigil_cuda_compile_kernel(const char* cuda_source, const char* kernel_name) {
    char* ptx = sigil_cuda_compile_to_ptx(cuda_source, kernel_name);
    if (!ptx) return -1;

    int64_t handle = sigil_cuda_load_ptx(ptx, kernel_name);
    free(ptx);
    return handle;
}

#else
/* Stub implementations when CUDA is not available */

int64_t sigil_cuda_init(void) {
    fprintf(stderr, "CUDA support not compiled in\n");
    return 0;
}

void sigil_cuda_cleanup(void) {}

int64_t sigil_cuda_get_device_count(void) { return 0; }

int64_t sigil_cuda_malloc(int64_t size) {
    (void)size;
    return 0;
}

void sigil_cuda_free(int64_t device_ptr) {
    (void)device_ptr;
}

int64_t sigil_cuda_memcpy_h2d(int64_t dst, void* src, int64_t size) {
    (void)dst; (void)src; (void)size;
    return 0;
}

int64_t sigil_cuda_memcpy_d2h(void* dst, int64_t src, int64_t size) {
    (void)dst; (void)src; (void)size;
    return 0;
}

int64_t sigil_cuda_memcpy_d2d(int64_t dst, int64_t src, int64_t size) {
    (void)dst; (void)src; (void)size;
    return 0;
}

void sigil_cuda_sync(void) {}

int64_t sigil_cuda_load_ptx(const char* ptx, const char* name) {
    (void)ptx; (void)name;
    return -1;
}

int64_t sigil_cuda_launch_kernel_1d(int64_t h, int64_t gx, int64_t bx, void** args, int64_t n) {
    (void)h; (void)gx; (void)bx; (void)args; (void)n;
    return 0;
}

int64_t sigil_cuda_launch_kernel_2d(int64_t h, int64_t gx, int64_t gy, int64_t bx, int64_t by, void** args, int64_t n) {
    (void)h; (void)gx; (void)gy; (void)bx; (void)by; (void)args; (void)n;
    return 0;
}

int64_t sigil_cuda_compile_kernel(const char* src, const char* name) {
    (void)src; (void)name;
    return -1;
}

/* Stubs for CUDA device property queries (Nihil framework linkage) */
int64_t sigil_cuda_get_compute_capability(void) { return 0; }
int64_t sigil_cuda_get_total_memory(void) { return 0; }

#endif /* SIGIL_CUDA_SUPPORT */

#endif /* SIGIL_CUDA_EXTERNAL */

/* ============================================================================
 * Random Number Generation (Nihil framework linkage)
 * sigil_random_f64 / sigil_random_normal are called by Nihil's CPU backend
 * (Tensor::randn, Cpu::fill_randn). Training uses lcg_f32_vec instead, so
 * these stubs satisfy the linker but are not called during actual training.
 * ============================================================================ */
double sigil_random_f64(void) {
    static uint64_t state = 0x853c49e6748fea9bULL;
    state ^= state >> 12;
    state ^= state << 25;
    state ^= state >> 27;
    return (double)(state * 0x2545f4914f6cdd1dULL) / (double)UINT64_MAX;
}

double sigil_random_normal(void) {
    /* Box-Muller transform */
    double u1 = sigil_random_f64();
    double u2 = sigil_random_f64();
    if (u1 < 1e-10) u1 = 1e-10;
    double r = sqrt(-2.0 * log(u1));
    return r * cos(2.0 * 3.14159265358979323846 * u2);
}

/* Returns a standard normal f32 sample as f32-bits-in-i64.
 * Sigil's LLVM ABI passes f32 as the low 32 bits of i64.
 * sigil_random_normal() returns f64 which gets mangled through the ABI;
 * this function does the f64→f32 cast in C and returns the correct bit pattern. */
int64_t sigil_random_normal_f32(void) {
    float f = (float)sigil_random_normal();
    uint32_t bits;
    memcpy(&bits, &f, sizeof(float));
    return (int64_t)bits;
}

/* Scaled normal f32 sample: returns (N(0,1) * scale) as f32-bits-in-i64.
 * scale_bits is f32-bits-in-i64 (Sigil's f32 calling convention). */
int64_t sigil_random_normal_f32_scaled(int64_t scale_bits) {
    int32_t lo32 = (int32_t)(scale_bits & 0xFFFFFFFFLL);
    float scale;
    memcpy(&scale, &lo32, sizeof(float));
    float f = (float)sigil_random_normal() * scale;
    uint32_t bits;
    memcpy(&bits, &f, sizeof(float));
    return (int64_t)bits;
}

/* Scaled fill: N(0,1)*scale into raw float* buffer.
 * Like sigil_fill_randn_f32_raw but with a scale factor.
 * scale_bits: f32-bits-in-i64 (Sigil's f32 calling convention).
 * Uses rand() — call srand(seed) first for reproducibility. */
void sigil_fill_randn_f32_scaled(int64_t ptr, int64_t len, int64_t scale_bits) {
    float* data = (float*)(uintptr_t)ptr;
    if (!data || len <= 0) return;
    int32_t lo32 = (int32_t)(scale_bits & 0xFFFFFFFFLL);
    float scale;
    memcpy(&scale, &lo32, sizeof(float));
    for (int64_t i = 0; i < len; i += 2) {
        double u1, u2;
        do { u1 = ((double)rand() + 0.5) / ((double)RAND_MAX + 1.0); } while (u1 <= 0.0);
        u2 = ((double)rand() + 0.5) / ((double)RAND_MAX + 1.0);
        double r = sqrt(-2.0 * log(u1));
        double theta = 2.0 * 3.14159265358979323846 * u2;
        data[i] = (float)(r * cos(theta)) * scale;
        if (i + 1 < len) data[i + 1] = (float)(r * sin(theta)) * scale;
    }
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

/* ============================================================================
 * Command Line Argument Marshalling
 * ============================================================================ */

/* Create a Vec<String> from C argc/argv */
void* sigil_vec_from_argv(int argc, char** argv) {
    // Create Vec with capacity for all args
    void* vec = sigil_vec_new((int64_t)argc);
    if (!vec) return NULL;

    // Convert each C string to a Sigil String and push to Vec
    for (int i = 0; i < argc; i++) {
        void* sigil_str = sigil_string_from(argv[i]);
        sigil_vec_push(vec, (int64_t)sigil_str);
    }

    return vec;
}

#ifndef SIGIL_RUNTIME_LIB_ONLY

/* Entry point - calls the Sigil main function with command line args */
extern int64_t main_sigil(int64_t args);

int main(int argc, char** argv) {
    // Marshal C argc/argv to Sigil Vec<String>
    void* args = sigil_vec_from_argv(argc, argv);

    // Call Sigil main with the arguments
    int64_t result = main_sigil((int64_t)args);

    // Note: We don't free args here as the program is exiting anyway
    return (int)result;
}

#endif /* SIGIL_RUNTIME_LIB_ONLY */
