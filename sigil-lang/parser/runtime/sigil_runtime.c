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

#endif /* SIGIL_CUDA_SUPPORT */

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
