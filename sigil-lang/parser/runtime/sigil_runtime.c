/*
 * Sigil Runtime Library
 *
 * Provides runtime functions for AOT-compiled Sigil programs.
 * This gets linked with the compiled object file to create
 * a standalone native executable.
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include <time.h>
#include <sys/time.h>

/* ============================================================================
 * Time Functions
 * ============================================================================ */

/* Get current time in milliseconds since epoch */
int64_t sigil_now(void) {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (int64_t)(tv.tv_sec * 1000 + tv.tv_usec / 1000);
}

/* ============================================================================
 * Print Functions
 * ============================================================================ */

/* Print an integer value */
void sigil_print_int(int64_t value) {
    printf("%ld\n", (long)value);
}

/* Print a float value */
void sigil_print_float(double value) {
    printf("%g\n", value);
}

/* Print a string */
void sigil_print_str(const char* str) {
    printf("%s\n", str);
}

/* ============================================================================
 * Math Functions (operate on i64 bits representing f64)
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

/* Exponential (e^x) */
int64_t sigil_exp(int64_t x) {
    return double_to_bits(exp(bits_to_double(x)));
}

/* Natural logarithm */
int64_t sigil_ln(int64_t x) {
    return double_to_bits(log(bits_to_double(x)));
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

/* Round */
int64_t sigil_round(int64_t x) {
    return double_to_bits(round(bits_to_double(x)));
}

/* Absolute value (float) */
int64_t sigil_fabs(int64_t x) {
    return double_to_bits(fabs(bits_to_double(x)));
}

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

/* ============================================================================
 * Entry Point
 * ============================================================================ */

/* Entry point - calls the Sigil main function */
extern int64_t main_sigil(void);

int main(int argc, char** argv) {
    (void)argc;
    (void)argv;
    int64_t result = main_sigil();
    return (int)result;
}
