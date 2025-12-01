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
#include <time.h>
#include <sys/time.h>

/* Get current time in milliseconds since epoch */
int64_t sigil_now(void) {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (int64_t)(tv.tv_sec * 1000 + tv.tv_usec / 1000);
}

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

/* Entry point - calls the Sigil main function */
extern int64_t main_sigil(void);

int main(int argc, char** argv) {
    (void)argc;
    (void)argv;
    int64_t result = main_sigil();
    return (int)result;
}
