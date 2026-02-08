#include <stdint.h>
int64_t matrix_compute(int64_t size, int64_t iterations) {
    int64_t result = 0;
    for (int64_t iter = 0; iter < iterations; iter++) {
        for (int64_t i = 0; i < size; i++) {
            for (int64_t j = 0; j < size; j++) {
                result += (i * size + j) * (iter + 1);
            }
        }
    }
    return result;
}
int main() {
    int64_t r = matrix_compute(100, 1000);
    return (r != 0) ? 0 : 1;
}
