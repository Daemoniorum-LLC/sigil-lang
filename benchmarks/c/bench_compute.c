#include <stdint.h>
int64_t compute_intensive(int64_t n) {
    int64_t a = 1, b = 2, c = 3, d = 4;
    for (int64_t i = 0; i < n; i++) {
        a = a * 3 + 7; b = b * 5 + 11; c = c * 7 + 13; d = d * 11 + 17;
        a ^= b; c ^= d;
    }
    return a + b + c + d;
}
int64_t division_heavy(int64_t n) {
    int64_t sum = 0;
    for (int64_t i = 1; i < n; i++) {
        sum += (i * 1000000) / (i + 1);
        sum += i % 7;
    }
    return sum;
}
int main() {
    int64_t r1 = compute_intensive(100000000);
    int64_t r2 = division_heavy(10000000);
    return (r1 != 0 && r2 > 0) ? 0 : 1;
}
