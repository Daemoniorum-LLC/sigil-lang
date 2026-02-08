#include <stdint.h>
int64_t cache_unfriendly(int64_t n, int64_t stride) {
    int64_t sum = 0, idx = 0;
    for (int64_t i = 0; i < n; i++) {
        idx = (idx + stride) % 10000;
        sum += idx;
    }
    return sum;
}
int64_t cache_friendly(int64_t n) {
    int64_t sum = 0;
    for (int64_t i = 0; i < n; i++) sum += i;
    return sum;
}
int main() {
    int64_t r1 = cache_unfriendly(10000000, 4096);
    int64_t r2 = cache_friendly(10000000);
    return (r1 > 0 && r2 > 0) ? 0 : 1;
}
