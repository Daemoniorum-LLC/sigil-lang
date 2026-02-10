#include <stdint.h>
int64_t predictable_branches(int64_t n) {
    int64_t sum = 0;
    for (int64_t i = 0; i < n; i++) {
        if (i < n / 2) sum += 1;
        else sum += 2;
    }
    return sum;
}
int64_t unpredictable_branches(int64_t n) {
    int64_t sum = 0, state = 12345;
    for (int64_t i = 0; i < n; i++) {
        state = (state * 1103515245 + 12345) % 2147483648;
        if (state % 2 == 0) sum += 1;
        else sum += 2;
    }
    return sum;
}
int main() {
    int64_t r1 = predictable_branches(100000000);
    int64_t r2 = unpredictable_branches(100000000);
    return (r1 > 0 && r2 > 0) ? 0 : 1;
}
