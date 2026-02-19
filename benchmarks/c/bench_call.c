#include <stdint.h>
__attribute__((noinline)) int64_t tiny_func(int64_t x) { return x + 1; }
int64_t call_chain(int64_t x) { return tiny_func(tiny_func(tiny_func(tiny_func(x)))); }
int64_t deep_recursion(int64_t n, int64_t acc) {
    if (n <= 0) return acc;
    return deep_recursion(n - 1, acc + 1);
}
int64_t many_calls(int64_t n) {
    int64_t sum = 0;
    for (int64_t i = 0; i < n; i++) sum += call_chain(i);
    return sum;
}
int main() {
    int64_t r1 = many_calls(10000000);
    int64_t r2 = deep_recursion(100000, 0);
    return (r1 > 0 && r2 == 100000) ? 0 : 1;
}
