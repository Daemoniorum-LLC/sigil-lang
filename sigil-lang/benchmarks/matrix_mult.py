#!/usr/bin/env python3
# Benchmark: Matrix Multiplication

def matrix_mult(size):
    n = size
    total = n * n

    # Flattened matrices
    a = [((i // n) + (i % n)) for i in range(total)]
    b = [((i // n) * (i % n)) for i in range(total)]
    c = [0] * total

    # Multiply
    for i in range(n):
        for j in range(n):
            s = 0
            for k in range(n):
                s += a[i * n + k] * b[k * n + j]
            c[i * n + j] = s

    return c[total - 1]

if __name__ == "__main__":
    result = matrix_mult(100)
    print(result)
