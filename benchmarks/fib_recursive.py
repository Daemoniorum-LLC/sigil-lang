#!/usr/bin/env python3
# Benchmark: Recursive Fibonacci

def fib(n):
    if n <= 1:
        return n
    return fib(n - 1) + fib(n - 2)

if __name__ == "__main__":
    result = fib(35)
    print(result)
