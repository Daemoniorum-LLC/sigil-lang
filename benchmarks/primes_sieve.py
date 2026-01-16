#!/usr/bin/env python3
# Benchmark: Sieve of Eratosthenes

def sieve(limit):
    is_prime = [True] * (limit + 1)
    is_prime[0] = is_prime[1] = False

    p = 2
    while p * p <= limit:
        if is_prime[p]:
            for multiple in range(p * p, limit + 1, p):
                is_prime[multiple] = False
        p += 1

    return sum(is_prime)

if __name__ == "__main__":
    count = sieve(10000)
    print(count)
