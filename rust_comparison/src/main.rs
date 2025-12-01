// Rust Comprehensive Benchmark - Direct comparison with Sigil JIT
// Run with: cargo run --release

use std::time::Instant;

fn fib(n: i64) -> i64 {
    if n <= 1 {
        return n;
    }
    fib(n - 1) + fib(n - 2)
}

fn ackermann(m: i64, n: i64) -> i64 {
    if m == 0 {
        return n + 1;
    }
    if n == 0 {
        return ackermann(m - 1, 1);
    }
    ackermann(m - 1, ackermann(m, n - 1))
}

fn tak(x: i64, y: i64, z: i64) -> i64 {
    if y >= x {
        return z;
    }
    tak(tak(x - 1, y, z), tak(y - 1, z, x), tak(z - 1, x, y))
}

fn gcd(mut a: i64, mut b: i64) -> i64 {
    while b != 0 {
        let temp = b;
        b = a % b;
        a = temp;
    }
    a
}

fn fib_iterative(n: i64) -> i64 {
    if n <= 1 {
        return n;
    }
    let mut a = 0i64;
    let mut b = 1i64;
    for _ in 2..=n {
        let temp = a + b;
        a = b;
        b = temp;
    }
    b
}

fn main() {
    println!("=== Rust Comprehensive Benchmark ===");
    println!();

    // Test 1: fib(30) recursive
    let start1 = Instant::now();
    let r1 = fib(30);
    let t1 = start1.elapsed();
    println!("fib(30): {}ms, result={}", t1.as_millis(), r1);

    // Test 2: fib(35) recursive
    let start2 = Instant::now();
    let r2 = fib(35);
    let t2 = start2.elapsed();
    println!("fib(35): {}ms, result={}", t2.as_millis(), r2);

    // Test 3: ackermann(3, 7)
    let start3 = Instant::now();
    let r3 = ackermann(3, 7);
    let t3 = start3.elapsed();
    println!("ackermann(3,7): {}ms, result={}", t3.as_millis(), r3);

    // Test 4: tak(18, 12, 6)
    let start4 = Instant::now();
    let r4 = tak(18, 12, 6);
    let t4 = start4.elapsed();
    println!("tak(18,12,6): {}us, result={}", t4.as_micros(), r4);

    // Test 5: GCD x10k
    let start5 = Instant::now();
    let mut g5 = 0i64;
    for i in 0..10000i64 {
        g5 = gcd(48 * i + 1, 18 * i + 1);
    }
    let t5 = start5.elapsed();
    println!("GCD x10k: {}us", t5.as_micros());
    std::hint::black_box(g5);

    // Test 6: Iterative fib x10k
    let start6 = Instant::now();
    let mut f6 = 0i64;
    for _ in 0..10000 {
        f6 = fib_iterative(50);
    }
    let t6 = start6.elapsed();
    println!("fib_iter(50) x10k: {}us", t6.as_micros());
    std::hint::black_box(f6);

    // Test 7: Sum 1M integers
    let start7 = Instant::now();
    let mut sum7 = 0i64;
    for i in 0..1000000i64 {
        sum7 += i;
    }
    let t7 = start7.elapsed();
    println!("Sum 1M integers: {}us", t7.as_micros());
    std::hint::black_box(sum7);

    // Test 8: Nested loops 1000x1000
    let start8 = Instant::now();
    let mut count8 = 0i64;
    for _ in 0..1000 {
        for _ in 0..1000 {
            count8 += 1;
        }
    }
    let t8 = start8.elapsed();
    println!("Nested 1000x1000: {}us", t8.as_micros());
    std::hint::black_box(count8);

    println!();
    println!("=== Benchmark Complete ===");
}
