// Equivalent Rust benchmark for JIT comparison
// Compile: rustc -O jit_bench.rs && ./jit_bench

fn fib(n: i64) -> i64 {
    if n <= 1 { n } else { fib(n - 1) + fib(n - 2) }
}

fn ackermann(m: i64, n: i64) -> i64 {
    if m == 0 { n + 1 }
    else if n == 0 { ackermann(m - 1, 1) }
    else { ackermann(m - 1, ackermann(m, n - 1)) }
}

fn tak(x: i64, y: i64, z: i64) -> i64 {
    if y >= x { z }
    else { tak(tak(x - 1, y, z), tak(y - 1, z, x), tak(z - 1, x, y)) }
}

fn main() {
    let r1 = fib(35);
    let r2 = ackermann(3, 7);
    let r3 = tak(18, 12, 6);
    println!("{}", r1 + r2 + r3);
}
