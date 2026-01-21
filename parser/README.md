# Sigil Parser

The Rust-based Sigil compiler. See the [main repository](https://github.com/Daemoniorum-LLC/sigil-lang) for full documentation.

## v0.2.0

- Native symbol vocabulary (`≔`, `→`, `λ`, `σ`, `θ`, middledot syntax)
- SIMD backend (AVX-512 F32x16 operations)
- CUDA backend (GPU compute via `--cuda` flag)
- LSP server, formatter, linter
- 414 tests passing (100%)

## Build

```bash
# Basic build (Cranelift JIT)
cargo build --release

# With LLVM backend (native compilation)
apt install llvm-18-dev libpolly-18-dev libzstd-dev clang-18
CC=clang-18 cargo build --release --features llvm
```

## Usage

```bash
# Interpreter
./target/release/sigil run program.sg

# Cranelift JIT
./target/release/sigil jit program.sg

# LLVM native compile
./target/release/sigil compile program.sg -o program

# With CUDA
./target/release/sigil compile program.sg -o program --cuda
```

## Test

```bash
cargo test
# Or via test suite
cd ../jormungandr/tests && ./run_tests_rust.sh
```

## License

MIT / Apache 2.0 - Daemoniorum, LLC
