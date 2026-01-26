# Sigil Language - Agent Guide

## Agent Coordination

**Register your presence** in the workspace `CONCLAVE.sigil` and follow:
- [SDD Methodology](../docs/methodologies/SPEC-DRIVEN-DEVELOPMENT.md)
- [Agent-TDD](../docs/methodologies/AGENT-TDD.md)

## The Canonical Compiler

The **Rust-based Sigil compiler** at `parser/` is the canonical compiler.

**Test Results**: 509/596 tests passing (85%), all P0 (stable) tests pass

```bash
cd parser
cargo build --release

# Run Sigil programs
./target/release/sigil run program.sg

# Run tests
cd ../jormungandr/tests
./run_tests_rust.sh
```

## Why Rust Compiler?

The Rust compiler:
- 85% test pass rate (509/596 tests), all P0 (stable) tests pass
- Full lexer, parser, interpreter, JIT (Cranelift), and LLVM backend
- Includes stdlib with Rc<T>, Cell<T>, Drop, HTTP, WebSocket

## Commands

```bash
# Run a Sigil program (interpreter mode)
./target/release/sigil run file.sg

# Compile to native binary (LLVM backend)
./target/release/sigil compile file.sg -o output

# Compile with CUDA support (GPU compute)
./target/release/sigil compile file.sg -o output --cuda

# Compile with LTO (link-time optimization)
./target/release/sigil compile file.sg -o output --lto

# JIT execution (Cranelift backend)
./target/release/sigil jit file.sg
```

## Key Components

- `parser/src/lexer.rs` - Tokenizer (50KB)
- `parser/src/parser.rs` - Parser (337KB)
- `parser/src/interpreter.rs` - Runtime interpreter (452KB)
- `parser/src/codegen.rs` - Cranelift JIT backend (155KB)
- `parser/src/llvm_codegen.rs` - LLVM AOT backend (195KB)
- `parser/src/stdlib.rs` - Standard library (1.2MB)
- `parser/src/typeck.rs` - Type checker (122KB)
- `parser/runtime/sigil_runtime.c` - C runtime (SIMD intrinsics, memory management)
- `parser/runtime/libsigil_runtime.a` - Standard runtime library
- `parser/runtime/libsigil_runtime_cuda.a` - CUDA-enabled runtime library

## Test Suite

The test suite is located at `jormungandr/tests/`.

```bash
cd jormungandr/tests
./run_tests_rust.sh                    # Run all tests
./run_tests_rust.sh --spec 03_types    # Run specific section
./run_tests_rust.sh --priority P0      # Run P0 tests only
```

**Current Status**: 414/414 passing (100%)

Notable implementations:
- Mutable reference semantics via sync-back mechanism
- Automatic Drop::drop() calls when values go out of scope

## Jormungandr (Legacy Self-Hosted Compiler)

The `jormungandr/` directory contains the legacy self-hosted compiler written in Sigil.

**Status**: Experimental. Use the Rust compiler for actual work.

```bash
cd jormungandr/build
gcc -g -O0 -o sigil2 sigil2.c -lm
./sigil2 compile ../src/main.sg -o output.c
```

This compiler is useful for:
- Understanding Sigil's self-hosting capabilities
- Testing language features in Sigil itself
- Future bootstrap experiments

## Ecosystem Libraries

The repo contains many Sigil libraries in subdirectories:
- `aegis/` - Security primitives
- `anima/` - Animation/graphics
- `chorus/` - Concurrency primitives
- And many more...

These can be used for testing the compiler with real-world code.

## Development Workflow

1. **Build the compiler**: `cd parser && cargo build --release`
2. **Run tests**: `cd ../jormungandr/tests && ./run_tests_rust.sh`
3. **Test your changes**: `../../parser/target/release/sigil run your_test.sg`
4. **Check coverage**: Test results show which features work

## Compute Backends

The LLVM backend supports compute primitives for numerical workloads.

### SIMD Backend (AVX-512)

Native 512-bit vector operations using AVX-512 intrinsics.

**Type:** `F32x16` - 16-lane packed f32 vector

```sigil
// Allocate aligned memory (64-byte for AVX-512)
≔ a = F32x16·alloc(16);
≔ b = F32x16·alloc(16);
≔ result = F32x16·alloc(16);

// Initialize vectors
F32x16·splat(a, 2.0);
F32x16·splat(b, 3.0);

// Vector operations
F32x16·add(result, a, b);      // result = a + b
F32x16·mul(result, a, b);      // result = a * b
F32x16·fmadd(result, a, b, c); // result = a * b + c

// Reductions
≔ sum = F32x16·reduce_add(a);  // horizontal sum
≔ dot = F32x16·dot(a, b);      // dot product
```

**Requirements:** AVX-512 capable CPU. Falls back to scalar on unsupported hardware.

### CUDA Backend

GPU compute via CUDA Driver API. Compile with `--cuda` flag.

```bash
./sigil compile program.sg -o program --cuda
```

**Module:** `Cuda`

```sigil
// Initialize CUDA
≔ ok = Cuda·init();
≔ devices = Cuda·device_count();

// Device memory management
≔ d_ptr = Cuda·malloc(1024);
Cuda·free(d_ptr);

// Memory transfers
Cuda·memcpy_h2d(d_ptr, h_ptr, size);  // Host → Device
Cuda·memcpy_d2h(h_ptr, d_ptr, size);  // Device → Host

// Synchronization
Cuda·sync();

// Kernel compilation (NVRTC)
≔ kernel = Cuda·compile_kernel(cuda_source, "kernel_name");
≔ result = Cuda·launch_1d(kernel, grid_x, block_x, args_ptr, num_args);

// Cleanup
Cuda·cleanup();
```

**Requirements:** NVIDIA GPU, CUDA toolkit. Links `-lcuda -lnvrtc`.

**Runtime:** Uses `libsigil_runtime_cuda.a` instead of standard runtime.

## Recent Changes (January 2026)

- Restored Rust compiler from git history
- Fixed critical bugs in type system and codegen
- Implemented Rc<T> and Cell<T> stdlib types
- Implemented mutable reference sync-back mechanism
- Implemented Drop trait with automatic destructor calls
- Added native symbol vocabulary (middledot syntax, arrows, etc.)
- Added SIMD backend (AVX-512 F32x16 operations)
- Added CUDA backend (GPU compute via Driver API)
- Added LSP server, formatter, linter, package manager
- Added HTTP and WebSocket clients
