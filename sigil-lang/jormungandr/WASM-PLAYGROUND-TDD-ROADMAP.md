# TDD Roadmap: Jormungandr WASM Playground Compiler

> **Principle:** Write failing tests FIRST, then implement to pass. No phase is complete until all quality gates pass.

## Overview

Compile the Jormungandr self-hosted Sigil compiler to WASM, enabling real Sigil compilation and execution in the browser playground.

**Target:** `website-qliphoth/wasm/jormungandr.wasm`

---

## Quality Gate Definitions

Every phase must pass these gates before proceeding:

### Gate 1: Test Coverage
- [ ] All new code has corresponding tests
- [ ] All tests pass (0 failures)
- [ ] No skipped tests without tracking issue

### Gate 2: No Stubs/TODOs
```bash
# Must return 0 results
grep -rn "TODO\|FIXME\|STUB\|unimplemented\|todo!\|panic!(\"not" src/*.sg
```

### Gate 3: No Dead Code
```bash
# Compiler warnings for unused code must be addressed
./sigil check --warn-unused src/*.sg
```

### Gate 4: Documentation
- [ ] All public functions have doc comments
- [ ] README updated with new functionality

### Gate 5: Integration Test
- [ ] End-to-end test passes with real Sigil code

---

## Phase 0: Test Infrastructure Setup

**Goal:** Establish testing framework for WASM compilation.

### 0.1 Create Test Directory Structure

```
jormungandr/
├── tests/
│   ├── wasm/
│   │   ├── run_wasm_tests.sh       # WASM test runner
│   │   ├── test_harness.js         # Node.js WASM loader
│   │   └── fixtures/
│   │       ├── hello.sg            # Basic test case
│   │       ├── arithmetic.sg       # Math operations
│   │       ├── morphemes.sg        # Morpheme operators
│   │       ├── evidentiality.sg    # Evidence tracking
│   │       └── errors.sg           # Error handling
│   └── unit/
│       ├── lexer_test.sg
│       ├── parser_test.sg
│       ├── typeck_test.sg
│       ├── interp_test.sg
│       └── wasm_bridge_test.sg
```

### 0.2 Tests to Write FIRST

**Test: `tests/wasm/fixtures/hello.sg`**
```sigil
// EXPECTED_OUTPUT: Hello, Sigil!
// EXPECTED_EXIT: 0

rite main() -> i64 {
    println("Hello, Sigil!");
    0
}
```

**Test: `tests/wasm/run_wasm_tests.sh`**
```bash
#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
WASM_FILE="$SCRIPT_DIR/../../build/jormungandr.wasm"

# Verify WASM exists
if [ ! -f "$WASM_FILE" ]; then
    echo "FAIL: jormungandr.wasm not found"
    exit 1
fi

# Run each fixture
for fixture in "$SCRIPT_DIR/fixtures/"*.sg; do
    name=$(basename "$fixture" .sg)
    expected_output=$(grep "EXPECTED_OUTPUT:" "$fixture" | sed 's/.*EXPECTED_OUTPUT: //')
    expected_exit=$(grep "EXPECTED_EXIT:" "$fixture" | sed 's/.*EXPECTED_EXIT: //')

    echo "Testing: $name"

    # Run via Node.js harness
    result=$(node "$SCRIPT_DIR/test_harness.js" "$WASM_FILE" "$fixture")
    actual_output=$(echo "$result" | head -n -1)
    actual_exit=$(echo "$result" | tail -1)

    if [ "$actual_output" != "$expected_output" ]; then
        echo "  FAIL: Output mismatch"
        echo "    Expected: $expected_output"
        echo "    Actual:   $actual_output"
        exit 1
    fi

    if [ "$actual_exit" != "$expected_exit" ]; then
        echo "  FAIL: Exit code mismatch"
        echo "    Expected: $expected_exit"
        echo "    Actual:   $actual_exit"
        exit 1
    fi

    echo "  PASS"
done

echo "All WASM tests passed!"
```

### 0.3 Quality Gate Checklist

- [ ] Test runner script exists and is executable
- [ ] All fixture files have EXPECTED_OUTPUT and EXPECTED_EXIT comments
- [ ] Node.js test harness can load WASM modules
- [ ] Running tests shows "jormungandr.wasm not found" (expected - not built yet)

---

## Phase 1: WASM Entry Point Module

**Goal:** Create the WASM-compatible entry point that bridges JS ↔ Sigil.

### 1.1 Tests to Write FIRST

**Test: `tests/unit/wasm_bridge_test.sg`**
```sigil
//! Tests for WASM bridge module

use crate::wasm_bridge::*;

#[test]
fn test_compile_and_run_hello() {
    let source = "rite main() -> i64 { println(\"Hello\"); 0 }";
    let result = compile_and_run(source);

    assert!(result.is_ok());
    assert_eq!(result.unwrap().output, "Hello\n");
    assert_eq!(result.unwrap().exit_code, 0);
}

#[test]
fn test_compile_and_run_arithmetic() {
    let source = "rite main() -> i64 { println(str(2 + 3)); 0 }";
    let result = compile_and_run(source);

    assert!(result.is_ok());
    assert_eq!(result.unwrap().output, "5\n");
}

#[test]
fn test_compile_error_syntax() {
    let source = "rite main( { }";  // Missing closing paren
    let result = compile_and_run(source);

    assert!(result.is_err());
    assert!(result.unwrap_err().contains("syntax error"));
}

#[test]
fn test_compile_error_type() {
    let source = "rite main() -> i64 { \"not an int\" }";
    let result = compile_and_run(source);

    assert!(result.is_err());
    assert!(result.unwrap_err().contains("type"));
}

#[test]
fn test_runtime_error_division_by_zero() {
    let source = "rite main() -> i64 { println(str(1 / 0)); 0 }";
    let result = compile_and_run(source);

    assert!(result.is_err());
    assert!(result.unwrap_err().contains("division by zero"));
}

#[test]
fn test_morpheme_transform() {
    let source = r#"
        rite main() -> i64 {
            let nums = [1, 2, 3];
            let doubled = nums |τ{ it * 2 };
            println(str(doubled));
            0
        }
    "#;
    let result = compile_and_run(source);

    assert!(result.is_ok());
    assert_eq!(result.unwrap().output, "[2, 4, 6]\n");
}

#[test]
fn test_evidentiality_known() {
    let source = r#"
        rite main() -> i64 {
            let x! = 42;
            println(evidence_of(x));
            0
        }
    "#;
    let result = compile_and_run(source);

    assert!(result.is_ok());
    assert_eq!(result.unwrap().output, "known\n");
}
```

### 1.2 Implementation

**File: `src/wasm_bridge.sg`**
```sigil
//! WASM Bridge - Entry point for browser-based compilation
//!
//! This module provides the interface between JavaScript and the
//! Jormungandr compiler pipeline.

use crate::lexer::Lexer;
use crate::parser::Parser;
use crate::typeck::TypeChecker;
use crate::lower::Lowerer;
use crate::interp::Interpreter;

/// Result of compilation and execution
pub struct ExecutionResult {
    /// Console output from the program
    pub output: !String,
    /// Exit code (0 = success)
    pub exit_code: !i64,
    /// Execution time in milliseconds
    pub duration_ms: !f64,
}

/// Compile and run Sigil source code
///
/// # Arguments
/// * `source` - The Sigil source code to compile and run
///
/// # Returns
/// * `Ok(ExecutionResult)` - Successful execution with output
/// * `Err(String)` - Compilation or runtime error message
///
/// # Example
/// ```sigil
/// let result = compile_and_run("rite main() -> i64 { 42 }");
/// assert!(result.is_ok());
/// ```
pub fn compile_and_run(source: !&str) -> !Result<ExecutionResult, String> {
    let start_time = now();

    // Phase 1: Lexical analysis
    let tokens = match Lexer::new(source).tokenize() {
        Result::Ok(t) => t,
        Result::Err(e) => return Result::Err(format!("Lexer error: {}", e)),
    };

    // Phase 2: Parsing
    let ast = match Parser::new(tokens).parse() {
        Result::Ok(a) => a,
        Result::Err(e) => return Result::Err(format!("Syntax error: {}", e)),
    };

    // Phase 3: Type checking
    let typed_ast = match TypeChecker::new().check(ast) {
        Result::Ok(t) => t,
        Result::Err(e) => return Result::Err(format!("Type error: {}", e)),
    };

    // Phase 4: Lower to IR
    let ir = match Lowerer::new().lower(typed_ast) {
        Result::Ok(i) => i,
        Result::Err(e) => return Result::Err(format!("Lowering error: {}", e)),
    };

    // Phase 5: Interpret
    let mut interpreter = Interpreter::new();
    interpreter.load_module(ir);

    let exit_code = match interpreter.run() {
        Result::Ok(val) => match val.value {
            Value::Int(n) => n,
            _ => 0,
        },
        Result::Err(e) => return Result::Err(format!("Runtime error: {}", e.message)),
    };

    let end_time = now();

    Result::Ok(ExecutionResult {
        output: interpreter.get_output(),
        exit_code,
        duration_ms: end_time - start_time,
    })
}

/// Check syntax without executing
///
/// Returns list of diagnostics (errors and warnings)
pub fn check_syntax(source: !&str) -> ![Diagnostic] {
    let mut diagnostics: ![Diagnostic] = [];

    // Lexer errors
    match Lexer::new(source).tokenize() {
        Result::Ok(tokens) => {
            // Parser errors
            match Parser::new(tokens).parse() {
                Result::Ok(ast) => {
                    // Type errors
                    match TypeChecker::new().check(ast) {
                        Result::Ok(_) => {},
                        Result::Err(e) => diagnostics.push(Diagnostic::error(e)),
                    }
                },
                Result::Err(e) => diagnostics.push(Diagnostic::error(e)),
            }
        },
        Result::Err(e) => diagnostics.push(Diagnostic::error(e)),
    }

    diagnostics
}

/// Diagnostic message
pub struct Diagnostic {
    pub severity: !DiagnosticSeverity,
    pub message: !String,
    pub line: ?u32,
    pub column: ?u32,
}

pub enum DiagnosticSeverity {
    Error,
    Warning,
    Info,
}

impl Diagnostic {
    pub fn error(message: !String) -> !Diagnostic {
        Diagnostic {
            severity: DiagnosticSeverity::Error,
            message,
            line: null,
            column: null,
        }
    }
}

// WASM exports - these are the functions callable from JavaScript
#[wasm_export]
pub fn wasm_compile_and_run(source_ptr: !i32, source_len: !i32) -> !i32 {
    // Read string from WASM memory
    let source = wasm_read_string(source_ptr, source_len);

    // Compile and run
    let result = compile_and_run(source);

    // Write result to WASM memory and return pointer
    match result {
        Result::Ok(exec) => {
            let json = format!(
                r#"{{"ok":true,"output":"{}","exit_code":{},"duration_ms":{}}}"#,
                escape_json(exec.output),
                exec.exit_code,
                exec.duration_ms
            );
            wasm_write_string(json)
        },
        Result::Err(err) => {
            let json = format!(
                r#"{{"ok":false,"error":"{}"}}"#,
                escape_json(err)
            );
            wasm_write_string(json)
        },
    }
}

#[wasm_export]
pub fn wasm_check_syntax(source_ptr: !i32, source_len: !i32) -> !i32 {
    let source = wasm_read_string(source_ptr, source_len);
    let diagnostics = check_syntax(source);

    // Serialize diagnostics to JSON
    let json = diagnostics_to_json(diagnostics);
    wasm_write_string(json)
}

// Memory management for WASM
#[wasm_export]
pub fn wasm_alloc(size: !i32) -> !i32 {
    // Allocate memory and return pointer
    alloc(size as usize) as i32
}

#[wasm_export]
pub fn wasm_free(ptr: !i32, size: !i32) {
    free(ptr as usize, size as usize);
}

// Internal helpers
fn wasm_read_string(ptr: !i32, len: !i32) -> !String {
    // Read UTF-8 bytes from WASM linear memory
    let bytes = read_memory(ptr as usize, len as usize);
    String::from_utf8(bytes).unwrap_or_default()
}

fn wasm_write_string(s: !String) -> !i32 {
    let bytes = s.as_bytes();
    let ptr = alloc(bytes.len() + 4);

    // Write length prefix (4 bytes)
    write_memory(ptr, (bytes.len() as u32).to_le_bytes());
    // Write string bytes
    write_memory(ptr + 4, bytes);

    ptr as i32
}

fn escape_json(s: !String) -> !String {
    s.replace("\\", "\\\\")
     .replace("\"", "\\\"")
     .replace("\n", "\\n")
     .replace("\r", "\\r")
     .replace("\t", "\\t")
}

fn diagnostics_to_json(diagnostics: ![Diagnostic]) -> !String {
    let items = diagnostics |τ{ d =>
        format!(
            r#"{{"severity":"{}","message":"{}","line":{},"column":{}}}"#,
            match d.severity {
                DiagnosticSeverity::Error => "error",
                DiagnosticSeverity::Warning => "warning",
                DiagnosticSeverity::Info => "info",
            },
            escape_json(d.message),
            d.line.map(|l| l.to_string()).unwrap_or("null"),
            d.column.map(|c| c.to_string()).unwrap_or("null")
        )
    } |join(",");

    format!("[{}]", items)
}

// Platform-specific time function
#[cfg(target_arch = "wasm32")]
fn now() -> !f64 {
    // Will be provided by JS
    extern "wasm" {
        fn performance_now() -> f64;
    }
    performance_now()
}

#[cfg(not(target_arch = "wasm32"))]
fn now() -> !f64 {
    // Native implementation
    0.0
}
```

### 1.3 Quality Gate Checklist

- [ ] All 7 unit tests pass
- [ ] `grep -rn "TODO\|FIXME" src/wasm_bridge.sg` returns 0 results
- [ ] All public functions have doc comments
- [ ] No compiler warnings

---

## Phase 2: WASM Compilation Target

**Goal:** Configure build system to compile Jormungandr to WASM.

### 2.1 Tests to Write FIRST

**Test: WASM module loads correctly**
```javascript
// tests/wasm/test_harness.js
const fs = require('fs');
const path = require('path');

async function loadWasm(wasmPath) {
    const bytes = fs.readFileSync(wasmPath);

    const imports = {
        env: {
            performance_now: () => Date.now(),
        },
        console: {
            log: (ptr, len) => { /* ... */ },
        },
    };

    const { instance } = await WebAssembly.instantiate(bytes, imports);
    return instance;
}

async function runTest(wasmPath, fixturePath) {
    const instance = await loadWasm(wasmPath);
    const source = fs.readFileSync(fixturePath, 'utf-8');

    // Write source to WASM memory
    const encoder = new TextEncoder();
    const sourceBytes = encoder.encode(source);
    const sourcePtr = instance.exports.wasm_alloc(sourceBytes.length);
    const memory = new Uint8Array(instance.exports.memory.buffer);
    memory.set(sourceBytes, sourcePtr);

    // Call compile_and_run
    const resultPtr = instance.exports.wasm_compile_and_run(sourcePtr, sourceBytes.length);

    // Read result
    const resultLen = new DataView(instance.exports.memory.buffer).getInt32(resultPtr, true);
    const resultBytes = memory.slice(resultPtr + 4, resultPtr + 4 + resultLen);
    const result = new TextDecoder().decode(resultBytes);

    // Free memory
    instance.exports.wasm_free(sourcePtr, sourceBytes.length);

    return JSON.parse(result);
}

// Test: WASM exports required functions
async function testExports(wasmPath) {
    const instance = await loadWasm(wasmPath);

    const requiredExports = [
        'wasm_compile_and_run',
        'wasm_check_syntax',
        'wasm_alloc',
        'wasm_free',
        'memory',
    ];

    for (const name of requiredExports) {
        if (!(name in instance.exports)) {
            throw new Error(`Missing export: ${name}`);
        }
    }

    console.log('All required exports present');
}

// Test: Memory allocation works
async function testMemory(wasmPath) {
    const instance = await loadWasm(wasmPath);

    const ptr1 = instance.exports.wasm_alloc(100);
    const ptr2 = instance.exports.wasm_alloc(200);

    if (ptr1 === 0 || ptr2 === 0) {
        throw new Error('Allocation returned null');
    }

    if (ptr1 === ptr2) {
        throw new Error('Allocations overlap');
    }

    instance.exports.wasm_free(ptr1, 100);
    instance.exports.wasm_free(ptr2, 200);

    console.log('Memory allocation works');
}

module.exports = { loadWasm, runTest, testExports, testMemory };
```

### 2.2 Build Configuration

**File: `Sigil.toml` (update)**
```toml
[project]
name = "jormungandr"
version = "0.2.0"
edition = "2025"

[lib]
name = "jormungandr"
path = "src/lib.sg"

[[bin]]
name = "sigil"
path = "src/driver.sg"

[dependencies]
# No external dependencies for WASM compatibility

[features]
default = ["native"]
native = []
wasm = []

[profile.wasm]
backend = "llvm"
target = "wasm32-unknown-unknown"
opt-level = "z"          # Optimize for size
lto = true               # Link-time optimization
strip = true             # Strip debug symbols

[profile.wasm.features]
simd128 = false          # Disable SIMD for broader compatibility
bulk-memory = true       # Enable bulk memory operations
```

**File: `Makefile` (update)**
```makefile
.PHONY: all build test wasm clean

SIGIL_COMPILER := ../../parser/target/release/sigil
SRC_FILES := $(wildcard src/*.sg)
WASM_OUT := build/jormungandr.wasm

all: build

build:
	$(SIGIL_COMPILER) build

test: build
	$(SIGIL_COMPILER) test

wasm: $(SRC_FILES)
	@mkdir -p build
	$(SIGIL_COMPILER) build --target wasm32-unknown-unknown --profile wasm -o $(WASM_OUT)
	@echo "WASM size: $$(wc -c < $(WASM_OUT)) bytes"
	@echo "Verifying exports..."
	@node tests/wasm/verify_exports.js $(WASM_OUT)

wasm-test: wasm
	./tests/wasm/run_wasm_tests.sh

clean:
	rm -rf build/

# Quality gates
lint:
	$(SIGIL_COMPILER) check --warn-unused src/*.sg

check-todos:
	@if grep -rn "TODO\|FIXME\|STUB\|unimplemented" src/*.sg; then \
		echo "ERROR: Found TODOs/stubs in source"; \
		exit 1; \
	fi

quality-gate: lint check-todos test wasm-test
	@echo "All quality gates passed!"
```

### 2.3 Quality Gate Checklist

- [ ] `make wasm` produces `build/jormungandr.wasm`
- [ ] WASM file size < 1MB
- [ ] All required exports present (verified by `verify_exports.js`)
- [ ] Memory tests pass
- [ ] `make quality-gate` passes

---

## Phase 3: JavaScript Bridge

**Goal:** Create the JS library that loads and interfaces with `jormungandr.wasm`.

### 3.1 Tests to Write FIRST

**Test: `tests/wasm/jormungandr_js_test.js`**
```javascript
const { Jormungandr } = require('../../build/jormungandr.js');

describe('Jormungandr JS Bridge', () => {
    let compiler;

    beforeAll(async () => {
        compiler = await Jormungandr.load('./build/jormungandr.wasm');
    });

    afterAll(() => {
        compiler.dispose();
    });

    test('compiles and runs hello world', async () => {
        const result = await compiler.run('rite main() -> i64 { println("Hello"); 0 }');

        expect(result.ok).toBe(true);
        expect(result.output).toBe('Hello\n');
        expect(result.exitCode).toBe(0);
    });

    test('returns syntax errors', async () => {
        const result = await compiler.run('rite main( { }');

        expect(result.ok).toBe(false);
        expect(result.error).toContain('syntax');
    });

    test('returns type errors', async () => {
        const result = await compiler.run('rite main() -> i64 { "string" }');

        expect(result.ok).toBe(false);
        expect(result.error).toContain('type');
    });

    test('handles runtime errors', async () => {
        const result = await compiler.run('rite main() -> i64 { 1 / 0 }');

        expect(result.ok).toBe(false);
        expect(result.error).toContain('division');
    });

    test('provides diagnostics for check()', async () => {
        const diagnostics = await compiler.check('rite main() -> i64 { "oops" }');

        expect(diagnostics.length).toBeGreaterThan(0);
        expect(diagnostics[0].severity).toBe('error');
    });

    test('handles morpheme operators', async () => {
        const result = await compiler.run(`
            rite main() -> i64 {
                let nums = [1, 2, 3, 4, 5];
                let sum = nums |Σ;
                println(str(sum));
                0
            }
        `);

        expect(result.ok).toBe(true);
        expect(result.output).toBe('15\n');
    });

    test('tracks evidentiality', async () => {
        const result = await compiler.run(`
            rite main() -> i64 {
                let known! = 42;
                let uncertain? = get_input();
                println(evidence_of(known));
                0
            }

            rite get_input() -> i64? { ?42 }
        `);

        expect(result.ok).toBe(true);
        expect(result.output).toBe('known\n');
    });

    test('reports execution time', async () => {
        const result = await compiler.run('rite main() -> i64 { 0 }');

        expect(result.ok).toBe(true);
        expect(result.durationMs).toBeGreaterThanOrEqual(0);
    });

    test('handles concurrent compilations', async () => {
        const promises = [
            compiler.run('rite main() -> i64 { println("1"); 0 }'),
            compiler.run('rite main() -> i64 { println("2"); 0 }'),
            compiler.run('rite main() -> i64 { println("3"); 0 }'),
        ];

        const results = await Promise.all(promises);

        expect(results.every(r => r.ok)).toBe(true);
    });
});
```

### 3.2 Implementation

**File: `build/jormungandr.js`**
```javascript
/**
 * Jormungandr - Sigil Compiler for the Browser
 *
 * @example
 * const compiler = await Jormungandr.load('./jormungandr.wasm');
 * const result = await compiler.run('rite main() -> i64 { 42 }');
 * console.log(result.output);
 * compiler.dispose();
 */

class Jormungandr {
    #instance = null;
    #memory = null;
    #encoder = new TextEncoder();
    #decoder = new TextDecoder();

    constructor(instance) {
        this.#instance = instance;
        this.#memory = new Uint8Array(instance.exports.memory.buffer);
    }

    /**
     * Load the Jormungandr WASM module
     * @param {string} wasmPath - Path or URL to jormungandr.wasm
     * @returns {Promise<Jormungandr>}
     */
    static async load(wasmPath) {
        const imports = {
            env: {
                performance_now: () => performance.now(),
            },
            console: {
                log_str: (ptr, len) => {
                    // Handled internally
                },
            },
        };

        let bytes;
        if (typeof fetch !== 'undefined') {
            // Browser
            const response = await fetch(wasmPath);
            bytes = await response.arrayBuffer();
        } else {
            // Node.js
            const fs = require('fs');
            bytes = fs.readFileSync(wasmPath);
        }

        const { instance } = await WebAssembly.instantiate(bytes, imports);
        return new Jormungandr(instance);
    }

    /**
     * Compile and run Sigil source code
     * @param {string} source - Sigil source code
     * @returns {Promise<{ok: boolean, output?: string, exitCode?: number, durationMs?: number, error?: string}>}
     */
    async run(source) {
        const sourceBytes = this.#encoder.encode(source);
        const sourcePtr = this.#alloc(sourceBytes.length);

        try {
            // Write source to WASM memory
            this.#refreshMemory();
            this.#memory.set(sourceBytes, sourcePtr);

            // Call WASM function
            const resultPtr = this.#instance.exports.wasm_compile_and_run(
                sourcePtr,
                sourceBytes.length
            );

            // Read result
            const result = this.#readResult(resultPtr);

            return result;
        } finally {
            this.#free(sourcePtr, sourceBytes.length);
        }
    }

    /**
     * Check syntax without executing
     * @param {string} source - Sigil source code
     * @returns {Promise<Array<{severity: string, message: string, line?: number, column?: number}>>}
     */
    async check(source) {
        const sourceBytes = this.#encoder.encode(source);
        const sourcePtr = this.#alloc(sourceBytes.length);

        try {
            this.#refreshMemory();
            this.#memory.set(sourceBytes, sourcePtr);

            const resultPtr = this.#instance.exports.wasm_check_syntax(
                sourcePtr,
                sourceBytes.length
            );

            const jsonStr = this.#readString(resultPtr);
            return JSON.parse(jsonStr);
        } finally {
            this.#free(sourcePtr, sourceBytes.length);
        }
    }

    /**
     * Release WASM resources
     */
    dispose() {
        this.#instance = null;
        this.#memory = null;
    }

    // Private helpers

    #alloc(size) {
        return this.#instance.exports.wasm_alloc(size);
    }

    #free(ptr, size) {
        this.#instance.exports.wasm_free(ptr, size);
    }

    #refreshMemory() {
        // Memory buffer may have grown
        this.#memory = new Uint8Array(this.#instance.exports.memory.buffer);
    }

    #readString(ptr) {
        this.#refreshMemory();
        const view = new DataView(this.#instance.exports.memory.buffer);
        const len = view.getInt32(ptr, true);
        const bytes = this.#memory.slice(ptr + 4, ptr + 4 + len);
        return this.#decoder.decode(bytes);
    }

    #readResult(ptr) {
        const jsonStr = this.#readString(ptr);
        const parsed = JSON.parse(jsonStr);

        if (parsed.ok) {
            return {
                ok: true,
                output: parsed.output,
                exitCode: parsed.exit_code,
                durationMs: parsed.duration_ms,
            };
        } else {
            return {
                ok: false,
                error: parsed.error,
            };
        }
    }
}

// Export for Node.js and ES modules
if (typeof module !== 'undefined') {
    module.exports = { Jormungandr };
}
if (typeof window !== 'undefined') {
    window.Jormungandr = Jormungandr;
}

export { Jormungandr };
```

### 3.3 Quality Gate Checklist

- [ ] All 10 JS tests pass
- [ ] TypeScript types are correct (if using TS)
- [ ] Works in Node.js
- [ ] Works in browser (Chrome, Firefox, Safari)
- [ ] No console errors
- [ ] Memory is properly freed (no leaks after 100 iterations)

---

## Phase 4: Playground Integration

**Goal:** Update the playground to use the real Jormungandr compiler.

### 4.1 Tests to Write FIRST

**Test: `tests/e2e/playground_test.js` (Playwright)**
```javascript
const { test, expect } = require('@playwright/test');

test.describe('Sigil Playground with Jormungandr', () => {
    test.beforeEach(async ({ page }) => {
        await page.goto('/playground.html');
        // Wait for WASM to load
        await page.waitForFunction(() => window.compiler !== undefined);
    });

    test('runs hello world', async ({ page }) => {
        // Clear editor and type code
        await page.fill('#code-input', 'rite main() -> i64 { println("Hello, Sigil!"); 0 }');

        // Click run
        await page.click('#run-btn');

        // Check console output
        const output = await page.textContent('#console-output');
        expect(output).toContain('Hello, Sigil!');
    });

    test('shows syntax errors', async ({ page }) => {
        await page.fill('#code-input', 'rite main( { }');
        await page.click('#run-btn');

        const output = await page.textContent('#console-output');
        expect(output).toContain('error');

        // Status should be error
        const statusDot = await page.$('#status-dot');
        expect(await statusDot.getAttribute('class')).toContain('status-error');
    });

    test('shows type errors', async ({ page }) => {
        await page.fill('#code-input', 'rite main() -> i64 { "not an int" }');
        await page.click('#run-btn');

        const output = await page.textContent('#console-output');
        expect(output).toContain('type');
    });

    test('runs morpheme examples', async ({ page }) => {
        // Select morphemes example
        await page.selectOption('#example-select', 'morphemes');
        await page.click('#run-btn');

        // Should complete without errors
        const statusText = await page.textContent('#status-text');
        expect(statusText).toBe('Success');
    });

    test('runs counter example', async ({ page }) => {
        await page.selectOption('#example-select', 'counter');
        await page.click('#run-btn');

        // Should render in preview
        const preview = await page.$('#preview-content');
        expect(await preview.textContent()).not.toContain("Click 'Run'");
    });

    test('keyboard shortcut Ctrl+Enter runs code', async ({ page }) => {
        await page.fill('#code-input', 'rite main() -> i64 { println("shortcut"); 0 }');
        await page.press('#code-input', 'Control+Enter');

        const output = await page.textContent('#console-output');
        expect(output).toContain('shortcut');
    });

    test('share button generates URL', async ({ page }) => {
        await page.fill('#code-input', 'rite main() -> i64 { 42 }');
        await page.click('#share-btn');

        // Check clipboard (may need permissions)
        const output = await page.textContent('#console-output');
        expect(output).toContain('copied');
    });

    test('loads code from URL hash', async ({ page }) => {
        const code = 'rite main() -> i64 { println("from url"); 0 }';
        const encoded = Buffer.from(encodeURIComponent(code)).toString('base64');

        await page.goto(`/playground.html#code=${encoded}`);
        await page.waitForFunction(() => window.compiler !== undefined);

        const editorContent = await page.inputValue('#code-input');
        expect(editorContent).toContain('from url');
    });

    test('clear console button works', async ({ page }) => {
        await page.fill('#code-input', 'rite main() -> i64 { println("test"); 0 }');
        await page.click('#run-btn');
        await page.click('#clear-console-btn');

        const output = await page.textContent('#console-output');
        expect(output).toBe('');
    });

    test('theme toggle works', async ({ page }) => {
        await page.click('#theme-toggle');

        const theme = await page.getAttribute('html', 'data-theme');
        expect(theme).toBe('light');

        await page.click('#theme-toggle');
        const theme2 = await page.getAttribute('html', 'data-theme');
        expect(theme2 || '').toBe('');
    });
});
```

### 4.2 Implementation

**File: `website-qliphoth/pages/playground.html` (update runCode function)**
```javascript
// Replace the simulated execution with real Jormungandr

let compiler = null;

async function initCompiler() {
    try {
        setStatus('compiling', 'Loading compiler...');
        compiler = await Jormungandr.load('../wasm/jormungandr.wasm');
        setStatus('idle', 'Ready');
        console.log('Jormungandr compiler loaded!');
    } catch (error) {
        console.error('Failed to load compiler:', error);
        setStatus('error', 'Compiler failed to load');
        appendToConsole(`Failed to load compiler: ${error.message}`, 'error');
    }
}

async function runCode() {
    if (!compiler) {
        appendToConsole('Compiler not loaded. Please refresh the page.', 'error');
        return;
    }

    const codeInput = document.getElementById('code-input');
    const code = codeInput.value;

    clearConsole();
    setStatus('compiling', 'Compiling...');

    try {
        const result = await compiler.run(code);

        if (result.ok) {
            // Show output
            if (result.output) {
                appendToConsole(result.output, 'success');
            }
            appendToConsole(`// Program exited with code ${result.exitCode} (${result.durationMs.toFixed(2)}ms)`, 'info');
            setStatus('success', 'Success');

            // If the code produces VDOM, render it
            renderPreview(code, result);
        } else {
            appendToConsole(`Error: ${result.error}`, 'error');
            setStatus('error', 'Error');
        }
    } catch (error) {
        appendToConsole(`Internal error: ${error.message}`, 'error');
        setStatus('error', 'Error');
    }
}

// Real-time syntax checking (debounced)
let checkTimeout = null;

function onCodeChange(e) {
    const code = e.target.value;
    updateLineNumbers(code);
    updateSyntaxHighlight(code);

    // Debounced syntax check
    clearTimeout(checkTimeout);
    checkTimeout = setTimeout(async () => {
        if (compiler) {
            const diagnostics = await compiler.check(code);
            showDiagnostics(diagnostics);
        }
    }, 500);
}

function showDiagnostics(diagnostics) {
    // Clear previous diagnostics
    const existing = document.querySelectorAll('.diagnostic-marker');
    existing.forEach(el => el.remove());

    // Show new diagnostics
    for (const diag of diagnostics) {
        if (diag.line) {
            highlightLine(diag.line, diag.severity);
        }
    }

    // Update status if errors
    const errors = diagnostics.filter(d => d.severity === 'error');
    if (errors.length > 0) {
        setStatus('error', `${errors.length} error${errors.length > 1 ? 's' : ''}`);
    }
}

// Initialize compiler on page load
init().then(() => {
    initCompiler();
});
```

### 4.3 Quality Gate Checklist

- [ ] All 10 Playwright E2E tests pass
- [ ] Compiler loads in < 2 seconds
- [ ] All 6 example programs run correctly
- [ ] No console errors during normal operation
- [ ] Works in Chrome, Firefox, Safari
- [ ] Mobile responsive (tablet at minimum)

---

## Phase 5: Performance & Polish

**Goal:** Optimize WASM size, execution speed, and user experience.

### 5.1 Performance Tests

**Test: `tests/perf/benchmark.js`**
```javascript
const { Jormungandr } = require('../../build/jormungandr.js');

async function benchmark() {
    const compiler = await Jormungandr.load('./build/jormungandr.wasm');

    const tests = [
        {
            name: 'Hello World',
            code: 'rite main() -> i64 { println("Hello"); 0 }',
            maxMs: 50,
        },
        {
            name: 'Fibonacci 20',
            code: `
                rite fib(n: i64) -> i64 {
                    if n <= 1 { n }
                    else { fib(n - 1) + fib(n - 2) }
                }
                rite main() -> i64 { println(str(fib(20))); 0 }
            `,
            maxMs: 500,
        },
        {
            name: 'Array map 1000',
            code: `
                rite main() -> i64 {
                    let arr = range(0, 1000);
                    let result = arr |τ{ it * 2 } |Σ;
                    println(str(result));
                    0
                }
            `,
            maxMs: 200,
        },
        {
            name: 'Type check large file',
            code: generateLargeFile(100), // 100 functions
            maxMs: 1000,
        },
    ];

    console.log('Performance Benchmarks\n');

    for (const test of tests) {
        const start = performance.now();
        const result = await compiler.run(test.code);
        const duration = performance.now() - start;

        const status = duration <= test.maxMs ? 'PASS' : 'FAIL';
        console.log(`${test.name}: ${duration.toFixed(2)}ms (max: ${test.maxMs}ms) [${status}]`);

        if (status === 'FAIL') {
            process.exit(1);
        }
    }

    compiler.dispose();
    console.log('\nAll benchmarks passed!');
}

function generateLargeFile(numFunctions) {
    let code = '';
    for (let i = 0; i < numFunctions; i++) {
        code += `rite func_${i}(x: i64) -> i64 { x + ${i} }\n`;
    }
    code += 'rite main() -> i64 { func_0(42) }';
    return code;
}

benchmark().catch(console.error);
```

### 5.2 Size Optimization

```bash
# Target: < 500KB for jormungandr.wasm

# Current techniques:
# - opt-level = "z" (optimize for size)
# - LTO enabled
# - Strip debug symbols
# - No SIMD

# Additional optimizations:
wasm-opt -Oz build/jormungandr.wasm -o build/jormungandr.opt.wasm
wasm-strip build/jormungandr.opt.wasm

# Compression for serving:
gzip -9 -k build/jormungandr.opt.wasm
brotli -9 -k build/jormungandr.opt.wasm
```

### 5.3 Quality Gate Checklist

- [ ] WASM size < 500KB (uncompressed)
- [ ] WASM size < 150KB (brotli compressed)
- [ ] Hello world executes in < 50ms
- [ ] Fib(20) executes in < 500ms
- [ ] No memory leaks after 1000 runs
- [ ] All performance benchmarks pass

---

## Final Quality Gate

Before declaring the project complete:

### Code Quality
- [ ] `grep -rn "TODO\|FIXME\|STUB\|unimplemented" src/` returns 0 results
- [ ] All functions have doc comments
- [ ] No compiler warnings (`sigil check --warn-unused`)
- [ ] Code formatted (`sigil fmt`)

### Test Coverage
- [ ] Unit tests: 100% of public functions
- [ ] Integration tests: All compiler phases
- [ ] E2E tests: All playground features
- [ ] Performance tests: All benchmarks pass

### Documentation
- [ ] README updated
- [ ] CHANGELOG updated
- [ ] API documentation generated
- [ ] Example code verified

### Deployment
- [ ] WASM builds in CI
- [ ] Tests run in CI
- [ ] WASM served with correct MIME type
- [ ] Brotli/gzip compression enabled

---

## Timeline

| Phase | Estimated Effort | Dependencies |
|-------|------------------|--------------|
| Phase 0: Test Infrastructure | 2-4 hours | None |
| Phase 1: WASM Entry Point | 4-8 hours | Phase 0 |
| Phase 2: WASM Compilation | 4-8 hours | Phase 1 |
| Phase 3: JS Bridge | 4-6 hours | Phase 2 |
| Phase 4: Playground Integration | 4-8 hours | Phase 3 |
| Phase 5: Performance & Polish | 4-8 hours | Phase 4 |

**Total: 22-42 hours (~3-5 days)**

---

## Success Criteria

The project is complete when:

1. User types Sigil code in playground
2. Clicks "Run" (or Ctrl+Enter)
3. Real Jormungandr compiler (in WASM) compiles the code
4. Real interpreter executes it
5. Output appears in console
6. All quality gates pass
7. No TODOs, FIXMEs, or stubs remain
