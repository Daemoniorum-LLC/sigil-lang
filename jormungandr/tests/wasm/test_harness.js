#!/usr/bin/env node
/**
 * Jormungandr WASM Test Harness
 * TDD Phase 0: Test Infrastructure
 *
 * Usage: node test_harness.js <wasm_path> <fixture_path>
 *
 * Loads the Jormungandr WASM module and runs a Sigil source file,
 * returning the result as JSON.
 */

const fs = require('fs');
const path = require('path');

async function main() {
    const args = process.argv.slice(2);

    if (args.length < 2) {
        console.error('Usage: node test_harness.js <wasm_path> <fixture_path>');
        process.exit(1);
    }

    const wasmPath = args[0];
    const fixturePath = args[1];

    try {
        const result = await runTest(wasmPath, fixturePath);
        console.log(JSON.stringify(result));
    } catch (error) {
        console.log(JSON.stringify({
            ok: false,
            error: `Harness error: ${error.message}`,
            exitCode: 1,
        }));
        process.exit(0); // Exit 0 so the runner can parse the JSON
    }
}

async function runTest(wasmPath, fixturePath) {
    // Load WASM module
    const wasmBytes = fs.readFileSync(wasmPath);
    const source = fs.readFileSync(fixturePath, 'utf-8');

    // Output buffer for console.log calls
    let outputBuffer = '';

    // Import object for WASM
    const imports = {
        env: {
            // Performance timer
            performance_now: () => Date.now(),

            // Memory allocation (will be overwritten by WASM exports)
            memory: null,
        },
        console: {
            // Log string (ptr, len)
            log_str: (ptr, len) => {
                const memory = new Uint8Array(instance.exports.memory.buffer);
                const bytes = memory.slice(ptr, ptr + len);
                const str = new TextDecoder().decode(bytes);
                outputBuffer += str;
            },
            // Log i64
            log_i64: (value) => {
                outputBuffer += value.toString();
            },
            // Log f64
            log_f64: (value) => {
                outputBuffer += value.toString();
            },
            // Print (general)
            print: (ptr, len) => {
                const memory = new Uint8Array(instance.exports.memory.buffer);
                const bytes = memory.slice(ptr, ptr + len);
                const str = new TextDecoder().decode(bytes);
                outputBuffer += str;
            },
            // Println
            println_str: (ptr, len) => {
                const memory = new Uint8Array(instance.exports.memory.buffer);
                const bytes = memory.slice(ptr, ptr + len);
                const str = new TextDecoder().decode(bytes);
                outputBuffer += str + '\n';
            },
            println_i64: (value) => {
                outputBuffer += value.toString() + '\n';
            },
            println_f64: (value) => {
                outputBuffer += value.toString() + '\n';
            },
        },
        // String operations
        string: {
            concat: (ptr1, len1, ptr2, len2) => 0,
            length: (ptr) => 0,
            eq: (ptr1, len1, ptr2, len2) => 0,
            from_int: (value) => 0,
            from_float: (value) => 0,
        },
        // Math operations
        math: {
            sqrt: Math.sqrt,
            sin: Math.sin,
            cos: Math.cos,
            tan: Math.tan,
            pow: Math.pow,
            exp: Math.exp,
            log: Math.log,
            floor: Math.floor,
            ceil: Math.ceil,
            round: Math.round,
            abs: Math.abs,
            random: Math.random,
        },
        // Memory operations (stubs - real impl in WASM)
        memory: {
            alloc: (size) => 0,
            free: (ptr) => {},
            realloc: (ptr, oldSize, newSize) => 0,
        },
    };

    // Instantiate WASM
    const { instance } = await WebAssembly.instantiate(wasmBytes, imports);

    // Check for required exports
    const requiredExports = ['wasm_compile_and_run', 'wasm_alloc', 'wasm_free', 'memory'];
    const missingExports = requiredExports.filter(name => !(name in instance.exports));

    if (missingExports.length > 0) {
        return {
            ok: false,
            error: `Missing WASM exports: ${missingExports.join(', ')}`,
            exitCode: 1,
        };
    }

    // Encode source string
    const encoder = new TextEncoder();
    const sourceBytes = encoder.encode(source);

    // Allocate memory for source
    const sourcePtr = instance.exports.wasm_alloc(sourceBytes.length);
    if (sourcePtr === 0) {
        return {
            ok: false,
            error: 'Failed to allocate memory for source',
            exitCode: 1,
        };
    }

    // Write source to WASM memory
    const memory = new Uint8Array(instance.exports.memory.buffer);
    memory.set(sourceBytes, sourcePtr);

    // Call compile_and_run
    const startTime = Date.now();
    const resultPtr = instance.exports.wasm_compile_and_run(sourcePtr, sourceBytes.length);
    const duration = Date.now() - startTime;

    // Read result (length-prefixed string)
    const view = new DataView(instance.exports.memory.buffer);
    const resultLen = view.getInt32(resultPtr, true);
    const resultBytes = new Uint8Array(instance.exports.memory.buffer).slice(
        resultPtr + 4,
        resultPtr + 4 + resultLen
    );
    const resultJson = new TextDecoder().decode(resultBytes);

    // Free memory
    instance.exports.wasm_free(sourcePtr, sourceBytes.length);

    // Parse and return result
    try {
        const parsed = JSON.parse(resultJson);

        // Normalize output (trim trailing newline for comparison)
        if (parsed.output) {
            parsed.output = parsed.output.replace(/\n$/, '');
        }

        return {
            ok: parsed.ok,
            output: parsed.output || outputBuffer.replace(/\n$/, ''),
            error: parsed.error,
            exitCode: parsed.ok ? (parsed.exit_code || 0) : 1,
            durationMs: parsed.duration_ms || duration,
        };
    } catch (e) {
        return {
            ok: false,
            error: `Failed to parse result: ${resultJson}`,
            exitCode: 1,
        };
    }
}

// Export for use as module
module.exports = { runTest };

// Run if called directly
if (require.main === module) {
    main().catch(err => {
        console.error(err);
        process.exit(1);
    });
}
