#!/usr/bin/env node
/**
 * Verify Jormungandr WASM Exports
 * TDD Phase 2: WASM Compilation Target
 *
 * Checks that the WASM module exports all required functions.
 */

const fs = require('fs');

async function main() {
    const wasmPath = process.argv[2];

    if (!wasmPath) {
        console.error('Usage: node verify_exports.js <wasm_path>');
        process.exit(1);
    }

    try {
        const wasmBytes = fs.readFileSync(wasmPath);
        const module = await WebAssembly.compile(wasmBytes);
        const exports = WebAssembly.Module.exports(module);

        const exportNames = exports.map(e => e.name);

        console.log('WASM Exports:');
        exports.forEach(e => {
            console.log(`  ${e.name}: ${e.kind}`);
        });
        console.log('');

        // Required exports
        const required = [
            { name: 'wasm_compile_and_run', kind: 'function' },
            { name: 'wasm_check_syntax', kind: 'function' },
            { name: 'wasm_alloc', kind: 'function' },
            { name: 'wasm_free', kind: 'function' },
            { name: 'memory', kind: 'memory' },
        ];

        const missing = [];
        const wrongKind = [];

        for (const req of required) {
            const found = exports.find(e => e.name === req.name);
            if (!found) {
                missing.push(req.name);
            } else if (found.kind !== req.kind) {
                wrongKind.push(`${req.name}: expected ${req.kind}, got ${found.kind}`);
            }
        }

        if (missing.length > 0) {
            console.error('Missing exports:');
            missing.forEach(name => console.error(`  - ${name}`));
        }

        if (wrongKind.length > 0) {
            console.error('Wrong export kinds:');
            wrongKind.forEach(msg => console.error(`  - ${msg}`));
        }

        if (missing.length > 0 || wrongKind.length > 0) {
            console.error('\nVerification FAILED');
            process.exit(1);
        }

        console.log('All required exports present!');
        console.log('Verification PASSED');

    } catch (error) {
        console.error(`Error: ${error.message}`);
        process.exit(1);
    }
}

main();
