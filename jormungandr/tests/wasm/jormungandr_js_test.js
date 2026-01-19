#!/usr/bin/env node
/**
 * Jormungandr JS Bridge Tests
 * TDD Phase 3: JavaScript Bridge
 *
 * These tests verify the JavaScript API for the Jormungandr WASM compiler.
 * Run with: node jormungandr_js_test.js
 */

const path = require('path');
const fs = require('fs');

// Test framework (minimal, no dependencies)
let passCount = 0;
let failCount = 0;
let skipCount = 0;

const GREEN = '\x1b[32m';
const RED = '\x1b[31m';
const YELLOW = '\x1b[33m';
const NC = '\x1b[0m';

async function test(name, fn) {
    process.stdout.write(`  ${name}... `);
    try {
        await fn();
        console.log(`${GREEN}PASS${NC}`);
        passCount++;
    } catch (err) {
        console.log(`${RED}FAIL${NC}`);
        console.log(`    Error: ${err.message}`);
        failCount++;
    }
}

function skip(name, reason) {
    console.log(`  ${name}... ${YELLOW}SKIP${NC} (${reason})`);
    skipCount++;
}

function expect(actual) {
    return {
        toBe(expected) {
            if (actual !== expected) {
                throw new Error(`Expected ${JSON.stringify(expected)}, got ${JSON.stringify(actual)}`);
            }
        },
        toContain(substr) {
            if (typeof actual !== 'string' || !actual.includes(substr)) {
                throw new Error(`Expected "${actual}" to contain "${substr}"`);
            }
        },
        toBeGreaterThan(n) {
            if (!(actual > n)) {
                throw new Error(`Expected ${actual} to be greater than ${n}`);
            }
        },
        toBeGreaterThanOrEqual(n) {
            if (!(actual >= n)) {
                throw new Error(`Expected ${actual} to be >= ${n}`);
            }
        },
        toBeTruthy() {
            if (!actual) {
                throw new Error(`Expected truthy value, got ${actual}`);
            }
        },
    };
}

// ============================================================================
// Tests
// ============================================================================

async function runTests() {
    console.log('============================================');
    console.log('  Jormungandr JS Bridge Tests');
    console.log('============================================');
    console.log('');

    const buildDir = path.join(__dirname, '../../build');
    const wasmPath = path.join(buildDir, 'jormungandr.wasm');
    const jsPath = path.join(buildDir, 'jormungandr.js');

    // Check if WASM exists
    if (!fs.existsSync(wasmPath)) {
        console.log(`${YELLOW}WARNING: jormungandr.wasm not found at:${NC}`);
        console.log(`  ${wasmPath}`);
        console.log('');
        console.log('This is expected in Phase 3 TDD (tests written before WASM is built).');
        console.log('Build with: make wasm');
        console.log('');
        console.log(`${YELLOW}Skipping all tests.${NC}`);
        return;
    }

    // Check if JS bridge exists
    if (!fs.existsSync(jsPath)) {
        console.log(`${YELLOW}WARNING: jormungandr.js not found at:${NC}`);
        console.log(`  ${jsPath}`);
        console.log('');
        console.log('Create the JS bridge to run these tests.');
        console.log('');
        console.log(`${YELLOW}Skipping all tests.${NC}`);
        return;
    }

    // Load the JS bridge
    const { Jormungandr } = require(jsPath);
    let compiler;

    try {
        console.log('Loading Jormungandr...');
        compiler = await Jormungandr.load(wasmPath);
        console.log('Loaded successfully.\n');
    } catch (err) {
        console.log(`${RED}Failed to load Jormungandr:${NC} ${err.message}`);
        return;
    }

    // ========================================================================
    // Basic Compilation Tests
    // ========================================================================

    console.log('Basic Compilation:');

    await test('compiles and runs hello world', async () => {
        const result = await compiler.run('rite main() -> i64 { println("Hello"); 0 }');
        expect(result.ok).toBe(true);
        expect(result.output).toContain('Hello');
        expect(result.exitCode).toBe(0);
    });

    await test('compiles arithmetic expressions', async () => {
        const result = await compiler.run('rite main() -> i64 { println(str(2 + 3)); 0 }');
        expect(result.ok).toBe(true);
        expect(result.output).toContain('5');
    });

    await test('returns exit code from main', async () => {
        const result = await compiler.run('rite main() -> i64 { 42 }');
        expect(result.ok).toBe(true);
        expect(result.exitCode).toBe(42);
    });

    // ========================================================================
    // Error Handling Tests
    // ========================================================================

    console.log('\nError Handling:');

    await test('returns syntax errors', async () => {
        const result = await compiler.run('rite main( { }');
        expect(result.ok).toBe(false);
        expect(result.error.toLowerCase()).toContain('parse');
    });

    await test('returns type errors', async () => {
        const result = await compiler.run('rite main() -> i64 { "not a number" }');
        expect(result.ok).toBe(false);
        expect(result.error.toLowerCase()).toContain('type');
    });

    await test('handles runtime errors (division by zero)', async () => {
        const result = await compiler.run('rite main() -> i64 { let x = 1 / 0; println(str(x)); 0 }');
        expect(result.ok).toBe(false);
        expect(result.error.toLowerCase()).toContain('division');
    });

    // ========================================================================
    // Sigil Language Features
    // ========================================================================

    console.log('\nSigil Features:');

    await test('handles morpheme transform (τ)', async () => {
        const result = await compiler.run(`
            rite main() -> i64 {
                let nums = [1, 2, 3];
                let doubled = nums |τ{ it * 2 };
                println(str(doubled));
                0
            }
        `);
        expect(result.ok).toBe(true);
        expect(result.output).toContain('2');
        expect(result.output).toContain('4');
        expect(result.output).toContain('6');
    });

    await test('handles morpheme sum (Σ)', async () => {
        const result = await compiler.run(`
            rite main() -> i64 {
                let nums = [1, 2, 3, 4, 5];
                let sum = nums |Σ;
                println(str(sum));
                0
            }
        `);
        expect(result.ok).toBe(true);
        expect(result.output).toContain('15');
    });

    await test('handles morpheme filter (φ)', async () => {
        const result = await compiler.run(`
            rite main() -> i64 {
                let nums = [1, 2, 3, 4, 5, 6];
                let evens = nums |φ{ it % 2 == 0 };
                println(str(evens));
                0
            }
        `);
        expect(result.ok).toBe(true);
        expect(result.output).toContain('2');
        expect(result.output).toContain('4');
        expect(result.output).toContain('6');
    });

    await test('tracks evidentiality (known !)', async () => {
        const result = await compiler.run(`
            rite main() -> i64 {
                let known! = 42;
                println(evidence_of(known));
                0
            }
        `);
        expect(result.ok).toBe(true);
        expect(result.output.toLowerCase()).toContain('known');
    });

    // ========================================================================
    // Syntax Checking (check API)
    // ========================================================================

    console.log('\nSyntax Checking:');

    await test('check() returns empty for valid code', async () => {
        const diagnostics = await compiler.check('rite main() -> i64 { 0 }');
        expect(Array.isArray(diagnostics)).toBe(true);
        expect(diagnostics.length).toBe(0);
    });

    await test('check() returns diagnostics for invalid code', async () => {
        const diagnostics = await compiler.check('rite main() -> i64 { "oops" }');
        expect(Array.isArray(diagnostics)).toBe(true);
        expect(diagnostics.length).toBeGreaterThan(0);
    });

    // ========================================================================
    // Performance & Metadata
    // ========================================================================

    console.log('\nPerformance:');

    await test('reports execution time', async () => {
        const result = await compiler.run('rite main() -> i64 { 0 }');
        expect(result.ok).toBe(true);
        expect(result.durationMs).toBeGreaterThanOrEqual(0);
    });

    await test('handles concurrent compilations', async () => {
        const promises = [
            compiler.run('rite main() -> i64 { println("1"); 0 }'),
            compiler.run('rite main() -> i64 { println("2"); 0 }'),
            compiler.run('rite main() -> i64 { println("3"); 0 }'),
        ];

        const results = await Promise.all(promises);
        const allOk = results.every(r => r.ok);
        expect(allOk).toBe(true);
    });

    // ========================================================================
    // Cleanup
    // ========================================================================

    console.log('\nCleanup:');

    await test('dispose() releases resources', async () => {
        compiler.dispose();
        expect(true).toBe(true); // No error thrown
    });

    // ========================================================================
    // Summary
    // ========================================================================

    console.log('');
    console.log('============================================');
    console.log('  Results');
    console.log('============================================');
    console.log(`  ${GREEN}Passed:${NC}  ${passCount}`);
    console.log(`  ${RED}Failed:${NC}  ${failCount}`);
    console.log(`  ${YELLOW}Skipped:${NC} ${skipCount}`);
    console.log('');

    if (failCount > 0) {
        console.log(`${RED}TESTS FAILED${NC}`);
        process.exit(1);
    } else {
        console.log(`${GREEN}ALL TESTS PASSED${NC}`);
        process.exit(0);
    }
}

runTests().catch(err => {
    console.error(`${RED}Fatal error:${NC}`, err);
    process.exit(1);
});
