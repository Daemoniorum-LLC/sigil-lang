#!/usr/bin/env node
/**
 * Jormungandr Memory Leak Test
 * TDD Phase 5: Performance & Polish
 *
 * Runs the compiler repeatedly to detect memory leaks.
 * Memory should remain stable after many iterations.
 *
 * Run with: node memory_leak_test.js
 */

const path = require('path');
const fs = require('fs');

// Colors
const GREEN = '\x1b[32m';
const RED = '\x1b[31m';
const YELLOW = '\x1b[33m';
const NC = '\x1b[0m';

// Test configuration
const ITERATIONS = 100;
const MEMORY_GROWTH_THRESHOLD = 1.5; // Max 50% growth allowed
const SAMPLE_INTERVAL = 10; // Sample memory every N iterations

const TEST_CODE = `
rite fib(n: i64) -> i64 {
    if n <= 1 { n }
    else { fib(n - 1) + fib(n - 2) }
}

rite main() -> i64 {
    let result = fib(10);
    println(str(result));
    0
}
`;

async function runMemoryLeakTest() {
    console.log('============================================');
    console.log('  Jormungandr Memory Leak Test');
    console.log('============================================\n');

    const buildDir = path.join(__dirname, '../../build');
    const wasmPath = path.join(buildDir, 'jormungandr.wasm');
    const jsPath = path.join(buildDir, 'jormungandr.js');

    // Check if WASM exists
    if (!fs.existsSync(wasmPath)) {
        console.log(`${YELLOW}WARNING: jormungandr.wasm not found${NC}`);
        console.log('Build with: make wasm');
        console.log(`\n${YELLOW}Skipping memory leak test.${NC}`);
        return;
    }

    // Load compiler
    const { Jormungandr } = require(jsPath);
    let compiler;

    try {
        console.log('Loading compiler...');
        compiler = await Jormungandr.load(wasmPath);
        console.log('Compiler loaded.\n');
    } catch (err) {
        console.log(`${RED}Failed to load compiler:${NC} ${err.message}`);
        process.exit(1);
    }

    // Force initial GC if available
    if (global.gc) {
        global.gc();
    }

    const memorySamples = [];
    const initialMemory = process.memoryUsage();

    console.log(`Running ${ITERATIONS} iterations...`);
    console.log(`Initial heap: ${(initialMemory.heapUsed / 1024 / 1024).toFixed(2)} MB\n`);

    // Run iterations
    for (let i = 1; i <= ITERATIONS; i++) {
        const result = await compiler.run(TEST_CODE);

        if (!result.ok) {
            console.log(`${RED}Iteration ${i} failed:${NC} ${result.error}`);
            process.exit(1);
        }

        // Sample memory periodically
        if (i % SAMPLE_INTERVAL === 0) {
            if (global.gc) global.gc();
            const mem = process.memoryUsage();
            memorySamples.push({
                iteration: i,
                heapUsed: mem.heapUsed,
                heapTotal: mem.heapTotal,
                external: mem.external,
            });
            process.stdout.write(`  Iteration ${i}/${ITERATIONS}: ${(mem.heapUsed / 1024 / 1024).toFixed(2)} MB\r`);
        }
    }

    console.log('\n');

    // Force final GC
    if (global.gc) {
        global.gc();
    }

    const finalMemory = process.memoryUsage();

    // Analyze results
    console.log('Memory Analysis:');
    console.log('-'.repeat(50));

    const initialHeap = initialMemory.heapUsed / 1024 / 1024;
    const finalHeap = finalMemory.heapUsed / 1024 / 1024;
    const growth = finalHeap / initialHeap;

    console.log(`  Initial heap:  ${initialHeap.toFixed(2)} MB`);
    console.log(`  Final heap:    ${finalHeap.toFixed(2)} MB`);
    console.log(`  Growth factor: ${growth.toFixed(2)}x`);
    console.log(`  Threshold:     ${MEMORY_GROWTH_THRESHOLD}x`);

    // Show trend
    if (memorySamples.length >= 2) {
        console.log('\n  Memory trend:');
        const first = memorySamples[0];
        const last = memorySamples[memorySamples.length - 1];

        for (const sample of memorySamples) {
            const mb = sample.heapUsed / 1024 / 1024;
            const bar = '█'.repeat(Math.min(50, Math.round(mb * 2)));
            console.log(`    ${String(sample.iteration).padStart(4)}: ${bar} ${mb.toFixed(2)} MB`);
        }
    }

    console.log('');

    // Cleanup
    compiler.dispose();

    // Verdict
    if (growth <= MEMORY_GROWTH_THRESHOLD) {
        console.log(`${GREEN}MEMORY TEST PASSED${NC}`);
        console.log(`Memory growth (${growth.toFixed(2)}x) is within acceptable limits.`);
    } else {
        console.log(`${RED}MEMORY TEST FAILED${NC}`);
        console.log(`Memory growth (${growth.toFixed(2)}x) exceeds threshold (${MEMORY_GROWTH_THRESHOLD}x).`);
        console.log('Possible memory leak detected.');
        process.exit(1);
    }
}

// Note: Run with --expose-gc for accurate GC:
// node --expose-gc memory_leak_test.js

if (!global.gc) {
    console.log(`${YELLOW}TIP: Run with --expose-gc for more accurate results:${NC}`);
    console.log('  node --expose-gc memory_leak_test.js\n');
}

runMemoryLeakTest().catch(err => {
    console.error(`${RED}Fatal error:${NC}`, err);
    process.exit(1);
});
