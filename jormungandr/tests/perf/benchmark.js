#!/usr/bin/env node
/**
 * Jormungandr Performance Benchmarks
 * TDD Phase 5: Performance & Polish
 *
 * Measures compilation and execution times to ensure
 * the WASM compiler meets performance requirements.
 *
 * Run with: node benchmark.js
 */

const path = require('path');
const fs = require('fs');

// Colors
const GREEN = '\x1b[32m';
const RED = '\x1b[31m';
const YELLOW = '\x1b[33m';
const CYAN = '\x1b[36m';
const NC = '\x1b[0m';

// Performance targets
const BENCHMARKS = [
    {
        name: 'Hello World',
        code: 'rite main() -> i64 { println("Hello"); 0 }',
        maxMs: 100,
        description: 'Basic compilation and execution',
    },
    {
        name: 'Arithmetic',
        code: `
rite main() -> i64 {
    let a = 10;
    let b = 20;
    let c = a + b * 2;
    println(str(c));
    0
}`,
        maxMs: 100,
        description: 'Variable binding and arithmetic',
    },
    {
        name: 'Fibonacci 15',
        code: `
rite fib(n: i64) -> i64 {
    if n <= 1 { n }
    else { fib(n - 1) + fib(n - 2) }
}
rite main() -> i64 {
    println(str(fib(15)));
    0
}`,
        maxMs: 500,
        description: 'Recursive function calls',
    },
    {
        name: 'Morpheme Transform (100 items)',
        code: `
rite main() -> i64 {
    let arr = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
               11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
               21, 22, 23, 24, 25, 26, 27, 28, 29, 30,
               31, 32, 33, 34, 35, 36, 37, 38, 39, 40,
               41, 42, 43, 44, 45, 46, 47, 48, 49, 50,
               51, 52, 53, 54, 55, 56, 57, 58, 59, 60,
               61, 62, 63, 64, 65, 66, 67, 68, 69, 70,
               71, 72, 73, 74, 75, 76, 77, 78, 79, 80,
               81, 82, 83, 84, 85, 86, 87, 88, 89, 90,
               91, 92, 93, 94, 95, 96, 97, 98, 99, 100];
    let doubled = arr |τ{ it * 2 };
    let sum = doubled |Σ;
    println(str(sum));
    0
}`,
        maxMs: 300,
        description: 'Array morpheme operations',
    },
    {
        name: 'Morpheme Filter + Sum',
        code: `
rite main() -> i64 {
    let arr = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
               11, 12, 13, 14, 15, 16, 17, 18, 19, 20];
    let evens = arr |φ{ it % 2 == 0 };
    let sum = evens |Σ;
    println(str(sum));
    0
}`,
        maxMs: 200,
        description: 'Filter and reduce operations',
    },
    {
        name: 'Struct Creation',
        code: `
sigil Point { x: i64, y: i64 }
sigil Line { start: Point, end: Point }

rite main() -> i64 {
    let p1 = Point { x: 0, y: 0 };
    let p2 = Point { x: 10, y: 20 };
    let line = Line { start: p1, end: p2 };
    println(str(line.end.x + line.end.y));
    0
}`,
        maxMs: 150,
        description: 'Struct instantiation and field access',
    },
    {
        name: 'Type Check (10 functions)',
        code: generateFunctions(10),
        maxMs: 200,
        description: 'Type checking multiple functions',
    },
    {
        name: 'Type Check (50 functions)',
        code: generateFunctions(50),
        maxMs: 500,
        description: 'Type checking many functions',
    },
    {
        name: 'Evidentiality Tracking',
        code: `
rite main() -> i64 {
    let known! = 42;
    let uncertain? = get_value();
    println(evidence_of(known));
    0
}
rite get_value() -> i64? { ?100 }`,
        maxMs: 150,
        description: 'Evidentiality type operations',
    },
    {
        name: 'String Operations',
        code: `
rite main() -> i64 {
    let s1 = "Hello, ";
    let s2 = "World!";
    let s3 = s1 ++ s2;
    println(s3);
    println(str(len(s3)));
    0
}`,
        maxMs: 150,
        description: 'String concatenation and length',
    },
];

function generateFunctions(count) {
    let code = '';
    for (let i = 0; i < count; i++) {
        code += `rite func_${i}(x: i64) -> i64 { x + ${i} }\n`;
    }
    code += `\nrite main() -> i64 { func_0(42) }`;
    return code;
}

async function runBenchmarks() {
    console.log('============================================');
    console.log('  Jormungandr Performance Benchmarks');
    console.log('============================================\n');

    const buildDir = path.join(__dirname, '../../build');
    const wasmPath = path.join(buildDir, 'jormungandr.wasm');
    const jsPath = path.join(buildDir, 'jormungandr.js');

    // Check if WASM exists
    if (!fs.existsSync(wasmPath)) {
        console.log(`${YELLOW}WARNING: jormungandr.wasm not found${NC}`);
        console.log('Build with: make wasm');
        console.log(`\n${YELLOW}Skipping benchmarks.${NC}`);
        return;
    }

    // Load compiler
    const { Jormungandr } = require(jsPath);
    let compiler;

    try {
        console.log('Loading compiler...');
        const loadStart = performance.now();
        compiler = await Jormungandr.load(wasmPath);
        const loadTime = performance.now() - loadStart;
        console.log(`Compiler loaded in ${loadTime.toFixed(2)}ms\n`);
    } catch (err) {
        console.log(`${RED}Failed to load compiler:${NC} ${err.message}`);
        process.exit(1);
    }

    // Run benchmarks
    let passCount = 0;
    let failCount = 0;
    const results = [];

    for (const bench of BENCHMARKS) {
        process.stdout.write(`${CYAN}${bench.name}${NC}... `);

        // Warm-up run (not counted)
        await compiler.run(bench.code);

        // Measured runs (average of 3)
        const times = [];
        for (let i = 0; i < 3; i++) {
            const start = performance.now();
            const result = await compiler.run(bench.code);
            const duration = performance.now() - start;
            times.push(duration);

            if (!result.ok) {
                console.log(`${RED}ERROR${NC}`);
                console.log(`  Code failed: ${result.error}`);
                failCount++;
                results.push({ ...bench, time: -1, status: 'ERROR' });
                continue;
            }
        }

        const avgTime = times.reduce((a, b) => a + b, 0) / times.length;
        const status = avgTime <= bench.maxMs ? 'PASS' : 'FAIL';

        if (status === 'PASS') {
            console.log(`${GREEN}${avgTime.toFixed(2)}ms${NC} (max: ${bench.maxMs}ms)`);
            passCount++;
        } else {
            console.log(`${RED}${avgTime.toFixed(2)}ms${NC} (max: ${bench.maxMs}ms) ${RED}SLOW${NC}`);
            failCount++;
        }

        results.push({ ...bench, time: avgTime, status });
    }

    // Summary
    console.log('\n============================================');
    console.log('  Results');
    console.log('============================================');
    console.log(`  ${GREEN}Passed:${NC} ${passCount}`);
    console.log(`  ${RED}Failed:${NC} ${failCount}`);

    // Detailed table
    console.log('\n  Detailed Results:');
    console.log('  ' + '-'.repeat(60));
    console.log(`  ${'Benchmark'.padEnd(30)} ${'Time'.padStart(10)} ${'Max'.padStart(10)} Status`);
    console.log('  ' + '-'.repeat(60));

    for (const r of results) {
        const timeStr = r.time >= 0 ? `${r.time.toFixed(2)}ms` : 'ERROR';
        const statusColor = r.status === 'PASS' ? GREEN : RED;
        console.log(`  ${r.name.padEnd(30)} ${timeStr.padStart(10)} ${(r.maxMs + 'ms').padStart(10)} ${statusColor}${r.status}${NC}`);
    }

    console.log('');

    // Cleanup
    compiler.dispose();

    if (failCount > 0) {
        console.log(`${RED}BENCHMARKS FAILED${NC}`);
        process.exit(1);
    } else {
        console.log(`${GREEN}ALL BENCHMARKS PASSED${NC}`);
    }
}

runBenchmarks().catch(err => {
    console.error(`${RED}Fatal error:${NC}`, err);
    process.exit(1);
});
